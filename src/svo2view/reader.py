"""Read a ZED SVO2 recording: frames, timestamps, sensors, rectification. No ZED SDK, no NVIDIA GPU.

    from svo2view import SVO2
    s = SVO2("recording.svo2")
    s.n, s.t_ns                      # frame count, per-frame image timestamps in ns (== SDK get_timestamp(IMAGE))
    s.image(i)                       # side-by-side BGR uint8 (H x 2W x 3): the stored frame, BT.601 full range
    s.image(i, look="sdk")           # + an approximation of the SDK's playback sharpening
    s.image(i, rect=True)            # rectified (needs a calibration, see calib.py)
    s.yuv(i)                         # the decoded planes exactly as stored (Y: H x 2W, U / V: H/2 x W)
    s.sensors()                      # every sensor sample (IMU, magnetometer, barometer, temperatures)

Format notes (measured on ZED SDK 5.5 files; see README): MCAP container; a video message is u32 rest_len, u32 hevc_len,
an Annex-B H.265 (or H.264) access unit, then a 56-byte trailer; GOP of `fps` frames (an IDR each second, P-frames
between); a sensor message is JSON {"data": base64} of a 360-byte struct.
"""
from __future__ import annotations

import base64
import json
import threading

import av
import cv2
import numpy as np

from . import calib as _calib
from .mcapidx import McapIndex

PARITY = {
    "timestamps": "exact vs SDK 5.5 (image timestamps of every frame)",
    "sensors": "exact float32 vs SDK 5.5 (IMU raw + calibrated, orientation, magnetometer, barometer, temperatures)",
    "image (camera)": "exact decode of the stored stream; BGR by BT.601 full range",
    "image (SDK look)": "approximation of the SDK's playback sharpening (mean |err| ~0.94 levels at the default sharpness)",
    "rectification": "SDK geometry, ~0.06 px median with an SDK calibration JSON; approximate from a factory .conf",
    "depth / tracking": "not reproduced (closed SDK models)",
}

# 360-byte sensor struct: float32 fields (offset, count) and u64 timestamps, matched value by value against the SDK
SENSOR_F32 = {"quat_xyzw": (60, 4), "gyro_raw": (88, 3), "acc_raw": (100, 3), "gyro": (112, 3), "acc": (124, 3),
              "temp_imu": (228, 1), "mag": (236, 3), "mag_raw": (248, 3), "temp_baro": (272, 1),
              "pressure_pa": (276, 1), "rel_alt_m": (284, 1), "temp_left": (292, 1), "temp_right": (296, 1)}
SENSOR_U64 = {"t_imu": 16, "t_baro": 32, "t_mag": 48}
SENSOR_STRUCT = 360


def _nal_types(b):
    out, j = [], b.find(b"\x00\x00\x01")
    while 0 <= j < len(b) - 3:
        out.append(b[j + 3])
        j = b.find(b"\x00\x00\x01", j + 3)
    return out


class SVO2:
    def __init__(self, path, calibration=None):
        self.path = path
        self.ix = McapIndex(path)
        vt = self.ix.topic("side_by_side") or next((t for t in self.ix.msgs if t.endswith(("/left", "/stereo"))), None)
        if vt is None:
            raise ValueError(f"{path}: no video channel")
        self._v = self.ix.msgs[vt]
        self.n = len(self._v)
        self.t_ns = np.array([m[0] for m in self._v], np.int64)
        hdr = self.ix.msgs.get("svo_header")
        self.header = json.loads(self.ix.read(*hdr[0][1:])) if hdr else {}
        self.serial = int(vt.split("_SN")[1].split("/")[0]) if "_SN" in vt else None
        h = base64.b64decode(self.header["header"]) if self.header.get("header") else b""
        self.width = self.height = self.fps = 0
        if len(h) >= 16:
            self.width, self.height, self.fps = (int.from_bytes(h[0:4], "little"), int.from_bytes(h[4:8], "little"),
                                                 int.from_bytes(h[12:16], "little"))
        if not self.width and self.n:
            d = self._raw(0)
            self.width, self.height = int.from_bytes(d[-56:-52], "little") // 2, int.from_bytes(d[-52:-48], "little")
        if not self.fps and self.n > 1:
            self.fps = int(round(1e9 / np.median(np.diff(self.t_ns))))
        self._codec_name = None
        self._mu = threading.Lock()
        self._codec = None
        self._last = None
        self._key = {}
        self._maps = None
        self._sensors = None
        self.calib, self.calib_source = _calib.find(path, self.serial, self.width, self.height, calibration)

    # ------------------------------------------------------------------ video
    def _raw(self, i):
        _, off, size = self._v[i]
        return self.ix.read(off, size)

    def _payload(self, i):
        d = self._raw(i)
        n = int.from_bytes(d[4:8], "little")
        return d[8:8 + n]

    def _codec_for(self, _b=None):
        if self._codec_name is None:          # decided from frame 0 (a keyframe, carries the parameter sets)
            types = _nal_types(self._payload(0)[:4096])
            hevc = any(((t >> 1) & 0x3F) in (32, 33, 34) for t in types)   # VPS / SPS / PPS
            self._codec_name = "hevc" if hevc else "h264"
        return self._codec_name

    def is_keyframe(self, i):
        if i not in self._key:
            b = self._payload(i)[:4096]
            types = _nal_types(b)
            if self._codec_for(b) == "hevc":
                self._key[i] = any(16 <= ((t >> 1) & 0x3F) <= 21 for t in types)
            else:
                self._key[i] = any((t & 0x1F) == 5 for t in types)
        return self._key[i]

    def yuv(self, i):
        """Decoded planes (Y: H x 2W, U, V: H/2 x W) of frame i, exactly as stored. Random access decodes from the
        previous keyframe; sequential access decodes one frame."""
        if not 0 <= i < self.n:
            raise IndexError(i)
        with self._mu:
            if self._codec is None or self._last is None or not (self._last < i <= self._last + 2 * max(self.fps, 1)):
                k = i
                while k > 0 and not self.is_keyframe(k):
                    k -= 1
                self._codec = av.CodecContext.create(self._codec_for(self._payload(k)), "r")
                self._codec.thread_type = "SLICE"   # frame threading would delay output by several packets
                start = k
            else:
                start = self._last + 1
            fr = None
            for j in range(start, i + 1):
                out = self._codec.decode(av.Packet(self._payload(j)))
                self._last = j
                fr = out[0] if out else None
            if fr is None:
                self._codec = None
                raise RuntimeError(f"frame {i}: the decoder returned no picture")
            a = fr.to_ndarray(format="yuv420p")
        H, W = a.shape[0] * 2 // 3, a.shape[1]
        return a[:H], a[H:H + H // 4].reshape(H // 2, W // 2), a[H + H // 4:].reshape(H // 2, W // 2)

    def image(self, i, look="camera", rect=False):
        """Side-by-side BGR uint8 (H x 2W x 3). look: "camera" (as stored) or "sdk" (approximate SDK sharpening)."""
        Y, U, V = self.yuv(i)
        H, W = Y.shape
        Uf = cv2.resize(U, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128
        Vf = cv2.resize(V, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128
        Yf = Y.astype(np.float32)
        bgr = np.stack([Yf + 1.772 * Uf, Yf - 0.344136 * Uf - 0.714136 * Vf, Yf + 1.402 * Vf], -1)
        if look == "sdk":
            k = np.array([[0, -.5, 0], [-.5, 3, -.5], [0, -.5, 0]], np.float32)
            bgr = cv2.filter2D(bgr, -1, k, borderType=cv2.BORDER_REPLICATE)
        out = np.clip(np.round(bgr), 0, 255).astype(np.uint8)
        if rect:
            w = W // 2
            (lx, ly), (rx, ry) = self.rect_maps()
            out = np.concatenate([cv2.remap(out[:, :w], lx, ly, cv2.INTER_LINEAR),
                                  cv2.remap(out[:, w:], rx, ry, cv2.INTER_LINEAR)], 1)
        return out

    def rect_maps(self):
        if self.calib is None:
            raise ValueError("no calibration found for this recording (see svo2view.calib)")
        if self._maps is None:
            self._maps = _calib.rect_maps(self.calib, self.width, self.height)
        return self._maps

    # ------------------------------------------------------------------ sensors
    def sensors(self):
        """Every sensor sample: dict of numpy arrays (values as stored, float32; times in ns)."""
        if self._sensors is None:
            t = self.ix.topic("sensors")
            msgs = self.ix.msgs.get(t, []) if t else []
            raw = []
            with open(self.path, "rb") as f:
                for _, off, size in msgs:
                    f.seek(off)
                    b = base64.b64decode(json.loads(f.read(size))["data"])
                    raw.append(b[:SENSOR_STRUCT].ljust(SENSOR_STRUCT, b"\0"))
            B = np.frombuffer(b"".join(raw), np.uint8).reshape(len(raw), SENSOR_STRUCT)
            out = {"t": np.array([m[0] for m in msgs], np.int64)}
            for k, (o, n) in SENSOR_F32.items():
                a = B[:, o:o + 4 * n].copy().view("<f4")
                out[k] = a[:, 0] if n == 1 else a
            for k, o in SENSOR_U64.items():
                out[k] = B[:, o:o + 8].copy().view("<u8")[:, 0].astype(np.int64)
            self._sensors = out
        return self._sensors

    def info(self):
        return {"path": self.path, "n": self.n, "width": self.width, "height": self.height, "fps": self.fps,
                "serial": self.serial, "sdk": self.header.get("ZED_SDK_version"),
                "t0_ns": int(self.t_ns[0]) if self.n else None,
                "duration_s": float((self.t_ns[-1] - self.t_ns[0]) / 1e9) if self.n > 1 else 0.0,
                "calibration": self.calib_source, "parity": PARITY}
