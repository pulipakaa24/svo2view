"""Write a small synthetic SVO2-layout file (no camera, no ZED SDK): MCAP with the ZED channel names, lossless H.265
side-by-side video with a GOP of `fps` frames, the 360-byte sensor struct, and a header. Used by the tests."""
import base64
import json
import struct

import av
import numpy as np
from mcap.writer import CompressionType, Writer

from svo2view.reader import SENSOR_F32, SENSOR_STRUCT, SENSOR_U64

SERIAL = 12345678


def frames_yuv(n, w, h, seed=0):
    """n random-textured side-by-side YUV420 frames (Y: h x 2w)."""
    rng = np.random.default_rng(seed)
    out = []
    base = rng.integers(0, 256, (h, 2 * w), dtype=np.uint8)
    for i in range(n):
        Y = np.roll(base, i, axis=1)
        U = np.full((h // 2, w), 100 + i % 50, np.uint8)
        V = np.full((h // 2, w), 150 - i % 40, np.uint8)
        out.append((Y, U, V))
    return out


def encode_hevc_lossless(frames, fps):
    W, H = frames[0][0].shape[1], frames[0][0].shape[0]
    cc = av.CodecContext.create("libx265", "w")
    cc.width, cc.height, cc.pix_fmt = W, H, "yuv420p"
    from fractions import Fraction
    cc.time_base = Fraction(1, fps)
    cc.options = {"x265-params": f"lossless=1:keyint={fps}:min-keyint={fps}:bframes=0:repeat-headers=1:log-level=error"}
    pkts = []
    for i, (Y, U, V) in enumerate(frames):
        f = av.VideoFrame(W, H, "yuv420p")
        for plane, arr in zip(f.planes, (Y, U, V)):
            buf = np.frombuffer(plane, np.uint8).reshape(-1, plane.line_size)
            buf[:arr.shape[0], :arr.shape[1]] = arr
        f.pts = i
        pkts += [bytes(p) for p in cc.encode(f)]
    pkts += [bytes(p) for p in cc.encode(None)]
    assert len(pkts) == len(frames), (len(pkts), len(frames))
    return pkts


def sensor_struct(k, t_ns):
    b = bytearray(SENSOR_STRUCT)
    vals = {}
    for name, (off, n) in SENSOR_F32.items():
        v = np.float32(np.arange(n) + k * 0.25 + off / 1000.0)
        b[off:off + 4 * n] = v.tobytes()
        vals[name] = v
    for name, off in SENSOR_U64.items():
        b[off:off + 8] = struct.pack("<Q", t_ns + off)
        vals[name] = t_ns + off
    return bytes(b), vals


def write(path, n=12, w=64, h=48, fps=5, sensors_per_frame=4, t0=1_700_000_000_000_000_000):
    frames = frames_yuv(n, w, h)
    pkts = encode_hevc_lossless(frames, fps)
    times = [t0 + i * 1_000_000_000 // fps for i in range(n)]
    sens = []
    with open(path, "wb") as f:
        wr = Writer(f, compression=CompressionType.NONE)
        wr.start(profile="", library="svo2view-tests")
        hdr = struct.pack("<IIII", w, h, 0, fps) + bytes(64)
        ch_h = wr.register_channel("svo_header", "json", 0)
        ch_v = wr.register_channel(f"Camera_SN{SERIAL}/side_by_side", "zed_sdk_encoded", 0)
        ch_s = wr.register_channel(f"Camera_SN{SERIAL}/sensors", "json", 0)
        wr.add_message(ch_h, t0 - 1, json.dumps({"header": base64.b64encode(hdr).decode(), "ZED_SDK_version": "synthetic"}).encode(), t0 - 1)
        k = 0
        for i, (t, p) in enumerate(zip(times, pkts)):
            trailer = struct.pack("<II", 2 * w, h) + bytes(48)
            body = struct.pack("<I", len(p)) + p + trailer
            wr.add_message(ch_v, t, struct.pack("<I", len(body)) + body, t)
            for j in range(sensors_per_frame):
                ts = t + j * (1_000_000_000 // fps // sensors_per_frame)
                b, vals = sensor_struct(k, ts)
                wr.add_message(ch_s, ts, json.dumps({"__title__": "sensors", "data": base64.b64encode(b).decode()}).encode(), ts)
                sens.append((ts, vals))
                k += 1
        wr.finish()
    return {"frames": frames, "times": times, "sensors": sens, "w": w, "h": h, "fps": fps}
