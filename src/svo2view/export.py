"""svo2view export: turn an SVO2 recording into files anyone can open without svo2view.

    svo2view export rec.svo2 [--out DIR] [--rect] [--view sbs|left|right|anaglyph] [--width 3840] [--crf 18]

writes, next to the recording or into DIR:

* <stem>.mp4          H.264 video at the recorded timing: frame times are the recorded image timestamps, so a dropped
                      frame holds the previous image for as long as the camera really missed it
* <stem>_frames.csv   one row per frame: index, timestamp (ns), seconds from the first frame, gap to the previous frame
                      (ms), keyframe
* <stem>_sensors.csv  every sensor sample: IMU (calibrated and raw), orientation quaternion, magnetometer, barometer,
                      temperatures, with each sensor's own timestamp
"""
import argparse
import csv
import os
import sys
from fractions import Fraction

import av
import numpy as np

from .reader import SVO2


def _tmp(path):
    return path + ".part"


def export_frames_csv(s, path):
    t = s.t_ns
    with open(_tmp(path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_ns", "t_s", "gap_ms", "keyframe"])
        for i in range(s.n):
            w.writerow([i, int(t[i]), f"{(t[i] - t[0]) / 1e9:.6f}", "" if i == 0 else f"{(t[i] - t[i - 1]) / 1e6:.3f}",
                        int(s.is_keyframe(i))])
    os.replace(_tmp(path), path)


def export_sensors_csv(s, path):
    S = s.sensors()
    cols, data = [], []
    t0 = int(s.t_ns[0]) if s.n else int(S["t"][0])
    cols += ["t_ns", "t_s"]
    data += [S["t"], (S["t"] - t0) / 1e9]
    axes = {"quat_xyzw": "xyzw", "acc": "xyz", "gyro": "xyz", "acc_raw": "xyz", "gyro_raw": "xyz", "mag": "xyz",
            "mag_raw": "xyz"}
    units = {"acc": "m_s2", "acc_raw": "m_s2", "gyro": "deg_s", "gyro_raw": "deg_s", "mag": "uT", "mag_raw": "uT"}
    for k in ["acc", "gyro", "quat_xyzw", "acc_raw", "gyro_raw", "t_imu", "mag", "mag_raw", "t_mag", "pressure_pa",
              "rel_alt_m", "temp_baro", "t_baro", "temp_imu", "temp_left", "temp_right"]:
        a = S[k]
        if a.ndim == 2:
            for j, ax in enumerate(axes[k]):
                cols.append(f"{k.replace('_xyzw', '')}_{ax}" + (f"_{units[k]}" if k in units else ""))
                data.append(a[:, j])
        else:
            cols.append(k + ("_C" if k.startswith("temp") else ""))
            data.append(a)
    with open(_tmp(path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in range(len(S["t"])):
            w.writerow([f"{c[r]:.6f}" if c.dtype.kind == "f" else int(c[r]) for c in data])
    os.replace(_tmp(path), path)


def export_mp4(s, path, view="sbs", rect=False, width=3840, crf=18, look="camera", progress=True):
    eye_w = s.width
    full = eye_w * (1 if view in ("left", "right", "anaglyph") else 2)
    out_w = min(width or full, full)
    out_w -= out_w % 2
    out_h = round(s.height * out_w / full)
    out_h -= out_h % 2
    tb = Fraction(1, 1000)
    with av.open(_tmp(path), "w", format="mp4") as box:
        st = box.add_stream("libx264", rate=s.fps)
        st.width, st.height, st.pix_fmt = out_w, out_h, "yuv420p"
        st.time_base = tb
        st.options = {"crf": str(crf), "preset": "medium", "tune": "film"}
        st.codec_context.time_base = tb
        t0, last = int(s.t_ns[0]), -1
        for i, Y, U, V in s.stream(0):
            bgr = s.display(Y, U, V, look=look, rect=rect, view=view, width=out_w)
            if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
                bgr = np.ascontiguousarray(bgr[:out_h, :out_w])
            fr = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            pts = int(round((int(s.t_ns[i]) - t0) / 1e6))
            fr.pts, fr.time_base = max(pts, last + 1), tb
            last = fr.pts
            for p in st.encode(fr):
                box.mux(p)
            if progress and (i % 50 == 0 or i == s.n - 1):
                print(f"\r  video {i + 1}/{s.n}", end="", file=sys.stderr, flush=True)
        for p in st.encode():
            box.mux(p)
    if progress:
        print(file=sys.stderr)
    os.replace(_tmp(path), path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="svo2view export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("svo2")
    ap.add_argument("--out", help="output folder (default: next to the recording)")
    ap.add_argument("--calib", help="calibration .json or factory SN<serial>.conf (for --rect)")
    ap.add_argument("--rect", action="store_true", help="rectified video (needs a calibration)")
    ap.add_argument("--view", default="sbs", choices=["sbs", "left", "right", "anaglyph"])
    ap.add_argument("--width", type=int, default=3840, help="video width in pixels (default 3840; 0 = full)")
    ap.add_argument("--crf", type=int, default=18, help="H.264 quality, lower is better (default 18)")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args(argv)
    s = SVO2(a.svo2, calibration=a.calib)
    out = a.out or os.path.dirname(os.path.abspath(a.svo2))
    os.makedirs(out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.svo2))[0]
    if a.rect and s.calib is None:
        print("no calibration found: pass --calib, or drop --rect", file=sys.stderr)
        return 2
    print(f"{a.svo2}: {s.n} frames, {s.width}x{s.height} per eye @ {s.fps} fps, calibration: {s.calib_source}",
          file=sys.stderr)
    export_frames_csv(s, os.path.join(out, stem + "_frames.csv"))
    export_sensors_csv(s, os.path.join(out, stem + "_sensors.csv"))
    if not a.no_video:
        export_mp4(s, os.path.join(out, stem + ".mp4"), view=a.view, rect=a.rect, width=a.width, crf=a.crf)
    print(f"wrote {stem}.mp4 / _frames.csv / _sensors.csv to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
