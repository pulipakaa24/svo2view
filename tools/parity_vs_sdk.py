#!/usr/bin/env python3
"""Parity of svo2view against the ZED SDK's own output (an .npz written by tools/zed_truth.py on a machine with the SDK).

    python tools/parity_vs_sdk.py REC.svo2 TRUTH.npz [--calib REC.calib.json]

Gates (exit 1 if one fails): frame timestamps exact; every sensor value exact (float32); rectification geometry
(the SDK's unrectified image remapped by our maps vs the SDK's rectified image, mean |err| <= 1.0 level). Reported, not
gated: our decoded image vs the SDK's (the SDK's playback sharpening is approximated).
"""
import sys

import numpy as np

import argparse

import cv2

from svo2view import SVO2

ap = argparse.ArgumentParser()
ap.add_argument("svo")
ap.add_argument("truth")
ap.add_argument("--calib")
a = ap.parse_args()
s, z = SVO2(a.svo, calibration=a.calib), np.load(a.truth)
ok = True


def gate(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")


gate("frame count", s.n == int(z["n_frames"]), f"{s.n} vs SDK {int(z['n_frames'])}")
gate("image timestamps", np.array_equal(s.t_ns, z["image_t"]), "all frames exact" if np.array_equal(s.t_ns, z["image_t"]) else "differ")
S = s.sensors()
gate("sensor sample count", len(S["t"]) == len(z["s_imu_t"]), f"{len(S['t'])} vs {len(z['s_imu_t'])}")
q = np.abs(S["quat_xyzw"].astype(np.float64) - z["s_imu_quat"]).max()
gate("sensor quat", q <= 1e-6, f"max |d| {q:.2g} (<= 1e-6: the SDK hands the stored float32 quaternion back renormalised)")
pairs = [("t_imu", "s_imu_t"), ("t_mag", "s_mag_t"), ("t_baro", "s_baro_t"), ("mag", "s_mag_cal"),
         ("mag_raw", "s_mag_unc"), ("pressure_pa", "s_baro_p"), ("rel_alt_m", "s_baro_alt"), ("temp_imu", "s_temp_imu"),
         ("temp_baro", "s_temp_baro"), ("temp_left", "s_temp_left"), ("temp_right", "s_temp_right")]
for a, b in pairs:
    x, y = np.asarray(S[a]), np.asarray(z[b])
    same = np.array_equal(x.astype(np.float64) if x.dtype.kind == "f" else x, y.astype(x.dtype).astype(np.float64) if x.dtype.kind == "f" else y)
    gate(f"sensor {a}", same, "exact" if same else f"max |d| {np.abs(x.astype(float) - y.astype(float)).max():.3g}")
# older truth dumps stored acc / gyro from a buffer pyzed reuses, so every row holds the LAST sample: compare that row
for a, b in (("acc", "s_imu_acc"), ("gyro", "s_imu_gyro"), ("acc_raw", "s_imu_acc_unc"), ("gyro_raw", "s_imu_gyro_unc")):
    y = z[b]
    stale = (y == y[-1]).all()
    cmp = (S[a][-1:], y[-1:]) if stale else (S[a], y)
    same = np.array_equal(cmp[0].astype(np.float64), cmp[1].astype(np.float32).astype(np.float64))
    gate(f"sensor {a}", same, ("exact" if same else "differ") + (" (truth dump has the stale-buffer bug: last row only)" if stale else ""))

frames = list(z["frames_dumped"])
w = s.width
(lx, ly), (rx, ry) = s.rect_maps()
m = np.zeros((s.height, w), bool)
m[20:-20, 20:-20] = True
for j, f in enumerate(frames):
    ours = s.image(int(f), look="sdk")
    cam = s.image(int(f), look="camera")
    sdk_u = np.concatenate([z["img_LEFT_UNRECTIFIED"][j][..., :3], z["img_RIGHT_UNRECTIFIED"][j][..., :3]], 1)
    sdk_r = np.concatenate([z["img_LEFT"][j][..., :3], z["img_RIGHT"][j][..., :3]], 1)
    e = np.abs(ours.astype(np.int16) - sdk_u)
    ec = np.abs(cam.astype(np.int16) - sdk_u)
    geo = []
    for src, tgt, mx, my in ((sdk_u[:, :w], sdk_r[:, :w], lx, ly), (sdk_u[:, w:], sdk_r[:, w:], rx, ry)):
        geo.append(float(np.abs(cv2.remap(src, mx, my, cv2.INTER_LINEAR).astype(np.int16) - tgt)[m].mean()))
    full = np.abs(s.image(int(f), look="sdk", rect=True).astype(np.int16) - sdk_r)
    print(f"      frame {f}: image sdk-look |err| {e.mean():.3f} ({(e == 0).mean() * 100:.1f} % exact, {(e <= 2).mean() * 100:.1f} % <=2); "
          f"camera-look {ec.mean():.3f}; rect geometry L/R {geo[0]:.3f}/{geo[1]:.3f}; full chain rect {full.mean():.3f}")
    gate(f"rectification geometry frame {f}", max(geo) <= 1.0, f"{max(geo):.3f} <= 1.0")
print("ALL GATES PASS" if ok else "SOME GATES FAILED")
sys.exit(0 if ok else 1)
