#!/usr/bin/env python3
"""Export the calibration the ZED SDK applies, for exact rectification in svo2view. Needs the ZED SDK + pyzed.

    python zed_export_calib.py RECORDING.svo2 [OUT.json]       # from a recording
    python zed_export_calib.py --live HD720 [OUT.json]          # from a connected camera at a resolution

Writes the raw intrinsics, the SDK's 8-coefficient rational distortion, the stereo extrinsics and the SDK's rectified
intrinsics (self-calibration off, so the factory calibration). Default output: <recording>.calib.json next to the
recording (svo2view finds it there), or SN<serial>_<W>x<H>.json for --live (copy it to ~/.svo2view/).
"""
import argparse
import json
import os

import numpy as np
import pyzed.sl as sl

ap = argparse.ArgumentParser()
ap.add_argument("src", nargs="?")
ap.add_argument("out", nargs="?")
ap.add_argument("--live", metavar="RESOLUTION", help="e.g. HD720, HD1080, HD2K, HD1200, SVGA, VGA")
a = ap.parse_args()

ip = sl.InitParameters()
ip.depth_mode = sl.DEPTH_MODE.NONE
ip.camera_disable_self_calib = True
if a.live:
    ip.camera_resolution = getattr(sl.RESOLUTION, a.live)
    out = a.src
else:
    ip.set_from_svo_file(a.src)
    out = a.out
cam = sl.Camera()
err = cam.open(ip)
if err != sl.ERROR_CODE.SUCCESS:
    raise SystemExit(f"open failed: {err}")
info = cam.get_camera_information()
cc = info.camera_configuration
raw, rect = cc.calibration_parameters_raw, cc.calibration_parameters
K = lambda c: [[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]]  # noqa: E731
res = [cc.resolution.width, cc.resolution.height]
d = {"serial": info.serial_number, "resolution": res, "sdk": sl.Camera.get_sdk_version(), "self_calib": False,
     "raw": {"left_K": K(raw.left_cam), "right_K": K(raw.right_cam), "left_disto": list(raw.left_cam.disto),
             "right_disto": list(raw.right_cam.disto),
             "stereo_R": np.array(raw.stereo_transform.get_rotation_matrix().r).tolist(),
             "stereo_T_m": list(raw.stereo_transform.get_translation().get())},
     "rect": {"K": K(rect.left_cam), "baseline_m": float(rect.stereo_transform.get_translation().get()[0])},
     "imu_to_left_cam": np.array(info.sensors_configuration.camera_imu_transform.m).tolist()}
cam.close()
if not out:
    out = (os.path.splitext(a.src)[0] + ".calib.json") if not a.live else f"SN{d['serial']}_{res[0]}x{res[1]}.json"
with open(out, "w") as f:
    json.dump(d, f, indent=1)
print(out)
