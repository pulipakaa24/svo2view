"""Stereo calibration for SVO2 recordings, and rectification maps that reproduce the ZED SDK's.

Two sources, best first:

1. **SDK calibration JSON** (exact): what the ZED SDK reports for the recording, written by
   ``tools/zed_export_calib.py`` on a machine with the SDK. The SDK applies an 8-coefficient rational distortion model
   (k1 k2 p1 p2 k3 k4 k5 k6) that is in neither the SVO2 file nor the factory ``SN<serial>.conf``.
2. **Factory ``SN<serial>.conf``** (approximate): the camera's calibration file (``/usr/local/zed/settings/`` on a
   machine with the SDK, or downloadable from Stereolabs by serial number). It holds a 5-coefficient Brown model per
   resolution; the rectified intrinsics then come from OpenCV (no black borders), not from the SDK.

Rectification rule (measured against SDK 5.5 output, 0.06 px median): split the stereo rotation in half between the two
cameras, then rotate the (half-rotated) baseline onto +x. JSON schema::

    {"serial": 123, "resolution": [1280, 720],
     "raw": {"left_K": 3x3, "right_K": 3x3, "left_disto": [k1..k6, ...], "right_disto": [...],
             "stereo_R": 3x3, "stereo_T_m": [tx, ty, tz]},
     "rect": {"K": 3x3}}            # optional; computed when absent
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

# SDK resolution (per-eye width, height) -> section suffix in SN<serial>.conf
CONF_SECTIONS = {(2208, 1242): "2K", (1920, 1080): "FHD", (1280, 720): "HD", (672, 376): "VGA",
                 (1920, 1200): "FHD1200", (960, 600): "SVGA", (3840, 2160): "4k"}


def _rod(v):
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def read_conf(path):
    out, sec = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                sec = line.strip("[]")
                out[sec] = {}
            elif "=" in line and sec:
                k, v = line.split("=", 1)
                try:
                    out[sec][k.strip()] = float(v)
                except ValueError:
                    out[sec][k.strip()] = v.strip()
    return out


def from_conf(path, width, height):
    """Calibration dict (schema above, without rect K) from a factory SN<serial>.conf for one resolution."""
    conf = read_conf(path)
    sec = CONF_SECTIONS.get((width, height))
    if sec is None or f"LEFT_CAM_{sec}" not in conf:
        raise ValueError(f"{path}: no section for {width}x{height}")
    raw = {}
    for side in ("left", "right"):
        c = conf[f"{side.upper()}_CAM_{sec}"]
        raw[f"{side}_K"] = [[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]]
        raw[f"{side}_disto"] = [c.get("k1", 0), c.get("k2", 0), c.get("p1", 0), c.get("p2", 0), c.get("k3", 0)]
    st = conf["STEREO"]
    rv = [st.get(f"RX_{sec}", 0.0), st.get(f"CV_{sec}", 0.0), st.get(f"RZ_{sec}", 0.0)]
    raw["stereo_R"] = _rod(rv).tolist()          # == the SDK's stereo rotation (checked to 3e-8)
    raw["stereo_T_m"] = [st["Baseline"] / 1000, st.get("TY", 0.0) / 1000, st.get("TZ", 0.0) / 1000]
    return {"resolution": [width, height], "raw": raw, "source": f"factory conf {os.path.basename(path)} (approximate)"}


def rect_rotations(calib):
    """(R_left, R_right): rotation from the rectified frame to each camera (x_cam = R @ x_rect)."""
    R = np.array(calib["raw"]["stereo_R"], np.float64)
    T = np.array(calib["raw"]["stereo_T_m"], np.float64)
    rv = cv2.Rodrigues(R)[0].ravel()
    hL, hR = _rod(-rv / 2), _rod(rv / 2)
    t = hR @ T
    t = t / np.linalg.norm(t)
    ax = np.cross(t, [1.0, 0.0, 0.0])
    s = np.linalg.norm(ax)
    Ra = _rod(ax / s * np.arcsin(min(s, 1.0))) if s > 0 else np.eye(3)
    return hL @ Ra, hR @ Ra   # both cameras share one rectified orientation (R_right @ R_left.T == stereo R exactly)


def _disto(c, side):
    d = np.array(c["raw"][f"{side}_disto"], np.float64)
    return d[:8] if len(d) >= 8 else d


def rect_K(calib, width, height):
    if calib.get("rect", {}).get("K"):
        return np.array(calib["rect"]["K"], np.float64)
    K1, K2 = (np.array(calib["raw"][f"{s}_K"], np.float64) for s in ("left", "right"))
    R = np.array(calib["raw"]["stereo_R"], np.float64)
    T = np.array(calib["raw"]["stereo_T_m"], np.float64).reshape(3, 1)
    P1 = cv2.stereoRectify(K1, _disto(calib, "left"), K2, _disto(calib, "right"), (width, height), R, T,
                           flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)[2]
    f = P1[0, 0]
    return np.array([[f, 0, P1[0, 2]], [0, f, P1[1, 2]], [0, 0, 1]], np.float64)


def rect_maps(calib, width, height):
    """cv2.remap maps ((left_x, left_y), (right_x, right_y)) for the rectified view."""
    RL, RR = rect_rotations(calib)
    Kr = rect_K(calib, width, height)
    out = []
    for side, Rc in (("left", RL), ("right", RR)):
        K = np.array(calib["raw"][f"{side}_K"], np.float64)
        out.append(cv2.initUndistortRectifyMap(K, _disto(calib, side), Rc.T, Kr, (width, height), cv2.CV_32FC1))
    return out


def find(svo_path, serial, width, height, explicit=None):
    """(calib dict or None, description). Search order: explicit path; next to the recording; ~/.svo2view; the SDK's
    settings folder."""
    d = os.path.dirname(os.path.abspath(svo_path))
    stem = os.path.splitext(os.path.basename(svo_path))[0]
    res = f"{width}x{height}"
    jsons = [explicit] if explicit and explicit.endswith(".json") else []
    jsons += [os.path.join(d, f"{stem}.calib.json"), os.path.join(d, "sdk_calibration.json"),
              os.path.join(d, f"SN{serial}_{res}.json"),
              os.path.join(os.path.expanduser("~/.svo2view"), f"SN{serial}_{res}.json")]
    for p in jsons:
        if p and os.path.exists(p):
            with open(p) as f:
                c = json.load(f)
            if list(c.get("resolution", [width, height])) == [width, height]:
                return c, f"SDK calibration {os.path.basename(p)} (exact)"
    confs = [explicit] if explicit and explicit.endswith(".conf") else []
    confs += [os.path.join(d, f"SN{serial}.conf"), os.path.join(os.path.expanduser("~/.svo2view"), f"SN{serial}.conf"),
              f"/usr/local/zed/settings/SN{serial}.conf", os.path.expanduser(f"~/zed/settings/SN{serial}.conf")]
    for p in confs:
        if p and os.path.exists(p):
            try:
                c = from_conf(p, width, height)
                return c, c["source"]
            except (ValueError, KeyError):
                continue
    return None, "none (rectification unavailable)"
