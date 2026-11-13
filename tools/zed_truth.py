#!/usr/bin/env python3
"""Ground truth from the ZED SDK for svo2view's parity check (tools/parity_vs_sdk.py). Needs the ZED SDK + pyzed.

    python zed_truth.py IN.svo2 OUT.npz [--self-calib] [--frames 0,1,2,100,-1]

Writes, through pyzed only (the reference implementation):
  * per-frame image timestamps (ns) and SVO positions for the whole file;
  * for a few frames: LEFT_UNRECTIFIED, RIGHT_UNRECTIFIED, LEFT, RIGHT (BGRA uint8, full resolution);
  * the calibration the SDK applied (rectified and raw: K, distortion, stereo R / T) and the camera / sensor configuration;
  * every sensor sample of the file (get_sensors_data_batch): IMU (calibrated + uncalibrated, orientation), magnetometer
    (calibrated + uncalibrated, heading), barometer (pressure, relative altitude), temperatures, each with its timestamp.
Self-calibration is OFF by default so the calibration is the factory file's; --self-calib leaves the SDK default.
"""
import argparse

import numpy as np

import pyzed.sl as sl

ap = argparse.ArgumentParser()
ap.add_argument("svo")
ap.add_argument("out")
ap.add_argument("--self-calib", action="store_true")
ap.add_argument("--frames", default="0,1,2,100,250,-1")
ap.add_argument("--no-enhance", action="store_true", help="InitParameters.enable_image_enhancement = False")
a = ap.parse_args()

ip = sl.InitParameters()
ip.set_from_svo_file(a.svo)
ip.svo_real_time_mode = False
ip.depth_mode = sl.DEPTH_MODE.NONE
ip.camera_disable_self_calib = not a.self_calib
if a.no_enhance:
    ip.enable_image_enhancement = False
ip.coordinate_units = sl.UNIT.METER
cam = sl.Camera()
err = cam.open(ip)
assert err == sl.ERROR_CODE.SUCCESS, err
info = cam.get_camera_information()
n = cam.get_svo_number_of_frames()
want = sorted({(int(f) % n) for f in a.frames.split(",")})
out = {"n_frames": n, "self_calib": a.self_calib, "enhance": not a.no_enhance, "serial": info.serial_number, "sdk": sl.Camera.get_sdk_version(),
       "frames_dumped": np.array(want)}


def cam_params(prefix, cp):
    out[prefix + "_K"] = np.array([[cp.fx, 0, cp.cx], [0, cp.fy, cp.cy], [0, 0, 1]], np.float64)
    out[prefix + "_disto"] = np.array(cp.disto, np.float64)
    out[prefix + "_size"] = np.array([cp.image_size.width, cp.image_size.height])


cc = info.camera_configuration
for tag, cal in (("rect", cc.calibration_parameters), ("raw", cc.calibration_parameters_raw)):
    cam_params(tag + "_left", cal.left_cam)
    cam_params(tag + "_right", cal.right_cam)
    out[tag + "_stereo_T"] = np.array(cal.stereo_transform.get_translation().get(), np.float64)
    out[tag + "_stereo_R"] = np.array(cal.stereo_transform.get_rotation_matrix().r, np.float64)
    out[tag + "_stereo_m"] = np.array(cal.stereo_transform.m, np.float64)
out["resolution"] = np.array([cc.resolution.width, cc.resolution.height])
out["fps"] = cc.fps
sc = info.sensors_configuration
out["imu_to_left_cam"] = np.array(sc.camera_imu_transform.m, np.float64)

ts, pos = [], []
imgs = {k: [] for k in ("LEFT_UNRECTIFIED", "RIGHT_UNRECTIFIED", "LEFT", "RIGHT")}
S = {k: [] for k in ("imu_t", "imu_acc", "imu_gyro", "imu_acc_unc", "imu_gyro_unc", "imu_quat", "mag_t", "mag_cal",
                     "mag_unc", "mag_heading", "baro_t", "baro_p", "baro_alt", "temp_imu", "temp_baro", "temp_left",
                     "temp_right", "batch_frame")}
mat = sl.Mat()
rt = sl.RuntimeParameters()
i = 0
while cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
    ts.append(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
    pos.append(cam.get_svo_position())
    if i in want:
        for k in imgs:
            cam.retrieve_image(mat, getattr(sl.VIEW, k))
            imgs[k].append(mat.get_data().copy())
    batch = []
    cam.get_sensors_data_batch(batch)
    for s in batch:
        imu, mag, bar, tmp = s.get_imu_data(), s.get_magnetometer_data(), s.get_barometer_data(), s.get_temperature_data()
        S["imu_t"].append(imu.timestamp.get_nanoseconds())
        S["imu_acc"].append(np.array(imu.get_linear_acceleration(), np.float64))  # copy: pyzed reuses the buffer
        S["imu_gyro"].append(np.array(imu.get_angular_velocity(), np.float64))
        S["imu_acc_unc"].append(np.array(imu.get_linear_acceleration_uncalibrated(), np.float64))
        S["imu_gyro_unc"].append(np.array(imu.get_angular_velocity_uncalibrated(), np.float64))
        S["imu_quat"].append(imu.get_pose().get_orientation().get())
        S["mag_t"].append(mag.timestamp.get_nanoseconds())
        S["mag_cal"].append(np.array(mag.get_magnetic_field_calibrated(), np.float64))
        S["mag_unc"].append(np.array(mag.get_magnetic_field_uncalibrated(), np.float64))
        S["mag_heading"].append(mag.magnetic_heading)
        S["baro_t"].append(bar.timestamp.get_nanoseconds())
        S["baro_p"].append(bar.pressure)
        S["baro_alt"].append(bar.relative_altitude)
        for k, loc in (("temp_imu", "IMU"), ("temp_baro", "BAROMETER"), ("temp_left", "ONBOARD_LEFT"),
                       ("temp_right", "ONBOARD_RIGHT")):
            S[k].append(tmp.get(getattr(sl.SENSOR_LOCATION, loc)))
        S["batch_frame"].append(i)
    i += 1
cam.close()
out["image_t"] = np.array(ts, np.int64)
out["svo_pos"] = np.array(pos, np.int64)
for k, v in imgs.items():
    out["img_" + k] = np.stack(v)
for k, v in S.items():
    out["s_" + k] = np.array(v, np.int64 if k.endswith("_t") or k == "batch_frame" else np.float64)
np.savez_compressed(a.out, **out)
print(f"frames {i}/{n}, dumped {want}, imu samples {len(S['imu_t'])}, unique imu t {len(set(S['imu_t']))}")
