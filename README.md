# svo2view

Read and view **Stereolabs ZED SVO2 recordings without the ZED SDK or an NVIDIA GPU**: on a Mac, on any Linux or Windows
machine. Frames, timestamps, IMU / magnetometer / barometer / temperatures, and rectification that reproduces the SDK's,
in a small Python library and a local web viewer.

```
pip install git+https://github.com/pulipakaa24/svo2view
svo2view path/to/recordings/          # opens http://127.0.0.1:8790 in the browser
```

The viewer shows side by side / left / right / red-cyan anaglyph, the stored image or an "SDK look", raw or rectified;
plays in real time at the recorded rate (or 0.25-4x), steps (← / →, shift = 10), scrubs, marks
dropped-frame gaps, saves any frame as a full-resolution PNG, and plots every sensor stream with a cursor at the current
frame (click a plot to seek). It listens on 127.0.0.1 only.

```python
from svo2view import SVO2

s = SVO2("recording.svo2")
s.n, s.fps, s.t_ns              # frame count, frame rate, per-frame image timestamps (ns)
bgr = s.image(100)              # side-by-side BGR uint8, H x 2W x 3
left = s.image(100, rect=True)[:, : s.width]
y, u, v = s.yuv(100)            # decoded planes exactly as stored
S = s.sensors()                 # dict of arrays: t, acc, gyro, acc_raw, gyro_raw, quat_xyzw, mag, mag_raw,
                                # pressure_pa, rel_alt_m, temp_imu / temp_baro / temp_left / temp_right, t_imu / t_mag / t_baro
```

### Export for people without svo2view

```
svo2view export rec.svo2 --rect          # -> rec.mp4, rec_frames.csv, rec_sensors.csv next to the recording
```

* `rec.mp4`: H.264, side by side (or `--view left|right|anaglyph`), raw or `--rect`ified, 3840 px wide by default
  (`--width 0` = full resolution). Frame times are the recorded image timestamps, so a dropped frame holds the previous
  image for as long as the camera actually missed it, and the video lasts exactly as long as the recording.
* `rec_frames.csv`: frame index, timestamp (ns), seconds from the first frame, gap to the previous frame (ms), keyframe.
* `rec_sensors.csv`: every sensor sample (about 400 Hz on a ZED 2i): acceleration and angular velocity (calibrated and
  raw), orientation quaternion, magnetometer, barometer pressure and relative altitude, temperatures, and each sensor's own
  timestamp.

A 103 s HD2K @ 15 recording (3.7 GB) exports to a 397 MB rectified MP4 in about a minute on an Apple-silicon Mac.

## How close to the ZED SDK it is

Measured against the ZED SDK 5.5.0's own output for the same file (ZED 2i, HD720 @ 30, `H265_LOSSLESS`; `tools/zed_truth.py`
on a Jetson with the SDK, then `tools/parity_vs_sdk.py`):

| what | result |
|---|---|
| frame count, image timestamps (`get_timestamp(IMAGE)`) | **exact**, every frame |
| sensors (`get_sensors_data_batch`, 6,129 samples) | **exact float32**: IMU raw + calibrated acceleration / angular velocity, magnetometer, barometer, temperatures and their timestamps; orientation quaternion within 2e-7 |
| decoded image | **exact** decode of the stored H.265 stream (`s.yuv`) |
| `LEFT_UNRECTIFIED` / `RIGHT_UNRECTIFIED` | **approximate** ("SDK look"): the SDK sharpens on playback; ours is within 0.94 grey levels on average, 94 % of pixels within 2 |
| rectification geometry | **matches** with an SDK calibration file: 0.06 px median (optical-flow fit); the SDK's own unrectified image through our maps differs from its rectified image by 0.83 levels, i.e. interpolation only |
| depth, positional tracking, object detection | **not reproduced** (closed SDK models) |

## Calibration (for rectification)

SVO2 files do not contain the lens calibration. svo2view looks for, in order:

1. **`<recording>.calib.json`** next to the recording (or `sdk_calibration.json`, or `--calib FILE`, or
   `~/.svo2view/SN<serial>_<W>x<H>.json`): the calibration the SDK applies, exported once per camera and resolution with
   `tools/zed_export_calib.py` on any machine with the ZED SDK. **Exact.** The SDK uses an 8-coefficient rational
   distortion model that is not in the factory file.
2. **`SN<serial>.conf`**, the factory calibration (next to the recording, `~/.svo2view/`, or `/usr/local/zed/settings/`;
   Stereolabs serves it by serial number). **Approximate**: the 5-coefficient model lands 1.8 px from the SDK's geometry
   (median; 5 px at most), and the rectified focal length is OpenCV's choice rather than the SDK's.

The viewer says which one it used.

## The SVO2 format, as measured

* **Container:** [MCAP](https://mcap.dev), uncompressed chunks. Channels: `svo_header` (JSON: IMU calibration and a base64
  header starting with u32 width, u32 height, u32 ?, u32 fps), `Camera_SN<serial>/side_by_side`, `/sensors`,
  `/integrated_sensors`, `/image_metadata`, `svo_footer` (timestamp index). The MCAP `log_time` of a video message is the
  SDK's image timestamp.
* **Video message:** `u32 rest_len`, `u32 bitstream_len`, an Annex-B H.265 (or H.264) access unit with the left and right
  images side by side, then a 56-byte trailer (u32 width x 2, u32 height, timestamps). **One keyframe per second** (GOP =
  fps), P-frames between; random access decodes from the previous keyframe.
* **Sensor message:** JSON `{"data": base64}` of a 360-byte little-endian struct: u64 timestamps at 16 (IMU), 32
  (barometer), 48 (magnetometer); float32 orientation x y z w at 60; angular velocity raw at 88, acceleration raw at 100,
  angular velocity at 112, acceleration at 124 (deg/s, m/s²); IMU temperature 228; magnetic field 236 and raw 248 (µT);
  barometer temperature 272, pressure 276 (Pa), relative altitude 284; left / right sensor temperatures 292 / 296.
* **HD2K is stored over-under** (2208 x 2484, left on top), since 2 x 2208 px exceeds the encoder's width; the reader returns
  side by side for every resolution.
* **Truncated files** (recording cut by a crash or power loss) open: every complete frame and sensor sample is recovered.
* **Rectification rule:** split the stereo rotation in half between the cameras, then turn the baseline onto x, with the
  SDK's rectified intrinsics; distortion is the SDK's rational model.

Tested with SVO2 files from ZED SDK 5.5 and a ZED 2i. Other cameras (ZED X, ZED Mini) and other SDK versions are likely to
work and are untested; issues with a sample file are welcome.

## Known limits and open questions

* **"SDK look" is approximate.** The SDK sharpens images on playback; the exact filter (and whether playback honours the
  camera's SHARPNESS / GAMMA settings) is not yet pinned down. The stored image (`look="camera"`) is exact.
* **Depth, tracking, object detection** are not reproduced; they need the SDK's closed models.
* **Factory-calibration rectification** is ~1.8 px off the SDK's; export the SDK calibration once per camera and resolution
  (`tools/zed_export_calib.py`) for an exact match.
* **Magnetometer:** values are reproduced exactly, but a camera whose magnetometer has not been calibrated (ZED Sensor
  Viewer) reports a field of only ~2 µT with calibrated == raw, and the SDK's heading stays 0. That is the camera's data,
  not a reader error; Earth's field is ~25-65 µT.
* **Dropped frames** are the recorder's, not the reader's: gaps show up in the timeline, in `_frames.csv`, and as holds in
  an exported MP4.

## Speed

Playback decodes in order on all cores (FFmpeg frame threading) and streams to the page, paced by the recorded
timestamps: on an Apple-silicon Mac, 7.4 ms per frame at HD720 and 11.3 ms at HD2K, so real time with no dropped frames.
Still frames (stepping, scrubbing, PNG export) decode at full quality; a jump costs up to one second of frames.

## Development

```
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest            # synthetic SVO2 files (H.265 lossless via libx265), no camera or SDK needed
```

Parity against the SDK (needs a machine with the ZED SDK for the first two steps):

```
python tools/zed_truth.py rec.svo2 truth.npz             # with pyzed
python tools/zed_export_calib.py rec.svo2                 # with pyzed -> rec.calib.json
python tools/parity_vs_sdk.py rec.svo2 truth.npz          # anywhere
```

On macOS, importing PyAV and OpenCV together prints an `objc ... AVFFrameReceiver is implemented in both` warning: both
wheels bundle FFmpeg's libavdevice. It is harmless here (no capture devices are used).

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Not affiliated with or endorsed by Stereolabs; "ZED"
and "SVO" are Stereolabs trademarks. svo2view contains no Stereolabs code; the optional scripts in `tools/` call the ZED
SDK's Python API, which Stereolabs licenses separately.
