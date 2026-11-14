# Changelog

## 0.1.0 (2026-11-13)

First release. Everything below was measured against the ZED SDK 5.5.0's own output for real ZED 2i recordings.

* **Reader** (`svo2view.SVO2`): MCAP indexer that reads only record headers and recovers every complete message of a
  truncated file; exact H.265 / H.264 decode with keyframe-aware random access; per-frame image timestamps identical to
  the SDK's; every sensor sample from the 360-byte sensor struct, identical to the SDK's float32 values; HD2K's over-under
  storage (2208 x 2484) returned as side by side like every other resolution.
* **Rectification** that reproduces the SDK's: half the stereo rotation per camera, then the baseline onto x, with the SDK's
  8-coefficient rational distortion (0.06 px median from the SDK's geometry). Calibration lookup: SDK export, then the
  factory `SN<serial>.conf` (approximate).
* **Viewer** (`svo2view PATH`): local web page; side by side / left / right / anaglyph, camera or SDK look, raw or
  rectified; real-time playback streamed from a frame-threaded decoder (7.4 ms / frame HD720, 11.3 ms HD2K on an M-series
  Mac); stepping, scrubbing, dropped-frame markers, full-resolution PNG export, sensor plots with a seek cursor.
* **Export** (`svo2view export`): H.264 MP4 at the recorded timing, per-frame CSV, per-sample sensor CSV.
* **Tools** for a machine with the ZED SDK: `zed_truth.py` (dump the SDK's frames, timestamps and sensors),
  `zed_export_calib.py` (dump the calibration the SDK applies), `parity_vs_sdk.py` (compare).
* **Tests** on synthetic SVO2 files (lossless H.265, MCAP), no camera or SDK needed; CI on Linux, macOS, Windows.
