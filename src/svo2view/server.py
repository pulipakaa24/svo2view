"""svo2view: a local web viewer for ZED SVO2 recordings.

    svo2view PATH [PATH ...] [--port 8790] [--calib FILE] [--no-browser]

PATH is an .svo2 file or a folder (searched recursively). The viewer listens on 127.0.0.1 only and opens in the browser.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .reader import SVO2

HERE = os.path.dirname(os.path.abspath(__file__))
RECS: list = []          # [{"id", "name", "path"}]
CACHE: dict = {}
PLAY: dict = {}   # playback stream id -> {i, done, shown, dropped}
MU = threading.Lock()
ARGS = None


def scan(paths):
    out = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        files = [p] if os.path.isfile(p) else sorted(glob.glob(os.path.join(p, "**", "*.svo2"), recursive=True))
        base = os.path.dirname(p) if os.path.isfile(p) else p
        for f in files:
            out.append({"id": str(len(out)), "name": os.path.relpath(f, base), "path": f,
                        "size_bytes": os.path.getsize(f)})
    return out


def get(rec_id):
    rec = next((r for r in RECS if r["id"] == rec_id), None)
    if rec is None:
        raise KeyError("unknown recording")
    with MU:
        if rec_id not in CACHE:
            CACHE[rec_id] = SVO2(rec["path"], calibration=ARGS.calib)
            while len(CACHE) > 3:
                CACHE.pop(next(iter(CACHE)))
        return CACHE[rec_id]


def info(rec_id):
    s = get(rec_id)
    d = s.info()
    t0 = s.t_ns[0] if s.n else 0
    d["frame_ms"] = ((s.t_ns - t0) / 1e6).round(2).tolist()
    dt = np.diff(s.t_ns) / 1e6
    nominal = 1000.0 / s.fps if s.fps else 0
    d["gaps"] = [{"frame": int(i + 1), "ms": round(float(v), 1)} for i, v in enumerate(dt) if nominal and v > 1.5 * nominal][:500]
    S = s.sensors()
    n = len(S["t"])
    step = max(1, n // 4000)
    sl = slice(0, n, step)
    q = S["quat_xyzw"][sl].astype(np.float64)
    x, y, z, w = q.T if n else (np.zeros(0),) * 4
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    r = lambda a, k=4: np.round(np.asarray(a, np.float64), k).tolist()  # noqa: E731
    d["sensors"] = {"n": n, "step": step, "t_s": r((S["t"][sl] - t0) / 1e9), "acc": r(S["acc"][sl]),
                    "gyro": r(S["gyro"][sl]), "mag": r(S["mag"][sl]), "euler": r(np.stack([roll, pitch, yaw], 1), 2),
                    "pressure_hpa": r(S["pressure_pa"][sl] / 100, 3),
                    "temp": r(np.stack([S["temp_imu"][sl], S["temp_left"][sl], S["temp_right"][sl]], 1), 2)}
    return d


def frame(rec_id, i, look, rect, view, width, fmt):
    s = get(rec_id)
    im = s.image(max(0, min(int(i), s.n - 1)), look=look, rect=rect)
    w = im.shape[1] // 2
    if view == "left":
        im = im[:, :w]
    elif view == "right":
        im = im[:, w:]
    elif view == "anaglyph":     # red = left eye, green + blue = right eye
        im = np.dstack([im[:, w:, 0], im[:, w:, 1], im[:, :w, 2]])
    if width and 0 < width < im.shape[1]:
        im = cv2.resize(im, (width, round(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
    if fmt == "png":
        return cv2.imencode(".png", im)[1].tobytes(), "image/png"
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), "image/jpeg"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, b, ctype, code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def _play(self, q):
        """MJPEG playback: decode in order on all cores, pace by the recorded timestamps x speed, drop frames from the
        stream (never from decoding) when behind; the page polls /api/playpos for the frame on screen."""
        sid = q.get("sid", "")
        s = get(q.get("id", ""))
        start = max(0, min(int(q.get("start", 0)), s.n - 1))
        speed = float(q.get("speed", 1) or 0)
        look, rect, view, width = q.get("look", "camera"), q.get("rect") == "1", q.get("view", "sbs"), int(q.get("w", 0) or 0)
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=svoframe")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        PLAY[sid] = {"i": start, "done": False, "shown": 0, "dropped": 0}
        while len(PLAY) > 50:
            PLAY.pop(next(iter(PLAY)))
        g = s.stream(start)
        t0, ts0 = None, int(s.t_ns[start])
        period = 1.0 / s.fps if s.fps else 0.033
        try:
            for i, Y, U, V in g:
                if t0 is None:
                    t0 = time.monotonic()
                if speed > 0:
                    late = time.monotonic() - t0 - (int(s.t_ns[i]) - ts0) / 1e9 / speed
                    if late > period / speed and i < s.n - 1:
                        PLAY[sid]["dropped"] += 1
                        continue
                    if late < 0:
                        time.sleep(-late)
                b = cv2.imencode(".jpg", s.display(Y, U, V, look, rect, view, width), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                self.wfile.write(b"--svoframe\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                self.wfile.flush()
                PLAY[sid].update(i=i, shown=PLAY[sid]["shown"] + 1)
            PLAY[sid]["done"] = True
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            g.close()

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "static", "viewer.html"), "rb") as f:
                    return self._send(f.read(), "text/html; charset=utf-8")
            if u.path == "/api/list":
                return self._json({"recordings": [{k: r[k] for k in ("id", "name", "size_bytes")} for r in RECS]})
            if u.path == "/api/info":
                return self._json(info(q.get("id", "")))
            if u.path == "/api/playpos":
                return self._json(PLAY.get(q.get("sid", ""), {}))
            if u.path == "/api/play.mjpg":
                return self._play(q)
            if u.path == "/api/frame":
                fmt = q.get("fmt", "jpg")
                b, ctype = frame(q.get("id", ""), int(q.get("i", 0)), q.get("look", "camera"), q.get("rect") == "1",
                                 q.get("view", "sbs"), int(q.get("w", 0) or 0), fmt)
                extra = {}
                if fmt == "png":
                    stem = os.path.splitext(os.path.basename(get(q["id"]).path))[0]
                    extra["Content-Disposition"] = f'attachment; filename="{stem}_{int(q.get("i", 0)):06d}.png"'
                return self._send(b, ctype, extra=extra)
            return self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main(argv=None):
    global ARGS, RECS
    ap = argparse.ArgumentParser(prog="svo2view", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".svo2 files or folders")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--calib", help="calibration .json (tools/zed_export_calib.py) or factory SN<serial>.conf")
    ap.add_argument("--no-browser", action="store_true")
    ARGS = ap.parse_args(argv)
    RECS = scan(ARGS.paths)
    if not RECS:
        print("no .svo2 files found", file=sys.stderr)
        return 2
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{ARGS.port}/"
    print(f"svo2view: {len(RECS)} recording(s) at {url}  (Ctrl-C to stop)")
    if not ARGS.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
