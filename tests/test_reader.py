import json
import os
import shutil
import threading
import urllib.request

import numpy as np
import pytest

from svo2view import SVO2, calib
from synth import SERIAL, write


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    d = tmp_path_factory.mktemp("rec")
    p = str(d / "synthetic.svo2")
    truth = write(p)
    return p, truth


def test_index_and_timestamps(rec):
    p, t = rec
    s = SVO2(p)
    assert s.n == len(t["frames"])
    assert s.t_ns.tolist() == t["times"]
    assert (s.width, s.height, s.fps, s.serial) == (t["w"], t["h"], t["fps"], SERIAL)


def test_decode_is_exact_sequential_and_random(rec):
    p, t = rec
    s = SVO2(p)
    for i, (Y, U, V) in enumerate(t["frames"]):
        y, u, v = s.yuv(i)
        assert np.array_equal(y, Y) and np.array_equal(u, U) and np.array_equal(v, V), i
    s2 = SVO2(p)
    for i in (9, 2, 7, 0, 11, 5, 6):          # jumps across GOPs and backwards
        assert np.array_equal(s2.yuv(i)[0], t["frames"][i][0]), i
    assert s.is_keyframe(0) and s.is_keyframe(t["fps"]) and not s.is_keyframe(1)


def test_stream_matches_random_access(rec):
    p, t = rec
    s = SVO2(p)
    got = [(i, Y) for i, Y, U, V in s.stream(3)]
    assert [i for i, _ in got] == list(range(3, s.n))
    assert all(np.array_equal(Y, t["frames"][i][0]) for i, Y in got)
    g = s.stream(0)                 # stopping early must not hang (frame-threaded decoder drained)
    next(g)
    g.close()
    for view in ("sbs", "left", "right", "anaglyph"):
        im = s.display(*s.yuv(4), view=view, width=60)
        assert im.ndim == 3 and im.shape[1] in (60, 30)


def test_over_under_layout_is_rearranged(tmp_path):
    p = str(tmp_path / "stacked.svo2")
    t = write(p, stacked=True)
    s = SVO2(p)
    for i in (0, 6, 3):
        Y, U, V = s.yuv(i)
        assert np.array_equal(Y, t["frames"][i][0]) and np.array_equal(U, t["frames"][i][1]), i
    assert [i for i, *_ in s.stream(0)] == list(range(s.n))


def test_image_shapes_and_looks(rec):
    p, t = rec
    s = SVO2(p)
    a, b = s.image(3), s.image(3, look="sdk")
    assert a.shape == b.shape == (t["h"], 2 * t["w"], 3) and a.dtype == np.uint8
    assert not np.array_equal(a, b)


def test_sensors_exact(rec):
    p, t = rec
    S = SVO2(p).sensors()
    assert len(S["t"]) == len(t["sensors"])
    for k, (ts, vals) in enumerate(t["sensors"]):
        assert S["t"][k] == ts
        for name, v in vals.items():
            got = S[name][k]
            assert np.array_equal(np.atleast_1d(got), np.atleast_1d(v)), (k, name)


def test_truncated_file_opens(rec, tmp_path):
    p, t = rec
    q = str(tmp_path / "cut.svo2")
    with open(p, "rb") as f:
        data = f.read()
    with open(q, "wb") as f:
        f.write(data[: int(len(data) * 0.6)])
    s = SVO2(q)
    assert 0 < s.n < len(t["frames"])
    assert np.array_equal(s.yuv(s.n - 1)[0], t["frames"][s.n - 1][0])


CONF = """[LEFT_CAM_HD]
fx=530
fy=530
cx=640
cy=360
k1=-0.05
k2=0.02
p1=0
p2=0
k3=0
[RIGHT_CAM_HD]
fx=531
fy=531
cx=650
cy=355
k1=-0.05
k2=0.02
p1=0
p2=0
k3=0
[STEREO]
Baseline=120
TY=0.2
TZ=0.5
CV_HD=0.002
RX_HD=0.003
RZ_HD=0.001
"""


def test_conf_calibration_and_rectification(tmp_path):
    c = tmp_path / "SN1.conf"
    c.write_text(CONF)
    cal = calib.from_conf(str(c), 1280, 720)
    assert np.allclose(cal["raw"]["stereo_T_m"], [0.12, 0.0002, 0.0005])
    RL, RR = calib.rect_rotations(cal)
    T = np.array(cal["raw"]["stereo_T_m"])
    R = np.array(cal["raw"]["stereo_R"])
    # both cameras end up with one rectified orientation: R_right @ R_left.T is the stereo rotation
    assert np.allclose(RR @ RL.T, R, atol=1e-9)
    (lx, ly), (rx, ry) = calib.rect_maps(cal, 1280, 720)
    assert lx.shape == (720, 1280) and np.isfinite(lx).all() and np.isfinite(ry).all()


def test_calibration_json_next_to_recording(rec, tmp_path):
    p, t = rec
    q = tmp_path / "r.svo2"
    shutil.copy(p, q)
    cal = {"resolution": [t["w"], t["h"]], "raw": {"left_K": [[50, 0, 32], [0, 50, 24], [0, 0, 1]],
           "right_K": [[50, 0, 32], [0, 50, 24], [0, 0, 1]], "left_disto": [0] * 8, "right_disto": [0] * 8,
           "stereo_R": np.eye(3).tolist(), "stereo_T_m": [0.12, 0, 0]}, "rect": {"K": [[50, 0, 32], [0, 50, 24], [0, 0, 1]]}}
    (tmp_path / "r.calib.json").write_text(json.dumps(cal))
    s = SVO2(str(q))
    assert "exact" in s.calib_source
    im, r = s.image(2), s.image(2, rect=True)
    assert np.abs(im[2:-2, 2:-2].astype(int) - r[2:-2, 2:-2]).max() <= 1   # identity calibration -> same image


def test_server(rec):
    from http.server import ThreadingHTTPServer
    from svo2view import server
    p, t = rec
    server.ARGS = type("A", (), {"calib": None})()
    server.RECS = server.scan([os.path.dirname(p)])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        L = json.load(urllib.request.urlopen(base + "/api/list"))
        rid = L["recordings"][0]["id"]
        info = json.load(urllib.request.urlopen(f"{base}/api/info?id={rid}"))
        assert info["n"] == len(t["frames"]) and info["sensors"]["n"] == len(t["sensors"])
        for view in ("sbs", "left", "right", "anaglyph"):
            r = urllib.request.urlopen(f"{base}/api/frame?id={rid}&i=4&view={view}")
            assert r.headers["Content-Type"] == "image/jpeg" and len(r.read()) > 100
        assert urllib.request.urlopen(base + "/").read().startswith(b"<!doctype html>")
        body = urllib.request.urlopen(f"{base}/api/play.mjpg?id={rid}&start=2&speed=0&sid=t1").read()
        assert body.count(b"--svoframe") == len(t["frames"]) - 2
        pos = json.load(urllib.request.urlopen(f"{base}/api/playpos?sid=t1"))
        assert pos["done"] and pos["i"] == len(t["frames"]) - 1
    finally:
        srv.shutdown()


def test_export_mp4_and_csvs(rec, tmp_path):
    import csv

    import av

    from svo2view.server import main as cli
    p, t = rec
    assert cli(["export", p, "--out", str(tmp_path), "--width", "0"]) == 0
    stem = os.path.splitext(os.path.basename(p))[0]
    rows = list(csv.DictReader(open(tmp_path / f"{stem}_frames.csv")))
    assert [int(r["t_ns"]) for r in rows] == t["times"]
    s = SVO2(p)
    S = s.sensors()
    srows = list(csv.DictReader(open(tmp_path / f"{stem}_sensors.csv")))
    assert len(srows) == len(S["t"]) and int(srows[-1]["t_ns"]) == int(S["t"][-1])
    assert abs(float(srows[0]["acc_x_m_s2"]) - float(S["acc"][0, 0])) < 1e-5
    with av.open(str(tmp_path / f"{stem}.mp4")) as c:
        v = c.streams.video[0]
        frames = list(c.decode(v))
        assert len(frames) == s.n and (v.width, v.height) == (2 * s.width, s.height)
        dur = (frames[-1].pts - frames[0].pts) * float(v.time_base)          # recorded timing, not n / fps
        assert abs(dur - (t["times"][-1] - t["times"][0]) / 1e9) < 0.002
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".part")]
