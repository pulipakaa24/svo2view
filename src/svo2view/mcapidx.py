"""Minimal MCAP indexer for SVO2 files (uncompressed chunks, as the ZED SDK writes them).

One pass over record headers (no message payloads read) -> per-topic arrays of (log_time, file_offset, size) so any message
can be read with one seek. Handles a truncated file (stops at the last complete record).
"""
import json
import struct

MAGIC = b"\x89MCAP0\r\n"
OP_SCHEMA, OP_CHANNEL, OP_MESSAGE, OP_CHUNK = 0x03, 0x04, 0x05, 0x06


def _str(b, o):
    n = struct.unpack_from("<I", b, o)[0]
    return b[o + 4:o + 4 + n].decode(), o + 4 + n


def _map(b, o):
    n = struct.unpack_from("<I", b, o)[0]
    end, o, m = o + 4 + n, o + 4, {}
    while o < end:
        k, o = _str(b, o)
        v, o = _str(b, o)
        m[k] = v
    return m, o


class McapIndex:
    def __init__(self, path):
        self.path = path
        self.channels = {}  # id -> dict(topic, encoding, metadata)
        self.msgs = {}  # topic -> list of (log_time, offset, size)
        with open(path, "rb") as f:
            if f.read(8) != MAGIC:
                raise ValueError(f"{path}: not an MCAP/SVO2 file")
            f.seek(0, 2)
            self.file_size = f.tell()
            self._scan(f, 8, self.file_size)

    def _channel(self, c):
        cid, _schema = struct.unpack_from("<HH", c, 0)
        topic, o = _str(c, 4)
        enc, o = _str(c, o)
        meta, _ = _map(c, o)
        self.channels[cid] = dict(topic=topic, encoding=enc, metadata=meta)
        self.msgs.setdefault(topic, [])

    def _scan(self, f, pos, end, in_chunk=False):
        while pos + 9 <= end:
            f.seek(pos)
            h = f.read(9)
            if len(h) < 9:
                return
            op, ln = struct.unpack("<BQ", h)
            body = pos + 9
            if body + ln > end:  # truncated record: recover the complete messages of a cut-off chunk, then stop
                if op == OP_CHUNK and not in_chunk:
                    self._chunk(f, body, end, end)
                return
            if op == OP_CHANNEL and not self._known_channel_at(f, body):
                f.seek(body)
                self._channel(f.read(ln))
            elif op == OP_MESSAGE:
                f.seek(body)
                cid, _seq, log_t, _pub = struct.unpack("<HIQQ", f.read(22))
                ch = self.channels.get(cid)
                if ch is not None:
                    self.msgs[ch["topic"]].append((log_t, body + 22, ln - 22))
            elif op == OP_CHUNK and not in_chunk:
                self._chunk(f, body, body + ln, end)
            elif op == 0x02:  # footer
                return
            pos = body + ln

    def _chunk(self, f, body, chunk_end, end):
        if body + 32 > end:
            return
        f.seek(body + 28)  # skip start/end time, uncompressed size, crc
        clen = struct.unpack("<I", f.read(4))[0]
        comp = f.read(clen).decode(errors="replace")
        if comp not in ("", "none"):
            raise NotImplementedError(f"compressed MCAP chunk ({comp}); SVO2 files from the ZED SDK are uncompressed")
        rstart = body + 28 + 4 + clen + 8
        if rstart > end:
            return
        rlen = struct.unpack("<Q", f.read(8))[0]
        self._scan(f, rstart, min(rstart + rlen, chunk_end, end), in_chunk=True)

    def _known_channel_at(self, f, body):
        f.seek(body)
        cid = struct.unpack("<H", f.read(2))[0]
        return cid in self.channels

    def read(self, offset, size):
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(size)

    def topic(self, suffix):
        for t in self.msgs:
            if t == suffix or t.endswith("/" + suffix):
                return t
        return None


if __name__ == "__main__":
    import sys
    ix = McapIndex(sys.argv[1])
    print(json.dumps({t: len(v) for t, v in ix.msgs.items()}))
    print(json.dumps({k: v for k, v in ix.channels.items()})[:600])
