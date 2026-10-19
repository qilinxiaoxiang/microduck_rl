"""Minimal DYNAMIXEL 2.0 sync read / sync write over pyserial for the Pi Zero W.

dynamixel_sdk parses each status packet in pure Python and needs about 44 ms (half of them failing) for one
14-servo sync read on the Zero W. Here the whole reply (one status packet per ID) is read in one call and parsed
with struct, which keeps a read under a few milliseconds.
"""
import struct
import time

import serial

_CRC = []
for _i in range(256):
    _c = _i << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x8005) if _c & 0x8000 else (_c << 1)
    _CRC.append(_c & 0xFFFF)


def crc16(data):
    c = 0
    for b in data:
        c = ((c << 8) ^ _CRC[((c >> 8) ^ b) & 0xFF]) & 0xFFFF
    return c


def _packet(dxl_id, inst, params):
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, dxl_id]) + struct.pack("<H", len(params) + 3) + bytes([inst]) + params
    return body + struct.pack("<H", crc16(body))


class FastBus:
    def __init__(self, port="/dev/serial0", baud=1_000_000):
        self.ser = serial.Serial(port, baud, timeout=0.004)

    def sync_write(self, addr, size, values):
        """values: {id: int}"""
        params = struct.pack("<HH", addr, size)
        for i, v in values.items():
            params += bytes([i]) + int(v).to_bytes(size, "little", signed=size == 4)
        self.ser.write(_packet(0xFE, 0x83, params))

    def sync_read(self, addr, size, ids, timeout=0.012):
        """Return {id: bytes} for the IDs that answered with a valid packet."""
        self.ser.reset_input_buffer()
        self.ser.write(_packet(0xFE, 0x82, struct.pack("<HH", addr, size) + bytes(ids)))
        want = len(ids) * (11 + size)
        buf = b""
        end = time.perf_counter() + timeout
        while len(buf) < want and time.perf_counter() < end:
            buf += self.ser.read(want - len(buf))
        out, k = {}, 0
        while True:
            k = buf.find(b"\xff\xff\xfd\x00", k)
            if k < 0 or k + 11 + size > len(buf):
                break
            dxl_id = buf[k + 4]
            n = struct.unpack_from("<H", buf, k + 5)[0]
            pkt = buf[k:k + 7 + n]
            if buf[k + 7] == 0x55 and n == 4 + size and len(pkt) == 7 + n and \
                    crc16(pkt[:-2]) == struct.unpack_from("<H", pkt, len(pkt) - 2)[0]:
                out[dxl_id] = pkt[9:9 + size]
            k += 7 + max(n, 1)
        return out
