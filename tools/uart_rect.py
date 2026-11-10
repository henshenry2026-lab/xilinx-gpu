"""Encode and send the xilinx-gpu v1 rectangle command (Python 3.10+)."""

import argparse
import struct
import sys


def crc8(data: bytes) -> int:
    """CRC-8/SMBUS: poly=0x07, init=0, refin=false, refout=false, xorout=0."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ (0x07 if crc & 0x80 else 0)) & 0xFF
    return crc


def encode_rect(x: int, y: int, width: int, height: int, color: int) -> bytes:
    """Return one 13-byte packet; reject any rectangle outside 640x480."""
    if not (0 <= x < 640 and 0 <= y < 480 and width > 0 and height > 0
            and x + width <= 640 and y + height <= 480):
        raise ValueError("rectangle must have positive dimensions and fit inside 640x480")
    if not 0 <= color <= 255:
        raise ValueError("RGB332 color must be between 0 and 255")
    payload = struct.pack("<BHHHHB", 1, x, y, width, height, color)
    return b"\xa5\x5a" + payload + bytes([crc8(payload)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="serial device, e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--x", type=int, default=160)
    parser.add_argument("--y", type=int, default=120)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--color", type=lambda value: int(value, 0), default=0x1F,
                        help="RGB332 integer, decimal or 0x-prefixed hex (default: 0x1f)")
    parser.add_argument("--dry-run", action="store_true", help="print hex packet without serial access")
    args = parser.parse_args(argv)
    if args.baud <= 0:
        parser.error("baud must be positive")
    try:
        packet = encode_rect(args.x, args.y, args.width, args.height, args.color)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(packet.hex(" ").upper())
        return 0
    if not args.port:
        parser.error("--port is required unless --dry-run is used")
    try:
        import serial
    except ImportError:
        print("Install serial support: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    try:
        with serial.Serial(args.port, args.baud, timeout=1, write_timeout=1) as device:
            device.write(packet)
            device.flush()
    except (serial.SerialException, OSError) as error:
        print(f"Serial write failed: {error}", file=sys.stderr)
        return 1
    print(f"Sent {len(packet)} bytes to {args.port}: {packet.hex(' ').upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
