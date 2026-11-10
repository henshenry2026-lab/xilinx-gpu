"""Host-side protocol tests, including an independent CRC check vector."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uart_rect", ROOT / "tools" / "uart_rect.py")
uart_rect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uart_rect)


class ProtocolTests(unittest.TestCase):
    def test_standard_crc8_check_vector(self):
        self.assertEqual(uart_rect.crc8(b"123456789"), 0xF4)
        self.assertEqual(uart_rect.crc8(b""), 0)

    def test_little_endian_packet_and_crc(self):
        packet = uart_rect.encode_rect(300, 260, 320, 200, 0x1F)
        self.assertEqual(len(packet), 13)
        self.assertEqual(packet[:12], bytes.fromhex("A5 5A 01 2C 01 04 01 40 01 C8 00 1F"))
        self.assertEqual(packet[-1], uart_rect.crc8(packet[2:-1]))

    def test_screen_edges_and_full_screen(self):
        self.assertEqual(len(uart_rect.encode_rect(639, 479, 1, 1, 255)), 13)
        self.assertEqual(len(uart_rect.encode_rect(0, 0, 640, 480, 0)), 13)

    def test_invalid_rectangles_and_colors(self):
        for values in [
            (-1, 0, 1, 1, 0), (0, -1, 1, 1, 0),
            (0, 0, 0, 1, 0), (0, 0, 1, 0, 0),
            (640, 0, 1, 1, 0), (0, 480, 1, 1, 0),
            (639, 0, 2, 1, 0), (0, 479, 1, 2, 0),
            (65535, 0, 65535, 1, 0), (0, 65535, 1, 65535, 0),
            (0, 0, 1, 1, -1), (0, 0, 1, 1, 256),
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                uart_rect.encode_rect(*values)

    def test_dry_run_without_serial_dependency(self):
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / "tools" / "uart_rect.py"),
             "--x", "300", "--y", "260", "--width", "320", "--height", "200",
             "--color", "0x1f", "--dry-run"],
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(bytes.fromhex(result.stdout.strip()), uart_rect.encode_rect(300, 260, 320, 200, 0x1F))

    def test_cli_rejects_invalid_input_and_missing_port(self):
        for extra in [["--width", "0", "--dry-run"], []]:
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "uart_rect.py"), *extra],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
