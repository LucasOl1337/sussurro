import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sussurro_hypr import Hypr

NATIVE = [
    dict(name='HDMI-A-1', x=0, y=0, width=1920, height=1080, scale=1, reserved=[0,26,0,0]),
    dict(name='DP-1', x=1920, y=0, width=2560, height=1440, scale=1, reserved=[0,26,0,0]),
    dict(name='DP-2', x=4480, y=0, width=2560, height=1440, scale=1, reserved=[0,26,0,0]),
]
XRANDR = '''HDMI-A-1 connected 1920x1080+0+0 (normal)
DP-2 connected 2560x1440+1920+0 (normal)
DP-1 connected primary 2560x1440+4480+0 (normal)
'''

class MonitorCoordinates(unittest.TestCase):
    def hypr(self, point):
        h = Hypr()
        h.available = True
        h._query = lambda cmd: NATIVE if cmd == 'monitors' else dict(x=point[0], y=point[1])
        return h

    @patch('sussurro_hypr.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=XRANDR))
    def test_bar_follows_center_monitor_in_xwayland_coordinates(self, _):
        h = self.hypr((3200, 720))
        cursor = h.cursorpos()
        self.assertEqual(cursor, (5760, 720))
        self.assertEqual(h.work_area_at(*cursor), (4480, 26, 7040, 1440))

    @patch('sussurro_hypr.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=XRANDR))
    def test_bar_follows_right_monitor_after_display_reordering(self, _):
        h = self.hypr((5760, 720))
        cursor = h.cursorpos()
        self.assertEqual(cursor, (3200, 720))
        self.assertEqual(h.work_area_at(*cursor), (1920, 26, 4480, 1440))

    @patch('sussurro_hypr.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=XRANDR))
    def test_left_monitor_stays_unchanged(self, _):
        h = self.hypr((960, 540))
        self.assertEqual(h.cursorpos(), (960, 540))
        self.assertEqual(h.work_area_at(*h.cursorpos()), (0, 26, 1920, 1080))

if __name__ == '__main__':
    unittest.main()
