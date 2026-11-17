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
    def test_successful_bar_placement_is_cached_from_repl_result(self):
        h = Hypr()
        monitor = NATIVE[0]
        h._native_mons = [monitor]
        with patch.object(h, 'monitor_at', return_value=monitor), \
             patch('sussurro_hypr.subprocess.run') as run:
            # Model hyprctl's actual CLI: eval discards the returned value.
            run.side_effect = lambda cmd, **kw: SimpleNamespace(
                returncode=0, stdout='true\n' if cmd[1] == 'repl' else 'ok\n')
            self.assertTrue(h.place_bar(100, 100))
            self.assertTrue(h.place_bar(100, 100))
            run.assert_called_once()

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

ROTATED = [
    dict(name='HDMI-A-1', x=0, y=0, width=1920, height=1080, scale=1, transform=0, reserved=[0,30,0,0]),
    dict(name='DP-3', x=1920, y=0, width=3440, height=1440, scale=1, transform=0, reserved=[0,30,0,0]),
    dict(name='DP-2', x=5360, y=0, width=2560, height=1440, scale=1, transform=3, reserved=[0,30,0,0]),
]
XRANDR_ROTATED = '''DP-3 connected primary 3440x1440+1920+0 left (normal)
HDMI-A-1 connected 1920x1080+0+0 (normal)
DP-2 connected 1440x2560+5360+0 (normal)
'''


class RotatedMonitor(unittest.TestCase):
    """Monitor em retrato: o Hyprland manda o modo sem girar (2560x1440, transform 3)."""

    def hypr(self, point):
        h = Hypr()
        h.available = True
        h._query = lambda cmd: ROTATED if cmd == 'monitors' else dict(x=point[0], y=point[1])
        return h

    @patch('sussurro_hypr.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=XRANDR_ROTATED))
    def test_cursor_low_on_portrait_monitor_is_found(self, _):
        h = self.hypr((5400, 2000))
        cursor = h.cursorpos()
        self.assertEqual(cursor, (5400, 2000))
        self.assertEqual(h.work_area_at(*cursor), (5360, 30, 6800, 2560))

    @patch('sussurro_hypr.subprocess.run')
    def test_bar_lands_inside_portrait_monitor(self, run):
        h = self.hypr((5400, 2000))
        run.side_effect = lambda cmd, **kw: SimpleNamespace(
            returncode=0, stdout=XRANDR_ROTATED if cmd[0] == 'xrandr' else 'true\n')
        h.monitors()
        self.assertTrue(h.place_bar(5360 + 644, 2366))
        code = run.call_args.args[0][2]
        self.assertIn('x=6004,y=2366', code)
        self.assertIn('monitor="DP-2"', code)

    def test_native_work_area_swaps_rotated_size(self):
        self.assertEqual(Hypr.native_work_area(ROTATED[2]), (5360, 30, 6800, 2560))
        self.assertEqual(Hypr.native_work_area(ROTATED[1]), (1920, 30, 5360, 1440))

    def test_native_monitor_at_uses_rotated_bounds(self):
        h = self.hypr((0, 0))
        self.assertEqual(h.native_monitor_at(5400, 2000)['name'], 'DP-2')
        self.assertEqual(h.native_monitor_at(9000, 2000)['name'], 'DP-2')  # fora: o mais perto


if __name__ == '__main__':
    unittest.main()
