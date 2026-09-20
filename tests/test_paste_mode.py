"""Modo colar deve enviar clipboard, inclusive no Codex, sem digitar texto."""
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from sussurro_hypr import Hypr


class PasteModeTests(unittest.TestCase):
    def setUp(self):
        perf = patch.object(app, '_perf')
        perf.start()
        self.addCleanup(perf.stop)

    def test_terminal_fallback_never_drops_shift(self):
        with patch.object(threading.Thread, 'start'), \
             patch.object(app.keyboard, 'Controller'):
            transcriber = app.Transcriber(queue.Queue(), queue.Queue())
        with patch.object(app, 'prepare_paste_target', return_value='foot'), \
             patch.object(app, '_is_wayland', return_value=False), \
             patch.object(app, 'backup_clipboard', return_value=b'original'), \
             patch.object(app, 'set_clipboard_text', return_value=True), \
             patch.object(transcriber, '_send_paste_key', return_value=False), \
             patch.object(threading.Timer, 'start'), \
             patch.object(app, '_perf'):
            transcriber._paste_linux('ditado')
        transcriber._keyboard.pressed.assert_called_once_with(
            app.keyboard.Key.ctrl, app.keyboard.Key.shift)

    def test_wayland_failure_does_not_send_unreliable_x11_ctrl_v(self):
        with patch.object(threading.Thread, 'start'), \
             patch.object(app.keyboard, 'Controller'):
            transcriber = app.Transcriber(queue.Queue(), queue.Queue())
        with patch.object(app, 'prepare_paste_target', return_value='foot'), \
             patch.object(app, '_is_wayland', return_value=True), \
             patch.object(app, 'backup_clipboard', return_value=b'original'), \
             patch.object(app, 'set_clipboard_text', return_value=True), \
             patch.object(transcriber, '_send_paste_key', return_value=False):
            with self.assertRaises(RuntimeError):
                transcriber._paste_linux('ditado')
        transcriber._keyboard.pressed.assert_not_called()

    def test_desktop_paste_uses_clipboard_and_ctrl_v(self):
        for window in ('ChatGPT', 'Codex', 'chatgpt-desktop'):
            with self.subTest(window=window), \
                 patch.object(threading.Thread, 'start'), \
                 patch.object(app.keyboard, 'Controller'):
                transcriber = app.Transcriber(queue.Queue(), queue.Queue())
                with patch.object(app, 'prepare_paste_target', return_value=window), \
                     patch.object(app, 'backup_clipboard', return_value=b'original'), \
                     patch.object(app, 'set_clipboard_text', return_value=True) as write, \
                     patch.object(transcriber, '_send_paste_key', return_value=True) as send, \
                     patch.object(transcriber, '_type_fallback') as type_text:
                    transcriber._paste('texto inteiro de uma vez')
                    type_text.assert_not_called()
                    write.assert_called_once_with('texto inteiro de uma vez')
                    send.assert_called_once_with('ctrl_v')

    def test_linux_paste_prepares_target_under_cursor(self):
        with patch.object(threading.Thread, 'start'), \
             patch.object(app.keyboard, 'Controller'):
            transcriber = app.Transcriber(queue.Queue(), queue.Queue())
            with patch.object(app, 'prepare_paste_target', return_value='chromium') as prepare, \
                 patch.object(app, 'backup_clipboard', return_value=b'original'), \
                 patch.object(app, 'set_clipboard_text', return_value=True), \
                 patch.object(transcriber, '_send_paste_key', return_value=True) as send, \
                 patch.object(transcriber, '_type_fallback') as type_text:
                transcriber._paste_linux('frase')
                prepare.assert_called_once()
                send.assert_called_once_with('ctrl_v')
                type_text.assert_not_called()

    def test_prepare_clicks_web_fields_but_not_terminals(self):
        win = {'address': '0x1', 'class': 'chromium'}
        hypr = Hypr()
        hypr.available = True
        hypr.focus_at_cursor = lambda: win
        with patch.object(app, '_hypr', return_value=hypr), \
             patch.object(app, 'click_at_cursor', return_value=True) as click, \
             patch.object(app.time, 'sleep'):
            self.assertEqual(app.prepare_paste_target(), 'chromium')
            click.assert_called_once()

        term = {'address': '0x2', 'class': 'foot'}
        hypr.focus_at_cursor = lambda: term
        with patch.object(app, '_hypr', return_value=hypr), \
             patch.object(app, 'click_at_cursor', return_value=True) as click, \
             patch.object(app.time, 'sleep'):
            self.assertEqual(app.prepare_paste_target(), 'foot')
            click.assert_not_called()


class WindowAtCursorTests(unittest.TestCase):
    def test_focus_uses_lua_result_instead_of_legacy_dispatcher(self):
        h = Hypr()
        with patch('sussurro_hypr.subprocess.run', return_value=SimpleNamespace(
                returncode=0, stdout='true\n')) as run:
            self.assertTrue(h.focus_window({'address': '0x123'}))
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[:2], ['hyprctl', 'repl'])
            self.assertIn('hl.dsp.focus', cmd[2])
            self.assertIn('address:0x123', cmd[2])
            run.return_value.stdout = 'false\n'
            self.assertFalse(h.focus_window({'address': '0x123'}))

    def test_window_at_skips_bar_and_prefers_smaller_floating(self):
        clients = [
            {'address': '0xa', 'class': 'foot', 'mapped': True, 'hidden': False,
             'floating': False, 'at': [0, 0], 'size': [1920, 1080], 'focusHistoryID': 1},
            {'address': '0xb', 'class': 'SussurroBar', 'mapped': True, 'hidden': False,
             'floating': True, 'at': [100, 100], 'size': [152, 40], 'focusHistoryID': 0},
            {'address': '0xc', 'class': 'chromium', 'mapped': True, 'hidden': False,
             'floating': True, 'at': [200, 200], 'size': [800, 600], 'focusHistoryID': 2},
        ]
        h = Hypr()
        h.available = True
        monitors = [{'x': 0, 'y': 0, 'width': 1920, 'height': 1080,
                     'scale': 1, 'activeWorkspace': {'id': 2},
                     'specialWorkspace': {'id': 0}}]
        for client in clients:
            client['workspace'] = {'id': 2}
        h._query = lambda cmd: clients if cmd == 'clients' else monitors
        hit = h.window_at(400, 400)
        self.assertEqual(hit['address'], '0xc')
        self.assertEqual(h.window_at(10, 10)['class'], 'foot')

    def test_focused_terminal_beats_invisible_overlapping_workspace(self):
        h = Hypr()
        h.available = True
        common = dict(mapped=True, hidden=False, floating=False,
                      at=[1920, 30], size=[3440, 1410])
        terminal = dict(common, address='terminal', **{'class': 'foot'},
                        workspace={'id': 2}, focusHistoryID=0)
        invisible = dict(common, address='invisible', **{'class': 'maestri-app'},
                         workspace={'id': 4}, focusHistoryID=1)
        monitors = [{'x': 1920, 'y': 0, 'width': 3440, 'height': 1440,
                     'scale': 1, 'activeWorkspace': {'id': 2},
                     'specialWorkspace': {'id': 0}}]
        h._query = lambda cmd: [invisible, terminal] if cmd == 'clients' else monitors
        self.assertEqual(h.window_at(2200, 300)['address'], 'terminal')
        invisible['workspace']['id'] = 2
        self.assertEqual(h.window_at(2200, 300)['address'], 'terminal')

    def test_special_workspace_covers_regular_workspace(self):
        h = Hypr()
        h.available = True
        common = dict(mapped=True, hidden=False, floating=False,
                      at=[0, 30], size=[1000, 700])
        regular = dict(common, address='regular', workspace={'id': 2},
                       focusHistoryID=0, **{'class': 'foot'})
        special = dict(common, address='special', workspace={'id': -98},
                       focusHistoryID=1, **{'class': 'chromium'})
        monitors = [{'x': 0, 'y': 0, 'width': 1000, 'height': 800,
                     'scale': 1, 'activeWorkspace': {'id': 2},
                     'specialWorkspace': {'id': -98}}]
        h._query = lambda cmd: [regular, special] if cmd == 'clients' else monitors
        self.assertEqual(h.window_at(300, 300)['address'], 'special')

    def test_failed_focus_is_not_reported_as_success(self):
        h = Hypr()
        h.available = True
        target = {'address': 'target', 'class': 'foot'}
        with patch.object(h, 'native_cursorpos', return_value=(10, 10)), \
             patch.object(h, 'window_at', return_value=target), \
             patch.object(h, 'activewindow', return_value={'address': 'other'}), \
             patch.object(h, 'focus_window', return_value=False):
            with self.assertRaises(RuntimeError):
                h.focus_at_cursor()

    def test_focus_command_success_requires_actual_target_focus(self):
        h = Hypr()
        h.available = True
        target = {'address': 'target', 'class': 'foot'}
        with patch.object(h, 'native_cursorpos', return_value=(10, 10)), \
             patch.object(h, 'window_at', return_value=target), \
             patch.object(h, 'activewindow', return_value={'address': 'other'}), \
             patch.object(h, 'focus_window', return_value=True), \
             patch('sussurro_hypr.time.sleep'):
            with self.assertRaises(RuntimeError):
                h.focus_at_cursor()


if __name__ == '__main__':
    unittest.main()
