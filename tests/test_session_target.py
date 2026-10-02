"""Alvo do ditado e capturado no start, sem teclado/clipboard/compositor reais."""
import queue
import sys
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from sussurro_hypr import Hypr


class SessionTargetTests(unittest.TestCase):
    @contextmanager
    def desktop(self, target_class='foot'):
        target = {'address': 'initial', 'class': target_class, 'mapped': True,
                  'at': [0, 0], 'size': [400, 400], 'workspace': {'id': 2}}
        other = {'address': 'other', 'class': 'chromium', 'mapped': True,
                 'at': [400, 0], 'size': [400, 400], 'workspace': {'id': 2}}
        state = {'cursor': (100, 100), 'active': target, 'clients': [target, other]}
        h = Hypr()
        h.available = True
        monitors = [{'x': 0, 'y': 0, 'width': 800, 'height': 600,
                     'activeWorkspace': {'id': 2}}]
        def query(cmd):
            if cmd == 'clients':
                return state['clients']
            if cmd == 'monitors':
                return monitors
            if cmd == 'activewindow':
                return state['active']
            if cmd == 'cursorpos':
                return dict(zip(('x', 'y'), state['cursor']))
            raise AssertionError(cmd)
        def focus(win):
            state['active'] = win
            return True
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            t = app.Transcriber(queue.Queue(), queue.Queue())
        deliveries = []
        with patch.object(app, '_hypr', return_value=h), \
             patch.object(h, '_query', side_effect=query), \
             patch.object(h, 'focus_window', side_effect=focus), \
             patch.object(app, 'IS_WIN', False), \
             patch.object(app.time, 'sleep'), \
             patch.object(app, '_perf'), \
             patch.object(app, 'click_at_cursor') as click, \
             patch.object(t, '_open_stream'), \
             patch.object(app, 'set_clipboard_text', return_value=True) as clipboard, \
             patch.object(t, '_send_paste_key', side_effect=lambda strategy:
                          deliveries.append((state['active']['address'], strategy)) or True):
            yield t, state, target, other, deliveries, click, clipboard

    def test_start_keeps_initial_window_after_cursor_and_focus_move(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            t.start(None, True)
            t._paste('primeiro')
            state['cursor'] = (500, 100)
            state['active'] = other
            t._paste('segundo')
            self.assertEqual(delivered, [('initial', 'terminal'), ('initial', 'terminal')])
            click.assert_not_called()

    def test_own_sussurro_window_never_receives_paste(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            target['class'] = 'Sussurro'
            state['clients'] = [target]
            with self.assertRaises(RuntimeError):
                t._paste('nao colar no Sussurro')
            self.assertEqual(delivered, [])
            clipboard.assert_not_called()
            click.assert_not_called()

    def test_missing_initial_window_falls_back_to_current_pointer(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            t.start(None, True)
            state['clients'].remove(target)
            state['cursor'] = (500, 100)
            state['active'] = other
            t._paste('janela inicial fechou')
            self.assertEqual(delivered, [('other', 'ctrl_v')])
            click.assert_called_once()

    def test_web_target_is_refocused_without_clicking_new_pointer_window(self):
        with self.desktop('chromium') as (t, state, target, other, delivered, click, clipboard):
            t.start(None, True)
            t._paste('primeiro')
            click.assert_called_once()
            click.reset_mock()
            state['cursor'] = (500, 100)
            state['active'] = other
            t._paste('segundo')
            self.assertEqual(delivered, [('initial', 'ctrl_v'), ('initial', 'ctrl_v')])
            click.assert_not_called()

    def test_start_only_reads_target_and_next_session_captures_new_address(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            t.start(None, True)
            self.assertEqual(t._session_target, {'address': 'initial'})
            click.assert_not_called()
            clipboard.assert_not_called()
            self.assertEqual(delivered, [])
            t.cancel()
            state['cursor'] = (500, 100)
            state['active'] = other
            t.start(None, True)
            t._paste('nova sessao')
            self.assertEqual(t._session_target, {'address': 'other'})
            self.assertEqual(delivered, [('other', 'ctrl_v')])

    def test_without_capture_uses_current_pointer(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            state['cursor'] = (900, 100)
            t.start(None, True)
            self.assertIsNone(t._session_target)
            state['cursor'] = (500, 100)
            t._paste('fallback sem alvo inicial')
            self.assertEqual(delivered, [('other', 'ctrl_v')])

    def test_non_injected_session_never_queries_compositor(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            with patch.object(app, '_hypr') as hypr:
                t.start(None, False)
                hypr.assert_not_called()
            self.assertIsNone(t._session_target)

    def test_failed_focus_never_sends_to_pointer_window(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            t.start(None, True)
            state['cursor'] = (500, 100)
            state['active'] = other
            with patch.object(app._hypr(), 'focus_window', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'focar o destino'):
                    t._paste('preservado no historico')
            self.assertEqual(delivered, [])
            clipboard.assert_not_called()
            click.assert_not_called()

    def test_own_window_is_not_target_but_is_detected_as_click_obstacle(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            target['class'] = 'Sussurro'
            self.assertIsNone(app._hypr().window_at(100, 100))
            self.assertEqual(app._hypr().window_at(100, 100, include_bar=True), target)

    def test_without_hypr_rejects_own_focused_window_but_preserves_foot(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            app._hypr().available = False
            with patch.object(app, 'focused_window_class', return_value='Sussurro'):
                with self.assertRaisesRegex(RuntimeError, 'proprio Sussurro'):
                    t._paste('nao colar aqui')
            clipboard.assert_not_called()
            with patch.object(app, 'focused_window_class', return_value='foot'):
                t._paste('terminal sem Hypr')
            self.assertEqual(delivered, [('initial', 'terminal')])

    def test_saved_target_focus_must_be_confirmed_before_clipboard(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            t.start(None, True)
            state['active'] = other
            with patch.object(app._hypr(), 'focus_window', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'foco mudou'):
                    t._paste('nao enviar sem confirmar')
            self.assertEqual(delivered, [])
            clipboard.assert_not_called()

    def test_unavailable_clients_is_not_treated_as_closed_window(self):
        with self.desktop() as (t, state, target, other, delivered, click, clipboard):
            t.start(None, True)
            state['clients'] = None
            with self.assertRaisesRegex(RuntimeError, 'verificar o destino'):
                t._paste('destino nao verificavel')
            self.assertEqual(delivered, [])
            clipboard.assert_not_called()

    def test_own_floating_window_is_not_clicked_over_saved_web_target(self):
        with self.desktop('chromium') as (t, state, target, other, delivered, click, clipboard):
            t.start(None, True)
            own = {**target, 'class': 'Sussurro', 'address': 'own',
                   'floating': True, 'size': [200, 200]}
            state['clients'].append(own)
            state['active'] = own
            t._paste('cola na janela salva, nao na propria')
            self.assertEqual(delivered, [('initial', 'ctrl_v')])
            click.assert_not_called()


if __name__ == '__main__':
    unittest.main()
