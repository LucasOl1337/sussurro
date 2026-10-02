"""IPC real em runtime temporario: instancia unica e clientes que falham."""
import errno
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import sussurro_ipc


class IpcRobustnessTests(unittest.TestCase):
    def setUp(self):
        trace = patch.object(app.traceback, 'print_exc')
        trace.start()
        self.addCleanup(trace.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'sussurro.sock'
        self.events = queue.Queue()
        self.server = app.IpcServer(self.events, self.path)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown, self.server, self.thread)
        self.wait_ready()

    def shutdown(self, server, thread):
        server._stop.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())

    def send(self, command):
        with patch.object(sussurro_ipc, 'IPC_SOCK', self.path):
            return sussurro_ipc.ipc_send(command)

    def wait_ready(self):
        deadline = time.monotonic() + 2
        while True:
            try:
                self.assertEqual(self.send('status'), 'ok\n')
                return
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    self.fail('socket temporario nao ficou pronto')
                time.sleep(.01)

    def test_second_application_exits_before_tk_audio_or_cuda(self):
        inode = self.path.stat().st_ino
        env = {**os.environ, 'XDG_RUNTIME_DIR': self.directory.name,
               'DISPLAY': '', 'WAYLAND_DISPLAY': ''}
        result = subprocess.run([sys.executable, '-S', app.__file__], env=env,
                                capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ja esta rodando', result.stdout)
        self.assertEqual(self.path.stat().st_ino, inode)
        self.assertEqual(self.events.get_nowait(), ('show', None))
        self.assertEqual(self.send('toggle'), 'ok\n')
        self.assertEqual(self.events.get_nowait(), ('toggle', None))

    def test_second_server_cannot_take_over_live_socket(self):
        inode = self.path.stat().st_ino
        other_events = queue.Queue()
        other = app.IpcServer(other_events, self.path)
        thread = threading.Thread(target=other.run, daemon=True)
        thread.start()
        self.addCleanup(self.shutdown, other, thread)
        thread.join(.3)
        self.assertFalse(thread.is_alive(), 'segundo servidor roubou o socket')
        self.assertEqual(self.path.stat().st_ino, inode)
        self.assertEqual(self.send('cancel'), 'ok\n')
        self.assertEqual(self.events.get_nowait(), ('cancel', None))
        self.assertTrue(other_events.empty())

    def test_client_that_sends_nothing_does_not_kill_hotkey_server(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(self.path))
            time.sleep(1.15)  # timeout de recv do servidor, com folga
        self.assertTrue(self.thread.is_alive(), 'timeout do cliente matou o IPC')
        self.assertTrue(self.path.exists())
        self.assertEqual(self.send('toggle'), 'ok\n')
        self.assertEqual(self.events.get_nowait(), ('toggle', None))

    def test_client_closing_before_reply_does_not_kill_server(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(self.path))
            client.sendall(b'cancel\n')
            client.shutdown(socket.SHUT_RDWR)
        time.sleep(.1)
        self.assertEqual(self.send('toggle'), 'ok\n')
        self.assertTrue(self.thread.is_alive())

    def test_handler_failure_and_unknown_command_leave_next_toggle_working(self):
        original = self.server._handle
        with patch.object(self.server, '_handle', side_effect=[RuntimeError('pedido falhou')]):
            self.assertEqual(self.send('status'), '')
        self.assertEqual(self.send('unknown'), 'err unknown\n')
        self.assertEqual(self.send('toggle'), 'ok\n')
        self.assertEqual(self.events.get_nowait(), ('toggle', None))
        self.assertEqual(original('status'), 'ok\n')

    def test_stale_socket_can_be_replaced_and_clean_shutdown_removes_own_socket(self):
        self.shutdown(self.server, self.thread)
        self.assertFalse(self.path.exists())
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
            stale.bind(str(self.path))
        replacement = app.IpcServer(self.events, self.path)
        thread = threading.Thread(target=replacement.run, daemon=True)
        thread.start()
        self.addCleanup(self.shutdown, replacement, thread)
        self.wait_ready()
        self.assertEqual(self.send('toggle'), 'ok\n')
        self.assertEqual(self.events.get_nowait(), ('toggle', None))
        self.shutdown(replacement, thread)
        self.assertFalse(self.path.exists())

    def test_transient_accept_error_recovers_for_next_cli_request(self):
        sock = self.server._sock
        accept = sock.accept
        calls = 0
        class Proxy:
            def accept(self):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError(errno.EINTR, 'interrompido')
                return accept()
            def close(self):
                sock.close()
        self.server._sock = Proxy()
        # O accept que ja estava bloqueado precisa acordar para usar o proxy.
        self.send('status')
        time.sleep(.1)
        self.assertEqual(self.send('toggle'), 'ok\n')
        self.assertGreaterEqual(calls, 2)
        self.assertTrue(self.thread.is_alive())

    def test_fatal_accept_error_preserves_socket_for_diagnosis(self):
        sock = self.server._sock
        class Proxy:
            def accept(self):
                raise OSError(errno.EBADF, 'socket fechado')
            def close(self):
                sock.close()
        self.server._sock = Proxy()
        self.send('status')
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())
        self.assertTrue(self.path.exists(), 'erro fatal deslinkou o socket')
        event, message = self.events.get_nowait()
        self.assertEqual(event, 'error')
        self.assertIn('atalho IPC indisponivel', message)


class InstanceProbeTests(unittest.TestCase):
    def test_absent_or_stale_socket_allows_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sussurro.sock'
            with patch.object(sussurro_ipc, 'IPC_SOCK', path):
                self.assertFalse(sussurro_ipc.already_running())
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as old:
                    old.bind(str(path))
                self.assertFalse(sussurro_ipc.already_running())

    def test_unresponsive_socket_refuses_startup_without_stealing(self):
        with patch.object(sussurro_ipc, 'ipc_send', side_effect=socket.timeout):
            with self.assertRaises(socket.timeout):
                sussurro_ipc.already_running()


if __name__ == '__main__':
    unittest.main()
