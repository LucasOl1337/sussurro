"""Modo colar deve enviar clipboard, inclusive no Codex, sem digitar texto."""
import queue
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class PasteModeTests(unittest.TestCase):
    def test_desktop_paste_uses_clipboard_and_ctrl_v(self):
        for window in ('ChatGPT', 'Codex', 'chatgpt-desktop'):
            with self.subTest(window=window), \
                 patch.object(threading.Thread, 'start'), \
                 patch.object(app.keyboard, 'Controller'):
                transcriber = app.Transcriber(queue.Queue(), queue.Queue())
                with patch.object(app, 'focused_window_class', return_value=window), \
                     patch.object(app, 'backup_clipboard', return_value=b'original'), \
                     patch.object(app, 'set_clipboard_text', return_value=True) as write, \
                     patch.object(transcriber, '_send_paste_key', return_value=True) as send, \
                     patch.object(transcriber, '_type_fallback') as type_text:
                    transcriber._paste('texto inteiro de uma vez')
                    type_text.assert_not_called()
                    write.assert_called_once_with('texto inteiro de uma vez')
                    send.assert_called_once_with('ctrl_v')


if __name__ == '__main__':
    unittest.main()
