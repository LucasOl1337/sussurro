"""Gasto da nuvem: livro por mes, custo dos cabecalhos da ElevenLabs e saldo da conta."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import app
import sussurro_cloud as cloud
from sussurro_cloud import ELEVENLABS, XAI, CloudModel
from sussurro_usage import UsageLedger, elevenlabs_balance, spend_line
from tests.test_cloud import ELEVEN_BODY, XAI_BODY, Response, Secrets


class HeaderResponse(Response):
    def __init__(self, body, headers):
        super().__init__(body=body)
        self.headers = headers


class UsageLedgerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / 'cloud-usage.json'
        self.ledger = UsageLedger(self.path)

    def test_elevenlabs_call_records_credits_and_dollars_from_headers(self):
        http = Mock()
        http.post.return_value = HeaderResponse(ELEVEN_BODY, {'character-cost': '45',
                                                              'fiat-cost-before-overages': '0.0045'})
        model = CloudModel(ELEVENLABS, http=http, secrets=Secrets(), usage=self.ledger)
        model.transcribe(np.zeros(cloud.SAMPLE_RATE * 4, dtype=np.float32), language='pt')
        month = self.ledger.month(ELEVENLABS)
        self.assertEqual((month['calls'], month['audio_s'], month['credits']), (1, 4.0, 45))
        self.assertAlmostEqual(month['usd'], 0.0045)
        saved = json.loads(self.path.read_text())['months']
        self.assertEqual(saved[datetime.now().strftime('%Y-%m')][ELEVENLABS]['credits'], 45)

    def test_fallback_charges_the_provider_that_answered(self):
        http = Mock()
        http.post.side_effect = [Response(status=500, body={'detail': 'fora'}), HeaderResponse(XAI_BODY, {})]
        CloudModel(ELEVENLABS, http=http, secrets=Secrets(), usage=self.ledger).transcribe(
            np.zeros(cloud.SAMPLE_RATE * 2, dtype=np.float32))
        self.assertEqual(self.ledger.month(ELEVENLABS)['calls'], 0)
        self.assertEqual(self.ledger.month(XAI)['calls'], 1)
        self.assertEqual(self.ledger.month(XAI)['credits'], 0)

    def test_fake_http_without_ledger_never_touches_the_real_file(self):
        http = Mock()
        http.post.return_value = Response(body=ELEVEN_BODY)
        with patch('sussurro_usage.shared_ledger') as shared:
            CloudModel(ELEVENLABS, http=http, secrets=Secrets()).transcribe(
                np.zeros(cloud.SAMPLE_RATE * 2, dtype=np.float32))
        shared.assert_not_called()

    def test_months_split_and_total_sums_them(self):
        self.ledger.record(XAI, 60, now=datetime(2026, 9, 30))
        self.ledger.record(XAI, 30, now=datetime(2026, 10, 1))
        self.assertEqual(self.ledger.month(XAI, now=datetime(2026, 10, 5))['audio_s'], 30)
        self.assertEqual(self.ledger.total(XAI)['audio_s'], 90)
        self.assertEqual(UsageLedger(self.path).total(XAI)['calls'], 2)  # sobrevive a reabrir

    def test_spend_line_shows_balance_or_why_it_is_missing(self):
        self.ledger.record(ELEVENLABS, 10, {'character-cost': '100', 'fiat-cost-before-overages': '0.01'})
        self.ledger.balance_error = 'saldo: falta a permissão user_read na chave'
        self.assertIn('user_read', spend_line(self.ledger, ELEVENLABS))
        self.ledger.balance, self.ledger.balance_error = {'used': 400, 'limit': 10000}, None
        self.ledger.record(ELEVENLABS, 10, {'character-cost': '100'})
        line = spend_line(self.ledger, ELEVENLABS)
        self.assertIn('200 créditos', line)
        self.assertIn('restam 9.500 de 10.000', line)  # gasto novo desconta antes da proxima busca
        self.assertIn('não informa custo', spend_line(self.ledger, XAI))


class BalanceTests(unittest.TestCase):
    def test_subscription_fields(self):
        http = Mock()
        http.get.return_value = Response(body={'character_count': 1200, 'character_limit': 100000,
                                               'next_character_count_reset_unix': 1794000000, 'tier': 'creator'})
        self.assertEqual(elevenlabs_balance(Secrets(), http),
                         {'used': 1200, 'limit': 100000, 'reset': 1794000000, 'tier': 'creator'})
        self.assertEqual(http.get.call_args.kwargs['headers'], {'xi-api-key': 'el-key'})

    def test_key_without_user_read_says_so(self):
        http = Mock()
        http.get.return_value = Response(status=401, body={'detail': {'status': 'missing_permissions'}})
        with self.assertRaisesRegex(RuntimeError, 'user_read'):
            elevenlabs_balance(Secrets(), http)


class RetryUnlocksPickerTests(unittest.TestCase):
    def test_refazer_marks_the_real_entry_so_the_picker_unlocks(self):
        """A lista desenha copias; marcar a copia deixava MODELO e EXECUTAR EM cinzas pra sempre."""
        real = {'wav': 'a.wav', 'text': 'oi', 'ts': '2026-10-17T10:00:00'}
        host = SimpleNamespace(entries=[real], transcriber=Mock(), record_btn=Mock(), status=Mock(),
                               _set_model_picker=Mock(), _render_history=Mock(), _model_load_done=Mock(),
                               _stats_dirty=False, _tab=None)
        with patch.object(app.Path, 'is_file', return_value=True):
            app.App._retry_entry(host, {**real, 'audio_removed': False})
        host._set_model_picker.assert_called_once_with('disabled')
        self.assertTrue(real['retrying'])
        app.App._add_history(host, {**real, 'text': 'oi refeito'})
        host._model_load_done.assert_called_once_with({}, False, None)


if __name__ == '__main__':
    unittest.main()
