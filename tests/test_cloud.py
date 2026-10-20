"""Nuvem (ElevenLabs e xAI) com HTTP falso: formato do pedido, segmentos, fallback e credenciais."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import sussurro_cloud as cloud
from sussurro_cloud import ELEVENLABS, XAI, CloudError, CloudModel, Credentials


class Response:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError('sem json')
        return self._body


class Secrets:
    def __init__(self, have=(ELEVENLABS, XAI), tokens=('tok-a',), login=True):
        self.have = set(have)
        self.tokens = list(tokens)
        self.login = login
        self.margins = []

    def has(self, provider):
        return provider in self.have

    def missing_message(self, provider):
        return f'sem {provider}'

    def elevenlabs_key(self):
        return 'el-key'

    def xai_token(self, margin=120):
        self.margins.append(margin)
        return self.tokens.pop(0) if len(self.tokens) > 1 else self.tokens[0]

    def xai_is_login(self):
        return self.login


ELEVEN_BODY = {'language_code': 'por', 'text': 'Oi Lucas. Tudo bem?', 'words': [
    {'text': 'Oi', 'start': 0.1, 'end': 0.3, 'type': 'word'},
    {'text': ' ', 'start': 0.3, 'end': 0.35, 'type': 'spacing'},
    {'text': 'Lucas.', 'start': 0.35, 'end': 0.8, 'type': 'word'},
    {'text': ' ', 'start': 0.8, 'end': 2.6, 'type': 'spacing'},
    {'text': '(risos)', 'start': 2.6, 'end': 2.7, 'type': 'audio_event'},
    {'text': 'Tudo', 'start': 2.6, 'end': 2.9, 'type': 'word'},
    {'text': ' ', 'start': 2.9, 'end': 2.95, 'type': 'spacing'},
    {'text': 'bem?', 'start': 2.95, 'end': 3.2, 'type': 'word'}]}
XAI_BODY = {'language': 'pt', 'text': 'Oi Lucas.', 'words': [
    {'text': 'Oi', 'start': 0.1, 'end': 0.3}, {'text': 'Lucas.', 'start': 0.35, 'end': 0.8}]}


class CloudModelTests(unittest.TestCase):
    def setUp(self):
        self.http = Mock()
        self.audio = np.zeros(cloud.SAMPLE_RATE * 4, dtype=np.float32)

    def model(self, provider=ELEVENLABS, secrets=None, fallback=True):
        return CloudModel(provider, http=self.http, secrets=secrets or Secrets(), fallback=fallback)

    def test_elevenlabs_request_and_segments_split_on_long_pause(self):
        self.http.post.return_value = Response(body=ELEVEN_BODY)
        segments, info = self.model().transcribe(self.audio, language='pt', hotwords='Sussurro, 9router')
        kwargs = self.http.post.call_args.kwargs
        self.assertEqual(self.http.post.call_args.args[0], cloud.ELEVENLABS_URL)
        self.assertEqual(kwargs['data']['model_id'], 'scribe_v2')
        self.assertEqual(kwargs['data']['tag_audio_events'], 'false')
        self.assertEqual(kwargs['data']['file_format'], 'pcm_s16le_16')
        self.assertEqual(kwargs['data']['language_code'], 'por')
        self.assertEqual(kwargs['data']['keyterms'], ['Sussurro', '9router'])
        self.assertEqual(kwargs['headers'], {'xi-api-key': 'el-key'})
        self.assertEqual(len(kwargs['files']['file'][1]), self.audio.size * 2)  # PCM16 cru
        self.assertEqual([s.text for s in segments], [' Oi Lucas.', ' Tudo bem?'])
        self.assertAlmostEqual(segments[1].start, 2.6)
        self.assertEqual([w.word for w in segments[0].words], ['Oi', 'Lucas.'])
        self.assertEqual(info.language, 'por')

    def test_xai_sends_wav_with_formatting_and_repeated_keyterm(self):
        self.http.post.return_value = Response(body=XAI_BODY)
        segments, info = self.model(XAI).transcribe(self.audio, language='pt', hotwords='Sussurro, Omarchy')
        kwargs = self.http.post.call_args.kwargs
        self.assertEqual(kwargs['data']['format'], 'true')
        self.assertEqual(kwargs['data']['keyterm'], ['Sussurro', 'Omarchy'])  # dict com lista: httpx repete
        self.assertTrue(kwargs['files']['file'][1].startswith(b'RIFF'))
        self.assertEqual(kwargs['headers'], {'Authorization': 'Bearer tok-a'})
        self.assertEqual(segments[0].text, ' Oi Lucas.')
        self.assertEqual(info.language, 'pt')

    def test_tiny_audio_returns_empty_without_network(self):
        segments, info = self.model().transcribe(np.zeros(800, dtype=np.float32), language='pt')
        self.assertEqual(segments, [])
        self.http.post.assert_not_called()

    def test_failure_tries_other_provider(self):
        self.http.post.side_effect = [Response(500, {'detail': 'caiu'}), Response(body=XAI_BODY)]
        model = self.model()
        segments, _ = model.transcribe(self.audio, language='pt')
        self.assertEqual(model.last_provider, XAI)
        self.assertEqual(segments[0].text, ' Oi Lucas.')

    def test_comparison_mode_does_not_fall_back(self):
        self.http.post.return_value = Response(500, {'detail': 'caiu'})
        with self.assertRaisesRegex(CloudError, 'ElevenLabs.*HTTP 500'):
            self.model(fallback=False).transcribe(self.audio, language='pt')
        self.assertEqual(self.http.post.call_count, 1)

    def test_unparseable_answer_becomes_cloud_error(self):
        self.http.post.return_value = Response(200, None, text='<html>')
        with self.assertRaisesRegex(CloudError, 'resposta invalida'):
            self.model(fallback=False).transcribe(self.audio, language='pt')

    def test_network_errors_become_cloud_error(self):
        import httpx
        self.http.post.side_effect = httpx.ConnectTimeout('lento')
        with self.assertRaisesRegex(CloudError, 'sem resposta a tempo'):
            self.model(fallback=False).transcribe(self.audio, language='pt')

    def test_provider_without_credentials_is_skipped(self):
        self.http.post.return_value = Response(body=XAI_BODY)
        model = self.model(secrets=Secrets(have=(XAI,)))
        model.transcribe(self.audio, language='pt')
        self.assertEqual(self.http.post.call_count, 1)
        self.assertEqual(model.last_provider, XAI)

    def test_error_message_masks_secrets(self):
        token = 'xai-' + 'A1b2C3d4E5f6G7h8I9j0K1l2'
        self.http.post.return_value = Response(401, {'detail': {'message': f'Incorrect API key {token}'}})
        with self.assertRaises(CloudError) as caught:
            self.model(ELEVENLABS, fallback=False).transcribe(self.audio, language='pt')
        self.assertNotIn(token, str(caught.exception))
        self.assertIn('HTTP 401', str(caught.exception))

    def test_xai_refused_login_token_renews_once(self):
        secrets = Secrets(tokens=['velho', 'novo'])
        self.http.post.side_effect = [Response(400, text='Incorrect API key'), Response(body=XAI_BODY)]
        self.model(XAI, secrets=secrets, fallback=False).transcribe(self.audio, language='pt')
        self.assertEqual(secrets.margins, [120, 600])
        self.assertEqual(self.http.post.call_args.kwargs['headers'], {'Authorization': 'Bearer novo'})

    def test_xai_api_key_is_not_retried(self):
        secrets = Secrets(tokens=['chave'], login=False)
        self.http.post.return_value = Response(400, text='Incorrect API key')
        with self.assertRaises(CloudError):
            self.model(XAI, secrets=secrets, fallback=False).transcribe(self.audio, language='pt')
        self.assertEqual(self.http.post.call_count, 1)

    def test_timeout_grows_with_audio(self):
        model = self.model()
        self.assertLess(model._timeout(5).read, model._timeout(60).read)
        self.assertEqual(model._timeout(60).connect, 4.0)


class HelperTests(unittest.TestCase):
    def test_keyterms_respect_provider_limits(self):
        terms = cloud.keyterms(['ok', 'x' * 60, 'um dois tres quatro cinco seis', 'a<b', 'ok', 'Claude  Code']
                               + [f't{i}' for i in range(200)])
        self.assertEqual(terms[:2], ['ok', 'Claude Code'])
        self.assertEqual(len(terms), 100)
        self.assertEqual(cloud.keyterms(None), [])
        self.assertEqual(cloud.keyterms('A, B'), ['A', 'B'])

    def test_text_without_word_times_becomes_one_segment(self):
        segments = cloud.segments_from_words([], 'Texto solto', 3.0)
        self.assertEqual([(s.start, s.end, s.text) for s in segments], [(0.0, 3.0, ' Texto solto')])
        self.assertEqual(cloud.segments_from_words([], '  ', 3.0), [])

    def test_short_language_from_either_provider(self):
        self.assertEqual([cloud.short_language(c) for c in ('por', 'eng', 'pt-BR', 'es-mx', 'auto', None, 'ja')],
                         ['pt', 'en', 'pt', 'es', None, None, 'ja'])

    def test_speaker_change_starts_a_new_segment(self):
        items = [('Oi', 0.0, 0.3, True, 0), (' ', 0.3, 0.3, False), ('tudo', 0.3, 0.6, True, 0),
                 (' ', 0.6, 0.6, False), ('Bem', 0.7, 1.0, True, 1)]
        segments = cloud.segments_from_words(items, '', 1.0)
        self.assertEqual([s.text for s in segments], [' Oi tudo', ' Bem'])
        self.assertEqual([w.speaker for w in segments[1].words], [1])


class MeetingRequestTests(unittest.TestCase):
    """Reuniao: falante por palavra, Opus no lugar de PCM e prazo longo."""

    def setUp(self):
        self.http = Mock()
        self.audio = np.zeros(cloud.SAMPLE_RATE * 4, dtype=np.float32)

    def model(self, provider):
        return CloudModel(provider, http=self.http, secrets=Secrets(), fallback=False)

    def test_xai_diarize_reads_speaker_per_word_and_sends_ogg(self):
        body = {'language': 'pt', 'text': 'Oi. Tudo bem?', 'words': [
            {'text': 'Oi.', 'start': 0.1, 'end': 0.3, 'speaker': 0},
            {'text': 'Tudo', 'start': 0.4, 'end': 0.6, 'speaker': 1},
            {'text': 'bem?', 'start': 0.6, 'end': 0.9, 'speaker': 1}]}
        self.http.post.return_value = Response(body=body)
        with patch.object(cloud, 'opus_bytes', return_value=b'OggS-falso'):
            segments, _ = self.model(XAI).transcribe(self.audio, language='pt', diarize=True, compress=True)
        kwargs = self.http.post.call_args.kwargs
        self.assertEqual(kwargs['data']['diarize'], 'true')
        self.assertEqual(kwargs['files']['file'], ('reuniao.ogg', b'OggS-falso', 'audio/ogg'))
        self.assertGreaterEqual(kwargs['timeout'].read, 60)
        self.assertEqual([(s.text, [w.speaker for w in s.words]) for s in segments],
                         [(' Oi.', [0]), (' Tudo bem?', [1, 1])])

    def test_elevenlabs_diarize_with_fixed_count_and_no_pcm_format(self):
        body = {'language_code': 'por', 'text': 'Oi. Bem.', 'words': [
            {'text': 'Oi.', 'start': 0.1, 'end': 0.3, 'type': 'word', 'speaker_id': 'speaker_0'},
            {'text': ' ', 'start': 0.3, 'end': 0.4, 'type': 'spacing', 'speaker_id': 'speaker_0'},
            {'text': 'Bem.', 'start': 0.4, 'end': 0.7, 'type': 'word', 'speaker_id': 'speaker_1'}]}
        self.http.post.return_value = Response(body=body)
        with patch.object(cloud, 'opus_bytes', return_value=b'OggS-falso'):
            segments, _ = self.model(ELEVENLABS).transcribe(self.audio, language='pt', diarize=True,
                                                            speakers=2, compress=True)
        data = self.http.post.call_args.kwargs['data']
        self.assertEqual((data['diarize'], data['num_speakers']), ('true', '2'))
        self.assertNotIn('file_format', data)  # Opus tem cabecalho; pcm_s16le_16 quebraria
        self.assertEqual([w.speaker for s in segments for w in s.words], ['speaker_0', 'speaker_1'])

    def test_without_ffmpeg_meeting_falls_back_to_raw_audio(self):
        self.http.post.return_value = Response(body=XAI_BODY)
        with patch.object(cloud, 'opus_bytes', return_value=None):
            self.model(XAI).transcribe(self.audio, language='pt', compress=True)
        self.assertTrue(self.http.post.call_args.kwargs['files']['file'][1].startswith(b'RIFF'))

    def test_dictation_does_not_ask_for_speakers(self):
        self.http.post.return_value = Response(body=XAI_BODY)
        self.model(XAI).transcribe(self.audio, language=None)
        data = self.http.post.call_args.kwargs['data']
        self.assertNotIn('diarize', data)
        self.assertNotIn('format', data)  # format=true sem idioma: a xAI responde 400

    def test_opus_bytes_is_a_real_ogg(self):
        import shutil
        if not shutil.which('ffmpeg'):
            self.skipTest('sem ffmpeg')
        t = np.arange(cloud.SAMPLE_RATE * 2) / cloud.SAMPLE_RATE
        packed = cloud.opus_bytes((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
        self.assertTrue(packed.startswith(b'OggS'))
        self.assertLess(len(packed), cloud.SAMPLE_RATE * 2 * 2 // 4)  # bem menor que o PCM16


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.auth = Path(self.folder.name) / 'auth.json'

    def write(self, expires):
        stamp = expires.strftime('%Y-%m-%dT%H:%M:%S') + '.250057481Z'
        self.auth.write_text(json.dumps({
            'outro::1': {'oidc_issuer': 'https://outro', 'key': 'nao'},
            'https://auth.x.ai::1': {'oidc_issuer': cloud.GROK_ISSUER, 'key': 'grok-token', 'expires_at': stamp}}))

    def creds(self, env=None):
        return Credentials(env=env or {}, auth_path=self.auth, grok_bin='/nao/existe')

    def test_env_key_wins_and_is_not_a_login(self):
        creds = self.creds({'XAI_API_KEY': 'chave'})
        self.assertEqual(creds.xai_token(), 'chave')
        self.assertFalse(creds.xai_is_login())

    def test_valid_grok_login_is_used_without_renewal(self):
        self.write(datetime.now(timezone.utc) + timedelta(hours=1))
        creds = self.creds()
        with patch.object(creds, '_renew') as renew:
            self.assertEqual(creds.xai_token(), 'grok-token')
        renew.assert_not_called()
        self.assertTrue(creds.has(XAI))

    def test_expiring_login_asks_cli_to_renew(self):
        self.write(datetime.now(timezone.utc) + timedelta(seconds=30))
        creds = self.creds()
        renewed = lambda: self.write(datetime.now(timezone.utc) + timedelta(hours=3))
        with patch.object(creds, '_renew', side_effect=renewed) as renew:
            self.assertEqual(creds.xai_token(), 'grok-token')
        renew.assert_called_once()

    def test_login_that_stays_expired_explains_what_to_do(self):
        self.write(datetime.now(timezone.utc) - timedelta(minutes=5))
        creds = self.creds()
        with patch.object(creds, '_renew'), self.assertRaisesRegex(CloudError, 'grok'):
            creds.xai_token()

    def test_missing_login_is_unavailable(self):
        self.assertFalse(self.creds().has(XAI))
        with self.assertRaisesRegex(CloudError, 'grok login'):
            self.creds().xai_token()

    def test_expiry_is_read_as_utc(self):
        when = datetime(2026, 10, 10, 16, 7, 24, tzinfo=timezone.utc)
        self.assertEqual(Credentials._expires({'expires_at': '2026-10-10T16:07:24.250057481Z'}), when.timestamp())
        self.assertEqual(Credentials._expires({}), 0.0)

    def test_elevenlabs_key_from_env_or_keyring(self):
        self.assertEqual(self.creds({'ELEVENLABS_API_KEY': 'k'}).elevenlabs_key(), 'k')
        creds = self.creds()
        with patch.object(cloud.shutil, 'which', return_value='/usr/bin/secret-tool'), \
             patch.object(cloud.subprocess, 'run', return_value=Mock(stdout='do-chaveiro\n')) as run:
            self.assertEqual(creds.elevenlabs_key(), 'do-chaveiro')
            self.assertEqual(creds.elevenlabs_key(), 'do-chaveiro')
        run.assert_called_once()  # chaveiro consultado uma vez por sessao
        self.assertEqual(run.call_args.args[0][1:], ['lookup', 'servico', 'elevenlabs', 'tipo', 'api-key'])

    def test_missing_elevenlabs_key_explains_keyring(self):
        creds = self.creds()
        with patch.object(cloud.shutil, 'which', return_value=None):
            self.assertFalse(creds.has(ELEVENLABS))
            with self.assertRaisesRegex(CloudError, 'secret-tool store'):
                cloud.require_credentials(ELEVENLABS, creds)


if __name__ == '__main__':
    unittest.main()
