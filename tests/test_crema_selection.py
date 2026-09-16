import io
import unittest
import wave
from scripts.prepare_crema_confirmation import REVISION, STATEMENTS, selection, validate_audio


class TestCremaSelection(unittest.TestCase):
    def test_incomplete_actor_excluded_before_model_and_nonxx_ignored(self):
        paths = [f'AudioWAV/{a}_{s}_{e}_XX.wav' for a in ('1001', '1002') for s in STATEMENTS for e in ('HAP', 'SAD')]
        paths.remove('AudioWAV/1002_DFA_SAD_XX.wav')
        paths.append('AudioWAV/1003_IEO_HAP_HI.wav')
        rows, excluded = selection({'sha': REVISION, 'tree': [{'path': p} for p in paths]})
        self.assertEqual(len(rows), 22)
        self.assertEqual({r['actor'] for r in rows}, {'1001'})
        self.assertEqual(excluded, {'1002': ['DFA']})
        self.assertEqual(len({r['pair_id'] for r in rows}), 11)
        with self.assertRaises(ValueError):
            selection({'sha': 'wrong', 'tree': []})

    def test_wav_validation_rejects_truncation_and_lfs_stub(self):
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(b'\0' * 320)
        raw = stream.getvalue()
        self.assertEqual(validate_audio(raw)['frames'], 160)
        with self.assertRaises(ValueError):
            validate_audio(raw[:-2])
        with self.assertRaises((wave.Error, EOFError)):
            validate_audio(b'version https://git-lfs.github.com/spec/v1\n')
