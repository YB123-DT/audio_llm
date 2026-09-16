import pytest
from scripts.llama_omni2_runtime import classify_checkpoint_keys, QUESTION


def test_weight_coverage_rejects_unexplained_missing_or_extra():
    keys = ['model.embed_tokens.weight', 'model.speech_projector.linear1.weight', 'model.speech_encoder.conv1.weight', 'lm_head.weight']
    missing, extra = classify_checkpoint_keys(keys, ['model.embed_tokens.weight', 'model.speech_projector.linear1.weight', 'speech_generator.model.weight'])
    assert missing == {'model.speech_encoder.conv1.weight', 'lm_head.weight'}
    assert extra == {'speech_generator.model.weight'}
    with pytest.raises(ValueError, match='missing keys'):
        classify_checkpoint_keys(keys, ['model.embed_tokens.weight'])
    with pytest.raises(ValueError, match='Unexpected checkpoint'):
        classify_checkpoint_keys(keys, keys + ['wrong_decoder.weight'])


def test_fixed_question_preserves_previous_experiment():
    assert QUESTION == 'Listen to the speech. Is the speaker HAPPY or SAD? Answer only HAPPY or SAD.'


def test_official_prefix_span_and_last_answer_position():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    from scripts.llama_omni2_runtime import prepare_prefix

    class Tokenizer:
        def convert_tokens_to_ids(self, token):
            assert token == '<speech>'
            return 151665
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role':'user', 'content':'<speech>\n' + QUESTION}]
            assert kwargs['add_generation_prompt']
            if kwargs.get('tokenize') is False:
                return 'rendered chat'
            return torch.tensor([[1, 151665, 2, 3]])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(4, 896)
            self.projector = torch.nn.Identity()
            self.device = torch.device('cpu')
        def get_speech_projector(self):
            return self.projector
        def prepare_inputs_labels_for_speech_and_text(self, ids, *_args):
            assert ids.tolist() == [[1, -200, 2, 3]]
            speech, lengths = _args[-2:]
            assert speech.shape == (1, 3000, 128)
            assert lengths.tolist() == [3000]
            projected = self.projector(torch.ones(1, 300, 896))
            embeddings = torch.cat((self.model.embed_tokens(ids[:, :1]), projected, self.model.embed_tokens(ids[:, 2:])), dim=1)
            return None, None, None, None, embeddings, None

    whisper = SimpleNamespace(load_audio=lambda path: path, pad_or_trim=lambda audio: audio,
        log_mel_spectrogram=lambda audio, n_mels: torch.zeros(n_mels, 3000))
    embeddings, positions = prepare_prefix(torch, Model(), Tokenizer(), whisper, 'dummy.wav')
    assert embeddings.shape == (1, 303, 896)
    assert positions['audio_start'] == 1
    assert positions['audio_length'] == 300
    assert positions['decision_position'] == 302
