"""Frozen text-path runtime for the unchanged official LLaMA-Omni2 sources."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

QUESTION = 'Listen to the speech. Is the speaker HAPPY or SAD? Answer only HAPPY or SAD.'
SOURCE_COMMIT = 'c8afa9061a9c2d2c1919f7293f5492d946869752'
MODEL_REVISION = 'a16aa9a4ea3f2f363c3db728e8e83ee08e60922c'
CHECKPOINT_SHA256 = 'a34de68dc82e051d01f4c6e32c43addaf6f52721cc4e89db06868b616b76fc5c'
CHECKPOINT_SIZE = 3856817560


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def classify_checkpoint_keys(expected, available):
    """Reject unexplained weights; permit only absent external encoder and tied head."""
    expected, available = set(expected), set(available)
    extra = available - expected
    invalid = sorted(key for key in extra if not key.startswith('speech_generator.'))
    missing = expected - available
    invalid_missing = sorted(key for key in missing if key != 'lm_head.weight' and not key.startswith('model.speech_encoder.'))
    if invalid or invalid_missing:
        raise ValueError(f'Unexpected checkpoint keys: {invalid}; missing keys: {invalid_missing}')
    return missing, extra


def _official_module(source_root):
    source_root = Path(source_root).resolve()
    revision = subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError(f'Unexpected official source revision: {revision}')
    sys.path.insert(0, str(source_root))
    # The package initializer imports obsolete, unused TTS generation internals.
    # Namespace loading leaves every imported text-path source file unchanged.
    for name, relative in [('llama_omni2', 'llama_omni2'), ('llama_omni2.model', 'llama_omni2/model')]:
        existing = sys.modules.get(name)
        directory = str(source_root / relative)
        if existing is not None and list(getattr(existing, '__path__', [])) != [directory]:
            raise RuntimeError(f'Conflicting imported package: {name}')
        if existing is None:
            package = types.ModuleType(name)
            package.__path__ = [directory]
            sys.modules[name] = package
    return importlib.import_module('llama_omni2.model.language_model.omni2_speech_qwen2')


def load_runtime(source_root, model_path, whisper_path, device):
    import torch
    import whisper
    import transformers
    from safetensors import safe_open
    from transformers import AutoTokenizer

    source_root, model_path, whisper_path = map(Path, (source_root, model_path, whisper_path))
    checkpoint_path = model_path / 'model.safetensors'
    if checkpoint_path.stat().st_size != CHECKPOINT_SIZE:
        raise ValueError('Official checkpoint file size mismatch')
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError('Official pinned checkpoint SHA256 mismatch')
    official = _official_module(source_root)
    raw = json.loads((model_path / 'config.json').read_text())
    if (raw['hidden_size'], raw['num_hidden_layers'], raw['speech_encoder_ds_rate']) != (896, 24, 5):
        raise ValueError('Unexpected checkpoint dimensions')
    config = official.Omni2SpeechQwen2Config.from_dict(raw)
    config.speech_encoder = str(whisper_path.resolve())
    config._attn_implementation = 'eager'
    # Instantiate normally, avoiding meta initialization of external Whisper.
    model = official.Omni2SpeechQwen2ForCausalLM(config).float()
    state = model.state_dict()
    with safe_open(str(model_path / 'model.safetensors'), framework='pt', device='cpu') as checkpoint:
        missing, extra = classify_checkpoint_keys(state, checkpoint.keys())
        verified = 0
        with torch.no_grad():
            for name, destination in state.items():
                if name in missing:
                    continue
                source = checkpoint.get_tensor(name).to(dtype=destination.dtype)
                if source.shape != destination.shape:
                    raise ValueError(f'Checkpoint shape mismatch: {name}')
                destination.copy_(source)
                if not torch.equal(destination, source):
                    raise AssertionError(f'Weight copy mismatch: {name}')
                verified += 1
        # Tied output head may legitimately be omitted by safetensors serialization.
        if 'lm_head.weight' in missing:
            if not config.tie_word_embeddings:
                raise ValueError('Missing untied LM head')
            model.tie_weights()
            if not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
                raise AssertionError('Tied head mismatch')
        # Recheck every key after all copies/ties: shared destinations can otherwise
        # invalidate an earlier successful per-tensor verification.
        final_state = model.state_dict()
        for name in state:
            if name not in missing and not torch.equal(final_state[name], checkpoint.get_tensor(name).to(final_state[name].dtype)):
                raise AssertionError(f'Final checkpoint/alias mismatch: {name}')
        if config.tie_word_embeddings and model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
            raise AssertionError('Expected tied embedding/head storage')
        del final_state

    encoder_missing = [name for name in missing if name.startswith('model.speech_encoder.')]
    if encoder_missing:
        encoder_expected = [name for name in state if name.startswith('model.speech_encoder.')]
        if len(encoder_missing) != len(encoder_expected):
            raise ValueError('Partially missing encoder is unsupported')
        # Official model post_init can reinitialize externally loaded submodules:
        # restore and verify every encoder tensor explicitly from official Whisper.
        external = torch.load(whisper_path, map_location='cpu', weights_only=False)['model_state_dict']
        encoder_state = {name.removeprefix('encoder.'): value for name, value in external.items() if name.startswith('encoder.')}
        model.get_speech_encoder().load_state_dict(encoder_state, strict=True)
        for name, value in model.get_speech_encoder().state_dict().items():
            if not torch.equal(value, encoder_state[name].to(value.dtype)):
                raise AssertionError(f'External encoder mismatch: {name}')
        del external, encoder_state
    del state
    model.eval().requires_grad_(False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    if tokenizer.convert_tokens_to_ids('<speech>') != 151665:
        raise ValueError('Unexpected speech placeholder id')
    provenance = dict(source_commit=SOURCE_COMMIT, model_revision=MODEL_REVISION,
        torch_version=torch.__version__, transformers_version=transformers.__version__, dtype='float32',
        attention_implementation='eager', model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
        verified_checkpoint_tensors=verified, unused_speech_generator_tensors=len(extra),
        external_encoder_tensors=len(encoder_missing), tied_lm_head_derived='lm_head.weight' in missing,
        question=QUESTION, user_content='<speech>\n' + QUESTION,
        config_override={'speech_encoder': str(whisper_path.resolve()), '_attn_implementation':'eager'},
        checkpoint_sha256=checkpoint_hash, checkpoint_size=CHECKPOINT_SIZE, final_checkpoint_alias_audit=True, whisper_sha256=sha256(whisper_path))
    return torch, model, tokenizer, whisper, provenance


def prepare_prefix(torch, model, tokenizer, whisper, audio_path):
    messages = [{'role':'user', 'content':'<speech>\n' + QUESTION}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(model.device)
    locations = (ids[0] == tokenizer.convert_tokens_to_ids('<speech>')).nonzero().flatten()
    if locations.numel() != 1:
        raise ValueError('Expected one speech placeholder')
    start = int(locations.item())
    ids = ids.clone()
    ids[ids == 151665] = -200
    audio = whisper.pad_or_trim(whisper.load_audio(str(audio_path)))
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).permute(1, 0)
    speech = mel.unsqueeze(0).to(device=model.device, dtype=next(model.parameters()).dtype)
    lengths = torch.tensor([len(mel)], device=model.device, dtype=torch.long)
    projected = []
    handle = model.get_speech_projector().register_forward_hook(lambda _module, _inputs, output: projected.append(output.detach()))
    try:
        with torch.inference_mode():
            prepared = model.prepare_inputs_labels_for_speech_and_text(ids, None, None, None, None, speech, lengths)
    finally:
        handle.remove()
    embeddings = prepared[4]
    if embeddings is None or embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[2] != 896:
        raise ValueError('Unexpected prepared embeddings')
    audio_length = embeddings.shape[1] - ids.shape[1] + 1
    if audio_length != 300 or not torch.isfinite(embeddings).all():
        raise ValueError('Unexpected audio length or nonfinite prefix')
    if len(projected) != 1 or not torch.equal(embeddings[:, start:start+audio_length], projected[0][:, :audio_length]):
        raise AssertionError('Official projector/prefix span mismatch')
    nonaudio_ids = torch.cat((ids[:, :start], ids[:, start+1:]), dim=1)
    nonaudio_embeds = torch.cat((embeddings[:, :start], embeddings[:, start+audio_length:]), dim=1)
    if not torch.equal(nonaudio_embeds, model.model.embed_tokens(nonaudio_ids)):
        raise AssertionError('Official nonaudio prefix embedding mismatch')
    return embeddings, dict(audio_start=start, audio_length=audio_length,
        prefix_length=embeddings.shape[1], decision_position=embeddings.shape[1]-1,
        prompt_token_ids=ids[0].cpu().tolist(), rendered_chat=tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))


def validate_forward_parity(torch, model, embeddings):
    with torch.inference_mode():
        official = model(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True, return_dict=True)
        direct = model.model(inputs_embeds=embeddings, use_cache=False, return_dict=True)
        state = direct.last_hidden_state
        logits = model.lm_head(state)[:, -1]
        state_error = (state - official.hidden_states[-1]).abs().max().item()
        logits_error = (logits - official.logits[:, -1]).abs().max().item()
    if not torch.isfinite(state).all() or not torch.isfinite(logits).all():
        raise AssertionError('Nonfinite parity output')
    if state_error != 0 or logits_error > 1e-4:
        raise AssertionError(f'Official/direct forward mismatch: {state_error}, {logits_error}')
    return dict(final_state_max_abs_error=state_error, last_logit_max_abs_error=logits_error)
