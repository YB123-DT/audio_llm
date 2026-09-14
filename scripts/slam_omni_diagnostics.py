#!/usr/bin/env python3
"""Run frozen SLAM-Omni diagnostics on a RAVDESS manifest.

The script deliberately separates free ``output``, teacher-forced
``forced-choice``, and ``hidden`` stages.  The hidden stage performs one
encoder/projector/LLM forward per clip and writes small pooled vectors plus
metadata.  None of the stages train, alter checkpoints, or update model
parameters.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.machinery
import json
import logging
import random
import re
import sys
import types
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("slam_omni_diagnostics")
PROMPT_TEMPLATE = "<SYSTEM>: {}\n "
TEXT_VOCAB_SIZE = 151936
TEXT_SPECIAL_TOKENS = 64
AUDIO_VOCAB_SIZE = 4096
AUDIO_SPECIAL_TOKENS = 64
PADDED_TEXT_VOCAB_SIZE = TEXT_VOCAB_SIZE + TEXT_SPECIAL_TOKENS
PADDED_AUDIO_VOCAB_SIZE = AUDIO_VOCAB_SIZE + AUDIO_SPECIAL_TOKENS
CODE_LAYER = 3
TOTAL_AUDIO_VOCAB_SIZE = PADDED_AUDIO_VOCAB_SIZE
TOTAL_VOCAB_SIZE = PADDED_TEXT_VOCAB_SIZE + TOTAL_AUDIO_VOCAB_SIZE
SHIFT = PADDED_TEXT_VOCAB_SIZE
EMOTION_TO_LABEL = {"happy": 1, "sad": 0}
DEFAULT_PROMPT = "Listen to the speech. Is the speaker HAPPY or SAD? Answer only HAPPY or SAD."

# A small 2x2 prompt/verbalizer matrix makes forced-choice results auditable.
# The leading space is intentional: it is the token boundary used when the
# candidate answer follows the ``answer_t`` marker in the SLAM-Omni stream.
FORCED_CHOICE_CONDITIONS: dict[str, dict[str, str]] = {
    "upper_prompt__upper_spaced": {
        "prompt": DEFAULT_PROMPT,
        "happy_verbalizer": " HAPPY",
        "sad_verbalizer": " SAD",
    },
    "upper_prompt__lower_spaced": {
        "prompt": DEFAULT_PROMPT,
        "happy_verbalizer": " happy",
        "sad_verbalizer": " sad",
    },
    "lower_prompt__upper_spaced": {
        "prompt": "Listen to the speech. Is the speaker happy or sad? Answer only happy or sad.",
        "happy_verbalizer": " HAPPY",
        "sad_verbalizer": " SAD",
    },
    "lower_prompt__lower_spaced": {
        "prompt": "Listen to the speech. Is the speaker happy or sad? Answer only happy or sad.",
        "happy_verbalizer": " happy",
        "sad_verbalizer": " sad",
    },
}


def normalize_prediction(text: str) -> tuple[str, str]:
    """Return a deterministic label and a short parse status for raw output."""
    matches = re.findall(r"\b(HAPPY|SAD)\b", text.upper())
    if not matches:
        return "unknown", "no_label"
    if len(set(matches)) > 1:
        return matches[0].lower(), "ambiguous_first_label"
    return matches[0].lower(), "single_label"


def format_prompt(prompt: str) -> str:
    """Serialize user prompt with the official SLAM-Omni dataset wrapper."""
    return PROMPT_TEMPLATE.format(prompt)


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "pair_id", "path", "actor", "statement", "repetition", "intensity", "emotion"}
    missing = required.difference(rows[0] if rows else required)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if limit is not None:
        rows = rows[:limit]
    return rows


def _import_runtime(slam_llm_root: Path) -> tuple[Any, Any, Any, Any]:
    """Import the official SLAM-LLM factory and config helpers lazily."""
    root = slam_llm_root.resolve()
    sys.path.insert(0, str(root / "examples" / "s2s"))
    sys.path.insert(0, str(root / "src"))
    try:
        import torch
        from omegaconf import OmegaConf
        # The official utility package imports one DeepSpeed checkpoint helper
        # even for single-GPU inference.  Keep this diagnostic independent of
        # the training-only DeepSpeed install while leaving the official code
        # untouched; the helper is never called by this script.
        try:
            import deepspeed  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            deepspeed = types.ModuleType("deepspeed")
            deepspeed_utils = types.ModuleType("deepspeed.utils")
            zero_to_fp32 = types.ModuleType("deepspeed.utils.zero_to_fp32")
            deepspeed.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
            deepspeed.__version__ = "0.0.0-diagnostic-stub"
            deepspeed_utils.__spec__ = importlib.machinery.ModuleSpec("deepspeed.utils", loader=None)
            zero_to_fp32.__spec__ = importlib.machinery.ModuleSpec("deepspeed.utils.zero_to_fp32", loader=None)
            zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[attr-defined]
                RuntimeError("DeepSpeed checkpoint conversion is unavailable in frozen diagnostics")
            )
            deepspeed.utils = deepspeed_utils  # type: ignore[attr-defined]
            deepspeed_utils.zero_to_fp32 = zero_to_fp32  # type: ignore[attr-defined]
            sys.modules.update(
                {
                    "deepspeed": deepspeed,
                    "deepspeed.utils": deepspeed_utils,
                    "deepspeed.utils.zero_to_fp32": zero_to_fp32,
                }
            )
        try:
            import wandb  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            wandb = types.ModuleType("wandb")
            wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
            wandb.init = lambda *args, **kwargs: None  # type: ignore[attr-defined]
            wandb.log = lambda *args, **kwargs: None  # type: ignore[attr-defined]
            wandb.finish = lambda *args, **kwargs: None  # type: ignore[attr-defined]
            sys.modules["wandb"] = wandb
        from model.slam_model_s2s import model_factory
    except ImportError as exc:  # pragma: no cover - exercised on runtime host
        raise RuntimeError(
            "SLAM-Omni runtime imports are unavailable. Install the official "
            "SLAM-LLM dependencies and openai-whisper in the selected environment."
        ) from exc
    return torch, OmegaConf, model_factory, importlib.import_module("whisper")


def _make_configs(OmegaConf: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    model_config = OmegaConf.create(
        {
            "llm_name": "qwen2-0.5b",
            "llm_path": str(args.qwen_path),
            "llm_dim": 896,
            "encoder_name": "whisper",
            "encoder_path": str(args.whisper_path),
            "encoder_path_hf": None,
            "encoder_dim": 768,
            "encoder_projector": "linear",
            "encoder_projector_ds_rate": 5,
            "encoder_type": "finetune",
            "codec_decode": False,
            "codec_decoder_type": "CosyVoice",
            "code_type": "CosyVoice",
            "tts_adapter": False,
            "group_decode": True,
            "group_decode_adapter_type": "linear",
            "whisper_decode": True,
            "vocab_config": {
                "text_vocabsize": TEXT_VOCAB_SIZE,
                "text_specialtokens": TEXT_SPECIAL_TOKENS,
                "audio_vocabsize": AUDIO_VOCAB_SIZE,
                "audio_specialtokens": AUDIO_SPECIAL_TOKENS,
                "code_layer": CODE_LAYER,
                "padded_text_vocabsize": PADDED_TEXT_VOCAB_SIZE,
                "padded_audio_vocabsize": PADDED_AUDIO_VOCAB_SIZE,
                "total_audio_vocabsize": TOTAL_AUDIO_VOCAB_SIZE,
                "total_vocabsize": TOTAL_VOCAB_SIZE,
                "eot": TEXT_VOCAB_SIZE,
                "pad_t": TEXT_VOCAB_SIZE + 1,
                "input_t": TEXT_VOCAB_SIZE + 2,
                "answer_t": TEXT_VOCAB_SIZE + 3,
                "asr": TEXT_VOCAB_SIZE + 4,
                "eoa": AUDIO_VOCAB_SIZE,
                "pad_a": AUDIO_VOCAB_SIZE + 1,
                "input_a": AUDIO_VOCAB_SIZE + 2,
                "answer_a": AUDIO_VOCAB_SIZE + 3,
                "split": AUDIO_VOCAB_SIZE + 4,
            },
        }
    )
    train_config = OmegaConf.create(
        {
            "model_name": "SLAM-Omni-0.5B-diagnostic",
            "task_type": "s2s",
            "enable_ddp": False,
            "enable_fsdp": False,
            "low_cpu_fsdp": False,
            "enable_deepspeed": False,
            "quantization": False,
            "freeze_encoder": True,
            "freeze_llm": True,
            "freeze_encoder_projector": True,
            "freeze_group_decode_adapter": True,
            "train_audio_embed_only": False,
            "train_embed_only": False,
            "train_embed": False,
            "use_peft": False,
        }
    )
    return train_config, model_config


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    torch, OmegaConf, model_factory, whisper = _import_runtime(args.slam_llm_root)
    seed = getattr(args, "seed", None)
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:  # pragma: no cover - numpy is a runtime dependency
            pass
    train_config, model_config = _make_configs(OmegaConf, args)
    LOGGER.info("loading frozen model")
    model, tokenizer = model_factory(
        train_config,
        model_config,
        ckpt_path=str(args.checkpoint),
        metric=None,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    return torch, model, tokenizer, whisper


def _build_audio_inputs(torch: Any, model: Any, tokenizer: Any, audio_mel: Any, prompt: str) -> tuple[dict[str, Any], dict[str, int]]:
    """Build the multimodal token streams for an already prepared mel."""
    audio_length = (audio_mel.shape[0] + 1) // 2 // 5

    vocab = model.model_config.vocab_config
    prompt_ids = tokenizer.encode(format_prompt(prompt))
    prompt_ids = [int(vocab.input_t)] + prompt_ids + [int(vocab.eot)]
    prompt_length = len(prompt_ids)

    prompt_layers: list[Any] = []
    for _ in range(CODE_LAYER):
        prompt_layers.append(torch.tensor([SHIFT + int(vocab.pad_a)] * prompt_length, dtype=torch.long))
    prompt_layers.append(torch.tensor(prompt_ids, dtype=torch.long))

    example_layers: list[Any] = []
    for _ in range(CODE_LAYER):
        example_layers.append(
            torch.tensor(
                [SHIFT + int(vocab.input_a)]
                + [SHIFT + int(vocab.pad_a)] * audio_length
                + [SHIFT + int(vocab.eoa), SHIFT + int(vocab.answer_a)],
                dtype=torch.long,
            )
        )
    example_layers.append(
        torch.tensor(
            [int(vocab.input_t)]
            + [int(vocab.pad_t)] * audio_length
            + [int(vocab.eot), int(vocab.answer_t)],
            dtype=torch.long,
        )
    )
    input_ids = torch.stack([torch.cat((p, e)) for p, e in zip(prompt_layers, example_layers)]).unsqueeze(0)
    attention_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
    modality_mask = torch.zeros_like(attention_mask)
    audio_start = prompt_length + 1
    modality_mask[:, audio_start : audio_start + audio_length] = True
    device = next(model.parameters()).device
    batch = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "audio_mel": audio_mel.unsqueeze(0).to(device),
        "audio_length": torch.tensor([audio_length], dtype=torch.long, device=device),
        "input_length": torch.tensor([audio_length], dtype=torch.long, device=device),
        "modality_mask": modality_mask.to(device),
        "task_types": ["s2s"],
        "mini_omni_modeling": False,
    }
    return batch, {"audio_start": audio_start, "audio_length": audio_length, "prompt_last": prompt_length - 1}


def _audio_inputs(torch: Any, whisper: Any, model: Any, tokenizer: Any, wav_path: Path, prompt: str) -> tuple[dict[str, Any], dict[str, int]]:
    audio_raw = _load_audio_16k(whisper, wav_path)
    audio_raw = whisper.pad_or_trim(audio_raw)
    audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=80).permute(1, 0)
    return _build_audio_inputs(torch, model, tokenizer, audio_mel, prompt)


def _load_audio_16k(whisper: Any, wav_path: Path) -> Any:
    """Load a WAV through Whisper, with a dependency-light ffmpeg fallback."""
    try:
        return whisper.load_audio(str(wav_path))
    except FileNotFoundError as exc:
        if exc.filename != "ffmpeg":
            raise
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            audio = resample_poly(audio, 16000, sample_rate)
        return np.asarray(audio, dtype=np.float32)


def _extract_whisper_layers(torch: Any, encoder: Any, audio_mel: Any) -> tuple[Any, Any, list[str]]:
    x = torch.nn.functional.gelu(encoder.conv1(audio_mel.permute(0, 2, 1)))
    x = torch.nn.functional.gelu(encoder.conv2(x))
    x = x.permute(0, 2, 1)
    x = (x + encoder.positional_embedding[: x.shape[1]]).to(x.dtype)
    means = [x.mean(dim=1)]
    names = ["conv2"]
    for index, block in enumerate(encoder.blocks):
        x = block(x)
        means.append(x.mean(dim=1))
        names.append(f"block_{index}")
    x = encoder.ln_post(x)
    means.append(x.mean(dim=1))
    names.append("final")
    return x, torch.stack(means, dim=1), names


def _embed_llm_input_ids(model: Any, input_ids: Any) -> Any:
    """Embed the four parallel SLAM-Omni streams with the official LLM."""
    llm = model.llm
    if hasattr(llm.model, "embed_tokens"):
        inputs_embeds = llm.model.embed_tokens(input_ids)
    elif hasattr(llm.model.model, "embed_tokens"):
        inputs_embeds = llm.model.model.embed_tokens(input_ids)
    else:
        inputs_embeds = llm.model.model.model.embed_tokens(input_ids)
    return inputs_embeds


def _compose_llm_inputs(torch: Any, model: Any, input_ids: Any, attention_mask: Any, encoder_outs: Any, projector_outs: Any, audio_start: int, audio_length: int) -> Any:
    inputs_embeds = _embed_llm_input_ids(model, input_ids)
    modality_mask = torch.zeros((input_ids.shape[0], input_ids.shape[-1]), dtype=torch.bool, device=input_ids.device)
    modality_mask[:, audio_start : audio_start + audio_length] = True
    modality_mask = modality_mask.unsqueeze(1).repeat(1, CODE_LAYER, 1)
    modality_lengths = min(projector_outs.shape[1], audio_length)
    encoder_outs_pad = torch.zeros_like(inputs_embeds)
    encoder_outs_pad[:, :CODE_LAYER, audio_start : audio_start + modality_lengths] = projector_outs[:, :modality_lengths].unsqueeze(1)
    inputs_embeds[:, :CODE_LAYER] = encoder_outs_pad[:, :CODE_LAYER] + inputs_embeds[:, :CODE_LAYER] * (~modality_mask[..., None])
    return inputs_embeds.mean(dim=1), attention_mask


def _append_text_candidate(torch: Any, model: Any, base_inputs_embeds: Any, candidate_ids: list[int]) -> Any:
    """Append teacher-forced text tokens and audio padding streams."""
    if not candidate_ids:
        raise ValueError("Candidate verbalizer tokenizes to an empty sequence")
    vocab = model.model_config.vocab_config
    candidate_streams = torch.full(
        (1, CODE_LAYER + 1, len(candidate_ids)),
        SHIFT + int(vocab.pad_a),
        dtype=torch.long,
        device=base_inputs_embeds.device,
    )
    candidate_streams[:, CODE_LAYER, :] = torch.tensor(candidate_ids, dtype=torch.long, device=base_inputs_embeds.device)
    candidate_embeds = _embed_llm_input_ids(model, candidate_streams).mean(dim=1)
    return torch.cat((base_inputs_embeds, candidate_embeds), dim=1)


def _teacher_forced_logprob(torch: Any, model: Any, base_inputs_embeds: Any, base_attention_mask: Any, candidate_ids: list[int]) -> float:
    """Return log P(candidate | multimodal prefix) for a token sequence."""
    with torch.inference_mode():
        full_inputs_embeds = _append_text_candidate(torch, model, base_inputs_embeds, candidate_ids)
        full_attention_mask = torch.ones(
            (full_inputs_embeds.shape[0], full_inputs_embeds.shape[1]),
            dtype=base_attention_mask.dtype,
            device=full_inputs_embeds.device,
        )
        outputs = model.llm(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        text_vocab_size = int(model.model_config.vocab_config.padded_text_vocabsize)
        logits = outputs.logits[:, :, :text_vocab_size]
        prefix_length = base_inputs_embeds.shape[1]
        positions = torch.arange(len(candidate_ids), device=logits.device) + prefix_length - 1
        target_ids = torch.tensor(candidate_ids, dtype=torch.long, device=logits.device)
        token_logprobs = torch.log_softmax(logits[0, positions, :], dim=-1)
        return float(token_logprobs.gather(1, target_ids[:, None]).sum().item())


def _resolve_forced_choice_conditions(names: list[str] | None) -> list[tuple[str, dict[str, str]]]:
    selected = names or list(FORCED_CHOICE_CONDITIONS)
    unknown = [name for name in selected if name not in FORCED_CHOICE_CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown forced-choice condition(s): {unknown}; choose from {list(FORCED_CHOICE_CONDITIONS)}")
    return [(name, FORCED_CHOICE_CONDITIONS[name]) for name in selected]


def run_output(args: argparse.Namespace) -> None:
    torch, model, tokenizer, whisper = load_model(args)
    rows = read_rows(args.manifest)
    if args.start:
        rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt
    fieldnames = ["sample_id", "pair_id", "actor", "statement", "repetition", "intensity", "emotion", "ground_truth", "model_output", "predicted", "parse_status", "correct", "error"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            result = {key: row.get(key, "") for key in fieldnames}
            result["ground_truth"] = row["emotion"]
            result["error"] = ""
            try:
                batch, _ = _audio_inputs(torch, whisper, model, tokenizer, args.ravdess_root / row["path"], prompt)
                output = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    decode_text_only=True,
                    do_sample=False,
                    top_p=1.0,
                    top_k=0,
                    temperature=1.0,
                    text_repetition_penalty=1.0,
                    audio_repetition_penalty=1.0,
                    num_latency_tokens=0,
                    do_layershift=False,
                )
                raw = tokenizer.decode(output[CODE_LAYER], add_special_tokens=False, skip_special_tokens=True)
                pred, status = normalize_prediction(raw)
                result.update(model_output=raw, predicted=pred, parse_status=status, correct=str(pred == row["emotion"]))
            except Exception as exc:  # keep the diagnostic audit complete
                LOGGER.exception("output failed for %s", row["sample_id"])
                result.update(model_output="", predicted="unknown", parse_status="error", correct="False", error=f"{type(exc).__name__}: {exc}")
            writer.writerow(result)
            handle.flush()
            if (index + 1) % 8 == 0:
                LOGGER.info("output stage: %d/%d", index + 1, len(rows))


def run_forced_choice(args: argparse.Namespace) -> None:
    """Score candidate emotion labels with teacher-forced sequence likelihood."""
    torch, model, tokenizer, whisper = load_model(args)
    rows = read_rows(args.manifest)
    if args.start:
        rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]
    conditions = _resolve_forced_choice_conditions(args.forced_choice_condition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_fields = ["sample_id", "pair_id", "actor", "statement", "repetition", "intensity", "emotion", "ground_truth"]
    fieldnames = [
        *metadata_fields,
        "condition_id",
        "prompt",
        "happy_verbalizer",
        "sad_verbalizer",
        "happy_token_ids",
        "sad_token_ids",
        "happy_logprob",
        "sad_logprob",
        "score_happy_minus_sad",
        "predicted",
        "correct",
        "error",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            try:
                audio_raw = whisper.pad_or_trim(_load_audio_16k(whisper, args.ravdess_root / row["path"]))
                audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=80).permute(1, 0)
                audio_mel_batch = audio_mel.unsqueeze(0).to(next(model.parameters()).device)
                with torch.inference_mode():
                    encoder_outs = model.encoder.extract_variable_length_features(audio_mel_batch.permute(0, 2, 1))
                    projected = model.encoder_projector(encoder_outs)
                for condition_id, condition in conditions:
                    result = {key: row.get(key, "") for key in metadata_fields}
                    result["ground_truth"] = row["emotion"]
                    result.update(
                        condition_id=condition_id,
                        prompt=condition["prompt"],
                        happy_verbalizer=condition["happy_verbalizer"],
                        sad_verbalizer=condition["sad_verbalizer"],
                        happy_token_ids="",
                        sad_token_ids="",
                        happy_logprob="",
                        sad_logprob="",
                        score_happy_minus_sad="",
                        predicted="unknown",
                        correct="False",
                        error="",
                    )
                    try:
                        batch, positions = _build_audio_inputs(torch, model, tokenizer, audio_mel, condition["prompt"])
                        base_inputs_embeds, base_attention_mask = _compose_llm_inputs(
                            torch,
                            model,
                            batch["input_ids"],
                            batch["attention_mask"],
                            encoder_outs,
                            projected,
                            positions["audio_start"],
                            positions["audio_length"],
                        )
                        happy_ids = [int(token_id) for token_id in tokenizer.encode(condition["happy_verbalizer"], add_special_tokens=False)]
                        sad_ids = [int(token_id) for token_id in tokenizer.encode(condition["sad_verbalizer"], add_special_tokens=False)]
                        happy_logprob = _teacher_forced_logprob(torch, model, base_inputs_embeds, base_attention_mask, happy_ids)
                        sad_logprob = _teacher_forced_logprob(torch, model, base_inputs_embeds, base_attention_mask, sad_ids)
                        score = happy_logprob - sad_logprob
                        predicted = "happy" if score > 0 else "sad" if score < 0 else "tie"
                        result.update(
                            happy_token_ids=json.dumps(happy_ids),
                            sad_token_ids=json.dumps(sad_ids),
                            happy_logprob=happy_logprob,
                            sad_logprob=sad_logprob,
                            score_happy_minus_sad=score,
                            predicted=predicted,
                            correct=str(predicted == row["emotion"]),
                        )
                    except Exception as exc:  # keep one condition failure auditable
                        LOGGER.exception("forced-choice failed for %s/%s", row["sample_id"], condition_id)
                        result["error"] = f"{type(exc).__name__}: {exc}"
                    writer.writerow(result)
            except Exception as exc:  # keep every condition represented on audio failures
                LOGGER.exception("forced-choice audio failed for %s", row["sample_id"])
                for condition_id, condition in conditions:
                    result = {key: row.get(key, "") for key in metadata_fields}
                    result["ground_truth"] = row["emotion"]
                    result.update(
                        condition_id=condition_id,
                        prompt=condition["prompt"],
                        happy_verbalizer=condition["happy_verbalizer"],
                        sad_verbalizer=condition["sad_verbalizer"],
                        happy_token_ids="",
                        sad_token_ids="",
                        happy_logprob="",
                        sad_logprob="",
                        score_happy_minus_sad="",
                        predicted="unknown",
                        correct="False",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    writer.writerow(result)
            handle.flush()
            if (index + 1) % 8 == 0:
                LOGGER.info("forced-choice stage: %d/%d", index + 1, len(rows))


def run_hidden(args: argparse.Namespace) -> None:
    torch, model, tokenizer, whisper = load_model(args)
    rows = read_rows(args.manifest, args.limit)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt
    metadata: list[dict[str, str]] = []
    audio_vectors: list[Any] = []
    projector_vectors: list[Any] = []
    llm_audio_vectors: list[Any] = []
    llm_decision_vectors: list[Any] = []
    llm_prompt_vectors: list[Any] = []
    audio_layer_names: list[str] | None = None
    llm_layer_names: list[str] | None = None
    errors: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        try:
            batch, positions = _audio_inputs(torch, whisper, model, tokenizer, args.ravdess_root / row["path"], prompt)
            encoder_outs, audio_means, names = _extract_whisper_layers(torch, model.encoder, batch["audio_mel"])
            projected = model.encoder_projector(encoder_outs)
            inputs_embeds, attention_mask = _compose_llm_inputs(
                torch,
                model,
                batch["input_ids"],
                batch["attention_mask"],
                encoder_outs,
                projected,
                positions["audio_start"],
                positions["audio_length"],
            )
            outputs = model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = torch.stack([state[0] for state in outputs.hidden_states])
            audio_length = min(positions["audio_length"], projected.shape[1])
            audio_slice = slice(positions["audio_start"], positions["audio_start"] + audio_length)
            llm_audio = hidden[:, audio_slice, :].mean(dim=1)
            llm_decision = hidden[:, -1, :]
            llm_prompt = hidden[:, positions["prompt_last"], :]
            metadata.append({**row, "representation_index": str(len(metadata)), "error": ""})
            audio_vectors.append(audio_means[0].float().cpu())
            projector_vectors.append(projected[:, :audio_length, :].mean(dim=1)[0].float().cpu())
            llm_audio_vectors.append(llm_audio.float().cpu())
            llm_decision_vectors.append(llm_decision.float().cpu())
            llm_prompt_vectors.append(llm_prompt.float().cpu())
            audio_layer_names = names
            llm_layer_names = ["embedding"] + [f"layer_{i}" for i in range(hidden.shape[0] - 1)]
        except Exception as exc:  # one corrupt clip should not erase the run
            LOGGER.exception("hidden failed for %s", row["sample_id"])
            errors.append({**row, "error": f"{type(exc).__name__}: {exc}"})
        if (index + 1) % 8 == 0:
            LOGGER.info("hidden stage: %d/%d", index + 1, len(rows))

    if not metadata:
        raise RuntimeError("No hidden-state rows were produced")
    torch.save(
        {
            "audio_encoder": torch.stack(audio_vectors),
            "projector": torch.stack(projector_vectors),
            "llm_audio_mean": torch.stack(llm_audio_vectors),
            "llm_decision": torch.stack(llm_decision_vectors),
            "llm_prompt_last": torch.stack(llm_prompt_vectors),
            "audio_layer_names": audio_layer_names,
            "llm_layer_names": llm_layer_names,
            "prompt": prompt,
            "model_checkpoint": str(args.checkpoint),
        },
        output_dir / "representations.pt",
    )
    with (output_dir / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*metadata[0].keys()])
        writer.writeheader()
        writer.writerows(metadata)
    with (output_dir / "errors.csv").open("w", newline="") as handle:
        fieldnames = [*rows[0].keys(), "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    LOGGER.info("saved %d hidden rows and %d errors", len(metadata), len(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["output", "forced-choice", "hidden"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ravdess-root", type=Path, required=True)
    parser.add_argument("--slam-llm-root", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--whisper-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="CSV path for output stage")
    parser.add_argument("--output-dir", type=Path, help="Directory for hidden-stage tensors")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--forced-choice-condition",
        action="append",
        choices=list(FORCED_CHOICE_CONDITIONS),
        help="Repeat to select forced-choice prompt/verbalizer conditions; default is the full 2x2 matrix",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0, help="Zero-based manifest row for chunked output runs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.stage in {"output", "forced-choice"} and args.output is None:
        parser.error(f"{args.stage} stage requires --output")
    if args.stage == "hidden" and args.output_dir is None:
        parser.error("hidden stage requires --output-dir")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.stage == "output":
        run_output(args)
    elif args.stage == "forced-choice":
        run_forced_choice(args)
    else:
        run_hidden(args)


if __name__ == "__main__":
    main()
