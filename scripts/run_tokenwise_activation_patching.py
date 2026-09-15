#!/usr/bin/env python3
"""Decompose two-token activation-patching effects by candidate position.

This runner mirrors :mod:`run_activation_patching`, but keeps the
teacher-forced log-probability for each verbalizer token instead of only the
sequence sum.  The resulting per-token CE values add up to the original
sequence-level CE, which separates first-token decision evidence from later
candidate dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.run_activation_patching import (
        DEFAULT_SEED,
        PATCH_KINDS,
        _capture_all_donors,
        _decoder_forward,
        _llm_layers,
        _pair_indices,
        _patch_hook,
        _prepare_prefixes,
        load_model,
        read_rows,
    )
    from scripts.slam_omni_diagnostics import CODE_LAYER, FORCED_CHOICE_CONDITIONS, SHIFT, _embed_llm_input_ids
except ModuleNotFoundError:  # Direct ``python scripts/run_tokenwise_activation_patching.py`` entry point.
    from run_activation_patching import (
        DEFAULT_SEED,
        PATCH_KINDS,
        _capture_all_donors,
        _decoder_forward,
        _llm_layers,
        _pair_indices,
        _patch_hook,
        _prepare_prefixes,
        load_model,
        read_rows,
    )
    from slam_omni_diagnostics import CODE_LAYER, FORCED_CHOICE_CONDITIONS, SHIFT, _embed_llm_input_ids


LOGGER = logging.getLogger("run_tokenwise_activation_patching")


def _append_candidate_batch(torch: Any, model: Any, base_inputs_embeds: Any, candidate_ids_batch: list[list[int]]) -> Any:
    if not candidate_ids_batch or not candidate_ids_batch[0]:
        raise ValueError("Candidate verbalizer tokenizes to an empty sequence")
    candidate_length = len(candidate_ids_batch[0])
    if any(len(ids) != candidate_length for ids in candidate_ids_batch):
        raise ValueError("Batched tokenwise patching requires same-length verbalizers")
    vocab = model.model_config.vocab_config
    batch_size = base_inputs_embeds.shape[0]
    if batch_size != len(candidate_ids_batch):
        raise ValueError("Candidate batch and prefix batch have different sizes")
    candidate_streams = torch.full(
        (batch_size, CODE_LAYER + 1, candidate_length),
        SHIFT + int(vocab.pad_a),
        dtype=torch.long,
        device=base_inputs_embeds.device,
    )
    candidate_streams[:, CODE_LAYER, :] = torch.tensor(
        candidate_ids_batch,
        dtype=torch.long,
        device=base_inputs_embeds.device,
    )
    candidate_embeds = _embed_llm_input_ids(model, candidate_streams).mean(dim=1)
    return torch.cat((base_inputs_embeds, candidate_embeds), dim=1)


def _score_candidate_pairs_tokenwise(
    torch: Any,
    model: Any,
    prefix_batch: Any,
    attention_mask: Any,
    happy_ids: list[int],
    sad_ids: list[int],
    prefix_length: int,
    *,
    layer_module: Any = None,
    donor: Any = None,
    patch_kind: str | None = None,
    audio_start: int | None = None,
    audio_length: int | None = None,
    decision_position: int | None = None,
) -> np.ndarray:
    """Return [batch, candidate, token] teacher-forced log-probabilities."""
    batch_size = prefix_batch.shape[0]
    prefix_repeated = prefix_batch.repeat_interleave(2, dim=0)
    candidate_ids_batch = [ids for _ in range(batch_size) for ids in (happy_ids, sad_ids)]
    full_inputs_embeds = _append_candidate_batch(torch, model, prefix_repeated, candidate_ids_batch)
    full_attention_mask = torch.ones(
        (full_inputs_embeds.shape[0], full_inputs_embeds.shape[1]),
        dtype=attention_mask.dtype,
        device=full_inputs_embeds.device,
    )
    handles = []
    if layer_module is not None:
        if donor is None or patch_kind is None or audio_start is None or audio_length is None or decision_position is None:
            raise ValueError("A complete patch specification is required when a layer is selected")
        if donor.shape[0] != batch_size:
            raise ValueError(f"Patch donor batch {donor.shape[0]} does not match prefix batch {batch_size}")
        donor_repeated = donor.repeat_interleave(2, dim=0)
        handles.append(
            layer_module.register_forward_hook(
                _patch_hook(donor_repeated, patch_kind, audio_start, audio_length, decision_position)
            )
        )
    try:
        with torch.inference_mode():
            text_vocab_size = int(model.model_config.vocab_config.padded_text_vocabsize)
            candidate_length = len(happy_ids)
            positions = torch.arange(candidate_length, device=full_inputs_embeds.device) + prefix_length - 1
            outputs = _decoder_forward(model, full_inputs_embeds, full_attention_mask)
            if hasattr(outputs, "last_hidden_state"):
                selected_hidden = outputs.last_hidden_state[:, positions, :]
                logits = model.llm.lm_head(selected_hidden)[..., :text_vocab_size]
            else:
                logits = outputs.logits[:, positions, :text_vocab_size]
            token_logprobs = torch.log_softmax(logits, dim=-1)
            target_ids = torch.tensor(candidate_ids_batch, dtype=torch.long, device=logits.device)
            row_indices = torch.arange(target_ids.shape[0], device=logits.device)[:, None]
            token_indices = torch.arange(candidate_length, device=logits.device)[None, :]
            selected = token_logprobs[row_indices, token_indices, target_ids]
            return selected.reshape(batch_size, 2, candidate_length).detach().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()


def _score_all_baselines_tokenwise(
    torch: Any,
    model: Any,
    prefixes: list[Any],
    positions: dict[str, int],
    happy_ids: list[int],
    sad_ids: list[int],
    device: Any,
    batch_size: int,
) -> np.ndarray:
    values = np.zeros((len(prefixes), 2, len(happy_ids)), dtype=np.float64)
    for start in range(0, len(prefixes), batch_size):
        indices = range(start, min(start + batch_size, len(prefixes)))
        prefix_batch = torch.stack([prefixes[index] for index in indices]).to(device)
        scores = _score_candidate_pairs_tokenwise(
            torch,
            model,
            prefix_batch,
            torch.ones((prefix_batch.shape[0], prefix_batch.shape[1]), dtype=torch.bool, device=device),
            happy_ids,
            sad_ids,
            positions["prefix_length"],
        )
        values[start : start + len(scores)] = scores
        LOGGER.info("tokenwise baseline likelihood: %d/%d", start + len(scores), len(prefixes))
    return values


def _score_direction_tokenwise(
    torch: Any,
    model: Any,
    prefixes: list[Any],
    positions: dict[str, int],
    target_indices: list[int],
    donor_indices: list[int],
    donor_store: dict[int, dict[str, Any]],
    layer_index: int,
    patch_kind: str,
    happy_ids: list[int],
    sad_ids: list[int],
    device: Any,
    batch_size: int,
) -> dict[int, np.ndarray]:
    if len(target_indices) != len(donor_indices):
        raise ValueError("Target and donor index lists differ in length")
    layers = _llm_layers(model)
    scores: dict[int, np.ndarray] = {}
    for start in range(0, len(target_indices), batch_size):
        target_chunk = target_indices[start : start + batch_size]
        donor_chunk = donor_indices[start : start + batch_size]
        prefix_batch = torch.stack([prefixes[index] for index in target_chunk]).to(device)
        donor_batch = donor_store[layer_index][patch_kind][donor_chunk].to(device)
        token_scores = _score_candidate_pairs_tokenwise(
            torch,
            model,
            prefix_batch,
            torch.ones((prefix_batch.shape[0], prefix_batch.shape[1]), dtype=torch.bool, device=device),
            happy_ids,
            sad_ids,
            positions["prefix_length"],
            layer_module=layers[layer_index],
            donor=donor_batch,
            patch_kind=patch_kind,
            audio_start=positions["audio_start"],
            audio_length=positions["audio_length"],
            decision_position=positions["decision_position"],
        )
        for offset, index in enumerate(target_chunk):
            scores[index] = token_scores[offset]
        LOGGER.info(
            "tokenwise patched %s layer_%d: %d/%d",
            patch_kind,
            layer_index,
            min(start + batch_size, len(target_indices)),
            len(target_indices),
        )
    return scores


def _tokenwise_effects(
    baseline_happy: np.ndarray,
    baseline_sad: np.ndarray,
    patched_happy: np.ndarray,
    patched_sad: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-token margins and CE; CE sums to sequence-level CE."""
    baseline_happy_margin = baseline_happy[0] - baseline_happy[1]
    baseline_sad_margin = baseline_sad[0] - baseline_sad[1]
    patched_happy_margin = patched_happy[0] - patched_happy[1]
    patched_sad_margin = patched_sad[0] - patched_sad[1]
    happy_effect = baseline_happy_margin - patched_happy_margin
    sad_effect = patched_sad_margin - baseline_sad_margin
    return {
        "baseline_happy_margin": baseline_happy_margin,
        "baseline_sad_margin": baseline_sad_margin,
        "patched_happy_target_margin": patched_happy_margin,
        "patched_sad_target_margin": patched_sad_margin,
        "happy_effect_toward_sad": happy_effect,
        "sad_effect_toward_happy": sad_effect,
        "counterfactual_effect": 0.5 * (happy_effect + sad_effect),
    }


def run(args: argparse.Namespace) -> None:
    torch, model, tokenizer, whisper = load_model(args)
    rows = read_rows(args.manifest, args.limit)
    if args.start:
        rows = rows[args.start :]
    pairs = _pair_indices(rows)
    if not pairs:
        raise ValueError("No complete happy/sad pairs were found")
    condition = FORCED_CHOICE_CONDITIONS[args.condition]
    happy_ids = [int(token_id) for token_id in tokenizer.encode(condition["happy_verbalizer"], add_special_tokens=False)]
    sad_ids = [int(token_id) for token_id in tokenizer.encode(condition["sad_verbalizer"], add_special_tokens=False)]
    if len(happy_ids) != len(sad_ids) or len(happy_ids) < 2:
        raise ValueError(f"Tokenwise decomposition requires equal verbalizers with at least two tokens: happy={happy_ids}, sad={sad_ids}")
    prefixes, positions = _prepare_prefixes(
        torch, model, whisper, tokenizer, rows, args.ravdess_root, condition["prompt"]
    )
    device = next(model.parameters()).device
    baseline = _score_all_baselines_tokenwise(
        torch, model, prefixes, positions, happy_ids, sad_ids, device, args.batch_size
    )
    layer_indices = tuple(sorted(set(args.layer)))
    layers = _llm_layers(model)
    invalid = [index for index in layer_indices if index < 0 or index >= len(layers)]
    if invalid:
        raise ValueError(f"Invalid layer indices {invalid}; model has {len(layers)} layers")
    donor_store = _capture_all_donors(
        torch, model, prefixes, positions, layer_indices, device, args.batch_size
    )
    output_path = args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id", "actor", "statement", "repetition", "intensity", "happy_sample_id", "sad_sample_id",
        "condition_id", "prompt", "happy_verbalizer", "sad_verbalizer", "happy_token_ids", "sad_token_ids",
        "token_position", "layer_index", "layer", "patch_site", "patch_kind",
        "baseline_happy_happy_token_logprob", "baseline_happy_sad_token_logprob",
        "baseline_sad_happy_token_logprob", "baseline_sad_sad_token_logprob",
        "baseline_happy_margin", "baseline_sad_margin",
        "patched_happy_target_happy_token_logprob", "patched_happy_target_sad_token_logprob",
        "patched_sad_target_happy_token_logprob", "patched_sad_target_sad_token_logprob",
        "patched_happy_target_margin", "patched_sad_target_margin",
        "happy_effect_toward_sad", "sad_effect_toward_happy", "counterfactual_effect",
        "happy_direction", "sad_direction", "error",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for layer_index in layer_indices:
            for patch_kind in args.patch_kind:
                sad_targets = [pair["sad_index"] for pair in pairs]
                happy_donors = [pair["happy_index"] for pair in pairs]
                happy_targets = [pair["happy_index"] for pair in pairs]
                sad_donors = [pair["sad_index"] for pair in pairs]
                patched_sad = _score_direction_tokenwise(
                    torch, model, prefixes, positions, sad_targets, happy_donors, donor_store,
                    layer_index, patch_kind, happy_ids, sad_ids, device, args.batch_size
                )
                patched_happy = _score_direction_tokenwise(
                    torch, model, prefixes, positions, happy_targets, sad_donors, donor_store,
                    layer_index, patch_kind, happy_ids, sad_ids, device, args.batch_size
                )
                for pair in pairs:
                    happy_index = pair["happy_index"]
                    sad_index = pair["sad_index"]
                    effects = _tokenwise_effects(
                        baseline[happy_index], baseline[sad_index],
                        patched_happy[happy_index], patched_sad[sad_index]
                    )
                    for token_offset in range(len(happy_ids)):
                        row = {
                            "pair_id": pair["pair_id"],
                            "actor": pair["happy"]["actor"],
                            "statement": pair["happy"]["statement"],
                            "repetition": pair["happy"]["repetition"],
                            "intensity": pair["happy"]["intensity"],
                            "happy_sample_id": pair["happy"]["sample_id"],
                            "sad_sample_id": pair["sad"]["sample_id"],
                            "condition_id": args.condition,
                            "prompt": condition["prompt"],
                            "happy_verbalizer": condition["happy_verbalizer"],
                            "sad_verbalizer": condition["sad_verbalizer"],
                            "happy_token_ids": json.dumps(happy_ids),
                            "sad_token_ids": json.dumps(sad_ids),
                            "token_position": token_offset + 1,
                            "layer_index": layer_index,
                            "layer": f"layer_{layer_index}",
                            "patch_site": "post_decoder_block_output",
                            "patch_kind": patch_kind,
                            "baseline_happy_happy_token_logprob": baseline[happy_index, 0, token_offset],
                            "baseline_happy_sad_token_logprob": baseline[happy_index, 1, token_offset],
                            "baseline_sad_happy_token_logprob": baseline[sad_index, 0, token_offset],
                            "baseline_sad_sad_token_logprob": baseline[sad_index, 1, token_offset],
                            "baseline_happy_margin": effects["baseline_happy_margin"][token_offset],
                            "baseline_sad_margin": effects["baseline_sad_margin"][token_offset],
                            "patched_happy_target_happy_token_logprob": patched_happy[happy_index][0, token_offset],
                            "patched_happy_target_sad_token_logprob": patched_happy[happy_index][1, token_offset],
                            "patched_sad_target_happy_token_logprob": patched_sad[sad_index][0, token_offset],
                            "patched_sad_target_sad_token_logprob": patched_sad[sad_index][1, token_offset],
                            "patched_happy_target_margin": effects["patched_happy_target_margin"][token_offset],
                            "patched_sad_target_margin": effects["patched_sad_target_margin"][token_offset],
                            "happy_effect_toward_sad": effects["happy_effect_toward_sad"][token_offset],
                            "sad_effect_toward_happy": effects["sad_effect_toward_happy"][token_offset],
                            "counterfactual_effect": effects["counterfactual_effect"][token_offset],
                            "happy_direction": effects["happy_effect_toward_sad"][token_offset] > 0,
                            "sad_direction": effects["sad_effect_toward_happy"][token_offset] > 0,
                            "error": "",
                        }
                        writer.writerow(row)
                handle.flush()
                LOGGER.info("finished tokenwise patch condition: layer_%d/%s", layer_index, patch_kind)
    metadata = {
        "manifest": str(args.manifest),
        "n_samples": len(rows),
        "n_pairs": len(pairs),
        "condition_id": args.condition,
        "happy_token_ids": happy_ids,
        "sad_token_ids": sad_ids,
        "layer_indices": list(layer_indices),
        "patch_kinds": list(args.patch_kind),
        "tokenwise": True,
        "patch_site": "post_decoder_block_output",
        "audio_start": positions["audio_start"],
        "audio_length": positions["audio_length"],
        "prefix_length": positions["prefix_length"],
        "checkpoint": str(args.checkpoint),
        "model_parameters_frozen": True,
        "sequence_ce_reconstruction": "sum(token_position counterfactual_effect) equals sequence-level CE up to floating-point error",
        "seed": args.seed,
    }
    output_path.with_name(output_path.stem + "_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    LOGGER.info("saved tokenwise activation patch rows to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ravdess-root", type=Path, required=True)
    parser.add_argument("--slam-llm-root", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--whisper-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--condition", choices=list(FORCED_CHOICE_CONDITIONS), default="upper_prompt__upper_spaced")
    parser.add_argument("--layer", nargs="+", type=int, default=[17])
    parser.add_argument("--patch-kind", nargs="+", choices=list(PATCH_KINDS), default=list(PATCH_KINDS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
