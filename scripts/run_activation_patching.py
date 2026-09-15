#!/usr/bin/env python3
"""Run frozen activation patching on matched RAVDESS happy/sad pairs.

The patch is applied to the post-output hidden state of a selected Qwen
decoder block. Audio-token patches replace the matched target audio-token
positions; decision-token patches replace the final "answer_t" position.
The donor and target use the same prompt and fixed-length audio layout.
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
    from scripts.slam_omni_diagnostics import (
        CODE_LAYER,
        DEFAULT_PROMPT,
        FORCED_CHOICE_CONDITIONS,
        SHIFT,
        _audio_inputs,
        _compose_llm_inputs,
        _embed_llm_input_ids,
        load_model,
        read_rows,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_activation_patching.py`` entry point.
    from slam_omni_diagnostics import (
        CODE_LAYER,
        DEFAULT_PROMPT,
        FORCED_CHOICE_CONDITIONS,
        SHIFT,
        _audio_inputs,
        _compose_llm_inputs,
        _embed_llm_input_ids,
        load_model,
        read_rows,
    )


LOGGER = logging.getLogger("run_activation_patching")
DEFAULT_LAYERS = (7, 14, 15, 17, 22, 23)
PATCH_KINDS = ("audio_tokens", "decision_token")
DEFAULT_SEED = 1234


def _llm_layers(model: Any) -> Any:
    """Return the decoder block list for the loaded causal LM."""
    candidates = (
        getattr(getattr(model, "llm", None), "model", None),
        getattr(getattr(getattr(model, "llm", None), "model", None), "model", None),
    )
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise RuntimeError("Could not locate Qwen decoder layers on model.llm")


def _hidden_tensor(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _replace_hidden(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def _decoder_forward(model: Any, inputs_embeds: Any, attention_mask: Any) -> Any:
    """Run only the decoder backbone when the Qwen module is available."""
    backbone = getattr(getattr(model, "llm", None), "model", None)
    if backbone is not None and hasattr(backbone, "layers"):
        return backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )


def _append_candidate_batch(
    torch: Any,
    model: Any,
    base_inputs_embeds: Any,
    candidate_ids_batch: list[list[int]],
) -> Any:
    """Append one same-length candidate sequence per batch row."""
    if not candidate_ids_batch or not candidate_ids_batch[0]:
        raise ValueError("Candidate verbalizer tokenizes to an empty sequence")
    candidate_length = len(candidate_ids_batch[0])
    if any(len(ids) != candidate_length for ids in candidate_ids_batch):
        raise ValueError("Batched activation patching requires same-length verbalizers")
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


def _patch_hook(
    donor: Any,
    patch_kind: str,
    audio_start: int,
    audio_length: int,
    decision_position: int,
):
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = _hidden_tensor(output)
        patched = hidden.clone()
        donor_cast = donor.to(device=hidden.device, dtype=hidden.dtype)
        if patch_kind == "audio_tokens":
            if donor_cast.shape[1] != audio_length:
                raise RuntimeError(
                    f"Donor audio length {donor_cast.shape[1]} does not match target {audio_length}"
                )
            patched[:, audio_start : audio_start + audio_length, :] = donor_cast
        elif patch_kind == "decision_token":
            patched[:, decision_position, :] = donor_cast
        else:
            raise ValueError(f"Unknown patch kind: {patch_kind}")
        return _replace_hidden(output, patched)

    return hook


def _score_candidate_pairs(
    torch: Any,
    model: Any,
    prefix_batch: Any,
    attention_mask: Any,
    happy_ids: list[int],
    sad_ids: list[int],
    prefix_length: int,
    layer_module: Any = None,
    donor: Any = None,
    patch_kind: str | None = None,
    audio_start: int | None = None,
    audio_length: int | None = None,
    decision_position: int | None = None,
    patch_specs: list[tuple[Any, Any, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return happy and sad sequence log-probabilities for each prefix."""
    batch_size = prefix_batch.shape[0]
    prefix_repeated = prefix_batch.repeat_interleave(2, dim=0)
    candidate_ids_batch = [ids for _ in range(batch_size) for ids in (happy_ids, sad_ids)]
    full_inputs_embeds = _append_candidate_batch(torch, model, prefix_repeated, candidate_ids_batch)
    full_attention_mask = torch.ones(
        (full_inputs_embeds.shape[0], full_inputs_embeds.shape[1]),
        dtype=attention_mask.dtype,
        device=full_inputs_embeds.device,
    )

    hook_specs: list[tuple[Any, Any, str]] = list(patch_specs or [])
    if layer_module is not None:
        if donor is None or patch_kind is None or audio_start is None or audio_length is None or decision_position is None:
            raise ValueError("A complete patch specification is required when a layer is selected")
        hook_specs.append((layer_module, donor, patch_kind))
    handles = []
    for hook_layer, hook_donor, hook_kind in hook_specs:
        if hook_donor.shape[0] != batch_size:
            raise ValueError(
                f"Patch donor batch {hook_donor.shape[0]} does not match prefix batch {batch_size}"
            )
        donor_repeated = hook_donor.repeat_interleave(2, dim=0)
        handles.append(
            hook_layer.register_forward_hook(
                _patch_hook(donor_repeated, hook_kind, audio_start, audio_length, decision_position)
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
            target_ids = torch.tensor(
                candidate_ids_batch,
                dtype=torch.long,
                device=logits.device,
            )
            row_indices = torch.arange(target_ids.shape[0], device=logits.device)[:, None]
            token_indices = torch.arange(candidate_length, device=logits.device)[None, :]
            selected = token_logprobs[row_indices, token_indices, target_ids]
            scores = selected.sum(dim=1).reshape(batch_size, 2).detach().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    return scores[:, 0], scores[:, 1]


def _prepare_prefixes(
    torch: Any,
    model: Any,
    whisper: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    ravdess_root: Path,
    prompt: str,
) -> tuple[list[Any], dict[str, int]]:
    """Encode every clip once and keep the multimodal prefix on CPU."""
    prefixes: list[Any] = []
    common_positions: dict[str, int] | None = None
    for index, row in enumerate(rows):
        batch, positions = _audio_inputs(
            torch,
            whisper,
            model,
            tokenizer,
            ravdess_root / row["path"],
            prompt,
        )
        with torch.inference_mode():
            encoder_outs = model.encoder.extract_variable_length_features(
                batch["audio_mel"].permute(0, 2, 1)
            )
            projected = model.encoder_projector(encoder_outs)
            base_inputs_embeds, _ = _compose_llm_inputs(
                torch,
                model,
                batch["input_ids"],
                batch["attention_mask"],
                encoder_outs,
                projected,
                positions["audio_start"],
                positions["audio_length"],
            )
        current = {
            "audio_start": positions["audio_start"],
            "audio_length": min(positions["audio_length"], projected.shape[1]),
            "prefix_length": int(base_inputs_embeds.shape[1]),
            "decision_position": int(base_inputs_embeds.shape[1] - 1),
        }
        if common_positions is None:
            common_positions = current
        elif current != common_positions:
            raise ValueError(f"Prefix layout changed for {row['sample_id']}: {current} != {common_positions}")
        prefixes.append(base_inputs_embeds[0].detach().cpu())
        if (index + 1) % 16 == 0:
            LOGGER.info("prepared prefixes: %d/%d", index + 1, len(rows))
    if common_positions is None:
        raise ValueError("No rows were provided")
    return prefixes, common_positions


def _capture_donor_chunk(
    torch: Any,
    model: Any,
    prefix_batch: Any,
    positions: dict[str, int],
    layer_indices: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    """Capture selected post-block states from a donor prefix batch."""
    layers = _llm_layers(model)
    captured: dict[int, dict[str, Any]] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = _hidden_tensor(output)
            captured[layer_index] = {
                "audio_tokens": hidden[
                    :,
                    positions["audio_start"] : positions["audio_start"] + positions["audio_length"],
                    :,
                ]
                .detach()
                .cpu(),
                "decision_token": hidden[:, positions["decision_position"], :].detach().cpu(),
            }

        return hook

    for layer_index in layer_indices:
        handles.append(layers[layer_index].register_forward_hook(make_hook(layer_index)))
    try:
        with torch.inference_mode():
            _decoder_forward(
                model,
                prefix_batch,
                torch.ones(
                    (prefix_batch.shape[0], prefix_batch.shape[1]),
                    dtype=torch.bool,
                    device=prefix_batch.device,
                ),
            )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layer_indices):
        raise RuntimeError(f"Missing captured layers: expected {layer_indices}, got {sorted(captured)}")
    return captured


def _capture_all_donors(
    torch: Any,
    model: Any,
    prefixes: list[Any],
    positions: dict[str, int],
    layer_indices: tuple[int, ...],
    device: Any,
    batch_size: int,
) -> dict[int, dict[str, Any]]:
    stores: dict[int, dict[str, list[Any]]] = {
        layer_index: {"audio_tokens": [], "decision_token": []} for layer_index in layer_indices
    }
    for start in range(0, len(prefixes), batch_size):
        prefix_batch = torch.stack(prefixes[start : start + batch_size]).to(device)
        captured = _capture_donor_chunk(torch, model, prefix_batch, positions, layer_indices)
        for layer_index in layer_indices:
            for kind in PATCH_KINDS:
                stores[layer_index][kind].append(captured[layer_index][kind])
        LOGGER.info("captured donor states: %d/%d", min(start + batch_size, len(prefixes)), len(prefixes))
    return {
        layer_index: {
            kind: torch.cat(values, dim=0) for kind, values in kind_store.items()
        }
        for layer_index, kind_store in stores.items()
    }


def _score_all_baselines(
    torch: Any,
    model: Any,
    prefixes: list[Any],
    positions: dict[str, int],
    happy_ids: list[int],
    sad_ids: list[int],
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    happy_scores = np.zeros(len(prefixes), dtype=np.float64)
    sad_scores = np.zeros(len(prefixes), dtype=np.float64)
    for start in range(0, len(prefixes), batch_size):
        indices = range(start, min(start + batch_size, len(prefixes)))
        prefix_batch = torch.stack([prefixes[index] for index in indices]).to(device)
        happy, sad = _score_candidate_pairs(
            torch,
            model,
            prefix_batch,
            torch.ones((prefix_batch.shape[0], prefix_batch.shape[1]), dtype=torch.bool, device=device),
            happy_ids,
            sad_ids,
            positions["prefix_length"],
        )
        end = start + len(happy)
        happy_scores[start:end] = happy
        sad_scores[start:end] = sad
        LOGGER.info("baseline likelihood: %d/%d", end, len(prefixes))
    return happy_scores, sad_scores


def _score_direction(
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
) -> tuple[dict[int, float], dict[int, float]]:
    if len(target_indices) != len(donor_indices):
        raise ValueError("Target and donor index lists differ in length")
    layers = _llm_layers(model)
    happy_scores: dict[int, float] = {}
    sad_scores: dict[int, float] = {}
    for start in range(0, len(target_indices), batch_size):
        target_chunk = target_indices[start : start + batch_size]
        donor_chunk = donor_indices[start : start + batch_size]
        prefix_batch = torch.stack([prefixes[index] for index in target_chunk]).to(device)
        donor_batch = donor_store[layer_index][patch_kind][donor_chunk].to(device)
        attention_mask = torch.ones(
            (prefix_batch.shape[0], prefix_batch.shape[1]),
            dtype=torch.bool,
            device=device,
        )
        happy, sad = _score_candidate_pairs(
            torch,
            model,
            prefix_batch,
            attention_mask,
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
            happy_scores[index] = float(happy[offset])
            sad_scores[index] = float(sad[offset])
        LOGGER.info(
            "patched %s layer_%d: %d/%d",
            patch_kind,
            layer_index,
            min(start + batch_size, len(target_indices)),
            len(target_indices),
        )
    return happy_scores, sad_scores


def _counterfactual_margin_effects(
    *,
    baseline_happy_happy_score: float,
    baseline_happy_sad_score: float,
    baseline_sad_happy_score: float,
    baseline_sad_sad_score: float,
    patched_happy_target_happy_score: float,
    patched_happy_target_sad_score: float,
    patched_sad_target_happy_score: float,
    patched_sad_target_sad_score: float,
) -> dict[str, float]:
    """Compute the two S-margin changes and the symmetric CE effect.

    Each margin uses both candidate likelihoods from the same target sample.
    The positive direction is donor-aligned: a happy target should move toward
    the sad donor, while a sad target should move toward the happy donor.
    """
    baseline_happy_margin = baseline_happy_happy_score - baseline_happy_sad_score
    baseline_sad_margin = baseline_sad_happy_score - baseline_sad_sad_score
    patched_happy_target_margin = patched_happy_target_happy_score - patched_happy_target_sad_score
    patched_sad_target_margin = patched_sad_target_happy_score - patched_sad_target_sad_score
    happy_effect = baseline_happy_margin - patched_happy_target_margin
    sad_effect = patched_sad_target_margin - baseline_sad_margin
    return {
        "baseline_happy_margin": baseline_happy_margin,
        "baseline_sad_margin": baseline_sad_margin,
        "patched_happy_target_margin": patched_happy_target_margin,
        "patched_sad_target_margin": patched_sad_target_margin,
        "happy_effect_toward_sad": happy_effect,
        "sad_effect_toward_happy": sad_effect,
        "counterfactual_effect": 0.5 * (happy_effect + sad_effect),
    }


def _self_patch_check(
    torch: Any,
    model: Any,
    prefixes: list[Any],
    positions: dict[str, int],
    donor_store: dict[int, dict[str, Any]],
    layer_indices: tuple[int, ...],
    patch_kinds: tuple[str, ...],
    happy_ids: list[int],
    sad_ids: list[int],
    baseline_happy: np.ndarray,
    baseline_sad: np.ndarray,
    device: Any,
) -> float:
    """Check that replacing a state with itself preserves the likelihood."""
    layers = _llm_layers(model)
    prefix_batch = torch.stack([prefixes[0]]).to(device)
    attention_mask = torch.ones((1, prefix_batch.shape[1]), dtype=torch.bool, device=device)
    max_error = 0.0
    for layer_index in layer_indices:
        for patch_kind in patch_kinds:
            happy, sad = _score_candidate_pairs(
                torch,
                model,
                prefix_batch,
                attention_mask,
                happy_ids,
                sad_ids,
                positions["prefix_length"],
                layer_module=layers[layer_index],
                donor=donor_store[layer_index][patch_kind][:1].to(device),
                patch_kind=patch_kind,
                audio_start=positions["audio_start"],
                audio_length=positions["audio_length"],
                decision_position=positions["decision_position"],
            )
            max_error = max(
                max_error,
                abs(float(happy[0]) - float(baseline_happy[0])),
                abs(float(sad[0]) - float(baseline_sad[0])),
            )
    return max_error


def _pair_indices(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row["pair_id"], {})[row["emotion"]] = index
    pairs = []
    for pair_id in sorted(grouped):
        emotions = grouped[pair_id]
        if set(emotions) != {"happy", "sad"}:
            raise ValueError(f"Pair {pair_id} does not contain exactly happy and sad rows")
        happy_index = emotions["happy"]
        sad_index = emotions["sad"]
        happy = rows[happy_index]
        sad = rows[sad_index]
        for key in ("actor", "statement", "repetition", "intensity"):
            if happy[key] != sad[key]:
                raise ValueError(f"Pair {pair_id} is not matched on {key}")
        pairs.append(
            {
                "pair_id": pair_id,
                "happy_index": happy_index,
                "sad_index": sad_index,
                "happy": happy,
                "sad": sad,
            }
        )
    return pairs


def run(args: argparse.Namespace) -> None:
    torch, model, tokenizer, whisper = load_model(args)
    rows = read_rows(args.manifest, args.limit)
    if args.start:
        rows = rows[args.start :]
    pairs = _pair_indices(rows)
    if not pairs:
        raise ValueError("No complete happy/sad pairs were found")
    layer_indices = tuple(sorted(set(args.layer)))
    layers = _llm_layers(model)
    invalid = [layer_index for layer_index in layer_indices if layer_index < 0 or layer_index >= len(layers)]
    if invalid:
        raise ValueError(f"Invalid layer indices {invalid}; model has {len(layers)} layers")
    patch_kinds = tuple(args.patch_kind)
    condition = FORCED_CHOICE_CONDITIONS[args.condition]
    happy_ids = [int(token_id) for token_id in tokenizer.encode(condition["happy_verbalizer"], add_special_tokens=False)]
    sad_ids = [int(token_id) for token_id in tokenizer.encode(condition["sad_verbalizer"], add_special_tokens=False)]
    if len(happy_ids) != len(sad_ids):
        raise ValueError(
            f"Selected verbalizers have different token lengths: happy={happy_ids}, sad={sad_ids}"
        )
    prefixes, positions = _prepare_prefixes(
        torch,
        model,
        whisper,
        tokenizer,
        rows,
        args.ravdess_root,
        condition["prompt"],
    )
    device = next(model.parameters()).device
    happy_baseline, sad_baseline = _score_all_baselines(
        torch,
        model,
        prefixes,
        positions,
        happy_ids,
        sad_ids,
        device,
        args.batch_size,
    )
    donor_store = _capture_all_donors(
        torch,
        model,
        prefixes,
        positions,
        layer_indices,
        device,
        args.batch_size,
    )
    self_patch_error = _self_patch_check(
        torch,
        model,
        prefixes,
        positions,
        donor_store,
        layer_indices,
        patch_kinds,
        happy_ids,
        sad_ids,
        happy_baseline,
        sad_baseline,
        device,
    )
    LOGGER.info("self-patch max absolute likelihood error: %.6g", self_patch_error)
    if self_patch_error > args.self_patch_tolerance:
        raise RuntimeError(
            f"Self-patch control changed likelihood by {self_patch_error:.6g}, "
            f"exceeding tolerance {args.self_patch_tolerance:.6g}"
        )
    output_path = args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "actor",
        "statement",
        "repetition",
        "intensity",
        "happy_sample_id",
        "sad_sample_id",
        "condition_id",
        "prompt",
        "happy_verbalizer",
        "sad_verbalizer",
        "happy_token_ids",
        "sad_token_ids",
        "layer_index",
        "layer",
        "patch_site",
        "patch_kind",
        "baseline_happy_happy_score",
        "baseline_happy_sad_score",
        "baseline_sad_happy_score",
        "baseline_sad_sad_score",
        "baseline_happy_margin",
        "baseline_sad_margin",
        "patched_happy_target_happy_score",
        "patched_happy_target_sad_score",
        "patched_sad_target_happy_score",
        "patched_sad_target_sad_score",
        "patched_happy_target_margin",
        "patched_sad_target_margin",
        "happy_effect_toward_sad",
        "sad_effect_toward_happy",
        "counterfactual_effect",
        "happy_direction",
        "sad_direction",
        "error",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for layer_index in layer_indices:
            for patch_kind in patch_kinds:
                sad_targets = [pair["sad_index"] for pair in pairs]
                happy_donors = [pair["happy_index"] for pair in pairs]
                happy_targets = [pair["happy_index"] for pair in pairs]
                sad_donors = [pair["sad_index"] for pair in pairs]
                patched_sad_happy_scores, patched_sad_sad_scores = _score_direction(
                    torch,
                    model,
                    prefixes,
                    positions,
                    sad_targets,
                    happy_donors,
                    donor_store,
                    layer_index,
                    patch_kind,
                    happy_ids,
                    sad_ids,
                    device,
                    args.batch_size,
                )
                patched_happy_happy_scores, patched_happy_sad_scores = _score_direction(
                    torch,
                    model,
                    prefixes,
                    positions,
                    happy_targets,
                    sad_donors,
                    donor_store,
                    layer_index,
                    patch_kind,
                    happy_ids,
                    sad_ids,
                    device,
                    args.batch_size,
                )
                for pair in pairs:
                    happy_index = pair["happy_index"]
                    sad_index = pair["sad_index"]
                    effects = _counterfactual_margin_effects(
                        baseline_happy_happy_score=float(happy_baseline[happy_index]),
                        baseline_happy_sad_score=float(sad_baseline[happy_index]),
                        baseline_sad_happy_score=float(happy_baseline[sad_index]),
                        baseline_sad_sad_score=float(sad_baseline[sad_index]),
                        patched_happy_target_happy_score=patched_happy_happy_scores[happy_index],
                        patched_happy_target_sad_score=patched_happy_sad_scores[happy_index],
                        patched_sad_target_happy_score=patched_sad_happy_scores[sad_index],
                        patched_sad_target_sad_score=patched_sad_sad_scores[sad_index],
                    )
                    writer.writerow(
                        {
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
                            "layer_index": layer_index,
                            "layer": f"layer_{layer_index}",
                            "patch_site": "post_decoder_block_output",
                            "patch_kind": patch_kind,
                            "baseline_happy_happy_score": float(happy_baseline[happy_index]),
                            "baseline_happy_sad_score": float(sad_baseline[happy_index]),
                            "baseline_sad_happy_score": float(happy_baseline[sad_index]),
                            "baseline_sad_sad_score": float(sad_baseline[sad_index]),
                            "baseline_happy_margin": effects["baseline_happy_margin"],
                            "baseline_sad_margin": effects["baseline_sad_margin"],
                            "patched_happy_target_happy_score": patched_happy_happy_scores[happy_index],
                            "patched_happy_target_sad_score": patched_happy_sad_scores[happy_index],
                            "patched_sad_target_happy_score": patched_sad_happy_scores[sad_index],
                            "patched_sad_target_sad_score": patched_sad_sad_scores[sad_index],
                            "patched_happy_target_margin": effects["patched_happy_target_margin"],
                            "patched_sad_target_margin": effects["patched_sad_target_margin"],
                            "happy_effect_toward_sad": effects["happy_effect_toward_sad"],
                            "sad_effect_toward_happy": effects["sad_effect_toward_happy"],
                            "counterfactual_effect": effects["counterfactual_effect"],
                            "happy_direction": effects["happy_effect_toward_sad"] > 0,
                            "sad_direction": effects["sad_effect_toward_happy"] > 0,
                            "error": "",
                        }
                    )
                handle.flush()
                LOGGER.info("finished patch condition: layer_%d/%s", layer_index, patch_kind)
    run_metadata = {
        "manifest": str(args.manifest),
        "n_samples": len(rows),
        "n_pairs": len(pairs),
        "condition_id": args.condition,
        "happy_token_ids": happy_ids,
        "sad_token_ids": sad_ids,
        "layer_indices": list(layer_indices),
        "patch_kinds": list(patch_kinds),
        "patch_site": "post_decoder_block_output",
        "audio_start": positions["audio_start"],
        "audio_length": positions["audio_length"],
        "prefix_length": positions["prefix_length"],
        "checkpoint": str(args.checkpoint),
        "model_parameters_frozen": True,
        "self_patch_max_abs_likelihood_error": self_patch_error,
        "self_patch_tolerance": args.self_patch_tolerance,
        "seed": args.seed,
    }
    output_path.with_name(output_path.stem + "_run.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n"
    )
    LOGGER.info("saved activation patch rows to %s", output_path)


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
    parser.add_argument("--layer", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--patch-kind", nargs="+", choices=list(PATCH_KINDS), default=list(PATCH_KINDS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--self-patch-tolerance", type=float, default=1e-3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.self_patch_tolerance < 0:
        raise SystemExit("--self-patch-tolerance must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
