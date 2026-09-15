#!/usr/bin/env python3
"""Causally patch selected attention heads at the decision position.

The preceding attention/value trace supplies candidate heads.  For each
selected ``layer:head`` this runner replaces only that head's pre-``o_proj``
attention output at the decision position with the matched donor state, then
measures the single-token forced-choice margin.  This is a targeted causal
follow-up, not a scan over every possible path.
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
        _append_candidate_batch,
        _counterfactual_margin_effects,
        _decoder_forward,
        _llm_layers,
        _pair_indices,
        _prepare_prefixes,
        _score_all_baselines,
    )
    from scripts.slam_omni_diagnostics import FORCED_CHOICE_CONDITIONS, load_model, read_rows
except ModuleNotFoundError:  # Direct ``python scripts/run_attention_head_patching.py`` entry point.
    from run_activation_patching import (
        DEFAULT_SEED,
        _append_candidate_batch,
        _counterfactual_margin_effects,
        _decoder_forward,
        _llm_layers,
        _pair_indices,
        _prepare_prefixes,
        _score_all_baselines,
    )
    from slam_omni_diagnostics import FORCED_CHOICE_CONDITIONS, load_model, read_rows


LOGGER = logging.getLogger("run_attention_head_patching")


def _parse_heads(values: list[str]) -> list[tuple[int, int]]:
    heads = []
    for value in values:
        try:
            layer, head = (int(part) for part in value.split(":", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Head must use layer:head syntax, got {value!r}") from exc
        heads.append((layer, head))
    if len(set(heads)) != len(heads):
        raise ValueError("Duplicate layer:head specification")
    return heads


def _capture_o_proj_inputs(
    torch: Any,
    model: Any,
    layers: Any,
    prefixes: list[Any],
    positions: dict[str, int],
    happy_id: int,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    """Capture each sample's pre-o_proj attention vectors [N,L,D]."""
    stores: list[np.ndarray] = []
    for start in range(0, len(prefixes), batch_size):
        prefix_batch = torch.stack(prefixes[start : start + batch_size]).to(device)
        full_inputs = _append_candidate_batch(torch, model, prefix_batch, [[happy_id]] * prefix_batch.shape[0])
        attention_mask = torch.ones((full_inputs.shape[0], full_inputs.shape[1]), dtype=torch.bool, device=device)
        captured: dict[int, Any] = {}
        handles = []
        for layer_index, layer in enumerate(layers):
            def make_hook(index: int):
                def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
                    if not inputs:
                        raise RuntimeError("o_proj pre-hook received no input")
                    captured[index] = inputs[0][:, positions["decision_position"], :].detach().cpu()
                    return None

                return hook

            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(layer_index)))
        try:
            with torch.inference_mode():
                _decoder_forward(model, full_inputs, attention_mask)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(range(len(layers))):
            raise RuntimeError("Missing o_proj captures for one or more decoder layers")
        stores.append(torch.stack([captured[index] for index in range(len(layers))], dim=1).numpy())
        LOGGER.info("captured o_proj inputs: %d/%d", min(start + batch_size, len(prefixes)), len(prefixes))
    return np.concatenate(stores, axis=0)


def _score_candidate_pairs_head(
    torch: Any,
    model: Any,
    prefix_batch: Any,
    happy_ids: list[int],
    sad_ids: list[int],
    prefix_length: int,
    *,
    layer: Any,
    donor_o_proj: Any,
    head_index: int,
    decision_position: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = prefix_batch.shape[0]
    prefix_repeated = prefix_batch.repeat_interleave(2, dim=0)
    candidate_ids_batch = [ids for _ in range(batch_size) for ids in (happy_ids, sad_ids)]
    full_inputs = _append_candidate_batch(torch, model, prefix_repeated, candidate_ids_batch)
    attention_mask = torch.ones((full_inputs.shape[0], full_inputs.shape[1]), dtype=torch.bool, device=full_inputs.device)
    attention = layer.self_attn
    head_dim = int(getattr(attention, "head_dim", attention.config.hidden_size // attention.config.num_attention_heads))
    start = head_index * head_dim
    end = start + head_dim
    donor_repeated = donor_o_proj.repeat_interleave(2, dim=0)

    def patch_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs:
            raise RuntimeError("o_proj pre-hook received no input")
        hidden = inputs[0]
        patched = hidden.clone()
        patched[:, decision_position, start:end] = donor_repeated[:, start:end].to(
            device=hidden.device, dtype=hidden.dtype
        )
        return (patched, *inputs[1:])

    handle = attention.o_proj.register_forward_pre_hook(patch_hook)
    try:
        with torch.inference_mode():
            text_vocab_size = int(model.model_config.vocab_config.padded_text_vocabsize)
            outputs = _decoder_forward(model, full_inputs, attention_mask)
            if hasattr(outputs, "last_hidden_state"):
                logits = model.llm.lm_head(outputs.last_hidden_state[:, prefix_length - 1, :])[:, :text_vocab_size]
            else:
                logits = outputs.logits[:, prefix_length - 1, :text_vocab_size]
            logprobs = torch.log_softmax(logits, dim=-1)
            target_ids = torch.tensor(candidate_ids_batch, dtype=torch.long, device=logprobs.device)
            selected = logprobs[torch.arange(target_ids.shape[0], device=logprobs.device), target_ids[:, 0]]
            scores = selected.reshape(batch_size, 2).detach().cpu().numpy()
    finally:
        handle.remove()
    return scores[:, 0], scores[:, 1]


def _score_direction_head(
    torch: Any,
    model: Any,
    prefixes: list[Any],
    positions: dict[str, int],
    target_indices: list[int],
    donor_indices: list[int],
    donor_o_proj: np.ndarray,
    layer_index: int,
    head_index: int,
    happy_ids: list[int],
    sad_ids: list[int],
    device: Any,
    batch_size: int,
) -> tuple[dict[int, float], dict[int, float]]:
    layers = _llm_layers(model)
    output_happy: dict[int, float] = {}
    output_sad: dict[int, float] = {}
    for start in range(0, len(target_indices), batch_size):
        target_chunk = target_indices[start : start + batch_size]
        donor_chunk = donor_indices[start : start + batch_size]
        prefix_batch = torch.stack([prefixes[index] for index in target_chunk]).to(device)
        donor_batch = torch.as_tensor(
            donor_o_proj[donor_chunk, layer_index], dtype=prefix_batch.dtype, device=device
        )
        happy, sad = _score_candidate_pairs_head(
            torch,
            model,
            prefix_batch,
            happy_ids,
            sad_ids,
            positions["prefix_length"],
            layer=layers[layer_index],
            donor_o_proj=donor_batch,
            head_index=head_index,
            decision_position=positions["decision_position"],
        )
        for offset, index in enumerate(target_chunk):
            output_happy[index] = float(happy[offset])
            output_sad[index] = float(sad[offset])
        LOGGER.info(
            "patched attention head L%dH%d: %d/%d",
            layer_index,
            head_index,
            min(start + batch_size, len(target_indices)),
            len(target_indices),
        )
    return output_happy, output_sad


def _self_patch_error(
    torch: Any,
    model: Any,
    prefixes: list[Any],
    positions: dict[str, int],
    donor_o_proj: np.ndarray,
    layer_index: int,
    head_index: int,
    happy_ids: list[int],
    sad_ids: list[int],
    baseline_happy: np.ndarray,
    baseline_sad: np.ndarray,
    device: Any,
) -> float:
    layers = _llm_layers(model)
    prefix_batch = torch.stack([prefixes[0]]).to(device)
    donor = torch.as_tensor(donor_o_proj[:1, layer_index], dtype=prefix_batch.dtype, device=device)
    happy, sad = _score_candidate_pairs_head(
        torch,
        model,
        prefix_batch,
        happy_ids,
        sad_ids,
        positions["prefix_length"],
        layer=layers[layer_index],
        donor_o_proj=donor,
        head_index=head_index,
        decision_position=positions["decision_position"],
    )
    return max(
        abs(float(happy[0]) - float(baseline_happy[0])),
        abs(float(sad[0]) - float(baseline_sad[0])),
    )


def run(args: argparse.Namespace) -> None:
    torch, model, tokenizer, whisper = load_model(args)
    rows = read_rows(args.manifest, args.limit)
    if args.start:
        rows = rows[args.start :]
    pairs = _pair_indices(rows)
    if not pairs:
        raise ValueError("No complete happy/sad pairs were found")
    heads = _parse_heads(args.head)
    condition = FORCED_CHOICE_CONDITIONS[args.condition]
    happy_ids = [int(token_id) for token_id in tokenizer.encode(condition["happy_verbalizer"], add_special_tokens=False)]
    sad_ids = [int(token_id) for token_id in tokenizer.encode(condition["sad_verbalizer"], add_special_tokens=False)]
    if len(happy_ids) != 1 or len(sad_ids) != 1:
        raise ValueError(f"Head patching requires single-token verbalizers: happy={happy_ids}, sad={sad_ids}")
    prefixes, positions = _prepare_prefixes(
        torch, model, whisper, tokenizer, rows, args.ravdess_root, condition["prompt"]
    )
    layers = _llm_layers(model)
    for layer_index, head_index in heads:
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"Invalid layer index {layer_index}; model has {len(layers)} layers")
        n_heads = int(getattr(layers[layer_index].self_attn, "num_heads", layers[layer_index].self_attn.config.num_attention_heads))
        if head_index < 0 or head_index >= n_heads:
            raise ValueError(f"Invalid head index {head_index} for layer {layer_index}; model has {n_heads} heads")
    device = next(model.parameters()).device
    happy_baseline, sad_baseline = _score_all_baselines(
        torch, model, prefixes, positions, happy_ids, sad_ids, device, args.batch_size
    )
    donor_o_proj = _capture_o_proj_inputs(
        torch, model, layers, prefixes, positions, happy_ids[0], device, args.batch_size
    )
    output_path = args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id", "actor", "statement", "repetition", "intensity", "happy_sample_id", "sad_sample_id",
        "condition_id", "prompt", "happy_verbalizer", "sad_verbalizer", "happy_token_ids", "sad_token_ids",
        "layer_index", "head_index", "layer", "patch_site", "patch_kind",
        "baseline_happy_happy_score", "baseline_happy_sad_score", "baseline_sad_happy_score", "baseline_sad_sad_score",
        "baseline_happy_margin", "baseline_sad_margin", "patched_happy_target_happy_score", "patched_happy_target_sad_score",
        "patched_sad_target_happy_score", "patched_sad_target_sad_score", "patched_happy_target_margin", "patched_sad_target_margin",
        "happy_effect_toward_sad", "sad_effect_toward_happy", "counterfactual_effect", "happy_direction", "sad_direction", "error",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for layer_index, head_index in heads:
            self_error = _self_patch_error(
                torch, model, prefixes, positions, donor_o_proj, layer_index, head_index,
                happy_ids, sad_ids, happy_baseline, sad_baseline, device
            )
            if self_error > args.self_patch_tolerance:
                raise RuntimeError(f"Self-patch L{layer_index}H{head_index} changed likelihood by {self_error:.6g}")
            sad_targets = [pair["sad_index"] for pair in pairs]
            happy_donors = [pair["happy_index"] for pair in pairs]
            happy_targets = [pair["happy_index"] for pair in pairs]
            sad_donors = [pair["sad_index"] for pair in pairs]
            patched_sad_happy, patched_sad_sad = _score_direction_head(
                torch, model, prefixes, positions, sad_targets, happy_donors, donor_o_proj,
                layer_index, head_index, happy_ids, sad_ids, device, args.batch_size
            )
            patched_happy_happy, patched_happy_sad = _score_direction_head(
                torch, model, prefixes, positions, happy_targets, sad_donors, donor_o_proj,
                layer_index, head_index, happy_ids, sad_ids, device, args.batch_size
            )
            for pair in pairs:
                happy_index, sad_index = pair["happy_index"], pair["sad_index"]
                effects = _counterfactual_margin_effects(
                    baseline_happy_happy_score=float(happy_baseline[happy_index]),
                    baseline_happy_sad_score=float(sad_baseline[happy_index]),
                    baseline_sad_happy_score=float(happy_baseline[sad_index]),
                    baseline_sad_sad_score=float(sad_baseline[sad_index]),
                    patched_happy_target_happy_score=patched_happy_happy[happy_index],
                    patched_happy_target_sad_score=patched_happy_sad[happy_index],
                    patched_sad_target_happy_score=patched_sad_happy[sad_index],
                    patched_sad_target_sad_score=patched_sad_sad[sad_index],
                )
                writer.writerow(
                    {
                        "pair_id": pair["pair_id"], "actor": pair["happy"]["actor"], "statement": pair["happy"]["statement"],
                        "repetition": pair["happy"]["repetition"], "intensity": pair["happy"]["intensity"],
                        "happy_sample_id": pair["happy"]["sample_id"], "sad_sample_id": pair["sad"]["sample_id"],
                        "condition_id": args.condition, "prompt": condition["prompt"],
                        "happy_verbalizer": condition["happy_verbalizer"], "sad_verbalizer": condition["sad_verbalizer"],
                        "happy_token_ids": json.dumps(happy_ids), "sad_token_ids": json.dumps(sad_ids),
                        "layer_index": layer_index, "head_index": head_index, "layer": f"layer_{layer_index}",
                        "patch_site": "pre_o_proj_decision_head", "patch_kind": "attention_head",
                        "baseline_happy_happy_score": float(happy_baseline[happy_index]), "baseline_happy_sad_score": float(sad_baseline[happy_index]),
                        "baseline_sad_happy_score": float(happy_baseline[sad_index]), "baseline_sad_sad_score": float(sad_baseline[sad_index]),
                        "baseline_happy_margin": effects["baseline_happy_margin"], "baseline_sad_margin": effects["baseline_sad_margin"],
                        "patched_happy_target_happy_score": patched_happy_happy[happy_index], "patched_happy_target_sad_score": patched_happy_sad[happy_index],
                        "patched_sad_target_happy_score": patched_sad_happy[sad_index], "patched_sad_target_sad_score": patched_sad_sad[sad_index],
                        "patched_happy_target_margin": effects["patched_happy_target_margin"], "patched_sad_target_margin": effects["patched_sad_target_margin"],
                        "happy_effect_toward_sad": effects["happy_effect_toward_sad"], "sad_effect_toward_happy": effects["sad_effect_toward_happy"],
                        "counterfactual_effect": effects["counterfactual_effect"], "happy_direction": effects["happy_effect_toward_sad"] > 0,
                        "sad_direction": effects["sad_effect_toward_happy"] > 0, "error": "",
                    }
                )
            handle.flush()
            LOGGER.info("finished causal attention head: L%dH%d", layer_index, head_index)
    output_path.with_name(output_path.stem + "_run.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest), "n_samples": len(rows), "n_pairs": len(pairs),
                "condition_id": args.condition, "happy_token_ids": happy_ids, "sad_token_ids": sad_ids,
                "heads": [{"layer": layer, "head": head} for layer, head in heads],
                "patch_site": "pre_o_proj_decision_head", "checkpoint": str(args.checkpoint),
                "model_parameters_frozen": True,
                "effect_definition": "symmetric donor-aligned forced-choice margin CE",
                "causal_status": "targeted head-level activation patch; donor pre-o_proj attention vector is replaced only at decision position",
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    LOGGER.info("saved causal attention head rows to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ravdess-root", type=Path, required=True)
    parser.add_argument("--slam-llm-root", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--whisper-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--condition", choices=list(FORCED_CHOICE_CONDITIONS), default="upper_prompt__lower_spaced")
    parser.add_argument("--head", nargs="+", required=True, help="Selected heads as layer:head, e.g. 18:4 21:5")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--self-patch-tolerance", type=float, default=1e-3)
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
