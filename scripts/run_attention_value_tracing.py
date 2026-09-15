#!/usr/bin/env python3
"""Trace decision-to-audio attention and value flow in frozen SLAM-Omni.

This is a descriptive path-tracing pass.  For each sample and Qwen decoder
layer/head it records the attention mass from the decision position to the
audio span and the norm of the attention-weighted value contribution from
that span.  Pair-level sad-minus-happy deltas are saved for later causal head
patching; no model state is modified.
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
        _decoder_forward,
        _llm_layers,
        _pair_indices,
        _prepare_prefixes,
    )
    from scripts.slam_omni_diagnostics import FORCED_CHOICE_CONDITIONS, load_model, read_rows
except ModuleNotFoundError:  # Direct ``python scripts/run_attention_value_tracing.py`` entry point.
    from run_activation_patching import (
        DEFAULT_SEED,
        _append_candidate_batch,
        _decoder_forward,
        _llm_layers,
        _pair_indices,
        _prepare_prefixes,
    )
    from slam_omni_diagnostics import FORCED_CHOICE_CONDITIONS, load_model, read_rows


LOGGER = logging.getLogger("run_attention_value_tracing")


def _set_eager_attention(layers: Any) -> None:
    """Force Qwen attention to return weights rather than SDPA placeholders."""
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise RuntimeError("Qwen decoder layer has no self_attn module")
        config = getattr(attention, "config", None)
        if config is not None:
            config._attn_implementation = "eager"


def _attention_shape(attention: Any) -> tuple[int, int]:
    heads = getattr(attention, "num_heads", None)
    head_dim = getattr(attention, "head_dim", None)
    if heads is None or head_dim is None:
        config = attention.config
        heads = int(config.num_attention_heads)
        head_dim = int(config.hidden_size) // heads
    return int(heads), int(head_dim)


def _value_audio_contribution(value_audio: Any, attention_audio: Any, attention_module: Any, torch: Any) -> np.ndarray:
    """Return one L2 norm per query head for the audio value contribution."""
    heads, head_dim = _attention_shape(attention_module)
    kv_heads_value = getattr(attention_module, "num_key_value_heads", None)
    if kv_heads_value is None:
        kv_heads_value = getattr(attention_module.config, "num_key_value_heads", heads)
    kv_heads = int(kv_heads_value)
    if value_audio.shape[-1] != kv_heads * head_dim:
        raise RuntimeError(
            f"Unexpected value projection width {value_audio.shape[-1]} != {kv_heads}*{head_dim}"
        )
    values = value_audio.reshape(value_audio.shape[0], value_audio.shape[1], kv_heads, head_dim)
    values = values.permute(0, 2, 1, 3)
    groups_value = getattr(attention_module, "num_key_value_groups", None)
    if groups_value is None:
        groups_value = getattr(attention_module.config, "num_key_value_groups", heads // kv_heads)
    groups = int(groups_value)
    values = values.repeat_interleave(groups, dim=1)
    if values.shape[1] != heads:
        raise RuntimeError(f"Expanded value heads {values.shape[1]} != attention heads {heads}")
    weights = attention_audio.to(dtype=values.dtype)
    contribution = torch.sum(weights[:, :, :, None] * values, dim=2)
    return torch.linalg.vector_norm(contribution, dim=-1).detach().cpu().numpy()[0]


def _trace_sample(
    torch: Any,
    model: Any,
    layers: Any,
    prefix: Any,
    positions: dict[str, int],
    happy_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture [layer, head] attention mass and value norm for one sample."""
    device = next(model.parameters()).device
    prefix_batch = prefix.unsqueeze(0).to(device)
    full_inputs = _append_candidate_batch(torch, model, prefix_batch, [[happy_id]])
    attention_mask = torch.ones(
        (1, full_inputs.shape[1]), dtype=torch.bool, device=full_inputs.device
    )
    audio_start = positions["audio_start"]
    audio_end = audio_start + positions["audio_length"]
    decision_position = positions["decision_position"]
    attention_rows: dict[int, Any] = {}
    value_audio: dict[int, Any] = {}
    handles = []

    for layer_index, layer in enumerate(layers):
        attention_module = layer.self_attn

        def make_attention_hook(index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                weights = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
                if weights is None:
                    raise RuntimeError(
                        "Qwen attention did not return weights; eager attention may not be active"
                    )
                attention_rows[index] = weights[:, :, decision_position, audio_start:audio_end].detach().cpu()

            return hook

        def make_value_hook(index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                value_audio[index] = output[:, audio_start:audio_end, :].detach().cpu()

            return hook

        handles.append(attention_module.register_forward_hook(make_attention_hook(layer_index)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_value_hook(layer_index)))
    try:
        with torch.inference_mode():
            _decoder_forward(model, full_inputs, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    if set(attention_rows) != set(range(len(layers))) or set(value_audio) != set(range(len(layers))):
        raise RuntimeError("Tracing hooks did not observe every decoder layer")
    attention_mass = np.stack(
        [attention_rows[index].sum(dim=-1).numpy()[0] for index in range(len(layers))]
    )
    value_norm = np.stack(
        [
            _value_audio_contribution(
                value_audio[index], attention_rows[index], layers[index].self_attn, torch
            )
            for index in range(len(layers))
        ]
    )
    return attention_mass, value_norm


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size != x.size:
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator == 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
    if len(happy_ids) != 1 or len(sad_ids) != 1:
        raise ValueError(f"Attention tracing requires single-token verbalizers: happy={happy_ids}, sad={sad_ids}")
    prefixes, positions = _prepare_prefixes(
        torch, model, whisper, tokenizer, rows, args.ravdess_root, condition["prompt"]
    )
    layers = _llm_layers(model)
    _set_eager_attention(layers)
    n_layers = len(layers)
    first_attention, first_value = _trace_sample(
        torch, model, layers, prefixes[0], positions, happy_ids[0]
    )
    n_heads = first_attention.shape[1]
    attention_mass = np.zeros((len(rows), n_layers, n_heads), dtype=np.float32)
    value_norm = np.zeros_like(attention_mass)
    attention_mass[0], value_norm[0] = first_attention, first_value
    LOGGER.info("traced sample 1/%d with %d layers × %d heads", len(rows), n_layers, n_heads)
    for index in range(1, len(rows)):
        current_attention, current_value = _trace_sample(
            torch, model, layers, prefixes[index], positions, happy_ids[0]
        )
        if current_attention.shape != first_attention.shape:
            raise RuntimeError("Attention head shape changed across samples")
        attention_mass[index], value_norm[index] = current_attention, current_value
        if (index + 1) % 16 == 0:
            LOGGER.info("traced samples: %d/%d", index + 1, len(rows))

    pair_attention = np.stack(
        [attention_mass[pair["sad_index"]] - attention_mass[pair["happy_index"]] for pair in pairs]
    )
    pair_value = np.stack(
        [value_norm[pair["sad_index"]] - value_norm[pair["happy_index"]] for pair in pairs]
    )
    patch_effect = None
    if args.patch_csv:
        with args.patch_csv.open(newline="") as handle:
            patch_rows = list(csv.DictReader(handle))
        patch_effect = {
            row["pair_id"]: float(row["counterfactual_effect"])
            for row in patch_rows
            if row.get("patch_kind") == "audio_tokens" and row.get("layer_index") == "17"
        }
        missing = [pair["pair_id"] for pair in pairs if pair["pair_id"] not in patch_effect]
        if missing:
            raise ValueError(f"Patch CSV is missing layer17 audio effects for {len(missing)} pairs")
    summary_rows: list[dict[str, Any]] = []
    for layer_index in range(n_layers):
        for head_index in range(n_heads):
            attention_values = pair_attention[:, layer_index, head_index]
            value_values = pair_value[:, layer_index, head_index]
            row: dict[str, Any] = {
                "layer_index": layer_index,
                "head_index": head_index,
                "n_samples": len(rows),
                "n_pairs": len(pairs),
                "mean_attention_mass": float(attention_mass[:, layer_index, head_index].mean()),
                "sd_attention_mass": float(attention_mass[:, layer_index, head_index].std(ddof=1)),
                "mean_audio_value_contribution_norm": float(value_norm[:, layer_index, head_index].mean()),
                "sd_audio_value_contribution_norm": float(value_norm[:, layer_index, head_index].std(ddof=1)),
                "mean_sad_minus_happy_attention_mass": float(attention_values.mean()),
                "mean_abs_sad_minus_happy_attention_mass": float(np.abs(attention_values).mean()),
                "mean_sad_minus_happy_value_norm": float(value_values.mean()),
                "mean_abs_sad_minus_happy_value_norm": float(np.abs(value_values).mean()),
                "attention_value_product": float(
                    attention_mass[:, layer_index, head_index].mean()
                    * value_norm[:, layer_index, head_index].mean()
                ),
            }
            if patch_effect is not None:
                ce = np.asarray([patch_effect[pair["pair_id"]] for pair in pairs], dtype=np.float64)
                row["corr_pair_attention_delta_with_layer17_ce"] = _pearson(attention_values, ce)
                row["corr_pair_value_delta_with_layer17_ce"] = _pearson(value_values, ce)
            summary_rows.append(row)

    output_path = args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, summary_rows)
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        attention_mass=attention_mass,
        value_norm=value_norm,
        pair_attention_delta=pair_attention,
        pair_value_delta=pair_value,
    )
    metadata = {
        "manifest": str(args.manifest),
        "n_samples": len(rows),
        "n_pairs": len(pairs),
        "condition_id": args.condition,
        "happy_token_ids": happy_ids,
        "sad_token_ids": sad_ids,
        "layer_count": n_layers,
        "head_count": n_heads,
        "audio_start": positions["audio_start"],
        "audio_length": positions["audio_length"],
        "decision_position": positions["decision_position"],
        "checkpoint": str(args.checkpoint),
        "model_parameters_frozen": True,
        "attention_implementation": "eager",
        "trace_definition": "attention mass from decision position to audio span; value norm is L2 norm of attention-weighted V contribution per head",
        "pair_delta_definition": "sad minus happy for the same actor/statement/repetition/intensity pair",
        "causal_status": "descriptive only; no head or value state was patched",
        "patch_csv": str(args.patch_csv) if args.patch_csv else None,
        "seed": args.seed,
    }
    output_path.with_name(output_path.stem + "_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    LOGGER.info("saved attention/value tracing summary to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ravdess-root", type=Path, required=True)
    parser.add_argument("--slam-llm-root", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--whisper-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--patch-csv", type=Path)
    parser.add_argument("--condition", choices=list(FORCED_CHOICE_CONDITIONS), default="upper_prompt__lower_spaced")
    parser.add_argument("--batch-size", type=int, default=1)  # Kept for CLI parity; tracing is sample-wise.
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
