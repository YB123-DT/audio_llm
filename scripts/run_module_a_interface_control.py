"""Six fixed text-interface controls using the official text batch builder."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from . import slam_omni_diagnostics as base
else:
    import slam_omni_diagnostics as base


def cases():
    return [dict(statement=s, cue=cue, text=text + (f" The speaker emotion is {cue.upper()}." if cue else ""))
            for s, text in [("01", "Kids are talking by the door."), ("02", "Dogs are sitting by the door.")]
            for cue in ("", "happy", "sad")]


def official_batch(torch, model, root, text, device):
    """Execute original official helper functions; intercept before generation."""
    source = root / "examples/s2s/generate/generate_s2s_online.py"
    tree = ast.parse(source.read_text())
    names = {"get_input_ids", "get_padded_input", "generate_from_text"}
    subset = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = {"torch": torch, "layershift": lambda token, layer: token + base.SHIFT}
    exec(compile(subset, str(source), "exec"), namespace)
    proxy = SimpleNamespace(tokenizer=model.tokenizer, stream_generate=lambda **kw: kw)
    dataset = SimpleNamespace(prompt=base.DEFAULT_PROMPT, vocab_config=model.model_config.vocab_config,
                              task_type="s2s", num_latency_tokens=0)
    return namespace["generate_from_text"](text, proxy, None, dataset, {}, None, device,
                                              model.model_config, None, inference_streaming=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("slam-llm-root", "qwen-path", "whisper-path", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.seed = 1234
    torch, model, tokenizer, whisper = base.load_model(args)
    model.requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    runtime = model.state_dict()
    audit = {"missing_keys": sorted(set(runtime) - set(checkpoint)),
             "unexpected_keys": sorted(set(checkpoint) - set(runtime)),
             "shape_mismatches": [k for k in set(runtime) & set(checkpoint) if runtime[k].shape != checkpoint[k].shape],
             "runtime_key_count": len(runtime), "checkpoint_key_count": len(checkpoint)}
    audit["lm_head_tied_to_embedding"] = model.llm.lm_head.weight.data_ptr() == model.llm.get_input_embeddings().weight.data_ptr()
    audit["embedding_checkpoint_loaded_max_error"] = float((model.llm.get_input_embeddings().weight.detach().cpu() - checkpoint["llm.model.embed_tokens.weight"]).abs().max())
    del checkpoint, runtime
    legacy_dir = Path("/data2/yb/audio_llm_runs/ravdess_happy_sad_intensity01_systemprompt/hidden")
    legacy = torch.load(legacy_dir / "representations.pt", map_location="cpu", weights_only=False)
    with (legacy_dir / "metadata.csv").open() as handle:
        legacy_rows = list(csv.DictReader(handle))
    projector_checks = []
    with torch.inference_mode():
        for index in (0, 1):
            row = legacy_rows[index]
            batch, positions = base._audio_inputs(torch, whisper, model, tokenizer,
                Path("/data2/yb/paper/RAVDESS") / row["path"], base.DEFAULT_PROMPT)
            enc = model.encoder.extract_variable_length_features(batch["audio_mel"].permute(0, 2, 1))
            mean = model.encoder_projector(enc)[0, :positions["audio_length"]].mean(dim=0).cpu()
            projector_checks.append({"sample_id": row["sample_id"], "emotion": row["emotion"],
                "max_absolute_error": float((mean - legacy["projector"][index]).abs().max()),
                "mean_absolute_error": float((mean - legacy["projector"][index]).abs().mean())})
    del legacy
    candidates = [tokenizer.encode(" happy"), tokenizer.encode(" sad")]
    assert candidates == [[6247], [12421]], candidates
    results = []
    with torch.inference_mode():
        for case in cases():
            batch = official_batch(torch, model, args.slam_llm_root, case["text"], args.device)
            assert batch["audio_mel"] is None and not batch["modality_mask"].any()
            embeds = base._embed_llm_input_ids(model, batch["input_ids"]).mean(dim=1)
            out = model.llm(inputs_embeds=embeds, attention_mask=batch["attention_mask"], use_cache=False, return_dict=True)
            margin = float((out.logits[0, -1, 6247] - out.logits[0, -1, 12421]).item())
            logs = [base._teacher_forced_logprob(torch, model, embeds, batch["attention_mask"], ids) for ids in candidates]
            parity = abs(margin - (logs[0] - logs[1]))
            assert parity < 1e-3
            generated = model.generate(**batch, max_new_tokens=32, decode_text_only=True, do_sample=False,
                                       top_p=1., top_k=0, temperature=1., text_repetition_penalty=1.,
                                       audio_repetition_penalty=1., num_latency_tokens=0, do_layershift=False)
            raw = tokenizer.decode(generated[base.CODE_LAYER], add_special_tokens=False, skip_special_tokens=True)
            parsed, status = base.normalize_prediction(raw)
            results.append({**case, "margin": margin, "happy_logprob": logs[0], "sad_logprob": logs[1],
                            "predicted": "happy" if margin > 0 else "sad", "parity_error": parity,
                            "raw_output": raw, "parsed_output": parsed, "parse_status": status,
                            "input_ids": batch["input_ids"].cpu().tolist()})
            print(case["statement"], case["cue"], margin, repr(raw), flush=True)
    payload = {"projector_spotchecks": projector_checks, "checkpoint_audit": audit, "system_prompt": base.DEFAULT_PROMPT, "candidate_ids": candidates,
               "max_new_tokens": 32, "frozen": True, "seed": args.seed, "cases": results,
               "torch": torch.__version__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "official_builder_sha256": hashlib.sha256((args.slam_llm_root / "examples/s2s/generate/generate_s2s_online.py").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
