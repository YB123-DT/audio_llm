from scripts.run_module_a_interface_control import cases


def test_fixed_cases_are_paired_cue_only_controls():
    rows = cases()
    assert len(rows) == 6
    for start in (0, 3):
        plain, happy, sad = rows[start:start + 3]
        assert plain['cue'] == ''
        assert happy['text'] == plain['text'] + ' The speaker emotion is HAPPY.'
        assert sad['text'] == plain['text'] + ' The speaker emotion is SAD.'
        assert plain['statement'] == happy['statement'] == sad['statement']


def test_official_text_builder_preserves_markers_and_disables_audio():
    import pytest
    torch = pytest.importorskip('torch')
    from pathlib import Path
    from types import SimpleNamespace
    from scripts.run_module_a_interface_control import official_batch, base
    root = Path('/data2/yb/paper/SLAM-LLM')
    if not root.exists():
        pytest.skip('local official source unavailable')
    vocab = SimpleNamespace(code_layer=3, pad_a=4097, input_a=4098, eoa=4096,
                            answer_a=4099, input_t=151938, pad_t=151937,
                            eot=151936, answer_t=151939)
    tokenizer = SimpleNamespace(encode=lambda text: [11, 12] if text == 'cue' else [7])
    model = SimpleNamespace(tokenizer=tokenizer, model_config=SimpleNamespace(vocab_config=vocab, code_type='CosyVoice'))
    batch = official_batch(torch, model, root, 'cue', 'cpu')
    assert batch['input_ids'][0, 3].tolist() == [151938, 7, 151936, 151938, 11, 12, 151936, 151939]
    assert batch['input_ids'][0, 0].tolist() == [156097]*3 + [156098, 156097, 156097, 156096, 156099]
    assert batch['audio_mel'] is None
    assert not batch['modality_mask'].any()
