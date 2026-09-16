"""Manifest and interruption recovery guards for the fixed independent confirmation."""
import json
import numpy as np
import pytest
from scripts.run_crema_confirmation import validate_rows, validate_resume, save_npz, CONDITIONS


def rows():
    return [dict(sample_id='sample_'+emotion, pair_id='1001_DFA_XX', actor='1001', statement='DFA', intensity='XX', repetition='01', emotion=emotion) for emotion in ('happy', 'sad')]


def test_pairs_require_matched_metadata_and_both_emotions():
    source = rows()
    assert validate_rows(source) == ['sample_happy', 'sample_sad']
    for key in ('actor', 'statement', 'intensity', 'repetition', 'emotion'):
        changed = [dict(r) for r in source]
        changed[1][key] = 'different'
        with pytest.raises(ValueError):
            validate_rows(changed)
    with pytest.raises(ValueError):
        validate_rows(source[:1])
    with pytest.raises(ValueError):
        validate_rows(source + source)


def test_resume_rejects_different_provenance_order_and_partial_checks(tmp_path):
    state = dict(sample_ids=np.array(['sample_happy']), conditions=np.array(CONDITIONS), block6=np.zeros((1,3,896)), final_answer=np.zeros((1,3,896)), clean_margins=np.zeros(1), fingerprint=np.array('same'), checks=np.array(json.dumps(dict(nonanswer_block_comparisons=48, suppressed_update_calls=12, unhooked_clean_final_max=0))))
    path = tmp_path/'state.npz'
    save_npz(path, state)
    with np.load(path, allow_pickle=False) as saved:
        assert validate_resume(saved, ['sample_happy','sample_sad'], 'same')[0] == 1
        with pytest.raises(ValueError):
            validate_resume(saved, ['sample_happy','sample_sad'], 'different')
        with pytest.raises(ValueError):
            validate_resume(saved, ['sample_sad','sample_happy'], 'same')
    assert not path.with_suffix('.npz.part').exists()
    for key, value in [('checks', np.array(json.dumps(dict(nonanswer_block_comparisons=47, suppressed_update_calls=12)))), ('final_answer', np.full((1,3,896), np.nan)), ('conditions', np.array(CONDITIONS[::-1]))]:
        broken = dict(state, **{key: value})
        with pytest.raises(ValueError):
            validate_resume(broken, ['sample_happy','sample_sad'], 'same')


def test_audio_fingerprint_rejects_changed_bytes_path_and_coverage(tmp_path):
    import csv
    from scripts.run_crema_confirmation import validate_audio_checksums, sha256
    source = rows()
    entries = []
    for row in source:
        row['path'] = row['sample_id'] + '.wav'
        audio = tmp_path / row['path']
        audio.write_bytes(row['sample_id'].encode())
        entries.append(dict(sample_id=row['sample_id'], path=row['path'], sample_rate=16000, frames=16000, duration_seconds=1, sha256=sha256(audio)))
    checksums = tmp_path / 'checksums.csv'
    def write(values):
        with checksums.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(values)
    write(entries)
    validate_audio_checksums(source, checksums, tmp_path)
    write(entries[:1])
    with pytest.raises(ValueError, match='coverage'):
        validate_audio_checksums(source, checksums, tmp_path)
    write([dict(entries[0], path='changed.wav'), entries[1]])
    with pytest.raises(ValueError, match='path'):
        validate_audio_checksums(source, checksums, tmp_path)
    write(entries)
    (tmp_path / source[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        validate_audio_checksums(source, checksums, tmp_path)
