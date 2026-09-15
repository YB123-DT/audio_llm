# Full-depth causal sweep results

All 192 samples (96 strict matched pairs) completed all 24 post-block interventions: 4,608 directional observations. The point-estimate curve peaks in blocks 11–13 and then generally declines. Post-block 17 is a useful previously studied intervention boundary, but the full curve does not identify it as an isolated peak or the globally strongest boundary.

| Post-block | Mean bidirectional CE | Pointwise actor-bootstrap 95% CI |
|---|---:|---:|
| 10 | 0.06966 | [0.00069, 0.13902] |
| 11 | 0.08269 | [0.02015, 0.14450] |
| 12 | 0.08002 | [0.01897, 0.14120] |
| 13 | 0.08301 | [0.02449, 0.14280] |
| 14 | 0.05899 | [0.00507, 0.11178] |
| 15 | 0.05275 | [0.00343, 0.10031] |
| 16 | 0.05678 | [0.01720, 0.09570] |
| 17 | 0.04555 | [0.00847, 0.08568] |
| 18 | 0.03625 | [0.00173, 0.07254] |
| 19 | 0.02558 | [−0.01317, 0.05955] |
| 23 | 0 | [0, 0] |

The approximate 95% simultaneous bootstrap band over all 24 CE estimates excludes zero at blocks 11 and 13 only; block 12 is just below that threshold (lower bound −0.000091). These near-threshold differences should not be interpreted as a discontinuous anatomical boundary. For the more relevant direct comparison, CE13−CE17 = 0.03746 with pointwise CI [−0.00333, 0.07955] and simultaneous CI [−0.03340, 0.10833]. No layer-minus17 contrast excludes zero under the 24-layer simultaneous band. Thus the observed broader middle-layer peak revises the *description* of the curve, without establishing a uniquely superior intervention layer.

The complete post23 zero is structural: replacing only audio states after the last decoder block cannot change the untouched decision residual. It is a numerical implementation check, not an emotion-encoding claim.

Verification: historical clean and post17 patched margins match exactly for all 192 samples; clean margins are invariant across all layers. Self-patch error is exactly zero at every layer for both samples of the first pair. Single-token teacher-forced parity errors are at most 1.53e−5 (clean all samples) and 1.34e−5 (patched first pair across all layers). Statistical tests pass locally and remotely (2 tests). The model remained frozen.

See `sweep-method.md`, `sweep_run.json`, `sweep_analysis/verification.json`, the complete CSVs, and `sweep_analysis/sweep.png` for methods and evidence. Uncertainty concerns actors and these specific two statements; it does not establish generalization to unrestricted speech. Full-audio replacement can carry residual non-emotion acoustic differences even under matching.
