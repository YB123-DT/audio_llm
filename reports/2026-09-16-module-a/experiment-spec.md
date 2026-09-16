# Module A: one comparable upstream / complete-answer / native-readout baseline

2026-09-16. Scope is limited to a unified table and an interface-validity judgment. No activation patching, calibration intervention, plugin/model training, path tracing or new full layer sweep.

## Main table

Same192 samples/96matched happy-sad pairs/24actors/speech intensity01, original uppercase emotion question, current single-token candidates ` happy`6247 and ` sad`12421. Objects:

1. Projector output temporal mean, external logistic probe.
2. **Complete final answer-position hidden state** immediately before LM head (after final RMSNorm), external logistic probe. This contains the whole answer residual, not an individual attention edge update.
3. Native unchanged LM-head log-likelihood margin, no fitted external classifier.

Reuse train-only StandardScaler, logistic regression C=1/lbfgs/max_iter5000, happy positive. Identical leave-one-actor-out, two statement-held-out directions, and joint actor+statement disjoint folds for both probes; no tuning or best-layer selection. Report accuracy and pooled OOF ROC-AUC with shared actor-cluster bootstrap, plus directional/fold AUC records to clarify pooled-score scale differences. Bootstrap conditions on fitted probes, does not refit; CIs unadjusted. Native predictions do not depend on fold training, so pooled native accuracy/AUC are repeated references, not three newly trained models.

## Reuse compatibility gate

Historical hidden archive: `/data2/yb/audio_llm_runs/ravdess_happy_sad_intensity01_systemprompt/hidden/representations.pt`. Its final answer vectors fail direct current-head historical-margin parity (max error0.75870), despite same prompt/sample identifiers, so do not mix them into the current table. Historical unseeded extraction predates the recorded seed1234 runtime; the cause of discrepancy must not be assumed without evidence.

Reuse the current clean archive `reports/2026-09-15-edge-representation/edge_vectors.npz`: it also contains **full** final decision pre-norm vectors and finalnorm weights. Reconstruct final RMSNorm using official epsilon, and validate all192 final-head margins against paired saved clean scores. This is not edge-vector probing and needs no new inference.

The historical projector is upstream of prompt/LLM state. Inspect the encoder/projector extraction and checkpoint loading, and numerically spot-check two existing samples before reuse. Report the spot-check scope honestly; if mismatch, pause use of this artifact and obtain only the missing compatible projector feature, not a new all-layer experiment.

## Minimal task-interface prerequisite

Inspect official local checkpoint Hydra config, upstream input wrappers and existing scoring parity. Audit checkpoint key sets/shapes against constructed model to expose strict=False omissions. This is verification, not an authorized model repair or new training run.

Existing records lack an explicit text-emotion-cue control. Six fixed supported-text-input cases: two RAVDESS statements × no cue / 'The speaker emotion is HAPPY.' / SAD, using official text-input wrapper, original classification system prompt and existing candidates. Record native forced-choice margins plus greedy max_new_tokens32 output/label compliance; no adaptive prompt search. No-cue cases have no emotion ground truth. Cue control is a task/label interface check, not evidence of audio recognition ability or a dataset accuracy estimate. Model seed1234 must match current score artifacts.

If the interface control fails, the table may still document representation/readout differences, but it does not license an audio-specific 'represented but ignored' mechanism claim. No extra interventions will be added in this round.
