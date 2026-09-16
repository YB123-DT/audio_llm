# Module B: decoder-depth readability under the Module A protocol

2026-09-16. One question: does cross-statement emotion readability weaken throughout Qwen, drop at a particular depth, or remain at audio positions without becoming stable at the answer position? This is an observational linear-readability comparison, not a causal routing diagnosis.

## Fixed representations and endpoints

- Same 192 speech samples, 96 happy/sad matched pairs, 24 actors, intensity01 as Module A.
- Projector temporal mean is the pre-decoder reference.
- At each of the 24 Qwen blocks, take the output residual **after the full block**, indexed 0–23. Save audio-position temporal mean and the complete last-prompt/answer-position state. Neither is an isolated attention update.
- Keep final RMSNorm separate from block23: its complete answer state must reproduce Module A and its native score. This avoids silently mixing pre-norm and post-norm endpoints.
- Reuse only compatible cached states. The legacy answer archive failed Module A score parity, so it cannot be assumed compatible merely from matching IDs. If a fresh observational pass is needed, extract all layers in one forward per sample, freeze the model and use the current seed1234 runtime. Verify projector and final answer against Module A for all samples.

## Unchanged probes and evaluation

Module A folds: speaker leave-one-actor-out; both statement train/test directions; joint folds excluding both the test actor and test statement from training. Reuse exact membership. Train-only StandardScaler and logistic regression C=1, lbfgs, max_iter5000; happy=1, fixed seed20260915, no tuning or layer selection. Every sample receives exactly one held-out score per object/regime.

Report accuracy, pooled out-of-fold ROC-AUC and descriptive mean within-fold AUC. Reuse 10,000 actor-cluster bootstrap draws and unadjusted percentile intervals, conditional on fitted probes; no refitting. Pooled AUC combines independently fitted score scales, so inspect within-fold AUC before interpreting abrupt pooled changes. Compare Module A projector and normalized final-answer endpoint metrics.

Endpoint audit update: fresh actual GPU final RMSNorm and Module A's CPU-reconstructed normalization differ by at most 3.05176e−5, while native margins are identical. An initially exact metric gate detected a speaker pooled-AUC difference of 0.00032552. Retain fresh actual vectors, require exact projector metrics, and record every final-answer metric/CI difference rather than choosing an after-the-fact tolerance or substituting the old endpoint. Fold/probe protocol and extraction parity gates remain unchanged. This is an explicit provenance distinction, not a tuned experimental condition.

## Deliverable and interpretation limits

One figure: three split columns and accuracy/AUC rows, with audio and complete-answer curves, pre-decoder projector reference and explicitly marked normalized endpoint. Save underlying metrics, predictions, fold membership, provenance and a short report. Describe abrupt versus gradual changes from the observed trajectory without claiming statistical change-point localization. A lower probe score does not prove information destruction; position, pooling, sample size and linear accessibility matter. Module A's task-interface limitation still applies.

No patching, calibration, attention-path search, plugins, model training or new interface experiments in this module.
