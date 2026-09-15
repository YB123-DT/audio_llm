# Clean causal-edge representation and full-depth validation

Specification recorded before this round's results, 2026-09-15. Frozen SLAM-Omni-0.5B, existing192-sample RAVDESS speech intensity01 manifest,96strict matched pairs/24actors; prompt and single-token verbalizers unchanged (` happy`6247, ` sad`12421). No model/plugin training; external diagnostic probes only.

## Primary: clean edge representation

For each block l=18,…,23 capture the natural decision-query audio-source attention contribution. Heads are concatenated in the existing Qwen layout, then mapped into residual coordinates using o_proj.weight with **no bias**:

`e_l = W_O,l concat_heads(sum_audio A_clean[decision,a] V_clean[a])`.

Save each layer separately, without summing across depths. This is the actual local attention update of the tested edge, not the total causal effect after subsequent nonlinear blocks. Current clean attention probabilities retain normalization over all allowed source positions; do not renormalize within audio. Verify attention reconstruction, source partition closure, hooked clean margin parity, finite vectors and fixed sample ordering.

For each layer, fixed C=1 logistic regression with training-only standardization and no hyperparameter/layer selection: leave-one-actor-out; bidirectional statement-held-out; joint leave-one-actor-out plus held-out statement (training actors and statement both disjoint from test). Keep matched happy/sad pairs within each split. Save folds, out-of-fold predictions, accuracy/AUC and10,000 actor-cluster bootstrap intervals. OOF bootstrap conditions on fitted folds and does not capture training instability. Statement-only generalization is over these two fixed sentences; it is not general language invariance.

This protocol differs from earlier unstandardized probes and their fixed speaker split; do not interpret differences from old probe percentages as a controlled representation-only comparison.

Orient probe coefficients toward happy. Convert standardized coefficients into residual coordinates `w_raw = w_scaled / scale_train` before computing cosine with `d_LM = W_U[happy]-W_U[sad]`. Report fold-specific cosine and a final-norm-weighted axis where available; do not silently compare standardized coefficients with raw vocabulary vectors. Compare edge projections onto the LM axis with external probe readouts.

High probe accuracy establishes accessible information in these updates, not sufficiency of that information for the model's natural task. Near-zero cosine in high dimension alone does not prove readout alignment failure: residual anisotropy, RMSNorm, downstream attention/MLP transformations and non-unique probe directions matter. These are geometric diagnostics, not an intervention on a verified final readout direction. Do not declare routing resolved or failure exclusively localized to LM head from probe/cosine alone.

## Secondary: full-depth matched full-audio causal sweep

Post-block0–23, replace target audio residual states with the opposite-emotion matched donor at that same layer, keeping nonaudio/prompt positions unchanged. Evaluate both directions and average donor-oriented shifts per pair (same CE convention as earlier). Report24-layer pointwise actor-bootstrap CIs plus simultaneous bands to limit selection claims. Compare layer17 against its historical result; examine neighboring layers without choosing an optimum and reporting it as preselected. Last-block audio-only patch should have zero effect at the already-computed decision position, serving as structural control.

The sweep validates a causal leverage window under this intervention/checkpoint/verbalizer; it does not establish natural encoding onset or emotion specificity relative to every possible donor control.
