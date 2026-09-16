# Final fixed early-answer intervention

2026-09-16. Exactly three conditions, normal frozen model except the specified answer-position residual update. This is the requested causal confirmation following Module C, not another observational survey or an adaptive repair search.

## Intervention and invariant checks

Use the existing zero-based Qwen numbering: intervene on blocks1,2,3,4,5,6 inclusive; block0 (initial audio integration) is unchanged. Same192 speech samples,24actors,96happy/sad pairs,intensity01,seed1234,prompt/checkpoint/runtime as Module B/C. Answer is the final prompt position330, not a generated candidate token.

- **clean:** no update suppressed.
- **no-attn:** compute attention normally, replace only its answer-position output row by zero immediately before residual addition, in blocks1–6. MLP runs normally on the resulting state.
- **no-MLP:** compute attention and MLP normally, replace only the MLP answer-position output row by zero immediately before residual addition, in blocks1–6.

Do not freeze the answer state, swap a donor, change attention weights/masks, suppress an entire block, zero other positions or restore later layers. Retained computations respond naturally to the altered answer state. Capture post-block6 complete answer state, then continue the normal decoder to the complete final RMSNorm answer state.

Run clean and the two interventions on the same prefix for every sample. Verify clean block6/final states/native margins against Module B. Because answer is the last causal token, all earlier positions should remain exactly clean; numerically check every block's non-answer states, including all audio positions, rather than only asserting that the hook targets one row. Check zeroed update rows, unchanged other update rows, exact target layers and hook cleanup.

## Fixed evaluation

**Primary endpoint: W6, block6 within-content mean actor-fold ROC-AUC.** For each testactor and teststatement, train on the other23 actors of the same statement (92train,4test,balanced labels). Each condition gets its own train-only StandardScaler and C=1 logistic probe, lbfgs/max_iter5000, happy=1, seed20260915. Reuse clean probe scores only after exact vector/source identity checks. Fit the intervention conditions under exactly the same folds. No cross-content probes are added in this experiment.

**Secondary endpoint:** within-content mean actor-fold AUC at final answer **after terminal RMSNorm**, explicitly separate from block6 and from native LM-head behavior. Retain per-statement results to expose the known directional asymmetry. Combined AUC is the equal mean across24actor folds and2statements; pooled AUC/accuracy, if saved, are supplemental rather than interchangeable with the primary metric.

Two planned primary contrasts:

- ΔA=W6(no-attn)−W6(clean).
- ΔA−M=W6(no-attn)−W6(no-MLP).

The no-MLP−clean contrast and final-endpoint contrasts are secondary. Bootstrap the same24 actor clusters10,000times,seed20260915, preserving both statements and both conditions in each draw. Compute paired contrasts inside each draw; fixed fitted probes, no refitting. Report percentile95% intervals, unadjusted for the planned contrasts and descriptive statement subgroups.

## Interpretation and stop rule

If no-attn improves the primary endpoint over both clean and no-MLP with supporting paired uncertainty, this supports a causal contribution of early answer-position attention updates to the observed loss of linearly decodable emotion structure under this specific intervention and probe protocol. It does not establish lexical semantics as the source, identify audio/prompt/task contributions, prove information erasure, or show improved native classification/instruction following. Suppression changes later answer computation and can move states away from the natural distribution; this bounds generalization even though only one row is changed.

If no-attn fails to restore W6, report that simple blocking did not repair the localized decline. If results are mixed or uncertain, report that uncertainty. In every case stop after these three conditions and the predeclared endpoints: no new block windows, heads, gain sweeps, calibration or plugins.
