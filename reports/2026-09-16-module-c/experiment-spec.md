# Module C: attention versus MLP boundaries in a normal forward

2026-09-16. Scope: localize changes in complete-answer linear readability and content-transfer gap to observed sublayer boundaries. No zeroing, ablation, patching, calibration, prompt search, head/path tracing or model training.

## Observational extraction and identities

Same192 samples, seed1234, original prompt/input/runtime as Module B. Inspect the installed Qwen2DecoderLayer forward. For each block0–23 capture the **complete answer-position residual**:

1. `in`: before the block's input RMSNorm.
2. `attn`: after attention residual addition, before post-attention RMSNorm.
3. `out`: after MLP residual addition, before the next block/final RMSNorm.

Observe all boundaries during the same normal frozen forward, one decoder forward per sample. Do not confuse post-attention RMSNorm output with the pre-norm residual, or an attention/MLP update with the full state. Verify each residual addition numerically, all24 out states against Module B, in[l+1]=out[l], and final normalized answer/native margins. The first block's answer input is expected to be constant across samples before audio enters that position; report this initialization separately from subsequent changes.

## Exact reuse of the matched probe protocol

At each boundary, for each held-out actor and test statement, within trains the other23 actors of the same statement, cross trains the other23 actors of the other statement. Both92 training samples and the identical4 test samples. Train-only StandardScaler, C=1 logistic regression, lbfgs/max_iter5000, happy=1, seed20260915, no tuning. Primary AUC is mean actor×statement test-fold AUC, not pooled AUC. Report each test statement and their equal-weight macro average. Accuracy and pooled AUC remain secondary.

Reuse the previous within/cross OOF predictions for exactly identical out states; reuse those same predictions for next-block inputs. Fit only the new post-attention states and the first input state. Verify cache identity, fold/label metadata, score completeness and numerical reproduction. This avoids stochastic or fitting differences at an identical state.

## Paired contrasts

At each block report W_in, W_attn, W_out, where W is within-content AUC, and G_in/G_attn/G_out, where G=W−cross-content AUC. Compute:

- ΔW_attention=W_attn−W_in; ΔW_MLP=W_out−W_attn.
- ΔG_attention=G_attn−G_in; ΔG_MLP=G_out−G_attn.

Use10,000 identical actor-cluster bootstrap draws across all conditions, states and both statements, seed20260915. Derive contrast intervals directly from paired draws, not by subtracting CI endpoints. Intervals condition on the fitted probes, are pointwise and unadjusted for multiple layers.

To summarize the observed post-initialization trajectory without selecting layers from results, also report signed sums of the per-component changes over blocks1–23, alongside block0 separately. The attention and MLP sums must telescope to out[23]−out[0], for both W and G. These are descriptive signed sums of AUC changes with cancellation, **not additive causal mediation fractions or information amounts**. Report all layers regardless of sign or significance. Do not infer a sublayer's intrinsic causal function from a before/after probe contrast alone.

## Deliverables and interpretation

Save compact vectors outside Git, OOF/fold records, all state metrics and component contrasts with paired CIs, source hashes, one readable set of W/G curves and component-delta plots, and a short report. Distinguish overall readability change from content-transfer change; a larger G can reflect worse cross-content ranking, better within-content ranking, or both. Preserve the previously observed test-statement asymmetry. Conclusions locate where a linear probe's performance changes in the natural computation, without proving erasure, direction rotation or that ablation would restore behavior.
