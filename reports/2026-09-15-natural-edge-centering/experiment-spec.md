# Natural model readout after clean edge offset correction

2026-09-15, recorded before results. Reuse fixed clean edge vectors from192 RAVDESS happy/sad samples,96pairs,24actors; block18–23 post-o_proj audio→decision updates, same checkpoint/prompt/single-token happy6247 and sad12421. No new path tracing, donor swap, model training, or external probe fitting.

## Calibration and intervention

For each held-out actor a, estimate mu_s,l from all other23actors at statement s using only actor/statement metadata and edge vectors, without emotion labels. Each statement calibration set has92examples (known balanced experimental design); reference mean is the equal average of mu01 and mu02. Fixed shifts c_s,l=mu_bar,l−mu_s,l are estimated from the historical clean run, separately per layer.

At the actual o_proj output in a natural forward, add c_s,l only at the decision row. Audio states, other rows, model weights and prompt are unchanged by the hook. This implements e'_l=e_current,l+c_s,l while retaining the current nonaudio attention contribution and shared projection bias. It does not require reconstructing current attention because the correction is additive after o_proj.

Primary condition: correct all blocks18–23. Secondary separately correct19 and23 (motivated by previous edge and offset results). Include unhooked clean and all-layer zero-shift no-op. No gain/threshold optimization. Follow original downstream computation naturally; earlier corrections can alter later decision queries/updates. Calibration means remain fixed clean-run means, so joint intervention does not assert that the evolving distribution at every later layer is exactly centered.

## Validation

Exclude every sample of test actor from each calibration mean. Verify no emotion field is accessed by calibration, opposite statement corrections cancel, and matched happy/sad receive identical correction vectors. Check no-op and nondecision-row isolation, clean historical margin parity, finite scores and single-token likelihood/logit equivalence. All192samples,96pairs evaluated.

## Outcomes

Model-native S=logP(happy)−logP(sad), fixed zero threshold: accuracy, ROC-AUC, happy prediction fraction, per-class mean margin, common score shift, matched H−S separation and its positive fraction. Compute paired changes versus clean with10,000 actor-cluster bootstrap draws (seed20260915), conditional on the fixed calibration means. Report per-statement descriptives and individual pair shifts to distinguish common domain/bias changes from emotion-dependent responses. Intervals unadjusted across secondary conditions/metrics; no refit of calibration during bootstrap.

The same actor/statement correction is identical for both emotions. Any pair-separation change therefore arises through sample-dependent downstream computation, not label-specific shifts. AUC/accuracy can also change through statement/domain calibration without improving within-statement ranking; include per-statement AUC.

This is actor-disjoint **unlabeled target-statement calibration**, not unseen-statement induction. Statement identity is known and supplied externally. Balanced label prevalence and only two fixed sentences limit deployment/generalization claims. Failure does not prove offsets irrelevant; this particular fixed-mean intervention can create distribution shifts, and pure readout bias or geometry may remain. Success would demonstrate a causal effect on the model's own readout, not necessarily reliable emotion recognition or a complete mechanism explanation.
