# Answer-state within-content versus cross-content readability

2026-09-16. Reuse Module B's complete answer states at post-block0–23 and its separate actual final RMSNorm endpoint. No model forward, audio extraction, intervention, calibration, interface search or model training.

## Matched comparison

For each test actor a and statement s, test the same four samples (two emotions × two repetitions).

- Within-content: train on the other23 actors speaking statement s (92 samples).
- Cross-content: reuse Module B **joint-held-out** predictions, trained on the other23 actors speaking the other statement (92 samples).

Thus test actor, test content, test samples, training size, class balance and probe hyperparameters match; the training statement changes. Neither test actor appears in training. The two directional statement-only folds from Module B are not the primary comparator because they do not exclude the test actor. The mixed-statement speaker probe is also not a within-statement probe.

Use train-only StandardScaler, logistic regression C=1/lbfgs/max_iter5000, happy=1, seed20260915. No tuning or layer selection. Cache reuse must validate unique sample/layer/fold coverage and identifiers. Save within-content predictions and their paired original cross-content scores.

## Metrics and gap

Primary AUC for each statement is the mean of its24 actor-held-out fold AUCs, comparing happy/sad only inside the same held-out actor and statement. Overall AUC is the equal average across both statements (48 folds). Define G_l=AUC_within(l)−AUC_cross(l) with exactly this same aggregation on both sides. Report statement01, statement02 and overall separately.

Also report pooled OOF AUC and accuracy, and paired gaps, as secondary metrics. Independently fitted probe score scales can affect pooled AUC; do not mix pooled cross-content AUC with fold-averaged within-content AUC when calculating G. Existing Module B headline pooled AUC~0.54 is therefore not directly the cross-content baseline for a fold-averaged primary G.

Use the same10,000 actor-cluster bootstrap draws for both conditions and statements, seed20260915, preserving all repetitions and both emotions of each sampled actor. Bootstrap the paired difference directly; do not subtract marginal CI endpoints. Intervals condition on fitted probes (no refitting), are pointwise and unadjusted across layers. Four samples per test fold mean individual AUCs are coarse; interpret actor-aggregated curves and intervals.

## Interpretation and delivery

One figure compares within/cross AUC and G along depth, separately for each statement and overall. Retain raw OOF scores, fold membership/metrics, summary/CIs and source hashes. If late within-content AUC stays high while cross-content declines, support content-dependent linear readability; this alone does not establish rotation of emotion directions. A constant scalar score offset within a statement cannot change its within-actor fold AUC, so it cannot alone explain a primary ranking gap; it can affect pooled AUC/accuracy. Covariance changes, finite-sample fitting and other encoding differences still limit a specific geometry claim. If both weaken, support a broader decline in linear accessibility under this probe/data protocol, not proof that emotion information is absent. Mixed trajectories are possible and should not be forced into a binary mechanism claim.
