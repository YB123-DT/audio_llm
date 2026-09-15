# Content-conditioned offset vs emotion geometry

2026-09-15, specified before results. Reuse frozen clean edge updates [192,6,896] from the previous experiment (NPZ SHA2560796d412b527dd281198c6e06b164b0b8ae990d54713b1009cbaebfc7283d490). All blocks18–23 reported, with19 a previously motivated focus. No new attention tracing, model inference, model modification or plugin training. External diagnostic linear probes only.

## Geometry

Compute class means separately for statement01/02 and happy/sad; d_s=mu_s,H−mu_s,S; o=(mu_02,H+mu_02,S−mu_01,H−mu_01,S)/2. Report cos(d01,d02), ||o||, ||d_s||, o·d_s and o·d_shared with d_shared=(d01+d02)/2, alongside normalized projections and relative scale. Use actor-cluster bootstrap for class-mean geometry where computed. Directions estimated from finite samples are noisy; low cosine alone cannot prove all transferable information absent.

Fit separate statement01/02 logistic probes with fixed C=1 and training-domain StandardScaler. Convert coefficient/intercept to raw residual coordinates. Report cos(w01,w02), raw and norm-normalized intercepts, w_s·o and within-domain class mean score gaps. Raw intercept differences from separately fitted models are not standalone evidence: direction, coefficient norm and score scale may differ. Positive class is happy throughout.

## Counterfactual conditions

Primary bidirectional statement-held-out baseline exactly reproduces the earlier standardized probe. Then center each domain by its own unlabeled sample mean, for both source training and target test, and fit the same probe. Test-domain mean uses all target examples and is **transductive**, not an inductive held-out score; no target emotion labels enter preprocessing, but balanced design and target-distribution access are explicit advantages.

With identical train-centered standardized fit, test centering is a constant score shift `score_centered = score_raw − w_raw·(mu_target−mu_source)`. Verify this identity and per-fold AUC invariance. Improvements in pooled AUC can arise solely from aligning offsets of different folds; they do not establish improved within-fold ranking.

Secondary actor-disjoint calibration: for each held-out actor and target statement, classifier labels come only from other actors' source-statement samples. Other actors' **unlabeled target-statement examples** may estimate the target mean/content offset. Test actor never contributes to these calibration estimates. Compare raw joint baseline, source-only centering no-op control, independent calibration mean-centering, and removal of rank1 offset direction estimated from source/target domain means of the other actors. Both train/test projected with the same fitted projector; train-only standardization after projection.

This nuisance condition is stricter about held-out test actors, but **is not strict unseen-statement induction**: target statement is observed unlabeled during calibration. With only one statement in a strict statement-held-out training set, its between-statement nuisance direction is unidentifiable from that set alone. Do not conceal this limitation or estimate it from test actor examples.

## Evidence and interpretation

Report per-layer and per-direction accuracy, pooled OOF AUC, within-fold AUC, predictions and paired10,000 actor-cluster percentile95% intervals (seed20260915), conditional on fitted probes/calibration means; not refitting models or estimating new means in bootstrap. No hyperparameter tuning, best-layer confirmation or multiplicity-adjusted probe claim.

A centering improvement supports an offset contribution to fixed-threshold failure, not that geometry is globally shared or that the natural model itself is corrected. A failed rank1 removal does not prove rotating emotion geometry: the removed direction may overlap emotion information, the content effect may be higher rank/nonlinear, and directions/probes are noisy. Assess geometry, changes and residual ranking together; avoid the proposed binary conclusion without supporting evidence.
