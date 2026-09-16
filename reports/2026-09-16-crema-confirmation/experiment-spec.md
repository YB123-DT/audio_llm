# Independent CREMA-D confirmation — frozen before model execution

Date: 2026-09-16. Status at protocol freeze: metadata inspected; no CREMA-D model outputs or probe results inspected. RAVDESS is the discovery dataset, including the preceding causal confirmation; it is not independent validation. This experiment tests replication on a separate corpus under the same checkpoint, not another search for an intervention window.

## Question and fixed intervention

Does suppressing early answer-position attention updates improve linearly decodable emotion at block6 on new corpus actors and a larger set of statements? Use the identical checkpoint, float32 runtime, audio preprocessing, prompt, label direction, seed1234, and zero-based blocks1–6 as commit296a8bd. Exactly three conditions: clean; no_attn (only answer attention output before residual add zeroed); no_mlp (only answer MLP output zeroed). Other component recomputes normally. Block0 and all other positions remain normal. Verify bitwise non-answer parity at all24blocks for each intervention. Never tune layers, prompts, intensity, actor selection, gain, probe C, or endpoints after viewing results.

## Dataset selected from metadata only

Official source: https://github.com/CheyneyComputerScience/CREMA-D at revision1658cd342dff90010aa843eaeebd53610a08b1dc. File conventions: https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/README.md .

Use intended filename HAP/SAD labels and intensity codeXX across all11available XX statements (DFA,IOM,ITH,ITS,IWL,IWW,MTI,TAI,TIE,TSI,WSI). XX means unspecified, NOT equal acoustic intensity or RAVDESS intensity01. IEO has separately specified intensities and is excluded by this uniform-code rule. Retain every actor with all11happy/sad cells complete. Exclude actors1008(missingWSI),1009(missingMTI),1019(missingITH), solely for metadata completeness. Result:88actors ×11statements ×2emotions =1936samples,968pairs. No performance, crowd-rating, or perceived-quality filtering. Single file per emotion/cell; repetition01 is only a loader placeholder.

Manifest SHA256: b7db3dff0af5a0e18a435ebbe888181d001501a190f4bba9f937ed2c8c6993a3 . Freeze manifest before forward; validate all WAVs decode and record checksums. Any missing/corrupt file halts the run for redownload; do not silently change the frozen cohort. No RAVDESS samples/vectors/predictions enter this analysis. Corpus actor IDs are separate; real-person cross-corpus identity overlap and model pretraining exposure cannot be independently ruled out.

## Probe and estimand

Primary endpoint W6: block6 complete answer residual within-statement mean actor-fold ROC-AUC. For each of11statements, leave out one of88actors; train on the other87actors (174samples) and test its happy/sad pair (2samples). Fit separate train-only StandardScaler and LogisticRegression(C=1,lbfgs,max_iter5000,random_state20260915), happy=1, for every condition/endpoint. Refit all clean probes on CREMA-D; no score reuse across datasets. Equal-weight average across88actors and11statements. Individual fold AUC is coarse (0,0.5,1) because there is one positive and one negative; report aggregate with uncertainty rather than treating each tiny fold as precise. This is speaker-held-out within-content decoding on a new corpus, NOT a classifier trained on RAVDESS nor unseen-statement transfer.

Co-primary directional contrasts: ΔA=W6(no_attn)−W6(clean); ΔA−M=W6(no_attn)−W6(no_mlp). Replication support requires both positive contrasts with paired95%bootstrap lower bounds above0. This is a conjunction, not selection of whichever comparison succeeds. No requirement to reproduce the RAVDESS point estimate0.922. Report all values even if effects reverse or clean is near ceiling.

Use10000paired actor-cluster percentile bootstrap draws, seed20260915, shared across conditions/endpoints; retain each actor's11statements together. Intervals are conditional on the fixed11statements and fitted probes; no refitting or generalization interval over arbitrary sentences. Per-statement breakdown is descriptive, never used to select a subset. Pooled AUC/accuracy may be supplemental; cannot replace the primary fold metric.

Secondary only: complete final answer after terminal RMSNorm; no_mlp−clean; native clean margins if retained are provenance, not intervention behavioral efficacy. No extra cross-content, projector, layerwise, head, or source analyses.

## Interpretation and stopping

Positive result supports replication of the specified intervention effect on emotion linear readability across these selected new-corpus actors/statements under the same checkpoint. It does not establish lexical-semantic suppression, native LM-head improvement, information deletion, efficacy for other checkpoints, or intensity-matched cross-corpus causality. Record changes in recording domain, intended-label noise, intensity scheme and probe training-set size as limits.

Negative/uncertain result limits external validity; it does not automatically identify which domain factor explains the discrepancy. Stop after the fixed experiment regardless of outcome. No rescue window search or new conditions.
