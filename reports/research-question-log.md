# 研究问题变更记录

这份记录追踪研究问题如何从“Audio-LLM 是否利用语气”逐步收窄到“跨文本可读但 context-conditioned 的语气信息，为什么没有成为最终决策依据”。问题变化由上一轮实验结果驱动；每一版保留当时的核心问题、实验、结果和转向理由。

| 版本 | 当时的核心问题 | 对应实验 | 关键结果 | 为什么发生问题变化 | 变化后的问题 |
|---|---|---|---|---|---|
| **Q0：文献起点** | Audio-LLM 不利用语气，是因为没编码，还是编码了但没用？ | 参考 LISTEN、Heard but Not Heeded 的行为与 layer-wise probing 思路。 | 文献提示 encoding 和 utilization 可能分离。 | 问题还太宽，没有区分“表征是否抽象”。 | 加入 abstraction 这一中间环节。 |
| **Q1：最初本地问题** | 模型有没有形成 content-invariant prosody state？如果有，最终是否使用？ | RAVDESS 严格 happy/sad 配对；layer-wise hidden extraction；linear probe；statement-held-out；global PS；free generation。 | Projector probe：speaker-held-out 95.8%，statement-held-out 91.1%；但 global PS 仅 0.114。 | 高 probe、低 PS 同时出现，说明“能读出”和“统一方向”不是一回事。 | 区分 linear separability 与 geometric parallelism。 |
| **Q2：第一次修正** | Emotion 是不是跨文本线性可读，但不是一根统一向量？ | 比较 statement-held-out probe 与 sad−happy difference-vector PS。 | 同一个线性边界可以迁移到另一 statement；差向量却不平行。 | 说明原来的“抽象/不抽象”二分太粗。 | 问：不平行到底由 lexical content、speaker 还是 repetition 造成？ |
| **Q3：第二次修正** | 低 global PS 是否主要来自 lexical content？ | 新增 \(PS_{\text{text}}\)、\(PS_{\text{speaker}}\)、\(PS_{\text{repetition}}\)。 | Projector：\(PS_{\rm all}=0.114\)、\(PS_{\rm text}=0.208\)、\(PS_{\rm speaker}=0.102\)、\(PS_{\rm repetition}=0.438\)。 | 低 PS 不能归因于文本；speaker、statement、repetition 都会改变 emotion direction，speaker 影响尤其明显。 | Emotion 更像 context/speaker-conditioned code，而不是全局向量。 |
| **Q4：利用问题** | 即使外部线性头能读出，模型自己的 likelihood readout 能不能使用？ | 2×2 prompt/verbalizer forced-choice sequence likelihood；free-generation sensitivity。 | 四个条件 accuracy 都是 50%；ROC-AUC 0.486–0.524；每个条件都退化为同一标签偏置。 | 证明当前模型 readout 没有形成可用 happy/sad likelihood separation；但仍未定位具体失败机制。 | 问信息在哪一步失去 decision relevance。 |
| **Q5：当前问题** | 跨文本可读但 context-conditioned 的语气信息，为什么没有成为最终决策依据？ | 待做：audio-token 与 decision-token 对照、联合 speaker+statement held-out、activation/value patching、forced-choice causal intervention。 | 现有线索：audio-token mean 中可读性高，decision state 的跨 statement probe 后期接近 chance；模型 likelihood 也是 chance。 | 可能不是“表示不存在”，而是 audio representation 没有成功 routing 到 decision token，或最终 readout 没读取它。 | **Routing failure 还是 readout failure？** |
| **Q6：activation patching 结果** | 在候选层直接交换 matched-pair 的 audio-token 或 decision-token state，哪一种能把最终 forced-choice margin 推向 donor emotion？ | `layer_7,14,15,17,22,23` × `{audio_tokens, decision_token}`，每格 96 对；固定 `seed=1234`，按双向 $S$ margin 计算 CE。 | 只有 `layer_17/audio_tokens` 的 CE 通过描述性 CI95 下界 >0：+0.061（[+0.020,+0.102]）；同层 decision patch 未通过；`layer_23/audio_tokens` 精确为 0；其余层未达到 CI 下界 >0。12 个条件未做多重比较校正。 | 局部 audio state 确有干预性影响，但直接 decision-state 替换不稳定，不能把它直接等同于自然 routing 或可用 readout。 | **该 layer-17 效应是自然 audio→decision routing，还是分布式 token computation / off-manifold patch artifact？** |
| **Q7：persistence 结果** | layer17 的 audio evidence 需要在后续 blocks 中持续存在多久，decision state 的一次性 patch 是否会被后续 computation 覆盖？ | Audio：layer17 donor patch 后在 layer18–23 分别 restore target，另加 no-restore；Decision：layer17–17/18/…/23 的 same-layer donor clamp；每格 96 对。 | Audio CE：restore18 +0.028、restore19 +0.036、restore20 +0.019、restore21 +0.028、restore22 +0.061、restore23/no-restore +0.061；曲线非单调。Decision clamp：−0.059、−0.034、−0.029、−0.043、−0.034、+0.004、−0.003，所有 CI95 都跨 0。 | layer17 audio swap 的影响可穿过至少一个后续 block，但没有出现持续时间越长越强的简单轨迹；decision persistent clamp 也没有变成 donor-aligned likelihood。 | **非单调的 layer17 audio influence 是真实的 audio→decision routing，还是 restore/clamp 的 state-distribution 效应？需要 attention/value tracing、audio-token ablation 和多 seed/verbalizer 复核。** |

## 当前证据索引

- [RAVDESS happy/sad 诊断报告](./2026-09-14-ravdess-happy-sad/diagnostic-report.md)
- [factor-controlled parallelism](./2026-09-14-ravdess-happy-sad/parallelism_conditioned.csv)
- [forced-choice 汇总](./2026-09-14-ravdess-happy-sad/forced_choice_summary.csv)
- [audio-token → decision-token transfer 诊断](./2026-09-14-ravdess-happy-sad/decision_transfer.csv)
- [activation patching 汇总](./2026-09-14-ravdess-happy-sad/activation_patch_summary.csv)
- [activation patching 机制表](./2026-09-14-ravdess-happy-sad/activation_patch_mechanism.csv)
- [persistence patching 汇总](./2026-09-14-ravdess-happy-sad/persistence_patch_summary.csv)
- [persistence patching 曲线](./2026-09-14-ravdess-happy-sad/persistence_patch_summary.png)
- [persistence patching 运行记录](./2026-09-14-ravdess-happy-sad/persistence_patch_run.json)

Q5 已通过 forced-choice 和 activation patching 收窄，但仍未完成 routing/readout 的机制判定。Q6 的 layer17 audio-token 局部正向 CE 在 persistence 实验中显示为非单调的短程影响；Q7 因而转向区分真实 audio→decision routing 与 restore/clamp 的 state-distribution 效应，需要结合 activation/value tracing、audio-token ablation、更多 prompt/verbalizer 和 held-out actor/statement 复核。
