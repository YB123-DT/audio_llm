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

## 当前证据索引

- [RAVDESS happy/sad 诊断报告](./2026-09-14-ravdess-happy-sad/diagnostic-report.md)
- [factor-controlled parallelism](./2026-09-14-ravdess-happy-sad/parallelism_conditioned.csv)
- [forced-choice 汇总](./2026-09-14-ravdess-happy-sad/forced_choice_summary.csv)
- [audio-token → decision-token transfer 诊断](./2026-09-14-ravdess-happy-sad/decision_transfer.csv)

Q5 仍是开放问题。现有 probe gap 只能定位下一轮因果干预的候选层，不能单独证明 routing failure 或 readout failure；需要结合 activation/value patching、audio-token ablation 和 forced-choice margin 的因果变化来区分两者。
