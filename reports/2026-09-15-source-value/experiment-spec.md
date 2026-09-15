# Source-specific V joint restore

运行前口径，2026-09-15。沿用 post-block17 matched opposite-emotion audio patch，单 token lower_spaced verbalizers、seed1234、192样本/96pairs/24actors。恢复 block18–23 的 clean target v_proj 输出，冻结全部参数。

## 来源分区

由实际 multimodal input 构造推导范围，零起点、右端不包含：

| 条件 | 位置 | 含义 |
|---|---|---|
| audio | [29,329) | 300个projected audio positions |
| prompt | [0,28) | prompt内容与其wrapper，包括prompt input_t/eot |
| other | {28,329,330} | 音频段输入标记、音频结束标记、decision；与audio/prompt互斥 |
| non_audio | prompt ∪ other | 所有非audio位置；与prompt存在包含关系，不得相加 |
| decision | {330} | answer_t/answer_a decision position |
| post_audio_marker | {329} | 音频结束eot/eoa位置 |
| all | [0,331) | 上轮all-V复现 |

前置prompt及position28在因果mask下不能接收后来的audio patch，prompt-only应为结构性零效应。单独报告post_audio_marker与decision，以区分后置非audio中间位置和decision自身的跨层反馈。

## 估计量和比较

M_source = donor_sign*(S_patch−S_restore)，happy target取−1、sad target取+1，先在pair内平均。报告CE_patch、CE_restore、M及24 actor-cluster bootstrap95%CI（10,000次，numpy seed20260915）。直接配对比较audio−all、audio−other、audio−non_audio、other−decision、other−post_audio_marker、prompt−0。另记录all−audio−prompt−other作为非加性诊断，不能把三个恢复效应强行加成总效应。

Audio≈all 用差值及其CI描述；没有事后选择等价界，不把差异不显著自动当作已证明等价。

Audio−all 为主要配对比较；其他 contrasts 和来源拆分为探索性，区间未做多重比较校正。Prompt 的 M 与其 CI 本身即 prompt−0 比较。

## 解释边界

这是source-restricted **所有query** 的V恢复：改写某source的V后，所有能关注它的query都会受影响。若audio-only接近all，只能说明该效应依赖audio-source values，不能仅凭此排除audio→other→decision路径，因为audio-only恢复也可能阻止最初写入other。若post-audio-marker/decision恢复有较大效应，则支持非audio位置参与传播或反馈，但不能由单个条件宣称唯一multi-hop机制。需要query/edge-specific干预才能更严格区分直接decision读取与间接传播；本轮不扩展执行该实验。

## 验证

来源分区互斥完备（audio/prompt/other）；self-restore no-op；全部sample clean/patched scorer parity；前置prompt V clean与patch不变；all-V逐sample复现上轮；保存实际范围、源代码hash、runtime、逐样本分数和配对统计。无训练、无权重修改、无新依赖。
