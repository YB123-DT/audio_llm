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
| **Q8：candidate token 分解** | 旧的 layer17 sequence-level CE 究竟来自首个 decision token，还是来自两 token verbalizer 的后续 candidate dynamics？ | 单 token `upper_prompt__lower_spaced` 复现 layer17 audio/decision patch；两 token `upper_prompt__upper_spaced` 的 first-token/second-token teacher-forced CE 分解；逐 pair sequence CE 重建。 | 单 token audio CE +0.0456（CI [+0.0111,+0.0800]），decision CE −0.0132（CI 跨 0）。两 token audio token1 +0.0582、token2 +0.0027；decision token1 −0.0617、token2 +0.0027；token 求和在约 5×10⁻⁶ 内重建旧 CE。 | 旧 sequence effect 几乎完全由第一个 candidate token 决定，第二 token 没有独立支持性效应；单 token 复现保留了 audio 正向/decision 不稳定的方向。 | **layer17 audio→decision 的局部效应是否在单 token 下持续存在，并且能否被 restore、decision clamp 与 donor controls 证明为 emotion-specific routing？** |
| **Q9：单 token persistence** | layer17 audio evidence 在首个 candidate token 的读出中能持续多久？一次性 decision patch 是否会被后续 blocks 覆盖？ | 单 token `upper_prompt__lower_spaced` 下运行 `audio_restore_at_18`、`audio_no_restore` 与 `decision_clamp_to_17…23`；9 个 schedule、每格 96 对。 | Audio restore18 +0.0097（CI 跨 0），no-restore +0.0456（CI [+0.0111,+0.0800]）；decision clamp 从 −0.0132 到 +0.0326，但所有 CI95 都跨 0。baseline 跨 schedule 对同一 pair 完全一致，self-patch 最大误差 1.14×10⁻⁵。 | 在首个 candidate token 上，layer17 audio 影响 no-restore 可复现，但 layer18 恢复 target audio 后不再达到描述性显著；decision clamp 没有形成稳定正向 readout。 | **layer17 的正向 audio effect 是否是 emotion-specific，而不是任意 donor state replacement？需要 actor-cluster bootstrap、same-emotion/random donor controls；通过后再做 attention/value path tracing。** |
| **Q10：donor controls** | layer17 的正向 audio effect 是否只是 broad state replacement，而不是 emotion-specific intervention？ | 单 token layer17 audio patch；`matched_swap`、同 emotion/同 statement/repetition/intensity/跨 actor 的 `same_emotion`、全局 `random_donor`；24 actor-cluster bootstrap（10,000 次）。 | matched swap +0.0456，actor bootstrap CI [+0.0082,+0.0862]；same-emotion +0.0142，CI [−0.0120,+0.0391]；random donor +0.0312，CI [−0.0002,+0.0648]。matched runner 与既有结果逐 pair 完全一致。 | 严格 emotion swap 的正向 effect 在 actor 重采样后仍保留，而 same-emotion 与 random donor 不稳定，但尚未直接检验 matched 与 controls 的配对差异，不能排除 broad replacement；random donor 的正均值仍混有 speaker/content 变化。 | **下一步才进入 attention/value path tracing：先定位哪些 head/value 路径把 layer17 audio state 的 emotion-specific 部分带到首个 decision candidate。** |
| **Q11：attention/value path map** | 哪些后续 decoder head 从 decision position 读取 audio span，并携带可能的 emotion-specific value？ | 单 token frozen forward；24 层×14 头 eager attention；记录 decision→audio attention mass、audio-weighted value norm、pair sad−happy delta，并与 layer17 CE 做相关性。 | 后续高 flow 分布在多个 head（如 L18H4、L20H12、L21H5、L22H3/H13、L23H9）；pair delta 与 CE 最大绝对相关性约 0.51，未出现单一集中 head。 | 描述性读写路径是分布式的；相关性只能用于候选选择，不能替代 causal intervention。 | **候选 head 的 donor state 注入是否能单独改变首个 decision token？** |
| **Q12：attention-head causal patch** | 目标 head 的 pre-`o_proj` decision state 是否足以传递 layer17 emotion evidence？ | 对 10 个由同一数据选出的探索性后续候选 head 做 matched donor patch；每个 head 96 对；actor-cluster bootstrap。 | 所有 head 的 bootstrap CI 都跨 0；最大均值 L23H9 +0.0089（CI [−0.0060,+0.0229]），L21H5 +0.0087（[−0.0042,+0.0211]）。 | 没有单 head 的稳定 causal CE；null 不能区分分布式计算、统计功效不足或 off-manifold 风险，也不是 layer17 效应的条件中介检验。 | **若继续机制定位，应测试组合 head/value patch、audio value ablation，并加入多 seed/verbalizer 与多重比较校正；暂不把单 head null 写成“没有 routing”。** |

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
- [tokenwise activation patching 汇总](./2026-09-14-ravdess-happy-sad/single_token_runs/tokenwise_analysis/audio/tokenwise_activation_patch_summary.csv)
- [单 token layer17 patch](./2026-09-14-ravdess-happy-sad/single_token_runs/layer17_audio_tokens_single_token.csv)
- [单 token persistence 汇总](./2026-09-14-ravdess-happy-sad/single_token_persistence_runs/analysis/persistence_patch_summary.csv)
- [donor controls 汇总](./2026-09-14-ravdess-happy-sad/donor_controls/analysis/donor_control_summary.csv)
- [attention/value path map](./2026-09-14-ravdess-happy-sad/attention_value_trace/attention_value_path_summary.csv)
- [attention-head causal patch 汇总](./2026-09-14-ravdess-happy-sad/attention_head_patching/analysis/attention_head_patch_summary.csv)

Q5 已通过 forced-choice 和 activation patching 收窄，但仍未完成 routing/readout 的机制判定。Q6 的 layer17 audio-token 局部正向 CE 在 persistence 实验中显示为非单调的短程影响；Q8 的 tokenwise 分解表明旧 sequence CE 主要落在第一个 candidate token；Q9 显示单 token 下 restore18 会使 audio effect 降到不显著，而 no-restore 可复现；Q10 支持 matched emotion swap 的正向效应，但未建立相对 controls 的特异性；Q11/Q12 的 path map 与目标化 head patch 均未定位到单一 head。若继续，应测试组合 head/value patch、audio value ablation，并扩展 seed/verbalizer 后再作机制结论。

## Q13–Q15：Residual alignment → component mediation → QK/V（2026-09-15）

| 版本 | 核心问题 | 实验与关键结果 | 结论边界与问题变化 |
|---|---|---|---|
| **Q13：logit alignment** | layer17 audio patch 的信号何时开始朝输出 happy/sad 方向变化？ | 保存每层 decision raw h·d 与 final-norm logit lens。Block18 lens Δ=+0.0329，19=+0.0638，20回落至+0.0263，final=+0.0456。 | Patch 信号在 block18 输出已有描述性正向 alignment，非单调。早于18的零是干预位置造成的结构性结果，不是自然 emotion 未编码；自然 final happy−sad 差值 CI 仍跨0。 |
| **Q14：component restore** | 后续 attention 还是 MLP 的 decision contribution 介导该 margin effect？ | 在同一 post17 audio swap 背景恢复 clean target contribution。Joint attention 移除+0.0456，joint MLP移除−0.0010；两者配对差+0.0465，actor CI [+0.0165,+0.0770]。 | 通过预先定义的 QK/V 门槛。但 joint attention 完全回到clean具有结构性原因；逐block attention CI均跨0，尚无唯一 mediator block，也未排除audio位置MLP的作用。 |
| **Q15：QK vs V** | Attention 的影响更依赖 routing 变化还是 projected value content？ | 恢复 decision Q+全部prefix K，与恢复全部prefix V对比。Joint QK移除−0.0236（CI跨0），V移除+0.0635；V−QK配对差+0.0871，CI [+0.0254,+0.1469]。Joint QKV closure误差0。 | 支持当前all-source干预下的value-content dependence；不是audio-only路径或自然emotion-specific机制的证明。不能声称routing无用，也不能把非加性移除量换算为独立中介百分比。当前问题收窄为：哪些source positions的value变化承载这个小margin效应？本轮未扩展执行该问题。 |

完整证据：[本轮报告](./2026-09-15-residual-component/report.md)、[运行前口径](./2026-09-15-residual-component/experiment-spec.md)、[验证记录](./2026-09-15-residual-component/verification.json)。本轮还更正了旧报告的“controls分别显著性即可证明特异性”和“同一数据选择head属于预注册”表述。冻结模型，未训练。

## Q16：Value effect 来自哪些 source positions？（2026-09-15）

| 核心问题 | 实验 | 关键结果 | 问题收窄与限制 |
|---|---|---|---|
| 上轮all-V移除效应主要来自audio values，还是audio先写入非audio位置后间接传播？ | 固定post17 audio patch，blocks18–23按source恢复V：audio、prompt、other、non-audio、decision、结束标记、all；7条件×192样本。 | Audio M=+0.06653，all=+0.06351，prompt=0，other/non-audio=+0.000788（CI跨0）；audio−all=+0.003018 [0.000016,0.005939]；audio−non-audio=+0.065740 [0.021715,0.106847]。All-V与旧结果逐样本精确一致。 | 当前margin effect主要依赖audio-position values，没有大的非audio V中介效应。恢复作用于所有query，可同时阻断audio→other的首次写入，因此未独立证明audio→decision直达，也未排除所有multi-hop。后续问题是decision-query/edge-specific的audio V恢复能否复现该效应；本轮未扩展执行。 |

[Source-specific报告](./2026-09-15-source-value/report.md) · [配对contrasts](./2026-09-15-source-value/analysis/source_contrasts.csv) · [验证](./2026-09-15-source-value/verification.json)。Audio-only略大于all-V及非零非加性项，提示不能将恢复效应相加解释为独立中介百分比。

## Q17：固定audio source，哪类query接收其value效应？（2026-09-15）

| 核心问题 | 实验 | 关键结果 | 结论与边界 |
|---|---|---|---|
| Q16的audio-source dependence主要来自decision直接读取，还是先写入audio/marker等其他query后传播？ | 在post17 audio swap下，blocks18–23的pre-o_proj按query加入A_current·(V_clean−V_current)，仅audio source非零；decision、audio、end-marker、non-decision、all五条件×192样本。 | Decision M=+0.06610，all=+0.06653；差值−0.000428 [−0.002036,+0.001176]。Non-decision M=−0.001845（CI跨0）；decision−non-decision=+0.067945 [+0.024225,+0.109851]。Edge-all与原source restore最大margin差4.29×10⁻⁶。 | 结果接近情况A：当前条件干预的主要margin效应位于后续audio-V→decision-query读取边。未看到大的非decision-query正向移除效应；但被读取的audio values此前仍可能经历传播/重编码，且实验未穷尽K/routing/residual间接路径。没有建立正式等价或可加的中介比例。 |

[Edge restore报告](./2026-09-15-audio-edge/report.md) · [配对contrasts](./2026-09-15-audio-edge/analysis/edge_contrasts.csv) · [验证](./2026-09-15-audio-edge/verification.json)。没有更换source、训练插件或修改checkpoint。更细的block/head定位及跨prompt验证仍未执行。


## Q18–Q20：自然edge信号、gain与block定位（2026-09-15）

| 版本 | 核心问题 | 关键结果 | 问题变化与限制 |
|---|---|---|---|
| Q18：clean necessity | 自然audio-V→decision边是否携带正确emotion差异？ | 无donor的blocks18–23联合移除：DeltaC=+0.04923，actor CI [+0.00794,+0.09497]；59/96对为正。C_H=−0.58060、C_S=−0.62983。 | 弱的正确平均差异与共同SAD偏移并存；不能说自然边没有emotion信息，也不是每个context均一致。仅测试这些层的总下游效应。 |
| Q19：clean gain | 放大自然边能否改善最终readout？ | alpha0/0.5/1/1.5/2的AUC为0.4979/0.5071/0.5212/0.5372/0.5493；全部预测SAD，accuracy均50%。alpha2相对1的AUC变化+0.02810，CI [+0.00271,+0.05111]。 | 相对排序改善有迹象，但alpha2绝对AUC CI [0.4992,0.5982]跨chance，pair gap提升CI跨0。尚不能归因于纯常数bias或声称gain已解决任务。 |
| Q20：block-specific direct edge | donor patch效应在哪些后续block被读取？ | 单block18/19 M=+0.01439/+0.03066，各自CI高于0；累计18–19 M=+0.04407，18–23=+0.06610并精确复现历史joint。 | 18–19已有明显作用；未证明19优于其他block，后续累计增量CI跨0不等于无作用。层效应不可相加，区间未校正，仍限当前prompt/verbalizer/checkpoint。 |

[完整报告](./2026-09-15-clean-edge/report.md) · [实验口径](./2026-09-15-clean-edge/experiment-spec.md) · [验证](./2026-09-15-clean-edge/verification.json)。本轮冻结推理，未拟合bias、阈值或最优gain，未训练。当前问题是：如何提高弱自然信号的稳定性与输出相关性，而不是仅寻找人工patch的传播路径。


## Q21–Q23：真实edge representation与全层杠杆验证（2026-09-15）

| 版本 | 核心问题 | 实验与关键结果 | 问题变化与限制 |
|---|---|---|---|
| Q21：edge decodability | 自然audio→decision update是否跨speaker/text高度可读？ | 分别提取blocks18–23 post-o_proj audio更新。Speaker accuracy66.7–72.9%；statement49.5–57.3%；joint49.5–58.3%。Joint19 pooled AUC0.5917、within-fold0.6615。 | 未出现跨文本80–90%可读性；有一定排序信息但阈值迁移弱。不能用历史不同probe协议直接量化“信息损失”。 |
| Q22：readout alignment | 外部probe方向与LM方向是否脱节？ | 将训练标准化权重还原到residual坐标。Joint平均raw cosine约−0.009至+0.070；block23 LM-axis AUC0.5596，与joint pooled AUC0.5574点估计接近；最终模型AUC0.5212。 | 低cosine不能单独证明alignment failure，高维各向异性及后续变换仍影响解释。当前更像有一定edge信息，但跨context排序/阈值与最终使用都不理想。未定位纯LM-head故障。 |
| Q23：selection验证 | Layer17是孤立峰，还是更广泛窗口的一点？ | 全24层matched双向audio patch，96pairs、actorbootstrap。点估计峰13 CE0.08301，11–13均约0.08；17为0.04555并精确复现。 | 17不是点估计峰或明显孤立尖峰；simultaneous band仅11/13排除0，但13−17配对pointwise CI仍跨0，不能声称唯一峰或13显著更强。17条件路径成立，不等于特有瓶颈。 |

[完整报告](./2026-09-15-edge-representation/report.md) · [验证](./2026-09-15-edge-representation/verification.json)。下一研究问题应区分edge中的context-conditioned读出、阈值迁移及后续计算，而不预设readout alignment failure；本轮未扩展训练或校准实验。


## Q24–Q26：Content offset、阈值迁移与残余geometry（2026-09-15）

| 版本 | 核心问题 | 关键结果 | 问题变化与限制 |
|---|---|---|---|
| Q24：classmean与probe geometry | 两句话的emotion方向是否一致，content shift是否沿readout污染分数？ | Block19 cos(d01,d02)=0.488、cos(w01,w02)=0.174；norm(o)=3.593、norm(d_shared)=0.358；w01·o=−2.761、w02·o=−13.535。 | Offset大且作用于固定probe分数，但方向并非高度一致；独立probe的intercept差不能直接当成同一方向阈值差。高维geometry区间有非线性估计偏差。 |
| Q25：去均值反事实 | 只改statement均值能否恢复跨文本分类？ | Block19 statement57.29%→62.50%，提升CI跨0；block23从49.48%→65.10%，+15.63pp [8.33,22.92]。所有centering的within-fold AUC精确不变。 | 支持offset导致部分阈值失效；测试batch均值属于transductive访问。Pooled AUC改善不是fold内排序改善，未恢复70–80%。 |
| Q26：独立actor校准 | 不用测试actor估计均值或content方向，改善是否保留？ | 其他23actors提供无标签target statement校准；block23 joint50.52%→60.42%，+9.90pp [3.13,17.71]；rank1同为60.42%。Block19改善小。 | 校准隔离测试actor，但见过target statement的无标签分布，不是严格未知文本induction。Offset是部分原因；残余失败不能仅凭rank1结果归因于emotion旋转。全部条件区间未校正。 |

[完整报告](./2026-09-15-content-stability/report.md) · [验证](./2026-09-15-content-stability/verification.json)。复用已有edge向量，无新模型推理、attention路径实验或模型训练。


## Q27：自然edge的均值校正能否修复模型自身readout？（2026-09-15）

| 核心问题 | 实验 | 关键结果 | 更新与限制 |
|---|---|---|---|
| 外部probe的offset修正能否转化为模型自身emotion读出的改善？ | 用其他23actors、无emotion标签估计每statement clean edge均值；在post-o_proj decision行加reference−mu_s，联合18–23及单独19/23。固定单token LM readout。 | 联合AUC0.5212→0.5126，Δ−0.00857 [−0.01617,−0.00033]；gap0.03258→0.02159，变化CI跨0；common shift−0.13683。全部条件192/192 SAD，accuracy50%。单独23 AUC0.5275，变化CI跨0。 | 简单clean-mean offset校正不足以修复自然readout；外部probe的阈值证据不能直接归因为原LM失败主因。已知statement、无标签target校准、固定clean均值及后续动态变化限制解释；未证明offset完全无关或故障只在LM head。 |

[完整报告](./2026-09-15-natural-edge-centering/report.md) · [验证](./2026-09-15-natural-edge-centering/verification.json)。均值按actor隔离，模型冻结、无donor，无新attention路径搜索或模型训练。


## Q28：模块A统一基线与任务接口（2026-09-16）

| 核心问题 | 可比协议与结果 | 判断边界 |
|---|---|---|
| 高上游probe、低edge probe与低输出分数能否定位利用失败？ | 改为同fold/train-only标准化，比较projector、完整final answer state及native head。Joint accuracy/AUC分别90.1%/0.9667、50.5%/0.5393、50.0%/0.5212。完整回答状态joint within-fold AUC0.5990。 | 弱跨文本线性可读性已经出现在完整回答状态，不能只凭局部edge或旧协议差异定位LM head忽略信息。不是信息完全消失或因果层定位证明。 |
| 当前任务/标签接口是否有效？ | 官方配置/输入/评分核对，tied head加载无异常；固定4条明确text cue的相对AUC1.0，但全部预测SAD、accuracy50%，6条生成均不合标签格式。 | 小控制提供相对likelihood方向的正面证据，未验证固定类别阈值与输出格式；音频措辞prompt、两个模板限制解释。不能直接认定音频特有的represented-but-ignored机制。 |

[模块A统一表与判断](./2026-09-16-module-a/report.md) · [验证](./2026-09-16-module-a/verification.json)。排除不兼容旧回答cache，复用当前完整decision state；仅补2条projector抽查和6条接口控制。没有新增patch、校准或插件。


## Q29：模块B统一decoder逐层可读性（2026-09-16）

| 核心问题 | 同协议结果 | 判断与限制 |
|---|---|---|
| Projector的强跨文本情绪信息进入Qwen后，是突然丢失、逐层减弱，还是只留在audio位置？ | 复用Module A的192样本/fold/train-only标准化/C=1 probe，对24个完整block的audio mean与完整answer state统一比较。Joint answer pooled AUC：block0 0.9154→block1 0.7885→block6 0.6033→block23 0.5412；audio的全部层mean within-fold AUC保持0.9583–0.9844。 | Answer早期已可读，随后分阶段、非单调减弱；audio排序信息持续保留。不能说从未到达answer，也不能由pooled曲线定位唯一因果故障层。Audio中后期pooled AUC下降/最终回升与within-fold轨迹不一致，不能直接称为信息丢失/恢复。 |

[统一曲线与判断](./2026-09-16-module-b/report.md) · [验证](./2026-09-16-module-b/verification.json)。旧逐层cache不兼容，补192条各一次冻结decoder forward；Projector/native margin及raw最后answer端点精确核对。实际GPU final RMSNorm与Module A CPU重建的微小差异及其指标变化全部披露；无patch、校准或插件训练。


## Q30：完整answer的within-statement读出（2026-09-16）

| 核心问题 | 匹配协议与结果 | 判断与限制 |
|---|---|---|
| 后期跨文本下降是句内线性可读性也退化，还是只剩content-conditioned code？ | 复用Module B answer states；within/cross均训练其他23actors、92样本，测试同actor同statement4样本。主指标mean actor-fold AUC：within block0 0.9375→block23 0.7083；cross block23 0.6042，G+0.1042 CI[−0.0052,0.2083]。N：within0.7031/cross0.5938，G+0.1094 CI[0.0052,0.2135]。 | 句内下降与额外跨文本损失同时存在。Block23测试01的G+0.2813，测试02的G−0.0729且CI跨零，不能称为两句对称旋转。合并block23 gap不确定，N仅点态未校正区间略正；尚无具体decoder因果机制或direction旋转证明。 |

[曲线与报告](./2026-09-16-answer-content/report.md) · [验证](./2026-09-16-answer-content/verification.json)。仅CPU外部probe，未重新forward；pooled AUC另报，不与折内平均AUC混算gap。


## Q31：Module C 正常forward的Attention/MLP边界（2026-09-16）

| 核心问题 | 观察与匹配协议 | 判断与边界 |
|---|---|---|
| Within可读性下降与content gap增加分别发生在哪种子层之后？ | 192样本每条正常forward一次，保存完整answer的in/post-attn/post-MLP；全部out与Module B精确一致。Block0 W=0.5→0.9375→0.9375。固定blocks1–23合并ΣΔW：Attention−0.2292 CI[−0.3802,−0.0833]，MLP≈0 CI[−0.1146,0.1094]。ΣΔG：Attention−0.0365 CI[−0.2344,0.1615]，MLP+0.1302 CI[−0.0417,0.3021]。 | 净within可读性下降主要位于Attention更新边界；MLP净和零包含正负抵消，不是无作用。合并G归属仍不确定，不能确立“Attention丢情绪、MLP造content dependence”的机制分工。差值和为观察性望远镜恒等式，不是因果mediation比例；未ablate。 |

[Module C图与报告](./2026-09-16-module-c/report.md) · [验证](./2026-09-16-module-c/verification.json)。沿用同actor同测试样本的within/cross probe；共享actor-bootstrap，区间点态未校正；声明并保留statement方向不对称。


## Q32：固定blocks1–6 answer-update因果确认（2026-09-16）

| 核心问题 | 预定干预与主终点 | 判断与停止 |
|---|---|---|
| 早期Attention更新是否对answer线性可读性下降有因果贡献？ | 仅阻断answer行，clean/no-attn/no-MLP；主终点block6 within mean actor-fold AUC为0.7656/0.9219/0.7552。ΔA+0.1563 CI[0.0729,0.2448]；ΔA−M+0.1667 CI[0.0677,0.2865]。所有非answer位置逐层bitwise保持clean。 | 支持指定干预下early answer-position Attention updates对线性可读性损失的因果贡献；未定位lexical/audio/prompt来源。Final状态ΔA+0.0417 CI[0,0.0885]，未形成可靠最终修复证据，未检验原LM行为改善。完成三个固定条件后停止，无追加搜索。 |

[最后一次固定干预报告](./2026-09-16-early-answer-causal/report.md) · [验证](./2026-09-16-early-answer-causal/verification.json)。区间为配对actor-bootstrap、固定拟合probe、未多重校正；statement02主对比子组CI下界触及零，保留泛化限制。


## Q33：独立语料确认固定早期干预（2026-09-16）

| 核心问题 | 冻结协议与结果 | 判断与可信范围 |
|---|---|---|
| 换语料演员与更多句子后，早期 answer-position Attention 更新的可读性效应是否复现？ | 运行前提交协议与样本；CREMA-D XX、88actors、11句、1936音频。沿用blocks1–6、三条件、block6终点。Clean/no-attn/no-MLP W6=0.9236/0.9618/0.9329；ΔA+0.0382 CI[0.0207,0.0558]，ΔA−M+0.0289 CI[0.0124,0.0455]，均满足预定判据。 | 固定新语料样本上的平均干预效应复现，幅度小于RAVDESS；11句中7正、2零、2负，不是逐句一致。Final ΔA−0.0134 CI[−0.0331,0.0052]，没有最终状态改善证据。XX为强度未指定，同checkpoint且within-statement probe，不能推广到其他模型或未见文本的分类器迁移。 |

[独立确认报告](./2026-09-16-crema-confirmation/report.md) · [事前冻结协议](./2026-09-16-crema-confirmation/experiment-spec.md) · [验证](./2026-09-16-crema-confirmation/verification.json)。RAVDESS保留为发现数据，未与确认数据合并；新结果未用于重选窗口或样本。按约定停止本轮。


## Q34：第二个同族模型的双语料固定窗口复验（2026-09-16）

| 核心问题 | 冻结协议与结果 | 判断与可信范围 |
|---|---|---|
| SLAM 上的早期 answer-Attention 可读性效应能否迁移到 LLaMA-Omni2-0.5B？ | 固定 blocks1–6、三条件、完整 block6 answer、同样本/折/probe。RAVDESS clean/no-attn/no-MLP=0.9479/0.9167/0.9219；ΔA−0.0313 CI[−0.0885,0.0208]，ΔA−M−0.0052 CI[−0.0729,0.0573]。CREMA-D=0.9329/0.9618/0.9380；ΔA+0.0289 CI[0.0103,0.0486]，ΔA−M+0.0238 CI[0.0093,0.0382]。 | CREMA-D 满足预定复现判据，RAVDESS 不满足；不能宣称双语料复现或同族模型普遍成立。CREMA 11句9正2负。Final ΔA 在两语料均有正的次要区间，但不能替代 block6 主终点，也不是 LM 原生行为改善。当前 4.57.6/float32 环境内官方路径一致，不等于已验证官方4.43.4/BF16等价性。 |

[双模型双语料报告](./2026-09-16-llama-omni2-confirmation/report.md) · [事前协议](./2026-09-16-llama-omni2-confirmation/experiment-spec.md) · [验证](./2026-09-16-llama-omni2-confirmation/verification.json)。仅新增 checkpoint/实现，仍属 speech-prefix/Qwen 同族；未据结果重选层、样本或主终点，未扩展路径搜索、校准或插件。
