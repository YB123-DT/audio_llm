# Audio source × query edge restore

运行前口径：2026-09-15。沿用冻结模型、192样本/96严格matched pairs、24actors、seed1234、单token lower_spaced readout；post-block17 audio donor swap，blocks18–23 joint restore。

## 局部干预定义

固定source audio=[29,329)。在每个被干预block，使用**当前干预轨迹**的Q/K与attention weights A，对选定query替换audio来源的V贡献：

O_restored[q] = O_current[q] + sum_{s in audio} A_current[q,s] (V_clean_target[s] − V_current[s])。

替换发生在pre-o_proj，每个head均参与；其余query在该block的pre-o_proj output保持不变。不是移植clean target的整段attention output，也不恢复clean Q/K/attention weights。V_clean_target来自同sample无audio patch的原始clean forward，在每层分别缓存。

| 条件 | Query位置 |
|---|---|
| decision | {330} |
| audio | [29,329) |
| end_marker | {329} |
| non_decision | [0,330) |
| all | [0,331) |

前置prompt与输入标记的query不能看后续audio，相关delta应为0。Non-decision包含audio、结束标记和这些结构性零query。当前Q/K保留是局部操作定义；早层edge干预仍会通过正常后续计算改变以后层的Q/K。

## 比较与统计

M = donor_sign*(S_patch−S_restore)，先happy/sad pair内平均，actor-cluster bootstrap10,000次，numpy seed20260915，报告95% CI。比较decision−all、decision−non_decision、audio−non_decision、end_marker−non_decision；记录all−decision−non_decision作为非加性诊断。所有区间未做多重比较校正，不把差值CI跨0当作已证等价，不把移除量换算为中介百分比。

若decision与all接近、indirect较小，支持这些block的audio→decision value读入是主要敏感边；若audio/non-decision恢复较大，则支持query侧audio内部重新计算或其他间接计算的重要性。若两者都有影响，则报告两类计算共同参与与交互，不能相加为独立路径比例。即使decision-only较大，也不排除被读取的audio values本身已经历更早的传播/重编码。

Non-decision条件合并的是audio-V→非decision-query边，不涵盖所有可能的K/routing或residual路径。不能将该条件的零/小效应表述为所有间接计算均不存在。

## 验证

保持SDPA后端；用真实RoPE、GQA和当前mask计算所选query的V贡献，selected-query masking必须保留绝对位置因果性。检查当前A@V可重建原pre-o_proj、self/zero-delta no-op、非选定query局部输出不变、前置query结构性零。全192样本比较edge-all与普通audio-source V restore，并与上一轮逐样本结果核对。保存运行误差、源代码hash、实际source/query范围。若parity失败先修复，不解读结果。无训练或权重修改。
