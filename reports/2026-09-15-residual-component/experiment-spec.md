# Layer17 audio effect: residual alignment and component mediation

本轮在运行前固定分析口径。冻结同一 checkpoint、seed=1234、192 样本/96 matched pairs，使用 `upper_prompt__lower_spaced` 的单 token happy/sad。层号均为零起点，audio donor patch 位于 block17 输出。

## 1. Residual logit attribution

令 d=W[happy]−W[sad]，分别保存 clean target 和 layer17 matched donor audio patch 后的每层 decision residual。报告 raw h·d 和 final RMSNorm(h)·d（logit lens），两者不能混为实际中间层预测。最终 logit lens 必须复现最终单 token margin。双向 donor-aligned change 对 happy target 取负号，对 sad target 取正号，然后在 pair 内取均值。另保留自然 clean happy−sad 作为参照。

## 2. Attention / MLP component restore

每次均从同一 layer17 audio intervention 出发，在 block18…23 分别将 decision position 的 attention contribution（o_proj 后、residual add 前）或 MLP contribution 恢复为 clean target 值。额外比较全部六层的 attention joint restore 和 MLP joint restore。其余 positions 和 residual skip stream 保持当前干预状态。

令 CE_base 为原 audio patch 效应，CE_restore 为 component restore 后效应，移除量 M=CE_base−CE_restore。M 是条件干预下移除的 margin effect，不是唯一或可加的自然中介比例；负值表示 restore 增强效应。所有条件使用相同 actor 重采样，报告 10,000 次 actor-cluster percentile bootstrap CI；同时直接比较配对的 M_attention−M_mlp。

进入 QK/V 的主门槛：预先指定的 joint attention restore 的 M>0 且其与 joint MLP 的差值>0，两者的 actor-bootstrap 95% CI 下界均大于 0。单 block 的探索性扫描不能单独触发下一阶段；报告其多重比较限制。

结构性边界：post17 decision 本身未被替换。若每个后续 block 的 decision attention contribution 都恢复 clean target，逐 token MLP 也会沿 clean decision 轨迹演化。因此 joint attention restore 应恢复 clean margin，这是结构性 sanity check，不能单独证明独特 mediator；具体 block 仍需逐层结果。此处比较的是写入 decision 的分量，不能排除 audio positions 上 MLP 的上游作用。

## 3. Conditional QK / V separation

仅通过上述门槛后执行。在 attention mediator 的相同干预背景下恢复 clean target 的 decision Q 和全部 prefix K（QK 条件），对照恢复全部 prefix V（V 条件）；固定位置，关闭 KV cache。此实验区分 all-source query/key routing 与 projected value content，不等价于 audio-only 路径隔离。优先联合恢复 blocks18…23，与主门槛对齐，逐 block 仅作探索。

第三阶段开始时加入 QKV joint closure 检查：同时恢复 Q/K/V 应重建 clean decision attention，得到与 attention joint restore 一致的结果。它只验证分离实现，不作为额外机制发现。QK 与 V 同样报告直接配对的移除量差值区间，不能仅比较各自对零的显著性。

## Verification and scope

检查 clean self restore、upstream self audio patch、prefix-only margin 与既有 teacher-forced scorer 一致、final logit lens 与最终 margin 一致。比较本轮 CE 与旧单 token CE。使用 actor-cluster CI，不能把“一个显著、另一个不显著”当成两者有显著差异。旧 head transplant 并未以 layer17 audio patch 为条件，因此不是该效应的 path-specific mediation test。本轮不训练、不改权重；没有新 prompt、seed 或数据集泛化验证。
