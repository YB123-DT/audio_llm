# 模块A：统一读取基线与任务接口判断

2026-09-16。终点是一张可比表与一个判断；本轮没有新增patch、校准、插件或逐层路径实验。

## 统一基线表

每格为 **accuracy / pooled OOF ROC-AUC**。192条同样的音频、96matched pairs、24actors；两种外部probe使用完全相同的fold、训练集StandardScaler、C=1 logistic regression、happy=1与指标。回答状态是**完整final answer位置、final RMSNorm后、LM head前的向量**，包含全部residual信息，不是audio-attention update。

| 读取对象与读出方式 | Speaker-held-out | Statement-held-out | Joint speaker＋statement |
|---|---:|---:|---:|
| Projector时间均值 → 外部linear probe | 90.1% / 0.9647 | 93.2% / 0.9882 | 90.1% / 0.9667 |
| 完整最终回答状态 → 外部linear probe | 69.3% / 0.7317 | 50.5% / 0.5442 | 50.5% / 0.5393 |
| 原模型LM head → 固定候选likelihood | 50.0% / 0.5212 | 50.0% / 0.5212 | 50.0% / 0.5212 |

Speaker为leave-one-actor-out，statement为01→02及反向，joint训练集同时排除测试actor和测试statement。Native LM不拟合参数，三列复用相同分数，不能当作三次独立训练验证。单token候选为` happy`6247与` sad`12421，原始分类prompt不变。

Pooled OOF分数来自不同拟合模型，可能受fold间offset/scale影响，因此同时保存各fold指标。完整回答状态的statement/joint **平均within-fold AUC分别为0.5918/0.5990**：有弱的跨文本排序信息，不能将50.5% accuracy写成信息完全消失。Projector对应within-fold AUC为0.9881/0.9792。

全部CI与逐样本结果见CSV。代表性joint结果：projector accuracy90.10% [84.38%,95.31%]、AUC0.9667 [0.9360,0.9905]；完整回答状态accuracy50.52% [50.00%,51.56%]、AUC0.5393 [0.5041,0.5748]；native AUC0.5212 [0.4761,0.5676]。区间为10,000次actor-cluster bootstrap，条件于已拟合fold，未重拟合、未校正多指标比较。探针表现不能单独证明信息被物理删除或定位因果层。

## 唯一补充前提：当前任务接口

官方本地Hydra配置、四流输入包装、最终候选评分已核对。Checkpoint不存在unexpected keys或shape mismatch。`lm_head.weight`未作为独立key保存，但其与已正确加载的embedding共享存储，权重差0；其余missing keys来自单独预训练加载的Whisper。当前审计没有发现独立LM head未加载的证据。

仅补6条固定官方text-input控制：两句RAVDESS文本分别不加cue、明确HAPPY cue、明确SAD cue。保留原分类system prompt和答案格式，无prompt搜索。

- 四条明确cue的 **pooled AUC=1.0（n=4）**；两对HAPPY−SAD cue margin差为+0.02428、+0.02448。相对likelihood排序有最小正对照证据，不能说标签likelihood接口完全无效。
- 但四条全部预测SAD，固定零阈值accuracy2/4；六条greedy生成的label compliance为0/6。No-cue没有emotion ground truth，不计算accuracy。
- 这只是两个模板、四个显式cue，不是泛化准确率估计；保持的原prompt含“Listen to the speech”，与text-only输入的措辞并不完全匹配，因此不能推断模型在所有受支持提示下都无法分类，也不能把控制当成音频识别能力。

## 一个判断

**统一协议下，projector中的跨文本线性可读性很强，但在完整回答状态中已经明显较弱；现有证据不支持“完整回答状态保留了强而稳定的emotion信息，只是LM head忽略它”。** 同时，文字cue对相对likelihood有响应，但固定分类阈值及输出格式未通过控制，故当前结果也不足以认定音频特有的“represented but ignored”机制。结论限定为可读性比较与当前任务接口诊断，不进一步定位routing/readout因果失败。

## 复用与兼容性

早期完整回答隐藏态直接乘当前LM方向，无法复现当前分数，最大误差0.75870，已排除。旧提取早于seed1234固定，但本轮未证明seed就是差异根因，不能仅靠这一历史线索归因。

本轮用已经保存的**当前clean完整pre-finalnorm decision state**和finalnorm weight重建最终回答状态（官方epsilon=1e−6），无需模型重跑。192条最终LM margin最大重建误差2.42e−6。文件名含edge，但读取字段为`final_decision_pre_norm`，没有拿`edge_updates`替代完整回答状态。

Projector复用早期cache：通过sample_id与representation_index显式对齐，并对首对happy/sad各做一次官方encoder/projector前向，二者逐元素差均为0；这只是2样本数值抽查，不声称重新提取验证了192条。源文件hash与兼容性记录全部保存。

## 证据与复现

- [统一表及fold内AUC](./baseline_analysis/baseline-table.md)、[完整指标与CI](./baseline_analysis/summary.csv)
- [划分成员](./baseline_analysis/fold_membership.csv)、[逐fold指标](./baseline_analysis/fold_metrics.csv)、[OOF分数](./baseline_analysis/oof_predictions.csv)、[协议与parity](./baseline_analysis/analysis.json)
- [来源兼容性](./baseline_provenance.json)、[接口核对](./interface-audit.md)、[文字cue原始结果](./interface-control.json)
- [运行口径](./experiment-spec.md)、[复现命令](./reproduce.sh)、[汇总验证](./verification.json)

二进制表示文件保留本地/远端运行存储，不提交Git；代码、表格、评分记录与报告提交。未修改模型权重或训练插件。

验证：远端78项测试通过；本地55通过、21因可选运行时依赖缺失跳过。语法编译、复现脚本语法、来源ID/index对齐、全部OOF覆盖与完整回答状态LM读出一致性均通过。
