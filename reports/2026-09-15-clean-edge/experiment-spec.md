# Clean edge necessity / gain，随后 block restore

2026-09-15，运行前口径。沿用192样本、96严格pairs、24actors、固定intensity01、seed1234、lower_spaced单token readout和SDPA冻结模型。

首先自然clean forward无donor patch，对blocks18–23的audio-V→decision-query贡献联合乘alpha=0,0.5,1,1.5,2。在pre-o_proj加(alpha−1)A_current V_current_audio；不改变局部Q/K，不重新归一attention；早层gain可改变之后的decision状态和query，因此是内生轨迹上的联合增益而非固定终点线性插值。

C(x)=S_alpha1(x)−S_alpha0(x)，DeltaC=C_happy−C_sad。报告C两类均值、DeltaC及正向pair比例、actor/statement分组。C为移除这些边的总下游效应，不是静态线性logit attribution。

每个alpha报告accuracy（margin>0为happy，否则sad；单列ties）、ROC-AUC、happy预测比例、matched H−S均值及正向比例、两类margin和相对alpha1平移。以actor-cluster bootstrap10,000次（numpy seed20260915）给CI并做相对alpha1的配对差值；AUC每次重采样后重算，不平均单actor AUC。所有区间未校正，不事后拟合阈值、gain或输出bias，不以不显著证明无信息。

先完成并分析clean，再执行较低优先级的post17 donor背景direct-edge restore：单block18…23，累计18:19…18:23。累计18与单block18相同，仅运行一次；11条件。保留当前A，恢复clean-target V，报告移除量M、累计增量和actor-bootstrap CI，累计终点必须复现上轮decision-only。结果不作可加中介分解，不混同自然与人工干预。

验证：alpha1无操作、scorer parity、attention重建、非decision局部不变、source/位置边界，所有分数finite；clean基线对旧结果、block累计终点对旧干预逐sample验证。无训练、无模型权重修改、无新依赖。
