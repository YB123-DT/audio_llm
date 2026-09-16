# LLaMA-Omni2-0.5B：两个语料上的固定早期干预复验

2026-09-16。本报告仅回答：在第二个 speech-prefix/Qwen 模型上，固定阻断 blocks 1–6 的 answer-position Attention update，是否提高 block6 的情绪线性可读性，并优于阻断 MLP？

**两个数据集均已完成并通过独立审计。固定 block6 主效应在 CREMA-D 复现，在 RAVDESS 未复现；因此不满足“两个数据集均复现”的事前判据。结果支持有条件的跨模型可迁移性，不能将原结论推广为同族模型中普遍成立。**

## 事前固定的比较

[实验协议](experiment-spec.md)在本模型结果产生前提交（`fa53398`）。沿用既有完整样本：RAVDESS 192 条、24 actors、2 statements、96 happy/sad pairs；CREMA-D 1936 条、88 actors、11 statements、968 pairs。没有据新模型结果重选样本、层或主终点。

三条件为 clean、no-attn、no-MLP。后两者仅在零基 blocks 1–6 阻断最后 prompt/answer 位置相应子层的 residual update；保留另一子层的正常重计算，其他 token 不改。主终点为完整 post-block6 answer residual，次终点为末层之后 terminal RMSNorm 的完整 answer state。

每个 statement 内 leave-one-actor-out；train-only StandardScaler，C=1 LogisticRegression，happy=1。所有条件独立拟合，使用同一折划分。主指标为按 actor×statement 等权平均的 fold AUC，**不是 pooled AUC、不是原 LM 分类准确率**。RAVDESS 每折 92 train / 4 test，CREMA-D 每折 174 train / 2 test；CREMA-D 单折 AUC 只能取 0、0.5、1。

共享 10000 次配对 actor-cluster bootstrap，seed=20260915；固定已经拟合的 probes，不重拟合、不重采样 statements。区间为未多重校正的 95% 区间。每个数据集只有当 ΔA=no-attn−clean、ΔA−M=no-attn−no-MLP 的区间下界都大于零才满足冻结判据；“两个数据集均复现”要求两者都满足。

## 主终点：block6

| 模型 / 数据集 | Clean AUC | No-attn AUC | No-MLP AUC | ΔA [95% CI] | ΔA−M [95% CI] | 冻结判据 |
|---|---:|---:|---:|---|---|---|
| SLAM-Omni / RAVDESS（既有发现结果） | 0.7656 | 0.9219 | 0.7552 | +0.1563 [0.0729, 0.2448] | +0.1667 [0.0677, 0.2865] | 满足 |
| SLAM-Omni / CREMA-D（既有语料确认） | 0.9236 | 0.9618 | 0.9329 | +0.0382 [0.0207, 0.0558] | +0.0289 [0.0124, 0.0455] | 满足 |
| LLaMA-Omni2 / RAVDESS（本轮） | 0.9479 | 0.9167 | 0.9219 | −0.0313 [−0.0885, 0.0208] | −0.0052 [−0.0729, 0.0573] | 不满足 |
| LLaMA-Omni2 / CREMA-D（本轮） | 0.9329 | 0.9618 | 0.9380 | +0.0289 [0.0103, 0.0486] | +0.0238 [0.0093, 0.0382] | 满足 |

SLAM 参考来自[原 RAVDESS 因果实验](../2026-09-16-early-answer-causal/report.md)和[CREMA-D 确认](../2026-09-16-crema-confirmation/report.md)，未重新选择或混合两模型的 probe 预测。不同模型的模板、输入表示和 checkpoint 不同；表中的绝对值差异不是只改变 Attention 机制的受控模型间比较。

RAVDESS 的新模型 clean block6 已有较高可读性。No-attn 的两个主对比均跨零，因而**未复现 SLAM 的固定早期改善**；也不能据此断言不存在任何 Attention 效应或效应确定相反。Statement 01 的 ΔA=+0.0104，statement 02 为 −0.0729；不能用一句的方向代替整体判断。

CREMA-D 两个主对比的区间下界均大于零，满足预定判据。11 句的 ΔA 点估计中 9 正、2 负（TAI −0.0455、TSI −0.0114）；改善不在所有句子上一致。不能只取这个正结果来宣称双数据集复现。

## 次终点：完整 final answer

| 数据集 | Clean AUC | No-attn AUC | No-MLP AUC | ΔA [95% CI] | ΔA−M [95% CI] |
|---|---:|---:|---:|---|---|
| RAVDESS | 0.8542 | 0.9323 | 0.8698 | +0.0781 [0.0052, 0.1510] | +0.0625 [−0.0104, 0.1406] |
| CREMA-D | 0.9132 | 0.9349 | 0.9091 | +0.0217 [0.0031, 0.0403] | +0.0258 [0.0041, 0.0475] |

RAVDESS 最终状态 no-attn 相对 clean 的次要对比为正，但相对 no-MLP 仍不确定。此结果不能替代失败的 block6 主终点，也不等于原模型 LM head 的情绪判断改善。CREMA-D 的两个 final 次要对比也为正；这些未校正的次要区间仍只涉及外部 probe。本轮没有做原生输出评分或生成实验。

## 模型接口与实施核对

使用[官方源码](https://github.com/ictnlp/LLaMA-Omni2/tree/c8afa9061a9c2d2c1919f7293f5492d946869752)的 text-answer branch 和[官方 0.5B checkpoint](https://huggingface.co/ICTNLP/LLaMA-Omni2-0.5B/tree/a16aa9a4ea3f2f363c3db728e8e83ee08e60922c)。源码和权重 revision、SHA256 见[模型资产记录](model-artifacts.json)。24 blocks，hidden size 896；Whisper-large-v3，128-bin mel，30 秒 pad/trim，300 个 speech tokens。配置名为 `linear` 的 projector 实际为 5 帧拼接后的 Linear(6400,2048)→ReLU→Linear(2048,896)。

保留官方 tokenizer 的默认 system prompt，用官方 speech/text preparation 组装 `<speech>` 加固定情绪问题。该 speech+text 问题是显式任务适配，官方 demo 使用 speech-only user message。实际 prefix 长度 352，audio positions 24–323，answer position 351，均为零基索引；没有借用 SLAM 的位置。保存了 rendered chat 和所有 prompt token IDs。

已逐项验证 782 个所需 checkpoint tensors（含 encoder），并在所有复制及权重绑定之后再次核对；297 个 speech-generator tensors 不参与 text branch。冻结 eval、float32、eager attention、batch=1、seed=1234。模型参数未训练，未安装新依赖。外部线性 probe 的拟合只是诊断读出。

运行使用现有 PyTorch 2.5.1 / Transformers 4.57.6。绕过了官方包 initializer 中不兼容的、未用到的 TTS-generation imports，实际 text/encoder/projector 源文件保持不变。首个预定样本上，官方 text wrapper 与直接 backbone 的完整最终 state 和最后位置 logits 差值均为零；clean observer hooks 与无 hook forward 也精确一致。**这个核对只验证当前安装环境内的两条执行路径，不证明与官方 Transformers 4.43.4 / BF16 环境完全等价。**

RAVDESS 的 192 条均完成；9216 次非 answer 位置逐 block 比较全部 bitwise 等于 clean，2304 次 update 阻断检查通过。[独立审计](ravdess/verification.json)从 OOF scores 独立重算 fold AUC 和配对 actor-bootstrap 区间，核对全部样本、折成员及隐藏态完整性，没有复用生产分析的统计函数。审计不重新拟合 probes，也未独立重算次要 pooled AUC / accuracy。

CREMA-D 的 1936 条全部完成，92928 次非 answer 比较、23232 次阻断检查通过；首样本两项官方路径误差和 clean-hook 误差也均为零。[CREMA-D 独立审计](crema-d/verification.json)核对了 11616 条 OOF 记录、170368 行折成员，并独立复算 72 个 AUC 汇总与 72 个配对对比。两语料共拟合 6096 个诊断 probes。实现阶段远端完整测试 140 项通过；新增审计脚本通过编译及两个实际结果集的审计。

实施过程记录：第一次运行因 shell PATH 未包含已有 ffmpeg 而在首条音频前停止；仅补入现有环境 bin，未替换音频处理方法。第二次 RAVDESS 运行在第 43 条之后收到外部 SIGKILL，未找到足以确定原因的日志；从已原子保存并通过 fingerprint 校验的前 24 条检查点恢复，随后完成全部样本。未将中断样本、未完成输出作为统计结果，未据模型指标修改实现。

## 可信范围与交付

这检验的是一个事前固定层窗口在另一同族 speech-prefix/Qwen checkpoint 与实现中的可迁移性。模型借用相关 SLAM-LLM encoder/projector 代码，不能称为跨架构成立。两个语料都已经用于此前研究；本轮新的是模型结果，而非声称所有测试数据或预训练历史都从未见过。

Within-statement probes 的训练和测试共享 statement，本轮不检验分类器迁移到未见文本。

即使某项干预改善 probe，也只说明在指定干预及线性读出下的可读性改变，不证明信息被彻底抹除、lexical semantics 抑制 emotion 或 LM head 已恢复利用。阻断 residual update 是人为干预，可能改变状态分布。当前没有据结果继续找其他窗口、attention paths、校准或插件。

- [RAVDESS 图](ravdess/analysis/model_replication.png)、[摘要](ravdess/analysis/summary.csv)、[配对对比](ravdess/analysis/contrasts.csv)、[逐折预测](ravdess/analysis/oof_predictions.csv)、[提取 provenance](ravdess/intervention_states.json)。
- [CREMA-D 图](crema-d/analysis/model_replication.png)、[摘要](crema-d/analysis/summary.csv)、[配对对比](crema-d/analysis/contrasts.csv)、[逐折预测](crema-d/analysis/oof_predictions.csv)、[提取 provenance](crema-d/intervention_states.json)。
- [统一模型×语料基线表](comparison.csv)、[整体交付核验](verification.json)。
- [复现脚本](reproduce.sh)：`bash reports/2026-09-16-llama-omni2-confirmation/reproduce.sh ravdess 6`；第二次将参数改为 `crema-d 7`。脚本复用已校验的完整状态；模型、官方源码与数据路径见协议。
- [独立审计脚本](../../scripts/verify_model_replication.py)：仓库根目录执行 `python scripts/verify_model_replication.py --dataset ravdess --output /tmp/omni2-ravdess-audit.json`；CREMA-D 使用 `--dataset crema-d`。需本地完整隐藏态及原始 provenance 所指向的文件。
- 完整隐藏态、模型和音频保留本地及 biggpu，不进入 Git；Git 保存代码、冻结协议、样本校验、预测、汇总、图和验证结果。
