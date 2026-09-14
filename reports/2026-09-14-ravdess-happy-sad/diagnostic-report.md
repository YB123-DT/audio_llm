# RAVDESS happy/sad frozen SLAM-Omni diagnosis

日期：2026-09-14

模型：SLAM-Omni-0.5B 官方英语单轮 checkpoint

目标：在不修改模型、不训练插件的条件下，检查 RAVDESS 的 emotion/prosody 信息何时出现、是否形成跨文本稳定方向，以及是否被最终输出使用。

## 数据和配对

只使用 RAVDESS speech audio（filename modality=`03`、channel=`01`），固定 intensity=`01`，并保留 emotion code `03=happy` 与 `04=sad`。配对键为 `(actor, statement, repetition, intensity)`，只有 emotion 不同。

- 192 条音频，96 个严格 matched pairs
- 24 个 actor（01–24）
- 两个 statement（01、02）
- happy/sad 各 96 条
- manifest：[ravdess_happy_sad_intensity01.csv](../../artifacts/ravdess_happy_sad_intensity01.csv)

构建命令：

```bash
python scripts/build_ravdess_manifest.py \
  --ravdess-root /data2/yb/paper/RAVDESS \
  --intensity 01 \
  --output artifacts/ravdess_happy_sad_intensity01.csv
```

## 阶段一：冻结输出

所有样本使用相同 prompt 内容：

```text
Listen to the speech. Is the speaker HAPPY or SAD?
Answer only HAPPY or SAD.
```

为和官方 SLAM-Omni 数据集保持一致，实际 token 序列化为 `<SYSTEM>: {prompt}\n `。推理为 greedy、`decode_text_only=True`；模型和所有参数保持冻结。原始第一轮使用 `max_new_tokens=8`，随后在完全相同的 prompt、manifest、checkpoint 和解码设置下追加 `max_new_tokens=32` 与 `64`，用来区分“输出被截断”和“模型没有按要求输出标签”。

原始 8-token 结果：[predictions.csv](./predictions.csv)、[output_summary.json](./output_summary.json)

- 192/192 条输出都没有出现独立的 `HAPPY` 或 `SAD` 标签
- `predicted=unknown`：192 条
- label compliance/coverage：0/192 = 0.0%
- 可解析样本数：0，因此可解析准确率不定义
- 典型原始输出：`It's be the weather. It you`、`It's be not be not be the`

8 个新 token 的确可能让句子看起来不完整，所以追加了两个更长的生成预算。两组重跑均覆盖同一 192 条样本，分块完整性为 6×32：

| `max_new_tokens` | 样本数 | 含 HAPPY/SAD 的输出 | label coverage | 正确标签数 | 原始输出字符数（min / median / max） |
|---:|---:|---:|---:|---:|---:|
| 8  | 192 | 0 | 0.0% | 0 | 25 / 26 / 28（截断前缀） |
| 32 | 192 | 0 | 0.0% | 0 | 123 / 132 / 148 |
| 64 | 192 | 0 | 0.0% | 0 | 231 / 243 / 299 |

这里的“正确标签数”仅统计输出中出现可解析标签且与 ground truth 相同的条目；三组的二分类准确率都不定义，因为 coverage 为 0%。

32-token 结果见 [predictions_nt32.csv](./predictions_nt32.csv) 和 [output_nt32_summary.json](./output_nt32_summary.json)，64-token 结果见 [predictions_nt64.csv](./predictions_nt64.csv) 和 [output_nt64_summary.json](./output_nt64_summary.json)；对比表见 [output_budget_comparison.csv](./output_budget_comparison.csv)。较长预算只让模型继续生成重复、语义不稳定的文本（例如反复出现 `of a question`、`of a list`），没有出现一个可解析的独立 `HAPPY` 或 `SAD`。因此第一轮的 0% 不能归因于 8-token 截断，但结论仍限定在这个冻结 checkpoint、这个 prompt 和自由生成解码设置；它不是对模型能力的因果证明。

对每个 matched pair 直接比较 happy 与 sad 的原始文本是否完全相同，结果见 [output_sensitivity.csv](./output_sensitivity.csv)、[output_sensitivity_pairs.csv](./output_sensitivity_pairs.csv) 和 [output_sensitivity_summary.json](./output_sensitivity_summary.json)：

| 生成预算 | 完全相同 | 发生变化 | output sensitivity |
|---:|---:|---:|---:|
| 8  | 93/96 | 3/96 | 3.125% |
| 32 | 23/96 | 73/96 | 76.042% |
| 64 | 16/96 | 80/96 | 83.333% |

在原始 8-token 设置中，statement 01 的 96 条输出只有 1 个模板，statement 02 有 2 个模板且众数占 94.8%；这支持“短输出主要被 statement 模板控制”的行为描述。32/64-token 下，少量早期 token 差异会被自回归续写放大，所以 pair sensitivity 上升不能单独解释为 emotion 被使用。

后面的隐藏态结果仍然可以判断 emotion 信息是否存在。

## 阶段二：隐藏态保存

对每条音频只做一次 encoder/projector/LLM forward，保存 mean-pooled vectors 和 metadata。原始 58 MB tensor 文件保留在远端运行目录，不放入 Git：

```text
/data2/yb/audio_llm_runs/ravdess_happy_sad_intensity01_systemprompt/hidden/representations.pt
```

保存内容：

- `audio_encoder`: `[192, 14, 768]`，`conv2`、12 个 Whisper block、`final`
- `projector`: `[192, 896]`，projector 时间维 mean
- `llm_audio_mean`: `[192, 25, 896]`，embedding + 24 个 LLM layer 的 audio-token mean
- `llm_decision`: `[192, 25, 896]`，每层最后一个输入位置（`answer_t` 位置）的 hidden state
- `llm_prompt_last`: `[192, 25, 896]`，每层最后一个系统 prompt token 的 hidden state，作为无 audio 影响的 control

全量提取：192/192，0 errors。对应 metadata 和错误审计文件见 [metadata.csv](./metadata.csv) 与 [errors.csv](./errors.csv)。

## 阶段三：matched-pair 差向量和 linear probe

对每个 pair 计算 `sad - happy`，parallelism 是 96 个差向量之间的 off-diagonal cosine 平均值。完整层级结果见 [parallelism.csv](./parallelism.csv) 与 [probe_accuracy.csv](./probe_accuracy.csv)，图见 [diagnostic_curves.png](./diagnostic_curves.png)。

### Parallelism（PS）

| 表示位置 | 起点 | 最高点 | 末层 |
|---|---:|---:|---:|
| Whisper audio encoder | conv2 0.716 | conv2 0.716 | final 0.155 |
| Projector mean | 0.114 | 0.114 | 0.114 |
| LLM audio-token mean | embedding 0.114 | layer 5 0.125 | layer 23 0.052 |
| LLM decision state | embedding 0.000 | layer 0 0.099 | layer 23 0.007 |
| LLM prompt-last control | 0.000 | 0.000 | 0.000 |

没有观察到预期的 global `PS` 随深度升高的轨迹。这个 global 指标混合了 actor、statement 和 repetition；具体的 factor-controlled 比较见下一节。

### Factor-controlled parallelism

`PS_all` 把 actor、statement 和 repetition 同时放进同一个两两比较集合，因而不能单独归因于 lexical content。基于同一份 `representations.pt`，新增三种控制比较：

$$
PS_{\rm text}=\mathbb E_{a,r}[\cos(\Delta_{a,01,r},\Delta_{a,02,r})]
$$

固定 actor 和 repetition，只改变 statement；共有 48 个比较。`PS_speaker` 固定 statement 和 repetition，在 actor 之间比较；4 个 statement×repetition context 内共有 $4\binom{24}{2}=1104$ 个 actor 对。另加 `PS_repetition`，固定 actor 和 statement，只比较两个 repetition，共 48 个比较。由于 cosine 对称，conditioned CSV 的 `comparisons` 使用 unordered pairs，均值与 ordered off-diagonal 定义相同。

完整数据见 [parallelism_conditioned.csv](./parallelism_conditioned.csv)，曲线见 [parallelism_conditioned.png](./parallelism_conditioned.png)。下表给出同一表示位置的 `PS_all / PS_text / PS_speaker / PS_repetition`：

| 表示位置 | PS_all | PS_text | PS_speaker | PS_repetition |
|---|---:|---:|---:|---:|
| Whisper conv2 | 0.716 | 0.853 | 0.711 | 0.815 |
| Whisper final | 0.155 | 0.377 | 0.143 | 0.473 |
| Projector mean | 0.114 | 0.208 | 0.102 | 0.438 |
| LLM audio mean final | 0.052 | 0.184 | 0.047 | 0.322 |
| LLM decision final | 0.007 | 0.019 | 0.008 | 0.150 |

控制 actor 和 repetition 后，`PS_text` 确实高于对应的 `PS_all`；但它仍没有随模型深度稳定升高。`PS_speaker` 在多数位置接近或低于 global 值，而 `PS_repetition` 在 projector/LLM 表示中更高。这个分解说明 global 低分不能写成“主要由 lexical content 导致”，更合适的结论是 emotion 差向量同时受到 speaker、statement 和 repetition 条件影响。

### Emotion linear probe

speaker-held-out 固定将 actor 19–24 作为测试集；statement-held-out 做 `statement 01 → 02` 和 `02 → 01` 两个方向，表中报告两方向均值。

| 表示位置 | speaker-held-out 最高准确率 | statement-held-out 最高均值 | 末层（speaker / statement） |
|---|---:|---:|---:|
| Whisper audio encoder | 0.938（block 11/final） | 0.896（final） | 0.938 / 0.896 |
| Projector mean | 0.958 | 0.911 | 0.958 / 0.911 |
| LLM audio-token mean | 0.958（layer 0） | 0.911（layer 1） | 0.875 / 0.792 |
| LLM decision state | 0.833（layer 7） | 0.693（layer 0） | 0.771 / 0.516 |
| LLM prompt-last control | 0.500 | 0.500 | 0.500 / 0.500 |

## 阶段四：audio-token → decision-token 的 probe transfer

为了定位 emotion 信息从 `llm_audio_mean` 进入 `llm_decision` 后何时变弱，按相同 LLM layer index 对齐两类表示，并计算：

$$
G_l=A_l-D_l,
\qquad
\Delta D_l=D_l-D_{l-1},
\qquad
R_l=\frac{D_l-0.5}{A_l-0.5}.
$$

其中 $A_l$ 是 audio-token mean 的 probe，$D_l$ 是 decision state 的 probe，$G_l$ 是两者的 readout gap，$\Delta D_l$ 用来找 decision probe 的单层突降，$R_l$ 是相对 chance 的保留率。`embedding` 行作为输入基线保留在 CSV 中，但不作为 LLM 层间 bottleneck；否则会把“尚未经过 LLM decision 位置”的基线差异误报成某一层的传递损失。

结果见 [decision_transfer.csv](./decision_transfer.csv)、[decision_transfer_summary.json](./decision_transfer_summary.json) 和 [decision_transfer.png](./decision_transfer.png)。候选位置如下（仅搜索 `layer_0`–`layer_23`）：

| held-out split | decision probe 最大单层下降 | 突降点 audio probe | LLM 层内最大 audio−decision gap | gap 点 audio / decision / 保留率 |
|---|---|---:|---|---|
| speaker | `layer_13 → layer_14`: 0.7708 → 0.6458（−0.1250） | 0.9167 | `layer_15` | 0.9167 / 0.6042 / 0.250 |
| statement | `layer_21 → layer_22`: 0.5781 → 0.5156（−0.0625） | 0.8281 | `layer_17` | 0.8333 / 0.5156 / 0.047 |

因此，speaker-held-out 上最符合“audio probe 仍高、decision probe 突降”的候选区域是 `layer_14`，而最大层内 gap 出现在 `layer_15`；statement-held-out 的最大 gap 在 `layer_17`，末段 `layer_22` 还有一次较小突降。由于每个 held-out 测试集只有 48 条样本、准确率步长较粗，这些层只能作为下一轮因果干预的候选点。该分析比较的是两个位置上独立训练的线性 readout，属于 probe-retention 代理，不能证明单一层完成了因果的信息传递或删除。

## 阶段五：forced-choice likelihood

为避免依赖模型主动遵循输出格式，对每条音频直接计算候选标签的 teacher-forced sequence likelihood：

$$
S(x)=\log P(\text{happy}\mid x,p)-\log P(\text{sad}\mid x,p)
$$

多 token verbalizer 对所有 token 的 log probability 求和；`S(x)>0` 判为 happy，`S(x)<0` 判为 sad。默认执行 2×2 的 prompt/verbalizer 矩阵，结果见 [forced_choice_summary.csv](./forced_choice_summary.csv)、[forced_choice_summary.json](./forced_choice_summary.json)，逐 pair margin 见 [forced_choice_pair_margins.csv](./forced_choice_pair_margins.csv)。

| 条件 | verbalizer token 数（happy / sad） | forced-choice accuracy | ROC-AUC | `S(happy)-S(sad)>0` | pair margin mean | 全部预测 |
|---|---:|---:|---:|---:|---:|---|
| upper prompt + upper spaced | 2 / 2 | 50.0% | 0.486 | 47/96 (48.96%) | -0.0051 | happy |
| upper prompt + lower spaced | 1 / 1 | 50.0% | 0.521 | 47/96 (48.96%) | 0.0326 | sad |
| lower prompt + upper spaced | 2 / 2 | 50.0% | 0.500 | 44/96 (45.83%) | 0.0163 | happy |
| lower prompt + lower spaced | 1 / 1 | 50.0% | 0.524 | 48/96 (50.00%) | 0.0414 | sad |

四个条件的 ROC-AUC 都接近机会水平，且每个条件对 192 条样本都输出同一个标签；50% accuracy 只是 balanced 数据上的先验偏置。matched-pair margin 也没有稳定地偏向正确方向，因此在当前输入重建和 checkpoint 下，forced-choice likelihood 没有显示可用的 emotion-to-decision 分离。

## 阶段六：matched-pair activation patching

为区分 audio→decision routing 和 downstream readout，本轮对每个严格 matched pair（同 actor、statement、repetition、intensity，仅 emotion 不同）做双向 donor swap。使用 forced-choice 的 `upper_prompt__upper_spaced` 条件（verbalizer 为 ` HAPPY` / ` SAD`，均为 2 个 token）；模型、encoder、projector 和 LLM 全部冻结。

- 在 Qwen decoder 的 `post_decoder_block_output` 上对候选层 `7, 14, 15, 17, 22, 23` 加 hook。
- `audio_tokens` patch 替换位置 `[29:329)` 的 300 个 audio tokens；`decision_token` patch 替换位置 `330`，prefix 长度为 331。
- 每个 target 保留 HAPPY 和 SAD 两个候选的完整 teacher-forced sequence log-likelihood，先在同一 target 上构造 $S(x)=\log P(\mathrm{HAPPY}\mid x,p)-\log P(\mathrm{SAD}\mid x,p)$，再计算
  $\mathrm{CE}_l=\tfrac12[(S(x_H)-S(x_H\leftarrow x_S))+(S(x_S\leftarrow x_H)-S(x_S))]$。
  因而正 CE 表示两侧 target 的 margin 都向 donor emotion 移动。
- 为控制长进程资源，12 个 layer×patch-site 条件分成独立进程，统一 `seed=1234`、`batch-size=1`；每个条件覆盖 96 对。固定 seed 后各条件 baseline candidate scores 的最大差异为 0。

原始 1,152 行及汇总见 [activation_patch.csv](./activation_patch.csv)、[activation_patch_summary.csv](./activation_patch_summary.csv)、[activation_patch_mechanism.csv](./activation_patch_mechanism.csv)、[activation_patch_summary.json](./activation_patch_summary.json)、[activation_patch_run.json](./activation_patch_run.json)，图见 [activation_patch_summary.png](./activation_patch_summary.png)。12 个子任务退出码均为 0，`error` 列为空；self-patch 的最大绝对 likelihood 误差为 $1.53\times10^{-5}$（容差 $10^{-3}$）。

表中 CI 是对 96 个 pair-level CE 的描述性正态近似区间，12 个 layer×patch-site 条件之间未做多重比较校正；“双向一致”是 happy-side 和 sad-side effect 同时为正的比例：

| LLM layer | audio-token CE（95% CI；双向一致） | decision-token CE（95% CI；双向一致） |
|---:|---:|---:|
| `layer_7` | -0.032 [-0.158, +0.093]；42.7% | +0.021 [-0.003, +0.046]；54.2% |
| `layer_14` | -0.001 [-0.078, +0.076]；52.1% | -0.022 [-0.104, +0.061]；45.8% |
| `layer_15` | -0.004 [-0.077, +0.068]；52.1% | -0.012 [-0.094, +0.069]；45.8% |
| `layer_17` | +0.061 [+0.020, +0.102]；59.4% | -0.059 [-0.156, +0.038]；39.6% |
| `layer_22` | -0.001 [-0.014, +0.013]；38.5% | +0.000 [-0.099, +0.099]；46.9% |
| `layer_23` | +0.000 [+0.000, +0.000]；0.0% | -0.004 [-0.104, +0.097]；47.9% |

按“CI95 下界 > 0”的描述性标准，只有 `layer_17/audio_tokens` 达到明确正向（CE=+0.061，CI=[+0.020,+0.102]，双向一致 59.4%）；同层 `decision_token` 为 −0.059，CI 跨过 0。`layer_7` 的 decision-token 均值为 +0.021 但 CI 下界略低于 0，不能当作有效 patch；`layer_23` 的 audio-token CE 精确为 0，符合最终 block 后没有 downstream token mixing 的结构控制。其余层的两类 patch 也未达到 CI 下界 > 0。由于未做多重比较校正，这个 layer-17 信号应视为待复核的假设线索。

因此，这一轮没有得到“所有层都存在 routing failure”或“decoder readout 已被证明可用”的结论。最具体的信号是 `layer_17` audio-token state 的 donor-aligned causal influence，而直接替换同层 decision state 没有稳定地产生相同方向；这与分布式 token computation 或 decision patch 的 off-manifold 风险相一致。该解释仍受单一 prompt/verbalizer、固定 30 秒音频边界和描述性 CI 限制，下一步应在更多 verbalizer/seed、audio-token ablation 和 value/activation tracing 上复核。

## 当前结论

1. **信息可读出。** Emotion 在 Whisper 后段、projector 以及 LLM audio-token mean 中都能被线性 probe 读出；projector 在本轮两个 split 上分别达到 95.8% 和 91.1%。
2. **global PS 不能单独归因于 lexical content。** 控制 actor 和 repetition 后，`PS_text` 高于 `PS_all`，但仍没有形成随深度上升的稳定轨迹；`PS_speaker` 与 `PS_repetition` 表明 speaker 和 repetition 也会改变差向量方向。当前更稳妥的表述是 emotion 差向量具有明显的 context dependence。
3. **audio-token 到 decision-token 的候选瓶颈已定位。** probe-retention 代理在 speaker split 的 `layer_14` 出现最大单层下降（−0.1250），`layer_15` 出现最大 LLM 层内 gap（0.3125）；statement split 的最大 gap 在 `layer_17`，`layer_22` 有另一处突降。它们是下一轮 activation patching、token ablation 或 causal tracing 的候选层，不是已经证明的因果瓶颈。
4. **决策位置仍带有部分信息，但不稳定。** `llm_decision` 的 probe 高于随机基线，statement-held-out 末层则接近 0.5；它没有表现出强的跨 context parallelism。
5. **free generation 没有完成标签任务。** 在 `max_new_tokens=8/32/64` 三个预算下，192 条生成结果都没有给出可解析的 HAPPY/SAD 标签；这对应的是 label compliance/coverage=0%，不是一个有定义的二分类 accuracy。原始 8-token 中 93/96 个 pair 的文本完全相同，长预算下差异增多但仍是重复续写。
6. **forced-choice 也没有显示可用的情绪决策信号。** 对每条音频计算候选序列的 teacher-forced log-likelihood，并用 $S(x)=\log P(\text{happy}\mid x,p)-\log P(\text{sad}\mid x,p)$ 决策。4 个 prompt/verbalizer 条件的 accuracy 都是 50.0%，ROC-AUC 为 0.486–0.524，matched-pair 的 `S(happy)-S(sad)` 正方向比例为 0.458–0.500；结果见 [forced_choice_summary.csv](./forced_choice_summary.csv)、[forced_choice_summary.json](./forced_choice_summary.json) 和 [forced_choice_pair_margins.csv](./forced_choice_pair_margins.csv)。这说明当前设置下 final likelihood readout 没有把内部可读出的 emotion 分离成稳定的 happy/sad 选择。

7. **activation patching 给出局部因果线索。** 在固定 `upper_prompt__upper_spaced`、seed=1234 的 12 个条件中，只有 `layer_17/audio_tokens` 的 CE 通过描述性 CI95 下界 >0（+0.061，[+0.020,+0.102]）；同层 `decision_token` 未通过，`layer_23/audio_tokens` 精确为 0。结果支持“layer 17 的 audio state 能影响后续 margin，但直接 decision-state 替换不稳定”的局部现象，不能单独证明自然 routing failure 或 downstream readout failure。

这是一轮冻结、二分类、固定 intensity 的诊断。forced-choice 结果只说明当前 checkpoint、手工重建配置、prompt/verbalizer 和自由音频输入下没有超过机会水平，不能写成“decoder 已被证明完全不使用 emotion”；instruction-following 能力不足或 checkpoint/config 加载问题仍需单独排除。activation patching 提供的是给定层、位置和 verbalizer 下的干预性因果效应，不等同于自然 routing 已被证明。RAVDESS 的 actor/acoustic 属性也可能和 emotion 混杂；本轮 audio encoder 使用固定 30 秒 Whisper 输入并做全时间 mean pooling，也可能削弱 token-level 的情绪方向。下一轮应先验证官方 checkpoint/config 加载和 text-only likelihood 校准，再做有效音频边界/token-level pooling 与按 actor 的多折 held-out。

## 可复现实验入口

代码位于：

- [build_ravdess_manifest.py](../../scripts/build_ravdess_manifest.py)
- [slam_omni_diagnostics.py](../../scripts/slam_omni_diagnostics.py)
- [analyze_slam_omni_diagnostics.py](../../scripts/analyze_slam_omni_diagnostics.py)
- [analyze_output_behavior.py](../../scripts/analyze_output_behavior.py)
- [analyze_forced_choice.py](../../scripts/analyze_forced_choice.py)
- [analyze_decision_transfer.py](../../scripts/analyze_decision_transfer.py)
- [run_activation_patching.py](../../scripts/run_activation_patching.py)
- [analyze_activation_patching.py](../../scripts/analyze_activation_patching.py)

forced-choice 运行阶段为 `slam_omni_diagnostics.py forced-choice`，默认执行 2×2 prompt/verbalizer 矩阵；候选标签按完整 token sequence 做 teacher-forced likelihood，不能用单 token argmax 替代。activation patching 的单条件入口如下（完整网格按 layer×patch-kind 拆分执行）：

```bash
python scripts/run_activation_patching.py \
  --manifest artifacts/ravdess_happy_sad_intensity01.csv \
  --ravdess-root /data2/yb/paper/RAVDESS \
  --slam-llm-root /data2/yb/paper/SLAM-LLM \
  --qwen-path /data2/yb/paper_runtime_models/Qwen2-0.5B \
  --whisper-path /data2/yb/paper_runtime_models/whisper/small.pt \
  --checkpoint /data2/yb/paper/SLAM-Omni-0.5B/model.pt \
  --output-csv /tmp/activation_patch_layer17_audio_tokens.csv \
  --condition upper_prompt__upper_spaced --layer 17 \
  --patch-kind audio_tokens --batch-size 1 --seed 1234 --device cuda:0

python scripts/analyze_activation_patching.py \
  --input-csv reports/2026-09-14-ravdess-happy-sad/activation_patch.csv \
  --output-dir reports/2026-09-14-ravdess-happy-sad
```

probe transfer 汇总命令为：

```bash
python scripts/analyze_decision_transfer.py \
  --probe-accuracy reports/2026-09-14-ravdess-happy-sad/probe_accuracy.csv \
  --output-dir reports/2026-09-14-ravdess-happy-sad
```

远端依赖代码使用官方 SLAM-LLM checkout `/data2/yb/paper/SLAM-LLM`；模型 checkpoint 仍在 `/data2/yb/paper/SLAM-Omni-0.5B/model.pt`，不纳入 Git。模型 SHA-256 为 `601055c1d7022f076a29f1c22693aa6038083631f54e51204f3099a4ec37249e`。
