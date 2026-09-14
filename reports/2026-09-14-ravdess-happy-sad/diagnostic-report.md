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

为和官方 SLAM-Omni 数据集保持一致，实际 token 序列化为 `<SYSTEM>: {prompt}\n `。推理为 greedy、`decode_text_only=True`、`max_new_tokens=8`；模型和所有参数保持冻结。

结果：[predictions.csv](./predictions.csv)、[output_summary.json](./output_summary.json)

- 192/192 条输出都没有出现独立的 `HAPPY` 或 `SAD` 标签
- `predicted=unknown`：192 条
- 全部样本 accuracy：0/192 = 0.0%
- 可解析样本数：0，因此可解析准确率不定义
- 典型原始输出：`It's be the weather. It you`、`It's be not be not be the`

这说明在这个冻结 checkpoint、这个任务 prompt 和解码设置下，模型没有把答案输出成要求的二分类标签；后面的隐藏态结果仍然可以判断信息是否存在。

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

没有观察到预期的 `PS` 随深度升高的轨迹。这个设置下，跨 lexical content 的 `sad - happy` 方向从 Whisper 早期层开始下降，经过 final/projector 后维持在约 0.1，LLM audio mean 的末层进一步下降到约 0.052。

### Emotion linear probe

speaker-held-out 固定将 actor 19–24 作为测试集；statement-held-out 做 `statement 01 → 02` 和 `02 → 01` 两个方向，表中报告两方向均值。

| 表示位置 | speaker-held-out 最高准确率 | statement-held-out 最高均值 | 末层（speaker / statement） |
|---|---:|---:|---:|
| Whisper audio encoder | 0.938（block 11/final） | 0.896（final） | 0.938 / 0.896 |
| Projector mean | 0.958 | 0.911 | 0.958 / 0.911 |
| LLM audio-token mean | 0.958（layer 0） | 0.911（layer 1） | 0.875 / 0.792 |
| LLM decision state | 0.833（layer 7） | 0.693（layer 0） | 0.771 / 0.516 |
| LLM prompt-last control | 0.500 | 0.500 | 0.500 / 0.500 |

## 当前结论

1. **信息可读出。** Emotion 在 Whisper 后段、projector 以及 LLM audio-token mean 中都能被线性 probe 读出；projector 在本轮两个 split 上分别达到 95.8% 和 91.1%。
2. **没有形成高 parallelism 的跨文本方向。** PS 没有从早期 audio 到 projector/LLM 持续升高，而是从 0.716 降到约 0.1 或更低。因此本轮证据更接近“emotion 可解码，但 `sad-happy` 全局差向量没有被组织成 context-invariant 方向”。
3. **决策位置仍带有部分信息，但不稳定。** `llm_decision` 的 probe 高于随机基线，statement-held-out 末层则接近 0.5；它没有表现出强的跨 context parallelism。
4. **最终文本输出没有使用这些信息。** 192 条生成结果都没有给出可解析的 HAPPY/SAD 标签。在“可读出 + 输出不使用”的意义上，这个 checkpoint/任务设置表现为 information available at internal states but not converted into the requested output。

这是一轮冻结、二分类、固定 intensity 的诊断，不构成因果证明。RAVDESS 的 actor/acoustic 属性仍可能和 emotion 混杂；本轮 audio encoder 使用固定 30 秒 Whisper 输入并做全时间 mean pooling，也可能削弱 token-level 的情绪方向。下一轮若继续，应先做有效音频边界/token-level pooling 和按 actor 的多折 held-out，再考虑任何模型改动或训练。

## 可复现实验入口

代码位于：

- [build_ravdess_manifest.py](../../scripts/build_ravdess_manifest.py)
- [slam_omni_diagnostics.py](../../scripts/slam_omni_diagnostics.py)
- [analyze_slam_omni_diagnostics.py](../../scripts/analyze_slam_omni_diagnostics.py)

远端依赖代码使用官方 SLAM-LLM checkout `/data2/yb/paper/SLAM-LLM`；模型 checkpoint 仍在 `/data2/yb/paper/SLAM-Omni-0.5B/model.pt`，不纳入 Git。模型 SHA-256 为 `601055c1d7022f076a29f1c22693aa6038083631f54e51204f3099a4ec37249e`。
