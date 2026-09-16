# Module A：任务接口静态核对（2026-09-16）

本文件先记录本地官方配置、上游实现与现有诊断代码的静态证据，再报告仅为缺失前提补充的 6 条 text 控制和 2 条 projector 一致性检查。不把评分数值一致性等同于任务接口有效性；没有新增 patch 或训练。

## 已核对的接口

- 官方本地配置 `/data2/yb/paper/SLAM-Omni-0.5B/.hydra/config.yaml`：Qwen2-0.5B（896 维）、Whisper small（768 维）、linear projector、downsample=5、CosyVoice、group size=3、vocabulary=156160、无 latency/layer shift。这些前向相关设置与 `scripts/slam_omni_diagnostics.py:_make_configs` 一致。
- 上游 `SLAM-LLM/examples/s2s/README.md` 的模型定位是语音对话；已下载 variant 的 `DOWNLOAD_SOURCE.md` 标明 single-round English / VoiceAssistant-400K。官方默认系统提示是 `Conduct a spoken conversation with the user.`。这不是经认证的 happy/sad 分类模型，也没有从这些材料获得其能遵循任意分类系统提示的保证。
- `speech_dataset_s2s.py:28` 和 `generate/generate_s2s_online.py` 使用 `<SYSTEM>: {}\n `。现有 `format_prompt` 使用相同包装；`_build_audio_inputs` 使用同样的 input/eot/answer markers、3 路音频 embedding 与 1 路 text embedding 平均。Whisper 30 秒 pad/trim、80 mel、300 projected positions 也与官方 helper 一致。
- 诊断配置关闭 `codec_decode`，保留 group decode adapter；关闭的是生成语音的 codec 部分，不是 text LM head。冻结、路径和 generation token budget 等是诊断设置差异，不能仅据这些差异断言 checkpoint 加载错误。
- `run_hidden` 保存 `llm_decision = hidden[:, -1, :]`；这是完整 prefix 最后 answer marker 的隐藏状态，不是 prompt 系统文字最后 token，也不是局部 attention update。最终 `outputs.hidden_states[-1]` 含模型 final norm，应明确用它作为完整最终回答状态。

## 评分验证覆盖与缺口

`reports/2026-09-15-edge-representation/edge_vectors_run.json` 已记录：hooked clean parity 0；teacher-forced candidate parity 最大误差 1.5259e-5；historical clean parity 0；final norm/LM readout parity 2.8610e-6。因此既有 single-token score 是实现内一致的 LM readout，不需要再全跑音频评分来证明同一件事。

但上游 `examples/s2s/model/slam_model_s2s.py:97` 执行 `load_state_dict(..., strict=False)`，未记录返回的 missing/unexpected keys。此次静态扫描未找到已有完整 key/shape 审计。最低补充是当前 runtime 与 checkpoint key/shape 比较，明确区分被刻意禁用的 codec 和真正参与推理的 encoder/projector/LLM；不应把上述 score parity 误当作权重完整加载证明。

未发现已完成的显式情绪文字线索控制。因此当前结论只能是：输入序列及 scoring 与上游实现兼容；当前任务/标签接口的行为有效性仍需控制。

## 最小支持的文字控制

官方 `examples/s2s/generate/generate_s2s_online.py:152` 的 `generate_from_text` 以及 `decode_config.input_text` 支持 text input；不是把未验证的 chat template 强行套给模型。该实现仍用相同 multimodal markers，将用户文字填入 text stream、audio streams 保持 pad，`audio_mel=None`，`modality_mask` 全 false，并调用同一个 `model.generate`。

建议只补固定、预先指定的小控制：两条 RAVDESS lexical statements 的 text-no-cue；相同文本分别加 `The speaker is happy.` / `The speaker is sad.` 的 text+cue。使用同一分类指令与已固定答案 verbalizers（单 token ` happy`/` sad`），记录原始 margin、分类、最多 32 token 的 greedy 输出和标签 compliance。若必须调整指令以避免音频指代与 text-only 输入矛盾，需明确标记这个 modality wording 差异，不混入统一音频基线表。

控制不使用音频情绪标签来训练或选择 prompt，不探索提示直到成功。它检验明示 emotion 能否进入同一答案映射；成功不能证明音频识别能力，失败也不能定位到特定 attention/readout 机制。

## 补充运行结果（固定 seed=1234）

随后按上述缺口完成最小补充，未改模型配置。`run_module_a_interface_control.py` 直接 AST 提取官方 `get_input_ids`、`get_padded_input`、`generate_from_text`，在其 `stream_generate` 调用点截获完整 batch，再用当前模型评分和 greedy generation。未移植或改写 token 序列。两个测试覆盖固定 6 cases 和官方 builder 的字面 token/模态 mask。

Checkpoint 有 296 keys，runtime 有 963 keys，无 unexpected keys、无 shape mismatch。667 个 missing keys 中，187 个属于外部预训练 Whisper encoder，479 个属于完整 Whisper（含与 encoder 共享的模块）；剩下 `llm.lm_head.weight` 经运行时 storage 地址核对，与 `llm.model.embed_tokens.weight` 确实 tied。已加载 embedding 与 checkpoint 对应权重最大差为 **0**。因此这一次 missing-key 审计没有发现随机初始化的独立 LM head；不能把缺少该别名误报成读出层未加载。Whisper 权重来自显式指定的预训练 small.pt，而非本 SLAM checkpoint。

为允许复用旧 projector cache，另外只对旧 metadata 的第 0/1 条严格 happy/sad 配对做 encoder/projector 前向；使用官方 encoder 方法得到的 projector mean 与历史 cache **两条均逐元素完全一致**。这只是两样本一致性检查，不声称全 192 样本重新提取验证。最终回答状态的旧 cache 问题由统一基线脚本单独处理。

| Statement | Text cue | H−S margin | Forced choice | Greedy compliance |
|---|---|---:|---|---|
| 01 | 无 | −0.332181 | SAD | 无标签 |
| 01 | HAPPY | −0.360524 | SAD | 无标签 |
| 01 | SAD | −0.384799 | SAD | 无标签 |
| 02 | 无 | −0.317719 | SAD | 无标签 |
| 02 | HAPPY | −0.337260 | SAD | 无标签 |
| 02 | SAD | −0.361738 | SAD | 无标签 |

原文固定为 `Kids are talking by the door.` 与 `Dogs are sitting by the door.`，有 cue 时分别拼接 ` The speaker emotion is HAPPY.` / ` The speaker emotion is SAD.`。系统指令保持原始音频分类系统指令完全不变；这确保答案接口一致，但其 “Listen to the speech” wording 与 text-only 模态不完全匹配，结论应局限于这个固定控制。没有尝试其他 prompt，避免针对结果选择接口。

四个有 cue 样本的 forced-choice accuracy 为 **2/4=50%**；pooled ROC-AUC 为 **1.0（仅 4 个控制样本）**，两条 HAPPY cue margin 均高于两条 SAD cue margin。两个 statement 内的 HAPPY−SAD cue margin 差分别为 **+0.024275、+0.024478**，方向均正确，但未跨分类阈值。因此有正面的相对 cue 排序证据，不能称 likelihood 接口完全无效。六条 greedy 输出均无可解析类别，例如 `It's the context of the phrase of the phrase ...`；原始完整输出见 `interface-control.json`。No-cue 无情绪 ground truth，不报告它的 accuracy。仅两个 lexical templates，也不对这个小控制作总体统计显著性推断。

**判断：结构/分数接口与官方实现一致；这 4 个显式 cue 控制支持相对 likelihood 排序方向有效，但固定零阈值类别决策和自由生成格式尚未得到有效验证。应区分“能够对明示 cue 排序”与“能够按当前答案接口正确分类”，不能把整个 likelihood 接口判为无效，也不能将音频 LM 分数弱直接归因为独立的“有信息但不利用”机制。** 这没有证明模型在所有受支持提示下均无法分类，也没有否定音频内部存在 emotion 信息。

运行版本及全部输入 token IDs、scores、checkpoint key 列表、hash、parity、两条 projector 对照均记录在 `interface-control.json`。本地 1 passed/1 skipped（无 torch），远端 2 passed。首次试运行未固定 seed，最终 artifact 已用 seed=1234 重跑覆盖；加入 tied-storage 审计后的同 seed 复跑保留相同固定 cases，没有适应性选择 prompt。
