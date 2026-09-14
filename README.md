# audio_llm

音频大模型研究工作区，统一保存项目代码、诊断报告、实验摘要和模型来源记录。

- GitHub：https://github.com/YB123-DT/audio_llm
- 本地仓库：`/data2/yb/paper/audio_llm`
- 远程仓库：`biggpu:/data2/yb/paper/audio_llm`
- 两端模型目录：`/data2/yb/paper/SLAM-Omni-0.5B`

## 当前模型

已下载并在冻结条件下完成第一轮 RAVDESS happy/sad 诊断的 SLAM-Omni 官方英语单轮检查点（Qwen2-0.5B + Whisper-small，group size 3）。
详见 [下载与校验报告](reports/2026-09-14-slam-omni-download.md) 和 [RAVDESS 诊断报告](reports/2026-09-14-ravdess-happy-sad/diagnostic-report.md)。

当前冻结诊断还包括 matched-pair activation patching；原始交换分数、机制汇总和运行元数据见 [activation patching 结果](reports/2026-09-14-ravdess-happy-sad/activation_patch_summary.csv)。

## 工作约定

后续本项目代码、诊断报告和实验摘要在此仓库维护，每项工作完成验证后提交并推送。
诊断报告放在 `reports/`，来源与校验清单放在 `artifacts/`。
大体积模型权重、数据集、缓存和凭据不进入 Git；通过来源记录与校验清单追踪。
