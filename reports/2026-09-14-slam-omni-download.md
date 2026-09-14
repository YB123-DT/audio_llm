# SLAM-Omni-0.5B 下载与校验记录

日期：2026-09-14

## 来源和版本

- 官方代码文档：https://github.com/X-LANCE/SLAM-LLM/blob/main/examples/s2s/README.md
- 官方英语单轮检查点：https://drive.google.com/drive/folders/1ZmM1h5ZTvS-piuN-msmctmZdi51GWLAu
- 权重文件 ID：`14UxpVbAmuq0wu8ztgsaLVlFDw6HRQAB8`
- 版本：Qwen2-0.5B、Whisper-small、group size 3，英语单轮。

## 两端保存位置

本地与 SSH 别名 `biggpu` 对应服务器：`/data2/yb/paper/SLAM-Omni-0.5B`。
先从官方 Google Drive 下载到本地，再通过 rsync 同步到服务器。

包含 `model.pt`、`.hydra/config.yaml`、`.hydra/hydra.yaml`、`.hydra/overrides.yaml`，
以及本地添加的来源说明和 SHA-256 清单。

## 验证证据

- 权重大小：2,237,875,274 字节（约 2.1 GiB）。
- 权重 SHA-256：`601055c1d7022f076a29f1c22693aa6038083631f54e51204f3099a4ec37249e`。
- 本地使用 Python zipfile 检查 300 个归档条目，CRC 全部通过。
- 本地为所有文件生成 SHA-256 清单；远程 `sha256sum -c SHA256SUMS` 五项全部通过。
- 完整校验清单见 [SHA256SUMS](../artifacts/slam-omni-0.5b/SHA256SUMS)，路径相对于模型目录。

## 下载诊断

gdown 成功下载配置文件，但获取大文件时无法解析 Google Drive 下载确认页。
改用官方 `drive.usercontent.google.com/download` 下载端点并设置 `confirm=t` 后完成下载，
随后验证归档与文件哈希。未将 HTML 确认页误作权重。

## 未验证项

本次仅验证下载文件完整性和两端一致性，没有加载模型或执行推理。
推理仍需 SLAM-LLM 代码与环境、Qwen2-0.5B 相关资源、Whisper small.pt、CosyVoice-300M-SFT，
并覆盖官方配置中原作者的绝对路径。
