# audio_llm 项目约定

## 持续授权与交付

用户已于 2026-09-14 指定 https://github.com/YB123-DT/audio_llm 为本项目仓库，
并授权后续本项目代码、诊断报告、实验摘要等成果提交并推送到此仓库。
在验证完成后执行 commit 和 push，无需重复询问普通上传是否继续。

- 本地和 biggpu 的仓库目录均为 `/data2/yb/paper/audio_llm`。
- 模型位于仓库外的 `/data2/yb/paper/SLAM-Omni-0.5B`。
- 仅提交当前任务相关文件，保留其他工作者的修改。
- 修改前检查 Git 状态；同步时仅快进，遇到分叉先检查差异，禁止自动强推。
- 报告放在 `reports/`，明确日期、环境、证据、结论和未验证项。
- 不提交模型权重、数据集、缓存、凭据或含敏感信息的原始日志。
- 公共报告保留复现所需的模型与校验信息，不记录无关账户或基础设施信息。
- 每项任务结束报告验证证据、提交号以及 push 是否成功。

## 提交格式

遵循 Lore 协议：首行说明变更目的；正文记录约束和选择原因。
按需使用原生 Git trailers：`Constraint:`、`Rejected:`、`Confidence:`、
`Scope-risk:`、`Directive:`、`Tested:`、`Not-tested:`。
