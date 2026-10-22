# OCT / OCTA 研究实验归档

阅读入口：[所有实验总览](../docs/experiments/README.md)。

`experiments/` 保存每轮实验代码、配置、数字记录和已渲染图像；`analysis/` 保存基线续训与小波完整训练分析；
`artifacts/` 保存早期验证、完整测试、扩展指标与故障排查；`server_runs/` 保存服务器训练目录的轻量快照。
`flowoct_reproduction/`、`flowoct3d/`、`upstream/` 和 `working/` 分别保留复现工程、三维扩展与历史代码。

模型权重、医学体积、原始标注、完整患者清单与环境不在本归档中。原报告中指向它们的引用有意保留。
源码中的原机器路径可能需要配置；导出的文字记录使用示例路径。源文件和导出文件的哈希分别记录于 `ARCHIVE_MANIFEST.json`。
FlowOCT 派生源码的 MIT 许可见 `THIRD_PARTY_FLOWOCT_LICENSE` 及各对应源码目录。

本次归档为静态快照；仍在运行或仅有阶段记录的组不会被当作最终完成结果。
