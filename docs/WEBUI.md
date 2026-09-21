# Web UI 使用说明

本地 UI 由 `webui.py` 提供，默认只监听 `127.0.0.1`，不会自动向外网暴露服务。

启动：

```powershell
python webui.py --open
```

或双击 `start_webui.bat`。默认地址：`http://127.0.0.1:7861`。

## 首次运行

如果项目目录中没有 `displacement.json`，UI 会读取 `displacement.example.json` 并自动生成一份本地 `displacement.json`。这个本地配置已被 `.gitignore` 忽略。

至少需要填写：

1. 源 OBJ 模型。
2. 至少三张同一人物、相近姿态/表情、不同主光方向的参考图（启用结构修正时）。
3. 项目目录内的输出路径，例如 `output/my_run`。
4. 关键点检测 Python、SOAP/head selection 源码目录。
5. 需要最终烘焙/渲染时填写 Blender 可执行文件。

## UI 与主程序关系

UI 只负责编辑配置、启动任务、显示日志和预览。点击运行后，它会把当次配置保存到 `output/webui_jobs/<时间>/config.json`，然后启动：

```text
run_displacement.py --config <job-config>
```

因此 UI 和命令行使用同一套求解代码。

## 常用参数

- `structure.cells`：默认 `[16, 4]`，控制两级低频结构修正。
- `displacement.levels`：默认 `4`，控制最多细分层数。
- `texture_resolution` / `bake_resolution`：当前示例默认 `8192`。
- `structure.maximum_displacement_mm`：低频结构阶段的总位移上限。
- `displacement.maximum_displacement_mm`：结构修正后细节阶段的法向位移上限。

完整调用链见 [`ARCHITECTURE.md`](ARCHITECTURE.md)，算法与输出解释见 [`WORKFLOW.md`](WORKFLOW.md)。
