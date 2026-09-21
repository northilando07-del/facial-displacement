# Multi-view Facial Displacement Reconstruction

一个研究型的人脸几何重建/位移生成工作流：输入带 UV 的 OBJ 头模，以及同一人物、相近姿态与表情、不同主光方向的多张参考图，先修正低频脸型与五官体块，再提取多尺度光度细节，最后输出真实高模、灰度位移、XYZ 向量位移和烘焙法线。

当前默认流程是：**16格结构修正 → 4格结构修正 → 1格基础残差 → 最多4级自适应细分 → 8K 贴图输出**。

> 当前仍属于研究原型。它依赖外部 SOAP/head selection 环境完成源模型绑定，并依赖 Blender 完成最终烘焙与对照渲染。

## 快速开始

安装 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

启动本地 UI：

```powershell
python webui.py --open
```

也可以双击 `start_webui.bat`。

首次启动时，如果仓库里没有 `displacement.json`，UI 会从 `displacement.example.json` 自动生成一份本地配置。`displacement.json` 已加入 `.gitignore`，可以安全填写你的模型、图片、SOAP 环境和 Blender 绝对路径。

命令行运行：

```powershell
Copy-Item displacement.example.json displacement.json
# 编辑 displacement.json 后：
python run_displacement.py --config displacement.json
```

仅生成几何、不启动 Blender 烘焙：

```powershell
python run_displacement.py --config displacement.json --geometry-only
```

强制重建 FULL 细节证据：

```powershell
python run_displacement.py --config displacement.json --rebuild-evidence
```

## 必要配置

| 字段 | 含义 |
| --- | --- |
| `mesh` | 带 UV 的源 OBJ 头模 |
| `references` | 同一人物/姿态/表情、不同主光方向的参考图；结构精修启用时至少 3 张 |
| `output` | 输出目录，必须位于项目目录内，例如 `output/my_run` |
| `automatic_inputs.detector_python` | 能运行人脸关键点检测脚本的 Python 可执行文件 |
| `automatic_inputs.soap_root` | SOAP/head selection 源码目录，需包含 `headlab_geometry.py` |
| `blender` | Blender 可执行文件；只跑 `--geometry-only` 时不需要 |

默认示例配置见 [`displacement.example.json`](displacement.example.json)。

## 工作流

```text
OBJ + 多光照参考图
        │
        ├─ 关键点检测 / SOAP 源模型绑定
        │
        ├─ 16格 -> 4格：低频 XYZ 脸型/五官结构修正
        │
        ├─ 在修正模型上重新投影并提取 FULL 多尺度法线证据
        │
        ├─ 1格基础残差 + 自适应三角细分
        │
        ├─ 灰度位移 / XYZ 向量位移导出
        │
        └─ Blender：最终高模、法线烘焙、对照渲染
```

更完整的模块关系、数据契约和调用链见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。当前 88 版工作流细节见 [`docs/WORKFLOW.md`](docs/WORKFLOW.md)。

## 关键输出

| 路径 | 内容 |
| --- | --- |
| `structure/coarse_16.obj` | 第一层低频结构修正 |
| `structure/coarse_04.obj` | 第二层低频结构修正 |
| `levels/level_00.obj` ... | 基础残差与自适应细分层 |
| `head_displaced.obj` | 最终高模 |
| `maps/<材质>/vector_displacement_16.png` | 相对原始模型的 XYZ 总向量位移 |
| `maps/<材质>/displacement_16.png` | 相对精修基础模型的灰度细节位移 |
| `maps/<材质>/normal_opengl16.png` | 从最终高模烘焙回源低模的切线法线 |
| `report.json` | 输入哈希、求解阶段、误差、位移量和输出摘要 |

生成结果、缓存和本地备份都已从 Git 跟踪中排除。

## 仓库结构

```text
.
├─ run_displacement.py          主工作流编排
├─ displacement_structure.py   低频 XYZ 结构修正
├─ displacement_geometry.py    位移求解与自适应细分
├─ prepare_inputs.py            参考图检测/源模型绑定入口
├─ run_layered.py               FULL 多尺度细节证据提取
├─ normal_*.py                  光照、法线、语义层、几何场基础模块
├─ export_displacement_maps.py  位移贴图导出
├─ render_displacement.py       Blender 烘焙与渲染
├─ webui.py + webui/            本地 Web UI
├─ tests/                       单元/算法回归测试
├─ displacement.example.json    可提交示例配置
└─ docs/                        架构、工作流与 UI 文档
```

## 依赖

`requirements.txt` 目前只有：

- NumPy
- SciPy
- Pillow

Blender 的 `bpy` 由 Blender 自带，不通过 pip 安装。SOAP/head selection 和关键点检测环境属于外部依赖，需要在本机配置中指定。

## 测试

从项目根目录运行：

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

这些测试覆盖法线/多光照层、几何场、参考图融合、语义层和位移细分等核心算法。Blender 渲染与外部 SOAP/FAN 环境属于集成依赖，不在纯 Python 单测中自动启动。

## 已知边界

- 结构恢复依赖多光照、关键点、轮廓和先验网格，不是无先验的真实 3D 扫描。
- 光源是估计值，输入姿态/表情差异过大会降低可靠性。
- 自适应细分是线性三角细分，不是 Catmull-Clark。
- 8K 代表输出贴图采样，不代表低分辨率参考图会产生额外真实信息。

仓库目前没有附带许可证；公开发布前请根据你希望的使用范围自行选择许可证。
