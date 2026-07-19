> ⚠️ **<samp>重要版权与商用法律声明</samp>**
>
> 本整合包基于 NVIDIA [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) 模型制作。**本整合包严格禁止商用，仅限学术与研究体验。**
> 
> 其核心依赖的模型权重遵循 [NVIDIA License](https://huggingface.co/nvidia/LocateAnything-3B)，仅限学术和非营利研究用途，**【绝对不允许商业使用】**。用户因违反上游协议产生的任何法律及商业后果，由使用者自行承担。

# LocateAnything-VideoClipper

基于 NVIDIA [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) 模型的**视频/图片智能切片工具**windows整合包，采用三阶段抗误检策略，支持 Gradio WebUI 一键操作。

## 功能特性

- **自然语言驱动**：输入任意提示词（如"黑猫"、"穿红衣服的人"），自动在视频中定位目标
- **批量处理**：指定输入文件夹，自动遍历所有视频文件（mp4/mkv/avi/mov）
- **三阶段抗误检**：置信度过滤 → 时序连续性校验 → 时间轴融合，大幅降低误检
- **动态步长**：盲搜模式快速跳帧，发现目标后自动切换精准追踪
- **无损切片**：调用 FFmpeg 进行 `-c copy` 无损流复制裁剪，画质零损失
- **智能命名**：输出文件自动按 `[视频名]_[提示词]_[时间段]_片段X.mp4` 格式命名
- **Gradio WebUI**：本地浏览器操作，实时日志显示处理进度

## 环境要求

- **操作系统**：Windows 10/11
- **GPU**：NVIDIA 显卡，显存 >= 8GB（推荐 RTX 3060 及以上）
- **CUDA**：CUDA 12.x 驱动

## 快速开始

### 1. 下载模型权重

模型权重约 7.4GB，需从 HuggingFace 下载。在项目根目录执行：

```bash
# 安装 huggingface_hub（如已安装可跳过）
pip install huggingface_hub

# 下载模型到 model_weights 目录
huggingface-cli download nvidia/LocateAnything-3B --local-dir model_weights
```

或使用 Python：

```python
from huggingface_hub import snapshot_download
snapshot_download("nvidia/LocateAnything-3B", local_dir="model_weights")
```

### 2. 安装 Python（可选）

本项目提供嵌入式 Python 方式（双击 `install_deps.bat` + `start.bat`），也可使用自行安装的 Python 3.10+。

### 3. 安装依赖

**方式 A：使用嵌入式 Python（推荐）**

双击 `install_deps.bat`，自动安装所有依赖。

**方式 B：使用自行安装的 Python**

```bash
pip install -r requirements.txt
```

> PyTorch 需根据你的 CUDA 版本单独安装，参考 [pytorch.org](https://pytorch.org/get-started/locally/)。

### 4. 启动

**方式 A：嵌入式 Python**

双击 `start.bat`，浏览器自动打开 `http://localhost:7860`。

**方式 B：自行安装的 Python**

```bash
python app.py
```

## 使用方法

1. 打开浏览器访问 `http://localhost:7860`
2. **输入文件夹路径**：填入包含视频/图片的文件夹路径
3. **提示词**：输入目标描述，多个提示词用逗号分隔（如 `黑猫, 穿红衣服的人, 汽车`）
4. **输出文件夹路径**：留空则默认输出到 `output_clips/`
5. 调整抗误检参数（可选）：
   - **置信度阈值**：越高越严格，过滤越多（默认 0.65）
   - **最小连续帧数**：目标必须连续出现 N 帧才保留（默认 3）
   - **最短输出片段**：低于此秒数的片段丢弃（默认 1s）
   - **盲搜步长**：无目标时每 N 秒抽 1 帧，越大数据越快（默认 5s）
6. 点击 **开始处理**，在日志区域查看实时进度

## 项目结构

```
LocateAnything-VideoClipper/
├── app.py                  # 主入口 + Gradio UI
├── model_worker.py         # 模型加载与推理
├── video_processor.py      # 视频帧提取与处理
├── ffmpeg_cutter.py        # FFmpeg 无损裁剪
├── timeline_merger.py      # 时间轴合并
├── start.bat               # 一键启动（嵌入式 Python）
├── install_deps.bat        # 依赖安装（嵌入式 Python）
├── requirements.txt        # Python 依赖列表
├── embedded_python/        # 嵌入式 Python 3.10（可选）
├── model_weights/          # LocateAnything-3B 模型权重（需下载）
├── input_videos/           # 输入视频放这里
└── output_clips/           # 输出切片目录
```

## 技术细节

- **模型**：NVIDIA LocateAnything-3B（3B 参数视觉语言模型，基于 Qwen2.5-3B-Instruct）
- **推理模式**：Hybrid（MTP 并行解码 + AR 回退，平衡速度与精度）
- **模型架构**：MoonViT 视觉编码器 + Qwen2.5 语言模型 + MLP 多模态投影器
- **训练数据**：12M 图片、138M+ 查询、785M 边界框

## 致谢

- [NVIDIA LocateAnything](https://huggingface.co/nvidia/LocateAnything-3B) — 核心视觉语言模型
- [NVlabs/Eagle](https://github.com/NVlabs/Eagle) — Eagle VLM 模型家族
- [FFmpeg](https://ffmpeg.org/) — 视频处理
- [Gradio](https://www.gradio.app/) — WebUI 框架

## 许可证
## ⚠️ 重要版权与商用法律声明
本项目自身的代码（如抗误检策略、UI 界面）采用 GPL v3 协议开源。然而，本项目核心依赖的后端模型为 NVIDIA LocateAnything-3B，该模型严格遵循 NVIDIA License for Non-Commercial Use。因此，无论本仓库采用何种协议，包含该模型的任何整合包、衍生版均【绝对禁止用于任何商业用途】。用户因违反上游协议产生的法律后果由使用者自行承担。

模型权重遵循 [NVIDIA License](https://huggingface.co/nvidia/LocateAnything-3B/blob/LICENSE)，仅限学术和非营利研究用途，**不允许商业使用**。

本项目工具代码仅供学习交流。

## 联系我们
🛠️ 如果您不想折腾复杂的 PyTorch、CUDA 显卡驱动配置，我们提供开箱即用、双击即运行的 GPU 加速绿色整合包，并提供以下支持：

⚙️ 完整的 Windows 10 一键免安装绿色运行环境

⚡ 针对 Nvidia 10系老旧显卡的 CUDA/cuDNN 极速推理优化

💬 专属的技术支持与定制功能开发

欢迎联系咨询： [andy615.white@gmail.com]
