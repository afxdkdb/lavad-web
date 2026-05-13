<div align="center">

# 🎬 LAVAD

**Language-guided Video Anomaly Detection**

CVPR 2024 · 零样本视频异常检测 · 大语言模型驱动

</div>

---

## 📖 项目简介

本项目是 CVPR 2024 论文 **[Harnessing Large Language Models for Training-free Video Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf)** 的完整复现与 Web 工程化部署。

系统核心思想：**将视频转化为文本描述，利用大语言模型进行"理解 + 评分"**，无需在目标数据集上训练即可检测监控视频中的异常事件。

> 📄 作者：Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, Elisa Ricci

---

## ✨ 核心特性

- ⚡ **零样本检测** — 无需训练数据，即插即用，直接预训练模型推理
- 🧠 **语言驱动** — BLIP-2（图像描述）+ ImageBind（跨模态检索）+ Qwen-7B-Chat（推理评分）
- 🎯 **多模型集成** — 5 种 BLIP-2 模型提升描述多样性
- 🔄 **8 步骤标准化流程** — 帧提取 → Caption → 索引 → 清洗 → LLM评分 → 细化 → 评估
- 🌐 **完整 Web 系统** — FastAPI 后端 + Streamlit 前端，4 标签页交互界面

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────┐
│          Streamlit 前端 (端口 8501)              │
│   上传与分析 │ 检测结果 │ 中间结果 │ 帮助文档     │
└───────────────────┬─────────────────────────────┘
                    │ HTTP REST API
┌───────────────────┴─────────────────────────────┐
│           FastAPI 后端 (端口 8000)                │
│  /health  /analyze  /intermediate_results  ...  │
│             ┌─────────────────────┐             │
│             │   LAVADPipeline     │             │
│             │  8步核心检测引擎     │             │
│             └─────────────────────┘             │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────┴─────────────────────────────┐
│   GPU 0: ImageBind + BLIP-2 (动态加载)           │
│   GPU 1: Qwen-7B-Chat                           │
│   GPU 2-3: 备用                                  │
└─────────────────────────────────────────────────┘
```

---

## 🧩 技术栈

| 类别     | 技术                | 版本                  |
| ------ | ----------------- | ------------------- |
| 后端框架   | FastAPI + Uvicorn | ≥ 0.104             |
| 前端框架   | Streamlit         | ≥ 1.28              |
| 深度学习   | PyTorch           | ≥ 2.0               |
| 图像描述   | BLIP-2 (5 models) | transformers ≥ 4.31 |
| 跨模态检索  | ImageBind         | Meta Research       |
| 大语言模型  | Qwen-7B-Chat      | transformers ≥ 4.31 |
| 向量索引   | FAISS (GPU)       | ≥ 1.7               |
| 视频处理   | OpenCV            | ≥ 4.8               |
| 数据校验   | Pydantic          | ≥ 2.0               |
| Python | 3.10              | -                   |

---

## 🔄 LAVAD 8 步检测流程

| Step | 名称             | 核心操作                         | 关键参数                     |
| :--: | -------------- | ---------------------------- | ------------------------ |
|   1  | 视频帧提取          | OpenCV 解码 → 自适应间隔采样          | 短 4 / 中 8 / 长 16 帧       |
|   2  | BLIP-2 Caption | 5 种模型顺序生成帧描述                 | BATCH_SIZE=4            |
|   3  | 文本索引           | Captions → Embedding → FAISS | dim=1024, 去重             |
|   4  | Caption 清洗     | 视觉 × 文本跨模态检索过滤               | k=1, NUM_SAMPLES=10     |
|   5  | LLM 摘要与评分      | Qwen-7B 摘要 + 0~1 评分         | temp=0.6, top_p=0.9     |
|   6  | 摘要索引           | 去重摘要 → Embedding → FAISS     | dim=1024                 |
|   7  | 分数细化           | 10 邻居指数加权平均                  | k=10                     |
|   8  | 最终评估           | 细化分 vs 阈值 → 异常判定             | threshold=max(mean,0.45) |

### 使用的 5 种 BLIP-2 模型

| 模型                    | 参数规模   | 特点               |
| --------------------- | ------ | ---------------- |
| blip2-flan-t5-xl      | ~7.5B | Flan-T5 解码器，通用场景 |
| blip2-flan-t5-xl-coco | ~7.5B | COCO 微调版，物体识别更准  |
| blip2-flan-t5-xxl     | ~12B  | 参数量最大，描述最准确      |
| blip2-opt-6.7b        | ~6.7B | OPT 解码器，多样性强     |
| blip2-opt-6.7b-coco   | ~6.7B | COCO 微调版 OPT     |

---

## 📊 性能指标

| 指标      | 数值            | 说明                   |
| ------- | ------------- | -------------------- |
| 数据集     | UCF-Crime     | 1900 个真实监控视频，13 类异常  |
| ROC-AUC | **0.3863**    | 实际复现指标（Qwen-7B-Chat） |
| PR-AUC  | **0.0392**    | 实际复现指标               |
| 训练方式    | **Zero-shot** | 未在目标数据集训练/微调         |

### 分析耗时

| 视频规模 | 帧数       | 间隔 | 耗时          |
| ---- | -------- | -- | ----------- |
| 短视频  | < 150    | 4  | ~3-8 min   |
| 中视频  | 150~300 | 8  | ~8-15 min  |
| 长视频  | > 300    | 16 | ~15-30 min |

---

## 🚀 快速开始

### 环境要求

- Python 3.10
- ≥ 2 张 NVIDIA GPU（合计 20GB+ 显存）
- Conda 环境管理

### 安装

```bash
# 创建环境
conda create -n lavad python=3.10
conda activate lavad

# 安装 PyTorch (根据 CUDA 版本调整)
pip install torch>=2.0.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r deploy/requirements.txt

# 配置环境变量
export PYTHONPATH=/path/to/lavad:/path/to/lavad/libs/ImageBind
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
```

### 模型准备

确保以下模型已下载至本地：

```
/path/to/models/
├── blip2-flan-t5-xl/
├── blip2-flan-t5-xl-coco/
├── blip2-flan-t5-xxl/
├── blip2-opt-6.7b/
├── blip2-opt-6.7b-coco/
└── Qwen-7B-Chat/
```

修改 `deploy/service/backend/lavad_pipeline.py` 中的 `BLIP2_MODELS` 列表与 `qwen_model_path` 以匹配实际路径。

### 启动服务

```bash
# 后端（端口 8000）
python -m uvicorn deploy.service.backend.main:app --host 0.0.0.0 --port 8000 --reload

# 前端（端口 8501，新终端）
streamlit run deploy/service/frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

打开浏览器访问：

| 服务               | 地址                            |
| ---------------- | ----------------------------- |
| 前端界面             | http://localhost:8501       |
| API 文档 (Swagger) | http://localhost:8000/docs  |
| API 文档 (ReDoc)   | http://localhost:8000/redoc |

---

## 📡 API 端点

| 方法   | 端点                       | 功能                  |
| ---- | ------------------------ | ------------------- |
| GET  | `/health`                | 健康检查（GPU 状态 + 模型加载） |
| GET  | `/model_status`          | 各模型详细加载状态           |
| GET  | `/demo_results`          | UCF-Crime 数据集性能指标   |
| GET  | `/sample_videos`         | 示例视频列表              |
| POST | `/upload`                | 上传视频文件              |
| POST | `/analyze`               | **核心**：执行完整 8 步异常检测 |
| GET  | `/intermediate_results`  | 获取各步骤中间数据           |
| POST | `/export_anomaly_frames` | 导出异常帧 ZIP（含图片 + 描述） |
| GET  | `/video_info/{video_id}` | 视频信息查询              |

---

## 🖥️ Web 界面

Streamlit 4 标签页布局：

| 标签           | 内容                                 |
| ------------ | ---------------------------------- |
| 🔍 **上传与分析** | 视频上传 + 8 步流程展示                     |
| 📊 **检测结果**  | Plotly 评分折线图 + 异常帧卡片 + 表格 + ZIP 导出 |
| 🔬 **中间结果**  | Step 2/4/5 详情 + 评分对比图              |
| ℹ️ **帮助文档**  | 8 步流程详解 + 论文引用                     |

侧边栏：系统状态指示器 + 模型列表 + 性能指标

---

## 🧪 关键技术参数

| 参数                    | 值               | 说明           |
| --------------------- | --------------- | ------------ |
| FRAME_INTERVAL       | 4 / 8 / 16      | 短/中/长视频帧间隔   |
| BATCH_SIZE           | 4               | BLIP-2 批处理   |
| CLIP_DURATION        | 10s             | 片段窗口长度       |
| INDEX_DIM            | 1024            | FAISS 向量维度   |
| NUM_NEIGHBORS_STEP6 | 10              | 分数细化邻居数      |
| LLM Temperature       | 0.6             | Qwen-7B 推理温度 |
| LLM Top-P             | 0.9             | 核采样概率        |
| 异常阈值                  | max(mean, 0.45) | 动态阈值公式       |
| 评分坍缩保护                | std < 0.005     | 自动回退原始评分     |

---

## ⚙️ 显存管理

|  级别 | 策略                                          |
| :-: | ------------------------------------------- |
|  L1 | BLIP-2 **逐个加载/卸载**，用完即释放                    |
|  L2 | ImageBind & Qwen 在 BLIP-2 处理期间**临时卸载至 CPU** |
|  L3 | 每次卸载后调用 `torch.cuda.empty_cache()` 回收碎片     |

| 模型           |   GPU  | 显存         |
| ------------ | :----: | ---------- |
| ImageBind    |    0   | ~3.4 GB   |
| Qwen-7B-Chat |    1   | ~15 GB    |
| BLIP-2 ×5    | 0 (临时) | 每约 7-12 GB |

---

## 📁 项目结构

```
LAVAD/
├── deploy/                  # Web 部署
│   ├── cloud/               # 云端脚本（启停/检查/测试）
│   ├── requirements.txt     # Python 依赖
│   └── service/
│       ├── backend/         # FastAPI 后端
│       │   ├── main.py              # API 入口（9 端点）
│       │   ├── lavad_pipeline.py    # 8 步核心引擎
│       │   ├── models.py            # Pydantic 数据模型
│       │   └── video_analyzer.py    # 视频分析器
│       └── frontend/        # Streamlit 前端
│           ├── app.py               # 主界面（4 标签页）
│           └── simple_app.py        # 简化版前端
├── src/                     # 论文原始核心代码
│   ├── models/              # 各步骤独立模块
│   │   ├── image_captioner.py
│   │   ├── create_index.py
│   │   ├── image_text_caption_cleaner.py
│   │   ├── llm_anomaly_scorer.py
│   │   └── video_text_score_refiner.py
│   ├── utils/               # 工具函数
│   └── eval.py              # 评估入口
├── libs/
│   ├── ImageBind/           # Meta ImageBind 库
│   └── llama/               # LLaMA 相关库
├── scripts/                 # 命令行流程脚本
├── slurm/                   # SLURM 集群脚本
├── docs/                    # 项目展示站
├── API接口文档.md
├── 项目技术文档.md
└── 云端配置指南.md
```

---

## 📚 文档

| 文档                         | 说明                |
| -------------------------- | ----------------- |
| [API接口文档.md](./API接口文档.md) | 全部 9 个 API 端点详细说明 |
| [项目技术文档.md](./项目技术文档.md)   | 系统架构、核心算法、数据模型    |
| [云端配置指南.md](./云端配置指南.md)   | GPU 部署、环境配置、启停命令  |

---

## 📝 参考文献

[1] Zanella L, Menapace W, Avogaro A, et al. LAVAD: Language-guided Video Anomaly Detection [C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[2] Sultani W, Chen C, Shah M. Real-world Anomaly Detection in Surveillance Videos [C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018: 6479-6488.

[3] Wu P, Liu J, Shi Y, et al. Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision [C]. European Conference on Computer Vision (ECCV), 2020: 322-339.

[4] Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models From Natural Language Supervision [C]. International Conference on Machine Learning (ICML), 2021: 8748-8763.

[5] Li J, Li D, Savarese S, et al. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models [C]. International Conference on Machine Learning (ICML), 2023: 19730-19742.

[6] Girdhar R, El-Nouby A, Liu Z, et al. ImageBind: One Embedding Space To Bind Them All [C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023: 15180-15190.

[7] Bai J, Bai S, Chu Y, et al. Qwen Technical Report [R]. arXiv preprint arXiv:2309.16609, 2023.

[8] Johnson J, Douze M, Jégou H. Billion-scale Similarity Search with GPUs [J]. IEEE Transactions on Big Data, 2019, 7(3): 535-547.

[9] Touvron H, Martin L, Stone K, et al. Llama 2: Open Foundation and Fine-Tuned Chat Models [R]. arXiv preprint arXiv:2307.09288, 2023.

---

## 📄 License

本项目基于 CVPR 2024 LAVAD 论文复现，遵循原始项目的开源协议。详见各子模块（`libs/ImageBind`、`libs/llama`）中的 LICENSE 文件。

---

<div align="center">

<sub>Made with ❤️ · CVPR 2024 · 综合课程设计</sub>

</div>