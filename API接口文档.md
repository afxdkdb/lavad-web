# LAVAD 视频异常检测系统 - API 接口文档

## 基本信息

| 项目 | 内容 |
|------|------|
| 接口地址 | `http://localhost:8000` (本地) / `http://<服务器IP>:8000` (云端) |
| 协议 | HTTP/1.1 |
| 数据格式 | JSON / multipart/form-data |
| 字符编码 | UTF-8 |
| Swagger文档 | http://localhost:8000/docs |
| ReDoc文档 | http://localhost:8000/redoc |

---

## 通用说明

### 响应格式

正常响应直接返回对应的 JSON 对象或流式数据。

错误响应格式：

```json
{
  "detail": "错误描述信息"
}
```

### HTTP状态码

| 状态码 | 含义 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误/格式不支持 |
| 404 | 资源不存在 |
| 422 | 请求体校验失败 (FastAPI 自动校验) |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 (模型未初始化) |

---

## 一、健康检查

### GET /health

检查后端服务健康状态、GPU可用性和模型加载情况。

**请求示例**

```bash
curl -X GET http://localhost:8000/health
```

**响应示例**

```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_count": 4,
  "model_loaded": true
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | `"healthy"` 表示正常，`"degraded"` 表示降级运行 |
| gpu_available | boolean | CUDA是否可用 |
| gpu_count | integer | 可用GPU数量 |
| model_loaded | boolean | 所有必需模型是否已加载 |

---

## 二、模型状态查询

### GET /model_status

获取各模型的详细加载状态。

**请求示例**

```bash
curl -X GET http://localhost:8000/model_status
```

**响应示例**

```json
{
  "blip2_loaded": [
    "blip2-flan-t5-xl",
    "blip2-flan-t5-xl-coco",
    "blip2-flan-t5-xxl",
    "blip2-opt-6.7b",
    "blip2-opt-6.7b-coco"
  ],
  "blip2_available": [
    "/data/jinanyang/models/blip2-flan-t5-xl",
    "/data/jinanyang/models/blip2-flan-t5-xl-coco",
    "/data/jinanyang/models/blip2-flan-t5-xxl",
    "/data/jinanyang/models/blip2-opt-6.7b",
    "/data/jinanyang/models/blip2-opt-6.7b-coco"
  ],
  "qwen_loaded": true,
  "imagebind_loaded": true
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| blip2_loaded | string[] | 当前已加载到显存的BLIP-2模型名称列表 |
| blip2_available | string[] | 服务器上可用的BLIP-2模型完整路径列表 |
| qwen_loaded | boolean | Qwen-7B-Chat是否已加载 |
| imagebind_loaded | boolean | ImageBind是否已加载 |

---

## 三、示例视频列表

### GET /sample_videos

获取预置的示例视频列表，供前端展示使用。

**请求示例**

```bash
curl -X GET http://localhost:8000/sample_videos
```

---

## 四、UCF-Crime评测结果

### GET /demo_results

获取LAVAD系统在UCF-Crime数据集上的性能指标。

**请求示例**

```bash
curl -X GET http://localhost:8000/demo_results
```

**响应示例**

```json
{
  "dataset": "UCF-Crime",
  "roc_auc": 0.7471008133366012,
  "pr_auc": 0.26,
  "description": "LAVAD with Qwen-7B-Chat model trained on UCF-Crime dataset"
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| dataset | string | 评测数据集名称 |
| roc_auc | float | ROC曲线下面积 (帧级检测) |
| pr_auc | float | PR曲线下面积 |
| description | string | 评测说明 |

---

## 五、视频上传

### POST /upload

上传视频文件到服务器（仅上传，不执行分析）。

**请求格式**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| video | file | 是 | 视频文件 (.mp4/.mov/.mkv) |

**请求示例**

```bash
curl -X POST http://localhost:8000/upload \
  -F "video=@/path/to/test_video.mp4"
```

**成功响应**

```json
{
  "video_id": "1234567890",
  "message": "Video test_video.mp4 uploaded successfully",
  "status": "uploaded"
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| video_id | string | 视频唯一标识 |
| message | string | 上传结果描述 |
| status | string | 状态: `"uploaded"` |

**错误响应**

| 状态码 | 原因 |
|--------|------|
| 400 | 不支持的视频格式 |
| 500 | 文件保存失败 |

---

## 六、视频异常检测分析

### POST /analyze

**核心接口**。上传视频文件并执行完整的LAVAD 8步异常检测分析。

**请求格式**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| video | file | 是 | 视频文件 (.mp4/.mov/.mkv) |

**请求示例**

```bash
curl -X POST http://localhost:8000/analyze \
  -F "video=@/path/to/test_video.mp4"
```

> ⚠️ 此接口为长时操作，分析时间取决于视频长度与帧数：
> - 短视频 (<150帧，约5秒): ~3-8分钟
> - 中视频 (150~300帧): ~8-15分钟
> - 长视频 (>300帧): ~15-30分钟
> 建议设置较长的超时时间 (如 1800 秒)。

**成功响应**

```json
{
  "video_id": "abc123def456",
  "video_name": "test_video.mp4",
  "total_frames": 120,
  "fps": 30.0,
  "duration": 150.0,
  "anomaly_frames": [
    {
      "frame_idx": 12,
      "timestamp": 6.0,
      "timestamp_str": "00:06",
      "score": 0.82,
      "caption": "A person is running while another person falls to the ground.",
      "summary": "A sudden altercation occurs between two individuals on a street corner, resulting in one person being pushed to the ground while bystanders look on.",
      "image_path": "/tmp/lavad_xxx/frame_000012.jpg",
      "image_base64": "/9j/4AAQSkZJRg... (Base64编码图像数据)"
    }
  ],
  "normal_frames": 118,
  "abnormal_frames": 2,
  "anomaly_ratio": 0.0167,
  "overall_score": 0.62,
  "max_score": 0.82,
  "threshold": 0.53,
  "mean_score": 0.53,
  "std_score": 0.15,
  "summary": "检测到 2 个异常帧 (占总视频的 1.7%)。最高异常评分: 0.82",
  "top_anomaly_captions": [
    "A sudden altercation occurs between two individuals..."
  ],
  "processing_time": 287.5,
  "steps_completed": [
    "Step 1: Frame Extraction",
    "Step 2: BLIP-2 Captioning (5 models)",
    "Step 3: ImageBind Text Index",
    "Step 4: Cross-modal Caption Cleaning",
    "Step 5: LLM Summaries & Scoring",
    "Step 6: Summary Index Creation",
    "Step 7: Multi-neighbor Score Refinement",
    "Step 8: Final Evaluation"
  ]
}
```

**顶层响应字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| video_id | string | 视频唯一标识 |
| video_name | string | 原始文件名 |
| total_frames | integer | 视频总帧数 |
| fps | float | 视频帧率 |
| duration | float | 视频时长（秒） |
| anomaly_frames | AnomalyFrame[] | 异常帧列表（详见下方） |
| normal_frames | integer | 正常帧数（未超过阈值） |
| abnormal_frames | integer | 异常帧数（超过阈值） |
| anomaly_ratio | float | 异常帧占比 (0~1) |
| overall_score | float | 整体异常评分（所有帧评分均值） |
| max_score | float | 最高帧级异常评分 |
| threshold | float | 判定阈值 |
| mean_score | float | 所有帧评分均值 |
| std_score | float | 所有帧评分标准差 |
| summary | string | 整体分析摘要 |
| top_anomaly_captions | string[] | 评分最高的5个异常帧摘要 |
| processing_time | float | 总处理时间（秒） |
| steps_completed | string[] | 已完成的步骤名称列表 |

**AnomalyFrame 子字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| frame_idx | integer | 帧编号（采样后的clip索引） |
| timestamp | float | 帧对应的时间戳（秒） |
| timestamp_str | string | 格式化时间戳 (MM:SS) |
| score | float | 异常评分 (0~1) |
| caption | string\|null | 清洗后的帧描述文本 |
| summary | string\|null | LLM生成的场景摘要 |
| image_path | string\|null | 服务器上帧图像文件路径 |
| image_base64 | string\|null | 帧图像的Base64编码（可嵌入HTML） |

**错误响应**

| 状态码 | 原因 |
|--------|------|
| 400 | 不支持的视频格式 |
| 500 | 分析过程异常 |
| 503 | 模型未加载，服务不可用 |

---

## 七、获取中间分析结果

### GET /intermediate_results

获取最近一次分析过程中产生的各步骤中间数据，包括原始Captions、清洗后Captions、LLM摘要、原始评分和细化评分。

**请求示例**

```bash
curl -X GET http://localhost:8000/intermediate_results
```

**响应示例**

```json
{
  "captions": {
    "blip2-flan-t5-xl": [
      {"clip_idx": 0, "caption": "A person is walking down a street"},
      {"clip_idx": 1, "caption": "A car is parked on the side of the road"}
    ],
    "blip2-flan-t5-xxl": [
      {"clip_idx": 0, "caption": "A man walking on a sidewalk"},
      {"clip_idx": 1, "caption": "A vehicle stopped near a building"}
    ]
  },
  "cleaned_captions": {
    "blip2-flan-t5-xl": [
      {
        "clip_idx": 0,
        "captions": [
          {"nn_idx": "0", "caption": "A person is walking down a street", "similarity": 1.0}
        ]
      }
    ]
  },
  "step4_summaries": [
    {"clip_idx": 0, "summary": "A person walks along a residential street...", "raw_score": 0.2},
    {"clip_idx": 1, "summary": "A vehicle remains stationary...", "raw_score": 0.1}
  ],
  "step6_refined_scores": [
    {"clip_idx": 0, "refined_score": 0.18, "raw_score": 0.2, "score_delta": -0.02},
    {"clip_idx": 1, "refined_score": 0.12, "raw_score": 0.1, "score_delta": 0.02}
  ]
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| captions | object | `{模型名: [{clip_idx, caption}, ...]}` Step 2原始输出 |
| cleaned_captions | object | `{模型名: [{clip_idx, captions: [{nn_idx, caption, similarity}]}]}` Step 4输出 |
| step4_summaries | object[] | `[{clip_idx, summary, raw_score}]` Step 5 LLM摘要与原始评分 |
| step6_refined_scores | object[] | `[{clip_idx, refined_score, raw_score, score_delta}]` Step 7细化后评分 |

---

## 八、导出异常帧

### POST /export_anomaly_frames

将最近一次分析中检测到的异常帧打包为ZIP文件下载。

**内容说明**: ZIP中包含异常帧JPG图片和 `anomaly_frames_descriptions.txt` 描述文件。

```
anomaly_frames.zip
├── frame_000120_score_0.8200.jpg
├── frame_000145_score_0.7600.jpg
└── anomaly_frames_descriptions.txt
```

**anomaly_frames_descriptions.txt 格式**:
```
帧 #120 | 时间: 60.0s | 评分: 0.8200
  描述: A person is running while another person falls to the ground.
  摘要: A sudden altercation occurs between two individuals...

帧 #145 | 时间: 72.5s | 评分: 0.7600
  描述: Two people fighting on the street corner.
  摘要: Two individuals engage in a physical confrontation near...
```

**请求示例**

```bash
curl -X POST http://localhost:8000/export_anomaly_frames \
  --output anomaly_frames.zip
```

**Python调用示例**

```python
import requests

resp = requests.post("http://localhost:8000/export_anomaly_frames", timeout=120)
with open("anomaly_frames.zip", "wb") as f:
    f.write(resp.content)
```

**JavaScript调用示例**

```javascript
fetch("http://localhost:8000/export_anomaly_frames", { method: "POST" })
  .then(res => res.blob())
  .then(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "anomaly_frames.zip";
    a.click();
  });
```

**响应格式**: `application/zip` 流式文件下载

**Content-Disposition**: `attachment; filename=anomaly_frames.zip`

**错误响应**

| 状态码 | 原因 |
|--------|------|
| 404 | 无可用异常帧，需先执行分析 |
| 503 | 管道未初始化 |

---

## 九、完整调用流程

### 典型前端调用流程

```
1. GET /health                → 确认服务可用
2. GET /model_status          → 确认模型就绪
3. POST /analyze              → 上传视频 + 执行分析
   ├── 等待 5~25 分钟...
   └── 返回 DetectionResult
4. GET /intermediate_results  → 获取详细中间数据
5. POST /export_anomaly_frames → 下载异常帧ZIP
```

### Python SDK 示例

```python
import requests
import json

API = "http://localhost:8000"

# 1. 健康检查
r = requests.get(f"{API}/health")
print(f"服务状态: {r.json()['status']}")

# 2. 执行分析
with open("test_video.mp4", "rb") as f:
    resp = requests.post(f"{API}/analyze", files={"video": f}, timeout=1800)
result = resp.json()

# 3. 查看结果
print(f"视频: {result['video_name']}")
print(f"时长: {result['duration']}s")
print(f"异常帧数: {result['abnormal_frames']}/{result['total_frames']}")
print(f"评分: mean={result['mean_score']:.4f}, max={result['max_score']:.4f}")
print(f"阈值: {result['threshold']:.4f}")

for af in result['anomaly_frames']:
    print(f"  帧 #{af['frame_idx']} @ {af['timestamp']:.1f}s: {af['score']:.4f}")
    print(f"    描述: {af.get('caption', 'N/A')}")

# 4. 获取中间结果
inter = requests.get(f"{API}/intermediate_results").json()
print(f"原始评分数: {len(inter['step4_summaries'])}")
print(f"细化评分数: {len(inter['step6_refined_scores'])}")

# 5. 导出异常帧
export = requests.post(f"{API}/export_anomaly_frames", timeout=120)
with open("anomaly_frames.zip", "wb") as f:
    f.write(export.content)
print("异常帧已导出到 anomaly_frames.zip")
```

---

## 十、新增/变更接口说明

### GET /video_info/{video_id}

获取指定视频的基本信息（当前为占位实现）。

**响应示例**

```json
{
  "video_id": "abc123",
  "status": "placeholder"
}
```

---

## 十一、错误处理建议

| 错误场景 | 处理策略 |
|----------|----------|
| 503 模型未就绪 | 等待30秒后重试，最多3次 |
| 500 分析失败 | 检查后端日志 `/tmp/backend_output.log` |
| 网络超时 | analyze接口建议超时设为1800秒 |
| ZIP导出无内容 | 先调用 analyze 接口确保已有分析结果 |
| 422 校验失败 | 检查请求参数类型与格式是否符合接口定义 |
