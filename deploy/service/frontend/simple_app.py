from flask import Flask, render_template_string, request, jsonify
import os

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LAVAD - 视频异常检测系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #1E88E5, #1565C0); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 30px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        .card { background: white; border-radius: 10px; padding: 30px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .upload-area { border: 3px dashed #1E88E5; border-radius: 10px; padding: 50px; text-align: center; cursor: pointer; transition: all 0.3s; }
        .upload-area:hover { background: #e3f2fd; border-color: #1565C0; }
        .upload-area input { display: none; }
        .upload-icon { font-size: 4rem; margin-bottom: 20px; }
        .btn { background: #1E88E5; color: white; border: none; padding: 15px 40px; font-size: 1.1rem; border-radius: 5px; cursor: pointer; transition: background 0.3s; }
        .btn:hover { background: #1565C0; }
        .btn:disabled { background: #ccc; cursor: not-allowed; }
        .status-bar { display: flex; gap: 20px; margin-top: 20px; }
        .status-item { flex: 1; background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
        .status-item .value { font-size: 1.5rem; font-weight: bold; color: #1E88E5; }
        .status-item .label { font-size: 0.9rem; color: #666; margin-top: 5px; }
        .results { margin-top: 30px; }
        .anomaly-item { background: #fff3e0; border-left: 4px solid #ff9800; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .anomaly-item.high { background: #ffebee; border-left-color: #f44336; }
        .anomaly-item.low { background: #e8f5e9; border-left-color: #4caf50; }
        .loading { display: none; text-align: center; padding: 50px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #1E88E5; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .info-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }
        .info-item { background: #e3f2fd; padding: 15px; border-radius: 5px; text-align: center; }
        .metric { font-size: 1.5rem; font-weight: bold; color: #1E88E5; }
        .metric-label { font-size: 0.85rem; color: #666; }
        .sidebar { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .sidebar h3 { color: #1E88E5; margin-bottom: 15px; }
        .sidebar-item { padding: 10px 0; border-bottom: 1px solid #eee; }
        .success { background: #e8f5e9; border: 1px solid #4caf50; padding: 15px; border-radius: 5px; color: #2e7d32; }
        .error { background: #ffebee; border: 1px solid #f44336; padding: 15px; border-radius: 5px; color: #c62828; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 LAVAD 视频异常检测系统</h1>
            <p>基于大语言模型的无训练视频异常检测 (CVPR 2024)</p>
        </div>

        <div style="display: flex; gap: 20px;">
            <div style="flex: 3;">
                <div class="card">
                    <h2 style="margin-bottom: 20px;">📤 上传视频</h2>
                    <form id="uploadForm" enctype="multipart/form-data">
                        <label class="upload-area" id="uploadArea">
                            <div class="upload-icon">📹</div>
                            <p style="font-size: 1.2rem; margin-bottom: 10px;">点击或拖拽视频文件到此处</p>
                            <p style="color: #666;">支持 MP4, MOV, MKV 格式</p>
                            <input type="file" id="videoFile" name="video" accept="video/*">
                        </label>
                        <div id="fileInfo" style="margin-top: 20px; display: none;">
                            <p id="fileName" style="font-weight: bold;"></p>
                            <p id="fileSize"></p>
                        </div>
                        <button type="submit" class="btn" id="analyzeBtn" style="margin-top: 20px; width: 100%;" disabled>
                            🚀 开始分析
                        </button>
                    </form>
                </div>

                <div class="card loading" id="loading">
                    <div class="spinner"></div>
                    <p style="font-size: 1.2rem;">正在分析视频，请耐心等待...</p>
                    <p style="color: #666; margin-top: 10px;">这可能需要几分钟时间</p>
                </div>

                <div class="card" id="results" style="display: none;">
                    <h2 style="margin-bottom: 20px;">📊 分析结果</h2>
                    <div class="info-grid" id="infoGrid"></div>
                    <div id="summary" style="margin-top: 20px; padding: 15px; border-radius: 5px;"></div>
                    <div class="results" id="anomalyResults"></div>
                </div>
            </div>

            <div style="flex: 1;">
                <div class="sidebar">
                    <h3>📊 系统状态</h3>
                    <div class="sidebar-item">
                        <span style="color: #4caf50;">●</span> 后端服务已连接
                    </div>
                    <div class="sidebar-item">
                        GPU 可用: <span id="gpuStatus">检查中...</span>
                    </div>
                </div>

                <div class="sidebar">
                    <h3>📈 数据集性能</h3>
                    <div class="sidebar-item">
                        <strong>ROC-AUC:</strong> <span style="color: #1E88E5;">0.7471</span>
                    </div>
                    <div class="sidebar-item">
                        <strong>PR-AUC:</strong> <span style="color: #1E88E5;">0.26</span>
                    </div>
                    <div class="sidebar-item">
                        <strong>数据集:</strong> UCF-Crime
                    </div>
                </div>

                <div class="sidebar">
                    <h3>ℹ️ 关于</h3>
                    <p style="font-size: 0.9rem; color: #666;">
                        LAVAD 利用 BLIP-2 进行视频描述生成，Qwen-7B 进行异常评分。
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = 'http://121.48.164.7:8000';

        async function checkHealth() {
            try {
                const res = await fetch(API_BASE + '/health');
                const data = await res.json();
                document.getElementById('gpuStatus').textContent = data.gpu_available ? '是 (' + data.gpu_count + ' GPU)' : '否';
            } catch (e) {
                document.getElementById('gpuStatus').textContent = '不可用';
            }
        }

        checkHealth();

        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('videoFile');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#e3f2fd';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = '';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = '';
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });

        fileInput.addEventListener('change', handleFileSelect);

        function handleFileSelect() {
            const file = fileInput.files[0];
            if (file) {
                fileName.textContent = '已选择: ' + file.name;
                fileSize.textContent = '大小: ' + (file.size / 1024 / 1024).toFixed(2) + ' MB';
                fileInfo.style.display = 'block';
                analyzeBtn.disabled = false;
            }
        }

        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const file = fileInput.files[0];
            if (!file) return;

            loading.style.display = 'block';
            results.style.display = 'none';

            const formData = new FormData();
            formData.append('video', file);

            try {
                const res = await fetch(API_BASE + '/analyze', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) throw new Error('分析失败');

                const data = await res.json();
                showResults(data);
            } catch (e) {
                alert('错误: ' + e.message);
            } finally {
                loading.style.display = 'none';
            }
        });

        function showResults(data) {
            results.style.display = 'block';

            const infoGrid = document.getElementById('infoGrid');
            infoGrid.innerHTML = `
                <div class="info-item">
                    <div class="metric">${data.total_frames}</div>
                    <div class="metric-label">总帧数</div>
                </div>
                <div class="info-item">
                    <div class="metric">${data.duration.toFixed(1)}s</div>
                    <div class="metric-label">时长</div>
                </div>
                <div class="info-item">
                    <div class="metric">${data.abnormal_frames}</div>
                    <div class="metric-label">异常帧</div>
                </div>
                <div class="info-item">
                    <div class="metric">${(data.anomaly_ratio * 100).toFixed(1)}%</div>
                    <div class="metric-label">异常比例</div>
                </div>
            `;

            const summary = document.getElementById('summary');
            const isWarning = data.anomaly_ratio > 0.1;
            summary.className = isWarning ? 'error' : 'success';
            summary.textContent = data.summary;

            const anomalyResults = document.getElementById('anomalyResults');
            if (data.anomaly_frames && data.anomaly_frames.length > 0) {
                anomalyResults.innerHTML = '<h3 style="margin: 20px 0 10px;">🚨 检测到的异常</h3>' +
                    data.anomaly_frames.slice(0, 10).map(f => {
                        const cls = f.score > 0.7 ? 'high' : (f.score > 0.5 ? '' : 'low');
                        const icon = f.score > 0.7 ? '🚨' : (f.score > 0.5 ? '⚠️' : '🔔');
                        return `
                            <div class="anomaly-item ${cls}">
                                <strong>${icon} 第${f.frame_idx}帧</strong> (${formatTime(f.timestamp)}) - 评分: ${f.score.toFixed(4)}<br>
                                <small>${f.caption || '无描述'}</small>
                            </div>
                        `;
                    }).join('');
            } else {
                anomalyResults.innerHTML = '<div class="success">🎉 未检测到异常</div>';
            }
        }

        function formatTime(seconds) {
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=False)