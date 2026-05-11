import streamlit as st
import requests
import json
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import base64

st.set_page_config(
    page_title="LAVAD - 视频异常检测系统",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    .model-loaded {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4caf50;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
    }
    .model-unloaded {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 2px solid #f44336;
        box-shadow: 0 4px 12px rgba(244, 67, 54, 0.2);
    }
    .model-available {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border: 2px solid #ffc107;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.2);
    }
    .model-icon {
        font-size: 2.5rem;
        margin-bottom: 8px;
    }
    .model-name {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .model-desc {
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 8px;
    }
    .model-status {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-loaded {
        background-color: #4caf50;
        color: white;
    }
    .status-available {
        background-color: #ffc107;
        color: #333;
    }
    .status-unloaded {
        background-color: #f44336;
        color: white;
    }
    .overall-status {
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .overall-ready {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .overall-warning {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border: 2px solid #ff9800;
        color: #e65100;
    }
    .step-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #1E88E5;
    }
    .step-completed {
        border-left-color: #4caf50;
        background: #e8f5e9;
    }
    .step-current {
        border-left-color: #ff9800;
        background: #fff3e0;
    }
    .anomaly-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .anomaly-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .anomaly-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
    }
    .success-box {
        background-color: #e8f5e9;
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 1rem;
    }
    .warning-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 5px;
        padding: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 5px;
        padding: 1rem;
    }
    .result-box {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def check_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_model_status():
    try:
        response = requests.get(f"{API_BASE_URL}/model_status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_demo_results():
    try:
        response = requests.get(f"{API_BASE_URL}/demo_results", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_sample_videos():
    try:
        response = requests.get(f"{API_BASE_URL}/sample_videos", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def analyze_video_api(video_file):
    try:
        files = {"video": (video_file.name, video_file.getvalue(), video_file.type)}
        with st.spinner("正在分析视频... 由于采用完整的8步流程，大视频可能需要3-15分钟，请耐心等待。"):
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                files=files,
                timeout=7200
            )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"服务器错误: {response.status_code}", "detail": response.text}
    except requests.exceptions.Timeout:
        return {"error": "请求超时", "detail": "视频分析时间过长，请尝试较短的视频。"}
    except Exception as e:
        return {"error": str(e)}


def render_blip2_model_card(model_name, available, loaded):
    if loaded:
        card_class = "model-loaded"
        status_class = "status-loaded"
        status_text = "✅ 已加载"
    elif available:
        card_class = "model-available"
        status_class = "status-available"
        status_text = "📥 可用"
    else:
        card_class = "model-unloaded"
        status_class = "status-unloaded"
        status_text = "❌ 未下载"

    display_name = model_name.replace("/home/jinanyang/models/", "")

    st.markdown(f"""
    <div class="model-card {card_class}">
        <div class="model-name">{display_name}</div>
        <span class="model-status {status_class}">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)


def render_step_card(step_num, step_name, status="pending"):
    if status == "completed":
        card_class = "step-card step-completed"
        icon = "✅"
    elif status == "current":
        card_class = "step-card step-current"
        icon = "⏳"
    else:
        card_class = "step-card"
        icon = "⬜"

    st.markdown(f"""
    <div class="{card_class}">
        {icon} <b>Step {step_num}:</b> {step_name}
    </div>
    """, unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">🎬 LAVAD 视频异常检测系统</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">基于大语言模型的视频异常检测</p>',
        unsafe_allow_html=True
    )

    health = check_health()
    model_status = get_model_status()
    demo_results = get_demo_results()

    with st.sidebar:
        st.markdown("### 📊 系统状态")

        if health:
            st.success("✅ 后端服务已连接")
        else:
            st.error("❌ 后端服务未连接")
            st.warning("请确保后端服务在 8000 端口运行")

        st.markdown("---")

        st.markdown("### 🤖 使用模型")

        if model_status:
            st.markdown("**BLIP-2 模型:**")
            blip_models = [
                "blip2-opt-6.7b-coco",
                "blip2-opt-6.7b",
                "blip2-flan-t5-xxl",
                "blip2-flan-t5-xl",
                "blip2-flan-t5-xl-coco"
            ]
            for model in blip_models:
                st.markdown(f"- {model}")

            st.markdown("**其他模型:**")
            st.markdown("- Qwen-7B-Chat")
            st.markdown("- ImageBind")
        else:
            st.info("使用模型: BLIP-2 (5种) / Qwen-7B / ImageBind")

        st.markdown("---")

        st.markdown("### 📈 数据集")
        st.info("本系统在 **UCF-Crime** 数据集上进行评估的数据为")
        if demo_results:
            roc_auc = demo_results.get('roc_auc', 0)
            st.metric("ROC-AUC", f"{roc_auc*100:.2f}%")
        else:
            st.metric("ROC-AUC", "74.71%")

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 上传与分析", "📊 检测结果", "🔬 中间结果", "ℹ️ 帮助与文档"])

    with tab1:
        st.markdown("### 📤 上传视频进行分析")
        st.info("📋 **LAVAD 8步异常检测流程:**")
        st.markdown("""
        **Step 1:** 视频帧提取  
        **Step 2:** BLIP-2 Caption生成 (5种模型)  
        **Step 3:** ImageBind文本索引创建  
        **Step 4:** 跨模态Caption清洗  
        **Step 5:** LLM摘要生成与评分  
        **Step 6:** 摘要索引创建  
        **Step 7:** 多邻居分数细化  
        **Step 8:** 最终评估
        """)

        video_file = st.file_uploader(
            "选择视频文件",
            type=['mp4', 'mkv', 'mov'],
            help="支持的格式: MP4, MKV, MOV"
        )

        if video_file:
            st.success(f"✅ 已选择文件: {video_file.name}")
            file_size_mb = video_file.size / 1024 / 1024
            st.info(f"📦 文件大小: {file_size_mb:.2f} MB")

            video_bytes = video_file.getvalue()
            st.video(video_bytes)

        analyze_button = st.button("🚀 开始8步分析", type="primary", use_container_width=True)

        if analyze_button and video_file:
            result = analyze_video_api(video_file)

            if "error" in result:
                st.error(f"❌ 错误: {result['error']}")
                if "detail" in result:
                    with st.expander("错误详情"):
                        st.code(result['detail'])
            else:
                st.session_state['analysis_result'] = result
                st.session_state['show_results'] = True
                st.success("✅ 分析成功完成！请查看下方结果预览，或切换到'检测结果'标签页")

                anomaly_count = result.get('abnormal_frames', 0)
                total_frames = result.get('total_frames', 0)
                anomaly_ratio = result.get('anomaly_ratio', 0)
                overall_score = result.get('overall_score', 0)

                st.markdown("### 📊 结果预览")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总帧数", total_frames)
                with col2:
                    st.metric("异常帧数", anomaly_count)
                with col3:
                    st.metric("异常比例", f"{anomaly_ratio*100:.1f}%")
                with col4:
                    st.metric("总体评分", f"{overall_score:.4f}")

                st.info(f"📋 摘要: {result.get('summary', '')}")

        elif analyze_button and not video_file:
            st.warning("⚠️ 请先上传视频文件")

    with tab2:
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']

            st.markdown("### 📈 分析结果")
            st.markdown(f"**视频:** {result.get('video_name', '未知')}")

            steps_completed = result.get('steps_completed', [])
            if steps_completed:
                st.markdown("#### 🔄 分析流程进度")
                all_steps = [
                    "Step 1: Frame Extraction",
                    "Step 2: BLIP-2 Captioning (5 models)",
                    "Step 3: ImageBind Text Index",
                    "Step 4: Cross-modal Caption Cleaning",
                    "Step 5: LLM Summaries & Scoring",
                    "Step 6: Summary Index Creation",
                    "Step 7: Multi-neighbor Score Refinement",
                    "Step 8: Final Evaluation"
                ]
                for idx, step in enumerate(all_steps):
                    if step in steps_completed:
                        st.success(f"✅ {step}")
                    else:
                        st.write(f"⬜ {step}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总帧数", result.get('total_frames', 0))
            with col2:
                st.metric("帧率", f"{result.get('fps', 0):.1f}")
            with col3:
                st.metric("时长", f"{result.get('duration', 0):.1f}秒")
            with col4:
                st.metric("处理时间", f"{result.get('processing_time', 0):.1f}秒")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("异常帧数", result.get('abnormal_frames', 0))
            with col2:
                anomaly_ratio = result.get('anomaly_ratio', 0)
                st.metric("异常比例", f"{anomaly_ratio*100:.1f}%")
            with col3:
                st.metric("总体评分", f"{result.get('overall_score', 0):.4f}")

            st.markdown("### 📝 分析摘要")
            st.info(result.get('summary', '暂无摘要'))

            anomaly_frames = result.get('anomaly_frames', [])

            if anomaly_frames:
                st.markdown("### 🚨 检测到的异常")
                st.write(f"发现 **{len(anomaly_frames)}** 个潜在异常帧")

                df = pd.DataFrame(anomaly_frames)
                if 'timestamp' in df.columns:
                    df['timestamp_str'] = df['timestamp'].apply(
                        lambda x: f"{int(x//60):02d}:{int(x%60):02d}"
                    )
                else:
                    df['timestamp_str'] = "00:00"
                
                if 'caption' not in df.columns:
                    df['caption'] = ""
                if 'summary' not in df.columns:
                    df['summary'] = ""

                fig = px.bar(
                    df,
                    x='timestamp_str',
                    y='score',
                    color='score',
                    color_continuous_scale='RdYlGn_r',
                    title="异常评分随时间变化",
                    labels={'timestamp_str': '时间戳 (分:秒)', 'score': '异常评分'}
                )
                fig.update_layout(
                    xaxis_title="时间戳",
                    yaxis_title="异常评分 (0-1)",
                    yaxis=dict(range=[0, 1.05]),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 📥 导出异常帧")
                with st.spinner("正在打包异常帧..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE_URL}/export_anomaly_frames",
                            timeout=120
                        )
                        if resp.status_code == 200:
                            st.download_button(
                                label="📥 下载 anomaly_frames.zip (含描述文件)",
                                data=resp.content,
                                file_name="anomaly_frames.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"请求失败: {e}")

                st.markdown("#### 📸 异常帧预览")
                image_cols = st.columns(min(4, len(anomaly_frames)))
                for i, (_, row) in enumerate(df.iterrows()):
                    score = row['score']
                    col = image_cols[i % len(image_cols)]
                    with col:
                        has_image = False
                        image_base64 = row.get('image_base64')
                        if image_base64 and isinstance(image_base64, str) and len(image_base64) > 100:
                            try:
                                import base64
                                img_bytes = base64.b64decode(image_base64)
                                col.image(img_bytes, caption=f"帧#{row['frame_idx']} T:{row['timestamp_str']}", use_container_width=True)
                                has_image = True
                            except Exception as e:
                                col.warning(f"图片解码失败")
                        if not has_image:
                            image_path = row.get('image_path', '')
                            if image_path and isinstance(image_path, str):
                                try:
                                    with open(image_path, 'rb') as f:
                                        img_bytes = f.read()
                                        col.image(img_bytes, caption=f"帧#{row['frame_idx']} T:{row['timestamp_str']}", use_container_width=True)
                                        has_image = True
                                except:
                                    pass
                        if not has_image:
                            col.info(f"📷 帧#{row['frame_idx']} @ {row['timestamp_str']}")

                        if score > 0.7:
                            col.markdown(f"🚨 **{score:.4f}**")
                        elif score > 0.5:
                            col.markdown(f"⚠️ **{score:.4f}**")
                        else:
                            col.markdown(f"🔔 **{score:.4f}**")

                st.markdown("#### 📋 异常帧详情")
                for idx, row in df.iterrows():
                    score = row['score']
                    if score > 0.7:
                        border_color = "#f44336"
                        bg_color = "#ffebee"
                        icon = "🚨"
                    elif score > 0.5:
                        border_color = "#ff9800"
                        bg_color = "#fff3e0"
                        icon = "⚠️"
                    else:
                        border_color = "#4caf50"
                        bg_color = "#e8f5e9"
                        icon = "🔔"

                    caption_text = row.get("caption", "") or "无描述"
                    summary_text = row.get("summary", "") or "无摘要"

                    frame_info = f"{icon} **第 {row['frame_idx']} 帧** | 时间: {row['timestamp_str']} | 评分: **{score:.4f}**"
                    st.markdown(frame_info)

                    image_base64 = row.get('image_base64')
                    if image_base64 and isinstance(image_base64, str) and len(image_base64) > 100:
                        try:
                            img_html = f'<img src="data:image/jpeg;base64,{image_base64}" style="max-width:250px; border-radius:4px; margin:8px 0; border:2px solid {border_color};">'
                            st.markdown(img_html, unsafe_allow_html=True)
                        except:
                            pass
                    else:
                        st.caption("📷 暂无预览图")

                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**📝 描述:** {caption_text[:200]}{'...' if len(caption_text) > 200 else ''}")
                        with col2:
                            st.markdown(f"**📋 摘要:** {summary_text[:200]}{'...' if len(summary_text) > 200 else ''}")

                    st.markdown("---")

                with st.expander("📋 表格视图"):
                    display_df = df[['frame_idx', 'timestamp_str', 'score', 'caption']].copy()
                    display_df.columns = ['帧编号', '时间', '评分', '描述']
                    display_df['评分'] = display_df['评分'].apply(lambda x: f"{x:.4f}")
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.success("🎉 未检测到异常！该视频显示正常活动。")

            if st.button("🔄 分析另一个视频"):
                if 'analysis_result' in st.session_state:
                    del st.session_state['analysis_result']
                st.rerun()
        else:
            st.info("👆 请先上传并分析视频以查看结果")

    with tab3:
        st.markdown("### 🔬 中间结果详情")

        if 'analysis_result' not in st.session_state or not st.session_state.analysis_result:
            st.info("👆 请先在「上传与分析」标签页上传并分析视频")
        else:
            with st.spinner("正在加载中间结果..."):
                try:
                    resp = requests.get(f"{API_BASE_URL}/intermediate_results", timeout=10)
                    if resp.status_code != 200:
                        st.error(f"获取中间结果失败: {resp.text}")
                        data = None
                    else:
                        data = resp.json()
                except Exception as e:
                    st.error(f"请求失败: {e}")
                    data = None

            if data:
                analysis_result = st.session_state.analysis_result
                num_clips = analysis_result.get('total_frames', 0)

                with st.expander("📋 Step 2: BLIP-2 原始Caption (每个clip的文本描述)", expanded=False):
                    captions_data = data.get("captions", {})
                    if captions_data:
                        selected_model = st.selectbox("选择模型", list(captions_data.keys()))
                        items = captions_data[selected_model]
                        st.markdown(f"**{selected_model}** - 共 {len(items)} 条Caption")
                        for item in items[:min(20, len(items))]:
                            ts = item['clip_idx'] * 16 / max(analysis_result.get('fps', 30), 1)
                            st.markdown(f"**Clip {item['clip_idx']}** (T≈{ts:.1f}s): {item['caption']}")
                        if len(items) > 20:
                            st.info(f"仅显示前20条，共 {len(items)} 条")
                    else:
                        st.info("无Caption数据")

                with st.expander("🧹 Step 4: 清洗后Caption (跨模态匹配过滤)", expanded=False):
                    cleaned_data = data.get("cleaned_captions", {})
                    if cleaned_data:
                        selected_model = st.selectbox("选择模型 (清洗后)", list(cleaned_data.keys()))
                        items = cleaned_data[selected_model]
                        matched = [it for it in items if it.get('captions')]
                        st.markdown(f"**{selected_model}** - 共 {len(items)} clips, {len(matched)} 个被清洗匹配")
                        for item in matched[:min(15, len(matched))]:
                            ts = item['clip_idx'] * 16 / max(analysis_result.get('fps', 30), 1)
                            caps_text = " | ".join([f"[{c['nn_idx']}] {c['caption'][:80]}" for c in item.get('captions', [])])
                            st.markdown(f"**Clip {item['clip_idx']}** (T≈{ts:.1f}s): {caps_text}")
                        if len(matched) > 15:
                            st.info(f"仅显示前15条，共 {len(matched)} 条")
                    else:
                        st.info("无清洗后Caption数据")

                with st.expander("📝 Step 5: LLM摘要与原始异常评分", expanded=False):
                    summaries_data = data.get("step4_summaries", [])
                    if summaries_data:
                        st.markdown(f"共 {len(summaries_data)} 个clip的摘要和评分")
                        for item in summaries_data[:min(20, len(summaries_data))]:
                            ts = item['clip_idx'] * 16 / max(analysis_result.get('fps', 30), 1)
                            score_str = f"{item['raw_score']:.4f}" if item.get('raw_score') is not None else "N/A"
                            st.markdown(f"**Clip {item['clip_idx']}** (T≈{ts:.1f}s) | Score: **{score_str}**")
                            st.markdown(f"_{item['summary'][:150]}{'...' if len(item['summary']) > 150 else ''}_")
                        if len(summaries_data) > 20:
                            st.info(f"仅显示前20条，共 {len(summaries_data)} 条")
                    else:
                        st.info("无摘要数据")

                with st.expander("📈 Step 5 vs Step 7 评分对比", expanded=True):
                    refined_data = data.get("step6_refined_scores", [])
                    if refined_data:
                        indices = [d['clip_idx'] for d in refined_data]
                        raw_scores = [d.get('raw_score') if d.get('raw_score') is not None else d['refined_score'] for d in refined_data]
                        refined_scores = [d['refined_score'] for d in refined_data]

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=indices, y=raw_scores,
                            mode='lines+markers',
                            name='Step 5 原始评分',
                            line=dict(color='royalblue', width=2),
                            marker=dict(size=6, symbol='circle')
                        ))
                        fig.add_trace(go.Scatter(
                            x=indices, y=refined_scores,
                            mode='lines+markers',
                            name='Step 7 细化评分',
                            line=dict(color='crimson', width=2),
                            marker=dict(size=6, symbol='x')
                        ))

                        threshold = analysis_result.get('threshold', 0.5)
                        fig.add_hline(
                            y=threshold, line_dash="dash", line_color="orange",
                            annotation_text=f"阈值 {threshold:.4f}"
                        )

                        fig.update_layout(
                            title="Step 5 原始评分 vs Step 7 细化评分",
                            xaxis_title="Clip 索引",
                            yaxis_title="异常评分 (0-1)",
                            yaxis=dict(range=[0, 1.05]),
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown(f"共 {len(refined_data)} 个 clips | 阈值: **{threshold:.4f}** | Step 5 评分范围: **{min(raw_scores):.4f} ~ {max(raw_scores):.4f}**")
                    else:
                        st.info("无细化评分数据")

                with st.expander("🎯 Step 8: 异常帧最终评分", expanded=False):
                    anomaly_frames = analysis_result.get('anomaly_frames', [])
                    if anomaly_frames:
                        st.markdown(f"**{len(anomaly_frames)}** 个异常帧 (阈值: {analysis_result.get('threshold', 'N/A')})")
                        for af in anomaly_frames:
                            st.markdown(f"- 帧#{af['frame_idx']} T={af.get('timestamp_str', 'N/A')} | Score: **{af['score']}** | Caption: _{af.get('caption', 'N/A')[:100]}_")
                    else:
                        st.info("无异常帧")
            else:
                st.warning("暂无中间结果数据")

    with tab4:
        st.markdown("### ℹ️ LAVAD 8步流程详解")

        with st.expander("Step 1: 视频帧提取", expanded=True):
            st.markdown("""
            **功能:** 从上传的视频中按固定间隔提取帧

            **参数:**
            - 帧间隔: 短视频4帧，中视频8帧，长视频16帧
            - 采样方式: 均匀采样
            """)

        with st.expander("Step 2: BLIP-2 Caption生成"):
            st.markdown("""
            **功能:** 使用5种不同的BLIP-2模型为每个视频帧生成文本描述

            **使用的模型:**
            1. blip2-flan-t5-xl
            2. blip2-flan-t5-xl-coco
            3. blip2-flan-t5-xxl
            4. blip2-opt-6.7b
            5. blip2-opt-6.7b-coco

            **特点:** 分批加载模型以节省显存
            """)

        with st.expander("Step 3: ImageBind文本索引"):
            st.markdown("""
            **功能:** 使用ImageBind模型将所有caption转换为文本嵌入向量，
            并使用FAISS构建向量索引以便快速相似度搜索
            """)

        with st.expander("Step 4: 跨模态Caption清洗"):
            st.markdown("""
            **功能:** 使用ImageBind的视觉特征与文本特征进行相似度匹配，
            过滤掉与对应帧视觉内容不匹配的caption，提高描述准确性
            """)

        with st.expander("Step 5: LLM摘要与评分"):
            st.markdown("""
            **功能:** 使用Qwen-7B大语言模型：
            1. 根据清洗后的caption生成场景摘要
            2. 对摘要进行0-1异常评分

            **Prompt:**
            - 摘要Prompt: 要求生成简洁的场景描述
            - 评分Prompt: 要求从0-1评分异常可能性
            """)

        with st.expander("Step 6: 摘要索引创建"):
            st.markdown("""
            **功能:** 将所有摘要转换为ImageBind文本嵌入向量，
            构建FAISS索引用于后续的相似度搜索
            """)

        with st.expander("Step 7: 多邻居分数细化"):
            st.markdown("""
            **功能:** 对每个视频片段，使用ImageBind找到视觉上最相似的
            10个邻居片段，结合邻居片段的异常评分进行加权细化

            **公式:** refined_score = Σ(neighbor_score × similarity) / Σ(similarity)
            """)

        with st.expander("Step 8: 最终评估"):
            st.markdown("""
            **功能:** 根据细化后的评分，使用 max(mean_score, 0.45) 作为阈值
            识别超过阈值的异常帧，并输出最终检测结果
            """)

        with st.expander("🔬 关于 LAVAD"):
            st.markdown("""
            #### LAVAD: 基于语言的视频异常检测

            本系统实现了 **LAVAD** 方法，来自 CVPR 2024:

            ```bibtex
            @inproceedings{zanella2024harnessing,
              title={Harnessing Large Language Models for Training-free Video Anomaly Detection},
              author={Zanella, Luca and Menapace, Willi and Mancini, Massimiliano and Wang, Yiming and Ricci, Elisa},
              booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
              pages={18527--18536},
              year={2024}
            }
            ```
            """)


if __name__ == "__main__":
    main()