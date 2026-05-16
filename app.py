import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from model import my_model
import utils
import plotly.graph_objects as go
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import asyncio
from aiortc import MediaStreamTrack
import threading
import queue

# 页面配置
st.set_page_config(
    page_title="面部情绪识别系统",
    page_icon="😊",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class EmotionDetector:
    """情绪检测器类"""

    def __init__(self):
        self.model = None
        self.data_trans = None
        self.device = None
        self.label_name = None
        self.face_cascade = None
        self.latest_emotion_probs = None
        self.latest_pred_emotion = None
        self.latest_confidence = None
        self.lock = threading.Lock()

    def load_model(self):
        """加载模型和人脸检测器"""
        config = utils.read_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.label_name = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

        self.data_trans = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        model = my_model(num_classes=7)
        model_path = "./checkpoints/best.pth"

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        self.model = model

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_emotion(self, frame_bgr):
        """检测单帧的情绪"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return None, None, None, frame_bgr

        # 取最大的人脸
        face_rect = max(faces, key=lambda x: x[2] * x[3])
        (x, y, w, h) = face_rect

        # 绘制人脸框
        frame_with_box = frame_bgr.copy()
        cv2.rectangle(frame_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 提取人脸区域
        face_bgr = frame_bgr[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        # 预处理和预测
        face_tensor = self.data_trans(face_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(face_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probabilities[0][pred_idx].item()

        emotion_probs = {self.label_name[i]: probabilities[0][i].item() for i in range(len(self.label_name))}
        pred_emotion = self.label_name[pred_idx]

        # 在帧上显示结果
        result_text = f"{pred_emotion} ({confidence:.2f})"
        cv2.putText(frame_with_box, result_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        return emotion_probs, pred_emotion, confidence, frame_with_box


# 缓存检测器
@st.cache_resource
def get_detector():
    detector = EmotionDetector()
    detector.load_model()
    return detector


def create_emotion_bar_chart(emotion_probs):
    """创建情绪概率柱状图"""
    if emotion_probs is None:
        return None

    emotions = list(emotion_probs.keys())
    probabilities = list(emotion_probs.values())

    emotion_cn = {
        'Anger': '愤怒',
        'Disgust': '厌恶',
        'Fear': '恐惧',
        'Happiness': '开心',
        'Neutral': '中性',
        'Sadness': '悲伤',
        'Surprise': '惊讶'
    }

    colors = ['#FF6B6B', '#9C27B0', '#FF9800', '#4CAF50', '#9E9E9E', '#2196F3', '#FFEB3B']

    fig = go.Figure(data=[
        go.Bar(
            x=[emotion_cn[e] for e in emotions],
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.2%}' for p in probabilities],
            textposition='outside',
        )
    ])

    fig.update_layout(
        title='情绪概率分布',
        xaxis_title='情绪类型',
        yaxis_title='概率',
        yaxis_range=[0, 1],
        height=400,
        template='plotly_white',
        showlegend=False
    )

    return fig


def main():
    st.markdown('<div class="main-header">😊 面部情绪识别系统</div>', unsafe_allow_html=True)

    # 侧边栏
    st.sidebar.title("功能选择")
    mode = st.sidebar.radio("选择模式", ["📸 图片识别", "🎥 实时摄像头"])

    # 加载检测器
    with st.spinner('正在加载模型...'):
        detector = get_detector()
    st.sidebar.success("✅ 模型加载成功")

    # 模式1：图片识别
    if mode == "📸 图片识别":
        st.markdown('<div class="sub-header">上传图片进行情绪识别</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("选择一张图片", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🖼️ 原始图片")
                image = Image.open(uploaded_file)
                image_rgb = np.array(image)

                if len(image_rgb.shape) == 2:
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
                elif image_rgb.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

                st.image(image_rgb, use_container_width=True)

            with col2:
                st.subheader("📊 识别结果")

                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                faces = detector.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_bgr = image_bgr[y:y + h, x:x + w]
                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)

                    emotion_probs, pred_emotion, confidence, _ = detector.detect_emotion(image_bgr)

                    st.success(f"**识别结果**: {pred_emotion}")
                    st.info(f"**置信度**: {confidence:.2%}")

                    fig = create_emotion_bar_chart(emotion_probs)
                    st.plotly_chart(fig, use_container_width=True)

                    emotion_cn = {
                        'Anger': '愤怒',
                        'Disgust': '厌恶',
                        'Fear': '恐惧',
                        'Happiness': '开心',
                        'Neutral': '中性',
                        'Sadness': '悲伤',
                        'Surprise': '惊讶'
                    }

                    prob_data = {
                        '情绪': [emotion_cn[k] for k in emotion_probs.keys()],
                        '概率': [f'{v:.2%}' for v in emotion_probs.values()]
                    }
                    st.dataframe(prob_data, use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ 未检测到人脸")

    # 模式2：实时摄像头
    elif mode == "🎥 实时摄像头":
        st.markdown('<div class="sub-header">实时情绪识别</div>', unsafe_allow_html=True)

        st.info("💡 **提示**: 点击下方按钮开启摄像头，将脸部对准镜头即可实时识别情绪")

        # 初始化session state
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False
        if 'latest_frame' not in st.session_state:
            st.session_state.latest_frame = None
        if 'latest_emotion_probs' not in st.session_state:
            st.session_state.latest_emotion_probs = None
        if 'latest_pred_emotion' not in st.session_state:
            st.session_state.latest_pred_emotion = None
        if 'latest_confidence' not in st.session_state:
            st.session_state.latest_confidence = None

        # 摄像头控制
        col_start, col_stop = st.columns(2)

        def process_frame(frame):
            """处理每一帧"""
            try:
                # 转换格式
                img = frame.to_ndarray(format="bgr24")

                # 检测情绪
                emotion_probs, pred_emotion, confidence, processed_frame = detector.detect_emotion(img)

                # 更新session state
                st.session_state.latest_frame = processed_frame
                st.session_state.latest_emotion_probs = emotion_probs
                st.session_state.latest_pred_emotion = pred_emotion
                st.session_state.latest_confidence = confidence

                # 返回处理后的帧
                return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
            except Exception as e:
                st.error(f"处理帧时出错: {e}")
                return frame

        with col_start:
            start_button = st.button("▶️ 开启摄像头", type="primary", use_container_width=True)

        with col_stop:
            stop_button = st.button("⏹️ 停止", use_container_width=True)

        if start_button or st.session_state.is_running:
            st.session_state.is_running = True

            # 显示摄像头和处理结果
            col_video, col_stats = st.columns([2, 1])

            with col_video:
                st.subheader("📹 实时画面")

                # 使用webrtc_streamer
                webrtc_ctx = webrtc_streamer(
                    key="emotion-detection",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTCConfiguration(
                        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                    ),
                    video_frame_callback=process_frame,
                    media_stream_constraints={"video": True, "audio": False},
                )

                if not webrtc_ctx.state.playing:
                    st.warning("摄像头未启动")

        if stop_button:
            st.session_state.is_running = False
            st.session_state.latest_frame = None
            st.session_state.latest_emotion_probs = None
            st.session_state.latest_pred_emotion = None
            st.session_state.latest_confidence = None
            st.rerun()


if __name__ == "__main__":
    main()
