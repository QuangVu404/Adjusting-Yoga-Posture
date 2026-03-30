"""
app.py — Giai đoạn 4: Streamlit Dashboard (Hoàn thiện)
Gọi YogaPipeline từ inference_pipeline.py và bọc giao diện Web lên.
Hỗ trợ: Upload video (.mp4) và Webcam realtime.
"""

import time
import cv2
import numpy as np
import streamlit as st

from inference_pipeline import YogaPipeline

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AI Yoga Guardian',
    page_icon='🧘',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1117; }
.metric-card {
    background: #1a1d26;
    border: 1px solid #2d3142;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.pose-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #a8e063;
    line-height: 1;
}
.pose-warning {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f87171; /* Màu đỏ cảnh báo */
    line-height: 1.2;
}
.feedback-ok   { color: #a8e063; font-size: 0.85rem; }
.feedback-warn { color: #f0c96b; font-size: 0.85rem; }
.feedback-bad  { color: #f87171; font-size: 0.85rem; }
.stProgress > div > div { background: #a8e063; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return YogaPipeline(model_dir='models')

pipeline = load_pipeline()

if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🧘 AI Yoga Guardian')
    st.caption('YOLOv8 Pose + PyTorch MLP 62d + State Machine')
    st.divider()

    source = st.radio('Nguồn video', ['📤 Upload Video', '📷 Webcam'], index=0)

    st.subheader('Tư thế mục tiêu')
    pose_options = ['(Tất cả)'] + pipeline.supported_poses
    target_sel = st.selectbox('Chọn pose', pose_options, index=0)
    target_pose = '' if target_sel == '(Tất cả)' else target_sel
    if st.button('Áp dụng', use_container_width=True):
        pipeline.set_target(target_pose)
        st.success(f'Đặt mục tiêu: {target_sel}')

    st.divider()

    st.subheader('Trạng thái')
    state_ph    = st.empty()
    rep_ph      = st.empty()
    conf_bar_ph = st.empty()

    if st.button('🔄 Reset Reps', use_container_width=True):
        pipeline.reset_reps()
        st.rerun()

    st.divider()
    
    # 💡 Lời khuyên chống méo hình không gian
    st.info("💡 **Mẹo tập luyện:** Hãy đặt camera ngang tầm ngực/eo và song song với cơ thể để AI đo góc và tỷ lệ khung xương chính xác nhất, tránh bị méo hình không gian.")

    st.divider()
    with st.expander('Cài đặt nâng cao'):
        pipeline.sm.CONFIRM_FRAMES = st.slider('Frames xác nhận', 5, 30, 10)
        pipeline.sm.CONF_MIN       = st.slider('Confidence tối thiểu', 0.4, 0.95, 0.70)
        pipeline.sm.HOLD_SECONDS   = st.slider('Giữ pose (giây/rep)', 1.0, 5.0, 2.0)
        frame_skip = st.slider('Xử lý mỗi N frames', 1, 5, 2)


frame_skip = 2   # default, bị override bởi slider trong expander

# ──────────────────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────────────────
st.title('AI Yoga Guardian')

col_video, col_info = st.columns([3, 1], gap='medium')

with col_video:
    video_ph = st.empty()

with col_info:
    st.markdown('#### Nhận diện')
    pose_name_ph  = st.empty()
    conf_text_ph  = st.empty()

    st.markdown('#### Phân bố xác suất')
    score_bars_ph = st.empty()

    st.markdown('#### Góc khớp (°)')
    angles_ph     = st.empty()

    st.markdown('#### Phản hồi tư thế')
    feedback_ph   = st.empty()
    infer_ph      = st.empty()


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
def render_sidebar(result):
    state_ph.markdown(f'**{result.status_text}**')
    rep_ph.metric('Reps hoàn thành', result.reps)
    conf_bar_ph.progress(int(result.confidence * 100), text=f'Conf {result.confidence*100:.0f}%')

def render_info(result):
    # XỬ LÝ GIAO DIỆN KHI BỊ KHUẤT CAMERA
    if result.pose == 'INCOMPLETE_BODY':
        pose_name_ph.markdown(f'<div class="pose-warning">Camera<br>Bị Khuất</div>', unsafe_allow_html=True)
        conf_text_ph.caption('Không thể dự đoán')
        score_bars_ph.markdown('*Chưa thấy toàn thân*')
        angles_ph.empty()
        feedback_ph.empty()
        return

    # Giao diện bình thường khi đủ người
    pose_name_ph.markdown(f'<div class="pose-title">{result.pose or "—"}</div>', unsafe_allow_html=True)
    conf_text_ph.caption(f'{result.confidence*100:.1f}% confidence')

    if result.all_scores:
        md = ''
        for pose, score in sorted(result.all_scores.items(), key=lambda x:-x[1]):
            pct  = int(score * 100)
            bold = '**' if pose == result.pose else ''
            md  += f'{bold}{pose}{bold}: `{pct}%`  \n'
        score_bars_ph.markdown(md)

    if result.angles:
        rows = ''
        for k, v in result.angles.items():
            label = k.replace('ang_','').replace('_',' ').title()
            rows += f'| {label} | {v:.0f}° |\n'
        angles_ph.markdown('| Khớp | Góc |\n|---|---|\n' + rows)

    if result.feedback:
        lines = []
        for f in result.feedback:
            if '✅' in f:  cls = 'feedback-ok'
            elif '🔴' in f: cls = 'feedback-bad'
            else:           cls = 'feedback-warn'
            lines.append(f'<p class="{cls}">{f}</p>')
        feedback_ph.markdown(''.join(lines), unsafe_allow_html=True)


def process_and_display(frame, t0):
    result = pipeline.process_frame(frame)
    rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)

    # VẼ OVERLAY LÊN VIDEO NẾU THIẾU BỘ PHẬN
    if result.pose == 'INCOMPLETE_BODY':
        overlay = rgb.copy()
        cv2.rectangle(overlay, (0, 0), (rgb.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, rgb, 0.4, 0, rgb)
        # Báo lỗi đỏ lên góc trái video
        cv2.putText(rgb, "⚠️ VUI LONG LUI LAI DE CAMERA THAY TOAN THAN", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 80), 2, cv2.LINE_AA)

    video_ph.image(rgb, channels='RGB', use_column_width='always')
    render_sidebar(result)
    render_info(result)
    ms = (time.perf_counter() - t0) * 1000
    infer_ph.caption(f'⏱ {ms:.0f} ms / frame')
    st.session_state.last_result = result


# ──────────────────────────────────────────────────────────
# RUN LOOP (Video / Webcam)
# ──────────────────────────────────────────────────────────
if source == '📤 Upload Video':
    uploaded = st.file_uploader('Tải video lên (.mp4, .avi, .mov)', type=['mp4','avi','mov'])
    col_btn1, col_btn2, _ = st.columns([1,1,4])
    with col_btn1:
        start_btn = st.button('▶ Bắt đầu', type='primary', use_container_width=True)
    with col_btn2:
        stop_btn  = st.button('■ Dừng', use_container_width=True)

    if stop_btn: st.session_state.running = False
    if start_btn and uploaded:
        st.session_state.running = True
        tmp_path = f'/tmp/yoga_upload_{int(time.time())}.mp4'
        with open(tmp_path, 'wb') as f: f.write(uploaded.read())
        cap = cv2.VideoCapture(tmp_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid  = cap.get(cv2.CAP_PROP_FPS) or 25
        progress_bar = st.progress(0, text='Đang xử lý...')
        frame_idx = 0

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            progress_bar.progress(min(frame_idx / max(n_frames,1), 1.0), text=f'Frame {frame_idx}/{n_frames}')
            if frame_idx % frame_skip != 0: continue
            t0 = time.perf_counter()
            process_and_display(frame, t0)
            time.sleep(max(0, 1/fps_vid - (time.perf_counter()-t0)))

        cap.release()
        progress_bar.empty()
        st.session_state.running = False
        st.success('Xử lý video hoàn tất!')

    elif not uploaded:
        video_ph.info('👆 Tải video lên để bắt đầu phân tích.')

else:
    col_w1, col_w2, _ = st.columns([1,1,4])
    with col_w1:
        cam_start = st.button('📷 Bật Webcam', type='primary', use_container_width=True)
    with col_w2:
        cam_stop  = st.button('■ Tắt', use_container_width=True)

    if cam_stop: st.session_state.running = False
    if cam_start:
        st.session_state.running = True
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            st.error('Không thể mở webcam. Kiểm tra lại thiết bị.')
            st.session_state.running = False
        else:
            frame_idx = 0
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1) 
                frame_idx += 1
                if frame_idx % frame_skip != 0: continue
                process_and_display(frame, time.perf_counter())

            cap.release()
            st.session_state.running = False