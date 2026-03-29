# AI Yoga Guardian — Streamlit Edition

Pipeline: **Video/Webcam → YOLOv8 keypoints → MLP (59d) → State Machine → Pose + Feedback**

## Cấu trúc

```
yoga-streamlit/
├── app.py
├── inference_pipeline.py
├── models/
│   ├── yoga_best_v2.pth
│   ├── label_map.json
│   ├── scaler.pkl
│   └── pose_stats.json
└── requirements.txt
```

## Setup & chạy local

```bash
pip install -r requirements.txt

streamlit run app.py
```

Mở http://localhost:8501

## Tính năng

### Nguồn video
- **Upload .mp4/.avi/.mov** — phân tích toàn bộ video, progress bar theo frame
- **Webcam realtime** — mirror effect, xử lý liên tục

### Sidebar
- Chọn **pose mục tiêu** → State Machine chỉ đếm rep cho pose đó
- Xem **trạng thái** (Chờ / Xác nhận / Đang tập / Giữ tốt!)
- **Đếm reps** tự động khi giữ pose đủ thời gian
- Reset reps bất kỳ lúc nào

### State Machine (Debouncing)
```
IDLE → PENDING (15 frames liên tiếp cùng pose, conf ≥ 70%)
     → ACTIVE  (xác nhận thành công)
     → HOLDING (giữ đủ 2 giây → +1 rep)
     → IDLE    (rời pose)
```
Tránh nhảy số rep khi model dao động.

### Feedback thông minh
Dùng **z-score** so với phân phối thực từ dataset:
- 🔴 Cần sửa gấp (|z| > 2.5)
- 🟡 Hơi lệch (|z| > 1.5)
- ✅ Tốt!

Skeleton vẽ **xanh lá** khi tư thế đúng, **đỏ** khi có lỗi.

## Sidebar
| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| Frames xác nhận | 15 | Bao nhiêu frame liên tiếp để ACTIVE |
| Confidence tối thiểu | 70% | Ngưỡng để bắt đầu đếm |
| Giữ pose (giây/rep) | 2.0 | Giữ bao lâu để +1 rep |
| Xử lý mỗi N frames | 2 | Frame skip để tăng tốc |