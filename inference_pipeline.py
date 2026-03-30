"""
inference_pipeline.py — Phiên bản Hoàn thiện (Production Ready)
- Mạng MLP 62 chiều (Bổ sung 3 đặc trưng hình học không gian).
- Chống rung nhiễu với DROP_FRAMES.
- Tích hợp Gatekeeper: Ép về trạng thái IDLE khi nhận diện tư thế vô lý (OOD).
"""

import json, pickle, time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────
class YogaMLP(nn.Module):
    def __init__(self, input_size=62, num_classes=5):  # SỬA: 62 chiều
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x): return self.net(x)


# ──────────────────────────────────────────────────────────
# COCO CONFIG & LOGIC MAX/MIN
# ──────────────────────────────────────────────────────────
SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
KP_GROUP = ['head','head','head','head','head',
            'left','right','left','right','left','right',
            'left','right','left','right','left','right']
KP_COLOR = {'head': (240,201,107), 'left': (100,220,80), 'right': (80,180,240)}

JOINT_PAIRS = [
    (5, 7, 9, 6, 8, 10, 'ang_elbow'),
    (11, 5, 7, 12, 6, 8, 'ang_shoulder'),
    (5, 11, 13, 6, 12, 14, 'ang_hip'),
    (11, 13, 15, 12, 14, 16, 'ang_knee')
]

JOINT_DISPLAY = {
    'ang_elbow_max': 'Tay duỗi',        'ang_elbow_min': 'Tay gập',
    'ang_shoulder_max': 'Vai (góc mở)', 'ang_shoulder_min': 'Vai (góc khép)',
    'ang_hip_max': 'Hông (trụ)',        'ang_hip_min': 'Hông (gập)',
    'ang_knee_max': 'Chân (duỗi)',      'ang_knee_min': 'Chân (gập)',
}

JOINT_DIRECTION = {
    'ang_elbow_max':    ('Duỗi thẳng cánh tay hơn',     'Giảm độ căng ở khuỷu tay'),
    'ang_elbow_min':    ('Mở rộng góc tay gập ra',      'Gập tay sâu hơn nữa'),
    'ang_shoulder_max': ('Nâng hoặc mở rộng vai hơn',   'Hạ bớt vai xuống'),
    'ang_shoulder_min': ('Mở rộng góc vai ra',          'Khép vai lại gần thân hơn'),
    'ang_hip_max':      ('Đẩy hông thẳng lên',          'Gập nhẹ hông lại'),
    'ang_hip_min':      ('Mở rộng góc hông (ngẩng lên)','Gập người sâu hơn nữa'),
    'ang_knee_max':     ('Duỗi thẳng gối (đứng thẳng)', 'Trùng nhẹ đầu gối xuống'),
    'ang_knee_min':     ('Mở rộng góc gối',             'Gập gối sâu hơn (hạ trọng tâm)'),
}

Z_WARN  = 1.5
Z_ALERT = 2.5
CONF_THRESH = 0.3


# ──────────────────────────────────────────────────────────
# STATE MACHINE (CHỐNG RUNG NHIỄU)
# ──────────────────────────────────────────────────────────
class PoseState(Enum):
    IDLE    = auto()
    PENDING = auto()
    ACTIVE  = auto()
    HOLDING = auto()

@dataclass
class StateMachine:
    CONFIRM_FRAMES: int   = 10
    CONF_MIN:       float = 0.70
    HOLD_SECONDS:   float = 2.0
    DROP_FRAMES:    int   = 5

    state:          PoseState = field(default=PoseState.IDLE)
    current_pose:   str       = field(default='')
    pending_pose:   str       = field(default='')
    pending_count:  int       = field(default=0)
    drop_count:     int       = field(default=0)
    reps:           int       = field(default=0)
    hold_start:     float     = field(default=0.0)
    target_pose:    str       = field(default='')

    def update(self, pred_pose: str, confidence: float, valid_form: bool) -> 'StateMachine':
        now = time.time()
        active_pose = self.target_pose if self.target_pose else pred_pose

        is_good = (confidence >= self.CONF_MIN) and valid_form and (pred_pose == active_pose)

        if self.state in (PoseState.ACTIVE, PoseState.HOLDING):
            if is_good and (pred_pose == self.current_pose):
                self.drop_count = 0
                held = now - self.hold_start
                if held >= self.HOLD_SECONDS and self.state == PoseState.ACTIVE:
                    self.state  = PoseState.HOLDING
                    self.reps  += 1
            else:
                self.drop_count += 1
                if self.drop_count >= self.DROP_FRAMES:
                    self.state         = PoseState.IDLE
                    self.drop_count    = 0
                    self.pending_count = 0
                    self.hold_start    = 0.0
            return self

        if is_good:
            if self.state == PoseState.IDLE:
                self.state         = PoseState.PENDING
                self.pending_pose  = pred_pose
                self.pending_count = 1
            elif self.state == PoseState.PENDING:
                if pred_pose == self.pending_pose:
                    self.pending_count += 1
                    if self.pending_count >= self.CONFIRM_FRAMES:
                        self.state        = PoseState.ACTIVE
                        self.current_pose = pred_pose
                        self.hold_start   = now
                        self.drop_count   = 0
                else:
                    self.pending_pose  = pred_pose
                    self.pending_count = 1
        else:
            self.state         = PoseState.IDLE
            self.pending_count = 0
            
        return self

    @property
    def status_text(self) -> str:
        if self.state == PoseState.IDLE:    return '⏸ Chờ...'
        if self.state == PoseState.PENDING: return f'🔄 Xác nhận... ({self.pending_count}/{self.CONFIRM_FRAMES})'
        if self.state == PoseState.ACTIVE:  return '✅ Đang tập'
        if self.state == PoseState.HOLDING: return '🏆 Giữ tốt!'
        return ''


# ──────────────────────────────────────────────────────────
# FEATURE & VALIDATION
# ──────────────────────────────────────────────────────────
def _calc_angle(p1, p2, p3):
    if np.any(p1 == 0) or np.any(p2 == 0) or np.any(p3 == 0): return -1.0
    v1, v2 = p1 - p2, p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

def _get_max_min_angles(a_left, a_right):
    if a_left < 0 and a_right < 0: return -1.0, -1.0
    if a_left < 0: return a_right, -1.0
    if a_right < 0: return a_left, -1.0
    return max(a_left, a_right), min(a_left, a_right)

def _extract_features(keypoints):
    kp_xy, kp_conf = keypoints[:, :2], keypoints[:, 2]
    hip_c = (kp_xy[11] + kp_xy[12]) / 2.0
    sho_c = (kp_xy[5]  + kp_xy[6])  / 2.0
    torso = np.linalg.norm(sho_c - hip_c)
    if torso < 1e-5: return None, {}
        
    # --- 3 Tín hiệu Không gian ---
    shoulder_width = np.linalg.norm(kp_xy[5] - kp_xy[6])
    norm_shoulder_width = shoulder_width / (torso + 1e-8)
    
    dx = sho_c[0] - hip_c[0]
    dy = sho_c[1] - hip_c[1]
    spine_angle = np.degrees(np.arctan2(dy, dx))
    norm_spine_angle = spine_angle / 180.0
    
    left_leg  = np.linalg.norm(kp_xy[11] - kp_xy[13]) + np.linalg.norm(kp_xy[13] - kp_xy[15])
    right_leg = np.linalg.norm(kp_xy[12] - kp_xy[14]) + np.linalg.norm(kp_xy[14] - kp_xy[16])
    avg_leg_length = (left_leg + right_leg) / 2.0
    bone_ratio = torso / (avg_leg_length + 1e-8)
    
    spatial_features = np.array([norm_shoulder_width, norm_spine_angle, bone_ratio])
    # -----------------------------

    norm_xy   = (kp_xy - hip_c) / torso
    conf_feat = np.clip(kp_conf, 0.0, 1.0)
    angles_raw, angle_feat = {}, []
    
    for i_L, j_L, k_L, i_R, j_R, k_R, name in JOINT_PAIRS:
        deg_L = _calc_angle(kp_xy[i_L], kp_xy[j_L], kp_xy[k_L])
        deg_R = _calc_angle(kp_xy[i_R], kp_xy[j_R], kp_xy[k_R])
        a_max, a_min = _get_max_min_angles(deg_L, deg_R)
        
        angles_raw[f"{name}_max"] = a_max
        angles_raw[f"{name}_min"] = a_min
        angle_feat.append(a_max / 180.0 if a_max >= 0 else -1.0)
        angle_feat.append(a_min / 180.0 if a_min >= 0 else -1.0)
        
    feat = np.concatenate([norm_xy.flatten(), conf_feat, np.array(angle_feat), spatial_features])
    return feat.astype(np.float32), angles_raw

def _validate_form(pose, angles_raw, pose_stats):
    stats = pose_stats.get(pose, {})
    if not stats: return True
    
    alert_count = 0
    for joint, deg in angles_raw.items():
        if deg < 0: continue
        s = stats.get(joint)
        if not s or s['std'] < 1e-3: continue
        
        z = abs(deg - s['mean']) / s['std']
        
        if z > 3.5: 
            return False
        if z > 2.5: 
            alert_count += 1
            
    if alert_count >= 3: 
        return False
        
    return True


# ──────────────────────────────────────────────────────────
# FEEDBACK & RENDERING
# ──────────────────────────────────────────────────────────
def _generate_feedback(pose, angles_raw, pose_stats, target_pose=''):
    feedback, effective = [], pose
    if target_pose and target_pose != pose:
        feedback.append(f"⚠️ Đang nhận '{pose}', chưa phải '{target_pose}'")
        effective = target_pose
        
    stats = pose_stats.get(effective, {})
    issues = []
    for joint, deg in angles_raw.items():
        if deg < 0: continue
        s = stats.get(joint)
        if not s or s['std'] < 1e-3: continue
        
        z = (deg - s['mean']) / s['std']
        if abs(z) < Z_WARN: continue
        
        label = JOINT_DISPLAY.get(joint, joint)
        dir_low, dir_high = JOINT_DIRECTION.get(joint, ('Điều chỉnh','Điều chỉnh'))
        direction = dir_low if z < 0 else dir_high
        icon = '🔴' if abs(z) >= Z_ALERT else '🟡'
        
        issues.append((abs(z), f"{icon} {label}: {direction} ({deg:.0f}° / tb {s['mean']:.0f}°±{s['std']:.0f}°)"))
        
    issues.sort(reverse=True)
    if not issues:
        feedback.append(f"✅ {effective} tốt! Mọi khớp trong phạm vi bình thường.")
    else:
        feedback.extend(m for _, m in issues)
    return feedback

def _draw_skeleton(frame, kps, is_correct: bool):
    h, w = frame.shape[:2]
    edge_color = (80, 220, 80) if is_correct else (80, 80, 240)
    for a, b in SKELETON_EDGES:
        pa, pb = kps[a], kps[b]
        if pa[2] < CONF_THRESH or pb[2] < CONF_THRESH: continue
        x1,y1 = int(pa[0]), int(pa[1])
        x2,y2 = int(pb[0]), int(pb[1])
        cv2.line(frame, (x1,y1), (x2,y2), edge_color, 2, cv2.LINE_AA)
    for i, kp in enumerate(kps):
        if kp[2] < CONF_THRESH: continue
        x, y = int(kp[0]), int(kp[1])
        col  = KP_COLOR[KP_GROUP[i]]
        cv2.circle(frame, (x, y), 5, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 8, col, 1,  cv2.LINE_AA)

def _draw_hud(frame, pose, confidence, state_machine: StateMachine, feedback: list[str]):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (300, 110), (15,15,20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, pose.upper(),        (12,36),  cv2.FONT_HERSHEY_DUPLEX,  0.9, (168,224,99), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{confidence*100:.0f}% conf', (12,58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.putText(frame, state_machine.status_text,     (12,78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(frame, f'Reps: {state_machine.reps}', (12,100),cv2.FONT_HERSHEY_SIMPLEX, 0.55,(240,200,80),  1)

    y0 = h - 10 - len(feedback) * 22
    for i, line in enumerate(feedback[:4]):
        clean = line.replace('🔴','[!]').replace('🟡','[~]').replace('✅','[ok]').replace('⚠️','[w]').replace('❌','[X]')
        col = (80,80,240) if ('[!]' in clean or '[X]' in clean) else \
              (80,200,240) if '[~]' in clean else (80,220,80)
        cv2.putText(frame, clean, (12, y0 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

# ──────────────────────────────────────────────────────────
# RESULT DATACLASS
# ──────────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    frame:        np.ndarray
    pose:         str
    confidence:   float
    all_scores:   dict
    angles:       dict
    feedback:     list[str]
    state:        PoseState
    reps:         int
    status_text:  str
    person_found: bool

# ──────────────────────────────────────────────────────────
# MAIN PIPELINE CLASS
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# BỘ LỌC 1: KEYPOINT SMOOTHER (EMA Filter)
# ──────────────────────────────────────────────────────────
class KeypointSmoother:
    """
    Exponential Moving Average filter cho 17 keypoints COCO.
    Làm mượt tọa độ (x, y) qua các frame liên tiếp.

    Công thức EMA: smoothed = alpha * new + (1 - alpha) * prev
      alpha gần 1 → bám sát frame mới (ít lag, ít mượt)
      alpha gần 0 → mượt hơn nhưng lag nhiều

    Chọn alpha = 0.4: đủ mượt cho demo, không lag nhìn thấy được.
    Confidence KHÔNG filter (giữ nguyên để gatekeeper hoạt động đúng).
    """
    def __init__(self, alpha: float = 0.4, n_kp: int = 17):
        self.alpha   = alpha
        self.n_kp    = n_kp
        self._prev   = None   # (17, 3) frame trước, None nếu chưa có

    def reset(self):
        self._prev = None

    def smooth(self, kpts: np.ndarray) -> np.ndarray:
        """
        Input : (17, 3) — [x, y, conf] raw từ YOLOv8
        Output: (17, 3) — [x_smooth, y_smooth, conf_raw]
        """
        if self._prev is None:
            self._prev = kpts.copy()
            return kpts.copy()

        smoothed = kpts.copy()
        # Chỉ EMA tọa độ (x, y), không chạm conf
        smoothed[:, :2] = (
            self.alpha * kpts[:, :2]
            + (1.0 - self.alpha) * self._prev[:, :2]
        )
        self._prev = smoothed.copy()
        return smoothed


# ──────────────────────────────────────────────────────────
# BỘ LỌC 2: FEEDBACK DEBOUNCER
# ──────────────────────────────────────────────────────────
class FeedbackDebouncer:
    """
    Ổn định lời khuyên feedback qua các frame.
    Tránh câu cảnh báo chớp nháy khi người đang điều chỉnh chậm.

    Quy tắc:
      - Lỗi MỚI phải xuất hiện liên tục N_SHOW frames mới được hiển thị.
      - Lỗi ĐÃ HIỆN phải biến mất liên tục N_HIDE frames mới bị gỡ.
      - Lời khuyên ✅ (tốt) luôn hiển thị ngay, không cần debounce.

    Cấu trúc nội tại:
      _counts[msg] > 0  : frame liên tiếp msg xuất hiện (chờ show)
      _counts[msg] < 0  : frame liên tiếp msg vắng mặt (chờ hide)
      _active[msg]      : msg đang được hiển thị
    """
    def __init__(self, n_show: int = 5, n_hide: int = 5):
        self.N_SHOW  = n_show   # frames liên tiếp để bật cảnh báo
        self.N_HIDE  = n_hide   # frames liên tiếp để tắt cảnh báo
        self._counts: dict[str, int] = {}
        self._active: set[str]       = set()

    def reset(self):
        self._counts.clear()
        self._active.clear()

    def update(self, feedback_raw: list[str]) -> list[str]:
        """
        Input : list feedback thô từ _generate_feedback (frame hiện tại)
        Output: list feedback đã debounce (ổn định)
        """
        # Tách riêng: lời khuyên tốt (✅) không cần debounce
        good_msgs = [f for f in feedback_raw if f.startswith('✅')]
        warn_msgs = [f for f in feedback_raw if not f.startswith('✅')]

        current_set = set(warn_msgs)

        # Cập nhật counter cho từng msg đang theo dõi
        all_tracked = set(self._counts.keys()) | current_set
        to_delete = []
        for msg in all_tracked:
            if msg in current_set:
                # Msg đang xuất hiện → tăng counter dương
                self._counts[msg] = max(self._counts.get(msg, 0), 0) + 1
                if self._counts[msg] >= self.N_SHOW:
                    self._active.add(msg)
            else:
                # Msg đã biến mất → giảm counter (đếm âm)
                self._counts[msg] = min(self._counts.get(msg, 0), 0) - 1
                if self._counts[msg] <= -self.N_HIDE:
                    self._active.discard(msg)
                    to_delete.append(msg)

        # stable_warns = msgs đang trong _active (đã được confirm show)
        # Quan trọng: lấy từ _active, KHÔNG lấy từ warn_msgs hiện tại
        # vì khi warn vừa biến mất, warn_msgs=[] nhưng _active vẫn còn msg → cần giữ lại
        stable_warns = list(self._active - {m for m in to_delete})

        for msg in to_delete:
            del self._counts[msg]

        return stable_warns + good_msgs if stable_warns else good_msgs


class YogaPipeline:
    def __init__(self, model_dir: str = 'models'):
        d = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # SỬA: Dùng YOLOv8s cho ổn định
        self.yolo = YOLO('yolov8s-pose.pt')

        with open(d / 'label_map.json') as f:
            lm = json.load(f)
        self.class_names = [k for k,_ in sorted(lm.items(), key=lambda x: x[1])]
        
        with open(d / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        ps_path = d / 'pose_stats.json'
        self.pose_stats = json.loads(ps_path.read_text()) if ps_path.exists() else {}

        # SỬA: input_size=62
        self.mlp = YogaMLP(62, len(self.class_names)).to(self.device)
        self.mlp.load_state_dict(torch.load(d / 'yoga_best_v2.pth', map_location=self.device))
        self.mlp.eval()
        self.sm       = StateMachine()
        self.smoother = KeypointSmoother(alpha=0.4)
        self.debouncer = FeedbackDebouncer(n_show=5, n_hide=5)

    def set_target(self, pose: str):
        self.sm.target_pose  = pose
        self.sm.state        = PoseState.IDLE
        self.sm.pending_count = 0
        self.smoother.reset()
        self.debouncer.reset()

    def reset_reps(self):
        self.sm.reps       = 0
        self.sm.state      = PoseState.IDLE
        self.smoother.reset()
        self.debouncer.reset()
        self.sm.hold_start = 0.0

    @property
    def supported_poses(self): return self.class_names

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        out = frame.copy()

        # 1. YOLOv8
        results = self.yolo(frame, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return PipelineResult(out,'',0,{},{},['Không phát hiện người trong frame.'],
                                  PoseState.IDLE, self.sm.reps, '⏸ Chờ...', False)

        boxes  = results[0].boxes
        kp_all = results[0].keypoints.data
        idx = int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy])) if len(boxes) > 1 else 0
        raw_kpts = kp_all[idx].cpu().numpy()

        # 1b. EMA Keypoint Smoother — làm mượt tọa độ trước mọi xử lý
        raw_kpts = self.smoother.smooth(raw_kpts)

        # 2. Gatekeeper (Người gác cổng nghiêm ngặt)
        kp_conf = raw_kpts[:, 2]
        
        # 💡 Bổ sung: Bắt buộc phải thấy ít nhất 1 bên Vai (để đảm bảo có thân trên)
        has_shoulder = kp_conf[5] > 0.4 or kp_conf[6] > 0.4
        
        has_hip = kp_conf[11] > 0.4 or kp_conf[12] > 0.4
        has_leg = (kp_conf[13] > 0.4 or kp_conf[14] > 0.4 or kp_conf[15] > 0.4 or kp_conf[16] > 0.4)
        
        # 💡 Điều kiện: Bắt buộc phải có đủ 3 bộ phận (Vai - Hông - Chân)
        if not (has_shoulder and has_hip and has_leg):
            self.sm.state = PoseState.IDLE
            self.sm.pending_count = 0
            self.smoother.reset()   # Xóa dữ liệu cũ để tránh ảnh hưởng khi thấy lại người
            self.debouncer.reset()
            return PipelineResult(out, 'INCOMPLETE_BODY', 0.0, {}, {}, 
                                  ['⚠️ Hãy lùi lại để camera thấy toàn thân (Vai - Hông - Chân)'], 
                                  PoseState.IDLE, self.sm.reps, '⚠️ Camera bị khuất', True)

        # 3. Features
        feat, angles_raw = _extract_features(raw_kpts)
        if feat is None:
            return PipelineResult(out,'',0,{},{},['Không đủ keypoint.'], PoseState.IDLE, self.sm.reps, '⏸ Chờ...', True)

        # 4. MLP inference
        feat_s = self.scaler.transform(feat.reshape(1,-1))
        t      = torch.tensor(feat_s, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.mlp(t), dim=1).cpu().numpy()[0]
        
        idx_pred   = int(np.argmax(probs))
        pred_pose  = self.class_names[idx_pred]
        confidence = float(probs[idx_pred])
        all_scores = {n: round(float(probs[i]),4) for i,n in enumerate(self.class_names)}

        # 5. Sanity Check (Bảo vệ vòng ngoài)
        valid_form = _validate_form(pred_pose, angles_raw, self.pose_stats)

        # 6 & 7. Ép về IDLE và Render
        if not valid_form:
            # 💡 ÉP VỀ IDLE NẾU DÁNG VÔ LÝ
            pred_pose = "IDLE"
            confidence = 0.0
            self.sm.update("IDLE", 1.0, True)  # Reset State Machine
            pred_pose_display = "IDLE (Đang chờ...)"
            feedback = ["🧘 Hãy bước vào tư thế Yoga để hệ thống nhận diện!"]
            is_correct = True  # Giữ khung xương màu xanh lá cho êm dịu
        else:
            self.sm.update(pred_pose, confidence, valid_form)
            if self.sm.state in (PoseState.ACTIVE, PoseState.HOLDING):
                pred_pose_display = pred_pose
                feedback_raw = _generate_feedback(pred_pose, angles_raw, self.pose_stats, self.sm.target_pose)
                feedback = self.debouncer.update(feedback_raw)
            else:
                pred_pose_display = pred_pose
                feedback = self.debouncer.update([self.sm.status_text])
            is_correct = (not any('🔴' in f or '🟡' in f or '❌' in f for f in feedback))

        _draw_skeleton(out, raw_kpts, is_correct)
        _draw_hud(out, pred_pose_display, confidence, self.sm, feedback)

        angles_display = {k: round(v,1) for k,v in angles_raw.items() if v >= 0}

        return PipelineResult(out, pred_pose_display, confidence, all_scores, angles_display, 
                              feedback, self.sm.state, self.sm.reps, self.sm.status_text, True)