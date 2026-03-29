"""
inference_pipeline.py — Giai đoạn 3 theo đặc tả
Ghép YOLOv8, MLP weights, pose_stats và State Machine thành một luồng duy nhất.
app.py chỉ cần gọi pipeline.process_frame(frame) → PipelineResult
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
    def __init__(self, input_size=59, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x): return self.net(x)


# ──────────────────────────────────────────────────────────
# COCO CONFIG
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

JOINT_TRIPLES = [
    (5, 7, 9,  'ang_L_elbow'),
    (6, 8, 10, 'ang_R_elbow'),
    (11,5, 7,  'ang_L_shoulder'),
    (12,6, 8,  'ang_R_shoulder'),
    (5, 11,13, 'ang_L_hip'),
    (6, 12,14, 'ang_R_hip'),
    (11,13,15, 'ang_L_knee'),
    (12,14,16, 'ang_R_knee'),
]
JOINT_DISPLAY = {
    'ang_L_elbow':'Khuỷu trái', 'ang_R_elbow':'Khuỷu phải',
    'ang_L_shoulder':'Vai trái', 'ang_R_shoulder':'Vai phải',
    'ang_L_hip':'Hông trái',    'ang_R_hip':'Hông phải',
    'ang_L_knee':'Gối trái',    'ang_R_knee':'Gối phải',
}
JOINT_DIRECTION = {
    'ang_L_elbow':    ('Duỗi thẳng khuỷu tay trái',    'Gập khuỷu tay trái lại'),
    'ang_R_elbow':    ('Duỗi thẳng khuỷu tay phải',   'Gập khuỷu tay phải lại'),
    'ang_L_shoulder': ('Nâng / mở rộng tay trái',      'Hạ tay trái xuống'),
    'ang_R_shoulder': ('Nâng / mở rộng tay phải',      'Hạ tay phải xuống'),
    'ang_L_hip':      ('Mở rộng hông trái',             'Đứng thẳng hơn ở hông trái'),
    'ang_R_hip':      ('Mở rộng hông phải',             'Đứng thẳng hơn ở hông phải'),
    'ang_L_knee':     ('Gập gối trái sâu hơn',          'Duỗi thẳng chân trái'),
    'ang_R_knee':     ('Gập gối phải sâu hơn',          'Duỗi thẳng chân phải'),
}

Z_WARN  = 1.5
Z_ALERT = 2.5
CONF_THRESH = 0.3


# ──────────────────────────────────────────────────────────
# STATE MACHINE
# ──────────────────────────────────────────────────────────
class PoseState(Enum):
    IDLE    = auto()   # chưa nhận diện được pose ổn định
    PENDING = auto()   # đang đếm frame liên tiếp để xác nhận
    ACTIVE  = auto()   # đã xác nhận, đang trong pose
    HOLDING = auto()   # giữ pose (dùng cho rep counter)

@dataclass
class StateMachine:
    """
    Debouncing: chỉ chuyển sang ACTIVE sau CONFIRM_FRAMES
    liên tiếp cùng một pose với confidence đủ cao.
    Rep counter: đếm số lần hoàn thành một chu kỳ pose.
    """
    CONFIRM_FRAMES: int  = 15    # frames liên tiếp để xác nhận
    CONF_MIN:       float = 0.70  # confidence tối thiểu
    HOLD_SECONDS:   float = 2.0   # giữ pose bao lâu để tính 1 rep

    state:          PoseState = field(default=PoseState.IDLE)
    current_pose:   str       = field(default='')
    pending_pose:   str       = field(default='')
    pending_count:  int       = field(default=0)
    reps:           int       = field(default=0)
    hold_start:     float     = field(default=0.0)

    # Target pose do người dùng chọn ('' = bất kỳ)
    target_pose: str = field(default='')

    def update(self, pred_pose: str, confidence: float) -> 'StateMachine':
        now = time.time()
        active_pose = self.target_pose if self.target_pose else pred_pose

        if confidence < self.CONF_MIN:
            # Confidence thấp → quay về IDLE
            self.state        = PoseState.IDLE
            self.pending_count = 0
            return self

        if self.state == PoseState.IDLE:
            if pred_pose == active_pose:
                self.state        = PoseState.PENDING
                self.pending_pose  = pred_pose
                self.pending_count = 1
            return self

        if self.state == PoseState.PENDING:
            if pred_pose == self.pending_pose:
                self.pending_count += 1
                if self.pending_count >= self.CONFIRM_FRAMES:
                    self.state        = PoseState.ACTIVE
                    self.current_pose  = pred_pose
                    self.hold_start    = now
            else:
                # Pose thay đổi → reset
                self.state         = PoseState.IDLE
                self.pending_count  = 0
            return self

        if self.state in (PoseState.ACTIVE, PoseState.HOLDING):
            if pred_pose == self.current_pose:
                held = now - self.hold_start
                if held >= self.HOLD_SECONDS and self.state == PoseState.ACTIVE:
                    self.state  = PoseState.HOLDING
                    self.reps  += 1
            else:
                # Rời pose → quay về IDLE, chuẩn bị rep mới
                self.state         = PoseState.IDLE
                self.pending_count  = 0
                self.hold_start     = 0.0
            return self

        return self

    @property
    def status_text(self) -> str:
        if self.state == PoseState.IDLE:    return '⏸ Chờ...'
        if self.state == PoseState.PENDING: return f'🔄 Xác nhận... ({self.pending_count}/{self.CONFIRM_FRAMES})'
        if self.state == PoseState.ACTIVE:  return '✅ Đang tập'
        if self.state == PoseState.HOLDING: return '🏆 Giữ tốt!'
        return ''


# ──────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────
def _calc_angle(p1, p2, p3):
    if np.any(p1 == 0) or np.any(p2 == 0) or np.any(p3 == 0):
        return -1.0
    v1, v2 = p1 - p2, p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

def _extract_features(keypoints):
    kp_xy, kp_conf = keypoints[:, :2], keypoints[:, 2]
    hip_c   = (kp_xy[11] + kp_xy[12]) / 2
    sho_c   = (kp_xy[5]  + kp_xy[6])  / 2
    torso   = np.linalg.norm(sho_c - hip_c)
    if torso < 1e-5:
        return None, {}
    norm_xy   = (kp_xy - hip_c) / torso
    conf_feat = np.clip(kp_conf, 0, 1)
    angles_raw, angle_feat = {}, []
    for i, j, k, name in JOINT_TRIPLES:
        deg = _calc_angle(kp_xy[i], kp_xy[j], kp_xy[k])
        angles_raw[name] = deg
        angle_feat.append(deg / 180.0 if deg >= 0 else -1.0)
    feat = np.concatenate([norm_xy.flatten(), conf_feat, np.array(angle_feat)])
    return feat.astype(np.float32), angles_raw


# ──────────────────────────────────────────────────────────
# FEEDBACK (z-score)
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
        label    = JOINT_DISPLAY.get(joint, joint)
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


# ──────────────────────────────────────────────────────────
# RENDERING
# ──────────────────────────────────────────────────────────
def _draw_skeleton(frame, kps, is_correct: bool):
    h, w = frame.shape[:2]
    edge_color = (80, 220, 80) if is_correct else (80, 80, 240)

    for a, b in SKELETON_EDGES:
        pa, pb = kps[a], kps[b]
        if pa[2] < CONF_THRESH or pb[2] < CONF_THRESH: continue
        x1,y1 = int(pa[0]*w), int(pa[1]*h)  # nếu kp normalized
        x2,y2 = int(pb[0]*w), int(pb[1]*h)
        # kp là pixel gốc → không cần nhân
        x1,y1 = int(pa[0]), int(pa[1])
        x2,y2 = int(pb[0]), int(pb[1])
        cv2.line(frame, (x1,y1), (x2,y2), edge_color, 2, cv2.LINE_AA)

    for i, kp in enumerate(kps):
        if kp[2] < CONF_THRESH: continue
        x, y = int(kp[0]), int(kp[1])
        col  = KP_COLOR[KP_GROUP[i]]
        cv2.circle(frame, (x, y), 5, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 8, col, 1,  cv2.LINE_AA)


def _draw_hud(frame, pose, confidence, state_machine: StateMachine,
              feedback: list[str]):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top-left panel
    cv2.rectangle(overlay, (0,0), (300, 110), (15,15,20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, pose.upper(),        (12,36),  cv2.FONT_HERSHEY_DUPLEX,  0.9, (168,224,99), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{confidence*100:.0f}% conf', (12,58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.putText(frame, state_machine.status_text,     (12,78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(frame, f'Reps: {state_machine.reps}', (12,100),cv2.FONT_HERSHEY_SIMPLEX, 0.55,(240,200,80),  1)

    # Feedback lines (bottom)
    y0 = h - 10 - len(feedback) * 22
    for i, line in enumerate(feedback[:4]):   # tối đa 4 dòng
        clean = line.replace('🔴','[!]').replace('🟡','[~]').replace('✅','[ok]').replace('⚠️','[w]')
        col = (80,80,240) if '[!]' in clean else \
              (80,200,240) if '[~]' in clean else (80,220,80)
        cv2.putText(frame, clean, (12, y0 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────
# RESULT DATACLASS
# ──────────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    frame:        np.ndarray         # frame đã vẽ đè HUD + skeleton
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
class YogaPipeline:
    def __init__(self, model_dir: str = 'models'):
        d = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo   = YOLO('yolov8n-pose.pt')

        with open(d / 'label_map.json') as f:
            lm = json.load(f)
        self.class_names = [k for k,_ in sorted(lm.items(), key=lambda x: x[1])]

        with open(d / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        ps_path = d / 'pose_stats.json'
        self.pose_stats = json.loads(ps_path.read_text()) if ps_path.exists() else {}

        self.mlp = YogaMLP(59, len(self.class_names)).to(self.device)
        self.mlp.load_state_dict(torch.load(d / 'yoga_best_v2.pth', map_location=self.device))
        self.mlp.eval()

        self.sm = StateMachine()
        print(f'✅ YogaPipeline ready | {self.class_names} | device={self.device}')

    def set_target(self, pose: str):
        """Đặt pose mục tiêu cho State Machine."""
        self.sm.target_pose = pose
        self.sm.state        = PoseState.IDLE
        self.sm.pending_count = 0

    def reset_reps(self):
        self.sm.reps       = 0
        self.sm.state      = PoseState.IDLE
        self.sm.hold_start = 0.0

    @property
    def supported_poses(self): return self.class_names

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """
        Input : BGR frame (numpy)
        Output: PipelineResult với frame đã được render
        """
        out = frame.copy()

        # 1. YOLOv8 → keypoints
        results = self.yolo(frame, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return PipelineResult(out,'',0,{},{},['Không phát hiện người trong frame.'],
                                  PoseState.IDLE, self.sm.reps, '⏸ Chờ...', False)

        # Lấy person có bounding box lớn nhất
        boxes  = results[0].boxes
        kp_all = results[0].keypoints.data
        if len(boxes) > 1:
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy]
            idx   = int(np.argmax(areas))
        else:
            idx = 0
        raw_kpts = kp_all[idx].cpu().numpy()   # (17, 3)

        # 2. Feature extraction
        feat, angles_raw = _extract_features(raw_kpts)
        if feat is None:
            return PipelineResult(out,'',0,{},{},['Không đủ keypoint để phân tích.'],
                                  PoseState.IDLE, self.sm.reps, '⏸ Chờ...', True)

        # 3. MLP inference
        feat_s = self.scaler.transform(feat.reshape(1,-1))
        t      = torch.tensor(feat_s, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.mlp(t), dim=1).cpu().numpy()[0]
        idx_pred   = int(np.argmax(probs))
        pred_pose  = self.class_names[idx_pred]
        confidence = float(probs[idx_pred])
        all_scores = {n: round(float(probs[i]),4) for i,n in enumerate(self.class_names)}

        # 4. State Machine
        self.sm.update(pred_pose, confidence)

        # 5. Feedback (chỉ khi ACTIVE/HOLDING)
        if self.sm.state in (PoseState.ACTIVE, PoseState.HOLDING):
            feedback = _generate_feedback(pred_pose, angles_raw, self.pose_stats,
                                          self.sm.target_pose)
        else:
            feedback = [self.sm.status_text]

        # 6. Render
        is_correct = (not any('🔴' in f or '🟡' in f for f in feedback))
        _draw_skeleton(out, raw_kpts, is_correct)
        _draw_hud(out, pred_pose, confidence, self.sm, feedback)

        angles_display = {k: round(v,1) for k,v in angles_raw.items() if v >= 0}

        return PipelineResult(
            frame       = out,
            pose        = pred_pose,
            confidence  = confidence,
            all_scores  = all_scores,
            angles      = angles_display,
            feedback    = feedback,
            state       = self.sm.state,
            reps        = self.sm.reps,
            status_text = self.sm.status_text,
            person_found= True,
        )
