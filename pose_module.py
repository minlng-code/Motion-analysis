import cv2
import mediapipe as mp
import numpy as np
import utils

class RehabDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        # Tăng độ tin cậy lên 0.7 để loại bỏ nhiễu background
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1)
        self.reset_session()
        
        # Biến cho bộ lọc làm mượt (Smoothing)
        self.prev_angle = 0
        self.smoothing_factor = 0.6  # 0.6 = Mượt vừa phải, phản hồi nhanh. 0.9 = Rất mượt nhưng lag.

    def reset_session(self):
        """Reset toàn bộ dữ liệu"""
        self.session_data = {
            "reps": 0,
            "stage": None, # 'up' hoặc 'down'
            "feedback": "Stand inside frame",
            "color": (255, 255, 255),
            "min_angle": 180,
            "max_angle": 0
        }
        self.angle_history = []
        self.prev_angle = 0

    def calculate_angle(self, a, b, c):
        """Tính góc hình học giữa 3 điểm"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Tính arctan2 để lấy góc
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def _smooth_angle(self, raw_angle):
        """
        Bộ lọc nhiễu (Exponential Moving Average).
        Giúp góc không bị nhảy loạn xạ.
        """
        # Nếu là frame đầu tiên
        if self.prev_angle == 0:
            self.prev_angle = raw_angle
            return raw_angle
        
        # Công thức: Smooth = alpha * Raw + (1 - alpha) * Prev
        smoothed_angle = (self.smoothing_factor * raw_angle) + ((1 - self.smoothing_factor) * self.prev_angle)
        self.prev_angle = smoothed_angle
        return int(smoothed_angle)

    def process_frame(self, frame, exercise_type):
        """Xử lý hình ảnh và trả về dữ liệu đã khử nhiễu"""
        # 1. Chuyển màu
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_angle = 0
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape 
            
            try:
                # --- CHECK 1: ĐỘ TIN CẬY (VISIBILITY) ---
                def get_landmark(idx):
                    lm = landmarks[idx]
                    # Nếu độ tin cậy < 65% hoặc tọa độ ngoài màn hình -> Bỏ qua
                    if lm.visibility < 0.65 or not (0 <= lm.x <= 1 and 0 <= lm.y <= 1):
                        raise ValueError("Low visibility")
                    return [lm.x, lm.y]

                # Xác định các điểm khớp cần lấy
                indices = []
                if exercise_type == "Bicep Curl":
                    indices = [11, 13, 15] # Vai, Khuỷu, Cổ tay (Trái)
                elif exercise_type in ["Squat", "Lunges"]:
                    indices = [23, 25, 27] # Hông, Gối, Cổ chân (Trái)

                try:
                    p1 = get_landmark(indices[0])
                    p2 = get_landmark(indices[1])
                    p3 = get_landmark(indices[2])
                except ValueError:
                    # Nếu không nhìn rõ khớp -> Báo lỗi ngay
                    self.session_data["feedback"] = "Adjust Camera / Body"
                    self.session_data["color"] = (0, 0, 255) # Đỏ
                    cv2.putText(image, "LOST TRACKING", (50, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return image, self.session_data, 0

                # --- CHECK 2: TÍNH GÓC & LÀM MƯỢT ---
                raw_angle = self.calculate_angle(p1, p2, p3)
                current_angle = self._smooth_angle(raw_angle) # Áp dụng bộ lọc
                
                # Tọa độ vẽ HUD
                joint_pos = (int(p2[0] * w), int(p2[1] * h))

                # --- CHECK 3: LOGIC BÀI TẬP (STRICT MODE) ---
                
                # >> BICEP CURL
                if exercise_type == "Bicep Curl":
                    # Xuống phải thật sâu (> 160) mới tính là giãn cơ
                    if current_angle > 160:
                        self.session_data["stage"] = "down"
                        self.session_data["feedback"] = "Curl Up"
                        self.session_data["color"] = (255, 255, 255)
                    
                    # Lên phải thật cao (< 30) và TRƯỚC ĐÓ phải đang ở 'down'
                    if current_angle < 30 and self.session_data["stage"] == "down":
                        self.session_data["stage"] = "up"
                        self.session_data["reps"] += 1
                        self.session_data["feedback"] = "Good Rep!"
                        self.session_data["color"] = (0, 255, 0)
                        utils.play_success()

                # >> SQUAT
                elif exercise_type == "Squat":
                    if current_angle < 85: # Xuống sâu hơn chút (85) để chắc chắn
                        self.session_data["stage"] = "down"
                        self.session_data["feedback"] = "Stand Up"
                        self.session_data["color"] = (0, 0, 255)
                    elif current_angle < 110 and self.session_data["stage"] != "down":
                        self.session_data["feedback"] = "Lower! (<90)"
                        self.session_data["color"] = (0, 165, 255)

                    if current_angle > 165 and self.session_data["stage"] == "down":
                        self.session_data["stage"] = "up"
                        self.session_data["reps"] += 1
                        self.session_data["feedback"] = "Perfect!"
                        self.session_data["color"] = (0, 255, 0)
                        utils.play_success()

                # >> LUNGES
                elif exercise_type == "Lunges":
                    if current_angle < 95:
                        self.session_data["stage"] = "down"
                        self.session_data["feedback"] = "Push Up"
                        self.session_data["color"] = (0, 0, 255)
                    
                    if current_angle > 165 and self.session_data["stage"] == "down":
                        self.session_data["stage"] = "up"
                        self.session_data["reps"] += 1
                        self.session_data["feedback"] = "Good Lunge!"
                        self.session_data["color"] = (0, 255, 0)
                        utils.play_success()

                # Cập nhật Metrics
                self.angle_history.append(current_angle)
                self.session_data["min_angle"] = min(self.session_data["min_angle"], current_angle)
                self.session_data["max_angle"] = max(self.session_data["max_angle"], current_angle)

                # --- VẼ HUD CHUYÊN NGHIỆP ---
                # Vòng tròn ngoài
                cv2.circle(image, joint_pos, 28, (255, 255, 255), -1)
                # Viền trạng thái (Xanh nếu đúng, Đỏ nếu sai form)
                status_color = (0, 255, 0) if "Good" in self.session_data["feedback"] or "Perfect" in self.session_data["feedback"] else (0, 0, 0)
                cv2.circle(image, joint_pos, 28, status_color, 3)
                
                # Số đo góc
                text = str(int(current_angle))
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = joint_pos[0] - text_size[0] // 2
                text_y = joint_pos[1] + text_size[1] // 2
                cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

                # Vẽ xương
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            except Exception as e:
                pass

        return image, self.session_data, current_angle