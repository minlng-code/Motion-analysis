import cv2
import mediapipe as mp
import numpy as np
import utils
from collections import deque


class OneEuroFilter:
    """
    Simple One Euro Filter for smoothing scalar signals (joint angles).
    Tham số mặc định tuned cho video ~30 FPS.
    """
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.003, dcutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0

    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        # First time: chỉ lưu lại
        if self.x_prev is None:
            self.x_prev = x
            return x

        # derivative
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)

        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class RehabDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        # Model pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
        )

        # Smoothing / filter
        self.prev_angle = 0
        self.smoothing_factor = 0.6
        self.angle_window = deque(maxlen=5)
        self.angle_filter = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.005, dcutoff=1.0)
        self.last_speed = 0.0
        self._prev_for_speed = None

        # Stage counters
        self.down_counter = 0
        self.up_counter = 0
        self.lost_counter = 0

        # === AUTO-CALIB DATA ===
        # Lưu max/min ước lượng theo từng bài để suy ra threshold
        self.calib_data = {
            "Bicep Curl": {"max": None, "min": None},
            "Squat": {"max": None, "min": None},
            "Lunges": {"max": None, "min": None},
        }
        # Ngưỡng hiện tại (sau auto-calib)
        self.thresholds = {}  # {exercise: {"DOWN_TH": x, "UP_TH": y}}
        # Đánh dấu đã auto-calibrate trong session này chưa
        self.auto_calibrated = {
            "Bicep Curl": False,
            "Squat": False,
            "Lunges": False,
        }

        # Ngưỡng mặc định (fallback trước khi auto-calib đủ dữ liệu)
        self.default_thresholds = {
            "Bicep Curl": {"DOWN_TH": 150, "UP_TH": 40},
            "Squat": {"DOWN_TH": 90, "UP_TH": 165},
            "Lunges": {"DOWN_TH": 95, "UP_TH": 165},
        }

        self.reset_session()

    def reset_session(self):
        """Reset dữ liệu theo session (không giữ ngưỡng auto-calib giữa các bài)."""
        self.session_data = {
            "reps": 0,
            "stage": None,
            "feedback": "Stand inside frame",
            "color": (255, 255, 255),
            "min_angle": 180,
            "max_angle": 0,
        }
        self.angle_history = []
        self.prev_angle = 0
        self.angle_window.clear()
        self.down_counter = 0
        self.up_counter = 0
        self.lost_counter = 0
        self.last_speed = 0.0
        self._prev_for_speed = None
        self.angle_filter = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.005, dcutoff=1.0)

        # Reset auto-calib cho session mới
        for ex in self.calib_data:
            self.calib_data[ex]["max"] = None
            self.calib_data[ex]["min"] = None
            self.auto_calibrated[ex] = False
        self.thresholds.clear()

    # ===================== AUTO-CALIB =====================

    def _recalc_thresholds(self, exercise_type: str):
        """Tính DOWN/UP threshold dựa trên max/min đã calibrate."""
        data = self.calib_data.get(exercise_type)
        if not data:
            return

        max_a = data.get("max")
        min_a = data.get("min")
        if max_a is None or min_a is None:
            return

        delta = max_a - min_a
        if delta <= 0:
            return

        margin = 0.15 * delta  # 15% biên độ làm vùng đệm

        if exercise_type == "Bicep Curl":
            # Góc lớn = tay duỗi, góc nhỏ = tay gập
            down_th = max_a - margin   # > down_th = tay duỗi đủ
            up_th = min_a + margin     # < up_th = gập đủ
        else:
            # Squat/Lunges: góc lớn = đứng thẳng, góc nhỏ = xuống sâu
            down_th = min_a + margin   # < down_th = xuống đủ sâu
            up_th = max_a - margin     # > up_th = đứng thẳng đủ

        self.thresholds[exercise_type] = {
            "DOWN_TH": int(down_th),
            "UP_TH": int(up_th),
        }

    def _get_thresholds(self, exercise_type: str):
        """Lấy ngưỡng hiện tại (ưu tiên auto-calib, nếu chưa thì dùng default)."""
        if exercise_type in self.thresholds:
            return (
                self.thresholds[exercise_type]["DOWN_TH"],
                self.thresholds[exercise_type]["UP_TH"],
            )
        base = self.default_thresholds.get(exercise_type, {"DOWN_TH": 150, "UP_TH": 40})
        return base["DOWN_TH"], base["UP_TH"]

    def _auto_calibrate_if_needed(self, exercise_type: str):
        """
        Tự động calibrate sau 3+ reps đầu:
        - Dùng percentile 95/5 của angle_history để ước lượng max/min
        - Dùng EMA (giống loss L2 mượt) để cập nhật max/min, giảm nhiễu
        """
        if self.auto_calibrated.get(exercise_type, False):
            return

        # Cần ít nhất 3 reps & đủ frame để ước lượng tin cậy
        if self.session_data["reps"] < 3:
            return
        if len(self.angle_history) < 60:  # ~2s @30fps
            return

        history = np.array(self.angle_history, dtype=np.float32)
        max_est = np.percentile(history, 95)
        min_est = np.percentile(history, 5)

        # Smoothing như loss: threshold mới = 0.7*old + 0.3*estimate
        prev = self.calib_data[exercise_type]
        alpha = 0.7

        if prev["max"] is None:
            max_smooth = max_est
        else:
            max_smooth = alpha * prev["max"] + (1.0 - alpha) * max_est

        if prev["min"] is None:
            min_smooth = min_est
        else:
            min_smooth = alpha * prev["min"] + (1.0 - alpha) * min_est

        # Cập nhật calib data
        self.calib_data[exercise_type]["max"] = float(max_smooth)
        self.calib_data[exercise_type]["min"] = float(min_smooth)

        # Tính threshold từ calib_data
        self._recalc_thresholds(exercise_type)
        self.auto_calibrated[exercise_type] = True

    # ===================== CORE POSE PROCESSING =====================

    def calculate_angle(self, a, b, c):
        """Tính góc hình học giữa 3 điểm a-b-c (b là đỉnh)."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def _smooth_angle(self, raw_angle: float) -> int:
        """
        Lọc nhiễu: One-Euro + Moving Average,
        và tính tốc độ góc để anti-cheat.
        """
        filtered = self.angle_filter(raw_angle)

        self.angle_window.append(filtered)
        smoothed_angle = float(np.mean(self.angle_window))

        if self._prev_for_speed is not None:
            self.last_speed = abs(smoothed_angle - self._prev_for_speed) * 30.0
        self._prev_for_speed = smoothed_angle

        return int(smoothed_angle)

    def process_frame(self, frame, exercise_type):
        """Xử lý 1 frame, trả về frame vẽ + session_data + current_angle."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_angle = 0
        h, w, _ = image.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            def get_landmark(idx):
                lm = landmarks[idx]
                # Nới visibility: <0.3 mới coi là mất
                if lm.visibility is not None and lm.visibility < 0.3:
                    raise ValueError("Low visibility")
                x = min(max(lm.x, 0.0), 1.0)
                y = min(max(lm.y, 0.0), 1.0)
                return [x, y]

            try:
                # Chọn khớp tùy bài tập
                if exercise_type == "Bicep Curl":
                    indices = [11, 13, 15]  # Vai, Khuỷu, Cổ tay (trái)
                elif exercise_type in ["Squat", "Lunges"]:
                    indices = [23, 25, 27]  # Hông, Gối, Cổ chân (trái)
                else:
                    indices = [11, 13, 15]

                p1 = get_landmark(indices[0])
                p2 = get_landmark(indices[1])
                p3 = get_landmark(indices[2])
                self.lost_counter = 0

                raw_angle = self.calculate_angle(p1, p2, p3)
                current_angle = self._smooth_angle(raw_angle)

                # Anti-cheat: tốc độ góc quá cao
                if self.last_speed > 1200:
                    self.session_data["feedback"] = "Too fast! Control your movement"
                    self.session_data["color"] = (0, 165, 255)

                # Threshold (mặc định hoặc đã auto-calib)
                DOWN_TH, UP_TH = self._get_thresholds(exercise_type)
                MIN_FRAMES_STAGE = 3

                # Tọa độ vẽ góc
                joint_pos = (int(p2[0] * w), int(p2[1] * h))

                # Extra landmarks cho form Squat/Lunge
                shoulder_L = None
                if exercise_type in ["Squat", "Lunges"]:
                    try:
                        shoulder_L = get_landmark(11)
                    except ValueError:
                        shoulder_L = None

                # === LOGIC THEO BÀI TẬP ===

                # BICEP CURL
                if exercise_type == "Bicep Curl":
                    if current_angle > DOWN_TH:
                        self.down_counter += 1
                        self.up_counter = 0
                    elif current_angle < UP_TH:
                        self.up_counter += 1
                        self.down_counter = 0
                    else:
                        self.down_counter = 0
                        self.up_counter = 0

                    if (
                        self.down_counter >= MIN_FRAMES_STAGE
                        and self.session_data["stage"] != "down"
                    ):
                        self.session_data["stage"] = "down"
                        if "Too fast" not in self.session_data["feedback"]:
                            self.session_data["feedback"] = "Curl Up"
                        self.session_data["color"] = (255, 255, 255)

                    if (
                        self.up_counter >= MIN_FRAMES_STAGE
                        and self.session_data["stage"] == "down"
                    ):
                        self.session_data["stage"] = "up"
                        self.session_data["reps"] += 1
                        if "Too fast" not in self.session_data["feedback"]:
                            self.session_data["feedback"] = "Good Rep!"
                        self.session_data["color"] = (0, 255, 0)
                        utils.play_success()
                        self.up_counter = 0
                        self.down_counter = 0

                # SQUAT
                elif exercise_type == "Squat":
                    if current_angle < DOWN_TH:
                        self.down_counter += 1
                        self.up_counter = 0
                    elif current_angle > UP_TH:
                        self.up_counter += 1
                        self.down_counter = 0
                    else:
                        if self.session_data["stage"] != "down":
                            if "Too fast" not in self.session_data["feedback"]:
                                self.session_data["feedback"] = "Lower! (Go deeper)"
                            self.session_data["color"] = (0, 165, 255)
                        self.down_counter = 0
                        self.up_counter = 0

                    if (
                        self.down_counter >= MIN_FRAMES_STAGE
                        and self.session_data["stage"] != "down"
                    ):
                        self.session_data["stage"] = "down"
                        if "Too fast" not in self.session_data["feedback"]:
                            self.session_data["feedback"] = "Stand Up"
                        self.session_data["color"] = (0, 0, 255)

                    if (
                        self.up_counter >= MIN_FRAMES_STAGE
                        and self.session_data["stage"] == "down"
                    ):
                        self.session_data["stage"] = "up"
                        self.session_data["reps"] += 1
                        if "Too fast" not in self.session_data["feedback"]:
                            self.session_data["feedback"] = "Perfect Squat!"
                        self.session_data["color"] = (0, 255, 0)
                        utils.play_success()
                        self.up_counter = 0
                        self.down_counter = 0

                    # FORM CHECK SQUAT
                    if self.session_data["stage"] == "down":
                        knee_x = p2[0]
                        ankle_x = p3[0]
                        if knee_x - ankle_x > 0.12:
                            self.session_data["feedback"] = "Knee too far forward"
                            self.session_data["color"] = (0, 0, 255)

                        if shoulder_L is not None:
                            trunk_angle = self.calculate_angle(shoulder_L, p1, p2)
                            if trunk_angle < 150:
                                self.session_data["feedback"] = "Keep your back more upright"
                                self.session_data["color"] = (0, 0, 255)

                # LUNGES
                elif exercise_type == "Lunges":
                    if current_angle < DOWN_TH:
                        self.down_counter += 1
                        self.up_counter = 0
                    elif current_angle > UP_TH:
                        self.up_counter += 1
                        self.down_counter = 0
                    else:
                        self.down_counter = 0
                        self.up_counter = 0

                    if (
                        self.down_counter >= MIN_FRAMES_STAGE
                        and self.session_data["stage"] != "down"
                    ):
                        self.session_data["stage"] = "down"
                        if "Too fast" not in self.session_data["feedback"]:
                            self.session_data["feedback"] = "Push Up"
                        self.session_data["color"] = (0, 0, 255)

                    if (
                        self.up_counter >= MIN_FRAMES_STAGE
                        and self.session_data["stage"] == "down"
                    ):
                        self.session_data["stage"] = "up"
                        self.session_data["reps"] += 1
                        if "Too fast" not in self.session_data["feedback"]:
                            self.session_data["feedback"] = "Good Lunge!"
                        self.session_data["color"] = (0, 255, 0)
                        utils.play_success()
                        self.up_counter = 0
                        self.down_counter = 0

                    # FORM CHECK LUNGES
                    if self.session_data["stage"] == "down":
                        knee_x = p2[0]
                        ankle_x = p3[0]
                        if knee_x - ankle_x > 0.12:
                            self.session_data["feedback"] = "Front knee too far forward"
                            self.session_data["color"] = (0, 0, 255)

                # Cập nhật thống kê session
                self.angle_history.append(current_angle)
                self.session_data["min_angle"] = min(
                    self.session_data["min_angle"], current_angle
                )
                self.session_data["max_angle"] = max(
                    self.session_data["max_angle"], current_angle
                )

                # Sau khi có đủ rep/frame -> auto-calib
                self._auto_calibrate_if_needed(exercise_type)

                # HUD
                cv2.circle(image, joint_pos, 28, (255, 255, 255), -1)
                status_color = (
                    (0, 255, 0)
                    if "Good" in self.session_data["feedback"]
                    or "Perfect" in self.session_data["feedback"]
                    else (0, 0, 0)
                )
                cv2.circle(image, joint_pos, 28, status_color, 3)

                text = str(int(current_angle))
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )[0]
                text_x = joint_pos[0] - text_size[0] // 2
                text_y = joint_pos[1] + text_size[1] // 2
                cv2.putText(
                    image,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

            except ValueError:
                self.lost_counter += 1
                self.session_data["feedback"] = "Adjust Camera / Body"
                self.session_data["color"] = (0, 0, 255)

                if self.lost_counter >= 5:
                    cv2.putText(
                        image,
                        "LOST TRACKING",
                        (50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

            except Exception:
                pass

        return image, self.session_data, current_angle

    def compute_rom_and_fatigue(self, exercise_type: str):
        """
        Tính:
        - rom_score (%): so sánh biên độ thực tế với biên độ auto-calib (max-min)
        - fatigue_flag: 'Low', 'Moderate', 'High' dựa trên việc biên độ giảm dần
        """
        calib = self.calib_data.get(exercise_type, {})
        max_a = calib.get("max")
        min_a = calib.get("min")
        rom_score = None

        if max_a is not None and min_a is not None and max_a > min_a:
            delta_expected = max_a - min_a
            delta_session = (
                self.session_data["max_angle"] - self.session_data["min_angle"]
            )
            if delta_expected > 0:
                rom_score = max(0.0, min(120.0, 100.0 * delta_session / delta_expected))

        fatigue_flag = None
        history = self.angle_history
        if len(history) >= 90:  # ~3s @30fps
            n = len(history)
            first = history[: n // 3]
            last = history[-n // 3 :]
            min_first = min(first)
            min_last = min(last)

            diff = min_last - min_first
            if exercise_type in ["Squat", "Lunges"]:
                if diff > 15:
                    fatigue_flag = "High"
                elif diff > 7:
                    fatigue_flag = "Moderate"
                else:
                    fatigue_flag = "Low"
            else:
                if diff > 10:
                    fatigue_flag = "High"
                elif diff > 5:
                    fatigue_flag = "Moderate"
                else:
                    fatigue_flag = "Low"

        return rom_score, fatigue_flag
