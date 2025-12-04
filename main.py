import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import time
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pose_module import RehabDetector
import utils


class RehabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rehab Center Management System (Pro Version)")
        self.root.geometry("1200x760")
        self.root.configure(bg="#1e272e")

        self.detector = RehabDetector()
        self.is_running = False
        self.cap = None

        self.current_exercise = tk.StringVar(value="Bicep Curl")
        self.patient_name = tk.StringVar(value="Patient_001")
        self.is_recording = tk.BooleanVar(value=False)
        self.video_writer = None

        self.prev_time = 0
        self.fps_avg = 0

        self.setup_ui()

    def setup_ui(self):
        # HEADER
        header = tk.Frame(self.root, bg="#0fb9b1", height=80)
        header.pack(fill="x", side="top")

        tk.Label(
            header,
            text="REHABILITATION MONITORING CENTER",
            bg="#0fb9b1",
            fg="white",
            font=("Segoe UI", 22, "bold"),
        ).pack(pady=(10, 0))

        tk.Label(
            header,
            text="AI-Powered Motion Analysis & Recovery Tracking",
            bg="#0fb9b1",
            fg="#dfe6e9",
            font=("Segoe UI", 11),
        ).pack(pady=(0, 10))

        # MAIN CONTAINER
        container = tk.Frame(self.root, bg="#1e272e")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # LEFT PANEL
        left_frame = tk.Frame(container, bg="#485460", width=380)
        left_frame.pack(side="left", fill="y", padx=(0, 20))
        left_frame.pack_propagate(False)

        # PATIENT PROFILE
        tk.Label(
            left_frame,
            text="PATIENT PROFILE",
            bg="#485460",
            fg="#0fb9b1",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(25, 5), anchor="w", padx=20)

        tk.Entry(
            left_frame,
            textvariable=self.patient_name,
            font=("Segoe UI", 12),
        ).pack(pady=5, padx=20, fill="x")

        # TRAINING CONFIG
        tk.Label(
            left_frame,
            text="TRAINING CONFIGURATION",
            bg="#485460",
            fg="#0fb9b1",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(20, 5), anchor="w", padx=20)

        ex_combo = ttk.Combobox(
            left_frame,
            textvariable=self.current_exercise,
            state="readonly",
            font=("Segoe UI", 12),
        )
        ex_combo["values"] = ("Bicep Curl", "Squat", "Lunges")
        ex_combo.pack(pady=5, padx=20, fill="x")
        ex_combo.bind("<<ComboboxSelected>>", self.on_exercise_change)

        # RECORDING
        chk_record = tk.Checkbutton(
            left_frame,
            text="Record Video Evidence (.avi)",
            variable=self.is_recording,
            bg="#485460",
            fg="white",
            selectcolor="#1e272e",
            activebackground="#485460",
            font=("Segoe UI", 10),
        )
        chk_record.pack(pady=10, anchor="w", padx=15)

        # START/STOP BUTTONS
        self.btn_start = tk.Button(
            left_frame,
            text="START SESSION",
            bg="#20bf6b",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            bd=0,
            pady=12,
            command=self.start_camera,
            cursor="hand2",
        )
        self.btn_start.pack(pady=20, padx=20, fill="x")

        self.btn_stop = tk.Button(
            left_frame,
            text="STOP & ANALYZE",
            bg="#eb3b5a",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            bd=0,
            pady=12,
            command=self.stop_camera,
            state="disabled",
            cursor="hand2",
        )
        self.btn_stop.pack(pady=5, padx=20, fill="x")

        # HELP BUTTON
        self.btn_help = tk.Button(
            left_frame,
            text="HƯỚNG DẪN CAMERA & TƯ THẾ",
            bg="#00a8ff",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            pady=8,
            command=self.show_help,
            cursor="hand2",
        )
        self.btn_help.pack(pady=10, padx=20, fill="x")

        # STATS
        stats_frame = tk.Frame(left_frame, bg="#1e272e", bd=1, relief="solid")
        stats_frame.pack(fill="x", padx=20, pady=20)

        self.lbl_reps = tk.Label(
            stats_frame,
            text="0",
            bg="#1e272e",
            fg="#0fb9b1",
            font=("Segoe UI", 60, "bold"),
        )
        self.lbl_reps.pack(pady=5)

        tk.Label(
            stats_frame,
            text="REPS COMPLETED",
            bg="#1e272e",
            fg="#b2bec3",
            font=("Segoe UI", 9, "bold"),
        ).pack(pady=(0, 10))

        self.lbl_feedback = tk.Label(
            left_frame,
            text="Ready",
            bg="#485460",
            fg="#f7b731",
            font=("Segoe UI", 16, "bold"),
        )
        self.lbl_feedback.pack(pady=(10, 0))

        self.lbl_angle = tk.Label(
            left_frame,
            text="Joint Angle: 0°",
            bg="#485460",
            fg="white",
            font=("Segoe UI", 13),
        )
        self.lbl_angle.pack()

        # RIGHT PANEL (VIDEO + IDLE SCREEN)
        right_frame = tk.Frame(container, bg="black", bd=2, relief="sunken")
        right_frame.pack(side="right", fill="both", expand=True)

        self.video_label = tk.Label(right_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # Lúc mở app hoặc khi chưa bấm START => màn hình chờ + hướng dẫn
        self.show_idle_screen()

    # ===== IDLE SCREEN & HƯỚNG DẪN BÀI TẬP =====

    def get_exercise_guide_text(self):
        ex = self.current_exercise.get()
        if ex == "Bicep Curl":
            return (
                "BICEP CURL (Tập tay trước)\n\n"
                "• Đứng nghiêng 45°–90° so với camera, tay tập gần camera.\n"
                "• Tư thế bắt đầu: tay duỗi gần như thẳng xuống.\n"
                "• Tư thế kết thúc: gập tay hết mức về phía vai.\n"
                "• Giữ vai cố định, không đánh người.\n"
                "• Hít vào khi hạ tay, thở ra khi gập tay."
            )
        elif ex == "Squat":
            return (
                "SQUAT (Tập chân & mông)\n\n"
                "• Đặt camera bên hông (side view).\n"
                "• Tư thế bắt đầu: đứng thẳng, chân rộng bằng vai.\n"
                "• Ngồi xuống như ngồi ghế, gối không vượt mũi chân quá nhiều.\n"
                "• Giữ lưng thẳng, ngực mở, mắt nhìn trước.\n"
                "• Đứng thẳng người trở lại, siết mông ở cuối chuyển động."
            )
        elif ex == "Lunges":
            return (
                "LUNGES (Tập chân trước & sau)\n\n"
                "• Đặt camera bên hông, thấy rõ chân trước và chân sau.\n"
                "• Bước một chân lên phía trước, hạ người xuống.\n"
                "• Gối trước gần 90°, không vượt mũi chân quá nhiều.\n"
                "• Giữ lưng thẳng, trọng tâm ổn định.\n"
                "• Đẩy người lên và đổi chân nếu cần."
            )
        else:
            return "Chọn một bài tập để xem hướng dẫn."

    def show_idle_screen(self):
        """Hiển thị màn hình chờ với hướng dẫn bài tập hiện tại."""
        width, height = 800, 600
        img = Image.new("RGB", (width, height), color=(15, 20, 30))
        draw = ImageDraw.Draw(img)

        # Load font Unicode (đường dẫn có thể cần chỉnh theo máy)
        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
            font_sub = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
            font_body = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
            font_hint = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
        except Exception:
            font_title = font_sub = font_body = font_hint = None

        title = "READY TO START"
        sub = "Chọn bài tập, đọc hướng dẫn và bấm START SESSION"

        draw.text((40, 40), title, fill=(0, 255, 200), font=font_title)
        draw.text((40, 80), sub, fill=(180, 200, 220), font=font_sub)

        guide = self.get_exercise_guide_text()
        y = 140
        for line in guide.split("\n"):
            draw.text((40, y), line, fill=(230, 230, 230), font=font_body)
            y += 26

        hint = "Hệ thống sẽ tự học biên độ của bạn trong vài rep đầu, không cần calibration."
        draw.text((40, height - 60), hint, fill=(150, 170, 190), font=font_hint)

        img_tk = ImageTk.PhotoImage(img)
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

    def on_exercise_change(self, event=None):
        """Khi đổi bài tập, nếu chưa chạy camera thì update màn hình chờ."""
        if not self.is_running:
            self.show_idle_screen()

    # ===== HƯỚNG DẪN CAMERA =====

    def show_help(self):
        help_text = (
            "HƯỚNG DẪN ĐẶT CAMERA & TƯ THẾ\n\n"
            "A. CÀI ĐẶT CAMERA CHUNG\n"
            "• Đặt camera ngang tầm ngực hoặc vai (không quá thấp).\n"
            "• Khoảng cách: ~1.5–2.0m để thấy trọn thân trên / chân tùy bài.\n"
            "• Tránh ngược sáng (đèn / cửa sổ ở sau lưng) → sẽ mất tracking.\n"
            "• Mặc đồ vừa người, tránh áo quá rộng che khuất khớp.\n\n"
            "B. BICEP CURL (Tay)\n"
            "• Đứng nghiêng 45–90° so với camera, tay tập là tay gần camera.\n"
            "• Tay duỗi gần như thẳng khi bắt đầu, gập hết về phía vai ở cuối.\n"
            "• Trong khi tập: giữ vai cố định, không đánh người.\n\n"
            "C. SQUAT (Gối)\n"
            "• Đặt camera nghiêng bên hông (side view).\n"
            "• Từ đứng thẳng, hạ người như ngồi ghế, gối không vượt mũi chân quá nhiều.\n"
            "• Giữ lưng thẳng, ngực mở.\n\n"
            "D. LUNGES\n"
            "• Đặt camera nghiêng bên hông, thấy rõ chân trước + chân sau.\n"
            "• Bước chân trước, hạ người, gối trước gần 90°.\n"
            "• Giữ lưng thẳng, trọng tâm ổn định.\n\n"
            "Hệ thống sẽ tự động học biên độ ROM của bạn trong vài rep đầu của mỗi bài.\n"
        )
        messagebox.showinfo("Hướng dẫn sử dụng", help_text)

    # ===== CAMERA CONTROL =====

    def start_camera(self):
        if not self.patient_name.get().strip():
            messagebox.showwarning("Warning", "Patient Name cannot be empty!")
            return

        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
                if not self.cap.isOpened():
                    messagebox.showerror("System Error", "No Camera Detected.")
                    return

            if self.is_recording.get():
                if not os.path.exists("recordings"):
                    os.makedirs("recordings")
                filename = (
                    f"recordings/{self.patient_name.get().strip().replace(' ', '_')}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                )
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (800, 600))

            self.detector.reset_session()
            self.is_running = True
            self.prev_time = time.time()
            self.fps_avg = 0

            self.btn_start.config(state="disabled", bg="#95a5a6")
            self.btn_stop.config(state="normal", bg="#eb3b5a")

            self.update_frame()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()

            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                messagebox.showinfo(
                    "Evidence Saved", "Video recording saved to 'recordings/' folder."
                )

            self.btn_start.config(state="normal", bg="#20bf6b")
            self.btn_stop.config(state="disabled", bg="#95a5a6")

            data = self.detector.session_data
            history = self.detector.angle_history
            p_name = self.patient_name.get()
            ex_name = self.current_exercise.get()

            # ROM% & fatigue (dựa trên auto-calib)
            rom_score, fatigue_flag = self.detector.compute_rom_and_fatigue(ex_name)

            if data["reps"] > 0:
                utils.log_session(
                    p_name,
                    ex_name,
                    data["reps"],
                    data["min_angle"],
                    data["max_angle"],
                    rom_score=rom_score,
                    fatigue_flag=fatigue_flag,
                )

                extra = ""
                if rom_score is not None:
                    extra += f"\nROM: {rom_score:.1f}%"
                if fatigue_flag is not None:
                    extra += f"\nFatigue: {fatigue_flag}"

                ans = messagebox.askyesno(
                    "Report",
                    f"Session Finished.\nReps: {data['reps']}{extra}\n\nView Analysis Chart?"
                )
                if ans:
                    utils.show_performance_chart(history, ex_name, p_name)
            else:
                messagebox.showinfo("Info", "No reps recorded.")

            # Quay lại idle screen
            self.show_idle_screen()

    def update_frame(self):
        if self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    frame = cv2.resize(frame, (800, 600))

                    curr_time = time.time()
                    dt = curr_time - self.prev_time
                    fps = 1 / dt if dt > 0 else 0
                    self.prev_time = curr_time
                    self.fps_avg = 0.9 * self.fps_avg + 0.1 * fps

                    processed_frame, data, angle = self.detector.process_frame(
                        frame, self.current_exercise.get()
                    )

                    # FPS
                    cv2.putText(
                        processed_frame,
                        f"FPS: {int(self.fps_avg)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    # REPS overlay
                    cv2.putText(
                        processed_frame,
                        f"REPS: {data['reps']}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )

                    if self.video_writer:
                        self.video_writer.write(processed_frame)

                    # Đồng bộ label bên trái
                    self.lbl_reps.config(text=str(data["reps"]))

                    fb_text = data["feedback"]
                    if "Good" in fb_text or "Perfect" in fb_text:
                        color = "#20bf6b"
                    elif "Ready" in fb_text:
                        color = "#f7b731"
                    elif (
                        "Adjust" in fb_text
                        or "Missing" in fb_text
                        or "Lost" in fb_text
                        or "LOST" in fb_text
                    ):
                        color = "#ff0000"
                    else:
                        color = "#eb3b5a"

                    self.lbl_feedback.config(text=fb_text, fg=color)
                    self.lbl_angle.config(text=f"Joint Angle: {angle}°")

                    img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
                    self.video_label.imgtk = img_tk
                    self.video_label.configure(image=img_tk)

                except Exception as e:
                    print(f"Frame Error: {e}")

            self.root.after(10, self.update_frame)

    def on_close(self):
        if self.is_running:
            self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RehabApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
