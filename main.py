import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import time
from datetime import datetime
from PIL import Image, ImageTk
from pose_module import RehabDetector
import utils

class RehabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rehab Center Management System (Pro Version)")
        self.root.geometry("1200x760")
        self.root.configure(bg="#1e272e")

        # --- Core Modules ---
        self.detector = RehabDetector()
        self.is_running = False
        self.cap = None
        
        # --- Variables ---
        self.current_exercise = tk.StringVar(value="Bicep Curl")
        self.patient_name = tk.StringVar(value="Patient_001")
        self.is_recording = tk.BooleanVar(value=False)
        self.video_writer = None
        
        # FPS Smoothing
        self.prev_time = 0
        self.fps_avg = 0

        self.setup_ui()

    def setup_ui(self):
        # 1. HEADER
        header = tk.Frame(self.root, bg="#0fb9b1", height=80)
        header.pack(fill="x", side="top")
        
        tk.Label(header, text="REHABILITATION MONITORING CENTER", 
                 bg="#0fb9b1", fg="white", font=("Segoe UI", 22, "bold")).pack(pady=(10, 0))
        tk.Label(header, text="AI-Powered Motion Analysis & Recovery Tracking", 
                 bg="#0fb9b1", fg="#dfe6e9", font=("Segoe UI", 11)).pack(pady=(0, 10))

        # 2. MAIN CONTAINER
        container = tk.Frame(self.root, bg="#1e272e")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # --- LEFT PANEL (Control) ---
        left_frame = tk.Frame(container, bg="#485460", width=360)
        left_frame.pack(side="left", fill="y", padx=(0, 20))
        left_frame.pack_propagate(False)

        # > PATIENT INFO
        tk.Label(left_frame, text="PATIENT PROFILE", bg="#485460", fg="#0fb9b1", font=("Segoe UI", 12, "bold")).pack(pady=(25,5), anchor="w", padx=20)
        tk.Entry(left_frame, textvariable=self.patient_name, font=("Segoe UI", 12)).pack(pady=5, padx=20, fill="x")

        # > EXERCISE SETTINGS
        tk.Label(left_frame, text="TRAINING CONFIGURATION", bg="#485460", fg="#0fb9b1", font=("Segoe UI", 12, "bold")).pack(pady=(20,5), anchor="w", padx=20)
        ex_combo = ttk.Combobox(left_frame, textvariable=self.current_exercise, state="readonly", font=("Segoe UI", 12))
        ex_combo['values'] = ('Bicep Curl', 'Squat', 'Lunges')
        ex_combo.pack(pady=5, padx=20, fill="x")

        # > RECORDING
        chk_record = tk.Checkbutton(left_frame, text="Record Video Evidence (.avi)", 
                                   variable=self.is_recording, bg="#485460", fg="white", 
                                   selectcolor="#1e272e", activebackground="#485460", font=("Segoe UI", 10))
        chk_record.pack(pady=10, anchor="w", padx=15)

        # > BUTTONS
        self.btn_start = tk.Button(left_frame, text="START SESSION", bg="#20bf6b", fg="white",
                                   font=("Segoe UI", 12, "bold"), bd=0, pady=12, command=self.start_camera, cursor="hand2")
        self.btn_start.pack(pady=20, padx=20, fill="x")

        self.btn_stop = tk.Button(left_frame, text="STOP & ANALYZE", bg="#eb3b5a", fg="white",
                                  font=("Segoe UI", 12, "bold"), bd=0, pady=12, command=self.stop_camera, state="disabled", cursor="hand2")
        self.btn_stop.pack(pady=5, padx=20, fill="x")

        # > REAL-TIME STATS
        stats_frame = tk.Frame(left_frame, bg="#1e272e", bd=1, relief="solid")
        stats_frame.pack(fill="x", padx=20, pady=30)
        
        self.lbl_reps = tk.Label(stats_frame, text="0", bg="#1e272e", fg="#0fb9b1", font=("Segoe UI", 60, "bold"))
        self.lbl_reps.pack(pady=5)
        tk.Label(stats_frame, text="REPS COMPLETED", bg="#1e272e", fg="#b2bec3", font=("Segoe UI", 9, "bold")).pack(pady=(0,10))

        self.lbl_feedback = tk.Label(left_frame, text="Ready", bg="#485460", fg="#f7b731", font=("Segoe UI", 16, "bold"))
        self.lbl_feedback.pack(pady=(10,0))
        self.lbl_angle = tk.Label(left_frame, text="Joint Angle: 0°", bg="#485460", fg="white", font=("Segoe UI", 13))
        self.lbl_angle.pack()

        # --- RIGHT PANEL (Camera Feed) ---
        right_frame = tk.Frame(container, bg="black", bd=2, relief="sunken")
        right_frame.pack(side="right", fill="both", expand=True)
        self.video_label = tk.Label(right_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

    def start_camera(self):
        if not self.patient_name.get().strip():
            messagebox.showwarning("Warning", "Patient Name cannot be empty!")
            return

        if not self.is_running:
            # Thử mở camera 0, nếu lỗi thử camera 1
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
                if not self.cap.isOpened():
                    messagebox.showerror("System Error", "No Camera Detected.")
                    return

            # Cấu hình Video Writer
            if self.is_recording.get():
                if not os.path.exists('recordings'): os.makedirs('recordings')
                filename = f"recordings/{self.patient_name.get().strip().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (800, 600))

            self.detector.reset_session()
            self.is_running = True
            self.prev_time = time.time()
            
            self.btn_start.config(state="disabled", bg="#95a5a6")
            self.btn_stop.config(state="normal", bg="#eb3b5a")
            
            self.update_frame()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap: self.cap.release()
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                messagebox.showinfo("Evidence Saved", "Video recording saved to 'recordings/' folder.")

            self.video_label.config(image="")
            self.btn_start.config(state="normal", bg="#20bf6b")
            self.btn_stop.config(state="disabled", bg="#95a5a6")

            # Xử lý báo cáo
            data = self.detector.session_data
            history = self.detector.angle_history
            p_name = self.patient_name.get()
            ex_name = self.current_exercise.get()

            if data["reps"] > 0:
                utils.log_session(p_name, ex_name, data["reps"], data["min_angle"], data["max_angle"])
                ans = messagebox.askyesno("Report", f"Session Finished.\nReps: {data['reps']}\n\nView Analysis Chart?")
                if ans:
                    utils.show_performance_chart(history, ex_name, p_name)
            else:
                messagebox.showinfo("Info", "No reps recorded.")

    def update_frame(self):
        if self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    # 1. Resize chuẩn (Bắt buộc cho VideoWriter)
                    frame = cv2.resize(frame, (800, 600))

                    # 2. Tính FPS (Smoothed)
                    curr_time = time.time()
                    fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
                    self.prev_time = curr_time
                    # Làm mượt FPS: alpha * new + (1-alpha) * old
                    self.fps_avg = 0.9 * self.fps_avg + 0.1 * fps 

                    # 3. AI Processing
                    processed_frame, data, angle = self.detector.process_frame(frame, self.current_exercise.get())

                    # 4. Vẽ FPS
                    cv2.putText(processed_frame, f"FPS: {int(self.fps_avg)}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # 5. Ghi hình
                    if self.video_writer:
                        self.video_writer.write(processed_frame)

                    # 6. Cập nhật UI (Logic màu sắc mới)
                    self.lbl_reps.config(text=str(data["reps"]))
                    
                    fb_text = data["feedback"]
                    # Màu sắc dựa trên keywords từ pose_module
                    if "Good" in fb_text or "Perfect" in fb_text: 
                        color = "#20bf6b" # Green
                    elif "Ready" in fb_text: 
                        color = "#f7b731" # Yellow
                    elif "Adjust" in fb_text or "Missing" in fb_text or "Lost" in fb_text:
                        color = "#ff0000" # Pure Red (Cảnh báo lỗi)
                    else: 
                        color = "#eb3b5a" # Soft Red (Sai tư thế)
                    
                    self.lbl_feedback.config(text=fb_text, fg=color)
                    self.lbl_angle.config(text=f"Joint Angle: {angle}°")

                    # 7. Hiển thị Tkinter
                    img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
                    self.video_label.imgtk = img_tk
                    self.video_label.configure(image=img_tk)

                except Exception as e:
                    print(f"Frame Error: {e}")
                    # Không crash app nếu lỗi 1 frame

            self.root.after(10, self.update_frame)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RehabApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()