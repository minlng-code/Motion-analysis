import pygame
import numpy as np
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import sqlite3 # Thêm thư viện Database

# --- 1. AUDIO SETUP ---
try:
    pygame.mixer.init()
except:
    pass

def make_tone(frequency=880, duration_ms=100, volume=0.1, sample_rate=44100):
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    note = np.sin(frequency * t * 2 * np.pi)
    audio = note * (2**15 - 1) * volume
    audio = audio.astype(np.int16)
    return audio

SUCCESS_TONE = make_tone(880, 150, 0.2)
ERROR_TONE = make_tone(300, 300, 0.2)

def play_success():
    try:
        sound = pygame.sndarray.make_sound(SUCCESS_TONE)
        sound.play()
    except: pass

def play_error():
    try:
        sound = pygame.sndarray.make_sound(ERROR_TONE)
        sound.play()
    except: pass

# --- 2. DATABASE SETUP (NÂNG CẤP) ---
def init_db():
    """Khởi tạo Database nếu chưa tồn tại"""
    try:
        conn = sqlite3.connect('rehab_data.db')
        c = conn.cursor()
        # Tạo bảng lưu trữ phiên tập
        c.execute('''CREATE TABLE IF NOT EXISTS sessions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      patient_name TEXT,
                      exercise TEXT,
                      reps INTEGER,
                      min_angle REAL,
                      max_angle REAL,
                      assessment TEXT)''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Init Error: {e}")

# --- 3. LOGGING (LƯU TRỮ KÉP) ---
def log_session(patient_name, exercise_name, reps, max_rom_flex, max_rom_ext):
    """
    Lưu dữ liệu vào cả CSV (để xem nhanh) và SQLite (để quản lý hệ thống)
    """
    # Đảm bảo DB đã sẵn sàng
    init_db()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assessment = "Excellent" if reps >= 10 else "Good" if reps >= 5 else "Keep Trying"

    try:
        # --- CÁCH 1: LƯU CSV (Backup file Excel) ---
        filename = 'rehab_log.csv'
        file_exists = os.path.isfile(filename)
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Patient ID', 'Exercise', 'Reps', 'Min_Angle', 'Max_Angle', 'Assessment'])
            writer.writerow([timestamp, patient_name, exercise_name, reps, max_rom_flex, max_rom_ext, assessment])

        # --- CÁCH 2: LƯU SQLITE (System Database) ---
        conn = sqlite3.connect('rehab_data.db')
        c = conn.cursor()
        c.execute("INSERT INTO sessions (timestamp, patient_name, exercise, reps, min_angle, max_angle, assessment) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (timestamp, patient_name, exercise_name, reps, max_rom_flex, max_rom_ext, assessment))
        conn.commit()
        conn.close()
        
        print(f"Data saved to System DB & CSV: {reps} reps")
        return True

    except Exception as e:
        print(f"Log Error: {e}")
        return False

# --- 4. VISUALIZATION ---
def show_performance_chart(angle_history, exercise_name, patient_name):
    """Vẽ biểu đồ dao động góc khớp"""
    if not angle_history: return
    
    plt.figure(figsize=(10, 5))
    plt.plot(angle_history, label='Joint Angle', color='#20bf6b', linewidth=2)
    
    # Vẽ đường ngưỡng tham chiếu
    plt.axhline(y=160, color='r', linestyle='--', label='Extension (Straight)')
    
    target = 30 if exercise_name == 'Bicep Curl' else 90
    plt.axhline(y=target, color='b', linestyle='--', label='Flexion (Bent)')
    
    plt.title(f'Analysis: {exercise_name} - Patient: {patient_name}')
    plt.xlabel('Frames (Time)')
    plt.ylabel('Angle (Degrees)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()