
import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
import cv2
import numpy as np
import mediapipe as mp
import joblib
import collections
import re
from PIL import Image
import customtkinter as ctk
import pyttsx3
import threading
import time


def resource_path(relative_path):
    """Mendapatkan path resource baik saat development maupun setelah dibuild dengan PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
# ==========================================
# KONFIGURASI TEMA UI
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
ctk.set_widget_scaling(1.0)
ctk.set_window_scaling(1.0)

# Palet warna kustom — satu aksen utama + warna semantik seperlunya
COL_BG          = "#0b0d13"
COL_CARD        = "#141722"
COL_CARD_BORDER = "#242838"
COL_ACCENT      = "#7C5CFF"
COL_ACCENT_DIM  = "#332966"
COL_TEXT_MUTED  = "#8890a3"
COL_SUCCESS     = "#3ddc97"
COL_SUCCESS_HOVER = "#2fbd82"
COL_DANGER      = "#ff5c72"
COL_DANGER_HOVER = "#e14a60"

# Kamus teks antarmuka (Indonesia / English)
TEXTS = {
    "id": {
        "subtitle": "Pengenalan Bahasa Isyarat Real-time",
        "offline": "●  OFFLINE",
        "scanning": "●  MEMINDAI",
        "live_detection": "DETEKSI LIVE",
        "idle_hint": "Arahkan tangan ke kamera",
        "waiting_hand": "Menunggu tangan...",
        "z_swipe": "Sapuan Z terdeteksi...",
        "accuracy": "Akurasi",
        "hold_progress": "PROGRESS TAHAN",
        "camera_source": "SUMBER KAMERA",
        "mirror": "Mirror Kamera",
        "start_cam": "▶  MULAI KAMERA",
        "stop_cam": "⏸  HENTIKAN KAMERA",
        "clear_history": "🗑  HAPUS RIWAYAT",
        "translated_text": "📝 Teks Diterjemahkan",
        "camera_offline": "📷\n\nKamera Offline",
        "camera_fail": "⚠️\n\nGagal membuka {cam}!",
        "empty_text": "Belum ada teks terdeteksi...",
        "backspace_tip": "⌨️  Tombol Backspace = hapus huruf terakhir",
        "cam_prefix": "Kamera",
    },
    "en": {
        "subtitle": "Real-time Sign Language Recognition",
        "offline": "●  OFFLINE",
        "scanning": "●  SCANNING",
        "live_detection": "LIVE DETECTION",
        "idle_hint": "Point your hand at the camera",
        "waiting_hand": "Waiting for hand...",
        "z_swipe": "Z swipe detected...",
        "accuracy": "Accuracy",
        "hold_progress": "HOLD PROGRESS",
        "camera_source": "CAMERA SOURCE",
        "mirror": "Mirror Camera",
        "start_cam": "▶  START CAMERA",
        "stop_cam": "⏸  STOP CAMERA",
        "clear_history": "🗑  CLEAR HISTORY",
        "translated_text": "📝 Translated Text",
        "camera_offline": "📷\n\nCamera Offline",
        "camera_fail": "⚠️\n\nFailed to open {cam}!",
        "empty_text": "No text detected yet...",
        "backspace_tip": "⌨️  Backspace key = delete last letter",
        "cam_prefix": "Camera",
    },
}


class ASLDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Translator")
        self.root.geometry("1060x740")
        self.root.minsize(900, 650)
        self.root.resizable(True, True)
        self.root.configure(fg_color=COL_BG)

        # 1. Inisialisasi Model AI
        print("Memuat AI...")
        self.model = joblib.load(resource_path("svm_asl_model.pkl"))
        self.classes = self.model.classes_

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 2. Sistem Audio (Text-to-Speech)
        self.audio_queue = []
        self.is_speaking = False
        threading.Thread(target=self.audio_worker, daemon=True).start()

        # 3. Variabel Kamera & Logika UI
        self.cap = None
        self.is_scanning = False
        self.lang = "id"

        # 4. Variabel Logika Live Inference Asli
        self.current_word = ""
        self.sentence = ""

        self.last_char = None
        self.frames_held = 0
        self.REQUIRED_FRAMES = 15

        self.hand_missing_frames = 0
        self.SPACE_TIMEOUT_FRAMES = 30

        self.index_path_x = collections.deque(maxlen=20)
        self.Z_SWEEP_THRESHOLD = 0.15
        self.cooldown_frames = 0
        self.COOLDOWN_LIMIT = 20

        self.setup_ui()

        # Shortcut keyboard untuk menghapus huruf
        self.root.bind("<BackSpace>", self.delete_last_char)

    # ==========================================
    # BAHASA UI (ID / EN)
    # ==========================================
    def t(self, key):
        return TEXTS[self.lang][key]

    def set_language(self, value):
        self.lang = "id" if value == "ID" else "en"
        self.apply_language()

    def apply_language(self):
        self.subtitle_label.configure(text=self.t("subtitle"))
        self.live_detection_label.configure(text=self.t("live_detection"))
        self.hold_progress_label.configure(text=self.t("hold_progress"))
        self.camera_source_label.configure(text=self.t("camera_source"))
        self.mirror_switch.configure(text=self.t("mirror"))
        self.btn_clear.configure(text=self.t("clear_history"))
        self.history_header_label.configure(text=self.t("translated_text"))
        self.tip_label.configure(text=self.t("backspace_tip"))

        self.status_pill.configure(
            text=self.t("scanning") if self.is_scanning else self.t("offline"),
            text_color=COL_SUCCESS if self.is_scanning else COL_DANGER
        )
        self.btn_toggle.configure(text=self.t("stop_cam") if self.is_scanning else self.t("start_cam"))

        if not self.is_scanning:
            self.video_label.configure(text=self.t("camera_offline"))
            self.letter_label.configure(text="🤚", text_color=COL_TEXT_MUTED)
            self.confidence_label.configure(text=self.t("idle_hint"))

        # Perbarui pilihan kamera tanpa mengubah kamera yang sedang dipilih
        try:
            current_index = int(re.search(r'\d+', self.cam_dropdown.get()).group())
        except Exception:
            current_index = 0
        new_values = [f'{self.t("cam_prefix")} {i}' for i in range(4)]
        self.cam_dropdown.configure(values=new_values)
        self.cam_dropdown.set(new_values[current_index])

        self.update_history_display()

    def audio_worker(self):
        """Pengatur lalu lintas antrean suara. Berjalan di latar belakang."""
        while True:
            if len(self.audio_queue) > 0 and not self.is_speaking:
                text = self.audio_queue.pop(0)

                def run_tts():
                    """Thread sekali pakai untuk menghindari kemacetan SAPI5 Windows"""
                    self.is_speaking = True
                    try:
                        tts = pyttsx3.init()
                        tts.setProperty('rate', 150)
                        voices = tts.getProperty('voices')
                        if len(voices) > 1:
                            tts.setProperty('voice', voices[1].id)
                        tts.say(text)
                        tts.runAndWait()
                    except Exception as e:
                        print("Audio Error:", e)
                    finally:
                        self.is_speaking = False

                threading.Thread(target=run_tts, daemon=True).start()

            time.sleep(0.05)

    def setup_ui(self):
        # ---------- PENGATURAN WINDOW ----------
        self.root.resizable(True, True) 
        self.root.minsize(960, 680) 

        # ---------- HEADER ----------
        header = ctk.CTkFrame(self.root, height=68, corner_radius=14, fg_color=COL_CARD, border_width=1, border_color=COL_CARD_BORDER)
        header.pack(fill="x", padx=20, pady=(15, 10))

        ctk.CTkLabel(header, text="🤟", font=("Segoe UI Emoji", 24)).place(x=18, y=16)
        ctk.CTkLabel(header, text="ASL Translator", font=("Segoe UI", 20, "bold"), text_color="white").place(x=56, y=10)
        
        self.subtitle_label = ctk.CTkLabel(header, text="", font=("Segoe UI", 12), text_color=COL_TEXT_MUTED)
        self.subtitle_label.place(x=56, y=39)

        self.status_pill = ctk.CTkLabel(header, text="", font=("Segoe UI", 13, "bold"), text_color=COL_DANGER)
        self.status_pill.pack(side="right", padx=20, pady=24)
        
        self.lang_switch = ctk.CTkSegmentedButton(
            header, values=["ID", "EN"], command=self.set_language,
            width=104, height=32, font=("Segoe UI", 12, "bold"), corner_radius=10,
            fg_color=COL_BG, selected_color=COL_ACCENT, selected_hover_color=COL_ACCENT,
            unselected_color=COL_CARD, unselected_hover_color=COL_CARD_BORDER,
            text_color="white"
        )
        self.lang_switch.set("ID")
        self.lang_switch.pack(side="right", padx=10, pady=18)

        # ---------- MAIN CONTENT (VIDEO + DASHBOARD) ----------
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=5)

        # VIDEO FRAME (KIRI) - Akan melebar dan menyusut otomatis
        self.video_frame = ctk.CTkFrame(main_container, corner_radius=16, fg_color=COL_CARD, border_width=2, border_color=COL_CARD_BORDER)
        self.video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_label = ctk.CTkLabel(self.video_frame, text="📷\n\nKamera Offline", font=("Segoe UI", 22), text_color=COL_TEXT_MUTED, justify="center")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # DASHBOARD (KANAN) - Ukuran tetap agar tombol tidak berantakan
        self.dashboard_frame = ctk.CTkFrame(main_container, width=320, corner_radius=16, fg_color=COL_CARD, border_width=1, border_color=COL_CARD_BORDER)
        self.dashboard_frame.pack(side="right", fill="y")
        self.dashboard_frame.pack_propagate(False) 

        INNER_W = 276  

        self.live_detection_label = ctk.CTkLabel(self.dashboard_frame, text="", font=("Segoe UI", 13, "bold"), text_color=COL_TEXT_MUTED)
        self.live_detection_label.place(x=22, y=15)

        self.letter_card = ctk.CTkFrame(self.dashboard_frame, width=INNER_W, height=150, corner_radius=14, fg_color=COL_BG, border_width=1, border_color=COL_CARD_BORDER)
        self.letter_card.place(x=22, y=40)

        self.letter_label = ctk.CTkLabel(self.letter_card, text="🤚", font=("Segoe UI Emoji", 64, "bold"), text_color=COL_TEXT_MUTED)
        self.letter_label.place(relx=0.5, rely=0.4, anchor="center")

        self.confidence_label = ctk.CTkLabel(self.letter_card, text="", font=("Segoe UI", 12), text_color=COL_TEXT_MUTED)
        self.confidence_label.place(relx=0.5, rely=0.82, anchor="center")

        self.hold_progress_label = ctk.CTkLabel(self.dashboard_frame, text="", font=("Segoe UI", 11, "bold"), text_color=COL_TEXT_MUTED)
        self.hold_progress_label.place(x=22, y=200)

        self.progress_bar = ctk.CTkProgressBar(self.dashboard_frame, width=INNER_W, height=10, corner_radius=5, progress_color=COL_ACCENT, fg_color=COL_CARD_BORDER)
        self.progress_bar.place(x=22, y=225)
        self.progress_bar.set(0.0)

        self.camera_source_label = ctk.CTkLabel(self.dashboard_frame, text="", font=("Segoe UI", 11, "bold"), text_color=COL_TEXT_MUTED)
        self.camera_source_label.place(x=22, y=255)

        self.cam_dropdown = ctk.CTkOptionMenu(self.dashboard_frame, values=["Kamera 0", "Kamera 1", "Kamera 2", "Kamera 3"], width=INNER_W, height=35, corner_radius=10, font=("Segoe UI", 13), fg_color=COL_BG, button_color=COL_CARD_BORDER, button_hover_color=COL_ACCENT_DIM, dropdown_fg_color=COL_CARD)
        self.cam_dropdown.place(x=22, y=280)

        self.mirror_var = ctk.BooleanVar(value=True)
        self.mirror_switch = ctk.CTkSwitch(self.dashboard_frame, text="", variable=self.mirror_var, font=("Segoe UI", 13), progress_color=COL_ACCENT)
        self.mirror_switch.place(x=22, y=330)

        self.btn_toggle = ctk.CTkButton(self.dashboard_frame, text="", font=("Segoe UI", 15, "bold"), corner_radius=12, width=INNER_W, height=45, fg_color=COL_SUCCESS, hover_color=COL_SUCCESS_HOVER, text_color="#0b0d13", command=self.toggle_camera)
        self.btn_toggle.place(x=22, y=370)

        self.btn_clear = ctk.CTkButton(self.dashboard_frame, text="", font=("Segoe UI", 13, "bold"), corner_radius=12, width=INNER_W, height=40, fg_color="transparent", border_width=1, border_color=COL_DANGER, text_color=COL_DANGER, hover_color=COL_CARD_BORDER, command=self.clear_history)
        self.btn_clear.place(x=22, y=425)

        # ---------- RIWAYAT TEKS (BAWAH) ----------
        self.history_frame = ctk.CTkFrame(self.root, height=120, corner_radius=16, fg_color=COL_CARD, border_width=1, border_color=COL_CARD_BORDER)
        self.history_frame.pack(fill="x", padx=20, pady=(5, 20))
        self.history_frame.pack_propagate(False) 

        # Ini dia label yang hilang sebelumnya!
        self.history_header_label = ctk.CTkLabel(self.history_frame, text="", font=("Segoe UI", 13, "bold"), text_color=COL_TEXT_MUTED)
        self.history_header_label.place(x=22, y=10)

        self.btn_backspace = ctk.CTkButton(self.history_frame, text="⌫", font=("Segoe UI", 16, "bold"), corner_radius=10, width=42, height=32, fg_color=COL_CARD_BORDER, hover_color=COL_ACCENT_DIM, text_color=COL_ACCENT, command=self.delete_last_char)
        self.btn_backspace.pack(side="right", anchor="n", padx=15, pady=10)

        self.history_box = ctk.CTkTextbox(self.history_frame, height=50, corner_radius=10, fg_color=COL_BG, font=("Consolas", 22), text_color="white", wrap="word", activate_scrollbars=False)
        self.history_box.pack(fill="x", padx=22, pady=(35, 5))
        self.history_box.configure(state="disabled")

        self.tip_label = ctk.CTkLabel(self.history_frame, text="", font=("Segoe UI", 11), text_color=COL_TEXT_MUTED)
        self.tip_label.place(x=22, y=90)

        self.apply_language()

    def toggle_camera(self):
        if not self.is_scanning:
            selected_cam_str = self.cam_dropdown.get()
            cam_index = int(re.search(r'\d+', selected_cam_str).group())

            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                self.video_label.configure(text=self.t("camera_fail").format(cam=selected_cam_str),
                                            text_color=COL_DANGER)
                return

            self.video_label.configure(text_color="white")
            self.video_frame.configure(border_color=COL_ACCENT)
            self.is_scanning = True
            self.status_pill.configure(text=self.t("scanning"), text_color=COL_SUCCESS)
            self.btn_toggle.configure(text=self.t("stop_cam"), fg_color=COL_DANGER, hover_color=COL_DANGER_HOVER,
                                       text_color="white")
            self.cam_dropdown.configure(state="disabled")
            self.update_frame()
        else:
            self.is_scanning = False
            self.status_pill.configure(text=self.t("offline"), text_color=COL_DANGER)
            self.video_frame.configure(border_color=COL_CARD_BORDER)
            self.btn_toggle.configure(text=self.t("start_cam"), fg_color=COL_SUCCESS, hover_color=COL_SUCCESS_HOVER,
                                       text_color="#0b0d13")
            self.cam_dropdown.configure(state="normal")
            if self.cap:
                self.cap.release()
            self.video_label.configure(image="", text=self.t("camera_offline"), text_color=COL_TEXT_MUTED)
            self.letter_label.configure(text="🤚", text_color=COL_TEXT_MUTED)
            self.confidence_label.configure(text=self.t("idle_hint"))
            self.progress_bar.set(0.0)

    def clear_history(self):
        self.current_word = ""
        self.sentence = ""
        self.index_path_x.clear()
        self.update_history_display()

    def delete_last_char(self, event=None):
        """Menghapus huruf terakhir dari kata yang sedang diketik,
        atau dari kalimat yang sudah terbentuk jika kata saat ini kosong."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence.strip():
            temp = self.sentence.rstrip()
            temp = temp[:-1]
            self.sentence = (temp + " ") if temp else ""

        # Reset status debounce agar tidak langsung menambah huruf lagi
        self.last_char = None
        self.frames_held = 0

        self.update_history_display()

    def update_history_display(self):
        full_text_display = f"{self.sentence}{self.current_word}"
        self.history_box.configure(state="normal")
        self.history_box.delete("1.0", "end")
        self.history_box.insert("1.0", full_text_display if full_text_display else self.t("empty_text"))
        self.history_box.configure(state="disabled")

    def update_frame(self):
        if not self.is_scanning or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        if self.mirror_var.get():
            frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        predicted_class = '0'
        confidence = 0.0

        if results.multi_hand_landmarks:
            self.hand_missing_frames = 0

            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                raw_coords = []
                for lm in hand_landmarks.landmark:
                    raw_coords.extend([lm.x, lm.y, lm.z])

                wx, wy, wz = raw_coords[0], raw_coords[1], raw_coords[2]
                normalized = []
                for i in range(21):
                    normalized.extend([raw_coords[i*3]-wx, raw_coords[(i*3)+1]-wy, raw_coords[(i*3)+2]-wz])

                input_data = np.array([normalized])
                probabilities = self.model.predict_proba(input_data)[0]
                max_index = np.argmax(probabilities)
                predicted_class = self.classes[max_index]
                confidence = probabilities[max_index] * 100

                index_x = hand_landmarks.landmark[8].x

                # ==========================================
                # FSM Z SWEEP DETECTION
                # ==========================================
                if self.cooldown_frames > 0:
                    self.cooldown_frames -= 1
                    predicted_class = 'Menunggu...'
                    confidence = 0.0
                    self.last_char = None
                    self.frames_held = 0
                else:
                    if predicted_class == 'D':
                        self.index_path_x.append(index_x)

                        if len(self.index_path_x) > 15:
                            min_x = min(self.index_path_x)
                            max_x = max(self.index_path_x)

                            if (max_x - min_x) > self.Z_SWEEP_THRESHOLD:
                                if len(self.current_word) > 0 and self.current_word[-1] == 'D':
                                    self.current_word = self.current_word[:-1]

                                self.current_word += 'Z'
                                self.audio_queue.append('Z')
                                self.cooldown_frames = self.COOLDOWN_LIMIT
                                self.index_path_x.clear()
                    else:
                        self.index_path_x.clear()

                # ==========================================
                # LOGIKA DEBOUNCER
                # ==========================================
                if self.cooldown_frames == 0:
                    if predicted_class != '0' and confidence > 75.0:
                        if predicted_class == self.last_char:
                            self.frames_held += 1
                            if self.frames_held == self.REQUIRED_FRAMES:
                                self.current_word += predicted_class
                                self.audio_queue.append(predicted_class)
                        else:
                            self.last_char = predicted_class
                            self.frames_held = 0
                    else:
                        self.last_char = None
                        self.frames_held = 0
        else:
            # ==========================================
            # LOGIKA DETEKSI SPASI SAAT TANGAN HILANG
            # ==========================================
            self.hand_missing_frames += 1

            if self.hand_missing_frames == self.SPACE_TIMEOUT_FRAMES:
                if len(self.current_word) > 0:
                    self.audio_queue.append(self.current_word)
                    self.sentence += self.current_word + " "
                    self.current_word = ""

            self.last_char = None
            self.frames_held = 0

        # Update Tampilan Teks Riwayat (Sentence + Current Word)
        self.update_history_display()

        # Update UI Dashboard Kanan
        display_letter = predicted_class if (predicted_class != '0' and confidence > 75.0) else "-"
        if self.cooldown_frames > 0:
            display_letter = "Z"

        shown_text = "🤚" if display_letter == "-" else display_letter
        letter_color = COL_TEXT_MUTED if shown_text == "🤚" else COL_ACCENT
        self.letter_label.configure(text=shown_text, text_color=letter_color)

        if display_letter != "-" and predicted_class != 'Menunggu...':
            self.confidence_label.configure(text=f"{self.t('accuracy')}: {confidence:.0f}%")
        elif predicted_class == 'Menunggu...':
            self.confidence_label.configure(text=self.t('z_swipe'))
        else:
            self.confidence_label.configure(text=self.t('waiting_hand'))

        self.progress_bar.set(min(self.frames_held / self.REQUIRED_FRAMES, 1.0))

        # Render Kamera Mengikuti Ukuran Window Saat Ini
        current_w = self.video_frame.winfo_width()
        current_h = self.video_frame.winfo_height()
        
        # Penjaga (Guard) agar tidak error saat aplikasi pertama kali ditarik/dibuka
        if current_w < 100: current_w = 640
        if current_h < 100: current_h = 480

        img = Image.fromarray(rgb_frame)
        img = img.resize((current_w, current_h))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(current_w, current_h))

        self.video_label.configure(image=ctk_img, text="")

        # Looping Frame
        self.root.after(15, self.update_frame)
#test

if __name__ == "__main__":
    root = ctk.CTk()
    app = ASLDesktopApp(root)
    root.mainloop()