import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading # Modul untuk menjalankan suara di latar belakang

print("1. Memuat Otak AI (SVM)...")
model = joblib.load('svm_asl_model.pkl')
classes = model.classes_

print(f" -> Kelas siap pakai: {classes}")

print("2. Menyalakan Radar MediaPipe...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# ==========================================
# MESIN SUARA ASINKRON (NON-BLOCKING)
# ==========================================
def speak_text(text):
    """Fungsi ini berjalan di latar belakang agar kamera tidak lag saat sistem bicara."""
    def run_tts():
        tts = pyttsx3.init()
        tts.setProperty('rate', 150) # Kecepatan normal
        tts.say(text)
        tts.runAndWait()
    
    # Memulai proses suara di thread terpisah
    threading.Thread(target=run_tts, daemon=True).start()

# ==========================================
# VARIABEL STATE MACHINE
# ==========================================
current_word = ""
sentence = ""

last_char = None
frames_held = 0
REQUIRED_FRAMES = 15 

hand_missing_frames = 0
SPACE_TIMEOUT_FRAMES = 30 

tracking_j = False
j_start_y = 0
j_frame_counter = 0
J_MOVEMENT_THRESHOLD = 0.05 

print("\n--- SISTEM PENERJEMAH AKTIF ---")
print("[KONTROL KEYBOARD]")
print(" - Tekan 'BACKSPACE' untuk hapus 1 huruf terakhir.")
print(" - Tekan 'R' untuk Reset semua kalimat.")
print(" - Tekan 'ESC' untuk Keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_missing_frames = 0 
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raw_coords = []
            for landmark in hand_landmarks.landmark:
                raw_coords.extend([landmark.x, landmark.y, landmark.z])
            
            wrist_x = raw_coords[0]
            wrist_y = raw_coords[1]
            wrist_z = raw_coords[2]
            
            normalized_coords = []
            for i in range(21):
                normalized_coords.append(raw_coords[i*3] - wrist_x)
                normalized_coords.append(raw_coords[(i*3)+1] - wrist_y)
                normalized_coords.append(raw_coords[(i*3)+2] - wrist_z)

            input_data = np.array([normalized_coords])
            
            probabilities = model.predict_proba(input_data)[0]
            max_index = np.argmax(probabilities)
            predicted_class = classes[max_index]
            confidence = probabilities[max_index] * 100

            # LOGIKA HEURISTIK JEBAKAN HURUF "J"
            if predicted_class == 'I':
                if not tracking_j:
                    tracking_j = True
                    j_start_y = wrist_y
                    j_frame_counter = 0
                else:
                    j_frame_counter += 1
                    if (wrist_y - j_start_y) > J_MOVEMENT_THRESHOLD:
                        predicted_class = 'J'
                        confidence = 99.0
                    
                    if j_frame_counter > 25:
                        tracking_j = False
            else:
                tracking_j = False 

            # LOGIKA DEBOUNCE PENGETIKAN & SUARA PER HURUF
            if predicted_class != 'X' and confidence > 75.0:
                if predicted_class == last_char:
                    frames_held += 1
                    if frames_held == REQUIRED_FRAMES:
                        current_word += predicted_class 
                        # speak_text(predicted_class) # Sistem menyebutkan huruf yang baru saja diketik
                else:
                    last_char = predicted_class
                    frames_held = 0
            else:
                last_char = None
                frames_held = 0

            color = (0, 0, 255) if predicted_class == 'X' or confidence < 75.0 else (0, 255, 0)
            text_radar = f"Deteksi: {predicted_class} ({confidence:.1f}%) | Loading: {frames_held}/{REQUIRED_FRAMES}"
            cv2.putText(frame, text_radar, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    else:
        hand_missing_frames += 1
        
        if hand_missing_frames == SPACE_TIMEOUT_FRAMES:
            if len(current_word) > 0:
                speak_text(current_word)
                sentence += current_word + " "
                current_word = "" 
        
        tracking_j = False
        last_char = None
        frames_held = 0

    cv2.putText(frame, f"Kata: {current_word}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Kalimat: {sentence}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('ASL Live Inference (SVM)', frame)

    # ==========================================
    # KONTROL KEYBOARD
    # ==========================================
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC untuk keluar
        break
    elif key == 8: # BACKSPACE (ASCII 8) untuk menghapus huruf terakhir
        if len(current_word) > 0:
            current_word = current_word[:-1]
        elif len(sentence) > 0:
            # Jika kata kosong, hapus spasi terakhir dari kalimat
            sentence = sentence[:-1] 
    elif key == ord('r'): # Tombol 'R' untuk Reset Semua
        current_word = ""
        sentence = ""

cap.release()
cv2.destroyAllWindows()