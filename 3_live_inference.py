import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading
import collections # Modul FSM untuk memori lintasan

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
# MESIN SUARA ASINKRON (KATA UTUH)
# ==========================================
def speak_text(text):
    def run_tts():
        tts = pyttsx3.init()
        tts.setProperty('rate', 150)
        tts.say(text)
        tts.runAndWait()
    threading.Thread(target=run_tts, daemon=True).start()

# ==========================================
# VARIABEL STATE MACHINE & MEMORI
# ==========================================
current_word = ""
sentence = ""

last_char = None
frames_held = 0
REQUIRED_FRAMES = 15 

hand_missing_frames = 0
SPACE_TIMEOUT_FRAMES = 30 
# Variabel FSM Khusus Z (Index Finger Tracking)
index_path_x = collections.deque(maxlen=20) 
force_z_frames = 0
Z_SWEEP_THRESHOLD = 0.15 # Seberapa lebar tarikan garis Z di layar
# Variabel FSM Khusus J (Pinky Tracking)
pinky_path = collections.deque(maxlen=30) # Mengingat posisi kelingking 30 frame ke belakang
force_j_frames = 0 # Durasi paksaan (State Lock) untuk membajak sistem
J_SWEEP_THRESHOLD = 0.15 # Ambang batas ayunan kelingking ke bawah (0.15 = 15% dari tinggi layar)

print("\n--- SISTEM PENERJEMAH AKTIF ---")

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

            # Ekstraksi Koordinat Mentah
            raw_coords = []
            for landmark in hand_landmarks.landmark:
                raw_coords.extend([landmark.x, landmark.y, landmark.z])
            
            wrist_x = raw_coords[0]
            wrist_y = raw_coords[1]
            wrist_z = raw_coords[2]
            
            # Ambil koordinat UJUNG KELINGKING (Index 20)
            pinky_y = hand_landmarks.landmark[20].y
            
            # Normalisasi Wrist-Centric untuk AI
            normalized_coords = []
            for i in range(21):
                normalized_coords.append(raw_coords[i*3] - wrist_x)
                normalized_coords.append(raw_coords[(i*3)+1] - wrist_y)
                normalized_coords.append(raw_coords[(i*3)+2] - wrist_z)

            input_data = np.array([normalized_coords])
            
            # Prediksi Normal
            probabilities = model.predict_proba(input_data)[0]
            max_index = np.argmax(probabilities)
            predicted_class = classes[max_index]
            confidence = probabilities[max_index] * 100

           # Ambil koordinat UJUNG TELUNJUK (Index 8) absolut
            index_x = hand_landmarks.landmark[8].x

            # FSM JEBAKAN HURUF Z (TRACKING TELUNJUK)
            if force_z_frames > 0:
                predicted_class = 'Z'
                confidence = 99.0
                force_z_frames -= 1
                index_path_x.clear()
            else:
                # Karena Z menggunakan pose D, pemicunya adalah 'D'
                if predicted_class == 'D':
                    index_path_x.append(index_x)
                    
                    if len(index_path_x) > 15:
                        # Cari titik paling kiri dan paling kanan dari jejak telunjuk
                        min_x = min(index_path_x)
                        max_x = max(index_path_x)
                        
                        # Jika selisihnya (lebar zig-zag) melampaui threshold
                        if (max_x - min_x) > Z_SWEEP_THRESHOLD:
                            # Tambahan + 5 frame agar loading bar pengetikan punya waktu untuk penuh
                            force_z_frames = REQUIRED_FRAMES + 5 
                else:
                    index_path_x.clear()

            # ==========================================
            # LOGIKA DEBOUNCE PENGETIKAN
            # ==========================================
            if predicted_class != '0' and confidence > 75.0:
                if predicted_class == last_char:
                    frames_held += 1
                    if frames_held == REQUIRED_FRAMES:
                        current_word += predicted_class 
                else:
                    last_char = predicted_class
                    frames_held = 0
            else:
                last_char = None
                frames_held = 0

            # --- Visualisasi UI Radar ---
            color = (0, 0, 255) if predicted_class == '0' or confidence < 75.0 else (0, 255, 0)
            text_radar = f"Deteksi: {predicted_class} ({confidence:.1f}%) | Loading: {frames_held}/{REQUIRED_FRAMES}"
            cv2.putText(frame, text_radar, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    else:
        # LOGIKA SPASI & SUARA PER KATA
        hand_missing_frames += 1
        
        if hand_missing_frames == SPACE_TIMEOUT_FRAMES:
            if len(current_word) > 0:
                speak_text(current_word) 
                sentence += current_word + " "
                current_word = "" 
        
        pinky_path.clear()
        force_j_frames = 0
        last_char = None
        frames_held = 0

    cv2.putText(frame, f"Kata: {current_word}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Kalimat: {sentence}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('ASL Live Inference (SVM)', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break
    elif key == 8: 
        if len(current_word) > 0:
            current_word = current_word[:-1]
        elif len(sentence) > 0:
            sentence = sentence[:-1] 
    elif key == ord('r'): 
        current_word = ""
        sentence = ""

cap.release()
cv2.destroyAllWindows()