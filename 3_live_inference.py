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
# MESIN SUARA ASINKRON (KATA UTUH - INDONESIA)
# ==========================================
def speak_text(text):
    def run_tts():
        tts = pyttsx3.init()
        tts.setProperty('rate', 140) # Sedikit diperlambat agar pelafalannya jelas
        
        # Mencari dan mengaktifkan suara Bahasa Indonesia
        voices = tts.getProperty('voices')
        for voice in voices:
            if 'Indonesian' in voice.name or 'ID' in voice.id or 'Andika' in voice.name or 'Gadis' in voice.name:
                tts.setProperty('voice', voice.id)
                break
                
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
cooldown_frames = 0 # Jeda waktu agar AI buta sementara setelah mengetik huruf spesial
COOLDOWN_LIMIT = 20 # Durasi buta (sekitar 1 detik)
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

            # ==========================================
            # FSM JEBAKAN HURUF Z (INSTANT BYPASS)
            # ==========================================
            if cooldown_frames > 0:
                # KAMERA DIBUTAKAN SEMENTARA
                cooldown_frames -= 1
                predicted_class = 'Menunggu...'
                confidence = 0.0
                last_char = None
                frames_held = 0
            else:
                if predicted_class == 'D':
                    index_path_x.append(index_x)
                    
                    if len(index_path_x) > 15:
                        min_x = min(index_path_x)
                        max_x = max(index_path_x)
                        
                        if (max_x - min_x) > Z_SWEEP_THRESHOLD:
                            # 1. RETROACTIVE WIPE: Hapus 'D' prematur di ujung kata
                            if len(current_word) > 0 and current_word[-1] == 'D':
                                current_word = current_word[:-1]
                            
                            # 2. INSTANT INJECT: Langsung masukkan Z tanpa loading bar!
                            current_word += 'Z'
                            
                            # 3. AKTIFKAN COOLDOWN: Butakan AI agar sisa jari 'D' tidak terketik
                            cooldown_frames = COOLDOWN_LIMIT
                            index_path_x.clear()
                else:
                    index_path_x.clear()

            # ==========================================
            # LOGIKA DEBOUNCE PENGETIKAN NORMAL
            # ==========================================
            # Hanya jalankan pengetikan normal jika sistem tidak sedang Cooldown
            if cooldown_frames == 0:
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

    # ==========================================
    # RENDER UI (SEMI-TRANSPARENT OVERLAY)
    # ==========================================
    # Buat salinan frame untuk layar transparan
    overlay = frame.copy()
    
    # Gambar kotak hitam di bagian atas (Untuk Radar Biasa)
    cv2.rectangle(overlay, (0, 0), (640, 50), (0, 0, 0), -1)
    
    # Gambar kotak hitam di bagian bawah (Untuk Kata & Kalimat)
    cv2.rectangle(overlay, (0, 380), (640, 480), (0, 0, 0), -1)
    
    # Gabungkan overlay dengan frame asli (Alpha 0.6 = 60% Transparan)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Tulis teks putih di atas area hitam tersebut agar sangat kontras
    cv2.putText(frame, f"Kata    : {current_word}", (10, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Kalimat : {sentence}", (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    # (Pastikan kamu menghapus cv2.putText kata dan kalimat yang lama agar tidak menumpuk)


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