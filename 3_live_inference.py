import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading
import collections
import time

# ==========================================
# main configuration
# ==========================================
CAMERA_INDEX = 2 # Indeks kamera (0 = default, 1 = kamera eksternal, 2 = kamera USB, dll)
MIRROR_CAMERA = False  # Set True jika ingin membalikkan kamera (efek cermin)
# ==========================================

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

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
audio_queue = []
is_speaking = False

def audio_worker():
    """Pengatur lalu lintas antrean suara. Berjalan di latar belakang."""
    global is_speaking
    while True:
        if len(audio_queue) > 0 and not is_speaking:
            text = audio_queue.pop(0)
            
            def run_tts():
                """Thread sekali pakai untuk menghindari kemacetan SAPI5 Windows"""
                global is_speaking
                is_speaking = True
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
                    is_speaking = False 
            
            threading.Thread(target=run_tts, daemon=True).start()
        
        time.sleep(0.05)

threading.Thread(target=audio_worker, daemon=True).start()

current_word = ""
sentence = ""

last_char = None
frames_held = 0
REQUIRED_FRAMES = 15 

hand_missing_frames = 0
SPACE_TIMEOUT_FRAMES = 30 

index_path_x = collections.deque(maxlen=20) 
force_z_frames = 0
Z_SWEEP_THRESHOLD = 0.15 
cooldown_frames = 0 
COOLDOWN_LIMIT = 20 

pinky_path = collections.deque(maxlen=30) 
force_j_frames = 0 
J_SWEEP_THRESHOLD = 0.15 

print("--- ASLxSVM ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    if MIRROR_CAMERA:
        frame = cv2.flip(frame, 1)
        
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
            
            pinky_y = hand_landmarks.landmark[20].y
            
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

            index_x = hand_landmarks.landmark[8].x

            # ==========================================
            # FSM Z SWEEP DETECTION
            # ==========================================
            if cooldown_frames > 0:
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
                            if len(current_word) > 0 and current_word[-1] == 'D':
                                current_word = current_word[:-1]
                            
                            current_word += 'Z'
                            audio_queue.append('Z') # SUNTIKKAN 'Z' KE ANTREAN SUARA
                            cooldown_frames = COOLDOWN_LIMIT
                            index_path_x.clear()
                else:
                    index_path_x.clear()

            # ==========================================
            # debounce logic for character recognition
            # ==========================================
            if cooldown_frames == 0:
                if predicted_class != '0' and confidence > 75.0:
                    if predicted_class == last_char:
                        frames_held += 1
                        if frames_held == REQUIRED_FRAMES:
                            current_word += predicted_class 
                            audio_queue.append(predicted_class) # SUNTIKKAN HURUF NORMAL KE ANTREAN
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
        # ==========================================
        # space detection logic when hand is missing
        # ==========================================
        hand_missing_frames += 1
        
        if hand_missing_frames == SPACE_TIMEOUT_FRAMES:
            if len(current_word) > 0:
                audio_queue.append(current_word) # SUNTIKKAN KATA UTUH KE ANTREAN
                sentence += current_word + " "
                current_word = "" 
        
        pinky_path.clear()
        force_j_frames = 0
        last_char = None
        frames_held = 0

    # ==========================================
    # ui overlay for current word and sentence
    # ==========================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 50), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, 380), (640, 480), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"Kata    : {current_word}", (10, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Kalimat : {sentence}", (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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