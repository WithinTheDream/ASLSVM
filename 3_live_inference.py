import cv2
import mediapipe as mp
import numpy as np
import joblib

print("1. Memuat Otak AI (SVM)...")
# HANYA memuat model, HAPUS baris load scaler
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

print("\n--- SISTEM AKTIF ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
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

            # Format array 2D
            input_data = np.array([normalized_coords])
            
            # --- MASUKKAN LANGSUNG KE MODEL TANPA SCALER ---
            probabilities = model.predict_proba(input_data)[0]
            max_index = np.argmax(probabilities)
            predicted_class = classes[max_index]
            confidence = probabilities[max_index] * 100

            # --- LOGIKA "NOTHING" ---
# --- LOGIKA KELAS "NOTHING" DAN THRESHOLD ---
            if predicted_class == 'X' or confidence < 75.0:
                # Jika sistem menebak 'X' (gestur acak) ATAU ragu-ragu
                text = "Gestur Tidak Dikenal..."
                color = (0, 0, 255) # Merah
            else:
                text = f"Deteksi: {predicted_class} ({confidence:.1f}%)"
                color = (0, 255, 0) # Hijau
            
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('ASL Live Inference (SVM)', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()