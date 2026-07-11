import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# csv for storing the dataset
CSV_FILE = 'dataset_asl_bersih.csv'
if not os.path.exists(CSV_FILE):
    cols = []
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])
    cols.append('label')
    pd.DataFrame(columns=cols).to_csv(CSV_FILE, index=False)
    print(f"File {CSV_FILE} baru dibuat.")

# Webcam (choose 0, or 1, or 2 depending on your system)
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

print("\n--- INSTRUKSI REKAM ---")
print("1. Posisikan tangan hingga kerangka muncul.")
print("2. Tekan huruf di keyboard (a, b, c,) untuk merekam 1 frame data ke huruf tersebut.")
print("3. Tekan 'ESC' untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raw_coords = []
            for landmark in hand_landmarks.landmark:
                raw_coords.extend([landmark.x, landmark.y, landmark.z])
            
            # wrist coordinates for normalization
            wrist_x = raw_coords[0]
            wrist_y = raw_coords[1]
            wrist_z = raw_coords[2]
            
            normalized_coords = []
            for i in range(21):
                normalized_coords.append(raw_coords[i*3] - wrist_x)
                normalized_coords.append(raw_coords[(i*3)+1] - wrist_y)
                normalized_coords.append(raw_coords[(i*3)+2] - wrist_z)

            key = cv2.waitKey(1) & 0xFF
            
            valid_keys = ['a', 'd', 'm', 'b', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'x']
            
            if chr(key) in valid_keys:
                label = chr(key).upper()
                normalized_coords.append(label)
                
                # Simpan ke CSV
                with open(CSV_FILE, 'a') as f:
                    pd.DataFrame([normalized_coords]).to_csv(f, header=False, index=False)
                print(f"Terekam: 1 data untuk kelas '{label}'")

    cv2.imshow('ASL Data Collector', frame)

    if cv2.waitKey(1) & 0xFF == 27: # Tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
print("Kamera ditutup. Data aman.")