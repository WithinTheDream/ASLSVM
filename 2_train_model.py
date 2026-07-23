import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time  # <-- Tambahkan library time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("1. Memuat Dataset Spasial...")
df = pd.read_csv('dataset_asl_bersih.csv')

print("1.5 Menyeimbangkan Kelas...")
min_samples = df['label'].value_counts().min()
df = df.groupby('label').sample(n=min_samples, random_state=42)
print(f" -> Distribusi Baru:\n{df['label'].value_counts()}")

X = df.drop('label', axis=1).values
y = df['label'].values

print("\n2. Membagi Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Melatih Model SVM...")
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True) 
model.fit(X_train, y_train)

print("4. Evaluasi & Ekspor...")

# --- MENGHITUNG INFERENCE TIME ---
start_time = time.time()          # Catat waktu mulai
y_pred = model.predict(X_test)    # Proses prediksi berjalan
end_time = time.time()            # Catat waktu selesai
# ---------------------------------

# Hitung total waktu dan rata-rata per frame (latency)
total_inference_time = end_time - start_time
jumlah_sampel = len(y_test)
latency_per_sample = (total_inference_time / jumlah_sampel) * 1000  # Dikali 1000 untuk mengubah detik ke milidetik

print(f" -> AKURASI VALIDASI: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f" -> TOTAL WAKTU PREDIKSI : {total_inference_time:.4f} detik (untuk {jumlah_sampel} sampel)")
print(f" -> LATENCY PER SAMPEL   : {latency_per_sample:.4f} ms\n")

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_,
            linewidths=.5)

plt.title('Confusion Matrix - ASL SVM Model', fontsize=18, pad=20)
plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
plt.ylabel('True Label', fontsize=14, labelpad=10)
plt.savefig('confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

joblib.dump(model, 'svm_asl_model.pkl')
print("SELESAI! Model telah dibekukan (Tanpa Scaler).")