import pandas as pd
import numpy as np
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
# Kita langsung pakai X, TANPA StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Melatih Model SVM...")
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True) 
model.fit(X_train, y_train)

print("4. Evaluasi & Ekspor...")
y_pred = model.predict(X_test)
print(f" -> AKURASI VALIDASI: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'svm_asl_model.pkl')
print("SELESAI! Model telah dibekukan (Tanpa Scaler).")