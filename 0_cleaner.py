import pandas as pd

# 1. Targetkan file CSV milikmu
file_name = 'dataset_asl_bersih.csv'
print(f"Membaca {file_name}...")

# 2. Muat data menggunakan Pandas
df = pd.read_csv(file_name)
jumlah_awal = len(df)

# 3. Tentukan target operasi penghapusan
target_hapus = []

# 4. Filter data: Ambil semua baris yang labelnya TIDAK ADA di daftar target_hapus
df_bersih = df[~df['label'].isin(target_hapus)]

# 5. Timpa dan simpan kembali ke file aslinya
df_bersih.to_csv(file_name, index=False)

jumlah_akhir = len(df_bersih)
terhapus = jumlah_awal - jumlah_akhir

print("\n--- OPERASI PEMBERSIHAN SELESAI ---")
print(f"Total baris awal   : {jumlah_awal}")
print(f"Total baris akhir  : {jumlah_akhir}")
print(f"Total data dihapus : {terhapus} baris untuk huruf {target_hapus}")