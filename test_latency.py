import cv2
import joblib
import time
import csv
import pandas as pd
import serial
from ultralytics import YOLO

# --- Konfigurasi ---
KAMERA_INDEX = 0
PORT_GPS = "/dev/ttyS0"
BAUDRATE_GPS = 9600
JUMLAH_ITERASI = 20  # Mengambil 10 sampel data untuk dirata-rata

# --- Load Model (Di luar perhitungan waktu karena ini inisialisasi awal) ---
print("Memuat model YOLO dan Decision Tree...")
model = YOLO("best.pt")
dt_model = joblib.load("decision_tree.pkl")
encoder_objek = joblib.load("encoder_objek.pkl")
encoder_bahaya = joblib.load("encoder_bahaya.pkl")

# --- Fungsi Tiling (Sesuai kodemu sebelumnya agar waktu YOLO akurat) ---
def split_into_4(frame):
    h, w = frame.shape[:2]
    half_h, half_w = h // 2, w // 2
    pad_h, pad_w = int(half_h * 0.15), int(half_w * 0.15)
    coords = [
        (0, 0, half_w + pad_w, half_h + pad_h),
        (half_w - pad_w, 0, w, half_h + pad_h),
        (0, half_h - pad_h, half_w + pad_w, h),
        (half_w - pad_w, half_h - pad_h, w, h),
    ]
    quadrants = []
    for (x1, y1, x2, y2) in coords:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        quadrants.append(frame[y1:y2, x1:x2])
    return quadrants

def run_latency_test():
    cap = cv2.VideoCapture(KAMERA_INDEX)
    gps_serial = serial.Serial(PORT_GPS, baudrate=BAUDRATE_GPS, timeout=1.0)
    
    nama_file = 'hasil_waktu_komputasi.csv'
    
    with open(nama_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Menyesuaikan header persis dengan tabel skripsimu
        writer.writerow(["Iterasi", "T1_Akuisisi_Citra", "T2_Pembacaan_GPS", "T3_Deteksi_YOLO", "T4_Klasifikasi_DT", "T5_Total_End_to_End"])
        
        print(f"\n--- Memulai {JUMLAH_ITERASI} Iterasi Pengujian Waktu ---")
        
        for i in range(1, JUMLAH_ITERASI + 1):
            t_start = time.perf_counter()
            
            # ---------------------------------------------------------
            # TAHAP 1: Akuisisi Citra
            # ---------------------------------------------------------
            ret, frame = cap.read()
            t_akuisisi = time.perf_counter()
            
            # ---------------------------------------------------------
            # TAHAP 2: Pembacaan GPS (Membaca 1 baris NMEA)
            # ---------------------------------------------------------
            try:
                # Membaca buffer serial mentah untuk mengukur kecepatan komunikasi hardware
                gps_serial.readline() 
            except:
                pass
            t_gps = time.perf_counter()
            
            # ---------------------------------------------------------
            # TAHAP 3: Deteksi Objek (YOLOv8 dengan metode 4 Kuadran)
            # ---------------------------------------------------------
            if ret:
                kuadran_list = split_into_4(frame)
                for q_img in kuadran_list:
                    # Jalankan inferensi pada setiap kuadran
                    _ = model(q_img, conf=0.65, imgsz=320, verbose=False)
            t_yolo = time.perf_counter()
            
            # ---------------------------------------------------------
            # TAHAP 4: Klasifikasi (Decision Tree)
            # ---------------------------------------------------------
            # Kita simulasikan ada 1 objek yang diproses ke Decision Tree
            try:
                objek_enc = encoder_objek.transform(["KERIKIL"])[0]
                fitur = pd.DataFrame([[objek_enc, 0.85, 4500]], columns=["objek", "confidence", "area_bbox"])
                pred = dt_model.predict(fitur)[0]
                _ = encoder_bahaya.inverse_transform([pred])[0]
            except:
                pass
            t_klasifikasi = time.perf_counter()
            
            # ---------------------------------------------------------
            # PERHITUNGAN SELISIH WAKTU (Dalam Detik)
            # ---------------------------------------------------------
            waktu_t1 = t_akuisisi - t_start
            waktu_t2 = t_gps - t_akuisisi
            waktu_t3 = t_yolo - t_gps
            waktu_t4 = t_klasifikasi - t_yolo
            waktu_t5_total = t_klasifikasi - t_start
            
            writer.writerow([i, waktu_t1, waktu_t2, waktu_t3, waktu_t4, waktu_t5_total])
            print(f"Iterasi {i:02d} selesai | Total Waktu: {waktu_t5_total:.4f} dtk (YOLO: {waktu_t3:.4f} dtk)")
            
    cap.release()
    gps_serial.close()
    print(f"\nSelesai! Data kelima tahap berhasil disimpan di {nama_file}")

if __name__ == "__main__":
    run_latency_test()