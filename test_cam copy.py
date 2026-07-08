import cv2
import time
import sys

# Sesuaikan dengan index kamera kamu (di kodemu sebelumnya pakai 2)
KAMERA_INDEX = 0

def test_kamera():
    print(f"--- Memulai Test Hardware Kamera di index {KAMERA_INDEX} ---")
    
    # Gunakan cv2.CAP_DSHOW jika di Windows, tapi karena ini Raspi (Linux), 
    # cv2.CAP_V4L2 (Video4Linux2) biasanya lebih stabil, atau biarkan default.
    cap = cv2.VideoCapture(KAMERA_INDEX)

    if not cap.isOpened():
        print("Gagal! Kamera tidak terdeteksi. Coba cek colokan USB atau ubah index ke 0 atau 1.")
        sys.exit()

    # Ambil resolusi default dari hardware
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[1] Resolusi Terbaca: {width} x {height}")

    # ==========================================
    # TEST 1: Kualitas Gambar & Auto-Exposure
    # ==========================================
    print("\n[2] Mengambil sampel foto (Warming up sensor 10 frame)...")
    
    # Pemanasan sensor: buang beberapa frame awal agar kamera 
    # sempat menyesuaikan pencahayaan (Auto-Exposure & White Balance)
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    ret, frame = cap.read()
    if ret:
        nama_file = "test_hardware_kamera.jpg"
        cv2.imwrite(nama_file, frame)
        print(f"    -> Foto berhasil disimpan sebagai '{nama_file}'.")
    else:
        print("    -> Gagal mengambil foto dari sensor.")

    # ==========================================
    # TEST 2: Kecepatan Mentah Hardware (Raw FPS)
    # ==========================================
    print("\n[3] Mengukur Raw FPS (Mengambil 100 frame tanpa henti)...")
    jumlah_frame = 100
    start_time = time.time()

    for i in range(jumlah_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"    -> Kamera terputus di frame ke-{i}!")
            break

    end_time = time.time()
    waktu_total = end_time - start_time
    fps = jumlah_frame / waktu_total

    print(f"    -> Waktu total untuk 100 frame: {waktu_total:.2f} detik")
    print(f"    -> Kemampuan Hardware Kamera: {fps:.2f} FPS")

    cap.release()
    print("\n--- Test Selesai ---")

if __name__ == '__main__':
    test_kamera()
