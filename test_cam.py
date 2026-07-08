import cv2
import time
import sys

# Sesuaikan dengan index kamera kamu
KAMERA_INDEX = 0

def test_kamera_with_overlay():
    print(f"--- Memulai Test Hardware Kamera di index {KAMERA_INDEX} ---")
    cap = cv2.VideoCapture(KAMERA_INDEX)

    if not cap.isOpened():
        print("Gagal! Kamera tidak terdeteksi. Coba cek colokan USB atau ubah index.")
        sys.exit()

    # 1. Ambil resolusi bawaan dari hardware
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[1] Resolusi Terbaca: {width} x {height}")

    # 2. Pemanasan sensor (Auto-Exposure & White Balance)
    print("\n[2] Pemanasan sensor (10 frame awal)...")
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    # 3. Pengukuran Raw FPS
    print("\n[3] Mengukur Raw FPS (Mengambil 100 frame berturut-turut)...")
    jumlah_frame = 100
    start_time = time.time()
    
    last_frame = None
    for i in range(jumlah_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"    -> Kamera terputus di frame ke-{i}!")
            break
        last_frame = frame  # Simpan frame terakhir untuk diberi teks

    end_time = time.time()
    waktu_total = end_time - start_time
    fps = jumlah_frame / waktu_total

    print(f"    -> Waktu total: {waktu_total:.2f} detik")
    print(f"    -> Kemampuan Hardware Kamera: {fps:.2f} FPS")

    # 4. Tambahkan Teks ke Gambar dan Simpan
    if last_frame is not None:
        text_res = f"Res: {width}x{height}"
        text_fps = f"FPS: {fps:.2f}"
        
        # Pengaturan Font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0) # Warna teks hijau (B, G, R)
        margin = 15       # Jarak dari pinggir frame
        
        # Hitung ukuran teks agar posisinya dinamis (selalu pas di pojok kanan bawah)
        (res_w, res_h), _ = cv2.getTextSize(text_res, font, font_scale, thickness)
        (fps_w, fps_h), _ = cv2.getTextSize(text_fps, font, font_scale, thickness)
        
        # Koordinat untuk teks FPS (Paling bawah)
        fps_x = width - fps_w - margin
        fps_y = height - margin
        
        # Koordinat untuk teks Resolusi (Tepat di atas teks FPS)
        res_x = width - res_w - margin
        res_y = fps_y - fps_h - (margin // 2)
        
        # Tempelkan teks ke frame terakhir yang ditangkap
        cv2.putText(last_frame, text_res, (res_x, res_y), font, font_scale, color, thickness)
        cv2.putText(last_frame, text_fps, (fps_x, fps_y), font, font_scale, color, thickness)
        
        # Simpan ke disk
        nama_file = "test_hardware_kamera_overlay.jpg"
        cv2.imwrite(nama_file, last_frame)
        print(f"\n[4] Sukses! Foto disimpan sebagai '{nama_file}' dengan teks di pojok kanan bawah.")
    else:
        print("\n[Error] Tidak ada frame yang berhasil ditangkap.")

    cap.release()
    print("--- Test Selesai ---")

if __name__ == '__main__':
    test_kamera_with_overlay()