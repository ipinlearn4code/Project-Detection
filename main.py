from ultralytics import YOLO
import cv2, joblib, asyncio
from telegram import Bot
from datetime import datetime
import pandas as pd
from config import TOKEN, CHAT_ID
import time
import os
import time
import serial_asyncio
import pynmea2
import sys
# import platform
import gps  # Pastikan untuk mengakses variabel global gps.GPS_TTFF
# from gps import get_gps_async
# from gps import get_latest_gps, 

import subprocess
import numpy as np

# Pastikan Anda mengimpor gps_loop dan get_latest_gps dari file gps.py Anda
# dari gps import gps_loop, get_latest_gps, GPS_TTFF
from gps import gps_loop, get_latest_gps, GPS_TTFF

async def startup_diagnostics():
    """
    (Permintaan No. 3, 4, 5: Mengambil data cold start, TTFF, ping Telegram, 
    lalu diprint dan dikirim ke Telegram).
    """
    print("--- Menjalankan Diagnostik Startup ---")
    stats = []
    
    # 3. Baca systemd-analyze (Cold Start Raspi)
    try:
        # Menjalankan command terminal dari Python
        boot_time = subprocess.check_output(['systemd-analyze']).decode('utf-8').strip()
        stats.append(f"⏱️ *Cold Start (systemd)*:\n`{boot_time}`")
    except Exception as e:
        stats.append("⏱️ *Cold Start (systemd)*: Gagal dibaca (Mungkin bukan Linux/Raspi)")

    # 4a. Telegram Ping
    t0_ping = time.time()
    try:
        await bot.get_me()
        ping_ms = (time.time() - t0_ping) * 1000
        stats.append(f"🌐 *Telegram Ping*: {ping_ms:.2f} ms")
    except Exception as e:
        stats.append(f"🌐 *Telegram Ping*: Gagal ({e})")

    # 4b. Kamera TTFF (Membuka port kamera dan membaca frame pertama)
    t0_cam = time.time()
    global cap
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            cam_ttff = (time.time() - t0_cam) * 1000
            stats.append(f"📷 *Kamera TTFF*: {cam_ttff:.2f} ms")
        else:
            stats.append("📷 *Kamera TTFF*: Gagal membaca frame")
    else:
        stats.append("📷 *Kamera TTFF*: Kamera tidak terbuka")

    # 4c. YOLO TTFF (Warmup model dengan dummy image agar loading ke RAM/VRAM selesai)
    t0_yolo = time.time()
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    _ = model(dummy_frame, verbose=False)
    yolo_ttff = (time.time() - t0_yolo) * 1000
    stats.append(f"🧠 *YOLO TTFF (Warmup)*: {yolo_ttff:.2f} ms")

    # 4d. GPS TTFF (Tunggu maksimal 15 detik untuk mendapatkan fix pertama kali)
    import gps # Pastikan untuk mengakses variabel global gps.GPS_TTFF
    print("Menunggu GPS lock (maks 15 detik)...")
    wait_time = 0
    while gps.GPS_TTFF == 0.0 and wait_time < 15:
        await asyncio.sleep(1)
        wait_time += 1
        
    if gps.GPS_TTFF > 0:
        stats.append(f"🛰️ *GPS TTFF*: {gps.GPS_TTFF:.2f} detik")
    else:
        stats.append("🛰️ *GPS TTFF*: Timeout (Belum Fix)")

    # 5. Print & Kirim ke Telegram sebelum sistem utama berjalan
    report_text = "🚀 *SYSTEM STARTUP DIAGNOSTICS*\n\n" + "\n\n".join(stats)
    
    print("\n" + report_text.replace("*", "").replace("`", "") + "\n")
    
    try:
        await bot.send_message(chat_id=CHAT_ID, text=report_text, parse_mode="Markdown")
        print("Laporan diagnostik berhasil dikirim ke Telegram.")
    except Exception as e:
        print(f"Gagal mengirim diagnostik ke Telegram: {e}")

# ... (Fungsi-fungsi lain tetap sama: capture_loop, detection_loop, dll) ...

# Modifikasi fungsi main()
async def main():
    try:
        # Jalankan loop GPS di background
        asyncio.create_task(gps.gps_loop())
        
        # Jalankan pengecekan dan kirim pesan
        await startup_diagnostics()
        
        # Lanjut ke operasi utama secara bersamaan
        print("--- Memulai Sistem Utama ---")
        await asyncio.gather(
            capture_loop(),
            detection_loop(),
            telegram_sender_loop(),
        )
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     asyncio.run(main())
    
CONF_THRES = 0.5             # dinaikkan dari 0.5 untuk kurangi false positive
OVERLAP_RATIO = 0.15          # overlap antar kuadran, supaya objek di pinggir tidak terpotong
MIN_CONFIRM_FRAMES = 1        # objek harus muncul minimal N frame berturut-turut sebelum dikirim
MAX_AREA_RATIO = 0.25         # bbox tidak boleh lebih dari 25% luas frame (filter wajah/tangan besar)
DETECTION_IMG_PATH = "deteksi.jpg"   # file tetap, selalu ditimpa, tidak buat file baru

model = YOLO("best.pt")
dt_model = joblib.load("decision_tree.pkl")
encoder_objek = joblib.load("encoder_objek.pkl")
encoder_bahaya = joblib.load("encoder_bahaya.pkl")
bot = Bot(token=TOKEN)

LABEL_MAPPING = {
    "Deteksi Mur": "Deteksi MUR",
    "Deteksi Baut": "Deteksi BAUT",
    "KERIKIL": "KERIKIL",
    "TUMPAHAN OLI": "TUMPAHAN OLI",
}
ICONS = {"deteksi mur": "🔩", "deteksi baut": "🔩", "kerikil": "🪨", "tumpahan oli": "🛢️"}

def cari_kamera():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Coba ambil satu frame untuk memastikan benar-benar kamera
            ret, frame = cap.read()
            if ret:
                print(f"Kamera ditemukan di indeks: {i}")
                return i
            cap.release()
    return None

index = cari_kamera()
if index is not None:
    cap = cv2.VideoCapture(index)
else:
    print("Kamera tidak ditemukan di indeks manapun.")
    sys.exit()
    
# cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera gagal dibuka"); exit()

# ---- Tracking berbasis posisi (IoU) -----------------------------------
# Tujuannya: membedakan "objek yang sama, masih ada di tempat yang sama"
# vs "objek baru yang posisinya berbeda", meski labelnya sama (misal 2 kerikil
# berbeda). Hanya membandingkan label saja tidak cukup untuk kasus ini.
IOU_THRESHOLD = 0.3      # seberapa besar overlap box dianggap "objek yang sama"
MAX_MISSED = 8           # toleransi berapa siklus objek boleh "hilang sebentar"
                         # sebelum dianggap benar-benar pergi (mengatasi deteksi flicker)

# "confirmed_frames": berapa kali objek ini sudah terdeteksi berturut-turut.
# Objek hanya dikirim ke Telegram kalau confirmed_frames >= MIN_CONFIRM_FRAMES.
# Ini mencegah false positive sesaat (misal deteksi 1 frame karena noise/blur).
active_objects = []      # [{"id", "label", "bbox", "missed", "confirmed_frames", "sent"}]
next_object_id = 0

AGGREGATION_TIME = 0.5  # detik

notification_buffer = []
notification_task = None

# Queue penghubung antara detection_loop (producer) dan telegram_sender_loop (consumer)
send_queue = asyncio.Queue()

# ---- Buffer antara capture & detection (async, bukan threading) ----
latest_frame = None
frame_ready = False
latest_output = None


def compute_iou(box_a, box_b):
    """Hitung Intersection-over-Union antara 2 bounding box (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def update_tracking(detections):
    """
    Cocokkan deteksi frame ini dengan objek aktif (berdasar label + IoU).
    Return: list deteksi yang BARU DIKONFIRMASI (sudah muncul >= MIN_CONFIRM_FRAMES
    frame berturut-turut) dan belum pernah dikirim ke Telegram.
    """
    global active_objects, next_object_id

    confirmed_to_send = []
    matched_active_ids = set()

    for d in detections:
        best_match = None
        best_iou = 0.0

        for obj in active_objects:
            if obj["label"] != d["label"] or obj["id"] in matched_active_ids:
                continue
            iou = compute_iou(obj["bbox"], d["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = obj

        if best_match is not None and best_iou >= IOU_THRESHOLD:
            # objek yang SAMA -> update posisi dan tambah hitungan konfirmasi
            best_match["bbox"] = d["bbox"]
            best_match["missed"] = 0
            best_match["confirmed_frames"] += 1
            matched_active_ids.add(best_match["id"])

            # Kalau baru mencapai threshold konfirmasi DAN belum pernah dikirim -> kirim
            if best_match["confirmed_frames"] >= MIN_CONFIRM_FRAMES and not best_match["sent"]:
                best_match["sent"] = True
                confirmed_to_send.append(d)
        else:
            # objek BARU -> tambahkan ke active_objects, belum dikonfirmasi
            new_obj = {
                "id": next_object_id,
                "label": d["label"],
                "bbox": d["bbox"],
                "missed": 0,
                "confirmed_frames": 1,   # sudah 1 frame (frame ini)
                "sent": False,           # belum pernah dikirim ke Telegram
            }
            next_object_id += 1
            # active_objects.append(new_obj)
            matched_active_ids.add(new_obj["id"])
            
            # Jika batas konfirmasi adalah 1, langsung tandai terkirim detik ini juga!
            if MIN_CONFIRM_FRAMES <= 1:
                new_obj["sent"] = True
                confirmed_to_send.append(d)

            active_objects.append(new_obj)

    # objek aktif yang tidak ketemu pasangannya -> tambah missed
    for obj in active_objects:
        if obj["id"] not in matched_active_ids:
            obj["missed"] += 1

    # buang objek yang sudah lama hilang
    active_objects = [obj for obj in active_objects if obj["missed"] <= MAX_MISSED]

    return confirmed_to_send


# =========================================================
# 1. CAPTURE LOOP  -> mengisi buffer (latest_frame) terus-menerus
# =========================================================
async def capture_loop():
    print("--- Memulai capture frame dari kamera ---")
    global latest_frame, frame_ready
    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue

        latest_frame = frame
        frame_ready = True
        await asyncio.sleep(0.01)  # kasih kesempatan task lain (detection_loop) jalan


# =========================================================
# 2. SPLIT 1 FRAME JADI 4 KUADRAN (dengan overlap) + deteksi tiap kuadran
#    Koordinat box dikembalikan ke posisi frame ASLI (bukan posisi di kuadran)
# =========================================================
def split_into_4_with_overlap(frame, overlap_ratio=OVERLAP_RATIO):
    h, w = frame.shape[:2]
    half_h, half_w = h // 2, w // 2
    pad_h, pad_w = int(half_h * overlap_ratio), int(half_w * overlap_ratio)

    coords = [
        (0, 0, half_w + pad_w, half_h + pad_h),                      # kiri atas
        (half_w - pad_w, 0, w, half_h + pad_h),                      # kanan atas
        (0, half_h - pad_h, half_w + pad_w, h),                      # kiri bawah
        (half_w - pad_w, half_h - pad_h, w, h),                      # kanan bawah
    ]

    quadrants = []
    for (x1, y1, x2, y2) in coords:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        quadrants.append({"img": frame[y1:y2, x1:x2], "offset": (x1, y1)})
    return quadrants


def detect_with_tiling(frame, conf_thres=CONF_THRES):
    """Jalankan YOLO ke 4 kuadran, kembalikan semua deteksi dengan koordinat global."""
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w
    all_detections = []

    for q in split_into_4_with_overlap(frame):
        ox, oy = q["offset"]
        if q["img"].size == 0:
            continue
        results = model(q["img"], conf=conf_thres, imgsz=320, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
                bbox_area = (x2 - x1) * (y2 - y1)

                # Buang deteksi yang areanya terlalu besar relatif terhadap frame
                # (kemungkinan besar wajah/tangan yang masuk frame, bukan objek di jalan)
                if bbox_area / frame_area > MAX_AREA_RATIO:
                    continue

                all_detections.append({
                    "label": model.names[int(box.cls)],
                    "conf": float(box.conf),
                    "bbox": (x1, y1, x2, y2),
                })
    return all_detections


def draw_detections(frame, detections):
    output = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"{d['label']} {d['conf']:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output


# =========================================================
# 3. PREDIKSI BAHAYA (Decision Tree) 
# =========================================================
def prediksi_bahaya(label, conf, area):
    # Override langsung untuk MUR dan BAUT — selalu tinggi
    if label in ["Deteksi Mur", "Deteksi Baut"]:
        return "tinggi"
    
    # Override langsung untuk KERIKIL — selalu rendah
    if label == "KERIKIL":
        return "rendah"
    
    # Hanya TUMPAHAN OLI yang pakai Decision Tree
    label_ds = LABEL_MAPPING.get(label)
    if label_ds is None:
        return "rendah"
    try:
        objek_enc = encoder_objek.transform([label_ds])[0]
    except ValueError:
        return "rendah"
    fitur = pd.DataFrame([[objek_enc, conf, area]], columns=["objek", "confidence", "area_bbox"])
    pred = dt_model.predict(fitur)[0]
    return encoder_bahaya.inverse_transform([pred])[0]


# =========================================================
# 4. KIRIM KE TELEGRAM 
# =========================================================
async def send_telegram(detections, image_path):
    lokasi = await get_latest_gps()

    # KEMBALIKAN PENGECEKAN INI: Sangat penting agar Telegram tidak error
    if lokasi is None:
        lokasi = "GPS belum mendapatkan sinyal"
    else:
        print(f"\nMengirim ke tele dengan lokasi : {lokasi}")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    obj_text = "\n".join(
        f"{ICONS.get(d['label'].lower(),'⚠️')} "
        f"{d['label']} "
        f"(conf: {d['confidence']:.2f}) "
        f"🔴 *{d['bahaya'].upper()}*"
        for d in detections
    )

    caption = (
        f"🚨 *DETEKSI OBJEK DI JALAN*\n\n"
        f"🕒 {ts}\n\n"
        f"{obj_text}\n\n"
        f"📍 Lokasi\n"
        f"{lokasi}"
    )

    try:
        with open(image_path, "rb") as photo:
            await bot.send_photo(
                chat_id=CHAT_ID,
                photo=photo,
                caption=caption,
                parse_mode="Markdown"
            )
    except Exception as e:
        print(f"Error dari Telegram Bot API: {e}")

async def send_notification_after_delay():
    global notification_buffer, notification_task, latest_output

    await asyncio.sleep(AGGREGATION_TIME)

    if notification_buffer:
        # 1. BUAT NAMA FILE UNIK BERDASARKAN WAKTU
        timestamp_ms = int(time.time() * 1000)
        unik_img_path = f"deteksi_{timestamp_ms}.jpg"

        try:
            # 2. SIMPAN DENGAN NAMA FILE UNIK TERSEBUT
            await asyncio.to_thread(cv2.imwrite, unik_img_path, latest_output)
        except Exception as e:
            print(f"Gagal menyimpan gambar: {e}")

        # 3. KIRIM NAMA FILE UNIK KE ANTREAN TELEGRAM
        await send_queue.put(
            (notification_buffer.copy(), unik_img_path)
        )
        print(f"Mengirim {len(notification_buffer)} objek sekaligus")
        notification_buffer.clear()

    notification_task = None
# =========================================================
# 5. DETECTION LOOP -> ambil frame dari buffer KALAU sudah ada yang baru
#    lalu split jadi 4 kuadran, deteksi.
#    Setiap deteksi dicocokkan ke objek aktif (IoU + label).
#    Kalau cocok -> dianggap objek yang sama, posisinya diupdate, TIDAK dikirim ulang.
#    Kalau tidak cocok dengan objek aktif manapun -> dianggap OBJEK BARU, dikirim ke queue.
# =========================================================
async def detection_loop():
    global frame_ready, latest_output, notification_buffer, notification_task
    print("--- Memulai deteksi objek ---")
    
    while True:
        try:
            if not frame_ready or latest_frame is None:
                await asyncio.sleep(0.01)
                continue

            frame_ready = False
            frame = latest_frame.copy()
            
            waktu_mulai_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"[{waktu_mulai_str}] Mulai deteksi YOLO...")
            start_time = time.perf_counter()
            
            detections = await asyncio.to_thread(detect_with_tiling, frame)
            
            end_time = time.perf_counter()
            waktu_selesai_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            durasi_ms = (end_time - start_time) * 1000  # Konversi ke milidetik
            print(f"[{waktu_selesai_str}] YOLO selesai dalam {durasi_ms:.2f} ms. Ditemukan {len(detections)} objek. confident={[d['conf'] for d in detections]}")
            
            output = draw_detections(frame, detections)
            latest_output = output.copy()

            confirmed_detections = update_tracking(detections)

            for d in confirmed_detections:
                label, conf, bbox = d["label"], d["conf"], d["bbox"]
                area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                
                bahaya = prediksi_bahaya(label, conf, area)
                # if not any(obj["label"] == label for obj in notification_buffer):
                notification_buffer.append({
                    "label": label,
                    "confidence": conf,
                    "bahaya": bahaya
                })

            if notification_buffer and notification_task is None:
                notification_task = asyncio.create_task(send_notification_after_delay())

            await asyncio.sleep(0.01)

        except Exception as e:
            print(f"Error pada detection_loop: {e}")
            await asyncio.sleep(1) # Beri jeda sejenak sebelum mencoba lagi


# =========================================================
# 6. TELEGRAM SENDER LOOP -> consumer terpisah, ambil dari queue lalu kirim
#    Dipisah dari detection_loop supaya proses kirim (network, bisa lambat)
#    tidak menghambat proses deteksi frame berikutnya.
# =========================================================
async def telegram_sender_loop():
    while True:
        detections, img_path = await send_queue.get()

        try:
            await send_telegram(detections, img_path)
            print(f"Terkirim ke Telegram ({len(detections)} objek)")

        except Exception as e:
            print(f"Gagal kirim ke Telegram: {e}")

        finally:
            # 4. HAPUS FILE SETELAH DIKIRIM (berhasil ataupun gagal)
            if os.path.exists(img_path):
                try:    
                    os.remove(img_path)
                except Exception as e:
                    print(f"Gagal menghapus file {img_path}: {e}")
                    
            send_queue.task_done()


# =========================================================
# 7. MAIN -> jalankan capture_loop, detection_loop, dan telegram_sender_loop
#    secara CONCURRENT (3 task async berjalan bersamaan)
# =========================================================
# async def main():
#     try:
#         await asyncio.gather(
#             capture_loop(),
#             detection_loop(),
#             telegram_sender_loop(),
#         )
#     except KeyboardInterrupt:
#         pass
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
