from ultralytics import YOLO
import cv2, joblib, asyncio
from telegram import Bot
from datetime import datetime
import pandas as pd
from config import TOKEN, CHAT_ID
import time
import os
import subprocess
import numpy as np
import sys
import threading
from flask import Flask, Response

# Pastikan module gps Anda ada di folder yang sama
import gps  

# =========================================================
# --- DITAMBAHKAN UNTUK FLASK WEB STREAMER ---
# =========================================================
app = Flask(__name__)

def generate_frames():
    global latest_output, latest_frame
    while True:
        # Gunakan frame hasil deteksi jika ada, jika belum gunakan frame mentah dari kamera
        frame_to_stream = latest_output if latest_output is not None else latest_frame
        
        if frame_to_stream is None:
            time.sleep(0.1)
            continue
        
        # Encode gambar ke JPEG
        ret, buffer = cv2.imencode('.jpg', frame_to_stream)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.05) # Jeda tipis agar CPU tidak 100%

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) # Sembunyikan log Flask yang spam di terminal
    app.run(host='0.0.0.0', port=5000, threaded=True)
# =========================================================


async def startup_diagnostics():
    print("--- Menjalankan Diagnostik Startup ---")
    stats = []
    
    # 3. Baca systemd-analyze (Cold Start Raspi)
    try:
        boot_time = subprocess.check_output(['systemd-analyze']).decode('utf-8').strip()
        stats.append(f"⏱️ *Cold Start (systemd)*:\n`{boot_time}`")
    except Exception as e:
        stats.append("⏱️ *Cold Start (systemd)*: Gagal dibaca")

    # 4a. Telegram Ping
    t0_ping = time.time()
    try:
        await bot.get_me()
        ping_ms = (time.time() - t0_ping) * 1000
        stats.append(f"🌐 *Telegram Ping*: {ping_ms:.2f} ms")
    except Exception as e:
        stats.append(f"🌐 *Telegram Ping*: Gagal ({e})")

    # 4b. Kamera TTFF
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

    # 4c. YOLO TTFF (Warmup)
    t0_yolo = time.time()
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    _ = model(dummy_frame, verbose=False)
    yolo_ttff = (time.time() - t0_yolo) * 1000
    stats.append(f"🧠 *YOLO TTFF (Warmup)*: {yolo_ttff:.2f} ms")

    # 4d. GPS TTFF
    print("Menunggu GPS lock (maks 15 detik)...")
    wait_time = 0
    while gps.GPS_TTFF == 0.0 and wait_time < 15:
        await asyncio.sleep(1)
        wait_time += 1
        
    if gps.GPS_TTFF > 0:
        stats.append(f"🛰️ *GPS TTFF*: {gps.GPS_TTFF:.2f} detik")
    else:
        stats.append("🛰️ *GPS TTFF*: Timeout (Belum Fix)")

    # 5. Print & Kirim
    report_text = "🚀 *SYSTEM STARTUP DIAGNOSTICS*\n\n" + "\n\n".join(stats)
    print("\n" + report_text.replace("*", "").replace("`", "") + "\n")
    
    try:
        await bot.send_message(chat_id=CHAT_ID, text=report_text, parse_mode="Markdown")
        print("Laporan diagnostik dikirim ke Telegram.")
    except Exception as e:
        print(f"Gagal mengirim diagnostik: {e}")

CONF_THRES = 0.5             
OVERLAP_RATIO = 0.15          
MIN_CONFIRM_FRAMES = 1        
MAX_AREA_RATIO = 0.25         
DETECTION_IMG_PATH = "deteksi.jpg"   

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
    print("Kamera tidak ditemukan.")
    sys.exit()
    
if not cap.isOpened():
    print("Kamera gagal dibuka"); exit()

IOU_THRESHOLD = 0.3      
MAX_MISSED = 8           
active_objects = []      
next_object_id = 0
AGGREGATION_TIME = 0.5  

notification_buffer = []
notification_task = None
send_queue = asyncio.Queue()

latest_frame = None
frame_ready = False
latest_output = None

def compute_iou(box_a, box_b):
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
            best_match["bbox"] = d["bbox"]
            best_match["missed"] = 0
            best_match["confirmed_frames"] += 1
            matched_active_ids.add(best_match["id"])

            if best_match["confirmed_frames"] >= MIN_CONFIRM_FRAMES and not best_match["sent"]:
                best_match["sent"] = True
                confirmed_to_send.append(d)
        else:
            new_obj = {
                "id": next_object_id,
                "label": d["label"],
                "bbox": d["bbox"],
                "missed": 0,
                "confirmed_frames": 1,   
                "sent": False,           
            }
            next_object_id += 1
            matched_active_ids.add(new_obj["id"])
            
            if MIN_CONFIRM_FRAMES <= 1:
                new_obj["sent"] = True
                confirmed_to_send.append(d)
            active_objects.append(new_obj)

    for obj in active_objects:
        if obj["id"] not in matched_active_ids:
            obj["missed"] += 1

    active_objects = [obj for obj in active_objects if obj["missed"] <= MAX_MISSED]
    return confirmed_to_send

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
        await asyncio.sleep(0.01) 

def split_into_4_with_overlap(frame, overlap_ratio=OVERLAP_RATIO):
    h, w = frame.shape[:2]
    half_h, half_w = h // 2, w // 2
    pad_h, pad_w = int(half_h * overlap_ratio), int(half_w * overlap_ratio)

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
        quadrants.append({"img": frame[y1:y2, x1:x2], "offset": (x1, y1)})
    return quadrants

def detect_with_tiling(frame, conf_thres=CONF_THRES):
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

                if bbox_area / frame_area > MAX_AREA_RATIO:
                    continue

                all_detections.append({
                    "label": model.names[int(box.cls)],
                    "conf": float(box.conf),
                    "bbox": (x1, y1, x2, y2),
                })
    return all_detections

# =========================================================
# --- DIMODIFIKASI: Menampilkan teks dari Decision Tree ---
# =========================================================
def draw_detections(frame, detections):
    output = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Ambil status bahaya (default kosong jika belum diprediksi)
        status_bahaya = d.get('bahaya', '').upper()
        
        # Warna teks: Merah jika TINGGI, Hijau muda jika RENDAH/SEDANG
        if status_bahaya == 'TINGGI':
            warna_teks = (0, 0, 255) # BGR Merah
        else:
            warna_teks = (0, 255, 0) # BGR Hijau
            
        teks_display = f"{d['label']} {d['conf']:.2f} | {status_bahaya}"
        
        cv2.putText(output, teks_display, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna_teks, 2)
    return output

def prediksi_bahaya(label, conf, area):
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

async def send_telegram(detections, image_path):
    # Asumsikan gps.get_gps_async sudah di-import atau disesuaikan
    lokasi = await gps.get_gps_async(timeout=10) 

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
        timestamp_ms = int(time.time() * 1000)
        unik_img_path = f"deteksi_{timestamp_ms}.jpg"

        try:
            await asyncio.to_thread(cv2.imwrite, unik_img_path, latest_output)
        except Exception as e:
            print(f"Gagal menyimpan gambar: {e}")

        await send_queue.put(
            (notification_buffer.copy(), unik_img_path)
        )
        print(f"Mengirim {len(notification_buffer)} objek sekaligus")
        notification_buffer.clear()

    notification_task = None

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
            
            start_time = time.perf_counter()
            detections = await asyncio.to_thread(detect_with_tiling, frame)
            
            # =========================================================
            # --- DIMODIFIKASI: Pindahkan proses Decision Tree ke sini ---
            # =========================================================
            for d in detections:
                bbox = d["bbox"]
                area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                d["bahaya"] = prediksi_bahaya(d["label"], d["conf"], area)
            
            end_time = time.perf_counter()
            durasi_ms = (end_time - start_time) * 1000  
            print(f"YOLO + DT selesai dalam {durasi_ms:.2f} ms. Ditemukan {len(detections)} objek.")
            
            # Sekarang gambar akan memiliki teks Bahaya Tinggi/Rendah
            output = draw_detections(frame, detections)
            latest_output = output.copy()

            confirmed_detections = update_tracking(detections)

            for d in confirmed_detections:
                # Ambil "bahaya" langsung dari dictionary d, tidak perlu prediksi ulang
                if not any(obj["label"] == d["label"] for obj in notification_buffer):
                    notification_buffer.append({
                        "label": d["label"],
                        "confidence": d["conf"],
                        "bahaya": d["bahaya"] 
                    })

            if notification_buffer and notification_task is None:
                notification_task = asyncio.create_task(send_notification_after_delay())

            await asyncio.sleep(0.01)

        except Exception as e:
            print(f"Error pada detection_loop: {e}")
            await asyncio.sleep(1) 

async def telegram_sender_loop():
    while True:
        detections, img_path = await send_queue.get()
        try:
            await send_telegram(detections, img_path)
            print(f"Terkirim ke Telegram ({len(detections)} objek)")
        except Exception as e:
            print(f"Gagal kirim ke Telegram: {e}")
        finally:
            if os.path.exists(img_path):
                try:    
                    os.remove(img_path)
                except Exception as e:
                    print(f"Gagal menghapus file {img_path}: {e}")
            send_queue.task_done()

# =========================================================
# --- DIMODIFIKASI: Menjalankan Thread Flask sebelum Loop ---
# =========================================================
async def main():
    try:
        # Jalankan Flask Web Streamer di background
        flask_thread = threading.Thread(target=start_flask, daemon=True)
        flask_thread.start()
        print("🌐 Web Streamer Aktif! Buka http://IP_RASPI:5000 di browser Anda.")

        # Jalankan loop GPS
        asyncio.create_task(gps.gps_loop())
        
        # Diagnostik
        await startup_diagnostics()
        
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

if __name__ == "__main__":
    asyncio.run(main())