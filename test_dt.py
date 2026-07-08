import os
import json
import joblib
import pandas as pd
import asyncio
import random
from telegram import Bot
from datetime import datetime, timedelta

from config import TOKEN, CHAT_ID

# =========================================================
# 1. SETUP MODEL & VARIABEL GLOBAL
# =========================================================
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

IMG_DIR = "data/img"
JSON_DIR = "data/json"

# =========================================================
# 2. FUNGSI PREDIKSI BAHAYA (DECISION TREE)
# =========================================================
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

# =========================================================
# 3. FUNGSI KIRIM TELEGRAM
# =========================================================
async def send_telegram(detections, image_path, lokasi):
    # Menggunakan timestamp saat ini
    # ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Ambil waktu saat ini, tapi ubah jamnya ke 16:00:00 (4 sore)
    waktu_awal = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

    # 2. Bikin angka acak untuk detik (0 sampai 3600 detik)
    detik_acak = random.randint(0, 3600)

    # 3. Tambahkan detik acak tersebut ke waktu_awal
    waktu_random = waktu_awal + timedelta(seconds=detik_acak)

    # 4. Format sesuai kebutuhanmu
    ts = waktu_random.strftime("%Y-%m-%d %H:%M:%S")

    print(ts)

    # Format teks objek (sama persis dengan kode asli)
    obj_text = "\n".join(
        f"{ICONS.get(d['label'].lower(),'⚠️')} "
        f"{d['label']} "
        f"(conf: {d['confidence']:.2f}) "
        f"🔴 *{d['bahaya'].upper()}*"
        for d in detections
    )

    # Format caption Telegram (sama persis dengan kode asli)
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
        print(f"  ✅ Laporan berhasil dikirim ke Telegram!")
    except Exception as e:
        print(f"  ❌ Error dari Telegram Bot API: {e}")

# =========================================================
# 4. FUNGSI UTAMA (MEMBACA DATA & TEST)
# =========================================================
async def main():
    print("--- Memulai Testing Decision Tree ---")
    
    if not os.path.exists(IMG_DIR) or not os.path.exists(JSON_DIR):
        print(f"Folder '{IMG_DIR}' atau '{JSON_DIR}' tidak ditemukan. Pastikan direktori sudah dibuat.")
        return

    for json_filename in os.listdir(JSON_DIR):
        if not json_filename.endswith(".json"):
            continue

        json_path = os.path.join(JSON_DIR, json_filename)
        print(f"\n📄 Membaca file JSON: {json_filename}")
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Format JSON rusak pada file: {json_filename}")
                continue

        imgdata_list = data.get("imgdata", [])
        
        for item in imgdata_list:
            nama_file = item.get("nama_file")
            lokasi = item.get("lokasi", "Lokasi tidak tersedia")
            detected_list = item.get("detected", [])

            img_path = os.path.join(IMG_DIR, nama_file)
            if not os.path.exists(img_path):
                print(f"⚠️ Gambar tidak ditemukan di folder img: {img_path}")
                continue

            print(f"  -> Memproses gambar : {nama_file}")
            print(f"  -> Lokasi           : {lokasi}")
            
            detections_to_send = []
            for det in detected_list:
                try:
                    parts = det.split(',')
                    label = parts[0].strip()
                    conf = float(parts[1].strip())
                    area = float(parts[2].strip())
                except Exception as e:
                    print(f"     [Error] Format array 'detected' salah pada '{det}': {e}")
                    continue

                # Prediksi bahaya dari DT
                bahaya = prediksi_bahaya(label, conf, area)
                
                # Print hasil eksekusi ke terminal
                print(f"     * Objek: {label} | Conf: {conf:.2f} | Area: {area} -> Prediksi Bahaya: {bahaya.upper()}")

                detections_to_send.append({
                    "label": label,
                    "confidence": conf,
                    "bahaya": bahaya
                })

            if detections_to_send:
                print(f"  -> Mengirim hasil {nama_file} ke Telegram...")
                await send_telegram(detections_to_send, img_path, lokasi)
            else:
                print("  -> Tidak ada objek yang dideteksi untuk dikirim.")

    print("\n--- Testing Selesai ---")

if __name__ == "__main__":
    asyncio.run(main())