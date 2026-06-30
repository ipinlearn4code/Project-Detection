from ultralytics import YOLO
import cv2, joblib, asyncio
from telegram import Bot
from datetime import datetime
import pandas as pd
from config import TOKEN, CHAT_ID

COOLDOWN = 30
WAIT_BEFORE_SEND = 2  # detik, tunggu untuk cari confidence terbaik sebelum kirim

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

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Kamera gagal dibuka"); exit()

last_sent = {}
pending = {}  # {label: {"data": {...}, "first_seen": ts, "best_frame": img}}
frame_count = 0


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
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    obj_text = "\n".join(
        f"{ICONS.get(d['label'].lower(),'⚠️')} {d['label']} (conf: {d['confidence']:.2f}) 🔴 *{d['bahaya'].upper()}*"
        for d in detections
    )
    caption = f"🚨 *DETEKSI OBJEK DI JALAN!*\n\n🕐 {ts}\n\n{obj_text}"

    with open(image_path, 'rb') as photo:
        await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=caption, parse_mode='Markdown')


async def main():
    global frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (320, 320))
        results = model(frame, conf=0.5, imgsz=320)

        output = frame.copy()
        now = datetime.now().timestamp()

        for result in results:
            output = result.plot(img=output)
            for box in result.boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0]
                area = float((x2 - x1) * (y2 - y1))
                bahaya = prediksi_bahaya(label, conf, area)

                # Skip kalau masih dalam cooldown dari pengiriman sebelumnya
                if now - last_sent.get(label, 0) <= COOLDOWN:
                    continue

                # Simpan/update deteksi terbaik (confidence tertinggi) untuk label ini
                if label not in pending:
                    pending[label] = {
                        "data": {"label": label, "confidence": conf, "bahaya": bahaya},
                        "first_seen": now,
                        "best_frame": output.copy()
                    }
                elif conf > pending[label]["data"]["confidence"]:
                    pending[label]["data"] = {"label": label, "confidence": conf, "bahaya": bahaya}
                    pending[label]["best_frame"] = output.copy()

        cv2.imshow("YOLO + Decision Tree", output)

        # Kirim yang sudah cukup lama ditunggu (ambil versi terbaiknya)
        to_remove = []
        for label, info in pending.items():
            if now - info["first_seen"] >= WAIT_BEFORE_SEND:
                img_path = "deteksi.jpg"
                cv2.imwrite(img_path, info["best_frame"])
                await send_telegram([info["data"]], img_path)
                print(f"Terkirim: {label} (conf={info['data']['confidence']:.2f})")
                last_sent[label] = now
                to_remove.append(label)

        for label in to_remove:
            del pending[label]

        if cv2.waitKey(1) & 0xFF == 27:
            break

        await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()


asyncio.run(main())