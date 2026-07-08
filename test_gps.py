import asyncio
import serial_asyncio
import pynmea2
import csv
import sys
import time
from datetime import datetime

# Konfigurasi Pengujian
PORT = "/dev/serial0"  # Sesuaikan dengan port raspi kamu
BAUDRATE = 9600
DURASI_TEST_MENIT = 15 # Lama pengujian (15-30 menit sangat disarankan untuk skripsi)

async def log_gps_data():
    filename = f"uji_gps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Buka/buat file CSV dan tulis Header
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header kolom untuk keperluan Bab 4
        writer.writerow(["Timestamp", "Latitude", "Longitude", "Num_Satellites", "HDOP", "Fix_Quality"])
        
        print(f"--- Memulai Pengujian GPS ---")
        print(f"Data akan disimpan di: {filename}")
        print(f"Menunggu lock satelit...")

        try:
            reader, writer_serial = await serial_asyncio.open_serial_connection(url=PORT, baudrate=BAUDRATE)
            
            waktu_mulai = time.time()
            durasi_detik = DURASI_TEST_MENIT * 60
            
            while (time.time() - waktu_mulai) < durasi_detik:
                try:
                    line_bytes = await asyncio.wait_for(reader.readline(), timeout=2.0)
                    line = line_bytes.decode('ascii', errors='ignore').strip()
                    
                    # Ambil data dari kalimat GNGGA atau GPGGA
                    if line.startswith(('$GNGGA', '$GPGGA')):
                        msg = pynmea2.parse(line)
                        
                        # Pastikan GPS sudah mendapat Fix (Valid)
                        if msg.gps_qual > 0 and msg.latitude != 0.0:
                            waktu_sekarang = datetime.now().strftime('%H:%M:%S')
                            lat = round(msg.latitude, 7)
                            lon = round(msg.longitude, 7)
                            sats = msg.num_sats
                            hdop = msg.horizontal_dil
                            qual = msg.gps_qual
                            
                            # Tulis ke CSV
                            writer.writerow([waktu_sekarang, lat, lon, sats, hdop, qual])
                            
                            # Tampilkan di layar untuk monitoring OS Headless
                            sys.stdout.write(f"\r[{waktu_sekarang}] Lat: {lat}, Lon: {lon} | Sats: {sats} | HDOP: {hdop} ")
                            sys.stdout.flush()
                            
                            # Beri jeda 1 detik tiap sampel (opsional, agar data tidak terlalu besar)
                            await asyncio.sleep(1)
                            
                except asyncio.TimeoutError:
                    continue
                except pynmea2.ParseError:
                    continue
                    
        except Exception as e:
            print(f"\n[Error] Gagal mengakses port serial: {e}")
            return
        finally:
            print(f"\n\nPengujian selesai. Data berhasil disimpan ke {filename}")

if __name__ == '__main__':
    # PENTING: Pastikan bot telegram utama sedang dimatikan saat menjalankan ini
    # agar port serial tidak "rebutan" (Device or resource busy)
    try:
        asyncio.run(log_gps_data())
    except KeyboardInterrupt:
        print("\nPengujian dihentikan manual oleh pengguna.")
        sys.exit(0)