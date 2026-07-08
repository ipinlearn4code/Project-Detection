import asyncio
import serial_asyncio
import pynmea2
import platform
import time

# Variabel global untuk menyimpan state GPS terakhir dan TTFF
LATEST_LOCATION = None
GPS_TTFF = 0.0

async def gps_loop():
    """
    Berjalan terus-menerus di background untuk memperbarui lokasi.
    (Permintaan No. 2: jalan terus mencari lokasi)
    """
    global LATEST_LOCATION, GPS_TTFF
    start_time = time.time()
    
    if platform.system() == "Windows":
        await asyncio.sleep(2) # Simulasi delay lock satelit
        GPS_TTFF = time.time() - start_time
        # (Permintaan No. 1: Perbaikan format link Google Maps standar)
        LATEST_LOCATION = "https://www.google.com/maps?q=-6.200000,106.816666"
        while True:
            await asyncio.sleep(1) # Standby

    port = "/dev/serial0"
    baudrate = 9600
    
    try:
        reader, writer = await serial_asyncio.open_serial_connection(url=port, baudrate=baudrate)
        while True:
            try:
                line_bytes = await asyncio.wait_for(reader.readline(), timeout=2.0)
                line = line_bytes.decode('ascii', errors='ignore').strip()
                
                if line.startswith(('$GNGGA', '$GPGGA')):
                    try:
                        msg = pynmea2.parse(line)
                        if msg.gps_qual > 0 and msg.latitude != 0.0:
                            # 1. Update ke format standar Google Maps
                            LATEST_LOCATION = f"https://www.google.com/maps?q={msg.latitude:.6f},{msg.longitude:.6f}"
                            
                            # Catat TTFF saat pertama kali mendapat sinyal (fix)
                            if GPS_TTFF == 0.0:
                                GPS_TTFF = time.time() - start_time
                                
                    except pynmea2.ParseError:
                        continue
                        
            except asyncio.TimeoutError:
                # Abaikan timeout, biarkan loop jalan terus
                continue
                
    except Exception as e:
        print(f"GPS Loop Error: {e}")

async def get_latest_gps():
    """Mengembalikan lokasi terakhir tanpa memblokir."""
    return LATEST_LOCATION