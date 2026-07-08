import asyncio
import serial_asyncio
import pynmea2
import logging
import sys
import platform

# Setup logging agar Anda bisa memantau jika GPS gagal lock tanpa mengganggu bot
# logger = logging.getLogger(__name__)

async def get_gps_async(timeout=10):
    if platform.system() == "Windows":
        await asyncio.sleep(0.5)

        return "https://maps.google.com/?q=-7.954321,112.614532"
    """
    Method async untuk membaca GPS. 
    Mengembalikan URL maps jika sukses, atau None jika gagal/timeout.
    """
    port = "/dev/serial0"
    baudrate = 9600
    
    try:
        # Membuka koneksi serial secara async
        reader, writer = await serial_asyncio.open_serial_connection(url=port, baudrate=baudrate)
        
        # Membungkus pembacaan dengan timeout agar tidak memblokir event loop
        try:
            end_time = asyncio.get_event_loop().time() + timeout
            
            while asyncio.get_event_loop().time() < end_time:
                # Membaca baris data
                line_bytes = await asyncio.wait_for(reader.readline(), timeout=1.0)
                line = line_bytes.decode('ascii', errors='ignore').strip()
                
                if line.startswith(('$GNGGA', '$GPGGA')):
                    try:
                        msg = pynmea2.parse(line)
                        if msg.gps_qual > 0 and msg.latitude != 0.0:
                            # Jika sudah dapat koordinat valid, langsung return
                            return f"http://maps.google.com/?q={msg.latitude:.6f},{msg.longitude:.6f}"
                    except pynmea2.ParseError:
                        continue
        
        except asyncio.TimeoutError:
            print("GPS: Timeout tercapai, gagal mendapatkan posisi.")
        
        finally:
            # Pastikan koneksi ditutup dengan benar
            writer.close()
            await writer.wait_closed()
            
    except Exception as e:
        print(f"GPS: Error pada serial port: {e}")
        
    return None

# Contoh cara panggil di dalam bot atau event loop Anda:
# async def command_lokasi(update, context):
#     lokasi = await get_gps_async()
#     if lokasi:
#         await context.bot.send_message(chat_id=..., text=f"Lokasi saat ini: {lokasi}")
#     else:
#         await context.bot.send_message(chat_id=..., text="GPS belum fix, coba lagi sebentar lagi.")

# contoh main buat nyoba
async def main():
    print("--- Memulai pencarian lokasi GPS ---")
    print("Mohon tunggu, sedang melakukan lock satelit (maks 10 detik)...")
    
    # Memanggil fungsi async
    lokasi = await get_gps_async(timeout=10)
    
    if lokasi:
        print("\n[SUKSES] Lokasi ditemukan:")
        print(f"URL: {lokasi}")
    else:
        print("\n[GAGAL] GPS tidak mendapatkan sinyal atau timeout.")

if __name__ == '__main__':
    try:
        # Menjalankan event loop utama
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")
        sys.exit(0)