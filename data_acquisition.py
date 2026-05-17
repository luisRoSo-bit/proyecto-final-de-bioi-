import numpy as np
import time
import threading
import serial
import serial.tools.list_ports
from scipy.signal import iirnotch, lfilter, find_peaks

class DataAcquisition:
    """
    Adquisicion de datos desde ESP32 por USB Serial.
    Trama esperada: ECG,TEMP,PPG,ANGULO
    """
    def __init__(self, fs=400):
        self.fs = fs
        self.running = False

        # Filtro Notch 60Hz
        self.f0 = 60.0
        self.Q = 30.0
        if self.fs > 120:
            self.b_notch, self.a_notch = iirnotch(self.f0, self.Q, self.fs)
            self.zi_ecg = np.zeros(max(len(self.a_notch), len(self.b_notch)) - 1)
            self.zi_ppg = np.zeros(max(len(self.a_notch), len(self.b_notch)) - 1)
            self.apply_notch = True
        else:
            self.apply_notch = False
            self.zi_ecg = None
            self.zi_ppg = None

        self.data_buffer = {
            'time': [], 'ecg': [], 'temp': [], 'ppg': [], 'posture': [], 'bpm': []
        }

        self.t0 = 0
        self.temp_baseline = None
        self.bpm = 0
        self.temperature = 0.0
        self.posture_state = "Calculando..."
        self.posture_angle = 0
        self.serial_port = None
        self._sample_count = 0
        self._debug_prints = 0  # Contador para imprimir las primeras tramas

    def _find_esp32_port(self):
        """Busca el puerto y muestra en consola lo que encuentra."""
        print("[INFO] Buscando puertos USB activos...")
        ports = list(serial.tools.list_ports.comports())
        
        if not ports:
            print("[ERROR] No se detecto NINGUN dispositivo USB conectado al PC.")
            return None

        for p in ports:
            desc = p.description.lower()
            print(f"   -> Encontrado: {p.device} | {p.description}")
            if "cp210" in desc or "ch340" in desc or "usb-serial" in desc or "uart" in desc:
                print(f"[OK] ESP32 detectado en {p.device}")
                return p.device
                
        print("[WARN] No se reconocio el nombre del chip. Si sabes el puerto, ponlo manual.")
        return None

    def start(self):
        self.running = True
        self.t0 = time.time()
        self._sample_count = 0
        self._debug_prints = 0

        for k in self.data_buffer:
            self.data_buffer[k] = []

        # 1. Busqueda de puerto
        port_name = self._find_esp32_port()
        
        # --- ZONA DE CONTROL MANUAL ---
        # Si el automatico falla, BORRA el '#' de la linea de abajo y pon tu puerto real (ej: "COM3")
        # port_name = "COM3"

        if not port_name:
            print("[ERROR] Imposible iniciar sin un puerto COM valido.")
            self.running = False
            return

        try:
            print(f"[INFO] Intentando abrir {port_name} a 115200 baudios...")
            self.serial_port = serial.Serial(port_name, 115200, timeout=1)
            
            # Control de energia para evitar que el ESP32 se quede trabado reiniciandose
            self.serial_port.dtr = False
            self.serial_port.rts = False
            
            time.sleep(2) # Esperar a que el hardware estabilice
            self.serial_port.reset_input_buffer()
            print("[OK] Puerto ABIERTO. Escuchando datos...")
            
        except serial.SerialException as e:
            print(f"[ERROR] ACCESO DENEGADO al {port_name}.")
            print("   -> Asegurate de tener el Monitor Serie de Arduino cerrado.")
            self.running = False
            return

        threading.Thread(target=self._update_loop, daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("[INFO] Puerto serial cerrado de forma segura.")

    def _update_loop(self):
        while self.running:
            try:
                if self.serial_port.in_waiting == 0:
                    continue # No hay datos, seguir esperando sin bloquear

                raw_line = self.serial_port.readline()
                if not raw_line: continue
                
                line = raw_line.decode('utf-8', errors='ignore').strip()
                if not line or line.startswith("="): continue

                # Mostrar las primeras 5 tramas para confirmar que llegan bien
                if self._debug_prints < 5:
                    print(f"[DEBUG] DATO RECIBIDO: {line}")
                    self._debug_prints += 1

                parts = line.split(',')
                if len(parts) != 4: continue

                ecg_raw  = float(parts[0])
                temp_raw = float(parts[1])
                ppg_raw  = float(parts[2])
                angulo   = float(parts[3])

                ecg_raw = max(0.0, min(4095.0, ecg_raw))
                temp_raw = max(0.0, min(4095.0, temp_raw))

                ecg_volts = (ecg_raw * 3.3 / 4095.0) - 1.65
                temp_celsius = ((temp_raw * 3.3 / 4095.0) * 1000.0) / 10.0

                t = time.time() - self.t0
                self._append_data(t, ecg_volts, temp_celsius, ppg_raw, angulo)
                self._sample_count += 1

            except Exception as e:
                continue

    def _append_data(self, t, ecg, temp, ppg, angulo):
        if self.apply_notch:
            ecg_f, self.zi_ecg = lfilter(self.b_notch, self.a_notch, [ecg], zi=self.zi_ecg)
            ppg_f, self.zi_ppg = lfilter(self.b_notch, self.a_notch, [ppg], zi=self.zi_ppg)
        else:
            ecg_f = [ecg]
            ppg_f = [ppg]

        self.temperature = temp

        if self.temp_baseline is None:
            self.temp_baseline = temp
            temp_csv = temp
        else:
            if abs(temp - self.temp_baseline) >= 1.0:
                self.temp_baseline = temp
                temp_csv = temp
            else:
                temp_csv = np.nan 

        self.data_buffer['time'].append(t)
        self.data_buffer['ecg'].append(ecg_f[0])
        self.data_buffer['temp'].append(temp_csv)
        self.data_buffer['ppg'].append(ppg_f[0])
        self.data_buffer['posture'].append(angulo)
        self.data_buffer['bpm'].append(self.bpm)

        max_samples = self.fs * 30
        if len(self.data_buffer['time']) > max_samples:
            for k in self.data_buffer:
                self.data_buffer[k].pop(0)

    def _process_loop(self):
        while self.running:
            if len(self.data_buffer['ppg']) > self.fs * 4:
                ppg_window = np.array(self.data_buffer['ppg'][-self.fs*4:])
                ppg_norm = ppg_window - np.mean(ppg_window)
                
                max_peak = np.max(ppg_norm)
                threshold = max_peak * 0.4
                if threshold < 50: threshold = 50
                
                peaks, _ = find_peaks(ppg_norm, distance=self.fs*0.33, height=threshold)
                
                if len(peaks) >= 2:
                    avg_rr = np.mean(np.diff(peaks)) / self.fs
                    self.bpm = int(60.0 / avg_rr)
                
                self.posture_angle = self.data_buffer['posture'][-1]
                angle_abs = abs(self.posture_angle)
                
                if angle_abs < 35: self.posture_state = "De pie"
                elif angle_abs < 65: self.posture_state = "Inclinado"
                else: self.posture_state = "Caida"

            time.sleep(0.5)