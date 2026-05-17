import sys
import numpy as np
import math
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QFileDialog, QGroupBox, QGraphicsPathItem
from PyQt5.QtCore import QTimer, Qt, QRectF
from PyQt5.QtGui import QFont, QPainterPath
import pyqtgraph as pg

class MonitorUI(QWidget):
    """
    Interfaz gráfica en tiempo real que implementa el Monitor de Signos Vitales (Opción 2 Pura).
    Utiliza PyQtGraph para visualización ultra fluida y renderizado de alta velocidad.
    """
    def __init__(self, daq):
        super().__init__()
        self.daq = daq
        self.initUI()
        
        # Iniciar la adquisición inmediatamente
        self.daq.start()
        
        # Timer para actualizar las gráficas continuamente a alta velocidad (~25 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(40)
        
    def initUI(self):
        self.setWindowTitle("Monitor de Signos Vitales - ESP32 DAQ (Opción 2)")
        self.setGeometry(50, 50, 1600, 950)
        self.setStyleSheet("background-color: #0A0A0A; color: white;")
        
        layout = QVBoxLayout(self)
        
        # ================= Controles Superiores =================
        control_layout = QHBoxLayout()
        self.btn_save = QPushButton("Guardar Historial de Monitor (CSV)")
        
        btn_style = """
            QPushButton {
                background-color: #1A1A1A; border: 2px solid #333; 
                padding: 12px 20px; border-radius: 6px; font-size: 14px; font-weight: bold;
                color: #00FF00;
            }
            QPushButton:hover { background-color: #2A2A2A; border-color: #00FF00; }
        """
        self.btn_save.setStyleSheet(btn_style)
        self.btn_save.clicked.connect(self.save_data)
        
        self.lbl_status = QLabel("Monitor Activo - Tiempo Real (ESP32 Serial)")
        self.lbl_status.setFont(QFont("Segoe UI", 12))
        self.lbl_status.setStyleSheet("color: #00FF00;")
        
        control_layout.addWidget(self.btn_save)
        control_layout.addStretch()
        control_layout.addWidget(self.lbl_status)
        layout.addLayout(control_layout)
        
        # ================= Área Principal =================
        main_layout = QHBoxLayout()
        
        # ----- Lado Izquierdo: 3 Gráficas simultáneas amplias -----
        plots_layout = QVBoxLayout()
        pg.setConfigOption('background', '#0A0A0A')
        pg.setConfigOption('foreground', '#AAAAAA')
        
        self.graphs = {}
        
        def add_plot(name, title, color):
            plot = pg.PlotWidget(title=title)
            plot.setLabel('bottom', 'Tiempo', units='s')
            plot.showGrid(x=True, y=True, alpha=0.3)
            # Desactivar auto-rango en X para control manual y fluidez
            plot.setMouseEnabled(x=False, y=False)
            # Activar escalado dinámico estricto en Y para evitar clipping
            plot.enableAutoRange(axis='y', enable=True)
            plot.setAutoVisible(x=False, y=True)
            # Aumentar tamaño de fuente del título para paneles grandes
            plot.setTitle(title, size="14pt", color="#CCCCCC")
            curve = plot.plot(pen=pg.mkPen(color, width=2.5))
            plots_layout.addWidget(plot)
            self.graphs[name] = {'plot': plot, 'curve': curve}
            
        add_plot('ecg', "Electrocardiograma (ECG - Derivación III)", '#00FF00')
        add_plot('ppg', "Fotopletismografía (PPG)", '#FFFF00')
        add_plot('posture', "Tendencia de Ángulo Postural (Grados)", '#FFAA00')
        
        main_layout.addLayout(plots_layout, stretch=3)
        
        # ----- Lado Derecho: 3 Indicadores Numéricos Grandes -----
        indicators_layout = QVBoxLayout()
        
        def create_indicator(title, color, size=64):
            group = QGroupBox(title)
            group.setStyleSheet("""
                QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #333; margin-top: 20px; border-radius: 8px; }
                QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px 0 8px; color: #AAAAAA; }
            """)
            l = QVBoxLayout()
            lbl_v = QLabel("--")
            lbl_v.setAlignment(Qt.AlignCenter)
            lbl_v.setFont(QFont("Segoe UI", size, QFont.Bold))
            lbl_v.setStyleSheet(f"color: {color}; margin: 20px;")
            l.addWidget(lbl_v)
            group.setLayout(l)
            return group, lbl_v
            
        g_bpm, self.lbl_bpm = create_indicator("Frec. Cardíaca (BPM)", "#00FF00")
        g_temp, self.lbl_temp = create_indicator("Temperatura (°C)", "#00FFFF")
        g_post, self.lbl_post = create_indicator("Estado Postural", "#FFFFFF", size=32)
        
        # Representación visual del ángulo postural
        self.angle_plot = pg.PlotWidget()
        self.angle_plot.setFixedSize(180, 180)
        self.angle_plot.setXRange(-1.5, 1.5)
        self.angle_plot.setYRange(-1.5, 1.5)
        self.angle_plot.hideAxis('bottom')
        self.angle_plot.hideAxis('left')
        self.angle_plot.setMouseEnabled(x=False, y=False)
        self.angle_plot.enableAutoRange(False)
        
        circle = QPainterPath()
        circle.addEllipse(QRectF(-1, -1, 2, 2))
        path_item = QGraphicsPathItem(circle)
        path_item.setPen(pg.mkPen('#333333', width=2))
        self.angle_plot.addItem(path_item)
        
        self.angle_line = self.angle_plot.plot([0, 0], [0, 1], pen=pg.mkPen('#00FFFF', width=6))
        g_post.layout().addWidget(self.angle_plot, alignment=Qt.AlignCenter)
        
        indicators_layout.addWidget(g_bpm)
        indicators_layout.addWidget(g_temp)
        indicators_layout.addWidget(g_post)
        indicators_layout.addStretch()
        
        main_layout.addLayout(indicators_layout, stretch=1)
        layout.addLayout(main_layout)
        
        self.update_counter = 0

    def closeEvent(self, event):
        self.daq.stop()
        event.accept()

    def save_data(self):
        import pandas as pd
        if len(self.daq.data_buffer['time']) == 0:
            QMessageBox.warning(self, "Sin datos", "Aún no hay datos para guardar.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Guardar Datos de Monitoreo", "datos_signos_vitales.csv", "CSV Files (*.csv)")
        if path:
            df = pd.DataFrame(self.daq.data_buffer)
            df.to_csv(path, index=False)
            self.lbl_status.setText(f"Datos exportados a {path}")
            QMessageBox.information(self, "Exportación Exitosa", f"Se guardaron exitosamente {len(df)} muestras.")

    def update_plots(self):
        if not self.daq.running:
            return
            
        t = np.array(self.daq.data_buffer['time'])
        if len(t) < 2:
            return
            
        # Ventana deslizante de 5 segundos
        window_size = 5.0
        mask = t > (t[-1] - window_size)
        t_plot = t[mask]
        
        if len(t_plot) > 0:
            try:
                # 1. ECG
                self.graphs['ecg']['curve'].setData(t_plot, np.array(self.daq.data_buffer['ecg'])[:len(t)][mask])
                self.graphs['ecg']['plot'].setXRange(t_plot[-1] - window_size, t_plot[-1], padding=0)
                
                # 2. PPG (IR channel)
                self.graphs['ppg']['curve'].setData(t_plot, np.array(self.daq.data_buffer['ppg'])[:len(t)][mask])
                self.graphs['ppg']['plot'].setXRange(t_plot[-1] - window_size, t_plot[-1], padding=0)
                
                # 3. Posture Angle
                angle = np.array(self.daq.data_buffer['posture'])[:len(t)][mask]
                self.graphs['posture']['curve'].setData(t_plot, angle)
                self.graphs['posture']['plot'].setXRange(t_plot[-1] - window_size, t_plot[-1], padding=0)
            except (IndexError, ValueError, KeyError):
                pass # Ignorar frame si el buffer fue modificado en otro hilo o faltan llaves

        # Actualizar Indicadores (cada ~400ms para evitar parpadeo)
        self.update_counter += 1
        if self.update_counter >= 10:
            self.update_counter = 0
            
            self.lbl_bpm.setText(f"{self.daq.bpm}" if self.daq.bpm > 0 else "--")
            self.lbl_temp.setText(f"{self.daq.temperature:.1f} °C")
            
            posture = self.daq.posture_state
            self.lbl_post.setText(f"{posture}\n({self.daq.posture_angle}°)")
            if posture == "Caída":
                self.lbl_post.setStyleSheet("color: #FF2222; margin: 10px;")
            elif posture == "Inclinado":
                self.lbl_post.setStyleSheet("color: #FFAA00; margin: 10px;")
            else:
                self.lbl_post.setStyleSheet("color: #00FF00; margin: 10px;")
                
            # Actualizar visualización gráfica del ángulo
            rad = math.radians(self.daq.posture_angle)
            x = math.sin(rad)
            y = math.cos(rad)
            self.angle_line.setData([0, x], [0, y])
