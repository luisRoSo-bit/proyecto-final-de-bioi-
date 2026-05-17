import sys
from PyQt5.QtWidgets import QApplication
from data_acquisition import DataAcquisition
from ui_monitor import MonitorUI

def main():
    """
    Punto de entrada principal.
    Inicializa el módulo de adquisición y levanta la interfaz gráfica.
    """
    # ¡Sincronizado a 400Hz con el ESP32 para activar el Filtro Notch!
    daq = DataAcquisition(fs=400) 
    
    app = QApplication(sys.argv)
    window = MonitorUI(daq)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()