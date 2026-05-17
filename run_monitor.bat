@echo off
echo ===================================================
echo Iniciando Monitor de Signos Vitales (ESP32)...
echo ===================================================
echo Instalando dependencias necesarias (por si acaso)...
pip install pyserial pyqt5 pyqtgraph scipy numpy pandas
echo.
echo Ejecutando la aplicacion...
python main.py
echo.
pause
