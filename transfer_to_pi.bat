@echo off
echo Transferring troubleshooting files to Raspberry Pi
echo.

echo 1. Transferring troubleshoot.sh
scp "%~dp0troubleshoot.sh" sahan@192.168.8.137:~/

echo 2. Transferring simple_monitor.py
scp "%~dp0simple_monitor.py" sahan@192.168.8.137:~/jaundice_monitor/

echo.
echo Files transferred. Now connect to your Raspberry Pi and run:
echo.
echo    ssh sahan@192.168.8.137
echo    chmod +x troubleshoot.sh
echo    ./troubleshoot.sh
echo.
echo Then try the simplified monitor:
echo.
echo    python3 ~/jaundice_monitor/simple_monitor.py --camera 0
echo.
pause
