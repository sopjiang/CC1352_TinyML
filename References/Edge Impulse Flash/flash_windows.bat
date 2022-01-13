@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION
setlocal
REM go to the folder where this bat script is located
cd /d d:\Programs\TI\Uniflash_7.0.0

dslite.bat -c user_files\configs\cc1352p1f3.ccxml -l user_files\settings\generated.ufsettings -e -f -v edge_impulse_firmware.out

@pause
exit /b 0
