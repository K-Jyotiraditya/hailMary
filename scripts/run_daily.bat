@echo off
:: HailMary Daily Pipeline — runs Mon-Fri at 18:30 IST (9:00 AM ET)
:: Logs to data/logs/daily_YYYY-MM-DD.log

set ROOT=C:\Users\yashj\Desktop\HailMary
set PYTHON=%ROOT%\venv\Scripts\python.exe
set LOGDIR=%ROOT%\data\logs

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

:: Date stamp for log file
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set LOGDATE=%DT:~0,4%-%DT:~4,2%-%DT:~6,2%
set LOGFILE=%LOGDIR%\daily_%LOGDATE%.log

echo ============================================================ >> "%LOGFILE%"
echo  HailMary Daily Run - %LOGDATE% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

:: Check Ollama is running — start it if not
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if errorlevel 1 (
    echo [BOOT] Starting Ollama... >> "%LOGFILE%"
    start /B "" ollama serve
    timeout /t 8 /nobreak >NUL
)

:: Run the pipeline
echo [RUN] Starting daily pipeline >> "%LOGFILE%"
cd /d "%ROOT%"
"%PYTHON%" orchestrator/daily_pipeline.py >> "%LOGFILE%" 2>&1
echo [DONE] Pipeline finished with exit code %errorlevel% >> "%LOGFILE%"
