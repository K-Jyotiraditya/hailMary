@echo off
:: HailMary Weekly Retrain — runs Sunday 02:00 IST
:: Retrains XGBoost on 10 years of data + refreshes universe cache
:: Logs to data/logs/retrain_YYYY-MM-DD.log

set ROOT=C:\Users\yashj\Desktop\HailMary
set PYTHON=%ROOT%\venv\Scripts\python.exe
set LOGDIR=%ROOT%\data\logs

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set LOGDATE=%DT:~0,4%-%DT:~4,2%-%DT:~6,2%
set LOGFILE=%LOGDIR%\retrain_%LOGDATE%.log

echo ============================================================ >> "%LOGFILE%"
echo  HailMary Weekly Retrain - %LOGDATE% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

cd /d "%ROOT%"

:: Refresh universe screen (force — weekly cache bust)
echo [1/3] Refreshing universe screen... >> "%LOGFILE%"
"%PYTHON%" -c "from config.universe import screen_universe; screen_universe(force_refresh=True)" >> "%LOGFILE%" 2>&1

:: Refresh macro regime cache
echo [2/3] Refreshing macro regime... >> "%LOGFILE%"
"%PYTHON%" -c "from data.macro_regime import get_macro_regime; r=get_macro_regime(force_refresh=True); print('Regime:', r['regime'], '| VIX:', r['vix'])" >> "%LOGFILE%" 2>&1

:: Walk-forward CV + retrain (only saves if mean IC > 0)
echo [3/3] Running walk-forward CV and retraining XGBoost... >> "%LOGFILE%"
"%PYTHON%" models/walk_forward.py --period 10y --train-final >> "%LOGFILE%" 2>&1

echo [DONE] Weekly retrain complete >> "%LOGFILE%"
