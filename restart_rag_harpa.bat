@echo off
echo Stopping existing Flask process...

for /f "tokens=2 delims= " %%a in ('tasklist ^| findstr python') do taskkill /F /PID %%a

timeout /t 2 /nobreak >nul

echo Starting Flask application...
start /B run_rag_harpa.bat

echo Flask restarted successfully.
exit
