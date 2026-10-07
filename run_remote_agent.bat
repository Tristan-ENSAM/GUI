@echo off
rem Remote-execution agent, to start on the COMPUTE PC in its Remote Desktop
rem session. It runs the Abaqus jobs that the GUI of another PC puts in the
rem queue folder on the shared drive. Close the Remote Desktop window with the
rem cross (disconnect) to keep it running; "Sign out" would stop it.
rem
rem Queue folder: first argument, else the one set in the GUI Preferences of
rem this PC (Preferences > Execution > Queue folder).
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -u -m gui.core.remote_exec agent %*
pause
