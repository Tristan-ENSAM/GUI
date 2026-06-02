@echo off
REM ============================================================================
REM  Abaqus Cutting Pre-processor — DEBUG launcher
REM ----------------------------------------------------------------------------
REM  Same as run_gui.bat but keeps the console open and uses python.exe (not
REM  pythonw.exe), so any Python exception or traceback stays visible.
REM ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_MARKER=%VENV_DIR%\.requirements_installed"

set "HOST_PY="

where py >nul 2>&1
if !errorlevel! == 0 (
    py -3 --version >nul 2>&1
    if !errorlevel! == 0 set "HOST_PY=py -3"
)

if not defined HOST_PY (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined HOST_PY (
            set "_CAND=%%P"
            echo !_CAND! | findstr /I "\\WindowsApps\\" >nul
            if !errorlevel! neq 0 set "HOST_PY=!_CAND!"
        )
    )
)

if not defined HOST_PY (
    echo [ERROR] No usable Python 3 found. Install from python.org and retry.
    pause
    exit /b 1
)

echo [INFO] Host Python: !HOST_PY!

if not exist "%VENV_PY%" (
    echo [INFO] Creating venv...
    !HOST_PY! -m venv "%VENV_DIR%"
)

if not exist "%REQ_MARKER%" (
    echo [INFO] Installing dependencies...
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install PySide6 matplotlib numpy
    if !errorlevel! neq 0 (
        echo [ERROR] Install failed.
        pause
        exit /b 1
    )
    echo installed > "%REQ_MARKER%"
)

if not exist "%~dp0gui\main.py" (
    echo [ERROR] Cannot find `gui\main.py` next to this .bat.
    echo Move the .py files into a `gui\` subfolder respecting the layout
    echo described in README.md.
    pause
    exit /b 1
)

echo [INFO] Launching GUI (console kept open for debug)...
"%VENV_PY%" -m gui.main
echo.
echo [INFO] GUI exited with code %errorlevel%
pause

endlocal
