@echo off
REM ============================================================================
REM  Abaqus Cutting Pre-processor - DEBUG launcher
REM  Uses python.exe (console stays open) so any traceback is visible. If the
REM  venv already has the core dependencies it just launches - no host Python,
REM  no network needed. Optional (experimental) deps are installed best-effort.
REM ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_MARKER=%VENV_DIR%\.requirements_installed"

REM --- Fast path: venv with the CORE deps already present -> just launch ------
REM Optional deps (Pillow, OpenCV) are NOT gated here so the GUI always starts.
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import PySide6, matplotlib, numpy" >nul 2>&1
    if !errorlevel! == 0 goto :optional
    "%VENV_PY%" -c "import sys" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [WARN] Existing .venv is not runnable here ^(copied from another
        echo        PC / Python version^). Rebuilding it...
        rmdir /s /q "%VENV_DIR%"
    )
)

REM --- Need a host Python only to CREATE a missing venv ----------------------
if not exist "%VENV_PY%" (
    set "HOST_PY="
    where py >nul 2>&1
    if !errorlevel! == 0 (
        py -3 --version >nul 2>&1
        if !errorlevel! == 0 set "HOST_PY=py -3"
    )
    if not defined HOST_PY (
        for /f "delims=" %%P in ('where python 2^>nul') do (
            if not defined HOST_PY (
                echo %%P | findstr /I "\WindowsApps\" >nul
                if !errorlevel! neq 0 set "HOST_PY=%%P"
            )
        )
    )
    if not defined HOST_PY (
        echo [ERROR] No .venv yet, and no host Python 3 found to create it.
        echo         Create it once, e.g.:
        echo           "C:\ProgramData\Anaconda3\python.exe" -m venv "%VENV_DIR%"
        pause
        exit /b 1
    )
    echo [INFO] Creating venv with: !HOST_PY!
    !HOST_PY! -m venv "%VENV_DIR%"
)

REM --- Full install (fresh venv): core + optional deps -----------------------
echo [INFO] Installing dependencies...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install PySide6 matplotlib numpy Pillow opencv-python
if !errorlevel! neq 0 (
    echo [ERROR] Install failed. If it is an SSL/proxy error, your machine may
    echo         block PyPI; use a corporate proxy, e.g.:
    echo           "%VENV_PY%" -m pip install --proxy http://USER:PASS@HOST:PORT PySide6 matplotlib numpy Pillow opencv-python
    pause
    exit /b 1
)
echo installed > "%REQ_MARKER%"
goto :launch

REM --- Optional deps: install only if missing, never block the launch --------
:optional
"%VENV_PY%" -c "import PIL, cv2" >nul 2>&1
if !errorlevel! neq 0 (
    echo [INFO] Installing optional experimental deps ^(Pillow, OpenCV^)...
    "%VENV_PY%" -m pip install Pillow opencv-python
    if !errorlevel! neq 0 (
        echo [WARN] Could not install Pillow/OpenCV. The Experimental Data
        echo        image and calibration features may be limited. Continuing.
    )
)
goto :launch

:launch
if not exist "%~dp0gui\main.py" (
    echo [ERROR] Cannot find gui\main.py next to this .bat.
    pause
    exit /b 1
)
echo [INFO] Launching GUI (console kept open for debug)...
"%VENV_PY%" -m gui.main
echo.
echo [INFO] GUI exited with code %errorlevel%
pause
endlocal
