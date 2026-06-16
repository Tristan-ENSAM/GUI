@echo off
REM ============================================================================
REM  Abaqus Cutting Pre-processor - DEBUG launcher
REM  Same as run_gui.bat but uses python.exe (console stays open) so any
REM  traceback is visible. If the venv already has the dependencies, it just
REM  launches - no host Python, no network needed. The create/install path
REM  only runs for a missing or incomplete venv.
REM ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_MARKER=%VENV_DIR%\.requirements_installed"

REM --- Fast path: a working venv with deps already present -> just launch ----
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import PySide6, matplotlib, numpy, PIL" >nul 2>&1
    if !errorlevel! == 0 goto :launch
    REM venv exists but can it even run python at all?
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
                echo %%P | findstr /I "\\WindowsApps\\" >nul
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

REM --- Install dependencies (with conda-venv SSL fix) ------------------------
echo [INFO] Installing dependencies...
set "BASE_PREFIX="
for /f "delims=" %%I in ('""%VENV_PY%" -c "import sys;print(sys.base_prefix)""') do set "BASE_PREFIX=%%I"
if defined BASE_PREFIX (
    if exist "!BASE_PREFIX!\Library\bin" (
        echo [INFO] Adding conda OpenSSL DLLs to PATH for pip.
        set "PATH=!BASE_PREFIX!\Library\bin;!BASE_PREFIX!\Library\usr\bin;!BASE_PREFIX!\Library\mingw-w64\bin;!PATH!"
    )
)
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install PySide6 matplotlib numpy Pillow
if !errorlevel! neq 0 (
    echo [ERROR] Install failed. If it is an SSL/proxy error, your machine may
    echo         block PyPI; use a corporate proxy, e.g.:
    echo           "%VENV_PY%" -m pip install --proxy http://USER:PASS@HOST:PORT PySide6 matplotlib numpy Pillow
    pause
    exit /b 1
)
echo installed > "%REQ_MARKER%"

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
