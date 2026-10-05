@echo off
REM ============================================================================
REM  Abaqus Cutting Pre-processor - DEBUG launcher
REM  Uses python.exe (console stays open) so any traceback is visible. If the
REM  venv already has the core dependencies it just launches - no host Python,
REM  no network needed. Optional (experimental) deps are installed best-effort.
REM
REM  A host Python 3.11+ is needed only to (re)create the venv. It is searched
REM  in this order: %%GUI_HOST_PY%% (full path to python.exe), "py -3", PATH,
REM  then the default Anaconda / Miniconda / python.org install folders.
REM ============================================================================
set "GUI_DEBUG=1"
set "GUI_DEBUG_DIR=%~dp0logs"
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_MARKER=%VENV_DIR%\.requirements_installed"
set "REQ_FILE=%~dp0requirements.txt"
set "REBUILD="

REM --- Fast path: venv with the CORE deps already present -> just launch ------
REM Optional deps (Pillow, OpenCV) are NOT gated here so the GUI always starts.
if exist "%VENV_PY%" (
    call :venv_conda_path
    "%VENV_PY%" -c "import PySide6, matplotlib, numpy, scipy" >nul 2>&1
    if !errorlevel! == 0 goto :optional
    "%VENV_PY%" -c "import sys" >nul 2>&1
    if !errorlevel! neq 0 set "REBUILD=1"
)
REM Runnable venv that only lacks packages: no host Python needed.
if exist "%VENV_PY%" if not defined REBUILD goto :install

REM --- Need a host Python only to CREATE (or rebuild) the venv ---------------
call :find_host
if not defined HOST_PY goto :no_host
if defined REBUILD (
    echo [WARN] Existing .venv is not runnable here ^(copied from another
    echo        PC / Python version^). It is kept until a host Python is found.
)

if defined REBUILD (
    echo [WARN] Existing .venv is not runnable here ^(copied from another
    echo        PC / Python version^). Rebuilding it...
    rmdir /s /q "%VENV_DIR%"
)
echo [INFO] Creating venv with: !HOST_PY!
!HOST_PY! -m venv "%VENV_DIR%"
if not exist "%VENV_PY%" (
    echo [ERROR] venv creation failed with: !HOST_PY!
    pause
    exit /b 1
)
call :venv_conda_path

REM --- Full install (fresh venv): requirements.txt --------------------------
:install
echo [INFO] Installing dependencies from requirements.txt...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r "%REQ_FILE%"
if !errorlevel! neq 0 (
    echo [ERROR] Install failed. If it is an SSL/proxy error, your machine may
    echo         block PyPI; use a corporate proxy, e.g.:
    echo           "%VENV_PY%" -m pip install --proxy http://USER:PASS@HOST:PORT -r "%REQ_FILE%"
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
echo [INFO] Launching GUI (console kept open while it runs)...
REM "%VENV_PY%" -m gui.main
"%VENV_PY%" -X faulthandler -m gui.main
set "GUI_RC=!errorlevel!"
REM Normal close (exit code 0): the console closes with the window.
REM Crash / error (non-zero code): keep it open so the traceback can be read
REM (the session log is also written to %GUI_DEBUG_DIR%).
if "!GUI_RC!" == "0" (
    endlocal
    exit
)
echo.
echo [INFO] GUI exited with code !GUI_RC! - console kept open to read the error.
pause
endlocal

:no_host
if defined REBUILD (
    echo [WARN] Existing .venv is not runnable here ^(copied from another
    echo        PC / Python version^). It is kept until a host Python is found.
)
echo [ERROR] No usable .venv here, and no host Python 3.11+ found to create it.
echo         Searched: GUI_HOST_PY, "py -3", PATH, and the default Anaconda,
echo         Miniconda and python.org install folders.
echo         Either install Python 3.11+ (python.org), or point this launcher
echo         at an existing python.exe and run it again, e.g. in a console:
echo           set GUI_HOST_PY=C:\path\to\python.exe
echo           "%~f0"
pause
exit /b 1

REM ============================================================================
REM  Subroutines
REM ============================================================================

REM --- find_host: sets HOST_PY (quoted command) to a Python 3.11+ ------------
:find_host
set "HOST_PY="
if defined GUI_HOST_PY call :try_host "%GUI_HOST_PY:"=%"
if not defined HOST_PY (
    where py >nul 2>&1
    if !errorlevel! == 0 (
        py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
        if !errorlevel! == 0 set "HOST_PY=py -3"
    )
)
if not defined HOST_PY (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined HOST_PY (
            echo %%P | findstr /I "\WindowsApps\" >nul
            if !errorlevel! neq 0 call :try_host "%%P"
        )
    )
)
for %%D in ("%ProgramData%\Anaconda3" "%ProgramData%\Miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\Continuum\anaconda3") do (
    if not defined HOST_PY call :try_host "%%~D\python.exe"
)
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*" "%ProgramFiles%\Python3*" "C:\Python3*") do (
    if not defined HOST_PY call :try_host "%%~D\python.exe"
)
goto :eof

REM --- try_host <python.exe>: accept it if it runs and is 3.11+ --------------
:try_host
if not exist "%~1" goto :eof
"%~1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set HOST_PY="%~1"
goto :eof

REM --- venv_conda_path: a venv built from Anaconda needs <base>\Library\bin --
REM on PATH, otherwise pip has no SSL (see TODO.md, "Venv non portable").
:venv_conda_path
if not exist "%VENV_DIR%\pyvenv.cfg" goto :eof
set "VHOME="
for /f "tokens=1,* delims==" %%A in ('findstr /B /I /C:"home" "%VENV_DIR%\pyvenv.cfg"') do set "VHOME=%%B"
if not defined VHOME goto :eof
for /f "tokens=*" %%H in ("!VHOME!") do set "VHOME=%%H"
if exist "!VHOME!\Library\bin" set "PATH=!VHOME!\Library\bin;!PATH!"
goto :eof
