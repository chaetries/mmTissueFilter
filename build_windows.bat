@echo off
REM ============================================================================
REM Build script for Tissue Annotation Tool - Windows
REM ============================================================================

echo ====================================
echo Tissue Annotation Tool - Windows Build
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/4] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo [3/4] Building executable with PyInstaller...
pyinstaller tissue_annotator.spec

if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo [4/4] Build complete!
echo.
echo ====================================
echo Your application is ready in: dist\TissueAnnotator\
echo.
echo To run the application:
echo   1. Navigate to: dist\TissueAnnotator\
echo   2. Double-click: TissueAnnotator.exe
echo.
echo To distribute:
echo   - Zip the entire "dist\TissueAnnotator" folder
echo   - Users can extract and run TissueAnnotator.exe
echo ====================================
echo.

pause
