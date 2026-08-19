#!/bin/bash
# ============================================================================
# Build script for Tissue Annotation Tool - macOS
# ============================================================================

set -e  # Exit on error

echo "===================================="
echo "Tissue Annotation Tool - macOS Build"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    echo "Visit: https://www.python.org/downloads/"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

echo "[1/4] Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo ""
echo "[2/4] Cleaning previous builds..."
rm -rf build dist

echo ""
echo "[3/4] Building application with PyInstaller..."
pyinstaller tissue_annotator.spec

echo ""
echo "[4/4] Build complete!"
echo ""
echo "===================================="
echo "Your application is ready in: dist/"
echo ""
echo "Application bundle: dist/TissueAnnotator.app"
echo ""
echo "To run the application:"
echo "  1. Navigate to dist/ folder"
echo "  2. Double-click TissueAnnotator.app"
echo ""
echo "To distribute:"
echo "  Option 1: Zip the .app bundle"
echo "    cd dist"
echo "    zip -r TissueAnnotator.zip TissueAnnotator.app"
echo ""
echo "  Option 2: Create a DMG (requires additional tools)"
echo "    You can use 'create-dmg' or similar tools"
echo "===================================="
echo ""

# Make the .app executable
chmod -R 755 dist/TissueAnnotator.app

echo "Build process completed successfully!"
