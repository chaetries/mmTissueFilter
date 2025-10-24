# ============================================================================
# main.py
# ============================================================================
"""
Tissue Annotation Tool - Main Application
PyQt5-based GUI for annotating tissue regions in Mueller matrix polarimetry images
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import TissueAnnotatorGUI


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Tissue Annotation Tool")
    app.setOrganizationName("MuellerMatrixAnalysis")

    window = TissueAnnotatorGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
