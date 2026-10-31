#!/usr/bin/env python3
"""DIP Lab — launch the desktop application.

    python main.py

Requires: numpy, pillow, PyQt5.
"""

from __future__ import annotations

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from diplab.gui.main_window import MainWindow
from diplab.gui.style import APP_QSS


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("DIP Lab")
    app.setApplicationDisplayName("DIP Lab")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
