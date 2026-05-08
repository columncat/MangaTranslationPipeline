from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .gui.main_window import MainWindow
from .paths import ensure_dirs


def main() -> int:
    ensure_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName("Manga Translation Pipeline")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
