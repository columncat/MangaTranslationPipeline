from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .config import AppConfig
from .gui.download_dialog import DownloadProgressDialog
from .gui.language_dialog import LanguageDialog
from .gui.main_window import MainWindow
from .i18n import set_language
from .ml.weights import ctd_weights_present
from .paths import ensure_dirs


def _maybe_show_language_dialog(config: AppConfig) -> None:
    """First-launch language picker. Skipped once ``ui_language`` is set."""
    if config.ui_language in ("ko", "en"):
        set_language(config.ui_language)
        return
    dlg = LanguageDialog(current=None)
    dlg.exec()
    config.ui_language = dlg.selected
    set_language(config.ui_language)
    config.save()


def _maybe_show_download_dialog(config: AppConfig) -> None:
    """First-launch model-weight download popup.

    Skipped if ``first_run_done`` is True AND the CTD weight file exists.
    The CTD check is a cheap way to detect a wiped ``models/`` cache
    between launches.
    """
    if config.first_run_done and ctd_weights_present():
        return
    dlg = DownloadProgressDialog()
    dlg.start()
    dlg.exec()
    if dlg.all_ok:
        config.first_run_done = True
        config.save()


def main() -> int:
    ensure_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName("Manga Translation Pipeline")

    config = AppConfig.load()

    _maybe_show_language_dialog(config)
    _maybe_show_download_dialog(config)

    win = MainWindow(config=config)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
