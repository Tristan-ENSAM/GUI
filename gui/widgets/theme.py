# -*- coding: utf-8 -*-
"""
Application look: Fusion style, a light palette and one style sheet, so
every tab shares the same panels, buttons, fields and tab bars.

`apply_theme(app)` is called once by gui.main before the main window is
built. Buttons that launch the main action of a panel are marked with
`mark_primary(button)` (accent colour).
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QPushButton, QStyleFactory

ACCENT = "#2f6fdf"
ACCENT_DARK = "#2459b8"
WINDOW = "#eef1f5"
PANEL = "#ffffff"
BORDER = "#cfd6df"
TEXT = "#1d2430"
MUTED = "#6b7480"

_QSS = """
QMainWindow, QDialog {{ background: {window}; }}
QToolTip {{ color: {text}; background: #fffef2; border: 1px solid {border};
           padding: 4px; }}

/* Panels: white cards with a bold title. */
QGroupBox {{ background: {panel}; border: 1px solid {border};
            border-radius: 6px; margin-top: 18px; padding: 10px 8px 6px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
                   left: 8px; top: 2px; padding: 0 4px; color: {accent_dark}; }}

/* Tab bars: flat, the current tab underlined with the accent. */
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 4px;
                   background: {window}; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {muted};
               padding: 6px 14px; border: none;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {text}; }}
QTabBar::tab:selected {{ color: {text}; border-bottom: 2px solid {accent};
                        font-weight: 600; }}

/* Buttons. */
QPushButton {{ background: {panel}; color: {text}; border: 1px solid {border};
              border-radius: 4px; padding: 4px 12px; min-height: 18px; }}
QPushButton:hover {{ background: #f3f6fa; border-color: #b5bfcc; }}
QPushButton:pressed {{ background: #e5ebf2; }}
QPushButton:disabled {{ color: #a7aeb8; background: #f6f7f9;
                       border-color: #e1e5ea; }}
QPushButton[primary="true"] {{ background: {accent}; color: white;
                              border-color: {accent}; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {accent_dark}; }}
QPushButton[primary="true"]:disabled {{ background: #9fb7e3;
                                       border-color: #9fb7e3; color: #eef3fb; }}

/* Fields. */
QLineEdit, QPlainTextEdit, QTextEdit {{
    background: {panel}; border: 1px solid {border}; border-radius: 4px;
    padding: 2px 6px; selection-background-color: {accent}; }}
QLineEdit:focus {{ border-color: {accent}; }}
QLineEdit:disabled {{ color: #9aa2ac; background: #f4f6f8; }}
QPlainTextEdit {{ font-family: Consolas, "DejaVu Sans Mono", monospace; }}

QTableView, QTableWidget {{ background: {panel}; border: 1px solid {border};
                          gridline-color: #e4e8ed;
                          selection-background-color: #dbe7fb;
                          selection-color: {text}; }}
QHeaderView::section {{ background: #f4f6f9; color: {text}; border: none;
                       border-right: 1px solid #e4e8ed;
                       border-bottom: 1px solid {border}; padding: 4px 6px;
                       font-weight: 600; }}
QScrollArea {{ border: none; background: transparent; }}
QSplitter::handle {{ background: {window}; }}
QProgressBar {{ border: 1px solid {border}; border-radius: 4px;
               background: {panel}; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}
"""


def style_sheet() -> str:
    return _QSS.format(accent=ACCENT, accent_dark=ACCENT_DARK, window=WINDOW,
                       panel=PANEL, border=BORDER, text=TEXT, muted=MUTED)


def apply_theme(app: QApplication) -> None:
    """Fusion style + light palette + the shared style sheet."""
    app.setStyle(QStyleFactory.create("Fusion"))
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(WINDOW))
    pal.setColor(QPalette.Base, QColor(PANEL))
    pal.setColor(QPalette.AlternateBase, QColor("#f6f8fa"))
    pal.setColor(QPalette.Button, QColor(PANEL))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor("#9aa2ac"))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("white"))
    pal.setColor(QPalette.ToolTipBase, QColor("#fffef2"))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    app.setPalette(pal)
    app.setStyleSheet(style_sheet())


def mark_primary(button: QPushButton) -> QPushButton:
    """Accent colour for the main action of a panel."""
    button.setProperty("primary", True)
    return button
