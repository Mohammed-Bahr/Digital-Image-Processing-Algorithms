"""Dark application stylesheet."""

APP_QSS = """
* {
    font-family: "Inter", "Segoe UI", "Noto Sans", "DejaVu Sans", sans-serif;
}

QWidget {
    background-color: #0f131b;
    color: #d7dee9;
    font-size: 13px;
}

QMainWindow, QDialog { background-color: #0f131b; }

QMenuBar {
    background-color: #131823;
    padding: 2px 6px;
    border-bottom: 1px solid #202838;
}
QMenuBar::item { padding: 5px 10px; border-radius: 5px; background: transparent; }
QMenuBar::item:selected { background: #222a3a; }

QMenu {
    background-color: #161c28;
    border: 1px solid #293245;
    padding: 5px;
}
QMenu::item { padding: 6px 24px 6px 16px; border-radius: 5px; }
QMenu::item:selected { background-color: #23535d; }
QMenu::separator { height: 1px; background: #293245; margin: 5px 8px; }

QToolBar {
    background-color: #131823;
    border: none;
    border-bottom: 1px solid #202838;
    padding: 5px 8px;
    spacing: 5px;
}
QToolButton {
    background: transparent;
    padding: 6px 11px;
    border-radius: 6px;
    color: #cfd7e3;
}
QToolButton:hover { background: #222a3a; }
QToolButton:pressed { background: #2b3548; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1a2030;
    border: 1px solid #2c3548;
    border-radius: 6px;
    padding: 4px 7px;
    selection-background-color: #2f6f7a;
    min-height: 18px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #3f8e99;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: #1a2030;
    border: 1px solid #2c3548;
    selection-background-color: #23535d;
    outline: none;
}

QPushButton {
    background-color: #1d2534;
    border: 1px solid #2e394d;
    border-radius: 7px;
    padding: 6px 14px;
    color: #d7dee9;
}
QPushButton:hover { background-color: #273246; }
QPushButton:pressed { background-color: #1a2231; }
QPushButton#primary {
    background-color: #2a7a86;
    border-color: #3d9aa7;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primary:hover { background-color: #338e9b; }

QTreeWidget {
    background-color: #141924;
    border: 1px solid #232c3d;
    border-radius: 8px;
    outline: none;
    padding: 4px;
}
QTreeWidget::item {
    padding: 6px 4px;
    border-radius: 5px;
    margin: 1px 2px;
}
QTreeWidget::item:hover { background-color: #1d2432; }
QTreeWidget::item:selected { background-color: #23535d; color: #ffffff; }
QTreeWidget::branch { background: transparent; }

QTabWidget::pane {
    border: 1px solid #232c3d;
    border-radius: 7px;
    top: -1px;
    background: #10151e;
}
QTabBar::tab {
    background: #151b26;
    color: #98a4b7;
    padding: 7px 14px;
    margin-right: 3px;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    border: 1px solid transparent;
}
QTabBar::tab:selected {
    background: #23535d;
    color: #eaf6f5;
    border-color: #2d6d78;
}
QTabBar::tab:hover:!selected { background: #1d2534; }

QPlainTextEdit {
    background-color: #0e131c;
    border: 1px solid #232c3d;
    border-radius: 7px;
    color: #b9c6d8;
    selection-background-color: #2f6f7a;
    padding: 6px;
}

QGraphicsView { background-color: #0d1118; border: none; }

QLabel { background: transparent; }
QLabel#opTitle { font-size: 20px; font-weight: 700; color: #f2f6fb; }
QLabel#opMeta { color: #5fc6d2; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QLabel#opDesc { color: #a8b4c6; font-size: 12.5px; }
QLabel#sectionLabel { color: #7b879b; font-size: 10.5px; font-weight: 700; letter-spacing: 1.2px; }
QLabel#hintLabel { color: #6f7c90; font-size: 11.5px; }
QLabel#paramLabel { color: #b6c1d2; }
QLabel#kernel {
    font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
    font-size: 11px;
    color: #9fd0c8;
    background-color: #11161f;
    border: 1px solid #232c3d;
    border-radius: 7px;
    padding: 8px;
}
QLabel#paneTitle {
    color: #8fa3bd;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.5px;
}
QLabel#paneDims { color: #5f6c80; font-size: 11px; }
QLabel#statusInfo { color: #7e8ba0; padding-right: 8px; }

QFrame#separator { color: #232c3d; background: #232c3d; max-height: 1px; }

QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }

QSplitter::handle { background-color: #0b0f16; }
QSplitter::handle:hover { background-color: #1f6a74; }
QSplitter::handle:horizontal { width: 4px; }
QSplitter::handle:vertical { height: 4px; }

QStatusBar {
    background-color: #131823;
    border-top: 1px solid #202838;
    color: #7e8ba0;
}
QStatusBar::item { border: none; }

QCheckBox { spacing: 7px; color: #c2ccdb; }
QCheckBox::indicator {
    width: 15px; height: 15px;
    border-radius: 4px;
    border: 1px solid #3a465c;
    background: #1a2030;
}
QCheckBox::indicator:hover { border-color: #4d7d8a; }
QCheckBox::indicator:checked {
    background: #3d9aa7;
    border-color: #59b6c2;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #2b3548;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #3a4862; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #2b3548;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QToolTip {
    background-color: #1a2231;
    color: #d7dee9;
    border: 1px solid #344055;
    padding: 5px;
}
"""
