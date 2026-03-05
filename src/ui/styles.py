"""
UI Styles and Theme Constants
"""

# Dark theme colors
DARK = {
    "bg":        "#0D0F14",
    "surface":   "#161A23",
    "surface2":  "#1E2433",
    "border":    "#2A3045",
    "accent":    "#4F8EF7",
    "accent2":   "#7C5CEF",
    "text":      "#E8ECF4",
    "text_dim":  "#6B7A99",
    "success":   "#3DD68C",
    "warning":   "#F5A623",
    "error":     "#F75F5F",
    "result_bg": "#181D2A",
}


def get_main_stylesheet():
    """Returns the main application stylesheet."""
    return f"""
        QMainWindow, QWidget {{
            background: {DARK['bg']};
            color: {DARK['text']};
        }}
        QWidget#Header {{
            background: {DARK['surface']};
            border-bottom: 1px solid {DARK['border']};
        }}
        QWidget#LeftPanel {{
            background: {DARK['surface']};
            border-right: 1px solid {DARK['border']};
        }}
        QWidget#RightPanel {{ background: {DARK['bg']}; }}
        QLineEdit {{
            background: {DARK['surface2']};
            border: 1.5px solid {DARK['border']};
            border-radius: 8px;
            padding: 0 14px;
            color: {DARK['text']};
        }}
        QLineEdit:focus {{ border-color: {DARK['accent']}; }}
        QPushButton {{
            background: {DARK['surface2']};
            border: 1px solid {DARK['border']};
            border-radius: 6px;
            color: {DARK['text']};
            padding: 0 12px;
        }}
        QPushButton:hover {{
            background: {DARK['accent']};
            color: white;
            border-color: {DARK['accent']};
        }}
        QComboBox {{
            background: {DARK['surface2']};
            border: 1px solid {DARK['border']};
            border-radius: 6px;
            color: {DARK['text']};
            padding: 0 10px;
            font-family: 'Courier New';
            font-size: 11px;
        }}
        QComboBox QAbstractItemView {{
            background: {DARK['surface2']};
            border: 1px solid {DARK['border']};
            color: {DARK['text']};
            selection-background-color: {DARK['accent']};
        }}
        QListWidget {{
            background: {DARK['bg']};
            border: 1px solid {DARK['border']};
            border-radius: 6px;
            color: {DARK['text']};
            outline: none;
        }}
        QListWidget::item {{ padding: 4px 8px; border-radius: 4px; }}
        QListWidget::item:selected {{ background: {DARK['accent']}; }}
        QListWidget::item:hover {{ background: {DARK['surface2']}; }}
        QGroupBox {{
            border: 1px solid {DARK['border']};
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 8px;
            color: {DARK['text_dim']};
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        QCheckBox {{ color: {DARK['text']}; spacing: 8px; }}
        QCheckBox::indicator {{
            width: 14px; height: 14px;
            border: 1px solid {DARK['border']};
            border-radius: 3px;
            background: {DARK['surface2']};
        }}
        QCheckBox::indicator:checked {{
            background: {DARK['accent']};
            border-color: {DARK['accent']};
        }}
        QSpinBox {{
            background: {DARK['surface2']};
            border: 1px solid {DARK['border']};
            border-radius: 4px;
            color: {DARK['text']};
            padding: 2px 4px;
        }}
        QScrollArea#ResultsScroll {{ background: transparent; border: none; }}
        QScrollBar:vertical {{
            background: {DARK['surface']};
            width: 8px; border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: {DARK['border']};
            border-radius: 4px; min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {DARK['accent']}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QProgressBar {{
            background: {DARK['surface2']};
            border: none; border-radius: 2px;
        }}
        QProgressBar::chunk {{
            background: {DARK['accent']}; border-radius: 2px;
        }}
        QToolButton {{
            background: {DARK['surface2']};
            border: 1px solid {DARK['border']};
            border-radius: 6px; color: {DARK['text']};
        }}
        QToolButton:hover {{ background: {DARK['accent']}; color: white; }}
    """
