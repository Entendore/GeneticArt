"""Application constants, default parameter values, and stylesheet."""

from __future__ import annotations

VERSION = "1.2.0"

IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "512 × 512": (512, 512),
    "600 × 600": (600, 600),
    "800 × 800": (800, 800),
    "1024 × 1024": (1024, 1024),
}

DEFAULTS: dict[str, int | float | str] = {
    "image_size_key": "600 × 600",
    "population_size": 6,
    "num_fractals": 3,
    "mutation_rate": 0.20,
    "frames_per_gen": 24,
    "perlin_scale": 0.08,
    "rotation_speed": 0.02,
    "anim_interval_ms": 50,
    "export_gens": 50,
}

# Catppuccin Mocha colour tokens
C = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "overlay0": "#6c7086",
    "overlay1": "#7f849c",
    "overlay2": "#9399b2",
    "subtext0": "#a6adc8",
    "subtext1": "#bac2de",
    "text": "#cdd6f4",
    "lavender": "#b4befe",
    "blue": "#89b4fa",
    "sapphire": "#74c7ec",
    "sky": "#89dceb",
    "teal": "#94e2d5",
    "green": "#a6e3a1",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "maroon": "#eba0ac",
    "red": "#f38ba8",
    "mauve": "#cba6f7",
    "pink": "#f5c2e7",
    "flamingo": "#f2cdcd",
    "rosewater": "#f5e0dc",
}

DARK_STYLE = f"""
/* ── Global ── */
QMainWindow, QWidget {{
    background: {C['base']};
    color: {C['text']};
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
}}

/* ── Group boxes ── */
QGroupBox {{
    border: 1px solid {C['surface1']};
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 8px 8px 8px;
    font-weight: 600;
    color: {C['text']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {C['blue']};
}}

/* ── Buttons ── */
QPushButton {{
    background: {C['surface1']};
    border: 1px solid {C['surface2']};
    border-radius: 6px;
    padding: 6px 14px;
    color: {C['text']};
    font-weight: 500;
}}
QPushButton:hover {{
    background: {C['surface2']};
    border-color: {C['overlay0']};
}}
QPushButton:pressed {{
    background: {C['surface0']};
}}
QPushButton:disabled {{
    background: {C['surface0']};
    color: {C['overlay0']};
    border-color: {C['surface1']};
}}

/* ── Primary button variant ── */
QPushButton[primary="true"] {{
    background: {C['blue']};
    border-color: {C['sapphire']};
    color: {C['crust']};
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{
    background: {C['sapphire']};
}}
QPushButton[primary="true"]:disabled {{
    background: {C['surface0']};
    color: {C['overlay0']};
    border-color: {C['surface1']};
}}

/* ── Danger button variant ── */
QPushButton[danger="true"] {{
    background: {C['surface0']};
    border-color: {C['red']};
    color: {C['red']};
}}
QPushButton[danger="true"]:hover {{
    background: {C['red']};
    color: {C['crust']};
}}

/* ── Input fields ── */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background: {C['surface0']};
    border: 1px solid {C['surface1']};
    border-radius: 4px;
    padding: 4px 8px;
    color: {C['text']};
    selection-background-color: {C['surface1']};
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
    border-color: {C['blue']};
}}

/* ── Slider ── */
QSlider::groove:horizontal {{
    height: 6px;
    background: {C['surface0']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {C['blue']};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {C['sapphire']};
}}
QSlider::sub-page:horizontal {{
    background: {C['surface1']};
    border-radius: 3px;
}}

/* ── Checkbox ── */
QCheckBox {{
    spacing: 8px;
    color: {C['subtext1']};
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
}}
QCheckBox::indicator:unchecked {{
    border: 2px solid {C['surface2']};
    background: {C['surface0']};
}}
QCheckBox::indicator:checked {{
    border: 2px solid {C['blue']};
    background: {C['blue']};
}}
QCheckBox:hover::indicator:unchecked {{
    border-color: {C['overlay0']};
}}

/* ── Scroll area ── */
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {C['mantle']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C['surface2']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {C['overlay0']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* ── Combo box dropdown ── */
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {C['overlay0']};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {C['surface0']};
    color: {C['text']};
    selection-background-color: {C['surface1']};
    border: 1px solid {C['surface1']};
    border-radius: 4px;
    outline: none;
}}

/* ── Labels ── */
QLabel {{
    background: transparent;
    color: {C['subtext1']};
}}
QLabel[heading="true"] {{
    font-size: 14px;
    font-weight: 700;
    color: {C['text']};
}}

/* ── Progress dialog ── */
QProgressDialog {{
    background: {C['base']};
    color: {C['text']};
}}
QProgressBar {{
    background: {C['surface0']};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {C['text']};
}}
QProgressBar::chunk {{
    background: {C['blue']};
    border-radius: 4px;
}}

/* ── Status bar ── */
QStatusBar {{
    background: {C['mantle']};
    color: {C['subtext0']};
    border-top: 1px solid {C['surface0']};
    font-size: 11px;
    padding: 2px 8px;
}}

/* ── Message box ── */
QMessageBox {{
    background: {C['base']};
}}
QMessageBox QLabel {{
    color: {C['text']};
    font-size: 13px;
}}
"""