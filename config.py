"""Application constants and default parameter values."""

IMAGE_SIZES = {
    "512×512": (512, 512),
    "600×600": (600, 600),
    "800×800": (800, 800),
    "1024×1024": (1024, 1024),
}

DEFAULTS = {
    "image_size_key": "600×600",
    "population_size": 5,
    "num_fractals": 3,
    "mutation_rate": 0.20,
    "frames_per_gen": 20,
    "perlin_scale": 0.08,
    "rotation_speed": 0.02,
    "anim_interval_ms": 50,
    "export_gens": 50,
}

DARK_STYLE = """
QMainWindow, QWidget { background:#1e1e2e; color:#cdd6f4; }
QGroupBox { border:1px solid #45475a; border-radius:6px; margin-top:10px;
            padding-top:14px; font-weight:bold; color:#cdd6f4; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; color:#89b4fa; }
QPushButton { background:#45475a; border:1px solid #585b70; border-radius:4px;
              padding:5px 8px; color:#cdd6f4; }
QPushButton:hover { background:#585b70; }
QPushButton:pressed { background:#313244; }
QPushButton:disabled { background:#313244; color:#6c7086; }
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background:#313244; border:1px solid #45475a; border-radius:3px;
    padding:3px; color:#cdd6f4; }
QSlider::groove:horizontal { height:6px; background:#45475a; border-radius:3px; }
QSlider::handle:horizontal { background:#89b4fa; width:14px; margin:-4px 0;
                             border-radius:7px; }
QScrollArea { border:none; }
QCheckBox { spacing:5px; }
QCheckBox::indicator { width:15px; height:15px; }
QCheckBox::indicator:unchecked { border:1px solid #585b70; background:#313244;
                                  border-radius:3px; }
QCheckBox::indicator:checked { border:1px solid #89b4fa; background:#89b4fa;
                                border-radius:3px; }
QComboBox::drop-down { border:none; }
QComboBox QAbstractItemView { background:#313244; color:#cdd6f4;
                              selection-background-color:#45475a; }
QProgressDialog { background:#1e1e2e; color:#cdd6f4; }
QLabel { background:transparent; }
"""