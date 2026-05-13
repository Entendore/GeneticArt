"""All custom widgets: canvas, fitness chart, gallery, control panel, and status bar."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QPainter, QPen, QColor, QLinearGradient, QRadialGradient,
    QPainterPath, QFont, QBrush, QPixmap,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QSlider, QLabel, QLineEdit, QScrollArea, QFileDialog,
    QSizePolicy, QFrame, QDockWidget,
)

from config import DEFAULTS, IMAGE_SIZES, C, PALETTE_PRESETS, EASING_MODES, SHORTCUTS
from genetics import GeneticAlgorithm, Gene, Individual, fitness
from renderers import draw_fractal_qp, draw_gradient_qp, render_thumbnail_qpixmap


# ─── Fractal Canvas ───────────────────────────────────────────────────────────

class FractalCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ga: Optional[GeneticAlgorithm] = None
        self.perlin_scale = DEFAULTS["perlin_scale"]
        self.rotation_speed = DEFAULTS["rotation_speed"]
        self.show_gradient = True
        self.show_swirl = True
        self.shimmer = True
        self.highlight_best = False
        self.show_leaves = True
        self.show_glow = True
        self.palette_name = DEFAULTS["palette"]
        self.easing = DEFAULTS["easing"]
        self.selected_index: int = -1  # -1 = show all
        self._interp_pop: List[Individual] = []
        self._gt = 0.0
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_ga(self, ga: GeneticAlgorithm) -> None:
        self.ga = ga

    def update_frame(self, interp_pop: List[Individual], global_time: float) -> None:
        self._interp_pop = interp_pop
        self._gt = global_time
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor(C["crust"]))

        if self.ga is None or not self._interp_pop:
            self._draw_placeholder(p)
            p.end()
            return

        w, h = self.ga.image_size
        scale = min(self.width() / w, self.height() / h) * 0.95
        ox = (self.width() - w * scale) / 2.0
        oy = (self.height() - h * scale) / 2.0

        p.save()
        p.translate(ox, oy)
        p.scale(scale, scale)

        self._draw_canvas_bg(p, w, h)
        self._draw_fractals(p, w, h)

        p.restore()
        self._draw_border(p, ox, oy, w * scale, h * scale)
        p.end()

    def _draw_placeholder(self, p: QPainter) -> None:
        p.setPen(QColor(C["overlay0"]))
        font = QFont("Segoe UI", 14)
        font.setItalic(True)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignCenter, "Press Start or Reset to begin")

    def _draw_canvas_bg(self, p: QPainter, w: int, h: int) -> None:
        if self.show_gradient:
            avg = self._get_average_colour()
            draw_gradient_qp(p, (w, h), avg * 0.8, avg * 0.1)
        else:
            p.fillRect(0, 0, w, h, QColor(0, 0, 0))

    def _draw_fractals(self, p: QPainter, w: int, h: int) -> None:
        pop = self._interp_pop
        n = len(pop)

        if self.selected_index >= 0 and self.selected_index < n:
            indices = [self.selected_index]
        elif self.highlight_best:
            indices = [0]
        else:
            indices = list(range(n))

        for rank, i in enumerate(indices):
            swirl_angle = (
                self._gt * self.rotation_speed * (i + 1) if self.show_swirl else 0.0
            )
            alpha = 1.0 if (self.highlight_best or self.selected_index >= 0) else (0.12 + 0.88 * (i / max(1, n - 1)))

            for gene in pop[i]:
                draw_fractal_qp(
                    p, gene, self._gt, (w, h),
                    alpha=alpha, swirl=swirl_angle,
                    perlin_scale=self.perlin_scale, shimmer=self.shimmer,
                    palette_name=self.palette_name,
                    show_leaves=self.show_leaves, show_glow=self.show_glow,
                )

    def _get_average_colour(self) -> np.ndarray:
        from color_utils import average_colour
        return average_colour(self.ga.population, self._gt, self.palette_name)

    def _draw_border(self, p: QPainter, x: float, y: float, w: float, h: float) -> None:
        pen = QPen(QColor(C["surface1"]), 1)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(x - 2, y - 2, w + 4, h + 4), 4, 4)


# ─── Fitness Chart ────────────────────────────────────────────────────────────

class FitnessChart(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.history: List[float] = []
        self.avg_history: List[float] = []
        self.diversity_history: List[float] = []
        self.show_diversity = True
        self.setMinimumHeight(140)
        self.setMaximumHeight(220)

    def set_history(self, history: List[float], avg_history: List[float] | None = None,
                    diversity_history: List[float] | None = None) -> None:
        self.history = list(history)
        self.avg_history = list(avg_history) if avg_history else []
        self.diversity_history = list(diversity_history) if diversity_history else []
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        ml, mb, mt, mr = 45, 24, 12, 12

        bg = QRadialGradient(w / 2, h / 2, w)
        bg.setColorAt(0, QColor(C["surface0"]))
        bg.setColorAt(1, QColor(C["mantle"]))
        p.fillRect(0, 0, w, h, bg)
        p.setPen(QPen(QColor(C["surface1"]), 1))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 6, 6)

        if len(self.history) < 2:
            p.setPen(QColor(C["overlay0"]))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "Waiting for data…")
            p.end()
            return

        pw, ph = w - ml - mr, h - mt - mb
        mn, mx = min(self.history), max(self.history)
        rng = (mx - mn) if mx != mn else 1.0

        # Grid lines
        p.setFont(QFont("Segoe UI", 7))
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = int(mt + (1.0 - frac) * ph)
            p.setPen(QPen(QColor(C["surface1"]), 1, Qt.DotLine))
            p.drawLine(ml, yy, w - mr, yy)
            p.setPen(QColor(C["overlay0"]))
            p.drawText(4, yy + 4, f"{mn + frac * rng:.0f}")

        # Axes
        p.setPen(QPen(QColor(C["surface2"]), 1))
        p.drawLine(ml, mt, ml, h - mb)
        p.drawLine(ml, h - mb, w - mr, h - mb)

        # Diversity line
        if self.show_diversity and len(self.diversity_history) >= 2:
            d_mn, d_mx = min(self.diversity_history), max(self.diversity_history)
            d_rng = (d_mx - d_mn) if d_mx != d_mn else 1.0
            self._draw_line(p, self.diversity_history, d_mn, d_rng, ml, mt, pw, ph,
                            QColor(C["teal"], 100), QColor(C["teal"], 15), 1.0)

        # Average fitness line
        if len(self.avg_history) >= 2:
            self._draw_line(p, self.avg_history, mn, rng, ml, mt, pw, ph,
                            QColor(C["mauve"], 120), QColor(C["mauve"], 20), 1.5)

        # Best fitness line
        self._draw_line(p, self.history, mn, rng, ml, mt, pw, ph,
                        QColor(C["blue"]), QColor(C["blue"], 40), 2.0)

        # End point dot
        lx = ml + pw
        ly = mt + (1.0 - (self.history[-1] - mn) / rng) * ph
        glow = QRadialGradient(lx, ly, 8)
        glow.setColorAt(0, QColor(C["rosewater"], 80))
        glow.setColorAt(1, QColor(C["rosewater"], 0))
        p.setBrush(glow)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(lx, ly), 8, 8)
        p.setBrush(QColor(C["rosewater"]))
        p.setPen(QPen(QColor(C["text"]), 1))
        p.drawEllipse(QPointF(lx, ly), 4, 4)

        # Labels
        p.setPen(QColor(C["overlay0"]))
        p.drawText(ml, h - 4, "0")
        p.drawText(w - mr - 25, h - 4, f"{len(self.history) - 1}")
        p.setPen(QColor(C["subtext1"]))
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(ml + 4, mt + 10, "Fitness")

        # Legend
        legend_x = ml + pw - 80
        legend_y = mt + 8
        for i, (color, label) in enumerate([
            (C["blue"], "Best"), (C["mauve"], "Avg"), (C["teal"], "Div")
        ]):
            yy = legend_y + i * 14
            p.setPen(QPen(QColor(color), 2))
            p.drawLine(legend_x, yy, legend_x + 14, yy)
            p.setPen(QColor(C["subtext0"]))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(legend_x + 18, yy + 4, label)

        p.end()

    def _draw_line(self, p, data, mn, rng, mx, my, pw, ph, line_color, fill_color, lw=2.0):
        n = len(data)
        path = QPainterPath()
        for i, val in enumerate(data):
            x = mx + (i / max(1, n - 1)) * pw
            y = my + (1.0 - (val - mn) / rng) * ph
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        fill = QPainterPath(path)
        fill.lineTo(mx + pw, my + ph)
        fill.lineTo(mx, my + ph)
        fill.closeSubpath()
        gradient = QLinearGradient(0, my, 0, my + ph)
        gradient.setColorAt(0, fill_color)
        gradient.setColorAt(1, QColor(fill_color.red(), fill_color.green(), fill_color.blue(), 0))
        p.fillPath(fill, gradient)
        p.setPen(QPen(line_color, lw))
        p.drawPath(path)


# ─── Gallery Item ─────────────────────────────────────────────────────────────

class GalleryItem(QLabel):
    """Clickable thumbnail for one individual."""
    clicked = Signal(int)

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self.index = index
        self.selected = False
        self.setFixedSize(120, 120)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"border: 2px solid {C['surface1']}; border-radius: 6px;")
        self.setToolTip(f"Individual #{index + 1}\nClick to focus")

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.index)
        super().mousePressEvent(event)

    def set_selected(self, sel: bool) -> None:
        self.selected = sel
        color = C["blue"] if sel else C["surface1"]
        self.setStyleSheet(f"border: 2px solid {color}; border-radius: 6px;")


# ─── Population Gallery ───────────────────────────────────────────────────────

class PopulationGallery(QWidget):
    """Horizontal strip of individual thumbnails."""
    individual_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items: List[GalleryItem] = []
        self._selected = -1
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(6)
        self._layout.addStretch()

        btn_clear = QPushButton("✕")
        btn_clear.setFixedSize(24, 24)
        btn_clear.setToolTip("Clear selection (show all)")
        btn_clear.clicked.connect(self._clear_selection)
        self._layout.addWidget(btn_clear)

    def update_thumbnails(self, population: List[Individual], global_time: float,
                          perlin_scale: float, palette_name: str,
                          image_size: Tuple[int, int]) -> None:
        # Remove old items
        for item in self._items:
            self._layout.removeWidget(item)
            item.deleteLater()
        self._items.clear()

        for i, ind in enumerate(population):
            item = GalleryItem(i)
            pm = render_thumbnail_qpixmap(
                ind, 112, global_time, perlin_scale, palette_name
            )
            item.setPixmap(pm.scaled(112, 112, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            item.clicked.connect(self._on_click)
            self._layout.insertWidget(self._layout.count() - 1, item)
            self._items.append(item)

        if self._selected >= 0:
            self._set_selected(self._selected)

    def _on_click(self, index: int) -> None:
        if self._selected == index:
            self._clear_selection()
        else:
            self._set_selected(index)
            self.individual_selected.emit(index)

    def _set_selected(self, index: int) -> None:
        self._selected = index
        for item in self._items:
            item.set_selected(item.index == index)

    def _clear_selection(self) -> None:
        self._selected = -1
        for item in self._items:
            item.set_selected(False)
        self.individual_selected.emit(-1)


# ─── Status Bar Widget ────────────────────────────────────────────────────────

class StatusBarWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(16)
        self._labels = {}
        items = [
            ("state", "State", C["text"]),
            ("gen", "Gen: 0", C["blue"]),
            ("frame", "Frame: 0/0", C["sapphire"]),
            ("fitness", "Fit: 0.0", C["green"]),
            ("mutation", "Mut: 0.20", C["peach"]),
            ("diversity", "Div: 0.0", C["teal"]),
            ("pop", "Pop: 0", C["lavender"]),
            ("fps", "FPS: 0", C["yellow"]),
        ]
        for key, text, color in items:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {color}; font-weight: 600;")
            layout.addWidget(lbl)
            self._labels[key] = lbl
        layout.addStretch()

    def _set(self, key: str, value: str) -> None:
        if key in self._labels:
            self._labels[key].setText(value)

    @property
    def state(self) -> str:
        return self._labels["state"].text()

    @state.setter
    def state(self, v: str) -> None:
        color = C["green"] if v == "Running" else C["yellow"]
        icon = "●" if v == "Running" else "○"
        self._labels["state"].setText(f"{icon} {v}")
        self._labels["state"].setStyleSheet(f"color: {color}; font-weight: 600;")

    @property
    def gen(self) -> int:
        return int(self._labels["gen"].text().split(":")[1].strip())

    @gen.setter
    def gen(self, v: int) -> None:
        self._set("gen", f"Gen: {v}")

    @property
    def frame(self) -> str:
        return self._labels["frame"].text()

    @frame.setter
    def frame(self, v: str) -> None:
        self._set("frame", f"Frm: {v}")

    @property
    def fitness(self) -> str:
        return self._labels["fitness"].text()

    @fitness.setter
    def fitness(self, v: str) -> None:
        self._set("fitness", f"Fit: {v}")

    @property
    def mutation_rate(self) -> str:
        return self._labels["mutation"].text()

    @mutation_rate.setter
    def mutation_rate(self, v: str) -> None:
        self._set("mutation", f"Mut: {v}")

    @property
    def diversity(self) -> str:
        return self._labels["diversity"].text()

    @diversity.setter
    def diversity(self, v: str) -> None:
        self._set("diversity", f"Div: {v}")

    @property
    def pop_size(self) -> int:
        return int(self._labels["pop"].text().split(":")[1].strip())

    @pop_size.setter
    def pop_size(self, v: int) -> None:
        self._set("pop", f"Pop: {v}")

    @property
    def fps(self) -> int:
        return int(self._labels["fps"].text().split(":")[1].strip())

    @fps.setter
    def fps(self, v: int) -> None:
        self._set("fps", f"FPS: {v}")


# ─── Control Panel ────────────────────────────────────────────────────────────

class ControlPanel(QScrollArea):
    sig_start = Signal()
    sig_pause = Signal()
    sig_reset = Signal()
    sig_step = Signal()
    sig_screenshot = Signal()
    sig_export_seq = Signal()
    sig_export_gif = Signal()
    sig_save = Signal()
    sig_load = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedWidth(330)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        root = QWidget()
        self._layout = QVBoxLayout(root)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(8, 8, 8, 8)

        self._build_title()
        self._build_evolution_group()
        self._build_visual_group()
        self._build_animation_group()
        self._build_export_group()
        self._build_info_group()
        self._build_chart()

        self._layout.addStretch()
        self.setWidget(root)

    def _build_title(self) -> None:
        title = QLabel("🧬 Fractal Genetic Art")
        title.setStyleSheet(f"font-size:16px; font-weight:700; color:{C['sky']}; padding:8px 0 4px 0;")
        self._layout.addWidget(title)
        subtitle = QLabel("Evolutionary fractal tree generator v2.0")
        subtitle.setStyleSheet(f"font-size:11px; color:{C['overlay0']}; padding-bottom:4px;")
        self._layout.addWidget(subtitle)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {C['surface1']};")
        self._layout.addWidget(sep)

    def _build_evolution_group(self) -> None:
        group = QGroupBox("Evolution")
        layout = QFormLayout()
        layout.setSpacing(8)
        layout.setLabelAlignment(Qt.AlignRight)

        self.sp_pop = self._spin(2, 30, DEFAULTS["population_size"])
        self.sp_pop.setToolTip("Number of individuals in the population")
        self.sp_fract = self._spin(1, 10, DEFAULTS["num_fractals"])
        self.sp_fract.setToolTip("Number of fractal trees per individual")
        self.sp_mut = self._dspin(0.0, 1.0, DEFAULTS["mutation_rate"], 0.05, 2)
        self.sp_mut.setToolTip("Base mutation rate (adaptive will adjust this)")
        self.sp_fpg = self._spin(1, 100, DEFAULTS["frames_per_gen"])
        self.sp_fpg.setToolTip("Animation frames between each generation")
        self.sp_seed = self._spin(0, 999999, DEFAULTS["seed"])
        self.sp_seed.setToolTip("Random seed for reproducible runs (0 = random)")
        self.chk_adaptive = QCheckBox("Adaptive mutation")
        self.chk_adaptive.setChecked(DEFAULTS["adaptive_mutation"])
        self.chk_adaptive.setToolTip("Automatically increase mutation when evolution stagnates")

        layout.addRow("Population:", self.sp_pop)
        layout.addRow("Trees/ind:", self.sp_fract)
        layout.addRow("Mutation:", self.sp_mut)
        layout.addRow("Frames/gen:", self.sp_fpg)
        layout.addRow("Seed:", self.sp_seed)
        layout.addRow(self.chk_adaptive)

        group.setLayout(layout)
        self._layout.addWidget(group)

    def _build_visual_group(self) -> None:
        group = QGroupBox("Visual")
        layout = QFormLayout()
        layout.setSpacing(8)
        layout.setLabelAlignment(Qt.AlignRight)

        self.cb_size = QComboBox()
        self.cb_size.addItems(list(IMAGE_SIZES.keys()))
        self.cb_size.setCurrentText(DEFAULTS["image_size_key"])
        self.cb_size.setToolTip("Canvas resolution")

        self.cb_palette = QComboBox()
        self.cb_palette.addItems(PALETTE_PRESETS)
        self.cb_palette.setCurrentText(DEFAULTS["palette"])
        self.cb_palette.setToolTip("Color palette preset for fractal rendering")

        self.cb_easing = QComboBox()
        self.cb_easing.addItems(EASING_MODES)
        self.cb_easing.setCurrentText(DEFAULTS["easing"])
        self.cb_easing.setToolTip("Interpolation easing function for smooth transitions")

        self.sp_perlin = self._dspin(0.0, 0.5, DEFAULTS["perlin_scale"], 0.01, 3)
        self.sp_perlin.setToolTip("Perlin noise scale for organic movement")
        self.sp_rot = self._dspin(0.0, 0.2, DEFAULTS["rotation_speed"], 0.005, 3)
        self.sp_rot.setToolTip("Rotation speed for swirl effect")

        self.chk_grad = QCheckBox("Gradient background")
        self.chk_grad.setChecked(True)
        self.chk_swirl = QCheckBox("Swirl effect")
        self.chk_swirl.setChecked(True)
        self.chk_shimmer = QCheckBox("Shimmer mode")
        self.chk_shimmer.setChecked(True)
        self.chk_best = QCheckBox("Show best only")
        self.chk_best.setChecked(False)
        self.chk_leaves = QCheckBox("Leaf particles")
        self.chk_leaves.setChecked(DEFAULTS["show_leaves"])
        self.chk_leaves.setToolTip("Draw leaf particles at branch tips")
        self.chk_glow = QCheckBox("Glow effect")
        self.chk_glow.setChecked(DEFAULTS["show_glow"])
        self.chk_glow.setToolTip("Add glow effect at branch tips")

        layout.addRow("Size:", self.cb_size)
        layout.addRow("Palette:", self.cb_palette)
        layout.addRow("Easing:", self.cb_easing)
        layout.addRow("Perlin:", self.sp_perlin)
        layout.addRow("Rotation:", self.sp_rot)

        chk_layout = QVBoxLayout()
        chk_layout.setSpacing(4)
        for chk in [self.chk_grad, self.chk_swirl, self.chk_shimmer, self.chk_best,
                     self.chk_leaves, self.chk_glow]:
            chk_layout.addWidget(chk)
        layout.addRow(chk_layout)

        group.setLayout(layout)
        self._layout.addWidget(group)

    def _build_animation_group(self) -> None:
        group = QGroupBox("Animation")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Interval:"))
        self.sl_speed = QSlider(Qt.Horizontal)
        self.sl_speed.setRange(10, 200)
        self.sl_speed.setValue(DEFAULTS["anim_interval_ms"])
        self.lb_speed = QLabel(f"{DEFAULTS['anim_interval_ms']} ms")
        self.lb_speed.setMinimumWidth(48)
        self.sl_speed.valueChanged.connect(lambda v: self.lb_speed.setText(f"{v} ms"))
        self.sl_speed.setToolTip("Animation frame interval in milliseconds")
        speed_layout.addWidget(self.sl_speed)
        speed_layout.addWidget(self.lb_speed)
        layout.addLayout(speed_layout)

        btn_row1 = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setProperty("primary", "true")
        self.btn_start.setToolTip(f"Start evolution ({SHORTCUTS['start_pause']})")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip(f"Pause evolution ({SHORTCUTS['start_pause']})")
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setToolTip(f"Reset population ({SHORTCUTS['reset']})")
        self.btn_start.clicked.connect(self.sig_start)
        self.btn_pause.clicked.connect(self.sig_pause)
        self.btn_reset.clicked.connect(self.sig_reset)
        btn_row1.addWidget(self.btn_start)
        btn_row1.addWidget(self.btn_pause)
        btn_row1.addWidget(self.btn_reset)
        layout.addLayout(btn_row1)

        self.btn_step = QPushButton(f"⏭  Step Generation ({SHORTCUTS['step']})")
        self.btn_step.setToolTip("Advance one full generation")
        self.btn_step.clicked.connect(self.sig_step)
        layout.addWidget(self.btn_step)

        group.setLayout(layout)
        self._layout.addWidget(group)

    def _build_export_group(self) -> None:
        group = QGroupBox("Export")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        dir_layout = QHBoxLayout()
        self.le_dir = QLineEdit("fractal_genetic_art")
        self.le_dir.setPlaceholderText("Output directory…")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse)
        dir_layout.addWidget(self.le_dir)
        dir_layout.addWidget(btn_browse)
        layout.addLayout(dir_layout)

        export_layout = QHBoxLayout()
        self.btn_snap = QPushButton("📸 Screenshot")
        self.btn_snap.setToolTip(f"Save current frame ({SHORTCUTS['screenshot']})")
        self.btn_seq = QPushButton("🎬 Sequence")
        self.btn_seq.setToolTip(f"Export frame sequence ({SHORTCUTS['export_seq']})")
        self.btn_gif = QPushButton("🎞️ GIF")
        self.btn_gif.setToolTip(f"Export animated GIF ({SHORTCUTS['export_gif']})")
        self.btn_snap.clicked.connect(self.sig_screenshot)
        self.btn_seq.clicked.connect(self.sig_export_seq)
        self.btn_gif.clicked.connect(self.sig_export_gif)
        export_layout.addWidget(self.btn_snap)
        export_layout.addWidget(self.btn_seq)
        export_layout.addWidget(self.btn_gif)
        layout.addLayout(export_layout)

        save_layout = QHBoxLayout()
        self.btn_save = QPushButton(f"💾 Save ({SHORTCUTS['save']})")
        self.btn_load = QPushButton(f"📂 Load ({SHORTCUTS['load']})")
        self.btn_save.setToolTip("Save population state to JSON")
        self.btn_load.setToolTip("Load population state from JSON")
        self.btn_save.clicked.connect(self.sig_save)
        self.btn_load.clicked.connect(self.sig_load)
        save_layout.addWidget(self.btn_save)
        save_layout.addWidget(self.btn_load)
        layout.addLayout(save_layout)

        group.setLayout(layout)
        self._layout.addWidget(group)

    def _build_info_group(self) -> None:
        group = QGroupBox("Statistics")
        layout = QFormLayout()
        layout.setSpacing(6)
        layout.setLabelAlignment(Qt.AlignRight)

        self.lb_gen = QLabel("0")
        self.lb_frame = QLabel("0 / 0")
        self.lb_fit = QLabel("0.0")
        self.lb_avg_fit = QLabel("0.0")
        self.lb_mut = QLabel("0.20")
        self.lb_div = QLabel("0.0")
        self.lb_time = QLabel("0.0")

        for lbl in [self.lb_gen, self.lb_frame, self.lb_fit, self.lb_avg_fit,
                     self.lb_mut, self.lb_div, self.lb_time]:
            lbl.setStyleSheet(f"color: {C['text']}; font-weight: 600;")

        layout.addRow("Generation:", self.lb_gen)
        layout.addRow("Frame:", self.lb_frame)
        layout.addRow("Best fitness:", self.lb_fit)
        layout.addRow("Avg fitness:", self.lb_avg_fit)
        layout.addRow("Mutation rate:", self.lb_mut)
        layout.addRow("Diversity:", self.lb_div)
        layout.addRow("Time:", self.lb_time)

        group.setLayout(layout)
        self._layout.addWidget(group)

    def _build_chart(self) -> None:
        label = QLabel("Fitness History")
        label.setStyleSheet(f"font-weight: 600; color: {C['subtext1']}; padding-top: 4px;")
        self._layout.addWidget(label)
        self.chart = FitnessChart()
        self._layout.addWidget(self.chart)

    # ── Helpers ──

    def _spin(self, mn, mx, default) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(mn, mx)
        sb.setValue(default)
        return sb

    def _dspin(self, mn, mx, default, step, decimals) -> QDoubleSpinBox:
        dsb = QDoubleSpinBox()
        dsb.setRange(mn, mx)
        dsb.setSingleStep(step)
        dsb.setDecimals(decimals)
        dsb.setValue(default)
        return dsb

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.le_dir.setText(d)

    # ── Properties ──

    @property
    def image_size(self) -> Tuple[int, int]:
        return IMAGE_SIZES[self.cb_size.currentText()]

    @property
    def ga_params(self) -> dict:
        seed = self.sp_seed.value()
        return dict(
            image_size=self.image_size,
            pop_size=self.sp_pop.value(),
            num_fractals=self.sp_fract.value(),
            mutation_rate=self.sp_mut.value(),
            frames_per_gen=self.sp_fpg.value(),
            seed=seed if seed != 0 else None,
            adaptive_mutation=self.chk_adaptive.isChecked(),
        )

    @property
    def interval(self) -> int:
        return self.sl_speed.value()

    @property
    def export_dir(self) -> str:
        return self.le_dir.text()

    @property
    def current_palette(self) -> str:
        return self.cb_palette.currentText()

    @property
    def current_easing(self) -> str:
        return self.cb_easing.currentText()

    # ── Public ──

    def refresh_info(self, ga: GeneticAlgorithm) -> None:
        stats = ga.get_stats()
        self.lb_gen.setText(str(stats["generation"]))
        self.lb_frame.setText(f"{stats['frame']} / {stats['frames_per_gen']}")
        self.lb_fit.setText(f"{stats['best_fitness']:.1f}")
        self.lb_avg_fit.setText(f"{stats['avg_fitness']:.1f}")
        self.lb_mut.setText(f"{stats['mutation_rate']:.2f}")
        self.lb_div.setText(f"{stats['diversity']:.1f}")
        self.lb_time.setText(f"{stats['global_time']:.1f}")
        self.chart.set_history(ga.fitness_history, ga.avg_fitness_history, ga.diversity_history)

    def set_running(self, on: bool) -> None:
        self.btn_start.setEnabled(not on)
        self.btn_pause.setEnabled(on)
        for w in (self.sp_pop, self.sp_fract, self.cb_size, self.sp_seed, self.chk_adaptive):
            w.setEnabled(not on)