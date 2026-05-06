"""All custom widgets: canvas, fitness chart, control panel, and status bar."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QColor, QLinearGradient,
    QPainterPath, QFont, QBrush, QRadialGradient,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QSlider, QLabel, QLineEdit, QScrollArea, QFileDialog,
    QSizePolicy, QFrame,
)

from config import DEFAULTS, IMAGE_SIZES, C
from genetics import GeneticAlgorithm, Gene, Individual
from renderers import draw_fractal_qp, draw_gradient_qp


# ─── Fractal Canvas ───────────────────────────────────────────────────────────

class FractalCanvas(QWidget):
    """Central painting surface rendered via QPainter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ga: Optional[GeneticAlgorithm] = None
        self.perlin_scale = DEFAULTS["perlin_scale"]
        self.rotation_speed = DEFAULTS["rotation_speed"]
        self.show_gradient = True
        self.show_swirl = True
        self.shimmer = True
        self.highlight_best = False
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

        # Background
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

        # Draw canvas background
        self._draw_canvas_bg(p, w, h)

        # Draw fractals
        self._draw_fractals(p, w, h)

        p.restore()

        # Draw border overlay
        self._draw_border(p, ox, oy, w * scale, h * scale)

        p.end()

    def _draw_placeholder(self, p: QPainter) -> None:
        """Draw placeholder when no data is available."""
        p.setPen(QColor(C["overlay0"]))
        font = QFont("Segoe UI", 14)
        font.setItalic(True)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignCenter, "Press Start or Reset to begin")

    def _draw_canvas_bg(self, p: QPainter, w: int, h: int) -> None:
        """Draw the canvas background (gradient or solid)."""
        if self.show_gradient:
            avg = self._get_average_colour()
            draw_gradient_qp(p, (w, h), avg * 0.8, avg * 0.1)
        else:
            p.fillRect(0, 0, w, h, QColor(0, 0, 0))

    def _draw_fractals(self, p: QPainter, w: int, h: int) -> None:
        """Draw all fractal trees."""
        pop = self._interp_pop
        n = len(pop)

        if self.highlight_best:
            indices = [0]
        else:
            indices = list(range(n))

        for rank, i in enumerate(indices):
            swirl_angle = (
                self._gt * self.rotation_speed * (i + 1)
                if self.show_swirl else 0.0
            )
            if self.highlight_best:
                alpha = 1.0
            else:
                alpha = 0.12 + 0.88 * (i / max(1, n - 1))

            for gene in pop[i]:
                draw_fractal_qp(
                    p, gene, self._gt, (w, h),
                    alpha=alpha,
                    swirl=swirl_angle,
                    perlin_scale=self.perlin_scale,
                    shimmer=self.shimmer,
                )

    def _get_average_colour(self) -> np.ndarray:
        """Get average colour from current population."""
        from color_utils import average_colour
        return average_colour(self.ga.population, self._gt)

    def _draw_border(self, p: QPainter, x: float, y: float, w: float, h: float) -> None:
        """Draw a subtle border around the canvas area."""
        pen = QPen(QColor(C["surface1"]), 1)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(x - 2, y - 2, w + 4, h + 4), 4, 4)


# ─── Fitness Chart ────────────────────────────────────────────────────────────

class FitnessChart(QWidget):
    """Sparkline chart showing fitness history with improved styling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.history: List[float] = []
        self.avg_history: List[float] = []
        self.setMinimumHeight(140)
        self.setMaximumHeight(220)

    def set_history(self, history: List[float], avg_history: List[float] | None = None) -> None:
        self.history = list(history)
        self.avg_history = list(avg_history) if avg_history else []
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        margin_left = 45
        margin_bottom = 24
        margin_top = 12
        margin_right = 12

        # Background
        bg = QRadialGradient(w / 2, h / 2, w)
        bg.setColorAt(0, QColor(C["surface0"]))
        bg.setColorAt(1, QColor(C["mantle"]))
        p.fillRect(0, 0, w, h, bg)

        # Border
        p.setPen(QPen(QColor(C["surface1"]), 1))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 6, 6)

        if len(self.history) < 2:
            p.setPen(QColor(C["overlay0"]))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "Waiting for data…")
            p.end()
            return

        # Compute plot area
        pw = w - margin_left - margin_right
        ph = h - margin_top - margin_bottom

        mn = min(self.history)
        mx = max(self.history)
        rng = (mx - mn) if mx != mn else 1.0

        # Grid lines
        p.setPen(QPen(QColor(C["surface1"]), 1, Qt.DotLine))
        p.setFont(QFont("Segoe UI", 7))

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = int(margin_top + (1.0 - frac) * ph)
            p.drawLine(margin_left, yy, w - margin_right, yy)
            val = mn + frac * rng
            p.setPen(QColor(C["overlay0"]))
            p.drawText(4, yy + 4, f"{val:.0f}")
            p.setPen(QPen(QColor(C["surface1"]), 1, Qt.DotLine))

        # Axes
        p.setPen(QPen(QColor(C["surface2"]), 1))
        p.drawLine(margin_left, margin_top, margin_left, h - margin_bottom)
        p.drawLine(margin_left, h - margin_bottom, w - margin_right, h - margin_bottom)

        # Average fitness line (if available)
        if len(self.avg_history) >= 2:
            self._draw_line(
                p, self.avg_history, mn, rng,
                margin_left, margin_top, pw, ph,
                QColor(C["mauve"], 120), QColor(C["mauve"], 20),
                line_width=1.5,
            )

        # Best fitness line
        self._draw_line(
            p, self.history, mn, rng,
            margin_left, margin_top, pw, ph,
            QColor(C["blue"]), QColor(C["blue"], 40),
            line_width=2.0,
        )

        # End point dot
        n = len(self.history)
        lx = margin_left + pw
        ly = margin_top + (1.0 - (self.history[-1] - mn) / rng) * ph

        # Glow effect
        glow = QRadialGradient(lx, ly, 8)
        glow.setColorAt(0, QColor(C["rosewater"], 80))
        glow.setColorAt(1, QColor(C["rosewater"], 0))
        p.setBrush(glow)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(lx, ly), 8, 8)

        # Dot
        p.setBrush(QColor(C["rosewater"]))
        p.setPen(QPen(QColor(C["text"]), 1))
        p.drawEllipse(QPointF(lx, ly), 4, 4)

        # X-axis labels
        p.setPen(QColor(C["overlay0"]))
        p.drawText(margin_left, h - 4, "0")
        p.drawText(w - margin_right - 25, h - 4, f"{n - 1}")

        # Title
        p.setPen(QColor(C["subtext1"]))
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(margin_left + 4, margin_top + 10, "Fitness")

        p.end()

    def _draw_line(
        self,
        p: QPainter,
        data: List[float],
        mn: float,
        rng: float,
        mx: float,
        my: float,
        pw: float,
        ph: float,
        line_color: QColor,
        fill_color: QColor,
        line_width: float = 2.0,
    ) -> None:
        """Draw a line with gradient fill beneath it."""
        n = len(data)

        # Build path
        path = QPainterPath()
        for i, val in enumerate(data):
            x = mx + (i / max(1, n - 1)) * pw
            y = my + (1.0 - (val - mn) / rng) * ph
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        # Fill area under curve
        fill = QPainterPath(path)
        fill.lineTo(mx + pw, my + ph)
        fill.lineTo(mx, my + ph)
        fill.closeSubpath()

        gradient = QLinearGradient(0, my, 0, my + ph)
        gradient.setColorAt(0, fill_color)
        gradient.setColorAt(1, QColor(fill_color.red(), fill_color.green(), fill_color.blue(), 0))
        p.fillPath(fill, gradient)

        # Draw line
        p.setPen(QPen(line_color, line_width))
        p.drawPath(path)


# ─── Status Bar Widget ────────────────────────────────────────────────────────

class StatusBarWidget(QWidget):
    """Custom status bar widget showing GA state."""

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
            ("fitness", "Fitness: 0.0", C["green"]),
            ("pop", "Pop: 0", C["lavender"]),
            ("fps", "FPS: 0", C["yellow"]),
        ]

        for key, text, color in items:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {color}; font-weight: 600;")
            layout.addWidget(lbl)
            self._labels[key] = lbl

        layout.addStretch()

    def _set_label(self, key: str, value: str) -> None:
        if key in self._labels:
            self._labels[key].setText(value)

    @property
    def state(self) -> str:
        return self._labels["state"].text()

    @state.setter
    def state(self, value: str) -> None:
        color = C["green"] if value == "Running" else C["yellow"]
        icon = "●" if value == "Running" else "○"
        self._labels["state"].setText(f"{icon} {value}")
        self._labels["state"].setStyleSheet(f"color: {color}; font-weight: 600;")

    @property
    def gen(self) -> int:
        return int(self._labels["gen"].text().split(":")[1].strip())

    @gen.setter
    def gen(self, value: int) -> None:
        self._set_label("gen", f"Gen: {value}")

    @property
    def frame(self) -> str:
        return self._labels["frame"].text()

    @frame.setter
    def frame(self, value: str) -> None:
        self._set_label("frame", f"Frame: {value}")

    @property
    def fitness(self) -> str:
        return self._labels["fitness"].text()

    @fitness.setter
    def fitness(self, value: str) -> None:
        self._set_label("fitness", f"Fitness: {value}")

    @property
    def pop_size(self) -> int:
        return int(self._labels["pop"].text().split(":")[1].strip())

    @pop_size.setter
    def pop_size(self, value: int) -> None:
        self._set_label("pop", f"Pop: {value}")

    @property
    def fps(self) -> int:
        return int(self._labels["fps"].text().split(":")[1].strip())

    @fps.setter
    def fps(self, value: int) -> None:
        self._set_label("fps", f"FPS: {value}")


# ─── Control Panel ────────────────────────────────────────────────────────────

class ControlPanel(QScrollArea):
    """Sidebar with all tunables, buttons, and info readouts."""

    sig_start = Signal()
    sig_pause = Signal()
    sig_reset = Signal()
    sig_step = Signal()
    sig_screenshot = Signal()
    sig_export_seq = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedWidth(320)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        root = QWidget()
        self._layout = QVBoxLayout(root)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(8, 8, 8, 8)

        # ── Title ──
        self._build_title()
        # ── Evolution group ──
        self._build_evolution_group()
        # ── Visual group ──
        self._build_visual_group()
        # ── Animation group ──
        self._build_animation_group()
        # ── Export group ──
        self._build_export_group()
        # ── Info group ──
        self._build_info_group()
        # ── Chart ──
        self._build_chart()

        self._layout.addStretch()
        self.setWidget(root)

    def _build_title(self) -> None:
        title = QLabel("🧬 Fractal Genetic Art")
        title.setStyleSheet(
            f"font-size:16px; font-weight:700; color:{C['sky']}; "
            f"padding:8px 0 4px 0;"
        )
        self._layout.addWidget(title)

        subtitle = QLabel("Evolutionary fractal tree generator")
        subtitle.setStyleSheet(
            f"font-size:11px; color:{C['overlay0']}; padding-bottom:4px;"
        )
        self._layout.addWidget(subtitle)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {C['surface1']};")
        self._layout.addWidget(sep)

    def _build_evolution_group(self) -> None:
        group = QGroupBox("Evolution")
        layout = QFormLayout()
        layout.setSpacing(8)
        layout.setLabelAlignment(Qt.AlignRight)

        self.sp_pop = self._create_spinbox(2, 30, DEFAULTS["population_size"])
        self.sp_fract = self._create_spinbox(1, 10, DEFAULTS["num_fractals"])
        self.sp_mut = self._create_doublespinbox(0.0, 1.0, DEFAULTS["mutation_rate"], 0.05, 2)
        self.sp_fpg = self._create_spinbox(1, 100, DEFAULTS["frames_per_gen"])

        layout.addRow("Population:", self.sp_pop)
        layout.addRow("Trees/ind:", self.sp_fract)
        layout.addRow("Mutation:", self.sp_mut)
        layout.addRow("Frames/gen:", self.sp_fpg)

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

        self.sp_perlin = self._create_doublespinbox(0.0, 0.5, DEFAULTS["perlin_scale"], 0.01, 3)
        self.sp_rot = self._create_doublespinbox(0.0, 0.2, DEFAULTS["rotation_speed"], 0.005, 3)

        self.chk_grad = QCheckBox("Gradient background")
        self.chk_grad.setChecked(True)
        self.chk_swirl = QCheckBox("Swirl effect")
        self.chk_swirl.setChecked(True)
        self.chk_shimmer = QCheckBox("Shimmer mode")
        self.chk_shimmer.setChecked(True)
        self.chk_best = QCheckBox("Show best only")
        self.chk_best.setChecked(False)

        layout.addRow("Size:", self.cb_size)
        layout.addRow("Perlin:", self.sp_perlin)
        layout.addRow("Rotation:", self.sp_rot)

        # Checkboxes in a sub-layout
        chk_layout = QVBoxLayout()
        chk_layout.setSpacing(4)
        for chk in [self.chk_grad, self.chk_swirl, self.chk_shimmer, self.chk_best]:
            chk_layout.addWidget(chk)
        layout.addRow(chk_layout)

        group.setLayout(layout)
        self._layout.addWidget(group)

    def _build_animation_group(self) -> None:
        group = QGroupBox("Animation")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Interval:"))
        self.sl_speed = QSlider(Qt.Horizontal)
        self.sl_speed.setRange(10, 200)
        self.sl_speed.setValue(DEFAULTS["anim_interval_ms"])
        self.lb_speed = QLabel(f"{DEFAULTS['anim_interval_ms']} ms")
        self.lb_speed.setMinimumWidth(48)
        self.sl_speed.valueChanged.connect(lambda v: self.lb_speed.setText(f"{v} ms"))
        speed_layout.addWidget(self.sl_speed)
        speed_layout.addWidget(self.lb_speed)
        layout.addLayout(speed_layout)

        # Control buttons
        btn_row1 = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setProperty("primary", "true")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_start.clicked.connect(self.sig_start)
        self.btn_pause.clicked.connect(self.sig_pause)
        self.btn_reset.clicked.connect(self.sig_reset)
        btn_row1.addWidget(self.btn_start)
        btn_row1.addWidget(self.btn_pause)
        btn_row1.addWidget(self.btn_reset)
        layout.addLayout(btn_row1)

        self.btn_step = QPushButton("⏭  Step Generation")
        self.btn_step.clicked.connect(self.sig_step)
        layout.addWidget(self.btn_step)

        group.setLayout(layout)
        self._layout.addWidget(group)

    def _build_export_group(self) -> None:
        group = QGroupBox("Export")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Directory selector
        dir_layout = QHBoxLayout()
        self.le_dir = QLineEdit("fractal_genetic_art")
        self.le_dir.setPlaceholderText("Output directory…")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse)
        dir_layout.addWidget(self.le_dir)
        dir_layout.addWidget(btn_browse)
        layout.addLayout(dir_layout)

        # Export buttons
        export_layout = QHBoxLayout()
        self.btn_snap = QPushButton("📸 Screenshot")
        self.btn_seq = QPushButton("🎬 Sequence")
        self.btn_snap.clicked.connect(self.sig_screenshot)
        self.btn_seq.clicked.connect(self.sig_export_seq)
        export_layout.addWidget(self.btn_snap)
        export_layout.addWidget(self.btn_seq)
        layout.addLayout(export_layout)

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
        self.lb_time = QLabel("0.0")

        # Style values
        for lbl in [self.lb_gen, self.lb_frame, self.lb_fit, self.lb_avg_fit, self.lb_time]:
            lbl.setStyleSheet(f"color: {C['text']}; font-weight: 600;")

        layout.addRow("Generation:", self.lb_gen)
        layout.addRow("Frame:", self.lb_frame)
        layout.addRow("Best fitness:", self.lb_fit)
        layout.addRow("Avg fitness:", self.lb_avg_fit)
        layout.addRow("Time:", self.lb_time)

        group.setLayout(layout)
        self._layout.addWidget(group)

    def _build_chart(self) -> None:
        label = QLabel("Fitness History")
        label.setStyleSheet(f"font-weight: 600; color: {C['subtext1']}; padding-top: 4px;")
        self._layout.addWidget(label)

        self.chart = FitnessChart()
        self._layout.addWidget(self.chart)

    # ── Helper methods ──

    def _create_spinbox(self, min_val: int, max_val: int, default: int) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(default)
        return sb

    def _create_doublespinbox(
        self, min_val: float, max_val: float, default: float,
        step: float, decimals: int
    ) -> QDoubleSpinBox:
        dsb = QDoubleSpinBox()
        dsb.setRange(min_val, max_val)
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
        return dict(
            image_size=self.image_size,
            pop_size=self.sp_pop.value(),
            num_fractals=self.sp_fract.value(),
            mutation_rate=self.sp_mut.value(),
            frames_per_gen=self.sp_fpg.value(),
        )

    @property
    def interval(self) -> int:
        return self.sl_speed.value()

    @property
    def export_dir(self) -> str:
        return self.le_dir.text()

    # ── Public methods ──

    def refresh_info(self, ga: GeneticAlgorithm) -> None:
        """Update all info labels from GA state."""
        self.lb_gen.setText(str(ga.generation))
        self.lb_frame.setText(f"{ga.frame_idx} / {ga.frames_per_gen}")
        self.lb_fit.setText(f"{ga.best_fitness:.1f}")
        self.lb_avg_fit.setText(f"{ga.avg_fitness:.1f}")
        self.lb_time.setText(f"{ga.global_time:.1f}")
        self.chart.set_history(ga.fitness_history, ga.avg_fitness_history)

    def set_running(self, on: bool) -> None:
        """Enable/disable controls based on running state."""
        self.btn_start.setEnabled(not on)
        self.btn_pause.setEnabled(on)
        # Disable evolution params while running
        for w in (self.sp_pop, self.sp_fract, self.cb_size):
            w.setEnabled(not on)