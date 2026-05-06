"""All custom widgets: canvas, fitness chart, and control panel."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QLinearGradient, QPainterPath, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QSlider, QLabel, QLineEdit, QScrollArea, QFileDialog,
    QMessageBox, QProgressDialog, QSizePolicy,
)

from colors import harmonic_color
from config import DEFAULTS, IMAGE_SIZES, DARK_STYLE
from genetics import GeneticAlgorithm, Gene
from renderer_qt import draw_fractal_qp, draw_gradient_qp
from renderer_pil import render_individual_pil, average_colour


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
        self._interp_pop: List[List[Gene]] = []
        self._gt = 0.0
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_ga(self, ga: GeneticAlgorithm) -> None:
        self.ga = ga

    def update_frame(self, interp_pop: List[List[Gene]],
                     global_time: float) -> None:
        self._interp_pop = interp_pop
        self._gt = global_time
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if self.ga is None or not self._interp_pop:
            p.fillRect(self.rect(), QColor(0, 0, 0))
            p.setPen(QColor(80, 80, 80))
            p.drawText(self.rect(), Qt.AlignCenter, "Press Start or Reset")
            p.end()
            return

        w, h = self.ga.image_size
        scale = min(self.width() / w, self.height() / h)
        ox = (self.width() - w * scale) / 2.0
        oy = (self.height() - h * scale) / 2.0

        p.save()
        p.translate(ox, oy)
        p.scale(scale, scale)

        # background
        if self.show_gradient:
            avg = average_colour(self.ga.population, self._gt)
            draw_gradient_qp(p, (w, h), avg * 0.9, avg * 0.15)
        else:
            p.fillRect(0, 0, w, h, QColor(0, 0, 0))

        # fractals
        pop = self._interp_pop
        n = len(pop)
        indices = [0] if self.highlight_best else range(n)

        for i in indices:
            swirl_a = (self._gt * self.rotation_speed * (i + 1)
                       if self.show_swirl else 0.0)
            alpha = 1.0 if self.highlight_best else (0.15 + 0.85 * i / max(1, n - 1))
            for gene in pop[i]:
                draw_fractal_qp(p, gene, self._gt, (w, h),
                                alpha=alpha, swirl=swirl_a,
                                perlin_scale=self.perlin_scale,
                                shimmer=self.shimmer)

        p.restore()
        p.end()


# ─── Fitness Chart ────────────────────────────────────────────────────────────
class FitnessChart(QWidget):
    """Minimal sparkline chart of best fitness per generation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.history: List[float] = []
        self.setMinimumHeight(130)
        self.setMaximumHeight(200)

    def set_history(self, history: List[float]) -> None:
        self.history = list(history)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        m = 35

        p.fillRect(0, 0, w, h, QColor(24, 24, 37))
        p.setPen(QPen(QColor(69, 71, 90), 1))
        p.drawRect(0, 0, w - 1, h - 1)

        if len(self.history) < 2:
            p.setPen(QColor(150, 150, 150))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "Waiting for data…")
            p.end()
            return

        p.setPen(QPen(QColor(108, 112, 134), 1))
        p.drawLine(m, h - m, w - 10, h - m)
        p.drawLine(m, 10, m, h - m)

        mn, mx = min(self.history), max(self.history)
        rng = (mx - mn) if mx != mn else 1.0
        pw, ph = w - m - 10, h - m - 10

        p.setFont(QFont("Segoe UI", 7))
        p.setPen(QColor(108, 112, 134))
        for frac in (0.0, 0.5, 1.0):
            yy = int(h - m - frac * ph)
            p.drawLine(m, yy, w - 10, yy)
            p.drawText(2, yy - 2, f"{mn + frac * rng:.0f}")

        path = QPainterPath()
        n = len(self.history)
        for i, f in enumerate(self.history):
            x = m + (i / max(1, n - 1)) * pw
            y = h - m - ((f - mn) / rng) * ph
            (path.moveTo if i == 0 else path.lineTo)(x, y)

        fill = QPainterPath(path)
        fill.lineTo(m + pw, h - m)
        fill.lineTo(m, h - m)
        fill.closeSubpath()
        fg = QLinearGradient(0, 10, 0, h - m)
        fg.setColorAt(0, QColor(137, 180, 250, 60))
        fg.setColorAt(1, QColor(137, 180, 250, 5))
        p.fillPath(fill, fg)

        p.setPen(QPen(QColor(137, 180, 250), 2))
        p.drawPath(path)

        lx = m + pw
        ly = h - m - ((self.history[-1] - mn) / rng) * ph
        p.setBrush(QColor(243, 139, 168))
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawEllipse(QPointF(lx, ly), 4, 4)

        p.setPen(QColor(166, 173, 200))
        p.drawText(m, h - 2, "Gen 0")
        p.drawText(w - 40, h - 2, f"Gen {n - 1}")
        p.end()


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
        self.setFixedWidth(310)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setSpacing(6)
        lay.setContentsMargins(6, 6, 6, 6)

        title = QLabel("🧬 Fractal Genetic Art")
        title.setStyleSheet(
            "font-size:15px;font-weight:bold;color:#89dceb;padding:4px 0;")
        lay.addWidget(title)

        # ── Evolution ──
        eg = QGroupBox("Evolution")
        ef = QFormLayout()
        self.sp_pop = QSpinBox()
        self.sp_pop.setRange(1, 30)
        self.sp_pop.setValue(DEFAULTS["population_size"])
        self.sp_fract = QSpinBox()
        self.sp_fract.setRange(1, 10)
        self.sp_fract.setValue(DEFAULTS["num_fractals"])
        self.sp_mut = QDoubleSpinBox()
        self.sp_mut.setRange(0.0, 1.0)
        self.sp_mut.setSingleStep(0.05)
        self.sp_mut.setValue(DEFAULTS["mutation_rate"])
        self.sp_fpg = QSpinBox()
        self.sp_fpg.setRange(1, 100)
        self.sp_fpg.setValue(DEFAULTS["frames_per_gen"])
        ef.addRow("Population:", self.sp_pop)
        ef.addRow("Trees / ind:", self.sp_fract)
        ef.addRow("Mutation rate:", self.sp_mut)
        ef.addRow("Frames / gen:", self.sp_fpg)
        eg.setLayout(ef)
        lay.addWidget(eg)

        # ── Visual ──
        vg = QGroupBox("Visual")
        vf = QFormLayout()
        self.cb_size = QComboBox()
        self.cb_size.addItems(list(IMAGE_SIZES.keys()))
        self.cb_size.setCurrentText(DEFAULTS["image_size_key"])
        self.sp_perlin = QDoubleSpinBox()
        self.sp_perlin.setRange(0.0, 0.5)
        self.sp_perlin.setSingleStep(0.01)
        self.sp_perlin.setDecimals(3)
        self.sp_perlin.setValue(DEFAULTS["perlin_scale"])
        self.sp_rot = QDoubleSpinBox()
        self.sp_rot.setRange(0.0, 0.2)
        self.sp_rot.setSingleStep(0.005)
        self.sp_rot.setDecimals(3)
        self.sp_rot.setValue(DEFAULTS["rotation_speed"])
        self.chk_grad = QCheckBox("Gradient bg")
        self.chk_grad.setChecked(True)
        self.chk_swirl = QCheckBox("Swirling")
        self.chk_swirl.setChecked(True)
        self.chk_shimmer = QCheckBox("Shimmer mode")
        self.chk_shimmer.setChecked(True)
        self.chk_best = QCheckBox("Highlight best only")
        self.chk_best.setChecked(False)
        vf.addRow("Size:", self.cb_size)
        vf.addRow("Perlin scale:", self.sp_perlin)
        vf.addRow("Rotation speed:", self.sp_rot)
        vf.addRow(self.chk_grad)
        vf.addRow(self.chk_swirl)
        vf.addRow(self.chk_shimmer)
        vf.addRow(self.chk_best)
        vg.setLayout(vf)
        lay.addWidget(vg)

        # ── Animation ──
        ag = QGroupBox("Animation")
        al = QVBoxLayout()
        sl = QHBoxLayout()
        sl.addWidget(QLabel("Interval:"))
        self.sl_speed = QSlider(Qt.Horizontal)
        self.sl_speed.setRange(10, 200)
        self.sl_speed.setValue(DEFAULTS["anim_interval_ms"])
        self.lb_speed = QLabel(f"{DEFAULTS['anim_interval_ms']} ms")
        self.sl_speed.valueChanged.connect(
            lambda v: self.lb_speed.setText(f"{v} ms"))
        sl.addWidget(self.sl_speed)
        sl.addWidget(self.lb_speed)
        al.addLayout(sl)

        bl = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_start.clicked.connect(self.sig_start)
        self.btn_pause.clicked.connect(self.sig_pause)
        self.btn_reset.clicked.connect(self.sig_reset)
        bl.addWidget(self.btn_start)
        bl.addWidget(self.btn_pause)
        bl.addWidget(self.btn_reset)
        al.addLayout(bl)

        self.btn_step = QPushButton("⏭  Step one generation")
        self.btn_step.clicked.connect(self.sig_step)
        al.addWidget(self.btn_step)
        ag.setLayout(al)
        lay.addWidget(ag)

        # ── Export ──
        xg = QGroupBox("Export")
        xl = QVBoxLayout()
        dl = QHBoxLayout()
        self.le_dir = QLineEdit("fractal_genetic_art")
        self.btn_browse = QPushButton("…")
        self.btn_browse.setFixedWidth(28)
        self.btn_browse.clicked.connect(self._browse)
        dl.addWidget(self.le_dir)
        dl.addWidget(self.btn_browse)
        xl.addLayout(dl)
        el = QHBoxLayout()
        self.btn_snap = QPushButton("📸 Screenshot")
        self.btn_snap.clicked.connect(self.sig_screenshot)
        self.btn_seq = QPushButton("🎬 Sequence")
        self.btn_seq.clicked.connect(self.sig_export_seq)
        el.addWidget(self.btn_snap)
        el.addWidget(self.btn_seq)
        xl.addLayout(el)
        xg.setLayout(xl)
        lay.addWidget(xg)

        # ── Info ──
        ig = QGroupBox("Info")
        il = QFormLayout()
        self.lb_gen = QLabel("0")
        self.lb_frame = QLabel("0 / 0")
        self.lb_fit = QLabel("0.0")
        self.lb_time = QLabel("0.0")
        il.addRow("Generation:", self.lb_gen)
        il.addRow("Frame:", self.lb_frame)
        il.addRow("Best fitness:", self.lb_fit)
        il.addRow("Time:", self.lb_time)
        ig.setLayout(il)
        lay.addWidget(ig)

        lay.addWidget(QLabel("Fitness History"))
        self.chart = FitnessChart()
        lay.addWidget(self.chart)
        lay.addStretch()
        self.setWidget(root)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output directory")
        if d:
            self.le_dir.setText(d)

    # ── convenience properties ──
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

    def refresh_info(self, ga: GeneticAlgorithm) -> None:
        self.lb_gen.setText(str(ga.generation))
        self.lb_frame.setText(f"{ga.frame_idx} / {ga.frames_per_gen}")
        self.lb_fit.setText(f"{ga.best_fitness:.1f}")
        self.lb_time.setText(f"{ga.global_time:.1f}")
        self.chart.set_history(ga.fitness_history)

    def set_running(self, on: bool) -> None:
        self.btn_start.setEnabled(not on)
        self.btn_pause.setEnabled(on)
        for w in (self.sp_pop, self.sp_fract, self.cb_size):
            w.setEnabled(not on)