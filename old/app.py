#!/usr/bin/env python3
"""
Fractal Genetic Art Studio
Combines genetic algorithms with fractal tree rendering, Perlin noise perturbation,
swirling animations, smooth interpolation, gradient backgrounds, and real-time evolution.
"""

import sys
import os
import math
import random
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QSlider, QLabel, QLineEdit, QScrollArea,
    QFileDialog, QMessageBox, QProgressDialog, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal, QPointF
from PySide6.QtGui import (
    QPainter, QPen, QColor, QLinearGradient, QPainterPath, QFont
)
from PIL import Image, ImageDraw

# ─── Perlin noise with built-in fallback ──────────────────────────────────────
try:
    from noise import pnoise1
except ImportError:
    _perm = list(range(256))
    random.shuffle(_perm)
    _perm = _perm + _perm
    _grad = [random.uniform(-1.0, 1.0) for _ in range(512)]

    def _fade(t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _lerp(a, b, t):
        return a + t * (b - a)

    def pnoise1(x, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024, base=0):
        x = x % repeat
        xi = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = _fade(xf)
        a = _perm[xi]
        b = _perm[xi + 1]
        return _lerp(_grad[a], _grad[b], u)


# ─── Utility functions ────────────────────────────────────────────────────────
def harmonic_color(index, total, t_shift=0.0):
    hue = (index / total + t_shift) % 1.0
    r = (math.sin(2.0 * math.pi * hue) + 1.0) / 2.0
    g = (math.sin(2.0 * math.pi * hue + 2.0 * math.pi / 3.0) + 1.0) / 2.0
    b = (math.sin(2.0 * math.pi * hue + 4.0 * math.pi / 3.0) + 1.0) / 2.0
    return np.clip([r, g, b], 0.0, 1.0)


def _gene_seed(gene):
    parts = []
    for k, v in sorted(gene.items()):
        parts.append(int(v * 10000) if isinstance(v, float) else int(v))
    return hash(tuple(parts))


def interpolate_gene(g1, g2, t):
    result = {}
    for key in g1:
        if isinstance(g1[key], float):
            result[key] = g1[key] * (1.0 - t) + g2[key] * t
        else:
            result[key] = int(g1[key] * (1.0 - t) + g2[key] * t)
    return result


def interpolate_individual(ind1, ind2, t):
    return [interpolate_gene(g1, g2, t) for g1, g2 in zip(ind1, ind2)]


# ─── Genetic Algorithm ────────────────────────────────────────────────────────
class GeneticAlgorithm:
    def __init__(self, image_size=(600, 600), pop_size=5, num_fractals=3,
                 mutation_rate=0.2, frames_per_gen=20):
        self.image_size = image_size
        self.pop_size = max(1, pop_size)
        self.num_fractals = max(1, num_fractals)
        self.mutation_rate = mutation_rate
        self.frames_per_gen = max(1, frames_per_gen)
        self.population = []
        self.next_population = []
        self.generation = 0
        self.frame_idx = 0
        self.global_time = 0.0
        self.best_fitness = 0.0
        self.fitness_history = []
        self.initialize()

    # ── Gene helpers ──
    def random_gene(self):
        w, h = self.image_size
        return {
            'x': random.randint(w // 4, 3 * w // 4),
            'y': h,
            'length': random.randint(80, 160),
            'angle': -math.pi / 2 + random.uniform(-0.3, 0.3),
            'depth': random.randint(3, 6),
            'branch_factor': random.randint(2, 4),
            'color_offset': random.random(),
            'curvature': random.uniform(-0.2, 0.2),
        }

    def create_individual(self):
        return [self.random_gene() for _ in range(self.num_fractals)]

    # ── Fitness (spread + branch complexity + colour variety) ──
    def fitness(self, individual):
        xs = [g['x'] for g in individual]
        spread = (max(xs) - min(xs)) if xs else 0
        branch_score = sum(g['branch_factor'] * g['depth'] for g in individual)
        color_var = sum(g['color_offset'] for g in individual)
        return spread + branch_score + color_var * 50.0

    # ── Selection (roulette) ──
    def select_parents(self, pop=None):
        pop = pop or self.population
        if len(pop) < 2:
            return pop[0], pop[0]
        fitnesses = np.array([self.fitness(ind) for ind in pop], dtype=np.float64)
        total = fitnesses.sum()
        probs = (fitnesses / total) if total > 0 else np.ones(len(pop)) / len(pop)
        idx1, idx2 = np.random.choice(len(pop), size=2, replace=False, p=probs)
        return pop[idx1], pop[idx2]

    # ── Genetic operators ──
    def mutate(self, individual):
        new_ind = []
        for gene in individual:
            ng = gene.copy()
            if random.random() < self.mutation_rate:
                trait = random.choice(list(gene.keys()))
                if trait in ('x', 'y'):
                    ng[trait] = max(0, gene[trait] + random.randint(-10, 10))
                elif trait == 'depth':
                    ng[trait] = max(1, min(8, gene[trait] + random.randint(-1, 1)))
                elif trait == 'branch_factor':
                    ng[trait] = max(1, min(5, gene[trait] + random.randint(-1, 1)))
                elif trait == 'length':
                    ng[trait] = max(20, gene[trait] + random.randint(-15, 15))
                else:
                    ng[trait] = gene[trait] + random.uniform(-0.1, 0.1)
            new_ind.append(ng)
        return new_ind

    def crossover(self, p1, p2):
        child = []
        for g1, g2 in zip(p1, p2):
            child.append({k: (g1[k] if random.random() < 0.5 else g2[k]) for k in g1})
        return child

    # ── Lifecycle ──
    def initialize(self):
        self.population = [self.create_individual() for _ in range(self.pop_size)]
        self.next_population = []
        for _ in range(self.pop_size):
            p1, p2 = self.select_parents()
            self.next_population.append(self.mutate(self.crossover(p1, p2)))
        self.generation = 0
        self.frame_idx = 0
        self.global_time = 0.0
        self.best_fitness = max(self.fitness(ind) for ind in self.population)
        self.fitness_history = [self.best_fitness]

    def get_interpolated_population(self):
        t = self.frame_idx / max(1, self.frames_per_gen)
        return [interpolate_individual(ind, nxt, t)
                for ind, nxt in zip(self.population, self.next_population)]

    def step(self):
        self.frame_idx += 1
        self.global_time += 0.1
        if self.frame_idx >= self.frames_per_gen:
            self._evolve()
            return True
        return False

    def _evolve(self):
        self.population = self.next_population[:]
        self.population.sort(key=self.fitness, reverse=True)
        self.best_fitness = self.fitness(self.population[0])
        self.fitness_history.append(self.best_fitness)
        # elitism + breed
        new_next = self.population[:max(1, self.pop_size // 3)]
        while len(new_next) < self.pop_size:
            p1, p2 = self.select_parents()
            new_next.append(self.mutate(self.crossover(p1, p2)))
        self.next_population = new_next
        self.generation += 1
        self.frame_idx = 0


# ─── Fractal Renderer ────────────────────────────────────────────────────────
class FractalRenderer:
    @staticmethod
    def draw_fractal_qp(painter, gene, global_time, image_size,
                        alpha=1.0, swirl=0.0, perlin_scale=0.08,
                        shimmer=True):
        w, h = image_size
        rng = random.Random() if shimmer else random.Random(_gene_seed(gene))

        def _rec(x, y, length, angle, depth, bf, co, curv):
            if depth <= 0 or length < 2:
                return
            angle_m = angle + pnoise1(global_time + x * perlin_scale, repeat=1024) * 0.5 + curv
            x2 = x + math.cos(angle_m) * length
            y2 = y + math.sin(angle_m) * length

            # swirl around bottom-center
            cx, cy = w / 2.0, float(h)
            cos_s, sin_s = math.cos(swirl), math.sin(swirl)
            dx, dy = x - cx, y - cy
            dx2, dy2 = x2 - cx, y2 - cy
            xr = cx + dx * cos_s - dy * sin_s
            yr = cy + dx * sin_s + dy * cos_s
            xr2 = cx + dx2 * cos_s - dy2 * sin_s
            yr2 = cy + dx2 * sin_s + dy2 * cos_s

            col = harmonic_color(int(co * 1000), 1000, global_time * 0.01)
            a = int(np.clip(alpha, 0.0, 1.0) * 255)
            pen = QPen(QColor(int(col[0] * 255), int(col[1] * 255), int(col[2] * 255), a))
            pen.setWidth(max(1, depth))
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(xr, yr), QPointF(xr2, yr2))

            for i in range(bf):
                _rec(x2, y2,
                     length * rng.uniform(0.6, 0.8),
                     angle_m + rng.uniform(-0.5, 0.5),
                     depth - 1, bf, co + i * 0.05, curv)

        _rec(gene['x'], gene['y'], gene['length'], gene['angle'],
             gene['depth'], gene['branch_factor'], gene['color_offset'],
             gene['curvature'])

    @staticmethod
    def draw_gradient_qp(painter, image_size, color_top, color_bottom):
        w, h = image_size
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor(int(color_top[0] * 255),
                                  int(color_top[1] * 255),
                                  int(color_top[2] * 255)))
        grad.setColorAt(1, QColor(int(color_bottom[0] * 255),
                                  int(color_bottom[1] * 255),
                                  int(color_bottom[2] * 255)))
        painter.fillRect(0, 0, w, h, grad)

    @staticmethod
    def render_pil(individual, image_size, global_time=0.0, perlin_scale=0.08):
        w, h = image_size
        img = Image.new("RGBA", image_size, (0, 0, 0, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        for gene in individual:
            rng = random.Random(_gene_seed(gene))

            def _rec(x, y, length, angle, depth, bf, co, curv, _rng=rng):
                if depth <= 0 or length < 2:
                    return
                angle_m = angle + pnoise1(global_time + x * perlin_scale, repeat=1024) * 0.5 + curv
                x2 = x + math.cos(angle_m) * length
                y2 = y + math.sin(angle_m) * length
                col = harmonic_color(int(co * 1000), 1000, global_time * 0.01)
                r, g, b = int(col[0] * 255), int(col[1] * 255), int(col[2] * 255)
                a = int(50 + 205 * depth / 6)
                draw.line((int(x), int(y), int(x2), int(y2)),
                          fill=(r, g, b, a), width=max(1, depth))
                for i in range(bf):
                    _rec(x2, y2,
                         length * _rng.uniform(0.6, 0.8),
                         angle_m + _rng.uniform(-0.5, 0.5),
                         depth - 1, bf, co + i * 0.05, curv, _rng)

            _rec(gene['x'], gene['y'], gene['length'], gene['angle'],
                 gene['depth'], gene['branch_factor'], gene['color_offset'],
                 gene['curvature'])
        return img


# ─── Canvas Widget ────────────────────────────────────────────────────────────
class FractalCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ga = None
        self.perlin_scale = 0.08
        self.rotation_speed = 0.02
        self.show_gradient = True
        self.show_swirl = True
        self.shimmer = True
        self.highlight_best = False
        self._interp_pop = []
        self._gt = 0.0
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_ga(self, ga):
        self.ga = ga

    def update_frame(self, interp_pop, global_time):
        self._interp_pop = interp_pop
        self._gt = global_time
        self.update()

    def paintEvent(self, event):  # noqa: N802
        if self.ga is None or not self._interp_pop:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            painter.setPen(QColor(80, 80, 80))
            painter.drawText(self.rect(), Qt.AlignCenter, "Press Start or Reset")
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.ga.image_size
        scale = min(self.width() / w, self.height() / h)
        ox = (self.width() - w * scale) / 2.0
        oy = (self.height() - h * scale) / 2.0

        painter.save()
        painter.translate(ox, oy)
        painter.scale(scale, scale)

        # background
        if self.show_gradient:
            avg = np.zeros(3)
            for ind in self.ga.population:
                for gene in ind:
                    avg += harmonic_color(int(gene['color_offset'] * 1000), 1000,
                                          self._gt * 0.01)
            avg /= max(1, len(self.ga.population) * self.ga.num_fractals)
            FractalRenderer.draw_gradient_qp(painter, (w, h), avg * 0.9, avg * 0.15)
        else:
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0))

        # fractals
        pop = self._interp_pop
        n = len(pop)
        if self.highlight_best:
            indices = [0]
        else:
            indices = range(n)

        for i in indices:
            swirl_a = self._gt * self.rotation_speed * (i + 1) if self.show_swirl else 0.0
            if n > 1 and not self.highlight_best:
                alpha = 0.15 + 0.85 * i / (n - 1)
            else:
                alpha = 1.0
            for gene in pop[i]:
                FractalRenderer.draw_fractal_qp(
                    painter, gene, self._gt, (w, h),
                    alpha=alpha, swirl=swirl_a,
                    perlin_scale=self.perlin_scale,
                    shimmer=self.shimmer)

        painter.restore()
        painter.end()


# ─── Fitness Chart Widget ─────────────────────────────────────────────────────
class FitnessChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.setMinimumHeight(130)
        self.setMaximumHeight(200)

    def set_history(self, history):
        self.history = list(history)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        m = 35  # margin

        p.fillRect(0, 0, w, h, QColor(24, 24, 37))
        p.setPen(QPen(QColor(69, 71, 90), 1))
        p.drawRect(0, 0, w - 1, h - 1)

        if len(self.history) < 2:
            p.setPen(QColor(150, 150, 150))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "Waiting for data…")
            p.end()
            return

        # axes
        p.setPen(QPen(QColor(108, 112, 134), 1))
        p.drawLine(m, h - m, w - 10, h - m)
        p.drawLine(m, 10, m, h - m)

        mn, mx = min(self.history), max(self.history)
        rng = (mx - mn) if mx != mn else 1.0
        pw, ph = w - m - 10, h - m - 10

        # grid lines & labels
        p.setFont(QFont("Segoe UI", 7))
        p.setPen(QColor(108, 112, 134))
        for frac in (0.0, 0.5, 1.0):
            yy = int(h - m - frac * ph)
            p.drawLine(m, yy, w - 10, yy)
            val = mn + frac * rng
            p.drawText(2, yy - 2, f"{val:.0f}")

        # path
        path = QPainterPath()
        n = len(self.history)
        for i, f in enumerate(self.history):
            x = m + (i / max(1, n - 1)) * pw
            y = h - m - ((f - mn) / rng) * ph
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        # fill under curve
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

        # endpoint dot
        lx = m + pw
        ly = h - m - ((self.history[-1] - mn) / rng) * ph
        p.setBrush(QColor(243, 139, 168))
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawEllipse(QPointF(lx, ly), 4, 4)

        # gen label
        p.setPen(QColor(166, 173, 200))
        p.drawText(m, h - 2, "Gen 0")
        p.drawText(w - 40, h - 2, f"Gen {n - 1}")
        p.end()


# ─── Control Panel ────────────────────────────────────────────────────────────
class ControlPanel(QScrollArea):
    sig_start = Signal()
    sig_pause = Signal()
    sig_reset = Signal()
    sig_step = Signal()
    sig_screenshot = Signal()
    sig_export_seq = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedWidth(310)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setSpacing(6)
        lay.setContentsMargins(6, 6, 6, 6)

        title = QLabel("🧬 Fractal Genetic Art")
        title.setStyleSheet("font-size:15px;font-weight:bold;color:#89dceb;padding:4px 0;")
        lay.addWidget(title)

        # ── Evolution ──
        eg = QGroupBox("Evolution")
        ef = QFormLayout()
        self.sp_pop = QSpinBox(); self.sp_pop.setRange(1, 30); self.sp_pop.setValue(5)
        self.sp_fract = QSpinBox(); self.sp_fract.setRange(1, 10); self.sp_fract.setValue(3)
        self.sp_mut = QDoubleSpinBox(); self.sp_mut.setRange(0.0, 1.0); self.sp_mut.setSingleStep(0.05); self.sp_mut.setValue(0.20)
        self.sp_fpg = QSpinBox(); self.sp_fpg.setRange(1, 100); self.sp_fpg.setValue(20)
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
        self.cb_size.addItems(["512×512", "600×600", "800×800", "1024×1024"])
        self.cb_size.setCurrentIndex(1)
        self.sp_perlin = QDoubleSpinBox(); self.sp_perlin.setRange(0.0, 0.5); self.sp_perlin.setSingleStep(0.01); self.sp_perlin.setDecimals(3); self.sp_perlin.setValue(0.08)
        self.sp_rot = QDoubleSpinBox(); self.sp_rot.setRange(0.0, 0.2); self.sp_rot.setSingleStep(0.005); self.sp_rot.setDecimals(3); self.sp_rot.setValue(0.02)
        self.chk_grad = QCheckBox("Gradient bg"); self.chk_grad.setChecked(True)
        self.chk_swirl = QCheckBox("Swirling"); self.chk_swirl.setChecked(True)
        self.chk_shimmer = QCheckBox("Shimmer mode"); self.chk_shimmer.setChecked(True)
        self.chk_best = QCheckBox("Highlight best only"); self.chk_best.setChecked(False)
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
        self.sl_speed = QSlider(Qt.Horizontal); self.sl_speed.setRange(10, 200); self.sl_speed.setValue(50)
        self.lb_speed = QLabel("50 ms")
        self.sl_speed.valueChanged.connect(lambda v: self.lb_speed.setText(f"{v} ms"))
        sl.addWidget(self.sl_speed); sl.addWidget(self.lb_speed)
        al.addLayout(sl)

        bl = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start")
        self.btn_pause = QPushButton("⏸ Pause"); self.btn_pause.setEnabled(False)
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_start.clicked.connect(self.sig_start)
        self.btn_pause.clicked.connect(self.sig_pause)
        self.btn_reset.clicked.connect(self.sig_reset)
        bl.addWidget(self.btn_start); bl.addWidget(self.btn_pause); bl.addWidget(self.btn_reset)
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
        self.btn_browse = QPushButton("…"); self.btn_browse.setFixedWidth(28)
        self.btn_browse.clicked.connect(self._browse)
        dl.addWidget(self.le_dir); dl.addWidget(self.btn_browse)
        xl.addLayout(dl)
        el = QHBoxLayout()
        self.btn_snap = QPushButton("📸 Screenshot"); self.btn_snap.clicked.connect(self.sig_screenshot)
        self.btn_seq = QPushButton("🎬 Sequence"); self.btn_seq.clicked.connect(self.sig_export_seq)
        el.addWidget(self.btn_snap); el.addWidget(self.btn_seq)
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

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Output directory")
        if d:
            self.le_dir.setText(d)

    @property
    def image_size(self):
        return [(512, 512), (600, 600), (800, 800), (1024, 1024)][self.cb_size.currentIndex()]

    @property
    def params(self):
        return dict(image_size=self.image_size, pop_size=self.sp_pop.value(),
                    num_fractals=self.sp_fract.value(), mutation_rate=self.sp_mut.value(),
                    frames_per_gen=self.sp_fpg.value())

    @property
    def interval(self):
        return self.sl_speed.value()

    @property
    def export_dir(self):
        return self.le_dir.text()

    def refresh_info(self, ga):
        self.lb_gen.setText(str(ga.generation))
        self.lb_frame.setText(f"{ga.frame_idx} / {ga.frames_per_gen}")
        self.lb_fit.setText(f"{ga.best_fitness:.1f}")
        self.lb_time.setText(f"{ga.global_time:.1f}")
        self.chart.set_history(ga.fitness_history)

    def set_running(self, on):
        self.btn_start.setEnabled(not on)
        self.btn_pause.setEnabled(on)
        self.btn_reset.setEnabled(True)
        for w in (self.sp_pop, self.sp_fract, self.cb_size):
            w.setEnabled(not on)


# ─── Main Window ──────────────────────────────────────────────────────────────
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
    background:#313244; border:1px solid #45475a; border-radius:3px; padding:3px; color:#cdd6f4; }
QSlider::groove:horizontal { height:6px; background:#45475a; border-radius:3px; }
QSlider::handle:horizontal { background:#89b4fa; width:14px; margin:-4px 0; border-radius:7px; }
QScrollArea { border:none; }
QCheckBox { spacing:5px; }
QCheckBox::indicator { width:15px; height:15px; }
QCheckBox::indicator:unchecked { border:1px solid #585b70; background:#313244; border-radius:3px; }
QCheckBox::indicator:checked { border:1px solid #89b4fa; background:#89b4fa; border-radius:3px; }
QComboBox::drop-down { border:none; }
QComboBox QAbstractItemView { background:#313244; color:#cdd6f4; selection-background-color:#45475a; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractal Genetic Art Studio")
        self.setMinimumSize(1050, 720)
        self.resize(1200, 800)
        self.setStyleSheet(DARK_STYLE)

        cw = QWidget()
        self.setCentralWidget(cw)
        ml = QHBoxLayout(cw)
        ml.setContentsMargins(4, 4, 4, 4)
        ml.setSpacing(4)

        self.canvas = FractalCanvas()
        self.canvas.setStyleSheet("background:#11111b; border-radius:6px;")
        ml.addWidget(self.canvas, 1)

        self.ctrl = ControlPanel()
        ml.addWidget(self.ctrl)

        # signals
        self.ctrl.sig_start.connect(self._start)
        self.ctrl.sig_pause.connect(self._pause)
        self.ctrl.sig_reset.connect(self._reset)
        self.ctrl.sig_step.connect(self._step_gen)
        self.ctrl.sig_screenshot.connect(self._screenshot)
        self.ctrl.sig_export_seq.connect(self._export_seq)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.ga = None
        self._running = False
        self._seq_active = False

        self._reset()

    # ── controls ──
    def _sync_visuals(self):
        c = self.canvas
        c.perlin_scale = self.ctrl.sp_perlin.value()
        c.rotation_speed = self.ctrl.sp_rot.value()
        c.show_gradient = self.ctrl.chk_grad.isChecked()
        c.show_swirl = self.ctrl.chk_swirl.isChecked()
        c.shimmer = self.ctrl.chk_shimmer.isChecked()
        c.highlight_best = self.ctrl.chk_best.isChecked()

    def _reset(self):
        self._pause()
        self.ga = GeneticAlgorithm(**self.ctrl.params)
        self.canvas.set_ga(self.ga)
        self._sync_visuals()
        ip = self.ga.get_interpolated_population()
        self.canvas.update_frame(ip, self.ga.global_time)
        self.ctrl.refresh_info(self.ga)
        self.ctrl.set_running(False)

    def _start(self):
        if self.ga is None:
            self._reset()
        self._running = True
        self.timer.start(self.ctrl.interval)
        self.ctrl.set_running(True)

    def _pause(self):
        self._running = False
        self.timer.stop()
        self.ctrl.set_running(False)

    def _tick(self):
        if self.ga is None:
            return
        self._sync_visuals()
        ip = self.ga.get_interpolated_population()
        gt = self.ga.global_time
        evolved = self.ga.step()
        self.canvas.update_frame(ip, gt)
        self.ctrl.refresh_info(self.ga)
        if evolved:
            self.ctrl.chart.set_history(self.ga.fitness_history)

    def _step_gen(self):
        if self.ga is None:
            return
        self._sync_visuals()
        while self.ga.frame_idx < self.ga.frames_per_gen:
            self.ga.step()
        ip = self.ga.get_interpolated_population()
        self.canvas.update_frame(ip, self.ga.global_time)
        self.ctrl.refresh_info(self.ga)
        self.ctrl.chart.set_history(self.ga.fitness_history)

    # ── export ──
    def _screenshot(self):
        if self.ga is None:
            return
        d = self.ctrl.export_dir
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, f"screenshot_gen{self.ga.generation:04d}_f{self.ga.frame_idx:03d}.png")
        pix = self.canvas.grab()
        pix.save(fn)
        QMessageBox.information(self, "Saved", f"Screenshot saved to:\n{fn}")

    def _export_seq(self):
        if self.ga is None:
            return
        params = self.ctrl.params
        num_gens = 50
        total = num_gens * params['frames_per_gen']

        d = self.ctrl.export_dir
        os.makedirs(d, exist_ok=True)

        dlg = QProgressDialog("Exporting sequence…", "Cancel", 0, total, self)
        dlg.setWindowTitle("Export")
        dlg.setMinimumDuration(0)
        dlg.show()

        seq_ga = GeneticAlgorithm(**params)
        exported = 0
        self._seq_active = True

        def _do_batch():
            nonlocal exported
            batch = 10
            for _ in range(batch):
                if not self._seq_active or dlg.wasCanceled():
                    break
                ip = seq_ga.get_interpolated_population()
                best = ip[0]
                img = FractalRenderer.render_pil(best, seq_ga.image_size,
                                                  seq_ga.global_time,
                                                  self.ctrl.sp_perlin.value())
                fn = os.path.join(d, f"frame_{exported:05d}.png")
                img.save(fn)
                seq_ga.step()
                exported += 1
                dlg.setValue(exported)
                dlg.setLabelText(f"Gen {seq_ga.generation}  frame {seq_ga.frame_idx}")
                QApplication.processEvents()

            if exported >= total or not self._seq_active or dlg.wasCanceled():
                dlg.close()
                if exported > 0:
                    QMessageBox.information(self, "Done",
                                            f"Exported {exported} frames to:\n{d}")
                self._seq_active = False
            else:
                QTimer.singleShot(0, _do_batch)

        QTimer.singleShot(0, _do_batch)

    # ── lifecycle ──
    def closeEvent(self, event):  # noqa: N802
        self._pause()
        self._seq_active = False
        event.accept()


# ─── Entry Point ──────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()