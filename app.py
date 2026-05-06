#!/usr/bin/env python3
"""Fractal Genetic Art Studio — application entry point."""

from __future__ import annotations

import os
import sys
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QWidget,
    QStatusBar,
    QToolBar,
    QSizePolicy,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QIcon

from config import DARK_STYLE, DEFAULTS, VERSION
from genetics import GeneticAlgorithm
from renderers import render_individual_pil
from widgets import FractalCanvas, ControlPanel, StatusBarWidget


class MainWindow(QMainWindow):
    """Main application window with improved layout structure."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Fractal Genetic Art Studio v{VERSION}")
        self.setMinimumSize(1100, 750)
        self.resize(1280, 860)
        self.setStyleSheet(DARK_STYLE)

        # ── Central widget with main layout ──
        central = QWidget()
        self.setCentralWidget(central)
        self._main_layout = self._build_main_layout(central)

        # ── Toolbar ──
        self._build_toolbar()

        # ── Status bar ──
        self._status_widget = StatusBarWidget()
        self.statusBar().addPermanentWidget(self._status_widget)
        self.statusBar().setStyleSheet("background:#181825; color:#a6adc8;")

        # ── Timer ──
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # ── State ──
        self.ga: Optional[GeneticAlgorithm] = None
        self._running = False
        self._seq_active = False
        self._fps_counter = 0
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

        self._reset()

    def _build_main_layout(self, parent: QWidget) -> None:
        """Build the main horizontal layout: canvas + control panel."""
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(8)

        # ── Canvas area with frame ──
        self.canvas = FractalCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)

        # ── Control panel ──
        self.ctrl = ControlPanel()
        layout.addWidget(self.ctrl)

        # ── Connect signals ──
        self.ctrl.sig_start.connect(self._start)
        self.ctrl.sig_pause.connect(self._pause)
        self.ctrl.sig_reset.connect(self._reset)
        self.ctrl.sig_step.connect(self._step_gen)
        self.ctrl.sig_screenshot.connect(self._screenshot)
        self.ctrl.sig_export_seq.connect(self._export_seq)

    def _build_toolbar(self) -> None:
        """Build the top toolbar with common actions."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.setIconSize(toolbar.iconSize())
        toolbar.setStyleSheet("""
            QToolBar {
                background: #181825;
                border-bottom: 1px solid #313244;
                spacing: 6px;
                padding: 4px 8px;
            }
            QToolButton {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 5px 12px;
                color: #cdd6f4;
                font-size: 12px;
            }
            QToolButton:hover { background: #45475a; }
            QToolButton:pressed { background: #1e1e2e; }
            QToolButton:disabled { color: #6c7086; }
        """)
        self.addToolBar(toolbar)

        actions = [
            ("▶ Start", "Start evolution", self._start),
            ("⏸ Pause", "Pause evolution", self._pause),
            ("↺ Reset", "Reset population", self._reset),
            ("⏭ Step", "Advance one generation", self._step_gen),
        ]
        for text, tooltip, slot in actions:
            action = QAction(text, self)
            action.setToolTip(tooltip)
            action.triggered.connect(slot)
            toolbar.addAction(action)

        toolbar.addSeparator()

        snap_action = QAction("📸 Screenshot", self)
        snap_action.setToolTip("Save current frame as image")
        snap_action.triggered.connect(self._screenshot)
        toolbar.addAction(snap_action)

    def _sync_visuals(self) -> None:
        """Sync visual parameters from control panel to canvas."""
        c = self.canvas
        c.perlin_scale = self.ctrl.sp_perlin.value()
        c.rotation_speed = self.ctrl.sp_rot.value()
        c.show_gradient = self.ctrl.chk_grad.isChecked()
        c.show_swirl = self.ctrl.chk_swirl.isChecked()
        c.shimmer = self.ctrl.chk_shimmer.isChecked()
        c.highlight_best = self.ctrl.chk_best.isChecked()

    def _update_status(self) -> None:
        """Update status bar with current GA state."""
        if self.ga is None:
            return
        sw = self._status_widget
        sw.gen = self.ga.generation
        sw.frame = f"{self.ga.frame_idx}/{self.ga.frames_per_gen}"
        sw.fitness = f"{self.ga.best_fitness:.1f}"
        sw.pop_size = self.ga.pop_size
        sw.state = "Running" if self._running else "Paused"

    def _update_fps(self) -> None:
        """Update FPS display."""
        self._status_widget.fps = self._fps_counter
        self._fps_counter = 0

    # ── Slot implementations ──

    def _reset(self) -> None:
        self._pause()
        self.ga = GeneticAlgorithm(**self.ctrl.ga_params)
        self.canvas.set_ga(self.ga)
        self._sync_visuals()
        ip = self.ga.get_interpolated_population()
        self.canvas.update_frame(ip, self.ga.global_time)
        self.ctrl.refresh_info(self.ga)
        self.ctrl.set_running(False)
        self._update_status()
        self.statusBar().showMessage("Population reset — ready to evolve", 3000)

    def _start(self) -> None:
        if self.ga is None:
            self._reset()
        self._running = True
        self.timer.start(self.ctrl.interval)
        self.ctrl.set_running(True)
        self._update_status()
        self.statusBar().showMessage("Evolution running…", 2000)

    def _pause(self) -> None:
        self._running = False
        self.timer.stop()
        self.ctrl.set_running(False)
        self._update_status()
        if self.ga and self.ga.generation > 0:
            self.statusBar().showMessage("Paused", 2000)

    def _tick(self) -> None:
        if self.ga is None:
            return
        self._fps_counter += 1
        self._sync_visuals()
        ip = self.ga.get_interpolated_population()
        gt = self.ga.global_time
        evolved = self.ga.step()
        self.canvas.update_frame(ip, gt)
        self.ctrl.refresh_info(self.ga)
        self._update_status()
        if evolved:
            self.ctrl.chart.set_history(self.ga.fitness_history)

    def _step_gen(self) -> None:
        if self.ga is None:
            return
        self._sync_visuals()
        while self.ga.frame_idx < self.ga.frames_per_gen:
            self.ga.step()
        ip = self.ga.get_interpolated_population()
        self.canvas.update_frame(ip, self.ga.global_time)
        self.ctrl.refresh_info(self.ga)
        self.ctrl.chart.set_history(self.ga.fitness_history)
        self._update_status()
        self.statusBar().showMessage(
            f"Stepped to generation {self.ga.generation}", 2000
        )

    def _screenshot(self) -> None:
        if self.ga is None:
            return
        d = self.ctrl.export_dir
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(
            d,
            f"screenshot_gen{self.ga.generation:04d}_f{self.ga.frame_idx:03d}.png",
        )
        self.canvas.grab().save(fn)
        self.statusBar().showMessage(f"Screenshot saved: {fn}", 4000)
        QMessageBox.information(self, "Screenshot Saved", f"Saved to:\n{fn}")

    def _export_seq(self) -> None:
        if self.ga is None:
            return
        params = self.ctrl.ga_params
        num_gens = DEFAULTS["export_gens"]
        total = num_gens * params["frames_per_gen"]
        d = self.ctrl.export_dir
        os.makedirs(d, exist_ok=True)
        perlin = self.ctrl.sp_perlin.value()

        dlg = QProgressDialog("Exporting frame sequence…", "Cancel", 0, total, self)
        dlg.setWindowTitle("Export Sequence")
        dlg.setMinimumDuration(0)
        dlg.setFixedSize(400, 150)
        dlg.show()

        seq_ga = GeneticAlgorithm(**params)
        exported = 0
        self._seq_active = True

        def _batch() -> None:
            nonlocal exported
            for _ in range(10):
                if not self._seq_active or dlg.wasCanceled():
                    break
                best = seq_ga.get_interpolated_population()[0]
                img = render_individual_pil(
                    best, seq_ga.image_size, seq_ga.global_time, perlin
                )
                img.save(os.path.join(d, f"frame_{exported:05d}.png"))
                seq_ga.step()
                exported += 1
                dlg.setValue(exported)
                dlg.setLabelText(
                    f"Gen {seq_ga.generation} · Frame {seq_ga.frame_idx}\n"
                    f"Progress: {exported}/{total}"
                )
                QApplication.processEvents()

            if exported >= total or not self._seq_active or dlg.wasCanceled():
                dlg.close()
                if exported > 0:
                    self.statusBar().showMessage(
                        f"Exported {exported} frames to {d}", 5000
                    )
                    QMessageBox.information(
                        self, "Export Complete",
                        f"Exported {exported} frames to:\n{d}"
                    )
                self._seq_active = False
            else:
                QTimer.singleShot(0, _batch)

        QTimer.singleShot(0, _batch)

    def closeEvent(self, event) -> None:
        self._pause()
        self._seq_active = False
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Fractal Genetic Art Studio")
    app.setApplicationVersion(VERSION)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()