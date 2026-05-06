#!/usr/bin/env python3
"""Fractal Genetic Art Studio — application entry point."""

import os
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QProgressDialog, QWidget
from PySide6.QtCore import QTimer

from config import DARK_STYLE, DEFAULTS
from genetics import GeneticAlgorithm
from renderer_pil import render_individual_pil
from widgets import FractalCanvas, ControlPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fractal Genetic Art Studio")
        self.setMinimumSize(1050, 720)
        self.resize(1200, 800)
        self.setStyleSheet(DARK_STYLE)

        cw = QWidget()
        self.setCentralWidget(cw)
        ml = QVBoxLayout(cw)  # Use VBoxLayout at top level
        ml.setContentsMargins(4, 4, 4, 4)
        ml.setSpacing(4)

        # Horizontal layout for canvas + controls
        hl = QHBoxLayout()
        hl.setSpacing(4)

        self.canvas = FractalCanvas()
        self.canvas.setStyleSheet("background:#11111b; border-radius:6px;")
        hl.addWidget(self.canvas, 1)

        self.ctrl = ControlPanel()
        hl.addWidget(self.ctrl)

        ml.addLayout(hl, 1)

        # Signals
        self.ctrl.sig_start.connect(self._start)
        self.ctrl.sig_pause.connect(self._pause)
        self.ctrl.sig_reset.connect(self._reset)
        self.ctrl.sig_step.connect(self._step_gen)
        self.ctrl.sig_screenshot.connect(self._screenshot)
        self.ctrl.sig_export_seq.connect(self._export_seq)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.ga: GeneticAlgorithm | None = None
        self._running = False
        self._seq_active = False

        self._reset()

    # ── sync visual params from control panel to canvas ──
    def _sync_visuals(self) -> None:
        c = self.canvas
        c.perlin_scale = self.ctrl.sp_perlin.value()
        c.rotation_speed = self.ctrl.sp_rot.value()
        c.show_gradient = self.ctrl.chk_grad.isChecked()
        c.show_swirl = self.ctrl.chk_swirl.isChecked()
        c.shimmer = self.ctrl.chk_shimmer.isChecked()
        c.highlight_best = self.ctrl.chk_best.isChecked()

    # ── slot implementations ──
    def _reset(self) -> None:
        self._pause()
        self.ga = GeneticAlgorithm(**self.ctrl.ga_params)
        self.canvas.set_ga(self.ga)
        self._sync_visuals()
        ip = self.ga.get_interpolated_population()
        self.canvas.update_frame(ip, self.ga.global_time)
        self.ctrl.refresh_info(self.ga)
        self.ctrl.set_running(False)

    def _start(self) -> None:
        if self.ga is None:
            self._reset()
        self._running = True
        self.timer.start(self.ctrl.interval)
        self.ctrl.set_running(True)

    def _pause(self) -> None:
        self._running = False
        self.timer.stop()
        self.ctrl.set_running(False)

    def _tick(self) -> None:
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
        QMessageBox.information(self, "Saved", f"Screenshot saved to:\n{fn}")

    def _export_seq(self) -> None:
        if self.ga is None:
            return
        params = self.ctrl.ga_params
        num_gens = DEFAULTS["export_gens"]
        total = num_gens * params["frames_per_gen"]
        d = self.ctrl.export_dir
        os.makedirs(d, exist_ok=True)
        perlin = self.ctrl.sp_perlin.value()

        dlg = QProgressDialog("Exporting sequence…", "Cancel", 0, total, self)
        dlg.setWindowTitle("Export")
        dlg.setMinimumDuration(0)
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
                    best, seq_ga.image_size, seq_ga.global_time, perlin)
                img.save(os.path.join(d, f"frame_{exported:05d}.png"))
                seq_ga.step()
                exported += 1
                dlg.setValue(exported)
                dlg.setLabelText(
                    f"Gen {seq_ga.generation}  frame {seq_ga.frame_idx}")
                QApplication.processEvents()

            if exported >= total or not self._seq_active or dlg.wasCanceled():
                dlg.close()
                if exported > 0:
                    QMessageBox.information(
                        self, "Done", f"Exported {exported} frames to:\n{d}")
                self._seq_active = False
            else:
                QTimer.singleShot(0, _batch)

        QTimer.singleShot(0, _batch)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._pause()
        self._seq_active = False
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()