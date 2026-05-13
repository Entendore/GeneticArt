#!/usr/bin/env python3
"""Fractal Genetic Art Studio v2.0 — application entry point."""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QWidget,
    QToolBar,
    QSizePolicy,
    QHBoxLayout,
    QDockWidget,
    QFileDialog,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence

from config import DARK_STYLE, DEFAULTS, VERSION, SHORTCUTS, C
from genetics import GeneticAlgorithm
from renderers import render_individual_pil
from widgets import FractalCanvas, ControlPanel, StatusBarWidget, PopulationGallery


class MainWindow(QMainWindow):
    """Main application window with menu, toolbar, gallery dock, and canvas."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Fractal Genetic Art Studio v{VERSION}")
        self.setMinimumSize(1100, 750)
        self.resize(1400, 900)
        self.setStyleSheet(DARK_STYLE)

        # ── Central widget with main layout ──
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(8)

        # Canvas
        self.canvas = FractalCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)

        # Control panel
        self.ctrl = ControlPanel()
        layout.addWidget(self.ctrl)

        # ── Gallery dock ──
        self.gallery = PopulationGallery()
        self.gallery.individual_selected.connect(self._on_gallery_select)
        self._gallery_dock = QDockWidget("Population Gallery", self)
        self._gallery_dock.setWidget(self.gallery)
        self._gallery_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self._gallery_dock.setStyleSheet(
            f"QDockWidget {{ color: {C['text']}; }}"
            f"QDockWidget::title {{ background: {C['mantle']}; padding: 6px;"
            f" border-bottom: 1px solid {C['surface0']}; }}"
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self._gallery_dock)

        # ── Menu bar ──
        self._build_menu_bar()

        # ── Toolbar ──
        self._build_toolbar()

        # ── Status bar ──
        self._status_widget = StatusBarWidget()
        self.statusBar().addPermanentWidget(self._status_widget)
        self.statusBar().setStyleSheet(
            f"background:{C['mantle']}; color:{C['subtext0']};"
        )

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
        self._gallery_timer = QTimer(self)
        self._gallery_timer.setSingleShot(True)
        self._gallery_timer.setInterval(2000)
        self._gallery_timer.timeout.connect(self._refresh_gallery)

        self._reset()

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menu_bar(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        act_save = QAction("💾 Save Population…", self)
        act_save.setShortcut(QKeySequence(SHORTCUTS["save"]))
        act_save.setToolTip("Save current population state to JSON")
        act_save.triggered.connect(self._save_population)
        file_menu.addAction(act_save)

        act_load = QAction("📂 Load Population…", self)
        act_load.setShortcut(QKeySequence(SHORTCUTS["load"]))
        act_load.setToolTip("Load population state from JSON")
        act_load.triggered.connect(self._load_population)
        file_menu.addAction(act_load)

        file_menu.addSeparator()

        act_screenshot = QAction("📸 Screenshot", self)
        act_screenshot.setShortcut(QKeySequence(SHORTCUTS["screenshot"]))
        act_screenshot.triggered.connect(self._screenshot)
        file_menu.addAction(act_screenshot)

        act_seq = QAction("🎬 Export Sequence…", self)
        act_seq.setShortcut(QKeySequence(SHORTCUTS["export_seq"]))
        act_seq.triggered.connect(self._export_seq)
        file_menu.addAction(act_seq)

        act_gif = QAction("🎞️ Export GIF…", self)
        act_gif.setShortcut(QKeySequence(SHORTCUTS["export_gif"]))
        act_gif.triggered.connect(self._export_gif)
        file_menu.addAction(act_gif)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence(SHORTCUTS["quit"]))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Evolution menu
        evo_menu = menubar.addMenu("&Evolution")

        act_start = QAction("▶ Start", self)
        act_start.setShortcut(QKeySequence(SHORTCUTS["start_pause"]))
        act_start.triggered.connect(self._toggle_start_pause)
        evo_menu.addAction(act_start)

        act_reset = QAction("↺ Reset", self)
        act_reset.setShortcut(QKeySequence(SHORTCUTS["reset"]))
        act_reset.triggered.connect(self._reset)
        evo_menu.addAction(act_reset)

        act_step = QAction("⏭ Step Generation", self)
        act_step.setShortcut(QKeySequence(SHORTCUTS["step"]))
        act_step.triggered.connect(self._step_gen)
        evo_menu.addAction(act_step)

        # View menu
        view_menu = menubar.addMenu("&View")

        act_gallery = QAction("Toggle Gallery", self)
        act_gallery.setShortcut(QKeySequence(SHORTCUTS["toggle_gallery"]))
        act_gallery.triggered.connect(self._toggle_gallery)
        view_menu.addAction(act_gallery)

        act_toolbar = QAction("Toggle Toolbar", self)
        act_toolbar.triggered.connect(self._toggle_toolbar)
        view_menu.addAction(act_toolbar)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

        act_shortcuts = QAction("Keyboard Shortcuts", self)
        act_shortcuts.triggered.connect(self._show_shortcuts)
        help_menu.addAction(act_shortcuts)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.setStyleSheet(
            "QToolBar {"
            f"  background: {C['mantle']};"
            f"  border-bottom: 1px solid {C['surface0']};"
            "  spacing: 6px; padding: 4px 8px;"
            "}"
            "QToolButton {"
            f"  background: {C['surface1']};"
            f"  border: 1px solid {C['surface2']};"
            "  border-radius: 4px; padding: 5px 12px;"
            f"  color: {C['text']}; font-size: 12px;"
            "}"
            f"QToolButton:hover {{ background: {C['surface2']}; }}"
            f"QToolButton:pressed {{ background: {C['surface0']}; }}"
            f"QToolButton:disabled {{ color: {C['overlay0']}; }}"
        )
        self.addToolBar(toolbar)

        actions = [
            ("▶ Start", "Start / Pause evolution", self._toggle_start_pause),
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

        gif_action = QAction("🎞️ GIF", self)
        gif_action.setToolTip("Export animated GIF")
        gif_action.triggered.connect(self._export_gif)
        toolbar.addAction(gif_action)

    # ── Sync visuals ──────────────────────────────────────────────────────────

    def _sync_visuals(self) -> None:
        """Sync visual parameters from control panel to canvas."""
        c = self.canvas
        c.perlin_scale = self.ctrl.sp_perlin.value()
        c.rotation_speed = self.ctrl.sp_rot.value()
        c.show_gradient = self.ctrl.chk_grad.isChecked()
        c.show_swirl = self.ctrl.chk_swirl.isChecked()
        c.shimmer = self.ctrl.chk_shimmer.isChecked()
        c.highlight_best = self.ctrl.chk_best.isChecked()
        c.show_leaves = self.ctrl.chk_leaves.isChecked()
        c.show_glow = self.ctrl.chk_glow.isChecked()
        c.palette_name = self.ctrl.cb_palette.currentText()
        c.easing = self.ctrl.cb_easing.currentText()

    # ── Status updates ────────────────────────────────────────────────────────

    def _update_status(self) -> None:
        if self.ga is None:
            return
        stats = self.ga.get_stats()
        sw = self._status_widget
        sw.gen = stats["generation"]
        sw.frame = f"{stats['frame']}/{stats['frames_per_gen']}"
        sw.fitness = f"{stats['best_fitness']:.1f}"
        sw.mutation_rate = f"{stats['mutation_rate']:.2f}"
        sw.diversity = f"{stats['diversity']:.1f}"
        sw.pop_size = stats["pop_size"]
        sw.state = "Running" if self._running else "Paused"

    def _update_fps(self) -> None:
        self._status_widget.fps = self._fps_counter
        self._fps_counter = 0

    # ── Slot implementations ──────────────────────────────────────────────────

    def _reset(self) -> None:
        self._pause()
        self.ga = GeneticAlgorithm(**self.ctrl.ga_params)
        self.canvas.set_ga(self.ga)
        self._sync_visuals()
        ip = self.ga.get_interpolated_population(self.ctrl.current_easing)
        self.canvas.update_frame(ip, self.ga.global_time)
        self.ctrl.refresh_info(self.ga)
        self.ctrl.set_running(False)
        self._update_status()
        self._refresh_gallery()
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
        if hasattr(self, "ctrl"):
            self.ctrl.set_running(False)
        self._update_status()

    def _toggle_start_pause(self) -> None:
        if self._running:
            self._pause()
            if self.ga and self.ga.generation > 0:
                self.statusBar().showMessage("Paused", 2000)
        else:
            self._start()

    def _tick(self) -> None:
        if self.ga is None:
            return
        self._fps_counter += 1
        self._sync_visuals()
        ip = self.ga.get_interpolated_population(self.ctrl.current_easing)
        gt = self.ga.global_time
        evolved = self.ga.step()
        self.canvas.update_frame(ip, gt)
        self.ctrl.refresh_info(self.ga)
        self._update_status()
        if evolved:
            self.ctrl.chart.set_history(
                self.ga.fitness_history,
                self.ga.avg_fitness_history,
                self.ga.diversity_history,
            )
            # Refresh gallery periodically (not every gen for performance)
            if self.ga.generation % 3 == 0:
                self._gallery_timer.start()

    def _step_gen(self) -> None:
        if self.ga is None:
            return
        self._sync_visuals()
        while self.ga.frame_idx < self.ga.frames_per_gen:
            self.ga.step()
        ip = self.ga.get_interpolated_population(self.ctrl.current_easing)
        self.canvas.update_frame(ip, self.ga.global_time)
        self.ctrl.refresh_info(self.ga)
        self.ctrl.chart.set_history(
            self.ga.fitness_history,
            self.ga.avg_fitness_history,
            self.ga.diversity_history,
        )
        self._update_status()
        self._refresh_gallery()
        self.statusBar().showMessage(
            f"Stepped to generation {self.ga.generation}", 2000
        )

    def _on_gallery_select(self, index: int) -> None:
        """Handle gallery individual selection."""
        self.canvas.selected_index = index
        self.canvas.update()

    def _refresh_gallery(self) -> None:
        """Update gallery thumbnails from current population."""
        if self.ga is None:
            return
        self.gallery.update_thumbnails(
            self.ga.population,
            self.ga.global_time,
            self.ctrl.sp_perlin.value(),
            self.ctrl.cb_palette.currentText(),
            self.ga.image_size,
        )

    # ── Screenshot ────────────────────────────────────────────────────────────

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

    # ── Export sequence ───────────────────────────────────────────────────────

    def _export_seq(self) -> None:
        if self.ga is None:
            return
        params = self.ctrl.ga_params
        num_gens = DEFAULTS["export_gens"]
        total = num_gens * params["frames_per_gen"]
        d = self.ctrl.export_dir
        os.makedirs(d, exist_ok=True)
        perlin = self.ctrl.sp_perlin.value()
        palette = self.ctrl.cb_palette.currentText()
        show_leaves = self.ctrl.chk_leaves.isChecked()

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
                best = seq_ga.get_interpolated_population(
                    self.ctrl.current_easing
                )[0]
                img = render_individual_pil(
                    best, seq_ga.image_size, seq_ga.global_time,
                    perlin, palette, show_leaves,
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
                        f"Exported {exported} frames to:\n{d}",
                    )
                self._seq_active = False
            else:
                QTimer.singleShot(0, _batch)

        QTimer.singleShot(0, _batch)

    # ── Export GIF ────────────────────────────────────────────────────────────

    def _export_gif(self) -> None:
        if self.ga is None:
            return
        d = self.ctrl.export_dir
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, f"evolution_gen{self.ga.generation:04d}.gif")
        params = self.ctrl.ga_params
        num_gens = min(20, DEFAULTS["export_gens"])
        frames_per_gif = 8
        total = num_gens * frames_per_gif
        perlin = self.ctrl.sp_perlin.value()
        palette = self.ctrl.cb_palette.currentText()
        show_leaves = self.ctrl.chk_leaves.isChecked()
        img_size = params["image_size"]

        dlg = QProgressDialog("Exporting animated GIF…", "Cancel", 0, total, self)
        dlg.setWindowTitle("Export GIF")
        dlg.setMinimumDuration(0)
        dlg.setFixedSize(400, 150)
        dlg.show()

        gif_ga = GeneticAlgorithm(**params)
        frames = []
        exported = 0
        self._seq_active = True

        def _batch() -> None:
            nonlocal exported
            for _ in range(5):
                if not self._seq_active or dlg.wasCanceled():
                    break
                best = gif_ga.get_interpolated_population(
                    self.ctrl.current_easing
                )[0]
                img = render_individual_pil(
                    best, img_size, gif_ga.global_time,
                    perlin, palette, show_leaves,
                ).convert("RGB")
                frames.append(img)
                gif_ga.step()
                exported += 1
                dlg.setValue(exported)
                dlg.setLabelText(
                    f"Gen {gif_ga.generation} · "
                    f"Frame {exported}/{total}\n"
                    f"Rendering frames…"
                )
                QApplication.processEvents()

            if exported >= total or not self._seq_active or dlg.wasCanceled():
                dlg.close()
                if frames:
                    try:
                        frames[0].save(
                            fn,
                            save_all=True,
                            append_images=frames[1:],
                            duration=80,
                            loop=0,
                            optimize=True,
                        )
                        self.statusBar().showMessage(
                            f"GIF saved: {fn} ({len(frames)} frames)", 5000
                        )
                        QMessageBox.information(
                            self, "GIF Export Complete",
                            f"Saved {len(frames)} frames to:\n{fn}",
                        )
                    except Exception as e:
                        QMessageBox.critical(
                            self, "GIF Export Error",
                            f"Failed to save GIF:\n{e}",
                        )
                self._seq_active = False
            else:
                QTimer.singleShot(0, _batch)

        QTimer.singleShot(0, _batch)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def _save_population(self) -> None:
        if self.ga is None:
            QMessageBox.warning(self, "Nothing to Save", "No active population.")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Population", "population.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not fn:
            return
        try:
            data = self.ga.to_dict()
            with open(fn, "w") as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"Population saved: {fn}", 4000)
            QMessageBox.information(
                self, "Saved",
                f"Population saved to:\n{fn}\n\n"
                f"Generation: {self.ga.generation}\n"
                f"Best fitness: {self.ga.best_fitness:.1f}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")

    def _load_population(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self, "Load Population", "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not fn:
            return
        try:
            with open(fn, "r") as f:
                data = json.load(f)
            self._pause()
            self.ga = GeneticAlgorithm.from_dict(data)
            self.canvas.set_ga(self.ga)
            self._sync_visuals()
            ip = self.ga.get_interpolated_population(self.ctrl.current_easing)
            self.canvas.update_frame(ip, self.ga.global_time)
            self.ctrl.refresh_info(self.ga)
            self.ctrl.set_running(False)
            self._update_status()
            self._refresh_gallery()
            self.statusBar().showMessage(
                f"Population loaded: Gen {self.ga.generation}, "
                f"Best fitness: {self.ga.best_fitness:.1f}",
                4000,
            )
            QMessageBox.information(
                self, "Loaded",
                f"Population loaded from:\n{fn}\n\n"
                f"Generation: {self.ga.generation}\n"
                f"Best fitness: {self.ga.best_fitness:.1f}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load:\n{e}")

    # ── View toggles ──────────────────────────────────────────────────────────

    def _toggle_gallery(self) -> None:
        visible = self._gallery_dock.isVisible()
        self._gallery_dock.setVisible(not visible)

    def _toggle_toolbar(self) -> None:
        toolbars = self.findChildren(QToolBar)
        for tb in toolbars:
            tb.setVisible(not tb.isVisible())

    # ── Dialogs ───────────────────────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About Fractal Genetic Art Studio",
            f"<h2>Fractal Genetic Art Studio v{VERSION}</h2>"
            f"<p>An evolutionary fractal tree generator built with PySide6.</p>"
            f"<p>🧬 Genetic algorithm with adaptive mutation<br>"
            f"🎨 9 color palette presets<br>"
            f"✨ Leaf particles &amp; glow effects<br>"
            f"🖼️ Population gallery with thumbnails<br>"
            f"🎞️ GIF &amp; PNG sequence export<br>"
            f"💾 Save/load population state<br>"
            f"🌱 Seed control for reproducibility</p>"
            f"<p style='color: {C['overlay0']};'>"
            f"Built with Python, PySide6, PIL/Pillow, NumPy</p>",
        )

    def _show_shortcuts(self) -> None:
        lines = []
        for name, key in SHORTCUTS.items():
            label = name.replace("_", " ").title()
            lines.append(f"<tr><td><b>{label}</b></td><td>{key}</td></tr>")
        table = "<table>" + "\n".join(lines) + "</table>"
        QMessageBox.information(
            self, "Keyboard Shortcuts",
            f"<h3>Keyboard Shortcuts</h3>{table}",
        )

    # ── Close ─────────────────────────────────────────────────────────────────

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