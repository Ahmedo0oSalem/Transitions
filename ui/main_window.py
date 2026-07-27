"""PyQt6 main window for the TRANSITIONS app — sidebar nav + glassmorphism dark theme."""

from __future__ import annotations

import contextlib
import io
import re
import sys
import traceback
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from ..io.paths import EPV_GRID_PATH, PROCESSED_DIR, RAW_TRACKING_DIR
from ..pipeline import runner
from . import viewer
from .theme import GlassCard, apply_glass_shadow, wrap_in_glass


# ====================================================================
# Helpers — unchanged from the original layout
# ====================================================================

class _SignalWriter(io.TextIOBase):
    """Redirect print output from long-running jobs into the UI log."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self._callback = callback

    def write(self, text: str) -> int:
        if text:
            self._callback(text.rstrip())
        return len(text)

    def flush(self) -> None:
        return None


class Worker(QObject):
    """Runs a callable in a background thread.

    Pipeline steps can report progress by printing lines like::

        PROGRESS: 50 Training model...

    Values: -1 = indeterminate, 0-100 = determinate.
    """

    log = pyqtSignal(str)
    progress = pyqtSignal(int, str)  # value, message
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str)

    _PROGRESS_RE = re.compile(r"^PROGRESS:\s*(-?\d+)\s*(.*)$")

    def __init__(self, label: str, callback: Callable[[], object]) -> None:
        super().__init__()
        self._label = label
        self._callback = callback

    def _on_output(self, text: str) -> None:
        m = self._PROGRESS_RE.match(text)
        if m:
            self.progress.emit(int(m.group(1)), m.group(2).strip())
        self.log.emit(text)

    def run(self) -> None:
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")
        writer = _SignalWriter(self._on_output)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                result = self._callback()
            self.finished.emit(self._label, result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class FigureTab(QWidget):
    """Widget hosting a matplotlib figure + navigation toolbar.

    The canvas grows to fill available space, but never shrinks below a
    legible minimum — below that, a scrollbar appears instead of the
    plot becoming unreadable.
    """

    # Tune these to taste — this is the floor below which we scroll
    # instead of shrinking further.
    MIN_CANVAS_WIDTH = 480
    MIN_CANVAS_HEIGHT = 320

    def __init__(self, title: str, figure) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        canvas = figure.canvas
        if not isinstance(canvas, FigureCanvasQTAgg):
            canvas = FigureCanvasQTAgg(figure)
        canvas.setParent(self)

        canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        # Floor, not a lock: canvas is free to grow past this, but won't
        # shrink below it. Below this size the QScrollArea scrolls instead.
        canvas.setMinimumSize(self.MIN_CANVAS_WIDTH, self.MIN_CANVAS_HEIGHT)

        toolbar = NavigationToolbar2QT(canvas, self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # <-- key: canvas fills viewport when room exists
        scroll.setWidget(canvas)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        layout.addWidget(toolbar)
        layout.addWidget(scroll, 1)

        canvas.draw_idle()
        self.canvas = canvas
        self.title = title


# ====================================================================
# Navigation Sidebar
# ====================================================================

class NavSidebar(QListWidget):
    PAGE_LABELS = [
        "🏠  Home",
        "⚙️  Preprocess",
        "🧩  Formations",
        "📈  EPV + DAS",
        "🕒  Timeline",
        "🎬  Viewer",
        "🚀  Pipeline",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(180)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for label in self.PAGE_LABELS:
            self.addItem(label)
        self.setCurrentRow(0)


# ====================================================================
# Top Settings Bar
# ====================================================================

class TopBar(QWidget):
    """Always-visible bar: match selector, paths, shared parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        # ---- Row 1: Match ID + numeric params ----
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        self.match_label = QLabel("MATCH ID")
        self.match_label.setObjectName("fieldLabel")
        self.match_combo = QComboBox()
        self.match_combo.setEditable(True)
        self.match_combo.setPlaceholderText("e.g. 12345")
        self.match_combo.setMinimumWidth(140)

        self.refresh_btn = QToolButton()
        self.refresh_btn.setText("↻")
        self.refresh_btn.setToolTip("Scan processed directory for available matches")

        self.window_label = QLabel("WINDOW (s)")
        self.window_label.setObjectName("fieldLabel")
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 7200)
        self.window_spin.setValue(300)
        self.window_spin.setFixedWidth(80)

        self.stride_label = QLabel("STRIDE (s)")
        self.stride_label.setObjectName("fieldLabel")
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(0, 7200)
        self.stride_spin.setSpecialValueText("default")
        self.stride_spin.setValue(0)
        self.stride_spin.setFixedWidth(80)

        self.speed_label = QLabel("SPEED")
        self.speed_label.setObjectName("fieldLabel")
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 25.0)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setFixedWidth(70)

        row1.addWidget(self.match_label)
        row1.addWidget(self.match_combo, 1)
        row1.addWidget(self.refresh_btn)
        row1.addSpacing(14)
        row1.addWidget(self.window_label)
        row1.addWidget(self.window_spin)
        row1.addWidget(self.stride_label)
        row1.addWidget(self.stride_spin)
        row1.addWidget(self.speed_label)
        row1.addWidget(self.speed_spin)
        row1.addStretch()

        # ---- Row 2: Path fields ----
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        self.processed_edit = QLineEdit(str(PROCESSED_DIR))
        self.raw_edit = QLineEdit(str(RAW_TRACKING_DIR))
        self.epv_edit = QLineEdit(str(EPV_GRID_PATH))

        processed_label = QLabel("PROCESSED")
        processed_label.setObjectName("fieldLabel")
        raw_label = QLabel("RAW")
        raw_label.setObjectName("fieldLabel")
        epv_label = QLabel("EPV GRID")
        epv_label.setObjectName("fieldLabel")

        row2.addWidget(processed_label)
        row2.addWidget(self.processed_edit, 3)
        self._browse_btn(row2, self.processed_edit, directory=True)
        row2.addSpacing(12)
        row2.addWidget(raw_label)
        row2.addWidget(self.raw_edit, 3)
        self._browse_btn(row2, self.raw_edit, directory=True)
        row2.addSpacing(12)
        row2.addWidget(epv_label)
        row2.addWidget(self.epv_edit, 2)
        self._browse_btn(row2, self.epv_edit, directory=False)
        row2.addStretch()

        outer.addLayout(row1)
        outer.addLayout(row2)

        self.refresh_btn.clicked.connect(self.refresh_matches)
        self.processed_edit.textChanged.connect(lambda _: self.refresh_matches())

        self.refresh_matches()

    # ---- helpers ----

    def _browse_btn(self, parent_layout: QHBoxLayout, line_edit: QLineEdit, directory: bool) -> None:
        btn = QToolButton()
        btn.setText("…")
        btn.setToolTip("Browse for directory" if directory else "Browse for file")
        btn.clicked.connect(lambda: self._browse(line_edit, directory))
        parent_layout.addWidget(btn)

    def _browse(self, line_edit: QLineEdit, directory: bool) -> None:
        if directory:
            selected = QFileDialog.getExistingDirectory(self, "Choose directory", line_edit.text())
        else:
            selected, _ = QFileDialog.getOpenFileName(self, "Choose file", line_edit.text())
        if selected:
            line_edit.setText(selected)

    def refresh_matches(self) -> None:
        """Scan *processed-dir* for subdirectories with metadata.json."""
        current = self.match_combo.currentText().strip()
        self.match_combo.clear()
        try:
            processed = Path(self.processed_edit.text().strip())
            if processed.is_dir():
                for p in sorted(processed.iterdir()):
                    if p.is_dir() and (p / "metadata.json").is_file():
                        self.match_combo.addItem(p.name)
        except OSError:
            pass
        if current:
            self.match_combo.setCurrentText(current)

    # ---- read-only properties used by run methods ----

    @property
    def match_id(self) -> str:
        return self.match_combo.currentText().strip()

    @property
    def processed_dir(self) -> str:
        return self.processed_edit.text().strip()

    @property
    def raw_dir(self) -> str:
        return self.raw_edit.text().strip()

    @property
    def epv_grid(self) -> str:
        return self.epv_edit.text().strip()

    @property
    def window_seconds(self) -> int:
        return self.window_spin.value()

    @property
    def stride_seconds(self) -> int | None:
        v = self.stride_spin.value()
        return v if v > 0 else None

    @property
    def speed(self) -> float:
        return self.speed_spin.value()


# ====================================================================
# Log Panel
# ====================================================================

class LogPanel(QWidget):
    """Bottom panel: progress bar + scrollable log output."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("RUN LOG")
        title.setObjectName("logTitle")
        header.addWidget(title)
        header.addStretch()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedWidth(64)
        header.addWidget(self.clear_btn)
        layout.addLayout(header)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedHeight(14)
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(60)
        self.log.setMaximumHeight(180)
        layout.addWidget(self.log)

        self.clear_btn.clicked.connect(self.log.clear)

    def append(self, text: str) -> None:
        if not text:
            return
        # HTML-escape and colourise
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if text.startswith(">"):
            colour = "#4a9eff"
        elif text.startswith("Done:"):
            colour = "#2ecc71"
        elif any(x in text for x in ("Error", "Failed", "Traceback", "traceback")):
            colour = "#e74c3c"
        elif "Warning" in text:
            colour = "#f39c12"
        else:
            colour = "#b0c0d0"
        html = (
            f'<span style="color:{colour}; white-space:pre;'
            f' font-family:Consolas,Courier New,monospace; font-size:11px;">'
            f"{escaped}</span><br>"
        )
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertHtml(html)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def update_progress(self, value: int, message: str) -> None:
        """value: -1 for indeterminate, 0-100 for determinate."""
        if value < 0:
            self.progress.setRange(0, 0)
            self.progress.setFormat(message or "Running...")
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(value)
            self.progress.setFormat(message or f"{value}%")
        self.progress.setVisible(True)

    def show_progress(self, running: bool) -> None:
        self.progress.setVisible(running)
        if running:
            self.progress.setRange(0, 0)
            self.progress.setFormat("Running...")
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("")


# ====================================================================
# Main Window
# ====================================================================

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TRANSITIONS")
        self.resize(1360, 840)
        self._thread: QThread | None = None
        self._worker: Worker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---- Left sidebar ----
        self.nav = NavSidebar()
        main_layout.addWidget(self.nav)

        # ---- Vertical separator ----
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        main_layout.addWidget(sep)

        # ---- Right content area ----
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(14, 14, 14, 10)
        content_layout.setSpacing(12)

        # Top bar, wrapped in a glass card
        self.top = TopBar()
        content_layout.addWidget(wrap_in_glass(self.top, outer_margins=0))

        # Stacked widget (one page per sidebar item)
        self.stack = QStackedWidget()
        self._build_pages()
        content_layout.addWidget(self.stack, 0)

        # Results tabs, wrapped in a glass card
        self.tabs = QTabWidget()
        self.tabs.addTab(self._empty_tab(), "Results")
        tabs_card = wrap_in_glass(self.tabs, outer_margins=0)
        content_layout.addWidget(tabs_card, 1)

        # Log panel, wrapped in a glass card
        self.log_panel = LogPanel()
        content_layout.addWidget(wrap_in_glass(self.log_panel, outer_margins=0))

        main_layout.addWidget(content, 1)

        self.statusBar().showMessage("Ready")

        # Wire sidebar to stacked widget
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)

        # Track match ID in status bar
        self._match_label = QLabel("")
        self._match_label.setObjectName("statusMatch")
        self.statusBar().addPermanentWidget(self._match_label)
        self.top.match_combo.currentTextChanged.connect(self._update_match_status)
        self._update_match_status(self.top.match_id)

    # ---- page builders ----

    def _build_pages(self) -> None:
        pages = [
            self._build_home_page(),
            self._build_page(
                "Preprocess",
                "Process raw tracking data into structured JSONL + metadata.",
                "▸ Run Preprocess",
                self.run_preprocess,
            ),
            self._build_page(
                "Formations",
                "Detect team formations from player positions using template matching.",
                "▸ Detect Formations",
                self.run_formations,
            ),
            self._build_page(
                "EPV + DAS",
                "Compute Expected Possession Value and Dangerous Action Sequences.",
                "▸ Run EPV + DAS",
                self.run_epv,
            ),
            self._build_page(
                "Timeline",
                "Plot formation changes over the match, one bar per possession sequence.",
                "▸ Show Timeline",
                self.run_timeline,
            ),
            self._build_page(
                "Viewer",
                "Interactive match viewer with scrubbable slider and playback controls.",
                "▸ Open Match Viewer",
                self.run_viewer,
            ),
            self._build_pipeline_page(),
        ]
        for page in pages:
            self.stack.addWidget(page)

    def _build_page(self, title: str, description: str,
                    button_text: str, callback: Callable) -> QWidget:
        card = GlassCard(margins=(20, 18, 20, 18), spacing=10)
        layout = card.body

        title_label = QLabel(title)
        title_label.setStyleSheet(
            "font-size: 19px; font-weight: 700; background: transparent;"
        )
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(
            "color: #8a9aaa; font-size: 12.5px; background: transparent;"
        )

        btn = QPushButton(button_text)
        btn.setObjectName("runButton")
        btn.setMinimumHeight(38)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(callback)

        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addSpacing(6)
        layout.addWidget(btn)
        layout.addStretch()

        # give the shadow room to breathe
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.addWidget(card)
        return outer

    def _build_home_page(self) -> QWidget:
        card = GlassCard(margins=(28, 26, 28, 26), spacing=6)
        layout = card.body

        title = QLabel("TRANSITIONS")
        title.setStyleSheet(
            "font-size: 30px; font-weight: 800; background: transparent;"
            "color: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #6cb2ff, stop:1 #9689ff);"
        )
        subtitle = QLabel("Football Analytics Framework")
        subtitle.setStyleSheet(
            "font-size: 14px; color: #8a9aaa; background: transparent;"
        )

        steps = QLabel(
            "Workflow:\n"
            "  1. Enter a match ID and paths in the top bar\n"
            "  2. Run Preprocess to prepare the data\n"
            "  3. Detect Formations to identify team shapes\n"
            "  4. Run EPV + DAS for expected-value analysis\n"
            "  5. Show Timeline or Open Match Viewer to explore results\n\n"
            "Or click Run Full Pipeline to execute all steps at once."
        )
        steps.setWordWrap(True)
        steps.setStyleSheet(
            "color: #b0c0d0; font-size: 12.5px; margin-top: 10px;"
            "background: transparent; line-height: 150%;"
        )

        quick_btn = QPushButton("▸ Run Full Pipeline")
        quick_btn.setObjectName("runButton")
        quick_btn.setMinimumHeight(42)
        quick_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        quick_btn.clicked.connect(self.run_full_pipeline)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(10)
        layout.addWidget(steps)
        layout.addSpacing(14)
        layout.addWidget(quick_btn)
        layout.addStretch()

        try:
            from importlib.metadata import version
            ver = version("TRANSITIONS")
        except Exception:
            ver = "0.1.0"
        ver_label = QLabel(f"v{ver}")
        ver_label.setStyleSheet(
            "color: #4a5a6a; font-size: 10px; background: transparent;"
        )
        layout.addWidget(ver_label)

        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.addWidget(card)
        return outer

    def _build_pipeline_page(self) -> QWidget:
        card = GlassCard(margins=(20, 18, 20, 18), spacing=10)
        layout = card.body

        title = QLabel("Full Pipeline")
        title.setStyleSheet(
            "font-size: 19px; font-weight: 700; background: transparent;"
        )
        desc = QLabel(
            "Run all stages in sequence: Preprocess → Detect Formations → "
            "EPV + DAS → Timeline. Uses the settings from the top bar."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(
            "color: #8a9aaa; font-size: 12.5px; background: transparent;"
        )

        btn = QPushButton("▸ Run Full Pipeline")
        btn.setObjectName("runButton")
        btn.setMinimumHeight(40)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self.run_full_pipeline)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addSpacing(6)
        layout.addWidget(btn)
        layout.addStretch()

        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.addWidget(card)
        return outer

    # ---- empty tab placeholder ----

    def _empty_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel("Run a step to display plots here.")
        label.setStyleSheet("font-size: 14px; color: #6b7a8a; background: transparent;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return tab

    # ---- helpers ----

    def _match_id(self) -> str | None:
        match_id = self.top.match_id
        if not match_id:
            QMessageBox.warning(self, "Missing match ID",
                                "Enter a match ID in the top bar.")
            return None
        return match_id

    def _set_running(self, running: bool, label: str = "") -> None:
        for btn in self.findChildren(QPushButton):
            if btn.objectName() == "runButton":
                btn.setEnabled(not running)
        self.nav.setEnabled(not running)
        self.statusBar().showMessage(label if running else "Ready")
        self.log_panel.show_progress(running)

    def _start_worker(self, label: str, callback: Callable[[], object]) -> None:
        if self._thread is not None:
            return
        self.log_panel.append(f"> {label}")
        self._set_running(True, label)
        self._thread = QThread(self)
        self._worker = Worker(label, callback)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_panel.append)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._job_finished)
        self._worker.failed.connect(self._job_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    def _cleanup_worker(self) -> None:
        self._set_running(False)
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    def _update_match_status(self, match_id: str) -> None:
        if match_id:
            self._match_label.setText(f"●  Match: {match_id}")
            self._match_label.setStyleSheet("color: #2ecc71; font-size: 11px;")
        else:
            self._match_label.setText("")
            self._match_label.setStyleSheet("")

    def _on_progress(self, value: int, message: str) -> None:
        self.log_panel.update_progress(value, message)
        if message:
            self.statusBar().showMessage(message)

    def _job_finished(self, label: str, result: object) -> None:
        self.log_panel.append(f"Done: {label}")
        self._show_result(label, result)

    def _job_failed(self, details: str) -> None:
        self.log_panel.append(details)
        QMessageBox.critical(self, "Run failed", details)

    def _show_result(self, label: str, result: object) -> None:
        figures = []
        if result is None:
            return
        if isinstance(result, tuple):
            figures = list(result)
        elif hasattr(result, "canvas"):
            figures = [result]
        else:
            return
        if self.tabs.count() == 1 and self.tabs.tabText(0) == "Results":
            self.tabs.removeTab(0)
        for index, figure in enumerate(figures, start=1):
            title = label if len(figures) == 1 else f"{label} {index}"
            self.tabs.addTab(FigureTab(title, figure), title)
            self.tabs.setCurrentIndex(self.tabs.count() - 1)

    # ---- run methods ----

    def run_preprocess(self) -> None:
        match_id = self.top.match_id or None
        raw_dir = self.top.raw_dir or None
        self._start_worker(
            "Preprocess",
            lambda: runner.preprocess_all_matches(match_id=match_id,
                                                  raw_tracking_dir=raw_dir),
        )

    def run_formations(self) -> None:
        match_id = self._match_id()
        if match_id is None:
            return
        self._start_worker(
            "Formations",
            lambda: runner.detect_formations(
                [match_id],
                processed_dir=self.top.processed_dir,
                window_seconds=self.top.window_seconds,
                stride_seconds=self.top.stride_seconds,
            ),
        )

    def run_epv(self) -> None:
        match_id = self._match_id()
        if match_id is None:
            return
        self._start_worker(
            "EPV + DAS",
            lambda: runner.run_epv(match_id, self.top.processed_dir,
                                   self.top.epv_grid),
        )

    def run_timeline(self) -> None:
        match_id = self._match_id()
        if match_id is None:
            return
        self._start_worker(
            "Timeline",
            lambda: runner.run_timeline(match_id, self.top.processed_dir),
        )

    def run_viewer(self) -> None:
        match_id = self._match_id()
        if match_id is None:
            return
        import matplotlib.pyplot as plt

        plt.switch_backend("QtAgg")
        viewer.run_app(
            match_id,
            self.top.processed_dir,
            self.top.speed,
            show=True,
            block=False,
        )

    def run_full_pipeline(self) -> None:
        match_id = self._match_id()
        if match_id is None:
            return
        processed_dir = self.top.processed_dir
        raw_dir = self.top.raw_dir or None
        epv_grid = self.top.epv_grid
        window_seconds = self.top.window_seconds
        stride_seconds = self.top.stride_seconds

        def _run() -> tuple:
            runner.preprocess_all_matches(match_id=match_id,
                                          raw_tracking_dir=raw_dir)
            runner.detect_formations(
                [match_id],
                processed_dir=processed_dir,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
            )
            epv_figures = runner.run_epv(match_id, processed_dir, epv_grid)
            timeline_figure = runner.run_timeline(match_id, processed_dir)
            return (*epv_figures, timeline_figure)

        self._start_worker("Full Pipeline", _run)


def main() -> None:
    app = QApplication(sys.argv)
    from .theme import setup_dark

    setup_dark(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()