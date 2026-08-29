"""PyQt6 main window for the TRANSITIONS app — accordion sidebar + full-width plot area."""
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
    QApplication, QComboBox, QDoubleSpinBox, QDialog, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSizePolicy, QSpinBox, QTabWidget, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from ..io.paths import EPV_GRID_PATH, PROCESSED_DIR, RAW_TRACKING_DIR
from ..pipeline import runner
from . import timeline, viewer
from .theme import GlassCard, apply_glass_shadow, wrap_in_glass
from .two_d_analysis import format_time, METRIC_COLS
import pandas as pd

# =====================================================================
# Helpers
# =====================================================================

class _SignalWriter(io.TextIOBase):
    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self._callback = callback
    def write(self, text: str) -> int:
        if text: self._callback(text.rstrip())
        return len(text)
    def flush(self) -> None: return None

class Worker(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str)
    _PROGRESS_RE = re.compile(r"^PROGRESS:\s*(-?\d+)\s*(.*)$")

    def __init__(self, label: str, callback: Callable[[], object]) -> None:
        super().__init__()
        self._label = label
        self._callback = callback

    def _on_output(self, text: str) -> None:
        m = self._PROGRESS_RE.match(text)
        if m: self.progress.emit(int(m.group(1)), m.group(2).strip())
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

class LegendPopup(QDialog):
    """Popup dialog to display the plot legend."""
    def __init__(self, parent=None, title="Legend", shape_categories=None, color_metric=None):
        super().__init__(parent)
        self.setWindowTitle(f"Legend - {title}")
        self.setMinimumSize(400, 500)
        
        layout = QVBoxLayout(self)
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)
        
        if shape_categories:
            shape_label = QLabel("<b>Formation (Shape):</b>")
            layout.addWidget(shape_label)
            shape_list = QListWidget()
            for cat in shape_categories:
                shape_list.addItem(str(cat))
            layout.addWidget(shape_list)
            
        if color_metric:
            color_label = QLabel(f"<b>Color: {color_metric}</b>")
            layout.addWidget(color_label)
            color_info = QLabel("Continuous color scale (viridis colormap)")
            color_info.setStyleSheet("color: #8a9aaa;")
            layout.addWidget(color_info)
            
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

class FigureTab(QWidget):
    """Widget hosting a matplotlib figure + navigation toolbar + details panel."""
    def __init__(self, title: str, figure) -> None:
        super().__init__()
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Left side: Plot container
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        canvas = figure.canvas
        if not isinstance(canvas, FigureCanvasQTAgg):
            canvas = FigureCanvasQTAgg(figure)
        canvas.setParent(self)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Toolbar row with Show Legend button
        toolbar_row = QHBoxLayout()
        toolbar = NavigationToolbar2QT(canvas, self)
        toolbar_row.addWidget(toolbar, 1)
        
        # Add Show Legend button if data exists
        if hasattr(figure, '_legend_data'):
            legend_btn = QPushButton("📋 Show Legend")
            legend_btn.setObjectName("runButton")
            legend_btn.setMinimumHeight(28)
            legend_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            legend_btn.clicked.connect(lambda: self._show_legend_popup(figure))
            toolbar_row.addWidget(legend_btn)
            
        plot_layout.addLayout(toolbar_row)
        plot_layout.addWidget(canvas, 1)
        canvas.draw_idle()
        self.canvas = canvas
        self.title = title
        
        main_layout.addWidget(plot_container, 3)
        
        # Right side: Details Panel
        self.details_panel = QTextEdit()
        self.details_panel.setReadOnly(True)
        self.details_panel.setMinimumWidth(250)
        self.details_panel.setMaximumWidth(350)
        self.details_panel.setStyleSheet(f"""
            QTextEdit {{
                background: rgba(0, 0, 0, 0.22);
                border: 1px solid #2a3a4a;
                border-radius: 9px;
                padding: 10px;
                color: #e0e0e0;
                font-family: "Segoe UI", sans-serif;
                font-size: 12px;
            }}
        """)
        self.details_panel.setHtml("<b>Segment Details</b><br><br>Hover or click a point to see details.")
        main_layout.addWidget(self.details_panel, 1)
        
        # --- Interaction Handlers ---
        self._setup_interactions(figure)

    def _show_legend_popup(self, figure):
        """Open a modal popup with legend information."""
        data = figure._legend_data
        popup = LegendPopup(
            self,
            title=data['title'],
            shape_categories=data['shape_categories'],
            color_metric=data['color_metric']
        )
        popup.exec()

    def _setup_interactions(self, figure):
        if not hasattr(figure, '_segment_records'):
            return
        canvas = self.canvas
        
        def on_hover(event):
            if event.inaxes != figure.axes[0]:
                figure._hover_annot.set_visible(False)
                canvas.draw_idle()
                return
                
            for sc, ids in figure._scatter_map.items():
                cont, ind = sc.contains(event)
                if cont:
                    idx = ind['ind'][0]
                    plot_id = ids[idx]
                    record = figure._segment_records[plot_id]
                    
                    # Update annotation text - safely handle numeric formatting
                    enc = figure._encodings
                    x_val = record.get(METRIC_COLS.get(enc['x_axis'], ''), 'N/A')
                    y_val = record.get(METRIC_COLS.get(enc['y_axis'], ''), 'N/A')
                    
                    # Format values safely - only apply :.2f if value is numeric
                    try:
                        x_formatted = f"{float(x_val):.2f}" if x_val != 'N/A' else 'N/A'
                    except (ValueError, TypeError):
                        x_formatted = str(x_val)
                        
                    try:
                        y_formatted = f"{float(y_val):.2f}" if y_val != 'N/A' else 'N/A'
                    except (ValueError, TypeError):
                        y_formatted = str(y_val)
                    
                    text = (f"<b>{record.get('variant', 'N/A')}</b> ({record.get('family', 'N/A')})\n"
                            f"Time: {format_time(record.get('start_sec', 0))} – {format_time(record.get('end_sec', 0))}\n"
                            f"Dur: {format_time(record.get('duration', 0))}\n"
                            f"{enc['x_axis']}: {x_formatted}\n"
                            f"{enc['y_axis']}: {y_formatted}")
                    
                    figure._hover_annot.set_text(text)
                    figure._hover_annot.set_position((event.xdata, event.ydata))
                    figure._hover_annot.set_visible(True)
                    canvas.draw_idle()
                    return
                    
            figure._hover_annot.set_visible(False)
            canvas.draw_idle()

        def on_click(event):
            if event.inaxes != figure.axes[0] or event.button != 1:
                return
                
            for sc, ids in figure._scatter_map.items():
                cont, ind = sc.contains(event)
                if cont:
                    idx = ind['ind'][0]
                    plot_id = ids[idx]
                    record = figure._segment_records[plot_id]
                    
                    # Highlight point
                    figure._selection_ring.set_offsets([[record[METRIC_COLS.get(figure._encodings['x_axis'], 'mean_width')], 
                                                         record[METRIC_COLS.get(figure._encodings['y_axis'], 'mean_depth')]]])
                    canvas.draw_idle()
                    
                    # Update details panel
                    self._update_details_panel(record)
                    return

        canvas.mpl_connect('motion_notify_event', on_hover)
        canvas.mpl_connect('button_press_event', on_click)

    def _update_details_panel(self, record):
        def fmt(val, decimals=2):
            if val is None or (isinstance(val, float) and pd.isna(val)): return "N/A"
            return f"{val:.{decimals}f}"
            
        html = f"""
        <h3 style="color: #2b8a3e; margin-bottom: 10px;">Segment Details</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td style="color: #8a9aaa; padding: 4px;">Formation:</td><td style="font-weight: bold; padding: 4px;">{record.get('variant', 'N/A')}</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Family:</td><td style="padding: 4px;">{record.get('family', 'N/A')}</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Team:</td><td style="padding: 4px;">{record.get('team', 'N/A').capitalize()}</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Period:</td><td style="padding: 4px;">{record.get('period', 'N/A')}</td></tr>
            <tr><td colspan="2" style="border-top: 1px solid #2a3a4a; margin: 5px 0;"></td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Start:</td><td style="padding: 4px;">{format_time(record.get('start_sec', 0))}</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">End:</td><td style="padding: 4px;">{format_time(record.get('end_sec', 0))}</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Duration:</td><td style="padding: 4px;">{format_time(record.get('duration', 0))}</td></tr>
            <tr><td colspan="2" style="border-top: 1px solid #2a3a4a; margin: 5px 0;"></td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">In Possession:</td><td style="padding: 4px;">{fmt(record.get('in_possession_sec', 0), 1)}s</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Out of Possession:</td><td style="padding: 4px;">{fmt(record.get('out_of_possession_sec', 0), 1)}s</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Loose Ball:</td><td style="padding: 4px;">{fmt(record.get('loose_ball_sec', 0), 1)}s</td></tr>
            <tr><td colspan="2" style="border-top: 1px solid #2a3a4a; margin: 5px 0;"></td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Mean Width:</td><td style="padding: 4px;">{fmt(record.get('mean_width'))} m</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Mean Depth:</td><td style="padding: 4px;">{fmt(record.get('mean_depth'))} m</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Mean Compactness:</td><td style="padding: 4px;">{fmt(record.get('mean_compactness'))} m</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Mean Center X:</td><td style="padding: 4px;">{fmt(record.get('mean_center_x'))}</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Mean Center Y:</td><td style="padding: 4px;">{fmt(record.get('mean_center_y'))}</td></tr>
            <tr><td colspan="2" style="border-top: 1px solid #2a3a4a; margin: 5px 0;"></td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Centroid Velocity:</td><td style="padding: 4px;">{fmt(record.get('mean_centroid_velocity'))} m/s</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Template Displacement:</td><td style="padding: 4px;">{fmt(record.get('mean_template_displacement'))} m</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Mean Confidence:</td><td style="padding: 4px;">{fmt(record.get('mean_confidence'), 4)}</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">Min Confidence:</td><td style="padding: 4px;">{fmt(record.get('min_confidence'), 4)}</td></tr>
            <tr><td colspan="2" style="border-top: 1px solid #2a3a4a; margin: 5px 0;"></td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">N Windows:</td><td style="padding: 4px;">{record.get('n_windows', 'N/A')}</td></tr>
            <tr><td style="color: #8a9aaa; padding: 4px;">N Frames:</td><td style="padding: 4px;">{record.get('n_frames', 'N/A')}</td></tr>
        </table>
        """
        self.details_panel.setHtml(html)

# =====================================================================
# Accordion Sidebar
# =====================================================================

class AccordionSidebar(QWidget):
    SECTIONS: list[tuple[str, str, str]] = [
        ("Home",              "Run the full pipeline, or execute individual steps below.", "▸ Run Full Pipeline"),
        ("Preprocess",        "Process raw tracking data into structured JSONL + metadata.", " Run Preprocess"),
        ("Formations",        "Detect team formations from player positions using template matching.", "▸ Detect Formations"),
        ("EPV + DAS",         "Compute Expected Possession Value and Dangerous Action Sequences.", "▸ Run EPV + DAS"),
        ("Pitch Control",     "Compute pitch control per formation window.", "▸ Compute Pitch Control"),
        ("OBSO",              "Compute Off-Ball Scoring Opportunity per formation window.", "▸ Compute OBSO"),
        ("Timeline",          "Plot formation changes over the match.", " Show Timeline"),
        ("2D Analysis",       "Tactical analysis of formation segments in 2D space.", "▸ Run 2D Analysis"),
        ("Viewer",            "Interactive match viewer with scrubbable slider and playback controls.", " Open Match Viewer"),
    ]

    def __init__(self, callbacks: list[Callable[[], None]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(220)
        self._callbacks = callbacks
        self._sections: list[dict] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("TRANSITIONS")
        title.setStyleSheet(
            "font-size: 16px; font-weight: 800; padding: 16px 14px 10px 14px;"
            "color: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #6cb2ff, stop:1 #9689ff);"
            "background: transparent;"
        )
        layout.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.08); margin: 0 10px;")
        layout.addWidget(sep)

        for idx, (label, desc, btn_text) in enumerate(self.SECTIONS):
            header, body = self._build_section(label, desc, btn_text, idx)
            layout.addWidget(header)
            layout.addWidget(body)
            self._sections.append({
                "header": header,
                "body": body,
                "is_open": False,
            })
        layout.addStretch()

    def _build_section(self, label: str, desc: str,
                       btn_text: str, idx: int) -> tuple[QPushButton, QWidget]:
        header = QPushButton(label)
        header.setObjectName("navHeader")
        header.setCursor(Qt.CursorShape.PointingHandCursor)
        header.setStyleSheet("""
            QPushButton#navHeader {
                text-align: left;
                padding: 9px 14px;
                border: none;
                border-radius: 0;
                font-size: 13px;
                font-weight: 600;
                color: #b0c0d0;
                background: transparent;
            }
            QPushButton#navHeader:hover {
                background: rgba(255, 255, 255, 0.06);
                color: #d0e0f0;
            }
        """)
        header.clicked.connect(lambda: self._toggle(idx))

        body = QWidget()
        body.setVisible(False)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 2, 14, 10)
        body_layout.setSpacing(8)

        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(
            "color: #7a8a9a; font-size: 11px; background: transparent;"
        )
        run_btn = QPushButton(btn_text)
        run_btn.setObjectName("runButton")
        run_btn.setMinimumHeight(32)
        run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        run_btn.clicked.connect(self._callbacks[idx])

        body_layout.addWidget(desc_label)
        body_layout.addWidget(run_btn)
        return header, body

    def _toggle(self, idx: int) -> None:
        target = self._sections[idx]
        target["is_open"] = not target["is_open"]
        target["body"].setVisible(target["is_open"])
        for i, sec in enumerate(self._sections):
            if i != idx and sec["is_open"]:
                sec["is_open"] = False
                sec["body"].setVisible(False)

    def set_enabled(self, enabled: bool) -> None:
        for sec in self._sections:
            sec["header"].setEnabled(enabled)
            for child in sec["body"].findChildren(QPushButton):
                child.setEnabled(enabled)

# =====================================================================
# Top Settings Bar
# =====================================================================

class TopBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

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

# =====================================================================
# Log Panel
# =====================================================================

class LogPanel(QWidget):
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

# =====================================================================
# Main Window
# =====================================================================

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

        self.sidebar = AccordionSidebar(callbacks=[
             self.run_full_pipeline,
             self.run_preprocess,
             self.run_formations,
             self.run_epv,
             self.run_pitch_control,
             self.run_obso,
             self.run_timeline,
             self.run_2d_analysis,
             self.run_viewer,
        ])
        main_layout.addWidget(self.sidebar)

        self._timeline_method = QComboBox()
        self._timeline_method.addItems(["Single line", "Piano roll"])
        self._timeline_method.setMinimumHeight(28)
        self._timeline_method.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        
        self._timeline_granularity = QComboBox()
        self._timeline_granularity.addItems(["Detail (variant)", "Overview (family)"])
        self._timeline_granularity.setMinimumHeight(28)
        self._timeline_granularity.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        
        timeline_body = self.sidebar._sections[6]["body"]
        timeline_body.layout().insertWidget(1, self._timeline_method)
        timeline_body.layout().insertWidget(2, self._timeline_granularity)

        self._2d_controls = self._build_2d_controls()
        analysis_body = self.sidebar._sections[7]["body"]
        analysis_body.layout().insertWidget(1, self._2d_controls)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        main_layout.addWidget(sep)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(14, 14, 14, 10)
        content_layout.setSpacing(12)

        self.top = TopBar()
        content_layout.addWidget(wrap_in_glass(self.top, outer_margins=0))

        self.tabs = QTabWidget()
        self.tabs.addTab(self._empty_tab(), "Results")
        tabs_card = wrap_in_glass(self.tabs, outer_margins=0)
        content_layout.addWidget(tabs_card, 1)

        self.log_panel = LogPanel()
        content_layout.addWidget(wrap_in_glass(self.log_panel, outer_margins=0))

        main_layout.addWidget(content, 1)
        self.statusBar().showMessage("Ready")

        self._match_label = QLabel("")
        self._match_label.setObjectName("statusMatch")
        self.statusBar().addPermanentWidget(self._match_label)
        self.top.match_combo.currentTextChanged.connect(self._update_match_status)
        # Connect match change to update formation dropdown
        self.top.match_combo.currentTextChanged.connect(self._update_formation_dropdown)
        self._update_match_status(self.top.match_id)

    def _empty_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel("Run a step to display plots here.")
        label.setStyleSheet("font-size: 14px; color: #6b7a8a; background: transparent;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return tab

    def _update_formation_dropdown(self, match_id: str) -> None:
        """Populate the Formation dropdown based on the selected match's segments."""
        if not match_id:
            self._2d_formation.clear()
            self._2d_formation.addItem("All")
            return
            
        try:
            folder = Path(self.top.processed_dir) / str(match_id)
            segments_path = folder / "formation_segments.csv"
            
            if segments_path.is_file():
                df = pd.read_csv(segments_path)
                # Clean variant names just like in two_d_analysis
                if 'variant' in df.columns:
                    variants = df['variant'].astype(str).apply(lambda x: x.split('\n')[0].strip()).unique()
                    variants = sorted([v for v in variants if v])
                else:
                    variants = []
                    
                self._2d_formation.blockSignals(True)
                self._2d_formation.clear()
                self._2d_formation.addItem("All")
                self._2d_formation.addItems(variants)
                self._2d_formation.blockSignals(False)
            else:
                self._2d_formation.clear()
                self._2d_formation.addItem("All")
        except Exception:
            # Fallback if reading fails
            self._2d_formation.clear()
            self._2d_formation.addItem("All")

    def _build_2d_controls(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        def make_combo(label_text: str, items: list[str]) -> QComboBox:
            lbl = QLabel(label_text)
            lbl.setObjectName("fieldLabel")
            combo = QComboBox()
            combo.addItems(items)
            combo.setMinimumHeight(28)
            combo.setStyleSheet("font-size: 11px; padding: 2px 4px;")
            layout.addWidget(lbl)
            layout.addWidget(combo)
            return combo

        def make_spin(label_text: str, min_val: int, max_val: int, default: int) -> QSpinBox:
            lbl = QLabel(label_text)
            lbl.setObjectName("fieldLabel")
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            spin.setFixedWidth(80)
            spin.setMinimumHeight(28)
            spin.setStyleSheet("font-size: 11px; padding: 2px 4px;")
            layout.addWidget(lbl)
            layout.addWidget(spin)
            return spin

        self._2d_team = make_combo("TEAM", ["All", "Home", "Away"])
        self._2d_period = make_combo("PERIOD", ["All", "1", "2"])
        self._2d_formation = make_combo("FORMATION", ["All"])
        self._2d_min_duration = make_spin("MIN DURATION (s)", 0, 3600, 0)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.08); margin: 4px 0;")
        layout.addWidget(sep)

        # --- STAGE 4 UPDATE: Added EPV/DAS metrics to dropdowns ---
        metrics_list = [
            "Width", "Depth", "Compactness", "Duration", 
            "Center X", "Center Y", "Elongation", 
            "Centroid Displacement", "Centroid Velocity", "Confidence",
            "Pitch Control", "Home Control", "Away Control",
            "OBSO",
            "Cumulative EPV", "Mean EPV", "EPV / min",
            "DAS Count", "DAS / min"
        ]
        
        self._2d_x_axis = make_combo("X-AXIS", metrics_list)
        self._2d_x_axis.setCurrentText("Width")
        
        self._2d_y_axis = make_combo("Y-AXIS", metrics_list)
        self._2d_y_axis.setCurrentText("Depth")
        
        self._2d_color = make_combo("COLOR", metrics_list)
        self._2d_color.setCurrentText("Compactness")
        
        self._2d_shape = make_combo("SHAPE", ["Formation Family", "Formation"])
        
        layout.addStretch()
        return container
    
    def _match_id(self) -> str | None:
        match_id = self.top.match_id
        if not match_id:
            QMessageBox.warning(self, "Missing match ID", "Enter a match ID in the top bar.")
            return None
        return match_id

    def _set_running(self, running: bool, label: str = "") -> None:
        for btn in self.findChildren(QPushButton):
            if btn.objectName() == "runButton":
                btn.setEnabled(not running)
        self.sidebar.set_enabled(not running)
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

    def run_pitch_control(self) -> None:
        match_id = self._match_id()
        if match_id is None: return
        from ..pipeline.runner import pitch_control_figure
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
        fig, df = pitch_control_figure(match_id, processed_dir=self.top.processed_dir)
        self._show_result("Pitch Control", fig)

    def run_obso(self) -> None:
        match_id = self._match_id()
        if match_id is None: return
        from ..pipeline.runner import obso_figure
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
        fig, df = obso_figure(match_id, processed_dir=self.top.processed_dir, epv_grid_path=self.top.epv_grid)
        self._show_result("OBSO", fig)

    def run_preprocess(self) -> None:
        match_id = self.top.match_id or None
        raw_dir = self.top.raw_dir or None
        self._start_worker("Preprocess", lambda: runner.preprocess_all_matches(match_id=match_id, raw_tracking_dir=raw_dir))

    def run_formations(self) -> None:
        match_id = self._match_id()
        if match_id is None: return
        self._start_worker("Formations", lambda: runner.detect_formations([match_id], processed_dir=self.top.processed_dir, window_seconds=self.top.window_seconds, stride_seconds=self.top.stride_seconds))

    def run_epv(self) -> None:
        match_id = self._match_id()
        if match_id is None: return
        self._start_worker("EPV + DAS", lambda: runner.run_epv(match_id, self.top.processed_dir, self.top.epv_grid))

    def run_timeline(self) -> None:
        match_id = self._match_id()
        if match_id is None: return
        import matplotlib.pyplot as plt
        method = "voted" if self._timeline_method.currentIndex() == 0 else "all_formations"
        granularity = "variant" if self._timeline_granularity.currentIndex() == 0 else "family"
        plt.switch_backend("QtAgg")
        timeline.plot_formation_timeline(match_id, self.top.processed_dir, method=method, granularity=granularity)
        plt.show(block=False)

    def run_2d_analysis(self) -> None:
        match_id = self._match_id()
        if match_id is None: return
        
        import matplotlib.pyplot as plt
        plt.switch_backend("QtAgg")
        
        from .two_d_analysis import plot_2d_analysis
        
        filters = {
            "team": self._2d_team.currentText(),
            "period": self._2d_period.currentText(),
            "formation": self._2d_formation.currentText(),
            "min_duration": self._2d_min_duration.value(),
        }
        
        encodings = {
            "x_axis": self._2d_x_axis.currentText(),
            "y_axis": self._2d_y_axis.currentText(),
            "color": self._2d_color.currentText(),
            "shape": self._2d_shape.currentText(),
        }
        
        try:
            fig = plot_2d_analysis(match_id, self.top.processed_dir, filters, encodings)
            self._show_result("2D Analysis", fig)
        except Exception as e:
            import traceback
            self.log_panel.append(traceback.format_exc())
            QMessageBox.critical(self, "2D Analysis Failed", str(e))

    def run_viewer(self) -> None:
        match_id = self._match_id()
        if match_id is None: return
        import matplotlib.pyplot as plt
        plt.switch_backend("QtAgg")
        viewer.run_app(match_id, self.top.processed_dir, self.top.speed, show=True, block=False)

    def run_full_pipeline(self) -> None:
        match_id = self._match_id()
        if match_id is None: return
        processed_dir = self.top.processed_dir
        raw_dir = self.top.raw_dir or None
        epv_grid = self.top.epv_grid
        window_seconds = self.top.window_seconds
        stride_seconds = self.top.stride_seconds

        def _run() -> tuple:
            runner.preprocess_all_matches(match_id=match_id, raw_tracking_dir=raw_dir)
            runner.detect_formations([match_id], processed_dir=processed_dir, window_seconds=window_seconds, stride_seconds=stride_seconds)
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