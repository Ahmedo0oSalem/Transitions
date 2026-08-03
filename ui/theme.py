"""Dark theme configuration for TRANSITIONS.

Usage:
    from Transitions.ui.theme import setup_dark
    app = QApplication(sys.argv)
    setup_dark(app)
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QVBoxLayout,
    QWidget,
)

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Colour palette — single source of truth for both Qt widgets and matplotlib
# figures.  Import these constants in viewer.py / timeline.py etc. instead of
# duplicating colour strings.
#
# NOTE: these are unchanged from the original theme so every existing import
# elsewhere in the codebase keeps working. The glassmorphism look comes from
# the QSS in resources/style.qss (built from these same colours) plus the
# GlassCard / apply_glass_shadow helpers below — not from changing the base
# palette values.
# ---------------------------------------------------------------------------

# Core surfaces
WINDOW_BG = "#0b1a2a"
BASE_BG = "#112238"
ALT_BG = "#162d45"
BUTTON_BG = "#1a3050"
SURFACE_BG = "#1e3555"

# Text
TEXT_PRIMARY = "#e0e0e0"
TEXT_SECONDARY = "#8a9aaa"
TEXT_DISABLED = "#4a5a6a"

# Accent
ACCENT_GREEN = "#2b8a3e"
ACCENT_GREEN_HOVER = "#34a34a"
ACCENT_BLUE = "#1f77b4"
ACCENT_GOLD = "#f1c40f"

# Borders / lines
BORDER = "#2a3a4a"
GRID = "#2a3a4a"
BASELINE = "#3a4a5a"

# Figure colours (matplotlib)
FIG_FACE = WINDOW_BG
AXES_FACE = "#112233"
AXES_EDGE = GRID
TITLE_COLOR = TEXT_PRIMARY
LABEL_COLOR = TEXT_PRIMARY
TICK_COLOR = TEXT_SECONDARY
LEGEND_FACE = "#0d1b2a"

# Match viewer colours
HOME_COLOR = "#d62728"
AWAY_COLOR = ACCENT_BLUE
BALL_COLOR = "#f7e017"
POSSESSION_RING_COLOR = "#2ecc71"

# Goal marker colour (used by timeline)
GOAL_COLOR = ACCENT_GOLD


# ---------------------------------------------------------------------------
# Glassmorphism additions — translucency tokens + corner radii.
# These are new; nothing else in the codebase depends on them yet, so it's
# safe to tune the alpha/blur values to taste.
# ---------------------------------------------------------------------------

GLASS_FILL = "rgba(255, 255, 255, 0.045)"
GLASS_FILL_HOVER = "rgba(255, 255, 255, 0.085)"
GLASS_BORDER = "rgba(255, 255, 255, 0.11)"
GLASS_BORDER_HOVER = "rgba(255, 255, 255, 0.20)"
GLASS_SHADOW_ALPHA = 150  # 0-255, used by apply_glass_shadow()

RADIUS = 16
RADIUS_SM = 9


def apply_glass_shadow(
    widget: QWidget, blur: int = 36, y_offset: int = 10,
    alpha: int = GLASS_SHADOW_ALPHA,
) -> None:
    """Attach a soft drop shadow to fake glass elevation.

    The widget needs breathing room around it (margins on its parent
    layout) or the shadow will be clipped — use wrap_in_glass() for that.
    """
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, y_offset)
    effect.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(effect)


class GlassCard(QFrame):
    """A translucent, rounded, elevated panel — the base glass surface.

    Styled via QSS selector `#glassCard` (see resources/style.qss).
    """

    def __init__(self, parent: QWidget | None = None, *, shadow: bool = True,
                 margins: tuple[int, int, int, int] = (16, 16, 16, 16),
                 spacing: int = 10) -> None:
        super().__init__(parent)
        self.setObjectName("glassCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)
        if shadow:
            apply_glass_shadow(self)

    @property
    def body(self) -> QVBoxLayout:
        return self.layout()  # type: ignore[return-value]


def wrap_in_glass(widget: QWidget, *, outer_margins: int = 6) -> QWidget:
    """Wrap an existing widget in a glass card, preserving its content.

    Returns an outer container (with margin so the shadow isn't clipped)
    holding a GlassCard holding `widget`.
    """
    outer = QWidget()
    outer_layout = QVBoxLayout(outer)
    outer_layout.setContentsMargins(
        outer_margins, outer_margins, outer_margins, outer_margins
    )
    card = GlassCard(margins=(0, 0, 0, 0), spacing=0)
    card.body.addWidget(widget)
    outer_layout.addWidget(card)
    return outer


# ---------------------------------------------------------------------------
# QPalette builder
# ---------------------------------------------------------------------------

def _make_palette() -> QPalette:
    p = QPalette()

    # Helper: set a role for all states, then override specific states
    def _set(role, base_hex, disabled_hex=None):
        col = QColor(base_hex)
        dis = QColor(disabled_hex) if disabled_hex else col.darker(150)
        p.setColor(QPalette.ColorGroup.Active, role, col)
        p.setColor(QPalette.ColorGroup.Inactive, role, col)
        p.setColor(QPalette.ColorGroup.Disabled, role, dis)

    _set(QPalette.ColorRole.Window, WINDOW_BG)
    _set(QPalette.ColorRole.WindowText, TEXT_PRIMARY, TEXT_DISABLED)
    _set(QPalette.ColorRole.Base, BASE_BG)
    _set(QPalette.ColorRole.AlternateBase, ALT_BG)
    _set(QPalette.ColorRole.ToolTipBase, SURFACE_BG)
    _set(QPalette.ColorRole.ToolTipText, TEXT_PRIMARY)
    _set(QPalette.ColorRole.Text, TEXT_PRIMARY, TEXT_DISABLED)
    _set(QPalette.ColorRole.Button, BUTTON_BG)
    _set(QPalette.ColorRole.ButtonText, TEXT_PRIMARY, TEXT_DISABLED)
    _set(QPalette.ColorRole.BrightText, "#ffffff")
    _set(QPalette.ColorRole.Link, ACCENT_BLUE)
    _set(QPalette.ColorRole.Highlight, ACCENT_GREEN)
    _set(QPalette.ColorRole.HighlightedText, "#ffffff")

    return p


# ---------------------------------------------------------------------------
# matplotlib rcParams
# ---------------------------------------------------------------------------

def _setup_mpl() -> None:
    """Apply dark theme defaults to matplotlib rcParams."""
    mpl.rcParams.update({
        "figure.facecolor": FIG_FACE,
        "figure.edgecolor": AXES_EDGE,
        "axes.facecolor": AXES_FACE,
        "axes.edgecolor": AXES_EDGE,
        "axes.labelcolor": LABEL_COLOR,
        "axes.titlecolor": TITLE_COLOR,
        "axes.titlelocation": "left",
        "text.color": TEXT_PRIMARY,
        "grid.color": GRID,
        "grid.alpha": 0.4,
        "xtick.color": TICK_COLOR,
        "ytick.color": TICK_COLOR,
        "legend.facecolor": LEGEND_FACE,
        "legend.edgecolor": BORDER,
        "legend.labelcolor": LABEL_COLOR,
        "legend.framealpha": 0.85,
        "savefig.facecolor": FIG_FACE,
        "savefig.edgecolor": AXES_EDGE,
    })


# ---------------------------------------------------------------------------
# QSS loader
# ---------------------------------------------------------------------------

def _qss_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "style.qss"


def _load_qss() -> str:
    """Load QSS from resources/style.qss if present, else fall back to the
    built-in glassmorphism stylesheet generated from the palette above."""
    path = _qss_path()
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return _build_glass_qss()


def _build_glass_qss() -> str:
    """Generate the glassmorphism stylesheet from the named colour
    constants, so the app looks right even before resources/style.qss
    is dropped in on disk."""
    return f"""
QMainWindow {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {WINDOW_BG}, stop:1 {BASE_BG});
}}

QWidget {{
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}

/* ---- Sidebar ---- */
QListWidget {{
    background: {GLASS_FILL};
    border: none;
    border-right: 1px solid {GLASS_BORDER};
    padding: 18px 8px;
    outline: none;
}}
QListWidget::item {{
    padding: 10px 12px;
    margin: 3px 2px;
    border-radius: {RADIUS_SM}px;
    color: {TEXT_SECONDARY};
}}
QListWidget::item:hover {{
    background: {GLASS_FILL_HOVER};
    color: {TEXT_PRIMARY};
}}
QListWidget::item:selected {{
    background: rgba(43, 138, 62, 0.22);
    color: {ACCENT_GREEN_HOVER};
    border: 1px solid rgba(43, 138, 62, 0.45);
    font-weight: 600;
}}

/* ---- Frames / separators ---- */
QFrame {{
    background: transparent;
    border: none;
}}
QFrame[frameShape="4"] {{
    background: {GLASS_BORDER};
    max-width: 1px;
}}
QFrame[frameShape="5"] {{
    background: {GLASS_BORDER};
    max-height: 1px;
}}
#topSeparator {{
    background: {GLASS_BORDER};
    max-height: 1px;
}}

/* ---- Glass card surface ---- */
#glassCard {{
    background: {GLASS_FILL};
    border: 1px solid {GLASS_BORDER};
    border-radius: {RADIUS}px;
}}

#fieldLabel {{
    color: {TEXT_SECONDARY};
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}

#logTitle {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

/* ---- Inputs ---- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid {GLASS_BORDER};
    border-radius: {RADIUS_SM}px;
    padding: 5px 10px;
    selection-background-color: rgba(43, 138, 62, 0.35);
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border: 1px solid {GLASS_BORDER_HOVER};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT_GREEN_HOVER};
    background: {GLASS_FILL_HOVER};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {BASE_BG};
    border: 1px solid {GLASS_BORDER};
    border-radius: {RADIUS_SM}px;
    selection-background-color: rgba(43, 138, 62, 0.35);
    outline: none;
    padding: 4px;
}}

/* ---- Buttons ---- */
QPushButton, QToolButton {{
    background: {GLASS_FILL_HOVER};
    border: 1px solid {GLASS_BORDER};
    border-radius: {RADIUS_SM}px;
    padding: 6px 12px;
    color: {TEXT_PRIMARY};
}}
QPushButton:hover, QToolButton:hover {{
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid {GLASS_BORDER_HOVER};
}}
QPushButton:pressed, QToolButton:pressed {{
    background: rgba(255, 255, 255, 0.04);
}}
QPushButton:disabled {{
    color: {TEXT_DISABLED};
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

#runButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT_GREEN}, stop:1 {ACCENT_GREEN_HOVER});
    border: none;
    border-radius: {RADIUS_SM}px;
    color: #ffffff;
    font-weight: 600;
    padding: 8px 16px;
}}
#runButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT_GREEN_HOVER}, stop:1 #3fbf5a);
}}
#runButton:pressed {{
    background: {ACCENT_GREEN};
}}
#runButton:disabled {{
    background: rgba(255, 255, 255, 0.06);
    color: {TEXT_DISABLED};
}}

/* ---- Tabs ---- */
QTabWidget::pane {{
    background: {GLASS_FILL};
    border: 1px solid {GLASS_BORDER};
    border-radius: {RADIUS}px;
    top: -1px;
    padding: 4px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_SECONDARY};
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: {RADIUS_SM}px;
    border-top-right-radius: {RADIUS_SM}px;
}}
QTabBar::tab:selected {{
    background: {GLASS_FILL_HOVER};
    color: {ACCENT_GREEN_HOVER};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT_PRIMARY};
}}

/* ---- Text edit / log ---- */
QTextEdit {{
    background: rgba(0, 0, 0, 0.22);
    border: 1px solid {GLASS_BORDER};
    border-radius: {RADIUS_SM}px;
    padding: 6px;
}}

/* ---- Progress bar ---- */
QProgressBar {{
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid {GLASS_BORDER};
    border-radius: 7px;
    text-align: center;
    color: {TEXT_SECONDARY};
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT_GREEN}, stop:1 {ACCENT_GREEN_HOVER});
    border-radius: 6px;
}}

/* ---- Status bar ---- */
QStatusBar {{
    background: {GLASS_FILL};
    border-top: 1px solid {GLASS_BORDER};
    color: {TEXT_SECONDARY};
}}
#statusMatch {{
    font-weight: 600;
    padding: 2px 8px;
}}

/* ---- Scrollbars ---- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: rgba(255, 255, 255, 0.15);
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: rgba(255, 255, 255, 0.25);
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: rgba(255, 255, 255, 0.15);
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QToolTip {{
    background: {SURFACE_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {GLASS_BORDER};
    border-radius: 6px;
    padding: 4px 8px;
}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_dark(app: QApplication) -> None:
    """Apply the dark theme to *app* (Palette + QSS + matplotlib rcParams)."""
    app.setStyle("Fusion")
    app.setPalette(_make_palette())

    font = QFont("Segoe UI", 9)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    icon_path = Path(__file__).resolve().parent / "resources" / "icon.svg"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    qss = _load_qss()
    if qss:
        app.setStyleSheet(qss)

    _setup_mpl()