"""
2D Tactical Analysis visualization.
Loads formation_segments.csv and plots one point per segment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QPushButton
from PyQt6.QtCore import Qt
from ..io.paths import match_dir
from ..analytics.formations.taxonomy import derive_hierarchy
from .theme import FIG_FACE, TEXT_PRIMARY, LABEL_COLOR, TICK_COLOR, GRID

# Map UI labels to CSV columns
METRIC_COLS = {
    "Width": "mean_width",
    "Depth": "mean_depth",
    "Compactness": "mean_compactness",
    "Duration": "duration",
    "Center X": "mean_center_x",
    "Center Y": "mean_center_y",
    "Elongation": "mean_elongation",
    "Centroid Displacement": "net_centroid_displacement",
    "Centroid Velocity": "mean_centroid_velocity",
    "Template Displacement": "mean_template_displacement",
    "Confidence": "mean_confidence",
    "N Windows": "n_windows",
    "N Frames": "n_frames",
}

CAT_COLS = {
    "Formation": "variant",
    "Formation Family": "family",
    "Team": "team",
    "Period": "period",
}

# Markers for shape encoding
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8', '+', 'x']

class LegendPopup(QDialog):
    """Popup dialog to display the plot legend."""
    def __init__(self, parent=None, title="Legend", shape_categories=None, color_metric=None):
        super().__init__(parent)
        self.setWindowTitle(f"Legend - {title}")
        self.setMinimumSize(400, 500)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Shape categories
        if shape_categories:
            shape_label = QLabel("<b>Formation (Shape):</b>")
            layout.addWidget(shape_label)
            
            shape_list = QListWidget()
            for cat in shape_categories:
                shape_list.addItem(str(cat))
            layout.addWidget(shape_list)
        
        # Color metric info
        if color_metric:
            color_label = QLabel(f"<b>Color: {color_metric}</b>")
            layout.addWidget(color_label)
            color_info = QLabel("Continuous color scale (viridis colormap)")
            color_info.setStyleSheet("color: #8a9aaa;")
            layout.addWidget(color_info)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

def _ensure_hierarchy(df):
    """
    Ensures 'variant' and 'family' columns are correct.
    Fixes the issue where 'family' might be 'other' in the CSV.
    """
    # Clean variant
    if 'variant' in df.columns:
        # Remove pandas artifacts if present
        df['variant'] = df['variant'].astype(str).apply(lambda x: x.split('\n')[0].strip())
    elif 'formation' in df.columns:
        df['variant'] = df['formation'].astype(str).apply(lambda x: x.split('\n')[0].strip().removesuffix('flat'))
    
    # Derive family if missing or invalid
    if 'family' not in df.columns or df['family'].isna().all() or (df['family'] == 'other').all():
        if 'variant' in df.columns:
            df['family'] = df['variant'].apply(lambda x: derive_hierarchy(x)['family'])
        else:
            df['family'] = 'other'
            
    return df

def plot_2d_analysis(match_id, processed_dir, filters, encodings):
    """
    Generate a 2D scatter plot of formation segments.
    """
    folder = match_dir(match_id, processed_dir)
    segments_path = folder / "formation_segments.csv"
    
    if not segments_path.is_file():
        raise FileNotFoundError(f"formation_segments.csv not found for match {match_id}.")
        
    df = pd.read_csv(segments_path)
    df = _ensure_hierarchy(df)
    
    # --- 1. Apply Filters ---
    if filters.get("team") != "All":
        df = df[df["team"] == filters["team"].lower()]
    if filters.get("period") != "All":
        df = df[df["period"] == int(filters["period"])]
    if filters.get("formation") != "All":
        # Filter by variant
        df = df[df["variant"] == filters["formation"]]
    if filters.get("min_duration", 0) > 0:
        df = df[df["duration"] >= filters["min_duration"]]
        
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No segments match the selected filters.", 
                ha="center", va="center", fontsize=12, color=TEXT_PRIMARY)
        ax.set_facecolor(FIG_FACE)
        fig.patch.set_facecolor(FIG_FACE)
        return fig
        
    # --- 2. Resolve Columns ---
    x_col = METRIC_COLS.get(encodings["x_axis"], "mean_width")
    y_col = METRIC_COLS.get(encodings["y_axis"], "mean_depth")
    
    # Drop NaNs for axes
    plot_df = df.dropna(subset=[x_col, y_col]).copy().reset_index(drop=True)
    
    # --- 3. Setup Figure ---
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(FIG_FACE)
    
    # --- 4. Visual Encodings ---
    color_col_label = encodings.get("color", "None")
    shape_col_label = encodings.get("shape", "None")
    
    is_color_metric = color_col_label in METRIC_COLS
    is_shape_cat = shape_col_label in CAT_COLS
    
    # --- Color Logic (Continuous for Stage 2) ---
    cmap = plt.cm.viridis
    colors = None
    cbar = None
    
    if is_color_metric:
        color_col = METRIC_COLS[color_col_label]
        colors = plot_df[color_col]
    else:
        # Fallback to single color
        colors = '#3498db'
        cmap = None

    # --- Shape Logic (Categorical) ---
    shape_map = {}
    shape_categories = []
    if is_shape_cat:
        shape_col = CAT_COLS[shape_col_label]
        shape_categories = sorted(plot_df[shape_col].unique())
        shape_map = {cat: MARKERS[i % len(MARKERS)] for i, cat in enumerate(shape_categories)}
    
    # --- 5. Plotting ---
    # If we have shapes, we plot group by group to assign markers
    if shape_map:
        for cat, marker in shape_map.items():
            mask = plot_df[shape_col] == cat
            sub_df = plot_df[mask]
            if sub_df.empty: continue
            
            sub_colors = colors[mask] if hasattr(colors, '__len__') else colors
            
            ax.scatter(
                sub_df[x_col], sub_df[y_col],
                c=sub_colors, cmap=cmap,
                marker=marker, s=80, alpha=0.7, 
                edgecolors='white', linewidth=0.5,
                label=str(cat), zorder=3
            )
    else:
        # No shape encoding, plot all
        ax.scatter(
            plot_df[x_col], plot_df[y_col],
            c=colors, cmap=cmap,
            marker='o', s=80, alpha=0.7,
            edgecolors='white', linewidth=0.5, zorder=3
        )

    # Colorbar for continuous color
    if is_color_metric and cmap is not None and hasattr(colors, '__len__'):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(colors.min(), colors.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(color_col_label, color=LABEL_COLOR)
        cbar.ax.yaxis.set_tick_params(color=TICK_COLOR)
        cbar.outline.set_color(GRID)

    # --- 6. Labels and Layout ---
    ax.set_xlabel(encodings["x_axis"], color=LABEL_COLOR, fontsize=12)
    ax.set_ylabel(encodings["y_axis"], color=LABEL_COLOR, fontsize=12)
    ax.set_title(f"Formation Segments: {encodings['x_axis']} vs {encodings['y_axis']}", 
                 color=TEXT_PRIMARY, fontsize=14, pad=15)
    ax.tick_params(colors=TICK_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, alpha=0.2, color=GRID)
    
    # Add "Show Legend" button as text annotation that can be clicked
    button_text = ax.text(0.02, 0.98, "📋 Show Legend", 
                          transform=ax.transAxes,
                          fontsize=10, 
                          fontweight='bold',
                          color=TEXT_PRIMARY,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#2b8a3e', alpha=0.8),
                          picker=True)  # Enable picking
    
    # Store data for the button click handler
    fig._legend_data = {
        'shape_categories': shape_categories if shape_categories else None,
        'color_metric': color_col_label if is_color_metric else None,
        'title': f"{encodings['x_axis']} vs {encodings['y_axis']}",
        'popup': None  # Store popup reference to prevent garbage collection
    }
    
    def on_button_click(event):
        """Handle button click to show legend popup."""
        if event.artist == button_text:
            # Create popup
            popup = LegendPopup(
                title=fig._legend_data['title'],
                shape_categories=fig._legend_data['shape_categories'],
                color_metric=fig._legend_data['color_metric']
            )
            # Store reference to prevent garbage collection
            fig._legend_data['popup'] = popup
            # Use exec() instead of show() to make it modal and block until closed
            popup.exec()
    
    # Connect the click event
    fig.canvas.mpl_connect('pick_event', on_button_click)
    
    plt.tight_layout()
    return fig