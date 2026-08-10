"""
2D Tactical Analysis visualization.
Loads formation_segments.csv and plots one point per segment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..io.paths import match_dir
from ..analytics.formations.taxonomy import derive_hierarchy
from .theme import FIG_FACE, TEXT_PRIMARY, LABEL_COLOR, TICK_COLOR, GRID

# Map UI dropdown labels to actual CSV column names
METRIC_COLS = {
    "Width": "mean_width",
    "Depth": "mean_depth",
    "Compactness": "mean_compactness",
    "Duration": "duration",
    "Centroid velocity": "mean_centroid_velocity",
    "Centroid displacement": "net_centroid_displacement",
    "Confidence": "mean_confidence",
    "Elongation": "mean_elongation",
}

CAT_COLS = {
    "Formation": "variant",
    "Formation Family": "family",
    "Team": "team",
    "Period": "period",
}

# Distinct markers for categorical shapes
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8', '+', 'x']

def _fix_families(df):
    """
    Fix the broken 'family' column by deriving it from the 'variant' column.
    Uses the same derive_hierarchy logic as timeline.py
    """
    if 'variant' in df.columns:
        # Clean variant names first
        df['variant'] = df['variant'].astype(str).str.strip()
        # Derive family from variant using taxonomy (same as timeline.py)
        df['family'] = df['variant'].apply(lambda x: derive_hierarchy(x)['family'])
    return df

def plot_2d_analysis(match_id, processed_dir, filters, encodings):
    """
    Generate a 2D scatter plot of formation segments.
    """
    folder = match_dir(match_id, processed_dir)
    segments_path = folder / "formation_segments.csv"
    
    if not segments_path.is_file():
        raise FileNotFoundError(
            f"formation_segments.csv not found for match {match_id}. "
            "Run Detect Formations first."
        )
        
    df = pd.read_csv(segments_path)
    
    # Fix the broken family column using taxonomy (like timeline.py does)
    df = _fix_families(df)
    
    # Apply Filters
    if filters.get("team") != "All":
        df = df[df["team"] == filters["team"].lower()]
    if filters.get("period") != "All":
        df = df[df["period"] == int(filters["period"])]
    if filters.get("formation") != "All":
        col = 'variant' if 'variant' in df.columns else 'formation'
        df = df[df[col] == filters["formation"]]
    if filters.get("min_duration", 0) > 0:
        df = df[df["duration"] >= filters["min_duration"]]
        
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No segments match the selected filters.", 
                ha="center", va="center", fontsize=12, color=TEXT_PRIMARY)
        ax.set_facecolor(FIG_FACE)
        fig.patch.set_facecolor(FIG_FACE)
        return fig
        
    x_col = METRIC_COLS.get(encodings["x_axis"], "mean_width")
    y_col = METRIC_COLS.get(encodings["y_axis"], "mean_depth")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(FIG_FACE)
    
    plot_df = df.dropna(subset=[x_col, y_col]).copy().reset_index(drop=True)
    
    color_col = encodings.get("color", "None")
    shape_col = encodings.get("shape", "None")
    size_col = encodings.get("size", "None")
    
    is_color_cat = color_col in CAT_COLS
    is_shape_cat = shape_col in CAT_COLS
    
    # Color Logic
    color_map_dict = None
    if is_color_cat:
        cat_col = CAT_COLS[color_col]
        categories = sorted([str(x) for x in plot_df[cat_col].unique()])
        cmap = plt.get_cmap("tab10")
        color_map_dict = {cat: cmap(i % 10) for i, cat in enumerate(categories)}
        colors = plot_df[cat_col].map(color_map_dict).tolist()
    elif color_col in METRIC_COLS:
        met_col = METRIC_COLS[color_col]
        colors = plot_df[met_col].tolist()
        cmap = plt.cm.viridis
    else:
        colors = '#3498db'
        cmap = None
        
    # Shape Logic
    shape_map = None
    if is_shape_cat:
        shape_cat_col = CAT_COLS[shape_col]
        shape_categories = sorted([str(c) for c in plot_df[shape_cat_col].unique()])
        shape_map = {cat: MARKERS[i % len(MARKERS)] for i, cat in enumerate(shape_categories)}
        
    # Size Logic
    if size_col == "Duration":
        durations = plot_df["duration"]
        if durations.max() > durations.min():
            sizes = (20 + 180 * (durations - durations.min()) / (durations.max() - durations.min())).tolist()
        else:
            sizes = [100] * len(plot_df)
    else:
        sizes = [80] * len(plot_df)
        
    # Plotting
    handles, labels = [], []
    
    if shape_map is not None:
        for shape_cat, marker in shape_map.items():
            mask = plot_df[shape_cat_col].astype(str) == shape_cat
            sub_df = plot_df[mask].reset_index(drop=True)
            if sub_df.empty:
                continue
                
            mask_values = mask.values
            sub_colors = [colors[i] for i, m in enumerate(mask_values) if m]
            sub_sizes = [sizes[i] for i, m in enumerate(mask_values) if m]
            
            sc = ax.scatter(
                sub_df[x_col], sub_df[y_col],
                c=sub_colors, cmap=cmap if not is_color_cat else None,
                marker=marker, s=sub_sizes,
                alpha=0.7, edgecolors='white', linewidth=0.5,
                label=str(shape_cat), zorder=3
            )
            handles.append(sc)
            labels.append(str(shape_cat))
    else:
        sc = ax.scatter(
            plot_df[x_col], plot_df[y_col],
            c=colors, cmap=cmap if not is_color_cat else None,
            marker='o', s=sizes,
            alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3
        )
        
    # Add color legend entries if color is categorical
    if is_color_cat and color_map_dict:
        for cat, color in color_map_dict.items():
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=color, markersize=8, 
                                      label=str(cat), linestyle='None'))
            labels.append(str(cat))
    
    # Labels and Layout
    ax.set_xlabel(encodings["x_axis"], color=LABEL_COLOR, fontsize=12)
    ax.set_ylabel(encodings["y_axis"], color=LABEL_COLOR, fontsize=12)
    ax.set_title(f"Formation Segments: {encodings['x_axis']} vs {encodings['y_axis']}", 
                 color=TEXT_PRIMARY, fontsize=14, pad=15)
    ax.tick_params(colors=TICK_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, alpha=0.2, color=GRID)
    
    # Add colorbar for continuous color
    if not is_color_cat and cmap is not None and isinstance(colors, list) and len(colors) > 0 and isinstance(colors[0], (int, float)):
        if 'sc' in locals() and hasattr(sc, 'get_array'):
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(color_col, color=LABEL_COLOR)
            cbar.ax.yaxis.set_tick_params(color=TICK_COLOR)
            cbar.outline.set_color(GRID)
        
    plt.tight_layout()
    
    # Store legend data in the figure for the PyQt6 popup button
    fig._legend_handles = handles
    fig._legend_labels = labels
    fig._legend_title = f"{color_col} / {shape_col}" if (is_color_cat and is_shape_cat) else (color_col if is_color_cat else shape_col)
    
    return fig