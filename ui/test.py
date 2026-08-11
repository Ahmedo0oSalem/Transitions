"""
2D Tactical Analysis visualization.
Loads formation_segments.csv and plots one point per segment.
Stage 3: Enhanced interactions, legends, and metadata attachment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# Markers for shape encoding (limited to distinct shapes)
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8', '+', 'x']

def format_time(seconds):
    """Convert seconds to MM:SS format."""
    if pd.isna(seconds):
        return "N/A"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def _ensure_hierarchy(df):
    """Ensures 'variant' and 'family' columns are correct."""
    if 'variant' in df.columns:
        df['variant'] = df['variant'].astype(str).apply(lambda x: x.split('\n')[0].strip())
    elif 'formation' in df.columns:
        df['variant'] = df['formation'].astype(str).apply(lambda x: x.split('\n')[0].strip().removesuffix('flat'))
    
    if 'family' not in df.columns or df['family'].isna().all() or (df['family'] == 'other').all():
        if 'variant' in df.columns:
            df['family'] = df['variant'].apply(lambda x: derive_hierarchy(x)['family'])
        else:
            df['family'] = 'other'
    return df

def plot_2d_analysis(match_id, processed_dir, filters, encodings):
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
        
    # --- 2. Resolve Columns ---
    x_col = METRIC_COLS.get(encodings["x_axis"], "mean_width")
    y_col = METRIC_COLS.get(encodings["y_axis"], "mean_depth")
    
    # Drop NaNs for axes
    plot_df = df.dropna(subset=[x_col, y_col]).copy().reset_index(drop=True)
    
    # Assign a unique ID to each row for interaction tracking
    plot_df['plot_id'] = range(len(plot_df))
    
    # --- 3. Setup Figure ---
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(FIG_FACE)
    
    # --- 4. Visual Encodings ---
    color_col_label = encodings.get("color", "None")
    shape_col_label = encodings.get("shape", "None")
    
    is_color_metric = color_col_label in METRIC_COLS
    is_shape_cat = shape_col_label in CAT_COLS
    
    # --- Color Logic (Sequential Scale) ---
    cmap = plt.cm.viridis
    norm = None
    cbar = None
    
    if is_color_metric:
        color_col = METRIC_COLS[color_col_label]
        vmin, vmax = plot_df[color_col].min(), plot_df[color_col].max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = None

    # --- Shape Logic ---
    shape_map = {}
    if is_shape_cat:
        shape_col = CAT_COLS[shape_col_label]
        categories = sorted(plot_df[shape_col].unique())
        # Limit shapes to avoid confusion
        shape_map = {cat: MARKERS[i % len(MARKERS)] for i, cat in enumerate(categories)}
    
    # --- 5. Plotting ---
    # We store scatter collections to map clicks back to data
    fig._scatter_map = {} 
    
    if shape_map:
        for cat, marker in shape_map.items():
            mask = plot_df[shape_col] == cat
            sub_df = plot_df[mask]
            if sub_df.empty: continue
            
            sub_colors = sub_df[METRIC_COLS[color_col_label]] if is_color_metric else '#3498db'
            
            sc = ax.scatter(
                sub_df[x_col], sub_df[y_col],
                c=sub_colors, cmap=cmap, norm=norm,
                marker=marker, s=80, alpha=0.7, 
                edgecolors='white', linewidth=0.5,
                label=str(cat), zorder=3
            )
            # Map this scatter collection to the plot_ids it contains
            fig._scatter_map[sc] = sub_df['plot_id'].values
    else:
        colors = plot_df[METRIC_COLS[color_col_label]] if is_color_metric else '#3498db'
        sc = ax.scatter(
            plot_df[x_col], plot_df[y_col],
            c=colors, cmap=cmap, norm=norm,
            marker='o', s=80, alpha=0.7,
            edgecolors='white', linewidth=0.5, zorder=3
        )
        fig._scatter_map[sc] = plot_df['plot_id'].values

    # Colorbar
    if is_color_metric and cmap is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label(color_col_label, color=LABEL_COLOR, fontsize=10)
        cbar.ax.yaxis.set_tick_params(color=TICK_COLOR)
        cbar.outline.set_color(GRID)

    # Shape Legend (Outside plot)
    if shape_map:
        ax.legend(title=shape_col_label, bbox_to_anchor=(1.05, 1), loc='upper left', 
                  facecolor=FIG_FACE, edgecolor=GRID, labelcolor=TEXT_PRIMARY)

    # --- 6. Labels and Layout ---
    ax.set_xlabel(encodings["x_axis"], color=LABEL_COLOR, fontsize=12)
    ax.set_ylabel(encodings["y_axis"], color=LABEL_COLOR, fontsize=12)
    ax.set_title(f"Formation Segments: {encodings['x_axis']} vs {encodings['y_axis']}", 
                 color=TEXT_PRIMARY, fontsize=14, pad=15)
    ax.tick_params(colors=TICK_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, alpha=0.2, color=GRID)
    
    plt.tight_layout()
    fig.subplots_adjust(right=0.85) # Make room for legend
    
    # --- 7. Attach Data for Interaction ---
    # Convert dataframe to list of dicts for easy lookup by plot_id
    fig._segment_records = plot_df.to_dict('records')
    fig._encodings = encodings # Store current encodings for tooltip generation
    
    # Create an annotation for hover (initially invisible)
    fig._hover_annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                                   bbox=dict(boxstyle="round", fc=FIG_FACE, ec=GRID, alpha=0.9),
                                   color=TEXT_PRIMARY, fontsize=9, zorder=10)
    fig._hover_annot.set_visible(False)
    
    # Create a highlight ring for selection (initially empty)
    fig._selection_ring = ax.scatter([], [], s=200, facecolors='none', edgecolors='#f1c40f', linewidths=2, zorder=4)

    return fig