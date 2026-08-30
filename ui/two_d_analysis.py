"""
2D Tactical Analysis visualization.
Loads formation_segments.csv and plots one point per segment.
Stage 3: Enhanced interactions, legends, and metadata attachment.
Stage 4: Integrated EPV/DAS metrics.
Stage 5: Graph Control UI + Split by Team Visualization.
Stage 6: Axis Range Configuration Integration.
FIXED: Safe metric loading + preserved Graph/Range controls.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from ..io.paths import match_dir
from ..analytics.formations.segments import _aggregate_obso_for_segment
from ..analytics.formations.taxonomy import derive_hierarchy
from .theme import FIG_FACE, TEXT_PRIMARY, LABEL_COLOR, TICK_COLOR, GRID
from .metric_ranges import get_axis_range

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
    "Pitch Control": "mean_home_control",
    "Home Control": "mean_home_control",
    "Away Control": "mean_away_control",
    "OBSO": "mean_obso",
    "N Windows": "n_windows",
    "N Frames": "n_frames",
    "Cumulative EPV": "cumulative_epv",
    "Mean EPV": "mean_epv",
    "EPV / min": "epv_per_min",
    "DAS Count": "das_count",
    "DAS / min": "das_per_min",
}

CAT_COLS = {
    "Formation": "variant",
    "Formation Family": "family",
    "Team": "team",
    "Period": "period",
}

MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8', '+', 'x']

def format_time(seconds):
    """Convert seconds to MM:SS format."""
    if pd.isna(seconds): return "N/A"
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

def _normalize_team_filter(team_value):
    """Normalize 2D-analysis team filter values from the UI to the CSV format."""
    if team_value is None:
        return "all"
    normalized = str(team_value).strip().lower()
    if normalized in {"all", ""}:
        return "all"
    if normalized in {"home", "away"}:
        return normalized
    return normalized

def _normalize_team_x_for_plot(x_value, team, period, home_team_start_left, pitch_length):
    """Mirror x-values for a single-team plot so team-only views use the correct field orientation."""
    if x_value is None or pd.isna(x_value):
        return x_value
    
    home_attacks_left_to_right = home_team_start_left if period % 2 == 1 else not home_team_start_left
    team_attacks_left_to_right = home_attacks_left_to_right if team == "home" else not home_attacks_left_to_right
    
    if not team_attacks_left_to_right:
        return pitch_length - float(x_value)
    return float(x_value)

def _apply_team_orientation_to_position_columns(plot_df, team_filter, metadata_path):
    """Mirror x-position metrics for a single-team view to keep the chart aligned to the pitch."""
    if team_filter == "all" or metadata_path is None or not metadata_path.is_file():
        return plot_df
        
    import json
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else {}
        
    pitches = metadata.get("stadium", {}).get("pitches", [])
    if pitches:
        pitch_length = float(pitches[0].get("length", 105.0))
    else:
        pitch_length = 105.0
        
    home_team_start_left = bool(metadata.get("homeTeamStartLeft", True))
    x_cols = ["mean_center_x", "std_center_x", "range_center_x"]
    
    for x_col in x_cols:
        if x_col not in plot_df.columns:
            continue
        plot_df[x_col] = plot_df.apply(
            lambda row: _normalize_team_x_for_plot(
                row[x_col], 
                team_filter,
                int(row.get("period", 1)),
                home_team_start_left,
                pitch_length,
            ),
            axis=1,
        )
    return plot_df

def _resolve_metric_column(metric_label, team_filter):
    """Resolve the metric column used for plotting."""
    if metric_label == "Pitch Control":
        if str(team_filter).lower() == "away":
            return "mean_away_control"
        return "mean_home_control"
    if metric_label == "Home Control":
        return "mean_home_control"
    if metric_label == "Away Control":
        return "mean_away_control"
    return METRIC_COLS.get(metric_label, "mean_width")

def _get_team_names(match_id, processed_dir):
    """Retrieve actual team names from metadata."""
    folder = match_dir(match_id, processed_dir)
    meta_path = folder / "metadata.json"
    home_name = "Home"
    away_name = "Away"
    try:
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta, list): meta = meta[0]
        home_name = meta.get("homeTeam", {}).get("name", "Home")
        away_name = meta.get("awayTeam", {}).get("name", "Away")
    except Exception:
        pass
    return home_name, away_name

def plot_2d_analysis(match_id, processed_dir, filters, encodings):
    folder = match_dir(match_id, processed_dir)
    segments_path = folder / "formation_segments.csv"
    
    if not segments_path.is_file():
        raise FileNotFoundError(f"formation_segments.csv not found for match {match_id}.")
        
    df = pd.read_csv(segments_path)
    df = _ensure_hierarchy(df)
    
    # --- FIX: Load missing heavy metrics BEFORE filtering/dropping NaN ---
    if "mean_obso" not in df.columns:
        try:
            from ..analytics.obso import compute_obso_for_match
            obso_df = compute_obso_for_match(match_id, processed_dir)
            if not obso_df.empty:
                obso_values = []
                for _, row in df.iterrows():
                    props = _aggregate_obso_for_segment(
                        obso_df,
                        str(row.get("team", "")).strip().lower(),
                        int(row.get("period", 0)),
                        float(row.get("start_sec", row.get("windowStartSec", 0))),
                        float(row.get("end_sec", row.get("windowEndSec", 0))),
                    )
                    obso_values.append(props.get("mean_obso", np.nan))
                df["mean_obso"] = obso_values
        except Exception:
            pass

    if ("mean_home_control" not in df.columns) or ("mean_away_control" not in df.columns):
        try:
            from ..analytics.pitch_control.control import compute_pitch_control_for_match
            control_df = compute_pitch_control_for_match(match_id, processed_dir)
            if not control_df.empty:
                home_vals = []
                away_vals = []
                for _, row in df.iterrows():
                    start = float(row.get("start_sec", row.get("windowStartSec", 0)))
                    end = float(row.get("end_sec", row.get("windowEndSec", 0)))
                    period = int(row.get("period", 0))
                    mask = (control_df["period"] == period) & (control_df["elapsed"] >= start) & (control_df["elapsed"] < end)
                    sub = control_df[mask]
                    home_vals.append(float(sub["home_control"].mean()) if not sub.empty else np.nan)
                    away_vals.append(float(sub["away_control"].mean()) if not sub.empty else np.nan)
                if "mean_home_control" not in df.columns:
                    df["mean_home_control"] = home_vals
                if "mean_away_control" not in df.columns:
                    df["mean_away_control"] = away_vals
        except Exception:
            pass
    # ---------------------------------------------------------
    
    # --- Handle Graph Mode ---
    graph_mode = encodings.get("graph_mode", "Combined")
    valid_modes = {"Combined", "Split by Team"}
    if graph_mode not in valid_modes:
        graph_mode = "Combined"
    
    # Get real team names for titles
    home_name, away_name = _get_team_names(match_id, processed_dir)
    
    # --- Apply Filters (Global) ---
    team_filter = _normalize_team_filter(filters.get("team", "All"))
    if team_filter != "all":
        team_values = df["team"].astype(str).str.strip().str.lower()
        df = df[team_values == team_filter].copy()
        
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
        ax.set_facecolor(FIG_FACE); fig.patch.set_facecolor(FIG_FACE)
        return fig
        
    # --- Resolve Columns ---
    x_axis_label = encodings.get("x_axis", "Width")
    y_axis_label = encodings.get("y_axis", "Depth")
    x_col = _resolve_metric_column(x_axis_label, team_filter)
    y_col = _resolve_metric_column(y_axis_label, team_filter)
    
    plot_df = df.dropna(subset=[x_col, y_col]).copy().reset_index(drop=True)
    
    # RESTORED: Apply team orientation mirroring for single-team views
    if team_filter != "all":
        plot_df = _apply_team_orientation_to_position_columns(
            plot_df, team_filter, folder / "metadata.json"
        )
        
    plot_df['plot_id'] = range(len(plot_df))
    
    # --- Prepare Color Data ---
    color_col_label = encodings.get("color", "None")
    shape_col_label = encodings.get("shape", "None")
    is_color_metric = color_col_label in METRIC_COLS
    is_shape_cat = shape_col_label in CAT_COLS
    
    cmap = None; norm = None; colors = '#3498db'
    if is_color_metric:
        color_col = METRIC_COLS[color_col_label]
        c_values = plot_df[color_col].values
        vmin, vmax = float(np.nanmin(c_values)), float(np.nanmax(c_values))
        cmap = plt.cm.viridis
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = c_values 
    
    # --- Shape Logic ---
    shape_map = {}; shape_categories = []
    if is_shape_cat:
        shape_col = CAT_COLS[shape_col_label]
        shape_categories = sorted(plot_df[shape_col].unique())
        shape_map = {cat: MARKERS[i % len(MARKERS)] for i, cat in enumerate(shape_categories)}
    
    # --- Helper to Render a Single Graph ---
    def render_graph(data, title_suffix, team_specific_filter=None):
        """Renders one complete graph based on provided data subset."""
        if team_specific_filter:
            data = data[data["team"] == team_specific_filter].copy()
            
        if data.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f"No data for {title_suffix}", 
                    ha="center", va="center", fontsize=12, color=TEXT_PRIMARY)
            ax.set_facecolor(FIG_FACE); fig.patch.set_facecolor(FIG_FACE)
            return fig
            
        data = data.reset_index(drop=True)
        data['plot_id'] = range(len(data))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(FIG_FACE); ax.set_facecolor(FIG_FACE)
        
        # Re-calculate color mapping for this specific subset
        local_cmap = cmap; local_norm = norm; local_colors = colors
        if is_color_metric:
            c_vals = data[METRIC_COLS[color_col_label]].values
            lvmin, lvmax = float(np.nanmin(c_vals)), float(np.nanmax(c_vals))
            local_cmap = plt.cm.viridis
            local_norm = Normalize(vmin=lvmin, vmax=lvmax)
            local_colors = c_vals
            
        # Plotting
        fig._scatter_map = {} 
        if shape_map:
            for cat, marker in shape_map.items():
                mask = data[shape_col] == cat
                sub_df = data[mask]
                if sub_df.empty: continue
                
                if is_color_metric:
                    sub_colors = local_colors[mask.values] 
                else:
                    cat_col = CAT_COLS.get(color_col_label, 'variant')
                    unique_cats = sorted(data[cat_col].unique())
                    cat_cmap = plt.cm.tab10
                    color_lookup = {c: cat_cmap(i % 10) for i, c in enumerate(unique_cats)}
                    sub_colors = [color_lookup[v] for v in sub_df[cat_col]]
                
                sc = ax.scatter(sub_df[x_col], sub_df[y_col], c=sub_colors, cmap=local_cmap, norm=local_norm,
                                marker=marker, s=80, alpha=0.7, edgecolors='white', linewidths=1.5, label=str(cat), zorder=3)
                fig._scatter_map[sc] = sub_df['plot_id'].values
        else:
            if is_color_metric:
                sc = ax.scatter(data[x_col], data[y_col], c=local_colors, cmap=local_cmap, norm=local_norm,
                                marker='o', s=80, alpha=0.7, edgecolors='white', linewidths=1.5, zorder=3)
            else:
                cat_col = CAT_COLS.get(color_col_label, 'variant')
                unique_cats = sorted(data[cat_col].unique())
                cat_cmap = plt.cm.tab10
                color_lookup = {c: cat_cmap(i % 10) for i, c in enumerate(unique_cats)}
                plot_colors = [color_lookup[v] for v in data[cat_col]]
                sc = ax.scatter(data[x_col], data[y_col], c=plot_colors,
                                marker='o', s=80, alpha=0.7, edgecolors='white', linewidths=1.5, zorder=3)
            fig._scatter_map[sc] = data['plot_id'].values
        
        ax.legend().set_visible(False)
        
        if is_color_metric and local_cmap is not None:
            sm = plt.cm.ScalarMappable(cmap=local_cmap, norm=local_norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label(color_col_label, color=LABEL_COLOR, fontsize=10)
            cbar.ax.yaxis.set_tick_params(color=TICK_COLOR); cbar.outline.set_color(GRID)
        
        ax.set_xlabel(x_axis_label, color=LABEL_COLOR, fontsize=12)
        ax.set_ylabel(y_axis_label, color=LABEL_COLOR, fontsize=12)
        ax.set_title(f"Formation Segments: {x_axis_label} vs {y_axis_label}\n{title_suffix}", 
                     color=TEXT_PRIMARY, fontsize=14, pad=15)
        ax.tick_params(colors=TICK_COLOR)
        for spine in ax.spines.values(): spine.set_color(GRID)
        ax.grid(True, alpha=0.2, color=GRID)
        
        plt.tight_layout(); fig.subplots_adjust(right=0.85)
        
        # --- STAGE 6: Apply Axis Ranges ---
        x_range_mode = encodings.get("x_range_mode", "Auto (Current Data)")
        y_range_mode = encodings.get("y_range_mode", "Auto (Current Data)")
        
        mode_map = {
            "Auto (Current Data)": "auto",
            "Fixed Metric Range": "fixed",
            "Shared Range": "shared"
        }
        x_mode = mode_map.get(x_range_mode, "auto")
        y_mode = mode_map.get(y_range_mode, "auto")
        
        x_range = get_axis_range(x_axis_label, x_mode, data)
        y_range = get_axis_range(y_axis_label, y_mode, data)
        
        if x_range is not None:
            ax.set_xlim(x_range)
        if y_range is not None:
            ax.set_ylim(y_range)
        # --------------------------------
        
        # Attach Interaction Data
        fig._segment_records = data.to_dict('records')
        fig._encodings = encodings
        fig._legend_data = {
            'shape_categories': shape_categories if shape_categories else None,
            'color_metric': color_col_label if is_color_metric else None,
            'title': f"{x_axis_label} vs {y_axis_label} ({title_suffix})"
        }
        
        fig._hover_annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                                       bbox=dict(boxstyle="round", fc=FIG_FACE, ec=GRID, alpha=0.9),
                                       color=TEXT_PRIMARY, fontsize=9, zorder=10)
        fig._hover_annot.set_visible(False)
        
        fig._selection_ring = ax.scatter([], [], s=200, facecolors='none', edgecolors='#f1c40f', linewidths=2, zorder=4)
        
        return fig

    # --- Execute Rendering Based on Mode ---
    if graph_mode == "Split by Team":
        if team_filter == "all":
            home_fig = render_graph(plot_df, home_name, team_specific_filter="home")
            away_fig = render_graph(plot_df, away_name, team_specific_filter="away")
            return (home_fig, away_fig)
        else:
            return render_graph(plot_df, f"{team_filter.capitalize()} Team")
    else:
        return render_graph(plot_df, "Combined View")