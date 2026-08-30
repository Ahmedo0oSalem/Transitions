"""
Central registry for metric axis range configuration.
Defines fixed bounds where meaningful and supports dynamic calculation modes.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Metric Registry Structure:
# key: UI Label (must match METRIC_COLS keys exactly)
# value: {
#   "fixed_min": float | None,      # Natural lower bound (e.g., 0 for probabilities)
#   "fixed_max": float | None,      # Natural upper bound (e.g., 1 for probabilities) 
#   "is_bounded": bool,             # True if fixed_min/max are definitive global bounds
#   "description": str              # Human-readable explanation of the range
# }

METRIC_RANGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- Spatial Metrics (Pitch-dependent, not globally fixed) ---
    "Width": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Team width in meters"},
    "Depth": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Team depth in meters"},
    "Compactness": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Mean distance to centroid (m)"},
    "Center X": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Centroid X position"},
    "Center Y": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Centroid Y position"},
    
    # --- Probabilistic / Normalized Metrics (Naturally Bounded [0,1]) ---
    "Confidence": {"fixed_min": 0.0, "fixed_max": 1.0, "is_bounded": True, "desc": "Formation detection confidence"},
    "Pitch Control": {"fixed_min": 0.0, "fixed_max": 1.0, "is_bounded": True, "desc": "Home team pitch control probability"},
    "Home Control": {"fixed_min": 0.0, "fixed_max": 1.0, "is_bounded": True, "desc": "Home team pitch control probability"},
    "Away Control": {"fixed_min": 0.0, "fixed_max": 1.0, "is_bounded": True, "desc": "Away team pitch control probability"},
    "OBSO": {"fixed_min": 0.0, "fixed_max": 1.0, "is_bounded": True, "desc": "Off-ball scoring opportunity probability"},
    
    # --- Temporal / Count Metrics (Lower bound 0, upper unbounded) ---
    "Duration": {"fixed_min": 0.0, "fixed_max": None, "is_bounded": False, "desc": "Segment duration (seconds)"},
    "DAS Count": {"fixed_min": 0.0, "fixed_max": None, "is_bounded": False, "desc": "Number of dangerous sequences"},
    "N Windows": {"fixed_min": 0.0, "fixed_max": None, "is_bounded": False, "desc": "Number of formation windows"},
    "N Frames": {"fixed_min": 0.0, "fixed_max": None, "is_bounded": False, "desc": "Total tracking frames"},
    
    # --- Rate / Value Metrics (Unbounded, sign-dependent) ---
    "Cumulative EPV": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Total expected goal contribution"},
    "Mean EPV": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Average EPV per second"},
    "EPV / min": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "EPV rate per minute"},
    "DAS / min": {"fixed_min": 0.0, "fixed_max": None, "is_bounded": False, "desc": "Dangerous sequences per minute"},
    
    # --- Movement Metrics (Unbounded) ---
    "Elongation": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Depth-to-width ratio"},
    "Centroid Displacement": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Net centroid movement (m)"},
    "Centroid Velocity": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Mean centroid speed (m/s)"},
    "Template Displacement": {"fixed_min": None, "fixed_max": None, "is_bounded": False, "desc": "Avg cost per player vs template"},
}


def get_axis_range(
    metric_label: str,
    mode: str,
    current_data: pd.DataFrame,
    shared_data: Optional[pd.DataFrame] = None,
    padding: float = 0.05
) -> Optional[Tuple[float, float]]:
    """
    Calculate axis range based on mode and metric registry.
    
    Args:
        metric_label: UI label (e.g., "Pitch Control")
        mode: One of "auto", "fixed", "shared"
        current_data: Filtered DataFrame for this specific graph
        shared_data: Combined DataFrame across all graphs (for shared mode)
        padding: Fractional padding for auto/shared modes (default 5%)
        
    Returns:
        (min, max) tuple or None if range cannot be determined
    """
    from .two_d_analysis import METRIC_COLS
    
    col = METRIC_COLS.get(metric_label)
    if col is None or col not in current_data.columns:
        return None
        
    values = current_data[col].dropna()
    if values.empty:
        return None
        
    registry_entry = METRIC_RANGE_REGISTRY.get(metric_label, {})
    is_bounded = registry_entry.get("is_bounded", False)
    fixed_min = registry_entry.get("fixed_min")
    fixed_max = registry_entry.get("fixed_max")
    
    if mode == "fixed" and is_bounded:
        # Use definitive natural bounds
        return (float(fixed_min), float(fixed_max))
        
    elif mode == "fixed" and not is_bounded:
        # Metric has no meaningful global fixed range; fall back to auto
        # In a future stage, this could use user-defined custom bounds
        pass  # Fall through to auto behavior below
        
    # Determine source data for range calculation
    source_values = values
    if mode == "shared" and shared_data is not None and col in shared_data.columns:
        source_values = shared_data[col].dropna()
        if source_values.empty:
            source_values = values
            
    # Auto or Shared (or unbounded Fixed fallback)
    vmin = float(source_values.min())
    vmax = float(source_values.max())
    
    if vmin == vmax:
        # Avoid zero-range; add small buffer
        vmin -= 0.5
        vmax += 0.5
        
    # Apply symmetric padding
    span = vmax - vmin
    pad = span * padding
    vmin -= pad
    vmax += pad
    
    # Clamp to fixed bounds if they exist (even in auto/shared mode)
    if fixed_min is not None:
        vmin = max(vmin, float(fixed_min))
    if fixed_max is not None:
        vmax = min(vmax, float(fixed_max))
        
    return (vmin, vmax)