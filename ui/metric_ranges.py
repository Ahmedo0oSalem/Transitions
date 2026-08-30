"""
Central registry for metric axis range configuration.
Defines fixed bounds where meaningful and supports dynamic calculation modes.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import pandas as pd

# Metric Registry Structure
METRIC_RANGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Width": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "Depth": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "Compactness": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "Center X": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "Center Y": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "Confidence": {"fixed_min": 0.0, "fixed_max": 1.0, "is_bounded": True},
    "Pitch Control": {"fixed_min": 0.0, "fixed_max": 1.0, "is_bounded": True},
    "Home Control": {"fixed_min": 0.0, "fixed_max": 1.0, "is_bounded": True},
    "Away Control": {"fixed_min": 0.0, "fixed_max": 1.0, "is_bounded": True},
    "OBSO": {"fixed_min": 0.0, "fixed_max": 1.0, "is_bounded": True},
    "Duration": {"fixed_min": 0.0, "fixed_max": None, "is_bounded": False},
    "DAS Count": {"fixed_min": 0.0, "fixed_max": None, "is_bounded": False},
    "N Windows": {"fixed_min": 0.0, "fixed_max": None, "is_bounded": False},
    "N Frames": {"fixed_min": 0.0, "fixed_max": None, "is_bounded": False},
    "Cumulative EPV": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "Mean EPV": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "EPV / min": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "DAS / min": {"fixed_min": 0.0, "fixed_max": None, "is_bounded": False},
    "Elongation": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "Centroid Displacement": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "Centroid Velocity": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
    "Template Displacement": {"fixed_min": None, "fixed_max": None, "is_bounded": False},
}


def get_axis_range(
    metric_label: str,
    mode: str,
    data_sources: List[pd.DataFrame],
    column_resolver: callable,
    padding: float = 0.05
) -> Optional[Tuple[float, float]]:
    """
    Calculate axis range based on mode and metric registry.
    
    Args:
        metric_label: UI label (e.g., "Compactness")
        mode: One of "auto", "fixed", "shared"
        data_sources: List of filtered DataFrames (one per graph)
        column_resolver: Function(metric_label, team_filter) -> column_name
        padding: Fractional padding for auto/shared modes
        
    Returns:
        (min, max) tuple or None if range cannot be determined
    """
    from .two_d_analysis import METRIC_COLS
    
    # Get column name using resolver (handles Pitch Control team-specific mapping)
    col = column_resolver(metric_label, "All") 
    if col is None:
        return None
        
    registry_entry = METRIC_RANGE_REGISTRY.get(metric_label, {})
    is_bounded = registry_entry.get("is_bounded", False)
    fixed_min = registry_entry.get("fixed_min")
    fixed_max = registry_entry.get("fixed_max")
    
    # FIXED MODE: Use definitive natural bounds
    if mode == "fixed" and is_bounded:
        return (float(fixed_min), float(fixed_max))
        
    # Collect all valid values across data sources for AUTO/SHARED
    all_values = []
    for df in data_sources:
        if df is not None and not df.empty and col in df.columns:
            vals = df[col].dropna()
            if not vals.empty:
                all_values.append(vals)
                
    if not all_values:
        return None
        
    combined = pd.concat(all_values)
    vmin = float(combined.min())
    vmax = float(combined.max())
    
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
        
    # Apply symmetric padding
    span = vmax - vmin
    pad = span * padding
    vmin -= pad
    vmax += pad
    
    # Clamp to fixed bounds if they exist
    if fixed_min is not None:
        vmin = max(vmin, float(fixed_min))
    if fixed_max is not None:
        vmax = min(vmax, float(fixed_max))
        
    return (vmin, vmax)