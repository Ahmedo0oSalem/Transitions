"""Formation hierarchy derivation and color logic.

Every raw formation name from mplsoccer (e.g. ``"3511flat"``) can be
structured into three levels:

  formation
    The raw string as written to ``formations.csv`` (e.g. ``"3511flat"``).

  variant
    The ``flat``-suffix stripped (e.g. ``"3511"``).  Two formations that
    differ only in the ``flat`` annotation represent the same tactical
    shape — the ``flat`` qualifier is a detail of the template's
    numbering scheme, not a distinct formation identity.  By collapsing
    to *variant* we merge those rows in the piano-roll view and pool
    their votes in the smoothed view.

  family
    Derived from the variant's **first digit**, which encodes the number
    of defensive-line players (e.g. ``"3xxx"`` → ``"back-3"``).  A small
    override dict handles the handful of non-numeric mplsoccer names
    (``pyramid``, ``metodo``, ``wm``).

The three levels are stored in every segment/bar dict regardless of
which layout consumes them, so the consumer can choose which key to
group, sort, or colour by.
"""

from __future__ import annotations

import colorsys

import matplotlib.colors as mcolors

# ====================================================================
# Hierarchy derivation
# ====================================================================

# Tiny override for non-numeric mplsoccer names that can't be parsed
# by the first-digit rule below.
FAMILY_OVERRIDE: dict[str, str] = {
    "pyramid": "back-2",   # 2-3-5 pyramid
    "metodo":  "back-3",   # 3-2-5 Metodo
    "wm":      "back-3",   # 3-2-2-3 WM
}

# Canonical display order — families are ordered by back-line count.
FAMILY_ORDER: list[str] = [
    "back-1", "back-2", "back-3", "back-4", "back-5",
]

# Fallback for anything that doesn't match (shouldn't happen in practice).
_FALLBACK_FAMILY = "other"


def infer_family(variant: str) -> str:
    """Return the defensive-line family for a *variant* name.

    The rule is: extract the first character of *variant* and produce
    ``"back-{digit}"``.  The override dict catches non-numeric names.
    """
    if variant in FAMILY_OVERRIDE:
        return FAMILY_OVERRIDE[variant]
    if variant and variant[0].isdigit():
        return f"back-{variant[0]}"
    return _FALLBACK_FAMILY


def derive_hierarchy(raw_formation: str) -> dict[str, str]:
    """Map a raw formation string to its three hierarchy levels.

    Returns ``{"formation": …, "variant": …, "family": …}``.
    """
    variant = raw_formation.removesuffix("flat")
    family = infer_family(variant)
    return {"formation": raw_formation, "variant": variant, "family": family}


# ====================================================================
# Colour logic — one hue per family, shades per variant
# ====================================================================

# Each family gets a base HLS hue (0-1 range).  These are hand-picked
# to be perceptually distinct.
_FAMILY_HUES: dict[str, float] = {
    "back-1": 0.75,   # purple
    "back-2": 0.50,   # teal
    "back-3": 0.58,   # blue
    "back-4": 0.33,   # green
    "back-5": 0.08,   # orange
}

_FAMILY_SATURATION = 0.55
_VARIANT_LIGHTNESS_MIN = 0.28
_VARIANT_LIGHTNESS_MAX = 0.78


def _variant_lightness(index: int, count: int) -> float:
    """Spread *count* variants evenly across the lightness range."""
    if count <= 1:
        return (_VARIANT_LIGHTNESS_MIN + _VARIANT_LIGHTNESS_MAX) / 2
    return _VARIANT_LIGHTNESS_MIN + (
        _VARIANT_LIGHTNESS_MAX - _VARIANT_LIGHTNESS_MIN
    ) * index / (count - 1)


def build_family_color_map(
    segments: list[dict],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build two colour maps from a sequence of segment/bar dicts.

    Each dict must carry at least a ``"variant"`` key (or fall back to
    ``"formation"``) and a ``"family"`` key.

    Returns
    -------
    variant_color_map : dict[str, str]
        Maps ``variant`` (e.g. ``"3511"``) to a hex color string.
        Different variants in the same family get different lightness
        levels of the same base hue.
    family_color_map : dict[str, str]
        Maps ``family`` (e.g. ``"back-3"``) to a single mid-lightness
        hex string, useful for overview-mode colouring or legends.
    """
    family_variants: dict[str, set[str]] = {}
    for s in segments:
        fam = s.get("family", _FALLBACK_FAMILY)
        var = s.get("variant", s.get("formation", ""))
        family_variants.setdefault(fam, set()).add(var)

    var_color: dict[str, str] = {}
    fam_color: dict[str, str] = {}

    # Process families in canonical order so the hue assignment is stable.
    for fam in FAMILY_ORDER:
        if fam not in family_variants:
            continue
        variants = sorted(family_variants[fam])
        hue = _FAMILY_HUES.get(fam, 0.0)
        n = len(variants)
        for i, var in enumerate(variants):
            lightness = _variant_lightness(i, n)
            r, g, b = colorsys.hls_to_rgb(hue, lightness, _FAMILY_SATURATION)
            var_color[var] = mcolors.to_hex((r, g, b))
        # Family swatch = mid lightness
        r, g, b = colorsys.hls_to_rgb(hue, 0.55, _FAMILY_SATURATION)
        fam_color[fam] = mcolors.to_hex((r, g, b))

    # Any remaining families not in the canonical list get a fallback.
    for fam in family_variants:
        if fam not in fam_color:
            variants = sorted(family_variants[fam])
            hue = 0.0
            n = len(variants)
            for i, var in enumerate(variants):
                lightness = _variant_lightness(i, n)
                r, g, b = colorsys.hls_to_rgb(hue, lightness, _FAMILY_SATURATION)
                var_color[var] = mcolors.to_hex((r, g, b))
            r, g, b = colorsys.hls_to_rgb(hue, 0.55, _FAMILY_SATURATION)
            fam_color[fam] = mcolors.to_hex((r, g, b))

    return var_color, fam_color
