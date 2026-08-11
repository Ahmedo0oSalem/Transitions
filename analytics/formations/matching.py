"""Formation matching helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def match_formation(player_xy, templates, orientation, align=True):
    """Find the best matching formation template for a window.

    When *align* is True, each template is translated so that its centroid
    coincides with the players' centroid before matching.  The reported cost
    then measures shape mismatch only (not positional offset), which is
    essential for fluid in-possession positions that are rarely centred on
    the template's default location.

    Returns
    -------
    (best_formation, best_cost, best_names, second_cost) where *best_cost* is
    the mean per-player distance of the best template and *second_cost* the
    mean per-player distance of the runner-up template.
    """

    best_formation, best_cost, best_names = None, np.inf, None
    second_cost = np.inf

    if align:
        player_centroid = np.asarray(player_xy).mean(axis=0)

    for formation, tmpl in templates.items():
        template_xy = np.asarray(tmpl[orientation], dtype=float)
        if align:
            template_xy = template_xy + (player_centroid - template_xy.mean(axis=0))
        cost_matrix = cdist(player_xy, template_xy)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        cost = cost_matrix[row_ind, col_ind].sum()
        norm_cost = cost / len(row_ind)

        if norm_cost < best_cost:
            second_cost = best_cost
            best_cost = norm_cost
            best_formation = formation
            best_names = [tmpl["names"][i] for i in col_ind]
        elif norm_cost < second_cost:
            second_cost = norm_cost

    return best_formation, best_cost, best_names, second_cost
