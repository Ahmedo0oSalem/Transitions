"""Formation matching helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def match_formation(player_xy, templates, orientation):
	"""Find the best matching formation template for a window."""

	best_formation, best_cost, best_names = None, np.inf, None

	for formation, tmpl in templates.items():
		template_xy = tmpl[orientation]
		cost_matrix = cdist(player_xy, template_xy)
		row_ind, col_ind = linear_sum_assignment(cost_matrix)
		cost = cost_matrix[row_ind, col_ind].sum()
		norm_cost = cost / len(row_ind)

		if norm_cost < best_cost:
			best_cost = norm_cost
			best_formation = formation
			best_names = [tmpl["names"][i] for i in col_ind]

	return best_formation, best_cost, best_names
