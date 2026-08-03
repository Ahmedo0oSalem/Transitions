"""EPV grid helpers."""

from __future__ import annotations

import numpy as np


def load_epv_grid(path):
	"""Load the EPV grid from CSV."""

	return np.loadtxt(path, delimiter=",")


def epv_value(epv_grid, x, y, pitch_length, pitch_width, direction):
	"""Lookup EPV at a location, oriented for the attacking direction."""

	n_rows, n_cols = epv_grid.shape
	gx = x if direction == 1 else (pitch_length - x)
	gy = y
	col = int(np.clip(gx / pitch_length * n_cols, 0, n_cols - 1))
	row = int(np.clip(gy / pitch_width * n_rows, 0, n_rows - 1))
	return float(epv_grid[row, col])
