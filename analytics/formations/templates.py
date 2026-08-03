"""Formation template helpers."""

from __future__ import annotations

from mplsoccer import Pitch


def build_templates(pitch_length, pitch_width):
	"""Build the full set of mplsoccer formation templates."""

	pitch = Pitch(pitch_type="custom", pitch_length=pitch_length, pitch_width=pitch_width)
	df = pitch.formations_dataframe
	templates = {}
	for formation in pitch.formations:
		sub = df[(df["formation"] == formation) & (df["name"] != "GK")]
		if len(sub) != 10:
			continue
		templates[formation] = {
			"names": sub["name"].tolist(),
			"normal": sub[["x", "y"]].to_numpy(dtype=float),
			"flipped": sub[["x_flip", "y_flip"]].to_numpy(dtype=float),
		}
	return templates
