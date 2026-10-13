"""Shared matplotlib style so all figures look consistent."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# one fixed colour per family (colour follows the entity, never its rank)
COLORS = {
    "powerlaw": "#2a78d6",     # blue
    "lognormal": "#eb6834",    # orange
    "exponential": "#1baf7a",  # aqua
    "student_t": "#6250d6",    # violet
    "reference": "#8a8984",    # grey for reference lines / bands
}
LABELS = {"powerlaw": "power law", "lognormal": "lognormal",
          "exponential": "exponential", "student_t": "Student-t(3) tail"}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "legend.frameon": False,
    "legend.fontsize": 8,
})
