"""Shared colors and restrained hatch encodings for paper bar charts."""

from matplotlib.patches import Patch


COLORS = {
    "dense": "#66c2a5",
    "tile_skip": "#fc8d62",
    "gather_scatter": "#8da0cb",
    "wiseconv": "#f6d186",
}
HATCHES = {
    "dense": "",
    "tile_skip": "//",
    "gather_scatter": "\\\\",
    "wiseconv": "xx",
}
EDGE = "#2b2b2b"


def add_bar(ax, x, height, width, style_key, *, linewidth=0.65, zorder=3):
    """Draw one consistently styled categorical bar."""
    return ax.bar(
        x,
        height,
        width,
        color=COLORS[style_key],
        edgecolor=EDGE,
        hatch=HATCHES[style_key],
        linewidth=linewidth,
        zorder=zorder,
    )[0]


def legend_handles(style_keys):
    return [
        Patch(
            facecolor=COLORS[key],
            edgecolor=EDGE,
            hatch=HATCHES[key],
            linewidth=0.55,
        )
        for key in style_keys
    ]
