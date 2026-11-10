"""Shared colors and restrained hatch encodings for paper bar charts.

Palette (2026-11-10): four categorical hues of matched chroma, validated with the dataviz skill's
validate_palette.js on a white surface (all pairs: CVD Delta E >= 9.9, normal vision >= 18.6, contrast >= 3:1).
WISEConv is the only warm hue.  Hatches are the secondary encoding for print and color-vision deficiency; draw them
in a light ink over the fill (``add_bar``) rather than in the outline color.  Wasted work is a neutral gray with a
dot texture, never a system color.
"""

from matplotlib.patches import Patch


COLORS = {
    "dense": "#2a5c9e",
    "tile_skip": "#26a08f",
    "gather_scatter": "#9877cc",
    "wiseconv": "#dd7428",
}
HATCHES = {
    "dense": "",
    "tile_skip": "////",
    "gather_scatter": "\\\\\\\\",
    "wiseconv": "",
}
WASTED_COLOR = "#e4e4e2"
WASTED_HATCH = "...."
WASTED_INK = "#a8a8a4"
HATCH_INK = "#ffffff"
EDGE = "#2b2b2b"  # outline of the older figures (plot_e2e.py and friends)
INK = "#222222"  # text and axes
MUTED = "#6b6b68"  # secondary text, reference lines


def add_bar(ax, x, height, width, style_key, *, bottom=0.0, linewidth=0.65, zorder=3, outline=False):
    """One categorical bar: the fill, then its hatch in a light ink (no outline unless ``outline``)."""
    bar = ax.bar(x, height, width, bottom=bottom, color=COLORS[style_key],
                 edgecolor=EDGE if outline else "none", linewidth=linewidth if outline else 0, zorder=zorder)[0]
    if HATCHES[style_key]:
        ax.bar(x, height, width, bottom=bottom, color="none", edgecolor=HATCH_INK, hatch=HATCHES[style_key],
               linewidth=0, zorder=zorder + 0.1)
    return bar


def add_wasted(ax, x, height, width, *, bottom, zorder=3):
    """The wasted share of a bar's throughput: neutral gray with a dot texture."""
    bar = ax.bar(x, height, width, bottom=bottom, color=WASTED_COLOR, edgecolor="none", linewidth=0, zorder=zorder)[0]
    ax.bar(x, height, width, bottom=bottom, color="none", edgecolor=WASTED_INK, hatch=WASTED_HATCH, linewidth=0,
           zorder=zorder + 0.1)
    return bar


def legend_handles(style_keys, *, outline=False):
    return [
        Patch(facecolor=COLORS[key], edgecolor=EDGE if outline else HATCH_INK, hatch=HATCHES[key],
              linewidth=0.55 if outline else 0)
        for key in style_keys
    ]


def wasted_handle():
    return Patch(facecolor=WASTED_COLOR, edgecolor=WASTED_INK, hatch=WASTED_HATCH, linewidth=0)
