"""Shared colors and restrained hatch encodings for paper bar charts.

Palette (2026-11-12): the baselines are three tonal steps of one slate blue (Dense darkest, solid; tile skipping mid,
"//"; gather-scatter light, "\\\\"), and WISEConv is the only saturated hue (the orange of the WISEConv frame in
fig:overview).  Adjacent pairs validated with the dataviz skill's validate_palette.js (normal vision >= 21.3, CVD >=
20.6); the low chroma of the baselines is deliberate and the hatches carry identity in print.  Hatches are drawn in a
per-path ink over the fill (``add_bar``).  Wasted work is a neutral gray with a dot texture, never a system color.
"""

from matplotlib.patches import Patch


COLORS = {
    "dense": "#2b3a4a",
    "tile_skip": "#6a86a0",
    "gather_scatter": "#b7c6d4",
    "wiseconv": "#dd7428",
}
HATCHES = {
    "dense": "",
    "tile_skip": "////",
    "gather_scatter": "\\\\\\\\",
    "wiseconv": "",
}
WASTED_COLOR = "#dddddd"
WASTED_HATCH = "...."
WASTED_INK = "#9a9a9a"
HATCH_INK = "#ffffff"
HATCH_INKS = {"tile_skip": "#ffffff", "gather_scatter": "#6a86a0"}
EDGE = "#2b2b2b"  # outline of the older figures (plot_e2e.py and friends)
INK = "#222222"  # text and axes
MUTED = "#6b6b68"  # secondary text, reference lines
IDEAL = "#14a05a"  # the Ideal (proportional) line: a hue no bar uses
IDEAL_LS = (0, (3.2, 1.3))


def add_bar(ax, x, height, width, style_key, *, bottom=0.0, linewidth=0.65, zorder=3, outline=False):
    """One categorical bar: the fill, then its hatch in a light ink (no outline unless ``outline``)."""
    bar = ax.bar(x, height, width, bottom=bottom, color=COLORS[style_key],
                 edgecolor=EDGE if outline else "none", linewidth=linewidth if outline else 0, zorder=zorder)[0]
    if HATCHES[style_key]:
        ax.bar(x, height, width, bottom=bottom, color="none", edgecolor=HATCH_INKS.get(style_key, HATCH_INK),
               hatch=HATCHES[style_key],
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
        Patch(facecolor=COLORS[key], edgecolor=EDGE if outline else HATCH_INKS.get(key, HATCH_INK), hatch=HATCHES[key],
              linewidth=0.55 if outline else 0)
        for key in style_keys
    ]


def wasted_handle():
    return Patch(facecolor=WASTED_COLOR, edgecolor=WASTED_INK, hatch=WASTED_HATCH, linewidth=0)
