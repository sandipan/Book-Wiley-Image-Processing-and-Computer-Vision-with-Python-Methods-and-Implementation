r"""
ieee_style.py
=============
Reusable matplotlib style helpers for IEEE-conference/journal-quality figures.

IEEE Transactions-style column widths (inches), Times-compatible serif fonts,
a colorblind-safe categorical palette (Okabe & Ito, 2008), panel labels
(a)/(b)/(c)..., and a save helper that writes both a vector PDF (for LaTeX
\includegraphics) and a high-DPI PNG (for quick preview / slides).

Usage
-----
    from ieee_style import set_ieee_style, fig_size, IEEE_COLORS, label_panels, savefig_ieee

    set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=fig_size("double", aspect=0.42))
    ...
    label_panels(axes)
    savefig_ieee(fig, "outputs/my_figure")   # writes my_figure.pdf and my_figure.png
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# IEEE column widths (inches). Most IEEE Transactions templates use a
# two-column layout: single-column figures should be ~3.5in wide, figures
# spanning both columns ~7.16in wide.
# ---------------------------------------------------------------------------
IEEE_SINGLE_COL = 3.5
IEEE_DOUBLE_COL = 7.16

# Colorblind-safe categorical palette (Okabe & Ito, 2008) -- the standard
# choice for accessible scientific figures.
IEEE_COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
# Convenience ordered list for cycling through categories
IEEE_PALETTE = [
    IEEE_COLORS["blue"], IEEE_COLORS["vermillion"], IEEE_COLORS["bluish_green"],
    IEEE_COLORS["reddish_purple"], IEEE_COLORS["orange"], IEEE_COLORS["sky_blue"],
]

# Colorblind-safe, perceptually-uniform sequential/diverging colormaps for
# imshow/contour/heatmap-style panels (viridis family is standard for IEEE Vis
# and is safe for grayscale print reproduction too).
IEEE_CMAP_SEQUENTIAL = "viridis"
IEEE_CMAP_UNCERTAINTY = "magma"
IEEE_CMAP_DIVERGING = "RdBu_r"


def set_ieee_style(base_font_size=8, use_tex=False):
    """
    Configure matplotlib rcParams for IEEE-style figures: Times-compatible
    serif text, STIX math (visually matches Times), small publication font
    sizes, thin lines, minimal chartjunk (no top/right spines by default via
    `style_axis`), and white background suitable for print.

    Parameters
    ----------
    base_font_size : int
        Body font size in points. IEEE body text is ~9-10pt in a column
        ~3.5in wide, so 7-8pt is the right scale for axis/tick labels.
    use_tex : bool
        If True, render text with a real LaTeX installation (requires a
        working `latex` binary on PATH). Left off by default since most
        environments (including this sandbox) don't have LaTeX installed;
        the STIX mathtext fallback below looks very close to Times/Computer
        Modern without that dependency.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)  # start from a clean slate

    mpl.rcParams.update({
        # ---- fonts: Times-compatible serif everywhere -----------------
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",       # STIX math closely matches Times
        "text.usetex": use_tex,
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "figure.titlesize": base_font_size + 2,

        # ---- lines / markers: thin, print-friendly ---------------------
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.6,
        "axes.linewidth": 0.7,

        # ---- axes: minimal chartjunk, white background ------------------
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": True,
        "grid.color": "#b0b0b0",
        "grid.linewidth": 0.4,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # ---- ticks -------------------------------------------------------
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        # ---- legend --------------------------------------------------------
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#888888",
        "legend.fancybox": False,

        # ---- output resolution ---------------------------------------------
        "figure.dpi": 150,       # on-screen
        "savefig.dpi": 600,      # print-quality raster export
        "pdf.fonttype": 42,      # embed fonts as TrueType (editable text in PDF)
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    # set the default color cycle to the colorblind-safe palette
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=IEEE_PALETTE)


def fig_size(width="single", aspect=0.75, height=None):
    """
    Return an IEEE-appropriate (width, height) figsize tuple in inches.

    Parameters
    ----------
    width : {"single", "double"} or float
        "single" -> 3.5in (one IEEE column), "double" -> 7.16in (full page
        width), or pass an explicit width in inches.
    aspect : float
        height = width * aspect (ignored if `height` is given explicitly).
    height : float, optional
        Explicit height in inches, overrides `aspect`.
    """
    if width == "single":
        w = IEEE_SINGLE_COL
    elif width == "double":
        w = IEEE_DOUBLE_COL
    else:
        w = float(width)
    h = height if height is not None else w * aspect
    return (w, h)


def style_axis(ax, grid=True):
    """Apply the minimal-chartjunk IEEE look to a single axis (useful when
    an axis was created by a library, e.g. ArviZ, that sets its own style)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.7)
    ax.tick_params(width=0.6, length=3)
    if grid:
        ax.grid(True, color="#b0b0b0", linewidth=0.4, linestyle=":", alpha=0.6)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    return ax


def label_panels(axes, labels=None, loc=(-0.14, 1.06), fontsize=9, fontweight="bold"):
    """
    Add IEEE-style panel labels "(a)", "(b)", "(c)", ... to each axis in
    `axes` (a flat list/array of matplotlib Axes), positioned just outside
    the top-left corner of each panel -- the standard convention for
    multi-panel IEEE figures.
    """
    import numpy as np
    axes_flat = np.atleast_1d(axes).ravel()
    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(len(axes_flat))]
    for ax, lab in zip(axes_flat, labels):
        ax.text(loc[0], loc[1], lab, transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight, va="bottom", ha="left")
    return axes_flat


def add_colorbar(fig, im, ax, label="", fraction=0.046, pad=0.04):
    """Thin, IEEE-proportioned colorbar with a label, attached to `ax`."""
    cbar = fig.colorbar(im, ax=ax, fraction=fraction, pad=pad)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2)
    cbar.outline.set_linewidth(0.6)
    if label:
        cbar.set_label(label, fontsize=8)
    return cbar


def savefig_ieee(fig, path_no_ext, formats=("pdf", "png")):
    """Save `fig` to `path_no_ext.<ext>` for each format in `formats`
    (default: a vector PDF for papers/LaTeX, plus a high-DPI PNG preview)."""
    saved = []
    for ext in formats[1:]:
        out = f"{path_no_ext}.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
        saved.append(out)
    return saved


if __name__ == "__main__":
    # Minimal self-test: render a small multi-panel demo figure so the style
    # can be sanity-checked without depending on the rest of the project.
    import numpy as np
    import os

    set_ieee_style()
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)

    fig, axes = plt.subplots(1, 2, figsize=fig_size("double", aspect=0.40))

    ax = axes[0]
    for i, label in enumerate(["MAP", "MMSE", "truth"]):
        ax.plot(x, np.sin(x + i) + 0.05 * i, label=label)
    ax.set_xlabel("Pixel index")
    ax.set_ylabel("Intensity")
    ax.legend()
    style_axis(ax)

    ax = axes[1]
    im = ax.imshow(rng.normal(size=(20, 20)), cmap=IEEE_CMAP_SEQUENTIAL)
    ax.set_xticks([]); ax.set_yticks([])
    add_colorbar(fig, im, ax, label="posterior std")

    label_panels(axes)
    os.makedirs("figs", exist_ok=True)
    paths = savefig_ieee(fig, "figs/ieee_style_selftest")
    print("wrote:", paths)