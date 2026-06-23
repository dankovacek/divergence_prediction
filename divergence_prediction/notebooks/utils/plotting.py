"""Bokeh visualisation helpers for performance analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import export_png
from bokeh.layouts import gridplot, row as bk_row
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    CustomJSTickFormatter,
    Label,
    LinearColorMapper,
    Plot,
    Range1d,
)
from bokeh.palettes import Category10, Viridis256
from bokeh.plotting import figure
from bokeh.transform import transform as bk_transform
from scipy.stats import kendalltau

try:
    from config import (
        COL_GAUGE_ID,
        COL_METHOD,
        COL_REGION,
        METHOD_DIRECT,
        METHOD_MLE,
        METHOD_MOM,
        METHOD_NAIVE,
        METHOD_NULL,
        METHOD_NULL_GLOBAL,
    )
except ImportError:
    COL_GAUGE_ID = 'Official_ID'
    COL_METHOD = 'Label'
    COL_REGION = 'region'
    METHOD_MLE = 'MLE'
    METHOD_DIRECT = 'PredictedLog'
    METHOD_MOM = 'PredictedMOM'
    METHOD_NAIVE = 'Mean_PMF'
    METHOD_NULL = 'Uniform'
    METHOD_NULL_GLOBAL = 'RandomDraw'

REPO_ROOT = Path(__file__).resolve().parents[1]

# Academic serif font stack used for all exported figures
ACADEMIC_FONT = "EB Garamond, Palatino Linotype, Palatino, Georgia, serif"

# ---------------------------------------------------------------------------
# Method style and label registries
# ---------------------------------------------------------------------------

_METHOD_PALETTE = Category10[8]

METHOD_STYLES: dict[str, dict] = {
    METHOD_MLE:         {'color': '#F244D5', 'line_dash': 'solid',   'line_width': 2.0},
    METHOD_DIRECT:      {'color': _METHOD_PALETTE[2], 'line_dash': 'solid',   'line_width': 2.0},
    METHOD_MOM:         {'color': _METHOD_PALETTE[1], 'line_dash': 'solid',   'line_width': 2.0},
    METHOD_NAIVE:       {'color': _METHOD_PALETTE[3], 'line_dash': 'dotted',  'line_width': 2.0},
    METHOD_NULL:        {'color': '#424242',           'line_dash': 'dotted',  'line_width': 1.5},
    METHOD_NULL_GLOBAL: {'color': '#030303',           'line_dash': 'dashed',  'line_width': 1.5},
}

METHOD_LABELS: dict[str, str] = {
    METHOD_MLE:         'MLE',
    METHOD_DIRECT:      'PARAM',
    METHOD_MOM:         'MOM',
    METHOD_NAIVE:       '10NN',
    METHOD_NULL:        'NAV(R)',
    METHOD_NULL_GLOBAL: 'NAV(P)',
}

METRIC_LABELS: dict[str, str] = {
    'w1':   'W1',
    'kld':  'KLD',
    'ed':   'ED',
    'isd':  'ISD',
    'rmse': 'RMSE',
}

METRIC_AXIS_LABELS: dict[str, str] = {
    'w1':   'W1 (-)',
    'kld':  'KLD (bits)',
    'ed':   'ED (-)',
    'isd':  'ISD (-)',
    'rmse': 'RMSE (-)',
}


# ---------------------------------------------------------------------------
# Style and export helpers
# ---------------------------------------------------------------------------

def apply_tufte_style(
    fig: Plot,
    font: str = ACADEMIC_FONT,
    title_font_size: str = '16pt',
    legend_text_font_size: str ='14pt',
) -> None:
    """Apply Tufte-inspired minimal style and academic serif font to a Bokeh figure."""
    fig.background_fill_color = None
    fig.border_fill_color = None
    fig.outline_line_color = None

    fig.grid.grid_line_color = "#CECECE"
    fig.grid.grid_line_alpha = 0.6
    fig.grid.grid_line_width = 0.7

    fig.axis.axis_line_color = '#333333'
    fig.axis.minor_tick_line_color = None
    fig.axis.major_tick_line_color = '#333333'
    fig.axis.major_tick_out = 4
    fig.axis.major_tick_in = 0
    
    fig.axis.axis_label_text_font = font
    fig.axis.axis_label_text_font_style = 'normal'
    fig.axis.major_label_text_font = font
    if legend_text_font_size is not None:
        fig.legend.label_text_font = font
        fig.legend.label_text_font_size = legend_text_font_size
    fig.title.text_font = font
    # fig.title.text_font_style = 'normal'
    # fig.title.text_font_size = title_font_size


def save_figure(layout, figure_name: str, tufte: bool = True) -> Path:
    """Save a Bokeh layout as PNG to docs/notebooks/images/<figure_name>.png."""
    if tufte:
        for obj in layout.references():
            if isinstance(obj, Plot):
                apply_tufte_style(obj)
    out_path = REPO_ROOT / 'images' / f'{figure_name}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_png(layout, filename=out_path)
    print(f'Saved: {out_path}')
    return out_path


def _compute_ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute an ECDF without pulling notebook data helpers into plotting."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    sorted_vals = np.sort(arr)
    probs = np.arange(1, arr.size + 1) / arr.size
    return sorted_vals, probs


def build_bitrate_kld_gridplot(
    bitrates: list[int],
    bitrate_dfs: dict[int, pd.DataFrame],
    mle_dfs_discrete: dict[int, pd.DataFrame],
    oracle_dict: dict[int, dict[str, dict[str, float]]],
    benchmark_styles: dict[str, tuple[str, str, str]],
    bitrate_colors: list[str],
    k_values: list[int],
    show_mle: bool = True,
    show_oracle: bool = True,
    export_path: str | Path | None = None,
):
    """Build the bitrate-sensitivity KLD grid and optional export layout."""
    frame_width, frame_height = 400, 460
    plots = []
    metric = 'KLD'

    for col_idx, bitrate in enumerate(bitrates):
        plot = figure(
            x_axis_type='log',
            frame_width=frame_width,
            frame_height=frame_height,
            x_range=(1e-3, 1e1),
            y_range=(0, 1.05),
            title=f'{bitrate} bits',
            x_axis_label='KLD [bits]',
            y_axis_label='CDF' if col_idx == 0 else '',
            min_border_left=10,
            min_border_bottom=10,
        )

        if show_mle:
            df_mle_ref = mle_dfs_discrete[bitrate]
            if metric in df_mle_ref.columns:
                x_vals, y_vals = _compute_ecdf(df_mle_ref[metric].dropna().values)
                if x_vals.size > 1:
                    plot.line(
                        x_vals,
                        y_vals,
                        line_width=5,
                        color=bitrate_colors[-1],
                        line_dash='dotted',
                        alpha=0.5,
                    )

        if show_oracle and bitrate in oracle_dict:
            oracle_best = pd.DataFrame(oracle_dict[bitrate]).T
            if metric in oracle_best.columns:
                x_vals, y_vals = _compute_ecdf(oracle_best[metric].values)
                style = benchmark_styles['Oracle']
                plot.line(
                    x_vals,
                    y_vals,
                    line_width=3,
                    color=bitrate_colors[-1],
                    line_dash=style[1],
                    alpha=0.8,
                )

        for index, k_value in enumerate(k_values):
            df_k = bitrate_dfs[bitrate][bitrate_dfs[bitrate]['k'] == k_value]
            if len(df_k) == 0 or metric not in df_k.columns:
                continue
            x_vals, y_vals = _compute_ecdf(df_k[metric].dropna().values)
            if x_vals.size == 0:
                continue
            plot.line(x_vals, y_vals, line_width=3, color=bitrate_colors[index], alpha=0.9)

        apply_tufte_style(plot)
        if col_idx > 0:
            plot.yaxis.major_tick_line_color = None
            plot.yaxis.minor_tick_line_color = None
            plot.yaxis.axis_label = None
            plot.yaxis.visible = False
        plot.axis.major_label_text_font_size = '24pt'
        plot.axis.axis_label_text_font_size = '32pt'
        plot.title.text_font_size = '32pt'
        plots.append(plot)

    grid = gridplot(plots, ncols=len(bitrates), toolbar_location=None)

    legend_fig = figure(
        width=frame_width,
        height=frame_height + 100,
        toolbar_location=None,
        x_range=(0, 1),
        y_range=(0, 1),
    )
    legend_fig.outline_line_color = None
    legend_fig.xaxis.visible = False
    legend_fig.yaxis.visible = False
    legend_fig.xgrid.visible = False
    legend_fig.ygrid.visible = False

    text_font = ACADEMIC_FONT
    text_font_size = '28pt'
    x_pos = 0.2
    y_pos = 1.0
    for index, k_value in enumerate(k_values):
        y_pos -= 0.08
        legend_fig.line([0.05, 0.15], [y_pos, y_pos], line_width=4, color=bitrate_colors[index])
        legend_fig.add_layout(
            Label(
                x=x_pos,
                y=y_pos,
                text=f'{k_value} NN',
                text_font=text_font,
                text_font_size=text_font_size,
                text_baseline='middle',
            )
        )

    if show_mle:
        y_pos -= 0.1
        legend_fig.line([0.05, 0.15], [y_pos, y_pos], line_width=5, color='black', line_dash='dotted')
        legend_fig.add_layout(
            Label(
                x=x_pos,
                y=y_pos,
                text='LN-MLE',
                text_font=text_font,
                text_font_size=text_font_size,
                text_baseline='middle',
            )
        )

    if show_oracle:
        y_pos -= 0.08
        style = benchmark_styles['Oracle']
        legend_fig.line([0.05, 0.15], [y_pos, y_pos], line_width=3, color=bitrate_colors[-1], line_dash=style[1])
        legend_fig.add_layout(
            Label(
                x=x_pos,
                y=y_pos,
                text='1NN Oracle',
                text_font=text_font,
                text_font_size=text_font_size,
                text_baseline='middle',
            )
        )

    layout = bk_row(grid, legend_fig)
    if export_path is not None:
        export_png(layout, filename=str(export_path))
        print(f'\nSaved to {export_path}')
    return layout


# ---------------------------------------------------------------------------
# Metric CDF panels
# ---------------------------------------------------------------------------

NULL_METHODS = {METHOD_NULL, METHOD_NULL_GLOBAL}


def metric_cdf_panel(
    df: pd.DataFrame,
    region: str,
    metric: str,
    x_label: str,
    method_styles: dict,
    method_labels: dict,
    x_log: bool = False,
    x_range: tuple[float, float] | None = None,
    show_legend: bool = True,
    show_y: bool = True,
    show_x: bool = True,
    grid_alpha: float = 0.3,
) -> figure:
    """CDF panel for one metric and region, coloured by method.

    Draws a grey band spanning MLE (best) to null model (worst) as context,
    then overlays CDF lines and median markers for each non-null method.
    """
    if metric == 'nse':
        x_label = ' (1 - NSE)'
    n_stns = len(set(df[COL_GAUGE_ID]))
    p = figure(
        frame_width=280, height=270,
        x_axis_type='log' if x_log else 'linear',
        title='',
        x_axis_label=x_label,
        y_axis_label=f'{region.upper()}: (N={n_stns}) \n Cumulative probability',
        x_range=x_range,
    )

    null_key = METHOD_NULL_GLOBAL if df[COL_REGION].nunique() > 1 else METHOD_NULL
    mle_raw  = df.loc[df[COL_METHOD] == METHOD_MLE,  metric].dropna()
    null_raw = df.loc[df[COL_METHOD] == null_key,    metric].dropna()
    if not mle_raw.empty and not null_raw.empty:
        mv = (1 - mle_raw  if metric == 'nse' else mle_raw).sort_values().values
        nv = (1 - null_raw if metric == 'nse' else null_raw).sort_values().values
        mc = np.arange(1, len(mv) + 1) / len(mv)
        nc = np.arange(1, len(nv) + 1) / len(nv)
        pos = np.concatenate([mv[mv > 0], nv[nv > 0]])
        if pos.size > 0:
            xlo, xhi = float(pos.min()), float(pos.max())
            xg = (np.logspace(np.log10(xlo), np.log10(xhi), 500) if x_log
                  else np.linspace(xlo, xhi, 500))
            ym = np.interp(xg, mv, mc, left=0.0, right=1.0)
            yn = np.interp(xg, nv, nc, left=0.0, right=1.0)
            vr = p.varea(x=xg, y1=yn, y2=ym, fill_color='#B3B3B3', alpha=0.7,
                         legend_label='MLE-NAIVE range')
            vr.level = 'underlay'

    if metric == 'nse':
        p.varea(x=[1, 10], y1=0, y2=1, fill_color='#F5B7B1', alpha=0.4,
                legend_label='NSE <= 0')

    for method, style in method_styles.items():
        if method in {METHOD_MLE} | NULL_METHODS:
            continue
        vals = df.loc[df['method'] == method, metric].dropna()
        if vals.empty:
            continue
        vals = 1 - vals if metric == 'nse' else vals
        vals = vals.sort_values()
        cdf    = np.arange(1, len(vals) + 1) / len(vals)
        median = vals.median()
        lbl    = f'{method_labels[method]}'
        p.line(vals.values, cdf, color=style['color'], line_dash=style['line_dash'],
               line_width=style['line_width'], legend_label=lbl)
        p.line([median, median], [0, 1], color=style['color'], line_dash='dashed',
               line_width=2.5, legend_label=lbl)

    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    p.legend.background_fill_alpha = 0.2
    # make the gridlines a little darker
    grid_color = "#696969"
    p.xgrid.grid_line_color = grid_color
    p.ygrid.grid_line_color = grid_color
    # increase alpha on gridlines
    p.xgrid.grid_line_alpha = grid_alpha
    p.ygrid.grid_line_alpha = grid_alpha
    p.xgrid.grid_line_width = 1.5
    p.ygrid.grid_line_width = 1.5
    if not show_y:
        p.yaxis.visible = False
        p.legend.visible = False
    if not show_x:
        p.xaxis.visible = False
    if not show_legend:
        p.legend.visible = False
    return p


# ---------------------------------------------------------------------------
# Headroom hexbin scatter
# ---------------------------------------------------------------------------

def headroom_hexbin_panel(
    gap_df: pd.DataFrame,
    metric: str,
    mle_bounds: tuple[float, float],
    ratio_bounds: tuple[float, float],
    title: str = '',
    x_label: str = '',
    y_label: str = '',
    hex_size: float = 0.1,
) -> figure:
    """Hexbin density (log10 space): MLE score (x) vs baseline/MLE ratio (y).

    White trend line shows binned median ratio, revealing whether harder
    stations have more or less headroom above the baseline.
    """
    xlo, xhi = mle_bounds
    ylo, yhi = ratio_bounds
    sub = gap_df[[f'{metric}_mle', f'{metric}_ratio']].dropna()
    sub = sub[(sub[f'{metric}_mle'] > 0) & (sub[f'{metric}_ratio'] > 0)]

    lx = np.log10(sub[f'{metric}_mle'].values)
    ly = np.log10(sub[f'{metric}_ratio'].values)
    lxlo, lxhi = np.log10(xlo), np.log10(xhi)
    lylo, lyhi = np.log10(ylo), np.log10(yhi)

    fmt = CustomJSTickFormatter(code="""
        var v = Math.pow(10, tick);
        if (v >= 10) return v.toFixed(0);
        if (v >= 1)  return v.toPrecision(2);
        return v.toExponential(0);
    """)

    p = figure(
        frame_width=210, height=210,
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        x_range=Range1d(lxlo, lxhi),
        y_range=Range1d(lylo, lyhi),
        toolbar_location=None,
    )
    p.xaxis.formatter = fmt
    p.yaxis.formatter = fmt
    p.line([lxlo, lxhi], [0, 0], color='#cc3333', line_dash='dashed', line_width=1.5, alpha=0.85)
    p.hexbin(lx, ly, size=hex_size, palette=Viridis256[::-1], line_color=None)

    edges = np.linspace(lxlo, lxhi, 21)
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (lx >= lo) & (lx < hi)
        if mask.sum() >= 8:
            bx.append(0.5 * (lo + hi))
            by.append(float(np.median(ly[mask])))
    if len(bx) > 2:
        p.line(bx, by, color='white', line_width=2.0, alpha=0.85)
    return p


# ---------------------------------------------------------------------------
# Skill score CDF panel
# ---------------------------------------------------------------------------

def skill_cdf_panel(
    skill_df: pd.DataFrame,
    n_hyd: pd.Series,
    baseline_name: str,
    title: str,
    region_colors: dict[str, str],
    regions: list[str],
    skill_clip: tuple[float, float] = (-1.9, 2.2),
    show_y: bool = True,
    cdf_envelope: pd.DataFrame | None = None,
    boot_ci_pos: float | None = None,
    boot_ci_half: float | None = None,
    grid_alpha: float = 0.3,
) -> figure:
    """CDF of skill scores with Path A CI annotations and optional Path B bootstrap envelope.

    Prints Path A (complete-year block count) CI statistics to stdout.
    skill_df must have columns 'skill' and 'region', indexed by gauge_id.
    cdf_envelope (optional): DataFrame with columns x, ci_lo, ci_hi on the P(S <= x) scale.
    boot_ci_pos / boot_ci_half: Path B bootstrap std of replicate percentages (optional).
    """
    sk_ny    = skill_df.join(n_hyd.rename('n_hyd_years'), how='left')
    n_eff    = sk_ny['n_hyd_years'].sum()
    pct_pos  = 100 * ((skill_df['skill'] > 0)   & (skill_df['skill'] <= 1)).mean()
    pct_half = 100 * ((skill_df['skill'] >= 0.5) & (skill_df['skill'] <= 1)).mean()
    z        = 1.96
    ci_pos_a  = z * np.sqrt((pct_pos  / 100) * (1 - pct_pos  / 100) / n_eff) * 100
    ci_half_a = z * np.sqrt((pct_half / 100) * (1 - pct_half / 100) / n_eff) * 100
    print(f'N_eff={n_eff:.0f}  0<S<=1: {pct_pos:.1f}% (Path A \u00b1{ci_pos_a:.1f}pp'
          + (f', Path B \u00b1{boot_ci_pos:.1f}pp' if boot_ci_pos is not None else ')')
          + f'  0.5<=S<=1: {pct_half:.1f}%')

    path_b_pos  = f'  B:\u00b1{boot_ci_pos:.1f}'  if boot_ci_pos  is not None else ''
    path_b_half = f'  B:\u00b1{boot_ci_half:.1f}' if boot_ci_half is not None else ''
    x_label       = f'S = ({baseline_name} \u2212 PARAM) / ({baseline_name} \u2212 MLE)'
    x_label_right = skill_clip[1] - 0.05

    p = figure(
        frame_width=320, height=300,
        title=title,
        x_axis_label=x_label,
        y_axis_label='P(S \u2265 s)' if show_y else '',
        x_range=Range1d(*skill_clip),
        toolbar_location=None,
    )
    p.line([0, 0], [0, 1], color='#666666', line_dash='dashed', line_width=1.2, alpha=0.8)
    p.line([1, 1], [0, 1], color='#666666', line_dash='dashed', line_width=1.2, alpha=0.8)

    for region in regions:
        vals = skill_df.loc[skill_df[COL_REGION] == region, 'skill'].sort_values().values
        if len(vals) < 5:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        p.line(vals, cdf, color=region_colors[region], line_width=2, alpha=0.85,
               legend_label=region)

    # Path B: bootstrap CDF envelope drawn before pooled line so line sits on top.
    # Envelope is stored as P(S <= x); chart shows P(S >= x), so we flip the band.
    if cdf_envelope is not None:
        env     = cdf_envelope.sort_values('x')
        surv_lo = 1 - env['ci_hi'].values
        surv_hi = 1 - env['ci_lo'].values
        p.varea(x=env['x'].values, y1=surv_lo, y2=surv_hi,
                fill_color='black', alpha=0.10, legend_label='boot 95% CI')

    pool     = skill_df['skill'].sort_values().values
    pool_cdf = np.arange(1, len(pool) + 1) / len(pool)
    p.line(pool, pool_cdf, color='black', line_dash='dashed', line_width=2.5, alpha=0.9,
           legend_label='pooled')

    if not show_y:
        p.yaxis.visible = False
    p.legend.location = 'top_left'
    p.legend.background_fill_alpha = 0.15
    p.legend.label_text_font_size = '8pt'
    p.x_grid.grid_alpha = grid_alpha
    p.y_grid.grid_alpha = grid_alpha

    null_colour = '#B9B9B9'
    p.varea(x=[skill_clip[0], 0], y1=0, y2=1, fill_color=null_colour, alpha=0.4)
    p.varea(x=[1, skill_clip[1]], y1=0, y2=1, fill_color=null_colour, alpha=0.4)

    p.add_layout(Label(
        x=x_label_right, y=0.13,
        text=f'0<S\u22641: {pct_pos:.0f}%  [A:\u00b1{ci_pos_a:.1f}{path_b_pos} pp]',
        text_align='right', text_baseline='middle',
        text_font_size='11px', text_color='#333333',
    ))
    p.add_layout(Label(
        x=x_label_right, y=0.05,
        text=f'0.5\u2264S\u22641: {pct_half:.0f}%  [A:\u00b1{ci_half_a:.1f}{path_b_half} pp]',
        text_align='right', text_baseline='middle',
        text_font_size='11px', text_color='#333333',
    ))
    return p


# ---------------------------------------------------------------------------
# Method vs naive scatter
# ---------------------------------------------------------------------------

def method_vs_naive_scatter(
    df: pd.DataFrame,
    metric: str,
    method: str,
    bounds: tuple[float, float],
    region_colors: dict[str, str],
    title: str = '',
    x_label: str = '',
    y_label: str = '',
    show_legend: bool = False,
) -> figure:
    """Scatter of method metric (y) vs naive metric (x), coloured by region."""
    lo, hi = bounds
    p = figure(
        frame_width=180, height=200,
        x_axis_type='log',
        y_axis_type='log',
        x_range=Range1d(lo, hi),
        y_range=Range1d(lo, hi),
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
    )
    p.line([lo, hi], [lo, hi], color='#555555', line_dash='dashed', line_width=1, alpha=0.6)
    for region, color in region_colors.items():
        sub = df[df[COL_REGION] == region]
        if sub.empty:
            continue
        p.scatter(sub[f'{metric}_naive'].values, sub[metric].values,
                  color=color, alpha=0.35, size=4, legend_label=region)
    p.legend.visible = show_legend
    if show_legend:
        p.legend.location = 'top_left'
        p.legend.background_fill_alpha = 0.6
        p.legend.label_text_font_size = '8pt'
    return p


# ---------------------------------------------------------------------------
# Skew vs divergence scatter rows
# ---------------------------------------------------------------------------

def divergence_vs_skew_heat_row(
    df: pd.DataFrame,
    metrics: list[str],
    x_col: str,
    heat_col_map: dict[str, str],
    metric_labels: dict[str, str],
    y_col_map: dict[str, str] | None = None,
    region_col: str | None = None,
    region_order: list[str] | None = None,
    show_region_legend: bool = False,
    x_label: str = 'Sample skew',
    heat_label: str = 'Heat value',
    title_prefix: str = '',
    clip_quantiles: tuple[float, float] = (0.02, 0.98),
    frame_width: int = 220,
    frame_height: int = 220,
    point_size: float = 4.0,
    point_alpha: float = 0.7,
    y_axis_type: str = 'linear',
) -> Plot:
    """Build one gridplot row: sample skew vs divergence metrics with heat colors.

    Parameters
    ----------
    df : table indexed by station with columns needed for x, y, and heat variables
    metrics : divergence metric keys to render as separate panels
    x_col : x-axis column shared across all panels
    heat_col_map : metric -> heat column used to color dots in each panel
    metric_labels : metric -> short label for panel title
    y_col_map : optional metric -> y column mapping, defaults to metric name
    region_col : optional region column for per-region renderer grouping
    region_order : optional ordered list of region names for legend display
    show_region_legend : show clickable region legend on first panel
    """
    y_col_map = y_col_map or {m: m for m in metrics}

    heat_vals: list[np.ndarray] = []
    for m in metrics:
        h_col = heat_col_map.get(m)
        if h_col is None or h_col not in df.columns:
            continue
        hv = df[h_col].values.astype(float)
        hv = hv[np.isfinite(hv)]
        if hv.size:
            heat_vals.append(hv)
    if not heat_vals:
        raise ValueError('No finite heat-map values found for requested metrics')

    h_all = np.concatenate(heat_vals)
    q_lo, q_hi = clip_quantiles
    h_lo, h_hi = np.quantile(h_all, [q_lo, q_hi])
    if not np.isfinite(h_lo) or not np.isfinite(h_hi) or h_lo == h_hi:
        h_lo, h_hi = float(np.nanmin(h_all)), float(np.nanmax(h_all))
        if h_lo == h_hi:
            h_hi = h_lo + 1e-9
    mapper = LinearColorMapper(palette=Viridis256, low=float(h_lo), high=float(h_hi), nan_color='#d9d9d9')

    row: list[figure] = []
    for mi, m in enumerate(metrics):
        y_col = y_col_map.get(m, m)
        h_col = heat_col_map.get(m)
        if y_col not in df.columns or h_col is None or h_col not in df.columns:
            continue

        required_cols = [x_col, y_col, h_col]
        if region_col is not None and region_col in df.columns:
            required_cols.append(region_col)
        sub = df[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue

        p = figure(
            frame_width=frame_width,
            frame_height=frame_height,
            title=f"{title_prefix}{metric_labels.get(m, m.upper())}",
            x_axis_label=x_label,
            y_axis_label=metric_labels.get(m, m.upper()) if mi == 0 else '',
            y_axis_type=y_axis_type,
            toolbar_location=None,
        )
        has_region = region_col is not None and region_col in sub.columns
        if has_region:
            ordered_regions = region_order or sorted(sub[region_col].astype(str).unique().tolist())
            seen_regions = set(sub[region_col].astype(str).unique())
            for region in ordered_regions:
                if region not in seen_regions:
                    continue
                sub_r = sub[sub[region_col].astype(str) == region]
                if sub_r.empty:
                    continue
                src = ColumnDataSource({
                    'x': sub_r[x_col].values,
                    'y': sub_r[y_col].values,
                    'heat': sub_r[h_col].values,
                })
                scatter_kwargs = {
                    'x': 'x',
                    'y': 'y',
                    'source': src,
                    'size': point_size,
                    'alpha': point_alpha,
                    'line_color': None,
                    'fill_color': bk_transform('heat', mapper),
                    'name': f'region::{region}',
                }
                if show_region_legend and mi == 0:
                    scatter_kwargs['legend_label'] = str(region)
                p.scatter(**scatter_kwargs)
        else:
            src = ColumnDataSource({
                'x': sub[x_col].values,
                'y': sub[y_col].values,
                'heat': sub[h_col].values,
            })
            p.scatter(
                x='x',
                y='y',
                source=src,
                size=point_size,
                alpha=point_alpha,
                line_color=None,
                fill_color=bk_transform('heat', mapper),
            )
        p.grid.grid_line_alpha = 0.25

        if show_region_legend and mi == 0 and has_region and p.legend:
            p.legend.location = 'top_left'
            p.legend.background_fill_alpha = 0.2
            p.legend.label_text_font_size = '8pt'
            p.legend.click_policy = 'hide'

        if mi > 0:
            p.yaxis.axis_label = None

        if mi == len(metrics) - 1:
            cb = ColorBar(
                color_mapper=mapper,
                ticker=BasicTicker(desired_num_ticks=6),
                label_standoff=6,
                border_line_color=None,
                location=(0, 0),
                width=10,
                title=heat_label,
            )
            p.add_layout(cb, 'right')

        row.append(p)

    if not row:
        raise ValueError('No panels built. Check x, y, and heat column availability')
    return row


# ---------------------------------------------------------------------------
# Kendall tau heatmap
# ---------------------------------------------------------------------------

N_TAU_PAL  = 256
TAU_PALETTE = [
    f'#{r:02x}00{b:02x}'
    for r, b in zip(
        np.linspace(27, 255, N_TAU_PAL).astype(int),
        np.linspace(69, 255, N_TAU_PAL).astype(int),
    )
]


def pairwise_kendall_tau(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Lower-triangular pairwise Kendall tau matrix; diagonal = 1."""
    n   = len(cols)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i):
            x = df[cols[i]].dropna()
            y = df[cols[j]].reindex(x.index).dropna()
            x = x.reindex(y.index)
            if len(x) < 4:
                mat[i, j] = mat[j, i] = np.nan
                continue
            tau, _ = kendalltau(x.values, y.values)
            mat[i, j] = mat[j, i] = float(tau)
    return pd.DataFrame(mat, index=cols, columns=cols)


def tau_heatmap(
    tau_df: pd.DataFrame,
    labels: list[str],
    title: str = '',
    frame_size: int = 150,
    show_colorbar: bool = False,
) -> figure:
    """Lower-triangular Kendall tau heatmap with adaptive text colour."""
    mapper = LinearColorMapper(palette=TAU_PALETTE, low=0.0, high=1.0, nan_color='#eeeeee')

    xs, ys, vals, txts, txt_colors = [], [], [], [], []
    for i, rl in enumerate(labels):
        for j, cl in enumerate(labels):
            if j >= i:
                continue
            v = float(tau_df.loc[rl, cl])
            xs.append(cl)
            ys.append(rl)
            vals.append(v)
            txts.append(f'{v:.2f}' if np.isfinite(v) else '')
            txt_colors.append('#1a1a1a' if np.isfinite(v) and v > 0.6 else '#f0f0f0')

    src = ColumnDataSource({'x': xs, 'y': ys, 'tau': vals, 'text': txts, 'txt_color': txt_colors})
    p = figure(
        frame_width=frame_size, frame_height=frame_size,
        x_range=labels,
        y_range=list(reversed(labels)),
        title=title,
        toolbar_location=None,
    )
    p.rect(x='x', y='y', width=1, height=1, source=src,
           fill_color=bk_transform('tau', mapper), line_color='white', line_width=0.5)
    p.text(x='x', y='y', text='text', source=src,
           text_align='center', text_baseline='middle',
           text_font_size='8px', text_color='txt_color')
    p.axis.major_label_text_font_size = '8px'
    p.axis.major_tick_line_color      = None
    p.axis.minor_tick_line_color      = None
    p.grid.grid_line_color            = None
    p.outline_line_color              = '#cccccc'
    if show_colorbar:
        cb = ColorBar(
            color_mapper=mapper, ticker=BasicTicker(desired_num_ticks=6),
            label_standoff=6, border_line_color=None, location=(0, 0),
            width=10, title='Kendall \u03c4',
        )
        p.add_layout(cb, 'right')
    return p
