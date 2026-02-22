"""Shared notebook styling — FT-inspired warm palette with clean typography."""
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# FT-inspired color palette
FT_BG = '#FFF1E5'        # warm cream background
FT_DARK = '#33302E'       # near-black text
FT_BLUE = '#0D7680'       # teal accent
FT_RED = '#CC0000'        # loss / danger
FT_GREEN = '#09814A'      # gain / profit
FT_ORANGE = '#F2DFCE'     # light accent
FT_GREY = '#A3A09E'       # secondary text

CRASH_PERIODS = [
    ('2008 GFC',   '2007-10-01', '2009-03-09', '#CC0000'),
    ('2020 COVID', '2020-02-19', '2020-03-23', '#FF8833'),
    ('2022 Bear',  '2022-01-03', '2022-10-12', '#9467bd'),
]

def apply_style():
    """Apply FT-inspired matplotlib style."""
    sns.set_theme(style='white', palette='muted')
    mpl.rcParams.update({
        'figure.facecolor': FT_BG,
        'axes.facecolor': FT_BG,
        'savefig.facecolor': FT_BG,
        'text.color': FT_DARK,
        'axes.labelcolor': FT_DARK,
        'xtick.color': FT_DARK,
        'ytick.color': FT_DARK,
        'axes.edgecolor': FT_GREY,
        'grid.color': '#E0D5C5',
        'grid.alpha': 0.6,
        'axes.grid': True,
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'figure.figsize': (16, 6),
        'figure.dpi': 110,
        'legend.framealpha': 0.9,
        'legend.edgecolor': FT_GREY,
    })


def shade_crashes(ax, alpha=0.15):
    """Add crash period shading."""
    for label, start, end, color in CRASH_PERIODS:
        import pandas as pd
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=alpha, color=color, label=label)


def fmt_pct(val):
    """Format percentage with color."""
    if val > 0:
        return f'<span style="color:{FT_GREEN};font-weight:bold">+{val:.2f}%</span>'
    elif val < 0:
        return f'<span style="color:{FT_RED}">−{abs(val):.2f}%</span>'
    return f'{val:.2f}%'


def style_returns_table(styler):
    """Apply FT-style to a pandas Styler."""
    return (styler
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#0D7680'), ('color', 'white'),
                ('font-weight', 'bold'), ('text-align', 'center'),
                ('padding', '8px 12px'), ('border-bottom', '2px solid #33302E')]},
            {'selector': 'td', 'props': [
                ('padding', '6px 12px'), ('border-bottom', f'1px solid {FT_ORANGE}')]},
            {'selector': 'tr:hover td', 'props': [
                ('background-color', FT_ORANGE)]},
            {'selector': 'caption', 'props': [
                ('font-size', '14px'), ('font-weight', 'bold'),
                ('color', FT_DARK), ('padding', '10px 0')]},
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'), ('font-family', 'Georgia, serif')]},
        ])
    )


def color_excess(val):
    """Color excess returns green/red."""
    if isinstance(val, (int, float)):
        if val > 0.1: return f'color: {FT_GREEN}; font-weight: bold'
        if val < -0.1: return f'color: {FT_RED}'
    return ''
