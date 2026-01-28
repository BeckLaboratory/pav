"""Figures for comparing FWD/FWD-REV/REV states for query vs reference sequences.

Uses density models to compare the density of k-mers in the query and reference sequences. Useful
for visualizing intra-alignment INV and alignment-independent visualization of variant regions.
"""

"""Inversion figures."""

__all__ = [
    'kde_density_base',
]

import numpy as np
import polars as pl

from ..region import Region
from .. import util


def kde_density_base(
        df_kde: pl.DataFrame | pl.LazyFrame,
        region_qry: Region,
        figsize: tuple[int, int] = (7, 7),
        dpi: int = 300,
        flank_whiskers: bool = False
):
    """Get a base k-mer density plot using a k-mer density DataFrame.

    :param df_kde: Inversion call DataFrame.
    :param region_qry: Query region plot is generated over. Must match the region in `df_kde`.
    :param figsize: Figure size (width, height).
    :param dpi: Figure DPI.
    :param flank_whiskers: If `True`, show whiskers above or below points indicating if they match the upstream or
        downstream flanking inverted duplication.

    :returns: Plot figure object.
    """
    if isinstance(df_kde, pl.LazyFrame):
        df_kde = df_kde.collect()

    mpl = util.require_optional_module('matplotlib')
    plt = util.require_optional_module('matplotlib.pyplot')

    # Make figure
    fig = plt.figure(figsize=figsize, dpi=dpi)

    ax1, ax2 = fig.subplots(2, 1)

    # Flanking DUP whiskers
    if flank_whiskers:
        raise NotImplementedError('Flanking whiskers are not yet implemented.')
        # for index, subdf in df_kde.loc[
        #     ~ pd.isnull(df_kde['MATCH'])
        # ].groupby(
        #     ['STATE', 'MATCH']
        # ):
        #
        #     # Whisker y-values
        #     ymin, ymax = sorted(
        #         [
        #             -1 * (subdf.iloc[0]['STATE_MER'] - 1),
        #             -1 * (subdf.iloc[0]['STATE_MER'] - 1) + (0.25 if index[1] == 'SAME' else -0.25)
        #         ]
        #     )
        #
        #     # Plot whiskers
        #     ax1.vlines(
        #         x=subdf['INDEX'] + region_qry.pos,
        #         ymin=ymin, ymax=ymax,
        #         color='dimgray',
        #         linestyles='solid',
        #         linewidth=0.5
        #     )

    # Points (top pane)
    df_kde_0 = df_kde.filter(pl.col('state_mer') == 0)
    df_kde_1 = df_kde.filter(pl.col('state_mer') == 1)
    df_kde_2 = df_kde.filter(pl.col('state_mer') == 2)

    ax1.scatter(
        x=df_kde_0['index'] + region_qry.pos,
        y=df_kde_0['state_mer'],
        color='blue',
        alpha=0.2
    )

    ax1.scatter(
        x=df_kde_1['index'] + region_qry.pos,
        y=df_kde_1['state_mer'],
        color='purple',
        alpha=0.2
    )

    ax1.scatter(
        x=df_kde_2['index'] + region_qry.pos,
        y=df_kde_2['state_mer'],
        color='red',
        alpha=0.2
    )

    # Max density line (smoothed state call)
    ax1.plot(
        df_kde['index'] + region_qry.pos,
        df_kde['state'],
        color='black'
    )

    # Plot aestetics
    ax1.get_xaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )

    ax1.set_yticks(np.asarray([0, 1, 2]))
    ax1.set_yticklabels(np.array(['Fwd', 'Fwd+Rev', 'Rev']))

    # Density (bottom pane)
    ax2.plot(
        df_kde_0['index'] + region_qry.pos,
        df_kde_0['kde_fwd'],
        color='blue'
    )

    ax2.plot(
        df_kde_1['index'] + region_qry.pos,
        df_kde_1['kde_fwdrev'],
        color='purple'
    )

    ax2.plot(
        df_kde_2['index'] + region_qry.pos,
        df_kde_2['kde_rev'],
        color='red'
    )

    # Plot aestetics
    ax2.get_xaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )

    ax2.set_xlabel('{} ({:,d} - {:,d})'.format(
        region_qry.chrom,
        region_qry.pos + 1,
        region_qry.end
    ))

    ax2.set_ylabel('K-mer Density')

    ax1.tick_params(labelbottom=False)

    for label in ax2.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')

    fig.tight_layout()

    return fig
