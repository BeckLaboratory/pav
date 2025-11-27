"""Routines for calling inversions."""

__all__ = [
    'KDE_STATE_FWD',
    'KDE_STATE_FWDREV',
    'KDE_STATE_REV',
    'try_intra_region',
    'get_inv_row',
    'get_state_table',
    'test_kde',
    'cluster_table',
    'cluster_table_insdel',
    'cluster_table_sig'
]

from typing import Any, Optional

import agglovar
from dataclasses import dataclass, field
import numpy as np
import polars as pl
import scipy.stats

from . import const
from .align import lift
from .params import PavParams
from .region import Region
from .seq import ref_kmers, region_seq_fasta
from .kde import Kde, KdeTruncNorm, rl_encoder
from .io import NullWriter

# KDE states
KDE_STATE_FWD = 0
"""KDE state for forward-oriented matches between a query region and reference k-mers."""

KDE_STATE_FWDREV = 1
"""KDE state for forward- and reverse-oriented matches between a query region and reference k-mers (i.e. query """
"""k-mer matches a forward and reverse reference k-mer)."""

KDE_STATE_REV = 2
"""KDE state for reverse-oriented matches between a query region and reference k-mers."""


# Walk states
_WS_FLANK_L = 0  # Left flank (unique sequence)
_WS_REP_L = 1    # Left inverted repeat
_WS_INV = 2      # Inverted core
_WS_REP_R = 3    # Right inverted repeat
_WS_FLANK_R = 4  # Right flank (unique sequence)

# String representation for each walk state
_WS_STATE_REPR = {
    _WS_FLANK_L: 'FL',
    _WS_REP_L: 'RL',
    _WS_INV: 'V',
    _WS_REP_R: 'RR',
    _WS_FLANK_R: 'FR'
}

# Walk state to KDE table column (for scoring over the appropriate KDE state for a walk state)
_WALK_STATE_COL = {
    _WS_FLANK_L: 'kde_fwd',
    _WS_REP_L: 'kde_fwdrev',
    _WS_INV: 'kde_rev',
    _WS_REP_R: 'kde_fwdrev',
    _WS_FLANK_R: 'kde_fwd'
}


#
# Public functions
#

def try_intra_region(
        region_flag: Region,
        ref_fa_filename: str,
        qry_fa_filename: str,
        df_ref_fai: pl.DataFrame,
        df_qry_fai: pl.DataFrame,
        align_lift: lift.AlignLift,
        pav_params: Optional[PavParams] = None,
        k_util: agglovar.kmer.util.KmerUtil = None,
        kde_model: Kde = None,
        stop_on_lift_fail: bool = True,
        log_file=None,
) -> Optional[dict[str, Any]]:
    """Scan region for inversions.

    Start with a flagged region (`region_flag`) where variants indicated that an inversion might be. Scan that region
    for an inversion expanding as necessary.

    :param region_flag: Flagged region to begin scanning for an inversion.
    :param ref_fa_filename: Reference FASTA filename.
    :param qry_fa_filename: Query FASTA filename.
    :param df_ref_fai: Reference sequence lengths.
    :param df_qry_fai: Query sequence lengths.
    :param align_lift: Alignment lift-over tool (pav.align.AlignLift).
    :param pav_params: PAV parameters.
    :param k_util: K-mer utility. If `None`, a k-mer utility is created with PAV's default k-mer size
        (`pav3.const.K_SIZE`).
    :param kde_model: Kernel density estimator function. Expected to be a `pav3.kde.KdeTruncNorm` object, but can
        be any object with a similar signature. If `None`, a default `kde` estimator is used.
    :param stop_on_lift_fail: If `True`, stop if the reference-to-query lift fails for any reason (i.e. breakpoints
        missing alignment records). If `False`, keep expanding until expansion is exhausted (reaches maximum number
        of expansions or covers a whole reference region).
    :param log_file: Log file to write to. If `None`, no logging is performed.

    :returns: A dict of inversion table fields or None if no inversion was found. Does not include "id" or "call_source"
        items, these should be set after an inversion table is created.

    :raises ValueError: If parameters are not valid.
    """
    if log_file is None:
        log_file = NullWriter()

    # Check arguments
    if region_flag.pos_align_index is None or region_flag.end_align_index is None:
        raise ValueError('region_flag must have both pos_align_index and end_align_index')

    if region_flag.pos_align_index != region_flag.end_align_index:
        raise ValueError(
            f'region_flag must have the same pos_align_index ({region_flag.pos_align_index}) and end_align_index: '
            f'{region_flag.end_align_index}'
        )

    # Get parameters
    if pav_params is None:
        pav_params = PavParams()

    repeat_match_prop = pav_params.inv_repeat_match_prop
    min_inv_kmer_run = pav_params.inv_min_kmer_run
    min_qry_ref_prop = pav_params.inv_min_qry_ref_prop
    region_limit = pav_params.inv_region_limit
    min_expand = pav_params.inv_min_expand
    max_expand = np.inf
    init_expand = pav_params.inv_init_expand
    min_kmers = pav_params.inv_min_kmers
    max_ref_kmer_count = pav_params.inv_max_ref_kmer_count

    if k_util is None:
        k_util = agglovar.kmer.util.KmerUtil(pav_params.inv_k_size)

    if kde_model is None:
        kde_model = KdeTruncNorm()

    # Init
    region_ref = region_flag.expand(init_expand, min_pos=0, max_end=df_ref_fai, shift=True)

    inv_iterations = 0

    if max_expand is None:
        max_expand = np.inf

    # Scan and expand
    df_rl = None
    df_kde = None
    region_qry = None

    while inv_iterations <= min_expand and inv_iterations <= max_expand:

        # Expand from last search
        if inv_iterations > 0:
            last_len = len(region_ref)
            region_ref = _expand_region(region_ref, df_rl, df_ref_fai)

            if len(region_ref) == last_len:
                # Stop if expansion had no effect
                return None

        inv_iterations += 1

        # Reset, signal to code outside loop that the loop ended before df_rl is created
        # (delete from previous iteration)
        df_rl = None

        # Max region size
        if region_limit is not None and 0 < region_limit < len(region_ref):
            return None

        # Get query region
        region_qry = align_lift.region_to_qry(region_ref, same_index=True)

        # Check for expansion through a break in the query sequence
        if region_qry is None:
            if stop_on_lift_fail:
                return None
            continue

        # Create k-mer table
        df_kde = get_state_table(
            region_ref=region_ref, region_qry=region_qry,
            ref_fa_filename=ref_fa_filename, qry_fa_filename=qry_fa_filename,
            df_ref_fai=df_ref_fai, df_qry_fai=df_qry_fai,
            is_rev=region_qry.is_rev,
            k_util=k_util, kde_model=kde_model,
            max_ref_kmer_count=max_ref_kmer_count,
            expand_bound=True
        )

        if df_kde is None:
            # No matching k-mers, ref & query are too divergent, not likely an inversion
            return None

        if df_kde.shape[0] < min_kmers:
            # Not enough k-mers, expand to find more
            continue

        # Get run-length encoded states (list of (state, count) tuples).
        df_rl = rl_encoder(df_kde)

        # Done if reference oriented k-mers (state == 0) found on both sides
        if df_rl.height > 2 and df_rl[0, 'state'] == 0 and df_rl[-1, 'state'] == 0:
            break  # Stop searching and expanding

    # Stop if no inverted sequence was found
    if df_rl is None or not df_rl.select((pl.col('state') == 2).any()).item():
        log_file.write(f'Intra-align inversion region {region_flag}: No inverted states found: {region_ref}\n')
        log_file.flush()

        return None

    if df_rl[0, 'state'] != 0 or df_rl[-1, 'state'] != 0:
        log_file.write(f'Intra-align inversion region {region_flag}: No inversion found after expansions')

    # Estimate inversion breakpoints
    inv_walk = _resolve_inv_from_rl(
        df_rl, df_kde,
        repeat_match_prop=repeat_match_prop,
        min_inv_kmer_run=min_inv_kmer_run
    )

    if inv_walk is None:
        log_file.write(
            f'Intra-align inversion region {region_flag}: Failed to resolve inversion breakpoints: {region_ref}: '
            f'No walk along KDE states was found\n'
        )
        log_file.flush()

        return None

    region_qry_inner, region_qry_outer = _walk_to_regions(inv_walk, df_rl, region_qry, k_size=k_util.k_size)

    # Lift to reference
    region_ref_inner = align_lift.region_to_ref(region_qry_inner)

    if region_ref_inner is None:
        log_file.write(
            f'Intra-align inversion region {region_flag}: Failed lifting inner INV region to reference: '
            f'{region_qry_inner}\n'
        )
        log_file.flush()

        return None

    region_ref_outer = align_lift.region_to_ref(region_qry_outer)

    if region_ref_outer is None or not region_ref_outer.contains(region_ref_inner):
        region_ref_outer = region_ref_inner
        log_file.write(
            f'Intra-align inversion region {region_flag}: Failed lifting outer INV region to reference: '
            f'{region_qry_outer}: Using inner coordinates\n'
        )
        log_file.flush()

    # Check size proportions
    if len(region_ref_inner) < len(region_qry_inner) * min_qry_ref_prop:
        log_file.write(
            f'Intra-align inversion region {region_flag}: Reference region too short: '
            f'Reference region length ({len(region_ref_inner):,d}) is not within {min_qry_ref_prop * 100:.2f}% '
            f'of the query region length ({len(region_qry_inner):,d})\n'
        )
        log_file.flush()

        return None

    if len(region_qry_outer) < len(region_ref_outer) * min_qry_ref_prop:
        log_file.write(
            f'Intra-align inversion region {region_flag}: Query region too short: '
            f'Query region length ({len(region_qry_outer):,d}) is not within {min_qry_ref_prop * 100:.2f}% '
            f'of the reference region length ({len(region_ref_outer):,d})'
        )
        log_file.flush()

        return None

    # Return inversion call
    log_file.write(
        f'Intra-align inversion region {region_flag}: INV Found: '
        f'inner={region_ref_inner}, outer={region_ref_outer} '
        f'(qry inner={region_qry_inner}, qry_outer={region_qry_outer})'
    )
    log_file.flush()

    return get_inv_row(
        region_ref_inner, region_ref_outer,
        region_qry_inner, region_qry_outer,
        region_flag.pos_align_index
    )


def get_inv_row(
        region_ref_inner: Optional[Region] = None,
        region_ref_outer: Optional[Region] = None,
        region_qry_inner: Optional[Region] = None,
        region_qry_outer: Optional[Region] = None,
        align_index: Optional[int] = None,
) -> dict[str, Any]:
    """Format an inversion into a dict.

    Separated from try_inv_region() so that place-holder inv dicts can be generated for inversion tables with no
    records (can be used to get a list of fields that would be generated).

    :param region_ref_inner: Inner reference region.
    :param region_ref_outer: Outer reference region.
    :param region_qry_inner: Inner query region.
    :param region_qry_outer: Outer query region.
    :param align_index: Alignment index.
    """
    none_vals = [region is None for region in [
        region_ref_inner, region_ref_outer, region_qry_inner, region_qry_outer, align_index
    ]]

    if all(none_vals):
        region_ref_inner = Region('NA', 0, 0)
        region_ref_outer = Region('NA', 0, 0)
        region_qry_inner = Region('NA', 0, 0)
        region_qry_outer = Region('NA', 0, 0)
        align_index = -1

    elif any(none_vals):
        raise ValueError('All arguments may be None, or all arguments must be non-None')

    return {
        'chrom': region_ref_inner.chrom,
        'pos': region_ref_inner.pos,
        'end': region_ref_inner.end,
        'vartype': 'INV',
        'varlen': len(region_ref_inner),
        'align_index': align_index,
        'qry_id': region_qry_inner.chrom,
        'qry_pos': region_qry_inner.pos,
        'qry_end': region_qry_inner.end,
        'outer_ref': region_ref_outer.as_dict(),
        'outer_qry': region_qry_outer.as_dict()
    }


def get_state_table(
        region_ref: Region,
        region_qry: Region,
        ref_fa_filename: str,
        qry_fa_filename: str,
        df_ref_fai: pl.DataFrame,
        df_qry_fai: pl.DataFrame,
        is_rev: Optional[bool] = None,
        k_util: Optional[agglovar.kmer.util.KmerUtil] = None,
        kde_model: Optional[Kde] = None,
        max_ref_kmer_count: int = const.INV_MAX_REF_KMER_COUNT,
        expand_bound: bool = True,
        force_norm: bool = False,
) -> Optional[pl.DataFrame]:
    """Initialize the state table for inversion calling by k-mers.

    :param region_ref: Reference region.
    :param region_qry: Query region.
    :param ref_fa_filename: Reference FASTA file name.
    :param qry_fa_filename: Query FASTA file name.
    :param df_ref_fai: Reference lengths.
    :param df_qry_fai: Query lengths.
    :param is_rev: Set to `True` if the query is reverse-complemented relative to the reference. Reference k-mers are
        reverse-complemented to match the query sequence. If `None`, get from `region_qry`
    :param k_util: K-mer utility.
    :param kde_model: KDE model for kernel density estimates. If `None`, KDE is not applied.
    :param max_ref_kmer_count: Remove high-count kmers greater than this value.
    :param expand_bound: Expand reference and query regions to include the convolution bandwidth and shrink it back down
        after performing convolutions. This keeps the edges of region from dropping from convolutions, when set, the
        edge density will still sum to 1 (approximately, FFT methods are not exact).
    :param force_norm: Normalize across states so that the sum of KDE_FWD, KDE_FWDREV, and KDE_REV is always 1.0. This
        is not needed for most KDE models.

    :returns: Initialized KDE table or None if a table could not be created.

    :raises ValueError: `expand_bound` is `True` and either FAI table is missing.
    :raises ValueError: `k_util.k_bit_size` is greater than 64.
    """
    if k_util is None:
        k_util = agglovar.kmer.util.KmerUtil(const.INV_K_SIZE)

    if k_util.np_int_type is None:
        raise ValueError(
            f'K-mer size {k_util.k_size} exceeds maximum for k-mer arrays '
            f'(numpy unsigned integer types, max {agglovar.kmer.util.NP_MAX_KMER_SIZE} bp k-mers)'
        )

    if is_rev is None:
        is_rev = region_qry.is_rev

    # Expand regions by band_bound
    if expand_bound and kde_model is not None and kde_model.band_bound is not None:

        if df_ref_fai is None or df_qry_fai is None:
            raise ValueError('Expanding regions by KDE band-bounds requires reference and query FAIs tables')

        region_ref_exp = region_ref.expand(kde_model.band_bound * 2, max_end=df_ref_fai, shift=False)
        region_qry_exp = region_qry.expand(kde_model.band_bound * 2, max_end=df_qry_fai, shift=False)
    else:
        region_ref_exp = region_ref
        region_qry_exp = region_qry

    # Get reference k-mer counts
    ref_kmer_count = ref_kmers(region_ref_exp, ref_fa_filename, k_util)

    if ref_kmer_count is None or len(ref_kmer_count) == 0:
        return None

    ref_kmer_fwd = np.array(list(ref_kmer_count.keys()), dtype=k_util.np_int_type)
    ref_kmer_rev = k_util.rev_complement_array(ref_kmer_fwd)

    # Skip low-complexity sites with repetitive k-mers
    if max_ref_kmer_count > 0:

        kmer_pass = (
            np.array(list(ref_kmer_count.values()), dtype=np.int32) +
            np.vectorize(lambda x: ref_kmer_count.get(x, 0))(ref_kmer_rev)
        ) <= max_ref_kmer_count

        ref_kmer_fwd = ref_kmer_fwd[kmer_pass]
        ref_kmer_rev = ref_kmer_rev[kmer_pass]

    if is_rev:
        ref_kmer_fwd, ref_kmer_rev = ref_kmer_rev, ref_kmer_fwd

    # Get query k-mers as list
    seq_qry = region_seq_fasta(region_qry_exp, qry_fa_filename, rev_compl=False)

    df = pl.DataFrame(
        agglovar.kmer.util.stream_index(seq_qry, k_util),
        schema={'kmer': pl.UInt64, 'index': pl.UInt32}
    )

    if df.height == 0:
        return None

    max_index = df.select(pl.col('index').last()).item()

    df = (
        df
        .lazy()
        .with_columns(
            pl.col('kmer').is_in(ref_kmer_fwd).alias('kmer_fwd'),
            pl.col('kmer').is_in(ref_kmer_rev).alias('kmer_rev')
        )
        .select(
            pl.col('kmer'),
            pl.col('index'),
            pl.when(pl.col('kmer_fwd') & pl.col('kmer_rev')).then(2)
            .when(pl.col('kmer_fwd')).then(0)
            .when(pl.col('kmer_rev')).then(1)
            .otherwise(-1)
            .alias('state_mer')
        )
        .filter(pl.col('state_mer') >= 0)
        .with_row_index('_row_index')
        .collect()
    )

    # Apply KDE
    if kde_model is not None:
        df = (
            df.with_columns(
                kde_fwd=kde_model(df.filter(pl.col('state_mer') == 0).select('_row_index'), df.height),
                kde_fwdrev=kde_model(df.filter(pl.col('state_mer') == 1).select('_row_index'), df.height),
                kde_rev=kde_model(df.filter(pl.col('state_mer') == 2).select('_row_index'), df.height)
            )
            .with_columns(
                state=pl.concat_list(['kde_fwd', 'kde_fwdrev', 'kde_rev']).list.arg_max()
            )
        )

        if force_norm:
            df = df.with_columns(
                pl.col('kde_fwd') / pl.sum_horizontal(['kde_fwd', 'kde_fwdrev', 'kde_rev']),
                pl.col('kde_fwdrev') / pl.sum_horizontal(['kde_fwd', 'kde_fwdrev', 'kde_rev']),
                pl.col('kde_rev') / pl.sum_horizontal(['kde_fwd', 'kde_fwdrev', 'kde_rev'])
            )

    # Collapse ends expanded by kde.band_bound
    diff_l = region_qry.pos - region_qry_exp.pos
    diff_r = region_qry_exp.end - region_qry.end

    if diff_l > 0 or diff_r > 0:
        first_index = diff_l + 1
        last_index = max_index - diff_l + 1

        df = (
            df
            .filter(
                (pl.col('index') >= first_index) & (pl.col('index') <= last_index)
            )
            .with_columns(
                pl.col('index') - diff_l
            )
        )

    df = df.drop('_row_index')

    # Return dataframe
    return df


def test_kde(
        df_rl: pl.DataFrame,
        binom_prob: float = 0.5,
        two_sided: bool = True,
        trim_fwd: bool = True
) -> float:
    """Test KDE for inverted states and complete inversion.

    Must also check for the proportion of inverted states, test will produce a low p-value if most states are

    :param df_rl: Run-length encoded state table (Generated by pav3.kde.rl_encoder()).
    :param binom_prob: Probability of inverted states for a binomial test of equal frequency.
    :param two_sided: `True`, for a two-sided test (p-val * 2).
    :param trim_fwd: Trim FWD oriented states from the start and end of `df_rl` if `True`; do not penalize test for
        including flanking un-inverted sequence.

    :returns: Binomial p-value.
    """
    # Trim forward-oriented flanking sequence
    if trim_fwd:
        if df_rl.height > 0 and df_rl[0, 'state'] == KDE_STATE_FWD:
            df_rl = df_rl[1:]

        if df_rl.height > 0 and df_rl[-1, 'state'] == KDE_STATE_FWD:
            df_rl = df_rl[:-1]

    # Get probability of inverted states based on a bionmial test of equal frequency of forward and reverse states
    state_fwd_n = df_rl.filter(pl.col('state') == KDE_STATE_FWD).select(pl.col('len_kde').sum()).item()
    state_rev_n = df_rl.filter(pl.col('state') == KDE_STATE_REV).select(pl.col('len_kde').sum()).item()

    return float(
            (
                    1 - scipy.stats.binom.cdf(state_rev_n, state_rev_n + state_fwd_n, binom_prob)
            ) * (2 if two_sided else 1)
    )


def cluster_table(
        df_snv: pl.LazyFrame,
        df_insdel: pl.LazyFrame,
        df_ref_fai: pl.LazyFrame,
        df_qry_fai: pl.LazyFrame,
        pav_params: Optional[PavParams]
) -> pl.DataFrame:
    """Create a table of sites with clustered variants following patterns of intra-alignment inversions.

    The returned table contains the following columns:

        * chrom (str): Chromosome.
        * pos (int): Start position.
        * end (int): End position.
        * align_index (int): Alignment index.
        * flag (list[str]): Variant type (CLUSTER_SNV, CLUSTER_INSDEL, MATCH_INSDEL).

    Region types (flag field):

        * CLUSTER_SNVs: Clusters of SNVs.
        * CLUSTER_INSDEL: Clusters of insertions and deletions.
        * MATCH_INSDEL: Insertions and deletions in close proximity with similar size.

    :param df_snv: SNV intra-alignment variant table.
    :param df_insdel: Insertion and deletion intra-alignment variant table.
    :param df_ref_fai: Reference sequence lengths.
    :param df_qry_fai: Query sequence lengths.
    :param pav_params: PAV Parameters.

    :returns: Clustered variant table.
    """
    if pav_params is None:
        pav_params = PavParams()

    # Generate all clusters, concat into a single table
    df = (
        pl.concat(
            [
                (
                    cluster_table_sig(
                        df={'snv': df_snv, 'insdel': df_insdel}[vartype],
                        df_qry_fai=df_qry_fai,
                        cluster_flank=pav_params.inv_sig_cluster_flank,
                        varlen_min=pav_params.inv_sig_cluster_varlen_min,
                        varlen_max=50,
                        min_depth={
                            'snv':  pav_params.inv_sig_cluster_snv_min,
                            'insdel': pav_params.inv_sig_cluster_indel_min
                        }[vartype],
                        min_window_size=pav_params.inv_sig_cluster_win_min,
                    )
                    .with_columns(pl.lit([f'CLUSTER_{vartype.upper()}']).alias('flag'))
                    .select(['chrom', 'pos', 'end', 'align_index', 'flag'])
                    .lazy()
                )
                for vartype in ('snv', 'insdel')
            ] + [
                (
                    cluster_table_insdel(
                        df=df_insdel,
                        df_ref_fai=df_ref_fai,
                        varlen_min=50,
                        offset_prop_max=pav_params.inv_sig_insdel_offset_prop,
                        size_ro_min=pav_params.inv_sig_insdel_varlen_ro,
                    )
                    .with_columns(pl.lit(['MATCH_INSDEL']).alias('flag'))
                    .select(['chrom', 'pos', 'end', 'align_index', 'flag'])
                    .lazy()
                )
            ]
        )
        .sort(['chrom', 'pos', 'end'])
        .with_row_index('index')
    )

    # Self-join to find overlapping regions
    # df_inter = (
    #     df
    #     .join(
    #         df, how='cross'
    #     )
    #     .filter(
    #         pl.col('chrom') == pl.col('chrom_right'),
    #         pl.col('index') >= pl.col('index_right'),
    #         pl.col('pos') < pl.col('end_right') + pav_params.inv_sig_merge_flank,
    #         pl.col('end') > pl.col('pos_right') - pav_params.inv_sig_merge_flank,
    #     )
    #     # .join_where(
    #     #     df,
    #     #     pl.col('chrom') == pl.col('chrom_right'),
    #     #     pl.col('pos') < pl.col('end_right') + pav_params.inv_sig_merge_flank,
    #     #     pl.col('end') > pl.col('pos_right') - pav_params.inv_sig_merge_flank,
    #     # )
    #     .drop('chrom_right')
    #     .sort(['index', 'index_right'])
    # ).collect()

    df_inter = (
        agglovar.bed.join.pairwise_join(
            df.select('chrom', 'pos', 'end'),
            df.select(
                pl.col('chrom'),
                pl.col('pos') - pav_params.inv_sig_merge_flank,
                pl.col('end') + pav_params.inv_sig_merge_flank,
            ),
        )
        .filter(
            pl.col('index_a') < pl.col('index_b')
        )
        .select('index_a', 'index_b')
        .sort('index_a', 'index_b')
        .collect()
    )

    # Group by join. Dict key is the index, value is the group ID
    group_map = dict()

    for row in df_inter.iter_rows(named=True):
        if row['index_a'] not in group_map:
            group_map[row['index_a']] = row['index_a']

        group_map[row['index_b']] = group_map[row['index_a']]

    df_schema = df.collect_schema()

    df_mg = (
        df
        .join(
            pl.DataFrame(
                {
                    'index': group_map.keys(),
                    'group_id': group_map.values()
                },
                schema={'index': df_schema['index'], 'group_id': df_schema['index']}
            ).lazy(),
            on='index',
            how='inner'
        )
        .group_by('group_id')
        .agg(
            pl.col('chrom').first(),
            pl.col('pos').min(),
            pl.col('end').max(),
            pl.col('align_index').first(),
            pl.col('flag').flatten(),
            pl.len().alias('clusters'),
        )
        .with_columns(
            pl.col('flag').list.unique().list.sort(),
        )
        .drop('group_id')
        .sort(['chrom', 'pos', 'end', 'align_index'])
    ).collect()

    return df_mg


def cluster_table_insdel(
        df: pl.DataFrame | pl.LazyFrame,
        df_ref_fai: pl.DataFrame | pl.LazyFrame,
        varlen_min: int = 50,
        offset_prop_max: float = 2.0,
        size_ro_min: float = 0.8,
) -> pl.DataFrame:
    """Identify clusters of matching INS & DEL variants.

    When an alignment crosses an inversion and is not truncated (split into multiple alignment records), it often
    creates a pair of INS and DEL variants of similar size (ref allele deleted, inverted allele inserted). This
    function identifies candidate clusters of such variants.

    The returned table contains the following columns:
        * chrom (str): Chromosome.
        * pos (int): Start position.
        * end (int): End position.
        * align_index (int): Alignment index.
        * depth_max (int): Maximum depth in the cluster.

    :param df: Variant table. Must include INS and DEL variants.
    :param df_ref_fai: Reference sequence lengths.
    :param varlen_min: Minimum variant length (Ignore smaller variants).
    :param offset_prop_max: Maximum offset proportion between INS & DEL variants.
    :param size_ro_min: Minimum size overlap of INS & DEL variants.

    :returns: A table of clustered variants.
    """
    if not isinstance(df, pl.LazyFrame):
        df = df.lazy()

    if not isinstance(df_ref_fai, pl.LazyFrame):
        df_ref_fai = df_ref_fai.lazy()

    df_ins = df.filter((pl.col('vartype') == 'INS') & (pl.col('varlen') >= varlen_min))
    df_del = df.filter((pl.col('vartype') == 'DEL') & (pl.col('varlen') >= varlen_min))

    # Join INS and DEL by proximity.
    pairwise_intersect = agglovar.pairwise.overlap.PairwiseOverlap(
        (
            agglovar.pairwise.overlap.PairwiseOverlapStage(
                offset_prop_max=offset_prop_max,
                size_ro_min=size_ro_min,
                join_predicates=(
                    pl.col('align_index_a').list.first() == pl.col('align_index_b').list.first(),
                )
            ),
        ),
        join_cols=(
            pl.col('chrom_a').alias('chrom'),
            pl.col('pos_a'), pl.col('end_a'),
            pl.col('pos_b'), pl.col('end_b'),
            pl.col('varlen_a'),
            pl.col('varlen_b'),
            pl.col('align_index_a').list.first().alias('align_index'),
            pl.col('qry_id_a').alias('qry_id'),
        ),
    )

    # pairwise_intersect.append_join_predicates(
    #     pl.col('align_index_a').list.first() == pl.col('align_index_b').list.first()
    # )

    # pairwise_intersect.append_join_cols([
    #     pl.col('chrom_a').alias('chrom'),
    #     pl.col('pos_a'), pl.col('end_a'),
    #     pl.col('pos_b'), pl.col('end_b'),
    #     pl.col('varlen_a'),
    #     pl.col('varlen_b'),
    #     pl.col('align_index_a').list.first().alias('align_index'),
    #     pl.col('qry_id_a').alias('qry_id'),
    # ])

    df_join = (
        pairwise_intersect.join(df_ins, df_del)
        .select(
            pl.col('chrom'),
            pl.min_horizontal(pl.col('pos_a'), pl.col('pos_b')).alias('pos'),
            pl.max_horizontal(pl.col('end_a'), pl.col('end_b')).alias('end'),
            pl.col('align_index'),
            pl.concat_list('varlen_a', 'varlen_b').alias('varlen'),
            pl.col('index_a'),
            pl.col('qry_id'),
        )
        # .join(
        #     df_ins
        #     .with_row_index('index_a')
        #     .select(['chrom', 'index_a']),
        #     on='index_a', how='inner'
        # )
        .select(['chrom', 'pos', 'end', 'align_index', 'varlen', 'qry_id'])
    )

    # Group clusters into one record
    return (
        pl.concat(
            [
                (  # Start
                    df_join
                    .select(
                        pl.col('chrom'),
                        pl.col('pos').alias('loc'),
                        pl.col('align_index'),
                        pl.col('qry_id'),
                        pl.lit(1).alias('depth')
                    )
                ),
                (  # End position
                    df_join
                    .select(
                        pl.col('chrom'),
                        pl.col('end').alias('loc'),
                        pl.col('align_index'),
                        pl.col('qry_id'),
                        pl.lit(0).alias('depth')
                    )
                ),
                (  # Adjust depth after end position
                    df_join
                    .select(
                        pl.col('chrom'),
                        pl.col('end').alias('loc'),
                        pl.col('align_index'),
                        pl.col('qry_id'),
                        pl.lit(-1).alias('depth')
                    )
                ),
            ]
        )
        .sort(['align_index', 'loc', 'depth'], descending=[False, False, True])
        .with_columns(
            pl.col('depth').cum_sum().over('align_index').alias('depth')
        )
        .with_columns(
            (pl.col('depth') > 0).rle_id().over('align_index').alias('rle_id'),
        )
        .filter(pl.col('depth') > 0)
        .group_by(['align_index', 'rle_id'])
        .agg(
            pl.col('chrom').first(),
            pl.col('loc').min().alias('pos'),
            pl.col('loc').max().alias('end'),
            pl.col('qry_id').first(),
            pl.col('depth').max().alias('depth_max'),
        )
        .select(['chrom', 'pos', 'end', 'align_index', 'depth_max'])
        .sort(['chrom', 'pos', 'end', 'align_index'])
        .collect()
    )


def cluster_table_sig(
        df: pl.DataFrame | pl.LazyFrame,
        df_qry_fai: pl.DataFrame | pl.LazyFrame,
        cluster_flank: int = 100,
        varlen_min: int = 0,
        varlen_max: Optional[int] = 50,
        min_depth: int = 20,
        min_window_size: int = 500,
) -> pl.DataFrame:
    """Create a table of cluster signatures.

    Identify regions of heavily clustered variants, which often indicate an inversion that did not truncate an alignment
    record (i.e. clusters of SNVs and indels).

    The returned table contains the following columns:

        * chrom (str): Chromosome.
        * pos (int): Start position.
        * end (int): End position.
        * count (int): Number of variants contributing to the cluster.
        * depth_max (int): Maximum depth in the cluster.

    :param df: Variant table.
    :param df_ref_fai: Reference sequence lengths.
    :param df_qry_fai: Query sequence lengths.
    :param cluster_flank: Flank size for clustering. Added upstream and downstream of each variant midpoint before
        clustering.
    :param varlen_min: Minimum variant length. Ignored if "varlen" is not a column in `df`.
    :param varlen_max: Maximum variant length. Ignored if "varlen" is not a column in `df`.
    :param min_depth: Minimum depth for variant clusters.
    :param min_window_size: Minimum window size for clusters (end - pos).

    :returns: Clustered variant regions.
    """
    if not isinstance(df, pl.LazyFrame):
        df = df.lazy()

    if not isinstance(df_qry_fai, pl.LazyFrame):
        df_qry_fai = df_qry_fai.lazy()

    if 'varlen' in df.collect_schema().names():
        df = df.filter(pl.col('varlen') >= varlen_min)

        if varlen_max is not None:
            df = df.filter(pl.col('varlen') <= varlen_max)

    return (
        pl.concat(
            [
                (  # Start
                    df
                    .select(
                        pl.col('chrom'),
                        (pl.col('pos') - cluster_flank).alias('loc'),
                        pl.col('align_index').list.first(),
                        pl.col('qry_id'),
                        pl.lit(1).alias('depth')
                    )
                ),
                (  # End
                    df
                    .select(
                        pl.col('chrom'),
                        (pl.col('pos') - cluster_flank).alias('loc'),
                        pl.col('align_index').list.first(),
                        pl.col('qry_id'),
                        pl.lit(1).alias('depth')
                    )
                ),
                (  # Adjust depth after end position
                    df
                    .select(
                        pl.col('chrom'),
                        (pl.col('end') + cluster_flank).alias('loc'),
                        pl.col('align_index').list.first(),
                        pl.col('qry_id'),
                        pl.lit(-1).alias('depth')
                    )
                )
            ]
        )
        .sort(['align_index', 'loc', 'depth'], descending=[False, False, True])
        .with_columns(
            pl.col('depth').cum_sum().over('align_index').alias('depth')
        )
        .with_columns(
            (pl.col('depth') >= min_depth).rle_id().over('align_index').alias('rle_id'),
        )
        .group_by(['align_index', 'rle_id'])
        .agg(
            pl.col('chrom').first(),
            pl.col('loc').min().alias('pos'),
            pl.col('loc').max().alias('end'),
            pl.col('qry_id').first(),
            pl.col('depth').max().alias('depth_max'),
        )
        .join(
            df_qry_fai.select(pl.col('qry_id'), pl.col('len').alias('_qry_len')),
            on='qry_id',
            how='left'
        )
        .with_columns(
            pl.col('pos').clip(0, pl.col('_qry_len')).alias('pos'),
            pl.col('end').clip(0, pl.col('_qry_len')).alias('end'),
        )
        .filter(
            (pl.col('end') - pl.col('pos')) >= min_window_size
        )
        .select(['chrom', 'pos', 'end', 'depth_max', 'align_index'])
        .sort(['chrom', 'pos', 'end', 'align_index'])
        .collect()
    )


#
# Private functions and classes
#

def _score_range(
        df_kde: pl.DataFrame,
        df_rl: pl.DataFrame,
        rl_index: int,
        i: int,
        kde_col: str
) -> float:
    """Private function - Score a segment of the KDE between two run-length encoded positions.

    :param df_kde: KDE DataFrame.
    :param df_rl: Run-length encoded DataFrame.
    :param rl_index: Index of first run-length encoded record (inclusive).
    :param i: Index of last run-length encoded record (exclusive).
    :param kde_col: KDE column to sum.

    :returns: A sum of all KDE records in `df` between run-length encoded records at index `rl_index` and `i`
        The KDE state summed is `kde_col` (name of the column in `df` to sum).
    """
    return (
        df_kde[df_rl[rl_index, 'index_kde']: df_rl[i - 1, 'index_kde'] + df_rl[i - 1, 'len_kde']]
        .select(pl.col(kde_col).sum())
        .item()
    )


def _range_len(
        df_rl: pl.DataFrame,
        rl_index: int,
        i: int
):
    """Private function - Get the length (number of KDE records) between two run-length encoded records (inclusive).

    :param df_rl: Run-length encoded DataFrame.
    :param rl_index: Index of first run-length encoded record (inclusive).
    :param i: Index of last run-length encoded record (inclusive).

    :returns: Number of KDE records in the range `rl_index` to `i` (inclusive).
    """
    return df_rl[i - 1, 'index_kde'] + df_rl[i - 1, 'len_kde'] - df_rl[rl_index, 'index_kde']


NEXT_WALK_STATE = {
    (_WS_FLANK_L, KDE_STATE_FWDREV): _WS_REP_L,  # Left flank -> left inverted repeat
    (_WS_FLANK_L, KDE_STATE_REV): _WS_INV,       # Left flank -> inverted core (no left inverted repeat)
    (_WS_REP_L, KDE_STATE_REV): _WS_INV,         # Left inverted repeat -> inverted core
    (_WS_INV, KDE_STATE_FWDREV): _WS_REP_R,      # Inverted core -> right inverted repeat
    (_WS_INV, KDE_STATE_FWD): _WS_FLANK_R,       # Inverted core -> right flank (no right inverted repeat)
    (_WS_REP_R, KDE_STATE_FWD): _WS_FLANK_R      # Right inverted repeat -> right flank
}
"""State transition given the current walk state and the next KDE state. Only legal transitions are defined"""


@dataclass(repr=False)
class _InvWalkState(object):
    """
    Private class - Tracks the state of a walk from the left-most run-length (RL) encoded record to a current state.

    :ivar walk_state: Walk state ("WS_" constants)
    :ivar rl_index: Index of the RL-encoded DataFrame where the current state starts.
    :ivar walk_score: Cumulative score of the walk from the left-most RL record to the current state and position.
    :ivar l_rep_len: Length of the right inverted repeat locus. Value is 0 if there was none or the state has not
        yet reached the left inverted repeat. Used to give a score boost for records with similar-length inverted
        repeats on both sides.
    :ivar l_rep_score: Score of the right inverted repeat locus. Value is 0.0 if there was none or the state has
        not yet reached the left inverted repeat. Used to give a score boost for records with similar-length
        inverted repeats on both sides.
    :ivar trace_list: List of walk state transitions to the current state. Each element is a tuple of an RL table index
        and a walk state ("WS_" constants).
    """

    walk_state: int
    rl_index: int
    walk_score: float = 0.0
    l_rep_len: int = 0
    l_rep_score: float = 0.0
    trace_list: list[tuple[int, int]] = field(default_factory=list)

    def __repr__(self):
        """Get a string representation of this state."""
        trace_str = ', '.join([f'{te[0]}:{_WS_STATE_REPR[te[1]]}' for te in self.trace_list])
        return (
            f'InvWalkState(walk_state={self.walk_state}, '
            f'rl_index={self.rl_index}, '
            f'walk_score={self.walk_score}, '
            f'l_rep_len={self.l_rep_len}, '
            f'l_rep_score={self.l_rep_score}, '
            f'trace_list=[{trace_str}])'
        )


def _resolve_inv_from_rl(
        df_rl: pl.DataFrame,
        df_kde: pl.DataFrame,
        repeat_match_prop: float = 0.2,
        min_inv_kmer_run: int = const.INV_MIN_INV_KMER_RUN,
        final_state_node_list: Optional[list[_InvWalkState]] = None
):
    """Private function - Resolve the optimal walk along run-length (RL) encoded states.

    This walk sets the inversion breakpoints.

    :param df_rl: RL-encoded table.
    :param df_kde: State table after KDE.
    :param repeat_match_prop: Give a bonus to the a walk score for similar-length inverted repeats. The bonus is
        calculated by taking the minimum score from both left and right inverted repeat loci, multiplying by the
        length similarity (min/max length), and finally multipyling by this value. Set to 0.0 to disable the bonus.
    :param min_inv_kmer_run: Minimum number of k-mers in an inversion run. Set to 0 to disable the filter.
    :param final_state_node_list: If not None, the list is populated with all InvWalkState objects reaching a final
        state and is sorted by walk scores (highest scores first). This list is cleared by the method before
        exploring RL walks. This can be used to see alternative inversion structures.

    :returns: An InvWalkState object representing the best walk along RL states.
    """
    if repeat_match_prop is None:
        repeat_match_prop = 0.0

    if min_inv_kmer_run is None:
        min_inv_kmer_run = 0

    if final_state_node_list is not None:
        final_state_node_list.clear()

    if df_rl.select((pl.col('state') == 2).sum()).item() == 0:
        raise ValueError('No inverted segments found in the RL table')

    df_rl_i = df_rl.with_row_index('_index')

    # Location of the last inverted segment. Used to limit the search space in states before the inversion.
    max_inv_index = df_rl_i.filter(pl.col('state') == 2).select(pl.col('_index').max()).item()

    # Initialize first state
    walk_state_node_stack = [
        _InvWalkState(0, 0, 0.0, 0, 0.0, [(0, _WS_FLANK_L)])
    ]

    max_score = 0.0
    max_score_node = None

    while walk_state_node_stack:
        walk_state_node = walk_state_node_stack.pop()

        # Walk through downstream states
        for i in range(
            walk_state_node.rl_index + 1,
            (df_rl.height if walk_state_node.walk_state >= _WS_INV else max_inv_index)
        ):

            # KDE state and next walk state
            kde_state = df_rl_i[i, 'state']
            next_walk_state = NEXT_WALK_STATE.get((walk_state_node.walk_state, kde_state), None)

            if next_walk_state is None:
                continue  # No transition from the current walk state to this KDE state

            # Stop if inverted region is too short
            if (
                    walk_state_node.walk_state == _WS_INV and
                    min_inv_kmer_run > 0 and
                    _range_len(df_rl, walk_state_node.rl_index, i) < min_inv_kmer_run
            ):
                continue

            # Score this step
            score_step = _score_range(
                df_kde, df_rl,
                walk_state_node.rl_index, i,
                _WALK_STATE_COL[walk_state_node.walk_state]
            )

            # Apply bonus for similar-length inverted repeats
            if next_walk_state == _WS_FLANK_R and repeat_match_prop > 0.0:
                rep_len_min, rep_len_max = sorted([
                    walk_state_node.l_rep_len, _range_len(df_rl, walk_state_node.rl_index, i)
                ])

                score_step += (
                    np.min([walk_state_node.l_rep_score, score_step]) * (
                        (rep_len_min / rep_len_max) if rep_len_max > 0 else 0
                    ) * repeat_match_prop
                )

            # Create next walk state node
            next_walk_node = _InvWalkState(
                next_walk_state, i,
                walk_state_node.walk_score + score_step,
                walk_state_node.l_rep_len,
                walk_state_node.l_rep_score,
                walk_state_node.trace_list + [(i, next_walk_state)]
            )

            # Save left inverted repeat
            if next_walk_state == _WS_INV and walk_state_node.walk_state == _WS_REP_L:
                next_walk_node.l_rep_len = _range_len(df_rl, walk_state_node.rl_index, i)
                next_walk_node.l_rep_score = score_step

            # Save/report state
            if next_walk_state == _WS_FLANK_R:
                # A final state

                # Add score for right flank
                next_walk_node.walk_score += _score_range(
                    df_kde, df_rl, i, df_rl.height, _WALK_STATE_COL[next_walk_state]
                )

                # Update max state
                if next_walk_node.walk_score > max_score:
                    max_score = next_walk_node.walk_score
                    max_score_node = next_walk_node

                # Append state list
                if final_state_node_list is not None:
                    final_state_node_list.append(next_walk_node)

            else:
                # Not a final state, push to node stack
                walk_state_node_stack.append(next_walk_node)

    # Sort final states
    if final_state_node_list is not None:
        final_state_node_list.sort(key=lambda sl: sl.walk_score, reverse=True)

    # Report max state (or None if no inversion found)
    return max_score_node


def _walk_to_regions(
        inv_walk: _InvWalkState,
        df_rl: pl.DataFrame,
        region_qry: Region,
        k_size: int = 0
) -> tuple[Region, Region]:
    """Private function - Translate a walk to inner and outer regions.

    :param inv_walk: Inversion state walk.
    :param df_rl: RL-encoded KDE table.
    :param region_qry: Query region.
    :param k_size: K-mer size.

    :returns: A tuple of inner and outer regions.
    """
    # Validate walk
    if len(set([walk_state for rl_index, walk_state in inv_walk.trace_list])) != len(inv_walk.trace_list):
        raise ValueError(f'Walk node trace has repeated states: {inv_walk}')

    for i in range(1, len(inv_walk.trace_list)):
        if inv_walk.trace_list[i][1] <= inv_walk.trace_list[i - 1][1]:
            raise ValueError(f'Walk node trace states are not monotonically increasing: {inv_walk}')

    for i in range(1, len(inv_walk.trace_list)):
        if inv_walk.trace_list[i][0] <= inv_walk.trace_list[i - 1][0]:
            raise ValueError(f'Walk node trace indexes are not monotonically increasing: {inv_walk}')

    for i in range(len(inv_walk.trace_list)):
        if inv_walk.trace_list[i][1] not in {_WS_FLANK_L, _WS_FLANK_R, _WS_INV, _WS_REP_L, _WS_REP_R}:
            raise ValueError(f'Walk node trace has invalid state: {inv_walk}')

    # Get regions
    state_dict = {
        walk_state: rl_index for rl_index, walk_state in inv_walk.trace_list
    }

    if _WS_INV not in state_dict:
        raise ValueError(f'Walk node trace does not contain an inverted state: {inv_walk}')

    if _WS_FLANK_R not in state_dict or _WS_FLANK_L not in state_dict:
        raise ValueError(f'Walk node trace does not contain flanking states: {inv_walk}')

    i_inner_l = state_dict[_WS_INV]
    i_inner_r = state_dict.get(_WS_REP_R, state_dict[_WS_FLANK_R])

    i_outer_l = state_dict.get(_WS_REP_L, i_inner_l)
    i_outer_r = state_dict[_WS_FLANK_R]

    region_qry_outer = Region(
        region_qry.chrom,
        df_rl[i_outer_l, 'pos_qry'] + region_qry.pos,
        df_rl[i_outer_r, 'pos_qry'] + region_qry.pos + k_size,
        is_rev=region_qry.is_rev
    )

    region_qry_inner = Region(
        region_qry.chrom,
        df_rl[i_inner_l, 'pos_qry'] + region_qry.pos,
        df_rl[i_inner_r, 'pos_qry'] + region_qry.pos + k_size,
        is_rev=region_qry.is_rev
    )

    return region_qry_inner, region_qry_outer


def _expand_region(
        region_ref: Region,
        df_rl: Optional[pl.DataFrame],
        df_ref_fai: pl.DataFrame
) -> Region:
    """Private function - Expands region.

    :param region_ref: Reference region to expand.
    :param df_rl: Run-length encoded table from the inversion search over `region_ref`.
    :param df_ref_fai: Reference lengths.

    :returns: Expanded region.
    """
    expand_bp = int(len(region_ref) * const.INV_EXPAND_FACTOR)

    if df_rl is not None and df_rl.height > 2:
        # More than one state. Expand disproportionately if reference was found up or downstream.

        if df_rl[0, 'state'] == 0:
            return region_ref.expand(
                expand_bp, min_pos=0, max_end=df_ref_fai, shift=True, balance=0.2
            )  # Ref upstream: +20% upstream, +80% downstream

        if df_rl[-1, 'state'] == 0:
            return region_ref.expand(
                expand_bp, min_pos=0, max_end=df_ref_fai, shift=True, balance=0.8
            )  # Ref downstream: +80% upstream, +20% downstream

    return region_ref.expand(
        expand_bp, min_pos=0, max_end=df_ref_fai, shift=True, balance=0.5
    )  # +50% upstream, +50% downstream
