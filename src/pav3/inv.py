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

from typing import Any, Optional, Iterable

import agglovar
import numpy as np
import polars as pl
import scipy.stats

from . import const
from .align import lift
from .params import PavParams
from .region import Region
from .seq import ref_kmers, region_seq_fasta
from .kde import Kde, KdeTruncNorm
from .io import NullWriter

# KDE states
KDE_STATE_FWD = 0
"""KDE state for forward-oriented matches between a query region and reference k-mers."""

KDE_STATE_FWDREV = 1
"""KDE state for forward- and reverse-oriented matches between a query region and reference k-mers (i.e. query """
"""k-mer matches a forward and reverse reference k-mer)."""

KDE_STATE_REV = 2
"""KDE state for reverse-oriented matches between a query region and reference k-mers."""


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

    # if region_flag.pos_align_index != region_flag.end_align_index:
    #     raise ValueError(
    #         f'region_flag must have the same pos_align_index ({region_flag.pos_align_index}) and end_align_index ({region_flag.end_align_index})'
    #     )

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

    df_inv = None
    df_kde = None
    region_qry = None

    while df_inv is None and inv_iterations <= min_expand and inv_iterations <= max_expand:

        # Expand from last search
        if inv_iterations > 0:
            last_len = len(region_ref)
            region_ref = _expand_region(region_ref, None, df_ref_fai)

            if len(region_ref) == last_len:
                # Stop if expansion had no effect
                log_file.write(f'Intra-align inversion region {region_flag}: Reach maximum expansion: {region_ref}\n')
                log_file.flush()
                break

        inv_iterations += 1

        # Max region size
        if region_limit is not None and 0 < region_limit < len(region_ref):
            log_file.write(f'Intra-align inversion region {region_flag}: Reach region size limit ({region_limit}): {region_ref}\n')
            break

        # Get query region
        region_qry = align_lift.region_to_qry(region_ref, same_index=True)

        # Check for expansion through a break in the query sequence
        if region_qry is None:
            if stop_on_lift_fail:
                log_file.write(f'Intra-align inversion region {region_flag}: Lift failed: {region_ref}\n')
                break
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
            log_file.write(f'Intra-align inversion region {region_flag}: No matching k-mers (likely divergent region): {region_ref}\n')
            break

        if df_kde.shape[0] < min_kmers:
            # Not enough k-mers, expand to find more
            continue

        # Test states for inversion
        df_kde_index = (
            df_kde
            .group_by('state')
            .agg(
                pl.col('index').min().alias('index_min'),
                pl.col('index').max().alias('index_max'),
            )
        )

        if not (
            (0 in df_kde_index['state'])
            and (2 in df_kde_index['state'])
        ):
            continue

        inv_fwd_min, inv_fwd_max = (
            df_kde_index
            .filter(pl.col('state') == 0)
            .select('index_min', 'index_max')
            .row(0)
        )

        inv_rev_min, inv_rev_max = (
            df_kde_index
            .filter(pl.col('state') == 2)
            .select('index_min', 'index_max')
            .row(0)
        )

        if inv_fwd_min > inv_rev_min or inv_fwd_max < inv_rev_max:
            continue

        df_inv = kde_to_inv(df_kde)

        # # Get run-length encoded states (list of (state, count) tuples).
        # df_rl = rl_encoder(df_kde)
        #
        # # Done if reference oriented k-mers (state == 0) found on both sides
        # if df_rl.height > 2 and df_rl[0, 'state'] == 0 and df_rl[-1, 'state'] == 0:
        #     break  # Stop searching and expanding

    # Stop if no inverted sequence was found
    #if df_rl is None or not df_rl.select((pl.col('state') == 2).any()).item():
    if df_inv is None:
        log_file.write(f'Intra-align inversion region {region_flag}: No inverted states found: {region_ref}\n')
        log_file.flush()

        return None

    region_qry_inner, region_qry_outer = inv_table_to_regions(df_inv, region_qry)

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
        [region_flag.pos_align_index]
    )


def get_inv_row(
        region_ref_inner: Optional[Region] = None,
        region_ref_outer: Optional[Region] = None,
        region_qry_inner: Optional[Region] = None,
        region_qry_outer: Optional[Region] = None,
        align_source: Optional[Iterable[int] | int] = None,
) -> dict[str, Any]:
    """Format an inversion into a dict.

    Separated from try_inv_region() so that place-holder inv dicts can be generated for inversion tables with no
    records (can be used to get a list of fields that would be generated).

    :param region_ref_inner: Inner reference region.
    :param region_ref_outer: Outer reference region.
    :param region_qry_inner: Inner query region.
    :param region_qry_outer: Outer query region.
    :param align_source: Index of the source alignment record.
    """
    none_vals = [region is None for region in [
        region_ref_inner, region_ref_outer, region_qry_inner, region_qry_outer, align_source
    ]]

    if all(none_vals):
        region_ref_inner = Region('NA', 0, 0)
        region_ref_outer = Region('NA', 0, 0)
        region_qry_inner = Region('NA', 0, 0)
        region_qry_outer = Region('NA', 0, 0)
        align_source = [-1]
    elif any(none_vals):
        raise ValueError('All arguments may be None, or all arguments must be non-None')

    try:
        align_source = list(align_source)
    except TypeError:
        align_source = [align_source]

    return {
        'chrom': region_ref_inner.chrom,
        'pos': region_ref_inner.pos,
        'end': region_ref_inner.end,
        'vartype': 'INV',
        'varlen': len(region_ref_inner),
        'align_source': align_source,
        'qry_id': region_qry_inner.chrom,
        'qry_pos': region_qry_inner.pos,
        'qry_end': region_qry_inner.end,
        'qry_rev': region_ref_inner.is_rev,
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
            pl.when(pl.col('kmer_fwd') & pl.col('kmer_rev'))
            .then(1)
            .when(pl.col('kmer_fwd')).then(0)
            .when(pl.col('kmer_rev')).then(2)
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


def  kde_to_inv(
        df_kde: pl.DataFrame,
) -> pl.DataFrame:
    """Create a table describing an inversion using a hidden Markov model (HMM) to set inversion
    breakpoints from a KDE table.

    In a KDE table, the smoothed state values do not always correspond to the true state. Setting
    breakpoints from a KDE table is challenging and arbitrary resulting in suboptimal inversion
    structures. This function uses an HMM across the smoothed states to set breakpoints.

    HMM states are:
        0. Left unique region (fwd)
        1. Left inverted repeat (fwd+rev)
        2. Inversion (rev)
        3. Right inverted repeat (fwd+rev)
        4. Right unique region (fwd)
        5. Inversion without inverted repeat (rev)

    Allowed state transitions are "0-1-2-3-4" (with inverted repeats) or "0-5-4" (without inverted
    repeats).

    Each row in the inversion table returned is a range in the query sequence belonging to one
    inversion state.

    The table returned has these columns:
        0. inv_state: DMM state.
        #. inv_state_label: Label ("fwd", "fwdrev", and "rev").
        #. index_start: First value of column "index" in the KDE table.
        #. index_end: Last value of column "index" in the KDE table.
        #. len: Length of the region.
        #. states_mer: A struct of counts across original k-mer states (fwd, fwdrev, and rev).
        #. states_kde: A struct of counts across smoothed states (fwd, fwdrev, and rev).

    :param df_kde: KDE table.

    :returns: Inversion table.
    """
    import numpy as np
    import hmmlearn.hmm as hmm

    # TODO: Train a real model and load weights. Hardcoded weights for now, tested on messy INVs.
    a_trans = np.array(
        [
            # 0   1   2   3   4   5
            [ 0.95, 0.04, 0.00, 0.00, 0.00, 0.01, ],  # 0: Left Ref
            [ 0.00, 0.95, 0.05, 0.00, 0.00, 0.00, ],  # 1: Left Invdup
            [ 0.00, 0.00, 0.95, 0.05, 0.00, 0.00, ],  # 2: Inv
            [ 0.00, 0.00, 0.00, 0.95, 0.05, 0.00, ],  # 3: Right Invdup
            [ 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, ],  # 4: Right Ref
            [ 0.00, 0.00, 0.00, 0.00, 0.05, 0.95, ],  # 5: Inv - no invdup
        ]
    )

    a_start = np.array([1.00, 0.00, 0.00, 0.00, 0.00, 0.00])

    a_emission = np.array(
        [
            [ 0.90, 0.05, 0.05 ],  # 0: Left Ref
            [ 0.05, 0.90, 0.05 ],  # 1: Left Invdup
            [ 0.05, 0.05, 0.90 ],  # 2: Inv
            [ 0.05, 0.90, 0.05 ],  # 3: Right Invdup
            [ 0.90, 0.05, 0.05 ],  # 4: Right Ref
            [ 0.05, 0.05, 0.90 ],  # 5: Inv - no invdup
        ]
    )

    model = hmm.CategoricalHMM(
        n_components=a_trans.shape[0],
    )

    model.startprob_ = a_start
    model.transmat_ = a_trans
    model.emissionprob_ = a_emission

    hidden_states = model.predict(np.array(df_kde['state']).reshape(1, -1))

    df_kde = (
        df_kde
        .drop('inv_state', strict=False)
        .with_columns(pl.Series(hidden_states).cast(pl.Int32).alias('inv_state'))
    )

    df_inv = (
        df_kde
        .with_columns(
            pl.col('inv_state').rle_id().alias('inv_state_rle'),
        )
        .group_by('inv_state_rle')
        .agg(
            pl.col('index').first().alias('index_start'),
            pl.col('index').last().alias('index_end'),
            (pl.col('index').last() - pl.col('index').first()).alias('len'),
            pl.col('inv_state').first().alias('inv_state'),
            pl.struct(
                (pl.col('state_mer') == 0).sum().alias('fwd'),
                (pl.col('state_mer') == 1).sum().alias('fwdrev'),
                (pl.col('state_mer') == 2).sum().alias('rev'),
            ).alias('states_mer'),
            pl.struct(
                (pl.col('state') == 0).sum().alias('fwd'),
                (pl.col('state') == 1).sum().alias('fwdrev'),
                (pl.col('state') == 2).sum().alias('rev'),
            ).alias('states_kde'),
        )
        .filter(
            ~pl.col('inv_state').is_in([0, 4]),
            pl.col('len') > 0,
        )
        .select(
            'inv_state',
            pl.col('inv_state').replace_strict({1: 'fwdrev', 2: 'rev', 3: 'fwdrev', 5: 'rev'}).alias('inv_state_label'),
            'index_start', 'index_end', 'len', 'states_mer', 'states_kde',
        )
    )

    return df_inv


def inv_table_to_regions(
        df_inv: pl.DataFrame,
        region_qry: Region,
) -> tuple[Region, Region]:
    """Convert an inversion table to a tuple of regions (inner, outer).

    :param df_inv: Inversion table.
    :param region_qry: Query region.

    :returns: Tuple of regions (inner, outer).

    :raises ValueError: If the inversion table does not progress through inversion states as
        expected. Suggests an error in `kde_to_inv()`.
    """

    df_inv = (
        df_inv
        .filter(
            pl.col('inv_state').is_in([1, 2, 3, 5])
        )
    )

    if tuple(df_inv['inv_state']) == (1, 2, 3):
        coord_outer = (
            df_inv['index_start'][0] + region_qry.pos,
            df_inv['index_end'][2] + region_qry.pos + 1
        )

        coord_inner = (
          df_inv['index_start'][1] + region_qry.pos,
          df_inv['index_end'][1] + region_qry.pos + 1
        )

    elif tuple(df_inv['inv_state']) == (5,):
        coord_outer = (
            df_inv['index_start'][0] + region_qry.pos,
            df_inv['index_end'][0] + region_qry.pos + 1
        )

        coord_inner = coord_outer

    else:
        raise ValueError(f'Unexpected inv_state progression: {" > ".join(df_inv["inv_state"].cast(pl.String))}')

    return (
        Region(
            region_qry.chrom,
            coord_inner[0],
            coord_outer[1],
            region_qry.is_rev,
        ),
        Region(
            region_qry.chrom,
            coord_outer[0],
            coord_outer[1],
            region_qry.is_rev,
        ),
    )


def _expand_region(
        region_ref: Region,
        df_rl: Optional[pl.DataFrame],
        df_ref_fai: pl.DataFrame
) -> Region:
    """Expands region.

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
