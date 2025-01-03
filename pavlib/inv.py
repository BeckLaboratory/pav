"""
Routines for calling inversions.
"""

import numpy as np
import pandas as pd

import kanapy
import pavlib
import scipy.stats
import svpoplib


#
# Constants
#

# Walk states
WS_FLANK_L = 0  # Left flank (unique sequence)
WS_REP_L = 1    # Left inverted repeat
WS_INV = 2      # Inverted core
WS_REP_R = 3    # Right inverted repeat
WS_FLANK_R = 4  # Right flank (unique sequence)

# String representation for each walk state
WS_STATE_REPR = {
    WS_FLANK_L: 'FL',
    WS_REP_L: 'RL',
    WS_INV: 'V',
    WS_REP_R: 'RR',
    WS_FLANK_R: 'FR'
}

# KDE states
KDE_STATE_FWD = 0
KDE_STATE_FWDREV = 1
KDE_STATE_REV = 2

# Walk state to KDE table column (for scoring over the appropriate KDE state for a walk state)
WALK_STATE_COL = {
    WS_FLANK_L: 'KDE_FWD',
    WS_REP_L: 'KDE_FWDREV',
    WS_INV: 'KDE_REV',
    WS_REP_R: 'KDE_FWDREV',
    WS_FLANK_R: 'KDE_FWD'
}


class InvCall:
    """
    Describes an inversion call with data supporting the call.

    :param region_ref: Reference region of the inversion (inner-breakpoints).
    :param region_qry: Query region of the inversion (inner-breakpoints) corresponding to `region_ref`.
    :param is_rev: `True` query sequence is on the reverse strand.
    :param svlen: SV length.
    :param id: Variant ID.
    :param score: Variant score.
    :param region_ref_outer: Outer-flanks of inversion including inverted repeats, if present, that likely drove the
        inversion. The actual inversion breakpoints are likely between the outer and inner coordinates. This is the
        coordinate most inversion callers use.
    :param region_qry_outer: Coordinates on the aligned query of outer breakpoints corresponding to `region_ref_outer`.
    """

    def __init__(
            self,
            region_ref, region_qry, is_rev=None,
            svlen=None, id=None,
            score=np.nan,
            region_ref_outer=None, region_qry_outer=None
    ):

        # Save INV regions and dataframe
        self.region_ref = region_ref
        self.region_qry = region_qry

        if is_rev is None:
            is_rev = region_qry.is_rev

        self.is_rev = is_rev

        if svlen is None:
            svlen = len(self.region_ref)

        self.svlen = svlen

        if id is None:
            id = '{}-{}-INV-{}'.format(region_ref.chrom, region_ref.pos + 1, self.svlen)

        self.id = id

        self.score = score

        self.region_ref_outer = region_ref_outer
        self.region_qry_outer = region_qry_outer

    def __repr__(self):
        return self.id


def get_inv_from_record(row):
    """
    Get an inversion object from a a tabular inversion call record (i.e. a row from an INV BED).

    :param row: Inversion record (Pandas Series).

    :return: An Inversion object.
    """

    missing_col = [col for col in ('#CHROM', 'POS', 'END', 'QRY_STRAND') if col not in row]

    if missing_col:
        raise RuntimeError(f'Missing columns in row: {", ".join(missing_col)}')

    region_ref = pavlib.seq.Region(row['#CHROM'], row['POS'], row['END'])
    region_qry = pavlib.seq.region_from_string(row['QRY_REGION'], is_rev=row['QRY_STRAND'] == '-')

    return InvCall(
        region_ref=region_ref,
        region_qry=region_qry,

        svlen=row['SVLEN'] if 'SVLEN' in row else None,
        id=row['ID'] if 'ID' in row else None,
        score=row['SCORE'] if 'SCORE' in row else np.nan,

        region_ref_outer=pavlib.seq.region_from_string(row['REGION_REF_OUTER']) if 'REGION_REF_OUTER' in row else None,
        region_qry_outer=pavlib.seq.region_from_string(row['REGION_QRY_OUTER'], is_rev=region_qry.is_rev) if 'REGION_QRY_OUTER' in row else None
    )


def scan_for_inv(
        region_flag,
        ref_fa_name,
        qry_fa_name,
        align_lift,
        k_util=None,
        nc_ref=None,
        nc_qry=None,
        index_set=None,
        match_index=True,
        stop_on_lift_fail=True,
        kde=None,
        log=None,
        region_limit=pavlib.const.INV_REGION_LIMIT,
        min_expand=pavlib.const.INV_MIN_EXPAND_COUNT,
        max_expand=np.inf,
        init_expand=pavlib.const.INV_INIT_EXPAND,
        min_kmers=pavlib.const.INV_MIN_KMERS,
        max_ref_kmer_count=pavlib.const.INV_MAX_REF_KMER_COUNT,
        repeat_match_prop=pavlib.const.INV_REPEAT_MATCH_PROP,
        min_inv_kmer_run=pavlib.const.INV_MIN_INV_KMER_RUN,
        min_qry_ref_prop=pavlib.const.INV_MIN_QRY_REF_PROP
    ):
    """
    Scan region for inversions. Start with a flagged region (`region_flag`) where variants indicated that an inversion
    might be. Scan that region for an inversion expanding as necessary.

    :param region_flag: Flagged region to begin scanning for an inversion.
    :param ref_fa_name: Reference FASTA. Must also have a .fai file.
    :param qry_fa_name: Query FASTA. Must also have a .fai file.
    :param align_lift: Alignment lift-over tool (pavlib.align.AlignLift).
    :param k_util: K-mer utility. If `None`, a k-mer utility is created with PAV's default k-mer size
        (`pavlib.const.K_SIZE').
    :param nc_ref: Locations of no-call regions in the reference to ignore. These may be known problematic loci or
        N-bases in the reference. If a region is expanded into these locations with any overlap (1+ bp), then searching
        for inversions is stopped. If defined, this should be dict-like object keyed by chromosome names where each
        value is an IntervalTree of no-call bases.
    :param nc_qry: Locations of no-call regions in the query sequences. Purpose and format is analagous to `nc_ref`,
        except query coordinates are referenced instead of reference coordinates.
    :param index_set: If not `None`, restrict search to this set of aligned indexes.
    :param match_index: If `True`, then both left and right regions must land on the same alignment record. Set to
        `True` to handle inversions expected to be covered by one contiguous alignment and `False` if the inversion
        might have broken alignments into several aligned fragments.
    :param stop_on_lift_fail: If `True`, stop if the reference-to-query lift fails for any reason (i.e. breakpoints
        missing alignment records). If `False`, keep expanding until expansion is exhausted (reaches maximum number of
        expansions or covers a whole reference region).
    :param kde: Kernel density estimator function. Expected to be a `pavlib.kde.KdeTruncNorm` object, but can
        be any object with a similar signature. If `None`, a default `kde` estimator is used.
    :param log: Log file (open file handle) or `None` for no log.
    :param region_limit: Max region size. If inversion (+ flank) exceeds this size, stop searching for inversions.
        If `None`, set to default, `MAX_REGION_SIZE`. If 0, ignore max and scale to arbitrarily large inversions.
    :param min_expand: The number of region expansions to try (including the initial expansion) and finding only
        forward-oriented k-mer states after smoothing before giving up on the region. `None` sets the default value,
        `pavlib.const.INV_MIN_EXPAND_COUNT`.
    :param max_expand: Maximum number of expansions before giving up. If `None`, no expansion limit. Set to 0 to explore
        only the query region with no expansions.
    :param init_expand: Initial expand bp. The flagged region is first expanded by this value before searching for
        an inversion. This ensures the region reaches into flanks, which is needed for the inversion method.
    :param min_kmers: Minimum number of informative k-mers in the inversion region. Uninformative k-mers are removed
        first (i.e. not in reference region FWD or REV). If the number of k-mers left over is smaller than this value,
        then the search is stopped.
    :param max_ref_kmer_count: K-mers present in the reference region more than this many times are discarded. This
        is the canonical k-mer count, both orientations (fwd and rev) of each k-mer is added together and tested
        against this limit. Any canonical k-mers exceeding the limit are discarded including their reverse-complement.
    :param repeat_match_prop: When considering alternative inversion structures, give a bonus to structures with
        similar length inverted repeats (if both are present) proportional to the lowest score of both scaled by
        size similarity (min / max) and multiplied by this value. If 0.0, no bonus for similar length inverted repeats
        is applied.
    :param min_inv_kmer_run: Minimum number of k-mers in the inverted core region to consider as an inversion.
    :param min_qry_ref_prop: The query and reference region sizes must be within this factor (reciprocal)

    :return: A `InvCall` object describing the inversion found or `None` if no inversion was found.
    """

    if k_util is None:
        k_util = kanapy.util.kmer.KmerUtil(pavlib.const.INV_K_SIZE)

    if kde is None:
        kde = pavlib.kde.KdeTruncNorm()

    # Init
    _write_log(
        'Scanning for inversions in flagged region: {} (flagged region record id = {})'.format(
            region_flag, region_flag.region_id()
        ),
        log
    )

    ref_fai = svpoplib.ref.get_df_fai(ref_fa_name + '.fai')
    qry_fai = svpoplib.ref.get_df_fai(qry_fa_name + '.fai')

    region_ref = region_flag.copy()
    region_ref.expand(init_expand, min_pos=0, max_end=ref_fai, shift=True)

    inv_iterations = 0

    # Get no-call tree for this reference location
    if nc_ref is not None and region_ref.chrom in nc_ref.keys():
        nc_tree_ref = nc_ref[region_ref.chrom]
    else:
        nc_tree_ref = None

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
            expand_region(region_ref, df_rl, ref_fai)

            if len(region_ref) == last_len:
                # Stop if expansion had no effect
                _write_log('Reached reference limits, cannot expand', log)
                return None

        inv_iterations += 1

        df_kde = None
        df_rl = None

        # Max region size
        if region_limit is not None and 0 < region_limit < len(region_ref):
            _write_log(f'Region size exceeds max: {region_ref} ({len(region_ref):,d} > {region_limit:,d})', log)
            return None

        # Stop over non-callable reference regions
        if nc_tree_ref is not None and len(nc_tree_ref[region_ref.pos:region_ref.end]) > 0:
            _write_log(f'Region overlaps no-call bases (reference): {region_ref}', log)
            return None

        # Get query region
        region_qry = lift_ref_to_qry(region_ref, align_lift, index_set=index_set, match_index=match_index)

        # Check for expansion through a break in the query sequence
        if region_qry is None:

            if stop_on_lift_fail:
                _write_log(f'Could not lift reference region onto query: {region_ref}', log)
                return None

            _write_log(f'Lift failed: {region_ref}', log)

            continue

        # Stop over non-callable query regions
        if nc_qry is not None and region_qry.chrom in nc_qry.keys():
            nc_tree_qry = nc_qry[region_qry.chrom]

            if len(nc_tree_qry[region_qry.pos:region_ref.end]) > 0:
                _write_log(f'Region overlaps no-call bases (query): {region_qry}', log)
                return None

        # Log search
        _write_log(f'Scanning region: {region_ref}', log)

        # Create k-mer table
        df_kde = get_state_table(
            region_ref=region_ref, region_qry=region_qry,
            ref_fa_name=ref_fa_name, qry_fa_name=qry_fa_name,
            ref_fai=ref_fai, qry_fai=qry_fai,
            is_rev=region_qry.is_rev,
            k_util=k_util, kde=kde,
            max_ref_kmer_count=max_ref_kmer_count,
            expand_bound=True,
            log=log
        )

        if df_kde is None:
            _write_log(
                'No informative reference k-mers in forward or reverse orientation in region',
                log
            )

            return

        if df_kde.shape[0] < min_kmers:
            _write_log(f'Not enough k-mers in region to call: {df_kde.shape[0]} < {min_kmers}', log)
            continue

        # Get run-length encoded states (list of (state, count) tuples).
        df_rl = pavlib.kde.rl_encoder(df_kde)

        # Done if reference oriented k-mers (state == 0) found on both sides
        if df_rl.shape[0] > 2 and df_rl.iloc[0]['STATE'] == 0 and df_rl.iloc[-1]['STATE'] == 0:
            break  # Stop searching and expanding

    # Stop if no inverted sequence was found
    if df_rl is None or not np.any(df_rl['STATE'] == 2):
        _write_log(f'No inverted states found: {region_ref}', log)
        return None

    if df_rl.iloc[0]['STATE'] != 0 or df_rl.iloc[-1]['STATE'] != 0:
        raise RuntimeError(f'Found INV region not flanked by reference sequence (program bug): {region_ref}')

    # Estimate inversion breakpoints
    inv_walk = resolve_inv_from_rl(
        df_rl, df_kde,
        repeat_match_prop=repeat_match_prop,
        min_inv_kmer_run=min_inv_kmer_run
    )

    if inv_walk is None:
        _write_log(f'Failed to resolve inversion breakpoints: {region_ref}: No walk along KDE states was found', log)
        return None

    region_qry_inner, region_qry_outer = walk_to_regions(inv_walk, df_rl, region_qry, k_size=k_util.k_size)

    # Lift to reference
    region_ref_inner = align_lift.lift_region_to_ref(region_qry_inner, gap=True, check_orientation=False)

    if region_ref_inner is None:
        _write_log(f'Failed lifting inner INV region to reference: {region_qry_inner}', log)
        return None

    region_ref_outer = align_lift.lift_region_to_ref(region_qry_outer, gap=True, check_orientation=False)

    if region_ref_outer is None or not region_ref_outer.contains(region_ref_inner):
        region_ref_outer = region_ref_inner
        _write_log(f'Failed lifting outer INV region to reference: {region_qry_outer}: Using inner coordinates', log)

    # Check size proportions
    if len(region_ref_inner) < len(region_qry_inner) * min_qry_ref_prop:
        _write_log(
            'Reference region too short: Reference region length ({:,d}) is not within {:.2f}% of the query region length ({:,d})'.format(
                len(region_ref_inner),
                min_qry_ref_prop * 100,
                len(region_qry_inner)
            ),
            log
        )

        return None

    if len(region_qry_outer) < len(region_ref_outer) * min_qry_ref_prop:
        _write_log(
            'Query region too short: Query region length ({:,d}) is not within {:.2f}% of the reference region length ({:,d})'.format(
                len(region_qry_outer),
                min_qry_ref_prop * 100,
                len(region_ref_outer)
            ),
            log
        )

        return None

    # Return inversion call
    _write_log(
        f'INV Found: inner={region_ref_inner}, outer={region_ref_outer} (qry inner={region_qry_inner}, outer={region_qry_outer})',
        log
    )

    inv_call = InvCall(
        region_ref=region_ref_inner,
        region_qry=region_qry_inner,
        region_ref_outer=region_ref_outer,
        region_qry_outer=region_qry_outer
    )

    _write_log('Found inversion: {}'.format(inv_call), log)

    return inv_call


def get_state_table(
        region_ref, region_qry,
        ref_fa_name, qry_fa_name,
        ref_fai, qry_fai,
        is_rev=None,
        k_util=None,
        kde=None,
        max_ref_kmer_count=pavlib.const.INV_MAX_REF_KMER_COUNT,
        expand_bound=True,
        force_norm=False,
        log=None
):
    """
    Initialize the state table for inversion calling by k-mers.

    :param region_ref: Reference region.
    :param region_qry: Query region.
    :param ref_fa_name: Reference FASTA file name. Reference k-mers are extracted from this file.
    :param ref_fa_name: Reference FASTA file name.
    :param qry_fa_name: Query FASTA file name.
    :param is_rev: Set to `True` if the query is reverse-complemented relative to the reference. Reference k-mers are
        reverse-complemented to match the query sequence. If `None`, get from `region_qry`
    :param k_util: K-mer utility.
    :param kde: KDE object for kernel density estimates. If `None`, KDE is not applied.
    :param max_ref_kmer_count: Remove high-count kmers greater than this value.
    :param force_norm: Normalize across states so that the sum of KDE_FWD, KDE_FWDREV, and KDE_REV is always 1.0. This
        is not needed for most KDE models.
    :param log: Open log file.

    :return: Initialized table.
    """

    if k_util is None:
        k_util = kanapy.util.kmer.KmerUtil(pavlib.const.INV_K_SIZE)

    if is_rev is None:
        is_rev = region_qry.is_rev

    # Expand regions by band_bound
    if expand_bound and kde is not None:

        if ref_fai is None or qry_fai is None:
            raise RuntimeError('Expanding regions by KDE band-bounds requires both reference and query FAIs to be defined')

        region_ref_exp = region_ref.copy().expand(kde.band_bound * 2, max_end=ref_fai, shift=False)
        region_qry_exp = region_qry.copy().expand(kde.band_bound * 2, max_end=qry_fai, shift=False)
    else:
        region_ref_exp = region_ref
        region_qry_exp = region_qry

    # Get reference k-mer counts
    ref_kmer_count = pavlib.seq.ref_kmers(region_ref_exp, ref_fa_name, k_util)

    if ref_kmer_count is None or len(ref_kmer_count) == 0:
        _write_log(f'No reference k-mers for region {region_ref_exp}', log)
        return None

    # Skip low-complexity sites with repetitive k-mers
    if max_ref_kmer_count is not None and max_ref_kmer_count > 0:
        canon_count_set = set()
        rm_set = set()

        for kmer, count in ref_kmer_count.items():
            c_kmer = k_util.canonical_complement(kmer)

            if c_kmer in canon_count_set:
                continue

            if c_kmer == kmer:  # re-define kmer to be the reverse-complement of the canonical k-mer
                kmer = k_util.rev_complement(c_kmer)

            if ref_kmer_count.get(c_kmer, 0) + ref_kmer_count.get(kmer, 0) > max_ref_kmer_count:

                if c_kmer in ref_kmer_count:
                    rm_set.add(c_kmer)

                if kmer in ref_kmer_count:
                    rm_set.add(kmer)

            canon_count_set.add(c_kmer)

        if len(rm_set) > 0:
            for kmer in rm_set:
                del(ref_kmer_count[kmer])

            _write_log(f'Removed {len(rm_set)} high-count k-mers (> {max_ref_kmer_count:,}) for region {region_ref_exp}', log)

    ref_kmer_set = set(ref_kmer_count)

    if is_rev:
        ref_kmer_set = {k_util.rev_complement(kmer) for kmer in ref_kmer_set}

    ## Get contig k-mers as list ##

    seq_qry = pavlib.seq.region_seq_fasta(region_qry_exp, qry_fa_name, rev_compl=False)

    df = pd.DataFrame(
        list(kanapy.util.kmer.stream(seq_qry, k_util, index=True)),
        columns=['KMER', 'INDEX']
    )

    max_index = df.iloc[-1]['INDEX']

    # Assign state
    df['STATE_MER'] = df['KMER'].apply(lambda kmer:
       pavlib.kde.KMER_ORIENTATION_STATE[
           int(kmer in ref_kmer_set),
           int(k_util.rev_complement(kmer) in ref_kmer_set)
       ]
    )

    # Subset to informative sites
    df = df.loc[df['STATE_MER'] != -1]

    # Set index in condensed space (without uninformative k-mers)
    df.reset_index(inplace=True, drop=True)

    # Apply KDE
    if kde is not None:
        df['KDE_FWD'] = kde(df.loc[df['STATE_MER'] == 0].index, df.shape[0])
        df['KDE_FWDREV'] = kde(df.loc[df['STATE_MER'] == 1].index, df.shape[0])
        df['KDE_REV'] = kde(df.loc[df['STATE_MER'] == 2].index, df.shape[0])

        if force_norm:
            kde_sum = df[['KDE_FWD', 'KDE_FWDREV', 'KDE_REV']].apply(np.sum, axis=1)

            df['KDE_FWD'] /= kde_sum
            df['KDE_FWDREV'] /= kde_sum
            df['KDE_REV'] /= kde_sum

        df['STATE'] = np.argmax(df[['KDE_FWD', 'KDE_FWDREV', 'KDE_REV']], axis=1)

    # Collapse ends expanded by kde.band_bound
    diff_l = region_qry.pos - region_qry_exp.pos
    diff_r = region_qry_exp.end - region_qry.end

    if diff_l > 0 or diff_r > 0:
        first_index = diff_l + 1
        last_index = max_index - diff_l + 1
        df = df.loc[
            df['INDEX'].apply(lambda val: first_index < val < last_index)
        ].copy()

        df['INDEX'] -= diff_l

    df.reset_index(inplace=True, drop=True)

    # Return dataframe
    return df


#
# INV structure tracing through run-length encoded KDE states
#


def score_range(df, df_rl, rl_index, i, kde_col):
    """
    Score a segment of the KDE between two run-length encoded positions.

    :param df: KDE DataFrame.
    :param df_rl: Run-length encoded DataFrame.
    :param rl_index: Index of first run-length encoded record (inclusive).
    :param i: Index of last run-length encoded record (inclusive).
    :param kde_col: KDE column to sum.

    :return: A sum of all KDE records in `df` between run-length encoded records at index `rl_index` and `i`
        (inclusive). The KDE state summed is `kde_col` (name of the column in `df` to sum).
    """

    return np.sum(
        df.loc[
            (
                df_rl.loc[rl_index, 'POS_KDE']
            ):(
                df_rl.loc[i - 1, 'POS_KDE'] + df_rl.loc[i - 1, 'LEN_KDE'] - 1
            ),
            kde_col
        ]
    )


def range_len(df_rl, rl_index, i):
    """
    Get the length (number of KDE records) between two run-length encoded records (inclusive).

    :param df_rl: Run-length encoded DataFrame.
    :param rl_index: Index of first run-length encoded record (inclusive).
    :param i: Index of last run-length encoded record (inclusive).

    :return: Number of KDE records in the range `rl_index` to `i` (inclusive).
    """
    return df_rl.loc[i - 1, 'POS_KDE'] + df_rl.loc[i - 1, 'LEN_KDE'] - df_rl.loc[rl_index, 'POS_KDE']


NEXT_WALK_STATE = {  # State transition given the current walk state and the next KDE state. Only legal transitions are defined
    (WS_FLANK_L, KDE_STATE_FWDREV): WS_REP_L,  # Left flank -> left inverted repeat
    (WS_FLANK_L, KDE_STATE_REV): WS_INV,       # Left flank -> inverted core (no left inverted repeat)
    (WS_REP_L, KDE_STATE_REV): WS_INV,         # Left inverted repeat -> inverted core
    (WS_INV, KDE_STATE_FWDREV): WS_REP_R,      # Inverted core -> right inverted repeat
    (WS_INV, KDE_STATE_FWD): WS_FLANK_R,       # Inverted core -> right flank (no right inverted repeat)
    (WS_REP_R, KDE_STATE_FWD): WS_FLANK_R      # Right inverted repeat -> right flank
}


class InvWalkState(object):
    """
    Tracks the state of a walk from the left-most run-length (RL) encoded record to a current state.

    Attributes:
    * walk_state: Walk state ("WS_" constants)
    * rl_index: Index of the RL-encoded DataFrame where the current state starts.
    * walk_score: Cumulative score of the walk walk from the left-most RL record to the current state and position..
    * l_rep_len: Length of the right inverted repeat locus. Value is 0 if there was none or the state has not
        yet reached the left inverted repeat. Used to give a score boost for records with similar-length inverted
        repeats on both sides.
    * l_rep_score: Score of the right inverted repeat locus. Value is 0.0 if there was none or the state has
        not yet reached the left inverted repeat. Used to give a score boost for records with similar-length inverted
        repeats on both sides.
    * trace_list: List of walk state transitions to the current state. Each element is a tuple of an RL table index and
        a walk state ("WS_" constants).
    """

    def __init__(self, walk_state, rl_index, walk_score=0.0, l_rep_len=0, l_rep_score=0.0, trace_list=None):
        """
        Create a new walk state object.

        All arguments are assigned to attributes.

        :param walk_state: Walk state.
        :param rl_index: RL-index where the state starts.
        :param walk_score: Cumulative ccore of the walk state.
        :param l_rep_len: Length of the right inverted repeat locus.
        :param l_rep_score: Score of the right inverted repeat locus.
        :param trace_list: List of walk state transitions to the current state.
        """

        self.walk_state = walk_state
        self.rl_index = rl_index
        self.walk_score = walk_score
        self.l_rep_len = l_rep_len
        self.l_rep_score = l_rep_score
        self.trace_list = trace_list if trace_list is not None else []

    def __repr__(self):
        trace_str = ', '.join([f'{te[0]}:{WS_STATE_REPR[te[1]]}' for te in self.trace_list])
        return f'InvWalkState(walk_state={self.walk_state}, rl_index={self.rl_index}, walk_score={self.walk_score}, l_rep_len={self.l_rep_len}, l_rep_score={self.l_rep_score} trace_list=[{trace_str}])'


def resolve_inv_from_rl(df_rl, df, repeat_match_prop=0.2, min_inv_kmer_run=pavlib.const.INV_MIN_INV_KMER_RUN, final_state_node_list=None):
    """
    Resolve the optimal walk along run-length (RL) encoded states. This walk sets the inversion breakpoints.

    :param df_rl: RL-encoded table.
    :param df: State table after KDE.
    :param repeat_match_prop: Give a bonus to the a walk score for similar-length inverted repeats. The bonus is
        calculated by taking the minimum score from both left and right inverted repeat loci, multiplying by the
        length similarity (min/max length), and finally multipyling by this value. Set to 0.0 to disable the bonus.
    :param min_inv_kmer_run: Minimum number of k-mers in an inversion run. Set to 0 to disable the filter.
    :param final_state_node_list: If not None, the list is populated with all InvWalkState objects reaching a final
        state and is sorted by walk scores (highest scores first). This list is cleared by the method before
        exploing RL walks. This can be used to see alternative inversion structures.

    :return: A InvWalkState object representing the best walk along RL states.
    """

    if repeat_match_prop is None:
        repeat_match_prop = 0.0

    if min_inv_kmer_run is None:
        min_inv_kmer_run = 0

    # Clear final state nodes
    if final_state_node_list is not None:
        final_state_node_list.clear()

    # Location of the last inverted segment. Used to limit the search space in states before the inversion.
    max_inv_index = np.max(df_rl.loc[df_rl['STATE'] == 2].index) + 1  # Last inversion state in the rl table

    # Initialize first state
    walk_state_node_stack = [
        InvWalkState(0, 0, 0.0, 0, 0.0, [(0, WS_FLANK_L)])
    ]

    max_score = 0.0
    max_score_node = None

    while walk_state_node_stack:
        walk_state_node = walk_state_node_stack.pop()

        # Walk through downstream states
        for i in range(
            walk_state_node.rl_index + 1,
            (df_rl.shape[0] if walk_state_node.walk_state >= WS_INV else max_inv_index)
        ):

            # KDE state and next walk state
            kde_state = df_rl.loc[i, 'STATE']
            next_walk_state = NEXT_WALK_STATE.get((walk_state_node.walk_state, kde_state), None)

            if next_walk_state is None:
                continue  # No transition from the current walk state to this KDE state

            # Stop if inverted region is too short
            if walk_state_node.walk_state == WS_INV and \
                min_inv_kmer_run > 0 and \
                range_len(df_rl, walk_state_node.rl_index, i) < min_inv_kmer_run:

                continue

            # Score this step
            score_step = score_range(df, df_rl, walk_state_node.rl_index, i, WALK_STATE_COL[walk_state_node.walk_state])

            # Apply bonus for similar-length inverted repeats
            if next_walk_state == WS_FLANK_R and repeat_match_prop > 0.0:
                rep_len_min, rep_len_max = sorted([walk_state_node.l_rep_len, range_len(df_rl, walk_state_node.rl_index, i)])

                score_step += (
                    np.min([walk_state_node.l_rep_score, score_step]) * (
                        (rep_len_min / rep_len_max) if rep_len_max > 0 else 0
                    ) * repeat_match_prop
                )

            # Create next walk state node
            next_walk_node = InvWalkState(
                next_walk_state, i,
                walk_state_node.walk_score + score_step,
                walk_state_node.l_rep_len,
                walk_state_node.l_rep_score,
                walk_state_node.trace_list + [(i, next_walk_state)]
            )

            # Save left inverted repeat
            if next_walk_state == WS_INV and walk_state_node.walk_state == WS_REP_L:
                next_walk_node.l_rep_len = range_len(df_rl, walk_state_node.rl_index, i)
                next_walk_node.l_rep_score = score_step

            # Save/report state
            if next_walk_state == WS_FLANK_R:
                # A final state

                # Add score for right flank
                next_walk_node.walk_score += score_range(df, df_rl, i, df_rl.shape[0], WALK_STATE_COL[next_walk_state])

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


def walk_to_regions(inv_walk, df_rl, region_qry, k_size=0):
    """
    Translate a walk to inner and outer regions.

    :param inv_walk: Inversion state walk.
    :param df_rl: RL-encoded KDE table.
    :param region_qry: Query region.
    :param k_size: K-mer size.

    :return: A tuple of inner and outer regions (`pavlib.seq.Region` objects).
    """

    # Validate walk
    if len(set([walk_state for rl_index, walk_state in inv_walk.trace_list])) != len(inv_walk.trace_list):
        raise RuntimeError(f'Walk node trace has repeated states: {inv_walk}')

    for i in range(1, len(inv_walk.trace_list)):
        if inv_walk.trace_list[i][1] <= inv_walk.trace_list[i - 1][1]:
            raise RuntimeError(f'Walk node trace states are not monotonically increasing: {inv_walk}')

    for i in range(1, len(inv_walk.trace_list)):
        if inv_walk.trace_list[i][0] <= inv_walk.trace_list[i - 1][0]:
            raise RuntimeError(f'Walk node trace indexes are not monotonically increasing: {inv_walk}')

    for i in range(len(inv_walk.trace_list)):
        if inv_walk.trace_list[i][1] not in {WS_FLANK_L, WS_FLANK_R, WS_INV, WS_REP_L, WS_REP_R}:
            raise RuntimeError(f'Walk node trace has invalid state: {inv_walk}')

    # Get regions
    state_dict = {
        walk_state: rl_index for rl_index, walk_state in inv_walk.trace_list
    }

    if WS_INV not in state_dict:
        raise RuntimeError(f'Walk node trace does not contain an inverted state: {inv_walk}')

    if WS_FLANK_R not in state_dict or WS_FLANK_L not in state_dict:
        raise RuntimeError(f'Walk node trace does not contain flanking states: {inv_walk}')

    i_inner_l = state_dict[WS_INV]
    i_inner_r = state_dict.get(WS_REP_R, state_dict[WS_FLANK_R])

    i_outer_l = state_dict.get(WS_REP_L, i_inner_l)
    i_outer_r = state_dict[WS_FLANK_R]

    region_qry_outer = pavlib.seq.Region(
        region_qry.chrom,
        df_rl.loc[i_outer_l, 'POS_QRY'] + region_qry.pos,
        df_rl.loc[i_outer_r, 'POS_QRY'] + region_qry.pos + k_size,
        is_rev=region_qry.is_rev
    )

    region_qry_inner = pavlib.seq.Region(
        region_qry.chrom,
        df_rl.loc[i_inner_l, 'POS_QRY'] + region_qry.pos,
        df_rl.loc[i_inner_r, 'POS_QRY'] + region_qry.pos + k_size,
        is_rev=region_qry.is_rev
    )

    return region_qry_inner, region_qry_outer


#
# Region lift and expand
#

def lift_ref_to_qry(region_ref, align_lift, index_set=None, match_index=True):
    """
    Lift reference region to query for inversion detection. Allow overlapping alignment records and choose the best one
    with the highest-scoring alignment records at flanks if there are overlapping alignments at this locus.

    :param region_ref: Reference region to lift.
    :param align_lift: Alignment lift object.
    :param index_set: If not `None`, restrict search to this set of aligned indexes.
    :param match_index: If `True`, then both left and right regions must land on the same alignment record. Set to
        `True` to handle inversions expected to be covered by one contiguous alignment and `False` if the inversion
        might have broken alignments into several aligned fragments.

    :return: A lifted query regio or `None` if to lift was found.
    """

    qry_lift = align_lift.lift_to_qry(region_ref.chrom, (region_ref.pos, region_ref.end))

    if qry_lift[0] is None or qry_lift[1] is None:
        return None

    if index_set is not None:
        qry_lift = [
            [
                l for l in lift_list if l[5] in index_set
            ] for lift_list in qry_lift
        ]

    if len(qry_lift[0]) == 0 or len(qry_lift[1]) == 0:
        return None

    # Choose best matches
    best_score = 0.0
    best_len = 0
    best_lift = None

    for lift_l in qry_lift[0]:
        for lift_r in qry_lift[1]:

            # Match reference and orientation
            if lift_l[0] != lift_r[0] or lift_l[2] != lift_r[2]:
                continue

            # Same alignment record if match_index
            if match_index and lift_l[5] != lift_r[5]:
                continue

            # Find best region (best score, break ties by length)
            lift_score = lift_l[6] + lift_r[6]
            lift_len = np.abs(lift_l[1] - lift_r[1])

            if lift_score > best_score or (lift_score == best_score and lift_len > best_len):
                best_score = lift_score
                best_len = lift_len

                best_lift = (lift_l, lift_r)

    if best_lift is None:
        return None

    return pavlib.seq.Region(
        best_lift[0][0], best_lift[0][1], best_lift[1][1], is_rev=best_lift[0][2]
    )


def expand_region(region_ref, df_rl, df_fai):
    """
    Expands region `region_ref` in place for the next inversion.

    :param region_ref: Reference region to expand inplace.
    :param df_rl: Run-length encoded table from the inversion search over `region_ref`.
    :param df_fai: Reference sequence index.

    """
    expand_bp = np.int32(len(region_ref) * pavlib.const.INV_EXPAND_FACTOR)

    if df_rl is not None and df_rl.shape[0] > 2:
        # More than one state. Expand disproportionately if reference was found up or downstream.

        if df_rl.iloc[0]['STATE'] == 0:
            region_ref.expand(
                expand_bp, min_pos=0, max_end=df_fai, shift=True, balance=0.25
            )  # Ref upstream: +25% upstream, +75% downstream

        elif df_rl.iloc[-1]['STATE'] == 0:
            region_ref.expand(
                expand_bp, min_pos=0, max_end=df_fai, shift=True, balance=0.75
            )  # Ref downstream: +75% upstream, +25% downstream

        else:
            region_ref.expand(
                expand_bp, min_pos=0, max_end=df_fai, shift=True, balance=0.5
            )  # +50% upstream, +50% downstream

    else:
        region_ref.expand(expand_bp, min_pos=0, max_end=df_fai, shift=True, balance=0.5)  # +50% upstream, +50% downstream


def test_kde(df_rl, binom_prob=0.5, two_sided=True, trim_fwd=True):
    """
    Test KDE for inverted states and complete inversion.

    Must also check for the proportion of inverted states, test will produce a low p-value if most states are

    :param df_rl: Run-length encoded state table (Generated by pavlib.kde.rl_encoder()).
    :param binom_prob: Probability of inverted states for a bionomial test of equal frequency.
    :param two_sided: `True`, for a two-sided test (p-val * 2).u
    :param trim_fwd: Trim FWD oriented states from the start and end of `df_rl` if `True`; do not penalize test for
        including flanking un-inverted sequence.

    :return: Binomial p-value.
    """

    # Trim forward-oriented flanking sequence
    if trim_fwd:
        if df_rl.shape[0] > 0 and df_rl.iloc[0]['STATE'] == pavlib.inv.KDE_STATE_FWD:
            df_rl = df_rl.iloc[1:]

        if df_rl.shape[0] > 0 and df_rl.iloc[-1]['STATE'] == pavlib.inv.KDE_STATE_FWD:
            df_rl = df_rl.iloc[:-1]

    df_rl = df_rl.reset_index(drop=True)

    # Get probability of inverted states based on a bionmial test of equal frequency of forward and reverse states
    state_fwd_n = np.sum(df_rl.loc[df_rl['STATE'] == pavlib.inv.KDE_STATE_FWD, 'LEN_KDE'])
    state_rev_n = np.sum(df_rl.loc[df_rl['STATE'] == pavlib.inv.KDE_STATE_REV, 'LEN_KDE'])

    p_null = (1 - scipy.stats.binom.cdf(state_rev_n, state_rev_n + state_fwd_n, binom_prob)) * (2 if two_sided else 1)

    return p_null


#
# Logging
#

def _write_log(message, log):
    """
    Write message to log.

    :param message: Message.
    :param log: Log or `None`.
    """

    if log is None:
        return

    log.write(message)
    log.write('\n')

    log.flush()
