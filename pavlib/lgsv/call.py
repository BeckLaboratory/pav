"""
Large alignment-truncating variant calling.
"""

import collections
import numpy as np
import os

import svpoplib

from .. import align
from .. import const
from .. import inv
from .. import io
from .. import kde

from .chain import AnchorChainNode
from .chain import get_chain_set

from .interval import AnchoredInterval
from .interval import get_segment_table

from .util import dot_graph_writer
from .util import get_min_anchor_score

from .variant import DeletionVariant
from .variant import ComplexVariant
from .variant import InsertionVariant
from .variant import InversionVariant
from .variant import NullVariant
from .variant import PatchVariant
from .variant import TandemDuplicationVariant
from .variant import score_segment_transitions
from .variant import try_variant

class VarRegionKde:
    def __init__(self,
                 interval, caller_resources,
                 qry_ref_max_ident=0.80,
                 max_seg_n=50,
                 max_qry_len_kde=1e6
                 ):
        """
        Create a variant region with KDE. Determine if the reference and query sequences should be tried for a variant.

        :param interval: Anchored interval over alignment fragments.
        :param caller_resources: Caller resources.
        :param qry_ref_max_ident: If the query and reference identity is this value or greater, do not try a variant
            call. This often occurs when alignments gap over divergent but similar sequences, but aligners leave the
            fragments unaligned. These look like substitutions (INS + DEL of similar size), but are not. These variants
            require better methods to match the unaligned sequences, which PAV does not currently have. For balanced
            inversions, the reference and query sequences must also be within this length
            (i.e. min / max >= qry_ref_max_ident).
        :param max_seg_n: If the number of segments is this value or greater, do not try a variant call. No limit if
            None.
        :param max_qry_len_kde: If the query sequence length is this value or greater, do not try a variant call. No
            limit if None.
        """

        # Default values
        self.df_kde = None
        self.df_rl = None

        self.try_inv = False
        self.try_var = False

        if interval.len_ref <= 0 or \
                np.min([interval.len_qry, interval.len_ref]) / np.max([interval.len_qry, interval.len_ref]) < qry_ref_max_ident:

            self.try_var = True
            return

        if max_seg_n is not None and interval.seg_n >= max_seg_n:
            return

        kmer_n = len(interval.region_qry) - caller_resources.k_util.k_size + 1  # Number of k-mers in the query sequence

        if kmer_n <= 0:
            self.try_var = True
            return

        # Build KDE
        region_ref_exp = interval.region_ref.expand(
            len(interval.region_ref) * 0.2,
            max_end=caller_resources.ref_fai[interval.region_ref.chrom],
            shift=False,
            inplace=False
        )

        region_qry_exp = interval.region_qry.expand(
            len(interval.region_qry) * 0.2,
            max_end=caller_resources.qry_fai[interval.region_qry.chrom],
            shift=False,
            inplace=False
        )

        if len(region_qry_exp) <= max_qry_len_kde:

            # Match complex sequence to the gap region
            self.df_kde = inv.get_state_table(
                region_ref=region_ref_exp,
                region_qry=region_qry_exp,
                ref_fa_name=caller_resources.ref_fa_name,
                qry_fa_name=caller_resources.qry_fa_name,
                ref_fai=caller_resources.ref_fai,
                qry_fai=caller_resources.qry_fai,
                is_rev=interval.region_qry.is_rev,
                k_util=caller_resources.k_util,
                kde_model=caller_resources.kde_model,
                max_ref_kmer_count=caller_resources.pav_params.inv_max_ref_kmer_count,
                expand_bound=True,
                log=caller_resources.log_file
            )

            if self.df_kde.shape[0] == 0:

                # No matching k-mers
                self.try_var = True
                return

            if self.df_kde is not None:

                kmer_n_kde = self.df_kde.shape[0]  # Number of k-mers not dropped by KDE (found in both query and ref in either orientation)

                self.df_rl = kde.rl_encoder(self.df_kde)

                qry_len_rl = np.sum(self.df_rl['LEN_QRY'])

                if qry_len_rl > 0:
                    prop_rev = np.sum(self.df_rl.loc[self.df_rl['STATE'] == inv.KDE_STATE_REV, 'LEN_KDE']) / qry_len_rl
                    prop_fwdrev = np.sum(self.df_rl.loc[self.df_rl['STATE'] == inv.KDE_STATE_FWDREV, 'LEN_KDE']) / qry_len_rl
                    prop_fwd = np.sum(self.df_rl.loc[self.df_rl['STATE'] == inv.KDE_STATE_FWD, 'LEN_KDE']) / qry_len_rl
                else:
                    prop_rev = 0.0
                    prop_fwdrev = 0.0
                    prop_fwd = 0.0

                # If forward k-mers make most of the original sequence, do not try a variant. The alignment dropped here, but
                # sequences are similar. Additional methods are needed to refine the missing alignments for these sequences,
                # which PAV currently does not have. This prevents false substitution CSVs (INS + DEL) will be created.
                if np.sum(self.df_kde['STATE'] == 0) / kmer_n > qry_ref_max_ident:
                    return

                # Try a non-INV variant call if tests to this point have passed.
                # Weed out misalignments around inverted repeats (human chrX)
                self.try_var = (
                    kmer_n_kde / kmer_n < 0.8  # Too many missing k-mers to make a call, try a variant
                ) or (
                    prop_fwdrev < 0.5 or prop_fwd + prop_fwdrev < 0.90  # Weed out misalignments around inverted repeats (human chrX)
                )

                # Test states for inverted or reference states
                if np.any(self.df_rl['STATE'] == inv.KDE_STATE_REV):
                    p_binom = inv.test_kde(self.df_rl)  # Test KDE for inverted state significance

                    self.try_inv = p_binom < 0.01 and prop_rev >= 0.5

                # Check all segments, should belong to the gap region
                qry_exp_diff = interval.region_qry.pos - region_qry_exp.pos

                if self.try_inv:
                    qry_start = interval.df_segment.iloc[-1 if interval.is_rev else 0]['QRY_END']

                    for index, row in interval.df_segment.loc[~ interval.df_segment['IS_ANCHOR']].iterrows():
                        pos, end = sorted([abs(row['QRY_POS'] - qry_start), abs(row['QRY_END'] - row['QRY_POS'])])

                        df_kde_seg = self.df_kde.loc[
                            (self.df_kde['INDEX'] >= (pos + qry_exp_diff)) * (self.df_kde['INDEX'] <= (end + qry_exp_diff))
                        ]

                        if df_kde_seg.shape[0] / len(interval.region_qry) < 0.05:  # Skip diminuitive segments
                            continue

                        df_kde_seg = df_kde_seg.loc[df_kde_seg['STATE'].isin({inv.KDE_STATE_FWDREV, inv.KDE_STATE_REV})]

                        prop_mer = df_kde_seg.shape[0] / (end - pos - caller_resources.k_util.k_size + 1)

                        if prop_mer < 0.5:
                            self.try_inv = False

        else:
            # KDE skipped
            self.try_var = True

            # try variant only if k-mer Jaccard distance is within the threshold
            # if svpoplib.aligner.jaccard_distance(
            #     seq.region_seq_fasta(interval.region_qry, caller_resources.qry_fa_name),
            #     seq.region_seq_fasta(interval.region_ref, caller_resources.ref_fa_name),
            #     caller_resources.k_util.k_size
            # ) < qry_ref_max_ident:
            #     self.try_var = True


def call_from_align(caller_resources, min_anchor_score=const.DEFAULT_MIN_ANCHOR_SCORE, dot_dirname=None):
    """
    Create a list of variant calls from alignment table.

    :param caller_resources: Caller resources.
    :param min_anchor_score: Minimum allowed score for an alignment segment to anchor a variant call.
    :param dot_dirname: Directory where graph dot files are written.

    :return: A list of variant call objects.
    """

    variant_call_list = list()
    variant_id_set = set()

    min_anchor_score = get_min_anchor_score(min_anchor_score, caller_resources.score_model)

    for query_id in caller_resources.df_align_qry['QRY_ID'].unique():
        if caller_resources.verbose:
            print(f'Query: {query_id}: Chaining', file=caller_resources.log_file, flush=True)

        df_align = caller_resources.df_align_qry.loc[
            caller_resources.df_align_qry['QRY_ID'] == query_id
        ].reset_index(drop=True)

        if list(df_align['QRY_ORDER']) != list(df_align.index):
            raise RuntimeError(f'Query {query_id} order is out of sync with QRY_ORDER (Program Bug)')

        # Chain alignment records
        chain_set = get_chain_set(df_align, caller_resources, min_anchor_score)

        if caller_resources.verbose:
            n_chains = len(chain_set)

            if n_chains > 0:
                max_chain = max([end_index - start_index for start_index, end_index in chain_set])
            else:
                max_chain = 0

            print(f'Query: {query_id}: chains={n_chains}, max_index_dist={max_chain}', file=caller_resources.log_file, flush=True)

        # Variant candidates
        sv_dict = dict()  # Key: interval range (tuple), value=SV object

        for start_index, end_index in chain_set:
            sv_dict[(start_index, end_index)] = call_from_interval(start_index, end_index, df_align, caller_resources)

        # Choose variants along the optimal path
        optimal_interval_list = list()

        new_var_list = find_optimal_svs(sv_dict, chain_set, df_align, caller_resources)

        for variant_call in new_var_list:
            variant_call.variant_id = svpoplib.variant.version_id_name(variant_call.variant_id, variant_id_set)
            variant_id_set.add(variant_call.variant_id)

        variant_call_list.extend(new_var_list)

        # Write dot file
        if dot_dirname is not None:
            dot_filename = os.path.join(dot_dirname, f'lgsv_graph_{query_id}.dot.gz')

            with io.PlainOrGzFile(dot_filename, 'wt') as out_file:
                dot_graph_writer(
                    out_file, df_align, chain_set, optimal_interval_list, sv_dict, graph_name=f'"{query_id}"', forcelabels=True
                )

    # Return variant calls
    return variant_call_list

def call_from_interval(start_index, end_index, df_align, caller_resources, min_sum=0.0):
    """
    Call variant from an interval.

    :param start_index: Start index in df_align.
    :param end_index: End index in df_align.
    :param df_align: Query alignment records for one query sequence sorted and indexed in query order.
    :param caller_resources: Caller resources.
    :param min_sum: The sum of the variant score and least anchor score (lesser score of the two anchors) must be
        greater than this value.

    :return: Variant call. If no variant is called, returns a `NullVariant` object.
    """

    chain_node = AnchorChainNode(start_index, end_index)

    interval = AnchoredInterval(chain_node, df_align, caller_resources)

    if caller_resources.verbose:
        print(f'Trying interval: interval=({interval.chain_node.start_index}, {interval.chain_node.end_index}), region_qry={interval.region_qry}, region_ref={interval.region_ref}', file=caller_resources.log_file, flush=True)

    # Get variant region KDE
    var_region_kde = VarRegionKde(interval, caller_resources)

    # Initialize to Null variant
    variant_call = NullVariant()

    # Try variants
    if var_region_kde.try_var:
        # Try INS
        variant_call = try_variant(
            InsertionVariant, interval, caller_resources, variant_call, var_region_kde
        )

        # Try DEL
        variant_call = try_variant(
            DeletionVariant, interval, caller_resources, variant_call, var_region_kde
        )

        # Try tandem duplication
        variant_call = try_variant(
            TandemDuplicationVariant, interval, caller_resources, variant_call, var_region_kde
        )

        # Try Inversion
        variant_call = try_variant(
            InversionVariant, interval, caller_resources, variant_call, var_region_kde
        )

        # Try complex
        variant_call = try_variant(
            ComplexVariant, interval, caller_resources, variant_call, var_region_kde
        )
    else:
        variant_call = PatchVariant(interval)

    if not variant_call.is_null():
        if (not interval.is_anchor_pass) or interval.aligned_pass_prop < caller_resources.pav_params.lg_cpx_min_aligned_prop:
            variant_call.filter = interval.align_filters if interval.align_filters else align.records.FILTER_LCALIGN

    if caller_resources.verbose:
        print(f'Call ({variant_call.filter}): interval=({interval.chain_node.start_index}, {interval.chain_node.end_index}), var={variant_call}', file=caller_resources.log_file, flush=True)

    # Set to Null variant if anchors cannot support the variant call
    if not variant_call.is_null() and variant_call.score_variant + variant_call.anchor_score_min < min_sum:
        variant_call = NullVariant()

    return variant_call

def find_optimal_svs(sv_dict, chain_set, df_align, caller_resources, optimal_interval_list=None):
    """

    :param sv_dict: SV dictionary.
    :param chain_set: Set of intervals in the chain along this query.
    :param df_align: Query alignment records for one query sequence sorted and indexed in query order.
    :param caller_resources: Caller resources
    :param optimal_interval_list: A list object to append optimal intervals to or None. If the optimal path through
        intervals is required (i.e. for DOT file generation), then pass a list object and it will be populated. Pass
        None (default) otherwise.

    :return: A tuple of two elements, a list of varuiant call objects and a list of intervals in the chosen optimal
        path through aligned fragments.
    """

    # Initialize Bellman-Ford
    top_score = np.full(df_align.shape[0], -np.inf)   # Score, top-sorted graph
    top_tb = np.full(df_align.shape[0], -2)           # Traceback (points to parent node with the best score), top-sorted graph

    top_score[0] = 0
    top_tb[0] = -1

    # Create a graph by nodes (anchor graph nodes are scored edges)
    node_link = collections.defaultdict(set)

    for start_index, end_index in chain_set:
        node_link[start_index].add(end_index)

    for start_index in range(df_align.shape[0] - 1):
        node_link[start_index].add(start_index + 1)

    # Update score by Bellman-Ford
    for start_index in range(df_align.shape[0]):
        base_score = top_score[start_index]

        # print(f'Start: {start_index:d}: {base_score:.2E}')  # DBGTMP

        if np.isneginf(base_score):  # Unreachable
            raise RuntimeError(f'Unreachable node at index {start_index}')

        for end_index in sorted(node_link[start_index]):

            # Score for this edge (Initialize to optimal score leading up to the edge)
            score = base_score

            sv_score = sv_dict[start_index, end_index].score_variant \
                if (start_index, end_index) in sv_dict else -np.inf

            if not np.isneginf(sv_score):
                # Variant call (or patch variant across alignment artifacts)
                # Edge weight: Variant score and half of each anchor.
                score +=  \
                    sv_score + \
                    df_align.loc[end_index]['SCORE'] / 2 + \
                    df_align.loc[start_index]['SCORE'] / 2

            else:
                # No variant call
                # Can be from in-chain edges (anchor candidates) or not-in-chain (edges added between sequential alignment records)
                # Edge weight: Score by complex structure (template switches and gap penalties for each segment).

                # Initial score: Template switches and gap penalties
                score += \
                    score_segment_transitions(
                        get_segment_table(start_index, end_index, df_align, caller_resources),
                        caller_resources
                    )

                if (start_index, end_index) in chain_set:
                    # In-chain edge
                    # Score: add half of anchor alignment scores and penalize by the reference gap
                    score += \
                        df_align.loc[end_index]['SCORE'] / 2 + \
                        df_align.loc[start_index]['SCORE'] / 2

            # print(f'\t* End: {end_index:d}: {score:.2E}')  # DBGTMP

            if score > top_score[end_index]:
                # print('\t\t* New top')  # DBGTMP
                top_score[end_index] = score
                top_tb[end_index] = start_index

    last_node = df_align.shape[0] - 1

    if optimal_interval_list is None:
        optimal_interval_list = list()

    while True:
        first_node = top_tb[last_node]

        if first_node < 0:
            break

        optimal_interval_list.append((first_node, last_node))

        last_node = first_node

    # Return variants
    new_var_list = [
        sv_dict[node_interval] for node_interval in optimal_interval_list \
            if node_interval in sv_dict and not sv_dict[node_interval].is_null() and not sv_dict[node_interval].is_patch
    ]

    if caller_resources.verbose:
        print(f'Call: {len(new_var_list)} optimal variants', file=caller_resources.log_file, flush=True)

    return new_var_list
