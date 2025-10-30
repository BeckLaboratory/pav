"""Large alignment-truncating variant calling."""


__all__ = [
    'call_from_align',
    'call_from_interval',
    'try_variant',
    'find_optimal_svs'
]

import collections
import os
from pathlib import Path
import polars as pl
from typing import Optional, Type

import numpy as np

from ..const import DEFAULT_MIN_ANCHOR_SCORE
from ..io import PlainOrGzFile

from .interval import AnchoredInterval, score_segment_transitions, get_segment_table
from .chain import get_chain_set, get_min_anchor_score
from .io import dot_graph_writer
from .region_kde import VarRegionKde
from .resources import CallerResources
from .variant import Variant, NullVariant, PatchVariant
from .variant import ComplexVariant, DeletionVariant, InsertionVariant, InversionVariant, TandemDuplicationVariant


def call_from_align(
        caller_resources: CallerResources,
        min_anchor_score: float = DEFAULT_MIN_ANCHOR_SCORE,
        dot_dirname: Optional[Path | str] = None
) -> list[Variant]:
    """Create a list of variant calls from alignment table.

    :param caller_resources: Caller resources.
    :param min_anchor_score: Minimum allowed score for an alignment segment to anchor a variant call.
    :param dot_dirname: Directory where graph dot files are written.

    :returns: A list of variant call objects.
    """
    variant_call_list = list()

    min_anchor_score = get_min_anchor_score(min_anchor_score, caller_resources.score_model)

    qry_id_list = (
        caller_resources.df_align_qry
        .select('qry_id').unique().sort('qry_id').collect()
        .to_series().to_list()
    )

    for qry_id in qry_id_list:
        if caller_resources.verbose:
            print(f'Query: {qry_id}: Chaining', file=caller_resources.log_file, flush=True)

        df_align = (
            caller_resources.df_align_qry
            .filter(pl.col('qry_id') == qry_id)
            .sort(['qry_order'])
            .drop(['align_ops'], strict=False)
            .collect()
        )

        # Chain alignment records
        chain_set = get_chain_set(df_align, caller_resources, min_anchor_score)

        if caller_resources.verbose:
            n_chains = len(chain_set)

            max_chain = max([end_index - start_index for start_index, end_index in chain_set]) if n_chains > 0 else 0

            print(
                f'Query: {qry_id}: chains={n_chains}, max_index_dist={max_chain}',
                file=caller_resources.log_file, flush=True
            )

        # Variant candidates
        sv_dict = dict()  # Key: interval range (tuple), value=SV object

        for start_index, end_index in chain_set:
            sv_dict[(start_index, end_index)] = call_from_interval(
                start_index, end_index, df_align, caller_resources
            )

        # Choose variants along the optimal path
        new_variant_list = find_optimal_svs(sv_dict, chain_set, df_align, caller_resources)

        variant_call_list.extend([variant for variant in new_variant_list if not variant.is_patch])

        # Write dot file
        if dot_dirname is not None and len(chain_set) > 0:
            dot_filename = os.path.join(dot_dirname, f'lgsv_graph_{qry_id}.dot.gz')
            optimal_path_intervals = {(variant.start_index, variant.end_index) for variant in new_variant_list}

            with PlainOrGzFile(dot_filename, 'wt') as out_file:
                dot_graph_writer(
                    out_file=out_file,
                    df_align=df_align,
                    sv_dict=sv_dict,
                    chain_set=chain_set,
                    optimal_path_intervals=optimal_path_intervals,
                    graph_name=f'"{qry_id}"',
                    force_labels=True
                )

    # Return variant calls
    return variant_call_list


def call_from_interval(
        start_index: int,
        end_index: int,
        df_align: pl.DataFrame,
        caller_resources: CallerResources,
        min_sum: float = 0.0,
) -> Variant:
    """Call variant from an interval.

    :param start_index: Start index in df_align.
    :param end_index: End index in df_align.
    :param df_align: Query alignment records for one query sequence sorted and indexed in query order.
    :param caller_resources: Caller resources.
    :param min_sum: The sum of the variant score and least anchor score (lesser score of the two anchors) must be
        greater than this value.

    :returns Variant call. If no variant is called, returns a `NullVariant` object.
    """
    interval = AnchoredInterval(start_index, end_index, df_align, caller_resources)

    if caller_resources.verbose:
        print(
            f'Trying interval: interval=({interval.start_index}, {interval.end_index}), '
            f'region_qry={interval.region_qry}, '
            f'region_ref={interval.region_ref}',
            file=caller_resources.log_file,
            flush=True
        )

    # Try variants
    var_region_kde = VarRegionKde(interval, caller_resources)

    variant_call = NullVariant(interval.start_index, interval.end_index)

    if var_region_kde.try_var:
        variant_call = try_variant(
            InsertionVariant, interval, caller_resources, variant_call, var_region_kde
        )

        variant_call = try_variant(
            DeletionVariant, interval, caller_resources, variant_call, var_region_kde
        )

        variant_call = try_variant(
            TandemDuplicationVariant, interval, caller_resources, variant_call, var_region_kde
        )

        variant_call = try_variant(
            InversionVariant, interval, caller_resources, variant_call, var_region_kde
        )

        variant_call = try_variant(
            ComplexVariant, interval, caller_resources, variant_call, var_region_kde
        )

    elif var_region_kde.try_inv:
        variant_call = try_variant(
            InversionVariant, interval, caller_resources, variant_call, var_region_kde
        )

    else:
        variant_call = PatchVariant(interval.start_index, interval.end_index)

    if caller_resources.verbose:
        print(
            f'Call ({variant_call.filter}): '
            f'interval=({interval.start_index}, {interval.end_index}), '
            f'var={variant_call}',
            file=caller_resources.log_file, flush=True
        )

    # Set to Null variant if anchors cannot support the variant call
    if (
            variant_call.var_score + variant_call.min_anchor_score < min_sum
    ):
        variant_call = PatchVariant(start_index, end_index)

    return variant_call


def find_optimal_svs(
        sv_dict: dict[tuple[int, int], Variant],
        chain_set: set[tuple[int, int]],
        df_align: pl.DataFrame,
        caller_resources: CallerResources,
) -> list[Variant]:
    """Find the optimal path through aligned fragments and produce variant calls.

    :param sv_dict: SV dictionary mapping intervals (tuple of start and end indices) to variant calls.
    :param chain_set: Set of intervals in the chain along this query.
    :param df_align: Query alignment records for one query sequence sorted and indexed in query order.
    :param caller_resources: Caller resources

    :returns: A list of alignment-truncating variants along the optimal path through aligned fragments.
    """
    # Initialize Bellman-Ford
    top_score = np.full(df_align.height, -np.inf)   # Score, top-sorted graph
    top_tb = np.full(df_align.height, -2)           # Traceback (points to parent node with the best score)

    top_score[0] = 0
    top_tb[0] = -1

    # Create a graph by nodes (anchor graph nodes are scored edges)
    node_link = collections.defaultdict(set)

    for start_index, end_index in chain_set:  # Chained nodes
        node_link[start_index].add(end_index)

    for start_index in range(df_align.height - 1):  # Implicit edges to next node
        node_link[start_index].add(start_index + 1)

    # Update score by Bellman-Ford
    for start_index in range(df_align.height):
        base_score = top_score[start_index]

        if np.isneginf(base_score):  # Unreachable
            raise RuntimeError(f'Unreachable node at index {start_index}')

        for end_index in sorted(node_link[start_index]):

            # Score for this edge (Initialize to optimal score leading up to the edge)
            score = base_score

            sv_score = sv_dict[start_index, end_index].var_score if (
                    (start_index, end_index) in sv_dict
            ) else -np.inf

            if not np.isneginf(sv_score):
                # Variant call (or patch variant across alignment artifacts)
                # Edge weight: Variant score and half of each anchor.
                score += (
                    sv_score +
                    df_align[end_index, 'score'] / 2 +
                    df_align[start_index, 'score'] / 2
                )

            else:
                # No variant call
                # Can be from in-chain edges (anchor candidates) or not-in-chain (edges added between sequential
                # alignment records)
                # Edge weight: Score by complex structure (template switches and gap penalties for each segment).

                # Initial score: Template switches and gap penalties
                score += (
                    score_segment_transitions(
                        get_segment_table(start_index, end_index, df_align, caller_resources),
                        caller_resources
                    )
                )

                if (start_index, end_index) in chain_set:
                    # In-chain edge
                    # Score: add half of anchor alignment scores and penalize by the reference gap
                    score += (
                        df_align[end_index, 'score'] / 2 +
                        df_align[start_index, 'score'] / 2
                    )

            if score > top_score[end_index]:
                # print('\t\t* New top')  # DBGTMP
                top_score[end_index] = score
                top_tb[end_index] = start_index

    last_node = df_align.height - 1

    optimal_variant_list: list[Variant] = []

    while True:
        first_node = int(top_tb[last_node])

        if first_node < 0:
            break

        variant = (
            sv_dict[first_node, last_node]
            if (first_node, last_node) in sv_dict
            else PatchVariant(first_node, last_node)
        )

        assert not variant.is_null, 'Found Null variant in optimal path: %s' % str(variant)

        optimal_variant_list.append(variant)
        last_node = first_node

    # Return variants
    if caller_resources.verbose:
        n_var = sum([not variant.is_patch for variant in optimal_variant_list])
        print(f'Call: {n_var} optimal variants', file=caller_resources.log_file, flush=True)

    return optimal_variant_list


def try_variant(
        var_type: Type[Variant],
        interval: AnchoredInterval,
        caller_resources: CallerResources,
        best_variant: Variant,
        var_region_kde: VarRegionKde,
) -> Variant:
    """Try calling a variant of a specific type.

    Call a varian of type `var_type` (a subclass of `Variant`). Check the variant against `best_variant` and return the
    variant of these two (new variant and best variant) with the highest score.

    :param var_type: Variant call type to try (A subclass of `Variant`).
    :param interval: Alignment interval.
    :param caller_resources: Caller resources.
    :param best_variant: Best variant call so far. If there is no variant to check against, a `NullVariant` object
        should be used as this parameter.
    :param var_region_kde: For regions where the inserted and deleted segments are of similar size, this object
        describes how well the inserted sequence matches the deleted sequence in forward or reverse orientation. If
        the forward match is too high, then do not attempt a variant call.

    :returns: Best variant call between the new variant of type `var_type` and `best_variant`. May return a
        `NullVariant` if both `best_variant` is a `NullVariant` and the interval does not match a variant of type
        `var_type`.
    """
    assert best_variant is not None

    # Try variant call
    variant = var_type(interval, caller_resources, var_region_kde)

    if not variant.is_null and variant.var_score <= best_variant.var_score:
        return best_variant

    return variant
