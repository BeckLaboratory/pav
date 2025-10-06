"""Functions for transforming files."""

__all__ = [
    'dot_graph_writer',
    'record_to_paf'
]

from typing import Iterable, Optional, TextIO

import polars as pl

from .variant import Variant


def record_to_paf(row_seg, ref_fai, qry_fai, mapq_summary='max'):
    """Convert the row of a segment table to PAF format.

    `row_seg` is a complex segment record with "MAPQ" and "CIGAR" fields added.

    :param row_seg: Segment table row.
    :param ref_fai: Reference FASTA index.
    :param qry_fai: Query FASTA index.
    :param mapq_summary: If multiple alignment records were aggregated, then the MAPQ value is a list of MAPQ values
        from the original alignments. When multiple MAPQ vaules are found, summarize them to a single value with this
        approach. "max": maximum value (default), "min": minimum value, "mean": average value.

    :returns: PAF record row.
    """
    raise NotImplementedError

    # match_n = 0
    # align_len = 0
    # cigar_index = -1
    #
    # cigar_list = list(align.op.as_tuples(row_seg['CIGAR']))
    #
    # # Remove clipping and adjust coordinates
    # if cigar_list[0][0] == align.op.H:
    #     cigar_list = cigar_list[1:]
    #     cigar_index += 1
    #
    # if cigar_list[0][0] == align.op.S:
    #     cigar_list = cigar_list[1:]
    #     cigar_index += 1
    #
    # if cigar_list[-1][0] == align.op.H:
    #     cigar_list = cigar_list[:-1]
    #
    # if cigar_list[-1][0] == align.op.S:
    #     cigar_list = cigar_list[:-1]
    #
    # cigar = ''.join([f'{op_len}{op_code}' for op_code, op_len in cigar_list])
    #
    # # Process CIGAR operations
    # for op_code, op_len in cigar_list:
    #     cigar_index += 1
    #
    #     if op_code == '=':
    #         match_n += op_len
    #         align_len += op_len
    #
    #     elif op_code in {align.op.X, align.op.I, align.op.D}:
    #         align_len += op_len
    #
    #     elif op_code in {align.op.H, align.op.S}:
    #         raise RuntimeError(
    #             f'Unhandled clipping in CIGAR string: {op_code} at CIGAR index {cigar_index}: '
    #             f'Expected clipped bases at the beginning and end of the CIGAR string only.'
    #         )
    #
    #     else:
    #         raise RuntimeError(f'Unrecognized CIGAR op code: {op_code} at CIGAR index {cigar_index}')
    #
    # # Set strand
    # if 'STRAND' in row_seg:
    #     strand = row_seg['STRAND']
    # elif 'IS_REV' in row_seg:
    #     strand = '-' if row_seg['IS_REV'] else '+'
    # elif 'IS_REV' in row_seg:
    #     strand = '-' if row_seg['IS_REV'] else '+'
    # else:
    #     raise RuntimeError(
    #         f'Missing "STRAND", "REV", or "IS_REV" column in segment table: '
    #         f'Record {row_seg["INDEX"] if "INDEX" in row_seg else row_seg.name}'
    #     )
    #
    # # Adjust MAPQ (might be a list of MAPQ values)
    # if isinstance(row_seg['MAPQ'], str):
    #     mapq_list = [int(v) for v in row_seg['MAPQ'].split(',')]
    #
    #     if mapq_summary == 'max':
    #         mapq = np.max(mapq_list)
    #     elif mapq_summary == 'min':
    #         mapq = np.min(mapq_list)
    #     elif mapq_summary == 'mean':
    #         mapq = np.mean(mapq_list)
    #     else:
    #         raise RuntimeError(f'Unrecognized mapq_summary: {mapq_summary}')
    # else:
    #     mapq = row_seg['MAPQ']
    #
    # # Create PAF record
    # return pd.Series(
    #     [
    #         row_seg['QRY_ID'],
    #         qry_fai[row_seg['QRY_ID']],
    #         row_seg['QRY_POS'],
    #         row_seg['QRY_END'],
    #         strand,
    #         row_seg['#CHROM'],
    #         ref_fai[row_seg['#CHROM']],
    #         row_seg['POS'],
    #         row_seg['END'],
    #         match_n,
    #         align_len,
    #         mapq,
    #         cigar
    #     ],
    #     index=[
    #         'QRY_NAME',
    #         'QRY_LEN',
    #         'QRY_POS',
    #         'QRY_END',
    #         'STRAND',
    #         'CHROM',
    #         'CHROM_LEN',
    #         'CHROM_POS',
    #         'CHROM_END',
    #         'MISMATCH_N',
    #         'ALIGN_BLK_LEN',
    #         'MAPQ',
    #         'CIGAR'
    #     ]
    # )


def dot_graph_writer(
        out_file: TextIO,
        df_align: pl.DataFrame,
        sv_dict: dict[tuple[int, int], Variant],
        chain_set: Optional[Iterable[tuple[int, int]]] = None,
        optimal_path_intervals: Optional[Iterable[tuple[int, int]]] = None,
        graph_name: str = 'Unnamed_Graph',
        force_labels: bool = True,
        anchor_width: float = 2.5,
        index_interval: Optional[tuple[int, int]] = None,
) -> None:
    """Write a DOT graph file for a set of alignments.

    :param out_file: Output DOT file (open filehandle, text mode, not filename).
    :param df_align: Table of aligned records.
    :param sv_dict: Dictionary of SVs where the key is an interval (tuple) and the value is a variant call object.
    :param chain_set: The set of chained elements. Derive from `sv_dict.keys()` if None.
    :param optimal_path_intervals: A collection of intervals along the optimal path.
    :param graph_name: Name of the graph.
    :param force_labels: Force all labels (do not omit).
    :param anchor_width: Line width of anchors.
    :param index_interval: Only output a graph for nodes in this interval (tuple of min and max indexes, inclusive).
    """
    df_align_graph = (
        df_align
        .lazy()
        .select(['align_index', 'chrom', 'pos', 'end', 'qry_id', 'qry_pos', 'qry_end', 'is_rev', 'score'])
        .with_columns(
            pl.when('is_rev').then(pl.lit('-')).otherwise(pl.lit('+')).alias('strand')
        )
        .collect()
    )

    # Header
    out_file.write(f'graph {graph_name} {{\n')

    # Attributes
    if force_labels:
        out_file.write('    forcelabels=true;\n')

    out_file.write('    overlap=false;\n')

    # Anchor and interval sets
    optimal_interval_set = set(optimal_path_intervals) if optimal_path_intervals is not None else set()
    chain_set = set(chain_set) if chain_set is not None else set(sv_dict.keys())
    variant_interval_set = {
        (start_index, end_index) for (start_index, end_index), variant in sv_dict.items()
        if not (variant.is_null or variant.is_patch)
    }

    optimal_anchor_set = {index for index_pair in optimal_interval_set for index in index_pair}
    variant_anchor_set = {index for index_pair in variant_interval_set for index in index_pair}

    if len(chain_set) == 0:
        raise ValueError('No intervals found: chain_set is empty')

    if index_interval is not None:
        min_index, max_index = min(index_interval), max(index_interval)
    else:
        index_set = {index for chain_tuple in chain_set for index in chain_tuple}

        if len(index_set) == 0:
            raise ValueError('No intervals found in the graph and ')

        min_index, max_index = min(index_set), max(index_set)

    # Add nodes
    for index in range(min_index, max_index + 1):
        row = df_align_graph.row(index, named=True)

        if index in optimal_anchor_set:
            color = 'blue'
        elif index in variant_anchor_set:
            color = 'black'
        else:
            color = 'gray33'

        width = anchor_width if index in variant_anchor_set else 1

        out_file.write(
            (
                '    '
                'n{index} ['  # noqa: E131
                    'label="'  # noqa: E131
                        '{align_index} (interval={index})\n'  # noqa: E131
                        '{chrom}:{pos:,d}-{end:,d} ({strand})\n'
                        '{qry_id}:{qry_pos:,d}-{qry_end:,d}\n'
                        's={score:.2f}'
                    '", penwidth={width}, color="{color}"'
                ']\n'
            ).format(index=index, width=width, color=color, **row)
        )

    # Add candidate edges
    for start_index, end_index in sorted(chain_set):
        start_index, end_index = sorted([start_index, end_index])

        if start_index < min_index or end_index > max_index:
            continue

        if (start_index, end_index) in optimal_interval_set:
            color = 'blue'
        elif (start_index, end_index) in variant_interval_set:
            color = 'black'
        else:
            color = 'gray33'

        width = anchor_width if (start_index, end_index) in variant_interval_set else 1

        variant = sv_dict[start_index, end_index]

        if variant.is_null or variant.is_patch:
            var_label = ''
        else:
            var_label = f'{variant.variant_id}\n(s={variant.var_score})'

        out_file.write(f'    n{start_index} -- n{end_index} [label="{var_label}", penwidth={width}, color="{color}"]\n')

    # Add adjacent edges (not anchor candidates)
    for start_index in range(min_index, max_index):
        end_index = start_index + 1

        if start_index < min_index or end_index > max_index:
            continue

        if (start_index, end_index) in optimal_interval_set:
            color = 'blue'
        elif (start_index, end_index) in variant_interval_set:
            color = 'black'
        else:
            color = 'gray33'

        if (start_index, end_index) not in chain_set:
            out_file.write(f'    n{start_index} -- n{end_index} [penwidth=1, color="{color}"]\n')

    # Done
    out_file.write('}\n')
