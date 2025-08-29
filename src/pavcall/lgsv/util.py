"""
Variant call utilities.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TextIO

import numpy as np
import polars as pl
import sys

import agglovar

from .. import align as pavcall_align
from .. import kde as pavcall_kde
from .. import seq as pavcall_seq
from .. import params as pavcall_params


@dataclass
class CallerResources(object):
    df_align_qry: pl.DataFrame
    df_align_qryref: pl.DataFrame
    df_align_none: pl.DataFrame
    ref_fa_filename: str
    qry_fa_filename: str
    score_model: Optional[pavcall_align.score.ScoreModel] = field(default_factory=pavcall_align.score.get_score_model)
    k_util: Optional[agglovar.kmer.util.KmerUtil] = None
    inv_params: Optional[dict[str, Any]] = field(default_factory=dict)
    kde_model: Optional[pavcall_kde.Kde] = field(default_factory=lambda: pavcall_kde.KdeTruncNorm())
    log_file: Optional[TextIO] = sys.stdout
    verbose: bool = True
    pav_params: Optional[pavcall_params.PavParams] = field(default_factory=lambda: pavcall_params.PavParams(), repr=False)

    """
    Container of resources needed by routines resolving large variants.

    Attributes:
        df_align_qry: Alignment table (QRY trimmed)
        df_align_qryref: Alignment table (QRY & REF trimmed)
        df_align_none: Alignment table (No trimming)
        ref_fa_filename: Reference FASTA filename.
        qry_fa_filename: Query FASTA filename.
        ref_fai_filename: Reference FAI filename.
        qry_fai_filename: Query FAI filename.
        ref_fai: Reference FAI table.
        qry_fai: Query FAI table.
        score_model: Alignemnt score model.
        k_util: K-mer utility.
        inv_params: Inversion parameters.
        kde_model: KDE model.
        align_lift: Object for lifting alignment coordinates between query and reference through the alignment.
        cache_qry_upper: Query sequence cache used for homology searches. Caches the last query sequence in
            upper-case.
        cache_ref_upper: Reference sequence cache used for homology searches. Caches the last reference sequence in
            upper-case.
    """

    def __post_init__(self):

        if self.k_util is None:
            self.k_util = agglovar.kmer.util.KmerUtil(self.pav_params.inv_k_size)

        self.ref_fai_filename = self.ref_fa_filename + '.fai'
        self.qry_fai_filename = self.qry_fa_filename + '.fai'

        self.ref_fai = agglovar.fa.read_fai(self.ref_fai_filename)
        self.qry_fai = agglovar.fa.read_fai(self.qry_fai_filename)

        self.align_lift = pavcall_align.lift.AlignLift(self.df_align_qry, self.qry_fai)

        self.cache_qry_upper = pavcall_seq.LRUSequenceCache(fa_filename=self.qry_fa_name, max_size=1, upper=True)
        self.cache_ref_upper = pavcall_seq.LRUSequenceCache(fa_filename=self.ref_fa_name, max_size=1, upper=True)

        self.inv_params = dict(self.inv_params)

        for key in list(self.inv_params.keys()):
            if key not in {'nc_ref', 'nc_qry', 'region_limit', 'init_expand', 'min_kmers',
                              'max_ref_kmer_count', 'repeat_match_prop', 'min_inv_kmer_run', 'min_qry_ref_prop'}:
                del self.inv_params[key]


def record_to_paf(row_seg, ref_fai, qry_fai, mapq_summary='max'):
    """
    Convert the row of a segment table to PAF format.

    `row_seg` is a complex segment record with "MAPQ" and "CIGAR" fields added.

    :param row_seg: Segment table row.
    :param ref_fai: Reference FASTA index.
    :param qry_fai: Query FASTA index.
    :param mapq_summary: If multiple alignment records were aggregated, then the MAPQ value is a list of MAPQ values
        from the original alignments. When multiple MAPQ vaules are found, summarize them to a single value with this
        approach. "max": maximum value (default), "min": minimum value, "mean": average value.

    :return: PAF record row.
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
    #         raise RuntimeError(f'Unhandled clipping in CIGAR string: {op_code} at CIGAR index {cigar_index}: Expected clipped bases at the beginning and end of the CIGAR string only.')
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
    #     raise RuntimeError(f'Missing "STRAND", "REV", or "IS_REV" column in segment table: Record {row_seg["INDEX"] if "INDEX" in row_seg else row_seg.name}')
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
        chain_set,
        optimal_interval_list,
        sv_dict,
        graph_name: str = 'Unnamed_Graph',
        force_labels: bool = True,
        anchor_width: float = 2.5,
        index_interval: Optional[tuple[int, int]] = None,
        discard_null: bool = True
):
    """
    Write a DOT graph file for a set of alignments.
    
    Params:
        out_file: Output DOT file (open filehandle, not filename).
        df_align: Table of aligned records (anchors).
        chain_set: The set of chained elements.
        optimal_interval_list: A list of intervals chosen for the optimal path through the alignment graph.
        sv_dict: Dictionary of SVs where the key is an interval (tuple) and the value is a variant call object.
        graph_name: Name of the graph.
        force_labels: Force all labels (do not omit).
        anchor_width: Line width of anchors.
        index_interval: Only output a graph for nodes in this interval (tuple of min and max indexes, inclusive).
        discard_null: Discard null variants if True.
    """

    raise NotImplementedError # Update for Polars tables

    if index_interval is not None:
        min_index, max_index = index_interval
    else:
        min_index, max_index = -np.inf, np.inf

    # Header
    out_file.write(f'graph {graph_name} {{\n')

    # Attributes
    if force_labels:
        out_file.write('    forcelabels=true;\n')

    out_file.write('    overlap=false;\n')

    # Anchor and interval sets
    optimal_interval_set = set(optimal_interval_list)
    variant_interval_set = set(sv_dict.keys())

    optimal_anchor_set = {index for index_pair in optimal_interval_set for index in index_pair}
    variant_anchor_set = {index for index_pair in sv_dict.keys() for index in index_pair}

    # optimal_interval_set = set(optimal_interval_list)

    # Add nodes
    for index, row in df_align.iterrows():

        if index < min_index or index > max_index:
            continue

        if index in optimal_anchor_set:
            color = 'blue'
        elif index in variant_anchor_set:
            color = 'black'
        else:
            color = 'gray33'

        width = anchor_width if index in variant_anchor_set else 1

        out_file.write(f'    n{index} [label="{index} ({row["INDEX"]})\n{row["#CHROM"]}:{row["POS"]}-{row["END"]} ({"-" if row["IS_REV"] else "+"})\ns={row["SCORE"]}", penwidth={width}, color="{color}"]\n')
        # out_file.write(f'    n{index} [label="{index} ({row["INDEX"]}) - {row["#CHROM"]}:{row["POS"]}-{row["END"]} {"-" if row["IS_REV"] else "+"} s={row["SCORE"]}", penwidth={width}, color="{color}"]\n')

    # Add candidate edges
    for start_index, end_index in chain_set:

        if start_index < min_index or end_index > max_index:
            continue

        if (start_index, end_index) in optimal_interval_set:
            color = 'blue'
        elif (start_index, end_index) in variant_interval_set:
            color = 'black'
        else:
            color = 'gray33'

        width = anchor_width if (start_index, end_index) in variant_interval_set else 1

        if (start_index, end_index) not in sv_dict or sv_dict[start_index, end_index].is_null():
            var_name = 'NullVar'
        elif sv_dict[start_index, end_index].is_patch:
            var_name = 'AlignPatch (No Variant)'
        else:
            var_name = sv_dict[start_index, end_index].variant_id

        if discard_null and sv_dict[start_index, end_index].is_null():
            var_label = ''
        else:
            var_label = f'{var_name}\n(s={sv_dict[start_index, end_index].score_variant})'
            # var_label = f'{var_name} (s={sv_dict[start_index, end_index].score_variant})'

        out_file.write(f'    n{start_index} -- n{end_index} [label="{var_label}", penwidth={width}, color="{color}"]\n')

    # Add adjacent edges (not anchor candidates)
    for start_index in range(df_align.shape[0] - 1):
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


def get_min_anchor_score(
        min_anchor_score: str | int | float,
        score_model: pavcall_align.score.ScoreModel
) -> float:
    """
    Get the minimum score of an anchoring alignment. The score may be expressed as an absolute alignment score (value
    is numeric or a string representing a number), or a number of matching basepairs (string ending in "bp").

    Each large variant is anchored by a pair of alignment records where the large variant appears between them.
    Anchoring alignments must be sufficiently confident or evidence for the variant is not well-supported.

    Args:
        min_anchor_score: Minimum score of an anchoring alignment. The score may be expressed as an absolute alignment
            score (numeric or string representing a number), or a number of matching basepairs (string ending in "bp").
        score_model: Score model to use for converting basepair scores to alignment scores.

    Returns:
        Minimum score for anchoring alignments.

    Raises:
        ValueError: If min_anchor_score is not a string or numeric.
    """

    if isinstance(min_anchor_score, str):
        min_anchor_score_str = min_anchor_score.strip()

        if min_anchor_score_str.lower().endswith('bp'):
            try:
                min_anchor_score_bp = int(min_anchor_score_str[:-2].strip())
            except ValueError:
                raise ValueError(f'min_anchor_score: "bp" must come before an integer: "{min_anchor_score}"')

            if min_anchor_score_bp < 0:
                raise ValueError(f'min_anchor_score: "bp" must come before a non-negative integer: "{min_anchor_score}"')

            if score_model is None:
                raise ValueError('score_model is None')

            return score_model.match(min_anchor_score_bp)

        else:
            try:
                return abs(float(min_anchor_score))
            except ValueError:
                raise ValueError(f'min_anchor_score is a string that does not represent a numeric value: {min_anchor_score}')

    else:
        try:
            # noinspection PyTypeChecker
            return float(min_anchor_score)
        except ValueError:
            raise ValueError(f'min_anchor_score is not a string or numeric: type={type(min_anchor_score)}')
