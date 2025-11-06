"""Variant classes."""

__all__ = [
    'CALL_SOURCE',
    'REF_TRACE_COLUMNS',
    'Variant',
    'InsertionVariant',
    'TandemDuplicationVariant',
    'DeletionVariant',
    'InversionVariant',
    'ComplexVariant',
    'NullVariant',
    'PatchVariant',
    'DevVariant'
]

from abc import ABC
import numpy as np
import polars as pl
from typing import Any, Optional

import Bio.Seq

from ..anno import perfect_homology
from ..region import Region

from .interval import AnchoredInterval
from .resources import CallerResources
from .region_kde import VarRegionKde
from .struct import get_ref_trace, smooth_ref_trace, qry_trace_str, ref_trace_str


CALL_SOURCE: str = 'INTER'
"""Variant call source column value."""

_REF_TRACE_COLUMNS_PRE = ['#CHROM', 'POS', 'END', 'DEPTH', 'QRY_ID', 'INDEX']
"""Head reference trace columns."""

_REF_TRACE_COLUMNS_POST = ['FWD_COUNT', 'REV_COUNT', 'TYPE', 'LEN']
"""Tail reference trace columns."""

REF_TRACE_COLUMNS = _REF_TRACE_COLUMNS_PRE + _REF_TRACE_COLUMNS_POST
"""Reference trace columns."""


class Variant(ABC):
    """Base class for variant call objects.

    :ivar interval: Variant interval. Only Null in NullVariant.
    :ivar caller_resources: Caller resources. Only Null in NullVariant.
    :ivar var_region_kde: Variant region KDE. Only Null in NullVariant.
    :ivar region_ref: Reference region. Only Null in NullVariant. Initialized to interval.region_ref, may be altered by
        variant call.
    :ivar region_qry: Query region. Only Null in NullVariant. Initialized to interval.region_qry, may be altered by
        variant call.
    :ivar vartype: Variant type.
    :ivar varsubtype: Variant subtype (e.g. TANDEM).
    :ivar varlen: Variant length.
    :ivar filter_set: Set of filter strings initialized to alignment filters.
    :ivar call_source: Type of evidence supporting the variant call.
    :ivar var_score: Variant score.
    :ivar templ_res: Template resolution flag. Set to True if a variant is aligned across segments and False if variant
        segments are unaligned. Common in complex variants, smaller template switches are left unaligned.
    :ivar seq: Variant sequence.
    :ivar is_patch: Variant is a patch variant. It does not represent a variant call, but should be traversed during
        graph resolution.
    :ivar start_index: Start index of the variant in the alignment table.
    :ivar end_index: End index of the variant in the alignment table.
    :ivar df_ref_trace: Reference trace table set by complex variants (None if not complex).
    :ivar var_index: Set on variant objects to track order and give them a unique index. This is set externally, not by
        variant classes. The PAV variant caller assigns each complex variant a unique index linking it to a
        table of reference and segment traces. With this key, df_ref_trace and df_segment from each variant are
        concatenated into one trace table and one segment table for multiple variant calls, and this index is a
        unique key between the variant table and the concatenated tables.
    """

    interval: Optional[AnchoredInterval]
    caller_resources: Optional[CallerResources]
    var_region_kde: Optional[VarRegionKde]
    region_ref: Optional[Region]
    region_qry: Optional[Region]
    vartype: str
    varsubtype: Optional[str]
    varlen: int
    filter_set: set[str]
    call_source: str
    var_score: float
    templ_res: bool
    seq: Optional[str]
    is_patch: bool
    start_index: int
    end_index: int
    df_ref_trace: Optional[pl.DataFrame]
    var_index: Optional[int]

    def __init__(
            self,
            interval: Optional[AnchoredInterval],
            caller_resources: Optional[CallerResources],
            var_region_kde: Optional[VarRegionKde],
            is_null: bool = False,
            is_patch: bool = False,
            start_index: Optional[int] = None,
            end_index: Optional[int] = None,
    ) -> None:
        """Initialize a variant object.

        If `is_null`, then `interval`, `caller_resources`, and `var_region_kde` must be None. If `is_patch`, then
        `interval` must not be None. If neither `is_null` nor `is_patch` is set, then `interval`, `caller_resources`,
        and `var_region_kde` must not be None.

        :param interval: Variant interval.
        :param caller_resources: Caller resources.
        :param var_region_kde: Variant region KDE.
        :param is_null: Set to True for null variants.
        :param is_patch: Set to True for patch variants.
        """
        if is_null or is_patch:
            assert all([interval is None, caller_resources is None, var_region_kde is None])
            assert all([start_index is not None, end_index is not None])

            self.start_index = start_index
            self.end_index = end_index

        else:
            assert all([interval is not None, caller_resources is not None, var_region_kde is not None])

            self.start_index = interval.start_index
            self.end_index = interval.end_index

        # Key structures
        self.interval = interval
        self.caller_resources = caller_resources
        self.var_region_kde = var_region_kde

        # Variant fields
        self.region_ref = interval.region_ref if interval is not None else None  # Fields: chrom, pos, end
        self.region_qry = interval.region_qry if interval is not None else None  # Fields: qry_id, qry_pos, qry_end

        self.vartype = 'NULL'
        self.varsubtype = None
        self.varlen = 0

        self.filter_set = set((
            interval.df_segment
            .filter(pl.col('is_anchor'))
            .select(pl.col('filter').explode().drop_nulls())
            .to_series().to_list()
        )) if interval is not None else set()

        self.call_source = CALL_SOURCE

        self.var_score = float('-inf')
        self.templ_res = False

        self.df_ref_trace = None
        self._seq = None  # Sequence cache

        self.var_index = None

        self._dup_list: list[dict[str, str | int]] = []

        # Non-variant fields
        self.is_patch = is_patch          # Set to True for patch variants

        # Private fields
        self._is_complete_anno = False  # Set to True after complete_anno() is called
        self._found_variant = False     # Null variant until set to True by the concrete variant class
        self._patch_start_index = None  # Set by patch variants to indicate the start index (no interval)
        self._patch_end_index = None    # Set by patch variants to indicate the end index (no interval)

    @property
    def chrom(self) -> str:
        """Reference chromosome."""
        if self.region_ref is None:
            raise ValueError('Property "chrom" is not set on a null variant')

        return self.region_ref.chrom

    @property
    def pos(self) -> int:
        """Reference start position."""
        if self.region_ref is None:
            raise ValueError('Property "pos" is not set on a null variant')

        return self.region_ref.pos

    @property
    def end(self) -> int:
        """Reference end position."""
        if self.region_ref is None:
            raise ValueError('Property "end" is not set on a null variant')

        return self.region_ref.end

    @property
    def chain_start_index(self) -> int:
        """Chain start index."""
        if self.interval is None:
            raise ValueError('Property "chain_start_index" is not set on a null variant')

        return self.interval.start_index

    @property
    def chain_end_index(self) -> int:
        """Chain end index."""
        if self.interval is None:
            raise ValueError('Property "chain_end_index" is not set on a null variant')

        return self.interval.end_index

    @property
    def seg_n(self) -> int:
        """Number of variant segments including anchors."""
        if self.interval is None:
            raise ValueError('Property "seg_n" is not set on a null variant')

        return self.interval.df_segment.height

    @property
    def df_segment(self) -> pl.DataFrame:
        """Table of segments including aligned and unaligned query segments."""
        if self.interval is None:
            raise ValueError('Property "df_segment" is not set on a null variant')

        return self.interval.df_segment

    @property
    def min_anchor_score(self) -> float:
        """Minimum score of both anchors."""
        return self.interval.min_anchor_score if self.interval is not None else float('-inf')

    @property
    def filter(self) -> list[str]:
        """Sorted list of filters."""
        return sorted(self.filter_set)

    @property
    def is_pass(self) -> bool:
        """Determine if there are no filters set for this variant."""
        return len(self.filter_set) == 0

    @property
    def qry_id(self) -> str:
        """Query sequence ID."""
        if self.region_qry is None:
            raise ValueError('Property "qry_id" cannot be accessed: Missing region_qry')

        return self.region_qry.chrom

    @property
    def qry_pos(self) -> int:
        """Query start position."""
        if self.region_qry is None:
            raise ValueError('Property "qry_pos" cannot be accessed: Missing region_qry')

        return self.region_qry.pos

    @property
    def qry_end(self) -> int:
        """Query end position."""
        if self.region_qry is None:
            raise ValueError('Property "qry_end" cannot be accessed: Missing region_qry')

        return self.region_qry.end

    @property
    def qry_rev(self) -> bool:
        """Query is in reverse orientation relative to the reference."""
        if self.interval is None:
            raise ValueError('Property "qry_rev" is not set on a null variant')

        return self.interval.is_rev

    @property
    def align_index(self) -> list[int]:
        """List of aligned segment indices."""
        if self.interval is None:
            raise ValueError('Property "align_index" is not set on a null variant')

        return (
            self.interval.df_segment
            .filter(pl.col('is_aligned'))
            .select(pl.col('align_index'))
        ).to_series().to_list()

    @property
    def dup(self) -> Optional[list[dict[str, str | int]]]:
        """Get a list of duplicated reference loci."""
        return list(self._dup_list) if self._dup_list is not None else None

    @property
    def seg(self) -> Optional[list[dict[str, str | int | bool]]]:
        """Get a list of templated reference loci."""
        if self.interval is None:
            return None

        return (
            self.interval.df_segment
            .filter(pl.col('is_aligned') & ~ pl.col('is_anchor'))
            .select(
                'chrom', 'pos', 'end',
                (pl.col('is_rev') ^ self.interval.is_rev).alias('is_rev'),
                'qry_id', 'qry_pos', 'qry_end'
            )
            .rows(named=True)
        )

    @property
    def variant_id(self) -> str:
        """Variant ID."""
        if self.is_null:
            return 'Null'

        return f'{self.chrom}-{self.vartype}-{self.pos}-{self.varlen}'

    @property
    def is_null(self) -> bool:
        """Determine if variant is a null call."""
        return (self.interval is None or not self._found_variant) and not self.is_patch

    @property
    def is_variant(self) -> bool:
        """Determine if a variant was found."""
        return self._found_variant

    @property
    def seq(self) -> str:
        """Variant sequence."""
        if self.is_null or self.is_patch:
            raise ValueError('Property "seq" is not set on a null or patch variant')

        if self._seq is None:

            if self.vartype == 'DEL':
                self._seq = self.caller_resources.cache_ref[
                    self.region_ref.chrom
                ][
                    self.region_ref.pos:self.region_ref.end
                ]

            else:

                self._seq = self.caller_resources.cache_qry[
                    self.region_qry.chrom
                ][
                    self.region_qry.pos:self.region_qry.end
                ]

                if self.qry_rev:
                    self._seq = str(Bio.Seq.Seq(self._seq).reverse_complement())

        return self._seq

    @property
    def hom_ref(self) -> dict[str, int]:
        """Reference homology."""
        hom = perfect_homology(
            seq_var=self.seq,
            seq_ref=self.caller_resources.cache_ref[self.region_ref.chrom],
            pos=self.region_ref.pos,
            end=self.region_ref.end,
            is_rev=False,
        )

        return {'up': hom[0], 'dn': hom[1]}

    @property
    def hom_qry(self) -> dict[str, int]:
        """Query homology."""
        hom = perfect_homology(
            seq_var=self.seq,
            seq_ref=self.caller_resources.cache_qry[self.region_qry.chrom],
            pos=self.region_qry.pos,
            end=self.region_qry.end,
            is_rev=self.qry_rev,
        )

        return {'up': hom[0], 'dn': hom[1]}

    def __repr__(self) -> str:
        """Get a string representation of a variant object."""
        state = 'NULL' if self.is_null else (
            'INCOMPLETE' if not self._found_variant else (
                'PATCH' if self.is_patch else 'FOUND'
            )
        )
        return (
            f'{type(self).__name__}('
            f'state={state} '
            f'vartype={self.vartype}, '
            f'ref={self.region_ref}, '
            f'qry={self.region_qry}, qry_rev={self.qry_rev if not (self.is_null or self.is_patch) else None}, '
            f'score={self.var_score}, filter=[{", ".join(self.filter)}])'
        )

    def complete_anno(self) -> None:
        """Complete annotations on this variant call setting all fields necessary to call row().

        Initial variant calls are not completed so they can be prioritized before spending the CPU cycles and IO time
        pulling sequences to complete annotations.
        """
        if self._is_complete_anno or self.is_null or self.is_patch:
            return

        # Variant type implementation
        self._complete_anno_variant()
        self._is_complete_anno = True

    @classmethod
    def row_set(
            cls,
            add_set: Optional[set[str]] = None
    ) -> set[str]:
        """Get a set of row names for this variant class.

        :param add_set: Optional set of additional row names to include.

        :returns: Set of row names for this variant class.
        """
        return (
            {
                'chrom', 'pos', 'end',
                'vartype', 'varlen', 'filter',
                'qry_id', 'qry_pos', 'qry_end', 'qry_rev',
                'call_source', 'var_score', 'align_index',
                'hom_ref', 'hom_qry', 'seq',
            } |
            cls._row_set() |
            (add_set if add_set is not None else set())
        )

    def row(
            self,
            add_set: Optional[set[str]] = None
    ) -> dict[str, Any]:
        """Get a dict of values representing a variant call record (not including "id").

        The order of records is undefined, the final table should be ordered by the variant schema. The "id" column
        should be set on the final table, it is not computed by the variant objects.

        :param add_set: Optional set of additional row names to include.

        :returns: Variant call record as a dict.
        """
        self.complete_anno()

        if self.is_null:
            raise ValueError(f'Cannot get row for null variant: {self}')

        return {k: getattr(self, k) for k in self.row_set(add_set)}

    #
    # To be overridden
    #

    def _complete_anno_variant(self) -> None:  # noqa: B027
        """Complete annotations. To be implemented by the variant subclass."""
        pass

    @classmethod
    def _row_set(cls) -> set[str]:
        """Get a set of row names for this variant call."""
        return set()


# TODO: correct untemplated insertion breakpoints for homology at breakpoints
#
# Find untemplated insertion
#
# Trim reference if ends overlap around an untemplated insertion. This is likely homology at the breakpoint
# where the homologous sequence at each end of the insertion was aligned to the reference.
#
#                             [    HOM REGION    ]
#     Ref: --------->--------->--------->--------->--------->--------->--------->
# Align 1: --------->--------->--------->--------->
# Align 2:                     --------->--------->--------->--------->--------->
#
# Alignments 1 & 2 skip query sequence between them (untemplated insertion). This does not occur for strict templated
# insertions because query trimming will remove redundancy.
class InsertionVariant(Variant):
    """Insertion variant call."""

    # Simple insertion (INS = unaligned or aligned to another site)
    # INS:                           -------------------
    #                                        ||
    # Qry: --->--------->---------->---------> --------->-------->--------->--------->-----
    # Ref: --------------------------------------------------------------------------------

    def __init__(
            self,
            interval: AnchoredInterval,
            caller_resources: CallerResources,
            var_region_kde: VarRegionKde,
            _try_td: bool = False,
    ) -> None:
        """Create variant call."""
        Variant.__init__(self, interval, caller_resources, var_region_kde)
        self.vartype = 'INS'

        if _try_td:  # Called by TandemDuplicationVariant constructor, do not try to resolve INS variant
            return

        # Return immediately to leave the variant call a Null type
        if self.interval.seg_n != 1 or interval.len_qry == 0:
            return

        # Reference gap penalty: Penalize insertions for overlaps or deletions in the reference at the INS breakpoint
        len_ref = abs(self.interval.len_ref)

        ref_overlap = len_ref if self.interval.len_ref < 0 else 0
        self.varlen = len(interval.region_qry) + ref_overlap

        self.var_score = (
            caller_resources.score_model.gap(self.varlen) +
            caller_resources.score_model.gap(abs(len_ref)) * caller_resources.pav_params.lg_off_gap_mult +
            caller_resources.score_model.gap(ref_overlap)
        )

        self.region_ref = Region(interval.region_ref.chrom, interval.region_ref.pos, interval.region_ref.pos + 1)

        self.resolved_templ = interval.qry_aligned_pass_prop >= caller_resources.pav_params.lg_cpx_min_aligned_prop

        self._found_variant = True

    def _complete_anno_variant(self) -> None:
        """Complete annotations."""
        pass

    @classmethod
    def _row_set(cls) -> set[str]:
        """Get a set of row names for this variant call."""
        return {'varsubtype', 'dup'}


class TandemDuplicationVariant(InsertionVariant):
    """Tandem duplication variant call."""

    # Tandem duplication (TD)
    # Repeats:                                [> REP >]            [> REP >]
    # Qry 1:    --------->--------->--------->--------->--------->--------->
    # Qry 2:                                  --------->--------->--------->--------->--------->--------->
    #
    # Directly-oriented repeats may mediated by tandem repeats. Look at alignment-trimming in three ways:
    # * Qry trimming: Identifies TD if redundant query bases are removed, but queries still overlap
    # * Qry & Ref trimming: Find a breakpoint for an insertion call.
    # * No trimming: Identify the repeats at the ends of the TD.
    #
    # The resulting call is an INS call with a breakpoint placed in a similar location if the TD was called as an
    # insertion event in the CIGAR string. The variant is annotated with the DUP locus, and the sequence of both copies
    # is

    def __init__(
            self,
            interval: AnchoredInterval,
            caller_resources: CallerResources,
            var_region_kde: VarRegionKde,
    ) -> None:
        """Create variant call."""
        InsertionVariant.__init__(self, interval, caller_resources, var_region_kde, _try_td=True)

        if (
            self.interval.seg_n != 0 or self.interval.len_ref >= 0 or
            not self.interval.df_segment[0, 'align_index'] in self.caller_resources.qryref_index_set or
            not self.interval.df_segment[-1, 'align_index'] in self.caller_resources.qryref_index_set
        ):
            return

        df_qryref = (
            self.caller_resources.df_align_qryref
            .join(
                (
                    self.interval.df_segment
                    .filter('is_aligned')
                    .select('align_index')
                    .lazy()
                ),
                on='align_index',
                how='inner',
                maintain_order='right'
            )
            .select(['align_index', 'pos', 'end', 'qry_pos', 'qry_end'])
            .collect()
        )

        # Determine left-most breakpoint using alignment trimming for homology
        if self.interval.is_rev:
            qry_pos = df_qryref[-1, 'qry_end']
            qry_end = df_qryref[0, 'qry_pos']
        else:
            qry_pos = df_qryref[0, 'qry_end']
            qry_end = df_qryref[-1, 'qry_pos']

        self.varlen = qry_end - qry_pos

        if self.varlen <= 0:
            return

        self.region_ref = Region(self.interval.chrom, df_qryref[0, 'end'] - 1, df_qryref[0, 'end'])

        self.var_score = caller_resources.score_model.gap(self.varlen)

        self.varsubtype = 'TD'

        self.region_qry = Region(interval.region_qry.chrom, qry_pos, qry_end)

        self._dup_list.append(self.interval.region_ref.as_dict())

        self.resolved_templ = interval.qry_aligned_pass_prop >= caller_resources.pav_params.lg_cpx_min_aligned_prop

        self._found_variant = True


class DeletionVariant(Variant):
    """Deletion variant call."""

    # Simple deletion
    # Qry: ---->---------->--------->                  --------->-------->--------->-------
    # Ref: --------------------------------------------------------------------------------

    def __init__(
            self,
            interval: AnchoredInterval,
            caller_resources: CallerResources,
            var_region_kde: VarRegionKde,
    ) -> None:
        """Create variant call."""
        Variant.__init__(self, interval, caller_resources, var_region_kde)
        self.vartype = 'DEL'

        # Return immediately to leave the variant call a Null type
        if interval.len_ref <= 0:
            return

        self.region_ref = interval.region_ref
        self.region_qry = interval.region_qry

        self.varlen = len(self.interval.region_ref)

        self.var_score = (
            caller_resources.score_model.gap(self.varlen) +
            caller_resources.score_model.gap(abs(self.interval.len_qry)) *
            caller_resources.pav_params.lg_off_gap_mult
        )

        self.resolved_templ = True
        self._found_variant = True

    @classmethod
    def _row_set(cls) -> set[str]:
        """Get a set of row names for this variant call."""
        return {'varsubtype', 'dup'}  # Same as INS, will be placed in the same table


class InversionVariant(Variant):
    """Balanced inversion variant call."""

    # Note: Breakpoints may not be placed consistently in inverted repeats on each end, the aligner makes an
    # alignment decision independently for each breakpoint. Therefore, the inverted segment may not align to the
    # reference gap between the repeats. An alignment penalty should be applied to discourage inversion calls
    # that do not fit the classic model (an inverted alignment in a reference gap), but not penalize alignment
    # this common alignment artifact.
    #
    #            [ > >  Repeat  > > ]                                        [ < <  Repeat  < < ]
    # Flank: --->--------->------->                                                          --->----
    #   INV:       <---------<---------<---------<---------<---------<---------
    # Trace:  NML    |   DUP      |                  INV                       |    DEL      |   NML
    #
    # The pattern above is apparent in trimmed alignments (by query trimming). The un-trimmed alignment will
    # typically align through both repeats:
    #
    #            [ > >  Repeat  > > ]                                        [ < <  Repeat  < < ]
    # Flank: --->--------->------->--                                        --------->------->------
    #   INV:     --<---------<---------<---------<---------<---------<---------<---------<-------
    # Trace:  NML    |   DUP      |                  INV                       |    DEL      |   NML
    #
    # These two sources of information are combined. The qry-trimmed alignment is used to score the inversion after
    # redundant alignments are removed so complex patterns can be penalized appropriately. The untrimmed alignment
    # is used for setting inner- and outer-breakpoint locations.

    def __init__(
            self,
            interval: AnchoredInterval,
            caller_resources: CallerResources,
            var_region_kde: VarRegionKde,
    ) -> None:
        """Create variant call."""
        Variant.__init__(self, interval, caller_resources, var_region_kde)
        self.vartype = 'INV'

        self.region_ref_outer = None  # Also a sentinel for a successful call when trying multiple approaches
        self.region_qry_outer = None
        self.size_gap = 0

        # Base inversion checks
        if self.interval.len_ref <= 0 or self.interval.len_qry <= 0:
            # Balanced inversions consume ref & query bases
            return

        if min(self.interval.len_ref, self.interval.len_qry) / max(self.interval.len_ref, self.interval.len_qry) < 0.5:
            # Neither ref or query region may be 2x greater than the other
            return

        df_int = (
            interval.df_segment
            .filter(~ pl.col('is_anchor'))
            .with_columns(
                (
                    pl.col('is_aligned') &
                    (pl.col('pos') < self.interval.region_ref.end) &
                    (pl.col('end') > self.interval.region_ref.pos)
                ).alias('is_prox')
            )
            .with_columns(
                (~ pl.col('is_prox') & pl.col('is_aligned')).alias('is_dist')
            )
        )

        # Try inversion by untrimmed alignment
        if (
                self.interval.seg_n_aligned == 1
                and df_int.filter('is_aligned').select((pl.col('is_rev') != self.interval.is_rev).all()).item()
                and df_int.filter('is_aligned').select((pl.col('chrom') == self.interval.chrom).all()).item()
        ):
            # Get a table of pre-trimmed alignments in the same order as the aligned segments
            df_align_none = (
                caller_resources.df_align_none
                .select(['align_index', 'pos', 'end', 'qry_pos', 'qry_end'])
                .join(
                    (
                        self.interval.df_segment.filter('is_aligned')
                        .lazy()
                        .select(['align_index'])
                    ),
                    on='align_index',
                    how='inner',
                    maintain_order='right'
                )
                .collect()
            )

            ref_pos_outer = df_align_none[1, 'pos']
            ref_end_outer = df_align_none[1, 'end']

            ref_pos_inner = df_align_none[0, 'end']
            ref_end_inner = df_align_none[2, 'pos']

            if not interval.is_rev:
                qry_pos_outer = df_align_none[1, 'qry_pos']
                qry_end_outer = df_align_none[1, 'qry_end']

                qry_pos_inner = df_align_none[0, 'qry_end']
                qry_end_inner = df_align_none[2, 'qry_pos']
            else:
                qry_pos_outer = df_align_none[1, 'qry_pos']
                qry_end_outer = df_align_none[1, 'qry_end']

                qry_pos_inner = df_align_none[2, 'qry_end']
                qry_end_inner = df_align_none[0, 'qry_pos']

            if (
                (ref_pos_outer <= ref_pos_inner) and
                (ref_end_outer >= ref_end_inner) and
                (ref_pos_inner <= ref_end_inner) and
                (qry_pos_outer <= qry_pos_inner) and
                (qry_end_outer >= qry_end_inner) and
                (qry_pos_inner <= qry_end_inner)
            ):

                self.region_ref_outer = Region(interval.chrom, ref_pos_outer, ref_end_outer)
                self.region_ref = Region(interval.chrom, ref_pos_inner, ref_end_inner)

                self.region_qry_outer = Region(interval.qry_id, qry_pos_outer, qry_end_outer)
                self.region_qry = Region(interval.qry_id, qry_pos_inner, qry_end_inner)

                self.align_gap = 0
                self.resolved_templ = True

                self.res_type = 'ALIGN'

        # Try call by KDE
        if self.region_ref_outer is None:

            # Pre-checks before trying inversion by KDE
            if df_int.filter('is_dist').select('len_qry').sum().item() > self.interval.len_qry * 0.1:
                # No more than 10% aligned outside of the inversion site (allow for small alignments)
                return

            if (
                not var_region_kde.try_inv
                or df_int.filter('is_prox').select(
                    (
                        pl.col('len_qry').filter(pl.col('is_rev') == interval.is_rev).sum()
                        > pl.col('len_qry').filter(pl.col('is_rev') != interval.is_rev).sum()
                    )
                ).item()
            ):
                # Inverted by KDE, or Inverted segment lengths should outweigh the
                # non-inverted segments if the inversion is called by alignment.
                return

            # Call variant by KDE
            self.region_ref = self.interval.region_ref
            self.region_qry = self.interval.region_qry

            self.region_ref_outer = self.region_ref
            self.region_qry_outer = self.region_qry

            self.align_gap = np.abs(len(self.region_ref) - len(self.region_qry))
            self.resolved_templ = (
                    self.interval.qry_aligned_pass_prop >=
                    caller_resources.pav_params.lg_cpx_min_aligned_prop
            )

            self.res_type = 'KDE'

        # Call variant
        if self.region_ref_outer is None:
            # No variant by any method
            return

        self.varlen = len(self.region_ref)

        self.var_score = (
            # Penalize reference gap and inverted alignment diff
            caller_resources.score_model.gap(self.align_gap) +
            # Penalize template switches
            caller_resources.score_model.template_switch() * 2
        )

        self.outer_ref = self.region_ref_outer.as_dict()
        self.outer_qry = self.region_qry_outer.as_dict()

        self._found_variant = True
        return

    @classmethod
    def _row_set(cls) -> set[str]:
        """Get a set of row names for this variant call."""
        return {'outer_ref', 'outer_qry', 'align_gap'}


class ComplexVariant(Variant):
    """Complex variant call."""

    def __init__(
            self,
            interval: AnchoredInterval,
            caller_resources: CallerResources,
            var_region_kde: VarRegionKde,
    ) -> None:
        """Create variant call."""
        Variant.__init__(self, interval, caller_resources, var_region_kde)

        # Null variant (for creating table headers when no variants are found)
        if interval.len_qry <= 0:
            return

        # Set base properties
        self.vartype = 'CPX'
        self.region_ref = interval.region_ref
        self.region_qry = interval.region_qry
        self.varlen = len(interval.region_qry)

        # Get reference trace
        self.df_ref_trace = get_ref_trace(
            interval=self.interval,
            df_ref_fai=self.caller_resources.df_ref_fai
        )

        # Compute variant score
        self.var_score = (
            interval.segment_transition_score +
            (
                self.df_ref_trace
                .filter(pl.col('type') == 'DEL')
                .select((pl.col('end') - pl.col('pos')).map_elements(
                    caller_resources.score_model.gap, return_dtype=pl.Float32
                ).sum())
                .item()
            )
        )

        # Set in complete_anno
        self.struct_qry = None
        self.struct_ref = None

        self.resolved_templ = interval.qry_aligned_pass_prop >= caller_resources.pav_params.lg_cpx_min_aligned_prop

        if self.interval.len_ref < 0:
            self._dup_list.append(self.interval.region_ref.as_dict())

        self._found_variant = True

    def _complete_anno_variant(self):
        """Complete variant annotations."""
        # Get smoothed segment table
        if self.caller_resources.pav_params.lg_smooth_segments > 0.0:
            self.ref_trace_smooth = smooth_ref_trace(
                df_ref_trace=self.df_ref_trace,
                varlen=self.varlen,
                smooth_factor=self.caller_resources.pav_params.lg_smooth_segments
            )
            ref_trace_smooth = self.ref_trace_smooth
        else:
            self.ref_trace_smooth = None
            ref_trace_smooth = self.df_ref_trace

        self.trace_qry = qry_trace_str(self.interval.df_segment, self.is_pass)
        self.trace_ref = ref_trace_str(self.df_ref_trace, with_len=True)
        self.varsubtype = ref_trace_str(ref_trace_smooth, with_len=False)

    @classmethod
    def _row_set(cls) -> set[str]:
        """Get a set of row names for this variant call."""
        return {'varsubtype', 'seg_n', 'trace_qry', 'trace_ref', 'seg', 'dup'}


class NullVariant(Variant):
    """Represents no variant call."""

    def __init__(
            self,
            start_index: int,
            end_index: int,
    ) -> None:
        """Create variant call."""
        Variant.__init__(
            self, None, None, None,
            is_null=True, start_index=start_index, end_index=end_index
        )
        self.vartype = 'NULL'

    @classmethod
    def _row_set(cls) -> set[str]:
        """No row sets, raisse an error."""
        raise ValueError('Null variant has no row')


class PatchVariant(Variant):
    """Patch variant call for spanning segments with no variant.

    Represents no-variant call around alignment errors that do not represent real variation. For example, alignment
    artifacts around inverted repeats often break the alignment into pieces, but inspection of the pieces finds that
    they are largely contiguous with the reference. This patch variant type has a score of 0 and is treated as a bridge
    from one alignment anchor to another by graph traversal and prevents it from being pushed down non-optimal paths.
    """

    def __init__(self, start_index: int, end_index: int) -> None:
        """Start and end indices of the patch variant."""
        Variant.__init__(
            self, None, None, None,
            is_patch=True, start_index=start_index, end_index=end_index
        )
        self.vartype = 'PATCH'

        self.var_score = 0.0
        self.is_patch = True

    @classmethod
    def _row_set(cls) -> set[str]:
        """No row sets, raisse an error."""
        raise RuntimeError('Null variant does not have a row')


class DevVariant(Variant):
    """A convenience class for development purposes. Not used by PAV."""

    def __init__(
            self,
            interval: AnchoredInterval,
            caller_resources: CallerResources,
            var_region_kde: VarRegionKde,
    ) -> None:
        """Create variant call."""
        Variant.__init__(self, interval, caller_resources, var_region_kde)

        self.vartype = 'DEV'

    @classmethod
    def _row_set(cls) -> set[str]:
        """No row sets, raisse an error."""
        raise ValueError('Dev variant does not have a row')
