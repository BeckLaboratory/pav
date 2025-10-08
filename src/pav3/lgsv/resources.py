"""Caller resources object passed among lgsv modules."""

__all__ = [
    'CallerResources',
]

from dataclasses import dataclass, field
import polars as pl
from typing import Optional, TextIO, Any
import sys

import agglovar

from ..align.score import ScoreModel, get_score_model
from ..align.lift import AlignLift
from ..io import NullWriter
from ..params import PavParams
from ..region import Region
from ..seq import LRUSequenceCache
from ..kde import Kde, KdeTruncNorm


@dataclass(repr=False)
class CallerResources(object):
    """Container of resources needed by routines resolving large variants.

    :ivar df_align_qry: Alignment table (QRY trimmed)
    :ivar df_align_qryref: Alignment table (QRY & REF trimmed)
    :ivar df_align_none: Alignment table (No trimming)
    :ivar ref_fa_filename: Reference FASTA filename.
    :ivar qry_fa_filename: Query FASTA filename.
    :ivar ref_fai_filename: Reference FAI filename.
    :ivar qry_fai_filename: Query FAI filename.
    :ivar score_model: Alignment score model.
    :ivar k_util: K-mer utility.
    :ivar inv_params: Inversion parameters.
    :ivar kde_model: KDE model.
    :ivar log_file: Write log to this file. If None, discard messages.
    :ivar verbose: Print verbose messages if True.
    :ivar pav_params: PAV pipeline parameters.
    :ivar ref_fai: Reference FAI table.
    :ivar qry_fai: Query FAI table.
    :ivar align_lift: Object for lifting alignment coordinates between query and reference through the alignment.
    :ivar cache_qry: Query sequence cache.
    :ivar cache_ref: Reference sequence cache.
    :ivar qry_index_set: Set of align_index values that were not removed by QRY trimming.
    :ivar qryref_index_set: Set of align_index values that were not removed by QRY and REF trimming.
    """

    df_align_qry: pl.LazyFrame
    df_align_qryref: pl.LazyFrame
    df_align_none: pl.LazyFrame
    ref_fa_filename: str
    qry_fa_filename: str
    ref_fai_filename: str = field(default=None)
    qry_fai_filename: str = field(default=None)
    score_model: Optional[ScoreModel] = field(default_factory=get_score_model)
    k_util: Optional[agglovar.kmer.util.KmerUtil] = None
    inv_params: Optional[dict[str, Any]] = field(default_factory=dict)
    kde_model: Optional[Kde] = field(default_factory=lambda: KdeTruncNorm())
    log_file: Optional[TextIO | NullWriter] = sys.stdout
    verbose: bool = True
    pav_params: Optional[PavParams] = field(
        default_factory=lambda: PavParams(), repr=False
    )
    align_lift: AlignLift = field(init=False)
    df_ref_fai: pl.DataFrame = field(init=False)
    df_qry_fai: pl.DataFrame = field(init=False)
    cache_qry: LRUSequenceCache = field(init=False)
    cache_ref: LRUSequenceCache = field(init=False)
    qry_index_set: set[int] = field(init=False)
    qryref_index_set: set[int] = field(init=False)

    def __post_init__(self) -> None:
        """Finalize object initialization."""
        if isinstance(self.df_align_qry, pl.DataFrame):
            self.df_align_qry = self.df_align_qry.lazy()

        if isinstance(self.df_align_qryref, pl.DataFrame):
            self.df_align_qryref = self.df_align_qryref.lazy()

        if isinstance(self.df_align_none, pl.DataFrame):
            self.df_align_none = self.df_align_none.lazy()

        self.qryref_index_set = set(
            self.df_align_qryref.select('align_index')
            .collect().to_series().to_list()
        )

        self.qry_index_set = set(
            self.df_align_qry.select('align_index')
            .collect().to_series().to_list()
        )

        self.df_align_qry = (
            self.df_align_qry
            .with_columns(
                pl.col('align_index').is_in(self.qryref_index_set).alias('in_qryref')
            )
            .collect()
            .lazy()
        )

        self.df_align_none = (
            self.df_align_none
            .with_columns(
                pl.col('align_index').is_in(self.qry_index_set).alias('in_qry'),
                pl.col('align_index').is_in(self.qryref_index_set).alias('in_qryref')
            )
            .collect()
            .lazy()
        )

        if self.k_util is None:
            self.k_util = agglovar.kmer.util.KmerUtil(self.pav_params.inv_k_size)

        if self.ref_fai_filename is None:
            self.ref_fai_filename = self.ref_fa_filename + '.fai'

        if self.qry_fai_filename is None:
            self.qry_fai_filename = self.qry_fa_filename + '.fai'

        self.df_ref_fai = agglovar.fa.read_fai(self.ref_fai_filename)
        self.df_qry_fai = agglovar.fa.read_fai(self.qry_fai_filename)

        self.align_lift = AlignLift(self.df_align_qry.collect(), self.df_qry_fai)

        if self.log_file is None:
            self.log_file = NullWriter()

        self.cache_qry = LRUSequenceCache(fa_filename=self.qry_fa_filename, max_size=5)
        self.cache_ref = LRUSequenceCache(fa_filename=self.ref_fa_filename, max_size=5)

        self.cache_qry.open()
        self.cache_ref.open()

        self.inv_params = dict(self.inv_params)

        for key in list(self.inv_params.keys()):
            if key not in {
                'nc_ref', 'nc_qry', 'region_limit', 'init_expand', 'min_kmers',
                'max_ref_kmer_count', 'repeat_match_prop', 'min_inv_kmer_run', 'min_qry_ref_prop'
            }:
                del self.inv_params[key]

    def seq_ref(self, region: Region) -> str:
        """Extract reference sequence for a region."""
        return self.cache_ref_upper.get(region.chrom, region.pos, region.end)

    def __repr__(self) -> str:
        """Get a string representation of this object."""
        return f'{self.__class__.__name__}(0x{id(self):X})'
