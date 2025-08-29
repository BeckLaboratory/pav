"""
Table schemas.
"""

import agglovar

import polars as pl

ALIGN = {
    'chrom': pl.String,
    'pos': pl.Int64,
    'end': pl.Int64,
    'align_index': pl.Int32,
    'filter': pl.List(pl.String),
    'qry_id': pl.String,
    'qry_pos': pl.Int64,
    'qry_end': pl.Int64,
    'qry_order': pl.Int32,
    'rg': pl.String,
    'mapq': pl.UInt8,
    'is_rev': pl.Boolean,
    'flags': pl.UInt16,
    'align_ops': pl.Struct({
        'op_code': pl.List(pl.UInt8),
        'op_len': pl.List(pl.Int32)
    })
}
"""dict[str, pl.DataType]: Schema of alignment tables excluding features (added by ailgn.feature)."""

VARIANT = {
    'chrom': agglovar.schema.VARIANT['chrom'],
    'pos': agglovar.schema.VARIANT['pos'],
    'end': agglovar.schema.VARIANT['end'],
    'id': agglovar.schema.VARIANT['id'],
    'vartype': agglovar.schema.VARIANT['vartype'],
    'varlen': agglovar.schema.VARIANT['varlen'],
    'ref': agglovar.schema.VARIANT['ref'],
    'alt': agglovar.schema.VARIANT['alt'],
    'align_index': pl.Int32,
    'filter': pl.List(pl.String),
    'qry_id': pl.String,
    'qry_pos': pl.Int64,
    'qry_end': pl.Int64,
    'call_source': pl.String,
    'seq': agglovar.schema.VARIANT['seq'],
}
"""dict[str, pl.DataType]: Schema of variant tables."""
