"""
Global rules
"""

import gzip
import polars as pl

# Convert parquet to TSV in BED format
rule global_pq_to_table:
    input:
        pq='{prefix}/{filename}.bed.parquet'
    output:
        bed='{prefix}/{filename}.bed.gz'
    run:

        with gzip.open(output.bed, 'wb') as out_file:
            pl.scan_parquet(input.pq).sink_csv(out_file, separator='\t')
