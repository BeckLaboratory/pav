import numpy as np
import polars as pl

from pav3.align import op
from pav3.align.score import get_score_model


def test_score_align_table_returns_float32_for_large_values():
    score_model = get_score_model()

    n_match = 25_414_036
    op_arr = np.array([[op.EQ, n_match]], dtype=np.int64)

    df = pl.DataFrame(
        {
            'align_ops': [op.arr_to_row(op_arr)],
        }
    )

    score = score_model.score_align_table(df)

    assert score.dtype == pl.Float32
    assert score.len() == 1
    assert score.item() == np.float32(2.0 * n_match)


def test_var_score_expr_yields_float32():
    """Pin the Float64 -> Float32 cast pattern used for var_score in intra.py.

    Polars >=1.x raises SchemaError when ``map_elements`` returns a Python float
    (Float64) but ``return_dtype=pl.Float32`` is declared. The intra-haplotype
    caller declares Float64 then explicitly casts to Float32 to preserve the
    Float32 schema contract for ``var_score``.
    """
    score_model = get_score_model()

    op_lens = [1, 5, 50, 500]
    df = pl.DataFrame({'op_len': op_lens}, schema={'op_len': pl.Int64})

    var_score = (
        df
        .select(
            pl.col('op_len')
            .map_elements(score_model.gap, return_dtype=pl.Float64)
            .cast(pl.Float32)
            .alias('var_score')
        )
        .get_column('var_score')
    )

    assert var_score.dtype == pl.Float32
    assert var_score.len() == len(op_lens)

    expected = [np.float32(score_model.gap(n)) for n in op_lens]
    assert var_score.to_list() == expected
