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
