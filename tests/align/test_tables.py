"""Round-trip tests for pav3.align.tables.{seq_to_align_ops,align_to_seq_ops}."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pysam
import pytest

from pav3.align import op
from pav3.align.tables import align_to_seq_ops, seq_to_align_ops


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_ROOT = PROJECT_ROOT / 'local' / 'test_data' / 'pav_short'
ALIGN_TABLE_GLOB = 'results/*/align/*/align_trim-none.parquet'
REF_FASTA = TEST_DATA_ROOT / 'ref_files' / 'hs1.fa.gz'
ASM_FASTA_TEMPLATE = 'asm_files/small1_{asm}_{hap}.fa.gz'

_INDEL_CODES = np.array([op.I, op.D])

# IUPAC complement table (covers ACGT/N plus standard ambiguity codes)
_REVCOMP_TABLE = bytes.maketrans(
    b'ACGTNRYWSKMBDHVacgtnrywskmbdhv',
    b'TGCANYRWSMKVHDBtgcanyrwsmkvhdb',
)


def _revcomp(seq: str) -> str:
    """Reverse-complement an ASCII nucleotide string."""
    return seq.encode('ascii').translate(_REVCOMP_TABLE)[::-1].decode('ascii')


def _discover_align_tables() -> list[tuple[str, str, Path]]:
    """Discover (asm, hap, path) tuples from the test data directory."""
    if not TEST_DATA_ROOT.is_dir():
        return []

    found: list[tuple[str, str, Path]] = []
    for path in sorted(TEST_DATA_ROOT.glob(ALIGN_TABLE_GLOB)):
        # path = <root>/results/<asm>/align/<hap>/align_trim-none.parquet
        hap = path.parent.name
        asm = path.parent.parent.parent.name
        found.append((asm, hap, path))
    return found


_TABLES = _discover_align_tables()
_PARAMS = (
    [
        pytest.param(asm, hap, path, id=f'{asm}-{hap}', marks=pytest.mark.sample(asm))
        for asm, hap, path in _TABLES
    ]
    if _TABLES
    else [pytest.param('', '', Path(), id='no-test-data')]
)


@pytest.fixture(scope='session')
def ref_fasta_handle():
    """Shared pysam FastaFile handle to the reference for all tests in the session."""
    if not REF_FASTA.is_file():
        pytest.skip(f'reference FASTA missing: {REF_FASTA}')
    fa = pysam.FastaFile(str(REF_FASTA))
    try:
        yield fa
    finally:
        fa.close()


def _indels_with_coords(op_arr: np.ndarray) -> np.ndarray:
    """Return I/D rows with reference and query coordinates attached.

    Strips clipping ops first because ``op_arr_add_coords`` rejects ops that
    advance neither ref nor qry.
    """
    non_clip = op_arr[~np.isin(op_arr[:, 0], list(op.CLIP_SET))]
    coords = op.op_arr_add_coords(non_clip, add_index=False)
    return coords[np.isin(coords[:, 0], _INDEL_CODES)]


@pytest.mark.requires_test_data
@pytest.mark.parametrize('asm,hap,path', _PARAMS)
def test_align_seq_op_round_trip(asm: str, hap: str, path: Path, ref_fasta_handle: pysam.FastaFile) -> None:
    """Each align_trim-none record round-trips =/X -> M -> =/X using real ref/qry sequences."""
    qry_path = TEST_DATA_ROOT / ASM_FASTA_TEMPLATE.format(asm=asm, hap=hap)
    if not qry_path.is_file():
        pytest.skip(f'assembly FASTA missing: {qry_path}')

    df = pl.read_parquet(path)

    with pysam.FastaFile(str(qry_path)) as qry_fa:
        for record_idx, row in enumerate(df.iter_rows(named=True)):
            orig_arr = op.row_to_arr(row['align_ops'])

            m_arr = seq_to_align_ops(orig_arr)

            orig_indels = _indels_with_coords(orig_arr)
            m_indels = _indels_with_coords(m_arr)
            assert np.array_equal(orig_indels, m_indels), (
                f'I/D positions differ after seq_to_align_ops '
                f'(asm={asm}, hap={hap}, record_idx={record_idx}, '
                f'align_index={row["align_index"]})'
            )

            ref_seq = ref_fasta_handle.fetch(row['chrom'], row['pos'], row['end'])
            qry_seq = qry_fa.fetch(row['qry_id'], row['qry_pos'], row['qry_end'])
            if row['is_rev']:
                qry_seq = _revcomp(qry_seq)

            recon_arr = align_to_seq_ops(m_arr, qry_seq, ref_seq)

            assert np.array_equal(recon_arr, orig_arr), (
                f'Round-trip differs from original '
                f'(asm={asm}, hap={hap}, record_idx={record_idx}, '
                f'align_index={row["align_index"]})'
            )
