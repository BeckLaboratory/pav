"""Region kernel density estimator (KDE).

An object for creating a k-mer kernel density estimate (KDE) for a variant region showing forward- and reverse-oriented
segments a query region relative to a corresponding reference region. Inter-alignment variants use this KDE to determine
if alignments are aberrant and unlikely to contain a variant and to reveal variant structures.
"""

__all__ = [
    'VarRegionKde',
]

import polars as pl

from ..inv import get_state_table, test_kde, KDE_STATE_FWD, KDE_STATE_FWDREV, KDE_STATE_REV
from ..kde import rl_encoder

from .interval import AnchoredInterval
from .resources import CallerResources

_MAX_QRY_LEN_KDE = int(1e6)


class VarRegionKde:
    """
    Describes how a query and reference region match by k-mer density.

    :ivar df_kde: KDE table see :func:`pav3.inv.get_state_table`.
    :ivar df_rl: Run-length table see :func:`pav3.kde.rl_encoder`.
    :ivar try_inv: If True, try to call an inversion.
    :ivar try_var: If True, try to call a variant. False indicates aberrant alignment patterns were identified and that
        calling this region would likely produce a false variant call.
    """

    def __init__(self,
                 interval: AnchoredInterval,
                 caller_resources: CallerResources,
                 qry_ref_max_ident: float = 0.90,
                 max_seg_n: int = 50,
                 max_qry_len_kde: int = _MAX_QRY_LEN_KDE
                 ) -> None:
        """Create a variant region with KDE.

        Determines if the reference and query sequences should be tried for a variant.

        :param interval: Anchored interval over alignment fragments.
        :param caller_resources: Caller resources.
        :param qry_ref_max_ident: If the query and reference identity is this value or greater, do not try a variant
            call. This often occurs when alignments gap over divergent but similar sequences, but aligners leave the
            fragments unaligned. These look like substitutions (INS + DEL of similar size), but are not. These
            variants require better methods to match the unaligned sequences, which PAV does not currently have. For
            balanced inversions, the reference and query sequences must also be within this length
            (i.e. min / max >= qry_ref_max_ident).
        :param max_seg_n: If the number of segments is this value or greater, do not try a variant call. No limit if
            None.
        :param max_qry_len_kde: If the query sequence length is this value or greater, do not try a variant call. No
            limit if None.
        """
        # Default values
        self.df_kde = None
        self.df_rl = None

        self.try_inv = False
        self.try_var = False

        if (
                interval.len_ref <= 0 or (
                    min([interval.len_qry, interval.len_ref]) / max([interval.len_qry, interval.len_ref])
                ) < qry_ref_max_ident
        ):
            self.try_var = True
            return

        if max_seg_n is not None and interval.seg_n >= max_seg_n:
            return

        kmer_n = len(interval.region_qry) - caller_resources.k_util.k_size + 1  # Number of k-mers in the query sequence

        if kmer_n <= 0:
            self.try_var = True
            return

        # Build KDE
        region_ref_exp = interval.region_ref.expand(
            len(interval.region_ref) * 0.2,
            max_end=caller_resources.df_ref_fai,
            shift=False,
        )

        region_qry_exp = interval.region_qry.expand(
            len(interval.region_qry) * 0.2,
            max_end=caller_resources.df_qry_fai,
            shift=False,
        )

        if len(region_qry_exp) <= max_qry_len_kde:

            # Match complex sequence to the gap region
            self.df_kde = get_state_table(
                region_ref=region_ref_exp,
                region_qry=region_qry_exp,
                ref_fa_filename=caller_resources.ref_fa_filename,
                qry_fa_filename=caller_resources.qry_fa_filename,
                df_ref_fai=caller_resources.df_ref_fai,
                df_qry_fai=caller_resources.df_qry_fai,
                is_rev=interval.region_qry.is_rev,
                k_util=caller_resources.k_util,
                kde_model=caller_resources.kde_model,
                max_ref_kmer_count=caller_resources.pav_params.inv_max_ref_kmer_count,
                expand_bound=True,
            )

            if self.df_kde.height == 0:

                # No matching k-mers
                self.try_var = True
                return

            self.df_rl = rl_encoder(self.df_kde)

            # Subset the run-length (RL) table to those in just the variant region, cut the flanks.
            expand_l = interval.region_qry.pos - region_qry_exp.pos
            expand_r = region_qry_exp.end - interval.region_qry.end

            df_kde_noexp = (
                self.df_kde
                .filter(
                    (pl.col('index') > expand_l) & (pl.col('index') < len(region_qry_exp) - expand_r)
                )
            )

            if df_kde_noexp.shape[0] > 0:

                df_rl_sv = rl_encoder(df_kde_noexp)

                # Number of k-mers not dropped by KDE (found in both query and ref in either orientation)
                kmer_n_kde = df_rl_sv['len_kde'].sum()

                kde_len_rl = df_rl_sv.select(pl.col('len_kde').sum()).item()

                prop_fwd, prop_fwdrev, prop_rev = (
                    (
                        df_rl_sv
                        .select(
                            prop_fwd=(
                                    pl.col('len_kde').filter(pl.col('state') == KDE_STATE_FWD).sum() / kde_len_rl
                            ),
                            prop_fwdrev=(
                                    pl.col('len_kde').filter(pl.col('state') == KDE_STATE_FWDREV).sum() / kde_len_rl
                            ),
                            prop_rev=(
                                    pl.col('len_kde').filter(pl.col('state') == KDE_STATE_REV).sum() / kde_len_rl
                            ),
                        )
                    ).row(0)
                ) if kde_len_rl > 0 else (0, 0, 0)

                # If forward k-mers make most of the original sequence, do not try a variant.
                # The alignment dropped here, but sequences are similar. Additional methods
                # are needed to refine the missing alignments for these sequences,
                # which PAV currently does not have. This prevents false substitution CSVs
                # (INS + DEL).
                if prop_fwd > qry_ref_max_ident:
                    return

                # Try a non-INV variant call if tests to this point have passed.
                # Weed out misalignments around inverted repeats (human chrX)
                self.try_var = (
                    kmer_n_kde / kmer_n < 0.8  # Too many missing k-mers to be sure, try a variant
                ) or (
                    # Weed out misalignments around inv repeats (human chrX)
                    prop_fwdrev < 0.5 or prop_fwd + prop_fwdrev < 0.90
                )

                # Test states for inverted or reference states
                if self.df_rl.select((pl.col('state') == KDE_STATE_REV).any()).item():
                    p_binom = test_kde(self.df_rl)  # Test KDE for inverted state significance

                    self.try_inv = p_binom < 0.01 and (prop_rev + prop_fwdrev) >= 0.5

                # Check all segments, should belong to the gap region
                qry_exp_diff = interval.region_qry.pos - region_qry_exp.pos

                if self.try_inv:
                    qry_start = interval.df_segment[-1 if interval.is_rev else 0, 'qry_end']

                    for row in interval.df_segment.filter(~ pl.col('is_anchor')).iter_rows(named=True):
                        pos, end = sorted([abs(row['qry_pos'] - qry_start), abs(row['qry_end'] - row['qry_pos'])])

                        df_kde_seg = self.df_kde.filter(
                            (pl.col('index') >= (pos + qry_exp_diff)) & (pl.col('index') <= (end + qry_exp_diff))
                        )

                        if df_kde_seg.height / len(interval.region_qry) < 0.05:  # Skip diminuitive segments
                            continue

                        df_kde_seg = df_kde_seg.filter(pl.col('state').is_in({KDE_STATE_FWDREV, KDE_STATE_REV}))

                        prop_mer = df_kde_seg.height / (end - pos - caller_resources.k_util.k_size + 1)

                        if prop_mer < 0.5:
                            self.try_inv = False
            else:
                # Non-expanded KDE skipped
                self.try_var = True

        else:
            # KDE skipped
            self.try_var = True
