"""K-mer density estimation."""

__all__ = [
    'SAMPLE_INDEX_CHUNK_SIZE',
    'Kde',
    'KdeTruncNorm',
    'rl_encoder',
]

from abc import ABC, abstractmethod
import importlib.util
import inspect
import polars as pl
from typing import Optional

import numpy as np
from scipy.stats import truncnorm

from . import const


SAMPLE_INDEX_CHUNK_SIZE = 400


class Kde(ABC):
    """
    Kernel Density Estimation (KDE).

    Attributes:
        band_bound: Bandwidth boundary. Number of positions in the truncated normal vector for truncated-normal KDE.
    """

    band_bound: Optional[int]

    def __init__(self):
        """Initialize KDE."""
        self.band_bound = None

    @abstractmethod
    def __call__(self, x, n=None):
        """
        Get the density of an n-length vector with ones at x positions and zeros elsewhere.

        :param x: Location of 1's in the array (if `n` is defined) or an array to directly convolve with the density
            kernel and `n` becomes the length of this array.
        :param n: Length of the array. If `None`, `x` is the full-length array to convolve.

        :returns: A density vector of length `n`.
        """
        pass


class KdeTruncNorm(Kde):
    """Kernel Density Estimation (KDE) with a truncated-normal distribution.

    Uses FFT convolution to solve density.
    """

    def __init__(
            self,
            bandwidth=const.INV_KDE_BANDWIDTH,
            trunc_z=const.INV_KDE_TRUNC_Z,
            conv=const.INV_KDE_FUNC
    ) -> None:
        """Initialize truncated-normal KDE.

        :param bandwidth: Bandwidth of the truncated-normal distribution.
        :param trunc_z: Truncation SD of the truncated-normal distribution.
        :param conv: Convolution method.
        """
        super().__init__()

        # Check parameters
        if bandwidth <= 0:
            raise ValueError(f'Bandwidth must be > 0: {bandwidth}')

        if trunc_z <= 0:
            raise ValueError(f'Truncation SD must be > 0: {trunc_z}')

        self.bandwidth = float(bandwidth)
        self.band_sd = float(trunc_z)

        # Get convolution method
        if isinstance(conv, str):
            conv_lower = conv.strip().lower()

            is_auto = conv_lower == 'auto'

            self.conv_method = None

            if is_auto:
                conv_lower = 'fft'

            if conv_lower == 'fft' and self.conv_method is None:

                spec = importlib.util.find_spec('scipy.signal')

                if spec is None:
                    if not is_auto:
                        raise RuntimeError(
                            f'Error initializing KdeTruncNorm: Missing package for KDE convolution method {conv}: '
                            f'scipy.signal'
                        )
                    else:
                        conv_lower = 'conv'  # Try next

                import scipy.signal
                self.conv_method = scipy.signal.fftconvolve

            if conv_lower == 'conv' and self.conv_method is None:
                self.conv_method = np.convolve

            if self.conv_method is None:
                if is_auto:
                    raise RuntimeError(
                        'Error initializing KdeTruncNorm: Could not automatically resolve convolution method'
                    )

                raise RuntimeError(f'Error initializing KdeTruncNorm: Unknown convolution method: {conv}')

        elif callable(conv):
            self.conv_method = conv

        # Inspect convolution method
        n_arg = len(inspect.getfullargspec(self.conv_method).args)
        n_default = len(inspect.getfullargspec(self.conv_method).defaults)

        if n_arg < 2:
            raise RuntimeError(
                f'Convolution method does not take at least 2 arguments (n = {n_arg}): {self.conv_method}'
            )

        if n_arg - n_default > 2:
            raise RuntimeError(
                f'Convolution method requires more than 2 arguments (n = {n_arg - n_default} without default values): '
                f'{self.conv_method}'
            )

        # Set normal vector
        tnorm = truncnorm(-trunc_z, trunc_z)  # Truncate at Z = band_sd

        self.band_bound = int(np.ceil(trunc_z * bandwidth))  # number of positions in the truncated normal vector

        self.v_tnorm = tnorm.pdf(
            # Range from -band_sd to +band_sd after accounting for bandwidth
            np.arange(-self.band_bound, self.band_bound + 1) / self.bandwidth
        )  # Pre-compute PDF at positions

        # Vector sums to 1 (density 1 is a position with ones from -band_sd to band_sd)
        self.v_tnorm = self.v_tnorm / self.v_tnorm.sum()

    def __call__(self, x, n=None):
        """Get the density of an n-length vector with ones at x positions and zeros elsewhere.

        :param x: Location of 1's in the array (if `n` is defined) or an array to directly convolve with the density
            kernel and `n` becomes the length of this array.
        :param n: Length of the array. If `None`, `x` is the full-length array to convolve.

        :returns: A density vector of length `n`.
        """
        if n is not None:
            v_state = np.zeros(n)

            for v in x:
                v_state[v] = 1
        else:
            if not isinstance(x, np.ndarray):
                v_state = np.array(x)
            else:
                v_state = x

        y = self.conv_method(v_state, self.v_tnorm)

        return y[self.band_bound:-self.band_bound]


def rl_encoder(
        df: pl.DataFrame
):
    """Take a full table of states and KDE estimates per site and generate a table of contiguous states.

    Recall that the state table has columns removed.

    The table returned has these fields:

        * state: State of a contiguous region.
        * index_kde: Position in the k-mer table.
        * LEN_KDE: Length in the KDE table.
        * POS_QRY: Relative position in the query.
        * LEN_QRY: Length including excluded k-mers
        * MAX_GAIN: Maximum difference between the highest value and the middle (next highest) value in the KDE values.

    :param df: Dataframe of states.

    :returns: DataFrame with STATE, POS_KDE, LEN_KDE, POS_QRY, LEN_QRY, and MAX_GAIN columns.
    """
    return (
        df
        .lazy()
        .with_columns(
            pl.col('state').rle_id().alias('_state_rle'),
            pl.int_range(0, pl.count()).alias('_index')
        )
        .group_by('_state_rle')
        .agg([
            pl.col('state').first().alias('state'),
            pl.col('_index').first().alias('index_kde'),
            pl.count().alias('len_kde'),
            pl.col('index').first().alias('pos_qry'),
            ((pl.col('index').last() + 1) - pl.col('index').first()).alias('len_qry'),
            (
                pl.max_horizontal(['kde_fwd', 'kde_fwdrev', 'kde_rev']) * 2 - (
                    pl.sum_horizontal(['kde_fwd', 'kde_fwdrev', 'kde_rev']) -
                    pl.min_horizontal(['kde_fwd', 'kde_fwdrev', 'kde_rev'])
                )
            ).max().alias('max_gain')
        ])
        .drop('_state_rle')
        .collect()
    )
