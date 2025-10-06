"""Region objects and associated functions."""

__all__ = [
    'Region',
    'region_from_string',
    'region_from_id',
]

from dataclasses import dataclass
from typing import Optional, Self

import polars as pl
import re


@dataclass(frozen=True)
class Region(object):
    """
    Represents a region (chromosome, pos, and end) in 0-based half-open coordinates (BED).

    Tracks orientation as `is_rev` for other code to use (not used by this object). By default, it assumes that `is_rev`
    is `True` if `pos` > `end`.

    If breakpoints are uncertain, minimum and maximum values can be set for pos and end.

    If the region is associated with alignment records, the alignment record index for pos and end can be also be
    tracked with this object.

    :ivar chrom: Chromosome or sequence name.
    :ivar pos: Start position (0-based).
    :ivar end: End position (0-based), exclusive.
    :ivar is_rev: Region is reverse-complemented if `True`.
    :ivar pos_align_index: Alignment index for pos.
    :ivar end_align_index: Alignment index for end.
    """

    chrom: str
    pos: int
    end: int
    is_rev: bool = False
    pos_align_index: Optional[int] = None
    end_align_index: Optional[int] = None

    def __post_init__(self):
        """Validate fields.

        :raises ValueError: If `pos` is greater than `end`.
        :raises ValueError: If any required fields are missing.
        :raises ValueError: If `pos` or `end` is negative.
        """
        if self.pos > self.end:
            raise ValueError(f'pos {self.pos} must not be greater than end ({self.end})')

        if self.pos < 0 or self.end < 0:
            raise ValueError(f'pos and end must be non-negative: pos={self.pos}, end={self.end}')

        if missing_vals := [key for key in ('chrom', 'pos', 'end', 'is_rev') if getattr(self, key) is None]:
            raise ValueError(f'Missing required fields: {", ".join(missing_vals)}')

    def __repr__(self):
        """Get coordinate string in 1-based closed coordinates (Samtools, UCSC browser)."""
        return f'{self.chrom}:{self.pos + 1}-{self.end}'

    def __len__(self) -> int:
        """Get the region length."""
        return self.end - self.pos

    def region_id(self) -> str:
        """Get a formatted region ID.

        :returns: Region ID formatted as "chrom-pos-RGN-len" where pos is 1-based coordinate, len is the region length,
            and "RGN" is a literal string.
        """
        return f'{self.chrom}-{self.pos + 1}-RGN-{self.end - self.pos}'

    def expand(
            self,
            expand_bp: int | float,
            min_pos: int = 0,
            max_end: Optional[int | pl.DataFrame | pl.LazyFrame] = None,
            shift: bool = True,
            balance: float = 0.5
    ) -> Self:
        """Expand this region in place by `expand_bp`.

        Do not expand beyond `min_pos` and `max_end`, if defined. If position limits are reached and `shift` is `True`,
        then set the boundary at the limit and expand in the free position up to its limit, if defined. The region is
        never expanded beyond `expand_bp`, but if limits are reached, it may expand less.

        Expand can be asymetrical if `balance` is defined and not 0.5 (must be between 0 and 1). `pos` is expanded by
        `int(expand_bp * balance)`, and `end` is expanded by `int(expand_bp * (1 - balance))`.

        :param expand_bp: Expand this number of bases. May be negative (untested).
        :param min_pos: Lower limit on the new position (default 0).
        :param max_end: Upper limit on the new end. This may be an integer defining the maximum end position or a
            Polars table (chrom is matched to column "chrom" or "qry_id", and max_end is derived from column "len").
        :param shift: If `True`, shift the expand window if one end exceeds a limit. Tries to preserve `expand_bp` while
            honoring the limits. If `
        :param balance: Shift `pos` by `int(expand_bp * balance)` and `end` by `int(expand_bp * (1 - balance))`. If
            `0.5`, both positions are shifted equally.

        :returns: Expanded region.

        :raises ValueError: If `balance` is not in range [0, 1].
        :raises ValueError: if max_end is not an integer, a Polars DataFrame or LazyFrame, or if there is an error
            getting the max end position from a table.
        """
        if not (0 <= balance <= 1):
            raise ValueError('balance must be in range [0, 1]: {}'.format(balance))

        # Get new positions
        expand_pos = int(expand_bp * balance)
        expand_end = max([0, int(expand_bp - expand_pos)])

        new_pos = self.pos - expand_pos
        new_end = self.end + expand_end

        # Shift pos
        if min_pos is not None and new_pos < min_pos:
            min_diff = min_pos - new_pos

            if shift:
                new_end += min_diff

            new_pos = min_pos

        # Shift end
        if max_end is not None:

            # Convert max_end to integer if it's a table
            if isinstance(max_end, pl.DataFrame):
                max_end = max_end.lazy()

            if isinstance(max_end, pl.LazyFrame):
                col_name = [col for col in max_end.collect_schema().names() if col in {'chrom', 'qry_id'}]

                if len(col_name) != 1:
                    raise ValueError(
                        f'Cannot expand with max_end as a table: Must have one of "chrom" or "qry_id": '
                        f'found="{", ".join(col_name)}"'
                    )

                if 'len' not in max_end.collect_schema().names():
                    raise ValueError('Cannot expand with max_end as a table: Missing "len" field')

                col_name = col_name[0]

                try:
                    max_end = int(max_end.filter(pl.col(col_name) == self.chrom).select(pl.col('len')).collect().item())
                except ValueError as e:
                    raise ValueError(
                        f'Failed retrieving max_end a table (Multiple rows or no rows found for "chrom"?): {e}'
                    ) from e

            else:
                try:
                    max_end = int(max_end)
                except ValueError as e:
                    raise ValueError(f'Failed casting max_end to int: {e}') from e

        if (max_diff := new_end - max_end) > 0:
            if shift:
                new_pos = max(new_pos - max_diff, min_pos)

            new_end = max_end

        # Check for over-contraction (if expand_bp was negative)
        if new_end < new_pos:
            new_end = new_pos = (new_end + new_pos) // 2

        # Assign new coordinates
        return Region(
            chrom=self.chrom,
            pos=new_pos,
            end=new_end,
            is_rev=self.is_rev,
            pos_align_index=self.pos_align_index,
            end_align_index=self.end_align_index
        )

    def contains(self, other: Self) -> bool:
        """Determine if this region contains another region.

        :param other: Other region.

        :returns: u`True` if `other` is contained within this region.
        """
        return other.chrom == self.chrom and other.pos >= self.pos and other.end <= self.end

    def is_contained(self, other: Self) -> bool:
        """Determine if this region is contained within another region.

        :param other: Other region.

        :returns: `True` if `other` contains this region.
        """
        return other.chrom == self.chrom and self.pos >= other.pos and self.end <= other.end

    def as_dict(self) -> dict[str, str | int]:
        """Get a dictionary of this region.

        Typically used for writing to a Polars table.

        :returns: Dict of region (chrom, pos, end).
        """
        return {
            'chrom': self.chrom,
            'pos': self.pos,
            'end': self.end,
        }

    def as_dict_with_rev(self) -> dict[str, str | int | bool]:
        """Get a dictionary of this region with reverse strand information.

        Typically used for writing to a Polars table.

        :returns: Dict of region (chrom, pos, end, is_rev).
        """
        return {
            'chrom': self.chrom,
            'pos': self.pos,
            'end': self.end,
            'is_rev': self.is_rev,
        }

    def as_qry_dict(self) -> dict[str, str | int]:
        """Get a dict with query coordinate keys.

        Typically used for writing to a Polars table.

        :returns: Dict of region (chrom, pos, end) with query coordinate keys (qry_id, qry_pos, qry_end).
        """
        return {
            'qry_id': self.chrom,
            'qry_pos': self.pos,
            'qry_end': self.end,
        }

    def __eq__(self, other: Self) -> bool:
        """Test equality on coordinates (chrom, pos, end).

        :param other: Other Region object.

        :returns: `True` if `other` is equal to `self`.
        """
        return (
            self.chrom == other.chrom and
            self.pos == other.pos and
            self.end == other.end
        )

    def __lt__(self, other: Self) -> bool:
        """Test relation on coordinates (chrom, pos, end).

        :param other: Other Region object.

        :returns: `True` if `other` is less than `self`.
        """
        return (self.chrom, self.pos, self.end) < (other.chrom, other.pos, other.end)

    def __add__(self, other: int | Self) -> Self:
        """Add to this region.

        :param other: Integer to add to this region (may be negative), or another Region object to add to this region
            where pos and end from both regions are added.

        :raises ValueError: If `other` is a Region object and `self.chrom` is not equal to `other.chrom`.
        :raises ValueError: If `other` is not an integer or Region object.
        """
        try:
            if isinstance(other, Region):
                if self.chrom != other.chrom:
                    raise ValueError('Cannot add regions with different chromosomes')

                return Region(
                    chrom=self.chrom,
                    pos=self.pos + other.pos,
                    end=self.end + other.end,
                    is_rev=self.is_rev,
                    pos_align_index=self.pos_align_index,
                    end_align_index=self.end_align_index
                )

            return Region(
                chrom=self.chrom,
                pos=int(self.pos + other),
                end=int(self.end + other),
                is_rev=self.is_rev,
                pos_align_index=self.pos_align_index,
                end_align_index=self.end_align_index
            )
        except ValueError as e:
            raise ValueError(f'Failed adding {other}: {e}') from e

    def __sub__(self, other: int | Self) -> Self:
        """Subtract from this region.

        :param other: Integer to subtract from this region (may be negative), or another Region object to add to this
            region where pos and end from both regions are subtracted.

        :returns: Expanded region.

        :raises ValueError: If `other` is a Region object and `self.chrom` is not equal to `other.chrom`.
        :raises ValueError: If `other` is not an integer or Region object.
        """
        try:
            if isinstance(other, Region):
                if self.chrom != other.chrom:
                    raise ValueError('Cannot subtract regions with different chromosomes')

                return Region(
                    chrom=self.chrom,
                    pos=self.pos - other.pos,
                    end=self.end - other.end,
                    is_rev=self.is_rev,
                    pos_align_index=self.pos_align_index,
                    end_align_index=self.end_align_index
                )

            return Region(
                chrom=self.chrom,
                pos=int(self.pos - other),
                end=int(self.end - other),
                is_rev=self.is_rev,
                pos_align_index=self.pos_align_index,
                end_align_index=self.end_align_index
            )
        except ValueError as e:
            raise ValueError(f'Failed subtracting {other}: {e}') from e


def region_from_string(
        region_str: str,
        is_rev: bool = False,
        base0half: bool = False,
        pos_align_index: Optional[int] = None,
        end_align_index: Optional[int] = None,
) -> Region:
    """Get a region object from a string (e.g. "chr1:123456-234567").

    :param region_str: Region string.
    :param is_rev: Region is reverse-complemented if `True`.
    :param base0half: If `False` (default), then string is in base-1 closed coordinates (the first base in the
        chromosome is 1, and the positions are inclusive). If `True`, expect BED coordinates (pos is 0-based instead
        of 1-based). All PAV-generated region strings are 1-based (False for this parameter), as are Samtools
        and IGV region strings. This should be False except in very specific cases.
    :param pos_align_index: Alignment index for pos.
    :param end_align_index: Alignment index for end.

    :returns: Region object.

    :raises ValueError: If `region_str` is not in the expected format (chrom:pos-end) or the region is invalid.
    """
    rgn_str_no_comma = region_str.replace(',', '')
    match_obj = re.search(r'^(.+):(\d+)-(\d+)$', rgn_str_no_comma)

    if match_obj is None:
        raise ValueError('Region is not in expected format (chrom:pos-end): {}'.format(region_str))

    pos = int(match_obj[2])
    end = int(match_obj[3])

    if not base0half:
        pos -= 1

    return Region(match_obj[1], pos, end, is_rev, pos_align_index, end_align_index)


def region_from_id(region_id: str) -> Region:
    """Translate a region ID (CHROM-POS-SVTYPE-LEN) to a Region object.

    :param region_id: Region ID.

    :returns: Region object.

    :raises RuntimeError: If `region_id` is not in the expected format (chrom-pos-svtype-len) or is not a valid region.
    """
    tok = region_id.split('-')

    if len(tok) != 4:
        raise RuntimeError('Unrecognized region ID: {}'.format(region_id))

    return Region(tok[0], int(tok[1]) - 1, int(tok[1]) - 1 + int(tok[3]))
