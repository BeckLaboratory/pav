"""General utility functions."""

__all__ = [
    'as_bool',
    'region_merge',
    'collapse_to_set'
]

from typing import Any, Callable, Iterable, Optional

import polars as pl


def as_bool(
        val: Any,
        fail_to_none: bool = False
) -> Optional[bool]:
    """Get a boolean value.

    True values: "true", "1", "yes", "t", "y", True, 1
    False values: "false", "0", "no", "f", "n", False, 0

    String values are case-insensitive.

    :param val: Value to interpret.
    :param fail_to_none: If `True`, return `None` if `val` is not a recognized boolean value (see above).

    :returns: Boolean value representing `val`.

    :raises ValueError: If `val` is not a recognized boolean value and `fail_to_none` is `False`.
    """
    if issubclass(val.__class__, bool):
        return val

    val = str(val).lower()

    if val in {'true', '1', 'yes', 't', 'y'}:
        return True

    if val in {'false', '0', 'no', 'f', 'n'}:
        return False

    if fail_to_none:
        return None

    raise ValueError('Cannot interpret as boolean value: {}'.format(val))


def region_merge(
        file_list: Iterable[str | pl.DataFrame],
        pad: int = 500
):
    """Merge regions from multiple BED files.

    Each input table must  have colums "chrom", "pos", and "end'.

    :param file_list: List of files to merge. Each element may be a string (filename) or a DataFrame (in-memory table
        of alignments). Files with zero size are skipped.
    :param pad: Pad interval matches by this amount, but do not add it to the output intervals. Similar to bedtools
        merge "slop" parameter.

    :returns: A table of merged regions.

    :raises ValueError: If a file name is not a string or a DataFrame.
    """
    raise NotImplementedError

    # # Get BED list
    # bed_list = list()
    #
    # for file_name in file_list:
    #     if isinstance(file_name, str):
    #         if os.stat(file_name).st_size > 0:  # Skip if file is empty
    #             bed_list.append(
    #                 pd.read_csv(
    #                     file_name, sep='\t', usecols=('#CHROM', 'POS', 'END')
    #                 )
    #             )
    #
    #     elif isinstance(file_name, pd.DataFrame):
    #         bed_list.append(file_name)
    #
    #     else:
    #         raise ValueError(f'File name is not a string or a DataFrame: {file_name} (type "{type(file_name)}")')
    #
    # df = (
    #     pd.concat(bed_list, axis=0)
    #     .sort_values(['#CHROM', 'POS', 'END'], ascending=[True, True, False])
    #     .reset_index(drop=True)
    # )
    #
    # # # Read regions
    # # df = pd.concat(
    # #     [
    # #         pd.read_csv(
    # #             file_name,
    # #             sep='\t',
    # #             usecols=('#CHROM', 'POS', 'END')
    # #         ) for file_name in file_list if os.stat(file_name).st_size > 0
    # #     ],
    # #     axis=0
    # # ).sort_values(['#CHROM', 'POS', 'END'], ascending=[True, True, False]).reset_index(drop=True)
    #
    # # Merge with intervals
    # df_list = list()
    #
    # chrom = None
    # pos = None
    # end = None
    #
    # for index, row in df.iterrows():
    #
    #     next_chrom = row['#CHROM']
    #     next_pos = row['POS'] - 500
    #     next_end = row['END'] + 500
    #
    #     if row['#CHROM'] != chrom:
    #
    #         # Write last record
    #         if chrom is not None:
    #             df_list.append(pd.Series([chrom, pos + pad, end - pad], index=['#CHROM', 'POS', 'END']))
    #
    #         chrom = next_chrom
    #         pos = next_pos
    #         end = next_end
    #
    #     else:
    #
    #         if next_pos <= end:
    #             pos = np.min([pos, next_pos])
    #             end = np.max([end, next_end])
    #
    #         else:
    #             df_list.append(pd.Series([chrom, pos + pad, end - pad], index=['#CHROM', 'POS', 'END']))
    #
    #             pos = next_pos
    #             end = next_end
    #
    # # Write last record
    # if chrom is not None:
    #     df_list.append(pd.Series([chrom, pos + pad, end - pad], index=['#CHROM', 'POS', 'END']))
    #
    # # Return
    # if len(df_list) > 0:
    #     return pd.concat(df_list, axis=1).T
    # else:
    #     return pd.DataFrame([], columns=['#CHROM', 'POS', 'END'])


def collapse_to_set(
        to_flatten: Iterable[Any],
        to_type: Optional[Callable] = None
) -> set[Any]:
    """Flatten an iterable and collapse into a set.

    For each element in the iterable, if it is not a tuple or list, `to_type` is applied (if defined) and the element
    is added to the set. Tuple or list elements are recursively unpacked.

    :param to_flatten: Iterable to flatten.
    :param to_type: A function to convert each element to a specific type (e.g. "int" or "float").

    :returns: Set of unique elements.

    :raises ValueError: If a value fails validation through `to_type`.
    """
    to_flatten = list(to_flatten)  # Copy so the original list is not modified
    s = set()

    if to_type is None:
        to_type = _ident

    while len(to_flatten) > 0:
        v = to_flatten.pop()

        if isinstance(v, (tuple, list)):
            to_flatten.extend(v)
        else:
            s.add(to_type(v))

    return s


def _ident(x):
    """Parameter identity function."""
    return x
