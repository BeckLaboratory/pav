"""I/O utilities."""

__all__ = [
    'PlainOrGzFile',
    'FastaReader',
    'SamStreamer',
    'DecodeIterator',
    'NullWriter',
    'ResourceReader',
]

import gzip
import importlib.resources
import os
import pathlib
import pysam
import subprocess
from types import TracebackType
from typing import IO, BinaryIO, Iterator, TextIO


class PlainOrGzFile:
    """Read a plain or a gzipped file using context guard.

    Example::

        with PlainOrGzReader('path/to/file.gz'): ...
    """

    def __init__(self, file_name, mode='rt') -> None:
        """Create a context guard for a plain or a gzipped file.

        :praam file_name: File name.
        :param mode: File mode.
        """
        if file_name is None:
            raise RuntimeError('File name is missing')

        file_name = file_name.strip()

        if not file_name:
            raise RuntimeError('File name is empty')

        if mode is not None:
            mode = mode.strip()

            if not mode:
                mode = 'rt'
        else:
            mode = 'rt'

        self.file_name = file_name
        self.is_gz = file_name.strip().lower().endswith('.gz')

        self.mode = mode

        self.file_handle = None

    def __enter__(self) -> IO:
        """Enter context guard."""
        if self.is_gz:
            self.file_handle = gzip.open(self.file_name, self.mode)
        else:
            self.file_handle = open(self.file_name, self.mode)

        return self.file_handle

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context guard."""
        if self.file_handle is not None:
            self.file_handle.__exit__(exc_type, exc_value, traceback)
            self.file_handle = None


class FastaReader:
    """
    Accepts a FASTA file name or an open FASTA file (pysam.FastaFile) and provides a context-guard for the file.

    Examples:
        with FastaReader('path/to/file.fa.gz'): ...
        with FastaReader(fasta_file): ...  # fasta_file is a pysam.FastaFile
    """

    def __init__(self, file_name) -> None:
        """Create a context guard for a FASTA file.

        :param file_name: File name or open FASTA file.
        """
        if file_name is None:
            raise RuntimeError('File name or open FASTA file is missing')

        if isinstance(file_name, str):
            file_name = file_name.strip()

            if not file_name:
                raise RuntimeError('File name is empty')

            if not os.path.isfile(file_name):
                raise RuntimeError(f'File name does not exist or is not a regular file: {file_name}')

            self.is_pysam = False

            self.file_name = file_name
            self.file_handle = None

        elif isinstance(file_name, pysam.FastaFile):
            self.is_pysam = True

            self.file_name = "<pysam.FastaFile Object>"
            self.file_handle = file_name

        else:
            raise RuntimeError(
                f'File name or open FASTA file is not a string or a pysam.FastaFile: '
                f'{file_name} (type "{type(file_name)}")'
            )

        self.file_handle = None

        self.is_open = False

    def __enter__(self) -> pysam.FastaFile:
        """Enter context guard."""
        if self.is_open:
            raise RuntimeError(f'Enter called: File is already open by this context guard: {self.file_name}')

        if not self.is_pysam:
            self.file_handle = pysam.FastaFile(self.file_name)

        self.is_open = True

        return self.file_handle

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context guard."""
        if not self.is_open:
            raise RuntimeError(f'Exit called: File is not open by this context guard: {self.file_name}')

        if not self.is_pysam:
            self.file_handle.__exit__(exc_type, exc_value, traceback)

        self.is_open = False


class SamStreamer(object):
    """Stream a SAM, BAM, or CRAM file as a line generator."""

    def __init__(self, filename, file_type=None, ref_fa=None) -> None:
        """Initialize streamer.

        :param filename: File name.
        :param file_type: File type.
        :param ref_fa: Reference FASTA file.
        """
        self.filename = filename.strip()
        self.ref_fa = ref_fa

        self.is_open = False
        self.is_closed = False

        # Set type
        if isinstance(filename, str):
            filename = filename.strip()

            if not filename:
                raise RuntimeError('File name is empty')

            filename_lower = filename.lower()

            if file_type is None:
                if filename_lower.endswith('.sam') or filename_lower.endswith('.sam.gz'):
                    file_type = 'sam'
                elif filename_lower.endswith('.bam'):
                    file_type = 'bam'
                elif filename_lower.endswith('.cram'):
                    file_type = 'cram'
                else:
                    raise RuntimeError(
                        f'File name is not a string (type "{type(filename)}"), '
                        f'expected SAM, BAM, or CRAM file, but file name does not end with '
                        f'".sam", ".sam.gz", ".bam", or ".cram"'
                    )

        else:
            if file_type is not None and file_type.strip().lower() != 'iter':
                raise RuntimeError(
                    f'File name is not a string (type "{type(filename)}"), '
                    f'expected iterator, but "type" argument is not "iter" (file_type="{file_type}"'
                )
            file_type = 'iter'

        self.file_type = file_type

    def __enter__(self) -> Iterator[str]:
        """Enter context guard."""
        if self.is_open:
            raise RuntimeError(f'Enter called: File is already open by this context guard: {self.filename}')

        if self.file_type == 'sam':
            self.iterator = PlainOrGzFile(self.filename, 'rt').__enter__()
            self.is_open = True

            return self.iterator

        if self.file_type in {'bam', 'cram'}:
            # self.iterator = DecodeIterator(
            #     subprocess.Popen(['samtools', 'view', '-h', self.filename], stdout=subprocess.PIPE).stdout
            # )

            if self.ref_fa is not None:
                samtools_cmd = ['samtools', 'view', '-h', '-T', self.ref_fa, self.filename]
            else:
                samtools_cmd = ['samtools', 'view', '-h', self.filename]

            self.iterator = DecodeIterator(
                subprocess.Popen(samtools_cmd, stdout=subprocess.PIPE).stdout
            )

            self.is_open = True

            return self.iterator

        if self.file_type == 'iter':
            self.iterator = self.filename

            self.is_open = True

            return self.iterator

        raise RuntimeError(f'Unknown file type: {self.file_type}')

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context guard."""
        if not self.is_open:
            raise RuntimeError(f'Exit called: File is not open by this context guard: {self.filename}')

        if self.file_type == {'bam', 'cram'}:
            self.iterator.close()

        elif self.file_type == 'sam':
            self.iterator.__exit__(exc_type, exc_value, traceback)

        self.is_open = False
        self.is_closed = True

    def __iter__(self):
        """Return SAM line iterator."""
        if self.is_closed:
            raise RuntimeError('Iterator is closed')

        if not self.is_open:
            self.__enter__()

        return self.iterator


class DecodeIterator(object):
    """Utility iterator for decoding bytes to strings.

    Needed by SamStreamer for streaming BAM & CRAM files.
    """

    def __init__(self, iterator) -> None:
        """Initialize iterator."""
        self.iterator = iterator

    def __iter__(self) -> Iterator[str]:
        """Return iterator."""
        return self

    def __next__(self) -> str:
        """Return next item."""
        return next(self.iterator).decode()

    def close(self) -> None:
        """Close iterator."""
        self.iterator.close()


class NullWriter(object):
    """A writer object that discards all messages."""

    def write(self, *args, **kwargs):
        """Discard message."""
        pass

    def flush(self):
        """Flush (has no effect)."""
        pass

    def writelines(self, *args, **kwargs):
        """Discard message."""
        pass

    def close(self):
        """Close (has no effect)."""
        pass


class ResourceReader(object):
    """Open a resource for reading within a context guard.

    Handles two resource types (`resource_type` attribute):

        * package: File in a package.
        * filesystem: File on the filesystem.

    In both cases, `anchor` refers to the directory where files are found. If the resource type is "package", then this
    should formatted like a package resource (i.e. "pav3.data.lcmodel.default"). If the resource type is
    "filesystem", then `anchor` is a path string or `pathlib.Path` object pointing to the location of the resource
    (relative to the current working directory if not an absolute path).

    In all cases, `name` is the name of the resource file within `anchor` to be opened.

    :ivar anchor: Directory or package location where the file is found.
    :ivar name: Name of the file within `anchor` to open.
    :ivar resource_type: "package" to locate a package resource, or "filesystem" to search the filesystem.
    :ivar text_mode: If True, open files in text mode, otherwise, open in binary mode.
    """
    anchor: str | pathlib.Path
    name: str
    resource_type: str
    text_mode: bool

    def __init__(
            self,
            anchor: str | pathlib.Path,
            name: str,
            resource_type: str = 'package',
            text_mode: bool = True
    ) -> None:
        """Init reader object."""
        self.anchor = anchor
        self.name = str(name)
        self.resource_type = resource_type
        self.text_mode = text_mode

        if self.resource_type == 'package':
            self.anchor = str(self.anchor)

        self.file_handle = None

    def __enter__(self) -> TextIO | BinaryIO:
        """Enter context."""
        if self.file_handle is not None:
            raise RuntimeError(f'Enter called: File is already open by this context guard: anchor={self.anchor}, name={self.name}')

        if self.resource_type == 'package':
            if self.text_mode:
                self.file_handle = importlib.resources.open_text(self.anchor, self.name)
            else:
                self.file_handle = importlib.resources.open_binary(self.anchor, self.name)

        elif self.resource_type == 'filesystem':
            self.file_handle = open(pathlib.Path(self.anchor) / self.name, 'r' + ('t' if self.text_mode else 'b'))

        else:
            raise ValueError(f'Unknown resource type: {self.resource_type}')

        return self.file_handle

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
    ) -> None:
        """Exit context"""

        self.file_handle.close()
        self.file_handle = None
