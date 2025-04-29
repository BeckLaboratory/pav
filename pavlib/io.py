"""
I/O utilities.
"""

import gzip
import os
import pysam
import subprocess

class PlainOrGzFile:
    """
    Read a plain or a gzipped file using context guard.

    Example:
        with PlainOrGzReader('path/to/file.gz'): ...
    """

    def __init__(self, file_name, mode='rt'):

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

    def __enter__(self):

        if self.is_gz:
            self.file_handle = gzip.open(self.file_name, self.mode)
        else:
            self.file_handle = open(self.file_name, self.mode)

        return self.file_handle

    def __exit__(self, exc_type, exc_value, traceback):

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

    def __init__(self, file_name):

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
            raise RuntimeError(f'File name or open FASTA file is not a string or a pysam.FastaFile: {file_name} (type "{type(file_name)}")')

        self.file_handle = None

        self.is_open = False

    def __enter__(self):

        if self.is_open:
            raise RuntimeError(f'Enter called: File is already open by this context guard: {self.file_name}')

        if not self.is_pysam:
            self.file_handle = pysam.FastaFile(self.file_name)

        self.is_open = True

        return self.file_handle

    def __exit__(self, exc_type, exc_value, traceback):

        if not self.is_open:
            raise RuntimeError(f'Exit called: File is not open by this context guard: {self.file_name}')

        if not self.is_pysam:
            self.file_handle.__exit__(exc_type, exc_value, traceback)

        self.is_open = False

class SamStreamer(object):
    """
    Stream a SAM, BAM, or CRAM file as a line generator.
    """

    def __init__(self, filename, file_type=None, ref_fa=None):
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
                    raise RuntimeError(f'File name is not a string (type "{type(filename)}"), expected SAM, BAM, or CRAM file, but file name does not end with ".sam", ".sam.gz", ".bam", or ".cram"')

        else:
            if file_type is not None and file_type.strip().lower() != 'iter':
                raise RuntimeError(f'File name is not a string (type "{type(filename)}"), expected iterator, but "type" argument is not "iter" (file_type="{file_type}"')
            file_type = 'iter'

        self.file_type = file_type

    def __enter__(self):

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

    def __exit__(self, exc_type, exc_value, traceback):

        if not self.is_open:
            raise RuntimeError(f'Exit called: File is not open by this context guard: {self.filename}')

        if self.file_type == {'bam', 'cram'}:
            self.iterator.close()

        elif self.file_type == 'sam':
            self.iterator.__exit__(exc_type, exc_value, traceback)

        self.is_open = False
        self.is_closed = True

    def __iter__(self):
        if self.is_closed:
            raise RuntimeError('Iterator is closed')

        if not self.is_open:
            self.__enter__()

        return self.iterator

class DecodeIterator(object):
    """
    Utility iterator for decoding bytes to strings. Needed by SamStreamer for streaming BAM & CRAM files.
    """

    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator).decode()

    def close(self):
        self.iterator.close()
