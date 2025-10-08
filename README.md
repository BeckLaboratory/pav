<h1 align="center"><img width="300px" src="img/logo/PAVLogo_Full.png"/></h1>
<p align="center">Phased Assembly Variant Caller</p>

***
<!-- Templated header from the pbsv github page: https://github.com/PacificBiosciences/pbsv -->

Variant caller for assembled genomes.


## Install

```
pip install pav3
```

To run PAV, use the `pav3` command after setting up configuration files.

```
python -m pav3
```


## Configuring PAV

PAV reads two configuration files:

* `config.json`: Points to the reference genome and can be used to set optional parameters.
* `assemblies.tsv`: A table of input assemblies.

Final results and temporary files are created in the analysis directory. Typically, these
configuration files will also be found in the analysis directory, although `config.json` may point
to an assembly outside the analysis directory.

Each analysis directory runs a single reference genome. Typically, all runs in the directory will use the same
configuration options in `config.json`, although per-sample configuration may be set with the "CONFIG" column in
`assemblies.json`.

### Base config: config.json

A JSON configuration file, `config.json`, configures PAV. Default options are built-in, and the only required option is
`reference` pointing to a reference FASTA file variants are called against.

Example:

```
{
  "reference": "/path/to/hg38.no_alt.fa.gz"
}
```

### Assembly table

The assembly table has one line per sample. The `NAME` column contains the assembly name (or sample name). This column
must be present and must be unique within the file.

PAV accepts one or more assembled haplotypes per sample, each with a column in the table:

* HAP_* (where * is a haplotype name): A named haplotype. For example, "HAP_h1", "HAP_h2" would be typical columns
  for a phased assembly. Other pseudo-haplotypes may be added, such as "HAP_unphased". Haplotypes must only contain
  alpha-numeric and "-", "+", "." characters. Underscores are currently not allowed to avoid filename format ambiguity.

Each entry in a "HAP_" column is the name of a file (FASTA, FASTQ, GFA, or FOFN). Multiple files can be input by
separating them by semi-colons (i.e. "path/to/file1.fasta.gz;path/to/file2.fasta.gz").

PAV will run for all haplotypes with a non-empty "HAP_" column. The variant calling process is done independently for
each haplotype, and variant calls are merged at the end in the order they are found in this table. To include a
haplotype with no input, add an entry to a zero-byte file to make the haplotype appear in the merged variant table and
in VCF genotypes.

#### Assembly-specific configuration

Global configuration parameters (those found in `config.json`) can be set per-sample, which wil override `config.json`
for specific assemblies. The configuration string can be placed into the optional "CONFIG" column and is a
semicolon-separated list of key-value pairs (i.e. "key1=val1;key2=val2"). The reference cannot be overridden.

### Suitable references

Do not use references with ALT, PATCH, or DECOY scaffolds for PAV, or generally, any assembly-based or long-read
variant calling tool. Reference redundancy may increase false-negative errors.

The GRCh38 HGSVC no-ALT reference for long reads can be found here:
ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC2/technical/reference/20200513_hg38_NoALT/

The T2T-CHM13v2.0 (hs1 on UCSC) is suitable without alteration. Custom per-sample assemblies containing a
single-haplotype or an unphased ("squashed") assembly typically also make a suitable reference as long as they are
free of large structural misassemblies and especially large false duplications.

### Assembly input files

Each sample has one or more "haplotype" assemblies. For example, a diploid human sample would have two complete
assemblies, one for haplotype 1 and one for haplotype 2.

The most efficient input is bgzipped and indexed FASTA files with ".fai" and ".gzi" indices. If files are not bgzipped,
they are bgzipped and stored in the "data" directory, which slows PAV significantly. Samtools can be used to prepare
files with `bgzip` and `samtools faidx` commands (Samtools package, not part of PAV).

Assemblies may be in FASTA, FASTQ, or GFA (hifiasm compatible) formats, and files may be gzipped. File indexes such as
".fai" files are generated if missing. Note that PAV never alters input or index files outside its own run directory.

PAV can also take an FOFN (File Of File Names) pointing to multiple input files and processing them as one. An FOFN
file ends with ".fofn" and is a plain text file with one filename per line. PAV will gather all input files from the
FOFN and create a compressed and indexed FASTA file in the "data" directory for it to use. If FOFN files contain
other FOFN files, they are followed recursively. Absolute paths should be used, and paths are relative to the run
directory otherwise.

### Input filename wildcards

PAV will attempt to replace "{asm_name}" and "{hap}" wildcards in paths with the assembly name (NAME column) and
the haplotype name making it easier to generate paths in the assembly table if input files follow the same file naming
conventions (can use the same path pattern for many samples and haplotype columns). For example, with sample "HG00733"
and haplotype "h2", path pattern "/path/to/assemblies/{asm_name}/{asm_name}_{hap}.fa.gz" becomes
"/path/to/assemblies/HG00733/HG00733_h1.fa.gz". Another wildcard "{sample}" is treated as an alias of "asm_name"
(i.e. the example above could have used either the "{asm_name}" or "{sample}" wildcards to achieve the same result).

## PAV versions

PAV uses Python package versioning with three fields:

* Major: Major changes or new features.
* Minor: Small changes, but may affect PAV's API or command-line interfaces.
* Patch: Small changes and minor new features. Patch versions do break API or command-line compatibility, but may
  add minor features or options to the API that were not previously supported.

PAV follows Python's packaging versioning scheme (https://packaging.python.org/en/latest/discussions/versioning/).

PAV may use pre-release versions with a suffix for development releases (".devN"), alpha ("aN"), beta ("bN"), or
release-candidate ("rcN") where "N" is an integer greater than 0. For example, "3.0.0.dev1" is a development version,
and "3.0.0a1" is an early alpha version, and "3.0.0rc1" is a release candidate, all of which precede the "3.0.0"
release and should not be considered production-ready.


## Cite PAV

Ebert et al., “Haplotype-Resolved Diverse Human Genomes and Integrated Analysis of Structural Variation,”
Science, February 25, 2021, eabf7117, https://doi.org/10.1126/science.abf7117 (PMID: 33632895).

PAV was also presented at ASHG 2021:

Audano et al., "PAV: An assembly-based approach for discovering structural variants, indels, and point mutations
in long-read phased genomes," ASHG Annual Meeting, October 20, 2021 (10:45 - 11:00 AM), PrgmNr 1160
