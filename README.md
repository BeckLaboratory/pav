<h1 align="center"><img width="300px" src="img/logo/PAVLogo_Full.png"/></h1>
<p align="center">Phased Assembly Variant Caller</p>

***
<!-- Templated header from the pbsv github page: https://github.com/PacificBiosciences/pbsv -->

Variant caller for assembled genomes.

## Development release (Please read)

PAV 3 is currently a development release and may contain bugs. Please report problems you encounter.

PAV now uses Polars for fast data manipulation. If your job fails with strange Polars errors, such as "capacity
overflow" or "PanicException", this is likely from Polars running out of memory.


### Notes for early adopters

If you have been using development versions, please read about these changes.

**Call subcommand is deprecated.** If you have been running pav3 with the "call" subcommand, switch it to "batch"
(i.e. ("pav3 batch ...")). A future version will use "call" for single assemblies (with multiple haplotypes) defined on
the command-line and "batch" to run all assemblies from an assembly table.

**Moving configuration to pav.json.** The configuration file "config.json" is being moved to "pav.json". A future
version will ignore "config.json".


## Install and run


### Install
```
pip install pav3
```

### Run
To run PAV, use the `pav3 batch` command after setting up configuration files (see below). A future version will add
`pav call` for single-assemblies without requiring configuration files. 

Some Python environments may require you to run `pav3` through the `python` command:
```
python -m pav3 batch
```


### Dependencies
Currently, PAV needs `minimap2` in the environment where it is run. This may change in future releases. All other
dependencies are handled by the installer.


### Output
PAV will output a VCF file for each sample called `NAME.vcf.gz`.

* `results/NAME/call_hap`: Unmerged variant calls.
  * Includes tables of callable regions in reference ("callable_ref") and query ("callable_qry") coordinates.
* `results/NAME/call`: Variant calls merged across haplotypes.

All tables are in parquet format.


## Configuring PAV for batch runs

To run assemblies in batches ("pav3 batch" command), PAV reads two configuration files:

* `pav.json`: Points to the reference genome and can be used to set optional parameters.
* `assemblies.tsv`: A table of input assemblies.

### Base config: pav.json

A JSON configuration file, `pav.json`, configures PAV. Default options are built-in, and the only required option is
`reference` pointing to a reference FASTA file.

Example:

```
{
  "reference": "/path/to/hg38.no_alt.fa.gz"
}
```

### Assembly table

The assembly table points PAV to input assemblies. It may be in TSV, CSV, Excel, or parquet formats (TSV and CSV may
optionally be gzipped). Each assembly has one row in the table.

Columns:
* NAME: Assembly or sample name.
* HAP_\*: One column for each assembled haplotype.


#### Name column

The `NAME` column contains the assembly name (or sample name). This column must be present and each row must have a
unique value.


#### Haplotype columns

PAV accepts one or more assembled haplotypes per assembly, each with a column in the table starting with "HAP_". Each
is a path to an input file for one assembly haplotype.

Common column names are "HAP_h1" for haplotype "h1" and "HAP_h2" for haplotype "h2". For some assemblies with known
parental origins, "HAP_mat" and "HAP_pat" are commonly used.

There must be at least one haplotype per assembly, and PAV has no limits on the number of haplotypes (i.e. 3 or more
are acceptable).

Not all assemblies need to have the same haplotypes. PAV will ignore empty the "HAP_" values for each assembly. For
example, if some assemblies have an "unphased" haplotype and other do not, include "HAP_unphased" and leave it blank
for assemblies that do not have it.

Note that genotypes in the VCF file will have one allele for each haplotype defined for the assembly. For an assembly
with haplotypes "h1", "h2", and "unphased", three genotype alleles will be possible (e.g. "1|0|." for a heterozygous
variant present in "h1", not present in "h2", and uncallable in "unphased"). The order of genotypes is determined by
the order of haplotype columns in the assembly table.

Each "HAP_" column contains paths to input files in FASTA, FASTQ, GFA, or FOFN format. FOFN may contain paths to these
same file types including other FOFNs (recursive FOFNs are not recommended, but PAV will detect cycles). Multiple files
can be input by separating them by semi-colons (i.e. "path/to/file1.fasta.gz;path/to/file2.fasta.gz") and a mix of
types is possible. PAV will be fastest if the input is a single bgzipped and indexed FASTA file, it will build its
own FASTA file for all other cases.


#### Configuration column for global overrides

An optional "CONFIG" column can override global configuration parameters per assembly. Global configuration parameters
are defined in `pav.json` or are PAV default values if not defined. Values in this column are semicolon-separated lists
of key-value pairs (i.e. "key1=val1;key2=val2"). The "reference" parameter cannot be overridden per assembly.


### A note on references

Do not use references with ALT, PATCH, or DECOY scaffolds for PAV, or generally, any assembly-based or long-read
variant calling tool. Reference redundancy may increase callset errors.

The GRCh38 HGSVC no-ALT reference for long reads can be found here:
ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC2/technical/reference/20200513_hg38_NoALT/

The T2T-CHM13v2.0 (hs1 on UCSC) is suitable without alteration. Custom per-sample assemblies containing a
single-haplotype or an unphased ("squashed") assembly typically also make a suitable reference as long as they are
free of large structural misassemblies and especially large false duplications.


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

PAV 3 does not yet have a citation. For now, use the citation for previous PAV versions, but check back for updates.

Ebert et al., “Haplotype-Resolved Diverse Human Genomes and Integrated Analysis of Structural Variation,”
Science, February 25, 2021, eabf7117, https://doi.org/10.1126/science.abf7117 (PMID: 33632895).
