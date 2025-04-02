# Low-Confidence alignment model training

Code and definitions in this directory are used to train models to recognize low-confidence alignment records.


## Input data

Input requirements:
* One or more PAV run directories executed at least up to alignment tables (PAV target `align_all`).
* A definition describing how the model should be trained.


## Definition JSON

The training script reads a JSON file that describes the model to be trained and paths to the data used to train it.

Top-level keys:
* name (required): Name of the output model. When using a model, this name is used to locate the model and to report
  errors.
* training: Describes parameters for splitting data into training, test, and cross-validation sets.
* data: Contains all paths to data to be used in training the model.

The definition may be split into multiple files, which are loaded in the order they appear on the command-line with
later files overwriting parameters defined in earlier files. The training JSONs for models packaged with PAV are
included, but without private data paths. A custom model can be trained on custom data using PAV's parameters by loading
specifying the PAV model JSON followed by a custom training JSON containing a "data" section. Similarly, default PAV
parameters may be overridden by specifying them in the custom JSON.

For an example, the default "prim_ap85" model was trained on HGSVC assemblies with this definition:
```
{
  "name": "prim_ap85",
  "type": "logistic",
  "description": "Low-confidence alignment model trained on HGSVC assemblies (Logsdon 2025)",
  "features": [
    "SCORE_PROP", "SCORE_MM_PROP", "ANCHOR_PROP", "QRY_PROP"
  ],
  "threshold": 0.5,
  "epochs": 16,
  "test_size": 50000,
  "k_fold": 4,
  "batch_size": 512,
  "data": [
    {
      "name": "hgsvc-vk-asm20-human-hs1",
      "comment": "HGSVC3 (Logsdon 2025) assemblies with hifiasm (asm20) - T2T-CHM13v2.0 - Human (no trios)",
      "path": "/path/to/pav/vk-asm20/hs1",
      "exclude": [
        "PanTro", "PanPan", "GorGor", "PonAbe", "PonPyg", "SymSyn",
        "HG00733", "HG00731", "HG00732",
        "HG00514", "HG00512", "HG00513",
        "NA19240", "NA19238", "NA19239"
      ]
    },
    {
      "name": "hgsvc-vk-asm20-humantrio-hs1",
      "eval": "true",
      "comment": "HGSVC3 (Logsdon 2025) assemblies with hifiasm (asm20) - T2T-CHM13v2.0 - Human trios",
      "path": "/path/to/pav/vk-asm20/hs1",
      "include": [
        "HG00733", "HG00731", "HG00732",
        "HG00514", "HG00512", "HG00513",
        "NA19240", "NA19238", "NA19239"
      ]
    }
  ]
}
```

This shortened version does not include all data sources, and paths to PAV run directories were removed. See
JSON files in PAV pipeline directory "files/lcmodel/model_def" for full examples (also with paths scrubbed).

### Base parameters

These base parameters should be defined for all training runs.

* name (required): Model name.
* description (optional): Model description.
* type (required): Model type. Currently, only "logistic" is recognized.
* features (required): List of features the model will ingest. These are keywords PAV understands, and many are columns
  in the alignment BED files PAV generates. See the [Features](#Features) section for details.

### Initial prediction parameters

The training process must make some initial predictions about alignment records to produce a dataset for supervised
learning models (i.e. a set of features (X) and labels (y) for each alignment record).

Several features include score proportions, which is defined as the score of the alignment (sum of match, mismatch, etc)
divided by the maximum possible score. See the [Features](#Features) section for details.

Low-confidence labels are set in three ways:
1. If the score proportion is below a threshold (score_prop_conf and score_mm_prop_conf).
2. If the alignment was completely removed by alignment trimming.
3. If the alignment clusters with other low-confidence alignmentns (1 and 2 above).

Values are defined below with defaults in paretheses (NA means not set).

* score_prop_conf (0.85): Minimum score proportion.
* score_prop_rescue (0.95): Do not cluster if the score proportion is at at least this value.
  * Must also meet rescue_length.
* score_mm_prop_conf (NA): Minimum score proportion over mismatches (gaps ignored).
* score_mm_prop_rescue (NA): Do not cluster if the mismatch score proportion is at at least this value.
  * Must also meet rescue_length.
* rescue_length (10000): Minimum length of an alignment to rescue.
* merge_flank (2000): Merge alignments marked by alignment proportion and alignment trimming (1 and 2 above) within this
  distance.


### Advanced parameters
Generally, these parameters should not be modified and can have consequences on model performance. 

* score_model (optional): Parameter string describing the alignment score model. PAV's default score model is used, and
  this parameter should not be used. Briefly, alignments are scored by by summing match, mismatch, and gap scores in
  an alignment record (CIGAR string). If there is a mismatch between this score model definition and the score model
  used to run PAV, then scores are recomputed for training and LC prediction when running PAV, which can be slow.


## Features

Several features are extraced from the alignments and used to train models and later to predict low-confidence
alignments. Some of these features are included in the alignment BED files PAV generates, and missing features are
generated automatically for training and inference.

Several features rely on scoring alignments with an alignment score model (pavlib.align.score). The trained LC model
saves a definition for the score model it uses. If the alignments for a PAV run 

* SCORE: Score of an alignment by using an alignment score model over the alignment events (CIGAR elements).
  * Not recommended
* SCORE_PROP: Proportion of SCORE to the maximum possible score for the query sequence aligned in a record. Value 1.0
  indicates that the alignment matched every base to a reference sequence with no mismatches or gaps.
* SCORE_MM: Like SCORE, but scores only mismatches across alignment events (gaps ignored).
  * Not recommended.
* SCORE_MM_PROP: Proportion of SCORE_MM to the maximum possible score for the query sequence aligned in a record.
* ANCHOR_PROP: A value indicating whether the alignment has a confident alignment upstream and/or downstream in the
  query sequence. Values are 0.0 (no confident alignment in the query sequence), 0.5 (a confident alignment either
  upstream or downstream), and 1.0 (a confident alignment both upstream and downstream). If the alignment record
  itself is a confident alignment, then it self-anchors (1.0).
* QRY_PROP: Proportion of the full query sequence aligned in a record. 



#### data
A list of data sources, each is a dictionary with the following keys:

* alias (optional): A short name for the data source, which will appear in model reports. If ommitted, a default
  name will be created.
* path (required): Path to a PAV run directory. This directory will have a "results" directory containing
  subdirectories for each sample.
* eval (optional): If true, then this data source is used only for evaluation and not for training. Samples are
  included for training by default.
* exclude (optional): A list of sample names to exclude.
* include (optional): A list of sample names to include. By default, all samples in the PAV run
  are included. If both `exclude_samples` and `include_samples` are specified, excluded samples will be removed.
* include_haps: A list of haplotype names to include from all samples. By default, all haplotypes in the PAV run
  are included. Regardless of this setting, missing haplotypes are ignored (i.e. if "h1" is included but an assembly
  has no "h1", an error is not generated).
* exclude_haps: A list of haplotype names to exclude from all samples. Haplotypes appearing in both `include_haps` and
  `exclude_haps` are excluded.

## Implementation notes

### Stage cache and UUIDs

Each stage writes a stage cache filename in its working subdirectory named "stage_cache.json.gz". This file contains
sections for tracking stage UUIDs (updated each time a stage or stage step is run) and parameters that stage was run
with. Mismatches with an upstream UUID or configuration items affecting a stage will cause it to re-run.

In this JSON file, section "stage_uuid" stores cached UUIDs. Keys are a stage name with an optional step name separated
with a "_". Each stage stores both it's own UUIDs (stage name will match the stage name that wrote the JSON) and UUIDs
for upstream stages it monitors (the stage name will be different). For example, the "features" stage will store
UUIDs it creates, which will all start with "features". The "features" stage will also store UUIDs it recorded from the
init stage, and when the training script runs, if the "init" UUIDs stored by the "features" stage do not match, in
indicates that init has updated files and features will re-run.

This system allows some parameters to be tweaked, such as training parameters or evaluation test sets, without
re-running all upstream steps. It also allows the training process to pick up after a crash or interruption.
