"""
Make variant calls from aligned contigs.
"""

import os
import sys

global expand
global shell
global workflow


#
# Global constants
#

PIPELINE_DIR = os.path.dirname(os.path.realpath(workflow.snakefile))


#
# Parameters
#

configfile: config.get('config_file', 'config.json')


### Parameters from config ###

# Reference FASTA & FAI
REF_FA = 'data/ref/ref.fa.gz'

REF_FAI = REF_FA + '.fai'

# VCF file pattern
VCF_PATTERN = f'{config.get("vcf_prefix", "")}{{asm_name}}{config.get("vcf_suffix", "")}.vcf.gz'

#
# Assembly library and dependency imports
#

sys.path.append(PIPELINE_DIR)  # pavlib
sys.path.append(os.path.join(PIPELINE_DIR, 'dep', 'svpop'))  # svpoplib
sys.path.append(os.path.join(PIPELINE_DIR, 'dep', 'svpop', 'dep'))  # kanapy
sys.path.append(os.path.join(PIPELINE_DIR, 'dep', 'svpop', 'dep', 'ply'))  # ply - lexer / parser

import pavlib


#
# Read sample config
#

ASM_TABLE_FILENAME = config.get('assembly_table', None)

if ASM_TABLE_FILENAME is None and os.path.isfile('assemblies.tsv'):
    ASM_TABLE_FILENAME = 'assemblies.tsv'

if ASM_TABLE_FILENAME is None and os.path.isfile('assemblies.xlsx'):
    ASM_TABLE_FILENAME = 'assemblies.xlsx'

if ASM_TABLE_FILENAME is None:
    raise RuntimeError('No input assembly table in config ("assembly_table") and the default table filename was not found ("assemblies.tsv")')

ASM_TABLE = pavlib.pipeline.read_assembly_table(ASM_TABLE_FILENAME, config)


#
# Rules
#

# Environment source file for shell commands
ENV_FILE = config.get('env_source', 'setenv.sh')

if not os.path.isfile(ENV_FILE) or pavlib.util.as_bool(config.get('ignore_env_file', False)):
    ENV_FILE = None

if ENV_FILE:
    shell.prefix(f'set -euo pipefail; source {ENV_FILE}; ')
else:
    shell.prefix('set -euo pipefail; ')


### Wildcard constraints ###

wildcard_constraints:
    asm_name=r'[A-Za-z_\-0-9\.]+'

### Default rule ###

localrules: pav_all

# pav_all
#
# Make all files for all samples.
rule pav_all:
    input:
        bed=expand('{asm_name}.vcf.gz', asm_name=ASM_TABLE.index)
        # bed=expand('vcf/merged/{asm_name}.vcf.gz', asm_name=ASM_TABLE.index)

### Includes ###

include: os.path.join(PIPELINE_DIR, 'rules/pipeline.snakefile')

include: os.path.join(PIPELINE_DIR, 'rules/data.snakefile')
include: os.path.join(PIPELINE_DIR, 'rules/align.snakefile')
include: os.path.join(PIPELINE_DIR, 'rules/call.snakefile')
include: os.path.join(PIPELINE_DIR, 'rules/call_inv.snakefile')
include: os.path.join(PIPELINE_DIR, 'rules/call_lg.snakefile')
include: os.path.join(PIPELINE_DIR, 'rules/tracks.snakefile')
include: os.path.join(PIPELINE_DIR, 'rules/figures.snakefile')
include: os.path.join(PIPELINE_DIR, 'rules/vcf.snakefile')
