PAV documentation
=================

The Phased Assembly Variant (PAV) caller generates variant calls from genome assemblies.

PAV 3 is a reimplementation of earlier PAV versions. It uses many of the same algorithms and
includes new features for identifying large and complex structural variants. Improved
low-confidence alignment prediction helps to filter variant calls from erroneous alignment records.

PAV 3 is fast. By utilizing modern data science tools, specifically Polars, PAV produces callsets
using a fraction of the memory and CPU resources as earlier versions.

This version is currently under active development, and a complete version has not yet been
released. If you would like to try it out, you can install it from source using the following
command:

.. code-block:: bash

   pip install pav3

PAV can then be run with a built-in command-line interface:

.. code-block:: bash

   python -m pav3 ...

Note that if you are in an environment where PAV is installed, ``python -m`` may be dropped from the
command.

PAV currently requires a configuration file and a table of assemblies to run, similar to earlier
versions. For details about setting these up, please see see
`CONFIG.md <https://github.com/BeckLaboratory/pav/blob/main/README.md>`_ in the PAV repository.

This documentation is also incomplete, check back for updates as development progresses.

See the `PAV Repository <https://github.com/BeckLaboratory/pav>`_ for additional information.

Contributions to the project are welcome.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

