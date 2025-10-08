# Contributing

Thank you for your interest in contributing! We appreciate your help including bugs reports and implementing new
features.

## Bugs, questions, and feature requests

The project's [GitHub issues page](https://github.com/BeckLaboratory/pav/issues) may be used to ask questions,
report bugs, and request features.

Please search for related issues before opening a new one. If you find a relevant open issue, you may monitor it for
updates (look for the "Subscribe" button on the issue's page). Add comments to existing issues as needed. Reference
related issues in your new issue or comments. Comments on closed issues may be missed.

Please make sure you understand the [Code of Conduct](CODE_OF_CONDUCT.md) before opening or commenting on issues.

### Reporting bugs

When reporting a bug, please provide the following information:
* A clear and concise description of the bug.
* The version of the project you are using.
* The steps to reproduce the bug.
* Any error messages you receive.
* Platform information (e.g. OS, Python version, etc).

Please include as many details as possible in your report, it helps us resolve the issue faster. Please also take a
moment to re-read your report to ensure it is clear and concise. When in doubt, more detail is generally better.

### Feature requests

We welcome ideas for improving on the project. Please provide a clear description of the feature you would like and
examples about why it is useful. Please help us understand the full context of your feature request so that we can
make an informed decision and properly prioritize the request.

## Contributing

Bugs are often fixed by the maintainers, but please feel free to look for issues and submit pull requests. Pick any
issue that is accepted and not assigned to another contributor. You may comment on the issue to express your interest
in working on it.

If you are a first time contributor, we will do our best to mark issues as a "good first issue".

If you would like to take on an issue, please comment on the issue to let others know. You may use the issue to discuss
possible solutions.

Please make sure you understand the [Code of Conduct](CODE_OF_CONDUCT.md) before becoming a contributor.

### Setting up your environment

Pull the codebase from GitHub:
```bash
git clone https://github.com/BeckLaboratory/pav.git
cd pav
```

A virtual environment is already setup with [uv](https://docs.astral.sh/uv/). See the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) to install uv if you do not have it.

Initialize your virtual environment:
```bash
uv sync
```

This will create a virtual environment in the `.venv` directory including all project dependencies and development tools
used directly by this project. Your IDE can use the virtual environment to provide features such as code completion.

Commands in the virtual environment can be run by prefixing them with `uv`.

Open an interactive session:
```bash
uv run ipython
```

### Setting up your IDE

This project was developed with [PyCharm](https://www.jetbrains.com/pycharm/), but no specific IDE is compulsory. When
opening the projcet in PyCharm, it should detect the uv virtual environment and use it. If not, you may need to
configure the IDE to use the environment in `.venv`.

### Linting and unit tests

Linting and unit tests are setup through tox environments [tox](https://tox.wiki/) already configured for this project.
A list of environments can be viewed in [tox.toml]. Environments starting with "py" are for unit testing various
Python versions, and a "flake8" environment is for linting. By default, running a tox environment will run all unit
tests or perform linting for the whole project.

Important environments are:
* flake8: Linting.
* sphinx: Build documentation.


To run a tox environment, use:
```bash
uv run tox -e <environment_name>
```

For example, to run the linting environment:
```bash
uv run tox -e flake8
```

The same tools are also available in the virtual environment and can be run with the `uv` prefix.

For example, lint a single file:
```bash
uv run flake8 <file_path>
```

Please ensure your contributions pass linting and do not break unit tests before submitting a pull request. When adding
a new feature, please consider adding unit tests for it.


### Pull requests

When your contribution is ready, open a pull request.

Before submitting a pull request:
* Make sure your changes are rebased against the main branch.
* Ensure your code is free of linting errors (See [Linting and unit tests])
* Ensure your code does not break any existing unit tests (See [Linting and unit tests]).
* Ensure your code follows the conventions outlined in the [Conventions](#conventions) section.

While submitting a pull request:
* Use a clear and concise title.
* Provide a clear and concise description.
* Reference the issue(s) you are working on.

A maintainer will review your request and may ask for changes or leave comments. Once all issues are resolved, the
pull request will be merged into the main branch, and your contribution will be included in the next release!

If you are stuck or unsure about your solution, feel free to open a draft pull request and ask for help.

## Conventions

* Use [reStructuredText docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)
  * Omit ":type:" and ":rtype:", allow type hints to fill it in.
* Use type hints.
* Use snake_case for variable names and function names.
* Use CamelCase for class names.
* Line length limit 120 characters
* Use 4 spaces for indentation.
* Place operators at the start of multi-line expressions rather than the end.
* Avoid backslashes, enclose in parethesis instead.
* All non-public members should be prefixed with an underscore.
* All public package members must appear in __all__.
* Use relative imports except for a few submodules with no dependencies on other submodules.
