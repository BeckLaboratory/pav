"""Shared pytest fixtures and markers for the PAV3 test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_ROOT = PROJECT_ROOT / 'local' / 'test_data' / 'pav_short'

DEFAULT_RUN_SAMPLES = 'HG02011,HG03683'


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register PAV3-specific CLI options."""
    parser.addoption(
        '--run-samples',
        action='store',
        default=DEFAULT_RUN_SAMPLES,
        help=(
            'Comma-separated list of assembly sample names whose parametrized '
            'tests should run. Use "all" to include every sample. Tests not '
            'tagged with @pytest.mark.sample(...) are unaffected. '
            f'Default: {DEFAULT_RUN_SAMPLES}.'
        ),
    )


def _parse_run_samples(value: str) -> set[str] | None:
    """Parse the ``--run-samples`` option. Return ``None`` to mean 'all'."""
    value = value.strip()
    if value.lower() == 'all':
        return None
    return {tok.strip() for tok in value.split(',') if tok.strip()}


@pytest.fixture(scope='session')
def test_data_dir() -> Path:
    """Return the root path of the bundled small test dataset.

    The dataset is not shipped with the package; it lives at
    ``local/test_data/pav_short`` in a developer checkout. Tests that need it
    should use this fixture together with ``@pytest.mark.requires_test_data``.
    """
    return TEST_DATA_ROOT


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply default-skip markers based on test data availability and ``--run-samples``."""
    test_data_missing = not TEST_DATA_ROOT.is_dir()
    test_data_skip = pytest.mark.skip(
        reason=f'test data not found at {TEST_DATA_ROOT} (see tests/README for setup)',
    )

    requested = _parse_run_samples(config.getoption('--run-samples'))

    for item in items:
        if test_data_missing and 'requires_test_data' in item.keywords:
            item.add_marker(test_data_skip)
            continue

        if requested is None:
            continue

        sample_markers = list(item.iter_markers(name='sample'))
        if not sample_markers:
            continue

        sample_names = {m.args[0] for m in sample_markers if m.args}
        if sample_names and not (sample_names & requested):
            item.add_marker(pytest.mark.skip(
                reason=(
                    f'sample(s) {sorted(sample_names)} not in --run-samples; '
                    'pass --run-samples=all or --run-samples=<name>[,<name>...] to include'
                ),
            ))
