"""Shared pytest fixtures and markers for the PAV3 test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_ROOT = PROJECT_ROOT / 'local' / 'test_data' / 'pav_short'


@pytest.fixture(scope='session')
def test_data_dir() -> Path:
    """Return the root path of the bundled small test dataset.

    The dataset is not shipped with the package; it lives at
    ``local/test_data/pav_short`` in a developer checkout. Tests that need it
    should use this fixture together with ``@pytest.mark.requires_test_data``.
    """
    return TEST_DATA_ROOT


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``requires_test_data`` tests when the dataset is not present."""
    if TEST_DATA_ROOT.is_dir():
        return

    skip_marker = pytest.mark.skip(
        reason=f'test data not found at {TEST_DATA_ROOT} (see tests/README for setup)',
    )
    for item in items:
        if 'requires_test_data' in item.keywords:
            item.add_marker(skip_marker)
