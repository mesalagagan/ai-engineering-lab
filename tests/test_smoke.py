"""Smoke tests for the project foundation."""

import ai_engineering_lab


def test_package_imports() -> None:
    """The package can be imported from the source layout."""
    assert ai_engineering_lab.__doc__
