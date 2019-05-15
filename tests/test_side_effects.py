"""
Unit and regression test for the side_effects package.
"""

# Import package, test suite, and other packages as needed
import side_effects
import pytest
import sys

def test_side_effects_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "side_effects" in sys.modules
