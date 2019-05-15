"""
side_effects
Predict DDI and their side effects
"""

# Add imports here
from .side_effects import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
