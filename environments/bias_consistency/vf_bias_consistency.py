"""
Verifiers Framework integration for bias-consistency environment.
This file provides the expected vf_*.py entry point for the Prime framework.
"""

# Import everything from the main package
from bias_consistency import *
from bias_consistency import BiasConsistency, load_environment

# Make sure the main class is available at module level
__all__ = ['BiasConsistency', 'load_environment']

# Optional: Create an alias for backwards compatibility
BiasConsistencyEval = BiasConsistency
