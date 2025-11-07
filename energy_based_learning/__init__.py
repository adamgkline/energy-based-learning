"""Energy-Based Learning Framework

This package provides implementations of energy-based learning algorithms
for analog computing, including equilibrium propagation and related methods.
"""

__version__ = "0.1.0"

# Import main modules for convenience
from . import datasets
from . import model
from . import training

__all__ = ["datasets", "model", "training"]
