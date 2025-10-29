"""
Archive Package

Contains legacy modules and utilities for GenericNER.

Modules:
- functions: Core utility functions
- entity_extraction_lib: Entity extraction functionality
- entity_linking_lib: Entity linking functionality
- evaluation: Evaluation metrics and tools
- translation_lib: Translation utilities
- test: Test utilities
"""

# Import main modules for easier access
from . import functions
from . import entity_extraction_lib
from . import entity_linking_lib
from . import evaluation
from . import translation_lib

# Make commonly used modules available at package level
__all__ = [
    "functions",
    "entity_extraction_lib",
    "entity_linking_lib", 
    "evaluation",
    "translation_lib"
] 