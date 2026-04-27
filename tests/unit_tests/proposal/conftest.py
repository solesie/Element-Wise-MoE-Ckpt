"""Conftest for proposal unit tests.

The top-level tests/conftest.py initialises distributed / model-parallel
state which is unavailable on single-process developer runs. These tests
exercise pure filesystem / scoring logic on CPU, so we override with an
empty conftest.
"""
