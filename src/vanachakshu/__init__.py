"""Vanachakshu — near-real-time forest disturbance alerts for the Western Ghats.

The package is split so that pure logic never imports Earth Engine:

    config     — validated settings, no I/O
    geometry   — bounding boxes, area maths, pure functions
    gee        — the only module that talks to Earth Engine

This separation is what lets the majority of the test suite run in CI without
Google credentials. See ``docs`` in the README for the rationale.
"""

__version__ = "0.1.0"
