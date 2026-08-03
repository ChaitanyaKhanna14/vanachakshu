"""Shared pytest fixtures.

Test bounding boxes here are deliberately small (0.1 degree, ~124 km2).

An earlier version of this suite reached for 1-degree boxes as "obviously
simple" fixtures. They were rejected by the AOI quota guardrail, correctly: a
1-degree box is ~12,400 km2, more than twice the largest area this project
should ever process in one run. Keeping fixtures inside the range the system
actually supports stops the tests drifting away from reality.
"""

from __future__ import annotations

import pytest

from vanachakshu.config import AreaOfInterest, BoundingBox

# ~124 km2 near the equator — comfortably inside MAX_AOI_SQ_KM.
TINY_BOX = {"west": 0.0, "south": 0.0, "east": 0.1, "north": 0.1}

# ~123 km2, positioned over the real project area so the derived UTM zone
# matches production (32643).
WESTERN_GHATS_BOX = {"west": 74.60, "south": 14.90, "east": 74.70, "north": 15.00}


@pytest.fixture
def tiny_bbox() -> BoundingBox:
    """A minimal valid bounding box, for tests that don't care where it is."""
    return BoundingBox(**TINY_BOX)


@pytest.fixture
def western_ghats_bbox() -> BoundingBox:
    """A small box over the project AOI, for tests that depend on the UTM zone."""
    return BoundingBox(**WESTERN_GHATS_BOX)


@pytest.fixture
def tiny_aoi(tiny_bbox: BoundingBox) -> AreaOfInterest:
    """A minimal valid area of interest."""
    return AreaOfInterest(name="Test AOI", bbox=tiny_bbox)
