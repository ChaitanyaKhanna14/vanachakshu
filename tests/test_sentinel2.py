"""Tests for Sentinel-2 cloud masking constants.

The mask class set is the highest-stakes constant in the optical pipeline. Every
one of its failure modes is silent: mask the wrong class and you either destroy
the signal you are looking for or manufacture the one you are not. So it gets
tested by *intent* — "bare soil must survive" — rather than by restating the
literal set, which would only assert that a copy-paste succeeded.
"""

from __future__ import annotations

import pytest

from vanachakshu.sentinel2 import SCL_CLASS_NAMES, SCL_MASK_CLASSES


class TestSclMaskClasses:
    def test_vegetation_is_never_masked(self) -> None:
        # Class 4 is the forest itself. Masking it empties the composite.
        assert 4 not in SCL_MASK_CLASSES

    def test_bare_soil_is_never_masked(self) -> None:
        # The single most important line in this file. A freshly cleared patch
        # IS bare soil (class 5). Masking it would remove precisely the pixels
        # the detector exists to find, and the failure would look like "the
        # model detects nothing" rather than like a masking bug.
        assert 5 not in SCL_MASK_CLASSES

    def test_water_is_not_masked(self) -> None:
        # Water is a legitimate ground observation. Masking it would leave
        # permanent holes in the composite around reservoirs and rivers.
        assert 6 not in SCL_MASK_CLASSES

    def test_unclassified_is_not_masked(self) -> None:
        assert 7 not in SCL_MASK_CLASSES

    @pytest.mark.parametrize(
        ("scl_class", "reason"),
        [
            (0, "no data"),
            (1, "saturated or defective"),
            (3, "cloud shadow"),
            (8, "cloud, medium probability"),
            (9, "cloud, high probability"),
            (10, "thin cirrus"),
        ],
    )
    def test_cloud_and_defect_classes_are_masked(self, scl_class: int, reason: str) -> None:
        assert scl_class in SCL_MASK_CLASSES, f"class {scl_class} ({reason}) must be masked"

    def test_topographic_shadow_is_masked(self) -> None:
        # Class 2 is a deliberate, non-obvious choice. In flat terrain it is
        # usually kept; in the Western Ghats, hillside shadow depresses NDVI
        # exactly the way real clearing does, so every north-facing slope would
        # read as disturbance. Precision over recall.
        assert 2 in SCL_MASK_CLASSES

    def test_every_masked_class_is_a_real_scl_class(self) -> None:
        # Guards a typo like 12 or 99 silently masking nothing at all.
        unknown = SCL_MASK_CLASSES - set(SCL_CLASS_NAMES)
        assert not unknown, f"not valid SCL classes: {sorted(unknown)}"

    def test_mask_keeps_some_classes(self) -> None:
        # A mask that discarded everything would produce an empty composite and
        # zero detections — which reads as "no deforestation found".
        kept = set(SCL_CLASS_NAMES) - SCL_MASK_CLASSES
        assert kept, "mask discards every class; the composite would be empty"


class TestSclClassNames:
    def test_covers_the_full_published_range(self) -> None:
        # ESA defines exactly 0-11.
        assert set(SCL_CLASS_NAMES) == set(range(12))

    def test_every_class_has_a_description(self) -> None:
        assert all(name.strip() for name in SCL_CLASS_NAMES.values())
