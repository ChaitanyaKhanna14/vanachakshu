"""A trained classifier in place of a hand-tuned threshold.

The previous detector asked one question — did NDVI fall by more than 0.15 —
and could not be made to work, because at this base rate a single feature with
Cohen's d of 1.36 cannot separate one loss pixel from five thousand stable ones.

This asks 130 questions at once and learns how to weigh them, from labels rather
than from a person guessing. The features are AlphaEarth embeddings; the labels
are Hansen's published forest loss.

Two design decisions worth understanding, because both are places this could
quietly go wrong:

**Balanced training, honest evaluation.** Training samples equal numbers of loss
and stable pixels, because a classifier fed the true 1-in-5000 ratio learns to
answer "stable" every time and scores 99.98%. But *evaluation* must use the real
ratio, or the reported precision is a fiction. The two are deliberately
separated here.

**Spatially separated train and test.** Neighbouring pixels are nearly identical,
so a random split leaks: the classifier is tested on pixels it effectively
memorised. Splitting by location is the only honest option, and the difference
between the two is usually large enough to turn a broken model into a
publishable one on paper.
"""

from __future__ import annotations

from typing import Final

import ee

from vanachakshu import embeddings, hansen
from vanachakshu.config import OpticalDetectionConfig

__all__ = [
    "PROBABILITY_BAND",
    "classify_change",
    "feature_names",
    "sample_points",
    "split_regions",
    "train",
]

PROBABILITY_BAND: Final = "loss_probability"

# Trees in the forest. Beyond a few hundred the gains flatten while cost keeps
# rising, and Earth Engine's memory limit is a real constraint here.
_N_TREES: Final = 200

# Fraction of the AOI held back for testing, split by longitude rather than at
# random. See the module docstring on leakage.
_TEST_FRACTION: Final = 0.3


def _label_image(base_year: int, target_year: int, config: OpticalDetectionConfig) -> ee.Image:
    """1 where Hansen recorded loss, 0 for forest that has never been lost.

    Everything else is masked out. Pixels that were never forest are not
    negatives — they are irrelevant, and including them would teach the
    classifier to separate forest from farmland rather than loss from
    persistence.
    """
    loss = hansen.loss_mask(base_year, target_year).unmask(0).gt(0)
    forest = hansen.forest_mask(base_year, config)
    # Never-lost forest across the whole Hansen record, so a pixel cleared in
    # some other year is not labelled stable.
    never_lost = hansen.loss_mask(2000, hansen.HANSEN_LAST_LOSS_YEAR).unmask(0).gt(0).Not()
    stable = forest.And(never_lost)

    return loss.rename("label").updateMask(loss.Or(stable))


def feature_names() -> list[str]:
    """Every column the classifier is trained on, in a stable order."""
    names = [*embeddings.EMBEDDING_BANDS, *[f"D{i:02d}" for i in range(64)]]
    return [*names, "emb_euclid", "emb_cosine"]


def sample_points(
    geometry: ee.Geometry,
    base_year: int,
    target_year: int,
    points_per_class: int = 1500,
    seed: int = 42,
    config: OpticalDetectionConfig | None = None,
) -> ee.FeatureCollection:
    """Draw balanced labelled points from one region.

    ``points_per_class`` is equal across classes on purpose. The real ratio is
    about 1:5000, and a classifier trained on that learns to answer "stable"
    always. Balancing is standard for training; evaluation elsewhere uses the
    true ratio.

    Sampling one region per call, rather than the whole AOI at once, is what
    keeps this inside Earth Engine's limits: a 130-band stack over 1,463 km²
    exceeded the per-tile output ceiling, and raising tileScale far enough to
    fix that made the request time out instead. Two smaller requests succeed
    where one large one cannot.
    """
    cfg = config if config is not None else OpticalDetectionConfig()

    stacked = embeddings.change_stack(geometry, base_year, target_year).addBands(
        _label_image(base_year, target_year, cfg)
    )

    result: ee.FeatureCollection = stacked.stratifiedSample(
        numPoints=points_per_class,
        classBand="label",
        region=geometry,
        scale=10,
        seed=seed,
        geometries=False,
        dropNulls=True,
        tileScale=8,
    )
    return result


def split_regions(bbox_coords: list[float]) -> tuple[ee.Geometry, ee.Geometry]:
    """Split an AOI west/east into train and test regions.

    Split by location, never at random. Neighbouring pixels are near-duplicates,
    so a random split tests the classifier on ground it effectively memorised —
    which routinely turns a broken model into a publishable-looking one.
    """
    west, south, east, north = bbox_coords
    line = west + (east - west) * (1 - _TEST_FRACTION)
    return (
        ee.Geometry.Rectangle([west, south, line, north]),
        ee.Geometry.Rectangle([line, south, east, north]),
    )


def train(training_points: ee.FeatureCollection) -> ee.Classifier:
    """Fit a random forest that outputs a probability, not a hard class.

    Probability rather than a label because the operating point is a decision
    about how many wasted field visits are acceptable — a project-level choice,
    not something to bake into the model. It also lets one trained classifier
    serve both a cautious alerting threshold and a permissive screening one.
    """
    classifier: ee.Classifier = (
        ee.Classifier.smileRandomForest(numberOfTrees=_N_TREES, seed=42)
        .setOutputMode("PROBABILITY")
        .train(
            features=training_points,
            classProperty="label",
            inputProperties=feature_names(),
        )
    )
    return classifier


def classify_change(
    geometry: ee.Geometry,
    base_year: int,
    target_year: int,
    classifier: ee.Classifier,
) -> ee.Image:
    """Per-pixel probability that a pixel lost forest between the two years."""
    features = embeddings.change_stack(geometry, base_year, target_year)
    result: ee.Image = features.classify(classifier).rename(PROBABILITY_BAND)
    return result
