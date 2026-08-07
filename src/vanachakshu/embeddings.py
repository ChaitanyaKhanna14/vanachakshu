"""AlphaEarth satellite embeddings as the detector's feature space.

Each 10 m pixel carries 64 numbers summarising a whole year of multi-sensor
observation — Sentinel-1 radar and Sentinel-2 optical already fused by a model
trained on the global archive.

**Why this replaces the NDVI difference.** Measured over this AOI, separating
Hansen's 2025 forest loss from stable forest:

    ndvi_drop     Cohen's d = 1.36
    embedding L2  Cohen's d = 2.20

That 62% gain is not incremental at this base rate. Hansen records 28.6 ha of
loss in 146,300 ha — **one loss pixel per five thousand stable ones**. At
d = 1.36, a threshold catching half of real loss also admits ~8.7% of stable
forest, which at that base rate yields precision near 0.001. No threshold fixes
that; it is arithmetic. Raising separability is the only lever that moves it,
which is why the previous detector could only ever produce 3,840 hectares or
zero.

Two features are exposed rather than one, because they answer different
questions. Euclidean distance measures *how far* a pixel moved in embedding
space. Cosine distance measures *in what direction*, ignoring magnitude — which
matters because a whole landscape can shift together in a dry year, exactly the
common-mode effect that produced the 2023 false-positive storm.
"""

from __future__ import annotations

from typing import Final

import ee

__all__ = [
    "EMBEDDING_ASSET",
    "EMBEDDING_BANDS",
    "annual",
    "available_years",
    "change_stack",
    "cosine_distance",
    "euclidean_distance",
    "latest_available_year",
]

EMBEDDING_ASSET: Final = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

# 64 dimensions, named A00-A63 in the source.
EMBEDDING_BANDS: Final[tuple[str, ...]] = tuple(f"A{i:02d}" for i in range(64))


def available_years(geometry: ee.Geometry) -> list[int]:
    """Years for which an embedding exists over ``geometry``, ascending.

    **The embeddings lag.** They are published annually and the most recent year
    appears months after it ends, so the newest comparable pair is always at
    least a year behind today. This is the detector's binding latency
    constraint, and it is worth knowing before a run fails with an opaque
    "Image with no bands" error at the point of use.
    """
    stamps = (
        ee.ImageCollection(EMBEDDING_ASSET)
        .filterBounds(geometry)
        .aggregate_array("system:time_start")
        .getInfo()
    ) or []
    return sorted({ee.Date(stamp).get("year").getInfo() for stamp in stamps})


def latest_available_year(geometry: ee.Geometry) -> int:
    """Most recent year with an embedding over ``geometry``."""
    years = available_years(geometry)
    if not years:
        raise ValueError(
            f"no embeddings from {EMBEDDING_ASSET} cover this area — "
            "check the AOI, or the collection may not extend here"
        )
    return years[-1]


def annual(geometry: ee.Geometry, year: int) -> ee.Image:
    """The embedding image for one calendar year.

    Mosaicked because the collection is tiled and an AOI usually straddles
    several tiles — the same trap as the DEM, where loading a single image
    silently covers only part of the area.

    Raises for a year with no embedding rather than returning an empty image.
    Earth Engine's own failure here is "Band pattern 'A00' was applied to an
    Image with no bands", raised far downstream at the point of use, which says
    nothing about the actual cause: the year simply is not published yet.
    """
    collection = (
        ee.ImageCollection(EMBEDDING_ASSET)
        .filterBounds(geometry)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    )
    if collection.size().getInfo() == 0:
        available = available_years(geometry)
        raise ValueError(
            f"no satellite embedding for {year}. Available: "
            f"{available[0]}-{available[-1] if available else '?'}. "
            "AlphaEarth is published annually and lags by roughly a year, so "
            "the newest comparable pair is always behind the current date."
        )
    result: ee.Image = collection.mosaic().select(list(EMBEDDING_BANDS))
    return result


def euclidean_distance(base: ee.Image, target: ee.Image) -> ee.Image:
    """Straight-line distance between two embedding vectors.

    The strongest single feature measured (d = 2.20). Sensitive to any change in
    what a pixel is, without needing to know which dimensions encode what.
    """
    result: ee.Image = (
        base.subtract(target).pow(2).reduce(ee.Reducer.sum()).sqrt().rename("emb_euclid")
    )
    return result


def cosine_distance(base: ee.Image, target: ee.Image) -> ee.Image:
    """Angular distance between two embedding vectors, ignoring magnitude.

    Embeddings are unit-length, so the dot product is the cosine of the angle
    between them and ``1 - dot`` is a bounded dissimilarity. Useful alongside
    Euclidean distance precisely because it discards overall magnitude: a
    landscape-wide shift, like the 2023 monsoon failure, moves every vector's
    length together while leaving directions largely intact.
    """
    result: ee.Image = (
        ee.Image.constant(1)
        .subtract(base.multiply(target).reduce(ee.Reducer.sum()))
        .rename("emb_cosine")
    )
    return result


def change_stack(geometry: ee.Geometry, base_year: int, target_year: int) -> ee.Image:
    """Feature image for classifying change between two years.

    Three groups, each earning its place:

    * ``A00``-``A63`` — the **baseline** embedding. Tells the classifier what
      the pixel *was*, which is how plantation gets separated from natural
      forest. A difference alone cannot: a harvested plantation and a cleared
      forest move similarly, and only the starting state distinguishes them.
      This is the project's largest measured error source.
    * ``D00``-``D63`` — the per-dimension **difference**. Direction of change,
      not just magnitude.
    * ``emb_euclid``, ``emb_cosine`` — summary distances, giving the classifier
      the aggregate signal directly rather than making it rediscover it.
    """
    if base_year >= target_year:
        raise ValueError(
            f"base_year ({base_year}) must be earlier than target_year ({target_year})"
        )

    base = annual(geometry, base_year)
    target = annual(geometry, target_year)

    difference = target.subtract(base).rename([f"D{i:02d}" for i in range(64)])

    result: ee.Image = (
        base.addBands(difference)
        .addBands(euclidean_distance(base, target))
        .addBands(cosine_distance(base, target))
    )
    return result
