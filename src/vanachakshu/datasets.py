"""Earth Engine asset identifiers, in one place.

Every dataset id the project depends on is named here rather than inlined at the
call site. Asset ids are versioned and do change — Hansen ships a new version
annually, NICFI basemaps roll forward monthly — and hunting a hard-coded string
through a codebase after an upstream bump is exactly the kind of avoidable work
this file prevents.

Each entry records where it came from and why that version, so the next person
(including you in three months) can tell whether a bump is safe.
"""

from __future__ import annotations

from typing import Final

# --- Optical -----------------------------------------------------------------
# Sentinel-2 Level-2A surface reflectance, "harmonized" variant. Harmonized
# matters: ESA shifted the radiometric offset in January 2022, and the plain
# S2_SR collection therefore has a step change in reflectance values across that
# date. Using the harmonized collection means a 2021-vs-2024 comparison measures
# forest change rather than a processing-baseline change.
SENTINEL2_SR: Final = "COPERNICUS/S2_SR_HARMONIZED"

# Cloud probability, produced by the s2cloudless model. Joined to Sentinel-2 by
# image id; more reliable than the scene classification (SCL) band alone,
# particularly for thin cirrus over tropical forest.
SENTINEL2_CLOUD_PROBABILITY: Final = "COPERNICUS/S2_CLOUD_PROBABILITY"

# --- Radar -------------------------------------------------------------------
# Sentinel-1 ground range detected. Phase 3's primary sensor: it sees through
# monsoon cloud, which optical cannot.
SENTINEL1_GRD: Final = "COPERNICUS/S1_GRD"

# --- Labels and masks --------------------------------------------------------
# Hansen Global Forest Change v1.13, covering 2000-2025. Supplies both the
# forest mask (treecover2000) and the training/validation labels (lossyear).
# Bump deliberately when v1.14 lands: the loss-year encoding is relative to the
# collection's own end year, so a version bump changes label semantics.
HANSEN_GFC: Final = "UMD/hansen/global_forest_change_2025_v1_13"

# ESA WorldCover 10 m land cover, v200 (2021 epoch). Used to exclude built-up,
# water and cropland from the forest mask.
ESA_WORLDCOVER: Final = "ESA/WorldCover/v200"

# --- Validation --------------------------------------------------------------
# NICFI Planet basemaps, Asia tile set: <5 m imagery, free for noncommercial
# use. This is the independent validation source for Phase 4 — checking against
# imagery rather than against Hansen, which is itself a model.
NICFI_BASEMAPS_ASIA: Final = "projects/planet-nicfi/assets/basemaps/asia"

# --- Ancillary ---------------------------------------------------------------
# CHIRPS daily precipitation. Needed for RESTREND in the degradation layer, to
# separate land-use degradation from ordinary drought.
CHIRPS_DAILY: Final = "UCSB-CHG/CHIRPS/DAILY"

# Digital elevation model used for radiometric terrain correction in Phase 3.
# Copernicus GLO-30 is newer and cleaner than SRTM over steep Western Ghats
# terrain, where layover and shadow are the dominant radar problem.
COPERNICUS_DEM: Final = "COPERNICUS/DEM/GLO30"
