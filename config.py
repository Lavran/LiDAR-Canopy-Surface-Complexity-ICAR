"""
config.py -- shared paths and constants for the Rainier canopy-complexity pipeline.

Every script imports paths from here so there is ONE place to change the project
location. This module is cheap (no heavy imports, no data reads), so importing it
from the modeling scripts does not require laspy / pdal / the raw LiDAR.
"""
from pathlib import Path

# Project root = the folder containing this file, so the repo works wherever it is cloned.
PROJECT_DIR = Path(__file__).resolve().parent

# --- coordinate reference + analysis grains ---
CRS      = "EPSG:6339"   # NAD83(2011) / UTM Zone 10N
LATITUDE = 46.85         # degrees N, for the McCune & Keon heat-load index
RES_30   = 30.0          # analysis grid resolution (m)
RES_1    = 1.0           # canopy height model resolution (m)

# Neighbour distance for the areal spatial model and for Moran's I. 45 m is queen
# contiguity on a 30 m lattice (rook + diagonal). Used by 06 (connected-component
# filter), 07 and 09 (Moran's I), 08 (ICAR graph) and 11 (neighbour pairs at tile
# seams) -- keep them on one definition.
NEIGHBOUR_DIST = 45.0

# --- LiDAR + raster products ---
LIDAR_DIR    = PROJECT_DIR / "Rainier_lidar_2019"
LAS_RAW      = LIDAR_DIR / "point_cloud"
LAS_CLEAN    = LIDAR_DIR / "point_cloud_clean"

DTM_DIR      = LIDAR_DIR / "DTM"
DTM_PATH     = DTM_DIR / "Rainier_lidar_2019_DTM.tiff"
TERRAIN_PATH = DTM_DIR / "Rainier_lidar_2019_terrain_metrics.tiff"

CANOPY_DIR   = LIDAR_DIR / "Canopy_metrics"
CANOPY_PATH  = CANOPY_DIR / "Rainier_lidar_2019_canopy_metrics.tiff"

CHM_DIR      = LIDAR_DIR / "Khosravipour_et_al_CHM"
CHM_PATH     = CHM_DIR / "Rainier_lidar_2019_Khosravipour_CHM.tiff"

RUMPLE_PATH  = LIDAR_DIR / "Rumple" / "pit free" / "rumple_pitfree.tiff"

# --- modeling artifacts ---
METRICS_PKL  = PROJECT_DIR / "metrics_spatial.pkl"

# --- point-cloud cleaning parameters ---
OUTLIER_K          = 8      # k nearest neighbours for the statistical outlier filter
OUTLIER_MULTIPLIER = 2.5    # std-dev multiplier

# --- structural forest mask (cover + height components of the FAO definition) ---
MIN_COVER  = 0.10   # >= 10% LiDAR canopy cover
MIN_HEIGHT = 5.0    # >= 5 m p95 height
