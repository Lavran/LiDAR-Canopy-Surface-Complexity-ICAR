"""
grid.py -- the survey-wide 30 m analysis lattice, derived once from the raw LiDAR
tile bounds. Every raster product is anchored to this lattice so that canopy,
terrain, and rumple share a common origin and the 1 m CHM grid nests exactly inside
the 30 m grid. Only the preprocessing steps (02-04) need this.
"""
import glob
import os
import numpy as np
import laspy
from rasterio.transform import from_origin

import config


def survey_grid(res_30=config.RES_30, res_1=config.RES_1):
    """Return the survey-aligned grid extent and affine transforms."""
    tiles = glob.glob(os.path.join(str(config.LAS_RAW), "*.laz"))
    if not tiles:
        raise FileNotFoundError(f"No .laz tiles found in {config.LAS_RAW}")

    bounds = []
    for f in tiles:
        with laspy.open(f) as las:
            bounds.append([
                las.header.mins[0], las.header.mins[1],
                las.header.maxs[0], las.header.maxs[1],
            ])
    bounds = np.asarray(bounds)

    left   = np.floor(bounds[:, 0].min() / res_30) * res_30
    bottom = np.floor(bounds[:, 1].min() / res_30) * res_30
    right  = np.ceil(bounds[:, 2].max() / res_30) * res_30
    top    = np.ceil(bounds[:, 3].max() / res_30) * res_30

    return {
        "left": left, "bottom": bottom, "right": right, "top": top,
        "transform_30": from_origin(left, top, res_30, res_30),
        "transform_1":  from_origin(left, top, res_1, res_1),
    }
