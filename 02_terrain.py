"""
02_terrain.py
Per-tile 30 m DTM from ground (class 2) returns, mosaicked, then slope + aspect +
elevation as a 3-band terrain raster. Runs on the RAW tiles (ground returns are
still classified there).
"""
import os
import glob
import json
import numpy as np
import laspy
import pdal
import rasterio
from scipy.ndimage import distance_transform_edt

import config
from grid import survey_grid
from utils import mosaic_rasters

g = survey_grid()
GRID_LEFT, GRID_BOTTOM = g["left"], g["bottom"]


def dtm_per_tile(in_folder, out_folder, grid_left=GRID_LEFT, grid_bottom=GRID_BOTTOM,
                 res=config.RES_30, crs=config.CRS):
    os.makedirs(str(out_folder), exist_ok=True)
    for i in glob.glob(os.path.join(str(in_folder), "*.laz")):
        out_tif = os.path.join(str(out_folder),
                               os.path.splitext(os.path.basename(i))[0] + "_dtm.tif")
        with laspy.open(i) as f:
            xmin, ymin = f.header.mins[:2]
            xmax, ymax = f.header.maxs[:2]
        # snap this tile's extent onto the survey-wide 30 m lattice
        left   = grid_left + np.floor((xmin - grid_left) / res) * res
        right  = grid_left + np.ceil((xmax - grid_left) / res) * res
        bottom = grid_bottom + np.floor((ymin - grid_bottom) / res) * res
        top    = grid_bottom + np.ceil((ymax - grid_bottom) / res) * res
        width  = int(round((right - left) / res))
        height = int(round((top - bottom) / res))
        pipeline = {"pipeline": [
            {"type": "readers.las", "filename": i},
            {"type": "filters.expression", "expression": "Classification == 2"},
            {"type": "writers.gdal", "filename": out_tif, "resolution": res,
             "origin_x": left, "origin_y": bottom, "width": width, "height": height,
             "output_type": "idw", "window_size": 3, "gdaldriver": "GTiff",
             "nodata": -9999, "override_srs": crs},
        ]}
        pdal.Pipeline(json.dumps(pipeline)).execute()
        print(f"wrote {os.path.basename(out_tif)} | {width} x {height} cells | "
              f"origin=({left:.1f}, {bottom:.1f})")


def terrain_metrics(dem_path, out_tif):
    with rasterio.open(str(dem_path)) as src:
        dem  = src.read(1).astype("float32")
        meta = src.meta.copy()
        nod  = src.nodata
        cell = src.transform.a                        # pixel size (m)
    dem[dem == nod] = np.nan
    valid = ~np.isnan(dem)
    # nearest-neighbour fill so gradients don't smear NaNs, then differentiate
    if (~valid).any():
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        dem_filled = dem[tuple(idx)]
    else:
        dem_filled = dem
    grad_row, grad_col = np.gradient(dem_filled, cell)
    fx, fy = grad_col, -grad_row
    slope  = np.degrees(np.arctan(np.hypot(fx, fy)))
    aspect = np.degrees(np.arctan2(-fx, -fy)) % 360
    # mask back to the observed extent
    elev   = np.where(valid, dem_filled, -9999).astype("float32")
    slope  = np.where(valid, slope,      -9999).astype("float32")
    aspect = np.where(valid, aspect,     -9999).astype("float32")
    bands = {"elevation": elev, "slope": slope, "aspect": aspect}
    meta.update(count=3, dtype="float32", nodata=-9999)
    os.makedirs(os.path.dirname(str(out_tif)) or ".", exist_ok=True)
    with rasterio.open(str(out_tif), "w", **meta) as dst:
        dst.write(np.stack(list(bands.values())))
        for b, name in enumerate(bands, 1):
            dst.set_band_description(b, name)
    print(f"wrote {os.path.basename(str(out_tif))}")


if __name__ == "__main__":
    dtm_per_tile(config.LAS_RAW, config.DTM_DIR)
    mosaic_rasters(config.DTM_DIR, config.DTM_PATH)
    terrain_metrics(config.DTM_PATH, config.TERRAIN_PATH)
