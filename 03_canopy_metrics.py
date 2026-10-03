"""
03_canopy_metrics.py
Per-cell canopy structure at 30 m from the first returns of the cleaned, height-
normalized point cloud: 95th-percentile height (p95), canopy relief ratio (CRR) and
canopy cover (share of first returns above 2 m). Processed in 1,200 m chunks anchored
to the survey lattice, then mosaicked to one 3-band raster (p95, crr, cover).
"""
import os
import glob
import numpy as np
import pandas as pd
import laspy
import rasterio
from rasterio.transform import from_origin

import config
from grid import survey_grid
from utils import mosaic_rasters

g = survey_grid()
GRID_LEFT, GRID_TOP, GRID_RIGHT, GRID_BOTTOM = g["left"], g["top"], g["right"], g["bottom"]


def canopy_cover_metrics(in_folder, out_folder, grid_left=GRID_LEFT, grid_top=GRID_TOP,
                         grid_right=GRID_RIGHT, grid_bottom=GRID_BOTTOM,
                         cell=config.RES_30, chunk_size=1200.0, crs=config.CRS):
    os.makedirs(str(out_folder), exist_ok=True)
    files = []
    for path in glob.glob(os.path.join(str(in_folder), "*.laz")):
        with laspy.open(path) as f:
            xmin, ymin = f.header.mins[:2]
            xmax, ymax = f.header.maxs[:2]
        files.append({"path": path, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    n_chunk_cols = int(np.ceil((grid_right - grid_left) / chunk_size))
    n_chunk_rows = int(np.ceil((grid_top - grid_bottom) / chunk_size))

    for rr in range(n_chunk_rows):
        top = grid_top - rr * chunk_size
        bottom = max(top - chunk_size, grid_bottom)
        for cc in range(n_chunk_cols):
            left = grid_left + cc * chunk_size
            right = min(left + chunk_size, grid_right)
            sources = [f["path"] for f in files
                       if (f["xmax"] > left and f["xmin"] < right and
                           f["ymax"] > bottom and f["ymin"] < top)]
            if not sources:
                continue
            xs, ys, zs = [], [], []
            for path in sources:
                las = laspy.read(path)
                first = las.return_number == 1
                x = np.asarray(las.x)[first]; y = np.asarray(las.y)[first]; z = np.asarray(las.z)[first]
                inside = (x >= left) & (x < right) & (y > bottom) & (y <= top)
                if inside.any():
                    xs.append(x[inside]); ys.append(y[inside]); zs.append(z[inside])
            if not xs:
                continue
            x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)

            ncols = int(round((right - left) / cell))
            nrows = int(round((top - bottom) / cell))
            col = ((x - left) / cell).astype(int)
            row = ((top - y) / cell).astype(int)
            valid = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
            col = col[valid]; row = row[valid]; z = z[valid]
            cid = row * ncols + col

            df = pd.DataFrame({"cid": cid, "z": z})
            df["tall"] = df["z"] > 2
            gb = df.groupby("cid")
            zmin, zmax = gb["z"].min(), gb["z"].max()
            m = pd.DataFrame({
                "p95": gb["z"].quantile(0.95),
                "crr": (gb["z"].mean() - zmin) / (zmax - zmin),
                "cover": gb["tall"].mean(),
            })
            # CRR is undefined where every return in the cell sits at one height
            # (zmax == zmin -> 0/0). Those cells are left NoData; all of them fall
            # inside the larger rumple NoData mask, so none reach the analysis table.
            m["crr"] = m["crr"].replace([np.inf, -np.inf], np.nan)

            bands = ["p95", "crr", "cover"]
            stack = np.full((len(bands), nrows * ncols), -9999.0, dtype="float32")
            for b, name in enumerate(bands):
                good = m[name].notna()
                stack[b, m.index[good]] = m.loc[good, name].to_numpy(dtype="float32")
            stack = stack.reshape(len(bands), nrows, ncols)

            out_tif = os.path.join(str(out_folder), f"metrics_r{rr:03d}_c{cc:03d}.tif")
            transform = from_origin(left, top, cell, cell)
            with rasterio.open(out_tif, "w", driver="GTiff", height=nrows, width=ncols,
                               count=len(bands), dtype="float32", crs=crs,
                               transform=transform, nodata=-9999, compress="deflate") as dst:
                dst.write(stack)
                for b, name in enumerate(bands, start=1):
                    dst.set_band_description(b, name)
            print(f"wrote {os.path.basename(out_tif)} | {len(z):,} first returns")


if __name__ == "__main__":
    canopy_cover_metrics(config.LAS_CLEAN, config.CANOPY_DIR)
    mosaic_rasters(config.CANOPY_DIR, config.CANOPY_PATH)
