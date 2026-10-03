"""
04_pitfree_chm.py
Pit-free 1 m canopy height model from the cleaned point cloud, following the layered-
TIN logic of Khosravipour et al. (2014): first returns stratified by height thresholds,
each triangulated with a max-edge trim, highest supported interpolated height retained.
Then mosaicked to a single 1 m CHM.
"""
import os
import glob
import gc
import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import Delaunay, QhullError

import config
from grid import survey_grid
from utils import mosaic_rasters

g = survey_grid()
GRID_LEFT, GRID_TOP, GRID_RIGHT, GRID_BOTTOM = g["left"], g["top"], g["right"], g["bottom"]


def pitfree_CHM(in_folder, out_folder, grid_left=GRID_LEFT, grid_top=GRID_TOP,
                grid_right=GRID_RIGHT, grid_bottom=GRID_BOTTOM, res=config.RES_1,
                chunk_size=600.0, buffer=20.0, thresholds=(0, 2, 5, 10, 15, 25),
                base_max_edge=5.0, upper_max_edge=1.5, crs=config.CRS, overwrite=False):
    os.makedirs(str(out_folder), exist_ok=True)
    files = []
    for path in glob.glob(os.path.join(str(in_folder), "*.laz")):
        with laspy.open(path) as f:
            xmin, ymin, _ = f.header.mins
            xmax, ymax, _ = f.header.maxs
        files.append({"path": path, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    if not np.isclose(chunk_size / res, round(chunk_size / res)):
        raise ValueError("chunk_size must be evenly divisible by res")

    n_chunk_cols = int(np.ceil((grid_right - grid_left) / chunk_size))
    n_chunk_rows = int(np.ceil((grid_top - grid_bottom) / chunk_size))

    for rr in range(n_chunk_rows):
        top = grid_top - rr * chunk_size
        bottom = max(top - chunk_size, grid_bottom)
        for cc in range(n_chunk_cols):
            left = grid_left + cc * chunk_size
            right = min(left + chunk_size, grid_right)
            out_tif = os.path.join(str(out_folder), f"pitfree_r{rr:03d}_c{cc:03d}.tif")
            if os.path.exists(out_tif) and not overwrite:
                print("skip", os.path.basename(out_tif)); continue

            bleft, bright = left - buffer, right + buffer
            bbottom, btop = bottom - buffer, top + buffer
            sources = [f["path"] for f in files
                       if (f["xmax"] >= bleft and f["xmin"] <= bright and
                           f["ymax"] >= bbottom and f["ymin"] <= btop)]
            if not sources:
                continue

            xs, ys, zs = [], [], []
            for path in sources:
                las = laspy.read(path)
                first = las.return_number == 1
                x = np.asarray(las.x)[first]; y = np.asarray(las.y)[first]; z = np.asarray(las.z)[first]
                inside = (x >= bleft) & (x < bright) & (y >= bbottom) & (y < btop)
                if inside.any():
                    xs.append(x[inside]); ys.append(y[inside]); zs.append(z[inside])
                del las
            if not xs:
                continue
            x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
            z = np.maximum(z, 0)                      # clamp tiny negative heights

            # highest first return per GLOBAL 1 m cell
            global_col = np.floor((x - grid_left) / res).astype(np.int64)
            global_row = np.floor((grid_top - y) / res).astype(np.int64)
            global_ncols = int(np.ceil((grid_right - grid_left) / res))
            cell_id = global_row * global_ncols + global_col
            order = np.lexsort((z, cell_id))
            ids_sorted = cell_id[order]
            keep = np.r_[ids_sorted[1:] != ids_sorted[:-1], True]
            idx = order[keep]
            x, y, z = x[idx], y[idx], z[idx]

            # output grid = UNBUFFERED chunk
            cols = int(round((right - left) / res))
            rows = int(round((top - bottom) / res))
            x_centers = left + (np.arange(cols) + 0.5) * res
            y_centers = top - (np.arange(rows) + 0.5) * res
            gx, gy = np.meshgrid(x_centers, y_centers)
            grid_xy = np.column_stack((gx.ravel(), gy.ravel()))
            result = np.full(len(grid_xy), np.nan, dtype=np.float32)

            for threshold in thresholds:
                sel = z >= threshold
                if sel.sum() < 3:
                    continue
                xy = np.column_stack((x[sel], y[sel])); zz = z[sel]
                try:
                    tri = Delaunay(xy)
                except QhullError:
                    print(f"Qhull failed: chunk {rr},{cc}, threshold {threshold}")
                    continue
                vertices = xy[tri.simplices]
                e1 = np.linalg.norm(vertices[:, 0] - vertices[:, 1], axis=1)
                e2 = np.linalg.norm(vertices[:, 1] - vertices[:, 2], axis=1)
                e3 = np.linalg.norm(vertices[:, 2] - vertices[:, 0], axis=1)
                longest = np.maximum.reduce((e1, e2, e3))
                good_triangle = longest <= (base_max_edge if threshold == 0 else upper_max_edge)

                sid = tri.find_simplex(grid_xy)
                inside = sid >= 0                     # -1 means outside every simplex
                if not inside.any():
                    del tri; continue
                pix = np.flatnonzero(inside); simplex = sid[inside]
                okay = good_triangle[simplex]
                pix = pix[okay]; simplex = simplex[okay]
                if len(pix) == 0:
                    del tri; continue
                pts = grid_xy[pix]
                X = tri.transform[simplex, :2]
                Y = pts - tri.transform[simplex, 2]
                bary12 = np.einsum("pij,pj->pi", X, Y)
                bary = np.column_stack((bary12, 1 - bary12.sum(axis=1)))
                z_interp = np.einsum("pi,pi->p", bary, zz[tri.simplices[simplex]])
                result[pix] = np.fmax(result[pix], z_interp.astype(np.float32))
                del tri, vertices, X, Y, bary, z_interp
                gc.collect()

            chm = result.reshape(rows, cols)
            transform = from_origin(left, top, res, res)
            with rasterio.open(out_tif, "w", driver="GTiff", height=rows, width=cols,
                               count=1, dtype="float32", crs=crs, transform=transform,
                               nodata=-9999, compress="deflate") as dst:
                dst.write(np.where(np.isfinite(chm), chm, -9999).astype("float32"), 1)
                dst.set_band_description(1, "pitfree_chm")
            print(f"wrote {os.path.basename(out_tif)} | {len(x):,} retained first returns")
            del x, y, z, grid_xy, gx, gy, result, chm
            gc.collect()


if __name__ == "__main__":
    pitfree_CHM(config.LAS_CLEAN, config.CHM_DIR)
    mosaic_rasters(config.CHM_DIR, config.CHM_PATH, res=config.RES_1, pattern="*.tif")
