"""
06_assemble_dataset.py
Assemble the modeling table from the aligned 30 m rasters: read terrain, canopy, and
rumple onto the common grid (refusing any misaligned input), add heat load, restrict
to the structural forest mask and then to the largest connected component of the 45 m
neighbour graph, standardize predictors, and save metrics_spatial.pkl -- the exact
dataframe the Bayesian models consume.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import config
from utils import heat_load

# reference grid = terrain raster
with rasterio.open(str(config.TERRAIN_PATH)) as ref:
    H, W = ref.shape
    dst_transform, dst_crs = ref.transform, ref.crs


def read_aligned(path, band):
    """Read a raster band after confirming it matches the 30 m reference grid exactly."""
    with rasterio.open(str(path)) as src:
        if src.shape != (H, W):
            raise ValueError(f"Grid shape mismatch for {path}: {src.shape} != {(H, W)}")
        if src.crs != dst_crs:
            raise ValueError(f"CRS mismatch for {path}: {src.crs} != {dst_crs}")
        if not src.transform.almost_equals(dst_transform):
            raise ValueError(f"Grid transform mismatch for {path}: {src.transform} != {dst_transform}")
        return src.read(band, masked=True).filled(np.nan).astype("float32").ravel()


# Canopy band order is set by 03_canopy_metrics.py: ["p95", "crr", "cover"].
# CRR is read here so the analysis table is complete and regenerable; it is NOT a
# model covariate (Spearman 0.84 with cover and 0.67 with p95). It is used only
# as a diagnostic against the fitted spatial field in the write-up.
metrics = pd.DataFrame({
    "elevation": read_aligned(config.TERRAIN_PATH, 1),
    "slope":     read_aligned(config.TERRAIN_PATH, 2),
    "aspect":    read_aligned(config.TERRAIN_PATH, 3),
    "p95":       read_aligned(config.CANOPY_PATH, 1),
    "crr":       read_aligned(config.CANOPY_PATH, 2),
    "cover":     read_aligned(config.CANOPY_PATH, 3),
    "rumple":    read_aligned(config.RUMPLE_PATH, 1),
})
r, c = np.divmod(np.arange(H * W), W)
metrics["x"], metrics["y"] = xy(dst_transform, r, c)
metrics = metrics.dropna().reset_index(drop=True)
metrics["heat_load"] = heat_load(metrics["slope"], metrics["aspect"],
                                 latitude=config.LATITUDE, equation=3)

# structural forest mask (cover + height components of the FAO definition)
forest = (metrics["cover"] >= config.MIN_COVER) & (metrics["p95"] >= config.MIN_HEIGHT)
print(f"{len(metrics):,} cells -> {forest.sum():,} forested "
      f"({100*forest.mean():.0f}%); dropped {100*(1-forest.mean()):.0f}% non-forest")
metrics = metrics[forest].reset_index(drop=True)

# Restrict to the largest connected component of the 45 m neighbour graph.
#
# The ICAR prior in 08 is improper along the constant vector of each connected
# component, so it needs a connected graph. The alternative -- linking every stray
# component to its nearest cell regardless of distance -- fabricates adjacencies
# between cells that are hundreds of metres apart, and the prior then smooths across
# them as though they were neighbours. Dropping the strays instead costs a small
# fraction of cells and invents nothing.
#
# This happens BEFORE standardization so that "per SD" in the results refers to the
# spread of the cells actually modelled, and so metrics_spatial.pkl is exactly the
# analysis set.
coords = metrics[["x", "y"]].to_numpy()
pairs = cKDTree(coords).query_pairs(r=config.NEIGHBOUR_DIST, output_type="ndarray")
adjacency = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(metrics),) * 2)
n_comp, labels = connected_components(adjacency + adjacency.T, directed=False)
sizes = np.bincount(labels)
keep = labels == sizes.argmax()
print(f"{n_comp} connected components at {config.NEIGHBOUR_DIST:.0f} m "
      f"({(sizes == 1).sum()} isolated cells); keeping the largest: "
      f"{keep.sum():,} cells ({100 * keep.mean():.2f}%), dropping {(~keep).sum():,}")
metrics = metrics[keep].reset_index(drop=True)

# standardize predictors on the analysed subset
for v in ["heat_load", "elevation", "slope", "p95"]:
    metrics[v + "_z"] = (metrics[v] - metrics[v].mean()) / metrics[v].std()

metrics.to_pickle(str(config.METRICS_PKL))
print(f"saved {config.METRICS_PKL.name} ({len(metrics):,} cells)")
