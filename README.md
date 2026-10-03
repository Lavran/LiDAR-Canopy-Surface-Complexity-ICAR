# Rainier Canopy Complexity

Airborne-LiDAR analysis of canopy surface complexity (rumple) and its relationship to
canopy height, elevation, and topographic heat load across a forested footprint in
eastern Mount Rainier National Park, with a Bayesian intrinsic conditional
autoregressive (ICAR) spatial model separating covariate effects from residual spatial
structure.

**Manuscript:** [LiDAR-Canopy-Surface-Complexity.html](https://lavran.github.io/LiDAR-Canopy-Surface-Complexity-ICAR/LiDAR-Canopy-Surface-Complexity.html)
(source: `LiDAR-Canopy-Surface-Complexity.qmd`)

## Pipeline

Run the scripts in numeric order. All paths are set in `config.py`; `grid.py` and
`utils.py` are shared helpers.

| Script | What it does |
|---|---|
| `01_clean_pointcloud.py` | Removes outliers and noise; normalizes heights to height above ground |
| `02_terrain.py` | 30 m terrain raster: elevation, slope, aspect |
| `03_canopy_metrics.py` | 30 m canopy metrics: p95 height, canopy relief ratio, cover |
| `04_pitfree_chm.py` | 1 m pit-free canopy height model |
| `05_rumple.py` | 30 m canopy surface complexity (rumple) from the canopy height model |
| `06_assemble_dataset.py` | Builds the analysis table, `metrics_spatial.pkl` |
| `07_nonspatial_models.py` | Nonspatial Bayesian regressions; residual Moran's I and semivariogram |
| `08_icar.py` | ICAR spatial model, the main result (about 37 min) |
| `09_spatial_diagnostics.py` | Moran's I and semivariogram of the ICAR residuals |
| `10_acquisition_check.py` | Checks whether the spatial field lines up with LiDAR tile or processing-chunk edges |

Steps 01–06 need the raw LiDAR. If you already have `metrics_spatial.pkl`, start at 07
(10 also reads the LiDAR file headers). To rebuild the manuscript, run
`quarto render LiDAR-Canopy-Surface-Complexity.qmd`; it needs the rasters from 02–05.

## Setup

```
conda env create -f environment.yml
conda activate rainier-canopy
```

The environment pins OpenBLAS on purpose: MKL-built packages add a second OpenMP
runtime, which crashes the nutpie sampler (seen on Windows).

## Data

LiDAR: USGS 3DEP *WA Eastern Cascades 2019 B19* (EPSG:6339), downloaded from the
[Washington State DNR LiDAR Portal](https://lidarportal.dnr.wa.gov/). Put the `.laz`
tiles in `Rainier_lidar_2019/point_cloud/`.

The point clouds, rasters, `metrics_spatial.pkl` and the saved posterior are too large
for the repo; the scripts regenerate them. The repo includes the scripts, the
manuscript, the per-cell spatial field (`spatial_field_icar.csv`) and the summary
tables.

## Reproducibility

Model fits and variograms use a fixed seed (1738). With the same package versions on
the same machine, reruns reproduce the results; MCMC draws can differ across operating
systems, BLAS libraries or thread counts.

## License

MIT (see `LICENSE`). © 2026 Lavran Pagano.
