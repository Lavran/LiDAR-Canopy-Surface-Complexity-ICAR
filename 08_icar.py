"""
08_icar.py
----------
Rumple regressed on canopy height, elevation and topographic heat load, with an
intrinsic conditional autoregressive (ICAR) spatial random effect over the 30 m
analysis lattice.

The spatial prior is pm.ICAR. Its log density, per the PyMC documentation, is

    log p(phi | W, sigma) = -1/(2 sigma^2) sum_{i~j} (phi_i - phi_j)^2
                            - 0.5 (sum_i phi_i / (0.001N))^2
                            - ln sqrt(2 pi) - ln(0.001N)

PyMC cites Morris et al. (2019), Spatial and Spatio-temporal Epidemiology 31:100301,
and Banerjee, Carlin & Gelfand (2015), Hierarchical Modeling and Analysis for Spatial
Data, 2nd ed.

The model follows the non-centred parameterisation from pm.ICAR's own docstring:
phi is drawn with ICAR sigma fixed at 1 and multiplied by a separately estimated
spatial scale, sd_spatial. Because an unscaled ICAR does not give every cell a
marginal SD of exactly 1, sd_spatial is best interpreted as a scale parameter.

Package versions and the number of divergent transitions are printed at run time, and
the per-draw divergence flag is saved with the posterior in spatial_fit_icar.nc.

Run:  python 08_icar.py
"""

import os
# Must be set BEFORE numpy / nutpie import. Windows conda can load two OpenMP
# runtimes at once (Intel libiomp via MKL, LLVM libomp via nutpie), which crashes
# hard (exit -1073740791). See also the libblas pin in environment.yml.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import nutpie  # imported for its version string; pm.sample drives it
from scipy.spatial import cKDTree

import config

print("versions | python", sys.version.split()[0], "| pymc", pm.__version__,
      "| nutpie", nutpie.__version__, "| arviz", az.__version__, "| numpy", np.__version__)

PREDICTORS = ["p95_z", "elevation_z", "heat_load_z"]
DRAWS, TUNE, CHAINS, CORES, TARGET_ACCEPT = 8000, 2000, 4, 2, 0.9
SEED = 1738

metrics = pd.read_pickle(config.METRICS_PKL).reset_index(drop=True)
y = metrics["rumple"].to_numpy()
X = metrics[PREDICTORS].to_numpy()
coords = metrics[["x", "y"]].to_numpy()
N = len(metrics)

# Adjacency: 1 where two cell centres are within 45 m of each other, which is queen
# contiguity (rook + diagonal) on a 30 m lattice. 06 has already restricted the table
# to one connected component, which ICAR requires.
pairs = cKDTree(coords).query_pairs(r=config.NEIGHBOUR_DIST, output_type="ndarray")
W = np.zeros((N, N), dtype=int)
W[pairs[:, 0], pairs[:, 1]] = 1
W[pairs[:, 1], pairs[:, 0]] = 1
print(f"{N:,} cells | {len(pairs):,} edges | mean degree {W.sum() / N:.2f}")

with pm.Model(coords={"predictor": PREDICTORS}) as model:
    intercept = pm.Normal("Intercept", 2.0, 1.0)
    beta = pm.Normal("beta", 0.0, 1.0, dims="predictor")
    sd_spatial = pm.HalfNormal("sd_spatial", 1.0)
    sigma = pm.HalfNormal("sigma", 0.5)

    phi = pm.ICAR("phi", W=W)

    mu = intercept + X @ beta + sd_spatial * phi
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

    idata = pm.sample(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        cores=CORES,
        nuts_sampler="nutpie",
        target_accept=TARGET_ACCEPT,
        random_seed=SEED,
        nuts_sampler_kwargs={"adaptation": "low_rank"},
    )

n_div = int(idata.sample_stats["diverging"].sum())
print(f"divergent transitions: {n_div}")

summary = az.summary(idata, var_names=["Intercept", "beta", "sd_spatial", "sigma"])
print(summary)
summary.to_csv(config.PROJECT_DIR / "spatial_summary_icar.csv")

# Posterior-mean spatial field and residuals, for mapping and diagnostics in 09.
post = idata.posterior

# Average the product within each posterior draw. This estimates
# E[sd_spatial * phi_i], rather than E[sd_spatial] * E[phi_i]; the latter ignores
# posterior dependence between sd_spatial and phi.
spatial = (
    post["sd_spatial"] * post["phi"]
).mean(dim=("chain", "draw")).values

intercept_hat = post["Intercept"].mean().item()
beta_hat = post["beta"].mean(dim=("chain", "draw")).values
resid = y - (intercept_hat + X @ beta_hat + spatial)

pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
              "resid_icar": resid, "spatial_effect": spatial}) \
    .to_csv(config.PROJECT_DIR / "spatial_field_icar.csv", index=False)

# phi is N-long per draw (>1 GB over the full trace), so it is dropped before saving.
# nutpie also attaches its settings as a dict attribute, which newer xarray refuses to
# write to netCDF; clearing attrs sidesteps that.
posterior = idata.posterior[[v for v in post.data_vars if v != "phi"]].copy()
posterior.attrs = {}
# Keep the per-draw divergence flag so the divergence count can be checked from the
# saved file; the rest of sample_stats is not needed.
diverging = idata.sample_stats[["diverging"]].copy()
diverging.attrs = {}
az.InferenceData(posterior=posterior, sample_stats=diverging).to_netcdf(
    config.PROJECT_DIR / "spatial_fit_icar.nc")
print("saved spatial_summary_icar.csv, spatial_field_icar.csv, spatial_fit_icar.nc")
