"""
07_nonspatial_models.py
The non-spatial Bayesian regression sequence (marginal -> conditional) plus the
pre-spatial residual diagnostics that motivate the spatial model: Moran's I on the
full-model residuals and a spherical semivariogram.

skgstat subsamples 3,000 points to build the variogram, so the nugget and sill move on
every run unless the draw is fixed. Its Variogram class swallows unknown keywords, so
passing `random_state=` does nothing -- the subsampling reads numpy's GLOBAL random
state (see skgstat.MetricSpace). Seeding numpy immediately before constructing the
Variogram is what actually fixes the draw. 09_spatial_diagnostics.py uses the same seed,
so the before/after variogram comparison rests on the same subsample.
"""
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
from libpysal.weights import DistanceBand
from esda.moran import Moran
import skgstat as skg

import config

SEED = 1738

metrics = pd.read_pickle(str(config.METRICS_PKL))
y_obs = metrics["rumple"].to_numpy()


# --- marginal + conditional model sequence ---
specs = {
    "m1_heat":            "rumple ~ heat_load_z",
    "m2_elev":            "rumple ~ elevation_z",
    "m3_p95":             "rumple ~ p95_z",
    "m4_elev_heat":       "rumple ~ elevation_z + heat_load_z",
    "m5_p95_heat":        "rumple ~ p95_z + heat_load_z",
    "m6_p95_elev_heat":   "rumple ~ p95_z + elevation_z + heat_load_z",
}
# The sequence is fit in the SAME Bayesian framework as the spatial model on purpose:
# the point is to compare posterior HDIs across spatial and non-spatial specifications,
# which is not a comparison OLS standard errors support without extra argument. At
# n = 21,897 with Bambi's default priors the posterior means are indistinguishable from
# least squares; it is the intervals, not the point estimates, that carry the argument.
#
# Explained variance is reported as residual sigma rather than Bayesian R^2. A per-cell
# spatial field makes conditional R^2 near-1 by construction, so the quantity is not
# comparable across the sequence and the spatial model, and a number that has to be
# disclaimed in the text is worse than one that is not reported.
summaries = []
models, fits = {}, {}
for name, formula in specs.items():
    models[name] = bmb.Model(formula, metrics)            # keep the Model object: predict()
    fits[name] = models[name].fit(draws=1000, tune=1000,  # must be called on the SAME instance
                                  chains=4, cores=1, random_seed=SEED)
    print(f"\n=== {name}: {formula} ===")
    summary = az.summary(fits[name])
    print(summary)
    summaries.append(summary.assign(model=name, formula=formula))

pd.concat(summaries).to_csv(config.PROJECT_DIR / "nonspatial_model_summaries.csv")
print("\nsaved nonspatial_model_summaries.csv")

# --- pre-spatial residual diagnostics from the full model ---
full = "m6_p95_elev_heat"
pred = models[full].predict(fits[full], kind="response_params", inplace=False)
# bambi names the Gaussian mean parameter "mu" (older) or "<response>_mean" (newer)
mu_name = "mu" if "mu" in pred.posterior else [v for v in pred.posterior.data_vars if v.endswith("_mean")][0]
mu = pred.posterior[mu_name].mean(dim=("chain", "draw")).values
metrics["resid"] = metrics["rumple"].to_numpy() - mu

coords = metrics[["x", "y"]].to_numpy()
w = DistanceBand(coords, threshold=config.NEIGHBOUR_DIST, binary=True, silence_warnings=True)
w.transform = "r"
mi = Moran(metrics["resid"].to_numpy(), w, permutations=999)
print("\nNon-spatial residual diagnostics")
print("Moran's I:", mi.I, "| Expected:", mi.EI, "| Permutation p:", mi.p_sim)

# Seed the global RNG here, not via a Variogram argument -- see the module docstring.
np.random.seed(SEED)
V = skg.Variogram(coords, metrics["resid"].to_numpy(), estimator="matheron",
                  model="spherical", n_lags=20, maxlag=1500, use_nugget=True, samples=3000)
print("Effective range:", V.parameters[0], "m | Sill:", V.parameters[1], "| Nugget:", V.parameters[2])

# Save the nonspatial residual variogram (empirical + fitted) + diagnostics so the manuscript
# figure can show the before/after against the ICAR residuals (09_spatial_diagnostics.py) with no
# re-subsampling. The fitted spherical range is poorly identified here (a modest structured
# fraction over a large nugget), so we report Moran's I, sill and nugget rather than one range.
x_fit = np.linspace(0, V.bins.max(), 500)
pd.DataFrame({"distance_m": V.bins, "semivariance": V.experimental}).to_csv(
    config.PROJECT_DIR / "nonspatial_residual_variogram_empirical.csv", index=False)
pd.DataFrame({"distance_m": x_fit, "fitted_semivariance": V.fitted_model(x_fit)}).to_csv(
    config.PROJECT_DIR / "nonspatial_residual_variogram_fitted.csv", index=False)
pd.DataFrame({
    "parameter": ["Moran's I", "Moran permutation p", "Effective range (m)", "Sill", "Nugget"],
    "value": [mi.I, mi.p_sim, V.parameters[0], V.parameters[1], V.parameters[2]],
}).to_csv(config.PROJECT_DIR / "nonspatial_residual_diagnostics.csv", index=False)
print("saved nonspatial residual variogram tables (empirical + fitted + diagnostics)")
