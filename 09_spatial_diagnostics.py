"""
09_spatial_diagnostics.py
Post-spatial-model residual diagnostics for the ICAR fit. Reads the per-cell residual
field written by 08 (spatial_field_icar.csv), recomputes Moran's I and a spherical
semivariogram, and writes the diagnostic tables + the variogram figure.
No re-sampling: everything is derived from the saved residuals.

skgstat subsamples 3,000 points to build the variogram, so the reported nugget and sill
move on every run unless the draw is fixed. Its Variogram class swallows unknown
keywords, so passing `random_state=` does nothing -- the subsampling reads numpy's
GLOBAL random state (see skgstat.MetricSpace). Seeding numpy immediately before
constructing the Variogram is what actually fixes the draw; this was verified by
fitting the same data twice each way.

07_nonspatial_models.py seeds the same way with the same SEED, so the before/after
variograms are built on the same subsample of cells.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libpysal.weights import DistanceBand
from esda.moran import Moran
import skgstat as skg

import config

MODEL = "icar"
SEED = 1738

field = pd.read_csv(config.PROJECT_DIR / f"spatial_field_{MODEL}.csv")
coords = field[["x", "y"]].to_numpy()
resid = field[f"resid_{MODEL}"].to_numpy()
print(f"{len(field):,} cells read from spatial_field_{MODEL}.csv")

# --- Moran's I (same neighbour band as the non-spatial diagnostics) ---
w = DistanceBand(coords, threshold=config.NEIGHBOUR_DIST, binary=True, silence_warnings=True)
w.transform = "r"
mi = Moran(resid, w, permutations=999)
print("Spatial-model residual diagnostics")
print("Moran's I:", mi.I, "| Expected:", mi.EI, "| Permutation p:", mi.p_sim)

# --- semivariogram ---
# Seed the global RNG here, not via a Variogram argument -- see the module docstring.
np.random.seed(SEED)
V = skg.Variogram(coords, resid, estimator="matheron", model="spherical",
                  n_lags=20, maxlag=1500, use_nugget=True, samples=3000)
print("Effective range:", V.parameters[0], "m | Sill:", V.parameters[1],
      "| Nugget:", V.parameters[2])

# --- export tables ---
pd.DataFrame({"distance_m": V.bins, "semivariance": V.experimental}).to_csv(
    config.PROJECT_DIR / f"spatial_residual_variogram_empirical_{MODEL}.csv", index=False)
x_fit = np.linspace(0, V.bins.max(), 500)
pd.DataFrame({"distance_m": x_fit, "fitted_semivariance": V.fitted_model(x_fit)}).to_csv(
    config.PROJECT_DIR / f"spatial_residual_variogram_fitted_{MODEL}.csv", index=False)
pd.DataFrame({
    "parameter": ["Moran's I", "Moran permutation p", "Effective range (m)", "Sill", "Nugget"],
    "value": [mi.I, mi.p_sim, V.parameters[0], V.parameters[1], V.parameters[2]],
}).to_csv(config.PROJECT_DIR / f"spatial_residual_diagnostics_{MODEL}.csv", index=False)

# --- figure ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(V.bins, V.experimental, label="Empirical")
ax.plot(x_fit, V.fitted_model(x_fit), linewidth=2, label="Fitted spherical model")
ax.set_xlabel("Distance (m)"); ax.set_ylabel("Semivariance")
ax.set_title("Semivariogram of ICAR spatial-model residuals")
ax.legend(); fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(config.PROJECT_DIR / f"spatial_residual_variogram_{MODEL}.{ext}",
                dpi=300, bbox_inches="tight")
print("saved variogram tables + figure")
