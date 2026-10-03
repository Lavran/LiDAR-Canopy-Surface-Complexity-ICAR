"""
05_rumple.py
Canopy surface complexity (rumple) from the 1 m pit-free CHM at 30 m resolution.
Mask-aware 3x3 smoothing (NoData contributes nothing, never 0 m), local surface-area
factor A_f = sqrt(1 + dz/dx^2 + dz/dy^2), aggregated to a 30 m mean with a >=90%
valid-support requirement per cell.
"""
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage as ndi

import config


def rumple_from_chm(chm_path, out_tif, cell=config.RES_30, smooth=3, min_valid_fraction=0.90):
    with rasterio.open(str(chm_path)) as src:
        chm = src.read(1).astype("float32")
        fine = abs(src.transform.a)
        left, top = src.transform.c, src.transform.f
        crs, nod = src.crs, src.nodata

        valid = np.isfinite(chm)
        if nod is not None:
            valid &= chm != nod
        values = np.where(valid, chm, 0.0).astype("float32")

        # mask-aware smoothing: numerator = smoothed valid heights, denominator = valid fraction
        numerator = ndi.uniform_filter(values, size=smooth, mode="constant", cval=0.0)
        denominator = ndi.uniform_filter(valid.astype("float32"), size=smooth, mode="constant", cval=0.0)
        smoothed = np.divide(numerator, denominator,
                             out=np.full_like(chm, np.nan), where=denominator > 0)

        gy, gx = np.gradient(smoothed, fine)
        factor = np.sqrt(1.0 + gx**2 + gy**2)
        factor[~valid] = np.nan                       # report surface area only where CHM had support

        # aggregate 1 m surface factors to 30 m rumple
        b = int(round(cell / fine))
        H = (factor.shape[0] // b) * b
        W = (factor.shape[1] // b) * b
        blocks = factor[:H, :W].reshape(H // b, b, W // b, b)
        supported = np.isfinite(blocks)
        valid_fraction = supported.mean(axis=(1, 3))
        count = supported.sum(axis=(1, 3))
        total = np.nansum(blocks, axis=(1, 3))
        rumple = np.divide(total, count,
                           out=np.full((H // b, W // b), np.nan, dtype="float32"),
                           where=count > 0)
        rumple[valid_fraction < min_valid_fraction] = np.nan
        rumple = rumple.astype("float32")
        transform = from_origin(left, top, cell, cell)

    os.makedirs(os.path.dirname(str(out_tif)) or ".", exist_ok=True)
    with rasterio.open(str(out_tif), "w", driver="GTiff", height=rumple.shape[0],
                       width=rumple.shape[1], count=1, dtype="float32", crs=crs,
                       transform=transform, nodata=-9999) as dst:
        dst.write(np.where(np.isfinite(rumple), rumple, -9999), 1)
        dst.set_band_description(1, "rumple")
    print(f"wrote {os.path.basename(str(out_tif))}")


if __name__ == "__main__":
    rumple_from_chm(config.CHM_PATH, config.RUMPLE_PATH, cell=config.RES_30, smooth=3)
