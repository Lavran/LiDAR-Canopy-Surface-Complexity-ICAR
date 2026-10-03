"""
01_clean_pointcloud.py
Statistical outlier removal + noise-class drop, then normalize heights to height
above ground (Delaunay TIN), writing cleaned tiles to point_cloud_clean/.
"""
import os
import glob
import json
import laspy
import pdal

import config


def clean_pointcloud(in_folder, out_folder, k, multi, crs=config.CRS):
    os.makedirs(str(out_folder), exist_ok=True)
    for i in glob.glob(os.path.join(str(in_folder), "*.laz")):
        with laspy.open(i) as f:                       # header-only read of point count
            n_in = f.header.point_count
        pipeline = {"pipeline": [
            {"type": "readers.las", "filename": i},
            # statistical outlier filter: k neighbours + std-dev multiplier
            {"type": "filters.outlier", "method": "statistical",
             "mean_k": k, "multiplier": multi},
            # drop low-noise (7) and USGS high-noise (18) classes
            {"type": "filters.expression",
             "expression": "Classification != 7 && Classification != 18"},
            # height above ground from a Delaunay TIN of ground points
            {"type": "filters.hag_delaunay"},
            # copy HeightAboveGround into Z (normalized heights)
            {"type": "filters.ferry", "dimensions": "HeightAboveGround=>Z"},
            {"type": "writers.las",
             "filename": os.path.join(str(out_folder), os.path.basename(i)),
             "a_srs": crs},
        ]}
        n_out = pdal.Pipeline(json.dumps(pipeline)).execute()
        flagged = n_in - n_out
        print(f"{os.path.basename(i)}: {n_in:,} -> {n_out:,} "
              f"({flagged:,} flagged, {100*flagged/n_in:.1f}%)")


if __name__ == "__main__":
    clean_pointcloud(config.LAS_RAW, config.LAS_CLEAN,
                     config.OUTLIER_K, config.OUTLIER_MULTIPLIER)
