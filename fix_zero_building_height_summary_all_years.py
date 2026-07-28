import os
import glob
import gc
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.geometry import box


# =========================================================
# YEARS TO FIX
# =========================================================
YEARS = {
    2017: {
        "base_folder": r"C:\Users\ravit\Documents\tree research work new\building_height_results_2017",
        "building_shapefile": r"C:\Users\ravit\Documents\tree research work new\Mecklenburg_Buildings_2017\Mecklenburg_Buildings_2017.shp",
        "county_output_folder": r"C:\Users\ravit\Documents\tree research work new\Countywide_Combined_Results_2017",
        "output_folder": r"C:\Users\ravit\Documents\tree research work new\Countywide_Combined_Results_2017\QOL_NPA_BuildingHeight_Min10ft_Max900ft_Outputs"
    },
    2012: {
        "base_folder": r"C:\Users\ravit\Documents\tree research work new\building_height_results_2012",
        "building_shapefile": r"C:\Users\ravit\Documents\tree research work new\Mecklenburg_2012_BuildingFootprints\Buildings.shp",
        "county_output_folder": r"C:\Users\ravit\Documents\tree research work new\Countywide_Combined_Results_2012",
        "output_folder": r"C:\Users\ravit\Documents\tree research work new\Countywide_Combined_Results_2012\QOL_NPA_BuildingHeight_Min10ft_Max900ft_Outputs"
    },
    2007: {
        "base_folder": r"C:\Users\ravit\Documents\tree research work new\building_height_results_2007",
        "building_shapefile": r"C:\Users\ravit\Documents\tree research work new\Mecklenburg_Buildings_2007\BLDG_OTLN_2007.shp",
        "county_output_folder": r"C:\Users\ravit\Documents\tree research work new\Countywide_Combined_Results_2007",
        "output_folder": r"C:\Users\ravit\Documents\tree research work new\Countywide_Combined_Results_2007\QOL_NPA_BuildingHeight_Min10ft_Max900ft_Outputs"
    },
    2002: {
        "base_folder": r"C:\Users\ravit\Documents\tree research work new\building_height_results_2002",
        "building_shapefile": r"C:\Users\ravit\Documents\tree research work new\Mecklenburg_Buildings_2002\Mecklenburg_Buildings_2002.shp",
        "county_output_folder": r"C:\Users\ravit\Documents\tree research work new\Countywide_Combined_Results_2002",
        "output_folder": r"C:\Users\ravit\Documents\tree research work new\Countywide_Combined_Results_2002\QOL_NPA_BuildingHeight_Min10ft_Max900ft_Outputs"
    }
}


qol_path = r"C:\Users\ravit\Documents\tree research work new\QOL_NPA_2020_final_projected\QOL_NPA_2020_final_projected.shp"

minimum_building_height_ft = 10.0
maximum_building_height_ft = 900.0
nodata_value = -9999.0
bad_value_limit = 1e10

skip_existing_bhm = True


# =========================================================
# HELPERS
# =========================================================
def clean_bhm(arr):
    arr = np.asarray(arr, dtype="float32")

    if arr.ndim == 3:
        arr = arr[0]

    bad = (
        ~np.isfinite(arr) |
        (np.abs(arr) > bad_value_limit) |
        (arr < minimum_building_height_ft) |
        (arr > maximum_building_height_ft)
    )

    arr[bad] = nodata_value
    return arr


def summarize_bhm_array(data):
    valid = data[
        np.isfinite(data) &
        (data != nodata_value) &
        (data >= minimum_building_height_ft) &
        (data <= maximum_building_height_ft)
    ]

    if valid.size == 0:
        return 0, 0, 0, 0

    return (
        float(np.mean(valid)),
        float(np.median(valid)),
        float(np.percentile(valid, 95)),
        float(np.max(valid))
    )


def safe_write_summary(summary_rows, summary_csv):
    df = pd.DataFrame(summary_rows)

    df = df.drop_duplicates(
        subset=["NPA_ID", "year"],
        keep="last"
    ).sort_values("NPA_ID")

    df.to_csv(summary_csv, index=False)


# =========================================================
# MAIN LOOP
# =========================================================
qol = gpd.read_file(qol_path)

for year, cfg in YEARS.items():

    print("\n==============================")
    print(f"Fixing year {year}")
    print("==============================")

    output_folder = cfg["output_folder"]

    summary_csv = os.path.join(
        output_folder,
        f"QOL_NPA_Building_Height_Summary_Min10ft_Max900ft_{year}.csv"
    )

    if not os.path.exists(summary_csv):
        print("Summary file not found. Skipping year.")
        continue

    df = pd.read_csv(summary_csv)
    summary_rows = df.to_dict("records")

    for i, row in enumerate(summary_rows):

        npa_id = int(row["NPA_ID"])

        bhm_path = os.path.join(
            output_folder,
            f"NPA_{npa_id}",
            f"NPA_{npa_id}_BHM_BuildingsOnly_Min10ft_Max900ft_{year}.tif"
        )

        if not os.path.exists(bhm_path):
            continue

        with rasterio.open(bhm_path) as src:
            data = src.read(1)

        data = clean_bhm(data)

        mean_ft, median_ft, p95_ft, max_ft = summarize_bhm_array(data)

        # overwrite zero values
        summary_rows[i]["mean_building_height_ft"] = mean_ft
        summary_rows[i]["median_building_height_ft"] = median_ft
        summary_rows[i]["p95_building_height_ft"] = p95_ft
        summary_rows[i]["max_building_height_ft"] = max_ft

        summary_rows[i]["mean_building_height_m"] = mean_ft * 0.3048
        summary_rows[i]["median_building_height_m"] = median_ft * 0.3048
        summary_rows[i]["p95_building_height_m"] = p95_ft * 0.3048
        summary_rows[i]["max_building_height_m"] = max_ft * 0.3048

        print(f"NPA {npa_id} fixed")

    safe_write_summary(summary_rows, summary_csv)

    print(f"Updated CSV saved: {summary_csv}")

print("\nAll years fixed.")
