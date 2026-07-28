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
# USER SETTINGS
# =========================================================
year = 2012

base_folder = r"F:\MYPHDRESEARCHOUTPUTS\Batch_Results2012_1m_Final"

qol_path = r"C:\Users\ravit\Documents\tree research work new\QOL_NPA_2020_final_projected\QOL_NPA_2020_final_projected.shp"

building_shapefile = r"F:\MYPHDRESEARCHDATA\Mecklenburg_2012_BuildingFootprints\Buildings.shp"

output_folder = r"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_2012\QOL_NPA_Min2m_Outputs"
os.makedirs(output_folder, exist_ok=True)

county_output_folder = r"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_2012"
os.makedirs(county_output_folder, exist_ok=True)

minimum_canopy_height_ft = 6.56
maximum_canopy_height_ft = 150.0
bad_value_limit = 1e10
nodata_value = -9999.0

selected_npa_ids = None
skip_existing = True

summary_csv = os.path.join(output_folder, f"QOL_NPA_Canopy_Summary_Min2m_{year}.csv")

county_chm_output = os.path.join(
    county_output_folder,
    f"Mecklenburg_CHM_CanopyOnly_NoBuildings_Min2m_ByNPA_{year}.tif"
)

county_crowns_output = os.path.join(
    county_output_folder,
    f"Mecklenburg_tree_crowns_NoBuildings_Min2m_ByNPA_{year}.shp"
)

# =========================================================
# HELPERS
# =========================================================
def clean_chm(arr, make_nodata=True):
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(nodata_value)

    arr = np.asarray(arr, dtype="float32")

    if arr.ndim == 3:
        arr = arr[0]

    bad = (
        ~np.isfinite(arr) |
        (np.abs(arr) > bad_value_limit) |
        (arr < minimum_canopy_height_ft) |
        (arr > maximum_canopy_height_ft)
    )

    arr[bad] = nodata_value if make_nodata else 0
    return arr.astype("float32")


def shapefile_exists(shp_path):
    base = os.path.splitext(shp_path)[0]
    return all(os.path.exists(base + ext) for ext in [".shp", ".shx", ".dbf", ".prj"])


def remove_shapefile(shp_path):
    base = os.path.splitext(shp_path)[0]
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx"]:
        p = base + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def safe_write_summary(summary_rows, summary_csv):
    df = pd.DataFrame(summary_rows)

    if df.empty:
        return

    df = df.drop_duplicates(
        subset=["NPA_ID", "year"],
        keep="last"
    ).sort_values("NPA_ID")

    temp_csv = summary_csv.replace(".csv", "_temp.csv")
    df.to_csv(temp_csv, index=False)

    for attempt in range(5):
        try:
            if os.path.exists(summary_csv):
                os.remove(summary_csv)
            os.rename(temp_csv, summary_csv)
            return
        except PermissionError:
            print("  Summary CSV is locked. Close it in Excel or ArcGIS. Retrying...")
            time.sleep(2)

    backup_csv = summary_csv.replace(".csv", f"_backup_{int(time.time())}.csv")
    df.to_csv(backup_csv, index=False)
    print(f"  Could not overwrite locked summary CSV. Backup written: {backup_csv}")


# =========================================================
# LOAD DATA
# =========================================================
print("Loading QOL polygons...")
qol = gpd.read_file(qol_path)

if qol.crs is None:
    raise ValueError("QOL shapefile has no CRS.")

print("Loading building footprints...")
buildings = gpd.read_file(building_shapefile)

if buildings.crs is None:
    raise ValueError("Building shapefile has no CRS.")

if selected_npa_ids is not None:
    qol = qol[qol["NPA_ID"].isin(selected_npa_ids)].copy()

qol = qol[qol.geometry.notnull() & ~qol.geometry.is_empty].copy()
qol = qol.sort_values("NPA_ID").copy()

print(f"QOL polygons to process: {len(qol)}")

# =========================================================
# FIND INPUT FILES
# =========================================================
print("Finding CHM rasters...")
chm_files = sorted(
    glob.glob(
        os.path.join(base_folder, "**", f"*_CHM_1m_{year}.tif"),
        recursive=True
    )
)

print(f"Found {len(chm_files)} CHM rasters.")

if len(chm_files) == 0:
    raise FileNotFoundError("No CHM rasters found. Check base_folder and file naming.")

print("Finding crown shapefiles...")
crown_files = sorted(
    glob.glob(
        os.path.join(base_folder, "**", f"*_tree_crowns_1m_{year}.shp"),
        recursive=True
    )
)

print(f"Found {len(crown_files)} crown shapefiles.")

# =========================================================
# BUILD CHM TILE INDEX
# =========================================================
print("Building CHM tile index...")

tile_records = []
raster_crs = None

for fp in chm_files:
    try:
        with rasterio.open(fp) as src:
            raster_crs = src.crs
            tile_records.append({
                "path": fp,
                "geometry": box(*src.bounds)
            })
    except Exception as e:
        print(f"Skipping raster: {fp}")
        print(e)

if len(tile_records) == 0:
    raise RuntimeError("No valid CHM rasters could be indexed.")

tile_index = gpd.GeoDataFrame(tile_records, crs=raster_crs)

if qol.crs != raster_crs:
    qol = qol.to_crs(raster_crs)

if buildings.crs != raster_crs:
    buildings = buildings.to_crs(raster_crs)

buildings = buildings[buildings.geometry.notnull() & ~buildings.geometry.is_empty].copy()
buildings = buildings[buildings.geometry.is_valid].copy()
buildings_sindex = buildings.sindex

# =========================================================
# LOAD EXISTING SUMMARY
# =========================================================
summary_rows = []

if os.path.exists(summary_csv):
    try:
        existing_summary = pd.read_csv(summary_csv)
        summary_rows = existing_summary.to_dict("records")
        print(f"Loaded existing summary rows: {len(summary_rows)}")
    except Exception:
        print("Existing summary CSV could not be read. A new summary will be created.")

# =========================================================
# PROCESS EACH NPA
# =========================================================
for idx, npa in qol.iterrows():
    npa_id = int(npa["NPA_ID"])
    npa_geom = npa.geometry

    print(f"\nProcessing NPA_ID: {npa_id}")

    npa_folder = os.path.join(output_folder, f"NPA_{npa_id}")
    os.makedirs(npa_folder, exist_ok=True)

    out_chm = os.path.join(
        npa_folder,
        f"NPA_{npa_id}_CHM_CanopyOnly_NoBuildings_Min2m_{year}.tif"
    )

    out_crowns = os.path.join(
        npa_folder,
        f"NPA_{npa_id}_tree_crowns_NoBuildings_Min2m_{year}.shp"
    )

    if skip_existing and os.path.exists(out_chm):
        print("  Already processed. Skipping this NPA.")
        continue

    tiles_hit = tile_index[tile_index.intersects(npa_geom)].copy()

    if tiles_hit.empty:
        print("  No CHM tiles intersect this NPA.")
        continue

    print(f"  CHM tiles found: {len(tiles_hit)}")

    src_files = []
    temp_npa_chm = os.path.join(npa_folder, f"NPA_{npa_id}_temp_mosaic.tif")

    try:
        for fp in tiles_hit["path"]:
            src_files.append(rasterio.open(fp))

        mosaic, out_transform = merge(
            src_files,
            bounds=npa_geom.bounds,
            method="max",
            nodata=0
        )

        mosaic = clean_chm(mosaic, make_nodata=False)

        out_profile = {
            "driver": "GTiff",
            "height": mosaic.shape[0],
            "width": mosaic.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": raster_crs,
            "transform": out_transform,
            "compress": "lzw",
            "BIGTIFF": "YES",
            "nodata": nodata_value
        }

        with rasterio.open(temp_npa_chm, "w", **out_profile) as dst:
            dst.write(mosaic, 1)

    finally:
        for src in src_files:
            src.close()

        if "mosaic" in locals():
            del mosaic

        gc.collect()

    with rasterio.open(temp_npa_chm) as src:
        clipped, clipped_transform = mask(
            src,
            [npa_geom],
            crop=True,
            filled=True,
            nodata=nodata_value
        )

        data = clean_chm(clipped, make_nodata=True)

        possible_idx = list(buildings_sindex.intersection(npa_geom.bounds))

        if len(possible_idx) > 0:
            bldg_subset = buildings.iloc[possible_idx].copy()
            bldg_subset = bldg_subset[bldg_subset.intersects(npa_geom)]

            if not bldg_subset.empty:
                building_mask = rasterize(
                    [
                        (geom, 1)
                        for geom in bldg_subset.geometry
                        if geom is not None and not geom.is_empty
                    ],
                    out_shape=data.shape,
                    transform=clipped_transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=True
                )

                data[building_mask == 1] = nodata_value

        data = clean_chm(data, make_nodata=True)

        out_meta = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": raster_crs,
            "transform": clipped_transform,
            "compress": "lzw",
            "BIGTIFF": "YES",
            "nodata": nodata_value
        }

        with rasterio.open(out_chm, "w", **out_meta) as dst:
            dst.write(data, 1)

    if os.path.exists(temp_npa_chm):
        os.remove(temp_npa_chm)

    crown_gdfs = []

    for shp in crown_files:
        try:
            gdf = gpd.read_file(shp)

            if gdf.empty:
                continue

            if gdf.crs is None:
                gdf = gdf.set_crs(raster_crs)

            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)

            if not gdf.intersects(npa_geom).any():
                continue

            npa_gdf = gpd.GeoDataFrame(geometry=[npa_geom], crs=raster_crs)
            gdf = gpd.clip(gdf, npa_gdf)

            if "height_ft" in gdf.columns:
                gdf = gdf[gdf["height_ft"] >= minimum_canopy_height_ft].copy()

            if gdf.empty:
                continue

            tile_id = os.path.basename(shp).replace(f"_tree_crowns_1m_{year}.shp", "")
            gdf["tile_id"] = tile_id
            gdf["NPA_ID"] = npa_id

            crown_gdfs.append(gdf)

        except Exception as e:
            print(f"  Skipping crown file because of error: {shp}")
            print(e)

    if len(crown_gdfs) > 0:
        crowns_all = gpd.GeoDataFrame(
            pd.concat(crown_gdfs, ignore_index=True),
            crs=raster_crs
        )

        possible_idx = list(buildings_sindex.intersection(npa_geom.bounds))

        if len(possible_idx) > 0:
            bldg_subset = buildings.iloc[possible_idx].copy()
            bldg_subset = bldg_subset[bldg_subset.intersects(npa_geom)]

            if not bldg_subset.empty:
                bldg_subset = bldg_subset[["geometry"]].copy()
                bldg_subset["building_hit"] = 1

                joined = gpd.sjoin(
                    crowns_all,
                    bldg_subset,
                    how="left",
                    predicate="intersects"
                )

                crowns_all = joined[joined["building_hit"].isna()].copy()

                drop_cols = [c for c in ["index_right", "building_hit"] if c in crowns_all.columns]
                crowns_all = crowns_all.drop(columns=drop_cols)

        remove_shapefile(out_crowns)
        crowns_all.to_file(out_crowns)
        crown_count = len(crowns_all)

    else:
        crown_count = 0
        print("  No crowns found for this NPA.")

    valid = data[data != nodata_value]

    pixel_area_ft2 = abs(clipped_transform.a * clipped_transform.e)
    pixel_area_m2 = pixel_area_ft2 * 0.092903

    canopy_area_m2 = valid.size * pixel_area_m2
    npa_area_m2 = npa_geom.area * 0.092903

    canopy_cover = canopy_area_m2 / npa_area_m2 if npa_area_m2 > 0 else 0

    mean_height_ft = float(np.mean(valid)) if valid.size > 0 else 0
    max_height_ft = float(np.max(valid)) if valid.size > 0 else 0

    summary_rows = [
        r for r in summary_rows
        if not (int(r["NPA_ID"]) == npa_id and int(r["year"]) == year)
    ]

    summary_rows.append({
        "NPA_ID": npa_id,
        "year": year,
        "chm_tiles": len(tiles_hit),
        "crown_count": crown_count,
        "canopy_area_m2": canopy_area_m2,
        "canopy_cover": canopy_cover,
        "mean_height_ft": mean_height_ft,
        "max_height_ft": max_height_ft,
        "mean_height_m": mean_height_ft * 0.3048,
        "max_height_m": max_height_ft * 0.3048,
        "output_chm": out_chm,
        "output_crowns": out_crowns if crown_count > 0 else ""
    })

    safe_write_summary(summary_rows, summary_csv)

    print(f"  Saved CHM: {out_chm}")
    print(f"  Crown count: {crown_count}")
    print(f"  Canopy cover: {canopy_cover:.4f}")

    del data
    gc.collect()

# =========================================================
# FINAL COMBINED COUNTY CHM
# =========================================================
print("\nCreating final combined countywide CHM from NPA outputs...")

npa_chm_files = sorted(
    glob.glob(
        os.path.join(output_folder, "NPA_*", f"NPA_*_CHM_CanopyOnly_NoBuildings_Min2m_{year}.tif")
    )
)

if len(npa_chm_files) == 0:
    print("No NPA CHM files found for countywide merge.")
else:
    src_files = []

    try:
        for fp in npa_chm_files:
            src_files.append(rasterio.open(fp))

        county_mosaic, county_transform = merge(
            src_files,
            method="max",
            nodata=nodata_value
        )

        county_mosaic = np.asarray(county_mosaic, dtype="float32")

        if county_mosaic.ndim == 3:
            county_mosaic = county_mosaic[0]

        county_mosaic[
            (~np.isfinite(county_mosaic)) |
            (np.abs(county_mosaic) > bad_value_limit) |
            (county_mosaic < minimum_canopy_height_ft) |
            (county_mosaic > maximum_canopy_height_ft)
        ] = nodata_value

        county_meta = {
            "driver": "GTiff",
            "height": county_mosaic.shape[0],
            "width": county_mosaic.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": raster_crs,
            "transform": county_transform,
            "compress": "lzw",
            "BIGTIFF": "YES",
            "nodata": nodata_value
        }

        with rasterio.open(county_chm_output, "w", **county_meta) as dst:
            dst.write(county_mosaic, 1)

        print(f"Countywide CHM written: {county_chm_output}")

    finally:
        for src in src_files:
            src.close()

        if "county_mosaic" in locals():
            del county_mosaic

        gc.collect()

# =========================================================
# FINAL COMBINED COUNTY TREE CROWNS
# =========================================================
print("\nCreating final combined countywide tree crown shapefile...")

npa_crown_files = sorted(
    glob.glob(
        os.path.join(output_folder, "NPA_*", f"NPA_*_tree_crowns_NoBuildings_Min2m_{year}.shp")
    )
)

if len(npa_crown_files) == 0:
    print("No NPA crown shapefiles found for countywide merge.")
else:
    crown_list = []

    for shp in npa_crown_files:
        try:
            gdf = gpd.read_file(shp)

            if gdf.empty:
                continue

            if gdf.crs is None:
                gdf = gdf.set_crs(raster_crs)

            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)

            crown_list.append(gdf)

        except Exception as e:
            print(f"Skipping NPA crown shapefile: {shp}")
            print(e)

    if len(crown_list) > 0:
        county_crowns = gpd.GeoDataFrame(
            pd.concat(crown_list, ignore_index=True),
            crs=raster_crs
        )

        county_crowns = county_crowns[
            county_crowns.geometry.notnull() & ~county_crowns.geometry.is_empty
        ].copy()

        remove_shapefile(county_crowns_output)
        county_crowns.to_file(county_crowns_output)

        print(f"Countywide crowns written: {county_crowns_output}")
        print(f"Countywide crown count: {len(county_crowns)}")

print("\nDone.")
print(f"Summary written: {summary_csv}")