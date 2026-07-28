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
year = 2007

base_folder = r"F:\MYPHDRESEARCHOUTPUTS\building_height_results_2007"

qol_path = r"C:\Users\ravit\Documents\tree research work new\QOL_NPA_2020_final_projected\QOL_NPA_2020_final_projected.shp"

building_shapefile = r"F:\MYPHDRESEARCHDATA\Mecklenburg_Buildings_2007\BLDG_OTLN_2007.shp"

output_folder = r"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_2007\QOL_NPA_BuildingHeight_Min10ft_Max900ft_Outputs"
os.makedirs(output_folder, exist_ok=True)

county_output_folder = r"F:\MYPHDRESEARCHOUTPUTS\Countywide_Combined_Results_2007"
os.makedirs(county_output_folder, exist_ok=True)

minimum_building_height_ft = 10.0
maximum_building_height_ft = 900.0
bad_value_limit = 1e10
nodata_value = -9999.0

selected_npa_ids = None
skip_existing = True


summary_csv = os.path.join(
    output_folder,
    f"QOL_NPA_Building_Height_Summary_Min10ft_Max900ft_{year}.csv"
)

county_bhm_output = os.path.join(
    county_output_folder,
    f"Mecklenburg_BHM_1m_BuildingsOnly_Min10ft_Max900ft_ByNPA_{year}.tif"
)

county_buildings_output = os.path.join(
    county_output_folder,
    f"Mecklenburg_building_heights_Min10ft_Max900ft_ByNPA_{year}.shp"
)


# =========================================================
# HELPERS
# =========================================================
def clean_bhm(arr, make_nodata=True):
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(nodata_value)

    arr = np.asarray(arr, dtype="float32")

    if arr.ndim == 3:
        arr = arr[0]

    bad = (
        ~np.isfinite(arr) |
        (np.abs(arr) > bad_value_limit) |
        (arr < minimum_building_height_ft) |
        (arr > maximum_building_height_ft)
    )

    arr[bad] = nodata_value if make_nodata else 0
    return arr.astype("float32")


def remove_shapefile(shp_path):
    base = os.path.splitext(shp_path)[0]

    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx"]:
        p = base + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def keep_only_polygons(gdf):
    if gdf is None or gdf.empty:
        return gdf

    gdf = gdf[
        gdf.geometry.notnull() &
        ~gdf.geometry.is_empty
    ].copy()

    gdf = gdf[
        gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    return gdf


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


def get_height_column(gdf):
    preferred_cols = [
        "preferred_h_ft",
        "preferre_1",
        "p95_h_ft",
        "mean_h_ft",
        "med_h_ft",
        "max_h_ft",
        "height_ft"
    ]

    for col in preferred_cols:
        if col in gdf.columns:
            return col

    return None


# =========================================================
# LOAD DATA
# =========================================================
print("Loading QOL polygons...")
qol = gpd.read_file(qol_path)

if qol.crs is None:
    raise ValueError("QOL shapefile has no CRS.")

print("Loading building footprints...")
building_fp = gpd.read_file(building_shapefile)

if building_fp.crs is None:
    raise ValueError("Building footprint shapefile has no CRS.")

building_fp = keep_only_polygons(building_fp)

if selected_npa_ids is not None:
    qol = qol[qol["NPA_ID"].isin(selected_npa_ids)].copy()

qol = qol[qol.geometry.notnull() & ~qol.geometry.is_empty].copy()
qol = qol.sort_values("NPA_ID").copy()

print(f"QOL polygons to process: {len(qol)}")


# =========================================================
# FIND BHM RASTERS AND BUILDING HEIGHT SHAPEFILES
# =========================================================
print("Finding BHM rasters...")

bhm_files = sorted(
    glob.glob(
        os.path.join(base_folder, "**", f"*_BHM_1m_{year}.tif"),
        recursive=True
    )
)

print(f"Found {len(bhm_files)} BHM rasters.")

if len(bhm_files) == 0:
    raise FileNotFoundError("No BHM rasters found. Check base_folder and file naming.")

print("Finding building height shapefiles...")

building_height_files = sorted(
    glob.glob(
        os.path.join(base_folder, "**", f"*_building_heights_{year}.shp"),
        recursive=True
    )
)

print(f"Found {len(building_height_files)} building height shapefiles.")


# =========================================================
# BUILD BHM TILE INDEX
# =========================================================
print("Building BHM tile index...")

tile_records = []
raster_crs = None

for fp in bhm_files:
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
    raise RuntimeError("No valid BHM rasters could be indexed.")

tile_index = gpd.GeoDataFrame(tile_records, crs=raster_crs)

if qol.crs != raster_crs:
    qol = qol.to_crs(raster_crs)

if building_fp.crs != raster_crs:
    building_fp = building_fp.to_crs(raster_crs)

building_fp = keep_only_polygons(building_fp)
building_fp = building_fp[building_fp.geometry.is_valid].copy()
building_fp_sindex = building_fp.sindex


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

    out_bhm = os.path.join(
        npa_folder,
        f"NPA_{npa_id}_BHM_BuildingsOnly_Min10ft_Max900ft_{year}.tif"
    )

    out_buildings = os.path.join(
        npa_folder,
        f"NPA_{npa_id}_building_heights_Min10ft_Max900ft_{year}.shp"
    )

    if skip_existing and os.path.exists(out_bhm):
        print("  Already processed. Skipping this NPA.")
        continue

    tiles_hit = tile_index[tile_index.intersects(npa_geom)].copy()

    if tiles_hit.empty:
        print("  No BHM tiles intersect this NPA.")
        continue

    print(f"  BHM tiles found: {len(tiles_hit)}")

    src_files = []
    temp_npa_bhm = os.path.join(
        npa_folder,
        f"NPA_{npa_id}_temp_bhm_mosaic.tif"
    )

    try:
        for fp in tiles_hit["path"]:
            src_files.append(rasterio.open(fp))

        mosaic, out_transform = merge(
            src_files,
            bounds=npa_geom.bounds,
            method="max",
            nodata=0
        )

        mosaic = clean_bhm(mosaic, make_nodata=False)

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

        with rasterio.open(temp_npa_bhm, "w", **out_profile) as dst:
            dst.write(mosaic, 1)

    finally:
        for src in src_files:
            src.close()

        if "mosaic" in locals():
            del mosaic

        gc.collect()


    # =====================================================
    # CLIP BHM TO NPA AND BUILDING FOOTPRINTS
    # =====================================================
    with rasterio.open(temp_npa_bhm) as src:
        clipped, clipped_transform = mask(
            src,
            [npa_geom],
            crop=True,
            filled=True,
            nodata=nodata_value
        )

        data = clean_bhm(clipped, make_nodata=True)

        possible_idx = list(building_fp_sindex.intersection(npa_geom.bounds))

        if len(possible_idx) > 0:
            fp_subset = building_fp.iloc[possible_idx].copy()
            fp_subset = fp_subset[fp_subset.intersects(npa_geom)]
            fp_subset = keep_only_polygons(fp_subset)

            if not fp_subset.empty:
                building_mask = rasterize(
                    [
                        (geom, 1)
                        for geom in fp_subset.geometry
                        if geom is not None and not geom.is_empty
                    ],
                    out_shape=data.shape,
                    transform=clipped_transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=True
                )

                data[building_mask != 1] = nodata_value

        data = clean_bhm(data, make_nodata=True)

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

        with rasterio.open(out_bhm, "w", **out_meta) as dst:
            dst.write(data, 1)

    if os.path.exists(temp_npa_bhm):
        os.remove(temp_npa_bhm)


    # =====================================================
    # PROCESS BUILDING HEIGHT POLYGONS
    # =====================================================
    building_gdfs = []

    for shp in building_height_files:
        try:
            gdf = gpd.read_file(shp)

            if gdf.empty:
                continue

            if gdf.crs is None:
                gdf = gdf.set_crs(raster_crs)

            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)

            gdf = keep_only_polygons(gdf)

            if gdf.empty:
                continue

            if not gdf.intersects(npa_geom).any():
                continue

            npa_gdf = gpd.GeoDataFrame(geometry=[npa_geom], crs=raster_crs)
            gdf = gpd.clip(gdf, npa_gdf)
            gdf = keep_only_polygons(gdf)

            if gdf.empty:
                continue

            height_col = get_height_column(gdf)

            if height_col is not None:
                heights_temp = pd.to_numeric(gdf[height_col], errors="coerce")
                gdf = gdf[
                    (heights_temp >= minimum_building_height_ft) &
                    (heights_temp <= maximum_building_height_ft)
                ].copy()

            gdf = keep_only_polygons(gdf)

            if gdf.empty:
                continue

            tile_id = os.path.basename(shp).replace(
                f"_building_heights_{year}.shp",
                ""
            )

            gdf["tile_id"] = tile_id
            gdf["NPA_ID"] = npa_id

            building_gdfs.append(gdf)

        except Exception as e:
            print(f"  Skipping building height file because of error: {shp}")
            print(e)

    if len(building_gdfs) > 0:
        buildings_all = gpd.GeoDataFrame(
            pd.concat(building_gdfs, ignore_index=True),
            crs=raster_crs
        )

        buildings_all = keep_only_polygons(buildings_all)

        if not buildings_all.empty:
            remove_shapefile(out_buildings)
            buildings_all.to_file(out_buildings)
            building_count = len(buildings_all)

            height_col = get_height_column(buildings_all)

            if height_col is not None:
                heights = pd.to_numeric(buildings_all[height_col], errors="coerce")
                heights = heights[
                    (heights >= minimum_building_height_ft) &
                    (heights <= maximum_building_height_ft)
                ]

                mean_building_height_ft = float(heights.mean()) if len(heights) > 0 else 0
                median_building_height_ft = float(heights.median()) if len(heights) > 0 else 0
                p95_building_height_ft = float(heights.quantile(0.95)) if len(heights) > 0 else 0
                max_building_height_ft = float(heights.max()) if len(heights) > 0 else 0

            else:
                mean_building_height_ft = 0
                median_building_height_ft = 0
                p95_building_height_ft = 0
                max_building_height_ft = 0

        else:
            building_count = 0
            mean_building_height_ft = 0
            median_building_height_ft = 0
            p95_building_height_ft = 0
            max_building_height_ft = 0
            print("  No valid polygon building height features found after clipping.")

    else:
        building_count = 0
        mean_building_height_ft = 0
        median_building_height_ft = 0
        p95_building_height_ft = 0
        max_building_height_ft = 0
        print("  No building height polygons found for this NPA.")


    # =====================================================
    # SUMMARY METRICS FROM BHM RASTER
    # =====================================================
    valid = data[data != nodata_value]

    valid = valid[
        (valid >= minimum_building_height_ft) &
        (valid <= maximum_building_height_ft)
    ]

    pixel_area_ft2 = abs(clipped_transform.a * clipped_transform.e)
    pixel_area_m2 = pixel_area_ft2 * 0.092903

    building_area_m2 = valid.size * pixel_area_m2

    npa_area_m2 = npa_geom.area * 0.092903
    npa_area_ha = npa_area_m2 / 10000.0 if npa_area_m2 > 0 else 0

    building_cover = building_area_m2 / npa_area_m2 if npa_area_m2 > 0 else 0
    building_density_per_ha = building_count / npa_area_ha if npa_area_ha > 0 else 0

    mean_bhm_height_ft = float(np.mean(valid)) if valid.size > 0 else 0
    median_bhm_height_ft = float(np.median(valid)) if valid.size > 0 else 0
    p95_bhm_height_ft = float(np.percentile(valid, 95)) if valid.size > 0 else 0
    max_bhm_height_ft = float(np.max(valid)) if valid.size > 0 else 0

    summary_rows = [
        r for r in summary_rows
        if not (int(r["NPA_ID"]) == npa_id and int(r["year"]) == year)
    ]

    summary_rows.append({
        "NPA_ID": npa_id,
        "year": year,
        "height_threshold_min_ft": minimum_building_height_ft,
        "height_threshold_max_ft": maximum_building_height_ft,
        "bhm_tiles": len(tiles_hit),
        "building_count": building_count,
        "building_area_m2": building_area_m2,
        "building_cover": building_cover,
        "building_density_per_ha": building_density_per_ha,
        "mean_bhm_height_ft": mean_bhm_height_ft,
        "median_bhm_height_ft": median_bhm_height_ft,
        "p95_bhm_height_ft": p95_bhm_height_ft,
        "max_bhm_height_ft": max_bhm_height_ft,
        "mean_bhm_height_m": mean_bhm_height_ft * 0.3048,
        "median_bhm_height_m": median_bhm_height_ft * 0.3048,
        "p95_bhm_height_m": p95_bhm_height_ft * 0.3048,
        "max_bhm_height_m": max_bhm_height_ft * 0.3048,
        "mean_building_height_ft": mean_building_height_ft,
        "median_building_height_ft": median_building_height_ft,
        "p95_building_height_ft": p95_building_height_ft,
        "max_building_height_ft": max_building_height_ft,
        "mean_building_height_m": mean_building_height_ft * 0.3048,
        "median_building_height_m": median_building_height_ft * 0.3048,
        "p95_building_height_m": p95_building_height_ft * 0.3048,
        "max_building_height_m": max_building_height_ft * 0.3048,
        "output_bhm": out_bhm,
        "output_buildings": out_buildings if building_count > 0 else ""
    })

    safe_write_summary(summary_rows, summary_csv)

    print(f"  Saved BHM: {out_bhm}")
    print(f"  Building count: {building_count}")
    print(f"  Mean BHM height ft: {mean_bhm_height_ft:.2f}")
    print(f"  Median BHM height ft: {median_bhm_height_ft:.2f}")
    print(f"  P95 BHM height ft: {p95_bhm_height_ft:.2f}")
    print(f"  Max BHM height ft: {max_bhm_height_ft:.2f}")
    print(f"  Mean polygon building height ft: {mean_building_height_ft:.2f}")
    print(f"  Building density per ha: {building_density_per_ha:.2f}")

    del data
    gc.collect()


# =========================================================
# FINAL COMBINED COUNTY BHM
# =========================================================
print("\nCreating final combined countywide BHM from NPA outputs...")

npa_bhm_files = sorted(
    glob.glob(
        os.path.join(
            output_folder,
            "NPA_*",
            f"NPA_*_BHM_BuildingsOnly_Min10ft_Max900ft_{year}.tif"
        )
    )
)

if len(npa_bhm_files) == 0:
    print("No NPA BHM files found for countywide merge.")
else:
    src_files = []

    try:
        for fp in npa_bhm_files:
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
            (county_mosaic < minimum_building_height_ft) |
            (county_mosaic > maximum_building_height_ft)
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

        with rasterio.open(county_bhm_output, "w", **county_meta) as dst:
            dst.write(county_mosaic, 1)

        print(f"Countywide BHM written: {county_bhm_output}")

    finally:
        for src in src_files:
            src.close()

        if "county_mosaic" in locals():
            del county_mosaic

        gc.collect()


# =========================================================
# FINAL COMBINED COUNTY BUILDING HEIGHT POLYGONS
# =========================================================
print("\nCreating final combined countywide building height shapefile...")

npa_building_files = sorted(
    glob.glob(
        os.path.join(
            output_folder,
            "NPA_*",
            f"NPA_*_building_heights_Min10ft_Max900ft_{year}.shp"
        )
    )
)

if len(npa_building_files) == 0:
    print("No NPA building height shapefiles found for countywide merge.")
else:
    building_list = []

    for shp in npa_building_files:
        try:
            gdf = gpd.read_file(shp)

            if gdf.empty:
                continue

            if gdf.crs is None:
                gdf = gdf.set_crs(raster_crs)

            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)

            gdf = keep_only_polygons(gdf)

            if not gdf.empty:
                building_list.append(gdf)

        except Exception as e:
            print(f"Skipping NPA building shapefile: {shp}")
            print(e)

    if len(building_list) > 0:
        county_buildings = gpd.GeoDataFrame(
            pd.concat(building_list, ignore_index=True),
            crs=raster_crs
        )

        county_buildings = keep_only_polygons(county_buildings)

        if not county_buildings.empty:
            remove_shapefile(county_buildings_output)
            county_buildings.to_file(county_buildings_output)

            print(f"Countywide building heights written: {county_buildings_output}")
            print(f"Countywide building count: {len(county_buildings)}")
        else:
            print("No valid polygon features found for countywide building output.")

print("\nDone.")
print(f"Summary written: {summary_csv}")