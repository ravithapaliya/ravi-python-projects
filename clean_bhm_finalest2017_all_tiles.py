import os
import time
import math
import warnings
import gc

import laspy
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.geometry import box
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 1. USER SETTINGS
# =========================================================
las_folder = r"F:\MecklenburgCountyGIS\LiDAR2017\Masspoints"
base_output_folder = r"C:\Users\ravit\Documents\building_height_results_2017"
building_shapefile = r"F:\MYPHDRESEARCHDATA\Mecklenburg_Buildings_2017\Mecklenburg_Buildings_2017.shp"

target_crs = CRS.from_epsg(2264)

# 1 meter in US Survey Feet
resolution = 3.28084

ground_classes = [2]
building_classes = [6]
noise_classes = [7]
water_classes = [9]
breakline_classes = [10]
road_classes = [13]
ignored_classes = [12]

max_surface_gap_ft = 4.0
min_points_required = 100
maximum_building_height_ft = 300.0
minimum_building_height_ft = 1.0

# Fallback buffer, in feet
fallback_buffer_ft = 2.0

skip_completed_tiles = True
chunk_rows = 50

# =========================================================
# 2. LOAD DATA
# =========================================================
os.makedirs(base_output_folder, exist_ok=True)

print("Loading 2017 building footprints...")
buildings = gpd.read_file(building_shapefile)

if buildings.crs is None:
    raise ValueError("Building shapefile has no CRS defined.")

if buildings.crs != target_crs:
    buildings = buildings.to_crs(target_crs)

buildings = buildings[buildings.geometry.notnull() & ~buildings.geometry.is_empty].copy()
buildings = buildings.reset_index(drop=True)
buildings["bldg_id"] = buildings.index.astype(int)

buildings_sindex = buildings.sindex

las_files = sorted([f for f in os.listdir(las_folder) if f.lower().endswith(".las")])
print(f"Found {len(las_files)} LAS tiles.")

# TEST MODE, run only one tile
# las_files = ["LA_37_10348708_.las"]

# =========================================================
# 3. HELPERS
# =========================================================
def write_raster(path, array, transform, crs, nodata=0.0):
    meta = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "nodata": nodata
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array.astype("float32"), 1)

def safe_stats(arr, name):
    arr = np.asarray(arr)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return f"{name}: no finite values"
    return f"{name}: min={valid.min():.2f}, max={valid.max():.2f}, mean={valid.mean():.2f}"

def build_knn_surface(xs, ys, zs, x_min, y_max, nrows, ncols, res, max_dist=None, chunk_rows=50):
    if len(xs) < 10:
        if max_dist is None:
            return np.zeros((nrows, ncols), dtype=np.float32)
        return np.full((nrows, ncols), np.nan, dtype=np.float32)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    zs = np.asarray(zs, dtype=np.float32)

    tree = cKDTree(np.column_stack((xs, ys)))

    x_c = x_min + (np.arange(ncols, dtype=np.float32) + 0.5) * res
    y_c = y_max - (np.arange(nrows, dtype=np.float32) + 0.5) * res

    if max_dist is None:
        grid = np.zeros((nrows, ncols), dtype=np.float32)
    else:
        grid = np.full((nrows, ncols), np.nan, dtype=np.float32)

    for start_r in range(0, nrows, chunk_rows):
        end_r = min(start_r + chunk_rows, nrows)
        yy = y_c[start_r:end_r]
        xx_mesh, yy_mesh = np.meshgrid(x_c, yy)
        query_pts = np.column_stack((xx_mesh.ravel(), yy_mesh.ravel())).astype(np.float32)

        if max_dist is None:
            _, idx = tree.query(query_pts, k=1, workers=1)
            vals = zs[np.asarray(idx, dtype=np.int64)].astype(np.float32)
        else:
            dist, idx = tree.query(query_pts, k=1, distance_upper_bound=max_dist, workers=1)
            dist = np.asarray(dist, dtype=np.float32).reshape(-1)
            idx = np.asarray(idx, dtype=np.int64).reshape(-1)

            vals = np.full(dist.shape, np.nan, dtype=np.float32)
            valid = np.isfinite(dist) & (dist <= max_dist) & (idx < len(zs))
            if np.any(valid):
                vals[valid] = zs[idx[valid]].astype(np.float32)

        grid[start_r:end_r, :] = vals.reshape((end_r - start_r, ncols))

        del xx_mesh, yy_mesh, query_pts, vals
        gc.collect()

    return grid

def get_buildings_for_tile(buildings_gdf, sindex, tile_bbox):
    idx = list(sindex.intersection(tile_bbox.bounds))
    if len(idx) == 0:
        return buildings_gdf.iloc[[]].copy()

    subset = buildings_gdf.iloc[idx].copy()
    subset = subset[subset.intersects(tile_bbox)].copy()
    return subset

def summarize_vals(vals):
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "pix_n": 0,
            "max_ft": np.nan,
            "mean_ft": np.nan,
            "med_ft": np.nan,
            "p95_ft": np.nan,
            "max_m": np.nan,
            "mean_m": np.nan,
            "med_m": np.nan,
            "p95_m": np.nan,
            "pref_ft": np.nan,
            "pref_m": np.nan
        }

    max_ft = float(np.max(vals))
    mean_ft = float(np.mean(vals))
    med_ft = float(np.median(vals))
    p95_ft = float(np.percentile(vals, 95))

    pref_ft = p95_ft if np.isfinite(p95_ft) else med_ft
    pref_m = pref_ft * 0.3048 if np.isfinite(pref_ft) else np.nan

    return {
        "pix_n": int(vals.size),
        "max_ft": max_ft,
        "mean_ft": mean_ft,
        "med_ft": med_ft,
        "p95_ft": p95_ft,
        "max_m": max_ft * 0.3048,
        "mean_m": mean_ft * 0.3048,
        "med_m": med_ft * 0.3048,
        "p95_m": p95_ft * 0.3048,
        "pref_ft": pref_ft,
        "pref_m": pref_m
    }

# =========================================================
# 4. MAIN LOOP
# =========================================================
all_tile_records = []

for filename in las_files:
    tile_id = os.path.splitext(filename)[0]
    tile_folder = os.path.join(base_output_folder, tile_id)
    os.makedirs(tile_folder, exist_ok=True)

    out_bhm = os.path.join(tile_folder, f"{tile_id}_BHM_1m_2017.tif")
    out_buildings = os.path.join(tile_folder, f"{tile_id}_building_heights_2017.shp")

    if skip_completed_tiles and os.path.exists(out_bhm) and os.path.exists(out_buildings):
        print(f"Skipping completed tile: {tile_id}")
        continue

    print(f"\nProcessing: {tile_id}")
    start_t = time.time()

    try:
        las_path = os.path.join(las_folder, filename)
        las = laspy.read(las_path)

        x = np.asarray(las.x, dtype=np.float32)
        y = np.asarray(las.y, dtype=np.float32)
        z = np.asarray(las.z, dtype=np.float32)
        cls = np.asarray(las.classification, dtype=np.int16)

        if len(x) == 0:
            print("  Skipping. Empty LAS.")
            continue

        print(f"  Total points: {len(x)}")
        print(f"  Classes present: {np.unique(cls)}")
        print(f"  {safe_stats(z, 'Raw Z')}")

        ground_m = np.isin(cls, ground_classes)

        excluded = ground_classes + noise_classes + water_classes + breakline_classes + road_classes + ignored_classes
        class6_roof_m = np.isin(cls, building_classes)
        broad_surface_m = ~np.isin(cls, excluded)

        if np.sum(ground_m) < min_points_required:
            print(f"  Skipping. Too few ground points: {np.sum(ground_m)}")
            continue

        roof_m = class6_roof_m.copy()
        if np.sum(roof_m) < min_points_required:
            roof_m = broad_surface_m
            print("  Building class 6 is sparse. Falling back to broad non ground surface.")

        if np.sum(roof_m) < min_points_required:
            print(f"  Skipping. Too few roof candidate points: {np.sum(roof_m)}")
            continue

        x_min = math.floor(float(np.min(x)))
        x_max = math.ceil(float(np.max(x)))
        y_min = math.floor(float(np.min(y)))
        y_max = math.ceil(float(np.max(y)))

        ncols = max(1, int(math.ceil((x_max - x_min) / resolution)))
        nrows = max(1, int(math.ceil((y_max - y_min) / resolution)))
        transform = from_origin(x_min, y_max, resolution, resolution)

        print(f"  Tile size: {ncols} cols x {nrows} rows at {resolution:.5f} ft")

        dtm = build_knn_surface(
            x[ground_m], y[ground_m], z[ground_m],
            x_min, y_max, nrows, ncols, resolution,
            max_dist=None,
            chunk_rows=chunk_rows
        )

        # Primary roof DSM from class 6 if possible
        if np.sum(class6_roof_m) >= 10:
            dsm_class6 = build_knn_surface(
                x[class6_roof_m], y[class6_roof_m], z[class6_roof_m],
                x_min, y_max, nrows, ncols, resolution,
                max_dist=max_surface_gap_ft,
                chunk_rows=chunk_rows
            )
        else:
            dsm_class6 = build_knn_surface(
                x[roof_m], y[roof_m], z[roof_m],
                x_min, y_max, nrows, ncols, resolution,
                max_dist=max_surface_gap_ft,
                chunk_rows=chunk_rows
            )

        # Broader non ground DSM for fallback
        dsm_broad = build_knn_surface(
            x[roof_m], y[roof_m], z[roof_m],
            x_min, y_max, nrows, ncols, resolution,
            max_dist=max_surface_gap_ft,
            chunk_rows=chunk_rows
        )

        dsm_class6[np.isnan(dsm_class6)] = dtm[np.isnan(dsm_class6)]
        dsm_broad[np.isnan(dsm_broad)] = dtm[np.isnan(dsm_broad)]

        bhm_class6 = np.maximum(dsm_class6 - dtm, 0).astype(np.float32)
        bhm_broad = np.maximum(dsm_broad - dtm, 0).astype(np.float32)

        bhm_class6[bhm_class6 > maximum_building_height_ft] = np.nan
        bhm_class6[bhm_class6 < minimum_building_height_ft] = np.nan

        bhm_broad[bhm_broad > maximum_building_height_ft] = np.nan
        bhm_broad[bhm_broad < minimum_building_height_ft] = np.nan

        tile_bbox = box(x_min, y_min, x_max, y_max)
        buildings_tile = get_buildings_for_tile(buildings, buildings_sindex, tile_bbox)

        if buildings_tile.empty:
            print("  No building footprints intersect this tile.")
            continue

        building_mask = rasterize(
            [(geom, 1) for geom in buildings_tile.geometry if geom.is_valid],
            out_shape=bhm_class6.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8"
        )

        bhm_class6[building_mask == 0] = np.nan
        bhm_broad[building_mask == 0] = np.nan

        print(f"  {safe_stats(dtm, 'DTM')}")
        print(f"  {safe_stats(dsm_class6, 'DSM Class6')}")
        print(f"  {safe_stats(dsm_broad, 'DSM Broad')}")
        print(f"  {safe_stats(bhm_class6, 'BHM Class6')}")
        print(f"  {safe_stats(bhm_broad, 'BHM Broad')}")

        # Write one BHM raster per tile, primary first, broad fill second
        bhm_write = np.where(np.isfinite(bhm_class6), bhm_class6, bhm_broad)
        bhm_write = np.nan_to_num(bhm_write, nan=0.0).astype(np.float32)
        write_raster(out_bhm, bhm_write, transform, target_crs, nodata=0.0)

        tile_rows = []
        tile_gdf_rows = []

        for _, row in buildings_tile.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            primary_mask = rasterize(
                [(geom, 1)],
                out_shape=bhm_class6.shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype="uint8"
            )

            vals_primary = bhm_class6[primary_mask == 1]
            vals_primary = vals_primary[np.isfinite(vals_primary)]
            primary_stats = summarize_vals(vals_primary)

            used_fallback = 0
            final_stats = primary_stats.copy()
            h_status = "ok" if primary_stats["pix_n"] > 0 else "no_height"

            if primary_stats["pix_n"] == 0:
                buffered_geom = geom.buffer(fallback_buffer_ft)

                fallback_mask = rasterize(
                    [(buffered_geom, 1)],
                    out_shape=bhm_broad.shape,
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype="uint8"
                )

                vals_fallback = bhm_broad[fallback_mask == 1]
                vals_fallback = vals_fallback[np.isfinite(vals_fallback)]
                fallback_stats = summarize_vals(vals_fallback)

                if fallback_stats["pix_n"] > 0:
                    final_stats = fallback_stats.copy()
                    h_status = "fb_recov"
                    used_fallback = 1

            record = row.copy()
            record["tile_id"] = tile_id
            record["pix_n"] = final_stats["pix_n"]
            record["max_ft"] = final_stats["max_ft"]
            record["mean_ft"] = final_stats["mean_ft"]
            record["med_ft"] = final_stats["med_ft"]
            record["p95_ft"] = final_stats["p95_ft"]
            record["max_m"] = final_stats["max_m"]
            record["mean_m"] = final_stats["mean_m"]
            record["med_m"] = final_stats["med_m"]
            record["p95_m"] = final_stats["p95_m"]
            record["pref_ft"] = final_stats["pref_ft"]
            record["pref_m"] = final_stats["pref_m"]
            record["h_status"] = h_status
            record["used_fb"] = used_fallback

            tile_gdf_rows.append(record)

            tile_rows.append({
                "bldg_id": int(row["bldg_id"]),
                "tile_id": tile_id,
                "pix_n": int(final_stats["pix_n"]),
                "max_ft": final_stats["max_ft"],
                "mean_ft": final_stats["mean_ft"],
                "med_ft": final_stats["med_ft"],
                "p95_ft": final_stats["p95_ft"],
                "max_m": final_stats["max_m"],
                "mean_m": final_stats["mean_m"],
                "med_m": final_stats["med_m"],
                "p95_m": final_stats["p95_m"],
                "pref_ft": final_stats["pref_ft"],
                "pref_m": final_stats["pref_m"],
                "h_status": h_status,
                "used_fb": used_fallback
            })

        tile_buildings_out = gpd.GeoDataFrame(tile_gdf_rows, crs=target_crs)
        tile_buildings_out.to_file(out_buildings)

        all_tile_records.extend(tile_rows)

        print(f"  Success. {len(tile_gdf_rows)} building polygons written for this tile.")
        print(f"  Finished in {((time.time() - start_t) / 60.0):.2f} minutes")

        del las, x, y, z, cls, dtm, dsm_class6, dsm_broad, bhm_class6, bhm_broad
        del bhm_write, building_mask, buildings_tile, tile_rows, tile_gdf_rows, tile_buildings_out
        gc.collect()

    except MemoryError:
        print("  Error: memory error. Reduce chunk_rows.")
        gc.collect()

    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        gc.collect()

# =========================================================
# 5. COUNTYWIDE MERGE
# =========================================================
print("\nBuilding countywide outputs...")

if len(all_tile_records) == 0:
    print("No tile records were created.")
else:
    all_df = pd.DataFrame(all_tile_records)

    all_df["status_rank"] = all_df["h_status"].map({
        "ok": 2,
        "fb_recov": 1,
        "no_height": 0
    }).fillna(0)

    all_df = all_df.sort_values(
        by=["bldg_id", "status_rank", "pix_n", "pref_ft"],
        ascending=[True, False, False, False]
    )

    best_df = all_df.drop_duplicates(subset="bldg_id", keep="first").copy()
    best_df = best_df.drop(columns=["status_rank"])

    buildings_final = buildings.merge(best_df, on="bldg_id", how="left")

    missing_mask = buildings_final["h_status"].isna()
    buildings_final.loc[missing_mask, "h_status"] = "no_data"
    buildings_final.loc[missing_mask, "pix_n"] = 0
    buildings_final.loc[missing_mask, "used_fb"] = 0

    out_all = os.path.join(base_output_folder, "Meck_BldgHt_2017_All.shp")
    buildings_final.to_file(out_all)

    valid_only = buildings_final[buildings_final["h_status"].isin(["ok", "fb_recov"])].copy()
    out_valid = os.path.join(base_output_folder, "Meck_BldgHt_2017_Valid.shp")
    valid_only.to_file(out_valid)

    fallback_only = buildings_final[buildings_final["h_status"] == "fb_recov"].copy()
    out_fallback = os.path.join(base_output_folder, "Meck_BldgHt_2017_Fallback.shp")
    fallback_only.to_file(out_fallback)

    print(f"Countywide all buildings written: {out_all}")
    print(f"Valid only buildings written: {out_valid}")
    print(f"Fallback recovered buildings written: {out_fallback}")
    print(f"Total county polygons written: {len(buildings_final)}")
    print(f"Valid polygons written: {len(valid_only)}")
    print(f"Fallback recovered polygons written: {len(fallback_only)}")

print("\nAll tiles processed.")