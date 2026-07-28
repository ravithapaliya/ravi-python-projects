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
from shapely.geometry import box, Point
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 1. USER SETTINGS
# =========================================================
bare_earth_folder = r"F:\MecklenburgCountyGIS\LiDAR2002\Bare Earth\LAS"
reflective_folder = r"F:\MecklenburgCountyGIS\LiDAR2002\Reflective Surface\New folder"
base_output_folder = r"C:\Users\ravit\Documents\building_height_results_2002"
building_shapefile = r"F:\MYPHDRESEARCHDATA\Mecklenburg_Buildings_2002\Mecklenburg_Buildings_2002.shp"

target_crs = CRS.from_epsg(2264)

# 1 meter in U.S. Survey Feet
resolution = 3.28084

# Reflective LAS classes seen in your 2002 data
ground_classes = [2]
primary_roof_classes = [6]  # building class, use first if present
noise_classes = [7]
water_classes = [9]
ignored_ground_like_classes = [20, 21]  # ignored ground, snow

max_surface_gap_ft = 6.0
max_ground_query_distance_ft = 30.0
min_points_required = 50
maximum_building_height_ft = 300.0
minimum_building_height_ft = 1.0

fallback_buffer_ft = 3.0

skip_completed_tiles = True
chunk_rows = 50

# =========================================================
# 2. LOAD DATA
# =========================================================
os.makedirs(base_output_folder, exist_ok=True)

print("Loading 2002 building footprints...")
buildings = gpd.read_file(building_shapefile)

if buildings.crs is None:
    raise ValueError("Building shapefile has no CRS defined.")

if buildings.crs != target_crs:
    buildings = buildings.to_crs(target_crs)

buildings = buildings[buildings.geometry.notnull() & ~buildings.geometry.is_empty].copy()
buildings = buildings.reset_index(drop=True)
buildings["bldg_id"] = buildings.index.astype(int)

buildings_sindex = buildings.sindex
building_bounds = buildings.total_bounds
building_bounds_poly = box(*building_bounds)

print(f"Building footprint CRS: {buildings.crs}")
print(f"Building footprint bounds: {building_bounds}")

reflective_files = sorted(
    [f for f in os.listdir(reflective_folder) if f.lower().endswith(".las")]
)
print(f"Found {len(reflective_files)} reflective LAS tiles.")

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

def find_matching_bare_earth_file(filename, bare_folder):
    candidates = [
        os.path.join(bare_folder, filename),
        os.path.join(bare_folder, filename.replace(".las", ".LAS")),
        os.path.join(bare_folder, filename.lower()),
        os.path.join(bare_folder, filename.upper()),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    base = os.path.splitext(filename)[0].lower()
    all_bare = [f for f in os.listdir(bare_folder) if f.lower().endswith(".las")]
    for f in all_bare:
        if os.path.splitext(f)[0].lower() == base:
            return os.path.join(bare_folder, f)

    return None

def normalize_point_heights(x_all, y_all, z_all, gx, gy, gz, max_dist):
    if len(gx) < 10 or len(x_all) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    gtree = cKDTree(np.column_stack([gx, gy]))
    dist, idx = gtree.query(np.column_stack([x_all, y_all]), k=1)

    z_ground = gz[idx]
    h = z_all - z_ground

    valid = np.isfinite(h) & np.isfinite(dist) & (dist <= max_dist)
    return x_all[valid], y_all[valid], h[valid]

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

def point_stats_in_polygon(poly, xs, ys, hs):
    if len(xs) == 0:
        return summarize_vals(np.array([], dtype=np.float32))

    minx, miny, maxx, maxy = poly.bounds
    bbox_mask = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)

    if np.count_nonzero(bbox_mask) == 0:
        return summarize_vals(np.array([], dtype=np.float32))

    xs_sub = xs[bbox_mask]
    ys_sub = ys[bbox_mask]
    hs_sub = hs[bbox_mask]

    inside_vals = []
    for xx, yy, hh in zip(xs_sub, ys_sub, hs_sub):
        pt = Point(float(xx), float(yy))
        if poly.contains(pt) or poly.touches(pt):
            inside_vals.append(hh)

    if len(inside_vals) == 0:
        return summarize_vals(np.array([], dtype=np.float32))

    return summarize_vals(np.asarray(inside_vals, dtype=np.float32))

# =========================================================
# 4. MAIN LOOP
# =========================================================
all_tile_records = []

for filename in reflective_files:
    tile_id = os.path.splitext(filename)[0]
    tile_folder = os.path.join(base_output_folder, tile_id)
    os.makedirs(tile_folder, exist_ok=True)

    out_bhm = os.path.join(tile_folder, f"{tile_id}_BHM_1m_2002.tif")
    out_buildings = os.path.join(tile_folder, f"{tile_id}_building_heights_2002.shp")

    if skip_completed_tiles and os.path.exists(out_bhm) and os.path.exists(out_buildings):
        print(f"Skipping completed tile: {tile_id}")
        continue

    print(f"\nProcessing: {tile_id}")
    start_t = time.time()

    try:
        reflective_path = os.path.join(reflective_folder, filename)
        bare_path = find_matching_bare_earth_file(filename, bare_earth_folder)

        if bare_path is None:
            print(f"  Skipping. No matching bare earth LAS found for {filename}")
            continue

        las_ref = laspy.read(reflective_path)
        las_be = laspy.read(bare_path)

        rx = np.asarray(las_ref.x, dtype=np.float32)
        ry = np.asarray(las_ref.y, dtype=np.float32)
        rz = np.asarray(las_ref.z, dtype=np.float32)
        rcls = np.asarray(las_ref.classification, dtype=np.int16)

        gx = np.asarray(las_be.x, dtype=np.float32)
        gy = np.asarray(las_be.y, dtype=np.float32)
        gz = np.asarray(las_be.z, dtype=np.float32)

        if len(rx) < min_points_required or len(gx) < min_points_required:
            print("  Skipping. Too few points in reflective or bare earth LAS.")
            continue

        print(f"  Reflective points: {len(rx)}")
        print(f"  Bare earth points: {len(gx)}")
        print(f"  Reflective classes: {np.unique(rcls)}")
        print(f"  {safe_stats(rz, 'Reflective Raw Z')}")
        print(f"  {safe_stats(gz, 'Bare Earth Raw Z')}")

        x_min = math.floor(float(np.min(rx)))
        x_max = math.ceil(float(np.max(rx)))
        y_min = math.floor(float(np.min(ry)))
        y_max = math.ceil(float(np.max(ry)))

        tile_bbox = box(x_min, y_min, x_max, y_max)

        if not tile_bbox.intersects(building_bounds_poly):
            print("  Tile is outside county building footprint extent.")
            continue

        buildings_tile = get_buildings_for_tile(buildings, buildings_sindex, tile_bbox)

        if buildings_tile.empty:
            print("  No building footprints intersect this tile. This is okay for non building tiles.")
            continue

        # Use true bare earth as DTM source
        ncols = max(1, int(math.ceil((x_max - x_min) / resolution)))
        nrows = max(1, int(math.ceil((y_max - y_min) / resolution)))
        transform = from_origin(x_min, y_max, resolution, resolution)

        print(f"  Tile size: {ncols} cols x {nrows} rows at {resolution:.5f} ft")

        dtm = build_knn_surface(
            gx, gy, gz,
            x_min, y_max, nrows, ncols, resolution,
            max_dist=None,
            chunk_rows=chunk_rows
        )

        # Primary roof candidates are class 6 if available in reflective LAS
        primary_roof_m = np.isin(rcls, primary_roof_classes)

        # Broad fallback uses all reflective points except ground like and obvious noise
        broad_excluded = noise_classes + water_classes + ignored_ground_like_classes
        broad_surface_m = ~np.isin(rcls, broad_excluded)

        if np.sum(primary_roof_m) < 10:
            print("  Too few class 6 roof points. Primary raster will use broad reflective surface.")

        if np.sum(broad_surface_m) < min_points_required:
            print(f"  Skipping. Too few broad reflective candidate points: {np.sum(broad_surface_m)}")
            continue

        if np.sum(primary_roof_m) >= 10:
            dsm_primary = build_knn_surface(
                rx[primary_roof_m], ry[primary_roof_m], rz[primary_roof_m],
                x_min, y_max, nrows, ncols, resolution,
                max_dist=max_surface_gap_ft,
                chunk_rows=chunk_rows
            )
        else:
            dsm_primary = build_knn_surface(
                rx[broad_surface_m], ry[broad_surface_m], rz[broad_surface_m],
                x_min, y_max, nrows, ncols, resolution,
                max_dist=max_surface_gap_ft,
                chunk_rows=chunk_rows
            )

        dsm_broad = build_knn_surface(
            rx[broad_surface_m], ry[broad_surface_m], rz[broad_surface_m],
            x_min, y_max, nrows, ncols, resolution,
            max_dist=max_surface_gap_ft,
            chunk_rows=chunk_rows
        )

        dsm_primary[np.isnan(dsm_primary)] = dtm[np.isnan(dsm_primary)]
        dsm_broad[np.isnan(dsm_broad)] = dtm[np.isnan(dsm_broad)]

        bhm_primary = np.maximum(dsm_primary - dtm, 0).astype(np.float32)
        bhm_broad = np.maximum(dsm_broad - dtm, 0).astype(np.float32)

        bhm_primary[bhm_primary > maximum_building_height_ft] = np.nan
        bhm_primary[bhm_primary < minimum_building_height_ft] = np.nan

        bhm_broad[bhm_broad > maximum_building_height_ft] = np.nan
        bhm_broad[bhm_broad < minimum_building_height_ft] = np.nan

        building_mask = rasterize(
            [(geom, 1) for geom in buildings_tile.geometry if geom.is_valid],
            out_shape=bhm_primary.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8"
        )

        bhm_primary[building_mask == 0] = np.nan
        bhm_broad[building_mask == 0] = np.nan

        print(f"  {safe_stats(dtm, 'DTM')}")
        print(f"  {safe_stats(dsm_primary, 'DSM Primary')}")
        print(f"  {safe_stats(dsm_broad, 'DSM Broad')}")
        print(f"  {safe_stats(bhm_primary, 'BHM Primary')}")
        print(f"  {safe_stats(bhm_broad, 'BHM Broad')}")

        bhm_write = np.where(np.isfinite(bhm_primary), bhm_primary, bhm_broad)
        bhm_write = np.nan_to_num(bhm_write, nan=0.0).astype(np.float32)
        write_raster(out_bhm, bhm_write, transform, target_crs, nodata=0.0)

        # Point based normalized heights for fallback recovery
        pnx, pny, pnh = normalize_point_heights(
            rx[primary_roof_m], ry[primary_roof_m], rz[primary_roof_m],
            gx, gy, gz, max_ground_query_distance_ft
        )

        bnx, bny, bnh = normalize_point_heights(
            rx[broad_surface_m], ry[broad_surface_m], rz[broad_surface_m],
            gx, gy, gz, max_ground_query_distance_ft
        )

        p_valid = (pnh >= minimum_building_height_ft) & (pnh <= maximum_building_height_ft)
        pnx, pny, pnh = pnx[p_valid], pny[p_valid], pnh[p_valid]

        b_valid = (bnh >= minimum_building_height_ft) & (bnh <= maximum_building_height_ft)
        bnx, bny, bnh = bnx[b_valid], bny[b_valid], bnh[b_valid]

        tile_rows = []
        tile_gdf_rows = []

        for _, row in buildings_tile.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            primary_mask = rasterize(
                [(geom, 1)],
                out_shape=bhm_primary.shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype="uint8"
            )

            vals_primary = bhm_primary[primary_mask == 1]
            vals_primary = vals_primary[np.isfinite(vals_primary)]
            primary_stats = summarize_vals(vals_primary)

            used_fb = 0
            final_stats = primary_stats.copy()
            h_status = "ok" if primary_stats["pix_n"] > 0 else "no_hgt"

            if primary_stats["pix_n"] == 0:
                buffered_geom = geom.buffer(fallback_buffer_ft)

                fb_primary_stats = point_stats_in_polygon(buffered_geom, pnx, pny, pnh)

                if fb_primary_stats["pix_n"] > 0:
                    final_stats = fb_primary_stats.copy()
                    h_status = "fb_recov"
                    used_fb = 1
                else:
                    fb_broad_stats = point_stats_in_polygon(buffered_geom, bnx, bny, bnh)

                    if fb_broad_stats["pix_n"] > 0:
                        final_stats = fb_broad_stats.copy()
                        h_status = "fb_recov"
                        used_fb = 1

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
            record["used_fb"] = used_fb

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
                "used_fb": used_fb
            })

        tile_buildings_out = gpd.GeoDataFrame(tile_gdf_rows, crs=target_crs)
        tile_buildings_out.to_file(out_buildings)

        all_tile_records.extend(tile_rows)

        print(f"  Success. {len(tile_gdf_rows)} building polygons written for this tile.")
        print(f"  Finished in {((time.time() - start_t) / 60.0):.2f} minutes")

        del las_ref, las_be, rx, ry, rz, rcls, gx, gy, gz, dtm, dsm_primary, dsm_broad
        del bhm_primary, bhm_broad, bhm_write, building_mask, buildings_tile
        del tile_rows, tile_gdf_rows, tile_buildings_out, pnx, pny, pnh, bnx, bny, bnh
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
        "no_hgt": 0
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

    out_all = os.path.join(base_output_folder, "Meck2002_BldgHt_All.shp")
    buildings_final.to_file(out_all)

    valid_only = buildings_final[buildings_final["h_status"].isin(["ok", "fb_recov"])].copy()
    out_valid = os.path.join(base_output_folder, "Meck2002_BldgHt_Valid.shp")
    valid_only.to_file(out_valid)

    fallback_only = buildings_final[buildings_final["h_status"] == "fb_recov"].copy()
    out_fallback = os.path.join(base_output_folder, "Meck2002_BldgHt_Fallback.shp")
    fallback_only.to_file(out_fallback)

    print(f"Countywide all buildings written: {out_all}")
    print(f"Valid only buildings written: {out_valid}")
    print(f"Fallback recovered buildings written: {out_fallback}")
    print(f"Total county polygons written: {len(buildings_final)}")
    print(f"Valid polygons written: {len(valid_only)}")
    print(f"Fallback recovered polygons written: {len(fallback_only)}")

print("\nAll tiles processed.")