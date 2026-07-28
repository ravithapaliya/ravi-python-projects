import os
import time
import math
import warnings
import gc

import laspy
import numpy as np
import geopandas as gpd
import rasterio

from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.geometry import box
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 1. USER SETTINGS
# =========================================================
bare_earth_folder = r"F:\MecklenburgCountyGIS\other_copy_by_me\LiDAR2023\BareEarth_las"
classified_las_folder = r"F:\MecklenburgCountyGIS\other_copy_by_me\LiDAR2023\Classified_las"
base_output_folder = r"C:\Users\ravit\Documents\building_height_results_2023"
building_shapefile = r"F:\MYPHDRESEARCHDATA\Mecklenburg_2023_BuildingFootprints\Mecklenburg_2023_BuildingFootprints.shp"

target_crs = CRS.from_epsg(2264)

# 1 meter in US Survey Feet
resolution = 3.28084

ground_classes = [2]
building_classes = [6]
noise_classes = [7]
water_classes = [9]
ignored_classes = [10, 11, 12, 13, 14, 15, 16, 17, 18]

max_surface_gap_ft = 4.0
min_points_required = 50
maximum_building_height_ft = 300.0
minimum_building_height_ft = 3.0

skip_completed_tiles = True
chunk_rows = 50

# =========================================================
# 2. LOAD DATA
# =========================================================
os.makedirs(base_output_folder, exist_ok=True)

print("Loading building footprints...")
buildings = gpd.read_file(building_shapefile)

if buildings.crs is None:
    raise ValueError("Building shapefile has no CRS defined.")

if buildings.crs != target_crs:
    buildings = buildings.to_crs(target_crs)

buildings = buildings[buildings.geometry.notnull() & ~buildings.geometry.is_empty].copy()
buildings_sindex = buildings.sindex

classified_files = sorted(
    [f for f in os.listdir(classified_las_folder) if f.lower().endswith(".las")]
)

print(f"Found {len(classified_files)} classified LAS tiles to process.")

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

def rasterize_buildings_for_tile(buildings_gdf, sindex, tile_bbox, out_shape, transform):
    idx = list(sindex.intersection(tile_bbox.bounds))
    if len(idx) == 0:
        return np.zeros(out_shape, dtype=np.uint8), buildings_gdf.iloc[[]].copy()

    subset = buildings_gdf.iloc[idx].copy()
    subset = subset[subset.intersects(tile_bbox)].copy()

    if subset.empty:
        return np.zeros(out_shape, dtype=np.uint8), subset

    mask = rasterize(
        [(geom, 1) for geom in subset.geometry if geom.is_valid],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )
    return mask, subset

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

# =========================================================
# 4. MAIN LOOP
# =========================================================
for filename in classified_files:
    tile_id = os.path.splitext(filename)[0]
    tile_folder = os.path.join(base_output_folder, tile_id)
    os.makedirs(tile_folder, exist_ok=True)

    out_bhm = os.path.join(tile_folder, f"{tile_id}_BHM_1m_2023.tif")
    out_buildings = os.path.join(tile_folder, f"{tile_id}_building_heights_2023.shp")

    if skip_completed_tiles and os.path.exists(out_bhm) and os.path.exists(out_buildings):
        print(f"Skipping completed tile: {tile_id}")
        continue

    print(f"\nProcessing: {tile_id}")
    start_t = time.time()

    try:
        classified_path = os.path.join(classified_las_folder, filename)
        bare_path = find_matching_bare_earth_file(filename, bare_earth_folder)

        if bare_path is None:
            print(f"  Skipping. No matching bare earth LAS found for {filename}")
            continue

        las_cls = laspy.read(classified_path)
        las_be = laspy.read(bare_path)

        x_cls = np.asarray(las_cls.x, dtype=np.float32)
        y_cls = np.asarray(las_cls.y, dtype=np.float32)
        z_cls = np.asarray(las_cls.z, dtype=np.float32)
        cls = np.asarray(las_cls.classification, dtype=np.int16)

        x_be = np.asarray(las_be.x, dtype=np.float32)
        y_be = np.asarray(las_be.y, dtype=np.float32)
        z_be = np.asarray(las_be.z, dtype=np.float32)

        if len(x_cls) == 0 or len(x_be) == 0:
            print("  Skipping. Empty LAS arrays.")
            continue

        print(f"  Classified points: {len(x_cls)}")
        print(f"  Bare earth points: {len(x_be)}")
        print(f"  Classes present: {np.unique(cls)}")
        print(f"  {safe_stats(z_cls, 'Classified Raw Z')}")
        print(f"  {safe_stats(z_be, 'Bare Earth Raw Z')}")

        x_min = math.floor(float(np.min(x_cls)))
        x_max = math.ceil(float(np.max(x_cls)))
        y_min = math.floor(float(np.min(y_cls)))
        y_max = math.ceil(float(np.max(y_cls)))

        ncols = max(1, int(math.ceil((x_max - x_min) / resolution)))
        nrows = max(1, int(math.ceil((y_max - y_min) / resolution)))
        transform = from_origin(x_min, y_max, resolution, resolution)

        print(f"  Tile size: {ncols} cols x {nrows} rows at {resolution:.5f} ft")

        dtm = build_knn_surface(
            x_be, y_be, z_be,
            x_min, y_max, nrows, ncols, resolution,
            max_dist=None,
            chunk_rows=chunk_rows
        )

        building_m = np.isin(cls, building_classes)

        if np.sum(building_m) < min_points_required:
            excluded = ground_classes + noise_classes + water_classes + ignored_classes
            building_m = ~np.isin(cls, excluded)
            print("  Building class 6 is sparse or absent. Falling back to non-ground candidate points.")

        if np.sum(building_m) < min_points_required:
            print(f"  Skipping. Too few building candidate points: {np.sum(building_m)}")
            continue

        dsm = build_knn_surface(
            x_cls[building_m], y_cls[building_m], z_cls[building_m],
            x_min, y_max, nrows, ncols, resolution,
            max_dist=max_surface_gap_ft,
            chunk_rows=chunk_rows
        )

        dsm[np.isnan(dsm)] = dtm[np.isnan(dsm)]

        bhm = np.maximum(dsm - dtm, 0).astype(np.float32)
        bhm[bhm > maximum_building_height_ft] = 0.0
        bhm[bhm < minimum_building_height_ft] = 0.0

        tile_bbox = box(x_min, y_min, x_max, y_max)
        building_mask, buildings_tile = rasterize_buildings_for_tile(
            buildings, buildings_sindex, tile_bbox, bhm.shape, transform
        )
        bhm[building_mask == 0] = 0.0

        bhm = median_filter(bhm, size=2).astype(np.float32)

        print(f"  {safe_stats(dtm, 'DTM')}")
        print(f"  {safe_stats(dsm, 'Building Surface DSM')}")
        print(f"  {safe_stats(bhm, 'Building Height Model')}")

        write_raster(out_bhm, bhm, transform, target_crs)

        if buildings_tile.empty:
            print("  No building footprints intersect this tile.")
            continue

        building_records = []
        for _, row in buildings_tile.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            mask = rasterize(
                [(geom, 1)],
                out_shape=bhm.shape,
                transform=transform,
                fill=0,
                dtype="uint8"
            )

            vals = bhm[mask == 1]
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > 0]

            if vals.size == 0:
                continue

            record = row.copy()
            record["max_h_ft"] = float(np.max(vals))
            record["mean_h_ft"] = float(np.mean(vals))
            record["med_h_ft"] = float(np.median(vals))
            record["max_h_m"] = float(np.max(vals) * 0.3048)
            record["mean_h_m"] = float(np.mean(vals) * 0.3048)
            record["med_h_m"] = float(np.median(vals) * 0.3048)
            building_records.append(record)

        if len(building_records) == 0:
            print("  No buildings with valid height values.")
            continue

        buildings_out = gpd.GeoDataFrame(building_records, crs=target_crs)
        buildings_out.to_file(out_buildings)

        print(f"  Success. {len(building_records)} buildings written.")
        print(f"  Finished in {((time.time() - start_t) / 60.0):.2f} minutes")

        del las_cls, las_be, x_cls, y_cls, z_cls, cls, x_be, y_be, z_be
        del dtm, dsm, bhm, building_mask, buildings_tile, building_records
        gc.collect()

    except MemoryError:
        print("  Error: memory error. Reduce chunk_rows.")
        gc.collect()

    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        gc.collect()

print("\nAll tiles processed.")