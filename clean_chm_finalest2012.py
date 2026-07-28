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
from shapely.geometry import box, shape
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from rasterio import features

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 1. USER SETTINGS
# =========================================================
las_folder = r"F:\MecklenburgCountyGIS\LiDAR2012"
base_output_folder = r"C:\Users\ravit\Documents\tree research work new\Batch_Results2012_1m_Final"
building_shapefile = r"F:\MYPHDRESEARCHDATA\Mecklenburg_2012_BuildingFootprints\Buildings.shp"

# NAD 1983 StatePlane North Carolina FIPS 3200 US Feet
target_crs = CRS.from_epsg(2264)

# 1 meter in feet
resolution = 3.28084

# LAS classes
ground_classes = [2]
noise_classes = [7]
water_classes = [9]
ignored_classes = [10, 12, 17, 18]

# Height thresholds in feet
minimum_canopy_write_height = 1.0       # keep CHM variation, remove only tiny noise
minimum_tree_height = 6.56              # about 2 meters, used for tree candidate points
segmentation_height_threshold = 10.0    # about 3.05 meters
maximum_tree_height = 150.0

# Distance limits in feet
max_ground_query_distance = 30.0
max_canopy_gap_ft = 4.0

# Minimum points
min_points_required = 100

# Segmentation settings for 1 meter CHM
gaussian_sigma = 1.2
peak_min_distance = 2

# Crown filters
min_crown_area_sqft = 20.0
max_crown_area_sqft = 2500.0
max_extent = 0.85
min_circularity = 0.20

skip_completed_tiles = True

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

las_files = sorted([f for f in os.listdir(las_folder) if f.lower().endswith(".las")])
print(f"Found {len(las_files)} LAS tiles.")

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

def build_knn_surface(xs, ys, zs, x_min, y_max, nrows, ncols, res, max_dist=None, chunk_rows=300):
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

def normalize_point_heights(x_all, y_all, z_all, ground_x, ground_y, ground_z, max_dist):
    tree = cKDTree(np.column_stack([ground_x, ground_y]))
    dist, idx = tree.query(np.column_stack([x_all, y_all]), k=1)

    z_ground = ground_z[idx]
    h = z_all - z_ground

    valid = np.isfinite(h) & np.isfinite(dist) & (dist <= max_dist)
    return h, valid

def rasterize_buildings_for_tile(buildings_gdf, sindex, tile_bbox, out_shape, transform):
    idx = list(sindex.intersection(tile_bbox.bounds))
    if len(idx) == 0:
        return np.zeros(out_shape, dtype=np.uint8)

    subset = buildings_gdf.iloc[idx]
    subset = subset[subset.intersects(tile_bbox)]

    if subset.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    mask = rasterize(
        [(geom, 1) for geom in subset.geometry if geom.is_valid],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )
    return mask

# =========================================================
# 4. MAIN LOOP
# =========================================================
for filename in las_files:
    tile_id = os.path.splitext(filename)[0]
    tile_folder = os.path.join(base_output_folder, tile_id)
    os.makedirs(tile_folder, exist_ok=True)

    out_chm = os.path.join(tile_folder, f"{tile_id}_CHM_1m_2012.tif")
    out_shp = os.path.join(tile_folder, f"{tile_id}_tree_crowns_1m_2012.shp")

    if skip_completed_tiles and os.path.exists(out_chm) and os.path.exists(out_shp):
        print(f"Skipping completed tile: {tile_id}")
        continue

    print(f"\n--- Processing: {tile_id} ---")
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

        excluded_classes = set(ground_classes + noise_classes + water_classes + ignored_classes)
        candidate_m = ~np.isin(cls, list(excluded_classes))

        if ground_m.sum() < min_points_required:
            print(f"  Skipping. Too few ground points: {ground_m.sum()}")
            continue

        if candidate_m.sum() < min_points_required:
            print(f"  Skipping. Too few non ground candidate points: {candidate_m.sum()}")
            continue

        x_min = math.floor(float(np.min(x)))
        x_max = math.ceil(float(np.max(x)))
        y_min = math.floor(float(np.min(y)))
        y_max = math.ceil(float(np.max(y)))

        ncols = max(1, int(math.ceil((x_max - x_min) / resolution)))
        nrows = max(1, int(math.ceil((y_max - y_min) / resolution)))
        transform = from_origin(x_min, y_max, resolution, resolution)

        print(f"  Tile size: {ncols} cols x {nrows} rows at {resolution:.5f} ft, about 1 meter")

        # Ground surface from class 2
        dtm = build_knn_surface(
            x[ground_m], y[ground_m], z[ground_m],
            x_min, y_max, nrows, ncols, resolution,
            max_dist=None,
            chunk_rows=300
        )

        print(f"  {safe_stats(dtm, 'DTM')}")

        # Normalize all non ground candidate points
        h_norm, valid_norm = normalize_point_heights(
            x[candidate_m], y[candidate_m], z[candidate_m],
            x[ground_m], y[ground_m], z[ground_m],
            max_ground_query_distance
        )

        cand_x = x[candidate_m][valid_norm]
        cand_y = y[candidate_m][valid_norm]
        cand_h = h_norm[valid_norm]

        # Keep only likely tree points
        veg_keep = (cand_h >= minimum_tree_height) & (cand_h <= maximum_tree_height)
        cand_x = cand_x[veg_keep]
        cand_y = cand_y[veg_keep]
        cand_h = cand_h[veg_keep]

        if len(cand_h) < min_points_required:
            print("  Skipping. Too few above ground vegetation candidate points after normalization.")
            continue

        print(f"  Candidate above ground points kept: {len(cand_h)}")
        print(f"  {safe_stats(cand_h, 'Normalized canopy heights')}")

        # Build canopy surface from normalized heights with a limited gap distance
        chm = build_knn_surface(
            cand_x, cand_y, cand_h,
            x_min, y_max, nrows, ncols, resolution,
            max_dist=max_canopy_gap_ft,
            chunk_rows=300
        )

        # Cells without nearby canopy become zero
        chm = np.nan_to_num(chm, nan=0.0).astype(np.float32)

        # Remove only tiny noise, keep low canopy variation
        chm[chm < minimum_canopy_write_height] = 0.0

        # Light despeckle
        chm = median_filter(chm, size=2).astype(np.float32)

        # Remove buildings
        tile_bbox = box(x_min, y_min, x_max, y_max)
        building_mask = rasterize_buildings_for_tile(
            buildings, buildings_sindex, tile_bbox, chm.shape, transform
        )
        chm[building_mask == 1] = 0.0

        print(f"  {safe_stats(chm, 'CHM before write')}")

        write_raster(out_chm, chm, transform, target_crs)

        # =====================================================
        # 5. TREE SEGMENTATION
        # =====================================================
        smoothed = gaussian(chm, sigma=gaussian_sigma, preserve_range=True).astype(np.float32)
        tree_mask = smoothed >= segmentation_height_threshold

        if np.count_nonzero(tree_mask) == 0:
            print("  No tree mask pixels found.")
            continue

        coords = peak_local_max(
            smoothed,
            min_distance=peak_min_distance,
            labels=tree_mask
        )

        if len(coords) == 0:
            print("  No tree peaks detected.")
            continue

        markers = np.zeros_like(smoothed, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

        labels = watershed(-smoothed, markers, mask=tree_mask).astype(np.int32)

        label_geoms = {
            int(v): shape(g)
            for g, v in features.shapes(labels, transform=transform)
            if v > 0
        }

        tree_recs = []
        for region in regionprops(labels, intensity_image=chm):
            area = region.area * (resolution ** 2)
            perimeter = region.perimeter * resolution
            circularity = (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

            if (
                min_crown_area_sqft <= area <= max_crown_area_sqft and
                region.extent <= max_extent and
                circularity >= min_circularity and
                region.label in label_geoms
            ):
                tree_recs.append({
                    "geometry": label_geoms[region.label],
                    "height_ft": float(region.intensity_max),
                    "height_m": float(region.intensity_max * 0.3048),
                    "area_sqft": float(area),
                    "area_sqm": float(area * 0.092903),
                    "circularity": float(circularity)
                })

        if len(tree_recs) == 0:
            print("  No segments passed the crown filters.")
            continue

        crowns_gdf = gpd.GeoDataFrame(tree_recs, crs=target_crs)
        crowns_gdf.to_file(out_shp)

        print(f"  Success. {len(tree_recs)} trees written.")
        print(f"  Finished in {((time.time() - start_t) / 60.0):.2f} minutes")

        del las, x, y, z, cls, dtm, h_norm, cand_x, cand_y, cand_h, chm, smoothed, labels
        gc.collect()

    except MemoryError:
        print(f"  Error processing {filename}: memory error. Reduce chunk_rows in build_knn_surface.")
        gc.collect()

    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        gc.collect()

print("\n--- All tiles processed ---")