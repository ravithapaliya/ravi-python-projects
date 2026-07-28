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
bare_earth_folder = r"F:\MecklenburgCountyGIS\other_copy_by_me\LiDAR2023\BareEarth_las"
classified_las_folder = r"F:\MecklenburgCountyGIS\other_copy_by_me\LiDAR2023\Classified_las"
base_output_folder = r"C:\Users\ravit\Documents\tree research work new\Batch_Results2023_1m_Final"
building_shapefile = r"F:\MYPHDRESEARCHDATA\Mecklenburg_2023_BuildingFootprints\Mecklenburg_2023_BuildingFootprints.shp"

target_crs = CRS.from_epsg(2264)

# 1 meter in US Survey Feet
resolution = 3.28084

# Expected ASPRS classes
ground_classes = [2]
vegetation_classes = [3, 4, 5]
building_classes = [6]
noise_classes = [7]
water_classes = [9]
ignored_classes = [10, 11, 12, 13, 14, 15, 16, 17, 18]

minimum_canopy_write_height = 1.0
minimum_tree_height = 6.56
segmentation_height_threshold = 10.0
maximum_tree_height = 150.0

max_canopy_gap_ft = 4.0
min_points_required = 100

gaussian_sigma = 1.2
peak_min_distance = 2

min_crown_area_sqft = 20.0
max_crown_area_sqft = 2500.0
max_extent = 0.85
min_circularity = 0.20

skip_completed_tiles = True
chunk_rows = 300

# =========================================================
# 2. LOAD DATA
# =========================================================
os.makedirs(base_output_folder, exist_ok=True)

print("Loading 2023 building footprints...")
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
print(f"Found {len(classified_files)} classified LAS tiles.")

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

    out_chm = os.path.join(tile_folder, f"{tile_id}_CHM_1m_2023.tif")
    out_shp = os.path.join(tile_folder, f"{tile_id}_tree_crowns_1m_2023.shp")

    if skip_completed_tiles and os.path.exists(out_chm) and os.path.exists(out_shp):
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
        print(f"  Classes present in classified LAS: {np.unique(cls)}")
        print(f"  {safe_stats(z_cls, 'Classified Raw Z')}")
        print(f"  {safe_stats(z_be, 'Bare Earth Raw Z')}")

        x_min = math.floor(float(np.min(x_cls)))
        x_max = math.ceil(float(np.max(x_cls)))
        y_min = math.floor(float(np.min(y_cls)))
        y_max = math.ceil(float(np.max(y_cls)))

        ncols = max(1, int(math.ceil((x_max - x_min) / resolution)))
        nrows = max(1, int(math.ceil((y_max - y_min) / resolution)))
        transform = from_origin(x_min, y_max, resolution, resolution)

        print(f"  Tile size: {ncols} cols x {nrows} rows at {resolution:.5f} ft, about 1 meter")

        # DTM from bare earth LAS
        dtm = build_knn_surface(
            x_be, y_be, z_be,
            x_min, y_max, nrows, ncols, resolution,
            max_dist=None,
            chunk_rows=chunk_rows
        )

        # Prefer vegetation classes 3, 4, 5
        canopy_m = np.isin(cls, vegetation_classes)

        # Fallback to non ground candidate points if vegetation classes are missing
        if np.sum(canopy_m) < min_points_required:
            excluded = ground_classes + building_classes + noise_classes + water_classes + ignored_classes
            canopy_m = ~np.isin(cls, excluded)
            print("  Vegetation classes 3, 4, 5 were sparse or absent. Falling back to non-ground candidate points.")

        if np.sum(canopy_m) < min_points_required:
            print(f"  Skipping. Too few canopy candidate points: {np.sum(canopy_m)}")
            continue

        ground_ref = np.median(z_be)
        canopy_valid = canopy_m & (z_cls >= ground_ref) & (z_cls <= ground_ref + maximum_tree_height + 20.0)

        if np.sum(canopy_valid) < min_points_required:
            print("  Skipping. Too few canopy candidate points after sanity filter.")
            continue

        dsm = build_knn_surface(
            x_cls[canopy_valid], y_cls[canopy_valid], z_cls[canopy_valid],
            x_min, y_max, nrows, ncols, resolution,
            max_dist=max_canopy_gap_ft,
            chunk_rows=chunk_rows
        )

        dsm[np.isnan(dsm)] = dtm[np.isnan(dsm)]

        chm = np.maximum(dsm - dtm, 0).astype(np.float32)

        print(f"  {safe_stats(dtm, 'DTM')}")
        print(f"  {safe_stats(dsm, 'DSM')}")
        print(f"  {safe_stats(chm, 'CHM before clamp')}")

        chm[chm > maximum_tree_height] = 0.0
        chm[chm < minimum_canopy_write_height] = 0.0

        chm = median_filter(chm, size=2).astype(np.float32)

        tile_bbox = box(x_min, y_min, x_max, y_max)
        building_mask = rasterize_buildings_for_tile(
            buildings, buildings_sindex, tile_bbox, chm.shape, transform
        )
        chm[building_mask == 1] = 0.0

        print(f"  {safe_stats(chm, 'CHM before write')}")

        write_raster(out_chm, chm, transform, target_crs)

        # Segmentation
        smoothed = gaussian(chm, sigma=gaussian_sigma, preserve_range=True).astype(np.float32)
        tree_mask = smoothed >= segmentation_height_threshold

        if not np.any(tree_mask):
            print("  No canopy above segmentation threshold.")
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

        del las_cls, las_be, x_cls, y_cls, z_cls, cls, x_be, y_be, z_be, dtm, dsm, chm, smoothed, labels
        gc.collect()

    except MemoryError:
        print("  Error: memory error. Reduce chunk_rows.")
        gc.collect()

    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        gc.collect()

print("\nAll tiles processed.")