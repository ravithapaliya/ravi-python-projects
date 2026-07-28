import arcpy
import os
import glob

arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")


# ======================================================
# INPUTS
# ======================================================

naip_base_folder = r"D:\Aerial Photos - NAIP"

meck_boundary = r"C:\Users\ravit\Documents\tree research work new\MecklenburgCounty_Boundary\MecklenburgCounty_Boundary.shp"

output_folder = r"D:\Aerial Photos - NAIP\NDVI_Outputs"
tile_ndvi_folder = r"D:\Aerial Photos - NAIP\NDVI_Outputs\Tile_NDVI"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(tile_ndvi_folder, exist_ok=True)


# ======================================================
# YEAR SETTINGS
# ======================================================
# Your 3 band false color imagery is assumed to be:
# Band 1 = NIR
# Band 2 = Red
# Band 3 = Blue
#
# NDVI = (NIR - Red) / (NIR + Red)

year_configs = {
    2012: {
        "input_folder": r"D:\Aerial Photos - NAIP\2012\3-band_False_Color_Imagery",
        "output_name": "Mecklenburg_NDVI_NAIP_2012_1m.tif"
    },
    2017: {
        "input_folder": r"D:\Aerial Photos - NAIP\2017\3-band_False_Color_Imagery",
        "output_name": "Mecklenburg_NDVI_NAIP_2017_1m.tif"
    },
    2023: {
        "input_folder": r"D:\Aerial Photos - NAIP\2023\3-band_False_Color_Imagery",
        "output_name": "Mecklenburg_NDVI_NAIP_2023_1m.tif"
    }
}


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def find_tif_files(folder):
    files = []

    files.extend(glob.glob(os.path.join(folder, "*.tif")))
    files.extend(glob.glob(os.path.join(folder, "*.tiff")))
    files.extend(glob.glob(os.path.join(folder, "**", "*.tif"), recursive=True))
    files.extend(glob.glob(os.path.join(folder, "**", "*.tiff"), recursive=True))

    clean = []
    seen = set()

    for f in files:
        lower = f.lower()

        if lower.endswith(".aux.xml"):
            continue

        if "_NDVI" in os.path.basename(f):
            continue

        if f not in seen:
            seen.add(f)
            clean.append(f)

    return sorted(clean)


def safe_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    name = name.replace(" ", "_")
    return name


def delete_if_exists(path):
    if arcpy.Exists(path):
        arcpy.management.Delete(path)
    elif os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def get_raster_min_max(raster_path):
    arcpy.management.CalculateStatistics(raster_path)

    min_val = arcpy.management.GetRasterProperties(raster_path, "MINIMUM")[0]
    max_val = arcpy.management.GetRasterProperties(raster_path, "MAXIMUM")[0]

    return float(min_val), float(max_val)


# ======================================================
# MAIN PROCESS
# ======================================================

for year, cfg in year_configs.items():

    print("")
    print("=================================================")
    print(f"Processing NDVI for {year}")
    print("=================================================")

    input_folder = cfg["input_folder"]
    final_output = os.path.join(output_folder, cfg["output_name"])

    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}")
        continue

    tif_files = find_tif_files(input_folder)

    print(f"GeoTIFF files found: {len(tif_files)}")

    if len(tif_files) == 0:
        print("No GeoTIFF files found. Skipping this year.")
        continue

    year_tile_folder = os.path.join(tile_ndvi_folder, str(year))
    os.makedirs(year_tile_folder, exist_ok=True)

    ndvi_tiles = []

    for i, tif in enumerate(tif_files, start=1):

        tile_name = safe_name(tif)
        out_tile = os.path.join(year_tile_folder, f"{tile_name}_NDVI.tif")

        if arcpy.Exists(out_tile):
            print(f"[{i}/{len(tif_files)}] Tile NDVI already exists. Skipping.")
            ndvi_tiles.append(out_tile)
            continue

        print(f"[{i}/{len(tif_files)}] Calculating NDVI for: {os.path.basename(tif)}")

        try:
            band_count = int(arcpy.management.GetRasterProperties(tif, "BANDCOUNT")[0])

            if band_count < 3:
                print("Skipping. Raster has fewer than 3 bands.")
                continue

            nir = arcpy.sa.Raster(tif + r"\Band_1")
            red = arcpy.sa.Raster(tif + r"\Band_2")

            nir_float = arcpy.sa.Float(nir)
            red_float = arcpy.sa.Float(red)

            denominator = nir_float + red_float

            ndvi_raw = (nir_float - red_float) / denominator

            ndvi_clean = arcpy.sa.SetNull(
                (denominator == 0) | (ndvi_raw < -1) | (ndvi_raw > 1),
                ndvi_raw
            )

            ndvi_clean.save(out_tile)

            arcpy.management.CalculateStatistics(out_tile)

            min_val, max_val = get_raster_min_max(out_tile)

            print(f"Saved tile NDVI: {out_tile}")
            print(f"Tile NDVI range: {min_val} to {max_val}")

            ndvi_tiles.append(out_tile)

        except Exception as e:
            print(f"Failed tile: {tif}")
            print(e)
            continue

    print(f"NDVI tiles created or found for {year}: {len(ndvi_tiles)}")

    if len(ndvi_tiles) == 0:
        print("No NDVI tiles available for final mosaic.")
        continue

    if arcpy.Exists(final_output):
        print(f"Final countywide NDVI already exists. Deleting old file.")
        delete_if_exists(final_output)

    print("Mosaicking NDVI tiles into final countywide NDVI.")

    temp_mosaic = os.path.join(output_folder, f"Mecklenburg_NDVI_NAIP_{year}_unclipped.tif")
    delete_if_exists(temp_mosaic)

    first_tile = ndvi_tiles[0]
    spatial_ref = arcpy.Describe(first_tile).spatialReference

    arcpy.management.MosaicToNewRaster(
        input_rasters=ndvi_tiles,
        output_location=output_folder,
        raster_dataset_name_with_extension=os.path.basename(temp_mosaic),
        coordinate_system_for_the_raster=spatial_ref,
        pixel_type="32_BIT_FLOAT",
        cellsize="#",
        number_of_bands=1,
        mosaic_method="LAST",
        mosaic_colormap_mode="FIRST"
    )

    arcpy.management.CalculateStatistics(temp_mosaic)

    print("Clipping final NDVI to Mecklenburg County boundary.")

    arcpy.management.Clip(
        in_raster=temp_mosaic,
        rectangle="#",
        out_raster=final_output,
        in_template_dataset=meck_boundary,
        nodata_value="#",
        clipping_geometry="ClippingGeometry",
        maintain_clipping_extent="NO_MAINTAIN_EXTENT"
    )

    arcpy.management.CalculateStatistics(final_output)

    final_min, final_max = get_raster_min_max(final_output)

    print(f"Final clipped NDVI exported: {final_output}")
    print(f"Final NDVI range: {final_min} to {final_max}")

print("")
print("All NDVI processing completed.")