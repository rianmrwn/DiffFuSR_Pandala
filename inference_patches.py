import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import subprocess
import shutil
from tqdm import tqdm
import math

def tile_and_process_image(input_tiff, output_folder, output_name, checkpoint_path, tile_size=1024, scale=4, overlap=0):
    """
    Tiles a large GeoTIFF image, processes each tile with DiffFuSR, and stitches results back together.

    Args:
        input_tiff: Path to input Sentinel-2 GeoTIFF (with bands 4,3,2)
        output_folder: Folder to save the final output
        output_name: Name of the final stitched output file
        checkpoint_path: Path to the DiffFuSR model checkpoint
        tile_size: Size of tiles (default 1024x1024)
        scale: Super-resolution scale factor (default 4)
        overlap: Overlap between tiles in pixels (default 0)
    """
    # Create temporary directories
    temp_tiles_dir = os.path.join(output_folder, "temp_tiles")
    temp_sr_dir = os.path.join(output_folder, "temp_sr")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_tiles_dir, exist_ok=True)
    os.makedirs(temp_sr_dir, exist_ok=True)

    # Open the input image
    with rasterio.open(input_tiff) as src:
        # Check if we have the right band order for Sentinel-2 RGB (4,3,2)
        # If not, we'll need to extract and reorder the bands
        band_count = src.count

        # Get image dimensions
        height = src.height
        width = src.width

        # Calculate number of tiles
        n_tiles_height = math.ceil(height / (tile_size - overlap))
        n_tiles_width = math.ceil(width / (tile_size - overlap))

        print(f"Image dimensions: {width}x{height}")
        print(f"Tiling into {n_tiles_width}x{n_tiles_height} tiles of size {tile_size}x{tile_size}")

        # Get metadata for output
        meta = src.meta.copy()

        # Create a list to store tile information
        tiles_info = []

        # Generate tiles
        for i in range(n_tiles_height):
            for j in range(n_tiles_width):
                # Calculate tile coordinates with overlap
                row_start = i * (tile_size - overlap)
                col_start = j * (tile_size - overlap)

                # Handle edge cases
                row_end = min(row_start + tile_size, height)
                col_end = min(col_start + tile_size, width)

                # Adjust start positions for tiles smaller than tile_size
                row_start = max(0, row_end - tile_size)
                col_start = max(0, col_end - tile_size)

                # Define window for reading
                window = Window(col_start, row_start, col_end - col_start, row_end - row_start)

                # Read the data
                data = src.read(window=window)

                # Create tile metadata
                tile_meta = src.meta.copy()
                tile_transform = rasterio.windows.transform(window, src.transform)
                tile_meta.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': tile_transform
                })

                # Save tile info
                tile_path = os.path.join(temp_tiles_dir, f"tile_{i}_{j}.tif")
                tiles_info.append({
                    'path': tile_path,
                    'window': window,
                    'transform': tile_transform,
                    'row': i,
                    'col': j
                })

                # Write the tile
                with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                    dst.write(data)

        # Process each tile with DiffFuSR
        for tile_info in tqdm(tiles_info, desc="Processing tiles"):
            tile_path = tile_info['path']
            sr_output_name = f"sr_tile_{tile_info['row']}_{tile_info['col']}.tif"
            sr_output_path = os.path.join(temp_sr_dir, sr_output_name)

            # Run the DiffFuSR inference
            cmd = [
                "python", "/content/DiffFuSR_Pandala/inference.py",
                "--input_tiff", tile_path,
                "--output_folder", temp_sr_dir,
                "--output_name", sr_output_name,
                "--checkpoint", checkpoint_path,
                "--scale", str(scale)
            ]

            subprocess.run(cmd, check=True)
            tile_info['sr_path'] = sr_output_path

        # Prepare for stitching
        # The output will be scale times larger
        out_height = height * scale
        out_width = width * scale

        # Update metadata for the final output
        out_meta = meta.copy()
        out_meta.update({
            'height': out_height,
            'width': out_width,
            'transform': Affine(
                src.transform.a / scale,
                src.transform.b,
                src.transform.c,
                src.transform.d,
                src.transform.e / scale,
                src.transform.f
            )
        })

        # Create the output file
        output_path = os.path.join(output_folder, output_name)
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            # Initialize with zeros or nodata
            if 'nodata' in out_meta:
                nodata = out_meta['nodata']
                dst_data = np.full((band_count, out_height, out_width), nodata, dtype=out_meta['dtype'])
            else:
                dst_data = np.zeros((band_count, out_height, out_width), dtype=out_meta['dtype'])

            # Stitch tiles together
            for tile_info in tqdm(tiles_info, desc="Stitching tiles"):
                if os.path.exists(tile_info['sr_path']):
                    with rasterio.open(tile_info['sr_path']) as sr_src:
                        sr_data = sr_src.read()

                        # Calculate position in the output image
                        row_start = tile_info['window'].row_off * scale
                        col_start = tile_info['window'].col_off * scale
                        row_end = row_start + sr_src.height
                        col_end = col_start + sr_src.width

                        # Handle edge cases
                        row_end = min(row_end, out_height)
                        col_end = min(col_end, out_width)

                        # Copy data to the output array
                        dst_data[:, row_start:row_end, col_start:col_end] = sr_data[:, :(row_end-row_start), :(col_end-col_start)]

            # Write the final stitched image
            dst.write(dst_data)

    print(f"Successfully processed and stitched image to: {output_path}")

    # Clean up temporary files if needed
    # shutil.rmtree(temp_tiles_dir)
    # shutil.rmtree(temp_sr_dir)
    print("Temporary files are kept in case you need them. You can delete them manually.")

if __name__ == "__main__":
    # Example usage
    input_tiff = "/content/sentinel2_Sawit_fix.tif"
    output_folder = "/content/output_SR/"
    output_name = "sr_sawit2025_baru_stitched.tif"
    checkpoint_path = "/content/DiffFuSR_Pandala/check/data/naip_harm/last.ckpt"

    tile_and_process_image(
        input_tiff=input_tiff,
        output_folder=output_folder,
        output_name=output_name,
        checkpoint_path=checkpoint_path,
        tile_size=1024,
        scale=4,
        overlap=32  # Small overlap to avoid edge artifacts
    )
