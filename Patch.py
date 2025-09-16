import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import subprocess
import shutil
from tqdm import tqdm
import math
import argparse
import sys

def tile_and_process_image(input_tiff, output_folder, output_name, checkpoint_path, 
                          tile_size=1024, scale=4, overlap=0, output_8bit=True, 
                          lzw_compression=True, write_world_files=True):
    """
    Tiles a large GeoTIFF image, processes each tile with DiffFuSR, and stitches results back together.

    Args:
        input_tiff: Path to input Sentinel-2 GeoTIFF
        output_folder: Folder to save the final output
        output_name: Name of the final stitched output file
        checkpoint_path: Path to the DiffFuSR model checkpoint
        tile_size: Size of tiles (default 1024x1024)
        scale: Super-resolution scale factor (default 4)
        overlap: Overlap between tiles in pixels (default 0)
        output_8bit: Convert output to 8-bit (default True)
        lzw_compression: Use LZW compression (default True)
        write_world_files: Write .prj and .tfw files (default True)
    """
    # Create temporary directories
    temp_tiles_dir = os.path.join(output_folder, "temp_tiles")
    temp_sr_dir = os.path.join(output_folder, "temp_sr")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_tiles_dir, exist_ok=True)
    os.makedirs(temp_sr_dir, exist_ok=True)

    # Open the input image
    with rasterio.open(input_tiff) as src:
        # Get image dimensions
        height = src.height
        width = src.width
        band_count = src.count

        # Calculate number of tiles
        n_tiles_height = math.ceil(height / (tile_size - overlap))
        n_tiles_width = math.ceil(width / (tile_size - overlap))

        print(f"Image dimensions: {width}x{height}")
        print(f"Number of bands: {band_count}")
        print(f"Tiling into {n_tiles_width}x{n_tiles_height} tiles of size {tile_size}x{tile_size}")

        # Get metadata for output
        meta = src.meta.copy()

        # Create a list to store tile information
        tiles_info = []

        # Generate tiles
        print("Creating tiles...")
        for i in range(n_tiles_height):
            for j in range(n_tiles_width):
                # Calculate tile coordinates with overlap
                row_start = i * (tile_size - overlap)
                col_start = j * (tile_size - overlap)

                # Handle edge cases
                row_end = min(row_start + tile_size, height)
                col_end = min(col_start + tile_size, width)

                # Adjust start positions for tiles smaller than tile_size
                actual_height = row_end - row_start
                actual_width = col_end - col_start

                # Define window for reading
                window = Window(col_start, row_start, actual_width, actual_height)

                # Read the data
                data = src.read(window=window)

                # Create tile metadata
                tile_meta = src.meta.copy()
                tile_transform = rasterio.windows.transform(window, src.transform)
                tile_meta.update({
                    'height': actual_height,
                    'width': actual_width,
                    'transform': tile_transform
                })

                # Save tile info
                tile_path = os.path.join(temp_tiles_dir, f"tile_{i}_{j}.tif")
                tiles_info.append({
                    'path': tile_path,
                    'window': window,
                    'transform': tile_transform,
                    'row': i,
                    'col': j,
                    'actual_height': actual_height,
                    'actual_width': actual_width
                })

                # Write the tile
                with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                    dst.write(data)

        # Process each tile with DiffFuSR
        print("Processing tiles with DiffFuSR...")
        for tile_info in tqdm(tiles_info, desc="Processing tiles"):
            tile_path = tile_info['path']
            sr_output_name = f"sr_tile_{tile_info['row']}_{tile_info['col']}.tif"
            sr_output_path = os.path.join(temp_sr_dir, sr_output_name)

            # Build the command with all options
            cmd = [
                "python", "/content/DiffFuSR_Pandala/inference.py",
                "--input_tiff", tile_path,
                "--output_folder", temp_sr_dir,
                "--output_name", sr_output_name,
                "--checkpoint", checkpoint_path,
                "--scale", str(scale)
            ]

            # Add optional flags based on parameters
            if not output_8bit:
                cmd.append("--no-8bit")
            if not lzw_compression:
                cmd.append("--no-compression")
            if not write_world_files:
                cmd.append("--no-world-files")

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                tile_info['sr_path'] = sr_output_path
            except subprocess.CalledProcessError as e:
                print(f"Error processing tile {tile_info['row']}_{tile_info['col']}: {e}")
                print(f"Command output: {e.stdout}")
                print(f"Command error: {e.stderr}")
                tile_info['sr_path'] = None

        # Prepare for stitching
        print("Preparing for stitching...")
        # The output will be scale times larger
        out_height = height * scale
        out_width = width * scale

        # Update metadata for the final output
        out_meta = meta.copy()
        
        # Update transform for super-resolution
        new_transform = Affine(
            src.transform.a / scale,  # pixel width
            src.transform.b,          # rotation
            src.transform.c,          # x offset
            src.transform.d,          # rotation  
            src.transform.e / scale,  # pixel height (negative)
            src.transform.f           # y offset
        )
        
        out_meta.update({
            'height': out_height,
            'width': out_width,
            'transform': new_transform
        })

        # Set data type and compression
        if output_8bit:
            out_meta['dtype'] = 'uint8'
        
        if lzw_compression:
            out_meta['compress'] = 'lzw'

        # Create the output file
        output_path = os.path.join(output_folder, output_name)
        print("Stitching tiles together...")
        
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            # Initialize with zeros or nodata
            if 'nodata' in out_meta and out_meta['nodata'] is not None:
                nodata = out_meta['nodata']
                fill_value = nodata
            else:
                fill_value = 0

            # Process and stitch tiles
            for tile_info in tqdm(tiles_info, desc="Stitching tiles"):
                if tile_info['sr_path'] and os.path.exists(tile_info['sr_path']):
                    try:
                        with rasterio.open(tile_info['sr_path']) as sr_src:
                            sr_data = sr_src.read()

                            # Calculate position in the output image
                            row_start = tile_info['window'].row_off * scale
                            col_start = tile_info['window'].col_off * scale
                            
                            # Get the actual dimensions of the SR tile
                            sr_height, sr_width = sr_data.shape[1], sr_data.shape[2]
                            
                            row_end = row_start + sr_height
                            col_end = col_start + sr_width

                            # Handle edge cases
                            row_end = min(row_end, out_height)
                            col_end = min(col_end, out_width)
                            
                            # Adjust sr_data if needed
                            actual_height = row_end - row_start
                            actual_width = col_end - col_start
                            
                            if actual_height < sr_height or actual_width < sr_width:
                                sr_data = sr_data[:, :actual_height, :actual_width]

                            # Write data to the output file
                            window_out = Window(col_start, row_start, actual_width, actual_height)
                            dst.write(sr_data, window=window_out)
                            
                    except Exception as e:
                        print(f"Error stitching tile {tile_info['row']}_{tile_info['col']}: {e}")
                else:
                    print(f"Skipping missing tile: {tile_info['row']}_{tile_info['col']}")

        # Write world file and projection file if requested
        if write_world_files:
            try:
                # Write .tfw file
                tfw_path = output_path.replace('.tif', '.tfw')
                with open(tfw_path, 'w') as f:
                    f.write(f"{new_transform.a}\n")      # pixel width
                    f.write(f"{new_transform.b}\n")      # rotation
                    f.write(f"{new_transform.d}\n")      # rotation
                    f.write(f"{new_transform.e}\n")      # pixel height
                    f.write(f"{new_transform.c}\n")      # x coordinate of upper left
                    f.write(f"{new_transform.f}\n")      # y coordinate of upper left
                print(f"World file saved: {tfw_path}")

                # Write .prj file (copy from source if available)
                prj_path = output_path.replace('.tif', '.prj')
                if src.crs:
                    with open(prj_path, 'w') as f:
                        f.write(src.crs.to_wkt())
                    print(f"Projection file saved: {prj_path}")
                    
            except Exception as e:
                print(f"Error writing world/projection files: {e}")

    print(f"Successfully processed and stitched image to: {output_path}")
    print(f"Output dimensions: {out_width}x{out_height}")

    # Ask user if they want to clean up temporary files (only in interactive mode)
    try:
        cleanup = input("Do you want to delete temporary files? (y/n): ").lower().strip()
        if cleanup == 'y':
            try:
                shutil.rmtree(temp_tiles_dir)
                shutil.rmtree(temp_sr_dir)
                print("Temporary files deleted.")
            except Exception as e:
                print(f"Error deleting temporary files: {e}")
        else:
            print(f"Temporary files kept in: {temp_tiles_dir} and {temp_sr_dir}")
    except:
        # If input() fails (like in non-interactive environments), just keep the files
        print(f"Temporary files kept in: {temp_tiles_dir} and {temp_sr_dir}")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Tile and Process Large Images with DiffFuSR")
    parser.add_argument("--input_tiff", type=str, required=True, help="Path to input GeoTIFF file")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the final output")
    parser.add_argument("--output_name", type=str, required=True, help="Name of the final stitched output file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DiffFuSR model checkpoint")
    parser.add_argument("--tile_size", type=int, default=1024, help="Size of tiles (default 1024)")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=4, help="Super-resolution scale factor")
    parser.add_argument("--overlap", type=int, default=32, help="Overlap between tiles in pixels")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit conversion")
    parser.add_argument("--no-compression", action="store_true", help="Disable LZW compression")
    parser.add_argument("--no-world-files", action="store_true", help="Don't write .prj and .tfw files")
    
    args = parser.parse_args()

    tile_and_process_image(
        input_tiff=args.input_tiff,
        output_folder=args.output_folder,
        output_name=args.output_name,
        checkpoint_path=args.checkpoint,
        tile_size=args.tile_size,
        scale=args.scale,
        overlap=args.overlap,
        output_8bit=not args.no_8bit,
        lzw_compression=not args.no_compression,
        write_world_files=not args.no_world_files
    )

if __name__ == "__main__":
    main()
