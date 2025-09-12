import argparse
import os
import torch
import tifffile
from litsr.utils import read_yaml
from models import load_model

# Sentinel-2 band order for 12-band input
S2_BAND_NAMES = [
    "CoastAerosal", "Blue", "Green", "Red", "RE1", "RE2", "RE3", "NIR", "RE4", "WaterVapor", "SWIR1", "SWIR2"
]

def run_multispectral_inference(input_tiff, output_folder, output_name, checkpoint):
    # Load config from checkpoint folder
    exp_path = os.path.dirname(os.path.dirname(checkpoint))

    # Load model
    device = torch.device("cuda")
    model = load_model(checkpoint, strict=False)
    model.to(device)
    model.eval()

    # Load input Sentinel-2 TIFF (assume shape: H x W x 12)
    img = tifffile.imread(input_tiff)

    # Prepare input tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)  # Shape: 1 x 12 x H x W

    # Run inference
    with torch.no_grad():
        # The model expects a tuple (lr, name)
        sr_result = model.test_step_lr_only((img_tensor, [output_name]))
        sr_img = sr_result["log_img_sr"]  # Shape: C x H x W or H x W x C

    # Save output
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, output_name)
    # If sr_img is torch.Tensor, convert to numpy
    if isinstance(sr_img, torch.Tensor):
        sr_img = sr_img.cpu().numpy()
    # If shape is C x H x W, transpose to H x W x C
    if sr_img.shape[0] == 12:
        sr_img = sr_img.transpose(1, 2, 0)
    tifffile.imwrite(out_path, sr_img.astype("uint16"))
    print(f"Super-resolved multispectral image saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffFuSR Multispectral Inference Script")
    parser.add_argument("--input_tiff", type=str, required=True, help="Path to input Sentinel-2 TIFF file (12 bands)")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save SR output")
    parser.add_argument("--output_name", type=str, required=True, help="Name of output SR image (e.g., sr_12band.tif)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DiffFuSR model checkpoint (.ckpt)")
    args = parser.parse_args()

    run_multispectral_inference(args.input_tiff, args.output_folder, args.output_name, args.checkpoint)

