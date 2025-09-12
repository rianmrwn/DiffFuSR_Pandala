import argparse
import os
import torch
import tifffile
from litsr.utils import read_yaml
from models import load_model

def run_inference(input_tiff, output_folder, output_name, checkpoint, scale):
    # Load config from checkpoint folder
    exp_path = os.path.dirname(os.path.dirname(checkpoint))
    config = read_yaml(os.path.join(exp_path, "hparams.yaml"))

    # Load model
    device = torch.device("cpu")
    model = load_model(config, checkpoint, strict=False)
    model.to(device)
    model.eval()

    # Load input Sentinel-2 TIFF (assume shape: H x W x 12)
    img = tifffile.imread(input_tiff)
    
    # Prepare input tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)  # Shape: 1 x 12 x H x W

    # Run inference
    with torch.no_grad():
        sr_result = model.test_step_lr_only((img_tensor, [output_name]))
        sr_img = sr_result["log_img_sr"]  # Shape: C x H x W or H x W x C

    # Save output
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, output_name)
    # If sr_img is torch.Tensor, convert to numpy
    if isinstance(sr_img, torch.Tensor):
        sr_img = sr_img.cpu().numpy()
    # If shape is C x H x W, transpose to H x W x C
    if sr_img.shape[0] in [3, 12]:
        sr_img = sr_img.transpose(1, 2, 0)
    # tifffile.imwrite(out_path, sr_img.astype("float32"))
    tifffile.imwrite(out_path, sr_img.astype("uint16"))
    print(f"Super-resolved image saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffFuSR Inference Script")
    parser.add_argument("--input_tiff", type=str, required=True, help="Path to input Sentinel-2 TIFF file")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save SR output")
    parser.add_argument("--output_name", type=str, required=True, help="Name of output SR image (e.g., sr.tif)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DiffFuSR model checkpoint (.ckpt)")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=4, help="Super-resolution scale factor (x2 or x4)")
    args = parser.parse_args()


    run_inference(args.input_tiff, args.output_folder, args.output_name, args.checkpoint, args.scale)

