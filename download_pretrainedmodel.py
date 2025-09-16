# Download model untuk RGB

import os
import wget

# download pretrained model dan hparams.yaml dari huggingface
# # Data Model Pretrained:

# *   NAIP Harmony =

# https://huggingface.co/NorskRegnesentralSTI/DiffFuSR/resolve/main/logs/blindsrsnf_aniso_naip_degraded_harm_large/version_1/checkpoint/last.ckpt?download=true
# https://huggingface.co/NorskRegnesentralSTI/DiffFuSR/resolve/main/logs/blindsrsnf_aniso_naip_degraded_harm_large/version_1/hparams.yaml?download=true

# *   NAIP Non-Harmony =

# https://huggingface.co/NorskRegnesentralSTI/DiffFuSR/resolve/main/logs/blindsrsnf_aniso_naip_degraded_not_harm_large/version_0/checkpoints/last.ckpt?download=true
# https://huggingface.co/NorskRegnesentralSTI/DiffFuSR/resolve/main/logs/blindsrsnf_aniso_naip_degraded_not_harm_large/version_0/hparams.yaml?download=true

# *   Worldstrat =

# https://huggingface.co/NorskRegnesentralSTI/DiffFuSR/resolve/main/logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/checkpoints/last.ckpt?download=true
# https://huggingface.co/NorskRegnesentralSTI/DiffFuSR/resolve/main/logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/hparams.yaml?download=true


download_url = "https://huggingface.co/NorskRegnesentralSTI/DiffFuSR/resolve/main/logs/blindsrsnf_aniso_naip_degraded_harm_large/version_1/checkpoint/last.ckpt?download=true"
destination_folder = "/content/DiffFuSR_Pandala/check/data/naip_harm"

download_url2 = "https://huggingface.co/NorskRegnesentralSTI/DiffFuSR/resolve/main/logs/blindsrsnf_aniso_naip_degraded_harm_large/version_1/hparams.yaml?download=true"
destination_folder2 = "/content/DiffFuSR_Pandala/check/data"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Extract the filename from the URL
filename = download_url.split('/')[-1].split('?')[0]
destination_path = os.path.join(destination_folder, filename)

filename2 = download_url2.split('/')[-1].split('?')[0]
destination_path2 = os.path.join(destination_folder2, filename2)

# Download the file using wget
wget.download(download_url, out=destination_path)
wget.download(download_url2, out=destination_path2)
