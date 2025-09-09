import json
from pathlib import Path

import fire
import geopandas as gpd
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from fm4cs.data.fm4cs_footprint_builder import FootprintBuilder
from fm4cs.data.fm4cs_sample_index_builder import FM4CSDatasetIndexBuilder
from fm4cs.data.utils import parse_products


def build_index(
    version="v2",
    products=[  # noqa: B006
        "S2-10m",
        "S2-20m",
        "S2-60m",
        "S1-10m",
        "S1-60m",
        "S3-250m",
        "S3-500m",
        "S3-1000m",
    ],
    split=None,
    ground_cover=2880,
    data_dir="/lokal-uten-backup-8tb/pro/fm4cs/usr/tforgaard/data/lumi/v2",
    kml_grid_path=None,  # "/nr/bamjo/user/tforgaard/Projects/fm4cs/data/drawn 1.kml",
    num_workers=0,
    save_fig=False,
    out_dir=None,
    include_filter=None,
):
    if version == "v1":
        print("Building dataset index")
        dataset = FM4CSDatasetIndexBuilder(
            paths=data_dir,
            products=parse_products(products),
            split=split,
            ground_cover=ground_cover,
            data_percent=1.0,
            out_dir=out_dir,
            include_filter=include_filter,
            save_fig=save_fig,
        )
    elif version == "v2":
        print("Building dataset index v2")
        dataset = FootprintBuilder(
            paths=data_dir,
            products=parse_products(products),
            split=split,
            kml_path=kml_grid_path,
            ground_cover=ground_cover,
            data_percent=1.0,
            out_dir=out_dir,
            include_filter=include_filter,
            save_fig=save_fig,
        )
    print("number of unique samples", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )

    dataset_metadata = {}
    misses = 0
    misses_files = []
    found_files = []
    total_gdf = None
    torch.manual_seed(402)  # 22)
    i = 0
    for sample_b in tqdm(dataloader, total=len(dataloader), desc="Building dataset"):
        sample = sample_b[0]
        file_path, gdf, total_area, success = sample
        if not success:
            misses += 1
            misses_files.append(str(file_path))
        else:
            gdf["id"] = i
            gdf["file_path"] = str(file_path)
            i += 1
            if total_gdf is None:
                total_gdf = gdf
                crs = gdf.crs
            else:
                total_gdf = pd.concat([total_gdf, gdf.to_crs(crs)], ignore_index=True)
            found_files.append(str(file_path))
            dataset_metadata[str(file_path)] = total_area

        # print(f"Batch {i_batch}, File: {sample[0][0]}")
        # for sample_area in sample[0][1]:
        #     for k,v in sample_area.items():
        #         print(f"{k}: {v}")
        #     print()

        # print("===================================")
        # break

    print(f"Misses: {misses}")
    print(f"Missing percentage: {misses / len(dataloader) * 100:.2f}%")
    print(f"Misses files: {misses_files}")
    print("===============================")
    print(f"Found {len(found_files)} files")

    # if out_dir is None:
    out_dir = f"{data_dir}_geometa.json"
    out_dir = Path(out_dir)

    if total_gdf is not None:
        gdf_file = out_dir.parent / "total_gdf.geojson"
        total_gdf.to_file(gdf_file, driver="GeoJSON")

    with open(out_dir, "w") as f:
        json.dump(dataset_metadata, f)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    fire.Fire(build_index)
