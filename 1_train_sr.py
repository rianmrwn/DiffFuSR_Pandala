import argparse

import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from data import create_data_module
from litsr.utils import read_yaml
from models import create_model, load_model

torch.backends.cudnn.benchmark = True


def train_pipeline(args):
    exp_name = args.expname or args.config.split("/")[-1].split(".")[0]
    config = read_yaml(args.config)

    logger = loggers.tensorboard.TensorBoardLogger(
        args.save_path, name=exp_name, default_hp_metric=False
    )
    checkpoint_period = ModelCheckpoint(
        save_last=False,
        save_top_k=-1,
        every_n_epochs=config.trainer.get("save_period", 50),
    )

    assert config.trainer.save_top_k > 0
    checkpoint_best = ModelCheckpoint(
        monitor="val/psnr",
        save_last=True,
        save_top_k=config.trainer.get("save_top_k", 2),
        mode="max",
        filename="epoch={epoch:d}-psnr={val/psnr:.2f}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )
    device_list = [int(d) for d in args.devices.split(",")]   # e.g. "0,1" → [0,1]

    trainer_args = {
        "accelerator": args.accelerator,      # "gpu", "cpu", "auto"…
        "devices": device_list,               # [0,1]  or  2
        "logger": logger,
        "default_root_dir": args.save_path,
        "limit_val_batches": 20,
        "check_val_every_n_epoch": config.trainer.check_val_every_n_epoch,
        "callbacks": [checkpoint_period, checkpoint_best],
        "max_epochs": config.trainer.max_epochs,
    }

    if args.resume:
        trainer_args["resume_from_checkpoint"] = args.resume

    trainer = Trainer(**trainer_args)#,accumulate_grad_batches=4
    if args.finetune:
        model = load_model(config, args.finetune, False)
    else:
        model = create_model(config)

    datamodule = create_data_module(config.data_module)
    trainer.fit(model=model, datamodule=datamodule)


def getTrainParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large.yaml", type=str, help="config file path") # 
    parser.add_argument( # 
        "-r", 
        "--resume",
        default= None, #
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-f",#
        "--finetune",
        default= None, 
        type=str,
        help="path to checkpoint (default: None)",
    )
    parser.add_argument("--accelerator", default="cuda", type=str,
                    help="Hardware accelerator: cpu, gpu, auto…")
    parser.add_argument("--devices",     default="2",  type=str,
                        help="Comma-separated list of device ids, "
                            "or an int for the number of devices")
    parser.add_argument("-s", "--save_path", default="logs", type=str, help="save path")
    parser.add_argument("-e", "--expname", default="", type=str, help="save path")

    return parser


train_parser = getTrainParser()

if __name__ == "__main__":
    args = train_parser.parse_args()
    train_pipeline(args)
