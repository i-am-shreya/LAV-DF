import argparse

import toml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.lavdf import LAVDFDataModule
from model import BATFD
from utils import LrLogger, EarlyStoppingLR

parser = argparse.ArgumentParser(description="BATFD training")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--precision", default=32)
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=1000)
parser.add_argument("--max_epochs", type=int, default=500)
parser.add_argument("--resume", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    config = toml.load(args.config)

    dm = LAVDFDataModule(
        root=args.data_root,
        frame_padding=config["num_frames"],
        max_duration=config["max_duration"],
        batch_size=args.batch_size, num_workers=args.num_workers,
        take_train=args.num_train, take_dev=args.num_val,
        get_meta_attr=BATFD.get_meta_attr
    )

    model = BATFD(
        v_encoder=config["model"]["video_encoder"]["type"],
        a_encoder=config["model"]["audio_encoder"]["type"],
        frame_classifier=config["model"]["frame_classifier"]["type"],
        ve_features=config["model"]["video_encoder"]["hidden_dims"],
        ae_features=config["model"]["audio_encoder"]["hidden_dims"],
        v_cla_feature_in=config["model"]["video_encoder"]["cla_feature_in"],
        a_cla_feature_in=config["model"]["audio_encoder"]["cla_feature_in"],
        boundary_features=config["model"]["boundary_module"]["hidden_dims"],
        boundary_samples=config["model"]["boundary_module"]["samples"],
        temporal_dim=config["num_frames"],
        max_duration=config["max_duration"],
        weight_frame_loss=config["optimizer"]["frame_loss_weight"],
        weight_modal_bm_loss=config["optimizer"]["modal_bm_loss_weight"],
        weight_contrastive_loss=config["optimizer"]["contrastive_loss_weight"],
        contrast_loss_margin=config["optimizer"]["contrastive_loss_margin"],
        weight_decay=config["optimizer"]["weight_decay"],
        learning_rate=config["optimizer"]["learning_rate"],
        distributed=args.gpus > 1
    )

    try:
        precision = int(args.precision)
    except ValueError:
        pass

    trainer = Trainer(log_every_n_steps=50, precision=precision, max_epochs=args.max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"./ckpt/{config['name']}", save_last=True, filename=config["name"] + "-{epoch}-{val_loss:.3f}",
                monitor="val_fusion_bm_loss", mode="min"
            ),
            LrLogger(),
            EarlyStoppingLR(lr_threshold=1e-7)
        ], enable_checkpointing=True,
        accelerator="auto",
        devices=args.gpus,
        strategy=None if args.gpus < 2 else "ddp",
        resume_from_checkpoint=args.resume
    )

    trainer.fit(model, dm)
