import json
import os
import warnings
from argparse import ArgumentParser
from datetime import datetime
from typing import Callable, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torchmetrics
import transformers
from callbacks import BagPredictionWriter
from data_wsi import TileLevelDataModule
from encoders import get_encoder
from metrics import BagAggregator
from rich import print
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.utilities.data import dim_zero_cat
from transforms import Transform

ZOO = os.path.join(os.path.dirname(os.path.realpath(__file__)), "zoo")

# Ignore useless warning to sync metrics in DDP
warnings.filterwarnings(
    "ignore",
    message=".*when logging on epoch level in distributed setting to accumulate the metric across devices.",
)


class ExtractionModel(pl.LightningModule):
    def __init__(
        self,
        encoder: Callable,
        label_predict: Optional[str] = None,
        tfms_test: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=(
                "encoder",
                "tfms_test",
                "regions_filter_by_label_func",
            )
        )
        self.encoder = encoder
        self.norm = nn.LayerNorm(self.encoder.num_features)  # Untrained LayerNorm is like (x - x.mean(1)) / x.std(1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        return x

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["img"]
        x = self.tfms_test(x)
        x = self(x)

        self.predict_aggregators[dataloader_idx].update(
            instance_preds=x.new_zeros(len(x)),
            bag_target=batch.get(self.hparams.label_predict),
            bag_indices=batch["slide_idx"],
            instance_indices=batch["region_idx"],
            instance_features=x,
        )

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument("--tfms_test", type=str, default="resize,normalize,gpu")
        return pl.utilities.argparse.add_argparse_args(cls, parser, **kwargs)


def main():

    # ------------
    # args
    # ------------

    # Load config
    parser = ArgumentParser(
        conflict_handler="resolve"
    )  # resolve enables overriding arguments
    parser.add_argument("--cfg", type=str, default=None)

    args, _ = parser.parse_known_args()
    if args.cfg is not None:
        with open(args.cfg) as stream:
            defaults = json.load(stream)

    # Args concerning this script
    args_script = {
        "name": dict(type=str, default="default-name"),
        "encoder_name": dict(type=str, default="resnet18-camelyon_catavgmax"),
        "encoder_ckpt": dict(type=str, default=None),
        "encoder_freeze_ratio": dict(type=float, default=1.0),
        "encoder_pretrained": dict(type=bool, default=True),
        "encoder_zoo": dict(type=str, default=ZOO),
        "seed": dict(type=int, default=42),
        "with_filter": dict(type=str, default="none"),
        "prediction_dir": dict(type=str, default=None),
    }
    for name, kwargs in args_script.items():
        parser.add_argument(f"--{name}", **kwargs)

    # Args from trainer, model, data and transforms
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ExtractionModel.add_argparse_args(parser)
    parser = TileLevelDataModule.add_argparse_args(parser)
    parser = Transform.add_argparse_args(parser)

    # Set defaults to cfg values
    if args.cfg is not None:
        parser.set_defaults(**defaults)

    hp, _ = parser.parse_known_args()

    if hp.image_size is None:
        hp.image_size = hp.regions_size

    hp_dict = vars(hp)
    print(hp_dict)

    # ------------
    # model
    # ------------
    pl.seed_everything(hp.seed)
    encoder, norm_mean, norm_std = get_encoder(**hp_dict)
    hp.norm_mean = norm_mean
    hp.norm_std = norm_std
    model = ExtractionModel(encoder=encoder, **hp_dict)

    # ------------
    # transforms
    # ------------

    tfms_test= Transform(tfms=hp.tfms_test, **hp_dict)
    model.tfms_test = tfms_test.on_gpu

    # ------------
    # data
    # ------------

    if hp.with_filter == "white_blurry_tumor":

        def white_blurry_tumor(labels):
            not_white_or_blurry = labels["white_or_blurry"] < 0.5
            tumor = np.argmax(labels["tile_classifier"], axis=1) == 8
            mask = not_white_or_blurry & tumor
            return mask

        hp.regions_filter_by_label_func = white_blurry_tumor

    elif hp.with_filter == "white_blurry":

        def white_blurry(labels):
            not_white_or_blurry = labels["white_or_blurry"] < 0.5
            return not_white_or_blurry

        hp.regions_filter_by_label_func = white_blurry
      
    elif hp.with_filter == "tumor_only":

        def tumor_only(labels):
            tumor = np.argmax(labels["tile_classifier"], axis=1) == 8
            return tumor

        hp.regions_filter_by_label_func = tumor_only
        

    dm = TileLevelDataModule(**hp_dict)
    dm.tfms_test = tfms_test.on_cpu

    # ------------
    # predicting
    # ------------

    no_target = hp.regions_return_labels_valid is None
    prediction_dir = os.path.join(hp.prediction_dir, hp.name)
    prediction_writer = BagPredictionWriter(prediction_dir, update_to_cpu=True, no_target=no_target)
    trainer = pl.Trainer.from_argparse_args(
        hp,
        logger=False,
        callbacks=[prediction_writer],
    )
    trainer.predict(model, datamodule=dm, ckpt_path=hp.resume_from_checkpoint)


if __name__ == "__main__":
    main()
