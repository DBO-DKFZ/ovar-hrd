import os
from typing import Any, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from metrics import BagAggregator
from pytorch_lightning.callbacks.callback import Callback
from rich import print


class BagPredictionWriter(Callback):
    def __init__(
        self, save_directory: str, no_target: bool = False, update_to_cpu: bool = False
    ):
        self.save_directory = save_directory
        self.no_target = no_target
        self.update_to_cpu = update_to_cpu

    def setup(self, trainer, pl_module, stage=None):
        if stage != "predict":
            raise ValueError(f"You shouldn't use this callback with {stage=}")

        print("Added prediction aggregators to pl_module")
        pl_module.predict_aggregators = [
            BagAggregator(
                pooling_fn=torch.mean,
                with_features=True,
                no_target=self.no_target,
                update_to_cpu=self.update_to_cpu,
            )
            for _ in range(len(trainer.datamodule.hparams.csv_predict))
        ]

    def on_predict_epoch_end(self, trainer, pl_module, outputs):

        datasets = trainer.datamodule.dss_predict
        aggregators = pl_module.predict_aggregators
        dataset_names = [
            os.path.basename(csv).split(".")[0]
            for csv in trainer.datamodule.hparams.csv_predict
        ]

        # Catch non-unique names
        if len(dataset_names) != len(np.unique(dataset_names)):
            dataset_names = [f"dataset_{i}" for i in range(len(dataset_names))]

        for aggregator, dataset, dataset_name in zip(
            aggregators, datasets, dataset_names
        ):

            (
                instance_level,
                bag_level,
                bag_instance_features,
                bag_instance_indices,
                bag_instance_preds,
            ) = aggregator.compute(
                return_indices=True,
                return_features=True,
                return_preds=True,
            )

            bag_pred, bag_target, bag_index = bag_level

            # {instance, bag}_level = (preds, target, index)
            # bag_instance_features = [bag_0_features, ...]
            # bag_instance_preds = [bag_0_preds, ...]
            # bag_instance_indices = [bag_0_indices, ...]

            slide_names = [
                os.path.basename(slide.image.path).split(".")[0]
                for slide in dataset.slides
            ]

            # Catch non-unique names
            if len(slide_names) != len(np.unique(slide_names)):
                slide_names = [f"slide_{i}" for i in range(len(slide_names))]

            save_dir = os.path.join(self.save_directory, dataset_name)

            print(f"Saving features to {save_dir}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            if bag_target is None:
                bag_target = [None for _ in range(len(slide_names))]

            for i in range(len(slide_names)):
                out = {
                    "bag_idx": bag_index[i],
                    "bag_pred": bag_pred[i],
                    "bag_target": bag_target[i],
                    "instance_preds": bag_instance_preds[i],
                    "instance_features": bag_instance_features[i],
                    "instance_indices": bag_instance_indices[i],
                }
                path = os.path.join(save_dir, f"{slide_names[i]}.pt")
                torch.save(out, path)
