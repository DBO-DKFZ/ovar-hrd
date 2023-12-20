import os
from argparse import ArgumentParser

from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.nn import Identity
from torch.utils.data import DataLoader

from slide_tools.objects.constants import (BalanceMode, LabelInterpolation,
                                           SizeUnit)
from slide_tools.tile_level import TileLevelDataset


class TileLevelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        csv_train: Optional[str] = None,
        csv_valid: Optional[Union[Sequence[str], str]] = None,
        csv_test: Optional[str] = None,
        csv_predict: Optional[Union[Sequence[str], str]] = None,
        tfms_train: Callable = Identity(),
        tfms_valid: Callable = Identity(),
        tfms_test: Callable = Identity(),
        column_slide: str = "slide",
        column_annotation: Optional[str] = None,
        column_label: Optional[str] = None,
        columns_global_label_train: Optional[Sequence[str]] = None,
        columns_global_label_valid: Optional[Sequence[str]] = None,
        slide_simplify_tolerance: int = 0,
        slide_interpolation: Union[str, LabelInterpolation] = "nearest",
        slide_load_keys: Optional[Sequence[str]] = None,
        slide_linear_fill_value: float = np.nan,
        regions_size: Optional[int] = None,
        regions_unit: Union[str, SizeUnit] = "pixel",
        regions_level: int = 0,
        regions_centroid_in_annotation: bool = False,
        regions_annotation_align: bool = False,
        regions_region_overlap: float = 0.0,
        regions_with_labels: bool = False,
        regions_return_index: bool = False,
        regions_return_labels_train: Optional[Union[Sequence[str], str]] = None,
        regions_return_labels_valid: Optional[Union[Sequence[str], str]] = None,
        regions_filter_by_label_func: Optional[Callable] = None,
        epoch_balance_size_by: Optional[Union[BalanceMode, int, str]] = None,
        epoch_balance_label_key: Optional[str] = None,
        epoch_balance_label_bins: int = 10,
        epoch_shuffle: bool = False,
        epoch_shuffle_chunk_size: int = 1,
        epoch_with_replacement: bool = True,
        epoch_strict_size_balance: bool = False,
        verbose: bool = False,
        pin_memory: bool = False,
        num_valid_samples_per_slide: Optional[int] = None,
        train_location_wiggle: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "tfms_train",
                "tfms_valid",
                "tfms_test",
                "regions_filter_by_label_func",
            ]
        )

        self.tfms_train = tfms_train
        self.tfms_valid = tfms_valid
        self.tfms_test = tfms_test
        self.regions_filter_by_label_func = regions_filter_by_label_func

        self.correct_hparams()

    def correct_hparams(self):
        # Helper to split off hparams beginning with prefix
        def split_off_by(prefix, hparams):
            return pl.utilities.parsing.AttributeDict(
                {
                    key[len(prefix) :]: hparams[key]
                    for key in hparams
                    if key.startswith(prefix)
                }
            )

        self.hparams.root = os.path.abspath(self.hparams.root)
        self.slide_kwargs = split_off_by("slide_", self.hparams)
        self.region_kwargs = split_off_by("regions_", self.hparams)
        self.region_kwargs["filter_by_label_func"] = self.regions_filter_by_label_func
        self.epoch_kwargs = split_off_by("epoch_", self.hparams)

        if isinstance(self.slide_kwargs.interpolation, str):
            self.slide_kwargs.interpolation = LabelInterpolation[
                self.slide_kwargs.interpolation.upper()
            ]
        if isinstance(self.region_kwargs.unit, str):
            self.region_kwargs.unit = SizeUnit[self.region_kwargs.unit.upper()]
        if isinstance(self.epoch_kwargs.balance_size_by, str):
            self.epoch_kwargs.balance_size_by = BalanceMode[
                self.epoch_kwargs.balance_size_by.upper()
            ]
        if isinstance(self.hparams.columns_global_label_train, str):
            self.hparams.columns_global_label_train = [self.hparams.columns_global_label_train]
        if isinstance(self.hparams.columns_global_label_valid, str):
            self.hparams.columns_global_label_valid = [self.hparams.columns_global_label_valid]
        if isinstance(self.hparams.csv_valid, str):
            self.hparams.csv_valid = [self.hparams.csv_valid]
        if isinstance(self.hparams.csv_predict, str):
            self.hparams.csv_valid = [self.hparams.csv_predict]

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        # Helper to join root onto relative path series
        rootify_ = lambda path: os.path.join(self.hparams.root, path)

        def rootify(series):
            if series is None:
                return None
            else:
                return series.apply(rootify_)

        # Helper to turn label frame (mulitple columns) to dict
        def get_columns_as_records(frame, columns):
            if columns is None:
                return None
            else:
                return frame[columns].to_dict("records")

        if stage == "fit" or stage is None:
            frame_train = pd.read_csv(rootify_(self.hparams.csv_train))
            frames_valid = [pd.read_csv(rootify_(p)) for p in self.hparams.csv_valid]

            self.ds_train = TileLevelDataset(
                slide_paths=rootify(frame_train[self.hparams.column_slide]),
                annotation_paths=rootify(
                    frame_train.get(self.hparams.column_annotation)
                ),
                label_paths=rootify(frame_train.get(self.hparams.column_label)),
                global_labels=get_columns_as_records(
                    frame_train, self.hparams.columns_global_label_train
                ),
                img_tfms=self.tfms_train,
                return_labels=self.region_kwargs.return_labels_train,
                location_wiggle=self.hparams.train_location_wiggle,
                **self.slide_kwargs,
                **self.region_kwargs,
                verbose=self.hparams.verbose,
            )

            self.dss_valid = [
                TileLevelDataset(
                    slide_paths=rootify(frm[self.hparams.column_slide]),
                    annotation_paths=rootify(frm.get(self.hparams.column_annotation)),
                    label_paths=rootify(frm.get(self.hparams.column_label)),
                    global_labels=get_columns_as_records(
                        frm, self.hparams.columns_global_label_valid
                    ),
                    img_tfms=self.tfms_valid,
                    return_labels=self.region_kwargs.return_labels_valid,
                    **self.slide_kwargs,
                    **self.region_kwargs,
                    verbose=self.hparams.verbose,
                )
                for frm in frames_valid
            ]

            if self.hparams.num_valid_samples_per_slide is not None:
                for ds in self.dss_valid:
                    ds.setup_epoch(
                        balance_size_by=self.hparams.num_valid_samples_per_slide,
                        with_replacement=False,
                        strict_size_balance=True,
                    )

        if stage == "test" or stage is None:
            frame_test = pd.read_csv(rootify_(self.hparams.csv_test))

            self.ds_test = TileLevelDataset(
                slide_paths=rootify(frame_test[self.hparams.column_slide]),
                annotation_paths=rootify(
                    frame_test.get(self.hparams.column_annotation)
                ),
                label_paths=rootify(frame_test.get(self.hparams.column_label)),
                global_labels=get_columns_as_records(
                    frame_test, self.hparams.columns_global_label_valid
                ),
                img_tfms=self.tfms_test,
                return_labels=self.region_kwargs.return_labels_valid,
                **self.slide_kwargs,
                **self.region_kwargs,
                verbose=self.hparams.verbose,
            )

        if stage == "predict" or stage is None:
            if isinstance(self.hparams.csv_predict, str):
                self.hparams.csv_predict = [self.hparams.csv_predict]
            frames_predict = [
                pd.read_csv(rootify_(p)) for p in self.hparams.csv_predict
            ]

            self.dss_predict = [
                TileLevelDataset(
                    slide_paths=rootify(frm[self.hparams.column_slide]),
                    annotation_paths=rootify(frm.get(self.hparams.column_annotation)),
                    label_paths=rootify(frm.get(self.hparams.column_label)),
                    global_labels=get_columns_as_records(
                        frm, self.hparams.columns_global_label_valid
                    ),
                    img_tfms=self.tfms_test,
                    return_labels=self.region_kwargs.return_labels_valid,
                    **self.slide_kwargs,
                    **self.region_kwargs,
                    verbose=self.hparams.verbose,
                )
                for frm in frames_predict
            ]

    def train_dataloader(
        self,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ):
        self.ds_train.setup_epoch(**self.epoch_kwargs)
        return DataLoader(
            self.ds_train,
            shuffle=False,
            batch_size=batch_size or self.hparams.batch_size,
            drop_last=True,
            num_workers=num_workers or self.hparams.num_workers,
            pin_memory=pin_memory or self.hparams.pin_memory,
        )

    def val_dataloader(
        self,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ):
        return [
            DataLoader(
                ds,
                shuffle=False,
                batch_size=batch_size or self.hparams.batch_size,
                num_workers=num_workers or self.hparams.num_workers,
                pin_memory=pin_memory or self.hparams.pin_memory,
            )
            for ds in self.dss_valid
        ]

    def test_dataloader(
        self,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ):
        return DataLoader(
            self.ds_test,
            shuffle=False,
            batch_size=batch_size or self.hparams.batch_size,
            num_workers=num_workers or self.hparams.num_workers,
            pin_memory=pin_memory or self.hparams.pin_memory,
        )

    def predict_dataloader(
        self,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ):
        return [
            DataLoader(
                ds,
                shuffle=False,
                batch_size=batch_size or self.hparams.batch_size,
                num_workers=num_workers or self.hparams.num_workers,
                pin_memory=pin_memory or self.hparams.pin_memory,
            )
            for ds in self.dss_predict
        ]

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument("--csv_valid", nargs="+", type=str, default=None)
        parser.add_argument("--csv_predict", nargs="+", type=str, default=None)
        parser.add_argument("--columns_global_label_train", nargs="+", type=str, default=None)
        parser.add_argument("--columns_global_label_valid", nargs="+", type=str, default=None)
        parser.add_argument("--slide_load_keys", nargs="+", type=str, default=None)
        parser.add_argument(
            "--regions_return_labels_train", nargs="+", type=str, default=None
        )
        parser.add_argument(
            "--regions_return_labels_valid", nargs="+", type=str, default=None
        )
        return pl.utilities.argparse.add_argparse_args(cls, parser, **kwargs)
