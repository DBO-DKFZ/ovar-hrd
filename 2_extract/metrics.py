import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class BagAggregator(Metric):

    is_differentiable: None
    higher_is_better: None
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]
    indices: List[Tensor]

    def __init__(
        self,
        pooling_fn: Union[Callable, str],
        with_features: bool = False,
        no_target: bool = False,
        update_to_cpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.pooling_fn = pooling_fn
        self.update_to_cpu = update_to_cpu
        self.add_state("instance_preds", default=[], dist_reduce_fx="cat")
        if not no_target:
            self.add_state("bag_target", default=[], dist_reduce_fx="cat")
        self.add_state("bag_indices", default=[], dist_reduce_fx="cat")
        if with_features:
            self.add_state("instance_features", default=[], dist_reduce_fx="cat")
            self.add_state("instance_indices", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `BagPool` will save all targets, predictions and indices in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, instance_preds: Tensor, bag_indices: Tensor, bag_target: Optional[Tensor] = None, instance_features: Optional[Tensor] = None, instance_indices: Optional[Tensor] = None) -> None:  # type: ignore
        """Update state with predictions, targets and indices.
        Args:
            instance_preds: Predictions from model (probabilities, or labels)
            bag_indices: Bag index
            bag_target: Ground truth labels
            features: Instance features
            instance_indices: Instance index
        """

        self.instance_preds.append(self.correct_device(instance_preds))
        self.bag_indices.append(self.correct_device(bag_indices))

        if bag_target is not None:
            self.bag_target.append(self.correct_device(bag_target))

        if instance_features is not None:
            self.instance_features.append(self.correct_device(instance_features))
            
        if instance_indices is not None:
            self.instance_indices.append(self.correct_device(instance_indices))

    def compute(
        self,
        return_indices: bool = False,
        return_features: bool = False,
        return_preds: bool = False,
    ) -> Tuple[Tensor]:
        """Compute the bag pooled predictions."""

        instance_level = (
            dim_zero_cat(self.instance_preds),
            dim_zero_cat(self.bag_target) if hasattr(self, "bag_target") else None,
            dim_zero_cat(self.bag_indices),
        )

        *bag_level, bag_instance_preds = self.groupby_reduce(*instance_level, self.pooling_fn)

        if return_indices:
            out = [instance_level, bag_level]
        else:
            out = [instance_level[:2], bag_level[:2]]

        if return_features:
            instance_features = dim_zero_cat(self.instance_features)
            instance_indices = dim_zero_cat(self.instance_indices)
            bag_indices = instance_level[2]
            unique_indices = bag_level[2]
            bag_instance_features = [instance_features[bag_indices == idx] for idx in unique_indices]
            bag_instance_indices = [instance_indices[bag_indices == idx] for idx in unique_indices]
            out.append(bag_instance_features)
            out.append(bag_instance_indices)

        if return_preds:
            out.append(bag_instance_preds)

        return out

    @staticmethod
    def groupby_reduce(
        s: Tensor,
        y: Optional[Tensor],
        ids: Tensor,
        pooling_fn: Callable,
    ) -> Tuple[Tensor]:
        """Group s and y by ids. Pool s with pooling_fn."""
        ids_g, c = ids.unique(return_counts=True)
        idx, y_idx = ids.argsort(), c.new_zeros(len(c))
        y_idx[1:] = c[:-1].cumsum(0)
        s_g = torch.split(s[idx], c.tolist())
        s_g_reduced = torch.stack(list(map(pooling_fn, s_g)))
        y_g = y[idx][y_idx] if y is not None else None
        return s_g_reduced, y_g, ids_g, s_g
    
    def correct_device(self, tensor):
        return tensor.to("cpu") if self.update_to_cpu else tensor
