from argparse import ArgumentParser
from collections import OrderedDict
from typing import List, Optional

import albumentations as A
import cv2
import itertools
import torch
from torch import nn
from torchvision import transforms as T

# Prevent opencv from using all cores in every worker
cv2.setNumThreads(1)


class AlbumentationWrapper(nn.Module):
    """Wrapper around albumentations."""

    def __init__(self, transforms: List):
        super().__init__()
        self.tfms = A.Compose(transforms)

    @torch.no_grad()
    def forward(self, image):
        image = self.tfms(image=image)["image"]
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image

    def __repr__(self):
        return f"{self.__class__.__name__}({self.tfms.__repr__()})"


class Normalize(nn.Module):
    """Normalize image."""

    def __init__(self, mean, std):
        super().__init__()
        self.normalize = T.Normalize(
            torch.tensor(mean),
            torch.tensor(std),
        )

    @torch.no_grad()
    def forward(self, image):
        if not image.is_floating_point():
            image = image / 255
        image = self.normalize(image)
        return image

    def __repr__(self):
        return f"{self.__class__.__name__}"


class BatchWrapper(nn.Module):
    """Wrapper enabling sample-wise batch transforms."""

    def __init__(self, tfms, same_on_batch=False):
        super().__init__()

        self.tfms = tfms
        self.same_on_batch = same_on_batch

    @torch.no_grad()
    def forward(self, x):
        if self.same_on_batch:
            return self.tfms(x)
        else:
            return torch.stack(
                [self.tfms(x_i) for x_i in torch.unbind(x, dim=0)],
                dim=0,
            )

    def __repr__(self):
        return f"{self.__class__.__name__}"
    
    
class TTA_on_GPU(nn.Module):
    """Deterministic TTA."""

    def __init__(self, brightness, contrast, saturation, hue, normalize):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.wants_whole_batch = True
        self.normalize = normalize

        self.params = tuple(itertools.product(brightness, contrast, saturation, hue))
    
    @staticmethod
    def transform(img, brightness, contrast, saturation, hue):
        # Always the same order!
        img = T.functional.adjust_brightness(img, brightness)
        img = T.functional.adjust_contrast(img, contrast)
        img = T.functional.adjust_saturation(img, saturation)
        img = T.functional.adjust_hue(img, hue)
        return img
        
    
    @torch.no_grad()
    def forward(self, batch):
        for key in batch:
            tensor = batch[key]
            if key == "img":
                tensor = torch.cat(
                    [torch.stack([self.transform(img, *params) for params in self.params]) for img in tensor]
                )
                tensor = self.normalize(tensor)
            else:
                tensor = tensor.repeat_interleave(len(self.params))
                
            batch[key] = tensor

        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}"


class Transform(nn.Module):
    """Assemble transforms from str to ops."""

    def __init__(
        self,
        image_size,
        tfms: Optional[str] = "",
        norm_mean: List[float] = [0.5, 0.5, 0.5],
        norm_std: List[float] = [0.5, 0.5, 0.5],
        same_on_batch: bool = False,
        **kwargs,
    ):
        super().__init__()

        size = image_size

        gpu_ops = dict(
            autoaugment=BatchWrapper(T.AutoAugment(), same_on_batch),
            randomresizedcrop=BatchWrapper(
                T.RandomResizedCrop(
                    size, scale=(0.25, 1.0), interpolation=T.InterpolationMode.BILINEAR
                ),
                same_on_batch,
            ),
            normalize=Normalize(norm_mean, norm_std),
            resize=T.Resize([size, size], T.InterpolationMode.BILINEAR),
            colorjitter=BatchWrapper(
                T.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.5),
                same_on_batch,
            ),
            trivial_augment=BatchWrapper(
                T.TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR),
                same_on_batch,
            ),
            tta=TTA_on_GPU(
                brightness=(0.75, 1., 1.25),
                contrast=(0.25, 1., 1.75),
                saturation=(0.75, 1., 1.25),
                hue=(-0.1, 0., 0.1),
                normalize=Normalize(norm_mean, norm_std),
            )
        )

        cpu_ops = dict(
            flip=A.Flip(p=0.5),
            noise=A.GaussNoise(p=0.5),
            shift_scale_rotate=A.ShiftScaleRotate(
                p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=45
            ),
            elastic=A.ElasticTransform(
                p=0.5, alpha=100, sigma=100 * 0.1, alpha_affine=100 * 0.03
            ),
            blur=A.Blur(p=0.5),
            colorjitter=A.ColorJitter(
                p=0.5, brightness=0.25, contrast=0.75, saturation=0.25, hue=0.5
            ),
            brightness_contrast=A.RandomBrightnessContrast(p=0.5),
            clahe=A.CLAHE(p=0.5),
            equalize=A.Equalize(p=0.5),
            solarize=A.Solarize(p=0.5),
            sharpen=A.Sharpen(p=0.5),
            posterize=A.Posterize(p=0.5),
            gamma=A.RandomGamma(p=0.5),
            cutout=A.CoarseDropout(p=0.5),
        )

        cpu_ops["trivial_augment"] = A.OneOf(list(cpu_ops.values()), p=1.0)
        cpu_ops["randomresizedcrop"] = A.RandomResizedCrop(
            size, size, scale=(0.2, 1.0), interpolation=cv2.INTER_LINEAR
        )
        cpu_ops["togray"] = A.ToGray(p=1.0)
        cpu_ops["resize"] = A.Resize(size, size, interpolation=cv2.INTER_LINEAR)
        cpu_ops["normalize"] = A.Normalize(
            mean=norm_mean, std=norm_std, max_pixel_value=255.0
        )

        on_cpu = []
        on_gpu = []

        if len(tfms) > 0:
            tfms = tfms.replace(" ", "")
            on_cpu, on_gpu = [t.strip(",").split(",") for t in tfms.split("gpu")]
            on_cpu = [] if (len(on_cpu) == 1) and (on_cpu[0] == "") else on_cpu
            on_gpu = [] if (len(on_gpu) == 1) and (on_gpu[0] == "") else on_gpu

        self.on_cpu = (
            AlbumentationWrapper([cpu_ops[t] for t in on_cpu])
            if len(on_cpu) > 0
            else nn.Identity()
        )
        self.on_gpu = nn.Sequential(OrderedDict([[t, gpu_ops[t]] for t in on_gpu]))

    @torch.no_grad()
    def forward(self, x):
        x = self.on_cpu(x)
        x = self.on_gpu(x)
        return x

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument("--image_size", type=int, default=None)
        parser.add_argument(
            "--norm_mean", nargs="+", type=float, default=[0.5, 0.5, 0.5]
        )
        parser.add_argument(
            "--norm_std", nargs="+", type=float, default=[0.5, 0.5, 0.5]
        )
        return parser
