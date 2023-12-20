import os
from argparse import Namespace
from collections import OrderedDict
from typing import Optional

import timm
import torch
from rich import print


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling."""

    def __init__(self, pooling="global", **kwargs):
        super().__init__(**kwargs)
        assert pooling in ("none", "global", "cls_token")
        self.pooling = pooling

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.pooling == "global":
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.norm(x)
        elif self.pooling == "cls_token":
            x = self.norm(x)
            outcome = x[:, 0]
        else:
            outcome = x

        return outcome

    def freeze(self):
        self.requires_grad_(False)
        if self.pooling == "cls_token":
            self.cls_token.requires_grad_(True)

    def unfreeze(self):
        self.requires_grad_(True)


def get_encoder(
    encoder_name: str,
    encoder_ckpt: Optional[str] = None,
    encoder_freeze_ratio: float = 0.0,
    encoder_zoo: Optional[str] = None,
    encoder_pretrained: bool = True,
    **kwargs,
):
    model, pool, *misc = encoder_name.split("_")

    supported = timm.list_models() + ["MAE", "resnet18-camelyon", "resnet50-wang"]

    # assert encoder_name in supported

    kwargs = {
        key[len("encoder") :]: kwargs[key]
        for key in kwargs
        if key.startswith("encoder")
    }

    print(f"Using {encoder_name} (pretrained={encoder_pretrained})")

    if model == "MAE":
        encoder = torch.load(encoder_ckpt)
        encoder.pooling = "cls_token"
        if encoder_freeze_ratio == 1:
            encoder.freeze()
        else:
            encoder.unfreeze()
        encoder_freeze_ratio = None
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

    elif model == "resnet18-camelyon":
        encoder = timm.create_model("resnet18")
        encoder.fc = torch.nn.Conv2d(512, 1, 1)
        ckpt = torch.load(
            os.path.join(encoder_zoo, "nvidia-resnet18.pt"), map_location="cpu"
        )
        encoder.load_state_dict(
            OrderedDict(zip(encoder.state_dict().keys(), ckpt.values()))
        )
        encoder.reset_classifier(0, global_pool=pool)
        # Timm does not update this automatically for some reason...
        encoder.num_features = encoder.num_features * encoder.global_pool.feat_mult()
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        
    elif model == "resnet18-ciga":
        encoder = timm.create_model("resnet18", num_classes=0, global_pool=pool)
        ckpt = torch.load(
            os.path.join(encoder_zoo, "ciga-resnet18.pt"), map_location="cpu"
        )['state_dict']
        for key in list(ckpt.keys()):
            new_key = key.replace("model.resnet.", "")
            value = ckpt.pop(key)
            if not new_key.startswith("fc."):
                ckpt[new_key] = value

        encoder.load_state_dict(ckpt)
        # Timm does not update this automatically for some reason...
        encoder.num_features = encoder.num_features * encoder.global_pool.feat_mult()
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        
    elif model == "resnet50-wang":
        encoder = timm.create_model("resnet50", num_classes=0, global_pool=pool)
        ckpt = torch.load(
            os.path.join(encoder_zoo, "wang-resnet50.pt"), map_location="cpu"
        )
        encoder.load_state_dict(ckpt)
        # Timm does not update this automatically for some reason...
        encoder.num_features = encoder.num_features * encoder.global_pool.feat_mult()
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

    elif "truncated" in misc:
        if model.find("resnet") == -1:
            raise ValueError("Truncation only supported for ResNets!")
        encoder = timm.create_model(
            model,
            global_pool=pool,
            pretrained=encoder_pretrained,
            num_classes=0,
            **kwargs,
        )
        encoder.layer4 = torch.nn.Identity()
        encoder.num_features = (
            encoder.layer3[-1].bn3.num_features * encoder.global_pool.feat_mult()
        )
        cfg = encoder.default_cfg
        norm_mean = cfg["mean"]
        norm_std = cfg["std"]

    else:
        encoder = timm.create_model(
            model,
            global_pool=pool,
            pretrained=encoder_pretrained,
            num_classes=0,
            **kwargs,
        )
        if pool == "catavgmax":
            encoder.num_features = encoder.num_features * encoder.global_pool.feat_mult()
        cfg = encoder.default_cfg
        norm_mean = cfg["mean"]
        norm_std = cfg["std"]

    if encoder_freeze_ratio is None:
        pass
    elif encoder_freeze_ratio == 0:
        encoder.requires_grad_(True)
    elif encoder_freeze_ratio == 1:
        encoder.requires_grad_(False)
    elif 0 < encoder_freeze_ratio < 1:
        if model.find("convnext") != -1:
            blocks = [encoder.stem]
            for stage in encoder.stages:
                blocks += [stage.downsample] + [b for b in stage.blocks]
            blocks += encoder.head

        elif model.find("resnet") != -1:
            downsample = [encoder.conv1, encoder.bn1, encoder.act1, encoder.maxpool]
            blocks = [downsample]
            for layer in [
                encoder.layer1,
                encoder.layer2,
                encoder.layer3,
                encoder.layer4,
            ]:
                has_children = len(list(layer.children())) > 0
                if has_children:
                    for block in layer:
                        has_parameters = sum(p.numel() for p in block.parameters()) > 0
                        if has_parameters:
                            blocks.append(block)
                else:
                    has_parameters = sum(p.numel() for p in layer.parameters()) > 0
                    if has_parameters:
                        blocks.append(layer)
        else:
            raise ValueError(f"Freeze ratios unsupported for {model}")

        num_unfreeze_blocks = int(len(blocks) * (1 - encoder_freeze_ratio))
        unfreeze_blocks = blocks[::-1][:num_unfreeze_blocks]

        print(f"Unfreezing {num_unfreeze_blocks}/{len(blocks)} blocks")

        encoder.requires_grad_(False)
        for block in unfreeze_blocks:
            if isinstance(block, list):
                for b in block:
                    b.requires_grad_(True)
            else:
                block.requires_grad_(True)

    print(f"Encoder has {encoder.num_features} features")
    print(f"Encoder uses {encoder.global_pool}")

    return encoder, norm_mean, norm_std
