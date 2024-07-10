# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224', 'deit_base_patch16_384',
    'deit_my_small_patch8_pure'
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_my_small_patch8_pure(pretrained=False, **kwargs):  # for miniImageNet, imageSize=96
    model = VisionTransformer(
        patch_size=8, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        img_size=96,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def deit_my_small_patch16_MultiLyaerOutput_224(output_blocks, pretrained=False, image_size=96,
                                               **kwargs):  # for miniImageNet, imageSize=96
    model = MultiLyaerOutputVisionTransformer(
        output_blocks,
        patch_size=16, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        img_size=image_size,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def deit_my_small_patch8_MultiLyaerOutput(output_blocks, pretrained=False, image_size=96,
                                          **kwargs):  # for miniImageNet, imageSize=96
    model = MultiLyaerOutputVisionTransformer(
        output_blocks,
        patch_size=8, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        img_size=image_size,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def deit_my_small_patch4_MultiLyaerOutput(output_blocks, pretrained=False, image_size=96,
                                          **kwargs):  # for miniImageNet, imageSize=96
    model = MultiLyaerOutputVisionTransformer(
        output_blocks,
        patch_size=4, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        img_size=image_size,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def deit_small_patch8_MultiLyaerOutput(output_blocks, pretrained=False, image_size=96,
                                       **kwargs):  # for miniImageNet, imageSize=96
    model = MultiLyaerOutputVisionTransformer(
        output_blocks,
        patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        img_size=image_size,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def deit_my_small_patch3_MultiLyaerOutput(output_blocks, pretrained=False, image_size=96,
                                          **kwargs):  # for miniImageNet, imageSize=96
    model = MultiLyaerOutputVisionTransformer(
        output_blocks,
        patch_size=3, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        img_size=image_size,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


class MultiLyaerOutputVisionTransformer(VisionTransformer):
    def __init__(self, output_blocks=None, *args, **kwargs):
        super(MultiLyaerOutputVisionTransformer, self).__init__(*args, **kwargs)
        assert isinstance(output_blocks, list)
        self.output_blocks = output_blocks

    def forward_features(self, x):
        output_features_per_block = {}
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.output_blocks:
                x_out = self.norm(x)
                # x_out = x
                output_features_per_block[f'layer{idx}'] = x_out[:, 0]
        x = self.norm(x)
        if self.dist_token is None:
            return {'pre_logits': self.pre_logits(x[:, 0]),
                    'output_features_per_block': output_features_per_block}
        else:
            assert False
            # return x[:, 0], x[:, 1]

    def forward(self, x, return_all=False):
        contents = self.forward_features(x)
        x = contents['pre_logits']
        output_features_per_block = contents['output_features_per_block']
        if self.head_dist is not None:
            assert False
            # x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            # if self.training and not torch.jit.is_scripting():
            #     # during inference, return the average of both classifier predictions
            #     return x, x_dist
            # else:
            #     return (x + x_dist) / 2
        else:
            x = self.head(x)
        if return_all:
            # return {'features': x, 'output_features_per_block': output_features_per_block}
            return x, output_features_per_block
        else:
            return x


class MyVisionTransformer(VisionTransformer):
    def __init__(self, *args, output_block=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_block = output_block

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx == self.output_block:
                break
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
