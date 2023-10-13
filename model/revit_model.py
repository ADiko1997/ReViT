"""ReViT and ReMViTv2  models. Modified from mvitv2 official repo."""

import math
from functools import partial
# from turtle import forward
from numpy import isin
# from sqlalchemy import true

import torch
import torch.nn as nn
from .attention import MultiScaleBlock
from .common import round_width
from timm.models.layers import trunc_normal_


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(7, 7),
        stride=(4, 4),
        padding=(3, 3),
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B HW C
        return x.flatten(2).transpose(1, 2), x.shape


class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x


# @MODEL_REGISTRY.register()
class ReViT(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2112.01526

    Multiscale Vision Transformers
    Haoqi Fan*, Bo Xiong*, Karttikeya Mangalam*, Yanghao Li*, Zhicheng Yan, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Prepare input.
        in_chans = 3
        spatial_size = cfg.DATA.crop_size
        # Prepare output.
        num_classes = cfg.MODEL.num_classes
        embed_dim = cfg.ReViT.embed_dim
        # MViT params.
        num_heads = cfg.ReViT.num_heads
        depth = cfg.ReViT.depth
        self.cls_embed_on = cfg.ReViT.cls_embed_on
        self.use_abs_pos = cfg.ReViT.use_abs_pos
        self.zero_decay_pos_cls = cfg.ReViT.zero_decay_pos
        if cfg.ReViT.alpha:
            self.alpha = nn.Parameter(torch.Tensor([0.5]))
        else:
            self.alpha = 0.5


        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # if cfg.MODEL.ACT_CHECKPOINT:
        #     validate_checkpoint_wrapper_import(checkpoint_wrapper)

        patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.ReViT.patch_kernel,
            stride=cfg.ReViT.patch_stride,
            padding=cfg.ReViT.patch_padding,
        )
        # if cfg.MODEL.ACT_CHECKPOINT:
        #     patch_embed = checkpoint_wrapper(patch_embed)
        self.patch_embed = patch_embed

        patch_dims = [
            spatial_size // cfg.ReViT.patch_stride[0],
            spatial_size // cfg.ReViT.patch_stride[1],
        ]
        num_patches = math.prod(patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, cfg.ReViT.drop_path, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        # MViT backbone configs
        dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv = _prepare_mvit_configs(
            cfg
        )

        input_size = patch_dims
        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if cfg.ReViT.dim_mul_in_att:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=cfg.ReViT.mlp_ratio,
                qkv_bias=cfg.ReViT.qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=cfg.ReViT.mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=cfg.ReViT.pool_first,
                rel_pos_spatial=cfg.ReViT.use_rel_pos,
                rel_pos_zero_init=cfg.ReViT.rel_pos_zero_init,
                residual_pooling=cfg.ReViT.residual_pooling,
                dim_mul_in_att=cfg.ReViT.dim_mul_in_att,
            )

            # if cfg.MODEL.ACT_CHECKPOINT:
            #     attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]
            embed_dim = dim_out

        self.norm = norm_layer(embed_dim)

        self.head = TransformerBasicHead(
            embed_dim,
            num_classes,
            dropout_rate=cfg.MODEL.dropout_rate,
            act_func=cfg.MODEL.head_act,
        )
        if self.use_abs_pos:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        #Flag to return the attention maps
        self.viz = cfg.ReViT.visualize

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            # add all potential params
            names = ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"]

        return names

    def forward(self, x):
        attn_res = None
        attn_container = []
        x, bchw = self.patch_embed(x)

        H, W = bchw[-2], bchw[-1]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            x = x + self.pos_embed

        thw = [H, W]
        for blk in self.blocks:
            x, thw, attn_res = blk(x, thw, attn_res, self.alpha)
            attn_container.append(attn_res)

        x = self.norm(x)
        if self.cls_embed_on:
            x = x[:, 0]
        else:
            x = x.mean(1)

        x = self.head(x)
        if self.viz:
            # print("Returning output dict")
            return {
                "output": x,
                "attn_maps": attn_container
            }
        return x


def _prepare_mvit_configs(cfg):
    """
    Prepare mvit configs for dim_mul and head_mul facotrs, and q and kv pooling
    kernels and strides.
    Credits to Meat (C)
    """
    depth = cfg.ReViT.depth
    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    for i in range(len(cfg.ReViT.dim_mul)):
        dim_mul[cfg.ReViT.dim_mul[i][0]] = cfg.ReViT.dim_mul[i][1]
    for i in range(len(cfg.ReViT.head_mul)):
        head_mul[cfg.ReViT.head_mul[i][0]] = cfg.ReViT.head_mul[i][1]

    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]

    # print("Len q stride: ", len(cfg.ReViT.pool_q_stride))
    for i in range(len(cfg.ReViT.pool_q_stride)):
        stride_q[cfg.ReViT.pool_q_stride[i][0]] = cfg.ReViT.pool_q_stride[i][1:]
        pool_q[cfg.ReViT.pool_q_stride[i][0]] = cfg.ReViT.pool_qkv_kernel
    

    # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
    if cfg.ReViT.pool_kv_stride_adaptive is not None:
        _stride_kv = cfg.ReViT.pool_kv_stride_adaptive
        cfg.ReViT.pool_kv_stride = []
        for i in range(cfg.ReViT.depth):
            if len(stride_q[i]) > 0:
                _stride_kv = [
                    max(_stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(_stride_kv))
                ]
            cfg.ReViT.pool_kv_stride.append([i] + _stride_kv)
    for i in range(len(cfg.ReViT.pool_kv_stride)):
        stride_kv[cfg.ReViT.pool_kv_stride[i][0]] = cfg.ReViT.pool_kv_stride[i][1:]
        pool_kv[cfg.ReViT.pool_kv_stride[i][0]] = cfg.ReViT.pool_qkv_kernel

    return dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv