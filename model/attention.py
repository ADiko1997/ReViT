from multiprocessing import pool
# from turtle import forward
import numpy as np
import numpy 
import torch 
import torch.nn as nn 
from timm.models.layers import DropPath, trunc_normal_

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def downsample(attn_res:torch.tensor, q_W:torch.tensor, q_H:torch.tensor):
    """
    Applies pooling on attention residuals when HW dimensons are changed

    Parameters
    ==========
        attn_res (torch.tensor): attention residuals from previous block
        q_N (int): number of patchs (i.e. attention map has shape [B, num_heads, q_N, q_N])
    
    Return
    ======
        attn_res (torch.tensor): pooled attention residuals

    Raises
    ======
        ValueError: Value error if q_N is <=0
    """
    # print('enters')
    B, n_heads, num_patches_W_2, num_patches_H_2 = attn_res.shape #(B, n_heads, num_patches_W_2, num_patches_H_2)
    num_patches_W = int(num_patches_W_2**(1/2))
    num_patches_H = int(num_patches_H_2**(1/2))
    q_W_prev = torch.as_tensor(num_patches_W_2)
    q_H_prev = torch.as_tensor(num_patches_H_2)

    # q_N = torch.as_tensor(q_N)
    if q_W_prev > q_W:

        kernel = torch.div(torch.sqrt(q_W_prev), torch.sqrt(torch.as_tensor(q_W)), rounding_mode='floor')
        stride = kernel
        padding = kernel - stride # will be 0 with the current config
        pool = nn.AvgPool2d(
            kernel_size=int(kernel),
            stride=int(stride),
            padding=int(padding),
            ceil_mode=False
        )

        # dim_2 pooling
        attn_res = attn_res.transpose(-1, -2).reshape(B * n_heads, num_patches_H_2, num_patches_W, num_patches_W) #(B * n_heads, num_patches_H_2, num_patches_W, num_patches_W)
        attn_res = pool(attn_res)
        attn_res = attn_res.reshape(B * n_heads, num_patches_H_2, q_W).transpose(-1, -2) #(B * n_heads, q_W, num_patches_H_2) 

    # dim_3
    if q_H_prev > q_H:
        kernel = torch.div(torch.sqrt(q_H_prev), torch.sqrt(torch.as_tensor(q_H)), rounding_mode='floor')
        stride = kernel
        padding = kernel - stride # will be 0 with the current config
        pool = nn.AvgPool2d(
            kernel_size=int(kernel),
            stride=int(stride),
            padding=int(padding),
            ceil_mode=False
        )

        attn_res = attn_res.reshape(B * n_heads, q_W, num_patches_H, num_patches_H) #(B * n_heads, q_W, num_patches_W, num_patches_H)
        attn_res = pool(attn_res)
        # attn_res = attn_res.reshape(B * n_heads, q_W, q_H) #(B * n_heads, q_W, q_H) 

    elif q_H > q_H_prev:
        upsampling_factor = torch.div(torch.sqrt(torch.as_tensor(q_H)), torch.sqrt(q_H_prev), rounding_mode='floor')
        upsample = torch.nn.Upsample(scale_factor=upsampling_factor, mode='nearest')
        attn_res = attn_res.reshape(B * n_heads, q_W, num_patches_H, num_patches_H) #(B * n_heads, q_W, num_patches, num_patches)
        attn_res = upsample(attn_res)
        # attn_res = attn_res.reshape(B, n_heads, q_W, q_H)

    attn_res = attn_res.reshape(B, n_heads, q_W, q_H)

    return attn_res



def attention_pool(tensor, pool, hw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, hw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    H, W = hw_shape
    tensor = tensor.reshape(B * N, H, W, C).permute(0, 3, 1, 2).contiguous()

    tensor = pool(tensor)

    hw_shape = [tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)

    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, hw_shape


def cal_rel_pos_spatial(
    attn,
    q,
    has_cls_embed,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.
    """
    # print('enters rel possitional')
    sp_idx = 1 if has_cls_embed else 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        mode="conv",
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_zero_init=False,
        residual_pooling=True,
    ):
        super().__init__()
        self.pool_first = pool_first

        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if pool_first:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()
        self.mode = mode

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            self.pool_q = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None

            self.pool_k = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None

            self.pool_v = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )

            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None

        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        # relative pos embedding
        self.rel_pos_spatial = rel_pos_spatial
        if self.rel_pos_spatial:
            assert input_size[0] == input_size[1]

            size = input_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

            if not rel_pos_zero_init:
                torch.nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                torch.nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, hw_shape, attn_res=None, alpha=0.5):
        B, N, _ = x.shape

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"

            qkv = (
                self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        if self.pool_first:
            q_N = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
            k_N = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
            v_N = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = self.v(v).reshape(B, v_N, self.num_heads, -1).permute(0, 2, 1, 3)

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = self.k(k).reshape(B, k_N, self.num_heads, -1).permute(0, 2, 1, 3)

        N = q.shape[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)


        ########################
        ## Residual Attention ##
        ########################
        if attn_res is not None:
            if attn_res.shape[1] != attn.shape[1]:
                attn_res = attn_res.mean(1, keepdim=True)
            
            if attn_res.shape[-1] != attn.shape[-1] or attn_res.shape[-2] != attn.shape[-2]:
                attn_res = downsample(attn_res=attn_res, q_W=attn.shape[-2], q_H=attn.shape[-1])    


            attn = (alpha * attn) + ((1-alpha) * attn_res)       
        attn_res = attn


        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
            )

        attn = attn.softmax(dim=-1)
        x = attn @ v
        # print("x:", x.shape)

        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q
        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        # attn_res = attn_res.transpose(1, 2).reshape(B, -1, self.dim_out)

        x = self.proj(x)
        # print(x.shape)
        return x, q_shape, attn_res


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        input_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_zero_init=False,
        residual_pooling=True,
        dim_mul_in_att=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att

        att_dim = dim_out if dim_mul_in_att else dim
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed

        mlp_dim_out = dim_out
        self.mlp = MLP(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if len(stride_q) > 0 and numpy.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.pool_skip = nn.MaxPool2d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
        else:
            self.pool_skip = None

    def forward(self, x, hw_shape, attn_res=None, alpha=0.5):
        x_norm = self.norm1(x)
        x_block, hw_shape_new, attn_res = self.attn(x_norm, hw_shape, attn_res, alpha)

        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = attention_pool(
            x, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed
        )
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)

        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, hw_shape_new, attn_res
