import torch
from torch import nn

import sys
import numpy as np

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

###############################
##      Complete Blocks      ##
###############################

class AsymmetricMHPAReversibleBlock(nn.Module):
    """
    Reversible Blocks for Reversible Vision Transformer.
    See Section 3.3.2 in paper for details.
    """

    def __init__(self, dim_c, dim_v, num_heads, enable_amp, drop_path, token_map_pool_size, kv_pool_size, const_patches_shape, block_id):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.

        self.drop_path_rate = drop_path

        self.F = MHPAModifiedFBlockC2V(
            dim_c=dim_c,
            dim_v=dim_v,
            num_heads=num_heads,
            patches_shape=const_patches_shape,
            token_pool_size=token_map_pool_size,
            kv_pool_size=kv_pool_size,
            enable_amp=enable_amp,
        )

        self.G = FFNConvTransposeSubblockV2C(
            dim_c=dim_c,
            dim_v=dim_v,
            const_patches_shape=const_patches_shape,
            token_pool_size=token_map_pool_size,
            enable_amp=False,  # standard for ViTs
        )

        # note that since all functions are deterministic, and we are
        # not using any stochastic elements such as dropout, we do
        # not need to control seeds for the random number generator.
        # To see usage with controlled seeds and dropout, see pyslowfast.

        self.seeds = {}
        self.block_id = block_id
        print(f"Block index : {self.block_id} | {num_heads} heads | Dpr : {self.drop_path_rate}")
    
    def seed_cuda(self, key):
        """
        Fix seeds to allow for stochastic elements such as
        dropout to be reproduced exactly in activation
        recomputation in the backward pass.
        """

        # randomize seeds
        # use cuda generator if available
        if (
            hasattr(torch.cuda, "default_generators")
            and len(torch.cuda.default_generators) > 0
        ):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        self.seed_cuda("attn")
        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        self.seed_cuda("droppath")
        f_X_2_dropped = drop_path(
            f_X_2, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_1 = X_1 + f(X_2)
        Y_1 = X_1 + f_X_2_dropped

        # free memory since X_1 is now not needed
        del X_1

        self.seed_cuda("FFN")
        g_Y_1 = self.G(Y_1)

        torch.manual_seed(self.seeds["droppath"])
        g_Y_1_dropped = drop_path(
            g_Y_1, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1_dropped

        # free memory since X_2 is now not needed
        del X_2

        return Y_1, Y_2

    def backward_pass(
        self,
        Y_1,
        Y_2,
        dY_1,
        dY_2,
    ):
        """
        equation for activation recomputation:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention

        And we use pytorch native logic carefully to
        calculate gradients on F and G.
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            torch.manual_seed(self.seeds["FFN"])
            g_Y_1 = self.G(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = drop_path(
                g_Y_1, drop_prob=self.drop_path_rate, training=self.training
            )

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence we do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dY_1 = dY_1 + Y_1.grad

            # free memory since Y_1.grad is now not needed
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.F(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = drop_path(
                f_X_2, drop_prob=self.drop_path_rate, training=self.training
            )

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():
            # recomputing X_1 from the rev equation
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            # the gradients for the previous block
            # note that it is called dY_2 but it in fact dX_2 in math.
            # reusing same variable to save memory
            dY_2 = dY_2 + X_2.grad

            # free memory since X_2.grad is now not needed
            X_2.grad = None

            X_2 = X_2.detach()

        # et voila~
        return X_1, X_2, dY_1, dY_2


class AsymmetricSwinReversibleBlock(nn.Module):
    """
    Reversible Blocks for Reversible Vision Transformer.
    See Section 3.3.2 in paper for details.
    """

    def __init__(self, f_block, dim_c, dim_v, num_heads, window_size, shift_size, enable_amp, drop_path, token_map_pool_size, const_patches_shape, block_id):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.

        self.drop_path_rate = drop_path

        SwinBlock=None
        if f_block == "swin-attention":
            SwinBlock = SwinFBlockC2V
        elif f_block == "swin-mlp":
            SwinBlock = SwinMLPFBlockC2V
        elif f_block == "swin-dw-mlp":
            SwinBlock = SwinDWMLPFBlockC2V
        else:
            print("Swin block selection error")

        self.F = SwinBlock(
            dim_c=dim_c,
            dim_v=dim_v,
            num_heads=num_heads,
            window_size=window_size, 
            shift_size=shift_size,
            patches_shape=const_patches_shape,
            token_pool_size=token_map_pool_size,
            enable_amp=enable_amp,
        )

        self.G = FFNConvTransposeSubblockV2C(
            dim_c=dim_c,
            dim_v=dim_v,
            const_patches_shape=const_patches_shape,
            token_pool_size=token_map_pool_size,
            enable_amp=False,  # standard for ViTs
        )

        # note that since all functions are deterministic, and we are
        # not using any stochastic elements such as dropout, we do
        # not need to control seeds for the random number generator.
        # To see usage with controlled seeds and dropout, see pyslowfast.

        self.seeds = {}
        self.block_id = block_id
        print(f"Block index : {self.block_id} | {num_heads} heads | Shift : {self.F.shift_size} | Dpr : {self.drop_path_rate}")
    
    def seed_cuda(self, key):
        """
        Fix seeds to allow for stochastic elements such as
        dropout to be reproduced exactly in activation
        recomputation in the backward pass.
        """

        # randomize seeds
        # use cuda generator if available
        if (
            hasattr(torch.cuda, "default_generators")
            and len(torch.cuda.default_generators) > 0
        ):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        self.seed_cuda("attn")
        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        self.seed_cuda("droppath")
        f_X_2_dropped = drop_path(
            f_X_2, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_1 = X_1 + f(X_2)
        Y_1 = X_1 + f_X_2_dropped

        # free memory since X_1 is now not needed
        del X_1

        self.seed_cuda("FFN")
        g_Y_1 = self.G(Y_1)

        torch.manual_seed(self.seeds["droppath"])
        g_Y_1_dropped = drop_path(
            g_Y_1, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1_dropped

        # free memory since X_2 is now not needed
        del X_2

        return Y_1, Y_2

    def backward_pass(
        self,
        Y_1,
        Y_2,
        dY_1,
        dY_2,
    ):
        """
        equation for activation recomputation:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention

        And we use pytorch native logic carefully to
        calculate gradients on F and G.
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            torch.manual_seed(self.seeds["FFN"])
            g_Y_1 = self.G(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = drop_path(
                g_Y_1, drop_prob=self.drop_path_rate, training=self.training
            )

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence we do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dY_1 = dY_1 + Y_1.grad

            # free memory since Y_1.grad is now not needed
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.F(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = drop_path(
                f_X_2, drop_prob=self.drop_path_rate, training=self.training
            )

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():
            # recomputing X_1 from the rev equation
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            # the gradients for the previous block
            # note that it is called dY_2 but it in fact dX_2 in math.
            # reusing same variable to save memory
            dY_2 = dY_2 + X_2.grad

            # free memory since X_2.grad is now not needed
            X_2.grad = None

            X_2 = X_2.detach()

        # et voila~
        return X_1, X_2, dY_1, dY_2
    

###############################
##       F SubBlocks         ##
###############################


class MHPAModifiedFBlockC2V(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        num_heads,
        patches_shape,
        token_pool_size,
        kv_pool_size,
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)

        self.dim_c, self.dim_v = dim_c, dim_v
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.N_v = (self.patches_shape[0]//token_pool_size) * (self.patches_shape[1]//token_pool_size)
        self.N_kv = (self.patches_shape[0]//kv_pool_size) * (self.patches_shape[1]//kv_pool_size)
        self.num_heads = num_heads

        assert self.dim_c % self.num_heads == 0
        assert self.dim_v % self.num_heads == 0

        self.scale = (self.dim_c // self.num_heads)**-0.5 # Attention is computed at size dim_c/num_heads

        self.qkv = nn.Linear(self.dim_c, 3*self.dim_c)

        dim_conv = self.dim_c // self.num_heads

        # GroupedConv for Pooling (N_c, d_c) --> (N_v, d_c)        
        self.pool_q = nn.Conv2d(
            in_channels=dim_conv,
            out_channels=dim_conv,
            kernel_size=(token_pool_size, token_pool_size),
            stride=(token_pool_size, token_pool_size),
            padding=0,
            groups=dim_conv,
            bias=False,
        )

        # GroupedConv for Pooling (N_c, d_c) --> (N_kv, d_c) (N_kv const across stages)
        self.pool_k = nn.Conv2d(
            dim_conv,
            dim_conv,
            (kv_pool_size, kv_pool_size),
            stride=(kv_pool_size, kv_pool_size),
            padding=0,
            groups=dim_conv,
            bias=False,
        )

        # Ungrouped Conv for (N_c, d_c) --> (N_kv, d_v)
        self.pool_v = nn.Conv2d(
            dim_conv,
            dim_v // num_heads,
            (kv_pool_size, kv_pool_size),
            stride=(kv_pool_size, kv_pool_size),
            padding=0,
            groups=1,
            bias=False,
        )

        # Norm
        self.norm_q = nn.LayerNorm(dim_conv)
        self.norm_k = nn.LayerNorm(dim_conv)
        self.norm_v = nn.LayerNorm(dim_v // num_heads)

        self.proj = nn.Linear(self.dim_v, self.dim_v)

        self.patches_shape = patches_shape
        self.enable_amp = enable_amp

    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            B, N_c, d_c = x.shape
            
            x = self.norm(x)    
            qkv = self.qkv(x).reshape(B, N_c, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N_c, d_c/H)


            q = q.reshape(B * self.num_heads, self.patches_shape[0], self.patches_shape[1], -1).permute(0, 3, 1, 2).contiguous() # (BH, d_c/H, H_c, W_c)
            k = k.reshape(B * self.num_heads, self.patches_shape[0], self.patches_shape[1], -1).permute(0, 3, 1, 2).contiguous() # (BH, d_c/H, H_c, W_c)
            v = v.reshape(B * self.num_heads, self.patches_shape[0], self.patches_shape[1], -1).permute(0, 3, 1, 2).contiguous() # (BH, d_c/H, H_c, W_c)


            q = self.norm_q(self.pool_q(q).reshape(B, self.num_heads, self.N_v, -1)) # (B, H, N_v, d_c/H)
            k = self.norm_k(self.pool_k(k).reshape(B, self.num_heads, self.N_kv, -1)) # (B, H, N_kv, d_c/H)
            v = self.norm_v(self.pool_v(v).reshape(B, self.num_heads, self.N_kv, -1)) # (B, H, N_kv, d_v/H)

            attn = (q * self.scale) @ k.transpose(-2, -1) # (B, H, N_v, N_kv)
            attn = attn.softmax(dim=-1)
            x = attn @ v # (B, H, N_v, d_v/H)

            x = x.transpose(1, 2).reshape(B, -1, self.dim_v) # (B, N_v, d_v)
            x = self.proj(x) # (B, N_v, d_v)

            return x
        

class SwinFBlockC2V(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        num_heads, 
        window_size, 
        shift_size,
        patches_shape,
        token_pool_size,
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)

        self.conv = nn.Conv2d(in_channels=dim_c, out_channels=dim_v, kernel_size=token_pool_size, stride=token_pool_size)

        self.dim_c, self.dim_v = dim_c, dim_v
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.N_v = (self.patches_shape[0]//token_pool_size) * (self.patches_shape[1]//token_pool_size)
        self.swin_input_sizes = (self.patches_shape[0] // token_pool_size, self.patches_shape[1] // token_pool_size)

        self.patches_shape = patches_shape
        self.enable_amp = enable_amp

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm_attn = nn.LayerNorm(dim_v, eps=1e-6, elementwise_affine=True)

        if min(self.swin_input_sizes) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.swin_input_sizes)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            dim_v, window_size=(self.window_size, self.window_size), num_heads=self.num_heads,
            qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.swin_input_sizes
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        
    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            H, W = self.swin_input_sizes
            C = self.dim_v
            B, _, _ = x.shape

            x = self.norm(x)

            x = x.transpose(1,2)
            x = x.reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])
            x = self.conv(x).reshape(B, self.N_v, self.dim_v)

            x = self.norm_attn(x)

            x = x.view(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                shifted_x = x
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

            # reverse cyclic shift
            if self.shift_size > 0:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = shifted_x
            x = x.view(B, H * W, C)

            return x


class SwinMLPFBlockC2V(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        num_heads, 
        window_size, 
        shift_size,
        patches_shape,
        token_pool_size,
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)

        self.conv = nn.Conv2d(in_channels=dim_c, out_channels=dim_v, kernel_size=token_pool_size, stride=token_pool_size)

        self.dim_c, self.dim_v = dim_c, dim_v
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.N_v = (self.patches_shape[0]//token_pool_size) * (self.patches_shape[1]//token_pool_size)
        self.swin_input_sizes = (self.patches_shape[0] // token_pool_size, self.patches_shape[1] // token_pool_size)

        self.patches_shape = patches_shape
        self.enable_amp = enable_amp

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.swin_input_sizes) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.swin_input_sizes)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        # use group convolution to implement multi-head MLP
        self.spatial_mlp = nn.Conv1d(
            self.num_heads * self.window_size ** 2,
            self.num_heads * self.window_size ** 2,
            kernel_size=1,
            groups=self.num_heads
        )
        
    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            H, W = self.swin_input_sizes
            C = self.dim_v
            B, _, _ = x.shape

            x = self.norm(x)

            x = x.transpose(1,2)
            x = x.reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])
            x = self.conv(x)

            x = x.reshape(B, H, W, C)

            # shift
            if self.shift_size > 0:
                P_l, P_r, P_t, P_b = self.padding
                shifted_x = torch.nn.functional.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
            else:
                shifted_x = x
            _, _H, _W, _ = shifted_x.shape

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # Window/Shifted-Window Spatial MLP
            x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
            x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size, C//nH
            x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                    C // self.num_heads)
            spatial_mlp_windows = self.spatial_mlp(x_windows_heads)  # nW*B, nH*window_size*window_size, C//nH
            spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size,
                                                        C // self.num_heads).transpose(1, 2)
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

            # merge windows
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)  # B H' W' C

            # reverse shift
            if self.shift_size > 0:
                P_l, P_r, P_t, P_b = self.padding
                x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
            else:
                x = shifted_x
            x = x.view(B, H * W, C)

            return x


class SwinDWMLPFBlockC2V(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        num_heads, 
        window_size, 
        shift_size,
        patches_shape,
        token_pool_size,
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)

        self.proj = nn.Linear(dim_c, dim_v)
        self.act = nn.ReLU()

        self.dim_c, self.dim_v = dim_c, dim_v
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.N_v = (self.patches_shape[0]//token_pool_size) * (self.patches_shape[1]//token_pool_size)

        self.patches_shape = patches_shape
        self.enable_amp = enable_amp

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.patches_shape) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.patches_shape)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size < self.token_pool_size:
            self.shift_size = 0

        assert self.window_size % self.token_pool_size == 0
        assert self.shift_size % self.token_pool_size == 0
        self.window_size_out = self.window_size // self.token_pool_size

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        # use group convolution to implement multi-head MLP
        self.spatial_mlp = nn.Conv1d(
            self.num_heads * self.window_size ** 2,
            self.num_heads * (self.window_size // self.token_pool_size) ** 2,
            kernel_size=1,
            groups=self.num_heads
        )

    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            H, W = self.patches_shape
            C = self.dim_c
            B, _, _ = x.shape

            x = self.norm(x)

            # x = self.act(self.lin(x))

            x = x.view(B, H, W, C)

            # shift
            if self.shift_size > 0:
                P_l, P_r, P_t, P_b = self.padding
                shifted_x = torch.nn.functional.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
            else:
                shifted_x = x
            _, _H, _W, _ = shifted_x.shape

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # Window/Shifted-Window Spatial MLP
            x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
            x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size, C//nH
            x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                    C // self.num_heads)
            spatial_mlp_windows = self.spatial_mlp(x_windows_heads)  # nW*B, nH*window_size*window_size, C//nH

            spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size_out * self.window_size_out,
                                                        C // self.num_heads).transpose(1, 2)
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size_out * self.window_size_out, C)

            # merge windows
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size_out, self.window_size_out, C)
            shifted_x = window_reverse(spatial_mlp_windows, self.window_size_out, _H // self.token_pool_size, _W // self.token_pool_size)  # B H' W' C
            # reverse shift
            if self.shift_size > 0:
                P_l, P_r, P_t, P_b = [x // self.token_pool_size for x in self.padding]
                x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
            else:
                x = shifted_x
            x = x.view(B, self.N_v, C)

            x = self.proj(self.act(x))
            return x
        

###############################
##       G SubBlocks         ##
###############################


class FFNConvTransposeSubblockV2C(nn.Module):

    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        const_patches_shape,
        token_pool_size,
        enable_amp=False,  # standard for ViTs
    ):
        super().__init__()

        self.patches_shape = (const_patches_shape[0]//token_pool_size, const_patches_shape[1]//token_pool_size)
        self.dim_c, self.dim_v = dim_c, dim_v

        self.norm = nn.LayerNorm(self.dim_v)

        self.fc1 = nn.Linear(self.dim_v, 4*self.dim_v)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4*self.dim_v, self.dim_v)

        self.convtranspose = nn.ConvTranspose2d(in_channels=dim_v, out_channels=dim_c, 
                                                kernel_size=token_pool_size, stride=token_pool_size, groups=dim_c)

        self.enable_amp = enable_amp

    def forward(self, x):
        # The reason for implementing autocast inside forward loop instead
        # in the main training logic is the implicit forward pass during
        # memory efficient gradient backpropagation. In backward pass, the
        # activations need to be recomputed, and if the forward has happened
        # with mixed precision, the recomputation must also be so. This cannot
        # be handled with the autocast setup in main training logic.
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x = self.norm(x)

            B, N_v, d_v = x.shape
            x = torch.nn.functional.relu(self.fc2(self.act(self.fc1(x)))) # FFN

            x = x.transpose(1,2).reshape(B, self.dim_v, self.patches_shape[0], self.patches_shape[1])
            x = self.convtranspose(x)
            x = x.reshape(B, self.dim_c, -1).transpose(1, 2)
            return x
        

class MLPSubblockV2C(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        const_patches_shape,
        token_pool_size,
        enable_amp=False,  # standard for ViTs
    ):
        super().__init__()

        self.patches_shape = (const_patches_shape[0]//token_pool_size, const_patches_shape[1]//token_pool_size)
        self.dim_c = dim_c

        self.norm = nn.LayerNorm(dim_v)

        self.fc1 = nn.Linear(dim_v, dim_c)
        self.act = nn.GELU()
        self.convtranspose = nn.ConvTranspose2d(in_channels=dim_c, out_channels=dim_c, 
                                                kernel_size=token_pool_size, stride=token_pool_size, groups=1)

        self.enable_amp = enable_amp

    def forward(self, x):
        # The reason for implementing autocast inside forward loop instead
        # in the main training logic is the implicit forward pass during
        # memory efficient gradient backpropagation. In backward pass, the
        # activations need to be recomputed, and if the forward has happened
        # with mixed precision, the recomputation must also be so. This cannot
        # be handled with the autocast setup in main training logic.
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x = self.norm(x)
            B, N, d_v = x.shape
            x = self.act(self.fc1(x))
            x = x.transpose(1,2).reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])

            x = self.convtranspose(x)
            x = x.reshape(B, self.dim_c, -1).transpose(1, 2)
            return x


####################################
##  Parts for Swin Attention      ##
####################################

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x