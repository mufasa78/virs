import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from models.base_model import BaseModel

class WindowAttention(nn.Module):
    """Window based multi-head self attention module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wt, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if self.scale is None:
            self.scale = head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"))  # 3, Wt, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wt*Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wt*Wh*Ww, Wt*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wt*Wh*Ww, Wt*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wt*Wh*Ww, Wt*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wt*Wh*Ww, Wt*Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Apply scaling to query
        scale_factor = torch.tensor(self.scale, dtype=q.dtype, device=q.device)
        q = q * scale_factor
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wt*Wh*Ww,Wt*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wt*Wh*Ww, Wt*Wh*Ww
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


class Mlp(nn.Module):
    """Multilayer perceptron."""

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


def window_partition(x, window_size):
    """
    Args:
        x: (B, T, H, W, C)
        window_size (tuple[int]): window size (Wt, Wh, Ww)

    Returns:
        windows: (num_windows*B, Wt, Wh, Ww, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, B, T, H, W):
    """
    Args:
        windows: (num_windows*B, Wt, Wh, Ww, C)
        window_size (tuple[int]): Window size (Wt, Wh, Ww)
        B (int): Batch size
        T (int): Number of frames
        H (int): Height
        W (int): Width

    Returns:
        x: (B, T, H, W, C)
    """
    x = windows.view(B, T // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block."""

    def __init__(self, dim, num_heads, window_size=(2, 8, 8), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must be less than window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must be less than window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must be less than window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix=None):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, T, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        B, T, H, W, C = x.shape
        window_size, shift_size = self.window_size, self.shift_size

        shortcut = x
        x = self.norm1(x)

        # Padding if needed
        pad_t = (window_size[0] - T % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - W % window_size[2]) % window_size[2]

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))

        _, Tp, Hp, Wp, _ = x.shape

        # Cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            if mask_matrix is not None:
                attn_mask = mask_matrix
            else:
                attn_mask = None
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, window_size)  # (num_windows*B, Wt, Wh, Ww, C)
        x_windows = x_windows.view(-1, window_size[0] * window_size[1] * window_size[2], C)  # (num_windows*B, Wt*Wh*Ww, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # (num_windows*B, Wt*Wh*Ww, C)

        # Merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], window_size[2], C)  # (num_windows*B, Wt, Wh, Ww, C)
        shifted_x = window_reverse(attn_windows, window_size, B, Tp, Hp, Wp)  # (B, T, H, W, C)

        # Reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        # Remove padding if needed
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :T, :H, :W, :].contiguous()

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0., scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize

        if self.scale_by_keep:
            x = x.div(keep_prob) * random_tensor
        else:
            x = x * random_tensor

        return x


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding."""

    def __init__(self, patch_size=(1, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # Input: (B, C, T, H, W)
        B, C, T, H, W = x.size()

        # FIXME: look at relaxing size constraints
        assert T % self.patch_size[0] == 0 and H % self.patch_size[1] == 0 and W % self.patch_size[2] == 0, \
            f"Input size ({T}*{H}*{W}) is not divisible by patch size ({self.patch_size[0]}*{self.patch_size[1]}*{self.patch_size[2]})."

        x = self.proj(x)  # (B, embed_dim, T', H', W')

        T, H, W = T // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', embed_dim)

        if self.norm is not None:
            x = self.norm(x)

        x = x.reshape(B, T, H, W, -1)  # (B, T', H', W', embed_dim)

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, T, H, W, C).

        Returns:
            x: Output feature, tensor size (B, T, H/2, W/2, 2*C).
        """
        B, T, H, W, C = x.shape

        # Ensure H and W are even
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, :, 0::2, 0::2, :]  # (B, T, H/2, W/2, C)
        x1 = x[:, :, 1::2, 0::2, :]  # (B, T, H/2, W/2, C)
        x2 = x[:, :, 0::2, 1::2, :]  # (B, T, H/2, W/2, C)
        x3 = x[:, :, 1::2, 1::2, :]  # (B, T, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, T, H/2, W/2, 4*C)
        x = x.view(B, T, H // 2, W // 2, 4 * C)  # (B, T, H/2, W/2, 4*C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, T, H/2, W/2, 2*C)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, depth, num_heads, window_size=(2, 8, 8),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, T, H, W, C).

        Returns:
            x: Output feature, tensor size (B, T, H, W, C).
        """
        # Calculate attention mask for SW-MSA
        B, T, H, W, C = x.shape
        window_size, shift_size = self.window_size, self.shift_size

        # Generate attention mask
        Tp = int(math.ceil(T / window_size[0])) * window_size[0]
        Hp = int(math.ceil(H / window_size[1])) * window_size[1]
        Wp = int(math.ceil(W / window_size[2])) * window_size[2]

        img_mask = torch.zeros((1, Tp, Hp, Wp, 1), device=x.device)  # (1, Tp, Hp, Wp, 1)

        cnt = 0
        for t in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, t, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # (nW, Wt, Wh, Ww, 1)
        mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])  # (nW, Wt*Wh*Ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, Wt*Wh*Ww, Wt*Wh*Ww)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # Forward through blocks
        for blk in self.blocks:
            x = blk(x, attn_mask)

        # Downsample if needed
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VRT(BaseModel):
    """Video Restoration Transformer (VRT)."""

    def __init__(self, img_size=(8, 256, 256), patch_size=(1, 4, 4), in_chans=3, out_chans=3,
                 embed_dim=96, depths=[8, 8, 8, 8], num_heads=[6, 6, 6, 6],
                 window_size=(2, 8, 8), mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False):
        super(VRT, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_layers = len(depths)

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        # Build decoder layers (upsampling)
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, 0, -1):
            layer = nn.Sequential(
                nn.ConvTranspose3d(
                    int(embed_dim * 2 ** i_layer),
                    int(embed_dim * 2 ** (i_layer - 1)),
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2)
                ),
                nn.LeakyReLU(0.2, True)
            )
            self.decoder_layers.append(layer)

        # Final output layer
        self.output_layer = nn.Conv3d(embed_dim, out_chans, kernel_size=3, stride=1, padding=1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Forward function for feature extraction.

        Args:
            x: Input tensor with shape (B, C, T, H, W)

        Returns:
            Feature maps at different scales
        """
        x = self.patch_embed(x)  # (B, T', H', W', C)

        # Store feature maps for skip connections
        features = []

        # Forward through encoder layers
        for layer in self.layers:
            features.append(x)
            x = layer(x)

        return x, features

    def forward(self, x):
        """Forward function.

        Args:
            x: Input tensor with shape (B, C, T, H, W)

        Returns:
            Output tensor with shape (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape

        # Extract features
        x, features = self.forward_features(x)  # (B, T', H', W', C')

        # Convert to channel-first format for decoder
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C', T', H', W')

        # Forward through decoder layers with skip connections
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x)  # (B, C'//2, T', H'*2, W'*2)

            # Get corresponding encoder feature map
            skip = features[-(i+2)]  # Skip connection from encoder
            skip = skip.permute(0, 4, 1, 2, 3).contiguous()  # (B, C'//2, T', H'*2, W'*2)

            # Add skip connection
            x = x + skip

        # Final output layer
        x = self.output_layer(x)  # (B, C, T, H, W)

        # Ensure output size matches input size
        if x.size(2) != T or x.size(3) != H or x.size(4) != W:
            x = F.interpolate(x, size=(T, H, W), mode='trilinear', align_corners=False)

        return x
