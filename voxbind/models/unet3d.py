import torch
from torch import nn
from typing import Tuple, Union, List

from voxbind.constants import N_POCKET_ELEMENTS, N_LIGAND_ELEMENTS


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_groups: int = 16,
        dropout: float = 0.1,
    ):
        """
        Residual block module for the UNet3D model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_groups (int, optional): Number of groups for group normalization. Defaults to 16.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()

        self.use_norm = n_groups > 0
        # first norm + conv layer
        if self.use_norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=(3, 3, 3), padding=1
        )

        # second norm + conv layer
        if self.use_norm:
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=(3, 3, 3), padding=1
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        else:
            self.shortcut = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.use_norm:
            h = self.norm1(x)
            h = self.act1(h)
        else:
            h = self.act1(x)
        h = self.conv1(h)

        if self.use_norm:
            h = self.norm2(h)
        h = self.act2(h)
        if hasattr(self, "dropout"):
            h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        d_k: int = None,
        n_groups: int = 16
    ):
        """
        Initializes an AttentionBlock.

        Args:
            n_channels (int): Number of input channels.
            n_heads (int, optional): Number of attention heads. Defaults to 1.
            d_k (int, optional): Dimensionality of the key and query vectors. Defaults to None.
                If None, it is set equal to n_channels.
            n_groups (int, optional): Number of groups for group normalization. Defaults to 16.
                If n_groups <= 0, no normalization is applied.
        """
        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.use_norm = n_groups > 0
        self.n_heads = n_heads
        self.d_k = d_k

        if self.use_norm:
            self.norm = nn.GroupNorm(n_groups, n_channels)

        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_channels, height, width, depth).
        """
        batch_size, n_channels, height, width, depth = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum("bihd, bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum("bijh, bjhd->bihd", attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width, depth)
        return res


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_groups: int, has_attn: bool, dropout: float):
        """
        DownBlock class represents a block in the down-sampling path of a U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_groups (int): Number of groups for group normalization.
            has_attn (bool): Whether to include attention mechanism in the block.
            dropout (float): Dropout rate.

        """
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, n_groups=n_groups, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DownBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_groups: int, has_attn: bool, dropout: float):
        """
        UpBlock is a module that represents an upsampling block in a 3D U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_groups (int): Number of groups for group normalization.
            has_attn (bool): Whether to include attention mechanism in the block.
            dropout (float): Dropout rate.

        """
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, n_groups=n_groups, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UpBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, n_groups: int, dropout: float):
        """
        Initializes a MiddleBlock instance.

        Args:
            n_channels (int): Number of input and output channels.
            n_groups (int): Number of groups for group normalization.
            dropout (float): Dropout rate.

        """
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, n_groups=n_groups, dropout=dropout)
        self.attn = AttentionBlock(n_channels, n_groups=n_groups)
        self.res2 = ResidualBlock(n_channels, n_channels, n_groups=n_groups, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the MiddleBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        """
        Upsample module that performs 3D transposed convolution to upsample the input tensor.

        Args:
            n_channels (int): Number of input and output channels.
        """
        super().__init__()
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, (4, 4, 4), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Upsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Upsampled tensor.
        """
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        """
        Downsample module that performs 3D convolution with stride 2.

        Args:
            n_channels (int): Number of input and output channels.
        """
        super().__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, (3, 3, 3), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Downsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after 3D convolution with stride 2.
        """
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        n_inp_channels: int = N_POCKET_ELEMENTS + N_LIGAND_ELEMENTS,
        n_out_channels: int = N_LIGAND_ELEMENTS,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2,
        n_groups: int = 32,
        dropout: float = 0.1,
        smooth_sigma: float = 0.0,
        verbose: bool = False
    ):
        """
        3D U-Net model for voxel-based binding affinity prediction.

        Args:
            n_inp_channels (int): Number of input channels.
            n_out_channels (int): Number of output channels.
            n_channels (int): Number of channels in the model.
            ch_mults (Union[Tuple[int, ...], List[int]]): Channel multipliers for each resolution level.
            is_attn (Union[Tuple[bool, ...], List[int]]): Attention flag for each resolution level.
            n_blocks (int): Number of blocks in each resolution level.
            n_groups (int): Number of groups for group normalization.
            dropout (float): Dropout rate.
            smooth_sigma (float): Standard deviation for Gaussian smoothing.
            verbose (bool): Whether to print the number of parameters in the model.
        """
        super().__init__()

        self.smooth_sigma = smooth_sigma
        n_resolutions = len(ch_mults)

        self.grid_projection = nn.Conv3d(n_inp_channels, n_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_groups, is_attn[i], dropout))
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_groups, dropout)

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_groups, is_attn[i], dropout))

            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_groups, is_attn[i], dropout))
            in_channels = out_channels

            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        if n_groups > 0:
            self.norm = nn.GroupNorm(n_groups, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv3d(in_channels, n_out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        if verbose:
            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f">> model has {(n_params/1e6):.02f}M parameters")

    def forward(self, ligand: torch.Tensor, pocket: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet3D model.

        Args:
            ligand (torch.Tensor): Input tensor representing the ligand.
            pocket (torch.Tensor): Input tensor representing the pocket.

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        if pocket is not None:
            x = torch.cat((ligand, pocket), axis=1)
        else:
            x = ligand
        x = self.grid_projection(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        if hasattr(self, "norm"):
            x = self.norm(x)
        x = self.act(x)
        x = self.final(x)
        return x
