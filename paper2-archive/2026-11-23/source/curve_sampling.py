import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class RoadSnakeAdapter(nn.Module):
    """Add a zero-gated local/curved residual while preserving the incoming feature exactly at initialization."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        expansion: float = 0.25,
        max_offset: float = 1.0,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd and >= 3, got {kernel_size}")
        if not 0 < expansion <= 1:
            raise ValueError(f"expansion must be in (0, 1], got {expansion}")
        if max_offset <= 0:
            raise ValueError(f"max_offset must be positive, got {max_offset}")

        hidden = max(8, int(round(channels * expansion / 8)) * 8)
        self.channels = channels
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.max_offset = float(max_offset)

        self.reduce = Conv(channels, hidden, 1)
        self.local = Conv(hidden, hidden, 3, g=hidden)
        self.offset = nn.Conv2d(hidden, 2 * kernel_size, 3, padding=1)
        self.horizontal_weight = nn.Parameter(
            torch.full((hidden, kernel_size), 1.0 / kernel_size)
        )
        self.vertical_weight = nn.Parameter(
            torch.full((hidden, kernel_size), 1.0 / kernel_size)
        )
        groups = max(1, min(8, hidden // 4))
        while hidden % groups:
            groups -= 1
        self.horizontal_norm = nn.GroupNorm(groups, hidden)
        self.vertical_norm = nn.GroupNorm(groups, hidden)
        self.act = nn.SiLU(inplace=True)
        self.fuse = Conv(3 * hidden, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # Runtime-only deployment control used by RoadSnake-Anneal.  Keeping this as a
        # plain attribute preserves all historical checkpoint/state_dict contracts.
        self.anneal_scale = 1.0
        self.capture_geometry_offsets = False
        self.geometry_offsets = None

        positions = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        self.register_buffer("positions", positions, persistent=False)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def set_anneal_scale(self, scale: float) -> None:
        """Set the externally scheduled residual scale in [0, 1]."""
        scale = float(scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError(f"anneal scale must be in [0, 1], got {scale}")
        self.anneal_scale = scale

    @staticmethod
    def _cumulative_offsets(offset: torch.Tensor) -> torch.Tensor:
        """Accumulate offsets outwards from the fixed kernel center without Python loops."""
        center = offset.shape[1] // 2
        accumulated = torch.zeros_like(offset)
        if center:
            left = torch.flip(
                torch.cumsum(torch.flip(offset[:, :center], dims=(1,)), dim=1),
                dims=(1,),
            )
            right = torch.cumsum(offset[:, center + 1 :], dim=1)
            accumulated[:, :center] = left
            accumulated[:, center + 1 :] = right
        return accumulated

    @staticmethod
    def _normalize_grid(coordinate: torch.Tensor, size: int) -> torch.Tensor:
        """Convert pixel coordinates to the align_corners=True grid_sample domain."""
        if size <= 1:
            return torch.zeros_like(coordinate)
        return coordinate.mul(2.0 / (size - 1)).sub(1.0)

    def _sample_curve(
        self,
        feature: torch.Tensor,
        orthogonal_offset: torch.Tensor,
        horizontal: bool,
    ) -> torch.Tensor:
        """Sample a continuous horizontal or vertical snake and collapse its K points channel-wise."""
        batch, channels, height, width = feature.shape
        dtype, device = feature.dtype, feature.device
        base_y, base_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        base_y = base_y.view(1, 1, height, width)
        base_x = base_x.view(1, 1, height, width)
        positions = self.positions.to(device=device, dtype=dtype).view(
            1, self.kernel_size, 1, 1
        )
        curved = self._cumulative_offsets(orthogonal_offset).mul(self.max_offset)

        if horizontal:
            grid_y = base_y + curved
            grid_x = (base_x + positions).expand(batch, -1, -1, -1)
            weight = self.horizontal_weight
        else:
            grid_y = (base_y + positions).expand(batch, -1, -1, -1)
            grid_x = base_x + curved
            weight = self.vertical_weight

        grid = torch.stack(
            (self._normalize_grid(grid_x, width), self._normalize_grid(grid_y, height)),
            dim=-1,
        )
        grid = grid.permute(0, 2, 3, 1, 4).reshape(
            batch, height, width * self.kernel_size, 2
        )
        sampled = F.grid_sample(
            feature, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        sampled = sampled.reshape(
            batch, channels, height, width, self.kernel_size
        ).permute(0, 1, 4, 2, 3)
        return (
            sampled * weight.to(dtype=dtype).view(1, channels, self.kernel_size, 1, 1)
        ).sum(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unmodified input at step zero, then learn a bounded curved residual."""
        # The native tail must be a real bypass: no RoadSnake kernels, grid sampling,
        # normalization updates, or adapter gradients are executed when the scale is zero.
        # Historical R1 checkpoints predate ``anneal_scale``. Pickle restores their
        # old instance dictionaries without calling __init__, so default them to the
        # original fully-active behavior.
        anneal_scale = float(getattr(self, "anneal_scale", 1.0))
        if anneal_scale == 0.0:
            return x
        reduced = self.reduce(x)
        offset_h, offset_v = self.offset(reduced).tanh().chunk(2, dim=1)
        if self.training and getattr(self, "capture_geometry_offsets", False):
            # A second offset-only pass on detached features ensures that the
            # auxiliary direction loss cannot update P4, reduce, Neck, or the
            # detector. The normal detection path above remains untouched.
            self.geometry_offsets = self.offset(reduced.detach()).tanh().chunk(2, dim=1)
        else:
            self.geometry_offsets = None
        horizontal = self.act(
            self.horizontal_norm(self._sample_curve(reduced, offset_h, horizontal=True))
        )
        vertical = self.act(
            self.vertical_norm(self._sample_curve(reduced, offset_v, horizontal=False))
        )
        local = self.local(reduced)
        residual = self.fuse(torch.cat((local, horizontal, vertical), dim=1))
        return x + (anneal_scale * self.gamma).to(dtype=x.dtype) * residual
