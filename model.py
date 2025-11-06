import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Temporal correlation information extraction: LSTM → tanh(L×L).

    Produces per-step hidden outputs [B,L,H]; the correlation module will form
    a time-time matrix aligned with the original TSADCNN design.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        out, _ = self.lstm(x)  # [B, L, H]
        return out


class TemporalCorrelation(nn.Module):
    """Compute time-time correlation matrix S = tanh(Y · Yᵀ / √H)."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scale = hidden_dim ** 0.5

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        # Y: [B, L, H]
        S = torch.bmm(Y, Y.transpose(1, 2))  # [B, L, L] 其实就是Y的时间序列的内积矩阵 Y^T * Y 然后得到B，L，L 矩阵
        S = S / (self.scale + 1e-6)
        S = torch.tanh(S)
        return S.unsqueeze(1)  # [B, 1, L, L]


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return x + y


class SpatialExtractor(nn.Module):
    """Spatial structure information extraction: fused 3×3 + 1×1, then residual convs."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.bn0 = nn.BatchNorm2d(channels)

        self.block1 = ConvResidualBlock(channels)
        self.block2 = ConvResidualBlock(channels)
        self.block3 = ConvResidualBlock(channels)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        # S: [B, 1, L, L]
        y = self.conv3(S) + self.conv1(S)
        y = self.bn0(y)
        y = self.act(y)
        y = self.block1(y)
        # y = self.act(y)
        y = self.block2(y)
        # y = self.act(y)
        y = self.block3(y)
        # y = self.act(y)
        return y  # [B, C, L, L]


class TSADCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, conv_channels: int, embed_dim: int, dropout: float):
        super().__init__()
        # Temporal correlation module
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.tcorr = TemporalCorrelation(hidden_dim)
        # Spatial extractor with residual connections
        self.spatial = SpatialExtractor(conv_channels)
        # Head: global pooling + FC
        self.head = nn.Sequential(
            nn.Linear(conv_channels, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim)
        )

    def encode(self, traj: torch.Tensor) -> torch.Tensor:
        Y = self.encoder(traj)            # [B, L, H]
        S = self.tcorr(Y)                 # [B, 1, L, L]
        Fmap = self.spatial(S)            # [B, C, L, L]
        pooled = F.adaptive_avg_pool2d(Fmap, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
        z = self.head(pooled)             # [B, E]
        z = F.normalize(z, dim=1)
        return z

    def forward(self, old_traj: torch.Tensor, new_traj: torch.Tensor):
        z_old = self.encode(old_traj)
        z_new = self.encode(new_traj)
        return z_old, z_new


def contrastive_loss(z_old: torch.Tensor, z_new: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    distances = torch.norm(z_old - z_new, dim=1)
    pos_loss = labels.float() * (distances ** 2)
    neg_loss = (1.0 - labels.float()) * (F.relu(margin - distances) ** 2)
    return 0.5 * (pos_loss + neg_loss).mean()