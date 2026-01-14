from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv3 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=2, padding=3)

    def forward(self, x: torch.Tensor, condense: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        res = x
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = x + res

        if condense:
            x = F.relu(self.conv3(x))

        return x, res


class Decoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.c2 = nn.Conv1d(in_ch + out_ch, out_ch, kernel_size=5, padding="same")
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.c3 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.up(x))
        x = torch.cat([x, skip], dim=1)

        x = F.relu(self.c2(x))
        x = self.bn1(x)

        x = F.relu(self.c3(x))
        x = self.bn2(x)

        return x


class GCNUNet(nn.Module):
    """
    1D U-Net that outputs a per-position "risk/fitness map" (B, L).
    Scalar prediction is typically mean(risk_map, dim=1).
    """
    def __init__(self, hidden: int = 128, bottleneck: int = 256):
        super().__init__()
        self.enc1 = Encoder(4, hidden)
        self.enc2 = Encoder(hidden, hidden)
        self.enc3 = Encoder(hidden, bottleneck)
        self.enc4 = Encoder(bottleneck, bottleneck)
        self.cb1 = Encoder(bottleneck, bottleneck)
        self.cb2 = Encoder(bottleneck, bottleneck)

        self.dec1 = Decoder(bottleneck, hidden)
        self.dec2 = Decoder(hidden, hidden)
        self.dec3 = Decoder(hidden, hidden)

        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 4) -> (B, 4, L)
        x = x.permute(0, 2, 1)

        x, s1 = self.enc1(x, condense=True)
        x, s2 = self.enc2(x, condense=True)
        x, s3 = self.enc3(x, condense=True)

        x, _ = self.enc4(x, condense=False)
        x, _ = self.cb1(x, condense=False)
        x, _ = self.cb2(x, condense=False)

        x = self.dec1(x, s3)
        x = self.dec2(x, s2)
        x = self.dec3(x, s1)

        # (B, 1, L) -> (B, L)
        return self.head(x).squeeze(1)
