"""
model.py — 1-D ResNet for multi-label ECG classification.

Architecture:
  Stem  : Conv1d(12→32, k=15, stride=2) + BN + ReLU
  Block1: ResBlock1D(32→64,  stride=2)
  Block2: ResBlock1D(64→128, stride=2)
  Block3: ResBlock1D(128→256, stride=2)
  Block4: ResBlock1D(256→256, stride=2)
  GAP   : AdaptiveAvgPool1d(1)
  Head  : Linear(256→4)

Input : (batch, 12, 1000)   — 12 leads × 1000 samples (10 s @ 100 Hz)
Output: (batch, 4)          — logits for NORM / AFIB / STD / STE
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelAUROC

LABELS = ["NORM", "AFIB", "STD", "STE"]
N_CLASSES = len(LABELS)


# ── building block ────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """Post-activation residual block for 1-D signals."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        # 1×1 projection when dimensions change
        self.downsample: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.drop(self.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


# ── Lightning module ──────────────────────────────────────────────────────────

class ECGResNet(pl.LightningModule):
    """
    1-D ResNet multi-label ECG classifier.

    Parameters
    ----------
    lr      : learning rate for Adam
    dropout : dropout probability applied inside each residual block
    pos_weight : optional (4,) tensor for BCEWithLogitsLoss class weighting
    """

    def __init__(self, lr: float = 1e-3, dropout: float = 0.2):
        super().__init__()
        self.save_hyperparameters()

        # ── stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # ── 4 residual blocks ─────────────────────────────────────────────────
        self.layer1 = ResBlock1D(32,  64,  stride=2, dropout=dropout)
        self.layer2 = ResBlock1D(64,  128, stride=2, dropout=dropout)
        self.layer3 = ResBlock1D(128, 256, stride=2, dropout=dropout)
        self.layer4 = ResBlock1D(256, 256, stride=2, dropout=dropout)

        # ── head ──────────────────────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(256, N_CLASSES)

        # ── loss & metrics ────────────────────────────────────────────────────
        self.loss_fn   = nn.BCEWithLogitsLoss()
        self.train_auc = MultilabelAUROC(num_labels=N_CLASSES, average=None)
        self.val_auc   = MultilabelAUROC(num_labels=N_CLASSES, average=None)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (batch, 4)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)   # (batch, 256)
        return self.fc(x)             # (batch, 4)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities of shape (batch, 4)."""
        return torch.sigmoid(self(x))

    # ── shared step ───────────────────────────────────────────────────────────

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        loss   = self.loss_fn(logits, y.float())
        probs  = torch.sigmoid(logits)
        return loss, probs, y

    # ── Lightning hooks ───────────────────────────────────────────────────────

    def training_step(self, batch, _batch_idx):
        loss, probs, y = self._step(batch)
        self.train_auc.update(probs, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        aucs = self.train_auc.compute()
        for label, auc in zip(LABELS, aucs):
            self.log(f"train_auc_{label}", auc)
        self.train_auc.reset()

    def validation_step(self, batch, _batch_idx):
        loss, probs, y = self._step(batch)
        self.val_auc.update(probs, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        aucs = self.val_auc.compute()
        mean_auc = aucs[~aucs.isnan()].mean()
        self.log("val_auc_mean", mean_auc, prog_bar=True)
        for label, auc in zip(LABELS, aucs):
            self.log(f"val_auc_{label}", auc)
        self.val_auc.reset()

    def configure_optimizers(self):
        opt   = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ── quick shape test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ECGResNet()
    x = torch.randn(4, 12, 1000)
    logits = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {logits.shape}")   # expect (4, 4)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
