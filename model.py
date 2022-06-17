import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.functional import accuracy

from efficientnet_pytorch import EfficientNet

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

CLASSES = 8


class WhaleEfficientNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.efficient_net = EfficientNet.from_name('efficientnet-b7')
        # self.efficient_net.load_state_dict(torch.load(PRETRAINED_PATH))
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7', num_classes=CLASSES)
        in_features = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Linear(in_features, CLASSES)

    def forward(self, x):
        out = self.efficient_net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)

        # Weighted Cross Entropy
        # nSamples = [2492, 2474, 2423, 2254, 2213, 2013, 1641, 978]
        # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        # normedWeights = torch.FloatTensor(normedWeights).to('cuda')
        # criterion = nn.CrossEntropyLoss(normedWeights)
        # loss = criterion(y_hat, y)

        loss = F.cross_entropy(y_hat, y)

        acc = accuracy(y_hat, y)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True),
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)
