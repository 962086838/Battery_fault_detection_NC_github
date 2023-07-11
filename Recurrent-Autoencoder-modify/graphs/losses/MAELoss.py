"""
MAE loss
"""

import torch
import torch.nn as nn

class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction = 'mean')

    def forward(self, y_hat, y_true):
        loss = self.loss(y_hat, y_true)
        return loss
