import torch
import torch.nn as nn

class RenderLoss3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loss, x):
        loss = torch.mean(torch.abs(loss))
        return loss

