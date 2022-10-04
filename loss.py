import torch
from torch import Tensor
from torch.nn import Module


class MaskedBMLoss(Module):

    def __init__(self, loss_fn: Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        loss = []
        for i, frame in enumerate(n_frames):
            loss.append(self.loss_fn(pred[i, :, :frame], true[i, :, :frame]))
        return torch.mean(torch.stack(loss))


class MaskedFrameLoss(Module):

    def __init__(self, loss_fn: Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        # input: (B, T)
        loss = []
        for i, frame in enumerate(n_frames):
            loss.append(self.loss_fn(pred[i, :frame], true[i, :frame]))
        return torch.mean(torch.stack(loss))


class MaskedContrastLoss(Module):

    def __init__(self, margin: float = 0.99):
        super().__init__()
        self.margin = margin

    def forward(self, pred1: Tensor, pred2: Tensor, labels: Tensor, n_frames: Tensor):
        # input: (B, C, T)
        loss = []
        for i, frame in enumerate(n_frames):
            # mean L2 distance squared
            d = torch.dist(pred1[i, :, :frame], pred2[i, :, :frame], 2)
            if labels[i]:
                # if is positive pair, minimize distance
                loss.append(d ** 2)
            else:
                # if is negative pair, minimize (margin - distance) if distance < margin
                loss.append(torch.clip(self.margin - d, min=0.) ** 2)
        return torch.mean(torch.stack(loss))
