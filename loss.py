from torch import nn
import torch
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    """
    input: logits, target
    output: CrossEntropyLoss2d
    """
    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()

        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        return F.nll_loss(F.log_softmax(logits, dim=1), targets, weight=self.weight,
                          ignore_index=self.ignore_index, reduction='mean')


class FocalLoss2d(nn.Module):
    """
    input: logits, target
    output: FocalLoss2d
    """
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        softmax_input = F.softmax(logits, dim=1)
        log_softmax_input = torch.log(softmax_input)
        return F.nll_loss(torch.pow(1-softmax_input, self.gamma)*log_softmax_input, targets,
                          weight=self.weight,ignore_index=self.ignore_index, reduction='mean')

