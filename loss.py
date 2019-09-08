from torch import nn
import torch
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        # self.loss=nn.NLLLoss2d(weight)
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.loss = nn.NLLLoss(weight)

    def forward(self, input, targets):
        softmax_input = F.softmax(input, dim=1)
        log_softmax_input = torch.log(softmax_input)
        return self.loss(torch.pow(1-softmax_input, self.gamma)*log_softmax_input, targets)