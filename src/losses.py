import torch
import torch.nn as nn

class MSEUncensored(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, durations, labels):
        logits, durations, labels = logits.float().view(-1), durations.float().view(-1), labels.float().view(-1)
        mses = (durations-logits)**2
        error = mses**labels*torch.zeros_like(mse)**(1-labels)
        return torch.mean(error)

class MSEHinge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, durations, labels):
        logits, durations, labels = logits.float().view(-1), durations.float().view(-1), labels.float().view(-1)
        mses = (durations-logits)**2
        error = mses**labels*((durations>=logits).int()*mses)**(1-labels)
        return torch.mean(error)