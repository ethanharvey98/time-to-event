import torch
import torch.nn as nn

class MSEUncensored(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, durations, labels):
        logits, durations, labels = logits.float().view(-1), durations.float().view(-1), labels.float().view(-1)
        squared_errors = (durations-logits)**2
        error = squared_errors**labels*torch.zeros_like(squared_errors)**(1-labels)
        return torch.mean(error)

class MSEHinge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, durations, labels):
        logits, durations, labels = logits.float().view(-1), durations.float().view(-1), labels.float().view(-1)
        squared_errors = (durations-logits)**2
        error = squared_errors**labels*((durations>=logits).int()*squared_errors)**(1-labels)
        return torch.mean(error)