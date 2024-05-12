import torch
from torch import nn

class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)
        self.use_focal_loss = use_focal_loss
        self.eps = torch.tensor(1e-6).requires_grad_()

    def forward(self, predicts, targets):
        labels, label_lengths = targets
        predicts = predicts.log_softmax(-1).permute(1, 0, 2)
        N, B, C = predicts.shape
        preds_lengths = torch.full(size=(B,), fill_value=N, dtype=torch.long)
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        loss = loss.mean()
        return loss
    
