import torch
import torch.nn as nn


class lift_struct(nn.Module):
    def __init__(self, alpha, multi, margin):
        super(lift_struct, self).__init__()
        self.alpha = alpha
        self.multi = multi
        self.margin = margin
        self.loss = nn.TripletMarginLoss(self.margin)

    def forward(self, anchor, positive, neglist):
        batch = anchor.size(0)
        D_ij = torch.pairwise_distance(anchor, positive)
        D_in = 0
        D_jn = 0
        # loss_tri=0.0
        for i in range(self.multi):
            a = torch.pairwise_distance(anchor, neglist[i])
            D_in += torch.exp(self.alpha - a)
            b = torch.pairwise_distance(positive, neglist[i])
            D_jn += torch.exp(self.alpha - b)
        D_n = D_in + D_jn
        J = torch.log(D_n) + D_ij
        J = torch.clamp(J, min=0)
        loss = J.sum() / (2 * batch)
        return loss
