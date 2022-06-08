import torch
import torch.nn as nn


def euclidean_dist(x, y, split=0):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
      split: When the CUDA memory is not sufficient, we can split the dataset into different parts
             for the computing of distance.
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    if split == 0:
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        distmat = xx + yy
        distmat.addmm_(x, y.t(), beta=1, alpha=-2)

    else:
        distmat = x.new(m, n)
        start = 0
        x = x.cuda()

        while start < n:
            end = start + split if (start + split) < n else n
            num = end - start

            sub_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, num) + \
                          torch.pow(y[start:end].cuda(), 2).sum(dim=1, keepdim=True).expand(num, m).t()
            # sub_distmat.addmm_(1, -2, x, y[start:end].t())
            sub_distmat.addmm_(x, y[start:end].cuda().t(), beta=1, alpha=-2)
            distmat[:, start:end] = sub_distmat.cpu()
            start += num

    distmat = distmat.clamp(min=1e-12).sqrt()  # for numerical stability
    return torch.diag(distmat)


"""class RankList(nn.Module):  # without penalty on negative samples
    def __init__(self, alpha, m, lamda, multi):
        super(RankList, self).__init__()

        self.alpha = alpha
        self.m = m
        # self.T = T
        self.lamda = lamda
        self.multi = multi  # numbers of negative samples

    def forward(self, anchor, positive, neglist):
        # L_mp= torch.norm(anchor-positive,0) #Euclidean distance
        batch = anchor.size(0)
        L_mp = torch.pairwise_distance(anchor, positive)
        L_p = torch.clamp(L_mp - (self.alpha - self.m), min=0)
        L_n = 0
        for i in range(self.multi):
            L_mn = torch.pairwise_distance(anchor, neglist[i])
            L_n += torch.clamp(self.alpha - L_mn, min=0)

        # w1 = torch.zeros(batch)
        # w2 = torch.zeros(batch)
        # for x in range(batch):
        #     w1[x] = torch.exp(self.T*(self.alpha-L_mn1[x]))
        #     w2[x] = torch.exp(self.T*(self.alpha-L_mn2[x]))

        loss = (L_p + self.lamda * L_n) / (self.multi + 1)
        return loss.sum() / batch"""

"""
直接拿的别人的代码，可以对照论文里的公式来看
"""
"""
class re_triplet(nn.Module):
    def __init__(self, margin):
        super(re_triplet, self).__init__()
        self.margin = margin
        #self.loss = nn.TripletMarginLoss(self.margin)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, anchor, positive, n_list):
        
        loss_2 = 0.0
        dist_ap=euclidean_dist(anchor,positive)
        y = dist_ap.new().resize_as_(dist_ap).fill_(1)
        dist_ap_size=dist_ap.size(0)
        for i in range(len(n_list)):
            dist_an=euclidean_dist(anchor,n_list[i])
            loss= self.ranking_loss(dist_an, dist_ap, y)
            loss_2+=loss+1/(1+ torch.exp(4*(sum(dist_an-max(dist_ap))/dist_ap_size)))
        loss_2 = loss_2 / len(n_list)

        return loss_2
"""


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
            # loss_tri += self.loss(anchor, positive, neglist[i])
        D_n = D_in + D_jn
        J = torch.log(D_n) + D_ij
        J = torch.clamp(J, min=0)
        loss = J.sum() / (2 * batch)
        return loss


"""
class re_triplet(nn.Module):
    def __init__(self, margin):
        super(re_triplet, self).__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(self.margin)

    def forward(self, anchor, positive, n_list):
        batch = anchor.size(0)
        loss = 0.0
        loss_pn = 0.0
        for i in range(len(n_list)):
            loss += self.loss(anchor, positive, n_list[i])
            loss_pn += torch.pairwise_distance(n_list[i],positive)
        loss2 = loss / len(n_list)+sum(torch.clamp(loss_pn, min=0))/(2 * batch)
            
        #loss = loss / len(n_list)
        return loss2
"""
"""
if __name__ == '__main__':
    L = lift_struct(1.0, 1)
    # L = RankList(1.2,0.4,1,2)
    # L = n_pair(2)
    anchor = torch.randn(64, 128)
    positive = torch.randn(64, 128)
    negative1 = torch.randn(64, 128)
    # negative2 = torch.randn(64,128)
    neglist = []
    # neglist.append(negative1)
    # neglist.append(negative2)
    loss = L(anchor, positive, negative1)
    print(loss)
    # print(loss)
    # count = distance_acc(anchor,positive,neglist)
    # print(count)
"""
