from torch.utils.data import DataLoader
from dataloader import *
from metric import *


def adjust_lr(optimizer, epoch):
    if epoch < 50 and (epoch + 1) % 10 == 0:
        for p in optimizer.param_groups:
            p['lr'] = p['lr'] * 0.1


"""

"""


def compute_metric(f, a1, a2, label, loss_fuc):
    label = label.cpu().numpy()
    mod0 = np.where(label == 0)
    mod1 = np.where(label == 1)
    f_0 = f[mod0[0]]
    f_1 = f[mod1[0]]
    a1_0 = a1[mod0[0]]
    a1_1 = a1[mod1[0]]
    a2_0 = a2[mod0[0]]
    a2_1 = a2[mod1[0]]
    n_0 = []
    n_1 = []
    n_0.append(a2_0)
    n_1.append(a1_1)
    loss = loss_fuc(f_0, a1_0, n_0) + loss_fuc(f_1, a2_1, n_1)
    return loss


"""

"""


def distance_acc(anchor, positive, n_list):
    num = len(n_list)
    count = 0
    d_p = torch.pairwise_distance(anchor, positive)
    d_n = []
    for i in range(num):
        a = torch.pairwise_distance(anchor, n_list[i])
        d_n.append(a)
    for j in range(anchor.size(0)):
        count_ = 0
        for k in range(num):
            if d_p[j] < d_n[k][j]:
                count_ += 1
        if count_ == num:
            count += 1
    return count


"""

"""


def label_acc(out, label):
    label = label.to('cuda')
    _, predicts = torch.max(out.data, 1)
    correct = (predicts == label).sum().item()
    return correct


"""

"""


def eval(feature_extractor, generator, Cls, Discri, epoch, acc_best):
    ismydataset = test_data_F2V()
    valdataloader = DataLoader(ismydataset, batch_size=50, shuffle=False, num_workers=10)
    feature_extractor.eval()
    Cls.eval()
    total_test = 0.0
    count_test = 0.0
    for index, data in enumerate(valdataloader):
        f, a1, a2, label = data
        f, a1, a2 = f.to('cuda'), a1.to('cuda'), a2.to('cuda')
        label = label.to('cuda')
        total_test += f.size(0)
        f, a1, a2 = feature_extractor(f, a1, a2)
        f, a1, a2 = generator(f, a1, a2)
        predict = Cls(f, a1, a2)
        count_test += label_acc(predict, label)
    acc2 = count_test / total_test
    if acc2 > acc_best:
        acc_best = acc2
        states1 = {
            'feature': feature_extractor.state_dict(),
            'G': generator.state_dict(),
            'D': Discri.state_dict(),
            'C': Cls.state_dict(),
        }
        name = 'F2V' + str(epoch) + '_' + str(acc2) + '.pkl'
        torch.save(states1, name)
    print('F2V test acc : ', acc2, 'best acc', acc_best)
    return acc_best
