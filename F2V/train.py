from utils import *
import warnings
from model import *

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np

"""
加载之前的模型，特征提取，生成器，鉴别器，分类器
"""


def load(feature, D, G, C):
    # states = torch.load('V2F11_0.8845315904139434.pkl')#epoch=16,5e-3,5e-4(0.5,0.999)
    # states = torch.load('V2F5_0.9132897603485839.pkl')#epoch=16 after,5e-4,5e-5(0,0.999)
    # states = torch.load('V2F5_0.9254901960784314.pkl')#epoch=21 after,5e-5,5e-6(0,0.999)
    states = torch.load('V2F16_0.9193899782135077.pkl')  # epoch=16 after,1e-2,5e-3(0.5,0.999)  #state=5e-2,5e-1,5e-3
    feature.load_state_dict(states['feature'])
    G.load_state_dict(states['G'])
    D.load_state_dict(states['D'])
    C.load_state_dict(states['C'])
    return feature, D, G, C


def adjust_lr(optimizer, optimizer3, epoch):
    if 1 < epoch < 50 and epoch % 10 == 0:
        for p1 in optimizer.param_groups:
            p1['lr'] = p1['lr'] * 0.1
        for p3 in optimizer3.param_groups:
            p3['lr'] = p3['lr'] * 0.1


def train():
    feature = feature_extractor()
    generator = Generator()
    Discri = Dis()
    Cls = Class()
    Rank_loss = lift_struct(1.2, 1, 0.3)
    # Rank_loss = lift_struct(1.2,1)  #
    # Rank_loss = re_triplet(0.3)
    Loss_Fc = nn.CrossEntropyLoss()  # 
    cuda = True if torch.cuda.is_available() else False
    # feature, Discri, generator, Cls = load(feature, Discri, generator, Cls)
    if cuda:
        feature = feature.to('cuda')
        generator = generator.to('cuda')
        Discri = Discri.to('cuda')
        Cls = Cls.to('cuda')
        Rank_loss = Rank_loss.to('cuda')
        Loss_Fc = Loss_Fc.to('cuda')
    batch_size = 50
    acc_best = 0
    MyDataSet = train_data_F2V()
    dataloader = DataLoader(MyDataSet, batch_size=batch_size, shuffle=False, num_workers=10)  # num_workers=10
    optimizer1 = torch.optim.Adam(
        [
            {"params": feature.parameters(), "lr": 5e-2},
            {"params": generator.parameters(), "lr": 5e-3},
            {"params": Cls.parameters(), "lr": 5e-2}
        ],
    )
    optimizer3 = torch.optim.Adam(Discri.parameters(), lr=5e-3, betas=(0.5, 0.999))
    for epoch in range(50):
        adjust_lr(optimizer1, optimizer3, epoch)
        feature.train()
        generator.train()
        Discri.train()
        Cls.train()
        count_train = 0.0  # 
        audio_count = 0.0  # 
        face_count = 0.0  # 
        total_train = 0.0  # 
        for i, data in enumerate(dataloader):
            f, a1, a2, label, face_m, audio_m = data
            f = f.to('cuda')
            a1 = a1.to('cuda')
            a2 = a2.to('cuda')
            label = label.to('cuda')
            total_train += f.size(0)
            face_m = face_m.to('cuda')
            audio_m = audio_m.to('cuda')
            f, a1, a2 = feature(f, a1, a2)
            f, a1, a2 = generator(f, a1, a2)
            #######################################
            for p1 in Discri.parameters():
                p1.requires_grad = True
            for p2 in feature.parameters():
                p2.requires_grad = False
            for p3 in generator.parameters():
                p3.requires_grad = False
            for p4 in Cls.parameters():
                p4.requires_grad = False

            for k in range(5):
                for p in Discri.parameters():
                    p.data.clamp_(-0.01, 0.01)  #
                out1, out2, out3 = Discri(f, a1, a2)
                loss_d = 2 * Loss_Fc(out1, face_m) + Loss_Fc(out2, audio_m) + Loss_Fc(out3, audio_m)
                optimizer3.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer3.step()

            face_count += label_acc(out1, face_m)
            audio_count += label_acc(out2, audio_count) + label_acc(out3, audio_count)
            #######################################
            for p1 in Discri.parameters():
                p1.requires_grad = False
            for p2 in feature.parameters():
                p2.requires_grad = True
            for p3 in generator.parameters():
                p3.requires_grad = True
            for p4 in Cls.parameters():
                p4.requires_grad = True

            out1, out2, out3 = Discri(f, a1, a2)
            predict = Cls(f, a1, a2)
            loss1_g = 2 * Loss_Fc(out1, audio_m) + Loss_Fc(out2, face_m) + Loss_Fc(out3, face_m)
            loss_p = Loss_Fc(predict, label)
            loss_m = compute_metric(f, a1, a2, label, Rank_loss)
            loss_total = loss1_g + 2 * loss_m + 3 * loss_p
            count_train += label_acc(predict, label)
            if i % 10 == 0:
                print(epoch, i, 'G ', loss1_g.item(), ' M ', loss_m.item(), ' C ', loss_p.item(), 'D ', loss_d.item())
                if count_train != 0:
                    print('counts =', count_train)

            optimizer1.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer1.step()
            # for p3 in generator.parameters():
            #    torch.nn.utils.clip_grad_norm(p3.data, 0.02)
            # optimizer1.step()

        audio_acc = audio_count / (total_train * 2)
        face_acc = face_count / total_train
        acc = count_train / total_train
        print('epoch:', epoch, 'F2V training acc :', acc)
        print('Audio acc : ', audio_acc, 'Face acc : ', face_acc)
        acc_best = eval(feature, generator, Cls, Discri, epoch, acc_best)

    print('training over')
    # print('acc_best= ', acc_best)


if __name__ == '__main__':
    seed = 25
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train()
    # eval()
