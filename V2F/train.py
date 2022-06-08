from utils import *
import warnings
from model import *

warnings.filterwarnings('ignore')


def load(feature):
    states = torch.load('V2F0_0.5396319886765747.pkl')
    feature.load_state_dict(states['feature'])
    return feature


def train():
    feature = feature_extractor()
    generator = Generator()
    Discri = Dis()
    Cls = Class()
    Rank_loss = lift_struct(1.2, 1, 0.3)
    Loss_Fc = nn.CrossEntropyLoss()
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
    MyDataSet = train_data_V2F()
    dataloader = DataLoader(MyDataSet, batch_size=batch_size, shuffle=False, num_workers=10)  # num_workers=10
    optimizer1 = torch.optim.Adam(
        [
            {"params": feature.parameters(), "lr": 5e-2},
            {"params": generator.parameters(), "lr": 5e-3},
            {"params": Cls.parameters(), "lr": 5e-2}
        ],
    )
    optimizer3 = torch.optim.Adam(Discri.parameters(), lr=5e-3, betas=(0.5, 0.999))
    for epoch in range(100):
        feature.train()
        generator.train()
        Discri.train()
        Cls.train()
        count_train = 0.0  # 
        audio_count = 0.0  # 
        face_count = 0.0  # 
        total_train = 0.0  # 
        for i, data in enumerate(dataloader):
            a, f1, f2, label, face_m, audio_m = data
            a = a.to('cuda')
            f1 = f1.to('cuda')
            f2 = f2.to('cuda')
            label = label.to('cuda')
            total_train += a.size(0)
            face_m = face_m.to('cuda')
            audio_m = audio_m.to('cuda')
            a, f1, f2 = feature(a, f1, f2)
            """a1 = a.cpu()
            a1 = a1.detach().numpy()
            audio_m1 = audio_m.cpu()
            audio_m1 = audio_m1.detach().numpy()
            showPointSingleModal(a1, audio_m1, './tsne/')
            pdb.set_trace()"""
            a, f1, f2 = generator(a, f1, f2)

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
                    p.data.clamp_(-0.01, 0.01)
                out1, out2, out3 = Discri(a, f1, f2)
                loss_d = 2 * Loss_Fc(out1, audio_m) + Loss_Fc(out2, face_m) + Loss_Fc(out3, face_m)
                optimizer3.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer3.step()

            audio_count += label_acc(out1, audio_m)
            face_count += label_acc(out2, face_m) + label_acc(out3, face_m)
            #######################################
            for p1 in Discri.parameters():
                p1.requires_grad = False
            for p2 in feature.parameters():
                p2.requires_grad = True
            for p3 in generator.parameters():
                p3.requires_grad = True
            for p4 in Cls.parameters():
                p4.requires_grad = True

            out1, out2, out3 = Discri(a, f1, f2)
            predict = Cls(a, f1, f2)
            loss1_g = 2 * Loss_Fc(out1, face_m) + Loss_Fc(out2, audio_m) + Loss_Fc(out3, audio_m)
            loss_p = Loss_Fc(predict, label)
            loss_m = compute_metric(a, f1, f2, label, Rank_loss)
            loss_total = loss1_g + 2 * loss_m + 3 * loss_p
            count_train += label_acc(predict, label)
            if i % 10 == 0:
                print(epoch, i, 'G ', loss1_g.item(), ' M ', loss_m.item(), ' C ', loss_p.item(), 'D ', loss_d.item())
                if count_train != 0:
                    print('counts =', count_train)

            optimizer1.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer1.step()

        audio_acc = audio_count / total_train
        face_acc = face_count / (total_train * 2)
        acc = count_train / total_train
        states1 = {
            'feature': feature.state_dict(),
            'G': generator.state_dict(),
            'D': Discri.state_dict(),
            'C': Cls.state_dict(),
        }
        name = 'V2F' + str(epoch) + '_' + str(acc) + '.pkl'
        torch.save(states1, name)
        print('epoch:', epoch, 'V2F training acc :', acc)
        print('Audio acc : ', audio_acc, 'Face acc : ', face_acc)

        # acc_best = eval(feature, generator, Cls, Discri, epoch, acc_best)

    print('training over')


def test():
    feature = feature_extractor()
    cuda = True if torch.cuda.is_available() else False
    feature = load(feature)

    if cuda:
        feature = feature.to('cuda')

    acc_best = eval_CAM(feature)
    print('training over')
    print('acc_best= ', acc_best)


if __name__ == '__main__':
    seed = 25
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    test()
