import cv2
from matplotlib import pyplot as plt
from sklearn import manifold
from torch.utils.data import DataLoader
from dataloader import *
from metric import *
import torch.nn.functional as F
from model import *
import os.path as osp


def showPointSingleModal(features, label, save_path):
    # label = self.relabel(label)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    features_tsne = tsne.fit_transform(features)
    COLORS = ['darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black']
    MARKS = ['x', 'o', '+', '^', 's']
    features_min, features_max = features_tsne.min(0), features_tsne.max(0)
    features_norm = (features_tsne - features_min) / (features_max - features_min)
    plt.figure(figsize=(20, 20))
    for i in range(features_norm.shape[0]):
        plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[label[i] % 6],
                    marker=MARKS[label[i] % 5])
    plt.savefig(save_path)
    plt.show()
    plt.close()


def adjust_lr(optimizer, epoch):
    if epoch < 50 and (epoch + 1) % 10 == 0:
        for p in optimizer.param_groups:
            p['lr'] = p['lr'] * 0.1


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


def label_acc(out, label):
    label = label.to('cuda')
    _, predicts = torch.max(out.data, 1)
    correct = (predicts == label).sum().item()
    return correct


def eval(feature, generator, Cls, Discri, epoch, acc_best):
    ismydataset = test_data_V2F()
    valdataloader = DataLoader(ismydataset, batch_size=50, shuffle=False, num_workers=10)
    feature.eval()
    Cls.eval()
    """model_dict = dict(type='feature_img', arch=feature_img, layer_name='frame', input_size=(224, 224))
    gradcam = GradACM(model_dict)"""

    total_test = 0.0
    count_test = 0.0
    for index, data in enumerate(valdataloader):
        a, f1, f2, label = data
        a, f1, f2 = a.to('cuda'), f1.to('cuda'), f2.to('cuda')
        label = label.to('cuda')
        total_test += a.size(0)
        a, f1, f2 = feature(a, f1, f2)

        """mask, logit = gradcam(f1, class_idx=10)
        heatmap, cam_result = visualize_cam(mask, f1)"""

        a, f1, f2 = generator(a, f1, f2)
        predict = Cls(a, f1, f2)
        count_test += label_acc(predict, label)
    acc2 = count_test / total_test
    if acc2 > acc_best:
        acc_best = acc2
        states1 = {
            'feature_img': feature.state_dict(),
            'G': generator.state_dict(),
            'D': Discri.state_dict(),
            'C': Cls.state_dict(),
        }
        name = 'V2F' + str(epoch) + '_' + str(acc2) + '.pkl'
        torch.save(states1, name)
    print('V2F test acc : ', acc2, 'best acc', acc_best)
    return acc_best


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(
        model,
        test_loader,
        save_dir,
        width,
        height,
        use_gpu,
        img_mean=None,
        img_std=None
):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    batch = 0
    for batch_idx, data in enumerate(test_loader):
        batch = batch + 1
        audio, imgs, imgs2 = data[0], data[1], data[2]
        if use_gpu:
            audio = audio.cuda()
            imgs = imgs.cuda()
            imgs2 = imgs2.cuda()

        # forward to get convolutional feature maps
        try:
            outputs = model(audio, imgs, imgs2, return_featuremaps=True)
            # outputs = model(imgs, return_featuremaps=True)
        except TypeError:
            raise TypeError(
                'forward() got unexpected keyword argument "return_featuremaps". '
                'Please add return_featuremaps as an input argument to forward(). When '
                'return_featuremaps=True, return feature maps only.'
            )

        if outputs.dim() != 4:
            raise ValueError(
                'The model output is supposed to have '
                'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                'Please make sure you set the model output at eval mode '
                'to be the last convolutional feature maps'.format(
                    outputs.dim()
                )
            )

        # compute activation maps
        outputs = (outputs ** 2).sum(1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h * w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)

        if use_gpu:
            imgs, outputs = imgs.cpu(), outputs.cpu()

        for j in range(outputs.size(0)):

            imnameWri = str(batch) + '_' + str(j)
            # RGB image
            img = imgs[j, ...]
            for t, m, s in zip(img, img_mean, img_std):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.numpy() * 255))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            # activation map
            am = outputs[j, ...].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (
                    np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # overlapped
            overlapped = img_np * 0.3 + am * 0.7
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            # save images in a single figure (add white spacing between images)
            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones(
                (height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8
            )
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:,
            width + GRID_SPACING:2 * width + GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
            cv2.imwrite(osp.join(r"E:\visCAM_train\feature", imnameWri + '.jpg'), grid_img)
            # print(osp.join(r"E:\visCAM", imnameWri + '.jpg'))

        if (batch_idx + 1) % 10 == 0:
            print(
                '- done batch {}/{}'.format(
                    batch_idx + 1, len(test_loader)
                )
            )


def eval_CAM(feature):
    train_data = train_data_V2F()
    data_loader = DataLoader(train_data, batch_size=50, shuffle=False, num_workers=10)
    feature.eval()

    visactmap(
        feature, data_loader, r"E:\袁帆\code", 224, 224, use_gpu=True
    )

    raise RuntimeError
