import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.trans = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(51200, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Tanh()
        )

    def forward(self, in1, in2, in3):
        in1 = in1.view(in1.size(0), -1)
        in2 = in2.view(in2.size(0), -1)
        in3 = in3.view(in3.size(0), -1)
        return self.trans(in1), self.trans(in2), self.trans(in3)


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(128, 2),
        )

    def forward(self, in1, in2, in3):
        in1 = self.trans(in1)
        in2 = self.trans(in2)
        in3 = self.trans(in3)
        return in1, in2, in3


class Class(nn.Module):
    def __init__(self):
        super(Class, self).__init__()
        self.trans = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128 * 3, 2),
            nn.BatchNorm1d(2)
        )

    def forward(self, in1, in2, in3):
        out = torch.cat([in1, in2, in3], dim=1)
        return self.trans(out)


class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        self.audio = nn.Sequential(
            nn.Conv2d(1, 3, 5, (2, 1), 1),
            # nn.Dropout2d(0.4),
            nn.AvgPool2d(kernel_size=(3, 5), stride=2, ceil_mode=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
            nn.Conv2d(3, 64, (3, 6), 2, 1),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=False),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),

            nn.Conv2d(64, 128, 3, 1, 1),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.AvgPool2d(kernel_size=(4, 5), stride=1, ceil_mode=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
        )
        self.frame = nn.Sequential(
            nn.Conv2d(3, 3, 7, 2, 3),
            # nn.Dropout2d(0.4),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
            # nn.Dropout2d(0.2),
            nn.Conv2d(3, 32, 5, 2, 2),
            # nn.Dropout2d(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),

            nn.Conv2d(32, 64, 2, 1, 4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
        )

    def forward(self, v1, f1, f2, return_featuremaps=False):
        # print("v1", v1.shape)
        v1 = self.audio(v1)
        # print("v1", v1.shape)
        f1 = self.frame(f1)
        # print("f1", f1.shape)
        f2 = self.frame(f2)
        if return_featuremaps:
            return f1

        return v1, f1, f2
