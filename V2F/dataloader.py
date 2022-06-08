import torch
import torch.nn.functional
from torch.utils.data import Dataset
from data import *


class train_data_V2F(Dataset):
    def __init__(self):
        Fusion_list = get_anchor_audio(0)
        print('train_data', len(Fusion_list))
        self.data = Fusion_list

    def __getitem__(self, item):
        a, f1, f2, label = self.data[item]
        label = int(label)

        a = load_audio(a)
        a = torch.from_numpy(a)
        a = torch.unsqueeze(a, dim=0)

        f1 = TransformToPIL(f1)
        f1 = target_transform_train(f1)

        f2 = TransformToPIL(f2)
        f2 = target_transform_train(f2)

        face_m = 0  
        audio_m = 1

        return a, f1, f2, label, face_m, audio_m

    def __len__(self):
        return len(self.data)


class test_data_V2F(Dataset):
    def __init__(self):
        Fusion_list = get_anchor_audio(1)
        print('test_data', len(Fusion_list))
        self.data = Fusion_list

    def __getitem__(self, item):
        a, f1, f2, label = self.data[item]
        label = int(label)

        a = load_audio(a)
        a = torch.from_numpy(a)
        a = torch.unsqueeze(a, dim=0)

        f1 = TransformToPIL(f1)
        f1 = target_transform_test(f1)

        f2 = TransformToPIL(f2)
        f2 = target_transform_test(f2)

        return a, f1, f2, label

    def __len__(self):
        return len(self.data)
