import torch
import torch.nn.functional
from torch.utils.data import Dataset
from data import *


class train_data_F2V(Dataset):
    def __init__(self):
        Fusion_list = get_anchor_audio(0)
        print('train_data', len(Fusion_list))
        self.data = Fusion_list

    def __getitem__(self, item):
        f, a1, a2, label = self.data[item]
        label = int(label)

        a1 = load_audio(a1)
        a1 = torch.from_numpy(a1)
        a1 = torch.unsqueeze(a1, dim=0)

        a2 = load_audio(a2)
        a2 = torch.from_numpy(a2)
        a2 = torch.unsqueeze(a2, dim=0)

        f = TransformToPIL(f)
        f = target_transform_train(f)

        face_m = 0  
        audio_m = 1

        return f, a1, a2, label, face_m, audio_m

    def __len__(self):
        return len(self.data)


class test_data_F2V(Dataset):
    def __init__(self):
        Fusion_list = get_anchor_audio(1)
        print('test_data', len(Fusion_list))
        self.data = Fusion_list

    def __getitem__(self, item):
        f, a1, a2, label = self.data[item]
        label = int(label)

        a1 = load_audio(a1)
        a1 = torch.from_numpy(a1)
        a1 = torch.unsqueeze(a1, dim=0)

        a2 = load_audio(a2)
        a2 = torch.from_numpy(a2)
        a2 = torch.unsqueeze(a2, dim=0)

        f = TransformToPIL(f)
        f = target_transform_train(f)

        return f, a1, a2, label

    def __len__(self):
        return len(self.data)
