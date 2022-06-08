import torchvision.transforms as transforms
import librosa
import numpy as np
from PIL import Image

"""

"""


def target_transform_train(PILImg):
    transf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transf(PILImg)


"""

"""


def target_transform_test(PILImg):
    transf = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transf(PILImg)


"""

"""


def TransformToPIL(PILimage):
    RealPILimage = Image.open(PILimage).convert("RGB")
    RealPILimage = RealPILimage.resize((224, 224))
    return RealPILimage


"""

"""


def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


"""
"""


def load_audio(audiopath):
    y, sr = librosa.load(audiopath)  
    y = y - y.mean()
    y = preemphasis(y)
    y = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    """

    """
    a = np.max(y)
    b = np.min(y)
    k = 2 / (a - b)
    y = -1 + (y - b) * k
    y = np.resize(y, [224, 125])
    # print(np.max(y),np.min(y))
    return y


"""

"""


def get_anchor_audio(flag):
    FusionList_anchor_audio = []
    if flag == 0:
        with open('./data/train_F2V_vox1_label50.txt', 'r') as f:
            for line in f:
                str = line.split()
                FusionList_anchor_audio.append((str[0],) + (str[1],) + (str[2],) + (str[3],)) 
    elif flag == 1:
        with open('./data/test_F2V_vox1_label50.txt', 'r') as f:
            for line in f:
                str = line.split()
                FusionList_anchor_audio.append((str[0],) + (str[1],) + (str[2],) + (str[3],))

    return FusionList_anchor_audio
