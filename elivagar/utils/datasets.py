import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
from torch.utils.data import Dataset
import torch
from torch import Tensor
from sklearn.preprocessing import StandardScaler, MinMaxScaler




def load_dataset(name, embedding_type='angle', num_reps=1,size=16):
    x_train=np.load(f'/mnt/gs21/scratch/jinhongn/elivagar/PA/experiment_data/{name}/x_train.npy')
    x_test=np.load(f'/mnt/gs21/scratch/jinhongn/elivagar/PA/experiment_data/{name}/x_test.npy')
    y_train=np.load(f'/mnt/gs21/scratch/jinhongn/elivagar/PA/experiment_data/{name}/y_train.npy')
    y_test=np.load(f'/mnt/gs21/scratch/jinhongn/elivagar/PA/experiment_data/{name}/y_test.npy')

    min_max_scaler=MinMaxScaler(feature_range=(0, np.pi))
    x_train=min_max_scaler.fit_transform(x_train[:, :int(size)])
    x_test=min_max_scaler.transform(x_test[:, :int(size)])

    min_max_scaler_y=MinMaxScaler(feature_range=(0,1))
    y_train=min_max_scaler_y.fit_transform(y_train.reshape(-1,1))
    y_test=min_max_scaler_y.transform(y_test.reshape(-1,1))
    
    
    if embedding_type == 'angle':
        pass

    elif embedding_type == 'iqp':
        p_1 = np.prod(x_train[:, :], 1).reshape((len(x_train), 1))
        p_1 = np.concatenate([p_1 for i in range((len(x_train[0]) * (len(x_train[0]) - 1)) // 2)], 1)

        x_train = np.concatenate((x_train, p_1), 1)

    elif embedding_type == 'supernet':
        p_1 = np.prod(np.pi - x_train[:, :], 1).reshape((len(x_train), 1))
        p_1 = np.concatenate([p_1 for i in range(len(x_train[0]) - 1)], 1)

        x_train = np.concatenate((x_train, p_1), 1)
    else:
        print('Dataset not supported!')
    print(x_train.shape)
    print(x_train.min(),x_train.max(),x_test.min(),x_test.max())
    print(y_train.min(),y_train.max(),y_test.min(),y_test.max())
    return Tensor(x_train), Tensor(y_train), Tensor(x_test), Tensor(y_test),min_max_scaler_y
        
