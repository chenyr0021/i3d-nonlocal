import scipy.io as sio
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
def mat2np():
    root = './Inertial'
    new_root = './Inertial_np'

    file_names = os.listdir(root)
    for file in file_names:
        data = sio.loadmat(os.path.join(root, file))['d_iner']
        data = np.resize(data, (300, 6))
        np.save(os.path.join(new_root, file.split('.')[0]+'.npy'), data)


class IMU_dataset(Dataset):
    def __init__(self, root='./Inertial_np/train'):
        super(IMU_dataset, self).__init__()
        self.path = []
        self.label = []
        for file in os.listdir(root):
            self.path.append(os.path.join(root, file))
            self.label.append(int(file.split('_')[0][1:])-1)

    def __getitem__(self, item):
        file_name = self.path[item]
        data = torch.from_numpy(np.load(file_name)).float()
        data = data.reshape((300, 3, 2))
        data = data.transpose(0, 2)
        label = self.label[item]
        return data, label

    def __len__(self):
        return len(self.label)



if __name__ == '__main__':

    dataset = IMU_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for data, label in dataloader:
        print(data, label)
        break