import scipy.io as sio
import numpy as np
import os

import torch
def mat2np():
    root = './Inertial'
    new_root = './Inertial_np'

    file_names = os.listdir(root)
    for file in file_names:
        data = sio.loadmat(os.path.join(root, file))['d_iner']
        data = np.resize(data, (300, 6))
        np.save(os.path.join(new_root, file.split('.')[0]+'.npy'), data)




