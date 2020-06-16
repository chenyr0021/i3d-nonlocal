import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from IMU_data_process.imu_network import IMU_Network
from IMU_data_process.imu_dataset import IMU_dataset


def visualize(input, label):
    # acc
    plt.subplot(211)
    frame = np.linspace(1, 300 ,300)
    # [acc/gyro, x/y/z, frame]
    x = input[0,0,:]
    y = input[0,1,:]
    z = input[0,2,:]
    # frame = np.linspace(1,3,3)
    # x = [1,2,3]
    # y = [4,5,6]
    # z = [7,8,9]
    # print(x)
    # print(y)
    # print(z)
    plt.plot(frame, x, frame, y, frame, z)
    plt.show()


def train(batch_size=8):
    dataset = IMU_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    imu_net = IMU_Network()
    imu_net.cuda()
    imu_net.train()
    # imu_net.load_state_dict(torch.load('utd_imu.pt'))
    optimizer = optim.SGD(params=imu_net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for i in range(20):
        loss = 0
        acc = 0
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = imu_net(inputs)
            running_loss = criterion(output, labels)
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()

            _, pred = torch.max(output, dim=1)
            running_acc = (pred==labels).float().mean()
            acc += (pred==labels).sum().item()
            loss += running_loss.item()

            # print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(i+1, running_loss.item(), running_acc.item()))
        print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(i+1, loss, acc/len(dataset)))
    torch.save(imu_net.state_dict(), 'utd_imu.pt')

def test(batch_size=8):
    dataset = IMU_dataset(root='./Inertial_np/test')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    imu_net = IMU_Network()
    imu_net.cuda()
    imu_net.load_state_dict(torch.load('utd_imu.pt'))
    imu_net.eval()

    true_nums = 0
    total_nums = 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        out = imu_net(inputs)
        _, pred = torch.max(out, 1)
        true_nums += (pred==labels).sum().float()
        total_nums += len(labels)

        if total_nums > len(dataset)/4:
            print(pred, '\n', labels)
            # for i in range(len(labels)):
            visualize(inputs[0].cpu().numpy(), labels[0].cpu().numpy())
            break

    print('Test acc: {:.4f}'.format(true_nums/total_nums))

if __name__ == '__main__':
    # train()
    test()

    # # 划分数据集
    # import os
    # import shutil
    # from random import shuffle
    # file_list = os.listdir('./Inertial_np')
    # shuffle(file_list)
    # os.mkdir('./Inertial_np/test')
    # for i in range(len(file_list)//4):
    #     shutil.move(os.path.join('./Inertial_np', file_list[i]), os.path.join('./Inertial_np/test', file_list[i]))