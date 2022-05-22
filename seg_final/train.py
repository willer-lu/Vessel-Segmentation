import os
import time
from glob import glob
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A
from mydataset import SegDataset
from model import build_unet
from unetplus import Unet_plus_plus
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
from albumentations.augmentations.transforms import Normalize

# 可以改的地方或者说可以调参的地方和第一个差不多
# 然后一些要求或者可以写的地方(比如画图表或者什么迭代 各种实验)和第一个一样

def train(model, loader, optimizer, loss, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()
        epoch_loss += l.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            l = loss(y_pred, y)
            epoch_loss += l.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    #固定随机种子
    seeding(42)

    """ 保存的路径 """
    create_dir("files")

    """ 超参数 """
    batch_size = 2
    num_epochs = 5
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ 加载 Dataloader """
    trans = A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise(p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
                A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
                A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            # 随机应用仿射变换：平移，缩放和旋转输入
            A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
            A.Normalize(),
    ])
    trans_mask = A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            A.Normalize(),
    ])

    train_dataset = SegDataset("../train",trans)
    valid_dataset = SegDataset("../test",trans_mask)
    print(f"Dataset Size:\nTrain: {len(train_dataset)} - Valid: {len(valid_dataset)}")
    if len(train_dataset) == 0:
        print("数据读取失败！")
        exit()
    

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用Unet
    # model = build_unet().to(device)
    # 使用Unet++
    model = Unet_plus_plus().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20)

    loss_fn = DiceBCELoss()

    """ 训练 """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step()

        """ 保存模型 """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
