import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from mydataset import SegDataset
from model import build_unet
from torch.utils.data import DataLoader
from utils import create_dir, seeding
import albumentations as A
from loss import DiceLoss, DiceBCELoss
from unetplus import Unet_plus_plus
from matplotlib import pyplot as plt


def calculate_metrics(y_true, y_pred):
    """ 
        将y转为ndarray 并拉平 便于后续的计算
    """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ 
        将预测的图片 y_pre 转为ndarray 
    """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    """
        使用sklearn的包来计算 jaccard_score， f1_score， 准确率， 召回率 和 精度
    """
    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

if __name__ == "__main__":
    """ 固定种子 """
    seeding(42)
    model_path = './files/detect.pth'

    """ 打开文件夹 """
    create_dir("results")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = build_unet()     # Unet
    model = Unet_plus_plus()   # Unet++

    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.eval()

    trans_mask = A.Compose([
                A.Resize(height=512, width=512, p=1.0),
                # A.Normalize(),
        ])
    """ 加载图片 """
    valid_dataset = SegDataset("test",trans_mask)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    if len(valid_dataset) == 0:
        print("数据读取失败！")
        exit()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    Dice_bce_lose = 0
    time_taken = []
    loss = DiceBCELoss()
    # loss = DiceLoss()

    for i,(x, y) in tqdm(enumerate(valid_loader),total = len(valid_dataset)):
        x = x.to('cuda:0',dtype=torch.float32)
        y = y.to('cuda:0',dtype=torch.float32)
        with torch.no_grad():
            """ 分割并计算准确率和loss """
            start_time = time.time()
            y_hat = model(x)
            l = loss(y_hat, y)
            Dice_bce_lose += l.item()
            y_hat = torch.sigmoid(y_hat)
            total_time = time.time() - start_time
            time_taken.append(total_time)

        score = calculate_metrics(y, y_hat)
        metrics_score = list(map(add, metrics_score, score))

        
        """ 保存分割结果 """
        # print(x.shape, y.shape, y_hat.shape)
        y_hat = y_hat > 0.3 #二值化
        _, axes = plt.subplots(1, 3,figsize = (15,15))
        axes = axes.flatten()
        axes[0].imshow(np.transpose(x[0].detach().cpu().numpy(),(1,2,0)))
        axes[0].set_title("X")
        axes[1].imshow(np.transpose(y[0].detach().cpu().numpy(),(1,2,0)), cmap=plt.get_cmap('gray'))
        axes[1].set_title("y")
        axes[2].imshow(np.transpose(y_hat[0].detach().cpu().numpy(),(1,2,0)), cmap=plt.get_cmap('gray'))
        axes[2].set_title("y_hat")
        plt.savefig(f'./results/results_{i}.png', bbox_inches='tight')


    jaccard = metrics_score[0]/len(valid_dataset)
    f1 = metrics_score[1]/len(valid_dataset)
    recall = metrics_score[2]/len(valid_dataset)
    precision = metrics_score[3]/len(valid_dataset)
    acc = metrics_score[4]/len(valid_dataset)
    Dice_bce_lose = Dice_bce_lose/len(valid_dataset)
    print(f"Jaccard: {jaccard:1.4f} \nF1: {f1:1.4f} \nRecall: {recall:1.4f} \nPrecision: {precision:1.4f} \nAcc: {acc:1.4f} \nDiceBCEloss:{Dice_bce_lose:1.4f}")

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)

