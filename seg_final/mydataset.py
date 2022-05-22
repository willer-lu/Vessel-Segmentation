import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from glob import glob
from PIL import Image
import imageio
from torch.utils.data import Dataset
"""
    创建加载数据用的DataSet  + 数据增强
"""

class SegDataset(Dataset):
    def __init__(self, data_path, trans = None):
        super().__init__()

        self.X = sorted(glob(os.path.join(data_path,"images",  "*.tif")))
        self.y = sorted(glob(os.path.join(data_path, "1st_manual", "*.gif")))
        self.trans = trans

    def __getitem__(self, index):
        """ 读取image """
        image = cv2.imread(self.X[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        """ 读取mask """

        mask = Image.open(self.y[index])
        mask = np.array(mask)
        
        if self.trans is not None:
            t = self.trans(image = image, mask = mask)
            image = t['image']
            mask = t['mask']

        return transforms.ToTensor()(image),transforms.ToTensor()(mask)

    def __len__(self):
        return len(self.X)

