import os
import time
import random
import numpy as np
import cv2
from unetplus import Unet_plus_plus
import torch
from matplotlib import pyplot as plt

def seeding(seed):
    """ 固定种子"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dir(path):
    """ 创建路径 """
    if not os.path.exists(path):
        os.makedirs(path)


def epoch_time(start_time, end_time):
    """ 计算时间xx分钟，xx秒 """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
