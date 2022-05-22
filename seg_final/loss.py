import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
        实现Dice 损失函数
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #如果在模型中输出前使用过了sigmoid函数，这里就可以不用sigmoid
        inputs = torch.sigmoid(inputs)

        #拉平 tensor
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    """
        Bce + Diceloss 混合损失
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #如果在模型中输出前使用过了sigmoid函数，这里就可以不用sigmoid
        inputs = torch.sigmoid(inputs)

        #拉平 tensor
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
