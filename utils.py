import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes   
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            # torch.nn.L
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input,0,0)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list



# 评估指标

def calculate_miou(input, target, classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''

    inputTmp = torch.zeros([input.shape[0], classNum, input.shape[1], input.shape[2]]).cuda()  
    targetTmp = torch.zeros([target.shape[0], classNum, target.shape[1], target.shape[2]]).cuda()  
    input = input.unsqueeze(1) 
    target = target.unsqueeze(1)  
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1)  
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)  
    batchMious = []  
    mul = inputOht * targetOht 
    for i in range(input.shape[0]):  
        ious = []
        for j in range(classNum):  
            intersection = torch.sum(mul[i][j]) 
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6 #FN+TP+FP
            if union == 1e-6:
                continue
            iou = intersection / union
            ious.append(iou.item())
        miou = np.mean(ious) 
        batchMious.append(miou)
    return np.mean(batchMious)


def calculate_mdice(input, target, classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0], classNum, input.shape[1], input.shape[2]]).cuda()  
    targetTmp = torch.zeros([target.shape[0], classNum, target.shape[1], target.shape[2]]).cuda()  
    input = input.unsqueeze(1) 
    target = target.unsqueeze(1)  
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1)  
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)  
    batchMious = [] 
    mul = inputOht * targetOht 
    for i in range(input.shape[0]):  
        dices = []
        for j in range(classNum): 
            intersection = 2 * torch.sum(mul[i][j]) + 1e-6  # 2TP
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) + 1e-6  #FN+2TP+FP
            dice = intersection / union
            dices.append(dice.item())
        Dice = np.mean(dices) 
        batchMious.append(Dice)
    return np.mean(batchMious)


def Pa(input, target):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    tmp = input == target
    x = torch.sum(tmp).float()
    y = input.nelement()
    # print('x',x,y)
    return (x / y)


def pre(input, target):
    input = input.data.cpu().numpy()
    target = target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    pre = (TP) / (TP + FP + 1e-6)
    return pre


def recall(input, target):
    input = input.data.cpu().numpy()
    target = target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    recall = TP / (TP + FN + 1e-6)
    return recall


def F1score(input, target):
    input = input.data.cpu().numpy()
    target = target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    pre = (TP) / (TP + FP + 1e-6)
    recall = (TP) / (TP + FN + 1e-6)
    F1score = (2 * (pre) * (recall)) / (pre + recall + 1e-6)
    return F1score


def calculate_fwiou(input, target, classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0], classNum, input.shape[1], input.shape[2]]).cuda() 
    targetTmp = torch.zeros([target.shape[0], classNum, target.shape[1], target.shape[2]]).cuda() 
    input = input.unsqueeze(1) 
    target = target.unsqueeze(1) 
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1) 
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)  
    batchFwious = [] 
    mul = inputOht * targetOht 
    for i in range(input.shape[0]): 
        fwious = []
        for j in range(classNum): 
            TP_FN = torch.sum(targetOht[i][j])
            intersection = torch.sum(mul[i][j]) + 1e-6
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            if union == 1e-6:
                continue
            iou = intersection / union
            fwiou = (TP_FN / (input.shape[2] * input.shape[3])) * iou
            fwious.append(fwiou.item())
        fwiou = np.mean(fwious)  
        # print(miou)
        batchFwious.append(fwiou)
    return np.mean(batchFwious)
