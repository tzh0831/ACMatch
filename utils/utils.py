import numpy as np
from torch import nn
import torch
import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable

sunrgbd_frq = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
           0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
           2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
           0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
           1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
           4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
           3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
           0.750738, 4.040773]

nyuv2_frq = [0.04636878, 0.10907704, 0.152566  , 0.28470833, 0.29572534,
       0.42489686, 0.49606689, 0.49985867, 0.45401091, 0.52183679,
       0.50204292, 0.74834397, 0.6397011 , 1.00739467, 0.80728748,
       1.01140891, 1.09866549, 1.25703345, 0.9408835 , 1.56565388,
       1.19434108, 0.69079067, 1.86669642, 1.908     , 1.80942453,
       2.72492965, 3.00060817, 2.47616595, 2.44053651, 3.80659652,
       3.31090131, 3.9340523 , 3.53262803, 4.14408881, 3.71099056,
       4.61082739, 4.78020462, 0.44061509, 0.53504894, 0.21667766]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

label_colours = [(0, 0, 0),
                 # 0=background
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),
                 (139, 110, 246)]


class CrossEntropyLoss2d_eval(nn.Module):
    def __init__(self, weight=sunrgbd_frq):
        super(CrossEntropyLoss2d_eval, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        inputs = inputs_scales
        targets = targets_scales
        # for inputs, targets in zip(inputs_scales, targets_scales):
        mask = targets > 0
        targets_m = targets.clone()
        targets_m[mask] -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=sunrgbd_frq,cfg='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           size_average=False, reduce=False,reduction=cfg)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss
    
# class CrossEntropyLoss2d_u(nn.Module):
#     def __init__(self, weight=sunrgbd_frq,cfg='mean'):
#         super(CrossEntropyLoss2d_u, self).__init__()
#         self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
#                                            size_average=False, reduce=False,reduction=cfg)

#     def forward(self, inputs_scales, targets_scales):
#         losses = []
#         # for inputs, targets in zip(inputs_scales, targets_scales):
#         mask = targets_scales > 0
#         targets_m = targets_scales.clone()
#         targets_m[mask] -= 1
#         loss_all = self.ce_loss(inputs_scales, targets_m.long())
#         losses=torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float())
#         # total_loss = sum(losses)
#         return losses
class CrossEntropyLoss2d_u(nn.Module):
    def __init__(self, mode='SUNRGBD',cfg='mean'):
        super(CrossEntropyLoss2d_u, self).__init__()
        if mode =='SUNRGBD':
            self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(sunrgbd_frq)).float(),
                                           size_average=False, reduce=False,reduction=cfg)
        else:
            self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(nyuv2_frq)).float(),
                                           size_average=False, reduce=False,reduction=cfg)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        # for inputs, targets in zip(inputs_scales, targets_scales):
        mask = targets_scales > 0
        targets_m = targets_scales.clone()
        targets_m[mask] -= 1
        loss_all = self.ce_loss(inputs_scales, targets_m.long())
        losses=torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float())
        # total_loss = sum(losses)
        return losses
# hxx add, focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                                 size_average=self.size_average, reduce=False)

    def forward(self, input, target):
        return self.loss((1 - F.softmax(input, 1))**2 * F.log_softmax(input, 1), target)


class FocalLoss2d(nn.Module):
    def __init__(self, weight=sunrgbd_frq, gamma=0):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.fl_loss = FocalLoss(gamma=self.gamma, weight=self.weight, size_average=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.fl_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss


def color_label_eval(label):
    # label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    # return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    return colored.transpose([0, 2, 1])

def color_label(label):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.data, time_inter, count_inter))


def save_ckpt(cfg,ckpt_dir, model, optimizer, global_step, best_epoch,best_miou):
    # usually this happens only on the start of a epoch
    # epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'best_miou':best_miou,
        'epoch': best_epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
   
    path = os.path.join(ckpt_dir, '%s_epoch%d_%.3f.pth' % (cfg['backbone'], best_epoch,best_miou))
    torch.save(state, path)
 


def load_ckpt(model, optimizer, model_file,device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        # checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        best_miou = checkpoint['best_miou']
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return best_miou,step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)


# added by hxx for iou calculation
def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1 # hxx
    # imLab += 1 # label 应该是不用加的
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def accuracy(preds, label):
    valid = (label > 0) # hxx
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def macc(preds, label, num_class):
    a = np.zeros(num_class)
    b = np.zeros(num_class)
    for i in range(num_class):
        mask = (label == i+1)
        a_sum = (mask * preds == i+1).sum()
        b_sum = mask.sum()
        a[i] = a_sum
        b[i] = b_sum
    return a,b

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg