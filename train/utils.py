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

nyuv2_frq2 = [0.272491,0.568953,0.432069,0.354511,0.82178,
           0.506488,1.133686,0.81217,0.789383,0.380358,
           1.650497,1,0.650831,0.757218,0.950049,
           0.614332,0.483815,1.842002,0.635787,1.176839,
           1.196984,1.111907,1.927519,0.695354,1.057833,
           4.179196,1.571971,0.432408,3.705966,0.549132,
           1.282043,2.329812,0.992398,3.114945,5.466101,
           1.085242,6.968411,1.093939,1.33652,1.228912]

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
    
class CrossEntropyLoss2d_u(nn.Module):
    def __init__(self, mode='SUNRGBD',cfg='mean'):
        super(CrossEntropyLoss2d_u, self).__init__()
        if mode =='SUNRGBD':
            self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(sunrgbd_frq)).float(),
                                           size_average=False, reduce=False,reduction=cfg)
        else:
            self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(nyuv2_frq2)).float(),
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
    
class CrossEntropyLoss2d_u_ALIA(nn.Module):
    def __init__(self, mode='SUNRGBD',cfg='mean'):
        super(CrossEntropyLoss2d_u_ALIA, self).__init__()
        if mode =='SUNRGBD':
            self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(sunrgbd_frq)).float(),
                                           size_average=False, reduce=False,reduction=cfg)
        else:
            self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(nyuv2_frq2)).float(),
                                           size_average=False, reduce=False,reduction=cfg)

    def forward(self, inputs_scales, targets_scales,logits,thresh=0.95):
        losses = []
        # for inputs, targets in zip(inputs_scales, targets_scales):
        thresh_mask = logits.ge(thresh).bool()#高于阈值的是true 其他是false
        mask = targets_scales > 0
        targets_m = targets_scales.clone()
        targets_m[mask] -= 1
        loss_all = self.ce_loss(inputs_scales, targets_m.long())
        losses=torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float())
        losses = losses * thresh_mask
        # total_loss = sum(losses)
        return losses.mean(),thresh_mask.float().mean()
    
class CrossEntropyLoss2d_u_ALIA_ad(nn.Module):
    def __init__(self, mode='SUNRGBD',cfg='mean'):
        super(CrossEntropyLoss2d_u_ALIA_ad, self).__init__()
        if mode =='SUNRGBD':
            self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(sunrgbd_frq)).float(),
                                           size_average=False, reduce=False,reduction=cfg)
        else:
            self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(nyuv2_frq2)).float(),
                                           size_average=False, reduce=False,reduction=cfg)

    def forward(self, inputs_scales, targets_scales,logits,thresh_mask=None):
        losses = []
        # for inputs, targets in zip(inputs_scales, targets_scales):
        # thresh_mask = logits.ge(thresh).bool()#高于阈值的是true 其他是false
        mask = targets_scales > 0
        targets_m = targets_scales.clone()
        targets_m[mask] -= 1
        loss_all = self.ce_loss(inputs_scales, targets_m.long())
        losses=torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float())
        losses = losses * thresh_mask
        # total_loss = sum(losses)
        return losses.mean(),thresh_mask.float().mean()
    
"""
相比于交叉熵损失函数，Focal Loss是一种针对困难样本的一种损失函数。
该函数会降低容易分类的样本的权重，增加难分类样本的权重，从而缓解训练中易出现的类别不平衡问题。
这里采用的Focal Loss是在交叉熵损失基础上进行求解，其中gamma参数表示困难样本的权重增加的因子。
具体而言，首先用交叉熵损失计算当前每个像素的误差，然后计算每个像素对应的pt值，
代表当前预测像素的正确概率，用该概率值来计算Focal Loss。最后根据阈值进行过滤，并返回损失和thresh_mask的均值。
"""

class FocalLoss2d_u_ALIA(nn.Module):
    def __init__(self, mode='SUNRGBD', gamma=2, cfg='mean'):
        super(FocalLoss2d_u_ALIA, self).__init__()
        if mode == 'SUNRGBD':
            self.alpha = torch.from_numpy(np.array(sunrgbd_frq)).float()
        else:
            self.alpha = torch.from_numpy(np.array(nyuv2_frq2)).float()
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(weight=self.alpha,
                                           size_average=False, reduce=False,reduction=cfg)

    def forward(self, inputs_scales, targets_scales, logits, thresh=0.95):
        losses = []
        thresh_mask = logits.ge(thresh).bool() #高于阈值的是true 其他是false
        mask = targets_scales > 0
        targets_m = targets_scales.clone()
        targets_m[mask] -= 1
        ce_loss_all = self.ce_loss(inputs_scales, targets_m.long())
        pt = torch.exp(-ce_loss_all)
        focal_loss_all = ((1 - pt) ** self.gamma) * ce_loss_all

        losses = torch.sum(torch.masked_select(focal_loss_all, mask)) / torch.sum(mask.float())
        losses = losses * thresh_mask.float()
        return losses.mean(), thresh_mask.float().mean()

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


# def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train,miou):
#     # usually this happens only on the start of a epoch
#     epoch_float = epoch + (local_count / num_train)
#     state = {
#         'global_step': global_step,
#         'epoch': epoch_float,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#     }
#     ckpt_model_filename = "best{:0.3f}_ckpt_epoch_{:0.2f}.pth".format(miou,epoch_float)
#     path = os.path.join(ckpt_dir, ckpt_model_filename)
#     torch.save(state, path)
#     print('{:>2} has been successfully saved'.format(path))


# def load_ckpt(model, optimizer, model_file,device):
#     if os.path.isfile(model_file):
#         print("=> loading checkpoint '{}'".format(model_file))
#         if device.type == 'cuda':
#             checkpoint = torch.load(model_file)
#         else:
#             checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
#         # checkpoint = torch.load(model_file)
#         model.load_state_dict(checkpoint['state_dict'])
#         if optimizer:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#         print("=> loaded checkpoint '{}' (epoch {})"
#               .format(model_file, checkpoint['epoch']))
#         step = checkpoint['global_step']
#         epoch = checkpoint['epoch']
#         return step, epoch
#     else:
#         print("=> no checkpoint found at '{}'".format(model_file))
#         os._exit(0)

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
    
def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.long()

def generate_unsup_data(data,depth, target, logits):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_depth = []
    new_target = []
    new_logits = []
    for i in range(batch_size):
        mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        
        #cutmix需要两张  所以+data[i+1]
        new_data.append(
            (
                data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_depth.append(
            (
                depth[i] * mix_mask + depth[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        
        new_target.append(
            (
                target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_logits.append(
            (
                logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )

    new_data, new_depth,new_target, new_logits = (
        torch.cat(new_data),
        torch.cat(new_depth),
        torch.cat(new_target),
        torch.cat(new_logits),
    )
    return new_data, new_depth, new_target.long(), new_logits

def compute_unsupervised_loss_conf_weight(predict, target, percent, pred_teacher):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        conf, ps_label = torch.max(prob, dim=1)
        conf = conf.detach()
        mask = target > 0
        targets_m = target.clone()
        targets_m[mask] -= 1
    
        # conf_thresh = np.percentile(
        #     conf.cpu().numpy().flatten(), 100 - percent
        # )
        # thresh_mask = conf.le(conf_thresh).bool() *mask   #le  小于是true
        # conf[thresh_mask] = 0
        # targets_m[thresh_mask] = 255
        # weight = batch_size * h * w / (torch.sum(targets_m != 255) + 1e-6)
    loss_all = F.cross_entropy(predict, targets_m.long())
    losses=torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float())
    # loss_ = F.cross_entropy(predict, targets_m, reduction='none')  # [10, 321, 321]
    ## v1
    # loss = torch.mean(conf * loss_)
    ## v2
    # conf = conf / conf.sum() * (torch.sum(target != 255) + 1e-6)
    # loss = torch.mean(conf * loss_)
    ## v3
    # conf = (conf + 1.0) / (conf + 1.0).sum() * (torch.sum(targets_m) + 1e-6)
    
    loss = torch.mean((conf >= 0.95) * losses)
    return loss


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)



class ThreshController:
    def __init__(self, nclass, momentum, thresh_init=0.85):

        self.thresh_global = torch.tensor(thresh_init).cuda()
        self.momentum = momentum
        self.nclass = nclass
        

    def new_global_mask_pooling(self, pred, ignore_mask=None):
        return_dict = {}
        # n, c, h, w = pred.shape
        # pred_gather = torch.zeros([n , c, h, w]).cuda()
        
        # pred = pred_gather
        pred = torch.as_tensor(pred).cuda() 
        # print(pred.shape)
        mask_pred = torch.argmax(pred, dim=1)
        pred_softmax = pred.softmax(dim=1)
        pred_conf = pred_softmax.max(dim=1)[0]
        # print(mask_pred.shape)
        unique_cls = torch.unique(mask_pred)
        cls_num = len(unique_cls)
        new_global = 0.0
        for cls in unique_cls:
            cls_map = (mask_pred == cls)
            if cls_map.sum() == 0:
                cls_num -= 1
                continue
            pred_conf_cls_all = pred_conf[cls_map]
            cls_max_conf = pred_conf_cls_all.max()
            new_global += cls_max_conf
        return_dict['new_global'] = new_global / cls_num

        return return_dict

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        thresh = self.new_global_mask_pooling(pred, ignore_mask)
        if update_g:
            self.thresh_global = self.momentum * self.thresh_global + (1 - self.momentum) * thresh['new_global']

    def get_thresh_global(self):
        return self.thresh_global
    
    


def get_adaptive_binary_mask_city(logit): ## VOC COCO
    conf = torch.softmax(logit, dim=1)
    # import ipdb
    # ipdb.set_trace()
    max_value, _ = torch.max(conf.reshape(logit.shape[0], logit.shape[1], -1), dim=2)
    # print("============================",max_value.shape,"====================================")
    # print(max_value)
    new_max = torch.where(max_value > 0.92, max_value * 0.96, max_value)
    thresh = new_max.unsqueeze(-1).unsqueeze(-1)
    # binary_mask = (conf > thresh*0+0.92)
    binary_mask = (conf > thresh)
    result = torch.sum(binary_mask, dim=1)
    return result#, max_value
