import sys
sys.path.append("..")
import argparse
from itertools import cycle
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml
from train.utils import load_ckpt, intersectionAndUnion, AverageMeter, accuracy, macc,CrossEntropyLoss2d,CrossEntropyLoss2d_u,save_ckpt
from copy import deepcopy
# import torchdatasets as td
from semi import SemiDataset,RGBD_Dataset
# from model.semseg.deeplabv3plus import DeepLabV3Plus
from ACNet_models_V1 import ACNet
from util.utils import count_params, init_log,set_random_seed,seed_worker
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter





def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    a_meter = AverageMeter()
    b_meter = AverageMeter()

    with torch.no_grad():
        for idx,data in enumerate(loader):
            img = data[0].cuda()
            depth = data[1].cuda()
            mask =data[2].numpy()
            # mask = mask.cuda()
            with torch.no_grad():
                pred = model(img, depth)

            output = torch.max(pred, 1)[1] + 1
            output = output.squeeze(0).cpu().numpy()

            acc, pix = accuracy(output, mask)
            intersection, union = intersectionAndUnion(output, mask, cfg['nclass'])
            acc_meter.update(acc, pix)
            a_m, b_m = macc(output, mask, cfg['nclass'])
            intersection_meter.update(intersection)
            union_meter.update(union)
            a_meter.update(a_m)
            b_meter.update(b_m)
                # print('[{}] iter {}, accuracy: {}'
                #       .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                #               batch_idx, acc))

                # img = image.cpu().numpy()
                # print('origin iamge: ', type(origin_image))
                # if args.visualize:
                #     visualize_result(origin_image, origin_depth, label-1, output-1, batch_idx, args)

    iou = (intersection_meter.sum / (union_meter.sum + 1e-10)).mean()
    mAcc = (a_meter.average() / (b_meter.average()+1e-10)).mean()
    Accuracy = acc_meter.average() * 100
    return iou,mAcc,Accuracy
        
parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default='/workspace/UniMatch-main/configs/NYUV2_sup.yaml')
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--start-epochs", type=int, default=0)
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=0.9, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-path', type=str, default='/workspace/UniMatch-main/exp/NYUV2/100/sup')
parser.add_argument('--summary-dir', default='/summary', metavar='DIR',
                    help='path to save summary')
#将所有数据路径单独加
parser.add_argument('--train-labeled-path', type=str, default='/workspace/NYUV2/100/train_label.txt')
parser.add_argument('--train-unlabeled-path', type=str, default='/workspace/NYUV2/100/train_unlabel.txt')
parser.add_argument('--val-path', type=str, default='/workspace/NYUV2/100/val.txt')


def main():
    args = parser.parse_args()
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=True)
    g = torch.Generator()
    g.manual_seed(args.seed)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    os.makedirs(args.save_path+args.summary_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    logger = init_log('global', args.save_path+'/log.txt',logging.INFO)#logging.INFO将log日志记录输出的级别调到INFO  即正常命令行的输出也记录
    logger.propagate = 0  #关闭反馈机制 不需要下级给上级反馈输出
    logger.info('{}\n'.format(pprint.pformat(cfg)))#pprint.pformat 将文件中的信息以字符串的格式打印
    
    #搜索适合当前卷积的最优实现算法算子  实现网络加速
    # cudnn.enabled = True
    # cudnn.benchmark = True 
    
    if args.last_ckpt:
        model = ACNet(num_class=cfg['nclass'], pretrained=False)
    else:
        model = ACNet(num_class=cfg['nclass'], pretrained=True)
    
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))#打印计算量并写入log

    #分段设置lr 他本身就有backbone模块，需要自己弄个
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'],
                                momentum=0.9, weight_decay=1e-4)    
    model.cuda()
    # #
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = CrossEntropyLoss2d_u(cfg['dataset']).cuda()
    
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    #reduction=None 表示不求平均
    # criterion_u = CrossEntropyLoss2d_u(cfg['dataset'],cfg='none').cuda()

    # trainset_u = RGBD_Dataset(cfg['dataset'],  'train_u',
    #                           args.train_unlabeled_path)
    trainset_l = RGBD_Dataset(cfg['dataset'], 'train_l',
                              args.train_labeled_path)
    valset = RGBD_Dataset(cfg['dataset'], 'val',args.val_path)
    
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True,generator=g)
    
    # trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
    #                            pin_memory=True, num_workers=2, drop_last=True,worker_init_fn=seed_worker)
    
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False,generator=g)

    # total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    best_epoch = 0
    global_step = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.last_ckpt:
        previous_best,global_step, best_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)
        args.start_epochs =  best_epoch + 1
    lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    writer = SummaryWriter(args.save_path+args.summary_dir)
    for epoch in range(args.start_epochs,cfg['epochs']):
        # if epoch-best_epoch>20:
        #     break
        scheduler.step(epoch)
        logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.3f}, Epoch best: {:}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best,best_epoch))

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_mask_ratio = 0.0
        
        loader = trainloader_l
        for i, ((img_x, depth_x,mask_x,mask2_x,mask3_x,mask4_x,mask5_x)) in enumerate(loader):
            img_x, depth_x,mask_x,mask2_x,mask3_x,mask4_x,mask5_x = img_x.cuda(), depth_x.cuda(),\
                mask_x.cuda(),mask2_x.cuda(),mask3_x.cuda(),mask4_x.cuda(),mask5_x.cuda()
            model.train()
            preds,preds2,preds3,preds4,preds5 = model(img_x,depth_x)
            loss_x1 = criterion_l(preds, mask_x)
            loss_x2 = criterion_l(preds2, mask2_x)
            loss_x3 = criterion_l(preds3, mask3_x)
            loss_x4 = criterion_l(preds4, mask4_x)
            loss_x5 = criterion_l(preds5, mask5_x)
            loss_x = (loss_x1+loss_x2+loss_x3+loss_x4+loss_x5)/5.0         

            loss = loss_x
  
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # total_loss_x += loss_x.item()

            
            global_step += 1
            
            if (i % (len(trainloader_l) // 8) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(
                    i, total_loss / (i+1)))
            if global_step % args.print_freq == 0 or global_step == 1:
                writer.add_scalar('CrossEntropyLoss', loss.item(), global_step=global_step)
        
        eval_mode = 'original'
        mIOU, mAcc,Accuracy = evaluate(model, valloader, eval_mode, cfg)

        logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.3f},mAcc: {} , Accuracy: {:.2f}%\n'.format(eval_mode, mIOU,mAcc,Accuracy))

        if mIOU > previous_best :
            # if previous_best != 0:
            #     os.remove(os.path.join(args.save_path, '%s_epoch%d_%.3f.pth' % (cfg['backbone'],best_epoch, previous_best)))
            previous_best = mIOU
            best_epoch = epoch
            save_ckpt(cfg,args.save_path,model,optimizer,global_step, best_epoch,mIOU)


if __name__ == '__main__':
    main()
