import sys
sys.path.append("..")
import argparse
from copy import deepcopy
import logging
import os
import pprint

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml
# import torchdatasets as td
from semi_s import SemiDataset,RGBD_Dataset
from train.builder_deatten_gp import EncoderDecoder as segmodel
# from model.semseg.deeplabv3plus import DeepLabV3Plus
from ACNet_models_V1 import ACNet
from supervised1 import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log,set_random_seed,seed_worker
from util.dist_helper import setup_distributed
from utils import CrossEntropyLoss2d,CrossEntropyLoss2d_u,save_ckpt,load_ckpt,CrossEntropyLoss2d_u_ALIA,ThreshController
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from augs_ALIA import cut_mix_label_adaptive
from torch.cuda.amp import autocast, GradScaler
parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default='/workspace/shiyan3/configs/NYUV2.yaml')
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
parser.add_argument('--pre', default='/workspace/ACNet-master/pretrained/mit_b4.pth', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--save-path', type=str, default='/workspace/shiyan3/exp/NYUV2/100/ACT-Unimatch-Gp-min')
parser.add_argument('--summary-dir', default='/summary', metavar='DIR',
                    help='path to save summary')
#将所有数据路径单独加
# parser.add_argument('--train-labeled-path', type=str, default='/workspace/SUNRGBD/1322/train_label.txt')
# parser.add_argument('--train-unlabeled-path', type=str, default='/workspace/SUNRGBD/1322/train_unlabel.txt')
# parser.add_argument('--val-path', type=str, default='/workspace/SUNRGBD/1322/test.txt')

parser.add_argument('--train-labeled-path', type=str, default='/workspace/NYUV2/100/train_label.txt')
parser.add_argument('--train-unlabeled-path', type=str, default='/workspace/NYUV2/100/train_unlabel.txt')
parser.add_argument('--val-path', type=str, default='/workspace/NYUV2/100/test.txt')
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
    # if args.last_ckpt:
    #     model = ACNet(num_class=cfg['nclass'], pretrained=False)
    # else:
    #     model = ACNet(num_class=cfg['nclass'], pretrained=True)
    model = segmodel(cfg=args, num_class=cfg['nclass'], criterion=None, norm_layer=nn.GroupNorm)
    
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))#打印计算量并写入log

    #分段设置lr 他本身就有backbone模块，需要自己弄个
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'],
                                momentum=0.9, weight_decay=1e-4)    
    model.cuda()
    # #
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = CrossEntropyLoss2d_u(cfg['dataset']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    #reduction=None 表示不求平均
    criterion_u = CrossEntropyLoss2d_u(cfg['dataset'],cfg='none').cuda()
    criterion_u_ALIA = CrossEntropyLoss2d_u_ALIA(cfg['dataset'],cfg='none').cuda()
    # mse_loss = nn.MSELoss().cuda()
    
    trainset_u = RGBD_Dataset(cfg['dataset'],  'train_u',
                              args.train_unlabeled_path)
    trainset_l = RGBD_Dataset(cfg['dataset'], 'train_l',
                              args.train_labeled_path, nsample=len(trainset_u.ids))
    valset = RGBD_Dataset(cfg['dataset'], 'val',args.val_path)
    
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True,generator=g)
    
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True,generator=g)
    
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False,generator=g)

    thresh_controller = ThreshController(nclass=cfg['nclass'], momentum=cfg['momentum'], thresh_init=cfg['thresh_init'])
    
    
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
    scaler = GradScaler()
    writer = SummaryWriter(args.save_path+args.summary_dir)
    for epoch in range(args.start_epochs,cfg['epochs']):
       
        scheduler.step(epoch)
        logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.3f}, Epoch best: {:}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best,best_epoch))

        total_loss, total_loss_x, total_loss_s, total_loss_w = 0.0, 0.0, 0.0, 0.0
        total_mask_ratio = 0.0
        loader = zip(trainloader_l, trainloader_u)
        
        for i, ((img_x, depth_x,mask_x,mask2_x,mask3_x,mask4_x,mask5_x),
                (img_u_w, depth_u_w,img_u_s1, depth_u_s1,img_u_s2,depth_u_s2)) in enumerate(loader):

            optimizer.zero_grad()
            # img_x, depth_x,mask_x,mask2_x,mask3_x,mask4_x,mask5_x = img_x.cuda().half(), depth_x.cuda().half(),\
            #     mask_x.cuda(),mask2_x.cuda(),mask3_x.cuda(),mask4_x.cuda(),mask5_x.cuda()
            # img_u_w,depth_u_w = img_u_w.cuda().half(),depth_u_w.cuda().half()
            # img_u_s1, img_u_s2 = img_u_s1.cuda().half(), img_u_s2.cuda().half()
            # depth_u_s1,depth_u_s2 = depth_u_s1.cuda().half(),depth_u_s2.cuda().half()
            img_x, depth_x,mask_x,mask2_x,mask3_x,mask4_x,mask5_x = img_x.cuda(), depth_x.cuda(),\
                mask_x.cuda(),mask2_x.cuda(),mask3_x.cuda(),mask4_x.cuda(),mask5_x.cuda()
            img_u_w,depth_u_w = img_u_w.cuda(),depth_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            depth_u_s1,depth_u_s2 = depth_u_s1.cuda(),depth_u_s2.cuda()
            
            
            
            # model.float().half()
            with torch.no_grad():
                with autocast():
                #val只有一个out输出
                    model.eval()

                    pred_u_w = model(img_u_w,depth_u_w).detach()
                    # print('--1--------')
                    # print(pred_u_w.shape)
                    pred_u_w = F.softmax(pred_u_w, dim=1)#对每个通道特征进行概率转化 使所有通道的特征加起来值为1
                    # print('--2--------')
                    # print(pred_u_w.shape)
                    # obtain pseudos  在通道维找出概率最大的  并返回那个通道特征与下标  即logit与label
                    logits_u_s, label_u_s = torch.max(pred_u_w, dim=1)
                    # print('--3--------')
                    # print(label_u_s.shape)
                    # obtain confidence
                    entropy = -torch.sum(pred_u_w * torch.log(pred_u_w + 1e-10), dim=1)
                    entropy /= np.log(cfg['nclass'])
                    confidence = 1.0 - entropy
                    confidence = confidence * logits_u_s
                    confidence = confidence.mean(dim=[1,2])  # 1*C
                    confidence = confidence.cpu().numpy().tolist()
            label_u_s1,label_u_s2 = deepcopy(label_u_s),deepcopy(label_u_s)
            logits_u_s1,logits_u_s2 = deepcopy(logits_u_s),deepcopy(logits_u_s)
            model.train()
           
            #计算cutmix并拼接
            with autocast():
                if np.random.uniform(0, 1) > 0.5:
                    img_u_s1, depth_u_s1,label_u_s1, logits_u_s1 = cut_mix_label_adaptive(
                            img_u_s1,
                            depth_u_s1,
                            label_u_s1,
                            logits_u_s1, 
                            img_x,
                            depth_x,
                            mask_x, 
                            confidence
                        )
                    img_u_s2,depth_u_s2,label_u_s2,logits_u_s2 = cut_mix_label_adaptive(
                            img_u_s2,
                            depth_u_s2,
                            label_u_s2,
                            logits_u_s2, 
                            img_x,
                            depth_x,
                            mask_x, 
                            confidence
                        )
                #depth 还有mask 应该和img一样大小 ，只用一个就行
                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            
                preds, preds_fp,preds2,preds3,preds4,preds5 = model(torch.cat((img_x, img_u_w)),torch.cat((depth_x, depth_u_w)),need_fp=True)
            
            #这里应该是5个mask_x对应的5个pred_x
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])
                pred_x_2, _ = preds2.split([num_lb, num_ulb])  
                pred_x_3, _ = preds3.split([num_lb, num_ulb])  
                pred_x_4, _ = preds4.split([num_lb, num_ulb])  
                pred_x_5, _ = preds5.split([num_lb, num_ulb])     
                pred_u_w_fp = preds_fp[num_lb:]
                
                
                pred_u_w = pred_u_w.detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

                pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)),torch.cat((depth_u_s1, depth_u_s2)),single=True).chunk(2)
            
            #有监督五个输出监督
                loss_x1 = criterion_l(pred_x, mask_x)
                loss_x2 = criterion_l(pred_x_2, mask2_x)
                loss_x3 = criterion_l(pred_x_3, mask3_x)
                loss_x4 = criterion_l(pred_x_4, mask4_x)
                loss_x5 = criterion_l(pred_x_5, mask5_x)
                loss_x = (loss_x1+loss_x2+loss_x3+loss_x4+loss_x5)/5.0
            #弱增强的预测作为强增强的监督  
            # print('--4--------')
            # print(pred_u_w.shape)
                thresh_controller.thresh_update(pred_u_w.detach(), None, update_g=True)
                thresh_global = thresh_controller.get_thresh_global()
                #改进的自适应阈值就是这一行取个最大值  
                thresh_global = max(thresh_global, torch.tensor(cfg['thresh_min']).cuda())
                loss_u_s1, pseduo_high_ratio_s1 = criterion_u_ALIA(
                            pred_u_s1, label_u_s1.detach(),
                            logits_u_s1.detach(), thresh=thresh_global)
                
                loss_u_s2, pseduo_high_ratio_s2 = criterion_u_ALIA(
                            pred_u_s2, label_u_s2.detach(),
                            logits_u_s2.detach(), thresh=thresh_global)
            
           
                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= thresh_global))
                # loss_u_w_fp = torch.sum(loss_u_w_fp)/conf_u_w.sum().item() 
                loss_u_w_fp = torch.mean(loss_u_w_fp)


                loss = (loss_x  + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2
            
                del loss_x1,loss_x2,loss_x3,loss_x4,loss_x5
                torch.cuda.empty_cache()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
         

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_s += (loss_u_s1.item() + loss_u_s2.item()) / 2.0
            total_loss_w += loss_u_w_fp.item()
            total_mask_ratio += (pseduo_high_ratio_s1.item()+pseduo_high_ratio_s2.item())/2.0
           
            global_step += 1
            
            if (i % (len(trainloader_u) // 8) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, '
                            'Loss s: {:.3f}, Loss w: {:.3f}, Mask: {:.3f}, thresh_global: {:.3f}'.format(
                    i, total_loss / (i+1), total_loss_x / (i+1), total_loss_s / (i+1),
                    total_loss_w / (i+1), total_mask_ratio / (i+1),thresh_global))
            if global_step % args.print_freq == 0 or global_step == 1:
                writer.add_scalar('CrossEntropyLoss', loss.item(), global_step=global_step)
                writer.add_scalar('loss_x', loss_x.item(), global_step=global_step)
                writer.add_scalar('loss_u_s1', loss_u_s1.item(), global_step=global_step)
                writer.add_scalar('loss_u_s2', loss_u_s2.item(), global_step=global_step)
                writer.add_scalar('loss_u_w_fp', loss_u_w_fp.item(), global_step=global_step)
                writer.add_scalar('thresh_global', thresh_global.item(), global_step=global_step)
           
        # model.float()
        eval_mode = 'original'
        mIOU, mAcc,Accuracy = evaluate(model, valloader, eval_mode, cfg)
        
        
        logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.3f},mAcc: {} , Accuracy: {:.2f}%\n'.format(eval_mode, mIOU,mAcc,Accuracy))

        if mIOU > previous_best :
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_epoch%d_%.3f.pth' % (cfg['backbone'],best_epoch, previous_best)))
            previous_best = mIOU
            best_epoch = epoch
            save_ckpt(cfg,args.save_path,model,optimizer,global_step, best_epoch,mIOU)
            # torch.save(model.state_dict(),
            #            os.path.join(args.save_path, '%s_epoch%d_%.3f.pth' % (cfg['backbone'], best_epoch,mIOU)))
        

if __name__ == '__main__':
    main()
