from transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
# import torchdatasets as td
from torchvision import transforms


class RGBD_Dataset(Dataset):
    def __init__(self,name,mode,path,nsample=None,size=None):
        
        self.name = name
        self.mode = mode
       
        with open(path, 'r') as f:  #id_path = 
            self.ids = f.read().splitlines()
        # with open(depth_path,'r') as f:
        #     self.ids_depth = f.read().splitlines()
        
        # if mode == 'train_l' or mode == 'val':
        
            # with open(label_path,'r') as f:
            #     self.ids_label = f.read().splitlines()
        if mode == 'train_l' and nsample is not None:
            self.ids *= math.ceil(nsample / len(self.ids))          
            random.shuffle(self.ids)       
            self.ids = self.ids[:nsample]
    
    def __getitem__(self, item):
        
      
        
        id = self.ids[item]
        
        
        img = np.array(Image.open(id.split(' ')[0]))  
        depth = np.array(Image.open(id.split(' ')[1]))
        if self.mode == 'train_l' or self.mode == 'val':
            
            
            #H,W,C
            if self.name == 'SUNRGBD':
                mask = np.load(id.split(' ')[2],allow_pickle=True)
            else:
                mask = np.array(Image.open(id.split(' ')[2]))
            img,depth,mask = scaleNorm()(img,depth,mask)
            if self.mode == 'val': 
                
                
                img,depth,mask = ToTensor()(img,depth,mask,self.mode)
                img,depth,mask = Normalize()(self.name,img,depth,mask)
                return img,depth,mask
            
            
            img,depth,mask = RandomScale((1.0,1.4))(img,depth,mask)
            img = RandomHSV((0.9, 1.1),(0.9, 1.1),(25,25))(img)
            img, depth,mask = RandomCrop()(img,depth, mask)
            img,depth,mask = RandomFlip()(img,depth,mask)
            img,depth,mask,mask2,mask3,mask4,mask5 = ToTensor()(img,depth,mask,self.mode)
            return Normalize()(self.name,img,depth,mask,mask2,mask3,mask4,mask5)
            # img,depth,mask = ToTensor()(img,depth,mask,self.mode)
            # return Normalize()(self.name,img,depth,mask)

        img,depth = scaleNorm()(img,depth)
        img,depth = RandomScale((1.0,1.4))(img,depth)
        img = RandomHSV((0.9, 1.1),(0.9, 1.1),(25,25))(img)
        img,depth = RandomCrop()(img,depth)
        img,depth = RandomFlip()(img,depth)
        
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        depth_w,depth_s1,depth_s2 = deepcopy(depth),deepcopy(depth),deepcopy(depth)
        
        #H x W x C    
        img_s1 = Image.fromarray(np.uint8(img_s1))
        depth_s1 = Image.fromarray(np.uint8(depth_s1))
        img_s2=Image.fromarray(np.uint8(img_s2))
        depth_s2=Image.fromarray(np.uint8(depth_s2))
        
   
        # #弱增强与强增强
        # if random.random() < 0.8:
        #     # 80%概率进行亮度等增强
        #     img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        #     depth_s1= transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(depth_s1)
        # # #20%转为灰度图
        # img_s1= transforms.RandomGrayscale(p=0.2)(img_s1)
        # depth_s1 = transforms.RandomGrayscale(p=0.2)(depth_s1)
        # # # # #50%的概率进行高斯模糊
        
        # img_s1,depth_s1 = blur(img_s1,depth_s1, p=0.5)
        # #cutmix操作
        cutmix_box1,cutmix_box1_depth = obtain_cutmix_box(img_s1.size[0],img_s1.size[1], p=0.5)
        # # cutmix_box1_depth= obtain_cutmix_box(depth_s1.size[0],depth_s1.size[1], p=0.5)

     
        # if random.random() < 0.8:
        #     img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        #     depth_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(depth_s2)
        # img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        # depth_s2 = transforms.RandomGrayscale(p=0.2)(depth_s2)
        # img_s2,depth_s2 = blur(img_s2,depth_s2, p=0.5)
        
        # # #cutmix_box与cutmix_box_depth是两个完全一样的东西，照葫芦画瓢画的，
        cutmix_box2,cutmix_box2_depth = obtain_cutmix_box(img_s2.size[0], img_s2.size[1],p=0.5)
        # # cutmix_box2_depth = obtain_cutmix_box(depth_s2.size[0],depth_s2.size[1], p=0.5)
        
        
        
        img_s1=np.array(img_s1)
        depth_s1 = np.array(depth_s1)
        img_s2 = np.array(img_s2)
        depth_s2 = np.array(depth_s2)
        
        img_w,depth_w = ToTensor()(img_w,depth_w)
        img_s1,depth_s1 = ToTensor()(img_s1,depth_s1)
        img_s2,depth_s2 = ToTensor()(img_s2, depth_s2)
        
        img_w,depth_w = Normalize()(self.name,img_w,depth_w)
        img_s1,depth_s1 = Normalize()(self.name,img_s1,depth_s1)
        img_s2,depth_s2 = Normalize()(self.name,img_s2, depth_s2)
        
    
        
 
        return  img_w,depth_w,img_s1,depth_s1,img_s2,depth_s2,\
                cutmix_box1, cutmix_box2,cutmix_box1_depth,cutmix_box2_depth

    def __len__(self):
        return len(self.ids)




class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        # super().__init__() 
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open('partitions/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
        
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        
        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    
    RGBD_Dataset()