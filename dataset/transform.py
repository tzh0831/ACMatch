import random
import math
import torchvision
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import skimage.transform
import matplotlib
import matplotlib.colors

image_h = 480
image_w = 640
class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, img):
        
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return img_new


class scaleNorm(object):
    def __call__(self, image, depth, label=None):
        

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        if label is not None:
            label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

            return image, depth, label
        
        return image, depth


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)#1.0
        self.scale_high = max(scale)#1.4

    def __call__(self, image, depth, label=None):
        
        #1.0-1.4之间随机生成一个数
        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        #先四舍五入然后int成整数
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear  线性插值
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor  最近邻插值
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        if label is not None:
            label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
            return image, depth, label
        
        return image, depth


class RandomCrop(object):
    def __init__(self, th=image_h, tw=image_w):
        self.th = th
        self.tw = tw

    def __call__(self, image, depth, label=None):
        
        h = image.shape[0]
        w = image.shape[1]
        # padw = self.tw - w if w < self.tw else 0
        # padh = self.th - h if h < self.th else 0
        # image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
        # depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)
        # label = ImageOps.expand(label, border=(0, 0, padw, padh), fill=ignore_value)

        
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)
        # img = img.crop((x, y, x + self.tw, y + self.th))
        # mask = mask.crop((x, y, x + self.tw, y + self.th))
        # depth = depth.crop((x, y, x + self.tw, y + self.th))
        if label is not None:
            return image[i:i + image_h, j:j + image_w, :],\
                   depth[i:i + image_h, j:j + image_w],\
                   label[i:i + image_h, j:j + image_w]
        
        return image[i:i + image_h, j:j + image_w, :],\
                   depth[i:i + image_h, j:j + image_w]


class RandomFlip(object):
    def __call__(self, image, depth, label=None):
        
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            if label is not None:
                label = np.fliplr(label).copy()
                return image, depth, label
        if label is not None:
            return image, depth, label
        return image,depth

# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, image, depth, label=None,label2=None,label3=None,label4=None,label5=None):
        
        image = image / 255
        # image2 = image2 / 255
        # NYU V2 归一化
        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(image)
        # image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
        #                                          std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
        # depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
        #                                          std=[0.9932836506164299])(depth)
        #SUNRGBD 归一化
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        # image2 = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(image2)
        depth = torchvision.transforms.Normalize(mean=[19050],
                                                 std=[9650])(depth)
        #需要追一下，看到底需不需要if 还是直接全部返回(暂定 没标签不需要label)
        if label is not None:
            label = torch.from_numpy(np.array(label))
            # if label2 is not None:
            #     label2 = torch.from_numpy(np.array(label2)).long()
            #     label3 = torch.from_numpy(np.array(label3)).long()
            #     label4 = torch.from_numpy(np.array(label4)).long()
            #     label5 = torch.from_numpy(np.array(label5)).long()
                # return image, image2,depth,label,label2,label3,label4,label5
            return image,depth,label
        return image, depth


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, depth,label=None,mode=None):
        # image, depth, label = sample['image'], sample['depth'], sample['label']
        
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        # Generate different label scales
        if label is not None :
            if mode == 'val': 
                return  torch.from_numpy(image).float(),torch.from_numpy(depth).float(),\
                        torch.from_numpy(label).float()
            # label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
            #                                 order=0, mode='reflect', preserve_range=True)
            # label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
            #                                 order=0, mode='reflect', preserve_range=True)
            # label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
            #                                 order=0, mode='reflect', preserve_range=True)
            # label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
            #                                 order=0, mode='reflect', preserve_range=True)

            # image2 = skimage.transform.resize(image, (image.shape[0] // 2, image.shape[1] // 2),
            #                                 order=0, mode='reflect', preserve_range=True)
            # image3 = skimage.transform.resize(image, (image.shape[0] // 4, image.shape[1] // 4),
            #                                 order=0, mode='reflect', preserve_range=True)
            # image4 = skimage.transform.resize(image, (image.shape[0] // 8, image.shape[1] // 8),
            #                                 order=0, mode='reflect', preserve_range=True)
            # image5 = skimage.transform.resize(image, (image.shape[0] // 16, image.shape[1] // 16),
            #                                 order=0, mode='reflect', preserve_range=True)
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            
            return  torch.from_numpy(image).float(),torch.from_numpy(depth).float(),\
                    torch.from_numpy(label).float()
            
            # return  torch.from_numpy(image).float(),torch.from_numpy(depth).float(),\
            #         torch.from_numpy(label).float(),torch.from_numpy(label2).float(),\
            #         torch.from_numpy(label3).float(),torch.from_numpy(label4).float(),\
            #         torch.from_numpy(label5).float()
            
            # return  torch.from_numpy(image).float(),torch.from_numpy(image2).float(),\
            #         torch.from_numpy(image3).float(),torch.from_numpy(image4).float(),\
            #         torch.from_numpy(image5).float(),torch.from_numpy(depth).float(),\
            #         torch.from_numpy(label).float(),torch.from_numpy(label2).float(),\
            #         torch.from_numpy(label3).float(),torch.from_numpy(label4).float(),\
            #         torch.from_numpy(label5).float()
        
        return  torch.from_numpy(image).float(),torch.from_numpy(depth).float()
       

def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img,p=0.5):
    if random.random() < p:
        # print((img.numpy()).shape)
        # img = Image.fromarray(np.uint8(img))
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        # depth = depth.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size_w,img_size_h, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size_h, img_size_w)
    depth = torch.zeros(img_size_h, img_size_w)
    if random.random() > p:
        return mask,depth

    size = np.random.uniform(size_min, size_max) * img_size_h * img_size_w
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size_h)
        y = np.random.randint(0, img_size_w)

        if x + cutmix_h <= img_size_h and y + cutmix_w <= img_size_w:
            break

    mask[y:y + cutmix_w, x:x + cutmix_h] = 1
    depth[y:y + cutmix_w, x:x + cutmix_h] = 1
    return mask,depth
