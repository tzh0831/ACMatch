import random
import math
import torchvision
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
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
                   depth[i:i + image_h, j:j + image_w,:],\
                   label[i:i + image_h, j:j + image_w]
        
        return image[i:i + image_h, j:j + image_w, :],\
                   depth[i:i + image_h, j:j + image_w,:]


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
    def __call__(self,name,image, depth, label=None,label2=None,label3=None,label4=None,label5=None,):
        
        image = image / 255
        # image2 = image2 / 255
        if name =='NYUV2':
        # NYU V2 归一化
        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(image)
            image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                    std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
            depth = torchvision.transforms.Normalize(mean=[2.8424503515351494,2.8424503515351494,2.8424503515351494],
                                                 std=[0.9932836506164299,0.9932836506164299,0.9932836506164299])(depth)
        #SUNRGBD 归一化
        else:
            image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])(image)
            # image2 = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                          std=[0.229, 0.224, 0.225])(image2)
            depth = torchvision.transforms.Normalize(mean=[19050,19050,19050],
                                                    std=[9650,9650,9650])(depth)
        #需要追一下，看到底需不需要if 还是直接全部返回(暂定 没标签不需要label)
        if label is not None:
            label = torch.from_numpy(np.array(label))
            if label2 is not None:
                label2 = torch.from_numpy(np.array(label2)).long()
                label3 = torch.from_numpy(np.array(label3)).long()
                label4 = torch.from_numpy(np.array(label4)).long()
                label5 = torch.from_numpy(np.array(label5)).long()
                return image,depth,label,label2,label3,label4,label5
            return image,depth,label
        return image, depth


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, depth,label=None,mode=None):
        # image, depth, label = sample['image'], sample['depth'], sample['label']
        
        image = image.transpose((2, 0, 1))
        # depth = np.expand_dims(depth, 0).astype(np.float)
        depth = depth.transpose((2, 0, 1)).astype(np.float)
        # Generate different label scales
        if label is not None :
            if mode == 'val': 
                return  torch.from_numpy(image).float(),torch.from_numpy(depth).float(),\
                        torch.from_numpy(label).float()
            if mode =='label':
                return  torch.from_numpy(image).float(),torch.from_numpy(depth).float(),\
                        torch.from_numpy(label).float()
            elif mode =='train_l':
                label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                                                order=0, mode='reflect', preserve_range=True)
                label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                                order=0, mode='reflect', preserve_range=True)
                label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                                order=0, mode='reflect', preserve_range=True)
                label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                                order=0, mode='reflect', preserve_range=True)

                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                
                return  torch.from_numpy(image).float(),torch.from_numpy(depth).float(),\
                        torch.from_numpy(label).float(),torch.from_numpy(label2).float(),\
                        torch.from_numpy(label3).float(),torch.from_numpy(label4).float(),\
                        torch.from_numpy(label5).float()
        
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


def blur(img,depth=None,p=0.5):
    if random.random() < p:
        # print((img.numpy()).shape)
        # img = Image.fromarray(np.uint8(img))
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        if depth is not None:
            depth = depth.filter(ImageFilter.GaussianBlur(radius=sigma))
            return img,depth
        else:
            return img
    elif depth is not None:
        return img,depth
    else:
        return img


def obtain_cutmix_box(img_size_h,img_size_w, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size_w, img_size_h)
    depth = torch.zeros(img_size_w, img_size_h)
    if random.random() > p:
        return mask,depth

    size = np.random.uniform(size_min, size_max) * img_size_h * img_size_w
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size_w)
        y = np.random.randint(0, img_size_h)

        if x + cutmix_w <= img_size_w and y + cutmix_h <= img_size_h:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1
    depth[y:y + cutmix_h, x:x + cutmix_w] = 1#修改
    return mask,depth


# # # # # # # # # # # # # # # # # # # # # # # # 
# # # 2. Strong Augmentation for image Only
# # # # # # # # # # # # # # # # # # # # # # # # 

#原图
def img_aug_identity(img, scale=None):
    return img

#自动优化对比度
def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)

#直方图均匀分布
def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)

# #将输入图像转换为反色图像
# def img_aug_invert(img, scale=None):
#     return ImageOps.invert(img)

# 高斯模糊
def img_aug_blur(img,depth=None, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    # print(f"sigma:{sigma}")
    img =  img.filter(ImageFilter.GaussianBlur(radius=sigma))
    # if depth is not None:
    #         depth = depth.filter(ImageFilter.GaussianBlur(radius=sigma))
    #         return img,depth
    # else:
    return img
#对比度增强
def img_aug_contrast(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # # print(f"final:{v}")
    # v = np.random.uniform(scale[0], scale[1])
    return ImageEnhance.Contrast(img).enhance(v)

#亮度增强
def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Brightness(img).enhance(v)

#色度增强
def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Color(img).enhance(v)

#锐度增强
def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    # print(f"Final-V:{hue_factor}")
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img



#将每个颜色通道上变量bits对应的低(8-bits)个bit置0。变量bits的取值范围为[4，8]。
def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.posterize(img, v)

#在指定的阈值范围内，反转所有的像素点。即像素点二进制所有位取反
def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.solarize(img, v)

def get_augment_list(flag_using_wide=False):  
    if flag_using_wide:
        l = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_blur, [0.1, 2.0]),
        (img_aug_contrast, [0.1, 1.8]),
        (img_aug_brightness, [0.1, 1.8]),
        (img_aug_color, [0.1, 1.8]),
        (img_aug_sharpness, [0.1, 1.8]),
        (img_aug_posterize, [2, 8]),
        (img_aug_solarize, [1, 256])
        ]
    else:
        l = [
            (img_aug_identity, None),
            (img_aug_autocontrast, None),
            # (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]),
            (img_aug_contrast, [0.05, 0.95]),
            (img_aug_brightness, [0.05, 0.95]),
            (img_aug_color, [0.05, 0.95]),
            (img_aug_sharpness, [0.05, 0.95]),
            (img_aug_posterize, [4, 8]),
            # (img_aug_solarize, [1, 256])
        ]
    return l


class strong_img_aug:
    def __init__(self, num_augs, flag_using_random_num=False):
        assert 1<= num_augs <= 11
        self.n = num_augs
        self.augment_list = get_augment_list(flag_using_wide=False)
        self.flag_using_random_num = flag_using_random_num
    
    def __call__(self, img):
        if self.flag_using_random_num:
            max_num = np.random.randint(1, high=self.n + 1)
        else:
            max_num =self.n
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            # print("="*20, str(op))
            img = op(img, scales)
        return img
