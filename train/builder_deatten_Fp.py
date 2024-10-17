import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/workspace/UniMatch-main')
from util.init_func import init_weight

from util.load_utils import load_pretrain
from functools import partial
from dual_segformer_cnn_cat import mit_b4 as backbone
from util.logger import get_logger

logger = get_logger()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class  decoder(nn.Module):
    def __init__(self,num_class = 40):
        super().__init__()
        
        transblock = TransBasicBlock
        
        
        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 320, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)
        
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        
        self.sa4 = SpatialAttention()
        
        self.ca1 =  ChannelAttention(320)
        self.ca2 =  ChannelAttention(128)
        self.ca3 =  ChannelAttention(64)
        
        self.ca4 =  ChannelAttention(64)
        # self.agant0 = self._make_agant_layer(64, 64)
        # self.agant1 = self._make_agant_layer(64*4, 64)
        # self.agant2 = self._make_agant_layer(128*4, 128)
        # self.agant3 = self._make_agant_layer(256*4, 256)
        # self.agant4 = self._make_agant_layer(512, 512)
        
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)
        self.final_deconv = nn.ConvTranspose2d(self.inplanes, num_class, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        self.out5_conv = nn.Conv2d(320, num_class, kernel_size=1, stride=1, bias=True)
        self.out4_conv = nn.Conv2d(128, num_class, kernel_size=1, stride=1, bias=True)
        self.out3_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)
        
        # self.out5_conv = nn.Conv2d(512, num_class, kernel_size=1, stride=1, bias=True)
        # self.out4_conv = nn.Conv2d(320, num_class, kernel_size=1, stride=1, bias=True)
        # self.out3_conv = nn.Conv2d(128, num_class, kernel_size=1, stride=1, bias=True)
        # self.out2_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
    
    def forward(self, x):
   
        fuse3 = x[4]    #512,15,20
        fuse2 = x[3]    #320,30,40
        fuse1 = x[2]    #128,60,80
        fuse0 = x[1]    #64,120,160
        m0 = x[0] #64,240,320
        
        x = self.deconv1(fuse3)# 修改
        x = x.mul(self.ca1(x))
        x = x.mul(self.sa1(x))
        if self.training:
            out5 = self.out5_conv(x)  #40,15,20
        # agant3 = self.agant3(fuse3)
        # upsample 1
        x = x + fuse2 #修改
        x = self.deconv2(x)   #通道变320
        x = x.mul(self.ca2(x))
        x = x.mul(self.sa2(x))
        if self.training:
            out4 = self.out4_conv(x)  #40,30,40
        x = x + fuse1
        # upsample 2
        x = self.deconv3(x)    #128,60,80
        x = x.mul(self.ca3(x))
        x = x.mul(self.sa3(x))
        if self.training:
            out3 = self.out3_conv(x)  #40,60,80
        x = x + fuse0
        # upsample 3
        x = self.deconv4(x)
        x = x.mul(self.ca4(x))
        x = x.mul(self.sa4(x))
        if self.training:
            out2 = self.out2_conv(x)    #40,120,160    修改
        
        x = x + m0

        x = self.final_conv(x)
        out = self.final_deconv(x)
        
        if self.training:
            #120,160||60,80||30,40||15,20
            return out,out2, out3, out4, out5

        return out  #修改
    

    
    
   
    
class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, num_class =40, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer

        logger.info('Using backbone: Segformer-B4')
        
        self.backbone = backbone(norm_fuse=norm_layer)
        self.aux_head = None
        self.decoder = decoder(num_class)
        self.init_weights(cfg, pretrained=cfg.pre)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decoder, nn.init.kaiming_normal_,
                self.norm_layer, 1e-3, 0.1,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, 1e-3, 0.1,
                mode='fan_in', nonlinearity='relu')
        
    
    def encode_decode(self, rgb, modal_x,need_fp,single):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        # out = self.decode_head.forward(x)
        
        if need_fp:
            ms,ms2,ms3,ms4,ms5 = self.decoder.forward([torch.cat((x[0], nn.Dropout2d(0.5)(x[0]), nn.Dropout2d(0.5)(x[0]))),
                             torch.cat((x[1], nn.Dropout2d(0.5)(x[1]), nn.Dropout2d(0.5)(x[1]))),
                             torch.cat((x[2], nn.Dropout2d(0.5)(x[2]), nn.Dropout2d(0.5)(x[2]))),
                             torch.cat((x[3], nn.Dropout2d(0.5)(x[3]), nn.Dropout2d(0.5)(x[3]))),
                             torch.cat((x[4], nn.Dropout2d(0.5)(x[4]), nn.Dropout2d(0.5)(x[4])))])
            
            out ,out_fp,out_Fp = ms.chunk(3)
            out2,_,_ = ms2.chunk(3)
            out3,_ ,_= ms3.chunk(3)
            out4,_ ,_= ms4.chunk(3)
            out5,_ ,_= ms5.chunk(3)
            out = F.interpolate(out, size=(orisize[2], orisize[3]), mode='bilinear', align_corners=False)
            out_fp = F.interpolate(out_fp, size=(orisize[2], orisize[3]), mode='bilinear', align_corners=False)
            out_Fp = F.interpolate(out_Fp, size=(orisize[2], orisize[3]), mode='bilinear', align_corners=False)
            out2 = F.interpolate(out2, size=(orisize[2]// 2, orisize[3]// 2), mode='bilinear', align_corners=False)
            out3 = F.interpolate(out3, size=(orisize[2]// 4, orisize[3]// 4), mode='bilinear', align_corners=False)
            out4 = F.interpolate(out4, size=(orisize[2]// 8, orisize[3]// 8), mode='bilinear', align_corners=False)
            out5 = F.interpolate(out5, size=(orisize[2]// 16, orisize[3]// 16), mode='bilinear', align_corners=False)
            return out,out_fp,out_Fp,out2,out3,out4,out5
        
       
        
        
        if self.training:
            out,out2,out3,out4,out5 = self.decoder.forward(x)#修改
            #480,640||240,320||120,160||60,80
            out = F.interpolate(out, size=(orisize[2], orisize[3]), mode='bilinear', align_corners=False)
            out2 = F.interpolate(out2, size=(orisize[2]// 2, orisize[3]// 2), mode='bilinear', align_corners=False)
            out3 = F.interpolate(out3, size=(orisize[2]// 4, orisize[3]// 4), mode='bilinear', align_corners=False)
            out4 = F.interpolate(out4, size=(orisize[2]// 8, orisize[3]// 8), mode='bilinear', align_corners=False)
            out5 = F.interpolate(out5, size=(orisize[2]// 16, orisize[3]// 16), mode='bilinear', align_corners=False)
            if self.aux_head:
                aux_fm = self.aux_head(x[self.aux_index])
                aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
                return out,out2,out3,out4,aux_fm
            if single:
                return out
            return out,out2,out3,out4,out5
        else:
            out = self.decoder.forward(x)
            out = F.interpolate(out, size=(orisize[2], orisize[3]), mode='bilinear', align_corners=False)
            return out

    def forward(self, rgb, modal_x, need_fp = False,single = False,label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x,need_fp,single)
        else:
            out = self.encode_decode(rgb, modal_x,need_fp,single)
        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss
        return out
    
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out