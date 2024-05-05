import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F
import functools
import math
import clip
from clip.model import ModifiedResNet
from .module_attention import ModifiedSpatialTransformer

'''
继承ModifiedResNet，用于提取图像中的语义特征，并且使用了与训练的CLIP模型的视觉编码器作为基础
利用预训练的CLIP模型的视觉编码器作为特征提取器,从而获得具有语义意义的特征表示。这种方法可以在各种计算机视觉任务中提高性能,因为CLIP模型是在大规模的图文数据上进行预训练的,能够学习到丰富的视觉语义信息。
通过继承ModifiedResNet类,这个模块可以灵活地定制ResNet的层数和输出维度,以满足不同的应用需求。同时,保留CLIP模型的预训练权重可以提高模型的泛化能力,减少对大量标注数据的依赖。
'''
class CLIP_Semantic_extractor(ModifiedResNet):
    def __init__(self, layers=(3, 4, 6, 3), pretrained=True, path=None, output_dim=1024, heads=32):
        super(CLIP_Semantic_extractor, self).__init__(layers=layers, output_dim=output_dim, heads=heads)

        # #接受几个参数来初始化模块,包括层数、输出维度和注意力头数等。
        # 如果pretrained参数为True,则会加载预训练的CLIP模型的视觉编码器权重。
        # 将CLIP模型的均值和标准差注册为缓冲区,并将梯度更新设为False。
        
        # 如果path参数为None,则ckpt设置为'RN50',表示使用CLIP的ResNet-50作为视觉编码器。否则,ckpt的值就是传入的path参数。
        ckpt = 'RN50' if path is None else path
        #如果pretrained为True，使用clip.load(ckpt, device='cpu')函数加载预训练的CLIP模型,并将其保存到model变量中。这里使用了CPU设备进行加载,因为CLIP模型比较大,如果使用GPU可能会导致内存不足。
        if pretrained:
            model, _ = clip.load(ckpt, device='cpu')
        #使用self.load_state_dict(model.visual.state_dict())将CLIP模型的视觉编码器权重复制到当前模块。
        self.load_state_dict(model.visual.state_dict())
        #注册两个缓冲区:
        # 分别存储CLIP模型使用的输入图像的均值和标准差。
        # 这些值将在forward方法中用于对输入图像进行标准化。
        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )
        
        # 将当前模块的梯度更新设为False:这样可以防止在训练过程中修改预训练的CLIP模型权重。
        self.requires_grad_(False)
        # 最后删除model变量,释放相应的内存空间  
        del model

    def forward(self, x):
        # 首先定义了一个stem函数,它执行CLIP模型视觉编码器的前几层操作。
        # 对输入进行标准化,使用CLIP模型的均值和标准差。
        # 将标准化后的输入传递给stem函数,然后依次通过剩余的ResNet层。
        # 最终返回经过ResNet编码的特征。
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = (x - self.mean) / self.std
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

'''
实现了一个基于 PatchGAN 的语义驱动 (Semantic-Driven, SeD) 判别器。该判别器利用从预训练的 CLIP 模型中提取的语义特征,融合到卷积特征中,以增强其语义感知能力。
'''
class SeD_P(nn.Module):
    def __init__(self, input_nc, ndf=64, semantic_dim=1024, semantic_size=16, use_bias=True, nheads=1, dhead=64):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        接受输入通道数 input_nc、特征通道数 ndf、语义特征维度 semantic_dim、语义特征空间大小 semantic_size、是否使用偏置 use_bias、注意力头数 nheads 和注意力头维度 dhead 等参数。
        """
        super().__init__()

        kw = 4  #kernel_size

        padw = 1    #padding
        # ss = [128, 64, 32, 31, 30]  # PatchGAN's spatial size
        # cs = [64, 128, 256, 512, 1]  # PatchGAN's channel size

        norm = spectral_norm
        
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.conv_first = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)

        self.conv1 = norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        #法将 x 向上舍入到最接近的整数
        upscale = math.ceil(64 / semantic_size)
        self.att1 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=128, up_factor=upscale)

        ex_ndf = int(semantic_dim / upscale**2)
        self.conv11 = norm(nn.Conv2d(ndf * 2 + ex_ndf, ndf * 2, kernel_size=3, stride=1, padding=padw, bias=use_bias))

        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        upscale = math.ceil(32 / semantic_size)
        self.att2 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=256, up_factor=upscale)

        ex_ndf = int(semantic_dim / upscale**2)
        self.conv21 = norm(nn.Conv2d(ndf * 4 + ex_ndf, ndf * 4, kernel_size=3, stride=1, padding=padw, bias=use_bias))

        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        upscale = math.ceil(31 / semantic_size)
        self.att3 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=512, up_factor=upscale, is_last=True)

        ex_ndf = int(semantic_dim / upscale**2)
        self.conv31 = norm(nn.Conv2d(ndf * 8 + ex_ndf, ndf * 8, kernel_size=3, stride=1, padding=padw, bias=use_bias))

        self.conv_last = nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)

        init_weights(self, init_type='normal')

    def forward(self, input, semantic):
        """Standard forward."""
        input = self.conv_first(input)
        input = self.lrelu(input)

        input = self.conv1(input)
        se = self.att1(semantic, input)
        input = self.lrelu(self.conv11(torch.cat([input, se], dim=1)))

        input = self.conv2(input)
        se = self.att2(semantic, input)
        input = self.lrelu(self.conv21(torch.cat([input, se], dim=1)))

        input = self.conv3(input)
        se = self.att3(semantic, input)
        input = self.lrelu(self.conv31(torch.cat([input, se], dim=1)))

        input = self.conv_last(input)
        return input


class SeD_U(nn.Module):

    def __init__(self, num_in_ch=3, num_feat=64, semantic_dim=1024, semantic_size=16, skip_connection=True, nheads=1, dhead=64):
        super(SeD_U, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        upscale = math.ceil(128 / semantic_size)
        self.att1 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=128, up_factor=upscale)
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv11 = norm(nn.Conv2d(num_feat * 2 + ex_ndf, num_feat * 2, 3, 1, 1))


        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        upscale = math.ceil(64 / semantic_size)
        self.att2 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=256, up_factor=upscale)
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv21 = norm(nn.Conv2d(num_feat * 4 + ex_ndf, num_feat * 4, 3, 1, 1))


        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        upscale = math.ceil(32 / semantic_size)
        self.att3 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=512, up_factor=upscale)
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv31 = norm(nn.Conv2d(num_feat * 8 + ex_ndf, num_feat * 8, 3, 1, 1))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        # init_weights(self, init_type='orthogonal')

    def forward(self, x, semantic):

        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)

        x3 = self.conv3(x2)
        x3 = self.conv31(torch.cat([x3, self.att3(semantic, x3)], dim=1))
        x3 = F.leaky_relu(x3, negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = self.conv21(torch.cat([x4, self.att2(semantic, x4)], dim=1))
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = self.conv11(torch.cat([x5, self.att1(semantic, x5)], dim=1))
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
