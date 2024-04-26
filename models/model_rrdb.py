import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
从创建一个sequential模块
'''
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

'''
5个卷积层的残差密集快
'''
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        #nf：输入/输出通道数默认为645
        #gc增长通道数，默认位32
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        #该模块包含五个卷积层,核大小为3x3,步长为1。每层的输出通道数如下:
        #conv1: gc个输出通道
        #conv2: gc个输出通道,输入来自x和x1
        #conv3: gc个输出通道,输入来自x, x1和x2
        #conv4: gc个输出通道,输入来自x, x1, x2和x3
        #conv5: nf个输出通道,输入来自x, x1, x2, x3和x4
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #每个卷积层后跟一个负斜率为0.2的LeakyReLU激活函数。
        self.init_weights()

    def init_weights(self):
        """Init weights for ResidualDenseBlock.
        Use smaller std for better stability and performance. We empirically
        use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
        Generative Adversarial Networks"
        init_weights方法使用较小的标准差(0.1)初始化卷积层的权重,这有助于提高稳定性和性能,如ESRGAN论文所建议的
        """
        for i in range(5):
            default_init_weights(getattr(self, f'conv{i+1}'), 0.1)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        #forward方法定义了模块的前向传播过程。它顺序应用这五个卷积层,在每个阶段连接输入,并将最终输出乘以0.2后加到原始输入x上
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

'''
函数会遍历模块中的所有子模块。对于每个子模块:
如果子模块是nn.Conv2d(二维卷积层):
使用Kaiming正态分布初始化权重,其中a=0(rectifier非线性的默认值)、mode='fan_in'(输入通道数决定权重方差)、nonlinearity='relu'(使用ReLU激活函数)。
将初始化后的权重乘以缩放因子scale。
如果子模块是nn.Linear(全连接层):
同样使用Kaiming正态分布初始化权重,参数与卷积层相同。
同样将初始化后的权重乘以缩放因子scale。
这种初始化方法被称为Kaiming初始化,是一种针对ReLU等非线性激活函数的有效初始化方法。将初始化后的权重乘以缩放因子scale可以帮助改善训练过程中的稳定性,特别是在使用残差块的情况下。
总的来说,这个函数提供了一个通用的权重初始化方法,可以应用于卷积层和全连接层。它有助于确保网络在训练开始时具有合适的激活分布,从而加快收敛速度和提高最终性能
'''
def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            m.weight.data *= scale

'''
这种RRDB结构是ESRGAN模型的关键组成部分。它由三个串联的残差密集块(ResidualDenseBlock_5C)组成,每个块都包含五个卷积层。这种嵌套的残差密集块结构可以有效地提取和融合多尺度特征,从而增强模型的表示能力。

最后将输出乘以0.2并加上原始输入,这是一种常见的残差连接技术,可以帮助优化训练过程并提高模型性能。

'''
class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    '''
    @parameters:
        n_nc: 输入通道数
        out_nc: 输出通道数
        nf: 特征通道数(默认为64)
        nb: RRDB块的数量(默认为23)
        gc: 每个RRDB块的增长通道数(默认为32)
    '''
    def __init__(self, in_nc, out_nc, nf=64, nb=23, gc=32):

        #该模块包含以下组件:
        #conv_first: 一个3x3卷积层,将输入映射到nf个特征通道
        #RRDB_trunk: 由nb个RRDB块串联而成的主干网络
        #trunk_conv: 一个3x3卷积层,用于融合主干网络的输出
        #上采样部分:
        #upconv1: 第一个上采样卷积层
        #upconv2: 第二个上采样卷积层
        #HRconv: 高分辨率卷积层
        #conv_last: 最终的3x3卷积层,将特征映射到输出通道数out_nc
        #lrelu: LeakyReLU激活函数
        super(RRDBNet, self).__init__()
        #functools.partial 函数允许你创建一个新的可调用对象,其参数集是原始可调用对象的一个子集
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # 首先使用conv_first将输入映射到nf个特征通道
        # 然后通过RRDB_trunk提取主干特征,并使用trunk_conv进行融合
        # 将融合后的特征与原始特征相加
        # 进行两次上采样,每次使用nearest-neighbor插值和卷积
        # 最后使用HRconv和conv_last输出最终的高分辨率结果
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out