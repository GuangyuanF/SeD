import torch
import torch.nn as nn
import torchvision.models as models


class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def get_target_label(self, input, target_is_real):
        """Get target label.
        获得目标标签
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.真假

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
            返回wgan的bool值否则返回张量
        """

        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        #返回一个与size大小相同的1填充的张量，× 标签个数
        return input.new_ones(input.size()) * target_val
        

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            损失模块的输入，即网络的预测
            target_is_real (bool): Whether the targe is real or fake.判断标签真假
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
            是否是鉴别器的loss，默认为否
        Returns:
            Tensor: GAN loss value.
            返回GAN的损失值
        """
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        # loss_weight is always 1.0 for discriminators 判别器的loss_weight为0
        return loss if is_disc else loss * self.loss_weight
        # return loss

"""
PerceptualVGG 感知损失VGG
"""
class PerceptualVGG(nn.Module):

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True):
        super().__init__()
        #指定要从中提取特征的层的名称
        self.layer_name_list = layer_name_list
        #确定是否对输入图像进行归一化处理
        self.use_input_norm = use_input_norm

        # get vgg model and load pretrained vgg weight获取VGG并加载预训练的权重
        # remove _vgg from attributes to avoid `find_unused_parameters` bug 从属性中移除_vgg以避免`find_unused_parameters`的bug

        assert vgg_type in ('vgg16', 'vgg19')
        if vgg_type == 'vgg16':
            _vgg = models.vgg16(pretrained=True)
        else:
            _vgg = models.vgg19(pretrained=True)

        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers
        # only borrow layers that will be used from _vgg to avoid unused params仅从VGG中借用相应的层，避免未使用的参数
        self.vgg_layers = _vgg.features[:num_layers]

        if self.use_input_norm:
            # the mean is for image with range [0, 1]图像的均值范围为[0,1]。
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [-1, 1]图像的均值范围为[-1,1]。
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for v in self.vgg_layers.parameters():
            v.requires_grad = False

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        #标准化
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'4': 1., '9': 1., '18': 1.}, which means the
            5th, 10th and 18th feature layer will be extracted with weight 1.0
            in calculating losses.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 norm_img=False,
                 criterion='l1'):
        super().__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = PerceptualVGG(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)

        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(x_features[k]),
                    self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    