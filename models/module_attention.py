'''
Codes are inherited from:
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py
Modified by Bingchen Li
'''


from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

#如果val不是none返回true
def exists(val):
    return val is not None

#创建字典吗，去除重复的元素
def uniq(arr):
    return{el: True for el in arr}.keys()

#val存在则返回val，不存在则检查d是否为函数如果为函数则调用函数返回结果，否则返回d
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

#计算张量t数据类型的最大负值
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

#对张量进行初始化，首先获得张量的最后一个维度，计算标准差std，使其满足均匀分布在[-std,std]取随机值
def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
"""
GEGLU激活函数是GLU激活的一种变体,其中门控机制被应用了两次。第一次门控是由GELU激活函数执行的,第二次门控是通过将输入张量x与门控版本的gate张量相乘而执行的。

这种激活函数通常用于基于transformer的架构中,如门控卷积transformer(GCT)或视觉transformer(ViT),因为它被证明可以提高性能并稳定训练。

GEGLU模块通常用于基于transformer的模型的前馈层中,在那里它将GEGLU激活函数应用于线性投影层的输出。这使得模型能够学习输入数据的更复杂和非线性表示。
"""
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        #输入张量x通过proj线性层,该层将通道数加倍
        #proj层的输出然后在最后一维上被分割为两个张量(x和gate),使用chunk方法
        #最终输出通过将x与GEGLU(Gated Gated Linear Unit)激活函数的结果相乘来计算,
        #GEGLU激活是x与门控版本的gate(使用高斯误差线性单元(GELU)激活函数计算)的逐元素乘积。
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

"""
前馈神经网络，基于transformer，引入非线性，允许模型学习输入数据的更复杂表示。
多用于多头注意力层之后
使用GEGLU代替GELU。
dropout层正则化网络，防止过拟合。
"""
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        #nner_dim计算为dim * mult,这在transformer架构中是一种常见做法,用于增加中间层的维数。
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    将模型参数归零
    """
    for p in module.parameters():
        #detach方法将tensor从计算图分离，以获得一个不需要梯度的新tenfor
        #zero方法将分离后的tensor所有元素设为0，并在原位修改
        #yuetac0解科い厨白入在PyTorch中,张量（Tensor)默认会跟踪其计算历史,也就是说,在张量中包含了其被创建以来的完整计算图。如果我们不需要计算图中的一部分，可以使用 detach（)方法将其与计算图分离，从而减少内存占用并提高代码执行效率。具体地，y.detach（）会返回一个新的张量，其值与 y 相同，但是不再跟踪计算图，即不再保留其计算历史。
        #因此，y.detach（)返回的张量是一个独立的张量，对其进行操作不会影响原始张量 y，同时也不会对计算图产生影响。
        p.detach().zero_()
    return module

"""
GroupNorm用于归一化，Group Normalization是一种归一化技术,它对一组称为"组"的通道进行操作。它将输入通道划分为多个组,并独立在每个组内执行归一化。
这种方法在输入数据具有大量通道的情况下很有用,因为它可以帮助缓解内部协变量偏移的潜在问题,并提高神经网络的训练稳定性和性能。
num_groups=32: 指定将输入通道划分为多少组。在这种情况下,设置为32,这意味着输入通道被分成32组,并且归一化独立应用于每个组。
num_channels=in_channels: 指定总的输入通道数,它来自 Normalize 函数的参数。
eps=1e-6: 指定在归一化过程中添加到方差的一个小正值,以避免除以零。在这里,它被设置为1e-6(0.000001)。
affine=True: 确定 GroupNorm 层是否应该为每个组学习仿射变换(即,缩放和偏移)。将其设置为 True 允许层学习这些变换,可以帮助提高模型的性能。
"""
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

"""
线性注意力机制：
首先将输入张量投影为qkv。然后,它通过对key进行softmax归一化来计算注意力权重。上下文张量通过将归一化后的key与value相乘得到。最后,输出张量通过将上下文张量与query相乘并将结果投影回原始通道维度计算得到。
与传统自注意力机制相比,这种线性注意力机制更加高效,尤其是对于高分辨率图像,因为它避免了自注意力计算的二次复杂度。作为权衡,线性注意力机制只能模拟卷积层感受野内的局部交互,而自注意力可以捕获全局依赖关系。
"""
class LinearAttention(nn.Module):
    #dim：输入通道维度
    #heads：注意力头数
    #dim_head：确定每个注意力头的通道维度
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        #x投影位qkv
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        #将注意力机制的输出张量投影回原始通道维度
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        #接收一个形状为(batch_size, channels, height, width)的输入张量x
        b, c, h, w = x.shape
        #qkv张量通过将x传入self.to_qkv层得到
        qkv = self.to_qkv(x)
        #q、k和v张量通过使用einops库中的rearrange函数重排qkv张量得到
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        #k张量(键)在最后一维上使用softmax函数进行归一化
        k = k.softmax(dim=-1)  
        #context张量通过使用torch.einsum函数对归一化后的k和v张量进行矩阵乘法计算得到。
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        #out张量通过使用torch.einsum对context和q张量进行矩阵乘法计算得到
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        #out张量使用rearrange函数重排回原始形状。
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        #投影回原始通道维度
        return self.to_out(out)

"""
空间自注意力的工作原理是首先用单独的卷积层将输入张量投影为qkv，然后通过对q和k进行点积运算计算注意力权重，再进行缩放和归一化。
注意力加权通过将value和注意力权重相乘得到。最后输出张量通过将注意力加权投影回原始通道维度并加到输入张量上(残差连接)计算得到
"""
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        #首先将x归一化
        h_ = self.norm(h_)
        #获得qkv
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        #重排qkv
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        #q×k
        w_ = torch.einsum('bij,bjk->bik', q, k)
        #缩放
        w_ = w_ * (int(c)**(-0.5))
        #在最后一维上使用softmax进行归一化
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        #注意力加权值h_通过对值v和转置的注意力权重w_进行矩阵乘法计算得到
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        #重排回原始形状并通过proj_out投射回原始通道
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)
        #残差链接
        return x+h_


class CrossAttention(nn.Module):
    """
    交叉注意力机制，首先将q和上下文张量分别投影到qkv中。然后通过对q和k进行缩放点积运算计算注意力分数，注意力分数可以使用注意力掩码进行掩饰。注意力权重通过对注意力分数应用softmax函数计算得到。
    最后注意力加权通过将注意力权重和v相乘得到，最后投影回原始查询维度。

    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        # query: semantic; key, value: image.
        #query：语义;
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        #注意力分数sim通过对q和k进行缩放点击计算得到
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        #如果提供了注意力掩码mask,它会将无效区域的注意力分数掩蔽为一个非常大的负值。
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        #注意力权重attn通过对注意力分数sim在最后一维应用softmax函数计算得到
        attn = sim.softmax(dim=-1)
        #注意力加权值out通过对注意力权重attn和值v进行矩阵乘法计算得到,
        out = einsum('b i j, b j d -> b i d', attn, v)
        #使用rearrange函数重排回原始形状
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """
    
    """
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # x = self.attn1(self.norm1(x)) + x
        # x = self.attn2(self.norm2(x), context=context) + x
        # x = self.ff(self.norm3(x)) + x
        x = self.attn1(self.norm1(x))
        x = self.attn2(self.norm2(x), context=context)
        x = self.ff(self.norm3(x))
        return x


class ModifiedSpatialTransformer(nn.Module):
    """
    ModifiedSpatialTransformer模块将标准的转换器架构与额外的操作相结合,以处理图像中的空间信息。
    它首先将输入和上下文张量投影到展平表示,应用转换器块,然后将输出张量重塑回空间维度。输出张量还会通过卷积层进行空间上采样
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    
    用于类图像数据的转换器块。
    首先，将输入投影(也就是嵌入)并重塑为b, t, d。
    然后应用标准的变压器动作.
    最后对图像reshape
    """
    #up_factor上采样系数
    #depth转换器块数
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=1024, up_factor=2, is_last=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        #self.proj_in和self.proj_context层是卷积层,分别将输入和上下文张量投影到所有注意力头的组合维度。
        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_context = nn.Conv2d(context_dim,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        #执行标准的transforer操作
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=inner_dim)
                for d in range(depth)]
        )
        #将输出张量投影回原始输入通道维度
        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        #通过up_factor对输出张量进行空间上采样
        up_channels = int(in_channels / up_factor / up_factor)
        if not is_last:
            self.conv_out = nn.Conv2d(up_channels, up_channels, 3, 1, 1)
        else:
            self.conv_out = nn.Conv2d(up_channels, up_channels, 4, 1, 1)
        self.up_factor = up_factor

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        # x_in = x
        #输入和上下文张量分别通过self.proj_in和self.proj_context层进行投影
        x = self.norm(x)
        x = self.proj_in(x)
        context = self.proj_context(context)
        #对投影后的张量重排列
        #contiguous()返回一个内存连续的有相同数据的tensor，如果原tensor内存连续，则返回原tensor；
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        context = rearrange(context, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        #输出张量使用rearrange函数进行up_factor倍的空间上采样,然后应用self.conv_out层。
        # return x + x_in
        x = rearrange(x, 'b (c uw uh) h w -> b c (h uh) (w uw)', uh=self.up_factor, uw=self.up_factor).contiguous()  # rescale the semantic
        x = self.conv_out(x)
        return x
    

if __name__ == '__main__':
    M = ModifiedSpatialTransformer(1024, n_heads=1, d_head=64, context_dim=128)
    img = torch.randn(1, 128, 32, 32)
    context = torch.randn(1, 1024, 16, 16)

    haha = M(context, img)
    print(haha.size())

    a = torch.randn(1, 9, 2, 2)
    l = nn.PixelShuffle(3)
    b = l(a)
    print(b)
    c = rearrange(a, 'b (c uh uw) h w -> b c (h uh) (w uw)', uh=3, uw=3)
    print(b==c)