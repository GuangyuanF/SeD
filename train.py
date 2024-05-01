import argparse
from collections import OrderedDict
import os

import numpy as np
import cv2
import yaml

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch import distributed as dist
import torch.optim as optim

from models import model_rrdb, model_swinir, sed
from datasets import srdata

import logging
from utils import utils_logger, util_calculate_psnr_ssim, losses


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--opt', type=str, help='path to option file', required=True) #指定配置文件的路径,该文件包含训练过程的配置设置
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to the checkpoints for pretrained model",
    )
    #指定预训练模型检查点的路径,可用于恢复训练
    parser.add_argument(
        '--distributed',
        action='store_true'
    )
    #启用分布式训练,允许模型在多个 GPU 上进行训练。
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    #指定当前进程在分布式训练设置中的本地排名

    
    parser.add_argument('--data_root', type=str, default='./DF2K')
    parser.add_argument('--data_test_root', type=str, default='./Evaluation')
    parser.add_argument('--out_root', type=str, default='./checkpoint')

    args = parser.parse_args()

    return args


#数据采样器，默认随机打乱，分布式训练
def data_sampler(dataset, shuffle=True, distributed=True):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def main():
    args = parse_args()

    # Initialization
    #读取args的opt属性
    with open(args.opt, 'r') as f:
        #使用yaml.safe_load()加载配置文件
        opt = yaml.safe_load(f)
    #将 opt 字典中的 name 字段中的字符串 "RRDB" 替换为 model_type 字段的值来修改它。
    opt['name'] = opt['name'].replace('RRDB', opt['model_type'])
    print(opt)
    #根据 out_root 参数和 opt 字典中的 name 字段构建检查点目录的路径。如果目录不存在,则在当前进程的本地排名为 0 (即主进程)时创建该目录。
    ckpt_path = os.path.join(args.out_root, opt['name'])
    if not os.path.exists(ckpt_path):
        if torch.cuda.current_device() == 0:
            os.makedirs(ckpt_path, exist_ok=True)
    #根据 out_root 参数和 opt 字典中的 name 字段构建检查点目录的路径。如果目录不存在,则在当前进程的本地排名为 0 (即主进程)时创建该目录。    
    logger_name = opt['stage']
    utils_logger.logger_info(logger_name, os.path.join(ckpt_path, logger_name+'.log'), mode='w')
    logger = logging.getLogger(logger_name)
    #torch.cuda.set_device(args.local_rank) 这一行确保了当前进程使用基于其本地排名的正确 GPU 设备。
    #torch.distributed.init_process_group(backend="nccl") 这一行初始化了 PyTorch 分布式包,它提供了分布式训练过程所需的通信基元。
    # 选择 NCCL 后端是因为它是一个高度优化和高效的 GPU 到 GPU 通信库,这对分布式训练的性能至关重要。
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    #设置随机种子
    
    if opt['manual_seed']:      #检查配置 (opt) 字典中是否指定了手动种子值
        torch.manual_seed(opt['manual_seed'])    # PyTorch 的 CPU 操作设置随机种子。这确保了诸如数据预处理之类的基于 CPU 的计算是确定性的,并且可以重现。
        torch.cuda.manual_seed(opt['manual_seed'])   #当前 CUDA 设备上的 PyTorch GPU 操作设置随机种子。这确保了诸如神经网络前向和反向传播之类的基于 GPU 的计算是确定性的,并且可以重现。
        torch.cuda.manual_seed_all(opt['manual_seed'])    #所有可用的 CUDA 设备上的 PyTorch GPU 操作设置随机种子。在多 GPU 设置中需要这样做,因为它确保所有 GPU 上的计算都是确定性的,并且可以重现。

    loss_weight = opt['loss_weights']

    # Models
    #如果 opt 字典中的 model_type 字段设置为 'RRDB'(不区分大小写),
    # 代码会创建一个定义在 model_rrdb 模块中的 RRDBNet 模型的实例。
    # 该模型使用 opt['model']['rrdb'] 字典中指定的参数进行初始化。
    # 然后将创建的模型移动到 'cuda' 设备(即 GPU)上。
    if opt['model_type'].lower() == 'rrdb':
        model = model_rrdb.RRDBNet(**opt['model']['rrdb']).to('cuda')

    # 如果 opt 字典中的 model_type 字段设置为 'SwinIR'(不区分大小写),
    # 代码会创建一个定义在 model_swinir 模块中的 SwinIR 模型的实例。
    # 该模型使用 opt['model']['swinir'] 字典中指定的参数进行初始化。
    # 然后将创建的模型移动到 'cuda' 设备(即 GPU)上。
    elif opt['model_type'].lower() == 'swinir':
        model = model_swinir.SwinIR(**opt['model']['swinir']).to('cuda')
    else:
        raise ValueError(f"Model {opt['model_type']} is currently unsupported!")
    #** 运算符用于展开字典 opt['model_ex'],并将其键值对作为关键字参数传递给 CLIP_Semantic_extractor 类的构造函数。
    #创建了 CLIP_Semantic_extractor 类的一个新实例
    model_ex = sed.CLIP_Semantic_extractor(**opt['model_ex']).to('cuda')

    if opt['name'].split('_')[-1] == 'P+SeD':
        #创建了 SeD_P 类的一个实例
        #使用 ** 运算符展开 opt['model_d'] 字典,将其参数传递给 SeD_P 类的构造函数。
        model_d = sed.SeD_P(**opt['model_d']).to('cuda')
    elif opt['name'].split('_')[-1] == 'U+SeD':
        model_d = sed.SeD_U(**opt['model_d']).to('cuda')

    # 首先检查当前设备是否为GPU 0(可能用于日志记录),然后验证 args.resume 指定的文件是否存在。如果文件不存在,它将引发 ValueError。
    # 如果文件存在,它使用 torch.load 加载检查点,并使用 model.load_state_dict(ckpt) 加载模型的状态字典。最后,如果当前设备是GPU 0,它会记录一条消息
    if args.resume is not None:
        if torch.cuda.current_device() == 0:
            logger.info(f"load pretrained model: {args.resume}")
        if not os.path.isfile(args.resume):
            raise ValueError
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt)
        if torch.cuda.current_device() == 0:
            logger.info("model checkpoint load!")
    # 使用 Adam 优化器为两个模型,并使用 MultiStepLR 学习率调度器。优化器是使用各自模型需要梯度的参数创建的。调度器可能用于在训练过程中调整学习率。        
    # Optimizers
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], **opt['optimizer'])
    optimizer_d = optim.Adam([p for p in model_d.parameters() if p.requires_grad], **opt['optimizer_d'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **opt['scheduler'])
    scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, **opt['scheduler_d'])

    # Loss
    # Loss settings are hard-coded for now!
    # 这行初始化一个L1损失函数,通常用于逐像素重建任务。第二行将损失函数移动到GPU
    loss_pix = torch.nn.L1Loss()
    loss_pix = loss_pix.to('cuda')
    #始化一个GAN损失函数,具体来说是"vanilla"GAN损失,即Goodfellow等人在开创性论文中提出的原始GAN损失。loss_weight['loss_g']部分意味着该GAN损失的损失权重在字典loss_weight中指定。
    loss_g = losses.GANLoss(gan_type='vanilla', loss_weight=loss_weight['loss_g']).to('cuda')
    #定义了一个字典loss_dict_per,它将层索引(可能是用于感知损失的预训练网络)映射到相应的权重。
    loss_dict_per = {'2': 0.1, '7': 0.1, '16': 1.0, '25': 1.0, '34': 1.0}
    #始化一个感知损失函数,通常用于图像到图像翻译任务。layer_weights参数设置为之前定义的loss_dict_per字典,指定了预训练网络不同层的权重。
    #perceptual_weight参数设置为loss_weight['loss_p'],可能是loss_weight字典中的另一个权重。criterion='l1'部分指定应使用L1范数来计算感知损失。
    loss_p = losses.PerceptualLoss(layer_weights=loss_dict_per, perceptual_weight=loss_weight['loss_p'], criterion='l1').to('cuda')

    # Datasets
    #从opt中检测一个布尔值，用于确定是否使用测试数据集
    use_eval = opt['datasets']['test']['use_test']
    #使用srdata.Train根据opt的配置和dataroot的配置创建数据加载器
    trainset = srdata.Train(**opt['datasets']['train'], data_root=args.data_root)
    #使用DataLoader进行加载
    data_loader = data.DataLoader(
        trainset, 
        **opt['dataloader']['train'],
        sampler=data_sampler(trainset, shuffle=True, distributed=args.distributed),
    )
    #如果use_eval为true，使用srdata.Test类对data_root参数初始化测试数据集
    #使用 opt['dataloader']['test'] 的配置为测试数据集创建一个单独的数据加载器
    if use_eval:
        testset = srdata.Test(data_root=args.data_test_root)
        data_loader_test = data.DataLoader(
            testset, 
            **opt['dataloader']['test'],
            sampler=data_sampler(testset, shuffle=False, distributed=False),
        )
    #如果 args.distributed 为 True,则将 model 和 model_d封装在PyTorch的 DistributedDataParallel 模块中,它启用了在多个GPU上的分布式训练。
    #设备ID和输出设备被设置为 args.local_rank,并且 broadcast_buffers 被设置为 True。
    #broadcast_buffers：设置为True时，在模型执行forward之前，gpu0会把buffer中的参数值全部覆盖
    # 到别的gpu上。注意这和同步BN并不一样，同步BN应该使用SyncBatchNorm。
    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )
        model_d = DistributedDataParallel(
            model_d,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )
    
    # Training settings
    #设置总训练epoches数为一个非常大的值(10,000,000)。在实践中,训练可能会基于其他标准停止,例如收敛或最大迭代次数,而不是真正运行这么多epoches。
    #从配置字典 opt 中获取当前的步数或迭代数。这可能用于从先前的检查点恢复训练,或者跟踪训练过程的进度。
    #用作一个标志来控制训练循环的终止。
    total_epochs = 10000000
    current_step = opt['train']['current_step']
    endflag = False

    # Training starts
    for epoch in range(total_epochs):
        #获取学习率，文件名
        #TODO:hr是什么
        for lr, hr, filename in data_loader:
            #它递增当前步数,根据最大步数检查是否应该停止训练,获取学习率,预处理数据,并初始化一个字典来存储损失值。
            current_step += 1
            #检查是否应该停止训练 
            if current_step > opt['train']['total_step']:
                endflag = True
                break
            #获取学习率      
            learning_r = optimizer.param_groups[0]['lr']
            #数据预处理
            filename = filename[0].split('/')[-1]
            lr = lr.to('cuda')
            hr = hr.to('cuda')
            hr_semantic = model_ex(hr)
            #初始化损失字典
            loss_dict = OrderedDict()
            #使用生成模型计算超分辨率图像
            #生成器更新
            sr = model(lr)

            for p in model_d.parameters():
                p.requires_grad = False
            #重置梯度
            optimizer.zero_grad()

            l_g_total = 0

           
            #计算并累计损失
            # pixel loss像素损失
            loss_pixel = loss_pix(sr, hr)
            l_g_total += loss_pixel * loss_weight['loss_pix']
            loss_dict['loss_pix'] = loss_pixel.item()
            # perceptual loss
            #感知损失
            loss_percep = loss_p(sr, hr)
            l_g_total += loss_percep
            loss_dict['loss_p'] = loss_percep.item()
            # gan loss
            #对抗损失 
            fake_g_pred = model_d(sr, hr_semantic)
            loss_gan = loss_g(fake_g_pred, True, is_disc=False)
            l_g_total += loss_gan
            #累计损失

            loss_dict['loss_g'] = loss_gan.item()
            l_g_total.backward()
            #反向传播
            optimizer.step()
            scheduler.step()

            # optimize net_d
            #d对model_d的参数进行自动求导，存放在grad中让优化器更新参数
            for p in model_d.parameters():
                p.requires_grad = True

            optimizer_d.zero_grad()
            # real
            #计算真实图像的损失
            real_d_pred = model_d(hr, hr_semantic)
            l_d_real = loss_g(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real.item()
            l_d_real.backward()
            # fake
            #计算假SR图像损失
            fake_d_pred = model_d(sr.detach().clone(), hr_semantic)  # clone for pt1.9
            l_d_fake = loss_g(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake.item()
            l_d_fake.backward()
            optimizer_d.step()
            scheduler_d.step()
            #在规律的间隔记录损失值,以及在指定的间隔保存模型检查点。
            if current_step % opt['train']['log_every'] == 0 and torch.cuda.current_device() == 0:
                logger.info('LR: {} | Step: {} | loss_pix: {:.3f} | loss_per: {:.3f} | loss_gan: {:.5f} | loss_d_real: {:.3f} | loss_d_fake: {:.3f}'.format(
                    learning_r,
                    current_step,
                    loss_dict['loss_pix'],
                    loss_dict['loss_p'],
                    loss_dict['loss_g'],
                    loss_dict['l_d_real'],
                    loss_dict['l_d_fake']))

            if current_step % opt['train']['save_every'] == 0 and torch.cuda.current_device() == 0:
                m = model.module if args.distributed else model
                model_dict = m.state_dict()
                if torch.cuda.current_device() == 0:
                    torch.save(
                        model_dict,
                        os.path.join(ckpt_path, 'model_{}.pt'.format(current_step))
                    )
            #如果use_eval标志被设置,这个块将在规律的间隔对测试数据评估模型,计算PSNR和SSIM指标。它将模型切换到评估模式,计算指标,记录它们,然后将模型切换回训练模式。
            if use_eval and current_step % opt['train']['test_every'] == 0 and torch.cuda.current_device() == 0:
                model.eval()
                p = 0
                s = 0
                count = 0
                for lr, hr, filename in data_loader_test:
                    count += 1
                    lr = lr.to('cuda')
                    filename = filename[0]
                    with torch.no_grad():
                        sr = model(lr)
                    sr = sr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
                    sr = sr * 255.
                    sr = np.clip(sr.round(), 0, 255).astype(np.uint8)
                    hr = hr.squeeze(0).numpy().transpose(1, 2, 0)
                    hr = hr * 255.
                    hr = np.clip(hr.round(), 0, 255).astype(np.uint8)

                    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
                    hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
                    
                    psnr = util_calculate_psnr_ssim.calculate_psnr(sr, hr, crop_border=4, test_y_channel=True)
                    ssim = util_calculate_psnr_ssim.calculate_ssim(sr, hr, crop_border=4, test_y_channel=True)
                    p += psnr
                    s += ssim
                    logger.info('{}: {}, {}'.format(filename, psnr, ssim))

                p /= count
                s /= count

                logger.info("Epoch: {}, Step: {}, psnr: {}. ssim: {}.".format(epoch, current_step, p, s))
                model.train()
        #如果endflag被设置(表示训练应该停止),则中断外层循环,并记录一条消息表示训练完成。
        if endflag:
            break
    logger.info('Done')

if __name__ == '__main__':
    main()