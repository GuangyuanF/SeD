import os
import argparse

import torch
import cv2
import numpy as np
import yaml

from models import model_rrdb, model_swinir
from datasets import srdata_test
from torch.utils import data

import logging
from utils import utils_logger, util_calculate_psnr_ssim

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--opt', type=str, help='path to option file', required=True)
    parser.add_argument('--output_path', type=str, help='path to your output', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Initialization
    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)
    opt['name'] = opt['name'].replace('RRDB', opt['model_type'])
    print(opt)

    ckpt_path = opt['ckpt_path']
    #加载预训练模型权重
    #map_location适用于修改模型能在gpu上运行还是cpu上运行,指定如何重新映射存储位置。
    #load的map_location中有storage和location两个参数，storage参数将初始化反序列化storage，存储在CPU上。
    #lambda storage, loc: storage：GPU->CPU
    #torch.load('modelparameters.pth', map_location=lambda storage, loc: storage.cuda(1))    cpu -> gpu 1
    #torch.load('modelparameters.pth', map_location={'cuda:1':'cuda:0'})  gpu 1 -> gpu 0
    weight = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    weight = weight['model']

    # Models初始化模型
    if opt['model_type'].lower() == 'rrdb':
        model = model_rrdb.RRDBNet(**opt['model']['rrdb']).to('cuda')
    elif opt['model_type'].lower() == 'swinir':
        model = model_swinir.SwinIR(**opt['model']['swinir']).to('cuda')
    else:
        raise ValueError(f"Model {opt['model_type']} is currently unsupported!")

    model.load_state_dict(weight)
    model = model.cuda()

    # Datasets设置测试数据集
    testset = srdata_test.Test(**opt['test'])
    data_loader_test = data.DataLoader(
        testset, 
        **opt['dataloader']['test'],
        shuffle=False,
    )
    #创建输出目录
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    #log记录
    if opt['test']['use_hr']:
        logger_name = opt['stage']
        utils_logger.logger_info(logger_name, os.path.join(args.output_path, logger_name+'.log'), mode='w')
        logger = logging.getLogger(logger_name)
        p = 0
        s = 0    
        count = 0

    # Start testing
    #创建测试循环，将模式设置为评估模式
    
    model.eval()
    for batch in data_loader_test:
        #对批处理中提取LR和文件名fn
        lr = batch['lr']
        fn = batch['fn'][0]
        #设置use_hr则提取HR真图像
        if opt['test']['use_hr']:
            hr = batch['hr']

        #调用model将LR生成SR
        lr = lr.to('cuda')
        with torch.no_grad():
            sr = model(lr)
        #处理SR，从计算图中分离，转化为numpy，剪切有效像素范围，并进行颜色图像空间转换，并使用文件名保存到输出目录中
        #transpose(1, 2, 0)（channels,imagesize,imagesize）转化为（imagesize,imagesize,channels）
        sr = sr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
        sr = sr * 255.
        sr = np.clip(sr.round(), 0, 255).astype(np.uint8)
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_path, fn), sr)
        #如果设置了use_hr则计算psnr和SSIm指标
        if opt['test']['use_hr']:
            hr = hr.squeeze(0).numpy().transpose(1, 2, 0)
            hr = hr * 255.
            hr = np.clip(hr.round(), 0, 255).astype(np.uint8)
            hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
            
            psnr = util_calculate_psnr_ssim.calculate_psnr(sr, hr, crop_border=4, test_y_channel=True)
            ssim = util_calculate_psnr_ssim.calculate_ssim(sr, hr, crop_border=4, test_y_channel=True)
            p += psnr
            s += ssim
            count += 1

            logger.info('{}: {}, {}'.format(fn, psnr, ssim))

    if opt['test']['use_hr']:
        p /= count
        s /= count
        logger.info("Avg psnr: {}. ssim: {}. count: {}.".format(p, s, count))

    print('Testing finished!')
