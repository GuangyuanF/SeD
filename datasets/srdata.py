import os
import glob
import random
import numpy as np
import torch.utils.data as data
import cv2
from torchvision.transforms import ToTensor


class Train(data.Dataset):
    def __init__(self, scale=4, patch_size=64, data_root='./DF2K'):
        #scale: 上采样因子
        self.scale = scale

        self.patch_size = patch_size

        self.dir_hr = os.path.join(data_root, 'train_sub')
        self.dir_lr = os.path.join(data_root, 'train_sub_x4')
        # glob 模块创建高分辨率(self.images_hr)和低分辨率(self.images_lr)图像的文件路径列表。
        self.images_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*.png'))
        )
        self.images_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*.png'))
        )

    def __getitem__(self, idx):
        #cv2.imread 加载高分辨率和低分辨率图像,并使用 cv2.cvtColor 将颜色空间从BGR转换为RGB
        filename = os.path.basename(self.images_hr[idx])
        hr = cv2.imread(self.images_hr[idx])  # BGR, n_channels=3
        lr = cv2.imread(self.images_lr[idx])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)  # RGB, n_channels=3
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        #在低分辨率图像中随机选择一个作物位置(croph, cropw),确保作物大小为 patch_size。
        h, w, _ = lr.shape
        croph = np.random.randint(0, h - self.patch_size + 1)
        cropw = np.random.randint(0, w - self.patch_size + 1)
        
        hr = hr[croph*self.scale: croph*self.scale+self.patch_size*self.scale, cropw*self.scale: cropw*self.scale+self.patch_size*self.scale, :]
        lr = lr[croph: croph+self.patch_size, cropw: cropw+self.patch_size, :]

        mode = random.randint(0, 7)
        #augment_img对低分辨率和高分辨率块应用随机数据增强。
        lr, hr = augment_img(lr, mode=mode), augment_img(hr, mode=mode)

        lr = ToTensor()(lr.copy())
        hr = ToTensor()(hr.copy())
        
        return lr, hr, filename

    def __len__(self):
        return len(self.images_hr)


class Test(data.Dataset):
    def __init__(self, data_root='./Evaluation'):

        self.dir_hr = os.path.join(data_root, 'Set5/GTmod12')
        self.dir_lr = os.path.join(data_root, 'Set5/LRbicx4')
        self.images_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*.png'))
        )
        self.images_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*.png'))
        )

    def __getitem__(self, idx):

        filename = os.path.basename(self.images_hr[idx])
        hr = cv2.imread(self.images_hr[idx])  # BGR, n_channels=3
        lr = cv2.imread(self.images_lr[idx])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)  # RGB, n_channels=3
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        lr = ToTensor()(lr.copy())
        hr = ToTensor()(hr.copy())

        return lr, hr, filename

    def __len__(self):
        return len(self.images_hr)

"""
ugment_img函数是一个用于对图像进行数据增强的实用函数。它接受两个参数:

img: 输入图像为NumPy数组。
mode(可选): 一个0到7之间的整数值,用于指定要应用的增强类型。默认值为0,返回原始图像而不做任何增强。
根据 mode 值,该函数对输入图像应用不同的变换:

mode=0: 返回原始图像,不做任何增强。
mode=1: 垂直翻转图像,然后逆时针旋转90度。
mode=2: 垂直翻转图像。
mode=3: 逆时针旋转图像270度(等同于顺时针旋转90度)。
mode=4: 垂直翻转图像,然后旋转180度。
mode=5: 逆时针旋转图像90度。
mode=6: 旋转图像180度。
mode=7: 垂直翻转图像,然后逆时针旋转270度(等同于顺时针旋转90度)。
"""
def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
