import numpy as np
import os
from torch.utils.data import Dataset
import imageio.v3 as imageio
import random
import einops
import torch
import torch.nn.functional as F
from torch.fft import fft2,ifft2

class HCInewDataset(Dataset):
    """
    return: u v h w c
    """
    def __init__(self, opt):
        super().__init__()

        self.iters_in_one_epoch = opt['iters_in_one_epoch']  # 设为1000
        self.patch_size = opt['patch_size'] # 设为48或32
        self.img_list = self._load_imgs(opt['path']) # 路径 
        self.traindata_num = len(self.img_list)

    def __getitem__(self, index):
        x = self.img_list[index%self.traindata_num]
        an2, img_h, img_w, _ = x.shape
        start_h = random.randint(0, img_h-self.patch_size)
        start_w = random.randint(0, img_w-self.patch_size)
        train_data = x[:, start_h:start_h+self.patch_size, start_w:start_w+self.patch_size, :]
        train_data = einops.rearrange(train_data, '(u v) h w c -> h w u v c', u=9)
        train_data = einops.rearrange(train_data, 'h w u v c -> (u h) (v w) c', u=9)
        if random.random() < 0.5:
            train_data = train_data[::-1, ...]
        if random.random() < 0.5:
            train_data = train_data[:, ::-1, :]
        if random.random() < 0.5:
            train_data = train_data.transpose(1, 0, 2)
        orientation_rand = random.randint(0, 4)
        if orientation_rand == 0:
            train_data = np.rot90(train_data, 1)
        if orientation_rand == 1:
            train_data = np.rot90(train_data, 2)
        if orientation_rand == 2:
            train_data = np.rot90(train_data, 3)
        train_data = einops.rearrange(train_data, '(u h) (v w) c ->u v h w c', u=9, v=9)
        train_data = train_data.copy()

        '''随机高斯噪声'''
        # gauss = np.random.normal(0.0, 5/255+random.random()*45/255, train_data.shape)
        '''固定高斯噪声'''
        gauss = np.random.normal(0.0, 20/255, train_data.shape)
        train_data = train_data +gauss

        return train_data
        
    def __len__(self):
        return self.iters_in_one_epoch

    def _load_imgs(self, dataset_path):

        name_list = [
            'antinous',
            'boardgames',
            'dishes',
            'greek',
            'medieval2',
            'pens',
            'pillows', 
            'platonic',
            'rosemary',
            'table',
            'tomb',
            'tower',
            'town',
            'bedroom',
            'bicycle',
            'herbs',
            'origami',
            'boxes',
            'cotton',
            'dino', 
            'sideboard', 
        ]

        img_list = []
        for name in name_list:
            print(name)
            lf_list = []
            for i in range(81):
                tmp = imageio.imread(f'{dataset_path}/{name}/input_Cam{i:03}.png')  # load LF images(9x9)
                lf_list.append(tmp)
                img = np.stack(lf_list, 0) # n h w c
            # img = imageio.imread(f'{dataset_path}/{name}')
            img = img/255
            img = np.float32(img)
            img_list.append(img)
            
        return img_list


class LFMDataset(Dataset):
    """
    return: u v h w c
    """
    def __init__(self, opt):
        super().__init__()

        self.iters_in_one_epoch = opt['iters_in_one_epoch']
        self.patch_size = opt['patch_size']
        self.img_list = self._load_imgs(opt['path'])
        self.traindata_num = len(self.img_list)
        # self.psfs = self._load_psfs("../../../datasets/PSF/")

    def __getitem__(self, index):
        x = self.img_list[index%self.traindata_num]
        an2, img_h, img_w = x.shape
        start_h = random.randint(0, img_h-self.patch_size)
        start_w = random.randint(0, img_w-self.patch_size)
        train_data = x[:, start_h:start_h+self.patch_size, start_w:start_w+self.patch_size]
        train_data = train_data + np.random.normal(0, 0.04, train_data.shape)
        
        return train_data
        
    def __len__(self):
        return self.iters_in_one_epoch


    
    def _load_imgs(self, dataset_path):

        # name_list = [
        #     'B_cell.tif'
        # ]
        name_list = os.listdir(dataset_path)
        print(name_list)
        img_list = []
        for name in name_list:
            tmp = imageio.imread(f'{dataset_path}/{name}')  # load LF images(9x9)
            tmp = tmp.astype(np.float64)
            tmp = tmp *1000
            img_list.append(tmp)
        return img_list

