import cv2
import numpy as np
import torch
from einops import rearrange, repeat
import os
import model as mymodel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import imageio.v2 as imageio
import utils
import datetime
from thop import profile, clever_format

def salt_and_pepper_noise(
    data: np.ndarray,
    noise_ratio: float = 0.05,
    salt_pepper_ratio: float = 0.5,
    salt_value: float = 1.0,
    pepper_value: float = 0.0,
    seed: int = None,
    mask: np.ndarray = None,
    clip_range: tuple = None
) -> np.ndarray:
    """
    椒盐噪声生成函数：随机将部分像素设为最大值（盐噪声）或最小值（椒噪声）
    :param data: 输入数据（numpy数组，支持向量、矩阵、图像张量）
    :param noise_ratio: 噪声占比（0~1，总噪声像素数占总像素数的比例）
    :param salt_pepper_ratio: 盐噪声占总噪声的比例（0~1，剩余为椒噪声）
    :param salt_value: 盐噪声值（默认255，适用于图像）
    :param pepper_value: 椒噪声值（默认0，适用于图像）
    :param seed: 随机种子（确保结果可复现）
    :param mask: 掩码矩阵（与data同shape，True/1表示该位置可添加噪声）
    :param clip_range: 编码后数据裁剪范围（如(0,255)，避免溢出）
    :return: 带椒盐噪声的数据（与输入同shape、同dtype）
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = data.copy()  # 避免修改原始数据
    total_pixels = data.size  # 总像素数
    noise_pixels = int(total_pixels * noise_ratio)  # 噪声像素数
    salt_pixels = int(noise_pixels * salt_pepper_ratio)  # 盐噪声像素数
    pepper_pixels = noise_pixels - salt_pixels  # 椒噪声像素数
    
    # 生成随机索引
    indices = np.random.permutation(total_pixels)
    salt_indices = indices[:salt_pixels]
    pepper_indices = indices[salt_pixels:salt_pixels+pepper_pixels]
    
    # 应用掩码（仅在掩码为True的位置添加噪声）
    if mask is not None:
        mask = mask.astype(bool).ravel()  # 展平为一维掩码
        # 筛选掩码内的有效索引
        valid_salt_indices = salt_indices[mask[salt_indices]]
        valid_pepper_indices = pepper_indices[mask[pepper_indices]]
    else:
        valid_salt_indices = salt_indices
        valid_pepper_indices = pepper_indices
    
    # 添加盐噪声（最大值）和椒噪声（最小值）
    data_raveled = data.ravel()
    data_raveled[valid_salt_indices] = salt_value
    data_raveled[valid_pepper_indices] = pepper_value
    
    # 裁剪数据
    if clip_range is not None:
        data = np.clip(data, clip_range[0], clip_range[1])
    
    return data.astype(data.dtype)


def eval_HCInew(checkpoint_path, device, model, resultdir, sigma=20, type='g'):
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    dataset_path = "../../../dataset/hci_dataset/additional"
    # name_list = os.listdir(dataset_path)
    name_list = [
        'boxes',
        # 'cotton',
        # 'dino', 
        # 'sideboard', 
        # 'bedroom',
        # 'bicycle',
        # 'herbs',
        # 'origami',
        # 'kitchen'
    ]

    np.random.seed(42)
    an = 7
    model.eval()
    if not os.path.exists(f'../log/img/{resultdir}'):
        os.makedirs(f'../log/img/{resultdir}')

    mask, train_index, out_index = utils.gen_mask(an)
    p_list = [0 for i in range(len(out_index))]
    s_list = [0 for i in range(len(out_index))]
    for name in name_list:
        lf_list = []
        for i in range(81):
            tmp = imageio.imread(f'{dataset_path}/{name}/input_Cam{i:03}.png')  # load LF images(9x9)
            lf_list.append(tmp)
            img_read = np.stack(lf_list, 0) # n h w c   
        img_read = rearrange(img_read, '(u v) h w c -> u v h w c', u=9) 
        img_raw = img_read/255
        img_read = np.float32(img_raw)

        gauss = np.random.normal(0.0, sigma/255, img_read.shape)
        # gauss = np.random.uniform(-50/255, 50/255, img_read.shape)
        if type == 'g':
            img_read = img_read +gauss  #########################################
        if type == 'p':
            img_read  = np.random.poisson((sigma)*img_read) /(sigma)  ##############

        U,V,H,W,C = img_read.shape

        img_read = rearrange(img_read, 'u v h w c -> (u h) (v w) c')

        img_read = img_read
        img_read = rearrange(img_read, '(u h) (v w) c -> u v h w c', u=U,v=V)
        eval_data0 = img_read[np.newaxis, ...]
        eval_data0 = torch.from_numpy(eval_data0).to(device).float()


        with torch.no_grad():
            eval_data0 = model(eval_data0)
        eval_data0 = eval_data0.cpu().numpy()[0, ...]

        out_all = eval_data0
        out_all = out_all[:, :, 8:-8, 8:-8]
        img_raw = img_raw[:an, :an, ...]
        img_raw = rearrange(img_raw[:, :, 8:-8, 8:-8, :], 'u v h w c -> (u v) h w c')
        img_raw = img_raw[out_index]
        out_all = rearrange(out_all, 'n c h w -> n h w c')


        for i in range(len(out_index)):
            img = img_raw[i, ...]
            out = out_all[i, ...]
            psnr = peak_signal_noise_ratio(img, out, data_range=1)
            ssim = structural_similarity(img, out, data_range=1, channel_axis=2, gaussian_weights=True, sigma=1.5, win_size=11)
            print(psnr, ssim)
            p_list[i] += psnr
            s_list[i] += ssim
    p_list = [round(n/8, 4).item() for n in p_list]
    s_list = [round(n/8, 4).item() for n in s_list]
    print(np.average(p_list), np.average(s_list))


if __name__ == "__main__":


    ckp = '../log/ckp/LFBSN/1000g20_0.0745.pth'

    device = torch.device('cuda:0')
    model = mymodel.LFBSN_Eval(None)
    resultdir = 'LFBSN'

    eval_HCInew(ckp, device, model, resultdir, sigma=20, type='g')
