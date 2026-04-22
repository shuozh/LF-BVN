from collections import OrderedDict
import json
import datetime
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from torch.fft import fft2,ifft2


def date_time():
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
    return date_time

def log(log_file, str, also_print=True, with_time=True):
    with open(log_file, 'a+') as F:
        if with_time:
            F.write(date_time() + '  ')
        F.write(str)
    if also_print:
        if with_time:
            print(date_time(), end='  ')
        print(str, end='')

def parse(opt_path):
    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    return opt

def warp_all(disparity, in_put, device, an=9):
    '''
    已知中心视角视差，将所有视图反向warp到中心视角
    :param device:
    :param disparity: b, h, w
    :param in_put:  b, c, n, h, w
    :return: b, c, n, h, w
    '''
    b, c, n, h, w = in_put.shape
    an2 = an // 2
    
    xx = torch.arange(0, w).view(1, 1, 1, w).expand(b, n, h, w).float().to(device)
    yy = torch.arange(0, h).view(1, 1, h, 1).expand(b, n, h, w).float().to(device)
    zz = torch.arange(0, n).view(1, n, 1, 1).expand(b, n, h, w).float().to(device)
    for i in range(n):
        ind_h_source = i // an
        ind_w_source = i % an
        xx[:, i, ...] = xx[:, i, ...] + disparity*(an2-ind_w_source)
        yy[:, i, ...] = yy[:, i, ...] + disparity*(an2-ind_h_source)
    xx = xx*2/(w-1)-1
    yy = yy*2/(h-1)-1
    zz = zz*2/(n-1)-1
    grid = torch.stack([xx, yy, zz], dim=4)  # N, d, h, w, 3
    grid = grid.to(device)
    out_put = F.grid_sample(in_put, grid, align_corners=True)
    return out_put


def recursive_print(src, dpth=0, key=None):
    """ Recursively prints nested elements."""
    tabs = lambda n: ' ' * n * 4 # or 2 or 8 or...

    if isinstance(src, dict):
        if key is not None:
            print(tabs(dpth) + '%s: ' % (key))
        for key, value in src.items():
            recursive_print(value, dpth + 1, key)
    elif isinstance(src, list):
        if key is not None:
            print(tabs(dpth) + '%s: ' % (key))
        for litem in src:
            recursive_print(litem, dpth)
    else:
        if key is not None:
            print(tabs(dpth) + '%s: %s' % (key, src))


def recursive_log(log_file, src, dpth=0, key=None):
    """ Recursively prints nested elements."""
    tabs = lambda n: ' ' * n * 4 # or 2 or 8 or...

    if isinstance(src, dict):
        if key is not None:
            log(log_file, tabs(dpth) + '%s: \n' % (key), with_time=False)
        for key, value in src.items():
            recursive_log(log_file, value, dpth + 1, key)
    elif isinstance(src, list):
        if key is not None:
            log(log_file, tabs(dpth) + '%s: \n' % (key), with_time=False)
        for litem in src:
            recursive_log(log_file, litem, dpth)
    else:
        if key is not None:
            log(log_file, tabs(dpth) + '%s: %s\n' % (key, src), with_time=False)


def render_mpi(mpi, index_ref, index_target, disp_min, disp_max, step_interval, device):
    '''
    mpi: b c n h w
    index_ref: central view index
    '''
    b, c, n, h, w = mpi.shape    
    xx = torch.arange(0, w).view(1, 1, 1, w).expand(b, n, h, w).float().to(device)
    yy = torch.arange(0, h).view(1, 1, h, 1).expand(b, n, h, w).float().to(device)
    zz = torch.arange(0, n).view(1, n, 1, 1).expand(b, n, h, w).float().to(device)
    d = torch.arange(disp_min, disp_max+step_interval, step_interval).float().to(device)
    d = d.view(1, -1, 1, 1)
    disp = torch.ones_like(xx)* d
    ind_h_source,ind_w_source = index_target
    ind_h, ind_w = index_ref
    xx = xx + disp*(ind_w_source-ind_w)
    yy = yy + disp*(ind_h_source-ind_h)
    xx = xx*2/(w-1)-1
    yy = yy*2/(h-1)-1
    zz = zz*2/(n-1)-1
    grid = torch.stack([xx, yy, zz], dim=4)  # N, d, h, w, 3
    grid = grid.to(device)
    mpi = F.grid_sample(mpi, grid, align_corners=True)
    # out_put = torch.sum(mpi[:, :3, ...]* mpi[:, 3:, ...], 2)
    return mpi


def gen_cost_volume(input, device, dis_max, num, an=7):
    """
    生成cost volume
    :param input: batch_size, c, n, h, w
    :param device:
    :return: b, c, n, d, h, w
    """
    b, c, n, h, w = input.shape
    warped_all = []
    step = (dis_max*2)/(num-1)
    for d in [(i-(num-1)/2)*step for i in range(num)]:
        dis = torch.full((b, h, w), fill_value=d).float().to(device)
        warped = warp_all(dis, input, device, an)  # b, c, n, h, w
        warped_all.append(warped)
    cost_volume = torch.stack(warped_all, dim=3)
    return cost_volume

def gen_mask(an):
    train_index_list = []
    out_index_list = []
    mask = torch.ones((an, an))

    h = an//2
    w = an//2+1
    index = np.arange(h*w)
    if h % 2 == 0:
        index = index.reshape(h, w)
    else:
        index = index.reshape(w, h) 
        index = np.rot90(index, 1)
    for i in range(h):
        for j in range(w):
            i90 = h-(j-h)
            j90 = h+(i-h)
            i180 = h-(i-h)
            j180 = h-(j-h)
            i270 = h+(j-h)
            j270 = h-(i-h)
            if index[i, j] % 4 == 0:
                mask[i, j] = 0
            elif index[i, j] % 4 == 1:
                mask[i90, j90] = 0
            if index[i, j] % 4 == 2:
                mask[i180, j180] = 0
            if index[i, j] % 4 == 3:
                mask[i270, j270] = 0
    mask[h, h] = 0
    for i in range(an*an):
        if mask[i % an, i//an]==1:
            train_index_list.append(i)
        else:
            out_index_list.append(i)

    return mask, train_index_list, out_index_list


