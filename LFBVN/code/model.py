import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import random
import utils
import imageio.v3 as imageio
from torch.nn.functional import l1_loss,relu
from torch.fft import fft2



class ConvBn(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding='same', dilation=1):
        super(ConvBn, self).__init__()
        if padding == 'same':
            pad = int((kernel_size-1)/2)*dilation
        else:
            pad = padding
        self.bone = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad, dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.bone(x)
        return x

class ResBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.bone = nn.Sequential(
            ConvBn(in_channel, out_channel, kernel_size, stride, dilation=dilation),
            nn.LeakyReLU(inplace=True),
            ConvBn(out_channel, out_channel, kernel_size, 1, dilation=dilation),
        )
        self.shoutcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shoutcut = nn.Sequential(
                ConvBn(in_channel, out_channel, 1, stride, dilation=dilation)
            )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.bone(x)
        out = out + self.shoutcut(x)
        out = self.relu(out)
        return out

def make_layer(block, p, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(p))
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(fn, fn, 3, padding=1)
        self.conv2 = nn.Conv2d(fn, fn, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(fn)
        self.norm2 = nn.BatchNorm2d(fn)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        return identity + out

class ModulateConv2d(nn.Module):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super(ModulateConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, x, h, w):

        Unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation)
        x_unfold = Unfold(x)
        x_unfold_modulated = x_unfold
        Fold = nn.Fold(output_size=(h, w), kernel_size=1, stride=1)
        out = Fold(x_unfold_modulated)  
        out = rearrange(out, 'b (c n) h w -> b c n h w', n=self.kernel_size**2)
        return out

class UNetFeature(nn.Module):
    def __init__(self, in_planes, out_planes, hidden):  # batch_size, c, d, h, w
        super(UNetFeature, self).__init__()
        self.feature = nn.Sequential()
        c = hidden
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, c, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            ConvBn(c, 2*c, 3, 2),
            nn.LeakyReLU(),
            ConvBn(2*c, 2*c, 3),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            ConvBn(2*c, 4*c, 3, 2),
            nn.LeakyReLU(),
            ConvBn(4*c, 4*c, 3),
            nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            ConvBn(4*c, 8*c, 3, 2),
            nn.LeakyReLU(),
            ConvBn(8*c, 8*c, 3),
            nn.LeakyReLU(),
        )
        self.conv5 = nn.Conv2d(8*c, 4*c, 3, 1, 1)
        self.conv6 = nn.Conv2d(4*c, 2*c, 3, 1, 1)
        self.conv7 = nn.Conv2d(2*c, c, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(2*c, c, 2, stride=2),
            nn.BatchNorm2d(c),
            nn.LeakyReLU())
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(4*c, 2*c, 2, stride=2),
            nn.BatchNorm2d(2*c),
            nn.LeakyReLU())
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(8*c, 4*c, 2, stride=2),
            nn.BatchNorm2d(4*c),
            nn.LeakyReLU())
        self.head1 = nn.Sequential(
            nn.Conv2d(c, out_planes, 3, padding=1)
        )

    def forward(self, x):  # batch_size, c, d, h, w
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = torch.cat([x3, self.up3(x4)], 1)
        x5 = self.conv5(x5)
        x6 = torch.cat([x2, self.up2(x5)], 1)
        x6 = self.conv6(x6)
        x7 = torch.cat([x1, self.up1(x6)], 1)
        x7 = self.conv7(x7)
        out = self.head1(x7)
        return out   # batch_size, 1, d, h, w

class BuildCost(nn.Module):
    '''input: b c n h w  --> output: b c n d h w'''
    def __init__(self, angRes, mindisp, maxdisp):
        super(BuildCost, self).__init__()
        self.angRes = angRes
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        self.oacc = ModulateConv2d(kernel_size=angRes, stride=1)

    def forward(self, x):
        b, c, n, h, w = x.shape
        x = rearrange(x, 'b c (a1 a2) h w -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes)
        bdr = (self.angRes // 2) * self.maxdisp
        pad = nn.ZeroPad2d((bdr, bdr, bdr, bdr))
        x_pad = pad(x)
        x_pad = rearrange(x_pad, '(b a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        h_pad, w_pad = h + 2 * bdr, w + 2 * bdr
        cost = []
        for d in range(self.mindisp, self.maxdisp + 1):
            dila = [h_pad - d, w_pad - d]
            self.oacc.dilation = dila
            crop = (self.angRes // 2) * (d - self.mindisp)
            if d == self.mindisp:
                feat = x_pad
            else:
                feat = x_pad[:, :, crop: -crop, crop: -crop]
            current_cost = self.oacc(feat, h, w)  # b c*n h w
            cost.append(current_cost)
        cost = torch.stack(cost, dim=3)
        return cost
    

class ConvBn3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding='same'):
        super(ConvBn3d, self).__init__()
        # padding = int((kernel_size-1)/2+stride-1)
        self.bone = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_planes)
        )

    def forward(self, x):
        return self.bone(x)


class Basic(nn.Module):
    def __init__(self, in_planes, hidden=150, out=3):  # batch_size, c, d, h, w
        super(Basic, self).__init__()
        feature = hidden
        self.res0 = nn.Sequential(
            ConvBn3d(in_planes, feature, 3, 1),
            nn.LeakyReLU(),
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU()
        )
        self.res1 = nn.Sequential(
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU(),
            ConvBn3d(feature, feature, 3, 1)
        )
        self.res2 = nn.Sequential(
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU(),
            ConvBn3d(feature, feature, 3, 1)
        )
        self.last = nn.Sequential(
            ConvBn3d(feature, feature, 3, 1),
            nn.LeakyReLU(),
            nn.Conv3d(feature, out, 3, 1, padding=1, bias=False)
        )

    def forward(self, cv):  # batch_size, c, d, h, w
        cost = self.res0(cv)
        res = self.res1(cost)
        cost = cost + res
        res = self.res2(cost)
        cost = cost + res
        cost = self.last(cost)
        return cost   # batch_size, 1, d, h, w


class Decoder(nn.Module):   
    def __init__(self, c, outc=3):
        super().__init__()
        self.bone = nn.Sequential(
            ResBlock(c, c),
            ResBlock(c, c),
            ResBlock(c, 64),
            nn.Conv2d(64, outc, 3, 1, 1)
        )

    def forward(self, x):
        x = self.bone(x)
        return x

class LFBSN_Base(nn.Module):   
    def __init__(self, opt):
        super().__init__()
        self.an = 7
        self.mask, self.train_index, self.out_index = utils.gen_mask(self.an)
        self.unet = UNetFeature(len(self.train_index)*3*9, 64, 64)
        self.decoder_list = nn.ModuleList([Decoder(64) for i in range(len(self.out_index))])
        self.build_psv = BuildCost(self.an, -4, 4)
        self.depth = Basic(3*len(self.train_index), 100, 1)
        self.softmax = nn.Softmax(dim=2) 

    def forward(self, x, epoch=None):
        device = x.get_device()
        xx_raw = x[:, :self.an, :self.an, ...]
        xx_raw = rearrange(xx_raw, 'b u v h w c -> b c (u h) (v w)')
        denoise_all = torch.zeros_like(xx_raw).to(device)
        denoise_all = rearrange(denoise_all, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        x = rearrange(xx_raw, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        label = x[:,:, self.out_index, ...]
        psv = self.build_psv(x)  # b c n d h w
        psv_mask = psv[:, :, self.train_index, ...].clone()
        psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
        volume = self.unet(psv_mask)

        out_list = []
        for i in range(len(self.out_index)):
            out_list.append(self.decoder_list[i](volume))
        views0 = torch.stack(out_list, 2)  # b c n h w
        denoise_all[:, :, self.out_index, :, :] = views0


        # with torch.no_grad():
            # x = torch.rot90(xx_raw, 1, (2, 3))
            # x = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
            # psv = self.build_psv(x)  # b c n d h w
            # psv_mask = psv[:, :, self.train_index, ...].clone()
            # psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
            # volume90 = self.unet(psv_mask)
            # out_list = []
            # for i in range(len(self.out_index)):
            #     out_list.append(self.decoder_list[i](volume90))
            # views90 = torch.stack(out_list, 2)  # b
            # denoise_all = rearrange(denoise_all, 'b c (u v) h w -> b c (u h) (v w)', u=self.an,v=self.an)
            # denoise_all = torch.rot90(denoise_all, 1, (2, 3))
            # denoise_all = rearrange(denoise_all, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
            # denoise_all[:, :, self.out_index, :, :] = views90

            # x = torch.rot90(xx_raw, 2, (2, 3))
            # x = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
            # psv = self.build_psv(x)  # b c n d h w
            # psv_mask = psv[:, :, self.train_index, ...].clone()
            # psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
            # volume180 = self.unet(psv_mask)
            # out_list = []
            # for i in range(len(self.out_index)):
            #     out_list.append(self.decoder_list[i](volume180))
            # views180 = torch.stack(out_list, 2)  # b
            # denoise_all = rearrange(denoise_all, 'b c (u v) h w -> b c (u h) (v w)', u=self.an,v=self.an)
            # denoise_all = torch.rot90(denoise_all, 1, (2, 3))
            # denoise_all = rearrange(denoise_all, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
            # denoise_all[:, :, self.out_index, :, :] = views180

            # x = torch.rot90(xx_raw, 3, (2, 3))
            # x = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
            # psv = self.build_psv(x)  # b c n d h w
            # psv_mask = psv[:, :, self.train_index, ...].clone()
            # psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
            # volume270 = self.unet(psv_mask)
            # out_list = []
            # for i in range(len(self.out_index)):
            #     out_list.append(self.decoder_list[i](volume270))
            # views270 = torch.stack(out_list, 2)  # b
            # denoise_all = rearrange(denoise_all, 'b c (u v) h w -> b c (u h) (v w)', u=self.an,v=self.an)
            # denoise_all = torch.rot90(denoise_all, 1, (2, 3))
            # denoise_all = rearrange(denoise_all, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
            # denoise_all[:, :, self.out_index, :, :] = views270

            # denoise_all = rearrange(denoise_all, 'b c (u v) h w -> b c (u h) (v w)', u=self.an,v=self.an)
            # denoise_all = torch.rot90(denoise_all, 1, (2, 3))
            # denoise_all = rearrange(denoise_all, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
            # denoise_all = denoise_all.clone()
        # return denoise_all
        loss = torch.mean(torch.abs(views0-label)[..., 8:-8, 8:-8])
        return loss
        # return rearrange(views0, 'b c n h w -> b n c h w')
        # return pred_center.unsqueeze(1)


class LFBSN(nn.Module):   
    def __init__(self, opt):
        super().__init__()
        self.an = 7
        self.mask, self.train_index, self.out_index = utils.gen_mask(self.an)
        self.unet = UNetFeature(len(self.train_index)*3*9, 64, 64)
        self.decoder_list = nn.ModuleList([Decoder(64) for i in range(len(self.out_index))])
        self.build_psv = BuildCost(self.an, -4, 4)
        self.depth = Basic(3*len(self.train_index), 100, 1)
        self.softmax = nn.Softmax(dim=2) 
        self.stage1 = 200
    def forward(self, x, epoch=0):
        device = x.get_device()
        xx_raw = x[:, :self.an, :self.an, ...]
        xx_raw = rearrange(xx_raw, 'b u v h w c -> b c (u h) (v w)')
        x = rearrange(xx_raw, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        label = x[:,:, self.out_index, ...]
        center = x[:, :, self.an*self.an//2, ...]
        psv = self.build_psv(x)  # b c n d h w
        refocus = torch.mean(psv, 2)  # b c d h w
        psv_mask = psv[:, :, self.train_index, ...].clone()
        depth = self.depth(rearrange(psv_mask, 'b c n d h w -> b (c n) d h w')) # b 1 d h w
        depth = self.softmax(depth)
        pred_center = torch.sum(depth * refocus, 2).squeeze()
        psv_mask = psv_mask * depth.unsqueeze(1) 
        psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
        volume = self.unet(psv_mask)

        out_list = []
        for i in range(len(self.out_index)):
            out_list.append(self.decoder_list[i](volume))
        views0 = torch.stack(out_list, 2)  # b c n h w
        if epoch > self.stage1:
            with torch.no_grad():
                x = torch.rot90(xx_raw, 1, (2, 3))
                x = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
                psv = self.build_psv(x)  # b c n d h w
                refocus90 = torch.mean(psv, 2)  # b c d h w
                psv_mask = psv[:, :, self.train_index, ...].clone()
                depth = self.depth(rearrange(psv_mask, 'b c n d h w -> b (c n) d h w')) # b 1 d h w
                depth = self.softmax(depth)
                pred_center = torch.sum(depth * refocus90, 2).squeeze()
                psv_mask = psv_mask * depth.unsqueeze(1) 
                psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
                volume90 = self.unet(psv_mask)

                x = torch.rot90(xx_raw, 2, (2, 3))
                x = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
                psv = self.build_psv(x)  # b c n d h w
                refocus180 = torch.mean(psv, 2)  # b c d h w
                psv_mask = psv[:, :, self.train_index, ...].clone()
                depth = self.depth(rearrange(psv_mask, 'b c n d h w -> b (c n) d h w')) # b 1 d h w
                depth = self.softmax(depth)
                pred_center = torch.sum(depth * refocus180, 2).squeeze()
                psv_mask = psv_mask * depth.unsqueeze(1) 
                psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
                volume180 = self.unet(psv_mask)

                x = torch.rot90(xx_raw, 3, (2, 3))
                x = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
                psv = self.build_psv(x)  # b c n d h w
                refocus270 = torch.mean(psv, 2)  # b c d h w
                psv_mask = psv[:, :, self.train_index, ...].clone()
                depth = self.depth(rearrange(psv_mask, 'b c n d h w -> b (c n) d h w')) # b 1 d h w
                depth = self.softmax(depth)
                pred_center = torch.sum(depth * refocus270, 2).squeeze()
                psv_mask = psv_mask * depth.unsqueeze(1) 
                psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
                volume270 = self.unet(psv_mask)

            volume90 = torch.rot90(volume90, 3, (2, 3))
            volume180 = torch.rot90(volume180, 2, (2, 3))
            volume270 = torch.rot90(volume270, 1, (2, 3))

            loss = torch.mean(torch.abs(views0-label)[..., 8:-8, 8:-8]) + torch.mean(torch.abs(pred_center-center)[..., 8:-8, 8:-8])*0.1 + \
            torch.mean(torch.abs(volume-volume90)[..., 8:-8, 8:-8]+torch.abs(volume-volume180)[..., 8:-8, 8:-8]+torch.abs(volume-volume270)[..., 8:-8, 8:-8])*0.3
        
        else:
            loss = torch.mean(torch.abs(views0-label)[..., 8:-8, 8:-8]) + torch.mean(torch.abs(pred_center-center)[..., 8:-8, 8:-8])*0.1
        return loss

class LFBSN_All(nn.Module):   
    def __init__(self, opt):
        super().__init__()
        self.an = 7
        self.mask, self.train_index, self.out_index = utils.gen_mask(self.an)
        self.unet = UNetFeature(len(self.train_index)*3*9, 64, 64)
        self.decoder_list = nn.ModuleList([Decoder(64) for i in range(len(self.out_index))])
        self.build_psv = BuildCost(self.an, -4, 4)
        self.depth = Basic(3*len(self.train_index), 100, 1)
        self.softmax = nn.Softmax(dim=2) 

    def forward(self, x):
        device = x.get_device()
        xx_raw = x[:, :self.an, :self.an, ...]
        xx_raw = rearrange(xx_raw, 'b u v h w c -> b c (u h) (v w)')
        denoise_all = torch.zeros_like(xx_raw).to(device)
        denoise_all = rearrange(denoise_all, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        x = rearrange(xx_raw, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        label = x[:,:, self.out_index, ...]
        center = x[:, :, self.an*self.an//2, ...]
        psv = self.build_psv(x)  # b c n d h w
        refocus = torch.mean(psv, 2)  # b c d h w
        psv_mask = psv[:, :, self.train_index, ...].clone()
        depth = self.depth(rearrange(psv_mask, 'b c n d h w -> b (c n) d h w')) # b 1 d h w
        depth = self.softmax(depth)
        
        print(depth.shape)
        depth_ = depth * repeat(torch.arange(-4, 5, 1), 'c -> a b c d e', a=1, b=1, d=1, e=1).to(device)
        depth_ = torch.sum(depth_, 2).squeeze()
        # np.save('./sideboard', depth_.cpu().numpy())

        pred_center = torch.sum(depth * refocus, 2).squeeze()
        psv_mask = psv_mask * depth.unsqueeze(1) 
        psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
        volume = self.unet(psv_mask)

        out_list = []
        for i in range(len(self.out_index)):
            out_list.append(self.decoder_list[i](volume))
        views0 = torch.stack(out_list, 2)  # b c n h w
        denoise_all[:, :, self.out_index, :, :] = views0

        x = torch.rot90(xx_raw, 1, (2, 3))
        x = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        psv = self.build_psv(x)  # b c n d h w
        refocus90 = torch.mean(psv, 2)  # b c d h w
        psv_mask = psv[:, :, self.train_index, ...].clone()
        depth = self.depth(rearrange(psv_mask, 'b c n d h w -> b (c n) d h w')) # b 1 d h w
        depth = self.softmax(depth)
        pred_center = torch.sum(depth * refocus90, 2).squeeze()
        psv_mask = psv_mask * depth.unsqueeze(1) 
        psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
        volume90 = self.unet(psv_mask)
        out_list = []
        for i in range(len(self.out_index)):
            out_list.append(self.decoder_list[i](volume90))
        views90 = torch.stack(out_list, 2)  # b
        denoise_all = rearrange(denoise_all, 'b c (u v) h w -> b c (u h) (v w)', u=self.an,v=self.an)
        denoise_all = torch.rot90(denoise_all, 1, (2, 3))
        denoise_all = rearrange(denoise_all, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        denoise_all[:, :, self.out_index, :, :] = views90

        x = torch.rot90(xx_raw, 2, (2, 3))
        x = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        psv = self.build_psv(x)  # b c n d h w
        refocus180 = torch.mean(psv, 2)  # b c d h w
        psv_mask = psv[:, :, self.train_index, ...].clone()
        depth = self.depth(rearrange(psv_mask, 'b c n d h w -> b (c n) d h w')) # b 1 d h w
        depth = self.softmax(depth)
        pred_center = torch.sum(depth * refocus180, 2).squeeze()
        psv_mask = psv_mask * depth.unsqueeze(1) 
        psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
        volume180 = self.unet(psv_mask)
        out_list = []
        for i in range(len(self.out_index)):
            out_list.append(self.decoder_list[i](volume180))
        views180 = torch.stack(out_list, 2)  # b
        denoise_all = rearrange(denoise_all, 'b c (u v) h w -> b c (u h) (v w)', u=self.an,v=self.an)
        denoise_all = torch.rot90(denoise_all, 1, (2, 3))
        denoise_all = rearrange(denoise_all, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        denoise_all[:, :, self.out_index, :, :] = views180

        x = torch.rot90(xx_raw, 3, (2, 3))
        x = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        psv = self.build_psv(x)  # b c n d h w
        refocus270 = torch.mean(psv, 2)  # b c d h w
        psv_mask = psv[:, :, self.train_index, ...].clone()
        depth = self.depth(rearrange(psv_mask, 'b c n d h w -> b (c n) d h w')) # b 1 d h w
        depth = self.softmax(depth)
        pred_center = torch.sum(depth * refocus270, 2).squeeze()
        psv_mask = psv_mask * depth.unsqueeze(1) 
        psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
        volume270 = self.unet(psv_mask)
        out_list = []
        for i in range(len(self.out_index)):
            out_list.append(self.decoder_list[i](volume270))
        views270 = torch.stack(out_list, 2)  # b
        denoise_all = rearrange(denoise_all, 'b c (u v) h w -> b c (u h) (v w)', u=self.an,v=self.an)
        denoise_all = torch.rot90(denoise_all, 1, (2, 3))
        denoise_all = rearrange(denoise_all, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        denoise_all[:, :, self.out_index, :, :] = views270

        denoise_all = rearrange(denoise_all, 'b c (u v) h w -> b c (u h) (v w)', u=self.an,v=self.an)
        denoise_all = torch.rot90(denoise_all, 1, (2, 3))
        denoise_all = rearrange(denoise_all, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        denoise_all = denoise_all.clone()
        return denoise_all

class LFBSN_Eval(nn.Module):   
    def __init__(self, opt):
        super().__init__()
        self.an = 7
        self.mask, self.train_index, self.out_index = utils.gen_mask(self.an)
        self.unet = UNetFeature(len(self.train_index)*3*9, 64, 64)
        self.decoder_list = nn.ModuleList([Decoder(64) for i in range(len(self.out_index))])
        self.build_psv = BuildCost(self.an, -4, 4)
        self.depth = Basic(3*len(self.train_index), 100, 1)
        self.softmax = nn.Softmax(dim=2) 

    def forward(self, x):
        device = x.get_device()
        xx_raw = x[:, :self.an, :self.an, ...]
        xx_raw = rearrange(xx_raw, 'b u v h w c -> b c (u h) (v w)')
        x = rearrange(xx_raw, 'b c (u h) (v w) -> b c (u v) h w', u=self.an,v=self.an)
        label = x[:,:, self.out_index, ...]
        center = x[:, :, self.an*self.an//2, ...]
        psv = self.build_psv(x)  # b c n d h w
        refocus = torch.mean(psv, 2)  # b c d h w
        psv_mask = psv[:, :, self.train_index, ...].clone()
        depth = self.depth(rearrange(psv_mask, 'b c n d h w -> b (c n) d h w')) # b 1 d h w
        depth = self.softmax(depth)
        
        depth_ = depth * repeat(torch.arange(-4, 5, 1), 'c -> a b c d e', a=1, b=1, d=1, e=1).to(device)
        depth_ = torch.sum(depth_, 2).squeeze()

        pred_center = torch.sum(depth * refocus, 2).squeeze()
        psv_mask = psv_mask * depth.unsqueeze(1) 
        psv_mask = rearrange(psv_mask, 'b c n d h w -> b (c n d) h w')
        volume = self.unet(psv_mask)

        out_list = []
        for i in range(len(self.out_index)):
            out_list.append(self.decoder_list[i](volume))
        views0 = torch.stack(out_list, 2)  # b c n h w
        return rearrange(views0, 'b c n h w -> b n c h w')




