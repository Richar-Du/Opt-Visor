from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)

def calculate_pooling_size(h, w, kernel_size):
    # 计算需要的填充
    # import ipdb; ipdb.set_trace()
    pad_h = (kernel_size - (h % kernel_size)) % kernel_size
    pad_w = (kernel_size - (w % kernel_size)) % kernel_size
    
    # 填充后的高度和宽度
    h_padded = h + pad_h
    w_padded = w + pad_w
    
    # 计算 pooling 后的大小
    output_height = h_padded // kernel_size
    output_width = w_padded // kernel_size
    
    return output_height * output_width

class MeanPoolingProjector(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, pool_after=False):
        super(MeanPoolingProjector, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel = kernel  # kernel同时用作length和stride

        # 参数: 输入特征维度*kernel*kernel，输出特征维度
        self.linear = build_mlp(2, input_dim, output_dim)
        self.pool_after = pool_after

    def forward(self, x, h_w=None):
        # print(x.shape)

        bs, hw, input_dim = x.shape
        if h_w:
            h, w = h_w[0], h_w[1]
        else:
            h = w = int((hw) ** 0.5)  # 假设h=w


        # 计算pad值以确保整除
        self.pad_h = 0
        self.pad_w = 0
        
        if h % self.kernel:
            self.pad_h = (self.kernel - h % self.kernel) % self.kernel

        if h_w:
            self.pad_w = (self.kernel - w % self.kernel) % self.kernel
        else:
            self.pad_w = self.pad_h
        # 重塑为(bs, h, w, input_dim)并进行必要的填充
        x = x.view(bs, h, w, input_dim)
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, 0, 0, self.pad_w, 0, self.pad_h), "constant", 0)

        if self.pool_after:
            x = self.linear(x)

        x = x.permute(0, 3, 1, 2)  # 调整为(bs, C, H, W)格式以适应unfold
        x_dim = x.shape[1]

        x_pooled = F.avg_pool2d(x, kernel_size=self.kernel, stride=self.kernel, padding=0)
        x_pooled = x_pooled.view(bs, x_dim, -1)
        # print(x_pooled.shape)

        x_pooled = x_pooled.permute(0, 2, 1)
        # print(x_pooled.shape)

        # 应用linear变换
        if not self.pool_after:
            x_pooled = self.linear(x_pooled)

        return x_pooled
    
class TemporalPoolingProjector(nn.Module):
    def __init__(self, input_dim, output_dim, spatial_kernel, pool_after=False):
        super(TemporalPoolingProjector, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spatial_kernel = spatial_kernel  # kernel同时用作length和stride

        # 参数: 输入特征维度*kernel*kernel，输出特征维度
        self.linear = build_mlp(2, input_dim, output_dim)
        self.pool_after = pool_after

    def forward(self, x, h_w=None, temp_kernel=None, aspect_ratio=None):       # 加一个参数说明是image还是video
        # print(x.shape)
        if aspect_ratio is not None:
            import ipdb
            ipdb.set_trace()

        t, hw, input_dim = x.shape
        if h_w:
            h, w = h_w[0], h_w[1]
        else:
            h = w = int((hw) ** 0.5)  # 假设h=w

        # 计算pad值以确保整除
        self.pad_h = 0
        self.pad_w = 0

        if h % self.spatial_kernel:
            self.pad_h = (self.spatial_kernel - h % self.spatial_kernel) % self.spatial_kernel
        if h_w:
            self.pad_w = (self.spatial_kernel - w % self.spatial_kernel) % self.spatial_kernel
        else:
            self.pad_w = self.pad_h

        # self.spatial_pad = 0
        # if h % self.spatial_kernel:
        #     self.spatial_pad = self.spatial_kernel - h % self.spatial_kernel
        self.pad_t = 0
        if t % temp_kernel:
            self.pad_t = temp_kernel - t % temp_kernel

        # 重塑为(t, h, w, input_dim)并进行必要的填充
        x = x.view(t, h, w, input_dim)
        if self.pad_h > 0 or self.pad_w > 0 or self.pad_t > 0:
            x = F.pad(x, (0, 0, 0, self.pad_w, 0, self.pad_h, 0, self.pad_t), "constant", 0)

        if self.pool_after:
            x = self.linear(x)

        x = x.unsqueeze(0)      # (T, H, W, C) -> (1, T, H, W, C)
        x = x.permute(0, 4, 1, 2, 3)  # 调整为(1, C, T, H, W)格式以适应unfold
        x_dim = x.shape[1]
        x_pooled = F.avg_pool3d(x, kernel_size=(temp_kernel, self.spatial_kernel, self.spatial_kernel), stride=(temp_kernel, self.spatial_kernel, self.spatial_kernel), padding=0)
        t_pooled = x_pooled.shape[2]
        x_pooled = x_pooled.view(1, x_dim, -1)      # (1, C, t_pooled, H, W) -> (1, C, t_pooled*H*W)
        x_pooled = x_pooled.permute(0, 2, 1)        # (1, C, t_pooled*H*W) -> (1, t_pooled*H*W, C)

        # 应用linear变换
        if not self.pool_after:
            x_pooled = self.linear(x_pooled)

        return x_pooled
