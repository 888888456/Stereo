"""
ONNX兼容版本的工具函数
针对 ONNX 1.10 / opset 12 / TensorRT 7.1.3 优化
"""
import torch
import torch.nn.functional as F
import numpy as np


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def bilinear_sampler_onnx_compatible(img, coords, mode='bilinear', mask=False):
    """
    兼容 opset 12 的 bilinear sampler
    
    主要改进:
    1. 移除 align_corners 的依赖（opset 12可能有问题）
    2. 使用更简单的 grid_sample 参数
    3. 确保所有操作都是 ONNX 可导出的
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    
    # 归一化到 [-1, 1]
    # 使用更安全的方式，避免除以0
    xgrid = 2.0 * xgrid / (W - 1.0) - 1.0
    if H > 1:
        ygrid = 2.0 * ygrid / (H - 1.0) - 1.0
    else:
        ygrid = torch.zeros_like(xgrid)
    
    grid = torch.cat([xgrid, ygrid], dim=-1)
    
    # 使用最简单的 grid_sample 参数组合
    # mode='bilinear', padding_mode='zeros' 是 opset 12 最安全的组合
    # 不使用 align_corners，这在旧版本中可能有兼容性问题
    img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros')
    
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    
    return img


def coords_grid(batch, ht, wd):
    """
    生成坐标网格 - ONNX 兼容版本
    使用 torch.arange 而不是 torch.meshgrid 的新版本语法
    """
    # 旧版 meshgrid 语法（更兼容 opset 12）
    y_coords = torch.arange(ht).float()
    x_coords = torch.arange(wd).float()
    
    # 手动创建网格以避免 indexing 参数问题
    y_grid = y_coords.view(-1, 1).repeat(1, wd)
    x_grid = x_coords.view(1, -1).repeat(ht, 1)
    
    # 堆叠为 [2, H, W]
    coords = torch.stack([x_grid, y_grid], dim=0)
    
    # 扩展 batch 维度
    return coords.unsqueeze(0).repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    """
    8倍上采样 - ONNX 兼容版本
    
    改进:
    1. 不使用 align_corners (opset 12 兼容性更好)
    2. 使用固定的 size 参数而不是 scale_factor
    """
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    # 移除 align_corners 参数以提高兼容性
    upsampled = F.interpolate(flow, size=new_size, mode=mode, align_corners=False)
    return 8.0 * upsampled


def gauss_blur(input, N=5, std=1):
    """高斯模糊 - 完全兼容 ONNX"""
    B, D, H, W = input.shape
    
    # 创建高斯核
    x = torch.arange(N).float() - N//2
    y = torch.arange(N).float() - N//2
    
    # 手动创建网格
    y_grid = y.view(-1, 1).repeat(1, N)
    x_grid = x.view(1, -1).repeat(N, 1)
    
    # 计算高斯权重
    unnormalized_gaussian = torch.exp(-(x_grid.pow(2) + y_grid.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / (unnormalized_gaussian.sum() + 1e-8)
    weights = weights.view(1, 1, N, N).to(input.device)
    
    # 应用卷积
    output = F.conv2d(input.reshape(B*D, 1, H, W), weights, padding=N//2)
    return output.view(B, D, H, W)


# 保持向后兼容的别名
bilinear_sampler = bilinear_sampler_onnx_compatible
