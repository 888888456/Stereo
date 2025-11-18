"""
ONNX兼容版本的相关性计算模块
针对 ONNX 1.10 / opset 12 / TensorRT 7.1.3 优化

主要改进:
1. 使用简化的 grid_sample (不依赖 align_corners)
2. 移除 CUDA 自定义算子依赖
3. 使用标准 PyTorch 操作确保 ONNX 可导出
"""
import torch
import torch.nn.functional as F


def bilinear_sampler_simple(img, coords):
    """
    简化的双线性采样器，专为 ONNX opset 12 优化
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    
    # 归一化到 [-1, 1]
    xgrid = 2.0 * xgrid / (W - 1.0) - 1.0
    ygrid = 2.0 * ygrid / (H - 1.0) - 1.0
    
    grid = torch.cat([xgrid, ygrid], dim=-1)
    
    # 使用最简单的参数组合，兼容 opset 12
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class CorrBlock1D_ONNX:
    """
    1D相关性块 - ONNX 兼容版本
    使用标准操作替代自定义CUDA算子
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # 计算所有对的相关性
        corr = CorrBlock1D_ONNX.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            
            # 创建采样偏移
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dx = dx.view(2*r+1, 1)
            
            # 计算采样位置
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / (2**i)
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0, y0], dim=-1)
            
            # 使用简化的双线性采样
            corr_sampled = bilinear_sampler_simple(corr, coords_lvl)
            corr_sampled = corr_sampled.view(batch, h1, w1, -1)
            out_pyramid.append(corr_sampled)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        """计算特征图之间的相关性"""
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        
        # 使用 einsum 计算相关性（ONNX 支持）
        corr = torch.einsum('bchw,bchv->bhwv', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        
        # 归一化
        return corr / torch.sqrt(torch.tensor(D, dtype=torch.float32))


class PytorchAlternateCorrBlock1D_ONNX:
    """
    交替相关性块 - ONNX 兼容版本
    移除 align_corners 依赖
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.fmap1 = fmap1
        self.fmap2 = fmap2

    def corr(self, fmap1, fmap2, coords):
        B, D, H, W = fmap2.shape
        
        # 归一化网格坐标到 [-1, 1]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2.0 * xgrid / (W - 1.0) - 1.0
        ygrid = 2.0 * ygrid / (H - 1.0) - 1.0

        grid = torch.cat([xgrid, ygrid], dim=-1)
        
        output_corr = []
        for grid_slice in grid.unbind(3):
            # 使用简化的 grid_sample 参数
            fmapw_mini = F.grid_sample(
                fmap2, grid_slice, 
                mode='bilinear', 
                padding_mode='zeros',
                align_corners=False  # 更好的 opset 12 兼容性
            )
            corr = torch.sum(fmapw_mini * fmap1, dim=1)
            output_corr.append(corr)
            
        corr = torch.stack(output_corr, dim=1).permute(0, 2, 3, 1)
        return corr / torch.sqrt(torch.tensor(D, dtype=torch.float32))

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        
        fmap1 = self.fmap1
        fmap2 = self.fmap2
        
        out_pyramid = []
        for i in range(self.num_levels):
            # 创建偏移网格
            dx = torch.zeros(1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            
            # 手动创建网格（避免 indexing 参数）
            dy_grid = dy.view(-1, 1).repeat(1, 1)
            dx_grid = dx.view(1, -1).repeat(2*r+1, 1)
            delta = torch.stack([dx_grid, dy_grid], dim=-1)
            
            # 计算采样位置
            centroid_lvl = coords.reshape(batch, h1, w1, 1, 2).clone()
            centroid_lvl[..., 0] = centroid_lvl[..., 0] / (2**i)
            coords_lvl = centroid_lvl + delta.view(-1, 2)
            
            # 计算相关性
            corr = self.corr(fmap1, fmap2, coords_lvl)
            
            # 下采样
            fmap2 = F.avg_pool2d(fmap2, [1, 2], stride=[1, 2])
            out_pyramid.append(corr)
            
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


# 导出接口 - 使用 ONNX 兼容版本
CorrBlock1D = CorrBlock1D_ONNX
PytorchAlternateCorrBlock1D = PytorchAlternateCorrBlock1D_ONNX

# CUDA 版本在 ONNX 导出时不可用，使用 Python 版本替代
CorrBlockFast1D = CorrBlock1D_ONNX

class AlternateCorrBlock:
    """占位符 - 不在 ONNX 导出中使用"""
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        raise NotImplementedError("AlternateCorrBlock requires CUDA and is not supported for ONNX export")
