"""
ONNX兼容版本的 RAFT-Stereo 主模型
针对 ONNX 1.10 / opset 12 / TensorRT 7.1.3 优化

主要改进:
1. 使用 ONNX 兼容的工具函数
2. 移除 CUDA 相关依赖
3. 简化 autocast 处理
4. 优化所有插值和采样操作
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update_onnx_compatible import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr_onnx_compatible import CorrBlock1D, PytorchAlternateCorrBlock1D


def coords_grid_onnx(batch, ht, wd, device):
    """
    生成坐标网格 - ONNX 兼容版本
    """
    y_coords = torch.arange(ht, dtype=torch.float32, device=device)
    x_coords = torch.arange(wd, dtype=torch.float32, device=device)
    
    # 手动创建网格
    y_grid = y_coords.view(-1, 1).repeat(1, wd)
    x_grid = x_coords.view(1, -1).repeat(ht, 1)
    
    coords = torch.stack([x_grid, y_grid], dim=0)
    return coords.unsqueeze(0).repeat(batch, 1, 1, 1)


def upflow8_onnx(flow, n_downsample=2):
    """
    上采样函数 - ONNX 兼容版本
    """
    factor = 2 ** n_downsample
    new_size = (factor * flow.shape[2], factor * flow.shape[3])
    # 使用 align_corners=False 以提高兼容性
    return factor * F.interpolate(flow, size=new_size, mode='bilinear', align_corners=False)


class RAFTStereoONNX(nn.Module):
    """
    RAFT-Stereo ONNX 兼容版本
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(
            output_dim=[args.hidden_dims, context_dims], 
            norm_fn=args.context_norm, 
            downsample=args.n_downsample
        )
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) 
            for i in range(self.args.n_gru_layers)
        ])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(
                output_dim=256, 
                norm_fn='instance', 
                downsample=args.n_downsample
            )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """初始化光流坐标"""
        N, _, H, W = img.shape
        coords0 = coords_grid_onnx(N, H, W, img.device)
        coords1 = coords_grid_onnx(N, H, W, img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """
        使用凸组合上采样光流 [H/8, W/8, 2] -> [H, W, 2]
        ONNX 兼容版本
        """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        # unfold 在 opset 12 中支持
        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """
        估计立体图像对之间的视差
        
        Args:
            image1: 左图像 [B, 3, H, W]
            image2: 右图像 [B, 3, H, W]
            iters: 迭代次数
            flow_init: 初始光流（可选）
            test_mode: 测试模式（只返回最终结果）
        """
        # 归一化输入图像
        image1 = (2.0 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2.0 * (image2 / 255.0) - 1.0).contiguous()

        # 运行上下文网络
        # 注意：在 ONNX 导出时，mixed_precision 会被忽略
        with torch.no_grad() if not self.training else torch.enable_grad():
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(
                    torch.cat((image1, image2), dim=0), 
                    dual_inp=True, 
                    num_layers=self.args.n_gru_layers
                )
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])
                
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # 预先计算 GRU 的卷积层
            inp_list = [
                list(conv(i).split(split_size=conv.out_channels//3, dim=1)) 
                for i, conv in zip(inp_list, self.context_zqr_convs)
            ]

        # 使用标准相关性实现（ONNX 兼容）
        if self.args.corr_implementation in ["reg", "alt"]:
            if self.args.corr_implementation == "reg":
                corr_block = CorrBlock1D
            else:
                corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        else:
            # CUDA 版本在 ONNX 导出时不可用，回退到标准版本
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
            
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # 索引相关性体积
            flow = coords1 - coords0
            
            # 更新块
            if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:
                net_list = self.update_block(
                    net_list, inp_list, 
                    iter32=True, iter16=False, iter08=False, update=False
                )
            if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:
                net_list = self.update_block(
                    net_list, inp_list, 
                    iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False
                )
            net_list, up_mask, delta_flow = self.update_block(
                net_list, inp_list, corr, flow, 
                iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2
            )

            # 在立体模式下，将光流投影到极线上
            delta_flow[:, 1] = 0.0

            # F(t+1) = F(t) + ΔF(t)
            coords1 = coords1 + delta_flow

            # 测试模式下不需要上采样或输出中间结果
            if test_mode and itr < iters-1:
                continue

            # 上采样预测
            if up_mask is None:
                flow_up = upflow8_onnx(coords1 - coords0, self.args.n_downsample)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            # 只保留水平视差
            flow_up = flow_up[:, :1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
