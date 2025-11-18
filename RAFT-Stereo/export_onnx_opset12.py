"""
RAFT-Stereo ONNX 导出脚本 - opset 12 兼容版本
专为 Jetson NX (ONNX 1.10 / TensorRT 7.1.3) 优化

使用方法:
python3 export_onnx_opset12.py --restore_ckpt ../raftstereo-realtime.pth \
    --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru \
    --valid_iters 7 --mixed_precision
"""
import sys
sys.path.append('core')

import argparse
import torch
import torch.nn as nn
from raft_stereo_onnx_compatible import RAFTStereoONNX

# 固定输入尺寸
INPUT_HEIGHT = 480
INPUT_WIDTH = 640


class RAFTStereoONNXWrapper(nn.Module):
    """
    ONNX导出封装类
    """
    def __init__(self, model, iters=12):
        super().__init__()
        self.model = model
        self.iters = iters
        
    def forward(self, image1, image2):
        """
        前向传播 - 简化为测试模式
        
        输入:
            image1: [B, 3, H, W] - 左图像 (0-255)
            image2: [B, 3, H, W] - 右图像 (0-255)
        
        输出:
            disparity: [B, 1, H, W] - 视差图
        """
        _, flow_up = self.model(image1, image2, iters=self.iters, test_mode=True)
        
        # 取负值得到视差（左图到右图的位移）
        disparity = -flow_up
        
        return disparity


def load_pretrained_model(args):
    """
    加载预训练模型
    
    兼容两种情况:
    1. DataParallel 保存的模型
    2. 直接保存的模型
    """
    print(f"Loading checkpoint: {args.restore_ckpt}")
    
    # 创建模型
    model = RAFTStereoONNX(args)
    
    # 加载权重
    checkpoint = torch.load(args.restore_ckpt, map_location='cpu')
    
    # 尝试加载 DataParallel 格式
    try:
        # 如果是 DataParallel 保存的，需要去掉 'module.' 前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                name = k[7:]  # 移除 'module.' 前缀
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("✓ Loaded DataParallel checkpoint")
    except:
        # 直接加载
        model.load_state_dict(checkpoint)
        print("✓ Loaded standard checkpoint")
    
    return model


def export_onnx(args):
    """
    导出 ONNX 模型
    """
    print("="*70)
    print("RAFT-Stereo ONNX Export for TensorRT 7.1.3 (opset 12)")
    print("="*70)
    print(f"Target: Jetson NX with TensorRT 7.1.3")
    print(f"ONNX version: 1.10")
    print(f"Opset version: {args.opset_version}")
    print(f"Input size: {INPUT_HEIGHT}x{INPUT_WIDTH}")
    print(f"Iterations: {args.valid_iters}")
    print("="*70)
    
    # 加载模型
    model = load_pretrained_model(args)
    model.eval()
    
    # 封装为 ONNX 导出版本
    onnx_model = RAFTStereoONNXWrapper(model, iters=args.valid_iters)
    onnx_model.eval()
    
    # 创建 dummy 输入
    dummy_left = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    dummy_right = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    
    # 生成输出文件名
    model_name = args.restore_ckpt.split('/')[-1].replace('.pth', '')
    output_name = f"{model_name}_{INPUT_HEIGHT}_{INPUT_WIDTH}_opset{args.opset_version}.onnx"
    
    print(f"\n🔧 Exporting to ONNX...")
    print(f"   Model: {model_name}")
    print(f"   Input shape: {dummy_left.shape}")
    print(f"   Output file: {output_name}")
    
    # 测试一次前向传播
    print(f"\n🧪 Testing forward pass...")
    try:
        with torch.no_grad():
            output = onnx_model(dummy_left, dummy_right)
        print(f"   ✓ Forward pass successful")
        print(f"   Output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return
    
    # 导出 ONNX
    print(f"\n📦 Exporting ONNX (opset {args.opset_version})...")
    try:
        torch.onnx.export(
            onnx_model,
            (dummy_left, dummy_right),
            output_name,
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=['left_image', 'right_image'],
            output_names=['disparity'],
            dynamic_axes=None if args.fixed_size else {
                'left_image': {0: 'batch'},
                'right_image': {0: 'batch'},
                'disparity': {0: 'batch'}
            },
            verbose=False
        )
        print(f"   ✓ ONNX export successful!")
    except Exception as e:
        print(f"   ✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 验证 ONNX 模型
    print(f"\n✅ Validating ONNX model...")
    try:
        import onnx
        onnx_model_check = onnx.load(output_name)
        onnx.checker.check_model(onnx_model_check)
        print(f"   ✓ ONNX model is valid!")
        print(f"   Opset version: {onnx_model_check.opset_import[0].version}")
    except ImportError:
        print(f"   ⚠ onnx package not installed, skipping validation")
    except Exception as e:
        print(f"   ⚠ Validation warning: {e}")
    
    # 成功信息
    print(f"\n" + "="*70)
    print(f"✅ Export completed successfully!")
    print(f"="*70)
    print(f"\n📄 Output file: {output_name}")
    print(f"\n📋 Next steps:")
    print(f"   1. Simplify ONNX model:")
    print(f"      onnxsim {output_name} {output_name.replace('.onnx', '_sim.onnx')}")
    print(f"\n   2. Transfer to Jetson NX:")
    print(f"      scp {output_name.replace('.onnx', '_sim.onnx')} nx@jetson:/path/to/model/")
    print(f"\n   3. Test with TensorRT 7.1.3 on NX")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export RAFT-Stereo to ONNX (opset 12)')
    
    # 模型参数
    parser.add_argument('--restore_ckpt', required=True, help="path to checkpoint (.pth)")
    parser.add_argument('--valid_iters', type=int, default=7, 
                        help='iterations (7 for realtime, 32 for sceneflow)')
    
    # 架构参数
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], 
                        default="reg", help="use 'reg' or 'alt' for ONNX")
    parser.add_argument('--shared_backbone', action='store_true', 
                        help="required for realtime model")
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2, 
                        help="3 for realtime, 2 for sceneflow")
    parser.add_argument('--context_norm', type=str, default="batch", 
                        choices=['group', 'batch', 'instance', 'none'])
    parser.add_argument('--slow_fast_gru', action='store_true', 
                        help="required for realtime model")
    parser.add_argument('--n_gru_layers', type=int, default=3, 
                        help="2 for realtime, 3 for sceneflow")
    parser.add_argument('--mixed_precision', action='store_true', 
                        help='flag for training, ignored during export')
    
    # ONNX 导出参数
    parser.add_argument('--opset_version', type=int, default=12,
                        help='ONNX opset (12 for TRT 7.1.3, can try 11 or 13)')
    parser.add_argument('--fixed_size', action='store_true', default=True,
                        help='use fixed input size (recommended)')
    
    args = parser.parse_args()
    
    # 检查参数
    if args.corr_implementation in ["reg_cuda", "alt_cuda"]:
        print(f"⚠ Warning: {args.corr_implementation} requires CUDA ops, switching to 'reg'")
        args.corr_implementation = "reg"
    
    export_onnx(args)
