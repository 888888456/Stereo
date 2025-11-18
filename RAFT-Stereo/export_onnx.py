import sys
sys.path.append('core')

import argparse
import torch
import torch.nn as nn
from raft_stereo import RAFTStereo

# 设置固定的输入尺寸
INPUT_HEIGHT = 480
INPUT_WIDTH = 640

class RAFTStereoONNX(nn.Module):
    """
    ONNX导出封装类，简化模型以便于ONNX转换
    """
    def __init__(self, model, iters=12):
        super().__init__()
        self.model = model
        self.iters = iters
        
    def forward(self, image1, image2):
        # 调用原始模型的forward，使用test_mode
        _, flow_up = self.model(image1, image2, iters=self.iters, test_mode=True)
        return flow_up


def export_onnx(args):
    """
    导出ONNX模型，针对ONNX 1.10 / TensorRT 7.1.3优化
    """
    print(f"Loading model from: {args.restore_ckpt}")
    
    # 加载模型
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, map_location='cpu'))
    model = model.module
    model.eval()
    
    # 封装为ONNX友好的模型
    onnx_model = RAFTStereoONNX(model, iters=args.valid_iters)
    onnx_model.eval()
    
    # 创建dummy输入 (batch_size, channels, height, width)
    dummy_left = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    dummy_right = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    
    # 输出文件名
    model_name = args.restore_ckpt.split('/')[-1].replace('.pth', '')
    output_name = f"{model_name}_{INPUT_HEIGHT}_{INPUT_WIDTH}.onnx"
    
    print(f"\nExporting to ONNX...")
    print(f"Input shape: {dummy_left.shape}")
    print(f"Output file: {output_name}")
    print(f"ONNX opset version: {args.opset_version}")
    print(f"Valid iterations: {args.valid_iters}")
    
    # 导出ONNX
    # 对于 ONNX 1.10 / TensorRT 7.1.3，使用 opset 12-13
    torch.onnx.export(
        onnx_model,
        (dummy_left, dummy_right),
        output_name,
        export_params=True,
        opset_version=args.opset_version,  # 使用较低的opset版本以兼容TRT 7.1.3
        do_constant_folding=True,
        input_names=['left_image', 'right_image'],
        output_names=['disparity'],
        dynamic_axes={
            # 不使用动态轴，固定尺寸更容易在TensorRT上推理
        } if args.fixed_size else {
            'left_image': {0: 'batch_size'},
            'right_image': {0: 'batch_size'},
            'disparity': {0: 'batch_size'}
        }
    )
    
    print(f"\n✓ Successfully exported to {output_name}")
    print(f"\nNext steps:")
    print(f"1. Simplify ONNX: onnxsim {output_name} {output_name.replace('.onnx', '_sim.onnx')}")
    print(f"2. Test on Jetson NX with TensorRT 7.1.3")
    
    # 验证导出的模型
    try:
        import onnx
        onnx_model_check = onnx.load(output_name)
        onnx.checker.check_model(onnx_model_check)
        print(f"✓ ONNX model validation passed")
    except ImportError:
        print("! Warning: onnx package not found, skipping validation")
    except Exception as e:
        print(f"! Warning: ONNX validation failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export RAFT-Stereo to ONNX for TensorRT 7.1.3')
    
    # 模型相关参数
    parser.add_argument('--restore_ckpt', help="path to model checkpoint", required=True)
    parser.add_argument('--valid_iters', type=int, default=7, help='number of flow-field updates (7 for realtime, 32 for sceneflow)')
    
    # 架构参数 (realtime model)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone (required for realtime)")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (3 for realtime, 2 for sceneflow)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels (2 for realtime, 3 for sceneflow)")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    
    # ONNX导出参数
    parser.add_argument('--opset_version', type=int, default=12, 
                        help='ONNX opset version (12-13 for ONNX 1.10 / TensorRT 7.1.3)')
    parser.add_argument('--fixed_size', action='store_true', default=True,
                        help='use fixed input size (recommended for TensorRT)')
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("="*60)
    print("RAFT-Stereo ONNX Export for TensorRT 7.1.3")
    print("="*60)
    print(f"Target platform: Jetson NX")
    print(f"TensorRT version: 7.1.3")
    print(f"ONNX version: 1.10")
    print(f"Opset version: {args.opset_version}")
    print(f"Input size: {INPUT_HEIGHT}x{INPUT_WIDTH}")
    print("="*60)
    
    export_onnx(args)
