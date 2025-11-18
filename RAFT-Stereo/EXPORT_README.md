# RAFT-Stereo ONNX 导出指南 (针对 Jetson NX TensorRT 7.1.3)

## 环境要求

### 当前PC环境（用于导出ONNX）
- PyTorch >= 1.6
- ONNX >= 1.10
- onnx-simplifier (可选但推荐)

### 目标设备 Jetson NX
- ONNX 1.10
- TensorRT 7.1.3
- CUDA + cuDNN

## 步骤1: 安装依赖

```bash
cd RAFT-Stereo

# 安装基础依赖
pip install torch torchvision
pip install onnx==1.10.0
pip install onnx-simplifier

# (可选) 如果需要使用mmcv的bilinear_grid_sample
# pip install mmcv-full
```

## 步骤2: 导出 realtime 模型

```bash
# 确保 raftstereo-realtime.pth 在根目录或指定正确路径
python3 export_onnx.py \
    --restore_ckpt ../raftstereo-realtime.pth \
    --shared_backbone \
    --n_downsample 3 \
    --n_gru_layers 2 \
    --slow_fast_gru \
    --valid_iters 7 \
    --mixed_precision \
    --opset_version 12
```

### 参数说明：
- `--restore_ckpt`: pth模型路径
- `--shared_backbone`: realtime模型必须启用
- `--n_downsample 3`: realtime模型使用3
- `--n_gru_layers 2`: realtime模型使用2层GRU
- `--slow_fast_gru`: 启用慢速-快速GRU
- `--valid_iters 7`: realtime模型使用7次迭代
- `--opset_version 12`: **关键！** 使用opset 12以兼容ONNX 1.10 / TensorRT 7.1.3

## 步骤3: 简化ONNX模型（推荐）

```bash
# 简化模型，移除冗余节点
onnxsim raftstereo-realtime_480_640.onnx raftstereo-realtime_480_640_sim.onnx
```

## 步骤4: 验证ONNX模型

```bash
# 检查ONNX模型
python3 -c "
import onnx
model = onnx.load('raftstereo-realtime_480_640_sim.onnx')
onnx.checker.check_model(model)
print('Model is valid!')
print(f'Opset version: {model.opset_import[0].version}')
"
```

## 步骤5: 传输到Jetson NX测试

```bash
# 将ONNX模型传输到Jetson NX
scp raftstereo-realtime_480_640_sim.onnx nx@jetson-nx:/path/to/model/
```

## 常见问题

### Q1: 为什么使用 opset 12 而不是 16？
**A:** TensorRT 7.1.3对应的ONNX支持有限，opset 12-13是比较安全的选择。opset 16的某些算子（如高级grid_sample）可能不支持。

### Q2: 如果转换失败怎么办？
**A:** 可以尝试：
1. 降低opset版本到11: `--opset_version 11`
2. 检查是否有不兼容的算子
3. 使用`--corr_implementation reg`而不是cuda版本

### Q3: 模型输入输出尺寸？
**A:** 
- 输入: 左右图像各 (1, 3, 480, 640) - 固定尺寸
- 输出: 视差图 (1, 1, 480, 640)

### Q4: 性能预期？
**A:** 
- Jetson Xavier-NX: ~120ms (realtime模型)
- Jetson TX2-NX: ~400ms (realtime模型)

## 导出其他模型版本

### SceneFlow 模型（更高精度，更慢）

```bash
python3 export_onnx.py \
    --restore_ckpt models/raftstereo-sceneflow.pth \
    --valid_iters 32 \
    --opset_version 12
```

注意：sceneflow模型不需要 `--shared_backbone`、`--n_downsample 3` 等参数。

## 下一步

转换完成后，使用C++项目中的TensorRT推理代码：
```
./build/RAFTStereo/test/raft_stereo_demo
```

确保：
1. StereoCalibration.yml 标定文件已生成
2. ONNX模型路径正确
3. 测试图像准备好
