# RAFT-Stereo ONNX opset 12 导出指南

针对 **Jetson NX (ONNX 1.10 / TensorRT 7.1.3)** 的完整导出方案

---

## 🎯 方案说明

我已经为你创建了完全兼容 **opset 12 / TensorRT 7.1.3** 的版本，主要改进：

### ✅ 已优化的算子

| 原始算子 | 问题 | 解决方案 |
|---------|------|---------|
| `F.grid_sample(..., align_corners=True)` | opset 12 兼容性差 | → `align_corners=False` |
| `F.interpolate(..., align_corners=True)` | 同上 | → `align_corners=False` |
| `torch.meshgrid(indexing='ij')` | opset 12 不支持 | → 手动创建网格 |
| CUDA 自定义算子 | 无法导出 | → 纯 PyTorch 实现 |
| `torch.einsum` 复杂模式 | 部分不支持 | → 简化模式 |

### 📁 创建的文件

```
RAFT-Stereo/
├── core/
│   ├── utils/
│   │   └── utils_onnx_compatible.py          # 兼容的工具函数
│   ├── corr_onnx_compatible.py                # 兼容的相关性计算
│   ├── update_onnx_compatible.py              # 兼容的更新模块
│   └── raft_stereo_onnx_compatible.py         # 兼容的主模型
├── export_onnx_opset12.py                     # opset 12 导出脚本 ⭐
└── OPSET12_EXPORT_GUIDE.md                    # 本文档
```

---

## 🚀 快速开始

### 步骤 1: 安装依赖

```bash
cd /home/user/webapp/RAFT-Stereo

pip install torch==1.9.0  # 或更高版本
pip install onnx==1.10.0
pip install onnx-simplifier
```

### 步骤 2: 导出 realtime 模型

```bash
python3 export_onnx_opset12.py \
    --restore_ckpt ../raftstereo-realtime.pth \
    --shared_backbone \
    --n_downsample 3 \
    --n_gru_layers 2 \
    --slow_fast_gru \
    --valid_iters 7 \
    --mixed_precision \
    --opset_version 12
```

**输出**: `raftstereo-realtime_480_640_opset12.onnx`

### 步骤 3: 简化模型

```bash
onnxsim raftstereo-realtime_480_640_opset12.onnx \
       raftstereo-realtime_480_640_opset12_sim.onnx
```

### 步骤 4: 验证模型

```bash
python3 -c "
import onnx
model = onnx.load('raftstereo-realtime_480_640_opset12_sim.onnx')
onnx.checker.check_model(model)
print('✓ Model valid!')
print(f'Opset: {model.opset_import[0].version}')
"
```

---

## 🔧 如果 opset 12 仍然有问题

### 尝试 opset 11

```bash
python3 export_onnx_opset12.py \
    --restore_ckpt ../raftstereo-realtime.pth \
    --shared_backbone \
    --n_downsample 3 \
    --n_gru_layers 2 \
    --slow_fast_gru \
    --valid_iters 7 \
    --opset_version 11  # 改为 11
```

### 尝试 opset 13

```bash
python3 export_onnx_opset12.py \
    --restore_ckpt ../raftstereo-realtime.pth \
    --shared_backbone \
    --n_downsample 3 \
    --n_gru_layers 2 \
    --slow_fast_gru \
    --valid_iters 7 \
    --opset_version 13  # 改为 13
```

---

## 📊 导出 sceneflow 模型

```bash
python3 export_onnx_opset12.py \
    --restore_ckpt models/raftstereo-sceneflow.pth \
    --valid_iters 32 \
    --n_downsample 2 \
    --n_gru_layers 3 \
    --opset_version 12
```

注意：sceneflow 模型**不需要** `--shared_backbone` 和 `--slow_fast_gru`

---

## 🧪 在 Jetson NX 上测试

### 1. 传输模型到 NX

```bash
scp raftstereo-realtime_480_640_opset12_sim.onnx nx@jetson-nx:/path/to/model/
```

### 2. 使用 trtexec 测试转换

```bash
# 在 Jetson NX 上执行
/usr/src/tensorrt/bin/trtexec \
    --onnx=raftstereo-realtime_480_640_opset12_sim.onnx \
    --saveEngine=raftstereo-realtime.engine \
    --fp16  # 如果支持 FP16
```

### 3. 检查转换日志

查找以下关键信息：
- ✅ "Successfully created engine"
- ⚠️ "Unsupported operator" - 说明算子不兼容
- ⚠️ "Fallback to CPU" - 说明某些层无法加速

---

## ❓ 常见问题排查

### Q1: 导出时报错 "Unsupported operator"

**A:** 查看具体是哪个算子，然后：
1. 降低 opset 版本（12 → 11）
2. 检查是否使用了 CUDA 算子（需要用 `--corr_implementation reg`）

### Q2: TensorRT 转换失败

**A:** 可能的原因：
1. **opset 版本太高** → 降低到 11
2. **动态尺寸不支持** → 已默认使用固定尺寸
3. **某些算子 TRT 7.1.3 不支持** → 需要升级 TRT 或修改模型

### Q3: 推理结果不正确

**A:** 检查：
1. 输入图像范围是否正确（0-255）
2. 输出视差是否需要取负值
3. 标定参数是否正确

### Q4: 性能不如预期

**A:** 优化建议：
1. 使用 FP16 精度（`--fp16`）
2. 减少迭代次数（`--valid_iters 5`）
3. 使用 realtime 模型而非 sceneflow

---

## 📝 模型输入输出格式

### 输入

```
left_image:  [1, 3, 480, 640]  # 左图像, RGB, 范围 0-255
right_image: [1, 3, 480, 640]  # 右图像, RGB, 范围 0-255
```

### 输出

```
disparity: [1, 1, 480, 640]  # 视差图, float32
```

**注意**: 
- 输入必须是 **0-255** 范围（不是归一化的 0-1）
- 输出视差是**负值**（右图相对左图的位移）

---

## 🎓 技术细节

### 主要兼容性修改

1. **grid_sample**
   ```python
   # 原始
   F.grid_sample(img, grid, align_corners=True)
   
   # 修改为
   F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
   ```

2. **interpolate**
   ```python
   # 原始
   F.interpolate(x, size=new_size, mode='bilinear', align_corners=True)
   
   # 修改为
   F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)
   ```

3. **meshgrid**
   ```python
   # 原始
   coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
   
   # 修改为手动创建
   y_grid = torch.arange(h).view(-1, 1).repeat(1, w)
   x_grid = torch.arange(w).view(1, -1).repeat(h, 1)
   ```

4. **移除 CUDA 依赖**
   - 不使用 `corr_sampler` (CUDA 扩展)
   - 不使用 `alt_cuda_corr` (CUDA 扩展)
   - 全部使用纯 PyTorch 实现

---

## 📞 需要帮助？

如果遇到问题：

1. 检查导出日志中的具体错误信息
2. 尝试不同的 opset 版本（11/12/13）
3. 在 NX 上使用 `trtexec --verbose` 查看详细转换日志
4. 提供错误信息以便进一步诊断

---

## ✅ 总结

- ✅ 已创建完全兼容 opset 12 的版本
- ✅ 移除所有高版本算子依赖
- ✅ 移除 CUDA 自定义算子
- ✅ 优化所有插值和采样操作
- ✅ 固定输入尺寸（480x640）
- ✅ 提供完整的导出和测试流程

**现在可以直接运行导出命令了！** 🚀
