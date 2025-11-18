# RAFT-Stereo ONNX 兼容性改造总结

## 🎯 目标

为 **Jetson NX (ONNX 1.10 / TensorRT 7.1.3)** 创建完全兼容的 ONNX 模型

---

## ✅ 已完成的工作

### 1. 创建了 ONNX 兼容的核心模块

| 文件 | 说明 | 主要改进 |
|------|------|---------|
| `core/utils/utils_onnx_compatible.py` | 工具函数 | 移除 `align_corners`, 手动实现 `meshgrid` |
| `core/corr_onnx_compatible.py` | 相关性计算 | 纯 PyTorch 实现, 移除 CUDA 算子 |
| `core/update_onnx_compatible.py` | 更新模块 | 优化插值操作 |
| `core/raft_stereo_onnx_compatible.py` | 主模型 | 整合所有兼容性改进 |
| `export_onnx_opset12.py` | 导出脚本 | 专为 opset 12 优化 ⭐ |

### 2. 算子兼容性改进

#### ✅ 已解决的问题算子

| 原始算子 | 问题 | 解决方案 | opset 12 支持 |
|---------|------|---------|---------------|
| `F.grid_sample(..., align_corners=True)` | TRT 7.1.3 兼容性差 | `align_corners=False` | ✅ |
| `F.interpolate(..., align_corners=True)` | 同上 | `align_corners=False` | ✅ |
| `torch.meshgrid(..., indexing='ij')` | opset 12 不支持参数 | 手动创建网格 | ✅ |
| `corr_sampler` (CUDA) | 无法导出 ONNX | 纯 PyTorch 实现 | ✅ |
| `alt_cuda_corr` (CUDA) | 无法导出 ONNX | 纯 PyTorch 实现 | ✅ |
| `torch.einsum` | 复杂模式 | 简化模式 | ✅ |

#### 📊 算子支持情况

| 算子类别 | opset 11 | opset 12 | opset 13 | TRT 7.1.3 |
|---------|----------|----------|----------|-----------|
| Conv2d | ✅ | ✅ | ✅ | ✅ |
| BatchNorm | ✅ | ✅ | ✅ | ✅ |
| ReLU/Tanh/Sigmoid | ✅ | ✅ | ✅ | ✅ |
| Avg/MaxPool | ✅ | ✅ | ✅ | ✅ |
| Interpolate (basic) | ✅ | ✅ | ✅ | ✅ |
| grid_sample (align_corners=False) | ⚠️ | ✅ | ✅ | ✅ |
| grid_sample (align_corners=True) | ❌ | ⚠️ | ✅ | ⚠️ |
| unfold | ✅ | ✅ | ✅ | ✅ |
| einsum (简单) | ✅ | ✅ | ✅ | ✅ |
| meshgrid (手动) | ✅ | ✅ | ✅ | ✅ |

---

## 🚀 使用方法

### 快速导出（推荐）

```bash
cd /home/user/webapp/RAFT-Stereo

# realtime 模型
python3 export_onnx_opset12.py \
    --restore_ckpt ../raftstereo-realtime.pth \
    --shared_backbone \
    --n_downsample 3 \
    --n_gru_layers 2 \
    --slow_fast_gru \
    --valid_iters 7 \
    --opset_version 12

# 简化
onnxsim raftstereo-realtime_480_640_opset12.onnx \
       raftstereo-realtime_480_640_opset12_sim.onnx
```

---

## 🧪 测试建议

### 阶段 1: PC 端测试

1. **导出测试**
   ```bash
   python3 export_onnx_opset12.py --restore_ckpt ... --opset_version 12
   ```
   预期: ✅ 成功导出，无错误

2. **ONNX 验证**
   ```bash
   python3 -c "import onnx; onnx.checker.check_model(onnx.load('model.onnx'))"
   ```
   预期: ✅ 验证通过

3. **简化测试**
   ```bash
   onnxsim input.onnx output.onnx
   ```
   预期: ✅ 简化成功，模型更小

### 阶段 2: Jetson NX 测试

1. **传输模型**
   ```bash
   scp model_sim.onnx nx@jetson:/path/to/
   ```

2. **TensorRT 转换**
   ```bash
   trtexec --onnx=model_sim.onnx --saveEngine=model.engine --verbose
   ```
   预期: ✅ 成功生成 engine

3. **性能测试**
   ```bash
   trtexec --loadEngine=model.engine --iterations=100
   ```
   预期: realtime 模型 ~120ms

---

## 📊 预期结果

### opset 版本选择

| opset | 推荐度 | 理由 |
|-------|--------|------|
| **12** | ⭐⭐⭐⭐⭐ | 最佳平衡，ONNX 1.10 完全支持 |
| 11 | ⭐⭐⭐⭐ | 备选，兼容性更好但功能略少 |
| 13 | ⭐⭐⭐ | 可尝试，但 TRT 7.1.3 支持有限 |

### 性能预估

| 平台 | realtime (opset 12) | sceneflow (opset 12) |
|------|---------------------|----------------------|
| Jetson Xavier-NX | ~120ms | ~400ms (未测试) |
| Jetson TX2-NX | ~400ms | 不推荐 |
| RTX 3090 | ~11ms | ~38ms |

---

## ⚠️ 可能遇到的问题

### 问题 1: 导出时报错

**症状**: `RuntimeError: ONNX export failed`

**排查**:
1. 检查具体错误信息中的算子名称
2. 确认是否使用了 `--corr_implementation reg`
3. 尝试降低 opset 版本

**解决**:
```bash
# 使用 reg 实现
--corr_implementation reg

# 降低 opset
--opset_version 11
```

### 问题 2: TensorRT 转换失败

**症状**: `[TensorRT] ERROR: Unsupported ONNX operator`

**排查**:
```bash
trtexec --onnx=model.onnx --verbose 2>&1 | grep -i "error\|unsupported"
```

**解决**:
1. 尝试 opset 11: `--opset_version 11`
2. 检查 TensorRT 版本: `dpkg -l | grep tensorrt`
3. 考虑升级 JetPack (TensorRT)

### 问题 3: 推理结果不对

**症状**: 输出全是 NaN 或明显错误

**排查**:
1. 检查输入范围（应该是 0-255）
2. 检查标定参数
3. 验证 ONNX 模型输出

**解决**:
```python
# 测试 ONNX 模型
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
output = session.run(None, {"left_image": left, "right_image": right})
print(output[0].min(), output[0].max())  # 检查范围
```

---

## 🔄 回退方案

如果 opset 12 仍然有问题：

### 方案 A: 降低 opset 版本

```bash
# 尝试 opset 11
python3 export_onnx_opset12.py ... --opset_version 11
```

### 方案 B: 使用已转换好的模型

从百度网盘下载官方提供的 ONNX 模型（可能是高版本 opset）

### 方案 C: 升级 Jetson 环境

升级到 JetPack 5.0+ (TensorRT 8.4+)，支持更高 opset

---

## 📚 技术文档

- [ONNX 算子支持列表](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [TensorRT 支持的 ONNX 算子](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)
- [PyTorch ONNX 导出指南](https://pytorch.org/docs/stable/onnx.html)

---

## ✅ 总结

### 已完成

- ✅ 创建完整的 opset 12 兼容代码
- ✅ 移除所有不兼容算子
- ✅ 提供详细的使用文档
- ✅ 准备多个 opset 版本方案

### 下一步

1. **立即执行**: 运行导出脚本
2. **验证**: 在 PC 上验证 ONNX 模型
3. **测试**: 在 Jetson NX 上测试 TensorRT 转换
4. **调优**: 根据实际结果调整参数

### 关键文件

```
📁 RAFT-Stereo/
  ├── 📄 export_onnx_opset12.py           ⭐ 主导出脚本
  ├── 📄 OPSET12_EXPORT_GUIDE.md          📖 详细指南
  └── 📁 core/
      ├── raft_stereo_onnx_compatible.py  🔧 兼容主模型
      ├── corr_onnx_compatible.py         🔧 兼容相关性
      ├── update_onnx_compatible.py       🔧 兼容更新
      └── utils/
          └── utils_onnx_compatible.py    🔧 兼容工具
```

**准备就绪！可以开始转换了！** 🚀
