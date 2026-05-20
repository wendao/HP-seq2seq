# HP-seq2seq 自主研究

Seq2seq Transformer 模型在 HP 序列→结构映射任务上的自动优化。

## 项目结构

```
train.py          # 模型 + 训练循环（唯一可编辑文件）
train_baseline.py # 原始多模型训练脚本（参考）
prepare.py        # 数据加载、词表、folds、数据集（只读）
datasets/
  dataset20int    # 输入数据：每行 "input_seq output_seq"
results.tsv       # 实验日志（制表符分隔，不要提交到git）
PROGRAM.md        # 本文件：项目说明 + Autoresearch 原则
AUTO.md           # 进度总结（手动更新）
```

## 编辑范围

**只编辑 `train.py`。** 其他所有文件都是只读的。

## 可调超参数

```python
HIDDEN_SIZE = 512    # FFN 隐藏层维度
EMBED_DIM = 96       # 模型维度（d_model）
NUM_LAYERS = 4       # 编码器/解码器层数
NUM_HEADS = 4        # 注意力头数（必须整除 EMBED_DIM）
DROPOUT = 0.1        # Dropout 率
BATCH_SIZE = 64      # 批次大小
LR = 0.001           # 学习率
WEIGHT_DECAY = 1e-4  # 权重衰减
GRAD_CLIP = 1.0      # 梯度裁剪阈值（None=禁用）
LR_SCHEDULER = 'warmup'  # 'none', 'cosine', 'plateau', 'warmup'
```

## 固定超参数

```python
EPOCHS = 50          # 训练轮数（固定）
SPLIT = 'grouped'    # 按输出序列分组划分（无输出重叠）
FOLD = 0             # 交叉验证折数（0-4）
```

## 优化目标

**最大化验证 hit rate（val_hit_rate）**——输出序列的完全匹配准确率。

每个实验运行 **固定的 10 分钟时间预算**（墙钟训练时间，不包括启动）。训练固定 50 个 epoch，因此如果 50 epoch 在 10 分钟内完成则自然结束，否则提前停止。

**你可以做的：**
- 修改 `train.py` 中的"Tunable hyperparameters"，优化器、超参数、训练循环、批次大小、模型大小、学习率调度等。

**你不能做的：**
- 修改 `prepare.py`。它是只读的。
- 创建并运行新的 py 文件。
- 安装新包或添加依赖。只能使用已有的包（torch, prepare.py）。
- 修改评估方式。`prepare.py` 中的数据加载和分组划分是 ground truth。

**显存** 是软约束。显存增加一些可以接受，但不应急剧膨胀。

**简洁性准则**：在其他条件相同的情况下，越简单越好。一个小的改进如果增加了丑陋的复杂性，那就不值得。相反，删除某些东西并获得相同或更好的结果是一个好结果。

## 输出格式

脚本完成后打印：

```
Best validation hit rate: 0.4525
```

从日志中提取关键指标：

```bash
grep "Best validation hit rate" run.log
```

如果提前结束，就看最后一行的最后一个数字：

```bash
tail -n1 run.log
```

例子

```
Epoch 30/500 | Train Loss: 0.2690, Hit Rate: 0.1115 | Val Loss: 0.1957, Hit Rate: 0.2188
```


## 记录结果

实验完成后，记录到 `results.tsv`（制表符分隔，**不是**逗号分隔）。

TSV 有 5 列：

```
commit	val_hit_rate	memory_gb	status	description
```

1. git commit 哈希（短，7 个字符）
2. 达到的 val_hit_rate（例如 0.4525）——崩溃用 0.0000
3. 峰值显存（GB），保留 1 位小数——崩溃用 0.0
4. 状态：`keep`（优于基线）、`discard`（差于或等于基线）、`crash`
5. 简短描述

示例：

```
commit	val_hit_rate	memory_gb	status	description
30798ac	0.4032	0.0	keep	baseline: HIDDEN=512 EMBED=96 LAYERS=4
c4fc6ef	0.4525	0.0	keep	warmup+cosine LR scheduler
735a00e	0.2971	0.0	discard	DROPOUT=0.0
6cb65d8	0.0000	0.0	crash	NUM_HEADS=8 OOM
```

## 优化循环

**无限循环：**

1. 查看当前分支和 git 状态
2. 调整 `train.py` 中的超参数或架构
3. `git commit`
4. 运行实验：`python train.py > run.log 2>&1`（重定向所有输出）
5. 读取结果：`grep "Best validation hit rate" run.log`
6. 如果输出为空，运行崩溃了。运行 `tail -n 50 run.log` 读取堆栈跟踪并尝试修复。
7. 记录结果到 results.tsv（不要提交到 git）
8. 如果 val_hit_rate 提高，保留 commit，推进分支
9. 如果持平或更差，`git reset --hard HEAD~1` 回退

**超时**：每个实验应约 10 分钟。如果超过 15 分钟，终止并视为失败。

**崩溃**：如果是容易修复的问题（拼写错误、缺少导入），修复并重新运行。如果想法本身根本不可行，跳过，记录为 "crash"，继续。

**永远不要停止**：一旦循环开始，不要停下来问人类是否继续。自主工作直到被手动停止。如果没有想法了，尝试组合之前的近失策略，尝试更激进的变化。

## 关键约束

- 只使用现有包（torch, prepare.py）
- VRAM 不应急剧膨胀
- 简洁性优先：小改进如果增加复杂性则不值得
- 永远不要停止循环——自主运行直到被中断

---

# Autoresearch 原则与经验

> 以下内容是经过多轮实验沉淀出的实战指南，供后续 agent 参考。

## 一、优化策略框架

### 1.1 两阶段优化法

**阶段一：架构搜索**
- 先固定训练技巧（LR=0.001, AdamW, 无调度器或简单调度器）
- 逐个调整架构参数：HIDDEN_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT
- 每次只改一个参数，观察趋势，找到峰值区域
- 不要在架构未稳定前引入复杂的训练技巧

**阶段二：训练技巧微调**
- 在找到最优架构后，固定架构参数
- 逐个调整训练技巧：学习率、调度器、batch size、正则化、梯度裁剪
- 同样遵循"一次只改一个"原则

### 1.2 参数敏感性判断

一个参数如果满足以下特征，说明它很敏感：
- 轻微调整（如 ±20%）导致性能显著下降（>0.01）
- 双侧调整（增大和减小）都导致下降

敏感参数需要精细搜索（小步长），不敏感参数可以快速排除。

### 1.3 有效修改的判定标准

| 结果变化 | 行动 |
|---------|------|
| 明显提升 (>0.01) | 保留，作为新基线继续优化 |
| 轻微提升 (0.005-0.01) | 保留，但优先尝试更大的改进 |
| 持平 (±0.005) | 回退，除非简化了代码 |
| 下降 | 回退 |

## 二、已验证的参数敏感性（本项目）

### 2.1 架构参数

| 参数 | 测试范围 | 最佳值 | 敏感性 | 趋势 |
|------|---------|--------|--------|------|
| HIDDEN_SIZE | 128-1024 | **512** | 高 | 单峰，512 是 sweet spot |
| EMBED_DIM | 64-128 | **96** | 高 | 单峰，96 是 sweet spot |
| NUM_LAYERS | 3-5 | **4** | 高 | 单峰，4 是 sweet spot |
| NUM_HEADS | 4,6,8 | **4** | 高 | 增大导致下降，8 会 OOM |
| DROPOUT | 0.0-0.15 | **0.1** | 高 | 0.0 降 26%，0.15 降 15% |

**教训**：架构参数都是单峰的，且峰值很窄。找到峰值后不要继续在该维度上微调（如 0.08, 0.12 都更差）。

### 2.2 训练参数

| 参数 | 测试范围 | 最佳值 | 敏感性 | 趋势 |
|------|---------|--------|--------|------|
| LR | 0.0005-0.002 | **0.001** | 极高 | 0.0005 降 26%，0.002 降 7% |
| BATCH_SIZE | 64, 128 | **64** | 中 | 128 降约 5%，且 train/val 差距变小 |
| GRAD_CLIP | 0.5, 1.0, 5.0 | **1.0** | 中 | 两侧都更差 |
| WEIGHT_DECAY | 1e-5, 1e-4 | **1e-4** | 中 | 1e-5 降约 6% |
| WARMUP_EPOCHS | 3, 5, 10 | **5** | 中 | 3 和 10 都更差 |
| MIN_LR (cosine) | 0.0x, 0.1x, 0.2x | **0.1x** | 高 | 0.0x 约 0.4525，0.1x **0.4705**，0.2x 0.4457 |

## 三、有效技巧清单

### 3.1 已验证有效

1. **Warmup + Cosine 学习率调度**
   - 5 epochs 线性 warmup（从 0.1*LR 到 1.0*LR）
   - 之后 cosine 衰减
   - 相比无调度器或纯 cosine 有明显提升

2. **保留最小学习率（Min LR）**
   - 防止 cosine 衰减到 0
   - 最佳：保留 10% 的 base LR（即衰减到 0.0001）
   - 提升约 +0.018（从 ~0.4525 到 0.4705）
   - 代码实现：
     ```python
     return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
     ```

3. **AdamW + 权重衰减 1e-4**
   - 比 SGD 或纯 Adam 更稳定
   - 1e-4 比 1e-5 更好

4. **梯度裁剪 1.0**
   - 防止 Transformer 的大梯度爆炸
   - 比不裁剪（None）或裁剪过松（5.0）更好

### 3.2 无效或有害

| 技巧 | 效果 | 分析 |
|------|------|------|
| Label Smoothing 0.1 | 有害 (-0.025) | 对 exact-match 任务，平滑会惩罚高置信度的正确预测 |
| Pre-LN (norm_first=True) | 轻微有害 (-0.009) | 对本项目的小模型和短序列，Post-LN 已足够稳定 |
| AdamW betas (0.9, 0.98) | 有害 (-0.036) | 默认 (0.9, 0.999) 更适合本项目 |
| BATCH_SIZE 128 | 有害 (-0.054) | 更新次数减少，可能未充分收敛 |
| LR 0.0005 | 严重有害 (-0.117) | 学习率过低，收敛缓慢 |
| LR 0.002 | 有害 (-0.020) | 略高，可能导致不稳定 |

## 四、随机性与可重复性

**重要发现**：当前代码未设置随机种子，导致结果存在显著波动。

### 4.1 观察到的波动

同一 commit（838ecde，最优配置）两次运行结果：
- 第一次：0.4705
- 第二次：0.4336
- 差距：**0.037**（约 8% 的相对波动）

### 4.2 波动来源

1. **PyTorch 随机种子未设置**：`torch.manual_seed()`, `torch.cuda.manual_seed_all()` 缺失
2. **NumPy/Python 随机种子未设置**
3. **DataLoader shuffle**：每次 epoch 的数据顺序不同
4. **CUDA 非确定性操作**：某些 cuDNN 操作不是确定性的

### 4.3 应对策略

**对于 Autoresearch 代理**：
- 不要依赖单次实验结果做绝对判断
- 如果一个修改的结果与基线差距 <0.02，建议运行 2-3 次取平均
- 优先关注差距 >0.02 的修改
- 如果资源允许，对关键修改做多次验证

**如果要提高可重复性**（需要修改 train.py）：
```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

注意：设置种子后会降低训练速度（cuDNN.benchmark=False）。

## 五、常见陷阱

### 5.1 过度拟合训练指标

- Train hit rate 通常远高于 val hit rate（差距可达 0.1-0.15）
- 不要以 train loss 或 train hit rate 作为优化目标
- 只看 **val hit rate**

### 5.2 学习率调度器的隐藏陷阱

- **Cosine 衰减到 0**：在训练后期学习率几乎为 0，模型无法继续优化
  - 解决方案：设置 min_lr（如 0.1 * base_lr）
- **Warmup 过长**：占用宝贵的训练时间，导致实际在高学习率下的迭代次数减少
- **Plateau 调度器**：需要小心设置 patience，太短会频繁降 LR，太长会错过最佳点

### 5.3 批次大小与学习率的耦合

- 增大 batch size 时，通常需要同比例增大学习率（线性缩放规则）
- 本项目 BATCH_SIZE=128 失败，可能部分原因是未相应调整 LR

### 5.4 显存监控

- 虽然本模型很小（峰值显存 <0.1GB），但仍应监控
- 如果修改导致 OOM，记录为 crash，不要浪费时间去微调

## 六、实验设计建议

### 6.1 优先级排序

当没有明确方向时，按以下优先级尝试：

1. **学习率调度器**（影响最大，已验证）
2. **学习率数值**（敏感但范围窄）
3. **Dropout / 正则化**（防止过拟合）
4. **Batch Size**（稳定性 vs 收敛速度）
5. **梯度裁剪**（训练稳定性）
6. **优化器参数**（beta, eps 等）
7. **架构微调**（norm_first, activation 等）

### 6.2 组合策略

当单个技巧都尝试过且没有突破时：
- 尝试组合"近失"策略（即单独效果接近基线但略差的技巧）
- 例如：min LR + 更长的 warmup，或 label smoothing + 更大的模型
- 注意：组合后如果更差，需要回退到单个最佳技巧

### 6.3 何时停止

满足以下条件时可以认为已进入收益递减区：
- 连续 5-8 个实验都没有超过当前最佳
- 已覆盖所有主要超参数维度
- 最佳结果在多次运行中稳定（考虑随机波动）

此时应：
- 记录最终最佳配置
- 更新 AUTO.md 和 PROGRAM.md
- 准备汇报

## 七、代码修改规范

### 7.1 提交信息格式

```
exp: <修改内容简要描述>
```

示例：
- `exp: min LR 0.1x`
- `exp: warmup 10 epochs`
- `exp: label smoothing 0.1`

### 7.2 回退规则

- 结果持平或更差：`git reset --hard HEAD~1`
- 结果提升：保留，继续基于新基线优化
- 崩溃：如果是简单错误（拼写、导入），修复后重试；否则记录为 crash，回退

### 7.3 不要修改的文件

- `prepare.py`：ground truth，只读
- `results.tsv`：本地实验记录，不提交到 git
- `AUTO.md`：手动更新的进度总结

## 八、快速参考：最优配置

```python
# 架构
HIDDEN_SIZE = 512
EMBED_DIM = 96
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1

# 训练
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# 调度器：warmup + cosine with min LR 0.1x
LR_SCHEDULER = "warmup"
def lr_lambda(epoch):
    if epoch < 5:
        return 0.1 + 0.9 * epoch / 5
    else:
        progress = (epoch - 5) / (EPOCHS - 5)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
```

**当前最佳 val_hit_rate：约 0.45-0.47**（存在随机波动，未设种子）
