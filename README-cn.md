# HP-seq2seq

HP序列→结构映射的Seq2seq训练框架。

## 任务

给定输入HP序列（长度20，字母表`{H, P}`），预测输出结构序列（长度18，字母表`{R, L, F}`）。

**目标：** 最大化hit rate（输出序列的完全匹配准确率）。

## 数据

- **文件:** `datasets/dataset20int`
- **大小:** 24,900 个样本
- **格式:** `HPHHPPHH... RFLLFR...`（空格分隔的输入和输出）
- **输入长度:** 20（字母表: H, P）
- **输出长度:** 18（字母表: R, L, F，外加特殊标记<pad>, <sos>, <eos>）

## 项目结构

```
HP-seq2seq/
├── datasets/
│   └── dataset20int       # 训练数据 (24900个样本)
├── prepare.py             # 数据处理（加载、词表、折叠、数据集、批处理）
├── train.py               # 训练脚本，支持RNN/LSTM/CNN模型
└── README.md / README-cn.md
```

## prepare.py（只读）

数据准备模块：

- `load_data(path)` → (input_str, output_str) 列表
- `create_vocabs(input_seqs, output_seqs)` → input_vocab, output_vocab
- `create_folds(data, n_folds=5)` → (train_idx, val_idx) 元组列表
- `encode_data(data, input_vocab, output_vocab)` → Seq2SeqDataset
- `collate_fn(batch)` → 填充后的序列

## train.py

支持模型选择和超参数调整的训练脚本。

### 使用方法（GPU）

```bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES="3"
python train.py --model lstm --fold 0 --epochs 30
```

### 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--model` | lstm | 模型类型: rnn, lstm, 或 cnn |
| `--fold` | 0 | 交叉验证折数 (0-4) |
| `--epochs` | 30 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--hidden_size` | 8 | 隐藏层维度 |
| `--embed_dim` | 4 | 嵌入维度 |
| `--batch_size` | 64 | 批量大小 |
| `--num_layers` | 1 | 层数：RNN/LSTM直接使用；CNN作为编码器层数（解码器使用num_layers-1） |
| `--cross_attn` | 0 | Cross-Attention 头数（0=禁用） |

### 模型

所有模型均支持 `--hidden_size` 和 `--embed_dim` 控制容量。

**RNN Seq2Seq:**
- `EncoderRNN`: 双向RNN编码器
- `DecoderRNN`: 带简化注意力机制的RNN解码器
- `--num_layers`: RNN层数（默认1，同时控制解码器）
- `--cross_attn`: Cross-Attention 头数（0=禁用）

**LSTM Seq2Seq:**
- `EncoderLSTM`: 双向LSTM编码器
- `DecoderLSTM`: LSTM解码器
- `--num_layers`: LSTM层数（默认1）
- `--cross_attn`: Cross-Attention 头数（0=禁用）

**CNN Seq2Seq:**
- `EncoderCNN`: 多层Conv1D编码器，带残差连接和全局池化（max + avg）
- `DecoderCNN`: 带可选Cross-Attention的Conv1D解码器
- `--num_layers`: CNN编码器层数；解码器使用num_layers-1
- `--cross_attn`: Cross-Attention 头数（0=禁用）

| num_layers | cross_attn | 参数量 |
|--------------|-----------|--------|
| 1 | false | 370 |
| 3 | false | 970 |
| 5 | false | 1,770 |
| 5 | true | 1,770 |

### 训练细节

- **损失函数:** CrossEntropyLoss (ignore_index=<pad>)
- **优化器:** Adam
- **指标:** Hit Rate = 输出序列的完全匹配准确率
- **5折交叉验证**（每折内80/20训练/验证划分）
- **Teacher forcing** 训练

### 输出词表

特殊标记: `<pad>=0`, `<sos>=1`, `<eos>=2`
内容标记: `R=3`, `L=4`, `F=5`

## 最新结果（5折CV，100轮）

### LSTM
| 折 | 最佳验证Hit Rate |
|------|-------------------|
| 0 | 0.0269 |
| 1 | 0.0235 |
| 2 | 0.0343 |
| 3 | 0.0309 |
| 4 | 0.0470 |
| **平均** | **0.0325** |

### RNN
| 折 | 最佳验证Hit Rate |
|------|-------------------|
| 0 | 0.0062 |
| 1 | 0.0042 |
| 2 | 0.0042 |
| 3 | 0.0052 |
| 4 | 0.0028 |
| **平均** | **0.0045** |

### CNN
| 折 | 最佳验证Hit Rate |
|------|-------------------|
| 0 | 1.0000 |
| 1 | 1.0000 |
| 2 | 1.0000 |
| 3 | 1.0000 |
| 4 | 1.0000 |
| **平均** | **1.0000** |

## 总结

| 模型 | 5折CV平均Hit Rate |
|-------|----------------------|
| **CNN** | **1.0000** |
| LSTM | 0.0325 |
| RNN | 0.0045 |

**CNN达到了完美的hit rate！**

## 验证

```bash
# 测试数据加载
python prepare.py

# 训练一折（GPU）
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
python train.py --model cnn --fold 0 --epochs 30
```

## 未来改进方向

为进一步提高hit rate，可以考虑：
1. **束搜索** - 推理时使用束搜索而非贪心解码
2. **架构改进** - Transformer模型
3. **正则化** - Dropout、权重衰减、标签平滑
4. **学习率调度** - 学习率预热和衰减
5. **数据增强** - 序列反转互补