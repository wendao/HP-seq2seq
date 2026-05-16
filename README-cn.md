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
- **输出长度:** 18（字母表: R, L, F，外加特殊标记'<pad>', '<sos>', '<eos>'）

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
python train.py --model lstm --fold 0 --epochs 100 --num_layers 1 --hidden_size 128 --embed_dim 64
```

### 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--model` | lstm | 模型类型: rnn, lstm, 或 cnn |
| `--fold` | 0 | 交叉验证折数 (0-4) |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--hidden_size` | 32 | 隐藏层维度 |
| `--embed_dim` | 16 | 嵌入维度 |
| `--batch_size` | 64 | 批量大小 |
| `--num_layers` | 1 | 层数 |
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
- `EncoderCNN`: 多层Conv1D编码器，全局最大池化
- `DecoderCNN`: 逐步Conv1D解码器（因果架构，类似RNN）
- `--num_layers`: 编码器和解码器共用层数
- `--cross_attn`: Cross-Attention 头数（0=禁用）

| num_layers | hidden_size | embed_dim | 参数量 |
|------------|-------------|-----------|--------|
| 1 | 128 | 64 | 50,822 |
| 3 | 128 | 64 | 247,942 |
| 3 | 256 | 128 | 987,398 |

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

参数: `--num_layers 1 --hidden_size 128 --embed_dim 64`（CNN: `--num_layers 3`）

### 无 Cross-Attention

| 模型 | 折0 | 折1 | 折2 | 折3 | 折4 | **平均** |
|------|------|------|------|------|------|---------|
| **LSTM** | 0.4267 | 0.4427 | 0.4416 | 0.4072 | 0.3995 | **0.4235** |
| RNN | 0.2241 | 0.2487 | 0.2040 | 0.1979 | 0.1968 | **0.2143** |
| CNN | 0.1053 | 0.1276 | 0.1404 | 0.1250 | 0.1243 | **0.1245** |

### 有 Cross-Attention (`--cross_attn 8`)

| 模型 | 折0 | 折1 | 折2 | 折3 | 折4 | **平均** |
|------|------|------|------|------|------|---------|
| **LSTM** | 0.5613 | 0.5569 | 0.5473 | 0.5509 | 0.5521 | **0.5537** |
| RNN | 0.3459 | 0.3856 | 0.3620 | 0.3699 | 0.3570 | **0.3641** |
| CNN | 0.1077 | 0.1224 | 0.1326 | 0.1401 | 0.1325 | **0.1271** |

## 总结

| 模型 | 无CA | 有CA |
|------|------|------|
| **LSTM** | 0.4235 | **0.5537** |
| RNN | 0.2143 | **0.3641** |
| CNN | 0.1245 | 0.1271 |

## 验证

```bash
# 测试数据加载
python prepare.py

# 运行所有基线（5折CV）
bash run_baseline.sh

# 或者训练单折
python train.py --model lstm --fold 0 --epochs 100 --num_layers 1 --hidden_size 128 --embed_dim 64
```

## 未来改进方向

1. **束搜索** - 推理时使用束搜索而非贪心解码
2. **Transformer模型** - 替换CNN/RNN为transformer架构
3. **正则化** - Dropout、权重衰减、标签平滑
4. **学习率调度** - 预热和衰减
5. **数据增强** - 序列反转互补
6. **Scheduled sampling** - 逐步降低teacher forcing比例
