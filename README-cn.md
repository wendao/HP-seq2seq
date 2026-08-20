# HP-seq2seq

HP 格点蛋白折叠映射的 seq2seq 建模：疏水/亲水序列 → 结构序列。

## 任务

| | |
|---|---|
| 输入 | HP 序列，长度 **20**，字母表 `{H, P}` |
| 输出 | 结构序列，长度 **18**，字母表 `{R, L, F}`（相对折叠方向） |
| 指标 | **hit rate** —— 验证集中**整条**输出序列完全预测正确的样本比例 |
| 数据 | `datasets/dataset20int`，24,900 条样本，5,310 种唯一输出 |

所有样本严格是 20/18 定长，因此不存在任何 padding，指标就是对 18 个结构 token
加 `<eos>` 的整序列完全匹配。错一位，该样本记 0 分。

## 项目结构

```
train.py              # Transformer seq2seq。autoresearch 循环唯一编辑的文件
train_baseline.py     # RNN / LSTM / CNN / Transformer 基线（命令行）
prepare.py            # 数据、词表、折划分、Dataset —— 只读 ground truth
run_transformer.sh    # train.py 的 5 折交叉验证，日志写入 logs_trf/
run_baseline.sh       # 基线的 5 折交叉验证，日志写入 logs/
plot_training_curves.py
datasets/dataset20int
logs/                 # 基线日志
extra_logs/           # Transformer 5 折日志（EPOCHS=200）
EXP.md GROUPS.md AUTO.md AUTO2.md PROGRAM.md    # 实验记录 + autoresearch 规则
results.tsv           # 本地实验追踪，不提交 git
```

## 快速开始

```bash
# Transformer 单折。超参是 train.py 顶部的常量，argv[1] 是折号 (0-4)。
# 设备自动选择：CUDA > MPS > CPU
python train.py 0 > run.log 2>&1
grep "Best validation hit rate" run.log

# Transformer 全 5 折 + 汇总（均值 / 标准差）
bash run_transformer.sh

# 基线
python train_baseline.py --model lstm --fold 0 --epochs 100 \
    --num_layers 1 --hidden_size 128 --embed_dim 64 --cross_attn 8
bash run_baseline.sh

# 数据自检
python prepare.py
```

在 Apple silicon 上 `train.py` 会自动选中 MPS。`run_transformer.sh` 导出了
`PYTORCH_ENABLE_MPS_FALLBACK=1`，这样 Metal 没实现的算子会回退到 CPU 而不是直接中断。

## train.py

`Seq2SeqTransformer` = `EncoderTransformer` + `DecoderTransformer`，Pre-LN
（`norm_first=True`），正弦位置编码，解码器带因果掩码缓冲区与对编码器 memory 的
cross-attention。当前配置下 1,243,654 个参数。

数据一次性 tokenize 后常驻训练设备（`LOAD_DATA_TO_GPU`），批处理走
`iter_tensor_batches` 而非 DataLoader。

### 超参数（文件顶部）

| 常量 | 当前值 | 说明 |
|---|---|---|
| `HIDDEN_SIZE` | 512 | FFN 的 `dim_feedforward` |
| `EMBED_DIM` | 96 | `d_model`，必须被 `NUM_HEADS` 整除 |
| `NUM_LAYERS` | 4 | 编码器与解码器共用 |
| `NUM_HEADS` | 4 | |
| `DROPOUT` | 0.1 | |
| `BATCH_SIZE` | 64 | |
| `LR` | 0.001 | |
| `WEIGHT_DECAY` | 1e-4 | AdamW |
| `GRAD_CLIP` | 1.0 | `None` 表示禁用 |
| `LR_SCHEDULER` | `warmup` | 5 轮线性 warmup 0.1×→1.0×，之后 cosine 衰减到 0.1× 下限 |
| `EPOCHS` | 200 | |
| `SPLIT` | `grouped` | 固定 |
| `SEED` | 42 | 固定 |

## 两种划分方式

`create_folds` 随机划分样本。由于 24,900 条样本只对应 5,310 种不同输出，同一条输出
序列会同时落进训练集和验证集，模型靠认出背过的输出就能拿到不错的分数。

`create_grouped_folds` 把共享同一输出序列的样本全部放进同一折，保证验证集的输出是
真正没见过的。这是诚实的评估方式，也是 `train.py` 使用的方式。参见 `GROUPS.md`——
两种划分下各架构的排名完全不同。

## 结果

### Transformer，分组划分（当前配置）

5 折，`EPOCHS=200`，来源 `extra_logs/trf_grp_*.log`：

| 折 | 0 | 1 | 2 | 3 | 4 | **均值** |
|---|---|---|---|---|---|---|
| val hit rate | 0.6204 | 0.6140 | 0.5514 | 0.6001 | 0.6102 | **0.5992** |
| 最优轮的 train hit rate | 0.8883 | 0.8870 | 0.8753 | 0.9056 | 0.8875 | 0.8887 |
| 最优轮次 | 182 | 179 | 186 | 200 | 191 | |

标准差 0.0277，范围 [0.5514, 0.6204]。折 2 始终是最难的一折。

相比 `AUTO.md` 里记录的 ~0.46，这一提升几乎全部来自训练更久（50 → 200 轮），而不是
进一步的超参搜索——最优轮次落在 179-200，说明 50 轮时训练还没走到三分之一。

### 基线，分组划分

100 轮，`--num_layers 1 --hidden_size 128 --embed_dim 64`（CNN 为 3 层），
来源 `logs/*_grp_*.log`：

| 模型 | 0 | 1 | 2 | 3 | 4 | **均值** |
|---|---|---|---|---|---|---|
| LSTM | 0.1053 | 0.1096 | 0.0959 | 0.0978 | 0.1107 | **0.1039** |
| CNN | 0.0293 | 0.0365 | 0.0246 | 0.0273 | 0.0319 | **0.0299** |
| RNN | 0.0290 | 0.0313 | 0.0180 | 0.0214 | 0.0266 | **0.0253** |

### 基线，随机划分

| 模型 | 0 | 1 | 2 | 3 | 4 | **均值** |
|---|---|---|---|---|---|---|
| LSTM + cross-attn | 0.5613 | 0.5569 | 0.5473 | 0.5509 | 0.5521 | **0.5537** |
| LSTM | 0.4267 | 0.4427 | 0.4416 | — | — | **0.4370**（3 折） |
| RNN + cross-attn | 0.3459 | 0.3856 | 0.3620 | 0.3699 | 0.3570 | **0.3641** |
| RNN | 0.2241 | 0.2487 | — | — | — | **0.2364**（2 折） |
| CNN + cross-attn | 0.1077 | 0.1224 | 0.1326 | 0.1401 | 0.1325 | **0.1271** |
| CNN | 0.1053 | 0.1276 | 0.1404 | 0.1250 | 0.1243 | **0.1245** |

`logs/lstm_3`、`logs/lstm_4`、`logs/rnn_2`、`logs/rnn_3`、`logs/rnn_4` 由一个
**未提交且已丢失的** `train_baseline.py` 版本产生，模型规模也不同（LSTM 496,774 vs
381,382 参数；RNN 150,406 vs 108,742），与折 0-2 不可比，因此上表排除了它们。本
README 的早期版本把它们平均了进去，那两个数字（LSTM 0.4235、RNN 0.2143）混合了两种
配置。

## 关于指标

`train.py` 和 `train_baseline.py` 都用 teacher forcing 评估——解码器在每个位置都被
喂入真实前缀，单次并行前向。对**逐 token** 指标而言这确实偏乐观，但对整序列完全匹配
来说，它与贪心自回归解码**完全等价**。

归纳法：第 1 步两者输入都是 `<sos>`；若 teacher-forced argmax 在前 *t-1* 个位置都
等于真实 token，则贪心解码已生成同样的前缀，于是第 *t* 步两者向同样的权重输入同样的
内容，argmax 必然相同。所以"teacher forcing 下整条正确" ⟺ "贪心解码复现整条"。曝光
偏差真实存在，但它只作用于模型本来就答错的样本，这些样本在两种口径下都是 0 分。

已直接验证：把 `train.py` 的模型训练到 val hit rate 0.0839，用同一份权重两种方式评估，
teacher forcing 得到 418 条完全匹配，贪心解码同样 418 条，4,980 条验证样本上零分歧
——而其中 4,264 条样本两种方式解出的序列其实是不同的。指标一致，序列不一致。

这个等价性**不适用于 beam search**，后者优化的是序列似然，可能捞回贪心错过的样本。
项目中未实现 beam search。

上面提到的那 5 份孤儿日志还报告了 `Autoregressive hit rate (greedy decoding): 0.0000`，
而同一次运行的 teacher-forced 值是 0.20-0.41。基于上述等价性，那是丢失脚本里的 bug，
不是模型的性质。

## 可复现性

`train.py` 用 `SEED = 42` 为 Python、CPU、CUDA 和 MPS 设置种子。cuDNN 确定性未开启
（仅 CUDA 有效且拖慢速度），所以运行是接近可复现但非逐比特一致。加种子之前，同一
commit 重复运行的差距可达 0.037；除非跨折平均，否则 0.02 以下的差异应视为噪声。

设备一致性：在 `DROPOUT=0.1`、`EPOCHS=200` 下，折 0 在 CUDA 上得 0.6204，MPS 上得
0.6039——相差 0.017，落在加种子前的运行波动范围内。

## 待办方向

- 推理时的 beam search —— 唯一可能真正推动指标的解码改动
- 200 轮以上：折 3 的峰值恰好落在第 200 轮，调度可能仍在截断训练
- 用已提交的脚本补齐 LSTM/RNN 随机划分缺失的那几折
- 跑一遍随机划分下的 Transformer，给 `GROUPS.md` 补上同口径的对照行
