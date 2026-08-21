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

# 基线。--split 默认 random，--seed 默认 42
python train_baseline.py --model lstm --split grouped --fold 0 --epochs 100 \
    --num_layers 1 --hidden_size 128 --embed_dim 64 --cross_attn 8

# 完整矩阵：70 次（transformer 不走 CA 那一维），100 轮在 MPS 上约 10 小时。
# 或只跑一部分：
bash run_baseline.sh
MODELS="lstm rnn" SPLITS=grouped CA=0 bash run_baseline.sh
FOLDS=0 EPOCHS=20 bash run_baseline.sh

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

5 折，`EPOCHS=200`，seed 42，MPS。来源 `logs_trf/trf_grp_*.log`：

| 折 | 0 | 1 | 2 | 3 | 4 | **均值** |
|---|---|---|---|---|---|---|
| val hit rate | 0.5922 | 0.6042 | 0.5904 | 0.5857 | 0.6248 | **0.5995** |
| 最优轮的 train hit rate | 0.8911 | 0.8855 | 0.8973 | 0.9042 | 0.9059 | 0.8968 |
| 最优轮次 | 190 | 177 | 193 | 191 | 195 | 189 |

标准差 0.0157，范围 [0.5857, 0.6248]。

相比 `AUTO.md` 里记录的 ~0.46，这一提升几乎全部来自训练更久（50 → 200 轮），而不是
进一步的超参搜索——最优轮次落在 177-195，说明 50 轮时训练还没走到三分之一。而且它们
仍然紧贴 200 轮的上限，这一轮很可能同样在被截断。

#### 设备与种子的对照

`extra_logs/trf_grp_*.log` 是同一配置更早的一批，跑在 CUDA 上，且**在 `SEED` 存在
之前**：

| | 0 | 1 | 2 | 3 | 4 | 均值 | 标准差 |
|---|---|---|---|---|---|---|---|
| CUDA，无种子 | 0.6204 | 0.6140 | 0.5514 | 0.6001 | 0.6102 | 0.5992 | 0.0277 |
| MPS，seed 42 | 0.5922 | 0.6042 | 0.5904 | 0.5857 | 0.6248 | 0.5995 | 0.0157 |

两者均值只差 0.0003——换后端加种子没有移动结果。但逐折数值完全对不上：折 2 在 CUDA
上是最低的离群点（0.5514），在 MPS 上却处于中游（0.5904），折 4 则反向移动。
**这里的折间差异是运行噪声，不是折的难度差异**——本文件的早期版本仅凭一次无种子运行
就断言"折 2 始终是最难的一折"，加种子的这批并不支持这个说法。

加种子还把波动砍掉了一半（标准差 0.0277 → 0.0157）。

### 基线，分组划分

100 轮，seed 42，MPS。`--num_layers 1 --hidden_size 128 --embed_dim 64`
（CNN 为 3 层），来源 `logs/*_grouped_?.log`：

| 模型 | 0 | 1 | 2 | 3 | 4 | **均值** | 标准差 | 最优轮 train |
|---|---|---|---|---|---|---|---|---|
| LSTM + cross-attn | 0.2012 | 0.1851 | 0.1739 | 0.1993 | 0.2041 | **0.1927** | 0.0128 | 1.0000 |
| RNN + cross-attn | 0.1071 | 0.1168 | 0.1034 | 0.1049 | 0.1169 | **0.1098** | 0.0066 | 0.5600 |
| LSTM | 0.1066 | 0.1056 | 0.0996 | 0.1022 | 0.1125 | **0.1053** | 0.0049 | 0.7302 |
| CNN + cross-attn | 0.0331 | 0.0276 | 0.0298 | 0.0337 | 0.0392 | **0.0327** | 0.0044 | 0.1533 |
| CNN | 0.0315 | 0.0292 | 0.0259 | 0.0224 | 0.0381 | **0.0294** | 0.0059 | 0.1467 |
| RNN | 0.0304 | 0.0267 | 0.0170 | 0.0297 | 0.0280 | **0.0264** | 0.0054 | 0.3124 |

### 基线，随机划分

同样设置，来源 `logs/*_random_?.log`：

| 模型 | 0 | 1 | 2 | 3 | 4 | **均值** | 标准差 | 最优轮 train |
|---|---|---|---|---|---|---|---|---|
| LSTM + cross-attn | 0.5673 | 0.5795 | 0.5765 | 0.5620 | 0.5361 | **0.5643** | 0.0172 | 0.9996 |
| LSTM | 0.4510 | 0.4374 | 0.4584 | 0.4319 | 0.4294 | **0.4416** | 0.0126 | 0.8936 |
| RNN + cross-attn | 0.3766 | 0.3809 | 0.3727 | 0.3886 | 0.3728 | **0.3783** | 0.0067 | 0.6450 |
| RNN | 0.2309 | 0.2333 | 0.2375 | 0.2345 | 0.2356 | **0.2344** | 0.0025 | 0.3716 |
| CNN + cross-attn | 0.1321 | 0.1237 | 0.1354 | 0.1197 | 0.1281 | **0.1278** | 0.0063 | 0.1559 |
| CNN | 0.1267 | 0.1249 | 0.1185 | 0.1338 | 0.1257 | **0.1259** | 0.0055 | 0.1465 |

`logs/baseline_boxplot.html` 是这两张表的箱线图。

### 两种划分说明了什么

区分开循环模型的是 cross-attention，而它的收益极不均匀：随机划分下 RNN 提升 61%
（0.2344 → 0.3783），LSTM 提升 28%，CNN 只有 1.5%。CNN 几乎不动的原因在
`Seq2SeqCNN.forward`——它把 `encoder_embed`（没过任何卷积层的原始输入 embedding）
当作注意力的 memory，而 RNN 和 LSTM 传的是 `encoder_output`。它的编码器还把 20 个
位置 max-pool 成 1 个。两处瓶颈串联，注意力却加在了错误的一端。

两种划分下的排序不同。随机划分：LSTM > RNN > CNN。分组划分：RNN+CA（0.1098）与
LSTM（0.1053）基本打平，两个 CNN 变体都塌到 0.03 附近——与裸 RNN 同一档。CNN 在随机
划分拿到的那 0.126，绝大部分是背下来的输出结构。

### 为什么 Transformer 不在基线矩阵里

`train_baseline.py` 仍然支持 `--model transformer`，但它已从默认矩阵和上面所有表格中
排除。在基线设置下——1 层、21,894 参数、朴素 Adam、无学习率调度、100 轮——它跑不到
收敛：loss 到最后一轮仍在单调下降，而且它是唯一一个 train hit rate **低于** val 的
模型，这正是"还没拟合训练集"的特征。在这里量到的分数反映的是训练预算而非架构，放进
六行已收敛的结果旁边会产生误导。

真要跑，就给它调度器和足够的轮数：
`MODELS=transformer EPOCHS=400 bash run_baseline.sh`。若只是想要这个任务上的
Transformer 结果，用 `train.py` / `run_transformer.sh`。

**目前还不存在一次受控的架构对比。** `train.py` 的 0.5995 来自一个参数量是 LSTM+CA
三倍、轮数翻倍、且调过调度器的模型。"这个任务上最好的结果由 transformer 取得"成立；
"transformer 架构在同等预算下优于 LSTM"不成立——这个实验从没人跑过。

## 基线的状态

直到 2026-08-20 为止，已提交的 `train_baseline.py` 有四个问题：

| 问题 | 后果 |
|---|---|
| `inputs, targets, _ = batch` 而 `collate_fn` 只返回 2 个值 | 第一个 batch 就 `ValueError`，脚本根本跑不起来 |
| 没有 `--split` 参数，硬编码 `create_folds` | 分组划分的基线结果不可能产出 |
| 没有种子 | 运行不可复现 |
| import 时设置 `os.environ['CUDA_LAUNCH_BLOCKING'] = '1'` | 串行化每一次 CUDA kernel 启动，大幅拖慢且无声无息 |

四个问题都已修复。脚本现在接受 `--split {random,grouped}` 和 `--seed`（默认 42），
并像 `train.py` 一样按 CUDA > MPS > CPU 选择设备。上面的表就是在修复后的脚本上完整
重跑 70 次的结果。

**重跑复现出了那个丢失驱动的数字**，这反过来验证了修复前 `logs/*_grp_*.log` 和
`logs/*_ca_*.log` 的可信度：

| | 重跑（seed 42，MPS） | 旧日志 | 差 |
|---|---|---|---|
| RNN，分组 | 0.0264 | 0.0253 | +0.0011 |
| CNN，分组 | 0.0294 | 0.0299 | −0.0005 |
| LSTM，分组 | 0.1053 | 0.1039 | +0.0014 |
| CNN + CA，随机 | 0.1278 | 0.1271 | +0.0007 |
| RNN + CA，随机 | 0.3783 | 0.3641 | +0.0142 |
| LSTM + CA，随机 | 0.5643 | 0.5537 | +0.0106 |

也就是说那个丢失的驱动脚本在功能上与修复后的等价，缺的只是复现它的代码。

加种子也让折间波动收窄了。旧的 RNN 随机划分折 0 与折 1 相差 0.025，新的 5 折标准差
只有 0.0025。全部十二种配置里最大的标准差是 0.0172（LSTM+CA 随机）——折间噪声已远
小于任何模型间差异，所以 5 折均值可以直接采信，不需要重复跑。

有两类日志仍然无法复现，已从所有表格中排除：`logs/lstm_3.log`、`logs/lstm_4.log`、
`logs/rnn_2.log`、`logs/rnn_3.log`、`logs/rnn_4.log` 报告的参数量（LSTM 496,774、RNN 150,406）
用本文件里的架构**无论怎么组合** `--num_layers` / `--hidden_size` / `--embed_dim`
都凑不出来——那是不同的架构，不只是不同的超参。这五份还打印了
`Autoregressive hit rate (greedy decoding): 0.0000`，而同一次运行的 teacher-forced
值是 0.20–0.41，下一节会说明这在数学上不可能。

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
- 用修复后的脚本重跑全部基线——`logs/` 里现在存的是本仓库任何代码都复现不出来的结果
- 跑一遍随机划分下的 Transformer，给 `GROUPS.md` 补上同口径的对照行
