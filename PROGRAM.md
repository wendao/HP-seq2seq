# HP-seq2seq 自主研究规则

Seq2seq Transformer 在 HP 序列→结构映射任务上的自动优化。
本文件由原 `PROGRAM.md` 与 `PROG.md` 合并而成（两者内容近乎重复，仅优化目标口径不同）。

## 编辑范围

**只编辑 `train.py`。** 其他文件只读。

不可以：
- 修改 `prepare.py`（数据加载与分组划分是 ground truth）
- 创建并运行新的 py 文件
- 安装新包（只能用 torch + prepare.py）
- 修改评估方式

**简洁性准则**：其他条件相同则越简单越好。小改进若带来丑陋的复杂性，不值得；删掉某些
东西还能拿到相同或更好的结果，是好结果。

## 优化目标

**最大化 val_hit_rate**（整条输出序列完全匹配的验证集比例）。

### 关于 `score = val_hit_rate - 0.5 * train_hit_rate`

`train.py` 现在会同时打印这个 score，但它是**诊断量，不是优化目标**。

引入它的初衷是惩罚过拟合，因为早期存在 train 虚高、val 尚可的配置。但 `EXP.md` 那一
轮（22 个实验、EPOCHS=50）在 score 口径下**全部**未超过 baseline，而当时的 baseline
是 val=0.3914 / train=0.4124 的欠拟合点——score 高只是因为模型还没学会。

后来把 EPOCHS 提到 200，val 从 0.39 升到 0.60（提升 53%），但 train 同时升到 0.89，
score 反而跌到 0.15。也就是说，**score 口径会拒绝这次真正的突破**。这个惩罚项系统性
地偏好训练不充分的配置。

结论：以 val_hit_rate 为准。score 只用来在两个 val 接近的配置之间做参考。

## 当前配置与结果

```python
HIDDEN_SIZE = 512    EMBED_DIM = 96      NUM_LAYERS = 4    NUM_HEADS = 4
DROPOUT = 0.1        BATCH_SIZE = 64     LR = 0.001        WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0      LR_SCHEDULER = 'warmup'               EPOCHS = 200
SPLIT = 'grouped'    SEED = 42
```

warmup + cosine，保留 0.1× 最小学习率：

```python
def lr_lambda(epoch):
    if epoch < 5:
        return 0.1 + 0.9 * epoch / 5
    progress = (epoch - 5) / (EPOCHS - 5)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
```

5 折 val hit rate 均值 **0.5992**（std 0.0277）。详见 README.md。

## 输出格式

脚本结束时打印：

```
Best validation hit rate: 0.6204
Train hit rate at best epoch: 0.8883 (epoch 182/200)
Score (val - 0.5*train): 0.1763
```

提取：

```bash
grep "Best validation hit rate" run.log
tail -n1 run.log        # 提前结束时看最后一行
```

## 记录结果

写入 `results.tsv`（**制表符**分隔，不提交 git），7 列：

```
commit	val_hit_rate	train_hit_rate	score	memory_gb	status	description
```

- commit：短哈希 7 位
- status：`keep`（val_hit_rate 优于基线）/ `discard` / `crash`
- 崩溃行的数值列填 0.0000

## 优化循环

1. 查看当前分支和 git 状态
2. 调整 `train.py`
3. `git commit -m "exp: <简要描述>"`
4. `python train.py 0 > run.log 2>&1`
5. `grep "Best validation hit rate" run.log`
6. 输出为空说明崩溃：`tail -n 50 run.log` 读堆栈，能修则修
7. 记录到 `results.tsv`
8. val_hit_rate 提高 → 保留 commit；持平或更差 → `git reset --hard HEAD~1`

**超时**：EPOCHS=200 的单折在 GPU 上约 30-40 分钟。超过预期一倍则终止视为失败。

**永远不要停止**：循环开始后不要停下来问人类。没有想法时，尝试组合之前的"近失"策略，
或者更激进的变化。

## 判定标准

| val_hit_rate 变化 | 行动 |
|---|---|
| > +0.01 | 保留，作为新基线 |
| +0.005 ~ +0.01 | 保留，但优先尝试更大的改进 |
| ±0.005 | 回退，除非简化了代码 |
| 下降 | 回退 |

单折差异小于 0.02 时不要下结论——见下方随机性一节。

---

# 经验沉淀

## 一、最重要的一条

**先确认训练轮数够不够，再调超参。**

`AUTO.md`（EPOCHS=50）、`AUTO2.md`（EPOCHS=20）、`EXP.md`（EPOCHS=50）三轮加起来
七十多个实验，把所有超参维度都扫成了"单峰、峰值很窄"。但把 EPOCHS 提到 200 之后，
val 从 0.46 直接到 0.60——**比之前所有超参微调加起来的收益都大**。

原因是最优轮次落在 179-200。在 50 轮预算下，所有超参都是在"如何在训练不充分的条件下
表现最好"这个错误问题上被优化的，得到的"sweet spot"是那个约束下的产物，不是模型的
真实最优。

推论：`AUTO.md` / `AUTO2.md` / `EXP.md` 里的敏感性结论**只在各自的 EPOCHS 下成立**。
`AUTO2.md` 自己就验证了这点——EPOCHS=20 时最优 LR 变成 0.002、BATCH 变成 32、
DROPOUT 变成 0，与 EPOCHS=50 的结论全部不同。在 200 轮下这些都需要重测。

## 二、两阶段优化法

**阶段一：架构搜索。** 固定训练技巧（LR=0.001, AdamW, 简单调度器），逐个调
HIDDEN_SIZE / EMBED_DIM / NUM_LAYERS / NUM_HEADS / DROPOUT，每次只改一个。

**阶段二：训练技巧。** 架构定下来后再调学习率、调度器、batch size、正则化、梯度裁剪。

前提是训练轮数已经足够（见上一节）。

## 三、已验证有效

1. **Warmup + Cosine 调度**：5 轮线性 warmup（0.1×→1.0×）后 cosine 衰减。相比无调度
   器或纯 cosine 有明显提升。
2. **保留最小学习率 0.1×**：防止 cosine 衰减到 0，在 EPOCHS=50 下提升约 +0.018。
3. **AdamW + weight decay 1e-4**：比纯 Adam 和 1e-5 都好。
4. **梯度裁剪 1.0**：两侧（0.5 / 5.0）都更差。
5. **Pre-LN（norm_first=True）**：3 次运行均值 0.4634，std 0.002，比 min-LR 方案
   （0.4461，std 0.017）更稳。

## 四、已验证无效或有害（EPOCHS=50 口径）

| 技巧 | 效果 | 分析 |
|---|---|---|
| Label smoothing 0.1 | -0.025 | exact-match 任务上平滑会惩罚高置信度的正确预测 |
| AdamW betas (0.9, 0.98) | -0.036 | 默认 (0.9, 0.999) 更合适 |
| BATCH_SIZE 128 | -0.054 | 更新次数减少 |
| LR 0.0005 | -0.117 | 收敛过慢 |
| LR 0.002 | -0.020 | 略高，不稳定 |
| weight tying | -0.031 | |
| plateau 调度器 | -0.043 | |

再次提醒：这些都是 50 轮预算下的结论。

## 五、随机性

`train.py` 现在有 `SEED = 42`，覆盖 Python / CPU / CUDA / MPS。cuDNN 确定性未开启
（仅 CUDA 有效且拖慢速度），所以是接近可复现而非逐比特一致。

加种子之前，同一 commit（838ecde）两次运行得到 0.4705 和 0.4336，差 0.037。当时被当
成"最优配置"记录的 0.4705 后来被证明是离群值。

**策略**：
- 差距 < 0.02 的单折结果不足以判断，跑多折或多次取平均
- 优先关注 > 0.02 的改动
- 关键结论做多次验证

## 六、常见陷阱

**用 train 指标做目标。** train hit rate 通常远高于 val（可达 0.29 的差距）。只看 val。

**cosine 衰减到 0。** 训练后期学习率几乎为 0，模型无法继续优化。设 min_lr = 0.1×。

**warmup 过长。** 占用高学习率的迭代次数。5 轮在 50/200 轮预算下都可用。

**batch size 与学习率耦合。** 增大 batch 通常需同比例增大 LR（线性缩放规则）。
BATCH_SIZE=128 的失败可能部分源于没调 LR。

**显存。** 模型很小（峰值 < 0.1GB），但 OOM 就记 crash，不要花时间微调。
`NUM_HEADS` 必须整除 `EMBED_DIM`，8 配 96 会崩。

## 七、何时停止

- 连续 5-8 个实验都没超过当前最佳
- 所有主要超参维度都覆盖过
- 最佳结果在多次运行中稳定

此时记录最终配置，更新 README.md 和 EXP.md，准备汇报。

## 八、提交规范

```
exp: <修改内容简要描述>
```

不要提交：`prepare.py` 的修改、`results.tsv`。
