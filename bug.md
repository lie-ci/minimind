


## miniMindMind的预训练

## 解释miniMindBlock的结构

`MiniMindBlock` 是一层 Transformer：它把归一化、注意力、前馈网络和残差连接组织起来。`MiniMindModel` 则把多个这样的 Block 串起来，组成模型主干。以下说明对应 [model/model_minimind.py](model/model_minimind.py) 中的实现。

### 1. Block 内部的数据流

这里采用 Pre-Norm 结构，即先归一化，再进入注意力或前馈网络，最后加上残差。

```text
输入 x
  │
  ├─────────────────────────────┐
  ↓                             │
RMSNorm → Attention             │
  ↓                             │
  + ←───────────────────────────┘
  │ 得到 h
  ├─────────────────────────────┐
  ↓                             │
RMSNorm → FeedForward / MoE      │
  ↓                             │
  + ←───────────────────────────┘
  │
输出 y
```

输入 `hidden_states` 通常是 `[batch_size, seq_len, hidden_size]`，即每个样本中每个 token 的向量。Block 的输出保持相同形状，可以继续传给下一层。

第一部分是注意力子模块：

```python
residual = hidden_states
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states), position_embeddings,
    past_key_value, use_cache, attention_mask
)
hidden_states += residual
```

先保存原始输入，再归一化并进行注意力计算，最后加回原始输入。省略位置、掩码和缓存参数后，可以写成：

```text
h = x + Attention(RMSNorm(x))
```

- `RMSNorm` 调整每个 token 向量的数值尺度，帮助训练保持稳定。
- `Attention` 让一个 token 从允许关注的其他 token 中汇集信息。
- `position_embeddings` 提供位置信息；本项目主干传入的是 RoPE 使用的余弦、正弦值。
- `attention_mask` 控制哪些位置可以被关注。
- `past_key_value` 传入历史 KV 缓存，`present_key_value` 返回本层缓存，配合 `use_cache` 用于加速逐 token 生成。
- 残差连接让注意力结果作为对原始表示的修正，同时为梯度传播提供直接路径。

第二部分是前馈子模块：

```python
hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
```

对应：

```text
y = h + MLP(RMSNorm(h))
```

前馈网络通常对每个 token 的向量分别做非线性变换，进一步加工特征。注意力负责 token 之间的信息交互，前馈网络负责每个 token 内部的特征加工。

`config.use_moe` 决定使用普通的 `FeedForward`，还是带专家和路由机制的 `MOEFeedForward`。`post_attention_layernorm` 虽然叫“注意力后的归一化”，但它位于 MLP 之前，因此仍然是 Pre-Norm。

最后，Block 返回本层输出和本层缓存：

```python
return hidden_states, present_key_value
```

### 2. 这里如何调用其他类中的模块

`__init__` 负责创建并保存模块实例，例如：

```python
self.self_attn = Attention(config)
```

`forward` 负责使用这些实例进行计算：

```python
hidden_states, present_key_value = self.self_attn(...)
```

因为这些模块继承了 `nn.Module`，调用 `self.self_attn(...)` 时，PyTorch 会通过模块的调用机制执行 `Attention.forward(...)`。通常直接调用模块实例，不手动调用 `.forward(...)`，这样也能保留 PyTorch 的钩子等机制。

| Block 中的调用 | 对应执行的方法 |
| --- | --- |
| `self.input_layernorm(...)` | 第一个 `RMSNorm` 实例的 `forward(...)` |
| `self.self_attn(...)` | `Attention.forward(...)` |
| `self.post_attention_layernorm(...)` | 第二个 `RMSNorm` 实例的 `forward(...)` |
| `self.mlp(...)` | `FeedForward.forward(...)` 或 `MOEFeedForward.forward(...)` |

因此，各个类实现具体计算，`MiniMindBlock` 组织计算顺序和残差连接。外部调用 `block(...)` 时，同样会执行 `MiniMindBlock.forward(...)`。

### 3. Model 主干做什么：依次调用多个 Block

`MiniMindModel` 负责更大范围的流程：把 token ID 转成向量，准备位置信息，依次调用多个 Block，再做最终归一化，并管理各层缓存。

```text
输入 token ID：[batch_size, seq_len]
       ↓
Embedding：ID → 向量
       ↓
Dropout
       ↓
Block 0
       ↓
Block 1
       ↓
……
       ↓
Block N-1
       ↓
最终 RMSNorm
       ↓
隐藏向量：[batch_size, seq_len, hidden_size]
```

主干在初始化时创建多个 Block：

```python
self.layers = nn.ModuleList([
    MiniMindBlock(l, config)
    for l in range(self.num_hidden_layers)
])
```

每个 Block 都是独立实例，有自己的一套参数。`nn.ModuleList` 负责注册这些子模块，让 PyTorch 能管理它们的参数；实际调用顺序由 `forward` 中的循环决定：

```python
for layer, past_key_value in zip(self.layers, past_key_values):
    hidden_states, present = layer(
        hidden_states,
        position_embeddings,
        past_key_value=past_key_value,
        use_cache=use_cache,
        attention_mask=attention_mask
    )
    presents.append(present)
```

上一层的输出成为下一层的输入，让表示逐层得到加工。本项目的主干最终返回 `hidden_states`、各层缓存 `presents` 和汇总的 MoE 辅助损失 `aux_loss`。

### 4. Model 外面还有语言模型封装

本项目的 `MiniMindForCausalLM` 包含主干 `MiniMindModel` 和输出头 `lm_head`：

```text
MiniMindForCausalLM
 ├─ MiniMindModel
 │   ├─ Embedding + Dropout
 │   ├─ 多个 MiniMindBlock
 │   │   ├─ RMSNorm
 │   │   ├─ Attention
 │   │   ├─ RMSNorm
 │   │   └─ FeedForward / MoE
 │   │      （Block 中还包含两次残差相加）
 │   └─ 最终 RMSNorm
 └─ lm_head：隐藏向量 → 词表中各 token 的分数
```

主干输出的是隐藏向量，`lm_head` 将隐藏向量映射为词表分数（logits）；分数本身还不是概率。传入 `labels` 时，外层封装还会计算预测下一个 token 的损失，生成时则根据分数选择后续 token。

这几个层级的分工是：**Model 调用多个 Block，Block 调用 Attention、MLP、RMSNorm 等模块，ForCausalLM 再使用 Model 的输出完成语言模型预测。**


## pretrain流程

Pretrain 的核心是：让模型阅读文本，学习“根据前面的内容预测下一个 token”，再根据预测误差更新模型参数。入口是 [trainer/train_pretrain.py](trainer/train_pretrain.py)。

以下参数以脚本默认值为例，启动命令中的参数会覆盖默认值。例如，`训练进程.md` 中记录的命令使用 `batch_size=4`、`max_seq_len=256`、`epochs=1`，对应输入形状是 `[4, 256]`。

### 1. 初始化环境、模型和优化器

- 初始化训练设备、随机种子、混合精度和可选的分布式训练环境。
- 默认模型隐藏维度为 `768`，包含 `8` 层 Transformer。
- `from_weight=none` 表示从随机初始化的模型开始；开启 `from_resume=1` 时，尝试恢复模型、优化器和训练进度。
- tokenizer 直接加载 `model/` 中已有的文件，pretrain 不会重新训练 tokenizer。
- 使用 AdamW 优化器。

### 2. 文本转成 token ID，并组成 batch

数据处理在 [dataset/lm_dataset.py](dataset/lm_dataset.py) 的 `PretrainDataset` 中。默认读取 `dataset/pretrain_t2t_mini.jsonl` 中每条数据的 `text` 字段。

```text
原始文本
   ↓
tokenizer 转成 token ID
   ↓
截断到最多 max_seq_len - 2 个 token
   ↓
添加开头 BOS 和结尾 EOS
   ↓
不足 max_seq_len 的部分补 PAD
   ↓
多条文本组成一个 batch
```

默认 `max_seq_len=340`、`batch_size=32`，因此 `input_ids` 的形状是 `[32, 340]`：32 条文本，每条包含 340 个 token ID（包括特殊 token 和补齐位置）。

`labels` 由 `input_ids` 复制而来，PAD 对应的标签设置为 `-100`，计算损失时忽略这些目标位置。

### 3. 前向计算：从 token ID 得到预测分数

模型实现在 [model/model_minimind.py](model/model_minimind.py)。这里的 token ID 就是上一步组成 batch 后的 `input_ids`。

```text
input_ids：[batch_size, seq_len]
   ↓
Embedding：每个 token ID → 一个 768 维向量
   ↓
Dropout（默认概率为 0）
   ↓
8 层 Transformer，每层包含：
   RMSNorm → 因果自注意力（使用 RoPE 位置信息）→ 残差相加
   RMSNorm → 前馈网络 FFN                    → 残差相加
   ↓
最终 RMSNorm
   ↓
LM Head：输出词表中每个 token 的预测分数 logits
```

Embedding 可以理解为一张可训练的查找表：每个 token ID 对应一个向量。不是把整个 batch 压成一个向量，而是为每个位置的 ID 分别查出向量，保留 batch 和序列两个维度。

```python
# 简化示例：2 条文本，每条 4 个 token
input_ids = [
    [1, 25, 316, 2],
    [1, 87, 429, 2],
]
# Embedding 前：[2, 4]
# Embedding 后：[2, 4, 768]
```

默认参数下，各阶段的形状为：

```text
[32, 340] → Embedding → [32, 340, 768]
          → Transformer 和最终 RMSNorm → [32, 340, 768]
          → LM Head → [32, 340, 6400]
```

其中 `6400` 是默认词表大小。logits 是分数，还不是概率。因果自注意力保证每个位置只能使用当前位置及之前的内容，不能偷看后面的 token。设置 `use_moe=1` 时，普通 FFN 会换成 MoE 专家网络。

**以上整块就是模型的前向计算；训练时还会继续根据预测分数和真实标签计算 loss。**

### 4. 计算下一个 token 的预测损失

假设分词后的序列如下（仅为示意）：

```text
用于预测的位置：BOS    我    喜欢    学习
对应预测目标：  我     喜欢  学习    EOS
```

模型看到 `BOS, 我` 时，要预测下一个 token 是“喜欢”。虽然数据集返回的 `labels` 最初与输入对齐，但模型内部会错开一位：

```python
x = logits[..., :-1, :].contiguous()
y = labels[..., 1:].contiguous()
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

一次前向计算就能同时训练一段文本中所有有效的预测位置，不需要像聊天生成那样逐个 token 调用模型。

- `logits_loss`：预测下一个 token 的交叉熵损失。
- `aux_loss`：MoE 辅助损失，默认普通模型为零。
- `loss`：以上两者之和。

### 5. 反向传播与参数更新

**反向传播不是额外设计的一套网络，而是沿着前向计算建立的计算图，通过链式法则求导。** 求的是每个可训练参数对最终 loss 的影响，不是孤立地对某一层公式求导。

设所有模型参数统称为 θ：

```text
前向计算：L = Loss(f_θ(x), y)
反向传播：g = ∂L / ∂θ
参数更新：优化器根据梯度 g 修改 θ
```

PyTorch 会自动记录前向运算，通过自动微分完成反向传播，通常不需要手写每一层的求导公式。代码中的关键调用为：

```python
res = model(input_ids, labels=labels)  # 前向计算，包含预测损失
loss = res.loss + res.aux_loss
loss = loss / args.accumulation_steps
scaler.scale(loss).backward()         # 反向传播，计算并累积梯度

# 达到梯度累积步数，或到达当前 epoch 的最后一个 batch 时：
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
scaler.step(optimizer)                # AdamW 根据梯度更新参数
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

反向传播负责计算梯度，优化器负责更新参数，这是两个不同的步骤。Embedding、注意力、FFN、归一化层和输出层中的可训练参数一起学习，并不是先训练注意力，再单独训练 FFN。

默认 `batch_size=32`、`accumulation_steps=8`，单卡通常处理 8 个 batch（256 条文本）后更新一次参数；轮末不足 8 个 batch 时也会更新。`训练进程.md` 中记录的命令使用 `batch_size=4`、`accumulation_steps=8`，单卡通常每处理 32 条文本更新一次。

### 6. 重复训练、记录日志和保存权重

```text
读取一个 batch → 前向计算 → loss → 反向传播，累积梯度
                                      ↓
                         达到更新条件：裁剪梯度、更新参数、清空梯度
                                      ↓
                              继续下一个 batch
                                      ↓
                         遍历完数据集后，进入下一个 epoch
```

- 默认训练 `2` 个 epoch，即遍历数据集两轮，每轮重新打乱顺序；学习率按余弦曲线逐渐下降。
- 默认每 `100` 个 batch 打印日志，每 `1000` 个 batch 以及每轮结束时保存权重和续训状态。
- 日志中的 `step` 是 batch 数，不是优化器更新次数。
- 日志中的 `epoch_time` 实际是当前轮预计剩余分钟数。
- 此脚本没有单独的验证集评估循环。

整个模型训练路线可以是：**Pretrain 学习文本续写 → SFT 学习按指令回答 → 可选的偏好对齐或强化学习**。`train_pretrain.py` 只负责第一阶段，完成后不会自动开始 SFT。

## 
