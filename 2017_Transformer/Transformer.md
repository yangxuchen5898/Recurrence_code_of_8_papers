## Transformer（2017）

```python
# 模拟输入输出流程
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设词表大小
    SRC_VOCAB = 1000
    TGT_VOCAB = 1000
    PAD_IDX = 1
    model = Transformer(
        src_pad_idx=PAD_IDX,
        trg_pad_idx=PAD_IDX,
        enc_voc_size=SRC_VOCAB,
        dec_voc_size=TGT_VOCAB,
        d_model=512,
        max_len=100,
        n_heads=8,
        ffn_hidden=2048,
        n_layers=6,
        drop_prob=0.1,
        device=device
    ).to(device)

    # 假输入：批次大小为2，序列长度为10
    src = torch.randint(2, SRC_VOCAB, (2, 10)).to(device)
    trg = torch.randint(2, TGT_VOCAB, (2, 10)).to(device)
    out = model(src, trg)
    print("输出 shape:", out.shape)  # 应该为 (batch, tgt_len, vocab_size)
```

### 代码逐行解析

大致理解模拟了怎样的训练过程

```python
SRC_VOCAB = 1000   # 源语言词表大小
TGT_VOCAB = 1000   # 目标语言词表大小
PAD_IDX = 1        # padding 符号的索引（约定为1）
```

#### 什么叫 padding 符号的索引（约定为1）

符号索引就是：在词表（vocabulary）里，每个 token（单词/符号）都有一个整数编号，如下

```
0 → <UNK> （未知词）
1 → <PAD> （padding 占位符）
2 → "I"
3 → "love"
4 → "AI"
...
```

这里我们人为约定：`<PAD>` 的编号是 1

#### 使用 `<PAD>` 的目的

掩码 (mask) 计算

- 在 Self-Attention 中，不能让模型把 `<PAD>` 当成有效输入参与计算。
- 所以我们会根据 `PAD_IDX` 生成一个 mask，把 `<PAD>` 的位置屏蔽掉。

损失函数 (loss) 计算

- 在计算交叉熵损失时，padding 位置不应该贡献梯度，否则会影响训练。

- 为了把 `<PAD>` 位置忽略掉，常见做法是：

  ```
  loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  ```

#### 定义模型

```python
model = Transformer(
    src_pad_idx=PAD_IDX,
    trg_pad_idx=PAD_IDX,
    enc_voc_size=SRC_VOCAB,
    dec_voc_size=TGT_VOCAB,
    d_model=512,
    max_len=100,
    n_heads=8,
    ffn_hidden=2048,
    n_layers=6,
    drop_prob=0.1,
    device=device
).to(device)
```

其中的

```python
src_pad_idx=PAD_IDX,
```

表示源语言（输入句子）里 padding 符号的编号

```python
trg_pad_idx=PAD_IDX,
```

表示目标语言（输出句子）里 padding 符号的编号，这样生成 decoder 的 mask 时，可以屏蔽 `<PAD>` ；计算损失函数时，可以忽略 `<PAD>`

```python
enc_voc_size=SRC_VOCAB,
```

表示 encoder 端的词表大小（源语言词表大小）

```python
dec_voc_size=TGT_VOCAB,
```

表示 decoder 端的词表大小（目标语言词表大小）

其中的其他参数的含义如下

```python 
d_model=512,       # 词向量维度 / 模型隐层维度
max_len=100,       # 位置编码的最大序列长度
n_heads=8,         # multi-head attention 头数
ffn_hidden=2048,   # 前馈层隐藏层大小
n_layers=6,        # encoder/decoder 堆叠层数
drop_prob=0.1,     # dropout 概率
```

#### 定义输入

```python
src = torch.randint(2, SRC_VOCAB, (2, 10)).to(device)
trg = torch.randint(2, TGT_VOCAB, (2, 10)).to(device)
```

表示随机生成一个 `[2, 最大词表]` 区间内的，形状为 `(2, 10)` 的二维张量，括号内的 2 代表 `batch_size` ，10 代表序列长度

