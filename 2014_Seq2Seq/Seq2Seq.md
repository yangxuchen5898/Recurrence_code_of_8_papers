## Seq2Seq——基于神经网络的序列间学习（2014）

### RNN 中的文本预处理

在序列模型中，如果我拥有了 $t=1$ 到 $t=t_1$ 时序的股价，可以预测未来 $t_2$ 时刻的股价

但如果我要处理一串一串的文本，应该怎么做，可以把一整段文本当作一个序列，然后做文字续写

```python
import re
def read_time_machine(filepath='timemachine.txt'):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # 每一行文本是
    # 只保留字母，用空格代替其他符号，并转小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])
```

我们要将文字输入到机器中去，让模型认出这些数据，所以要对文本进行词元化（文本转为数字）

设置两个参数（ `lines` 表示列表元素，`token='word'` 表示以一个单词作为一个词元）

```python
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)
tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

```python
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
	# 求词表的长度
    def __len__(self):
        return len(self.idx_to_token)
	# 把词元转换为数字索引
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk())
        return [self.__getitem__(token) for token in tokens]
	# 把数字索引转换为单词（反过来）
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
	# 未知词元的索引为 0
    # 在开始时会放入一些标记词元表示这是开始部分，如<bos><eos>等
    def unk(self):
        return 0
	# 返回这个词在这个小说中出现的频率
    def token_freqs(self):
        return self._token_freqs
def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

把上面的所有功能包装成以下函数

```python
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

采样可以在 $0 \sim num_{steps}$ 做随机采样，采样固定长度的一个子序列，然后对采样出的子序列再做截断，这个截断叫做分时间步，前一个时间步做为输入 $x$ ，后一个时间步作为输出 $y$ 

<img src="https://mmbiz.qpic.cn/sz_mmbiz_png/XQ1PTdTm4Niba9DQB53iaOVibyjWVRibLciatunFrVgAp3rLM6a6C1HkFLGOJRQy8cMnU4wBJWzIpJbGJAjj1E3NHng/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1" alt="图片" style="zoom: 67%;" />

随机采样再做截断的过程如上图所示，代码如下

```python
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
```

RNN 的表示，需要传入词典长度和隐藏单元数

```python
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

然后还要进行初始化

使用张量来初始化隐状态，它的形状是（隐藏层数，批量大小，隐藏单元数）

```python
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
```

### RNN 图示

<img src="https://i0.wp.com/colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png?w=1200&ssl=1" alt="展開的遞歸神經網絡。" style="zoom:50%;" />

```python
output, h_n = nn.RNN(input)
```

`output`

- 维度：`(seq_len, batch, hidden_size)`
- 含义：和 LSTM 一样，包含了**每个时间步的隐藏状态**。

`h_n`

- 维度：`(num_layers * num_directions, batch, hidden_size)`
- 含义：最后一个时间步的隐藏状态。
- 区别在于 RNN 没有 LSTM 的 `c_n`（记忆单元），只有 `h_n`。

### 循环神经网络及其代码实现

现有从开始时刻到 $t-1$ 时刻 $x_1$ 、 $x_2$ 、......、 $x_{t-1}$ 的数据输入模型，让其预测 $t$ 时刻的数据。

把历史数据 $x_1$ 、 $x_2$ 、......、 $x_{t-1}$ 用隐藏状态 $h_{t-1}$ 表示

当前信息 $x_t$ 和所有历史信息 $h_{t-1}$ 一起来表示 $h_t$ ，写作 $h_t=f(h_{t-1},x_t)$ ，这属于一种抽象形式的表示

之所以称为循环神经网络，是因为它的 $h_t$ 在不断循环更新

循环神经网络具体可表示为
$$
\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1}\mathbf{W}_{hh} + \mathbf{b}_h)
$$
其中， $\phi$ 表示激活函数

输出为
$$
\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q
$$
想要评价语言模型的好与坏，需要在交叉熵损失的基础上，构建“困惑度”，困惑度等于 $\text{exp}(交叉熵损失)$ ，所以困惑度越接近于 $1$ ，说明模型越好，困惑度越高，说明模型越差。

词的标量最终会转化为独热编码向量，最终使用向量进行计算

向量可以设定为一个 $28$ 维的（ $26$ 个字母、 `<bos>` 和 `<eos>` ）

每次取一个小批量的数据，形状为（批量大小，时间步数），加上独热编码以后，就会变成（批量大小，时间步数， $28$ ）

为了方便训练，可以将形状变成（时间步数，批量大小， $28$ ），每次将一个时间步一起放进去训练

```python
import re
import torch
from torch.utils.data import DataLoader, Dataset

def read_time_machine(filepath='timemachine.txt'):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
tokens = [list(line) for line in lines]  # 字符级建模
vocab = Vocab(tokens)

class TimeMachineDataset(Dataset):
    def __init__(self, tokens, vocab, num_steps):
        self.vocab = vocab
        # 展平成一个长序列
        self.corpus = [vocab[token] for line in tokens for token in line]
        self.num_steps = num_steps

    def __len__(self):
        # 每个样本长度 num_steps
        return (len(self.corpus) - 1) // self.num_steps

    def __getitem__(self, idx):
        start = idx * self.num_steps
        X = self.corpus[start:start + self.num_steps]
        Y = self.corpus[start + 1:start + self.num_steps + 1]
        return torch.tensor(X), torch.tensor(Y)

# 建立数据集和迭代器
num_steps = 35
batch_size = 32
dataset = TimeMachineDataset(tokens, vocab, num_steps)
train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

### LSTM 图示

<img src="https://i0.wp.com/colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png?w=1200&ssl=1" alt="A LSTM neural network." style="zoom:50%;" />

<img src="https://i0.wp.com/colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png?w=1200&ssl=1" alt="img" style="zoom:50%;" />

<img src="https://i0.wp.com/colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png?w=1200&ssl=1" alt="img" style="zoom:50%;" />

<img src="https://i0.wp.com/colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png?w=1200&ssl=1" alt="img" style="zoom:50%;" />

<img src="https://i0.wp.com/colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png?w=1200&ssl=1" alt="img" style="zoom:50%;" />



```python
output, (h_n, c_n) = nn.LSTM(input)
```

`output`

- 维度：`(seq_len, batch, hidden_size)`（如果 `batch_first=True` 就是 `(batch, seq_len, hidden_size)`）
- 含义：记录了**每一个时间步**的隐藏状态，也就是整个序列的所有时刻的输出。常常用于后续接全连接层做分类、翻译等。

`h_n`

- 维度：`(num_layers * num_directions, batch, hidden_size)`
- 含义：最后一个时间步的 **隐藏状态**（hidden state）。如果是多层或双向 LSTM，会把最后一层/各方向的状态堆叠起来。
- 用途：可以看作序列的**最终摘要**，经常在序列分类任务里直接用这个向量。

`c_n`

- 维度同 `h_n`。
- 含义：最后一个时间步的 **细胞状态**（cell state）。这是 LSTM 独有的“记忆单元”，帮助模型保持长程依赖。
- 用途：一般在继续接着输入序列时需要（比如做序列拼接训练）；如果只是拿最后结果做预测，很多时候不直接用到 `c_n`。

### 编码器解码器的架构

编码器，在这里我们定义其基类，具体实现由后面的子类来继承

```python
from torch import nn

# 定义编码器类（Encoder）继承自nn.Module，作为编码器-解码器架构中的基础编码器类
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""

    # 初始化方法，用于构造类对象
    def __init__(self, **kwargs):
        # 调用父类nn.Module的初始化方法，确保父类属性的正确初始化
        super(Encoder, self).__init__(**kwargs)

    # 定义前向传播函数，X为输入，args为可选的额外参数
    def forward(self, X, *args):
        # 抛出“未实现错误”，表示该函数需要在子类中实现具体的逻辑
        raise NotImplementedError
```

其中， `**kwargs` 表示传入多余的预定义的参数， `*args` 代表其他的可选参数

最后一行 `raise NotImplementedError` 表示这个方法必须由子类来具体实现，基类只是规定了接口

解码器基类：

```python
# 定义解码器类（Decoder），继承自 nn.Module，是编码器-解码器架构中的基础解码器类
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""

    # 初始化函数，构造解码器对象
    def __init__(self, **kwargs):
        # 调用父类 nn.Module 的初始化方法，确保继承的属性正确初始化
        super(Decoder, self).__init__(**kwargs)

    # 初始化解码器状态的方法，输入为编码器的输出 enc_outputs 和其他可选参数
    def init_state(self, enc_outputs, *args):
        # 抛出“未实现错误”，提示需要在具体子类中实现该方法的逻辑
        raise NotImplementedError

    # 定义前向传播函数，X 为输入，state 为解码器的状态
    def forward(self, X, state):
        # 抛出“未实现错误”，表示前向传播的逻辑需要在子类中实现
        raise NotImplementedError
```

其中， `init_state` 表示设置解码器的一个初始化状态，传入参数 `enc_outputs` 表示编码器输出的最后一个时间步的隐状态，隐状态就是储存的记忆

合并编码器和解码器：

```python
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""

    # 初始化函数，接受编码器和解码器对象，以及其他可选参数
    def __init__(self, encoder, decoder, **kwargs):
        # 调用父类 nn.Module 的初始化方法
        super(EncoderDecoder, self).__init__(**kwargs)
        # 将传入的编码器对象存储为类的属性
        self.encoder = encoder
        # 将传入的解码器对象存储为类的属性
        self.decoder = decoder

    # 定义前向传播函数，enc_X 是编码器的输入，dec_X 是解码器的输入，*args 是其他可选参数
    def forward(self, enc_X, dec_X, *args):
        # 通过编码器处理输入序列 enc_X，得到编码器的输出 enc_outputs
        enc_outputs = self.encoder(enc_X, *args)
        # 初始化解码器的状态，传入编码器的输出 enc_outputs
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # 使用解码器处理解码器输入 dec_X，并结合初始化的解码器状态 dec_state 生成输出
        return self.decoder(dec_X, dec_state)
```

### 进入到 Seq2Seq

在 Seq2Seq 中，如何使用编码器解码器

#### Seq2SeqEncoder

编码器接收序列输入后，会向解码器传递隐状态作为初始化条件

- 一是使用最终的隐状态来初始化解码器的隐状态（即只传一次）
- 二是将最终的隐状态在每一个时间步都作为解码器的输入序列的一部分传递给解码器（即每次都传）

下面用编码器-解码器架构构建一个英语到法语的机器翻译 Seq2Seq 模型并训练

```python
import collections  # 导入集合模块，提供容器数据类型如字典、列表等
import math         # 导入数学模块，包含基本数学函数
import torch        # 导入PyTorch库，用于构建和训练神经网络
from torch import nn  # 从PyTorch导入神经网络模块（nn），提供常用的神经网络层

class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""

    # 初始化函数，定义编码器的参数
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        # 调用父类d2l.Encoder的初始化函数，继承其属性
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 定义嵌入层，将词汇表大小vocab_size映射到embed_size的嵌入向量空间
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 定义GRU循环神经网络层，输入嵌入向量大小为embed_size，隐藏单元为num_hiddens，层数为num_layers，指定dropout
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    # 定义前向传播函数
    def forward(self, X, *args):
        # 将输入X通过嵌入层进行词嵌入操作，X的形状变为：(batch_size, num_steps, embed_size)
        X = self.embedding(X)
        # 将输入X的维度顺序重新排列，将时间步维度放在最前面，新的形状为：(num_steps, batch_size, embed_size)
        X = X.permute(1, 0, 2)
        # 使用GRU循环神经网络处理输入，返回输出output和隐藏状态state
        # 如果没有提供初始隐藏状态，默认初始化为0
        output, state = self.rnn(X)
        # output的形状为：(num_steps, batch_size, num_hiddens)，表示每个时间步的输出
        # state的形状为：(num_layers, batch_size, num_hiddens)，表示每层的隐藏状态
        return output, state
```

其中， `def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):` ，其中 `vocab_size` 的形状为 `(batch_size, num_steps, feature_size)` ，如果采用了独热编码，则 `feature_size` 就是 `vocab_size` ；而这里我们用的是嵌入层，所以 `feature_size` 是 `embed_size` 

`self.embedding = nn.Embedding(vocab_size, embed_size)` 这样的映射有什么好处：空间中的向量可以表示词与词之间的相似性（例如，he, she, I, him这种词就可以在空间中属于相似的词，进而被映射在相近的区域），一些参数它可以自己学习

`self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)` 因为刚才 `embedding` 把维度变成了 `embed_size` ，所以 `GRU` 的输入是 `embed_size` 

#### Seq2SeqDecoder

```python
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""

    # 初始化函数，定义解码器的结构
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        # 调用父类Decoder的初始化方法
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 定义嵌入层，解码器的输入不仅仅是来自词汇表的嵌入向量，还包括来自编码器的隐状态信息。将词汇表大小vocab_size映射到embed_size维的嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 定义GRU层，输入为(embed_size + num_hiddens)，输出为num_hiddens维度的隐藏状态，使用num_layers层的GRU，支持dropout
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        # 定义一个全连接层，将GRU的输出（隐藏状态）映射到词汇表大小的输出空间，用于词汇预测
        self.dense = nn.Linear(num_hiddens, vocab_size)

    # 初始化解码器的状态，接受编码器的输出并返回解码器的初始状态
    def init_state(self, enc_outputs, *args):
        # enc_outputs[1] 是编码器的隐藏状态，作为解码器的初始状态返回
        return enc_outputs[1]

    # 前向传播函数，X是解码器的输入（词汇索引序列），state是解码器的初始状态
    def forward(self, X, state):
        # 将输入X通过嵌入层映射为嵌入向量，并将其形状从(batch_size, num_steps, embed_size)变为(num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 在递归神经网络（如 GRU 或 LSTM）中，隐藏状态 state 是一个三维张量，形状是 (num_layers, batch_size, num_hiddens)
        # 将解码器的最后一层隐藏状态重复num_steps次，形状与X保持一致，均为(num_steps, batch_size, num_hiddens)，用于每个时间步的上下文信息
        context = state[-1].repeat(X.shape[0], 1, 1)
        # 将嵌入向量X和上下文context在最后一个维度拼接，形状变为(num_steps, batch_size, embed_size + num_hiddens)
        X_and_context = torch.cat((X, context), 2)
        # 将拼接后的向量通过GRU进行处理，output是GRU的输出，state是GRU的更新后隐藏状态
        output, state = self.rnn(X_and_context, state)
        # 将GRU的输出通过全连接层转换为词汇表大小的输出，形状为(num_steps, batch_size, vocab_size)，然后将维度转换为(batch_size, num_steps, vocab_size)
        output = self.dense(output).permute(1, 0, 2)
        # 返回解码器的输出和更新后的隐藏状态
        # output的形状为(batch_size, num_steps, vocab_size)，表示每个时间步上词汇的预测分布
        # state的形状为(num_layers, batch_size, num_hiddens)，表示GRU每层的隐藏状态
        return output, state
```

`self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)` 为什么输入维度是两者相加 `embed_size + num_hiddens` ： `num_hiddens` 是刚刚 `encoder` 编码出的特征维度大小， `embed_size` 是输入的特征维度大小，相加是因为每一个时间步的输入都是 `x + num_hiddens` ，要把 `num_hiddens` 输入到每一个 `decoder` 的时间步，这样 `decoder` 可以在每一个时间步参考 `x`

![../_images/seq2seq.svg](https://d2l.ai/_images/seq2seq.svg)

`def init_state(self, enc_outputs, *args): return enc_outputs[1]` 为什么是返回编码器输出的 `[1]` ：因为这是编码器的最后一个时间步的隐状态， `enc_outputs[0]` 是编码器的预测输出，形状为 `(batch_size, num_steps, num_hiddens)` 

#### sequence_mask

在构建英语到法语的机器翻译模型中，我们的数据集是英语句子列表（输入）和与其对应的法语句子列表（标签）。在这样的翻译任务中，英语句子和法语句子有长有短，即输入是一个变长的序列，输出也是一个变长的序列，而我们的编码器的输入通常是由长度一致的子序列构成的小批量，所以数据在加载到编码器之前，需要统一进行处理。我们可以设置一个简单的规则，长度超过时间步长（num_steps）则对源序列进行截断（truncation），否则进行填充（padding），即在其末尾添加特定的“<pad>”词元，直到其长度达到num_steps，这样不同长度的序列可以以相同形状的小批量加载。 

但是，我们应该将填充词元的预测排除在损失函数的计算之外。通过零值化屏蔽不相关的项，可以让任何不相关预测的计算都是与零的乘积，结果都等于零。屏蔽函数sequence_mask实现如下：

```python
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""

    # 获取序列的最大长度，即X的第二个维度（num_steps），代表每个序列的长度
    maxlen = X.size(1)

    # 生成一个形状为 (1, maxlen) 的张量，每个位置上的值为 0 到 maxlen-1，用于构建掩码
    # 使用 [None, :] 将这个张量扩展为形状 (1, maxlen)
    # valid_len[:, None] 扩展为 (batch_size, 1)，用于对比每个序列的有效长度
    # 通过比较生成的掩码，结果形状为 (batch_size, maxlen)，每个位置为 True 或 False
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]

    # 将 X 中 mask 为 False 的位置替换为指定的 value（默认是0）
    X[~mask] = value

    # 返回屏蔽后的 X
    return X
```

其中， `def sequence_mask(X, valid_len, value=0):` 的 `value=0` 参数表示用 0 把原有的替代掉

` mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]` 前面的 `[None, :]` 表示生成一个 $0 \sim \text{maxlen}-1$ 长度的一个序列，把这里的值放在张量中的第二个维度，第一个维度设为 `None` ，即成为”列“

同理，后面的 `[:, None]` 表示把这里的值放在张量中的第一个维度，第二个维度设为 `None` ，即成为”行“

然后通过 $<$ 的比较，生成 `True` 和 `False` 的矩阵

然后把 `False` 的地方改成 $0$

#### 带屏蔽的 Softmax 交叉熵损失函数

```python
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # pred 的形状：(batch_size, num_steps, vocab_size)，即模型的预测值
    # label 的形状：(batch_size, num_steps)，即真实标签
    # valid_len 的形状：(batch_size,)，即每个序列的有效长度
    def forward(self, pred, label, valid_len):
        # 创建与 label 形状相同的全 1 张量 weights，用于标识每个词元的权重
        weights = torch.ones_like(label)

        # 调用 sequence_mask 函数，屏蔽掉无效的词元（超出 valid_len 的部分），
        # 将 weights 中的超出有效长度的部分设置为 0
        weights = sequence_mask(weights, valid_len)

        # 设置损失函数的 reduction 为 'none'，确保损失不被自动缩减（保持每个时间步的损失）
        self.reduction = 'none'

        # 计算未加权的交叉熵损失，使用 permute 将 pred 的维度从 (batch_size, num_steps, vocab_size)
        # 调整为 (batch_size, vocab_size, num_steps)，以符合 CrossEntropyLoss 的输入要求
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)

        # 根据 weights，对未加权的损失进行加权（只保留有效长度的损失，屏蔽无效部分）
        # 然后对每个序列的损失求均值（按 dim=1，即 num_steps 维度）
        weighted_loss = (unweighted_loss * weights).mean(dim=1)

        # 返回每个序列的加权损失，形状为 (batch_size,)
        return weighted_loss
```

`model_and_train.py`

```python
import os
import re
import time
import collections
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter

# Utils: Animator / Timer / Accumulator
# 训练可视化
class TBAnimator:
    def __init__(self, metric_names=('loss',), log_dir='logs'):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.metric_names = metric_names
        self.history = []

    def add(self, step, metrics):
        self.history.append((step, *metrics))
        # 写入 TensorBoard
        for name, val in zip(self.metric_names, metrics):
            self.writer.add_scalar(name, float(val), global_step=step)
        # 同时在控制台打印一行
        printable = ", ".join(f"{n}={v:.4f}" for n, v in zip(self.metric_names, metrics))
        print(f"epoch {step}: {printable}")

    def close(self):
        self.writer.close()

# 通过计算 token总数 ÷ Timer测到的秒数 来获取 tokens/sec
class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
    def stop(self):
        return time.perf_counter() - self.t0

# 累加器
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]

# 数据预处理
class Vocab:
    def __init__(self, tokens, min_freq=2,
                 reserved_tokens=('<pad>', '<bos>', '<eos>', '<unk>')):
        counter = collections.Counter([t for line in tokens for t in line])
        self.idx_to_token = list(reserved_tokens)
        self.token_to_idx = {t:i for i,t in enumerate(self.idx_to_token)}
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.token_to_idx:
                self.token_to_idx[tok] = len(self.idx_to_token)
                self.idx_to_token.append(tok)
    def __len__(self): return len(self.idx_to_token)
    def __getitem__(self, token):
        if isinstance(token, (list, tuple)):
            return [self.__getitem__(t) for t in token]
        return self.token_to_idx.get(token, self.token_to_idx['<unk>'])
    def to_tokens(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self.to_tokens(i) for i in idx]
        return self.idx_to_token[idx]

def tokenize(s):
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s.split(' ') if s else []

def truncate_pad(tokens, num_steps, pad_token):
    tokens = tokens[:num_steps]
    length = min(len(tokens), num_steps)
    return tokens + [pad_token] * (num_steps - len(tokens)), length

def build_arrays(src_lines, tgt_lines, num_steps, src_vocab, tgt_vocab):
    # Y 不包含 <bos>，只在训练时拼接 bos + Y[:, :-1]
    src_idxs, src_lens, tgt_idxs, tgt_lens = [], [], [], []
    for s, t in zip(src_lines, tgt_lines):
        s_tok = tokenize(s)
        t_tok = tokenize(t) + ['<eos>']
        s_tok, s_len = truncate_pad(s_tok, num_steps, '<pad>')
        t_tok, t_len = truncate_pad(t_tok, num_steps, '<pad>')
        src_idxs.append(src_vocab[s_tok]); src_lens.append(s_len)
        tgt_idxs.append(tgt_vocab[t_tok]); tgt_lens.append(t_len)
    X = torch.tensor(src_idxs, dtype=torch.long)
    X_len = torch.tensor(src_lens, dtype=torch.long)
    Y = torch.tensor(tgt_idxs, dtype=torch.long)
    Y_len = torch.tensor(tgt_lens, dtype=torch.long)
    return X, X_len, Y, Y_len

def load_data_nmt_local(batch_size, num_steps, path='fra.txt', num_examples=10000, direction=(0,1)):
    """
    读取本地 'fra.txt'（或你自己的并行语料）:
    每行形如: 'go .\tje vais !'
    direction=(src_col, tgt_col) 选择列方向。
    返回: train_iter, src_vocab, tgt_vocab
    """
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('\t') for line in f if '\t' in line]
        src_col, tgt_col = direction
        pairs = [(l[src_col], l[tgt_col]) for l in lines[:num_examples]]
    else:
        # 小样本内置示例（英文->法文）
        pairs = [
            ("go .", "va !"),
            ("i love you .", "je t aime ."),
            ("he is a student .", "il est etudiant ."),
            ("she is reading a book .", "elle lit un livre ."),
            ("we like machine learning .", "nous aimons l apprentissage automatique ."),
        ]

    src_tokens = [tokenize(s) for s,_ in pairs]
    tgt_tokens = [tokenize(t) for _,t in pairs]

    src_vocab = Vocab(src_tokens, min_freq=1)
    tgt_vocab = Vocab([['<eos>']] + tgt_tokens, min_freq=1)

    X, X_len, Y, Y_len = build_arrays([s for s,_ in pairs], [t for _,t in pairs],
                                      num_steps, src_vocab, tgt_vocab)
    dataset = TensorDataset(X, X_len, Y, Y_len)
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_iter, src_vocab, tgt_vocab

# Seq2Seq Model
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError
    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        dec_output, dec_state = self.decoder(dec_X, dec_state)
        return dec_output, dec_state

class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
    def forward(self, X, *args):
        # X: (batch, seq_len)
        X = self.embedding(X)            # (batch, seq_len, embed)
        X = X.permute(1, 0, 2)           # (seq_len, batch, embed)
        output, state = self.rnn(X)      # output: (seq_len, batch, hidden)
        return output, state             # state: (num_layers, batch, hidden)

class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def init_state(self, enc_outputs, *args):
        # enc_outputs = (enc_outputs_all_steps, enc_state)
        return enc_outputs[1]  # 仅用最终隐状态初始化解码器
    def forward(self, X, state):
        # X: (batch, seq_len)
        X = self.embedding(X).permute(1, 0, 2)   # (seq_len, batch, embed)
        context = state[-1].repeat(X.shape[0], 1, 1)  # (seq_len, batch, hidden)
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)    # output: (seq_len, batch, hidden)
        output = self.dense(output).permute(1, 0, 2)      # (batch, seq_len, vocab)
        return output, state

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len=None, *args, **kwargs):
        # pred: (batch, seq_len, vocab), label: (batch, seq_len)
        weights = torch.ones_like(label)
        if valid_len is not None:
            weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted = super().forward(pred.permute(0, 2, 1), label)  # (batch, seq_len)
        weighted = (unweighted * weights).mean(dim=1)               # (batch,)
        return weighted

# =========================
# Train
# =========================
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.GRU):
            for name, p in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(p)

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = MaskedSoftmaxCELoss().to(device)
    net.train()

    animator = TBAnimator(metric_names=('loss',), log_dir='logs')
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # total_loss, num_tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            # Teacher forcing: dec_input = <bos> + Y[:-1]
            bos_id = tgt_vocab['<bos>']
            bos = torch.full((Y.shape[0], 1), fill_value=bos_id, dtype=torch.long, device=device)
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)

            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss_fn(Y_hat, Y, Y_valid_len)  # (batch,)
            l.sum().backward()

            nn_utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            num_tokens = Y_valid_len.sum()
            optimizer.step()

            with torch.no_grad():
                metric.add(l.sum(), num_tokens)

        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))

    elapsed = timer.stop()
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / elapsed:.1f} tokens/sec on {str(device)}')

if __name__ == "__main__":
    # 参数
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 50  # 训练轮数先设少一点试跑
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据
    train_iter, src_vocab, tgt_vocab = load_data_nmt_local(
        batch_size=batch_size, num_steps=num_steps, path='fra.txt'
    )

    # 模型
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout).to(device)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout).to(device)
    net = EncoderDecoder(encoder, decoder).to(device)
    # 训练
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

输出

```
epoch 10: loss=0.2465
epoch 20: loss=0.1887
epoch 30: loss=0.1612
epoch 40: loss=0.1438
epoch 50: loss=0.1316
loss 0.132, 11055.4 tokens/sec on cuda
```

损失函数下降如下

![image-20250928101906237](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250928101906237.png)
