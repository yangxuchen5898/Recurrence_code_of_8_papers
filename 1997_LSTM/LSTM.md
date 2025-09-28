## LSTM——长短时记忆（1997）

### 摘要

RNN存在的问题：通过循环反向传播在较长时间区间内学习并存储信息往往需要很长时间，主要原因是误差反向传播不足且会衰减。

本文贡献：我们提出一种 **新颖、高效的基于梯度的方法** ——“长短期记忆”来解决该问题。

1. 在不影响效果的情况下 **对梯度进行截断** ，LSTM 能够通过在特殊单元中的“恒定误差旋转器”（constant error carrousels）来保持恒定的误差流，从而学习跨越超过 $1000$ 个离散时间步的最小时间延迟。

2. 乘性门单元（multiplicative gate units）可以学习开启或关闭对恒定误差流的访问。

结构特性：LSTM在空间和时间上都是局部的；其每个时间步、每个权重的计算复杂度为 $O(1)$ 。

实验数据：我们在人工数据上的实验涉及局部、分布式、实值和带噪声的模式表示。

实验结果：与RTRL、BPTT、递归级联相关（Recurrent Cascade-Correlation）、Elman网络以及神经序列分块（Neural Sequence Chunking）相比， **LSTM在更多运行中取得成功** ，而且学习速度更快。LSTM还能解决一些复杂的、人工设计的长时间延迟任务，这些任务此前从未被其他循环网络算法成功解决过。

### 引言

原则上，循环网络可以利用其反馈连接，将最近输入事件的表征以激活的形式存储起来（称为“短期记忆”，与由缓慢变化的权重所体现的“长期记忆”相对）。这一能力在许多应用中都具有潜在的重要意义，但往往耗时过长，或者根本效果不好。尽管这些方法在理论上颇具吸引力，但与例如在前馈网络中使用有限时间窗口的反向传播相比，现有方法并未提供明确的实际优势。本文将回顾这一问题的分析，并提出一种解决方案。

**问题：** 在传统的“时间反向传播”或“实时循环学习”中，沿时间向后传播的误差信号往往会出现两种情况之一：（1）爆炸，或（2）消失。反向传播误差的时间演化会指数性地依赖于权重的大小。在情况（1）中，可能会导致权重振荡；而在情况（2）中，想要学习跨越较长时间延迟的依赖关系将需要极其漫长的时间，甚至完全无法运行。

**解决方案：** 本文提出了“长短期记忆”，这是一种新型的 **循环网络结构** ，并配合合适的 **基于梯度的学习算法** 使用。LSTM的设计目标是克服这些误差反向传播中的问题。它能够学习跨越超过 $1000$ 步的时间间隔，即使在输入序列带有噪声且不可压缩的情况下，也不会丧失对短时间延迟的处理能力。这是通过一种高效的、基于梯度的算法实现的，该算法作用于一种结构，这种结构可以在特殊单元的内部状态中维持 **恒定** 的误差流（因此既不会爆炸，也不会消失），前提是在某些特定于结构的位置 **截断梯度计算** ——这样做不会影响长期误差流。

**论文结构概述**：第2节将简要回顾以往的相关工作。第3节首先概述Hochreiter（1991）关于梯度消失问题的详细分析，然后会介绍一种 **用于教学目的的恒定误差反向传播的朴素方法** ，并指出其在信息存储与提取方面 **存在的问题** 。这些问题将引出第4节所描述的LSTM架构。第5节将展示大量实验，并与其他竞争方法进行比较。LSTM的表现优于这些方法，并且能够解决其他循环网络算法从未解决过的复杂人工任务。第6节将讨论LSTM的局限性和优势。附录将包含对算法的详细描述（A.1）以及明确的误差传播公式（A.2）。

### 此前的工作

### 恒定误差反向传播

#### 指数衰减误差

**常规的时间反向传播** 

$d_k(t)$ 表示输出单元 $k$ 在时间 $t$ 的目标，使用均方误差， $k$ 的损失信号是
$$
\vartheta_k(t) = f'_k\left(net_k(t)\right) \left(d_k(t) - y^k(t)\right)
$$
其中，
$$
y^i(t)=f_i\left(net_i(t)\right)
$$
是非输入单元 $i$ 的激活值，使用了可微分的激活函数 $f_i$ ，
$$
net_i(t) = \sum_j w_{ij} y^j(t - 1)
$$
是单元 $i$ 目前的网络输入， $w_{ij}$ 是从 $j$ 到 $i$ 的连接的权重，一些非输出单元 $j$ 的反向传播误差信号是
$$
\vartheta_j(t) = f'_j\left(net_j(t)\right) \sum_i w_{ij} \vartheta_i(t + 1)
$$
对于 $w_{jl}$ 的权重更新的相应的贡献是 $\alpha \vartheta_j(t) \, y^l(t - 1)$ ，其中 $\alpha$ 是学习率， $l$ 代表了连接到 $j$ 的任意单元

**公式解析**

这个公式是在描述传统的 BPTT（Backpropagation Through Time）反向传播算法中，输出层误差信号的计算方式，可以分成几个部分理解：

1 背景

BPTT 是训练循环神经网络（RNN）的标准方法，它的思路是：

1. 把时间展开，把 RNN 看成一个深度前馈网络。
1. 在每个时间步计算误差，并反向传播到更早的时间步。
1. 按照梯度下降更新权重。

2 公式结构

公式：
$$
\vartheta_k(t) = f'_k(net_k(t)) \cdot (d_k(t) - y^k(t))
$$
解释：

- $\vartheta_k(t)$ ：时间 $t$ 时刻输出单元 $k$ 的误差信号（error signal）。
- $d_k(t)$ ：时间 $t$ 时刻，单元 $k$ 的目标输出（target）。
- $y^k(t))$ ：时间 $t$ 时刻，单元 $k$ 的实际输出（network output）。
- $net_k(t))$ ：时间 $t$ 时刻，单元 $k$ 的加权输入（weighted sum）。
- $f'_k$ ：激活函数对输入的导数（derivative of activation function）。

含义：

**误差信号 = 激活函数导数 $\times$ 输出误差（目标值 $-$ 实际值）**

这是梯度计算的第一步，直接来源于均方误差（MSE）损失函数的链式法则。

3 隐含单元的激活定义

公式：
$$
y^i(t) = f_i(net_i(t))
$$

- 单元 $i$ 在时刻 $t$ 的输出 $=$ 激活函数 $f_i$ 作用于它的净输入 $net_i(t)$ 。

4 净输入的计算

公式：

$$
net_i(t) = \sum_j w_{ij} \, y^j(t-1)
$$

- $w_{ij}$ ：从单元 $j$ 到单元 $i$ 的权重
- 输出是上一时刻 $t-1$ 的单元 $j$ 的激活值 $y^j(t-1)$ 乘以权重，再求和。

5 非输出单元的误差信号

公式：

$$
\vartheta_j(t) = f'_j(net_j(t)) \sum_i w_{ij} \, \vartheta_i(t+1)
$$

- 对于非输出层单元 $j$ ，误差信号是：
  - 激活函数导数 $\times$ 所有下一时刻单元的误差信号 $\vartheta_i(t+1)$ 乘以权重。
- 这反映了误差在时间上传播的特性： $t+1$ 时刻的误差信号会反馈到 $t$ 时刻。

6 权重更新规则

最后，权重更新：

$$
\Delta w_{jl} = \alpha \, \vartheta_j(t) \, y^l(t-1)
$$

- $\alpha$：学习率

- $\vartheta_j(t)$：当前时刻单元 $j$ 的误差信号

- $y^l(t-1)$：前一时刻单元 $l$ 的输出

含义：越早的时间步也能影响当前时刻的误差信号，但会出现梯度消失的问题——这正是“Exponentially Decaying Error”要讨论的核心。

![微信图片_2025-08-13_201308_415](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\微信图片_2025-08-13_201308_415.jpg)

**Hochreiter的分析的概述**

假设我们有一个全连接网络，它的非输入单元的索引从 $1$ 到 $n$ 。我们关注从单元 $u$ 到单元 $v$ 的局部误差流（稍后我们将看到这个分析迅速推广到全局误差流）。在时间步 $t$，某个任意单元 $u$ 产生的误差，会向时间回溯 $q$ 步，传递到任意单元 $v$ 。这种“回到过去”的误差传递，会被以下系数缩放， **它表示时刻 $t$ 的某个误差 $\vartheta_u(t)$ 对更早 $t−q$ 的误差 $\vartheta_v(t-q)$ 有多大影响， $f'_v$ 是梯度衰减因子** ：
$$
\frac{\partial \vartheta_v(t-q)}{\partial \vartheta_u(t)} =
\begin{cases}
f'_v\left(net_v(t-1)\right) w_{uv}, & q = 1; \\
f'_v\left(net_v(t-q)\right) \sum\limits_{l=1}^n \frac{\partial \vartheta_l(t-q+1)}{\partial \vartheta_u(t)} w_{lv}, & q > 1.
\end{cases}
$$
当 $l_q = v$ 且 $l_0 = u$ 时，我们得到：
$$
\frac{\partial \vartheta_v(t-q)}{\partial \vartheta_u(t)}
= \sum_{l_1=1}^n \cdots \sum_{l_{q-1}=1}^n
\prod_{m=1}^q f'_{l_m}\left(net_{l_m}(t-m)\right) w_{l_m\,l_{m-1}}
$$
（可用数学归纳法证明）

总共有 $n^{q-1}$ 项，每一项的形式是：
$$
\prod_{m=1}^q f'_{l_m} \left( net_{l_m}(t - m) \right) w_{l_m\, l_{m-1}}
$$
这些项的总和决定了总的误差反向传播量（total error back flow）。需要注意的是，由于这些求和项的符号可能不同，它们之间会发生相互抵消，因此即使增加了单元数 $n$，也不一定会增加误差流动量（error flow）。

**如果这个公式里的 $f' \cdot w>1$ ，则连乘后梯度会爆炸，如果这个公式里的 $f' \cdot w<1$ ，则连乘后梯度会消失**

![微信图片_2025-08-13_200908_444](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\微信图片_2025-08-13_200908_444.jpg)

**对公式2的直观解释**

如果都是
$$
\left| f'_{l_m}\left(net_{l_m}(t - m)\right) w_{l_m\,l_{m-1}} \right| > 1.0
$$
则会误差爆炸

如果都是
$$
\left| f'_{l_m}\left(net_{l_m}(t - m)\right) w_{l_m\,l_{m-1}} \right| < 1.0
$$
则会误差消失

sigmoid函数的导数的最大值是 $0.25$ 

如果 $y^{l_{m-1}}$ 是常数且不等于 $0$ ， $\left| f'_{l_m}\left(net_{l_m}(t - m)\right) w_{l_m\,l_{m-1}} \right|$ 当 $w_{l_m\,l_{m-1}}=\frac{1}{y^{l_{m-1}}} \, \coth\left( \frac{1}{2} net_{l_m} \right)$ 时取最大值

![微信图片_2025-08-13_200940_030](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\微信图片_2025-08-13_200940_030.jpg)

当 $|w_{l_m l_{m-1}}| \to \infty$ 时，这个表达式趋于 $0$ ；当 $|w_{l_m l_{m-1}}| < 4.0$ 时，它小于 $1.0$ （例如，如果权重的绝对最大值 $w_{max}$ 小于 $4.0$ ）。

![微信图片_2025-08-13_200946_103](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\微信图片_2025-08-13_200946_103.jpg)

因此，对于常规的逻辑Sigmoid激活函数来说，权重的绝对值小于 $4.0$ ，或趋近于 $ \infty$ 时，都会衰减。同样地，增加学习率也无济于事——它不会改变长程误差传播与短程误差传播的比例。

BPTT（反向传播通过时间）对近期的扰动过于敏感。（Bengio 等人在 1994 年给出了一个非常类似但更新的分析。）

**全局误差流**

全局由局部组成，局部会消失，那么全局也一样

**缩放因子的弱上界**

下面这个稍作扩展的梯度消失分析还不考虑 $n$（单元数量）。当 $q > 1$ 时，公式 (2) 可改写为：

$$
(W_u^T) W_v' (t-1) \prod_{m=2}^{q-1} (W F'(t-m)) \, W_v \, f_v'(net_v(t-q)),
$$

其中：

- 权重矩阵 $W$ 定义为 $[W]_{ij} := w_{ij}$。
- 单元 $v$ 的**输出权重**矩阵 $W_{u}$ 定义为 $[W_u]_{iv} := W_{iv} = w_{iv}$。
- 单元 $u$ 的**输入权重**矩阵 $W_{x}$ 定义为 $[W_x]_{ui} := W_{ui} = w_{ui}$。
- 对于 $m = 1, \dots, q$，$F'(t-m)$ 是一阶导数的对角矩阵，定义为：
  
  $$
  [F'(t-m)]_{ii} := 
  \begin{cases} 
  0, & \text{若 } i \ne j, \\ 
  f'_i(net_i(t-m)), & \text{否则} 
  \end{cases}
  $$
  
- $T$ 是转置算子；$[A]_{ij}$ 表示矩阵 $A$ 的第 $i$ 列，第 $j$ 行的元素；$|x|$ 是向量 $x$ 的第 $i$ 个分量。

使用与向量范数 $\|\cdot\|_x$ 相容的矩阵范数 $\|\cdot\|_A$，定义

$$
f'_{\max} := \max_{m=1,\dots,q}\{ \|F'(t-m)\|_A\}
$$

对于 $\max_{i=1,\dots,n} \{x_i\} \le \|x\|_x$，有

$$
|x^T y| \le n \|x\|_x \|y\|_x.
$$

并且由于 $|f'_v(net_v(t-q))| \le \|F'(t-q)\|_A \le f'_{\max}$，我们得到：

$$
\left| \frac{\partial y_v(t-q)}{\partial y_u(t)} \right|
\le n \,(f'_{\max})^q \, \|W_v\|_x \, \|W_{u^T}\|_x \, \|W\|_A^{q-2} 
\le n \,(f'_{\max} \, \|W\|_A)^q.
$$

这个不等式来自：

$$
\|W_v\|_x = \|W e_v\|_x \le \|W\|_A \, \|e_v\|_x = \|W\|_A,
$$

$$
\|W_{u^T}\|_x = \|e_u W\|_x \le \|W\|_A \, \|e_u\|_x = \|W\|_A,
$$

其中 $e_k$ 是第 $k$ 个分量为 1，其余为 0 的单位向量。

关于这个上界的说明

这是一个**弱且极端**的上界，只有当所有 $\|F'(t-m)\|_A$ 都取到最大值，并且所有从单元 $u$ 到单元 $v$ 的误差信号路径上每条路径的贡献都同向叠加时，才能达到该上界。

然而，通常情况下一般的 $\|F'(t-m)\|_A$ 反而更小

例如，取矩阵范数

$$
\|W\|_A := \max_r \sum_s |w_{rs}|
$$

以及向量范数

$$
\|x\|_x := \max_r |x_r|
$$

对于 logistic sigmoid，有 $f'_{\max} = 0.25$。

如果

$$
|w_{ij}| \le w_{\max} < \frac{4.0}{n}, \quad \forall i,j,
$$

则 $\|W\|_A \le n w_{\max} < 4.0$ 会导致指数级衰减。令

$$
\tau := \frac{n w_{\max}}{4.0} < 1.0,
$$

得到

$$
\left| \frac{\partial y_v(t-q)}{\partial y_u(t)} \right| \le n (\tau)^q.
$$

这意味着梯度会随 $q$ 呈指数衰减（梯度消失）。

以上证明看不懂

**结论概括：当网络权重的绝对值足够小（如 $|w_{ij}| < 4.0/n$ ）时，梯度会随时间步数 q 呈指数衰减，导致全局误差信号迅速消失，即出现严重的梯度消失问题。**

#### 恒定误差流：朴素的方法

**单个单元**

为了避免误差信号消失，我们应该如何通过一个只有自身连接的单元 $j$ 得到恒定的误差流？那么根据前面规则，在时间 $t$ ， $j$ 的局部误差反向传播量满足
$$
\vartheta_j(t) = f'_j\left(net_j(t)\right) \, \vartheta_j(t+1) \, w_{jj}
$$
为了让通过 $j$ 的误差流保持恒定，必须满足
$$
f'_j\left(net_j(t)\right) \, w_{jj} = 1.0
$$
**要让单个自连接单元保持恒定的误差流，其激活函数导数与自连接权重的乘积必须等于 $1$ 。**

**恒定误差传送**

通过积分前面的微分方程，对于任意的 $\text{net}_{j}(t)$ ，可以得到
$$
f_{j}\left(\text{net}_{j}(t)\right) = \frac{\text{net}_{j}(t)}{w_{jj}}
$$
这说明 $f_j$ 必须是线性的（ $\frac{1}{w_{jj}}$ ），并且单元 $j$ 的激活值必须保持不变：
$$
y_{j}(t+1)=f_{j}(\text{net}_{j}(t+1))=f_{j}\left( w_{j j} y^{j}(t)\right)=y^{j}(t)
$$
**为了保持误差信号恒定，LSTM 采用恒等激活函数并将自循环权重设为 1，从而形成恒定误差回转器（CEC），这是 LSTM 的核心机制。**

当然，单元 $j$ 不仅会与自己相连，还会与其他单元相连。这会引发两个显而易见且相关的问题（这些问题在所有基于梯度的方法中也同样存在）：

1. **输入权重冲突**：为简单起见，我们先关注单个额外的输入权重 $w_{ji}$ 。假设总误差可以通过在某个输入出现时打开单元 $j$ ，并在较长时间内保持其激活（直到它有助于计算出期望的输出）来减少。如果输入 $i$ 不为零，由于同一个输入权重必须同时用于存储某些输入和忽略其他输入， $w_{ji}$ 在这段时间内往往会收到相互冲突的权重更新信号（回忆 $j$ 是线性的）：这些信号会尝试让 $w_{ji}$ 同时参与 (1) 存储输入（通过打开 $j$ ）以及 (2) 保护该输入（防止 $j$ 被后续无关输入关闭）。这种冲突使得学习变得困难，因此需要一种对上下文敏感的机制来控制通过输入权重的“写入操作”。

   >假设神经元 $j$ 有一个输入来自神经元 $i$，权重是 $w_{ji}$ 。
   >
   >- 当某个输入信号出现时，系统希望把 $j$ 打开并保持很久（因为它会在未来一段时间对最终结果有用）。
   >- 但是，这条输入线 $w_{ji}$ 同时要负责“记住有用的输入”和“忽略不相关的输入”。
   >- 在学习过程中，这会产生冲突：
   >  1. 有的信号希望 $w_{ji}$ 去存储信息（把 $j$ 打开）。
   >  2. 另一些信号希望 $w_{ji}$ 保护信息（不要让后续无关的输入关掉 $j$ ）。
   >
   >就像一个开关按钮既要负责“开启”功能，又要负责“锁定”它不被别人关掉，这两种指令在训练时可能互相矛盾，导致难以学习。
   >
   >所以作者说，需要一种更聪明的机制，能根据上下文控制什么时候允许“写入”这个神经元。

2. 假设单元 $j$ 已经被激活，并且当前存储了一些先前的输入。为了简单起见，我们只关注一个额外的输出权重 $w_{kj}$ 。同一个 $w_{kj}$ 既需要在某些时刻用来读取 $j$ 的内容，又需要在其他时刻防止 $j$ 去干扰单元 $k$ 。只要 $j$ 的值不为零， $w_{kj}$ 就会在序列处理过程中收到相互冲突的权重更新信号：这些信号会让 $w_{kj}$ 在不同时间分别参与（1）访问存储在 $j$ 中的信息，以及（2）保护 $k$ 不被 $j$ 干扰。比如，在许多任务中，某些“短时延误差”在训练早期可以被减少；但在训练后期， $j$ 可能突然开始引入新的可避免误差，因为它试图参与减少更难的“长时延误差”。这种冲突会让学习过程变得困难，因此需要更具上下文敏感性的机制，通过输出权重来控制“读取操作”。

   >**核心意思**
   >
   >假设神经元 $j$ 已经“记住”了一些信息，这时候它的一个输出权重 $w_{kj}$ 同时要完成两件事：
   >
   >1. **传递信息** —— 当需要用到 $j$ 记住的内容时， $w_{kj}$ 必须把这些信息传给下游的 $k$ 。
   >2. **防止干扰** —— 当不需要用到 $j$ 的内容时， $w_{kj}$ 还得阻止 $j$ 去影响 $k$ 。
   >
   >**为什么会冲突**
   >
   >- 训练时的梯度信号在某些时刻会要求增大 $w_{kj}$ （方便读取），
   >- 而在其他时刻会要求减小 $w_{kj}$ （避免干扰）。
   >  这样就出现了“拉扯”——同一个权重被两个相反的目标更新，学习就变得困难。

输入权重冲突和输出权重冲突不仅会出现在长时间延迟的任务中，在短时间延迟的任务中也会发生。只是这种影响在长时间延迟的情况下会更加明显：

1. 时间延迟越长，存储的信息需要在更长时间里避免被扰动；
2. 尤其是在学习的后期阶段，更多已经正确的输出也需要避免被干扰。

由于这些问题，朴素的方法在大多数情况下效果并不好，只有在一些非常简单的任务中（例如输入输出表示很局部、输入模式不重复的情况）才会有效。下一节将介绍正确的解决方法。

### 长短时记忆

**记忆细胞和门控单元**

为了设计一种架构，让误差信号能够在特殊的、自连接的单元中持续流动，同时避免朴素方法的缺点，我们扩展了在第 3.2 节中介绍的自连接线性单元 $j$ 所体现的常量误差环（CEC），并引入了额外功能。具体来说，引入了 **乘性输入门单元** （input gate unit），用来保护存储在 $j$ 中的记忆内容不被无关输入干扰；同样地，引入了 **乘性输出门单元** （output gate unit），用来保护其他单元不被当前 $j$ 中无关的记忆内容干扰。

>在原本的自连接单元（Constant Error Carousel, CEC）基础上，加两个 **乘性门（multiplicative gates）**
>
>1. **输入门（input gate）** ：像一道门，决定当前时刻哪些输入可以进入记忆单元，避免无关输入扰乱已经存好的内容。
>2. **输出门（output gate）** ：像另一道门，决定当前时刻记忆单元的内容能不能输出，避免无关的记忆影响别的单元。

由此得到的更复杂的单元称为 **记忆单元（memory cell）** （见图 1）。第 $j$ 个记忆单元记作 $c_j$ 。每个记忆单元都是围绕一个带有固定自连接（即 CEC, constant error carousel）的中心线性单元构建的。除了自身的净输入 $net_{c_j}$ 外，$c_j$ 还会从一个乘法单元 $out_j$ （称为“输出门”）接收输入，并从另一个乘法单元 $in_j$ （称为“输入门”）接收输入。
 输入门 $in_j$ 在时刻 $t$ 的激活值记为 $y^{in_j}(t)$ ，输出门 $out_j$ 的激活值记为 $y^{out_j}(t)$ 。

它们满足：
$$
y^{out_j}(t) = f_{out_j}(net_{out_j}(t)), \quad y^{in_j}(t) = f_{in_j}(net_{in_j}(t))
$$
其中：
$$
net_{out_j}(t) = \sum_u w_{out_j u} y^u(t-1) \\
net_{in_j}(t) = \sum_u w_{in_j u} y^u(t-1)
$$
同时：
$$
net_{c_j}(t) = \sum_u w_{c_j u} y^u(t-1)
$$

>这是因为 LSTM 的设计要同时解决 **信息存储** 和 **信息过滤** 的问题，所以 $c$ （记忆单元）除了本身的净输入，还要额外接受来自输入门和输出门的控制信号：
>
>1. **自身净输入（ $net_{c_j}$ ）**
>   - 这是记忆单元的主要信息来源，用来 **写入新信息** 。
>   - 如果直接把外界输入写进去，容易破坏已有的记忆，所以需要输入门来控制是否更新。
>2. **输入门（ $in_j$ ）**
>   - 类似一个“写入开关”，决定当前时刻的新信息能否进入 $c$ 。
>   - 当输入门接收到无关或干扰信息时，可以关上门，保护记忆单元不被覆盖。
>3. **输出门（ $out_j$ ）**
>   - 类似一个“读取开关”，决定 $c$ 里存储的内容能否被输出到其他单元。
>   - 当当前任务不需要用到这段记忆时，关上输出门可以避免它对其他单元造成干扰。

求和符号中的下标 $u$ 可以代表输入单元、门单元、记忆单元，甚至是常规隐藏单元（如果有的话）（后面“网络拓扑”一节也有提到）。这些不同类型的单元都能提供有关网络当前状态的有用信息。例如，输入门（或输出门）可能会利用其他记忆单元的输出来决定是否在它的记忆单元中存储（或访问）某些信息。甚至还可能存在像 $w_{c_j c_j}$ 这样的循环自连接。网络的具体拓扑结构由用户自行定义，图 2 给出了一个示例。

**LSTM中的门控单元输入可以来自不同类型的单元（输入单元、门单元、记忆单元，常规隐藏单元），并且这些连接可以非常灵活，包括循环自连接。**

在时间 $t$ 时，第 $j$ 个记忆单元 $c_j$ 的输出 $y^{c_j}$ 计算公式为
$$
y^{c_j}(t) = y^{out_j}(t) \cdot h(s_{c_j}(t))
$$
其中 $s_{c_j}(t)$ 是该单元的内部状态。

- 初始状态为 $s_{c_j}(0) = 0$ 。
- 对于 $t > 0$ ，状态更新为：

$$
s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t) \cdot g(net_{c_j}(t))
$$

其中：

- $g$ 是一个可微函数，用来压缩（squash）$net_{c_j}$ 的值（类似激活函数）。
- $h$ 是一个可微函数，用来根据内部状态 $s_{c_j}$ 调整记忆单元的输出。

**描述了LSTM记忆单元的输出由输出门控制并依赖于内部状态，而内部状态通过输入门控制的新信息与之前的状态累积而成。**

**为什么需要门控单元**

为了避免输入权重冲突，输入门 $in_j$ 控制误差信号流向记忆单元 $c_j$ 的输入连接 $w_{c_j,i}$ 。为了避免输出权重冲突，输出门 $out_j$ 控制来自单元 $j$ 的输出连接的误差信号流。换句话说，网络可以利用 $in_j$ 决定何时保留或覆盖记忆单元 $c_j$ 中的信息，并利用 $out_j$ 决定何时访问记忆单元 $c_j$ ，以及何时阻止其他单元受到 $c_j$ 的影响（参见图 1）。

被困在记忆单元 CEC（Constant Error Carousel，恒定误差环）的误差信号 **无法** 发生改变——但是，通过输出门在不同时刻流入单元的不同误差信号可能会被叠加。输出门必须学会 **应该将哪些误差困在它的 CEC 中** ，方法是对这些误差进行适当的缩放。输入门则必须学会 **什么时候释放这些误差** ，同样是通过适当的缩放来实现。本质上，乘法型门单元通过开关，控制着 CEC 内恒定误差流的访问权限。

**记忆单元的 CEC 会保持已捕获的误差信号不变，但不同时间输入的误差信号可能叠加，因此输出门需要学会挑选并缩放要保留的误差，输入门需要学会挑选并缩放要释放的误差，从而通过开关控制恒定误差流的进入与退出。**

分布式输出表示通常确实需要输出门，但并不是两种门都必须存在——有时只需一种就够了。例如，在第 5 节的实验 2a 和 2b 中，只使用输入门就可以实现。实际上，如果是本地输出编码的情况，则不需要输出门——可以通过简单地将相应权重设为零来防止记忆单元干扰已经学到的输出。即便在这种情况下，输出门仍然有用：它能防止网络在尝试存储难以学习的长时延记忆时，去干扰表示易于学习的短时延记忆的激活。（例如，这在实验 1 中会非常有用。）

**在某些情况下只需要一种门（输入门或输出门）就能防止记忆单元干扰已学输出，但输出门依然有助于避免长时延记忆学习时干扰短时延记忆。**

**网络拓扑结构**

我们使用由一个输入层、一个隐藏层和一个输出层组成的网络。隐藏层是（完全）自连接的，包含记忆单元和相应的门控单元（为方便起见，我们将记忆单元和门控单元都视为位于隐藏层中）。隐藏层还可以包含“常规”的隐藏单元，用于向门控单元和记忆单元提供输入。所有层中的所有单元（除了门控单元）都与其上一层（或更高层，见实验 2a 和 2b）中的所有单元建立了有向连接（作为输入）。

**记忆单元快**

共享相同输入门和相同输出门的 $S$ 个记忆单元组成一种结构，称为“大小为 $S$ 的记忆单元块”。记忆单元块有助于信息存储——就像传统神经网络一样，在单个单元中对分布式输入进行编码并不容易。由于每个记忆单元块的门单元数量与单个记忆单元相同（即两个），这种块结构在计算上甚至可能略微更高效（见“计算复杂度”一节）。当 $S=1$ 时，记忆单元块就是一个简单的记忆单元。在实验（第 5 节）中，我们会使用不同大小的记忆单元块。

>想象有好几个记忆单元（memory cell），它们就像几个小抽屉用来存数据。 **如果这些抽屉共用同一个“输入门”和同一个“输出门”** ，那么它们就组成了一个“记忆单元块”（memory cell block），大小就是有几个抽屉（比如 $S$ 个）。
>
>这样做有两个好处：
>
>1. **方便存信息** ——多个单元共享同一套门控制机制，省去单独控制的麻烦。
>2. **效率更高** ——虽然块里有很多单元，但门的数量和一个单元一样（两个门：输入门和输出门），所以计算成本不变甚至更低。
>
>当 $S = 1$ 时，这个块其实就是一个普通的单元。实际实验中（第五节），作者会用不同大小的单元块来测试效果。

**学习**

我们使用了一种改进版的 RTRL（例如 Robinson 和 Fallside 1987 提出的方法），它能正确考虑输入门和输出门引入的变化和乘法动态效应。然而，为了确保在记忆单元内部状态中进行误差反向传播时，误差信号不会衰减（类似于截断 BPTT，例如 Williams 和 Peng 1990），当误差到达“记忆单元的网络输入”时（对于单元 $c_j$ ，包括 $net_{c_j}$ 、 $net_{in_j}$ 、 $net_{out_j}$ ），这些误差不会再向更早的时间步传播（尽管它们会用来改变输入权重）。只有在记忆单元内部，误差才会通过先前的内部状态 $s_{c_j}$ 反向传播。为了形象化说明：一旦误差信号到达记忆单元的输出，它会被输出门激活值和 $h'$ 缩放。接着它进入记忆单元的 CEC（恒等循环连接），在这里它可以无限期反向流动而不被缩放。只有当它通过输入门和 $g$ 离开记忆单元时，才会再次被输入门激活值和 $g'$ 缩放。随后它会在被截断之前用于改变输入权重（具体公式见附录）。

>1 他们用的是什么方法
>
>- 作者用的是改进版的 **RTRL（实时循环学习）** 方法，并且考虑了输入门和输出门的影响。
>- 不过为了避免误差在时间上不断衰减，他们让误差在 **记忆单元内部** 可以“保存得住”，有点像水在水池里来回流动，不会慢慢漏光。
>
>2 误差是怎么传播的
>
>- 当误差到达 **记忆单元的输入端** （比如 $net_{c_j}, net_{in_j}, net_{out_j}$ 这些）时，它不会再往更早的时间去传了（但它还是能用来更新权重）。
>- **只有在记忆单元内部** ，误差才会通过之前的状态 $s_{c_j}$ 继续往回传。
>- 当误差到达 **记忆单元的输出端** 时：
>  1. 它会先被输出门的值和 $h'$ 缩放一次。
>  2. 然后它会进入单元的 CEC（恒等循环连接），这里的误差可以 **无限期往回传** ，不会衰减。
>  3. 当它要通过输入门离开记忆单元时，会再被输入门的值和 $g'$ 缩放一次，然后才用来改权重。
>
>3 重点类比
>
>你可以想象：
>
>- **记忆单元是一个水池（CEC）** ，水流（误差）可以在里面来回流动很久而不减弱。
>- **输入门和输出门像水闸** ，水要进来或出去都要经过它们的“缩放”，这就决定了误差流动的强度。

**计算复杂度**

LSTM 的计算复杂度和 Mozer 提出的聚焦递归反向传播算法类似，只需要存储和更新导数 $\frac{\partial s_{c_j}}{\partial w_{il}}$ ，因此更新效率很高，复杂度是 $O(W)$ ，其中 $W$ 是权重数量。这样，LSTM 和完全递归网络的 BPTT 每个时间步的更新复杂度是相同的（而 RTRL 要差得多）。与完整的 BPTT 不同，LSTM 在空间和时间上是“局部”的——也就是说，不需要像 BPTT 那样存储整个序列处理中每一步的激活值，这避免了可能无限增长的栈空间占用。

**滥用问题及解决办法**

在训练早期，网络可能在不需要长期存储信息的情况下就能降低误差，因此会“滥用”记忆单元，比如把它们当作偏置单元（让它们的激活保持常量，并把输出连接当作其他单元的自适应阈值）。问题是，这些被滥用的记忆单元可能需要很长时间才能被释放出来并用于后续学习。类似的问题还会出现在两个记忆单元存储相同（冗余）信息的情况下。

作者提出了两个解决方案：

1. **顺序网络构建** ：当误差不再下降时，才向网络中添加新的记忆单元及其对应的门控单元。
2. **输出门偏置** ：给每个输出门设置一个负的初始偏置，把记忆单元初始激活值推向零；具有更负偏置的记忆单元会在之后被“分配”使用。

>1 什么是“滥用问题”
>
>在训练刚开始的时候，网络可能还不需要真的去“记住”很久以前的东西，也能把误差降下来。
>
>这时，它可能会让某些记忆单元一直保持一个固定值（比如当作一个“恒定的开关”），而不去用它们真正的记忆功能。
>
>- 这样做虽然短期有用，但会导致这些单元长期“闲置”，以后想用来记真正信息时，释放和重新利用它们会很慢。
>- 另外，如果两个记忆单元存的内容是一样的（冗余信息），也浪费了存储能力。
>
>2 为什么这是个问题
>
>想象你有一个笔记本专门记重要的事情，但你一直在第一页写“天气很好”，从不更新。
>
>这个笔记本看似在“工作”，但实际上没有用来记录真正有价值的东西。等到真的需要记重要事情时，你还得先擦掉那句没用的话，非常麻烦。
>
>3 两个解决方法
>
>1. **逐步增加记忆单元（Sequential network construction）**
>    一开始不要给太多记忆单元，当网络误差不再下降时，再增加新的单元，这样避免一开始就闲置。
>2. **输出门负偏置（Output gate bias）**
>    给每个输出门加一个负的初始偏置，这会让记忆单元的初始输出值更接近零。
>    那些初始值更负的单元会在训练的后期才被“激活”，这样避免一开始就被滥用。

**记忆单元内部状态漂移（Internal state drift）问题及解决方法**

如果记忆单元 $c_j$ 的输入大多为正或大多为负，那么它的内部状态 $s_j$ 会随着时间漂移（变得越来越大或越来越小）。这很危险，因为此时 $h'(s_j)$ 会变得非常小，梯度就会消失。一种解决方法是选择合适的输出函数 $h$ ，但比如 $h(x) = x$ 虽然简单，却会导致记忆单元输出范围不受限制。作者提出了一个更简单有效的办法：在学习初期，给输入门 $in_j$ 一个趋近于零的初始偏置，防止内部状态漂移。这种方法有一个权衡：虽然会影响 $h'(s_j)$ 和 $y^{in_j}$、$f'_{in_j}$ 的幅度，但相比漂移造成的负面影响，这种代价可以忽略。对于 logistic sigmoid 激活函数，实验 4 和 5（第 5.4 节）表明不需要额外微调这个初始偏置。

>LSTM 在训练初期容易因为输入长期偏向一个方向而导致记忆单元状态“漂移”，作者通过 **让输入门初始偏置接近零** 的方法，有效抑制了漂移并避免梯度消失。

### 代码复现

- 遗忘门：将值朝 $0$ 减少
- 输入门：决定是不是忽略掉输入数据
- 输出门：决定是不是使用隐藏状态

$$
I_t = \sigma\left(X_t W_{xi} + H_{t-1} W_{hi} + b_i\right) \\
F_t = \sigma\left(X_t W_{xf} + H_{t-1} W_{hf} + b_f\right) \\
O_t = \sigma\left(X_t W_{xo} + H_{t-1} W_{ho} + b_o\right) \\
\tilde{C}_t = \tanh\left( X_t W_{xc} + H_{t-1} W_{hc} + b_c \right) \ \text{候选记忆单元} \\
C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t \ \text{记忆单元}
$$

如何理解候选记忆单元和记忆单元？

记忆单元就是把上一时间段的记忆和累积的候选记忆加在一起，通过输入门和遗忘门控制每次每个放进来多少
$$
H_t = O_t \odot \tanh(C_t) \ 隐藏状态
$$
以上公式都用 `torch.nn.LSTM` 封装起来

```python
class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
```

#### 实验一

[Embedded Reber Grammar的python实现](https://blog.csdn.net/qq_36810398/article/details/105898090)

```python
import random
import pandas as pd
import numpy as np

def ReberGrammar(n):
    '''
    :param n:
        how many reber string to generate
    :return:
        a list of unembedded reber strings
    '''
    graph={"B":["T1","P1"],
             "T1":["S1","X1"],
             "P1":["T1","V1"],
             "S1":["S1","X1"],
             "T2":["T2","V1"],
             "X1":["X2","S2"],
             "X2":["T2","V1"],
             "V1":["P2","V2"],
             "P2":["X2","S2"],
             "S2":["E"],
             "V2":["E"],
             "E":["end"]}
    rebers=[]
    for i in range(n):
        str_i=""
        edge_ = "B"
        while edge_ != "end":
            str_i+=edge_[0]
            sub_edge_=graph[edge_]
            edge_=random.sample(sub_edge_,1)[0]
        rebers.append(str_i)
    return(rebers)

def EmbeddedReberGrammar(rebers):
    '''
    :param rebers:
        unembedded reber strings
    :return:
        embedded reber strings
    '''
    newRebers=[]
    for _,str_ in enumerate(rebers):
        type=random.randint(0,1)
        if type==0:
            newRebers.append("BT"+str_+"TE")
        else:
            newRebers.append("BP"+str_+"PE")
    return(newRebers)
def illegalReberGrammarString(rebers):
    '''
    :param rebers:
        a list of legal reber strings
    :return:
        a list of illegal reber strings
    '''
    graph={"B":["T1","P1"],
             "T1":["S1","X1"],
             "P1":["T1","V1"],
             "S1":["S1","X1"],
             "T2":["T2","V1"],
             "X1":["X2","S2"],
             "X2":["T2","V1"],
             "V1":["P2","V2"],
             "P2":["X2","S2"],
             "S2":["E"],
             "V2":["E"],
             "E":["end"]}
    all_edges=["B","T","P","S","X","V","E"]
    illegalRebers=[]
    for i in range(len(rebers)):
        str_i=rebers[i]
        type=random.randint(0,1)
        if type==0:
            if str_i[1]=="P":
                str_i=str_i[0]+"T"+str_i[2:]
            else:
                str_i=str_i[0]+"P"+str_i[2:]
            illegalRebers.append(str_i)
        if type==1:
            L=len(str_i)
            edge_index=random.sample(range(2,L-2),1)[0]
            edge_=str_i[edge_index]
            if edge_ in ["B","E"]:
                sub_edge_=["P","T"]
            else:
                sub_edge_=graph[edge_+"1"]+graph[edge_+"2"]
            wrong_edge_=set(all_edges)-set([_[0] for _ in sub_edge_])
            replace=random.sample(wrong_edge_,1)[0]
            str_i=str_i[0:edge_index]+replace+str_i[edge_index+1:]
            illegalRebers.append(str_i)
    return(illegalRebers)
```

`model.py`

```python
import torch
from torch import nn
from torch.nn import LSTM, Linear
from torch.nn.utils.rnn import pack_padded_sequence
class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstm = LSTM(
            input_size = 7,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True,
            bidirectional = False
        )
        self.fc = Linear(64, 2)
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first = True, enforce_sorted = False)
        _, (h_n, c_n) = self.lstm(packed)
        last = h_n[-1]
        logits = self.fc(last)
        return logits
if __name__ == '__main__':
    lstm = Lstm()
    if torch.cuda.is_available():
        lstm.cuda()
    print(lstm)
```

`train.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Lstm
from torch.utils.tensorboard import SummaryWriter
VOCAB = ['B','T','P','S','X','V','E']
stoi = {c:i for i,c in enumerate(VOCAB)}
num_classes = 2
def encode_seq(s):
    idxs = [stoi[c] for c in s]
    onehots = torch.zeros(len(s), len(VOCAB))
    onehots[torch.arange(len(s)), idxs] = 1.0
    return onehots
class ReberDataset(Dataset):
    def __init__(self, legal_txt, illegal_txt):
        self.data = []
        for line in open(legal_txt, 'r', encoding='utf-8'):
            s = line.strip()
            if s: self.data.append((encode_seq(s), 1))
        for line in open(illegal_txt, 'r', encoding='utf-8'):
            s = line.strip()
            if s: self.data.append((encode_seq(s), 0))
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x, y = self.data[i]
        return x, len(x), y
def collate_fn(batch):
    batch.sort(key=lambda t: t[1], reverse=True)
    xs, lens, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    lens = torch.tensor(lens, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, lens, ys
if __name__ == "__main__":
    writer = SummaryWriter("logs")
    device = torch.device("cuda")
    legal_txt   = r".\train\10000_legal_ReberGrammar.txt"
    illegal_txt = r".\train\10000_illegal_ReberGrammar.txt"
    full = ReberDataset(legal_txt, illegal_txt)
    n = len(full)
    idx = int(n*0.8)
    train_ds, val_ds = torch.utils.data.random_split(full, [idx, n-idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    lstm = Lstm().to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    learning_rate = 0.01
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    lstm.train()
    epoch = 15
    for i in range(epoch):
        print("第", i + 1, "轮")
        train_loss = 0.0
        for x, lens, y in train_loader:
            optimizer.zero_grad()
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            logits = lstm(x, lens)
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=i + 1)
        print(train_loss)
        torch.save(lstm.state_dict(), "lstm_model_{}.pth".format(i + 1))
    writer.close()
```

输出

```
第 1 轮
172.43341886997223
第 2 轮
167.46716529130936
第 3 轮
162.64892929792404
第 4 轮
147.06039384007454
第 5 轮
111.93979389965534
第 6 轮
63.05713144689798
第 7 轮
44.57964085042477
第 8 轮
83.83688966743648
第 9 轮
31.47366362437606
第 10 轮
21.637346650473773
第 11 轮
16.053511993028224
第 12 轮
22.293177555315197
第 13 轮
14.057508518453687
第 14 轮
22.017977605573833
第 15 轮
13.253058027941734
```

![image-20250815014835857](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250815014835857.png)

`test.py`

```python
import torch
from torch.utils.data import DataLoader
from model import Lstm
from train import ReberDataset, collate_fn


device = torch.device("cuda")
lstm = Lstm()
legal_txt   = r".\test\500_legal_ReberGrammar.txt"
illegal_txt = r".\test\500_illegal_ReberGrammar.txt"
test_dataset = ReberDataset(legal_txt, illegal_txt)
n = len(test_dataset)
idx = int(n*0.8)
test_ds, val_ds = torch.utils.data.random_split(test_dataset, [idx, n-idx])
test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
lstm.to(device).eval()
state = torch.load("lstm_model_15.pth", map_location=device)
msg = lstm.load_state_dict(state, strict=True)
with torch.no_grad():
    total, correct = 0, 0
    for x, lens, y in test_loader:
        x, lens, y = x.to(device), lens.to(device), y.to(device)
        logits = lstm(x, lens)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
acc = correct/total
print(acc)
```

输出

```
0.97375
```

