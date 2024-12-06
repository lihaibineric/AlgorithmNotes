# Transformer

> 论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
>
> [大模型八股文总结文档](https://mp.weixin.qq.com/s/CtzEquBcCsFp5Wgz_ijbww)
>
> [transformer基本结构](https://zhuanlan.zhihu.com/p/338817680)

## 核心模块

### Attention Block

#### 基本定义

- Attention通过**学习不同部分的权重，将输入的序列中的重要部分显式地加权**，从而使得模型可以更好地关注与输出有关的信息。
- Attention机制的关键是**引入一种机制来动态地计算输入序列中各个位置的权重**，从而在每个时间步上，对输入序列的不同部分进行加权求和，得到当前时间步的输出。

#### Self-Attention

- **计算查询（Query）**：查询是当前时间步的输入，用于和序列中其他位置的信息进行比较
- **计算键（Key）和值（Value）**：键表示序列中其他位置的信息，值是对应位置的表示。键和值用来和查询进行比较
- **计算注意力权重**：通过将查询和键进行**内积运算**，然后应用softmax函数，得到注意力权重。这些权重表示了在当前时间步，模型应该关注序列中其他位置的重要程度
- **加权求和**：根据注意力权重将值进行加权求和，得到当前时间步的输出

在Transformer中，Self-Attention 被称为"Scaled Dot-Product Attention"，其计算过程如下：

1. 对于输入序列中的每个位置，通过计算其与所有其他位置之间的相似度得分（通常通过点积计算）
2. 对得分进行**缩放处理，以防止梯度爆炸**
3. 将得分用softmax函数转换为注意力权重，以便计算每个位置的加权和
4. 使用注意力权重对输入序列中的所有位置进行加权求和，得到每个位置的自注意输出

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})
$$

<img src="https://gitee.com/lihaibineric/picgo/raw/master/pic/image-20241206185715346.png" alt="image-20241206185715346" style="zoom:50%;" />

```python
def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
```

#### Scale处理

[https://www.zhihu.com/question/339723385/answer/3513306407](https://www.zhihu.com/question/339723385/answer/3513306407)

scaling后进行softmax操作可以使得输入的数据的分布变得更好，你可以想象下softmax的公式，数值会进入敏感区间，防止梯度消失，让模型能够更容易训练。
$$
\begin{align*}
\operatorname{var}\left[q \cdot k^{\top}\right] &= \operatorname{var}\left[\sum_{i=1}^{d_k} q_i \times k_i\right] \\
&= \sum_{i=1}^{d_k} \operatorname{var}\left[q_i \times k_i\right] \\
&= \sum_{i=1}^{d_k} \operatorname{var}\left[q_i\right] \times \operatorname{var}\left[k_i\right] \\
&= \sum_{i=1}^{d_k} 1 \\
&= d_k
\end{align*}
$$

$$

$$

1. 如果 d_k 变大，q⋅k⊤ 方差会变大。
2. 方差变大会导致向量之间元素的差值变大。
3. 元素的差值变大会导致 softmax 退化为 argmax, 也就是最大值 softmax 后的值为 1， 其他值则为0。
4. softmax 只有一个值为 1 的元素，其他都为 0 的话，反向传播的梯度会变为 0, 也就是所谓的梯度消失。

**scale 的值为 dk^0.5 其实是把 q⋅k⊤ 归一化成了一个 均值为 0，** **方差为 1 的向量。**

#### Padding处理

在 Attention 机制中，同样需要忽略 padding 部分的影响，这里以transformer encoder中的self-attention为例：self-attention中，Q和K在点积之后，需要先经过mask再进行softmax，因此，**对于要屏蔽的部分，mask之后的输出需要为负无穷**，这样softmax之后输出才为0。

```python
def prepare_mask(self, mask, query_shape, key_shape):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)
        # [seq_len_q, seq_len_k, batch_size, heads]
        return mask 
```

#### MHA

变换张量对Q，K，V分别进行线性变换，**这些变换不会改变原有张量的尺寸**，因此每个变换矩阵都是方阵，得到输出结果后，多头的作用才开始显现，每个头开始从词义层面分割输出的张量，也就是每个头都想获得一组Q，K，V进行注意力机制的计算，但是句子中的每个词的表示只获得一部分，也就是只分割了最后一维的词嵌入向量。

**采用 Multi-head Attention的原因**

1. 原始论文中提到进行Multi-head Attention的原因是将模型分为多个头，可以形成多个子空间间，让模型去关注不同方面的信息，最后再将各个方面的信息综合起来得到更好的效果。
2. 多个头进行attention计算最后再综合起来，类似于CNN中采用多个卷积核的作用，不同的卷积核提取不同的特征，关注不同的部分，最后再进行融合。

Multi-head attention允许模型**共同关注来自不同位置的不同表示子空间的信息**，如果只有一个attention head，它的平均值会削弱这个信息。

$$MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O \\ ~ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

**多头注意力作用**

这种结构设计能**让每个注意力机制去优化每个词汇的不同特征部分**，从而均衡同一种注意力机制可能产生的偏差，让词义拥有来自更多元的表达，实验表明可以提升模型效果.

**为什么要做多头注意力机制呢**？

- 一个 dot product 的注意力里面，没有什么可以学的参数。具体函数就是内积，为了识别不一样的模式，希望有不一样的计算相似度的办法。加性 attention 有一个权重可学，也许能学到一些内容。
- multi-head attention 给 h 次机会去学习 不一样的投影的方法，使得在投影进去的度量空间里面能够去匹配不同模式需要的一些相似函数，然后把 h 个 heads 拼接起来，最后再做一次投影。
- 每一个头 $h_i$ 是把 Q,K,V 通过 可以学习的 Wq, Wk, Wv 投影到 dv 上，再通过注意力函数，得到 $head_i$ 。

**多头注意力机制的实现**

1. Multi-head Attention和单一head的Attention唯一的区别就在于，其**对特征张量的最后一个维度进行了分割**，一般是对词嵌入的embedding_dim=512进行切割成head=8，这样每一个head的嵌入维度就是512/8=64，后续的Attention计算公式完全一致，只不过是在64这个维度上进行一系列的矩阵运算而已。
2. 在head=8个头上分别进行注意力规则的运算后，简单采用**拼接concat的方式对结果张量进行融合**就得到了Multi-head Attention的计算结果。

**代码实现**

```python
def forward(self, query, key, value, mask):
        # qkv: `[seq_len, batch_size, d_model]`; mask: `[seq_len, seq_len, batch_size]`
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        attn_scores = th.enisum('ibhd,jbhd->ijbh', query, key)

        attn_scores = attn_scores*self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(attn_scores)

        attn = self.dropout(attn)

         # Multiply by values
        x = th.einsum("ijbh,jbhd->ibhd", attn, value)

        self.attn = attn.detach() # 将梯度从计算图中分离出来

        # 合并多头注意力机制
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)
```

#### Attention优势

self-attention是一种通过**自身和自身进行关联的attention机制**，从而得到更好的representation来表达自身。

- self-attention是attention机制的一种特殊情况：在self-attention中，Q=K=V，序列中的每个单词(token)都和该序列中的其他所有单词(token)进行attention规则的计算。
- attention机制计算的特点在于，可以直接跨越一句话中不同距离的token，可以远距离的学习到序列的**知识依赖和语序结构**。
- self-attention可以远距离的捕捉到语义层面的特征（it的指代对象是animal）。

应用传统的RNN,LSTM,在获取长距离语义特征和结构特征的时候，需要按照序列顺序依次计算，**距离越远的联系信息的损耗越大，有效提取和捕获的可能性越小**。

但是应用self-attention时，计算过程中会直接将句子中任意两个token的联系通过一个计算步骤直接联系起来，

****



### FFN Block

FFN模块主要由两个全连接层组成，线性变换中间增添一个Relu激活函数，具体的维度采用4倍关系，即多头自注意力的**d_model=512**，则层内的变换维度**d_ff=2048**。

1. **增加模型容量**：通过先升维再降维，模型可以学习到更复杂的特征表示。升维允许模型在更广阔的空间中进行线性变换，这有助于捕捉输入数据的复杂性。
2. **非线性变换**：在升维和降维的线性变换之间，通常会加入非线性激活函数（如ReLU）。这种非线性变换使得模型能够学习到输入数据的非线性关系，从而提高模型的表达能力。
3. **减少参数数量**：如果FFN不进行升维，而是直接在原始维度上进行线性变换，那么模型的参数数量可能会非常大，这会导致过拟合的风险增加，并且计算成本也会更高。通过先升维再降维，可以在一定程度上减少参数数量，同时仍然保持模型的表达能力。

```python
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

****

### 位置编码

不同于RNN、CNN等模型，对于Transformer模型来说，位置编码的加入是必不可少的，因为**纯粹的Attention模块是无法捕捉输入顺序的，即无法区分不同位置的Token**。为此我们大体有两个选择：

1. 想办法将位置信息融入到输入中，这构成了绝对位置编码的一般做法；
2. 想办法微调一下Attention结构，使得它有能力分辨不同位置的Token，这构成了相对位置编码的一般做法。

形式上来看，绝对位置编码是相对简单的一种方案，但即便如此，也不妨碍各路研究人员的奇思妙想，也有不少的变种。一般来说，绝对位置编码会加到输入中：**在输入的第$$k$$个向量$$x_k$$中加入位置向量$$p_k$$变为$$x_k+p_k$$，其中$$p_k$$只依赖于位置编号$$k$$**。

相当于：**词向量+位置编码向量**

三角函数式位置编码，一般也称为Sinusoidal位置编码，是Google的论文[《Attention is All You Need》](https://arxiv.org/abs/1706.03762)所提出来的一个显式解：
$$
\left\{\begin{array}{l}\boldsymbol{p}_{k, 2 i}=\sin \left(k / 10000^{2 i / d}\right) \\ \boldsymbol{p}_{k, 2 i+1}=\cos \left(k / 10000^{2 i / d}\right)\end{array}\right)
$$
其中$$p_{k,2i}$$,$$p_{k,2i+1}$$分别是位置$$k$$的编码向量的第$$2i$$,$$2i+1$$个分量，$$d$$是位置向量的维度。

很明显，三角函数式位置编码的特点是**有显式的生成规律，因此可以期望于它有一定的外推性**。另外一个使用它的理由是：由于$$\sin (\alpha+\beta)=\sin \alpha \cos \beta+\cos \alpha \sin \beta$$以及$$\cos (\alpha+\beta)=\cos \alpha \cos \beta-\sin \alpha \sin \beta$$，这表明位置$$\alpha+\beta$$的向量可以表示成位置$$\alpha$$和位置$$\beta$$的向量组合，这提供了表达相对位置信息的可能性。

```python

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
```

****

### Layer Norm

#### 基本方法

LayerNorm是大模型也是transformer结构中最常用的归一化操作，简而言之，它的作用是 对特征张量按照某一维度或某几个维度进行均值为0，方差为1的归一化操作，计算公式为：
$$
\mathrm{y}=\frac{\mathrm{x}-\mathrm{E}(\mathrm{x})}{\sqrt{\mathrm{V} \operatorname{ar}(\mathrm{x})+\epsilon}} * \gamma+\beta
$$
这里的 $$x$$ 可以理解为**张量中具体某一维度的所有元素**，比如对于 shape 为 (2,2,4) 的张量 input，若指定归一化的操作为第三个维度，则会对第三个维度中的四个张量（2,2,1），各进行上述的一次计算，详细形式：

$$a_{i}=\sum_{j=1}^{m} w_{ij} x_{j}, y_{i}=f\left(a_{i}+b_{i}\right)$$

$$\bar{a}_{i}=\frac{a_{i}-\mu}{\sigma} g_{i}, \quad y_{i}=f\left(\bar{a}_{i}+b_{i}\right)$$

$$\mu=\frac{1}{n} \sum_{i=1}^{n} a_{i}, \quad \sigma=\sqrt{\frac{1}{n} \sum_{i=1}^{n}\left(a_{i}-\mu\right)^{2}}$$

这里结合PyTorch的nn.LayerNorm算子来看比较明白

```Python
nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
```

- `normalized_shape`：归一化的维度，int（最后一维）list（list里面的维度），还是以（2,2,4）为例，如果输入是int，则必须是4，如果是list，则可以是[4], [2,4], [2,2,4]，即最后一维，倒数两维，和所有维度
- `eps`：加在分母方差上的偏置项，防止分母为0
- `elementwise_affine`：是否使用可学习的参数 $$\gamma$$ 和 $$\beta$$ ，前者开始为1，后者为0，设置该变量为True，则二者均可学习随着训练过程而变化

Layer Normalization (LN) 的一个优势是不需要批训练，在单条数据内部就能归一化。LN不依赖于batch size和输入sequence的长度，因此可以用于batch size为1和RNN中。LN用于RNN效果比较明显，但是在CNN上，效果不如BN。

#### 为什么要用layer norm

任何norm的意义都是为了让使用norm的网络的输入的数据分布变得更好，也就是转换为标准正态分布，数值进入敏感度区间，以减缓梯度消失，从而更容易训练。

**如果在一个维度内进行normalization，那么在这个维度内，相对大小有意义的，是可以比较的**

在normalization后的不同的维度之间，相对大小这是没有意义的，因为NLP中：

- 对不同样本同一特征的信息进行归一化没有意义：
- 三个样本（为中华之崛起而读书；我爱中国；母爱最伟大）中，“为”、“我”、“母”归一到同一分布没有意义。
- 舍弃不了BN中舍弃的其他维度的信息，也就是同一个样本的不同维度的信息：
- “为”、“我”、“母”归一到同一分布后，第一句话中的“为”和“中”就没有可比性了，何谈同一句子之间的注意力机制？

加强一下，我们再回顾CV中：

- 对不同样本同一特征（channel）的信息进行归一化有意义：
- 因为同一个channel下的所有信息都是遵循统一规则下的大小比较的，比如黑白图中越白越靠近255，反之越黑越靠近0
- 可以舍弃其他维度的信息，也就是同一个样本的不同维度间（channel）的信息。举例来说，RGB三个通道之间互相比较意义不大

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
```

****

## 组成结构

<img src="https://gitee.com/lihaibineric/picgo/raw/master/pic/image-20241206181734587.png" alt="image-20241206181734587" style="zoom: 33%;" />

### Encoder block

每个Encoder Block包含两个子模块，分别是**多头自注意力层，和前馈块**。

- **多头自注意力层**采用的是一种 Scaled Dot-Product Attention的计算方式，实验结果表明，Multi-head可以在更细致的层面上提取不同head的特征，比单一head提取特征的效果更佳。
- **前馈全连接层**是由两个全连接层组成，线性变换中间增添一个Relu激活函数，具体的维度采用4倍关系，即多头自注意力的**d_model=512**，则层内的变换维度**d_ff=2048**。

****

### Decoder block

经典的 Transformer架构中的 Decoder模块包含 6个 Decoder Block。

每个Decoder Block包含3个子模块，分别是多头自注意力层，Encoder-Decoder Attention层，和前馈全连接层。

Attention的计算方式，最大的区别在于需要添加**look-ahead-mask，即遮掩“未来的信息”。**

- Encoder-Decoder Attention层和上一层多头自注意力层最主要的区别在于Q!=K=V，矩阵 Q来源于上一层 Decoder Block的输出，同时 K, V来源于 Encoder端的输出。
- 前馈全连接层和 Encoder中完全一样。

### Add&Norm block

Add& Norm模块接在**每一个 Encoder Block和** **Decoder** **Block中的每一个子层的后面**。（Post_norm）

- 对于每一个Encoder Block，里面的两个子层后面都有Add&Norm。
- 对于每一个Decoder Block，里面的三个子层后面都有Add&Norm。
- Add表示残差连接，作用是为了将信息无损耗的传递的更深，来增强模型的拟合能力。
- Norm表示LayerNorm，层级别的数值标准化操作，作用是防止参数过大过小导致的学习过程异常，模型收敛特别慢的问题。

##  Decoder训练和预测

1. 在Transformer结构中的Decoder模块的输入，区分于不同的 Block，最底层的Block输入有其特殊的地方。第二层到第六层的输入一致，都是上一层的输出和Encoder的输出。
2. 最底层的Block在训练阶段，每一个time step的输入是上一个time step的输入**加上真实标签序列向后移一位**。（相当于这一步会输入一个**类似于<Start>的符号告诉模型要开始输出**）具体来看，就是每一个time step的输入序列会越来越长，不断的将之前的输入融合进来。

![image-20241206190532087](https://gitee.com/lihaibineric/picgo/raw/master/pic/image-20241206190532087.png)

1. 最底层的Block在**训练阶段**，真实的代码实现中，采用的是MASK机制来**模拟输入序列不断添加**的过程。
2. 最底层的Block在预测阶段，每一个time step的输入是从time step=0开始，一直到上一个time step的预测值的累积拼接张量。具体来看，也是随着每一个time step的输入序列会越来越长。相比于训练阶段最大的不同是这里不断拼接进来的**token是每一个time step的预测值（这里的拼接内容是真实预测值，区分预测和训练）**，而不是训练阶段每一个time step取得的ground truth值。

![image-20241206190603806](https://gitee.com/lihaibineric/picgo/raw/master/pic/image-20241206190603806.png)

## Transformer相关问答

**Transformer为何使用多头注意力机制？**（为什么不使用一个头）

- 多头保证了transformer可以注意到不同子空间的信息，捕捉到更加丰富的特征信息。可以类比CNN中同时使用**多个滤波器**的作用，直观上讲，多头的注意力**有助于网络捕捉到更丰富的特征/信息。**

**Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？** （注意和第一个问题的区别）

- 使用Q/K/V不相同可以保证在不同空间进行投影，增强了表达能力，提高了泛化能力。
- 同时，由softmax函数的性质决定，实质做的是一个soft版本的arg max操作，得到的向量接近一个one-hot向量（接近程度根据这组数的数量级有所不同）。如果令Q=K，那么得到的模型大概率会得到一个类似单位矩阵的attention矩阵，**这样self-attention就退化成一个point-wise线性映射**。这样至少是违反了设计的初衷。

**在计算attention score的时候如何对padding做mask操作？**

- padding位置置为负无穷(一般来说-1000就可以)，再对attention score进行相加。对于这一点，涉及到batch_size之类的，具体的大家可以看一下实现的源代码，位置在这里：[https://github.com/huggingface/transformers/blob/aa6a29bc25b663e1311c5c4fb96b004cf8a6d2b6/src/transformers/modeling_bert.py#L720](https://link.zhihu.com/?target=https://github.com/huggingface/transformers/blob/aa6a29bc25b663e1311c5c4fb96b004cf8a6d2b6/src/transformers/modeling_bert.py#L720)
- padding位置置为负无穷而不是0，是因为后续在softmax时，$$e^0=1$$，不是0，计算会出现错误；而$$e^{-\infty} = 0$$，所以取负无穷

