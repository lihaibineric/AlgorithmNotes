import torch as th
from torch import nn
import math
class PreAttention(nn.Module):
    def __init__(self, d_model, heads, d_k, bias):
        super().__init__()
        self.linear = nn.Linear(d_model, heads*d_k, bias = bias)
        self.heads = heads
        self.d_k = d_k
    def forward(self, x):
        #  输入`[seq_len, batch_size, d_model]` 或者`[batch_size, d_model]`.
        head_shape = x.shape[:-1]
        x = self.linear(x)
        # 将最后一个维度按照头数划分展开得到下面的张量
        x = x.view(*head_shape, self.heads, self.d_k)
        # 输出`[seq_len, batch_size, heads, d_k]` 或者 `[batch_size, heads, d_model]`
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool=True):
        super().__init__()
        # 按照注意力的头数划分维度
        self.d_k = d_model//heads
        self.heads = heads

        # 将Q，K，V矩阵初始化
        self.query = PreAttention(d_model, heads, self.d_k, bias)
        self.key = PreAttention(d_model, heads, self.d_k, bias)
        self.value = PreAttention(d_model, heads, self.d_k, bias)

        self.softmax

        # 输出层
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # 放缩
        self.scale = 1 / math.sqrt(self.d_k)
        
    
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