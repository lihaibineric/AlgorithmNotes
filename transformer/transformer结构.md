- [1. 基础结构与组成](#1-基础结构与组成)
## 1. 基础结构与组成
**(1)Encoder模块**

经典的 Transformer架构中的Encoder模块包含6个Encoder Block。每个Encoder Block包含两个子模块，分别是**多头自注意力层，和前馈块**。

- **多头自注意力层**采用的是一种 Scaled Dot-Product Attention的计算方式，实验结果表明，Multi-head可以在更细致的层面上提取不同head的特征，比单一head提取特征的效果更佳。
- **前馈全连接层**是由两个全连接层组成，线性变换中间增添一个Relu激活函数，具体的维度采用4倍关系，即多头自注意力的**d_model=512**，则层内的变换维度**d_ff=2048**。


```python
if __name__ == '__main__':
    draw(mode='loss')
    draw(mode='bleu')
```

`强调` 

![image-20241205183039568](https://gitee.com/lihaibineric/picgo/raw/master/pic/image-20241205183039568.png)