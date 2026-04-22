# 注意力就是你所需要的一切 (Attention Is All You Need)

**作者**: Ashish Vaswani*, Noam Shazeer*, Niki Parmar*, Jakob Uszkoreit*, Llion Jones*, Aidan N. Gomez*†, Łukasz Kaiser*, Illia Polosukhin*‡

*Equal contribution. Listing order is random.

†Work performed while at Google Brain.

‡Work performed while at Google Research.

31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.

---

## 摘要

主导的序列转换模型基于复杂的循环神经网络或卷积神经网络，包含编码器和解码器。表现最佳的模型还通过注意力机制连接编码器和解码器。我们提出了一种新的简单网络架构——**Transformer**，完全基于注意力机制，完全摒弃了循环和卷积。在两个机器翻译任务上的实验表明，这些模型的质量更优，同时更易于并行化，并且训练时间显著减少。我们的模型在 WMT 2014 英语到德语翻译任务上达到了 28.4 BLEU 分数，比现有最佳结果（包括集成模型）提高了超过 2 BLEU。在 WMT 2014 英语到法语翻译任务上，我们的模型在 8 个 GPU 上训练 3.5 天后，达到了 41.0 的新单模型最佳 BLEU 分数，这只是文献中最佳模型训练成本的一小部分。

---

## 1 引言

循环神经网络、长短期记忆网络 [12] 和门控循环神经网络 [7] 特别是已被牢固地确立为序列建模和转换问题（如语言建模和机器翻译 [29, 2, 5]）的最先进方法。此后，众多研究继续努力推动循环语言模型和编码器 - 解码器架构的边界 [31, 21, 13]。

循环模型通常沿着输入和输出序列的符号位置进行计算。将位置与计算时间中的步骤对齐，它们生成隐藏状态序列 $h_t$，作为先前隐藏状态 $h_{t-1}$ 和位置 $t$ 输入的函数。这种固有的顺序性排除了训练样本内的并行化，这在更长序列长度时变得至关重要，因为内存限制限制了跨样本的批处理。最近的工作通过因式分解技巧 [18] 和条件计算 [26] 在计算效率方面取得了显著改进，同时在后一种情况下也提高了模型性能。然而，顺序计算的基本限制仍然存在。

注意力机制已成为各种任务中引人注目的序列建模和转换模型的组成部分，允许对依赖关系进行建模，而不考虑它们在输入或输出序列中的距离 [2, 16]。然而，在少数情况外 [22]，这种注意力机制与循环网络结合使用。

在这项工作中，我们提出了 **Transformer**，这是一种避免循环的模型架构，而是完全依赖注意力机制来绘制输入和输出之间的全局依赖关系。Transformer 允许显著更多的并行化，并且可以在 8 个 P100 GPU 上训练仅 12 小时后达到翻译质量的最新水平。

---

## 2 背景

减少顺序计算的目标也构成了扩展神经 GPU [20]、ByteNet [15] 和 ConvS2S [8] 的基础，它们都使用卷积神经网络作为基本构建块，为所有输入和输出位置并行计算隐藏表示。在这些模型中，将两个任意输入或输出位置的信号相关联所需的操作数量随着位置之间距离的增长而增长，ConvS2S 为线性增长，ByteNet 为对数增长。这使得学习远距离位置之间的依赖关系变得更加困难 [11]。在 Transformer 中，这被减少为恒定数量的操作，尽管以平均注意力加权位置降低有效分辨率为代价，我们通过第 3.2 节描述的多头注意力来抵消这种影响。

**自注意力**，有时称为内部注意力，是一种注意力机制，关联单个序列的不同位置以计算序列的表示。自注意力已成功用于各种任务，包括阅读理解、抽象摘要、文本蕴含和学习任务无关的句子表示 [4, 22, 23, 19]。

端到端记忆网络基于循环注意力机制而不是序列对齐循环，已被证明在简单语言问题回答和语言建模任务上表现良好 [28]。

然而，据我们所知，**Transformer 是第一个完全依赖自注意力来计算其输入和输出表示的转换模型，而不使用序列对齐的 RNN 或卷积**。在以下各节中，我们将描述 Transformer，说明自注意力的动机，并讨论其相对于 [14, 15] 和 [8] 等模型的优势。

---

## 3 模型架构

大多数有竞争力的神经序列转换模型具有编码器 - 解码器结构 [5, 2, 29]。这里，编码器将输入序列的符号表示 $(x_1, ..., x_n)$ 映射到连续表示序列 $z = (z_1, ..., z_n)$。给定 $z$，解码器然后一次一个元素地生成输出符号序列 $(y_1, ..., y_m)$。在每一步，模型都是自回归的 [9]，在生成下一个符号时将先前生成的符号作为额外输入消耗。

Transformer 遵循这一整体架构，为编码器和解码器使用堆叠的自注意力和逐点全连接层。

### 3.1 编码器和解码器堆栈

**编码器**：编码器由 $N = 6$ 个相同层的堆栈组成。每层有两个子层。第一个是多头自注意力机制，第二个是简单的逐点全连接前馈网络。我们在两个子层中的每一个周围采用残差连接 [10]，然后是层归一化 [1]。也就是说，每个子层的输出是：

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

其中 $\text{Sublayer}(x)$ 是子层自身实现的函数。为了促进这些残差连接，模型中的所有子层以及嵌入层都产生维度 $d_{\text{model}} = 512$ 的输出。

**解码器**：解码器也由 $N = 6$ 个相同层的堆栈组成。除了每个编码器层中的两个子层外，解码器插入第三个子层，该子层对编码器堆栈的输出执行多头注意力。与编码器类似，我们在每个子层周围采用残差连接，然后是层归一化。我们还修改了解码器堆栈中的自注意力子层，以防止位置关注后续位置。这种掩码与输出嵌入偏移一个位置的事实相结合，确保位置 $i$ 的预测只能依赖于位置小于 $i$ 的已知输出。

### 3.2 注意力

注意力函数可以描述为将查询和一组键值对映射到输出，其中查询、键、值和输出都是向量。输出计算为值的加权和，其中分配给每个值的权重由查询与相应键的兼容性函数计算。

#### 3.2.1 缩放点积注意力

我们将我们的特定注意力称为"**缩放点积注意力**"。输入由维度为 $d_k$ 的查询和键以及维度为 $d_v$ 的值组成。我们计算查询与所有键的点积，每个除以 $\sqrt{d_k}$，然后应用 softmax 函数以获得值的权重。

在实践中，我们同时在一组查询上计算注意力函数，将它们打包成矩阵 $Q$。键和值也被打包成矩阵 $K$ 和 $V$。我们计算输出矩阵为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

最常用的两种注意力函数是加法注意力 [2] 和点积（乘法）注意力。点积注意力与我们的算法相同，除了缩放因子 $1/\sqrt{d_k}$。加法注意力使用具有单个隐藏层的前馈网络计算兼容性函数。虽然两者在理论复杂性上相似，但点积注意力在实践中更快且更节省空间，因为它可以使用高度优化的矩阵乘法代码实现。

虽然对于较小的 $d_k$ 值，这两种机制表现相似，但对于较大的 $d_k$ 值，加法注意力优于未缩放的点积注意力 [3]。我们怀疑对于较大的 $d_k$ 值，点积变得很大，将 softmax 函数推入梯度极小的区域。为了抵消这种影响，我们将点积缩放 $1/\sqrt{d_k}$。

#### 3.2.2 多头注意力

我们发现，与其执行单个具有 $d_{\text{model}}$ 维度键、值和查询的注意力函数，不如用不同的学习线性投影线性地将查询、键和值投影 $h$ 次到 $d_k$、$d_k$ 和 $d_v$ 维度更有益。在这些投影版本的查询、键和值上，我们并行执行注意力函数，产生 $d_v$ 维输出值。将这些连接起来并再次投影，得到最终值。

多头注意力允许模型在不同位置共同关注来自不同表示子空间的信息。使用单个注意力头时，平均会抑制这一点。

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

其中投影是参数矩阵：

- $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

在这项工作中，我们使用了 $h = 8$ 个并行注意力层或头。对于每一个，我们使用 $d_k = d_v = d_{\text{model}}/h = 64$。由于每个头的维度降低，总计算成本与全维度的单头注意力相似。

#### 3.2.3 注意力在我们模型中的应用

Transformer 以三种不同方式使用多头注意力：

1. **在"编码器 - 解码器注意力"层中**：查询来自先前的解码器层，记忆键和值来自编码器的输出。这允许解码器中的每个位置关注输入序列中的所有位置。这模仿了 [31, 2, 8] 等序列到序列模型中典型的编码器 - 解码器注意力机制。

2. **编码器包含自注意力层**：在自注意力层中，所有键、值和查询都来自同一个地方，在本例中是编码器中前一层的输出。编码器中的每个位置都可以关注编码器前一层中的所有位置。

3. **解码器中的自注意力层**：允许解码器中的每个位置关注解码器中直到并包括该位置的所有位置。我们需要防止左向信息流在解码器中以保持自回归属性。我们在缩放点积注意力内通过掩蔽（设置为 $-\infty$）softmax 输入中对应于非法连接的所有值来实现这一点。

### 3.3 逐点前馈网络

除了注意力子层外，我们编码器和解码器中的每一层都包含一个全连接前馈网络，该网络分别且相同地应用于每个位置。这包括两个线性变换，中间有一个 ReLU 激活：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

虽然线性变换在不同位置之间是相同的，但它们逐层使用不同的参数。另一种描述方式是作为两个核大小为 1 的卷积。

输入和输出的维度是 $d_{\text{model}} = 512$，内层的维度是 $d_{ff} = 2048$。

### 3.4 嵌入和 Softmax

与其他序列转换模型类似，我们使用学习到的嵌入将输入和输出标记转换为维度 $d_{\text{model}}$ 的向量。我们还使用通常的学习线性变换和 softmax 函数将解码器输出转换为预测的下一个标记概率。在我们的模型中，我们在两个嵌入层和预 softmax 线性变换之间共享相同的权重矩阵，类似于 [24]。在嵌入层中，我们将这些权重乘以 $\sqrt{d_{\text{model}}}$。

### 3.5 位置编码

由于我们的模型不包含循环和卷积，为了让模型利用序列的顺序，我们必须注入关于序列中标记的相对或绝对位置的一些信息。为此，我们在编码器和解码器堆栈的底部向输入嵌入添加"**位置编码**"。位置编码与嵌入具有相同的维度 $d_{\text{model}}$，因此两者可以相加。位置编码有很多选择，包括学习和固定的 [8]。

在这项工作中，我们使用不同频率的正弦和余弦函数：

$$PE(pos, 2i) = \sin\left(pos/10000^{2i/d_{\text{model}}}\right)$$
$$PE(pos, 2i+1) = \cos\left(pos/10000^{2i/d_{\text{model}}}\right)$$

其中 $pos$ 是位置，$i$ 是维度。也就是说，位置编码的每个维度对应一个正弦曲线。波长形成从 $2\pi$ 到 $10000 \cdot 2\pi$ 的几何级数。我们选择这个函数是因为我们假设它可以让模型轻松学习按相对位置进行关注，因为对于任何固定偏移 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。

我们还尝试使用学习到的位置嵌入 [8]，发现两个版本产生几乎相同的结果（见表 3 行 (E)）。我们选择正弦版本，因为它可能允许模型外推到比训练期间遇到的更长的序列长度。

---

## 4 为什么自注意力

在本节中，我们将自注意力层的各个方面与通常用于将一个可变长度符号表示序列 $(x_1, ..., x_n)$ 映射到另一个等长序列 $(z_1, ..., z_n)$ 的循环层和卷积层进行比较。在激励我们使用自注意力时，我们考虑三个期望：

1. **每层的总计算复杂性**
2. **可以并行化的计算量**，以所需的最小顺序操作数量来衡量
3. **网络中长程依赖之间的路径长度**。学习长程依赖是许多序列转换任务中的关键挑战。一个关键因素是前向和后向信号在网络中必须遍历的路径长度。输入和输出序列中任意两个位置之间的这些路径越短，学习长程依赖就越容易 [11]。

**表 1：不同层类型的最大路径长度、每层复杂性和最小顺序操作数量**

| 层类型       | 每层复杂性               | 顺序操作 | 最大路径长度   |
| ------------ | ------------------------ | -------- | -------------- |
| 自注意力     | $O(n^2 \cdot d)$         | $O(1)$   | $O(1)$         |
| 循环         | $O(n \cdot d^2)$         | $O(n)$   | $O(n)$         |
| 卷积         | $O(k \cdot n \cdot d^2)$ | $O(1)$   | $O(\log_k(n))$ |
| 受限自注意力 | $O(r \cdot n \cdot d)$   | $O(1)$   | $O(n/r)$       |

其中 $n$ 是序列长度，$d$ 是表示维度，$k$ 是卷积的核大小，$r$ 是受限自注意力中的邻域大小。

如表 1 所示，自注意力层用恒定数量的顺序执行操作连接所有位置，而循环层需要 $O(n)$ 顺序操作。就计算复杂性而言，当序列长度 $n$ 小于表示维度 $d$ 时，自注意力层比循环层更快，这在机器翻译中最先进的模型使用的句子表示（如词片 [31] 和字节对 [25] 表示）中是最常见的情况。

为了提高涉及非常长序列任务的计算性能，自注意力可以限制为仅考虑输入序列中以各自输出位置为中心的大小为 $r$ 的邻域。这会将最大路径长度增加到 $O(n/r)$。我们计划在未来的工作中进一步研究这种方法。

单个核宽度 $k < n$ 的卷积层不连接所有输入和输出位置对。这样做需要 $O(n/k)$ 个卷积层堆栈（对于连续核），或者 $O(\log_k(n))$（对于膨胀卷积 [15]），增加了网络中任意两个位置之间最长路径的长度。卷积层通常比循环层更昂贵，相差 $k$ 倍。可分离卷积 [6] 大大降低了复杂性到 $O(k \cdot n \cdot d + n \cdot d^2)$。即使 $k = n$，可分离卷积的复杂性也等于自注意力层和逐点前馈层的组合，这是我们模型中采用的方法。

作为附带好处，自注意力可以产生更具可解释性的模型。我们检查模型中的注意力分布，并在附录中展示和讨论示例。单个注意力头不仅清楚地学习执行不同的任务，许多还表现出与句子的句法和语义结构相关的行为。

---

## 5 训练

### 5.1 训练数据和批处理

我们在标准 WMT 2014 英语 - 德语数据集上训练，该数据集由约 450 万个句子对组成。句子使用字节对编码 [3] 进行编码，具有约 37000 个标记的共享源 - 目标词汇表。对于英语 - 法语，我们使用了明显更大的 WMT 2014 英语 - 法语数据集，由 3600 万个句子组成，并将标记分割成 32000 个词片词汇表 [31]。句子对按近似序列长度批处理在一起。每个训练批包含一组句子对，包含约 25000 个源标记和 25000 个目标标记。

### 5.2 硬件和计划

我们在一台有 8 个 NVIDIA P100 GPU 的机器上训练模型。

- 对于基础模型：每个训练步骤大约需要 0.4 秒，训练 100,000 步或 12 小时
- 对于大型模型：步骤时间为 1.0 秒，训练 300,000 步（3.5 天）

### 5.3 优化器

我们使用 Adam 优化器 [17]，$\beta_1 = 0.9$，$\beta_2 = 0.98$，$\epsilon = 10^{-9}$。我们在训练过程中根据以下公式改变学习率：

$$\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})$$

这对应于在前 $warmup\_steps$ 训练步骤中线性增加学习率，此后与步数的平方根成反比降低。我们使用 $warmup\_steps = 4000$。

### 5.4 正则化

我们在训练期间采用三种类型的正则化：

**残差 Dropout** 我们将 dropout [27] 应用于每个子层的输出，在它添加到子层输入并归一化之前。此外，我们在编码器和解码器堆栈中的嵌入和位置编码之和应用 dropout。对于基础模型，我们使用 $P_{drop} = 0.1$ 的比率。

**标签平滑** 在训练期间，我们采用值为 $\epsilon_{ls} = 0.1$ 的标签平滑 [30]。这会损害困惑度，因为模型学会更加不确定，但会提高准确率和 BLEU 分数。

---

## 6 结果

### 6.1 机器翻译

**表 2：Transformer 在英语 - 德语和英语 - 法语 newstest2014 测试上取得了比先前最先进模型更好的 BLEU 分数，而训练成本只是一小部分**

| 模型                            | EN-DE BLEU | EN-FR BLEU | EN-DE 训练成本 (FLOPs)  | EN-FR 训练成本 (FLOPs) |
| ------------------------------- | ---------- | ---------- | ----------------------- | ---------------------- |
| ByteNet [15]                    | 23.75      | -          | -                       | -                      |
| Deep-Att + PosUnk [32]          | -          | 39.2       | -                       | $1.0 \cdot 10^{20}$    |
| GNMT + RL [31]                  | 24.6       | 39.92      | $2.3 \cdot 10^{19}$     | $1.4 \cdot 10^{20}$    |
| ConvS2S [8]                     | 25.16      | 40.46      | $9.6 \cdot 10^{18}$     | $1.5 \cdot 10^{20}$    |
| MoE [26]                        | 26.03      | 40.56      | $2.0 \cdot 10^{19}$     | $1.2 \cdot 10^{20}$    |
| Deep-Att + PosUnk Ensemble [32] | -          | 40.4       | -                       | $8.0 \cdot 10^{20}$    |
| GNMT + RL Ensemble [31]         | 26.30      | 41.16      | $1.8 \cdot 10^{20}$     | $1.1 \cdot 10^{21}$    |
| ConvS2S Ensemble [8]            | 26.36      | 41.29      | $7.7 \cdot 10^{19}$     | $1.2 \cdot 10^{21}$    |
| **Transformer (base model)**    | **27.3**   | **38.1**   | **$3.3 \cdot 10^{18}$** | -                      |
| **Transformer (big)**           | **28.4**   | **41.0**   | **$2.3 \cdot 10^{19}$** | -                      |

在 WMT 2014 英语 - 德语翻译任务上，大型 Transformer 模型比之前报告的最佳模型（包括集成）高出 2.0 BLEU 以上，建立了 28.4 的新最先进 BLEU 分数。训练在 8 个 P100 GPU 上花费了 3.5 天。即使是我们的基础模型也超过了所有先前发布的模型和集成，而训练成本仅为任何竞争模型的一小部分。

在 WMT 2014 英语 - 法语翻译任务上，我们的大型模型取得了 41.0 的 BLEU 分数，超过了所有先前发布的单一模型，而训练成本不到先前最先进模型的 1/4。用于英语 - 法语的 Transformer (big) 模型使用 dropout 率 $P_{drop} = 0.1$，而不是 0.3。

对于基础模型，我们使用单个模型，通过对最后 5 个检查点（以 10 分钟间隔写入）进行平均获得。对于大型模型，我们对最后 20 个检查点进行平均。我们使用束搜索，束大小为 4，长度惩罚 $\alpha = 0.6$。这些超参数是在开发集上实验后选择的。我们将推理期间的最大输出长度设置为输入长度 + 50，但尽可能提前终止。

### 6.2 模型变体

为了评估 Transformer 不同组件的重要性，我们以不同方式改变了基础模型，测量了开发集 newstest2013 上英语 - 德语翻译性能的变化。

**表 3：Transformer 架构的变体**

| 模型             | N    | d_model | d_ff | h    | d_k  | d_v  | P_drop | ε_ls | 训练步数 | PPL (dev) | BLEU (dev) | 参数 (×10^6) |
| ---------------- | ---- | ------- | ---- | ---- | ---- | ---- | ------ | ---- | -------- | --------- | ---------- | ------------ |
| base             | 6    | 512     | 2048 | 8    | 64   | 64   | 0.1    | 0.1  | 100K     | 4.92      | 25.8       | 65           |
| (A) 单头         | 6    | 512     | 512  | 1    | 512  | 512  | 0.1    | 0.1  | 100K     | 5.29      | 24.9       | 65           |
| (A) 4 头         | 6    | 512     | 2048 | 4    | 128  | 128  | 0.1    | 0.1  | 100K     | 5.00      | 25.5       | 65           |
| (A) 16 头        | 6    | 512     | 2048 | 16   | 32   | 32   | 0.1    | 0.1  | 100K     | 4.91      | 25.8       | 65           |
| (A) 32 头        | 6    | 512     | 2048 | 32   | 16   | 16   | 0.1    | 0.1  | 100K     | 5.01      | 25.4       | 65           |
| (B) h=16         | 6    | 512     | 2048 | 16   | 64   | 64   | 0.1    | 0.1  | 100K     | 5.16      | 25.1       | 58           |
| (B) h=32         | 6    | 512     | 2048 | 32   | 64   | 64   | 0.1    | 0.1  | 100K     | 5.01      | 25.4       | 60           |
| (C) N=2          | 2    | 512     | 2048 | 8    | 64   | 64   | 0.1    | 0.1  | 100K     | 6.11      | 23.7       | 36           |
| (C) N=4          | 4    | 512     | 2048 | 8    | 64   | 64   | 0.1    | 0.1  | 100K     | 5.19      | 25.3       | 50           |
| (C) N=8          | 8    | 512     | 2048 | 8    | 64   | 64   | 0.1    | 0.1  | 100K     | 4.88      | 25.5       | 80           |
| (D) 无 dropout   | 6    | 512     | 2048 | 8    | 64   | 64   | 0.0    | 0.1  | 100K     | 5.77      | 24.6       | 65           |
| (D) dropout=0.2  | 6    | 512     | 2048 | 8    | 64   | 64   | 0.2    | 0.1  | 100K     | 4.95      | 25.5       | 65           |
| (E) 学习位置嵌入 | 6    | 512     | 2048 | 8    | 64   | 64   | 0.1    | 0.1  | 100K     | 4.92      | 25.7       | 65           |
| big              | 6    | 1024    | 4096 | 16   | -    | -    | 0.3    | 0.1  | 300K     | 4.33      | 26.4       | 213          |

- 在行 (A)，我们改变注意力头的数量以及注意力键和值的维度，保持计算量恒定。虽然单头注意力比最佳设置差 0.9 BLEU，但头数过多时质量也会下降。
- 在行 (B)，我们观察到减小注意力键大小 $d_k$ 会损害模型质量。这表明确定兼容性并不容易，比点积更复杂的兼容性函数可能更有益。
- 在行 (C) 和 (D)，正如预期的那样，更大的模型更好，dropout 对于避免过拟合非常有帮助。
- 在行 (E)，我们将正弦位置编码替换为学习到的位置嵌入，观察到与基础模型几乎相同的结果。

---

## 7 结论

在这项工作中，我们提出了 **Transformer**，这是第一个完全基于注意力的序列转换模型，用多头自注意力替换了编码器 - 解码器架构中最常用的循环层。

对于翻译任务，Transformer 的训练速度明显快于基于循环或卷积层的架构。在 WMT 2014 英语 - 德语和 WMT 2014 英语 - 法语翻译任务上，我们都实现了新的最先进水平。在前一个任务中，我们的最佳模型甚至超过了所有先前报告的集成模型。

我们对基于注意力的模型的未来感到兴奋，并计划将它们应用于其他任务。我们计划将 Transformer 扩展到涉及文本以外输入和输出模态的问题，并研究局部受限注意力机制以有效处理大型输入和输出，如图像、音频和视频。减少生成的顺序性是我们的另一个研究目标。

我们用于训练和评估模型的代码可在 https://github.com/tensorflow/tensor2tensor 获取。

---

## 参考文献

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.

[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.

[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.

[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.

[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.

[7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

[8] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.

[9] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[11] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[12] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.

[13] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

[14] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.

[15] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.

[16] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.

[17] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

[18] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.

[19] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.

[20] Samy Bengio Łukasz Kaiser. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.

[21] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.

[22] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.

[23] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.

[24] Oﬁr Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.

[25] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

[26] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

[27] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929–1958, 2014.

[28] Sainbayar Sukhbaatar, arthur szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440–2448. Curran Associates, Inc., 2015.

[29] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.

[30] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.

[31] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

[32] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.

---

### 总结

> 本文提出了 **Transformer** 架构，这是自然语言处理（NLP）乃至整个人工智能领域的一项革命性突破。该篇文章直接催生了目前全球最大的开源 AI 社区——Hugging Face（抱抱脸）。现在所有开源的金融大模型（如基于 Llama、ChatGLM 微调的模型），其代码库和模型权重的开源，都是建立在这篇论文的理论基础之上的。文章的核心贡献与启示如下：
>
> 1. **架构颠覆**：完全摒弃了传统的循环神经网络（RNN）和卷积神经网络（CNN），首次提出仅通过**自注意力机制（Self-Attention）**和前馈神经网络来构建序列转换模型。
> 2. **核心机制**：
>    * **缩放点积注意力**：通过计算序列中各个词汇之间的关联度来捕捉全局依赖关系。
>    * **多头注意力（Multi-Head Attention）**：允许模型在不同的表示子空间中并行关注信息的不同维度。
>    * **位置编码（Positional Encoding）**：巧妙地弥补了丢弃 RNN 后丢失的序列位置信息。
> 3. **性能飞跃**：不仅在翻译准确度（BLEU分数）上达到了当时的最佳水平（SOTA），更重要的是，Transformer 极大地提升了模型训练的**并行计算能力**，显著缩短了训练时间。
> 4. **商业与数据分析视角**：Transformer 架构解决了长文本序列处理中的长程依赖难题和计算效率瓶颈，为后来处理海量非结构化文本数据（如公告、研报）的大语言模型（如 GPT、BERT）奠定了绝对的底层技术基石。
