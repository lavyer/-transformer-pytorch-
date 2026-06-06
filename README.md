# 基于Transformer的中英-机器翻译的自然语言处理课程设计
机器翻译是自然语言处理领域的核心任务之一，旨在利用计算机自动将一种自然语言转换为另一种自然语言。随着全球化进程的加速和互联网信息的爆炸式增长，机器翻译在跨国交流、信息获取、跨境电商等领域具有重要的应用价值。
传统的机器翻译方法包括基于规则的翻译、统计机器翻译（SMT）等，但这些方法往往依赖大量人工特征工程或复杂的对齐模型。2017年，Google提出的Transformer模型完全基于自注意力机制（Self-Attention），摒弃了循环神经网络（RNN）和卷积神经网络（CNN）的序列结构，实现了并行计算和长距离依赖的有效捕捉，在机器翻译任务上取得了突破性进展。Transformer已成为当前主流机器翻译系统（如Google Translate、DeepL）的核心架构。
在约3万对中英日常会话数据上训练的8.8M参数Transformer模型，经过15个epoch训练后，BLEU达到0.152（Beam Search）。模型对**简单句**（如"他是一名医生"、"今天天气真好"）的翻译准确率很高，能够正确捕捉主谓宾结构和常用表达。
模型采用《Attention Is All You Need》论文的经典Transformer架构：

**编码器（Encoder）**：N层EncoderBlock堆叠

- 多头自注意力子层（Multi-Head Self-Attention）
- 前馈神经网络子层（FFN：d_model → d_ff → d_model，ReLU激活）
- 每个子层后：残差连接 + 层归一化

**解码器（Decoder）**：N层DecoderBlock堆叠

- 掩码多头自注意力子层（Masked Multi-Head Self-Attention）
- 编码器-解码器交叉注意力子层（Cross-Attention）
- 前馈神经网络子层

**位置编码**：正弦位置编码（Sinusoidal Positional Encoding）
**2. Greedy vs Beam Search**：
Beam Search在所有指标上均优于Greedy Decode，尤其在复杂句子上差异明显。例如"汤姆决定不去参加玛丽的派对"：Greedy译为"Tom had to decide whether..."（语义偏差），Beam正确译为"Tom decided not to go to Mary's party"。

**3. 中文字符级分词的适用性**：

- 优势：中文词表仅3,080个字符，模型参数少，且避免了分词错误
- 劣势：丢失了词语级语义信息，字符级编码的序列更长

**4. 局限性与改进方向**：

- 训练数据规模有限（29,909对），模型对复杂长句和罕见词翻译效果不佳
- CPU训练限制了模型规模和训练轮数
- 改进方向：使用更大规模语料（WMT/OPUS）、BPE子词分词、GPU训练、预训练模型初始化
