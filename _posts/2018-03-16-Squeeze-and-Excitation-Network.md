---
author: linhu
title: "Squeeze-and-Excitation Network"
comments: true
classes: wide
date: 2018-03-15 17:42:32+00:00
categories:
  - backbone network
tags:
  - Deep Learning
  - Computer Vision
  - backbone network
  - CVPR 2017
---

>[论文链接](https://arxiv.org/abs/1709.01507) 
>[代码地址](https://github.com/hujie-frank/SENet)

# 摘要

&emsp;&emsp;卷积神经网络建立在卷积运算的基础上，通过融合本地感受域内的空间信息和信道信息来提取信息特征。为了提高网络的表现力，许多现有的工作已经显示出增强空间编码的好处。在这项工作中，我们专注于频道并提出一种新颖的架构单元，我们称之为“挤压与激励”（SE）模块，通过显式建模通道之间的相关性来自适应重新校准通道特性响应。我们证明，通过将这些块堆叠在一起，我们可以构建SENet体系结构，在具有挑战性的数据集中进行非常好的泛化。至关重要的是，我们发现SE块以微小的计算成本为现有最先进的深架构产生显着的性能改进。 SENET组成了我们ILSVRC 2017分类提交的基础，该分数提交获得了第一名，并将前5名的误差显着降低至2.251％，相对于2016年的获胜条目获得了25％的相对改善。
&emsp;&emsp;卷积神经网络建立在卷积操作的基础上，使用局部感受区域（local receptive field）融合空间信息和通道信息来提取包含信息的特征。有很多工作从增强空间编码（spatial encoding）的角度来提升网络的表示能力并取得了成果，而本文主要聚焦于通道维度，并提出一种新的结构单元——“ Squeeze-and-Excitation(SE) “单元，通过显式建模通道之间的相关性，可以自适应的调整各通道的特征响应值。我们论证了通过将这些单元堆叠在一起，我们可以构建SENet体系结构，并在具有挑战性的数据集上达到非常好的泛化。更重要的是，如果将SE单元添加到已有的state-of-the-art深度神经网络中，只会增加很小的计算开销但却可以显著地提升性能。依靠SENet作者获得了ILSVRC2017分类任务的第一名，并将top-5错误率显着降低至2.251%，与2016年的获胜者相比达到了~25％的相对提升。

# 1. 介绍

&emsp;&emsp;在卷积神经网络中，每个卷积层有若干滤波器，学习表达所有输入通道的局部空间连接模式。也就是说，卷积滤波器用于提取局部感受区域中的空间和通道维度的融合信息。通过交错叠加一系列卷积层以及非线性激活层和降采样层，CNN可以获得具有全局感受区域的分层模式来作为有力的图像描述。最近的一些工作表明，可以通过加入有助于获取空间相关性的学习机制来改善网络的性能，而且不需要额外的监督。例如Inception架构[^14,39]，通过在模块中加入多尺度处理来取得较高的精度。另有一些工作试图建模更好的空间相关性[^1,27]或者添加空间注意力机制（spatial attention）[^17]。
&emsp;&emsp;与上述方法不同，本文主要探索网络架构设计的另一个方面——通道关联性。本文提出一种新的网络单元——“ Squeeze-and-Excitation(SE) ” 单元，希望通过对卷积特征各通道的依赖性进行明确地建模来提高网络的表达能力，为此我们提出一种机制使得网络可以对特征进行微校（recalibration），这样网络就可以学习使用全局信息来有选择性地加强包含有用信息的特征并抑制无用特征。
&emsp;&emsp;SE单元基本结构如图1。对于任意给定的变换$$\mathrm{F_{tr}:X\rightarrow U,X\in\mathbb{R}^{{W ^\prime}\times{H ^\prime}\times{C^\prime}},U\in\mathbb{R}^{W\times H\times C}}$$，（即一个或一系列卷积），我们可以构建一个相应的SE单元来执行如下特征微校。特征$$U$$首先通过一个*squeeze*操作，将空间 $$W\times H$$ 范围内的特征图聚合起来形成一个通道描述符，这个描述符嵌入了通道维度特征响应的全局分布，使来自网络全局接受域的信息能够被其较低层所利用。接下来是*excitation*操作，通过一个基于通道置信度的self-gating机制为每个通道学习针对特定样本的激活（sample-specific activations），来决定每个通道的激励， 特征图 $$U$$ 接着被重新加权以产生SE块的输出，然后可以将其直接提供给后续的层。

![1520924591103]({{ site.url }}{{ site.baseurl }}/assets/images/1520924591103.png)

&emsp;&emsp;SE网络可以通过简单地堆叠一系列SE构件单元来生成。 SE单元也可以直接用来替换网络结构任何深度的drop-in单元。 然而，虽然构建单元的模板是通用的（如6.3节所示），它在不同深度扮演的角色需要适应网络的需求。 在较早的层中，它以类别无关的方式（class agnostic manner）学习提取富含信息的特征，加强共享低层表示的质量。 在后面的层中，SE块越来越特化，以高度类别特定的方式（class-specific manner）响应不同的输入。 因此，通过SE块进行特征微校产生的裨益可以通过整个网络进行累积。
&emsp;&emsp;新的CNN架构的开发是一项具有挑战性的工程任务，通常涉及许多新的超参数和层配置的选择。 相比之下，上面概述的SE单元的设计则很简单，并且可以直接与现有state-of-the-art的架构一起使用，这些架构的卷积层可以通过直接替换其对应SE副本来得到增强。 而且如第4节所示，SE单元在计算上是轻量级的，在模型复杂性和计算负担上仅略微增加。 为了支持这个论断，我们开发了几个SENets，分别是SE-ResNet，SE-Inception，SE-ResNeXt和SE-Inception-ResNet，并在ImageNet 2012数据集[^30]上对SENets进行了广泛的评估。 此外，为了证明SE单元的泛化能力，本文还提供了在其他数据集上的结果，表明所提出的方法不限定于特定数据集或任务。
&emsp;&emsp;使用SENets，本文作者赢得了ILSVRC 2017分类竞赛的第一名，在测试集上达到了2.251％的top-5 error。 与前一年的获胜者相比，这比上一年的冠军有25％的相对提升（其top-5 error为2.991％）。 

# 2. 相关工作

**Deep architectures**
&emsp;&emsp;有很多工作通过调整卷积神经网络架构使模型更容易地学习深层特征以提升模型性能。VGGNets[^35]和Inception模型[^39]证明了可以通过增加深度来提升性能，显著超越了之前ILSVRC 2014上的方法。Batch normalization (BN)[^14]在网络中添加可以正则化每一层输入数据的单元来稳定学习过程，从而改善梯度在网络中的传播，使得更深层的网络也可以工作。[ResNet](https://github.com/binLearning/caffe_toolkit/tree/master/ResNet)、[ResNet-v2](https://github.com/binLearning/caffe_toolkit/tree/master/ResNet-v2)在网络中加入恒等映射形式的跳跃连接（identity-based skip connections）使网络学习残差函数，有利于信息在单元间流动，极大推进了网络架构向更深层的发展。[DenseNet](https://github.com/binLearning/caffe_toolkit/tree/master/DenseNet)、[DPN](https://github.com/binLearning/caffe_toolkit/tree/master/DPN)通过调整网络各层间的连接机制来提升深层网络的学习和表示性能。 
&emsp;&emsp;另一个方向是调整网络中模块组件的功能形式。分组卷积（grouped convolutions）可以用于增加基数（cardinality，变换集合的大小）[^13,43]，如Deep roots、[ResNeXt](https://github.com/binLearning/caffe_toolkit/tree/master/ResNeXt)中所示，网络可以学习到更丰富的表示。多分支卷积（multi-branch convolutions）可以视为分组卷积的泛化，网络模块可以进行更灵活的卷积操作组合，如Inception系列。跨通道相关是一种新的特征组合方式，可以独立于空间结构[^6,18]（如Xception），也可以联合使用1x1标准卷积滤波器[^22]进行处理（如NIN），一般来说这些工作主要是为了降低模型和计算复杂度。这种方法的前提假设是通道关系在局部接受域可以被表述为实例无关（instance-agnostic）的函数组合，也就是说输出对于输入数据各通道的依赖性是相同的，不是类别相关的。与之相反，本文提出一种新的机制，使用全局信息明确地建模各通道之间的动态非线性依赖关系，从而改善学习过程并提升网络的表示能力。
**Attention and gating mechanisms**
&emsp;&emsp;注意力机制（attention）引导可用计算资源的分配偏向于输入信号中信息量最大的部分，这些机制的研究和发展一直是神经科学界长期以来的研究领域[^15,16,28]，近几年开始大量用于深度神经网络中，在很多任务中对性能有极大提升，从图像的定位和理解[^3,17]到基于序列的模型[^2,24]。它一般是和门限函数（如softmax、sigmoid）或者序列方法联合使用[^11, 37]。最近的研究表明，它适用于图像标题[^4,44]和唇读[^7]等任务，利用它来有效地聚合多模态数据。在这些应用中，它一般在表示较高层抽象语义的最高一层或多层用于模式匹配。
&emsp;&emsp;highway网络使用门限机制来正则化快捷（shortcut）连接，使得可以训练非常深的网络。 Wang等人在《Residual attention network for image classification》[^42]中介绍了一种有效的trunk-and-mask注意力机制并使用了沙漏模块（hourglass module）[^27]，它成功的运用于语义分割任务。这个高容量单元被插入到深度残差网络的中间阶段之间。 相比之下，本文提出的SE单元是一种轻量级的门限机制，专门用于对通道维度的关系进行建模，并用于增强整个网络中模块的表示能力。

# 3. Squeeze-and-Excitation Blocks

&emsp;&emsp;Squeeze-and-Excitation block是一个计算单元，可构建在任何给定的变换上$$\mathrm{F_{tr}:X\rightarrow U,X\in\mathbb{R}^{{W ^\prime}\times{H ^\prime}\times{C^\prime}},U\in\mathbb{R}^{W\times H\times C}}$$。为了简化说明，在下面的符号中，我们将$$\mathrm{F_{tr}}$$作为标准的卷积运算符。令 $$\mathrm{V=[v_1,v_2,\cdots,v_c]}$$ 表示学习到的滤波器卷积核集合，其中$$\mathrm{v_c}$$ 指第c个滤波器的参数。我们接着可以将$$\mathrm{F_{tr}}$$ 的输出写作$$\mathrm{U=[u_1,u_2,\cdots,u_c]}$$，那么：

$$\mathrm{u_c=v_c{*}X=\sum^{C^\prime}_{s=1}v^s_c {*} x^s}.\tag{1}$$ 

&emsp;&emsp;此处 $$^{*}$$ 表示卷积， $$\mathrm{v_c=[v_c^1,v_c^2,\cdots,v_c^{C^\prime}]}$$ 以及 $$\mathrm{X=[x^1,x^2,\cdots,x^{C^\prime}]}$$ (为了简化符号，省略了bias参数)。此处$$\mathrm{v_c^s}$$ 是一个2D空间卷积核，表示作用于 $$\mathrm{X}$$ 相应通道的 $$\mathrm{v_c}$$ 的一个通道。由于输出是所有通道的总和，因此通道依赖关系隐式嵌入到了 $$\mathrm{v_c}$$ 中，但这些依赖性与滤波器捕获的空间相关性纠缠在一起。我们的目标是使网络能够提高对富含信息的特征的敏感度，以便后续转换利用它们，并抑制不太有用的特征。 我们建议在进入下一个转换之前通过两个步骤（即squeeze和excitation）来明确建模通道相互依赖关系，从而重新校准滤波器响应。SE构件单元的图示如图1。

## 3.1 Squeeze: Global Information Embedding

&emsp;&emsp;&emsp;&emsp;为了利用通道依赖关系，我们首先考虑输出特征中每个通道的信号。 每一个所学的过滤器操作一个局部接受域，因此每个转换单元输出的U都不能利用该区域之外的上下文信息。 这个问题在接受域比较小的网络较低层中变得更加严重。

&emsp;&emsp;为了缓解这个问题，我们建议将全局空间信息压缩成一个通道描述符，通过使用全局平均池化（global average pooling）生成通道统计数据来实现。形式上，统计量 $$\mathrm z\in \mathbb R^C$$是通过压缩$$\mathrm U$$通过空间维度 $$\mathrm{W\times H}$$ 生成的，其中 $$\mathrm z$$ 的第c个元素由下式计算：

$$z_c=\mathrm{F_{sq}(u_c)=\frac{1}{W\times H}\sum^W_{i=1}\sum^H_{j=1}}u_c(i,j).\tag{2}$$

> *Discussion.*
>
> 转换的输出 $$U$$ 可以被解释为局部描述符的集合，其统计数据表示整个图像，利用这些信息在特征工程的工作中很普遍[^31,34,45]。 我们选择了最简单的全局平均池化，这里也可以采用更复杂的聚合策略。

## 3.2. Excitation: Adaptive Recalibration

&emsp;&emsp;为了利用squeeze操作中聚合的信息，我们紧接着进行第二个操作，该操作旨在完全提取通道相关性。为此，这个函数必须符合两个标准：

- 首先，它必须是灵活的（特别是它必须能够学习通道之间的非线性相互作用）;
- 其次，它必须学习一个非互斥的关系，因为多通道 可以强调与单次激活相反。 为了符合这些标准，我们选择使用一个简单的门控机制和一个sigmoid激活：

$$\mathbf{s} = \mathbf{F}_{ex}(\mathbf{z}, \mathbf{W}) = \sigma(g(\mathbf{z}, \mathbf{W})) = \sigma(\mathbf{W}_2\delta(\mathbf{W}_1\mathbf{z}))\tag{3}$$

&emsp;&emsp;其中$$\delta$$是指ReLU[^26]函数，$$\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$$ 和 $$\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$$。为了限制模型复杂度和辅助泛化，我们通过围绕在非线性ReLU前后的两个全连接（FC）层形成的瓶颈来参数化门机制，即降维层参数为$$\mathbf{W}_1$$，降维比例为 $$r$$（我们把它设置为16，这个参数选择在6.3节中讨论），一个ReLU，然后是一个参数为$$\mathbf{W}_2$$的升维层。块的最终输出通过使用上述激活值来rescaling之前的变换输出$$\mathbf{U}$$得到：
$$\widetilde{\mathbf{x}}_c = \mathbf{F}_{scale}(\mathbf{u}_c, s_c) = s_c \cdot \mathbf{u}_c\tag{4}$$
其中$$\widetilde{\mathbf{X}} = [\widetilde{\mathbf{x}}_1, \widetilde{\mathbf{x}}_2, \dots, \widetilde{\mathbf{x}}_{C}]$$，$$\mathbf{F}_{scale}(\mathbf{u}_c, s_c)$$指特征图 $$\mathbf{u}_c \in \mathbb{R}^{W \times H}$$ 和标量$$s_c$$之间的对应通道乘积。

> *Discussion.*
>
> 激活在此处作为适应特定输入描述符 $$z$$ 的通道权重。在这方面，SE单元本质上引入了以输入为条件的动态特性，有助于提高特征的辨别力。

## 3.3. Exemplars: SE-Inception and SE-ResNet

&emsp;&emsp;SE块的灵活性意味着它可以直接应用于标准卷积之外的变换。为了说明这一点，我们通过将SE块集成到两个流行的网络架构系列，Inception和ResNet中来开发SENets。通过将变换$$\mathbf{F}_{tr}$$看作一个整体的Inception模块（参见图2），为Inception网络构建SE单元。通过对架构中的每个模块进行更改，我们构建了一个SE-Inception网络。

![52112481026]({{ site.url }}{{ site.baseurl }}/assets/images/1521124810265.png)

&emsp;&emsp;残差网络及其变种在学习深度表示方面非常有效。我们开发了一系列的SE单元，分别与ResNet[^9]，ResNeXt[^43]和Inception-ResNet[^38]集成。图3描述了SE-ResNet模块的架构。在这里，SE单元变换 $$\mathbf{F}_{tr}$$ 被认为是残差模块的非恒等分支。压缩和激励都在与恒等分支相加之前起作用。

![52112487610]({{ site.url }}{{ site.baseurl }}/assets/images/1521124876101.png)

# 4. Model and Computational Complexity

&emsp;&emsp;SENet通过堆叠一组SE单元来构建。实际上，它是通过用原始块对应的SE部分（即SE残差块）替换每个原始块（即残差块）而产生的。我们在表1中描述了SE-ResNet-50和SE-ResNeXt-50的架构。

![52112498314]({{ site.url }}{{ site.baseurl }}/assets/images/1521124983140.png)

&emsp;&emsp;在实践中SE单元的模型复杂度和计算开销必须在可接受范围内才是实用可行的，这对于可伸缩性是重要的。为了说明模块的开销，我们比较了ResNet-50和SE-ResNet-50来作为例子，其中SE-ResNet-50的精确度明显优于ResNet-50，接近网络层数更深的ResNet-101网络（如表2所示）。对于$$224\times 224$$像素的输入图像，ResNet-50单次前向传播需要$$\sim$$3.86 GFLOP。每个SE单元需要使用一个压缩阶段的全局平均池化操作和两个激励阶段中的小全连接层，以及接下来的轻量的通道缩放操作。总的来说，SE-ResNet-50需要$$\sim$$3.87 GFLOP，相对于原始的ResNet-50只相对增加了0.26%的开销。

![52112504608]({{ site.url }}{{ site.baseurl }}/assets/images/1521125046086.png)

&emsp;&emsp;在实验中，训练的批数据大小为256张图像，ResNet-50的一次前向传播和反向传播花费190ms，而SE-ResNet-50则花费209ms（两个时间都是在具有88个NVIDIA Titan X GPU的服务器上执行得到的）。我们认为这是一个合理的开销，因为在现有的GPU库中，全局池化和小型内积操作的优化程度较低。此外，考虑到嵌入式设备使用CPU计算，我们还对每个模型的CPU推断时间进行了基准测试：对于$$224\times 224$$像素的输入图像，ResNet-50花费了164ms，相比之下，SE-ResNet-50花费了167ms。SE单元所需的少量的额外计算开销相对于其对模型性能的提升来说是合理的（在第6节中详细讨论）。

&emsp;&emsp;接下来，我们考虑所提出的单元引入的附加参数。所有附加参数都包含在门机制的两个全连接层中，只构成网络总容量的一小部分。更确切地说，引入的附加参数的数量由下式给出：

$$\frac{2}{r} \sum_{s=1}^S N_s \cdot {C_s}^2\tag{5}$$

&emsp;&emsp;其中$$r$$表示压缩率（我们在所有的实验中将 $$r$$ 设置为16），$$S$$ 指的是阶段数量（每个阶段是指在共同空间维度的特征图上运行的块的集合），$$C_s$$表示阶段$$s$$的输出通道的维度，$$N_s$$表示重复的块编号。总的来说，SE-ResNet-50在ResNet-50所要求的$$\sim$$2500万参数之外引入了$$\sim$$250万附加参数，相对增加了$$\sim$$10%的参数总数量。这些附加参数中的大部分来自于网络的最后阶段，在这个阶段激励（excitation）所操作的通道维度最大。SE单元在最终阶段的开销相对较大，然而我们发现其可以在性能的边际成本上被移除（即ImageNet数据集上top-1错误率$$<0.1\%$$ ），将相对参数增加减少到$$\sim$$4%，这在参数量是重要考虑的情况下可能是有用的。

# 5. Implementation

&emsp;&emsp;在训练过程中，我们遵循标准的做法，使用随机尺度裁剪[^39]到$$224\times 224$$像素（在Inception-ResNet-v2[^38]和SE-Inception-ResNet-v2上为$$299\times 299$$），并用随机水平翻转来进行数据增强。输入图像通过减去通道均值来进行归一化。另外，我们采用[^32]中描述的数据均衡策略进行mini-batch采样，以弥补类别分布不均匀。网络在我们的分布式学习系统“ROCS”上进行训练，其能够进行大型网络的高效并行训练。我们使用synchronous SGD进行优化，动量为0.9，mini-batch的大小为1024（在每个GPU上分为32张图像的 sub-batch，共4个服务器，每个服务器包含8个GPU）。初始学习率设为0.6，每30个迭代周期减少10倍。使用[^8]中描述的权重初始化策略，所有模型都从零开始训练100个迭代周期。

# 6. Experiments

&emsp;&emsp;在这节我们在ImageNet 2012数据集上进行了大量的实验[^30]，其目的是：首先探索所提出的SE单元对不同深度的基础网络的影响；其次，调查它与state-of-the-art的网络架构集成后的能力，旨在公平比较SENets和非SENets，而不是追求性能提升。接下来，我们将介绍提交到ILSVRC 2017分类任务的模型的结果以及详细信息。此外，我们在Places365-Challenge场景分类数据集[^48]上进行了实验，以研究SENets是否能够很好地泛化到其它数据集。最后，我们研究了激励的作用，并根据实验现象给出了一些分析。

## 6.1. ImageNet Classification

&emsp;&emsp;ImageNet 2012数据集包含来自1000个类别的128万张训练图像和5万张验证图像。我们在训练集上训练网络，并在验证集上使用中心裁剪图像评估来报告`top-1`和`top-5`错误率，其中每张图像短边首先归一化为256，然后从每张图像中裁剪出$$224\times 224$$个像素，（对于Inception-ResNet-v2和SE-Inception-ResNet-v2，每幅图像的短边首先归一化到352，然后裁剪出$$299\times 299$$个像素）。

**网络深度。**我们首先将SE-ResNet与一系列标准ResNet架构进行比较。每个ResNet及其相应的SE-ResNet都使用相同的优化方案进行训练。验证集上不同网络的性能如表2所示，表明SE块在不同深度上的网络上始终能提高性能，而计算复杂度却增加的很少。

&emsp;&emsp;值得注意的是，SE-ResNet-50实现了单裁剪图像6.62%的`top-5`验证错误率，超过了ResNet-50（7.48%）0.86%，接近更深的ResNet-101网络（6.52%的`top-5`错误率），但只有ResNet-101一半的计算开销（3.87 GFLOPs vs. 7.58 GFLOPs）。这种规律在更大的深度上重复，SE-ResNet-101（6.07%的`top-5`错误率）不仅达到而且超过了更深的ResNet-152网络（6.34%的`top-5`错误率）。图4分别描绘了SE-ResNets和ResNets的训练和验证曲线。虽然注意到SE单元本身增加了深度，但是它们的计算效率极高，即使在扩展基础架构的深度达到收益递减的点上也能产生良好的回报。而且，我们看到通过对各种不同深度的训练，性能改进是一致的，这表明SE单元引起的改进可以与增加基础架构深度结合使用。

![52112508662]({{ site.url }}{{ site.baseurl }}/assets/images/1521125086621.png)

**与现代架构集成。**接下来我们将研究SE单元与另外两种state-of-the-art的架构Inception-ResNet-v2[^38]和ResNeXt[^43]的结合效果。Inception架构将卷积模块构造为分解滤波器的多分支组合，体现了Inception假设[^6]可以独立映射空间相关性和跨通道相关性。相比之下，ResNeXt架构宣称可以通过聚合稀疏连接（在通道维度中）卷积特征的组合来获得更丰富的表示。两种方法都在模块中引入了先前结构化（prior-structured）的相关性。我们构造了这些网络的SENet版本，SE-Inception-ResNet-v2和SE-ResNeXt（表1给出了SE-ResNeXt-50（$$32\times4d$$）的配置）。像前面的实验一样，原始网络和它们对应的SENet网络都使用相同的优化方案。

&emsp;&emsp;表2中给出的结果说明在将SE块引入到两种架构中会引起显著的性能改善。尤其是SE-ResNeXt-50的`top-5`错误率是5.49%，优于它对应的ResNeXt-50（5.90%的`top-5`错误率）以及更深的ResNeXt-101（5.57%的`top-5`错误率），而ResNeXt-101模型几乎有两倍的参数和计算开销。对于Inception-ResNet-v2的实验，我们猜测可能是裁剪策略的差异导致了其报告结果与我们重新实现的结果之间的差距，因为它们的原始图像大小尚未在[^38]中澄清，所以我们从相对较大的图像（其中较短边被归一化为352）中裁剪出$$299\times 299$$大小的区域。SE-Inception-ResNet-v2（4.79%的`top-5`错误率）比我们重新实现的Inception-ResNet-v2（5.21%的`top-5`错误率）提升了0.42%（相对改进了8.1%），也优于[^38]中报告的结果。每个网络的优化曲线如图5所示，说明了在整个训练过程中SE单元产生了始终如一的提升。

![52112514647]({{ site.url }}{{ site.baseurl }}/assets/images/1521125146478.png)

&emsp;&emsp;最后，我们通过对BN-Inception架构[^14]进行实验来评估SE单元在非残差网络上的效果，该架构在较低的模型复杂度下提供了良好的性能。比较结果如表2所示，训练曲线如图6所示，表现出的现象与残差网络架构中出现的现象一样。尤其是与BN-Inception 7.89%的错误率相比，SE-BN-Inception获得了更低7.14%的`top-5`错误。这些实验表明SE单元引起的改进可以与多种架构结合使用。并且，这个结论对于残差网络和非残差网络都适用。

![52112517264]({{ site.url }}{{ site.baseurl }}/assets/images/1521125172642.png)

**ILSVRC 2017分类竞赛的结果。**ILSVRC[^30]是一个年度计算机视觉竞赛，被证明是图像分类模型发展的沃土。ILSVRC 2017分类任务的训练和验证数据来自ImageNet 2012数据集，而测试集包含额外的未标记的10万张图像。为了竞争的目的，使用`top-5`错误率度量来对输入条目进行排序。

&emsp;&emsp;SENets是我们在挑战中赢得第一名的基础。我们的获胜模型由一小群SENets集成（ensemble），它们采用了标准的多尺度和多裁剪图像融合策略，在测试集上获得了2.251%的`top-5`错误率。这个结果在2016年获胜者（2.99%的`top-5`错误率）的基础上相对改进了$$\sim$$25%。我们的高性能网络之一是将SE单元与修改后的ResNeXt[43]集成在一起构建的（附录A提供了这些修改的细节）。在表3中我们将所提出的架构与state-of-the-art的模型在ImageNet验证集上进行了比较。我们的模型在每一张图像使用$$224\times 224$$中间裁剪评估（短边首先归一化到256）取得了18.68%的`top-1`错误率和4.47%的`top-5`错误率。为了与以前的模型进行公平的比较，我们也提供了$$320\times320$$的中心裁剪图像评估，在`top-1`(17.28%)和`top-5`(3.79%)的错误率度量中也获得了最低的错误率。

![52112522679]({{ site.url }}{{ site.baseurl }}/assets/images/1521125226797.png)

## 6.2. 场景分类

&emsp;&emsp;ImageNet数据集的大部分由单个对象支配的图像组成。为了在更多不同的场景下评估我们提出的模型，我们还在Places365-Challenge数据集[48]上对场景分类进行评估。该数据集包含800万张训练图像和365个类别的36500张验证图像。相对于分类，场景理解的任务可以更好地评估模型泛化和处理抽象的能力，因为它需要捕获更复杂的数据关联以及对更大程度外观变化的鲁棒性。

&emsp;&emsp;我们使用ResNet-152作为强大的基线来评估SE单元的有效性，并遵循[^33]中的评估准则。表4显示了针对给定任务训练ResNet-152模型和SE-ResNet-152的结果。具体而言，SE-ResNet-152（11.01%的`top-5`错误率）取得了比ResNet-152（11.61%的`top-5`错误率）更低的验证错误率，证明了SE单元可以在不同的数据集上表现良好。这个SENet也超过了先前的state-of-the-art的模型Places-365-CNN [^33]，它在这个任务上有11.48%的`top-5`错误率。

![52112526109]({{ site.url }}{{ site.baseurl }}/assets/images/1521125261091.png)

## 6.3. Analysis and Discussion

**压缩率。**公式（5）中引入的压缩率 $$r$$ 是一个重要的超参数，它允许我们改变模型中SE块的容量和计算成本。为了研究这种关系，我们基于SE-ResNet-50架构进行了一系列不同 $$r$$ 值的实验。表5中的比较表明，性能并没有随着容量的增加而单调上升。这可能是因为SE单元会过拟合训练集的通道相关性。我们发现设置 $$r=16$$ 时在精度和复杂度之间取得了很好的平衡，因此我们将这个值用于所有的实验。

![52112529369]({{ site.url }}{{ site.baseurl }}/assets/images/1521125293694.png)

**激励的作用。**虽然SE单元从经验上显示出其可以改善网络性能，但我们也想了解自门激励机制（self-gating excitation mechanism）在实践中是如何运作的。为了更清楚地描述SE块的行为，本节我们研究SE-ResNet-50模型的样本激活，并考察它们在不同块不同类别下的分布情况。具体而言，我们从ImageNet数据集中抽取了四个类，这些类具有语义和外观多样性，即金鱼，哈巴狗，刨和悬崖（图7中显示了这些类别的示例图像）。然后，我们从验证集中为每个类抽取50个样本，并计算每个阶段最后的SE块中50个均匀采样的通道的平均激活值（紧接在下采样之前），并在图8中绘制它们的分布。作为参考，我们也绘制所有1000个类的平均激活分布。

![52112532204]({{ site.url }}{{ site.baseurl }}/assets/images/1521125322041.png)

&emsp;&emsp;我们对SENets中Excitation的作用提出以下三点看法。**首先，不同类别的分布在较低层中几乎相同**，例如，SE_2_3。这表明在网络的最初阶段特征通道的权重很可能由不同的类别共享。然而有趣的是，第二个观察结果是**在更深的层，每个通道的权重变得更类别相关，因为不同类别对特征的判别性值具有不同的偏好，** 如SE_4_6和SE_5_1。这两个观察结果与以前的研究结果一致[^21,46]，即低层特征通常更普遍（即分类中类别不可知），而高层特征具有更高的特异性。因此，特征表示学习从SE块引起的重新校准中受益，其自适应地促进特征提取和特化（specialisation）到所需要的程度。最后，我们在网络的最后阶段观察到一个有些不同的现象。SE_5_2呈现出朝向饱和状态的有趣趋势，其中大部分激活接近于1，其余激活接近于0。在所有激活值取1的点处，该块将成为标准残差块。在网络的末端SE_5_3中（紧接着是在分类器之前的全局池化），类似的模式出现在不同的类别上，尺度上只有轻微的变化（可以通过分类器来调整）。这表明，SE_5_2和SE_5_3在为网络提供重新校准方面比前面的块更不重要。这一发现与第四节实证研究的结果是一致的，这表明，通过删除最后一个阶段的SE块，总体参数数量可以显著减少，而性能只有一点损失（<0.1%的`top-1`错误率）。

![52112537872]({{ site.url }}{{ site.baseurl }}/assets/images/1521125378724.png)
![52112542795]({{ site.url }}{{ site.baseurl }}/assets/images/1521125427954.png)


# 7. 结论

&emsp;&emsp;在本文中，我们提出了SE块，这是一种新颖的架构单元，旨在通过使网络能够执行动态通道特征重新校准来提高网络的表示能力。大量实验证明了SENets的有效性，其在多个数据集上取得了state-of-the-art的性能。此外，它们也让我们认识到一些以前的架构在建模通道特征依赖性上的局限性，我们任务可能SENets对其它需要强判别性特征的任务是有用的。最后，由SE块引入的特征权重可能有助于一些相关领域，如网络修剪压缩。

# A. ILSVRC 2017分类竞赛输入细节

&emsp;&emsp;表3中的SENet是通过将SE块集成到 $$64\times4d$$ 的ResNeXt-152的修改版本中构建的，通过遵循ResNet-152[^9]的块堆叠来扩展原始ResNeXt-101[^43]。更多设计和训练差异（除了SE块的使用之外）如下：

（a）对于每个瓶颈构建块，首先 $$1\times1$$ 卷积通道的数量减半，以性能下降最小的方式降低网络的计算成本。

（b）第一个 $$7\times7$$ 卷积层被三个连续的 $$3\times3$$ 卷积层所取代。

（c）步长为2的 $$1\times1$$ 卷积的下采样投影被替换步长为2的 $$3\times3$$ 卷积以保留信息。

（d）在分类器层之前插入一个dropout层（丢弃比为0.2）以防止过拟合。

（e）训练期间使用标签平滑正则化（如[^40]中所介绍的）。

（f）在最后几个训练迭代周期，所有BN层的参数都被冻结，以确保训练和测试之间的一致性。

（g）使用8个服务器（64个GPU）并行训练，以实现大batch大小（2048），初始学习率为1.0。



# References

[^1]: S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR, 2016.

[^2]: T. Bluche. Joint line segmentation and transcription for end-to-end handwritten paragraph recognition. In NIPS, 2016.

[^3]: C.Cao, X.Liu, Y.Yang, Y.Yu, J.Wang, Z.Wang, Y.Huang, L. Wang, C. Huang, W. Xu, D. Ramanan, and T. S. Huang. Look and think twice: Capturing top-down visual attention with feedback convolutional neural networks. In ICCV, 2015.

[^4]: L. Chen, H. Zhang, J. Xiao, L. Nie, J. Shao, W. Liu, and T. Chua. SCA-CNN: Spatial and channel-wise attention in convolutional networks for image captioning. In CVPR, 2017.

[^5]: Y. Chen, J. Li, H. Xiao, X. Jin, S. Yan, and J. Feng. Dual path networks. arXiv:1707.01629, 2017.

[^6]: F. Chollet. Xception: Deep learning with depthwise separable convolutions. In CVPR, 2017.

[^7]: J. S. Chung, A. Senior, O. Vinyals, and A. Zisserman. Lip reading sentences in the wild. In CVPR, 2017.

[^8]: K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In ICCV, 2015.

[^9]: K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[^10]: K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.

[^11]: S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 1997.

[^12]: G. Huang, Z. Liu, K. Q. Weinberger, and L. Maaten. Densely connected convolutional networks. In CVPR, 2017.

[^13]: Y. Ioannou, D. Robertson, R. Cipolla, and A. Criminisi. Deep roots: Improving CNN efficiency with hierarchical filter groups. In CVPR, 2017.

[^14]: S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.

[^15]: L. Itti and C. Koch. Computational modelling of visual attention. Nature reviews neuroscience, 2001.

[^16]: L. Itti, C. Koch, and E. Niebur. A model of saliency-based visual attention for rapid scene analysis. IEEE TPAMI, 1998.

[^17]: M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu. Spatial transformer networks. In NIPS, 2015.

[^18]: M. Jaderberg, A. Vedaldi, and A. Zisserman. Speeding up convolutional neural networks with low rank expansions. In BMVC, 2014.

[^19]: A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.

[^20]: H. Larochelle and G. E. Hinton. Learning to combine foveal glimpses with a third-order boltzmann machine. In NIPS, 2010.

[^21]: H. Lee, R. Grosse, R. Ranganath, and A. Y. Ng. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. In ICML, 2009.

[^22]: M. Lin, Q. Chen, and S. Yan. Network in network. arXiv:1312.4400, 2013.

[^23]: J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[^24]: A. Miech, I. Laptev, and J. Sivic. Learnable pooling with context gating for video classification. arXiv:1706.06905, 2017.

[^25]: V. Mnih, N. Heess, A. Graves, and K. Kavukcuoglu. Recurrent models of visual attention. In NIPS, 2014.

[^26]: V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In ICML, 2010.

[^27]: A. Newell, K. Yang, and J. Deng. Stacked hourglass networks for human pose estimation. In ECCV, 2016.

[^28]: B. A. Olshausen, C. H. Anderson, and D. C. V. Essen. A neurobiological model of visual attention and invariant pattern recognition based on dynamic routing of information. Journal of Neuroscience, 1993.

[^^29]: S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

[^30]: O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet large scale visual recognition challenge. IJCV, 2015.

[^31]: J. Sanchez, F. Perronnin, T. Mensink, and J. Verbeek. Image classification with the fisher vector: Theory and practice. RR-8209, INRIA, 2013.

[^32]: L. Shen, Z. Lin, and Q. Huang. Relay backpropagation for effective learning of deep convolutional neural networks. In ECCV, 2016.

[^33]: L. Shen, Z. Lin, G. Sun, and J. Hu. Places401 and places365 models. <https://github.com/lishen-shirley/> Places2-CNNs, 2016.

[^34]: L. Shen, G. Sun, Q. Huang, S. Wang, Z. Lin, and E. Wu. Multi-level discriminative dictionary learning with application to large scale image classification. IEEE TIP, 2015.

[^35]: K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[^36]: R. K. Srivastava, K. Greff, and J. Schmidhuber. Training very deep networks. In NIPS, 2015.

[^37]: M. F. Stollenga, J. Masci, F. Gomez, and J. Schmidhuber. Deep networks with internal selective attention through feedback connections. In NIPS, 2014.

[^38]: C.Szegedy, S.Ioffe, V.Vanhoucke, and A.Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. arXiv:1602.07261, 2016.

[^39]: C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

[^40]: C.Szegedy, V.Vanhoucke, S.Ioffe, J.Shlens, and Z.Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.

[^41]: A. Toshev and C. Szegedy. DeepPose: Human pose estimation via deep neural networks. In CVPR, 2014.

[^42:]: F. Wang, M. Jiang, C. Qian, S. Yang, C. Li, H. Zhang, X. Wang, and X. Tang. Residual attention network for image classification. In CVPR, 2017.

[^43]: S. Xie, R. Girshick, P. Dollar, Z. Tu, and K. He. Aggregated residual transformations for deep neural networks. In CVPR, 2017.

[^44]: K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhudinov, R. Zemel, and Y. Bengio. Show, attend and tell: Neural image caption generation with visual attention. In ICML, 2015.

[^45]: J. Yang, K. Yu, Y. Gong, and T. Huang. Linear spatial pyramid matching using sparse coding for image classification. In CVPR, 2009.

[^46]: J. Yosinski, J. Clune, Y. Bengio, and H. Lipson. How transferable are features in deep neural networks? In NIPS, 2014.

[^47]: X. Zhang, Z. Li, C. C. Loy, and D. Lin. Polynet: A pursuit of structural diversity in very deep networks. In CVPR, 2017.

[^48]: B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba. Places: A 10 million image database for scene recognition. IEEE TPAMI, 2017.
