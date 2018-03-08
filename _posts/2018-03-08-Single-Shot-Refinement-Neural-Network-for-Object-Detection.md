---
author: linhu
title: "Single-Shot Refinement Neural Network for Object Detection"
date: 2018-03-08 17:42:32+00:00
categories:
  - Object Detection
tags:
  - Object Detection
  - Computer Vision
  - Deep Learning
---

# Single-Shot Refinement Neural Network for Object Detection

## 摘要

对于目标检测，两阶段的方法如faster R-CNN达到了最高的精度，而单阶段的方法如SSD则更高效。为了继承这二者的优点同时克服其缺点，在本文中，我们提出了一种称为RefineDet的新型single-shot检测器，它比两阶段方法更准确，并保持单阶段方法的可比效率。RefineDet由两个相互连接的模块组成，即锚点优化模块（anchor refinement module）和对象检测模块（object detection module）。

具体而言，前者旨在（1）滤除负锚以减少分类器的搜索空间，以及（2）粗略地调整锚的位置和大小以为随后的回归器提供更好的初始化。

后一模块将前者改良后的锚作为输入，进一步改进回归并预测多类别标签。同时，我们设计了一个传输连接块（transfer connection block）来传输锚点优化模块中的特征，用来预测目标检测模块中对象的位置，大小和类别标签。多任务损失函数使我们能够以端到端的方式训练整个网络。在PASCAL VOC 2007，PASCAL VOC 2012和MS COCO上进行的大量实验表明，RefineDet能够高效地达到state-of-the-art 的检测精度。代码位于 [github](https://github.com/sfzhang15/RefineDet)。

## 介绍

单阶段方法通过对位置，比例和高宽比进行规则且密集地采样来检测对象。这样做的主要优点是计算效率高。然而，它的检测精度通常落后于两阶段方法，其中一个主要原因是类别失衡问题（class imbalance problem）。

一些最近的一阶段方法旨在解决类别不平衡问题，以提高检测精度。 Kong等人 [24]使用卷积特征图上的对象先验约束来显著减少对象的搜索空间。 Lin等人 [28]通过重塑（reshape）标准交叉熵损失将训练集中在一组稀疏的困难样本上，并对指定给良好分类样本的损失权重进行下调，来解决类别失衡问题。 Zhang等人 [53]设计了一个max-out标记机制来减少类别不平衡导致的虚警。

我们认为目前先进的两阶段方法如Faster R-CNN [36], R-FCN [5], 和FPN[27]，相比单阶段方法有三个优点：

1. 采用采样启发式两阶段结构处理类别失衡;

2. 使用两级级联来回归对象框参数;

3. 使用两阶段特征来描述对象。


本文设计了一个名为RefineDet的新型对象检测框架，以继承这两种方法（即一阶段和两阶段方法）的优点并克服它们的缺点。 它通过使用两个相互连接的模块（参见图1）改进了单阶段方法的架构，即锚点细化模块（ARM）和物体检测模块（ODM）。具体来说，ARM被设计为（1）识别和移除负锚以减少分类器的搜索空间，并且（2）粗略地调整锚的位置和大小来为随后的回归器提供更好的初始化。 ODM将前者的细化的锚定作为输入，以进一步改进回归和预测多类别标签。 如图1所示，这两个相互连接的模块模仿两级结构，从而继承了上述三个优点，以高效率产生精确的检测结果。 另外，我们设计了一个传输连接块（TCB）来传输ARM中的特征，以预测ODM中对象的位置，大小和类别标签。 多任务损失函数使我们能够以端到端的方式训练整个网络。

在PASCAL VOC 2007，PASCAL VOC 2012和MS COCO基准上进行的大量实验表明，RefineDet优于目前最先进的方法。具体来说，在VOC 2007和2012，使用VGG-16网络的RefineDet实现了85.8％和86.8％的mAPs。 同时，使用ResNet-101的RefineDet在MS COCO test-dev上获得41.8％的AP，超过了以往发表的最好的一阶段和两阶段方法的结果。 此外，RefineDet的时间效率高，在NVIDIA Titan X GPU上对输入图像尺寸为320* 320和512* 512分别以40.2 FPS和24.1 FPS运行。

这项工作的主要贡献总结如下。

1. 我们引入了一个新的物体检测一阶段框架，由两个互连模块组成，即ARM和ODM。 使得性能比两阶段方法更好，同时保持单阶段方法的高效率。
2. 为确保效率，我们设计了TCB来传输ARM中的特征，以便处理更具挑战性的任务，即预测ODM中准确的物体位置，尺寸和类别标签。
3. RefineDet在通用对象检测（即PASCAL VOC 2007 [10]，PASCAL VOC 2012 [11]和MS COCO [29]）达到了最新的state-of-the-art成果。

##网络结构

![图1：RefineDet的架构， 为了更好的可视化，我们只显示用于检测的层。 灰绿色平行四边形表示与不同特征层相关联的细化锚， 星星代表refined anchor boxes的中心，在图像上不规则分布。]({{ site.url }}{{ site.baseurl }}/assets/images/1520242669895.png)

图1：RefineDet的架构， 为了更好的可视化，我们只显示用于检测的层。 灰绿色平行四边形表示与不同特征层相关联的细化锚， 星星代表refined anchor boxes的中心，在图像上不规则分布。

参考图1所示的总体网络架构。与SSD类似，RefineDet基于前馈卷积网络，该网络生成固定数量的bboxs，以及表示这些bboxs中存在不同类别的对象的score， 接着通过非极大抑制来产生最终结果。 RefineDet由两个相互连接的模块组成，即ARM和ODM。 ARM旨在消除负向锚，以减少分类器的搜索空间，并粗略调整锚点的位置和大小，以便为随后的回归器提供更好的初始化，而ODM基于细化的锚点，旨在回归准确的对象位置并预测多类别标签 。

通过去除两个基础网络的分类层并添加辅助结构（即在ImageNet上预训练的VGG-16和ResNet-101）来构建ARM。ODM由TCBs预测层（即具有3*3卷积核大小的卷积层）的输出组成，其产生物体类别的评分以及相对于细化的锚box坐标的形状偏移。 下面说明RefineDet中的三个核心组件，即：
1. 传输连接块（TCB），将特征从ARM转换到ODM以供检测; 
2. 两步级联回归，准确回归物体的位置和大小;
3. 负锚过滤，尽早滤除良好分类的负锚，缓解不平衡问题。

**Transfer Connection Block**

为了在ARM和ODM之间建立链接，我们引入了TCB，将来自ARM的不同层的特征转换为ODM所需的形式，以便ODM可以共享来自ARM的特征。 值得注意的是，在ARM中，我们只在与锚点关联的特征图上使用TCB。 TCB的另一个功能是通过将高层特征添加到传输的特征中来集成大规模上下文，以提高检测精度。为了匹配它们之间的尺寸，我们使用反卷积操作来放大高层特征图并按元素对它们进行求和。 然后，我们在求和之后添加卷积层以确保特征的可区分性。 TCB的结构如图2所示。

![TCB overview]({{ site.url }}{{ site.baseurl }}/assets/images/1520243299571.png)

**Two-Step Cascaded Regression**

目前的一阶段方法依赖于基于不同尺度的各种特征层的一步回归预测物体的位置和大小，这在一些具有挑战性的场景下是相当不准确的，特别是对于小物体。 为此，我们提出了一个两步级联回归策略来回归对象的位置和大小。 也就是说，我们使用ARM来首先调整锚的位置和大小，以便为ODM中的回归提供更好的初始化。 具体而言，我们将n个anchor boxes与特征图上的每个规则划分的cell单元相关联。 每个anchor box相对于其相应cell单元的初始位置是固定的。 在每个特征图cell单元中，我们预测细化 anchor boxes相对于原始平铺anchor的四个偏移量以及表示这些框中存在前景对象的两个置信度分数。 因此，我们可以在每个特征图cell单元中生成n个细化锚点框（refined anchor boxes）。

获得细化锚点框后，我们将它们传递到ODM中的相应特征图上，以进一步生成对象类别和准确的对象位置和大小，如图1所示。ARM和ODM中对应的特征图具有相同的维度。 我们计算c类分数以及相对于 refined anchor boxes的四个精确的对象偏移量，为每个细化锚点框产生c + 4个输出以完成检测任务。 该过程与SSD中使用的默认框类似。 然而，与SSD直接使用规则平铺默认boxes进行检测相比，RefineDet使用两步策略，即ARM生成refined anchor boxes，ODM将refined anchor boxes作为输入用于进一步检测， 得到更精确的检测结果，特别是对于小物体。

**Negative Anchor Filtering**

为了尽早滤除良好分类的负锚，并缓解不平衡问题，我们设计了一个负锚过滤机制。 具体而言，在训练阶段，对于一个refined anchor box，如果其负置信度大于预设的阈值（经验地设定为 $\theta=0.99$），我们将在训练ODM时放弃它。 也就是说，我们只传递refined hard negative anchor boxes和refined
positive anchor boxes用来训练ODM。 同时在推理阶段，如果一个refined anchor box被分配了一个大于$\theta$的负置信度，它将在ODM中被丢弃。

## 训练和推断

**数据增强 Data Augmentation**

我们使用[30]中提出的几种数据增强策略来构建一个适应对象变化的鲁棒模型。也就是说，我们随机扩大和裁剪附加了随机光度测量扭曲的原始训练图像[20]并翻转生成训练样本。请参阅[30]了解更多详情。

**骨干网络 Backbone Network**

在我们的RefineDet中，我们使用VGG-16和ResNet-101作为骨干网络，它们在ILSVRC CLS-LOC数据集上预训练。值得注意的是，RefineDet也可以用其他预训练网络，如Inception V2，Inception ResNet 和ResNeXt-101。类似于DeepLab-LargeFOV，我们通过子采样参数将VGG-16的fc6和fc7转换为卷积层conv_fc6和conv_fc7。由于conv4_3和conv5_3与其他层相比具有不同的特征尺度，因此我们使用L2归一化将conv4_3和conv5_3中的特征尺度缩放到10和8，然后在back propagation期间学习尺度。同时，为了在多尺度上捕获高层信息和驱动对象检测，我们还在截断的VGG-16的末端上增加了两个额外的卷积层（即conv6_1和conv6_2），在截断的ResNet-101的末尾增加了一个额外的残差块（即res6）。

**Anchors Design and Matching**

为了处理不同尺度的对象，我们选择了四个特征层，其中VGG-16和ResNet-101的总stride尺寸分别为8,16,32和64像素，与预测的几种不同尺度的锚相关联。 每个特征层与一个特定尺度的锚点（尺度是相应层的总步幅stride大小的4倍）和三个长宽比（即0.5，1.0和2.0）相关联。 我们遵循[53]中不同层的锚尺度的设计，这可以确保不同尺度的锚在图像上具有相同的平铺密度。 同时，在训练阶段，我们根据jaccard overlap确定锚点与ground truth boxes之间的对应关系，并相应地对整个网络进行端到端的训练。具体来说，我们首先将每个ground truth与最佳overlap score的anchor box匹配，然后将anchor box与任何overlap高于0.5的ground truth进行匹配。

**困难负样本挖掘 Hard Negative Mining**

在匹配步骤之后，大多数 anchor boxes都是负的，对于ODM也是如此，即使有一些简单的负锚已被ARM滤除。 与SSD类似，我们使用困难负样本挖掘来缓解极端的前景 - 背景类别失衡，也就是说，我们选择一些具有高loss值的负 anchor boxes，以使负样本与正样本之间的比率低于3：1，而不是使用所有负锚或在训练时随机选择负锚。

**Loss Function**

RefineDet的损失函数由两部分组成，即ARM中的损失和ODM中的损失。 对于ARM，我们为每个锚点分配一个二元类标签（是对象或不是对象），并同时对其位置和大小进行回归，以获得细化的锚点。 之后，我们将负置信度小于阈值的细化锚点 refined anchors 传递给ODM，以进一步预测对象类别和准确的对象位置和大小。 有了这些定义，我们将损失函数定义为：


$$\begin{align}\mathcal L\left(\left\{ p_i \right\},\left\{ x_i \right\},\left\{ c_i \right\},\left\{ t_i \right\} \right)=\frac{1}{N_{arm}}\left( \sum_i\mathcal L_b\left({p_i,\left[{l_i^*\geq 1}\right]}\right)+ \sum_i\left[{l_i^*\geq 1}\right]\mathcal L_r\left({x_i,g_i^*}\right)\right)\\+\frac{1}{N_{odm}}\left( \sum_i\mathcal L_m\left({c_i,l_i^*}\right)+ \sum_i\left[{l_i^*\geq 1}\right]\mathcal L_r\left({t_i,g_i^*}\right)\right)\end{align}$$
其中$i$ 是指一个mini-batch中anchor的索引，$l_i^*$ 是anchor $i$ 的ground truth类标签，$g_i^*$ 是anchor $i$ 的ground truth位置与大小。$p_i$ 和$x_i$ 是anchor $i$ 是对象的预测置信度得分以及在ARM中anchor $i$ 的精确坐标。$c_i$ 和$t_i$ 是anchor $i$ 是预测对象类别以及在ODM中bbox的坐标。$N_{arm}$ 和$N_{odm}$ 分别是在ARM和ODM中的正锚的数量。二元分类损失函数$\mathcal L_b$ 是二类（对象 vs. 非对象）cross-entropy/log 损失函数，而多类分类损失函数$\mathcal L_m$ 是基于多类别置信度的softmax损失函数。与Fast R-CNN类似，我们使用smooth L1 loss作为回归损失$L_r$ 。艾弗森（Iverson）括号指示器函数$\left[{l_i^*\geq 1}\right]$ 在条件为真是输出1，即$l_i^*\geq1$ (anchor不是负的)，否则输出0。因此$\left[{l_i^*\geq 1}\right]\mathcal L_r$ 表明回归损失忽略了负锚。 值得注意的是，如果$N_{arm}=0$ ，我们令$\mathcal L_b\left({p_i,\left[{l_i^*\geq 1}\right]}\right)=0$ 以及$\mathcal L_r\left({x_i,g_i^*}\right)=0$ ;相应的如果$N_{odm}=0$ ，我们令$\mathcal L_m\left({c_i,\left[{l_i^*\geq 1}\right]}\right)=0$ 以及$\mathcal L_r\left({t_i,g_i^*}\right)=0$ 。
**优化 Optimization**
如上所述，RefineDet使用的骨干网络（即VGG-16和ResNet-101）在ILSVRC CLS-LOC数据集上预训练[37]。 对于基于VGG-16的RefineDet，我们使用“xavier”方法[17]随机初始化其两个额外附加卷积层（即conv6_1和conv6_2）的参数；对于基于ResNet-101的RefineDet，从标准偏差为0.01的零均值高斯分布中绘制额外残差块的参数 （即res6）。 我们在训练中将默认batch大小设置为32。然后使用SGD以0.9的动量和0.0005的权重衰减对整个网络进行微调fine-tune。 我们将初始学习率设置为$10^{-3}$，并针对不同的数据集使用稍微不同的学习率衰减策略，稍后将对此进行详细描述。
**推断 Inference**
在推断阶段，ARM首先滤除负置信度分数大于阈值$\theta$的平铺锚点，然后细化剩余锚点的位置和大小。 之后，ODM接收这些细化锚点，并且每张图像输出前400个高置信度检测结果。 最后，我们将jaccard overlap为0.45的非极大抑制应用于每个类别，并保留每张图像前200个高置信度检测结果以产生最终检测结果。
## 实验
实验在三个数据集上进行：PASCAL VOC 2007，PASCAL VOC 2012和MS COCO。 PASCAL VOC和MS COCO数据集分别包括20个和80个对象类，PASCAL VOC中的类别是MS COCO中的类别的子集。 我们用Caffe中实现了RefineDet，所有训练和测试代码以及经过训练后的模型均可在 [github](https://github.com/sfzhang15/RefineDet)上获得。
**PASCAL VOC 2007**
所有模型都在VOC 2007和VOC 2012训练集进行训练，并在VOC 2007测试集上进行测试。 前80k次迭代的学习率设置为$10^{-3}$，额外两个的20k次迭代分别将其衰减到$10^{-4}$和$10^{-5}$。 训练时使用默认batch大小为32，并且仅使用VGG-16作为PASCAL VOC数据集上所有实验的骨干网络，包括VOC 2007和VOC 2012。
我们比较了RefineDet和表1中state-of-the-art的探测器。对于低维度的输入（即320$*$320），RefineDet可达到80.0％的mAP（无附加功能），这是第一种使用这种小尺寸输入图像达到80％mAP的方法，比几个当前的方法要好得多。通过使用更大的输入尺寸512$*$512，RefineDet达到81.8％的mAP，超过了所有的单阶段方法，例如RON384 [24]，SSD513 [13]，DSSD513 [13]等。与两阶段方法相比，RefineDet512的表现要好于其中的大多数，除了CoupleNet [54]，其基于ResNet-101并且使用了比我们的RefineDet512更大的输入尺寸（即1000$*$600）。正如[21]指出的那样，输入大小会显着影响检测精度。原因是高分辨率输入使检测器能够清楚地“看到”小物体，以增加检测成功率。为了减少输入大小对公平比较的影响，我们使用多尺度测试策略来评估RefineDet，实现83.1％（RefineDet320 +）和83.8％（RefineDet512 +）的mAP，大幅超过目前最先进的方法。
![1520243501036]({{ site.url }}{{ site.baseurl }}/assets/images/1520243501036.png)
**1. 运行时间**
在表1的第五列中提供了RefineDet和state-of-the-art方法在NVIDIA Titan X，CUDA 8.0和cuDNN v6的机器上，batch大小为1时的推断速度。如表1所示，我们发现RefineDet在输入尺寸为320$*$320和512$*$512时，分别以24.8ms（40.3 FPS）和41.5ms（24.1 FPS）的速度处理图像。据我们所知，RefineDet是第一个在PASCAL VOC 2007上实现检测精度高于80％mAP的实时方法。与SSD，RON，DSSD和DSOD相比，RefineDet可以在特征图上关联更少的锚点框（SSD512关联24564个锚定框vs.RefineDet512关联16320个锚定框）。然而，RefineDet仍然以高效率实现高精度，主要得益于两个互连模块的设计（即两步回归），这使RefineDet能够适应对象的不同尺度和长宽比。同时，只有YOLO和SSD300$^*$比我们的RefineDet320稍快，但它们的准确度比我们的差16.6％和2.5％。总之，RefineDet在精度和速度之间实现了最佳平衡。
**2. 消融实验**
为了论证RefineDet中不同组件的有效性，我们构建了四个变体，并在VOC 2007上对它们进行评估，如表3所示。具体来说，为了公平比较，我们在评估中使用相同的参数设置和输入大小（320$*$320）。 所有模型都在VOC 2007和VOC 2012训练集上进行训练，并在VOC 2007测试集上进行测试。

![1520243694234]({{ site.url }}{{ site.baseurl }}/assets/images/1520243694234.png)
- **Negative Anchor Filtering**

为了论证负锚过滤的有效性，在训练和测试中，我们把锚点判断为负的置信度阈值$\theta$均设置为1.0。此时所有细化锚点都将被发送到ODM进行检测。 RefineDet的其他部分保持不变。 去除负锚过滤导致mAP下降0.5％（即80.0％ vs. 79.5％）。原因在于这些良好分类的负锚大多会在训练过程中被滤除，从而在一定程度上解决了类别失衡问题。

- **Two-Step Cascaded Regression**

为了验证两步级联回归的有效性，我们重新设计了网络结构，直接使用规则平铺的锚而不是ARM中的细化锚（见表3第四列）。 如表3所示，我们发现mAP从79.5％降至77.3％。 这种急剧下降（即2.2％）表明，两步锚级联回归明显有助于提升性能。

- **Transfer Connection Block**

我们通过去除RefineDet中的TCB来构建一个新网络，并重新定义ARM中的损失函数像SSD一样直接检测多类别对象，以便演示TCB的效果。 模型的检测精度在表3的第五列中给出。我们比较表3中第四和第五列的结果（77.3％对比76.2％），发现TCB改善了1.1％mAP。 主要原因是该模型能够继承ARM中的判别特征，并通过使用TCB整合大规模上下文信息来提高检测精度。

**PASCAL VOC 2012**

> 省略精度说明

![1520243501036]({{ site.url }}{{ site.baseurl }}/assets/images/1520243501036.png)
**MS COCO**
> 省略精度说明
另外，RetinaNet800的主要贡献：焦点损失focal loss，可与我们的方法相辅相成。 我们相信它可以在RefineNet中使用以进一步提高性能。
![1520243660055]({{ site.url }}{{ site.baseurl }}/assets/images/1520243660055.png)
**From MS COCO to PASCAL VOC**
我们研究了如何用MS COCO数据集帮助提升在PASCAL VOC上的检测精度。由于PASCAL VOC中的对象类别是MS COCO的子集，因此我们通过对参数进行二次抽样subsampling，来对在MS COCO预训练的检测模型直接进行微调fine-tune，在VOC 2007测试集上达到了84.0％ mAP（RefineDet320）和85.2％ mAP（RefineDet512），在VOC 2012测试集上达到了82.7％ mAP（RefineDet320）和85.0％ mAP（RefineDet512），如表4所示。在使用多尺度测试后，检测精度分别提高到85.6％，85.8％，86.0％和86.8％。
如表4所示，使用MS COCO和PASCAL VOC的训练数据，我们的RefineDet获得了VOC 2007和VOC 2012的最高mAP分数。最重要的是，我们基于VGG-16的单一模型RefineNet512+在VOC 2012排行榜排名前五位（见[9]），这是所有一阶段方法中最准确的。其他实现更好结果的两阶段方法是基于更深的网络（例如，ResNet-101 [19]和ResNeXt-101 [49]）或者使用了集成机制。
![1520243755921]({{ site.url }}{{ site.baseurl }}/assets/images/1520243755921.png)
## 总结
在本文中，我们提出了一个基于单发细化神经网络的检测器，它由两个互连的模块组成，即ARM和ODM。 ARM旨在过滤出负锚以减少分类器的搜索空间，并粗略地调整锚的位置和大小，以便为后续的回归器提供更好的初始化，而ODM则将前面ARM细化的锚作为输入，回归出准确的物体位置和大小，并预测相应的多类别标签。整个网络以端到端的方式按多任务损失进行训练。我们在PASCAL VOC 2007，PASCAL VOC 2012和MS COCO数据集上进行了多次实验，以证明RefineDet能够高效地达到state-of-the-art的检测精度。未来，我们计划使用RefineDet来检测其他特定类型的物体，例如行人，车辆和人脸，并在RefineDet中引入注意机制以进一步提高性能。