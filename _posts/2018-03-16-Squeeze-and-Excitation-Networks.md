---
typora-copy-images-to: assets/images
title: "Squeeze-and-Excitation Networks"
last_modified_at: 2018-03-08T21:28:04-22:00:00
categories:
  - backbone network
tags:
  - Deep Learning
  - Computer Vision
  - backbone network
  - CVPR 2017
---

>[��������](https://arxiv.org/abs/1709.01507) 
>[�����ַ](https://github.com/hujie-frank/SENet)

# ժҪ

&emsp;&emsp;��������罨���ھ������Ļ����ϣ�ͨ���ںϱ��ظ������ڵĿռ���Ϣ���ŵ���Ϣ����ȡ��Ϣ������Ϊ���������ı�������������еĹ����Ѿ���ʾ����ǿ�ռ����ĺô�����������У�����רע��Ƶ�������һ����ӱ�ļܹ���Ԫ�����ǳ�֮Ϊ����ѹ�뼤������SE��ģ�飬ͨ����ʽ��ģͨ��֮��������������Ӧ����У׼ͨ��������Ӧ������֤����ͨ������Щ��ѵ���һ�����ǿ��Թ���SENet��ϵ�ṹ���ھ�����ս�Ե����ݼ��н��зǳ��õķ�����������Ҫ���ǣ����Ƿ���SE����΢С�ļ���ɱ�Ϊ�������Ƚ�����ܹ��������ŵ����ܸĽ��� SENET���������ILSVRC 2017�����ύ�Ļ������÷����ύ����˵�һ��������ǰ5����������Ž�����2.251���������2016��Ļ�ʤ��Ŀ�����25������Ը��ơ�
&emsp;&emsp;��������罨���ھ�������Ļ����ϣ�ʹ�þֲ���������local receptive field���ںϿռ���Ϣ��ͨ����Ϣ����ȡ������Ϣ���������кܶ๤������ǿ�ռ���루spatial encoding���ĽǶ�����������ı�ʾ������ȡ���˳ɹ�����������Ҫ�۽���ͨ��ά�ȣ������һ���µĽṹ��Ԫ������ Squeeze-and-Excitation(SE) ����Ԫ��ͨ����ʽ��ģͨ��֮�������ԣ���������Ӧ�ĵ�����ͨ����������Ӧֵ��������֤��ͨ������Щ��Ԫ�ѵ���һ�����ǿ��Թ���SENet��ϵ�ṹ�����ھ�����ս�Ե����ݼ��ϴﵽ�ǳ��õķ���������Ҫ���ǣ������SE��Ԫ��ӵ����е�state-of-the-art����������У�ֻ�����Ӻ�С�ļ��㿪����ȴ�����������������ܡ�����SENet���߻����ILSVRC2017��������ĵ�һ��������top-5���������Ž�����2.251%����2016��Ļ�ʤ����ȴﵽ��~25�������������

# 1. ����

&emsp;&emsp;�ھ���������У�ÿ��������������˲�����ѧϰ�����������ͨ���ľֲ��ռ�����ģʽ��Ҳ����˵������˲���������ȡ�ֲ����������еĿռ��ͨ��ά�ȵ��ں���Ϣ��ͨ���������һϵ�о�����Լ������Լ����ͽ������㣬CNN���Ի�þ���ȫ�ָ�������ķֲ�ģʽ����Ϊ������ͼ�������������һЩ��������������ͨ�����������ڻ�ȡ�ռ�����Ե�ѧϰ������������������ܣ����Ҳ���Ҫ����ļල������Inception�ܹ�[^14,39]��ͨ����ģ���м����߶ȴ�����ȡ�ýϸߵľ��ȡ�����һЩ������ͼ��ģ���õĿռ������[^1,27]������ӿռ�ע�������ƣ�spatial attention��[^17]��
&emsp;&emsp;������������ͬ��������Ҫ̽������ܹ���Ƶ���һ�����桪��ͨ�������ԡ��������һ���µ����絥Ԫ������ Squeeze-and-Excitation(SE) �� ��Ԫ��ϣ��ͨ���Ծ��������ͨ���������Խ�����ȷ�ؽ�ģ���������ı��������Ϊ���������һ�ֻ���ʹ��������Զ���������΢У��recalibration������������Ϳ���ѧϰʹ��ȫ����Ϣ����ѡ���Եؼ�ǿ����������Ϣ����������������������
&emsp;&emsp;SE��Ԫ�����ṹ��ͼ1��������������ı任$$\mathrm{F_{tr}:X\rightarrow U,X\in\mathbb{R}^{{W ^\prime}\times{H ^\prime}\times{C^\prime}},U\in\mathbb{R}^{W\times H\times C}}$$������һ����һϵ�о���������ǿ��Թ���һ����Ӧ��SE��Ԫ��ִ����������΢У������$$U$$����ͨ��һ��*squeeze*���������ռ� $$W\times H$$ ��Χ�ڵ�����ͼ�ۺ������γ�һ��ͨ�������������������Ƕ����ͨ��ά��������Ӧ��ȫ�ֲַ���ʹ��������ȫ�ֽ��������Ϣ�ܹ�����ϵͲ������á���������*excitation*������ͨ��һ������ͨ�����Ŷȵ�self-gating����Ϊÿ��ͨ��ѧϰ����ض������ļ��sample-specific activations����������ÿ��ͨ���ļ����� ����ͼ $$U$$ ���ű����¼�Ȩ�Բ���SE��������Ȼ����Խ���ֱ���ṩ�������Ĳ㡣

![1520924591103]({{ site.url }}{{ site.baseurl }}/assets/images/1520924591103.png)

&emsp;&emsp;SE�������ͨ���򵥵ضѵ�һϵ��SE������Ԫ�����ɡ� SE��ԪҲ����ֱ�������滻����ṹ�κ���ȵ�drop-in��Ԫ�� Ȼ������Ȼ������Ԫ��ģ����ͨ�õģ���6.3����ʾ�������ڲ�ͬ��Ȱ��ݵĽ�ɫ��Ҫ��Ӧ��������� �ڽ���Ĳ��У���������޹صķ�ʽ��class agnostic manner��ѧϰ��ȡ������Ϣ����������ǿ����Ͳ��ʾ�������� �ں���Ĳ��У�SE��Խ��Խ�ػ����Ը߶�����ض��ķ�ʽ��class-specific manner����Ӧ��ͬ�����롣 ��ˣ�ͨ��SE���������΢У�������������ͨ��������������ۻ���
&emsp;&emsp;�µ�CNN�ܹ��Ŀ�����һ�������ս�ԵĹ�������ͨ���漰����µĳ������Ͳ����õ�ѡ�� ���֮�£����������SE��Ԫ�������ܼ򵥣����ҿ���ֱ��������state-of-the-art�ļܹ�һ��ʹ�ã���Щ�ܹ��ľ�������ͨ��ֱ���滻���ӦSE�������õ���ǿ�� �������4����ʾ��SE��Ԫ�ڼ��������������ģ���ģ�͸����Ժͼ��㸺���Ͻ���΢���ӡ� Ϊ��֧������۶ϣ����ǿ����˼���SENets���ֱ���SE-ResNet��SE-Inception��SE-ResNeXt��SE-Inception-ResNet������ImageNet 2012���ݼ�[^30]�϶�SENets�����˹㷺�������� ���⣬Ϊ��֤��SE��Ԫ�ķ������������Ļ��ṩ�����������ݼ��ϵĽ��������������ķ������޶����ض����ݼ�������
&emsp;&emsp;ʹ��SENets����������Ӯ����ILSVRC 2017���ྺ���ĵ�һ�����ڲ��Լ��ϴﵽ��2.251����top-5 error�� ��ǰһ��Ļ�ʤ����ȣ������һ��Ĺھ���25���������������top-5 errorΪ2.991������ 

# 2. ��ع���

**Deep architectures**
&emsp;&emsp;�кܶ๤��ͨ���������������ܹ�ʹģ�͸����׵�ѧϰ�������������ģ�����ܡ�VGGNets[^35]��Inceptionģ��[^39]֤���˿���ͨ������������������ܣ�������Խ��֮ǰILSVRC 2014�ϵķ�����Batch normalization (BN)[^14]����������ӿ�������ÿһ���������ݵĵ�Ԫ���ȶ�ѧϰ���̣��Ӷ������ݶ��������еĴ�����ʹ�ø���������Ҳ���Թ�����[ResNet](https://github.com/binLearning/caffe_toolkit/tree/master/ResNet)��[ResNet-v2](https://github.com/binLearning/caffe_toolkit/tree/master/ResNet-v2)�������м�����ӳ����ʽ����Ծ���ӣ�identity-based skip connections��ʹ����ѧϰ�в������������Ϣ�ڵ�Ԫ�������������ƽ�������ܹ�������ķ�չ��[DenseNet](https://github.com/binLearning/caffe_toolkit/tree/master/DenseNet)��[DPN](https://github.com/binLearning/caffe_toolkit/tree/master/DPN)ͨ������������������ӻ�����������������ѧϰ�ͱ�ʾ���ܡ� 
&emsp;&emsp;��һ�������ǵ���������ģ������Ĺ�����ʽ����������grouped convolutions�������������ӻ�����cardinality���任���ϵĴ�С��[^13,43]����Deep roots��[ResNeXt](https://github.com/binLearning/caffe_toolkit/tree/master/ResNeXt)����ʾ���������ѧϰ�����ḻ�ı�ʾ�����֧�����multi-branch convolutions��������Ϊ�������ķ���������ģ����Խ��и����ľ��������ϣ���Inceptionϵ�С���ͨ�������һ���µ�������Ϸ�ʽ�����Զ����ڿռ�ṹ[^6,18]����Xception����Ҳ��������ʹ��1x1��׼����˲���[^22]���д�����NIN����һ����˵��Щ������Ҫ��Ϊ�˽���ģ�ͺͼ��㸴�Ӷȡ����ַ�����ǰ�������ͨ����ϵ�ھֲ���������Ա�����Ϊʵ���޹أ�instance-agnostic���ĺ�����ϣ�Ҳ����˵��������������ݸ�ͨ��������������ͬ�ģ����������صġ���֮�෴���������һ���µĻ��ƣ�ʹ��ȫ����Ϣ��ȷ�ؽ�ģ��ͨ��֮��Ķ�̬������������ϵ���Ӷ�����ѧϰ���̲���������ı�ʾ������
**Attention and gating mechanisms**
&emsp;&emsp;ע�������ƣ�attention���������ü�����Դ�ķ���ƫ���������ź�����Ϣ�����Ĳ��֣���Щ���Ƶ��о��ͷ�չһֱ���񾭿�ѧ�糤���������о�����[^15,16,28]�������꿪ʼ������������������У��ںܶ������ж������м�����������ͼ��Ķ�λ�����[^3,17]���������е�ģ��[^2,24]����һ���Ǻ����޺�������softmax��sigmoid���������з�������ʹ��[^11, 37]��������о���������������ͼ�����[^4,44]�ʹ���[^7]����������������Ч�ؾۺ϶�ģ̬���ݡ�����ЩӦ���У���һ���ڱ�ʾ�ϸ߲������������һ���������ģʽƥ�䡣
&emsp;&emsp;highway����ʹ�����޻��������򻯿�ݣ�shortcut�����ӣ�ʹ�ÿ���ѵ���ǳ�������硣 Wang�����ڡ�Residual attention network for image classification��[^42]�н�����һ����Ч��trunk-and-maskע�������Ʋ�ʹ����ɳ©ģ�飨hourglass module��[^27]�����ɹ�������������ָ����������������Ԫ�����뵽��Ȳв�������м�׶�֮�䡣 ���֮�£����������SE��Ԫ��һ�������������޻��ƣ�ר�����ڶ�ͨ��ά�ȵĹ�ϵ���н�ģ����������ǿ����������ģ��ı�ʾ������

# 3. Squeeze-and-Excitation Blocks

&emsp;&emsp;Squeeze-and-Excitation block��һ�����㵥Ԫ���ɹ������κθ����ı任��$$\mathrm{F_{tr}:X\rightarrow U,X\in\mathbb{R}^{{W ^\prime}\times{H ^\prime}\times{C^\prime}},U\in\mathbb{R}^{W\times H\times C}}$$��Ϊ�˼�˵����������ķ����У����ǽ�$$\mathrm{F_{tr}}$$��Ϊ��׼�ľ����������� $$\mathrm{V=[v_1,v_2,\cdots,v_c]}$$ ��ʾѧϰ�����˲�������˼��ϣ�����$$\mathrm{v_c}$$ ָ��c���˲����Ĳ��������ǽ��ſ��Խ�$$\mathrm{F_{tr}}$$ �����д��$$\mathrm{U=[u_1,u_2,\cdots,u_c]}$$����ô��

$$\mathrm{u_c=v_c{*}X=\sum^{C^\prime}_{s=1}v^s_c {*} x^s}.\tag{1}$$ 

&emsp;&emsp;�˴� $$^{*}$$ ��ʾ����� $$\mathrm{v_c=[v_c^1,v_c^2,\cdots,v_c^{C^\prime}]}$$ �Լ� $$\mathrm{X=[x^1,x^2,\cdots,x^{C^\prime}]}$$ (Ϊ�˼򻯷��ţ�ʡ����bias����)���˴�$$\mathrm{v_c^s}$$ ��һ��2D�ռ����ˣ���ʾ������ $$\mathrm{X}$$ ��Ӧͨ���� $$\mathrm{v_c}$$ ��һ��ͨ�����������������ͨ�����ܺͣ����ͨ��������ϵ��ʽǶ�뵽�� $$\mathrm{v_c}$$ �У�����Щ���������˲�������Ŀռ�����Ծ�����һ�����ǵ�Ŀ����ʹ�����ܹ���߶Ը�����Ϣ�����������жȣ��Ա����ת���������ǣ������Ʋ�̫���õ������� ���ǽ����ڽ�����һ��ת��֮ǰͨ���������裨��squeeze��excitation������ȷ��ģͨ���໥������ϵ���Ӷ�����У׼�˲�����Ӧ��SE������Ԫ��ͼʾ��ͼ1��

## 3.1 Squeeze: Global Information Embedding

&emsp;&emsp;&emsp;&emsp;Ϊ������ͨ��������ϵ���������ȿ������������ÿ��ͨ�����źš� ÿһ����ѧ�Ĺ���������һ���ֲ����������ÿ��ת����Ԫ�����U���������ø�����֮�����������Ϣ�� ��������ڽ�����Ƚ�С������ϵͲ��б�ø������ء�

&emsp;&emsp;Ϊ�˻���������⣬���ǽ��齫ȫ�ֿռ���Ϣѹ����һ��ͨ����������ͨ��ʹ��ȫ��ƽ���ػ���global average pooling������ͨ��ͳ��������ʵ�֡���ʽ�ϣ�ͳ���� $$\mathrm z\in \mathbb R^C$$��ͨ��ѹ��$$\mathrm U$$ͨ���ռ�ά�� $$\mathrm{W\times H}$$ ���ɵģ����� $$\mathrm z$$ �ĵ�c��Ԫ������ʽ���㣺

$$z_c=\mathrm{F_{sq}(u_c)=\frac{1}{W\times H}\sum^W_{i=1}\sum^H_{j=1}}u_c(i,j).\tag{2}$$

> *Discussion.*
>
> ת������� $$U$$ ���Ա�����Ϊ�ֲ��������ļ��ϣ���ͳ�����ݱ�ʾ����ͼ��������Щ��Ϣ���������̵Ĺ����к��ձ�[^31,34,45]�� ����ѡ������򵥵�ȫ��ƽ���ػ�������Ҳ���Բ��ø����ӵľۺϲ��ԡ�

## 3.2. Excitation: Adaptive Recalibration

&emsp;&emsp;Ϊ������squeeze�����оۺϵ���Ϣ�����ǽ����Ž��еڶ����������ò���ּ����ȫ��ȡͨ������ԡ�Ϊ�ˣ���������������������׼��

- ���ȣ������������ģ��ر����������ܹ�ѧϰͨ��֮��ķ������໥���ã�;
- ��Σ�������ѧϰһ���ǻ���Ĺ�ϵ����Ϊ��ͨ�� ����ǿ���뵥�μ����෴�� Ϊ�˷�����Щ��׼������ѡ��ʹ��һ���򵥵��ſػ��ƺ�һ��sigmoid���

$$\mathbf{s} = \mathbf{F}_{ex}(\mathbf{z}, \mathbf{W}) = \sigma(g(\mathbf{z}, \mathbf{W})) = \sigma(\mathbf{W}_2\delta(\mathbf{W}_1\mathbf{z}))\tag{3}$$

&emsp;&emsp;����$$\delta$$��ָReLU[^26]������$$\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$$ �� $$\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$$��Ϊ������ģ�͸��ӶȺ͸�������������ͨ��Χ���ڷ�����ReLUǰ�������ȫ���ӣ�FC�����γɵ�ƿ�����������Ż��ƣ�����ά�����Ϊ$$\mathbf{W}_1$$����ά����Ϊ $$r$$�����ǰ�������Ϊ16���������ѡ����6.3�������ۣ���һ��ReLU��Ȼ����һ������Ϊ$$\mathbf{W}_2$$����ά�㡣����������ͨ��ʹ����������ֵ��rescaling֮ǰ�ı任���$$\mathbf{U}$$�õ���
$$\widetilde{\mathbf{x}}_c = \mathbf{F}_{scale}(\mathbf{u}_c, s_c) = s_c \cdot \mathbf{u}_c\tag{4}$$
����$$\widetilde{\mathbf{X}} = [\widetilde{\mathbf{x}}_1, \widetilde{\mathbf{x}}_2, \dots, \widetilde{\mathbf{x}}_{C}]$$��$$\mathbf{F}_{scale}(\mathbf{u}_c, s_c)$$ָ����ͼ $$\mathbf{u}_c \in \mathbb{R}^{W \times H}$$ �ͱ���$$s_c$$֮��Ķ�Ӧͨ���˻���

> *Discussion.*
>
> �����ڴ˴���Ϊ��Ӧ�ض����������� $$z$$ ��ͨ��Ȩ�ء����ⷽ�棬SE��Ԫ������������������Ϊ�����Ķ�̬���ԣ���������������ı������

## 3.3. Exemplars: SE-Inception and SE-ResNet

&emsp;&emsp;SE����������ζ��������ֱ��Ӧ���ڱ�׼���֮��ı任��Ϊ��˵����һ�㣬����ͨ����SE�鼯�ɵ��������е�����ܹ�ϵ�У�Inception��ResNet��������SENets��ͨ�����任$$\mathbf{F}_{tr}$$����һ�������Inceptionģ�飨�μ�ͼ2����ΪInception���繹��SE��Ԫ��ͨ���Լܹ��е�ÿ��ģ����и��ģ����ǹ�����һ��SE-Inception���硣

![52112481026]({{ site.url }}{{ site.baseurl }}/assets/images/1521124810265.png)

&emsp;&emsp;�в����缰�������ѧϰ��ȱ�ʾ����ǳ���Ч�����ǿ�����һϵ�е�SE��Ԫ���ֱ���ResNet[^9]��ResNeXt[^43]��Inception-ResNet[^38]���ɡ�ͼ3������SE-ResNetģ��ļܹ��������SE��Ԫ�任 $$\mathbf{F}_{tr}$$ ����Ϊ�ǲв�ģ��ķǺ�ȷ�֧��ѹ���ͼ����������ȷ�֧���֮ǰ�����á�

![52112487610]({{ site.url }}{{ site.baseurl }}/assets/images/1521124876101.png)

# 4. Model and Computational Complexity

&emsp;&emsp;SENetͨ���ѵ�һ��SE��Ԫ��������ʵ���ϣ�����ͨ����ԭʼ���Ӧ��SE���֣���SE�в�飩�滻ÿ��ԭʼ�飨���в�飩�������ġ������ڱ�1��������SE-ResNet-50��SE-ResNeXt-50�ļܹ���

![52112498314]({{ site.url }}{{ site.baseurl }}/assets/images/1521124983140.png)

&emsp;&emsp;��ʵ����SE��Ԫ��ģ�͸��ӶȺͼ��㿪�������ڿɽ��ܷ�Χ�ڲ���ʵ�ÿ��еģ�����ڿ�����������Ҫ�ġ�Ϊ��˵��ģ��Ŀ��������ǱȽ���ResNet-50��SE-ResNet-50����Ϊ���ӣ�����SE-ResNet-50�ľ�ȷ����������ResNet-50���ӽ�������������ResNet-101���磨���2��ʾ��������$$224\times 224$$���ص�����ͼ��ResNet-50����ǰ�򴫲���Ҫ$$\sim$$3.86 GFLOP��ÿ��SE��Ԫ��Ҫʹ��һ��ѹ���׶ε�ȫ��ƽ���ػ����������������׶��е�Сȫ���Ӳ㣬�Լ���������������ͨ�����Ų������ܵ���˵��SE-ResNet-50��Ҫ$$\sim$$3.87 GFLOP�������ԭʼ��ResNet-50ֻ���������0.26%�Ŀ�����

![52112504608]({{ site.url }}{{ site.baseurl }}/assets/images/1521125046086.png)

&emsp;&emsp;��ʵ���У�ѵ���������ݴ�СΪ256��ͼ��ResNet-50��һ��ǰ�򴫲��ͷ��򴫲�����190ms����SE-ResNet-50�򻨷�209ms������ʱ�䶼���ھ���88��NVIDIA Titan X GPU�ķ�������ִ�еõ��ģ���������Ϊ����һ������Ŀ�������Ϊ�����е�GPU���У�ȫ�ֳػ���С���ڻ��������Ż��̶Ƚϵ͡����⣬���ǵ�Ƕ��ʽ�豸ʹ��CPU���㣬���ǻ���ÿ��ģ�͵�CPU�ƶ�ʱ������˻�׼���ԣ�����$$224\times 224$$���ص�����ͼ��ResNet-50������164ms�����֮�£�SE-ResNet-50������167ms��SE��Ԫ����������Ķ�����㿪����������ģ�����ܵ�������˵�Ǻ���ģ��ڵ�6������ϸ���ۣ���

&emsp;&emsp;�����������ǿ���������ĵ�Ԫ����ĸ��Ӳ��������и��Ӳ������������Ż��Ƶ�����ȫ���Ӳ��У�ֻ����������������һС���֡���ȷ�е�˵������ĸ��Ӳ�������������ʽ������

$$\frac{2}{r} \sum_{s=1}^S N_s \cdot {C_s}^2\tag{5}$$

&emsp;&emsp;����$$r$$��ʾѹ���ʣ����������е�ʵ���н� $$r$$ ����Ϊ16����$$S$$ ָ���ǽ׶�������ÿ���׶���ָ�ڹ�ͬ�ռ�ά�ȵ�����ͼ�����еĿ�ļ��ϣ���$$C_s$$��ʾ�׶�$$s$$�����ͨ����ά�ȣ�$$N_s$$��ʾ�ظ��Ŀ��š��ܵ���˵��SE-ResNet-50��ResNet-50��Ҫ���$$\sim$$2500�����֮��������$$\sim$$250�򸽼Ӳ��������������$$\sim$$10%�Ĳ�������������Щ���Ӳ����еĴ󲿷���������������׶Σ�������׶μ�����excitation����������ͨ��ά�����SE��Ԫ�����ս׶εĿ�����Խϴ�Ȼ�����Ƿ�������������ܵı߼ʳɱ��ϱ��Ƴ�����ImageNet���ݼ���top-1������$$<0.1\%$$ ��������Բ������Ӽ��ٵ�$$\sim$$4%�����ڲ���������Ҫ���ǵ�����¿��������õġ�

# 5. Implementation

&emsp;&emsp;��ѵ�������У�������ѭ��׼��������ʹ������߶Ȳü�[^39]��$$224\times 224$$���أ���Inception-ResNet-v2[^38]��SE-Inception-ResNet-v2��Ϊ$$299\times 299$$�����������ˮƽ��ת������������ǿ������ͼ��ͨ����ȥͨ����ֵ�����й�һ�������⣬���ǲ���[^32]�����������ݾ�����Խ���mini-batch���������ֲ����ֲ������ȡ����������ǵķֲ�ʽѧϰϵͳ��ROCS���Ͻ���ѵ�������ܹ����д�������ĸ�Ч����ѵ��������ʹ��synchronous SGD�����Ż�������Ϊ0.9��mini-batch�Ĵ�СΪ1024����ÿ��GPU�Ϸ�Ϊ32��ͼ��� sub-batch����4����������ÿ������������8��GPU������ʼѧϰ����Ϊ0.6��ÿ30���������ڼ���10����ʹ��[^8]��������Ȩ�س�ʼ�����ԣ�����ģ�Ͷ����㿪ʼѵ��100���������ڡ�

# 6. Experiments

&emsp;&emsp;�����������ImageNet 2012���ݼ��Ͻ����˴�����ʵ��[^30]����Ŀ���ǣ�����̽���������SE��Ԫ�Բ�ͬ��ȵĻ��������Ӱ�죻��Σ���������state-of-the-art������ܹ����ɺ��������ּ�ڹ�ƽ�Ƚ�SENets�ͷ�SENets��������׷�����������������������ǽ������ύ��ILSVRC 2017���������ģ�͵Ľ���Լ���ϸ��Ϣ�����⣬������Places365-Challenge�����������ݼ�[^48]�Ͻ�����ʵ�飬���о�SENets�Ƿ��ܹ��ܺõط������������ݼ�����������о��˼��������ã�������ʵ�����������һЩ������

## 6.1. ImageNet Classification

&emsp;&emsp;ImageNet 2012���ݼ���������1000������128����ѵ��ͼ���5������֤ͼ��������ѵ������ѵ�����磬������֤����ʹ�����Ĳü�ͼ������������`top-1`��`top-5`�����ʣ�����ÿ��ͼ��̱����ȹ�һ��Ϊ256��Ȼ���ÿ��ͼ���вü���$$224\times 224$$�����أ�������Inception-ResNet-v2��SE-Inception-ResNet-v2��ÿ��ͼ��Ķ̱����ȹ�һ����352��Ȼ��ü���$$299\times 299$$�����أ���

**������ȡ�**�������Ƚ�SE-ResNet��һϵ�б�׼ResNet�ܹ����бȽϡ�ÿ��ResNet������Ӧ��SE-ResNet��ʹ����ͬ���Ż���������ѵ������֤���ϲ�ͬ������������2��ʾ������SE���ڲ�ͬ����ϵ�������ʼ����������ܣ������㸴�Ӷ�ȴ���ӵĺ��١�

&emsp;&emsp;ֵ��ע����ǣ�SE-ResNet-50ʵ���˵��ü�ͼ��6.62%��`top-5`��֤�����ʣ�������ResNet-50��7.48%��0.86%���ӽ������ResNet-101���磨6.52%��`top-5`�����ʣ�����ֻ��ResNet-101һ��ļ��㿪����3.87 GFLOPs vs. 7.58 GFLOPs�������ֹ����ڸ����������ظ���SE-ResNet-101��6.07%��`top-5`�����ʣ������ﵽ���ҳ����˸����ResNet-152���磨6.34%��`top-5`�����ʣ���ͼ4�ֱ������SE-ResNets��ResNets��ѵ������֤���ߡ���Ȼע�⵽SE��Ԫ������������ȣ��������ǵļ���Ч�ʼ��ߣ���ʹ����չ�����ܹ�����ȴﵽ����ݼ��ĵ���Ҳ�ܲ������õĻر������ң����ǿ���ͨ���Ը��ֲ�ͬ��ȵ�ѵ�������ܸĽ���һ�µģ������SE��Ԫ����ĸĽ����������ӻ����ܹ���Ƚ��ʹ�á�

![52112508662]({{ site.url }}{{ site.baseurl }}/assets/images/1521125086621.png)

**���ִ��ܹ����ɡ�**���������ǽ��о�SE��Ԫ����������state-of-the-art�ļܹ�Inception-ResNet-v2[^38]��ResNeXt[^43]�Ľ��Ч����Inception�ܹ������ģ�鹹��Ϊ�ֽ��˲����Ķ��֧��ϣ�������Inception����[^6]���Զ���ӳ��ռ�����ԺͿ�ͨ������ԡ����֮�£�ResNeXt�ܹ����ƿ���ͨ���ۺ�ϡ�����ӣ���ͨ��ά���У�����������������ø��ḻ�ı�ʾ�����ַ�������ģ������������ǰ�ṹ����prior-structured��������ԡ����ǹ�������Щ�����SENet�汾��SE-Inception-ResNet-v2��SE-ResNeXt����1������SE-ResNeXt-50��$$32\times4d$$�������ã�����ǰ���ʵ��һ����ԭʼ��������Ƕ�Ӧ��SENet���綼ʹ����ͬ���Ż�������

&emsp;&emsp;��2�и����Ľ��˵���ڽ�SE�����뵽���ּܹ��л��������������ܸ��ơ�������SE-ResNeXt-50��`top-5`��������5.49%����������Ӧ��ResNeXt-50��5.90%��`top-5`�����ʣ��Լ������ResNeXt-101��5.57%��`top-5`�����ʣ�����ResNeXt-101ģ�ͼ����������Ĳ����ͼ��㿪��������Inception-ResNet-v2��ʵ�飬���ǲ²�����ǲü����ԵĲ��쵼�����䱨��������������ʵ�ֵĽ��֮��Ĳ�࣬��Ϊ���ǵ�ԭʼͼ���С��δ��[^38]�г��壬�������Ǵ���Խϴ��ͼ�����н϶̱߱���һ��Ϊ352���вü���$$299\times 299$$��С������SE-Inception-ResNet-v2��4.79%��`top-5`�����ʣ�����������ʵ�ֵ�Inception-ResNet-v2��5.21%��`top-5`�����ʣ�������0.42%����ԸĽ���8.1%����Ҳ����[^38]�б���Ľ����ÿ��������Ż�������ͼ5��ʾ��˵����������ѵ��������SE��Ԫ������ʼ����һ��������

![52112514647]({{ site.url }}{{ site.baseurl }}/assets/images/1521125146478.png)

&emsp;&emsp;�������ͨ����BN-Inception�ܹ�[^14]����ʵ��������SE��Ԫ�ڷǲв������ϵ�Ч�����üܹ��ڽϵ͵�ģ�͸��Ӷ����ṩ�����õ����ܡ��ȽϽ�����2��ʾ��ѵ��������ͼ6��ʾ�����ֳ���������в�����ܹ��г��ֵ�����һ������������BN-Inception 7.89%�Ĵ�������ȣ�SE-BN-Inception����˸���7.14%��`top-5`������Щʵ�����SE��Ԫ����ĸĽ���������ּܹ����ʹ�á����ң�������۶��ڲв�����ͷǲв����綼���á�

![52112517264]({{ site.url }}{{ site.baseurl }}/assets/images/1521125172642.png)

**ILSVRC 2017���ྺ���Ľ����**ILSVRC[^30]��һ����ȼ�����Ӿ���������֤����ͼ�����ģ�ͷ�չ��������ILSVRC 2017���������ѵ������֤��������ImageNet 2012���ݼ��������Լ����������δ��ǵ�10����ͼ��Ϊ�˾�����Ŀ�ģ�ʹ��`top-5`�����ʶ�������������Ŀ��������

&emsp;&emsp;SENets����������ս��Ӯ�õ�һ���Ļ��������ǵĻ�ʤģ����һСȺSENets���ɣ�ensemble�������ǲ����˱�׼�Ķ�߶ȺͶ�ü�ͼ���ںϲ��ԣ��ڲ��Լ��ϻ����2.251%��`top-5`�����ʡ���������2016���ʤ�ߣ�2.99%��`top-5`�����ʣ��Ļ�������ԸĽ���$$\sim$$25%�����ǵĸ���������֮һ�ǽ�SE��Ԫ���޸ĺ��ResNeXt[43]������һ�𹹽��ģ���¼A�ṩ����Щ�޸ĵ�ϸ�ڣ����ڱ�3�����ǽ�������ļܹ���state-of-the-art��ģ����ImageNet��֤���Ͻ����˱Ƚϡ����ǵ�ģ����ÿһ��ͼ��ʹ��$$224\times 224$$�м�ü��������̱����ȹ�һ����256��ȡ����18.68%��`top-1`�����ʺ�4.47%��`top-5`�����ʡ�Ϊ������ǰ��ģ�ͽ��й�ƽ�ıȽϣ�����Ҳ�ṩ��$$320\times320$$�����Ĳü�ͼ����������`top-1`(17.28%)��`top-5`(3.79%)�Ĵ����ʶ�����Ҳ�������͵Ĵ����ʡ�

![52112522679]({{ site.url }}{{ site.baseurl }}/assets/images/1521125226797.png)

## 6.2. ��������

&emsp;&emsp;ImageNet���ݼ��Ĵ󲿷��ɵ�������֧���ͼ����ɡ�Ϊ���ڸ��಻ͬ�ĳ������������������ģ�ͣ����ǻ���Places365-Challenge���ݼ�[48]�϶Գ���������������������ݼ�����800����ѵ��ͼ���365������36500����֤ͼ������ڷ��࣬��������������Ը��õ�����ģ�ͷ����ʹ���������������Ϊ����Ҫ��������ӵ����ݹ����Լ��Ը���̶���۱仯��³���ԡ�

&emsp;&emsp;����ʹ��ResNet-152��Ϊǿ��Ļ���������SE��Ԫ����Ч�ԣ�����ѭ[^33]�е�����׼�򡣱�4��ʾ����Ը�������ѵ��ResNet-152ģ�ͺ�SE-ResNet-152�Ľ����������ԣ�SE-ResNet-152��11.01%��`top-5`�����ʣ�ȡ���˱�ResNet-152��11.61%��`top-5`�����ʣ����͵���֤�����ʣ�֤����SE��Ԫ�����ڲ�ͬ�����ݼ��ϱ������á����SENetҲ��������ǰ��state-of-the-art��ģ��Places-365-CNN [^33]�����������������11.48%��`top-5`�����ʡ�

![52112526109]({{ site.url }}{{ site.baseurl }}/assets/images/1521125261091.png)

## 6.3. Analysis and Discussion

**ѹ���ʡ�**��ʽ��5���������ѹ���� $$r$$ ��һ����Ҫ�ĳ����������������Ǹı�ģ����SE��������ͼ���ɱ���Ϊ���о����ֹ�ϵ�����ǻ���SE-ResNet-50�ܹ�������һϵ�в�ͬ $$r$$ ֵ��ʵ�顣��5�еıȽϱ��������ܲ�û���������������Ӷ��������������������ΪSE��Ԫ������ѵ������ͨ������ԡ����Ƿ������� $$r=16$$ ʱ�ھ��Ⱥ͸��Ӷ�֮��ȡ���˺ܺõ�ƽ�⣬������ǽ����ֵ�������е�ʵ�顣

![52112529369]({{ site.url }}{{ site.baseurl }}/assets/images/1521125293694.png)

**���������á�**��ȻSE��Ԫ�Ӿ�������ʾ������Ը����������ܣ�������Ҳ���˽����ż������ƣ�self-gating excitation mechanism����ʵ��������������ġ�Ϊ�˸����������SE�����Ϊ�����������о�SE-ResNet-50ģ�͵�������������������ڲ�ͬ�鲻ͬ����µķֲ������������ԣ����Ǵ�ImageNet���ݼ��г�ȡ���ĸ��࣬��Щ������������۶����ԣ������㣬���͹����ٺ����£�ͼ7����ʾ����Щ����ʾ��ͼ�񣩡�Ȼ�����Ǵ���֤����Ϊÿ�����ȡ50��������������ÿ���׶�����SE����50�����Ȳ�����ͨ����ƽ������ֵ���������²���֮ǰ��������ͼ8�л������ǵķֲ�����Ϊ�ο�������Ҳ��������1000�����ƽ������ֲ���

![52112532204]({{ site.url }}{{ site.baseurl }}/assets/images/1521125322041.png)

&emsp;&emsp;���Ƕ�SENets��Excitation����������������㿴����**���ȣ���ͬ���ķֲ��ڽϵͲ��м�����ͬ**�����磬SE_2_3������������������׶�����ͨ����Ȩ�غܿ����ɲ�ͬ�������Ȼ����Ȥ���ǣ��ڶ����۲�����**�ڸ���Ĳ㣬ÿ��ͨ����Ȩ�ر�ø������أ���Ϊ��ͬ�����������б���ֵ���в�ͬ��ƫ�ã�** ��SE_4_6��SE_5_1���������۲�������ǰ���о����һ��[^21,46]�����Ͳ�����ͨ�����ձ飨����������𲻿�֪�������߲��������и��ߵ������ԡ���ˣ�������ʾѧϰ��SE�����������У׼�����棬������Ӧ�شٽ�������ȡ���ػ���specialisation��������Ҫ�ĳ̶ȡ������������������׶ι۲쵽һ����Щ��ͬ������SE_5_2���ֳ����򱥺�״̬����Ȥ���ƣ����д󲿷ּ���ӽ���1�����༤��ӽ���0�������м���ֵȡ1�ĵ㴦���ÿ齫��Ϊ��׼�в�顣�������ĩ��SE_5_3�У����������ڷ�����֮ǰ��ȫ�ֳػ��������Ƶ�ģʽ�����ڲ�ͬ������ϣ��߶���ֻ����΢�ı仯������ͨ�������������������������SE_5_2��SE_5_3��Ϊ�����ṩ����У׼�����ǰ��Ŀ������Ҫ����һ��������Ľ�ʵ֤�о��Ľ����һ�µģ��������ͨ��ɾ�����һ���׶ε�SE�飬����������������������٣�������ֻ��һ����ʧ��<0.1%��`top-1`�����ʣ���

![52112537872]({{ site.url }}{{ site.baseurl }}/assets/images/1521125378724.png)
![52112542795]({{ site.url }}{{ site.baseurl }}/assets/images/1521125427954.png)


# 7. ����

&emsp;&emsp;�ڱ����У����������SE�飬����һ����ӱ�ļܹ���Ԫ��ּ��ͨ��ʹ�����ܹ�ִ�ж�̬ͨ����������У׼���������ı�ʾ����������ʵ��֤����SENets����Ч�ԣ����ڶ�����ݼ���ȡ����state-of-the-art�����ܡ����⣬����Ҳ��������ʶ��һЩ��ǰ�ļܹ��ڽ�ģͨ�������������ϵľ����ԣ������������SENets��������Ҫǿ�б������������������õġ������SE�����������Ȩ�ؿ���������һЩ��������������޼�ѹ����

# A. ILSVRC 2017���ྺ������ϸ��

&emsp;&emsp;��3�е�SENet��ͨ����SE�鼯�ɵ� $$64\times4d$$ ��ResNeXt-152���޸İ汾�й����ģ�ͨ����ѭResNet-152[^9]�Ŀ�ѵ�����չԭʼResNeXt-101[^43]��������ƺ�ѵ�����죨����SE���ʹ��֮�⣩���£�

��a������ÿ��ƿ�������飬���� $$1\times1$$ ���ͨ�����������룬�������½���С�ķ�ʽ��������ļ���ɱ���

��b����һ�� $$7\times7$$ ����㱻���������� $$3\times3$$ �������ȡ����

��c������Ϊ2�� $$1\times1$$ ������²���ͶӰ���滻����Ϊ2�� $$3\times3$$ ����Ա�����Ϣ��

��d���ڷ�������֮ǰ����һ��dropout�㣨������Ϊ0.2���Է�ֹ����ϡ�

��e��ѵ���ڼ�ʹ�ñ�ǩƽ�����򻯣���[^40]�������ܵģ���

��f������󼸸�ѵ���������ڣ�����BN��Ĳ����������ᣬ��ȷ��ѵ���Ͳ���֮���һ���ԡ�

��g��ʹ��8����������64��GPU������ѵ������ʵ�ִ�batch��С��2048������ʼѧϰ��Ϊ1.0��



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