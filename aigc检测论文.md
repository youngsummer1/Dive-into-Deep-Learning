# 1
- [ ] [[1]刘安安,苏育挺,王岚君,等.AIGC视觉内容生成与溯源研究进展[J].中国图象图形学报,2024,29(06):1535-1554.](https://kns.cnki.net/kcms2/article/abstract?v=UbUZFcLhzIKQnW5Hc7ZFJRFdM-MchkSGHVgQFyDt2V9Q87ZcWxdyFVo1jzkZwrXQHNriVv60-FyyaRhYi9R3Qk_42BDzIqXjgY-hwrg5pWElyuB_2t7fsBgz37pYhXJbBj_CUD2geVVvsRAG9eTv4RdYxqDkKcyOrCke8lGPgCOLxZKBuKzdkojWmTMigI-imwGdn1DVOiDhvoFxWeVBqQ==&uniplatform=NZKPT)



##### 水印相关的生成图像溯源方法
![ZGTB202406003_04400.jpg](https://cdn.jsdelivr.net/gh/youngsummer1/joplinResource@main/20241231190706011.jpg)

* 无水印嵌入的生成图像溯源
	* 将模型生成时所遗留的生成痕迹视做指纹
	* 优势：
		* 无需额外嵌入信息，不会损害生成图像的质量
	* 局限：
		* 随着发展，图像的生成痕迹越发微弱，溯源变得困难
* 水印前置嵌入的生成图像溯源
	* 将水印嵌入到噪声、图像等输入训练数据中，利用嵌入水印的数据来训练生成模型
	* 优势：
		* 预先嵌入的信息能够在生成过程中得到有效保留，并且对生成图像质量的损害较小
	* 局限：
		* 需要对水印编解码器进行预训练或者水印嵌入训练数据进行预处理，代价较高
		* 生成图像中的水印具有单一性
* 水印后置嵌入的生成图像溯源
	* 水印嵌入是在图像生成之后进行
	* 优势：
		* 生成和嵌入之间相互独立，避免了相互干扰
		* 对生成图像质量的影响比较微弱
	* 局限：
		* 这两个阶段之间存在信息被篡改的风险
* 联合生成的生成图像溯源
	* 在图像生成过程中实现水印信息的自适应嵌入
	* 优势：
		* 无需事先构建携带水印的训练数据
		* 不存在后置水印嵌入方法存在的多阶段间逃逸攻击风险
		* 模型泛化性较强
		* 不会随着生成技术的革新或者数据集的变更而失效

# 2

- [ ] [[1]冯尊磊,娄恒瑞,贝毅君,等.人脸视频伪造检测技术进展与趋势[J].人工智能,2024,(02):63-69.DOI:10.16453/j.2096-5036.202417.](https://kns.cnki.net/kcms2/article/abstract?v=UbUZFcLhzILyFqivucoNu1DgE7F6kWfb483t99BKHvyn-vKIDPbOHwNYiGslPYLlHZO3c2UtCFFE9TPFN1zBevQxNPu2FpIX2XTlCZ_Gjws46I5412pqvjMBzQWdlbqKSKLvaQEE310o7EQX2aNvuhSkkMk2VEiiCW3mmn2E12W_LT3Q2sjmsqDg26ozQaF8qC1xN1D_4QGJdeR84pPMyQ==&uniplatform=NZKPT)

* 人脸视频伪造
	* 借助视觉信息的单模态人脸视频伪造
		* 人脸生成、人脸交换、局部伪造和人脸重现
	* 借助听觉信息的单模态人脸视频伪造
		* 通常利用TTS将源说话人的语言文本替换为伪造文本，接着利用语音转换技术实现源说话人音色的模仿，达到语音克隆的效果
	* 结合视觉与听觉层面的伪造手段
		* 在涵盖视觉与听觉伪造的同时，使用如Wav2lip等嘴部动作驱动技术实现音视频同步
	
* 视频伪造检测的主流方法
	* 基于听觉信息的防伪检测
		* 通过分析音频的微小特征，比如呼吸声、口腔移动声等生理和非言语特征，识别是否为合成音频
		* 将输入视频中提取的音频梅尔频率倒谱作为鉴别特征
	* 基于视觉信息的防伪检测
	* 基于视听结合的防伪检测
# 3
- [ ] [[1]刘明录,郑彦,韩雪,等.基于生成式因果语言模型的水印嵌入与检测[J].电信科学,2023,39(09):32-42.](https://kns.cnki.net/kcms2/article/abstract?v=UbUZFcLhzIIXu9awSHHV8oHnLX-6v3xl160nNirBiNmmg6G2nd4de4X6vQVg3vYMExCLuNm5cFbawbWyIkzTlxEt6iFFPuhJSsjIJ9syvKjeJqL-tJTav0d3MEwOA9sweL2o9jrMDZqDABTkUJhOrPlm3yQ43efnhocIeUduQ-CJv28O17asXw3EJi-PaN7idRkdCGw067ZF_LdLfU7NDQ==&uniplatform=NZKPT)
* 文本水印
* 文章生成，事中水印添加

* 因果语言模型（causal language model,CLM）
	* 通过对前文的输入进行建模，并预测下一个文字的概率分布
	* （用的是transformer的decoder结构）
##### 文本水印添加的几种策略
* 1、将文本视为一个通过文本行进行排版的图片，从而能够将图像领域成熟的水印技术直接应用于文本领域中
	* 具体
		* 嵌入背景图片，改变文字的字体、颜色等
	* 特点
		* 易被用户感知发现
		* 易清除
* 2、基于文本本身的字符串特征和水印位置计算策略，在文本的特定位置对原始文本进行直接修改
	* 具体
		* 同义词替换、句式替换等
	* 特点
		* 更改原始文本内容，可能导致文本语义的变化
	
* 3、利用文本字符串、段落序号、页码等文字版式特征，使用零宽度的Unicode字符集对上述特征进行编码形成标记字符串，并隐写在文本相对不易感知的隐蔽位置（如标题、页码、目录等）
	* 特点
		* 易清除

##### 生成式因果语言模型的文本水印技术
![DXKX202309004_03800.jpg](https://cdn.jsdelivr.net/gh/youngsummer1/joplinResource@main/20241231190706012.jpg)
（水印添加）
* 感觉与文章1中的“联合生成的生成图像溯源”差不多

* 引导上下文生成
![DXKX202309004_19300.jpg](https://cdn.jsdelivr.net/gh/youngsummer1/joplinResource@main/20241231190706013.jpg)
	* 用户prompt --> 基于prompt生成的一段无水印的先导文本（用于水印检出阶段建模）

* 水印嵌入文本生成
	![DXKX202309004_18900.jpg](https://cdn.jsdelivr.net/gh/youngsummer1/joplinResource@main/20241231190706014.jpg)
	![DXKX202309004_18901.jpg](https://cdn.jsdelivr.net/gh/youngsummer1/joplinResource@main/20241231190706015.jpg)
	* 多了根据水印选择候选子句的步骤
	（一位水印，值为0/1）
	* 水印会循环
		* 所以用户在一定程度增删改的基础上仍旧能够实现水印的检出



![DXKX202309004_08200.jpg](https://cdn.jsdelivr.net/gh/youngsummer1/joplinResource@main/20241231190706016.jpg)
（水印检测）
* 是水印嵌入过程的逆过程
* 水印特征编码抽取
![DXKX202309004_19000.jpg](https://cdn.jsdelivr.net/gh/youngsummer1/joplinResource@main/20241231190706017.jpg)
* 使用Levenstein距离进行编码的匹配度计算

# 4
- [ ] [[1]吴春生,佟晖,范晓明.生成对抗网络人脸生成及其检测技术研究[J].数字通信世界,2023,(07):28-33.](https://kns.cnki.net/kcms2/article/abstract?v=UbUZFcLhzIIGwp2HH4Y-hvRXZofTCi_o2iH4tnDzY2BKnI7yterOWLrRpAbVrhkCrrkSpzNPphjKET2odUg6Dek5TSgvPAmFQqnSFrdeuNiQ3-PHtmT-tnyf7ERdMcz_KsmINwxACt2fMUMnMRBgoM9_wLZUP4lUerwVo6BcEPFWsSt9cUhzcbr34I05IRYmFDezu-jAgysJqPIjKRadfw==&uniplatform=NZKPT)
* 主要分析以GAN为技术基础的人脸生成技术，对于相关的检测技术没有详细展开


* 基于GAN的人脸语义生成
	* 多视图姿势生成
	* 面部年龄改写
	* 人脸的属性风格生成
# 5
- [ ] [[1]王新哲,杨建,马多贺.面向AIGC对抗的人脸深度伪造检测方法研究[J].工业信息安全,2022,(11):35-45.](https://kns.cnki.net/kcms2/article/abstract?v=AhJL6SqmbxCDGAFiEXKCMhHgL8WZPaw2RmuqPNeiZ6DXHqyblZq4oWdxRUs03IaGki97VxCAL2JngoHqPbDwTTkmHHa-O-qLDapZvjbjCSSqvr1w9P-5Dl5lP2Tyrx0keO9RPSVABg337UbyN3EIHCKsCZ1qw9s9ADP9ZyCIniaAc7bCqWMy8Te2e_QL7w1g4nRaWqG6_VNVHYy0uqZcVA==&uniplatform=NZKPT)

* 人脸深度伪造
	* 自编码器法：简单，图像质量往往不高
	* 生成对抗网络法：逼真，普适性强，花费高
	* 神经网络法：资源开销大，训练周期久，且具有很强的针对性

* 人脸深度伪造检测
	* 图像检测
		* 深度学习检测法：
		* GAN指纹检测法：用GAN本身所具有的一部分特点来检测
		* 高压缩比图像检测方法：对压缩图像进行数据增强，缓解压缩导致的关键噪声损失
	* 视频检测
		* 帧内检测法：将视频划分为视频帧，检测单视频帧中存在的伪造特征
		* 帧间检测法：利用视频中帧间的时间、空间等条件的关系，通过一个时间段内时空的有序性和一致性来检测视频是否受到伪造

# 6
- [ ] [Optimizing AIGC Image Detection: Strategies in Data Augmentation and Model Architecture](https://dl.acm.org/doi/abs/10.1145/3664647.3689002)
* “马栏山杯“国际音视频算法大赛第一名的论文
	* 使用 NPR + Resnet50
	* 在未知的伪造技术上也能表现良好
* 数据增强
	* 小于224像素的图像：中心裁剪。
	* 224到512像素之间的图像：随机裁剪。
	* 大于512像素的图像：先缩放，然后随机裁剪。

* "neighboring pixel relations" (NPR)
	* 方法提出
		* [Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection.](https://arxiv.org/abs/2312.10461)
	* 作用
		* 用于捕捉生成模型中上采样操作引入的局部结构伪影，可增强模型检测伪造图像的能力

# 7
- [ ] [Exposing AI-generated Videos: A Benchmark Dataset and a Local-and-Global Temporal Defect Based Detection Method ](https://arxiv.org/abs/2405.04133)

* 提出了一个包含多种视频生成算法的AIGC**视频数据集**
	* 四种扩散模型
	* 模拟有损操作
* 提出了一个基于局部运动信息和全局外观变化的**检测框架**
	* 局部运动 + 全局外观 + 通道注意力特征融合
* 通过实验评估了不同方法的泛化能力和鲁棒性
	* 评估方法：CNNSpot 、 NPR、传统视频分类网络、本文的方法



# 8
- [ ] [Seeing is not always believing: Benchmarking Human and Model Perception of AI-Generated Images](https://proceedings.neurips.cc/paper_files/paper/2023/hash/505df5ea30f630661074145149274af0-Abstract-Datasets_and_Benchmarks.html)

* [仓库](https://github.com/Inf-imagine/Sentry)

* 构建Fake2M数据集
![fb0ee146131a052f3e9e5a6bbd8b5e79.png](https://cdn.jsdelivr.net/gh/youngsummer1/joplinResource@main/20241231190706018.png)
* 设置人类感知评估（HPBench）基准
	* 评估人类在区分AI生成图像和真实照片方面的能力
* 设计模型感知评估（MPBench）基准
	* 评估模型在检测最新生成模型生成的虚假图像方面的能力
	
# 9
- [ ] [FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models](https://arxiv.org/abs/2410.02761)

* [仓库](https://github.com/zhipeixu/FakeShield)
	* 论文未见刊，代码还没发布

* 多模态数据集（MMTD-Set）
	* 包含伪造图像、掩码和（哪部分是伪造的）详细描述
	* 用于训练FakeShield
* FakeShield框架
	![f7f036d0a18061ce469594db7c5c10d7.png](https://cdn.jsdelivr.net/gh/youngsummer1/joplinResource@main/20241231190706019.png)
	* 能够检测和定位多种伪造技术，并提供（哪部分是伪造的）详细的解释
	* 包括
		* 大语言模型：用于生成解释
		* DTE-FDM：通过域标签区分不同类型的伪造数据（PS/DeepFake/AIGC）
		* MFLM：结合视觉和文本特征，精确定位伪造区域

# 10 
- [ ] [Research about the Ability of LLM in the Tamper-Detection Area](https://arxiv.org/abs/2401.13504)
* 使用GPT-4、LLaVA、Bard、ERNIE Bot4、Tongyi Qianwen五种LLM 进行 AIGC检测 和 篡改检测
	* 结论是效果很差？
	* 该论文未见刊，感觉不太对

# 11
- [ ] [Towards Adversarially Robust AI-Generated Image Detection](https://www.scitepress.org/Papers/2023/128163/128163.pdf)
* 构建对抗数据集
* 创建一个能够在对抗攻击和自然变换下实现更高鲁棒性的模型
	* 结合对抗性训练和数据增强技术，显著增强了面对对抗攻击时的鲁棒性 ，同时保持了基础分类器的准确性

* 文中将 对分类器的对抗攻击被视为优化问题
	* 即，使模型的损失函数最大化，同时保证扰动尽可能小
	* $\max_{\delta} L(f(\mathbf{x} + \delta; \theta), y) \quad \text{subject to} \quad \|\delta\|_p \leq \epsilon$
	* 常见优化方法
		* 快速梯度符号法（FGSM）
		* 投影梯度下降法（PGD）
		* Auto-PGD

# 12
- [ ] [A Survey on Detection of LLMs-Generated Content](https://arxiv.org/abs/2310.15654)
* LLMs生成内容检测的全面综述


* LLMs 生成内容的检测方法
	* 基于训练的分类器：通过微调预训练语言模型来区分
	* 零样本检测器：利用LLMs的固有特性进行检测
	* 水印：在生成文本中嵌入可检测的标识信息

# 13
- [ ] [SoK: On the Role and Future of AIGC Watermarking in the Era of Gen-AI](https://arxiv.org/abs/2411.11478)
* AIGC水印生成、属性、功能、安全性、治理等方面的全面综述

* 不同模态生成水印
	* 文本
		* 法1：修改每个标记的logit值
		* 法2：略微修改大语言模型（LLM）的采样方法
		* 法3：使用带水印的文本数据微调模型
	* 图像
		* 法1：图像生成过程中向扩散模型的噪声中注入特定模式以嵌入水印
		* 法2：在AI生成管道中添加额外的水印模块或微调生成模型参数
	* 音频
		* 法1：添加额外的水印模块
		* 法2：使用预水印音频作为训练数据
	* 视频
		* 法1：主流的视频生成技术通过逐帧生成图像来生成视频，因此应用于图像的水印技术也可以用于嵌入视频水印
	* 跨模态
		* 法1：通过跨模态特征对齐来增强水印的鲁棒性或减少水印对生成内容的影响
# 
- [ ] [Fake-GPT: Detecting Fake Image via Large Language Model](https://link.springer.com/chapter/10.1007/978-981-97-8685-5_9)
	* 学校买了springer，但PRCV这部分的看不了

- [ ] [Advancing Video Quality Assessment for AIGC](https://arxiv.org/abs/2409.14888)




