# GALE: MEMORY-EFFICIENT GLOBAL APPROXIMATE AND LOCAL EXACT FEATURES

Alberto Ancilotto

Elisabetta Farella

Fondazione Bruno Kessler (FBK), Trento, Italy

## ABSTRACT

Embedded devices typically lack the resources of GPUequipped machines, and existing inference methods suffer from either high computational overhead (patch-based) or accuracy loss (approximation-based). We propose GaLe, a memory-efficient technique that enables the deployment of pretrained networks on constrained devices without retraining. GaLe partitions feature maps into two components: a local exact (L ) representation that preserves fine details and a global approximate $( G _ { A } )$ representation that retains long-range dependencies. Unlike standard tiling, GaLe supports global operations and attention mechanisms found in hybrid CNN-transformer models. Validated on ImageNet, our method matches exact-inference performance while achieving up to 65% speedup and 90% RAM reduction on a Cortex-M33 compared to patch-based inference. We further demonstrate GaLe’s versatility across classification, detection, and generation tasks, highlighting its potential as a foundation for resource-efficient architecture design.

Index Terms— Embedded Systems, TinyML, Memory-Efficient Inference, Microcontrollers, RAM Reduction

## 1. INTRODUCTION

Deep neural networks have grown increasingly resourcedemanding, with modern attention-based architectures pushing the limits of available hardware. This is particularly true in embedded and TinyML systems, where the benefits of edge processing (in terms of privacy, latency, and energy efficiency) clash with strict hardware limitations. Typical MCU constraints include static memory (FLASH) restricting parameters to under 2MB, compute capacities orders of magnitude lower than GPUs, and most critically, dynamic memory (RAM) often below 1MB. RAM capacity is the primary bottleneck for modern building blocks like inverted residuals [1, 2, 3] and attention mechanisms [4, 5]. To address this challenge, we propose a novel memory-efficient inference strategy enabling the deployment of pre-trained networks on resource-constrained devices without retraining, requiring only a small calibration set. Our method relies onfeature map partitioning, where layer outputs are decomposed into two complementary representations: (i) a local exact $( L _ { E } )$ component for fine-grained details and (ii) a global approximate $( G _ { A } )$ component for long-range dependencies. Unlike previous reduction methods [6, 7], this approach supports global receptive fields and attention blocks, minimizing accuracy loss while potentially guiding memory-aware architecture search. It also remains compatible with standard deployment runtimes such as ONNX, TFLite, and STM32Cube.Ai. Our contributions are two-fold. First, we introduce a hardwareaware partial-patch-based inference method that optimizes memory usage via contiguous RAM slices. We reduce computational overhead by estimating the minimal patch overlap required to achieve target precision and by leveraging $G _ { A }$ features to compensate for missing data. Second, we extend this approach to hybrid CNN-transformer architectures. We derive an attention formulation based on $G _ { A }$ and $L _ { E }$ features that enables sequential block processing with lower overhead than standard memory-efficient attention implementations [8, 9], and can operate alongside or replace them in memory-constrained settings.

![](images/f8a9ed19c6d3637f19e1c6fd3c0caa6528fe886624a142c099bbfdb3f2b4b108.jpg)  
Fig. 1. Graphical representation of local exact and global approximate decomposition for a convolutional layer

## 2. RELATED WORKS

Training-based approaches optimize network structure to reduce computational complexity. Foundational efficient architectures, such as the MobileNet family [2, 3, 10] and Xception [11], leverage depth-wise separable convolutions and inverted residual blocks. These concepts evolved into hardware-aware Neural Architecture Search (NAS) frameworks targeting MCUs, exemplified by MCUNet [12, 6, 13], and scaling strategies like EfficientNet [14], PhiNets [15], and XiNets [16], which jointly optimize memory, parameters, and latency. Recent efforts extend these principles to efficient transformer designs [17, 18] and fine-tuning based pruning strategies [19, 20, 21]. Training-free approaches reduce the memory usage of pre-trained models at runtime. The simplest method, reducing input resolution [14], lowers RAM usage quadratically but degrades fine-grained details. Full patchbased inference [22] preserves resolution but sacrifices global context. Partial patching, as adopted in TinyEngine [6] and other embedded engines [23], achieves mathematically exact results by tiling only the early layers; however, it is incompatible with global receptive field operators such as Squeezeand-Excitation [24] or attention. For transformers, runtime memory reduction often relies on token merging [25, 26] or exact algorithms like FlashAttention [9, 8], although the latter can incur significant computational overhead when aggressive memory reduction is required, limiting their applicability to embedded platforms.

## 3. GALE FOR CONVOLUTIONAL NETWORKS

Classical memory-reduction approaches like partial patchbased inference (PPBI) suffer from high computational overhead [6] and incompatibility with global operations like Squeeze-and-Excite (SE) modules [24] or attention. To address these limitations, we propose a compute-efficient feature map slicing approach that approximates outputs using two complementary representations (Figure 1): a Local Exact $( L _ { E } )$ component, preserving full-resolution details for a spatial sub-region, and a Global Approximate $( G _ { A } )$ component, capturing low-resolution context over the entire map. Unlike traditional patching, our method supports hybrid vision transformers and global receptive field operators while drastically reducing computational overhead (from over 100% to less than 20%) through a hardware-aware tensor layout that optimizes cache locality.

## 3.1. Computing Local Exact feature maps

In partial patch-based inference, the input is divided into N smaller, partially overlapping areas, with spatial dimensions $P _ { H } \times P _ { W }$ that are processed independently - reducing the size of intermediate activation tensors. To ensure accurate outputs at patch boundaries, patches must overlap sufficiently so that the receptive field of the layer at each output position is fully covered by valid input data. The required overlap O between patches is then equal to the receptive field of the deepest layer considered for patch-based execution: $\mathcal { O } _ { p p b i } = \mathcal { R } _ { 0 } ( L )$ , where L is the layer index and $\mathcal { R } _ { 0 }$ maps each layer to its receptive field on the input (layer 0). For example, considering a CNN composed only of convolutional layers with kernel size $k _ { i } ,$ stride $s _ { i }$ for layer i, receptive field size $\mathcal { R } _ { 0 } ( L )$ on the input layer [27] can be computed as:

![](images/49fd915037bdf2d61368420d2f0c5da2da734933fd3b06fa4cbfd3229ae99bed.jpg)  
Fig. 2. Proposed slicing vs classical PPBI. Our learned padding results in significantly lower overhead than padding to the full receptive field. Moreover, for NHWC tensor inference, the proposed method relies on data stored in a single continuous memory area, thereby optimizing cache usage.

$$
\mathcal { R } _ { 0 } ( L ) = \sum _ { l = 1 } ^ { L } \left( ( k _ { l } - 1 ) \prod _ { i = 1 } ^ { l - 1 } s _ { i } \right) + 1\tag{1}
$$

However, the receptive field size from Equation 1 is only smaller than the input size for networks composed exclusively of local operations such as convolutional blocks and maxpooling layers. As such, PPBI can only reduce RAM requirements of a limited number of existing networks. In fact, this approach is not directly applicable to architectures that aggregate information across the entire input or operations with very large receptive fields, such as spatial pyramid pooling. Our approach generalizes partial patch-based inference to all network architectures by reducing the required overlap so that $\mathcal { O } _ { G a L e } \ \leq \ \mathcal { O } _ { p p b }$ , allowing for a user-controllable approximation error ϵ between the original and reconstructed feature maps. We rely on a calibration pass to determine the appropriate overlap $\mathcal { O } _ { i }$ for each network layer patch. During calibration, for each layer i, we iteratively increase the overlap $\mathcal { O } _ { i }$ by $\mathscr { R } _ { i } ( i + 1 )$ if layer i + 1 is a convolutional layer, or $s _ { i + 1 }$ otherwise. This process continues until either the MSE between the reconstructed output tensor and the original output tensor falls below ϵ or the overlap reaches the maximum allowed based on memory constraints. In this case, we increase the number N of subdivisions for the current layer. Thanks to this iterative approximation strategy, the proposed method can approximate arbitrary network operators that may not be compatible with PPBI. When applied instead to architectures that can already be patched using traditional methods, GaLe can achieve significantly lower computational overhead, even for very small values of ϵ. The proposed method is based on dividing the input tensor into horizontal slices composed of multiple full rows $( P _ { W } = W )$ instead of square patches, as shown in Figure 2. When working with tensors in NHWC format (the default for the majority of embedded runtimes [28, 29]), slices align naturally with memory layout as shown in Figure 2. Each slice consists of values stored contiguously in memory, reducing cache misses and improving cache efficiency, resulting in lower latency, as shown in Section 5.

![](images/0eaacfb894d5b475fd5a9b9de6c221a645cfded35108de713f792f8796ae0e54.jpg)  
Fig. 3. (Top) Example of partial patch-based inference on 3 layers with 4 patches resulting in 12 kernel loading operations. (Bottom) Slice-based inference on 3 layers with 4 patches, resulting in 6 kernel loading operations

## 3.2. Optimizing memory transfers via Adaptive Slicing

Standard PPBI enforces a fixed patch count across all layers, ignoring the natural reduction in spatial dimensions typical of convolutional backbones. This creates unnecessary fragmentation in deeper layers, leading to redundant computation due to patch overlaps and excessive memory traffic. To address this, GaLe employs adaptive slicing. During calibration, we compute the minimal number of patches required for each block to strictly satisfy the peak memory constraint. As feature map resolution decreases, GaLe progressively reduces the patch count, merging slices whenever possible. This dynamic strategy minimizes the total overlap area, directly reducing redundant operations. Furthermore, by processing larger contiguous feature map sections, we drastically reduce the frequency of weight reloading from Flash to RAM (Figure 3). Consequently, GaLe achieves significantly lower latency than fixed-pattern inference (Figure 5), particularly on MCUs where Flash bandwidth is a critical bottleneck.

## 3.3. $G _ { A }$ feature maps maintain global information

Unlike standard PPBI, which relies on a large patch overlap to maintain mathematical equivalence, our $L _ { E }$ patches are spatially isolated approximations that inherently lack global context. To address this, we introduce a Global Approximate $( G _ { A } )$ branch. The $G _ { A }$ input is generated by downsampling the original feature map by a factor N, which is determined during calibration to ensuring the entire reduced tensor fits within the target memory budget. While various compression functions are possible, we find that simple bilinear downsampling offers the best trade-off between efficiency and representational quality. To construct the final output, we upsample the processed $G _ { A }$ result and merge it with the fine-grained $L _ { E }$ output using linear interpolation. Although computing the $G _ { A }$ branch introduces a fixed overhead, it remains significantly faster than the overlapping required by PPBI (Figure 5), particularly as the reduction factor increases. Table 1 details the impact of combining $L _ { E }$ and $G _ { A }$ components on MobileNetV4 performance.

<table><tr><td>Feature Maps</td><td>Accuracy (%)</td></tr><tr><td>Ga + Le</td><td>80.66</td></tr><tr><td>Le only</td><td>80.34</td></tr><tr><td>Ga only</td><td>79.16</td></tr></table>

Table 1. Comparison of feature map configurations on accuracy.

## 4. GALE FOR HYBRID NETWORKS

We extend the approach to modern attention-based and hybrid architectures, such as EfficientViT [4], FastViT [30] and MobileNetV4 [10]. As in the convolutional case, our method can offer superior performance than approximation-based techniques, with a fraction of the overhead required by math exact approaches based on memory-efficient attention (e.g. FlashAttention [31]), which recompute parts of the attention map multiple times during the forward pass. Our technique proves particularly effective for hybrid architectures. In fact, unlike existing approaches, GaLe can be used to reduce the memory of both the convolutional and attention-based parts in a unified framework. By analyzing RAM usage patterns in these models, we tailor GaLe to enable deployment on ultralow-resource devices such as MCUs, typically unsupported by existing methods.

## 4.1. Attention via $L _ { E }$ and $G _ { A }$ Approximation

Standard scaled dot-product attention computes an output O from query, key, and value matrices $Q , K , { \bar { V } } \in \mathbb { R } ^ { N \times d }$ as:

$$
\mathbf { O } = \mathrm { S o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d } } \right) V .\tag{2}
$$

This operation requires $\mathcal { O } ( N ^ { 2 } )$ memory to store the attention map $\mathbf { A } = Q K ^ { T }$ . To mitigate this bottleneck without the overhead of recomputation-based methods [8], we propose an approximation scheme based on structured sparsity and lowrank components. We model the attention map A as a tiling of $b \times b$ blocks $\mathbf { A } _ { i j }$ , where each block is approximated as a weighted combination of a local sparse component and a global low-rank component:

$$
\mathbf { A } _ { i j } \approx \alpha \mathbf { L } _ { i j } + ( 1 - \alpha ) \mathbf { G } _ { i j } .\tag{3}
$$

Here, ${ \bf L } _ { i j } = v _ { i j } { \bf I } _ { b }$ represents the fine-grained local correlation (where $\mathbf { I } _ { b }$ is the identity matrix, preserving only diagonal dependencies within the block), and $\mathbf { G } _ { i j } = c _ { i j } \mathbf { J } _ { b }$ represents the coarse global context (where $\mathbf { J } _ { b }$ is the all-ones matrix of size $b \times b )$ . Transforming this structural approximation into a computable operation, we decouple the attention mechanism into b independent “strided” heads (Local Exact) and one “downsampled” head (Global Approximate). This allows us to approximate the full operation on N tokens as a weighted sum of operations on smaller subsets of size $N / b$

## 4.1.1. Algorithm Formulation

Let the input sequences be partitioned into b interleaved subsets (strides). We define the Local Exact attention term, which captures dependencies between tokens at the same phase $k \in$ $\{ 1 , \ldots , b \}$ , and the Global Approximate term, which captures dependencies on downsampled features:

$$
\begin{array} { r } { \widetilde { \mathrm { A t t n } } ( Q , K , V ) = \alpha \displaystyle \sum _ { k = 1 } ^ { b } \mathcal { F } \left( Q ^ { ( k ) } , K ^ { ( k ) } , V ^ { ( k ) } \right) } \\ { + \left( 1 - \alpha \right) \mathcal { F } \left( Q _ { d s } , K _ { d s } , V _ { d s } \right) , } \end{array}\tag{4}
$$

where:

$\mathcal F ( \cdot )$ denotes the standard Attention operation (Eq. 2);

$Q ^ { ( k ) } , K ^ { ( k ) } , V ^ { ( k ) } \in \mathbb { R } ^ { \frac { N } { b } \times d }$ are the k-th strided slices of the inputs $( \mathrm { e . g . , } Q ^ { ( k ) }$ contains rows $k , k { + } b , k { + } 2 b . . . ) ;$

$Q _ { d s } , K _ { d s } , V _ { d s } \in \mathbb { R } ^ { \frac { N } { b } \times d }$ are the inputs downsampled by factor b to capture global context.

## 4.1.2. Complexity Analysis

By decomposing the problem, we avoid materializing the full $N \times N$ attention matrix. Instead, we compute b + 1 smaller attention maps, each of size $\begin{array} { r } { \frac { N } { b } \times \frac { N } { b } } \end{array}$ . This reduces the peak dynamic memory requirement for the attention map from $\mathcal { O } ( N ^ { 2 } )$ to $\mathcal { O } ( ( N / b ) ^ { 2 } )$ . Furthermore, because the operations in Eq. 4 are independent, they can be executed sequentially, ensuring that the peak RAM usage remains bounded by the requirements of a single subset.

## 4.1.3. Hybrid Architectures

State-of-the-art efficient architectures often adopt hybrid designs, integrating convolutional layers for local feature extraction in early stages and attention mechanisms for global context in later stages [10, 30, 4]. As shown in Figure 4, this structural heterogeneity creates a bi-modal memory profile, where peak RAM usage is governed by two distinct bottlenecks: Early Convolutional layers require memory to store the high-resolution activation maps, while attention operations generate peaks when the $\mathcal { O } ( \mathcal { H } \mathcal { W } ^ { \in } )$ attention map is computed.

![](images/5d003a94d8f5db38579e7f482eae1a58e246ca1706b507bf6e8dd05090d0f3ec.jpg)  
Fig. 4. RAM requirements for EfficientVit-B2 [4] hybrid architecture, and savings achieved through the proposed approach. The contribution of conv blocks and attention to peak RAM usage are clearly seen.

## 5. RESULTS

We benchmarked GaLe on different hardware platforms, ranging from a Raspberry Pi4 and STM32U585 to the resource-constrained STM32H743 and GAP9. Comparing our approach against resolution scaling and patch-based inference (FPBI/PPBI), we observe that GaLe consistently achieves superior memory reduction with minimal accuracy loss (< 1%) and significantly lower overhead (Figure 5, Table 2). Our results highlight specific architectural limitations of existing methods. While PPBI achieves an 88% RAM reduction on MobileNetV2, it incurs an 80% computational overhead; GaLe improves this reduction to 92% with only 18% overhead. Furthermore, GaLe succeeds where PPBI is structurally inapplicable: it handles the global dependencies of MobileNetV3’s Squeeze-and-Excitation blocks via $G _ { A }$ features, whereas standard tiling fails. Finally, for hybrid models like MobileNetV4 and FastViT, GaLe effectively compresses the entire model by addressing both convolutional and attention layers, where PPBI and Token Merging (ToMe) individually prove insufficient. Additional results for diffusion models are available in the appendix.

## 5.1. Case study: Object Detection

We evaluated GaLe on object detection, deploying quantized RT-DETR-L (requiring 9.8MB RAM) and YOLOv11n (4.2MB RAM) onto constrained targets: a high-performance 2.5MB platform (e.g., GAP9, STM32H7) and a low-power 512KB platform (e.g., STM32L4). Figure 6 illustrates the trade-offs. Resolution scaling degrades performance for small objects, while FPBI degrades performance for large objects due to spatial fragmentation. PPBI preserves accuracy but incurs in huge computational overhead. In contrast, GaLe effectively balances efficiency and accuracy, achieving negligible mAP reduction by maintaining both local detail and global context via the $L _ { E } / G _ { A }$ decomposition.

<table><tr><td colspan="2">Network</td><td>RAM [KB]</td><td>Top1%</td><td>Overhead%</td></tr><tr><td rowspan="5">MobileNetV2 a110</td><td></td><td>6110</td><td>76.44</td><td></td></tr><tr><td>Res</td><td>1527</td><td>64.16</td><td>-74.7</td></tr><tr><td>PBI</td><td>1588</td><td>66.72</td><td>+2.1</td></tr><tr><td>PPBI</td><td>763</td><td>76.44</td><td>+78.3</td></tr><tr><td>GaLe</td><td>476</td><td>76.42</td><td>+18.7</td></tr><tr><td rowspan="5">MobileNetV3 a100</td><td></td><td>4069</td><td>76.94</td><td></td></tr><tr><td>Res</td><td>1017</td><td>62.50</td><td>-73.9</td></tr><tr><td>FPBI</td><td>1017</td><td>66.04</td><td>+7.3</td></tr><tr><td>PPBI</td><td>1</td><td>1</td><td>1</td></tr><tr><td>GaLe</td><td>317</td><td>76.56</td><td>+16.3</td></tr><tr><td rowspan="5">ResNet50</td><td></td><td>9010</td><td>80.46</td><td></td></tr><tr><td>Res</td><td>1216</td><td>66.94</td><td>-85.8</td></tr><tr><td>FPBI</td><td>1261</td><td>71.96</td><td>+5.1</td></tr><tr><td>PPBI</td><td>1171</td><td>80.46</td><td>+167.9</td></tr><tr><td>GaLe</td><td>847</td><td>79.52</td><td>+40.9</td></tr></table>

<table><tr><td colspan="2">Network</td><td>RAM [KB]</td><td>Top1%</td><td>Overhead%</td></tr><tr><td rowspan="6">MobileNetV4 HM</td><td></td><td>6314</td><td>81.80</td><td></td></tr><tr><td>Res</td><td>1577</td><td>68.21</td><td>-74.9</td></tr><tr><td>FPBI</td><td>1623</td><td>69.08</td><td>+1.6</td></tr><tr><td>ToMe</td><td>6314</td><td>81.60</td><td>-5.1</td></tr><tr><td>PPBI</td><td>4922</td><td>81.81</td><td>+26.4</td></tr><tr><td>GaLe</td><td>884</td><td>80.66</td><td>+14.6</td></tr><tr><td rowspan="6">FastViT</td><td>=</td><td>8342</td><td>81.96</td><td>=</td></tr><tr><td>Res</td><td>1042</td><td>65.70</td><td>-79.7</td></tr><tr><td>FPBI</td><td>2085</td><td>74.60</td><td>+5.4</td></tr><tr><td>ToMe</td><td>8342</td><td>81.80</td><td>-9.2</td></tr><tr><td>PPBI</td><td>1668</td><td>81.96</td><td>+43.4</td></tr><tr><td>GaLe</td><td>992</td><td>81.10</td><td>+5.9</td></tr><tr><td rowspan="6">EfficientViT</td><td>=</td><td>9732</td><td>82.46</td><td></td></tr><tr><td>Res</td><td>1430</td><td>68.72</td><td>-83.4</td></tr><tr><td>PPBI</td><td>7687</td><td>82.46</td><td>+31.4</td></tr><tr><td>ToMe</td><td>9732</td><td>82.26</td><td>-6.1</td></tr><tr><td>FPBI</td><td>1625</td><td>72.21</td><td>+4.1</td></tr><tr><td>GaLe</td><td>1294</td><td>81.48</td><td>+15.6</td></tr></table>

Table 2. Comparison of GaLe against state-of-the-art methods on ImageNet. GaLe achieves the best trade-off between RAM reduction and computational overhead across all architectures.

![](images/9545244d5817a7e3d0e2f3b39224a491697b761bd94a3cedfa51bdc0046c9d0d.jpg)  
Fig. 5. Latency vs RAM for MobileNetV2 (256px, Cortex-M33). GaLe enables significant RAM savings at lower latencies than PPBI, achieving a 65% speedup for a 90% RAM reduction compared to tiling.

## 6. CONCLUSION

This work introduced GaLe, a feature map partitioning strategy enabling the deployment of pretrained deep learning models on memory-constrained devices without retraining. By decomposing activations into local exact and global approximate components, GaLe resolves the limitations of existing tiling methods, ensuring compatibility with global operators and attention mechanisms. Validation on ImageNet classification demonstrates similar performance with exact inference methods while delivering up to 90% memory reduction and substantial speedups over patch-based methods. Further experiments on object detection and diffusion highlight GaLe’s versatility, positioning it as a foundational technique for efficient edge AI and a potential driver for future memory-aware neural architecture search.

![](images/a00aa1c9055aa0e811068a652904d9b8e9dc8ee639efa968fb865329bd5817cc.jpg)  
Fig. 6. Performance comparison for RT-DETR-L (target < 2.5MB) and YOLOv11n (target < 512KB). GaLe maintains mAP comparable to exact methods with minimal overhead.

## References

[1] Howard, Andrew G et al, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.

[2] Sandler, Mark, et al, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018.

[3] Andrew Howard et al., “Searching for mobilenetv3,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 1314–1324.

[4] Han Cai et al., “Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17302–17313.

[5] Alexey Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[6] Ji Lin, Wei-Ming Chen, Han Cai, Chuang Gan, and Song Han, “Mcunetv2: Memory-efficient patch-based inference for tiny deep learning,” in Neural Information Processing Systems, 2021.

[7] Shishir Patil et al., “Poet: Training neural networks on tiny devices with integrated rematerialization and paging,” in International Conference on Machine Learning. PMLR, 2022.

[8] Markus N Rabe and Charles Staats, “Self-attention does not need o(n<sup>2</sup>) memory,” arXiv preprint arXiv:2112.05682, 2021.

[9] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re, “Flashattention: Fast and memory-efficient exact attention´ with io-awareness,” Advances in neural information processing systems, vol. 35, pp. 16344–16359, 2022.

[10] Danfeng Qin et al., “Mobilenetv4 - universal models for the mobile ecosystem,” European Conference on Computer Vision (ECCV), vol. abs/2404.10518, 2024.

[11] Franc¸ois Chollet, “Xception: Deep learning with depthwise separable convolutions,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1251–1258.

[12] Ji Lin, Wei-Ming Chen, Yujun Lin, John Cohn, Chuang Gan, and Song Han, “Mcunet: Tiny deep learning on iot devices,” arXiv preprint arXiv:2007.10319, 2020.

[13] Ji Lin et al., “On-device training under 256kb memory,” Advances in Neural Information Processing Systems, vol. 35, pp. 22941–22954, 2022.

[14] Mingxing Tan and Quoc Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in International Conference on Machine Learning. PMLR, 2019.

[15] Francesco Paissan, Alberto Ancilotto, and Elisabetta Farella, “Phinets: A scalable backbone for low-power ai at the edge,” ACM Trans. Embed. Comput. Syst., feb 2022.

[16] Alberto Ancilotto, Francesco Paissan, and Elisabetta Farella, “Xinet: Efficient neural networks for tinyml,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 16968–16977.

[17] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10012–10022.

[18] Sachin Mehta and Mohammad Rastegari, “Separable selfattention for mobile vision transformers,” Transactions on Machine Learning Research, 2023.

[19] Siyuan Wei et al., “Joint token pruning and squeezing towards more aggressive compression of vision transformers,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 2092–2101.

[20] Sehoon Kim et al., “Learned token pruning for transformers,” in Proceedings ofthe 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 784–794.

[21] Yunke Wang, Bo Du, Wenyuan Wang, and Chang Xu, “Multitailed vision transformer for efficient inference,” Neural Networks, vol. 174, pp. 106235, 2024.

[22] Fatih Akyon et al., “Slicing aided hyper inference and finetuning for small object detection,” in 2022 IEEE international conference on image processing (ICIP). IEEE, 2022, pp. 966– 970.

[23] GreenWaves Technologies, “Automated design intelligence with gapflow,” https:// greenwaves-technologies.com/.

[24] Jie Hu, Li Shen, and Gang Sun, “Squeeze-and-excitation networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 7132–7141.

[25] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman, “Token merging: Your vit but faster,” arXiv preprint arXiv:2210.09461, 2022.

[26] Minchul Kim et al., “Token fusion: Bridging the gap between token pruning and token merging,” in Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, 2024, pp. 1383–1392.

[27] Andre Araujo, Wade Norris, and Jack Sim, “Computing recep- ´ tive fields of convolutional neural networks,” Distill, vol. 4, no. 11, pp. e21, 2019.

[28] google, “tflite-micro,” https://github.com/ tensorflow/tflite-micro.

[29] ST Microelectronics, “X-cube-ai,” https://www.st. com/en/embedded-software/x-cube-ai.html.

[30] Pavan Kumar Anasosalu Vasu et al., “Fastvit: A fast hybrid vision transformer using structural reparameterization,” in ICCV, 2023.

[31] Tri Dao, “Flashattention-2: Faster attention with better parallelism and work partitioning,” arXiv preprint arXiv:2307.08691, 2023.

[32] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach, “Adversarial diffusion distillation,” in European Conference on Computer Vision. Springer, 2024, pp. 87–103.

[33] Justin Johnson, Alexandre Alahi, and Li Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14. Springer, 2016, pp. 694–711.

## A. APPENDIX

## A.1. Case study: diffusion models

We present an additional application of GaLe in the context of diffusion models as a replacement for self-attention layers to reduce memory consumption while avoiding the computational overhead associated with memory-efficient attention techniques [8]. To evaluate the effectiveness of our approach, we benchmark SD-Turbo [32] comparing GaLe against memory-efficient attention and token merging (ToMe) for attention memory reduction. Additionally, we introduce a hybrid variant in which token merging is applied together with GaLe, computing $G _ { A }$ through ToMe, while the $L _ { E }$ features remain unchanged.

![](images/428088909142722cc5e77553fc2d1f31253e03f04f29324768a72ae83d8cf033.jpg)  
Fig. 7. Our approach allows for significant RAM savings with minimal quality loss when applied to SD-turbo [32]. Token merging strategies can be used for computing the Ga feature maps, increasing performance.

As shown in Figure 8, GaLe achieves a substantially greater reduction in memory usage compared to ToMe alone, preserving the fidelity of generated images, as measured by Frechet Inception Distance (FID) and Perceptual Simi-´ larity score (LPIPS [33]) relative to the baseline network, while proving faster than standard memory-efficient attention. When GaLe is combined with ToMe, extreme memory compression factors can be achieved, which would otherwise be unattainable using ToMe alone, while maintaining a low

FID vs RAM Usage for Different Methods  
![](images/6fe16f3e26efd07858162cf7ecdbc0d4933d2f7c2988e89f4e3b0a3922a371f4.jpg)

Time vs RAM Usage for Different Methods  
![](images/d56f47cbbf6df3bf270d766eec74f07958e484a0d48271f39489bf7b8b4ab889.jpg)  
Fig. 8. (Top) Time required for generating five $5 1 2 \times 5 1 2$ images on a mobile RTX3000 Ada. GaLe allows for significantly reduced RAM usage without the computational overhead of using memory-efficient attention. (Bottom) Compared to token merging approaches, GaLe allows for significantly reduced RAM usage. Token merging can be used for computing the Ga feature map, allowing for lower FID.

FID score. Figure 9 shows examples of images generated from SD-Turbo using GaLe, with the achieved RAM reductions and perceptual similarity scores. Table 5 shows the performance obtained when generating images at different resolutions. We can see how the proposed approach offers high performance even at extreme (25×) compression ratios both for lower (512×512) and higher (1024×1024) generation resolutions.

## A.2. Detailed platform performance

Table 3 shows the real-world speedup achieved using GaLe compared to traditional partial-patch based inference. Higher speedups are obtained for simpler MCUs (eg Cortex M33), while more powerful devices with tiered caches (e.g. a Raspberry Pi4) do not benefit as much from the memory-aware slice layout.

## A.3. GaLe calibration pass

The following pseudocode shows the calibration pass logic.   
Slice count is increased until the target performance are met.

![](images/d06a500f880f1b8da0e398e94909d39f78fc01028fc8b09ba065d1619c3d4494.jpg)  
Fig. 9. Example images using sd-turbo with GaLe.

<table><tr><td></td><td>N</td><td>RAM [KB]</td><td>H743</td><td>GAP9</td><td>M33</td><td>RPi4</td></tr><tr><td>Base</td><td>1</td><td>1016</td><td>432</td><td>37.08</td><td>3132</td><td>一</td></tr><tr><td>PPBI</td><td>4</td><td>295</td><td>532</td><td>40.8</td><td>3632</td><td>62.9</td></tr><tr><td>PPBI</td><td>8</td><td>178</td><td>808</td><td>60.4</td><td>6512</td><td>68.3</td></tr><tr><td>PPBI</td><td>16</td><td>111</td><td>1378</td><td>93.96</td><td>10159</td><td>74.5</td></tr><tr><td>GaLe</td><td>4</td><td>254</td><td>492</td><td>37.48</td><td>3424</td><td>57.3</td></tr><tr><td>GaLe</td><td>8</td><td>127</td><td>532</td><td>37.8</td><td>3572</td><td>60.6</td></tr><tr><td>GaLe</td><td>16</td><td>84</td><td>604</td><td>41.2</td><td>3702</td><td>62.9</td></tr></table>

Table 3. Performance comparison of MBV2 A05 256x on different platforms. Times in ms

<table><tr><td>Split</td><td>Accuracy</td><td>RAM Usage (M)</td></tr><tr><td>1</td><td>91.98</td><td>6170.89</td></tr><tr><td>2</td><td>91.797</td><td>3396.60</td></tr><tr><td>4</td><td>90.732</td><td>1851.10</td></tr><tr><td>16</td><td>89.859</td><td>519.27</td></tr></table>

Table 4. Fine-grained Classification (iNaturalist) Results

At each iteration, one forward pass is performed, the error is evaluated, and the overlap for the i − th layer is increased.

Algorithm 1 Calibration of slice Overlap Parameters   
Require: Network, error tolerance ϵ, total RAM   
Ensure: Overlap values $\{ O _ { i } \} _ { i = 1 } ^ { L }$   
1: for i = 1 to L do   
2: $O _ { i }  0$   
3: $\begin{array} { r } { N \gets c e i l ( \frac { R A M _ { i } } { t o t a l ~ R A M } ) } \end{array}$   
4: repeat   
5: repeat   
6: Run N-Le-based inference, overlap $O _ { i }$   
7: Compute MSE<sub>i</sub> for output of layer i   
8: if Layer i + 1 is a convolutional layer then   
9: $O _ { i }  O _ { i } + \mathcal { R } _ { i } ( i + 1 )$   
10: else   
11: $O _ { i }  O _ { i } + s _ { i + 1 }$   
12: end if   
13: until $\mathrm { M S E } _ { i } < \epsilon \mathbf { o r }$ memory limit exceeded   
14: $N \gets N + 1$   
15: until not memory limit exceeded   
16: end for

## B. GENERALIZATION

We report additional results in Table 4, obtained by applying GaLe to a pretrained ViT on iNaturalist to evaluate finegrained classification performance. We can see that the approach can maintain high performance even on this task, with significant RAM reductions and minimal performance loss, demonstrating generalization also when used with huge numbers of classes.

## C. SENSITIVITY ANALYSIS OF WEIGHTING FACTOR α

To validate the necessity of our hybrid decomposition strategy, we conducted a sensitivity analysis on the weighting factor α, which controls the fusion between the Local Exact $( L _ { E } )$ and Global Approximate $( G _ { A } )$ feature maps. As defined in the method, the approximated output is given by a weighted combination where α governs the contribution of fine-grained local details versus global semantic context. Figure 10 illustrates the impact of α on Top-1 Accuracy for ImageNet classification. Results indicate that the method is highly stable within the range $\alpha \in [ 0 . 7 , 0 . 9 ]$ , with peak performance observed at $\alpha \approx 0 . 8$ . Notably, performance degrades at the extremes:

<table><tr><td>Attn split</td><td>LPIPS (512x512)</td><td>RAM (GB)</td><td>LPIPS (768x768)</td><td>RAM (GB)</td><td>LPIPS (1024x1024)</td><td>RAM (GB)</td></tr><tr><td>1</td><td>0.00</td><td>5.77</td><td>0.00</td><td>12.98</td><td>0.00</td><td>23.07</td></tr><tr><td>4</td><td>0.15</td><td>1.44</td><td>0.14</td><td>3.24</td><td>0.13</td><td>5.77</td></tr><tr><td>9</td><td>0.16</td><td>0.64</td><td>0.20</td><td>1.44</td><td>0.09</td><td>2.56</td></tr><tr><td>25</td><td>0.30</td><td>0.23</td><td>0.21</td><td>0.52</td><td>0.15</td><td>0.92</td></tr></table>

Table 5. Image Generation Results (High-resolution)

![](images/b19fa986e4cede27fc47833fdb5cc470fe73dbef277dd14c15c8ce14cbe76a02.jpg)  
Fig. 10. Effect of the weighting factor α on Top-1 Accuracy. The peak at $\alpha = 0 . 8$ demonstrates that a hybrid representation outperforms both pure local slicing $( \alpha = 1 . 0 )$ and aggressive global approximation (α = 0.5).

• At lower values $( \mathbf { e . g . } , \alpha = 0 . 5 )$ , the model relies too heavily on the downsampled $G _ { A }$ map, resulting in a loss of high-frequency details.

• At $\alpha ~ = ~ 1 . 0$ (pure slicing), the accuracy drops (≈ 77.1%) compared to the hybrid peak $( \approx 7 7 . 6 \% )$ . This confirms that when memory constraints prevent full receptive field overlap, the $L _ { E }$ component alone suffers from insufficient global context, which is effectively recovered by the $G _ { A }$ component.

## D. CALIBRATION EFFICIENCY AND SAMPLE SIZE

We analyzed the relationship between the size of the calibration dataset and the resulting model accuracy to assess data efficiency. As shown in Figure 11, the calibration process exhibits rapid convergence. The Top-1 Accuracy stabilizes significantly with as few as 16 to 32 samples, with negligible performance gains observed when increasing the sample size to 256. This data efficiency implies that the calibration step incurs minimal computational overhead and can be performed rapidly offline before deployment, without requiring large-scale validation sets or extensive processing time.

![](images/4bb876a1c0957629de47e896b1c230fa6ff3c62e2950f3447756ed15fef12ac1.jpg)  
Fig. 11. Effect of the number of calibration samples on Top-1 Accuracy. The method achieves optimal performance stability with as few as 32 samples, demonstrating high data efficiency and low setup overhead.

## E. ADDITIONAL RESULTS ON TIMM MODELS

Tables 6 and 7 show additional results achieved on common timm models using GaLe. Configurations are identified with the starting slice number (S), the number of split blocks $( B ) .$ the allowed MSE error for the calibration pass (ϵ) and the feature maps are used (for small RAM reductions, $L _ { E }$ features already provide the same performance as the full network, so no $G _ { A }$ features are used to reduce overhead).

<table><tr><td>Network</td><td>Configuration</td><td>RAM Reduction %</td><td>Top-1 %</td><td>Overhead %</td></tr><tr><td rowspan="5">MobilenetV2 a50</td><td></td><td></td><td>67.18</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>49.42%</td><td>67.14</td><td>+2.34%</td></tr><tr><td>4S-2B-€0.1-Le</td><td>73.62%</td><td>67.08</td><td>+5.86%</td></tr><tr><td>8S-3B-€1-Le</td><td>85.86%</td><td>67.42</td><td>+9.90%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>93.75%</td><td>64.88</td><td>+10.94%</td></tr><tr><td rowspan="6">MobilenetV2 a120</td><td></td><td></td><td>78.38</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>44.11%</td><td>78.38</td><td>+2.34%</td></tr><tr><td>4S-2B-€0.1-Le</td><td>72.16%</td><td>78.38</td><td>+7.42%</td></tr><tr><td>8S-3B-€1-GaLe</td><td>85.16%</td><td>78.38</td><td>+19.53%</td></tr><tr><td>8S-3B-€1-GaLe</td><td>85.07%</td><td>78.38</td><td>+19.53%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>92.33%</td><td>78.24</td><td>+34.38%</td></tr><tr><td rowspan="5">MobilenetV2 a140</td><td></td><td></td><td>77.30</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>48.12%</td><td>77.28</td><td>+2.34%</td></tr><tr><td>4S-2B-€0.1-Le</td><td>73.15%</td><td>77.28</td><td>+5.86%</td></tr><tr><td>8S-3B-€1-GaLe</td><td>85.26%</td><td>77.32</td><td>+14.32%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>92.16%</td><td>77.28</td><td>+27.34%</td></tr><tr><td rowspan="4">MobilenetV3 Large a100</td><td></td><td></td><td>76.94</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>48.44%</td><td>76.94</td><td>+2.34%</td></tr><tr><td>4S-2B-€0.1-Le</td><td>72.98%</td><td>76.94</td><td>+5.86%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>92.13%</td><td>76.56</td><td>+16.30%</td></tr><tr><td rowspan="6">MobilenetV3 L minimal a100</td><td></td><td></td><td>74.18</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>47.61%</td><td>74.18</td><td>+2.34%</td></tr><tr><td>4S-2B-€0.1-Le</td><td>74.18%</td><td>74.18</td><td>+5.86%</td></tr><tr><td>8S-3B-€1-GaLe</td><td>85.05%</td><td>74.16</td><td>+15.36%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>91.36%</td><td>74.18</td><td>+32.23%</td></tr><tr><td>32S-5B-€10-GaLe</td><td>95.31%</td><td>72.56</td><td>+39.06%</td></tr><tr><td rowspan="5">MobilenetV4 Hybrid M</td><td></td><td></td><td>81.74</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>48.42%</td><td>81.74</td><td>+1.56%</td></tr><tr><td>4S-2B-€0.1-Le</td><td>73.50%</td><td>81.74</td><td>+11.72%</td></tr><tr><td>8S-3B-€50-GaLe</td><td>85.99%</td><td>80.66</td><td>+14.6%</td></tr><tr><td>16S-4B-€20-GaLe</td><td>93.87%</td><td>75.66</td><td>+37.50%</td></tr><tr><td rowspan="3">MobilenetV4 Hybrid L</td><td></td><td></td><td>80.86</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>48.48%</td><td>80.86</td><td>+1.56%</td></tr><tr><td>4S-2B-€0.1-GaLe</td><td>73.48%</td><td>78.61</td><td>+18.72%</td></tr></table>

Table 6. Additional results on timm models.

<table><tr><td>Network</td><td>Configuration</td><td>RAM Reduction %</td><td>Top-1 %</td><td>Overhead %</td></tr><tr><td rowspan="4">ResNet18</td><td></td><td></td><td>72.56</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>42.25%</td><td>72.54</td><td>+14.06%</td></tr><tr><td>4S-2B-€0.1-GaLe</td><td>60.80%</td><td>72.50</td><td>+32.03%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>90.72%</td><td>64.12</td><td>+13.28%</td></tr><tr><td rowspan="4">ResNet50</td><td></td><td></td><td>80.06</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>45.17%</td><td>80.10</td><td>+9.38%</td></tr><tr><td>4S-2B-€0.1-Le</td><td>65.61%</td><td>80.08</td><td>+13.44%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>92.05%</td><td>79.52</td><td>+40.9%</td></tr><tr><td rowspan="4">ResNet200</td><td></td><td></td><td>83.60</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>43.82%</td><td>83.60</td><td>+12.50%</td></tr><tr><td>4S-2B-€0.1-GaLe</td><td>62.39%</td><td>83.60</td><td>+40.62%</td></tr><tr><td>16S-4B-e20-GaLe</td><td>90.66%</td><td>81.94</td><td>+45.12%</td></tr><tr><td rowspan="5">EfficientNetB0</td><td></td><td></td><td>78.50</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>48.54%</td><td>78.54</td><td>+2.34%</td></tr><tr><td>4S-2B-€0.1-GaLe</td><td>70.05%</td><td>78.54</td><td>+2.81%</td></tr><tr><td>8S-3B-€1-GaLe</td><td>85.98%</td><td>78.54</td><td>+14.58%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>92.34%</td><td>78.30</td><td>+21.88%</td></tr><tr><td rowspan="5">EfficientNetB2</td><td></td><td></td><td>80.18</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>47.59%</td><td>80.18</td><td>+3.91%</td></tr><tr><td>4S-2B-€0.1-Le</td><td>62.50%</td><td>80.14</td><td>+18.75%</td></tr><tr><td>8S-3B-€1-GaLe</td><td>84.35%</td><td>80.12</td><td>+19.53%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>92.07%</td><td>79.46</td><td>+23.83%</td></tr><tr><td rowspan="5">EfficientViT B1</td><td>1</td><td></td><td>80.06</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>48.46%</td><td>80.10</td><td>+2.34%</td></tr><tr><td>4S-2B-€0.1-Le</td><td>72.76%</td><td>80.08</td><td>+5.86%</td></tr><tr><td>8S-3B-€1-Le</td><td>85.85%</td><td>80.00</td><td>+12.50%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>92.86%</td><td>78.56</td><td>+12.70%</td></tr><tr><td rowspan="5">EfficientViT B2</td><td>=</td><td></td><td>82.90</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>48.47%</td><td>82.88</td><td>+2.34%</td></tr><tr><td>4S-2B-€0.1-Le</td><td>72.62%</td><td>82.92</td><td>+7.42%</td></tr><tr><td>8S-3B-€1-GaLe</td><td>85.86%</td><td>82.90</td><td>+15.10%</td></tr><tr><td>16S-4B-€5-GaLe</td><td>93.07%</td><td>82.72</td><td>+20.12%</td></tr><tr><td rowspan="4">ResNext101 32x8</td><td></td><td></td><td>82.88</td><td></td></tr><tr><td>2S-1B-€0.1-Le</td><td>43.71%</td><td>82.88</td><td>+12.50%</td></tr><tr><td>4S-2B-€0.1-GaLe</td><td>62.44%</td><td>82.88</td><td>+29.69%</td></tr><tr><td>16S-4B-e80-GaLe</td><td>93.69%</td><td>81.06</td><td>+18.75%</td></tr></table>

Table 7. Additional results on timm models.