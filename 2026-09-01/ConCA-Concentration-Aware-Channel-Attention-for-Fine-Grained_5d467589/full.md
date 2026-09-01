# ConCA: Concentration-Aware Channel Attention for Fine-Grained Visual Recognition

Yu-Sheng Liu<sup>a</sup>, Yu-Chen Tung<sup>a,∗</sup>

<sup>a</sup>Department of Physics, National Kaohsiung Normal University, Kaohsiung, 824, Taiwan

## Abstract

Lightweight channel attention mechanisms are widely used in image classification, yet their efectiveness in fine-grained visual recognition (FGVR) remains limited. Most modules summarize each channel by global average pooling (GAP), which captures activation magnitude but ignores spatial concentration, so channels with diferent spatial distributions but identical means receive6 the same descriptor. We propose Concentration-Aware Channel Attention (ConCA), which pairs the mean with a shift-invariant<sup>2</sup> negative-input entropy (NegEnt), computed via a softmax over the negated activations, forming a dual descriptor that jointly encodes<sup>0</sup> magnitude and concentration. A depthwise 1-D convolutional multi-layer perceptron (MLP), whose parameter count is linear in the number of channels, maps the pair to a per-channel weight. On six fine-grained benchmarks, ConCA improves over attention-<sup>g</sup> free, SE-Net, and ECA-Net baselines as well as four richer descriptor-based modules under a controlled from-scratch protocol,<sup>u</sup> and it generalizes across eight backbones on iNat2021-mini. These results indicate that the channel descriptor, together with the per-channel gating that maps it to attention weights, is an important but underexplored aspect of lightweight channel attention in1 FGVR.3

Keywords: Channel attention, Fine-grained visual recognition, Channel descriptor, Entropy, Convolutional neural networks

## 1. Introductionp

Fine-grained visual recognition (FGVR) distinguishes visually similar subcategories, such as bird species or aircraft models, whose discriminative cues are often confined to small local regions, a beak shape or a wing stripe, rather than the object’s global appearance. Feature representations must therefore preserve the spatial characteristics of discriminative regions while retaining their semantic information.

3 Channel attention is a widely used lightweight mechanism   
that adaptively recalibrates channel responses. Representative8   
modules such as SE-Net [1] and ECA-Net [2] difer in how the<sup>0</sup> attention weights are produced, but both summarize each channel by global average pooling (GAP), which captures activation magnitude while discarding the spatial distribution of responses.<sup>v</sup> Localized and difuse responses with identical means therefore receive the same descriptor, an ambiguity that matters in FGVR,   
where discriminative evidence is highly localized. Consistent<sup>a</sup> with this, SE-Net and ECA-Net rarely improve on the attentionfree baseline in our fine-grained experiments, with only modest gains on iNat2021-mini. This behavior motivates our central question: can a channel descriptor that also encodes spatial concentration, rather than the mean alone, improve lightweight channel attention in FGVR?

We propose Concentration-Aware Channel Attention (ConCA), which describes each channel by two statistics: the mean, reflecting the overall response level, and a negative-input entropy (NegEnt), the entropy of a softmax over the negated activations, which is shift-invariant and measures spatial concentration independently of magnitude. A lightweight depthwise 1-D convolutional multi-layer perceptron (MLP) maps this pair to a per-channel attention weight, operating independently on each channel while requiring only a number of parameters linear in the channel count, and ConCA serves as a drop-in module in standard convolutional backbones, where surrounding layers already provide suficient cross-channel interaction. Existing channel-attention research has largely focused on designing the descriptor-to-gate mapping. This work instead treats the channel descriptor, the information supplied to that mapping, as the central design axis, deliberately keeping the mapping simple.

The main contributions of this work are as follows:

• The channel descriptor as a design axis. We highlight the channel descriptor used to generate attention weights as an underexplored design axis for lightweight channel attention in FGVR, and show that a mean-only descriptor cannot distinguish channels with identical means but diferent spatial concentrations.

• Complementary descriptor design. We propose a dual descriptor combining the channel mean and the NegEnt, capturing both activation magnitude and spatial concentration with minimal parameter overhead, and instantiate it as ConCA, which couples this descriptor with per-channel gating.

• Empirical validation. We evaluate ConCA on six FGVR benchmarks, where it consistently improves over SE-Net, ECA-Net, the attention-free baseline, and four richer descriptor-based modules under a controlled from-scratch protocol, and we confirm cross-architecture generalization across eight backbones on iNat2021-mini.

Table 1: Design choices of representative channel-attention modules. Shiftinv. statistic indicates whether the descriptor includes a shift-invariant spatial statistic, and per-ch. gate indicates whether the attention weight depends only on the channel’s own descriptor. Parameter cost is reported as an order with respect to the channel count � (�: ECA kernel size, a small constant). PosEnt and NegEnt are the softmax entropies of the activations and of their negation, respectively.
<table><tr><td>Method</td><td>Descriptor</td><td>Shift-inv. statistic</td><td>Per-ch. gate</td><td>Params</td></tr><tr><td>SE-Net [1]</td><td>mean</td><td>X</td><td>X</td><td>O(C2)</td></tr><tr><td>ECA-Net [2]</td><td>mean</td><td>X</td><td>X</td><td>O(k)</td></tr><tr><td>SRM [5]</td><td>mean, std</td><td>√</td><td>√</td><td>O(C)</td></tr><tr><td>FcaNet [4]</td><td>DCT freqs.</td><td>X</td><td>X</td><td>O(C2)</td></tr><tr><td>CBAM-c [3]</td><td>mean, max</td><td>X</td><td>X</td><td>O(C2)</td></tr><tr><td>CAT-c [8]</td><td>mean, max, PosEnt</td><td>√</td><td>X</td><td>O(C2)</td></tr><tr><td>ConCA (Ours)</td><td>mean, NegEnt</td><td>√</td><td>√</td><td>O(C)</td></tr></table>

## 2. Related Work

SE-Net [1] introduced the GAP → fully connected $( \mathrm { F C } ) $ gate paradigm, and ECA-Net [2] replaced the FC bottleneck with a shared-kernel 1-D convolution, suggesting that lightweight local channel interaction can be suficient for efective channel recalibration. Later modules extend channel attention by enriching either the attention mechanism or the descriptor. CBAM [3] augments channel attention with a spatial branch and combines GAP with global max pooling (GMP) in its channel descriptor. FcaNet [4] interprets GAP as the lowest discrete cosine transform (DCT) frequency and introduces additional frequencies, and SRM [5] uses channel mean and standard deviation. These designs difer mainly in the descriptor-to-gate mapping or the pooling statistic. However, most descriptors remain first- or second-order summaries and do not explicitly characterize the spatial distribution of activations.

A related line of work uses entropy as a descriptor. Wan et al. [6] weight features by entropy before pooling. Filus and Domańska [7] propose a parameter-free Global Entropy Pooling layer and argue analytically that entropy captures distribution shape. CAT [8] fuses GAP, GMP, and an entropy pooler through a shared MLP and couples channel with spatial attention, making it the most closely related design since it also uses an entropy-based descriptor. Unlike CAT’s positive-input entropy (PosEnt) and shared channel-mixing MLP, ConCA pairs the mean with a negative-input entropy and gates each channel independently, difering along both the descriptor and the gate axes. We compare against CAT directly in Section 4.3. Table 1 compares these modules by descriptor, gating strategy, and parameter cost. Because CBAM and CAT couple channel with spatial attention, the table lists only their channel branches, denoted CBAM-c and CAT-c. Along these axes, SRM is the closest counterpart, likewise pairing the mean with a shift-invariant statistic under a per-channel gate at linear cost, but it measures dispersion (std) rather than concentration.

Because fine-grained categories difer mainly in subtle local cues, global features alone are weakly discriminative [9]. Existing approaches localize discriminative parts [10, 11], model high-order feature interactions such as bilinear pooling [12], or learn to attend end-to-end [13]. Channel attention is a lower-cost alternative, yet GAP-based modules gain little in this setting. ConCA instead operates at the descriptor level, requiring neither extra annotations nor additional feature-encoding modules.

## 3. Method

## 3.1. Motivation and Theory

Let a feature map at an intermediate layer be X $\mathbf { \Psi } \in \mathbb { R } ^ { B \times C \times H \times W }$ where $B , C , H ,$ , and � denote the batch size, number of channels, and spatial dimensions, respectively. Let $N = H \times W$ be the number of spatial positions, and denote the activation vector of channel � as $\mathbf { x } _ { c } = ( x _ { c } ^ { ( 1 ) } , \ldots , x _ { c } ^ { ( N ) } ) \in \mathbb { R } ^ { N }$ . GAP collapses it to the spatial mean

$$
\mu _ { c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } x _ { c } ^ { ( i ) } .\tag{1}
$$

Ambiguity of the mean descriptor. As a map $\mathbb { R } ^ { N } \to \mathbb { R }$ , (1) is many-to-one: very diferent spatial patterns can share the same mean. A channel with a single strong activation over a low background and a channel with a spatially uniform response can share the same mean $\mu ,$ yet GAP maps both to the same value and cannot distinguish them.

NegEnt: a complementary concentration descriptor. To recover the discarded shape information, we apply a softmax to the negated activations and take the entropy of the resulting distribution,

$$
\mathcal { E } _ { c } = - \sum _ { i = 1 } ^ { N } \tilde { p } _ { i } ^ { ( c ) } \log \tilde { p } _ { i } ^ { ( c ) } , \mathrm { ~ w h e r e ~ } \tilde { p } _ { i } ^ { ( c ) } = \frac { e ^ { - x _ { c } ^ { ( i ) } } } { \sum _ { j = 1 } ^ { N } e ^ { - x _ { c } ^ { ( j ) } } } .\tag{2}
$$

Since $\tilde { p } _ { i } ^ { ( c ) }$ is large where the activation is low, $\mathcal { E } _ { c }$ measures the spread of this induced softmax distribution and thus reflects the spatial concentration of the response rather than its absolute activation level. Two properties make it complementary to the mean descriptor. (i) It is shift-invariant: a constant ofset � cancels in the softmax, $\begin{array} { r } { e ^ { - ( x + \alpha ) } / { \sum _ { j } e ^ { - ( x _ { j } + \alpha ) } } = e ^ { - x } / { \sum _ { j } e ^ { - x _ { j } } } } \end{array}$ so $\mathcal { E } _ { c } ( \mathbf { x } _ { c } + \alpha \mathbf { 1 } ) = \mathcal { E } _ { c } ( \mathbf { x } _ { c } )$ (1 the all-ones vector) while the mean shifts by �. (ii) It provides non-redundant spatial-concentration information. For a localized high-activation pattern over a broad low background (typical in FGVR), the mass of $\tilde { p } ^ { ( c ) }$ spreads across the numerous low-activation positions, yielding a high $\mathcal { E } _ { c } ~ \approx$ log �. As the activated region expands and the lowactivation background shrinks, the mass of $\bar { \tilde { p } } ^ { ( c ) }$ concentrates on the fewer remaining low-activation positions, so $\mathcal { E } _ { c }$ decreases. The mean and NegEnt therefore capture diferent aspects of the response, a shift-sensitive activation level and a shift-invariant concentration. The pair $( \mu _ { c } , \mathcal { E } _ { c } )$ retains the mean while adding a concentration axis, separating channels that GAP alone conflates (Fig. 1).

![](images/b2191f6df5d197120ccf675fb6e3dd9fb26b22e66b0ccc07bb4b017e18dde494.jpg)  
Figure 1: Descriptor space (conceptual illustration). Channels with localized responses and with broadly activated responses overlap along the mean-response axis but separate along spatial concentration: two channels of equal mean (circled) receive the same value from a mean-only descriptor, whereas the pair $( \mu _ { c } , \mathcal { E } _ { c } )$ ) distinguishes them, since $\varepsilon _ { c }$ decreases as the high-activation region expands and the low-activation background shrinks.

Choice of entropy polarity. The entropy of a softmax can be computed from either the activations (PosEnt) or their negation (NegEnt), and both are shift-invariant. CAT [8] adopts PosEnt, whereas ConCA uses NegEnt, and the two difer once paired with the mean. For the localized responses typical of FGVR, softmax $\left( \mathbf { x } _ { c } \right)$ concentrates on the few dominant activations, so its entropy saturates near zero and PosEnt mainly reflects the magnitude of the strongest responses, information the mean already carries (Section 4.6 quantifies this overlap). Negating the input instead spreads mass across the low-activation background, yielding the concentration measure described above. The ablation in Section 4.5 supports this choice: mean+NegEnt (52.76) outperforms mean+PosEnt (51.83), which falls below the mean alone (51.96).

## 3.2. Module Design

ConCA (Fig. 2) maps an input $\mathbf { X } \in \mathbb { R } ^ { B \times C \times H \times W }$ to ${ \textbf { Y } } =$ X ⊙ g, where $\mathbf { g } = ( g _ { 1 } , \dotsc , g _ { C } )$ and $g _ { c } \in ( 0 , 1 )$ . Each gate is broadcast over the spatial dimensions, recalibrating its channel while preserving the spatial resolution.

Descriptors. The input is flattened to $\mathbb { R } ^ { B \times C \times N } \left( N = H \times W \right)$ and summarized by two per-channel statistics: the mean $\mu _ { c }$ of (1) and the NegEnt $\mathcal { E } _ { c }$ of (2). For numerical stability, $\mathcal { E } _ { c }$ is evaluated through log softmax $\left( - \mathbf { x } _ { c } \right)$ , whose internal log-sumexp avoids forming $e ^ { - x _ { c } ^ { ( i ) } }$ directly. Stacking these statistics gives the channel descriptor $\mathbf { f } _ { c } = [ \mu _ { c } , { \mathcal { E } } _ { c } ] \in \mathbb { R } ^ { m }$ with $m = 2$

Depthwise MLP. The descriptor $\mathbf { f } _ { c }$ is mapped to a gate by a twolayer depthwise 1-D convolutional MLP (kernel size 1, groups $= C )$ , applied identically and independently to every channel:

$$
{ \mathbf { h } } _ { c } = \mathrm { R e L U } \bigl ( \mathrm { C o n v 1 } \mathrm { d } _ { \mathrm { d w } } ^ { ( 1 ) } (  { \mathbf { f } } _ { c } ) \bigr ) , \quad  { \mathbf { h } } _ { c } \in \mathbb { R } ^ { 4 m } ,\tag{3}
$$

$$
g _ { c } = \mathrm { S i g m o i d } \big ( \mathrm { C o n v 1 d } _ { \mathrm { d w } } ^ { ( 2 ) } ( \mathbf { h } _ { c } ) \big ) , \quad g _ { c } \in ( 0 , 1 ) .\tag{4}
$$

The first layer expands the � descriptors to a hidden width 4� and the second compresses them to the single gate $g _ { c }$ , applied as $\mathbf { Y } _ { c , h , w } = g _ { c } \mathbf { X } _ { c , h , w }$ . Because both convolutions are depthwise $( \mathrm { g r o u p s } = C )$ $g _ { c }$ depends only on $( \mu _ { c } , \mathcal { E } _ { c } )$ , with no crosschannel interaction. Since the surrounding convolutional layers in standard backbones already mix channels, this isolates the descriptor’s efect, spending capacity on the descriptor-to-gate mapping rather than on further cross-channel interaction. This design assumes dense cross-channel interaction in the backbone, which keeps per-channel statistics meaningful and suits ConCA to standard convolutional backbones. Architectures that structurally restrict such mixing lie outside its intended scope by design (Section 4.4).

## 3.3. Complexity Analysis

The learnable parameters of ConCA reside entirely in the depthwise MLP. For $m = 2$ , the first depthwise layer contributes $4 m ^ { 2 } C + 4 m C = 2 4 C$ parameters (weights and biases), and the second contributes 4� $C + C = 9 C$ , giving

$$
P _ { \mathrm { C o n C A } } = 3 3 C ,\tag{5}
$$

which grows linearly with the number of channels. For comparison, SE-Net’s two fully connected layers require $P _ { \mathrm { S E } } \approx 2 C ^ { 2 } / r$ parameters, which grow quadratically with $C \left( r = 1 6 \right)$ , whereas ECA-Net uses a shared 1-D convolution with a small adaptive kernel $( P _ { \mathrm { E C A } } ~ = ~ k$ , a small constant). The counts of ConCA and SE-Net are comparable near $C \ = \ 2 5 6$ , whereas at $C = 2 0 4 8$ , typical of the final stages of ResNet-like convolutional neural networks (CNNs), ConCA uses only about 13% of SE-Net’s parameters. Both the mean and NegEnt are computed in $O ( C N )$ , so NegEnt does not change the asymptotic complexity. Table 2 reports the measured cost on ResNet-50 using fvcore.FlopCountAnalysis. ConCA adds negligible parameters (+0.50M over the attention-free baseline, against +2.53M for SE) and essentially no FLOPs, but its eagermode latency is higher because NegEnt launches several small reduction and element-wise kernels rather than added arithmetic. torch.compile (reduce-overhead) reduces the latency from 452.6 to 210.8 ms, indicating that much of the overhead stems from framework execution. Despite nearly identical GFLOPs (4.11), the remaining gap reflects memory movement from multiple small kernels rather than additional computation. Fusing the softmax and entropy reductions of NegEnt into a single on-chip kernel is a promising direction for reducing this overhead and will be investigated in future work.

## 4. Experiments

## 4.1. Setup

Datasets. We evaluate ConCA on six FGVR benchmarks: CUB -200-2011 [14], FGVC-Aircraft [15], Stanford Cars [16], Stanford Dogs [17], NABirds [18], and Oxford-IIIT Pets [19]. These datasets contain 37–555 sub-categories with strong intraclass variation and subtle inter-class diferences. We additionally evaluate cross-architecture generalization on the large-scale iNat2021-mini [20, 21], which contains 10,000 species with approximately 50 images per class. CUB, Cars, Dogs, NABirds, and Pets use a fixed 70/15/15 split (seed 42), and Aircraft uses its oficial split. CUB and Aircraft use the provided bounding boxes. For iNat2021-mini, we train on 45 images per class, validate on 5 images per class, and evaluate on the oficial validation set.

![](images/88e2b45a458a8a6cf68a64853747aaeee915b933b240f9cc11f40b782c72884f.jpg)  
Figure 2: ConCA overview. Each channel of X is summarized by the mean $\mu _ { c }$ and the NegEnt $\mathcal { E } _ { c } .$ . The pair $\mathbf { f } _ { c } = [ \mu _ { c } , \mathcal { E } _ { c } ]$ is mapped to a per-channel gate $g _ { c } \in ( 0 , 1 )$ by a two-layer depthwise 1-D convolutional MLP and applied as $\mathbf { Y } = \mathbf { X } \odot \mathbf { g } .$ The depthwise mapping makes � depend only on channel $^ { c , }$ meaning that each gate depends only on its own channel descriptor.

Table 2: Whole-model eficiency on ResNet-50 (ImageNet-1K setting: 224 × 224 input, 1,000 classes) on a single NVIDIA RTX 4090 in PyTorch (FP32). Latency is per batch of 512, averaged over 100 iterations after 10 warm-ups. The compiled column uses torch.compile (reduce-overhead).
<table><tr><td rowspan="2">Method Params (M)</td><td rowspan="2">GFLOPs</td><td colspan="2">Latency (ms)</td></tr><tr><td>eager</td><td>compiled</td></tr><tr><td>Baseline</td><td>25.56</td><td>4.11 218.80</td><td>142.77</td></tr><tr><td>SE [1]</td><td>28.09 4.12</td><td>255.76</td><td>178.81</td></tr><tr><td>ECA [2]</td><td>25.56 4.12</td><td>255.61</td><td>178.57</td></tr><tr><td>ConCA (Ours)</td><td>26.06</td><td>4.11 452.59</td><td>210.79</td></tr></table>

Training protocol. Fine-grained recognition and channelattention research follow two diferent conventions, and because ConCA sits at their intersection we evaluate under both. Following the standard FGVR protocol, we fine-tune an ImageNet [22]-pretrained ResNet-50 [23] on three benchmarks (Section 4.2). Following the common channel-attention protocol, we train ResNet-18 from scratch on all six benchmarks, which isolates the module’s contribution from dominant pretrained features and confirms that the baseline behavior is not an artifact of either regime (Section 4.3). Cross-architecture generalization is then evaluated on iNat2021-mini using fromscratch training (Section 4.4).

Compared methods. We compare ConCA with the attentionfree Baseline, the GAP-based SE-Net [1] and ECA-Net [2], and four richer descriptor-based modules: SRM [5] (mean and standard deviation), FcaNet [4] (multi-frequency DCT), CBAM-c [3] (GAP + GMP), and CAT-c [8] (GAP + GMP + PosEnt). CAT-c is the most direct comparison, as it also incorporates an entropy descriptor. For CBAM and CAT we evaluate only their channel branches, denoted CBAM-c and CAT-c, so that the comparison stays on the channel-descriptor axis. Their spatial branches are orthogonal to descriptor design and out of scope here. All methods use their recommended configurations, with hyper-parameters shared unless a method-specific default is required.

Hyper-parameters. For the six FGVR benchmarks, all methods are trained with ResNet-18 using AdamW [24] $( \mathrm { l r } \ ) = \ 1 0 ^ { - 3 }$ weight decay $1 0 ^ { - 2 } )$ , cosine annealing learning rate scheduling $( T _ { \mathrm { m a x } } ~ = ~ 9 0 )$ , batch size 128, and 90 epochs. Training uses cross-entropy loss, mixed precision, gradient clipping (1.0), and standard ImageNet normalization. Data augmentation includes RandomResizedCrop(224), horizontal flipping, and ColorJitter. CenterCrop(224) is used during testing. For all experiments, ConCA uses hidden width 4� = 8 and is inserted at the same position as SE-Net. We report mean ± std over five seeds (43–47) for FGVR benchmarks and three seeds (43–45) for iNat2021-mini. Model selection uses validation accuracy, and final results are reported as test top-1 accuracy.

## 4.2. Validation in the Pretrained Setting

FGVR models are commonly initialized from ImageNetpretrained backbones, so we first ask whether GAP-based channel attention behaves diferently under this standard transferlearning setting. Table 3 reports fine-tuning of an ImageNetpretrained ResNet-50 on CUB-200, Aircraft, and Cars. Because SE, ECA, and ConCA are newly inserted layers absent from the pretrained weights, their final gating layers are initialized so that each gate (the Sigmoid output) is close to one, an approximate identity mapping that leaves the pretrained features unchanged at the start of fine-tuning. All methods perform similarly, with average accuracies difering by only 0.65 points. Since every module starts from a near-identity mapping and the pretrained features are already highly discriminative, the room for lightweight channel recalibration is limited. We therefore focus on the from-scratch setting (Section 4.3), where the descriptor’s contribution is more directly observable.

## 4.3. Fine-Grained Benchmark Comparison

Table 4 reports the from-scratch comparison using ResNet-18, which isolates the contribution of the channel-attention module itself. Both GAP-only modules yield lower accuracy than the baseline across all six datasets, with average drops of 5.34 and 2.34 percentage points for SE and ECA, respectively. ConCA exceeds the baseline on all six datasets (+1.59 on average) and achieves the highest mean accuracy among the evaluated channel-attention modules on every benchmark. One-sided Welch’s �-tests over the five seeds (marked in Table 4) show that ConCA significantly improves $( p < 0 . 0 5 )$ over SE on all six datasets and over ECA on four. Over the strong attention-free baseline, the gains are significant on CUB-200, NABirds, and Pets and lie within run-to-run variation on Aircraft, Cars, and Dogs, where the margins are small. ConCA also has lower runto-run variance than the Baseline, SE, and ECA on every dataset except Cars, where variance is large for all methods yet ConCA still attains the highest mean accuracy, 1.67 points above the baseline.

Table 3: Top-1 accuracy (%, mean ± std over 5 seeds) under the standard FGVR transfer-learning setting. An ImageNet-pretrained ResNet-50 is fine-tuned on CUB-200, Aircraft, and Cars. The Avg column reports the unweighted mean of the three dataset means, with uncertainty computed by propagating the perdataset standard deviations assuming independent runs across datasets.
<table><tr><td>Method</td><td>CUB</td><td> $\operatorname { A i r } .$ </td><td> $\mathrm { C a r s }$ </td><td> $\mathbf { A v \mathrm { g } }$ </td></tr><tr><td>Baseline</td><td> $7 0 . 7 6 \pm 1 . 0 9$ </td><td> $4 8 . 2 6 \pm 0 . 5 9$ </td><td> $8 1 . 8 3 \pm 0 . 6 9$ </td><td> $6 6 . 9 5 \pm 0 . 4 7$ </td></tr><tr><td>SE [1]</td><td> $7 0 . 2 7 \pm 0 . 8 2$ </td><td> $4 8 . 5 6 \pm 0 . 1 8$ </td><td> $8 1 . 2 8 \pm 0 . 3 8$ </td><td> $6 6 . 7 0 \pm 0 . 3 1$ </td></tr><tr><td>ECA [2]</td><td> $7 0 . 5 4 \pm 0 . 3 9$ </td><td> $4 8 . 3 8 \pm 1 . 0 0$ </td><td> $8 1 . 7 0 \pm 0 . 8 5$ </td><td> $6 6 . 8 7 \pm 0 . 4 6$ </td></tr><tr><td>ConCA (Ours)</td><td> ${ \bf 7 1 . 0 7 \pm 0 . 7 9 }$ </td><td> ${ \bf 4 8 . 9 3 \pm 0 . 5 0 }$ </td><td> $\mathbf { 8 2 . 0 5 \pm 0 . 7 0 }$ </td><td> ${ \bf 6 7 . 3 5 \pm 0 . 3 9 }$ </td></tr></table>

The richer descriptor-based modules also fail to consistently improve over the baseline. CAT-c, despite using an entropy descriptor, trails ConCA by 6.88 points, while SRM, the strongest competing module, remains 3.79 points lower on average. ConCA is the only method that improves over the baseline on all six benchmarks. These comparisons suggest that efective channel attention in fine-grained recognition benefits from pairing a non-redundant descriptor with per-channel gating, rather than from entropy or a richer descriptor in isolation.

## 4.4. Cross-Architecture Generalization

For the cross-architecture study we retain SE and ECA, the most widely adopted baselines, since the richer modules in Table 4 gave no consistent advantage over ECA in Section 4.3. Table 5 reports results on eight architectures spanning the ResNet [23], Inception [25, 26], DenseNet [27], and HRNet [28] families. These eight backbones are built predominantly from standard convolutions and therefore satisfy ConCA’s assumption of dense cross-channel interaction, providing a fair testbed for its cross-architecture generalization. ConCA achieves the highest accuracy on all eight architectures. The gains are particularly pronounced on ResNet-50 (+1.85 percentage points over the second-best method) and Inception-v4 (+1.07), while smaller but consistent improvements are observed on the remaining architectures. Unlike on the small FGVR datasets, SE and ECA often improve over the baseline on iNat2021-mini. However, the efect remains architecture dependent, as both fall below the baseline on both DenseNet variants.

A one-sided Welch’s �-test $( \alpha = 0 . 0 5 )$ indicates that ConCA significantly outperforms the baseline on seven of the eight architectures, SE on six, and ECA on seven. The remaining comparisons do not reach statistical significance with three training seeds. The dual descriptor’s benefit thus generalizes across backbones with substantially diferent connectivity and featureaggregation strategies.

## 4.5. Descriptor Ablation

We evaluate descriptor combinations on CUB-200, Aircraft, and Cars over five seeds. The candidates comprise level descriptors (mean, max, min), which describe activation magnitude, and shape descriptors (std, skew, PosEnt, NegEnt), which characterize the spatial distribution. Preliminary experiments showed little benefit from min, so it appears only in the all-7 combination.

Table 6 indicates that combining complementary descriptor types outperforms any single statistic: the best single descriptor, mean (51.96%), rises to 52.76% when paired with NegEnt, the highest average of all combinations. Among shape complements to the mean, NegEnt outperforms std (52.45%), PosEnt (51.83%), and skew (50.62%). PosEnt and NegEnt perform almost identically in isolation (50.93% vs. 50.94%), yet only NegEnt gains substantially with the mean, a redundancy efect that Section 4.6 quantifies directly. Skew is unreliable when estimated from only $N = H \times W$ spatial samples. Adding more descriptors does not help: mean+max+NegEnt and the all-7 combination fall below mean+NegEnt, the latter sharply (48.49%), suggesting that descriptor quality matters more than quantity. A one-sided Welch’s �-test on Avg indicates that ConCA significantly outperforms several alternatives, including NegEnt alone $( p = 0 . 0 1 8 )$ , std alone $( p = 0 . 0 4 3 )$ , mean+skew $( p = 0 . 0 1 0 )$ max+NegEnt $( p = 0 . 0 2 6 )$ , and all 7 $( p < 0 . 0 0 1 )$ , while the remaining comparisons do not reach significance with five seeds.

Descriptor versus gate. The ablation above holds the perchannel gate fixed, isolating the descriptor’s contribution given that gate. Table 7 instead crosses the two descriptors (mean, mean+NegEnt) with two gating strategies (full cross-channel interaction as in SE, and the per-channel gate). Per-channel gating alone already exceeds the attention-free baseline, whereas full interaction falls well below it, and NegEnt adds a further gain on top of the per-channel gate.

## 4.6. Descriptor Redundancy Analysis

A natural objection is that entropy, as a concentration measure, might merely restate the magnitude already summarized by the mean. We test this directly, without reference to accuracy, by measuring how much of each entropy descriptor is fixed by the channel mean. Using the five attention-free base line ResNet-18 models of Table 4, we extract features at the layer-4 ConCA insertion point on the CUB-200 test split, pool $4 . 5 3 \times 1 0 ^ { 6 }$ channel samples, and regress each descriptor on $\mu _ { c } ,$ reporting the coeficient of determination $r ^ { 2 }$ . As Figure 3 shows, PosEnt varies systematically with $\mu _ { c } .$ , so the mean accounts for a sizable share of its variation $( r ^ { 2 } = 0 . 3 7 )$ , whereas NegEnt stays largely flat while retaining spread at each $\mu _ { c }$ , leaving most

Table 4: Top-1 accuracy (%, mean ± std over 5 seeds) on six fine-grained benchmarks using ResNet-18 trained from scratch. The best result in each column is shown in bold. †/‡/§: $p < 0 . 0 5$ (one-sided Welch’s �-test, 5 seeds) for ConCA vs. Baseline/SE/ECA, respectively. Avg follows the averaging convention of Table 3.
<table><tr><td>Method</td><td> $\mathrm { C U B - } 2 0 0$ </td><td>Aircraft</td><td>Cars</td><td>Dogs</td><td> $\mathrm { N A B i r d s }$ </td><td> $\mathrm { P e t s }$ </td><td> $\mathbf { A v } \mathbf { g }$ </td></tr><tr><td>Baseline</td><td> $5 8 . 9 8 \pm 0 . 9 6$ </td><td> $4 1 . 5 7 \pm 0 . 7 6$ </td><td> $5 3 . 8 0 \pm 4 . 3 7$ </td><td> $6 0 . 7 9 \pm 0 . 5 6$ </td><td> $6 9 . 2 9 \pm 0 . 6 1$ </td><td> $5 5 . 9 4 \pm 1 . 9 5$ </td><td> $5 6 . 7 3 \pm 0 . 8 3$ </td></tr><tr><td>SE [1]</td><td> $5 3 . 8 5 \pm 0 . 9 6$ </td><td> $3 8 . 0 6 \pm 1 . 1 8$ </td><td> $4 1 . 1 1 \pm 1 . 0 1$ </td><td> $5 6 . 2 1 \pm 0 . 6 0$ </td><td> $6 7 . 1 3 \pm 0 . 3 2$ </td><td> $5 1 . 9 9 \pm 2 . 5 1 $ </td><td> $5 1 . 3 9 \pm 0 . 5 3$ </td></tr><tr><td>ECA [2]</td><td> $5 5 . 3 9 \pm 1 . 3 8$ </td><td> $4 0 . 1 5 \pm 1 . 4 3$ </td><td> $5 0 . 4 6 \pm 3 . 4 4$ </td><td> $5 8 . 2 4 \pm 0 . 8 0$ </td><td> $6 7 . 1 0 \pm 0 . 4 0$ </td><td> $5 5 . 0 0 \pm 2 . 7 9$ </td><td> $5 4 . 3 9 \pm 0 . 8 2$ </td></tr><tr><td>SRM [5]</td><td> $5 7 . 8 8 \pm 1 . 0 4$ </td><td> $3 9 . 1 1 \pm 0 . 9 1$ </td><td> $4 8 . 0 7 \pm 0 . 5 8$ </td><td> $5 7 . 9 5 \pm 0 . 8 0$ </td><td> $6 7 . 9 9 \pm 0 . 2 3 $ </td><td> $5 6 . 1 9 \pm 1 . 7 6$ </td><td> $5 4 . 5 3 \pm 0 . 4 1$ </td></tr><tr><td>FcaNet [4]</td><td> $5 5 . 8 3 \pm 0 . 2 5$ </td><td>40.78 ± 0.55</td><td> $4 6 . 1 0 \pm 1 . 7 9$ </td><td>58.18 ± 0.92</td><td> $6 7 . 5 8 \pm 0 . 2 6$ </td><td> $5 5 . 5 8 \pm 1 . 9 9$ </td><td> $5 4 . 0 1 \pm 0 . 4 8$ </td></tr><tr><td> $\mathrm { C B A M - c } \ [ 3 ]$ </td><td> $5 4 . 9 1 \pm 1 . 0 6$ </td><td> $4 0 . 7 9 \pm 1 . 2 5$ </td><td> $4 4 . 1 3 \pm 2 . 6 5$ </td><td> $5 8 . 3 9 \pm 0 . 7 4$ </td><td> $6 6 . 4 6 \pm 0 . 5 2$ </td><td> $5 7 . 1 4 \pm 2 . 7 6$ </td><td> $5 3 . 6 4 \pm 0 . 7 1$ </td></tr><tr><td>CAT-c [8]</td><td> $5 2 . 0 4 \pm 0 . 8 0$ </td><td> $3 9 . 2 0 \pm 0 . 7 5$ </td><td> $4 1 . 0 8 \pm 1 . 4 7$ </td><td> $5 6 . 1 6 \pm 0 . 8 0$ </td><td> $6 3 . 1 1 \pm 0 . 2 0$ </td><td> $5 7 . 0 3 \pm 2 . 5 0 $ </td><td> $5 1 . 4 4 \pm 0 . 5 3$ </td></tr><tr><td>ConCA (Ours)</td><td> $\mathbf { 6 1 . 1 5 \pm 0 . 5 7 } ^ { \div \ddagger }$ </td><td> ${ \bf 4 1 . 6 6 \pm 0 . 4 3 ^ { \ddagger } }$ </td><td> $\pm 5 5 . 4 7 \pm 3 . 8 7 ^ { \ddagger }$ </td><td> ${ \bf 6 1 . 4 0 \pm 0 . 4 3 ^ { \div } } ^ { \xi }$ </td><td> $\mathbf { 7 1 . 1 5 \pm 0 . 2 2 } ^ { \div \div }$ </td><td> $\pm 9 . 0 9 \pm 1 . 8 3 ^ { \dagger \ddagger }$ </td><td> ${ \bf 5 8 . 3 2 \pm 0 . 7 3 }$ </td></tr></table>

Table 6: Descriptor-combination ablation: top-1 accuracy (%, mean ± std over 5 seeds) and the across-dataset average (Avg). Single-descriptor rows feed only that descriptor to the depthwise MLP (hidden width 4�). Avg follows the averaging convention of Table 3.  
Table 5: Top-1 accuracy (%, mean ± std over 3 seeds) on iNat2021-mini across eight architectures. ConCA attains the best accuracy on every architecture.

$$
\mathrm { R e s N e t } { \mathrm { - } } 1 8
$$

$$
4 3 . 6 4 \pm 0 . 2 8
$$

$$
4 3 . 6 2 \pm 0 . 1 9
$$

$$
4 8 . 6 3 \pm 0 . 1 3
$$

$$
5 0 . 4 3 \pm 0 . 8 8
$$

$$
4 4 . 4 7 \pm 0 . 1 2
$$

$$
{ \bf 4 5 . 1 0 \pm 0 . 2 7 }
$$

$$
4 8 . 8 4 \pm 0 . 4 5
$$

$$
4 9 . 6 6 \pm 1 . 3 8
$$

$$
5 0 . 3 1 \pm 0 . 7 4
$$

$$
5 0 . 7 2 \pm 0 . 2 8
$$

$$
{ \pm 2 . 2 8 \pm 0 . 7 9 }
$$

$$
5 6 . 9 1 \pm 0 . 2 1
$$

$$
5 8 . 6 7 \pm 0 . 0 8 
$$

$$
{ \bf 5 1 . 2 1 \pm 0 . 2 6 }
$$

$$
5 8 . 7 6 \pm 0 . 0 5
$$

$$
5 6 . 1 8 \pm 1 . 3 1
$$

$$
{ \bf 5 9 . 0 1 \pm 0 . 2 1 }
$$

$$
5 9 . 8 8 \pm 0 . 1 4
$$

$$
5 2 . 7 6 \pm 0 . 0 8
$$

$$
5 0 . 1 6 \pm 0 . 2 2
$$

$$
4 7 . 7 6 \pm 1 . 1 1
$$

$$
{ \bf 6 0 . 9 5 \pm 0 . 1 1 }
$$

$$
5 5 . 1 3 \pm 0 . 1 2
$$

$$
5 4 . 2 2 \pm 0 . 3 0
$$

$$
{ \bf 5 3 . 5 5 \pm 0 . 1 7 }
$$

$$
\mathrm { H R N e t - W 1 8 – C }
$$

$$
5 4 . 4 3 \pm 0 . 6 6
$$

$$
4 9 . 0 4 \pm 0 . 9 1 
$$

$$
5 0 . 4 0 \pm 0 . 2 1
$$

$$
4 8 . 9 6 \pm 0 . 2 4
$$

$$
{ \pm } 5 5 . 3 2 \pm 0 . 4 1
$$

$$
{ \bf 5 0 . 5 9 \pm 0 . 2 4 }
$$

iNat schedules (all AdamW + CosineAnnealingLR, 90 epochs unless noted)   
Abbreviations: bs (batch size), lr (learning rate), wd (weight decay), and +w (a 5-epoch linear warm-upfrom 10% ofthe base lr). ResNet-18: bs 256, lr $1 0 ^ { - 3 } , \mathrm { w d } \ 1 0 ^ { - 2 } .$ . ResNet-50: bs 1 $\lbrack 8 , \mathrm { l r } \bar { 5 } \times 1 0 ^ { - 4 } ,$ wd $1 0 ^ { - 2 } .$ . ResNet-101: bs 128, lr $1 0 ^ { - 3 }$ , wd $1 0 ^ { - 4 } .$ Inception-v3: bs $2 5 6 , \mathrm { l r } 1 0 ^ { - 3 }$ , wd $1 0 ^ { - 4 } , + \mathrm { w } .$   
Inception-v4: bs $1 2 8 , \mathrm { l r } 1 0 ^ { - 3 } ,$ , wd $1 0 ^ { - 4 } ,$ +w. DenseNet-121: bs 256, lr $1 0 ^ { - 3 } ,$   
wd $\bar { 1 0 } ^ { - 2 } .$ , +w. DenseNet-201: bs 128, lr $1 0 ^ { - 3 } .$ , wd 0.05, +w. HRNet-W18-C:   
bs 256, lr $1 0 ^ { - 3 }$ , wd $1 0 ^ { - 4 }$ , 120 epochs, +w. All use ImageNet normalization, RandomResizedCrop(224), horizontal flip, RandomRotation(15) and ColorJitter(0.2, 0.2, 0.2, 0.1).

<table><tr><td>Descriptor</td><td>CUB</td><td> $\operatorname { A i r } .$ </td><td> $\mathbf { C a r s }$ </td><td> $\mathbf { A v } \mathbf { g }$ </td></tr><tr><td>mean</td><td> $6 0 . 0 6 \pm 0 . 8 1$ </td><td> $4 1 . 2 3 \pm 0 . 9 5$ </td><td> $5 4 . 5 9 \pm 1 . 5 7$ </td><td> $5 1 . 9 6 \pm 0 . 6 7$ </td></tr><tr><td>PosEnt</td><td> $5 8 . 5 4 \pm 0 . 9 7 $ </td><td> $4 1 . 0 3 \pm 0 . 4 1$ </td><td> $5 3 . 2 3 \pm 1 . 3 1$ </td><td> $5 0 . 9 3 { \scriptstyle \pm 0 . 5 6 }$ </td></tr><tr><td>NegEnt</td><td> $5 8 . 4 3 \pm 0 . 6 5$ </td><td> $4 0 . 8 7 \pm 0 . 7 8$ </td><td> $5 3 . 5 2 \pm 2 . 0 8$ </td><td> $5 0 . 9 4 \pm 0 . 7 7$ </td></tr><tr><td>max</td><td> $5 9 . 4 6 \pm 0 . 7 5$ </td><td> $4 1 . 5 4 \pm 1 . 2 8$ </td><td> $5 2 . 9 9 \pm 1 . 6 2 $ </td><td> $5 1 . 3 3 \pm 0 . 7 3$ </td></tr><tr><td>std</td><td> $5 9 . 8 1 \pm 0 . 2 2$ </td><td> $4 1 . 0 2 \pm 1 . 1 0$ </td><td> $5 3 . 2 6 \pm 1 . 6 5$ </td><td> $5 1 . 3 6 \pm 0 . 6 7$ </td></tr><tr><td> $\mathrm { m e a n } + \mathrm { s t d }$ </td><td> $6 1 . 1 4 \pm 1 . 0 7$ </td><td> $4 0 . 9 1 \pm 0 . 8 7$ </td><td> $5 5 . 3 1 \pm 1 . 7 5$ </td><td> $5 2 . 4 5 { \scriptstyle \pm 0 . 7 4 }$ </td></tr><tr><td> $\mathrm { m e a n } + \mathrm { s k e w }$ </td><td> $5 8 . 3 7 \pm 0 . 6 7 $ </td><td> $4 1 . 3 1 \pm 0 . 3 9$ </td><td> $5 2 . 1 9 \pm 1 . 4 0$ </td><td> $5 0 . 6 2 \pm 0 . 5 3 $ </td></tr><tr><td> $\mathrm { m e a n + m a x }$ </td><td> $6 1 . 5 2 \pm 0 . 6 0$ </td><td> $4 1 . 1 5 \pm 0 . 4 3$ </td><td> $5 5 . 2 4 \pm 0 . 9 3$ </td><td> $5 2 . 6 4 \pm 0 . 4 0$ </td></tr><tr><td>max + NegEnt</td><td> $5 9 . 8 9 _ { \pm 0 . 6 6 }$ </td><td> $4 0 . 3 8 \pm 0 . 5 9$ </td><td> $5 2 . 8 6 \pm 2 . 2 4$ </td><td> $5 1 . 0 4 \pm 0 . 8 0$ </td></tr><tr><td> $\mathrm { s t d } + \mathrm { N e g E n t }$ </td><td> $6 0 . 7 3 \pm 0 . 9 7$ </td><td> $4 1 . 0 2 \pm 0 . 6 8$ </td><td> $5 3 . 9 6 \pm 0 . 9 6$ </td><td> $5 1 . 9 0 { \scriptstyle \pm 0 . 5 1 }$ </td></tr><tr><td> $\mathrm { m e a n } + \mathrm { P o s E n t }$ </td><td> $6 0 . 3 4 \pm 0 . 5 1$ </td><td> $4 1 . 0 9 _ { \pm 0 . 7 9 }$ </td><td> $5 4 . 0 6 \pm 3 . 2 1$ </td><td> $5 1 . 8 3 \pm 1 . 1 1 $ </td></tr><tr><td> $\mathbf { m e a n } + \mathbf { N e g E n t }$ </td><td> ${ \bf 6 1 . 1 5 \pm 0 . 5 7 }$ </td><td> $\mathbf { 4 1 . 6 6 \substack { \pm 0 . 4 3 } }$ </td><td> ${ \pm 5 5 . 4 7 \pm 3 . 8 7 }$ </td><td> $\pm 2 . 7 6 \pm 1 . 3 1$ </td></tr><tr><td>mean + PosEnt + NegEnt</td><td> $6 0 . 4 3 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $4 1 . 2 9 \pm 0 . 9 2 $ </td><td> $5 4 . 0 1 \pm 1 . 5 4$ </td><td> $5 1 . 9 1 \pm 0 . 6 3$ </td></tr><tr><td> $\mathrm { m e a n } + \mathrm { m a x } + \mathrm { P o s E n t }$ </td><td> $6 1 . 0 2 \pm 1 . 0 5$ </td><td> $4 1 . 4 9 \pm 1 . 4 1$ </td><td> $5 2 . 9 2 \pm 0 . 8 7$ </td><td> $5 1 . 8 1 \pm 0 . 6 5$ </td></tr><tr><td> $\mathrm { m e a n } + \mathrm { m a x } + \mathrm { N e g E n t }$ </td><td> $6 1 . 7 8 \pm 0 . 9 4$ </td><td> $4 1 . 9 5 \pm 1 . 2 6$ </td><td> $5 2 . 7 5 \pm 1 . 4 5$ </td><td> $5 2 . 1 6 \pm 0 . 7 1$ </td></tr><tr><td>all 7</td><td> $5 6 . 9 0 \pm 1 . 1 2$ </td><td> $4 1 . 0 3 \pm 1 . 1 7$ </td><td> $4 7 . 5 4 \pm 0 . 6 1$ </td><td> $4 8 . 4 9 \pm 0 . 5 8 $ </td></tr></table>

of its variation unexplained $( r ^ { 2 } = 0 . 1 1 )$ . PosEnt thus appears substantially more redundant with the mean than NegEnt, while NegEnt carries spatial structure that the mean does not.

This accounts for the ablation independently of accuracy: because PosEnt is largely co-determined with the mean, adding it helps little (mean+PosEnt 51.83% vs. mean 51.96%), whereas the near mean-independent NegEnt supplies a complementary axis and yields the best pair (52.76%).

Table 7: Gate × descriptor study (top-1 %, Avg over CUB-200/Aircraft/Cars, 5 seeds). Avg follows the averaging convention of Table 3. Full-interaction and per-channel gates use the same hidden width and training schedule, difering only in cross-channel interaction.
<table><tr><td>Gate</td><td>Descriptor</td><td>Avg</td></tr><tr><td>Baseline (none)</td><td>一</td><td> $5 1 . 4 5 \pm 1 . 5 1$ </td></tr><tr><td>Full interaction</td><td>mean</td><td> $4 4 . 3 4 \pm 0 . 6 1$ </td></tr><tr><td>Full interaction</td><td> ${ \mathrm { m e a n } } + { \mathrm { N e g E n t } }$ </td><td> $4 4 . 7 8 \pm 0 . 5 5$ </td></tr><tr><td>Per-channel</td><td>mean</td><td> $5 1 . 9 6 \pm 0 . 6 7$ </td></tr><tr><td>Per-channel</td><td> $\mathbf { m e a n } + \mathbf { N e g E n t } \ ( \mathbf { C o n C A } )$ </td><td> ${ \pm 2 . 7 6 \pm 1 . 3 1 }$ </td></tr></table>

## 5. Conclusion

descriptor design to alternative concentration measures, gating strategies, and transformer architectures.

## CRediT authorship contribution statement

We identified the channel descriptor, together with the perchannel gating that maps it to attention weights, as an underexplored design axis in lightweight channel attention for finegrained recognition. ConCA pairs the shift-sensitive mean with the shift-invariant NegEnt under a depthwise per-channel gate, and it consistently outperforms existing channel-attention modules across six benchmarks and eight CNN architectures at minimal parameter cost, with the gain attributed jointly to the gating and the non-redundant descriptor rather than to entropy alone. ConCA assumes that channels carry meaningful spatial structure and, by design, targets dense-convolution backbones rather than strongly grouped architectures. Future work will extend

Yu-Sheng Liu: Writing – original draft, Methodology, Software, Investigation. Yu-Chen Tung: Writing – review & editing, Conceptualization, Supervision, Funding acquisition.

## Declaration of competing interest

The authors declare no competing interests.

![](images/a37cdc9a7d7dc073e8399fbc4d8d2a8f65bb04d4e2ee70c056c15d32b386ccad.jpg)  
Figure 3: Redundancy of the entropy descriptors with the channel mean, measured on the attention-free baseline ResNet-18 (CUB-200 test split, five seeds pooled, $4 . 5 3 \times 1 0 ^ { 6 }$ channel samples). Each curve traces the per-bin fitted center of an entropy descriptor as a function of the channel mean $\mu _ { c } ,$ and the shaded band shows the fitted ±1� width within each $\mu _ { c }$ bin. PosEnt varies systematically with �<sub>�</sub> $( r ^ { 2 } = 0 . 3 7 ) $ , whereas NegEnt stays largely flat and is only weakly determined by $\mu _ { c } \ ( r ^ { 2 } = 0 . 1 1 )$ , indicating that NegEnt carries channel-level structure not encoded by the mean.

## Data availability

All datasets used in this study are publicly available.

## Acknowledgements

The authors acknowledge the computational resources provided through the support of the National Science and Technology Council (NSTC), Taiwan, under Grant No. 114-2112-M-017-001.

## References

[1] J. Hu, L. Shen, G. Sun, Squeeze-and-excitation networks, in: Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132–7141.

[2] Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, Q. Hu, ECA-Net: Eficient channel attention for deep convolutional neural networks, in: Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 11534–11542.

[3] S. Woo, J. Park, J.-Y. Lee, I. S. Kweon, CBAM: Convolutional block attention module, in: Proc. European Conf. on Computer Vision (ECCV), 2018, pp. 3–19.

[4] Z. Qin, P. Zhang, F. Wu, X. Li, FcaNet: Frequency channel attention networks, in: Proc. IEEE/CVF Int. Conf. on Computer Vision (ICCV), 2021, pp. 763–772.

[5] H. Lee, H. Kim, H. Nam, SRM: A style-based recalibration module for convolutional neural networks, in: Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 1854–1863.

[6] W. Wan, J. Chen, T. Li, Y. Huang, J. Tian, C. Yu, Y. Xue, Information entropy based feature pooling for convolutional neural networks, in: Proc. IEEE/CVF Int. Conf. on Computer Vision (ICCV), 2019, pp. 4311–4320.

[7] K. Filus, J. Domańska, Global entropy pooling layer for convolutional neural networks, Neurocomputing 555 (2023) 126615.

[8] Z. Wu, M. Wang, W. Sun, Y. Li, T. Xu, F. Wang, K. Huang, CAT: Learning to collaborate channel and spatial attention from multi-information fusion, IET Comput. Vis. 17 (3) (2023) 309–318.

[9] X.-S. Wei, Y.-Z. Song, O. Mac Aodha, J. Wu, Y. Peng, J. Tang, J. Yang, Fine-grained image analysis with deep learning: A survey, IEEE Trans. Pattern Anal. Mach. Intell. 44 (12) (2022) 9609–9630.

[10] N. Zhang, J. Donahue, R. Girshick, T. Darrell, Part-based R-CNNs for fine-grained category detection, in: Proc. European Conf. on Computer Vision (ECCV), 2014, pp. 834– 849.

[11] S. Branson, G. Van Horn, S. Belongie, P. Perona, Bird species categorization using pose normalized deep convolutional nets, in: Int. Conf. on Learning Representations (ICLR), 2014.

[12] T.-Y. Lin, A. RoyChowdhury, S. Maji, Bilinear CNN models for fine-grained visual recognition, in: Proc. IEEE Int. Conf. on Computer Vision (ICCV), 2015, pp. 1449–1457.

[13] H. Zheng, J. Fu, T. Mei, J. Luo, Learning multi-attention convolutional neural network for fine-grained image recognition, in: Proc. IEEE Int. Conf. on Computer Vision (ICCV), 2017, pp. 5209–5217.

[14] C. Wah, S. Branson, P. Welinder, P. Perona, S. Belongie, The Caltech-UCSD Birds-200-2011 dataset, Tech. Rep., California Institute of Technology, 2011.

[15] S. Maji, E. Rahtu, J. Kannala, M. Blaschko, Fine-grained visual classification of aircraft, arXiv:1306.5151, 2013.

[16] J. Krause, M. Stark, J. Deng, L. Fei-Fei, 3D object representations for fine-grained categorization, in: Proc. IEEE Int. Conf. on Computer Vision Workshops (ICCVW), 2013, pp. 554–561.

[17] A. Khosla, N. Jayadevaprakash, B. Yao, F.-F. Li, Novel dataset for fine-grained image categorization: Stanford Dogs, in: CVPR Workshop on Fine-Grained Visual Categorization (FGVC), 2011.

[18] G. Van Horn, S. Branson, R. Farrell, S. Haber, J. Barry, P. Ipeirotis, P. Perona, S. Belongie, Building a bird recognition app and large scale dataset with citizen scientists, in: Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 595–604.

[19] O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, Cats and dogs, in: Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2012, pp. 3498–3505.

[20] G. Van Horn, O. Mac Aodha, Y. Song, Y. Cui, C. Sun, A. Shepard, H. Adam, P. Perona, S. Belongie, The iNaturalist species classification and detection dataset, in: Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 8769–8778.

[21] G. Van Horn, E. Cole, S. Beery, K. Wilber, S. Belongie, O. Mac Aodha, Benchmarking representation learning for natural world image collections, in: Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 12884–12893.

[22] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, L. Fei-Fei, ImageNet large scale visual recognition challenge, Int. J. Comput. Vis. 115 (3) (2015) 211– 252.

[23] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770– 778.

[24] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, in: Int. Conf. on Learning Representations (ICLR), 2019.

[25] C. Szegedy, V. Vanhoucke, S. Iofe, J. Shlens, Z. Wojna, Rethinking the Inception architecture for computer vision, in: Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2818–2826.

[26] C. Szegedy, S. Iofe, V. Vanhoucke, A. A. Alemi, Inception-v4, Inception-ResNet and the impact of residual connections on learning, in: Proc. AAAI Conf. on Artificial Intelligence, 2017.

[27] G. Huang, Z. Liu, L. Van Der Maaten, K. Q. Weinberger, Densely connected convolutional networks, in: Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 4700–4708.

[28] J. Wang, K. Sun, T. Cheng, B. Jiang, C. Deng, Y. Zhao, D. Liu, Y. Mu, M. Tan, X. Liu, W. Liu, B. Xiao, Deep highresolution representation learning for visual recognition, IEEE Trans. Pattern Anal. Mach. Intell. 43 (10) (2020) 3349–3364.