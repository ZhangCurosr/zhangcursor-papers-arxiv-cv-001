# Dynamics-Aware Sparse Attention for Efficient Autoregressive Video Diffusion

Yuanyu He<sup>1,2∗</sup> Zhuokun Chen<sup>3∗</sup> Yefei He<sup>1,†</sup> Zhiwei Tang<sup>2</sup> Jiasheng Tang<sup>2</sup> Yinghao Yu<sup>2</sup> Jianfei Cai<sup>3</sup> Bohan Zhuang<sup>1,‡</sup>

<sup>1</sup>Zhejiang University <sup>2</sup>Alibaba Group <sup>3</sup>Monash University

## Abstract

Autoregressive Diffusion Transformers have emerged as a powerful paradigm for long video generation. However, they are often constrained by prohibitive computational overhead and substantial memory footprints. This inefficiency arises from dense computation mechanisms that allocate uniform processing resources across all spatiotemporal regions, failing to leverage the inherent temporal redundancy of video where motion is typically confined to sparse areas. In this paper, we propose Dynamics-Aware Sparse Attention (DASA), a unified framework that shifts from dense computation to sparse, motion-driven generation to eliminate both storage and computational redundancies. Inspired by video codec principles, our approach explicitly decouples video content into dynamic and static regions. To mitigate memory bottlenecks, we selectively compress the historical KV cache by retaining only dynamic features within the generated chunk. Furthermore, to reduce computational overhead, we employ a region-level computation allocation strategy based on localized dynamics. Additionally, a dynamics-guided distillation method is introduced to enhance performance via a lightweight distillation training process. Experiments on VBench demonstrate that our approach accelerates autoregressive video generation by 2.16×, while preserving high visual fidelity. Code is available on our project page: https://yuanyu0.github.io/DASA/.

## 1 Introduction

Video generation has witnessed unprecedented breakthroughs, propelled by the rapid progress of both proprietary [2, 27, 35, 34] and open-source [40, 26, 44, 29] foundational models. While these models demonstrate remarkable spatial-temporal fidelity, their outputs remain largely constrained to brief 5- to 20-second clips due to the quadratic computational complexity inherent in full-attention mechanisms. To achieve causally consistent, minute-level synthesis, the field is shifting toward Autoregressive (AR) Diffusion Models [22, 39, 3, 42, 47]. By factorizing the generation process into sequential chunks, AR diffusion offers a promising pathway toward infinite-length video synthesis.

Despite these compelling advantages, chunk-wise AR diffusion currently faces two severe efficiency bottlenecks in long video generation. First, prohibitive memory expansion. To maintain long-term causal consistency, AR video models typically retain the full historical KV cache across chunks. For example, when generating a 480P 60-second video with LingBot-World [39], the KV cache can occupy 300GB of memory space, while the model weights only require 56GB. Moreover, the KV cache grows further as video resolution and duration increase. Second, severe latency stems from redundant computation. Current AR architectures blindly allocate identical dense compute resources to every token per chunk, despite most scenes having high temporal redundancy.

While recent acceleration methods [12, 32, 48] for AR diffusion have analyzed attention patterns to prune inactive tokens or heads, they often overlook the inherent nature of video—where most scenes exhibit high temporal redundancy and motion is confined to sparse regions. As shown in Figure 1 (a), real-world videos are dominated by static backgrounds and continuous dynamics, demonstrating massive spatiotemporal redundancy in adjacent frames. Addressing these intertwined memory and computational bottlenecks is therefore crucial for efficient AR-diffusion based video generation.

![](images/1b48fd69c10dd97f94d4e627fd4c7b2deb82413601a019001a5dcc37e2d0faf3.jpg)  
Figure 1: Spatiotemporal dynamic analysis of video generation under different motion priors (left: camera panning; right: subject motion). (a) Real-world scenes. The videos are synthesized via Self Forcing, conditioned on a single text prompt provided at the initialization stage. (b) Attention distributions. Visualizations reveal the distribution of temporal attention scores, indicating that the generation process predominantly focuses on active dynamic elements alongside adjacent temporal context. (c) Cosine similarity between the currently generated frame and preceding frames. The consistently high similarity observed in static background regions demonstrates the potential for extensive feature reuse during computation.

To address these challenges, we propose Dynamics-Aware Sparse Attention (DASA), an efficient framework for AR video generation. Inspired by modern video coding principles [59, 33], we tackle both memory and computational bottlenecks through a unified dynamic and static separation perspective. From a storage standpoint, our analysis reveals that AR models primarily attend to historical dynamic elements and immediate temporal contexts (Figure 1 (b)). Based on this observation, we introduce a Frame-wise Memory Management module that employs layer-wise similarity pruning and a static token refresh strategy. This allows the framework to aggressively compress the KV cache while preserving essential high-frequency details. From a computational perspective, we implement a Region-level Computation Allocation strategy. By identifying static regions that exhibit minimal feature variation across chunks (Figure 1 (c)), we utilize an early-exit mechanism to bypass redundant forward passes, focusing resources exclusively on dynamic updates. By integrating these mechanisms, our framework curtails the KV cache memory footprint while reducing the computational overhead during chunk-wise video generation.

While these mechanisms are inherently training-free, pursuing higher acceleration ratios and framework scalability necessitates their integration into the training phase. However, directly exposing the network to static-feature reuse during training creates an optimization shortcut, resulting in motion sluggishness. To counteract this degradation, we introduce a Dynamics-guided Distillation strategy. By leveraging motion priors, we apply a spatially varying loss reweighting that explicitly amplifies the learning penalty in dynamic regions, successfully preserving the natural dynamics of the generated videos.

In summary, our main contributions are as follows:

• We identify that current autoregressive video diffusion suffers from inefficient resource allocation across highly redundant sequences. We observe that the generation process predominantly focuses on active dynamic elements, while static backgrounds exhibit extensive feature redundancy across adjacent chunks.

• We propose Dynamics-Aware Sparse Attention (DASA) to eliminate spatiotemporal redundancy from both storage and computation perspectives. By integrating adaptive memory management, motion-aware computation skipping, and dynamics-guided distillation, DASA facilitates scalable video synthesis while avoiding static degradation.

• Extensive experiments demonstrate DASA’s effectiveness in accelerating autoregressive video synthesis while preserving high visual fidelity, achieving a 2.16× inference speedup at 1080P with a minimal quality drop of less than 0.2%.

## 2 Related Work

## 2.1 Video Diffusion Models

The pursuit of ultra-long and interactive video generation [23, 19, 16, 7, 8, 57, 6, 53, 5, 18, 28, 38] has been fundamentally shaped by the struggle against computational and memory bottlenecks. Early video diffusion models [40, 26, 65, 17] relied on full bidirectional attention to generate short clips, but the quadratic complexity of attention mechanisms imposed a strict ceiling on video length. To circumvent this, the community shifted toward autoregressive diffusion [10, 11, 58, 36], which decomposed image-to-video generation into either frame-wise [62, 43] or chunk-wise [1] steps. Approaches like Causvid [56] apply DMD [54] to distill 50-step diffusion model into a 4-step generator, enabling faster streaming generation. However, scaling this autoregressive paradigm for long-term consistency requires caching extensive historical tokens, which inevitably triggers an unsustainable memory explosion. Consequently, recent models [20, 50] often employ a sliding window strategy as a trade-off to maintain a fixed-length memory. Simultaneously, to address the exposure bias in long rollouts, various training-inference alignment mechanisms were proposed, including Diffusion-Forcing [56], Self-Forcing [20], Rolling-Forcing [31], and Causal-Forcing [66]. Currently, this trajectory has culminated in interactive world models [3, 42, 39, 22, 61]. For these world models, efficiency is no longer merely about generating faster, but rather a strict prerequisite to achieving real-time interaction , necessitating a fundamental rethinking of how foundational models allocate computational resources.

## 2.2 Efficient Autoregressive Video Generation

To meet the stringent efficiency demands of AR video generation, recent efforts have explored various acceleration strategies, which broadly fall into two directions. The first direction focuses on altering the generation paradigm or introducing auxiliary modules to reduce computational latency. For instance, MAGI-1 [1] breaks sequential bottlenecks by enabling the parallel generation of multiple video chunks with varying noise levels. Other works focus on context compression. For example, Framepack [62] compresses historical frames using fixed patterns based on frame-wise importance, while LoViC [24] introduces an autoencoder to compress historical contexts into compact representations.

The second direction accelerates inference by exploiting the model’s intrinsic properties, predominantly through attention sparsification. Specifically, Dummy Forcing [12] prunes inactive attention heads during generation. Light Forcing [32] introduces a progressive, chunk-aware sparsity allocation and hierarchical sparse attention to capture information in a coarse-to-fine manner. Furthermore, Sparse Forcing [48] integrates sparsity into the training phase, motivated by the empirical observation that attention concentrates on salient visual blocks. Yet, it primarily optimizes storage and computation on the KV cache side, lacking optimization for the current query computation.

Fundamentally, while existing approaches primarily derive acceleration strategies from model-centric observations like attention distributions or structural priors [45, 63, 60, 51, 49, 15, 14, 37, 46, 64], they often overlook the intrinsic physical nature of the video data itself. We argue that a more natural source of efficiency lies in the data-centric characteristics: video motion is inherently sparse and highly concentrated, whereas the background remains largely static.

## 3 Method

To address the redundancy inherent in long video generation, we present Dynamics-Aware Sparse Attention, a unified framework that jointly optimizes AR video generation from the perspectives of memory storage and computational allocation. The key idea is to exploit the temporal redundancy of video by explicitly distinguishing dynamic regions. We first introduce our Frame-wise Memory Management in Sec 3.1, which alleviates memory bottlenecks by selectively retaining dynamic features in the historical KV cache, together with a static refresh mechanism to maintain long-range consistency. We then describe Region-level Computation Allocation in Sec 3.2, which reduces redundant computation by assigning different processing budgets to static and dynamic regions within each generated chunk. Finally, in Sec 3.3, we incorporate these sparse generation strategies into training through Dynamics-Guided Distillation Adaptation, a lightweight distillation procedure that mitigates the potential static bias introduced by sparsification and improves robustness across generative settings. Figure 2 provides an overview of the proposed framework.

![](images/e68f24b42bd472a66abad5687aa2cd63317ee92bd6bc602135bcbdb90239f848.jpg)  
Figure 2: Overall architecture of the proposed framework. We introduce a unified approach to optimize AR diffusion generation for both storage and computational efficiency. Storage Management (Left): Tokens from historical chunks are selectively cached in memory based on a layer-wise threshold $\sigma _ { l } ,$ , discarding redundant information. The final frame of the immediately preceding chunk serves as a referenceframe, with all its tokens unconditionally preserved. Computation Optimization (Right): This reference frame acts as an anchor to guide token selection for the current chunk<sub>C</sub>. Redundant tokens are filtered out via a threshold γ to reduce computational overhead. AR Diffusion Generation (Middle): Finally, the retained historical tokens from memory and the active tokens from the current chunk are concatenated and processed by the AR diffusion.

## 3.1 Frame-wise Memory Management

Due to the high spatial-temporal redundancy in continuous video streams, storing all tokens indiscriminately not only incurs severe memory footprint but is also fundamentally unnecessary. Current AR video models typically employ a sliding window approach [56, 20], sometimes paired with sink frames [50], to constrain memory by retaining only recent context. However, rigidly discarding older memory creates severe temporal inconsistencies in interactive world generation, especially when navigating back to past scenes. To overcome this limitation, rather than simply dropping old frames, we propose a Frame-wise Memory Management module, which dynamically regulates token caching through similarity-based pruning and a static token refresh strategy. We provide more algorithm details in Appendix A.1.

Layer-wise retention ratio. Motivated video coding standards like H.264, which utilize Intra-coded and Predictive-coded frames, we determine whether a token should be retained in memory based on its feature similarity with its adjacent frame. Specifically, let $\boldsymbol { x } _ { t } ^ { k }$ denote the latent representation of the k-th spatial token at frame t. We compute the cosine similarity between temporally adjacent tokens:

$$
{ \mathrm { s i m } } _ { t } ^ { k } = \cos \left( { \boldsymbol x } _ { t } ^ { k } , { \boldsymbol x } _ { t + 1 } ^ { k } \right) .\tag{1}
$$

Unlike H.264, which uses the previous frame as a reference, our approach retains the last generated frame as the reference frame and save all its tokens. This strategy results in significantly smoother generated videos. We hypothesize that this entire frame serves as an anchor for the newly generated chunk, which aligns with our observations in Figure 1 (b). If a token’s representation exhibits minimal changes compared to its latter frame, it is deemed redundant and can be safely discarded from the memory buffer. Thus, a preliminary retention mask is generated by thresholding the similarity: $M _ { \mathrm { s i m } } ^ { k } = \mathbb { I } \left( \mathrm { s i m } _ { t } ^ { k } < \sigma \right)$ , where I(·) is the indicator function and $\sigma$ is the threshold.

As visualized in the Appendix C.1, we observe that the models attend to more dynamic tokens as the layer goes deeper. We assume that shallow layers capture high-frequency details that change rapidly, while deeper layers extract abstract, low-frequency semantic information that remains highly stable across frames. Leveraging this property, we apply a layer-wise retention rate by linearly decaying the threshold $\sigma _ { l }$ for layer l as the layer depth increases:

$$
\sigma _ { l } = \sigma _ { m a x } - \frac { \sigma _ { m a x } - \sigma _ { m i n } } { L } \cdot { l }\tag{2}
$$

where $\sigma _ { m a x }$ and $\sigma _ { m i n }$ are the predefined upper and lower bounds of the threshold, and L is the total number of layers. A lower threshold in deeper layers enforces a stricter retention condition, further compressing the memory footprint.

Static token refresh strategy. While similarity-based pruning significantly reduces redundancy, we observe temporal degradation in static regions as they are continuously pruned. If a background token remains similar across multiple frames and is repeatedly dropped, the model is forced to rely on a heavily outdated token. Consequently, the model fails to capture subtle temporal variations—such as global illumination shifts—leading to error accumulation that severely degrades overall visual fidelity.

To mitigate this, we introduce a Static Token Refresh strategy. We maintain an aging counter $\Delta t ^ { k }$ for each spatial position k, defined as the number of frames elapsed since a token was last written to the memory at its location: $\Delta t ^ { k } = t - t _ { \mathrm { w r i t e } } ^ { k }$ . If a token at a specific location has been continuously dropped for a predefined maximum age threshold $\tau _ { : }$ , we force the model to retain this token. The final boolean mask $M _ { f i n a l } ^ { k } .$ , which determines whether the k-th token is saved to the memory bank, is defined as the union of the similarity and aging conditions:

$$
M _ { \mathrm { f i n a l } } ^ { k } = M _ { \mathrm { s i m } } ^ { k } \vee \mathbb { I } \left( \Delta t ^ { k } \geq \tau \right) .\tag{3}
$$

When $M _ { \mathrm { f i n a l } } ^ { k } = 1$ , the token is stored, and its timestamp $t _ { \mathrm { w r i t e } } ^ { k }$ is reset to the current frame t. This mandatory refresh acts as a periodic calibration, effectively preventing the fading of high-frequency background details while keeping the overall memory growth well-controlled.

## 3.2 Region-level Computation Allocation

Unlike traditional video diffusion models [40, 26] that compute full spatial-temporal attention across the entire video sequence simultaneously, AR diffusion models adopt a chunk-wise generation paradigm. While this can efficiently scale to long video generation, we observe a significant computational bottleneck: the standard diffusion process iteratively denoises all tokens within a chunk, ignoring the strong inter-chunk temporal coherence. In reality, the current generating chunk shares substantial similarities with the preceding synthesized chunk, particularly in backgrounds and smoothly moving subjects. To effectively eliminate this redundant computation, we propose a Region-level Computation Allocation strategy that dynamically reuses previously computed features from the preceding chunk, thereby skipping unnecessary computations for highly redundant regions. More algorithm details are provided in Appendix A.2.

Motion-aware early exit. In the first denoising step of the current chunk $C ,$ the model predicts an estimated clean state $\hat { X } _ { C }$ . We leverage this intermediate $\hat { X } _ { C }$ to evaluate temporal redundancy against the previously finalized chunk $X _ { C - 1 }$ . Since physical motion in videos is typically local and continuous, we perform a local alignment search rather than comparing strictly identical spatial coordinates.

Specifically, for a predicted token $\hat { \pmb { x } } _ { C } ^ { ( i , j ) }$ at spatial coordinate $( i , j )$ in chunk $C ,$ , we search for its best matching token within a local w × w spatial window $\mathcal { N } _ { w } ( i , j )$ in the previous chunk $C - 1$ . The aligned token $\tilde { \pmb { x } } _ { C - 1 } ^ { ( i , j ) }$ and its corresponding maximum similarity score sim $1 _ { C } ^ { ( i , j ) }$ are defined as:

$$
\tilde { x } _ { C - 1 } ^ { ( i , j ) } = x _ { C - 1 } ^ { { ( u ^ { * } , v ^ { * } ) } } , \quad \mathrm { w h e r e } \quad ( u ^ { * } , v ^ { * } ) = \underset { ( u , v ) \in \mathcal { N } _ { w } ( i , j ) } { \arg \operatorname* { m a x } } \cos \left( \hat { x } _ { C } ^ { ( i , j ) } , x _ { C - 1 } ^ { ( u , v ) } \right) ,\tag{4}
$$

and sim $\mathbf { \mathbf { \mathbf { \mathbf { \mathbf { \mathbf { \mathbf { \Lambda } } } } } } } _ { C } ^ { ( i , j ) } = \cos \left( \hat { \mathbf { \mathbf { \mathbf { x } } } } _ { C } ^ { ( i , j ) } , \tilde { \mathbf { \mathbf { x } } } _ { C - 1 } ^ { ( i , j ) } \right)$ . This local search efficiently tracks moving objects, while static backgrounds naturally align at the exact same coordinate $( i , j )$ . Furthermore, by keeping the window size w as a small fixed constant, this local search avoids the quadratic $\mathcal { O } ( L ^ { 2 } )$ complexity of a global matching process (where L is the spatial sequence length). Instead, it scales linearly with $L ,$ ensuring the additional alignment overhead remains minimal.

Based on the alignment results, a token is classified as redundant if its maximum similarity exceeds a predefined threshold $\gamma .$ . This yields a binary mask $M \colon$

$$
M ^ { ( i , j ) } = \mathbb { I } \left( \sin _ { C } ^ { ( i , j ) } \geq \gamma \right) .\tag{5}
$$

For these redundant tokens, we halt their subsequent denoising iterations—triggering an early exit—and directly copy the corresponding aligned tokens from the previous chunk. Let $\tilde { X } _ { C - 1 }$ denote the feature map composed of all aligned tokens. The output latent $\hat { X } _ { C } ^ { \mathrm { o u t } }$ is updated via spatial blending:

$$
\hat { X } _ { C } ^ { \mathrm { o u t } } = M \odot \tilde { X } _ { C - 1 } + ( 1 - M ) \odot \mathcal { F } _ { \theta } ( \hat { X } _ { C } )\tag{6}
$$

where $\mathcal { F } _ { \theta } ( \cdot )$ denotes the rest of the denoising forward passes applied exclusively to the remaining unaligned (dynamic) tokens. This streamlined process securely bypasses heavy, unnecessary forward passes for visually redundant areas without compromising generation quality.

## 3.3 Dynamics-Guided Distillation Adaptation

While our inference technique efficiently accelerates generation and maintains overall quality, minor domain shifts or artifacts can still occur. To further elevate visual quality, we integrate the aforementioned mechanisms into the training phase. However, during this native adaptation, we find that while the spatial quality of the generated videos improves, they suffer from motion lag, causing actions and object movements to become noticeably restricted.

We attribute this degradation to an optimization shortcut. When the region-level static scene reuse mechanism is exposed during training, the network discovers a trivial solution to minimize the reconstruction error. Since predicting static or minimal changes yields a safely lower loss than taking the risk to generate complex mo-

![](images/e0ead541c706f0906d8ba2300a80c14db3bb3ca77ebe71bc9d63147c533bdc2e.jpg)  
Figure 3: Dynamics-guided Distribution Matching Distillation. To mitigate static bias, we introduce a $\mathrm { d y } .$ namic weight mask based on temporal cosine similarity. By amplifying loss in high-motion regions, it penalizes static predictions and forces the model to learn complex dynamics.

tions, the model develops a “static bias.” It over-relies on copying historical tokens and inadvertently suppresses its internal capability to generate dynamic motions.

To counteract this static bias without discarding the efficiency gains, we introduce a Dynamics-Guided Distillation Adaptation built upon the standard Distribution Matching Distillation [55] framework. We leverage the robust motion priors from the data sample to explicitly penalize static predictions. Specifically, let $\boldsymbol { z } _ { t } ^ { ( i , j ) }$ denote the token representation of the sample video at frame t and spatial coordinate $( i , j )$ . We compute the temporal similarity sim $\mathbf { \boldsymbol { 1 } } _ { t } ^ { ( i , j ) }$ between adjacent frames using the previously defined cosine similarity. To identify areas of significant change and penalize “lazy” predictions, we map this similarity into a spatially-varying dynamic weight mask $W _ { t } ^ { ( i , j ) }$

$$
\boldsymbol { W } _ { t } ^ { ( i , j ) } = 1 + \alpha \cdot \operatorname* { m a x } \left( 0 , 1 - \sin _ { t } ^ { ( i , j ) } \right) ,\tag{7}
$$

where α is a dynamic scaling factor.

Finally, this weight mask is applied token-wise to the standard DMD loss $\mathcal { L } _ { \mathrm { b a s e } }$ . The proposed dynamics-guided distillation loss $\mathcal { L } _ { \mathrm { d y n a m i c } }$ is formulated as:

$$
\mathcal { L } _ { \mathrm { d y n a m i c } } = \frac { 1 } { N } \sum _ { t , i , j \in \Omega } W _ { t } ^ { ( i , j ) } \cdot \mathcal { L } _ { \mathrm { b a s e } } ^ { ( t , i , j ) } ,\tag{8}
$$

where N denotes the total number of tokens. Regions exhibiting larger temporal variations in the loss weight. This dynamic guidance acts as a strong penalty against the Student model’s optimization shortcut, forcing it to faithfully reconstruct the real data’s motion dynamics in the generated videos.

## 4 Experiments

## 4.1 Experimental Setup

Implementation details. We integrate our approach into representative baselines, including Self-Forcing [20] and LongLive [50], and further finetune them to enhance sample quality and generation speed. To comprehensively validate our method, our experimental scope spans multiple settings: standard 5s video generation, 30s long video generation, and high-resolution (780P and 1080P) video generation. All efficiency experiments are conducted on NVIDIA H100 GPUs. Due to page limit, we provide more implementation details in the Appendix B.

Evaluation. We evaluate our method on VBench [21], a comprehensive video generation benchmark. Specifically, we adopt the extended version of its 946 evaluation prompts introduced by [20]. Following standard practice, we sample 5 videos per prompt and assess the quality score and semantic score across all 16 quantitative dimensions defined in VBench. All evaluation experiments are conducted on NVIDIA H20 GPUs.

Table 1: Quantitative results for 5-second short video generation on VBench [21]. The best results are in bold, and the second best are underlined. The color box denotes the baseline model.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td rowspan="2">Params</td><td colspan="2">Efficiency</td><td colspan="3">Visual Quality</td></tr><tr><td>Throughput (FPS)↑</td><td>Speedup↑</td><td>Quality Score↑</td><td>Semantic Score↑</td><td>Total Score↑</td></tr><tr><td rowspan="5">Diffusion</td><td>LTX-Video [13]</td><td>1.9B</td><td>8.98</td><td></td><td>82.30</td><td>70.79</td><td>80.00</td></tr><tr><td>CogVideoX [52]</td><td>5B</td><td>0.85</td><td></td><td>82.68</td><td>79.48</td><td>82.04</td></tr><tr><td>Wan2.1 [41]</td><td>1.3B</td><td>0.78</td><td></td><td>85.30</td><td>80.09</td><td>84.26</td></tr><tr><td>Open-Sora [65]</td><td>11B</td><td>0.08</td><td></td><td>84.28</td><td>79.89</td><td>83.40</td></tr><tr><td>Hunyuan Video [26]</td><td>13B</td><td>0.22</td><td></td><td>85.07</td><td>76.88</td><td>83.43</td></tr><tr><td rowspan="14">Autoregressive</td><td>SkyReels-V2 [4]</td><td>1.3B</td><td>0.49</td><td></td><td>84.70</td><td>74.53</td><td>82.67</td></tr><tr><td>MAGI-1 [1]</td><td>4.5B</td><td>0.19</td><td></td><td>82.04</td><td>67.74</td><td>79.18</td></tr><tr><td>NOVA [9]</td><td>0.6B</td><td>0.88</td><td></td><td>80.39</td><td>79.05</td><td>80.12</td></tr><tr><td>Pyramid Flow [25]</td><td>2B</td><td>6.77</td><td></td><td>84.74</td><td>69.62</td><td>81.72</td></tr><tr><td>CausVid [56]</td><td>1.3B</td><td>0.69</td><td></td><td>83.73</td><td>78.49</td><td>82.69</td></tr><tr><td>Self Forcing [20]</td><td>1.3B</td><td>18.42</td><td>1.00×</td><td>84.97</td><td>81.06</td><td>84.19</td></tr><tr><td>TeaCache [30]</td><td>1.3B</td><td>21.69</td><td>1.18×</td><td>84.22</td><td>80.47</td><td>83.47</td></tr><tr><td>Dummy Forcing [12]</td><td>1.3B</td><td>25.19</td><td>1.37×</td><td>84.47</td><td>81.08</td><td>83.79</td></tr><tr><td>DASA</td><td>1.3B</td><td>26.21</td><td>1.42×</td><td>84.57</td><td>80.79</td><td>83.81</td></tr><tr><td>DASA w/ adaptation</td><td>1.3B</td><td>27.58</td><td>1.50×</td><td>84.64</td><td>81.32</td><td>83.98</td></tr><tr><td>LongLive [50]</td><td>1.3B</td><td>18.30</td><td>1.00×</td><td>83.77</td><td>80.73</td><td>83.16</td></tr><tr><td>TeaCache [30]</td><td>1.3B</td><td>21.82</td><td>1.19×</td><td>83.10</td><td>80.73</td><td>82.63</td></tr><tr><td>Dummy Forcing [12]</td><td>1.3B</td><td>25.10</td><td>1.37×</td><td>83.16 83.18</td><td>80.99</td><td>82.73</td></tr><tr><td>DASA</td><td>1.3B</td><td>26.50</td><td>1.45×</td><td></td><td>80.87</td><td>82.72</td></tr><tr><td>DASA w/ adaptation</td><td>1.3B</td><td>27.91</td><td>1.53×</td><td>83.78</td><td>80.35</td><td>83.09</td></tr></table>

## 4.2 Main Results

## 4.2.1 Short Video Generation

This subsection evaluates our method on standard 5-second video generation using the VBench dataset, as presented in Table 1. We integrate DASA into state-of-the-art AR diffusion models and compare it with TeaCache [30] and Dummy Forcing [12]. Since modern AR diffusion models typically use a low-step generation paradigm of 4 denoising steps, cross-step caching methods like TeaCache face inherent acceleration limits and yield a marginal speedup of 1.18×. While Dummy Forcing improves this by dropping attention heads based on the first generated chunk, its static pruning pattern may lead to information loss and degraded visual quality.

Unlike these baselines that rely on structural or step-wise approximations, our method exploits the inherent spatial-temporal redundancy of videos to dynamically allocate computational resources based on content complexity. By adaptively assigning more resources to highly dynamic regions while bypassing static areas, our adaptive pipeline achieves a speedup of 1.53×, consistently preserving the high visual and semantic quality. Furthermore, it is worth noting that our method optimizes AR diffusion models at a different level, making it orthogonal and compatible with TeaCache and Dummy Forcing.

![](images/ed36f320163add34cdb70ec1b3fbce63c6b0ed4401e8b89f67b92a503e239d62.jpg)

![](images/72b65e95487a6ffcacbca887f9b9b0266bac3d358d991c3423c83817303ce474.jpg)  
Figure 4: Ablation study on the impact of early-exit threshold γ and maximum age threshold τ .

Table 2: Quantitative results for 30-second long video and high-resolution video generation on VBench [21].
<table><tr><td rowspan="2">Model</td><td colspan="2">Efficiency</td><td colspan="3">Visual Quality</td></tr><tr><td>Throughput (FPS)↑</td><td>Speedup↑</td><td>Quality Score↑</td><td>Semantic Score↑</td><td>Total Score↑</td></tr><tr><td>SkyReels-V2 [4]</td><td>0.49</td><td></td><td>80.77</td><td>53.37</td><td>75.29</td></tr><tr><td>FramePack [62]</td><td>0.92</td><td></td><td>83.61</td><td>75.32</td><td>81.95</td></tr><tr><td colspan="6">30s Video Generation</td></tr><tr><td>Self Forcing</td><td>18.42</td><td>1.00×</td><td>84.15</td><td>80.24</td><td>83.37</td></tr><tr><td>Dummy Forcing DASA</td><td>25.19</td><td>1.37×</td><td>84.11</td><td>79.18</td><td>83.12</td></tr><tr><td></td><td>26.21</td><td>1.42×</td><td>84.05</td><td>79.78</td><td>83.20</td></tr><tr><td>LongLive Dummy Forcing</td><td>18.30</td><td>1.00×</td><td>83.57</td><td>80.96</td><td>83.05</td></tr><tr><td>DASA</td><td>25.10 26.50</td><td>1.37× 1.45×</td><td>83.23 83.39</td><td>79.59 80.59</td><td>82.50 82.83</td></tr><tr><td colspan="6">720P Video Generation</td></tr><tr><td>Self Forcing</td><td>5.92</td><td>1.00×</td><td>84.30</td><td>81.21</td><td>83.68</td></tr><tr><td>Dummy Forcing DASA</td><td>9.52</td><td>1.61×</td><td>84.16</td><td>80.86</td><td>83.50</td></tr><tr><td></td><td>10.10</td><td>1.71×</td><td>84.40</td><td>80.92</td><td>83.71</td></tr><tr><td>LongLive</td><td>5.98</td><td>1.00×</td><td>83.76</td><td>81.50</td><td>83.31</td></tr><tr><td>Dummy Forcing DASA</td><td>9.74 10.37</td><td>1.63× 1.73×</td><td>83.49 83.38</td><td>81.63 81.54</td><td>83.12 83.01</td></tr><tr><td colspan="6">1080P Video Generation</td></tr><tr><td>Self Forcing</td><td>1.35</td><td>1.00×</td><td>84.12</td><td>79.42</td><td></td></tr><tr><td>Dummy Forcing</td><td>2.64</td><td>1.96×</td><td>83.10</td><td>79.89</td><td>83.18 82.46</td></tr><tr><td>DASA</td><td>2.81</td><td>2.08×</td><td>83.55</td><td>79.73</td><td>82.79</td></tr><tr><td>LongLive</td><td>1.36</td><td>1.00×</td><td>82.13</td><td>80.40</td><td>81.78</td></tr><tr><td>Dummy Forcing</td><td>2.74</td><td>2.02×</td><td>81.92</td><td>80.24</td><td>81.58</td></tr><tr><td>DASA</td><td>2.94</td><td>2.16×</td><td>81.90</td><td>80.55</td><td>81.63</td></tr></table>

## 4.2.2 Long Video Generation

Crucially, we highlight the scalability of our approach in 30-second long video generation, with quantitative results presented in Table 2. By maintaining contextual coherence over extended frames, DASA seamlessly scales to longer sequences. When integrated with baselines like Self Forcing and LongLive, our method achieves speedups of up to 1.45×, demonstrating its capability to accelerate extended generation without compromising overall semantic integrity.

## 4.2.3 High-resolution Video Generation

To further validate the robustness of our method under high computational demands, we extend our evaluation to 720P and 1080P high-resolution generation, as detailed in Table 2. Notably, the acceleration benefits of DASA scale with resolution. Our approach delivers speedups of up to 2.16×

at 1080P. Despite massive reductions in memory and computational overhead, DASA successfully preserves superior visual fidelity and fine spatial details, yielding good trade-off between throughput and total video quality.

## 4.3 Ablation Study

Effect of memory and computation retention. We evaluate two thresholds controlling spatialtemporal redundancy: memory retention σ and early-exit γ. To avoid a rigid speed-quality trade-off, we adopt a dynamic layer-wise σ (denser in shallow layers, sparser in deep ones), which successfully reduces memory footprint while preserving fidelity, as detailed in Table 3. For the Motion-Aware Early Exit, Figure 4 (a) demonstrates that γ dictates the speed-quality balance; an optimal $\gamma = 0 . 9 2$ achieves a 1.42× speedup with a competitive Vbench total score of 83.98. In combination, these approaches bypass unnecessary memory operations and computational overhead.

Effect of static token refresh. As shown in the Figure 4 (b), increasing τ boosts throughput, but excessively large values $( \tau \geq 6 )$ severely degrade visual quality. This confirms our premise that unconstrained token dropping causes severe error accumulation. Conversely, the baseline $( \tau = 1 )$ preserves details but limits acceleration. We select $\tau = 4$ as the optimal balance, achieving a 1.42× speedup with highly competitive visual fidelity.

Table 3: Ablation study on the impact of memory retention ratio σ.
<table><tr><td colspan="2">σ</td><td rowspan="2">Throughput (FPS)↑</td><td rowspan="2">Quality Score↑</td><td rowspan="2">Semantic Score↑</td><td rowspan="2">Total Score↑</td></tr><tr><td> $\sigma _ { \mathrm { m a x } }$ </td><td> $\sigma _ { \mathrm { m i n } }$ </td></tr><tr><td>Self Forcing</td><td></td><td>18.42</td><td>84.97</td><td>81.06</td><td>84.19</td></tr><tr><td>0.8</td><td>0.8</td><td>22.34 (1.21×)</td><td>84.89</td><td>81.00</td><td>84.11</td></tr><tr><td>0.6</td><td>0.6</td><td>24.11 (1.31×)</td><td>84.12</td><td>80.04</td><td>83.30</td></tr><tr><td>0.8</td><td>0.2</td><td>26.21 (1.42×)</td><td>84.57</td><td>80.79</td><td>83.81</td></tr></table>

Table 4: Ablation study on the impact of local alignment and adaptation.
<table><tr><td></td><td>Throughput (FPS)↑</td><td>Quality Score↑</td><td>Semantic Score↑</td><td>Total Score↑</td></tr><tr><td>Self Forcing</td><td>18.42</td><td>84.97</td><td>81.06</td><td>84.19</td></tr><tr><td>w/ align.</td><td>26.21 (1.42×)</td><td>84.57</td><td>80.77</td><td>83.81</td></tr><tr><td>w/ adapt.</td><td>26.32 (1.43×)</td><td>84.70</td><td>81.24</td><td>84.01</td></tr><tr><td>ours</td><td>27.58 (1.50×)</td><td>84.64</td><td>81.32</td><td>83.98</td></tr></table>

Effect of local alignment. As shown in Table 4, by dynamically identifying and reusing redundant tokens via local window search, our motion-aware early exit bypasses unnecessary denoising tokens, yielding a significant 1.42× throughput increase.

Effect of dynamics-guided adaptation. Introducing the region-reuse mechanism during training can induce a "static bias," restricting motion generation. Our dynamics-guided adaptation explicitly penalizes these static predictions. As shown in Table 4, the dynamics-guided adaptation effectively mitigates the motion degradation, achieving the highest Semantic Score by preserving authentic dynamics. Furthermore, this harmonized training boosts the inference throughput to 27.58 FPS (1.50×), validating that our design achieves a desirable balance between efficiency and visual fidelity.

## 5 Conclusion

In this work, we explore the intrinsic dynamics and spatial-temporal redundancies in autoregressive long video generation. To exploit this nature for faster generation, we decompose this redundancy from two orthogonal perspectives: memory footprint and computational overhead. Specifically, we introduce a Frame-wise Memory Management module to dynamically prune temporally redundant tokens and strictly bound KV cache growth. Concurrently, we propose a Region-level Computation Allocation scheme that utilizes a motion-aware early exit mechanism to bypass unnecessary denoising computations for static backgrounds. Extensive experiments demonstrate that our token-level optimizations significantly accelerate video synthesis while maintaining high visual fidelity.

Limitations and future work. While our approach effectively reduces spatial-temporal redundancy at the token level, it currently does not explore redundancies across other architectural dimensions. Notably, our method is fundamentally orthogonal and highly compatible with existing state-of-the-art acceleration techniques. For instance, TeaCache exploits redundancies across the denoising step dimension, whereas Dummy Forcing accelerates generation from the attention head dimension. In future work, we plan to systematically integrate our token-level strategy with step-wise TeaCache and head-wise Dummy Forcing. We anticipate that unifying these three orthogonal dimensions will unlock unprecedented synergistic acceleration ratios for autoregressive video generation.

## A Algorithm Details

This part provides the precise algorithmic formulations of our proposed optimization framework, complementing the conceptual methodology described in the main text. To explicitly demonstrate how we exploit spatial-temporal redundancy for efficient long video generation, we detail the execution pipelines of our two core modules. First, Algorithm 1 formalizes the Frame-wise Memory Management strategy, illustrating the dynamic KV cache pruning and static refresh mechanisms. Second, Algorithm 2 outlines the Region-level Computation Allocation scheme, showing how the motion-aware early exit mechanism is practically implemented during the chunk-wise denoising loop.

## A.1 Memory Management

Algorithm 1 presents the core computational workflow of our Frame-wise Memory Management module, which dynamically optimizes the KV cache for efficient long video generation. The algorithm executes the methodology described in the main text, primarily integrating layer-wise retention threshold decay, similarity-based temporal redundancy pruning, and a mandatory static token refresh strategy. By synergistically applying these mechanisms, our approach reduces memory footprint while effectively preventing the temporal degradation of background details, ensuring high overall visual fidelity.

Algorithm 1 Frame-wise Memory Management for Video Generation   
Input: Latent tensor $X \in \mathbb { R } ^ { B \times N _ { f } \times N _ { s } \times D }$ , Keys K, Values $V$   
Input: Layer index l, Total layers $L ,$ Sim thresholds $\sigma _ { \mathrm { m a x } } , \sigma _ { \mathrm { m i n } } ,$ , Max token age $\tau$   
Input: Reference frame $X _ { \mathrm { r e f } } ,$ Token age tracker $T _ { \mathrm { u p d a t e } } ,$ Current frame num $t _ { \mathrm { c u r r } }$   
Output: Output tensor O, Updated KV Cache state   
1: // 1. Layer-wise Retention Threshold Decay   
2: $\begin{array} { r } { \sigma _ { l } \gets \sigma _ { \mathrm { m a x } } - \frac { \sigma _ { \mathrm { m a x } } - \sigma _ { \mathrm { m i n } } } { L } \cdot ( l + 1 ) } \end{array}$   
3: // 2. Temporal Context Concatenation   
4: if $X _ { \mathrm { r e f } }$ is not null then   
5: $\overrightharpoon { X } _ { \mathrm { c o m b } } \gets \mathrm { C o n c a t } ( X _ { \mathrm { r e f } } , X , \mathrm { d i m } = 1 )$   
6: $K _ { \mathrm { c o m b } } \gets \mathrm { C o n c a t } ( K _ { \mathrm { r e f } } , K , \mathrm { d i m = 1 } )$   
7: $V _ { \mathrm { c o m b } } \gets \mathrm { C o n c a t } ( V _ { \mathrm { r e f } } , V , \mathrm { d i m } = 1 )$   
8: else   
9: $X _ { \mathrm { c o m b } }  X , \quad K _ { \mathrm { c o m b } }  K , \quad V _ { \mathrm { c o m b } }  V$   
10: end if   
11: // 3. Similarity-based Token Pruning   
12: $X _ { \mathrm { p r e v } }  X _ { \mathrm { c o m b } } [ : , : - N _ { \mathrm { r e f } } , \dotsc ]$   
13: $X _ { \mathrm { n e x t } } ^ { \cdot }  X _ { \mathrm { c o m b } } [ : , 1 : ( - N _ { \mathrm { r e f } } + 1 ) , \dots ]$   
14: sim ← CosineSimilarity $( X _ { \mathrm { p r e v } } , X _ { \mathrm { n e x t } } ,$ dim = −1)   
15: $M _ { \mathrm { s i m } } \gets \mathrm { s i m } < \sigma _ { l }$ // Preliminary boolean mask based on similarity   
16: // 4. Static Token Refresh Strategy   
17: for each frame index $t \in \{ 1 , \ldots , N _ { f } \}$ do   
18: $M _ { \mathrm { c u r r } } \gets M _ { \mathrm { s i m } } [ : , t ]$   
19: $\Delta t \gets t _ { \mathrm { c u r r } } - \dot { T } _ { \mathrm { u } }$ pdate // Calculate token age   
20: $M _ { \mathrm { a g e } }  \Delta t \geq \dot { \tau }$   
21: $M _ { \mathrm { f i n a l } } [ : , t ] \gets M _ { \mathrm { c u r r } } \vee M _ { \mathrm { a g e } }$ // Force refresh if token is too old   
22: $T _ { \mathrm { u p d a t e } } [ M _ { \mathrm { f i n a l } } [ : , t ] ] \gets t _ { \mathrm { c u r r } }$ // Reset aging counter for retained tokens   
23: $t _ { \mathrm { c u r r } } \gets t _ { \mathrm { c u r r } } + 1$   
24: end for   
25: // 5. Reference Frame Preservation & Cache Update   
26: $M _ { \mathrm { t a i l } }  \mathbf { 1 } \in \{ 0 , 1 \} ^ { B \times N _ { \mathrm { r e f } } \times N _ { s } }$ // Unconditionally retain the last frame(s)   
27: $M _ { \mathrm { a l l } }  \mathrm { C o n c a t } ( M _ { \mathrm { f i n a l } } , M _ { \mathrm { t a i l } } , \mathrm { d i m } = 1 )$   
28: $K _ { \mathrm { s a v e } }  K _ { \mathrm { c o m b } } [ M _ { \mathrm { a l l } } ] , \quad V _ { \mathrm { s a v e } }  V _ { \mathrm { c o m b } } [ M _ { \mathrm { a l l } } ]$ // Extract retained tokens via masking   
29: KVCache $. \mathrm { A l }$ ppend $( K _ { \mathrm { s a v e } } , V _ { \mathrm { s a v e } } )$   
30: $X _ { \mathrm { r e f } }  X [ : , - N _ { \mathrm { r e f } } : , \dots ]$ // Update reference frame for the next chunk   
31: // 6. Standard Attention Computation with Compact Cache   
32: O ← Attention(Q, KVCache , KVCache )   
33: return O

## A.2 Computation Allocation

Algorithm 2 details the execution pipeline of our Region-level Computation Allocation strategy, specifically highlighting the Motion-Aware Early Exit mechanism. Operating within a chunk-wise generation framework, the algorithm leverages the first denoising step to perform a local spatialtemporal alignment between the current chunk and the preceding finalized chunk. By extracting a binary mask via thresholding and spatial dilation, it accurately segregates dynamic tokens from static or redundant backgrounds. In all subsequent denoising iterations, redundant tokens securely bypass the heavy forward passes (early exit) and are directly blended using cached historical features, thereby significantly eliminating redundant computations while preserving temporal coherence.

Algorithm 2 Region-level Computation Allocation (Motion-Aware Early Exit)   
Input: Noisy latent chunk $Z _ { C }$ , Previous clean chunk $X _ { C - 1 }$   
Input: Denoising steps $\{ t _ { 0 } , t _ { 1 } , \dotsc , t _ { S - 1 } \}$ , Search window size w   
Input: Similarity threshold $\gamma ,$ Morphological dilation kernel size k   
Output: Denoised output chunk $X _ { C }$   
1: $\bar { \boldsymbol { M } } _ { \mathrm { d y n } } \gets \mathbf { 1 }$ // Initialize mask to compute all tokens at the first step   
2: $\tilde { X } _ { C - 1 } $ null   
3: // 1. Spatial Denoising Iterations for Current Chunk C   
4: for each step index $s \in \{ 0 , 1 , \ldots , S - 1 \}$ do   
5: // Execute sparse Generator forward pass for dynamic tokens   
6: $\hat { X } _ { C } \gets$ Generator $( Z _ { C }$ , timestep $\mathbf { \epsilon } _ { | } = t _ { s }$ , active\_mask $= M _ { \mathrm { d y n } } )$   
7: $/ / 2 .$ Local Alignment & Mask Generation (Executed only at step 0)   
8: if $s = 0$ and $X _ { C - 1 }$ is not null then   
9: // Local motion search within $w \times w$ window to track objects   
10: $\tilde { X } _ { C - 1 } , S _ { \mathrm { s i m } } \gets$ LocalNxNMatch $( X _ { C - 1 } , \hat { X } _ { C } ,$ range = w)   
11: // Determine dynamic tokens via max-pooling & thresholding   
12: $M _ { \mathrm { d y n } } \gets \mathrm { M i n P o o l } 2 \mathrm { D } ( S _ { \mathrm { s i m } } ) < \gamma$   
13: $M _ { \mathrm { d y n } } \gets \mathrm { D i l a t e } ( M _ { \mathrm { d y n } } , \mathrm { k e r n e l } = k )$ // Spatial dilation for dynamic region   
14: $M _ { \mathrm { d y n } } \gets \mathrm { U p s a m p l e } ( M _ { \mathrm { d y n } } )$ // Expand to match original sequence length   
15: $M _ { \mathrm { s t a t i c } }  \neg M _ { \mathrm { d y n } }$ // Mask for visually redundant regions   
16: end if   
17: // 3. Spatial Blending & Computation Early Exit   
18: if $s > 0$ and $M _ { \mathrm { s t a t i c } }$ is available then   
19: // Bypass computation for static areas, copy from aligned history   
20: $\hat { X } _ { C } \gets M _ { \mathrm { s t a t i c } } \odot \tilde { X } _ { C - 1 } + \tilde { M } _ { \mathrm { d y n } } \odot \hat { X } _ { C }$   
21: end if   
22: // 4. Prepare Latent for Next Denoising Step   
23: if $s < S - 1$ then   
24: $Z _ { C } \gets \boldsymbol { S }$ cheduler.AddNoise $( \hat { X } _ { C } , t _ { s + 1 } )$   
25: end if   
26: end for   
27: // Return the fully denoised chunk   
28: $X _ { C }  \hat { X } _ { C }$   
29: return $X _ { C }$

## B More Implementation Details

Setup for frame-wise memory management implementation. Here, we provide the specific implementation details of the proposed Frame-wise Memory Management module. For the layer-wise retention ratio, the similarity threshold decays linearly from an upper bound $\sigma _ { m a x }$ to a lower bound $\sigma _ { m i n }$ across the transformer layers. During the training phase, to enhance the model’s robustness and prevent overfitting to strictly static backgrounds, we introduce a stochastic retention mechanism where naturally dropped tokens have a 5% probability of being forcefully retained. For the static token refresh strategy, a token is unconditionally revived if its continuous dropping age exceeds a predefined maximum token age threshold. Furthermore, to ensure seamless temporal transition across autoregressive chunks, we unconditionally preserve the exact last frame of the previously generated chunk to serve as the structural anchor for the current chunk.

Setup for computation allocation implementation. In this section, we detail the engineering setup for the Region-level Computation Allocation and Motion-Aware Early Exit mechanism. During the local alignment search between adjacent chunks, we constrain the spatial search window to a $3 \times 3$ grid to balance matching accuracy and computational overhead. To distinguish redundant tokens, we apply a strict cosine similarity threshold of $\gamma = 0 . 9 2$ To conservatively encompass moving subjects and prevent visual artifacts at motion boundaries, the resulting dynamic binary mask undergoes morphological spatial dilation using a $3 \times 3$ kernel (with stride 1 and padding 1). Crucially, to maximize inference efficiency, this motion search and mask generation process is strictly executed only once during the first denoising timestep. The acquired mask is then kept fixed for all subsequent denoising steps within the current chunk. Finally, after the spatial denoising loop concludes, an additional forward pass infused with clean context noise is executed to reliably update the KV cache for the upcoming autoregressive chunk.

## C More Qualitative Results

## C.1 Attention Distributions

To validate the motivation behind our layer-wise token retention strategy, we investigate how the model allocates its spatial attention at different stages of generation. To illustrate this progression, we extract and normalize the attention weights from layers 5, 15, and 25, utilizing them as representative examples of shallow, intermediate, and deep network layers, respectively. This visualization helps to confirm our hypothesis regarding the network’s transition from capturing fine-grained details to abstract semantic concepts.

Prompt: A bird landing on water and turning into a fish.  
![](images/2310035ea38054d9cd8b2fbc71f94e0262410ad816656e04ea1a58b6a80ac483.jpg)  
Figure 5: Attention distribution in layers 5, 15, and 25 of LongLive.

Prompt: A blonde teenage girl sits gracefully on a bent oak limb.  
![](images/f5966545381a894fd8c78d7285b3fa47af477d4ea937ef0dd64e7991c3b34006.jpg)  
Figure 6: Attention distribution in layers 5, 15, and 25 of LongLive.

![](images/8c4a9eb61437bcea488fa4e1d4b9c07433acc03b7c67691891405640af077ed6.jpg)  
Figure 7: Attention distribution in layers 5, 15, and 25 of LongLive.

Prompt: A sea creature with translucent, iridescent blue-green skin.  
![](images/4fe642909909aa4bee70f3f02356cf74cd6734245d112e777dd6f40b573f06f7.jpg)  
Figure 8: Attention distribution in layers 5, 15, and 25 of LongLive.

## C.2 Visualization

The following qualitative comparison demonstrates that our DASA framework generates video sequences that are visually nearly identical to the Self Forcing. This visual consistency across diverse dynamic scenarios—including fluid human actions, complex camera panning, detailed static backgrounds, and fast-moving vehicles—validates that our spatiotemporal redundancy reduction strategy achieves high generation quality while providing substantial computational acceleration.

"A building collapsing into a puddle of lava.”  
![](images/4c1dc0a54ede4d84335dca3b9da0471d1d12abfdddb3c885e2e90b78abeae96b.jpg)

"A cat is chasing a mice across a field, the mice runs towards an underground hole and the cat is left disappointed.”  
![](images/76c5d8e53bbd98556efb6b64f6481b66746fca5eb1a34d2200c540d5207bf6cd.jpg)  
Figure 9: Visualization of 480P videos generated by Self Forcing.

original 1.00 ×

![](images/708ea08f91cb67e5f86883d5d7b2b17f9eaa017c3187a43cde167c3b9767bde3.jpg)  
"A close up portrait of a woman lit by the side, the camera pulls back.”

![](images/e02d38411340e25feaa30d38710bd5b5f6873ae50a66e8a80db81edeb28d1c40.jpg)

![](images/fe5fe9231a21b4cd5206cfe3e8467dd9ab6e1f9b848dcc0628da5bcf7be649b9.jpg)

![](images/4d0d12dfcfc36b376eec7200703831bd976b344b1efffd02c2aa92aa61a07d82.jpg)

DASA 1.42 ×

![](images/c0096662c771258b4818d267aa28a102e3689b4ae64e59dbde93655873e86b74.jpg)

![](images/924086b58603385037a527194878d353ce41c3a60d33944a016af887fd323237.jpg)

![](images/d751aa5cad9b6a0a71bdf84d70aff2ddb4aff05b7e2255c280670b410f42fc25.jpg)

![](images/fee6645671f3e732135cc1e9494888d6c2eaa0f88228ad811ceb5aefe8765aa8.jpg)

DASA w/ adapt. 1.50 ×

![](images/f3de863188b1bbe708127014603941ed5212bcc0f204dd5997491e0093126d8b.jpg)

![](images/b274a9c085361eab474b9b82fd380c77031bce99acc8e62421885bb40606041e.jpg)

![](images/48c3afb3afc6be9f8cb5d0ca331320e4a12bd81a70e18a93e6b802f9f47023f0.jpg)

![](images/d7d60f7adadf3f9502b9c6a0d89fabb82005d23a88f822a20ece26cf598dad46.jpg)  
" A first-person POV shot rapidly flies towards a house's front door at 10x speed.”

original 1.00 ×

![](images/067a8d44508bb5015aa83bfc24d0de65e6a219fcaacd6cb948057a977d666a8c.jpg)

![](images/867afdd6fe0292add18167df40152268671685ebfcc20cf836e231f6cbcc2abf.jpg)

![](images/862b9fea571352a8c10570818a40996324e10542f263db40d15d3cdb4c4d7bd2.jpg)

![](images/4331e13eee9cd87d6e1a457cc5a9d1f65095d4b02538c5a5fc5ec16efab624dc.jpg)

DASA 1.42 ×

![](images/180bfe06d0e693e35702f99b7aef89fe279a040114600713e8005f64e816c1c7.jpg)

![](images/ae9e16aecd1322b87024a2e59ff004c43fcf69bb54ebde80835bd688534842cc.jpg)

![](images/b1974b360b4e8efc48d8d5073c23df488b792b2e3547408c5196aafa8375a460.jpg)

![](images/9ef33b2cabbcbb3db8c496608508719a626a10c09b1ad2545e958be95c89752f.jpg)

DASA w/ adapt. 1.50 ×

![](images/be33c5dd4c1421dd60af4e942a2c97c55ded945792185808a46b7fb74185dc89.jpg)

![](images/f8e92af61bb52ebb405667595fb9ef5054eddcd1fcfc4f28fb34fb29245a953a.jpg)

![](images/2178a14ecbd5eb3036d635663443e6640926267fc46540011bd068164956bdac.jpg)

![](images/515e9d001d7b4328b4ce0d6d5114f87aa5f555c8d3312ec0284850b48d3a74e9.jpg)  
" Zooming in hyper-fast to a dandelion to reveal macro dream-like abstract world.”

original 1.00 ×

![](images/88592a7224eafa1885f285d4e581b2719bb1d20f81b0cc29ca9de7c05ad60a05.jpg)

![](images/77c0a7102086aa71df3ee63e28c595638d3f36e702f48b4bbf392a4ef95969ab.jpg)

![](images/68fec44787f0343e90994747511c7f3f7e4f6a71910066a2241f7b2b3d37bb4d.jpg)

![](images/d80048211b9c7c7e165d36a412a5623a8b6aa70f7acd6059d76ad093117c704c.jpg)

DASA 1.42 ×

![](images/0726817498059205d4ff46b1d5e9737b9722187d31021e364e815517ff131ec9.jpg)

![](images/1d21edd47064c0db7ed954fed600b86cb757b21a8f756a64de530d24654fa6ed.jpg)

![](images/cc36a33d71abafaff4a1d9847642a0f786c13383c980664248629f66a513ec79.jpg)

![](images/69a0d61628c8d75daf71ca1dafa91e605839d7972a8eda71f16de7bfe7467eaf.jpg)

DASA w/ adapt. 1.50 ×

![](images/ed178c43f261db75b4c437f4de8a32ae7e86018c3d179b645f8a5503448c0d71.jpg)

![](images/2e62df5571dcbb7c58905d9ec19769966fe5023896ab0e902fd90331da7c4e6a.jpg)

![](images/770cf59293adebd22526876eab19557ec84deaecd0b0a184c9febf6b517a19d7.jpg)

![](images/90444bc354ddedfb8490bcfaca7a0b917f39975b08f35f829e4a90baffa9c270.jpg)  
Figure 10: Visualization of 720P videos generated by Self Forcing.

"A man riding a horse through the Gobi Desert with a beautiful sunset behind him, movie quality.”

![](images/41fcbe81cb13900457affe38bb799acf410f8820130eab30ca452a829f8c5a47.jpg)  
Figure 11: Visualization of 1080P videos generated by Self Forcing.

## References

[1] S. ai, H. Teng, H. Jia, L. Sun, L. Li, M. Li, M. Tang, S. Han, T. Zhang, W. Q. Zhang, W. Luo, X. Kang, Y. Sun, Y. Cao, Y. Huang, Y. Lin, Y. Fang, Z. Tao, Z. Zhang, Z. Wang, Z. Liu, D. Shi, G. Su, H. Sun, H. Pan, J. Wang, J. Sheng, M. Cui, M. Hu, M. Yan, S. Yin, S. Zhang, T. Liu, X. Yin, X. Yang, X. Song, X. Hu, Y. Zhang, and Y. Li. Magi-1: Autoregressive video generation at scale, 2025.

[2] T. Brooks, B. Peebles, C. Holmes, J. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, et al. Video generation models as world simulators, 2024. Accessed: 2026-05-02.

[3] J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, et al. Genie: Generative interactive environments. In ICML, 2024.

[4] G. Chen, D. Lin, J. Yang, C. Lin, J. Zhu, M. Fan, H. Zhang, S. Chen, Z. Chen, C. Ma, W. Xiong, W. Wang, N. Pang, K. Kang, Z. Xu, Y. Jin, Y. Liang, Y. Song, P. Zhao, B. Xu, D. Qiu, D. Li, Z. Fei, Y. Li, and Y. Zhou. Skyreels-v2: Infinite-length film generative model, 2025.

[5] J. Chen, Y. Zhao, J. Yu, R. Chu, J. Chen, S. Yang, X. Wang, Y. Pan, D. Zhou, H. Ling, et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025.

[6] S. Chen, C. Wei, S. Sun, P. Nie, K. Zhou, G. Zhang, M.-H. Yang, and W. Chen. Context forcing: Consistent autoregressive video generation with long context, 2026.

[7] X. Chen, Y. Wang, L. Zhang, S. Zhuang, X. Ma, J. Yu, Y. Wang, D. Lin, Y. Qiao, and Z. Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In ICLR, 2023.

[8] J. Cui, J. Wu, M. Li, T. Yang, X. Li, R. Wang, A. Bai, Y. Ban, and C.-J. Hsieh. Lol: Longer than longer, scaling video generation to hour. arXiv preprint arXiv:2601.16914, 2026.

[9] H. Deng, T. Pan, H. Diao, Z. Luo, Y. Cui, H. Lu, S. Shan, Y. Qi, and X. Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.

[10] K. Gao, J. Shi, H. Zhang, C. Wang, J. Xiao, and L. Chen. Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. In ICML. PMLR, 2025.

[11] Y. Gu, w. Mao, and M. Z. Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.

[12] H. Guo, Z. Jia, J. Li, B. Li, Y. Cai, J. Wang, Y. Li, and Y. Lu. Efficient autoregressive video diffusion with dummy head. arXiv preprint arXiv:2601.20499, 2026.

[13] Y. HaCohen, N. Chiprut, B. Brazowski, D. Shalem, D. Moshe, E. Richardson, E. Levin, G. Shiran, N. Zabari, O. Gordon, P. Panet, S. Weissbuch, V. Kulikov, Y. Bitterman, Z. Melumian, and O. Bibi. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

[14] S. He, B. Cui, B. Huang, Z. Ye, Y. Sun, L. Huang, H. Xue, Y. Yang, H. Hong, J. Tang, et al. Tc-pade: Trajectory-consistent pade approximation for diffusion acceleration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 35768–35778, 2026.

[15] Y. He, Y. He, S. He, F. Chen, H. Zhou, K. Zhang, and B. Zhuang. Neighboring autoregressive modeling for efficient visual generation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 19000–19010. IEEE, 2025.

[16] R. Henschel, L. Khachatryan, H. Poghosyan, D. Hayrapetyan, V. Tadevosyan, Z. Wang, S. Navasardyan, and H. Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In CVPR, pages 2568–2577, 2025.

[17] W. Hong, M. Ding, W. Zheng, X. Liu, and J. Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

[18] Y. Hong, Y. Mei, C. Ge, Y. Xu, Y. Zhou, S. Bi, Y. Hold-Geoffroy, M. Roberts, M. Fisher, E. Shechtman, et al. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025.

[19] J. Huang, Z. Ye, X. Hu, T. He, G. Zhang, S. Shi, J. Bian, and L. Jiang. Live: Long-horizon interactive video world modeling. arXiv preprint arXiv:2602.03747, 2026.

[20] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

[21] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, pages 21807–21818, 2024.

[22] T. HunyuanWorld. Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint, 2025.

[23] L. Ji, X. Liu, J. Shang, S. Wang, Y. Sun, H. Wu, and H. Wang. Videoar: Autoregressive video generation via next-frame & scale prediction. arXiv preprint arXiv:2601.05966, 2026.

[24] J. Jiang, W. Li, J. Ren, Y. Qiu, Y. Guo, X. Xu, H. Wu, and W. Zuo. Lovic: Efficient long video generation with context compression. arXiv preprint arXiv:2507.12952, 2025.

[25] Y. Jin, Z. Sun, N. Li, K. Xu, K. Xu, H. Jiang, N. Zhuang, Q. Huang, Y. Song, Y. Mu, and Z. Lin. Pyramidal flow matching for efficient video generative modeling. 2024.

[26] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[27] Kuaishou Technology. Kling v2.5 turbo: Video generation model, 2025. Accessed: 2026-01-13.

[28] W. Li, W. Pan, P.-C. Luan, Y. Gao, and A. Alahi. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025.

[29] B. Lin, Y. Ge, X. Cheng, Z. Li, B. Zhu, S. Wang, X. He, Y. Ye, S. Yuan, L. Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.

[30] F. Liu, S. Zhang, X. Wang, Y. Wei, H. Qiu, Y. Zhao, Y. Zhang, Q. Ye, and F. Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In CVPR, pages 7353–7363, 2025.

[31] K. Liu, W. Hu, J. Xu, Y. Shan, and S. Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.

[32] C. Lv, Y. Shi, Y. Huang, R. Gong, S. Ren, and W. Wang. Light forcing: Accelerating autoregressive video diffusion via sparse attention. arXiv preprint arXiv:2602.04789, 2026.

[33] S. D. Sarkar, R. Pautrat, O. Miksik, M. Pollefeys, I. Armeni, M. Rad, and M. Dusmanu. Cope-videolm: Leveraging codec primitives for efficient video language modeling. arXiv preprint arXiv:2602.13191, 2026.

[34] T. Seedance, D. Chen, L. Chen, X. Chen, Y. Chen, Z. Chen, Z. Chen, F. Cheng, T. Cheng, Y. Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

[35] T. Seedance, H. Chen, S. Chen, X. Chen, Y. Chen, Y. Chen, Z. Chen, F. Cheng, T. Cheng, X. Cheng, et al. Seedance 1.5 pro: A native audio-visual joint generation foundation model. arXiv preprint arXiv:2512.13507, 2025.

[36] M. Sun, W. Wang, G. Li, J. Liu, J. Sun, W. Feng, S. Lao, S. Zhou, Q. He, and J. Liu. Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. In CVPR, pages 7364–7373, 2025.

[37] Z. Tang, Y. He, Y. Han, W. Zhao, J. Tang, F. Wang, and B. Zhuang. K-forcing: Joint next-k-token decoding via push-forward language modeling, 2026.

[38] I. Team, T. Feng, Y. Han, J. He, Y. He, X. Lin, T. Liu, H. Lu, J. Tang, W. Wang, et al. Inferix: A blockdiffusion based next-generation inference engine for world simulation. arXiv preprint arXiv:2511.20714, 2025.

[39] R. Team, Z. Gao, Q. Wang, Y. Zeng, J. Zhu, K. L. Cheng, Y. Li, H. Wang, Y. Xu, S. Ma, Y. Chen, J. Liu, Y. Cheng, Y. Yao, J. Zhu, Y. Meng, K. Zheng, Q. Bai, J. Chen, Z. Shen, Y. Yu, X. Zhu, Y. Shen, and H. Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[40] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[41] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng, J. Wang, J. Zhang, J. Zhou, J. Wang, J. Chen, K. Zhu, K. Zhao, K. Yan, L. Huang, M. Feng, N. Zhang, P. Li, P. Wu, R. Chu, R. Feng, S. Zhang, S. Sun, T. Fang, T. Wang, T. Gui, T. Weng, T. Shen, W. Lin, W. Wang, W. Wang, W. Zhou, W. Wang, W. Shen, W. Yu, X. Shi, X. Huang, X. Xu, Y. Kou, Y. Lv, Y. Li, Y. Liu, Y. Wang, Y. Zhang, Y. Huang, Y. Li, Y. Wu, Y. Liu, Y. Pan, Y. Zheng, Y. Hong, Y. Shi, Y. Feng, Z. Jiang, Z. Han, Z.-F. Wu, and Z. Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[42] Z. Wang, Z. Liu, J. Li, K. Huang, B. Xu, F. Kang, M. An, P. Wang, B. Jiang, Y. Wei, et al. Matrixgame 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

[43] W. Weng, R. Feng, Y. Wang, Q. Dai, C. Wang, D. Yin, Z. Zhao, K. Qiu, J. Bao, Y. Yuan, et al. Art-v: Auto-regressive text-to-video generation with diffusion models. In CVPR, pages 7395–7405, 2024.

[44] B. Wu, C. Zou, C. Li, D. Huang, F. Yang, H. Tan, J. Peng, J. Wu, J. Xiong, J. Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.

[45] H. Xi, S. Yang, Y. Zhao, C. Xu, M. Li, X. Li, Y. Lin, H. Cai, J. Zhang, D. Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

[46] Y. Xia, S. Ling, F. Fu, Y. Wang, H. Li, X. Xiao, and B. Cui. Training-free and adaptive sparse attention for efficient long video generation. In CVPR, pages 15982–15993, 2025.

[47] D. Xie, Z. Xu, Y. Hong, H. Tan, D. Liu, F. Liu, A. Kaufman, and Y. Zhou. Progressive autoregressive video diffusion models. In CVPR, pages 6322–6332, 2025.

[48] B. Xu, Y. Du, Z. Liu, S. Yang, Z. Jiang, S. Yan, R. Saha, A. Pumarola, W. Wang, and P. Li. Sparse forcing: Native trainable sparse attention for real-time autoregressive diffusion video generation. arXiv preprint arXiv:2604.21221, 2026.

[49] R. Xu, G. Xiao, H. Huang, J. Guo, and S. Han. Xattention: Block sparse attention with antidiagonal scoring. arXiv preprint arXiv:2503.16428, 2025.

[50] S. Yang, W. Huang, R. Chu, Y. Xiao, Y. Zhao, X. Wang, M. Li, E. Xie, Y. Chen, Y. Lu, and S. H. Y. Chen. Longlive: Real-time interactive long video generation. 2025.

[51] S. Yang, H. Xi, Y. Zhao, M. Li, J. Zhang, H. Cai, Y. Lin, X. Li, C. Xu, K. Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. arXiv preprint arXiv:2505.18875, 2025.

[52] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

[53] J. Yi, W. Jang, P. H. Cho, J. Nam, H. Yoon, and S. Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025.

[54] T. Yin, M. Gharbi, T. Park, R. Zhang, E. Shechtman, F. Durand, and W. T. Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024.

[55] T. Yin, M. Gharbi, R. Zhang, E. Shechtman, F. Durand, W. T. Freeman, and T. Park. One-step diffusion with distribution matching distillation. In CVPR, pages 6613–6623, 2024.

[56] T. Yin, Q. Zhang, R. Zhang, W. T. Freeman, F. Durand, E. Shechtman, and X. Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025.

[57] S. Yuan, Y. Yin, Z. Li, X. Huang, X. Yang, and L. Yuan. Helios: Real real-time long video generation model. arXiv preprint arXiv:2603.04379, 2026.

[58] S. Zhai, Z. Ye, J. Liu, W. Xie, J. Hu, Z. Peng, H. Xue, D. Chen, X. Wang, L. Yang, et al. Stargen: A spatiotemporal autoregression framework with video diffusion model for scalable and controllable scene generation. In CVPR, pages 26822–26833, 2025.

[59] H. Zhang, W. Chai, S. He, A. Li, and Y. Fu. Dense video understanding with gated residual tokenization. arXiv preprint arXiv:2509.14199, 2025.

[60] J. Zhang, C. Xiang, H. Huang, J. Wei, H. Xi, J. Zhu, and J. Chen. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025.

[61] K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X.-X. Long, et al. Epona: Autoregressive diffusion world model for autonomous driving. In ICCV, pages 27220–27230, 2025.

[62] L. Zhang, S. Cai, M. Li, G. Wetzstein, and M. Agrawala. Frame context packing and drift prevention in next-frame-prediction video diffusion models. In NeurIPS, 2025.

[63] P. Zhang, Y. Chen, R. Su, H. Ding, I. Stoica, Z. Liu, and H. Zhang. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025.

[64] Y. Zhang, J. Xing, B. Xia, S. Liu, B. Peng, X. Tao, P. Wan, E. Lo, and J. Jia. Training-free efficient video generation via dynamic token carving. arXiv preprint arXiv:2505.16864, 2025.

[65] Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.

[66] H. Zhu, M. Zhao, G. He, H. Su, C. Li, and J. Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.