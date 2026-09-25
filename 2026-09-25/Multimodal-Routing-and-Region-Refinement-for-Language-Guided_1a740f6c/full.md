# Multimodal Routing and Region Refinement for Language-Guided Medical Image Segmentation

Md Maklachur Rahman<sup>1</sup>, Md Hasan Al Banna<sup>1</sup>, Saraf Anjum<sup>2</sup>, Assame Arnob<sup>1</sup>, and Tracy Hammond<sup>1</sup>

<sup>1</sup> Texas A&M University, College Station, TX 77843, USA <sup>2</sup> Independent Researcher

{maklachur,mdhasanalbanna,assamearnob,hammond}@tamu.edu, sarafanjumeva@gmail.com

Abstract. Textual descriptions can reduce ambiguity in medical image segmentation by specifying the finding and location to be delineated. Existing text-guided methods mainly improve where image and language features interact but generally retain a single learned update pathway across all image–text pairs. We propose MRSeg, a parameter-eficient framework that uses each image–text pair to route the adaptation of visual and textual features before dense prediction. Frozen ConvNeXt-Tiny and PubMedBERT encoders provide multiscale visual features and clinical text tokens. A joint router uses the deepest visual feature and pooled text to predict a sparse mixture over low-rank adapter bases. The resulting route is shared across separate adapter banks for two visual scales and text, coordinating their adaptation while keeping the featurespecific parameters separate. Region Bridge uses text-derived queries to aggregate dense visual tokens into latent regions, refines these regions through self-attention and text cross-attention, and redistributes the refined information back to the feature maps. Finally, a multiscale decoder combines refined semantic features with shallow image evidence. On QaTa-COV19 and MosMedData+, MRSeg achieves 90.90/83.32 and 81.53/68.82 Dice/mIoU, respectively, with 7.11M trainable parameters and 7.60 GFLOPs. Code: https://github.com/maklachur/MRSeg.

Keywords: Text-guided segmentation · Vision-language learning · LoRA · Parameter-eficient fine-tuning · Multimodal routing

## 1 Introduction

Medical image segmentation supports diagnosis, treatment planning, and longitudinal assessment by delineating anatomical structures and pathological regions. Encoder–decoder models such as U-Net [20], U-Net++ [27], and nnU-Net [11] remain strong foundations, while transformer-based designs improve long-range context modeling [3, 2], and recent state-space models such as Mamba provide an eficient alternative for capturing long-range dependencies [19]. Yet image-only segmentation [17, 18] remains dificult when abnormalities are difuse, low contrast, heterogeneous in appearance, or visually similar to surrounding tissue. In these cases, the image may contain several plausible regions without explicitly indicating which finding is clinically relevant.

Clinical descriptions provide complementary information about the target, including its presence, laterality, location, number, or extent. This motivates text-guided medical image segmentation, where an image and its associated description jointly determine the mask. LViT [12] established the foundation of language–vision interaction on pulmonary infection benchmarks. Later work explored encoder matching [1], bidirectional and language-guided adapters [25, 9], decoder attention [7], and reconstruction-based alignment [10]. Related methods also use text-guided attention [21], specialized decoders [26], memory and mixture-of-experts [5, 24], and robust prompting [22]. DD-CMD [16] introduced dual-domain cross-modal decoding that combines text-guided spatial attention with frequency-domain feature modulation. These approaches demonstrate that text can sharpen target specification, but they primarily ask where and how the modalities should interact.

We therefore ask whether representation adaptation should be conditioned on the specific image–text pair. Although cross-attention is input-dependent, its learned update subspace remains shared across samples. Diferent cases may require distinct adaptation patterns. A small unilateral opacity may demand greater sensitivity to localized texture and position, whereas widespread infection requires broader spatial context. Fully fine-tuning both encoders, however, is computationally costly and prone to overfitting on limited medical vision– language data.

Another challenge is the granularity mismatch between language and dense segmentation. Clinical descriptions specify findings and anatomical locations at a semantic or regional level, whereas the output must delineate them at the pixel level. Direct token-to-pixel fusion asks a single interaction to identify the target, organize spatial evidence, and preserve boundaries. We instead decouple these roles. The image–text pair first determines how its representations should adapt; the adapted visual features are then organized and refined at the region level, and the decoder reconstructs the dense mask.

To address this, we propose MRSeg, a parameter-eficient framework for routed multimodal adaptation and region refinement. MRSeg retains frozen ConvNeXt-Tiny [13] and PubMedBERT [6] backbones and learns lightweight LoRA [8], normalization, routing, refinement, and decoding parameters. A router derives a sparse policy from the globally pooled Stage-4 feature and meanpooled text. Pair Adapter reuses this policy across separate low-rank banks for the Stage-3 feature, Stage-4 feature, and text tokens. The route is shared, but adapter parameters remain feature-specific. Two parallel Region Bridge modules form text-aware latent regions and return their refined information to the dense Stage-3 (f ) and Stage-4 (f ) maps through identity-safe residual updates.

Our main contributions are summarized as follows: (1) We propose MRSeg, a parameter-eficient framework that separates pair-conditioned adaptation, regionlevel refinement, and dense mask reconstruction using frozen visual and language backbones. (2) We introduce Pair Adapter, which applies a shared sparse image– text route across separate visual and textual low-rank banks, and Region Bridge, which refines dense features through text-aware latent regions. (3) Extensive experiments on QaTa-COV19 and MosMedData+ demonstrate the complementary benefits of both modules, achieving 90.90%/83.32% and 81.53%/68.82% Dice/mIoU with 7.11M trainable parameters and 7.60 GFLOPs.

![](images/4d49980c6c319453b902515561f14bbd13d445bb852435e85363acf294235212.jpg)  
Fig. 1. Overview of MRSeg. Frozen, parameter-eficiently adapted encoders produce multiscale visual features and text tokens. The router uses only $\mathrm { G A P } ( f _ { 4 } )$ and Mean(t) to derive a sparse route α. Pair Adapter reuses this route across separate $f _ { 3 } , \ f _ { 4 } ,$ , and text adapter banks. Two Region Bridge modules independently produce $f _ { 3 } ^ { \prime \prime }$ and $f _ { 4 } ^ { \prime \prime }$ which are decoded together with $f _ { 1 } , f _ { 2 }$ , and shallow features from the adapted image.

## 2 Methodology

The overall pipeline is illustrated in Fig. 1, and Fig. 2 details the two proposed components. The design follows three principles. First, pretrained encoders should retain their general visual and biomedical priors, so we freeze their base parameters and learn only lightweight residual updates. Second, the current image–text pair should determine which update subspaces are active. Third, cross-modal refinement should pass through an intermediate regional representation before dense decoding. Given an image $I \in \mathbb { R } ^ { H \times W \times C }$ and clinical text $T ,$ MRSeg predicts logits $\hat { Y } \in \mathbb { R } ^ { H \times W \times 1 }$ . The complete forward path of our proposed method is the following:

$$
\begin{array} { r l } & { I _ { 3 } = \mathcal { A } _ { \mathrm { a d a p t } } ( I ) , } \\ & { \left( f _ { 1 } , f _ { 2 } , f _ { 3 } , f _ { 4 } \right) = E _ { v } ( I _ { 3 } ) , \qquad t = E _ { t } ( T ) , } \\ & { \qquad \alpha = \mathcal { R } ( \mathrm { G A P } ( f _ { 4 } ) , \mathrm { M e a n } ( t ) ) , } \\ & { \qquad ( f _ { 3 } ^ { \prime } , f _ { 4 } ^ { \prime } , t ^ { \prime } ) = \mathcal { A } _ { \mathrm { p a i r } } ( f _ { 3 } , f _ { 4 } , t ; \alpha ) , } \\ & { \qquad f _ { s } ^ { \prime \prime } = \mathcal { B } _ { s } ( f _ { s } ^ { \prime } , t ^ { \prime } ) , \quad s \in \{ 3 , 4 \} , } \\ & { \qquad \hat { Y } = \mathcal { D } ( I _ { 3 } , f _ { 1 } , f _ { 2 } , f _ { 3 } ^ { \prime \prime } , f _ { 4 } ^ { \prime \prime } ) . } \end{array}\tag{1}
$$

The low-level features $f _ { 1 }$ and $f _ { 2 }$ retain local structure and skip the proposed adaptation modules. In contrast, the more semantic features $f _ { 3 }$ and $f _ { 4 }$ are pro-

cessed by routed adaptation and region refinement. The two Region Bridge modules operate independently and in parallel.

## 2.1 Vision and Language Encoding

Medical image adaptation: For grayscale inputs, direct channel replication satisfies the three-channel requirement of a natural-image backbone but cannot correct modality-specific intensity statistics. We use a residual Medical Image Adapter:

$$
I _ { 3 } = I _ { \mathrm { b a s e } } + A _ { \mathrm { i m g } } ( I _ { \mathrm { g r a y } } ) ,\tag{2}
$$

where $I _ { \mathrm { b a s e } }$ is the replicated or retained three-channel image. The correction branch uses pointwise projection, depthwise $3 \times 3$ filtering, normalization, GELU, and a final pointwise projection. The last projection is zero-initialized, so the module initially reproduces the standard three-channel path and learns only the correction required by the task.

Vision encoder: A frozen ConvNeXt-Tiny [13] extracts four feature maps $f _ { 1 } ,$ $f _ { 2 } , f _ { 3 } ,$ , and $f _ { 4 }$ with channels {96, 192, 384, 768} and strides {4, 8, 16, 32}. LoRA [8] is added into selected pointwise transformations in the later stages, and the corresponding normalization parameters are trainable. In the visual branch, the reduced-rank representation is additionally processed by depthwise $3 \times 3$ spatial mixing. This keeps the pretrained projection fixed while allowing local taskspecific corrections. Fig. 1 denotes these updates compactly as ${ } ^ { \mathrm { \scriptsize { \sc \mathrm { \ 4 } } } } \mathrm { \Sigma } \mathrm { \mathrm { L o R A } } + \mathrm { N o r m } . { } ^ { \mathrm { \scriptsize { \cdot } } }$

Language encoder: The prompt is padded or truncated to $L = 2 4$ tokens and encoded by frozen PubMedBERT, yielding $t \in \mathbb { R } ^ { L \times 7 6 8 }$ . LoRA is applied to the query and value projections of the final four transformer layers, together with their normalization parameters. The full sequence is retained for region-to-text interaction, while $\bar { t } = \mathrm { M e a n } ( t )$ provides a compact routing descriptor. Encoder LoRA learns a dataset-level adjustment, whereas the routed feature adapters introduced next provide sample-specific updates.

## 2.2 Routed Multimodal Adaptation

A fixed adapter is task-specific but not case-specific. The conventional adapter applies the same residual transformation to every sample. Pair Adapter instead maintains M low-rank bases and lets each image–text pair activate a sparse subset. We first compute $\bar { f } _ { 4 } = \mathrm { G A P } ( f _ { 4 } )$ and $\bar { t } = \mathrm { M e a n } ( t )$ . The router produces:

$$
\begin{array} { r } { \pmb { \alpha } = \mathrm { T o p K N o r m } \big ( \mathrm { s o f t m a x } \big ( g _ { \theta } [ \bar { f } _ { 4 } \| \bar { t } ] \big ) \big ) \in \mathbb { R } ^ { M } , } \end{array}\tag{3}
$$

where TopKNorm retains and renormalizes the K largest weights. We use $M = 4$ available bases and $K = 2$ active bases. The router excludes $f _ { 3 }$ as an input because $f _ { 4 }$ provides the most compact global context, while $f _ { 3 }$ remains a higherresolution feature to be adapted.

![](images/f37f914b917c58b025753a686d85d713486e0d7081cb037b45625e780fa6dcfd.jpg)  
Fig. 2. Core components of MRSeg. (a) Routed multimodal adaptation: the image– text route controls separate adapter banks for $t , \ f _ { 3 } ,$ , and $f _ { 4 } . \ ( \mathrm { b } )$ Text-aware region refinement: text-derived queries aggregate dense visual tokens, regional and cross-modal interactions refine them, and dense reprojection returns the correction to the original feature map.

The same route α controls three separate adapter banks. For $h \in \{ t , f _ { 3 } , f _ { 4 } \}$

$$
h ^ { \prime } = h + \frac { \eta } { r } \sum _ { m = 1 } ^ { M } \alpha _ { m } B _ { m } ^ { ( h ) } \Big ( A _ { m } ^ { ( h ) } \mathrm { L N } ( h ) \Big ) ,\tag{4}
$$

where $r$ is the low-rank dimension and η is the LoRA scale. Visual maps are flattened to token sequences and restored after adaptation. The key distinction is that the route is shared, but the adapter parameters are not. This encourages compatible specialization across modalities without forcing text and visual features through identical projections. All up-projections are zero-initialized, making Pair Adapter an identity mapping at initialization. Fig. 2(a) summarizes the route and its three applications.

## 2.3 Text-Aware Region Refinement and Mask Decoding

Region formation and refinement: For scale $s \in \{ 3 , 4 \}$ , let $F = f _ { s } ^ { \prime } \in$ $\mathbb { R } ^ { H _ { s } \times W _ { s } \times C _ { s } }$ and $N = H _ { s } W _ { s } . \mathrm { ~ A ~ } 1 \times 1$ projection and reshaping produce $Z \in { }$ $\mathbb { R } ^ { N \times D }$ . Mean-pooled adapted text generates region queries $Q ~ \in ~ \mathbb { R } ^ { R _ { s } \times D }$ . We use $D = 1 9 2 , R _ { 3 } = 1 2$ , and $R _ { 4 } ~ = ~ 6$ . After LayerNorm, the assignment and aggregation are

$$
A = \operatorname { s o f t m a x } _ { N } \left( { \frac { Q { \bar { Z } } ^ { \top } } { \sqrt { D } } } \right) \in \mathbb { R } ^ { R _ { s } \times N } , \qquad G = A Z ,\tag{5}
$$

where $\bar { Z } = \mathrm { L N } ( Z )$ . Softmax is applied over the N spatial tokens, so each region query gathers a normalized distribution of visual evidence. The resulting region tokens are refined by two blocks, each containing region self-attention, crossattention to projected $t ^ { \prime } ,$ and a feed-forward network: $\widehat { G } = \mathcal { F } _ { \mathrm { r e g } } ( G , t ^ { \prime } )$

The same assignment redistributes the refined tokens to the dense grid:

$$
F ^ { \prime } = F + \gamma P _ { \mathrm { o u t } } \Bigl ( \mathrm { r e s h a p e } ( A ^ { \top } \widehat { G } ) \Bigr ) ,\tag{6}
$$

where $\gamma$ is learnable. The output projection is zero-initialized, so Region Bridge initially preserves its input. Applying Eq. 6 independently at the two scales gives $f _ { 3 } ^ { \prime \prime }$ and $f _ { 4 } ^ { \prime \prime }$ . These latent regions are learned solely through the segmentation objective and should not be interpreted as supervised anatomical entities.

Mask decoding and optimization: The decoder projects $f _ { 4 } ^ { \prime \prime }$ to 192 channels, upsamples ${ \mathrm { i t } } ,$ and fuses it with $f _ { 3 } ^ { \prime \prime }$ . It then introduces $f _ { 2 }$ and $f _ { 1 }$ through residual semantic skip fusion. Given decoder feature x and skip s, a lightweight gate predicts $g =$ tanh $\mathcal { G } ( [ \mathrm { U p } ( x ) ; s ] )$ and uses $\tilde { s } = s + s \odot g \mathrm { }$ . The gate output is zero-initialized, recovering ordinary skip fusion at the start of training. Two shallow stems extracted directly from $I _ { 3 }$ provide half $( H / 2 )$ and full-resolution $( H )$ detail. A coarse $1 \times 1$ classifier and a depthwise detail branch produce additive logits, $\hat { Y } = \hat { Y } _ { \mathrm { c o a r s e } } + \varDelta \hat { Y } _ { \mathrm { d e t a i l } }$ . We optimize equally weighted binary cross-entropy with logits and soft Dice loss: $\mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + \mathcal { L } _ { \mathrm { D i c e } }$ . Sigmoid and a 0.5 threshold are used to produce the mask output.

## 3 Experiments and Results

Datasets, Evaluation Metrics, and Implementation Details: Following prior methods [12, 7, 10, 5], we evaluate on two public image–text–mask benchmarks for pulmonary infection segmentation. QaTa-COV19 [4, 12] contains 9,258 chest X-rays and is split into 5,716/1,429/2,113 as training, validation, and test samples. MosMedData+ [15, 12] contains 2,729 CT slices and is split into 2,183/273/273 samples. We primarily report the Dice coeficient and mean Intersection over Union (mIoU) as overlap-based evaluation, and the 95th percentile of the Hausdorf Distance (HD95), measured in pixels, as the boundary error metric in the ablation studies.

All images are resized to 224×224, and descriptions use $L = 2 4$ tokens. Training applies diferent augmentations, followed by channel-wise normalization. We use AdamW [14], batch = 8, initial learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , cosine annealing to $1 0 ^ { - 6 }$ , and early stopping with patience 60. QaTa-COV19 is trained for at most 200 epochs and MosMedData+ for 150. The LoRA rank is 8 with scale 16 and dropout 0.05. Both Region Bridges use two refinement blocks and four attention heads. Experiments run on one NVIDIA RTX 3090Ti (24GB).

Comparison with State-of-the-Art Methods: Table 1 compares MRSeg with representative image-only and text-guided segmentation methods. MRSeg achieves the best performance on both datasets, obtaining 90.90% Dice and 83.32% mIoU on QaTa-COV19, and 81.53% Dice and 68.82% mIoU on MosMed-Data+. It improves over TGCAM [7], the strongest prior method on QaTa-COV19, by 0.30 Dice and 0.51 mIoU points. On MosMedData+, MRSeg surpasses the best prior Dice from MAdapter [25] by 2.91 points and the best prior mIoU from RecLMIS [10] by 3.75 points. Importantly, MRSeg requires only 7.11M trainable parameters and 7.60 GFLOPs, demonstrating strong performance with parameter-eficient adaptation. The qualitative comparisons in Fig. 3 further show more complete lesion coverage and fewer missed or spurious regions across both datasets.

Table 1. Comparison with SOTA methods on QaTa-COV19 and MosMedData+. Best and second-best results are shown in bold and underlined, respectively. ↑/↓ indicate higher/lower is better. — indicates values not reported by the original paper.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td rowspan="2">Text</td><td rowspan="2">Params ↓ (M)</td><td rowspan="2">FLOPs↓ (G)</td><td colspan="2">QaTa-COV19</td><td colspan="2">MosMedData+</td></tr><tr><td>Dice ↑</td><td>mIoU ↑</td><td>Dice ↑</td><td>mIoU ↑</td></tr><tr><td>U-Net [20]</td><td>MICCAI&#x27;15</td><td>X</td><td>14.8</td><td>50.3</td><td>79.02</td><td>69.46</td><td>64.60</td><td>50.73</td></tr><tr><td>nnUNet [11]</td><td>Nature&#x27;21</td><td>X</td><td>19.1</td><td>412.7</td><td>80.42</td><td>70.81</td><td>72.59</td><td>60.36</td></tr><tr><td>Swin-UNet [2]</td><td>ECCV&#x27;22</td><td>X</td><td>82.3</td><td>67.3</td><td>78.07</td><td>68.34</td><td>63.29</td><td>50.19</td></tr><tr><td>LAVT [23]</td><td>CVPR&#x27;22</td><td>√</td><td>118.6</td><td>83.8</td><td>79.28</td><td>69.89</td><td>73.29</td><td>60.41</td></tr><tr><td>LViT [12]</td><td>IEEE TMI&#x27;23</td><td>√</td><td>29.7</td><td>54.1</td><td>83.66</td><td>75.11</td><td>74.57</td><td>61.33</td></tr><tr><td>LGA [9]</td><td>MICCAI&#x27;24</td><td>√</td><td>8.24</td><td>381.1</td><td>84.65</td><td>76.23</td><td>75.63</td><td>62.52</td></tr><tr><td>TGCAM [7]</td><td>MICCAI&#x27;24</td><td></td><td></td><td></td><td>90.60</td><td>82.81</td><td>77.82</td><td>63.69</td></tr><tr><td>MAdapter [25]</td><td>MICCAI&#x27;24</td><td></td><td></td><td></td><td>90.22</td><td>82.16</td><td>78.62</td><td>64.78</td></tr><tr><td>RecLMIS [10]]</td><td>IEEE TMI&#x27;24</td><td>√</td><td>23.7</td><td>24.1</td><td>85.22</td><td>77.00</td><td>77.48</td><td>65.07</td></tr><tr><td>ARSeg [22]</td><td>MICCAI&#x27;25</td><td>了</td><td></td><td></td><td>84.09</td><td>72.64</td><td>73.24</td><td>59.82</td></tr><tr><td>TextMoE [24]</td><td>MICCAI&#x27;25</td><td>√</td><td></td><td></td><td>89.08</td><td>80.32</td><td>74.66</td><td>59.57</td></tr><tr><td>MG-UNet [5]</td><td>MICCAI&#x27;25</td><td>√</td><td>30.5</td><td>11.0</td><td>88.10</td><td>77.80</td><td>76.39</td><td>61.79</td></tr><tr><td>MRSeg (Ours)</td><td></td><td>√</td><td>7.11</td><td>7.60</td><td>90.90</td><td>83.32</td><td>81.53</td><td>68.82</td></tr></table>

![](images/9fa46bfda9481eba4057566a763307df0912f7c1ca1c0af953221e89255928ff.jpg)  
Fig. 3. Qualitative comparison on QaTa-COV19 and MosMedData+. Overlays indicate true positives (yellow), false negatives (red), and false positives (green).

Ablation Study on Diferent Core Modules: Table 2 shows that each component contributes to MRSeg. Removing text causes the largest drop, confirming that clinical descriptions provide complementary guidance beyond the image alone. The routed adaptation and region bridges are also important, as removing either consistently degrades Dice, mIoU, and HD95 on both datasets. Independent image and text routers perform worse than our shared pair-conditioned router, supporting joint adaptation from the paired inputs. The medical image adapter and encoder LoRA provide further gains, while the full model achieves the best overall performance across all metrics.

Table 2. Ablation study on QaTa-COV19 and MosMedData+. Dice and mIoU are reported in %, while HD95 in pixels. w/o means without. Best values are bolded.
<table><tr><td rowspan="2">Model Variant</td><td colspan="3">QaTa-COV19</td><td colspan="3">MosMedData+</td></tr><tr><td>Dice ↑</td><td>mIoU ↑</td><td>HD95↓</td><td>Dice ↑</td><td>mIoU ↑</td><td>HD95↓</td></tr><tr><td>Image-only (No Text)</td><td>87.56</td><td>77.87</td><td>27.27</td><td>78.94</td><td>65.21</td><td>19.22</td></tr><tr><td> $w / o$  Medical Image Adapter</td><td>90.68</td><td>82.95</td><td>16.45</td><td>80.99</td><td>68.06</td><td>15.18</td></tr><tr><td>w/o Encoder LoRA</td><td>90.49</td><td>82.64</td><td>16.90</td><td>80.76</td><td>67.73</td><td>16.11</td></tr><tr><td>Independent Image/Text Routers</td><td>90.59</td><td>82.72</td><td>16.68</td><td>80.82</td><td>67.82</td><td>15.94</td></tr><tr><td> $w / o$  Region Bridges</td><td>89.21</td><td>80.52</td><td>23.07</td><td>79.94</td><td>66.58</td><td>17.83</td></tr><tr><td> $w / o$  Routed Adaptation</td><td>88.96</td><td>80.11</td><td>24.20</td><td>80.03</td><td>66.71</td><td>17.61</td></tr><tr><td>Full model</td><td>90.90</td><td>83.32</td><td>14.34</td><td>81.53</td><td>68.82</td><td>14.73</td></tr></table>

Table 3. Robustness under reduced supervision. Models are trained on randomly sampled {30%, 70%, 100%} subsets of the training split and evaluated on the unchanged full test set. Dice and mIoU are reported in ${ \% } ,$ and HD95 in pixels.
<table><tr><td rowspan="2">Training Data</td><td colspan="3">QaTa-COV19</td><td colspan="3">MosMedData+</td></tr><tr><td>Dice ↑</td><td>mIoU ↑</td><td>HD95↓</td><td>Dice ↑</td><td>mIoU ↑</td><td>HD95↓</td></tr><tr><td>30%</td><td>89.60</td><td>81.17</td><td>17.07</td><td>77.83</td><td>63.71</td><td>18.53</td></tr><tr><td>70%</td><td>90.48</td><td>82.62</td><td>15.57</td><td>80.11</td><td>66.83</td><td>17.94</td></tr><tr><td>100%</td><td>90.90</td><td>83.32</td><td>14.34</td><td>81.53</td><td>68.82</td><td>14.73</td></tr></table>

Ablation Study on Data Eficiency: As shown in Table 3, our model remains competitive even under limited supervision. With only 30% of the training data, MRSeg already approaches several state-of-the-art methods trained on the full dataset, achieving 89.60% Dice on QaTa-COV19 and 77.83% on MosMed-Data+. As we increase the training data, performance improves steadily and HD95 consistently decreases, showing that MRSeg learns efectively from limited annotations while continuing to benefit from additional supervision.

## 4 Conclusion

We presented MRSeg, a parameter-eficient framework that decomposes languageguided medical image segmentation into pair-conditioned adaptation, text-aware region refinement, and dense reconstruction. Pair Adapter coordinates featurespecific visual and textual updates through a shared sparse route, while Region Bridge organizes and refines visual evidence at the region level before mask decoding. Experiments and ablations on QaTa-COV19 and MosMedData+ demonstrate that these components provide complementary gains, enabling strong segmentation performance with only 7.11M trainable parameters and 7.60 GFLOPs. Building on these results, future work will extend evaluation beyond the current text-guided pulmonary infection benchmarks and examine robustness to variations in the input text.

## References

1. Bui, P.N., Le, D.T., Choo, H.: Visual-textual matching attention for lesion segmentation in chest images. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 702–711. Springer (2024)

2. Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., Wang, M.: Swinunet: Unet-like pure transformer for medical image segmentation. In: European conference on computer vision. pp. 205–218. Springer (2022)

3. Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L., Zhou, Y.: Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306 (2021)

4. Degerli, A., Kiranyaz, S., Chowdhury, M.E., Gabbouj, M.: Osegnet: Operational segmentation network for covid-19 detection using chest x-ray images. In: 2022 IEEE International Conference on Image Processing (ICIP). pp. 2306–2310. IEEE (2022)

5. Ding, S., Li, M., Wang, C.: Mg-unet: A memory-guided unet for lesion segmentation in chest images. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 355–365. Springer (2025)

6. Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., Poon, H.: Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare 3(1), 2:1– 2:23 (Jan 2022). https://doi.org/10.1145/3458754

7. Guo, Y., Zeng, X., Zeng, P., Fei, Y., Wen, L., Zhou, J., Wang, Y.: Common visionlanguage attention for text-guided medical image segmentation of pneumonia. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 192–201. Springer (2024)

8. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. Iclr 1(2), 3 (2022)

9. Hu, J., Li, Y., Sun, H., Song, Y., Zhang, C., Lin, L., Chen, Y.W.: Lga: A language guide adapter for advancing the sam model’s capabilities in medical image segmentation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 610–620. Springer (2024)

10. Huang, X., Li, H., Cao, M., Chen, L., You, C., An, D.: Cross-modal conditioned reconstruction for language-guided medical image segmentation. IEEE Transactions on Medical Imaging 44(4), 1821–1835 (2024)

11. Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., Maier-Hein, K.H.: nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18(2), 203–211 (2021)

12. Li, Z., Li, Y., Li, Q., Wang, P., Guo, D., Lu, L., Jin, D., Zhang, Y., Hong, Q.: Lvit: language meets vision transformer in medical image segmentation. IEEE transactions on medical imaging 43(1), 96–107 (2023)

13. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11976–11986 (2022)

14. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

15. Morozov, S.P., Andreychenko, A.E., Pavlov, N.A., Vladzymyrskyy, A., Ledikhova, N.V., Gombolevskiy, V.A., Blokhin, I.A., Gelezhe, P.B., Gonchar, A., Chernina, V.Y.: Mosmeddata: Chest ct scans with covid-19 related findings dataset. arXiv preprint arXiv:2005.06465 (2020)

16. Rahman, M.M., Hammond, T.: Dual-domain cross-modal decoding for clinical textguided medical image segmentation. arXiv preprint arXiv:2608.11335 (2026)

17. Rahman, M.M., Jung, S.K., Hammond, T.: Aulunet: An adaptive ultra-lightweight u-net framework for eficient skin lesion segmentation in resource-constrained environments. In: 36th British Machine Vision Conference 2025, BMVC 2025, Shefield, UK. BMVA (2025)

18. Rahman, M.M., Jung, S.K., Hammond, T.: Mambaliteunet: Cross-gated adaptive feature fusion for robust skin lesion segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8556–8565 (June 2026)

19. Rahman, M.M., Tutul, A.A., Nath, A., Laishram, L., Jung, S.K., Hammond, T.: Mamba in vision: A comprehensive survey of techniques and applications. arXiv preprint arXiv:2410.03105 (2024)

20. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)

21. Tomar, N.K., Jha, D., Bagci, U., Ali, S.: Tganet: Text-guided attention for improved polyp segmentation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 151–160. Springer (2022)

22. Wang, Q., Lin, X., Yan, Z.: Towards robust medical image referring segmentation with incomplete textual prompts. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 636–646. Springer (2025)

23. Yang, Z., Wang, J., Tang, Y., Chen, K., Zhao, H., Torr, P.H.: Lavt: Languageaware vision transformer for referring image segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18155– 18165 (2022)

24. Zeng, Q., Luo, H., Ma, X., Lu, Z., Hu, Y., Xia, Y.: Exploring text-enhanced mixture-of-experts for semi-supervised medical image segmentation with composite data. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 226–236. Springer (2025)

25. Zhang, X., Ni, B., Yang, Y., Zhang, L.: Madapter: A better interaction between image and language for medical image segmentation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 425–434. Springer (2024)

26. Zhong, Y., Xu, M., Liang, K., Chen, K., Wu, M.: Ariadne’s thread: Using text prompts to improve segmentation of infected areas from chest x-ray images. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 724–733. Springer (2023)

27. Zhou, Z., Rahman Siddiquee, M.M., Tajbakhsh, N., Liang, J.: Unet++: A nested u-net architecture for medical image segmentation. In: Deep learning in medical image analysis and multimodal learning for clinical decision support, pp. 3–11. Springer (2018)