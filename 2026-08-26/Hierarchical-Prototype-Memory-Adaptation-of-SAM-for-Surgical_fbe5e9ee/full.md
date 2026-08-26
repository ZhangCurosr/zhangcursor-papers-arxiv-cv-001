# Hierarchical Prototype-Memory Adaptation of SAM for Surgical Instrument Segmentation

Xinning Yao<sup>1</sup>, Jingjing Wang<sup>1</sup>, Jinghua Yue<sup>1</sup>, Xiaoyan Luo<sup>1</sup>, Fugen Zhou<sup>1</sup>, Bo Liu<sup>1</sup>

Abstract— Surgical instrument segmentation (SIS) is fundamental for computer-assisted surgery, where reliable instrument masks enable precise scene understanding and clinical assistance. Recently, adapting foundation models like the Segment Anything Model (SAM) to the surgical domain via promptlearning has shown encouraging results. However, the performance of these adapted models under challenging surgical conditions is constrained by suboptimal adaptation mechanisms. Specifically, optimizing prompts or prototypes purely via downstream segmentation loss tends to cause them to degenerate into task-specific parameters rather than serving as persistent, stable category memory, thereby degrading their robustness against complex intraoperative variations. Moreover, routing multi-scale visual cues through a single prompt pathway creates a bottleneck that hinders effective scale-matched coupling. To address these limitations, we propose HPMA, a Hierarchical Prototype-Memory Adaptation framework for SAM. Specifically, HPMA constructs a frozen, multi-scale visual prototype memory bank from annotated surgical scenes and integrates it into SAM’s feature space using lightweight adapters to preserve stable category evidence. To maximize the utility of multi-scale cues, we introduce a scale-matched coupling mechanism where global prototypes calibrate class-level prompt features, structural prototypes guide decoder object queries, and local prototypes align high-resolution feature maps through a local alignment objective. Extensive experiments on the public EndoVis2017 and EndoVis2018 datasets demonstrate that our approach achieves state-of-the-art performance, outperforming existing foundation model adaptation methods.

## INTRODUCTION

Accurate surgical instrument segmentation (SIS) plays a fundamental role in computer-assisted surgery, where reliable instrument masks are essential for precise scene understanding and clinical assistance. In recent years, vision founda tion models, represented by the Segment Anything Model (SAM) and its multimodal evolution SAM3 [1], [2], [3], have demonstrated promising potential in general-purpose segmentation tasks. However, directly applying these models to the surgical domain is hindered by a substantial domain gap. Unlike objects in natural images, surgical instrument categories must be recognized from highly domain-specific visual evidence. Instruments often exhibit delicate structures, while surgical scenes commonly involve strong specular reflections, mutual occlusions, and ambiguous boundaries [4], [5]. Consequently, conventional parameter fine-tuning methods are often insufficient to bridge this gap effectively.

To address this challenge, existing prompt-learning methods [6], [7], such as SurgicalSAM [8], have attempted to incorporate domain-specific knowledge into foundation models, demonstrating their value in mitigating the discrepancy between generic representations and surgical semantics. Nevertheless, their performance under complex and highly variable surgical conditions remains constrained by suboptimal adaptation mechanisms. Specifically, existing methods suffer from two major limitations, as shown in Fig. 1(a). First, optimizing prompts or prototypes exclusively via downstream segmentation objectives frequently causes them to overfit as task-specific parameters, eroding their capacity to act as persistent, stable category memories and diminishing model robustness against intraoperative variations [7], [9]. Second, existing approaches typically route multi-scale visual cues through a single prompting pathway. This design creates a representational bottleneck that prevents effective scalematched coupling between visual features and the corresponding model components [10], [11].

![](images/736feea2cdf3395475c414b9a99414ccca03d834281a248202cc329e54fbab1d.jpg)  
Fig. 1: The conceptual comparison of (a) previous promptlearning method and our (b) hierarchical prototype-memory adaptation method.

To overcome these limitations, we propose Hierarchical Prototype-Memory Adaptation (HPMA), a novel adaptation framework for SAM. To address the first limitation of prompt degeneration, HPMA explicitly decouples domain knowledge storage from downstream feature adaptation. As shown in Fig. 1, instead of optimizing prompts blindly, HPMA extracts visual evidence from annotated surgical scenes to construct a frozen multi-scale prototype memory bank. By clustering and preserving diverse visual evidence of surgical instruments, the memory bank serves as a persistent statistical memory insulated from downstream optimization objectives. We then introduce lightweight adapters that learn to transform this stable, class-specific visual evidence into effective feature enhancement, seamlessly injecting them into the feature space of SAM.

Furthermore, we argue that surgical instrument segmentation is not merely a single-scale adaptation problem. A model should capture not only the overall appearance of an instrument but also its complex internal structures and fine-grained details. Accordingly, visual evidence at different scales should be delivered to the model components best suited to exploit it. To this end, we design a hierarchical representation level-to-module coupling mechanism. At the global semantic level, global prototypes calibrate class-level text-prompt features before feature fusion. At the structural component level, structural prototypes are injected into the Transformer decoder to guide and refine object queries, enabling the model to better capture complex instrument structures. At the local detail level, local prototypes serve as local alignment targets for high-resolution feature maps during training, effectively sharpening fine-grained instrument details without introducing additional computational overhead at inference time.

In summary, our main contributions are as follows:

• We propose HPMA, a framework that explicitly models visual evidence from the surgical domain as a prototype memory. Through lightweight adapters, HPMA transforms this frozen visual memory into model-specific feature enhancement, improving segmentation robustness in complex surgical scenes.

• We develop a hierarchical representation level-tomodule coupling mechanism that reduces the representational bottleneck imposed by a single adaptation pathway. By adapting prototype memory at different scales to the appropriate stages of the model, the proposed mechanism more effectively alleviates class-level semantic mismatch, insufficient structural perception, and fine-grained detail deficiencies in surgical instrument segmentation.

• We conduct extensive experiments on two public benchmarks, EndoVis2017 and EndoVis2018. The results demonstrate that HPMA substantially outperforms existing state-of-the-art methods and establishes new benchmark performance in terms of Challenge IoU and mean class IoU, with minimal inference overhead compared to existing adaptation methods.

## RELATED WORK

## Surgical Instrument Segmentation

Surgical instrument segmentation is challenging due to specular reflections, occlusions, ambiguous boundaries, and high intra-class variations. Specialized methods have tackled these issues via mask-attended classification [12], maskedattention transformers with temporal modeling [13], visionlanguage representation [14], [15], or instance-level representation learning [16]. Despite their effectiveness, these specialist models encode surgical knowledge implicitly in task-specific parameters, limiting their robustness against complex and changing intraoperative conditions.

## SAM Adaptation for Medical and Surgical Segmentation

The domain gap between natural and medical images has driven extensive adaptation of the Segment Anything Model (SAM) [1], [2], [3]. Methods like MedSAM [17] perform full fine-tuning, whereas SAMed [18] and Medical SAM Adapter [19] introduce LoRA or lightweight adapters for parameterefficient adaptation. Other variants tune specialized components via text guidance [20], hierarchical decoding [21], or decoder adaptation prompts [22], [23]. However, domain knowledge in these approaches is predominantly encoded in optimized model parameters, causing the degeneration of foundation model priors and leaving models susceptible to semantic drift under challenging surgical artifacts.

## Prompt Learning for Surgical Segmentation

Prompt learning reduces manual interaction in SAM adaptation. AutoSAM [24] and Surgical-DeSAM [25] automatically generate prompts via auxiliary encoders or detectors. SurgicalSAM [8] and SurgicalPart-SAM [10] introduce prototype-based or part-level prompts, while other works leverage hierarchical prompt trees [11] or prompt distillation [7]. Nevertheless, existing prompt-learning approaches route multi-scale visual cues through a single, congested prompting pathway, creating a representational bottleneck. In contrast, HPMA explicitly preserves multi-scale visual evidence in a frozen prototype memory and couples each representation level with its optimal model component.

## METHODOLOGY

## Problem Formulation and Overall Architecture

We formulate surgical instrument segmentation as a classprompted binary mask prediction task. Given a surgical image $\boldsymbol { I } \in \mathbb { R } ^ { H \times \mathbf { \tilde { W } } \times 3 }$ and a text prompt $t _ { c }$ for category $c \in { \mathcal { C } } ,$ the network $f _ { \Theta }$ predicts a binary mask $\hat { Y } _ { c } = f _ { \Theta } ( I , t _ { c } ) \in$ $\{ 0 , 1 \} ^ { H \times W }$ , where $\Theta = \{ \Theta _ { \mathrm { f r o z e n } } , \Theta _ { \mathrm { a d a p t e r } } \}$ separates frozen foundation parameters from trainable adapters. Multi-class masks are obtained by querying each category independently and taking the element-wise maximum.

We adopt SAM3 [3] as our baseline, which comprises an image encoder $E _ { \mathrm { i m g } }$ , text encoder $E _ { \mathrm { t x t } } ,$ fusion encoder $E _ { \mathrm { f u s } } ,$ transformer decoder $D _ { \mathrm { t r a n s } }$ , and segmentation head $H _ { \mathrm { s e g } } . \mathrm { N a } .$ tive SAM3 extracts multi-scale visual features $F = \{ \bar { F } _ { l } \} _ { l = 1 } ^ { 3 }$ and text embeddings $T _ { c } ,$ fuses them via $E _ { \mathrm { f u s } }$ , and predicts masks using decoder object queries $Q .$

As shown in Fig. 2, HPMA introduces a frozen prototype memory bank $\mathcal { P }$ paired with lightweight adapters across three hierarchical levels: (1) Global level: Calibrating raw text features $T _ { c }$ with global prototypes $\mathcal { P } _ { c } ^ { g }$ into $\tilde { T } _ { c }$ before fusion; (2) Structural level: Modifying object queries $Q$ with structural prototypes $\mathcal P _ { c } ^ { p }$ into $\tilde { Q } _ { c } \mathrm { : }$ and (3) Local level: $_ \mathrm { A l }$ igning high-resolution features $F _ { 0 }$ with local prototypes $\mathcal { P } _ { c } ^ { l }$ via a training-only objective. Formally, given intermediate features $( F , T _ { c } ) = ( E _ { \mathrm { i m g } } ( I ) , E _ { \mathrm { t x t } } ( t _ { c } ) )$ , the refined execution flow is:

![](images/6d0942e06a68d017fc7f5d5bde5be896dc8609b4bff08acf956f20478dc4d3b1.jpg)  
Fig. 2: Overview of the proposed HPMA framework. (a) The Multi-scale Prototype-Memory Construction extracts multi scale FPN features from labeled surgical images using a frozen SAM3 image encoder and constructs global-, structural-, and local-level prototype banks. (b) The Hierarchical Representation Level-to-Module Coupling mechanism injects global prototypes into text features before multimodal fusion, uses structural prototypes to refine decoder object queries, and aligns high-resolution image features with local prototypes during training. (c) The Adaptation Detail of the modules at each level.

$$
\hat { Y } _ { c } = H _ { \mathrm { s e g } } \Big ( D _ { \mathrm { t r a n s } } \big ( E _ { \mathrm { f u s } } ( F , \tilde { T } _ { c } ) , \tilde { Q } _ { c } \big ) \Big ) .\tag{1}
$$

Construction and Adaption of Multi-scale Prototype-Memory

The core of our framework is the construction of a frozen visual prototype memory bank and the design of lightweight adapters that inject this evidence into the model’s backbone to enhance feature representations.

To construct a stable repository of category-specific evidence, we extract multi-scale feature maps from the Feature Pyramid Network (FPN) of the frozen SAM3 image encoder. For each representation scale $s \in$ {global, structural, local} and instrument category $c \in { \mathcal { C } } ,$ we aggregate image-level features via masked average pooling governed by the groundtruth binary masks $M _ { c }$ . To encapsulate the severe intra-class visual variations inherent to surgical scenes, these aggregated representations are subsequently partitioned offline into $K$ distinct prototype vectors via K-Means clustering prior to training, formulating the comprehensive memory bank $\mathcal { P } \in$ <sub>R</sub>3×|C|×K×d

To facilitate rapid and parameter-efficient adaptation, we design lightweight adapters that read from the frozen memory bank and inject visual evidence into the active optimization pathway. The bank $\mathcal { P }$ remains strictly frozen to prevent the prototypes from degenerating into task-specific parameters during downstream optimization, avoiding catastrophic forgetting of SAM3’s foundational priors. For a given scale $s ,$ the adapter takes a target SAM3 representation $Z _ { c } ^ { s }$ and computes a residual update using a lightweight mapping function $\Psi _ { s } \mathbf { : }$

$$
\tilde { Z } _ { c } ^ { s } = Z _ { c } ^ { s } + \alpha _ { s } \Psi _ { s } ( Z _ { c } ^ { s } , \mathcal { P } _ { c } ^ { s } ) ,\tag{2}
$$

where $\alpha _ { s }$ is a learnable scalar initialized to a small value. This formulation explicitly isolates evidence storage from feature adaptation: the memory dictates what an instrument looks like under varying conditions, while the adapter learns how to optimally integrate this evidence into the backbone.

## Hierarchical Representation Level-to-Module Coupling

Unlike methods constrained by a singular projection bottleneck, our approach capitalizes on the rich, three-tiered hierarchical FPN features. Inspired by [11], [10], we introduce scale-matched adapters to explicitly route these visual cues into the most receptive SAM3 modules, maximizing representational efficiency. Specifically, the global level addresses category appearance, the structural level guides object-query formation, and the local level refines high-resolution features.

At the global semantic level, feature maps capture the overall context of the instrument. We design a global adapter to inject this evidence into class prompts prior to imagetext fusion. Acting as a cross-attention mechanism with text tokens $T _ { c }$ as queries, we define $Q _ { c } = T _ { c } W _ { Q } , K _ { c } =$ $\Phi _ { g } ( \mathcal { P } _ { c } ^ { g } ) W _ { K }$ , and $V _ { c } ~ = ~ \Phi _ { g } ( \mathcal { P } _ { c } ^ { g } ) W _ { V }$ , and the update is

formulated as:

$$
A _ { c } ^ { g } = \mathrm { s o f t m a x } \left( \frac { Q _ { c } K _ { c } ^ { \top } } { \sqrt { d _ { k } } } \right) , \quad \widetilde { T } _ { c } = T _ { c } + \alpha _ { g } ( A _ { c } ^ { g } V _ { c } ) W _ { O } ,\tag{3}
$$

where $W _ { \{ Q , K , V , O \} }$ are learnable projection matrices, $d _ { k }$ is the scaling dimension, and $\alpha _ { g }$ is a learnable scalar. The projection $\Phi _ { g }$ comprises a LayerNorm and a two-layer MLP with GELU. This formulation explicitly anchors abstract text embeddings into the domain-specific surgical visual manifold $( \mathcal { P } _ { c } ^ { g } )$ to overcome modality gaps from intraoperative variations, ensuring strictly domain-aligned representations for the fusion encoder.

At the intermediate structural level, geometric subcomponents of the instruments are represented. We inject this information directly into the SAM3 transformer decoder by modifying the learnable object queries $Q .$ Since these queries function as spatial anchors, adding the structural component prototype endows them with the geometric bias of specific instrument parts. A projection $\Phi _ { p } ,$ structurally identical to $\Phi _ { g } ,$ , maps the structural prototype $\mathcal { P } _ { c } ^ { p }$ into a query residual:

$$
\begin{array} { r } { \widetilde Q _ { c } = Q + \alpha _ { p } \Phi _ { p } ( \mathcal P _ { c } ^ { p } ) . } \end{array}\tag{4}
$$

Rather than relying on generic initialization, this explicitly biases queries toward inherent instrument structures, which is crucial for highly articulated tools defined by distinct components like shafts, jaws, and tips.

At the finest granularity, local-level features encode critical boundaries and textures. To exploit this without inference latency, we formulate a local alignment objective during training. The high-resolution feature map $F _ { 0 }$ is spatially pooled over the target object region to yield a compact representation:

$$
\phi _ { i } = \frac { \sum _ { x } w _ { i } ( x ) F _ { 0 } ( x ) } { \sum _ { x } w _ { i } ( x ) + \epsilon } ,\tag{5}
$$

where $w _ { i } ( x ) ~ \in ~ \{ 0 , 1 \}$ denotes the resized ground-truth mask value at spatial coordinate x. This is then explicitly regularized by aligning it toward the corresponding local prototype $\mathcal { P } _ { c } ^ { l }$

By systematically aggregating information across these dimensions, HPMA circumvents the representational bottleneck of single-pathway prompting. This hierarchical coupling seamlessly bridges the perceptual gap—from semantic calibration to structural awareness and local detail refinement—comprehensively improving multi-class segmentation capabilities.

## Training Objective

The overall training objective integrates the standard SAM3 loss formulation with our proposed local prototype alignment term. The base loss, denoted as ${ \mathcal { L } } _ { \mathrm { S A M 3 } }$ , comprises focal (for mask classification), dice, bounding box, generalized IoU, category classification, and presence supervision losses:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S A M 3 } } = \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { m a s k } } + \lambda _ { \mathrm { d i c e } } \mathcal { L } _ { \mathrm { d i c e } } + \lambda _ { \mathrm { b o x } } \mathcal { L } _ { \mathrm { b o x } } } \\ { + \lambda _ { \mathrm { g i o u } } \mathcal { L } _ { \mathrm { g i o u } } + \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { p r e s } } \mathcal { L } _ { \mathrm { p r e s } } , } \end{array}\tag{6}
$$

where the λ coefficients balance the respective loss components.

To supervise the high-resolution features, we formulate a local alignment loss $\mathcal { L } _ { \mathrm { a l i g n } }$ based on a cosine distance metric. We explicitly utilize cosine similarity to optimize the directional alignment of features rather than their $L _ { 2 }$ magnitude. This design choice renders the high-resolution representations invariant to absolute feature scaling and highly resilient to illumination variations, significantly improving the model’s robustness when delineating fine-grained edges amidst strong surgical specular reflections.

For a mini-batch containing N target object regions, the loss is computed between the spatially pooled high-resolution feature $\phi _ { i }$ and its most semantically relevant projected local prototype $p _ { c _ { i } , k } ^ { l }$ (where $k \in \{ 1 , \ldots , K \}$ indexes the prototypes for the ground-truth category $c _ { i } )$ :

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m i n } _ { k \in \{ 1 , . . . , K \} } \left( 1 - \cos ( \phi _ { i } , p _ { c _ { i } , k } ^ { l } ) \right) .\tag{7}
$$

The complete optimization objective is formulated as:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { S A M 3 } } + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } ,\tag{8}
$$

where $\lambda _ { \mathrm { a l i g n } }$ is the weighting hyperparameter for the local alignment regularization.

## EXPERIMENTS

## Datasets and Evaluation Metrics

We evaluate our proposed HPMA on two widely adopted public benchmarks for surgical instrument segmentation: EndoVis2017 and EndoVis2018.

EndoVis2017 [26] contains 8 videos for cross-validation, featuring 7 distinct instrument categories: Bipolar Forceps (BF), Prograsp Forceps (PF), Large Needle Driver (LND), Vessel Sealer (VS), Grasping Retractor (GR), Monopolar Curved Scissors (MCS), and Ultrasound Probe (UP). We use the 4-fold cross-validation provided by Shvets et al. [27].

EndoVis2018 [28] comprises 11 training videos and 4 validation videos. It also evaluates 7 categories, substituting Vessel Sealer and Grasping Retractor with Suction Instrument (SI) and Clip Applier (CA). We use the instrument-type segmentation annotation given by Gonzalez et al. [29].´

To comprehensively evaluate the performance, we utilize four standard evaluation metrics [29], [7]: Challenge IoU, IoU, mean class IoU (mc IoU) and mean Dice (mDice). Challenge IoU is a specialized metric that evaluates intersection over union exclusively for the instrument categories actively visible in a given frame.

## Implementation Details

The framework is implemented in PyTorch and trained on NVIDIA A100 GPU. During the adaptation phase, the primary image encoder, language backbone, geometry encoder, and fusion encoder of SAM3 are frozen to retain foundational priors. We selectively unfreeze the final four vision trunk blocks, the final two language encoder blocks, and the final two fusion encoder layers.

TABLE I: Comparative Results on the EndoVis2017 Dataset. The best results are presented in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Ref.</td><td rowspan="2">Challenge IoU</td><td rowspan="2">IoU</td><td rowspan="2">mc IoU</td><td rowspan="2">mDice</td><td colspan="7">Instrument Categories</td></tr><tr><td>PF</td><td></td><td>LND</td><td>VS</td><td>GR</td><td>MCS</td><td>UP</td></tr><tr><td>ISINet</td><td>MICCAI&#x27; 20</td><td>55.62</td><td>52.20</td><td>28.96</td><td>52.00</td><td>38.70</td><td>38.50</td><td>50.09</td><td>27.43</td><td>2.01</td><td>28.72</td><td>12.56</td></tr><tr><td>MATIS Frame</td><td>ISBI&#x27; 23</td><td>68.79</td><td>62.74</td><td>37.30</td><td>53.06</td><td>66.18</td><td>50.99</td><td>52.23</td><td>32.84</td><td>15.71</td><td>19.27</td><td>23.90</td></tr><tr><td>S3Net</td><td>WACV&#x27;23</td><td>72.54</td><td>71.99</td><td>46.55</td><td>66.44</td><td>75.08</td><td>54.32</td><td>61.84</td><td>35.50</td><td>27.47</td><td>43.23</td><td>28.38</td></tr><tr><td>SCI-Net</td><td>JBHI&#x27; 26</td><td>72.26</td><td>69.08</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TrackAnything</td><td></td><td>67.41</td><td>64.50</td><td>62.97</td><td></td><td>55.42</td><td>44.46</td><td>62.43</td><td>67.03</td><td>65.17</td><td>83.68</td><td>62.59</td></tr><tr><td>PerSAM</td><td></td><td>42.47</td><td>42.47</td><td>41.80</td><td></td><td>53.99</td><td>25.89</td><td>50.17</td><td>52.87</td><td>24.24</td><td>47.33</td><td>38.16</td></tr><tr><td>SurgicalSAM</td><td>AAAI&#x27; 24</td><td>69.94</td><td>69.94</td><td>67.03</td><td>80.34</td><td>68.30</td><td>51.77</td><td>75.52</td><td>68.24</td><td>57.63</td><td>86.95</td><td>60.80</td></tr><tr><td>MA-SAM2</td><td>MICCAI&#x27; 25</td><td>62.49</td><td>62.49</td><td>59.89</td><td></td><td>54.41</td><td>50.41</td><td>64.73</td><td>73.72</td><td>32.66</td><td>72.64</td><td>70.85</td></tr><tr><td>Distillation-SAM</td><td>TMI&#x27; 26</td><td></td><td></td><td>75</td><td>80</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td>79.63</td><td>79.63</td><td>76.34</td><td>83.87</td><td>83.71</td><td>69.26</td><td>87.25</td><td>83.13</td><td>48.52</td><td>80.32</td><td>75.02</td></tr><tr><td>GT Centroid + SAM3</td><td>ICLR&#x27; 26</td><td>50.52</td><td>50.52</td><td>50.60</td><td>53.56</td><td>51.96</td><td>43.64</td><td>39.18</td><td>46.80</td><td>42.66</td><td>75.18</td><td>66.87</td></tr></table>

TABLE II: Comparative Results on the EndoVis2018 Dataset. The best results are presented in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Ref.</td><td rowspan="2">Challenge IoU</td><td rowspan="2">IoU</td><td rowspan="2">mc IoU</td><td rowspan="2">mDice</td><td colspan="7">Instrument Categories</td></tr><tr><td>BF</td><td>PF</td><td>LND</td><td>MCS</td><td>UP</td><td>SI</td><td>CA</td></tr><tr><td>ISINet</td><td>MICCAI’ 20</td><td>73.03</td><td>70.97</td><td>40.21</td><td>71.10</td><td>73.83</td><td>48.61</td><td>30.98</td><td>88.16</td><td>2.16</td><td>37.68</td><td>0.00</td></tr><tr><td>MATIS Frame</td><td>ISBI&#x27; 23</td><td>82.37</td><td>77.01</td><td>48.65</td><td>63.66</td><td>83.35</td><td>38.82</td><td>40.19</td><td>93.18</td><td>16.17</td><td>64.49</td><td>4.32</td></tr><tr><td>S3Net</td><td>WACV&#x27; 23</td><td>75.81</td><td>74.02</td><td>42.58</td><td>68.02</td><td>77.22</td><td>50.87</td><td>19.83</td><td>92.12</td><td>7.44</td><td>50.59</td><td>0.00</td></tr><tr><td>SCI-Net</td><td>JBHI&#x27; 26</td><td>83.59</td><td>81.10</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TrackAnything</td><td></td><td>65.72</td><td>60.88</td><td>38.60</td><td></td><td>72.90</td><td>31.07</td><td>64.73</td><td>61.05</td><td>17.93</td><td>10.24</td><td>12.28</td></tr><tr><td>PerSAM</td><td></td><td>49.21</td><td>49.21</td><td>34.55</td><td></td><td>51.26</td><td>34.40</td><td>46.75</td><td>52.28</td><td>25.62</td><td>16.45</td><td>15.07</td></tr><tr><td>SurgicalSAM</td><td>AAAI&#x27; 24</td><td>80.33</td><td>80.33</td><td>58.87</td><td>69.61</td><td>83.66</td><td>65.63</td><td>58.75</td><td>88.56</td><td>21.23</td><td>54.48</td><td>39.78</td></tr><tr><td>MA-SAM2</td><td>MICCAI&#x27; 25</td><td>64.40</td><td>64.40</td><td>62.13</td><td></td><td>56.93</td><td>44.91</td><td>66.73</td><td>75.35</td><td>74.51</td><td>36.82</td><td>37.65</td></tr><tr><td>Ours</td><td></td><td>84.12</td><td>84.12</td><td>77.20</td><td>87.43</td><td>80.69</td><td>81.29</td><td>78.23</td><td>92.12</td><td>67.73</td><td>80.34</td><td>60.04</td></tr><tr><td>GT Centroid + SAM3</td><td>ICLR&#x27; 26</td><td>60.45</td><td>60.45</td><td>58.39</td><td>51.32</td><td>42.56</td><td>51.42</td><td>47.60</td><td>82.93</td><td>62.97</td><td>51.94</td><td>69.29</td></tr></table>

Ground Truth  
MATIS  
SurgicalSAM  
SAM3  
Ours  
![](images/aab80b534792d635ea1aa5147632ab6823ea452c93abcaa04405d0fa2307b1f9.jpg)  
Bipolar ForcepsPrograsp ForcepsLarge Needle DriverMonopolar Curved ScissorsUltrasound ProbeClip Applier

Fig. 3: The visual results of the comparative experiments. The green boxes highlight the comparison details.

The model is optimized with AdamW [30] using a base learning rate of $8 \times 1 0 ^ { - 5 }$ , while applying smaller learning rates to the pre-trained vision and language backbones, 2.5×

$1 0 ^ { - 5 }$ and $5 \times 1 0 ^ { - 6 } .$ , respectively [31]. All input images are resized to $1 0 0 8 \times 1 0 0 8$ [3] and trained for 20 epochs. The multi-scale visual prototype memory bank is pre-constructed using K-means clustering on the training set features and K is set to 4. For the training objective, the original SAM3 loss weights are retained, the local alignment loss weight is set to 0.01.

## Main Results

We compare HPMA with two groups of representative methods: specialized surgical segmentation models, including ISINet [29], MATIS Frame [13], S3Net [12], SCI-Net [32], and SAM-based approaches, including TrackAnything [33], PerSAM [34], SurgicalSAM [8], MA-SAM2 [35], Distillation-SAM [7]. We additionally report a baseline utilizing ground-truth centroids with SAM3 (GT Centroid + SAM3) [3]. Following previous works [8], [10], [32], the results for the compared methods are reproduced using their officially released pre-trained weights. However, to ensure a fair comparison and present the strongest baselines, we directly cite the results from the original papers whenever our reproduced performance falls short of their originally reported metrics.

As shown in Table I and Table II, HPMA achieves the best overall performance on both benchmarks. Regarding specialized baselines, although they incorporate carefully designed components, they still face challenges such as temporal dependency or limited data diversity. ISINet aggregates class predictions across frames, but relies on a dedicated detection pipeline and temporal association. S3Net employs multi-scale mask-attended features, but remains dependent on task-specific training with limited surgical data. MATIS introduces masked attention and temporal modeling, yet its category-level performance is constrained, particularly for less frequent instruments. More recently, SCI-Net integrates global-context aggregation, frequency-domain attention, and scale-aware dilation, but it can still be affected by severe reflections and motion blur. In contrast, HPMA explicitly preserves category-specific surgical evidence while retaining the pre-trained knowledge of SAM3, resulting in more balanced performance across instrument categories.

SAM-based methods exhibit strong competitiveness by exploiting foundation-model priors. Notably, the sub-optimal performance of the baseline (GT Centroid + SAM3) reveals that spatial prompts are insufficient without category-level semantics, which explicitly highlights the necessity of injecting stable, category-specific visual evidence. TrackAnything and PerSAM rely on point prompts or reference masks, mak ing their performance sensitive to prompt quality and limiting fully automatic segmentation. MA-SAM2 exploits temporal memory for training-free video segmentation, showing suboptimal performance. SurgicalSAM introduces a prototypebased class prompt encoder and contrastive prototype learning, enabling automatic segmentation without explicit spatial prompts. However, it treats each instrument as a single entity without utilizing the information from different scales. By maintaining a multi-scale prototype memory and coupling hierarchical evidence with appropriate model components, HPMA overcomes these limitations and achieves the best performance among both specialized and SAM-based methods.

Ground Truth  
SAM3  
Ours  
![](images/3133693b795f46b45f517f85b21ea70365c0ec86fd136408d264005fc24619a4.jpg)  
Fig. 4: The attention map visualization of SAM3 and our HPMA method.

Visual comparisons further validate the superiority of HPMA over baseline SAM-based methods. As shown in Fig. 3, traditional zero-shot or single-pathway prompt models often fail to precisely delineate boundaries due to the complex visual variations in surgery, such as overlapping tissues. They frequently misclassify the highly similar categories. In contrast, by grounding the adaptation process in our hierarchical visual prototype memory, HPMA maintains robust semantic stability. The scale-matched injection ensures that the model successfully captures holistic semantic meaning while simultaneously refining sharp, localized boundaries, even under severe intraoperative occlusions.

To intuitively demonstrate the effectiveness of our hierarchical coupling mechanism, we visualize the decoder crossattention maps in Fig. 4. As shown in the middle column, SAM3 frequently scatters its attention across background tissues. In contrast, after incorporating HPMA (right column), the high-response regions explicitly converge on distinct physical components of the surgical instruments, such as jaws, joints, and shafts. This visual evidence validates that our hierarchical adapters successfully resolve the semantic domain gap and better capture complex instrument structures.

We further evaluate the computational complexity and inference efficiency of HPMA compared to the SurgicalSAM baseline. Specifically, when using the SAM ViT-H backbone, HPMA achieves a lower computational cost of 4917.91 GFLOPs, markedly reducing the computational burden compared to the 5989.78 GFLOPs required by SurgicalSAM. Crucially, this reduction in GFLOPs translates directly to a faster inference speed. HPMA operates at 2.69 FPS, outperforming the 1.98 FPS achieved by SurgicalSAM.

## Ablation Study

To validate our hierarchical coupling mechanism, we conduct an ablation study on the EndoVis2018 dataset. As shown in Table III, sequentially integrating hierarchical evidence steadily improves performance compared to a partially finetuned baseline lacking prototype injection. The addition of global prototypes initially bridges the semantic domain gap.

<table><tr><td>No.</td><td>Global</td><td>Structural</td><td>Local</td><td>Challenge IoU</td><td>IoU</td><td>mc IoU</td></tr><tr><td>1</td><td>×</td><td>×</td><td>X</td><td>81.11</td><td>81.11</td><td>70.22</td></tr><tr><td>2</td><td>√</td><td>X</td><td>×</td><td>81.90</td><td>81.90</td><td>75.38</td></tr><tr><td>3</td><td>X</td><td>√</td><td>×</td><td>82.66</td><td>82.66</td><td>75.03</td></tr><tr><td>4</td><td>×</td><td>X</td><td>√</td><td>83.16</td><td>83.16</td><td>74.05</td></tr><tr><td>5</td><td>√</td><td>√</td><td>×</td><td>83.44</td><td>83.44</td><td>76.89</td></tr><tr><td>6</td><td>√</td><td>√</td><td>√</td><td>84.12</td><td>84.12</td><td>77.20</td></tr></table>

TABLE IV: Ablation Study on the Injection Strategies.
<table><tr><td>Injection Strategy</td><td>Challenge IoU</td><td>IoU</td><td>mc IoU</td></tr><tr><td>Multi-scale mean fusion</td><td>82.20</td><td>82.20</td><td>74.08</td></tr><tr><td>Multi-scale learned fusion</td><td>82.45</td><td>82.45</td><td>71.74</td></tr><tr><td>G-Text, S-Dec</td><td>83.44</td><td>83.44</td><td>76.89</td></tr><tr><td>G-Dec, S-Text</td><td>82.60</td><td>82.60</td><td>72.50</td></tr><tr><td>G-Text, L-SegHead</td><td>82.98</td><td>82.98</td><td>76.44</td></tr><tr><td>G-Text, S-Dec, L-SegHead</td><td>82.51</td><td>82.51</td><td>76.59</td></tr><tr><td>G-Text, S-Dec, L-Opt.</td><td>84.12</td><td>84.12</td><td>77.20</td></tr></table>

\* G refers to Global Level, S refers to Structural Level, L refers to Local Level.

![](images/363c7732cebf909f16c0d1c058daeda2b7a7ffad461557b61d00217e234ba3a2.jpg)  
Fig. 5: Ablation Study on the value of K.

Crucially, incorporating structural-level adapters provides essential geometric priors, substantially boosting the Challenge IoU from 81.90% to 83.44%. This confirms that explicit structural bias is vital for parsing highly articulated surgical tools. Finally, integrating the local alignment objective refines high-resolution boundary details, achieving a peak Challenge IoU of 84.12% and mc IoU of 77.20%. These progressive gains demonstrate that visual evidence across global, structural, and local levels is complementary and indispensable for robust adaptation.

We further compared different prototype injection strategies. As demonstrated in Table IV, a naive multi-scale fusion yields suboptimal results, indicating that congested prompting pathways create representational bottlenecks. In contrast, routing Global (G) prototypes to the Text features and Structural (S) prototypes to the Decoder significantly outperforms inverted mappings. This validates our core design hypothesis: global features are optimally suited to calibrate abstract semantic classes, while structural features naturally align with the geometric spatial anchors of decoder object queries. Furthermore, directly fusing Local (L) prototypes into the segmentation head degrades performance. Formulating the local level purely as an auxiliary optimization objective circumvents this degradation, yielding the highest Challenge IoU.

Fig. 5 evaluates the impact of the prototype count K per category. A single prototype (K = 1) fails to capture the severe intra-class visual variations. Performance peaks at K=4, which is adopted as the optimal configuration.

We present HPMA, a Hierarchical Prototype-Memory Adaptation framework that robustly adapts the Segment Anything Model 3 (SAM3) for surgical instrument segmentation. By decoupling visual evidence storage into a frozen multi-scale prototype memory bank, HPMA mitigates the prompt degeneration typical of single-pathway methods. Furthermore, our scale-matched adapters seamlessly integrate semantic calibration, structural awareness, and local detail refinement into respective SAM3 modules. Extensive experiments on the EndoVis2017 and EndoVis2018 benchmarks demonstrate HPMA’s state-of-the-art accuracy and robustness against complex intraoperative variations. Building on our computational efficiency over existing baseline, future work will explore model compression and knowledge distillation to transfer these learned capabilities into lightweight backbones. This will facilitate seamless integration with intraoperative platforms, unlocking the potential of foundation models for real-time surgical navigation and robotic assistance.

## REFERENCES

[1] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., Segment anything, in: Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.

[2] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson, et al., Sam 2: Segment anything¨ in images and videos, in: International Conference on Learning Representations, Vol. 2025, 2025, pp. 28085–28128.

[3] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. S. Coll-Vinent, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, J. Lei, T. Ma, B. Guo, A. Kalla, M. Marks, J. Greer, M. Wang, P. Sun, R. Radle,¨ T. Afouras, E. Mavroudi, K. Xu, T.-H. Wu, Y. Zhou, L. Momeni, R. HAZRA, S. Ding, S. Vaze, F. Porcher, F. Li, S. Li, A. Kamath, H. K. Cheng, P. Dollar, N. Ravi, K. Saenko, P. Zhang, C. Feichtenhofer, SAM 3: Segment anything with concepts, in: The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=r35clVtGzw

[4] J. Ma, Z. Yang, S. Kim, B. Chen, M. Baharoon, A. Fallahpour, R. Asakereh, H. Lyu, B. Wang, Medsam2: Segment anything in 3d medical images and videos, arXiv preprint arXiv:2504.03600 (2025).

[5] S. Zhao, L. Bai, K. Yuan, F. Li, J. Yu, W. Dong, G. Wang, M. I. Hoque, N. Padoy, N. Navab, et al., Rethinking data imbalance in class incremental surgical instrument segmentation, Medical Image Analysis 105 (2025) 103728.

[6] H. Liu, M. Gao, X. Luo, Z. Wang, G. Qin, J. Wu, Y. Jin, Resurgsam2: Referring segment anything in surgical video via credible long-term tracking, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2025, pp. 435–445.

[7] J. Tang, H. Han, S. Shan, X. Chen, Distillation-sam: Knowledge distillation based auto-prompt embedding learning for surgical image segmentation, IEEE Transactions on Medical Imaging (2026).

[8] W. Yue, J. Zhang, K. Hu, Y. Xia, J. Luo, Z. Wang, Surgicalsam: Efficient class promptable surgical instrument segmentation, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, 2024, pp. 6890–6898.

[9] Y. Lu, J. Li, S. Ju, Y. Su, H. Yao, Y. Liu, M. Zhu, J. Cheng, Segmote: Token-level mixture of experts for medical image segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 36332–36342.

[10] W. Yue, J. Zhang, K. Hu, Q. Wu, Z. Ge, Y. Xia, J. Luo, Z. Wang, Surgicalpart-sam: Part-to-whole collaborative prompting for surgical instrument segmentation, arXiv preprint arXiv:2312.14481 (2023).

[11] Y. Zhu, K. Li, Z. Li, P.-A. Heng, Unlocking positive transfer in incrementally learning surgical instruments: A self-reflection hierarchical prompt framework, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 21006– 21015.

[12] B. Baby, D. Thapar, M. Chasmai, T. Banerjee, K. Dargan, A. Suri, S. Banerjee, C. Arora, From forks to forceps: A new framework for instance segmentation of surgical instruments, in: 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE, 2023, pp. 6180–6190.

[13] N. Ayobi, A. Perez-Rond´ on, S. Rodr´ ´ıguez, P. Arbelaez, Matis:´ Masked-attention transformers for surgical instrument segmentation, in: 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), IEEE, 2023, pp. 1–5.

[14] Y. Rao, W. Zhao, G. Chen, Y. Tang, Z. Zhu, G. Huang, J. Zhou, J. Lu, Denseclip: Language-guided dense prediction with contextaware prompting, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 18082–18091.

[15] Z. Zhou, O. Alabi, M. Wei, T. Vercauteren, M. Shi, Text promptable surgical instrument segmentation with vision-language models, Advances in Neural Information Processing Systems 36 (2023) 28611– 28623.

[16] L. Sestini, B. Rosa, E. De Momi, G. Ferrigno, N. Padoy, Saf-is: A spatial annotation free framework for instance segmentation of surgical tools, Medical Image Analysis 101 (2025) 103471.

[17] J. Ma, Y. He, F. Li, L. Han, C. You, B. Wang, Segment anything in medical images, Nature communications 15 (1) (2024) 654.

[18] K. Zhang, D. Liu, Customized segment anything model for medical image segmentation, arXiv preprint arXiv:2304.13785 (2023).

[19] J. Wu, Z. Wang, M. Hong, W. Ji, H. Fu, Y. Xu, M. Xu, Y. Jin, Medical sam adapter: Adapting segment anything model for medical image segmentation, Medical image analysis 102 (2025) 103547.

[20] J. N. Paranjape, N. G. Nair, S. Sikder, S. S. Vedula, V. M. Patel, Adaptivesam: Towards efficient tuning of sam for surgical scene segmentation, in: M. H. Yap, C. Kendrick, A. Behera, T. Cootes, R. Zwiggelaar (Eds.), Medical Image Understanding and Analysis, Springer Nature Switzerland, Cham, 2024, pp. 187–201.

[21] Z. Cheng, Q. Wei, H. Zhu, Y. Wang, L. Qu, W. Shao, Y. Zhou, Unleashing the potential of sam for medical adaptation via hierarchical decoding, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 3511–3522.

[22] J. G. Tejero, M. Schmid, P. M. Neila, M. S. Zinkernagel, S. Wolf, R. Sznitman, Sam-da: Decoder adapter for efficient medical domain adaptation, in: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE, 2025, pp. 6775–6784.

[23] D. Yang, C. Jiang, S. Le, J. Liu, X. Li, Z. Li, Rsg-sam2: Robust semantics-guided sam2 for multi-class surgical instruments semantic segmentation, in: 2025 International Conference on Frontiers Technology in Circuits and Systems (FTCS), IEEE, 2025, pp. 425–430.

[24] T. Shaharabany, A. Dahan, R. Giryes, L. Wolf, Autosam: Adapting sam to medical images by overloading the prompt encoder, arXiv preprint arXiv:2306.06370 (2023).

[25] Y. Sheng, S. Bano, M. J. Clarkson, M. Islam, Surgical-desam: decoupling sam for instrument segmentation in robotic surgery, International Journal of Computer Assisted Radiology and Surgery 19 (7) (2024) 1267–1271.

[26] M. Allan, A. Shvets, T. Kurmann, Z. Zhang, R. Duggal, Y.-H. Su, N. Rieke, I. Laina, N. Kalavakonda, S. Bodenstedt, et al., 2017 robotic instrument segmentation challenge, arXiv preprint arXiv:1902.06426 (2019).

[27] A. A. Shvets, A. Rakhlin, A. A. Kalinin, V. I. Iglovikov, Automatic instrument segmentation in robot-assisted surgery using deep learning, in: 2018 17th IEEE international conference on machine learning and applications (ICMLA), IEEE, 2018, pp. 624–628.

[28] M. Allan, S. Kondo, S. Bodenstedt, S. Leger, R. Kadkhodamohammadi, I. Luengo, F. Fuentes, E. Flouty, A. Mohammed, M. Pedersen, et al., 2018 robotic scene segmentation challenge, arXiv preprint arXiv:2001.11190 (2020).

[29] C. Gonzalez, L. Bravo-S ´ anchez, P. Arbelaez, Isinet: an instance- ´ based approach for surgical instrument segmentation, in: International conference on medical image computing and computer-assisted intervention, Springer, 2020, pp. 595–605.

[30] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, in: International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7

[31] H. Bao, L. Dong, S. Piao, F. Wei, BEit: BERT pre-training of image transformers, in: International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=p-BhZSz59o4

[32] J. Mei, Y. Zhang, X. He, T. Zhou, Accurate segmentation of surgical instruments via spectral-attentive contextual interaction network, IEEE Journal of Biomedical and Health Informatics (2026).

[33] J. Yang, M. Gao, Z. Li, S. Gao, F. Wang, F. Zheng, Track anything: Segment anything meets videos, arXiv preprint arXiv:2304.11968 (2023).

[34] R. Zhang, Z. Jiang, Z. Guo, S. Yan, J. Pan, H. Dong, Y. Qiao, G. Peng, H. Li, Personalize segment anything model with one shot, in: International Conference on Learning Representations, Vol. 2024, 2024, pp. 18250–18279.

[35] M. Yin, F. Wang, X. Ye, Y. Meng, Z. Fu, Memory-augmented sam2 for training-free surgical video segmentation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2025, pp. 328–337.