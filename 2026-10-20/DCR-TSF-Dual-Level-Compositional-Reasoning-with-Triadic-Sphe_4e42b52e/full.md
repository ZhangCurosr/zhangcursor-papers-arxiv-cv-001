# DCR-TSF: Dual-Level Compositional Reasoning with Triadic Spherical Fusion for Training-Free Zero-Shot Composed Image Retrieval

Gwang-Ho Na<sup>1</sup> , Ho-Joong Kim<sup>1</sup> , and Seong-Whan Lee<sup>1</sup>

Dept. of Artificial Intelligence, Korea University, Seoul, Korea {g\_h\_na, hojoong\_kim, sw.lee}@korea.ac.kr

Abstract. Zero-shot composed image retrieval (ZS-CIR) aims to retrieve a target image that reflects an intended textual modification while maintaining semantic consistency with a reference image. Existing training-free ZS-CIR methods typically rely on fine-grained generated target descriptions, which often overemphasize reference-specific visual details and lead to appearance-biased retrieval. Moreover, since these descriptions are generated through multimodal large language models (MLLMs), the direct influence of the original image-text query can be weakened in the final retrieval representation. In this paper, we propose DCR-TSF, a training-free ZS-CIR framework that combines Dual-level Compositional Reasoning (DCR) and Triadic Spherical Fusion (TSF). DCR generates fine- and coarse-grained descriptions from the reference image and modification text, capturing complementary target-side semantics across diferent granularities. To preserve the original image-text query information, TSF constructs a composed-query anchor by spherically interpolating the reference image feature and modification text feature, and fuses it with the fine- and coarse-grained representations using a weighted Karcher mean on the unit hypersphere. The resulting final query representation balances detailed visual evidence, high-level semantic abstraction, and direct preservation of the original composed query. Extensive experiments on CIRCO, CIRR, and Fashion-IQ demonstrate that our method achieves state-of-the-art performance among training-free methods.

Keywords: Zero-shot composed image retrieval · Multimodal reasoning

## 1 Introduction

Zero-shot composed image retrieval (ZS-CIR) [4, 6, 21, 25] is the task of retrieving an image that reflects a user-specified textual modification of a given reference image while maintaining semantic consistency. Unlike conventional image retrieval [15, 19], ZS-CIR operates under a fuzzy matching paradigm, which means multiple diferent images can match with the modification text provided by the user. This inherent ambiguity makes it dificult to construct large-scale annotated datasets for ZS-CIR due to the combinatorial explosion of possible referencemodification pairs. To address this issue, recent works [27, 33, 34] have explored training-free paradigms leveraging pretrained vision-language models. Among them, existing ZS-CIR methods [27,33,34] commonly adopt a text-guided matching approach that generates a target image description predicting what the target image should look like using multimodal large language models (MLLMs) [3, 29]. However, existing methods still sufer from a fundamental limitation: they rely on appearance-biased fine-grained descriptions that overemphasize reference-specific visual details.

![](images/ae559adbbe639c3bace7ff1f7abd2ce3c7dc0d6ad9a32ca84581af21351b8b0b.jpg)  
Fig. 1: Biases in Fine- and Coarse-grained Descriptions for ZS-CIR. Colored text refers to observed elements in the reference image. Background color indicates phrases appearing in the modification text. The bar colors in the text-to-image similarity score plots correspond to the candidate images. (a): Fine-grained description is biased towards the visual details of the reference image. (b): Coarse-grained description lacks details of the reference image.

To demonstrate the issue of existing fine-grained descriptions, we additionally construct a coarse-grained description that describes only the key modificationrelated elements while omitting reference-specific details. Fig. 1 shows how diferent granularities of generated target-image descriptions afect retrieval results. As shown in Fig. 1(a), the existing fine-grained description overemphasizes reference-specific appearance details such as the blue frisbee and grassy field. This overemphasis introduces a visual bias, causing CLIP [24] to highlight visually similar but semantically irrelevant candidates (e.g., the third image). In contrast, the coarse-grained description excludes these reference-specific details such as blue and grassy field, retrieving correct images. Conversely, Fig. 1(b) shows the complementary information-deficient case for coarse-grained description. The coarse-grained description only includes child and snowboard. In this case, the modification text only specifies “has a snowboard instead of a skateboard,” where visual cues in the reference image such as young, helmet, barefoot, and indoors play a critical role in understanding the modification intent. Since the coarsegrained description omits these details, the coarse-grained case fails to capture the fine-grained semantics. These examples illustrate that relying on a single reasoning granularity inevitably introduces two complementary failure cases.

In this paper, we propose Dual-level Compositional Reasoning with Triadic Spherical Fusion (DCR-TSF) for training-free ZS-CIR. Our method captures dual-granular textual reasoning implied by the reference image and modification text, addressing the complementary limitations of appearance-biased fine-grained descriptions and information-deficient coarse-grained descriptions. To achieve this, DCR-TSF first generates fine- and coarse-grained textual representations from the reference image and modification text. The fine-grained representation preserves detailed appearance cues, while the coarse-grained representation abstracts semantic intent at a conceptual level. However, because the fine- and coarse-grained representations are obtained through MLLM-generated descriptions, the direct influence of the original image-text query can be weakened in the final retrieval representation. To explicitly preserve the original composed query, we introduce Triadic Spherical Fusion (TSF). TSF first constructs a composedquery anchor by spherically interpolating the reference image feature and the modification text feature. TSF then fuses the fine-grained representation, coarsegrained representation, and composed-query anchor using a weighted Karcher mean [11] on the unit hypersphere. The resulting final query representation balances visual detail, semantic abstraction, and direct preservation of the original image-text query. Extensive comparisons and ablation studies on Fashion-IQ [30], CIRR [21], and CIRCO [4] demonstrate the efectiveness of DCR-TSF, which achieves state-of-the-art performance among training-free ZS-CIR methods. Our contributions are summarized as three-fold: Qur contributions ar arized as three-fold

– We introduce Dual-level Compositional Reasoning (DCR), which generates fine- and coarse-grained textual representations to preserve detailed visual attributes and capture high-level semantic intent.

– We propose Triadic Spherical Fusion (TSF), which preserves the original composed-query information by fusing a composed-query anchor with fineand coarse-grained representations on the unit hypersphere.

– Extensive comparisons and ablation studies on three standard benchmarks demonstrate the efectiveness of DCR-TSF, which achieves state-of-the-art performance among training-free ZS-CIR methods.

## 2 Related Work

Composed Image Retrieval (CIR). Early supervised CIR methods [7, 21] jointly learned multimodal embedding spaces from annotated reference-target pairs, requiring large-scale labeled CIR datasets. Due to the dificulty of collecting triplet labels, recent studies have explored ZS-CIR, which performs retrieval without task-specific training. A basic approach in ZS-CIR is to construct a composed query representation by combining the reference image feature and the modification text feature in the CLIP [24] embedding space. This composition is commonly implemented by weighted feature aggregation, while spherical linear interpolation can also be used to better respect the unit-hypersphere geometry of normalized CLIP embeddings [10]. With the progress of MLLMs [3, 29], another line of training-free ZS-CIR methods has emerged, which generates target descriptions by jointly interpreting the reference image and modification text. CIReVL [12] adopts a two-stage pipeline that uses a captioner and a large language model to generate a target description. LDRE [34] and SEIZE [33] improve description-based retrieval by leveraging multiple generated descriptions. OSrCIR [28] introduces a chain-of-thought one-stage framework to mitigate the potential information loss of a two-stage pipeline. IP-CIR [17] leverages a difusion model to generate a proxy image, directly capturing visual changes. CoTMR [27] utilizes an MLLM to infer existent and nonexistent objects and incorporates these reasoning results into the retrieval process. Despite their efectiveness, existing description-based ZS-CIR methods generally rely on fine-grained target descriptions or multiple generated descriptions centered on fine-grained visual details. This reliance can overemphasize reference-specific appearance cues, leading to appearance-biased retrieval. Furthermore, because these retrieval representations are derived from MLLM-generated descriptions, the direct influence of the original image-text query can be weakened in the final retrieval representation. In contrast, our method combines Dual-level Compositional Reasoning (DCR) with Triadic Spherical Fusion (TSF), using coarse-grained descriptions to complement fine-grained target semantics while explicitly preserving the original composed query through a composed-query anchor.

Vision-Language Models (VLMs). VLMs demonstrate strong cross-modal reasoning capabilities, enabling zero-shot generalization across a wide range of visual-linguistic tasks. CLIP [24] pioneers large-scale contrastive pretraining on image-text pairs, establishing a unified embedding space that efectively bridges visual and textual information. Recent MLLMs, such as InternVL [29] and Qwen-VL [3], further advance fine-grained perception and instruction following, enabling contextual reasoning over image content. These advances make it possible to generate complementary textual representations without any task-specific supervision. Our method leverages pretrained VLMs for image-text embedding and MLLMs for generating fine- and coarse-grained descriptions.

## 3 Method

## 3.1 Preliminary

Let a composed query be defined as $\boldsymbol { Q } = \{ I _ { r } , D _ { m } \}$ , where $I _ { r }$ denotes the reference image and $D _ { m }$ denotes the modification text. Given $N _ { D }$ candidate images $\{ I _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { N _ { D } }$ , the goal of CIR is to retrieve the top-K target images that best match the query, with $K \ll N _ { D }$ . To enable cross-modal comparison between the query and candidate images, all visual and textual inputs are projected into a shared multimodal embedding space using CLIP [24].

## 3.2 DCR-TSF

In this part, we briefly describe DCR-TSF, which mainly consists of two parts: (1) Dual-level Compositional Reasoning (DCR), which generates fine- and coarsegrained descriptions to capture complementary target-side semantics; (2) Triadic Spherical Fusion (TSF), which constructs a composed-query anchor and fuses it with the fine- and coarse-grained representations on the unit hypersphere. The overall architecture of DCR-TSF is illustrated in Fig. 2.

![](images/8a84ce8c116ec6d30e8bccc73d64e641bb1f5240ecad5601077e9ea97f69d73e.jpg)  
Fig. 2: Overview of DCR-TSF. Given a reference image $I _ { r }$ and a modification text $D _ { m }$ , DCR generates fine- and coarse-grained descriptions, which are encoded into $T _ { \mathrm { f i n } }$ e and $T _ { \mathrm { c o a r s e } } .$ . TSF constructs a composed-query anchor $Q _ { \mathrm { c o m p } }$ from $V _ { r }$ and $T _ { m } ,$ and fuses it with $T _ { \mathrm { f i n e } }$ and $T _ { \mathrm { c o a r s e } }$ using a weighted Karcher mean $[ 1 1 ]$ . The resulting final query representation $T _ { \mathrm { f i n a l } }$ is used to compute the final retrieval score $S ^ { ( i ) }$

## 3.3 Dual-level Compositional Reasoning (DCR)

Dual-level Description Generation. Following previous description-based methods [27, 28, 33], we exploit an MLLM to infer target-image descriptions from the reference image $I _ { r }$ and the modification text $D _ { m }$ . The modification text $D _ { m }$ represents a user-specified textual instruction. We first generate a fine-grained description $D _ { \mathrm { f i n e } } .$ following the conventional target-description generation scheme used in existing ZS-CIR methods. This description preserves detailed visual evidence from the reference image and describes what the target image should look like. However, as discussed in Sec. 1, fine-grained descriptions can overemphasize reference-specific visual details that are not directly relevant to the intended modification. To complement this, we additionally generate a coarse-grained description $D _ { \mathrm { c o a r s e } }$ , which describes the target image at a more abstract semantic level by focusing on stable visual anchors and key modification-related elements. Fig. 3 compares our coarse-grained description generation with OSrCIR [28]. It shows that our prompt suppresses excessive reference-specific details and produces a coarse-grained description complementary to the fine-grained description.

The two descriptions are generated as follows:

$$
D _ { \mathrm { c o a r s e } } = \mathrm { M L L M } ( I _ { r } , D _ { m } , P _ { \mathrm { c o a r s e } } ) , \quad D _ { \mathrm { f i n e } } = \mathrm { M L L M } ( I _ { r } , D _ { m } , P _ { \mathrm { f i n e } } ) ,\tag{1}
$$

where $P _ { \mathrm { c o a r s e } }$ and $P _ { \mathrm { f i n e } }$ denote the prompting templates for coarse- and finegrained generation, respectively. $D _ { \mathrm { c o a r s e } }$ captures global, scene-level semantics and the intended modification, while $D _ { \mathrm { f i n e } }$ preserves local visual details in the reference image. Detailed prompts for $P _ { \mathrm { c o a r s e } }$ and $P _ { \mathrm { f i n e } }$ are provided in the supplementary material.

![](images/bd24ea6f80a012b2f310a8b6eaf31c20db77eaf85ee73444b4e18efdf48b1e52.jpg)  
Fig. 3: Comparison between OSrCIR and our coarse-grained description prompt. OSrCIR [28] generates a detailed target description through a CoT prompt, preserving rich reference-specific visual evidence that may be irrelevant to the modification condition. In contrast, our coarse-grained description prompt identifies stable visual anchors and focuses on modification-related semantics, omitting excessive reference specific details and providing a complementary abstraction for DCR.

Dual-level Representation Encoding. The coarse-grained description $D _ { \mathrm { c o a r s e } }$ and fine-grained description $D _ { \mathrm { f i n e } }$ are fed into the frozen text encoder of CLIP [24] to obtain their corresponding representations:

$$
T _ { \mathrm { c o a r s e } } = \mathrm { T E } ( D _ { \mathrm { c o a r s e } } ) , \quad T _ { \mathrm { f i n e } } = \mathrm { T E } ( D _ { \mathrm { f i n e } } ) ,\tag{2}
$$

where $\operatorname { T E } ( \cdot )$ denotes the CLIP [24] text encoder. $T _ { \mathrm { c o a r s e } }$ captures global, conceptlevel semantics and the intended modification, while $T _ { \mathrm { f i n e } }$ preserves local visual details grounded in the reference image.

## 3.4 Triadic Spherical Fusion (TSF)

Composed-query Anchor. Although $T _ { \mathrm { f i n e } }$ and $T _ { \mathrm { c o a r s e } }$ provide complementary target-side semantics, they are derived from MLLM-generated descriptions. Therefore, relying only on these representations may dilute the original image-text query information. To explicitly preserve the original composed query, we construct a composed-query anchor from the reference image and modification text. Specifically, we first extract the reference image feature $V _ { r }$ and the modification text feature $T _ { m }$

$$
V _ { r } = \mathrm { V E } ( I _ { r } ) , ~ T _ { m } = \mathrm { T E } ( D _ { m } ) ,\tag{3}
$$

where $\operatorname { V E } ( { \cdot } )$ and TE(·) denote the CLIP visual and text encoders, respectively. Since CLIP features are normalized on the unit hypersphere, we use Slerp [26] to interpolate the reference image feature and the modification text feature:

$$
Q _ { \mathrm { c o m p } } = \frac { \sin ( ( 1 - \lambda ) \theta ) } { \sin ( \theta ) } V _ { r } + \frac { \sin ( \lambda \theta ) } { \sin ( \theta ) } T _ { m } ,\tag{4}
$$

where $\theta = \cos ^ { - 1 } ( V _ { r } \cdot T _ { m } )$ is the angle between the reference image feature and the modification text feature, and λ is a hyperparameter that balances the contribution of $V _ { r }$ and $T _ { m }$ . A larger λ emphasizes the modification text feature $T _ { m }$ , while a smaller λ preserves more reference-level visual information from $V _ { r }$ The resulting $Q _ { \mathrm { { c o m p } } }$ serves as a composed-query anchor that directly preserves the original image-text query information.

Triadic Spherical Fusion. To obtain the final query representation, TSF fuses three complementary representations: the fine-grained representation $T _ { \mathrm { f i n e } } .$ , the coarse-grained representation $T _ { \mathrm { c o a r s e } }$ , and the composed-query anchor Q<sub>comp</sub>. Since sequentially applying Slerp to multiple representations can introduce order dependence, we use a weighted Karcher mean [11] on the unit hypersphere. Unlike Euclidean averaging, the Karcher mean finds a point on the unit hypersphere that minimizes the weighted sum of squared geodesic distances to the three representations:

$$
\begin{array} { r l } & { T _ { \mathrm { f i n a l } } = \arg \underset { \| Y \| _ { 2 } = 1 } { \operatorname* { m i n } } w _ { \mathrm { f i n e } } \left( \cos ^ { - 1 } ( Y ^ { \top } T _ { \mathrm { f i n e } } ) \right) ^ { 2 } + w _ { \mathrm { c o a r s e } } \left( \cos ^ { - 1 } ( Y ^ { \top } T _ { \mathrm { c o a r s e } } ) \right) ^ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad + w _ { \mathrm { c o m p } } \left( \cos ^ { - 1 } ( Y ^ { \top } Q _ { \mathrm { c o m p } } ) \right) ^ { 2 } , } \end{array}\tag{5}
$$

where $Y$ is the unit-norm optimization variable, and w<sub>fine</sub>, w<sub>coarse</sub>, and w<sub>comp</sub> are fusion weights satisfying $w _ { \mathrm { f i n e } } + w _ { \mathrm { c o a r s e } } + w _ { \mathrm { c o m p } } = 1$ . Here, $\cos ^ { - 1 } ( a ^ { \top } \dot { b } )$ computes the geodesic distance between two unit-normalized representations on the hypersphere. In practice, we compute this mean using an iterative solver based on logarithmic and exponential maps. This fusion produces the final query representation $T _ { \mathrm { f i n a l } }$ , which balances detailed visual evidence, high-level semantic abstraction, and direct preservation of the original composed query. More details are provided in the supplementary material.

Final Retrieval Score. Given each candidate image $I _ { t } ^ { ( i ) }$ , we extract its visual feature. The final retrieval score is computed as the cosine similarity between the final query representation and each candidate visual feature:

$$
V _ { t } ^ { ( i ) } = \mathrm { V E } ( I _ { t } ^ { ( i ) } ) , \qquad S ^ { ( i ) } = \frac { V _ { t } ^ { ( i ) } \cdot T _ { \mathrm { f i n a l } } } { \| V _ { t } ^ { ( i ) } \| \| T _ { \mathrm { f i n a l } } \| } .\tag{6}
$$

Finally, all candidate images are ranked by the final retrieval score $S ^ { ( i ) }$

## 4 Experiments

## 4.1 Setup

Datasets. We evaluate our method on three CIR benchmarks: CIRCO [4], CIRR [21], and Fashion-IQ [30]. CIRCO is constructed from real-world images in the COCO 2017 unlabeled set [20] and is the first CIR dataset to provide multiple ground-truth targets per query. CIRR is the first large-scale natural image dataset specifically curated for CIR. Fashion-IQ is a fashion-domain dataset consisting of garment images categorized into dresses, shirts, and tops/tees.

Table 1: Comparison with the state-of-the-art methods on the CIRCO and CIRR test sets. Best results are highlighted in bold, and the second best are underlined.
<table><tr><td rowspan="2" colspan="2">Backbone Method</td><td rowspan="2">Training -free</td><td colspan="4">CIRCO</td><td colspan="7">CIRR</td></tr><tr><td colspan="4">mAP@5 mAP@10 mAP@25 mAP@50</td><td colspan="2">R@1 R@5</td><td colspan="4">R@10 Rsub@1 Rsub@2 Rsub@3</td></tr><tr><td rowspan="8">ViT-B/32</td><td>PALAVRA [6]</td><td>x</td><td>4.61</td><td>5.32</td><td>6.33</td><td>6.80</td><td>16.62</td><td>43.49</td><td>58.51</td><td>41.61</td><td>65.30</td><td>80.94</td><td>42.55</td></tr><tr><td>SEARLE [4]</td><td>x</td><td>9.35</td><td>9.94</td><td>11.13</td><td>11.84</td><td>24.00</td><td>53.42</td><td>66.82</td><td>54.89</td><td>76.60</td><td>88.19</td><td>54.16</td></tr><tr><td>Slerp [10]</td><td>√</td><td>6.35</td><td>7.11</td><td>8.12</td><td>8.75</td><td>24.22</td><td>50.94</td><td>63.49</td><td>57.86</td><td>78.27</td><td>89.25</td><td>54.40</td></tr><tr><td>CIReVL [12]</td><td>√</td><td>14.94</td><td>15.42</td><td>17.00</td><td>17.82</td><td>23.94</td><td>52.51</td><td>66.00</td><td>60.17</td><td>80.05</td><td>90.19</td><td>56.34</td></tr><tr><td>LDRE [34]</td><td>√</td><td>17.96</td><td>18.32</td><td>20.21</td><td>21.11</td><td>25.69</td><td>55.13</td><td>69.04</td><td>60.53</td><td>80.65</td><td>90.70</td><td>57.83</td></tr><tr><td>OSrCIR [28]</td><td>√</td><td>18.04</td><td>19.17</td><td>20.94</td><td>21.85</td><td>25.42</td><td>54.54</td><td>68.19</td><td>62.31</td><td>80.86</td><td>91.13</td><td>58.43</td></tr><tr><td>SEIZE [33]</td><td>√</td><td>19.04</td><td>19.64</td><td>21.55</td><td>22.49</td><td>27.47</td><td>57.42</td><td>70.17</td><td>65.59</td><td>84.48</td><td>92.77</td><td>61.51</td></tr><tr><td>CoTMR [27]</td><td>√</td><td>22.23</td><td>22.78</td><td>24.68</td><td>25.74</td><td>31.50</td><td>60.80</td><td>73.04</td><td>66.61</td><td>84.50</td><td>92.55</td><td>63.71</td></tr><tr><td rowspan="8"></td><td>DCR-TSF (Ours)</td><td>√</td><td>25.07</td><td>25.81</td><td>28.05</td><td>29.09</td><td>35.93 67.28</td><td></td><td>79.28</td><td>67.66</td><td>84.99</td><td>93.74</td><td>67.47</td></tr><tr><td>SEARLE [4]</td><td>x</td><td>11.68</td><td>12.73</td><td>14.33</td><td>15.12</td><td>24.24</td><td>52.48</td><td>66.29</td><td>53.76</td><td>75.01</td><td>88.19</td><td>53.12</td></tr><tr><td>Slerp [10]</td><td>√</td><td>8.76</td><td>9.84</td><td>11.30</td><td>11.99</td><td>24.43</td><td>49.93</td><td>62.29</td><td>57.71</td><td>77.59</td><td>88.80</td><td>53.82</td></tr><tr><td>CIReVL [12]</td><td>√</td><td>18.57</td><td>19.01</td><td>20.89</td><td>21.80</td><td>24.55</td><td>52.31 64.92</td><td></td><td>59.54</td><td>79.88</td><td>89.69</td><td>55.92</td></tr><tr><td>LDRE [34]</td><td>√</td><td>23.35</td><td>24.03</td><td>26.44</td><td>27.50</td><td>26.53</td><td>55.57</td><td>67.54</td><td>60.43</td><td>80.31</td><td>89.90</td><td>58.00</td></tr><tr><td>OSrCIR [28]</td><td>√</td><td>23.87</td><td>25.33</td><td>27.84</td><td>28.97</td><td>29.45</td><td>57.68</td><td>69.86</td><td>62.12</td><td>81.92</td><td>91.10</td><td>59.90</td></tr><tr><td>SEIZE [33]</td><td>√</td><td>24.98</td><td>25.82 27.41</td><td>28.24</td><td>29.35</td><td>28.65</td><td>57.16</td><td>69.23</td><td>66.22</td><td>84.05</td><td>92.34</td><td>61.69</td></tr><tr><td>IP-CIR [17] CoTMR [27]</td><td>√ √</td><td>26.43 27.61</td><td>28.22</td><td>29.87 30.61</td><td>31.07 31.70</td><td>29.76 35.02</td><td>58.82 64.75</td><td>71.21</td><td>62.48</td><td>81.64 85.75</td><td>90.89 93.33</td><td>60.65</td></tr><tr><td></td><td>DCR-TSF (Ours)</td><td></td><td>31.80</td><td>32.65</td><td></td><td>36.25</td><td>39.13 69.95 80.94</td><td>76.18</td><td>69.39 70.34</td><td></td><td>86.53</td><td></td><td>67.07 70.15</td></tr><tr><td rowspan="7">ViT-G/14</td><td></td><td>√</td><td></td><td></td><td>35.15</td><td></td><td></td><td></td><td></td><td></td><td></td><td>93.81</td><td></td></tr><tr><td>CIReVL [12]</td><td>√ √</td><td>26.77 31.12</td><td>27.59 32.24</td><td>29.96</td><td>31.03</td><td>34.65</td><td>64.29</td><td>75.06</td><td>67.95</td><td>84.87</td><td>93.21</td><td>66.12</td></tr><tr><td>LDRE [34]</td><td>√</td><td>30.47</td><td>31.14</td><td>34.95 35.03</td><td>36.03 36.59</td><td>36.15 37.26</td><td>66.39 67.25</td><td>77.25 77.33</td><td>68.82 69.22</td><td>85.66 85.28</td><td>93.76</td><td>67.60 68.24</td></tr><tr><td>OSrCIR [28]</td><td>√</td><td>32.46</td><td>33.77</td><td>36.46</td><td>37.55</td><td>38.87</td><td>69.42</td><td>79.42</td><td>74.15</td><td>89.23</td><td>93.55 95.71</td><td>71.79</td></tr><tr><td>SEIZE [33]</td><td>√</td><td>32.75</td><td>34.26</td><td>36.86</td><td>38.03</td><td>39.25</td><td>70.07</td><td>80.00</td><td>69.96</td><td>86.87</td><td>94.22</td><td>70.02</td></tr><tr><td>IP-CIR [17] CoTMR [27]</td><td>√</td><td>32.23</td><td>32.72</td><td>35.60</td><td>36.83</td><td>36.36</td><td>67.52</td><td>77.82</td><td>71.19</td><td>86.34</td><td>93.87</td><td></td></tr><tr><td>DCR-TSF (Ours)</td><td>√</td><td>37.47</td><td>38.63</td><td>41.26</td><td>42.45</td><td></td><td>41.40 71.40 81.59</td><td></td><td>70.27</td><td>87.30</td><td>94.36</td><td>69.36 70.84</td></tr></table>

Evaluation Metrics. For Fashion-IQ [30] and CIRR [21], we adopt Recall@K (R@K), which measures the fraction of queries where the ground-truth item appears among the top-K retrieved results. We also report ${ \mathrm { R } } _ { \mathrm { m e a n } } .$ , defined as the average of all R@K values, representing overall retrieval performance. For CIRCO [4], we use mean Average Precision at K (mAP@K). For CIRR [21], we report R@K, $\mathrm { R e c a l l _ { s u b s e t } @ K ( R _ { s u b } @ K ) }$ , and the average score (Avg.) following [21]. $\mathrm { R } _ { \mathrm { s u b } } @ \mathrm { K }$ evaluates retrieval accuracy within a curated subset of visually similar candidates. Following the previous works [21, 27], the Avg. metric is defined as the mean of R@5 and $\mathrm { R } _ { \mathrm { s u b } } @ 1$ for CIRR.

Implementation Details. For the MLLM backbone, we employ Qwen2.5-VL-32B [3]. For the image and text embeddings, following previous methods [27,28], we conduct experiments using OpenCLIP [9] models, including ViT-B/32, ViT-L/14, and ViT-G/14. Our experiments are conducted on four NVIDIA A6000 GPUs. For constructing the composed-query anchor $Q _ { \mathrm { c o m p } } ,$ we set the Slerp interpolation ratio λ to 0.8 for CIRCO and Fashion-IQ, and 0.9 for CIRR, following the setting of a previous method [10]. For TSF, the fusion weights are assigned to the fine-grained representation, coarse-grained representation, and composed-query anchor, respectively. Specifically, in the order of $( T _ { \mathrm { f i n e } } , T _ { \mathrm { c o a r s e } } , Q _ { \mathrm { c o m p } } )$ , we set the weights to (0.3, 0.4, 0.3) for CIRCO, (0.2, 0.3, 0.5) for CIRR, and (0.3, 0.1, 0.6) for Fashion-IQ. The number of Karcher mean iterations is fixed to 2 for all datasets.

Table 2: Comparison with the state-of-the-art methods on Fashion-IQ. $\mathrm { R } _ { \mathrm { m e a n } }$ represents the average results across all the metrics. Best results are highlighted in bold, and the second best are underlined.
<table><tr><td rowspan="2" colspan="2">Backbone Method</td><td rowspan="2">Training -free</td><td colspan="2">Shirt</td><td colspan="2">Dress</td><td colspan="2">Tops&amp;Tee</td><td colspan="3">Avg.</td></tr><tr><td>R@10 R@50</td><td></td><td></td><td>R@10 R@50</td><td>R@10 R@50</td><td></td><td></td><td></td><td>R@10 R@50 Rmean</td></tr><tr><td rowspan="9">ViT-B/32</td><td>PALAVRA [6]</td><td>x</td><td>21.49</td><td>37.05</td><td>17.25</td><td>35.94</td><td>20.55</td><td>38.76</td><td>19.76</td><td>37.25</td><td>28.50</td></tr><tr><td>SEARLE [4]</td><td>x</td><td>24.44</td><td>41.61</td><td>18.54</td><td>39.51</td><td>25.70</td><td>46.46</td><td>22.89</td><td>42.53</td><td>32.71</td></tr><tr><td>Slerp [10]</td><td>√</td><td>20.53</td><td>41.00</td><td>23.75</td><td>40.92</td><td>26.98</td><td>46.77</td><td>23.75</td><td>42.90</td><td>33.33</td></tr><tr><td>CIReVL [12]</td><td>√</td><td>28.36</td><td>47.84</td><td>25.29</td><td>46.36</td><td>31.21</td><td>53.85</td><td>28.29</td><td>49.35</td><td>38.82</td></tr><tr><td>LDRE [34]</td><td>√</td><td>27.38</td><td>46.27</td><td>19.97</td><td>41.84</td><td>27.07</td><td>48.78</td><td>24.81</td><td>45.63</td><td>35.22</td></tr><tr><td>SEIZE [33]</td><td>√</td><td>29.38</td><td>47.97</td><td>25.37</td><td>46.84</td><td>32.07</td><td>54.78</td><td>28.94</td><td>49.86</td><td>39.40</td></tr><tr><td>OSrCIR [28]</td><td>√</td><td>31.16</td><td>51.13</td><td>29.35</td><td>50.37</td><td>36.51</td><td>58.71</td><td>32.34</td><td>53.40</td><td>42.87</td></tr><tr><td>CoTMR [27]</td><td>√</td><td>33.42</td><td>53.93</td><td>31.09</td><td>54.54</td><td>38.40</td><td>61.14</td><td>34.30</td><td>56.54</td><td>45.42</td></tr><tr><td>DCR-TSF (Ours)</td><td>√</td><td>38.35</td><td>58.40</td><td>37.79</td><td>60.63</td><td>45.80</td><td>66.99</td><td>40.65</td><td>62.01</td><td>51.33</td></tr><tr><td rowspan="8">ViT-L/14</td><td>SEARLE [4]</td><td>x</td><td>26.89</td><td>45.58</td><td>20.48</td><td>43.13</td><td>29.32</td><td>49.97</td><td>25.56</td><td>46.23</td><td>35.89</td></tr><tr><td>Slerp [10]</td><td>√</td><td>21.96</td><td>41.55</td><td>28.66</td><td>43.96</td><td>30.24</td><td>48.34</td><td>26.95</td><td>44.62</td><td>35.79</td></tr><tr><td>CIReVL [12]</td><td>√</td><td>29.49</td><td>47.40</td><td>24.79</td><td>44.76</td><td>31.36</td><td>53.65</td><td>28.55</td><td>48.57</td><td>38.56</td></tr><tr><td>LDRE [34]</td><td>√</td><td>31.04</td><td>51.22</td><td>22.93</td><td>46.76</td><td>31.57</td><td>53.64</td><td>28.51</td><td>50.54</td><td>39.52</td></tr><tr><td>SEIZE [33]</td><td>√</td><td>33.04</td><td>53.22</td><td>30.93</td><td>50.76</td><td>35.57</td><td>58.64</td><td>33.18</td><td>54.21</td><td>43.69</td></tr><tr><td>OSrCIR [28]</td><td>√</td><td>33.17</td><td>52.03</td><td>29.70</td><td>51.81</td><td>36.92</td><td>59.27</td><td>33.26</td><td>54.37</td><td>43.82</td></tr><tr><td>CoTMR [27]</td><td>√</td><td>35.43</td><td>54.91</td><td>31.18</td><td>55.04</td><td>38.55</td><td>61.33</td><td>35.05</td><td>57.09</td><td>46.50</td></tr><tr><td>DCR-TSF (Ours)</td><td>√</td><td>40.10</td><td>58.14</td><td>37.87</td><td>59.65</td><td>46.15</td><td>67.19</td><td>41.37</td><td>61.66</td><td>51.52</td></tr><tr><td rowspan="6">ViT-G/14</td><td>CIReVL [12]</td><td>√</td><td>33.71</td><td>51.42</td><td>27.07</td><td>49.53</td><td>35.80</td><td>56.14</td><td></td><td>52.36</td><td>42.27</td></tr><tr><td>LDRE [34]</td><td>√</td><td>35.94</td><td>58.58</td><td>26.11</td><td>51.12</td><td>35.42</td><td>56.67</td><td>32.19</td><td></td><td></td></tr><tr><td>SEIZE [33]</td><td>√</td><td>43.60</td><td>65.42</td><td>39.61</td><td>61.02</td><td>45.94</td><td>71.12</td><td>32.49 43.05</td><td>55.46 65.85</td><td>43.97 54.45</td></tr><tr><td>OSrCIR [28]</td><td>√</td><td>38.65</td><td>54.71</td><td>33.02</td><td>54.78</td><td>41.04</td><td>61.83</td><td>37.57</td><td>57.11</td><td>47.34</td></tr><tr><td>CoTMR [27]</td><td>√</td><td>38.32</td><td>62.24</td><td>34.51</td><td>57.36</td><td>41.90</td><td>64.30</td><td>38.25</td><td>61.32</td><td>49.78</td></tr><tr><td>DCR-TSF (Ours)</td><td>√</td><td>45.17</td><td>62.75</td><td></td><td>42.00 63.56</td><td>49.86</td><td>70.91</td><td>45.68</td><td>65.74</td><td>55.71</td></tr></table>

## 4.2 Main Results

CIRCO. Table 1 shows the performance comparison on CIRCO, which measures retrieval performance using mAP. Unlike R@K, mAP reflects not only the accuracy of top-ranked results but also the overall ordering quality across all retrieved items, providing a more comprehensive evaluation of retrieval consistency. Across all backbones, our method achieves the highest mAP, showing a substantial improvement over the existing training-free and fine-tuned approaches. These consistent improvements indicate that combining dual-level target descriptions with the composed-query anchor helps capture target semantics more robustly and stabilizes the overall retrieval ranking.

CIRR. Table 1 also shows the comparison with state-of-the-art methods on CIRR. Our method achieves strong performance across most evaluation metrics, demonstrating its efectiveness on natural-image composed retrieval. While our method obtains slightly lower scores on several ViT-G/14 settings, it remains competitive with existing training-free methods. $\mathrm { R } _ { \mathrm { s u b } } @ \mathrm { K }$ focuses on a small curated subset based on cosine similarity using ResNet [8], which emphasizes appearance-based similarity. The overall performance indicates that combining dual-level target descriptions with the composed-query anchor efectively captures target semantics beyond appearance-level similarity, demonstrating its robustness in real-world retrieval settings.

Table 3: Ablation study with CLIP ViT-G/14. FD, CD, and Comp denote fine-grained descriptions, coarse-grained descriptions, and the composed-query anchor, respectively. The last row corresponds to the full DCR-TSF.
<table><tr><td rowspan="2">FD</td><td rowspan="2">CD</td><td rowspan="2">Comp</td><td colspan="2">CIRCO</td><td colspan="2">CIRR</td><td colspan="2">Fashion-IQ</td></tr><tr><td>mAP@5</td><td>mAP@10</td><td>R@5</td><td> $\mathbf { R _ { s u b } @ 1 }$ </td><td>R@10</td><td>R@50</td></tr><tr><td rowspan="3">√</td><td></td><td></td><td>25.55</td><td>26.66</td><td>60.00</td><td>61.54</td><td>35.57</td><td>56.03</td></tr><tr><td>√</td><td></td><td>32.03</td><td>33.34</td><td>64.29</td><td>69.06</td><td>32.83</td><td>52.84</td></tr><tr><td></td><td>√</td><td>17.32</td><td>17.88</td><td>62.75</td><td>67.42</td><td>40.58</td><td>60.39</td></tr><tr><td rowspan="3">√ √</td><td>√</td><td></td><td>34.77</td><td>35.95</td><td>67.04</td><td>68.15</td><td>38.43</td><td>59.35</td></tr><tr><td></td><td>√</td><td>32.01</td><td>32.88</td><td>69.33</td><td>67.66</td><td>43.86</td><td>65.23</td></tr><tr><td>√</td><td>√</td><td>34.31</td><td>34.99</td><td>69.74</td><td>70.58</td><td>41.49</td><td>63.22</td></tr><tr><td>√</td><td>√</td><td>√</td><td>37.47</td><td>38.63</td><td>71.40</td><td>70.27</td><td>45.68</td><td>65.74</td></tr></table>

Fashion-IQ. Table 2 reports the comparison with state-of-the-art methods on Fashion-IQ with CLIP. Our method achieves the best overall performance in terms of ${ \mathrm { R } } _ { \mathrm { m e a n } } ,$ demonstrating its efectiveness on fashion-domain composed retrieval. Although our method shows slightly lower performance than SEIZE [33] on a few R@50 metrics with the $\mathrm { V i T - G / 1 4 }$ backbone, it still achieves the highest overall ${ \mathrm { R } } _ { \mathrm { m e a n } } .$ . Moreover, our method consistently improves as the backbone capacity increases from ViT-B/32 to $\mathrm { V i T - G / 1 4 }$ , indicating its scalability and robustness across diferent CLIP backbones.

## 4.3 Further Analysis

Ablation study on DCR-TSF. Table 3 shows the ablation results of the three components in DCR-TSF across CIRCO, CIRR, and Fashion-IQ using CLIP $\mathrm { V i T - G / 1 4 }$ . In this experiment, FD, CD, and Comp indicate the fine-grained representation $T _ { \mathrm { f i n e } } ,$ the coarse-grained representation $T _ { \mathrm { c o a r s e } } .$ , and the composedquery anchor $Q _ { \mathrm { c o m p } } ,$ , respectively. The single-component results reveal that the most efective retrieval cue difers depending on the dataset characteristics. On open-domain benchmarks such as CIRCO and CIRR, CD generally provides strong performance, suggesting that coarse semantic abstraction is useful when queries involve diverse scenes and multiple plausible target images. In contrast, on Fashion-IQ, FD and Comp are more efective than CD, indicating that fashiondomain retrieval relies more heavily on fine-grained visual attributes and direct preservation of the reference-modification query. The pairwise combinations further demonstrate the complementarity among the three components. Combining FD and CD balances detailed visual evidence with coarse semantic abstraction, while combining Comp with FD or CD provides information that is not fully captured by MLLM-generated descriptions. Finally, using all three components achieves the best performance on most metrics, confirming that DCR-TSF benefits from jointly exploiting fine-grained visual evidence, coarse-grained semantic abstraction, and direct preservation of the original composed query.

![](images/622d2d6b0a5f5ec0e7432f2b50092f793a6302f5b5adf182605db065598f81ae.jpg)  
(a) CIRCO

![](images/ee0173532b5db76b0fd6d044cbb5dea4f2b450754224b53d3f3addc20902b8f2.jpg)  
(b) CIRR

![](images/aec31746ee34499f7e862cbba6da3189d786e94b8643ff73a4b3991129a84b39.jpg)  
(c) Fashion-IQ  
Fig. 4: Hyperparameter sensitivity of TSF fusion weights. Ternary plots show validation results for diferent TSF fusion-weight configurations on each dataset using CLIP ViT-G/14. The star marker indicates the best-performing configuration selected for the main experiments.

## 4.4 Hyperparameter Sensitivity

Fig. 4 shows the sensitivity analysis of TSF fusion weights using ternary plots on the validation splits of CIRCO, CIRR, and Fashion-IQ. We vary the fusion weights $\mathbf { w } = ( w _ { \mathrm { f i n e } } , w _ { \mathrm { c o a r s e } } , w _ { \mathrm { c o m p } } )$ under the constraint $w _ { \mathrm { f i n e } } + w _ { \mathrm { c o a r s e } } + w _ { \mathrm { c o m p } } = 1$ , and report the retrieval performance for each weight combination. Each point in the ternary plot corresponds to a specific weight configuration, and the star marker indicates the best-performing configuration used in our main experiments: $\mathbf { w } = ( 0 . 3 , 0 . 4 , 0 . 3 )$ for CIRCO, $\mathbf { w } = ( 0 . 2 , 0 . 3 , 0 . 5 )$ for CIRR, and $\mathbf { w } = ( 0 . 3 , 0 . 1 , 0 . 6 )$ for Fashion-IQ. Overall, TSF is not highly sensitive to a precise weight configuration. Although the best-performing weights difer across datasets, highperformance regions are broadly formed around balanced combinations of the three components, especially near the center of the ternary plots. This suggests that TSF provides stable performance as long as fine-grained descriptions, coarsegrained descriptions, and the composed-query anchor are jointly considered. On CIRCO, strong performance is obtained when the three components are well balanced, indicating that open-domain retrieval benefits from both semantic abstraction and original query preservation. On CIRR and Fashion-IQ, relatively larger weights for the composed-query anchor are preferred, reflecting the importance of preserving the original reference-modification information in natural-image retrieval with visually similar candidates and fashion-domain retrieval. These results support the role of TSF in balancing fine-grained visual evidence, coarse-grained semantic abstraction, and direct preservation of the original composed query.

Disjointness Analysis of Fine- and Coarse-Descriptions. Table 4 shows the Jaccard similarity and overlap coeficient, which measure the degree of intersection between the Top-K retrieval sets obtained from the fine- and coarse-grained descriptions on CIRCO. For each reference image, we retrieve the Top-K images using both the fine- and coarse- descriptions, compare the overlap between the two corresponding retrieval images within ground truth, and then average the results over all queries. The Top-K overlap between fine- and coarse-grained retrieval sets is low at small K, implying that the two description levels capture distinct information. As K increases, the overlap ratio gradually rises, suggesting that fine- and coarse-descriptions share a consistent high-level concept. This trend indicates that the two reasoning levels follow similar semantic information but emphasize diferent aspects of the reference image and modification text.

Table 4: Disjointness analysis on CIRCO with ViT-G/14. Jaccard and Overlap Coeficient are computed between Top-K retrieval sets from fine- and coarselevel representations.
<table><tr><td colspan="2">Top-K Jaccard Overlap Coefficient</td></tr><tr><td>5</td><td>0.3508 0.4237</td></tr><tr><td>10 0.4140</td><td>0.5050</td></tr><tr><td>25 0.5350</td><td>0.6550</td></tr><tr><td>50 0.6355</td><td>0.7733</td></tr></table>

![](images/5c92db73f64774901386f6adf6b90a3e06e390e3d853accd5d8098b5b6c47e32.jpg)  
Fig. 5: Comparison of caption ensemble and DCR on the CIRCO test set with ViT-G/14.

Comparative Analysis of Caption Ensemble and DCR. Existing methods [17, 33, 34] report that utilizing multiple generated captions can improve the performance, raising the possibility that the gains of our DCR may stem from caption ensemble. To investigate this hypothesis, we generate multiple descriptions with varying temperature using either $P _ { \mathrm { { f i n e } } }$ or $P _ { \mathrm { c o a r s e } }$ independently and compare their performance with DCR under the same number of descriptions. Fig. 5 shows that increasing the number of fine- or coarse-only descriptions yields only marginal improvements and quickly saturates. In contrast, DCR consistently achieves superior performance when using the same number of descriptions, indicating that the performance gain does not arise from simple caption ensemble. These results suggest that DCR benefits from complementary semantic granularity, where fine- and coarse-grained descriptions provide mutually informative cues for compositional understanding. Although performance generally increases as more descriptions are used, generating multiple captions requires repeated MLLM inference, introducing substantial computational overhead. Therefore, in all experiments, we employ one fine- and one coarse-grained description for DCR. Analysis on Scalability and Model Variation in MLLMs. Table 5 shows the retrieval performance of various MLLMs with diferent parameter scales on CIRCO. As shown in Table 5, larger MLLMs generally achieve higher retrieval performance, but the improvement becomes marginal beyond 32B parameters. Specifically, Qwen2.5-VL-32B achieves a balance between performance and eficiency, while further scaling to 72B provides only marginal gains. We also report results with the latest Qwen3-VL series and GPT-5, which show comparable performance to Qwen2.5-VL-32B. This suggests that our framework is not highly sensitive to a specific MLLM backbone among suficiently capable models. However, to ensure fair comparison and maintain consistency with previous studies, we adopt Qwen2.5-VL-32B for all experiments.

Computational Cost Analysis. We analyze the computational overhead of our method when processing a single composed query. The overall latency is mainly dominated by MLLM inference for generating fine- and coarse-grained descriptions, while the remaining retrieval operations introduce only marginal additional cost. In particular, TSF only performs Slerp-based anchor construction and weighted Karcher mean fusion in the CLIP embedding space, introducing negligible computational overhead compared to MLLM inference. Table 6 compares the inference time of our method with SEIZE [33] and CoTMR [27] under the same experimental setting. Under the same environment with an NVIDIA A6000 GPU, our method requires 3.467±0.030 seconds per query. This is slightly faster than CoTMR and substantially more eficient than SEIZE, which require 3.553±0.023 and 5.472±0.791 seconds per query, respectively. Specifically, generating the fine- and coarse-grained descriptions takes 1.883±0.019 and 1.584±0.018 seconds, respectively, showing that the computational cost mainly comes from MLLM-based description generation. Despite this overhead, the dual-level reasoning stage provides complementary semantic cues, and TSF efectively integrates them with the original composed-query information. These results indicate that our method achieves a favorable balance between retrieval performance and computational eficiency.

Table 5: Analysis on scalability and model variation in MLLMs on the CIRCO test set with CLIP ViT-G/14.
<table><tr><td>Method</td><td>mAP@5</td><td>mAP@10</td><td>mAP@25</td><td>mAP@50</td></tr><tr><td>Qwen2.5-VL-3B [3]</td><td>23.72</td><td>24.89</td><td>27.09</td><td>28.09</td></tr><tr><td>Qwen2.5-VL-7B [3]</td><td>30.62</td><td>31.86</td><td>34.29</td><td>35.42</td></tr><tr><td>Qwen2.5-VL-32B [3]</td><td>34.77</td><td>35.95</td><td>38.56</td><td>39.79</td></tr><tr><td>Qwen2.5-VL-72B [3]</td><td>35.26</td><td>36.69</td><td>39.46</td><td>40.62</td></tr><tr><td>Qwen2-VL-72B [32]</td><td>34.38</td><td>36.11</td><td>38.81</td><td>39.89</td></tr><tr><td>Qwen3-VL-4B [31]</td><td>33.21</td><td>34.48</td><td>37.23</td><td>38.37</td></tr><tr><td>Qwen3-VL-8B [31]</td><td>34.81</td><td>35.87</td><td>38.50</td><td>39.60</td></tr><tr><td>Qwen3-VL-32B [31]</td><td>34.49</td><td>35.72</td><td>38.35</td><td>39.48</td></tr><tr><td>GPT-5 [22]</td><td>35.18</td><td>36.68</td><td>39.49</td><td>40.64</td></tr></table>

Table 6: Inference time comparison with existing training-free ZS-CIR methods. We report the average latency per composed query in seconds. Each column decomposes the total inference time into the main reasoning stages of each method.
<table><tr><td></td><td></td><td>BLIP-2 &amp; GPT-3.5-turbo Multi-caption generator LLM-based editing reasoner</td><td>Total</td></tr><tr><td>SEIZE [33]</td><td>1.560 ± 0.056</td><td>3.912 ± 0.812</td><td>5.472 ± 0.791</td></tr><tr><td>Qwen2.5-VL-32B</td><td>Image-scale reasoning</td><td>Object-scale reasoning</td><td>Total</td></tr><tr><td>CoTMR [27]</td><td>1.892 ± 0.011</td><td>1.661 ± 0.014</td><td>3.553 ± 0.023</td></tr><tr><td>Qwen2.5-VL-32B</td><td>Fine-grained description Coarse-grained description</td><td></td><td>Total</td></tr><tr><td>DCR-TSF (Ours)</td><td>1.883 ± 0.019</td><td>1.584 ± 0.018</td><td>3.467 ± 0.030</td></tr></table>

## 4.5 Qualitative Results

Fig. 6 presents qualitative retrieval examples on CIRCO. The ground-truth images highlighted in green often appear at suboptimal ranks when relying solely on either fine- or coarse-grained descriptions. DCR improves the ranking by combining fine-grained visual details with coarse-grained semantic cues. By further incorporating the composed-query anchor, DCR-TSF produces more balanced retrieval results and raises the ranks of the ground-truth images.

![](images/7b012c92339c1396651c62a4c34739a5c1c508ce79e440d38480eaf1116010b2.jpg)  
Fig. 6: Qualitative comparison on CIRCO with ViT-G/14. Each example shows a reference image, its modification text, and the retrieved images with their ranks under four settings: Fine-Grained Description, Coarse-Grained Description, DCR, and DCR-TSF. The ground-truth image is highlighted in green.

## 5 Conclusion

In this paper, we propose DCR-TSF, a training-free framework for ZS-CIR that combines Dual-level Compositional Reasoning (DCR) and Triadic Spherical Fusion (TSF). Our method generates fine- and coarse-grained descriptions to capture complementary semantics, balancing detailed visual evidence with high-level semantic abstraction. To preserve the original image-text query information, TSF constructs a composed-query anchor from the reference image and modification text and fuses it with the fine- and coarse-grained representations on the unit hypersphere. By exploiting fine-grained visual evidence, coarse-grained semantic abstraction, and direct preservation of the original composed query, DCR-TSF mitigates the limitations of relying on fine-grained generated descriptions alone. Experiments on CIRCO, CIRR, and Fashion-IQ demonstrate that DCR-TSF achieves state-of-the-art performance among training-free ZS-CIR methods.

Acknowledgments. This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2019-II190079, Artificial Intelligence Graduate School Program, Korea University), and by the Information Technology Research Center (ITRC) support program (No. IITP-2026-RS-2024-00436857).

## References

1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)

2. An, N.M., Kim, E., Thorne, J., Shim, H.: I0t: Embedding standardization method towards zero modality gap. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 27182– 27199 (2025)

3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)

4. Baldrati, A., Agnolucci, L., Bertini, M., Del Bimbo, A.: Zero-shot composed image retrieval with textual inversion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15338–15347 (2023)

5. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. In: Advances in Neural Information Processing Systems. vol. 33, pp. 1877–1901 (2020)

6. Cohen, N., Gal, R., Meirom, E.A., Chechik, G., Atzmon, Y.: “this is my unicorn, flufy”: Personalizing frozen vision-language representations. In: Proceedings of the European Conference on Computer Vision. pp. 558–577 (2022)

7. Delmas, G., de Rezende, R.S., Csurka, G., Larlus, D.: Artemis: Attentionbased retrieval with text-explicit matching and implicit similarity. arXiv preprint arXiv:2203.08101 (2022)

8. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 770–778 (2016)

9. Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., Schmidt, L.: Openclip (2021)

10. Jang, Y.K., Huynh, D., Shah, A., Chen, W.K., Lim, S.N.: Spherical linear interpolation and text-anchoring for zero-shot composed image retrieval. In: Proceedings of the European Conference on Computer Vision. pp. 239–254 (2024)

11. Karcher, H.: Riemannian center of mass and mollifier smoothing. Communications on pure and applied mathematics 30(5), 509–541 (1977)

12. Karthik, S., Roth, K., Mancini, M., Akata, Z.: Vision-by-language for training-free compositional image retrieval. arXiv preprint arXiv:2310.09291 (2023)

13. Lee, S.H., Choi, H.Y., Kim, S.B., Lee, S.W.: Hierspeech++: Bridging the gap between semantic and acoustic representation of speech by hierarchical variational inference for zero-shot speech synthesis. IEEE Transactions on Neural Networks and Learning Systems 36(10), 18422–18436 (2025)

14. Lee, S.H., Kim, S.B., Lee, J.H., Song, E., Hwang, M.J., Lee, S.W.: Hierspeech: Bridging the gap between text and speech by hierarchical variational inference using self-supervised representations for speech synthesis 35, 16624–16636 (2022)

15. Lee, Y.B., Park, U., Jain, A.K., Lee, S.W.: Pill-id: Matching and retrieval of drug pill images. Pattern Recognition Letters 33(7), 904–910 (2012)

16. Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International Conference on Machine Learning. pp. 19730–19742 (2023)

17. Li, Y., Ma, F., Yang, Y.: Imagine and seek: Improving composed image retrieval with an imagined proxy. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3984–3993 (2025)

18. Liang, V.W., Zhang, Y., Kwon, Y., Yeung, S., Zou, J.Y.: Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. In: Advances in Neural Information Processing Systems. vol. 35, pp. 17612–17625 (2022)

19. Lim, Y.K., Choi, S.H., Lee, S.W.: Text extraction in mpeg compressed video for content-based indexing. In: Proceedings International Conference on Pattern Recognition. vol. 4, pp. 409–412 (2000)

20. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: Proceedings of the European Conference on Computer Vision. pp. 740–755 (2014)

21. Liu, Z., Rodriguez-Opazo, C., Teney, D., Gould, S.: Image retrieval on real-life images with pre-trained vision-and-language models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2125–2134 (2021)

22. OpenAI: Introducing gpt-5. https://openai.com/index/introducing- gpt- 5/ (2025)

23. Prabhakar, S.K., Lee, S.W.: Holistic approaches to music genre classification using eficient transfer and deep learning techniques. Expert Systems with Applications 211, 118636 (2023)

24. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748–8763 (2021)

25. Saito, K., Sohn, K., Zhang, X., Li, C.L., Lee, C.Y., Saenko, K., Pfister, T.: Pic2Word: Mapping pictures to words for zero-shot composed image retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19305–19314 (2023)

26. Shoemake, K.: Animating rotation with quaternion curves. In: Proceedings of the ACM SIGGRAPH Computer Graphics. vol. 19, pp. 245–254 (1985)

27. Sun, Z., Jing, D., Lu, Z.: CoTMR: Chain-of-thought multi-scale reasoning for training-free zero-shot composed image retrieval. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2025)

28. Tang, Y., Zhang, J., Qin, X., Yu, J., Gou, G., Xiong, G., Lin, Q., Rajmohan, S., Zhang, D., Wu, Q.: Reason-before-retrieve: One-stage reflective chain-of-thoughts for training-free zero-shot composed image retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14400–14410 (2025)

29. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J.: InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265 (2025)

30. Wu, H., Gao, Y., Guo, X., Al-Halah, Z., Rennie, S., Grauman, K., Feris, R.: Fashion IQ: A new dataset towards retrieving images by natural language feedback. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11307–11317 (2021)

31. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)

32. Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin, H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J., Yang, J., Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K.,

Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P., Wang, P., Peng, R., Men, R., Gao, R., Lin, R., Wang, S., Bai, S., Tan, S., Zhu, T., Li, T., Liu, T., Ge, W., Deng, X., Zhou, X., Ren, X., Zhang, X., Wei, X., Ren, X., Liu, X., Fan, Y., Yao, Y., Zhang, Y., Wan, Y., Chu, Y., Liu, Y., Cui, Z., Zhang, Z., Guo, Z., Fan, Z.: Qwen2 technical report. arXiv preprint arXiv:2407.10671 (2024)

33. Yang, Z., Qian, S., Xue, D., Wu, J., Yang, F., Dong, W., Xu, C.: Semantic editing increment benefits zero-shot composed image retrieval. In: Proceedings of the ACM International Conference on Multimedia. pp. 1245–1254 (2024)

34. Yang, Z., Xue, D., Qian, S., Dong, W., Xu, C.: LDRE: Llm-based divergent reasoning and ensemble for zero-shot composed image retrieval. In: Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 80–90 (2024)

# DCR-TSF: Dual-Level Compositional Reasoning with Triadic Spherical Fusion for Training-Free Zero-Shot Composed Image Retrieval Supplementary

Gwang-Ho $\mathrm { N a } ^ { 1 } \oplus$ , Ho-Joong Kim<sup>1</sup> , and Seong-Whan Lee<sup>1</sup>

Dept. of Artificial Intelligence, Korea University, Seoul, Korea {g\_h\_na, hojoong\_kim, sw.lee}@korea.ac.kr

## A Additional Details

## A.1 Prompt for Fine-Coarse Description

Fine Description Prompt $\mathbf { P _ { f i n e } }$ Based on the existing methods [27, 28], we design a fine-grained prompt that explicitly enumerates object attributes, spatial relations, and all visually grounded semantics in the reference image. This prompt guides the model to focus on visual details of the scene. Fig. A illustrates the full template.

Coarse Description Prompt $\mathbf { P _ { c o a r s e } }$ While $\mathbf { P _ { f i n e } }$ focuses on detailed attributes, the coarse prompt provides a short summary that captures only the shared, global semantics of the reference image. Fig. B describes the coarse template.

## A.2 Preliminary on Triadic Spherical Fusion

In this section, we provide additional details on the implementation of Triadic Spherical Fusion (TSF). TSF fuses three unit-normalized representations: the fine-grained representation $T _ { \mathrm { f i n e } } .$ , the coarse-grained representation $T _ { \mathrm { c o a r s e } } .$ and the composed-query anchor $Q _ { \mathrm { { c o m p } } }$ . We denote this set of representations as

$$
\mathcal { Z } = \{ T _ { \mathrm { f i n e } } , T _ { \mathrm { c o a r s e } } , Q _ { \mathrm { c o m p } } \} .\tag{A}
$$

Each representation lies on the unit hypersphere $\mathbb { S } ^ { d - 1 }$ , where d denotes the embedding dimension. The fusion weight assigned to each $Z \in { \mathcal { Z } }$ is denoted as w<sub>Z</sub>, with $\stackrel { \smile } { \sum } _ { Z \in \mathcal { Z } } w _ { Z } = 1$

Weighted Karcher Mean. The Karcher mean, also known as the Riemannian center of mass, generalizes the Euclidean mean to Riemannian manifolds by minimizing the sum of squared geodesic distances [11]. On the unit hypersphere, we define the weighted Karcher mean of $\mathcal { Z }$ as:

$$
T _ { \mathrm { f i n a l } } = \arg \operatorname* { m i n } _ { Y \in \mathbb { S } ^ { d - 1 } } \sum _ { Z \in { \mathcal { Z } } } w _ { Z } d _ { \mathbb { S } } ^ { 2 } ( Y , Z ) ,\tag{B}
$$

where Y denotes an optimization variable on the unit hypersphere and $d _ { \mathrm { S } } ( \cdot , \cdot )$ denotes the geodesic distance on the unit hypersphere. For two unit vectors a and $b ,$ this distance is given by:

$$
d _ { \mathbb { S } } ( a , b ) = \cos ^ { - 1 } ( a ^ { \top } b ) .\tag{C}
$$

This formulation difers from Euclidean averaging because it computes the mean with respect to the intrinsic geometry of the unit hypersphere. Riemannian geometry and tangent-space mappings have also been employed for transfer learning and classification in other domains [23].

Numerical Solver. The weighted Karcher mean generally does not have a simple closed-form solution for more than two points on the sphere. Therefore, we compute it using a standard iterative procedure based on the logarithmic and exponential maps. Given the current estimate $Y ^ { ( r ) }$ , each representation $Z \in { \mathcal { Z } }$ is first mapped to the tangent space at $Y ^ { ( r ) }$

$$
\xi _ { Z } ^ { ( r ) } = \log _ { Y ^ { ( r ) } } ( Z ) .\tag{D}
$$

The tangent vectors are then averaged using the fusion weights:

$$
\varDelta ^ { ( r ) } = \sum _ { Z \in \mathcal { Z } } w _ { Z } \xi _ { Z } ^ { ( r ) } .\tag{E}
$$

The estimate is updated by mapping the averaged tangent vector back to the unit hypersphere:

$$
Y ^ { \left( r + 1 \right) } = \exp _ { Y ^ { \left( r \right) } } ( \varDelta ^ { \left( r \right) } ) .\tag{F}
$$

After a fixed number of iterations, the final estimate is used as $T _ { \mathrm { f i n a l } }$

Sphere-specific Maps. For completeness, we describe the logarithmic and exponential maps used on the unit hypersphere. Given two unit vectors x and $y ,$ the logarithmic map at x is:

$$
\log _ { x } ( y ) = { \frac { \theta } { \sin \theta } } \left( y - \cos \theta x \right) , \quad \theta = \cos ^ { - 1 } ( x ^ { \top } y ) .\tag{G}
$$

Given a tangent vector $\xi$ at $x _ { \mathrm { { i } } }$ , the exponential map is:

$$
\exp _ { x } ( \xi ) = \cos ( \| \xi \| ) x + \sin ( \| \xi \| ) \frac { \xi } { \| \xi \| } .\tag{H}
$$

These maps allow TSF to perform weighted averaging intrinsically on the unit hypersphere rather than in the ambient Euclidean embedding space.

Initialization and Stability. We initialize the iterative solver with the normalized weighted Euclidean average:

$$
Y ^ { ( 0 ) } = \frac { \sum _ { Z \in \mathcal { Z } } w _ { Z } Z } { \left\| \sum _ { Z \in \mathcal { Z } } w _ { Z } Z \right\| } .\tag{I}
$$

For numerical stability, inner products used in $\cos ^ { - 1 } ( \cdot )$ are clipped to a valid range. When $\theta$ or ∥ξ∥ is close to zero, we use the corresponding first-order approximation. In practice, a small fixed number of iterations is suficient for stable fusion.

![](images/42da80ce54bac3cf5fe209764f67fe19bff564766ccfc952ac3489bd3bb2b37a.jpg)  
Fig. A: Prompt of fine descriptions P<sub>fine</sub>

![](images/8347ea3a9c40ced984ef7a7a716837464b9aff2c79c9bf57bfcba0d9a1aa9cf5.jpg)  
Fig. B: Prompt of coarse descriptions P<sub>coarse</sub>

Table A: Extended ablation study across CLIP backbones. FD, CD, and Comp denote the fine-grained representation, coarse-grained representation, and composed query anchor, respectively. We report results on CIRCO, CIRR, and Fashion-IQ using ViT-B/32, ViT-L/14, and ViT-G/14.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">FD</td><td rowspan="2">CD</td><td rowspan="2">Comp</td><td colspan="2">CIRCO</td><td colspan="2">CIRR</td><td colspan="2">Fashion-IQ</td></tr><tr><td>mAP@5</td><td>mAP@10</td><td>R@5</td><td> $\mathbf { R _ { s u b } @ 1 }$ </td><td>R@10</td><td>R@50</td></tr><tr><td rowspan="8">ViT-B/32</td><td>√</td><td></td><td></td><td>18.21</td><td>18.83</td><td>55.88</td><td>58.96</td><td>32.08</td><td>52.84</td></tr><tr><td></td><td>√</td><td></td><td>21.28</td><td>21.89</td><td>59.98</td><td>65.88</td><td>29.91</td><td>50.87</td></tr><tr><td></td><td></td><td>√</td><td>11.00</td><td>11.41</td><td>59.21</td><td>63.90</td><td>33.97</td><td>54.49</td></tr><tr><td>√</td><td>√</td><td></td><td>22.95</td><td>23.76</td><td>62.29</td><td>65.16</td><td>35.57</td><td>57.02</td></tr><tr><td>√</td><td></td><td>√</td><td>22.15</td><td>22.86</td><td>65.74</td><td>64.89</td><td>39.50</td><td>60.82</td></tr><tr><td></td><td>√</td><td>√</td><td>23.53</td><td>24.05</td><td>64.46</td><td>68.07</td><td>37.75</td><td>59.03</td></tr><tr><td>√</td><td>√</td><td>√</td><td>25.07</td><td>25.81</td><td>67.28</td><td>67.66</td><td>40.65</td><td>62.01</td></tr><tr><td rowspan="7">ViT-L/14</td><td>√</td><td></td><td></td><td>21.95</td><td>22.59</td><td>58.17</td><td>60.19</td><td>32.38</td><td>52.42</td></tr><tr><td></td><td>√</td><td></td><td>27.28</td><td>28.11</td><td>62.78</td><td>68.07</td><td>29.10</td><td>49.83</td></tr><tr><td></td><td></td><td>√</td><td>13.62</td><td>14.15</td><td>62.19</td><td>66.55</td><td>36.66</td><td>56.17</td></tr><tr><td>√</td><td>√</td><td></td><td>30.05</td><td>30.88</td><td>64.87</td><td>66.72</td><td>34.40</td><td>55.70</td></tr><tr><td>√</td><td></td><td>√</td><td>27.59</td><td>28.47</td><td>67.59</td><td>67.35</td><td>39.90</td><td>60.79</td></tr><tr><td></td><td>√</td><td>√</td><td>29.46</td><td>30.13</td><td>67.66</td><td>70.75</td><td>37.41</td><td>58.28</td></tr><tr><td>√</td><td>√</td><td>√</td><td>31.80</td><td>32.65</td><td>69.95</td><td>70.34</td><td>41.37</td><td>61.66</td></tr><tr><td rowspan="7">ViT-G/14</td><td>√</td><td></td><td></td><td>25.55</td><td>26.66</td><td>60.00</td><td>61.54</td><td>35.57</td><td>56.03</td></tr><tr><td></td><td>√</td><td></td><td>32.03</td><td>33.34</td><td>64.29</td><td>69.06</td><td>32.83</td><td>52.84</td></tr><tr><td></td><td></td><td>√</td><td>17.32</td><td>17.88</td><td>62.75</td><td>67.42</td><td>40.58</td><td>60.39</td></tr><tr><td>√</td><td>√</td><td></td><td>34.77</td><td>35.95</td><td>67.04</td><td>68.15</td><td>38.43</td><td>59.35</td></tr><tr><td>√</td><td></td><td>√</td><td>32.01</td><td>32.88</td><td>69.33</td><td>67.66</td><td>43.86</td><td>65.23</td></tr><tr><td></td><td>√</td><td>√</td><td>34.31</td><td>34.99</td><td>69.74</td><td>70.58</td><td>41.49</td><td>63.22</td></tr><tr><td>√</td><td>√</td><td>√</td><td>37.47</td><td>38.63</td><td>71.40</td><td>70.27</td><td>45.68</td><td>65.74</td></tr></table>

## B Additional Experiments

## B.1 Ablation Studies

Table A provides extended ablation results across diferent CLIP backbones. The results show a consistent trend across ViT-B/32, ViT-L/14, and ViT-G/14. Using all three components generally achieves the best performance across CIRCO, CIRR, and Fashion-IQ, confirming the efectiveness of jointly exploiting finegrained descriptions, coarse-grained descriptions, and the composed-query anchor. The single-component results indicate that each component captures diferent retrieval cues. CD is particularly efective on CIRCO and CIRR, suggesting that coarse semantic abstraction is useful for open-domain composed retrieval. In contrast, Comp shows strong performance on Fashion-IQ, indicating that preserving the original reference-modification information is important for finegrained fashion retrieval. Pairwise combinations further improve performance over most single-component settings, demonstrating the complementarity among the three components. Overall, the extended ablation results confirm that the performance gains of DCR-TSF are stable across backbone scales.

Table B: Comparison of fusion strategies across CLIP backbones. Results are reported on CIRCO, CIRR, and Fashion-IQ with ViT-B/32 and ViT-L/14.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Fusion Strategy</td><td>CIRCO</td><td>CIRR</td><td>Fashion-IQ</td></tr><tr><td>mAP@5</td><td>R@5</td><td> $\mathrm { R } _ { \mathrm { m e a n } }$ </td></tr><tr><td rowspan="3">ViT-B/32</td><td>Weighted Avg.</td><td>25.11</td><td>67.18</td><td>51.32</td></tr><tr><td>Sequential Slerp</td><td>25.05</td><td>67.22</td><td>51.31</td></tr><tr><td>Karcher Mean</td><td>25.07</td><td>67.28</td><td>51.33</td></tr><tr><td rowspan="3"> $\mathrm { V i T - L } / 1 4$ </td><td>Weighted Avg.</td><td>31.92</td><td>69.95</td><td>51.46</td></tr><tr><td>Sequential Slerp</td><td>31.56</td><td>69.88</td><td>51.41</td></tr><tr><td>Karcher Mean</td><td>31.80</td><td>69.95</td><td>51.52</td></tr></table>

Table C: Comparison of reasoning backbones on the CIRCO test set with CLIP ViT-G/14.
<table><tr><td>Method</td><td>Reasoning Backbone</td><td>mAP@5</td><td>mAP@10</td></tr><tr><td>CIReVL [12]</td><td>BLIP-2 [16] + GPT-3.5-Turbo [5]</td><td>26.77</td><td>27.59</td></tr><tr><td>OSrCIR [28]</td><td>GPT-4o [1]</td><td>30.47</td><td>31.14</td></tr><tr><td>SEIZE [33]</td><td>BLIP-2 [16] + GPT-3.5-Turbo [5]</td><td>32.46</td><td>33.77</td></tr><tr><td>CoTMR [27]</td><td>Qwen2-VL-72B [32]</td><td>32.23</td><td>32.72</td></tr><tr><td>DCR (Ours)</td><td>Qwen2-VL-72B [32]</td><td>34.38</td><td>36.11</td></tr><tr><td>DCR (Ours)</td><td>Qwen2.5-VL-32B [3]</td><td>34.77</td><td>35.95</td></tr></table>

## B.2 Analysis of Fusion Strategies

To examine whether the efectiveness of TSF depends on the choice of fusion operator, we compare normalized weighted averaging, sequential Slerp, and the weighted Karcher mean. For a controlled comparison, all methods fuse the same three representations, $T _ { \mathrm { f i n e } } , T _ { \mathrm { c o a r s e } } .$ , and $Q _ { \mathrm { { c o m p } } }$ , using identical fusion weights. As shown in Table B, all three fusion strategies achieve nearly identical performance across CIRCO, CIRR, and Fashion-IQ. This indicates that the performance gains of TSF primarily arise from integrating the complementary information provided by the three representations, rather than from a particular fusion operator. We adopt the weighted Karcher mean as a principled intrinsic formulation for multi-way fusion on the unit hypersphere.

## B.3 Efect of Reasoning Backbone

To examine whether the performance improvement stems from the use of a stronger reasoning backbone, Table C summarizes the reasoning backbones used by existing training-free ZS-CIR methods. In particular, when using the same Qwen2-VL-72B backbone as CoTMR, DCR alone, without TSF, consistently outperforms CoTMR. These results demonstrate that the performance improvement is not attributable to the use of a stronger reasoning backbone.

Table D: Sensitivity analysis of fusion weights. Comparison of equal and datasetspecific fusion weights on CIRCO, CIRR, and Fashion-IQ using CLIP ViT-B/32 and ViT-L/14.
<table><tr><td rowspan="2"></td><td rowspan="2">Backbone Fusion Weights</td><td>CIRCO</td><td>CIRR</td><td>Fashion-IQ</td></tr><tr><td> $\mathrm { m A P @ 5 }$ </td><td>R@5</td><td> $\mathrm { R } _ { \mathrm { m e a n } }$ </td></tr><tr><td rowspan="2">ViT-B/32</td><td>Equal Weights</td><td>25.02</td><td>66.77</td><td>50.81</td></tr><tr><td>Dataset-specific</td><td>25.07</td><td>67.28</td><td>51.33</td></tr><tr><td rowspan="2"> $\mathrm { V i T - L } / 1 4$ </td><td>Equal Weights</td><td>31.60</td><td>69.25</td><td>50.46</td></tr><tr><td>Dataset-specific</td><td>31.80</td><td>69.95</td><td>51.52</td></tr></table>

Table E: Analysis of modality gap reduction. Comparison of retrieval performance and modality gap with and without $\mathrm { I 0 T _ { p o s t } }$ on the validation splits of three benchmarks. Values are reported as DCR-TSF $/ \mathrm { \Pi } \mathrm { I 0 } \mathrm { T _ { p o s t } } .$
<table><tr><td></td><td>CIRCO (mAP@5)</td><td>CIRR (R@5)</td><td>Fashion-IQ  $\mathbf { ( R _ { m e a n } ) }$ </td></tr><tr><td>Performance ↑</td><td>35.35 34.63</td><td>72.61 70.80</td><td>55.71 58.30</td></tr><tr><td>Modality Gap ↓</td><td>0.5524 0.0084</td><td>0.5366 0.0133</td><td>0.5990 0.0195</td></tr></table>

## B.4 Hyperparameter Sensitivity Analysis

Table D evaluates the sensitivity of DCR-TSF to the fusion weights by assigning equal weights of $1 / 3$ to all three representations. DCR-TSF maintains strong performance across CIRCO, CIRR, and Fashion-IQ even without dataset-specific weight configurations. These results show that the performance of DCR-TSF does not rely on carefully tuned fusion weights, demonstrating its robustness to the choice of fusion weights.

## B.5 Analysis of Modality Gap

Recent studies have identified a modality gap in multimodal embedding models such as CLIP, where visual and textual representations can remain separated despite being mapped into a shared embedding space [13, 14, 18]. We further analyze how reducing this modality gap afects the retrieval performance of DCR-TSF. To this end, we employ $\mathrm { I 0 T _ { p o s t } }$ [2], a training-free method that reduces the modality gap through post-hoc alignment. As a simple diagnostic, we measure the modality gap as the $\ell _ { 2 }$ distance between the mean visual and textual embeddings. As shown in Table E, $\mathrm { I 0 T _ { p o s t } }$ substantially reduces the modality gap, while retrieval performance does not consistently improve across datasets. These results indicate that reducing the modality gap alone does not necessarily lead to improved composed image retrieval performance.

## B.6 Comparative Analysis of Caption Ensemble and DCR

We further investigate whether the performance gains of DCR can be explained by simply ensembling multiple captions. To this end, we compare DCR with caption ensemble baselines constructed from descriptions at the same granularity. As shown in Fig. C, the relative efectiveness of coarse- and fine-grained caption ensembles varies across datasets, indicating that the preferred level of description granularity can depend on the characteristics of each benchmark. Despite this variation, DCR consistently outperforms the corresponding caption ensemble baselines on both CIRCO and Fashion-IQ. These results suggest that the benefit of DCR does not arise merely from aggregating multiple captions, but from efectively leveraging complementary information across descriptions with diferent levels of granularity.

![](images/fc66f526cb3efea85caa008a6033da8ee6b800e75dea437088693020c1ab011d.jpg)  
(a) CIRCO

![](images/2ddc8a9d77dc8c274250070377f680cb428f85c0ddec32954265bc0b3f8b2d9b.jpg)  
(b) Fashion-IQ  
Fig. C: Comparison of DCR with caption ensemble baselines on CIRCO and Fashion-IQ.

## C Additional Qualitative Results

## C.1 Retrieval Results

To further illustrate the behavior of our method beyond quantitative metrics, we present qualitative retrieval results on Fashion-IQ and CIRR. Each example compares four retrieval settings: Fine-Grained Description, Coarse-Grained Description, Composed-query Anchor, and DCR-TSF. The ground-truth image is highlighted in green.

Fashion-IQ. Fig. D shows representative examples from the Shirt, Dress, and Tops&Tee categories. Fashion-IQ requires fine-grained attribute understanding, since the modification text often describes detailed changes in color, shape, texture, sleeve length, or garment style. Fine-grained descriptions and the composedquery anchor help preserve such detailed reference-modification cues, while coarse-grained descriptions provide complementary garment-level semantics. By combining these cues, DCR-TSF more efectively captures both detailed visual attributes and the intended modification, ranking the ground-truth item higher. CIRR. Fig. E shows qualitative examples from CIRR. CIRR contains diverse natural images, where the modification text often requires changes in object category, action, or scene context. Fine- and coarse-grained descriptions capture complementary target semantics, while the composed-query anchor preserves the original reference-modification condition. By combining these cues, DCR-TSF improves the rank of the ground-truth image.

![](images/9e67b21a4b78127f65fc275d5edecf6bd56bc203ad432cd2c667ee8ba16ed1b5.jpg)  
(c) Tops&Tee  
Fig. D: Qualitative comparison on Fashion-IQ with ViT-G/14. Each example shows a reference image, its modification text, and the retrieved images with their ranks under four settings: Fine-Grained Description, Coarse-Grained Description, Composedquery Anchor, and DCR-TSF. The ground-truth image is highlighted in green.

## C.2 Failure Case Analysis

Despite the strong performance, our method still inherits limitations from both the MLLM and CLIP used in training-free ZS-CIR. Figs. F and G illustrate representative failure cases.

![](images/969888168f72706bd21ea5b169522537e1fb7e25ab77e6e5e6951d58d9164115.jpg)  
Fig. E: Qualitative comparison on CIRR with ViT-G/14. Each example shows a reference image, its modification text, and the retrieved images with their ranks under four settings: Fine-Grained Description, Coarse-Grained Description, Composed-query Anchor, and DCR-TSF. The ground-truth image is highlighted in green.

![](images/700c3df37c22d9703da3f71e9197cb746ef3a1145833802b4a98a1bf9cce5d0a.jpg)

![](images/7f592e0afeb25929a061eefd395df57ee595987fa323e9535757206bb613e61b.jpg)  
Fig. F: Failure cases caused by MLLM generation errors on CIRCO with ViT-G/14. These examples illustrate failure cases caused by hallucinated or inaccurate MLLM-generated descriptions. When incorrect semantics are introduced into the generated target description, they can propagate to the retrieval stage and degrade the target ranking. After manually correcting the erroneous attributes, the ranking improves substantially, highlighting the importance of description quality.

MLLM generation errors. Similar to existing MLLM-based ZS-CIR methods, our method relies on generated target descriptions. Therefore, when the MLLM produces incorrect or hallucinated fine- or coarse-grained descriptions, these errors can be directly propagated to the retrieval stage. As shown in Fig. F, correcting the generated descriptions leads to clear rank improvements, indicating that description quality remains an important bottleneck.

Failure on negation. Fig. G shows failure cases arising from CLIP’s dificulty in handling negation in the original modification text. For example, expressions such as “no books” or “no human interacting” may not be suficiently reflected in the

![](images/c802b0a064b06be0779632502cdd178e51f4179ab1a84da1a53bae7a7ade80fb.jpg)  
Fig. G: Failure cases caused by negation on CIRR with ViT-G/14. CLIP assigns high similarity to images containing objects or interactions explicitly excluded by negated modification texts (e.g., “no books” or “no human interacting”). Since TSF directly incorporates the modification text, this limitation propagates to retrieval and can rank such images above the ground-truth target.

CLIP similarity score. Since TSF directly incorporates the original modification text, this limitation can propagate to the retrieval stage, causing images that violate the negated condition to be ranked above the ground-truth target.