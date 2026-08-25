# Direct, Parallel, or Sequential? A Comparative Study of Training-Free Multi-Subject Image-to-Video Generation

Yanliang Qi University of Memphis Memphis, TN, USA

Muchao Ye University of Iowa Iowa City, IA, USA

Kexi Chen<sup>∗</sup> University of Pennsylvania Philadelphia, PA, USA

Haomiao Ni<sup>†</sup> University of Memphis Memphis, TN, USA

## Abstract

Text-conditioned image-to-video (I2V) generation has advanced rapidly, yet generating videos with multiple subjects remains challenging. A model must simultaneously preserve the appearance of each subject, assign distinct motions, and maintain coherent spatial and temporal interactions. This paper presents a systematic study of three representative paradigms for training-free multi-subject I2V generation: direct, parallel, and sequential generation. Direct generation applies a pretrained I2V model to the complete reference image and prompt, requiring all subjects and motions to be synthesized jointly. Parallel and sequential generation instead decompose the reference image and prompt into subject-specific visual and textual conditions. Parallel generation synthesizes each subject independently and subsequently composes the resulting videos, reducing the complexity of each generation step at the cost of weaker inter-subject context. Sequential generation first synthesizes a background video and then progressively introduces individual subjects. This preserves accumulated scene context but introduces sensitivity to subject ordering and error propagation. We empirically evaluate the three paradigms across diverse multi-subject scenes, comparing appearance preservation, motion fidelity, temporal consistency, and inter-subject coherence, while also characterizing their distinct failure modes. Our findings reveal the strengths and limitations of each paradigm and ofer practical insights for designing controllable multi-subject video generation systems.

## CCS Concepts

• Computing methodologies → Computer vision; Motion processing; Neural networks.

## Keywords

Image-to-video generation; Multi-subject generation; Training-free generation; Video difusion models.

ACM Reference Format:   
Yanliang Qi, Kexi Chen, Muchao Ye, and Haomiao Ni. 2026. Direct, Parallel, or Sequential? A Comparative Study of Training-Free Multi-Subject Imageto-Video Generation. In The fourth International Workshop on Rich Media with Generative AI (RichMediaGAI ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3841458. 3841542

## 1 Introduction

Text-conditioned image-to-video (I2V) generation has gained significant attention for its ability to synthesize expressive videos from a single static image and a textual motion prompt [10]. Formally, given an input image � and a prompt � , the task is to generate a video � that begins with � and reflects the semantics of�. Recent difusion-based models [9, 24, 25] have achieved impressive singlesubject video generation, supporting applications in entertainment, virtual reality, and human-computer interaction [13, 31].

Despite these successes, multi-subject I2V generation, in which distinct subjects may follow diferent prompt-specified motions, remains underexplored. Preserving multiple subject appearances while following individual and contextual motion descriptions remains challenging, raising the question of how frozen pretrained I2V models can support multi-subject generation without additional training. To address this question, we investigate three representative training-free paradigms: direct, parallel, and sequential generation.

As shown in Fig. 1(a), a direct strategy applies existing I2V models [29–31] to generate the entire scene and all subject motions in a single generation process. This strategy is simple and preserves the complete image–prompt context and overall scene semantics. However, jointly preserving multiple identities, synthesizing distinct motions, and maintaining realistic inter-subject relationships places a high burden on the model and can produce entangled appearance, motion, or interaction artifacts.

Parallel generation, illustrated in Fig. 1(b), alleviates this joint synthesis complexity through decomposition. It decomposes the image and prompt into subject-specific pairs, generates their videos in parallel with a pretrained I2V model, and merges the resulting videos using video composition techniques. This strategy reduces the per-subject generation burden and can better preserve individual appearance and motion. However, inter-subject context is not jointly modeled during the independent generation stage. The post-hoc composition stage must therefore reconcile independently

<sup>∗</sup>Work done as a research scholar at the University of Memphis.   
<sup>†</sup>Corresponding author: hni@memphis.edu.

![](images/6d6df42ac3aa43f8002b43db1696474afe8cf82b9b8c9f805c03acfbcd370897.jpg)  
(a) Direct Generation

High joint-synthesis complexity <sup>�</sup>� Global-context preservation  
![](images/71529c1b67236134bd30bc80c0c0ae711c7e02504021f89f020858ed360d0c9f.jpg)  
(b) Parallel Generation

Limited inter-subject coordination <sup>��</sup> Low per-subject complexity  
![](images/3fd025c5e996ddc569492203dcd42adbbe55a22e63d53cc3e9a512847b9796b3.jpg)  
Figure 1: Illustration of three training-free strategies for multi-subject image-to-video generation. Direct generation synthesizes all subjects jointly in one pass; parallel genera tion independently generates subject-level videos and merges them post hoc; sequential generation injects subjects one by one while conditioning on previously generated video context. The pretrained image-to-video (I2V) models remain frozen across all methods. For simplicity, the decomposition of the input multi-subject image and text prompt is omitted in (b) and (c).

generated motions, spatial layouts, occlusions, and interactions, which may produce inconsistent or physically implausible videos.

The third strategy is sequential generation, which introduces subjects progressively while propagating the previously generated context. As illustrated in Fig. 1(c), each stage introduces one new subject while conditioning on the video containing previously synthesized subjects, reducing the synthesis burden while preserving evolving context. However, its progressive generation increases inference time and may propagate ordering-dependent or accumulated errors.

We compare direct, parallel, and sequential training-free multi subject I2V generation on a self-created dataset under a unified experimental protocol. Our results show that no paradigm is uni versally superior: direct generation is simple and eficient but faces high joint synthesis complexity; parallel generation reduces persubject complexity but loses inter-subject context; and sequential generation preserves progressively accumulated context but increases inference time and is susceptible to error propagation.

## 2 Related Work

Image-to-Video Generation. Based on the availability of motion cues, image-to-video (I2V) generation can be categorized into stochastic methods, which rely solely on a given image � [2], and conditional methods, which utilize both the image � and an external condition � [18]. In this work, we primarily focus on textconditioned I2V generation [7, 17, 31]. Hu et al. [10] introduced MAGE, a text-driven I2V generator that employs a motion anchor structure to capture appearance-motion-aligned representations using three-dimensional axial transformers. Zhang et al. [31] proposed I2VGen-XL, a cascaded framework consisting of two stages: a base stage that preserves input semantics through dual hierarchical encoders, and a refinement stage that incorporates the text prompt to enhance visual details and improve video resolution. While these general-purpose I2V frameworks can produce promising results, they are not specifically designed for multi-subject scenarios and often struggle in such settings due to the need to simultaneously synthesize the appearance and motion of multiple subjects. This limitation motivates our study of how pretrained I2V models can be used under diferent training-free multi-subject generation strategies.

Multi-Subject Video Generation. Recent work on multi-subject video generation primarily focuses on customized multi-subject text-to-video (T2V) generation [3–6, 28]. Chen et al. [3] introduced VideoDreamer, which utilizes a combination of Disen-Mix finetuning and human-in-the-loop re-fine-tuning to adapt a pretrained T2V generator for multiple subjects. Chen et al. [5] proposed Video Alchemist, a difusion-transformer-based framework that fuses each conditional reference image with its corresponding subject-level text prompt via cross-attention layers. Compared to multi-subject T2V, multi-subject I2V is more challenging due to the additional constraints imposed by the input image. The generation must not only synthesize coherent motions for multiple subjects but also preserve their appearances, spatial positioning, and inter-subject relationships as defined by the provided image. In contrast to these customization-oriented T2V methods, our work focuses on trainingfree multi-subject I2V and compares direct, parallel, and sequential generation strategies using frozen pretrained components.

## 3 Methodology

## 3.1 Problem Formulation and Paradigm Overview

We study training-free multi-subject image-to-video (I2V) generation as a comparison of three generation paradigms: direct generation, parallel generation, and sequential generation. Formally, given an input image � containing � subjects $\{ S _ { i } \} _ { i = 1 } ^ { N }$ and a text prompt � describing their motions and scene context, the goal is to synthesize a video sequence $V = \langle v ^ { 1 } , \ldots , v ^ { L } \rangle$ with � frames. The generated video should preserve the appearance of each subject, follow the motion described in the prompt, and maintain coherent spatial and temporal relationships among subjects and the background.

As shown in Fig. 2, the three paradigms organize the generation process in diferent ways. Direct generation applies the pretrained I2V model M to the full image and prompt in one pass, requiring the model to jointly synthesize all subjects and their motions. Parallel generation first decomposes the input image and prompt into background-level and subject-level components, generates each component independently, and then composes the resulting videos into a multi-subject video. Sequential generation also uses decomposed components, but injects subjects one by one into a shared evolving video context, followed by temporal refinement. Below, we first introduce direct generation, then describe the image and text-prompt decomposition, and finally present the parallel and sequential generation paradigms built upon this decomposition.

![](images/d431e42b43fb90769c9277eda70f3c54a069d86b25286e82a9e31de95addb79b.jpg)  
Figure 2: Overview of the compared training-free paradigms for multi-subject image-to-video generation. We compare three strategies: (a) direct generation, which uses the original image and prompt in a single call to a frozen I2V model; (b) parallel generation via video composition, which independently generates decomposed background and subject videos and subsequently composes them; and (c) sequential generation with progressive injection, which inserts subjects one by one into an evolving video. For the decomposed paradigms in (b) and (c), the input prompt � and image � are decomposed into a background pair $( T _ { 0 } , I _ { 0 } )$ and subject-specific pairs $( T _ { i } , I _ { i } ) _ { i = 1 } ^ { N }$ . In (c), stages I, II, and III denote global prior generation, spatial subject injection, and temporal motion refinement, respectively.

## 3.2 Direct Generation

Direct generation is the most straightforward strategy for multisubject I2V generation. It directly applies the pretrained I2V model

M to the full text prompt � and the original multi-subject image �:

$$
V ^ { \mathrm { d i r } } = { \cal M } ( T , I ) .\tag{1}
$$

Unlike parallel and sequential generation, this paradigm does not perform text-image decomposition or post-hoc composition. Thus, the model can access the complete image-prompt context throughout generation, preserving the original spatial layout and global semantic description.

However, direct generation requires M to synthesize all subjects, their individual motions, and their inter-subject relationships in a single pass. This joint synthesis process places a high burden on the frozen I2V model and may lead to identity confusion, motion entanglement, missing subjects, or weak inter-subject interactions. Therefore, direct generation serves as a simple but challenging reference paradigm for evaluating multi-subject I2V generation.

## 3.3 Image and Text Prompt Decomposition

Direct generation operates on the original input image and prompt, while parallel and sequential generation both rely on decomposed text-image components. Inspired by the “divide-and-conquer” principle, we first decompose both the text prompt � and the input image � into modular components that can later be recombined in a controlled manner.

This shared decomposition step provides a common input representation for the two decomposed paradigms, ensuring that their diferences mainly come from the subsequent generation process rather than from diferent preprocessing procedures. In particular, text decomposition is performed before image decomposition, since subject identities are first established from the prompt and then used to guide the image decomposition module to isolate each subject.

Text Decomposition. Given the input text prompt � as global guidance, we decompose it into subject-level descriptions $\{ T _ { i } \} _ { i = 1 } ^ { N }$ and a scene-level description �<sub>0</sub> via a large language model (LLM) W:

$$
\{ T _ { i } \} _ { i = 0 } ^ { N } = { \mathcal { W } } ( T ) .\tag{2}
$$

The subject-level prompt $T _ { i }$ specifies the identity, attributes, and motion of subject $S _ { i } ,$ while removing descriptions of the other subjects as much as possible. The scene-level prompt $T _ { 0 }$ captures the background environment and global scene context. This decomposition establishes an explicit correspondence between textual subjects and visual subjects before the image is processed.

Image Decomposition. With $\{ T _ { i } \} _ { i = 0 } ^ { N }$ obtained, we apply an image decomposition pipeline $\mathcal { P }$ to the input image �:

$$
\{ I _ { i } \} _ { i = 0 } ^ { N } = \mathcal { P } ( I , \{ T _ { i } \} _ { i = 1 } ^ { N } ) .\tag{3}
$$

Specifically, we first apply pretrained object detection and segmentation models to identify each subject in the input image $I ,$ yielding subject masks and corresponding bounding boxes. To obtain the background image $I _ { 0 } ,$ all foreground subjects are removed and the missing regions are reconstructed using a pretrained inpainting model. To obtain single-subject images, we mask out all other subjects through segmentation masks and use a pretrained inpainting model to reconstruct the background in the masked regions. Finally, we crop the preserved subject region using the bounding box, producing a set of single-subject images $\{ I _ { i } \} _ { i = 1 } ^ { N }$ and a clean background image $I _ { 0 }$ (with all subjects removed). The resulting text-image pairs $\{ ( T _ { i } , I _ { i } ) \} _ { i = 0 } ^ { N }$ are then used as the shared decomposed inputs for parallel and sequential generation.

## 3.4 Parallel Generation via Video Composition

Parallel generation first synthesizes decomposed components independently and then composes them into a multi-subject video. Given the decomposed text-image pairs $\{ ( T _ { i } , I _ { i } ) \} _ { i = 0 } ^ { N }$ , we independently generate a background video and � subject-level videos:

$$
V _ { i } ^ { \mathrm { p a r } } = M ( T _ { i } , I _ { i } ) , i = 0 , 1 , . . . , N .\tag{4}
$$

Here, $V _ { 0 } ^ { \mathrm { p a r } }$ denotes the generated background video, and $V _ { i } ^ { \mathrm { p a r } }$ denotes the generated video object for subject $S _ { i }$ . Thus, each I2V call only handles one subject or the background.

To merge the independently generated component videos, we adopt a training-free video object composition module from [27]. Diferent from the original video object composition setting, where the inputs are multiple source video objects, our parallel paradigm first converts each decomposed text-image pair $( T _ { i } , I _ { i } )$ into a generated component video $V _ { i } ^ { \mathrm { { \hat { p } a r } } }$ . The resulting component videos are then provided to the composition module as the source videos corresponding to the individual subjects.

Algorithm 1 Parallel generation via video composition.   
Require: Decomposed text-image pairs $\{ ( T _ { i } , I _ { i } ) \} _ { i = 0 } ^ { N } ;$ global prompt �; pre   
trained I2V model M; mask extraction module Q; video composition   
module C.   
Ensure: A synthesized video $V ^ { \mathrm { p a r } }$ with � frames.   
1: for $i = 0 , 1 , \cdots , N \ ,$ do   
2: $V _ { i } ^ { \mathrm { p a r } } \gets { \mathcal { M } } ( T _ { i } , I _ { i } )$ // Independent I2V generation   
3: $\dot { M } _ { i } ^ { \mathrm { p a r } } \gets Q ( V _ { i } ^ { \mathrm { p a r } } )$ // Mask extraction   
4: $Z _ { i } ^ { \mathrm { { \dot { p a r } } } } \gets \mathrm { D D I M I n v } ( V _ { i } ^ { \mathrm { { p a r } } } )$ // Latent inversion   
5: end for   
6: $V ^ { \mathrm { p a r } }  C ( T , I , \{ Z _ { i } ^ { \mathrm { p a r } } , M _ { i } ^ { \mathrm { p a r } } \} _ { i = 0 } ^ { N } )$ // Video object composition   
7: return $V ^ { \mathrm { p a r } }$

Following [27], for each generated component video $V _ { \underline { { i } } } ^ { \mathrm { p a r } }$ , we first use the segmentation model $\boldsymbol { Q }$ to extract a sequence of masks for the corresponding subject:

$$
M _ { i } ^ { \mathrm { p a r } } = \mathcal { Q } ( V _ { i } ^ { \mathrm { p a r } } ) .\tag{5}
$$

We also apply DDIM inversion [24] to obtain its inverted latent trajectory:

$$
Z _ { i } ^ { \mathrm { p a r } } = \mathrm { D D I M I n v } ( V _ { i } ^ { \mathrm { p a r } } ) .\tag{6}
$$

The masks localize object regions, while the inverted latents preserve motion and appearance cues.

The composition module C then utilizes the original image and prompt, the object masks, and the inverted latents to generate the final multi-subject video:

$$
V ^ { \mathrm { p a r } } = { C } ( T , I , \{ Z _ { i } ^ { \mathrm { p a r } } , M _ { i } ^ { \mathrm { p a r } } \} _ { i = 0 } ^ { N } ) .\tag{7}
$$

During composition, C leverages feature and attention guidance from the component videos to preserve subject identity and motion consistency in the final video. Further details can be found in [27].

The advantage of parallel generation lies in its modularity: each I2V call handles only one subject or the background, simplifying appearance preservation and motion synthesis. However, independently generated subject videos do not model inter-subject context during generation. Consequently, spatial compatibility, occlusion, shared shadows, and coordinated interactions are dificult to reconstruct during post-hoc composition, making this paradigm less reliable for strongly interacting subjects.

## 3.5 Sequential Generation with Progressive Injection

Sequential generation progressively inserts decomposed subjects into a shared, evolving video state, thereby retaining more visual context than independent parallel generation while avoiding the joint synthesis of all subjects in a single pass. Inspired by coarse-tofine video generation and object insertion frameworks [8, 31, 32], we instantiate this paradigm as a cascaded pipeline comprising global prior generation, spatial subject injection, and temporal motion refinement. All stages use of-the-shelf pretrained models without additional training or fine-tuning.

Global prior generation. We first obtain a global motion prior from the same frozen I2V backbone used in direct generation. Given the original multi-subject image � and the full prompt � , the backbone generates a direct-generation video $V ^ { \mathrm { d i r } }$ using Eq. 1. Based on $V ^ { \mathrm { d i r } }$ , the global prior generation module $\mathcal { G }$ produces an initial background video and subject-level trajectories:

Algorithm 2 Sequential generation with progressive injection.   
Require: Original image � with � subjects; global prompt � ; decomposed   
images $\{ I _ { i } \} _ { i = 0 } ^ { N } ;$ ; pretrained I2V model $M ; \mathrm { g }$ lobal prior module $\mathcal { G } ;$ image   
composition model H; temporal refinement model R.   
Ensure: A synthesized video $V ^ { s e q }$ with � frames.   
<sub>�</sub>dir $\gets \dot { \mathcal { M } } ( T , I )$ // Direct-generation motion prior   
2: $V _ { 0 } ^ { \mathrm { s e q } } , \{ \mathcal { B } _ { i } \} _ { i = 1 } ^ { N }  \mathcal { G } ( V ^ { \mathrm { d i r } } )$ // Background video and trajectories   
3: for $i = 1 , 2 , \cdots , N$ do   
4: for $t = 1 , 2 , \cdots , L$ do   
5: $v _ { i } ^ { t }  \mathcal { H } ( I _ { i } , b _ { i } ^ { t } , v _ { i - 1 } ^ { t } )$ // Spatial composition   
6: end for   
7: end for   
8: $V ^ { \mathrm { s e q } } \gets \mathcal { R } ( V _ { N } ^ { \mathrm { s e q } } , T , I )$ // Temporal refinement   
9: return $V ^ { s e q }$

$$
V _ { 0 } ^ { \mathrm { s e q } } , \{ \mathcal { B } _ { i } \} _ { i = 1 } ^ { N } = \mathcal { G } ( V ^ { \mathrm { d i r } } ) .\tag{8}
$$

Here, $V _ { 0 } ^ { \mathrm { s e q } } = \langle \boldsymbol { v } _ { 0 } ^ { 1 } , \boldsymbol { v } _ { 0 } ^ { 2 } , \ldots , \boldsymbol { v } _ { 0 } ^ { L } \rangle$ denotes the background video obtained by removing foreground subjects from the direct-generation video ${ \dot { V } } ^ { \dim }$ , and $\mathcal { B } _ { i } ^ { - } = \langle b _ { i } ^ { \bar { 1 } } , b _ { i } ^ { 2 } , \dots , b _ { i } ^ { L } \rangle$ denotes the bounding-box sequence of subject $S _ { i }$ . These trajectories are used as coarse spatiotemporal anchors for later insertion.

Spatial subject injection. Starting from $V _ { 0 } ^ { s e q }$ , subjects are inserted sequentially according to a predefined order. At step �, the current video state $\dot { V } _ { i - 1 } ^ { \mathrm { s e q } }$ already contains subjects $\{ S _ { j } \} _ { j = 1 } ^ { i - 1 }$ . Given the image $I _ { i }$ of the new subject $S _ { i } ,$ its trajectory ${ \mathcal { B } } _ { i } ,$ , and the �-th frame $v _ { i - 1 } ^ { t }$ of $V _ { i - 1 } ^ { \mathrm { s e q } }$ , a pretrained image composition model H produces the updated frame containing $S _ { i } { \mathrm { : } }$

$$
\boldsymbol { v } _ { i } ^ { t } = \mathcal { H } ( \boldsymbol { I } _ { i } , \boldsymbol { b } _ { i } ^ { t } , \boldsymbol { v } _ { i - 1 } ^ { t } ) ,\tag{9}
$$

where $b _ { i } ^ { t }$ specifies the spatial location of subject $S _ { i }$ in frame �. Ap plying this operation across all frames gives

$$
V _ { i } ^ { \mathrm { s e q } } = \langle { v _ { i } ^ { 1 } , v _ { i } ^ { 2 } , \ldots , v _ { i } ^ { L } } \rangle .\tag{10}
$$

After all � subjects are inserted, the procedure produces a coarse multi-subject video $V _ { N } ^ { \mathrm { s e q } }$ . This stage explicitly controls subject placement and preserves the previously generated visual context, but its frame-wise nature may introduce temporal artifacts.

Temporal motion refinement. Since the coarse video $V _ { N } ^ { \mathrm { s e q } }$ is obtained through frame-level composition, it may contain interframe inconsistency, boundary artifacts, or unnatural local motion. We therefore apply a pretrained text- and image-conditioned videoto-video (V2V) refinement model R:

$$
V ^ { \mathrm { s e q } } = \mathcal { R } \Big ( V _ { N } ^ { \mathrm { s e q } } , T , I \Big ) .\tag{11}
$$

Here, � and � denote the original multi-subject image and global text prompt, respectively. This refinement stage uses the temporal prior of the pretrained V2V model to improve motion smoothness and visual continuity while preserving the composed subject layout.

Overall, sequential generation represents a middle ground be tween direct and parallel paradigms. By progressively updating the intermediate video state $\dot { V } _ { i } ^ { \mathrm { s e q } }$ , it reduces the burden of synthesizing all subjects simultaneously while retaining more shared visual con text than fully independent parallel generation. This cumulative context allows later inserted subjects to be composed with previously inserted ones, which can better preserve spatial compatibility and local interactions. However, sequential generation is inherently order-dependent: subjects inserted earlier are generated without awareness of those introduced later, which may limit coordinated motion and interaction modeling. In addition, errors from early insertion steps may propagate and accumulate throughout the sequence. The final output is therefore sensitive to the insertion order, the quality of the extracted trajectories, frame-level composition errors, and the robustness of the temporal refinement module.

## 4 Experiments

We conduct extensive experiments to compare direct, parallel, and sequential generation for multi-subject image-to-video synthesis. We describe the self-constructed benchmark, evaluation metrics, and implementation protocol, then analyze the three paradigms through quantitative results, qualitative comparisons, ablation studies, eficiency evaluation, and failure cases.

## 4.1 Dataset and Metrics

To comprehensively compare diferent paradigms for multi-subject I2V generation, we construct a controlled multi-subject dataset. The construction process involves two steps: (1) Text Prompt Generation. We manually design text prompts that describe natural scenes involving two subjects with explicit motion descriptions. This choice emphasizes the multi-subject nature of our task while keeping the scenarios interpretable and controlled. (2) Image Prompt Generation. Given each text prompt, we use Stable Difusion 3.5-Large [22] to synthesize a corresponding multi-subject image at the resolution of 512 × 512. Low-quality outputs are manually filtered out, resulting in a dataset of 20 high-quality text–image pairs. All three paradigms, including direct, parallel, and sequential generation, are evaluated on the same dataset to ensure a fair comparison.

For evaluation, we adopt five representative metrics from VBench [11], including: subject consistency (SC), background consistency (BC), motion smoothness (MS), aesthetic quality (AQ), and imaging quality (IQ).

## 4.2 Implementation Details

Compared Paradigms and Backbones. We implement and compare three paradigms for multi-subject I2V generation: direct, parallel, and sequential generation. To examine whether the comparison is consistent across diferent pretrained I2V models, we evaluate the three paradigms with two backbones, I2VGen-XL [31] and DynamiCrafter [29]. Unless otherwise specified, I2VGen-XL is used as the default I2V backbone M due to its strong video synthesis capability, and the guidance scale � is set to 9.

Shared Decomposition Modules. We use the same text and image decomposition pipeline introduced in Sec. 3.3 for both Parallel Generation and Sequential Generation. For text decomposition, we employ GPT-4 [1] as $\mathcal { W }$ to decompose the original prompt � into a set of prompts $\begin{array} { r } { \{ T _ { i } \} _ { i = 0 } ^ { N } , } \end{array}$ where $T _ { 0 }$ describes the background and �<sub>�</sub> describes subject � for $i = 1 , \ldots , N$ . For image decomposition, we use Grounding DINO [15] for object detection and SAM [12] for segmentation, while background inpainting is carried out by Stable Difusion-1.5 [22]. This shared decomposition design ensures that the comparison between parallel and sequential generation is not afected by diferent decomposition modules.

<table><tr><td colspan="4">Setting</td><td colspan="5">Metrics</td></tr><tr><td>Paradigm</td><td>BB</td><td>CFG</td><td>Steps</td><td>SC(%)</td><td>BC(%)</td><td>MS(%)</td><td>AQ(%)</td><td>IQ(%)</td></tr><tr><td rowspan="8">Direct</td><td rowspan="4">DC</td><td rowspan="2">7</td><td>25</td><td>92.34</td><td>95.15</td><td>97.38</td><td>60.11</td><td>57.25</td></tr><tr><td>50</td><td>91.54</td><td>95.20</td><td>96.87</td><td>60.56</td><td>57.83</td></tr><tr><td>9</td><td>25</td><td>91.36</td><td>94.45</td><td>97.16</td><td>58.80</td><td>56.90</td></tr><tr><td rowspan="2"></td><td>50</td><td>90.91</td><td>94.78</td><td>96.81</td><td>59.43</td><td>56.53</td></tr><tr><td>25</td><td>85.37</td><td>90.27</td><td>95.83</td><td>58.07</td><td>71.26</td></tr><tr><td rowspan="3">I2V</td><td>7</td><td>50</td><td>91.43</td><td>94.66</td><td>98.49</td><td>64.28</td><td>73.48</td></tr><tr><td>9</td><td>25</td><td>85.34</td><td>90.63</td><td>95.97</td><td>58.90</td><td>70.26</td></tr><tr><td></td><td>50</td><td>91.26</td><td>94.79</td><td>98.44</td><td>64.05</td><td>72.51</td></tr><tr><td rowspan="6">Parallel</td><td rowspan="3">DC</td><td>7</td><td>25</td><td>89.56</td><td>93.72</td><td>96.14</td><td>58.06</td><td>66.62</td></tr><tr><td rowspan="2">9</td><td>50</td><td>87.64</td><td>92.83</td><td>94.74</td><td>59.81</td><td>68.04</td></tr><tr><td>25</td><td>89.75</td><td>93.59</td><td>95.39</td><td>58.82</td><td>68.09</td></tr><tr><td rowspan="3">I2V</td><td rowspan="2">7</td><td>50</td><td>87.35</td><td>92.64</td><td>94.31</td><td>58.05</td><td>68.35</td></tr><tr><td>25</td><td>85.50</td><td>90.97</td><td>94.35</td><td>56.67</td><td>71.86</td></tr><tr><td rowspan="2"></td><td>50</td><td>91.77</td><td>94.17</td><td>96.34</td><td>61.05</td><td>71.69</td></tr><tr><td>9</td><td>25</td><td>85.57</td><td>90.73</td><td>94.29</td><td>56.44</td><td>69.91</td></tr><tr><td rowspan="8">Sequential</td><td rowspan="4">DC</td><td rowspan="2">7</td><td>50 25</td><td>90.32</td><td>93.58</td><td>96.18</td><td>59.76</td><td>70.03</td></tr><tr><td></td><td>93.41</td><td>95.49</td><td>96.73</td><td>65.88</td><td>73.93</td></tr><tr><td rowspan="2"></td><td>50 25</td><td>94.13</td><td>95.46</td><td>96.71</td><td>64.15</td><td>73.77</td></tr><tr><td>9</td><td>93.25</td><td>94.83</td><td>95.87</td><td>63.03</td><td>72.28</td></tr><tr><td rowspan="3">I2V</td><td></td><td>50</td><td>92.69</td><td>95.00</td><td>96.75</td><td>61.73</td><td>72.13</td></tr><tr><td rowspan="2">7</td><td>25 50</td><td>91.50</td><td>93.74</td><td>94.59</td><td>59.83</td><td>71.64</td></tr><tr><td>25</td><td>93.46 91.43</td><td>95.03 93.83</td><td>95.74 93.91</td><td>60.01 58.53</td><td>70.27 68.06</td></tr><tr><td rowspan="2"></td><td rowspan="2">9</td><td>50</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>93.98</td><td>94.94</td><td>95.69</td><td>59.77</td><td>69.31</td></tr></table>

Table 1: Quantitative comparison of three multi-subject I2V generation paradigms under diferent backbone and sampling settings. BB denotes the pretrained I2V backbone, where DC and I2V refer to DynamiCrafter and I2VGen-XL, respectively. CFG denotes the classifier-free guidance scale ${ \mathit { g } } ,$ and Steps denotes the number of denoising steps used in the I2V generation stage. We report five VBench-style metrics: subject consistency (SC), background consistency (BC), motion smoothness (MS), aesthetic quality (AQ), and imaging quality (IQ). All scores are reported as percentages, where higher values indicate better performance. Within each paradigm, the best score over diferent backbones, CFG, and step settings is highlighted in bold.

Parallel Generation. Given the decomposed text-image pairs, background and subject videos are independently generated using the backbone, guidance scale, and denoising steps specified in Table 1. The mask extraction module Q is instantiated as a fully pretrained Grounded-SAM-2 pipeline [21]. Specifically, for each independently generated subject video, Grounding DINO Tiny [15] detects candidate object boxes in the first frame using the corre sponding subject prompt, with box and text thresholds set to 0.20 and 0.15, respectively. CLIP ViT-L/14 [19] re-ranks the candidate boxes according to text–image similarity, and the selected box is provided as the initial prompt to SAM 2.1 Hiera Large [20], which extracts the first-frame instance mask and propagates it through the remaining video frames to produce masks for the whole video. No manually annotated masks or additional training are used. Each video branch is subsequently inverted into DDIM latents using 500 inversion steps with $g = 1 . 0$ . We then employ MVOC [27] as the video composition module C to compose the inverted branches through mask-guided Plug-and-Play feature and attention injection [26], using 50 denoising steps with $g \ = \ 9 . 0 $ . All remaining composition hyperparameters follow the oficial MVOC defaults. Sequential Generation. The image composition model H is implemented with TF-ICON [16], where the coeficients are set to $\sigma _ { i a } = 0 . 2$ for the interaction area and $\sigma _ { o } = 0 . 1 5$ for the subject mask.

The video-to-video translation model R is realized by AnyV2V [14], with 50 denoising steps for temporal refinement.

Common Settings. All models are executed on one NVIDIA RTX 6000 Ada GPU with mixed precision, using bfloat16 or fp16 depending on the model. We standardize all outputs to $5 1 2 \times 5 1 2$ resolution, 16 frames, and 8 FPS across all experiments. To ensure fairness, each paradigm is evaluated under the same dataset, metrics, and output protocol, while each pretrained model is used with its oficial default hyperparameters when applicable. We generate 100 videos for each paradigm.

## 4.3 Result Analysis

Quantitative comparison. Table 1 compares the three paradigms using I2VGen-XL [31] and DynamiCrafter [29] under matched classifier-free guidance scales and sampling steps. Sequential generation achieves the highest subject consistency (SC) and background consistency (BC) in all matched configurations, whereas direct generation consistently provides the highest motion smoothness (MS). This reveals a stable trade-of: progressive insertion preserves accumulated subject and scene context, while single-pass generation avoids temporal disturbances introduced by composition and refinement.

“In a quiet suburban backyard, a rabbit is hopping quickly across the yard, while a dog is running fast across the scene.”

![](images/2a8e468a59f7052a3e36bb9d7b375450799f57a8de51a10351e29db6b5dee40c.jpg)

“In a quiet riverside meadow, a cow is grazing, while a duck is waddling toward the water.”  
![](images/2e110050e8e506edc0a534fe5515d5da36fb272cadfbc84308a4e9370e1fbf6a.jpg)  
Figure 3: Qualitative comparison of the three paradigms for multi-subject I2V generation. All paradigms use the same pretrained I2VGen-XL [31] backbone. Columns in each block display the 1st, 3rd, 7th, 11th, and 15th frames of the generated videos. The input image � is highlighted with a red box, and the text prompt � is shown above each block. Representative unnatural details are highlighted with red circles.

Aesthetic quality (AQ) and imaging quality (IQ) are more dependent on the I2V backbone. Sequential generation performs the best with DynamiCrafter in most settings, while direct generation generally retains stronger frame-level quality with I2VGen-XL, particularly at 50 sampling steps. Parallel generation reduces the com plexity of each I2V call through independent component generation, but the lack of shared temporal context and the subsequent compo sition stage prevent this modularity from consistently improving the final metrics. Overall, no paradigm dominates all dimensions: direct generation favors temporal smoothness, sequential generation favors compositional consistency, and frame-level quality depends on compatibility with the underlying backbone.

Paradigm-level observations. Direct generation synthesizes all subjects and their motions jointly through a single pretrained I2V sampling process. This design consistently produces the highest mo tion smoothness because it does not require independent video composition or progressive frame-level modification. However, jointly handling multiple subjects places the full compositional burden on the pretrained backbone, and its SC and BC remain consistently below those of sequential generation.

Parallel generation independently animates the decomposed background and subject components before merging them through video object composition. Although this design simplifies each individual generation call and permits the component streams to be generated independently, the resulting videos do not share a unified temporal and contextual representation. Consequently, the reduced per-call synthesis complexity does not consistently translate into stronger final-video metrics, while the subsequent composition process may introduce additional inconsistency.

Sequential generation progressively inserts subjects into an evolving video using frame-level composition and temporal refinement. By retaining the video context accumulated after each insertion, it consistently achieves the strongest subject and background consistency across both backbones and all evaluated settings. However, repeated insertion and refinement reduce motion smoothness relative to direct generation. Their efect on frame-level quality is backbone-dependent: sequential generation is particularly efective with DynamiCrafter, whereas direct generation retains stronger AQ and IQ with I2VGen-XL. These findings suggest that progressive insertion reliably shifts generation toward stronger compositional consistency, but its overall perceptual benefit depends on its compatibility with the underlying I2V backbone.

Qualitative comparison. Figure 3 compares direct, parallel, and sequential generation across two input image–prompt pairs, using the same input conditions and I2VGen-XL [31] backbone within each example. In the rabbit–dog example, Direct generation preserves the rabbit reasonably well, but the dog loses its recognizable facial structure and gradually appears as an ambiguous long-eared animal. Parallel generation also maintains the rabbit, whereas the dog progressively acquires rabbit-like characteristics, indicating cross-subject identity confusion. Sequential generation preserves the identities ofboth animals more consistently throughout the sampled frames. In the cow–duck example, however, all three paradigms exhibit diferent visual artifacts. Direct generation produces an anatomically implausible number of cow legs from the early frames, while parallel generation develops a similar leg-count inconsistency toward the end of the video. Sequential generation maintains the cow and duck identities over most frames, but the duck’s head becomes visibly flattened when it turns to a side view in the final frame.

Computational trade-of. Table 2 reports the end-to-end inference time and peak VRAM consumption of the three paradigms for a single input. Direct generation achieves the lowest runtime and memory consumption because it requires only one I2V sampling pass using the original multi-subject image and text prompt. Parallel generation performs separate I2V generation for the background and individual subjects, followed by an additional video object composition stage. The complete pipeline therefore remains considerably more expensive than direct generation because it requires multiple I2V calls, stores multiple generated streams, and performs an additional composition process. Sequential generation incurs the highest computational cost because subjects are inserted progressively. Each insertion requires a DDIM-inversion-based composition operation [24, 26], followed by temporal refinement of the updated video. Consequently, inserting � subjects requires repeated composition and refinement rather than a single post-hoc composition step. Its higher peak VRAM further reflects the additional video latents and intermediate states maintained during the multi-stage composition and refinement pipeline. These results reveal a clear runtime–memory–consistency trade-of: direct generation minimizes computational resource usage, whereas sequential generation consumes substantially more time and GPU memory to preserve the context accumulated after each subject insertion.

<table><tr><td>Backbone</td><td>Metric</td><td>Direct</td><td>Parallel</td><td>Sequential</td></tr><tr><td rowspan="2">I2VGen-XL</td><td>Time (s) ↓</td><td>29.48</td><td>240.25</td><td>441.45</td></tr><tr><td>Peak VRAM (GB) ↓</td><td>8.98</td><td>12.93</td><td>26.42</td></tr><tr><td rowspan="2">DynamiCrafter</td><td>Time (s) ↓</td><td>59.00</td><td>251.97</td><td>430.14</td></tr><tr><td>Peak VRAM (GB) ↓</td><td>10.58</td><td>14.21</td><td>23.94</td></tr></table>

Table 2: Computational costs of the three paradigms with two pretrained I2V backbones. Inference time and peak VRAM are measured during end-to-end generation with a single input on an NVIDIA RTX 6000 Ada GPU. Peak VRAM denotes the maximum GPU memory usage observed throughout the complete generation pipeline.

Failure case analysis. The three paradigms exhibit distinct failure modes reflecting their generation strategies, as illustrated in Fig. 4. For direct generation, jointly synthesizing multiple subjects with distinct motions may exceed the capability of the pretrained I2V backbone. In the first row, the bufalo moves visibly while the bird remains nearly static, suggesting dificulty in assigning plausible motions to all subjects simultaneously.

Parallel generation inherits errors from independently generated subject-level videos. Errors in appearance or motion are directly propagated to the final composition, as shown by the horse in the second row gradually developing an abnormal blue appearance. Since the composition stage mainly integrates existing streams, it has limited ability to correct such errors.

Sequential generation is sensitive to errors introduced by intermediate processing and insertion. In the third row, inaccurate masks produce residual wing artifacts and an abnormal wing structure, while trajectory or motion estimation errors cause the dog to move backward. Such intermediate errors may persist or accumulate across subsequent stages.

Overall, direct generation is limited by the backbone’s generation capability, parallel generation by inherited subject-wise errors, and sequential generation by errors propagated through intermediate modules.

## 5 Conclusion and Discussion

We present a systematic comparison of three paradigms for multi subject image-to-video generation: direct, parallel, and sequential generation. Under matched backbones and sampling settings, the results reveal a consistent division of strengths. Sequential generation achieves the strongest subject and background consistency, while direct generation consistently produces the smoothest motion and requires substantially less inference time. Frame-level aesthetic and imaging quality are more backbone-dependent, favoring sequential generation with DynamiCrafter but direct generation with I2VGen-XL. Parallel generation provides a modular decomposition of multi-subject synthesis, although independent generation and post-hoc composition do not consistently improve the final quantitative results. These indicate that no paradigm is universally optimal: direct generation favors eficiency and temporal smoothness, parallel generation favors modularity, and sequential generation favors compositional consistency at a higher computational cost.

“In a green rice field, an egret is taking short flights above the plants, while a buffalo is walking slowly through the field. “  
![](images/768cf1b89a55f6939644f39bc8e3cb7b79df0b312a259f606462b7aea077585b.jpg)

“In a sunlit farmland, a horse is trotting along a dirt path, while a chicken is pecking and moving across the ground.”  
![](images/14a55a4e7ca697bafcaf9d72f0a01c4f7c47bdc41368a2ec049e18447f29ae41.jpg)

“In a sunlit grassy park, a dog is running across the field, while a pigeon is flying low above the ground.”  
![](images/ea0ca213eba8f28b835d30377c5a78296ae8c0cca3e3546e56c960dfa6c28e02.jpg)  
Figure 4: Failure cases of the three paradigms for multisubject I2V generation. Columns in each block display the 1st, 3rd, 8th, and 13th frames of the generated videos. Representative failure regions are highlighted with red circles.

Limitations and Future Work. Our study is limited to a relatively small synthetic benchmark, two pretrained I2V backbones, and five automatic evaluation metrics. The observed trade-ofs may therefore change with larger real-world datasets, stronger video generation models, or human evaluation. Moreover, all paradigms remain constrained by the capability of their pretrained I2V backbones, while parallel and sequential generation additionally depend on the reliability of decomposition and composition modules. Future work may extend this comparison to more backbones and subjects, introduce human-aligned evaluations, and investigate hybrid strategies that combine the eficiency of direct generation with the context preservation of progressive composition.

## Acknowledgments

Research reported in this publication was supported in part by computing resources from the iTiger GPU cluster [23], funded by NSF award CNS-2318210 and partially by generous contributions from the College of Arts and Sciences and Information Technology Services at the University of Memphis.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).

[2] Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H Campbell, and Sergey Levine. 2017. Stochastic variational video prediction. arXiv preprint arXiv:1710.11252 (2017).

[3] Hong Chen, Xin Wang, Guanning Zeng, Yipeng Zhang, Yuwei Zhou, Feilin Han, and Wenwu Zhu. 2023. Videodreamer: Customized multi-subject text-to-video generation with disen-mix finetuning. arXiv preprint arXiv:2311.00990 (2023).

[4] Hong Chen, Xin Wang, Yipeng Zhang, Yuwei Zhou, Zeyang Zhang, Siao Tang, and Wenwu Zhu. 2024. Disenstudio: Customized multi-subject text-to-video generation with disentangled spatial control. In Proceedings ofthe 32nd ACM International Conference on Multimedia. 3637–3646.

[5] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. 2025. Multi-subject Open-set Personalization in Video Generation. arXiv preprint arXiv:2501.06187 (2025).

[6] Yufan Deng, Xun Guo, Yizhi Wang, Jacob Zhiyuan Fang, Angtian Wang, Shenghai Yuan, Yiding Yang, Bo Liu, Haibin Huang, and Chongyang Ma. 2025. Cinema: Coherent multi-subject video generation via mllm-based guidance. arXiv preprint arXiv:2503.10391 (2025).

[7] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. 2024. Ltx-video: Realtime video latent difusion. arXiv preprint arXiv:2501.00103 (2024)

[8] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. 2022. Imagen video: High definition video generation with difusion models. arXiv preprint arXiv:2210.02303 (2022).

[9] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising difusion probabilistic models. Advances in Neural Information Processing Systems 33 (2020), 6840–6851.

[10] Yaosi Hu, Chong Luo, and Zhenzhong Chen. 2022. Make it move: controllable image-to-video generation with text descriptions. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18219–18228.

[11] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21807– 21818.

[12] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. 2023. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4015–4026.

[13] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024).

[14] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. 2024. Anyv2v: A tuning-free framework for any video-to-video editing tasks. arXiv preprint arXiv:2403.14468 (2024).

[15] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. 2024. Grounding dino: Marry ing dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision. Springer, 38–55.

[16] Shilin Lu, Yanzhu Liu, and Adams Wai-Kin Kong. 2023. Tf-icon: Difusion-based training-free cross-domain image composition. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2294–2305.

[17] Haomiao Ni, Bernhard Egger, Suhas Lohit, Anoop Cherian, Ye Wang, Toshiaki Koike-Akino, Sharon X Huang, and Tim K Marks. 2024. Ti2v-zero: Zero-shot image conditioning for text-to-video difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9015–9025.

[18] Haomiao Ni, Changhao Shi, Kai Li, Sharon X Huang, and Martin Renqiang Min. 2023. Conditional image-to-video generation with latent flow difusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18444–18455.

[19] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning. PMLR, 8748–8763.

[20] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. 2025. Sam 2: Segment anything in images and videos. In International Conference on Learning Representations, Vol. 2025. 28085–28128.

[21] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. 2024. Grounded SAM: Assembling

Open-World Models for Diverse Visual Tasks. arXiv:2401.14159 [cs.CV]

[22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10684–10695.

[23] Mayira Sharif, Guangzeng Han, Weisi Liu, and Xiaolei Huang. 2025. Cultivating Multidisciplinary Research and Education on GPU Infrastructure for Mid-South Institutions at the University of Memphis: Practice and Challenge. arXiv:2504.14786 [cs.DC] https://arxiv.org/abs/2504.14786

[24] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 (2020).

[25] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020. Score-based generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456 (2020).

[26] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. 2023. Plug-and-play difusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1921–1930.

[27] Wei Wang, Yaosen Chen, Yuegen Liu, Qi Yuan, Shubin Yang, and Yanru Zhang. 2024. MVOC: a training-free multiple video object composition method with difusion models. arXiv preprint arXiv:2406.15829 (2024).

[28] Zhao Wang, Aoxue Li, Lingting Zhu, Yong Guo, Qi Dou, and Zhenguo Li. 2024. Customvideo: Customizing text-to-video generation with multiple subjects. arXiv preprint arXiv:2401.09962 (2024).

[29] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. 2024. Dynamicrafter: Animating open-domain images with video difusion priors. In European Conference on Computer Vision. Springer, 399–417.

[30] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. Cogvideox: Text-to-video difusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024).

[31] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. 2023. I2vgen-xl: High-quality imageto-video synthesis via cascaded difusion models. arXiv preprint arXiv:2311.04145 (2023).

[32] Qi Zhao, Zhan Ma, and Pan Zhou. 2025. DreamInsert: Zero-Shot Image-to-Video Object Insertion from A Single Image. arXiv preprint arXiv:2503.10342 (2025).