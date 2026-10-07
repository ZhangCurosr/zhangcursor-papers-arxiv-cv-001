# ALIVE: INTERACTION-ALIGNED OBJECT INSERTION FOR FIRST-FRAME-GUIDED VIDEO EDITING

Zhenghong Zhou<sup>1†</sup> Zhe Lin<sup>2</sup> Jiebo Luo<sup>1,∗</sup> Yuqian Zhou<sup>2,∗</sup> <sup>1</sup>University of Rochester <sup>2</sup>Adobe Research <sup>∗</sup>Advising authors. Project page

![](images/ce94b3f00e4407c0b47ed8dee3b4150ba3d519e8478f711b2204e26c2752e1fe.jpg)

![](images/1e7eb2ccefd5167e37ebdfb4fdc37f3824fad7fadee211b1dadc7b6c2079e339.jpg)  
Figure 1: Bringing inserted objects to life. Each panel shows the source video, edited first frame and object-only instruction, followed by corresponding results from Señorita and ALIVE. ALIVE enables added objects to be picked up, worn, opened, or cut in response to source actions.

## ABSTRACT

Current video editors can insert objects but often struggle to make them participate in interactions such as being picked up or manipulated. We introduce ALIVE, a framework that makes inserted objects “alive” through coherent interactions with the source video’s contents, using an edited first frame and an instruction naming only the added object. We curate 35,800 editing pairs combining 3D-rendered, model-generated, and real-world videos with general editing pairs from ROSE. Each pair differs in the target object’s presence while preserving the surrounding action, teaching editors coordinated object behavior and source preservation. We further train a vision-language model (VLM) to predict interaction guidance from the same inputs. We introduce the ALIVE-interaction benchmark to assess interaction fidelity, source preservation, and visual coherence using a unified VLM-based protocol, and evaluate on the general video object insertion benchmark. Without VLM guidance, ALIVE improves Overall over the strongest evaluated baseline by 43.9% and 4.4% on the two benchmarks, respectively. VLM-predicted guidance further improves the ALIVE-interaction score by 0.95 points without additional user inputs.

## 1 INTRODUCTION

Recent advances in video generation and editing have enabled high-quality results across diverse tasks, including object insertion with realistic shadows and reflections (Liu et al., 2025b; Fu et al., 2026). However, making these objects respond coherently to source actions remains challenging. In Figure 1, an added mug should move with the hand that lifts it, and added dough should separate as a knife cuts through it. Our goal is to make inserted objects “alive”: not merely visible in the video, but part of its world, responding to the actions unfolding around them.

This capability poses two challenges: data construction and interaction modeling. Interaction editing pairs are scarce: their construction requires finding interaction videos and changing the target object’s presence while preserving the surrounding scene and action. For modeling, the source video provides action cues but does not directly show the added object’s response. The editor must infer the object’s evolving motion or state as these actions unfold.

ALIVE addresses these challenges, combining interaction-focused editing data with a visionlanguage model (VLM) that infers how added objects should respond to source actions. We curate 35,800 editing pairs combining 3D-rendered, model-generated, and real-world videos with general editing pairs from ROSE. These sources cover varied interactions and visual domains, supporting general object insertion. Quality checks exclude implausible interactions and unintended changes to the surrounding scene. We annotate multi-level interaction descriptions, including temporally localized prompts, to study which guidance benefits editing and supervise VLM prediction.

Using these data, we train a first-frame-guided diffusion editor (Ouyang et al., 2024; Liu et al., 2025b). Users provide a source video, an edited first frame, and an object-only instruction. The edited frame fixes the object’s initial appearance and placement, leaving subsequent behavior to follow source actions. We train the VLM to predict temporally localized interaction prompts from the same inputs, without user-provided interaction descriptions, per-frame masks, or trajectories.

We introduce the ALIVE-interaction benchmark to test whether added objects participate coherently in source actions while preserving the surrounding content. The general video object insertion benchmark further assesses broader editing quality. Without VLM guidance, ALIVE improves Overall scores over the strongest evaluated baseline by 43.9% and 4.4% on the two benchmarks, respectively. Predicted temporal prompt guidance further improves interaction performance. Ablations examine contributions from data composition and interaction guidance.

Our contributions are:

• A dataset of 35,800 video editing pairs combining interaction-focused and general insertion data, with complementary visual sources and multi-level prompt annotations.

• A diffusion editing framework that uses a VLM to predict interaction descriptions from the original inputs and aligns their guidance with the corresponding video intervals.

• We introduce the ALIVE-interaction benchmark and show that ALIVE outperforms baselines on interaction and general insertion, supported by ablations on data and guidance.

## 2 RELATED WORK

Video editing and paired video. EditVerse, UNIC, and VACE unify diverse video editing tasks through shared conditioning interfaces (Ju et al., 2026; Ye et al., 2025; Jiang et al., 2025). Goku-Edit and OpenVE-Edit support instruction-guided editing (Liang et al., 2026; He et al., 2025), while Kiwi-Edit unifies instruction and reference guidance (Lin et al., 2026). First-frame-guided methods propagate initial visual edits through source videos. AnyV2V and I2VEdit build on pretrained image-to-video models (Ku et al., 2024; Ouyang et al., 2024), while LoRA-Edit uses mask-aware adaptation (Gao et al., 2026a). Señorita, GenProp, and PropFly learn editing or propagation from paired or synthetic supervision (Zi et al., 2025; Liu et al., 2025b; Seo et al., 2026). PISCO and NovaEdit support sparse-keyframe-guided video editing (Gao et al., 2026b; Pan et al., 2026).

Paired datasets support these advances across tasks. Señorita-2M, Goku, and OpenVE cover broad editing tasks, while EffectErase and ROSE provide object removal / insertion pairs (Zi et al., 2025; Liang et al., 2026; He et al., 2025; Fu et al., 2026; Miao et al., 2025). ALIVE focuses on first-frameguided object insertion with coherent responses to source actions, supported by interaction-focused editing pairs and multi-level prompt annotations for studying interaction guidance.

Interactive video editing. Interactive editing spans user control in streaming workflows and physical interactions within scenes. EditStream, JoyAI-Video-Edit, StreamEdit, and Vidu S2-Editing support streaming editing (Zhou et al., 2026; Xiao et al., 2026; Jiao et al., 2026; Zhang et al., 2026). EgoPlay times edits via user-specified source events (Mai et al., 2026). EgoEdit supports instructionguided egocentric editing and real-time streaming (Li et al., 2026). Within scenes, VOID revises downstream physical interactions after object removal (Motamed et al., 2026), while DynaEdit edits actions and dynamics through text (Kulikov et al., 2026). We study first-frame-guided object insertion, where added objects respond to source actions without user-provided interaction descriptions.

VLM guidance for video editing. UniVideo and Omni-Video 2 integrate multimodal understanding with video generation and editing (Wei et al., 2026; Yang et al., 2026). DynVFX uses a VLM to describe a scene augmented with dynamic content (Yatim et al., 2025); Aurora plans edits and obtains missing conditions through tools (Yu et al., 2026). VOID identifies regions affected by object removal to guide counterfactual generation (Motamed et al., 2026). ALIVE trains a VLM to infer the inserted object’s possible responses from the source video, edited first frame, and object-only instruction, expressing them as temporally localized prompts guiding corresponding video intervals.

## 3 ALIVE DATASET AND BENCHMARK

Interaction-focused editing pairs are scarce, and evaluating object insertion requires assessing responses to source actions beyond appearance. We introduce ALIVE, comprising editing pairs from four complementary sources, multi-level prompt annotations for studying interaction guidance, and the ALIVE-interaction benchmark for evaluating interaction quality and source preservation.

## 3.1 INTERACTION PAIR CONSTRUCTION

Constructing these pairs requires finding suitable interaction videos and varying the target object’s presence while preserving the surrounding scene and action. We combine 3D-rendered videos, Model-generated videos, and Real-world videos for interaction examples across visual domains, supplemented by General editing pairs (ROSE) for insertion without object interaction. Each pair comprises an object-absent source video X and an object-present target video Y; its first frame provides the edited first frame E<sub>0</sub>. Figure 2 summarizes construction.

3D-rendered videos. To obtain precisely aligned editing pairs, we use 3D assets from SpaceTimePilot (Huang et al., 2025b; Lu et al., 2025). We manually select segments featuring object interactions and render each scene from multiple first- and third-person viewpoints. For each viewpoint, we render sequences with and without the target object while keeping the surrounding scene, camera trajectory, and animation aligned. These renders directly provide Y, X, and object masks.

Model-generated videos. To broaden interaction and scene coverage beyond available assets, we use GPT to design prompts covering common interaction types and synthesize object-present videos with Wan2.2 (Wan et al., 2025). SAM3 (Carion et al., 2026) segments and tracks the target object, and UnderEraser (Liu et al., 2026) removes it to construct the paired source video.

Real-world videos. For real-world visual supervision, we curate interaction clips from ARC-TIC (Fan et al., 2023), HOCap (Wang et al., 2025), HOI4D (Liu et al., 2022), and HOT3D (Banerjee et al., 2025). Target masks come from provided segmentations or projected object geometry, with refinement as needed. UnderEraser removes the target while retaining the surrounding action.

General editing pairs (ROSE). Objects should respond to interactions but also behave appropriately in their absence, for example by remaining stationary in the scene. We include general editing pairs from ROSE (Miao et al., 2025) to support this behavior, alongside effects such as shadows and reflections. We reverse its removal pairs: the object-absent result becomes X, and the original object-present video becomes Y.

Pair quality. We use Qwen3.6-27B (Qwen Team, 2026) and GPT-5.6 Luna/Sol for source-specific quality checks, excluding interaction videos with no visible object interaction or poor visual quality, and pairs with incomplete or visibly incorrect object removal.

Our 35,800 unique pairs comprise 14,362 3D-rendered, 5,896 model-generated, 8,542 real-world, and 7,000 ROSE general editing pairs. They provide precisely aligned pairs, common interactions, real-world examples, and cases without object interaction. Section 5.3.1 evaluates different sources.

![](images/da4efd992e88a20b68e1f475286a18b170a899ac395f2d1b4ea69c94f4314ada.jpg)  
Figure 2: ALIVE data construction and prompt annotation. Top: construction and quality filtering. Bottom: editing pairs from four sources (left) and multi-level prompts (right). P0 (no text) and P1 (object identity only) omit interaction descriptions; P2–P4 provide interaction guidance. Gold outlines mark edited first frames; P4 shading marks chunks with inclusive frame ranges.

## 3.2 MULTI-LEVEL INTERACTION ANNOTATION

We investigate whether prompts improve object interactions in video editing and which descriptions are most effective. We annotate P1–P4 with varying semantic detail and temporal structure, alongside P0 (no text). P1 names only the added object. Neither P0 nor P1 describes interactions, leaving the editor to infer plausible responses from the source video and $E _ { 0 }$ . P2 describes the principal interaction; P3 details its sequence and outcome; P4 describes temporal chunks. P2–P4 provide interaction information to the editor. P4 can span the full clip when no phase decomposition is needed. Figure 2 illustrates these differences. These annotations enable VLM-predicted interaction prompts from the source video, $E _ { 0 } ,$ , and P1, without user-specified interaction descriptions (Section 4.3).

Qwen3.6-27B assists annotation using ordered target-video frames, target masks, and available ob ject or action metadata, followed by selective GPT verification and refinement against the videos. Inconsistent descriptions are corrected or excluded at the affected level. For ROSE cases without object manipulation, prompts describe scene motion, occlusion, and visible object-related effects.

## 3.3 EVALUATION BENCHMARKS

ALIVE-interaction benchmark. We construct insertion tasks to assess coherent object responses to source actions. Each case provides an object-absent source video, an edited first frame, an objectpresent reference video, and P1–P4 annotations for prompt-level comparisons (Table 3), with P0 using no text. Its 128 cases include held-out samples from the 3D-rendered (10), model-generated (42), and real-world sources (19: ARCTIC 7, HOCap 1, HOI4D 5, and HOT3D 6), plus two outof-domain subsets: 31 HOIGen-1M (Liu et al., 2025a) cases and 26 videos we recorded (RealShot). General video object insertion benchmark. Alongside interaction-focused editing, we evaluate general insertion on 103 cases: 60 from ROSE and 43 from EffectErase. A common protocol evaluates editing fidelity, source preservation, and visual coherence on both benchmarks (Section 5.1).

## 4 METHOD

We present a first-frame-guided diffusion editor that learns inserted objects’ responses to source actions (Figure 3). Section 4.2 details its visual and prompt conditioning and training; Section 4.3 describes how a VLM uses visual understanding to infer P4 chunk prompts from the same inputs for temporal guidance.

![](images/d10f3c6c5e200041a1c95193d55b02575fe16bdd54dd31d395070a75db63388b.jpg)  
Figure 3: ALIVE overview. (a) Source and noisy target latents are channel-concatenated (C); the edited first-frame latent stays clean (gold). Flames mark editor and VLM LoRA fine-tuning. (b) A supervised VLM predicts P4 descriptions and intervals from the same visual inputs and P1. P4 texts are independently encoded. Attention maps contrast shared P0–P3 with time-local P4 conditioning.

## 4.1 TASK DEFINITION

Given a source video $X = ( x _ { 0 } , \dots , x _ { T - 1 } )$ , an edited first frame $E _ { 0 } ,$ and an identity-only insertion instruction $p _ { 1 }$ , we aim to generate an edited video in which the added object participates coherently in the source action while unrelated scene content is preserved. The edited frame specifies the object’s appearance and initial placement, $p _ { 1 }$ names the object, and X provides the action context. The editor $G _ { \theta }$ infers subsequent object interactions from these inputs:

$$
\hat { Y } = G _ { \theta } ( X , E _ { 0 } , p _ { 1 } ) .\tag{1}
$$

The complete target video Y provides training supervision but is unavailable at inference.

## 4.2 VIDEO EDITING MODEL

Visual conditioning. To propagate the initial insertion while preserving source actions, we condition the editor on $E _ { 0 }$ and the full source video. Our main implementation adapts LTX-2.5 (HaCohen et al., 2026). A frozen video encoder maps source–target pairs to aligned latent grids z<sub>X</sub> and $z _ { Y }$ . We concatenate clean source latents with noisy target latents $z _ { t }$ at matching spatiotemporal positions:

$$
h _ { t } = W _ { \mathrm { i n } } [ z _ { t } ; z _ { X } ] _ { \mathrm { c h a n n e l } } + b .\tag{2}
$$

This supplies source context without increasing the number of video tokens. We initialize the target half of $W _ { \mathrm { i n } }$ from the pretrained projection and the source half to zero. The encoded $E _ { 0 }$ anchors the first target frame; its conditioned tokens remain clean throughout denoising.

Prompt conditioning. The editor supports clip-level and temporally localized text conditioning. P0 uses an empty prompt while retaining the source video and $E _ { 0 } ^ { \mathrm { { \bar { \alpha } } } }$ . For P1–P3, the backbone’s standard text-conditioning path encodes the full prompt as one context shared across frames: P1 specifies object identity; P2 and P3 also describe interactions.

P4 aligns text guidance with source-action stages through descriptions $d _ { k }$ and temporal intervals $I _ { k }$ Each $d _ { k }$ specifies the added object and its interaction during $I _ { k }$ . We independently encode these descriptions into contexts $C _ { k }$ of length $L _ { k }$ and concatenate them. For video token i, let $w _ { i k }$ denote its normalized temporal overlap with $I _ { k }$ . Attention to text token $j$ in context k is

$$
A _ { i j } = \operatorname { s o f t m a x } _ { j } \left( { \frac { q _ { i } ^ { \top } k _ { j } } { \sqrt { d } } } + \log { \frac { w _ { i k } } { L _ { k } } } \right) , \qquad j \in C _ { k } ,\tag{3}
$$

where $q _ { i } , k _ { j }$ , and $d$ are the query, key, and key dimension. Zero overlap masks the context; dividing by $L _ { k }$ prevents longer descriptions from gaining prior attention mass solely through length. This routing modifies only video-to-text cross-attention; all video chunks are denoised jointly.

Training. We train a single editor with a mixture of prompt levels P0–P4 using LoRA (Hu et al., 2021) on ALIVE pairs. For each sampled level, we use pairs with valid annotations for that level. For unconditioned target tokens, the noisy latent at noise level t is $z _ { t } = ( 1 - t ) z _ { Y } + t \epsilon$ , with $\epsilon \sim \mathcal { N } ( 0 , I )$ . We optimize the native flow-matching objective (Lipman et al., 2022),

$$
\mathcal { L } _ { \mathrm { e d i t } } = \mathbb { E } _ { Y , X , p , t , \epsilon } \big [ \| v _ { \theta } ( z _ { t } , t ; z _ { X } , E _ { 0 } , p ) - ( \epsilon - z _ { Y } ) \| _ { M } ^ { 2 } \big ] ,\tag{4}
$$

where $v _ { \theta }$ is the editor’s velocity prediction with parameters $\theta , p$ is the sampled prompt, and $\| \cdot \| _ { M } ^ { 2 }$ averages squared error over target tokens not conditioned on $E _ { 0 }$ . For VLM-guided editing, we condition the editor on P4 prompts predicted from the original inputs, as described next.

## 4.3 VLM-BASED INTERACTION GUIDANCE

To guide object interactions over time, we use a VLM’s visual understanding to infer P4 chunk prompts from the source video, edited first frame, and object-only instruction. These prompts de scribe plausible interactions and their temporal intervals without additional interaction description.

We LoRA-adapt Qwen3.6-27B on the P4 annotations in Section 3.2. Given temporally ordered source frames, $E _ { 0 } ,$ and P1, denoted by $\boldsymbol { u } = ( X , E _ { 0 } , p _ { 1 } )$ , the predictor learns chunk descriptions and intervals through cross-entropy on response tokens only. Annotation may use the target video, but prediction uses only u during training and inference. At inference, the VLM $F _ { \phi }$ predicts P4 guidance $\hat { p } _ { 4 } .$ , replacing $p _ { 1 }$ as the editor’s text condition:

$$
\hat { p } _ { 4 } = F _ { \phi } ( X , E _ { 0 } , p _ { 1 } ) , \qquad \hat { Y } = G _ { \theta } ( X , E _ { 0 } , \hat { p } _ { 4 } ) .\tag{5}
$$

The routing in Section 4.2 applies each description to its temporal interval, guiding joint generation of the complete video.

## 5 EXPERIMENTS

We evaluate whether ALIVE enables inserted objects to participate in source actions while retaining general insertion quality. We then examine how training-data composition supports these capabilities and whether interaction prompts provide further improvements.

## 5.1 EXPERIMENTAL SETUP

We evaluate on the two benchmarks introduced in Section 3.3.

Compared methods. We compare AnyV2V, Señorita, NovaEdit, PropFly, and I2VEdit using their official conditioning interfaces and prompt formats. Both ALIVE variants share an LTX-2.5 model trained on our 35,800 pairs with mixed prompt levels: ALIVE uses object-identity text (P1), while ALIVE + VLM uses P4 predicted from the same source video, edited first frame, and P1. P2– P4 add interaction guidance beyond ${ \bf P } 1 \ ' \mathrm { s }$ object identity. Following their official prompt formats, AnyV2V and PropFly receive P2 descriptions, marked † for this additional guidance. We exclude instruction-only editors because text specifies initial object placement less precisely than $\begin{array} { r } { E _ { 0 } ; } \end{array}$ placement differences may prevent the intended interaction, confounding comparisons against methods sharing the edited first frame. We also exclude mask-guided VACE and LoRA-Edit configurations because per-frame masks provide object states and motion that our task requires the editor to infer. See Appendix A.1–A.3 for training, inference, and additional ablations.

Evaluation. A reference-aware VLM judge (gpt-5.6-sol, reasoning effort high) assesses interaction fidelity, source preservation, and visual coherence under a shared rubric across both benchmarks. It receives 16 uniformly sampled output frames, including both endpoints, with corresponding source and reference frames, $E _ { 0 } ,$ and the object-addition instruction. For I2VEdit, we evaluate all 14 output frames from its official implementation. Overall aggregates six dimensions on a 0–100 scale; Table 1 reports Overall, task fidelity, and motion/propagation. Beyond VLM scores, we follow EditVerse (Ju et al., 2026), reporting PickScore, frame-level CLIP and video-level ViCLIP alignment, and CLIP/DINOv2-based temporal consistency. Instructions, preprocessing alignment, and scoring details appear in Appendix A.4.

## 5.2 COMPARISON WITH EXISTING VIDEO EDITORS

ALIVE substantially improves interaction-focused insertion while retaining strong general insertion quality (Table 1). With object-identity text alone, it exceeds the strongest evaluated baseline by

Table 1: Comparison on interaction-focused and general object insertion. Both ALIVE variants share the LTX-2.5 (22B) editor; VLM P4 is predicted from source/E /P1. † marks interaction descriptions supplied. VLM scores use a 0–100 scale; automatic metrics retain native scales. Higher is better throughout; bold/underline denote best/second-best scores, including ties.
<table><tr><td></td><td></td><td colspan="3">VLM evaluation</td><td colspan="5">Automatic metrics</td></tr><tr><td>Method</td><td>Guidance</td><td>Overall</td><td>Task</td><td>Motion</td><td>PickScore</td><td>Frame CLIP</td><td>Video ViCLIP</td><td>TC-CLIP</td><td>TC-DINO</td></tr><tr><td colspan="10">ALIVE-interaction benchmark (128 cases)</td></tr><tr><td>Señorita</td><td>P1</td><td>60.24</td><td>53.52</td><td>36.52</td><td>19.95</td><td>24.04</td><td>19.09</td><td>0.9572</td><td>0.9494</td></tr><tr><td>NovaEdit</td><td>PO</td><td>36.36</td><td>8.59</td><td>0.39</td><td>18.94</td><td>21.15</td><td>15.41</td><td>0.9744</td><td>0.9734</td></tr><tr><td>I2VEdit</td><td>PO</td><td>56.55</td><td>48.44</td><td>33.79</td><td>19.68</td><td>23.48</td><td>18.61</td><td>0.9363</td><td>0.9132</td></tr><tr><td>AnyV2V</td><td>P2†</td><td>42.75</td><td>44.53</td><td>31.25</td><td>19.77</td><td>24.12</td><td>19.48</td><td>0.9141</td><td>0.8916</td></tr><tr><td>PropFly</td><td>P2†</td><td>42.31</td><td>35.94</td><td>21.68</td><td>19.40</td><td>22.81</td><td>17.58</td><td>0.9390</td><td>0.9274</td></tr><tr><td>ALIVE</td><td>P1</td><td>86.71</td><td>93.75</td><td>86.13</td><td>20.34</td><td>25.00</td><td>20.18</td><td>0.9795</td><td>0.9759</td></tr><tr><td>ALIVE + VLM</td><td>VLM P4</td><td>87.66</td><td>93.95</td><td>86.91</td><td>20.33</td><td>25.00</td><td>20.22</td><td>0.9791</td><td>0.9753</td></tr><tr><td colspan="10">general video object insertion benchmark (103 cases)</td></tr><tr><td>Señorita</td><td>P1</td><td>88.39</td><td>88.83</td><td>83.74</td><td>20.30</td><td>23.98</td><td>19.35</td><td>0.9713</td><td></td></tr><tr><td>NovaEdit</td><td>PO</td><td>45.66</td><td>27.18</td><td>19.66</td><td>19.49</td><td>21.49</td><td>16.74</td><td>0.9820</td><td>0.9723 0.9813</td></tr><tr><td>I2VEdit</td><td>PO</td><td>82.21</td><td>84.95</td><td>79.13</td><td>20.09</td><td>23.50</td><td>19.38</td><td>0.9556</td><td>0.9403</td></tr><tr><td>AnyV2V</td><td>P2†</td><td>48.03</td><td>56.07</td><td>47.09</td><td>20.09</td><td>23.71</td><td>20.15</td><td>0.9220</td><td>0.9020</td></tr><tr><td>PropFly</td><td>P2†</td><td>64.02</td><td>64.32</td><td>55.58</td><td>19.93</td><td>22.88</td><td>18.62</td><td>0.9603</td><td>0.9545</td></tr><tr><td>ALIVE</td><td>P1</td><td>92.28</td><td>94.17</td><td>89.32</td><td>20.37</td><td>23.74</td><td>19.47</td><td>0.9845</td><td>0.9835</td></tr><tr><td>ALIVE + VLM</td><td>VLM P4</td><td>91.71</td><td>93.20</td><td>88.35</td><td>20.38</td><td>23.85</td><td>19.63</td><td>0.9849</td><td>0.9849</td></tr></table>

26.46 points on the ALIVE-interaction benchmark (86.71 vs. 60.24) and by 3.89 points on general insertion (92.28 vs. 88.39). Its motion/propagation score on the interaction benchmark rises to 86.13, compared with 36.52 for Señorita, supporting more coherent coordination between inserted objects and source actions. VLM-predicted P4 further raises the interaction Overall score to 87.66. For general insertion, P1 remains stronger than predicted P4 (92.28 vs. 91.71), although both outperform the evaluated baselines. Predicted interaction guidance therefore builds on an already capable editor. Section 5.3.2 examines this contribution in detail.

Figure 4 compares book handling and basket carrying samples. ALIVE + VLM moves the inserted objects with the hands. In these examples, baselines can leave objects stationary, omit them, or distort their shape.

## 5.3 ABLATION STUDIES

## 5.3.1 DATA SOURCES

We examine how each data source contributes to interaction-focused and general object insertion, and whether combining them benefits both. We compare single-source training with a four-source mixture sampled in proportion to source size. All conditions use LTX-2.5, rank-128 LoRA, P1, and 5k training steps, with fixed evaluation cases and generation seeds. Table 2 lists source-pool sizes.

Different sources favor different capabilities: model-generated pairs perform best on interaction, while ROSE leads individual sources on general insertion. However, ROSE-only training scores just 60.12 on the ALIVE-interaction benchmark, compared with 82.35 for the mixture, highlighting the value of interaction-focused training pairs. The mixture remains within 0.46 points of modelgenerated-only training on interaction while improving general insertion by 7.61 points. Its highest general-insertion and pooled scores support the complementary value of our data sources.

## 5.3.2 PROMPT LEVELS AND VLM GUIDANCE

Having established the editor’s interaction capability, we test whether explicit descriptions of object behavior provide useful additional guidance. Table 3 compares annotated and VLM-predicted prompts while keeping editor weights, visual inputs, and generation seeds fixed. P0 supplies no text, P1 names the object, and annotated P2–P4 additionally describe interactions. VLM variants predict those descriptions from source $\mathrm { \Delta E _ { 0 } / P 1 }$

![](images/88b311a5b61dd896f3783be4d30ca7024883fa57a742b52453d060f03d23ae83.jpg)  
(a) Add the stack of hardcover books.  
(b) Add the laundry basket.  
Figure 4: Interaction-focused object insertion. Book handling (a) and basket carrying (b), each shown at the first frame, 40%, and 90% of the video. Square AnyV2V frames are displayed more narrowly. † marks provided interaction guidance, following the methods’ official prompt formats.

Table 2: Single-source and mixed-source training with the same 5k-step budget. All scores use our 16-frame evaluation (0–100; ↑). Pooled averages all 231 cases with equal weight: $( 1 2 8 S _ { \mathrm { i n t } } +$ 103 $S _ { \mathrm { g e n } } ) / 2 3 1$ , where $S _ { \mathrm { i n t } }$ and $S _ { \mathrm { g e n } }$ are the benchmark means. Bold and underline mark the best and second-best scores per column.
<table><tr><td rowspan="2">Training source</td><td rowspan="2">Pairs</td><td rowspan="2">Pooled</td><td colspan="3">ALIVE-interaction benchmark (N = 128)</td><td colspan="3">general video object insertion benchmark (N = 103)</td></tr><tr><td>Overall</td><td>Task</td><td>Motion</td><td>Overall</td><td>Task</td><td>Motion</td></tr><tr><td>3D-rendered videos</td><td>14,362</td><td>74.85</td><td>73.08</td><td>76.37</td><td>62.89</td><td>77.06</td><td>78.16</td><td>71.12</td></tr><tr><td>Real-world videos</td><td>8,542</td><td>81.51</td><td>80.19</td><td>88.48</td><td>79.49</td><td>83.16</td><td>83.50</td><td>80.10</td></tr><tr><td>Model-generated videos</td><td>5,896</td><td>81.26</td><td>82.81</td><td>90.82</td><td>82.23</td><td>79.32</td><td>79.85</td><td>71.84</td></tr><tr><td>General editing pairs (ROSE)</td><td>7,000</td><td>70.93</td><td>60.12</td><td>60.74</td><td>45.12</td><td>84.36</td><td>88.35</td><td>81.31</td></tr><tr><td>All four sources</td><td>35,800</td><td>84.39</td><td>82.35</td><td>89.65</td><td>82.03</td><td>86.93</td><td>88.83</td><td>83.01</td></tr></table>

Effects of prompt levels. P0 and P1 achieve Overall scores of 85.38 and 86.71, respectively, showing that the editor can infer object interactions from visual inputs. P2 provides little additional benefit, while the more detailed P3 and temporally localized P4 improve Overall to 87.37 and 88.69. These results suggest that the form of interaction guidance matters, with P4 performing best among the annotated formats.

VLM-predicted guidance. Predicted P4 achieves the highest Overall among the VLM variants, raising P1 from 86.71 to 87.66 using the original editing inputs. P4’s gains include temporal consistency and background preservation as well as motion/propagation. Thus, predicted P4 offers an improvement in overall interaction-editing quality.

Figure 5 connects the predicted descriptions to the resulting object behavior. P4 specifies “lower it onto the head” before adjusting the helmet straps under the chin, clarifying the placement sequence; its output avoids the strap artifacts around the eyes seen with P1. For the seedling, it describes lowering, releasing, and firming the surrounding soil; the generated plant is better seated in the soil as the hands press around it.

(a) P1: Add the bicycle helmet.  
![](images/d22557cc09cad3642621195687801d2833cf8bdd1fea7d7a8742f0e2303873b4.jpg)

(b) P1: Add the seedling.  
![](images/1a740ed66dad40dca26d16c74b446062fd745d21d7b675cc1f35be1f42374e8a.jpg)  
0–15 Add a seedling. Hands hold a seedling and lower it into a<sub>hole</sub> <sub>in</sub> <sub>the</sub> <sub>soil.</sub>  
<sub>16–39</sub> Add a seedling. Hands release the seedling and begin<sub>patting</sub> <sub>soil</sub> <sub>around</sub> <sub>the</sub> <sub>base.</sub>

<sub>40–80</sub> Add a seedling. Hands continue patting and firming the<sub>soil</sub> <sub>around</sub> <sub>the</sub> <sub>planted</sub> <sub>seedling.</sub>  
Figure 5: Effect of VLM-predicted interaction guidance. P1 and predicted P4 condition the same editor with fixed visual inputs. Predicted prompts and inclusive frame ranges appear below. Red boxes highlight strap artifacts around the eyes (a) and an exposed root ball (b) in the P1 results.  
Table 3: Prompt guidance on the ALIVE-interaction benchmark. All rows use the same editor, visual inputs, and case-specific seeds. Annotated P2–P4 supply interaction information; VLM variants infer it from source/E<sub>0</sub>/P1. P4 uses temporal chunks. Scores are 0–100 (↑); bold and underline mark first and second place. ID: identity; BG: background; Temp.: temporal consistency.
<table><tr><td>Guidance</td><td>N</td><td>Task</td><td>ID Motion</td><td></td><td>BG</td><td>Temp.</td><td>Visual</td><td>Overall</td></tr><tr><td>PO</td><td>128</td><td>91.41</td><td>81.84</td><td>83.40</td><td>93.36</td><td>80.47</td><td>75.00</td><td>85.38</td></tr><tr><td>P1</td><td>128</td><td>93.75</td><td>83.98</td><td>86.13</td><td>91.80</td><td>81.05</td><td>75.20</td><td>86.71</td></tr><tr><td>P2</td><td>128</td><td>93.16</td><td>82.23</td><td>86.91</td><td>92.38</td><td>80.66</td><td>77.15</td><td>86.68</td></tr><tr><td>P3</td><td>128</td><td>94.73</td><td>83.40</td><td>88.48</td><td>91.60</td><td>81.25</td><td>75.59</td><td>87.37</td></tr><tr><td>P4</td><td>128</td><td>95.12</td><td>85.94</td><td>88.67</td><td>93.55</td><td>82.81</td><td>78.32</td><td>88.69</td></tr><tr><td colspan="9">VLM-enhanced P1</td></tr><tr><td>VLM P2</td><td>128</td><td>93.55</td><td>84.57</td><td>87.89</td><td>91.80</td><td>81.84</td><td>76.37</td><td>87.33</td></tr><tr><td>VLM P3</td><td>128</td><td>94.34</td><td>83.20</td><td>86.33</td><td>92.19</td><td>81.84</td><td>77.34</td><td>87.17</td></tr><tr><td>VLM P4</td><td>128</td><td>93.95</td><td>84.57</td><td>86.91</td><td>93.36</td><td>82.81</td><td>76.76</td><td>87.66</td></tr></table>

Limitations and future work. Despite these improvements, ALIVE faces limitations. The source video and edited first frame do not fully specify geometry, contact conditions, or forces, leaving multiple plausible object responses. For example, a soft object may deform differently under pressure depending on stiffness, while a partially occluded grasp may allow a cup to remain upright or tilt during lifting. ALIVE aims to generate plausible object responses consistent with source actions, without implausible deformation, interpenetration, or unsupported floating. Additionally, VLMpredicted guidance errors can cause incorrect behavior, and the editor still struggles with complex cloth deformations and rapid, large-angle rotations. Future work includes improving guidance reliability, strengthening the editor’s ability to generate complex interactions, and incorporating explicit physical information for greater accuracy and control. Combining ALIVE with Self Forcing (Huang et al., 2025a) or EditStream (Zhou et al., 2026) could enable real-time streaming object insertion.

## 6 CONCLUSION

We presented ALIVE, a framework for inserting objects that participate coherently in a source video’s interactions. Our dataset combines complementary construction routes, and our benchmarks assess both interaction-focused and general object insertion. Experiments show the benefit of interaction-focused editor training and clarify the effects of data composition and prompt guidance. VLM-predicted guidance further improves interaction editing without additional user input, with P4 performing best among the evaluated prediction formats.

## ACKNOWLEDGMENTS

The implementation and experiments reported in this paper were carried out by Zhenghong Zhou at the University of Rochester. This work was partially supported by the university’s Goergen Institute for Data Science and Artificial Intelligence. We gratefully acknowledge use of the research computing resources of the Empire AI Consortium, Inc. (Bloom et al., 2025), with support from Empire State Development of the State of New York, the Simons Foundation, and the Secunda Family Foundation.

## REFERENCES

Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang Zhang, Jade Fountain, Edward Miller, Selen Basol, et al. Hot3d: Hand and object tracking in 3d from egocentric multi-view videos. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7061–7071. IEEE, 2025.

Stacie Bloom, Joshua C. Brumberg, Ian Fisk, Robert J. Harrison, Robert Hull, Melur Ramasubramanian, Krystyn Van Vliet, and Jeannette Wing. Empire AI: A new model for provisioning AI and HPC for academic research in the public good. In Practice and Experience in Advanced Research Computing (PEARC ’25), page 4, Columbus, OH, USA, July 2025. ACM. doi: 10.1145/3708035.3736070. URL https://doi.org/10.1145/3708035.3736070.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International conference on learning representations, volume 2026, pages 138846–138923, 2026.

Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J Black, and Otmar Hilliges. Arctic: A dataset for dexterous bimanual hand-object manipulation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12943– 12954. IEEE, 2023.

Yang Fu, Yike Zheng, Ziyun Dai, and Henghui Ding. Effecterase: Joint video object removal and insertion for high-quality effect erasing. arXiv preprint arXiv:2603.19224, 2026.

Chenjian Gao, Lihe Ding, Cai Cai, Zhanpeng Huang, Zibin Wang, and Tianfan Xue. Controllable first-frame-guided video editing via mask-aware lora fine-tuning. In International Conference on Learning Representations, volume 2026, pages 61741–61765, 2026a.

Xiangbo Gao, Renjie Li, Xinghao Chen, Yuheng Wu, Suofei Feng, Qing Yin, and Zhengzhong Tu. Pisco: Precise video instance insertion with sparse control. arXiv preprint arXiv:2602.08277, 2026b.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

Haoyang He, Jie Wang, Jiangning Zhang, Zhucun Xue, Xingyuan Bu, Qiangpeng Yang, Shilei Wen, and Lei Xie. Openve-3m: A large-scale high-quality dataset for instruction-guided video editing. arXiv preprint arXiv:2512.07826, 2025.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025a.

Zhening Huang, Hyeonho Jeong, Xuelin Chen, Yulia Gryaditskaya, Tuanfeng Y Wang, Joan Lasenby, and Chun-Hao Huang. Spacetimepilot: Generative rendering of dynamic scenes across space and time. arXiv preprint arXiv:2512.25075, 2025b.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 17191–17202. IEEE, 2025.

Guanlong Jiao, Chenyangguang Zhang, Jia Jun Cheng Xian, Zewei Zhang, and Renjie Liao. Streamedit: Training-free video editing via few-step streaming video generation. In European Conference on Computer Vision, pages 1–20. Springer, 2026.

Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Cherry Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, et al. Editverse: Unifying image and video editing and generation with in-context learning. In International Conference on Learning Representations, volume 2026, pages 137234–137255, 2026.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. Anyv2v: A tuning-free framework for any video-to-video editing tasks. arXiv preprint arXiv:2403.14468, 2024.

Vladimir Kulikov, Roni Paiss, Andrey Voynov, Inbar Mosseri, Tali Dekel, and Tomer Michaeli. Versatile editing of video content, actions, and dynamics without training. In European Conference on Computer Vision, pages 448–466. Springer, 2026.

Runjia Li, Moayed Haji-Ali, Ashkan Mirzaei, Chaoyang Wang, Arpit Sahni, Ivan Skorokhodov, Aliaksandr Siarohin, Tomas Jakab, Junlin Han, Sergey Tulyakov, et al. Egoedit: Dataset, real-time streaming model, and benchmark for egocentric video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16042–16053, 2026.

Sen Liang, Cong Wang, Zhentao Yu, Fengbin Guan, Zhengguang Zhou, Teng Hu, Youliang Zhang, Yuan Zhou, Xin Li, Qinglin Lu, et al. Goku: A million-scale universal dataset and benchmark for instruction-based video editing. arXiv preprint arXiv:2606.30599, 2026.

Yiqi Lin, Guoqiang Liang, Ziyun Zeng, Zechen Bai, Yanzhe Chen, and Mike Zheng Shou. Kiwi-edit: Versatile video editing via instruction and reference guidance. arXiv preprint arXiv:2603.02175, 2026.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Dingming Liu, Wenjing Wang, Chen Li, Jing Lyu, and Haotian Dong. From understanding to erasing: Towards complete and stable video object removal. arXiv preprint arXiv:2604.01693, 2026.

Kun Liu, Qi Liu, Xinchen Liu, Jie Li, Yongdong Zhang, Jiebo Luo, Xiaodong He, and Wu Liu. Hoigen-1m: A large-scale dataset for human-object interaction video generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24001– 24010. IEEE, 2025a.

Shaoteng Liu, Tianyu Wang, Jui-Hsien Wang, Qing Liu, Zhifei Zhang, Joon-Young Lee, Yijun Li, Bei Yu, Zhe Lin, Soo Ye Kim, et al. Generative video propagation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17712–17722. IEEE, 2025b.

Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. Hoi4d: A 4d egocentric dataset for category-level human-object interaction. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20981–20990. IEEE, 2022.

Jiaxin Lu, Chun-Hao Paul Huang, Uttaran Bhattacharya, Qixing Huang, and Yi Zhou. Humoto: A 4d dataset of mocap human object interactions. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10886–10897. IEEE, 2025.

Jinjie Mai, Gordon Guocheng Qian, Willi Menapace, Arpit Sahni, Chaoyang Wang, Ashkan Mirzaei, Runjia Li, Sergey Tulyakov, Bernard Ghanem, Peter Wonka, et al. Egoplay: Eventtriggered video editing for egocentric streams. arXiv preprint arXiv:2607.24560, 2026.

Chenxuan Miao, Yutong Feng, Jianshu Zeng, Zixiang Gao, Hantang Liu, Yunfeng Yan, Donglian Qi, Xi Chen, Bin Wang, and Hengshuang Zhao. Rose: Remove objects with side effects in videos. arXiv preprint arXiv:2508.18633, 2025.

Saman Motamed, William Harvey, Benjamin Klein, Luc Van Gool, Zhuoning Yuan, and Ta-Ying Cheng. Void: Video object and interaction deletion. In European Conference on Computer Vision, pages 245–261. Springer, 2026.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. I2vedit: First-frame-guided video editing via image-to-video diffusion models. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.

Tianlin Pan, Jiayi Dai, Chenpu Yuan, Zhengyao Lv, Binxin Yang, Hubery Yin, Chen Li, Jing Lyu, Caifeng Shan, and Chenyang Si. Nova: Sparse control, dense synthesis for pair-free video editing. arXiv preprint arXiv:2603.02802, 2026.

Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27b dense model, April 2026. URL https: //qwen.ai/blog?id=qwen3.6-27b.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Wonyong Seo, Jaeho Moon, Jaehyup Lee, Soo Ye Kim, and Munchurl Kim. Propfly: Learning to propagate via on-the-fly supervision from pre-trained video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 43228–43238, 2026.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Jikai Wang, Qifan Zhang, Yu-Wei Chao, Bowen Wen, Xiaohu Guo, and Yu Xiang. HO-cap: A capture system and dataset for 3d reconstruction and pose tracking of hand-object interaction. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. URL https://openreview.net/forum?id=hpu6r8oLw9.

Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In International Conference on Learning Representations, volume 2024, pages 42055–42079, 2024.

Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. In International Conference on Learning Representations, volume 2026, pages 113905–113933, 2026.

Yicheng Xiao, Wenxun Dai, Xinran Qin, Lin Song, Maoquan Zhang, Hang Xu, Yukang Chen, Yitong Li, Guohui Zhang, Yuan Zhang, et al. Joyai-video-edit: Real-time open-ended video editing with autoregressive diffusion. arXiv preprint arXiv:2608.03974, 2026.

Hao Yang, Zhiyu Tan, Jia Gong, Luozheng Qin, Hesen Chen, Xiaomeng Yang, Yuqing Sun, Yuetan Lin, Mengping Yang, and Hao Li. Omni-video 2: Scaling mllm-conditioned diffusion for unified video generation and editing. arXiv preprint arXiv:2602.08820, 2026.

Danah Yatim, Rafail Fridman, Omer Bar-Tal, and Tali Dekel. Dynvfx: Augmenting real videos with dynamic content. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–12, 2025.

Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216, 2025.

Yongsheng Yu, Ziyun Zeng, Zhiyuan Xiao, Zhenghong Zhou, Hang Hua, Wei Xiong, and Jiebo Luo. Aurora: Unified video editing with a tool-using agent. arXiv preprint arXiv:2605.18748, 2026.

Jintao Zhang, Kai Jiang, Jintao Chen, Xu Wang, Deyuan Liu, Jungang Li, Dechuang Chen, Ming Lin, Jingjiang Zhou, Haopeng Jin, et al. Vidu s2: Real-time interactive, editable, and spatial video generation. arXiv preprint arXiv:2609.11638, 2026.

Yuqian Zhou, Zhenghong Zhou, Zongze Wu, Cameron Smith, Richard Zhang, Jiebo Luo, Eli Shechtman, and Zhe Lin. Editstream: A unified autoregressive framework for interactive video generation and editing. arXiv preprint arXiv:2608.21424, 2026.

Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and Kam-Fai Wong. Señorita-2m: A high-quality instruction-based dataset for general video editing by video specialists. In NeurIPS D&B, 2025.

## A APPENDIX

## A.1 TRAINING DETAILS

Diffusion editor training. We adapt LTX-2.5 with rank-128 LoRA on 35,800 pairs, using 81-frame clips at $8 3 2 \times 4 8 0$ . Training uses AdamW with an initial learning rate of $1 0 ^ { - 4 }$ and linear decay, a global batch size of 16, and 20,000 updates. We use 16 GPUs per run (A100 80GB, H100, or H200), with one sample per GPU. Training first selects P0–P4 with probabilities 0.1/0.3/0.2/0.2/0.2, then samples a video pair with the corresponding prompt annotation (no text for P0). Data-source ablations use P1 and 5,000 updates with the same batch size, evaluation cases, and generation seeds. Both ALIVE variants share the main editor; at inference, they generate 81 frames at $7 6 8 \times 5 1 2$ with 30 denoising steps.

VLM training. We adapt Qwen3.6-27B with rank-64 LoRA $( \alpha = 1 2 8$ , dropout 0.05), training separate predictors for P2, P3, and P4. Each uses 16 A100 80GB GPUs, one sample per GPU, and two-step gradient accumulation, giving a global batch size of 32. We use AdamW with learning rate $5 \times 1 0 ^ { - 5 }$ , weight decay 0.01, and 32 warm-up updates followed by a constant learning rate. Inputs comprise 21 uniformly sampled source frames, $E _ { 0 }$ , and P1. We use the 2,048-update checkpoint for each predictor.

## A.2 INFERENCE SETTINGS AND EFFICIENCY

Table 4 profiles the same six videos (three per benchmark) using one A100 80GB per video, with batch size one and a separate warm-up video. We report mean wall-clock time and the largest peak allocated GPU memory across the six cases. Timing includes conditioning, required inversion or per-video adaptation, sampling, decoding, and export; offline data preparation and one-time setup are excluded. Costs reflect each method’s native output configuration and the memory-management settings specified below.

Table 4: Measured inference cost on A100 80GB. Output is width × height × frames; only denoising iterations are counted under Steps. Time includes each editor’s required inversion and adaptation. Peak denotes PyTorch peak allocated memory in GiB. VLM prompt prediction is measured separately in the last row; dashes mean not applicable.
<table><tr><td>Method</td><td>Output</td><td>Steps</td><td>Time (s)</td><td>Peak (GiB)</td></tr><tr><td>Señorita</td><td> $7 6 8 \times 4 4 8 \times 3 3$ </td><td>30</td><td>109.18</td><td>28.04</td></tr><tr><td>NovaEdit</td><td> $8 3 2 \times 4 8 0 \times 8 1$ </td><td>50</td><td>407.70</td><td>19.63</td></tr><tr><td>I2VEdit</td><td> $1 0 2 4 \times 5 7 6 \times 1 4$ </td><td>25</td><td>863.62</td><td>65.16</td></tr><tr><td>AnyV2V</td><td> $5 1 2 \times 5 1 2 \times 1 6$ </td><td>50</td><td>172.79</td><td>30.31</td></tr><tr><td>PropFly</td><td> $8 3 2 \times 4 8 0 \times 2 5$ </td><td>50</td><td>71.63</td><td>21.14</td></tr><tr><td>Wan 14B, P1</td><td> $7 6 8 \times 5 1 2 \times 8 1$ </td><td>30</td><td>304.64</td><td>64.46</td></tr><tr><td>LTX 22B, P1</td><td> $7 6 8 \times 5 1 2 \times 8 1$ </td><td>30</td><td>154.16</td><td>39.43</td></tr><tr><td>LTX 22B, predicted P4</td><td> $7 6 8 \times 5 1 2 \times 8 1$ </td><td>30</td><td>184.50</td><td>39.43</td></tr><tr><td>VLM P4 predictor</td><td>一</td><td>一</td><td>22.75</td><td>52.27</td></tr></table>

AnyV2V uses 500 inversion steps; I2VEdit includes inversion, 250 motion-LoRA updates, and native per-video model loading. Señorita and PropFly run without CPU model offloading. Wan retains both diffusion experts on GPU while staging its text encoder and VAE; LTX uses its native no-offload setting. The P4 editor uses the fixed predicted prompts from the quality evaluation. Separately, the 2,048-update VLM is profiled on the same editing inputs. Summing these separately measured stages gives an estimated 207.25 s per edit, excluding the overhead of switching models.

## A.3 ADDITIONAL ABLATIONS

## A.3.1 TRAINING LENGTH

We compare independently trained P1-only LTX-2.5 editors with 5k and 20k training steps on the same 35,800 pairs (Table 5). Evaluation cases and generation seeds are fixed, following the proto-

col in Section 5.1. The longer budget raises interaction Overall from 82.35 to 86.46 and general insertion Overall from 86.93 to 89.20.

Table 5: Effect of training budget on P1-only LTX-2.5 editors (LoRA rank 128, batch 16). Scores report Overall (↑) on both benchmarks.
<table><tr><td>Optimizer steps</td><td>Pairs</td><td>benchmark</td><td>ALIVE-interaction general video object insertion benchmark</td></tr><tr><td></td><td></td><td>82.35</td><td>86.93</td></tr><tr><td>5k 20k</td><td>35,800 35,800</td><td>86.46</td><td>89.20</td></tr></table>

## A.3.2 BASE MODELS AND MODEL SIZE

We compare Wan and LTX editors trained exclusively with P1 for 5k steps on the same 35,800 pairs (Table 6). Both benchmarks use the evaluation protocol in Section 5.1.

Table 6: Backbone comparison after 5k steps of P1-only training. Scores report Overall (↑) on both benchmarks; bold and underline mark the best and second-best scores.
<table><tr><td colspan="2"></td><td rowspan="2">ALIVE-interaction benchmark</td><td rowspan="2">general video object insertion benchmark</td></tr><tr><td>Backbone</td><td>Steps</td></tr><tr><td>Wan 2.2 TI2V (5B)</td><td>5k</td><td>71.85</td><td>81.54</td></tr><tr><td>Wan 2.2 I2V (14B)</td><td>5k</td><td>89.82</td><td>89.44</td></tr><tr><td>LTX-2.5 (22B)</td><td>5k</td><td>82.35</td><td>86.93</td></tr></table>

Training efficiency. For 81-frame clips at $8 3 2 \times 4 8 0 .$ , LoRA rank 128, and batch 16 per model or expert, LTX-2.5 takes 4.95 s per update on 16 A100 80GB GPUs; its 5k-step run takes 8.1 hours including checkpoint writes. Wan 14B takes approximately 45.4 s per expert update on 32 A100 80GB GPUs (16 per noise expert), implying approximately 63 hours for 5k updates at this rate. Update times are steady-state medians; the Wan duration is an extrapolation, excluding setup, checkpoint overhead, and interruptions. LTX’s stronger VAE compression $( 8 \times 3 2 \times 3 2$ versus $4 \times 8 \times 8$ in time, height, and width) reduces the video-token sequence length. This supports its practical efficiency. Wan 14B achieves higher scores in the 5k-step comparison, while LTX-2.5 trains substantially faster and serves as our main experimental backbone.

## A.4 EVALUATION PROTOCOL

We assess interaction fidelity, source preservation, and visual coherence using the VLM judge introduced in Section 5.1. Comparisons within each table use matched cases and generation seeds. Each candidate is evaluated independently with the same rubric and dimension weights across both benchmarks and all prompt levels. We reuse judgments for identical videos and evaluation inputs, retrying only failed requests without resampling successful scores.

Evaluation inputs. For each case, the judge receives source, reference, and candidate frames, the edited first frame $E _ { 0 } ,$ , and an object-addition instruction shared across methods (Table 7). The ALIVE-interaction benchmark uses P1; the general video object insertion benchmark uses “Add the object or person shown in the edited first frame.” Reference frames beyond $E _ { 0 }$ are provided only to the judge, not the editor.

Frame sampling and alignment. We align source and reference evidence to each generator’s input preprocessing so that evaluation reflects the field of view and time span available during generation. We uniformly sample 16 native output frames, including both endpoints, and map them to the source frames selected by the generator. For I2VEdit, we use all 14 native frames without repetition or interpolation. Spatial alignment follows the generator’s documented aspect-preserving cover resize and fixed center crop, including interpolation and rounding conventions.

At each timepoint, SOURCE, REFERENCE, and CANDIDATE appear side by side with their aspect ratios preserved. Four contact sheets show 5, 5, 5, and 1 timepoints; I2VEdit uses three sheets with 5, 5, and 4. The aligned $E _ { 0 }$ is supplied separately as a PNG. Each labeled cell measures $3 8 4 \times 2 4 0$ pixels, and sheets use JPEG quality 88. Time labels indicate normalized positions rather than seconds. Layout margins, labels, and gaps are excluded from scoring; artifacts within video regions remain scoreable.

Table 7: Inputs to the evaluation judge. The same roles and presentation are used for all benchmarks and methods. Only CANDIDATE is scored.
<table><tr><td>Input</td><td>Role</td></tr><tr><td>Basic instruction</td><td>Identify the intended addition without prescribing an action sequence.</td></tr><tr><td>SOURCE</td><td>Establish non-target scene content, camera motion, and interaction context.</td></tr><tr><td>Edited first frame  $E _ { 0 }$ </td><td>Specify the intended object or person and initial edit.</td></tr><tr><td>REFERENCE</td><td>Provide identity, contact, occlusion, interaction, and natural-effects evidence.</td></tr><tr><td>CANDIDATE</td><td>Generated result to evaluate at every displayed timepoint.</td></tr></table>

Scoring and aggregation. Table 8 defines six dimensions and their weights. For candidate i and dimension $d ,$ the judge assigns an integer $s _ { i , d } \in \{ 0 , 1 , 2 , 3 , 4 \}$ : 0 denotes failure or contradictory evidence, 2 partial correctness with a material problem, and 4 full satisfaction supported by visible evidence. Scores 1 and 3 fall between these anchors. We assess plausible interactions consistent with SOURCE and $E _ { 0 }$ , allowing reasonable spatial and temporal differences from the reference (Figure 6). Missing required interactions, broken contact, impossible motion, and source corruption remain errors.

Table 8: Evaluation dimensions. Weights are identical across benchmarks and ablations.
<table><tr><td>Dimension</td><td>Criterion</td><td>Weight</td></tr><tr><td>Task fidelity</td><td>Complete addition and coherent participation in interactions sup- ported by SOURCE and  $E _ { 0 }$ </td><td>0.25</td></tr><tr><td>Object identity</td><td>Stable identity, geometry, and attributes consistent with  $E _ { 0 }$ </td><td>0.15</td></tr><tr><td>Motion / propagation</td><td>Plausible motion or appropriate stability, contact, timing, and occlu- sion.</td><td>0.20</td></tr><tr><td>Background preserva- tion</td><td>Faithful non-target source content and camera motion.</td><td>0.15</td></tr><tr><td>Temporal consistency</td><td>Stable appearance and smooth transitions without flicker or disap- pearance.</td><td>0.15</td></tr><tr><td>Visual quality</td><td>Realistic boundaries, lighting, and natural effects.</td><td>0.10</td></tr></table>

We aggregate scores as

$$
\mathrm { O v e r a l l } _ { i } = 2 5 \sum _ { d = 1 } ^ { 6 } w _ { d } s _ { i , d } , \qquad \mathrm { O v e r a l l } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { O v e r a l l } _ { i } , \qquad \sum _ { d } w _ { d } = 1 .\tag{6}
$$

Here, $N$ is the number of evaluated cases and $w _ { d }$ the weight of dimension $d .$ Dimension scores are also multiplied by 25 for reporting. The judge returns structured scores and explanations; its confidence indicates uncertainty but does not affect aggregation.

Complete judge prompt. Figure 6 reproduces the full task prompt supplied with the visual evidence. The placeholder <basic instruction> is replaced by the case’s P1 instruction on the ALIVE-interaction benchmark, or “Add the object or person shown in the edited first frame.” on the general video object insertion benchmark. For I2VEdit, only “The first four images show sixteen normalized-time samples” changes to “The first three images show fourteen normalized-time samples”; all scoring instructions remain identical.

![](images/d89b37252d04d217b7a1bed8d53d69a05021644c031a8f22f959719a46270d09.jpg)  
Figure 6: Complete evaluation task prompt. Wording is reproduced verbatim with line wrapping normalized. The case instruction replaces <basic instruction>; the I2VEdit sampling exception and structured response requirements are specified in the text.