# GameVerse: A Minute-Scale Gameplay Dataset for Long-Horizon Interactive World Modeling

Kang He<sup>1,2,3,†</sup>, Wenshuo Peng<sup>4</sup>, Chuanhao Li<sup>1</sup>, Zihui Gao<sup>1</sup>, Xiaojie Xu<sup>1</sup>, Zhengyuan Lin<sup>1</sup>, Yuzhe Ding<sup>3</sup>, Donghong Ji<sup>3</sup>, Kaipeng Zhang<sup>1,2,∗,‡</sup>, Yongtao Ge<sup>1,∗,‡</sup> 1Alava Lab 2Shanghai Innovation Institute Alaya Lab Shanghai Innovation Institute

<sup>3</sup>School of Cyber Science and Engineering, Wuhan University <sup>4</sup>Tsinghua University hekang0225@whu.edu.cn, kaipeng.zhang@shanda.com

<sup></sup> Project Page § GitHub Datasets

![](images/28aa02a8e017cbcb90a82111f1821fada3c032320c5fd3119443fec3bdb2acfa.jpg)

![](images/aab8e561921f8246207c27653ec65b71ef7a32aa0af89034893cfb70313863bc.jpg)  
(a) Original Video  
(b) Scene Clips

![](images/3e0262dcf3d5cec96568319790b810e88407d7c503819a5417c4f2b63bcfce57.jpg)  
(c) Quality Pipeline

![](images/71a3aa5b717dc83ad7bc63e7e6c5deb6f40384dce5db6807d26a1c31ad12d766.jpg)  
(d) 2min clips

![](images/ddbf60f73f775457e98d3e35f49de03b8e7d0346c5e5c3983c7f7f346156ff18.jpg)  
(e) Annotation

![](images/37d0cd23ec21e196827321abb923447b3bf3ff20690ecc8f71fd027be2524485.jpg)  
(I) Clip Duration

![](images/ba88375ac9909daec37d6862420155b3c1b260e46826d080185d7ffa49e37b8e.jpg)  
(II) Game Genre

![](images/fa7c388fc5f95914069b2947a3e56bf4a581e5da5a673aa47f3be2f437891c86.jpg)  
(III) Action diversity  
Figure 1: Overview of the proposed GameVerse dataset and construction pipeline. The top portion (a)–(e) illustrates the construction pipeline, with panel (e.2) detailing the four-level structured caption schema (global static, local static, dynamic events, agent actions). The bottom panels present statistical analyses of GameVerse: (I) clip-duration distribution, (II) game-category composition, and (III) interaction diversity across gameplay behaviors.

## Abstract

Recent advances in interactive world models have made it possible to generate controllable, dynamic environments from action inputs. Yet current models struggle to maintain coherence beyond roughly one minute of rollout—a limitation rooted not in model capacity, but in the scarcity of long-horizon training data. Existing gameplay video datasets typically consist of clips under ten seconds, with captions

that describe whole clips in coarse and undifferentiated text, which is insufficient for controllable long-horizon generation. We address this with GameVerse, a minute-scale gameplay video dataset purpose-built for long-horizon interactive world modeling. GameVerse contains 69.7K clips spanning 1,655 hours and 349.6 million frames across diverse first- and third-person gameplay, including 1,358 hours of continuous two-minute sequences—an order of magnitude much longer than prior datasets. Clips pass through a unified pipeline of scene-consistent segmentation, quality filtering, camera pose estimation, and event-based caption annotation. Each clip is paired with a temporally stable camera trajectory that remains coherent across minutes, together with captions that separate static scene context from dynamic foreground events and align to event boundaries rather than fixed time windows. Comprehensive analysis shows that GameVerse surpasses existing public datasets in scale, diversity, and annotation quality. Experiments further demonstrate that training on GameVerse substantially extends the temporal horizon of interactive world models.

## 1 Introduction

Interactive world models [1, 28, 31] aim to learn controllable, dynamic environments that can be rolled out from action and camera inputs, a capability central to embodied decision-making, simulation, and game-style content generation. Recent gameplay-oriented systems such as GameGen-X [2], Matrix-Game [34], Hunyuan-GameCraft [16], and HY-World 1.5 [13] demonstrate compelling short-horizon controllability, yet they share a striking ceiling: rollouts collapse into drift, identity loss, and scene incoherence after roughly one minute of generation. We argue that this ceiling is, first and foremost, a data problem rather than a model-capacity problem. The training corpora behind these systems consist almost entirely of clips shorter than ten seconds, paired with whole-clip captions that conflate static context with dynamic events. Such data fundamentally cannot supervise minute-scale temporal coherence, fine-grained controllability, or event-level semantics, no matter how large the model.

This limitation is visible across recent gameplay and exploration datasets, which exhibit a consistent pattern of single-axis optimization (Table 1). Sekai [18, 8] provides large-scale exploration video but lacks dense interactions, validated camera trajectories, and structured semantics. Matrix-Game and HY-World 1.5 expose engine-level camera poses but cap continuous footage at short windows. GameGen-X and Hunyuan-GameCraft support action-conditioned interaction but lack reliable trajectories and event-aligned captions. None of them simultaneously provides minute-scale continuous footage, validated frame-level camera trajectories, dense first- and third-person interaction coverage, and multi-granular event-aligned semantic annotations. So long as these four signals remain fragmented across separate datasets, any model trained on a single source inherits its blind spot.

We close this gap with GameVerse, a minute-scale gameplay dataset purpose-built for long-horizon interactive world modeling. Curated from 11,002 source videos at 1080p and 30–60 FPS through a unified pipeline of scene-consistent segmentation, multi-stage quality filtering, ViPE-based [11] pose estimation with trajectory validation, and event-aligned multi-granular captioning, GameVerse delivers 69.7K clips, 1,655 hours, and 349.6M frames, including 1,358 hours of continuous twominute sequences—roughly an order of magnitude longer than any prior gameplay or exploration dataset. Every clip is paired with a temporally stable frame-level camera trajectory that survives explicit jump-, jitter-, and drift-filtering; a four-level structured caption (global static, local static, dynamic events, agent actions) that disentangles scene context from foreground change; and dense behavior coverage across first- and third-person perspectives, spanning exploration, combat, walking, driving, climbing, and complex object interaction. To our knowledge, GameVerse is the first dataset to ship these four signals jointly under a single quality bar, and fine-tuning a strong video generator on it improves VBench [12] temporal-consistency and visual quality, and WorldScore [4] content alignment and 3D consistency, over gameplay-domain baselines on a 30-second benchmark.

Our main contributions are summarized as follows:

• The largest minute-scale gameplay dataset to date. 69.7K clips, 1,655 hours, and 349.6M frames, including 1,358 hours of continuous two-minute sequences—roughly an order of magnitude longer than existing gameplay corpora and the first to directly supervise the minute-scale rollout regime where current interactive world models fail.

Table 1: Comparison with recent world modeling datasets. Long-duration refers to minute-level continuous video sequences. Long-horizon denotes the presence of temporally coherent state evolution and interaction dynamics over extended time. Pose indicates the availability of publicly released, reliable camera trajectories (∼ marks systems that obtain camera poses internally but do not release them as supervision). Multi-gran. denotes multi-level semantic decomposition of scene and behavior. ∼ indicates partial support or not explicitly specified.
<table><tr><td></td><td></td><td>Sekai [18] GameGen-X [2]</td><td></td><td></td><td>WorldMem [31] Matrix-Game [34, 9, 22] Hunyuan-GameCraft [16] HY-World [13] Ours</td><td></td><td></td></tr><tr><td>Long-duration</td><td>2</td><td>x</td><td>x</td><td>2</td><td>x</td><td>2</td><td></td></tr><tr><td>Long-horizon</td><td>x</td><td>2</td><td>x</td><td>√</td><td>x</td><td>√</td><td></td></tr><tr><td>Pose</td><td>~</td><td>x</td><td>2</td><td>2</td><td>x</td><td>2</td><td>V</td></tr><tr><td>Multi-gran.</td><td>x</td><td>2</td><td>x</td><td>x</td><td>x</td><td>x</td><td></td></tr><tr><td>Interaction</td><td>x</td><td>V</td><td>x</td><td>√</td><td>√</td><td>√</td><td></td></tr></table>

• A unified construction pipeline with reliable camera trajectories. Scene-consistent segmentation that combines hard-cut detection with CLIP-based gradual-transition detection, multi-stage visual / content / temporal quality filtering, and ViPE-based pose estimation with explicit jump-, jitter-, and drift-filtering—delivering temporally stable trajectories at minute scale, where most prior datasets either omit poses or expose noisy ones.

• A multi-granular event-aligned caption schema. A four-level structured annotation (global static, local static, dynamic events, agent actions) that disentangles environment context from controllable behavior and supports hierarchical conditioning—replacing the single-line whole-clip captions that dominate prior gameplay data.

• Empirical validation on long-horizon generation. Training on GameVerse improves VBench temporal-consistency and visual-quality metrics and most WorldScore controllability metrics over gameplay-domain baselines on a held-out long-duration benchmark.

## 2 Related Work

## 2.1 Interactive World Models and Gameplay Datasets

Early work on interactive video generation established the methodological foundation but relied on closed, narrow training corpora. Genie [1] learns latent actions from an unreleased, curated 30K-hour platformer collection, leaving the community without a reusable substrate. GameNGen [28] and Oasis [3] demonstrate that diffusion models can serve as real-time game engines, yet rely on a single game (e.g.DOOM) or a closed Decart corpus, limiting cross-game generalization and broader scientific study. WorldMem [31] introduces an explicit memory module for long-term Minecraft simulation, showing that long-horizon coherence is methodologically tractable, but remains limited to a narrow evaluation domain without a transferable, large-scale gameplay dataset.

More recent gameplay-oriented systems advance controllability with action and camera signals [25, 33], but fall short on the long-horizon data axis. GameGen-X [2] is trained predominantly on gameplay clips of a few seconds, providing little supervision for minute-scale rollouts. The Matrix-Game family—the Minecraft-centered Matrix-Game [34] and Matrix-Game 2.0 [9], and the recent Matrix-Game 3.0 [22] targeting long-horizon coherence via memory-augmented diffusion on an industrial-scale Unreal-Engine and AAA-game data engine—releases model weights and code but keeps its gameplay training corpora and Video-Pose-Action-Prompt quadruplet data proprietary, leaving no shared substrate for scaling, ablation, and reproducibility. Our work complements Matrix-Game 3.0: while it addresses the long-horizon problem from the model side through memory on a closed data engine, GameVerse addresses it from the data side with an openly released, multi-game, minute-scale corpus that any model can consume. Hunyuan-GameCraft [16] and HY-World 1.5 [13] report strong qualitative results across diverse environments but rely on unreleased internal gameplay recordings; WorldPlay [24] pursues real-time long-horizon generation under the same proprietary-data constraint, while VMem [17] explores surfel-indexed memory but remains bound to a method-specific corpus. Across this landscape, the lack of a public, diverse, multi-game, minute-scale gameplay dataset remains the principal bottleneck, rather than model capacity.

## 2.2 Camera-Aware Video Supervision

Explicit camera control is a key axis of controllable video generation [32]. On the method side, CameraCtrl [7] and MotionCtrl [30] demonstrate that trajectory conditioning materially improves geometric consistency in generated video, but both rely on the data side to provide reliable trajectories for training. On the data side, ViPE [11] provides scalable monocular pose estimation from unconstrained video and is the de-facto tool for retrofitting trajectories onto large video corpora; we leverage it as a tool in our pipeline rather than as a competing dataset. Among existing datasets, Matrix-Game [34, 9, 22], Hunyuan-GameCraft [16], and HY-World 1.5 [13] obtain engine-level camera poses in their closed training pipelines, which are never released as supervision; Sekai [18, 8] ships estimated trajectories alongside long-duration walking video but does not validate them against sudden jumps, high-frequency jitter, or long-term drift, leaving its trajectories noisy in practice at the minute scale; and the bulk of general-purpose video-language datasets ship no calibrated poses at all. GameVerse instead pairs every clip with a frame-level ViPE trajectory that has passed explicit jump-, jitter-, and drift-filtering, delivering minute-scale trajectories that remain physically coherent across the full clip, and we release these trajectories openly together with the video sources.

## 2.3 Structured Caption Annotations for Video

Caption supervision drives controllable generation, but existing video datasets stop short of the structure required for long-horizon and behavior-conditioned synthesis. HowTo100M [20] relies on ASR transcripts that are loosely aligned and dominated by narration rather than scene description. InternVid [29] and ActivityNet-style dense-captioning data [15, 10] contribute auto-generated or sparse event captions but treat each clip as a flat sequence of events, with no separation of static scene context from dynamic foreground and no isolation of agent behavior from environmental change. MiraData [14] pushes toward longer-duration video with longer structured captions, but its semantic schema is tuned to real-world video and does not include an agent-action vocabulary suitable for interactive generation. On the gameplay side, GameGen-X [2] provides single-line clip-level captions that conflate scene context with dynamic events, and Sekai [18] similarly attaches one whole-clip description to each minute-long video; in either case the controllable channel cannot be modulated independently of the scene. Matrix-Game [34, 9, 22], Hunyuan-GameCraft [16], and HY-World 1.5 [13] do not publicly release caption supervision at all. GameVerse instead annotates every clip with a four-level hierarchical schema—global static, local static, dynamic events, and agent actions—generated by a vision-language model [26] and aligned to event boundaries rather than fixed time windows, so that environment context is disentangled from controllable agent behavior and can be used directly as a hierarchical conditioning signal.

## 3 Dataset Construction Pipeline

Existing video datasets for world modeling still exhibit limitations along three axes: fragmented temporal structure caused by short clip segmentation, unstable or absent camera pose estimation, and flat semantic annotations that conflate static context with dynamic events. To address these limitations, we design a unified data construction pipeline that takes a curated pool of long-form gameplay videos and transforms it into structured, minute-scale data units, each consisting of an aligned video segment, a validated camera trajectory, and a multi-granular caption set. The pipeline begins with a data collection step (Section 3.1) that builds the raw pool, followed by four processing stages: scene-consistent segmentation, multi-stage video quality evaluation (illustrated in Figure 2), camera pose estimation and validation, and multi-granular structured captioning. Each stage targets a specific limitation of prior gameplay datasets.

## 3.1 Data Collection

Constructing a long-horizon gameplay corpus at the scale GameVerse targets requires a source that is simultaneously diverse—across games, perspectives, and behaviors—and openly releasable to the community. Engine recordings used by Genie [1], the Matrix-Game family [34, 9, 22], Hunyuan-GameCraft [16], and HY-World 1.5 [13] satisfy diversity within a single title but are proprietary and typically centered on one or two games. We instead build on YouTube, which is, to our knowledge, the only open source that simultaneously offers long-form gameplay recordings across many AAA and indie titles, both first- and third-person perspectives, and a wide range of player behaviors. All videos are retained at the highest available quality, up to 1080p at 30–60 FPS.

![](images/ff36abeb148f5242a99c1c0d4cea4c9d6053d3dc049a9a9641ea9f1a9ef777e2.jpg)  
Figure 2: Overview of the proposed multi-stage video quality evaluation pipeline.

Videos are collected on a per-title basis from a curated set of open-world AAA and indie games chosen to together span both first- and third-person viewpoints. Because the chosen titles already exhibit a wide range of player activities and environments, the resulting source pool naturally covers the eight behavior categories reported in Section 4 (exploring, combat, walking, running, driving, interacting, climbing, and idle) together with diverse environments under varied weather, time-of-day, season, and crowd-density conditions. The collected corpus consists of 11,002 source videos, totalling 4,692 hours (195.5 days) and approximately 986.6 M frames, which the four processing stages below convert into the final 188K structured clips.

## 3.2 Scene-consistent Segmentation

Existing datasets typically rely on short clips of a few seconds, which disrupt temporal continuity and limit long-horizon modeling. We start from long-form videos and explicitly construct minute-level continuous segments by removing scene transitions. We first employ TransNetV2 [23] to detect abrupt scene transitions (hard cuts) based on frame-level transition probabilities. To capture gradual transitions (e.g., camera panning or UI state changes) that are difficult to capture by hard-cut detection alone, we additionally compute a CLIP-based cosine semantic distance between sampled frames, treating large jumps in semantic distance as smooth-transition boundaries. Combining hard-cut detection with semantic-change signals, we extract scene-consistent segments of up to 120 seconds, which can be further concatenated into longer sequences. After segmentation, we obtain 272,128 clips totalling 4,661 hours, among which 120-second clips contribute approximately 2,773 hours.

## 3.3 Multi-stage Video Quality Evaluation

Even after scene-consistent segmentation, raw gameplay videos may still contain subtitles, watermarks, UI overlays, compression artifacts, duplicated content, and unstable temporal dynamics, all of which can introduce distracting visual signals and inconsistent supervision for world modeling. To ensure both visual fidelity and temporal coherence, we design a multi-stage video quality evaluation pipeline (Figure 2) that progressively performs (i) basic filtering to remove low-resolution or corrupted videos; (ii) content and quality assessment using OCR-based detection to filter subtitles, HUD elements, and other overlays; (iii) stability and UI detection to identify unstable motion patterns and temporal artifacts; (iv) deduplication and diversity selection to eliminate highly similar samples while preserving diverse gameplay scenarios; and (v) temporal refinement using temporal-quality metrics such as COVER [6] and VBench [12] consistency scores. These complementary stages jointly address frame-level visual degradation, temporal inconsistencies, and dataset redundancy. The overall quality of each clip is computed as a weighted sum of its per-stage scores, while stage-specific constraints prevent severe quality defects from being compensated for by high scores in other dimensions. Additionally, we recover a subset of previously filtered clips where the camera remains stationary while scene content continues to evolve, preserving meaningful object and agent dynamics under fixed viewpoints for world-model training. Only clips passing the corresponding quality checks are retained. Detailed thresholds and per-stage criteria are provided in Appendix B.

![](images/687dfe6168415b094b1630c4c5d9aa70a5db03d01506f8686e74730fbf9e7914.jpg)  
(a) Clip-duration Interval

![](images/4a9ea9e84112d9d7c1717dc056fb7b87d82dc5c2526f3d57c4f99ed603dab4b7.jpg)  
(b) Hierarchical Scene Attributes  
Figure 3: Statistical analysis of GameVerse. (a) Distribution of clip durations and total video hours across different intervals. (b) Hierarchical distribution of scene attributes across six dimensions.

## 3.4 Camera Pose Estimation and Validation

Most existing datasets lack reliable camera pose information. Even when available, pose annotations are often noisy or unstable, limiting their usefulness for geometric modeling. To address this, we apply ViPE [11] to estimate frame-wise camera poses for each video segment, obtaining both intrinsic and extrinsic parameters.

However, pose estimation in dynamic scenes is inherently challenging and often suffers from noise and tracking failures. To ensure trajectory reliability, we validate every estimated pose against three common error modes: sudden jumps (large non-physical pose changes between consecutive frames, typically from tracking failures or relocalisation errors), high-frequency jitter (local instability producing unnatural oscillations), and long-term drift (accumulated errors over time). We score each clip with a unified trajectory-quality function combining local relative-pose increments (capturing jumps and jitter) and a global trajectory smoothness term (capturing drift), and apply threshold-based filtering to remove unreliable samples.

## 3.5 Multi-granular Structured Captioning

Semantic annotations in prior gameplay datasets, such as GameGen-X [2] and Hunyuan-GameCraft [16], typically take the form of single-level captions or simple action labels that mix static scene context with dynamic behavior. Even structured annotations often provide limited separation across temporal and semantic scales.

To capture fine-grained gameplay semantics, we introduce a multi-granular structured captioning framework for GameVerse using the vision-language model Kimi-K2.5 [26]. As illustrated in Figure 1(e.2), the framework separates static context from dynamic processes and organizes each clip’s semantics into four complementary components: global static, clip-level environment and scene context (terrain, weather, lighting, and viewpoint); local static, relatively stable scene configurations within each temporal segment (functional areas, landmarks, and visible objects); dynamic events, changes in the surrounding world (crowd movement, traffic, and environmental transitions); and agent actions, the main subject’s behavior, described using a unified action vocabulary that separates action semantics from viewpoint. Together, these components provide explicit semantic supervision at multiple granularities, supporting hierarchical and conditional generation. Annotation details and examples are provided in Appendix E.

## 4 Dataset Statistics

After the construction pipeline of Section 3.1, GameVerse contains 69.7K high-quality gameplay clips totalling 1,655 hours, including 1,358 hours of two-minute continuous video. We analyse the dataset along four axes that mirror the contribution claims of Section 1: long-horizon temporal scale (Section 4.1), interaction and environment diversity (Section 4.2), camera-trajectory diversity (Section 4.3), and caption annotation statistics (Section 4.4).

![](images/6e6f3f76a190a770e69edc7c8545eb75725badddd6bd4b8f77855460bd24d7f6.jpg)  
(a) Pose Motion Distribution

![](images/9c5f5956cde8874dc4e0de0510af6fa40eaf6a36547c9efe8abad2ccb015acfe.jpg)  
(b) Video Caption Length  
Figure 4: Analysis of camera motion and semantic annotations in GameVerse. (a) Distribution of camera motion patterns, including static, rotational, translational, and complex motions. (b) Distribution of caption lengths across different semantic levels, including global static context, local static structure, dynamic events, and agent actions.

## 4.1 Scale and Long-Horizon Distribution

Figure 3(a) shows the clip-duration distribution of GameVerse. Unlike prior gameplay video datasets such as GameGen-X [2], Matrix-Game [34, 9, 22], and Hunyuan-GameCraft [16], which are dominated by clips shorter than ten seconds, GameVerse preserves substantially longer temporal continuity. Over 40.6K clips reach the 120-second upper bound and together contribute approximately 1,358 hours of continuous gameplay video, an order of magnitude longer than the continuous footage exposed in any prior gameplay corpus that we are aware of. Clips of 60 s or longer account for a further substantial portion of the dataset, providing the kind of minute-scale supervision that is essentially absent from existing gameplay corpora and is required to extend interactive world-model rollouts beyond the one-minute coherence ceiling discussed in Section 1.

## 4.2 Interaction and Environment Diversity

GameVerse spans at least 110 distinct game titles, with the genre composition shown in panel (II) of Figure 1. The distribution is long-tailed: the most-represented titles—God of War, Red Dead Redemption 2, Ghost of Tsushima, Elden Ring, and The Witcher 3—each contribute over 1,800 clips, while the tail extends across dozens of additional AAA and indie titles. Across the corpus, approximately 24% of clips are first-person, 73% are third-person follow camera, and 3% use specialised perspectives (drone, vehicle-mounted, gimbal-follow, stationary). This 1:3 first-to-thirdperson ratio supplies substantial supervision for both FPV and TPV settings, in sharp contrast to the Minecraft-only Matrix-Game [34, 9] and to Sekai [18], which is restricted to first-person walking footage. Figure 1 also presents the distribution of player-behavior categories—exploring, combat, walking, driving, interacting, climbing, running, and idle—spanning basic locomotion to complex object interaction and high-speed vehicle control; exploring, combat, walking, and driving together contribute the largest share of total video duration. Figure 3(b) further shows the hierarchical scene-attribute distribution: the corpus spans urban, rural, forest, highway, temple, and industrial environments under varied architectural styles, weather conditions, time-of-day, seasons, and crowd-density levels.

## 4.3 Camera Trajectory Diversity

Figure 4(a) shows the distribution of camera-motion patterns in GameVerse. The corpus covers static, rotational, translational, and complex motions. Forward motion accounts for the largest share (34.7%), reflecting the prevalence of continuous navigation and exploration in gameplay. Complex motions that couple translation with rotation contribute a further 22.0%; these clips are particularly challenging for long-horizon world modeling because they require simultaneous geometric and identity consistency under non-trivial viewpoint change. Lateral movement, orbit motion, and mixed rotational patterns are also well represented, indicating substantial diversity along the trajectory axis. By contrast, real-world walking-video datasets such as Sekai are typically dominated by forward translation with limited rotational complexity, and gameplay datasets that obtain engine-level poses are restricted to a single title’s controller scheme. The richer camera dynamics in GameVerse therefore enable supervision for geometry-aware generation, camera-controllable video synthesis, and interactive world modeling under genuine viewpoint change.

## 4.4 Caption Annotation Statistics

Figure 4(b) shows the caption-length distributions across the four semantic levels introduced in Section 3.5. Local static descriptions and dynamic events are substantially longer, with median lengths of 895 and 775 characters respectively, reflecting the rich environmental structure and dynamic scene changes in gameplay video. Global static and agent action captions are more concise: global-static captions describe high-level scene context, and agent actions are drawn from a constrained vocabulary so that the controllable channel can be modulated cleanly during conditional generation. Compared to the single-line whole-clip captions of Sekai and the clip-level captions of GameGen-X, the four-level schema produces substantially more total semantic information per clip while remaining structured enough to act as a hierarchical conditioning signal.

![](images/09cb2c7ecb9603a2aa611a6478585fb474159f98069e7d86d886173fdcee406b.jpg)  
Figure 5: Word-cloud visualization of the four semantic caption components in GameVerse.

Figure 5 shows that the four levels concentrate on largely disjoint vocabularies—scene attributes, spatial layouts, environment transitions, and explicit motion—confirming that the schema produces semantically separated annotations rather than overlapping rephrasings of the same clip.

## 5 Experiments

## 5.1 Experimental Setup

We evaluate 60-second image-to-video generation on the same 103 inputs using LingBot [5], HY-World 1.5 [13, 24], HY-World-FT, and LTX-FT. HY-World-FT is obtained by fine-tuning HY-World 1.5 on approximately 600 hours of GameVerse for 3,000 steps, using the step-3,000 checkpoint without changing the architecture. LTX-FT is a GameVerse-adapted LTX-Video 2.3 model [19]. The fixed-backbone comparison measures the benefit of additional GameVerse fine-tuning under a shared evaluation protocol; comparisons across model families characterize complete systems with different architectures and control interfaces.

## 5.2 Evaluation Metrics

We evaluate generation performance from two complementary perspectives. Video generation quality is assessed using VBench [12] across six dimensions: Subject Consistency, Background Consistency, Motion Smoothness, Aesthetic Quality, Imaging Quality, and Overall Consistency, collectively measuring temporal coherence, motion stability, visual fidelity, and text–video alignment. World generation quality and controllability are evaluated using WorldScore [4], covering camera and object control, content alignment, 3D consistency, photometric consistency, style consistency, and subjective quality. We report all seven normalized WorldScore components alongside their arithmetic mean (WS-Static). Higher scores indicate better performance across all reported metrics. Detailed definitions and evaluation protocols are provided in Appendix C.1.

## 5.3 Minute-Scale Generation Performance

VBench Evaluation. As shown in Table 2, LTX-FT achieves the best temporal consistency across 60-second rollouts, leading in Subject Consistency (79.17), Background Consistency (90.63), and Motion Smoothness (99.53). Meanwhile, LingBot performs better on perceptual quality and text– video alignment, achieving the highest Aesthetic Quality (6.14), Imaging Quality (56.70), and Overall Consistency (18.25). Notably, under the same HY-World 1.5 backbone, fine-tuning on GameVerse consistently improves all six VBench metrics. In particular, Subject Consistency increases from

Table 2: Overall VBench evaluation results on the full evaluation set. ↑ indicates higher is better. Bold indicates the best reported value in each column.
<table><tr><td>Model</td><td>Subj. Cons.↑</td><td>Bkg. Cons.↑</td><td>Mot. Smth.↑</td><td>Aes. Qual.↑</td><td>Img. Qual.↑</td><td>Overall Cons.↑</td></tr><tr><td>LingBot</td><td>74.30</td><td>88.94</td><td>97.36</td><td>6.14</td><td>56.70</td><td>18.25</td></tr><tr><td>HY-World 1.5</td><td>66.92</td><td>81.77</td><td>96.26</td><td>4.29</td><td>48.99</td><td>14.48</td></tr><tr><td>HY-World-FT</td><td>68.45</td><td>83.30</td><td>96.81</td><td>4.65</td><td>49.34</td><td>15.69</td></tr><tr><td>LTX-FT</td><td>79.17</td><td>90.63</td><td>99.53</td><td>5.79</td><td>49.36</td><td>17.57</td></tr></table>

Table 3: Overall WorldScore evaluation results on the full evaluation set. ↑ indicates higher is better. Best results are highlighted in bold.
<table><tr><td>Model</td><td>WS-Static↑</td><td>Cam. Ctrl.↑</td><td>Obj. Ctrl.↑</td><td>Content↑</td><td>3D Cons.↑</td><td>Photo.↑</td><td>Style↑</td><td>Subj. Qual.↑</td></tr><tr><td>LingBot</td><td>46.07</td><td>31.69</td><td>45.56</td><td>50.18</td><td>63.25</td><td>22.71</td><td>86.06</td><td>23.05</td></tr><tr><td>HY-World 1.5</td><td>53.33</td><td>28.58</td><td>45.54</td><td>48.16</td><td>66.94</td><td>83.15</td><td>87.52</td><td>13.42</td></tr><tr><td>HY-World-FT</td><td>55.13</td><td>28.90</td><td>46.26</td><td>49.59</td><td>73.42</td><td>84.08</td><td>88.11</td><td>15.58</td></tr><tr><td>LTX-FT</td><td>59.65</td><td>24.33</td><td>44.93</td><td>48.47</td><td>95.13</td><td>98.45</td><td>90.04</td><td>16.22</td></tr></table>

66.92 to 68.45, Background Consistency from 81.77 to 83.30, and Overall Consistency from 14.48 to 15.69, demonstrating the effectiveness of GameVerse in enhancing temporal coherence and generation quality without architectural modifications.

WorldScore Evaluation. Table 3 evaluates world generation quality and controllability. LTX-FT achieves the highest WS-Static score (59.65), leading in 3D Consistency (95.13), Photometric Consistency (98.45), and Style Consistency (90.04). LingBot performs best in Camera Control, Content Alignment, and Subjective Quality, while HY-World-FT leads in Object Control. LingBot’s stronger camera control may partly benefit from ground-truth pose supervision and dedicated conditioning mechanisms, whereas GameVerse relies on ViPE-estimated trajectories (see Appendix C.2). Importantly, fine-tuning the same HY-World backbone (Hy-World-FT) on GameVerse improves all seven WorldScore components, increasing WS-Static from 53.33 to 55.13, 3D Consistency from 66.94 to 73.42 (+6.48), and Camera Control from 28.58 to 28.90. These results demonstrate the effectiveness of GameVerse in improving geometric consistency and controllability for minute-scale generation.

## 5.4 Long-Horizon Consistency Analysis

Reference-frame Similarity. To further assess long-horizon stability, we sample generated frames at 2 Hz and compute their CLIP ViT-B/32 [21] similarity to the input frame. As shown in Table 4(a), LTX-FT achieves the highest similarity at 30 seconds (0.785) and the latest reported threshold crossing (50.7s). Notably, under the same HY-World backbone, GameVerse fine-tuning improves the 30-second similarity from 0.609 to 0.642 and delays the first similarity drop below 0.6 from 30.4s to 37.9s. These results suggest improved reference preservation and reduced visual drift over extended rollouts, although viewpoint changes may also affect similarity.

Performance across Gameplay Activities. Table 4(b) further evaluates Subject Consistency across four gameplay activities. LTX-FT achieves the highest scores across driving, riding, running, and walking. Under the same backbone, HY-World-FT consistently outperforms HY-World 1.5, with improvements of 1.15, 1.85, 2.49, and 1.58 points, respectively. These consistent gains across diverse gameplay activities further support the effectiveness of GameVerse for long-horizon generation.

## 5.5 Annotation Quality Validation

Camera trajectories. We assess trajectory completeness and temporal stability after quality filtering, and compare ViPE estimates with an independent DROID-SLAM estimator [27]. After Sim(3) alignment, rotation RPE ranges from 0.75 to 1.47 degrees and translation-direction error from 4.84 to 11.03 degrees across six motion categories. The audited retained trajectories have complete valid poses; mixed rotation exhibits the largest cross-estimator errors. These measurements characterize consistency between two estimators, rather than absolute pose accuracy against ground truth. Full motion-wise results are provided in Appendix C.3.

Table 4: Temporal stability and activity-wise Subject Consistency of 60s generations. ↑ indicates higher is better. Best results are highlighted in bold.  
(a) Reference-frame Similarity
<table><tr><td>Model</td><td>First-drop (s)↑ CLIP@30s↑</td></tr><tr><td>LingBot</td><td>49.0 0.680</td></tr><tr><td>HY-World 1.5</td><td>30.4 0.609</td></tr><tr><td>HY-World-FT</td><td>37.9 0.642</td></tr><tr><td>LTX-FT</td><td>50.7 0.785</td></tr></table>

(b) Subject Consistency
<table><tr><td>Model</td><td>Driving↑</td><td>Riding↑</td><td>Running↑</td><td>Walking↑</td></tr><tr><td>LingBot</td><td>73.23</td><td>76.37</td><td>74.01</td><td>73.69</td></tr><tr><td>HY-World 1.5</td><td>65.02</td><td>63.47</td><td>66.05</td><td>71.48</td></tr><tr><td>HY-World-FT</td><td>66.17</td><td>65.32</td><td>68.54</td><td>73.06</td></tr><tr><td>LTX-FT</td><td>81.44</td><td>79.13</td><td>76.16</td><td>80.74</td></tr></table>

Structured Captions. To assess the quality of our multigranular semantic annotations, we conduct a video-grounded human evaluation on 400 clips stratified by game, viewpoint, motion type, and duration. Three independent annotators evaluate the accuracy and temporal alignment of each caption against its corresponding video. As shown in Table 5, the annotations achieve high scores in scene/event accuracy (4.82/5) and agent-action accuracy (4.66/5), while temporal alignment (4.61/5) and camera-motion accuracy (4.55/5) also show favorable results. The overall alignment score reaches 4.64/5, with consistent judgments among annotators (Krippendorff’s $\alpha = 0 . 7 6 )$ . Furthermore, the clip-level hallucination and temporal-mismatch rates are limited to 5.5% and 4.0%, respectively. These results demonstrate the effectiveness of our structured captioning pipeline in capturing fine-grained gameplay semantics, while highlighting remaining challenges in annotation accuracy and temporal alignment.

Table 5: Human evaluation of structured captions on 400 clips.
<table><tr><td>Human-audit Measure</td><td>Result</td></tr><tr><td>Scene/event accuracy</td><td>4.82 / 5</td></tr><tr><td>Agent-action accuracy</td><td>4.66 / 5</td></tr><tr><td>Temporal alignment</td><td>4.61 / 5</td></tr><tr><td>Camera-motion accuracy</td><td>4.55 / 5</td></tr><tr><td>Overall alignment</td><td>4.64 / 5</td></tr><tr><td>Clip-level hallucination</td><td>5.5%</td></tr><tr><td>Temporal mismatch</td><td>4.0%</td></tr><tr><td>Krippendorff&#x27;s α</td><td>0.76</td></tr></table>

Case Study. We provide representative gameplay visualizations in Appendix D and multi-granular structured annotation examples in Appendix E, illustrating the diversity of environments, continuous scene evolution, and fine-grained semantic supervision in GameVerse. We further present qualitative generation results of LTX-Video 2.3 fine-tuned on GameVerse in Appendix F. Conditioned on an initial frame, camera trajectories, and structured semantic captions, the model generates temporally coherent gameplay sequences across diverse scenarios, including outdoor exploration, urban naviga tion, horseback riding, and nighttime driving. The results exhibit consistent scene structures, smooth camera-aware motion, and coherent environmental evolution. Additional camera-controlled examples demonstrate the model’s ability to follow complex trajectories while maintaining spatial and temporal consistency, highlighting the potential of GameVerse for long-horizon controllable world generation.

## 6 Conclusion and Limitations

We present GameVerse, a minute-scale gameplay dataset for long-horizon interactive world modeling, comprising 69.7K clips totaling 1,655 hours, including 1,358 hours of continuous two-minute gameplay. Built through a unified pipeline, GameVerse provides quality-filtered frame-level camera trajectories and multi-granular structured captions, jointly capturing long-term scene evolution, camera motion, and fine-grained interaction dynamics. Experiments on 60-second generation demonstrate improvements in temporal coherence, geometric consistency, and visual quality, with same-backbone comparisons further validating the benefits of GameVerse fine-tuning. Additional annotation-quality assessments support the reliability of its semantic and geometric supervision. We envision GameVerse as a valuable resource for advancing long-horizon interactive world modeling and enabling more coherent, controllable, and persistent virtual worlds.

Limitations. GameVerse focuses on gameplay environments, with cross-domain generalization to real-world scenarios remaining unexplored. Despite multi-stage validation, estimated camera trajectories and generated captions may retain residual errors. Continuous clips are limited to two minutes, while generation experiments cover up to 60 seconds, leaving longer rollouts and closedloop interaction for future work. Moreover, the individual contributions of clip duration, camera supervision, and structured captions remain to be isolated, and large-scale data construction is computationally intensive.

## Acknowledgments and Disclosure of Funding

This work was conducted during an internship at Alaya Lab. We sincerely thank Kaipeng Zhang and Yongtao Ge, the project leader, for their valuable guidance and continuous support throughout this research. We are also grateful to our colleagues and collaborators for their insightful discussions, constructive feedback, and contributions to the development of GameVerse.

Funding and Competing Interests. This research was primarily supported by Alaya Lab through the provision of research data resources. Alaya Lab has also authorized the public release of the resulting dataset metadata and derived annotations. The authors declare no competing interests.

## References

[1] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Proceedings of the International Conference on Machine Learning, pages 4603–4623, 2024.

[2] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. In Proceedings ofthe Thirteenth International Conference on Learning Representations, 2025.

[3] Etched Decart, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A universe in a transformer. URL: https://oasis-model. github. io, 2(3):6, 2024.

[4] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 27713–27724, 2025.

[5] Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[6] Chenlong He, Qi Zheng, Ruoxi Zhu, Xiaoyang Zeng, Yibo Fan, and Zhengzhong Tu. Cover: A comprehensive video quality evaluator. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5799–5809, 2024.

[7] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.

[8] Kang He, Wenshuo Peng, Zihui Gao, Jiaming Tan, Kaipeng Zhang, and Yongtao Ge. Sekai2: From world exploration to interactive world modeling. arXiv preprint arXiv:2608.09449, 2026.

[9] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.

[10] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pages 961–970, 2015.

[11] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934, 2025.

[12] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[13] Team HunyuanWorld. Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint, 2025.

[14] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. Advances in Neural Information Processing Systems, 37:48955–48970, 2024.

[15] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Densecaptioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715, 2017.

[16] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2(3):6, 2025.

[17] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 25690–25699, 2025.

[18] Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Xinyue Li, Yukang Feng, Jianwen Sun, Zizhen Li, et al. Sekai: A video dataset towards world exploration. In Proceedings of the Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.

[19] Lightricks. LTX-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

[20] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 2630–2640, 2019.

[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763, 2021.

[22] Skywork AI Matrix-Game Team. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

[23] Tomás Soucek and Jakub Lokoc. Transnet v2: An effective deep network architecture for fast shot transition detection. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11218–11221, 2024.

[24] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

[25] AlayaWorld Team, Kaipeng Zhang, Chuanhao Li, Yifan Zhan, Yongtao Ge, Yuanyang Yin, Jiaming Tan, Kang He, Liaoyuan Fan, Mingliang Zhai, et al. Alayaworld: Interactive longhorizon world modeling–full technical report. arXiv preprint arXiv:2607.18367, 2026.

[26] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

[27] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34:16558–16569, 2021.

[28] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are realtime game engines. In Proceedings of the thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025, 2025.

[29] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In Proceedings of the Twelfth International Conference on Learning Representations, 2024.

[30] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In Proceedings ofthe ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.

[31] Zeqi Xiao, LAN Yushi, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory. In Proceedings ofthe Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[32] Xiaojie Xu, Zhengyuan Lin, Kang He, Yukang Feng, Xiaofeng Mao, Yuanyang Yin, Yongtao Ge, and Kaipeng Zhang. Worldmark: A unified benchmark suite for interactive video world models. arXiv preprint arXiv:2604.21686, 2026.

[33] Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuanhao Li, Kaipeng Zhang, and Feng Zhao. Alaya-evoke: From linear-scaling supervision to endless world. arXiv preprint arXiv:2608.13546, 2026.

[34] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.

## A Implementation Details

Training Configuration. We fine-tune LTX-Video 2.3 (22B parameters) from the official base checkpoint on a curated GameVerse subset of 24,380 game-video segments (12–15 s each) at 540p / 24 fps. Since GameVerse targets visual interactive world modeling, we remove the audio branch of LTX-Video 2.3 prior to fine-tuning and train only the video stream. Training uses a mixed objective with weights 60% image-to-video (i2v), 20% video-to-video (v2v), and 20% text-to-video (t2v), runs for 3,700 steps at batch size 1 per GPU under FSDP across 64 GPUs. Camera conditioning is provided by additive Plücker injection and action conditioning by 81-class direction-angle embeddings. We optimise with AdamW at learning rate $1 \times 1 0 ^ { - 4 }$ on a constant schedule with a 200-step warmup, and retain the base model’s RoPE (max $\mathtt { - p o s } = 2 0 )$ under fractional-position normalisation so that all training samples remain within the position range seen during pretraining.

Data Processing Efficiency. To support large-scale data construction, we optimize each stage of the GameVerse pipeline for high-throughput processing. For scene-consistent segmentation, we adopt FFmpeg with NVENC acceleration, which significantly improves video decoding and preprocessing efficiency. Combined with TransNetV2, the segmentation pipeline achieves a processing throughput of approximately 400 FPS per worker. For camera trajectory estimation, we employ a distributed ViPE inference pipeline deployed on 256 GPUs. On average, pose estimation for each gameplay video requires approximately 6 minutes, enabling scalable processing of long-duration videos. For structured semantic annotation, we deploy Kimi-K2.5 using the SGLang inference framework on H20 90GB GPUs. The caption generation stage is executed on 16 compute nodes with a total of 128 GPUs, allowing efficient large-scale generation of multi-granular semantic annotations.

## B Detailed Multi-stage Video Quality Assessment Pipeline

Even after segmentation, raw gameplay clips often contain undesirable artifacts such as subtitles, watermarks, UI overlays, compression noise, and unstable motion. These artifacts introduce spurious correlations and degrade temporal consistency, which are particularly harmful for world modeling. To address this, we design a multi-stage quality assessment pipeline with explicit criteria and calibrated thresholds, jointly considering visual quality, content cleanliness, and temporal stability.

Overall Formulation. We define the overall quality of a video clip x as a weighted combination of multiple metrics:

$$
Q ( x ) = \sum _ { k } w _ { k } \cdot q _ { k } ( x ) ,\tag{1}
$$

where $q _ { k } ( x )$ denotes the k-th quality metric and $w _ { k }$ is its corresponding weight. In practice, we adopt a hybrid strategy that combines hard constraints with soft scoring, ensuring both strict filtering and relative ranking.

Visual Quality. We first evaluate low-level visual properties to remove clips with obvious degradation. Specifically, we require all clips to satisfy a minimum resolution of 720p, frame rate within [24, 65] fps, and valid encoding formats (e.g., H.264, HEVC, or AV1). We further compute luminance using the BT.601 standard on frames sampled at 5 fps, and discard clips that contain prolonged underexposed or overexposed segments, defined as more than 15 consecutive frames with luminance below 16 or above 240. In addition, we detect border artifacts such as letterboxing and pillarboxing by measuring pixel variance along image boundaries; regions with standard deviation below 4.0 are treated as uniform borders, and clips are removed if such regions occupy more than 10% of the frame.

Content Cleanliness. To eliminate non-world signals, we explicitly filter out text overlays and UI elements. We apply OCR-based detection to identify subtitles and watermarks, focusing on the lower portion of the frame $( y \in [ 0 . 7 , 1 . 0 ] )$ . Text regions that persist across multiple frames or occupy more than 2% of the image area trigger rejection, with a minimum detectable text height ratio of 0.008 to capture small HUD elements. In addition, we detect UI overlays based on their temporal stability and spatial structure. Specifically, we compute the temporal standard deviation $\sigma _ { t } .$ mean intensity $\mu _ { t } .$ , and spatial gradient magnitude |∇I| for each pixel, and identify UI regions as those satisfying $\left( \sigma _ { t } < 4 \right) \wedge \left( \mu _ { t } > 2 5 \right) \wedge \left( | \nabla I | > 1 2 \right)$ . Clips with UI coverage exceeding 4% are removed. This step is crucial to prevent models from overfitting to game-specific interface signals.

Temporal Stability. We then evaluate whether clips exhibit stable and physically plausible dynamics. Motion is estimated using optical flow (Farneback) at 4 fps, where frames with mean flow magnitude below 0.3 are considered static; clips with more than 50% such frames are discarded. We further detect freezing artifacts by measuring frame-to-frame luminance differences: consecutive frames with differences below 0.5 are marked as frozen, and clips are rejected if freezing persists for more than approximately 2 seconds or exceeds 30% of the clip. To capture unstable camera motion, we analyze displacement statistics within sliding windows and identify jitter spikes when instantaneous motion exceeds five times the local mean; clips with more than four such spikes per minute are filtered out.

Beyond these rule-based filters, we incorporate learned quality metrics. We compute multidimensional quality scores using COVER, including semantic, technical, and aesthetic components, and retain only the top 70% of clips ranked by the sum of aesthetic and technical scores. We further evaluate temporal consistency using VBench [12] metrics, including subject consistency, background consistency, and temporal flickering, with thresholds set to 0.65, 0.85, and 0.83 respectively.

Final Selection. The final decision integrates all criteria. Hard constraints (e.g., format validity, OCR detection, UI filtering, motion and freeze rejection) ensure that low-quality samples are strictly removed, while soft scoring mechanisms (e.g., COVER ranking and VBench thresholds) provide finer-grained quality control. Only clips that satisfy all constraints and exceed the required thresholds are retained.

Effectiveness. This multi-stage pipeline yields a dataset with high visual clarity, clean semantic content, and stable temporal dynamics. Compared to simple filtering strategies, it provides a more robust and comprehensive quality control mechanism, which is essential for training reliable and generalizable world models.

## C Additional Experimental Details and Analyses

## C.1 Evaluation Metrics

VBench. We adopt six evaluation dimensions from VBench [12] to assess temporal consistency, perceptual quality, and semantic alignment of generated videos. Subject Consistency measures the stability of subject appearance across frames using DINO feature similarity, capturing identity drift and unintended changes in visual attributes. Background Consistency evaluates scene-level coherence through CLIP feature similarity, reflecting whether background structures remain visually consistent over time. Motion Smoothness assesses the temporal continuity of generated motion using a video frame interpolation model, penalizing abrupt or unnatural transitions. For frame-level quality, Aesthetic Quality measures visual appeal using the LAION aesthetic predictor, considering aspects such as composition and color harmony, while Imaging Quality employs MUSIQ to evaluate visual distortions, including blur, noise, and overexposure. Finally, Overall Consistency measures semantic and stylistic alignment between generated videos and their textual descriptions using ViCLIP. Together, these metrics characterize complementary aspects of video generation quality, particularly temporal coherence and visual fidelity over extended rollouts.

WorldScore. We further employ WorldScore [4] to evaluate world generation quality and controllability from geometric, semantic, and perceptual perspectives. The benchmark includes three controllability metrics and four quality metrics. Camera Control measures adherence to the target camera trajectory by comparing estimated camera poses from generated videos with the prescribed motion, incorporating scale-invariant rotation and translation errors. Object Control evaluates whether prompt-specified objects appear in generated scenes using open-set object detection, while Content Alignment measures semantic agreement with the complete input description using CLIPScore. For generation quality, 3D Consistency evaluates geometric stability across frames through reprojection errors based on DROID-SLAM reconstruction. Photometric Consistency assesses the stability of visual appearance and textures using forward–backward optical-flow consistency, capturing artifacts such as texture shifts and flickering. Style Consistency evaluates whether the visual style remains consistent by comparing feature Gram matrices across generated frames. Finally, Subjective Quality estimates perceptual quality through an automatic combination of CLIP-IQA+ and CLIP Aesthetic scores, rather than direct human ratings.

Table 6: Trajectory quality across six motion types after filtering. Rotation RPE and translationdirection error are computed against DROID-SLAM after Sim(3) alignment. They measure crossestimator agreement, not ground-truth accuracy. Jump and jerk columns reproduce the source statistics.
<table><tr><td>Motion</td><td>Valid poses ↑</td><td>Jump ↓</td><td>Rot. jerk ↓</td><td>Rot. RPE (°) ↓</td><td>Trans. dir. (°) ↓</td></tr><tr><td>Forward</td><td>100.00%</td><td>0.0000</td><td>0.0055</td><td>0.75</td><td>4.84</td></tr><tr><td>Lateral</td><td>100.00%</td><td>0.0000</td><td>0.0057</td><td>1.31</td><td>6.42</td></tr><tr><td>Orbit</td><td>100.00%</td><td>0.0014</td><td>0.0099</td><td>1.31</td><td>6.59</td></tr><tr><td>Tilt</td><td>100.00%</td><td>0.0006</td><td>0.0084</td><td>1.40</td><td>9.10</td></tr><tr><td>Mixed-rot</td><td>100.00%</td><td>0.0220</td><td>0.0126</td><td>1.47</td><td>11.03</td></tr><tr><td>Complex</td><td>100.00%</td><td>0.0006</td><td>0.0061</td><td>0.88</td><td>4.94</td></tr></table>

## C.2 Camera-Pose Supervision Sources

LingBot-World. LingBot-World [5] adopts a hybrid data engine combining general videos, synchronized game recordings, and synthetic Unreal Engine (UE) rendering. Game recordings provide temporally aligned visual observations, user actions, and camera parameters, while UE-rendered sequences offer ground-truth camera intrinsics and extrinsics. For general videos lacking camera annotations, LingBot employs MegaSAM to estimate camera poses. Its camera-conditioning mechanism further incorporates continuous camera information through Plücker embeddings for fine-grained viewpoint control.

HY-World 1.5. HY-World 1.5 [13, 24] constructs its training corpus from approximately 320K clips, including 170K AAA-game recordings, 60K reconstructed scenes from DL3DV, 50K synthetic UE-rendered sequences, and 40K real-world videos from Sekai. The UE-rendered subset provides ground-truth camera trajectories, while the DL3DV subset uses 3D reconstruction and customized trajectory rendering for geometrically consistent supervision. For videos without available camera poses, ViPE is used for trajectory estimation and subsequent motion-consistency filtering. HY-World thus combines rendered, reconstruction-derived, and estimated trajectories with dedicated camera-conditioning mechanisms.

Comparison with GameVerse. Unlike these systems, GameVerse primarily collects natural gameplay recordings without direct access to game-engine camera parameters. Camera trajectories are recovered using ViPE and subjected to multi-stage quality filtering to remove abrupt jumps, jitter, and unreliable motion. While estimated trajectories lack the precision guarantees of engine-rendered ground truth, they enable scalable coverage of diverse gameplay motions, including irregular naviga tion, coupled translation and rotation, and complex viewpoint transitions. Therefore, differences in camera-control performance may reflect both pose-supervision sources and conditioning architectures. Nevertheless, improvements from HY-World 1.5 to HY-World-FT demonstrate the utility of GameVerse’s estimated trajectories for geometric supervision.

## C.3 Camera-Trajectory Reliability by Motion Type

To assess the reliability of GameVerse’s estimated trajectories, we evaluate pose completeness, temporal stability, and cross-estimator agreement across six camera-motion categories: forward, lateral, orbit, tilt, mixed rotation, and complex motion. As shown in Table 6, all audited retained trajectories contain valid poses throughout their duration, with generally low jump rates and rotational jerk. We further compare ViPE trajectories against an independent DROID-SLAM estimator after Sim(3) alignment. Rotation RPE ranges from 0.75 to 1.47 degrees, while translation-direction error ranges from 4.84 to 11.03 degrees, with mixed rotation exhibiting the largest errors. These results support the temporal stability and cross-estimator consistency of the retained trajectories, although they do not establish absolute pose accuracy without ground-truth trajectories. The reported 100% valid-pose ratios refer to the retained audit set rather than all raw source videos.

## C.4 Diagnostic Comparison of Original Backbones

To examine the influence of backbone architectures, we compare the original Wan2.2 and LTX-Video 2.3 models without camera conditioning or GameVerse fine-tuning at 5-, 10-, and 15-second generation durations. Both models use the same 103 input images and captions, a resolution of 1024 × 576, seed 42, 40 sampling steps, and a single-stage, full-resolution, non-distilled pipeline. Their native frame rates differ (16 fps for Wan2.2 and 24 fps for LTX-Video 2.3), which may affect motion-dependent metrics.

Table 7: Diagnostic comparison of original backbones without GameVerse fine-tuning or camera conditioning. Both use the same 103 inputs, 1024 × 576 resolution, seed 42 and 40 sampling steps. Wan2.2 uses 16 fps and LTX-2.3 uses 24 fps. Scores are multiplied by 100.
<table><tr><td>Duration</td><td>Model</td><td>Subj. Cons.</td><td>Bkg. Cons.</td><td>Mot. Smth.</td><td>Aes. Qual.</td><td>Img. Qual.</td><td>Overall Cons.</td></tr><tr><td rowspan="2">5s</td><td>Wan2.2</td><td>76.72</td><td>89.91</td><td>95.87</td><td>13.30</td><td>63.97</td><td>18.89</td></tr><tr><td>LTX-2.3</td><td>72.86</td><td>87.36</td><td>96.35</td><td>11.13</td><td>52.36</td><td>19.31</td></tr><tr><td rowspan="2">10s</td><td>Wan2.2</td><td>77.69</td><td>90.15</td><td>96.92</td><td>13.45</td><td>65.30</td><td>19.34</td></tr><tr><td>LTX-2.3</td><td>74.18</td><td>86.97</td><td>96.78</td><td>11.89</td><td>51.25</td><td>18.71</td></tr><tr><td rowspan="2">15s</td><td>Wan2.2</td><td>75.13</td><td>88.86</td><td>97.46</td><td>10.04</td><td>63.54</td><td>18.75</td></tr><tr><td>LTX-2.3</td><td>71.37</td><td>85.90</td><td>97.52</td><td>8.16</td><td>54.29</td><td>18.81</td></tr></table>

![](images/3e2da3a2eca6e5e08faafbdacda679093843d3b5a8c8646645fe6183cc50b3cb.jpg)  
Figure 6: Example long-duration gameplay sequences from GameVerse. Each row shows temporally continuous frames sampled from a single gameplay clip across diverse game environments, including indoor exploration, natural scenes, urban driving areas, sunset city environments, and nighttime interaction scenarios. The examples demonstrate the strong temporal continuity, environmental diversity, camera motion consistency, and rich visual dynamics provided by GameVerse.

As shown in Table 7, Wan2.2 consistently achieves higher aesthetic and imaging quality across all three durations, with average advantages of 1.87 and 11.64 points, respectively. These results suggest that appearance-related performance differences between LingBot and LTX-FT may partly originate from their underlying backbones rather than GameVerse fine-tuning alone. However, this diagnostic does not isolate the causes of cross-model differences in 60-second generation. The controlled comparison between HY-World 1.5 and HY-World-FT in the main text provides more direct evidence of the benefits of GameVerse.

## D Visualization

Figure 6 presents representative long-duration gameplay sequences from GameVerse. Each row corresponds to temporally continuous frames sampled from a single gameplay clip, illustrating the diversity of environments, lighting conditions, gameplay perspectives, and interaction dynamics contained in the dataset. The examples cover indoor exploration scenes, open-world natural environments, urban roads, sunset cityscapes, and nighttime gameplay scenarios across multiple game titles. Notably, the sampled sequences exhibit strong temporal continuity and stable camera motion over long horizons, while preserving diverse scene layouts and environmental transitions. These examples further demonstrate the ability of GameVerse to capture complex world dynamics and long-term scene evolution for interactive world modeling.

![](images/2a781bf6a4bf03d7beb6decd7f634c4096e5ab943a0e056d79954a9e9b7ee3c0.jpg)  
Figure 7: Overview of the structured semantic annotation format in GameVerse. Each gameplay video is decomposed into global metadata and temporally aligned segment-level annotations, jointly modeling scene states, dynamic events, agent behaviors, environmental changes, and temporal scene evolution.

Table 8: Example annotation from a historical urban environment (Assassin’s Creed-style gameplay).
<table><tr><td>Category</td><td>Annotation Example</td></tr><tr><td>Global Description</td><td>Third-person gameplay footage showing a hooded protagonist exploring a historical Middle Eastern city with arcades, markets, plazas, fountains, and crowded urban streets.</td></tr><tr><td>Dynamic Environ- ment</td><td>Clear weather, midday lighting, historic sandstone architecture, pedestrian streets, moderate crowd density.</td></tr><tr><td>Scene Evolution</td><td>Covered stone arcade → narrow residential street → market district → central plaza → elevated garden terrace.</td></tr><tr><td>Dynamic Events</td><td>Exiting covered arcade, navigating market crowds, interacting with NPC flow, turning around central fountain, ascending garden stairs.</td></tr><tr><td>Agent Behaviors</td><td>Walking through crowded streets, exploring urban plazas, navigating around market vendors and animals.</td></tr><tr><td>Key Scene Elements</td><td>Stone archways, fabric awnings, domed buildings, market stalls, fountains, palm trees, flower beds, garden terraces.</td></tr></table>

Table 9: Example annotation from a third-person cinematic exploration sequence.
<table><tr><td>Category</td><td>Annotation Example</td></tr><tr><td>Global Description</td><td>Third-person gameplay footage showing a warrior character exploring a rocky desert en- vironment, passing through cave tunnels, ancient ruins, and lakeside areas under dynamic lighting conditions.</td></tr><tr><td>Weather / Time / Style Camera Motion</td><td>Clear weather, midday lighting, cinematic visual style, third-person follow camera. Walking-forward camera motion with stabilized chest-level perspective.</td></tr><tr><td>Scene Segment 1</td><td>Wide desert landscape with large rock formations and sparse vegetation. The character walks toward a distant cave entrance while lightning flashes in the background.</td></tr><tr><td>Scene Segment 2</td><td>The character enters a narrow cave tunnel, transitioning from bright sunlight into a dark enclosed environment with rough stone walls.</td></tr><tr><td>Scene Segment 3</td><td>The scene opens into ancient ruins dominated by a massive tree structure, surrounded by stone pillars and distant mountains.</td></tr><tr><td>Dynamic Events</td><td>Lightning strikes, scene transitions, entering enclosed spaces, approaching environmental landmarks.</td></tr><tr><td>Agent Actions</td><td>Walking forward, exploring ruins, traversing shoreline areas, entering shallow water. Environment Elements Ancient ruins, cave archways, mountain ranges, lakeside shoreline, rocky terrain, stone</td></tr></table>

## E Four-level Structured Semantic Annotations

To better support long-horizon interactive world modeling, GameVerse introduces a four-level structured semantic annotation framework that explicitly decomposes gameplay videos into complementary semantic components. Unlike conventional video captions that describe an entire clip using a single sentence, our annotations separately model global static, local static, dynamic events, and agent actions, enabling more structured supervision for controllable generation and temporal world understanding.

Figure 7 summarizes the overall annotation format of GameVerse. Each gameplay video is first associated with global metadata describing environmental conditions, camera properties, and high-level scene semantics. The video is then divided into temporally aligned segments, where each segment contains structured annotations for local scene structure, dynamic events, interaction behaviors, environmental evolution, and natural-language scene descriptions.

Global Static. The first level focuses on globally consistent environmental properties throughout the video sequence. These annotations describe high-level scene attributes such as environment type, architectural style, weather conditions, lighting conditions, season, camera perspective, and overall scene composition. As shown in Tables 8–10, the annotations cover highly diverse environments including historical Middle Eastern cities, rocky desert landscapes, coastal ruins, underground caves, and large-scale natural environments. Such annotations provide stable world-level context and coarse-grained environmental priors for long-horizon generation.

Table 10: Example annotation from a first-person cinematic exploration sequence.
<table><tr><td>Category</td><td>Annotation Example</td></tr><tr><td>Global Description</td><td>First-person gameplay footage showing exploration of coastal ruins, medieval stone architec- ture, beaches, and underground cave systems.</td></tr><tr><td>ment</td><td>Dynamic Environ- Partly cloudy weather, cinematic rendering, first-person camera, coastal environment, mixed indoor-outdoor transitions.</td></tr><tr><td>Scene Evolution</td><td>Rocky canyon → covered corridor → open beach → castle entrance → underground cave.</td></tr><tr><td>Dynamic Events</td><td>Transition from canyon to open waterfront, entering covered structures, discovering cave interiors illuminated by natural light shafts.</td></tr><tr><td>Agent Behaviors</td><td>Walking through environmental transitions, approaching landmarks, exploring caves and hidden interiors.</td></tr><tr><td>Key Scene Elements</td><td>Stone bridges, water canals, medieval castle, armored statues, cave grotto, large underground tree.</td></tr></table>

Table 11: Example annotation from a large-scale natural exploration environment.
<table><tr><td>Category</td><td>Annotation Example</td></tr><tr><td>Global Description</td><td>Third-person gameplay footage of a warrior character exploring rocky deserts, ancient ruins, caves, and lakeside environments under cinematic lighting.</td></tr><tr><td>ment</td><td>Dynamic Environ- Clear weather, desert terrain, natural sunlight, wide-open spatial layout, sparse vegetation.</td></tr><tr><td>Scene Evolution</td><td>Rocky desert → cave tunnel → ancient ruins → lakeside shoreline.</td></tr><tr><td>Dynamic Events</td><td>Lightning strikes in the distance, entering enclosed cave areas, approaching monumental ruins, transitioning toward waterfront scenery.</td></tr><tr><td>Agent Behaviors</td><td>Exploring ruins, traversing cave systems, approaching environmental landmarks, entering shallow water.</td></tr><tr><td>Key Scene Elements</td><td>Rock formations, cave entrances, ancient ruins, giant world tree, mountain ranges, lakeshore environments.</td></tr></table>

Local Static. The second level captures fine-grained spatial structures within temporally aligned segments. Compared with global context annotations, this level focuses on local scene layouts and environment details that evolve over time. For example, the annotations explicitly describe transitions between covered stone arcades, narrow residential streets, open plazas, cave tunnels, lakeside shorelines, and underground grotto environments. These local structural descriptions preserve spatial continuity and geometric organization across long-duration gameplay sequences.

Dynamic Events. The third level models temporally evolving events and environmental transitions. Instead of only describing static scenes, the annotations explicitly capture scene evolution over time, including entering enclosed caves, exiting urban districts, approaching environmental landmarks, navigating through crowded markets, and transitioning between indoor and outdoor environments. As illustrated in the examples, GameVerse also captures environmental events such as lightning flashes, motion changes, and camera-view transitions, enabling richer supervision for temporal causality and long-horizon scene evolution.

Agent Actions. The fourth level focuses on interaction-centric behaviors of the player character and surrounding agents. These annotations describe actions such as walking, exploration, climbing, stealth movement, driving, environmental interaction, and NPC activities. For example, the examples include navigating crowded urban streets, traversing cave systems, approaching historical landmarks, and exploring large-scale outdoor environments. Compared with conventional action labels, the proposed annotations jointly model both the action itself and its surrounding interaction context, enabling richer supervision for controllable interaction modeling and embodied world simulation.

Together, these four semantic levels provide complementary supervision signals for long-horizon video understanding and generation. By explicitly separating global context, local structure, dynamic events, and interaction behaviors, GameVerse provides substantially richer annotations than conventional flat captioning approaches and better supports controllable interactive world modeling.

![](images/f204d45a0adb94f609877aa45f20fb48faa5665acf4115f2acc6b27f2c95fb4e.jpg)  
Figure 8: Qualitative image-to-video generation results of the LTX-2.3 model fine-tuned on GameVerse. Given a single initial frame together with camera trajectories and structured semantic captions, the model generates long-duration gameplay videos with consistent scene layouts, stable camera motion, and coherent temporal dynamics across diverse environments, including natural exploration, urban interaction, horse riding, and nighttime driving scenarios.

## F Case Study

## F.1 Qualitative Long-horizon Generation.

Figure 8 presents representative image-to-video generation results using the LTX-Video-2 model fine-tuned on GameVerse. Given only a single initial frame, together with the corresponding camera trajectories and structured semantic captions, the model is able to generate temporally coherent long-duration gameplay videos across diverse environments and interaction scenarios.

The generated sequences demonstrate strong scene consistency and stable camera-aware motion over long temporal horizons. For example, the model successfully preserves environmental geometry and spatial layouts in large-scale outdoor exploration scenes, while maintaining smooth camera transitions and consistent illumination changes in urban streets, horseback riding, and nighttime driving scenarios. Moreover, the generated videos exhibit coherent interaction dynamics and realistic environmental evolution, indicating that the structured semantic supervision provided by GameVerse effectively improves long-horizon controllable video generation.

These qualitative results further demonstrate the effectiveness of GameVerse for geometry-aware, temporally consistent, and interaction-centric world modeling

## F.2 Qualitative Camera-Controlled Generation.

Figure 9 presents qualitative image-to-video generation results conditioned on both camera trajectories and structured semantic captions. The overlaid blue trajectories visualize the input pose control signals provided to the generation model.

![](images/104ba1ac20c08a0935259a86a7c21d09ae753d6ea2c0e79aff5fe1bd7ac776c9.jpg)  
Figure 9: Qualitative image-to-video generation results conditioned on camera trajectories and structured semantic captions. The blue trajectory overlays visualize the input camera motion signals used for controllable generation. Given a single initial frame, the LTX-Video-2 model fine-tuned on GameVerse is able to generate temporally coherent long-duration gameplay videos with stable camera-aware motion, consistent scene geometry, and smooth environmental transitions across diverse gameplay scenarios.

As shown in the examples, the model is able to follow complex camera motions while preserving longterm scene consistency and coherent environmental structure. Across diverse gameplay scenarios, including road navigation, horseback riding, forest exploration, snowy environments, and urban scenes, the generated videos exhibit smooth camera-aware motion and stable temporal evolution over long horizons.

Moreover, the generated sequences maintain consistent scene geometry and spatial layouts under large camera movements, suggesting that the camera trajectory supervision provided by GameVerse effectively improves geometry-aware and controllable long-duration video generation.