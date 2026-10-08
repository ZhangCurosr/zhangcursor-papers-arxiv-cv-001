# DO BETTER VISUAL REPRESENTATIONS ALWAYS LEAD TO BETTER END-TO-END AUTONOMOUS DRIVING?

Zihao Zhang<sup>1,2,3</sup> Haochen Tian<sup>2</sup> Tianyu Li<sup>2</sup> Changhui Jing<sup>2</sup> Jingliang He<sup>3</sup> Naisheng Ye<sup>2</sup> Ziyuan Pu<sup>1‡</sup> Zhenjie Yang<sup>2‡</sup>

<sup>1</sup>Southeast University <sup>2</sup>OpenDriveLab at The University of Hong Kong

<sup>3</sup>Shenzhen Loop Area Institute

<sup>‡</sup>Correspondence Authors. Primary contact: zihao.zhang@opendrivelab.com

## ABSTRACT

Visual foundation models (VFMs) are increasingly integrated into end-to-end autonomous driving for their powerful representations, yet it remains unclear when these representations improve driving performance. To investigate this question, we introduce ViRA, a planner-agnostic visual representation alignment framework that keeps the planner architecture and inference cost unchanged. Our study reveals three findings: (1) VFM-guided visual representations consistently improve driving performance across diverse end-to-end planners, with gains extending to zeroshot closed-loop evaluation. (2) The choice of VFM target matters for planning performance, and alignment to a different VFM can further benefit planners with pre-trained VFM encoders. (3) Auxiliary perception supervision reduces sensitivity to VFM target selection, narrowing the EPDMS spread across five targets from 2.7 to 0.5 points and potentially compensating for less effective VFM targets. Guided by these findings, we develop ViRA-Diffusion, a diffusion-based planner trained without auxiliary perception supervision, which achieves 92.3 EPDMS on NAVSIM v2 navtest, outperforming recent methods in our comparison by at least 1.9 points. The results motivate jointly considering target selection and planner supervision when integrating VFMs into end-to-end autonomous driving. The results and demo are available at https://github.com/OpenDriveLab/ViRA.

## 1 INTRODUCTION

End-to-end autonomous driving is considered a promising route to scalable real-world deployment (Chen et al., 2024a; Guan et al., 2026). Within this pipeline, visual representations provide the planner with information about the environment, making the learning of high-quality visual representations an important research problem (Jiang et al., 2023; Li et al., 2024). As visual foundation models (VFMs) advance in semantic understanding and 3D geometry (Siméoni et al., 2025; Wang et al., 2025; Carion et al., 2025; Lin et al., 2025), an increasing number of studies either use VFMs directly (Sima et al., 2024; Wang et al., 2026a; Naeinian et al., 2026) or incorporate their pre-trained representations through distillation and representation alignment (Strong et al., 2026; Li et al., 2026a; Xie et al., 2026). However, most studies focus on a single planner–VFM pairing (Zuo et al., 2026b; Wang et al., 2026b), and systematic comparisons of planning gains across different pairings remain limited. It remains unclear whether the advantages of VFMs on visual tasks consistently translate into improved driving performance.

To answer this question, we introduce ViRA, a VFM-guided representation alignment framework illustrated in Figure 1(a). ViRA aligns the planner’s intermediate features with those of a frozen VFM target while keeping the planner architecture and inference cost unchanged (Yu et al., 2025b; Singh et al., 2026; Li et al., 2026a). Using this framework, we compare different combinations of end-to-end planners and VFM targets on NAVSIM v2 (Cao et al., 2025) and HUGSIM (Zhou et al., 2025a). We find that VFM-guided visual representations consistently improve driving performance across the diverse planners, but the gains depend on both the VFM target and planner design. We further examine these findings below, focusing on overall driving gains, the effects of VFM target selection, and the role of auxiliary perception supervision.

![](images/e026891d940fb543d820350bb1cd6aa454bdf197aaa12d37f39fe3cf0c7dbda6.jpg)

![](images/5c38a5122774e91f9c5520b2f6bd065add7a19bd91ebb927059165d8965290e3.jpg)

![](images/3c7a8bb82e0a7a6f1891514b40b21fdac50913c5f5d2cd3336ab8abc84f334ef.jpg)

![](images/3c457cee10e671b1874478635b7024fb4d18831145ae34c0410256611e07846e.jpg)

![](images/d780e4684df7d1971f68311895f835f773a89b0864361bccf646941fe91b3a82.jpg)  
Figure 1: Better visual representations improve planning, with gains depending on both VFMs and planners. (a) ViRA aligns planner representations with a frozen VFM during training, leaving inference unchanged. (b) Segmentation and visual attention suggest improved representations. (c) Better visual representations improve planning, but the gains depend on the VFM target and planner.

VFM-guided visual representations improve end-to-end driving performance. As shown in Figure 1(c-i), ViRA improves driving performance across regression-based (Chitta et al., 2023), diffusion-based (Liao et al., 2025), and scoring-based (Feng et al., 2026) planners, yielding average gains of 7.9 and 8.6 EPDMS points over their respective baselines on NAVSIM v2 navtest and navhard, respectively. These benefits extend to zero-shot closed-loop evaluation on HUGSIM, where overall HD-Score improves by an average of 4.0 points. Figure 1(b) shows that driving gains are accompanied by improved perception performance. To test whether the gains come from pre-training rather than loss regularization alone, we compare pre-trained and randomly initialized targets under the same alignment objectives. Randomly initialized targets fail to reproduce these perception and planning gains, supporting the importance of pre-trained VFM representations. This raises the question of whether VFMs with different pre-training objectives yield different planning benefits.

The choice of VFM target matters. As shown in Figure 1(c-ii), gains for the same Rap<sup>∗</sup> (Feng et al., 2026) architecture range from 6.6 to 12.1 EPDMS points across VFM targets (a 5.5-point spread), with DINOv3 (Siméoni et al., 2025) yielding the largest gain. Yet even when Rap<sup>∗</sup> uses DINOv3 as its pre-trained backbone, alignment to VGGT (Wang et al., 2025), a 3D geometric foundation model, further improves EPDMS from 83.8 to 86.3 (+2.5 points). This dependence on the VFM target also appears in TransFuser (Chitta et al., 2023) when trained without auxiliary perception supervision, with DINOv3 again performing best. With auxiliary perception supervision, however, TransFuser and DiffusionDrive (Liao et al., 2025) achieve similar planning performance across these targets. We hypothesize that auxiliary perception supervision affects how planners learn from VFM targets.

Auxiliary perception supervision reduces sensitivity to VFM target selection. To examine this factor, we compare TransFuser with and without auxiliary perception supervision. Across the same five VFM targets, planning gains over the respective baselines without representation alignment range from 5.2 to 5.7 EPDMS points with auxiliary supervision and from 3.8 to 6.5 points without it. As shown in Figure 1(c-iii), removing auxiliary supervision widens the spread in gains across targets from 0.5 to 2.7 EPDMS points. More specifically, the highest score stays nearly unchanged, while the lowest drops from 88.8 to 86.4, indicating that auxiliary supervision mainly compensates for the less effective targets. These results show that, in this setting, auxiliary perception supervision narrows the differences in gains across alignment targets, making planning performance less sensitive to VFM target selection.

These findings make a concrete prediction: a planner can achieve strong planning performance when its auxiliary perception supervision is removed, and a suitable VFM target is supplied in its place. We verify this with ViRA-Diffusion, a diffusion-based (Liao et al., 2025) planner trained with ViRA using DINOv3 (Siméoni et al., 2025) as the target and no auxiliary perception supervision. It reaches 92.3 EPDMS on NAVSIM $\mathbf { v } 2$ (Cao et al., 2025) navtest, outperforming the recent state-of-the-art method by 1.9 points. We highlight the main contributions of this paper below:

• We systematically study the driving benefits of pre-trained VFM representations across planners and targets using ViRA, a planner-agnostic experimental framework that preserves the deployed planner architecture and inference cost.

• We show that VFM target selection affects planning performance, while auxiliary perception supervision narrows performance differences across targets, highlighting the importance of considering target selection and planner supervision jointly.

• We demonstrate the practical applicability of VFM-guided alignment through ViRA-Diffusion, which uses DINOv3 alignment to achieve 92.3 EPDMS on NAVSIM $\mathbf { v } 2$ navtest without auxiliary perception supervision.

## 2 PRELIMINARIES

## 2.1 VFM-GUIDED END-TO-END PLANNER

End-to-end Planner. Given a single-frame multi-view observation ${ \mathcal { O } } = \{ \mathbf { I } ^ { i } \} _ { i = 1 } ^ { N }$ , an end-to-end planner predicts a future ego trajectory $\hat { \tau } = \{ \hat { \mathbf { p } } _ { k } \} _ { k = 1 } ^ { K }$ , where N is the number of camera views and $\hat { \mathbf p } _ { k } = \left( \hat { x } _ { k } , \hat { y } _ { k } \right)$ denotes the predicted position at future step k in the ego coordinate system. We formulate the planner as an encoder $E _ { \theta }$ followed by a policy head $\pi _ { \psi } \mathbf { : }$

$$
{ \bf F } ^ { s } = E _ { \theta } ( { \mathcal O } ) , \qquad \hat { \boldsymbol { \tau } } = \pi _ { \psi } ( { \bf F } ^ { s } ) .\tag{1}
$$

The encoder maps visual observations into an intermediate representation $\mathbf { F } ^ { s }$ , from which the policy head extracts information for trajectory planning. The policy $\pi _ { \psi }$ can follow regression-based (Chitta et al., 2023), diffusion-based (Liao et al., 2025), or scoring-based (Feng et al., 2026) designs. This formulation provides a common interface for studying visual representations across planners with different trajectory prediction mechanisms.

VFM-Guided Training. To incorporate powerful VFM representations into planner representation learning, we design a VFM-guided training pipeline. During training, the same observation O is fed to a frozen VFM $T _ { \phi }$ to obtain $\mathbf { F } ^ { t } = T _ { \phi } \bar { ( \mathcal { O } ) }$ . These representations provide representation-level supervision for the planner encoder without being passed to the policy head $\pi _ { \psi }$ or requiring additional annotations for planner training. Only the original planner is kept at inference, leaving its deployed architecture and inference cost unchanged. Unless otherwise specified, we use DVGT (Zuo et al., 2026a) as $T _ { \phi } ,$ , while the same pipeline also supports other VFMs (Carion et al., 2025; Wang et al., 2025; Siméoni et al., 2025; Lin et al., 2025). Throughout the paper, DINOv3 refers to DINOv3-L unless otherwise specified.

Visual Representation Alignment. To support the spatial reasoning and scene understanding required for end-to-end autonomous driving, we design two objectives of ViRA to transfer highquality VFM representations to the planner encoder: Spatial Dependency Alignment (SDA) and Scene Semantic Alignment (SSA). SDA matches spatial attention distributions between $\mathbf { F } ^ { s }$ and $\mathbf { F } ^ { t }$ encouraging the planner to learn the spatial dependencies encoded by the VFM. SSA aligns their global embeddings using a bidirectional contrastive objective, encouraging scene-level semantic consistency between the planner and the VFM. The overall training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p l a n n e r } } + \mathcal { L } _ { \mathrm { a l i g n } } , \quad \mathcal { L } _ { \mathrm { a l i g n } } = \lambda _ { \mathrm { S D A } } \mathcal { L } _ { \mathrm { S D A } } + \lambda _ { \mathrm { S S A } } \mathcal { L } _ { \mathrm { S S A } } . } \end{array}\tag{2}
$$

where $\mathcal { L } _ { \mathrm { p l a n n e r } }$ is the original planner objective, including trajectory prediction and any auxiliary supervision used in the corresponding configuration, $\mathcal { L } _ { \mathrm { a l i g n } }$ is the representation alignment objective combining SDA and SSA, and $\lambda _ { \mathrm { S D A } }$ and $\lambda _ { \mathrm { S S A } }$ are scalar weights. Details of ViRA are provided in Appendix A.

## 2.2 BENCHMARK AND EVALUATION

NAVSIM (Dauner et al., 2024) is a data-driven, non-reactive simulation and benchmarking framework for autonomous driving built on OpenScene (OpenScene Contributors, 2023) and nuPlan (Karnchanachari et al., 2024). We use navtrain for training and evaluate models on two NAVSIM v2 (Cao et al., 2025) benchmarks, navtest and navhard. Both benchmarks adopt the rule-based planning metric EPDMS (Li et al., 2025a), whose sub-metrics include No-at-fault Collisions (NC), Drivable Area Compliance (DAC), Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Timeto-Collision (TTC), Ego Progress (EP), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC).

HUGSIM (Zhou et al., 2025a) is a real-time, photorealistic closed-loop simulator that reconstructs 3D scenes from captured RGB images with 3D Gaussian Splatting and updates ego and actor states at each step from the planner output, so that surrounding vehicles react to the ego. We evaluate planners trained only on NAVSIM in a zero-shot manner on scenarios grouped into four difficulty levels: Easy, Medium, Hard, and Extreme. The evaluation metric is HD-Score, which weights route completion (RC) by per-step safety and comfort scores. Beyond planning scores, we use mean Intersection-over-Union (mIoU) over seven BEV semantic classes to measure TransFuser’s perception performance and assess the quality of its learned representations. More details are provided in Appendix B.

![](images/fc49f8991ec60ad00a3dda63661206fd4538847598a289121537873005258549.jpg)  
(a) TransFuser

![](images/685464b9a37f07c4e9466aafc740623e6a223585ccdb9226bfb8afa6a2a2b848.jpg)  
(b) Rap\*  
Figure 2: t-SNE visualization of visual representations. Planner representations without and with ViRA are shown alongside the reference VFM representations for (a) TransFuser and (b) $\mathsf { R a p } ^ { * }$ . The visualization suggests that representation alignment with ViRA shifts planner representations toward those of the reference VFM in the t-SNE space.

## 3 VFM-GUIDED VISUAL REPRESENTATIONS DO IMPROVE DRIVING

## 3.1 DRIVING PERFORMANCE ACROSS DIFFERENT BENCHMARKS

Results on NAVSIM v2 navtest. Table 1 shows that VFM-guided representation alignment consistently improves driving performance across three representative planners. With ViRA, Trans-Fuser (Chitta et al., 2023), DiffusionDrive (Liao et al., 2025), and Rap<sup>∗</sup> (Feng et al., 2026) achieve EPDMS gains of 5.4, 7.4, and 10.9 points over their respective baselines. DAC and LK show substantial improvements, reflecting gains in drivable-area compliance and lane keeping, respectively. Both capabilities are closely tied to visual perception. These improvements suggest that the benefits of VFM-guided visual representations extend across different planner architectures in standard driving scenarios.

Table 1: Performance on the NAVSIM v2 navtest benchmark. Rap<sup>∗</sup> is our reimplementation with a different backbone from the official codebase and without the original data augmentation. For a fair comparison, all methods use camera input only, without LiDAR.
<table><tr><td>Methods</td><td>Decoder</td><td colspan="9">|NC ↑ DAC ↑ DDC ↑ TLC ↑ EP ↑ TTC ↑ LK ↑ HC ↑ EC ↑ EPDMS ↑</td></tr><tr><td>Human Agent</td><td></td><td>100 100</td><td>99.8</td><td></td><td>100</td><td>97.8 100</td><td></td><td>100</td><td>98.1</td><td>90.1</td><td>97.9</td></tr><tr><td colspan="10">Perception-based</td><td></td><td></td></tr><tr><td>TransFuser (Chitta et al., 2023) TransFuser-ViRA</td><td>Regression</td><td>|97.1 89.6 97.7</td><td>94.1</td><td>99.0 99.5</td><td>99.8 99.8</td><td>97.9 95.9 98.2 96.7</td><td></td><td>95.998.3 96.8</td><td>98.3</td><td>87.3 87.9</td><td>83.6 89.0 +5.4</td></tr><tr><td>DiffusionDrive (Liao et al., 2025)</td><td>Diffusion</td><td>98.2</td><td>95.9</td><td>99.4</td><td>99.8</td><td>87.5 97.3</td><td></td><td>96.8</td><td>98.3</td><td>87.7</td><td>84.5</td></tr><tr><td colspan="10">DiffusionDrive-ViRA 98.2 96.5 99.6 99.8</td><td>98.3</td><td>88.0 91.9 +7.4</td></tr><tr><td colspan="10"></td></tr><tr><td>Rap* (Feng et al., 2026) Rap*-ViRA</td><td></td><td>|91.6 88.8</td><td>Perception-free</td><td>93.5</td><td>99.8</td><td>87.2 92.5</td><td></td><td>90.9 97.2</td><td></td><td></td><td></td></tr><tr><td></td><td>Scoring</td><td>97.1 96.9</td><td></td><td>99.2</td><td>99.9</td><td>87.5 96.9</td><td></td><td>96.2</td><td>97.7</td><td>56.7 72.8 48.1 83.7 +10.9</td></tr></table>

Table 2: Performance on the NAVSIM v2 navhard benchmark. Rap<sup>∗</sup> is our reimplementation with a different backbone from the official codebase and without the original data augmentation. PDM-Closed plans with GT symbolic inputs, while all other methods use camera data only as input.
<table><tr><td>Methods</td><td>Decoder</td><td colspan="9">|StageNC ↑ DAC ↑ DDC ↑ TLC ↑ EP ↑ TTC ↑ LK ↑ HC ↑ EC ↑ EPDMS ↑</td></tr><tr><td>PDM-Closed</td><td></td><td>S1 S2</td><td>|94.4 98.8 90.5 90.6</td><td>100 95.4</td><td>99.5 98.4</td><td>100 100</td><td>93.5 98.4</td><td>99.3 74.2</td><td>87.7 36.0 91.9 29.7</td><td>56.6</td></tr><tr><td colspan="9"></td></tr><tr><td rowspan="3">TransFuser (Chitta et al., 2023)</td><td rowspan="3">Regression</td><td>S1</td><td>Perception-based |96.4 73.8</td><td>95.6</td><td>99.3</td><td>94.8 95.6</td><td></td><td>88.0 97.6</td><td>82.2</td><td></td></tr><tr><td>S2</td><td>82.6 62.8</td><td>77.9</td><td>98.2</td><td></td><td>91.0 79.9</td><td>40.9</td><td>97.5 80.8</td><td>27.6</td></tr><tr><td>S1</td><td>|97.4 80.0</td><td>98.7</td><td>99.3</td><td>98.0 95.6</td><td></td><td>95.8 97.8</td><td>81.3</td><td>31.7 +4.1</td></tr><tr><td>DiffusionDrive (Liao et al., 2025)</td><td rowspan="3">Diffusion</td><td>S2 S1</td><td>81.2 73.9</td><td>86.0</td><td>98.0</td><td>97.0 78.3</td><td></td><td>47.3 96.1</td><td>71.1</td><td></td></tr><tr><td rowspan="3">DiffusionDrive-ViRA</td><td>S2</td><td>|96.8 86.0 80.172.8</td><td>98.8 84.4</td><td>99.3 98.4</td><td>84.0 95.8 85.976.6</td><td></td><td>96.7 97.6 46.4 96.3</td><td>79.6 72.8</td><td>30.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>S1 S2</td><td>|96.4 87.8 |80.574.1</td><td>99.2 86.4</td><td>99.3 98.4</td><td>97.577.2</td><td>98.5 95.3</td><td>96.4 47.6</td><td>97.6 95.8</td><td>78.7 35.8 +5.3 71.3</td></tr><tr><td colspan="2"></td><td></td><td>Perception-free</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Rap* (Feng et al., 2026)</td><td rowspan="3">Scoring</td><td>S1 S2</td><td>|89.4 71.8</td><td>82.3</td><td>99.6</td><td>86.6 89.6</td><td></td><td>85.8 95.8</td><td>47.1</td><td></td></tr><tr><td></td><td>90.677.6</td><td>85.6</td><td>98.8</td><td>71.4 88.9</td><td></td><td>50.8 98.4</td><td>68.9</td><td>32.8</td></tr><tr><td>S1 S2</td><td>|96.7 90.7 95.5 87.6</td><td>97.2 94.9</td><td>100.0 99.6</td><td>85.2 96.2 69.3 94.5</td><td></td><td>94.2 96.4</td><td>42.2</td><td>49.2 +16.4 68.3</td></tr></table>

Results on NAVSIM v2 navhard. The benefits extend to challenging driving scenarios: all three planners achieve higher EPDMS with ViRA on navhard (Table 2). Notably, although Rap<sup>∗</sup> already achieves the highest baseline EPDMS among the three planners, representation alignment further improves its score from 32.8 to 49.2, a gain of 16.4 points. DAC and DDC also improve, reflecting better drivable-area compliance and driving direction compliance in challenging scenarios. These improvements highlight the value of VFM-guided visual representations even for the strongest evaluated planner in challenging scenarios.

Zero-shot Results on HUGSIM. To assess whether the gains extend beyond NAVSIM, we evaluate NAVSIM-trained planners on the closed-loop HUGSIM benchmark (Zhou et al., 2025a) without further training. As shown in Table 3, ViRA improves the overall HD-Score of TransFuser, DiffusionDrive and Rap<sup>∗</sup> by 2.7, 6.4 and 3.0 points, respectively. Together with the NAVSIM results, these zero-shot gains support our finding that VFM-guided visual representations do improve driving performance across different benchmarks and evaluation settings.

## 3.2 REPRESENTATION QUALITY IMPROVES MEASURABLY

To examine whether the driving gains are accompanied by improved perception, we analyze BEV segmentation and attention maps. Figure 6(a) shows that ViRA improves TransFuser’s BEV segmentation mIoU by 0.5–3.0 points across the evaluated VFM targets. The t-SNE visualizations in Figure 2 suggest that ViRA shifts the representations of TransFuser and Rap<sup>∗</sup> toward those of the DVGT target in the t-SNE embedding space. The qualitative examples in Figure 3 further illustrate improvements in BEV predictions and planned trajectories, together with changes in attention to surrounding road users. These results provide evidence that alignment with high-quality VFM representations improves perception alongside driving performance.

Agent EgoPredicted Trajectory Human-Expert Trajectory  
(a) BEV perception and planning  
Table 3: Zero-shot Performance on the HUGSIM Benchmark. Rap<sup>∗</sup> is our reimplementation with a different backbone from the official codebase and without the original data augmentation.
<table><tr><td rowspan="2">Method</td><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="2">Hard</td><td colspan="2">Extreme</td><td colspan="2">Overall</td></tr><tr><td>RC HD-Score RC HD-Score</td><td></td><td></td><td></td><td></td><td>RC HD-Score</td><td></td><td>RC HD-Score|</td><td></td><td>RC HD-Score</td></tr><tr><td colspan="9">Perception-based</td><td rowspan="3"></td></tr><tr><td>TransFuser (Chitta et al., 2023)</td><td>66.8</td><td>56.7</td><td>28.1</td><td>11.6</td><td>24.5</td><td>9.2</td><td>26.2</td><td>10.9</td><td>36.4 22.1</td></tr><tr><td>TransFuser-ViRA</td><td>72.7</td><td>62.6</td><td>31.9</td><td>15.3</td><td>27.8</td><td>11.4</td><td>25.5</td><td>9.9</td><td>39.5 24.8 +2.7</td></tr><tr><td colspan="10">DiffusionDrive (Liao et al., 2025)</td></tr><tr><td>DiffusionDrive-ViRA</td><td>|68.9 77.5</td><td>56.8 69.3</td><td>29.8 37.7</td><td>12.4 20.7</td><td>25.9 30.0</td><td>9.7 12.0</td><td>25.7 27.9</td><td>9.8 12.3</td><td rowspan="2">|37.6 22.2 43.3 28.6 +6.4</td></tr><tr><td colspan="10"></td></tr><tr><td>Rap* (Feng et al., 2026)</td><td>11.8</td><td>10.6</td><td>11.1</td><td>9.9</td><td>7.8</td><td>6.3</td><td>4.1</td><td>2.3</td><td>8.7 7.3</td></tr><tr><td>Rap*-ViRA</td><td>18.1</td><td>15.9</td><td>16.7</td><td>14.6</td><td>10.5</td><td>8.2</td><td>4.3</td><td>2.3</td><td>12.410.3 +3.0</td></tr></table>

![](images/8145a279b3dcd73aff66547b03890dfd52afea419246d1a006e7932173d2d467.jpg)

![](images/f6ed5b5c6e59910e5ab1c0cc0219054c0603357207a9ca91526d55afc8966429.jpg)  
(b) Visual attention and planning  
Figure 3: Qualitative results showing improved perception and planning with ViRA.

Table 4: VFMs dependency on BEV segmentation of TransFuser. Static objects and pedestrians are omitted from the class-wise results; mIoU is computed over all seven classes.
<table><tr><td>VFM</td><td>Background ↑</td><td>Road ↑</td><td>Walkway ↑</td><td>Centerline ↑</td><td>Vehicle ↑</td><td>mIoU ↑</td></tr><tr><td>Baseline 1</td><td>85.4</td><td>62.3</td><td>56.3</td><td>23.6</td><td>27.1</td><td>36.4</td></tr><tr><td>Random-init</td><td>82.4</td><td>58.3</td><td>47.2</td><td>12.5</td><td>14.7</td><td>30.7-5.7</td></tr><tr><td>DVGT (Zuo et al., 2026a)</td><td>86.9</td><td>64.6</td><td>59.3</td><td>28.9</td><td>33.4</td><td>39.1 +2.7</td></tr></table>

## 3.3 THE GAINS REQUIRE A PRE-TRAINED TARGET

To test whether the gains depend on VFM pre-training, we replace the pre-trained target with a randomly initialized one while retaining the same alignment objectives. As shown in Table 4, randomtarget alignment reduces TransFuser’s BEV segmentation mIoU from 36.4 to 30.7, whereas alignment with pre-trained DVGT increases it to 39.1. The same pattern holds for driving performance: randomtarget alignment lowers EPDMS by 1.6 points for TransFuser and 1.7 points for Rap<sup>∗</sup>, while the pre-trained target improves both planners (Table 5). These results support the importance of pretrained visual knowledge: applying the same alignment objectives to random representations does not reproduce the gains observed with pre-trained VFM representations.

Table 5: Effect of different VFM teachers on TransFuser and Rap\*. Performance on the NAVSIM v2 navtest benchmark.
<table><tr><td>Methods</td><td>VFMs</td><td>NC↑</td><td>DAC ↑</td><td>DDC ↑</td><td>TLC ↑</td><td>LK↑</td><td>EPDMS ↑</td></tr><tr><td colspan="8">Perception-based</td></tr><tr><td rowspan="3">TransFuser (Chitta et al., 2023)</td><td>Baseline</td><td>97.1</td><td>89.6</td><td>99.0</td><td>99.8</td><td>95.9</td><td>83.6</td></tr><tr><td>Random-init</td><td>96.8</td><td>90.2</td><td>99.0</td><td>99.7</td><td>95.5</td><td>82.0 -1.6</td></tr><tr><td>DVGT (Zuo et al., 2026a)</td><td>97.7</td><td>94.1</td><td>99.5</td><td>99.8</td><td>96.8</td><td>89.0 +5.4</td></tr><tr><td colspan="8">Perception-free</td></tr><tr><td rowspan="3">Rap* (Feng et al., 2026)</td><td>Baseline</td><td>91.6</td><td>88.8</td><td>93.5</td><td>99.8</td><td>90.9</td><td>72.8</td></tr><tr><td>Random-init</td><td>94.6</td><td>88.1</td><td>95.1</td><td>99.8</td><td>93.1</td><td>71.1 -1.7</td></tr><tr><td>DVGT (Zuo et al., 2026a)</td><td>97.1</td><td>96.9</td><td>99.2</td><td>99.9</td><td>96.2</td><td>83.7 +10.9</td></tr></table>

## 3.4 GAINS WITHOUT ADDITIONAL INFERENCE COST

A practical question is whether these representation gains require replacing the planner encoder with a larger VFM at inference. Figure 4 compares ViRA with direct VFM backbone replacement in terms of planning performance, inference latency, and parameter count. Latency is measured on a single NVIDIA H100 GPU. DVGTguided alignment improves EPDMS from 72.8 to 83.7 while retaining the baseline latency of 20 ms and parameter count of 69M. This nearly matches the strongest evaluated backbone replacement, DINOv3-L, which achieves 83.8 EPDMS but requires 65 ms and 351M parameters. Since the VFM and alignment heads are used only during training, ViRA enables the planner to benefit from pre-trained visual representations without changing its deployed architecture or increasing inference cost.

![](images/b87a060baec08092be2db679c1732d566b8cecda8a5aa49f308ee41bba21a7e6.jpg)  
Figure 4: ViRA (star) vs. backbone replacement (circles). Bubble size indicates parameter count.

## 4 PLANNING GAINS DEPEND ON BOTH VFMS AND PLANNERS

## 4.1 THE CHOICE OF VFM TARGET MATTERS

Table 6: Rap<sup>∗</sup> (Feng et al., 2026) with DINOv3 (Siméoni et al., 2025) backbones. Performance on the NAVSIM v2 navtest benchmark under different VFM teachers.
<table><tr><td>Backbone</td><td>Teachers</td><td>NC↑</td><td>DAC ↑</td><td>DDC ↑</td><td>TLC↑</td><td>LK↑</td><td>EPDMS ↑</td></tr><tr><td>DINOv3-vit16l</td><td></td><td>97.2</td><td>96.4</td><td>98.9</td><td>99.9</td><td>96.1</td><td>83.8</td></tr><tr><td>DINOv3-vit16l</td><td>VGGT (Wang et al., 2025)</td><td>98.5</td><td>97.3</td><td>99.3</td><td>100.0</td><td>96.7</td><td>86.3 +2.5</td></tr></table>

As shown in Figure 1(c-ii), changing only the representation alignment target for the perception-free Rap<sup>∗</sup> yields EPDMS scores ranging from 79.4 to 84.9, a spread of 5.5 points. This difference is comparable to the gains from many method-level improvements. These results highlight the importance of alignment target selection, with DINOv3, which learns from more general data, providing the largest planning gain in this setting. To further examine whether VFMs with different pre-training objectives provide complementary representations, we use Rap<sup>∗</sup> with a DINOv3-L backbone as the baseline and introduce alignment to VGGT, a VFM trained for 3D geometry.

As shown in Table 6, VGGT alignment improves EPDMS by 2.5 points, from 83.8 to 86.3. Thus, even when the backbone already provides strong semantic representations, alignment to a geometryfocused VFM can further improve planning performance. These results suggest that alignment target selection deserves careful consideration, as its benefits may depend on the complementarity between the target VFM and the planner’s existing representations.

## 4.2 AUXILIARY SUPERVISION SHAPES SENSITIVITY TO VFM TARGETS

Extending the experiments from the preceding subsection to perception-based planners, we find that TransFuser and DiffusionDrive are less sensitive to the choice of VFM target. Across VFM targets, EPDMS varies by only 0.5 points for TransFuser (88.8 to 89.3) and 0.6 points for DiffusionDrive (91.3 to 91.9), compared with 5.5 points for the perception-free $\mathrm { { R a p } ^ { * } }$ . One important difference between these planners is the use of auxiliary perception supervision. TransFuser and DiffusionDrive learn planner representations with BEV segmentation and agent detection supervision, whereas Rap<sup>∗</sup> does not. We therefore hypothesize that auxiliary perception supervision reduces the sensitivity of the planner to VFM target selection.

![](images/2d9f0fee96a73a58359ee5fdd2f9cf3a12cd7f9d2a10f60f762e75078add90ff.jpg)  
Figure 5: Different VFM targets across planners.

To test this hypothesis, we compare TransFuser (Chitta et al., 2023) with and without auxiliary perception supervision across five alignment targets: DVGT (Zuo et al., 2026a), VGGT (Wang et al., 2025), DA3 (Lin et al., 2025), DINOv3 (Siméoni et al., 2025), and SAM3 (Carion et al., 2025). As shown in Figure 6(b), EPDMS ranges from 88.8 to 89.3 with auxiliary supervision. Removing this supervision expands the range to 86.4 to 89.1, increasing the spread from 0.5 to 2.7 points. Examining the endpoints of these ranges provides further insight: after removing auxiliary supervision, the maximum score remains nearly unchanged (89.3 to 89.1), whereas the minimum drops substantially (88.8 to 86.4). This indicates that auxiliary supervision mainly compensates for the less effective targets. Across these targets, TransFuser’s BEV segmentation results under auxiliary supervision show a broadly similar ordering to its planning results without auxiliary supervision. Overall, these controlled comparisons show that auxiliary perception supervision reduces TransFuser’s sensitivity to VFM target selection.

![](images/6a28281043b145e4342e3f4ee5c3e14709e081e3875b4a6a7d23330cf97447d2.jpg)  
(a) BEV segmentation results

![](images/25855586011b6111fcc210954d9b18983683e869b8aeb6eeaee974e56ead153f.jpg)  
(b) The gain of w/ and w/o perception supervision  
Figure 6: Auxiliary perception supervision narrows the spread of planning gains across VFM targets. (a) BEV mIoU across VFM targets; the dashed line marks the baseline (36.4). (b) Planning gain ∆EPDMS with and without auxiliary perception supervision (baselines 83.6 and 82.6, respectively).

Table 7: Comparison with state-of-the-art methods on the NAVSIM v2 navtest benchmark. ViRA-Diffusion is DiffusionDrive (Liao et al., 2025) trained with ViRA and without auxiliary perception supervision.
<table><tr><td>Method</td><td>|NC ↑ DAC ↑ DDC ↑ TLC ↑ EP↑ TTC ↑ LK ↑ HC ↑ EC ↑ EPDMS ↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Epona (Zhang et al., 2025)</td><td>97.1 95.7</td><td></td><td>99.3</td><td>99.7</td><td>88.6</td><td>96.3</td><td>97.0</td><td>98.0</td><td>67.8 85.1</td></tr><tr><td>DiffusionDriveV2 (Zou et al., 2025)</td><td>97.7</td><td>96.6</td><td>99.2</td><td>99.8 88.9</td><td>97.2</td><td>96.0</td><td>97.8</td><td>91.0</td><td>87.5</td></tr><tr><td>Latent-WAM (Wang et al., 2026b)</td><td>98.1 97.3</td><td>99.6</td><td>99.8</td><td>87.7</td><td>97.3</td><td>97.6</td><td>98.1</td><td>87.3</td><td>89.3</td></tr><tr><td>DriveFuture (Hong et al., 2026)</td><td>98.8</td><td>99.1</td><td>99.6</td><td>99.9</td><td>86.6</td><td>98.4</td><td>96.4 98.3</td><td>74.8</td><td>89.9</td></tr><tr><td>SparseDriveV2 (Sun et al., 2026)</td><td>98.1</td><td>98.1 99.6</td><td></td><td>99.8 91.1</td><td>97.3</td><td>96.9</td><td>98.2</td><td>78.4</td><td>90.1</td></tr><tr><td>Discrete-WAM (Yao et al., 2026)</td><td>98.5</td><td>98.2</td><td>99.7</td><td>99.8</td><td>90.5</td><td>97.9</td><td>97.2 98.3</td><td>78.1</td><td>90.4</td></tr><tr><td>ViRA-Diffusion</td><td>98.3 96.8</td><td>99.5</td><td></td><td>99.8 98.2</td><td>97.5</td><td>97.4</td><td>98.3</td><td>88.0</td><td>92.3</td></tr></table>

## 4.3 PRACTICAL IMPLICATIONS

The preceding analysis suggests that a planner can maintain comparable planning performance without auxiliary perception supervision when aligned with a suitable VFM target. Building on these findings, we develop ViRA-Diffusion, a diffusion-based planner (Liao et al., 2025) trained with ViRA using DINOv3 as the target and no auxiliary perception supervision. As shown in Table 7, the resulting ViRA-Diffusion achieves 92.3 EPDMS on NAVSIM v2 navtest, comparing favorably with the recent results included in our comparison. This result illustrates the practical value of our analysis in guiding the joint choice of VFM targets and supervision strategies for end-to-end planners.

## 5 RELATED WORK

End-to-End Autonomous Driving. End-to-end autonomous driving learns planning and control directly from sensory inputs within a unified framework (Hu et al., 2023; Yu et al., 2025a; Sun et al., 2025). Joint optimization can mitigate cascading errors between separately trained modules and facilitate learning from large-scale driving data, making end-to-end autonomous driving a promising route toward scalable real-world deployment (Chen et al., 2024a; Jia et al., 2024; Guan et al., 2026). Recent studies focus on improving policy heads (Liao et al., 2025; Gao et al., 2025; Zheng et al., 2026; Yang et al., 2026), introducing auxiliary perception supervision (Strong et al., 2026; Chen et al., 2024b; Xing et al., 2025), and scaling training data through synthetic scene generation and augmentation (Yang et al., 2023; Zhou et al., 2025b; Xu et al., 2025; Li et al., 2025b; Tian et al., 2026; Li et al., 2026b;c). Despite this progress, how visual representation quality contributes to planning gains remains insufficiently understood. Some recent methods enhance visual representations for driving with semantic and geometric priors from pre-trained VFMs (Zheng et al., 2025; Wang et al., 2026b). These methods demonstrate benefits within specific systems, but provide limited guidance on how the gains vary across VFM targets and planner designs. Complementing these directions, we study when better visual representations improve planning performance.

Representation Learning for Driving. Learning visual representations that capture scene semantics and spatial structure is crucial for end-to-end autonomous driving (Chen et al., 2024a). To enrich these representations, recent driving studies turn to VFMs pre-trained on large-scale visual data, drawing on their capabilities in semantics (Kirillov et al., 2023; Carion et al., 2025), depth (Yang et al., 2024a;b; Lin et al., 2025), and 3D geometry (Zuo et al., 2026a; Wang et al., 2024; 2025). These VFMs are used through visual backbone replacement (Feng et al., 2026; Naeinian et al., 2026), auxiliary supervision from VFM predictions (Jia et al., 2025; Strong et al., 2026), or feature integration within task-specific architectures and training objectives (Zheng et al., 2025; Wang et al., 2026b;a; Xie et al., 2026). Recent studies show that representation alignment, which supervises intermediate features with frozen visual representations, can improve training efficiency and task performance (Yu et al., 2025b; Li et al., 2026a). iREPA (Singh et al., 2026) further shows that the spatial structure of target representations predicts image generation quality better than their ImageNet classification accuracy. In end-to-end autonomous driving, how planner architecture and auxiliary perception supervision shape the benefits of VFM-guided representations remains unclear. We use representation alignment as a controlled experimental interface to study these interactions across planners and VFM targets.

## 6 CONCLUSION AND OUTLOOK

In this paper, we systematically study when VFM-guided visual representations improve end-to-end autonomous driving, using ViRA as an experimental interface that keeps the planner’s architecture and inference cost unchanged. Experiments across planners and benchmarks show consistent driving gains, with benefits extending to zero-shot closed-loop evaluation. VFM target selection affects these gains, and alignment to a different VFM can further improve planners that already use pre-trained VFM backbones. Controlled comparisons on TransFuser further show that auxiliary perception supervision narrows performance differences across VFM targets. These results highlight the importance of considering VFM target selection and planner supervision jointly. Guided by this analysis, we develop ViRA-Diffusion with DINOv3 alignment and no auxiliary perception supervision, achieving 92.3 EPDMS on NAVSIM v2 navtest. Future work will examine whether these findings extend to additional planner architectures and other physical intelligence tasks involving decision-making.

## ACKNOWLEDGMENTS

We would like to thank Fanghua Yu, Jiazhi Yang, and other members of OpenDriveLab for their valuable discussions and technical support.

## REFERENCES

Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. Pseudosimulation for autonomous driving. In CoRL, 2025.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-toend autonomous driving: Challenges and frontiers. IEEE TPAMI, 2024a.

Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243, 2024b.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE TPAMI, 2023.

Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. Navsim: Data-driven nonreactive autonomous vehicle simulation and benchmarking. In NeurIPS, 2024.

Lan Feng, Yang Gao, Eloi Zablocki, Quanyi Li, Wuyang Li, Sichao Liu, Matthieu Cord, and Alexandre Alahi. Rap: 3d rasterization augmented end-to-end planning. In ICLR, 2026.

Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, Ying Zhang, Wenyu Liu, Qian Zhang, and Xinggang Wang. Rad: Training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. In NeurIPS, 2025.

Yanchen Guan, Xingcheng Liu, Bin Rao, Chengyue Wang, Guofa Li, Yunjian Li, Lishengsa Yue, Zhiyong Cui, Chengzhong Xu, and Zhenning Li. Planning-oriented end-to-end autonomous driving: Architectures, evaluation, and emerging paradigms, 2026.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

Yufeng Hong, Xiaotian Zhou, Yingyan Li, Xiangpo Zhou, Lin Liu, Yadan Luo, Shaoqing Xu, Lei Yang, and Ziying Song. Drivefuture: Future-aware latent world models for autonomous driving, 2026.

Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In CVPR, 2023.

Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, and Junchi Yan. Bench2drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In NeurIPS, 2024.

Xiaosong Jia, Yanhao Liu, Junqi You, Renqiu Xia, Yu Hong, and Junchi Yan. Drivevggt: Visual geometry transformer for autonomous driving. arXiv preprint arXiv:2511.22264, 2025.

Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8306–8316. IEEE, 2023.

Napat Karnchanachari, Dimitris Geromichalos, Kok Seang Tan, Nanxiang Li, Christopher Eriksen, Shakiba Yaghoubi, Noushin Mehdipour, Gianmarco Bernasconi, Whye Kit Fong, Yiluan Guo, et al. Towards learning-based planning: The nuplan benchmark for real-world autonomous driving. In ICRA, 2024.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023.

Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial forcing: Implicit spatial representation alignment for vision-language-action model. In ICLR, 2026a.

Kailin Li, Zhenxin Li, Shiyi Lan, Yuan Xie, Zhizhong Zhang, Jiayi Liu, Zuxuan Wu, Zhiding Yu, and Jose M. Alvarez. Hydra-mdp++: Advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820, 2025a.

Shihao Li, Naisheng Ye, Tianyu Li, Kashyap Chitta, Tuo An, Peng Su, Boyang Wang, Haiou Liu, Chen Lv, and Hongyang Li. Optimization-guided diffusion for interactive scene generation. In ECCV, 2026b.

Tianyu Li, Yihang Qiu, Zhenhua Wu, Carl Lindström, Peng Su, Matthias Nießner, and Hongyang Li. Mtgs: Multi-traversal gaussian splatting. arXiv preprint arXiv:2503.12552, 2025b.

Tianyu Li, Li Chen, Caojun Wang, Haochen Liu, Kashyap Chitta, Zhenjie Yang, Yuhang Lu, Naisheng Ye, Yihang Qiu, Yufei Wang, et al. World engine: Towards the era of post-training for autonomous driving, 2026c.

Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE TPAMI, 2024.

Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In CVPR, 2025.

Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

Fatemeh Naeinian, Ali Hamza, Haoran Zhu, and Anna Choromanska. Zero-shot cross-city generalization in end-to-end autonomous driving: Self-supervised versus supervised representations. arXiv preprint arXiv:2603.11417, 2026.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

OpenScene Contributors. Openscene: The largest up-to-date 3d occupancy prediction benchmark in autonomous driving. https://github.com/OpenDriveLab/OpenScene, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In ECCV, 2024.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Jaskirat Singh, Xingjian Leng, Zongze Wu, Liang Zheng, Richard Zhang, Eli Shechtman, and Saining Xie. What matters for representation alignment: Global information or spatial structure? In ICLR, 2026.

Matthew Strong, Wei-Jer Chang, Quentin Herau, Jiezhi Yang, Yihan Hu, Chensheng Peng, and Wei Zhan. Learning to drive is a free gift: Large-scale label-free autonomy pretraining from unposed in-the-wild videos. In CVPR, 2026.

Wenchao Sun, Xuewu Lin, Yining Shi, Chuang Zhang, Haoran Wu, and Sifa Zheng. Sparsedrive: End-to-end autonomous driving via sparse scene representation. In ICRA, 2025.

Wenchao Sun, Xuewu Lin, Keyu Chen, Zixiang Pei, Xiang Li, Yining Shi, and Sifa Zheng. Sparsedrive-v2: Scoring is all you need for end-to-end autonomous driving, 2026.

Haochen Tian, Tianyu Li, Haochen Liu, Jiazhi Yang, Yihang Qiu, Guang Li, Junli Wang, Yinfeng Gao, Zhang Zhang, Liang Wang, Hangjun Ye, Long Chen, and Hongyang Li. Simscale: Learning to drive via real-world simulation at scale. In CVPR, 2026.

Frederick Tung and Greg Mori. Similarity-preserving knowledge distillation. In ICCV, 2019.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025.

Jie Wang, Guang Li, Zhijian Huang, Chenxu Dang, Hangjun Ye, Yahong Han, and Long Chen. Vggdrive: Empowering vision-language models with cross-view geometric grounding for autonomous driving. In CVPR, 2026a.

Linbo Wang, Yupeng Zheng, Qiang Chen, Shiwei Li, Yichen Zhang, Zebin Xing, Qichao Zhang, Xiang Li, Deheng Qian, Pengxuan Yang, et al. Latent-wam: Latent world action modeling for end-to-end autonomous driving. arXiv preprint arXiv:2603.24581, 2026b.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024.

Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling. In ICLR, 2026.

Chengen Xie, Bin Sun, Tianyu Li, Junjie Wu, Zhihui Hao, XianPeng Lang, and Hongyang Li. Latentvla: Efficient vision-language models for autonomous driving via latent action prediction. arXiv preprint arXiv:2601.05611, 2026.

Zebin Xing, Xingyu Zhang, Yang Hu, Bo Jiang, Tong He, Qian Zhang, Xiaoxiao Long, and Wei Yin. Goalflow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In CVPR, 2025.

Zhiyuan Xu, Bohan Li, Huan-ang Gao, Mingju Gao, Yong Chen, Ming Liu, Chenxu Yan, Hang Zhao, Shuo Feng, and Hao Zhao. Challenger: Affordable adversarial driving video generation. arXiv preprint arXiv:2505.15880, 2025.

Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024a.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In NeurIPS, 2024b.

Zhenjie Yang, Xiaosong Jia, Hongyang Li, and Junchi Yan. Llm4drive: A survey of large language models for autonomous driving, 2023.

Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, and Junchi Yan. Drivemoe: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving. In CVPR, 2026.

Ziyang Yao, Haochen Liu, Yuncheng Jiang, Zeyu Zhu, Zibin Guo, Jingru Wang, Tianle Liu, Jianwei Cui, Kuiyuan Yang, Hongwei Xie, et al. Discrete-wam: Unified discrete vision-action token editing for world-policy learning, 2026.

Rui Yu, Xianghang Zhang, Runkai Zhao, Huaicheng Yan, and Meng Wang. Distilldrive: End-to-end multi-mode autonomous driving distillation by isomorphic hetero-source planning model. In ICCV, 2025a.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In ICLR, 2025b.

Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928, 2016.

Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiao-Xiao Long, et al. Epona: Autoregressive diffusion world model for autonomous driving. In ICCV, 2025.

Yinan Zheng, Tianyi Tan, Bin Huang, Enguang Liu, Ruiming Liang, Jianlin Zhang, Jianwei Cui, et al. Unleashing the potential of diffusion models for end-to-end autonomous driving. arXiv preprint arXiv:2602.22801, 2026.

Yupeng Zheng, Pengxuan Yang, Zebin Xing, Qichao Zhang, Yuhang Zheng, Yinfeng Gao, Pengfei Li, Teng Zhang, Zhongpu Xia, Peng Jia, et al. World4drive: End-to-end autonomous driving via intention-aware physical latent world model. In ICCV, 2025.

Hongyu Zhou, Longzhong Lin, Jiabao Wang, Yichong Lu, Dongfeng Bai, Bingbing Liu, Yue Wang, Andreas Geiger, and Yiyi Liao. Hugsim: A real-time, photo-realistic and closed-loop simulator for autonomous driving. IEEE TPAMI, 2025a.

Yunsong Zhou, Naisheng Ye, William Ljungbergh, Tianyu Li, Jiazhi Yang, Zetong Yang, Hongzi Zhu, Christoffer Petersson, and Hongyang Li. Decoupled diffusion sparks adaptive scene generation. In ICCV, 2025b.

Jialv Zou, Shaoyu Chen, Bencheng Liao, Zhiyu Zheng, Yuehao Song, Lefei Zhang, Qian Zhang, Wenyu Liu, and Xinggang Wang. Diffusiondrive-v2: Reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving, 2025.

Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Shengyin Jiang, Long Chen, Zhi-Xin Yang, and Jiwen Lu. Dvgt: Driving visual geometry transformer. In CVPR, 2026a.

Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Hanbing Li, Long Chen, Zhi-Xin Yang, and Jiwen Lu. Dvgt-2: Vision-geometry-action model for autonomous driving at scale. arXiv preprint arXiv:2604.00813, 2026b.

## Appendix

The appendix is organized as follows. Appendix A provides the method details. Appendix B provides evaluation and implementation details. Appendix C reports additional quantitative results. Appendix D presents additional qualitative visualizations and failure cases. Appendix E discusses limitations and broader impact. Appendix F lists the licenses of the assets used in this work.

## A VFM-GUIDED REPRESENTATION ALIGNMENT

We present the ViRA framework as follows. Section 2 defines the notation and problem setup. Appendix A.1 provides an overview of the ViRA framework for E2E planners. Appendix A.2 introduces Spatial Dependency Alignment (SDA) and Scene Semantic Alignment (SSA). Appendix A.3 summarizes the training objective.

![](images/f7f1b73995ab692e5649829bc0d30a553cdbc6c1aeadbdad04e913fcbfc88c83.jpg)  
Figure S.1: Pipeline of ViRA. Multi-view images are encoded by both the planner encoder and a frozen VFM target, producing student and VFM representations. The VFM representations supervise latent alignment only, without being injected into the policy head. The policy head operates solely on planner representations, and the VFM branch is removed after training, yielding no inference overhead. Q, K, and V denote query, key, and value; KL denotes Kullback–Leibler divergence; CE denotes cross-entropy.

## A.1 VFM-GUIDED E2E PLANNER

As illustrated in Figure S.1, given a single-frame multi-view observation O, a conventional E2E planner first extracts an intermediate representation $\mathbf { F } ^ { s }$ using the planner encoder $E _ { \theta }$ , and then predicts the future trajectory $\hat { \tau }$ with a driving policy $\pi _ { \psi }$

$$
{ \bf F } ^ { s } = E _ { \theta } ( { \mathcal O } ) , \qquad \hat { \boldsymbol { \tau } } = \pi _ { \psi } ( { \bf F } ^ { s } ) .\tag{S.1}
$$

Here, $\pi _ { \psi }$ denotes a general policy head, which can be instantiated by different E2E planner designs, such as regression-based (Chitta et al., 2023), diffusion-based (Liao et al., 2025), or scoringbased (Feng et al., 2026) policies.

To improve the representation quality of the planner encoder, we additionally feed the same observation O into a pre-trained VFM $T _ { \phi }$ to obtain a VFM representation $\mathbf { F } ^ { t } = T _ { \phi } ( \mathcal { O } )$ , where ϕ is frozen throughout training. The VFM representation is used only to compute the representation alignment loss and is then discarded, without being passed to the policy head. Unless otherwise specified, we use DVGT (Zuo et al., 2026a) as the default VFM target $\dot { T } _ { \phi } ,$ a VFM pre-trained on autonomous driving data, while our framework is also compatible with other VFMs (Carion et al., 2025; Wang et al., 2025; Siméoni et al., 2025; Lin et al., 2025). Importantly, the VFM and alignment heads are used only during training and discarded during inference, keeping the original planner architecture and inference pipeline unchanged with no additional inference cost. Since ViRA uses frozen VFM representations as supervision, it requires no extra annotations.

## A.2 LATENT REPRESENTATION ALIGNMENT

To transfer driving-relevant knowledge from VFMs to planner encoders, we align their representations from two complementary perspectives: spatial distribution and scene semantic structure. As shown in Figure S.1, Spatial Dependency Alignment (SDA) aligns the spatial attention distribution of the planner representation with that of the VFM representation, encouraging the planner to capture fine-grained spatial dependencies. Scene Semantic Alignment (SSA) further aligns the global latent representations of the E2E planner and the VFMs through a bidirectional contrastive objective, preserving scene-level semantic consistency across training samples. Together, these two objectives improve the representation quality of the E2E planner encoder for downstream trajectory planning.

Spatial Dependency Alignment. SDA aims to transfer the spatial relational structure encoded by the VFM to the E2E planner representation. Different from directly matching feature values at each spatial location, we align the spatial dependency distributions between planner and VFM features, following the spirit of attention-based knowledge distillation (Zagoruyko & Komodakis, 2016; Tung & Mori, 2019). Given projected and flattened planner and VFM features, we define the SDA loss as

$$
\mathcal { L } _ { \mathrm { S D A } } = \mathbb { E } _ { b , i } \left[ D _ { \mathrm { K L } } \left( \mathbf { A } _ { b , i , : } ^ { t } \parallel \mathbf { A } _ { b , i , : } ^ { s } \right) \right] ,\tag{S.2}
$$

where the expectation is taken over all samples in the batch and all spatial tokens. Here, $D _ { \mathrm { K I } }$ denotes the Kullback–Leibler divergence, $\mathbf { A } ^ { s } \ = \ \mathrm { s o f t m a x } ( \mathbf { Q } ^ { s } ( \mathbf { K } ^ { t } ) ^ { \top } / \gamma _ { s d a } )$ and $\begin{array} { r l } { \mathbf { A } ^ { t } } & { { } = } \end{array}$ softmax $( \mathbf { Q } ^ { t } ( \mathbf { K } ^ { t } ) ^ { \top } / \gamma _ { s d a } )$ denote the planner-to-VFM dependency distribution and the VFM selfdependency distribution, respectively. $\mathbf { \bar { Q } } ^ { s }$ and $\mathbf { Q } ^ { t }$ are query projections of the planner and VFM features, and $\dot { \mathbf { K } } ^ { t }$ is the key projection of the VFM features. We set the SDA temperature to $\gamma _ { \mathrm { s d a } } = 0 . 0 7$ This objective encourages the planner encoder to reproduce the spatial dependencies distilled from the frozen VFM.

Scene Semantic Alignment. SSA encourages the planner representation to preserve global semantic consistency with the VFM, following the standard contrastive learning formulation (Oord et al., 2018; Chen et al., 2020; Radford et al., 2021). After aggregating the projected planner and VFM features into global embeddings $\mathbf { z } ^ { s }$ and $\mathbf { z } ^ { t }$ , we apply a bidirectional contrastive loss:

$$
\mathcal { L } _ { \mathrm { S S A } } = \frac { 1 } { 2 } \left[ \mathrm { C E } ( \mathbf { M } , \mathbf { y } ) + \mathrm { C E } ( \mathbf { M } ^ { \top } , \mathbf { y } ) \right] ,\tag{S.3}
$$

where CE denotes the cross-entropy loss, ${ \bf M } = { \bf z } ^ { s } ( { \bf z } ^ { t } ) ^ { \top } / \gamma _ { \mathrm { s s a } }$ is the planner–VFM similarity matrix, $\mathbf { y } = [ 1 , \dots , B ]$ denotes the matched scene indices within the batch, with B being the batch size. We set the SSA temperature to $\gamma _ { \mathrm { s s a } } = 1 . 0$ . The row-wise and column-wise cross-entropy terms enforce planner-to-VFM and VFM-to-planner matching, respectively.

## A.3 TRAINING OBJECTIVE

Different E2E planners use different training objectives depending on their architectural designs. Some are trained with trajectory supervision only, while others include auxiliary perception objectives, such as detection, segmentation, or depth estimation. ViRA is compatible with both settings: we keep the original E2E planner architecture and training protocol unchanged, and simply add representation alignment losses as representation-level supervision. Let $\mathcal { L } _ { \mathrm { p l a n n e r } }$ denote the original training objective of a planner, including its trajectory loss and any native auxiliary supervision if used. The overall training objective is:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p l a n n e r } } + \lambda _ { \mathrm { S D A } } \mathcal { L } _ { \mathrm { S D A } } + \lambda _ { \mathrm { S S A } } \mathcal { L } _ { \mathrm { S S A } } . } \end{array}\tag{S.4}
$$

Here, $\mathcal { L } _ { \mathrm { S D A } }$ and $\mathcal { L } _ { \mathrm { S S A } }$ denote the proposed spatial and semantic latent alignment losses, respectively. Motivated by recent findings that spatial structure is more critical than global semantics in representation alignment (Singh et al., 2026), we set $\lambda _ { \mathrm { S D A } } = 1$ and $\lambda _ { \mathrm { S S A } } = 0 . 1$ . For planning-only planners, $\mathcal { L } _ { \mathrm { p l a n n e r } }$ contains only the trajectory prediction objective. For planners with auxiliary objectives, the original auxiliary supervision terms are included in $\mathcal { L } _ { \mathrm { p l a n n e r } }$ and kept unchanged.

Table S.1: Model and training hyperparameters.
<table><tr><td>Hyperpara.</td><td>TransFuser (Chitta et al., 2023)</td><td>DiffusionDrive (Liao et al., 2025)</td><td>Rap* (Feng et al., 2026)</td></tr><tr><td colspan="4">Model Configuration</td></tr><tr><td>Sensors</td><td> $3 \times \mathbf { C a m } .$ </td><td>3 × Cam.</td><td>4 × Cam.</td></tr><tr><td>Resolution</td><td>2048 × 512</td><td>2048 × 512</td><td>4 × 768 × 448</td></tr><tr><td>Horizon</td><td>4s</td><td>4s</td><td>5s</td></tr><tr><td>Frequency</td><td>2Hz</td><td>2Hz</td><td>2Hz</td></tr><tr><td>Backbone</td><td>R34 (He et al., 2016)</td><td>R34 (He et al., 2016)</td><td>R34 (He et al., 2016)</td></tr><tr><td>Parameters</td><td>56M</td><td>61M</td><td>69M</td></tr><tr><td>Aux. Tasks</td><td>Det. Seg.</td><td>Det. Seg.</td><td>None</td></tr><tr><td colspan="4">Training Configuration</td></tr><tr><td>GPUs</td><td>8 × H100</td><td>8 × H100</td><td>8 × H100</td></tr><tr><td>Epochs</td><td>100</td><td>100</td><td>30</td></tr><tr><td>Total BS</td><td>192</td><td>192</td><td>48</td></tr><tr><td>Initial LR</td><td>1 × 10−4</td><td>6 × 10−4</td><td>1 × 10−4</td></tr><tr><td>Schedule</td><td>Constant</td><td>Cosine Decay</td><td>Cosine Decay</td></tr><tr><td>Optimizer</td><td>Adam (Kingma &amp; Ba, 2015)</td><td>AdamW (Loshchilov &amp; Hutter, 2019) Adam (Kingma &amp; Ba, 2015)</td><td></td></tr></table>

## B IMPLEMENTATION AND EVALUATION DETAILS

## B.1 BENCHMARK AND METRICS

NAVSIM. We evaluate driving performance on navtest and navhard using NAVSIM v2 (Cao et al., 2025). The navtest evaluation covers 12,146 real-world scenarios in a single stage, providing a broad assessment across driving conditions. For navhard, we follow the same two-stage evaluation setup as SimScale (Tian et al., 2026), using 244 challenging real-world scenarios and 4,164 synthetic followup scenarios rendered with 3D Gaussian Splatting. Both evaluations use the Extended Predictive Driver Model Score (EPDMS) (Li et al., 2025a; Cao et al., 2025), which combines multiplicative penalties with a weighted average of driving-quality scores.

$$
\mathrm { E P D M S } = \underbrace { \prod _ { m \in \mathcal { M } _ { \mathrm { p e n } } } S _ { m } } _ { \mathrm { p e n a l t i e s } } . \underbrace { \frac { \sum _ { m \in \mathcal { M } _ { \mathrm { a v g } } } w _ { m } S _ { m } } { \sum _ { m \in \mathcal { M } _ { \mathrm { a v g } } } w _ { m } } } _ { \mathrm { w e i g h t e d a v e r a g e } } ,\tag{S.5}
$$

where $S _ { m }$ denotes the evaluated subscore for metric m, including human-reference filtering where enabled, and $w _ { m }$ is its weight. The penalty set $\mathcal { M } _ { \mathrm { p e n } }$ comprises No-at-fault Collisions (NC), Drivable Area Compliance (DAC), Driving Direction Compliance (DDC), and Traffic Light Compliance (TLC). The weighted set $\mathcal { M } _ { \mathrm { a v g } }$ comprises Time-to-Collision (TTC), Ego Progress (EP), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC). Human-reference filtering assigns a subscore of one when the corresponding human-reference score is zero. For navhard, evaluation uses reactive traffic agents and combines scores from the initial and synthetic follow-up scenarios through two-stage aggregation.

HUGSIM. HUGSIM (Zhou et al., 2025a) is a real-time, photorealistic closed-loop simulator that reconstructs 3D scenes from captured RGB images with 3D Gaussian Splatting and updates ego and actor states at each step from the planner output, so that surrounding vehicles react to the ego. We evaluate planners trained only on NAVSIM in a zero-shot manner on scenarios grouped into four difficulty levels: Easy, Medium, Hard, and Extreme. Each episode is scored by HD-Score (Zhou et al., 2025a), reported on a 0–100 scale per level and on average:

$$
\mathrm { H D - S c o r e } = \mathrm { R C } \times \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \mathrm { N C } _ { t } \times \mathrm { D A C } _ { t } \times \frac { 5 \cdot \mathrm { T T C } _ { t } + 2 \cdot \mathrm { C o m f } _ { t } } { 7 } \right) ,\tag{S.6}
$$

where RC is the fraction of the intended route completed within an episode of T steps, NC and DAC are defined as above, and TTC and Comfort (Comf.) take values in [0, 1].

mIoU. We measure the semantics encoded in planner representations with the BEV segmentation head of TransFuser. Mean Intersection-over-Union (mIoU) is the unweighted average of per-class IoU

over seven classes: background, road, walkways, centerline, static objects, vehicles, and pedestrians.

$$
\mathrm { I o U } _ { c } = \frac { | P _ { c } \cap G _ { c } | } { | P _ { c } \cup G _ { c } | } , \quad \mathrm { m I o U } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathrm { I o U } _ { c } ,\tag{S.7}
$$

where $C = 7 , P _ { c }$ is the set of BEV cells predicted as class c over the evaluation set, and $G _ { c }$ is the corresponding ground-truth set. The main-text table reports class-wise IoU only for background, road, walkways, centerline, and vehicles. Static objects and pedestrians remain below 0.5 IoU for every evaluated target, so they are omitted from that table; the reported mIoU still averages all seven classes. Table S.6 lists the score of every class.

Table S.2: Effect of alignment objectives on navtest. MSE, Cosine, KL, and Contrastive denote mean squared error, cosine similarity, Kullback–Leibler divergence, and contrastive loss, respectively.
<table><tr><td>√</td><td>x</td><td>x</td><td>MSE Cosine KL Contrastive x</td><td>DAC DDC 90.9 98.6</td><td>TLC 99.8</td><td>LK 92.3</td><td>EPDMS ↑ 84.4</td></tr><tr><td>x</td><td>√</td><td>x</td><td>x</td><td>90.8 99.0</td><td>99.8</td><td>95.6</td><td>85.1</td></tr><tr><td>x</td><td>x</td><td>√</td><td>x</td><td>90.3 98.9</td><td>99.7</td><td>94.6</td><td>84.3</td></tr><tr><td>x</td><td>x</td><td>x</td><td>√</td><td>90.4 98.9</td><td>99.8</td><td>95.8</td><td>84.6</td></tr><tr><td>√</td><td>√</td><td>x</td><td>x</td><td>89.6 99.0</td><td>99.7</td><td>95.9</td><td>83.6</td></tr><tr><td>√</td><td>x</td><td>√</td><td>x</td><td>92.9 99.4</td><td>99.8</td><td>95.8</td><td>87.8</td></tr><tr><td>x</td><td>√</td><td>x</td><td>√</td><td>93.5 99.4</td><td>99.8</td><td>96.4</td><td>88.1</td></tr><tr><td>x</td><td>x</td><td>√</td><td>√</td><td>94.1 99.5</td><td>99.8</td><td>96.8</td><td>89.0</td></tr></table>

## B.2 TRAINING DETAILS

This section complements the experimental setup in Section 2.2 by providing the model configurations and training hyperparameters for all evaluated planners. As summarized in Table S.1, we follow the NAVSIM (Dauner et al., 2024) training protocol unless otherwise specified, and train all models on the training logs (Karnchanachari et al., 2024) of navtrain. Each planner is trained with its corresponding official settings, while ViRA leaves the planner architecture and inference pipeline unchanged.

TransFuser (Chitta et al., 2023) is a regression-based planner built on fused sensory representations. It uses three camera views with a resolution of $2 0 4 8 \times 5 1 2$ , predicts a 4s future trajectory at 2Hz, and adopts ResNet-34 (He et al., 2016) as the image backbone. The model contains 56M parameters and uses detection and segmentation as auxiliary tasks. We train TransFuser for 100 epochs on 8 H100 GPUs with a total batch size of 192, an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , a constant learning-rate schedule, and the Adam optimizer (Kingma & Ba, 2015).

DiffusionDrive (Liao et al., 2025) is a generative planner that predicts future trajectories through diffusion-based trajectory modeling. It uses the same three-camera input resolution, prediction horizon, and frequency as TransFuser, i.e., 2048 × 512 images, a 4s horizon, and 2Hz trajectory frequency. It also adopts ResNet-34 (He et al., 2016) as the image backbone and uses detection and segmentation auxiliary tasks. The model contains 61M parameters. We train DiffusionDrive for 100 epochs on 8 H100 GPUs with a total batch size of 192, an initial learning rate of $6 \times 1 0 ^ { - 4 }$ , a cosine decay schedule, and the AdamW optimizer (Loshchilov & Hutter, 2019).

Rap<sup>∗</sup> (Feng et al., 2026) is a scoring-based planner that selects trajectories from a predefined trajectory vocabulary. It takes four camera views with a resolution of $4 \times 7 6 8 \times 4 4 { \bar { 8 } }$ , predicts a 5s future trajectory at $2 \mathrm { H z } ,$ and uses ResNet-34 (He et al., 2016) as the image backbone. The corresponding model size is 69M. Unlike TransFuser and DiffusionDrive, Rap<sup>∗</sup> does not use auxiliary detection or segmentation tasks. We train Rap<sup>∗</sup> for 30 epochs on 8 H100 GPUs with a total batch size of 48, an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , a cosine decay schedule, and the Adam optimizer (Kingma & Ba, 2015).

## C ADDITIONAL QUANTITATIVE EXPERIMENTS

## C.1 LATENT ALIGNMENT IMPLEMENTATION

We provide additional implementation details of the representation alignment module. In our experiments, we use the official public checkpoints of DVGT (Zuo et al., 2026a), VGGT (Wang et al., 2025), DINOv3 (Siméoni et al., 2025), SAM3 (Carion et al., 2025), and DA3 (Lin et al., 2025) as frozen VFM targets, and keep all VFM parameters frozen throughout training. For all VFM targets, we use the intermediate representations before their task-specific prediction heads, rather than the final task outputs. This design avoids relying on explicit pseudo-labels and instead transfers the VFM’s learned visual representations to the planner encoder.

For DVGT (Zuo et al., 2026a), we follow its original feature extraction design and take the multi-level features before the DPT prediction head. Specifically, we extract four intermediate token features from the DVGT aggregator, fuse them with learnable weights, and reshape the patch tokens into a spatial feature map. For other VFM targets, we follow the same principle and use the corresponding feature maps before their prediction heads.

Given the planner feature $\mathbf { F } ^ { s }$ and the VFM feature $\mathbf { F } ^ { t }$ , we first resize $\mathbf { F } ^ { s }$ to the spatial resolution of $\mathbf { F } ^ { t }$ . Both features are then projected into a shared latent space by lightweight 1×1 convolutional projection heads. The projected features are used to compute the spatial distribution alignment loss and the semantic consistency loss described in Appendix A.2. The VFM and alignment heads are used only during training and discarded after training, introducing no additional inference cost.

## C.2 DIFFERENT ALIGNMENT OBJECTIVES ABLATION

To validate the effectiveness of the latent alignment objective, we conduct an ablation study on TransFuser (Chitta et al., 2023) with different alignment objectives. We consider representative losses for feature-level, distribution-level, and contrastive supervision, and compare different combinations for spatial and semantic alignment (Yu et al., 2025b; Li et al., 2026a; Wu et al., 2026). As shown in Table S.2, jointly aligning spatial dependencies and semantic consistency generally yields better planning performance than using a single alignment objective. Among all combinations, KL divergence for spatial alignment and contrastive loss for semantic alignment achieve the best performance, reaching an EPDMS of 89.0.

![](images/d4ebb9a13d83fc90112904efd732ca28aac5cdef4340ff789380489e1c50c8e4.jpg)  
(a) Before Alignment (TransFuser)

![](images/3ec7921d36b4f8ce2f4bd98e26bc4ef5a8d6a6e5d0d7f71cbee6100fcb5550d6.jpg)  
(b) After Alignment (TransFuser-ViRA)  
Figure S.2: Feature correlation analysis.

## C.3 SUPPLEMENTARY ALIGNMENT ANALYSIS

To further verify the alignment effectiveness of ViRA, we compute the Pearson correlation between planner encoder features and VFM features. As shown in Figure S.2, the original TransFuser encoder exhibits weak correlation with the VFM features, and the resulting correlation matrix lacks clear structural patterns. After DVGT supervision, TransFuser with DVGT shows stronger and more structured correlation patterns, indicating that ViRA effectively narrows the representation gap between the planner encoder and the VFM. This result provides feature-level evidence for the effectiveness of ViRA.

## C.4 VFM TARGET DEPENDENCY ANALYSIS

This section provides additional quantitative results for the VFM-target analysis. We evaluate ViRA on both TransFuser (Chitta et al., 2023) and Rap<sup>∗</sup> (Feng et al., 2026). TransFuser uses detection and segmentation as perception auxiliary tasks, whereas Rap<sup>∗</sup> does not rely on explicit perception supervision, allowing us to examine whether ViRA behaves consistently across different planner designs.

Table S.3: Effect of different VFM teachers on TransFuser. Performance on the NAVSIM v2 navtest benchmark.
<table><tr><td>Teachers</td><td>|NC ↑ DAC ↑</td><td>DDC ↑</td><td></td><td>TLC ↑</td><td>EP↑</td><td>TTC ↑LK↑</td><td></td><td>HC ↑ EC ↑</td><td></td><td>EPDMS ↑</td></tr><tr><td>Baseline</td><td>|97.1</td><td>89.6</td><td>99.0</td><td>99.8</td><td>97.9</td><td>95.9</td><td>95.9</td><td>98.3</td><td>87.3</td><td>83.6</td></tr><tr><td>Random init</td><td>|96.8</td><td>90.2</td><td>99.0</td><td>99.7</td><td>98.0</td><td>96.1</td><td>95.5</td><td>98.3</td><td>86.7</td><td>82.0 -1.6</td></tr><tr><td>DVGT (Zuo et al., 2026a)</td><td>97.7</td><td>94.1</td><td>99.5</td><td>99.8</td><td>98.2</td><td>96.7</td><td>96.8</td><td>98.3</td><td>87.9</td><td>89.0 +5.4</td></tr><tr><td>VGGT (Wang et al., 2025)</td><td>97.8</td><td>94.6</td><td>99.3</td><td>99.8</td><td>97.8</td><td>97.0</td><td>96.7</td><td>98.3</td><td>87.2</td><td>89.3 +5.7</td></tr><tr><td>DA3 (Lin et al., 2025)</td><td>97.9</td><td>94.2</td><td>99.5</td><td>99.8</td><td>98.0</td><td>97.0</td><td>95.9</td><td>98.3</td><td>87.5</td><td>89.1 +5.5</td></tr><tr><td>DINOv3-L (Siméoni et al., 2025)</td><td>97.6</td><td>94.4</td><td>99.3</td><td>99.7</td><td>98.0</td><td>97.0</td><td>96.7</td><td>98.3</td><td>86.6</td><td>88.8 +5.2</td></tr><tr><td>SAM3 (Carion et al., 2025)</td><td>97.9</td><td>94.2</td><td>99.5</td><td>99.8</td><td>98.0</td><td>96.9</td><td>95.1</td><td>98.3</td><td>88.1</td><td>89.0 +5.4</td></tr></table>

Table S.4: Effect of different VFM teachers on TransFuser without auxiliary perception supervision. Performance on the NAVSIM v2 navtest benchmark.
<table><tr><td>Teachers</td><td>|NC ↑ DAC ↑</td><td></td><td>DDC ↑</td><td>TLC↑</td><td>EP↑</td><td>TTC ↑LK↑</td><td></td><td>HC ↑EC ↑</td><td></td><td>EPDMS ↑</td></tr><tr><td>Baseline</td><td>|97.0</td><td>89.9</td><td>98.5</td><td>99.8</td><td>97.9</td><td>96.3</td><td>94.3</td><td>98.4</td><td>86.8</td><td>82.6</td></tr><tr><td>Random init</td><td>|95.4</td><td>90.9</td><td>97.7</td><td>99.5</td><td>98.1</td><td>94.2</td><td>93.6</td><td>98.3</td><td>84.8</td><td>80.1 -2.5</td></tr><tr><td>DVGT (Zuo et al., 2026a)</td><td>97.9</td><td>90.9</td><td>99.1</td><td>99.7</td><td>98.0</td><td>96.8</td><td>96.5</td><td>98.3</td><td>87.2</td><td>88.2 +5.6</td></tr><tr><td>VGGT (Wang et al., 2025)</td><td>97.7</td><td>92.9</td><td>99.1</td><td>99.8</td><td>98.0</td><td>96.6</td><td>95.9</td><td>98.3</td><td>87.5</td><td>87.6 +5.0</td></tr><tr><td>DA3 (Lin et al., 2025)</td><td>97.9</td><td>92.4</td><td>99.1</td><td>99.7</td><td>97.9</td><td>96.7</td><td>95.9</td><td>98.3</td><td>85.8</td><td>87.2 +4.6</td></tr><tr><td>DINOv3-L (Siméoni et al., 2025)</td><td>97.9</td><td>90.9</td><td>99.3</td><td>99.8</td><td>98.2</td><td>96.9</td><td>96.7</td><td>98.5</td><td>88.6</td><td>89.1 +6.5</td></tr><tr><td>SAM3 (Carion et al., 2025)</td><td>97.6</td><td>91.5</td><td>99.0</td><td>99.8</td><td>97.9</td><td>96.4</td><td>96.0</td><td>98.4</td><td>88.2</td><td>86.4 +3.8</td></tr></table>

VFM-target dependency across planners. Across the five targets used in the main comparison, ViRA improves TransFuser by 5.2–5.7 EPDMS and Rap\* by 6.6–12.1 EPDMS (Tables S.3 and S.5). Table S.4 further shows that ViRA improves TransFuser without auxiliary perception supervision by 3.8–6.5 EPDMS points across the five VFM targets. Compared with Table S.3, removing auxiliary perception supervision increases the EPDMS spread across the same targets from 0.5 to 2.7 points, indicating greater sensitivity to VFM target selection.

Random-target control. To test whether the gains depend on VFM pre-training, we replace the pretrained target with a randomly initialized VFM target while retaining the same alignment objectives. As shown in Tables S.3 and S.5, random-target alignment lowers EPDMS by 1.6 points for TransFuser and 1.7 points for Rap<sup>∗</sup>. Pre-trained DVGT improves both planners, by +5.4 and +10.9 points, respectively. Applying the same alignment objectives to random features therefore does not reproduce the gains observed with pre-trained VFM representations.

BEV semantic map analysis. We further report the BEV semantic map results of TransFuser in Table S.6. All pre-trained VFM targets improve BEV segmentation mIoU, but the gains vary from +0.5 (SAM3) to +3.0 (DINOv3).

## D ADDITIONAL QUALITATIVE VISUALIZATIONS

## D.1 ADDITIONAL QUALITATIVE ANALYSIS

We provide additional qualitative visualizations to further analyze the effect of ViRA across different planners, VFM targets, and driving scenarios. These examples include perception-based planners in complex intersection scenes Figure S.4, and comparisons under different VFM targets Figure S.3. Figure S.5 also shows that the benefits of VFM-guided alignment for perception-free planning vary across targets and scenarios. Together, these visualizations complement the quantitative results and show that ViRA consistently improves scene understanding and planning behavior across different settings.

## D.2 FAILURE CASES

Although ViRA improves representation quality and downstream planning performance, challenging failure cases remain. For perception-free planners Figure S.6, different VFM targets exhibit slightly different failure patterns, suggesting that their representations may differ in subtle but meaningful ways and lead to small performance variations. For perception-based planners Figure S.7, even when ViRA produces more complete BEV semantic maps, the planned trajectory may still fail to follow the human expert in challenging scenarios. These cases show that high-quality visual representations are fundamental, but not a silver bullet for end-to-end autonomous driving. Robust planning still requires better trajectory decoding, temporal reasoning, interaction modeling, and decision-making under uncertainty.

Table S.5: Effect of different VFM teachers on $\mathbf { R a p } ^ { * }$ . Performance on the NAVSIM v2 navtest benchmark. Rap<sup>∗</sup> is our reimplementation with a different backbone from the official codebase and without the original data augmentation.
<table><tr><td>Teachers</td><td>|NC ↑ DAC ↑ DDC ↑ TLC ↑ EP ↑</td><td></td><td></td><td></td><td></td><td>TTC ↑LK↑</td><td></td><td>HC ↑ EC ↑</td><td></td><td>EPDMS↑</td></tr><tr><td>Baseline</td><td>|91.6</td><td>88.8</td><td>93.5</td><td>99.8</td><td>87.2</td><td>92.5</td><td>90.9</td><td>97.2</td><td>56.7</td><td>72.8</td></tr><tr><td>Random init</td><td>|94.6</td><td>88.1</td><td>95.1</td><td>99.8</td><td>84.2</td><td>95.1</td><td>93.1</td><td>96.7</td><td>41.9</td><td>71.1-1.7</td></tr><tr><td>DVGT (Zuo et al., 2026a)</td><td>97.1</td><td>96.9</td><td>99.2</td><td>99.9</td><td>87.5</td><td>96.9</td><td>96.2</td><td>97.7</td><td>48.1</td><td> $8 3 . 7 + 1 0 . 9 $ </td></tr><tr><td>VGGT (Wang et al., 2025)</td><td>96.4</td><td>95.6</td><td>98.6</td><td>99.9</td><td>84.0</td><td>96.9</td><td>95.1</td><td>97.4</td><td>36.5</td><td> $7 9 . 4 + 6 . 6$ </td></tr><tr><td>DA3 (Lin et al., 2025)</td><td>96.1</td><td>95.6</td><td>98.6</td><td>99.7</td><td>88.8</td><td>96.1</td><td>96.0</td><td>97.7</td><td>54.9</td><td> $8 2 . 5 + 9 . 7 $ </td></tr><tr><td>DINOv3 (Siméoni et al., 2025)</td><td>97.6</td><td>96.4</td><td>98.9</td><td>100.0</td><td>87.8</td><td>97.7</td><td>95.9</td><td>98.0</td><td>58.4</td><td> $8 4 . 9 + 1 2 . 1 $ </td></tr><tr><td>SAM3 (Carion et al., 2025)</td><td>97.1</td><td>95.5</td><td>98.4</td><td>99.8</td><td>89.9</td><td>96.9</td><td>95.5</td><td>96.8</td><td>60.1</td><td> $8 4 . 0 + 1 1 . 2 $ </td></tr></table>

Table S.6: VFMs dependency on BEV segmentation of TransFuser. Class-wise BEV segmentation performance of TransFuser under different VFM targets. BG indicates background, SO indicates static objects.
<table><tr><td>Teachers</td><td></td><td>BG ↑ Road ↑ Walkways ↑ Centerline ↑ SO ↑ Vehicles ↑ Pedestrians ↑ mIoU ↑</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Baseline</td><td>85.4</td><td>62.3</td><td>56.3</td><td>23.6</td><td>0.1</td><td>27.1</td><td>0.0</td><td>36.4</td></tr><tr><td>DA3 (Lin et al., 2025)</td><td>86.6</td><td>63.8</td><td>59.1</td><td>28.2</td><td>0.3</td><td>32.3</td><td>0.0</td><td> $3 8 . 6 + 2 . 2 $ </td></tr><tr><td>VGGT (Wang et al., 2025)</td><td>86.5</td><td>63.9</td><td>58.5</td><td>27.6</td><td>0.3</td><td>33.6</td><td>0.0</td><td>38.6 +2.2</td></tr><tr><td>SAM3 (Carion et al., 2025)</td><td>85.8</td><td>62.4</td><td>56.7</td><td>25.0</td><td>0.1</td><td>28.0</td><td>0.0</td><td>36.9 +0.5</td></tr><tr><td>DVGT (Zuo et al., 2026a)</td><td>86.9</td><td>64.6</td><td>59.3</td><td>28.9</td><td>0.5</td><td>33.4</td><td>0.2</td><td>39.1 +2.7</td></tr><tr><td>DINOv3 (Siméoni et al., 2025)</td><td>87.1</td><td>64.7</td><td>60.1</td><td>29.8</td><td>0.3</td><td>33.5</td><td>0.2</td><td>39.4 +3.0</td></tr></table>

## E LIMITATIONS AND BROADER IMPACT

Training-time VFM Cost. Although ViRA introduces no additional inference overhead, it relies on external frozen VFM representations during training. Extracting and storing VFM representations may introduce extra training cost, especially when using large-scale VFMs or high-resolution multi view inputs. A practical future direction is to reduce this cost through feature caching, lightweight

VFM distillation, or selective alignment over only the most informative layers and spatial regions. This would make VFM-guided representation alignment more scalable for larger datasets and more diverse planner architectures.

Scope of Architectures and Benchmarks. This work evaluates ViRA on three representative E2E planners, on NAVSIM v2 navtest and navhard, and in zero-shot closed-loop evaluation on HUGSIM. It does not cover multi-modal sensor fusion, vision-language-action models, or real-vehicle deployment. Further validation on a wider set of planners and real-world data is needed to assess how far the observed planner-dependent conversion from representation quality to planning gains generalizes.

Broader Impact. By improving the visual representations used by E2E planners, ViRA may contribute to safer and more reliable autonomous driving systems. Meanwhile, the use of large-scale VFMs also raises potential concerns. VFMs may inherit biases from their pre-training data, including geographical, weather, cultural, and traffic-distribution biases. If such biases are transferred to autonomous driving planners without careful validation, they may cause degraded performance in underrepresented regions or rare safety-critical scenarios. Therefore, we believe that VFM-guided planning should be accompanied by rigorous evaluation across diverse environments, transparent failure analysis, and safety-oriented validation before real-world deployment.

![](images/60a379e577c70baca618b4a06274fee16be63a9f75db219133ad531bea79c06e.jpg)  
Figure S.3: Additional qualitative analysis for perception-based planners across different VFMs.

## F LICENSE OF ASSETS

All training and evaluation are conducted on publicly available datasets and benchmarks, including nuPlan (Karnchanachari et al., 2024), OpenScene (OpenScene Contributors, 2023), and NAVSIM v2 (Cao et al., 2025). We use these datasets only for research purposes and follow their official licenses and terms of use. We adopt several publicly available VFMs as VFM targets, including DVGT (Zuo et al., 2026a), VGGT (Wang et al., 2025), DINOv3 (Siméoni et al., 2025), SAM3 (Carion et al., 2025), and DA3 (Lin et al., 2025). DVGT is released under the Apache License 2.0. VGGT, DINOv3, and SAM3 are used under their respective official licenses. For DA3, we follow the license associated with the specific released model checkpoint used in our experiments. We also build upon publicly available E2E planners, including Rap<sup>∗</sup> (Feng et al., 2026), DiffusionDrive (Liao et al., 2025), and TransFuser (Chitta et al., 2023). Rap<sup>∗</sup> is released under the Apache License 2.0, while DiffusionDrive and TransFuser are released under the MIT License. We do not redistribute third-party datasets, model weights, or checkpoints. All third-party assets are used in accordance with their corresponding licenses. Our released code will include proper attribution and license notices for all used assets.

![](images/4069817cc28d0aee6f24b4f50dc7145e7cfb1200642b2c589ea5743bee3092d1.jpg)  
Figure S.4: Additional qualitative analysis for perception-based planners.

![](images/bb7fb5b46a625bf8d3576f646e37f99c439b3f291e930cfbf823b38da658adf5.jpg)  
Figure S.5: Additional qualitative analysis for perception-free planners across different VFMs.

![](images/d11501475327c51d95dc41d3657fbd46d3a0faf42998598164fdec9477717fbc.jpg)  
Figure S.6: Failure case analysis for a perception-free planner.

![](images/2695bb5c05cc0f7c951b32a4a6157e00376f97a1890c5c6d8febc289b55ecb2a.jpg)  
Figure S.7: Failure case analysis for perception-based planners.