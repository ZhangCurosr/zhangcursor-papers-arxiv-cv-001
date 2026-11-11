# DroneSegEdit: Adapting Pretrained Generative Editors for UAV Segmentation with Language-Grounded Verification

Dehua Chen Long Xin Jiangtao Wei<sup>\*</sup>

Fudan University, China

26113090032@m.fudan.edu.cn longxin25@m.fudan.edu.cn jtwei@fudan.edu.cn

## Abstract

Recent work has shown that pretrained image generators can support diverse visual tasks. Yet directly applying them to language-guided unmanned aerial vehicle (UAV) segmentation exposes a key limitation: a plausible mask may correspond to the wrong instance among small and repeated objects. We call this mismatch between mask rendering and expression-level target selection the Grounding–Rendering Gap. We present DRONESEGEDIT, which combines RGB proposal rendering with language-grounded, proposal-constrained verification to select supported foreground while retaining rendered geometry. DRONESEGEDIT adapts the open-weight LongCat-Image-Edit model using task-specific LoRA and fixed color-to-label decoding. For referring segmentation, a Language-Grounded Spatial Verifier (LGSV) reads hierarchical states from the frozen multimodal large language model backbone to derive expression-conditioned spatial evidence, which is mapped back to the rendered proposal without introducing foreground outside it. On RIS-LAD, DRONESEGEDIT reaches a new state of the art at 52.78 mIoU, 7.16 points above the previous best. It remains competitive overall on UAVid-RIS and VDD-RIS, with its clearest class-wise gains on compact, application-relevant categories, notably vehicles and humans. A separate verifier-free experiment extends the RGB interface to full-scene parsing: a task-specific adapterjointly renders all classes from one task-level instruction without enumerating class names or palette colors.

## 1. Introduction

Referring segmentation converts a natural-language expression into a pixel-level mask. Unmanned aerial vehicle (UAV) imagery supports civilian applications such as traffic and infrastructure monitoring, where reliable perception of vehicles and people is essential [2, 26]. Yet these application-relevant targets often occupy few pixels and occur in dense groups of visually similar instances. A referring expression may distinguish the intended instance only through subtle spatial or contextual cues. A successful system must therefore answer two coupled questions: which instance is being referred to, and which pixels belong to that instance.

Most referring systems rely on discriminative cross-modal decoders or multimodal reasoning followed by promptable segmentation [6, 14, 15, 17, 25, 28]. In parallel, image-space generalist models show that heterogeneous visual tasks can share an RGB prediction interface [11, 31, 37, 38, 48], while recent evidence further indicates that generative pretraining provides transferable visual priors [10]. For UAV perception, this formulation can retain fine spatial structure, produce visually inspectable predictions, and provide a common output space across heterogeneous pixel-level tasks. Its use for language-guided aerial segmentation, however, remains largely unexplored.

We adopt the open-weight, instruction-guided LongCat-Image-Edit [27]. Its backbone combines Qwen2.5-VL for multimodal instruction understanding with a 6B-parameter diffusion transformer [29] for image generation. It reports leading performance among open-weight image-editing models on standard benchmarks, providing a strong generative prior for UAV segmentation. We keep the pretrained editor frozen and adapt it with lightweight LoRA updates. A unified RGB output interface encodes segmentation targets as task-defined colors and recovers discrete masks through fixed color-to-label decoding. For referring segmentation, the referred region is rendered in a specified color while the remainder of the input image is preserved.

Pretrained generative editors offer useful visual priors for rendering spatially coherent RGB masks, but plausible mask generation does not guarantee expression-level target selection among small, repeated, or visually similar UAV instances. We call this mismatch between rendered regions and expression-supported regions the Grounding–Rendering Gap. DRONESEGEDIT addresses it by rendering an RGB proposal and then applying language-grounded, proposal-constrained verification: LGSV estimates expression-conditioned support, and the final mask remains restricted to the rendered proposal. The combination preserves rendered boundaries while using coarse language evidence to select foreground; matched-candidate controls test this complementary role (Table 4).

![](images/50d05a8d16fd397f7e7eaaf9ee71da34951e817608596ed381a4a556449e791e.jpg)  
(a) Overall framework

Language-Grounded Spatial Verifier (LGSV)  
![](images/8ff770234d5bca5a88ec90a26a66bc78ddd8de8ec64873c3fdb6c095135e6a8d.jpg)  
(b) LGSV architecture  
Figure 1. DroneSegEdit architecture. (a) The adapted editor renders an RGB proposal, deterministic decoding yields $M _ { \mathrm { r a w } }$ , and LGSV removes expression-unsupported regions. Green and red contour overlays mark retained and rejected candidates, respectively. (b) Four FiLM-conditioned probes reuse frozen Qwen2.5-VL states from the editor’s conditioning forward pass and fuse complementary spatial-semantic cues into P. The editor supplies proposal geometry, while LGSV supplies expression-conditioned support.

The Language-Grounded Spatial Verifier (LGSV; Fig. 1) reads four hierarchical layers of the editor’s frozen Qwen2.5-VL backbone to check whether proposed regions match the referring expression. This separates target selection from RGB generation while keeping the rendered regions as the spatial reference. Given a palette-decoded candidate $M _ { \mathrm { r a w } }$ and LGSV support map P, the shared verification operator is

$$
\begin{array} { r l } & { M _ { \mathrm { f i n a l } } = \mathcal { V } _ { t } ( M _ { \mathrm { r a w } } , P ) } \\ & { \qquad : = M _ { \mathrm { r a w } } \cap S _ { t } ( M _ { \mathrm { r a w } } , P ) \subseteq M _ { \mathrm { r a w } } , } \end{array}\tag{1}
$$

Equation 1 makes verification one-sided: rendering supplies candidate geometry and LGSV removes unsupported foreground without synthesizing new regions. Single-instance and region-union tasks share this operator; only the support-set definition follows the required output topology.

Our main contributions are:

• We repurpose a pretrained open-weight generative editor for UAV RGB mask rendering. A unified RGB output interface with fixed color-to-label decoding supports both referring segmentation and full-scene parsing through task-specific LoRA adaptation over the same frozen editor.

• We introduce LGSV and proposal-constrained verification to resolve expression-dependent target selection using frozen hierarchical multimodal states while restricting corrections to editor proposals.

• Across three UAV referring benchmarks, DRONESEGEDIT establishes a new state of the art on RIS-LAD while remaining competitive overall on UAVid-RIS and VDD-RIS, with its clearest class-wise gains on compact vehicle and human targets. Matched-candidate direct-generation and simple proposal-filtering controls isolate the complementary value of language-grounded selection. A separate verifier-free experiment extends the RGB interface to full-scene UAV parsing from one task-level instruction without per-class prompt enumeration.

## 2. Related Work

Referring and aerial segmentation. Conventional RIS uses cross-modal decoders to predict the requested mask [9, 13, 20, 39, 45]. Prompt- and mask-grounding variants refine instance contrast or pixel–language alignment [8, 35]; LISA, SAM4MLLM, and ReSaP route multimodal reasoning to dedicated mask backbones [6, 17, 25]. UAV and remote-sensing imagery further introduces arbitrary orientation, scale variation, compact targets, and repeated instances [4, 19, 21, 22, 36, 46, 47]. In contrast, our adapted editor supplies proposal geometry and language evidence selects within that candidate rather than redrawing its boundaries.

Generative segmentation and image-space interfaces. Painter, SegGPT, and InstructDiffusion formulate heterogeneous visual tasks as image-space prediction [11, 37, 38]. GSS, GenMask, and recursive noise diffusion study direct mask or semantic-map generation, including aerial views [3, 16, 44]. Concurrent work evaluates released image editors, including LongCat-Image-Edit, as zero-shot dense predictors on generic benchmarks without task-specific adaptation, and excludes instance segmentation because of inconsistent instance-level outputs [23]. In contrast, we study parameter-efficient adaptation for UAV segmentation: background-preserving RGB rendering supplies a candidate, and language-grounded verification addresses the Grounding–Rendering Gap for compact and repeated referents. The separate dense experiment uses a fixed task-level instruction rather than a per-image ground-truth class list.

Generative-feature readouts and language grounding. CLIP and DenseCLIP transfer image–text alignment to dense prediction [32, 33]. ODISE and VPD reuse diffusion representations for perception [43, 49], whereas DMP adapts a pretrained diffusion model for dense prediction [18]. Readout Guidance controls sampling through a learned readout [24]. Related PEFT methods adapt vision, diffusion, and geospatial models with small trainable modules [5, 34, 42]. LGSV instead reads hierarchical Qwen states from the editor’s existing conditioning forward pass and supplies expression-conditioned evidence within the rendered candidate; it neither reruns nor controls sampling. Dense parsing uses a closed-set, task-specific adapter.

## 3. Method

## 3.1. Unified RGB Rendering

Let $t \in \tau$ index a UAV task, $I \in [ 0 , 2 5 5 ] ^ { H \times W \times 3 }$ an input image, e a referring expression when present, Ω the $H \times$ W pixel lattice, and $Y _ { t }$ the target. Referring tasks form $\mathcal { T } _ { \mathrm { r e f } } = \mathcal { T } _ { \mathrm { s i n g l e } } \cup \mathcal { T } _ { \mathrm { u n i o n } } ;$ dense semantic tasks form $\mathcal { T } _ { \mathrm { s e m } } .$ We freeze the pretrained parameters $\theta _ { 0 }$ and learn standard task-specific low-rank updates $W _ { j } ^ { ( t ) } = W _ { j } + \Delta W _ { j } ^ { ( t ) }$ , with ran $\mathfrak { x } ( \Delta W _ { j } ^ { ( t ) } ) \ \le \ r _ { j }$ [12]; the factorization and injection details are provided in the appendix. Let $\mathbf { c } _ { \mathrm { r e f } } ^ { ( t ) }$ denote the task-defined reference color for referring segmentation and $\{ \mathbf { c } _ { k } ^ { ( t ) } \} _ { k \in \mathcal { C } _ { t } }$ the palette over the semantic label set $\mathcal { C } _ { t } .$ The common RGB supervision canvas is

$$
Y _ { \mathrm { r g b } } ^ { ( t ) } ( x ) = \left\{ \begin{array} { l l } { \mathbf { c } _ { \mathrm { r e f } } ^ { ( t ) } , } & { t \in \mathcal { T } _ { \mathrm { r e f } } , Y _ { t } ( x ) = 1 , } \\ { I ( x ) , } & { t \in \mathcal { T } _ { \mathrm { r e f } } , Y _ { t } ( x ) = 0 , } \\ { \mathbf { c } _ { Y _ { t } ( x ) } ^ { ( t ) } , } & { t \in \mathcal { T } _ { \mathrm { s e m } } . } \end{array} \right.\tag{2}
$$

Thus, the target for referring segmentation is a background-preserving edited photograph, whereas the target for dense semantic segmentation is a complete palette label image; in both cases, the model predicts three-channel RGB rather than mask logits. The editor learns the color–label correspondence implicitly from RGB supervision, while the fixed palette is used to encode targets and decode predictions. For referring tasks, the instruction $u _ { t } ( e )$ casts prediction as sparse editing: render only the referred region in the task-defined color and preserve all remaining pixels. Let $\Delta _ { t } ~ = ~ \{ \Delta W _ { j } ^ { ( t ) } \} _ { j }$ <sub>j</sub> collect the task adapter. The adapted editor G renders

$$
I _ { g } = G _ { \theta _ { 0 } , \Delta _ { t } } ( I , u _ { t } ( e ) ) , \qquad M _ { \mathrm { r a w } } = \mathcal { D } _ { \mathrm { p a l } } ^ { ( t ) } ( I _ { g } ) .\tag{3}
$$

The fixed decoder ${ \mathcal { D } } _ { \mathrm { p a l } } ^ { ( t ) }$ uses one tolerance around the reference-color region for all samples. $M _ { \mathrm { r a w } }$ is the rendered candidate before verification. The background-preserving RGB output $I _ { g }$ can also be inspected directly before decoding.

For dense semantic segmentation, a task-specific adapter receives the fixed instruction “Convert the full image into an RGB semantic label map.” and renders all classes jointly in one generation. The instruction enumerates neither class names nor palette colors; paired RGB targets teach their correspondence. Nearest-color decoding gives

$$
\widehat { Y } _ { t } ( x ) = \arg \operatorname* { m i n } _ { k \in \mathcal { C } _ { t } } \| I _ { g } ( x ) - \mathbf { c } _ { k } ^ { ( t ) } \| _ { 2 } ^ { 2 } .\tag{4}
$$

All tasks therefore share an RGB output interface over the same frozen pretrained editor. Dense VDD uses the verifier-free palette path to render all classes jointly in one generation from a single task-level instruction.

## 3.2. Language-Grounded Spatial Evidence

The frozen Qwen2.5-VL backbone [1] processes $( I , u _ { t } ( e ) )$ in the same conditioning forward pass used by the editor. LGSV taps its hierarchical hidden states and pools the token positions corresponding to the referring expression $e ,$ as indicated by $m _ { n } ;$ no additional Qwen pass is required during integrated inference. At layer $\ell \in \mathsf { \Omega }$ $\mathcal { L } ~ = ~ \{ 8 , 1 6 , 2 4 , 2 8 \}$ , LGSV reads visual states $F _ { \ell } ^ { v } ~ \in$ $\mathbb { R } ^ { N _ { v } \times d _ { \ell } }$ and expression-token states $F _ { \ell } ^ { e } \in \mathbb { R } ^ { N _ { e } \times d _ { \ell } }$ . The image-contextualized expression state and FiLM-modulated visual grid are

$$
\begin{array} { r l } & { q _ { \ell } = \cfrac { \sum _ { n } m _ { n } F _ { \ell , n } ^ { e } } { \sum _ { n } m _ { n } } , \qquad ( \gamma _ { \ell } , \beta _ { \ell } ) = g _ { \ell } ( q _ { \ell } ) , } \\ & { U _ { \ell } = \mathrm { G r i d } \big ( \phi _ { \ell } ( \mathrm { L N } ( F _ { \ell } ^ { v } ) ) \big ) , } \\ & { \widetilde { U } _ { \ell } = U _ { \ell } \odot ( 1 + \kappa \gamma _ { \ell } ) + \kappa \beta _ { \ell } . } \end{array}\tag{5}
$$

Here $\phi _ { \ell }$ projects visual tokens, $g _ { \ell }$ generates the channel-wise FiLM parameters $( \gamma _ { \ell } , \beta _ { \ell } )$ , and Grid restores the token layout. The fixed scale κ controls the residual modulation magnitude [30]. Two convolution–SiLU blocks and a pointwise head $h _ { \ell }$ produce logits $Z _ { \ell } = h _ { \ell } ( \widetilde { U } _ { \ell } )$ . Uniform logit averaging followed by bilinear upsampling yields

$$
P = \mathrm { U p } _ { H , W } \left( \sigma \left( { \frac { 1 } { 4 } } \sum _ { \ell \in { \mathcal { L } } } Z _ { \ell } \right) \right) ,\tag{6}
$$

where $P \in [ 0 , 1 ] ^ { H \times W }$ is a relative language-conditioned support map, not a calibrated posterior or an independent boundary mask. LGSV is a readout over the frozen backbone. This separates RGB rendering from expression verification: the editor produces an inspectable RGB candidate, and LGSV removes foreground unsupported by the expression.

The two branches are deliberately asymmetric. The editor predicts in image space and supplies coherent proposal contours, whereas Qwen evidence is produced on a $3 7 \times 3 7$ token grid. Although language-conditioned, this grid is too coarse to recover thin or irregular boundaries independently. Treating P as the final mask would therefore require the readout to solve boundary reconstruction. We instead use $P$ only to select support within $M _ { \mathrm { r a w } }$ , enforcing $M _ { \mathrm { f i n a l } } \subseteq M _ { \mathrm { r a w } }$ and preventing verification from introducing foreground not proposed by the editor.

Every active depth is supervised before aggregation. Let $O _ { \ell } = { \bf 1 } [ \mathrm { A d a M a x P o o l } _ { \ell } ( Y _ { t } ) > 0 ]$ denote target occupancy on the ℓ-th evidence grid, and let $\delta _ { t } = \mathbf { 1 } [ t = \mathrm { R I S - L A D } ]$ Full-grid BCE labels non-target cells as negatives under expression conditioning, while Dice penalizes excessive predicted support. For single-instance RIS-LAD, center-cell supervision additionally encourages the evidence peak to fall near the target centroid:

$$
\mathcal { L } _ { \mathrm { L G S V } } ^ { ( t ) } = \frac { 1 } { L } \sum _ { \ell \in \mathcal { L } } \Bigl [ \lambda _ { b } \mathcal { L } _ { \mathrm { B C E } } ^ { ( \ell ) } + \lambda _ { d } \mathcal { L } _ { \mathrm { D i c e } } ^ { ( \ell ) }\tag{7}
$$

For region-union tasks, where multiple same-class regions may be valid, $\delta _ { t } ~ = ~ 0$ and only the center-cell term is disabled; the full-grid BCE and Dice losses remain active. All fixed coefficients and target constructions are given in the appendix.

## 3.3. Proposal-Constrained Verification

LGSV applies the set operator $\nu _ { t }$ in Eq. 1 to both single-referent and region-union outputs. Let $\begin{array} { r l } { \Omega } & { { } = } \end{array}$ $\{ 1 , \ldots , H \} \times \{ 1 , \ldots , W \}$ be the image lattice. The task topology changes only the construction of the evidence support set ${ \cal S } _ { t } \subseteq \Omega$

For component competition, decompose $\begin{array} { r l } { M _ { \mathrm { r a w } } } & { { } = } \end{array}$ $\textstyle \bigcup _ { i = 1 } ^ { N } C _ { i }$ into 8-connected components and define $\mathcal { E } = \{ i$ $| \dot { C _ { i } } | \geq A _ { \operatorname* { m i n } } \}$ . Each eligible component receives the mean LGSV evidence within its rendered contour,

$$
s _ { i } = { \frac { 1 } { | C _ { i } | } } \sum _ { x \in C _ { i } } P ( x ) , \qquad i \in { \mathcal { E } } .\tag{8}
$$

For a single referent, retain eligible components whose evidence is within a factor $\rho$ of the strongest component. For a region union, dilate the high-evidence seed set. When $| \mathcal { E } | > 1$ , define

$$
\begin{array} { r l } & { { \kappa _ { \rho } } = \left\{ i \in \mathcal { E } : s _ { i } \ge \rho \underset { j \in \mathcal { E } } { \operatorname* { m a x } } s _ { j } \right\} , } \\ & { { \mathcal { H } _ { t } } = \{ x \in \Omega : P ( x ) \ge \tau _ { t } \} . } \end{array}
$$

Let $\begin{array} { r } { \mathcal { B } _ { d } ( A ) = \bigcup _ { x \in A } \{ y \in \Omega : \| y - x \| _ { 1 } \leq d \} } \end{array}$ denote $\ell _ { 1 }$ dilation. The two support constructions are

$$
\begin{array} { l } { { \displaystyle \mathcal { S } ^ { \mathrm { s i n g l e } } = \bigcup _ { i \in \mathcal { K } _ { \rho } } C _ { i } , \quad \vert \mathcal { E } \vert > 1 , } } \\ { { \displaystyle \mathcal { S } ^ { \mathrm { u n i o n } } = \mathcal { B } _ { d _ { t } } ( \mathcal { H } _ { t } ) . } } \end{array}\tag{9}
$$

When $| \mathcal { E } | \leq 1$ , the single-referent branch sets $S _ { t } ~ = ~ \Omega$ leaving the prediction unchanged. Otherwise, $S _ { t } = S ^ { \mathrm { s i n g l e } }$ for single-referent tasks and $S _ { t } = S ^ { \mathrm { u n i o n } }$ for region-union tasks. Here $\rho , \tau _ { t }$ , and $d _ { t }$ are fixed on validation data; all constants are listed in the appendix.

The upsampled $3 7 \times 3 7$ expression-conditioned grid can assign spatially diffuse responses to visually similar instances. The RIS-LAD rule therefore compares mean evidence across rendered components for relative proposal selection.

The single-referent rule performs competition among eligible rendered components. Averaging within each contour normalizes for component area, so ranking reflects mean expression support rather than the total evidence accumulated by a large region. The relative criterion can retain several similarly supported components. For region unions, thresholding and dilation instead construct spatial neighborhoods around multiple evidence seeds before intersection with the rendered foreground.

## 4. Experiments

## 4.1. Task Suite and Evaluation

RIS-LAD. RIS-LAD is a particularly demanding single-instance benchmark [46]. Across the complete official test set, the referred mask occupies a median of only 0.246% of the image; 69.9% and 82.7% of the targets occupy less than 0.5% and 1% of the image area, respectively. These compact referents frequently appear amid repeated or visually similar vehicles and people, and may be distinguished only through attributes, spatial relations, or scene context. Its varied illumination, including daytime and nighttime scenes, further changes target appearance. The benchmark therefore couples compact-object localization with fine-grained instance-level language grounding under substantial appearance variation. We evaluate on the complete official test split using the benchmark’s published metrics; checkpoint and decision settings are selected on validation only.

UAVid-RIS and VDD-RIS. UAVid-RIS includes disconnected class-level unions [21, 26]; VDD-RIS provides the analogous setting over VDD [2, 21]. Both follow the official protocols, with fixed settings reported in the appendix.

VDD dense semantic segmentation. For dense VDD parsing, a task-specific adapter renders a complete semantic map from one fixed task-level instruction on the official 280/80/40 split [2]. The evaluation uses nearest-palette decoding and reports mIoU over the complete label set.

For referring benchmarks, we report P@0.5–P@0.9, mIoU, and oIoU under the official protocols. P@t denotes the percentage of expressions whose predicted-mask IoU is at least t. mIoU averages mask IoU over expressions, whereas oIoU divides the accumulated intersection over the evaluation set by the accumulated union. The LongCat-Image-Edit and DRONESEGEDIT rows share identical rendered candidates; their difference is verification alone. Additional diagnostics appear in the appendix.

## 4.2. Implementation Details

We train each benchmark’s adapter independently under its official training and evaluation protocol. Qwen2.5-VL, the VAE, and the 6B base DiT remain frozen. The rendering LoRA uses rank 32 and $\alpha = 8 ,$ is injected into DiT attention and FFN projections, and is optimized with BF16 AdamW. It contains 47.19M trainable parameters, approximately 0.79% of the 6B DiT; referring tasks additionally train a 15.79M LGSV, for 62.98M trainable parameters in total. For scale, this combined trainable count is approximately 1.05% of the 6B DiT parameter count, and an even smaller fraction of the complete frozen Qwen–DiT–VAE system. Referring inference uses eight-step FlowMatch Euler sampling with CFG 2.5. Across independently adapted referring benchmarks, complete adaptation takes around 20–30 NVIDIA A800 GPU-hours per benchmark, depending on the benchmark-specific schedule. Resolution, batch size, learning rate, checkpoint, and decoder constants are provided in the appendix.

At inference, RIS-LAD uses 8-connected components with a minimum eligible area of 16 pixels and a relative evidence ratio of 0.75. The region-union settings are $( \tau _ { t } , d _ { t } ) ~ = ~ ( 0 . 6 0 , 1 6 )$ for UAVid-RIS and (0.30, 8) for VDD-RIS. These decisions operate on fixed RGB candidates; the rendering adapter is frozen before paired Raw/LGSV evaluation.

## 4.3. Referring Segmentation Results

RIS-LAD. DRONESEGEDIT reaches 52.78 mIoU and 54.01 oIoU, outperforming the paired editor by 5.42/6.92 points and ReSaP-8B by 7.16/0.57 points. Gains hold from P@0.5 (46.83→54.83) through P@0.9 (23.13→29.13). Verification raises precision from 52.93 to 62.10 while recall changes only from 81.04 to 80.57, and moves Pred./GT from 1.531 to 1.297, consistent with removing excess foreground. Figure 2 spans daylight and nighttime conditions.

Region-union and compact-target evidence. LGSV adds 3.66/5.65 mIoU/oIoU on UAVid-RIS and 1.06/4.94 on VDD-RIS. It obtains the best UAVid-RIS mIoU and P@0.5–P@0.7, but its oIoU remains 2.87 points below AeroReformer. On VDD-RIS, it obtains the best P@0.8 but trails AeroReformer by 1.03 mIoU and 3.80 oIoU; P@0.9 also decreases from 44.65 to 41.33 relative to the raw editor. Figure 3 provides matched examples, class-wise changes, and target-area distributions: the qualitative panels show fewer merged vehicle regions and fewer missed compact human regions, while VDD-RIS Vehicles reaches 82.98 IoU, 12.24 points above AeroReformer, and UAVid-RIS Human reaches 49.68 versus 26.68. These categories are also among the most compact in their respective benchmarks, complementing the small-target-dominated RIS-LAD evidence in Sec. 4.1.

Table 1. RIS-LAD test results (%). The Qwen3-VL SAM4MLLM configuration follows the SFT baseline in ReSaP [25]; LongCat-Image-Edit and DRONESEGEDIT share rendered candidates. Best/second-best values are bold/underlined.
<table><tr><td>Method</td><td>Core backbone(s)</td><td>P@0.5</td><td>P@0.6 P@0.7 P@0.8</td><td></td><td></td><td>P@0.9</td><td>mIoU</td><td>oIoU</td></tr><tr><td colspan="7">Traditional RIS methods</td><td></td><td></td></tr><tr><td>LAVT [45]</td><td>Swin-B/BERT</td><td>31.67</td><td>27.04</td><td>23.02</td><td>17.55</td><td>8.58</td><td>30.1441.97</td><td></td></tr><tr><td>FIANet [19]</td><td>Swin-B/BERT</td><td>40.21</td><td>35.22</td><td>29.93</td><td>23.71</td><td>13.21</td><td>37.44</td><td>43.39</td></tr><tr><td>RMSIN [22]</td><td>Swin-B/BERT</td><td>43.36</td><td>38.11</td><td>32.36</td><td>25.08</td><td>13.75</td><td>39.60</td><td>48.82</td></tr><tr><td>RSRefSeg [4]</td><td>SigLIP/SAM</td><td>44.73</td><td>40.21</td><td>36.30</td><td>30.91</td><td>21.46</td><td>41.16</td><td>47.71</td></tr><tr><td>SAARN [46]</td><td>Swin-B/BERT</td><td>45.02</td><td>39.34</td><td>33.37</td><td>26.24</td><td>15.31</td><td>41.67</td><td>49.60</td></tr><tr><td colspan="7">MLLM-based segmentation models</td><td></td><td></td></tr><tr><td>LISA [17]</td><td>LLaVA-7B/SAM</td><td>33.20</td><td>28.17</td><td>25.57</td><td>19.62</td><td>6.28</td><td></td><td>28.5038.39</td></tr><tr><td>SAM4MLLM [6]</td><td>Qwen3-VL-8B/SAM</td><td>45.28</td><td>39.98</td><td>34.56</td><td>30.77</td><td>11.88</td><td>39.79</td><td>47.39</td></tr><tr><td>ReSaP (8B) [25]</td><td>Qwen3-VL-8B/SAM</td><td>51.28</td><td>47.88</td><td>43.06</td><td>36.62</td><td>19.33</td><td>45.62</td><td>53.44</td></tr><tr><td colspan="7">Generative RGB-rendering models</td><td></td><td></td></tr><tr><td>LongCat-Image-Edit Qwen2.5-VL/DiT</td><td></td><td>46.83</td><td>41.26</td><td>37.60</td><td>32.50</td><td>23.13</td><td>47.36</td><td>47.09</td></tr><tr><td>DRONESEGEDIT</td><td>Qwen2.5-VL/DiT</td><td>54.83</td><td>49.84</td><td>46.44</td><td>40.75</td><td>29.13</td><td>52.78 54.01</td><td></td></tr></table>

Table 2. Region-union UAV referring results (%). Each LongCat-Image-Edit–DRONESEGEDIT pair shares rendered candidates. Best/second-best values per benchmark are bold/underlined.
<table><tr><td>Dataset</td><td>Method</td><td>Core backbone(s)</td><td>P@0.5</td><td>P@0.6 P@0.7 P@0.8 P@0.9</td><td></td><td></td><td>mIoU</td></tr><tr><td rowspan="6">UAVid-RIS</td><td>LAVT [45]</td><td>Swin-B/BERT</td><td>84.45</td><td>75.75</td><td>59.60</td><td>38.71</td><td>10.91 69.32 78.76</td></tr><tr><td>RMSIN [22]</td><td>Swin-B/BERT</td><td>85.71</td><td>77.91</td><td>64.06</td><td>46.52 17.76</td><td>72.05 81.10</td></tr><tr><td>MAFN [36]</td><td>Swin-B/BERT</td><td>85.10</td><td>77.76</td><td>63.68</td><td>44.26</td><td>16.33 71.64 80.67</td></tr><tr><td>AeroReformer [21]</td><td>Swin-B/BERT</td><td>86.34</td><td>79.07</td><td>65.60</td><td>47.12</td><td>18.52 72.79 81.53</td></tr><tr><td>LongCat-Image-Edit</td><td>Qwen2.5-VL/DiT</td><td>81.37</td><td>72.26</td><td>59.24</td><td>42.23</td><td>15.05 69.82 73.01</td></tr><tr><td>DRONESEGEDIT</td><td>Qwen2.5-VL/DiT</td><td>88.37</td><td>79.68</td><td>66.24</td><td>46.82</td><td>17.24 73.4878.66</td></tr><tr><td rowspan="6">VDD-RIS</td><td>LAVT [45]</td><td>Swin-B/BERT</td><td>90.04</td><td>86.72</td><td>76.01</td><td>56.09</td><td>34.32 77.80 82.51</td></tr><tr><td>RMSIN [22]</td><td>Swin-B/BERT</td><td>91.14</td><td>85.98</td><td>76.38</td><td>57.20</td><td>35.06 78.22 83.58</td></tr><tr><td>MAFN [36]</td><td>Swin-B/BERT</td><td>91.51</td><td>87.82</td><td>79.70</td><td>60.89</td><td>35.06 79.21 83.97</td></tr><tr><td>AeroReformer [21]</td><td>Swin-B/BERT</td><td>92.99</td><td>89.30</td><td>81.55</td><td>63.47</td><td>38.75 80.72 85.38</td></tr><tr><td>LongCat-Image-Edit</td><td>Qwen2.5-VL/DiT</td><td>86.72</td><td>81.18</td><td>74.91</td><td>62.73 44.65</td><td>78.6376.64</td></tr><tr><td>DRONESEGEDIT</td><td>Qwen2.5-VL/DiT</td><td>89.67</td><td>83.39</td><td>76.38</td><td>64.21 41.33</td><td>79.69 81.58</td></tr></table>

(a) RIS-LAD - Car · Multi-distractor correction  
(b) RIS-LAD - Truck · Distractor correction  
“The white car is driving on the road, positioned near the bottom left corner of the image.”  
“The truck is white, a Flatbed truck, driving on the road, positioned near the center-right, with a visible license plate.”

![](images/f2e55455d501a803a30ef12a05042ca965ec9389cdfc580b418d1a7c93364add.jpg)  
Figure 2. Proposal-constrained verification on RIS-LAD. (a) Car; (b) truck. All panels for each example use the same target-centered visualization crop. The evidence panels display $P \odot M _ { \mathrm { r a w } } ,$ with contours inherited from the decoded candidates. Green and red contours mark retained and rejected components, respectively; numerical labels show mean component evidence s<sub>i</sub>.

The task topology helps interpret these gains: RIS-LAD requires rejecting same-class distractors to select one instance, whereas region unions must preserve multiple valid regions. Subset-constrained filtering can trade recall for precision and cannot recover omitted foreground.

## (a) VDD-RIS - Vehicles

“Vehicles parked near trees and buildings.”

![](images/d68d01b338ef2cb7d40b98fb3aecc6f36a75e5fa191e939102cd9579c7c4439e.jpg)  
(b) VDD-RIS - Vehicles  
(c) UAVid-RIS - Human  
“People walking on sidewalk near houses.”  
“Vehicles parked in lot near trees.”

![](images/c1a2b54ca8de6d882ee18251c1ffd3bd61a770ea3ff944c7a458281602231644.jpg)  
(d) UAVid-RIS - Static Car

![](images/e6963bd25a3cfce959d0a715695111aec1623d97f3c99be11cf079609b0abcfb.jpg)

![](images/3a83c02a92213b2427a6cb192bc0a726a4847f11ee718434f699f36a91e670d2.jpg)  
“Vehicles parked along paved area near building.”

![](images/2af0e95351bc883ec4fe0a0ac3eecddde7422b786525cb67df22e8044c7f840a.jpg)

![](images/2b6ba01f06f304fcd1e3c5cbf5f41198d40b8ed3928c1d050481c7af3911ccc3.jpg)  
Figure 3. Cross-benchmark compact-target evidence. (a–d) For each UAVid-RIS and VDD-RIS example, all methods are shown using the same target-centered visualization crop, enlarged for readability. DRONESEGEDIT keeps adjacent vehicle regions separated in (a), (b), and (d), and retains both compact human regions in (c), more closely matching GT. (e) Class-wise IoU changes. (f) Per-expression target area 100|Y|/(HW) on a log scale; boxes show quartiles and 5th–95th-percentile whiskers.

## 4.4. Beyond Referring Masks: Single-Instruction Full-Scene UAV Parsing

We evaluate the shared RGB output interface with a separately trained, verifier-free dense VDD adapter (Sec. 3.1).

At 80.96 mIoU, the verifier-free editor ranks second to UPerNet–Swin-L under the matched-resolution comparison in Table 3. Figure 4 shows that a single generation jointly renders the scene label map, while nearest-palette decoding retains the dominant layout and visible object boundaries. At the selected checkpoint, the mean nearest-palette distance is 11.96, well below the minimum inter-palette separation of 87.61, supporting stable deterministic decoding.

## 5. Analysis

Table 4(a) examines LGSV under fixed rendered proposals. Four-depth fusion improves over the single-layer readout by 1.44 mIoU and raises Peak-cell/Peak-in-mask by 1.91/2.06 points. Replacing the matched expression with a mismatched one reduces mIoU from 53.15 to 47.78, while the equal-capacity image-only control reaches only 44.93. The gain therefore depends on hierarchical, expression-specific evidence rather than added capacity or image saliency alone.

Table 3. VDD full-scene mIoU (%) at matched 1152 × 864 resolution. All rows use the same split and metric implementation; best/second-best are bold/underlined.
<table><tr><td>Method</td><td>Backbone</td><td>mIoU</td></tr><tr><td>Mask2Former [7]</td><td>Swin-T</td><td>74.32</td></tr><tr><td>Mask2Former [7]</td><td>ResNet-50</td><td>79.90</td></tr><tr><td>SegFormer-B0 [41]</td><td>MiT-B0</td><td>74.94</td></tr><tr><td>SegFormer-B5 [41]</td><td>MiT-B5</td><td>78.54</td></tr><tr><td>UPerNet [40]</td><td>Swin-T</td><td>80.39</td></tr><tr><td>UPerNet [40]</td><td>Swin-L</td><td>82.17</td></tr><tr><td>DRONESEGEDIT</td><td>Qwen2.5-VL/DiT</td><td>80.96</td></tr></table>

Table 4(b) separates evidence from geometry.

![](images/1679645ee636bd0ada2bbea33ade55f8ca5d1b6cb0a76ade56d4983eedb2ba01.jpg)  
Figure 4. Single-instruction dense VDD parsing. One generation renders the full RGB label map; nearest-palette decoding recovers discrete classes, including densely arranged vehicles, without per-class prompting.

Table 4. RIS-LAD validation ablations. Peak-cell tests whether the highest-evidence cell overlaps target occupancy; Peak-in-mask tests whether its mapped center lies inside the target. Pred./GT is the predicted-to-target area ratio. Boundary IoU and mIoU are in %.

(a) LGSV Design and Language Conditioning
<table><tr><td>Configuration</td><td>Peak-cell</td><td>Peak-in-mask</td><td>mIoU</td></tr><tr><td>Single layer (L16)</td><td>59.57</td><td>53.01</td><td>51.71</td></tr><tr><td>Mismatched expression</td><td>41.76</td><td>37.43</td><td>47.78</td></tr><tr><td>Equal-capacity image-only</td><td>36.43</td><td>31.48</td><td>44.93</td></tr><tr><td>DRONESEGEDIT</td><td>61.48</td><td>55.07</td><td>53.15</td></tr></table>

(b) Generate-and-Verify Component Decomposition
<table><tr><td>Configuration</td><td>Pred./GT</td><td>Boundary IoU</td><td>mIoU</td></tr><tr><td>LGSV evidence only</td><td>2.489</td><td>27.68</td><td>26.12</td></tr><tr><td>Raw rendered candidate</td><td>1.507</td><td>53.83</td><td>50.03</td></tr><tr><td>Largest component</td><td>1.243</td><td>50.46</td><td>48.71</td></tr><tr><td>DRONESEGEDIT</td><td>1.230</td><td>58.24</td><td>53.15</td></tr></table>

Thresholding the complete P, unlike the proposal-restricted visualization in Fig. 2, yields only 26.12 mIoU and 27.68 Boundary IoU, with Pred./GT 2.489. The over-coverage is consistent with coarse occupancy supervision and upsampling. Thus the supervised readout supplies coarse selection evidence, while rendered proposals provide detailed geometry. Combining them raises candidate mIoU from 50.03 to 53.15 and Boundary IoU from 53.83 to 58.24.

The largest-component control and LGSV produce similar Pred./GT ratios (1.243 and 1.230), while LGSV improves mIoU by 4.44 points and Boundary IoU by 7.78 points. This comparison highlights the difference between foreground amount and spatial selection. Together with the mismatched-expression and image-only controls, it supports the role of expression-conditioned selection within the shared rendered proposals. Verification cannot recover omitted targets or reliably distinguish candidates with indistinguishable 37 × 37 evidence.

## 6. Discussion

Application-facing evidence. In UAV monitoring, unsupported foreground can become a false alarm. LGSV keeps corrections traceable to editor proposals, and the results support improved selection of compact vehicles and people.

Across all 2,763 RIS-LAD test expressions, verification lowers IoU in 40 cases (1.4%), with decreases of at least 0.05 IoU in only 20 cases (0.7%).

Scope and limitations. Experiments use LongCat-Image-Edit with LoRA, and matched controls assess the complementarity of rendering and verification. New dense label systems require adaptation, and larger palettes may increase color confusion. Connected color collisions, missed targets, coarse evidence, and reduced-resolution dense evaluation remain limitations. The large frozen editor and iterative sampling favor offline analysis; reducing memory use and sampling cost is a direction for onboard deployment.

Video and satellite extensions. Future work could align evidence across video frames and propagate masks while addressing motion, occlusion, and error accumulation. Extending the framework to satellite imagery also calls for adaptation to scale and viewpoint shifts.

## 7. Conclusion

DRONESEGEDIT pairs adapted RGB rendering with proposal-constrained language verification. It establishes a new state of the art on RIS-LAD and supports compact-target selection on two region-union benchmarks, with mixed aggregate results. A separately trained dense adapter uses the same RGB interface for full-scene parsing. Evaluating additional backbones and reducing generation time remain priorities.

## References

[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

[2] Wenxiao Cai, Ke Jin, Jinyan Hou, Cong Guo, Letian Wu, and Wankou Yang. VDD: Varied drone dataset for semantic segmentation. Journal ofVisual Communication and Image Representation, 109:104429, 2025.

[3] Jiaqi Chen, Jiachen Lu, Xiatian Zhu, and Li Zhang. Generative semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7111–7120, 2023.

[4] Keyan Chen, Jiafan Zhang, Chenyang Liu, Zhengxia Zou, and Zhenwei Shi. RSRefSeg: Referring remote sensing image segmentation with foundation models. In 2025 IEEE International Geoscience and Remote Sensing Symposium, pages 1070–1074, 2025.

[5] Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, and Ping Luo. AdaptFormer: Adapting vision transformers for scalable visual recognition. In Advances in Neural Information Processing Systems, pages 16664–16678, 2022.

[6] Yi-Chia Chen, Wei-Hua Li, Cheng Sun, Yu-Chiang Frank Wang, and Chu-Song Chen. SAM4MLLM: Enhance multi-modal large language model for referring expression segmentation. In European Conference on Computer Vision, pages 323–340, 2024.

[7] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1290–1299, 2022.

[8] Yong Xien Chng, Henry Zheng, Yizeng Han, Xuchong Qiu, and Gao Huang. Mask grounding for referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26573–26583, 2024.

[9] Henghui Ding, Chang Liu, Suchen Wang, and Xudong Jiang. Vision-language transformer and query generation for referring segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16321–16330, 2021.

[10] Valentin Gabeur, Shangbang Long, Songyou Peng, Paul Voigtlaender, Shuyang Sun, Yanan Bao, Karen Truong, Zhicheng Wang, Wenlei Zhou, Jonathan T. Barron, Kyle Genova, Nithish Kannen, Sherry Ben, Yandong Li, Mandy Guo, Suhas Yogin, Yiming Gu, Huizhong Chen, Oliver Wang, Saining Xie, Howard Zhou, Kaiming He, Thomas Funkhouser, Jean-Baptiste Alayrac, and Radu Soricut. Image generators are generalist vision learners. arXiv preprint arXiv:2604.20329, 2026.

[11] Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Houqiang Li, Han Hu, Dong Chen, and Baining Guo. InstructDiffusion: A generalist modeling interface for vision tasks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12709–12720, 2024.

[12] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

[13] Ronghang Hu, Marcus Rohrbach, and Trevor Darrell. Segmentation from natural language expressions. In European Conference on Computer Vision, pages 108–124, 2016.

[14] Koichiro Ito. Feature design for bridging SAM and CLIP toward referring image segmentation. In Proceedings of the Winter Conference on Applications ofComputer Vision, pages 8357–8367, 2025.

[15] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.

[16] Benedikt Kolbeinsson and Krystian Mikolajczyk. Multi-class segmentation from aerial views using recursive noise diffusion. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 8439–8449, 2024.

[17] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. LISA: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579–9589, 2024.

[18] Hsin-Ying Lee, Hung-Yu Tseng, Hsin-Ying Lee, and Ming-Hsuan Yang. Exploiting diffusion prior for generalizable dense prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7861–7871, 2024.

[19] Sen Lei, Xinyu Xiao, Tianlin Zhang, Heng-Chao Li, Zhenwei Shi, and Qing Zhu. Exploring fine-grained image-text alignment for referring remote sensing image segmentation. IEEE Transactions on Geoscience and Remote Sensing, 63: 1–11, 2025.

[20] Muchen Li and Leonid Sigal. Referring transformer: A one-step approach to multi-task visual grounding. In Advances in Neural Information Processing Systems, pages 19652–19664, 2021.

[21] Rui Li and Xiaowei Zhao. AeroReformer: Aerial referring transformer for UAV-based referring image segmentation. International Journal of Applied Earth Observation and Geoinformation, 143:104817, 2025.

[22] Sihan Liu, Yiwei Ma, Xiaoqing Zhang, Haowei Wang, Jiayi Ji, Xiaoshuai Sun, and Rongrong Ji. Rotated multi-scale interaction network for referring remote sensing image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26658–26668, 2024.

[23] Wei Liu, Jiaxin Lin, and Rui Chen. Open-source image editing models are zero-shot vision learners. arXiv preprint arXiv:2605.04566, 2026.

[24] Grace Luo, Trevor Darrell, Oliver Wang, Dan B. Goldman, and Aleksander Holynski. Readout guidance: Learning control from diffusion features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8217–8227, 2024.

[25] Ning Lv, Jisheng Dang, Teng Wang, Bimei Wang, Yichu Liu, Hong Peng, Haowen Yan, and Bin Hu. ReSaP: Reasoning-enhanced and scale-aware prompting for referring remote sensing image segmentation. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 19:8372–8383, 2026.

[26] Ye Lyu, George Vosselman, Gui-Song Xia, Alper Yilmaz, and Michael Ying Yang. UAVid: A semantic segmentation dataset for UAV imagery. ISPRS Journal ofPhotogrammetry and Remote Sensing, 165:108–119, 2020.

[27] Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, Xunliang Cai, Yayong Guan, and Jie Hu. LongCat-Image technical report. arXiv preprint arXiv:2512.07584, 2025.

[28] Hai Nguyen-Truong, E-Ro Nguyen, Tuan-Anh Vu, Minh-Triet Tran, Binh-Son Hua, and Sai-Kit Yeung. Vision-aware text features in referring image segmentation: From object understanding to context understanding. In Proceedings of the Winter Conference on Applications of Computer Vision, pages 4988–4998, 2025.

[29] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

[30] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018.

[31] Lu Qi, Lehan Yang, Weidong Guo, Yu Xu, Bo Du, Varun Jampani, and Ming-Hsuan Yang. UniGS: Unified representation for image generation and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6305–6315, 2024.

[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763, 2021.

[33] Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong Tang, Zheng Zhu, Guan Huang, Jie Zhou, and Jiwen Lu. DenseCLIP: Language-guided dense prediction with context-aware prompting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18082–18091, 2022.

[34] Linus Scheibenreif, Michael Mommert, and Damian Borth. Parameter efficient self-supervised geospatial

domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27841–27851, 2024.

[35] Chao Shang, Zichen Song, Heqian Qiu, Lanxiao Wang, Fanman Meng, and Hongliang Li. Prompt-driven referring image segmentation with instance contrasting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4124–4134, 2024.

[36] Leideng Shi and Juan Zhang. Multimodal-aware fusion network for referring remote sensing image segmentation. IEEE Geoscience and Remote Sensing Letters, 22:1–5, 2025.

[37] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6830–6839, 2023.

[38] Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, and Tiejun Huang. SegGPT: Towards segmenting everything in context. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1130–1140, 2023.

[39] Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yandong Guo, Mingming Gong, and Tongliang Liu. CRIS: CLIP-driven referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11686–11695, 2022.

[40] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In Proceedings of the European Conference on Computer Vision, pages 418–434, 2018.

[41] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, and Ping Luo. SegFormer: Simple and efficient design for semantic segmentation with transformers. In Advances in Neural Information Processing Systems, pages 12077–12090, 2021.

[42] Enze Xie, Lewei Yao, Han Shi, Zhili Liu, Daquan Zhou, Zhaoqiang Liu, Jiawei Li, and Zhenguo Li. DiffFit: Unlocking transferability of large diffusion models via simple parameter-efficient fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4230–4239, 2023.

[43] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2955–2966, 2023.

[44] Yuhuan Yang, Xianwei Zhuang, Yuxuan Cai, Chaofan Ma, Shuai Bai, Jiangchao Yao, Ya Zhang, Junyang Lin, and Yanfeng Wang. GenMask: Adapting DiT for segmentation via direct mask generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20455–20467, 2026.

[45] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip H. S. Torr. LAVT: Language-aware vision transformer for referring image segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18155–18165, 2022.

[46] Kai Ye, YingShi Luan, Zhudi Chen, Guangyue Meng, Pingyang Dai, and Liujuan Cao. RIS-LAD: A benchmark and model for referring image segmentation in low-altitude drone imagery. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 11937–11945, 2026.

[47] Zhenghang Yuan, Lichao Mou, Yuansheng Hua, and Xiao Xiang Zhu. RRSIS: Referring remote sensing image segmentation. IEEE Transactions on Geoscience and Remote Sensing, 62:1–12, 2024. Art. no. 5613312.

[48] Canyu Zhao, Yanlong Sun, Mingyu Liu, Huanyi Zheng, Muzhi Zhu, Zhiyue Zhao, Hao Chen, Tong He, and Chunhua Shen. DICEPTION: A generalist diffusion model for visual perceptual tasks. In Advances in Neural Information Processing Systems, pages 40108–40155, 2025.

[49] Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing text-to-image diffusion models for visual perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5729–5739, 2023.

Ground truth

## Appendix

Scope. This appendix provides qualitative evidence, operating limits, and implementation details. Paired referring comparisons reuse identical rendered RGB candidates, isolating verification; all settings and decisions are fixed before complete-split evaluation.

## A Qualitative Results and Operating Limits

## A.1 Successful verification

Figure A.1 shows the intended division of labor: the editor renders candidate geometry, while LGSV compares those regions using evidence conditioned on the complete expression.

Input

Rendered candidate

"The tricycle is pink and being driven by a person. It is located near the center of the image, slightly to the left."

![](images/822b9a70fee7a1b891b0ea774ff32ed1cf4b57158d33f8dc86d0416e750f27e2.jpg)

![](images/7288cfec092a5a9a542eaa659f31bbd9d5414f5189ab21e7395b6686db089c1f.jpg)  
Proposal-restricted LGSV evidence

![](images/3fbae86c10485bfe96dba431b8be8a7d3416201250935a53e1e762498a1f63f8.jpg)  
Verified mask

![](images/681c4c47ac69219c4e7066884cbda0a8f8612dd7f273915480cac7facb22e2e4.jpg)

![](images/d0d8694e7377168ac19fdc380a7f64f093abbb3d57ec79bf2a253ca2be3555b3.jpg)  
(b) RIS-LAD - Bus· Distractor correction  
"The bus is blue and driving on the road, positioned near the left edge of the image."

![](images/05331d70ecae52f67136d90d4c61d1e4cc2b0a9e29a191b98e03b09fa4800961.jpg)

![](images/b6a2cee2b64d8a8aa4ffdbfb34458713ac7616287a48be367e2fbc2376296793.jpg)

![](images/7045dd087b3cd3baa1d1e0b02c9b5401622f98b09f3080ce34fe8b419dc75f32.jpg)

![](images/76418f6e6cb3323238a719bb9572a2a1967dc8aa2fd9afbb64eb72b4dd3399c5.jpg)

![](images/745085393a74672098a12d024f4366e5e51aa1bdfce9d8fb2017d1c3d9e39052.jpg)

(c) RIS-LAD - Car · Multi-distractor correction  
![](images/d0e2605c51c182eaf2167516d378d800fafba7a926dd922d8f4a4ce5020cadb5.jpg)

![](images/94498173f12d537707238ad985ea006eede5620aa58b807d4eeb191960cb0737.jpg)  
(d) RIS-LAD - Car· Conservative verification

![](images/8488bae469f006be756c61d2824429c3fdfdd270b9d3ff3cc80dbdb82cffdaf0.jpg)

![](images/5993d61e0cdb396af4f0b5ee80b527843475ee655d7eded069367fad65fa1037.jpg)

![](images/ecfcae09ab8aabedc2c74de1e72f9e601b6e0876d38a1ab1156af8c8e6199e5e.jpg)  
"The white car is parked on the left side of the image, near the bottom, with a sunroof visible."

![](images/dea8e77b480d9bf91785b884ab3bf6371696f6784d1d07fe15e1f2bf999d00ba.jpg)

![](images/b09ec10caf01948768aabfffc49ccd71d70939d3bde43f2ff873530f08d7982c.jpg)

![](images/1e5d4c9b7651e5386a39971e19c51f0bd70ffcd20a621b22b374a14ce3134378.jpg)

![](images/4eef4aa9edf7cf2edd04f20bd0bd9d154bca7842f9b8ff05f49d7d05338c9032.jpg)

![](images/955a80df36214c15ea138b20f4713aecd754d67d83624eb88839818ce5e1de77.jpg)  
Figure A.1. RIS-LAD verification successes. Each row shows the input, fixed RGB-rendered candidate, proposal-restricted LGSV evidence, verified mask, and ground truth. The evidence panel displays $P \odot M _ { \mathrm { r a w } } \colon$ responses outside decoded candidates are suppressed only for visualization, while its sharp contours come from $M _ { \mathrm { r a w } }$ rather than $P .$ Green/red contours denote retained/rejected components; numerical labels report mean component evidence $s _ { i } .$ Panels (a–c) remove unsupported rendering, while (d) conservatively retains an accurate candidate.

For RIS-LAD, component scores are computed only over pixels inside each rendered component, and the final mask is constrained to $M _ { \mathrm { r a w } }$ . We therefore show proposal-restricted evidence for decision interpretability; this visualization does not imply that the full evidence map is zero outside the proposals.

## A.2 Region-union benchmark comparisons

We present UAVid-RIS and VDD-RIS jointly as region-union benchmarks because one expression can refer to multiple disconnected regions. Figure A.2 compares RMSIN, AeroReformer, and DroneSegEdit against the same ground truth using source-resolution crops. The examples complement the complete-split tables without repeating the main-paper cases.

Input

AeroReformer

DroneSegEdit

Ground truth

![](images/61e316adea14e81a1581fa64acfba729b70519fecd8919eb71f8f6bf8bf7f941.jpg)  
Figure A.2. Region-union benchmark comparisons. UAVid-RIS examples (a–b) and VDD-RIS examples (c–d) are shown in one aligned layout. DroneSegEdit more closely follows the requested moving, parked, and distributed vehicle unions than the comparison methods, while avoiding unrelated foreground. Every panel is rendered directly from the original image and its corresponding prediction mask; no per-example metric is displayed because complete-split results are reported in the main paper.

## A.3 Failure cases and diagnostic frequency

LGSV is proposal-constrained: $M _ { \mathrm { f i n a l } } \subseteq M _ { \mathrm { r a w } }$ . It can remove unsupported foreground already present in the candidate but cannot recover a target omitted by the editor. Its coarse evidence can also misrank nearby proposals, and component selection cannot divide an indivisible rendered component.

On all 2,763 RIS-LAD test expressions, verification lowers IoU in 40 cases (1.4%); only 20 cases (0.7%) decrease by at least 0.05 IoU.

![](images/af420ded55ff477e6242e3a62ec3b1822afe39dd631cb1ee002ef2536a845027.jpg)  
Figure A.3. Representative RIS-LAD operating limits. Top: a connected rendering merges the referred car with adjacent vehicles and cannot be divided by component selection. Bottom: proposal omission leaves no target foreground for LGSV to recover. Both cases have unambiguous referents and expose the one-sided scope of verification without changing the fixed candidate-generation protocol.

## B Implementation Details

## B.1 Notation

Table B.1 fixes notation across the main paper and appendix. Roman subscripts name semantic roles $( \mathrm { e . g . , \hbar ^ { * } r a w ^ { \prime } } )$ and “final”), t indexes tasks, ℓ indexes frozen MLLM depth, and x indexes pixels. Calligraphic letters denote sets or operators; bold lowercase symbols denote RGB colors or feature vectors.

Table B.1. Symbols shared by the main paper and appendix.
<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td> $t \in \tau$ </td><td>Task index, with  ${ \mathcal { T } } = { \mathcal { T } } _ { \mathrm { r e f } } \cup { \mathcal { T } } _ { \mathrm { s e m } } .$ </td></tr><tr><td> $\mathcal { T } _ { \mathrm { r e f } }$ </td><td>Referring tasks, partitioned as  $\mathcal T _ { \mathrm { s i n g l e } } \cup \mathcal T _ { \mathrm { u n i o n . } }$ </td></tr><tr><td> $I , e , Y _ { t }$ </td><td>Input RGB image, optional referring expression, and task target.</td></tr><tr><td> $\Omega , { \mathcal { C } } _ { t }$ </td><td>Pixel lattice and task label set.</td></tr><tr><td> $\theta _ { 0 } , \Delta _ { t }$ </td><td>Frozen editor parameters and task-specific LoRA update.</td></tr><tr><td> $I _ { g } , \mathcal { D } _ { \mathrm { p a l } } ^ { ( t ) }$ </td><td>Generated RGB canvas and deterministic task palette decoder.</td></tr><tr><td> $M _ { \mathrm { r a w } } , P , M _ { \mathrm { f i n a l } }$ </td><td>Decoded referring candidate, LGSV support map, and verified mask.</td></tr><tr><td> $F _ { \ell } ^ { v } , F _ { \ell } ^ { e } , q _ { \ell } , Z _ { \ell }$ </td><td>Frozen visual states, expression-token states, pooled expression state, and LGSV logits at depth l.</td></tr><tr><td> $( \gamma _ { \ell } , \beta _ { \ell } ) , \kappa$ </td><td>FiLM channel scale/shift and fixed residual modulation scale.</td></tr><tr><td> $O _ { \ell }$ </td><td>Coarse target occupancy at depth l.</td></tr><tr><td> $C _ { i } , \mathcal { E } , \mathcal { K } _ { \rho } , s _ { i }$ </td><td>Rendered component, eligible and retained index sets, and mean evidence score.</td></tr><tr><td> $\mathcal { H } _ { t } , \mathcal { N } _ { d _ { t } } ( x )$ </td><td>Thresholded evidence set and radius-dt Manhattan neighborhood.</td></tr></table>

## B.2 Data, RGB Interface, and Configurations

All reported referring results follow the official evaluation protocols and are computed on the complete official test splits with the benchmark-provided annotations and metrics. Checkpoints, decoding thresholds, and LGSV decision settings are selected on validation only; test annotations are not used for selection. Decoded masks are resized to each benchmark’s official evaluation resolution using nearest-neighbor interpolation before metric computation.

All tasks use the same frozen LongCat-Image-Edit base [27] with task-specific LoRA adaptation [12]. For fair comparison, each dataset/task follows its official training and evaluation protocol with an independently trained adapter, avoiding confounding gains from cross-task data mixing. For an adapted projection $W _ { j } \in \mathbb { R } ^ { d _ { j } ^ { \mathrm { o u t } } \times d _ { j } ^ { \mathrm { i n } } }$ , task t learns

$$
\begin{array} { c } { W _ { j } ^ { ( t ) } = W _ { j } + \Delta W _ { j } ^ { ( t ) } , \qquad \Delta W _ { j } ^ { ( t ) } = \displaystyle \frac { \alpha _ { j } } { r _ { j } } B _ { j } ^ { ( t ) } A _ { j } ^ { ( t ) } , } \\ { \mathrm { r a n k } ( \Delta W _ { j } ^ { ( t ) } ) \leq r _ { j } , } \end{array}\tag{B.1}
$$

where $B _ { i } ^ { ( t ) } \in \mathbb { R } ^ { d _ { j } ^ { \mathrm { o u t } } \times r _ { j } }$ and $A _ { i } ^ { ( t ) } \in \mathbb { R } ^ { r _ { j } \times d _ { j } ^ { \mathrm { i n } } }$ . For referring tasks, the deterministic palette decoder identifies magenta-like pixels from the generated 8-bit channels: red and blue must be high, green must be low, and both red–green and blue–green must exceed a fixed margin. A pixel is decoded as foreground if and only if

$$
R \geq t _ { R } \wedge B \geq t _ { B } \wedge G \leq t _ { G } \wedge R - G \geq t _ { \Delta } \wedge B - G \geq t _ { \Delta } .\tag{B.2}
$$

This joint rule rejects bright or gray scene pixels that satisfy only the absolute-channel tests. The resulting binary mask is $M _ { \mathrm { r a w } }$ and the same thresholds are used for every sample.

RIS-LAD. We use the fixed instruction template

Color the only {target class} described as "{expression}" pure magenta. Keep   
everything else unchanged.

The category placeholder does not introduce external label information. The official RIS-LAD formulation obtains its class-level feature by extracting the category token from the referring expression [46]. Accordingly, {target class} is deterministically extracted from the provided expression; no ground-truth category metadata or test annotation is accessed.

UAVid-RIS and VDD-RIS. Similarly, for the two region-union benchmarks, we use

Paint the {class name} region matching "{expression}" with pure magenta.

Dense VDD. We deliberately use a simple task-level instruction:

Convert the full image into an RGB semantic label map.

The prompt contains neither class names nor palette-color constraints. Instead, the complete class-to-color correspondence and joint full-scene parsing behavior are learned from paired RGB targets during task-specific LoRA SFT. Thus language specifies only the task, while supervision supplies the label inventory and output protocol. All templates are fixed before evaluation, and no per-example prompt selection is performed.

Shared settings. All LoRAs use rank 32 and α = 8 in the DiT self-/cross-attention Q/K/V/output and self-/context-FFN projections; Qwen2.5-VL, the base DiT, and the VAE remain frozen. Rendering LoRAs use BF16 AdamW, weight decay $\bar { 1 } 0 ^ { - 2 }$ , uniform diffusion-timestep sampling, and no stochastic image augmentation. Referring inference uses a $1 0 2 \bar { 4 } ^ { 2 }$ canvas, FlowMatch Euler sampling with dynamic shifting, 8 steps, CFG 2.5, and $( t _ { R } , t _ { B } , t _ { G } , t _ { \Delta } ) = ( 1 9 0 , 1 7 0 , 1 0 0 , 3 0 )$ . LGSV is optimized separately over frozen Qwen2.5-VL states using BF16 AdamW, batch 2, and weight decay $1 0 ^ { - 2 }$ . It uses layers {8, 16, 24, 28} and width 256; its fixed settings are $( \lambda _ { b } , \lambda _ { d } , \lambda _ { c } ) = ( 1 , 1 , 0 . 5 0 ) , \kappa = 0 . 1 0$ , positive-weight cap $\omega _ { \mathrm { m a x } } = 5 0$ , and Dice smoothing ϵ = 1. Region-union tasks omit only the center-cell term. LoRA contains 47.19M optimized parameters, increasing to 62.98M when the 15.79M LGSV heads are trained for a referring task.

Table B.2. Training and inference configurations. Rendering-LoRA schedules and checkpoints are listed separately from LGSV optimization. LoRA checkpoints determine the fixed RGB candidates, whereas LGSV checkpoints are selected independently on validation. RIS-LAD uses consecutive constant-LR adaptation and lower-LR cosine-decay continuation stages; all other LoRA schedules are single-stage.
<table><tr><td>Module / task</td><td>Optimization</td><td>Schedule</td><td>Selected checkpoint</td><td>Inference</td></tr><tr><td>Rendering LoRA: RIS-LAD Rendering LoRA:</td><td>AdamW; LR 10−4; batch 4. Continuation: 5× 10−5 AdamW; LR 10−4; batch 7k constant</td><td>2k constant-LR adaptation; 2k cosine-decay continuation</td><td>2k adaptation + 2k continuation (val.)</td><td>10242; 8 steps; CFG 2.5; fixed decoder 10242; 8 steps; CFG 2.5; fixed</td></tr><tr><td>UAVid-RIS Rendering LoRA:</td><td>8 AdamW; LR 10−4; batch 7k constant</td><td></td><td>6k (val.)</td><td>decoder 10242; 8 steps; CFG 2.5; fixed</td></tr><tr><td>VDD-RIS Rendering LoRA:</td><td>8</td><td></td><td>5k (val.)</td><td>decoder 1152 × 864; 12 steps; CFG 1.5;</td></tr><tr><td>VDD dense</td><td>AdamW; LR 10−4; batch 7k constant 2</td><td></td><td>5k (val.)</td><td>nearest palette; no LGSV</td></tr><tr><td>LGSV readout: referring tasks</td><td>BF16 AdamW; batch 2; weight decay 10−2; LR  $1 0 ^ { - 4 }$ </td><td>At most 24k optimizer updates</td><td>Best on validation</td><td>Layers {8, 16, 24, 28}; width 256</td></tr></table>

Complete adaptation takes around 20 NVIDIA A800 GPU-hours on RIS-LAD and around 30 GPU-hours on each of UAVid-RIS and VDD-RIS. GPU-hours denote the number of GPUs multiplied by training wall-clock time for one benchmark-specific run. Across RIS-LAD, UAVid-RIS, and VDD-RIS, LGSV shares the same readout layers, width, FiLM parameterization, optimizer family, batch size, weight decay, and BCE/Dice weights. RIS-LAD additionally activates the center-cell term, while each benchmark retains its validation-selected LGSV checkpoint and output-topology decision rule. The fixed decision settings are: RIS-LAD uses $A _ { \operatorname* { m i n } } = 1 6$ , mean component evidence, $\rho = 0 . 7 5$ , and 8-connectivity; UAVid-RIS uses $( \tau _ { t } , d _ { t } ) = ( 0 . 6 0 , 1 6 )$ and VDD-RIS uses $( \tau _ { t } , d _ { t } ) = ( 0 . 3 0 , 8 )$

During integrated inference, LGSV reuses the multi-level hidden states already produced by the editor’s Qwen conditioning pass, so verification requires no second Qwen forward. LGSV training and cached-output diagnostics instead run the frozen Qwen encoder separately for offline feature extraction.

The rendering-LoRA checkpoint is frozen before paired Raw/LGSV evaluation. LGSV therefore changes only the language-conditioned verification readout and does not resample or update the RGB candidate generator. Dense VDD uses the rendering LoRA alone.

## B.3 LGSV Supervision Details

The main paper specifies the verifier architecture, loss, and shared subset operator. Here we make the coarse supervision target explicit. At depth $\ell ,$ adaptive max pooling maps the task target mask $Y _ { t }$ to the evidence grid:

$$
O _ { \ell } = \mathbf { 1 } [ \mathrm { A d a M a x P o o l } _ { \ell } ( Y _ { t } ) > 0 ] .\tag{B.3}
$$

Here $O _ { \ell }$ is target occupancy. Full-grid BCE labels non-target cells as negatives under expression conditioning, while Dice penalizes excessive predicted support. The BCE positive weight uses the background-to-target occupancy ratio capped by $\omega _ { \mathrm { m a x } } .$ and the center term uses the target-centroid cell only for RIS-LAD. Adaptive max pooling intentionally supplies occupancy rather than downsampled contours: LGSV learns coarse expression support, while the rendered candidate supplies the proposal geometry. The fixed coefficients are listed above Table B.2.

## B.4 Ablation Protocol Details

All ablations follow the same RIS-LAD validation protocol. Matched LGSV design comparisons reuse fixed rendered proposals. The mismatched-expression control disrupts language–target correspondence, whereas the equal-capacity image-only control removes expression modulation while retaining readout capacity. The evidence-only control directly uses the validation-fixed evidence map without a rendered proposal, and the largest-component control retains the largest eligible decoded component.

## C Additional Diagnostics

## C.1 Single-Instruction Full-Scene VDD Parsing

A separately trained dense VDD adapter receives a complete UAV image and jointly renders seven labels at $1 1 5 2 \times 8 6 4$ preserving the native 4:3 aspect ratio [2]. The fixed instruction in Sec. B.2 contains neither class names nor colors; paired RGB targets teach their correspondence, without per-class prompt enumeration or repeated class-specific calls.

To measure acquisition of this implicit output protocol, we compute the distance from each generated pixel to its nearest valid palette color,

$$
d _ { \mathrm { p a l } } ( I _ { g } ) = \frac { 1 } { | \Omega | } \sum _ { x \in \Omega } \operatorname* { m i n } _ { \mathbf { c } \in \mathcal { C } _ { \mathrm { p a l } } } \| I _ { g } ( x ) - \mathbf { c } \| _ { 2 } .\tag{C.1}
$$

At the selected checkpoint, the mean distance is 11.96 in 8-bit RGB space, while the minimum distance between any two palette colors is 87.61. Generated pixels are, on average, substantially closer to a valid palette entry than the minimum inter-palette separation, supporting stable nearest-palette decoding. Figures C.1 and C.2 show representative outputs and the convergence of this diagnostic.

![](images/887b32312d5082440d69c8ac6af3adea742b13b26fe36a4bb4a62715cf25da7a.jpg)  
Figure C.1. Single-instruction dense VDD outputs. The adapter renders all classes jointly as RGB, and nearest-palette decoding maps the continuous output to discrete labels. The two rows illustrate a multi-class urban scene and a water–vegetation boundary.

![](images/e007785badeff29554038f19155b749e14e08b9e7661d3fb85807356a42828ee.jpg)  
Figure C.2. Nearest-palette distance during dense VDD adaptation. The selected checkpoint lies in the converged low-distance regime; th final mean distance (11.96) remains far below the minimum inter-palette separation (87.61).

For reproducibility, the ordered palette is
<table><tr><td> $\mathbf { c } _ { 0 }$ </td><td>other</td><td>(35, 35, 35)</td><td> $\mathbf { c } _ { 1 }$ </td><td>wall</td><td>(190, 160, 110)</td></tr><tr><td> $\mathbf { c } _ { 2 }$ </td><td>road</td><td>(115, 115, 115)</td><td> $\mathbf { c } _ { 3 }$ </td><td>vegetation</td><td>(45,170,60)</td></tr><tr><td> $\mathbf { c } _ { 4 }$ </td><td>vehicle</td><td>(245, 220, 40)</td><td> $\mathbf { c } _ { 5 }$ </td><td>roof</td><td>(190, 55, 35)</td></tr><tr><td> $\mathbf { c } _ { 6 }$ </td><td>water</td><td>(0, 145, 220)</td><td></td><td></td><td></td></tr></table>

The generated RGB image is decoded by nearest Euclidean distance to these colors, with the lower class ID used only to break an exact tie. All seven decoded classes, including other, enter the reported mIoU; no pixel is ignored or left unassigned.

Under the complete seven-class protocol at 1152×864 input, the selected checkpoint obtains 80.96 mIoU. This verifier-free experiment evaluates the shared RGB output interface. The dense adapter is trained separately, without transferring the trained referring adapter or LGSV. It remains a closed-set, task-specific adaptation: changing the label system requires retraining or adaptation, and larger label sets may introduce palette crowding or color confusion. A frozen-VAE reconstruction check reaches 99.296 mIoU on 64 validation images after the same palette readout, indicating limited reconstruction-and-decoding loss on this validation subset.

## C.2 Complementary Foreground Diagnostics

Table C.1. Complementary diagnostics not repeated in the main comparison tables. Each LongCat-Image-Edit–DRONESEGEDIT pair reuses the same rendered RGB candidate. mDice is averaged over expressions, whereas global Precision, Recall, and Pred/GT pool pixels over the evaluation set. Values are percentages except Pred/GT, for which values closer to one are preferred.
<table><tr><td>Dataset</td><td>Output</td><td>mDice</td><td>Global Precision</td><td>Global Recall</td><td>Pred/GT</td></tr><tr><td rowspan="2">RIS-LAD</td><td>LongCat-Image-Edit</td><td>54.37</td><td>52.93</td><td>81.04</td><td>1.531</td></tr><tr><td>DRONESEGEDIT</td><td>58.88</td><td>62.10</td><td>80.57</td><td>1.297</td></tr><tr><td rowspan="2">VDD-RIS</td><td>LongCat-Image-Edit</td><td>85.69</td><td>80.44</td><td>94.20</td><td>1.171</td></tr><tr><td>DRONESEGEDIT</td><td>86.90</td><td>87.70</td><td>92.11</td><td>1.050</td></tr><tr><td rowspan="2">UAVid-RIS</td><td>LongCat-Image-Edit</td><td>80.04</td><td>76.93</td><td>93.47</td><td>1.215</td></tr><tr><td>DRONESEGEDIT</td><td>83.05</td><td>85.84</td><td>90.38</td><td>1.053</td></tr></table>

Across all three paired evaluations, verification raises global precision and moves Pred/GT toward one, with a small-to-moderate global-recall trade-off. These diagnostics support the main paper’s interpretation that LGSV primarily removes unsupported rendered foreground from a fixed candidate.

## C.3 Scope of Deployment and Future Extensions

Model scale and deployment. The referring system combines a 47.19M-parameter rendering LoRA with a 15.79M-parameter LGSV (62.98M trainable parameters total), while Qwen2.5-VL, the 6B base DiT, and the VAE remain frozen. Referring inference uses eight sampling steps with CFG 2.5. Integrated LGSV reuses the editor’s Qwen conditioning states, avoiding an additional Qwen forward. The frozen backbone and iterative sampling favor offline analysis. Model compression and fewer-step sampling are directions for onboard deployment.

Temporal UAV imagery. The current method processes individual images. A video extension could align consecutive evidence maps and propagate verified masks using motion or feature correspondences while retaining the per-frame proposal constraint. Sequence-level evaluation would assess motion, viewpoint changes, occlusion, identity switches, and error accumulation.

Satellite imagery. Extending the RGB rendering interface to satellite imagery is another direction for future work. Differences in ground sampling distance, viewpoint, object scale, and referring expressions motivate task-specific adaptation and evaluation.

## D Responsible Use

The intended scope is civilian research on public UAV benchmarks. The method is not intended for military targeting or weapon guidance, coercive surveillance, biometric identification, or autonomous high-stakes decisions. Any deployment requires legal review, data minimization, human oversight, and task-specific validation; third-party data and models remain governed by their original licenses.