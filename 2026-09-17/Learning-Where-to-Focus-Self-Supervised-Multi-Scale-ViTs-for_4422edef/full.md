# Learning Where to Focus: Self-Supervised Multi-Scale ViTs for Histopathology

<sub>Anabel</sub> <sub>Stammer</sub>⋆[0009-0009-4885-0630]<sub>,</sub> <sub>Valay</sub>

<sub>Bundele</sub>⋆⋆[0000-0003-2140-9019]<sub>,</sub> <sub>Mehran</sub>

<sub>Hosseinzadeh</sub>⋆⋆[0009-0000-1114-1595]<sub>,</sub> <sub>and</sub> <sub>Hendrik</sub> <sub>P.A.</sub>

<sub>Lensch</sub>[0000-0003-3616-8668]

Eberhard Karls Universit¨at T¨ubingen, Germany {anabel.stammer,valay.bundele,mehran.hosseinzadeh, hendrik.lensch}@uni-tuebingen.de

Abstract. Pathologists diagnose diseases by first locating suspicious tissue and then examining it at higher magnification, whereas self-supervised vision transformers (ViTs) allocate the same spatial resolution to every image region despite diagnostic evidence being sparse and spanning multiple biological scales. Recent pathology foundation models have substantially improved representation quality by scaling training data and model capacity, but largely retain uniform tokenization. We instead investigate whether pathology representations can be improved by learning where to allocate spatial resolution during self-supervised learning. To this end, we propose CRAFT (Coarse-to-fine Region-Adaptive Feature Tokenization), a DINO-based framework that learns image-dependent mixed-scale representations by using self-supervised attention to selectively refine informative regions while preserving coarse context, together with a symmetric cross-scale regularization objective that encourages complementary coarse and fine representations. Across CAMELYON16, TCGA-Lung subtype classification, and TCGA-LUAD survival prediction, CRAFT consistently outperforms comparable-scale self-supervised methods while requiring lower inference computation. Despite using only a compact 22M parameter backbone trained on comparatively small pathology datasets, CRAFT remains competitive with, and often surpasses, substantially larger pathology foundation models.

Keywords: Histopathology · Multi-Scale Learning · Self-Supervision

## 1 Introduction

Pathology is as much a search problem as a recognition problem: in whole-slide images (WSIs), diagnostically relevant tissue is sparse and spans multiple biological scales [9]. Pathologists navigate this by scanning the slide broadly, then examining only suspicious regions at higher magnification. Most vision transformers (ViTs) for computational pathology instead tokenize every region at a single fixed scale, forcing one representation to capture both global context and fine cellular detail.

![](images/44553b23826ee6ec8a4df3c0c855c92e3f1bf308666949952d6e472a4b446a42.jpg)  
(a)

![](images/b1dd864e0113133f7475df0e3696359cbb71c41665f536077d8c383ea35b7153.jpg)  
(b)  
Fig. 1: (a) Class attention maps guide mixed-scale tokenization by refining salient regions while keeping others coarse. (b) Tumor recall via linear probing on CAMELYON16 after SSL pretraining at two token sizes (16 vs 32).

Recently, self-supervised learning (SSL) has substantially advanced pathology representation learning, leading to foundation models such as UNI [10], Virchow [47], and Prov-GigaPath [53]. Their success is largely driven by scaling training data and model capacity while retaining uniform tokenization. We instead ask a complementary question: can representation quality be improved by learning where to allocate spatial resolution during self-supervised learning?

Our motivation comes from a simple but surprising observation. Uniformly finer tokenization is not necessarily beneficial for pathology. As shown in Fig. 1(b), on CAMELYON16 [3], a DINO-pretrained ViT using coarse 32 × 32 tokens achieves substantially higher tumor recall under linear probing than the same architecture pretrained with uniformly finer 16 × 16 tokens (81.4% vs. 69.0%). This does not suggest that cellular detail is unimportant. Rather, it suggests that much of the diagnostic signal is already captured at coarse resolution, while allocating fine resolution uniformly can spend representational capacity where coarse context is suficient. The appropriate inductive bias is therefore not uniformly coarse or uniformly fine, but coarse first, fine where needed.

Learning such adaptive resolution during self-supervised pretraining is challenging. Without patch-level annotations, the model must first determine which regions deserve refinement. Moreover, once coarse and refined representations coexist, the refined representation must contribute complementary information rather than simply reproduce the coarse one. Both problems must be solved for adaptive tokenization to improve representation quality.

Existing methods address only parts of this problem. Token reduction methods such as ToMe [4], EViT [33], STELLAR [58], and SimPrune [32] reduce computation by pruning or merging tokens after they are formed. WSI-specific methods typically improve eficiency after tile features have been extracted through instance selection, sparse aggregation, or redesigned Multiple Instance Learning (MIL) pipelines [2,43,26,54,40]. Multi-scale pathology encoders such as HIPT [9] and PLUTO [27] use predefined hierarchies rather than adapting resolution to each input, while CF-ViT [8] explores coarse-to-fine tokenization for natural images using supervised region labels that are rarely available in histopathology. Consequently, existing methods do not learn where additional spatial resolution should be allocated during self-supervised representation learning.

We address this challenge with CRAFT (Coarse-to-fine Region-Adaptive Feature Tokenization), a DINO-based self-supervised framework that learns mixed-resolution representations without region supervision. CRAFT first constructs a coarse representation of the image and uses coarse-stage class attention as a label-free saliency signal to identify informative regions. Only these regions are re-tokenized at finer resolution, while the remaining regions retain their coarse representation (Fig. 1(a)). The refined features are fused with their parent coarse features, allowing local morphology to complement broader tissue context. Crucially, refinement is not heuristic nor supervised: DINO’s cross-view alignment objective implicitly drives the model to allocate higher resolution to regions that most reduce representation discrepancy across views. Moreover, to ensure refinement contributes genuinely new information, CRAFT further introduces a symmetric cross-scale KL regularizer that aligns coarse and refined predictions while discouraging redundant representations.

On CAMELYON16, TCGA-Lung subtyping, and TCGA-LUAD survival prediction, CRAFT outperforms comparable-scale methods while requiring lower inference compute. Despite using only a compact 22M-parameter backbone trained on comparatively small datasets, CRAFT achieves performance competitive with large-architecture pathology foundation models trained on substantially larger datasets. These results demonstrate that learning where to allocate spatial resolution during SSL pretraining yields more efective representations than uniformly processing all WSI regions.

## 2 Related Works

Self-Supervised Learning in Histopathology. Self-supervised learning has become the dominant paradigm for learning pathology representations from large unlabeled WSI data, where dense expert annotation is impractical. Contrastive learning [13,51], self-distillation [5,36], and masked-image modeling [48,19] have demonstrated strong transfer across downstream pathology tasks. More recently, pathology foundation models such as UNI [10] and Virchow [47] have shown that scaling DINO-style pretraining with larger datasets and backbones further improves representation quality. Despite these advances, existing SSL methods uniformly tokenize every image region. In contrast, CRAFT learns where to allocate spatial resolution during self-supervised representation learning.

Eficient and Adaptive Vision Transformers. Eficient Vision Transformers reduce computation by adaptively processing the token sequence. Representative approaches prune, merge, or sample informative tokens [33,17,30,4,59,32], while others introduce hierarchical or coarse-to-fine representations [45,23,8]. Although efective for natural-image recognition, these methods either discard tokens or rely on supervised guidance for adaptive refinement. CRAFT instead performs region selection using self-supervised class attention and learns coarse and refined representations through cross-scale regularization, enabling adaptive spatial allocation without region-level supervision.

Multi-Scale and Eficient WSI Modeling. The hierarchical organization of tissue has motivated multi-scale pathology models such as HIPT [9] and PLUTO [27], which explicitly model information across magnification levels. Separately, eficient WSI analysis has been explored through instance selection, sparse aggregation, zooming strategies, and feature compression within multipleinstance learning pipelines [25,57,41,46,2,43,26,54,40]. These approaches primarily operate after patch representations have been extracted or rely on predefined multi-scale hierarchies. In contrast, CRAFT learns image-dependent resolution allocation during patch-level self-supervised learning, producing unified mixedscale representations before slide-level aggregation.

## 3 Methodology

Preliminaries. DINO [6] trains a student network to match a stop-gradient teacher over multiple augmented views of the same image. The student receives global and local crops, the teacher receives global crops, and both networks map the [CLS] token through a projection head to a probability distribution over prototypes. The teacher is an exponential moving average of the student, with centering and sharpening used to avoid collapse. We build on DINO because local-to-global self-distillation naturally couples fine evidence with broader context, matching the hierarchical structure of histopathology [11].

CRAFT. CRAFT turns the coarse-first principle into a self-supervised tokenization module. Instead of assigning one patch size to every image region, CRAFT forms a full-coverage mixed-scale sequence: a coarse pass preserves global tissue context, and a refined pass spends fine tokens only on regions selected from the coarse representation. The mixed-scale tokenization module is inserted symmetrically in the student and teacher branches of DINO (Fig. 2a), so resolution allocation is learned during feature formation rather than applied as post-hoc compression.

Unlike CF-ViT [8], which uses label supervision to guide coarse-to-fine refinement, CRAFT learns region selection directly from self-supervised pre-training. This introduces a constraint single-network methods do not face: the tokenizer must act in both branches, and student and teacher must select consistently or the distillation target becomes unstable. We find the two branches converge to near-identical selections without an explicit cross-branch mechanism (App. I), unlike single-branch pruning schemes [32]. Refinement is further optimized jointly with DINO through a novel symmetric KL objective encouraging refined representations to capture complementary information while remaining semantically aligned with their coarse counterparts.

![](images/f00ace8c40c31fa2aea42c4a4f404d9a4afa347d2d415924c39cb93a848aa29e.jpg)  
Fig. 2: (a) Mixed-scale tokenization (green, Ours) is inserted into both DINO branches before the ViT; the rest is standard DINO, with gradients through CRAFT: Coarse-to-fine Regthe student only. Coarse and refined predictions $P _ { c } ^ { s } , P _ { r } ^ { s }$ ive Feature Tokeniare aligned by ${ \mathcal { L } } _ { \mathrm { K L } }$ -sym (Eq. 5). (b) Coarse-to-fine pipeline: Stage 1 encodes coarse tokens and selects key regions by class attention; Stage 2 re-splits and fuses them (⊕).

Mixed-scale Tokenization. As shown in Fig. 2b, our mixed-scale tokenization operates in two stages, applied in both teacher and student branches.

Stage 1: Coarse encoding and region selection. The input is first partitioned into $N _ { c }$ non-overlapping coarse tokens of size $3 2 \times 3 2$ pixels to which learnable coarse positional encodings are added. The resulting token sequence $X _ { c } ^ { 0 }$ is processed by a ViT backbone $\mathcal { E }$ with M transformer layers, producing the coarse representation $X _ { c } ^ { M }$ . Following DINO [6], we use the [CLS]-to-patch attention as a label-free estimate of regional importance. Let $s _ { \ell } \in \mathbb { R } ^ { N _ { c } }$ denote the attention scores at layer ℓ. Following CF-ViT [8], attention maps from intermediate-todeep layers are aggregated using an exponential moving average,

$$
\bar { s } _ { \ell } = \beta \bar { s } _ { \ell - 1 } + ( 1 - \beta ) s _ { \ell } , \qquad \ell = 4 , \ldots , M ,\tag{1}
$$

where $\beta$ is the momentum coeficient. The top $\alpha N _ { c }$ tokens according to $\bar { s } _ { M }$ are selected for refinement, where α controls the accuracy–compute trade-of.

Stage 2: Refinement and Multi-scale Fusion. Selected regions are refined at higher resolution; the rest keep their coarse tokens. We partition the coarse set as $S _ { C } = S _ { C F } \cup S _ { C C }$ into a selected part $S _ { C F }$ (refined) and a retained part $S _ { C C } .$ and let $S _ { F }$ be the fine tokens obtained by subdividing each selected $3 2 \times 3 2$ token into four $1 6 \times 1 6$ tokens. Re-initializing fine tokens would discard the context learned during coarse encoding; instead, we project each parent feature through a lightweight projector $g$ and inject it into its children, so refinement builds upon the coarse representation [8]:

$$
\tilde { X } _ { r } ^ { 0 } = [ \mathrm { C L S } ] _ { r } \cup S _ { C C } \cup \big ( S _ { F } \oplus g ( S _ { C F } ) \big ) .\tag{2}
$$

Here $[ \mathrm { C L S } ] _ { r }$ is the refined-stage class token and $\oplus$ adds each parent coarse feature to its four children after broadcasting to the fine grid. The mixed-scale sequence has $N _ { r } = \lfloor ( 1 - \alpha ) N _ { c } \rfloor + 4 \lceil \alpha N _ { c } \rceil$ , spatial tokens and is encoded by the same ViT backbone, $\tilde { X } _ { r } ^ { M } = \mathcal { E } ( \tilde { X } _ { r } ^ { 0 } )$ , giving the refined [CLS] output $z _ { r } = \tilde { X } _ { r } ^ { M } [ 0 ]$

Loss Formulation. Let $z _ { c } ^ { s } = X _ { c } ^ { M } [ 0 ]$ and $z _ { r } ^ { s } = \tilde { X } _ { r } ^ { M } [ 0 ]$ denote the coarse and refined student representations, with $z _ { c } ^ { t }$ and $z _ { r } ^ { t }$ the corresponding teacher representations. To encourage representation learning and refinement toward diagnostically relevant regions, we apply standard DINO objective to the refinedstage outputs of teacher and student (Fig. 2a). Let $P _ { c } ^ { s } ( x ) = h ^ { \prime } ( z _ { c } ^ { s } ( x ) ) , P _ { r } ^ { s } ( x ) =$ $h ^ { \prime } ( z _ { r } ^ { s } ( x ) )$ denote the coarse and refined student distributions, and $P _ { c } ^ { t } ( x ) , P _ { r } ^ { t } ( x )$ the corresponding teacher distributions. The standard DINO objective is

$$
L _ { \mathrm { D I N O } } = \sum _ { x _ { 1 } , x _ { 2 } } { \mathrm { C E } } \left( P _ { r } ^ { t } ( x _ { 1 } ) , P _ { r } ^ { s } ( x _ { 2 } ) \right) ,\tag{3}
$$

where $h ^ { \prime } ( \cdot ) = \sigma ( h ( \cdot ) )$ ) denotes the softmax-normalized DINO projection head, and $x _ { 1 } , x _ { 2 }$ are two augmented views of the same image. Since the refined representation already contains both coarse and refined tokens, an additional coarsestage supervision is unnecessary.

Although the DINO objective supervises the final mixed-scale representation, it does not explicitly regulate the relationship between the coarse and refined representations. A natural solution is to align them through a cross-scale $\mathrm { K L } -$ divergence objective, as used in CF-ViT [8]. Let $P _ { c } ^ { s } ( x _ { i } )$ and ${ P } _ { r } ^ { s } ( x _ { i } )$ denote the coarse and refined student predictions for view $x _ { i }$ . The asymmetric objective is

$$
\mathcal { L } _ { \mathrm { K L - a s y m } } = \sum _ { i \in \{ 1 , 2 \} } \mathrm { K L } ( \mathrm { s g } ( P _ { r } ^ { s } ( x _ { i } ) ) \parallel P _ { c } ^ { s } ( x _ { i } ) ) ,\tag{4}
$$

where sg(·) denotes the stop-gradient operator. This objective encourages the coarse representation to align with the refined prediction, but places no constraint on what the refined representation should contribute beyond coarse context. Since coarse representations already encode substantial diagnostic information, the refined branch can converge to a redundant copy, limiting the benefit of selective refinement. To encourage complementary specialization, we instead introduce a symmetric cross-scale KL objective on the student outputs,

$$
{ \mathcal { L } } _ { \mathrm { K L - s y m } } = \sum _ { i \in \{ 1 , 2 \} } { \mathrm { K L } } ( \mathrm { s g } ( P _ { r } ^ { s } ( x _ { i } ) ) \parallel P _ { c } ^ { s } ( x _ { i } ) ) - \lambda { \mathrm { K L } } ( \mathrm { s g } ( P _ { c } ^ { s } ( x _ { i } ) ) \parallel P _ { r } ^ { s } ( x _ { i } ) ) ,\tag{5}
$$

where $\lambda = 0 . 5$ . The first term preserves semantic alignment between coarse and refined predictions, while the second discourages the refined representation from collapsing to the coarse representation, encouraging it to capture complementary information. The weighting parameter λ balances these competing objectives: small values lead to redundant representations, whereas excessively large values weaken the DINO alignment signal.

We formulate cross-scale regularization in the prediction space rather than the embedding space. Since the DINO projection head already outputs probability distributions over prototypes, KL divergence naturally compares coarse and refined predictions without introducing additional contrastive objectives, negative samples, or pairwise similarity computations $( \mathrm { e . g . }$ , InfoNCE [37]/CLIP [39]). This keeps the auxiliary objective consistent with DINO while allowing us to explicitly control the relationship between the two representations through the proposed push–pull formulation. The symmetric formulation is particularly wellsuited to our setting, where the coarse and refined branches are intended to be complementary rather than identical. Together, the DINO objective and symmetric cross-scale regularization encourage refined tokens to preserve global tissue context while contributing additional local morphological information. The overall training objective is, thus defined as, $\mathcal { L } = L _ { \mathrm { D I N O } } + \mathcal { L } _ { \mathrm { K L - s y m } }$

## 4 Experiments

Experimental Setup. We evaluate slide-level discrimination, survival risk prediction, computational cost, and representation quality across our experiments.

Datasets. We use two public WSI cohorts. (1) CAMELYON16 [3] contains 270 training and 129 test breast-cancer WSIs. (2) TCGA-NSCLC [14] contains 1,042 lung WSIs (530 LUAD / 512 LUSC) and supports two evaluations: LUAD-vs.-LUSC subtype classification and LUAD survival prediction. We use the patient-stratified Snufy split [26]; after quality filtering, 638 slides (333 LUAD / 305 LUSC) are used for training and 256 slides (115 LUAD / 141 LUSC) form the test set. All splits are patient-disjoint. For survival prediction, we restrict evaluation to the LUAD subset, following standard single-subtype WSI survival benchmarks.

Patch extraction and inference modes. After background removal, WSIs are tiled into $2 2 4 \times 2 2 4$ patches at 20× magnification. CRAFT and all samerecipe SSL baselines use ViT-S, matching the method’s eficiency goal and prior evidence that ViT-S is a strong DINO backbone for pathology [28]. We evaluate two inference modes: (1) Coarse-32 uses a global $3 2 \times 3 2$ token grid, yielding 49 tokens at 1.16 GMACs per tile; (2) Mixed-16/32 refines the top 50% highattention regions into $1 6 \times 1 6$ tokens, yielding 124 tokens at 4.01 GMACs.

Downstream Tasks. Patch encoders are frozen and evaluated with four MIL aggregators: ABMIL (AB), DSMIL (DS), TransMIL (Trans), and maxpooling (Max). CAMELYON16 uses the oficial test set with a fixed 5-seed MIL protocol. TCGA-NSCLC follows the DSMIL protocol [31]: five-fold model selection, fixed test set. Classification performance is reported as AUC. Patch classification uses linear probing on frozen encoder features [6], with labels derived from the oficial CAMELYON16 tumor annotations [3].

For TCGA-LUAD survival, we report concordance index (C-index), the fraction of comparable patient pairs whose predicted risk ordering matches observed survival. Following [55], we use CLAM-Survival on our encoder, whose first fully connected layer maps each encoder’s feature dimension to a shared hidden size, controlling for feature-dimension efects across encoders. Published C-indices from [55] are used for the six reported baselines; CRAFT and H0-mini are evaluated in our pipeline with the same setup.

Comparison and compute protocol. Unless otherwise specified, published encoders are run under our evaluation pipeline. External-pretraining rows are treated as foundation-model references; controlled claims are made against same-size or same-recipe baselines. Compute is reported as GMACs for one 224×224 forward pass, measured directly where possible; ResNet-50, ViT-B/16, and Swin-T use standard literature estimates. For HIPT, cost is dominated by the cell-level ViT-S/16 (≈4.6 GMACs), while its $\mathrm { V i T _ { 4 0 9 6 } / V i T _ { W S I } }$ stages amortize to <0.02 GMACs/tile.

Statistical and label-free analyses. Significance uses Welch’s t-test over five seeds (CRAFT mixed-16/32 vs. same-recipe DINO ViT-S, CAMELYON16 DSMIL). To assess ${ \mathcal { L } } _ { \mathrm { K L - s y m } }$ beyond downstream labels, we probe the teacher’s refined [CLS] feature $z _ { r }$ on CAMELYON16 test patches. We report alignment [49] and three spectral descriptors defined in App. A.

Implementation Details. We implement CRAFT using ViT-Small within the DINO framework [6], training on RTX 4090 GPUs. Pretraining follows a cosine learning rate schedule with AdamW and base LR of $5 \times 1 0 ^ { - 4 }$ . In CRAFT, we set $\beta = 0 . 9 9$ and $\alpha = 0 . 5$ , resulting in $N _ { c } = 4 9$ and $N _ { r } = 1 2 4$ . For reference, uniform ViT-S/16 uses $N _ { f } = 1 9 6$ tokens. On CAMELYON16 and TCGA-LUNG, DINO variants are pretrained for 100 and 50 epochs, respectively. Afterwards, each MIL training is performed for 50 epochs.

Results. We report WSI classification on CAMELYON16 and TCGA-NSCLC, followed by TCGA-LUAD survival prediction.

CAMELYON16 classification. Table 1 evaluates frozen patch encoders on the oficial split using four MIL aggregators. This setting tests whether the encoder improvement is stable across slide-level heads.

The primary control is the same-recipe DINO ViT-S baseline, which matches the backbone, pretraining data, and MIL protocol, thus isolating the efect of pretraining objective. CRAFT mixed-16/32 improves DSMIL by +2.22 AUC points $( p = 0 . 0 0 4$ , Welch’s t-test) and maintains AUC above 96% for all four aggregators. Since the MIL heads are trained on frozen patch features, this crossaggregator consistency indicates an encoder-level gain rather than an interaction with a particular slide classifier.

This improvement is obtained under a lower inference budget than uniform ViT-S/16. Mixed-16/32 evaluates 124 tokens per tile instead of 196, reducing compute from 4.6 to 4.01 GMACs while improving the matched DINO baseline.

Table 1: WSI classification AUC (%) on CAMELYON16 (mean ± std over 5 runs). Among reproduced rows, best bold, second underlined; † external pretraining; Avg = mean over DS/AB/Trans/Max MILs.
<table><tr><td>Method</td><td>Encoder</td><td>Pretrain</td><td>DS</td><td>AB</td><td>Trans</td><td>Max</td><td>Avg</td><td>#P</td><td>GMACs</td></tr><tr><td colspan="10">Reproduced under our 5-seed MIL protocol</td></tr><tr><td>UNI†[10]</td><td>ViT-L/16 DINOv2</td><td></td><td>96.06±1.85</td><td>90.04±2.62</td><td>97.26±0.74</td><td>97.10±1.19</td><td>95.12 307M</td><td></td><td>59.70</td></tr><tr><td>Virchow†[47]</td><td>ViT-H/14 DINOv2</td><td></td><td>95.00±1.74</td><td>93.95±2.60</td><td>94.90±1.68</td><td>93.59±0.63</td><td>94.36 632M</td><td></td><td>161.99</td></tr><tr><td>CONCH†[34]</td><td></td><td>ViT-B/16 iBOT+CoCa</td><td>92.47±0.98</td><td>92.34±0.89</td><td>95.77±1.16 90.49±0.33</td><td></td><td>92.77</td><td>86M</td><td>16.87</td></tr><tr><td>H0-mini†[18]]</td><td>ViT-B/14 DINOv2</td><td></td><td>98.23±0.81</td><td></td><td>97.58±0.3497.87±0.37 99.22±0.0098.2386M</td><td></td><td></td><td></td><td>22.31</td></tr><tr><td>CTransPath†[50]</td><td>Swin-T</td><td>SRCL</td><td>85.62±2.34</td><td>88.08±1.95</td><td>96.31±0.56</td><td>90.45±2.22</td><td>90.12</td><td>28M</td><td>4.51</td></tr><tr><td>Lunit†[28]</td><td>ViT-S/16 DINO</td><td></td><td>86.44±18.5</td><td>92.96±3.17</td><td>95.98±1.51</td><td>87.82±17.9</td><td>90.80</td><td>22M</td><td>4.6</td></tr><tr><td>DINOv2[38]</td><td>ViT-S/16 DINOv2</td><td></td><td>95.75±4.83</td><td>94.78±1.33</td><td>96.27±0.94</td><td>89.52±17.6</td><td>94.08</td><td>22M</td><td>4.6</td></tr><tr><td>DINO[6]</td><td>ViT-S/16 DINO</td><td></td><td>96.21±0.30</td><td>96.26±0.80</td><td>95.69±1.47</td><td>95.95±0.98</td><td>96.03</td><td>22M</td><td>4.6</td></tr><tr><td>CRAFT coarse-32</td><td>ViT-S/16 DINO</td><td></td><td>94.89±0.62</td><td>95.36±0.90</td><td>93.19±0.59</td><td>95.21±0.71</td><td>94.66</td><td>22M</td><td>1.16</td></tr><tr><td>CRAFT mixed-16/32 ViT-S/16 DINO</td><td></td><td></td><td>98.43±0.93</td><td>96.29±1.36</td><td>96.71±0.88</td><td>97.18±0.74</td><td>97.15</td><td>22M</td><td>4.01</td></tr></table>

Coarse-32 provides a more aggressive operating point, reducing compute to 1.16 GMACs and still reaching 94.66 mean AUC. Owing to cross-scale alignment during training, coarse tokens remain discriminative, enabling competitive performance with significantly reduced computation. Thus, CRAFT improves the controlled high-accuracy setting and, with the same trained encoder, provides a substantially cheaper inference mode when throughput is prioritized.

The foundation-model rows contextualize this trade-of. Although CRAFT uses no external pretraining and retains a 22M ViT-S backbone, mixed-16/32 achieves the second-best average AUC in the table and the best DSMIL AUC. Relative to H0-mini, the only method with higher average AUC, CRAFT is +0.20 AUC higher on DSMIL and within 1.1 AUC points on average while requiring 5.6× fewer GMACs per tile. It also uses 14.9× fewer GMACs than UNI and 40.4× fewer than Virchow. These results place CRAFT on a favorable accuracy–compute frontier: near-foundation-model CAMELYON16 performance with a compact encoder and single-digit tile-level GMACs.

TCGA-NSCLC subtyping. Table 2 reports WSI subtyping on TCGA-NSCLC (LUAD vs. LUSC), averaged over four MIL aggregators; per-aggregator results with std are in App. C. This tests whether the mixed-scale encoder generalizes beyond CAMELYON16 to a diferent organ and task. Among comparable 22M encoders, CRAFT mixed-16/32 obtains the highest mean AUC, exceeding MoCov3/DINO [35] by +1.24 points and CAMELYON16-pretrained DINO transfer baseline by +10.92 points. The two CRAFT inference modes then separate the role of adaptive refinement: coarse-32 already reaches 95.89 AUC at 1.16 GMACs per tile, while mixed-16/32 raises performance to 96.24 AUC at 4.01 GMACs. The consistent per-aggregator gains in App. C indicate that refined tokens add subtype-discriminative information without degrading coarse representation.

The foundation-model comparison shows how far this compact setting can be pushed against models trained at much larger scale. CRAFT is within 0.6 AUC of UNI and Virchow while using a 22M backbone and requiring 14.9× and 40.4× fewer GMACs per tile, respectively. It also exceeds Lunit by +2.11 points and CTransPath by +3.43 points; these rows are marked with ¶ because their pretraining includes TCGA slides and may overlap the fixed test distribution. H0-mini remains the strongest row, but combines a larger ViT-B encoder, distillation from a billion-scale teacher, and the same TCGA-overlap caveat. Taken together, the matched 22M rows establish the controlled gain, while the foundation-model rows show that CRAFT approaches large-scale pretrained encoders on TCGA-NSCLC at substantially lower tile-level compute.

Table 2: WSI subtype classification on TCGA-Lung (LUAD vs. LUSC), AUC (%) on the fixed 256-slide test set, averaged over four MIL aggregators. Each aggregator value is the mean over five model-selection folds. Top: public encoders trained on massive data; bottom: low-data 22M encoders. Best bold, second underlined among 22M encoders. ¶ pretraining overlaps our TCGA test split, so AUC may be inflated.
<table><tr><td>Method</td><td>Encoder</td><td>Pretraining</td><td>Mean AUC</td><td>#P</td></tr><tr><td colspan="5">Frozen public Encoder trained on Massive Data</td></tr><tr><td>UNI [10]</td><td>ViT-L/16</td><td>DINOv2, Mass-100K</td><td>96.81</td><td>307M</td></tr><tr><td>Virchow [47]</td><td>ViT-H/14</td><td>DINOv2, MSK-1.5M</td><td>96.67</td><td>632M</td></tr><tr><td> $\mathrm { { H 0 } { - } \mathrm { { m i n i } ^ { \dag } \left[ 1 \dot { 8 } \right] } }$ </td><td>ViT-B/14</td><td>DINOv2 distill, TCGA</td><td>97.79</td><td>86M</td></tr><tr><td> $\mathrm { C T r a n s P a t h } ^ { \bullet }$  [50]</td><td>Swin-T</td><td>SRCL, TCGA+PAIP</td><td>92.81</td><td>28M</td></tr><tr><td>Maate Dassita RetCCL</td><td>ResNet-50</td><td>TCGA+PAIP (CCL)</td><td>85.61</td><td>23.5M</td></tr><tr><td>Lunit [28]</td><td>ViT-S/16</td><td>DINO, TCGA</td><td>94.13</td><td>22M</td></tr><tr><td>Barlow Twins [35] [56]</td><td>ViT-S/16</td><td>Barlow, TCGA-LUNG</td><td>91.7</td><td>22M</td></tr><tr><td>MoCo v3 [35][12]</td><td>ViT-S/16</td><td>MoCo v3, TCGA-LUNG</td><td>95.0</td><td>22M</td></tr><tr><td>DINO [35] [6]</td><td>ViT-S/16</td><td>DINO, TCGA-LUNG</td><td>95.0</td><td>22M</td></tr><tr><td>DINO ViT-S</td><td>ViT-S/16</td><td>DINO, CAM16</td><td>85.32</td><td>22M</td></tr><tr><td>IN ViT-S [16,44]</td><td>ViT-S/16</td><td>sup., IN-21k→1k</td><td>84.08</td><td>22M</td></tr><tr><td>Ours coarse-32</td><td>ViT-S/16</td><td>DINO, TCGA-LUNG</td><td>95.89</td><td>22M</td></tr><tr><td>Ste Dm-Dcata Ours mixed-16/32</td><td>ViT-S/16</td><td>DINO, TCGA-LUNG</td><td>96.24</td><td>22M</td></tr></table>

TCGA-LUAD survival. Fig. 3 compares TCGA-LUAD C-index; full values are in App. D. Survival prediction evaluates whether the frozen encoder transfers from diagnostic discrimination to patient-level risk ordering under a fixed survival head. CRAFT mixed-16/32 attains 56.9±2.06 C-index, exceeding H0-mini by +0.4 points, GigaPath by +0.7 points, and trailing only UNI by 0.2 points. It also has lower cross-fold variance than the other top-scoring methods except GigaPath. This performance requires only 4.01 GMACs per tile: about 5.9× less than H0-mini, over 15× less than UNI, and over 50× less than GigaPath. Coarse-32 reaches 55.49±2.30 C-index at 1.16 GMACs, preserving a low-compute survival setting.

To further assess representation quality, we evaluate driver-gene mutation prediction on TCGA-LUAD (App. B). Despite using a compact ViT-S backbone, CRAFT remains competitive with pathology foundation models up to 7× larger.

Ablation Studies. Table 3 evaluates the contributions of coarse-to-fine tokenization, multi-scale fusion, and cross-scale regularization. Linear probing measures tile-level representation quality, while MIL evaluates the transfer of frozen features to slide-level prediction. The ablations support three main conclusions.

![](images/f4ed958aaeeb78cd68aaaa69667f7e7ff4d44640456a3cc361ce2f5818b40489.jpg)  
Fig. 3: Survival prediction on TCGA-LUAD (selected encoders). Bars show mean ± std across 5 folds; color encodes compute (GMACs).

Table 3: Ablations on CAMELYON16. MIL metrics average four aggregators (DS, AB, Trans, Max). Best and second-best marked. TS: Token Sizes; MF: Multi-scale Fusion.
<table><tr><td rowspan="2">ID Model</td><td rowspan="2">TS</td><td rowspan="2"></td><td rowspan="2">MF LKL-asym LKL-sym</td><td rowspan="2"></td><td colspan="3">Linear Probing</td><td colspan="2">MIL</td><td rowspan="2"></td></tr><tr><td></td><td></td><td>ACC Prec Recall Avg AUC Avg Recall GMACs</td><td></td><td></td></tr><tr><td>B1 Baseline</td><td>16</td><td></td><td></td><td></td><td>97.35</td><td>90.8</td><td>69.0</td><td>96.03</td><td>95.61</td><td>4.6</td></tr><tr><td>B2 Baseline</td><td>32</td><td></td><td></td><td></td><td>98.05</td><td>89.2</td><td>81.4</td><td>94.59</td><td>93.68</td><td>1.16</td></tr><tr><td colspan="10">CRAFT operating points</td><td></td></tr><tr><td>O1 CRAFT 16</td><td></td><td>√</td><td>√</td><td>X</td><td>98.16 92.3</td><td></td><td>80.7</td><td>96.74</td><td>96.74</td><td>5.76</td></tr><tr><td>O2 CRAFT 16/32</td><td></td><td>√</td><td>√</td><td>X</td><td>97.80 93.8</td><td></td><td>73.2</td><td>95.65</td><td>94.19</td><td>4.01</td></tr><tr><td>O3 CRAFT 16/32</td><td></td><td>√</td><td>X</td><td>V</td><td>98.62 93.2</td><td></td><td>77.8</td><td>97.15</td><td>97.25</td><td>4.01</td></tr><tr><td colspan="10">Coarse-branch operating points</td></tr><tr><td>E1 CRAFT</td><td>32</td><td>√</td><td>√</td><td>X</td><td></td><td>97.85 93.1</td><td>74.7</td><td>94.83</td><td>94.47</td><td>1.16</td></tr><tr><td>E2 CRAFT</td><td>32</td><td>√</td><td>X</td><td>√</td><td>97.98</td><td>93.4</td><td>76.5</td><td>94.66</td><td>92.45</td><td>1.16</td></tr></table>

Table 4: Label-free representation analysis on CAME-LYON16 test (L<sub>KL-asym</sub> vs. ${ \mathcal { L } } _ { \mathrm { K L - s y m } } )$ . Mean±std over 5 runs; best bold.
<table><tr><td>Metric</td><td> $\mathcal { L } _ { \mathrm { K L - a s y m } }$ </td><td> $\mathcal { L } _ { \mathrm { K L - s y m } }$ </td></tr><tr><td>Alignment ↓</td><td></td><td>0.234±0.003 0.225±0.003</td></tr><tr><td>RankMe ↑</td><td>263.0±0.8</td><td>264.4±0.8</td></tr><tr><td>Effective rank ↑</td><td>275.4±0.7</td><td>279.7±0.7</td></tr><tr><td>α-ReQ (→ 1)</td><td>1.61±0.00</td><td>1.48±0.00</td></tr><tr><td>Max-MIL AUC ↑</td><td>95.7</td><td>97.2</td></tr></table>

First, fine-scale representations benefit from incorporating coarse contextual information. The coarse-only baseline (B2) outperforms the fine-only baseline (B1) on tile-level recall (81.4% vs. 69.0%) yet underperforms at slide level, indicating it lacks the fine detail needed for robust WSI classification. The uniformly refined CRAFT variant (O1) instead improves on both: MIL Avg AUC by 0.71 pp over B1 (96.03 → 96.74) and tile-level recall by 0.11 pp over B2, despite the same 16 × 16 token resolution. Combining coarse context with fine detail thus yields a small but cross-task-robust gain.

Second, selectively allocating fine resolution is more efective than uniformly refining every region, but only when coupled with proposed ${ \mathcal { L } } _ { \mathrm { K L - s y m } }$

Compared with uniform refinement (O1), refining only the most salient half of the image (O2) reduces computation from 5.76 to 4.01 GMACs, but also degrades downstream performance when trained with the asymmetric cross-scale objective. Replacing the asymmetric objective with the proposed symmetric formulation (O2→O3) recovers and further improves performance, increasing MIL Avg AUC from 95.65 to 97.15 at the same computational cost. Notably, O3 outperforms the more expensive uniformly refined model (O1), showing that selectively allocating spatial resolution is more efective than uniformly increasing it.

![](images/9a6ffa09491eaa7131f29ba3abcd2248dff9375ba5e97d8f5b1d0949046488ea.jpg)  
(a)

![](images/be8edbee2faffe900595e4de4f82363e710442a8d9ee463f63b1ac7acb62a4ab.jpg)  
(b)  
Fig. 4: (a) Mixed-scale tokenization on CAMELYON16 patches: key regions are refined to fine tokens while background and stroma stay coarse. (b) Selfattention maps for selected heads, comparing DINO (fine) with CRAFT under selective and coarse inference; input shown left.

Finally, $\mathcal { L } _ { \mathrm { K L - a s y m } }$ encourages redundancy: coarse-only features (E1) outperform mixed-scale (O2) in linear probing (97.85 vs. 97.80). In contrast, L<sub>KL-sym</sub> restores complementarity: mixed-scale features (O3) surpass coarse-only (E2) in both LP (98.62 vs. 97.98) and MIL (97.15 vs. 94.66). Interestingly, ${ \mathcal { L } } _ { \mathrm { K L - s y m } }$ also improves coarse-only features (97.98 vs. 97.85), indicating that enforcing complementary specialization yields better-structured representations at both scales, not only when combined. This supports our design: the symmetric term specializes the two scales rather than collapsing them together. The same trend holds across λ (App. E). A component analysis can be found at App. F.

Representation-Level Efect of ${ \mathcal { L } } _ { \mathbf { K L - s y m } } .$ . As shown in Table 4, the symmetric loss yields more view-invariant features: alignment, which is the distance between embeddings of two augmented views, drops (0.225 vs. 0.234, a ∼3σ gap). The three spectrum-entropy measures (RankMe [22], efective rank [42], and the eigenspectrum-decay exponent α [1], all quantifying how well the embedding space is utilized) favour ${ \mathcal { L } } _ { \mathrm { K L - s y m } }$ , indicating capacity is used broadly with no sign of collapse. These entropy- and invariance-based metrics are validated by downstream predictors (RankMe by design; alignment intrinsic to the SSL objective) and track the Max-MIL AUC gain, so the symmetric loss redistributes capacity towards a flatter spectrum and stronger cross-view consistency.

Visual Analysis. Fig. 4 (a) illustrates the adaptive mixed-scale tokenization: high-attention regions with dense cellular structures are refined into fineresolution tokens, while background and stromal regions remain coarse. Fig. 4 (b) shows self-attention maps from selected heads. CRAFT (mixed-scale) closely matches DINO (fine), confirming that mixed-scale tokenization preserves diagnostic features. Even under coarse-only inference, similar attention patterns persist, demonstrating efective multi-scale distillation.

## 5 Conclusion

We introduced CRAFT, a self-supervised framework that learns where to allocate spatial resolution during DINO pretraining, selectively refining salient regions while preserving coarse context. A symmetric cross-scale KL objective encourages complementary coarse and fine representations. Across multiple pathology tasks, CRAFT surpasses comparable-scale SSL baselines and remains competitive with substantially larger pathology foundation models at much lower inference cost, despite using a compact 22M-parameter backbone trained on modest data. These results suggest that selective spatial resolution allocation is an efficient alternative to uniform tokenization. Future work includes replacing the fixed refinement ratio α with an image-adaptive budget and extending the symmetric objective to patch-level distillation (e.g., iBOT), where masked patch targets remain misaligned across scales.

## Acknowledgements

The work described in this paper was conducted in the framework of the Graduate School 2543/2 “Intraoperative Multi-Sensory Tissue Diferentiation in Oncology” (project ID 40947457) funded by the German Research Foundation (DFG - Deutsche Forschungsgemeinschaft). This work has been supported by the Deutsche Forschungsgemeinschaft (DFG) – EXC number 2064/1 – Project number 390727645 and SFB 1233, TP 1, Project number 276693517. The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting A. Stammer, V. Bundele and M. Hosseinzadeh.

## References

1. Agrawal, K.K., Mondal, A.K., Ghosh, A., Richards, B.A.: α-ReQ: Assessing representation quality in self-supervised learning by measuring eigenspectrum decay. In: NeurIPS (2022), https://openreview.net/forum?id=ii9X4vtZGTZ

2. Ara´ujo, D.J., Verdelho, M.R., Bissoto, A., Nascimento, J.C., Santiago, C., Barata, C.: Key patches are all you need: A multiple instance learning framework for robust medical diagnosis (2024), https://arxiv.org/abs/2405.01654

3. Bejnordi, B.E., Veta, M., Van Diest, P.J., Van Ginneken, B., Karssemeijer, N., Litjens, G., Van Der Laak, J.A., Hermsen, M., Manson, Q.F., Balkenhol, M., et al.: Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. Jama 318(22), 2199–2210 (2017)

4. Bolya, D., Fu, C.Y., Dai, X., Zhang, P., Feichtenhofer, C., Hofman, J.: Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461 (2022)

5. Campanella, G., Kwan, R., Fluder, E., Zeng, J., Stock, A., Veremis, B., Polydorides, A.D., Hedvat, C., Schoenfeld, A., Vanderbilt, C., et al.: Computational pathology at health system scale–self-supervised foundation models from three billion images. arXiv preprint arXiv:2310.07033 (2023)

6. Caron, M., Touvron, H., Misra, I., J´egou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers (2021), https://arxiv. org/abs/2104.14294

7. Cerami, E., Gao, J., Dogrusoz, U., Gross, B.E., Sumer, S.O., Aksoy, B.A., Jacobsen, A., Byrne, C.J., Heuer, M.L., Larsson, E., Antipin, Y., Reva, B., Goldberg, A.P., Sander, C., Schultz, N.: The cbio cancer genomics portal: an open platform for exploring multidimensional cancer genomics data. Cancer Discov. 2(5), 401–404 (May 2012)

8. Chen, M., Lin, M., Li, K., Shen, Y., Wu, Y., Chao, F., Ji, R.: Cf-vit: A general coarse-to-fine method for vision transformer. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 7042–7052 (2023)

9. Chen, R.J., Chen, C., Li, Y., Chen, T.Y., Trister, A.D., Krishnan, R.G., Mahmood, F.: Scaling vision transformers to gigapixel images via hierarchical self-supervised learning (2022), https://arxiv.org/abs/2206.02647

10. Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., Jaume, G., Chen, B., Zhang, A., Shao, D., Song, A.H., Shaban, M., Williams, M., Vaidya, A., Sahai, S., Oldenburg, L., Weishaupt, L.L., Wang, J.J., Williams, W., Le, L.P., Gerber, G., Mahmood, F.: A general-purpose self-supervised model for computational pathology (2023), https://arxiv.org/abs/2308.15474

11. Chen, R.J., Krishnan, R.G.: Self-supervised vision transformers learn visual concepts in histopathology (2022), https://arxiv.org/abs/2203.00585

12. Chen, X., Xie, S., He, K.: An empirical study of training self-supervised vision transformers (2021), https://arxiv.org/abs/2104.02057

13. Ciga, O., Xu, T., Martel, A.L.: Self supervised contrastive learning for digital histopathology. Machine Learning with Applications 7, 100198 (2022)

14. Cooper, L.A., Demicco, E.G., Saltz, J.H., Powell, R.T., Rao, A., Lazar, A.J.: Pancancer insights from the cancer genome atlas: the pathologist’s perspective. The Journal of pathology 244, 512–524 (2018)

15. Coudray, N., Ocampo, P.S., Sakellaropoulos, T., Narula, N., Snuderl, M., Feny¨o, D., Moreira, A.L., Razavian, N., Tsirigos, A.: Classification and mutation prediction from non-small cell lung cancer histopathology images using deep learning. Nat. Med. 24(10) (Oct 2018)

16. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale (2021), https://arxiv.org/abs/2010.11929

17. Fayyaz, M., Koohpayegani, S.A., Jafari, F.R., Sengupta, S., Joze, H.R.V., Sommerlade, E., Pirsiavash, H., Gall, J.: Adaptive token sampling for eficient vision transformers. In: European Conference on Computer Vision. pp. 396–414. Springer (2022)

18. Filiot, A., Dop, N., Tchita, O., Riou, A., Dubois, R., Peeters, T., Valter, D., Scalbert, M., Saillard, C., Robin, G., Olivier, A.: Distilling foundation models for robust and eficient models in digital pathology (2025), https://arxiv.org/abs/2501.16239

19. Filiot, A., Ghermi, R., Olivier, A., Jacob, P., Fidon, L., Camara, A., Mac Kain, A., Saillard, C., Schiratti, J.B.: Scaling self-supervised learning for histopathology with masked image modeling. medRxiv pp. 2023–07 (2023)

20. Gao, J., Aksoy, B.A., Dogrusoz, U., Dresdner, G., Gross, B., Sumer, S.O., Sun, Y., Jacobsen, A., Sinha, R., Larsson, E., Cerami, E., Sander, C., Schultz, N.: Integrative analysis of complex cancer genomics and clinical profiles using the cBioPortal. Sci. Signal. 6(269), l1 (Apr 2013)

21. Gao, P., Trautmann, E., Yu, B., Santhanam, G., Ryu, S., Shenoy, K., Ganguli, S.: A theory of multineuronal dimensionality, dynamics and measurement. Journal Name (11 2017). https://doi.org/10.1101/214262

22. Garrido, Q., Balestriero, R., Najman, L., Lecun, Y.: Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank (2023), https://arxiv.org/abs/2210.02885

23. Havtorn, J.D., Royer, A., Blankevoort, T., Bejnordi, B.E.: Msvit: Dynamic mixedscale tokenization for vision transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 838–848 (2023)

24. Howard, F., Dolezal, J., Kochanny, S., Schulte, J., Chen, H., Heij, L., Huo, D., Nanda, R., Olopade, O., Kather, J., Cipriani, N., Grossman, R., Pearson, A.: The impact of site-specific digital histology signatures on deep learning model accuracy and bias. Nature Communications 12 (07 2021). https://doi.org/10.1038/ s41467-021-24698-1

25. Ilse, M., Tomczak, J.M., Welling, M.: Attention-based deep multiple instance learning (2018), https://arxiv.org/abs/1802.04712

26. Jafarinia, H., Alipanah, A., Razavi, S., Mirzaie, N., Rohban, M.H.: Snufy: Eficient whole slide image classifier. In: European Conference on Computer Vision. pp. 243– 260. Springer (2024)

27. Juyal, D., Padigela, H., Shah, C., Shenker, D., Harguindeguy, N., Liu, Y., Martin, B., Zhang, Y., Nercessian, M., Markey, M., Finberg, I., Luu, K., Borders, D., Javed, S.A., Krause, E., Biju, R., Sood, A., Ma, A., Nyman, J., Shamshoian, J., Chhor, G., Sanghavi, D., Thibault, M., Yu, L., Najdawi, F., Hipp, J.A., Fahy, D., Glass, B., Walk, E., Abel, J., Pokkalla, H., Beck, A.H., Grullon, S.: Pluto: Pathologyuniversal transformer (2024), https://arxiv.org/abs/2405.07905

28. Kang, M., Song, H., Park, S., Yoo, D., Pereira, S.: Benchmarking self-supervised learning on diverse pathology datasets (2023), https://arxiv.org/abs/2212.04690

29. Kather, J.N., Krisam, J., Charoentong, P., Luedde, T., Herpel, E., Weis, C.A., Gaiser, T., Marx, A., Valous, N.A., Ferber, D., Jansen, L., Reyes-Aldasoro, C.C., Z¨ornig, I., J¨ager, D., Brenner, H., Chang-Claude, J., Hofmeister, M., Halama, N.: Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. PLoS Med 16(1), e1002730 (Jan 2019)

30. Kong, Z., Dong, P., Ma, X., Meng, X., Niu, W., Sun, M., Shen, X., Yuan, G., Ren, B., Tang, H., et al.: Spvit: Enabling faster vision transformers via latency-aware soft token pruning. In: European conference on computer vision. pp. 620–640. Springer (2022)

31. Li, B., Li, Y., Eliceiri, K.W.: Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14318–14328 (2021)

32. Li, S., Tan, Q., Dai, Y., Kong, Z., Wang, T., Liu, J., Li, A., Liu, N., Ding, Y., Tang, X., Yuan, G.: Mutual efort for eficiency: A similarity-based token pruning for vision transformers in self-supervised learning. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum? id=GTcEe5fayC

33. Liang, Y., Ge, C., Tong, Z., Song, Y., Wang, J., Xie, P.: Not all patches are what you need: Expediting vision transformers via token reorganizations. arXiv preprint arXiv:2202.07800 (2022)

34. Lu, M.Y., Chen, B., Williamson, D.F.K., Chen, R.J., Liang, I., Ding, T., Jaume, G., Odintsov, I., Le, L.P., Gerber, G., Parwani, A.V., Zhang, A., Mahmood, F.: A visual-language foundation model for computational pathology. Nature Medicine 30(3), 863–874 (Mar 2024)

35. Mammadov, A., Folgoc, L.L., Adam, J., Buronfosse, A., Hayem, G., Hocquet, G., Gori, P.: Self-supervision enhances instance-based multiple instance learning methods in digital pathology: A benchmark study (2025), https://arxiv.org/abs/ 2505.01109

36. Nasiri-Sarvi, A., Trinh, V.Q.H., Rivaz, H., Hosseini, M.S.: Vim4path: Selfsupervised vision mamba for histopathology images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6894– 6903 (2024)

37. van den Oord, A., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding (2019), https://arxiv.org/abs/1807.03748

38. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision (2024), https://arxiv.org/abs/ 2304.07193

39. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision (2021), https://arxiv.org/abs/ 2103.00020

40. Rahman, T., Tarkhan, A., Chellappa, R., Baras, A.: DTC-WSI: Dynamic token compression for whole slide images. In: Medical Imaging with Deep Learning (2026), https://openreview.net/forum?id=yMkSDJc445

41. Raza, M., Awan, R., Bashir, R.M.S., Qaiser, T., Rajpoot, N.M.: Dual attention model with reinforcement learning for classification of histology whole-slide images. Computerized Medical Imaging and Graphics 118, 102466 (2024). https://doi.org/ https://doi.org/10.1016/j.compmedimag.2024.102466, https://www.sciencedirect. com/science/article/pii/S0895611124001435

42. Roy, O., Vetterli, M.: The efective rank: A measure of efective dimensionality. 2007 15th European Signal Processing Conference pp. 606–610 (2007), https:// api.semanticscholar.org/CorpusID:12184201

43. Stegm¨uller, T., Bozorgtabar, B., Spahr, A., Thiran, J.P.: Scorenet: Learning nonuniform attention and augmentation for transformer-based histopathological image classification (2022), https://arxiv.org/abs/2202.07570

44. Steiner, A., Kolesnikov, A., Zhai, X., Wightman, R., Uszkoreit, J., Beyer, L.: How to train your vit? data, augmentation, and regularization in vision transformers (2022), https://arxiv.org/abs/2106.10270

45. Tang, S., Zhang, J., Zhu, S., Tan, P.: Quadtree attention for vision transformers. arXiv preprint arXiv:2201.02767 (2022)

46. Thandiackal, K., Chen, B., Pati, P., Jaume, G., Williamson, D.F.K., Gabrani, M., Goksel, O.: Diferentiable zooming for multiple instance learning on whole-slide images (2022), https://arxiv.org/abs/2204.12454

47. Vorontsov, E., Bozkurt, A., Casson, A., Shaikovski, G., Zelechowski, M., Liu, S., Severson, K., Zimmermann, E., Hall, J., Tenenholtz, N., Fusi, N., Mathieu, P., van Eck, A., Lee, D., Viret, J., Robert, E., Wang, Y.K., Kunz, J.D., Lee, M.C.H., Bernhard, J., Godrich, R.A., Oakley, G., Millar, E., Hanna, M., Retamero, J., Moye, W.A., Yousfi, R., Kanan, C., Klimstra, D., Rothrock, B., Fuchs, T.J.: Virchow: A million-slide digital pathology foundation model (2024), https://arxiv.org/abs/2309.07778

48. Wang, J., Quan, H., Wang, C., Yang, G.: Pyramid-based self-supervised learning for histopathological image classification. Computers in Biology and Medicine 165, 107336 (2023)

49. Wang, T., Isola, P.: Understanding contrastive representation learning through alignment and uniformity on the hypersphere (2022), https://arxiv.org/abs/2005. 10242

50. Wang, X., Yang, S., Zhang, J., Wang, M., Zhang, J., Yang, W., Huang, J., Han, X.: Transformer-based unsupervised contrastive learning for histopathological image classification. Medical Image Analysis 81, 102559 (2022)

51. Wessels, F., Schmitt, M., Krieghof-Henning, E., Nientiedt, M., Waldbillig, F., Neuberger, M., Kriegmair, M.C., Kowalewski, K.F., Worst, T.S., Steeg, M., et al.: A self-supervised vision transformer to predict survival from histopathology in renal cell carcinoma. World Journal of Urology 41(8), 2233–2241 (2023)

52. Wong, B., Hong, S., Yi, M.Y.: Rethinking pre-trained feature extractor selection in multiple instance learning for whole slide image classification (2025), https: //arxiv.org/abs/2408.01167

53. Xu, H., Usuyama, N., Bagga, J., Zhang, S., Rao, R., Naumann, T., Wong, C., Gero, Z., Gonz´alez, J., Gu, Y., Xu, Y., Wei, M., Wang, W., Ma, S., Wei, F., Yang, J., Li, C., Gao, J., Rosemon, J., Bower, T., Lee, S., Weerasinghe, R., Wright, B.J., Robicsek, A., Piening, B., Bifulco, C., Wang, S., Poon, H.: A whole-slide foundation model for digital pathology from real-world data. Nature 630, 181–188 (2024). https://doi.org/10.1038/s41586-024-07441-w

54. Yang, J., Chen, H., Zhao, Y., Yang, F., Zhang, Y., He, L., Yao, J.: Remix: A general and eficient framework for multiple instance learning based whole slide image classification (2022), https://arxiv.org/abs/2207.01805

55. Yang, Z., Wei, T., Liang, Y., Yuan, X., Gao, R., Xia, Y., Zhou, J., Zhang, Y., Yu, Z.: A foundation model for generalizable cancer diagnosis and survival prediction from histopathological images. Nature Communications 16, 2366 (2025). https: //doi.org/10.1038/s41467-025-57587-y

56. Zbontar, J., Jing, L., Misra, I., LeCun, Y., Deny, S.: Barlow twins: Self-supervised learning via redundancy reduction (2021), https://arxiv.org/abs/2103.03230

57. Zhang, J., Ma, K., Arnam, J., Gupta, R., Saltz, J., Vakalopoulou, M.: A joint spatial and magnification based attention framework for large scale histopathology classification (06 2021). https://doi.org/10.1109/CVPRW53098.2021.00418

58. Zhao, T.Z., Kiblawi, S., Yang, J., Usuyama, N., Tan, R., Codella, N.C., Naumann, T., Poon, H., Wei, M.: Learning sparse visual representations via spatial-semantic factorization (2026), https://arxiv.org/abs/2602.01905

59. Zong, Z., Li, K., Song, G., Wang, Y., Qiao, Y., Leng, B., Liu, Y.: Self-slimmed vision transformer. In: European Conference on Computer Vision. pp. 432–448. Springer (2022)

## Appendix

## A Label-free metrics

All metrics use the frozen refined-stage [CLS] features. Arrows in the table of the main paper mark the preferred direction. Alignment [49]: mean squared distance between ℓ<sub>2</sub>-normalised embeddings of two views (lower = more view-invariant). RankMe [22]: entropy-based smooth efective rank of the feature singular values. Efective rank [42]: exponential Shannon entropy of the normalised singular values (higher = flatter spectrum). α-ReQ [1]: power-law exponent of eigenspectrum decay (α ≈ 1 well-conditioned). Participation ratio [21]: $( \sum _ { i } \lambda _ { i } ) ^ { 2 } / \sum _ { i } \lambda _ { i } ^ { 2 }$ efective dimensionality dominated by the leading eigenvalues.

## B Driver-Gene Mutation Prediction on TCGA-LUAD

Task. As a further probe of representation quality, we evaluate weakly-supervised prediction of somatic driver-gene mutation status from H&E WSIs of lung adenocarcinoma (LUAD) [15]. We treat four of the driver genes highlighted by Coudray et al. [15] for LUAD histology—KRAS, EGFR, TP53, STK11 —as independent binary slide-level tasks (mutated vs. wild-type).

Dataset and labels. We use the diagnostic slides of the TCGA-LUAD cohort. Mutation calls are taken from the cBioPortal [7,20] PanCancer Atlas file; following standard practice we retain only non-silent mutations and reduce them to a patient-level binary label per gene.

Protocol. As for other tasks in the main paper, encoders are frozen and only the DSMIL [31] aggregator is trained, so the encoder is the only variable. Encoders include Lunit [28], UNI [10], Virchow [47], H0-mini [18], and CRAFT. For each gene, we train a single-logit MIL classifier for 50 epochs under patient-level 5- fold stratified CV (seed 42; no patient spans train/test). We report mean ± std over the 5 folds, following [52].

Table 5: Driver-gene mutation prediction on TCGA-LUAD. Weakly-supervised ROC-AUC (%), DSMIL [31] aggregation; encoder is the only variable. Mean ± std over 5 patient-level, label-stratified folds; Macro = mean over genes. Bold and Underlined show the first and second best methods per column.
<table><tr><td>Encoder</td><td>Arch. (dim)</td><td>KRAS</td><td>EGFR</td><td>TP53</td><td>STK11</td><td>Macro</td></tr><tr><td>Virchow</td><td>ViT-H/14 (2560)</td><td>54.3 ±5.9</td><td></td><td>54.0 ±4.3 60.6 ±5.8 49.7 ±11.0 54.65</td><td></td><td></td></tr><tr><td>H0-mini</td><td>ViT-B/14 (1536)</td><td></td><td></td><td>56.4 ±5.1 54.6 ±7.6 69.4 ±6.2 64.1 ±6.7 61.13</td><td></td><td></td></tr><tr><td>UNI</td><td>ViT-L/16 (1024)</td><td>56.3 ±4.7</td><td></td><td>55.6 ±3.7 63.7 ±5.1 55.2 ±5.4</td><td></td><td>57.71</td></tr><tr><td>CRAFT-mixed ViT-S/16 (384)</td><td></td><td></td><td></td><td>57.8 ±4.5 56.8 ±6.0 65.5 ±3.4 61.3 ±8.9</td><td></td><td>60.35</td></tr></table>

Driver-mutation status is partly reflected in H&E morphology and can be inferred under weak supervision, though signal strength varies by gene and drops out-of-domain [15]. We therefore read this as a representation probe rather than a clinical result. Despite its ViT-S backbone, CRAFT is competitive with foundation models up to 7× larger, leading on KRAS and EGFR. TCGA slides originate from many submitting sites with distinct staining and scanning signatures; models can exploit these rather than biology, inflating absolute AUCs across all encoders [24]. H0-mini’s edge on TP53/STK11 likely reflects distillation from a much larger teacher rather than same-size training; moreover, if this distillation used TCGA, H0-mini would also carry data leakage.

## C TCGA-NSCLC: Per-Aggregator Results

Table 6 expands the main-text mean AUC (Table 2) into per-aggregator scores. Model selection uses 5-fold CV on the 638 training slides, with evaluation applied on the fixed 256-slide test set. Public encoders are frozen and evaluated on our split under our MIL protocol. Barlow Twins, MoCo v3, and DINO results are quoted from Mammadov et al. [35]. Encoders marked ¶ are pretrained on TCGA, overlapping our test split, so their AUC is potentially inflated (H0-mini, Lunit, CTransPath, RetCCL); UNI and Virchow training distribution doesn’t contain TCGA.

Among the 22M encoders, CRAFT mixed-16/32 is the only method above 94 AUC for all four aggregators (94.4–97.7). Same-size baselines are far less stable under the weakest aggregator: Max-pooling drops Lunit to 89.80±5.69 and the CAMELYON16-pretrained DINO baseline to 68.58±6.50, whereas CRAFT retains 95.34±0.41. Since all rows share the frozen-encoder, fixed-aggregator protocol, this consistency reflects encoder quality rather than aggregator tuning.

Table 6: Per-aggregator WSI subtype classification on TCGA-NSCLC (LUAD vs. LUSC), AUC (%). DS/AB/Trans/Max = DSMIL, ABMIL, TransMIL, Maxpooling; Avg = mean of the four. Among 22M encoders (Low Data), best bold, second underlined. ¶ TCGA overlap may inflate AUC.
<table><tr><td colspan="2">Method</td><td>Encoder</td><td>Pretraining</td><td>DS</td><td>AB</td><td>Trans</td><td>Max</td><td>Avg</td><td>#P</td></tr><tr><td colspan="10">Frozen public Encoder trained on Massive Data</td></tr><tr><td></td><td>UNI [10]</td><td>ViT-L/16</td><td>DINOv2, Mass-100K</td><td>97.82 ±0.49</td><td>95.47 ±1.87</td><td>98.10 ±1.15</td><td>95.83 ±0.34</td><td>96.81</td><td>307M</td></tr><tr><td></td><td>Virchow [47]</td><td>ViT-H/14</td><td>DINOv2, MSK-1.5M</td><td>97.53 ±0.21</td><td>95.37 ±1.12</td><td>97.80 ±1.37</td><td>95.96 ±0.28</td><td>96.67</td><td>632M</td></tr><tr><td></td><td>CTransPath&quot; [50]</td><td>Swin-T</td><td>SRCL, TCGA+PAIP</td><td>93.80 ±0.28</td><td>91.21 ±2.20</td><td>96.88±0.99</td><td>89.34 ±0.46</td><td>92.81</td><td>28M</td></tr><tr><td></td><td>RetCCL</td><td>ResNet-50</td><td>TCGA+PAIP (CCL)</td><td>87.80±1.26</td><td>83.57±1.38</td><td>95.14±1.97</td><td>75.92±2.11</td><td>85.61</td><td>23.5M</td></tr><tr><td>Maee Dsta</td><td>H0-mini [18]</td><td>ViT-B/14</td><td>DINOv2 distill, TCGA</td><td>98.35 ±0.19</td><td>96.90 ±1.10</td><td>98.18 ±0.90</td><td>97.71 ±0.24</td><td>97.79</td><td>86M</td></tr><tr><td></td><td>Lunit [28]</td><td>ViT-S/16</td><td>DINO, TCGA</td><td>95.13 ±0.32</td><td>93.45 ±1.58</td><td>98.12 ±0.76</td><td>89.80 ±5.69</td><td>94.13</td><td>22M</td></tr><tr><td></td><td>Barlow Twins [35] [56]</td><td>ViT-S/16</td><td>Barlow, TCGA-LUNG</td><td>86.0</td><td>91.7</td><td>94.4</td><td>94.6</td><td>91.7</td><td>22M</td></tr><tr><td></td><td>MoCo v3 [35][12]</td><td>ViT-S/16</td><td>MoCo v3, TCGA-LUNG</td><td>92.3</td><td>95.2</td><td>95.7</td><td>96.9</td><td>95.0</td><td>22M</td></tr><tr><td>Dota Dta</td><td>DINO [35] [6]</td><td>ViT-S/16</td><td>DINO, TCGA-LUNG</td><td>93.0</td><td>94.1</td><td>96.1</td><td>96.8</td><td>95.0</td><td>22M</td></tr><tr><td></td><td>DINO ViT-S</td><td>ViT-S/16</td><td>DINO, CAM16</td><td>91.40 ±1.43</td><td>84.87 ±1.71</td><td>96.41 ±1.02</td><td>68.58 ±6.50</td><td>85.32</td><td>22M</td></tr><tr><td></td><td>IN ViT-S [16,44]</td><td>ViT-S/16</td><td>sup., IN-21k→1k</td><td>85.89 ±1.21</td><td>84.86 ±7.01</td><td>94.96 ±1.38</td><td>70.62 ±5.50</td><td>84.08</td><td>22M</td></tr><tr><td></td><td>Ours coarse-32</td><td>ViT-S/16</td><td>DINO, TCGA-LUNG</td><td>97.36 ±0.42</td><td>94.06 ±2.15</td><td>97.49 ±0.98</td><td>94.65 ±0.16</td><td>95.89</td><td>22M</td></tr><tr><td></td><td>Ours mixed-16/32</td><td>ViT-S/16</td><td>DINO, TCGA-LUNG</td><td>97.65±0.45</td><td>94.44 ±1.97</td><td>97.54 ±0.93</td><td>95.34 ±0.41</td><td>96.24</td><td>22M</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## D Survival Prediction

Table 7 reports C-index on TCGA-LUAD using the fixed CLAMSurvival aggregator [55] with 5-fold cross-validation. Results for all competing encoders are taken directly from [55], while CRAFT is evaluated under the identical protocol. Absolute C-indices lie between 0.51 and 0.57, reflecting the inherent dificulty of TCGA-LUAD survival prediction due to its limited cohort size, heavy censoring, and weak slide-level prognostic signal rather than the evaluation setup. Similar performance ranges have been reported in prior work under diferent survival pipelines (e.g., DS-MIL achieves 0.537 [31]).

Using a common survival head isolates diferences in the learned representations. Notably, encoders with nearly identical computational cost can exhibit substantially diferent survival performance (e.g., HIPT: 53.8 vs. CTransPath: 51.2 at ∼4.5 GMACs), indicating that representation quality, rather than computational budget, is the dominant factor. While a survival head individually tuned to each encoder could improve absolute performance, such tuning would confound representation quality with downstream optimization. We therefore adopt a fixed aggregator to enable a controlled comparison across encoders.

Table 7: Survival prediction (C-index) on TCGA-LUAD. †External pretraining. All methods use CLAMSurvival [55]; C-indices except Ours are quoted from [55]. GMACs: single 224 × 224 forward pass (ResNet-50/ViT-B/Swin-T from literature; HIPT dominated by its cell-level ViT-S/16, ≈4.6).
<table><tr><td>Model</td><td>Encoder C-index</td><td>GMACs</td></tr><tr><td>DINO-HistoPretrain† [55] ViT-B/16 GigaPath† [53]</td><td> $5 6 . 9 \pm 5 . 5$  ViT-G/14  $5 6 . 2 \pm 0 . 5 2$ </td><td>17.6 228.1</td></tr><tr><td>UNI† [10]</td><td>ViT-L/16  ${ \bf 5 7 . 1 \pm 9 . 1 }$ </td><td>61.55</td></tr><tr><td>H0-mini† [18]</td><td>ViT-B/14  $5 6 . 5 \pm 2 . 4 3$ </td><td>23.5</td></tr><tr><td> $\mathrm { H I P T ^ { \dagger } \ [ 9 ] }$  CTransPath† [50]</td><td>Hier. ViT  $5 3 . 8 \pm 4 . 0$ </td><td>4.6</td></tr><tr><td> $\mathrm { R e s N e t - I m a g e N e t } ^ { \dagger }$  [55]</td><td>Swin-T  $5 1 . 2 \pm 5 . 0$  ResNet-50  $5 5 . 2 \pm 1 0 . 0$ </td><td>4.5 4.1</td></tr><tr><td>Ours coarse-32</td><td></td><td></td></tr><tr><td>Ours mixed-16/32</td><td>ViT-S/16  $5 5 . 4 9 \pm 2 . 3 0$  ViT-S/16  $5 6 . 9 \pm 2 . 0 6$ </td><td>1.16 4.01</td></tr></table>

## E λ-Sensitivity of the Symmetric-KL Loss on NCT-CRC-HE-100K

To isolate the efect of the symmetric-KL weight λ $\mathrm { ( E q . 5 ) }$ , we sweep it on NCT-CRC-HE-100K[29], a nine-class colorectal tile-classification benchmark of 100k 224 × 224 H&E tiles (20×). We use two disjoint evaluation sets. In-domain, we split the NCT tiles into an 80k training partition and a 20k held-out validation partition. Externally, we use the patient-disjoint CRC-VAL-HE-7K test set (7,180 tiles, 50 patients), which is touched only for final evaluation.

We pretrain CRAFT $( \mathrm { V i T - S } / 1 6 , \alpha = 0 . 5 )$ for 100 epochs on the 80k training tiles (self-supervised; labels unused), varying only λ and holding the rest of the recipe fixed. Following the linear-probing protocol of the main paper, we then freeze the teacher and fit a logistic-regression head on the mixed-scale (refined) [CLS] features of the 80k labeled training tiles, selecting the probe’s regularization strength on the 20k validation partition (reported as $^ { 6 6 } \mathrm { V a l }$ . Bal. Acc.”). We report the resulting balanced accuracy on the external CRC-VAL-HE-7K set. As label-free measures of representational capacity, we also report the efective rank [42] and participation ratio [21] of the [CLS] features — soft counts of how many dimensions they span (higher = more spread, less collapsed).

We fix $\lambda = 0 . 5$ , the default in our main experiments; among the reported settings it also attains the best accuracy on the held-out NCT validation split. At this weight, CRAFT reaches 90.97 balanced accuracy on the external test set using 124 tokens and 4.0 GMACs. Its [CLS] features span a high-dimensional subspace, with efective rank 13.3 and participation ratio 6.7.

To probe the efect of the weight, we report larger values. Increasing λ drives the coarse- and fine-stage head distributions apart—the coarse–fine KL grows from $2 \times 1 0 ^ { - 4 }$ at $\lambda = 0 . 5$ to $5 \times 1 0 ^ { - 2 }$ at λ = 5—but this divergence is detrimental rather than beneficial: balanced accuracy declines mildly at $\lambda = 2 ~ ( 8 9 . 6 4 )$ and collapses at λ = 5 (77.87), with the efective rank and participation ratio falling in step (to 6.1 and 3.6). An excessively large λ over-regularizes the two scales and degrades the representation; the small default $\lambda = 0 . 5$ stays well clear of this regime.

Table 8: λ-sensitivity on NCT-CRC-HE-100K (mixed-scale [CLS]; single run per row). Row shading marks the regime: safe → caution → over-regularized collapse . cls sim/KL are model-level (final epoch). <sup>∗</sup>Operating point selected on validation Bal. Acc.
<table><tr><td rowspan="2"> $\lambda \ ( \mathbf { K L _ { s y m } } )$ </td><td colspan="2">Coarse-fine</td><td rowspan="2">Val.</td><td colspan="2">Test (mixed-scale [CLS])</td><td colspan="2">Label-free</td></tr><tr><td>cls_sim KL</td><td> $( \times 1 0 ^ { - 3 } )$ </td><td>Bal. Acc. Bal. Acc.</td><td>Macro-F1</td><td></td><td>Eff. rank ↑ Part. ratio ↑</td></tr><tr><td>0.5 (default) 0.993</td><td></td><td>0.16</td><td>98.81</td><td>90.97</td><td>90.46</td><td>13.3</td><td>6.7</td></tr><tr><td>2.0</td><td>0.975</td><td>0.35</td><td>98.09</td><td>89.64</td><td>89.55</td><td>10.4</td><td>5.5</td></tr><tr><td>5.0</td><td>0.824</td><td>54.3</td><td>89.47</td><td>77.87</td><td>77.32</td><td>6.1</td><td>3.6</td></tr></table>

The coarse stage pinpoints where λ acts first. Mixed-scale accuracy and the coarse–fine KL hold up well through $\lambda { = } 2$ and collapse only at λ=5, whereas the coarse-only representation degrades earlier and monotonically (Fig. 5a). Coarse and mixed inference nearly coincide at λ=0.5; as λ grows, the coarse−mixed gap widens monotonically $( - 0 . 9  - 3 . 0  - 3 . 2 \mathrm { p p } )$ . Over-regularization thus hits the coarse stream first: coarse tokens rely on cross-scale alignment to stay discriminative and lose it before the mixed representation is afected.

![](images/1320d3af52cbd547bc013a25511423e1d97dd102a3049f97cb81bef8a378c00f.jpg)  
(a) Coarse-only vs. mixed-scale probe accuracy; the coarse−mixed gap widens monotonically with $\lambda ( - 0 . 9  - 3 . 2 \mathrm { p p } )$ .

![](images/6a9de99172d57ebfcb5316bd1a35f016b80b88dfba53ff48f7b51753b10ee2b1.jpg)  
(b) Training dynamics: alignment holds for $\lambda \leq 1$ , destabilizes at $\lambda = 5 ;$ ; loss converges higher and stalls (≈8).  
Fig. 5: λ-sensitivity of CRAFT (ViT-S/16; single run per λ) on NCT-CRC-HE-100K / CRC-VAL-HE-7K.

Training dynamics. The training-loss trajectories show how optimization fails as λ grows (Fig. 5b). At λ=0.5 the loss converges smoothly (≈3.6); at λ=2 optimization slows and plateaus higher (≈5.2); at λ=5 training destabilizes early (∼epoch 30) and stalls near 8.0.

These trajectories mirror the alignment in Table 8: the symmetric term opposes coarse–fine [CLS] alignment more strongly as λ grows, so stronger repulsion yields a higher, unconverged loss. This tug-of-war between cross-scale alignment and symmetric repulsion explains the collapse at large λ; the default λ=0.5 leaves optimization essentially undisturbed.

## F Component Ablation

Table 9 isolates each component of CRAFTduring the adaptive selection, on CAMELYON16, on the frozen-feature tile-classification (Linear Probing) and slide-level (MIL) protocols. TS denotes the token scale(s) used at inference (16, 32, or mixed 16/32), MF multi-scale fusion, and $\mathcal { L } _ { \mathrm { K L - a s y m } } / \mathcal { L } _ { \mathrm { K L - s y m } }$ the asymmetric and symmetric cross-scale alignment losses. Best and second-best per column are bold and underlined; MIL Max is reported over five seeds (± std).

We isolate multi-scale fusion (MF) and cross-scale alignment $\left( \mathcal { L } _ { \mathrm { K L - a s y m } } \right)$ . For fine tokens (16), removing MF (O1→C1) cuts tumor recall by 5.1 pp (80.7 → 75.6), confirming the value of reusing coarse context. For mixed tokens (16/32), the efect shifts to slide level: removing MF (O2→C2) drops MIL Avg AUC by 2.9 pp $( 9 5 . 6 5  9 2 . 7 6 )$ , indicating coarse context matters most when fine tokens are only partially available. Finally, removing $\mathcal { L } _ { \mathrm { K L - a s y m } }$ (O2→C3) causes the largest MIL Avg-Recall drop (4.4 pp, 94.19 → 89.79), showing cross-scale alignment is necessary for efective slide-level classification.

Table 9: Component ablation of CRAFT. Each block removes one component from a CRAFT operating point (full → ablated); ∆ is the change on the key metric of that block. LP Recall is tumor recall.
<table><tr><td colspan="6"></td><td colspan="2">Linear Probing</td><td colspan="2">MIL</td><td></td></tr><tr><td>ID Model</td><td>TS</td><td>MF</td><td> $\mathcal { L } _ { \mathrm { K L - a s y m } }$ </td><td> $\mathcal { L } _ { \mathrm { K L - s y m } }$ </td><td>ACC Prec Recall</td><td></td><td> $\operatorname { A v g }$  AUC</td><td> $\operatorname { A v g }$  Recall</td><td>Max</td><td>GMACs</td></tr><tr><td>O1 CRAFT</td><td>16</td><td>√</td><td>√</td><td>×</td><td>98.16 92.3</td><td>80.7</td><td>96.74</td><td>96.74</td><td>97.67±0.48</td><td>5.76</td></tr><tr><td>C1 CRAFT</td><td>16</td><td>×</td><td>√</td><td>×</td><td>97.97 94.2</td><td>75.6</td><td>95.59</td><td>94.80</td><td>95.66±1.18</td><td>5.76</td></tr><tr><td>MF effect (mixed, 16/32)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>O2 CRAFT 16/32√</td><td></td><td></td><td>√</td><td>×</td><td>97.80 93.8</td><td>73.2</td><td>95.65</td><td>94.19</td><td>95.66±2.76</td><td>4.01</td></tr><tr><td>C2 CRAFT 16/32 ×</td><td></td><td></td><td>√</td><td>X</td><td>97.73 92.9</td><td>73.0</td><td>92.76</td><td>91.02</td><td>94.37±0.63</td><td>4.01</td></tr><tr><td> $\mathcal { L } _ { \mathrm { K L - a s y m } }$  effect (mixed, 16/32)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>O2 CRAFT 16/32√</td><td></td><td></td><td>√</td><td>×</td><td>97.80 93.8</td><td>73.2</td><td>95.65</td><td>94.19</td><td>95.66±2.76</td><td>4.01</td></tr><tr><td>C3 CRAFT 16/32√</td><td></td><td></td><td>×</td><td>×</td><td>97.60 92.9</td><td>71.0</td><td>93.77</td><td>89.79</td><td>94.23±0.01</td><td>4.01</td></tr></table>

## G Attention Head Outputs

We further examine the self-attention maps of the six heads in the final teacher layer, complementing the main-text analysis.

Figure 6 shows three tissue regions, each under the selective (top) and coarse (bottom) grid. The heads divide labor: diferent heads light up on diferent parts of the patch rather than all tracking the same region. More importantly, the coarse grid (bottom rows) reproduces the attention structure of the selective grid (top rows) closely—the same regions remain salient at lower resolution. This is the visual counterpart of our quantitative finding that coarse-only inference stays discriminative.

## H Mixed-Scale Grids

Figure 7 shows additional mixed-scale tokenization examples. Refinement consistently concentrates on cellular regions, while stromal and background tissue stay coarse, showing that learned selection is spatially selective rather than uniform.

## I Student–Teacher Comparison

Figure 8 compares the [CLS] attention and token selection of the teacher and student branches on the same input. The two branches converge to a near-identical selection: the global [CLS] attention (left), the selected-token grid overlaid on the input (center), and the per-layer [CLS] attention (right) all coincide across the two rows, concentrating on the same salient regions.

This agreement indicates that the two-branch design does not push the branches toward divergent selections, and that the added adaptivity does not destabilize optimization (final training loss 1.93 vs. 1.26 for the baseline). Unlike SimPrune [32], where single-branch selection prunes the two branches inconsistently, our branches converge to the same selection without an explicit cross-branch mechanism.

![](images/f37d545e31b3981eaa6945b6c1b799185fa64b18a28073c32c8633e0844ff937.jpg)  
Fig. 6: Self-attention maps from the six final-layer teacher heads for three tissue regions (input on the left). Top row of each pair: selective grid; bottom row: coarse grid. Brighter is higher attention.

![](images/c8cef95e5dc9fdf1b2fbbe9d27aab3120e5f8576f765cee2afce2bac9f5b2a07.jpg)

Fig. 7: Visualization of adaptive mixed-scale tokenization applied to histopathology patches. High-attention regions, typically rich in cellular structures, are refined into fine-grained tokens, while background and less informative stromal areas remain at a coarse resolution. This hierarchical tokenization strategy preserves diagnostically relevant morphological features while optimizing computational eficiency.  
![](images/8f1a25135de2c043b909e4c214f834b49fa58c1e9ca75e76b9842ccca247d0a1.jpg)  
Fig. 8: Teacher–student selection agreement. Top: teacher branch; bottom: student branch. Left: global CLS attention. Center: selected tokens overlaid on the input. Right: CLS attention. Both branches attend to and select the same regions, showing the two-branch design preserves a consistent token selection.

## J Limitations

CRAFT has some limitations. First, although the learned refinement policy consistently improves downstream performance, the selected regions are not explicitly optimized for diagnostic interpretability and often follow general morphological structure rather than clinically salient tissue (Fig. 9). Second, CRAFT uses a fixed refinement ratio (α = 0.5), whereas adapting the refinement budget to image complexity could further improve the accuracy–eficiency trade-of. Finally, while we evaluate CRAFT across multiple pathology tasks, broader validation on additional cohorts and imaging modalities is needed to establish its generality.

![](images/6d3f192eb54d0afe42b7ff1d6421a39e0d132b410632450bd65fd35c3d3955a1.jpg)  
Fig. 9: Mixed-scale tokenization vs. tumor annotations on CAMELYON16. Each pair (a–d): CRAFT tokenization grid (left) and pathologist annotation of tumor outlined in blue (right). Finer tokens follow dense/high-contrast morphology, not the annotated tumor extent.