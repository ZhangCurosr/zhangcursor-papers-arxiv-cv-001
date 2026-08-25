# How Merge-Tolerant Are Vision Transformers for Wheat Phenotyping?

Simon Ravé<sup>1⋆</sup>, Pejman Rasti<sup>1,2</sup>, and David Rousseau<sup>1,2</sup> {simon.rave,pejman.rasti,david.rousseau}@univ-angers.fr

<sup>1</sup> LARIS, University of Angers, Angers, France <sup>2</sup> UMR INRAE-IRHS, Angers, France

Abstract. Vision-based wheat phenotyping requires repeated measurements under deployment constraints, from growth-stage recognition to wheat-head counting and organ segmentation. Plain Vision Transformers (ViTs) provide a common architecture for these tasks, but quadratic attention limits high-throughput and edge inference. Training-free token merging is attractive because it can be inserted into trained models without retraining. We provide a systematic benchmark of ToMe and Mutual Pair Merging across growth-stage classification, wheat-head detection, and wheat-organ segmentation, measuring task quality, throughput, token count, and peak GPU memory, with additional Raspberry Pi 5 measurements. The benchmark reveals a clear hierarchy: classification is highly merge-tolerant, while detection and segmentation are constrained by repeated instances, thin organs, dense boundaries, reconstruction, and runtime overhead. Optimized attention backends can erase apparent speedups, so deployment value must be profiled on the target runtime rather than inferred from token count.

Keywords: Plant phenotyping · Vision transformers · Token merging · Eficient inference

## 1 Introduction

Image-based plant phenotyping has moved from isolated image analysis toward repeated measurement in fields, greenhouses, and breeding nurseries. Cameras on fixed stations, UAVs, robots, and low-cost edge devices produce visual streams across sites, growth stages, and acquisition platforms under varying device budgets. A breeder may need the growth stage of a wheat canopy, the number and location of wheat heads, or a pixel-level separation of leaves, stems, and spikes. A model that is accurate but too slow, too memory-hungry, or too dependent on workstation hardware may be dificult to use in field robots or low-cost phenotyping systems [12]. Recent work has therefore explored general visual representations and compact models for plant recognition [3, 17, 22].

Wheat images contain strong visual redundancy (Fig. 5). Repeated biological structures create both an opportunity and a risk: canopy evidence may be compressed, whereas small instances and boundaries may be merged away. Plain ViTs provide the same sequence interface for classification, segmentation, and detection [8, 9, 25], but self-attention scales poorly with sequence length while crop images often require high resolution. The question is whether their token sequences can be compressed without losing phenotype-relevant evidence.

![](images/7b3b01557016dc808c29305e05fbf52040742b9aad039ee48aa9fca82aea2cb1.jpg)  
Fig. 1: Overview of our merging protocol. We benchmark classification, segmentation, and detection using plain Vision Transformers [8]. After training and freezing each model, we apply ToMe [2] or MPM [20] only at inference. We evaluate multiple backbone sizes and report task quality, token count, latency, and throughput on an NVIDIA RTX PRO 6000 system and a Raspberry Pi 5, with peak memory reported for GPU runs.

Training-free token merging addresses this deployment pressure by aggregating similar tokens inside an already trained transformer. ToMe showed that this intervention can improve throughput without retraining [2]. MPM emphasizes end-to-end latency and dense-feature reconstruction [20]. Yet generic computervision conclusions do not directly transfer to plant phenotyping: a merger that is harmless for classification may be harmful when the signal is one wheat head among many, a narrow stem, or a leaf boundary. The answer is also hardwaredependent because merge construction and reconstruction can dominate on optimized GPUs even when shorter sequences help embedded processors [5].

We therefore ask when an already trained ViT can be accelerated by merging tokens and when merging destroys the biological signal of interest. We study WGSP growth-stage classification [23], GWHD 2021 wheat-head detection [7], and GWFSS wheat-organ segmentation [26], using a ViT classifier, Segmenter, and YOLOS across multiple backbone sizes. We report primary and secondary task metrics together with measured GPU and Raspberry Pi 5 behavior. Classification tolerates substantial token reduction, detection provides limited practical gains in the tiled pipeline, and segmentation degrades under aggressive merging and dense reconstruction. Our contributions are:

1. We provide a benchmark of training-free token merging for ViT-based wheat phenotyping across classification, detection, and segmentation under a unified inference-time protocol.

2. We compare ToMe and MPM on plain ViT-based backbones and report task quality together with measured throughput, latency, final token count, and peak GPU memory, so that accuracy tolerance is separated from deployment usefulness.

3. We analyze why merge tolerance difers across plant tasks, including merge locality, dense reconstruction cost, adaptive padding, attention backends, and Raspberry Pi 5 edge behavior, and derive task- and runtime-specific deployment guidance.

4. The code is available at https://forge.inrae.fr/imhorphen/wheattoken-merging.

## 2 Related Work

Vision Transformers for Plant Vision. Vision Transformers represent an image as patch tokens processed by self-attention [8]. The abstraction extends beyond classification: Segmenter applies a plain ViT encoder with a mask-transformer decoder [25], while YOLOS formulates detection with a minimal ViT-style sequence interface [9]. This common interface matters because token merging acts on the sequence rather than a task-specific convolutional pyramid. Plant-vision work has adapted foundation models and compact networks to recognition and measurement [3, 17, 22]. We instead ask how much of an already trained plant ViT sequence can be merged without losing task utility.

Token Merging. Token reduction can prune tokens [10, 13, 19, 21, 27] or merge them. We focus on merging because it preserves information by aggregation rather than deletion. Token Pooling formulates aggregation as clustering [16]. ToMe pairs similar tokens inside a pretrained ViT without retraining [2], whereas MPM uses mutual nearest-neighbor pairs and records mappings for dense reconstruction [20]. Prior comparisons show that accuracy–speed behavior cannot be inferred from token count alone [11]. The trade-of is also task-dependent [1, 15, 18]: classification often tolerates coarse reduction, while dense prediction requires recoverable spatial evidence. We therefore evaluate both regimes using task quality and measured runtime rather than token count as a proxy.

Wheat Phenotyping Benchmarks and Deployment. Wheat phenotyping tasks difer in the spatial evidence they require. WGSP/GSP-AI uses canopy imagery for growth-stage recognition [23], GWHD evaluates head detection and counting across diverse conditions [6, 7], and GWFSS requires semantic separation of leaves, stems, and spikes [26]. These datasets expose global canopy structure, many small repeated objects, and dense boundary-sensitive organs. Embedded benchmarks and Raspberry Pi studies further motivate operation on constrained agricultural hardware [12,22]. To our knowledge, prior plant-vision work has not evaluated training-free token merging across all three regimes, multiple plain-ViT sizes, and both edge and workstation hardware.

## 3 Method

## 3.1 Benchmark Design

We study token merging as a deployment intervention, not as a new plant architecture. For each task, we first train a plain ViT-based model without token merging. We then freeze all weights and insert a training-free token merger only at inference time. Thus, for a fixed checkpoint, any change in task metrics or deployment measurements comes from the modified token sequence and from the overhead of the merge operator itself.

Let $\boldsymbol { x } \in \mathbb { R } ^ { H \times W \times 3 }$ be an RGB image. A ViT with patch size P maps x to a sequence of image tokens $Z _ { 0 } ~ \in ~ \mathbb { R } ^ { N _ { 0 } \times d }$ , where $N _ { 0 } ~ = ~ H W / P ^ { 2 }$ for a square image divisible by $P$ [8]. Depending on the task, the sequence may also contain protected tokens $U _ { l }$ , such as a classification token or detection tokens. At a selected transformer layer $\ell ,$ a merger replaces only the image-token sequence,

$$
X _ { \ell } = [ U _ { \ell } , Z _ { \ell } ] \quad \longrightarrow \quad { \widetilde { X } } _ { \ell } = [ U _ { \ell } , M _ { \ell } ( Z _ { \ell } ) ] ,\tag{1}
$$

where $M _ { \ell }$ returns fewer image tokens. Protected tokens are never merged. The task head is left unchanged.

We define merge tolerance by two quantities. The first is the task-performance drop relative to the unmerged checkpoint. The second is measured speedup,

$$
s = { \frac { T _ { \mathrm { b a s e } } } { T _ { \mathrm { m e r g e } } } } ,\tag{2}
$$

where $T$ is wall-clock inference time under the same batch size, precision, device, and attention backend. A merging setting is deployment-useful only if it mostly preserves the task metric while giving $s > 1$ in the measured path.

## 3.2 Tasks and Datasets

The benchmark covers one global, one instance-level, and one dense prediction task. This triplet mirrors a common phenotyping progression from coarse cropstate assessment to object-level counting and dense organ measurement. It also makes merge tolerance interpretable because the tasks difer not only by metric, but also by the spatial scale and identity of the biological evidence they require. The merging scheme for each task is summarized in Fig. 1.

Table 1: Tasks, dataset splits, and metrics. WGSP and GWHD list training / validation / test, GWFSS lists training / evaluation.
<table><tr><td>Task</td><td>Dataset</td><td>Split</td><td>Metrics</td></tr><tr><td></td><td>Classification WGSP/GSP-AI [23] 36,440</td><td>5,206</td><td>10,412 Top-1, macro-F1</td></tr><tr><td>Detection</td><td>GWHD 2021 [7]</td><td>3,605 / 1,448 / 1,334</td><td>mAP,  $\mathrm { A P _ { 5 0 } , }$  count  ${ \mathrm { M A E } } ,$  count  $R ^ { 2 }$ </td></tr><tr><td>Segmentation GWFSS [26]</td><td></td><td>876  / 220</td><td>mIoU, per-class IoU, boundary F1</td></tr></table>

Representative samples appear in Fig. 2. WGSP assigns one Zadoks growthstage label to a canopy image, so evidence may be pooled over broad plant regions. GWHD instead requires the model to retain the location and identity of individual wheat heads despite their repeated appearance and variable density. GWFSS requires pixel-level separation of Background, Head/Spike, Stem, and Leaf, including narrow organs and interfaces between adjacent classes. Its provided 220-image evaluation split is also used for checkpoint selection and final reporting, so it is not an independently held-out test estimate. The primary metrics are Top-1, mAP, and mIoU. Macro-F1, $\mathrm { A P _ { 5 0 } }$ , count MAE, count $R ^ { 2 }$ per-class IoU, and boundary F1 test whether an apparently tolerant operating point conceals class-, counting-, or boundary-specific damage.

## 3.3 Backbones and Training

For WGSP, pretrained timm ViT-T/16, ViT-S/16, and ViT-B/16 classifiers [8] are fine-tuned for 50 epochs with AdamW [14], batch size 256, weight decay 0.05, and seed 1337. The constant learning rates are $4 \times 1 0 ^ { - 4 }$ for Tiny and Small and $2 \times 1 0 ^ { - 4 }$ for Base. Cross-entropy training uses random resized crops ([0.7, 1.0]), horizontal flips, and RandAugment (two operations, magnitude 9) [4]. The best validation macro-F1 checkpoint is evaluated at $2 2 4 ^ { 2 }$

For GWFSS [26], Segmenter combines the corresponding pretrained timm ViT-T/16, ViT-S/16, or ViT-B/16 encoder with a two-layer mask-transformer decoder [25]. All variants use AdamW for 80 epochs, weight decay 0.05, cosine decay after a five-epoch warm-up, weighted cross-entropy $( 0 . 5 7 , 1 . 7 8 , 1 . 1 7 , 0 . 4 9 )$ bfloat16, and seed 1337. Tiny uses a learning rate of $2 \times 1 0 ^ { - 4 }$ and a batch size of 128. Small and Base use $1 0 ^ { - 4 }$ with batch sizes of 64 and 32. Augmentation comprises random resized crops to $5 1 2 ^ { 2 }$ ([0.45, 1.0]), horizontal and vertical flips, rotations up to $1 5 ^ { \circ }$ , color jitter, grayscale conversion, and Gaussian blur. The best evaluation-split mIoU checkpoint is reported at $5 1 2 ^ { 2 }$

For GWHD, we use pretrained YOLOS models [9]. Each model is fine-tuned first on full images for 30 epochs at $6 4 0 ^ { 2 }$ (AdamW, learning rate $5 \times 1 0 ^ { - 5 }$ , weight decay $1 0 ^ { - 4 }$ , batch 128, two warm-up epochs, seed 1337), then on tiles with 25% overlap and a size of 512 pixels for six epochs at $6 4 0 ^ { 2 } ~ ( 1 0 ^ { - 5 }$ , batch 64, one warm-up epoch) and four at $7 6 8 ^ { 2 } \ : ( 5 { \times } 1 0 ^ { - 6 }$ , batch 40, half-epoch warm-up), with cosine decay, bfloat16, gradient clipping at 1.0, and seed 20260615. Random resized crops are used only in the initial full-image stage. Flips, color jitter, grayscale conversion, and blur are used throughout. Checkpoints are selected by validation mAP. Validation $\mathrm { A P _ { 5 0 } }$ , then ${ \mathrm { m A P } } ,$ selects the tiled protocol, while count MAE, then RMSE, selects the count threshold. The checkpoint fine-tuned at $7 6 8 ^ { 2 }$ is evaluated at $6 4 0 ^ { 2 }$ on nine tiles per image, with NMS IoU 0.3 and count threshold 0.9.

The resolutions reflect the pipelines’ spatial demands: $2 2 4 ^ { 2 }$ is native to the classifiers, $5 1 2 ^ { 2 }$ retains a dense organ grid, and the selected detector evaluates tiles at $6 4 0 ^ { 2 }$ . With patch size 16 these yield 196, 1024, and 1600 image tokens. YOLOS adds 100 detection tokens and nine tiled forward passes. Resolution and tiling therefore confound cross-task throughput comparisons.

## 3.4 Training-Free Token Merging

We compare ToMe [2] and MPM [20] on fixed checkpoints without learned parameters. They represent two practical controls: ToMe exposes a fixed per-layer reduction, whereas MPM forms an input-dependent number of mutual pairs at selected layers. This contrast tests a fixed token-removal budget against adaptive aggregation under the same inference-only protocol.

ToMe. ToMe removes a fixed number r of image tokens at each selected transformer block. Similarity is computed from normalized attention-key features before tokens are paired and merged. Unless specified otherwise, the schedule is applied in every block. We sweep r with and without proportional attention (Tab. 2). Stored unmerge maps can be applied in reverse order when a downstream decoder requires the original grid.

MPM. At each insertion layer, MPM computes cosine similarity, forms mutual nearest-neighbor pairs, and replaces each accepted pair by its average. Unpaired tokens are retained. MPM has no continuous keep ratio: the discrete schedule L controls both compression and merge-map overhead. Because the retained count is input-dependent, we report efective final tokens and the padding required for batched execution.

## 3.5 Reconstruction and Protected Tokens

Segmenter’s decoder expects a dense image-token grid. We reconstruct it after the final ViT block and before the unchanged mask-transformer decoder, using MPM’s recorded original-to-merged mapping or ToMe’s reverse unmerge maps. Reconstruction preserves tensor compatibility but duplicates merged features at their source positions. It does not recover removed spatial detail.

For YOLOS, only image patch tokens are merged. The 100 detection tokens remain protected and are concatenated with the shortened patch sequence at each modified block. The unchanged head reads them after the final block, testing whether patch-token compression accelerates their interaction without removing evidence required for localization and counting.

## 3.6 Evaluation and Profiling

Each checkpoint is evaluated unmerged and with ToMe or MPM using identical trained weights. Task and timing metrics for a comparison therefore refer to the same frozen model.

Latency is measured in native $\mathrm { P y }$ Torch evaluation mode without gradients and includes merge construction, aggregation, reconstruction when required, and the unchanged task head. The scope is the classification forward benchmark, segmentation metric pass, or full tiled detection evaluation. It is fixed within each task but difers across tasks. Detection count $R ^ { 2 } = 1 - \textstyle \sum _ { i } ( \hat { c } _ { i } - c _ { i } ) ^ { 2 } / \textstyle \sum _ { i } ( c _ { i } - \bar { c } ) ^ { 2 }$ is computed over per-image head counts at the count-MAE score threshold.

![](images/b064ffd5100e48191c46cc39e4c1d5cdbaabd5bb93895afbf6e027d2461da362.jpg)  
Fig. 2: The three wheat datasets: WGSP/GSP-AI growth-stage classification (left), GWHD 2021 head detection (center), and GWFSS organ segmentation (right). They require global canopy evidence, localization of repeated instances, and dense separation of boundary-sensitive organs, respectively.

Peak allocated CUDA memory is reported in decimal GB after resetting Py-Torch’s counter. It describes framework allocations in the measured path, not total workstation memory, and is comparable only within a task.

We report workstation measurements on an NVIDIA RTX PRO 6000 Blackwell GPU and batch-one edge measurements on a Raspberry Pi 5. The edge experiment measures latency and throughput. We separately study batch size because adaptive merging may require padding.

Finally, we analyze task-specific Pareto curves. A configuration is useful only when it improves measured wall-clock performance while preserving the task metric to an application-appropriate tolerance. This distinction is central for dense prediction, where merge construction, padding, tiling, and grid reconstruction can dominate the nominal saving from shorter sequences.

## 4 Results

Tab. 3 provides a post-hoc comparison of the closest available ToMe–MPM finaltoken pairs in the evaluated sweep. At approximately 101 tokens, the two classification methods difer by only 0.02 Top-1 points and MPM is faster. At approximately 450 tokens, the two segmentation methods difer by 0.06 mIoU points, again with higher MPM throughput. Detection behaves diferently: ToMe and

Table 2: Selected token-merging settings. ToMe uses r and optional proportional attention. MPM uses insertion layers L.
<table><tr><td colspan="2">ToMe</td><td>MPM</td></tr><tr><td>r</td><td>Prop. attn.</td><td>Layers L</td></tr><tr><td>8</td><td>Yes</td><td>{2}</td></tr><tr><td>8</td><td>No</td><td>{6}</td></tr><tr><td>16</td><td>Yes</td><td>{8}</td></tr><tr><td>16</td><td>No</td><td>{10}</td></tr><tr><td>24</td><td>Yes</td><td>{0, 3}</td></tr><tr><td>24</td><td>No</td><td>{1, 3}</td></tr><tr><td>32</td><td>Yes</td><td>{2,5}</td></tr><tr><td>32</td><td>No</td><td>{2,7}</td></tr><tr><td>48</td><td>Yes</td><td>{6,9}</td></tr><tr><td>48</td><td>No</td><td>{3, 6, 9}</td></tr><tr><td>64</td><td>Yes</td><td>{0, 2, 4, 6, 8}</td></tr><tr><td>64</td><td>No</td><td>{0, 2, 4, 6, 8, 10}</td></tr></table>

Table 3: Matched final-token comparison on Base backbones at batch size 32 with common within-task hardware, backend, and forward-timing scope. Tokens denotes the mean final total. ∆ is relative to Full. Detection FPS aggregates nine tiled forward passes.
<table><tr><td>Task</td><td>Method</td><td>Tokens</td><td>Metric (%)</td><td>∆pp</td><td>FPS</td></tr><tr><td>Classification</td><td>ToMe r = 8</td><td>101.0</td><td>99.19</td><td>+0.05</td><td>2975.7</td></tr><tr><td>Classification</td><td>MPM L = {3, 6, 9}</td><td>100.6</td><td>99.17</td><td>+0.03</td><td>3189.9</td></tr><tr><td>Segmentation</td><td>ToMe r = 48</td><td>449.0</td><td>62.46</td><td>-1.17</td><td>266.2</td></tr><tr><td>Segmentation</td><td>MPM L = {3, 6, 9}</td><td>457.7</td><td>62.52</td><td>-1.11</td><td>284.0</td></tr><tr><td>Detection</td><td>ToMe r = 32</td><td>1317.0</td><td>26.60</td><td>+1.10</td><td>5.471</td></tr><tr><td>Detection</td><td>MPM L = {2}</td><td>1305.2</td><td>22.18</td><td>-3.32</td><td>5.596</td></tr></table>

MPM retain approximately 1,310 tokens but difer by 4.42 mAP points. Equal token count therefore does not imply equal information retention, particularly when the merger and insertion schedule modify representation dynamics diferently. Positive deltas remain descriptive fixed-checkpoint estimates.

We report the main throughput–accuracy trade-ofs in Fig. 3 for the Base backbones on GPU with batch size B = 32. The evaluated ToMe and MPM configurations are listed in Tab. 2. Across the three tasks, the Pareto curves show that merge tolerance follows the spatial scale of the phenotyping signal. Growth-stage classification is the clearest positive case: the model can discard many patch tokens while preserving canopy-level information. Segmentation is intermediate: moderate merging can preserve organ masks, but the mIoU loss grows as token reduction becomes aggressive. Detection is the most constrained in practice: mild merging can preserve or numerically increase mAP, but stronger compression harms localization and the tiled inference pipeline limits wall-clock gains. For GWHD, the original images are high-resolution and wheat heads are small, so each image is divided into nine tiles before inference. This makes detection substantially slower than the classification and segmentation settings and makes throughput gains harder to realize end to end.

Table 4: Selected Base-backbone points from Fig. 3. FPS is measured in images/s. Speedup is the ratio to the within-task Full baseline. Metric denotes Top-1, mIoU, or mAP. ∆ denotes its percentage-point change. Tokens denotes the mean final count per image.
<table><tr><td>Task</td><td>Model</td><td>Method</td><td>FPS</td><td>Speedup</td><td>Metric (%)</td><td>∆pp</td><td>Tokens</td></tr><tr><td>Classification</td><td>ViT-B/16</td><td>Full</td><td>2743.46</td><td>1.00</td><td>99.15</td><td>0.00</td><td>197.0</td></tr><tr><td>Classification</td><td>ViT-B/16</td><td>ToMe r = 16, no prop.</td><td>4362.48</td><td>1.59</td><td>99.01</td><td>-0.13</td><td>11.0</td></tr><tr><td>Classification</td><td>ViT-B/16</td><td>MPM L = {0, 2, 4, 6, 8}</td><td>4022.86</td><td>1.47</td><td>98.89</td><td>-0.26</td><td>79.0</td></tr><tr><td>Segmentation</td><td>Segmenter-B/16</td><td>Full</td><td>177.80</td><td>1.00</td><td>63.63</td><td>0.00</td><td>1025.0</td></tr><tr><td>Segmentation</td><td>Segmenter-B/16</td><td>ToMe r = 48, no prop.</td><td>236.16</td><td>1.33</td><td>62.80</td><td>-0.82</td><td>449.0</td></tr><tr><td>Segmentation</td><td>Segmenter-B/16</td><td>MPM L = {6, 9}</td><td>205.96</td><td>1.16</td><td>63.26</td><td>-0.36</td><td>591.9</td></tr><tr><td>Detection</td><td>YOLOS-B</td><td>Full</td><td>5.00</td><td>1.00</td><td>25.50</td><td>0.00</td><td>1701.0</td></tr><tr><td>Detection</td><td>YOLOS-B</td><td>ToMe r = 48</td><td>5.54</td><td>1.11</td><td>26.77</td><td>+1.27</td><td>1125.0</td></tr><tr><td>Detection</td><td>YOLOS-B</td><td>MPM L = {10}</td><td>5.09</td><td>1.02</td><td>26.71</td><td>+1.21</td><td>1268.1</td></tr></table>

Representative operating points appear in Tab. 4. They preserve the primary metric while improving or approximately preserving throughput, rather than claiming exhaustive optima. For classification, ToMe and MPM reach 1.59× and 1.47× speedups for Top-1 losses of 0.13 and 0.26 points. They end with 11 and 79 tokens, reinforcing that token count is not itself semantic information. For segmentation, 1.33× and 1.16× speedups cost 0.82 and 0.36 mIoU points. Detection mAP increases numerically by 1.27 and 1.21 points, but speedups are only 1.11× and 1.02×. Additionally, we report secondary metrics for the main configurations in Tab. 5. We see that peak memory is higher when merging for segmentation. This is due to the merge maps being stored for reconstruction. For detection, $\mathrm { A P _ { 5 0 } }$ is numerically preserved while count MAE rises from 11.44 to 11.92 and 13.91 and count $R ^ { 2 }$ falls from 0.703 to 0.682 and 0.585, showing that preserved detection AP does not guarantee preserved counting utility. For segmentation, all class IoUs and boundary F1 are reduced by merging, showing that the degradation seems to be uniform across classes. These phenotype-relevant metrics identify where degradation appears, but they do not prove that particular merges crossed a head instance or organ boundary.

We also report batch-one inference on a Raspberry Pi 5 in Fig. 4. The results are qualitatively similar to those on the GPU: classification is most mergetolerant, while segmentation and detection degrade faster. Aggressive settings reach about 2× CPU speedup, but compact detectors have low absolute mAP. We must also note that the device reported thermal throttling.

![](images/9e8cfb792988a140cea44cbd9f1ed10c122967f6d9ff8f406f0182174f177ca2.jpg)

![](images/5517166fa1f87f550b52b94938f7d6a7eba5162754e701cf14d49a2b7366bbcc.jpg)

![](images/0f15307334a8d97dc20bb3d958815df78180271d380840a8bcb523c7d2e2375e.jpg)  
Fig. 3: GPU performance–throughput trade-ofs on Base backbones. Each point is a merging configuration. The x-axis is FPS and the y-axis is Top-1, mAP, or mIoU. Dashed lines show method-specific Pareto frontiers.

![](images/c3f9a99d358fbf271fcb8ca902c3fa5938fd7e44bc18e5bba540e7a8c96898ec.jpg)

![](images/c13a2dab0e256225b9caeb592bff11e347f35a8e891843c3349302cb98cf73a9.jpg)

![](images/7d997d486814c0f02d198ee1311cd9711520de6a1be6927d8b3c83563bcf83d0.jpg)

![](images/df2a77eab54d7471bccabbb444a65561a6672f93a77ef8b93e5f0b7173fb6677.jpg)

![](images/1c109ede797f8c3d3373af42f174dec1f3dffa07c2eaf572575fbbcfd38aedcc.jpg)

![](images/061cfbeb1a38a648f0b9fdda5625991a17f0b009720751d4e77eb24ec6d86446.jpg)  
Fig. 4: Raspberry Pi 5 batch-one performance–throughput trade-ofs on fixed 100- image subsets. The x-axis is FPS and the y-axis is Top-1, mAP, or mIoU. Dashed lines show method-specific Pareto frontiers.

Table 5: Secondary metrics and peak allocated CUDA memory for Tab. 4. IoUs follow Background, Head/Spike, Stem, and Leaf. Boundary F1 excludes Background and uses a two-pixel tolerance.
<table><tr><td>Task</td><td>Method</td><td>Secondary metric</td><td></td><td>Value</td><td>Peak memory (GB)</td></tr><tr><td>Classification</td><td>Full</td><td>Macro-F1 (%)</td><td></td><td>99.15</td><td>0.636</td></tr><tr><td>Classification</td><td> $\mathrm { T o M e } ~ r = 1 6 , \mathrm { n o ~ p r o p } .$ </td><td>Macro-F1 (%)</td><td></td><td>99.01</td><td>0.607</td></tr><tr><td>Classification</td><td> $\mathrm { M P M } \ L = \{ 0 , 2 , \dot { 4 } , 6 , \dot { 8 } \}$ </td><td>Macro-F1 (%)</td><td></td><td>98.89</td><td>0.614</td></tr><tr><td>Detection</td><td>Full</td><td></td><td> $\mathrm { A P } _ { 5 0 } ~ ( \% ) ~ / \mathrm { c o u n t ~ M A E } ~ / \mathrm { c o u n t } ~ R ^ { 2 }$ </td><td> $6 4 . 4 6 \ : / \ : 1 1 . 4 4 \ : / \ : 0 . 7 0 3$ </td><td>4.226</td></tr><tr><td>Detection</td><td> $\mathrm { T o M e } \ r = 4 8$ </td><td></td><td> $\mathrm { A P _ { 5 0 } } \ \mathrm { ( \% ) } / \ \mathrm { c o u n t \ M A E } / \ \mathrm { c o u n t \ } R ^ { 2 }$ </td><td> $6 5 . 2 0 \ / \ 1 1 . 9 2 \ / \ 0 . 6 8 2$ </td><td>4.655</td></tr><tr><td>Detection</td><td> $\mathrm { M P M } \ L = \{ 1 0 \}$ </td><td></td><td> $\mathrm { A P _ { 5 0 } } \ \mathrm { ( \% ) } / \ \mathrm { c o u n t \ M A E } / \ \mathrm { c o u n t \ } R ^ { 2 }$ </td><td>65.47 / 13.91 / 0.585</td><td>4.214</td></tr><tr><td>Segmentation</td><td>Full</td><td> $\mathrm { I o U _ { 0 / 1 / 2 / 3 } \ / \ b o u n d a r y \ F 1 \ ( \% ) }$ </td><td></td><td> $7 0 . 1 4 \textrm { / } 3 3 . 3 3 \textrm { / } 7 4 . 7 1 \textrm { / } 7 6 . 3 2 \textrm { / } 3 4 . 8 8$ </td><td>6.023</td></tr><tr><td>Segmentation</td><td> $\mathrm { T o M e } ~ r = 4 8 , \mathrm { n o ~ p r o p . }$ </td><td> $\mathrm { I o U _ { 0 / 1 / 2 / 3 } \ / \ b o u n d a r y \ F 1 \ ( \% ) }$ </td><td></td><td>69.06 / 32.52 / 74.00 / 75.63 / 33.79</td><td>9.142</td></tr><tr><td>Segmentation</td><td>MPM L = {6, 9}</td><td> $\mathrm { I o U _ { 0 / 1 / 2 / 3 } \Omega / \ b o u n d a r y { \ F 1 \ ( \% ) } }$ </td><td></td><td> $6 9 . 8 1 \ / \ 3 2 . 7 9 \ / \ 7 4 . 4 1 \ / \ 7 6 . 0 5 \ / \ 3 4 . 1 4$ </td><td>6.719</td></tr></table>

Table 6: Batch-size-32 bfloat16 forward throughput with FlashAttention-2: median FPS ± standard error over three runs. Detection aggregates all tiled forward passes per image. ToMe uses no proportional attention.
<table><tr><td>Task</td><td>Model</td><td>Method</td><td>FPS</td><td>Speedup</td></tr><tr><td>Cls.</td><td>ViT-B/16</td><td>Baseline</td><td> $\mathbf { 4 0 4 7 . 8 \ : \pm { \ : 1 6 0 . 2 } }$ </td><td>1.00</td></tr><tr><td>Cls.</td><td>ViT-B/16</td><td>ToMe r = 24</td><td> $3 7 2 3 . 3 \pm 1 4 7 . 7$ </td><td>0.92</td></tr><tr><td>Cls.</td><td>ViT-B/16</td><td>MPM  $L = \{ 0 , 2 , 4 , 6 , 8 \}$ </td><td> $1 4 0 1 . 6 \pm 3 . 5$ </td><td>0.35</td></tr><tr><td>Seg.</td><td>Segmenter-B/16</td><td>Baseline</td><td> $5 8 3 . 2 \pm 4 . 3$ </td><td>1.00</td></tr><tr><td>Seg.</td><td>Segmenter-B/16</td><td>ToMe r = 24</td><td> $5 8 8 . 2 \pm 5 . 1 $ </td><td>1.01</td></tr><tr><td>Seg.</td><td>Segmenter-B/16</td><td>MPM  $L = \{ 0 , 2 , 4 , 6 , 8 \}$ </td><td> $\mathbf { 7 7 4 . 6 \pm 1 7 . 3 }$ </td><td>1.33</td></tr><tr><td>Det.</td><td>YOLOS-B</td><td>Baseline</td><td> ${ \bf 8 . 3 2 5 \pm 0 . 0 2 1 }$ </td><td>1.00</td></tr><tr><td>Det.</td><td>YOLOS-B</td><td>ToMe  $r = 2 4$ </td><td> $5 . 3 4 8 \pm 0 . 0 0 9$ </td><td>0.64</td></tr><tr><td>Det.</td><td>YOLOS-B</td><td> $\mathrm { M P M } \ L = \{ 0 , 2 , 4 , 6 , 8 \}$ </td><td> $7 . 0 6 2 \pm 0 . 0 1 4$ </td><td>0.85</td></tr></table>

## 4.1 FlashAttention-2

On recent GPUs, FlashAttention-2 [5] reduces the attention cost of the unmerged baseline, so token merging no longer guarantees higher throughput. Tab. 6 reports forward FPS with FlashAttention-2 enabled. For classification, ToMe and aggressive MPM reach only 0.92× and 0.35× baseline throughput. Aggressive MPM improves Segmenter-B/16 throughput by 1.33×. Both ToMe (0.64×) and MPM (0.85×) remain slower for tiled YOLOS-B. Once attention is optimized, merge construction, reconstruction, padding, and task-specific pipeline costs can dominate the savings from shorter sequences. Token reduction therefore cannot be treated as backend-independent acceleration.

## 4.2 Merge Locality

The locality analysis connects token merging to a specific property of plant imagery: self-similarity, visualized in Fig. 5. Wheat canopies contain many similar leaves, stems, and heads, so a feature-space merger may find visually plausible pairs that are spatially distant. Neither ToMe nor MPM explicitly enforces spatial locality when selecting token pairs. ToMe forms pairs from attentionkey similarity, whereas MPM uses mutual nearest neighbors in feature space. As a result, both methods may merge tokens that are far apart on the image grid if their representations are similar. Whether this is harmful depends on the task. For canopy classification, distant but similar regions can support the same global decision. For wheat-head detection and wheat-organ segmentation, the same operation can blur instance identity and organ boundaries or erase thin organs.

![](images/05d63f940827b4c3feef36d07088647641079ab95c8a70600e536d1fd0b6e435.jpg)  
Fig. 5: DINOv3 [24] patch-level self-similarity in a wheat canopy image. The red point marks the query patch, warm colors indicate higher cosine similarity at other locations, illustrating repeated visual structure in wheat imagery. This illustration is not a ToMe or MPM merge map and is not used quantitatively.

Table 7: Patch-grid Manhattan distances pooled over all merged token pairs for each representative task–method configuration.
<table><tr><td>Task</td><td>Method</td><td>Mean</td><td>Median</td><td>P95</td><td>P99</td><td>d ≤ 2 (%) d ≤ 4 (%)</td></tr><tr><td>WGSP cls.</td><td>ToMe r = 24</td><td>5.03</td><td>4.29</td><td>12.0</td><td>16.0</td><td>28.4 48.8</td></tr><tr><td>WGSP cls.</td><td>MPM L = {2}</td><td>3.43</td><td>2.00</td><td>11.0</td><td>14.0</td><td>56.4 72.7</td></tr><tr><td>GWFSS seg.</td><td>ToMe r = 24</td><td>3.24</td><td>1.50</td><td>13.0</td><td>29.0</td><td>69.5 84.7</td></tr><tr><td>GWFSS seg.</td><td>MPM L = {2}</td><td>2.23</td><td>1.00</td><td>5.0</td><td>18.0</td><td>78.6 93.7</td></tr><tr><td>GWHD det.</td><td>ToMe r = 24</td><td>1.52</td><td>1.00</td><td>3.0</td><td>6.0</td><td>89.6 97.6</td></tr><tr><td>GWHD det.</td><td>MPM L = {2}</td><td>1.31</td><td>1.00</td><td>2.0</td><td>3.0</td><td>96.3 100.0</td></tr></table>

To quantify this behavior, we measure the Manhattan distance between merged token pairs on the patch grid. For representative ToMe and MPM configurations, Tab. 7 reports the mean, median, 95th and 99th percentiles, and the fractions of pairs within distances 2 and 4.

The resulting merge patterns reflect the task structure. In WGSP classification, merged pairs are substantially more spread out: ToMe has median distance 4.29 and only 48.8% of pairs lie within four steps. The corresponding MPM values are 2.00 and 72.7%. In GWHD, both methods have median distance 1.00 and at least 97.6% of pairs lie within four steps. GWFSS is intermediate, although its tail remains broad for ToMe. These distributions are consistent with the canopy-level nature of classification, where similar visual evidence can occur across large regions, while dense tasks preferentially match nearby structures.

## 4.3 Batch Size and Padding Analysis

MPM’s input-dependent compression pads each batch to its longest merged sequence. This cost is absent at batch one but can reduce ofline throughput. On 100 fixed batches of 32 test images, Tab. 8 reports merged-token counts, retainedtoken distributions, and padding. Aggressive merging averages 20.3, 70.7, and 62.9 padded tokens per classification, segmentation, and detection image, with 95th-percentile costs of 25.5, 91.6, and 86.7. Mean retained count alone is therefore an incomplete eficiency proxy.

Table 8: MPM adaptive-padding on 100 fixed batches of 32 test images, reused across schedules. Pad/img is final padding per image after the last MPM insertion.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Level</td><td rowspan="2">MPM layers L</td><td rowspan="2">Merged/img</td><td colspan="2">Final tokens/img</td><td colspan="2">Pad/img</td></tr><tr><td>Median</td><td>p95</td><td>Avg.</td><td>p95</td></tr><tr><td>Cls.</td><td>Low</td><td>{10}</td><td>42.0</td><td>153.0</td><td>167.0</td><td>18.9</td><td>26.4</td></tr><tr><td>Cls.</td><td>Medium</td><td>{6, 9}</td><td>75.6</td><td>120.0</td><td>130.0</td><td>12.4</td><td>17.9</td></tr><tr><td>Cls.</td><td>Aggressive</td><td>{0, 2, 4, 6, 8, 10}</td><td>136.8</td><td>59.0</td><td>75.0</td><td>20.3</td><td>25.5</td></tr><tr><td>Seg.</td><td>Low</td><td>{10}</td><td>238.1</td><td>782.0</td><td>824.0</td><td>58.3</td><td>81.5</td></tr><tr><td>Seg.</td><td>Medium</td><td>{6, 9}</td><td>433.4</td><td>587.0</td><td>646.0</td><td>66.7</td><td>88.3</td></tr><tr><td>Seg.</td><td>Aggressive</td><td>{0, 2, 4, 6, 8, 10}</td><td>768.0</td><td>250.0</td><td>311.0</td><td>70.7</td><td>91.6</td></tr><tr><td>Det.</td><td>Low</td><td>{10}</td><td>438.1</td><td>1162.0</td><td>1182.0</td><td>26.2</td><td>37.2</td></tr><tr><td>Det.</td><td>Medium</td><td>{6,9}</td><td>734.2</td><td>866.0</td><td>895.0</td><td>38.7</td><td>54.8</td></tr><tr><td>Det.</td><td>Aggressive</td><td>{0, 2, 4, 6, 8, 10}</td><td>1264.6</td><td>334.0</td><td>383.0</td><td>62.9</td><td>86.7</td></tr></table>

## 5 Discussion and Limitations

Plant-Specific Interpretation. Training-free token merging helps plant phenotyping only when the task, merging schedule, and deployment path align. Growthstage classification is the clearest case: redundant canopy evidence can be compressed while preserving the global decision. Wheat-head detection and wheatorgan segmentation expose the limits, because repeated heads, thin leaves, stems, and organ boundaries are biological signal rather than mere visual redundancy. The same self-similarity that makes wheat images attractive for compression can therefore make aggressive merging unsafe. The secondary metrics make this limitation concrete: AP<sub>50</sub> can remain stable while count MAE worsens and count R<sup>2</sup> falls, and modest mIoU loss can coexist with lower Head/Spike IoU and boundary F1.

However, these associations do not establish a biological mechanism. The benchmark was not stratified by Zadoks stage, canopy or head density, occlusion, organ scale, perspective, or acquisition height. Locality cannot reveal which semantics a pair connects. Classification tolerance is consistent with global evidence requirements and dense-task fragility with spatial precision, but distance alone does not explain the curves.

Deployment Guidance. Token merging should be selected per task and runtime, not enabled by default. On the standard GPU path, classification is the strongest case: ToMe $r = 1 6$ without proportional attention and MPM $L = \{ 0 , 2 , 4 , 6 , 8 \}$ yield 1.59× and 1.47× speedups with losses below 0.3 Top-1 points. These are measured operating points, not universal loss tolerances.

Segmentation is conditional. ToMe $r \ = \ 4 8$ and MPM $L = \{ 6 , 9 \}$ improve throughput by 1.33× and 1.16×, but reduce mIoU, every class IoU, and boundary F1. Region and boundary losses must be checked against the application’s tolerance. Detection is the weakest case: mild ToMe preserves mAP only approximately, gives modest speedup, and increases count MAE, while MPM is nearly throughput-neutral. With FlashAttention-2, both tested detection mergers are slower than Full, suggesting that the tiled pipeline is the larger bottleneck.

The same task-level pattern also appears across the tested deployment environments, from the GPU to the Raspberry Pi 5. Measurements show more consistent CPU gains, but the device reported throttling. On every platform, the full baseline should first be profiled with the intended precision, batch size, backend, and complete pipeline. A shorter sequence is useful only if it improves performance along that deployment path.

Scope and Limitations. Plain ViT classifiers, Segmenter, and YOLOS isolate sequence behavior. Hierarchical backbones, modern detectors, and foundationmodel decoders may difer. We compare two training-free mergers rather than task-trained reduction. Joint training may improve robustness but removes the fixed-checkpoint control.

We also did not profile the energy cost of merging, which may be relevant for edge deployment. Finally, the benchmark is limited to wheat phenotyping. Other crops and plant organs may have diferent self-similarity and spatial requirements.

## Acknowledgements

This research was funded by the European Union’s Horizon Europe Research and Innovation Programme under the PHENET project, Grant Agreement Number 101094587. This work was granted access to the HPC resources of IDRIS under the allocation 2024-AD010115553 made by GENCI.

## References

1. Bergner, B., Lippert, C., Mahendran, A.: Token cropr: Faster ViTs for quite a few tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9740–9750 (June 2025)

2. Bolya, D., Fu, C.Y., Dai, X., Zhang, P., Feichtenhofer, C., Hofman, J.: Token merging: Your ViT but faster. In: International Conference on Learning Representations (2023)

3. Chen, F., Giufrida, M.V., Tsaftaris, S.A.: Adapting vision foundation models for plant phenotyping. In: Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. pp. 604–613 (October 2023). https://doi.org/10. 1109/ICCVW60793.2023.00067

4. Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.V.: RandAugment: Practical automated data augmentation with a reduced search space. In: Advances in Neural Information Processing Systems. vol. 33, pp. 18613–18624 (2020)

5. Dao, T.: FlashAttention-2: Faster attention with better parallelism and work partitioning. In: International Conference on Learning Representations (2024)

6. David, E., Ogidi, F., Smith, D., Chapman, S., de Solan, B., Guo, W., Baret, F., Stavness, I.: Global wheat head detection challenges: Winning models and application for head counting. Plant Phenomics 5, 0059 (2023). https://doi.org/10. 34133/plantphenomics.0059

7. David, E., Serouart, M., Smith, D., Madec, S., Velumani, K., Liu, S., Wang, X., Pinto, F., Shafiee, S., Tahir, I.S.A., Tsujimoto, H., Nasuda, S., Zheng, B., Kirchgessner, N., Aasen, H., Hund, A., Sadeghi-Tehran, P., Nagasawa, K., Ishikawa, G., Dandrifosse, S., Carlier, A., Dumont, B., Mercatoris, B., Evers, B., Kuroki, K., Wang, H., Ishii, M., Badhon, M.A., Pozniak, C., LeBauer, D.S., Lillemo, M., Poland, J., Chapman, S., de Solan, B., Baret, F., Stavness, I., Guo, W.: Global wheat head detection 2021: An improved dataset for benchmarking wheat head detection methods. Plant Phenomics 2021, 9846158 (2021). https://doi.org/10.34133/2021/ 9846158

8. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)

9. Fang, Y., Liao, B., Wang, X., Fang, J., Qi, J., Wu, R., Niu, J., Liu, W.: You only look at one sequence: Rethinking transformer in vision through object detection. In: Advances in Neural Information Processing Systems. vol. 34 (2021)

10. Fayyaz, M., Koohpayegani, S.A., Jafari, F.R., Sengupta, S., Joze, H.R.V., Sommerlade, E., Pirsiavash, H., Gall, J.: Adaptive token sampling for eficient vision transformers. In: Computer Vision – ECCV 2022. pp. 396–414. Springer (2022)

11. Haurum, J.B., Escalera, S., Taylor, G.W., Moeslund, T.B.: Which tokens to use? investigating token reduction in vision transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. pp. 773–783 (October 2023). https://doi.org/10.1109/ICCVW60793.2023.00085

12. Joice, A., Tufaique, T., Tazeen, H., Igathinathane, C., Zhang, Z., Whippo, C., Hendrickson, J., Archer, D.: Applications of Raspberry Pi for precision agriculture—a systematic review. Agriculture 15(3), 227 (2025). https://doi.org/10.3390/ agriculture15030227

13. Liang, Y., Ge, C., Tong, Z., Song, Y., Wang, J., Xie, P.: EViT: Expediting vision transformers via token reorganizations. In: International Conference on Learning Representations (2022)

14. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019)

15. Lu, C., de Geus, D., Dubbelman, G.: Content-aware token sharing for eficient semantic segmentation with vision transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 23631–23640 (June 2023). https://doi.org/10.1109/CVPR52729.2023.02263

16. Marin, D., Chang, J.H.R., Ranjan, A., Prabhu, A., Rastegari, M., Tuzel, O.: Token pooling in vision transformers for image classification. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 12–21 (2023). https://doi.org/10.1109/WACV56688.2023.00010

17. Murphy, K.M., Ludwig, E., Gutierrez, J., Gehan, M.A.: Deep learning in imagebased plant phenotyping. Annual Review of Plant Biology 75, 771–795 (2024). https://doi.org/10.1146/annurev-arplant-070523-042828

18. Norouzi, N., Orlova, S., de Geus, D., Dubbelman, G.: ALGM: Adaptive localthen-global token merging for eficient semantic segmentation with plain vision transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15773–15782 (June 2024). https://doi.org/10. 1109/CVPR52733.2024.01493

19. Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., Hsieh, C.J.: DynamicViT: Eficient vision transformers with dynamic token sparsification. In: Advances in Neural Information Processing Systems. vol. 34 (2021)

20. Ravé, S., Rasti, P., Rousseau, D.: MPM: Mutual pair merging for eficient vision transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Findings. pp. 2998–3008 (2026)

21. Ryoo, M.S., Piergiovanni, A., Arnab, A., Dehghani, M., Angelova, A.: Token-Learner: Adaptive space-time tokenization for videos. In: Advances in Neural Information Processing Systems. vol. 34 (2021)

22. Sehaba, M.E.A., Crispim-Junior, C., Tougne Rodet, L.: Embedded plant recognition: A benchmark for low footprint deep neural networks. In: Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. pp. 670–677 (October 2023). https://doi.org/10.1109/ICCVW60793.2023.00074

23. Shen, L., Ding, G., Jackson, R., Ali, M., Liu, S., Mitchell, A., Shi, Y., Lu, X., Dai, J., Deakin, G., Frels, K.A., Cen, H., Ge, Y.F., Zhou, J.: GSP-AI: An AI-powered platform for identifying key growth stages and the vegetative-to-reproductive transition in wheat using trilateral drone imagery and meteorological data. Plant Phenomics 6, 0255 (2024). https://doi.org/10.34133/plantphenomics.0255

24. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S.E., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., Bojanowski, P.: DINOv3. Trans. Mach. Learn. Res. 2026 (2026), https://openreview.net/ forum?id=2NlGyqNjns

25. Strudel, R., Garcia, R., Laptev, I., Schmid, C.: Segmenter: Transformer for semantic segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7262–7272 (2021)

26. Wang, Z., Zenkl, R., Greche, L., de Solan, B., Samatan, L.B., Ouahid, S., Visioni, A., Robles-Zazueta, C.A., Pinto, F., Perez-Olivera, I., Reynolds, M.P., Zhu, C., Liu, S., D’argaignon, M.P., Lopez-Lozano, R., Weiss, M., Marzougui, A., Roth, L., Dandrifosse, S., Carlier, A., Dumont, B., Mercatoris, B., Fernandez, J., Chapman, S., Najafian, K., Stavness, I., Wang, H., Guo, W., Virlet, N., Hawkesford, M.J., Chen, Z., David, E., Gillet, J., Irfan, K., Comar, A., Hund, A.: The global wheat full semantic organ segmentation (GWFSS) dataset. Plant Phenomics 7(3), 100084 (2025). https://doi.org/10.1016/j.plaphe.2025.100084

27. Yin, H., Vahdat, A., Alvarez, J.M., Mallya, A., Kautz, J., Molchanov, P.: A-ViT: Adaptive tokens for eficient vision transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10799–10808 (2022)