# Lightweight Deep Learning Framework for Astrocyte-Centric 3D Reconstruction across Normal and Stress-associated Ultrastructural States

Bowen Deng,<sup>1,</sup> <sup>∗</sup> Gwang-Jin Jeong,<sup>2,</sup> <sup>3</sup> Seulgi Noh,<sup>2</sup> Ji Young Mun,<sup>2,</sup> <sup>†</sup> and Na Young Kim<sup>1,</sup> <sup>4,</sup> <sup>‡</sup>

<sup>1</sup>Institute for Quantum Computing, Department of Electrical and

Computer Engineering, University of Waterloo, Waterloo, Canada

<sup>2</sup>Department of Neuroscience, Korea Brain Research Institute, Daegu, Republic of Korea

<sup>3</sup>Department of New Biology, Daegu Gyeongbuk Institute of Science and Technology, Daegu, Republic of Korea

<sup>4</sup>Waterloo Institute for Nanotechnology, University of Waterloo, Waterloo, Canada

The morphology of astrocytes is closely associated with their ability to support and modulate neural circuit functions; however, its reconstruction from volume electron microscopy (vEM) data remains labor-intensive and computationally resource-intensive, placing substantial barriers on its broader adoption and scalability despite the high spatial resolution of vEM. We present AstroSeg-vEM, a lightweight, astrocyte-centric framework for morphology-preserving astrocyte segmentation, 3D reconstruction, and quantitative morphological analysis in vEM images. AstroSeg-vEM combines transformer-based 2D segmentation with dedicated post-processing methods and operates directly on high-resolution EM sections, making it well suited to anisotropic vEM data and downstream proofreading workflows while preserving the full in-plane resolution of the original EM data. Across four mouse hippocampal datasets representing normal and morphologically defined stress-associated astrocyte ultrastructural states, AstroSeg-vEM demonstrates robust performance and generalization, achieving average Dice and weighted Dice values of 0.92 and 0.95, respectively, on a held-out test dataset. Its lightweight architecture enables eficient, high-fidelity reconstruction on consumer-grade hardware, providing a scalable and accessible platform for large-scale astrocyte ultrastructure analysis and future astrocyte-neuron interaction studies.

## I. INTRODUCTION

Astrocytes are essential regulators of synaptic function, neural circuit homeostasis, and neurovascular coupling in the hippocampus<sup>1–3</sup>. Their fine perisynaptic and perivascular processes form highly ramified, spatially interwoven territories that interact with dendrites, synapses, and blood vessels. Notably, the astrocytic processes<sup>4–6</sup> from specialized nano-domains mediate neurotransmitter uptake, ion bufering, lipid traficking, and secretion of antioxidants and trophic factors that support neuronal survival and plasticity<sup>7</sup>; however, these structures are beyond the resolution of light microscopy<sup>6</sup> and quantitative analysis remains technically challenging<sup>8,9</sup>. Astrocyte structural heterogeneity<sup>10</sup>, defined here as nanoscale changes in irregular process organization and spatial coverage within the synaptic microenvironment<sup>11,12</sup>, is especially dificult to quantify in 3D analysis. Hence, high-resolution volume electron microscopy (vEM) is essential for ultrastructural analysis, enabling direct visualization of how astrocytes organize the local synaptic microenvironment, including perisynaptic process geometry, synaptic apposition, and spatial occupancy. Most astrocyte vEM studies still rely on manual or semi-automated segmentation and proofreading<sup>13–19</sup>, making reconstruction laborintensive, operator-dependent, dificult to scale, and susceptible to split and merge errors. These challenges are particularly problematic for astrocytes because their thin, fragmented, and highly interconnected processes occupy only a small fraction of each section while extending through dense ultrastructures containing dendrites, axons, synapses, and vascular structures. Thus, scalable and reproducible astrocyte-centered segmentation remains a major bottleneck for quantitative analysis of astrocyte morphology and cell–cell interactions in vEM datasets.

Recent advances in deep learning have substantially improved EM image segmentation and have provided greater automation of downstream reconstruction workflows. Fully 3D segmentation architectures have demonstrated strong performance in vEM segmentation, especially in large-scale neuron-centric connectomics<sup>20</sup>. While achieving strong segmentation performance, such 3D approaches are computationally demanding and typically require high-memory GPUs (> 16 GB VRAM), hampering scalability and throughput. For astrocyte segmentation in vEM, 3D deep learning-based pipelines have also been applied to large-scale connectomic researches<sup>21–26</sup>. A flood-filling network-based architecture<sup>20,22</sup> has been implemented on the human brain tissue vEM dataset to segment neurons and other cellular components, including astrocytes, in the human temporal cortex. A more recent 3D EM segmentation approach has further improved the eficiency of dense connectomic reconstruction through strategies such as local shape descriptor-based afinity learning and scalable block-wise processing<sup>27</sup>. Since this approach was developed for general connectomics, it prioritizes large-scale neural circuit reconstruction rather than astrocyte-centric reconstruction. Although such general-purpose approaches may also be applicable to glial structures, they are not specifically optimized for preserving the irregular, and highly branched morphology of astrocytic processes or for astrocyte-focused morphometric workflows. This leaves a methodological gap for computationally eficient, astrocyte-centric segmentation and reconstruction tailored to anisotropic vEM data and downstream quantitative morphological analysis. In addition, current 3D approaches are often constrained by high computational demands, reduced robustness to anisotropic vEM characterized by high in-plane resolution but substantially coarser axial sampling, and dificulty in directly processing high-resolution inputs. They further require large-scale annotated volumetric datasets, whose acquisition is costly and labor-intensive. To overcome these limitations, two-dimensional (2D) segmentation architectures provide a practical and lightweight alternative for astrocyte vEM, balancing reconstruction fidelity with computational eficiency. By leveraging rich planar details while avoiding the heavy memory and computational demands associated with volumetric 3D models, 2D models consume less GPU memory, facilitate faster inference, and can be trained efectively on smaller annotated datasets.

In practice, this 2D modeling paradigm is commonly implemented using convolutional neural network (CNN) architectures such as $\mathrm { U - N e t ^ { 2 8 } }$ and its widely adopted 2D variants<sup>29–32</sup>, as well as context-aggregation models including DeepLab<sup>33–36</sup>, which have demonstrated strong performance in 2D image segmentation. More recently, transformer-based<sup>37–39</sup> and hybrid CNN–transformer architectures<sup>40,41</sup> have further improved the modeling of long-range spatial dependencies. Nevertheless, their limited input sizes, typically a few hundred pixels per side, restrict their applicability to high-resolution EM images. As a practical solution, patch-based 2D segmentation pipelines are widely adopted in EM image analysis<sup>42–46</sup>, enabling scalable training and inference on high-resolution EM sections by subdividing them into manageable patches while preserving local ultrastructural details. Yet the design and impact of patch sampling strategies remain underexplored. This limitation is particularly pronounced in astrocyte vEM segmentation, where astrocyte-positive regions (foreground) occupy only a small and spatially fragmented fraction of each section due to the characteristic star-shaped morphology of astrocytes (Fig. 1A). The complex geometry of astrocytes makes accurate segmentation and subsequent volumetric reconstruction from vEM data challenging<sup>8,9</sup>. Because section-wise 2D segmentation does not explicitly model inter-section continuity, morphology-preserving post-processing is used as a lightweight 3D-aware step to compensate for this missing volumetric context and to assemble consistent 3D astrocyte structures from 2D predictions. Compared with native 3D model training and inference, this post-processing step uses a small number of interpretable parameters, runs without GPU acceleration, and adds relatively low computational overhead while improving volumetric consistency.

These limitations are particularly critical in the hippocampal stratum radiatum, a region densely populated with CA1 apical dendrites and excitatory synapses, where astrocyte processes are positioned to regulate synaptic transmission, extracellular difusion, and activity-dependent plasticity<sup>5,47</sup>. EM enables direct visualization of perisynaptic astrocytic processes and their physical relationship with synapses, providing a more accurate and disease-relevant assessment of astrocyte morphology and its potential impact on circuit function<sup>47</sup>.

Here, we propose AstroSeg-vEM, a lightweight, astrocyte-centric deep learning framework for astrocyte labeling and analysis. AstroSeg-vEM integrates automated 2D segmentation with morphology-preserving reconstruction for large-scale astrocyte analysis in vEM data (Fig. 2). To our knowledge, this represents one of the first demonstrations of an eficient 2D deep learning framework for automated astrocyte segmentation and morphology-preserving 3D reconstruction, with outputs designed to support downstream expert-guided proofreading and analysis. The framework operates directly on high-resolution EM sections and is optimized for the nonregional, highly interconnected morphology of astrocytes, balancing between reconstruction fidelity and computational eficiency. Crucially, this unified framework not only enables quantitative characterization of astrocytic ultrastructure states but also provides an intrinsic metric for evaluating segmentation performance.

## II. RESULTS

## A. Astrocyte datasets and experimental setup

We evaluate the robustness of AstroSeg-vEM across diferent ultrastructural conditions using four independent vEM datasets (D1, D2, D3, and D4 in Fig. 1). All datasets were acquired from the mouse hippocampus by array tomography, and key acquisition parameters, including strongly anisotropic voxel spacing, image and physical volumes, are summarized in Fig. 1B. Note that the datasets exhibit pronounced voxel anisotropy with axial thicknesses (70 or 100 nm) that are 14 to 20 times larger than the in-plane resolution $( 5 \times 5 ~ \mathrm { n m ^ { 2 } } )$ They capture expansive in-plane fields of view of up to $6 0 \times 6 0$ µm<sup>2</sup> (D1) and $5 5 \times 7 0 ~ \mathrm { { \textmu m } ^ { 2 } \ ( D 4 ) }$ , while their axial coverage is limited by the finite number of sections. Based on these characteristics, the datasets encompass two distinct astrocyte profiles: normal and stress-associated astrocyte ultrastructural states (Fig. 1A). Here, astrocyte ultrastructural states (AUS) refer to distinct electron microscopyresolvable three-dimensional configurations of astrocyte morphology, including process architecture under normal (N-AUS) and stress-associated (S-AUS) conditions.

Three distinct experimental settings (Table 1) are defined to systematically evaluate the robustness of 2D segmentation and post-processing pipeline in AstroSegvEM across N-AUS and S-AUS. Experiments 1 (Exp 1) and 2 (Exp 2) establish a morphological cross-validation by training on S-AUS datasets and testing on N-AUS datasets and vice versa. Experiment 3 (Exp 3) assesses the full AstroSeg-vEM framework for biological analysis, training on three datasets (D1, D2, D3) and reserving D4 as a held-out test dataset for 3D reconstruction and quantitative morphological analysis.

A  
![](images/f4c47d60c50a032965dfe51352b9593a2201ac5dbcd1b82f9760640483b3112e.jpg)

![](images/8ea51ca2cb851e5af809b9591a61752ac8dd719e898dd76cbf6be06c59147c64.jpg)

![](images/03fdb19e75d216df9289b7e894f975c1b17e2f0f857934d953f0c529d8aa16c2.jpg)

![](images/ea176105e9b7e1502ca562b86a1de2465c6854c175fe0a64fcb82eff596d04b4.jpg)

![](images/0183d963de965a6374df2490a072c4a7f4f72632b57becf89edec8814a443364.jpg)

![](images/a83754af7b0a363bf5358a93ae5e4f13ab0e9fb079cb8dd12b95c404edc62dba.jpg)

![](images/09f2ce189619ab17459e65e88a6c7e4096eee03c4309eb6fd8cf7e18de927729.jpg)

![](images/7c203e1aa52b5d5336c628f9ae87f6c9f20f22bfce992c5f906a0fe1f775522d.jpg)

B
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Phenotype</td><td rowspan=1 colspan=1>xy Resolution(nm×nm)</td><td rowspan=1 colspan=1>z Thickness(nm)</td><td rowspan=1 colspan=1>Image Volume(pixel × pixel × sections)</td><td rowspan=1 colspan=1>Physical Volume(μm×μm× μm)</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>S-AUS</td><td rowspan=1 colspan=1>5×5</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1> $1 2 , 0 0 0 \times 1 2 , 0 0 0 \times 1 1 0$ </td><td rowspan=1 colspan=1> $6 0 \times 6 0 \times 7 . 7$ </td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>N-AUS</td><td rowspan=1 colspan=1>5×5</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1> $6 , 0 0 0 \times 6 , 0 0 0 \times 8 8$ </td><td rowspan=1 colspan=1> $3 0 \times 3 0 \times 6 . 1 6$ </td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>N-AUS</td><td rowspan=1 colspan=1>5×5</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1> $1 0 , 0 0 0 \times 1 0 , 0 0 0 \times 5 3$ </td><td rowspan=1 colspan=1> $5 0 \times 5 0 \times 5 . 3$ </td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>S-AUS</td><td rowspan=1 colspan=1>5×5</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1> $1 1 , 0 0 0 \times 1 4 , 0 0 0 \times 8 3$ </td><td rowspan=1 colspan=1> $5 5 \times 7 0 \times 8 . 3$ </td></tr></table>

FIG. 1. Visualization and statistics of astrocyte morphological profiles across vEM datasets. A Representative EM sections from astrocytes under stress (S-AUS) datasets (D1, D4) featuring complex, hyper-ramified morphologies and normal (N-AUS) datasets (D2, D3) exhibiting standard baseline morphologies. Red overlays indicate astrocyte masks. B Summary of dataset statistics and imaging parameters across four vEM datasets. Panel A overlays are downsampled for display from the native $\mathrm { 5 \times 5 \ n m ^ { 2 } }$ in-plane acquisition resolution to efective in-plane pixel (px) spacings of 58.6 nm/px (D1), 29.3 nm/px (D2), 48.8 nm/px (D3), and 68.4 nm/px (D4).

## B. Astrocyte-centric segmentation across diverse ultrastructural states

To validate the core capabilities of the transformerbased segmentation model within AstroSeg-vEM, we compared its raw, unprocessed segmentation performance with representative baselines, including $\mathrm { \bar { U } { - } N e t ^ { 2 8 } }$ and SegFormer<sup>39</sup>, as well as $\mathrm { S w i n – U n e t ^ { 3 7 } }$ , under the same experimental setting (Exp 3). Segmentation performance was evaluated using four metrics: Dice score (Dice), weighted Dice (wDice), precision, and recall; the mathematical definition of wDice is written in Methods. Table 2 summarizes that the proposed model achieves higher Dice and precision, while maintaining comparable performance in wDice and recall. It attains the highest Dice score of $0 . 7 0 \pm 0 . 0 2$ and precision of $0 . 5 8 \pm 0 . 0 3$ , outperforming both SegFormer and Swin-Unet, with a modest but consistent paired advantage over SegFormer across sections (Fig. S3), whereas U-Net exhibits markedly degraded performance $( \mathrm { D i c e } = 0 . 3 6 \pm 0 . 0 5 )$ . Consistently high performance of AstroSeg-vEM with wDice $( 0 . 9 4 \pm 0 . 0 2 ) $ and recall $( 0 . 8 9 \pm 0 . 0 4 ) $ indicates that primary astrocyte structures are consistently captured. Improvements in Dice and precision reflect enhanced delineation of fragmented astrocytic processes, highlighting the advantage of transformer-based architectures in modeling fine-scale spatial dependencies. Despite superior segmentation performance, our AstroSeg-vEM introduces only a small increase in model size (+11.7%) and computational cost (+28.7% GFLOPs) relative to SegFormer, while achieving 7.5% faster inference time per section. Compared with U-Net, it reduces computational cost by 89.2% and inference time by 50.8% with 20.1% fewer parameters. Relative to Swin-Unet, AstroSeg-vEM further reduces parameter count by 33.3%, computational cost by 39.4%, and inference time by 14.0%. These results demonstrate that AstroSeg-vEM optimizes the trade-of between segmentation accuracy, runtime eficiency, and model complexity, making it well-suited for capturing the complex and highly ramified astrocyte morphology.

A  
![](images/aee20b6826aca9f5b1ef9cc49386f0ecdfec94570cd2fb07d215e760293fc68e.jpg)  
FIG. 2. Lightweight 2D astrocyte segmentation and morphology-preserving reconstruction in AstroSeg-vEM. A Patch-level dataset generation leveraging foreground-aware sampling. B Hierarchical transformer-based U-Net with a MiT-B2 encoder capturing multi-scale morphological features through self-attention. C Overlapping sliding-window inference strategy ensuring in-plane spatial continuity via probabilistic aggregation (See Methods). D Morphology-preserving post-processing by sequentially performing soma-guided component refinement, volumetric structural consolidation, and in-plane morphological correction to filter spatially inconsistent artifacts and refine the connectivity of astrocytic processes across vEM sections. Full section overviews and post-processing renders in A, C, and D are downsampled for display to an efective in-plane pixel (px) spacing of 29.3 nm/px.

TABLE 1. Experimental design and dataset allocation
<table><tr><td>Experiment</td><td>Objective</td><td>Training Set</td><td>Validation Set</td><td>Testing Set</td><td>Downstream Tasks</td></tr><tr><td>Exp 1</td><td>Cross-domain (S-AUS → N-AUS)</td><td>D1, D4 (first 80%)</td><td>D1, D4 (last 20%)</td><td>D2, D3</td><td>Segmentation Accuracy</td></tr><tr><td>Exp 2</td><td>Cross-domain</td><td>D2, D3</td><td>D2, D3</td><td>D1, D4</td><td>Segmentation Accuracy</td></tr><tr><td>Exp 3</td><td>(N-AUS → S-AUS) Full Pipeline &amp; Morphology</td><td>(first 80%) D1, D2, D3 (first 80%)</td><td>(last 20%) D1, D2, D3 (last 20%)</td><td>D4</td><td>3D Reconstruction &amp; Analysis</td></tr></table>

Building upon robust raw predictions, we evaluated the full segmentation pipeline including a dedicated postprocessing module. The distributions of Dice, wDice, precision, and recall across Exp 1–3 are summarized in Fig. 3A and Table 6. Robustness was assessed through morphological cross-validation (Exp 1 and Exp 2). Within this cross-validation phase, N-AUS datasets (D2, D3) generally exhibit more compact interquartile ranges of four metrics compared to the S-AUS datasets (D1, D4). The increased variance and extended lower-scoring tails in the S-AUS datasets are concentrated in a subset of sections where the ground-truth astrocytic processes are sparse and thin, whereas the post-processing threshold is mainly tuned for large and coherent astrocyte regions (Fig. S1). Across datasets, Dice and recall generally show broader interquartile ranges and lower medians than precision and wDice. Nevertheless, performance remains suficient for large-scale morphological discovery, as the median Dice and recall values are consistently above 0.69 and 0.78, respectively. Notably, this cross-validation utilizes a uniform set of hyperparameters, demonstrating that our method efectively generalizes across topological features without individual tuning for each dataset. A detailed discussion of why proposed wDice is a good complementary metric for astrocyte segmentation is provided in the Discussion.

Following the cross-validation, we trained a more comprehensive model on combined N-AUS and S-AUS datasets to evaluate it on the held-out D4 (Exp 3). Recall remains stable with a median of 0.93, while wDice reaches its peak median of 0.96. For Exp 3, the postprocessing parameters are further optimized on D1–D3, without using D4, to maximize the structural continuity necessary for high-fidelity 3D reconstruction. By bridging the gap between raw pixel identification and volumetric connectivity, the pipeline provides cleaner section-wise predictions for high-fidelity 3D reconstruction and downstream proofreading. In order to test how the amount of training data afects performance, we additionally evaluated two protocols with fewer training volumes, twofold cross-validation (2-fold) and leave-one-volume-out (LOVO) cross-validation. Their training, validation and test volumes are listed in Supplementary Table S1. Exp 3 was trained on D1, D2 and D3, with the last 20% of sections of each volume held out for validation, whereas LOVO fold 1 was trained on D1 and D2 and validated on D3, so the two protocols difer in the amount of training data as well as in validation strategy. On the 83 sections of D4, post-processed Dice decreased from $0 . 9 2 \pm 0 . 0 2$ in Exp 3 to $0 . 7 1 \pm 0 . 0 4$ in LOVO fold 1 (Supplementary Table S1), and raw Dice decreased from $0 . 7 0 \pm 0 . 0 2$ to 0.58 ± 0.04, which we interpret as limited robustness when fewer training volumes are available. In 2-fold-1, which was trained on D1 and validated on D4, post-processed Dice and wDice on the 88 sections of D2 were $0 . 4 2 \pm 0 . 4 3$ and $0 . 4 7 \pm 0 . 4 3$ , compared with $0 . 5 3 \pm 0 . 1 8$ and 0.94 ± 0.05 before post-processing. D4 served as a test-only volume in Exp 3 and LOVO.

AstroSeg-vEM incorporates three post-processing modules designed to refine segmentation outputs and improve structural coherence: soma-guided component refinement (SCR), volumetric structural consolidation (VSC), and in plane morphological correction (IMC). A systematic abla tion study (Table 3) shows that the AstroSeg-vEM postprocessing pipeline consistently improves Dice in all five evaluated settings, but not always wDice. Already high at baseline, wDice is unchanged or slightly higher in Exp 1 and Exp 3 and decreases in Exp 2, from 0.91 to 0.82 in D1 and from 0.94 to 0.89 in D4. Although the baseline segmentation model already achieves high wDice (0.87–0.94), indicating that primary astrocyte structures are largely captured, Dice remains low due to false predictions. The SCR module reduces spatial inconsistencies, leading to higher Dice, especially in Exp 1 D2 and Exp 2 D1. The VSC module provides the largest gain by enforcing global 3D connectivity, raising Dice from 0.72 to 0.88 in Exp 1 D2 and from 0.43 to 0.67 in Exp 2 D1. The IMC module further refines topology, yielding smaller but consistent gains (e.g. 0.88 to 0.90 in Exp 1 D2). Note that wDice remains relatively stable throughout the ablation process in Exp 1 and Exp 3, suggesting that core astrocytic structures are already identified by the segmentation model, whereas post-processing primarily restores structural continuity rather than improving pixel-wise overlap. These results highlight the importance of morphology-preserving post-processing for producing structurally coherent representations suitable for 3D reconstruction and expertguided proofreading. Consistent with this, a per-section decomposition of residual errors into the false discovery rate (FDR) and the miss rate (Methods) shows that postprocessing lowers the FDR markedly in all five settings (Fig. S2A), whereas the miss rate is similar or lower in Exp 1 and Exp 3 and higher in Exp 2 (Fig. S2B). Because the two rates have diferent denominators, predicted-positive pixels for the FDR and ground-truth-positive pixels for the miss rate, they are presented as complementary error measures. Component-level analysis of the Exp 3 D4 volume gives the same picture (Supplementary Table S2).

TABLE 2. Segmentation Architectures Performance and Eficiency Comparison Dice, wDice, precision, and recall are mean ± standard deviation across $n = 8 3$ sections from a single held-out volume $\left( \mathrm { E x p ~ 3 ~ D 4 } \right)$ , not independent biological replicates. Values are computed at the native 5 nm resolution and are not directly comparable with the 20 nm common-grid values of the 3D nnU-Net comparison (see Discussion).
<table><tr><td>Architecture</td><td>Dice  $\left( \mathbf { E x p } \ 3 \right)$ </td><td>wDice  $\left( \mathbf { E x p } \ 3 \right)$ </td><td>Precision  $\left( \mathbf { E x p } \ 3 \right)$ </td><td></td><td>Recall  $\left( \mathbf { E x p } \ 3 \right)$ </td><td>Inference Time Parameters per Section (s)</td><td></td><td>GFLOPs</td></tr><tr><td>AstroSeg-vEM (Ours)</td><td> ${ \bf 0 . 7 0 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 4 } \pm \mathbf { 0 . 0 2 }$ </td><td></td><td> $\mathbf { 0 . 5 8 \ : \pm { \ : 0 . 0 3 } }$ </td><td> $\mathbf { 0 . 8 9 \ : \pm { \ : 0 . 0 4 } }$ </td><td>101.96</td><td>27.60M</td><td>10.80</td></tr><tr><td>SegFormer39</td><td> $0 . 6 8 \pm 0 . 0 2$ </td><td> $0 . 9 4 \pm 0 . 0 3$ </td><td></td><td> $0 . 5 5 \pm 0 . 0 3$ </td><td> $\mathbf { 0 . 8 9 \ : \pm { \ : 0 . 0 4 } }$ </td><td>110.23</td><td>24.72M</td><td>8.39</td></tr><tr><td> $\mathrm { U - N e t ^ { 2 8 } }$ </td><td> $0 . 3 6 \pm 0 . 0 5$ </td><td> $0 . 9 1 \pm 0 . 0 2$ </td><td></td><td> $0 . 2 3 \pm 0 . 0 4$ </td><td> $0 . 8 7 \pm 0 . 0 4$ </td><td>207.37</td><td>34.53M</td><td>100.07</td></tr><tr><td> $\mathrm { S w i n – U n e t ^ { 3 7 } }$ </td><td> $0 . 6 5 \pm 0 . 0 6$ </td><td> $0 . 9 0 \pm 0 . 1 0$ </td><td></td><td> $0 . 5 4 \pm 0 . 0 3$ </td><td> $0 . 8 4 \pm 0 . 1 2$ </td><td>118.53</td><td>41.38M</td><td>17.82</td></tr></table>

Post-processing dramatically reduced the mean number of predicted connected components per section by approximately 90%, from 655.7 to 66.3 (ground truth, 29.1), and the mean number of spurious components per section by approximately 95%, from 599.9 to 30.3. The mean number of split ground-truth components per section also decreased from 10.0 to 5.0, whereas missed ground-truth components increased from 0.9 to 3.4 and merges were rare and unchanged (0.5 per section).

The S-AUS datasets pose extreme challenges for automated 2D segmentation due to fragmented membrane signals and atypical morphological distortions, whose biological complexities often lead to severe topological discontinuities in standard frameworks. As shown in the pixellevel comparison (Fig. 3B), our pipeline demonstrates remarkable robustness across representative z sections. Sections were selected by jointly considering quantitative performance (Dice and wDice) and morphological diversity spanning diferent score ranges and astrocytic structures across both N-AUS and S-AUS datasets rather than only the highest-scoring examples. Predominant yellow regions confirm a precise pixel-wise overlap that is particularly striking in the proximal regions, where the soma and primary branches are recovered with high fidelity. This high fidelity in the morphological center is critical, as these structures define the spatial territory and basic scafold of the astrocyte.

While minor misclassifications (indicated by red and blue pixels) are occasionally detectable in the distal processes far from the astrocyte center, these are primarily restricted to the peripheral fringes where EM signals approach the limits of resolvable contrast. Importantly, even in these challenging distal zones, the pipeline maintains a high degree of directional consistency and reduces merge errors that can compromise downstream reconstruction<sup>48</sup>. This high precision in the primary and secondary branching structures provides the necessary confidence to proceed with the full 3D reconstruction of the 2D stacks. By preserving the integrity of the main morphological scafold, our pipeline enables volumetric representations that reflect the underlying biological structure rather than reconstruction artifacts.

## C. 3D reconstruction and comparative morphological analysis

To visually assess the reconstruction quality, we generated a 3D surface mesh from the ten times downsampled post-processed segmented vEM stack in Exp 3 (Fig. 4B) and anisotropically stretched it back to the original voxel scale. The AstroSeg-vEM-derived mesh successfully extracts the complex, hyper-ramified morphology of astrocytes, demonstrating high visual fidelity and structural congruence with the ground-truth manual annotations across both somatic regions and complex distal processes. Sholl analysis is commonly used to quantify astrocyte morphological complexity<sup>11,49–53</sup>. For evaluating whether our predicted labels preserve astrocyte morphology for downstream analysis, we performed Sholl analysis (Figs. 4C and 4D). Intersection counts were computed using three approaches with the Fiji SNT plugin<sup>54</sup>: Direct analysis of segmented images (original), analysis after autoskeletonization (thinning), and analysis after manual skeletonization (manual) in Figs. 4C and 4D. Auto-skeletonization resulted in an overestimation of mean intersections and a shift in the Sholl profile along the radial axis, likely due to spurious branches introduced by simple thinningbased algorithms. In contrast, direct mask-based analysis and manual skeletonization produced similar mean intersections and comparable Sholl plot profiles. These results indicate that the AstroSeg-vEM framework preserves astrocyte morphology and is suitable for downstream morphometric analysis.

## III. DISCUSSION

In this study, we develop AstroSeg-vEM as a lightweight, astrocyte-centric vEM reconstruction pipeline that combines transformer-based 2D segmentation with morphology-preserving post-processing. We position AstroSeg-vEM not as evidence of a general superiority of 2D over 3D segmentation, but as a substantially lighter, more memory-eficient alternative suited to accelerating native-resolution, manual vEM annotation workflows. The approach demonstrates excellent performance across diferent ultrastructure and morphological conditions, achieving high segmentation accuracy with eficient computation.

![](images/0c8682ae4e27f2af8beb1af9febdabb32192e0804c756440092ba22f4c63250b.jpg)  
FIG. 3. Quantitative robustness and pixel-level morphological fidelity of the astrocyte segmentation pipeline. A Quantitative segmentation performance. Boxplots (median and interquartile ranges) and individual section scores (scatter) summarize metrics across three experiments. B Segmentation visual comparison. Representative pixel-level results for selected sections are collected across Exp 1–3. Yellow = true positives (overlap), red = false negatives (missed), and blue = false positives (over-segmentation). Sections span a range of Dice and wDice scores and morphological regions, not only the highest-scoring examples. Panel B overlays are downsampled for display from the native 5 × 5 nm<sup>2</sup> in-plane acquisition resolution to efective in-plane pixel (px) spacings of 58.6 nm/px (D1), 29.3 nm/px (D2), 48.8 nm/px (D3), and 68.4 nm/px (D4).

Astrocytes are increasingly recognized as structurally complex partners of synapses rather than passive supporting cells. In particular, perisynaptic astrocytic processes occupy nanometer-scale domains around excitatory synapses and regulate neurotransmitter clearance, ionic homeostasis, extracellular difusion, and metabolic support<sup>5,12,47</sup>. Classical serial EM studies in hippocampal CA1 demonstrated that astrocytic processes contact only a subset of synapses and often cover only part of the synaptic perimeter, indicating that astrocyte-synapse relationships are spatially selective rather than uniform<sup>5,6</sup>. Therefore, the ultrastructural diferences detected in the present study may reflect remodeling of the local synaptic microenvironment rather than global astrocyte hypertrophy alone. By enabling 3D reconstruction of astrocytic processes under normal and stress-associated conditions,

TABLE 3. Ablation study of the post-processing pipeline for astrocyte segmentation Dice and wDice are mean ± standard deviation across sections within each held-out dataset (Exp 1 D2, n = 88; Exp 1 D3, n = 53; Exp 2 D1, n = 110; Exp 2 D4, n = 83; Exp 3 D4, n = 83), not independent biological replicates. All values are computed at the native 5 nm resolution and are not directly comparable with the 20 nm common-grid values of the 3D nnU-Net comparison (see Discussion).
<table><tr><td colspan="3">Post-processing</td><td colspan="2">Exp 1 D2</td><td rowspan="2">Exp 1 D3</td><td rowspan="2"></td><td colspan="2">Exp 2 D1</td><td colspan="2">Exp 2 D4</td><td colspan="2">Exp 3D4</td></tr><tr><td>SCR</td><td>VSC</td><td>IMC</td><td>Dice wDice</td><td>Dice</td><td>wDice</td><td>Dice</td><td>wDice</td><td>Dice</td><td>wDice</td><td>Dice wDice</td></tr><tr><td rowspan="4"></td><td rowspan="2"></td><td></td><td>0.57</td><td>0.91</td><td>0.50</td><td>0.87</td><td>0.38</td><td>0.91</td><td>0.71</td><td>0.94</td><td>0.70</td><td>0.94</td></tr><tr><td></td><td>±0.19</td><td>±0.06</td><td>± 0.13</td><td>±0.05</td><td>±0.09</td><td>±0.05</td><td>±0.02</td><td>±0.02</td><td>±0.02</td><td>±0.02</td></tr><tr><td rowspan="2"></td><td></td><td>0.72</td><td>0.90</td><td>0.52</td><td>0.87</td><td>0.43</td><td>0.89</td><td>0.71</td><td>0.92</td><td>0.70</td><td>0.93</td></tr><tr><td></td><td>±0.09</td><td>±0.07</td><td>± 0.12</td><td>±0.05</td><td>±0.10</td><td>±0.08</td><td>±0.03</td><td>±0.03</td><td>±0.02</td><td>± 0.02</td></tr><tr><td rowspan="2">√</td><td>√</td><td></td><td>0.88</td><td>0.90</td><td>0.75</td><td>0.86</td><td>0.67</td><td>0.81</td><td>0.85</td><td>0.87</td><td>0.89</td><td>0.93</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>±0.07</td><td>±0.07</td><td>±0.10</td><td>±0.05</td><td>± 0.12</td><td>±0.18</td><td>±0.05</td><td>± 0.06</td><td>±0.02</td><td>±0.03</td></tr><tr><td>√</td><td>0.90</td><td>0.91</td><td>0.76</td><td>0.88</td><td>0.68</td><td>0.82</td><td>0.87</td><td>0.89</td><td>0.92</td><td>0.95</td></tr><tr><td rowspan="2">√</td><td>√</td><td></td><td>±0.06</td><td>±0.06</td><td>±0.10</td><td>±0.05</td><td>±0.12</td><td>±0.18</td><td>±0.05</td><td>±0.06</td><td>±0.02</td><td>±0.02</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

AstroSeg-vEM provides a practical framework for quantifying astrocyte process architecture, synaptic apposition, and spatial occupancy at a resolution that cannot be achieved by light microscopy.

Note that S-AUS should not be interpreted as synonymous with molecularly defined reactive astrogliosis or as direct evidence of causal remodeling. Instead, we operationally define structural remodeling as nanoscale reorganization of astrocytic processes within the synaptic microenvironment, including changes in perisynaptic process geometry, synaptic apposition, and spatial occupancy. These features are not readily accessible using light microscopy<sup>5,11</sup>, but can be evaluated by vEM-based reconstruction. Accordingly, our results describe vEMdefined structural states characterized by altered astrocytic process organization, providing a foundation for future studies to investigate how these structural changes influence spatial relationships with neighboring dendrites. Future studies integrating vEM with molecular markers of astrocyte identity, reactivity, metabolic stress, and inflammatory signaling will be required to determine these ultrastructural states in terms of transcriptomic or immunohistochemical astrocyte subtypes.

For better structural fidelity assessment, we introduced wDice as a complementary metric whose formal definition is provided in the Methods. By placing greater emphasis on local structural agreement while reducing the influence of spatially distant artifacts, wDice is particularly informative for evaluating preservation of the main morphological structures of astrocytes. According to the ablation study (Table 3), wDice remains high (0.81–0.95) across all experimental stages and changes by at most 0.02 in Exp 1 and Exp 3, whereas it decreases in Exp 2, indicating consistent preservation of astrocytic structure, even in branches with peripheral noise. The biological relevance of wDice is further supported by the 3D reconstruction and Sholl analysis results (Fig. 4), which show that the predicted segmentations preserve the overall morphological profiles of manual annotations.

For possible condition batch confounding, all samples were processed leveraging the same solutions and processing time to minimize preparation-related variability. Although section thickness difered between datasets (70 or 100 nm), comparison of one control and one stress dataset at each thickness did not reveal an apparent thickness-dependent efect on segmentation performance, and we therefore consider section thickness unlikely to be the main source of between-dataset diferences. Nevertheless, biological variability among animals and technical variability related to sectioning and imaging cannot be completely excluded. D1 and D2 were prepared on the same day, as were D3 and D4, and the four datasets were therefore prepared in two separate batches. D3 and D4 were imaged in the same session, whereas D1 and D2 were each imaged in a separate session. No apparent batchor imaging-session-level efect was observed. Because an independent inter-annotator agreement analysis could not be performed retrospectively for the full dataset, the ground-truth annotations may contain annotator-specific uncertainty, and annotator and dataset efects cannot be completely separated in the current dataset. Finally, cross-volume robustness is limited when fewer training volumes are available. Post-processed Dice on D4 decreased from $0 . 9 2 \pm 0 . 0 2$ with three training volumes (Exp 3) to $0 . 7 1 \pm 0 . 0 4$ with two (LOVO fold 1), and all experiments rest on four volumes from four animals.

The AstroSeg-vEM framework also demonstrates high computational eficiency. Across Exp 1 and Exp 2, the per-section inference time ranged from 23.63 s to 102.47 s depending on dataset size and image dimensions (Table 4). In Exp 3, our transformer-based model maintains superior segmentation performance while achieving substantially lower inference time than U-Net and Swin-Unet on D4 (Table 2). This high throughput is achieved on consumergrade hardware (NVIDIA RTX 2060, 12 GB VRAM), and the morphology-preserving post-processing adds minimal total overhead (Table 5). By bypassing the prohibitive memory requirements of native 3D CNNs, AstroSeg-vEM eficiently leverages the rich in-plane resolution of 2D sections to recover 3D structures.

![](images/ffd6d60c5d37a299ccd6f1fabb68dd8d8a4c776c70c42d1535a1b1afef57a57c.jpg)

![](images/487ceb0f62b4db0f4fb99ecdc0ae626eb9bd3a3d38914a81c8776c4b6d3a60f1.jpg)

![](images/0791d9b8d4fe1f53f3ce9986b0c626fe41fa9aabf7e2f7d3a94e38b27e4c5c1e.jpg)

![](images/1ce35b66ff41c30412d6af84a7c3659f70ff37231687a3b30f9e9b331dd1b561.jpg)

![](images/86869095b825ab94d6a69a3fb9b1b6df262290ab2182e155ecde789359d061ab.jpg)

![](images/1693fb95a0b0c15c9f6a9a030ca2088555a9431a181db6b0e6fb37f8af55b2f7.jpg)  
FIG. 4. Morphology-preserving 3D reconstruction (downsampled ten times and anisotropically stretched back to the original voxel scale) and quantitative analysis of astrocytes using AstroSeg-vEM. AstroSeg-vEM reconstruction (left) and the ground-truth masks (right) of N-AUS D3 from Exp 1 (A) and S-AUS D4 from Exp 3 (B). C Sholl plots (left) and histograms of mean intersections (right) between manual masks and AstroSeg-vEM-predicted masks of S-AUS D4 from Exp 3 for original, thinning, and manual methods. D Visualization of Sholl analysis results with AstroSeg-vEM (top) and manual (bottom) masks for three Fiji SNT plugin approaches. Voxel spacing after stretching back to native scale: $5 \times 5 \times 1 0 0 \mathrm { n m } ^ { \bar { 3 } }$ (D3 in A, D4 in B).

At the backbone level, AstroSeg-vEM outperforms its closest competitor SegFormer by a mean paired Dice diference of only 0.021 across the 83 D4 sections (Table 2 and Fig. S3), which suggests that backbone performance is close to saturation for this task. This within-volume comparison is descriptive and makes no claim of interanimal significance. The larger gain comes from the lightweight post-processing, which raises Dice on the same volume from 0.70 to 0.92 (Table 3), while the framework processes native 5 nm sections on a consumer-grade 12 GB GPU. The practical value of AstroSeg-vEM therefore lies in this post-processing stage and eficiency–accuracy tradeof rather than in a higher backbone accuracy ceiling.

TABLE 4. Segmentation Runtime Performance on Exp 1 and Exp 2 Leveraging Transformer-based model Note: ‘Inference Time per Section’ is the wall-clock time for loading, inferring and saving a single section and ‘Total Inference Time’ is the wall-clock time of the whole evaluation run and additionally includes disk I/O and data-transfer overhead that is not attributed to individual sections.
<table><tr><td>Dataset</td><td>Number of Sections</td><td>Inference Time per Section (s)</td><td>Total Inference Time (s)</td></tr><tr><td>D1 S-AUS</td><td>110</td><td>96.18</td><td>12598.69</td></tr><tr><td>D2 N-AUS</td><td>88</td><td>23.63</td><td>2443.60</td></tr><tr><td>D3 N-AUS</td><td>53</td><td>66.55</td><td>4152.01</td></tr><tr><td>D4 S-AUS</td><td>83</td><td>102.47</td><td>10148.54</td></tr></table>

TABLE 5. Morphology-preserving Post-processing Run Time Performance on Exp 1 and Exp 2
<table><tr><td>Dataset</td><td>Number of Sections</td><td>Total Inference Time (s)</td></tr><tr><td>D1 S-AUS</td><td>110</td><td>520.81</td></tr><tr><td>D2 N-AUS</td><td>88</td><td>67.88</td></tr><tr><td>D3 N-AUS</td><td>53</td><td>111.47</td></tr><tr><td>D4 S-AUS</td><td>83</td><td>377.19</td></tr></table>

A medium-weight, 3D block-wise system is a promising alternative. In a feasibility experiment, a 3D nnU-Net trained at 20 nm in-plane resolution with four times downsampled input, under the LOVO protocol with training on D1 and D2, reached a per-section Dice of $0 . 6 9 6 \pm 0 . 0 5 5$ on D4 without post-processing, compared with $0 . 9 1 6 \pm 0 . 0 2 0$ for the complete AstroSeg-vEM pipeline on the same 20 nm grid (Methods). Because the two models were not trained on the same volumes and nnU-Net received downsampled input and had no post-processing stage, we do not attribute this diference to the architecture. The 3D run required an NVIDIA A100 data-center GPU with 40 GB of memory and about 5.1 hours for 250 training epochs, and we did not attempt a 3D pipeline at native 5 nm resolution. We regard this as a limitation of the present study and a direction for future work.

Fok et al.<sup>55</sup> recently introduced a curvature-consistency deep learning approach that accelerates astrocyte reconstruction 12-fold and enables cross-species comparison of astrocyte nanoarchitecture. It was developed for nearisotropic FIB-SEM data $( 4 \times 4 \times 8 \mathrm { n m ^ { 3 } } )$ , whereas our array-tomography data are strongly anisotropic, with 5 nm pixels and 70–100 nm sections, which motivated our 2D design. Extending such 3D approaches to anisotropic data is a promising direction for future work.

TABLE 6. Quantitative robustness and pixel-level morphological fidelity of the astrocyte segmentation pipeline Values are median and IQR across sections within each held-out dataset (Exp 1 D2, $n = 8 8 ;$ Exp 1 D3, $n = 5 3 ;$ Exp 2 D1, $n =$ 110; Exp 2 D4, $n = 8 3 ;$ Exp 3 D4, $n = 8 3 )$ , not independent biological replicates.
<table><tr><td></td><td colspan="2">Exp 1 D2</td><td colspan="2">Exp 1 D3</td><td colspan="2"> $\mathbf { E x p \textbf { 2 D 1 } }$ </td><td colspan="2"> $\mathbf { E x p \ 2 \ D 4 }$ </td><td colspan="2">Exp 3 D4</td></tr><tr><td>Metric</td><td>Median</td><td>IQR</td><td>Median</td><td>IQR</td><td>Median</td><td>IQR</td><td>Median</td><td>IQR</td><td>Median</td><td>IQR</td></tr><tr><td>Dice</td><td>0.91</td><td>0.07</td><td>0.77</td><td>0.11</td><td>0.70</td><td>0.23</td><td>0.90</td><td>0.08</td><td>0.92</td><td>0.02</td></tr><tr><td>wDice</td><td>0.92</td><td>0.07</td><td>0.88</td><td>0.07</td><td>0.94</td><td>0.36</td><td>0.93</td><td>0.10</td><td>0.96</td><td>0.02</td></tr><tr><td>Precision</td><td>0.97</td><td>0.03</td><td>0.77</td><td>0.19</td><td>0.70</td><td>0.17</td><td>0.95</td><td>0.03</td><td>0.93</td><td>0.06</td></tr><tr><td>Recall</td><td>0.86</td><td>0.13</td><td>0.79</td><td>0.11</td><td>0.89</td><td>0.50</td><td>0.87</td><td>0.16</td><td>0.93</td><td>0.05</td></tr></table>

While AstroSeg-vEM provides a high-fidelity foundation for mouse hippocampus astrocyte reconstruction, several aspects of the framework could be further improved. Further improvement in scalability for larger vEM volumes may be achieved by incorporating astrocytelocalization modules before high-resolution segmentation, thereby reducing computationally redundant processing of non-target tissue. Extending AstroSeg-vEM toward multiclass segmentation may also broaden its utility for studying astrocyte-synapse interactions in the hippocampal nano-environment. We envision that in future biological applications, analyses of process thickness distribution, synapse proximity, and organelle inclusion or contact-site quantification could further link astrocyte ultrastructural changes to functional alterations in neuron-glia communication.

## IV. METHODS

## A. Sample preparation for vEM imaging

Two controls and two with stress-associated astrocyte morphological changes identified by immunohistochemical analysis were used for electron microscopy (EM) analysis. These morphological changes were used to define the stress-associated condition and were not intended to indicate molecularly defined reactive astrogliosis. All animal experiments were conducted in accordance with the Institutional Animal Care and Use Committee (IACUC) guidelines and were approved by the IACUC under approval number IACUC-24-00067-M1. Mice were anesthetized with isoflurane and euthanized by cervical dislocation prior to tissue collection. They were subsequently perfused with 0.15 M cacodylate bufer (pH 7.4), followed by fixation with 2% paraformaldehyde (PFA) and 2.5% glutaraldehyde in the same bufer. The brains were removed and further fixed overnight at $4 ~ ^ { \circ } \mathrm { C }$ in the same fixative. The fixed brains were then sectioned into 150-µmthick hippocampal slices using a vibratome (Leica). For scanning electron microscopy (SEM), tissue slices were post-fixed with 2% osmium tetroxide $( \mathrm { O s O } _ { 4 } ;$ EMS) and

1.5% potassium ferrocyanide (Sigma) in 0.15 M cacodylate bufer for 1 h on ice. The tissues were subsequently incubated in 1% thiocarbohydrazide (TCH; TCI) for 20 min, followed by $2 \% \mathrm { O s O _ { 4 } }$ in distilled water for 30 min at room temperature. The samples were then stained with 2% uranyl acetate (EMS) overnight at 4 ${ } ^ { \circ } \mathrm { C } ,$ followed by incubation in lead aspartate solution for 30 min. The tissues were dehydrated through a graded ethanol series (20%, 50%, 70%, 90%, and 100%) on ice. Following dehydration, the samples were sequentially infiltrated with 100% acetone, mixtures of acetone and epoxy resin at ratios of 3:1, 1:1, and 1:3, and finally 100% epoxy resin. The samples were then embedded using an EMbed-812 embedding kit (EMS) according to the manufacturer’s instructions. For SEM imaging of the mouse hippocampus, serial thin sections (70 or 100 nm thick) were prepared from the resin-embedded samples using an ultramicrotome (EM UC7; Leica) and mounted on indium-tin-oxidecoated glass coverslips. Serial sections were imaged using a GeminiSEM 300 scanning electron microscope (Zeiss) equipped with Atlas 5 software at an accelerating voltage of 3.5 kV. Astrocytes in the stratum radiatum (SR) of the CA1 region were imaged at approximately 100–150 µm from the CA1 pyramidal cell layer. Images were acquired at an x–y pixel size of 5 nm. D1–D4 were obtained from four independent mouse samples. D1 and D4 were tissue blocks from the hippocampus SR region exhibiting a stress-associated astrocyte phenotype, as identified by immunohistochemical analysis, and were sectioned at 70 nm and 100 nm thickness, respectively. D2 and D3 were tissue blocks from the hippocampus SR region without this phenotype, sectioned at 70 nm and 100 nm thickness, respectively.

## B. Manual annotation procedure

vEM datasets were aligned using Fiji software TrakEM2<sup>56</sup> plug-in. The ground-truth annotations were generated using Reconstruct or PyReconstruct<sup>57</sup>. Astrocytes were identified based on characteristic ultrastructural features, including spongiform cytoplasm, the absence of synaptic inputs, and, when present, electrondense cytoplasmic filaments. Annotators then traced membrane-enclosed cellular processes continuously across serial sections to reconstruct connected astrocytic processes throughout the volume. D1 and D2 were initially annotated by one annotator, and D3 and D4 by a second annotator. All four datasets were subsequently proofread by the same primary annotator, who was diferent from the initial annotator of each dataset and who checked the continuity of astrocytic processes across consecutive sections, excluded clearly non-astrocytic profiles, corrected disconnected regions, and ensured consistency with adjacent sections. Annotations judged ambiguous were additionally reviewed jointly by multiple annotators. The finalized annotations were subsequently exported as 2D TIFF masks using custom scripts for model training and evaluation.

## C. Overview of the vEM analysis and reconstruction pipeline

Our lightweight and morphology-aware pipeline of astrocyte segmentation and reconstruction was designed as an astrocyte-centric framework to overcome the extreme spatial sparsity, complex branching, and the anisotropic resolution of vEM datasets. Instead of relying on computationally intensive 3D segmentation models, our framework leverages high-resolution 2D planar details and eficiently recovers 3D structures of astrocytes. The overall workflow consists of four integrated stages: (i) Foregroundaware patch sampling to address class imbalance (Fig. 2A); (ii) Transformer-based 2D segmentation (Fig. 2B) with sliding-window inference (Fig. 2C) to extract astrocyte ultrastructures; (iii) Morphology-preserving 2D-to-3D hybrid post-processing (Fig. 2D) to restore inter-section continuity and ensure volumetric coherence; (iv) Mesh reconstruction and Sholl analysis results (Fig. 4).

## D. Foreground-aware patch sampling from vEM data

Foreground of astrocyte vEM data occupies only a small fraction of the tissue volume and appears as sparse, spatially fragmented features in 2D sections, resulting in severe class imbalance and background-dominated sampling. This challenge is further exacerbated by anisotropic voxel spacing. To address this, we employ a foreground-aware patch sampling strategy illustrated in Fig. 2A. Fixed-size patches (224 × 224 pixels) are extracted from each 2D section, with patch centers sampled using a probabilistic mixture: 30% from foreground pixels, and 70% from the entire section. We also apply boundary-aware filtering to improve sample quality. Patches with more than 10% low-intensity background are discarded, and patches consisting entirely of foreground or background are largely excluded, with 5% retained to preserve extreme cases. This astrocyte-centric sampling strategy increases exposure to rare foreground and boundary regions where thin processes are most likely to be missed by random patch extraction.

## E. Transformer-based astrocyte segmentation

Astrocyte segmentation is performed using a transformer-based U-Net architecture with a MiT-B2 (Mix Transformer) encoder (Fig. 2B) originally introduced in SegFormer, which was initialized with ImageNet pretrained weights. The MiT encoder utilizes overlapping patch embeddings and a hierarchical multi-stage design, enabling progressive feature abstraction while maintaining spatial coherence. Its multi-scale representations capture both coarse somatic regions and fine distal astrocytic processes, facilitating segmentation of sparse and morphologically complex structures. Self-attention modules in transformer blocks model long-range spatial dependencies, integrating spatially distributed ultrastructure beyond the local convolutional receptive field. To improve computational eficiency on large vEM sections, attention is implemented with spatial reduction of key and value features in the first three stages. Subsequently, a U-Net– based decoder with skip connections and squeeze-andexcitation attention modules refines high-resolution features, improving boundary delineation of thin astrocytic processes.

Model training is performed on $2 2 4 \times 2 2 4$ patches extracted from individual vEM sections. Given the strong spatial continuity of astrocyte morphology along the zaxis, particular care is taken to avoid information leakage between training and validation data. Sections within each dataset are sequentially sorted and divided into 80% training and 20% validation subsets, ensuring that all patches from a given section were assigned exclusively to one split. This partition was applied only during training designated for each experiment (Table 1). Grayscale images are replicated across three channels to match the transformer encoder (Fig. 2B), normalized using fixed mean and standard deviation values of [0.5, 0.5, 0.5], and converted to PyTorch tensors. No additional random geometric or intensity data augmentation was applied during training.

For scalable inference on high-resolution vEM sections, we adopt an overlapping sliding-window strategy (Fig. 2C). Each vEM section $\breve { I _ { z } } \in \mathbb { R } ^ { X \times Y }$ is partitioned into 224 $\times ~ 2 2 4$ patches with 50% overlap in both horizontal (x) and vertical (y) directions. As each pixel $( x , y )$ is covered by multiple patches, the aggregated probability map $P _ { z , c } ( x , y )$ is computed as:

$$
P _ { z , c } ( x , y ) = \frac { 1 } { N ( x , y ) } \sum _ { k = 1 } ^ { N ( x , y ) } p _ { k , c } ( x , y ) ,\tag{1}
$$

where $p _ { k , c } ( x , y )$ denotes the predicted probability for class c from patch k and $N ( x , y )$ is the number of overlapping patches covering pixel (x, y). The final segmenta-

tion mask $\hat { S } _ { z } \in \{ 0 , 1 \} ^ { X \times Y }$ is obtained by performing an argmax operation over the class dimension $( { \mathrm { i . e . } }$ , selecting the index c that yields the highest aggregated probability)

$$
\hat { S } _ { z } ( x , y ) = \mathop { \mathrm { a r g m a x } } _ { c } P _ { z , c } ( x , y ) .\tag{2}
$$

This aggregation reduces boundary artifacts and preserves spatial continuity, while all predictions are performed at native resolution without rescaling. In the binary setting used in this paper, the maximum-probability assignment is equivalent to applying a 0.5 decision threshold to the foreground probability after softmax aggregation.

The training and inference are implemented in the $\mathrm { P y }$ Torch framework and executed on a single NVIDIA RTX 2060 GPU (12 GB VRAM). To accelerate the computational pipeline eficiently and optimize memory utilization, automatic mixed precision (AMP) is employed during training. Model parameters are optimized using the AdamW optimizer with an initial learning rate of $6 \times 1 0 ^ { - 5 }$ and a weight decay of $1 \times 1 0 ^ { - 4 }$ . The network is trained for up to 50 epochs with a batch size of 64. To prevent overfitting, an early stopping mechanism with a patience of 5 epochs is applied based on the validation Dice score. To address class imbalance, training is supervised with a hybrid loss comprising equally weighted cross-entropy and Dice losses. All experiments were performed using a fixed random seed (2025) leveraging Python, NumPy, PyTorch, CUDA, and DataLoader workers.

The 2D segmentation experiments (Exp 1–3) used Python 3.11.14, PyTorch 2.5.1, torchvision 0.20.1, segmentation-models-pytorch 0.5.0, timm 1.0.24 and CUDA 12.1 on an NVIDIA RTX 2060 GPU with 12 GB of memory. The 2-fold and leave-one-volume-out models in Supplementary Table S1 used Python 3.11.5, PyTorch 2.13.0, torchvision 0.28.0, segmentation-models-pytorch 0.3.4, timm 0.9.7 and CUDA 13.0 on an NVIDIA A100- SXM4-40GB GPU of the Narval cluster of the Digital Research Alliance of Canada. The 3D nnU-Net comparison used nnunetv2 2.5.1 with Python 3.11.5, PyTorch 2.5.1, torchvision 0.20.1 and CUDA 13.0 on the same type of GPU. Manual annotation used Fiji (version 1.54t) with the TrakEM2 plugin, Reconstruct (version 1.1.0.0) and PyReconstruct (version 1.19.0), and Sholl analysis used the Fiji SNT plugin (version 5.0.2).

## F. Morphology-preserving post-processing

Post-processing is a critical step for astrocyte segmentation due to the complexity of astrocyte morphological features and the reconstruction failure modes associated with thin, spatially fragmented processes. Although transformer-based 2D segmentation produces spatially coherent probabilistic maps, section-by-section predictions often inevitably contain fragmented components, spurious foreground regions, and discontinuities that hinder accurate 3D reconstruction. To enforce structural consistency across sections and preserve biologically meaningful astrocyte morphology, we implement an eficient, multi-stage, morphology-aware post-processing pipeline (Fig. 2D) that operates collaboratively in both the 2D and 3D domains.

Soma-guided component refinement: Section-wise segmentation lacks explicit 3D contextual information, leading to inconsistent false positives. For each section $I _ { z }$ we extract a set of foreground connected components $\{ C _ { z , i } \} _ { i = 1 } ^ { N _ { z } }$ from the predicted mask $\hat { S } _ { z }$ . Each component $C _ { z , i }$ is characterized by its area $A _ { z , i }$ and centroid $\mathbf { r } _ { z , i } = ( x _ { z , i } , \ y _ { z , i } )$ . To capture their cross-section distribution of these foreground components in physical space, centroids are embedded into 3D space as:

$$
\mathbf { q } _ { z , i } = ( x _ { z , i } \bullet s _ { x } , y _ { z , i } \bullet s _ { y } , z \bullet s _ { z } ) ,\tag{3}
$$

where $s _ { x } , s _ { y }$ and $s _ { z }$ denote anisotropic voxel spacing along their respective axes. To account for size diferences, each point is replicated proportionally to $\sqrt { A _ { z , i } }$ $\mathrm { H D B S C A N ^ { 5 8 } }$ clustering is then applied to identify spatially coherent components, and the cluster with the largest cumulative area is selected as the soma cluster. The actual soma-associated component in section $I _ { z }$ is identified as the foreground component situated closest to this reference coordinate. Let $\mathbf { r } _ { z } ^ { \mathrm { s o m a } }$ and $A _ { z } ^ { \mathrm { s o m a } }$ represent its centroid and area, respectively. The Euclidean distance to the estimated soma center is computed as

$$
\begin{array} { r } { d _ { z , i } = \| \mathbf { r } _ { z , i } - \mathbf { r } _ { z } ^ { \mathrm { s o m a } } \| _ { 2 } . } \end{array}\tag{4}
$$

And components are removed only if they satisfy both size and spatial distance criteria:

$$
A _ { z , i } \geq \alpha A _ { z } ^ { \mathrm { s o m a } } , \qquad d _ { z , i } > \beta D _ { z } ,\tag{5}
$$

where $D _ { z }$ is the diagonal length of section $I _ { z } ,$ and α and $\beta$ are relative thresholds for area and distance, respectively. This soma-guided refinement suppresses large, spatially inconsistent connected components arising from the absence of volumetric context, while preserving somacentered structures and their associated distal processes. By enforcing joint size and spatial distance constraints within a soma-centered reference frame, the procedure enhances volumetric coherence without introducing artificial morphological alterations.

Volumetric structural consolidation: While somaguided component refinement enforces cross-sectional consistency and removes prominent outliers, isolated false predictions may persist due to the inherently limited spatial context of section-wise segmentation. To further ensure global structural consistency, we perform volumetric connectivity analysis on the refined vEM stack. The segmentation is treated as a 3D binary volume, and unsupported single-section artifacts are first suppressed based on their lack of continuity across adjacent sections. Connected components are then identified in voxel space under 26-connectivity, and the k components with the largest volumetric support are retained as the principal astrocytic structures. This volumetric consolidation step removes residual disconnected fragments and segmentation artifacts that lack axial continuity and enforces global morphological coherence across the reconstructed astrocyte volume.

In-plane morphological correction: The preceding somaguided component refinement and volumetric consolidation efectively suppress spatially inconsistent false positives and noise, making the remaining cross-sectional masks highly confident and suficiently clean. To further improve morphological plausibility, a minimal in-plane morphological operation is applied following volumetric consolidation. Specifically, a standard 2D binary hole-filling algorithm is independently employed on each section to eliminate small, artificial internal cavities caused by local segmentation inconsistencies. This final refinement ensures the solid topological structure of the reconstructed astrocyte without altering its external boundaries and distal processes.

## G. 3D reconstruction and morphology analysis

To facilitate high-fidelity visualization and qualitative structural assessment, the marching cubes algorithm<sup>59</sup> implemented by mcubes (PyMCubes) library<sup>60</sup> is applied to the stacked and post-processed 2D vEM sections, converting the resulting 3D volume into a continuous geometric surface model. This reconstruction process employs isosurface extraction to transform volumetric stacks of 2D binary masks into triangular mesh representations. With an intensity threshold of 0.5, a high-resolution geometric model was generated to capture the complex branching morphology of astrocytes. To control the computational complexity of grid generation for large-scale vEM datasets, we directly implement isosurface extraction based on the post-processed 2D stack (downsampled ten times on the xy-plane). The resulting surfaces are further processed and represented using PyVista<sup>61</sup>. To preserve morphological fidelity, a stretch-and-scale transformation is applied to mesh vertices to compensate for anisotropic voxel resolution and any sampling factors introduced during segmentation.

## H. Evaluation metrics

Dice, precision and recall are applied to quantitatively evaluate 2D segmentation performance. While these metrics provide a general evaluation of spatial overlap in plane, they exhibit two critical limitations in the context of astrocyte vEM segmentation. First, their volume-centric nature causes the large soma to dominate the score, potentially masking errors in thin astrocytic processes.

Second, all false positives are penalized uniformly, despite distant artifacts being easily removed during postprocessing.

To address this, we introduce a weighted Dice (wDice) as an astrocyte-centric complement to conventional Dice. Let $G$ and $\check { P }$ represent the binary ground-truth and predicted masks, respectively. We first compute the Euclidean distance transform map $D ( x , y )$ , which measures the shortest distance from any spatial location (x, y) to the nearest foreground pixel in G. A spatial weight map $W ( x , y )$ is then defined using an exponential decay function:

$$
W ( x , y ) = e ^ { - \gamma \cdot D ( x , y ) } ,\tag{6}
$$

where $\gamma$ is the distance decay factor empirically set to 0.1. The wDice is subsequently defined as:

$$
w \mathrm { D i c e } = \frac { 2 \sum _ { x , y } W ( x , y ) P ( x , y ) G ( x , y ) } { \sum _ { x , y } W ( x , y ) P ( x , y ) + \sum _ { x , y } W ( x , y ) G ( x , y ) } ,\tag{7}
$$

which strategically ignores distant false positives while heavily penalizing false positives that distribute near the true astrocyte boundaries.

For the per-section error decomposition, we additionally report the false discovery rate (FDR), FDR = false positives / (false positives + true positives) = 1 − preci sion, and the miss rate = false negatives / (false negatives + true positives) = 1 − recall, with all counts taken as pixels within a section. In comparison with a 3D nnU-Net (Discussion), predictions and ground truth were evaluated on a common grid of 20 nm in-plane resolution. The native 5 nm ground-truth masks and final AstroSeg-vEM predictions from the complete pipeline were downsampled fourfold with a block-OR operation, whereby each 20 nm pixel was assigned as foreground if any of its 4 × 4 native pixels was foreground. This is the same operation applied to build the nnU-Net training data. The nnU-Net predictions are natively at 20 nm, and both models are scored against the same downsampled ground truth on all 83 D4 sections. Values on this grid are therefore not comparable with the native 5 nm values in Tables 2 and 3.

## I. Statistics and reproducibility

Unless otherwise noted, n refers to the number of 2D sections evaluated within a single held-out volume per experimental condition, not independent volumes or animals. These sections are not independent biological replicates, as they derive from the same animal and tissue volume.

## DATA AVAILABILITY

The datasets generated and/or analyzed during the current study are not publicly available due to institutional

policies but are available from the corresponding author on reasonable request.

## CODE AVAILABILITY

Source code is available under the GNU General Public License v3.0 (GPL-3.0) at https://github.com/ QuIN-Lab/AstroSeg-vEM.

## REFERENCES

[1] Kol, A. et al. Astrocytes contribute to remote memory formation by modulating hippocampal–cortical communication during learning. Nat. Neurosci. 23, 1229–1239 (2020).

[2] Zhou, Z. et al. Astrocytic cAMP modulates memory via synaptic plasticity. Proc. Natl. Acad. Sci. 118, e2016584118 (2021).

[3] Wang, Y., Fu, A. K. Y. & Ip, N. Y. Instructive roles of astrocytes in hippocampal synaptic plasticity: neuronal activity-dependent regulatory mechanisms. FEBS J. 289, 2202–2218 (2022).

[4] Octeau, J. C. et al. An Optical Neuron-Astrocyte Proximity Assay at Synaptic Distance Scales. Neuron 98, 49–66.e9 (2018).

[5] Ventura, R. & Harris, K. M. Three-dimensional relationships between hippocampal synapses and astrocytes. J. Neurosci. 19, 6897–6906 (1999).

[6] Witcher, M. R., Kirov, S. A. & Harris, K. M. Plasticity of perisynaptic astroglia during synaptogenesis in the mature rat hippocampus. Glia 55, 13–23 (2007).

[7] Kim, H. Y., Kim, S., Akaydin, A. N. et al. The rise of astrocytes: are they guardians or troublemakers of the brain disorder? Exp. Mol. Med. 58, 301–318 (2026).

[8] Bushong, E. A., Martone, M. E., Jones, Y. Z. & Ellisman, M. H. Protoplasmic Astrocytes in CA1 Stratum Radiatum Occupy Separate Anatomical Domains. J. Neurosci. 22, 183–192 (2002).

[9] Verkhratsky, A. & Nedergaard, M. Physiology of Astroglia. Physiol. Rev. 98, 239–389 (2018).

[10] Verkhratsky, A., Butt, A., Li, B. et al. Astrocytes in human central nervous system diseases: a frontier for new therapies. Signal Transduct. Target. Ther. 8, 396 (2023).

[11] Baldwin, K. T., Murai, K. K. & Khakh, B. S. Astrocyte morphology. Trends Cell Biol. 34, 547–565 (2024).

[12] Won, W., Bhalla, M., Lee, J.-H. & Lee, C. J. Astrocytes as Key Regulators of Neural Signaling in Health and Disease. Annu. Rev. Neurosci. 48, 251–276 (2025).

[13] Salmon, C. K. et al. Organizing principles of astrocytic nanoarchitecture in the mouse cerebral cortex. Curr. Biol. 33, 957–972.e5 (2023).

[14] Schertel, A. et al. Cryo FIB-SEM: Volume imaging of cellular ultrastructure in native frozen specimens. J. Struct. Biol. 184, 355–360 (2013).

[15] Špaček, J. Three-dimensional reconstructions of astroglia and oligodendroglia cells. Z. Zellforsch. 112, 430–442 (1970).

[16] Witcher, M. R. et al. Three-dimensional relationships between perisynaptic astroglia and human hippocampal synapses. Glia 58, 572–587 (2010).

[17] Aten, S. et al. Ultrastructural view of astrocyte arborization, astrocyte-astrocyte and astrocyte-synapse contacts, intracellular vesicle-like structures, and mitochondrial network. Prog. Neurobiol. 213, 102264 (2022).

[18] Kikuchi, T. et al. Volume Electron Microscopy Study of the Relationship Between Synapses and Astrocytes in the Developing Rat Somatosensory Cortex. Cereb. Cortex 30, 3800–3819 (2020).

[19] Chiappini, V. et al. Ultrastructure of astrocytes using volume electron microscopy: A scoping review. J. Physiol. 604, 1498–1517 (2026).

[20] Shapson-Coe, A. et al. A petavoxel fragment of human cerebral cortex reconstructed at nanoscale resolution. Science 384, eadk4858 (2024).

[21] Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T. & Ronneberger, O. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016 (eds Ourselin, S., Joskowicz, L., Sabuncu, M. R., Unal, G. & Wells, W.) vol. 9901 424–432 (Springer International Publishing, Cham, 2016).

[22] Januszewski, M. et al. High-precision automated reconstruction of neurons with flood-filling networks. Nat. Methods 15, 605–610 (2018).

[23] Lin, Z., Wei, D., Lichtman, J. & Pfister, H. PyTorch Connectomics: A Scalable and Flexible Segmentation Framework for EM Connectomics. Preprint at https: //doi.org/10.48550/arXiv.2112.05754 (2021).

[24] Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J. & Maier-Hein, K. H. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat. Methods 18, 203–211 (2021).

[25] Yener, Y., Motta, A. & Helmstaedter, M. Connectomic analysis of astrocyte-synapse interactions in the cerebral cortex. bioRxiv 2025.02.20.639274 (2025).

[26] Dorkenwald, S., Li, P. H., Januszewski, M. et al. Multilayered maps of neuropil with segmentation-guided contrastive learning. Nat. Methods 20, 2011–2020 (2023).

[27] Sheridan, A. et al. Local shape descriptors for neuron segmentation. Nat. Methods 20, 295–303 (2023).

[28] Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015 (eds Navab, N., Hornegger, J., Wells, W. M. & Frangi, A. F.) vol. 9351 234–241 (Springer International Publishing, Cham, 2015).

[29] Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N. & Liang, J. UNet++: A Nested U-Net Architecture for Medical Image Segmentation. in Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (eds Stoyanov, D. et al.) 3–11 (Springer International Publishing, Cham, 2018).

[30] Oktay, O. et al. Attention U-Net: Learning Where to Look for the Pancreas. Preprint at https://doi.org/10. 48550/arXiv.1804.03999 (2018).

[31] Alom, M. Z., Hasan, M., Yakopcic, C., Taha, T. M. & Asari, V. K. Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation. Preprint at https://doi.org/10.48550/ arXiv.1802.06955 (2018).

[32] Cao, Y., Liu, S., Peng, Y. & Li, J. DenseUNet: densely connected UNet for electron microscopy image segmentation. IET Image Process. 14, 2682–2689 (2020).

[33] Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K. & Yuille, A. L. Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs. Preprint at https://doi.org/10.48550/arXiv. 1412.7062 (2016).

[34] Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K. & Yuille, A. L. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE Trans. Pattern Anal. Mach. Intell. 40, 834–848 (2018).

[35] Chen, L.-C., Papandreou, G., Schrof, F. & Adam, H. Rethinking Atrous Convolution for Semantic Image Segmentation. Preprint at https://doi.org/10.48550/arXiv. 1706.05587 (2017).

[36] Chen, L.-C., Zhu, Y., Papandreou, G., Schrof, F. & Adam, H. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. in Computer Vision – ECCV 2018 (eds Ferrari, V., Hebert, M., Sminchisescu, C. & Weiss, Y.) vol. 11211 833–851 (Springer International Publishing, Cham, 2018).

[37] Cao, H. et al. Swin-Unet: Unet-Like Pure Transformer for Medical Image Segmentation. in Computer Vision – ECCV 2022 Workshops (eds Karlinsky, L., Michaeli, T. & Nishino, K.) vol. 13803 205–218 (Springer Nature Switzerland, Cham, 2023).

[38] Zheng, S. et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition 6877–6886 (2021).

[39] Xie, E. et al. SegFormer: Simple and eficient design for semantic segmentation with transformers. Adv. Neural Inf. Process. Syst. 34, 12077–12090 (2021).

[40] Chen, J. et al. TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. Preprint at https://doi.org/10.48550/arXiv.2102.04306 (2021).

[41] Wang, L. et al. UNetFormer: A UNet-like transformer for eficient semantic segmentation of remote sensing urban scene imagery. ISPRS J. Photogramm. Remote Sens. 190, 196–214 (2022).

[42] Xing, C., Xie, R. & Bader, G. D. RETINA: Reconstruction-based pre-trained enhanced TransUNet for electron microscopy segmentation on the CEM500K dataset. PLOS Comput. Biol. 21, e1013115 (2025).

[43] Ciresan, D., Giusti, A., Gambardella, L. & Schmidhuber, J. Deep neural networks segment neuronal membranes in electron microscopy images. Adv. Neural Inf. Process. Syst. 25 (2012).

[44] Khadangi, A., Boudier, T. & Rajagopal, V. EM-net: Deep learning for electron microscopy image segmentation. in 2020 25th international conference on pattern recognition (ICPR) 31–38 (IEEE, 2021).

[45] Belevich, I. & Jokitalo, E. DeepMIB: User-friendly and open-source software for training of deep learning network for biological image segmentation. PLoS Comput. Biol. 17, e1008374 (2021).

[46] Shah, A. et al. Automated image segmentation of scanning electron microscopy images of graphene using U-Net Neural Network. Mater. Today Commun. 35, 106127 (2023).

[47] Nam, A. J. et al. Perisynaptic astroglial response to in vivo long-term potentiation and concurrent long-term depression in the hippocampal dentate gyrus. J. Neurosci. 45, e0943252025 (2025).

[48] Li, H., Januszewski, M., Jain, V. & Li, P. H. Neuronal Subcompartment Classification and Merge Error Correc-

tion. in Medical Image Computing and Computer Assisted Intervention – MICCAI 2020 (eds Martel, A. L. et al.) 88–98 (Springer International Publishing, Cham, 2020).

[49] Binley, K. E., Ng, W. S., Tribble, J. R., Song, B. & Morgan, J. E. Sholl analysis: a quantitative comparison of semi-automated methods. J. Neurosci. Methods 225, 65–70 (2014).

[50] Zhang, A. Q. et al. Acute systemic inflammation induces region-specific morphological remodeling of astrocytes and microglia concurrent with depression-like behavior. Histochem. Cell Biol. 163, 112 (2025).

[51] Augustin, E. et al. Hippocampal Astrocyte Morphology Follows an Unexpected Trajectory with Age in a Transgenic Rodent Model of Tauopathy. Glia 73, 1502–1519 (2025).

[52] Ortiz-Valladares, M. et al. Long-term intermittent fasting induces region- and sex-specific changes in astrocyte morphology and expression and anxiety-like behaviors in CD-1 mice. Behav. Brain Res. 494, 115745 (2025).

[53] Viana, J. F. et al. Astrocyte structural heterogeneity in the mouse hippocampus. Glia 71, 1667–1682 (2023).

[54] Longair, M. H., Baker, D. A. & Armstrong, J. D. Simple Neurite Tracer: open source software for reconstruction, visualization and analysis of neuronal processes. Bioinformatics 27, 2453–2454 (2011).

[55] Fok, A. H. K. et al. Deep learning segmentation with curvature consistency reveals astrocyte nanostructure across species. Cell Rep. 45, 117725 (2026).

[56] Cardona, A., Saalfeld, S., Schindelin, J., Arganda-Carreras, I., Preibisch, S., Longair, M., Tomancak, P., Hartenstein, V. & Douglas, R. J. TrakEM2 software for neural circuit reconstruction. PLoS One 7, e38011 (2012).

[57] Chirillo, M. A., Falco, J. N., Musslewhite, M. D., Lindsey, L. F. & Harris, K. M. PyReconstruct: A fully open-source, collaborative successor to Reconstruct. Proc. Natl. Acad. Sci. 122, e2505822122 (2025).

[58] McInnes, L., Healy, J. & Astels, S. hdbscan: Hierarchical density based clustering. J. Open Source Softw. 2, 205 (2017).

[59] Lorensen, W. E. & Cline, H. E. Marching cubes: A high resolution 3D surface construction algorithm. ACM SIG-GRAPH Comput. Graph. 21, 163–169 (1987).

[60] Neila, P. M. PyMCubes: Marching cubes for Python. GitHub repository (2024).

[61] Sullivan, B. & Kaszynski, A. PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). J. Open Source Softw. 4, 1450 (2019).

## ACKNOWLEDGEMENTS

B.D. and N.Y.K. acknowledge the support from Industry Canada and Natural Sciences and Engineering Research Council of Canada (Discovery Grant RGPIN-2025-04474). G.J.J., S.N., and J.Y.M. are supported by the KBRI basic research program through the Korea Brain Research Institute, funded by the Ministry of Science and ICT (26-BR-01-03, 26-BR-07-01) and the National Research Foundation of Korea (NRF, RS-2024-00419699).

# AUTHOR INFORMATION

## Contributions

N.Y.K. and J.Y.M. conceived the idea, designed the research direction, and supervised the project. B.D. developed an automatic deep learning-based volume electron microscopy segmentation and reconstruction pipeline and performed related benchmark experiments. G.J.J. performed manual image analysis and S.N. acquired the electron microscopy images. B.D., J.Y.M., and N.Y.K. wrote the manuscript. All authors discussed the results and contributed to the paper.

## Corresponding author

Correspondence and requests for materials should be addressed to: Na Young Kim (nayoung.kim@uwaterloo.ca, 0000-0002-0290-7728), B. Deng (b28deng@uwaterloo.ca, 0000-0001-9625-829X), J.Y. Mun (jymun@kbri.re.kr, 0000-0003-0820-6233).

## ETHICS DECLARATIONS

## Competing interests

The authors declare no competing financial or nonfinancial interests.

Dice = 0.4436, w���� = 0.4943

# Supplementary Material

Lightweight Deep Learning Framework for Astrocyte-Centric 3D Reconstruction across Normal and Stress-associated Ultrastructural States

Bowen Deng<sup>1</sup>, Gwang-Jin Jeong<sup>2,3</sup>, Seulgi Noh<sup>2</sup>, Ji Young Mun<sup>2</sup> & Na Young Kim<sup>1,4</sup>

![](images/7b3bd07c6a0222972df4c05068e44debeb9b59b2ad46bab7ee77fb0acc1f7d07.jpg)

B  
![](images/3511be06a14632784a0e22222113404578190573432985de918f96174cf77e3c.jpg)

C  
![](images/c5120f1b0769b8f5ed3f0f2685865cee87501b5bd455dfd78b9637bff6604e83.jpg)

![](images/26a37cc95d50b6dd4180053a9fd1295082111daaa785d08d4bcfd9e46458e701.jpg)

![](images/fbd6b92096e96238c3023f17e2a6b3c8f5cb5c1edb73f41b0aad1b3e7d76adba.jpg)  
Dice = 0.4773, w���� = 0.4994

![](images/1dfff9a28c518015dd71b03ce932475914fe6ba0acad0f39f035e1967732d041.jpg)  
Dice = 0.4181, w���� = 0.4707

D  
![](images/45aa7ed0c812e1da651e962d3fd076e3e84c7cff1a8b98c5475895b1b3e83d2d.jpg)  
Dice = 0.9399, w���� = 0.9577

![](images/522005c9dcdc18142a9a60a37f4882f699a5651e1e39f2f3436381adb4e5a782.jpg)  
Dice = 0.9166, w���� = 0.9429

![](images/c68fddbbf08c9b4d97aaf5950ed832f713df98c82d49d45823094e9d13b8caa0.jpg)  
Dice = 0.9472, w���� = 0.9590

![](images/32b1234d952846063dd54b23708a0b19195e5c4fc970e6880699359c9fabff53.jpg)  
Dice = 0.8338, w���� = 0.8407

FIG. S1. Comparison of per-section segmentation quality before and after post-processing across the D1 and D4 volumes. A Per-section raw (before post-processing) and final (post-processed) wDice across the D1 volume. B Per-section raw and final wDice across the D4 volume, shown for comparison. C Overlays for the four post-processed outlier sections from D1. D Overlays for four representative sections from D4. Yellow = correctly predicted overlap, red = mask missed by the prediction, and blue = false positive prediction.

A  
![](images/710608da7ca3f7bc1cd2cdaddfa0d426a5c419a2e4490c35b4a332fee0b832c0.jpg)

B  
![](images/a7b0871ff2f8afa1d10f25b42315a5400a1d2be393c2705bca3e0aef920e4fb6.jpg)  
FIG. S2. Post-processing error decomposition into false discovery rate and miss rate across experiments and datasets. A False discovery rate (FDR = FP / (FP+TP) = 1 − precision) before and after post-processing for each experiment and test dataset. B Miss rate (FN $/ \ ( \mathrm { F N + T P } ) \stackrel { } { = } 1 - \mathrm { r e c a l l } )$ before and after post-processing for each experiment and test dataset. The two rates have diferent denominators (predicted-positive and ground-truth-positive pixels) and are shown as complementary measures; bars show the mean across the sections of each held-out volume. TP, FP and FN are the numbers of true-positive, false-positive and false-negative pixels in a section.

A  
![](images/665e996f68b0e977810b8157f85bb8509d837368bffb512c42df4a1a58f00363.jpg)  
B

![](images/eb4f326f9ebd218d442d258c175a9904452e99d3d1f49f5f19bf93eb72c23cf1.jpg)  
FIG. S3. Paired, per-section comparison of AstroSeg-vEM and SegFormer Dice on Exp 3 D4. A Per-section paired Dice (n = 83 sections, same D4 volume evaluated by both models). Points above the diagonal are sections where AstroSeg-vEM scores higher. B Distribution of per-section paired diferences (mean paired diference +0.021, moving-block bootstrap 95% confidence interval 0.018–0.024 (block length 5–10 sections; 0.018–0.023 if serial dependence is ignored); positive in 81 of 83 sections; lag-1 autocorrelation of the paired diferences 0.26, AR(1) efective sample size ≈ 49). This is a descriptive comparison within one volume; the sections are not independent replicates.

TABLE S1. Segmentation metrics before and after post-processing for all data-split protocols Exp 1–3 are the main-text experiments, trained on the first 80% and validated on the last 20% of the sections of the listed volumes, whereas the other protocols use fewer training volumes. Values are mean ± standard deviation across the sections of the held-out test volume. Post-processed values for 2-fold-1 D2 vary strongly across sections.
<table><tr><td colspan="5"></td><td colspan="2">Raw</td><td colspan="2">Post-processed</td></tr><tr><td>Protocol</td><td>Training</td><td>Validation</td><td>Test set</td><td>n</td><td>Dice</td><td>wDice</td><td>Dice</td><td>wDice</td></tr><tr><td rowspan="2">Exp 1</td><td rowspan="2">D1, D4 (first 80%) D1, D4 (last 20%)</td><td rowspan="2"></td><td>D2 (N-AUS)</td><td>88</td><td> $0 . 5 7 \pm 0 . 1 9$ </td><td> $0 . 9 1 \pm 0 . 0 6$ </td><td> $0 . 9 0 \pm 0 . 0 6$ </td><td> $0 . 9 1 \pm 0 . 0 6$ </td></tr><tr><td>D3 (N-AUS)</td><td>53</td><td> $0 . 5 0 \pm 0 . 1 3$ </td><td> $0 . 8 7 \pm 0 . 0 5$ </td><td> $0 . 7 6 \pm 0 . 1 0$ </td><td> $0 . 8 8 \pm 0 . 0 5$ </td></tr><tr><td>Exp 2</td><td></td><td>D2, D3 (first 80%) D2, D3 (last 20%)</td><td>D1 (S-AUS)</td><td>110</td><td> $0 . 3 8 \pm 0 . 0 9$ </td><td> $0 . 9 1 \pm 0 . 0 5$ </td><td> $0 . 6 8 \pm 0 . 1 2$ </td><td> $0 . 8 2 \pm 0 . 1 8$ </td></tr><tr><td></td><td></td><td></td><td>D4 (S-AUS)</td><td>83</td><td> $0 . 7 1 \pm 0 . 0 2$ </td><td> $0 . 9 4 \pm 0 . 0 2$ </td><td> $0 . 8 7 \pm 0 . 0 5$ </td><td> $0 . 8 9 \pm 0 . 0 6$ </td></tr><tr><td>Exp 3</td><td>D1–D3 (first 80%) D1–D3 (last 20%)</td><td></td><td>D4 (S-AUS)</td><td>83</td><td> $0 . 7 0 \pm 0 . 0 2$ </td><td> $0 . 9 4 \pm 0 . 0 2$ </td><td> $0 . 9 2 \pm 0 . 0 2$ </td><td> $0 . 9 5 \pm 0 . 0 2$ </td></tr><tr><td rowspan="2">2-fold-1</td><td>D1</td><td>D4</td><td>D2 (N-AUS)</td><td>88</td><td> $0 . 5 3 \pm 0 . 1 8$ </td><td> $0 . 9 4 \pm 0 . 0 5$ </td><td> $0 . 4 2 \pm 0 . 4 3$ </td><td> $0 . 4 7 \pm 0 . 4 3$ </td></tr><tr><td></td><td></td><td>D3 (N-AUS)</td><td>53</td><td> $0 . 2 0 \pm 0 . 0 7$ </td><td> $0 . 7 1 \pm 0 . 1 0$ </td><td> $0 . 4 6 \pm 0 . 1 0$ </td><td> $0 . 7 1 \pm 0 . 1 2$ </td></tr><tr><td rowspan="2">2-fold-2</td><td>D2</td><td>D3</td><td>D1 (S-AUS)</td><td>110</td><td> $0 . 3 6 \pm 0 . 1 0$ </td><td>0.92 ± 0.04</td><td> $0 . 7 0 \pm 0 . 0 6$ </td><td> $0 . 8 9 \pm 0 . 0 9$ </td></tr><tr><td></td><td></td><td>D4 (S-AUS)</td><td>83</td><td> $0 . 6 4 \pm 0 . 0 3$ </td><td>0.91 ± 0.02</td><td> $0 . 8 9 \pm 0 . 0 6$ </td><td> $0 . 9 1 \pm 0 . 0 6$ </td></tr><tr><td>LOVO fold 1 D1, D2</td><td></td><td>D3</td><td>D4 (S-AUS)</td><td>83</td><td> $0 . 5 8 \pm 0 . 0 4$ </td><td> $0 . 9 5 \pm 0 . 0 2$ </td><td> $0 . 7 1 \pm 0 . 0 4$ </td><td> $0 . 9 2 \pm 0 . 0 5$ </td></tr></table>

TABLE S2. Connected-component analysis of the Exp 3 D4 predictions before and after post-processing on the common 20 nm grid Spurious components overlap no ground truth, missed components are ground-truth components without an overlapping prediction, and split and merged counts are ground-truth components covered by several predictions and predictions covering several ground-truth components. Values are mean ± standard deviation across the 83 D4 sections.
<table><tr><td>Section set</td><td>Connected components per section</td><td>Spurious components</td><td>Missed GT components components</td><td>Split GT</td><td>Merged predicted components</td></tr><tr><td>Ground truth</td><td> $2 9 . 1 \pm 2 . 7$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Raw prediction</td><td> $6 5 5 . 7 \pm 3 9 . 1$ </td><td> $5 9 9 . 9 \pm 3 7 . 6$ </td><td> $0 . 9 \pm 1 . 0$ </td><td> $1 0 . 0 \pm 2 . 0$ </td><td> $0 . 5 \pm 0 . 7$ </td></tr><tr><td>Post-processed prediction</td><td> $6 6 . 3 \pm 1 7 . 7$ </td><td> $3 0 . 3 \pm 1 3 . 6$ </td><td> $3 . 4 \pm 2 . 7$ </td><td> $5 . 0 \pm 2 . 3$ </td><td> $0 . 5 \pm 0 . 7$ </td></tr></table>