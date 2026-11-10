# GHIML: VLM-GUIDED HIERARCHICAL MULTI-LABEL LEARNING WITH APPLICATIONS TO BENTHIC IMAGERY

Isaac Xu<sup>1</sup> Devi Ayyagari<sup>1</sup> H. Martin Gillis<sup>1</sup> Benjamin Misiuk<sup>1,2</sup> Craig J. Brown<sup>3</sup> Thomas Trappenberg<sup>1</sup>

<sup>1</sup>Faculty of Computer Science, Dalhousie University, Halifax, NS, Canada

<sup>2</sup>Departments of Geography and Earth Sciences, Memorial University of Newfoundland, St. John’s, NL, Canada <sup>3</sup>Department of Oceanography, Dalhousie University, Halifax, NS, Canada

isaac.xu@dal.ca, tt@cs.dal.ca

## ABSTRACT

Vision language models (VLMs) encode substantial expert knowledge within their weights, yet their direct application to specialised classification domains such as benthic imagery remains limited. Meanwhile, vision transformers trained on these tasks rely predominantly on low-level visual latents (colours, textures, and shapes) that lack the higher-level reasoning humans employ for identification. We propose Guided Hierarchical Multi-Label (GHiML) learning, a pipeline that extracts conceptual latent variables from a VLM conditioned on training data, generates sample-relevant captions grounded in those variables, and fuses the resulting textual embeddings with image embeddings via cross-attention and a per-node gated router. This approach provides semantic information complementary to low-level visual features, without requiring the VLM to be fine-tuned or a generative model to synthesise training images. We also introduce the inverted HML score $( d _ { \mathrm { i n v } } )$ , a similarity score that evaluates hierarchical multi-label predictions through graph traversal distance and a cardinality penalty, providing an interpretable alternative to existing measures, which are insensitive to realistic degradations in sparse, deep hierarchies. Experiments on BenthicNet, FathomNet and iNaturalist show that the router improves macro $F _ { 1 }$ over the vision-only baseline by up to 6.7 pp and $d _ { \mathrm { i n v } }$ by up to 21.1 pp on in-domain benthic data, with gains persisting under distribution shift. Ablations locate the contribution of the text itself: on the multi-label biotic hierarchies, removing the per-sample text signal degrades $d _ { \mathrm { i n v } }$ while leaving macro $F _ { 1 }$ unchanged, consistent with text acting through prediction cardinality rather than node discrimination, whereas iNaturalist, whose annotations are single-label, loses 18.7% of macro $F _ { 1 }$ . Our results support the hypothesis that VLMs contain exploitable specialised knowledge that, when appropriately elicited, can augment classification in domains where direct visual learning is insufficient<sup>1</sup>.

## 1 Introduction

For general tasks, vision language models (VLMs) have seen widespread adoption (Radford et al., 2021; Alayrac et al., 2022; Yang et al., 2025). However, recent research suggests that out-of-the-box VLMs remain limited in their application to specialised domains such as ocean and medical sciences (Wong et al., 2025; Nath et al., 2025). Consequently, in conjunction with more traditional convolutional neural networks (CNNs) (Fukushima, 1980; He et al., 2016), vision transformers (ViTs) (Dosovitskiy et al., 2021) remain the standard for classification tasks in these research areas (Langenkämper et al., 2017; Battach et al., 2025; Lowe et al., 2025; Aburass et al., 2025). Vision neural networks tend to work through the extraction of low-level latent variables (latents), such as colours, textures, and shapes (Zeiler and Fergus, 2014; Geirhos et al., 2019; Raghu et al., 2021; Ribeiro et al., 2016), resembling the hierarchical visual cortex (Fukushima, 1980). However, these latents lack the generalisable reasoning characteristics that humans use to identify objects. The lack of reasoning in visual tasks within specialised domains therefore remains a research gap. This motivation is illustrated in Figure 1.

![](images/681e17b4d2e1a6ec9accdefb8635a7f09034a8964c34429ea83de801a995bf12.jpg)  
Figure 1: Difference in Latents. A hypothetical example. The red (left) branch depicts classification typical of contemporary vision-based neural networks, whose low-level latents learned from a limited dataset are insufficient to classify the object correctly. The green (right) branch additionally draws on general conceptual knowledge that cannot be learned from visual characteristics alone.

Our application domain, benthic (seafloor) imagery classification, supports ecological and economic monitoring of maritime environments under increasing anthropogenic pressure (Scovil et al., 2025), at a data volume that far exceeds the capacity of manual analysis.

Benthic image classification is often a hierarchical multi-label (HML) task (Katija et al., 2022; Lowe et al., 2025), meaning a single image may contain multiple organism subjects or substrate types, each situated within a taxonomic or visually derived hierarchy that spans broad morphological categories down to fine-grained species-level distinctions. This combination of multi-label outputs and structured hierarchical dependencies poses challenges that extend beyond standard image classification (Giunchiglia and Lukasiewicz, 2020; Wehrmann et al., 2018). The problem is further compounded by substantial visual variability in seafloor imagery, arising from inconsistent lighting conditions, colour casts, and diverse substrate appearances, all of which undermine the robustness of learned representations (Lowe et al., 2025; Katija et al., 2022; Battach et al., 2025; Chen et al., 2021; Langenkämper et al., 2017). Moreover, rare classes, often those of the greatest ecological interest, remain poorly understood and difficult to detect by existing HML methods, as limited training examples prevent models from confidently learning discriminative features (Xu et al., 2024). Collectively, these factors make benthic image classification a particularly demanding domain in which current approaches struggle to achieve the robustness and reliability that ecological monitoring demands.

In an effort to improve robustness in image detection for this domain, we propose Guided Hierarchical Multi-Label (GHiML) learning, an approach that seeks to unite the general knowledge contained in large VLMs with the specialised training of ViTs. GHiML consists of four steps: (1) identifying high-level latents relevant to reasoning about the domain, data points, and their annotations, (2) generating sample-relevant captions for extracted latents, (3) cross-attention fusion between image embeddings and generated textual embeddings for downstream classification, and (4) training a gated mixture of all three modalities across classes (hierarchical nodes). Our hypothesis is that the relationships between tokens encoded in a sufficiently trained VLM contain useful oceanographic connections beyond what low-level visual latents represent. Conditioning on those latents then helps to elicit this stored knowledge as a sample-specific textual signal.

Additionally, we propose an evaluation score for HML predictions, based on graph traversal distance and comparisons of multi-label cardinality (the degree of “multi-labelled-ness”) (Tsoumakas and Katakis, 2009) between prediction and annotation. We refer to it as the inverted HML score $( d _ { \mathrm { i n v } } )$ , a bounded similarity score ranging from 0 to 1, where higher values indicate better predictions. In practice, $d _ { \mathrm { i n v } }$ is reported alongside the node-wise (macro-averaged) $F _ { 1 }$ score (Xu et al., 2024; Lin et al., 2025). The two evaluate different objects, with $F _ { 1 }$ scoring per-node detection and $d _ { \mathrm { i n v } }$ scoring the predicted label set as a whole, and we do not treat either as a substitute for the other.

Our main contributions are therefore:

1. A use-inspired pipeline that elicits domain knowledge from a VLM as a sample-specific textual signal and integrates it with an existing HML framework, rather than a new HML method in itself;

2. $d _ { \mathrm { i n v } }$ , an evaluation score for HML predictions that accounts for both hierarchical proximity and prediction cardinality; and

3. An analysis of what the elicited text contributes, which we find acts as a coarse prior over regions of the hierarchy rather than as a discriminative signal within them.

## 2 Background

## 2.1 Hierarchical Multi-Label Learning

In HML, annotations may be envisioned as a path starting from a root and terminating at a node. Multi-label implies that these annotations can consist of any combination of viable paths. The viability of an annotation is determined by the hierarchical constraint, that is $p _ { \mathrm { p a r e n t } } \ge p _ { \mathrm { c h i l d } }$ . To enforce this condition during training, we apply Coherent Hierarchical Multi-Label Classification Neural Networks (C-HMCNN) (Giunchiglia and Lukasiewicz, 2020), which has established itself as the predominant framework for constrained HML classification and remains a robust standard (Xu et al., 2026; Liu et al., 2023). C-HMCNN converts the hierarchy of $| \nu |$ nodes (V being the set of nodes) into an adjacency matrix $( \mathbf { A } \in \mathbb { R } ^ { | \mathcal { V } | \times | \mathcal { V } | } )$ that encodes ancestor-descendant relationships, then applies a maximum operation over each node’s sub-hierarchy to ensure no child exceeds its parent’s predicted probability. This constraint mechanism $( f _ { \mathrm { C M } } )$ is insufficient alone, and a modified binary cross-entropy (BCE) loss is required to ensure gradients point in the correct direction:

$$
{ \mathcal { L } } _ { \mathrm { M C } } = \mathrm { B C E } \Bigg ( \underbrace { \overbrace { ( 1 - \mathbf { Y } ) f _ { \mathrm { C M } } \bigl ( \theta ( \mathbf { X } ) , \mathbf { A } \bigr ) } ^ { \mathrm { N e g a t i v e - n o d e ~ c o n t r i b u t i o n } } + \overbrace { f _ { \mathrm { C M } } \bigl ( \mathbf { Y } \theta ( \mathbf { X } ) , \mathbf { A } \bigr ) } ^ { \mathrm { P o s i t i v e - n o d e ~ c o n t r i b u t i o n } } } _ { \mathrm { F u l l ~ c o n s t r a i n e d ~ o u t p u t } } , ~ \mathbf { Y } \Bigg ) ;\tag{1}
$$

where $\mathcal { L } _ { \mathrm { M C } }$ is referred to as the max constraint loss (MCLoss) (Giunchiglia and Lukasiewicz, 2020; Xu et al., 2026), $\mathbf { X } \in \mathbb { R } ^ { n _ { \mathrm { b } } \times d }$ is the input of batch size $n _ { \mathrm { b } }$ and feature size d, θ is the model which maps from d to |V|, and $\mathbf { Y } \in \mathbb { R } ^ { n _ { \mathrm { b } } \times | \nu | }$ is the annotation in binary representation. Finally, the A in Equation 1 has been expanded to be in $\mathbb { R } ^ { n _ { \mathrm { b } } \times | \nu | \times | \nu | }$ . Most notably, the positive-node contribution must be filtered by the positive annotations prior to constraint to remove the effect of negatively-annotated descendants on their ancestors.

## 3 Methods

Our approach is generalised to HML annotations which are not obligated to reach a leaf node (partial-depth). Technical details required for reproduction are provided in Appendix A.

## 3.1 Extracting Useful Latents

The training partition of an image dataset is supplied to a VLM with an appropriate prompt, as shown in the first part of Figure 2. Formally, using terms from Figure 1, we sample a set of conceptual latents $\bar { \mathbf Z } _ { 2 } \sim p ( \mathbf Z _ { 2 } \mid \mathbf X , \mathbf Y ; \phi )$ , where the model ϕ must first leverage low-level visual latents $\mathbf { Z } _ { 1 }$ . The variables in $\mathbf { Z } _ { 2 }$ should not resemble those we might expect from $\mathbf { Z } _ { 1 }$ . The goal is not to generate a description of the image and introduce redundant information for a visual encoder. Instead, the usefulness of $\mathbf { Z } _ { 2 }$ is predicated on its ability to bias the model towards conceptual knowledge acquired through pretraining on large-scale language data.

With a sufficiently large and diverse dataset, this procedure can return thousands of candidate variables, many of which are redundant or weakly informative. We therefore refine the proposed set in four steps, specified in full in Appendix E. (1) We compute the empirical distribution $p ( v )$ over nodes from the training annotations. (2) Since samples may be multi-labelled, an EM procedure resolves the ambiguous association between each proposed variable $z _ { 2 , j }$ and individual nodes, yielding a distribution $q _ { j } ( v )$ over nodes. (3) Each $q _ { j }$ is smoothed towards the prior, and candidates are scored by $D _ { \mathrm { K L } } ( \tilde { q } _ { j } \parallel p )$ ; variables with high divergence concentrate their annotation mass on rarer nodes and are hypothesised to carry the most discriminative information beyond the prior, and we retain only those scoring $\geq 1$ . (4) After deduplication, a uniform per-node budget B selects the top-B candidates by KL score for each leaf node, as providing smaller VLMs with a large quantity of keywords can introduce anchoring bias (Shi et al., 2023; Wei et al., 2023). The final variable set typically comprises 15-20 variables in our runs.

## 3.2 Latent Captioning

We refer again to Figure 2. This step consists of supplying the processed set of variables to the VLM for captioning. In addition to captioning for each variable, the model is prompted to assign a binary score for the relevance of the variable

![](images/57e689bb19de3b9039297998d4d93930ff72eb5d784b4fb6418375b62f80121e.jpg)  
Figure 2: VLM Variable Proposal and Captioning. A sea star image and its annotation are used to propose the variable “ambush”; during captioning, the VLM is prompted to reason about each variable for each sample.

to each sample. This is done for analysis purposes, and is inspired by Liu et al. (2024). A quality check is performed to ensure that all variables are accounted for in the captions of each sample.

## 3.3 Models and Architecture

We refer to Figure 3 for an overview of the models and training pipeline we use. Images are embedded with a DINOv3 ViT-H backbone (Siméoni et al., 2025), Qwen3-VL-8B-Instruct is used for variable proposal and captioning, and Qwen3-Embedding-8B for embedding captions (Yang et al., 2025). These models were chosen for their open-source availability, reliability, and ability to fit on consumer-grade GPUs accessible to domain-specialised research labs.

To summarise our models, we train self-attention-based pooling mechanisms to transform sequential image and text embeddings into dense vectors. These unimodal models serve as baselines and reveal the overlap between textual and visual performance, specifically the degree to which each modality carries redundant or complementary information. A third cross-attention mechanism is trained with image as the query modality and text as the key-value modality<sup>2</sup>. Finally, we train a router (Jacobs et al., 1991) that takes as input the concatenated pooled embeddings and outputs softmax weights over the three modalities for each node, so that the contribution of each modality may vary by node and by sample. We run two versions of this router: one that loads frozen pretrained pooling layers from our unimodal runs and trains the classifiers and router mechanism (Router), and one with all parameters trained from scratch (Router<sup>†</sup>).

Every row of Table 1 uses the same hierarchical classification head, an implementation of C-HMCNN (Giunchiglia and Lukasiewicz, 2020) with the max-based constraint f<sub>CM</sub> and its accompanying loss, and the same per-node thresholding procedure (subsection 3.5). The rows therefore differ only in which modalities are available and how they are combined, so C-HMCNN is held constant as a control rather than compared against.

## 3.4 Inverted HML Score $d _ { \mathbf { i n v } }$

Evaluating HML predictions requires a score that captures both closeness between prediction and ground truth and multi-label cardinality, penalising predictions that are deficient or excessive, given that real-world annotations may be incomplete (Lowe et al., 2025; Katija et al., 2022). Existing scores do not fully satisfy these requirements. Node-wise

![](images/8f7841c61068120ee24d2cf7603b7415481155578e7fbbe7e57a5412a6dc0ec7.jpg)  
Figure 3: Model Architectures. The (1) image and (2) text streams apply self-attention pooling followed by an HML-constrained classifier; (3) applies crossattention followed by pooling; (4) a router predicts per-node weights that mix the logits of (1)–(3).

$F _ { 1 }$ penalises cardinality errors but forfeits hierarchical proximity, particularly at standard 0.5 thresholding, where deeper nodes frequently score zero regardless of prediction closeness (Xu et al., 2026). Micro-averaged AP (Giunchiglia and Lukasiewicz, 2020; Wehrmann et al., 2018) pools across all nodes, so evaluation in sparse hierarchies lets easily ranked negatives dominate, producing overly optimistic scores. A comparison of these scores is presented in Appendix I.

We address these gaps with $d _ { \mathrm { i n v } }$ , which directly exploits hierarchical distance and explicitly accounts for multi-label cardinality (with user-stipulated flexibility). Because it maps graph traversal distance onto a normalised [0, 1] scale, the score is directly interpretable: a value of one indicates a perfect prediction, while intermediate values communicate precisely how far the prediction sits in terms of the hierarchy’s own structure, and how far its cardinality departs from the annotation. We define:

$$
d _ { \mathrm { i n v } } = s _ { \mathrm { m a x } } \gamma ,\tag{2}
$$

where $s _ { \mathrm { m a x } }$ encapsulates both normalisation and “identification” (determining which multi-label component in the prediction corresponds to which component in the ground truth). Both $\hat { \mathcal { V } }$ and $\mathcal { V }$ are ancestor-closed by construction, and $d _ { \mathrm { i n v } }$ is computed over their endpoints: the active nodes carrying no further active descendants. Ancestors are therefore not scored twice, and a prediction is evaluated by where its most specific claims fall. For normalisation, we map the prediction distance $d _ { \mathrm { p r e d } } ^ { ( i j ) }$ between ground-truth endpoint i and predicted endpoint $j$ onto a [0, 1] scale via $\begin{array} { r } { [ \mathbf { D } _ { \mathrm { i n v } } ] _ { i j } = ( d _ { \mathrm { m a x } } ^ { ( i ) } - d _ { \mathrm { p r e d } } ^ { ( i j ) } ) / d _ { \mathrm { m a x } } ^ { ( i ) } } \end{array}$ , where $d _ { \operatorname* { m a x } } ^ { ( i ) }$ is the eccentricity of ground-truth endpoint i, i.e. the greatest tree distance from it to any other node in the hierarch $\mathrm { 1 y ^ { 3 } }$ . For identification, we construct a matrix $\mathbf { D } _ { \mathrm { i n v } } \in \mathbb { R } ^ { n _ { \mathrm { t r u t h } } \times n _ { \mathrm { p r e d } } }$ of pairwise similarity scores for all component combinations, and solve the resulting rectangular assignment problem using the Jonker–Volgenant algorithm (Crouse, 2016). Writing $f _ { \mathrm { J V } } ( \hat { \mathcal { V } } , \mathcal { V } ; \mathbf { D } _ { \mathrm { i n v } } )$ for the total similarity of the optimal assignment, we normalise by the number of ground-truth components to obtain

$$
s _ { \mathrm { m a x } } = \frac { 1 } { n _ { \mathrm { t r u t h } } } f _ { \mathrm { J V } } ( \hat { \mathcal { V } } , \mathcal { V } ; \mathbf { D } _ { \mathrm { i n v } } ) ,\tag{3}
$$

the optimistically matched combination score between prediction $( \hat { \mathcal { V } } )$ and ground truth (Y). The cardinality factor γ penalises excessive predictions, an important consideration in HML settings. Defining the number of excessive predictions as $n _ { \mathrm { E } } = n _ { \mathrm { p r e d } } - n _ { \mathrm { t r u t h } }$ and the maximum number of excessive predictions as $n _ { \mathrm { M E } } = | \mathcal { V } | - n _ { \mathrm { t r u t h } }$ , we model

γ with exponential decay:

$$
\gamma ( n _ { \mathrm { E } } ; k _ { \mathrm { P } } , n _ { \mathrm { M E } } ) = \left\{ \begin{array} { l l } { a e ^ { - b n _ { \mathrm { E } } } + c } & { \mathrm { i f } \ k _ { \mathrm { P } } \neq k _ { \mathrm { P } } ^ { \ast } , } \\ { 1 - \displaystyle \frac { n _ { \mathrm { E } } } { n _ { \mathrm { M E } } } } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{4}
$$

where $k _ { \mathrm { P } } \in ( 0 , 1 )$ governs the penalty incurred by a single excess prediction, $k _ { \mathrm { P } } ^ { * } = ( n _ { \mathrm { M E } } - 1 ) / n _ { \mathrm { M E } }$ is the critical value at which the decay reduces to the linear case, and the parameters $a = ( 1 - e ^ { - b n _ { \mathrm { M E } } } ) ^ { - 1 }$ and $c = 1 - a$ are determined by boundary conditions $\gamma ( 0 ) = 1$ and $\gamma ( n _ { \mathrm { M E } } ) = 0$ , with b obtained as the unique non-trivial root of $( k _ { \mathrm { P } } - 1 ) e ^ { - b n _ { \mathrm { M E } } } + e ^ { - b } - k _ { \mathrm { P } } = 0$ (formal proofs of existence and uniqueness are provided in Appendix J). Edge cases (disjoint roots, empty and deficient predictions) are defined in Appendix J.

As $d _ { \mathrm { i n v } }$ is defined per sample, we report it macro-averaged across nodes, as with $F _ { 1 }$ : each node receives the mean $d _ { \mathrm { i n v } }$ of the test samples annotated with $\mathbf { i t } ,$ and these node scores are averaged with equal weight. We set $k _ { \mathrm { P } } = 0 . 5 ,$ , as a significant contemporary challenge in the benthic domain is the possibility of incomplete annotations (Barnard and Chrobak, 2026). Consequently, the annotation cardinalities cannot be fully trusted.

## 3.5 Optimal $F _ { 1 }$ Thresholding

Deeper and rare nodes are poorly detected in HML (Pereira et al., 2021; Lipton et al., 2014), possibly in part due to poor calibration (Xu et al., 2024). Following Lipton et al. (2014), we grid-search per-node thresholds maximising $F _ { 1 }$ (0.01 to 0.99 in 0.01 increments, starting from 0.50), subject to the hierarchical constraint $\begin{array} { r } { t _ { \mathrm { p a r e n t } } \leq \operatorname* { m i n } _ { v \in \mathcal { V } _ { \mathrm { d } } } t _ { v } } \end{array}$ , where $\nu _ { \mathrm { d } }$ denotes the set of descendants. Thresholds are fitted on training predictions only, and the procedure is universally applied to every model reported.

## 4 Experiments

## 4.1 Datasets

We evaluate on three datasets, described in full in Appendix C. BenthicNet (Lowe et al., 2025) provides 188,679 annotated images labelled under the CATAMI scheme (Althaus et al., 2015), from which we form two overlapping tasks, substrate (BN-Sub, 23 nodes) and biota (BN-Bio, 293 nodes). FathomNet (2025) (FN) (FathomNet 2025) is a benthic biota competition dataset of 9,305 full images, for which we extracted a 192-node taxonomic hierarchy from WoRMS (WoRMS Editorial Board, 2026), making the problem HML (full-depth). For both, the innate test partitions serve as out-of-domain (OOD) test sets, and we additionally construct in-domain (ID) test sets from a random 20% of the data. These partitions are $\mathrm { \bf { \tilde { \rho } } { \bf { \tilde { \rho } } } O O D } ^ { * }$ because BenthicNet’s test set is geospatially partitioned and FathomNet’s competition test set was hand-selected. Lastly, iNaturalist (iNat) (Van Horn et al., 2018) spans terrestrial fauna, flora, and fungi with single-label annotations over a hierarchy of over 16,000 nodes; we train on the $5 0 0 { , } 0 0 0$ training samples of iNat (mini) and test on its 100,000 validation samples, replacing MCLoss with a softmax over the $1 0 { , } 0 0 0$ endpoints, marginalised up the hierarchy. Since each prediction and annotation consists of exactly one endpoint, $n _ { \mathrm { E } } = 0$ and $\gamma \equiv 1$ , so $d _ { \mathrm { i n v } }$ reduces to normalised traversal similarity on this dataset.

## 4.2 Results and Discussion

As shown in Table 1, the router models achieve the strongest performance in nearly all settings. The ranking across methods is consistent for each metric across all random seeds. Although text underperforms vision on $F _ { 1 }$ at nearly every node, the two metrics disagree in a specific way: at 75 of the 171 BN-Bio nodes with sufficient support, text scores higher on $d _ { \mathrm { i n v } }$ while failing to identify the node on $F _ { 1 }$ , against 6 nodes where the reverse holds. This indicates that text contributes generalisable semantic information about where predictions belong in the hierarchy, rather than finer discrimination between nodes.

One contributing factor is that several of the selected variables encode functional traits, such as mobility, attachment strategy, and predation mode, that are particularly informative for sessile organisms. Algae and sponges, for instance, share many such traits, and the resulting captions place them in close proximity within the hierarchy even when exact classification fails.

Table 2 reports the same comparison on iNat, where the hierarchy is roughly 55 times larger and annotations are single-label. The ordering is unchanged: both routers exceed the image branch on both metrics at every seed, and text alone remains the weakest by a wide margin on $F _ { 1 }$ while retaining half of the image branch’s $d _ { \mathrm { i n v } }$ . Because $d _ { \mathrm { i n v } }$ has no cardinality term on iNat, this is a direct measure of proximity. A text stream with poor $F _ { 1 }$ on its own still places predictions within reach of the target.

Table 1: HML Results. Macro-average performance across datasets, modalities, and distribution settings. Values are mean ± std (%). Superscript indicates mean rank across seeds. Ten seeds were tested for BN datasets, and thirty for FN. Coloured values differ significantly from Img alone (Wilcoxon signed-rank test; green: $p < 0 . 0 5 ;$ bold dark green: $p < 0 . 0 1 )$ .
<table><tr><td rowspan="2" colspan="2"></td><td colspan="2">BN-Sub</td><td colspan="2">BN-Bio</td><td colspan="2">FN</td></tr><tr><td>F1</td><td> $d _ { \mathbf { i n v } }$ </td><td>F1</td><td> $d _ { \mathbf { i n v } }$ </td><td>F1</td><td> $d _ { \mathbf { i n v } }$ </td></tr><tr><td rowspan="5">D</td><td>Txt</td><td> $4 8 . 5 _ { \pm 2 . 9 } ( 5 . 0 )$ </td><td> $4 0 . 5 _ { \pm 5 . 3 } ( 5 . 0 )$ </td><td> $7 . 8 _ { \pm 1 . 6 } ^ { ( 5 . 0 ) }$ </td><td> $1 . 2 _ { \pm 1 . 7 } \AA ^ { ( 5 . 0 ) }$ </td><td> $1 5 . 6 _ { \pm 2 . 9 } ( ^ { 5 . 0 ) }$ </td><td> ${ 0 . 3 } _ { \pm 0 . 2 } { ^ { ( 5 . 0 ) } }$ </td></tr><tr><td>Img</td><td> $7 7 . 9 _ { \pm 3 . 5 } ( 3 . 4 )$ </td><td> $6 8 . 1 _ { \pm 4 . 5 } \mathrm { \Lambda } ^ { ( 3 . 5 ) }$ </td><td> $1 4 . 3 _ { \pm 3 . 8 } ( ^ { 3 . 2 ) }$ </td><td> $5 . 1 _ { \pm 5 . 5 } \mathrm { } ^ { ( 3 . 5 ) }$ </td><td> $4 4 . 5 { \scriptstyle \pm 5 . 1 } ^ { ( 3 . 5 ) }$ </td><td> $1 6 . 5 _ { \pm 3 . 2 } ( ^ { 3 . 5 ) }$ </td></tr><tr><td> $\mathrm { I m g + T x t }$ </td><td> $7 8 . 3 _ { \pm 3 . 7 } ( 3 . 0 ) $ </td><td> $6 8 . 7 _ { \pm 4 . 3 } ( ^ { 3 . 0 ) }$ </td><td> $1 4 . 0 _ { \pm 3 . 5 } \AA ^ { ( 3 . 8 ) }$ </td><td> $4 . 9 _ { \pm 5 . 6 } \AA ^ { ( 3 . 5 ) }$ </td><td> $4 4 . 5 _ { \pm 5 . 2 } ( 3 . 5 )$ </td><td> $1 6 . 7 _ { \pm 3 . 3 } ( ^ { 3 . 4 ) }$ </td></tr><tr><td>Router</td><td> ${ \bf 8 1 . 4 { \scriptstyle \pm 2 . 5 } } ^ { ( { \bf 1 . 1 } ) }$ </td><td> $\mathbf { 7 2 . 7 { \pm } 3 . 0 } ^ { ( \mathbf { 1 . 1 } ) }$ </td><td> ${ \bf 1 6 . 6 { \scriptstyle \pm 3 . 5 } } ^ { ( 2 . 0 ) }$ </td><td> ${ \bf 7 . 6 { \scriptstyle \pm 6 . 5 } } ^ { ( 1 . 5 ) }$ </td><td> ${ \bf 5 1 . 2 \pm _ { 4 . 2 } ( 1 . 1 ) }$ </td><td> $\mathbf { 3 7 . 6 { \scriptstyle \pm 4 . 1 } } ^ { ( 1 . 0 ) }$ </td></tr><tr><td> ${ \mathrm { R o u t e r } } ^ { \dagger }$ </td><td> $7 9 . 4 { \scriptstyle \pm 2 . 3 } ^ { ( 2 . 5 ) }$ </td><td> $6 9 . 4 { \scriptstyle \pm 3 . 9 } ^ { ( 2 . 4 ) }$ </td><td> $\mathbf { 1 7 . 8 { \scriptstyle \pm 4 . 0 } } ^ { ( 1 . 0 ) }$ </td><td> ${ \bf 7 . 7 { \pm 6 . 9 } } ^ { ( 1 . 5 ) }$ </td><td> ${ \bf 4 9 . 7 { \scriptstyle \pm 4 . 6 } } ^ { ( 1 . 9 ) }$ </td><td> $\mathbf { 2 3 . 0 { \scriptstyle \pm 4 . 6 } } ^ { ( 2 . 0 ) }$ </td></tr><tr><td rowspan="5">DOOD</td><td>Txt</td><td> $3 8 . 6 _ { \pm 3 . 2 } ( ^ { 5 . 0 ) }$ </td><td> $3 6 . 6 _ { \pm 5 . 7 } \AA ^ { ( 5 . 0 ) }$ </td><td> $7 . 4 _ { \pm 1 . 5 } ^ { } ( 5 . 0 )$ </td><td> $2 . 5 _ { \pm 3 . 4 } ( ^ { 4 . 8 ) }$ </td><td> $1 2 . 4 _ { \pm 4 . 8 } ( ^ { 5 . 0 ) }$ </td><td> $0 . 4 { \scriptstyle \pm 0 . 3 } ^ { ( 5 . 0 ) }$ </td></tr><tr><td>Img</td><td> $6 4 . 8 _ { \pm 4 . 5 } \mathrm { \Omega } ^ { ( 2 . 5 ) }$ </td><td> $6 0 . 2 { \scriptstyle \pm 5 . 3 } ^ { ( 3 . 2 ) }$ </td><td> $1 1 . 4 _ { \pm 2 . 7 } ( 3 . 4 )$ </td><td> $4 . 4 _ { \pm 5 . 1 } \AA ^ { ( 3 . 8 ) }$ </td><td> $3 7 . 4 { \scriptstyle \pm 1 0 . 0 } ^ { ( 3 . 4 ) }$ </td><td> $1 8 . 3 _ { \pm 4 . 8 } \AA ^ { ( 3 . 5 ) }$ </td></tr><tr><td> $\mathrm { I m g + T x t }$ </td><td> $6 4 . 2 _ { \pm 4 . 6 } \AA ^ { ( 3 . 1 ) }$ </td><td> $6 0 . 4 { \scriptstyle \pm 5 . 1 } ^ { ( 2 . 7 ) }$ </td><td> $1 1 . 3 _ { \pm 2 . 8 } \AA ^ { ( 3 . 6 ) }$ </td><td> $4 . 7 _ { \pm 5 . 8 } { } ^ { ( 3 . 0 ) }$ </td><td> $3 7 . 4 { \scriptstyle \pm 1 0 . 0 } ^ { ( 3 . 4 ) }$ </td><td> ${ 1 8 . 4 _ { \pm 5 . 1 } } ^ { ( 3 . 3 ) }$ </td></tr><tr><td>Router</td><td> ${ \bf 6 7 . 7 { \pm 3 . 1 } } ^ { ( { \bf 1 . 1 } ) }$ </td><td> ${ \bf 6 4 . 5 { \scriptstyle \pm 3 . 6 } } ^ { ( 1 . 0 ) }$ </td><td> $\mathbf { 1 3 . 2 _ { \pm 3 . 0 } } ^ { ( 1 . 9 ) }$ </td><td> ${ \bf 5 . 9 { \bf _ { \pm 5 . 8 } } } ^ { ( 2 . 0 ) }$ </td><td> ${ \bf 4 0 . 9 { \scriptstyle \pm 9 . 2 } } ^ { ( 1 . 7 ) }$ </td><td> $\mathbf { 3 2 . 5 { \scriptstyle \pm 6 . 4 } } ^ { ( 1 . 0 ) }$ </td></tr><tr><td> ${ \mathrm { R o u t e r } } ^ { \dagger }$ </td><td> $6 4 . 0 _ { \pm 4 . 8 } \AA ^ { ( 3 . 3 ) }$ </td><td> $6 0 . 4 _ { \pm 5 . 3 } ( ^ { 3 . 1 ) }$ </td><td> ${ \bf 1 4 . 3 _ { \pm 3 . 7 } ( 1 . 1 ) }$ </td><td> ${ \bf 6 . 4 . { _ { \pm 6 . 4 } } } ^ { ( 1 . 4 ) }$ </td><td> ${ \bf 4 1 . 7 { \scriptstyle \pm 9 . 2 } } ^ { ( 1 . 5 ) }$ </td><td> $\mathbf { 2 3 . 1 _ { \pm 6 . 5 } } ^ { ( 2 . 2 ) }$ </td></tr></table>

<table><tr><td rowspan="3">Txt</td><td>F1</td><td> $d _ { \mathbf { i n v } }$ </td></tr><tr><td></td><td></td></tr><tr><td> ${ 0 . 9 } _ { \pm 0 . 1 } \mathrm { } ^ { ( 5 . 0 ) }$ </td><td> $2 8 . 4 _ { \pm 2 . 9 } ( ^ { 5 . 0 ) }$ </td></tr><tr><td>Img</td><td> $2 2 . 8 _ { \pm 4 . 2 } \ L ^ { ( 4 . 0 ) }$ </td><td> $5 6 . 3 _ { \pm 2 . 7 } \AA ^ { ( 4 . 0 ) }$ </td></tr><tr><td> $\mathrm { I m g + T x t }$ </td><td> $2 9 . 8 _ { \pm 6 . 0 } \mathrm { ^ { ( 3 . 0 ) } }$ </td><td> $6 1 . 2 _ { \pm 4 . 2 } ( ^ { 3 . 0 ) }$ </td></tr><tr><td>Router</td><td> $5 5 . 6 { \scriptstyle \pm 2 . 1 } ^ { ( 2 . 0 ) }$ </td><td> $7 7 . 9 { \scriptstyle \pm 1 . 7 } ^ { ( 1 . 0 ) }$ </td></tr><tr><td> $\scriptstyle { \mathrm { R o u t e r } } ^ { \dagger }$ </td><td> $5 9 . 0 { \scriptstyle \pm 2 . 9 } ^ { ( 1 . 0 ) }$ </td><td> $7 5 . 9 _ { \pm 2 . 6 } ( ^ { 2 . 0 ) }$ </td></tr></table>

Table 2: iNaturalist Results. Macro-average performance over the 100,000 held-out images. Values are mean ± std (%) across five seeds; superscript indicates mean rank across seeds. Coloured values are significantly higher than Img (one-sided Wilcoxon signed-rank test; green: $p < 0 . 0 5 )$

We next ask whether text and vision contribute distinct information, and whether the router successfully exploits both Among the top-performing nodes for text, there is low overlap between the two modalities, and the router frequently surpasses vision while retaining a margin of advantage comparable to that of text alone (Appendix H). There, we also study the relevant variables and their correlations with text advantage, as well as with individual nodes. We observe that the proposed variables provide strong contextual information that benefits classification, even when they do not directly improve prediction of the original nodes with which they were associated via KL divergence.

This last point warrants emphasis. Although variables were assigned to modal nodes, the captions they generate encode a far richer set of combinations. The variable ambush, for example, was initially selected for its association with sea stars. In practice, it also describes rocky substrates (habitats in which ambush predators tend to occur), detects lobsters (themselves ambush predators), and identifies organisms or environments facilitating camouflage consistent with ambush strategy. A single functional concept thus provides signal across multiple nodes. This is consistent with considering ecological function and context rather than relying solely on low-level visual latents such as colour and texture.

## 4.3 Text as a Sub-Tree Prior

Beyond the trained branches, we also examine the embeddings themselves. We fit two k-NN classifiers over the frozen image and text embeddings, mean-pooled across the sequence dimension, and evaluate them on the same samples by cross-validation (Appendix B). No trained component of our pipeline is used, so any complementarity observed is a property of the representations rather than of our architecture. Were the text embedding correct only on samples that the image embedding already correctly classifies, it would be strictly redundant and the independence ratio P(text correct | image wrong) / P(text correct) would be zero. Across the BN and iNat datasets, 3.5–10.8% of samples across hierarchy levels are classified correctly by the text embedding but not by the image embedding. A more detailed k-NN comparison against generic captioning (image description) embeddings in BN-Bio (Appendix B)

Table 3: Mean Replacement. Relative change in scores across five seeds when each sample’s text embedding is replaced by the mean. Negative values indicate degradation. Bolded indicate $p \leq 0 . 0 5$ on paired one-sided Wilcoxon signed-rank test.
<table><tr><td>dataset</td><td>annotation</td><td> $\Delta F _ { 1 }$ </td><td> $\Delta d _ { \mathrm { i n v } }$ </td></tr><tr><td>BN-Sub (23 nodes)</td><td>multi-label</td><td>+1.5%</td><td>-0.1%</td></tr><tr><td>BN-Bio (293 nodes)</td><td>multi-label</td><td>+0.6%</td><td>-21.7%</td></tr><tr><td>iNaturalist (10,000 endpoints)</td><td>single-label</td><td>-18.7%</td><td>-10.2%</td></tr></table>

demonstrates that our extracted keywords help reduce redundancy with images precisely at the clade levels which benefit classification the most.

In Table 3, a mean embedding vector is supplied to the models in lieu of the original text embedding, such that no per-sample textual information survives. On BN-Bio the degradation occurs only for $d _ { \mathrm { i n v } } ,$ while $F _ { 1 }$ is unaffected. On iNat both metrics degrade. BN-Sub, the smallest hierarchy, is unaffected in either metric. The text signal therefore appears to matter more for larger and more complex hierarchies. The two biotic datasets differ in which metrics respond, and we attribute this to their label structure. Under multi-label annotations, the clade signal appears to suppress spurious subtree activations, which is expressed through prediction cardinality rather than hierarchical placement (subsection I.2). However, iNat’s single-label annotations foreclose that channel. Replacing the text with scale-matched Gaussian noise degrades performance below the image-only baseline on three of the four datasets, indicating that the gains do not follow from the added dimensionality alone (subsection G.2).

We also report on zero-shot VLM classification, which produced unscorable results. A zero-shot VLM cannot produce scheme-conformant predictions for this task without conditioning, and even with scheme-adherent conditioning, it over-predicts substantially (subsection G.1). Additionally, we test a descriptor-matching baseline, which recovers roughly half the node $F _ { 1 }$ of a trained image branch and under a tenth of its $d _ { \mathrm { i n v } }$ , with descriptors reused heavily between sibling nodes (subsection G.3). Many benthic images are homogeneous in colour, texture and tone despite differing content, so generic appearance-based descriptions tend to be interchangeable across nodes. Finally, we analyse the router’s gate weights on BN-Bio and confirm that they remain open to text (subsection H.2).

## 5 Related Works

Our work is influenced by the causal discovery literature, most notably COAT (Liu et al., 2024). COAT employs LLMs to discover a phenomenon’s Markov Blanket by proposing candidate variables and annotating each sample with a trinary encoding per variable (relevant, irrelevant, or inversely relevant). COAT demonstrates that LLMs can actively mediate between unstructured observations and structured representations. Our pipeline differs in that we target hierarchical multi-label classification, select variables per hierarchy node by KL divergence, and have a VLM score each variable with a brief visual justification. The resulting captions are embedded as dense text and fused with the image representation through a learned per-node router, so the variables serve as contextual signal for a vision model and are never used as discrete features for causal analysis.

Closer in line with our motivation are diffusion-based augmentation strategies (Yerramilli et al., 2025; Li et al., 2024; Trabucco et al., 2024; Samuel et al., 2024), which leverage the visual priors of large-scale generative models to augment downstream classification. SGID (Li et al., 2024) uses BLIP-generated captions (Li et al., 2023) and CLIP-based selection (Radford et al., 2021) to guide Stable Diffusion toward class-consistent samples, DA-Fusion (Trabucco et al., 2024) semantically augments real images via Stable Diffusion, and SeedSelect (Samuel et al., 2024) optimises the initial noise seed to improve rare class generation. However, these approaches are largely inapplicable to benthic habitat classification, where many taxa remain underrepresented in the web-scale data on which diffusion models are trained (Zhang et al., 2025). Instead, much of our knowledge of benthic taxa and habitat is encoded textually, motivating our alternative strategy of extracting higher-level semantic information from a VLM rather than synthesising novel training samples.

Within the HML literature, research into improving detection and addressing class imbalance remains sparse, with existing methods broadly employing either resampling techniques (Pereira et al., 2021) or weighted objectives (Xu et al., 2026). Our work enhances node detection broadly by leveraging the general knowledge of an external foundation model.

Lastly, we compare $d _ { \mathrm { i n v } }$ with the hierarchical $F _ { 1 }$ of Lin et al. (2025) and with CoPHE (Falis et al., 2021) in subsection I.1. We find $h F _ { 1 }$ comparatively lenient toward excessive prediction cardinality, since excess predictions that share correct ancestors inflate its ancestor-expanded sets only marginally, whereas CoPHE, which addresses this leniency, does not measure how far a prediction sits from its target and requires a well-defined depth for every node.

## 6 Conclusion

We have presented GHiML, a pipeline for extracting and leveraging higher-level semantic knowledge from VLMs to augment hierarchical multi-label classification in specialised visual domains. Through a structured variable proposal, EM-based filtering, and sample-conditioned captioning procedure, we demonstrated that a VLM can be positioned to generate complementary contextual information to standard visual features. The gated router architecture exploits this information on a per-node basis, consistently outperforming both unimodal baselines and the cross-attention fusion model across two benthic datasets under both in-domain and out-of-domain considerations, and on iNat.

We additionally introduced $d _ { \mathrm { i n v } } ,$ a metric that grounds HML evaluation in the structure of the hierarchy itself. By combining graph traversal distance with an explicit cardinality penalty governed by a single interpretable parameter, $d _ { \mathrm { i n v } }$ addresses gaps left by existing methods, and offers an intuitive understanding of model performance. Limitation of the work are discussed in greater detail in Appendix F.

## Acknowledgements

This research was supported by the Ocean Frontier Institute (OFI) through an award from the Canada First Research Excellence Fund, and as part of the Benthic Ecosystem Mapping and Engagement (BEcoME) project. Computational resources were provided by the Digital Research Alliance of Canada. We thank Dr. Mariano Maisonnave for helpful discussions.

## References

Sanad Aburass, Osama Dorgham, Jamil Al Shaqsi, Maha Abu Rumman, and Omar Al-Kadi. Vision transformers in med ical imaging: a comprehensive review of advancements and applications across multiple diseases. Journal ofimaging informatics in medicine, 38(6):3928–3971, December 2025. ISSN 2948-2933. doi: 10.1007/s10278-025-01481-y. URL https://europepmc.org/articles/PMC12701147.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikoł aj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan.´ Flamingo: a visual language model for few-shot learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 23716–23736. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ 960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf.

Franziska Althaus, Nicole Hill, Renata Ferrari, Luke Edwards, Rachel Przeslawski, Christine H. L. Schönberg, Rick Stuart-Smith, Neville Barrett, Graham Edgar, Jamie Colquhoun, Maggie Tran, Alan Jordan, Tony Rees, and Karen Gowlett-Holmes. A standardised vocabulary for identifying benthic biota and substrata from underwater imagery: The CATAMI classification scheme. PLOS ONE, 10(10):1–18, 10 2015. doi: 10.1371/journal.pone.0141039. URL https://doi.org/10.1371/journal.pone.0141039.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016. URL https://arxiv.org/ abs/1607.06450.

Kevin Barnard and Laura Chrobak. FathomNetCLEF2026 @ LifeCLEF & CVPR-FGVC. https://kaggle.com/ competitions/fathomnet-2026, 2026. Kaggle.

Yahia Battach, Abdulwahab Felemban, Faizan Farooq Khan, Yousef A. Radwan, Xiang Li, Fabio Marchese, Sara Beery, Burton H. Jones, Francesca Benzoni, and Mohamed Elhoseiny. ReefNet: A large scale, taxonomically enriched dataset and benchmark for hard coral classification. arXiv, 2025. URL https://arxiv.org/abs/2510.16822.

Qimin Chen, Oscar Beijbom, Stephen Chan, Jess Bouwmeester, and David Kriegman. A new deep learning engine for CoralNet. In IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 3686–3695, 10 2021. doi: 10.1109/ICCVW54120.2021.00412.

David F. Crouse. On implementing 2D rectangular assignment algorithms. IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679–1696, 2016. doi: 10.1109/TAES.2016.140952.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

eor123, Kevin Barnard, Laura Chrobak, and picekl. FathomNet 2025 @ CVPR-FGVC. https://kaggle.com/ competitions/fathomnet-2025, 2025. Kaggle.

Matúš Falis, Hang Dong, Alexandra Birch, and Beatrice Alex. CoPHE: A count-preserving hierarchical evaluation metric in large-scale multi-label text classification. CoRR, abs/2109.04853, 2021. URL https://arxiv.org/abs/ 2109.04853.

Kunihiko Fukushima. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biological Cybernetics, 36(4):193–202, 4 1980. ISSN 1432-0770. doi: 10.1007/ BF00344251. URL https://doi.org/10.1007/BF00344251.

Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bygh9j09KX.

Eleonora Giunchiglia and Thomas Lukasiewicz. Coherent hierarchical multi-label classification networks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 9662–9673. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper\_files/paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

Dan Hendrycks and Kevin Gimpel. Bridging nonlinearities and stochastic regularizers with Gaussian Error Linear Units. CoRR, abs/1606.08415, 2016. URL http://arxiv.org/abs/1606.08415.

Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive mixtures of local experts. Neural Computation, 3(1):79–87, 1991. doi: 10.1162/neco.1991.3.1.79.

Kakani Katija, Eric Orenstein, Brian Schlining, Lonny Lundsten, Kevin Barnard, Giovanna Sainz, Oceane Boulais, Megan Cromwell, Erin Butler, Benjamin Woodward, and Katherine L. C. Bell. FathomNet: A global image database for enabling artificial intelligence in the ocean. Scientific Reports, 12(1):15914, 2022. ISSN 2045-2322. doi: 10.1038/s41598-022-19939-2. URL https://doi.org/10.1038/s41598-022-19939-2.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015. URL https://arxiv.org/abs/1412.6980.

Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, and Ali Farhadi. Matryoshka representation learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 30233–30249. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ c32319f4868da7613d78af9993100e42-Paper-Conference.pdf.

Daniel Langenkämper, Martin Zurowietz, Timm Schoening, and Tim W. Nattkemper. BIIGLE 2.0 - browsing and annotating large marine image collections. Frontiers in Marine Science, 4, 2017. ISSN 2296-7745. doi: 10.3389/fmars.2017.00083. URL https://www.frontiersin.org/journals/marine-science/articles/ 10.3389/fmars.2017.00083.

Bohan Li, Xiao Xu, Xinghao Wang, Yutai Hou, Yunlong Feng, Feng Wang, Xuanliang Zhang, Qingfu Zhu, and Wanxiang Che. Semantic-guided generative image augmentation method with diffusion models for image classification. In Proceedings ofthe Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications ofArtificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’24/IAAI’24/EAAI’24. AAAI Press, 2024. ISBN 978-1-57735-887-9. doi: 10.1609/aaai.v38i4.28084. URL https://doi.org/10.1609/aaai.v38i4.28084.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19730–19742. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/li23q.html.

Sanne Lin, Flavius Frasincar, and Jasmijn Klinkhamer. Hierarchical deep learning for multi-label imbalanced text classification of economic literature. Applied Soft Computing, 176:113189, 2025. ISSN 1568-4946. doi: https://doi.org/10.1016/j.asoc.2025.113189. URL https://www.sciencedirect.com/science/article/pii/ S1568494625005009.

Zachary C. Lipton, Charles Elkan, and Balakrishnan Naryanaswamy. Optimal thresholding of classifiers to maximize F1 measure. In Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2014), volume 8725 of Lecture Notes in Computer Science, pages 225–239, Germany, 2014. Springer. doi: 10.1007/978-3-662-44851-9\_15.

Chenxi Liu, Yongqiang Chen, Tongliang Liu, Mingming Gong, James Cheng, Bo Han, and Kun Zhang. Discovery of the hidden world with large language models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 102307–102365. Curran Associates, Inc., 2024. doi: 10.52202/079017-3249. URL https://proceedings.neurips.cc/paper\_ files/paper/2024/file/b99a07486702417d3b1bd64ec2cf74ad-Paper-Conference.pdf.

Rundong Liu, Wenhan Liang, Weijun Luo, Yuxiang Song, He Zhang, Ruohua Xu, Yunfeng Li, and Ming Liu. Recent advances in hierarchical multi-label text classification: A survey. CoRR, abs/2307.16265, 2023. doi: 10.48550/ARXIV.2307.16265. URL https://doi.org/10.48550/arXiv.2307.16265.

Scott C. Lowe, Benjamin Misiuk, Isaac Xu, Shakhboz Abdulazizov, Amit R. Baroi, Alex C. Bastos, Merlin Best, Vicki Ferrini, Ariell Friedman, Deborah Hart, Ove Hoegh-Guldberg, Daniel Ierodiaconou, Julia Mackin-McLaughlin, Kathryn Markey, Pedro S. Menandro, Jacquomo Monk, Shreya Nemani, John O’Brien, Elizabeth Oh, Luba Y. Reshitnyk, Katleen Robert, Chris M. Roelfsema, Jessica A. Sameoto, Alexandre C. G. Schimel, Jordan A. Thomson, Brittany R. Wilson, Melisa C. Wong, Craig J. Brown, and Thomas Trappenberg. BenthicNet: A global compilation of seafloor images for deep learning applications. Scientific Data, 12(1):230, 2025. ISSN 2052-4463. doi: 10.1038/s41597-025-04491-1. URL https://doi.org/10.1038/s41597-025-04491-1.

Sachit Menon and Carl Vondrick. Visual classification via description from large language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= jlAjNL8z5cs.

Vishwesh Nath, Wenqi Li, Dong Yang, Andriy Myronenko, Mingxin Zheng, Yao Lu, Zhijian Liu, Hongxu Yin, Yucheng Tang, Pengfei Guo, Can Zhao, Ziyue Xu, Yufan He, Greg Heinrich, Yee Man Law, Benjamin Simon, Stephanie Harmon, Stephen Aylward, Marc Edgar, Michael Zephyr, Song Han, Pavlo Molchanov, Baris Turkbey, Holger Roth, and Daguang Xu. VILA-M3: Enhancing vision-language models with medical expert knowledge, 2025. URL https://arxiv.org/abs/2411.12915.

Rodolfo M. Pereira, Yandre M.G. Costa, and Carlos N. Silla Jr. Toward hierarchical classification of imbalanced data using random resampling algorithms. Information Sciences, 578:344–363, 2021. ISSN 0020-0255. doi: https://doi.org/10.1016/j.ins.2021.07.033. URL https://www.sciencedirect.com/science/article/pii/ S0020025521007234.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 8748–8763. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/radford21a.html.

Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 12116–12128. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/file/ 652cf38361a209088302ba2b8b7f51e0-Paper.pdf.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "why should i trust you?": Explaining the predictions of any classifier. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, page 1135–1144, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450342322. doi: 10.1145/2939672.2939778. URL https://doi.org/10.1145/2939672.2939778.

Dvir Samuel, Rami Ben-Ari, Simon Raviv, Nir Darshan, and Gal Chechik. Generating images of rare concepts using pre-trained diffusion models. In Proceedings ofthe Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications ofArtificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’24/IAAI’24/EAAI’24. AAAI Press, 2024. ISBN 978-1-57735- 887-9. doi: 10.1609/aaai.v38i5.28270. URL https://doi.org/10.1609/aaai.v38i5.28270.

Allie M. Scovil, Heather Breeze, Jessica Wingfield, Sarah Marteinson, Alexa Goodman, and Noreen E. Kelly. Effects of marine debris on Atlantic Canadian species and ecosystems. Canadian Technical Report of Fisheries and Aquatic

Sciences 3662, Fisheries and Oceans Canada, Maritimes Region, Bedford Institute of Oceanography, Dartmouth, Nova Scotia, 2025.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 31210–31227. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/shi23a.html.

Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3. arXiv, 2025. doi: 10.48550/arXiv.2508.10104. URL https://arxiv.org/abs/2508.10104.

Brandon Trabucco, Kyle Doherty, Max A Gurinas, and Ruslan Salakhutdinov. Effective data augmentation with diffusion models. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=ZWzUA9zeAg.

Grigorios Tsoumakas and Ioannis Katakis. Multi-label classification: An overview. International Journal of Data Warehousing and Mining, 3:1–13, 09 2009. doi: 10.4018/jdwm.2007070101.

Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The iNaturalist species classification and detection dataset, 2018. URL https://arxiv.org/abs/ 1707.06642.

Jonatas Wehrmann, Ricardo Cerri, and Rodrigo Barros. Hierarchical multi-label classification networks. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 5075–5084. PMLR, 10–15 Jul 2018. URL https:// proceedings.mlr.press/v80/wehrmann18a.html.

Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, and Tengyu Ma. Larger language models do in-context learning differently, 2023. URL https: //arxiv.org/abs/2303.03846.

Yuk-Kwan Wong, Tuan-An To, Jipeng Zhang, Ziqiang Zheng, and Sai-Kit Yeung. MarineEval: Assessing the marine intelligence of vision-language models, 2025. URL https://arxiv.org/abs/2512.21126.

WoRMS Editorial Board. World Register of Marine Species. Available from https://www.marinespecies.org at VLIZ, 2026. Accessed 2026-08-25. https://doi.org/10.14284/170.

Isaac Xu, Benjamin Misiuk, Scott C. Lowe, Martin Gillis, Thomas Trappenberg, and Craig J. Brown. Hierarchical multilabel classification with missing information for benthic habitat imagery. In 2024 International Joint Conference on Neural Networks (IJCNN), pages 1–10, 2024. doi: 10.1109/IJCNN60899.2024.10650176.

Isaac Xu, Martin Gillis, Ayushi Sharma, Benjamin Misiuk, Craig J. Brown, and Thomas Trappenberg. Improving detection of rare nodes in hierarchical multi-label learning. Transactions on Machine Learning Research, 2026. ISSN 2835-8856. URL https://openreview.net/forum?id=hf4zEWWIvE.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv, 2025. URL https://arxiv.org/abs/2505.09388.

Sahiti Yerramilli, Jayant Sravan Tamarapalli, Tanmay Girish Kulkarni, Jonathan Francis, and Eric Nyberg. Semantic augmentation in images using language, 2025. URL https://arxiv.org/abs/2404.02353.

Matthew D. Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Computer Vision – ECCV 2014, pages 818–833, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10590-1.

Tianyi Zhang, Weiming Zhi, Joshua Mangelson, and Matthew Johnson-Roberson. Infinite leagues under the sea: Photorealistic 3D underwater terrain generation by latent fractal diffusion models, 2025. URL https://arxiv. org/abs/2503.06784.

## A Technical Details

Table 4: VLM Inference Diagnostics. Throughput, GPU memory allocation, and KV cache overhead for the variable proposal and captioning stages. Throughput is reported as mean ± std across batches. Dataset sizes differ between tasks due to filtering (variable proposal operates on the training partition and captioning operates on the full dataset).
<table><tr><td>Task</td><td>Dataset</td><td>Dataset Size [samples]</td><td>Throughput [samples/s]</td><td>Peak Alloc [GB] (mean/max)</td><td>Peak Reserved [GB] (mean/max)</td><td>KV Overhead [GB]</td></tr><tr><td rowspan="3">Var. Proposal</td><td>BN-Sub</td><td>102,118</td><td> $5 . 7 3 \pm 0 . 2 2$ </td><td>19.09 / 19.23</td><td>21.61 / 22.75</td><td>2.75</td></tr><tr><td>BN-Bio</td><td>105,133</td><td> $5 . 6 1 \pm 0 . 4 0$ </td><td>19.02 / 19.21</td><td>20.64 /21.15</td><td>2.68</td></tr><tr><td>FN</td><td>8,980</td><td> $1 . 6 5 \pm 0 . 1 0$ </td><td>21.57 / 21.94</td><td>22.78 / 22.88</td><td>5.23</td></tr><tr><td rowspan="3">Captioning</td><td>BN-Sub</td><td>132,639</td><td> $0 . 3 8 \pm 0 . 0 4$ </td><td>17.06 / 17.13</td><td>21.15 /21.15</td><td>0.72</td></tr><tr><td>BN-Bio</td><td>142,637</td><td> $0 . 2 9 \pm 0 . 0 1$ </td><td>17.04 / 17.12</td><td>21.15 /21.15</td><td>0.70</td></tr><tr><td>FN</td><td>9,305</td><td> $2 . 0 9 \pm 0 . 1 7$ </td><td>18.90 / 19.17</td><td>22.80 / 22.80</td><td>2.56</td></tr></table>

Table 5: Training Model Diagnostics. Trainable parameters, throughput (mean ± std across batches), and GPU memory usage for each model variant.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td rowspan="2">Num. Classes</td><td rowspan="2">Batch Size</td><td rowspan="2">Trainable Params.</td><td rowspan="2">Throughput [samples/s]</td><td rowspan="2">Model Mem [GB]</td><td rowspan="2">Peak Alloc. [GB] (mean/max)</td><td rowspan="2">Peak Reserved [GB] (mean/max)</td></tr><tr><td></td></tr><tr><td rowspan="5">BN-Sub</td><td>Txt</td><td>23</td><td>64</td><td>19M</td><td> $9 8 5 . 9 6 \pm 5 0 . 7 7$ </td><td>0.09</td><td>2.57 / 2.62</td><td>3.81 / 3.88</td></tr><tr><td>Img</td><td>23</td><td>64</td><td>19M</td><td> $1 , 5 7 5 . 7 4 \pm 6 4 . 1 4$ </td><td>0.07</td><td>1.64 / 1.64</td><td>1.74 / 1.74</td></tr><tr><td>Img + Txt</td><td>23</td><td>64</td><td>19M</td><td> $1 , 2 8 1 . 5 8 \pm 5 9 . 6 4$ </td><td>0.09</td><td>1.86 / 1.87</td><td>2.11 /2.11</td></tr><tr><td>Router</td><td>23</td><td>64</td><td>10M</td><td> $1 , 0 4 6 . 1 4 \pm 6 5 . 8 8$ </td><td>0.26</td><td>1.28 / 1.30</td><td>2.96 / 3.01</td></tr><tr><td>Router†</td><td>23</td><td>64</td><td>66M</td><td> $4 4 3 . 7 9 \pm 2 4 . 7 0$ </td><td>0.26</td><td>5.44 / 5.49</td><td>7.09 / 7.16</td></tr><tr><td rowspan="5">BN-Bio</td><td>Txt</td><td>293</td><td>64</td><td>19M</td><td> $9 8 0 . 6 1 \pm 1 2 1 . 9 8 $ </td><td>0.09</td><td>2.52 / 2.58</td><td>3.71 / 3.78</td></tr><tr><td>Img</td><td>293</td><td>64</td><td>19M</td><td> $1 , 5 9 6 . 0 1 \pm 1 3 4 . 2 4$ </td><td>0.09</td><td>1.65 / 1.65</td><td>1.76 / 1.76</td></tr><tr><td> $\mathrm { I m g + T x t }$ </td><td>293</td><td>64</td><td>19M</td><td> $1 , 3 0 6 . 8 7 \pm 5 0 . 5 0$ </td><td>0.09</td><td>1.86 / 1.87</td><td>2.18 / 2.20</td></tr><tr><td>Router</td><td>293</td><td>64</td><td>11M</td><td> $1 , 0 5 3 . 1 4 \pm 4 4 . 2 1$ </td><td>0.28</td><td>1.28 / 1.30</td><td>2.91 / 2.96</td></tr><tr><td>Router†</td><td>293</td><td>64</td><td>69M</td><td> $4 4 6 . 3 0 \pm 9 . 7 4$ </td><td>0.28</td><td>5.42 / 5.48</td><td>7.06 / 7.15</td></tr><tr><td rowspan="5">FN</td><td>Txt</td><td>192</td><td>64</td><td>19M</td><td> $9 9 6 . 8 1 \pm 8 2 . 1 7$ </td><td>0.09</td><td>2.57 / 2.97</td><td>4.53 / 4.66</td></tr><tr><td>Img</td><td>192</td><td>64</td><td>19M</td><td> $1 , 6 3 4 . 5 2 \pm 5 1 . 5 2$ </td><td>0.07</td><td>1.65 / 1.65</td><td>1.74 / 1.74</td></tr><tr><td>Img + Txt</td><td>192</td><td>64</td><td>19M</td><td> $1 , 3 1 0 . 3 8 \pm 3 9 . 9 7$ </td><td>0.09</td><td>1.87 / 1.94</td><td>2.15 / 2.16</td></tr><tr><td>Router</td><td>192</td><td>64</td><td>11M</td><td> $1 , 0 0 2 . 5 8 \pm 1 0 2 . 5 9$ </td><td>0.27</td><td>1.29 / 1.43</td><td>3.56 / 3.65</td></tr><tr><td>Router†</td><td>192</td><td>64</td><td>68M</td><td> $4 5 0 . 1 8 \pm 1 7 . 9 7$ </td><td>0.27</td><td>5.45 / 5.86</td><td>7.22 / 7.35</td></tr></table>

Table 6: Training Hyperparameters. Learning rates and training epochs across models.
<table><tr><td>Dataset</td><td>Model</td><td>Learning Rate</td><td>Num. Epochs</td></tr><tr><td rowspan="5">BN*</td><td>Txt</td><td>3e-5</td><td>35</td></tr><tr><td>Img</td><td>1e-4</td><td>35</td></tr><tr><td> $\mathrm { I m g + T x t }$ </td><td>1e-4</td><td>35</td></tr><tr><td>Router</td><td>1e-4</td><td>35</td></tr><tr><td>Router†</td><td>1e-4</td><td>35</td></tr><tr><td rowspan="5">FN</td><td>Txt</td><td>3e-5</td><td>35</td></tr><tr><td>Img</td><td>1e-4</td><td>15</td></tr><tr><td> $\mathrm { I m g + T x t }$ </td><td>1e-4</td><td>15</td></tr><tr><td>Router</td><td>1e-4</td><td>8</td></tr><tr><td>Router†</td><td>1e-4</td><td>8</td></tr></table>

Applies to both BN-Sub and BN-Bio.

Table 7: Complementarity of Frozen Embeddings. Independence ratio P(text correct image wrong) / P(text correct) by hierarchy level, from k-NN classifiers on the frozen image and text embeddings. iNaturalist level 1 is excluded (image accuracy 99.6%).
<table><tr><td></td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td></tr><tr><td>BN-Sub</td><td>0.43</td><td>0.39</td><td>0.50</td><td></td><td></td></tr><tr><td>BN-Bio</td><td>0.68</td><td>0.27</td><td>0.33</td><td>0.29</td><td></td></tr><tr><td>iNaturalist</td><td>excl.</td><td>0.61</td><td>0.63</td><td>0.64</td><td>0.58</td></tr></table>

## A.1 Attention-Based Pooling

We consider two architectures for producing a fixed-dimensional representation from variable-length token sequences, both sharing a common structure consisting of a contextual encoding stage followed by learned query attention pooling. In the self-attention pooling variant, given an input sequence $\mathbf { X } \in \breve { \mathbb { R } } ^ { n _ { \mathrm { b } } \times T \times d }$ of batch size $n _ { \mathrm { b } } .$ sequence length T, and embedding dimension d, we first apply multi-headed (four heads) self-attention with residual connections and layer normalisation (Ba et al., 2016), followed by a position-wise feed-forward network, yielding a contextually enriched sequence $\tilde { \mathbf { X } } \in \mathbb { R } ^ { n _ { \mathrm { b } } \times T \times d }$ . In the cross-attention fusion variant, two distinct token sequences are available, a query sequence $\mathbf { X } _ { q } \in \mathbb { R } ^ { n _ { \mathrm { b } } \times N \times d }$ (image embeddings) and a key-value sequence $\mathbf { X } _ { k v } \in \mathbb { R } ^ { n _ { \mathrm { b } } \times \hat { M } \times d }$ (text embeddings), and we replace the self-attention encoder with an analogous cross-attention layer in which $\mathbf { X } _ { q }$ attends over ${ \bf X } _ { k v } .$ , followed by the same residual connections, layer normalisation, and feed-forward network, producing a fused sequence $\tilde { \mathbf { X } } _ { q } \in \mathbb { R } ^ { n _ { \mathrm { b } } \times N }$ ×d that integrates information from ${ \bf X } _ { k v }$ into the positions of $\mathbf { X } _ { q } .$ . In both cases, the final pooling stage is identical: a single learnable query parameter $\mathbf { q } \in \mathbb { R } ^ { d }$ attends over the encoded sequence via multi-head cross-attention, followed by layer normalisation, yielding a fixed-dimensional representation $\mathbf { z } \in \bar { \mathbb { R } } ^ { d }$ . Padding masks are propagated through all attention stages to ensure that padded positions contribute neither to contextual encoding nor to the final pooled representation. The resulting vector z is subsequently passed to an HML constrained feed-forward classifier.

## A.2 Gated Router

Given image, text, and cross-modal streams, we extract pooled feature vectors $\mathbf { z } _ { \mathrm { i m g } } , \mathbf { z } _ { \mathrm { t x t } } , \mathbf { z } _ { \mathrm { c r o s s } } \in \mathbb { R } ^ { d }$ and pass each through a dedicated classification head for each stream. This yields per-stream logit matrices stacked into a tensor of shape $( n _ { \mathrm { b } } , 3 , | \nu | )$ . A gating network ingests the concatenated features $[ { \bf z } _ { \mathrm { i m g } } ; { \bf z } _ { \mathrm { t x t } } ; { \bf z } _ { \mathrm { c r o s s } } ] \in \mathbb { R } ^ { 3 d }$ and produces persample, per-node softmax weights over the three modes. The gate applies layer normalisation (Ba et al., 2016), a linear projection from 3d to d, GELU activation with dropout, and a final linear layer from d to 3|V|, reshaped to $( n _ { \mathrm { b } } , | \nu | , 3 )$ and softmax-normalised along the last axis. The fused logit for node v is then

$$
l _ { v } = \sum _ { s \in \{ \mathrm { i m g , t x t , c r o s s } \} } g _ { v } ^ { ( s ) } \cdot l _ { v } ^ { ( s ) } ,\tag{5}
$$

where $g _ { v } ^ { ( s ) }$ is the gate weight and $l _ { v } ^ { ( s ) }$ is the stream logit. Sigmoid activation and the hierarchical constraint from C-HMCNN are applied after blending.

## A.3 Diagnostics

All diagnostic experiments were conducted on a single NVIDIA GeForce RTX 4090 GPU. Computational costs for VLM inference are reported in Table 4, and training diagnostics are provided in Table 5. We set the embedding dimension d to 1024 for Qwen3-Embedding-8B (which uses Matryoshka representations (Kusupati et al., 2022)), matching the output dimensionality of DINOv3’s ViT-H backbone. Each HML classifier head consists of a 3-layer MLP with hidden dimension 1024, GELU activations (Hendrycks and Gimpel, 2016), and dropout of 0.7, mapping from the 1024-dimensional embedding space to the label space R<sup>|V|</sup>. We optimise with Adam (Kingma and Ba, 2015) using a weight decay of 1e−4 across all experiments. Learning rates and training epochs are reported in Table 6. Training epochs were selected based on validation loss curves. BenthicNet’s larger dataset was less sensitive to overfitting, permitting 35 epochs to convergence, whereas FathomNet required more careful tuning to mitigate overfitting.

Table 8: Caption Sources. Independence ratio (as in Table 7) on BN-Bio (OOD test, 20,000 samples) for five caption sources, each embedded with Qwen3-Embedding-8B. Lower values indicate greater redundancy with the image embedding.
<table><tr><td>Captions</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td></tr><tr><td>Elicited (ours)</td><td>0.68</td><td>0.27</td><td>0.33</td><td>0.29</td></tr><tr><td>Keyword prose (ours)</td><td>0.64</td><td>0.27</td><td>0.33</td><td>0.32</td></tr><tr><td>Description</td><td>0.48</td><td>0.27</td><td>0.34</td><td>0.33</td></tr><tr><td>Visual</td><td>0.41</td><td>0.27</td><td>0.33</td><td>0.28</td></tr><tr><td>Plain</td><td>0.36</td><td>0.26</td><td>0.32</td><td>0.27</td></tr></table>

## B Captioning Embeddings

Table 7 reports the frozen-embedding probe summarised in subsection 4.3 by hierarchy level. iNaturalist level 1 is excluded because image accuracy there reaches 99.6%, leaving a conditioning set of only ∼89 samples. We then ask whether this complementarity depends on the captions being built on the extracted variables.

We compare the elicited captions against four alternatives generated by the same VLM for the same images: a plain caption, a visual description restricted to shapes, colours and textures, a description of the organisms and seafloor, and the same description additionally asked to relate the image to the extracted variables (Appendix D). At the coarsest level of the hierarchy, the captions built on the extracted variables are markedly less redundant with the image than the generic captions (Table 8). The description and keyword prose prompts differ only by the reference to the extracted variables, and this addition alone accounts for most of the gap. At deeper levels the five sources are comparable; however, we note that these deeper levels are where we expect text to be less useful in general.

## C Dataset Details

## C.1 BenthicNet

BenthicNet (Lowe et al., 2025) is one of the largest publicly available ocean science datasets, comprising over 11 million benthic images aggregated from 2,058 constituent datasets spanning a wide variety of geographic regions, depth ranges, and imaging platforms, sourced from research groups and government agencies around the world. Its label taxonomy follows the Collaborative and Annotation Tools for Analysis of Marine Imagery (CATAMI) visual classification scheme (Althaus et al., 2015), and is unique in that annotations span both substrate and biota categories, facilitating both habitat and organism classification. We draw from it its full set of annotated imagery (188,679 images), forming two overlapping subsets for substrate and biota (BN-Sub and BN-Bio, respectively), which we treat as independent classification tasks. BN-Sub contains 23 nodes while BN-Bio contains 293. BenthicNet’s test partition is geospatially subsampled, meaning test images originate from regions unseen during training. We refer to this innate partition as the out-of-domain (OOD) test set, and additionally construct an in-domain (ID) test set through random stratified sampling (of 20% of the data) for evaluation under more conventional conditions. The combination of global geographic coverage and diversity across both biota and substrate categories makes BenthicNet a suitable testbed for HML methods that must generalise across heterogeneous visual conditions.

## C.2 FathomNet (2025)

FathomNet (2025) (FN) (FathomNet 2025) is a benthic biota classification competition dataset consisting of 9,305 full images and 24,487 regions-of-interest (ROIs). Full images may contain multiple ROIs. The dataset is not inherently hierarchical, despite using taxonomic annotations. Furthermore, it has been curated to be balanced across classes. We extracted a hierarchy of 192 nodes spanning phylum, class, order, family, genus, and species<sup>4</sup> from the World Register of Marine Species (WoRMS) (WoRMS Editorial Board, 2026). Subsequently, the classes in the dataset become leaf nodes, and the problem becomes HML (full-depth). While earlier imagery from FathomNet was contained in BenthicNet, the 2025 competition dataset is novel with a carefully curated OOD test set (Lowe et al., 2025; FathomNet 2025). As with BenthicNet, we use both the innate test partition and a random partition (of 20%) for OOD and ID evaluation respectively.

## C.3 iNaturalist

To demonstrate the generalisability of the proposed approach, experiments are also performed on the iNaturalist (iNat) (Van Horn et al., 2018) dataset. The iNat dataset encompasses a broader domain than BenthicNet, capturing terrestrial fauna, flora, and fungi. Additionally, it presents a distinct structural classification challenge. While the dataset inherently contains 10,000 distinct species classes, its biological taxonomic hierarchy allows for the construction of a hierarchical classification problem spanning over 16,000 nodes, characterised by mutually exclusive, single-label assignments. In adapting our framework for iNat, we replace MCLoss with a softmax over the 10,000 endpoints, marginalised up the hierarchy, so that each node receives the summed probability of its descendants and the hierarchical structure is preserved. Training occurs on the 500,000 samples from iNat (mini), and the 100,000 validation samples act as a test set (since the true test partition is not publicly available).

## D Prompt Templates

We use two prompt templates for variable proposal and captioning, and four further templates for the caption comparison in Appendix B. Where applicable, templates are parameterised by task-specific variables denoted with braces.

## D.1 Variable Proposal Prompt

Variable Proposal Prompt   
You are a marine benthic ecologist.   
This seafloor image has the following annotations: {annotations}.   
Use these annotations to guide which ecological properties would be most diagnostic for   
identifying the annotated categories.   
Task: Infer functional ecological properties that help distinguish between different   
{task}.   
Go beyond what a simple visual description would capture –- use ecological reasoning to   
infer properties from what you observe. Focus especially on diagnostic properties where   
ecological and world knowledge would help resolve highly specific or rare categories.   
Do NOT return simple visual descriptions (e.g., color, shape, texture). Do not restate the   
annotation categories themselves. Return up to 10 of the most diagnostic properties as 1–3   
lowercase words each, separated by commas. Return nothing else.

## D.2 Captioning Prompt

Captioning Prompt   
You are a marine benthic ecologist analyzing a seafloor photograph.   
Task: For each ecological property below, carefully reason about whether it is supported or   
contradicted by visible evidence of the {task} in this image. Connect what you SEE to what   
you KNOW about benthic ecology. Score each property:   
1 = visually supported   
0 = not applicable or visually contradicted   
Properties:   
{variables}   
Format:   
[property\_term]: [brief reasoning and justification (one sentence, \~5 words)] [score value]   
Adhere strictly to the format. Output one property per line.

Here, {annotations} denotes the ground-truth categories expressed in natural language, {task} specifies the ecological domain (e.g., “biota” or “substrate characteristics”), and {variables} is the alphabetically sorted set of all processed variables for the domain.

The following prompts generate the alternative captions compared in Appendix B. All four use the same VLM and decoding settings as the captioning prompt.

## Plain Caption Prompt

Describe the image in a single paragraph of about 150 words.

## Visual Caption Prompt

Describe in detail what is directly visible in the image: shapes, colours, textures. Write a single paragraph of about 150 words.

## Description Prompt

You are looking at a seafloor photograph.   
Task: Describe in detail what is visible in the image: the organisms present and the   
seafloor.   
Write a single paragraph of about 150 words. Return nothing else.

## Keyword Prose Prompt

You are looking at a seafloor photograph.   
Task: Describe in detail what is visible in the image: the organisms present and the   
seafloor. Where relevant, relate your description to the following ecological properties:   
{variables}.   
Write a single paragraph of about 150 words. Return nothing else.

## E Latent Extraction Details

This section provides the complete mathematical specification of the variable extraction pipeline summarised in subsection 3.1.

## E.1 Step 1: Node Frequency Prior

We compute the empirical distribution over nodes from the training annotations:

$$
p ( v ) = \frac { n _ { v } } { \sum _ { v ^ { \prime } \in \mathcal { V } } n _ { v ^ { \prime } } } , \quad \forall v \in \mathcal { V } .\tag{6}
$$

## E.2 Step 2: EM Decomposition

We define the observed annotations for a candidate $z _ { 2 , j } \in \mathbf { Z } _ { 2 }$ as a set of label-count pairs $\{ ( \mathcal { A } _ { k } , m _ { k } ) \}$ , where $\mathcal { A } _ { k } \subseteq \mathcal { V }$ is a set of co-occurring partial-depth paths within a single sample and $m _ { k }$ is the number of samples associated with $z _ { 2 , j }$ that carry annotation set $\mathcal { A } _ { k }$ . We partition these into single-path pairs $S _ { j } = \{ ( \mathcal { A } _ { k } , m _ { k } ) : | \bar { \mathcal { A } } _ { k } | = 1 \}$ } and multi-path pairs $\tilde { \mathcal { M } _ { j } } = \{ ( \mathcal { A } _ { k } , m _ { k } ) : | \mathcal { A } _ { k } | > \mathrm { \bar { 1 } } \}$ . Let $S _ { j } ( v ) = \{ k : \bar { ( A _ { k } , m _ { k } ) } \in \bar { S _ { j } , ~ v } \in \bar { A _ { k } } \}$ and $\mathcal { M } _ { j } ( v ) = \{ k : ( \mathcal { A } _ { k } , m _ { k } ) \in$ $\mathcal { M } _ { j } , \ v \in \mathcal { A } _ { k } \}$

Counts are initialised from unambiguous single-path instances:

$$
n _ { j } ^ { ( 0 ) } ( v ) = \sum _ { k \in S _ { j } ( v ) } m _ { k } .\tag{7}
$$

Multi-path counts are then iteratively distributed in proportion to current estimates:

$$
n _ { j } ^ { ( t + 1 ) } ( v ) = n _ { j } ^ { ( 0 ) } ( v ) \ + \ \sum _ { k \in \mathcal { M } _ { j } ( v ) } m _ { k } \cdot \frac { n _ { j } ^ { ( t ) } ( v ) } { \sum _ { v ^ { \prime } \in \mathcal { A } _ { k } } n _ { j } ^ { ( t ) } ( v ^ { \prime } ) } ,\tag{8}
$$

iterating until convergence:

$$
\operatorname* { m a x } _ { v \in \mathcal { V } } \left| n _ { j } ^ { ( t + 1 ) } ( v ) - n _ { j } ^ { ( t ) } ( v ) \right| < \tau .\tag{9}
$$

The resulting distribution over nodes for each $z _ { 2 , j }$ is:

$$
q _ { j } ( v ) = { \frac { n _ { j } ( v ) } { \sum _ { v ^ { \prime } \in \mathcal { V } } n _ { j } ( v ^ { \prime } ) } } .\tag{10}
$$

## E.3 Step 3: Smoothing and KL Scoring

Before computing divergence, we smooth each $q _ { j }$ towards the prior p by adding a pseudo-count volume equivalent to Laplace smoothing, distributed according to p:

$$
\tilde { q } _ { j } ( v ) = \frac { n _ { j } ( v ) + | \mathcal { V } | \cdot p ( v ) } { N _ { j } + | \mathcal { V } | } ,\tag{11}
$$

where $\begin{array} { r } { N _ { j } = \sum _ { v ^ { \prime } } n _ { j } ( v ^ { \prime } ) } \end{array}$ is the total count mass for $z _ { 2 , j }$ . This smoothing serves as a sample-size-aware regulariser. A variable supported by many annotated instances will have large $N _ { j }$ , rendering the pseudo-counts negligible and preserving the empirical distribution, whereas a variable supported by only a handful of instances will be pulled substantially towards the prior.

A key assumption underlies this scoring procedure. The empirical posterior $q _ { j } ( v )$ is constructed from the same annotation source from which the candidate variables were originally proposed, so high $q _ { j } ( v )$ reflects co-occurrence within the annotation pipeline rather than an independently observed predictive relationship. We assume that this co-occurrence is indicative of genuine discriminative value. In other words, a candidate $z _ { 2 , j }$ whose annotation mass concentrates on a node v does indeed capture a visual or semantic property informative for distinguishing v from other nodes. This assumption may not hold universally, but provides the working basis for using the KL divergence as a proxy for discriminative value. Candidates are then scored by their divergence from the prior,

$$
D _ { \mathrm { K L } } \big ( \tilde { q } _ { j } \parallel p \big ) = \sum _ { v \in \mathcal { V } } \tilde { q } _ { j } ( v ) \log \frac { \tilde { q } _ { j } ( v ) } { p ( v ) } ,\tag{12}
$$

and only candidates scoring $\geq 1$ are retained.

## E.4 Step 4: Deduplication and Selection

The candidate set emerging from $D _ { \mathrm { K I } }$ filtering undergoes three stages of refinement.

Distribution-based deduplication. Each surviving candidate is represented by its normalised distribution ${ \tilde { q } } _ { j } .$ Processing in descending KL order, a candidate is discarded if its cosine similarity with any previously retained distribution meets or exceeds a threshold $\tau _ { \mathrm { d i s t } }$

$$
\frac { \tilde { q } _ { j } \cdot \tilde { q } _ { k } } { \| \tilde { q } _ { j } \| \| \tilde { q } _ { k } \| } \geq \tau _ { \mathrm { d i s t } } .\tag{13}
$$

We use $\tau _ { \mathrm { d i s t } } = 0 . 9 9$

Coverage-based deduplication. Let ${ \mathcal { T } } ( z _ { 2 , j } ) \subseteq \mathbf { X }$ denote the set of training images whose proposed-variable field contains $z _ { 2 , j }$ . Candidates are processed in descending KL order. A candidate is discarded if its image set has Jaccard similarity at or above $\tau _ { \mathrm { c o v } }$ with any previously retained candidate:

$$
J ( \mathcal { T } ( z _ { 2 , j } ) , \mathcal { T } ( z _ { 2 , k } ) ) = \frac { | \mathcal { T } ( z _ { 2 , j } ) \cap \mathcal { T } ( z _ { 2 , k } ) | } { | \mathcal { T } ( z _ { 2 , j } ) \cup \mathcal { T } ( z _ { 2 , k } ) | } \ \ge \ \tau _ { \mathrm { c o v } } .\tag{14}
$$

We use $\tau _ { \mathrm { c o v } } = 0 . 8$

Per-node top-B selection. Surviving candidates are assigned to their modal node $v _ { j } ^ { * } = \arg \operatorname* { m a x } _ { v \in \mathcal { V } } n _ { j } ( v )$ , inducing per-node groups $\mathcal { Z } ( v ) = \{ z _ { 2 , j } : v _ { i } ^ { * } = v \}$ . A uniform budget B is applied to leaf nodes $\nu _ { L } \subseteq \nu$ , prioritising rarity. For each leaf node v with $\mathcal { Z } ( v ) \neq \emptyset$ , the selected subset is:

$$
S _ { v } = \mathrm { t o p } _ { B } \Big ( \mathcal { Z } ( v ) , \mathrm { K L } \big ( \tilde { q } _ { j } \| p \big ) \Big ) .\tag{15}
$$

The final variable set is $\textstyle \bigcup _ { v \in \mathcal { V } _ { L } } S _ { v }$

## F Limitations

## F.1 Variable Proposal and Captioning

The quality of proposed variables and captions is bounded by the VLM’s pretraining corpus. Domains or taxa that are poorly represented in training data may yield superficial or inaccurate variables, and the model’s familiarity is not uniform across the label space. Well-studied organisms (e.g., hard corals) are likely to be better detected than understudied groups (e.g., soft corals), introducing a systematic bias in caption informativeness.

The uniform per-node budget B and conservative variable selection reflect pragmatic constraints imposed by the limited token capacity of smaller VLMs. When provided with too many variables, models at this scale tend to anchor onto variable names without meaningful contextual processing (Shi et al., 2023; Wei et al., 2023). A larger VLM or a multi-pass captioning scheme could improve caption quality. Lastly, while it may only need to be performed once, the captioning cost may be quite high, as demonstrated in Table 4. Computation cost in this step can also vary significantly depending on the phrasing used in the prompt.

## F.2 Architecture and Training

The text and image embedding models (Qwen3-Embedding-8B and DINOv3 ViT-H respectively) are frozen throughout training. Their representation spaces are therefore not adapted to the target domain, and any distributional mismatch between pretraining data and benthic imagery is inherited by the downstream classifier. Fine-tuning these backbones could improve performance but would substantially increase computational requirements and reduce accessibility for domain-specialised labs with limited hardware.

The router models introduce additional parameters relative to the unimodal baselines (by a factor of 3, given each stream, and the router mechanism), and one variant (Router) loads pretrained pooling layers from unimodal runs. We therefore assess the text contribution with ablations that hold the architecture and parameter count fixed while removing the per-sample text signal (subsection 4.3; subsection G.2).

## F.3 Evaluation Scope

Our benthic experiments, our primary application focus, are restricted to two datasets (BenthicNet (Lowe et al., 2025) and FathomNet (Katija et al., 2022; FathomNet 2025)), which represent the few large-scale, globally diverse, and taxonomically broad benchmarks available in this field. Beyond the benthic domain, we evaluate only on iNaturalist. While the core methodology does not rely on domain-specific assumptions, its extension to other specialised domains, such as medical imaging, remains empirically undemonstrated and is left to future work.

## F.4 Evaluation Metric

The inverted HML score $d _ { \mathrm { i n v } }$ relies on a user-specified parameter $k _ { \mathrm { P } }$ that governs the severity of the cardinality penalty. Our choice of $k _ { \mathrm { P } } = 0 . 5$ represents a balanced default. Where the expected unannotated multi-label cardinality of a dataset is suspected, k<sub>P</sub> can be adjusted to reflect it.

Additionally, $d _ { \mathrm { i n v } }$ treats all edges in the hierarchy as having unit weight, which may not reflect the true semantic distance between nodes. For example, the distinction between two congeneric species is typically far smaller than that between two phyla, yet both correspond to two edge traversals in a taxonomic tree. Depth-specific edge weights could be incorporated where expert knowledge provides meaningful inter-node distances, but we do not explore this idea in this work.

## G Additional Ablations

## G.1 Zero-Shot VLM Classification

A natural question is whether the VLM can perform the classification directly, without the elicitation and training pipeline. We evaluate this on BenthicNet in two prompting regimes under greedy decoding, and report structural information rather than performance (Table 9).

Free-form output cannot be scored. Both macro $F _ { 1 }$ and $d _ { \mathrm { i n s } }$ require an ancestor-closed subset of V, whereas the VLM emits unconstrained text. Matching that text back to the scheme recovers 0.10 nodes per image on BN-Sub and 0.87 on BN-Bio, against ground truth cardinalities of 1.42 and 3.09. Only 0.5% and 3.3% of emitted terms correspond to any

Table 9: Zero-Shot VLM Classification on BenthicNet. Scheme conformance and prediction cardinality for free-form prompting and for prompting conditioned on the classification scheme.
<table><tr><td></td><td>BN-Sub</td><td>BN-Bio</td></tr><tr><td>images yielding no scheme-conformant term (free-form)</td><td>92.0%</td><td>77.6%</td></tr><tr><td>conformance when conditioned on the scheme</td><td>99.9%</td><td>98.0%</td></tr><tr><td>predicted / annotated labels per image (conditioned)</td><td>2.08</td><td>6.98</td></tr></table>

Table 10: Scale-Matched Noise Ablation. Median scores across seeds for the Router with clean text embeddings and retrained with scale-matched Gaussian noise in their place; p is a two-sided Wilcoxon signed-rank test over paired seeds. Since iNat has only five seeds, the lowest attainable p-value is 0.063.
<table><tr><td rowspan="2">dataset</td><td rowspan="2">seeds</td><td colspan="3">Macro  $F _ { 1 }$ </td><td colspan="3">Macro  $d _ { \mathrm { i n v } }$ </td></tr><tr><td>clean</td><td>noise</td><td>p</td><td>clean</td><td>noise</td><td>p</td></tr><tr><td>BN-Sub</td><td>10</td><td>0.678</td><td>0.241</td><td>0.002</td><td>0.654</td><td>0.006</td><td>0.002</td></tr><tr><td>BN-Bio</td><td>10</td><td>0.129</td><td>0.047</td><td>0.002</td><td>0.037</td><td>0.001</td><td>0.002</td></tr><tr><td>FathomNet</td><td>30</td><td>0.410</td><td>0.414</td><td>0.670</td><td>0.323</td><td>0.316</td><td>0.008</td></tr><tr><td>iNaturalist</td><td>5</td><td>0.549</td><td>0.190</td><td>0.063</td><td>0.770</td><td>0.388</td><td>0.063</td></tr></table>

CATAMI category, and 92.0% and 77.6% of images yield no scheme-conformant prediction at all. Any score reported under this regime would be a property of the matching rule rather than of the model.

Supplying the vocabulary in the prompt helps with conformance but worsens cardinality. Conformance rises to 99.9% on BN-Sub and 98.0% on BN-Bio, while predicted cardinality inflates to 2.08 and 6.98 times ground truth. On BN-Bio the failure is bimodal: the median image returns 6 categories, 6.3% return more than 54, and the worst case returns 148 against a single annotated endpoint. This is the regime $d _ { \mathrm { i n v } }$ penalises and micro-averaged AP tolerates (Appendix I): conformant predictions that share correct ancestors while carrying almost no discriminative information. It is also the reason a per-node budget B is necessary (Appendix E), since instability scales with the size of the vocabulary placed in the prompt.

Neither regime yields a usable classifier. Knowledge held in the weights does not make a specialised label space available as a prediction target, which motivates using the VLM to elicit properties rather than to predict nodes.

## G.2 Scale-Matched Noise Ablation

To separate the contribution of the text content from that of the additional input dimensions, we replace the text embedding with Gaussian noise of matched scale at a mixing coefficient of one, so that no original signal remains, and retrain under the identical Router protocol. Architecture, embedding dimensionality and trainable parameter count are held constant, and the model trains from the start on an uninformative stream, free to disregard it in favour of the image branch.

Results are given in Table 10. On both BenthicNet domains and on iNat the corrupted model degrades on both metrics at every seed, and falls below the image-only baseline of Table 1 and Table 2. Additional feature dimensions should not make a model worse than not having them, so a gain arising from dimensionality alone would not produce this. Thi ablation separates content from dimensionality. A comparison against other semantically valid captions is given in Appendix B.

On FathomNet the effect is significant but small, and macro $F _ { 1 }$ is unchanged. Two factors plausibly limit what the VLM can supply for this dataset: annotations given as Latin binomials rather than descriptive terms, and the limited representation of deep-sea organisms in general-purpose corpora relative to terrestrial fauna. We do not attempt to separate these.

## G.3 Descriptor-Matching Baseline

We compare against a descriptor-matching baseline following Menon and Vondrick (2023), which classifies images by their similarity to LLM-generated visual descriptors rather than to class names alone. Because the method produces flat outputs, we evaluate it both with and without the hierarchical constraint of section 3.

Table 11: Descriptor Matching on BN-Bio. Macro $F _ { 1 }$ and macro $d _ { \mathrm { i n v } }$ (%) of a DCLIP-style baseline (Menon and Vondrick, 2023), with and without the hierarchical constraint f<sub>CM</sub>, alongside the trained models. Descriptors are embedded with the CLIP ViT-L-14-quickgelu text encoder. DCLIP rows are single deterministic runs; trained rows are the ten-seed OOD values of Table 1.
<table><tr><td></td><td>Macro  $F _ { 1 }$ </td><td>Macro  $d _ { \mathrm { i n v } }$ </td></tr><tr><td>DCLIP, no constraint</td><td>5.8</td><td>0.12</td></tr><tr><td>DCLIP + fcm</td><td>5.8</td><td>0.17</td></tr><tr><td>Text-only (trained)</td><td>7.4</td><td>2.5</td></tr><tr><td>Image-only (trained)</td><td>11.4</td><td>4.4</td></tr><tr><td>Router (trained)</td><td>13.2</td><td>5.9</td></tr><tr><td>Router† (trained)</td><td>14.3</td><td>6.4</td></tr></table>

For each of the 293 BN-Bio nodes, the VLM used elsewhere in our pipeline generates six visual descriptors, prompted with the node and its position in the hierarchy and restricted to appearance, giving 1,758 descriptors in total. Each descriptor is embedded once with the CLIP text encoder and scored against every image by cosine similarity, and a node’s score is the mean over its six descriptors. Normalisation and per-node thresholds are fitted on the 105,133 training images and applied unchanged to the 37,504 test images, so the baseline receives the same per-node thresholding as every model in Table 1.

The baseline reaches roughly half the macro $F _ { 1 }$ of the trained image branch and under a tenth of its $d _ { \mathrm { i n v } }$ (Table 11). Adding the hierarchical constraint leaves macro $F _ { 1 }$ unchanged and raises macro $d _ { \mathrm { i n v } }$ from 0.12 to 0.17, while the sample-mean $d _ { \mathrm { i n v } }$ stays at 0.11, so the gain falls on nodes that carry little weight in the sample mean.

The descriptors account for much of this. Of the 1,758 generated, 22.5% are verbatim duplicates of a descriptor issued for another node: smooth or slightly rough surface is assigned to 35 nodes and pale beige to brownish hue to 47. Duplication is most common between siblings, where 24% of the 769 sibling pairs share an identical descriptor. For example, Macroalgae > Laminate > Brown and > Red receive five identical descriptors out of six, none of which mentions colour, although the prompt supplied both hierarchy paths and asked for distinguishing features. Node scores reflect this overlap: sibling nodes correlate at r = 0.825, against 0.469 for non-siblings, and the correlation among nodes at the same depth rises from 0.36 at depth one to 0.95 at depth six. In this domain, appearance-based descriptors separate coarse categories but not the morphological distinctions on which CATAMI subdivides.

Our variables also operate at the clade level; the difference lies in redundancy. A visual descriptor names something a frozen visual encoder already extracts, whereas the properties elicited in section 3 are not recoverable from the image alone. This is consistent with Appendix B, where, at the coarsest level of the hierarchy, generic image descriptions are markedly more redundant with the image embedding than captions built on the elicited variables.

## H BN-Bio (OOD) Case Study

The correlation analysis between proposed variables and text advantage (Figure 5a) reveals that behavioural and habitat traits (ambush, high mobility, benthic habitat) are significantly positively correlated with text modality performance gains, while morphological plant traits (light interception, leafmorphology) exhibit significant negative correlations. However, examining text advantage at the level of individual nodes (Figure 5b) reveals that a variable annotated as absent (Property: [0]) for a given sample may still contribute positively to the discriminability of relevant classes. The explicit negation of a property, along with its caption, may serve as part of a combinatorial signature that distinguishes a node from its siblings.

Despite this argument, we emphasise that these correlations are observational and do not establish a causal mechanism by which any individual variable improves classification. While connections can be manually verified in specific cases, we cannot generalise this reasoning to all variable–node pairs. Some variables may contribute indirectly through interactions with other annotations, or may serve primarily as prompt structure that encourages the captioning VLM to attend to relevant image regions. Disentangling these effects would require controlled ablations at the individual variable level, beyond the scope of this work.

![](images/c990904eea3e273971f167e28813586baa2deefef40cb41346bfa84e1295e599.jpg)  
(a) BN-Sub (OOD)

![](images/ca4626870262b394a8c421f4d744da994b1caf12c54b50f5de640022d5984cd2.jpg)  
(b) BN-Bio (OOD)  
Figure 4: Modality Information Exclusivity. Displayed are the top-5 nodes in which text holds the greatest advantages over vision in the BenthicNet datasets in terms of $d _ { \mathrm { i n v } }$ . The bars depict the percentage of samples for each node in which the models perform less than, equal to, or greater than vision.

Table 12: Gate Weights under Text Ablation on BN-Bio. Text gate weight $w _ { \mathrm { t x t } }$ with clean captions and with every caption replaced by the mean embedding, the relative change, and the number of seeds in which the value decreases.
<table><tr><td></td><td>clean</td><td>constant text</td><td> $\Delta$ </td><td>seeds lower</td></tr><tr><td>mean  $w _ { \mathrm { t x t } }$ </td><td>0.5658</td><td>0.5455</td><td>-3.6%</td><td>5/5</td></tr><tr><td>per-sample std. of  $w _ { \mathrm { t x t } }$ </td><td>0.1315</td><td>0.1306</td><td>-0.68%</td><td>一</td></tr></table>

## H.1 Node-Wise Analysis

To characterise where the text modality contributes, we compare the text and image branches node by node on every BN-Bio node with at least 20 positive test samples (171 nodes).

On its own, the text branch has lower $F _ { 1 }$ than the image branch at 157 of the 171 nodes. The gap is recall-driven at 105 of these nodes and precision-driven at $5 2 ,$ so captions more often omit a discriminating property than assert an incorrect one. This matches the clade-level character of the elicited variables: filter feeding, for example, separates sponges from other taxa, placing a prediction within the correct subtree without resolving the morphotype within it.

The two metrics diverge in a consistent direction. At 75 of the 171 nodes, the text branch lands closer to the target on $d _ { \mathrm { i n v } }$ while failing to identify the node on $F _ { 1 } ;$ the reverse holds at only 6. Text therefore contributes hierarchical proximity where it does not contribute discrimination.

In the fused model, routing recovers the text branch’s $F _ { 1 }$ deficit at 130 of the 157 nodes, and at all 21 measurable BN-Sub nodes, with a median gain of 0.018 $F _ { 1 }$ over the image branch. At the remaining 27 nodes, the median cost is 0.005 $F _ { 1 }$ . These nodes are concentrated in sponges, which make up 24.6% of the nodes examined but 55.6% of those not recovered, a 2.3-fold concentration. CATAMI subdivides sponges by morphology, which our proposal prompt excludes by design (subsection D.1), so these nodes fall where the clade-level character of the variables predicts.

## H.2 Gate Weights

The router assigns per-node weights over the image, text and cross-attention branches. Table 12 reports the mean text weight and its per-sample standard deviation with and without per-sample text information.

The gate does not ignore the text branch. On BN-Bio, the mean text weight is 0.57. When every caption is replaced by the same payload, the mean text weight falls by 3.6% (0.566 → 0.546), lower at every seed, while its per-sample standard deviation changes by less than 1%. The sample-to-sample variation of the gate is therefore driven mainly by the image side of its input rather than by the content of individual captions.

![](images/2c5a9c765d0508f7bd045375d269822249afe9f6122fda2b0d924f0a8933eb13.jpg)

![](images/66b2b6bf028145ce4b4d445146cc2131689d976be9d9069fa777c0705bd5aa10.jpg)  
Figure 5: Variable Correlations with Text Modality Advantage. (a) Pearson correlation between each variable’s positive annotation in captions and text advantage across all nodes. Dark shading indicates statistical significance $( p < 0 . 0 5 )$ . (b) For each top-correlated variable (sorted by r), we select a representative class where the variable benefits text when it is positive [1] (blue) and one where it is negative [0]. Sample coverage is noted in brackets. A soft<sup>23</sup> diversity penalty discourages repeated selection of the same class across properties. Since the majority of samples have a positive annotation for “benthic habitat”, it is the only variable without a negative bar.

![](images/41f4da8227f41d76396db89fd18d45e0c15512399dfb20294cf7b457f4f0fe54.jpg)  
(a) False Negative (FN) Rate

![](images/69b35ca9941e8f4f60f4f31dc776192f3591a17ce73dda6a9e74d5d8f7996b04.jpg)  
(b) False Positive (FP) Rate  
Figure 6: Random Noise Profile. The behaviour of the random noise added is such that it slightly increases the FP rate at higher-level nodes, while substantially increasing the FN rate at deeper nodes, simulating a more realistic type of HML model degradation.

These weights describe how much the fused output draws on each branch, not how the text branch transforms its input. Because the gate mixes branch logits, it does not show how the text embedding is used within the model; that contribution is measured directly by the ablations in subsection 4.3 and subsection G.2.

## I HML Metrics Noise Experiment

Algorithm 1 Random Hierarchy Generation   
Require: Number of nodes $| \nu |$   
Ensure: Descendant matrix $\mathbf { D } ^ { ' } \in \{ 0 , 1 \} ^ { | \mathcal { V } | \times | \mathcal { V } | }$   
1: Select number of root nodes $n _ { r } \in [ 1 , \operatorname* { m a x } ( 1 , \lfloor \lvert \mathcal { V } \rvert / 1 0 0 \rfloor ) ]$ uniformly at random   
2: Initialise a forest with root nodes $v = 0 , \ldots , n _ { r } - 1$   
3: for each remaining node $v = n _ { r } , \ldots , | \mathcal { V } | - 1$ do   
4: Assign v as a child of a node chosen uniformly at random from $\{ 0 , \ldots , v - 1 \}$   
5: end for   
6: $\mathbf { D } \gets \mathbf { 0 } ^ { | \nu | \times | \nu | }$   
7: for each node $v \in \mathcal V$ do   
8: Set $D _ { v , u } \gets 1$ for all descendants u of v (including v itself)   
9: end for   
10: return D

A natural notion of a “random” model in HML is one that predicts each node independently with uniform probability, analogous to the random baseline in multi-class classification. Such a model produces predictions with expected cardinality near $| \nu | / 2$ , where the number of combinatorial states is greatest. Crucially, the hierarchical constraint forces false positives to propagate upward since activating a node requires activating all of its ancestors. Consequently, higher-level nodes are disproportionately likely to be active. This stacking effect means that even moderate noise inflates cardinality rapidly. Both AP and $F _ { 1 }$ decay under this noise regime, and do so more steeply as hierarchy size grows, since the propagation affects a larger number of ancestor nodes in absolute terms. Recall, however, remains high throughout, because the noise mechanism adds false positives without removing true positives. Instead, the decrease in $F _ { 1 }$ is driven mostly by precision. By contrast, $d _ { \mathrm { i n v } }$ falls off sharply under this noise, owing to its direct penalisation of excessive prediction cardinality, correctly reflecting that a model whose cardinality has inflated toward $| \nu | / 2$ retains little predictive value. The key point is not that one metric is right and the other wrong here, but that this regime is the only one in which AP behaves acceptably.

Indeed, this type of performance degradation is almost never relevant in real-world scenarios. HML classification is a difficult problem because natural imbalances arise at deeper nodes (Xu et al., 2026; Pereira et al., 2021), with detection and recall falling in these specific and rare regions. In contrast, higher-level nodes, being more frequently observed, are rarely confused with one another. Instead, a realistic noise environment sharply increases the rate of false negatives with node depth, and only slightly increases false positives, which decrease with depth. Following these guidelines, we simulate data by generating synthetic hierarchies as in Algorithm 1, with fictitious data generated using Algorithm 2, and add noise with Algorithm 3. The effects of the added noise are shown in Figure 6. Our synthetic experiment targets an expected cardinality of three (to simulate the low cardinality and sparsity of benthic data). We tested five distinct hierarchies at each of five sizes (25 hierarchies in total), with 500 data points per hierarchy. The noise levels spanned 0.0 to 1.0 in 0.1 increments.

Algorithm 2 Random Hierarchical Data Generation   
Require: Number of nodes |V|, number of samples N, descendant matrix D, sparsity s   
Ensure: Data matrix $\mathbf { Y } \in \{ 0 , \dot { 1 } \} ^ { N \times | \mathcal { V } | }$   
1: Define neighbourhood $\ r _ { \ r } \dot { \delta } = \ r \operatorname* { m a x } ( 1 , \lfloor s / 2 \rfloor )$   
2: $\mathbf { Y } \gets \mathbf { 0 } ^ { N \times | \mathcal { V } | }$   
3: for each sample $i = 1 , \ldots , N$ do   
4: Sample k uniformly from $\left[ \operatorname* { m a x } ( 1 , s - \delta ) \right.$ , min $( | \nu | , s + \delta ) ]$   
5: Select k nodes uniformly at random from V   
6: Activate selected nodes: $\mathbf { r } \in \{ 0 , 1 \} ^ { | \nu | }$   
7: Propagate through hierarchy: $\mathbf { \dot { Y } } _ { i } \gets \mathrm { c l i p } ( \mathbf { D } \cdot \mathbf { r } , 0 , 1 )$ ▷ Enforce ancestor activation   
8: end for   
9: return Y

Algorithm 3 Hierarchical Noise Generation   
Require: Ground-truth matrix $\mathbf { Y } \in \{ 0 , 1 \} ^ { N \times | \mathcal { V } | } .$ , descendant matrix D, noise level $\eta \in [ 0 , 1 ]$   
Ensure: Noisy predictions Y<sup>ˆ</sup> , prediction scores S<sup>ˆ</sup>   
1: Compute depth $\begin{array} { r } { \ell _ { v } \gets \sum _ { u } \dot { D } _ { u , v } - 1 } \end{array}$ and normalised depth $d _ { v } \gets \ell _ { v } /$ max ℓ for each node $v \in \mathcal V$   
2: Define FN rate per node: $r _ { v } ^ { \mathrm { F N } } = 0 . 5 \eta d _ { v } ^ { 1 . 5 }$ ▷ Increases with depth   
3: Define FP rate per node: $r _ { v } ^ { \mathrm { F P } } = 1 0 ^ { - 4 } \eta \left( 1 + 0 . 5 \left( 1 - d _ { v } \right) \right)$ ▷ Higher near root   
4: $\hat { \mathbf Y } \gets \mathbf Y$ ▷ Initialise predictions from ground truth   
5: for each node $v \in \mathcal V$ do   
6: for each sample i where $Y _ { i v } = 1$ do   
7: Flip $\hat { Y } _ { i v } \gets 0$ with probability $r _ { v } ^ { \mathrm { F N } }$ ▷ Inject false negatives   
8: end for   
9: for each sample i where $Y _ { i v } = 0$ do   
10: Flip $\hat { Y } _ { i v } \gets 1$ with probability $r _ { v } ^ { \mathrm { F P } }$ ▷ Inject false positives   
11: end for   
12: end for   
13: for each sample $i = 1 , \ldots , N$ do ▷ Enforce hierarchical consistency   
14: $\hat { \mathbf Y } _ { i } \gets \mathrm { c l i p } ( \mathbf D \cdot \hat { \mathbf Y } _ { i } , 0 , 1 )$   
15: end for   
16: for each sample i and node $v \in \mathcal V$ do ▷ Scores for AP   
17: if $\hat { Y } _ { i v } = 1$ then   
18: $\hat { S } _ { i v } \sim \mathcal { U } ( 0 . 5 5 , 0 . 9 5 )$   
19: else   
20: $\hat { S } _ { i v } \sim \mathcal { U } ( 0 . 0 5 , 0 . 4 5 )$   
21: end if   
22: end for   
23: return Y<sup>ˆ</sup> , S<sup>ˆ</sup>

In this more realistic setting (Figure 7), we observe that AP is both insensitive to disruptive noise and misleading as to model quality, leading to an optimistic overestimation of model performance. This behaviour matches results in existing studies (Xu et al., 2024; Giunchiglia and Lukasiewicz, 2020; Pereira et al., 2021; Xu et al., 2026). AP behaves this way because of the sparsity of nodes in large hierarchies. Low ground-truth cardinality means that the vast majority of nodes are inactive for any given sample. Micro-average AP computes precision and recall across all node-sample pairs simultaneously, so by its inability to distinguish the trivial discrimination of inactive nodes from the genuinely difficult and important task of detecting active ones, the metric becomes inflated. On the other hand, $d _ { \mathrm { i n v } }$ is not susceptible to this inflation because it operates directly on predicted and ground-truth cardinalities. Consequently, $d _ { \mathrm { i n v } }$ degrades more steeply than both $F _ { 1 }$ and $\mathbf { A P }$ as noise intensifies, reflecting the practical reality that the model has lost its ability to resolve the fine-grained distinctions important to HML classification.

![](images/b05205be316279e817520b905bec158871fa2858260763e05839994dc1a6fb2f.jpg)  
(a) F<sub>1</sub>

![](images/fc7814d818f65b9c5c4928909b3d2099ec24c1743837dfe0b46b9a1b0745fb9e.jpg)  
(b) Precision

![](images/0ecea50f7505dcb334bb1fba85f50c1449ee1760e23bbf8dd1b40fae19cd1637.jpg)  
(c) Recall

![](images/d2391d142a17bd004313b1ffb18d6028f84f8ce615fcf078107fa89b5b2c1a66.jpg)  
(d) AP

![](images/18f01d48e07a965d418a7605a442dc5dea5e17d607c72b42221380b2c839a53e.jpg)  
(e) d<sub>inv</sub>  
Figure 7: HML Metric Behaviours. The top row displays the behaviour of node-wise $F _ { 1 }$ , and its components in the simulated noisy model experiment across hierarchy sizes $| \nu |$ and added η. The bottom row compares $\mathbf { A P }$ to $d _ { \mathrm { i n v } }$

![](images/4ae3aec91bff3a4c5aacf93c7990051a712abaaa439289c68a37d2030b74dc38.jpg)  
Figure 8: Sample Hierarchy. A toy hierarchy of eight nodes, intended as the subject for comparisons between $h F _ { 1 }$ and $d _ { \mathrm { i n v } }$

## I.1 Comparisons to Hierarchical $F _ { 1 }$

We compare $d _ { \mathrm { i n v } }$ and $h F _ { 1 }$ (Lin et al., 2025) on two possible prediction strategies that we dub “sibling hedging” and “depth over-prediction”. Throughout this section, we refer to the example hierarchy depicted in Figure 8. Consider a hierarchy with root node r, internal nodes A and B as children of r, and leaf nodes $a _ { 1 } , a _ { 2 } , a _ { 3 }$ as children of A and $b _ { 1 } , b _ { 2 }$ as children of B. The hierarchy therefore contains $| \nu | = 8$ nodes. Suppose the ground truth annotation for a given sample is ${ \mathcal { Y } } = \{ a _ { 1 } \} , \operatorname { s o } n _ { \mathrm { t r u t h } } = { \mathrm { \dot { 1 } } }$ . The maximum tree distance from $a _ { 1 }$ to any other node is $d _ { \operatorname* { m a x } } = 4$ , attainable by $b _ { 1 }$ and $b _ { 2 }$ . For reference, the perfect prediction $\hat { \mathcal { V } } = \{ a _ { 1 } \}$ achieves $h F _ { 1 } = 1 . 0$ and $d _ { \mathrm { i n v } } = 1 . 0$

First, consider sibling hedging, where a risk-averse classifier predicts all siblings of the true label, producing $\hat { \mathcal { V } } =$ $\{ a _ { 1 } , a _ { 2 } , a _ { 3 } \}$ with $n _ { \mathrm { p r e d } } = 3$ . To compute $h F _ { 1 }$ , each label is expanded into the set of itself and its ancestors. The expanded ground truth is ${ \mathcal { V } } ^ { \uparrow } = \{ a _ { 1 } , A \}$ (ignoring the root) and the expanded prediction is $\hat { \mathcal { V } } ^ { \uparrow } = \{ a _ { 1 } , a _ { 2 } , a _ { 3 } , A \}$ Precision is $\vert \mathcal { V } ^ { \uparrow } \cap \hat { \mathcal { V } } ^ { \uparrow } \vert / \vert \hat { \mathcal { V } } ^ { \uparrow } \vert = \vert \{ a _ { 1 } , A \} \vert / \vert \{ a _ { 1 } , a _ { 2 } , a _ { 3 } , A \} \vert = 2 / 4$ , and recall is $| \mathcal { V } ^ { \uparrow } \cap \hat { \mathcal { V } } ^ { \uparrow } | \mathrm { ~ / ~ } | \mathcal { V } ^ { \uparrow } | = 2 / 2 = 1$ . The harmonic mean gives $h F _ { 1 } = 2 \cdot ( 1 / 2 ) \cdot 1 / \left( 1 / 2 + 1 \right) = 1 / \left( 3 / 2 \right) = 2 / 3 = 0 . 6 7$ . The ancestor overlap substantially attenuates the penalty for the two spurious predictions, because siblings share their full ancestral lineage with the true label. This leniency intensifies as hierarchy depth increases relative to the count of sibling nodes, since the shared ancestral path increasingly dominates the expanded sets.

To compute $d _ { \mathrm { i n v } }$ for the same scenario, we first construct the score matrix $\mathbf { D } _ { \mathrm { i n v } } \in \mathbb { R } ^ { n _ { \mathrm { t r u t h } } \times n _ { \mathrm { p r e d } } } = \mathbb { R } ^ { 1 \times 3 }$ , where each entry is the normalised similarity $( d _ { \mathrm { m a x } } - d _ { \mathrm { p r e d } } ) / d _ { \mathrm { m a x } }$ . The tree distances from $a _ { 1 } ~ \mathrm { t o }$ each prediction are $d _ { T } ( a _ { 1 } , a _ { 1 } ) = 0$ $d _ { \mathcal { T } } ( a _ { 1 } , a _ { 2 } ) = 2$ , and $d _ { T } ( a _ { 1 } , a _ { 3 } ) = 2 .$ , giving $\mathbf { D } _ { \mathrm { i n v } } = [ ( 4 { - } 0 ) / 4 \quad ( 4 { - } 2 ) / 4 \quad ( 4 { - } 2 ) / 4 ] = [ 1 . 0 \quad 0 . 5 \quad 0 . 5 ]$ . Solving the rectangular assignment problem yields $f _ { \mathrm { J V } } ( \hat { \mathcal { V } } , \mathcal { V } ; \mathbf { D } _ { \mathrm { i n v } } ) = 1 . 0 ,$ , so $s _ { \mathrm { m a x } } = f _ { \mathrm { J V } } / n _ { \mathrm { t r u t h } } = 1 . 0$ , since the optimal match assigns predicted $a _ { 1 }$ to true $a _ { 1 }$ at a score of 1.0. For the cardinality factor, we have $n _ { \mathrm { E } } = n _ { \mathrm { p r e d } } - n _ { \mathrm { t r u t h } } = 3 - 1 = 2$ excess predictions and $n _ { \mathrm { M E } } = | \mathcal { V } | - n _ { \mathrm { t r u t h } } = 8 - 1 = 7$ . Evaluating $\gamma ( 2 ; k _ { \mathrm { P } } { = } 0 . 5 , n _ { \mathrm { M E } } { = } 7 )$ numerically via Equation 4 gives $\gamma \approx 0 . 2 4 8$ . The final score is therefore $d _ { \mathrm { i n v } } = s _ { \mathrm { m a x } } \cdot \gamma = 1 . 0 \cdot 0 . 2 4 8 \approx 0 . 2 5$ . While $h F _ { 1 }$ awards $0 . 6 7$ to this prediction, $d _ { \mathrm { i n v } }$ assigns only 0.25, reflecting the intuition that predicting all children when only one is correct constitutes a significant cardinality violation.

Now consider depth over-prediction, where the classifier predicts every leaf: $\hat { \mathcal { V } } = \{ a _ { 1 } , a _ { 2 } , a _ { 3 } , b _ { 1 } , b _ { 2 } \}$ with $n _ { \mathrm { p r e d } } = 5$ For $h F _ { 1 }$ , the expanded prediction becomes $\hat { y } ^ { \uparrow } = \{ a _ { 1 } , a _ { 2 } , a _ { 3 } , b _ { 1 } , b _ { 2 } , A , B \}$ , containing seven elements, while the expanded ground truth remains ${ \mathcal { V } } ^ { \uparrow } = \{ a _ { 1 } , A \}$ . Precision is $| \{ a _ { 1 } , A \} | \ / \ | \{ a _ { 1 } , a _ { 2 } , a _ { 3 } , b _ { 1 } , b _ { 2 } , A , B \} | = 2 / 7$ , recall is $2 / 2 = 1$ , and $h F _ { 1 } = 2 \cdot ( 2 / 7 ) \cdot 1 / ( 2 / 7 + 1 ) = ( 4 / 7 ) / ( 9 / 7 ) \stackrel { \cdot } { = } 4 / 9 \stackrel { \cdot } { \approx } 0 . 4 4$ . Despite predicting every leaf, the metric remains well above zero because several nodes share intermediate ancestors.

For $d _ { \mathrm { i n v } } ,$ the score matrix is $\mathbf { D } _ { \mathrm { i n v } } \in \mathbb { R } ^ { 1 \times 5 }$ . The tree distances from $a _ { 1 }$ are $d _ { \mathcal { T } } ( a _ { 1 } , a _ { 1 } ) = 0 , d _ { \mathcal { T } } ( a _ { 1 } , a _ { 2 } ) = 2 ,$ $d _ { \mathcal { T } } ( a _ { 1 } , a _ { 3 } ) = 2 , d _ { \mathcal { T } } ( a _ { 1 } , b _ { 1 } ) = 4 ,$ , and $d _ { T } ( a _ { 1 } , b _ { 2 } ) = 4 _ { \ O }$ , yielding $\mathbf { D } _ { \mathrm { i n v } } = [ 1 . 0 \quad 0 . 5 \quad 0 . 5 \quad 0 . 0 \quad 0 . 0 ]$ . The optimal assignment again matches predicted $a _ { 1 }$ to true $a _ { 1 } ,$ , giving $f _ { \mathrm { J V } } = 1 . 0$ and hence $s _ { \mathrm { m a x } } = f _ { \mathrm { J V } } / n _ { \mathrm { t r u t h } } = 1 . 0$ . Now $n _ { \mathrm { E } } = 5 - 1 = 4$ and $n _ { \mathrm { M E } } = 7 ,$ , so the cardinality factor is $\gamma ( 4 ; 0 . 5 , 7 ) \approx 0 . 0 5 7$ , and the final score is $d _ { \mathrm { i n v } } = 1 . 0 \cdot 0 . 0 5 7 \approx 0 . 0 6$ . Where $h F _ { 1 }$ still awards 0.44 to a degenerate prediction that labels every leaf in the hierarchy, $d _ { \mathrm { i n v } }$ reduces the score to near zero, faithfully capturing that such a prediction carries almost no discriminative information.

The divergence between the two metrics in these scenarios stems from a fundamental design difference. Because $h F _ { 1 }$ operates on ancestor-expanded sets without solving an explicit assignment between predicted and true labels, any overlap in ancestral paths contributes positively to both precision and recall, regardless of whether those shared ancestors reflect genuine predictive accuracy or are simply artefacts of the hierarchical structure. In shallow hierarchies with few shared ancestors this effect can be minor, but in deep or broad hierarchies it can substantially dampen the penalty for spurious predictions. By contrast, $d _ { \mathrm { i n v } }$ isolates the best possible matching through the assignment problem and then applies the cardinality factor $\gamma ,$ which decays monotonically with the number of excess predictions. This ensures that cardinality violations are penalised in proportion to their severity, independently of where in the hierarchy the spurious predictions happen to fall. We view this property as particularly important for applications where downstream decisions depend on the precise set of predicted labels rather than merely on their neighbourhood in the hierarchy.

This leniency has been identified independently. Falis et al. (2021) observe the same structural property in set-based hierarchical evaluation, of which $h F _ { 1 }$ is an instance. Because the expanded sets are unions, several predicted descendants of a common ancestor reduce to a single entry, and the measure does not consider how many predictions reach the ancestor. They address this by propagating descendant counts rather than truth values, and report that the resulting measure returns consistently lower scores than set-based evaluation across three trained models on the MIMIC-III ICD-9 task, attributing the gap to over- and under-prediction within families of codes that set-based evaluation cannot express.

Our own results are in agreement. In Table 13, the degradation observed under text ablation is registered most weakly by $h F _ { 1 }$ among the cardinality-aware scores, and disappears entirely once $\gamma$ is fixed to one, indicating that the change being measured is one of prediction cardinality. That two measures constructed for different domains exhibit the same insensitivity suggests it follows from ancestor expansion itself rather than from any particular hierarchy or dataset.

## I.2 Decomposing the $d _ { \mathbf { i n v } }$ Ablation

Removing the per-sample text signal lowers $d _ { \mathrm { i n v } }$ on BN-Bio by 21.7% (Table 3), but this alone does not show which component of $d _ { \mathrm { i n v } }$ is responsible. $d _ { \mathrm { i n v } }$ is the product of a proximity term $s _ { \mathrm { m a x } } ,$ the assignment-matched hierarchical similarity, and a cardinality factor $\gamma ,$ which penalises excess predictions. We therefore rescore the BN-Bio meanreplacement runs under alternative cardinality factors, holding the matched assignment fixed so that only the cardinality term changes (Table 13).

Table 13: Decomposition of the $d _ { \mathbf { i n v } }$ Change under Mean Replacement. BN-Bio, five seeds (Table 3), rescored with alternative cardinality factors on the same matched predictions; setting $\gamma \equiv 1$ leaves only the matched proximity $s _ { \mathrm { m a x } }$ Per-node: nodes that degrade out of nodes evaluated, with p-value. Without the text signal, mean predicted endpoints rise from 10.16 to 10.29 (ground truth: 2.34).
<table><tr><td>score</td><td>clean</td><td>ablated</td><td> $\Delta$ </td><td>per-node</td></tr><tr><td> $d _ { \mathrm { i n v } } , k _ { P } = 0 . 5$ </td><td>0.0391</td><td>0.0306</td><td>-21.7%</td><td> $2 1 4 / 2 4 8 , 1 . 6 \times 1 0 ^ { - 3 2 }$ </td></tr><tr><td>min/max ratio</td><td>0.2553</td><td>0.2452</td><td>-4.0%</td><td> $1 9 3 \mathrm { / 2 4 8 , 1 . 1 \times 1 0 ^ { - 2 1 } }$ </td></tr><tr><td> $F _ { \mathrm { 1 } } \mathrm { - s t y l e }$  count penalty</td><td>0.3695</td><td>0.3594</td><td>-2.7%</td><td> $1 8 3 \mathrm { ^ { \prime } } 2 4 8 , 2 . 8 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td> $h F _ { 1 }$ </td><td>0.3857</td><td>0.3811</td><td>-1.2%</td><td> $1 5 8 / 2 4 6 , 4 . 7 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>proximity,  $\gamma = 1$ </td><td>0.8371</td><td>0.8408</td><td> $+ 0 . 4 3 \%$ </td><td> $6 0 / 2 4 6 , p = 1 . 0$ </td></tr></table>

The degradation appears under every cardinality-aware score, including three that we did not introduce: a min/max ratio, an $F _ { \mathrm { 1 } } \mathrm { - s t y l e }$ count penalty, and $h F _ { 1 }$ , which registers it most weakly. Setting $\gamma \equiv 1$ , which leaves proximity alone, removes the degradation, while the mean number of predicted endpoints rises from 10.16 to 10.29 against a ground-truth mean of 2.34. The change is therefore attributable to excess prediction cardinality rather than to matched proximity. This is the signature expected of a sub-tree prior. Since $s _ { \mathrm { m a x } }$ scores only the best-matched prediction for each ground-truth endpoint, suppressing predictions in incorrect subtrees can only come through in $\gamma$ . This concerns the fused model with and without per-sample text, and is distinct from the proximity advantage in subsection 4.2, which compares the text branch with the image branch.

The weak response of $h F _ { 1 }$ matches subsection I.1. Because $h F _ { 1 }$ expands each label to itself and its ancestors, spurious predictions that share a lineage with the target are only lightly penalised, so a change expressed through cardinality is precisely the kind to which $h F _ { 1 }$ is least sensitive.

## J Additional Details for Inverted HML Score

This section contains edge case definitions and proofs for the main claims made in the introduction of the inverted HML score $( d _ { \mathrm { i n v } } )$ in section $^ { 3 , }$ These claims and proofs serve to demonstrate the mathematical behaviour and robustness of the cardinality component of $d _ { \mathrm { i n v } }$

## J.1 Edge Cases

Three edge cases for $d _ { \mathrm { i n v } }$ are defined as follows. Where no path exists between a ground-truth endpoint and a predicted one, as is the case between disjoint roots, the pairwise similarity is set to zero rather than left undefined. Where the prediction set is empty, $d _ { \mathrm { i n v } } = 0$ . Where the prediction is deficient rather than excessive, $n _ { \mathrm { E } } \leq 0$ and $\gamma = 1$ , so underprediction is penalised instead in $s _ { \mathrm { m a x } }$ through unmatched ground-truth components, rather than in the cardinality factor.

Theorem 1 (Critical Value for the Trivial Root of $g ) .$ . Given $n _ { M E } > 1$ and $k _ { P } \in ( 0 , 1 )$ , then at $\begin{array} { r } { k _ { P } = k _ { P } ^ { * } : = \frac { n _ { M E } - 1 } { n _ { M E } } } \end{array}$ , the only root of:

$$
g ( b ) = ( k _ { P } - 1 ) e ^ { - b n _ { M E } } + e ^ { - b } - k _ { P } ,\tag{16}
$$

is the trivial solution $b = 0 .$

Proof. We begin by observing that $b = 0$ is always a root of $^ { g , }$ since $g ( 0 ) = ( k _ { \mathrm { P } } - 1 ) + 1 - k _ { \mathrm { P } } = 0$ for every $k _ { \mathrm { P } }$ We then compute the first derivative of $g ( b )$ at $b = 0 \mathrm { : }$

$$
\frac { \mathrm { d } g } { \mathrm { d } b } = - n _ { \mathrm { M E } } ( k _ { \mathrm { P } } - 1 ) e ^ { - b n _ { \mathrm { M E } } } - e ^ { - b }\tag{17}
$$

$$
\left( \frac { \mathrm { d } g } { \mathrm { d } b } \right) _ { b = 0 } = - n _ { \mathrm { M E } } ( k _ { \mathrm { P } } - 1 ) - 1\tag{18}
$$

Setting $\left( { \frac { \mathrm { d } g } { \mathrm { d } b } } \right) _ { b = 0 } = 0 .$ , we have:

$$
k _ { \mathrm { P } } ^ { * } = \frac { n _ { \mathrm { M E } } - 1 } { n _ { \mathrm { M E } } } ,\tag{19}
$$

as the form $k _ { \mathrm { P } } ^ { * }$ must take for $b = 0$ to be a critical point of g. Plugging this back into $\frac { \mathrm { d } g } { \mathrm { d } b }$ , we can observe the behaviour of the function to the left and right of $b = 0$

$$
\frac { \mathrm { d } g } { \mathrm { d } b } = n _ { \mathrm { M E } } \frac { 1 } { n _ { \mathrm { M E } } } e ^ { - b n _ { \mathrm { M E } } } - e ^ { - b }\tag{20}
$$

$$
= e ^ { - b n _ { \mathrm { M E } } } - e ^ { - b } .\tag{21}
$$

Since $n _ { \mathrm { M E } } > 1$ , for $b > 0$ we have $b n _ { \mathrm { M E } } > b$ and hence $e ^ { - b n _ { \mathrm { M E } } } < e ^ { - b }$ , giving $\frac { \mathrm { d } g } { \mathrm { d } b } < 0 .$ . Symmetrically, for $b < 0$ we have $b n _ { \mathrm { M E } } < b$ and hence e<sup>−bnME</sup> $> e ^ { - b }$ , giving $\frac { \mathrm { d } g } { \mathrm { d } b } > 0$ . This means that for $k _ { \mathrm { P } } ^ { * } , b = 0$ is a maximum, and the function monotonically increases on its left to meet this value and decreases monotonically on its right.

For completeness, we can examine the bounds of the function. It is observable from the form presented in Equation 16 that:

$$
\operatorname* { l i m } _ { b \to \infty } g ( b ) = 0 + 0 - k _ { \mathrm { P } } ^ { \ast } = - k _ { \mathrm { P } } ^ { \ast } < 0 .\tag{22}
$$

For the other direction, we factor g as:

$$
g ( b ) = e ^ { - b } \big [ ( k _ { \mathrm { P } } - 1 ) e ^ { - b ( n _ { \mathrm { M E } } - 1 ) } + 1 \big ] - k _ { \mathrm { P } } .\tag{23}
$$

As $b \to - \infty$ , the prefactor $e ^ { - b } \to + \infty$ . Inside the bracket, $e ^ { - b ( n _ { \mathrm { M E } } - 1 ) } \to + \infty$ (since $n _ { \mathrm { M E } } - 1 > 0 )$ , and its coefficient $\left( k _ { \mathrm { P } } - 1 \right) < 0$ drives the bracket $\mathrm { t o } - \infty$ . The product of $e ^ { - b } \to + \infty$ with a bracket tending $\mathbf { { t o - } \infty \ g i v e s } \colon$

$$
\operatorname* { l i m } _ { b  - \infty } g ( b ) = - \infty .\tag{24}
$$

Therefore, for $k _ { \mathrm { P } } ^ { * } , b = 0$ is the global maximum with $g ( 0 ) = 0$ , so $g ( b ) \leq 0$ for all $b \in \mathbb { R }$ with equality only at $b = 0$ Thus, $b = 0$ is the only root of $g ( b )$ □

Theorem 2. $A t k _ { P } = k _ { P } ^ { * } ,$ , the exponential decay function $\gamma ( n _ { E } ; k _ { P } , n _ { M E } ) = a e ^ { - b n _ { E } } +$ c reduces to a linear function of n<sub>E</sub>.

Proof. From Theorem 1, at $k _ { \mathrm { P } } = k _ { \mathrm { P } } ^ { \ast }$ the only solution is $b = 0$ , so it suffices to evaluate lim $_ { b \to 0 } \gamma ( n _ { \mathrm { E } } ; k _ { \mathrm { P } } , n _ { \mathrm { M E } } )$

Returning to the exponential decay function and plugging in the expressions for a and c derived earlier, namely $a = ( 1 - \overset { \cdot } { e } ^ { - b n _ { \mathrm { M E } } } ) ^ { - 1 }$ and $c = 1 - a \colon$

$$
\gamma ( n _ { \mathrm { E } } ; k _ { \mathrm { P } } , n _ { \mathrm { M E } } ) = \frac { e ^ { - b n _ { \mathrm { E } } } } { 1 - e ^ { - b n _ { \mathrm { M E } } } } + 1 - \frac { 1 } { 1 - e ^ { - b n _ { \mathrm { M E } } } }\tag{25}
$$

$$
= 1 - \frac { 1 - e ^ { - b n _ { \mathrm { E } } } } { 1 - e ^ { - b n _ { \mathrm { M E } } } } .\tag{26}
$$

Now, we can perform a Taylor expansion of the exponential terms around $b = 0 \mathrm { : }$

$$
\gamma ( n _ { \mathrm { E } } ; k _ { \mathrm { P } } , n _ { \mathrm { M E } } ) = 1 - \frac { 1 - \left( 1 - b n _ { \mathrm { E } } + \mathcal { O } ( b ^ { 2 } ) \right) } { 1 - \left( 1 - b n _ { \mathrm { M E } } + \mathcal { O } ( b ^ { 2 } ) \right) }\tag{27}
$$

$$
= 1 - \frac { b n _ { \mathrm { E } } + \mathcal { O } ( b ^ { 2 } ) } { b n _ { \mathrm { M E } } + \mathcal { O } ( b ^ { 2 } ) }\tag{28}
$$

$$
= 1 - \frac { n _ { \mathrm { E } } + \mathcal { O } ( b ) } { n _ { \mathrm { M E } } + \mathcal { O } ( b ) } .\tag{29}
$$

Therefore, we have:

$$
\operatorname* { l i m } _ { b  0 } \gamma ( n _ { \mathrm { E } } ; k _ { \mathrm { P } } , n _ { \mathrm { M E } } ) = 1 - \frac { n _ { \mathrm { E } } } { n _ { \mathrm { M E } } } ,\tag{30}
$$

which is linear in $n _ { \mathrm { E } }$ . As a final note, plugging in $n _ { \mathrm { E } } = 1$ into this linear form gives $\frac { n _ { \mathrm { M E } } - 1 } { n _ { \mathrm { M E } } }$ , which is the form for k<sup>∗</sup> derived in Equation 19, providing a consistency check. □

Theorem 3 (Non-Trivial Existence of b). For a given $n _ { M E } >$ 1 and $k _ { P } \in ( 0 , 1 ) \setminus \{ k _ { P } ^ { * } \}$ , there exists for $g ( b )$ a unique non-trivial root. Moreover, this root is positive when $k _ { P } < k _ { P } ^ { * }$ and negative when $k _ { P } > k _ { P } ^ { * }$

Proof. We return to our derivative for $g ( b )$ from Equation 17, which we restate and set equal to zero:

$$
- n _ { \mathrm { M E } } ( k _ { \mathrm { P } } - 1 ) e ^ { - b n _ { \mathrm { M E } } } - e ^ { - b } = 0 .\tag{31}
$$

In this form, we are now able to isolate for b, attaining:

$$
- n _ { \mathrm { M E } } ( k _ { \mathrm { P } } - 1 ) e ^ { - b n _ { \mathrm { M E } } } = e ^ { - b }\tag{32}
$$

$$
\implies n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) = e ^ { b ( n _ { \mathrm { M E } } - 1 ) }\tag{33}
$$

$$
\therefore b _ { \mathrm { c r i t } } = { \frac { \ln \left( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \right) } { n _ { \mathrm { M E } } - 1 } } .\tag{34}
$$

We note that the exponential term in the intermediate Equation 33 is strictly monotonically increasing in the domain $( - \infty , \infty )$ ; therefore, $b _ { \mathrm { c r i t } }$ must be the unique intersection with the constant on the left-hand side, and hence the unique critical point of $g .$ . As a quick sanity check, we can observe that $b _ { \mathrm { c r i t } } \to 0$ as $k _ { \mathrm { P } }  k _ { \mathrm { P } } ^ { * }$

Now, plugging in $b _ { \mathrm { c r i t } }$ in for the second derivative of $g ( b )$ , we get:

$$
\left( { \frac { \mathrm { d } ^ { 2 } g } { \mathrm { d } b ^ { 2 } } } \right) _ { b = b _ { \mathrm { c i t } } } = n _ { \mathrm { M E } } ^ { 2 } ( k _ { \mathrm { P } } - 1 ) e ^ { - b _ { \mathrm { c r i t } } n _ { \mathrm { M E } } } + e ^ { - b _ { \mathrm { c r i t } } }\tag{35}
$$

$$
= n _ { \mathrm { M E } } ^ { 2 } ( k _ { \mathrm { P } } - 1 ) \big ( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \big ) ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } + \big ( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \big ) ^ { - \frac { 1 } { n _ { \mathrm { M E } } - 1 } }\tag{36}
$$

$$
= \left( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \right) ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } \left( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) + n _ { \mathrm { M E } } ^ { 2 } ( k _ { \mathrm { P } } - 1 ) \right)\tag{37}
$$

$$
= \left( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \right) ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } \left( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) - n _ { \mathrm { M E } } ^ { 2 } ( 1 - k _ { \mathrm { P } } ) \right)\tag{38}
$$

$$
= \left( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \right) ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } \cdot ( 1 - k _ { \mathrm { P } } ) n _ { \mathrm { M E } } ( 1 - n _ { \mathrm { M E } } ) .\tag{39}
$$

We see that all the terms above are positive except for $1 - n _ { \mathrm { M E } }$ , which confirms:

$$
\left( \frac { \mathrm { d } ^ { 2 } g } { \mathrm { d } b ^ { 2 } } \right) _ { b = b _ { \mathrm { c r i t } } } < 0 ,\tag{40}
$$

and $b _ { \mathrm { c r i t } }$ as a maximum. Since $b _ { \mathrm { c r i t } }$ is also the only critical point and $g ( b )$ is continuous, it is therefore the global maximum.

We can now plug $b _ { \mathrm { c r i t } }$ back into the original $g ( b )$

$$
g ( b _ { \mathrm { c r i t } } ) = ( k _ { \mathrm { P } } - 1 ) e ^ { - b _ { \mathrm { c r i t } } n _ { \mathrm { M E } } } + e ^ { - b _ { \mathrm { c r i t } } } - k _ { \mathrm { P } }\tag{41}
$$

$$
= ( k _ { \mathrm { P } } - 1 ) \big ( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \big ) ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } + \big ( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \big ) ^ { - \frac { 1 } { n _ { \mathrm { M E } } - 1 } } - k _ { \mathrm { P } }\tag{42}
$$

$$
= \left( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \right) ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } \left( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) - ( 1 - k _ { \mathrm { P } } ) \right) - k _ { \mathrm { P } }\tag{43}
$$

$$
\begin{array} { r } { = \big ( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \big ) ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } ( 1 - k _ { \mathrm { P } } ) ( n _ { \mathrm { M E } } - 1 ) - k _ { \mathrm { P } } . } \end{array}\tag{44}
$$

When we insert $k _ { \mathrm { P } } ^ { \ast }$ for $k _ { \mathrm { P } } .$ , keeping in mind that $\begin{array} { r } { 1 - k _ { \mathrm { P } } ^ { \ast } = \frac { 1 } { n _ { \mathrm { M E } } } } \end{array}$ , it becomes clear that the expression becomes zero.

We therefore proceed to look at the expression and $b _ { \mathrm { c r i t } }$ itself in the cases where $k _ { \mathrm { P } } < k _ { \mathrm { P } } ^ { * }$ and $k _ { \mathrm { P } } > k _ { \mathrm { P } } ^ { * }$ . We rewrite the expression in Equation 44 as:

$$
g ( b _ { \mathrm { c r i t } } ) = n _ { \mathrm { M E } } ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } ( 1 - k _ { \mathrm { P } } ) ^ { 1 - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } ( n _ { \mathrm { M E } } - 1 ) - k _ { \mathrm { P } }\tag{45}
$$

$$
= n _ { \mathrm { M E } } ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } ( 1 - k _ { \mathrm { P } } ) ^ { - \frac { 1 } { n _ { \mathrm { M E } } - 1 } } ( n _ { \mathrm { M E } } - 1 ) - k _ { \mathrm { P } } .\tag{46}
$$

Now, to determine how $g ( b _ { \mathrm { c r i t } } )$ behaves as $k _ { \mathrm { P } }$ changes, we take the derivative of $g ( b _ { \mathrm { c r i t } } )$ with respect to $k _ { \mathrm { P } } ^ { 5 }$

$$
\frac { \mathrm { d } g ( b _ { \mathrm { c r i t } } ) } { \mathrm { d } k _ { \mathrm { P } } } = - \frac { 1 } { n _ { \mathrm { M E } } - 1 } ( n _ { \mathrm { M E } } - 1 ) n _ { \mathrm { M E } } ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } ( 1 - k _ { \mathrm { P } } ) ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } ( - 1 ) - 1\tag{47}
$$

$$
= n _ { \mathrm { M E } } ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } ( 1 - k _ { \mathrm { P } } ) ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } - 1\tag{48}
$$

$$
\begin{array} { r } { = \left( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \right) ^ { - \frac { n _ { \mathrm { M E } } } { n _ { \mathrm { M E } } - 1 } } - 1 . } \end{array}\tag{49}
$$

We can now attain the following statements:

$$
\frac { \mathrm { d } g ( b _ { \mathrm { c r i t } } ) } { \mathrm { d } k _ { \mathrm { P } } } = 0
$$

$$
k _ { \mathrm { P } } = k _ { \mathrm { P } } ^ { \ast }\tag{50}
$$

$$
\frac { \mathrm { d } g ( b _ { \mathrm { c r i t } } ) } { \mathrm { d } k _ { \mathrm { P } } } < 0 \qquad \iff \qquad n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) > 1 \qquad \iff
$$

$$
k _ { \mathrm { P } } < k _ { \mathrm { P } } ^ { \ast }\tag{51}
$$

$$
\frac { \mathrm { d } g ( b _ { \mathrm { c r i t } } ) } { \mathrm { d } k _ { \mathrm { P } } } > 0
$$

$$
k _ { \mathrm { P } } > k _ { \mathrm { P } } ^ { * } .\tag{52}
$$

It follows that $g ( b _ { \mathrm { c r i t } } )$ attains a global minimum at $k _ { \mathrm { P } } ^ { * }$ , where it equals zero: to the left of $k _ { \mathrm { P } } ^ { * }$ , the function is descending towards the minimum, while to the right, it is ascending away from it. Therefore, $g ( b _ { \mathrm { c r i t } } ) > 0$ for all $k _ { \mathrm { P } } \in ( 0 , 1 ) \setminus \{ k _ { \mathrm { P } } ^ { * } \}$ Next, returning to Equation 34, we see that:

$$
k _ { \mathrm { P } } < k _ { \mathrm { P } } ^ { * } \implies 1 - k _ { \mathrm { P } } > 1 / n _ { \mathrm { M E } }\tag{53}
$$

$$
\implies n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) > 1\tag{54}
$$

$$
\implies \ln \left( n _ { \mathrm { M E } } ( 1 - k _ { \mathrm { P } } ) \right) > 0\tag{55}
$$

$$
\therefore b _ { \mathrm { c r i t } } > 0 .\tag{56}
$$

If we apply the same logic to the case where $k _ { \mathrm { P } } > k _ { \mathrm { P } } ^ { * }$ , we get the opposite result, where $b _ { \mathrm { c r i t } } < 0$

To summarise, we now have the following relevant information:

1. $b _ { \mathrm { c r i t } }$ is the unique critical point of $g ( b )$ and a global maximum, since $g ( b )$ is continuous with no other critical points;

2. $b _ { \mathrm { c r i t } }$ is positive for $k _ { \mathrm { P } } < k _ { \mathrm { P } } ^ { * }$ , negative for $k _ { \mathrm { P } } > k _ { \mathrm { P } } ^ { * }$ , and zero for $k _ { \mathrm { P } } = k _ { \mathrm { P } } ^ { * }$

3. $g ( b _ { \mathrm { c r i t } } ) > 0$ for all $k _ { \mathrm { P } } \in ( 0 , 1 ) \setminus \{ k _ { \mathrm { P } } ^ { * } \}$ , and zero at $k _ { \mathrm { P } } = k _ { \mathrm { P } } ^ { \ast }$ ;

4. $g ( 0 ) = 0$ for all k<sub>P</sub>; and

5. lim $\quad 1 _ { b \to - \infty } g ( b ) = - \infty$ and li $\begin{array} { r } { \mathrm { n } _ { b \to \infty } g ( b ) = - k _ { \mathrm { P } } < 0 , } \end{array}$ , by the same argument as in Theorem 1 (which holds for all $k _ { \mathrm { P } } \in ( 0 , 1 ) )$ .

Since $b _ { \mathrm { c r i t } }$ is the only critical point, $g$ is strictly increasing on $( - \infty , b _ { \mathrm { c r i t } } )$ and strictly decreasing on $( b _ { \mathrm { c r i t } } , \infty )$ . In particular, g can cross zero at most once on each of these two intervals.

In the case where $k _ { \mathrm { P } } < k _ { \mathrm { P } } ^ { * }$ , we have $b _ { \mathrm { c r i t } } > 0$ , so the trivial root $b = 0$ lies in the increasing interval. Since g is strictly increasing on $( - \infty , b _ { \mathrm { c r i t } } )$ and $g ( 0 ) = 0$ , it follows that $g ( b ) < 0$ for all $b < 0$ . On the decreasing interval $( b _ { \mathrm { c r i t } } , \infty )$ , g falls from $g ( b _ { \mathrm { c r i t } } ) > 0 \mathrm { t o } - k _ { \mathrm { P } } < 0 $ , so by the intermediate value theorem and strict monotonicity, there is exactly one root, which is positive.

In the case where $k _ { \mathrm { P } } > k _ { \mathrm { P } } ^ { * }$ , we have $b _ { \mathrm { c r i t } } < 0$ , so the trivial root $b = 0$ lies in the decreasing interval. Since g is strictly decreasing on $( b _ { \mathrm { c r i t } } , \infty )$ and $g ( 0 ) = 0$ , it follows that $g ( b ) < 0$ for all $b > 0$ . On the increasing interval $( - \infty , b _ { \mathrm { c r i t } } ) , g$ rises from −∞ to $g ( b _ { \mathrm { c r i t } } ) > 0$ , so by the intermediate value theorem and strict monotonicity, there is exactly one root, which is negative.

Lastly, in the case $k _ { \mathrm { P } } = k _ { \mathrm { P } } ^ { * } , b _ { \mathrm { c r i t } } = 0 \mathrm { a n d } g ( b _ { \mathrm { c r i t } } ) = 0$ , so only the trivial root exists, recovering the result of Theorem 1. □