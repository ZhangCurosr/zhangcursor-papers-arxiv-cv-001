# Cost-efficient Active Learning for Referring Image Segmentation and Grounding

Junbeom Hong\*<sup>,1</sup> Seonghoon Yu\*<sup>,2</sup> Hyung Rok Jung<sup>3</sup> Sundong Kim<sup>3</sup> Jeany Son<sup>4</sup>

<sup>1</sup>Meissa <sup>2</sup>KAIST <sup>3</sup>GIST <sup>4</sup>POSTECH jbhong@meissa.ai seonghoon.yu@kaist.ac.kr jeany@postech.ac.kr

## Abstract

Collecting natural-language referring expressions along with region annotations, such as masks or boxes, is a major bottleneck in visual grounding (VG), as annotators must write descriptions that distinguish target regions from visually similar ones. We tackle this by formulating active learning (AL) for VG under the realistic setting where only raw images are available without accompanying text. Since ground-truth text is unavailable, sample selection must estimate which images contain ambiguous regions that would require discriminative referring expressions. To address this, we generate auxiliary region-text pairs using foundation models, and introduce Referred Region Ambiguity, a new acquisition function that measures whether the model’s confidence collapses onto a single region or disperses across multiple candidates. It allows our method to prioritize images with strong cross-region competition, which are more in formative due to their visual ambiguity. We also design a referring-expression annotation interface that helps annotators quickly focus on writing discriminative language with a few clicks. Experiments on RIS and REC bench marks show that our AL framework consistently outperforms several AL baselines, while a user study shows up to 1.6× faster description labeling of ours. Code is available at https://github.com/junbum766/ALRIS.

## 1 Introduction

Deep learning models often require large amounts of high-quality human annotations, making data collection a major bottleneck. Active learning (AL) addresses this issue by selecting the most informative unlabeled samples for annotation, reducing labeling costs while maintaining performance. It has been widely applied to computer vision tasks such as image classification (Gal et al., 2017; Sener and Savarese, 2018), object detection (Yang et al., 2024; Lyu et al., 2023), and semantic segmentation (Cai et al., 2021; Hwang et al., 2023).

![](images/eea1b9b8ff41440519fe1fe212fc1a26bf167f7b8191ff23af8a1a3657f1422c.jpg)  
Figure 1: Illustration of AL settings in vision-language domains: (a) existing unrealistic scenario (Kim et al., 2021; Lin and Parikh, 2017) in VQA assumes that costly textual questions related to each image are already available in the unlabeled pool, whereas (b) our realistic setting in VG considers a practical scenario where only raw images are provided without textual descriptions.

Despite its success in vision-only tasks, active learning (AL) remains underexplored for textconditioned vision-language (VL) tasks such as visual question answering (VQA) and visual grounding (VG). Existing AL studies mainly focus on VQA and assume that image-text pairs are already available in the unlabeled pool (Karamcheti et al., 2021; Lin and Parikh, 2017; Kim et al., 2021), requiring annotators only to provide answer labels (Fig. 1a). This assumption is often unrealistic: realworld unlabeled data typically consists of raw images, where both linguistic labels, such as questions in VQA or referring expressions in VG, and their corresponding labels (answers or referred regions) must be annotated. Moreover, AL for VG tasks such as referring image segmentation (RIS) and referring expression comprehension (REC) has not been explored despite their high annotation cost.

Another challenge in VG is that informativeness is characterized differently than in standard visiononly tasks. In VG, the model must pinpoint one referred region among many visually or semantically similar candidates. Therefore, when the model assigns high confidence to multiple regions simultaneously, such samples are, in fact, highly informative: they reveal exactly where the model lacks fine-grained discriminative ability (Fig. 2). However, existing acquisition functions designed for segmentation or object detection fail to capture this property, since they primarily select samples with moderate overall confidence (e.g., pixel entropy (Mackowiak et al., 2018), box-level confidence (Choi et al., 2021)). As a result, cases where the model strongly activates multiple regions are mistakenly treated as low-informative, although they correspond to informative VG samples.

![](images/51be1b07bdba4ca1a21322508dde7f05275f9cbcac24a19003017be1d96b441d.jpg)  
(a) Informative example  
(b) Non-informative example  
Figure 2: Illustration of informativeness in VG: (a) the model assigns high confidence to multiple regions, indicating an informative sample, while (b) a referred one is solely activated by the model, showing low informativeness.

In this paper, we introduce a new active learning task for visual grounding that selects and annotates samples directly from raw images, reducing manual effort by labeling only a small number of highly informative samples (Fig. 1b). First, to address the lack of textual descriptions in the unlabeled image pool, we generate auxiliary region-text pairs, providing the required multi-modal input for a VG model and sample selection. Second, to capture VG-specific informativeness, we introduce a novel acquisition function termed Referred Region Ambiguity. This metric measures how the model’s confidence is distributed across multiple candidate regions instead of being concentrated in one referred region. Its high ambiguity indicates that multiple regions are similarly plausible under the same expression, exposing genuine ambiguity that is worth labeling. By prioritizing such ambiguous samples, our framework enables the VG model to rapidly improve its fine-grained discriminative capability with minimal annotation costs.

Furthermore, we propose an efficient referringexpression annotation interface along with mask annotations for VG. Our interface first lets annotators specify the target region, then assists expression writing with region-aware top-k word suggestions. Rather than typing full sentences from scratch, annotators construct distinct referring expressions by sequentially selecting suggested top-k words relevant to the target region with a few clicks, achieving a significant reduction in the labeling cost.

Under the realistic AL scenarios in VG, our active learning framework achieves state-of-the-art performance over several AL baselines on both RIS and REC benchmarks. Notably, the proposed labeling tool achieves up to 1.6× faster labeling speeds for referring expression annotations, compared to the naïve labeling in a user study.

Our contributions can be summarized as follows:

• We introduce a new task setting for active learning in visual grounding, where sample selection must be performed from raw images without preexisting image-text pairs, reflecting a realistic data collection scenario.

• We propose a novel acquisition, Referred Region Ambiguity, built on auxiliary region-text pairs, which selects ambiguous samples where the model assigns high confidence to multiple regions rather than a referred one, enabling faster learning of fine-grained discrimination.

• We present an efficient referring-expression labeling tool with region annotations that reduces annotation cost by reformulating expression construction as a word-by-word click process.

• Our approach surpasses several AL baselines on RIS and REC benchmarks while concurrently delivering superior labeling efficiency in both speed and annotation cost.

## 2 Related Work

Active Learning. Active learning (AL) reduces annotation costs by selecting informative samples from unlabeled data pools. Prior work has explored uncertainty (Wang and Shang, 2014; Gal et al., 2017; Yoo and Kweon, 2019; Gwon et al., 2025), diversity (Tran et al., 2019), and hybrid (Zhdanov, 2019; Shui et al., 2020) criteria in image classification, and then extended to detection (Choi et al., 2021; Lyu et al., 2023), segmentation (Popp et al., 2025; Kim et al., 2023b, 2024a), and VQA (Karamcheti et al., 2021; Lin and Parikh, 2017; Kim et al., 2021). However, VL AL typically assumes pre-existing language inputs. Although ACTRESS (Kang et al., 2024) actively selects pseudo labels for semi-supervised VG from image-text pairs, it does not address human annotation cost incurred in text modality. In contrast, we study realistic active VG data collection from raw images, where both target regions and referring expressions are newly annotated.

![](images/bb88fbd7aed0e21f8d6d8a9272f96dfa760ecd8c68a3f1384e41398828ad87d8.jpg)  
Figure 3: The proposed AL framework for visual grounding. We first generate auxiliary region-text pairs for all raw images in the entire unlabeled pool (Sec. 3.2). Following this, iterative rounds of sample selection and labeling begin. In each round, our referred region ambiguity (Sec. 3.3) scores the informativeness of all raw images by utilizing their auxiliary region-expression pairs. The top-scoring images are then sequentially annotated via our cost-efficient interface (Sec. 4) until the round budget is exhausted.

Cost-efficient Annotation Tool. Reducing persample labeling cost is crucial for AL efficiency. Prior works have reduced visual annotation costs for segmentation using superpixels (Colling et al., 2020; Hwang et al., 2023) or foundation models (Kim et al., 2024a; Popp et al., 2025), but mainly target mask annotation. In contrast, referring expression annotation remains largely unexplored despite a bottleneck in VG. To this end, we introduce a first interface for referring expressions, together with referred region annotations.

Visual Grounding. Visual grounding (VG) localizes regions from language, producing masks for RIS (Kim et al., 2024b; Yu et al., 2024a) or boxes for REC (Deng et al., 2023; Liu et al., 2023). Since VG annotation requires both regions and detailed expressions, prior work has reduced supervision via weakly-(Kim et al., 2023a; Jin et al., 2023), semi-supervised (Sun et al., 2023; Zang et al., 2024), zero-shot (Yu et al., 2023), and pseudolabeling (Jiang et al., 2022; Yu et al., 2024b) approaches. However, their gap to fully supervised methods (Yu et al., 2025; Xiao et al., 2024) highlights the value of human labels. In this work, we explore active learning for VG to minimize annotation cost while preserving human-label quality.

## 3 Active Learning for Visual Grounding

We introduce an active learning framework for visual grounding that reduces manual labeling costs by selecting a small subset of informative raw images. Our goal is to identify challenging samples where the model struggles to ground an expression to the intended region among similar candidates. Such ambiguity reveals the fine-grained discrimination required for VG. We describe the task setup, auxiliary region-text pair construction, and our acquisition function in this section.

## 3.1 Overall Framework

We start by formalizing the active learning task in the context of visual grounding under a realistic setting, where only raw images are available in an unlabeled pool. Formally, let $\mathcal { I } = \{ I _ { 1 } , \ldots , I _ { | \mathcal { Z } | } \}$ denote an unlabeled image set. The goal is to iteratively select a subset of informative images $\{ I _ { n } \} _ { n = 1 } ^ { N _ { r } } \subset \mathcal { T }$ for human labeling at each round r. To enable VG sample selection on raw images, our framework begins with generation of auxiliary region-text pairs (Sec. 3.2), where it produces auxiliary pairs for each text-less image to infer the informativeness.

In every round r, our acquisition function (Sec. 3.3) assesses informativeness scores of every unlabeled image using its corresponding auxiliary samples. Based on these scores, the top informative raw images $\{ I _ { n } \} _ { n = 1 } ^ { N _ { r } } \subset \mathcal { T }$ are sequentially annotated by humans via our cost-efficient interface (Sec. 4) until the round’s budget $B _ { r }$ is exhausted. These newly labeled samples $\mathcal { D } _ { r }$ are added to the manually labeled set $\textstyle \bigcup _ { i = 1 } ^ { r - 1 } { \mathcal { D } } _ { i }$ , and then the VG model $\theta _ { r - 1 }$ is retrained on these updated labeled data $\mathsf { U } _ { i = 1 } ^ { r } \mathscr { D } _ { i }$ to begin the next AL round. The overall procedure is summarized in Algorithm 1 and illustrated in Fig. 3. In the following sections, we further elaborate on each step in more detail.

Algorithm 1 Proposed Active Learning Framework for VG   
Require: Unlabeled image set I, the number of AL rounds R, budget per round $\overline { { B _ { r } } } ,$ , an initial model $\overline { { \theta _ { 0 } } }$   
1: Build a set of auxiliary pairs $\mathcal { D } _ { \mathrm { a u x } }$ on $\mathcal { T }$ \triangleright Sec. 3.2   
2: for $r = 1 , 2 , \ldots , R$ do   
3: Score informativeness of all raw images $\forall I \in \mathcal { L }$ via $\theta _ { r - 1 }$ and $\mathcal { D } _ { \mathrm { a u x } }$ \triangleright Sec. 3.3   
4: Annotate top-scored images with our annotation interface \triangleright Sec. 4   
5: Build a set of labeled images $\mathcal { D } _ { r }$ within budget $B _ { r }$   
6: Get the model $\theta _ { r }$ trained on labeled dataset $\mathsf { U } _ { i = 1 } ^ { r } \mathscr { D } _ { i }$   
7: end for   
8: Return The final model $\theta _ { R }$

## 3.2 Generation of Auxiliary Region-Text Pairs

A core challenge in applying AL to VG lies in estimating raw-image informativeness without textual descriptions. We address this by generating a set of auxiliary region-text pairs for each unlabeled image, offering potential semantics to estimate its informativeness. Specifically, for a given image $I ,$ we extract a set of N potential regions $\{ r _ { i } ^ { \mathrm { a u x } } \} _ { i = 1 } ^ { N }$ using Grounded-SAM (Ren et al., 2024) guided by COCO 80 categories as semantic prompts, where N depends on the number of detected regions.

Next, we generate corresponding auxiliary captions $\{ t _ { i } ^ { \mathrm { a u x } } \} _ { i = 1 } ^ { N }$ using ViP-LLaVA (Cai et al., 2024) for all regions, with each region highlighted by a red box to guide generation. These region-caption pairs form $\mathcal { D } _ { \mathrm { a u x } }$ over the unlabeled pool I, which provide potential signals to score the informativeness of each raw image via the acquisition function in the next subsection.

## 3.3 Acquisition Function

Our acquisition function, termed Referred Region Ambiguity (Fig. 3), is designed to capture the unique source of informativeness in VG: ambiguity among candidate referred regions. While standard acquisition functions measure uncertainty at the pixel or box level (Mackowiak et al., 2018; Choi et al., 2021), they overlook the cross-region competition induced by a referring expression. In contrast, our method treats candidate regions as competing referents and computes a region-level confidence distribution for each auxiliary caption. It assigns high scores when the model distributes confidence across multiple plausible regions, revealing its difficulty in identifying the intended referent, and low scores when confidence is concentrated on a single region. This formulation directly targets the finegrained discrimination required in VG and provides a more faithful informativeness signal than generic vision-only acquisition criteria that do not account for language-conditioned region ambiguity.

Given a raw image I with its corresponding N auxiliary pairs $\{ ( r _ { i } ^ { \mathrm { a u x } } , t _ { i } ^ { \mathrm { a u x } } ) \} _ { i = 1 } ^ { N }$ , where $r _ { i } ^ { \mathrm { a u x } }$ is an auxiliary region (i.e., a mask for RIS or a box for REC) and $t _ { i } ^ { \mathrm { a u x } }$ is its corresponding caption, we quantify how exclusively the model attends to the intended region for each auxiliary caption. For each caption $t _ { i } ^ { \mathrm { a u x } }$ , the model produces a confidence map (i.e., a final logit scoring map before thresholding for RIS or attention map from the final decoder layer for REC), $p ^ { i } = \theta ( I , t _ { i } ^ { \mathrm { a u x } } ) \in \mathbb { R } ^ { h \times w }$ . We then compute region-level confidence scores $s _ { j } ^ { i }$ over N+ 1 regions (N auxiliary regions plus a background region $r _ { 0 } ^ { \mathrm { a u x } } )$ by averaging the confidence values within each region, as follows:

$$
s _ { j } ^ { i } = \frac { 1 } { \vert r _ { j } ^ { \mathrm { a u x } } \vert } \sum _ { ( x , y ) \in r _ { j } ^ { \mathrm { a u x } } } p _ { ( x , y ) } ^ { i } \in \mathbb { R } , \forall j \in \{ 0 , \ldots , N \} ,
$$

where $p _ { ( x , y ) } ^ { i }$ denotes the confidence score at the pixel coordinate $( x , y )$ . These scores are normalized with a softmax over all regions, producing a probability distribution $\begin{array} { r } { \mathbf q ^ { i } = \frac { \exp ( s _ { j } ^ { i } ) } { \sum _ { j = 0 } ^ { N } \exp ( s _ { j } ^ { i } ) } \in \mathbb { R } ^ { N + 1 } } \end{array}$ for the i-th caption. We then compute the normalized entropy $H _ { n } ( \mathbf { q } ^ { i } )$ of $\mathbf { q } ^ { i }$ , as follows:

$$
H _ { n } ( \mathbf { q } ^ { i } ) = - \frac { \sum _ { j = 0 } ^ { N } q _ { j } ^ { i } \log ( q _ { j } ^ { i } ) } { \log ( N + 1 ) } \in \mathbb { R } ,\tag{1}
$$

where $H _ { n } ( \mathbf { q } ^ { i } )$ is computed by taking the entropy of $\mathbf { q } ^ { i }$ and then dividing it by the maximum entropy value over $N + 1$ regions, $\log ( N + 1 )$ . A higher value of $H _ { n } ( \mathbf { q } ^ { i } )$ indicates greater ambiguity, meaning the model assigns similar confidence to multiple candidate regions rather than focusing on a single dominant one.

Finally, the informativeness of an image I is then computed by averaging $H ( \mathbf { q } ^ { i } )$ over all its auxiliary captions, as follows:

$$
E ( I ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } H _ { n } ( \mathbf { q } ^ { i } ) ,\tag{2}
$$

which we define as Referred Region Ambiguity.

![](images/dd6cb7e3be2038ed9a1f603f85386434ad89f97e748e4464de3957b4fb0b51f2.jpg)  
(a) Target-Mask Specification  
(b) Referring-Expression Labeling Interface  
Figure 4: The proposed cost-efficient annotation interface. (a) target-mask specification, where the user refines a presented mask by clicking on a desired region in sequence, and (b) text labeling interface, where the annotator composes a referring expression for the green mask by clicking on suggested word candidates sequentially, with the option to manually type a word if none of the suggested words are suitable. Demo videos are available in our GitHub repository.

Images with the highest $E ( I )$ scores are then annotated with our cost-efficient labeling tool (Sec. 4) until the next AL round budget is exhausted.

## 4 Cost-efficient Annotation Interface

To mitigate annotation overhead in our AL pipeline, we further present a cost-efficient tool for labeling referring expressions. As shown in Fig. 4, it enables click-driven target-region annotation and word-by-word expression construction via simple clicks, avoiding labor-intensive mask drawing and free-form text typing. We first detail our annotation workflow and then present its cost measurement.

Target-Region Specification. We first specify the target region to be described by a referring expression. For RIS, annotators start from an auxiliary mask from Sec. 3.2 and refine it through simple clicks, each selecting and merging a pre-extracted SAM (Kirillov et al., 2023) mask. For REC, we use two-click bounding-box specification.

Referring-Expression Annotation. For each finalized region, annotators construct a referring expression word by word, as shown in Fig. 4b. At each step, the interface presents k suggested words, which annotators can select with a click or replace by choosing the typing option. This process repeats until the expression is complete. The word candidates are generated by a region multimodal LLM (i.e., ViP-LLaVA (Cai et al., 2024)): at each auto-regressive decoding step, the model predicts the next-word distribution conditioned on the image, region, and previously selected words, and the top-k words are shown as clickable candidates.

Cost Measurement. We define each sample’s labeling cost as the sum of its region cost (mask for RIS or box for REC) and text cost, computed as:

1) Target-region costs $( C _ { \mathrm { r e g i o n } } ) { : }$ for RIS, it is defined as the number of clicks required to finalize the mask labeling; for REC, it is fixed to two clicks to specify the diagonal corners of a single box.

2) Referring-expression costs $( C _ { \mathrm { e x p } } ) { \mathrm { : } }$ we define this cost as the sum of each word selection cost $C _ { \mathrm { w } }$ over an expression. For each word selection, if the desired word appears in the top-k suggestions with probability $p _ { k }$ , the user selects it from $k + 1$ options, costing $\log _ { 2 } ( k + 1 )$ bits. Otherwise, the user selects the typing option with the same cost and additionally pays a typing cost $C _ { \mathrm { t y p e } \cdot } C _ { \mathrm { t y p e } }$ is computed as the conditional information content of typed letters, estimated by a frequency-based letter-level 3-gram model. Formally,

$$
C _ { \mathrm { w o r d } } = \underbrace { \mathrm { l o g } _ { 2 } ( k + 1 ) } _ { \mathrm { C l i c k ~ a ~ w o r d ~ o r ~ t y p i n g ~ o p t i o n } } + \underbrace { ( 1 - p _ { k } ) C _ { \mathrm { t y p e } } } _ { \mathrm { T y p e ~ a ~ w o r d ~ m a n u a l l y } } ,
$$

$$
\mathrm { w h e r e ~ } C _ { \mathrm { t y p e } } = - \sum _ { i = 1 } ^ { n } \log _ { 2 } p _ { l } ( l _ { i } | l _ { i - 1 } , l _ { i - 2 } ) ,\tag{3}
$$

and $p _ { l } ( l _ { i } | l _ { i - 1 } , l _ { i - 2 } )$ is the conditional probability of letter l<sub>i</sub> given previous two letters. Statistics for $p _ { k }$ and $p _ { l } ( l _ { i } | l _ { i - 1 } , l _ { i - 2 } )$ are in Appendix B.1.

## 5 Experiments

## 5.1 Implementation Details

For generation of auxiliary region-text pairs (Sec. 3.2), We extract potential regions from Grounded-SAM (Ren et al., 2024) using a confidence threshold of 0.23, and limit the number of extracted ones per image to 7. We then prompt ViP-LLaVA (Cai et al., 2024) with the instruction “describe the object in the red box” to generate the corresponding expression. For acquisition function (Sec. 3.3), the background region is obtained as the remaining area after excluding all extracted instance regions. For cost-efficient annotation interface (Sec. 4), we set the number of suggested word candidates to $k = 5$ . The annotation interface runs on a single RTX 3090 GPU. For the experiments, we train DETRIS-B (Huang et al., 2025) model for RIS and MaPPER-B model (Liu et al., 2024) for REC on two A6000 GPUs. In both models, we follow the training configurations as detailed in their respective models. Following previous AL works (Munjal et al., 2022), we train the initial model on 1% random samples to address the cold-start problem (Mahmood et al., 2021). The GPU costs incurred by the auxiliary generation and the annotation interface are in Appendix A.5.

<table><tr><td></td><td> Efficient labeling (ours) + Referred region ambiguity (ours)</td></tr><tr><td>Naïve labeling + Referred region ambiguity (ours)</td><td>Efficient labeling (ours) + Random acquisition Naïve labeling + Random acquisition</td></tr></table>

![](images/2a5f0115731022cb91c34c7cf0881f763c1017661b9a967e074902ebdbf19382.jpg)  
(a) RefCOCO

![](images/42d2feab538ff91449402279268ca66497d886972a53f8c02fec76deeae6367e.jpg)  
(b) RefCOCO+

![](images/3eda038e3b9ba2cfa16711222bdb50756a230d8c6d2e30db8a806e8d31e7db43.jpg)  
(c) RefCOCOg

Figure 5: Comparison with the naïve labeling and random acquisition on a RIS task.  
![](images/b873358c182c3fba84204b72f2eabb1f22a8b94db85367ef27b1e956eb1a4944.jpg)  
(a) RefCOCO

![](images/c1d4be090de87f94f72e14bb73d00be6c0e27e5a093d8fe6421a18561b4b8fdf.jpg)  
(b) RefCOCO+

![](images/96e74396492bd756a48c6497a3a75d7e186cbd4e117a8a47828da58410f8cb38.jpg)  
(c) RefCOCOg  
Figure 6: Comparison with the naïve labeling and random acquisition on a REC task. The reported results are averaged on the val and test (A,B) splits of each dataset. The annotation cost (x-axis) is calculated as the measured cost of the acquired samples via used labeling tool divided by the total manual annotation cost of the dataset.

## 5.2 Dataset and Metric

For dataset, we evaluate our active learning framework on standard benchmarks for visual grounding tasks (RIS and REC): RefCOCO (Yu et al., 2016), RefCOCO+ (Yu et al., 2016), RefCOCOg (Mao et al., 2016). Details on each dataset are in Appendix. For evaluation metric, we use standard metrics for both RIS and REC. For RIS, we adopt mIoU (mean IoU), which is the average of the persample IoU scores. For REC, we employ Prec@50, which is an accuracy only when IoU is over 0.5.

## 5.3 Main Results

Referring Image Segmentation. In Fig. 5, we compare the performance of our active learning framework with random acquisition and naïve labeling (i.e., polygon drawing for masks and manual typing for expressions) on RIS benchmark under annotation budgets of 2%, 5%, 10%, 20%, and 30%. The x-axis denotes the measured annotation cost of acquired samples divided by the total naïve labeling cost of the dataset. For clarity, the reported mIoU results are averaged across the val and test(A/B) of each RefCOCO(+/g) and three independent runs to ensure statistical significance. The proposed acquisition outperforms random acquisition under both labeling settings (i.e., our efficient tool and naïve tool), while our annotation interface delivers strong performance across all budgets, proving its efficiency.

Referring Expression Comprehension. We also present the performance of our method on REC compared to random acquisition and naïve labeling (i.e., solely manual typing for expressions) under 10%, 20%, and 30% annotation budgets in Fig. 6. Results are averaged over the val and test(A/B) splits of RefCOCO(+/g) and three separate runs. Our acquisition consistently outperforms random selection, while our text annotation tool achieves higher performance than the naïve tool at the same annotation cost, demonstrating its cost efficiency.

Comparisons on Acquisition Function. To isolate the effect of our acquisition, we compare it with other baselines on RIS and REC using the number of selected region-text pairs as the budget, removing the influence of labeling cost efficiency. We consider random selection, pixel-entropy (Mackowiak et al., 2018), and coreset (Sener and Savarese, 2018), detailed in Appendix B.2. As shown in Figs. 7 and 8, our referred region ambiguity consistently outperforms all baselines, demonstrating its effectiveness in selecting informative VG samples.

Verification of Annotation Tool with User Study. In Tab. 2, we empirically validate the efficiency of our proposed labeling tool via a user study with 30 annotators on Amazon Mechanical Turk. We

Referred region ambiguity (ours)Pixel-entropyCoreset▲Random

![](images/61b33aa39fdc238ad27fa354e2a06c1990bcab8ee103cba11ff4ec628702f0fd.jpg)  
(a) RefCOCO

![](images/7430fdbd52afefd9e1931108fcddcad3ce5a7544591247a73d36846274bca644.jpg)  
(b) RefCOCO+

![](images/8e770de3cef1d569d8a191a49987c8994adba63caf094ebcbfe338c63f6401a3.jpg)  
(c) RefCOCOg

Figure 7: Comparisons with several acquisition baselines on a RIS task.  
![](images/8080dcd8afc4a5967a688fc73fac01b912013a323454414f00e663cac8061b28.jpg)  
(a) RefCOCO

![](images/c0f81be272011663f93a69287503011499d64937f8ee60b12f56f064ed12c053.jpg)  
(b) RefCOCO+

![](images/af408a2e5348612e5800aa1a062c98ceef3335e0017a1ce66cb997b2b69a8e06.jpg)  
(c) RefCOCOg

Figure 8: Comparisons with several acquisition baselines on a REC task. By comparing performance based on the number of acquired instance-text pairs, we isolate the effectiveness of each acquisition from the annotation tool.
<table><tr><td rowspan="2">Acquisition Function</td><td colspan="2">Labeling Interface</td><td colspan="5">Annotation Cost</td><td rowspan="2">Average</td></tr><tr><td>Mask</td><td>Text</td><td>2%</td><td>5%</td><td>10%</td><td>20%</td><td>30%</td></tr><tr><td></td><td></td><td></td><td> $3 1 . 2 0 { \scriptstyle \pm 1 . 9 8 }$ </td><td> $4 5 . 0 4 _ { \pm 0 . 9 7 }$ </td><td> $5 6 . 1 0 { \scriptstyle \pm 1 . 4 4 }$ </td><td> $6 6 . 3 7 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $6 9 . 5 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td>53.65</td></tr><tr><td>√</td><td></td><td></td><td> $3 6 . 5 3 _ { \pm 0 . 9 0 }$ </td><td> $5 2 . 0 5 _ { \pm 1 . 2 8 }$ </td><td> $6 0 . 8 4 _ { \pm 0 . 5 6 }$ </td><td> $6 8 . 2 0 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $7 0 . 6 5 _ { \pm 0 . 2 2 }$ </td><td>57.66</td></tr><tr><td>√</td><td>√</td><td></td><td> $3 8 . 0 8 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $5 4 . 0 9 _ { \pm 1 . 3 0 }$ </td><td> $6 2 . 7 1 _ { \pm 0 . 4 9 }$ </td><td> $6 9 . 8 3 _ { \pm 0 . 1 7 }$ </td><td> $7 1 . 9 2 _ { \pm 0 . 4 0 }$ </td><td>59.33</td></tr><tr><td>√</td><td></td><td>√</td><td> $3 8 . 6 6 _ { \pm 1 . 1 8 }$ </td><td> $5 6 . 4 0 { \scriptstyle \pm 1 . 0 5 }$ </td><td> $6 3 . 4 2 _ { \pm 0 . 4 4 }$ </td><td> $7 0 . 0 1 _ { \pm 0 . 2 4 }$ </td><td> $7 2 . 3 7 { \scriptstyle \pm 0 . 3 1 }$ </td><td>60.17</td></tr><tr><td>√</td><td>√</td><td>√</td><td> ${ \bf 4 5 . 4 1 } _ { \pm 1 . 0 7 }$ </td><td> ${ \bf 5 8 . 9 4 _ { \pm 0 . 8 6 } }$ </td><td> ${ \bf 6 6 . 2 2 _ { \pm 0 . 5 1 } }$ </td><td> ${ \bf 7 1 . 7 4 _ { \pm 0 . 1 9 } }$ </td><td> $7 3 . 5 0 _ { \pm 0 . 2 9 }$ </td><td>63.16</td></tr></table>

Table 1: Ablation study within proposed components. Additional analyses are provided in Appendix A.

<table><tr><td rowspan="2">Tools</td><td colspan="2">Mask Labeling</td><td colspan="3">Referring-Expression Labeling</td></tr><tr><td>Time (s)</td><td>mIoU</td><td>Time (s)</td><td>Sen.-BERT Sim.</td><td>CLIP Score</td></tr><tr><td>Naïve</td><td>60.6</td><td>73.0</td><td>45.8</td><td>39.9</td><td>19.2</td></tr><tr><td>Ours</td><td>9.9</td><td>80.4</td><td>28.6</td><td>44.4</td><td>22.2</td></tr></table>

Table 2: User study comparisons with the naïve labeling. We involve 30 annotators from Amazon Mechanical Turk. Time (s) means the average time per sample.

compare it with the naïve labeling approach (i.e., polygon clicks for masks and manual typing for text), using per-sample time and quality metrics, including mIoU for masks, Sentence-BERT (Reimers and Gurevych, 2019) similarity between expressions and ground-truth one, and CLIP score (Hessel et al., 2021) between cropped images around masks and expressions. Our interface achieves faster annotation while maintaining high-quality mask specification and referring-expression labeling. Examples of user-annotated samples for each tool are visualized in Fig. 16 of Appendix C.1.

<table><tr><td rowspan="2">Ablations</td><td colspan="3">Annotation Budget</td><td rowspan="2">Measured Cost over Dataset</td></tr><tr><td>2%</td><td>10%</td><td>20%</td></tr><tr><td>(a) Mask Annotations</td><td></td><td></td><td></td><td></td></tr><tr><td>Naïve polygon</td><td>38.66</td><td>63.42</td><td>70.01</td><td>1,616,155</td></tr><tr><td>Superpixel</td><td>43.94</td><td>64.80</td><td>71.08</td><td>527,770</td></tr><tr><td>Pre-extracted SAM (ours)</td><td>45.41</td><td>66.22</td><td>71.74</td><td>53,970</td></tr><tr><td>(b) Text Annotations</td><td></td><td></td><td></td><td></td></tr><tr><td>Naïve typing</td><td>38.08</td><td>62.71</td><td>69.83</td><td>6,493,300</td></tr><tr><td>Sentence selection</td><td>40.18</td><td>64.34</td><td>70.16</td><td>2,814,811</td></tr><tr><td>Word selection (ours)</td><td>45.41</td><td>66.22</td><td>71.74</td><td>4,238,677</td></tr></table>

Table 3: Ablation on the annotation interface. Measured cost denotes the total labeling cost required to label all ground-truth data in RefCOCO with each tool.

## 5.4 Ablation Study

We conduct ablation studies on RefCOCO for RIS, reporting average mIoU over the val, test(A/B) splits to validate the effectiveness of our approach.

Effect of Each Proposed Component. Tab. 1 presents the contribution of each component in our AL framework. Starting from random acquisition with naïve annotation, replacing random selection with our referred region ambiguity notably improves performance, especially under low budgets. Adding efficient mask and text-labeling tools further reduces annotation costs and improves performance, with their combination yielding the best results. Additional analyses (e.g., auxiliary regions with other class vocabularies and GPU cost) are in Appendix A.

Ablation in Annotation Interface. To confirm the effectiveness of the proposed labeling interface, Tab. 3 compares our tool with several baseline labeling strategies. For mask labeling, we compare 1) naïve polygon, where the user clicks a series of vertices around region boundaries, and 2) superpixel (Van den Bergh et al., 2012), where the user selects over-segmented superpixels to produce a mask. Compared to them, our mask tool, based on pre-extracted SAM masks, achieves the best results across all budgets and the lowest mask click cost, demonstrating its superiority. For text labeling, we compare 1) manual typing, where annotators manually write full sentences from scratch, and 2) sentence selections, where annotators select the most suitable expression among several MLLMgenerated candidates. Our word-level selection approach with MLLM-guided decoding achieves the highest performance with the reduced manual effort over all compared methods, enabling more high-quality and faster text labeling, as validated in the user study (Tab. 2).

![](images/6de73fe1450393279b012c56e56fb78ff3f63b66a8c18f5128ae93f457d0b286.jpg)  
(a) Ablation with GT annotations

![](images/78f7811c5ef9d9827e0c497e2e29dd3ab958385e7de97cd903982995b13aed86.jpg)  
(b) Ablation with other generators

Figure 9: Ablation on the auxiliary generator. (a) we use GT labels as auxiliary pairs and (b) we use UnSAM (Wang et al., 2024) (unsupervised model) for region generator and CoCa (Yu et al., 2022) (trained solely on web-crawled data without human annotations) for text generator.
<table><tr><td rowspan="2">Quality</td><td colspan="3">Auxiliary Region Generator</td><td colspan="3">Auxiliary Text Generator</td></tr><tr><td>GT mask</td><td>Grounded-SAM</td><td>UnSAM</td><td>GT text</td><td>ViP-LLaVA</td><td>CoCa</td></tr><tr><td>mIoU or Sim.</td><td>100</td><td>83.56</td><td>56.85</td><td>100</td><td>54.72</td><td>32.97</td></tr></table>

Table 4: Quality of auxiliary pairs generated from the auxiliary generators in the ablation (Fig. 9b). For the region quality, we use mIoU with the ground-truth masks. For the text quality, we use Sentence-BERT (Reimers and Gurevych, 2019) similarity with the ground-truth text.

Ablation on Auxiliary Generator. To examine the sensitivity of our active learning framework to auxiliary generators, we conduct ablations using diverse auxiliary sources in Fig. 9. When using the ground-truth annotations as auxiliary region-text pairs (Fig. 9a), our AL method consistently remains robust. For the other generators (Fig. 9b), we use UnSAM (Wang et al., 2024) (unsupervised model) for region generator and CoCa (Yu et al., 2022) (with the weights trained solely on web-crawled data without human annotations) for text generation. Despite quality differences among auxiliary sources (Tab. 4), our active learning framework consistently maintains robust performance. These results indicate that our approach is not sensitive to the auxiliary generators. We attribute this robustness to the role of auxiliary pairs as candidate proposals for acquisition rather than final supervision; active learning mainly depends on the relative ranking of informative candidates, so moderate noise in masks or text often does not substantially change the priority of selection.

## 5.5 Qualitative Analysis

Analysis on Selected Samples via Acquisition. To highlight that our acquisition selects more complex VG images, we present qualitative examples of selected images in Fig. 10 and a statistical analysis in Fig. 11 compared with pixel-entropy (Mackowiak et al., 2018) and coreset (Sener and Savarese, 2018). Our method often selects images with visual similar regions, which is a hard scenario that can confuse the model, and thus promotes its discrimination capability. For a statistical comparison, we measure the feature similarity between the referred region and all other regions within an image, and then average this over all selected samples. The high similarity achieved by our acquisition confirms the presence of many visually similar distractors in the selected samples, indicating that ours effectively prioritizes challenging VG samples.

Analysis of Referred Region Ambiguity. In Fig. 12, we visualize the process of computing our informativeness score and compare it with the pixelentropy (Mackowiak et al., 2018). Pixel-entropy estimates the informativeness by averaging the entropy values (visualized in the figure) across all pix-

![](images/6abb61ffbb31d7f6b189f829fb56a065e0e8a26c83c56f2d1250151a2796495e.jpg)  
(a) Ours

![](images/8c722811f9ee731970afcbafe516874be0065d729712c08f5e5a07ed64edd331.jpg)  
(b) Pixel-entropy

![](images/781516d0495e1e0ab3961fc38b13dc546d83aa64cf7ec37aea270b88517920bc.jpg)  
(c) Coreset  
GT region  
Ours  
Pixel-entropy

Figure 10: Visualization of selected images compared with other acquisition functions. Paired ground-truth regiontext annotations are denoted by matching colors. More visualization are present in Fig. 20 of Appendix C.4.  
![](images/43293ee03134e04d9a4839ba965b8083185fc706cc0ae4fe4101ed18cc032cbe.jpg)  
Figure 11: Statistical comparison of selected pool with other methods. The measured similarity is the average feature similarity between the referred region and all other regions within the image, calculated across all samples in the selected pool.  
Image

## 7 Limitations

While our annotation tool reduces human labor costs, it introduces additional GPU overhead. In particular, generating word-level suggestions with multimodal LLMs during annotation requires extra GPU memory and computation, as analyzed in Tab. 9 of Appendix A.5. Since our annotation tool is designed to reduce human labeling costs,

In this paper, we present a novel active learning framework for visual grounding that operates under the realistic setting of a purely raw-image pool by utilizing auxiliary region-text samples as potential signals. By estimating VG-specific informativeness via our tailored acquisition and simplifying the labor-intensive labeling process via the proposed interface, our approach significantly reduces overall manual effort while consistently outperforming all AL baselines on RIS and REC benchmarks.

## 6 Conclusion

els in the confidence map. In contrast, our referred region ambiguity quantifies the region-level ambiguity (also visualized in the figure) from regionwise confidence scores, and then averages these region-wise entropy values. By focusing on regionlevel scores, our approach effectively selects complex yet informative VG examples. More visualization are provided in Fig. 18 of Appendix C.5.

![](images/4a2a065f9677dc996b2683afad94e7fc98cc4ada4ce10b080b21bc7aae92175b.jpg)  
“a car with only its rear end visible" Avg. score : 0.03Avg. score : 0.7  
Figure 12: Qualitative analysis on the process of computing the informativeness score compared to pixel-entropy. The pixel-entropy displays a raw entropy map derived from the confidence map, while ours shows the measured region-wise entropy computed from region-wise confidence distribution.

we do not explicitly consider the GPU costs incurred by the tool in its design. However, these costs are marginal, as the interface runs on a single RTX 3090 and requires only modest computation compared to the saved human labor. Overall, our tool reduces the annotation cost from \$18,296 with naïve labeling to \$6,752, achieving a 2.7× cost reduction. We leave GPU-aware annotation-tool design for future work.

## 8 Ethical Consideration

Our work aims to reduce human annotation burden in visual grounding. Since our annotation tool involves human annotators, they should be clearly informed of the task and fairly compensated. Although our interface reduces repetitive manual effort, human verification remains important to ensure accurate and unbiased referring expressions.

## Acknowledgments

This work was supported by the IITP grants (RS-2019-II191906 (5%), 2019-0-01842 (5%), RS-2022-II220926 (35%), RS-2026-25518317 (30%)), the NRF grant (RS-2025-24535146 (10%)), the InnoCORE program (N10250156 (10%)), GIST (Future-leading Specialized Research Project (5%)), and the AI Computing Support Project for R&D, funded by MSIT, Korea.

## References

Lile Cai, Xun Xu, Jun Hao Liew, and Chuan Sheng Foo. 2021. Revisiting superpixels for active learning in semantic segmentation with realistic annotation costs. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. 2024. Vip-llava: Making large multimodal models understand arbitrary visual prompts. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Jiwoong Choi, Ismail Elezi, Hyuk-Jae Lee, Clement Farabet, and Jose M Alvarez. 2021. Active learning for deep object detection via probabilistic modeling. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Pascal Colling, Lutz Roese-Koerner, Hanno Gottschalk, and Matthias Rottmann. 2020. Metabox+: A new region based active learning method for semantic segmentation using priority maps. arXiv preprint arXiv:2010.01884.

Aysen Degerli, Serkan Kiranyaz, Muhammad EH Chowdhury, and Moncef Gabbouj. 2022. Osegnet: Operational segmentation network for covid-19 detection using chest x-ray images. In IEEE International Conference on Image Processing (ICIP).

Jiajun Deng, Zhengyuan Yang, Daqing Liu, Tianlang Chen, Wengang Zhou, Yanyong Zhang, Houqiang Li, and Wanli Ouyang. 2023. Transvg++: End-to-end visual grounding with language conditioned vision transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 45(11):13636– 13652.

Dan Friedman and Adji Bousso Dieng. 2023. The vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research (TMLR).

Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017. Deep bayesian active learning with image data. In International Conference on Machine Learning (ICML).

Yeho Gwon, Sehyun Hwang, Hoyoung Kim, Jungseul Ok, and Suha Kwak. 2025. Enhancing cost efficiency in active learning with candidate set query. Transactions on Machine Learning Research (TMLR).

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A reference-free evaluation metric for image captioning. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Jiaqi Huang, Zunnan Xu, Ting Liu, Yong Liu, Haonan Han, Kehong Yuan, and Xiu Li. 2025. Densely connected parameter-efficient tuning for referring image segmentation. In AAAI Conference on Artificial Intelligence (AAAI).

Sehyun Hwang, Sohyun Lee, Hoyoung Kim, Minhyeon Oh, Jungseul Ok, and Suha Kwak. 2023. Active learning for semantic segmentation with multi-class label query. In Conference on Neural Information Processing Systems (NeurIPS).

Haojun Jiang, Yuanze Lin, Dongchen Han, Shiji Song, and Gao Huang. 2022. Pseudo-q: Generating pseudo language queries for visual grounding. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Lei Jin, Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Guannan Jiang, Annan Shu, and Rongrong Ji. 2023. Refclip: A universal teacher for weakly supervised referring expression comprehension. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Weitai Kang, Mengxue Qu, Yunchao Wei, and Yan Yan. 2024. Actress: Active retraining for semi-supervised visual grounding. arXiv preprint arXiv:2407.03251.

Siddharth Karamcheti, Ranjay Krishna, Li Fei-Fei, and Christopher D Manning. 2021. Mind your outliers! investigating the negative impact of outliers on active learning for visual question answering. In the Associationfor Computational Linguistics (ACL).

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. 2014. Referitgame: Referring to objects in photographs of natural scenes. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Dong-Jin Kim, Jae Won Cho, Jinsoo Choi, Yunjae Jung, and In So Kweon. 2021. Single-modal entropy based active learning for visual question answering. In British Machine Vision Conference (BMVC).

Dongwon Kim, Namyup Kim, Cuiling Lan, and Suha Kwak. 2023a. Shatter and gather: Learning referring image segmentation with text supervision. In IEEE International Conference on Computer Vision (ICCV).

Hoyoung Kim, Sehyun Hwang, Suha Kwak, and Jungseul Ok. 2024a. Active label correction for semantic segmentation with foundation models. In International Conference on Machine Learning (ICML).

Hoyoung Kim, Minhyeon Oh, Sehyun Hwang, Suha Kwak, and Jungseul Ok. 2023b. Adaptive superpixel for active learning in semantic segmentation. In IEEE International Conference on Computer Vision (ICCV).

Seoyeon Kim, Minguk Kang, Dongwon Kim, Jaesik Park, and Suha Kwak. 2024b. Extending clip’s image-text alignment to referring image segmentation. In Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics (NAACL).

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo,

and 1 others. 2023. Segment anything. In IEEE International Conference on Computer Vision (ICCV).

Bo Li and Tommy Sonne Alstrøm. 2020. On uncertainty estimation in active learning for image segmentation. arXiv preprint arXiv:2007.06364.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In European Conference on Computer Vision (ECCV).

Xiao Lin and Devi Parikh. 2017. Active learning for visual question answering: An empirical study. arXiv preprint arXiv:1711.01732.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and 1 others. 2023. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499.

Ting Liu, Zunnan Xu, Yue Hu, Liangtao Shi, Zhiqiang Wang, and Quanjun Yin. 2024. Mapper: Multimodal prior-guided parameter efficient tuning for referring expression comprehension. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Mengyao Lyu, Jundong Zhou, Hui Chen, Yijie Huang, Dongdong Yu, Yaqian Li, Yandong Guo, Yuchen Guo, Liuyu Xiang, and Guiguang Ding. 2023. Boxlevel active detection. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Radek Mackowiak, Philip Lenz, Omair Ghori, Ferran Diego, Oliver Lange, and Carsten Rother. 2018. Cereals-cost-effective region-based active learning for semantic segmentation. In British Machine Vision Conference (BMVC).

Rafid Mahmood, Sanja Fidler, and Marc T Law. 2021. Low-budget active learning via wasserstein distance: An integer programming approach. In International Conference on Learning Representations (ICLR).

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. 2016. Generation and comprehension of unambiguous object descriptions. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Prateek Munjal, Nasir Hayat, Munawar Hayat, Jamshid Sourati, and Shadab Khan. 2022. Towards robust and reproducible active learning using neural networks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Varun K Nagaraja, Vlad I Morariu, and Larry S Davis. 2016. Modeling context between objects for referring expression understanding. In European Conference on Computer Vision (ECCV).

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, and 1 others. 2024. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR).

Niclas Popp, Dan Zhang, Jan Hendrik Metzen, Matthias Hein, and Lukas Schott. 2025. Foundation modelbased data selection for dense prediction tasks. In International Conference on Learning Representations (ICLR) Workshop on Foundation Models in the Wild.

Practical Cryptography. English letter frequencies. http://www.practicalcryptography.com.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, and 1 others. 2024. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159.

Ozan Sener and Silvio Savarese. 2018. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations (ICLR).

Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. 2019. Objects365: A large-scale, high-quality dataset for object detection. In IEEE International Conference on Computer Vision (ICCV).

Changjian Shui, Fan Zhou, Christian Gagné, and Boyu Wang. 2020. Deep active learning: Unified and principled method for query and training. In International Conference on Artificial Intelligence and Statistics (AISTATS).

Yawar Siddiqui, Julien Valentin, and Matthias Nießner. 2020. Viewal: Active learning with viewpoint entropy for semantic segmentation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Jiamu Sun, Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Guannan Jiang, Zhiyu Wang, and Rongrong Ji. 2023. Refteacher: A strong baseline for semi-supervised referring expression comprehension. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Toan Tran, Thanh-Toan Do, Ian Reid, and Gustavo Carneiro. 2019. Bayesian generative active deep learning. In International Conference on Machine Learning (ICML).

Michael Van den Bergh, Xavier Boix, Gemma Roig, Benjamin de Capitani, and Luc Van Gool. 2012. Seeds: superpixels extracted via energy-driven sampling. In European Conference on Computer Vision (ECCV).

AV Vladzymyrskyy, AV Gonchar, and V Yu Chernina. 2020. Mosmeddata: Chest ct scans with covid-19 related findings dataset. arXiv preprint arXiv:2005.06465.

Dan Wang and Yi Shang. 2014. A new active labeling method for deep learning. In IJCNN.

XuDong Wang, Jingfeng Yang, and Trevor Darrell. 2024. Segment anything without supervision. In Conference on Neural Information Processing Systems (NeurIPS).

Linhui Xiao, Xiaoshan Yang, Fang Peng, Yaowei Wang, and Changsheng Xu. 2024. Oneref: Unified onetower expression grounding and segmentation with mask referring modeling. In Conference on Neural Information Processing Systems (NeurIPS).

Chenhongyi Yang, Lichao Huang, and Elliot J Crowley. 2024. Plug and play active learning for object detection. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Donggeun Yoo and In So Kweon. 2019. Learning loss for active learning. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. 2022. Coca: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research (TMLR).

Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. 2016. Modeling context in referring expressions. In European Conference on Computer Vision (ECCV).

Seonghoon Yu, Joonbeom Hong, Joonseok Lee, and Jeany Son. 2025. Latent expression generation for referring image segmentation and grounding. In IEEE International Conference on Computer Vision (ICCV).

Seonghoon Yu, Ilchae Jung, Byeongju Han, Taeoh Kim, Yunho Kim, Dongyoon Wee, and Jeany Son. 2024a. A simple baseline with single-encoder for referring image segmentation. arXiv preprint arXiv:2408.15521.

Seonghoon Yu, Paul Hongsuck Seo, and Jeany Son. 2023. Zero-shot referring image segmentation with global-local context features. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Seonghoon Yu, Paul Hongsuck Seo, and Jeany Son. 2024b. Pseudo-ris: Distinctive pseudo-supervision generation for referring image segmentation. In European Conference on Computer Vision (ECCV).

Ying Zang, Chenglong Fu, Runlong Cao, Didi Zhu, Min Zhang, Wenjun Hu, Lanyun Zhu, and Tianrun Chen. 2024. Resmatch: Referring expression segmentation in a semi-supervised manner. arXiv preprint arXiv:2402.05589.

Fedor Zhdanov. 2019. Diverse mini-batch active learning. arXiv preprint arXiv:1901.05954.

# Cost-efficient Active Learning for Referring Image Segmentation and Grounding - Appendix -

## Table of Contents

A. Additional Experiments A.1. Comparisons on Ambiguous Subset A.2. Comparisons with Additional Acquisition A.3. Adaptation Results on Medical Domains A.4. Auxiliary Region with Other Class A.5. Analysis on GPU Cost A.6. Comparisons with Visual-Similarity Baselines A.7. Validity of the Bit-Based Cost Model A.8. Analysis on Collected Expressions

B. Additional Details B.1. Theoretical Text Annotation Cost B.2. Acquisition Baselines B.3. User Study B.4. Dataset

C. Additional Qualitative Results

C.1. Examples of User Study Results

C.2. Examples of Auxiliary Region-Text Pairs

C.3. Examples on Ambiguous Subset

C.4. Examples of Selected Samples via Acquisitions

C.5. Examples of Referred Region Ambiguity

## A Additional Experiments

## A.1 Comparisons on Ambiguous Subset

Our acquisition function focuses on ambiguous cases, where several objects similar to the target appear in the same image. To verify that it effectively improves the model’s ability to distinguish the target from similar ones, we evaluate the RIS model trained under various acquisitions on an ambiguous subset in Tab. 5. We construct this subset from a val split of RefCOCO by selecting 100 heavily confused images that contain more than three objects of the same class (provided in the ground-truth labels). Our approach yields the best performance and among the smallest relative performance drops (%) on this subset, highlighting that it indeed enhances the discriminative ability in complex scenes. The qualitative results of predictions for each method on this set are visualized in Fig. 19.

<table><tr><td rowspan="2">Acquisition</td><td colspan="5"># of Instances</td></tr><tr><td colspan="2">2k</td><td colspan="2">5k</td><td colspan="2">10k</td></tr><tr><td></td><td>All</td><td>Subset</td><td>All</td><td>Subset</td><td>All</td><td>Subset</td></tr><tr><td>Random</td><td>38.46</td><td>32.93 (-14.4%)</td><td>55.64</td><td>48.80 (-12.3%)</td><td>65.23</td><td>60.89 (-6.7%)</td></tr><tr><td>Coreset</td><td>33.54</td><td>30.51 (-9.0%)</td><td>53.77</td><td>43.82 (-18.5%)</td><td>64.33</td><td>54.73 (-14.9%)</td></tr><tr><td>Pixel-entropy</td><td>39.25</td><td>35.74 (-8.9%)</td><td>56.62</td><td>48.04 (-15.2%)</td><td>65.48</td><td>60.56(-7.5%)</td></tr><tr><td>Ours</td><td>49.23</td><td>47.03 (-4.5%)</td><td>62.91</td><td>58.49(-7.0%)</td><td>68.96</td><td>64.00 (-7.2%)</td></tr></table>

Table 5: Comparisons with several acquisition baselines on an ambiguous subset of 100 highly complex images, each containing more than three objects of the same class.

<table><tr><td rowspan="2">Acquisition</td><td colspan="3"># of Instances</td></tr><tr><td>2k</td><td>5k</td><td>10k</td></tr><tr><td>PPAL (Yang et al., 2024)</td><td>36.77</td><td>55.59</td><td>65.98</td></tr><tr><td>Learning Loss (Yoo and Kweon, 2019)</td><td>39.79</td><td>54.82</td><td>64.03</td></tr><tr><td>Referred Region Ambiguity (ours)</td><td>49.14</td><td>61.98</td><td>67.94</td></tr></table>

Table 6: Comparisons with additional acquisition baselines for a RIS task on the validation set of RefCOCO.
<table><tr><td rowspan="2">Acquisition</td><td colspan="3">QaTa-COV19</td><td colspan="4">MosMedData+</td></tr><tr><td colspan="3"># of instances</td><td colspan="4"># of instances</td></tr><tr><td></td><td>0k†</td><td>0.1k 0.5k</td><td>1k</td><td>0k†</td><td>0.1k</td><td>0.3k</td><td>0.5k</td></tr><tr><td>Random</td><td></td><td>48.72</td><td>60.47</td><td>64.09</td><td></td><td>39.27</td><td>46.89 49.52</td></tr><tr><td>Coreset</td><td>12.23</td><td>46.96 61.01</td><td>64.28</td><td>4.52</td><td>37.17</td><td>45.60</td><td>48.57</td></tr><tr><td>Pixel-entropy</td><td></td><td>48.95 59.16</td><td>64.22</td><td></td><td>37.10</td><td>46.22</td><td>47.54</td></tr><tr><td>Ours</td><td></td><td>49.79 62.91</td><td>65.40</td><td></td><td>39.68</td><td>47.12</td><td>50.26</td></tr></table>

Table 7: Adaptation results on medical domains compared with several acquisitions. † denotes the zero-shot results of the RIS model fully-trained on RefCOCO.

![](images/1c8fa526e9c7edf4b492ccffbd413535dd7157412a03fada67a00920a4d94998.jpg)  
Figure 13: Comparison with different class sets for the auxiliary region extraction (Sec. 3.2 in the manuscript) on RefCOCO dataset for a RIS task. Each class set is prompted into Grounded-SAM to extract auxiliary regions. The generic 46-class set is shown in Tab. 8.

## A.2 Comparisons with Additional Acquisition

In Tab. 6, we compare our referred region ambiguity with additional acquisition baselines, including PPAL (Yang et al., 2024) and Learning Loss (Yoo and Kweon, 2019), on the RefCOCO validation set for a RIS task. To isolate the effect of acquisition from annotation tools, we evaluate all methods using the number of acquired samples as the budget. Our acquisition achieves the best mIoU, consistent with the results in Fig. 7 of the main manuscript. These results further demonstrate the effectiveness of our acquisition in improving the fine-grained discriminative ability required for VG.

## Generic 46-class set

person, animal, pet, bird, mammal, sea creature, insect, vehicle, car, truck, motorcycle, bicycle, aircraft, watercraft, furniture, appliance, electronics, kitchenware, decoration, clothing, footwear, accessories, bag, jewelry, food, fruit, vegetable, meal, snack, beverage, ingredient, tool, utensil, sports equipment, musical instrument, device, outdoor object, sign, light, plant, building element, object, container, item, product, material

Table 8: List of a generic 46-class set, constructed by abbreviating Object365 categories using ChatGPT.

## A.3 Adaptation Results on Medical Domains

To examine how well our AL framework efficiently supports the domain adaptation, we fine-tune the RIS model (pretrained on RefCOCO) on medical domains with various acquisitions in Tab. 7. We conduct experiments on two medical referring image segmentation datasets: QaTa-COV19 (Degerli et al., 2022) (5,716 train samples) and MosMed-Data+ (Vladzymyrskyy et al., 2020) (2,183 train samples). Due to a large domain gap (leading to poor zero-shot results), other methods may be less reliable, making random selection a competitive baseline. Nevertheless, our acquisition exhibits faster adaptation than all the compared methods, suggesting its effectiveness in the data-sparse domains with few samples.

## A.4 Auxiliary Region with Other Class

In Generation of Auxiliary Region-Text Pairs (Sec. 3.2 of the main manuscript), we prompt COCO (Lin et al., 2014) 80 classes into Grounded-SAM (Ren et al., 2024) to extract auxiliary regions across all unlabeled images. To further validate the generality of our method beyond COCO 80 categories, we conduct auxiliary region extraction with a more generic 46-class set and report the mIoU results under varying numbers of labeled instances on RefCOCO dataset (averaged over val, testA, and testB results for a RIS task) in Fig. 13. To construct the generic 46-class vocabulary, we aggregate the fine-grained Object365 (Shao et al., 2019) categories into 46 broader superclasses using ChatGPT. The performance gap between a COCO 80-class set and a generic 46-class set is minimal, indicating that our method is robust to the choice of object vocabulary used for generating auxiliary regions.

<table><tr><td>Types</td><td>Generation of Auxiliary Pairs</td><td>Annotation</td><td>Total Cost</td><td>GPU Memory</td></tr><tr><td>GPU overhead (w/ ours)</td><td>16 hours</td><td>844 hours</td><td>$120.4</td><td>8.5GB</td></tr><tr><td>Human labor (w/ ours)</td><td></td><td>844 hours</td><td>$6,752</td><td></td></tr><tr><td>Human labor (w/o ours)</td><td></td><td>2,287 hours</td><td>$18,296</td><td></td></tr></table>

Table 9: Comparisons of total annotation cost in U.S. dollars, including GPU overhead and human labor in our approach, compared with the standard baseline with naïve labeling and random selection. GPU cost is estimated using an RTX 3090 at \$0.14/hour on Vast.ai, and human labor cost is calculated assuming an hourly wage of \$8/hour.

![](images/702e75df33130e7ead5a42697f3b22c117fdc0d451ff48e750d4e8b525cb40dc.jpg)  
Figure 14: Comparisons under a fixed U.S. dollar budget that accounts for the GPU overhead in our approach. The cost estimates are summarized in Tab. 9.

## A.5 Analysis on GPU Cost

The proposed AL framework introduces additional GPU overhead beyond the standard model training in two parts: (1) generating auxiliary region-text pairs using Grounded-SAM (Ren et al., 2024) and ViP-LLaVA (Cai et al., 2024), and (2) supporting annotation by running ViP-LLaVA in the text labeling interface for text annotation and pre-extracting SAM (Kirillov et al., 2023) masks offline for the region labeling. In this subsection, we quantify these additional GPU costs in U.S. dollars (Tab. 9), compare them with human labor costs, and report experiments that explicitly account for resulting GPU overhead from both components (Fig. 14).

GPU Overhead in Generation of Auxiliary Pairs. To generate auxiliary pairs, we use a single RTX 3090 GPU rented from Vast.ai at \$0.14/hour. We run Grounded-SAM for 4 hours to produce auxiliary regions and ViP-LLaVA for 12 hours to generate auxiliary expressions, resulting in a total of 16 GPU hours. This corresponds to an overall cost of \$2.24 (16 hours × \$0.14/hour).

GPU Overhead in Labeling Interface. For text annotation, running ViP-LLaVA in the labeling interface requires 840 GPU hours on a single RTX 3090, computed from the per-sample latency in the user study (Tab. 2) of the main manuscript over the entire 42,404 RefCOCO samples. At \$0.14/hour on Vast.ai, this amounts to \$117.6. For region annotation, we pre-extract SAM masks offline, which takes 4 GPU hours on the same GPU and costs \$0.56. In total, the labeling interface requires 844 GPU hours, corresponding to a total GPU cost of \$118.16. In addition, our labeling tool still requires human annotation, assuming an hourly wage of \$8, the 844 hours of human labor correspond to \$6,752, as reported in Tab. 9. We also report the incurred GPU memory (GB) of ViP-LLaVA-7B on an NVIDIA RTX 3090 GPU in Tab. 9.

Compared to the Human Labor Costs. To compare these costs incurred in our approach with human manual labeling costs, we estimate the total annotation time for all 42,404 RefCOCO samples, based on the per-sample latency reported in the user study (Tab. 2) of the main manuscript. The naïve polygon drawing for the region annotation requires 761 hours, while manually typing full sentences for the text annotation demands 1,526 hours. In total, this amounts to 2,287 hours of human labor. Assuming an hourly wage of \$8, the total human labor cost is \$18,296, which is substantially higher than the GPU cost incurred by our labeling interface and auxiliary region-text generation. In Fig. 14, we further report results under a fixed U.S. dollar budget (%) that accounts for GPU costs, comparing naïve labeling with random acquisition against our labeling with our acquisition. Despite the additional GPU overhead in our approach (Tab. 9), the results in Fig. 14 show that our method remains more costeffective under the same budget. This is because the additional GPU cost is minimal compared with the human labor cost required for large-scale manual annotation.

<table><tr><td rowspan="2">Acquisition</td><td colspan="3"># of Instances</td></tr><tr><td>2k</td><td>5k</td><td>10k</td></tr><tr><td>Random</td><td>37.79</td><td>54.36</td><td>63.87</td></tr><tr><td>DINOv2 region similarity</td><td>46.20</td><td>56.00</td><td>65.36</td></tr><tr><td>Max-region selection</td><td>46.32</td><td>59.99</td><td>66.86</td></tr><tr><td>Referred Region Ambiguity (ours)</td><td>49.14</td><td>61.98</td><td>67.94</td></tr></table>

Table 10: Comparisons with acquisition baselines that rely only on visual similarity or object density, for a RIS task on the validation set of RefCOCO.

<table><tr><td rowspan="2">Interface</td><td colspan="2">Modeled Cost</td><td colspan="2">Measured Time</td></tr><tr><td>Bits</td><td>Reduction</td><td>Time (s)</td><td>Reduction</td></tr><tr><td>Naïve typing</td><td>6,493,300</td><td>0.0%</td><td>45.82</td><td>0.0%</td></tr><tr><td>Sentence selection</td><td>2,814,811</td><td>56.7%</td><td>21.10</td><td>53.9%</td></tr><tr><td>Word selection (ours)</td><td>4,238,677</td><td>34.7%</td><td>28.61</td><td>37.6%</td></tr></table>

Table 11: Validation of the bit-based text annotation cost. We compare the reduction in the modeled bit cost with the reduction in the annotation time measured in user studies, both relative to naïve typing.

## A.6 Comparisons with Visual-Similarity Baselines

To examine whether the gains of our acquisition arise primarily from selecting images that contain many visually similar objects, we compare referred region ambiguity with two additional baselines that rely only on visual signals. For DINOv2 region similarity, we extract DINOv2 (Oquab et al., 2024) features for all candidate regions in an image and use their average pairwise cosine similarity as the acquisition score. For max-region selection, we rank images by the number of regions proposed by Grounded-SAM (Ren et al., 2024), prioritizing images with more candidate objects. Tab. 10 reports the results on the RefCOCO validation set for a RIS task, following the protocol of Appendix A.2. Both baselines outperform random selection, suggesting that visual similarity and object density are useful acquisition signals. However, they consistently underperform referred region ambiguity across all annotation budgets. These results indicate that our gains cannot be explained solely by selecting images with many visually similar objects. Rather, our criterion captures languageconditioned, model-specific uncertainty, i.e., cases in which the current grounding model cannot reliably distinguish the referred target from similar candidates given the expression.

## A.7 Validity of the Bit-Based Cost Model

To further examine the validity of the bit-based text annotation cost (Eq. (3) in Sec. 4 of the main manuscript), we conduct an additional user study for the sentence-selection interface in Tab. 3 on Amazon Mechanical Turk. Twenty annotators each label 10 samples, yielding 200 annotations in total. We compare the reduction in the modeled bit cost with the reduction in the measured annotation time, both relative to naïve typing. As shown in Tab. 11, the modeled bit-cost reduction of sentence selection closely aligns with the measured time reduction (56.7% vs. 53.9%), and a similar agreement is observed for our word-selection interface (34.7% vs. 37.6%). Although the model cannot capture every aspect of human effort, these consistent trends across both interfaces provide additional evidence that it is a reasonable proxy for the relative annotation cost.

## A.8 Analysis on Collected Expressions

Beyond the annotation time and similarity metrics in Tab. 2, we further analyze the referring expressions collected with naïve typing and with our interface in the user study. Tab. 12 reports the following statistics. We use ChatGPT to categorize the words in each expression as attributes, spatial relations, or object categories, and report their proportions. We measure diversity using the Vendi Score (Friedman and Dieng, 2023), which quantifies the semantic diversity of the collected expressions based on LLM embeddings. To assess ambiguity, we conduct a paired comparison in which ChatGPT identifies the more ambiguous expression between those produced by naïve typing and our interface.

<table><tr><td rowspan="2">Tool</td><td rowspan="2"> $\operatorname { A v g } .$  Len.</td><td colspan="2">Word Type</td><td rowspan="2">Diversity ↑</td><td rowspan="2">Ambiguity ↓</td><td rowspan="2">Quality ↑</td><td rowspan="2">Refer. Acc. ↑</td></tr><tr><td>Attribute</td><td>Spatial Category</td></tr><tr><td>Naïve typing</td><td>12.0</td><td>23.6%</td><td>15.8%</td><td>9.8% 20.33</td><td>55.5%</td><td>3.88</td><td>34.70%</td></tr><tr><td>Ours</td><td>6.2</td><td>23.2%</td><td>17.4% 17.4%</td><td>17.71</td><td>44.5%</td><td>3.95</td><td>65.30%</td></tr></table>

Table 12: Language-side analysis of the referring expressions collected with naïve typing and with our interface. Avg. Len. is the average number of words. Word Type is the proportion of attribute, spatial-relation, and object-category words. Diversity is the Vendi Score (Friedman and Dieng, 2023). Ambiguity is the ratio of being judged as the more ambiguous expression in a paired comparison. Quality is a human-rated five-point score from 90 annotators. Refer. Acc. is the ratio of being judged as the expression that more accurately refers to the target.

For human-rated quality, we conduct an additional user study with 90 annotators from Amazon Mechanical Turk, where each annotator evaluates 20 expressions and rates their quality on a five-point scale. To assess referring accuracy, we ask Chat-GPT to identify which expression more accurately refers to the target object.

Naïve typing yields a higher diversity score, likely because its expressions are longer and contain more concepts. However, our manual inspection shows that some of these annotations describe the overall image rather than the target region. For example, one annotation states: “Three ladies are seated close together on a sofa; one is using headphones to unwind, another is wearing a robe and face mask and using a laptop to examine herfingernails, and afourthfigure is depicted as a blue, overlaid silhouette sitting on the right.” This expression describes the overall scene and includes inconsistent details rather than clearly identifying the target region. Such errors may increase diversity but also reduce clarity and increase ambiguity. In contrast, our interface produces clearer and more target-focused expressions, as reflected in the lower ambiguity, higher human-rated quality, and substantially higher referring accuracy.

## B Additional Details

## B.1 Theoretical Text Annotation Cost

In our text annotation cost measurement (Eq. (3) in the main manuscript, rewritten below for convenience), we rely on two types of probabilities: (1) the probability $p _ { k }$ that a user-desired word appears among the suggested candidates in the annotation interface $( i . e . ,$ , top-k coverage probability), and (2) the conditional probability of a letter given the previous two letters, $p _ { l } ( l _ { i } | l _ { i - 1 } , l _ { i - 2 } )$ (i.e., trigram probability). In this subsection, we provide detailed statistics and empirical measurements for

these two probabilities.

$$
C _ { \mathrm { w o r d } } = \underbrace { \mathrm { l o g } _ { 2 } ( k + 1 ) } _ { \mathrm { C l i c k ~ a ~ w o r d ~ o r ~ t y p i n g ~ o p t i o n } } + \underbrace { ( 1 - p _ { k } ) C _ { \mathrm { t y p e } } } _ { \mathrm { T y p e ~ a ~ w o r d ~ m a n u a l l y } } ,
$$

$$
\mathrm { w h e r e ~ } C _ { \mathrm { t y p e } } = - \sum _ { i = 1 } ^ { n } \log _ { 2 } p _ { l } ( l _ { i } | l _ { i - 1 } , l _ { i - 2 } ) .
$$

Top-k Suggestions in Text Annotation Tool. To estimate top-k coverage probability $p _ { k }$ , we measure the top-k accuracy of the k-word suggestions produced by our text annotation tool on the RefCOCO dataset. At each step of the annotation process, we mark the prediction as correct if the ground-truth next word appears within the top-k suggested candidates. The ratio of correct predictions across all steps yields the observed top-k accuracy, which we use as $p _ { k }$ . As shown in Tab. 13, $p _ { k }$ naturally increases as k grows. However, the overall measured cost still increases because selecting from a larger candidate set incurs a higher option cost, represented by the $\log _ { 2 } ( k + 1 )$ term. We choose $k = 5$ as it provides a more intuitive word selection while achieving the lowest measured annotation cost.

Conditional Probability of Letters. For the trigram conditional probability of a letter, $p _ { l } ( l _ { i } | l _ { i - 1 } , l _ { i - 2 } )$ in Eq. (3) of text cost measurement, we use letter-level 3-gram frequencies measured from approximately 4.5 billion characters of English text (we use the English letter frequency data presented in (Practical Cryptography)). For the first and second letters of a sequence, we use 1-gram and 2-gram frequencies, respectively. Subsets of these letter frequencies are shown in Tab. 14.

## B.2 Acquisition Baselines

We conduct acquisition comparisons with pixel entropy (Mackowiak et al., 2018) and coreset (Sener and Savarese, 2018) in Fig. 7 (RIS) and Fig. 8 (REC) of the main manuscript. In this subsection, we provide a detailed implementation for each of the compared baselines.

<table><tr><td rowspan="2">Metric</td><td colspan="3">Top-k candidates</td></tr><tr><td>3 5 (ours)</td><td>10</td><td>20</td></tr><tr><td>Accuracy (%)</td><td>56.48</td><td>61.75</td><td>67.10 71.21</td></tr><tr><td>Measured Cost over Dataset</td><td>4,274,759 4,238,677</td><td>4,324,874</td><td>4,524,190</td></tr></table>

Table 13: Text annotation costs and accuracy across varying k suggested word candidates in our text annotation interface for estimating $p _ { k }$ , which represents the top-k accuracy of including the ground-truth next word among the suggested k candidates over the RefCOCO dataset.

<table><tr><td></td><td colspan="2">1-gram</td><td colspan="2">2-gram</td><td colspan="2">3-gram</td></tr><tr><td>Rank</td><td>Letters</td><td>Probability</td><td>Letters</td><td>Probability</td><td>Letters</td><td>Probability</td></tr><tr><td>1</td><td>E</td><td>0.1210</td><td>TH</td><td>0.0272</td><td>THE</td><td>0.0181</td></tr><tr><td>2</td><td>T</td><td>0.0894</td><td>HE</td><td>0.0234</td><td>AND</td><td>0.0073</td></tr><tr><td>3</td><td>A</td><td>0.0855</td><td>IN</td><td>0.0204</td><td>ING</td><td>0.0072</td></tr><tr><td>4</td><td>0</td><td>0.0747</td><td>ER</td><td>0.0179</td><td>ENT</td><td>0.0042</td></tr><tr><td>5</td><td>I</td><td>0.0733</td><td>AN</td><td>0.0162</td><td>ION</td><td>0.0042</td></tr><tr><td>6</td><td>N</td><td>0.0717</td><td>RE</td><td>0.0142</td><td>HER</td><td>0.0036</td></tr><tr><td>7</td><td>S</td><td>0.0673</td><td>ES</td><td>0.0133</td><td>FOR</td><td>0.0034</td></tr><tr><td>8</td><td>R</td><td>0.0633</td><td>ON</td><td>0.0132</td><td>THA</td><td>0.0033</td></tr><tr><td>9</td><td>H</td><td>0.0496</td><td>ST</td><td>0.0126</td><td>NTH</td><td>0.0033</td></tr><tr><td>10</td><td>L</td><td>0.0421</td><td>NT</td><td>0.0118</td><td>INT</td><td>0.0032</td></tr></table>

Table 14: Top-10 most frequent letter n-grams with their probabilities. We use the English letter frequency data presented in (Practical Cryptography).

Pixel Entropy. This method is a conventional acquisition function widely used in active learning for segmentation (Mackowiak et al., 2018; Li and Alstrøm, 2020; Siddiqui et al., 2020). It quantifies the informativeness of an auxiliary region-text pair by measuring how uncertain the model’s prediction is at the pixel level, and it selects region-level samples for human annotations instead of image-level in ours. Pixel entropy outputs high informativeness when the model’s confidence map is uniformly spread across the whole image with moderate values $( e . g .$ , probability near 0.5). Conversely, if the model confidently activates the whole image with the lowest or highest values $( i . e .$ , probability near 0.0 or 1.0), it outputs low informativeness. We adapt this method for both RIS and REC models, as follows.

1) RIS model: Given a specific auxiliary pair $( r ^ { \mathrm { a u x } } , t ^ { \mathrm { a u x } } )$ , we obtain a confidence map (i.e., a final logit scoring map before thresholding), $p ^ { i } =$ $\theta ( I , t ^ { \mathrm { a u x } } ) \in \mathbb { R } ^ { h \times w }$ . We then apply a sigmoid function to convert these logits into a pixel-level probability map, as follows: $\hat { p } = \sigma ( p ) \in [ 0 , 1 ] ^ { h \times w }$ . The binary entropy $\hat { H } _ { ( x , y ) }$ at each pixel $( x , y )$ is defined as:

$$
\begin{array} { r l } & { \hat { H } _ { ( x , y ) } = - \hat { p } _ { ( x , y ) } \log ( \hat { p } _ { ( x , y ) } ) } \\ & { \qquad - \left( 1 - \hat { p } _ { ( x , y ) } \right) \log ( 1 - \hat { p } _ { ( x , y ) } ) , } \end{array}\tag{4}
$$

where $\hat { H } _ { ( x , y ) }$ is the binary entropy at pixel coordinate $( x , y )$ conditioned on a given auxiliary caption $t ^ { \mathrm { a u x } }$ . The final informativeness score $E ( r ^ { \mathrm { a u x } } , t ^ { \mathrm { a u x } } )$ for an auxiliary pair is obtained by averaging this pixel-level entropy $\hat { H } _ { ( x , y ) }$ across all $h \times w$ pixels, as follows:

$$
\boldsymbol { E } ( \boldsymbol { r } ^ { \mathrm { a u x } } , t ^ { \mathrm { a u x } } ) = \frac { 1 } { \boldsymbol { h } \times \boldsymbol { w } } \sum _ { x = 1 } ^ { \boldsymbol { h } } \sum _ { y = 1 } ^ { \boldsymbol { w } } \hat { \boldsymbol { H } } _ { ( x , y ) } ^ { i } .\tag{5}
$$

Based on this $E ( r ^ { \mathrm { a u x } } , t ^ { \mathrm { a u x } } )$ score computed for all auxiliary pairs, we select the top-scoring auxiliary region-text pairs and annotate them at the regionlevel sample.

2) REC model: Unlike RIS models that directly generate segmentation masks from a probability map, REC models predict bounding boxes by passing only the [CLS] token through an MLP layer, making pixel-level entropy inapplicable. Instead, we leverage the attention map $A \in \mathbb { R } ^ { h \times w }$ extracted from the final decoder layer, conditioned on the given auxiliary caption $t ^ { \mathrm { a u x } }$ and image I. We compute the normalized entropy $H _ { n }$ of this attention map and then average across all N auxiliary captions, as follows:

$$
H _ { n } ( A ) = - { \frac { \sum _ { x = 1 } ^ { h } \sum _ { y = 1 } ^ { w } A _ { ( x , y ) } \log ( A _ { ( x , y ) } ) } { \log ( h \times w ) } } ,\tag{6}
$$

where $A _ { ( x , y ) }$ is an attention score at a pixel coordinate $( x , y )$ . We use this $H _ { n } ( A )$ as a pixel-entropy score for each auxiliary pair in a REC model. We then select and annotate the auxiliary pairs with the highest $H _ { n } ( A )$ scores sequentially.

Coreset. This method aims to select diverse samples from an unlabeled pool to maximize coverage of acquired samples. To achieve this, it first extracts feature representations for all auxiliary pairs in the unlabeled auxiliary region-text pair set $\mathcal { D } _ { \mathcal { U } }$ and then iteratively selects the auxiliary pair whose feature is farthest from the features of all currently labeled region-text pair set $\mathcal { D } _ { \mathcal { L } }$

Formally, given a specific auxiliary caption $t ^ { \mathrm { a u x } }$ with its image (for a labeled case, we use a groundtruth expression instead of an auxiliary caption), we extract the feature representation of a given auxiliary caption from the model encoder $\theta _ { \mathrm { e n c } }$ , as follows:

$$
F = \theta _ { \mathrm { e n c } } ( I , t ^ { \mathrm { a u x } } ) \in \mathbb { R } ^ { h \times w \times d } ,\tag{7}
$$

where $h ,$ w are the size of the feature map and d is a feature dimension. The resulting feature map $F$ is then aggregated via spatial pooling and $\ell _ { 2 ^ { - } }$ normalized to obtain the final image feature vector $\hat { Z } \colon$

$$
{ \hat { Z } } = { \frac { Z } { \| Z \| } } , { \mathrm { ~ w h e r e ~ } } Z = { \frac { \sum _ { x = 1 } ^ { h } \sum _ { y = 1 } ^ { w } F _ { ( x , y ) } } { h \times w } } \in \mathbb { R } ^ { d } ,\tag{8}
$$

and $F _ { ( x , y ) }$ denotes the feature at a pixel coordinate $( x , y )$ . This normalized feature vector $\hat { Z }$ represents the auxiliary pair feature and is used for distance calculation. For each unlabeled auxiliary pair, we compute its minimum distance over all labeled region-text pairs in the acquired pair set $\mathcal { D } _ { \mathcal { L } } \colon$

$$
d _ { j } = \operatorname* { m i n } _ { k \in \mathcal { D } _ { \mathcal { L } } } \| \hat { Z } _ { j } - \hat { Z } _ { k } \| _ { 2 } ,\tag{9}
$$

where k denotes the index for the labeled pairs. We select the auxiliary pair with the maximum distance to the current labeled set:

$$
m ^ { * } = \arg \operatorname* { m a x } _ { j \in \mathcal { D } _ { \mathcal { U } } } d _ { j } ,\tag{10}
$$

where $m ^ { * }$ is a selected index over all auxiliary pairs in the unlabeled pairs set $\mathcal { D } _ { \mathcal { U } } . \ \mathrm { W e }$ add it to the labeled set, $\mathcal { D } _ { \mathcal { L } } \gets \mathcal { D } _ { \mathcal { L } } \cup \{ ( r _ { m ^ { * } } ^ { \mathrm { a u x } } , t _ { m ^ { * } } ^ { \mathrm { a u x } } ) \}$ with its annotations and an image. We then update the distances for all remaining auxiliary pairs: $d _ { j } \gets$ min $( d _ { j } , \Vert \hat { Z } _ { j } - \hat { Z } _ { m ^ { * } } \Vert ) , \forall j \in \{ 1 , \dots , | \mathcal { D } _ { \mathcal { U } } | \} \backslash \{ m ^ { * } \}$ This process iteratively continues until the round budget is exhausted.

## B.3 User Study

To verify the efficiency of the proposed annotation interface, we conduct a user study with 30 annotators from Amazon Mechanical Turk in Tab. 2 of the main manuscript. In this subsection, we provide the details of the user study. Each annotator is required to complete annotations for 10 images, following the instructions in Fig. 15; the instructions also include example videos for easy understanding. We record the time required to annotate both the mask and referring expression and save the annotation results (Fig. 16) to compare the quality with the ground truth labels. Besides our proposed interface, we also conduct the same user study for a naïve annotation $( i . e .$ , polygon drawing for masks and manual typing for expressions) to evaluate the benefits of our proposed tool. The example videos using our labeling interface are provided in our GitHub repository.

## B.4 Dataset

RefCOCO. RefCOCO (Yu et al., 2016) provides images paired with masks or boxes and their corresponding referring text. This is created using the ReferItGame (Kazemzadeh et al., 2014), where one player writes a textual description for a specific instance, and a second player locates the correct object based on that description. The dataset contains 142,210 expressions referring to 50,000 instances across 19,994 images. These instances are divided into the following splits: 42,404 for a training set, 3,811 for a validation set, 1,975 for a testA set, and 1,810 for a testB set. The expressions in RefCOCO are typically concise and often include directional instructions, such as $^ { \circ } l e f i ^ { \prime }$ or “right”, to disambiguate similar objects.

RefCOCO+. RefCOCO+ (Yu et al., 2016) is constructed using the same ReferItGame (Kazemzadeh et al., 2014) method as RefCOCO. The difference in textual form is that it explicitly prohibits the use of positional terms. This constraint forces the expressions to be based primarily on object attributes, such as clothing color or object size. This distinction allows researchers to evaluate a model’s performance on attribute comprehension, separate from its understanding of spatial cues. It includes

![](images/7fcce373b08d4e5a88148807af4f5b7b1c6dd4411b1c537c0a9567bc761472a2.jpg)  
Figure 15: Annotation task instructions for the user study on Amazon Mechanical Turk.

141,564 expressions for 49,856 instances across 19,992 images. These instances are divided into 42,278 for a train set, 3,805 for a val set, 1,975 for a testA set, and 1,798 for a testB set, following a similar split structure to RefCOCO.

RefCOCOg. RefCOCOg (Nagaraja et al., 2016) is characterized by much longer expressions, averaging 8.43 words per expression, compared to 3.61 and 3.53 words for RefCOCO and RefCOCO+, respectively. Therefore, it demands a model’s ability for more complex and contextual comprehension and is commonly used to evaluate multi-modal reasoning under longer linguistic context. Compared to RefCOCO and RefCOCO+, which are built on ReferItGame (Kazemzadeh et al., 2014),

RefCOCOg is collected via Amazon Mechanical Turk. This dataset includes 95,010 expressions for 49,822 object instances across 25,799 images. These instances are separated into 42,226 for a train split, 2,573 for a validation split, and 5,023 for a test split.

## C Additional Qualitative Results

## C.1 Examples of User Study Results

We conduct a user study with 30 annotators from Amazon Mechanical Turk in Tab. 2 of the main manuscript. In Fig. 16, we provide qualitative examples of those results compared with the naïve labeling tool (i.e., polygon drawing for masks and manual typing for expressions). The masks annotated via our interface are more accurate than those drawn manually, since polygon drawing is labor-intensive and struggles with complex object boundaries. Our mask tool leverages precise preextracted SAM masks, enabling high-quality annotations with faster times. Similarly, the expressions annotated through our interface accurately describe the referred regions. This is confirmed by the higher Sentence-BERT similarity scores compared to manual typing in Tab. 2 of the main manuscript. This is because, at each step, the decoding process of the multi-modal language model (i.e., ViP-LLaVA (Cai et al., 2024)) suggests contextually appropriate words relevant to the target region, guiding annotators to concise and accurate descriptions.

![](images/992bdf2b0751739c2a997f35538fff28bdc15580900ce55384e9bfb4b4ec3806.jpg)

Figure 16: Examples of user study results with our cost-efficient annotation interface compared to the naïve annotation interface (i.e., polygon drawing for masks and manual typing for expressions).  
![](images/0309fc2197faf96da9ae1026cb4f990db957283a17abc612f2ba4b1889d0e1ca.jpg)  
Figure 17: Examples of the generated auxiliary region-text pairs. The paired region-text are denoted by matching colors.

## C.2 Examples of Auxiliary Region-Text Pairs

We provide qualitative examples of the generated auxiliary region-text pairs (Sec. 3.2 of the main manuscript) in Fig. 17, where each region-text pair is highlighted using the same colors. Although some pairs contain minor noise, the auxiliary regions accurately localize the objects, and the corresponding captions provide clear and detailed referring expressions for each region, as quantified in Tab. 4.

Pixel-entropy  
Ours  
GT region  
Image  
![](images/6c0925a75c153b0e85bd50d60a2a68c49cc9d4ff2738584db1978ff2c0165881.jpg)  
Figure 18: Additional qualitative analysis on the process of computing the informativeness score compared to the pixel-entropy.

## C.3 Examples on Ambiguous Subset

In Tab. 5 of Appendix A.1, we conduct comparisons with several acquisition baselines (i.e., random, coreset, pixel-entropy, and ours) on the curated ambiguous test subset. In this subsection, we present their qualitative results from this ambiguous subset in Fig. 19. Compared with other acquisitions, the RIS model trained under our acquisition performs better at distinguishing the target from visually similar distractors. This result suggests that the proposed referred region ambiguity is effective in improving the model’s discriminative ability in challenging ambiguous cases.

## C.4 Examples of Selected Samples via Acquisitions

In Fig. 20, we present additional examples of selected samples via each acquisition. Compared with other approaches (i.e., pixel-entropy and coreset), our referred region ambiguity tends to prioritize images containing complex scenes with many objects, leading to more informative selections.

## C.5 Examples of Referred Region Ambiguity

We show additional examples of the process for computing the informativeness score in our referred region ambiguity, compared with the pixel-entropy in Fig. 18. Although the images contain visually similar objects, pixel-entropy assigns low informativeness because it averages entropy over all pixels. In contrast, our acquisition function yields high scores by operating at the region level, effectively capturing the complexity of these scenes.

Random

Pixel-entropy

GT region & Text

![](images/f2007377a4e7726b751315dadf2b4aaa97103d38f822e6c8fbed2893d5de4cd4.jpg)

Ours  
![](images/42aedeefed4f21f058024c394092c5085008c8cee45004ec0ff2b8480a778ad4.jpg)  
“man with yellow tie"

![](images/d22b80381949a492348fb78ac97bba73ff4caae9e2f982892b9c28d4267d2ac3.jpg)  
Coreset

![](images/2d5690e88bb40aef40e3aeb979df0c18d992c055d99c3e2600d0e3a6946c5533.jpg)

![](images/b6e2c830f6ddb3ec8bc3f1dd1a33cbaba87d0089a6a66371b4d1d756202e04ff.jpg)

![](images/e3cf57b16e08a0bb3c42ec4031df975ce45651e093f4d4fd969df6d3abc8eb05.jpg)

![](images/343dc8680687077936c60aef47a6eea9ce68bec3d05c3f6d60d32de8047fa971.jpg)

![](images/d11b85f906778d2e90be499181f2b72aac550724ac3e0c8517ef38b644250a86.jpg)  
“older woman center of photo"

![](images/d125832a06782314b5c2a96aff6ddd3c7d2600bfe5b9c77562b55c09b3684e9f.jpg)

![](images/267f1a13d294b67d1005a2ff1b8d2ca9dca3f24ea88431c6d32418d7eeb14730.jpg)

![](images/820b975c4b1a61101c76e957e1e88630e6da426fa966a617509085d7d7e2e6a8.jpg)

![](images/92f4775913345eccc2176988bc79b8b5748356ed67251b1fbcf73829cbffc036.jpg)  
“guy in middle of pic white shirt”

![](images/22f14cd0e8dc41ab58c852e649f48629446d5e8978fa0d40c297073b4026d5f3.jpg)

![](images/734f7db21fa2c805581750ab1abb2265dbe082db9fc0b89cfe0cb2487253a93d.jpg)

![](images/3a616e592f7668eaae56d4667ccd8553bfeeee0ebcbbc7942190229fe3fbddcc.jpg)  
“zebra in front"

![](images/ef527c572041756a14283b2310cd32ebc0a15a8a920a1c772e1dcdc74837eba1.jpg)

![](images/2cb1e566fcf5ef2f7a95bbe50feea08565ad53394d6d7eee13895d2515e22f70.jpg)

![](images/1a85b731b3755cef1e8aac10c4261fecf72343c1638a42c9bacf7f4b1fa6ac8b.jpg)  
“man on right gray suit"

![](images/8512480da7dd83ca4cd728c39adc13c3c4e57148fbb5a9a60c0d8650085c010a.jpg)

![](images/5a59efc49b2a40dfe54fbc49db44447b37337fc66b80e47145067a9651772fba.jpg)

![](images/798ce2cb3a9caa69113f0e810125ac0e01c43cb71efd048b56894bca2f5b0cbd.jpg)

![](images/4b575d6b166c6d169839a1d274c68c4c9567e380195c9fc233abb6de63151e53.jpg)  
"top right orange"

![](images/1a308406f60d6f6110e266bfdda511dcefbf38c572d6eeab204720c37bbf217d.jpg)

![](images/a34075ca58f1d1b1bdb306c2b3329ce5948a7235061bb0ad2a24a0fbd9136ff1.jpg)

![](images/e8d4367fbc99faede1f08a591b2603351f93743a25297d6eb8b3d8b932ffbac3.jpg)

![](images/b5186c7065b413d8e77868b9fe82384705617d6216f0c6e133f1348a101fbfd9.jpg)

![](images/da6d48c3dc96a3f160a293c74888976347afa5451b996e416e00352a16f42827.jpg)

![](images/34b0e3c2c3e4951019c849d6ce7ccb6b044828dac8d1beae0d63a497c415a608.jpg)

![](images/faefe2b8695b2bf2e0d164e4efb9b89dc1d1df527eab33eacfb29b1a2fe77bf6.jpg)

![](images/df102e39e859e8532e451feb718432be1b3c91a114301d8a7b7fd7adab9c892d.jpg)  
Figure 19: Qualitative examples from the ambiguous subset compared with the RIS models trained under different acquisition strategies.

![](images/de992cf8843c87be5ecb71b31a5ff6e84dcfd670eda3b5c05a3cc27a309a8179.jpg)  
(a) Referred region ambiguity

![](images/a38f949c9d47a463c9c6e4785e4da36affc87c37bb2d34421714886c84d87f71.jpg)  
(b) Pixel-entropy

![](images/5f160cb9d1a3bd48206613918e9a84f050cf5184856c2fbfc25b6784e0022f40.jpg)  
(c) Coreset  
Figure 20: Additional examples of selected samples via acquisitions. The paired GT region-text are denoted by matching colors.