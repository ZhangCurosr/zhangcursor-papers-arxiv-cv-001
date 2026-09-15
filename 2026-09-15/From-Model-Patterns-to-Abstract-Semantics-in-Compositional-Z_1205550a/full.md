# From Model Patterns to Abstract Semantics in Compositional Zero-Shot Learning

Weize Li<sup>1</sup> Zhicheng Zhao<sup>1,2,3†</sup> Fei Su<sup>1,2,3</sup>

<sup>1</sup>Beijing University of Posts and Telecommunications

<sup>2</sup>Beijing Key Laboratory of Network System and Network Culture

<sup>3</sup>Key Laboratory of Interactive Technology and Experience System

{bupt lwz, zhaozc, sufei}@bupt.edu.cn

Abstract—Compositional Zero Shot Learning aims to recognize unseen compositions by recombining learned primitives. Recent methods rely on vision language models and attempt to explicitly model contextual variations of primitives through multiple representations. However, such approaches are limited by fixed variant capacity and competition between abstract and concrete semantics. In this work, we present a new perspective that views primitive variations as the context-driven activation of concrete visual cues rather than independent entities. Based on it, we propose CLEAR, a CLoze-style rEAsoning-based Re-ranking framework inspired by human perceptual processes. CLEAR extracts conditional variants from the primitive candidate set in a coarse-to-fine manner, performs cloze-style reasoning to infer high-level semantics, and re-ranks predictions to correct biases toward salient concrete primitives. Extensive experiments demonstrate that CLEAR consistently improves the Base Model and outperforms state-of-the-art methods on the challenging C-GQA and MIT-States datasets. Code is available at https: //github.com/buptLwz/CLEAR.

Index Terms—compositional zero-shot learning, reasoning, vision-language models, re-ranking

## I. INTRODUCTION

The ability to recognize unseen concepts by recombining previously acquired visual and semantic primitives constitutes a fundamental capability of the human brain. This process, termed compositional generalization, enables humans to efficiently acquire new knowledge. Inspired by it, Compositional Zero-Shot Learning (CZSL) has emerged as an important direction, aiming to endow models with similar generalization abilities. CZSL seeks to disentangle attributes and objects within compositions, so that novel compositions can be correctly recognized. For example, after learning a green bottle and a yellow pear, a model is expected to correctly recognize a green pear and a yellow bottle, even though this new composition have never been encountered during training.

CZSL frameworks are mainly built upon pre-trained vision–language models (VLMs) such as CLIP [1]. By designing task-specific prompts and lightweight adaptations [2], existing methods aim to learn disentangled primitive textual representations. Although CLIP exhibits strong generalization capability, its performance on CZSL remains unsatisfactory. A primary reason lies in the strong contextual dependency of primitive semantics: the same primitive may exhibit substantially different visual manifestations across compositions, such as “old cat” versus “old building”. Even within the same composition, different instances may emphasize different visual aspects of the same primitive, making it difficult to learn a single fixed representation that can simultaneously cover such variations.

![](images/9a3d3c6fab15dc30edb2feec325acad8a572af1e9ce61ccecfb833dac619ba02.jpg)  
Fig. 1. (a) Prior methods enumerate and model conditional variants for each primitive, but the number of variants is limited, and abstract and concrete semantics compete with each other; (b) Our method first constructs a primitive candidate set and then reasons over it to infer high-level semantics. Built from primitive combinations, the candidate set not only has a higher information capacity but also avoids competition between semantics. (c) The proposed CLEAR performs cloze-style reasoning over the candidate set derived from base predictions, and then re-ranks the base predictions to correct errors.

Therefore, recent CZSL approaches attempt to explicitly model such contextual variations as shown in Fig. 1 (a), using techniques such as clustering [3], object-prioritized focus [4], or external structured knowledge [5]. They aim to enumerate contextual variants from the training data as comprehensively as possible, so that the same primitive can be recognized from multiple conditional perspectives.

However, rethinking this issue, we find that the contextual variation of primitives can not be the root cause, but rather a superficial manifestation. Many primitives in CZSL, especially abstract ones, are not atomic concepts, but emerge from the aggregation of multiple concrete visual cues. The conditional variants of a primitive can be viewed as emerging naturally from the context-driven activation of different visual cues. As shown in Fig. 1 (b), “old” in “old building” is inferred from cues such as “weathered” and “cracked”, while “old” in “old cat” is inferred from cues such as “faded” and “wrinkled”.

From this viewpoint, existing methods [3]–[5] essentially attempt to pre-model primitive variations by extracting commonly co-occurring visual patterns at the dataset level. While effective in capturing frequent contextual variations, such approaches inevitably face inherent limitations. First, they approximate primitive variations by a fixed and limited number of representations per primitive, failing to comprehensively cover long-tail unseen combinations or instance-specific finegrained cues. Second, as shown in Fig. 1 (a), existing methods overlook an important fact: in the modern CZSL benchmarks, concrete primitives and abstract primitives typically coexist in the vocabulary and compete during prediction. Even if learned variants can summarize multiple concrete cues, highly salient visual details may still dominate the prediction and induce a bias toward concrete primitives.

To address these issues, inspired by the human perceptual process of progressively focusing when recognizing ambiguous concepts [6], [7], we propose to model recognition as a coarse-to-fine procedure: instead of directly matching an exact answer in the full semantic space, the model first attend to multiple concrete cues and form a set of plausible, instance-specific hypotheses. Second, by reasoning over these hypotheses, it gradually infers the abstract concept that best explains the observed cues, alternatively, when the hypotheses are insufficient, it switches to selecting the most appropriate concrete primitive.

In brief, as shown in Fig. 1 (c), we propose a CLoze-style rEAsoning-based Re-ranking strategy (CLEAR). Specifically, we first employ a Base Model following a standard CZSL formulation, but instead of treating its predictions as the final output, we use the predicted candidate set to cover concrete primitives that are informative for inferring the abstract primitive. Then, CLEAR performs candidate-conditioned clozestyle reasoning in the textual space, mimicking the human process of summarizing abstract categories from plausible hypotheses. Finally, by re-comparing the reasoning results with primitive features, CLEAR re-ranks the Base Model’s predictions, thereby enabling reflective correction of the bias toward concrete primitives induced by salient visual details. Our contributions can be summarized as follows:

1. We provide a new perspective on contextual variation in CZSL by arguing that primitive variants are not independent entities, but rather natural outcomes of context-driven activation of different visual cues.

2. We propose CLEAR, a coarse-to-fine CZSL framework that avoids explicit modeling of conditional structures and comprehensively covers semantic variations through instancespecific candidate sets.

3. CLEAR introduces a re-ranking strategy for CZSL, which effectively mitigates the bias toward concrete primitives induced by salient visual details.

4. CLEAR significantly improves the performance of the Base Model, outperforming SOTA methods on the challenging C-GQA and MIT-States datasets, while achieving competitive results on the simpler UT-Zappos dataset.

## II. RELATED WORK

## A. Compositional Zero-Shot Learning

Recently, VLMs have been increasingly applied to CZSL [8], [9], and the issue of conditional semantics has also garnered attention. CLUSPRO [3] performs clustering in the visual space to enumerate contextual pattens and matches specific instances to prototypes. CPF [4] prioritizes object localization, aggregating visual details across instances to form object-level posterior attribute representations. LOGICZSL [5] leverages LLMs to parse different primitives into external knowledge, explicitly training the conditional relationships. Other approaches [10], [11] often place more emphasis on mitigating modality gaps and capturing visual details.

## B. Re-ranking

Re-ranking has been extensively studied in retrieval systems [12], [13], where it serves as a second-stage refinement to reorder a candidate list under a fixed semantic query. Such approaches focus on improving ranking accuracy by applying more expressive scoring functions, without altering the underlying semantic target. In CZSL, re-ranking acts as a reasoning mechanism that revisits hypotheses and reshapes primitive semantics, instead of merely reordering candidates under a fixed criterion.

## III. METHOD

## A. Problem Formulation

In CZSL, we define an attribute set $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } , . . . , a _ { M } \}$ and an object set $\mathcal { O } = \{ o _ { 1 } , o _ { 2 } , . . . , o _ { N } \}$ . The complete space of compositional concepts is given by their Cartesian product, i.e., $\mathcal { C } = \mathcal { A } \times \mathcal { O }$ . We further consider an image set X, where each image is annotated with a compositional label, and the set of image-associated compositions forms a subset ${ \mathcal { C } } ^ { x } \subseteq { \mathcal { C } }$ The image set X and its corresponding composition set ${ \mathcal { C } } ^ { x }$ are partitioned into two disjoint subsets: the seen compositions $\mathcal { X } _ { s }$ with $\mathcal { C } _ { s } ^ { x }$ , and the unseen compositions $\mathcal { X } _ { u }$ with $\mathcal { C } _ { u } ^ { x } .$ , which defines the training set ${ \mathcal { T } } = \{ ( x , c ) ~ | ~ x \in { \mathcal { X } } ^ { s } , c \in { \mathcal { C } } ^ { s } \}$ , and the test set $\mathcal T ^ { t e } = \{ ( x , c ) \mid x \in \mathcal X , c \in \mathcal C ^ { x } \}$ , respectively. In the open-world setting, the training data and image collection remain unchanged, while the model is required to perform classification over the entire composition space C at test time.

## B. Overview of CLEAR

Inspired by human behavior, CLEAR models the recogni tion process as a pipeline of coarse-grained classification, finegrained reasoning, and re-ranking. Specifically, as illustrated in Fig. 2 (a), the final recognition results (the dashed box in the center) consist of two components: (1) the coarse-grained predictions for attributes, objects, and compositions, denoted as $\mathbf { z } ^ { a } , \mathbf { z } ^ { o }$ , and $\mathbf { z } ^ { c } .$ , respectively; and (2) the re-ranking scores for attributes and objects, denoted as $\hat { \mathbf { z } } ^ { a }$ and $\hat { \mathbf { z } } ^ { o }$ . Notably, the re-ranking is designed for primitives and does not include the compositional branch, where the conditional effects are explicitly constrained by a well-defined context. In CLEAR, the coarse-grained recognition is obtained by a Base Model through cross-modal matching between the visual features and the primitive feature sets. Based on the primitive candidates derived from the coarse-grained predictions and the patch-level visual features $f _ { v } ^ { P }$ , CLEAR employs masked prompts $M ^ { a }$ and $M ^ { o }$ to perform cloze-style reasoning and infer high-level primitive representations. The re-ranking scores are finally computed by matching the high-level primitives with the primitive sets in a single-modality textual space.

![](images/cb4c4a9a20dcdbff9c0282cc2dfb92af8ef189aed2724edfdb820e5bf7c359e6.jpg)  
Fig. 2. (a) Overview of CLEAR. The Base Model adopts a three-branch architecture to jointly fine-tune the CLIP visual encoder with adapters and learn CLIP-based textual primitive prompts $\theta ^ { a } , \theta ^ { o }$ and $\theta ^ { c } .$ . Based on the predictions of the Base Model, CLEAR uses learnable masked tokens $M ^ { a }$ and $M ^ { o }$ to decode high-level semantics from primitive candidates and visual patch features $f _ { v } ^ { P }$ . Finally, the re-ranking scores are computed by matching the highlevel semantics and the primitive features in the textual space. (b) The Reasoner consists of a set of lightweight Transformer decoders, which take masked representations as queries and the candidate set with visual patch features $f _ { v } ^ { P }$ to serve as keys and values.

## C. The Base Model

We introduce a classical three-branch (attribute, object, and composition) CZSL framework as the Base Model to perform coarse-grained recognition, where logits are obtained by crossmodal matching between visual representations and primitive representations in each branch.

Visual Representations. Given a batch of images with batch size B, we employ the CLIP image encoder $E _ { v }$ equipped with lightweight adapters to extract the global image representation $f _ { v } ^ { c l s } ~ \in ~ \mathbb { R } ^ { B \times D }$ , which is directly used as the image feature $f _ { v } ^ { c }$ for the composition branch. To facilitate feature disentanglement, $f _ { v } ^ { c }$ is further processed by two independent MLPs, yielding an attribute-oriented image feature $f _ { v } ^ { a }$ and an objectoriented image feature $f _ { v } ^ { o } ,$ respectively.

Primitive Representations. Inspired by [2], we construct three types of learnable textual prompts for each attribute–object primitive pair $w _ { i } ^ { a } \in \mathcal { A }$ and $w _ { j } ^ { o } ~ \in ~ { \mathcal { O } } _ { \mathrm { : } }$ , as well as their composition $( w _ { i } ^ { a } , w _ { i } ^ { o } ) \in \mathcal { C }$ . Specifically, we define the attribute prompt $\theta _ { i } ^ { a } = [ \bar { p } _ { 0 } ^ { a } , \ldots , p _ { m } ^ { a } , w _ { i } ^ { a } ]$ , the object prompt $\theta _ { j } ^ { o } = [ p _ { 0 } ^ { o } , \ldots , p _ { m } ^ { o } , w _ { j } ^ { o } ]$ , and the composition prompt $\theta _ { i , j } ^ { c } = [ p _ { 0 } ^ { c } , \ldots , p _ { m } ^ { c } , w _ { i } ^ { a } , w _ { j } ^ { o } ]$ . Here, $p _ { 0 : m } ^ { a } , p _ { 0 : m } ^ { o }$ , and $p _ { 0 : m } ^ { c }$ denote learnable prefix tokens, which are initialized with the phrase ${ } ^ { \ast } a$ photo $o f '$ . We employ the CLIP text encoder $E _ { t }$ to encode all primitives, obtaining the corresponding text features $t ^ { a } \in \mathbb { R } ^ { | \boldsymbol { A } | \times \boldsymbol { D } } , t ^ { o } \in \mathbb { R } ^ { | \mathcal { O } | \times \boldsymbol { D } }$ , and $t ^ { c } \in \mathbb { R } ^ { | \mathcal { C } | \times D }$

Three-Branch Basic Logits. Given the visual features and primitive representations from the three branches, for branch $r \in \{ a , o , c \}$ , the base logit of the image n corresponding to class k with the temperature τ can be computed as

$$
z _ { n , k } ^ { r } = { \frac { f _ { v , n } ^ { r } \cdot t _ { k } ^ { r } } { \lVert f _ { v , n } ^ { r } \rVert _ { 2 } \lVert t _ { k } ^ { r } \rVert _ { 2 } } } \cdot { \frac { 1 } { \tau } } ,\tag{1}
$$

## D. Cloze-style Reasoning

Given the predictions of the Base Model, we regard the resulting candidate primitive set as having eliminated most instance-irrelevant categories. In an ideal case, it contains multiple sub-primitives of the instance’s ground-truth (GT) primitive, which jointly characterize the GT primitive (e.g., “tall”, “wide” and “long” collectively describe “large”). However, in practice, the candidate set often includes noisy primitives or suffers from missing ones. Therefore, we design a cloze-style reasoning process to abstract the most appropriate primitive text from this noisy candidate set, and it includes two parts:

1) Masked Representations: Serving as carriers of conditional information for reasoning, following the commonly used Masked Language Modeling (MLM) paradigm, we introduce two additional learnable [MASK] tokens $\bar { m } ^ { a } , m ^ { o } \in \mathbb { R } ^ { 1 \times D }$ which are used to denote the attribute and object to be decoded, respectively. They are initialized using the average pooling of the primitive tokens, namely $\bar { w } ^ { a }$ and $\bar { w } ^ { o }$

Then, we replace the original attribute and object primitive with the corresponding [MASK] token, yielding two masked prompts $M ^ { a } = [ p _ { 0 } ^ { a } , \dots , p _ { m } ^ { a } , m ^ { a } ]$ and $M ^ { o } = [ p _ { 0 } ^ { o } , \ldots , p _ { m } ^ { o } , m ^ { o } ]$ Here, the prefix tokens are kept identical to those in the Base Model to ensure that the masked token reside in the same feature space as the corresponding primitives.

However, unlike MLM, we typically align primitives with images at the sentence level in CZSL. Consequently, recovering the word at the [MASK] position in the same way as

MLM thus contradicts the basic training objective. Therefore, we instead use the sentence-level features $\hat { t } ^ { a }$ and $\hat { t } ^ { o }$ encoded by the CLIP text encoder as the final masked representations for subsequent reasoning, i.e.,

$$
\hat { t } ^ { a } = E _ { t } ( M ^ { a } ) , \quad \hat { t } ^ { o } = E _ { t } ( M ^ { o } )\tag{2}
$$

2) Reasoner: During reasoning, the masked representations are progressively transformed into high-level primitives, both of which reside in the same CLIP feature space. Therefore, as shown in Fig. 2 (b), we implement the Reasoner as a lightweight 6-layer Transformer decoder.

Concretely, taking the attribute branch as an example, given a batch of images of batch size B, we first repeat the masked representations $\hat { t } ^ { a }$ along the batch dimension as queries. Then, based on the Base Model’s prediction $p ( a \mid f _ { v } ^ { a } ) \bar { \in } \mathbb { R } ^ { B \times | A | }$ , we select, for each instance, the top-K primitives with the highest predicted probabilities from the attribute primitive set $t ^ { a }$ , to form a candidate set $t _ { C a n } ^ { a } \in \mathbb { R } ^ { B \times K \times D }$ , i.e.,

$$
I n d e x = T o p K \big ( p ( a \mid f _ { v } ^ { a } ) \big ) , t _ { C a n } ^ { a } = t ^ { a } [ I n d e x ] .\tag{3}
$$

In addition, we further feed the fine-grained visual features from CLIP, denoted as $f _ { v } ^ { P } \in \mathbb { R } ^ { B \times S \times D }$ , into the reasoning process, thereby enabling the model to identify the most suitable concrete primitive based on image details when the candidate set is insufficient, where S denotes the number of CLIP patches. As a result, the complete contextual feature $f _ { k v } \in \mathbb { R } ^ { B \times ( K + S ) \times D }$ is constructed by concatenating the candidate primitives $t _ { C a n } ^ { a }$ and the patch-level visual features:

$$
f _ { k v } = C a t ( t _ { C a n } ^ { a } ; f _ { v } ^ { P } ) ,\tag{4}
$$

where $C a t ( \cdot )$ denotes the concatenation operation. In practice, K is defined as a fixed proportion of the total number of primitive categories in each branch, and its effect is further analyzed in Sec. IV-C1.

## E. Re-ranking and Loss

After the reasoning process, the masked representations $\hat { t } ^ { a }$ and $\hat { t } ^ { o }$ aggregate visual details and prior semantic knowledge, yielding the primitive variants $\hat { t } _ { o u t } ^ { a }$ and $\hat { t } _ { o u t } ^ { o } .$ Although the reasoning is conditioned on the candidate set, it aims to summarize the most appropriate primitive; therefore, $\hat { t } _ { o u t } ^ { a }$ and $\hat { t } _ { o u t } ^ { o }$ may exhibit primitive characteristics beyond the candidate set. Therefore, taking the attribute branch as an example, unlike classical re-ranking strategies that only reweight the candidate score, we compute re-ranking scores $\hat { \mathbf { z } } ^ { a } \in \breve { \mathbb { R } } ^ { B \times | \mathcal { A } | }$ over the entire primitive space, i.e.,

$$
\hat { \mathbf { z } } ^ { a } = \frac { \hat { t } _ { o u t } ^ { a } \cdot t ^ { a } } { \left\| \hat { t } _ { \mathrm { o u t } } ^ { a } \right\| _ { 2 } \left\| t ^ { a } \right\| _ { 2 } } \cdot \frac { 1 } { \tau } ,\tag{5}
$$

Here, $\tau$ is the temperature parameter. We perform re-ranking directly at the logit level, so that the Base Model is supervised only through the fused results, granting some tolerance to its coarse-grained classification. Let $y _ { n } ^ { a } \in \{ 1 , \ldots , | A | \}$ denote the ground-truth attribute label of the n-th sample, the final loss is defined using a standard cross-entropy loss:

$$
\hat { \mathcal { L } } ^ { a } = - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } p ( y _ { n } ^ { a } | n ) ,\tag{6}
$$

$$
p ( a _ { i } | n ) = \frac { \exp ( z _ { n , a _ { i } } ^ { a } + \hat { z } _ { n , a _ { i } } ^ { a } ) } { \sum _ { k = 1 } ^ { | A | } \exp ( z _ { n , k } ^ { a } ) } .\tag{7}
$$

The object branch is trained in the same manner, by replacing $( a , \mathcal { A } , \mathbf { z } ^ { a } , \hat { \mathbf { z } } ^ { a } )$ with $( o , \mathcal { O } , { \bf z } ^ { o } , \hat { { \bf z } } ^ { o } )$ . The compositional branch adopts the loss function of the Base Model:

$$
\mathcal { L } ^ { c } = - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } p ( y _ { n } ^ { c } | n ) = - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { \exp ( z _ { n , y _ { n } ^ { a } } ^ { c } ) } { \sum _ { k = 1 } ^ { | \mathcal { C } ^ { s } | } \exp ( z _ { n , k } ^ { c } ) } .\tag{8}
$$

The overall learning objective is defined as

$$
\mathcal { L } = \lambda ^ { a } \hat { \mathcal { L } } ^ { a } + \lambda ^ { o } \hat { \mathcal { L } } ^ { o } + \lambda ^ { c } \mathcal { L } ^ { c } ,\tag{9}
$$

where $\lambda ^ { a } = \lambda ^ { o } = \lambda ^ { c } = 1$ , following empirical practice. The final probability for a given sample is calculated as the sum of the attribute, object and composition probability:

$$
p ^ { \prime } ( c _ { i , j } | x ) = p ( c _ { i , j } | x ) + p ( a _ { i } | x ) + p ( o _ { j } | x ) .\tag{10}
$$

## IV. EXPERIMENT

## A. Experiment Setup

Datasets and Metrics. We evaluate CLEAR on three widely used benchmarks for CZSL: UT-Zappos [14], MIT-States [15], and C-GQA [16]. Following prior works [16], we adopt the standard dataset splits for fair comparison and report Seen (S) and Unseen (U) composition accuracy, the Harmonic Mean (HM) of Seen and Unseen accuracy, and the Area Under Curve (AUC), with AUC being the most representative metric.

Implementation Details. CLEAR is trained for 15 epochs with Adam optimizer for all datasets. We jointly train the Base Model and the reasoner rather than in a stage-wise manner, using the pre-trained CLIP ViT-L/14 model [1] as both the text and image encoder. All training and evaluation are conducted on 2 NVIDIA 4090 GPUs.

## B. Main Results

To intuitively illustrate CLEAR’s capability in addressing the contextual dependency of primitives, we compare CLEAR with state-of-the-art methods that are explicitly designed for the same problem in Table I, including CLUSPRO [3], LOG-ICZSL [5], and CPF [4]. They respectively model primitive contextual dependencies through clustering, the incorporation of external knowledge, and an object-prioritized focus. Note that CPF reports results only on C-GQA, we thus reproduce its performance on the other two datasets.

The results show that CLEAR’s advantage is positively correlated with the scale of the dataset. In particular, on the large-scale C-GQA benchmark, CLEAR significantly and consistently outperforms all other SOTA approaches, demonstrating that our strategy of decoupling conditional information via candidate sets is better suited for complex real-world scenarios. On MIT-States, CLEAR still maintains a performance advantage. However, on the small-scale UT-Zappos dataset, CLEAR outperforms CPF and LOGICZSL, which uses additional knowledge, but only achieves comparable performance to the clustering-based CLUSPRO. This may be attributed to the fact that when the number of primitives is extremely small, modeling-based approaches can easily enumerate and capture almost all the compositional patterns seen.

TABLE I  
THE EXPERIMENTAL RESULTS FOR BOTH CLOSED/OPEN-WORLD SETTINGS. THE BEST PERFORMANCE ARE HIGHLIGHTED IN BOLD.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">C-GQA</td><td colspan="4">MIT-States</td><td colspan="4"> $\mathrm { U T - Z a p p o s }$ </td></tr><tr><td>S</td><td>U</td><td>HM</td><td>AUC</td><td>S</td><td>U</td><td>HM</td><td>AUC</td><td>S</td><td>U</td><td>HM</td><td>AUC</td></tr><tr><td colspan="10">Closed-world Results</td><td></td><td></td><td></td><td></td></tr><tr><td>PLID [8]</td><td>ECCV&#x27;24</td><td>38.8</td><td>33.0</td><td>27.9</td><td>11.0</td><td>49.7</td><td>52.4</td><td>39.0</td><td>22.1</td><td>67.3</td><td>68.8</td><td>52.4</td><td>38.7</td></tr><tr><td>CDS-CZSL [9]</td><td>CVPR&#x27;24</td><td>38.3</td><td>34.2</td><td>28.1</td><td>11.1</td><td>50.3</td><td>52.9</td><td>39.2</td><td>22.4</td><td>63.9</td><td>74.8</td><td>52.7</td><td>39.5</td></tr><tr><td>Troika [2]</td><td>CVPR&#x27;24</td><td>41.0</td><td>35.7</td><td>29.4</td><td>12.4</td><td>49.0</td><td>53.0</td><td>39.3</td><td>22.1</td><td>66.8</td><td>73.8</td><td>54.6</td><td>41.7</td></tr><tr><td>IMAX [11]</td><td>TPAMI&#x27;25</td><td>39.7</td><td>35.8</td><td>29.8</td><td>12.8</td><td>48.7</td><td>53.8</td><td>39.1</td><td>21.9</td><td>69.3</td><td>70.7</td><td>54.2</td><td>40.6</td></tr><tr><td>CLUSPRO [3]</td><td>ICLR’25</td><td>44.3</td><td>37.8</td><td>32.8</td><td>14.9</td><td>52.1</td><td>54.0</td><td>40.7</td><td>23.8</td><td>70.7</td><td>76.0</td><td>58.5</td><td>46.6</td></tr><tr><td>LOGICZSL [5]</td><td>CVPR&#x27;25</td><td>44.4</td><td>39.4</td><td>33.3</td><td>15.3</td><td>50.8</td><td>53.9</td><td>40.5</td><td>23.4</td><td>69.6</td><td>74.9</td><td>57.8</td><td>45.8</td></tr><tr><td>CPF [4]</td><td>ICCV&#x27;25</td><td>44.8</td><td>39.6</td><td>33.6</td><td>15.4</td><td>51.6</td><td>53.4</td><td>40.3</td><td>23.4</td><td>69.3</td><td>76.3</td><td>57.9</td><td>45.8</td></tr><tr><td colspan="2">Base Model</td><td>38.5</td><td>33.2</td><td>27.9</td><td>11.0</td><td>49.2</td><td>52.6</td><td>38.7</td><td>21.8</td><td>64.4</td><td>70.7</td><td>51.9</td><td>37.8</td></tr><tr><td colspan="2">CLEAR (Ours)</td><td>45.5</td><td>41.9</td><td>34.9</td><td>16.6</td><td>52.4</td><td>53.7</td><td>40.6</td><td>24.0</td><td>69.9</td><td>76.4</td><td>58.2</td><td>46.3</td></tr><tr><td colspan="2">Δ</td><td>↑7.0</td><td>↑8.7</td><td>↑7.0</td><td>↑5.6</td><td>↑3.2</td><td>↑1.1</td><td>↑1.9</td><td>↑2.2</td><td>↑5.5</td><td>↑5.7</td><td>↑6.3</td><td>↑8.5</td></tr><tr><td colspan="10">Open-world Results</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">PLID [8] ECCV&#x27;24</td><td>39.1</td><td>7.5</td><td>10.6</td><td>2.5</td><td>49.1</td><td>18.7</td><td>20.4</td><td>7.3</td><td>67.6</td><td>55.5</td><td>46.6</td><td>30.8</td></tr><tr><td>CDS-CZSL [9] Troika [2]</td><td>CVPR&#x27;24</td><td>37.6</td><td>8.2</td><td>11.6</td><td>2.7</td><td>49.4</td><td>21.8</td><td>22.1</td><td>8.5</td><td>64.7</td><td>61.3</td><td>48.2</td><td>32.3</td></tr><tr><td></td><td>CVPR&#x27;24</td><td>40.8</td><td>7.9</td><td>10.9</td><td>2.7</td><td>48.8</td><td>18.4</td><td>20.1</td><td>7.2</td><td>66.4</td><td>61.2</td><td>47.8</td><td>33.0</td></tr><tr><td>IMAX [11]</td><td>TPAMI&#x27;25</td><td>38.7</td><td>7.9</td><td>11.2</td><td>2.5</td><td>50.2</td><td>18.6</td><td>21.4</td><td>7.6</td><td>68.4</td><td>57.3</td><td>47.5</td><td>32.3</td></tr><tr><td>CLUSPRO [3]</td><td>ICLR’25</td><td>41.6</td><td>8.3</td><td>11.6</td><td>3.0</td><td>51.2</td><td>22.1</td><td>23.0</td><td>9.3</td><td>71.0</td><td>66.2</td><td>54.1</td><td>39.5</td></tr><tr><td>LOGICZSL [5]</td><td>CVPR&#x27;25 ICCV’25</td><td>43.7</td><td>9.3</td><td>12.6</td><td>3.4</td><td>50.7</td><td>21.4</td><td>22.4</td><td>8.7</td><td>69.6</td><td>63.7</td><td>50.8</td><td>36.2</td></tr><tr><td colspan="2">CPF [4]</td><td>44.5</td><td>9.3</td><td>13.0</td><td>3.6</td><td>51.0</td><td>20.8</td><td>22.2</td><td>8.6</td><td>69.3</td><td>63.8</td><td>50.7</td><td>36.6</td></tr><tr><td colspan="2">Base Model</td><td>38.9</td><td>8.1</td><td>11.3</td><td>2.6</td><td>48.5</td><td>18.0</td><td>19.8</td><td>7.0</td><td>66.8</td><td>59.0</td><td>47.6</td><td>32.5</td></tr><tr><td colspan="2">CLEAR (Ours)</td><td>45.8</td><td>11.5</td><td>15.4</td><td>4.6</td><td>52.4</td><td>21.5</td><td>22.7</td><td>9.1</td><td>68.7</td><td>65.8</td><td>52.6</td><td>38.1</td></tr><tr><td colspan="2">Δ</td><td>↑6.9</td><td>↑3.4</td><td>↑4.1</td><td>↑2.0</td><td>↑3.9</td><td>↑3.5</td><td>↑2.9</td><td>↑2.1</td><td>↑1.9</td><td>↑6.8</td><td>↑5.0</td><td>↑5.6</td></tr></table>

In addition, we also report the performance of the independently trained Base Model in Table I. Without extra components, its performance is even lower than that of the commonly used baseline Troika [2]. CLEAR consistently yields substantial improvements over the Base Model across all three datasets, indicating that re-ranking effectively guides the Base Model to capture complex contextual information.

## C. Ablation Study

![](images/a2211577a1fa7bc697b2d937a910e6af6655a51e303900791c6fb2dd76a4e937.jpg)  
Fig. 3. Effect of the TopK ratio K on model performance. The AUC on each dataset is normalized by its maximum value to facilitate comparison.

1) The Top-K in CLEAR: We investigate the effect of the TopK parameter K in Fig. 3. In CLEAR, the K for the attribute and object branches are computed by multiplying their respective category cardinalities by a given ratio. We perform a grid search over different ratio on three datasets.

For clarity, we normalize the AUC of each dataset by its own maximum value to highlight relative performance trends. As shown in Fig. 3, increasing the ratio from 0.05 to 1.0 leads to a consistent pattern across all datasets, where performance first improves and then degrades. This observation suggests that K plays a critical role in balancing candidate coverage and noise introduction: a small K tends to miss instance-relevant primitives, whereas an excessively large K introduces many irrelevant candidates. Due to differences in the total number of primitives and the strength of their internal conditional dependencies, the optimal K varies across datasets. The bestperforming ratios are 0.1 on C-GQA, 0.3 on MIT-States and 0.4 on UT-Zappos, respectively.

TABLE II  
THE ABLATION STUDY ABOUT REASONING CONFIGURATION.
<table><tr><td>Candidate Context</td><td>Visual Context</td><td>CLIP-Encoded Query</td><td>Type Embedding</td><td>AUC↑</td></tr><tr><td>x</td><td>x</td><td>x</td><td>x</td><td>11.0</td></tr><tr><td>√</td><td>x</td><td>x</td><td>x</td><td>14.2</td></tr><tr><td>√</td><td>√</td><td>x</td><td>x</td><td>15.6</td></tr><tr><td>√</td><td>√</td><td>√</td><td>x</td><td>16.6</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>16.1</td></tr></table>

2) Reasoning Configuration: We evaluate the reasoning configuration on the C-GQA dataset in Table II, with all experiments performed under the optimal K. Here, the candidate and visual contexts correspond to different components of $f _ { k v } .$ The $\mathbf { \cdots } \mathbf { x } ^ { , , }$ in “CLIP-Encoded Query” indicates that the masked representations $\hat { t } ^ { a }$ are replaced with directly learnable parameters. “Type Embedding” is a learnable embedding added to $f _ { k v }$ to indicate the positions of different modalities.

As shown in Table II, introducing candidate context yields a substantial improvement over the Base Model (11.0→14.2 AUC), indicating that candidate primitives provide a critical semantic prior. Further incorporating visual context leads to additional gains, suggesting that patch-level visual details offer necessary information. The best performance is achieved when CLIP-encoded queries are used together with both candidate and visual contexts (16.6 AUC). This comparison highlights the importance of projecting masked prompts into the same textual feature space as primitives, which helps maintain representation consistency during reasoning. In contrast, adding type embeddings slightly degrades performance, which can be attributed to the interference introduced by explicit modality cues in the highly aligned CLIP feature space.

![](images/0456655dabeb0d1ba3d5d741b164a7ddde6716dff2f4e68474983d6b7286bedb.jpg)

<table><tr><td rowspan="3">(b) Gain</td><td>AUC↑</td><td>C-GQA</td><td></td><td>MIT-States UT-Zappos</td></tr><tr><td>Base Model †</td><td>15.7</td><td>22.5</td><td>45.4</td></tr><tr><td>After Re-ranking</td><td>16.6</td><td>24.0</td><td>46.3</td></tr></table>

Fig. 4. Re-ranking effects of CLEAR. (a) Representative cases comparing the Base Model outputs before re-ranking with the final predictions. (b) Quantitative AUC gains of re-ranking across three datasets.

3) Re-ranking and Visualization: Fig. 4 provides an analysis of the proposed re-ranking strategy. In this figure, different from Table I, the reported Base Model<sup>†</sup> here refers to the predictions of the Base Model after joint training with CLEAR, but before applying the re-ranking module. As shown in Fig. 4 (a), the Base Model can be misled by salient but concrete visual cues, resulting in incorrect predictions. In contrast, CLEAR effectively corrects these errors by leveraging conditional reasoning to infer high-level semantics. Fig. 4 (b) further reports the quantitative improvements brought by re-ranking on three datasets. The consistent AUC gains demonstrate that CLEAR does not merely enhance the knowledge of the Base Model but exhibits an independent error-correction ability.

## V. CONCLUSION

In this paper, we revisit CZSL from the perspective that contextual variations emerge from instance-specific visual cues rather than fixed conditional forms. Based on this insight, we propose CLEAR, a coarse-to-fine framework that performs cloze-style reasoning over candidate primitives and re-ranks base predictions to mitigate biases toward concrete semantics. By reasoning from plausible hypotheses instead of explicitly enumerating primitive variants, CLEAR achieves more flexible and robust compositional generalization.

## ACKNOWLEDGMENT

This work was supported by the Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing.

## REFERENCES

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 8748–8763. [Online]. Available: https://proceedings.mlr.press/v139/radford21a.html

[2] S. Huang, B. Gong, Y. Feng, M. Zhang, Y. Lv, and D. Wang, “Troika: Multi-path cross-modal traction for compositional zero-shot learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024.

[3] H. Qu, J. Wei, X. Shu, and W. Wang, “Learning clustering-based prototypes for compositional zero-shot learning,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=eE2PXlNydB

[4] P. Wu, Q. Lai, H. Fang, G.-S. Xie, Y. Yin, X. Lu, and W. Wang, “A conditional probability framework for compositional zero-shot learning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025.

[5] P. Wu, X. Lu, H. Hu, Y. Xian, J. Shen, and W. Wang, “Logiczsl: Exploring logic-induced representation for compositional zero-shot learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 30 301–30 311.

[6] L. Muttenthaler, K. Greff, F. Born, B. Spitzer, S. Kornblith, M. C. Mozer, K.-R. Muller, T. Unterthiner, and A. K. Lampinen, “Aligning¨ machine and human visual representations across abstraction levels,” Nature, 2025.

[7] M. N. Hebart, C. Y. Zheng, F. Pereira, and C. I. Baker, “Revealing the multidimensional mental representations of natural objects underlying human similarity judgements,” Nature human behaviour, 2020.

[8] W. Bao, L. Chen, H. Huang, and Y. Kong, “Prompting languageinformed distribution for compositional zero-shot learning,” in Proceedings of the European Conference on Computer Vision (ECCV), 2024.

[9] Y. Li, Z. Liu, H. Chen, and L. Yao, “Context-based and diversity-driven specificity in compositional zero-shot learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[10] S. Zhang, C. Yan, Y. Liu, C. Jing, L. Zhou, and W. Wang, “Learning visual proxy for compositional zero-shot learning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025.

[11] C. Jiang, S. Wang, Y. Long, Z. Li, H. Zhang, and L. Shao, “Imaginaryconnected embedding in complex space for unseen attribute-object discrimination,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 3, pp. 1395–1413, 2025.

[12] Q. Liu, B. Wang, N. Wang, and J. Mao, “Leveraging passage embeddings for efficient listwise reranking with large language models,” in Proceedings of the ACM on Web Conference 2025, ser. WWW ’25. New York, NY, USA: Association for Computing Machinery, 2025. [Online]. Available: https://doi.org/10.1145/3696410.3714554

[13] R. Nogueira, Z. Jiang, R. Pradeep, and J. Lin, “Document ranking with a pretrained sequence-to-sequence model,” in Findings of the Association for Computational Linguistics: EMNLP 2020, T. Cohn, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, Nov. 2020, pp. 708–718. [Online]. Available: https://aclanthology.org/2020.findings-emnlp.63/

[14] A. Yu and K. Grauman, “Fine-grained visual comparisons with local learning,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2014, pp. 192–199.

[15] P. Isola, J. J. Lim, and E. H. Adelson, “Discovering states and transformations in image collections,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015.

[16] M. F. Naeem, Y. Xian, F. Tombari, and Z. Akata, “Learning graph embeddings for compositional zero-shot learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.