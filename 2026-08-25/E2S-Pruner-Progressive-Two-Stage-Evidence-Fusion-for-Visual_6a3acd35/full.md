# E2S-Pruner: Progressive Two-Stage Evidence Fusion for Visual Token Pruning in Vision-Language Models

Taoyu Qian<sup>1</sup> Qi Wang<sup>∗1</sup> Daqian Shi<sup>2</sup> Yuanhao Jiang<sup>3,4</sup> Shang Gao<sup>1</sup> Hualong Yu<sup>1</sup>

<sup>1</sup>School of Computer Science and Engineering, Jiangsu University of Science and Technology

<sup>2</sup>Digital Environment Research Institute (DERI), Queen Mary University of London

<sup>3</sup>Shanghai Institute of Artificial Intelligence for Education, East China Normal University

<sup>4</sup>National Institute of Education, Nanyang Technological University

Corresponding author: wangqi@just.edu.cn

## Abstract

Vision-language models typically encode an image into hundreds of visual tokens, incurring substantial inference latency and GPU memory overhead. Existing pruning methods largely rely on attention scores and directly aggregate outputs across attention heads and network layers, making it difficult to characterize evidential uncertainty and conflict. We propose E2S-Pruner, a progressive two-stage evidence-fusion framework for visual token pruning that requires no auxiliary model, trainable parameters, orfinetuning. In thefirst stage, E2S-Pruner treats each attention head as an independent evidence source, estimates its reliability from evidence clarity and inter-head consistency, and represents each visual token using three states: important, unimportant, and uncertain. In the second stage, Dempster–Shafer evidence theory is used to quantify interlayer conflict andfuse complementary evidencefrom multiple network layers. We further introduce a spatial novelty constraint that promotes coverage ofdistinct image regions and prevents the retained tokens from concentrating in a few locally salient areas. On LLaVA-1.5-7B, E2S-Pruner retains 98.0%, 96.8%, and 90.6% ofthe aggregate performance when the average numbers of retained visual tokens are 192, 128, and 64, respectively, while improving throughput by 1.96× and 2.09× under the 128-token and 64-token settings. Experiments on Qwen2-VL-7Bfurther demonstrate cross-model generalization. Code is available at https: //github.com/taoyu-qian/E2S-Pruner.git.

## 1. Introduction

Recent advances in multimodal learning and large-scale pretraining have driven vision-language models (VLMs) from early exploration to rapid capability gains[12, 23]. Meanwhile, edge-computing applications, including unmanned aerial vehicles[25] and autonomous driving systems[15], increasingly rely on the cross-modal understanding and reasoning capabilities of VLMs.

Covering the diversity and complexity of real-world tasks requires VLMs to learn from massive training corpora. Absorbing and representing such rich information generally entails a correspondingly large parameter count and substantial computation, which limits deployment in resourceconstrained edge environments. This bottleneck has motivated a growing body of work on efficient VLM and visual Transformer architectures[8, 21, 24].

Most contemporary large language models (LLMs) are built on the Transformer architecture[22], whose computational cost depends strongly on the input sequence length. Compressing redundant tokens can therefore reduce inference cost while preserving task performance. Existing tokencompression methods fall broadly into two categories: (i) similarity-based token merging, represented by ToME[2], and (ii) attention-score-based token pruning, represented by FastV[4]. The latter directly exploits attention information produced during inference and has consequently become a dominant approach. We adopt visual token pruning to improve VLM efficiency; its basic principle is introduced in Section 3.1.

We identify a limitation of existing approaches, illustrated in Fig. 1. Different attention heads and network layers exhibit distinct visual attention patterns[5]. Early LLM layers tend to capture low-level features such as local textures and edges, middle layers emphasize object-level features such as an individual road sign, and deep layers focus more strongly on global semantic structure. These representations therefore progress from local detail to high-level semantics[1, 4]. To exploit this diversity, existing methods typically aggregate attention weights across heads and layers and retain the top-K tokens according to the resulting scores. Consider the example at the top of Fig. 1, where three tokens receive scores of (0.02, 0.65, 0.62), (0.84, 0.61, 0.60), and (0.86, 0.61, 0.64) in an early, middle, and deep layer, respectively. The first token is strongly supported by object-level and semantic evidence but lacks texture-level support. Simple averaging assigns it a score of only 0.57, substantially weakening its apparent importance and increasing the risk of erroneous pruning. Thus, direct aggregation does not adequately model the joint distribution of evidence across heads and layers. A principled fusion mechanism that preserves complementary and conflicting information is required.

![](images/25b93adc23d0408eb4229b44e7474bd46c3be592f74316f90c23bdb4a7257124.jpg)  
Figure 1. Motivation and overview of E2S-Pruner. Direct aggregation can suppress tokens supported by complementary evidence across layers, whereas the proposed two-stage fusion explicitly models evidential uncertainty and conflict.

Unlike conventional probability theory, which assigns precise probabilities to individual hypotheses, Dempster– Shafer (D–S) evidence theory[10] supports belief assignment over sets of hypotheses and explicitly represents uncertainty during evidence fusion. It therefore provides a mathematical basis for modeling uncertain and conflicting information from multiple heads and layers. Building on this property, we introduce D–S evidence theory into visual token pruning and propose E2S-Pruner, a progressive two-stage evidencefusion method. In the first stage, evidence from all heads within each selected early, middle, or deep LLM layer is fused to construct a D–S frame of discernment with three states: important, unimportant, and uncertain. In the second stage, inter-layer conflict is quantified and the layer-wise frames are recursively fused to obtain a global decision.

The main contributions of this work are summarized as follows:

• We introduce D–S evidence theory into VLM token pruning. At the attention-head level, an evidence reliability measure is defined for each visual token to characterize agreement among heads, and the resulting reliability is used to construct a layer-wise D–S frame of discernment.

• We define an inter-layer conflict measure and use it to fuse evidence from multiple layer-wise frames. Belief and plausibility for visual-token importance are then derived within the D–S framework to determine token retention priority.

![](images/4e4619a55fcc734127522eda09b0818c75769bf845607f1e082884e0db27c19d.jpg)  
Figure 2. Radar-chart comparison of E2S-Pruner on LLaVA-1.5-7B across the evaluated benchmarks, showing the balance between task performance and aggressive visual-token reduction.

• We propose a spatial novelty constraint to prevent retained tokens from concentrating in a small image region. The image is partitioned into spatial cells, and a soft bonus is assigned to the highest-priority token in each cell to improve spatial coverage.

## 2. Related Work

As image resolution and visual context length increase, visual tokens become a major source of VLM inference cost[27]. Existing visual token compression techniques can be broadly categorized as token merging or token pruning[18]. ToME[2] progressively merges redundant tokens according to feature similarity and reduces Transformer computation without retraining. However, similarity-based merging lacks explicit guidance from the text instruction and may prematurely combine regions that are visually similar but semantically distinct. LLaVA-PruMerge[19] uses sparse class-to-patch attention in the vision encoder to identify key visual tokens and aggregates removed tokens into retained ones according to key-feature similarity, thereby mitigating information loss caused by direct pruning.

Another line of work estimates visual-token importance from internal attention or semantic relevance. FastV[4] observes that deep LLM layers assign little attention to many visual tokens and therefore performs one-shot pruning in a shallow layer to reduce subsequent self-attention and feedforward computation. HiRED[1] targets high-resolution inputs by using class-token attention in the vision encoder to allocate token budgets dynamically across image partitions and selecting informative tokens within each partition. SparseVLM[28] identifies text tokens related to visual content, uses the LLM self-attention matrix to score visual tokens, and combines progressive sparsification, dynamic budget allocation, and token recycling. These methods establish the utility of attention for visual-token selection, but they generally aggregate scores directly across heads or layers without explicitly modeling source reliability, conflict, or uncertainty.

Beyond saliency scoring, several studies reconsider visual token compression from the perspectives of redundancy, representation dynamics, and causal association. DART[26] argues that token importance alone is not a sufficiently reliable pruning criterion and retains complementary information by measuring redundancy between candidate tokens and a small set of pivot tokens. V2Drop[3] progressively prunes tokens according to representation changes between adjacent LLM layers, preferentially removing slowly changing tokens to reduce positional bias while remaining compatible with efficient attention kernels. CaVIN[16] further notes that conventional relevance scores may overlook dependencies between target regions and their context; it therefore learns visual-token associations from weak causal signals in chain-of-thought data and recovers contextual tokens that conventional pruning may discard.

Although these approaches improve compression through attention saliency, feature similarity, information redundancy, inter-layer dynamics, or contextual dependence, evidence quality varies across attention heads and network layers. Simple summation or averaging can obscure local details, complementary semantics, and potentially important visual evidence associated with high conflict. E2S-Pruner addresses this limitation by jointly modeling multi-head and multilayer information with D–S evidence theory and by improving regional coverage through a spatial constraint, yielding a more stable performance–efficiency trade-off across token budgets. Figure 2 summarizes its performance on LLaVA-1.5-7B across multiple benchmarks.

## 3. Preliminary

## 3.1. Attention-Score-Based Pruning

A VLM comprises a vision encoder and an LLM. Images and text are encoded as visual and text tokens, respectively, before being passed to the LLM. Because visual tokens substantially outnumber text tokens, pruning is applied primarily to the visual sequence. Attention-based pruning has several variants; in this work, visual-token importance is measured using text-to-visual attention. This head-level representation exploits the interaction between the attention mechanism and the CLIP vision encoder[17]: intermediate visual tokens encode rich image features while incorporating semantic guidance from the textual prompt, providing discriminative evidence for token selection.

For the h-th head in the l-th LLM layer, let $N _ { T }$ and $N _ { V }$ denote the numbers of text and visual tokens, respectively, where $N _ { V } \gg N _ { T }$ . The two token sets are $\mathcal { T } =$ $\{ T _ { 1 } , T _ { 2 } , \dots , T _ { N _ { T } } \}$ and $\mathcal { V } = \{ V _ { 1 } , V _ { 2 } , \ldots , V _ { N _ { V } } \}$ . The attention score of the n-th visual token in this head is

$$
a _ { l , h , n } = \frac { 1 } { N _ { T } } \sum _ { t \in \mathcal { T } } \frac { Q _ { t } K _ { v _ { n } } ^ { T } } { \sqrt { d } }\tag{1}
$$

where $Q _ { t }$ is the query of the t-th text token, $K _ { v _ { n } }$ is the key of the n-th visual token, and d is the head dimension.

Conventional pruning methods first aggregate the singlehead scores $^ { a _ { l , h , n } }$ across heads and then across layers to obtain one attention-based importance score for each visual token. Tokens are ranked by this score, and the top K are retained.

## 3.2. D-S Theory

For a decision problem, the set of all possible hypotheses is called the frame of discernment and is denoted by Θ. Evidence comprises observations, computed quantities, or prior knowledge that support, oppose, or remain uncertain about propositions in this frame. Different information sources may yield distinct assessments of the same problem; n such sources form a collection $\mathcal { E } = \{ \Theta _ { 1 } , \Theta _ { 2 } , . . . , \Theta _ { n } \}$ . D–S evidence theory provides a principled mechanism for fusing these sources.

For any subset $A \subseteq \Theta$ , the basic probability assignment $m ( A )$ quantifies the support assigned to proposition A and satisfies

$$
\sum _ { A \subset \Theta } ^ { m ( \emptyset ) = 0 } m ( A ) = 1\tag{2}
$$

Given propositions B and C from two evidence sources, their conflict degree is defined as

$$
K = \sum _ { B \cap C = \emptyset } m _ { 1 } ( B ) m _ { 2 } ( C ) \qquad K \in [ 0 , 1 ]\tag{3}
$$

This expression sums the products of masses assigned to mutually exclusive proposition pairs and thereby quantifies global conflict between the two sources. A value of K closer to 1 indicates stronger disagreement.

Dempster’s rule combines the two sources into the following mass function:

$$
m ( A ) = \frac { \displaystyle \sum _ { B \cap C = A } m _ { 1 } ( B ) m _ { 2 } ( C ) } { 1 - K } , \quad A \neq \emptyset\tag{4}
$$

Unlike conventional probability theory, D–S theory allows mass to be assigned directly to a composite proposition. For example,

$$
m ( \{ A , B \} ) = 0 . 4\tag{5}
$$

indicates that the evidence supports “the outcome is A or $B ^ { \prime \prime }$ but cannot distinguish between the two. Similarly, m(Θ) represents mass assigned to complete uncertainty.

Given a mass function, the belief function accumulates all masses that fully support A and therefore gives a lower bound on its support:

$$
\operatorname { B e l } ( A ) = \sum _ { B \subseteq A } m ( B )\tag{6}
$$

The plausibility function accumulates all masses that do not contradict A and therefore gives an upper bound on its support:

$$
\operatorname { P l } ( A ) = \sum _ { B \cap A \neq \emptyset } m ( B )\tag{7}
$$

## 4. Methodology

This section presents the E2S-Pruner algorithm. Its overall workflow is illustrated in Fig. 1, and the pseudocode is provided in Algorithm 1. Three pruning operations are deployed in shallow, middle, and deep LLM layers. The visual-token count decreases stepwise as pruning progresses, producing a coarse-to-fine compression schedule.

## 4.1. Attention-head Fusion

To address inconsistent attention responses across heads and layers, we use D–S evidence theory to model their conflict explicitly. This subsection introduces the first stage: multihead evidence fusion.

Each selected LLM layer yields a frame of discernment Θ for every candidate token. The frames from k pruning layers form the collection $\mathcal { E } = \{ \Theta _ { 1 } , \Theta _ { 2 } , \dots , \Theta _ { k } \}$ . Each layer-wise frame is constructed as follows.

Algorithm 1 E2S-Pruner   
Require: Text tokens T ; visual tokens V; pruning layers P;   
budgets $\{ K _ { l } ^ { \mathrm { k e e p } } \} ; G = 4$ and $\lambda = 0 . 1$   
Ensure: Retained visual tokens V   
for $l \in \mathcal { P }$ do   
for all $v _ { n } \in \mathcal V$ and attention heads h do   
Compute $^ { a _ { l , h , n } }$ using text-to-visual attention   
Normalize: $r _ { l , h , n } \gets \mathrm { S o f t m a x } _ { n } ( a _ { l , h , n } )$   
Map to support: $\begin{array} { r } { p _ { l , h , n } \gets \frac { N _ { l } r _ { l , h , n } } { 1 + N _ { l } r _ { l , h , n } } } \end{array}$   
end for   
for all $v _ { n } \in \mathcal V$ do   
Compute reliability $\rho _ { l , n }$ from head clarity and con  
sistency   
Construct   
$\begin{array} { r } { m _ { l , n } ( I ) = \rho _ { l , n } \bar { p } _ { l , n } , \quad m _ { l , n } ( U ) = \rho _ { l , n } ( 1 - \bar { p } _ { l , n } ) , \quad m _ { l , n } ( \Theta ) = 1 - \rho _ { l , n } } \end{array}$   
Recursively fuse $^ { m _ { l , n } }$ with previous-layer evi  
dence using Dempster’s rule to obtain $\widetilde { m } _ { n } ^ { ( l ) }$   
Compute   
$q _ { l , n } \gets 1 - \bigl ( 1 - \mathrm { P l } _ { n } ^ { ( l ) } ( I ) \bigr ) \bigl ( 1 - K _ { n } ^ { ( l ) } \bigr )$   
end for   
Select the highest-q representative in each occupied   
G × G spatial cell   
$\widetilde { q } _ { l , n } \gets ( 1 - \lambda ) q _ { l , n } + \lambda b _ { l , n }$   
$\mathcal { V }  \mathrm { T o p K } ( \{ \widetilde { q } _ { l , n } \} , K _ { l } ^ { \mathrm { k e e p } } )$   
end for   
return V

Consider the h-th attention head in the l-th pruning layer, where l indexes only layers at which pruning is performed. Equation (1) gives the attention score of the n-th visual token $v _ { n }$ in this head. We normalize the scores as

$$
r _ { l , h , n } = \frac { \exp ( a _ { l , h , n } ) } { \sum _ { j } \exp ( a _ { l , h , j } ) }\tag{8}
$$

We then define the support score

$$
p _ { l , h , n } = \frac { N _ { l } \cdot r _ { l , h , n } } { 1 + N _ { l } \cdot r _ { l , h , n } }\tag{9}
$$

This mapping converts relative attention into a support score in [0, 1). Because the mean normalized attention over the $N _ { l }$ tokens in pruning layer l is $1 / N _ { l } , r _ { l , h , n } = 1 / N _ { l }$ gives $p _ { l , h , n } = 0 . 5$ . This value corresponds to average attention and indicates no clear selection preference.

As illustrated in Fig. 3, suppose that the current layer has four attention heads. For a given token, their support scores may be [0.87, 0.91, 0.85, 0.89], indicating consistent support and reliable evidence. Alternatively, the scores may be [0.91, 0.83, 0.08, 0.12], indicating severe inter-head conflict. Direct averaging would obscure this distinction. We therefore compute the evidence reliability $\rho _ { l , n }$ as

![](images/e51e16ee0d507d6218624bfdcbd31424cc86bcf8a098c4b23d2b322edd3bddb0.jpg)  
Figure 3. Reliability-aware fusion of multi-head evidence. Clear and consistent head responses receive high reliability, whereas ambiguous or conflicting responses contribute greater uncertainty.

$$
\rho _ { l , n } = \frac { 2 } { H } \bigg \vert \sum _ { h = 1 } ^ { H } ( p _ { l , h , n } - 0 . 5 ) \bigg \vert \cdot \bigg ( 1 - \frac { 2 } { H } \sum _ { h = 1 } ^ { H } | p _ { l , h , n } - \bar { p } _ { l , n } | \bigg )\tag{10}
$$

Evidence reliability is defined as the product of decision clarity and support consistency. The first factor measures the selection clarity of individual heads, whereas the second characterizes inter-head agreement through deviations from the mean score.

D–S theory represents three states: (i) important, denoted by I, when the evidence supports retaining $v _ { n } ;$ (ii) unimportant, denoted by U, when the evidence does not support retaining $v _ { n } ;$ and (iii) uncertain, denoted by Θ, when the evidence is insufficient or conflicting. The frame is therefore $\Theta = \{ I , U \}$ . The mass function of the n-th candidate token in pruning layer l is

$$
\begin{array} { r l } & { m _ { l , n } ( I ) = \rho _ { l , n } \bar { p } _ { l , n } } \\ & { m _ { l , n } ( U ) = \rho _ { l , n } ( 1 - \bar { p } _ { l , n } ) } \\ & { m _ { l , n } ( \Theta ) = 1 - \rho _ { l , n } } \end{array}\tag{11}
$$

The model now represents each token not by a single attention magnitude but by mass assigned to the important, unimportant, and uncertain states. Disagreement among attention heads is thus quantified as uncertainty within the frame of discernment. This representation avoids premature decisions that could remove critical information and provides a discriminative basis for subsequent multi-source fusion.

## 4.2. Layer-wise Evidence Fusion

To balance inference efficiency and evidential completeness, multi-layer fusion is performed only at the designated pruning layers. This design exploits edge and texture cues from shallow layers, object structure from middle layers, and global semantics from deep layers. Evidence is recursively fused in pruning-layer order. Let $\widetilde { m } _ { n } ^ { ( l ) }$ denote the cumulative mass function of token n after fusion through pruning layer l. It is initialized as

$$
\widetilde { m } _ { n } ^ { ( 1 ) } ( A ) = m _ { 1 , n } ( A ) , \qquad A \in \{ I , U , \Theta \}\tag{12}
$$

At pruning layer $l ,$ we first compute the conflict between ${ m } _ { l , n }$ and the accumulated historical mass $\widetilde { m } _ { n } ^ { ( l - 1 ) }$ :

$$
K _ { n } ^ { ( l ) } = \widetilde { m } _ { n } ^ { ( l - 1 ) } ( I ) m _ { l , n } ( U ) + \widetilde { m } _ { n } ^ { ( l - 1 ) } ( U ) m _ { l , n } ( I )\tag{13}
$$

The fused mass assigned to the important hypothesis I is

$$
\begin{array} { r } { \widetilde { m } _ { n } ^ { ( l ) } ( I ) = \displaystyle \frac { 1 } { 1 - K _ { n } ^ { ( l ) } } \Big [ \widetilde { m } _ { n } ^ { ( l - 1 ) } ( I ) m _ { l , n } ( I ) + \widetilde { m } _ { n } ^ { ( l - 1 ) } ( I ) m _ { l , n } ( \Theta ) } \\ { + \widetilde { m } _ { n } ^ { ( l - 1 ) } ( \Theta ) m _ { l , n } ( I ) \Big ] } \end{array}\tag{14}
$$

The fused mass assigned to the unimportant hypothesis U is

$$
\begin{array} { r } { \widetilde { m } _ { n } ^ { ( l ) } ( U ) = \displaystyle \frac { 1 } { 1 - K _ { n } ^ { ( l ) } } \Big [ \widetilde { m } _ { n } ^ { ( l - 1 ) } ( U ) m _ { l , n } ( U ) + \widetilde { m } _ { n } ^ { ( l - 1 ) } ( U ) m _ { l , n } ( \Theta ) } \\ { + \widetilde { m } _ { n } ^ { ( l - 1 ) } ( \Theta ) m _ { l , n } ( U ) \Big ] } \end{array}\tag{15}
$$

The fused mass assigned to uncertainty Θ is

$$
\widetilde { m } _ { n } ^ { ( l ) } ( \Theta ) = \frac { \widetilde { m } _ { n } ^ { ( l - 1 ) } ( \Theta ) m _ { l , n } ( \Theta ) } { 1 - K _ { n } ^ { ( l ) } }\tag{16}
$$

![](images/bbd0ff8470aeb7fa2cb82c09e20bc118c5f8608f283e68627eb8e985337b5103.jpg)  
Figure 4. Spatial novelty regularization. The visual-token grid is partitioned into spatial cells, and the highest-priority candidate in each occupied cell receives a soft bonus before global Top-K selection.

This yields the fused mass function $\widetilde { m } _ { n } ^ { ( l ) }$ for token n at pruning layer l. Its belief and plausibility for importance hypothesis I are, respectively,

$$
\mathrm { B e l } _ { n } ^ { ( l ) } ( I ) = \widetilde { m } _ { n } ^ { ( l ) } ( I )\tag{17}
$$

$$
\mathrm { P l } _ { n } ^ { ( l ) } ( I ) = \widetilde { m } _ { n } ^ { ( l ) } ( I ) + \widetilde { m } _ { n } ^ { ( l ) } ( \Theta ) = 1 - \widetilde { m } _ { n } ^ { ( l ) } ( U )\tag{18}
$$

To avoid incorrectly pruning tokens with insufficient evidence, we use plausibility rather than belief as the pruning criterion. The retention priority at pruning layer l is defined as

$$
q _ { l , n } = 1 - \bigl ( 1 - P l _ { n } ^ { ( l ) } ( I ) \bigr ) ( 1 - K _ { n } ^ { ( l ) } )\tag{19}
$$

where $q _ { n } \in [ 0 , 1 ]$ is the token retention-priority score. Tokens with greater plausibility or stronger conflict receive higher scores.

## 4.3. Spatial Novelty Regularization and Token Pruning

Although the evidence-based retention priority reflects semantic relevance between visual tokens and the text query, direct global Top-K selection may concentrate the retained tokens in a small image region and discard useful information elsewhere. We therefore introduce the lightweight spatial novelty constraint shown in Fig. 4. It improves the spatial coverage of retained visual tokens without altering the evidence-fusion result.

For LLaVA-1.5-7B, the initial 576 visual tokens form a $2 4 \times 2 4$ grid. We partition this grid into $G \times G$ spatial cells. $\operatorname { W i t h } G = 4$ , each cell contains $6 \times 6$ visual tokens. Let V<sub>l</sub> denote the candidate set at the current layer and $\gamma _ { l , g }$ the candidates in spatial cell g. The representative token of that cell is

$$
r _ { l , g } = \arg \operatorname* { m a x } _ { v _ { n } \in \mathcal { V } _ { l , g } } q _ { l , n }\tag{20}
$$

This operation identifies the highest-priority token in each spatial cell. We further define the spatial representative indicator

$$
b _ { l , n } = \left\{ \begin{array} { l l } { 1 , } & { v _ { n } = r _ { l , g } \mathrm { ~ f o r ~ s o m e ~ o c c u p i e d ~ c e l l ~ } g , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{21}
$$

The spatially adjusted score is

$$
\widetilde { q } \boldsymbol { l } , n = ( 1 - \lambda ) q _ { l , n } + \lambda b _ { l , n }\tag{22}
$$

where λ is the spatial novelty weight, set to 0.1 in all experiments. Global Top-K selection is then performed according to $\widetilde { q } _ { n }$ :

$$
\mathcal { V } _ { \mathrm { k e e p } } = \mathrm { T o p K } \left( \{ \widetilde { q } _ { l , n } ~ | ~ v _ { n } \in \mathcal { V } \} , K \right)\tag{23}
$$

where $\mathcal { V } _ { \mathrm { k e e p } }$ is the set of visual tokens retained after pruning the current layer.

This mechanism applies only a soft bonus to the representative token with the highest evidence score in each occupied cell; it does not force every cell to retain a token. Consequently, it preserves the global evidence-based ranking while reducing excessive concentration of visual tokens in local regions.

## 5. Experiments and Analysis

All experiments were conducted on an NVIDIA GeForce RTX 4080 Super GPU with 32 GB of memory. This section presents the experimental protocol and analyzes the results of E2S-Pruner.

## 5.1. Experiment Settings

## 5.1.1. Datasets

We use seven benchmarks to evaluate pruning performance from complementary perspectives. The main experiments adopt LLaVA-1.5-7B[12] as the dense baseline and evaluate on GQA, SQA, TextVQA, POPE, MME, and MMBench. Cross-model experiments use Qwen2-VL-7B[23] and evaluate on SQA, POPE, MME, and AI2D. For fair and reproducible comparison, all datasets use their official prompt templates and evaluation code.

• GQA[7] is a comprehensive visual question answering benchmark built from real images and structured scene representations. It emphasizes reasoning over object attributes, spatial relations, and compositional semantics rather than simple object recognition.

• ScienceQA (SQA)[14] contains multiple-choice science questions from elementary- and secondary-school curricula. Some questions include images, charts, or diagrams, enabling evaluation of textual understanding, visual comprehension, and scientific reasoning.

• TextVQA[20] evaluates the ability to read and reason about text in images. Answering its questions often requires recognizing scene text, menus, signs, or document content and integrating it with visual context.

• POPE[11], or Polling-based Object Probing Evaluation, measures object hallucination in VLMs by asking whether specific objects are present and testing whether the model incorrectly reports nonexistent objects.

• MME[6] is a comprehensive multimodal benchmark spanning visual perception and cognitive reasoning. Its tasks assess object existence, counting, position, optical character recognition, commonsense knowledge, and reasoning.

• MMBench[13] is a multiple-choice benchmark for multimodal large models that covers fine-grained abilities such as attribute recognition, spatial relations, logical reasoning, and chart understanding.

• AI2D[9] evaluates the understanding of scientific diagrams and instructional illustrations. It requires models to parse objects, structures, and relations and to reason jointly over visual and textual information.

## 5.1.2. Pruning Settings

Although pruning methods differ in pruning locations and the number of tokens removed at each operation, they are compared under the same average-token budget. The average number of retained tokens is defined as

$$
\mathrm { A v g T o k e n s } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } T _ { i }\tag{24}
$$

where N is the number of LLM layers and $T _ { i }$ is the number of visual tokens retained in layer i.

Existing pruning schedules can be grouped into three types: (i) reducing the sequence to AvgTokens before it enters the LLM; (ii) performing one pruning operation at an intermediate LLM layer; and (iii) pruning progressively at multiple intermediate layers. We adopt the third strategy and evaluate average-token budgets of 192, 128, and 64.

LLaVA-1.5-7B has 32 LLM layers and receives 576 visual tokens. We perform three pruning operations after layers 3, 17, and 22; consequently, the visual-token count decreases at the inputs to layers 4, 18, and 23. For average-token budgets of 192, 128, and 64, the three operations retain $1 9 4 \to 1 4 8 \to 9 6 , 1 2 2 \to 4 8 \to 4 2 ,$ and $2 0  8  0$ visual tokens, respectively.

Figure 5 visualizes the pruning outcomes for the same examples under different average-token budgets. Green markers indicate the visual tokens retained in the final layer.

## 5.1.3. Evaluation Metrics

Performance on every dataset is measured using its official evaluation protocol. To summarize performance across heterogeneous benchmarks, we compute a normalized performance-retention score relative to the dense baseline. For LLaVA-1.5-7B, this score is

$$
\begin{array} { r } { \mathrm { R e t e n t i o n } = \displaystyle \frac { 1 } { 6 } \bigg ( \frac { S _ { \mathrm { G Q A } } } { 6 1 . 9 } + \frac { S _ { \mathrm { S Q A } } } { 6 9 . 5 } + \frac { S _ { \mathrm { T e x t V Q A } } } { 5 8 . 2 } + \frac { S _ { \mathrm { P O P E } } } { 8 5 . 9 } } \\ { + \frac { S _ { \mathrm { M M E } } } { 1 8 6 2 } + \frac { S _ { \mathrm { M M B e n c h } } } { 6 4 . 6 } \bigg ) \times 1 0 0 \% . } \end{array}\tag{25}
$$

where each numerator is the score of the pruned model on one benchmark and the corresponding denominator is the score of the dense baseline.

## 5.2. Comparative Experiment

Table 1 compares E2S-Pruner with representative visualtoken pruning methods under three average-token budgets. E2S-Pruner achieves the highest aggregate performance at every budget. With average budgets of 192, 128, and 64 visual tokens, it retains 98.0%, 96.8%, and 90.6% of the dense LLaVA-1.5-7B performance, respectively, demonstrating that it preserves multimodal understanding despite substantial token reduction.

At an average budget of 192 tokens, E2S-Pruner retains 98.0% of aggregate performance, exceeding SparseVLM and V2Drop by 2.1 and 0.4 percentage points, respectively. It obtains the best results on SQA, TextVQA, and MMBench, outperforming V2Drop by 0.5, 1.6, and 0.7 points on these benchmarks. Although V2Drop is slightly better on POPE and MME, the higher aggregate retention of E2S-Pruner indicates a more balanced preservation of capabilities across multimodal tasks.

The advantage becomes more pronounced at an average budget of 128 tokens. E2S-Pruner achieves the best result among the compared methods on all six benchmarks and retains 96.8% of aggregate performance. Relative to V2Drop, it improves GQA, SQA, TextVQA, POPE, MME, and MM-Bench by 1.5, 1.1, 2.2, 2.2, 67, and 1.7 points, respectively, and improves aggregate retention by 2.8 percentage points. It also exceeds SparseVLM by 3.6 percentage points in aggregate retention. These results suggest that E2S-Pruner identifies and preserves task-relevant visual evidence more accurately than methods driven primarily by local attention magnitude or token redundancy.

Under the more challenging 64-token budget, E2S-Pruner still retains 90.6% of dense-model performance, exceeding V2Drop and SparseVLM by 3.7 and 4.1 percentage points, respectively. Compared with V2Drop, the gains on GQA, SQA, POPE, MME, and MMBench are 5.0, 0.5, 3.4, 186, and 2.1 points. The particularly large gains on GQA and MME show that E2S-Pruner preserves visual information essential to image understanding and multimodal reasoning even after most visual tokens are removed.

One limitation emerges on TextVQA under the 64-token setting: E2S-Pruner scores 49.5, below the 54.0 achieved by LLaVA-PruMerge. At extremely low budgets, aggressive pruning may remove image regions containing fine-grained text and therefore disproportionately affect scene-text recognition and reasoning. Protecting textual regions under severe compression is an important direction for future work.

Across the three budgets, E2S-Pruner degrades more slowly than most competing methods as the pruning rate increases. Aggregate retention decreases from 98.0% at 192 tokens to 96.8% at 128 tokens and remains 90.6% at 64 tokens. This trend indicates that the proposed components jointly improve the accuracy and stability of visual-token importance estimation, making E2S-Pruner particularly suitable for resource-constrained applications or scenarios requiring aggressive compression.

![](images/0b69d5048d08a4805990dfa5e9a78cb0bef482894a2a49275646d673df6120cc.jpg)  
Figure 5. Qualitative visualization of final-layer visual-token retention under average budgets of 192, 128, and 64 tokens across representative TextVQA, ScienceQA, and MMBench examples. Green markers denote retained tokens, whereas red points indicate pruned patch locations.

## 5.3. Ablation Experiment

We evaluate each component of E2S-Pruner through ablation experiments at an average budget of 128 visual tokens. As shown in Table 2, all variants use identical models, datasets, prompts, evaluation protocols, and pruning locations; only the target component is removed or replaced. The complete method retains 97.7% of aggregate performance while processing only approximately 22.2% of the original visualtoken budget.

• Complete evidence reasoning. Replacing E2S-Pruner with conventional attention-based Top-K selection (A1) reduces performance retention from 97.7% to 92.9%. The complete method improves SQA, TextVQA, MME, and

MMBench by 5.0, 0.2, 87, and 4.3 points, respectively. Text-to-visual attention alone therefore does not identify critical evidence consistently, whereas explicit evidence modeling substantially improves token selection.

• Multi-head reliability modeling. Removing headreliability estimation (A2) reduces retention to 95.8%, 1.9 percentage points below the complete method; SQA, MME, and MMBench decrease by 1.9, 10, and 2.4 points. Evidence quality clearly differs across heads. Computing $\rho _ { l , n }$ from evidence clarity and consistency reduces interference from ambiguous or contradictory responses.

• Cross-layer evidence fusion. Using only the current evidence layer at each pruning location (A3) reduces retention to 96.2%. The complete method improves SQA, MME, and MMBench by 1.0, 16, and 2.1 points. Visual representations from different layers are complementary: combining shallow details, intermediate structures, and deep semantics reduces incidental single-layer bias and stabilizes token-importance estimates.

• Conflict protection. Removing conflict protection (A4) reduces retention to 96.8%. The complete method improves SQA and MMBench by 1.0 and 1.2 points, indicating that tokens associated with inter-layer disagreement may still contain important information. Incorporating conflict K through $q = 1 - ( 1 - P l ) ( 1 - K )$ reduces the risk of prematurely deleting such tokens. Although A4 attains a slightly higher MME score, its lower aggregate performance shows that conflict protection primarily improves cross-task stability.

Table 1. Performance comparison of visual-token pruning methods on LLaVA-1.5-7B, with average scores normalized to the dense baseline.
<table><tr><td>Method</td><td>GQA SQA</td><td>TextVQA</td><td>POPE</td><td>MME</td><td>MMBench</td><td>Avg</td></tr><tr><td colspan="7">Vanilla 576 Tokens</td></tr><tr><td>LLaVA-1.5-7B 61.9</td><td>69.5</td><td>58.2</td><td>85.9</td><td>1862</td><td>64.6</td><td>100.0%</td></tr><tr><td colspan="7">Retain 192 Tokens (↓ 66.7%)</td></tr><tr><td>ToME (ICLR&#x27;23) 54.3</td><td>65.2</td><td>52.1</td><td>72.4</td><td>1563</td><td>60.5</td><td>88.8%</td></tr><tr><td>FastV (ECCV’24)</td><td>52.7 67.3</td><td></td><td>52.5</td><td>64.8</td><td>1612 61.2</td><td>88.2%</td></tr><tr><td>HiRED (AAAI&#x27;25)</td><td>58.7 68.4</td><td>47.4</td><td></td><td>82.8 1737</td><td>62.8</td><td>93.6%</td></tr><tr><td>LLaVA-PruMerge (ICCV’25)</td><td>54.3 67.9</td><td></td><td>54.3</td><td>71.3</td><td>1632 59.6</td><td>90.3%</td></tr><tr><td>SparseVLM (ICML&#x27;25)</td><td>57.6 69.1</td><td>56.1</td><td></td><td>83.6 1721</td><td>62.5</td><td>95.9%</td></tr><tr><td>CaVIN (2026)</td><td>58.7 69.0</td><td>55.8</td><td></td><td>84.4 1788</td><td>63.3</td><td>97.0%</td></tr><tr><td>V2Drop (CVPR’26)</td><td>58.5 69.3</td><td></td><td>55.6</td><td>85.1 1826</td><td>63.7</td><td>97.6%</td></tr><tr><td>E2S-Pruner (Ours)</td><td>58.6 69.8</td><td></td><td>57.2</td><td>83.6 1814</td><td>64.4</td><td>98.0%</td></tr><tr><td colspan="7">Retain 128 Tokens (↓ 77.8%)</td></tr><tr><td>ToME (ICLR&#x27;23)</td><td>52.4 59.6</td><td>49.1</td><td>62.8</td><td>1343</td><td>53.3</td><td>80.4%</td></tr><tr><td>FastV (ECCV&#x27;24)</td><td>49.6 60.2</td><td></td><td>50.6</td><td>59.6 1490</td><td>56.1</td><td>81.7%</td></tr><tr><td>HiRED (AAAI&#x27;25)</td><td>57.2 68.1</td><td></td><td>46.1</td><td>79.8 1710</td><td>61.5</td><td>91.6%</td></tr><tr><td>LLaVA-PruMerge (ICCV’25)</td><td>53.3 67.1</td><td></td><td>54.3</td><td>67.2 1554</td><td>58.1</td><td>87.9%</td></tr><tr><td>SparseVLM (ICML&#x27;25)</td><td>56.0 67.1</td><td>54.9</td><td></td><td>80.5 1696</td><td>60.0</td><td>93.2%</td></tr><tr><td>CaVIN (2026)</td><td>56.6 68.6</td><td></td><td>53.4</td><td>81.0 1702</td><td>61.2</td><td>93.7%</td></tr><tr><td>V2Drop (CVPR’26)</td><td>56.3 68.8</td><td>53.8</td><td></td><td>80.9 1712</td><td>61.8</td><td>94.0%</td></tr><tr><td>E2S-Pruner (Ours)</td><td>57.8 69.9</td><td>56.0</td><td>83.1</td><td>1779</td><td>63.5</td><td>96.8%</td></tr><tr><td colspan="7">Retain 64 Tokens (↓ 88.9%)</td></tr><tr><td>ToME (ICLR’23)</td><td>48.6 50.0</td><td>45.3</td><td>52.5</td><td>1138</td><td>53.7</td><td>72.3%</td></tr><tr><td>FastV (ECCV’24)</td><td>46.1</td><td>51.1</td><td>47.8</td><td>48.0</td><td>1256 48.0</td><td>71.3%</td></tr><tr><td>LLaVA-PruMerge (ICCV&#x27;25)</td><td>51.9</td><td>68.1</td><td>54.0</td><td>65.3 1549</td><td>55.2</td><td>86.5%</td></tr><tr><td>SparseVLM (ICML’25)</td><td>52.7</td><td>62.2</td><td>51.8</td><td>75.1</td><td>1505 56.2</td><td>86.5%</td></tr><tr><td>CaVIN (2026)</td><td>51.2</td><td>68.5</td><td>51.2</td><td>75.4</td><td>1450 55.0</td><td>86.7%</td></tr><tr><td>V2Drop (CVPR’26)</td><td>50.5</td><td>68.9</td><td>51.8</td><td>75.1</td><td>1470</td><td>55.2 86.9%</td></tr><tr><td>E2S-Pruner (Ours)</td><td>55.5</td><td>69.4</td><td>49.5</td><td>78.5</td><td>1656</td><td>57.3 90.6%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

• Spatial novelty constraint. Removing spatial novelty (A5) reduces retention to 96.3%, with MME and MM-Bench decreasing by 5 and 2.3 points. Global Top-K selection tends to concentrate tokens in a few high-response regions. Spatial novelty improves regional coverage and reduces the loss of information from other important areas.

## 5.4. Analysis of Visual Token Retention Strategies

To study how token allocation across layers affects performance, we evaluate five pruning schedules under the same average budget of 128 visual tokens. The schedules differ only in the token counts assigned to the three pruning stages; the model, datasets, pruning locations, and evaluation protocols remain unchanged.

![](images/fbb1a4f4a3150ddd7d2ba61d9301d53fa0617ec60c0c23281f6eb29ccdc216cc.jpg)  
Figure 6. Qualitative comparison of visual-token retention patterns and model outputs for representative test examples, illustrating how different pruning methods preserve task-relevant image regions.

Table 2. Ablation study of the proposed E2S components.
<table><tr><td>Variant</td><td>SQA</td><td>TextVQA</td><td>MME</td><td>MMBench</td><td>Avg.</td></tr><tr><td>LLaVA-1.5-7B</td><td>69.5</td><td>58.2</td><td>1862</td><td>64.6</td><td>100.0%</td></tr><tr><td>A1: Vanilla Attention Top-K</td><td>64.9</td><td>55.8</td><td>1692</td><td>59.2</td><td>92.9%</td></tr><tr><td>A2: w/o Head Reliability</td><td>68.0</td><td>55.8</td><td>1769</td><td>61.1</td><td>95.8%</td></tr><tr><td>A3: w/o Layer-wise Fusion</td><td>68.9</td><td>55.8</td><td>1763</td><td>61.4</td><td>96.2%</td></tr><tr><td>A4: w/o Conflict Protection</td><td>68.9</td><td>55.9</td><td>1781</td><td>62.3</td><td>96.8%</td></tr><tr><td>A5: w/o Spatial Novelty</td><td>68.9</td><td>55.9</td><td>1774</td><td>61.2</td><td>96.3%</td></tr><tr><td>Full E2S</td><td>69.9</td><td>56.0</td><td>1779</td><td>63.5</td><td>97.7%</td></tr></table>

Table 3. Ablation study of different visual-token retention schedules.
<table><tr><td>Schedule</td><td>SQA</td><td>TextVQA</td><td>MME</td><td>MMBench</td><td>Avg</td><td>Throughput</td></tr><tr><td>576 → 122 → 48 → 42</td><td>69.9</td><td>56.0</td><td>1779</td><td>63.5</td><td>97.7%</td><td>10.37</td></tr><tr><td>576 → 132 → 40 → 32</td><td>69.9</td><td>55.9</td><td>1783</td><td>63.8</td><td>97.8%</td><td>10.57</td></tr><tr><td>576 → 142 → 32 → 22</td><td>70.1</td><td>55.6</td><td>1784</td><td>63.7</td><td>97.7%</td><td>10.70</td></tr><tr><td>576 → 147 → 38 → 12</td><td>69.9</td><td>55.6</td><td>1782</td><td>63.9</td><td>97.7%</td><td>10.41</td></tr><tr><td>576 → 152 → 44 → 2</td><td>69.9</td><td>55.8</td><td>1785</td><td>63.7</td><td>97.7%</td><td>10.71</td></tr></table>

As shown in Table 3, all five schedules retain 97.7%– 97.8% of aggregate performance, indicating that E2S-Pruner is robust to the precise allocation of tokens across layers. The 576 → 132 → 40 → 32 schedule achieves the highest retention of 97.8%, but its advantage is only 0.1 percentage

![](images/d044214493ef1b107fb050536bbda65a37a1cebfc8db31f2d7e8820f95c207ab.jpg)  
Figure 7. Performance–throughput trade-off on LLaVA-1.5-7B under different average visual-token budgets, showing the practical acceleration achieved as the retained sequence becomes shorter.

points.

Different tasks favor different allocations. Retaining more visual tokens in deep layers, as in 576 → 122 → 48 → 42, gives the best TextVQA score of 56.0, suggesting that scenetext recognition benefits from continued visual participation in deep semantic reasoning. By contrast, allocating more tokens to intermediate layers increases MME from 1779 to 1785, indicating that richer intermediate visual information supports more complete semantic representations. SQA and MMBench vary only slightly and are comparatively insensitive to the allocation schedule.

Considering aggregate performance, TextVQA accuracy, and inference efficiency, we select 576 → 122 → 48 → 42 as the default schedule. Although its aggregate retention is 0.1 percentage points below the maximum, it preserves more deep-layer visual information, performs better on textintensive tasks, and maintains strong overall efficiency.

## 5.5. Inference Efficiency Analysis

To evaluate practical acceleration, we compare LLM generation latency, end-to-end latency, peak GPU memory, throughput, and task performance on MMBench. All relative changes use dense LLaVA-1.5-7B as the baseline. Table 4 reports the measurements, and Fig. 7 visualizes the performance–efficiency trade-off.

At an average budget of 128 visual tokens, E2S-Pruner reduces LLM generation latency from 676.11 s to 312.48 s (53.8%) and end-to-end latency from 820.15 s to 422.28 s (48.5%). Throughput increases from 5.28 to 10.37 items/s, corresponding to a 1.96× speedup. Compared with Sparse-VLM, E2S-Pruner reduces end-to-end latency by a further 13.44 s, improves throughput by approximately 3.2%, and raises the task score from 60.0 to 63.5. Compared with V2Drop, it reduces end-to-end latency by 30.3%, improves throughput by 43.4%, and increases the task score by 1.7 points.

When the average budget is reduced to 64 visual tokens, E2S-Pruner decreases LLM generation and end-to-end latency by 57.7% and 51.5%, respectively. Throughput reaches 11.02 items/s, or 2.09× that of the dense model, while retaining 88.7% of its MMBench performance. Relative to SparseVLM, E2S-Pruner reduces end-to-end latency by 14.97 s, improves throughput by approximately 3.8%, and increases the task score by 1.1 points. Relative to V2Drop, the corresponding improvements are 25.8%, 34.9%, and 2.1 points.

As the budget decreases, E2S-Pruner consistently reduces generation latency, and throughput increases from 10.37 items/s at 128 tokens to 11.02 items/s at 64 tokens. The reduction in visual-token count therefore translates into practical acceleration rather than being offset by the additional evidence computation.

E2S-Pruner reaches peak GPU memory usages of 18,847 and 18,807 MB under the two budgets, approximately 21% above the dense baseline. This overhead arises mainly from intermediate states used for evidence modeling and crosslayer fusion. Although its peak memory is lower than that of SparseVLM, it remains higher than those of FastV and V2Drop. Reducing evidence caches and intermediate tensors is therefore an important optimization target. Overall, E2S-Pruner achieves the lowest end-to-end latency and highest throughput while preserving strong task performance, demonstrating its utility for efficient multimodal inference.

## 5.6. Cross-Model Generalization Analysis

To assess generalization across multimodal large language models, we apply E2S-Pruner to Qwen2-VL-7B and compare it with existing methods at two visual-token reduction ratios. As shown in Table 5, E2S-Pruner retains 98.9% and 95.4% of dense-model aggregate performance at reduction ratios of 66.7% and 77.8%, respectively, achieving the highest average retention in both settings.

At a 66.7% reduction ratio, E2S-Pruner retains 98.9% of aggregate performance, exceeding FastV, DART, and V2Drop by 4.8, 2.1, and 1.3 percentage points. It achieves the best SQA and AI2D scores, 83.8 and 80.2, respectively. Although its POPE and MME results are slightly below the best individual scores, its superior aggregate result demonstrates more balanced preservation of visual question answering, hallucination assessment, and general perception capabilities.

At a 77.8% reduction ratio, E2S-Pruner retains 95.4% of aggregate performance, outperforming FastV, DART, and V2Drop by 4.5, 1.5, and 0.5 percentage points. Its SQA score reaches 82.9, 3.3 points above the second-best method, indicating that critical information for complex visual reasoning is preserved even at a low token budget. Although it does not achieve the best individual result on POPE, MME, or AI2D, its aggregate performance remains superior to all competing methods.

These results show that E2S-Pruner is not tied to a particular architecture or model family. Its evidence modeling and token-selection mechanisms transfer effectively to Qwen2- VL-7B and preserve stable aggregate performance across reduction ratios, confirming strong cross-model generalization.

Table 4. Efficiency and performance comparison on LLaVA-1.5-7B, where percentages denote changes or performance retention relative to the dense baseline.
<table><tr><td>Method</td><td>LLM Generation↓ Latency (s)</td><td>Total Latency↓ (s)</td><td>GPU Peak↓ Memory (MB)</td><td>Throughput↑ (item/s)</td><td>Accuracy↑</td></tr><tr><td>LLaVA-1.5-7B</td><td>676.11</td><td>820.15</td><td>15566</td><td>5.28</td><td>64.6</td></tr><tr><td colspan="6">Avg. Retention 128 Tokens (↓ 77.8%)</td></tr><tr><td>FastV (ECCV’24)</td><td>334.87 ( 50.5%)</td><td>450.09(↓ 45.1%)</td><td> $1 5 5 5 8 ( \downarrow 0 . 1 \% )$ </td><td> $9 . 7 3 \ ( \uparrow 1 . 8 4 \times )$ </td><td>56.1 (86.8%)</td></tr><tr><td>SparseVLM (ICML’25)</td><td>322.43 (↓ 52.3%)</td><td>435.72 (↓ 46.9%)</td><td> $1 9 2 2 9 ( \uparrow 2 3 . 5 \% )$ </td><td> $1 0 . 0 5 \ : ( \uparrow 1 . 9 0 \times )$ </td><td>60.0 (92.9%)</td></tr><tr><td>V2Drop (CVPR’26)</td><td>410.02 (↓ 39.4%)</td><td> $6 0 5 . 6 9 \ ( \downarrow \ 2 6 . 1 \% )$ </td><td> $\mathbf { 1 5 0 4 6 } \ ( \downarrow \ 3 . 3 \% )$ </td><td> $7 . 2 3 \ : ( \uparrow 1 . 3 7 \times )$ </td><td>61.8 (95.7%)</td></tr><tr><td>E2S-Pruner (Ours)</td><td>312.48 (↓ 53.8%)</td><td>422.28 (↓ 48.5%)</td><td> $1 8 8 4 7 ( \uparrow 2 1 . 1 \% )$ </td><td> $\mathbf { 1 0 . 3 7 \ ( \uparrow 1 . 9 6 \times ) }$ </td><td>63.5 (98.3%)</td></tr><tr><td colspan="6">Avg. Retention 64 Tokens (↓ 88.9%)</td></tr><tr><td>FastV (ECCV’24)</td><td>310.86 (↓ 54.0%)</td><td> $4 2 9 . 6 4 \ ( \downarrow \ 4 7 . 6 \% )$ </td><td> $1 5 3 3 0 ( \downarrow 1 . 5 \% )$ </td><td> $1 0 . 1 9 \left( \uparrow 1 . 9 3 \times \right)$ </td><td>48.0 (74.3%)</td></tr><tr><td>SparseVLM (ICML’25)</td><td>305.13 (↓ 54.9%)</td><td> $4 1 2 . 3 4 ( \downarrow 4 9 . 7 \% )$ </td><td> $1 9 1 2 2 ( \uparrow 2 2 . 8 \% )$ </td><td> $1 0 . 6 2 \left( \uparrow 2 . 0 1 \times \right)$ </td><td>56.2 (87.0%)</td></tr><tr><td>V2Drop (CVPR&#x27;26)</td><td>354.73 (↓ 47.5%)</td><td> $5 3 5 . 8 6 ( \downarrow 3 4 . 7 \% )$ </td><td> $\mathbf { 1 4 9 7 8 } \ ( \downarrow \ 3 . 8 \% )$ </td><td> $8 . 1 7 \ ( \uparrow \ 1 . 5 5 \times )$ </td><td>55.2 (85.4%)</td></tr><tr><td>E2S-Pruner (Ours)</td><td>286.11 (↓ 57.7%)</td><td> $3 9 7 . 3 7 \ ( \downarrow \ 5 1 . 5 \% )$ </td><td> $1 8 8 0 7 ( \uparrow 2 0 . 8 \% )$ </td><td> $\mathbf { 1 1 . 0 2 } \left( \uparrow 2 . 0 9 \times \right)$ </td><td>57.3 (88.7%)</td></tr></table>

Table 5. Comparison on Qwen2-VL-7B under different token reduction ratios.
<table><tr><td>Method</td><td>SQA</td><td>POPE</td><td>MME</td><td>AI2D</td><td>Avg.</td></tr><tr><td>Qwen2-VL-7B</td><td>84.7</td><td>86.1</td><td>2317</td><td>80.5</td><td>100.0%</td></tr><tr><td colspan="6">Token Reduction (↓ 66.7%)</td></tr><tr><td>FastV (ECCV’24) DART (EMNNLP&#x27;25)</td><td>80.0 81.4</td><td>82.1 83.9</td><td>2130 2245</td><td>76.1 78.0</td><td>94.1% 96.8%</td></tr><tr><td>V2Drop (CVPR’26)</td><td>81.6</td><td>87.2</td><td>2224</td><td>78.0</td><td>97.6%</td></tr><tr><td>E2S-Pruner (Ours)</td><td>83.8</td><td>86.8</td><td>2228</td><td>80.2</td><td>98.9%</td></tr><tr><td colspan="6">Token Reduction (↓ 77.8%)</td></tr><tr><td>FastV (ECCV’24)</td><td>78.3</td><td>79.2</td><td>2031</td><td>73.8</td><td>90.9%</td></tr><tr><td>DART (EMNNLP&#x27;25)</td><td>79.6</td><td>82.1</td><td>2175</td><td>74.4</td><td>93.9%</td></tr><tr><td>V2Drop (CVPR’26)</td><td>78.9</td><td>85.1</td><td>2173</td><td>75.6</td><td>94.9%</td></tr><tr><td>E2S-Pruner (Ours)</td><td>82.9</td><td>83.9</td><td>2156</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>74.9</td><td>95.4%</td></tr></table>

## 6. Conclusion

We presented E2S-Pruner, a progressive two-stage evidencefusion method that reduces the inference cost caused by excessive visual tokens in VLMs. The method first estimates the reliability of evidence from different attention heads within each selected layer and then recursively fuses information across pruning layers using D–S evidence theory. Conflict protection reduces the risk of removing potentially important tokens associated with strong disagreement, while spatial novelty regularization promotes coverage of distinct image regions and prevents excessive concentration in locally salient areas.

On LLaVA-1.5-7B, E2S-Pruner retains 98.0%, 96.8%, and 90.6% of aggregate performance at average budgets of 192, 128, and 64 visual tokens. Under the 128-token and 64- token settings, it reduces end-to-end latency by 48.5% and 51.5% and reaches 1.96× and 2.09× the baseline throughput. On Qwen2-VL-7B, it retains 98.9% and 95.4% of aggregate performance at two reduction ratios, demonstrating that its evidence modeling and spatial constraint generalize across model families.

E2S-Pruner nevertheless leaves room for improvement. Evidence fusion and spatial novelty computation require additional intermediate states and therefore increase peak GPU memory. Moreover, tasks such as TextVQA, which depend on fine-grained textual regions, remain vulnerable to information loss at extremely low token budgets. Future work will investigate lower-overhead evidence caching and parallel fusion, together with adaptive spatial protection for text and fine-grained objects, to improve deployment efficiency and robustness on resource-constrained devices.

## References

[1] Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S. Nikolopoulos, Hans Vandierendonck, Deepu John, and Bo Ji. HiRED: Attention-guided token dropping for efficient inference of high-resolution vision-language models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1773– 1781, 2025.

[2] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging:

Your ViT but faster. In International Conference on Learning Representations, 2023.

[3] Junjie Chen, Xuyang Liu, Zichen Wen, Yiyu Wang, Siteng Huang, and Honggang Chen. Variation-aware vision token dropping for faster large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3489–3499, 2026.

[4] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35, 2024.

[5] Lin Cheng, Yanjie Liang, Yang Lu, and Yiu-ming Cheung. GradToken: Decoupling tokens with class-aware gradient for visual explanation of transformer network. Neural Networks, 181:106837, 2025.

[6] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. MME: A comprehensive evaluation benchmark for multimodal large language models. In Advances in Neural Information Processing Systems, 2025.

[7] Drew A. Hudson and Christopher D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6700–6709, 2019.

[8] Yanfeng Jiang, Ning Sun, Xueshuo Xie, Fei Yang, and Tao Li. ADFQ-ViT: Activation-distribution-friendly post-training quantization for vision transformers. Neural Networks, 186: 107289, 2025.

[9] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European Conference on Computer Vision, pages 235–251, 2016.

[10] Eric Lefevre, Olivier Colot, and Philippe Vannoorenberghe. Belief function combination and conflict management. Information Fusion, 3(2):149–162, 2002.

[11] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large visionlanguage models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305. Association for Computational Linguistics, 2023.

[12] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.

[13] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. MMBench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233, 2024.

[14] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, pages 2507–2521, 2022.

[15] Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup Mallik, Alessandro G Allievi, Senem Velipasalar, and Liu Ren. VLP: Vision language planning for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14760–14769, 2024.

[16] Taoyu Qian, Qi Wang, Shang Gao, and Hualong Yu. Find what you missed: Causal recovery for visual tokens in visionlanguage models. Knowledge-Based Systems, 350:116589, 2026.

[17] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

[18] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. DynamicViT: Efficient vision transformers with dynamic token sparsification. In Advances in Neural Information Processing Systems, 2021.

[19] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. LLaVA-PruMerge: Adaptive token reduction for efficient large multimodal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22857–22867, 2025.

[20] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8317–8326, 2019.

[21] Keke Su, Lihua Cao, Botong Zhao, Ning Li, Di Wu, Xiyu Han, and Yangfan Liu. DctViT: Discrete cosine transform meet vision transformers. Neural Networks, 172:106139, 2024.

[22] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017.

[23] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution, 2024.

[24] Yunke Wang, Bo Du, Wenyuan Wang, and Chang Xu. Multitailed vision transformer for efficient inference. Neural Networks, 174:106235, 2024.

[25] Yifan Wang, Jian Zhao, Zhaoxin Fan, Xin Zhang, Xuecheng Wu, Yudian Zhang, Lei Jin, Xinyue Li, Gang Wang, Mengxi Jia, Ping Hu, Zheng Zhu, and Xuelong Li. JTD-UAV: MLLMenhanced joint tracking and description framework for anti-UAV systems. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1633– 1644, 2025.

[26] Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. Stop looking for “important tokens” in multimodal language

models: Duplication matters more. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9961–9980. Association for Computational Linguistics, 2025.

[27] Tiantao Xian, Zhiheng Zhou, Wenlve Zhou, and Zhipeng Zhang. Refining visual token sequence for efficient image captioning. Neural Networks, 191:107759, 2025.

[28] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis A. Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. SparseVLM: Visual token sparsification for efficient visionlanguage model inference. In Proceedings of the 42nd International Conference on Machine Learning, pages 74840–74857. PMLR, 2025.