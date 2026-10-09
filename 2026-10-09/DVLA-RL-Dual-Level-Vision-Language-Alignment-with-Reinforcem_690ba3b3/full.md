# DVLA-RL++: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning

Wenhao Li , Xianjing Meng , Qiangchang Wang , Zhongyi Han , Yilong Yin , and Liqiang Nie , Senior Member, IEEE

Abstract—Few-shot learning aims to recognize novel categories from limited labeled examples. Recent studies incorporate textual semantics to compensate for limited visual observations and improve class representations. However, high image-text agreement may reflect both intrinsic object properties and incidental context, making support prototypes susceptible to contextual contamination. To address this problem, we propose DVLA-RL++, which extends DVLA-RL with complementary semantic purification (CSP) and counterfactual reinforcement-learning gating (CRG). Specifically, CSP generates intrinsic and nuisance descriptions from labeled supports and compares their agreement with each support token. An ambiguity-dependent rejection margin guides sparse evidence allocation, while an intrinsic semantic anchor fills the unassigned mass to provide a fallback when visual evidence is unreliable. CRG learns layer-wise semantic fusion strengths using a reward that balances recognition performance and nuisance exposure. An independently executed reference trajectory on the same episode provides a paired learning signal. Theoretical analysis relates retained evidence and anchor quality to prototype stability and establishes conditions for unbiased on-policy gradient estimation. Experiments on standard, fine-grained, and cross-domain benchmarks show state-of-the-art accuracy, with an average gain of 1.4% over DVLA-RL. The project page is available at https://peacelwh.github.io/TPAMI27-DVLA-RLpp/.

Index Terms—Few-shot learning, Vision-language alignment, Support prototype, Reinforcement learning.

## I. INTRODUCTION

Few-shot learning (FSL) aims to recognize novel categories from limited labeled examples by transferring knowledge acquired on base categories [1, 2]. It addresses settings in which collecting representative annotations for every new category is expensive or impractical. In episodic recognition, a learner uses a small labeled support set to classify separate query images. Metric-based methods construct a prototype for each class by aggregating its support features. With only a few observations, recognition depends not only on the quality of the learned embedding but also on which support evidence is included in the prototype.

Existing approaches improve either visual representations or the semantic information available to the learner. Visual methods strengthen local matching, embedding adaptation, and feature transfer [3, 4, 5]. Semantic methods introduce class names, prompts, attributes, or descriptions to supplement limited visual observations [6, 7, 8, 9]. These directions are complementary. Visual features capture the appearance of observed instances, whereas language supplies class knowledge that a few images cannot reliably establish. Our earlier framework DVLA-RL [10] combines local attributes and global descriptions with reinforcement-learning gates for hierarchical vision-language alignment. This alignment enriches visual features at multiple semantic granularities. However, it does not explicitly determine which local support features should define a class. A description may accurately characterize an image while matching both intrinsic object properties and incidental context. Therefore, reliable prototypes require distinguishing class-defining evidence from contextual cues, even when both are well explained by language.

Fig. 1(a) illustrates this ambiguity with a bird among reeds. Striped plumage and vertical reeds share visual patterns, and both intrinsic and contextual descriptions can be factually correct. However, contextual cues become unreliable when the background changes. Averaging support features allows these cues to enter the class prototype, a problem we call support-evidence contamination. Fig. 1(b) shows that competing semantic explanations help distinguish the evidence. Feathers favor the intrinsic descriptions, whereas reeds favor the nuisance descriptions. The reed plume is a subtler case, since its high intrinsic agreement is accompanied by a small advantage over nuisance agreement, motivating an explicit rejection margin. Unlike spatial foreground suppression [11], semantic evidence selection concerns incidental cues that may occur within or alongside the target object. It must also handle supports whose visual tokens are all unreliable without forcing prototype mass onto poor evidence. Furthermore, selective aggregation alone does not control whether semantic fusion reinforces contextual cues across visual layers. The fusion policy should therefore account for support-evidence reliability while preserving useful class semantics.

Motivated by these observations, we propose DVLA-RL++ with complementary semantic purification (CSP) and counterfactual reinforcement-learning gating (CRG). CSP compares each support token against intrinsic and nuisance descriptions generated from labeled supports. Its rejection margin increases with the semantic overlap between the two description banks, demanding stronger relative evidence when the explanations are ambiguous. Sparse allocation retains selected visual evidence, while an intrinsic semantic anchor fills the unassigned prototype mass and provides a defined fallback when no token is retained. CRG learns layer-wise semantic fusion strengths using a nuisance-aware recognition reward. A sampled gate policy and a frozen reference gate run on the same episode with the same representation but propagate their own support states. Their paired utilities guide policy learning toward gate choices that improve recognition while limiting nuisance exposure. At inference, the semantic banks, gate schedule, and prototypes are constructed from labeled supports and reused for independently processed queries.

![](images/ecde2233ed11ea8ca0630a1fd282e0df86caaac8be79aba6645d0dcd84f6e5a6.jpg)

![](images/788111fb31de9e4d60dfce4fcb300f2fc719d357ee33741f9e9ef86375660366.jpg)  
Fig. 1. Illustration of support-evidence contamination. (a) Intrinsic bird cues and incidental reed patterns coexist in a support image. (b) Comparing intrinsic and nuisance agreement keeps the feathers and rejects the reeds. The reed plume has positive relative evidence but falls below the rejection margin of 0.15, illustrating the need for explicit rejection.

This article extends our ICLR 2026 conference paper, DVLA-RL [10], in three respects. (1) We introduce complementary semantic purification to replace uniform averaging of support tokens with selective evidence aggregation and theoretically analyze the stability of the resulting prototypes. (2) We develop counterfactual reinforcement-learning gating using a nuisance-aware reward and a paired learning signal from an independently executed reference policy, and establish conditions for unbiased on-policy gradient estimation. (3) We expand the evaluation with context interventions, evidence audits, and policy diagnostics to examine the role of supportevidence reliability in the observed performance gains. Our main contributions are summarized as follows.

• DVLA-RL++ is proposed to purify support evidence against contextual contamination, combining selective prototype construction with adaptive semantic fusion.

• A CSP module is proposed to separate intrinsic evidence from contextual cues with competing descriptions and ambiguity-dependent rejection, backed by an intrinsic anchor and prototype stability bounds.

• A CRG module is proposed to limit nuisance exposure during semantic fusion with an independent reference trajectory, yielding unbiased on-policy gradient estimation.

• The proposed method achieves state-of-the-art results on nine FSL benchmarks across diverse scenarios with an average accuracy gain of 1.4% over DVLA-RL.

## II. RELATED WORK

## A. Visual and Semantic Few-Shot Learning

Visual few-shot learning transfers representations learned from base categories to novel classes with limited labeled examples. Matching Networks [1] and Prototypical Networks [2] establish embedding-based recognition through support-query matching and class prototypes, respectively. Subsequent methods improve local correspondence, episodic adaptation, and transferable representations. DeepEMD [3] models local feature correspondence through differentiable matching, FEAT [4] adapts embeddings using relationships within an episode, and Tian et al. [5] demonstrate the effectiveness of strong pretrained embeddings. These approaches improve visual recognition, while prototype estimation remains dependent on the limited evidence available in the support set.

Semantic methods supplement visual observations with class knowledge conveyed by names, prompts, attributes, and descriptions. AM3 [6] adaptively combines visual and semantic representations, Semantic Prompt [7] introduces textual knowledge for image recognition, and KTPP [12] incorporates hierarchical cross-modal prompts. SemFew [8] and ECER [9] further enrich the semantic information available for novel classes. Vision-language pretraining [13], prompt learning [14], and support-cache adaptation [15] provide additional mechanisms for transferring textual knowledge to fewshot recognition. Our earlier DVLA-RL [10] integrates local attributes and global descriptions through adaptive layer-wise fusion. Building on this framework, the present work examines the reliability of the support evidence used to construct prototypes. High image-text agreement may reflect both intrinsic object properties and incidental context, motivating explicit evidence selection beyond semantic enrichment.

## B. Complementary Evidence and Selective Aggregation

Context suppression offers a related perspective on improving support representations. Zha et al. [11] combine background suppression and foreground alignment for few-shot fine-grained recognition. Such spatial separation is relevant to contextual interference, although incidental cues may also occur within the foreground. Complementary information can additionally be expressed through negative semantic signals. For example, TDA [16] incorporates negative pseudo-labels into dynamic test-time caches for vision-language models. These signals concern class assignments, whereas our nuisance descriptions characterize incidental visual cues within labeled support images. Intrinsic and nuisance evidence can therefore coexist in the same image without changing its class label.

![](images/482f18e50b69339ff9919d4d1322b6fbfcf73a769f9705a07d81689371db63c8.jpg)  
Fig. 2. Overview of the proposed DVLA-RL++ framework. (A) A description generator produces intrinsic and nuisance semantic banks from the labeled supports of each class, and a frozen text encoder embeds them. (B) The intrinsic semantics guide local and global alignment inside the visual backbone through RL-gated attention. (C) Support tokens compete against the two banks, the retained evidence is mixed with an intrinsic anchor, and class-wise aggregation yields the prototypes used for nearest-prototype recognition. (D) During training only, a sampled gate policy and a frozen reference gate follow separate state trajectories on the same episode, and their paired utilities update the gate policy. Query labels enter only the training objectives and the evaluation metrics.

Sparse aggregation provides a mechanism for selecting informative features. Sparsemax [17] produces exact zero weights by projecting scores onto the probability simplex. CSP adopts a related projection principle but allows the total visual allocation mass to remain below one. This enables unreliable tokens to be rejected without redistributing all available mass among the remaining visual candidates. An intrinsic semantic anchor supplies the unassigned mass, including a fallback when no visual token is retained. The resulting formulation connects complementary semantic evidence, explicit rejection, and prototype construction, with theoretical analysis relating retained nuisance mass and anchor quality to prototype.

## C. Adaptive Fusion and Policy Learning

Adaptive fusion determines how semantic information contributes to visual representations. AM3 [6] learns adaptive combinations of visual and semantic representations, while DVLA-RL [10] uses reinforcement-learning gates to regulate semantic fusion across network layers. Policy-gradient methods provide the optimization foundation for such decisions: REINFORCE [18] estimates gradients through the scorefunction identity, and PPO [19] optimizes a clipped surrogate using sampled trajectories. The present extension incorporates support-evidence reliability into policy learning through a reward that accounts for both recognition performance and nuisance exposure.

Reference-based learning signals have also been explored in reinforcement learning. Self-critical sequence training [20] uses the reward of a decoded output as a baseline for sampled outputs, and COMA [21] constructs a counterfactual baseline by marginalizing an agent’s action. CRG applies referencebased learning to nuisance-aware semantic fusion through paired policy and reference rollouts on the same episode. The two rollouts share the representation but propagate their own states, allowing their utilities to reflect the gate consequences. Our analysis specifies the conditions under which the reference utility can serve as an action-independent baseline for unbiased on-policy gradient estimation.

## III. METHOD

DVLA-RL++ combines selective prototype construction with adaptive semantic fusion. As illustrated in Fig. 2, the dual-level alignment inherited from DVLA-RL produces semantically enriched visual features, complementary semantic purification (CSP) selects support evidence for prototype construction, and counterfactual reinforcement-learning gating (CRG) learns fusion strengths using recognition and evidencereliability signals. We first define the episodic setting and the alignment module, then describe CSP and CRG, and finally analyze prototype stability and the inference procedure.

## A. Problem Formulation

An N-way K-shot episode consists of a labeled support set and a separate query set,

$$
\mathcal { S } = \{ ( x _ { c , k } , c ) \} _ { c = 1 , k = 1 } ^ { N , K } , \qquad Q = \{ ( x _ { j } , y _ { j } ) \} _ { j = 1 } ^ { n _ { q } } ,\tag{1}
$$

where N denotes the number of classes, K the number of supports per class, and $n _ { q }$ the number of queries. The base, validation, and test categories are disjoint. Class names are available as semantic supervision, while query labels are used only for training objectives and evaluation. The learner constructs a prototype $p _ { c }$ for each class and classifies each query independently of the other queries. Support-evidence contamination refers to the inclusion of incidental contextual cues in these prototypes, even when support labels are correct.

The visual backbone produces image tokens at L selected layers, and a frozen text encoder embeds the descriptions generated from labeled supports. We retain the dual-level alignment and scalar attention gate of DVLA-RL [10], while introducing selective support aggregation and a nuisanceaware gate-learning objective. CSP determines which support tokens contribute to a prototype, and CRG determines the strength of semantic fusion at each selected layer. Semantic banks, gate decisions, and prototypes are constructed from labeled supports before a test query is processed.

## B. Dual-Level Semantic Alignment

a) Support-conditioned semantics: For each class, the generator receives its labeled supports and constructs local intrinsic and nuisance phrases together with global descriptions for the two roles. Intrinsic descriptions characterize class-defining object properties, whereas nuisance descriptions characterize incidental context. Both use affirmative statements; a nuisance description is neither a negated sentence nor a competing class label. Generation is performed once per episode and cached. When no reliable intrinsic phrase is returned, the class name provides the intrinsic fallback. An empty nuisance bank yields a zero nuisance response and a zero rejection margin.

b) Gated attention: At each selected layer, projected intrinsic semantic tokens from all candidate classes are concatenated according to a fixed local-to-global layer schedule. The same episode-level semantic tokens guide every support and query image, so description selection requires no query class information. Nuisance descriptions are used for evidence scoring and policy-state construction. Let $\begin{array} { r c l } { Z _ { \ell } } & { \in } & { \mathbb { R } ^ { M _ { \ell } \times d _ { \ell } } } \end{array}$ denote the image tokens and $\boldsymbol { T _ { \ell } } ~ \in ~ \mathbb { R } ^ { n _ { \ell } ^ { t } \times d _ { \ell } }$ the projected intrinsic semantic tokens at layer ℓ, where $M _ { \ell }$ and $n _ { \ell } ^ { t }$ are their respective token counts. As shown in Fig. 3, semantic queries produce image-guided and text-guided responses,

$$
\begin{array} { r } { V _ { \ell } = \mathrm { A t t n } ( T _ { \ell } W _ { q } , Z _ { \ell } W _ { k } ^ { v } , Z _ { \ell } W _ { v } ^ { v } ) , } \\ { S _ { \ell } = \mathrm { A t t n } ( T _ { \ell } W _ { q } , T _ { \ell } W _ { k } ^ { t } , T _ { \ell } W _ { v } ^ { t } ) , } \\ { B _ { \ell } ( g _ { \ell } ) = g _ { \ell } V _ { \ell } + ( 1 - g _ { \ell } ) S _ { \ell } , \qquad } \end{array}\tag{2}
$$

where Attn denotes scaled dot-product attention, $W _ { q }$ is the shared query projection, and $\left( W _ { k } ^ { v } , W _ { v } ^ { v } \right)$ and $( W _ { k } ^ { t } , W _ { v } ^ { t } )$ are the key and value projections of the two branches. The gate action $g _ { \ell } \in ( 0 , 1 )$ is shared across images within an episode. Larger gate values emphasize image-conditioned semantic responses, while smaller values emphasize text-conditioned responses. The pooled fused context is broadcast to image tokens, and the fused semantic tokens are appended before the transformer block,

![](images/5998b94f242761451cba2c74c85f103cfc014f95c6345b04aec9eb6073c2c38d.jpg)  
Fig. 3. Illustration of the RL-gated attention block. A support-derived gate combines image-guided and text-guided semantic responses. The fused context enriches image tokens through joint transformer processing, after which onl image tokens are propagated.

$$
Z _ { \ell + 1 } = \Bigl [ F _ { \phi , \ell } \bigl ( [ Z _ { \ell } + { \bf 1 } \mathrm { m e a n } ( B _ { \ell } ) ; B _ { \ell } ] \bigr ) \Bigr ] _ { \mathrm { i m g } } ,\tag{3}
$$

where $F _ { \phi , \ell }$ denotes the transformer block, 1 is a column of ones, the brackets concatenate token rows, and $[ \cdot ] _ { \mathrm { i m g } }$ retains the image positions. Semantic tokens are discarded after each block, preventing their accumulation across layers.

c) Cross-modal projections: Scoring projections $P _ { \ell }$ map image tokens into the text embedding space. They are calibrated on labeled base supports before policy training and subsequently frozen, keeping the scoring geometry fixed during gate optimization. An anchor projection G maps pooled intrinsic semantics into the final visual feature space and is trained jointly with the visual encoder through the classification loss.

d) Classification objective: For a normalized query feature $q _ { j }$ and normalized class prototypes $\{ p _ { c } \} _ { c = 1 } ^ { N } ,$ , classification uses cosine logits with a shared temperature $\tau _ { \mathrm { c l s } } > 0$

$$
\begin{array} { c c c } { { P _ { \phi } ( y _ { j } = c \mid x _ { j } , S ) = \displaystyle \frac { \exp ( q _ { j } ^ { \top } p _ { c } / \tau _ { \mathrm { c l s } } ) } { \sum _ { d = 1 } ^ { N } \exp ( q _ { j } ^ { \top } p _ { d } / \tau _ { \mathrm { c l s } } ) } , } } \\ { { \mathcal { L } _ { \mathrm { c l s } } = \displaystyle - \frac { 1 } { n _ { q } } \sum _ { j } \log P _ { \phi } ( y _ { j } \mid x _ { j } , S ) . } } \end{array}\tag{4}
$$

The representation is optimized through $\mathcal { L } _ { \mathrm { c l s } }$ , while the gate policy also accounts for evidence reliability. The conference model is evaluated separately as a baseline. The reference gate used by CRG operates on the representation and prototype constructor of DVLA-RL++.

![](images/9f0198413a06578ec36cf92b1742c2fc7e7f3bddca2268bb70dc3ef0cf7d952e.jpg)  
Fig. 4. Illustration of complementary semantic purification. Competing intrinsic and nuisance descriptions guide sparse support-token allocation. An intrinsic semantic anchor fills the unallocated mass before class-wise aggregation and normalization.

## C. Complementary Semantic Purification

CSP constructs prototypes through complementary evidence scoring, sparse allocation, semantic anchoring, and support aggregation, as illustrated in Fig. 4. Scoring uses image tokens only; appended semantic tokens are excluded. At the final layer, normalized visual tokens $z _ { c , k , i }$ provide the prototype features, while their normalized projected counterparts $v _ { c , k , i }$ provide the semantic scores.

a) Complementary evidence scoring: Let $e _ { c , r } ^ { + , h }$ and $e _ { c , r } ^ { - , h }$ denote the normalized embeddings of the r-th intrinsic and nuisance descriptions of class c at semantic level $\begin{array} { r l } { h } & { { } \in } \end{array}$ {loc, glob}. Let $R _ { c } ^ { \pm , h }$ denote the corresponding bank sizes. For a nonempty bank, the response is computed through sizenormalized log-sum-exp aggregation,

$$
\begin{array} { r l } & { u _ { c , k , i } ^ { \pm , h } = \tau _ { b } \log \left[ \frac { 1 } { R _ { c } ^ { \pm , h } } \displaystyle \sum _ { r = 1 } ^ { R _ { c } ^ { \pm , h } } \exp ( v _ { c , k , i } ^ { \top } e _ { c , r } ^ { \pm , h } / \tau _ { b } ) \right] , } \\ & { u _ { c , k , i } ^ { \pm } = \displaystyle \sum _ { h } \omega _ { h } u _ { c , k , i } ^ { \pm , h } , \qquad d _ { c , k , i } = u _ { c , k , i } ^ { + } - u _ { c , k , i } ^ { - } , } \end{array}\tag{5}
$$

where $\tau _ { b } ~ > ~ 0$ is the bank temperature and $\omega _ { h } ~ \ge ~ 0$ are fixed weights satisfying $\textstyle \sum _ { h } \omega _ { h } = 1$ . The same level weights are used for both semantic roles. Size normalization removes the direct dependence on bank cardinality introduced by an unnormalized sum, while the fixed level weights prevent phrase counts from determining the relative contribution of local and global semantics. A positive signed score indicates stronger agreement with the intrinsic bank. An entirely empty nuisance bank has $u _ { c , k , i } ^ { - } = 0$

b) Ambiguity-dependent rejection: Let $\mathcal { E } _ { c } ^ { \pm } ~ = ~ \{ e _ { c , r } ^ { \pm , h }$ $h \in \{ \mathrm { l o c } , \mathrm { g l o b } \}$ , $1 ~ \le ~ r ~ \le ~ R _ { c } ^ { \pm , h } \}$ denote the union of the available embeddings for each semantic role. When both banks are nonempty, their maximum cross-bank similarity defines the overlap,

$$
\begin{array} { r l } & { o _ { c } = \frac { 1 + \operatorname* { m a x } _ { e \in \mathcal { E } _ { c } ^ { + } , f \in \mathcal { E } _ { c } ^ { - } } e ^ { \top } f } { 2 } , } \\ & { \xi _ { c } = \left\{ \xi _ { 0 } + \kappa _ { o } o _ { c } , \quad \mathcal { E } _ { c } ^ { - } \neq \emptyset , \right. } \\ & { \left. \mathcal { E } _ { c } ^ { - } = \emptyset , \right. } \end{array}\tag{6}
$$

where $\xi _ { 0 } \ge 0$ and $\kappa _ { o } \geq 0$ are selected on validation episodes. The intrinsic fallback ensures that $\mathcal { E } _ { c } ^ { + }$ is nonempty. Greater

overlap increases the required intrinsic advantage because semantically similar explanations are harder to distinguish. The overlap is evaluated only for nonempty nuisance banks.

c) Sparse evidence allocation: CSP allocates at most unit visual mass to each support image by projecting the marginadjusted scores onto the nonnegative mass ball,

$$
\begin{array} { r l } & { \mathbf { a } _ { c , k } = \underset { a \geq 0 , 1 ^ { \top } a \leq 1 } { \arg \operatorname* { m i n } } \frac { 1 } { 2 } \left\| a - \frac { \mathbf { d } _ { c , k } - \xi _ { c } \mathbf { 1 } } { \tau _ { s } } \right\| _ { 2 } ^ { 2 } , } \\ & { \rho _ { c , k } = \mathbf { 1 } ^ { \top } \mathbf { a } _ { c , k } , } \end{array}\tag{7}
$$

where $\tau _ { s } > 0$ is the allocation temperature and $\rho _ { c , k } \in [ 0 , 1 ]$ is the retained visual mass. The projection is unique. As in simplex-based sparse projection [17], it admits a sorting-based threshold solution,

$$
a _ { c , k , i } = [ ( d _ { c , k , i } - \xi _ { c } ) / \tau _ { s } - \delta _ { c , k } ] _ { + } , \quad \delta _ { c , k } \geq 0 ,\tag{8}
$$

where $[ t ] _ { + } ~ = ~ \operatorname* { m a x } ( t , 0 )$ . The threshold is zero when the positive margin-adjusted scores already sum to at most one; otherwise, it enforces unit total allocation. Consequently, $a _ { c , k , i } > 0$ requires $d _ { c , k , i } > \xi _ { c } ,$ although this condition alone does not guarantee retention. The unallocated mass $1 - \rho _ { c , k }$ is neutral and is not redistributed to rejected tokens.

d) Semantic anchoring and prototype construction: The prototype uses the visual features $z _ { c , k , i }$ rather than their scoring projections. Let $r _ { c }$ be the unit anchor obtained by projecting the pooled intrinsic bank into the visual feature space. For $\lambda \in \ [ 0 , 1 ]$ , each raw support representation and its class-wise average are

$$
\begin{array} { l } { { \displaystyle w _ { c , k } = \lambda \sum _ { i } a _ { c , k , i } z _ { c , k , i } + ( 1 - \lambda \rho _ { c , k } ) r _ { c } } , } \\ { ~ } \\ { { \displaystyle \bar { w } _ { c } = \frac { 1 } { K } \sum _ { k } w _ { c , k } } . } \end{array}\tag{9}
$$

The anchor coefficient consists of the prescribed semantic share 1 − λ and the unallocated visual mass $\lambda ( 1 - \rho _ { c , k } )$ . Thus, visual evidence and the anchor form a convex combination, and a support with no retained token contributes exactly $r _ { c } .$ The class prototype is normalized with a numerical safeguard,

$$
\begin{array} { r } { p _ { c } = \left\{ \begin{array} { l l } { \bar { w } _ { c } / \| \bar { w } _ { c } \| _ { 2 } , } & { \| \bar { w } _ { c } \| _ { 2 } > \varepsilon _ { n } , } \\ { r _ { c } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{10}
$$

where $\varepsilon _ { n } > 0$ is a numerical threshold.

For policy learning, we also compute the pre-allocation evidence deficit,

$$
D _ { c , k } = \frac { 1 } { M } \sum _ { i } [ \xi _ { c } - d _ { c , k , i } ] + ,\tag{11}
$$

where M is the final-layer image-token count. This quantity penalizes tokens whose intrinsic advantage falls below the rejection margin. It serves as a surrogate for nuisance exposure and can also reflect ambiguous or insufficient intrinsic evidence.

## D. Counterfactual Reinforcement-Learning Gating

CRG learns fusion strengths that improve recognition while limiting contextual interference. As shown in Fig. 5, a sampled gate trajectory is evaluated alongside an independently executed reference trajectory on the same episode. Their paired utilities provide a learning signal for the gate policy.

a) Policy state and actions: Before the action at layer $\ell ,$ the policy observes a history $\mathcal { H } _ { \ell }$ containing pooled incoming support features, intrinsic and nuisance bank summaries, rejection statistics, the layer index, and previous gate actions. These previous actions determine the current support state. The scalar action retains the Beta parameterization of the conference model,

$$
g _ { \ell } \sim \pi _ { \boldsymbol { \theta } } ( \cdot \vert \mathcal { H } _ { \ell } ) = \mathrm { B e t a } ( \kappa _ { g } p _ { \boldsymbol { \theta } } ( \mathcal { H } _ { \ell } ) , \kappa _ { g } [ 1 - p _ { \boldsymbol { \theta } } ( \mathcal { H } _ { \ell } ) ] ) ,\tag{12}
$$

where $p _ { \theta } ( \mathcal { H } _ { \ell } ) \in ( 0 , 1 )$ is the predicted mean, bounded away from zero and one, and $\kappa _ { g } > 0$ is a fixed concentration.

b) Independent reference rollout: The frozen DVLA-RL gate π<sub>0</sub> supplies a reference schedule using its own supportpooled state $\mathcal { H } _ { \ell } ^ { 0 }$

$$
g _ { \ell } ^ { 0 } = \mathbb { E } _ { \pi _ { 0 } } [ g \mid \mathcal { H } _ { \ell } ^ { 0 } ] .\tag{13}
$$

The sampled and reference branches share the episode, augmentation, semantic banks, representation, and prototype rule. Each branch propagates its own support states, and the reference never receives states generated by sampled policy actions. Conditional on the episode and shared exogenous randomness, the reference rollout therefore does not depend on those actions.

c) Nuisance-aware utility: For a gate sequence ${ \textbf { g } } =$ $( g _ { 1 } , \dots , g _ { \cal L } )$ , we evaluate the evidence deficit on post-action support tokens at every selected layer,

$$
C ( \mathbf { g } ) = \frac { 1 } { L N K } \sum _ { \ell , c , k } \frac { 1 } { M _ { \ell } } \sum _ { i } [ \xi _ { c } - d _ { \ell , c , k , i } ^ { \mathrm { p o s t } } ( \mathbf { g } ) ] _ { + } ,\tag{14}
$$

where $d _ { \ell , c , k , i } ^ { \mathrm { p o s t } } ( \mathbf { g } )$ is the signed evidence from Eq. (5), evaluated on the post-action tokens using the frozen scoring projection. Averaging within each layer and across supports makes the cost comparable across layers with different token counts. This pre-selection cost prevents the policy from concealing unreliable candidate evidence by assigning it zero allocation weight.

For nonnegative coefficients $\gamma$ and $\eta ,$ the sampled reward, reference reward, and centered advantage are

$$
\begin{array} { r l } & { R ( \mathbf { g } ) = - \mathcal { L } _ { \mathrm { c l s } } ( \mathbf { g } ) - \gamma C ( \mathbf { g } ) - \displaystyle \frac { \eta } { L } \sum _ { \ell } ( g _ { \ell } - g _ { \ell } ^ { 0 } ) ^ { 2 } , } \\ & { \quad R _ { 0 } = - \mathcal { L } _ { \mathrm { c l s } } ( \mathbf { g } ^ { 0 } ) - \gamma C ( \mathbf { g } ^ { 0 } ) , } \\ & { \quad A _ { \ell } = \mathrm { s g } [ R ( \mathbf { g } ) - R _ { 0 } - b _ { \omega } ( \mathcal { H } _ { \ell } ) ] , } \end{array}\tag{15}
$$

where $b _ { \omega }$ is a history-dependent value baseline and sg stops gradients. The sampled reward balances recognition, the evidence-deficit surrogate for nuisance exposure, and deviation from the reference schedule. The reference has zero deviation cost. The policy state uses incoming tokens, whereas the cost uses post-action tokens, allowing an action to affect its utility without accessing information available only after execution.

![](images/300541e213d2db4e0c7ec0cb08594e6b12321f2b5b5b66d98f9f7c5a8cd21677.jpg)  
Fig. 5. Illustration of counterfactual reinforcement-learning gating. Sampled and frozen reference policies propagate their own states on the same episode. Their paired utilities guide gate-policy updates while the representation remains fixed.

d) Alternating optimization: Training alternates representation and policy updates after a warm-up using the reference gate. During policy-data collection and the associated policy updates, the representation, scoring projections, text encoder, reference policy, and value baseline are fixed. The rewards are computed from the final prototypes and query predictions, and policy optimization updates only θ. A separate classification step updates the representation using stopped gate actions, after which the value baseline is refitted. Its parameters are separate from the policy and representation, and its regression objective is

$$
\mathcal { L } _ { \mathrm { v a l u e } } = \mathbb { E } \Big [ ( b _ { \omega } ( \mathcal { H } _ { \ell } ) - \mathrm { s g } ( R - R _ { 0 } ) ) ^ { 2 } \Big ] .\tag{16}
$$

Query labels enter only through the classification utility. The baseline is held fixed for the next policy batch and its associated updates.

e) On-policy gradient estimation: With the environment parameters fixed, the policy maximizes expected trajectory reward. Its score-function estimator is

$$
J ( \theta ) = \mathbb { E } _ { \pi _ { \theta } } R ( \mathbf { g } ) , \qquad { \widehat { g } } = \sum _ { \ell = 1 } ^ { L } A _ { \ell } \nabla _ { \theta } \log \pi _ { \theta } ( g _ { \ell } \mid \mathcal { H } _ { \ell } ) .\tag{17}
$$

Sampled actions and histories are treated as fixed inputs when evaluating the score functions. The cosine logits and semantic responses are bounded, and the Beta policy has finite logarithmic moments; the analysis assumes integrability of the score-weighted rewards.

Proposition 1. Suppose that trajectories are sampled onpolicy, the environment and baseline parameters are fixed, policy densities are positive and differentiable with the regularity required to interchange differentiation and expectation, and score-weighted rewards are integrable. If, conditional on the episode and shared exogenous randomness, the reference rollout is independent of the sampled policy actions, then the estimator in Eq. (17) is unbiased for $\nabla _ { \boldsymbol { \theta } } J ( \boldsymbol { \theta } )$ . Conditioning on the pre-action history and shared randomness, the policy score has zero expectation, so subtracting the actionindependent reference utility and history-dependent baseline leaves the expected gradient unchanged. The paired reward difference may reduce reward variance, but this alone does not establish reduced gradient variance; Section IV evaluates gradient variance directly.

f) Clipped policy optimization: For repeated minibatch updates, we use the PPO surrogate [19],

$$
\begin{array} { c } { { r _ { \ell } = \displaystyle \frac { \pi _ { \theta } ( g _ { \ell } \mid \mathcal { H } _ { \ell } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( g _ { \ell } \mid \mathcal { H } _ { \ell } ) } , } } \\ { { \bar { r } _ { \ell } = \mathrm { c l i p } ( r _ { \ell } , 1 - \epsilon _ { p } , 1 + \epsilon _ { p } ) , } } \\ { { \mathcal { L } _ { \mathrm { C F ^ { - } P P O } } = \displaystyle - \mathbb { E } _ { \mathrm { o l d } } \sum _ { \ell } \operatorname* { m i n } ( r _ { \ell } A _ { \ell } , \bar { r } _ { \ell } A _ { \ell } ) , } } \end{array}\tag{18}
$$

where $\theta _ { \mathrm { o l d } }$ denotes the behavior-policy parameters and ${ \epsilon _ { p } } > 0$ is the clipping radius. Advantages and sampled histories remain fixed during these updates. At the behavior parameters, the surrogate gradient agrees with the negative on-policy reward gradient. After the policy changes, the clipped surrogate is not generally an unbiased estimator of that gradient. Validation evaluates the deterministic mean policy used at inference.

Overall, the representation minimizes ${ \mathcal { L } } _ { \mathrm { c l s } } ,$ the gate policy minimizes $\mathcal { L } _ { \mathrm { C F - P P O } }$ , and the baseline minimizes L<sub>value</sub> through separate alternating updates.

## E. Prototype Stability and Inference

a) Prototype error: For the analysis, let the intrinsic and nuisance token sets ${ \mathcal { T } } _ { c , k }$ and $\mathcal { N } _ { c , k }$ form a disjoint partition of the support tokens. Their retained masses are $\kappa _ { c , k } =$ $\sum _ { i \in \mathbb { Z } _ { c , k } } a _ { c , k , i }$ and $\begin{array} { r } { \nu _ { c , k } = \sum _ { i \in \mathcal { N } _ { c , k } } a _ { c , k , i } , \mathrm { s o } \kappa _ { c , k } + \nu _ { c , k } = \rho _ { c , k } } \end{array}$ Let $\mu _ { c }$ be a unit ideal class direction, and suppose intrinsic tokens, nuisance tokens, and the anchor have distances from $\mu _ { c }$ bounded by $B _ { I } , ~ B _ { N }$ , and $B _ { A }$ , respectively. Applying the triangle inequality to the convex combination in Eq. (9) yields

$$
\begin{array} { r l r } {  { \| \bar { \boldsymbol { w } } _ { c } - \boldsymbol { \mu } _ { c } \| _ { 2 } \le E _ { c } , } } \\ & { } & { E _ { c } = \frac { 1 } { K } \sum _ { k } \Big [ \lambda ( \kappa _ { c , k } B _ { I } + \nu _ { c , k } B _ { N } ) } \\ & { } & { + ( 1 - \lambda \rho _ { c , k } ) B _ { A } \Big ] . } \end{array}\tag{19}
$$

The bound separates the contributions of retained intrinsic evidence, retained nuisance evidence, and the semantic anchor. Reducing nuisance mass improves this bound when the replacement anchor is more reliable than the rejected nuisance evidence.

b) Retained nuisance mass: Proposition 2. Suppose that, conditional on the semantic banks and fixed model parameters, each nuisance signed score has mean at most $- \Delta < 0$ and a centered sub-Gaussian error with scale $\sigma > 0$ . For $M _ { N }$ nuisance tokens, the conditional expected retained nuisance mass satisfies

$$
\mathbb { E } \nu _ { c , k } \leq \operatorname* { m i n } \left\{ 1 , \frac { M _ { N } \sigma ^ { 2 } } { \tau _ { s } ( \Delta + \xi _ { c } ) } \exp \left[ - \frac { ( \Delta + \xi _ { c } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right] \right\} ,\tag{20}
$$

where the conditioning is suppressed for readability. The bound follows from the positive-part allocation bound and the sub-Gaussian tail inequality, without requiring independence among tokens. It decreases exponentially with the squared combined separation $\Delta + \xi _ { c }$ . Increasing the rejection margin suppresses nuisance allocation while increasing reliance on the anchor, making anchor quality central to the resulting stability guarantee.

c) Prediction preservation: If $E _ { c } ~ < ~ 1 - \varepsilon _ { n } ,$ then $\| \bar { w } _ { c } \| _ { 2 } ~ > ~ \varepsilon _ { n }$ and normalization gives $\| p _ { c } - \mu _ { c } \| _ { 2 } \leq 2 E _ { c } .$ For a fixed unit query representation with ideal predicted class y, the prediction is preserved whenever its ideal cosine margin exceeds $2 ( E _ { y } + \operatorname* { m a x } _ { c \neq y } E _ { c } )$ . This guarantee concerns prototype perturbations for a fixed query representation.

d) Inference: At inference, the support-conditioned semantic banks, deterministic mean gate schedule, and class prototypes are computed and cached once per episode. Every query reuses the same semantic tokens and gate schedule, keeping its representation compatible with the support prototypes. Each query is processed independently through one forward pass followed by prototype comparison, with no reference rollout or parameter update.

## IV. EXPERIMENTS

We evaluate DVLA-RL++ in standard, fine-grained, and cross-domain few-shot recognition, followed by component ablations and diagnostics of evidence reliability and policy learning.

## A. Experimental Settings

a) Datasets: We consider three evaluation scenarios. Standard FSL uses miniImageNet [1], tieredImageNet [22], and CIFAR-FS [23]. Fine-grained FSL uses CUB-200- 2011 [24], Stanford Cars [25], and Stanford Dogs [26]. Crossdomain FSL trains on miniImageNet and evaluates on CUB, Places [27], and ChestX [28]. These protocols cover nine benchmark settings across eight distinct datasets. We follow the standard class splits, with disjoint training, validation, and test categories.

b) Implementation Details: Following recent semantic FSL studies [12, 7, 9], we use Visformer-Tiny [29] as the visual backbone, the frozen ViT-B/16 CLIP text encoder [13] for 512-dimensional text embeddings, and Qwen2.5-VL-32B [30] for semantic generation. Inputs are resized to 224×224. Training consists of pre-training and episodic meta-tuning [31]. The representation uses AdamW [32] with an initial learning rate of $5 \times 1 0 ^ { - 4 }$ and a cosine schedule [33]. Pre-training runs for 300 epochs on tieredImageNet and 800 epochs on the other training datasets with a batch size of 512, followed by 100 epochs of meta-tuning.

We set $\kappa _ { g } = 1 0 , \tau _ { \mathrm { c l s } } = 0 . 2 , \tau _ { b } = 0 . 1 , \tau _ { s } = 0 . 2 , \xi _ { 0 } =$ 0.05, and $\kappa _ { o } ~ = ~ 0 . 1$ . The remaining defaults are $\lambda { \mathrm { ~ = ~ } } 0 . 7 ,$ $\gamma = 0 . 5 , \eta = 0 . 1$ , and $R = 8$ local phrases per class and semantic role. Scoring projections are calibrated on labeled base supports with $\tau _ { a } = 0 . 1$ and frozen during policy learning. Hyperparameters are selected by base-validation accuracy.

TABLE I  
RESULTS (%) ON MINIIMAGENET, TIEREDIMAGENET, AND CIFAR-FS UNDER THE 5-WAY 1-SHOT AND 5-SHOT SETTINGS. MEAN ACCURACY AND 95% CONFIDENCE INTERVALS ARE REPORTED. BOLD INDICATES THE HIGHEST MEAN ACCURACY.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="2">miniImageNet</td><td colspan="2">tieredImageNet</td><td colspan="2">CIFAR-FS</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>ProtoNet [2]</td><td>NeurIPS 2017</td><td> $6 2 . 3 9 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $8 0 . 5 3 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $6 8 . 2 3 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $8 4 . 0 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $7 2 . 2 0 { \scriptstyle \pm 0 . 7 0 }$ </td><td>83.50±0.50</td></tr><tr><td>AM3 [6]</td><td>NeurIPS 2019</td><td> $6 5 . 3 0 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $7 8 . 1 0 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $6 9 . 0 8 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $8 2 . 5 8 { \scriptstyle \pm 0 . 3 1 }$ </td><td></td><td></td></tr><tr><td>DeepEMD [3]</td><td>CVPR 2020</td><td> $6 5 . 9 1 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $8 2 . 4 1 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $7 1 . 1 6 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $8 6 . 0 3 { \scriptstyle \pm 0 . 5 8 }$ </td><td></td><td></td></tr><tr><td>SUÑ [31]</td><td>ECCV 2022</td><td> $6 7 . 8 0 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $8 3 . 2 5 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $7 2 . 9 9 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $8 6 . 7 4 { \scriptstyle \pm 0 . 3 3 }$ </td><td></td><td></td></tr><tr><td>FewTURE [37]</td><td>NeurIPS 2022</td><td> $6 8 . 0 2 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $8 4 . 5 1 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $7 2 . 9 6 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $8 6 . 4 3 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $7 2 . 8 0 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $8 6 . 1 4 { \scriptstyle \pm 0 . 6 4 }$ </td></tr><tr><td>SVAE [35]</td><td>CVPR 2022</td><td> $7 4 . 8 4 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $8 3 . 2 8 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $7 6 . 9 8 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $8 5 . 7 7 { \scriptstyle \pm 0 . 5 0 }$ </td><td></td><td></td></tr><tr><td>Meta-AdaM [38]</td><td>NeurIPS 2023</td><td> $5 9 . 8 9 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $7 7 . 9 2 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $6 5 . 3 1 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $8 5 . 2 4 { \scriptstyle \pm 0 . 3 5 }$ </td><td></td><td></td></tr><tr><td>ProtoDiff [39]</td><td>NeurIPS 2023</td><td> $6 6 . 6 3 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $8 3 . 4 8 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $7 2 . 9 5 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $8 5 . 1 5 { \scriptstyle \pm 0 . 1 8 }$ </td><td></td><td></td></tr><tr><td>CPEA [40]</td><td>ICCV 2023</td><td> $7 1 . 9 7 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $8 7 . 0 6 { \pm } 0 . 3 8 $ </td><td> $7 6 . 9 3 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $9 0 . 1 2 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $7 7 . 8 2 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $8 8 . 9 8 { \scriptstyle \pm 0 . 4 5 }$ </td></tr><tr><td>SP [7]</td><td>CVPR 2023</td><td> $7 2 . 3 1 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $8 3 . 4 2 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $7 8 . 0 3 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $8 8 . 5 5 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $8 2 . 1 8 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $8 8 . 2 4 { \scriptstyle \pm 0 . 3 2 }$ </td></tr><tr><td>ALFA [41]</td><td>TPAMI 2024</td><td> $6 6 . 6 1 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $8 1 . 4 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $7 0 . 2 9 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $8 6 . 1 7 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $7 6 . 3 2 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $8 6 . 7 3 { \scriptstyle \pm 0 . 3 1 }$ </td></tr><tr><td>LastShot [42]</td><td>TPAMI 2024</td><td> $6 7 . 3 5 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $8 2 . 5 8 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $7 2 . 4 3 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $8 5 . 8 2 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $7 6 . 7 6 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $8 7 . 4 9 { \scriptstyle \pm 0 . 1 2 }$ </td></tr><tr><td>MetaKernel [43]</td><td>TPAMI 2024</td><td> $6 6 . 9 { \pm } 0 . 3 $ </td><td> $8 1 . 3 { \pm } 0 . 2 $ </td><td></td><td></td><td> $7 6 . 1 \pm 0 . 4$ </td><td> $8 6 . 9 { \pm } 0 . 5 $ </td></tr><tr><td>SIFT [44]</td><td>IJCV 2024</td><td> $7 7 . 3 1 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $8 6 . 9 5 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $7 7 . 8 6 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $8 9 . 8 9 { \scriptstyle \pm 0 . 5 2 }$ </td><td></td><td></td></tr><tr><td>KTPP [12]</td><td>ACM MM 2024</td><td> $7 6 . 7 1 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $8 6 . 4 6 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $8 0 . 8 0 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $9 0 . 0 1 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $8 3 . 6 3 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $9 0 . 1 9 { \scriptstyle \pm 0 . 3 0 }$ </td></tr><tr><td>SemFew [8]</td><td>CVPR 2024</td><td> $7 8 . 9 4 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $8 6 . 4 9 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $8 2 . 3 7 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $8 9 . 8 9 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $8 4 . 3 4 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $8 9 . 1 1 { \scriptstyle \pm 0 . 5 4 }$ </td></tr><tr><td>ECER [9]</td><td>AAAI 2025</td><td> $8 1 . 1 4 { \pm } 0 . 1 5$ </td><td></td><td> $8 1 . 8 1 { \scriptstyle \pm 0 . 5 1 }$ </td><td></td><td> $8 6 . 0 1 { \scriptstyle \pm 0 . 3 5 }$ </td><td></td></tr><tr><td>CPL [45]</td><td>TPAMI 2025</td><td> $7 2 . 8 2 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $8 7 . 9 3 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $7 8 . 0 5 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $9 0 . 8 9 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $7 8 . 8 2 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $8 9 . 9 8 { \scriptstyle \pm 0 . 4 4 }$ </td></tr><tr><td>FRN+IAM [46]</td><td>TPAMI 2025</td><td> $6 6 . 9 6 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $8 3 . 1 9 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $7 1 . 8 5 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $8 6 . 5 5 { \scriptstyle \pm 0 . 1 5 }$ </td><td></td><td></td></tr><tr><td>FKFRN [47]</td><td>TIP 2026</td><td> $6 6 . 7 8 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $8 3 . 2 6 { \pm } 0 . 1 9$ </td><td> $7 1 . 3 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $8 6 . 3 5 { \scriptstyle \pm 0 . 1 4 }$ </td><td></td><td></td></tr><tr><td>LE-ProtoPNet [48]</td><td>TIP 2026</td><td> $8 1 . 3 4 { \pm } 0 . 3 2 $ </td><td> $8 7 . 1 7 { \scriptstyle \pm 0 . 2 6 }$ </td><td></td><td></td><td> $7 9 . 6 9 { \scriptstyle \pm 0 . 4 2 }$ </td><td>86.40±0.33</td></tr><tr><td>DVLA-RL [10]</td><td>ICLR 2026</td><td> $8 1 . 6 9 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $8 8 . 2 5 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $8 3 . 0 2 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $9 1 . 7 1 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $8 7 . 1 8 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $9 0 . 5 9 { \scriptstyle \pm 0 . 3 1 }$ </td></tr><tr><td>DVLA-RL++</td><td>Ours</td><td> $\mathbf { 8 3 . 4 1 \pm 0 . 3 4 }$ </td><td> $\mathbf { 8 9 . 6 3 { \scriptstyle \pm 0 . 2 6 } }$ </td><td> $\mathbf { 8 4 . 5 7 { \scriptstyle \pm 0 . 4 1 } }$ </td><td> $\mathbf { 9 2 . 8 9 { \scriptstyle \pm 0 . 2 7 } }$ </td><td> $\mathbf { 8 8 . 7 0 { \scriptstyle \pm 0 . 3 8 } }$ </td><td> $\mathbf { 9 1 . 7 5 { \scriptstyle \pm 0 . 2 9 } }$ </td></tr></table>

After a five-epoch warm-up with the reference gate, policy and representation updates alternate using 32 episodes per batch. PPO uses four optimization epochs per policy batch, minibatches of 16 episodes, and $\epsilon _ { p } = 0 . 2 .$ The policy and value baseline use Adam with learning rates of $1 0 ^ { - 4 } \mathrm { a n d } 1 0 ^ { - 3 }$ respectively; policy gradients are clipped to a norm of 1.0. All experiments run on an NVIDIA RTX 6000 Ada GPU.

DVLA-RL++ exceeds these results by 2.1%–2.7% and 1.6%– 2.0%, respectively. These results highlight the effectiveness of purifying support evidence and adaptively fusing it through counterfactual gating.

c) Evaluation Protocol: We follow the episodic FSL protocol [7, 12, 6, 34, 35, 36]. Main comparisons use 5-way 1- shot and 5-shot episodes. Each setting includes 2,000 episodes sampled uniformly from novel classes, with 15 queries per class sampled separately from the supports. We report mean episode accuracy and its 95% confidence interval. At test time, queries are processed independently using cached supportderived semantics, gates, and prototypes. Component ablations share episode lists, training seeds, and semantic caches. Published competitor results are taken from their original reports. DVLA-RL [10] is the direct conference baseline and uses the same backbone, semantic generator, and evaluation protocol. A dash denotes an unreported setting.

b) Fine-Grained Recognition: Table II reports results on CUB, Stanford Cars, and Stanford Dogs. DVLA-RL++ achieves the highest mean accuracy in all six settings. Relative to DVLA-RL, the gains are 2.18%–2.41% in 1-shot recognition and 0.55%–1.25% in 5-shot recognition. DVLA-RL++ improves on the strongest other methods, SUITED and FKFRN, by 5.2%–12.2% in 1-shot recognition and by up to 2.3% in 5-shot recognition. The larger 1-shot gains reflect the sensitivity of prototype estimation to limited support evidence. This shows that DVLA-RL++ captures subtle inter-class differences and preserves intra-class consistency by suppressing incidental context.

## B. Main Results

a) Standard Few-Shot Recognition: Table I compares performance on the three standard benchmarks. DVLA-RL++ achieves the highest mean accuracy in all six settings and improves on DVLA-RL by 1.2%–1.7%. Since both models share the backbone, semantic generator, and protocol, this gain reflects the benefit of CSP and CRG. The strongest other methods are LE-ProtoPNet, SemFew, or ECER in 1- shot recognition and CPL or KTPP in 5-shot recognition.

c) Cross-Domain Recognition: Table III evaluates transfer from miniImageNet to CUB, Places, and ChestX. DVLA-RL++ obtains the highest reported mean accuracy in all six settings and improves on DVLA-RL by 1.0% on average. On CUB and Places, the gains over DVLA-RL are 1.22%–1.68%. Relative to other published methods, DVLA-RL++ exceeds the strongest 1-shot results of CSC by 10.3% on CUB and 7.1% on Places. In 5-shot recognition, it exceeds STL DeepBDC+IAM by 3.9% on CUB and SVasP by 4.1% on Places. These results demonstrate that DVLA-RL++ learns more transferable representations by retaining intrinsic object evidence, generalizing well to novel categories under distribution shifts.

TABLE II  
RESULTS (%) ON CUB, STANFORD CARS, AND STANFORD DOGS UNDER THE 5-WAY 1-SHOT AND 5-SHOT SETTINGS. MEAN ACCURACY AND 95% CONFIDENCE INTERVALS ARE REPORTED. BOLD INDICATES THE HIGHEST MEAN ACCURACY.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="2">CUB</td><td colspan="2">Stanford Cars</td><td colspan="2">Stanford Dogs</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>ProtoNet [2]</td><td>NeurIPS 2017</td><td> $6 3 . 4 4 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $8 3 . 1 7 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $4 5 . 0 1 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $8 7 . 1 9 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $4 1 . 6 1 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $7 6 . 7 8 { \scriptstyle \pm 0 . 3 6 }$ </td></tr><tr><td>FRN [49]</td><td>CVPR 2021</td><td> $8 3 . 5 5 { \pm } 0 . 1 9$ </td><td> $9 2 . 9 2 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $5 8 . 9 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $7 9 . 6 5 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $4 9 . 3 7 { \scriptstyle \pm 0 . 2 0 }$ </td><td>67.13±0.17</td></tr><tr><td>DAN [50]</td><td>AAAI 2022</td><td> $7 2 . 8 9 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $8 6 . 6 0 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $7 0 . 2 1 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $8 5 . 5 5 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $5 9 . 8 1 { \scriptstyle \pm 0 . 5 0 }$ </td><td>77.19±0.35</td></tr><tr><td>MFGN [51]</td><td>IJCAI 2022</td><td> $8 4 . 0 1 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $9 1 . 8 5 { \scriptstyle \pm 0 . 2 1 } $ </td><td></td><td></td><td> $7 4 . 8 1 { \scriptstyle \pm 0 . 4 4 }$ </td><td>86.52±0.26</td></tr><tr><td>TDM [52]</td><td>CVPR 2022</td><td> $8 4 . 3 6 { \pm } 0 . 1 9$ </td><td> $9 3 . 3 7 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $6 8 . 3 6 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $8 6 . 1 4 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $5 7 . 6 4 { \scriptstyle \pm 0 . 2 2 }$ </td><td>75.03±0.16</td></tr><tr><td>BSFA [11]</td><td>TCSVT 2023</td><td> $8 6 . 0 0 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $9 2 . 5 3 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $8 8 . 9 3 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $9 5 . 2 0 { \scriptstyle \pm 0 . 2 0 } $ </td><td> $6 9 . 5 8 { \scriptstyle \pm 0 . 5 0 }$ </td><td>82.59±0.33</td></tr><tr><td>MLI [53]</td><td>TIP 2024</td><td> $8 5 . 9 4 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $9 3 . 5 0 { \scriptstyle \pm 0 . 2 9 }$ </td><td></td><td></td><td> $7 6 . 3 2 { \scriptstyle \pm 0 . 4 7 }$ </td><td>88.25±0.27</td></tr><tr><td>C2-Net [54]</td><td>AAAI 2024</td><td> $8 3 . 3 1 { \scriptstyle \pm 0 . 4 1 }$ </td><td>92.18±0.23</td><td> $8 8 . 9 6 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $9 5 . 1 6 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $7 5 . 5 0 { \scriptstyle \pm 0 . 4 9 }$ </td><td>87.65±0.28</td></tr><tr><td>SUITED [55]</td><td>AAAI 2025</td><td> $8 6 . 0 2 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $9 4 . 1 3 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $8 9 . 9 7 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $9 6 . 5 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $7 6 . 5 5 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $8 8 . 8 6 { \scriptstyle \pm 0 . 2 7 }$ </td></tr><tr><td>FRN+TDM+IAM [46]</td><td>TPAMI 2025</td><td> $8 4 . 8 4 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $9 3 . 6 0 { \scriptstyle \pm 0 . 1 0 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>FKFRN [47]</td><td>TIP 2026</td><td> $8 4 . 7 0 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $9 4 . 1 1 { \pm } 0 . 1 3 $ </td><td> $8 9 . 6 0 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $9 7 . 1 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $7 9 . 8 8 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $9 0 . 4 1 { \scriptstyle \pm 0 . 1 2 }$ </td></tr><tr><td>LE-ProtoPNet [48]</td><td>TIP 2026</td><td> $8 3 . 4 3 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $9 2 . 0 7 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $8 5 . 3 8 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $9 2 . 2 0 { \scriptstyle \pm 0 . 1 6 }$ </td><td></td><td></td></tr><tr><td>DVLA-RL [10]</td><td>ICLR 2026</td><td> $9 1 . 9 3 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $9 5 . 0 6 { \pm } 0 . 1 9$ </td><td> $9 2 . 9 5 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $9 6 . 5 9 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $8 9 . 6 4 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $9 1 . 4 2 { \scriptstyle \pm 0 . 2 5 }$ </td></tr><tr><td>DVLA-RL++</td><td>Ours</td><td> $\mathbf { 9 4 . 3 1 { \scriptstyle \pm 0 . 2 6 } }$ </td><td> ${ \bf 9 6 . 2 8 { \scriptstyle \pm 0 . 1 7 } }$ </td><td> $\mathbf { 9 5 . 1 3 { \scriptstyle \pm 0 . 2 2 } }$ </td><td> $\mathbf { 9 7 . 1 4 { \scriptstyle \pm 0 . 1 4 } }$ </td><td> $\mathbf { 9 2 . 0 5 { \scriptstyle \pm 0 . 2 8 } }$ </td><td> $\mathbf { 9 2 . 6 7 { \scriptstyle \pm 0 . 2 3 } }$ </td></tr></table>

TABLE III

CROSS-DOMAIN RESULTS (%) WITH TRAINING ON MINIIMAGENET AND EVALUATION ON CUB, PLACES, AND CHESTX. MEAN ACCURACY AND 95% CONFIDENCE INTERVALS ARE REPORTED. BOLD INDICATES THE HIGHEST MEAN ACCURACY.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="2">CUB</td><td colspan="2">Places</td><td colspan="2">ChestX</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>GNN [56]</td><td>ICLR 2018</td><td> $4 4 . 4 0 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $6 2 . 8 7 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $5 2 . 4 2 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $7 0 . 9 1 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $2 2 . 0 0 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $2 5 . 2 7 { \scriptstyle \pm 0 . 5 9 }$ </td></tr><tr><td>FWT [57]</td><td>ICLR 2020</td><td> $4 5 . 5 0 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $6 4 . 9 7 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $5 3 . 4 4 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $7 0 . 7 0 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $2 2 . 0 4 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $2 5 . 1 8 { \scriptstyle \pm 0 . 4 5 }$ </td></tr><tr><td>AFA [58]</td><td>ECCV 2022</td><td> $4 6 . 8 6 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $6 8 . 2 5 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $5 4 . 0 4 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $7 6 . 2 1 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $2 2 . 9 2 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $2 5 . 0 2 { \scriptstyle \pm 0 . 2 0 }$ </td></tr><tr><td>UCD [59]</td><td>NeurIPS 2022</td><td> $4 0 . 6 5 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $5 8 . 5 4 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $5 1 . 8 4 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $7 2 . 1 9 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $2 2 . 6 4 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $2 6 . 2 6 { \scriptstyle \pm 0 . 4 5 }$ </td></tr><tr><td>ATA [60]</td><td>AIJ 2023</td><td> $4 5 . 0 0 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $6 6 . 2 2 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 3 . 5 7 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $7 5 . 4 8 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $2 2 . 1 0 { \scriptstyle \pm 0 . 2 0 }$ </td><td>24.32±0.40</td></tr><tr><td>LDP-Net [61]</td><td>CVPR 2023</td><td> $4 9 . 8 2 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $7 0 . 3 9 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $5 3 . 8 2 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $7 2 . 9 0 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $2 3 . 0 1 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $2 6 . 6 7 { \scriptstyle \pm 0 . 4 0 }$ </td></tr><tr><td>StyleAdv [62]</td><td>CVPR 2023</td><td> $4 8 . 4 9 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $6 8 . 7 2 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $5 8 . 5 8 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $7 7 . 7 3 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $2 2 . 6 4 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $2 6 . 0 7 { \scriptstyle \pm 0 . 3 7 }$ </td></tr><tr><td>FÁP [63]</td><td>IJCAI 2024</td><td> $5 0 . 5 6 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $6 4 . 1 7 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $5 7 . 3 4 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $7 2 . 0 5 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $2 1 . 5 6 { \scriptstyle \pm 0 . 2 0 }$ </td><td>24.15±0.20</td></tr><tr><td>FLoR [64]</td><td>CVPR 2024</td><td> $4 9 . 9 9 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $7 0 . 3 9 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $5 3 . 1 8 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $7 2 . 3 1 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $2 3 . 1 1 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $2 6 . 7 0 { \scriptstyle \pm 0 . 4 0 }$ </td></tr><tr><td>MEFP [65]</td><td>NeurIPS 2024</td><td> $5 1 . 5 5 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $7 3 . 6 1 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $5 2 . 0 6 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $7 3 . 7 8 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $2 2 . 8 2 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $2 6 . 5 3 { \scriptstyle \pm 0 . 4 0 }$ </td></tr><tr><td>SVasP [66]</td><td>AAAI 2025</td><td> $4 9 . 4 9 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $6 8 . 9 5 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $5 9 . 0 7 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $7 7 . 7 8 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $2 3 . 2 3 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $2 6 . 8 7 { \scriptstyle \pm 0 . 3 8 }$ </td></tr><tr><td>STL DeepBDC+IAM [46]</td><td>TPAMI 2025</td><td> $5 5 . 9 2 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $7 6 . 4 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>CSC [67]</td><td>TIP 2026</td><td> $5 8 . 8 6 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $7 6 . 1 7 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $6 3 . 7 4 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $7 7 . 2 8 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $2 2 . 7 2 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $2 6 . 2 8 { \scriptstyle \pm 0 . 4 5 }$ </td></tr><tr><td> $\mathrm { { D V L A - R L } \ [ 1 0 ] }$ </td><td>ICLR 2026</td><td> $6 7 . 4 6 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $7 8 . 9 9 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $6 9 . 2 6 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $8 0 . 7 0 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $2 3 . 4 7 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $2 6 . 9 4 { \scriptstyle \pm 0 . 2 2 }$ </td></tr><tr><td> $\mathbf { D V L A - R L + + }$ </td><td>Ours</td><td> ${ \bf 6 9 . 1 4 } { \bf \pm 0 . 4 5 }$ </td><td> ${ \bf 8 0 . 3 0 { \scriptstyle \pm 0 . 3 3 } }$ </td><td> $\mathbf { 7 0 . 8 7 { \scriptstyle \pm 0 . 4 3 } }$ </td><td> $\mathbf { 8 1 . 9 2 { \scriptstyle \pm 0 . 3 4 } }$ </td><td> $\mathbf { 2 3 . 6 5 \pm 0 . 1 9 }$ </td><td> $\mathbf { 2 7 . 1 0 { \scriptstyle \pm 0 . 2 1 } }$ </td></tr></table>

## C. Model Analysis

Component ablations use the 5-way 1-shot setting with matched semantic caches and training budgets. Each variant is trained with five seeds and evaluated on 2,000 episodes per seed. Clean accuracy is measured on the original queries, while Shift accuracy is measured on matched queries after replacing the background or an incidental co-occurring object while preserving the foreground object and its label. The paired evaluation uses the same supports and trained model, isolating the effect of the query-context intervention.

TABLE IV  
COMPONENT ABLATIONS (%) ON MINIIMAGENET AND CUB UNDER THE 5-WAY 1-SHOT SETTING. CLEAN AND SHIFT DENOTE ACCURACY BEFORE AND AFTER THE MATCHED CONTEXT INTERVENTION.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">CSP</td><td rowspan="2">CRG</td><td colspan="2">miniImageNet</td><td colspan="2">CUB</td></tr><tr><td>Clean</td><td>Shift</td><td>Clean</td><td>Shift</td></tr><tr><td>DVLA-RL</td><td>X</td><td>X</td><td>81.7</td><td>75.8</td><td>91.9</td><td>85.0</td></tr><tr><td>+ CSP</td><td>√</td><td>×</td><td>82.6</td><td>79.4</td><td>93.4</td><td>89.4</td></tr><tr><td>+ CRG</td><td>X</td><td>√</td><td>82.4</td><td>78.4</td><td>93.0</td><td>88.1</td></tr><tr><td> $\mathbf { D V L A - R L + + }$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>83.4</td><td>81.2</td><td>94.3</td><td>92.0</td></tr></table>

a) Component Ablations: We conduct component ablations on miniImageNet and CUB to evaluate CSP and CRG, as shown in Table IV. The CSP-only variant retains the conference gate, and the CRG-only variant retains uniform visual support prototypes. It is to be noted that (1) CSP alone improves Shift accuracy over DVLA-RL by 3.6% and 4.4%, showing that selecting support tokens with complementary semantics suppresses incidental context. (2) CRG alone improves Shift accuracy by 2.6% and 3.1%, indicating that reliability-aware gating reduces the influence of unreliable semantics during fusion. (3) The best performance is achieved when both are used. The full model improves Shift accuracy by 5.4% and 7.0% and Clean accuracy by 1.7% and 2.4%. Its Clean-to-Shift drop decreases from 5.9% to 2.2% on miniImageNet and from 6.9% to 2.3% on CUB. These results demonstrate the complementary roles of selective support aggregation and reliability-aware semantic fusion.

![](images/6a7d1f39f707c678b497ecbc9085e5336e240c3f9fc436c85bd51a9ce0708b0a.jpg)

![](images/8dd80876b0e5a62636c1a61c544785625cd4488ed2bab790ef14d87c09e064b8.jpg)

![](images/b676de3168cd695fb2cf3848a0df8f93c4e7aae5863cd2ef25711c63c9d607d9.jpg)  
Fig. 6. Parameter sensitivity on miniImageNet (blue solid) and CUB (orange dashed) under the 5-way 1-shot setting. Panels vary (a) τ , (b) λ, and ${ \bf ( c ) } \gamma .$ Accuracy changes are relative to each dataset’s default setting.

![](images/ed41cd81e255c145e59f2fbcb51bec78fb59d21619578881ecd07eec05d357a0.jpg)  
Fig. 7. Signed-margin distributions of intrinsic (blue) and nuisance (orange) evidence on miniImageNet and CUB. Rows compare DVLA-RL and DVLA-RL++. OVL denotes histogram overlap; the dotted line marks the rejection boundary $d - \xi _ { c } = 0$

b) Internal Design Choices: Table V replaces or removes individual design elements from the complete model using the same Clean and Shift protocol. Uniform allocation preserves the anchor mixture but sets token weights to $1 / M ;$ positive-only semantics removes the nuisance bank; and the no-neutral control forces unit allocation mass. Other controls use a support-independent gate, a fixed rejection margin, mean visual evidence in place of the intrinsic fallback, a state baseline without the paired reference, or a zero nuisance-cost weight.

Every individual control reduces Shift accuracy more than Clean accuracy. Uniform allocation produces the largest Shift degradation among these design controls, at 3.6% on miniImageNet and 4.8% on CUB. Removing nuisance descriptions costs 2.2% and 2.9%, while removing the neutral option or intrinsic fallback costs 1.5%–2.2%. The static gate costs

TABLE V  
INTERNAL DESIGN CONTROLS (%) ON MINIIMAGENET AND CUB UNDER THE 5-WAY 1-SHOT SETTING. CLEAN AND SHIFT FOLLOW THE SAME PROTOCOL AS TABLE IV.
<table><tr><td rowspan="2">Variant</td><td colspan="2">miniImageNet</td><td colspan="2">CUB</td></tr><tr><td>Clean</td><td>Shift</td><td>Clean</td><td>Shift</td></tr><tr><td>DVLA-RL</td><td>81.7</td><td>75.8</td><td>91.9</td><td>85.0</td></tr><tr><td>Uniform allocation</td><td>82.2</td><td>77.6</td><td>92.8</td><td>87.2</td></tr><tr><td>Positive semantics only</td><td>82.7</td><td>79.0</td><td>93.5</td><td>89.1</td></tr><tr><td>No neutral option</td><td>83.0</td><td>79.7</td><td>93.8</td><td>90.0</td></tr><tr><td>Static gate</td><td>82.8</td><td>79.8</td><td>93.6</td><td>90.0</td></tr><tr><td>Fixed margin</td><td>83.1</td><td>80.1</td><td>94.0</td><td>90.6</td></tr><tr><td>No intrinsic fallback</td><td>82.9</td><td>79.6</td><td>93.7</td><td>89.8</td></tr><tr><td>State baseline only</td><td>83.0</td><td>80.4</td><td>94.0</td><td>90.9</td></tr><tr><td>No nuisance cost</td><td>83.1</td><td>80.1</td><td>94.0</td><td>90.5</td></tr><tr><td>DVLA-RL++</td><td>83.4</td><td>81.2</td><td>94.3</td><td>92.0</td></tr></table>

1.4% and 2.0%, and using the state baseline alone costs 0.8% and 1.1%. Fixing the margin or removing the nuisance cost also lowers Shift accuracy by 1.1%–1.5%. The pattern links contextual robustness to evidence selection, rejection and fallback handling, and the paired policy-learning signal.

c) Parameter Sensitivity: Fig. 6 varies the allocation temperature $\tau _ { s } ,$ visual-evidence coefficient λ, and nuisancecost weight γ one at a time under the 5-way 1-shot setting. The reference-deviation weight and local-bank budget remain fixed at $\eta = 0 . 1$ and $R = 8 .$ . The temperature curve favors an intermediate allocation regime: very low temperatures concentrate weight on fewer tokens, while higher temperatures spread it across more candidates. The default $\tau _ { s } = 0 . 2$ balances these effects. On miniImageNet, accuracy remains within 0.3% of the default for sampled λ values in [0.5, 0.9] and within 0.2% for sampled γ values in [0.1, 0.75]. Similar trends on CUB indicate a stable range around the selected settings.

d) Context and Evidence Diagnostics: Fig. 7 compares the signed margins $m \ : = \ : d \ : - \ : \xi _ { c }$ of intrinsic and nuisance evidence, as defined by Eqs. (5) and (6). Intrinsic evidence corresponds to class-defining object cues, whereas nuisance evidence corresponds to incidental context. Both distributions use fixed common bins and unit-integral densities; histogram overlap is the area shared by the two densities. Overlap decreases from 0.74 to 0.16 on miniImageNet and from 0.75 to 0.12 on CUB, indicating increased separation around the rejection boundary.

(b) Policy diagnostics  
![](images/0fe06c90d0887d72d946e2b0fa3c17a921d4f5ebf9ff251462959a9f97a7586e.jpg)  
Fig. 9. Attention maps of DVLA-RL, CSP alone, CRG alone, and DVLA-RL++ on nine miniImageNet images. Columns (a)–(i) cover object co-occurrence occlusion, clutter, and camouflage.

We further audit evidence independently on 600 matched 5-way 1-shot test episodes per dataset on miniImageNet and CUB, comprising 3,000 support-image occurrences per dataset. Intrinsic and nuisance tokens are identified using independently annotated class-defining object regions and incidental context. Annotators are blinded to semantic scores, signed margins, and allocation weights. For each support image, we compute the fraction of tokens with positive allocation weights, intrinsic precision over unambiguously annotated retained tokens, and neutral mass $1 - \textstyle \sum _ { i } a _ { i }$ . Metrics are averaged within and across episodes. At the default setting, the audit reports 24% token retention, 94% intrinsic precision over unambiguously annotated retained tokens, and 0.08 neutral mass.

e) Policy Diagnostics and Inference: Fig. 8(a) evaluates the gradient estimator using 200 independent fresh minibatches per frozen checkpoint. The representation, scorer, objective, and behavior policy are fixed; estimators share the reward and differ only in their control variates. We measure the trace of the on-policy gradient covariance and normalize it by that of the state-baseline estimator. The rawreturn estimator has a normalized covariance trace of 1.25, while the paired estimator with the state baseline has a value of 0.60, corresponding to a 40% reduction relative to the state baseline. This empirical comparison complements the unbiasedness conditions in Proposition 1 by directly measuring gradient variance.

Fig. 8(b) compares policy diagnostics for the state-baseline and paired estimators. The frequency of $C ( \mathbf { g } ) > C ( \mathbf { g } ^ { 0 } )$ decreases from 24% to 9%, where C is the evidence-deficit surrogate for nuisance exposure. Active PPO clipping is recorded after updates when $A _ { \ell } > 0$ and $\begin{array} { r } { r _ { \ell } > 1 + \epsilon _ { p } , } \end{array}$ or when $A _ { \ell } < 0$ and $r _ { \ell } < 1 - \epsilon _ { p } ;$ its frequency decreases from 22% to 15%. At inference, semantic banks, prototypes, and the support-derived mean gate schedule are cached once per episode. Each query requires one forward pass and prototype comparison, without a reference rollout or value-network evaluation.

![](images/b3bf16e8e77490c913b9b5fcc20c80ab0473a5b42973dec6c739c2fedcd29d7c.jpg)

![](images/058c3fbb4a91c6d8cea3ac133738fb834fc2d6bc8d11f82b54189cd0a4115953.jpg)  
Fig. 8. Policy diagnostics on miniImageNet. (a) Gradient-covariance trace normalized by the state-baseline estimator. (b) Frequencies of nuisance-cost increase and active PPO clipping for the state baseline (blue) and paired estimator (green).

f) Qualitative Analysis: The baseline is constructed by removing CSP and CRG while retaining the same backbone and the same sampled support images to ensure a fair comparison. Fig. 9 compares the baseline, CSP alone, CRG alone, and DVLA-RL++ on nine miniImageNet images containing co-occurring objects, occlusion, clutter, or camouflage. The baseline attention frequently includes competing objects and contextual regions, such as the cat beside the dog in (a), branches around the bird in (c), and snow around the fox in (e). CSP alone concentrates more on object details but provides incomplete coverage, while CRG alone covers the object more broadly with residual contextual responses. The combined model shows broader object coverage and weaker contextual responses in the displayed examples, consistent with the complementary behavior observed in the component ablations.

![](images/1dd6d9d839b8a1028592f80901656b11e9981106580747e1ce8714c882738100.jpg)  
Fig. 10. T-SNE visualization on novel classes from four datasets.

Fig. 10 visualizes query features with t-SNE [68] on mini-ImageNet, CIFAR-FS, Stanford Dogs, and Places. Each dataset includes five novel classes and 100 queries per class. DVLA-RL++ exhibits more compact class neighborhoods and clearer separation in the displayed projections.

## V. CONCLUSION

We presented DVLA-RL++, an extension of DVLA-RL that addresses support-evidence contamination in few-shot recognition. Complementary semantic purification contrasts intrinsic and nuisance semantics to select reliable support tokens, while an intrinsic semantic anchor stabilizes prototype construction. Counterfactual reinforcement-learning gating adapts semantic fusion using a nuisance-aware reward and independently executed policy and reference trajectories. Theoretical analysis relates retained nuisance mass to prototype error and classification margins, and establishes conditions for unbiased on-policy gradient estimation. Experiments on standard, fine-grained, and cross-domain benchmarks show consistent improvements over DVLA-RL, with an average accuracy gain of 1.4%. Matched context interventions, independent evidence audits, and policy diagnostics further support the effectiveness of evidence purification and counterfactual gating in improving contextual robustness and reducing gradient variance. Future work will strengthen semantic grounding under larger domain shifts and reduce the cost of support-conditioned generation.

## REFERENCES

[1] O. Vinyals, C. Blundell, T. Lillicrap, D. Wierstra et al., “Matching networks for one shot learning,” in NeurIPS, 2016, pp. 3630–3638.

[2] J. Snell, K. Swersky, and R. Zemel, “Prototypical networks for few-shot learning,” in NeurIPS, 2017, pp. 4077–4087.

[3] C. Zhang, Y. Cai, G. Lin et al., “Deepemd: Few-shot image classification with differentiable earth mover’s distance and structured classifiers,” in CVPR, 2020, pp. 12 203–12 213.

[4] H.-J. Ye, H. Hu, D.-C. Zhan, and F. Sha, “Few-shot learning via embedding adaptation with set-to-set functions,” in CVPR, 2020, pp. 8808–8817.

[5] Y. Tian, Y. Wang, D. Krishnan, J. B. Tenenbaum, and P. Isola, “Rethinking few-shot image classification: a good embedding is all you need?” in ECCV. Springer, 2020, pp. 266–282.

[6] C. Xing, N. Rostamzadeh, B. Oreshkin, and P. O. O Pinheiro, “Adaptive cross-modal few-shot learning,” in NeurIPS, vol. 32, 2019.

[7] W. Chen, C. Si, Z. Zhang, L. Wang, Z. Wang, and T. Tan, “Semantic prompt for few-shot image recognition,” in CVPR, 2023, pp. 23 581–23 591.

[8] H. Zhang, J. Xu, S. Jiang, and Z. He, “Simple semanticaided few-shot learning,” in CVPR, 2024, pp. 28 588– 28 597.

[9] M. Liu, F. Wu, B. Li, Z. Lu et al., “Envisioning class entity reasoning by large language models for few-shot learning,” in AAAI, vol. 39, no. 18, 2025, pp. 18 906– 18 914.

[10] W. Li, X. Meng, Q. Wang, Z. Han, Z. Wu, and Y. Yin, “DVLA-RL: Dual-level vision-language alignment with reinforcement learning gating for few-shot learning,” in ICLR, 2026.

[11] Z. Zha, H. Tang, Y. Sun, and J. Tang, “Boosting few-shot fine-grained recognition with background suppression and foreground alignment,” TCSVT, vol. 33, no. 8, pp. 3947–3961, 2023.

[12] W. Li, Q. Wang, P. Zhao, and Y. Yin, “Knn transformer with pyramid prompts for few-shot learning,” in ACM MM, 2024, pp. 1082–1091.

[13] A. Radford, J. W. Kim, C. Hallacy et al., “Learning transferable visual models from natural language supervision,” in ICML. PMLR, 2021, pp. 8748–8763.

[14] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Learning to prompt for vision-language models,” IJCV, vol. 130, no. 9, pp. 2337–2348, 2022.

[15] R. Zhang, W. Zhang, R. Fang, P. Gao, K. Li, J. Dai, Y. Qiao, and H. Li, “Tip-adapter: Training-free adaption of clip for few-shot classification,” in ECCV. Springer, 2022, pp. 493–510.

[16] A. Karmanov, D. Guan, S. Lu, A. El Saddik, and E. Xing, “Efficient test-time adaptation of vision-language models,” in CVPR, 2024, pp. 14 162–14 171.

[17] A. Martins and R. Astudillo, “From softmax to sparsemax: A sparse model of attention and multi-label classification,” in ICML, 2016, pp. 1614–1623.

[18] R. J. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning,” Mach. Learn., vol. 8, pp. 229–256, 1992.

[19] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and

O. Klimov, “Proximal policy optimization algorithms,” arXiv:1707.06347, 2017.

[20] S. J. Rennie, E. Marcheret, Y. Mroueh, J. Ross, and V. Goel, “Self-critical sequence training for image captioning,” in CVPR, 2017.

[21] J. Foerster, G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson, “Counterfactual multi-agent policy gradients,” in AAAI, vol. 32, no. 1, 2018.

[22] M. Ren, E. Triantafillou, S. Ravi, J. Snell, K. Swersky et al., “Meta-learning for semi-supervised few-shot classification,” arXiv:1803.00676, 2018.

[23] K. Lee, S. Maji et al., “Meta-learning with differentiable convex optimization,” in CVPR, 2019, pp. 10 657–10 665.

[24] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The caltech-ucsd birds-200-2011 dataset,” California Institute of Technology, Tech. Rep., 2011.

[25] J. Krause, M. Stark, J. Deng, and L. Fei-Fei, “3d object representations for fine-grained categorization,” in ICCVW, 2013, pp. 554–561.

[26] A. Khosla, N. Jayadevaprakash, B. Yao, and F.-F. Li, “Novel dataset for fine-grained image categorization: Stanford dogs,” in CVPRW, vol. 2, no. 1, 2011.

[27] B. Zhou, A. Lapedriza et al., “Places: A 10 million image database for scene recognition,” TPAMI, vol. 40, no. 6, pp. 1452–1464, 2017.

[28] X. Wang, Y. Peng, L. Lu et al., “Chestx-ray8: Hospitalscale chest x-ray database and benchmarks on weaklysupervised classification and localization of common thorax diseases,” in CVPR, 2017, pp. 2097–2106.

[29] Z. Chen, L. Xie, J. Niu, X. Liu et al., “Visformer: The vision-friendly transformer,” in ICCV, 2021, pp. 589– 598.

[30] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang et al., “Qwen2.5-vl technical report,” arXiv:2502.13923, 2025.

[31] B. Dong, P. Zhou, S. Yan, and W. Zuo, “Selfpromoted supervision for few-shot transformer,” in ECCV. Springer, 2022, pp. 329–347.

[32] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv:1711.05101, 2017.

[33] ——, “Sgdr: Stochastic gradient descent with warm restarts,” arXiv:1608.03983, 2016.

[34] A. Li, W. Huang, X. Lan, J. Feng, Z. Li, and L. Wang, “Boosting few-shot learning with adaptive margin loss,” in CVPR, 2020, pp. 12 576–12 584.

[35] J. Xu and H. Le, “Generating representative samples for few-shot classification,” in CVPR, 2022, pp. 9003–9013.

[36] R. Zhang, X. Hu, B. Li, S. Huang, H. Deng, Y. Qiao, P. Gao, and H. Li, “Prompt, generate, then cache: Cascade of foundation models makes strong few-shot learners,” in CVPR, 2023, pp. 15 211–15 222.

[37] M. Hiller, R. Ma et al., “Rethinking generalization in few-shot classification,” NeurIPS, vol. 35, pp. 3582– 3595, 2022.

[38] S. Sun and H. Gao, “Meta-adam: An meta-learned adaptive optimizer with momentum for few-shot learning,” NeurIPS, vol. 36, pp. 65 441–65 455, 2023.

[39] Y. Du, Z. Xiao, S. Liao, and C. G. M. Snoek, “Protodiff:

Learning to learn prototypical networks by task-guided diffusion,” in NeurIPS, vol. 36, 2023, pp. 46 304–46 322.

[40] F. Hao et al., “Class-aware patch embedding adaptation for few-shot image classification,” in ICCV, 2023, pp. 18 905–18 915.

[41] S. Baik et al., “Learning to learn task-adaptive hyperparameters for few-shot learning,” TPAMI, vol. 46, no. 3, pp. 1441–1454, 2024.

[42] H.-J. Ye, L. Ming, D.-C. Zhan, and W.-L. Chao, “Fewshot learning with a strong teacher,” TPAMI, vol. 46, no. 3, pp. 1425–1440, 2024.

[43] Y. Du, H. Sun, X. Zhen, J. Xu, Y. Yin, L. Shao, and C. G. M. Snoek, “MetaKernel: Learning variational random features with limited labels,” TPAMI, vol. 46, no. 3, pp. 1464–1478, 2024.

[44] M.-H. Pan, H.-Y. Xin, and H.-B. Shen, “Semantic-based implicit feature transform for few-shot classification,” IJCV, vol. 132, no. 11, pp. 5014–5029, 2024.

[45] Y. Guo, R. Du, A. Sain, K. Liang et al., “Understanding episode hardness in few-shot learning,” TPAMI, vol. 47, no. 1, pp. 616–633, 2025.

[46] S. Lee, W. Moon, H. S. Seong, and J.-P. Heo, “Taskoriented channel attention for fine-grained few-shot classification,” TPAMI, vol. 47, no. 3, pp. 1448–1463, 2025.

[47] Y. Li and W. Li, “Few-shot fine-grained classification with foreground-aware kernelized feature reconstruction network,” TIP, vol. 35, pp. 150–165, 2026.

[48] Z. Ji, R. Wei, J. Liu, Y. Pang, and J. Han, “Interpretable few-shot image classification via prototypical conceptguided mixture of LoRA experts,” TIP, vol. 35, pp. 930– 942, 2026.

[49] D. Wertheimer, L. Tang, and B. Hariharan, “Few-shot classification with feature map reconstruction networks,” in CVPR, 2021, pp. 8012–8021.

[50] S.-L. Xu, F. Zhang, X.-S. Wei, and J. Wang, “Dual attention networks for few-shot fine-grained recognition,” in AAAI, vol. 36, no. 3, 2022, pp. 2911–2919.

[51] Y. Yu, D. Zhang, and Z. Ji, “Masked feature generation network for few-shot learning.” in IJCAI, 2022, pp. 3695–3701.

[52] S. Lee, W. Moon, and J.-P. Heo, “Task discrepancy maximization for fine-grained few-shot classification,” in CVPR, 2022, pp. 5331–5340.

[53] L.-J. Zhao, Z.-D. Chen, Z.-X. Ma, X. Luo, and X.-S. Xu, “Angular isotonic loss guided multi-layer integration for few-shot fine-grained image classification,” TIP, vol. 33, pp. 3778–3792, 2024.

[54] Z.-X. Ma, Z.-D. Chen et al., “Cross-layer and crosssample feature optimization network for few-shot finegrained image classification,” in AAAI, vol. 38, no. 5, 2024, pp. 4136–4144.

[55] Z.-X. Ma et al., “Few-shot fine-grained image classification with progressively feature refinement and continuous relationship modeling,” in AAAI, vol. 39, no. 6, 2025, pp. 6036–6044.

[56] V. Garcia and J. Bruna, “Few-shot learning with graph neural networks,” in ICLR, 2018.

[57] H.-Y. Tseng, H.-Y. Lee, J.-B. Huang, and M.-H.

Yang, “Cross-domain few-shot classification via learned feature-wise transformation,” in ICLR, 2020.

[58] Y. Hu and A. J. Ma, “Adversarial feature augmentation for cross-domain few-shot classification,” in ECCV. Springer, 2022, pp. 20–37.

[59] J. Oh, S. Kim, N. Ho, J.-H. Kim, H. Song, and S.-Y. Yun, “Understanding cross-domain few-shot learning based on domain similarity and few-shot difficulty,” NeurIPS, vol. 35, pp. 2622–2636, 2022.

[60] H. Wang, H. Mai, Y. Gong, and Z.-H. Deng, “Towards well-generalizing meta-learning via adversarial task augmentation,” AIJ, vol. 317, p. 103875, 2023.

[61] F. Zhou, P. Wang, L. Zhang, W. Wei, and Y. Zhang, “Revisiting prototypical network for cross domain fewshot learning,” in CVPR, 2023, pp. 20 061–20 070.

[62] Y. Fu, Y. Xie, Y. Fu, and Y.-G. Jiang, “Styleadv: Meta style adversarial training for cross-domain few-shot learning,” in CVPR, 2023, pp. 24 575–24 584.

[63] T. Zhang, Q. Cai, F. Gao, L. Qi, and J. Dong, “Exploring cross-domain few-shot classification via frequency-aware prompting,” IJCAI, 2024.

[64] Y. Zou, Y. Liu et al., “Flatten long-range loss landscapes for cross-domain few-shot learning,” in CVPR, 2024, pp. 23 575–23 584.

[65] F. Zhou, P. Wang, L. Zhang, Z. Chen, W. Wei, C. Ding, G. Lin, and Y. Zhang, “Meta-exploiting frequency prior for cross-domain few-shot learning,” NeurIPS, vol. 37, pp. 116 783–116 814, 2024.

[66] W. Li, P. Fang, and H. Xue, “Svasp: Self-versatility adversarial style perturbation for cross-domain few-shot learning,” in AAAI, vol. 39, no. 15, 2025, pp. 15 275– 15 283.

[67] F. Zhou, P. Wang, L. Zhang, W. Wei, C. Ding, G. Lin, and Y. Zhang, “Meta-exploiting complementary semantic consistency for cross-domain few-shot learning promotion,” TIP, vol. 35, pp. 8543–8557, 2026.

[68] L. v. d. Maaten and G. Hinton, “Visualizing data using t-sne,” JMLR, vol. 9, no. Nov, pp. 2579–2605, 2008.

## APPENDIX A

## COMPUTATIONAL DETAILS AND THEORETICAL ANALYSIS

This appendix specifies the implementation interfaces and proves the results used in the main text. The analysis concerns the defined estimator and policy objective, and statistical assumptions about semantic score quality are stated separately from algebraic properties of the allocator.

## A. Feature, Semantic, and Policy Interfaces

At layer $\ell ,$ image tokens belong to $\mathbb { R } ^ { d _ { \ell } }$ and text embeddings belong to $\mathbb { R } ^ { d _ { t } }$ . The image-position count $M _ { \ell }$ excludes temporarily injected semantic tokens. Let $P _ { \ell } : \mathbb { R } ^ { d _ { \ell } }  \mathbb { R } ^ { d _ { t } }$ be a scoring projection and $G : \mathbb { R } ^ { d _ { t } }  \mathbb { R } ^ { d _ { f } }$ the anchor projection into the final feature space. Define

$$
\begin{array} { r l r } {  { v _ { \ell , c , k , i } = \mathrm { u n i t } ( P _ { \ell } Z _ { \ell , c , k , i } ) , } } \\ & { } & { z _ { c , k , i } = \mathrm { u n i t } ( Z _ { f , c , k , i } ) , } \\ & { } & { r _ { c } = \mathrm { u n i t } ( G \sum _ { h } { \omega _ { f , h } } \frac { 1 } { R _ { c } ^ { + , h } } \sum _ { r } { e _ { c } ^ { + , h , r } } ) . } \end{array}\tag{A1}
$$

Here $h \in \{ \mathrm { l o c } , \mathrm { g l o b } \}$ , the nonnegative level weights sum to one, and uni $\mathbf { \boldsymbol { \cdot } } ( \boldsymbol { v } ) = \boldsymbol { v } / \| \boldsymbol { v } \| _ { 2 }$ for nonzero v. An exactly zero vector is mapped to a fixed unit coordinate vector of the appropriate dimension. This deterministic numerical fallback is counted in the diagnostics. The query feature is the unitnormalized mean of its final image tokens, in the same $d _ { f ^ { - } }$ dimensional space as z and $^ { r } \cdot$

The local-to-global weight schedule is fixed using validation categories. At layer ℓ, the nonnegative weights $\omega _ { \ell , h }$ sum to one. In the main text, $\omega _ { h }$ denotes the weights of the layer being scored. The overlap maximum ranges over the unions of local and global phrases in each role, so the flattened indices $r , t$ in Eq. (6) include both levels. We use $\tau _ { b } , \tau _ { s } , \tau _ { c } > 0 , 0 \leq \lambda \leq 1$ $0 < \varepsilon _ { n } < 1$ , and $0 < \epsilon _ { p } < 1$ . For an explicit attention input, concatenate the positive embedding rows across classes within each level as $\bar { E _ { h } ^ { + } } \in \mathbb { R } ^ { R _ { h } \times d _ { t } }$ . A learned map $A _ { \ell } \in \mathbb { R } ^ { d _ { t } \times d _ { \ell } }$ gives

$$
T _ { \ell } = [ \omega _ { \ell , \mathrm { l o c } } E _ { \mathrm { l o c } } ^ { + } A _ { \ell } ; \omega _ { \ell , \mathrm { g l o b } } E _ { \mathrm { g l o b } } ^ { + } A _ { \ell } ] .\tag{A2}
$$

The attention projections in Eq. (2) map $d _ { \ell }$ dimensions to $d _ { \ell }$ dimensions, and attention logits are divided by $\sqrt { d _ { \ell } }$ . These projections and $A _ { \ell }$ are representation parameters, which classification updates train and policy updates freeze. The scorer maps $P _ { \ell }$ remain separate and frozen after calibration.

Scoring uses the same weights for the two evidence roles. Missing positive phrases are replaced by the encoded classname template in that level, and the response of a missing nuisance level is zero. If both nuisance levels are missing, set the overlap and rejection margin to zero. These conventions make every formula well defined.

The generator is prompted to return concise visible object properties in the intrinsic field and visible scene context, cooccurring objects, occlusions, or acquisition cues in the nuisance field. It is allowed to leave a nuisance field empty. Each role’s global description summarizes its own local phrases, and phrases are never reassigned on the basis of a query prediction. Exact duplicate phrases are removed within each field. A cache entry is identified by the support identifiers, class names, generation seed, prompt, generator version, and text-encoder version. Neither query images nor query labels are part of its construction.

During warm-up, the scoring maps are calibrated using base-support labels. Warm-up fits the visual and anchor parameters with classification loss and calibrates $P _ { \ell } .$ . After warmup, the scoring maps are frozen. Subsequent representation updates affect the encoder and anchor map through classification, and policy updates freeze all of them. A fixed scoring map prevents direct optimization of the probe to lower its own cost, and independent audits evaluate the final representations and policies used for prediction.

The gate state uses permutation-invariant pooling over labeled supports and candidate classes. It contains the pooled visual state, positive and nuisance semantic summaries, mean signed evidence, mean retained and neutral masses, and a layer indicator. A finite-dimensional policy consumes this summary, whereas the history $\mathcal { H } _ { \ell }$ in the proofs includes the complete support history and preceding actions, without assuming that the summary itself is Markov. For $0 < \epsilon _ { g } < 1 / 2$ , a smooth mean parameterization is

$$
p _ { \theta } = \epsilon _ { g } + ( 1 - 2 \epsilon _ { g } ) \mathrm { s i g m o i d } ( f _ { \theta } ) .\tag{A3}
$$

The concentration $\kappa _ { g }$ is positive and fixed. The frozen reference gate receives the support-pooled visual and intrinsic semantic state in the base policy’s original input dimensions. Its added nuisance-state coordinates are not used. The full conference model remains a separate experimental baseline, and this reference is its gate embedded in the current recognition pipeline.

For every selected layer, incoming supports determine the action, followed by the image-token transition in $\operatorname { E q . }$ (3). The cost in Eq. (14) is evaluated on the resulting post-action tokens using the scoring map for that output space. A query subsequently replays the support-derived action schedule and uses all candidate classes’ intrinsic tokens. Thus an unknown query class is never needed as an input, and processing one query cannot alter another query’s prediction.

## B. Projection, Rejection, and Sensitivity

Set $b = ( d - \xi { \bf 1 } ) / \tau _ { s }$ and $\mathcal { A } = \left\{ a \in \mathbb { R } ^ { M } : a \geq 0 , \mathbf { 1 } ^ { \top } a \leq \right\}$ 1}. This is a nonempty closed convex set, so its Euclidean projection is unique. The Lagrangian of Eq. (7) is

$$
\begin{array} { r } { \mathcal { F } ( \boldsymbol { a } , \delta , \zeta ) = \frac 1 2 \| \boldsymbol { a } - \boldsymbol { b } \| _ { 2 } ^ { 2 } + \delta ( \mathbf { 1 } ^ { \top } \boldsymbol { a } - 1 ) - \zeta ^ { \top } \boldsymbol { a } , \quad \delta \geq 0 , \quad \zeta \geq 0 . } \end{array}\tag{A4}
$$

Stationarity and complementary slackness give $a _ { i } = [ b _ { i } - \delta ] _ { + }$ and $\begin{array} { r } { \delta ( \sum _ { i } a _ { i } - 1 ) = 0 . \mathrm { I f } \sum _ { i } [ b _ { i } ] _ { + } \leq 1 } \end{array}$ , the multiplier is zero. Otherwise it is the unique threshold yielding unit mass, found by sorting $b _ { i } .$ . Consequently,

$$
a _ { i } > 0 \implies d _ { i } > \xi , ~ a _ { i } [ \xi - d _ { i } ] _ { + } = 0 .\tag{A5}
$$

Zero-score ties have zero allocation. If all scores are below the margin, $a = 0$ , and Eq. (9) gives $w = r$ exactly. No division by the retained mass is required, even when it approaches zero.

The candidate deficit is distinct from retained contamination. Its value can be positive when all nuisance tokens have been rejected, and it can be small when a misgrounded bank assigns nuisance tokens large intrinsic scores. Thus candidate deficit, neutral mass, and externally audited nuisance retention address different questions, and the last equality above shows that selection-weighted deficits cannot serve as an independent audit.

Projection also gives a deterministic sensitivity statement. For two score vectors and margins,

$$
\| a - a ^ { \prime } \| _ { 2 } \leq \frac { 1 } { \tau _ { s } } \| ( d - d ^ { \prime } ) - ( \xi - \xi ^ { \prime } ) { \bf 1 } \| _ { 2 } .\tag{A6}
$$

This follows from nonexpansiveness of Euclidean projection. For fixed visual tokens, write $Z = [ z _ { 1 } , \dots , z _ { M } ]$ . Allowing the anchor to change as well,

$$
\| w - w ^ { \prime } \| _ { 2 } \leq \lambda \| Z - r \mathbf { 1 } ^ { \top } \| _ { \mathrm { o p } } \| a - a ^ { \prime } \| _ { 2 } + \| r - r ^ { \prime } \| _ { 2 } .\tag{A7}
$$

Indeed, subtract the two definitions and write the difference as $\lambda ( Z - r \mathbf { 1 } ^ { \top } ) ( a - a ^ { \prime } ) + ( 1 - \lambda \rho ^ { \prime } ) ( r - r ^ { \prime } )$ . Its second coefficient lies in [0, 1]. This statement concerns perturbations of scoring and anchoring for fixed image tokens.

## C. Retained Nuisance Mass and Prototype Error

Fix a class and condition on its semantic banks. Let the final support token indices be partitioned into intrinsic indices I and nuisance indices ${ \mathcal { N } } ,$ with $M _ { N } = | \mathcal { N } |$ . This partition is used for analysis and independent audits, not provided to the selector. Define $\textstyle { \boldsymbol { \kappa } } = \sum _ { i \in { \mathcal { I } } } a _ { i }$ and $\textstyle \nu = \sum _ { i \in { \mathcal { N } } } a _ { i }$ , so $\rho = \kappa { + } \nu \leq 1$ . For a nuisance token, assume its final signed score can be written as $d _ { i } = m _ { i } + \epsilon _ { i }$ with $m _ { i } \le - \Delta < 0$ and

$$
\mathbb { E } [ \exp ( t \epsilon _ { i } ) \mid { \mathcal T } , i \in \mathcal { N } ] \le \exp ( t ^ { 2 } \sigma ^ { 2 } / 2 ) , \qquad t \in \mathbb { R } .\tag{A8}
$$

The conditioning includes everything determining the rejection margin. The assumption concerns final scores under the evaluated policy, not scores from a different warm-up representation. No independence among tokens is assumed.

Proof of the retained-mass bound. Writing $h = \Delta { + } \xi > 0 .$ the KKT solution implies $a _ { i } \leq [ d _ { i } - \xi ] _ { + } / \tau _ { s }$ . Hence, for $\sigma > 0 .$

$$
\begin{array} { r l r } {  { \mathbb { E } a _ { i } \leq \frac { 1 } { \tau _ { s } } \int _ { 0 } ^ { \infty } \operatorname* { P r } ( d _ { i } - \xi > t ) d t } } \\ & { } & { \leq \frac { 1 } { \tau _ { s } } \int _ { h } ^ { \infty } \exp [ - u ^ { 2 } / ( 2 \sigma ^ { 2 } ) ] d u } \\ & { } & { \leq \frac { \sigma ^ { 2 } } { \tau _ { s } h } \exp [ - h ^ { 2 } / ( 2 \sigma ^ { 2 } ) ] . } \end{array}\tag{A9}
$$

The last inequality uses $1 \leq u / h$ on the integration interval. Summing over nuisance indices and using $\nu ~ \leq ~ 1$ proves Eq. (20). If $\sigma = 0 { \mathrm { . } }$ , every nuisance score is below the margin and $\nu = 0 .$ . A related familywise bound is

$$
\operatorname* { P r } ( \nu > 0 \mid \mathcal { T } ) \le \operatorname* { m i n } \{ 1 , M _ { N } \exp [ - h ^ { 2 } / ( 2 \sigma ^ { 2 } ) ] \} .\tag{A10}
$$

This follows by a union bound. The retained-mass result additionally controls the amount allocated to erroneous tokens, which the familywise probability alone does not quantify. □

Proof of the prototype bound. Let $\mu _ { c }$ be a unit ideal class direction. Suppose intrinsic token errors, nuisance token errors, and anchor errors are bounded by $B _ { I } , B _ { N } , B _ { A }$ , respectively. The coefficients in the raw support estimator are nonnegative and sum to one. Subtracting $\mu _ { c }$ therefore yields

$$
w _ { c , k } - \mu _ { c } = \lambda \sum _ { i } a _ { c , k , i } ( z _ { c , k , i } - \mu _ { c } ) + ( 1 - \lambda \rho _ { c , k } ) ( r _ { c } - \mu _ { c } ) .\tag{A11}
$$

The triangle inequality and averaging over supports prove $\operatorname { E q . } \ ( 1 9 )$ . The bound is tighter than the anchor-only upper bound $B _ { A }$ if

$$
\frac { 1 } { K } \sum _ { k } ( \kappa _ { c , k } B _ { I } + \nu _ { c , k } B _ { N } ) < \frac { 1 } { K } \sum _ { k } \rho _ { c , k } B _ { A } ,\tag{A12}
$$

provided $\lambda > 0$ . Comparing upper bounds does not establish that every realized prototype improves on its realized anchor. If $E _ { c } ~ < ~ 1 - \varepsilon _ { n } ,$ the reverse triangle inequality implies $\| \bar { w } _ { c } \| _ { 2 } \ge 1 - E _ { c } > \varepsilon _ { n } ,$ so the fallback is inactive. Using $\| \mu _ { c } \| _ { 2 } = 1$ gives

$$
\begin{array} { r l } & { \| p _ { c } - \mu _ { c } \| _ { 2 } \leq \| p _ { c } - \bar { w } _ { c } \| _ { 2 } + \| \bar { w } _ { c } - \mu _ { c } \| _ { 2 } } \\ & { \qquad = | 1 - \| \bar { w } _ { c } \| _ { 2 } | + \| \bar { w } _ { c } - \mu _ { c } \| _ { 2 } \leq 2 E _ { c } . } \end{array}\tag{A13}
$$

If the fallback is active, $p _ { c } ~ = ~ r _ { c }$ and its error is instead bounded by $B _ { A }$ . Both cases are well defined, including

complete rejection. For a fixed unit query $q ,$ comparison with any competing class gives

$$
q ^ { \top } ( p _ { y } - p _ { c } ) \geq q ^ { \top } ( \mu _ { y } - \mu _ { c } ) - 2 ( E _ { y } + E _ { c } ) .\tag{A14}
$$

When the normalization condition holds for every class, the margin condition in the main text preserves the ideal label. This establishes the effect of prototype estimation for the representation actually used, and changes in q under a different gate schedule are treated separately. □

These results identify where empirical grounding is needed. An inaccurate intrinsic anchor increases $B _ { A }$ , and incorrectly assigned nuisance phrases can violate the score-gap assumption. Larger rejection margins reduce the nuisance tail bound but can increase anchor reliance. The method’s evaluation therefore reports these quantities together rather than using rejection rate alone as evidence of improved recognition.

## D. Paired Policy Gradients and Clipping

Let ζ contain the episode, frozen semantic banks, and exogenous augmentation randomness. During a policy update, the encoder, scoring maps, anchor map, reference gate, and value baseline are fixed. The current policy parameters appear only in the action densities, not directly in the reward or feature transition at a fixed action sequence. The deterministic reference trajectory is a function of $\zeta$ and its own previous states, so $R _ { 0 }$ and $\bar { \mathbf { a } ^ { 0 } }$ are independent of sampled policy actions conditional on ζ.

Proof of Proposition 1. The conditional trajectory density is

$$
p _ { \theta } ( \mathbf { a } \mid \zeta ) = \prod _ { \ell = 1 } ^ { L } \pi _ { \theta } ( a _ { \ell } \mid \mathcal { H } _ { \ell } ) .\tag{A15}
$$

Assume differentiation can be interchanged with integration, for example under a locally dominating integrable scoreweighted reward. The score-function identity gives

$$
\nabla _ { \theta } J = \mathbb { E } \left[ R \sum _ { \ell } \nabla _ { \theta } \log \pi _ { \theta } ( a _ { \ell } \mid \mathcal { H } _ { \ell } ) \right] .\tag{A16}
$$

Conditioned on the complete history and $\zeta ,$ the expectation of each policy score is zero. Since $R _ { 0 }$ is already fixed under this conditioning, and $b _ { \omega } ( \mathcal { H } _ { \ell } )$ is independent of the current action,

$$
\begin{array} { r } { \mathbb { E } [ ( R _ { 0 } + b _ { \omega } ( \mathcal { H } _ { \ell } ) ) \nabla _ { \theta } \log \pi _ { \theta } ( a _ { \ell } \mid \mathcal { H } _ { \ell } ) ] = 0 . } \end{array}\tag{A17}
$$

Subtracting these terms proves $\mathbb { E } \widehat { g } = \nabla _ { \theta } J$ . Dependence of later states on earlier sampled actions is represented by the trajectory density. It is not an additional pathwise derivative through the frozen environment. The action-deviation cost is part of R and is therefore included in the same identity. □

Detaching a tensor is an implementation operation, whereas conditional action independence is a statistical condition. For example, if a Bernoulli action has reward $R = a ,$ choosing a detached but action-dependent baseline $\boldsymbol { R _ { 0 } } ~ = ~ \boldsymbol { a }$ makes every advantage zero although the expected reward has a nonzero policy gradient. The separate reference rollout avoids this issue. Updating shared encoder parameters through the policy surrogate would introduce further direct dependencies not covered by this proof, which is why the optimization schedule explicitly separates those parameters.

Reward variance and gradient variance. For any paired returns,

$$
\operatorname { V a r } ( R - R _ { 0 } ) = \operatorname { V a r } ( R ) + \operatorname { V a r } ( R _ { 0 } ) - 2 \operatorname { C o v } ( R , R _ { 0 } ) .\tag{A18}
$$

Positive covariance improves this expression relative to an independently sampled reference with the same marginals. It does not automatically improve it relative to using R alone, or control score-weighted gradient variance. For a concrete example, take $a \sim \mathrm { B e r n o u l l i } ( 0 . 1 )$ and an independent equiprobable $X ~ \in ~ \{ - 1 , 1 \}$ . Let $R \ : = \ : ( 1 \ : - \ : 2 a ) X$ and $R _ { 0 } \ = \ X$ . Then $\mathrm { C o v } ( R , R _ { 0 } ) = 0 . 8$ and the reward variance decreases from 1 to 0.4, whereas the logit-score gradient variance increases from 0.09 to 0.324. Thus the diagnostics in Fig. 8 measure gradient variance directly rather than substituting paired reward variance for it. These values are computed analytically for this example.

Clipped surrogate deviation. Define the unclipped importance surrogate on the fixed behavior batch as $\begin{array} { r l } { J _ { \mathrm { I S } } } & { { } = } \end{array}$ $\mathbb { E } _ { \mathrm { o l d } } \sum _ { \ell } r _ { \ell } A _ { \ell }$ and let $J _ { \mathrm { c l i p } } = - \mathcal { L } _ { \mathrm { C F - P P O } }$ . Away from clipping boundaries, set

$$
\mathcal { T } _ { \ell } = \mathbf { 1 } \{ A _ { \ell } > 0 , ~ r _ { \ell } > 1 + \epsilon _ { p } \} + \mathbf { 1 } \{ A _ { \ell } < 0 , ~ r _ { \ell } < 1 - \epsilon _ { p } \} .\tag{A19}
$$

The derivative of a likelihood ratio is $r _ { \ell } \nabla$ log $\pi _ { \theta }$ , while the active clipped branch is locally constant. Consequently,

$$
\| \nabla J _ { \mathrm { I S } } - \nabla J _ { \mathrm { c l i p } } \| _ { 2 } \leq \mathbb { E } _ { \mathrm { o l d } } \sum _ { \ell } r _ { \ell } | A _ { \ell } | \ \| \nabla \log \pi _ { \theta } ( a _ { \ell } \mid \mathcal { H } _ { \ell } ) \| _ { 2 } \mathcal { L } _ { \ell } .\tag{A20}
$$

The bound follows by subtracting derivatives and applying the triangle inequality, assuming integrability. It uses no uniform bound on Beta score gradients. At $\theta = \theta _ { \mathrm { o l d } }$ , every ratio equals one and the surrogate gradient has the on-policy expectation proved above. After policy updates, the per-layer importance surrogate does not correct the changed state-visitation distribution and need not equal the gradient of J. Neither this result nor clipping alone establishes monotonic improvement.

For an explicit clipping counterexample, consider one sampled contribution with $A = 1 , r = 1 . 2 1$ , clipping radius $\epsilon _ { p } =$ 0.2, and scalar score $\nabla _ { \theta } \log \pi _ { \theta } = 1$ . The unclipped derivative is 1.21 and the active clipped derivative is zero. Their difference is therefore 1.21, whereas $| r - \mathrm { c l i p } ( r , 0 . 8 , 1 . 2 ) | = 0 . 0 1$ This local configuration can occur in a valid two-action policy. $\operatorname { A t } \theta = 0 ,$ , let $p _ { \theta } ( a = 1 ) = 0 . 2 4 2 \exp ( \theta )$ in a sufficiently small neighborhood and let $p _ { \mathrm { o l d } } ( a = 1 ) = 0 . 2$ . The other action has advantage zero. The expected gradient gap is then 0.242, while an excess-ratio bound weighted by the same behavior probability would give 0.002. Thus the excess-ratio expression cannot bound the clipping-gradient error.

## E. Training Algorithm

We describe the entire training procedure of DVLA-RL++ in detail in Algorithm 1. The training process follows an episodic paradigm and alternates between policy and representation updates after a warm-up with the reference gate. It incorporates two key modules: (1) Complementary Semantic

Algorithm 1: Training algorithm of the proposed   
DVLA-RL++.   
Input: Base classes $\mathcal { C } _ { \mathrm { b a s e } } ,$ representation $F _ { \phi }$ with   
anchor map G, frozen text encoder and scoring   
projections $P _ { \ell } .$ frozen reference gate $\pi _ { 0 } ,$ , gate   
policy $\pi _ { \theta } ,$ , and value baseline $b _ { \omega }$   
1. Warm up ϕ with the reference gate and calibrate $P _ { \ell }$   
using base-support labels, then freeze $P _ { \ell } ;$   
while not converged do   
2. Sample an N-way K-shot episode {S, Q} from   
$\mathcal { C } _ { \mathrm { b a s e } }$ according to Eq. (1);   
3. Generate or retrieve the intrinsic and nuisance   
banks $\mathcal { E } _ { c } ^ { \pm }$ using support identifiers only;   
4. Compute the reference schedule $ { \mathbf { g } } ^ { 0 }$ from the   
support states of $\pi _ { 0 }$ according to Eq. (13), and   
obtain $R _ { 0 }$ according to Eq. (15);   
for each layer $\ell = 1 , \ldots , L$ do   
5. Sample $g _ { \ell } \sim \pi _ { \theta } ( \cdot \mid \mathcal { H } _ { \ell } )$ according to   
Eq. (12), and fuse the visual and semantic   
branches according to Eqs. (2) and (3);   
6. Record the log density, the state summary,   
and the post-action evidence deficit;   
7. Compute the signed evidence, rejection margins,   
and token allocations according to Eqs. (5)–(7);   
8. Construct the purified prototypes $p _ { c }$ according to   
Eqs. (9) and (10), and encode the queries with the   
recorded schedule;   
9. Compute the nuisance cost $C ( \mathbf { g } )$ , the reward   
$R ( \mathbf { g } )$ , and the stopped advantages $A _ { \ell }$ according   
to Eqs. (14) and (15);   
10. Update θ with L <sub>-</sub> according to Eq. (18),   
keeping the sampled actions and advantages fixed;   
11. Fit $b _ { \omega }$ to the stopped paired returns according   
to Eq. (16) for subsequent policy batches;   
12. Draw a fresh episode, recompute the support   
states, gate actions, and prototypes with the   
current policy, and stop gradients through the   
actions;   
for each query image $x _ { j }$ in Q do   
13. Compute the prediction $P _ { \phi } ( y _ { j } \mid x _ { j } , S )$ and   
the cross-entropy loss according to Eq. (4);   
14. Update ϕ with ${ \mathcal L } _ { \mathrm { c l s } }$ and discard the stored   
policy trajectories;   
Output: The checkpoint selected by the sequential   
mean policy on base-validation episodes,   
together with the cached mean gate schedule   
and purified prototypes for testing.

Purification (CSP), which contrasts intrinsic and nuisance semantics to allocate support tokens and anchors each support prototype with intrinsic class semantics; and (2) Counterfactual Reinforcement-Learning Gating (CRG), which samples layer-wise fusion gates and compares each sampled trajectory with an independent reference rollout under a nuisance-aware reward. During policy updates, the representation, scoring projections, text encoder, reference gate, and value baseline remain fixed. During representation updates, the gate actions are stopped and only ϕ is optimized with the classification loss. These modules jointly construct purified prototypes that retain intrinsic object evidence and suppress incidental context.

At inference, only the current mean policy is executed, and the support state is updated after every mean action. The resulting gate schedule and class prototypes are cached once per episode, and queries are encoded individually. The reference gate and the value baseline are unnecessary at test time.