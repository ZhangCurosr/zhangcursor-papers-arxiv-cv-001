# Hub-Spectral Activation of Latent Multimodal Knowledge

Ying Guo<sup>∗</sup>, Haidong Chen<sup>∗</sup>, Linrui Xu, Xiaohao Liu, Chuancheng Shi, Canran Xiao, Dan Zhang, Fei Shen, Senior Member, IEEE, Li Shen, Senior Member, IEEE, and Tat-Seng Chua

Abstract—Multimodal representation learning seeks shared representations for cross-modal retrieval and knowledge transfer. Hub-based binding reduces pairwise supervision costs, but separate hub connections cannot guarantee reliable alignment between modalities without direct joint training. We introduce Hub-Spectral Activation (HSA), a closed-form method for recovering and activating the hub-readable component of latent multimodal knowledge in frozen representations. We formalize this knowledge as source-induced cross-modal dependence and characterize the component determined by the second-order statistics of two trained hub edges. Under a second-order source model, we establish conditions for exact recovery of the complete source-induced relation and bound the dimension of its hub-readable component by the hub covariance rank. HSA composes and standardizes hub-edge statistics, extracts paired spectral directions, and combines reliability-weighted matching evidence with source-gated candidate resolution for bidirectional retrieval and prototype classification. HSA requires no targetpair supervision, gradient optimization, or backbone updates. Across 19 retrieval and 11 prototype-classification relations on ImageBind and LanguageBind, HSA raises mean bidirectional Recall@10 from 18.27% to 31.15% and mean macro Top-1 accuracy from 29.01% to 52.43%, respectively. Controlled analyses further identify valid hub-edge correspondence and leading spectral directions as key sources of retrieval gains, demonstrating the utility of latent multimodal knowledge beyond native similarity scores. Code and models are publicly available at https://github.com/Luo1Yan/HSA.

Index Terms—Cross-modal alignment, cross-modal classification, cross-modal retrieval, latent multimodal knowledge, multimodal binding, multimodal representation learning.

![](images/626921396dc1492259d6b246a6e086e632ce4855e5dc2b4542d97accbd4be32a.jpg)  
Fig. 1. From Multimodal Training to Hub-Spectral Activation. (a) Targetpair training learns one relation from paired data. (b) Hub-based binding reduces pairwise supervision but leaves held-out retrieval uneven. (c) HSA reads the relation supported by two trained hub connections and produces retrieval rankings. HSA improves held-out retrieval in closed form, with frozen encoders and without positive target-pair examples or gradient optimization.

## I. INTRODUCTION

The same object or event [1], [2], [3], [4] can be observed through images, language, audio, depth, thermal signals, and motion. Multimodal representation learning organizes these heterogeneous observations so that information acquired in one modality can support matching and transfer in another [5], [6], [7]. Joint objectives learn such relations from paired or co-occurring observations [8], [9], [10], [11], [12], [13]. Direct target-pair training can provide precise alignment for a chosen relation, as illustrated in Fig. 1(a) [14], [15]. Extending this strategy to every modality pair requires paired data and objectives for a number of relations that grows quadratically with the number of modalities.

Hub-based binding models reduce this supervision burden by aligning each modality with a shared hub. ImageBind uses image to connect text, audio, depth, thermal, and inertial modalities [16]; LanguageBind uses language to connect video, infrared, depth, audio, and image [17]. Text-hub binding has also been extended to medical imaging modalities [18]. We call two non-hub modalities connected only through separately trained hub edges a held-out pair. The indirect path can support transfer, yet native retrieval remains highly uneven, as illustrated in Fig. 1(b). This variability motivates a specific question: after binding-model training is complete, what relation between a held-out pair can be identified and activated using only its two trained hub connections?

We answer this question with Hub-Spectral Activation (HSA), shown in Fig. 1(c). We define latent multimodal knowledge as source-induced dependence preserved in frozen representations and identify the hub-readable component determined by the second-order statistics of the two trained hub edges. HSA composes their moment systems, standardizes the resulting relation, and identifies paired spectral carriers that couple informative directions across the target spaces. It maps samples into these low-dimensional knowledge coordinates and combines reliability-weighted carrier agreement with source-gated candidate resolution. HSA thereby turns the recovered hub-readable relation into a basis for comparing candidates, supporting both bidirectional retrieval and prototype classification. Our analysis establishes when the hub-readable component equals the complete source-induced relation and further bounds its dimension by the rank of the hub representation. Every fitted quantity is obtained in closed form, without positive target-pair examples, pair identities, gradient optimization, or backbone updates.

Existing approaches connect indirect modalities through target-pair correspondences, common anchors, overlapping modalities, unpaired target marginals, learned proxies, or target-encoder adaptation [19], [20], [21], [22], [23], [24]. Related frozen-encoder methods still optimize lightweight maps from paired target data [25], [26]. HSA operates in a fully frozen post-training setting, using the two hub-edge datasets to construct and calibrate the held-out relation readout. Across nineteen ImageBind and LanguageBind retrieval relations, it raises mean Recall@10 from 18.27% to 31.15%, compared with 31.49% for fixed-duration supervised target-pair training. Across the eleven prototype-classification relations, HSA estimates the target-modality relation and constructs its readout using the classification source features, raising mean macro Top-1 accuracy from 29.01% to 52.43%. Hub-edge shuffling, disjoint-source estimation, and carrier controls jointly show that HSA uses valid sample correspondence within the two hub edges to recover the hub-readable relation, with leading paired spectral carriers providing its main retrieval gain.

Before delving into details, we summarize our contributions as follows:

• We formalize latent multimodal knowledge as sourceinduced dependence, identify the component readable through two hub edges, and give its exact-recovery condition. A rank bound further quantifies the capacity available through the hub.

• We develop HSA, a closed-form readout that recovers the target-modality relation from two hub edges, extracts paired spectral carriers, and uses the resulting relation for bidirectional retrieval and prototype classification. Relation fitting uses no positive target-pair examples, gradient optimization, or backbone updates.

• Across nineteen retrieval relations and eleven classification relations from ImageBind and LanguageBind, HSA improves every reported relation-level point estimate over frozen cosine. Correspondence, disjoint-source, and spectral controls link the retrieval gains to valid hub-edge statistics and the leading paired carriers.

## II. RELATED WORK

## A. Building Unified Multimodal Spaces

Unified multimodal spaces are learned with complementary objectives. Canonical-correlation methods learn shared coordinates in classical, multiset, and deep settings [27], [28], [29], [30]. Other approaches use reconstruction [31], joint generative modeling [32], or pairwise correspondence and contrastive alignment [8], [9], [10], [11], [12], [13], [14], [15]. Vision-language pretraining provides an extensively studied setting for these objectives [33]. ViLBERT, LXMERT, and UNITER learn joint image-text representations through transformer-based interaction and paired pretraining [34], [35], [36]. FLAVA combines contrastive alignment with multimodal fusion [37]; ALBEF aligns image and text representations before cross-modal attention, while BLIP learns understanding and generation from filtered and synthetic captions [38], [39]. Frozen visual encoders can also support trainable cross-modal connections: LiT tunes a text encoder against a locked image encoder, and BLIP-2 trains a querying transformer between frozen image and language models [40], [41]. Multimodal conditioning also supports image-and-pose-guided generation in IMAGPose [42] and garment-conditioned synthesis in IMAGDressing-v1 [43].

When modalities rarely co-occur, binding methods exploit shared structure across separately collected datasets [44]. ImageBind uses image as the hub for text, audio, depth, thermal, and IMU, while LanguageBind uses language to connect video, infrared, depth, audio, and image [16], [17]. Medical binding similarly connects text, imaging, and physiological signals [18], [45], [46]. UniBind learns modality-agnostic alignment centers, and ULIP places point clouds, images, and language in one representation space [47], [48]. PMRL uses a rank-1 Gram objective for anchor-free simultaneous alignment; CalMRL handles missing observations through anchor-shift analysis and representation-level imputation [49], [50]. These methods construct or adapt a unified space. Their objectives determine what cross-modal structure is encoded and how it is organized. HSA operates after training and reads the hub-readable component of a held-out relation from two frozen hub edges.

## B. Connecting Indirectly Related Modalities

Methods that connect indirectly related modalities differ in the information available for alignment. ReAlign uses training-free Anchor, Trace, and Centroid Alignment to map text embeddings toward the image distribution from large unpaired image and text marginals [23]. ASIF forms a coupled dictionary from paired target examples [19]. C-MCR and Ex-MCR connect independently trained spaces through an overlapping modality [21], [22], whereas FreeBind and OmniBind fuse pretrained expert spaces [51], [52]. COX couples known and unseen modalities without instance-level pairs by learning variational bottlenecks and emergent correspondences [53]. TextME projects unseen modalities into an LLM embedding space from text descriptions [54]. Related frozen-encoder methods learn lightweight maps from limited paired supervision [25], [26]. Together, these approaches use target pairs, overlapping modalities, text, unpaired marginals, or learned correspondences. Their supervision and modelupdate requirements therefore vary across settings.

EmergentBridge strengthens weak emergent transfer by learning a proxy generator from an anchor to an already aligned modality [24]. It then adapts the target-modality encoder with anchor-paired data and an orthogonal-subspace proxy objective, while keeping the anchor encoder frozen. This procedure uses no direct pairs between the evaluated modalities, but it requires proxy training and gradient-based target-encoder adaptation. HSA keeps the hub and both target encoders frozen. Its fit uses only the existing A–H and H–B hub-edge datasets and excludes A–B identities, target losses, gradient optimization, and target-informed rank selection.

## C. Linear Recovery and Representation Geometry

Analyses of frozen representations examine the structure retained across models and modalities [55], [56], [57], [58], [59]. Representational similarity analysis, canonicalcorrelation comparisons, centered-kernel alignment, and statistical tests compare internal organization [60], [61], [62], [63], [64], [65], [66]. Modality-gap and cross-encoder studies show that semantic structure can coexist with modality-dependent separation [64], [67]. The Platonic Representation Hypothesis and subsequent matching work study compatible relational geometry across independently learned representations [68], [69]. Alignment also depends on modality similarity and the balance of redundant and unique information [70]. Relative representations expose shared organization by expressing samples through similarities to common anchors [20]. These studies characterize cross-modal compatibility; HSA constructs a held-out retrieval rule from two frozen hub edges.

Classical linear methods offer transparent tools for crossspace recovery. CCA finds correlated coordinates from paired two-view observations, with multiset extensions for more than two views [27], [28]. Ridge regression stabilizes linear prediction under correlated coordinates [71], and Orthogonal Procrustes estimates an optimal orthogonal map from paired correspondences [72]. Latent Space Translation derives a closed-form transformation from parallel semantic anchors that denote the same high-level concept in both spaces [73]. We use these constructions as protocol references. Bidirectional ridge [71] and Procrustes [72] map each target independently into the hub space, while Paired Orthogonal Procrustes (Paired-OP) [72] solves one orthogonal map per retrieval direction from paired A–B rows. Building on these classical linear tools, HSA studies which component of latent multimodal knowledge can be identified from the second-order statistics of two trained hub connections. It characterizes the recovery limits of this component and uses it for held-out retrieval and prototype classification, connecting representation analysis with cross-modal transfer without direct target-pair supervision or backbone updates.

## III. PRELIMINARY

Held-Out Hub Setting. A real-world object, scene, or event may give rise to several modality-specific observations that retain dependence through their common source. We consider two target modalities A and B that are connected during training only through a hub modality H. Let $O _ { i }$ denote the common source and let $\mathbf { L } _ { i } = \ell ( O _ { i } ) \in \mathbb { R } ^ { q }$ collect the factors shared across its observations. For each $m \in \{ A , H , B \} , X _ { i } ^ { m }$ is a partial observation of $O _ { i } ,$ , and a frozen encoder maps it into the common d-dimensional binding space:

$$
\begin{array} { r l } & { O _ { i } \sim \mathcal { P } _ { O } , \qquad X _ { i } ^ { m } \mid O _ { i } \sim p _ { m } ( \cdot \mid O _ { i } ) , \qquad \mathbf { L } _ { i } = \ell ( O _ { i } ) , } \\ & { \qquad \mathbf { a } _ { i } = \operatorname { n r m } ( f _ { A } ( X _ { i } ^ { A } ) ) , \qquad \mathbf { h } _ { i } = \operatorname { n r m } ( f _ { H } ( X _ { i } ^ { H } ) ) , } \\ & { \qquad \mathbf { b } _ { i } = \operatorname { n r m } ( f _ { B } ( X _ { i } ^ { B } ) ) . } \end{array}\tag{1}
$$

Although fitting receives no direct A–B pairs, the target representations can retain a source-induced relation through $\mathbf { L } _ { i } .$ . We formalize this relation next.

Definition 1 (Source-induced latent multimodal knowledge): For square-integrable frozen representations A and B from a common source, latent multimodal knowledge is the crosscovariance of their source-conditioned means:

$$
\mathcal { K } _ { A B } : = \operatorname { C o v } ( \mathbb { E } [ \mathbf { A } \mid \mathbf { L } ] , \mathbb { E } [ \mathbf { B } \mid \mathbf { L } ] ) .\tag{2}
$$

It collects the dependence carried by shared source factors in the two frozen target spaces.

Available Information. During fitting, HSA receives only two paired hub-edge datasets,

$$
\mathcal { D } _ { A H } = \{ ( { \bf a } _ { i } , { \bf h } _ { i } ^ { A H } ) \} _ { i = 1 } ^ { n _ { A H } } , \qquad \mathcal { D } _ { H B } = \{ ( { \bf h } _ { j } ^ { H B } , { \bf b } _ { j } ) \} _ { j = 1 } ^ { n _ { H B } } .
$$

HSA estimates the two edge-moment systems separately, without requiring cross-edge row correspondence. The edge datasets may therefore use disjoint source instances. When synchronized observations are available, one source row may instead contribute a pair to each dataset, so the two hub representations can coincide. HSA summarizes each hub edge through its moments, including $\Sigma _ { A H }$ and $\Sigma _ { H B }$ , without forming $\Sigma _ { A B }$ or an A–B training objective. At population level, the available information is

$$
\begin{array} { c } { { { \mathfrak { S } } _ { H } = \{ \pmb { \mu } _ { A } , \pmb { \mu } _ { H } , \pmb { \mu } _ { B } , \pmb { \Sigma } _ { A A } , \pmb { \Sigma } _ { A H } , } } \\ { { \Sigma _ { H H } , \Sigma _ { H B } , \Sigma _ { B B } \} . } } \end{array}\tag{3}
$$

Its empirical counterpart, $\widehat { \mathfrak { S } } _ { H }$ , supplies the second-order statistics for estimating the hub-readable relation. HSA uses the two hub-edge datasets for rank selection and score calibration. Relation fitting excludes the empirical A–B cross-covariance, an A–B training loss, target labels, and target-informed rank selection. The statistical task is to identify the component of ${ { \kappa } _ { A B } }$ fixed by ${ \mathfrak { S } } _ { H }$ and convert it into a score for matching new sample pairs.

Second-Order Source Model. To connect the shared factors, frozen representations, and observed hub edges, we use a second-order source model. The model permits target-specific residual dependence while requiring both hub edges to reflect the shared factors consistently. Intuitively, the shared factors L link the two observed hub edges, while A and B may retain residual dependence beyond HSA’s stated recovery scope.

Assumption 1 (Second-order consistency across the two hub edges): All variables have finite second moments, 1 $\begin{array} { r l r } { \vec { \bf \ " } } & { { } \mathbf { L } } & { = \quad \mathbf { 0 } , } \end{array}$ , and $\begin{array} { l l l } { \displaystyle \mathrm { C o v } ( { \bf L } ) } & { = } & { { I _ { q } } . } \end{array}$ For each $( m , \mathbf { M } ) \in$ $\{ ( A , \mathbf { A } ) , ( H , \mathbf { H } ) , ( B , \mathbf { B } ) \} , \ \mathbb { E } [ \mathbf { M } \ | \ \textbf { L } ] \ = \ \pmb { \mu _ { m } } + G _ { m } \mathbf { L } ,$ , and $\boldsymbol { \varepsilon } _ { m } : = \mathbf { M } - \mathbb { E } [ \mathbf { M } \mid \mathbf { L } ]$ satisfies $\mathbb { E } [ \pmb { \varepsilon } _ { m } \mid \mathbf { L } ] = \mathbf { 0 }$ . The observed hub edges satisfy $\mathrm { C o v } ( \pmb { \varepsilon } _ { A } , \pmb { \varepsilon } _ { H } ) = 0$ and $\mathrm { C o v } ( \pmb { \varepsilon } _ { H } , \pmb { \varepsilon } _ { B } ) = 0$ Their samples provide compatible estimates of the population means and the second-order statistics of the hub.

![](images/d09b990f25b6ed098cc6737b2336f67451a9a5b2f6c520f06dfa6c495b49db3a.jpg)  
Fig. 2. Overview of Hub-Spectral Activation. Frozen representations from the observed $A { - } H$ and $H { - } B$ hub edges provide the moment systems used by HSA. Stages ①–③ construct and standardize the hub-readable relation $\widehat { \cal K } _ { A B } ^ { H } .$ , identify paired carriers $\Phi _ { A }$ and $\Phi _ { B } ,$ and project candidates into readout coordinates $\bar { \mathbf { z } ^ { A } }$ and $\mathbf { z } ^ { B }$ . Their strength-aware agreement defines the carrier score $s _ { C } .$ , which is calibrated and combined with gated candidate resolution to form s for retrieval and prototype classification.

Assumption 1 yields

$$
\begin{array} { r l r } & { \mathbf { A } = \pmb { \mu } _ { A } + G _ { A } \mathbf { L } + \boldsymbol { \varepsilon } _ { A } , \quad \quad } & { \mathbf { H } = \pmb { \mu } _ { H } + G _ { H } \mathbf { L } + \boldsymbol { \varepsilon } _ { H } , } \\ & { \mathbf { B } = \pmb { \mu } _ { B } + G _ { B } \mathbf { L } + \boldsymbol { \varepsilon } _ { B } , \quad \quad \mathcal { K } _ { A B } = G _ { A } G _ { B } ^ { \top } . } \end{array}\tag{4}
$$

The population target cross-covariance may also contain target-residual dependence: $\Sigma _ { A B } ~ = ~ { \cal K } _ { A B } + \Omega _ { A B }$ , where $\Omega _ { A B } : = \mathrm { C o v } ( \pmb { \varepsilon } _ { A } , \pmb { \varepsilon } _ { B } )$ is unrestricted. This separation ties the recovery target to the common source. The derivation of this decomposition and the exact information interface supplied by the two hub edges are detailed in the appendix.

## IV. HUB-SPECTRAL ACTIVATION

HSA proceeds from identification to task readout, as summarized in Fig. 2. It first isolates the relation determined by the two observed hub edges, identifies paired spectral carriers that preserve this relation, converts their sample-wise evidence into a calibrated score, and applies that score to retrieval rankings or class prototypes. During fitting, HSA estimates the carrier matrices from sample statistics of the two hub-edge datasets. During task readout, these matrices remain fixed as each input sample is mapped to a carrier coordinate vector for query– candidate scoring.

## A. Hub-Readable Relation and Capacity

Hub-Readable Relation. The available information contains the two observed hub edges while omitting the target crossrelation. The following theorem identifies the component determined by these edges and states when that component recovers the complete relation induced by the common source.

Theorem 1 (Identifiable hub-readable latent knowledge): Under Assumption 1, let $\Psi _ { H } = \mathrm { C o v } ( \varepsilon _ { H } )$ and define

$$
\begin{array} { r l } & { \mathcal { P } _ { H } \ = G _ { H } ^ { \top } ( G _ { H } G _ { H } ^ { \top } + \Psi _ { H } ) ^ { \dagger } G _ { H } , \qquad 0 \preceq \mathcal { P } _ { H } \preceq I _ { q } , } \\ & { { \mathbf A } ^ { H } \ = \Sigma _ { A H } \Sigma _ { H H } ^ { \dagger } ( { \mathbf H } - \mu _ { H } ) , } \\ & { \underline { { \mathbf B } } ^ { H } \ = \Sigma _ { B H } \Sigma _ { H H } ^ { \dagger } ( { \mathbf H } - \mu _ { H } ) , } \\ & { \underline { { \mathcal { K } _ { A B } ^ { H } } } \Big | : = \Sigma _ { A H } \Sigma _ { H H } ^ { \dagger } \Sigma _ { H B } , } \\ & { \qquad = \mathrm { C o v } ( { \mathbf A } ^ { H } , { \mathbf B } ^ { H } ) = G _ { A } \mathcal { P } _ { H } G _ { B } ^ { \top } . } \end{array}\tag{5}
$$

Then

$$
\mathcal { K } _ { A B } = \mathcal { K } _ { A B } ^ { H } + G _ { A } ( I _ { q } - \mathcal { P } _ { H } ) G _ { B } ^ { \top } ,\tag{6}
$$

and exact hub readout holds if and only if $G _ { A } ( I _ { q } - \mathcal { P } _ { H } ) G _ { B } ^ { \top } =$ 0. The component ${ \cal { K } } _ { A B } ^ { H }$ is identified by ${ \mathfrak { S } } _ { H }$ , while the same interface generally underidentifies ${ \kappa } _ { A B }$

The matrix product in Eq. (5) is the covariance of the two best linear hub predictions, giving it a classical regression interpretation. Theorem 1 specifies its role under the restricted information interface: $\mathcal { P } _ { H }$ measures how much of the shared source state is linearly resolved by the frozen hub, and ${ \cal { K } } _ { A B } ^ { H }$ transports that resolved component into the two target spaces. Eq. (6) fixes the claim boundary by separating this identified component from the generally underidentified remainder of ${ { \kappa } _ { A B } }$ . The theorem’s proof and an explicit non-identifiability construction are provided in the appendix.

Hub Capacity. The same result limits how many independent relation channels the hub can expose.

Corollary 1 (Hub-capacity bound): The hub-readable relation satisfies

$$
\operatorname { r a n k } ( { K } _ { A B } ^ { H } ) \leq \operatorname { r a n k } ( \Sigma _ { H H } ) .\tag{7}
$$

If the hub representation has at most $N _ { H }$ distinct population states, then ran $\mathrm { k } ( \mathcal { K } _ { A B } ^ { H } ) \le N _ { H } - 1$

## B. Spectral Carrier Identification

① Standardized Hub Operator. HSA next estimates and standardizes this operator. Applying the standard leadingsingular-subspace variational principle [74] yields paired lowdimensional directions that preserve the greatest relation strength at a fixed dimension. Let $\widehat { \Sigma } _ { H H } ^ { A }$ and $\widehat { \Sigma } _ { H H } ^ { B }$ denote the hub covariances estimated on the two edges. HSA pools them with equal edge weight as $\begin{array} { r } { \overline { { \Sigma } } _ { H H _ { \bullet } } = \frac { 1 } { 2 } ( \widehat { \Sigma } _ { H H } ^ { A } + \widehat { \Sigma } _ { H H } ^ { B } ) } \end{array}$ . It uses the trace-scaled ridges $\rho _ { X } = \lambda \mathrm { t r } ( \widehat { \Sigma } _ { X X } ) / d _ { X }$ for $X \in \{ A , B \}$ and $\rho _ { H } = \lambda \mathrm { t r } ( \overline { { \Sigma } } _ { H H } ) / d _ { H }$ . The empirical hub-readable relation and its standardized decomposition are

$$
\begin{array} { r l } & { \widehat { \mathcal { K } } _ { A B } ^ { H } = \widehat { \Sigma } _ { A H } \big ( \overline { { \Sigma } } _ { H H } + \rho _ { H } I \big ) ^ { - 1 } \widehat { \Sigma } _ { H B } , } \\ & { \quad T _ { H } : = W _ { A } \widehat { \mathcal { K } } _ { A B } ^ { H } W _ { B } ^ { \top } = U _ { C } \mathrm { d i a g } \big ( \sigma _ { 1 } , \ldots , \sigma _ { r } \big ) V _ { C } ^ { \top } , } \\ & { \quad W _ { X } = ( \widehat { \Sigma } _ { X X } + \rho _ { X } I ) ^ { - 1 / 2 } . } \end{array}\tag{8}
$$

A rounded-state empirical proxy motivated by Corollary 1 sets a heuristic cap on the carrier count $k _ { H }$ . Details of this rule and its numerical safeguards are provided in the appendix.

## ② Optimal Paired Carriers.

Proposition 1 (Optimal paired spectral carriers): For any $1 \ \leq \ k \ \leq \ r .$ , among all k-dimensional orthonormal paired projections in the standardized target spaces,

$$
\operatorname* { m a x } _ { P ^ { \top } P = I _ { k } } \mathrm { t r } ( P ^ { \top } T _ { H } Q ) = \sum _ { j = 1 } ^ { k } \sigma _ { j } .\tag{9}
$$

The maximum is attained by $P = U _ { C , k }$ and $Q \ = \ V _ { C , k }$ Mapping these directions to the original target coordinates as $\Phi _ { A } ^ { ( k ) } { = } \bar { W } _ { A } U _ { C , k }$ and $\Phi _ { B } ^ { ( k ) } = W _ { B } V _ { C , k }$ gives

$$
( \Phi _ { A } ^ { ( k ) } ) ^ { \top } \widehat { \mathcal { K } } _ { A B } ^ { H } \Phi _ { B } ^ { ( k ) } = \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { k } ) .\tag{10}
$$

③ Carrier Coordinates. Proposition 1 turns the recovered dense operator into ordered coordinate-wise evidence: the chosen pair of subspaces maximizes retained hub-readable relation strength, and each resulting channel has the known strength $\sigma _ { j }$ . HSA sets $\boldsymbol { k } ~ = ~ k _ { H }$ , writes $\Phi _ { A } \ = \ \Phi _ { A } ^ { ( k _ { H } ) }$ and $\Phi _ { B } = \Phi _ { B } ^ { ( k _ { H } ) }$ , and defines the projected knowledge coordinates

$$
\mathbf { z } ^ { A } ( \mathbf { a } ) = \boldsymbol \Phi _ { A } ^ { \top } ( \mathbf { a } - \widehat { \pmb \mu } _ { A } ) , \qquad \mathbf { z } ^ { B } ( \mathbf { b } ) = \boldsymbol \Phi _ { B } ^ { \top } ( \mathbf { b } - \widehat { \pmb \mu } _ { B } ) .\tag{11}
$$

Each of the $k _ { H }$ columns of $\Phi _ { A }$ represents a selected carrier direction. The component $z _ { j } ^ { A } ( \mathbf { a } )$ is the inner product of $\mathbf { a } - \widehat { \pmb { \mu } } _ { A }$ with column $j ,$ for $j = 1 , \dotsc , k _ { H }$ . These components form the coordinate vector $\mathbf { z } ^ { A } ( \mathbf { a } )$ , with the same construction applying to modality B. Together, these steps perform spectral carrier localization and readout: the paired singular directions locate the hub-readable relation, while the projected coordinates expose it as sample-wise evidence. The retained singular triplets also give the best rank- $- k _ { H }$ Frobenius approximation of $T _ { H }$ The proof of the proposition and derivation of the sample score below are provided in the appendix.

## C. HSA Scoring and Task Readouts

Carrier Evidence. The paired carriers convert the hubreadable relation into aligned coordinates. HSA scores a candidate pair by asking whether these coordinates are more likely under a matched model than under a mismatched model. Set $c _ { j } = \mathrm { c l i p } ( \sigma _ { j } , 0 , 1 - 1 0 ^ { - 6 } )$ . For channel $j ,$ HSA models a matched coordinate pair as a standardized bivariate Gaussian with correlation $c _ { j }$ and a mismatched pair as independent standardized coordinates. Let

Algorithm 1 HSA: Identify, Calibrate, and Read Out   
Require: Hub-edge data, task readout sets, and registered HSA   
settings   
Ensure: Retrieval rankings or prototype-classification labels   
1: Identify: estimate $\widehat { \mathfrak { S } } _ { H }$ from the two hub edges as in Eq. (3).   
2: Compute $\widehat { \mathcal { K } } _ { A B } ^ { H } , W _ { A } , W _ { B } ,$ , and $T _ { H }$ using Eq. (8).   
3: Select k<sub>H</sub> by the hub-capacity rule (see the appendix) and obtain   
$\Phi _ { A } , \Phi _ { B }$ from Proposition 1.   
4: Define carrier evidence $s _ { C }$ using Eq. (12).   
5: Select k<sub>R</sub> by the shuffled-edge null (see the appendix) and define   
s<sub>R</sub> using Eq. (13).   
6: Calibrate: estimate $\widehat { \Omega } _ { A } , \widehat { \Omega } _ { B }$ and compute $g _ { R }$ using Eq. (14).   
7: Use five fixed derangements of the equal-count target marginals   
in $\mathcal { D } _ { A H } , \mathcal { D } _ { H B }$ to compute $\tau _ { C } , \tau _ { R }$ using Eq. (15a).   
8: Read Out: map $\mathcal { A } _ { \mathrm { r o } } , \hat { \boldsymbol B } _ { \mathrm { r o } }$ using Eq. (11) and evaluate all pairs   
with Eq. (15b).   
9: For retrieval, form score matrices and bidirectional rankings using   
Eqs. (16)–(17).   
10: For classification, treat $\scriptstyle A _ { \mathrm { r o } }$ as the prototype bank and predict   
labels using Eq. (18).   
11: return The task-specific rankings or predicted labels

$$
\ell _ { j } ( x , y ; c _ { j } ) : = \log \frac { p ( [ x , y ] ^ { \top } \mid Y = 1 ) } { p ( [ x , y ] ^ { \top } \mid Y = 0 ) }
$$

be the coordinate-wise log-likelihood ratio. The spectral carrier score is

$$
s _ { C } ( { \bf a } , { \bf b } ) = \sum _ { j = 1 } ^ { k _ { H } } \ell _ { j } \left( z _ { j } ^ { A } ( { \bf a } ) , z _ { j } ^ { B } ( { \bf b } ) ; c _ { j } \right) .\tag{12}
$$

The singular value therefore controls how strongly each carrier coordinate contributes to the same-source evidence. The closed-form expansion is provided in the appendix, where the Gaussian working model is confined to these standardized coordinates.

Candidate Resolution. Several gallery items can share similar carrier coordinates, so HSA distinguishes them in the orthogonal complement of the leading raw relation subspace. A fixed shuffled-edge null determines the subspace rank $k _ { R } ;$ details of the selection rule and numerical safeguards are provided in the appendix. The normalized vectors $\mathbf { r } ^ { A }$ and $\mathbf { r } ^ { B }$ provide candidate-resolution coordinates, while $s _ { R }$ measures fine-grained query-candidate compatibility after removing that subspace. Let $\tilde { \mathcal { K } } _ { A B } ^ { H } ~ = ~ U _ { R } \mathrm { d i a g } \mathrm { ( } \eta ) V _ { R } ^ { \top }$ , form the rank- $\displaystyle - k _ { R }$ projectors $\Pi _ { A } ^ { R }$ and $\Pi _ { B } ^ { R }$ , and define

$$
\begin{array} { r l } & { \mathbf { r } ^ { A } ( \mathbf { a } ) = \mathrm { n r m } \big ( ( I - \Pi _ { A } ^ { R } ) ( \mathbf { a } - \widehat { \pmb { \mu } } _ { A } ) \big ) , } \\ & { \mathbf { r } ^ { B } ( \mathbf { b } ) = \mathrm { n r m } \big ( ( I - \Pi _ { B } ^ { R } ) ( \mathbf { b } - \widehat { \pmb { \mu } } _ { B } ) \big ) , } \\ & { s _ { R } ( \mathbf { a } , \mathbf { b } ) = \mathbf { r } ^ { A } ( \mathbf { a } ) ^ { \top } \mathbf { r } ^ { B } ( \mathbf { b } ) . } \end{array}\tag{13}
$$

Reliability and Calibration. The candidate-resolution term should be used only when residual geometry is organized consistently across the two target spaces. HSA measures this reliability using only the observed hub edges. Let $\widehat { \Omega } _ { A }$ and $\widehat { \Omega } _ { B }$ be the ridge-regularized covariance matrices. Applying the same feature permutation $P _ { \pi }$ to the rows and columns of $\widehat { \Omega } _ { B }$ preserves its spectrum but breaks its coordinate correspondence with $\widehat { \Omega } _ { A } .$ . The excess of the observed affinity over this permutation null defines a reliability gate:

$$
\begin{array} { r l } & { \alpha _ { R } = \frac { \langle \widehat \Omega _ { A } , \widehat \Omega _ { B } \rangle _ { F } } { \| \widehat \Omega _ { A } \| _ { F } \| \widehat \Omega _ { B } \| _ { F } } , } \\ & { q _ { R } = \mathrm { Q u a n t i l e } _ { 0 . 9 5 } \left( \left\{ \frac { \langle \widehat \Omega _ { A } , P _ { \pi } \widehat \Omega _ { B } P _ { \pi } ^ { \top } \rangle _ { F } } { \| \widehat \Omega _ { A } \| _ { F } \| \widehat \Omega _ { B } \| _ { F } } \right\} _ { \pi \in \Pi _ { G } } \right) , } \\ & { g _ { R } = \mathrm { c l i p } \left( \frac { \alpha _ { R } - q _ { R } } { 1 - q _ { R } } , 0 , 1 \right) . } \end{array}\tag{14}
$$

The resulting gate uses only these hub-edge covariance matrices and leaves the carrier evidence unchanged. The edgespecific estimators, null construction, and zero-denominator rule are detailed in the appendix. In all reported runs, the two target-side training marginals have equal row counts: synchronized source rows supply equal-count marginals in the main protocol, and the disjoint-source control uses matched equal-count splits. Five fixed derangements of these marginals estimate mismatch scales $\tau _ { C }$ and $\tau _ { R } ,$ placing the carrier and candidate-resolution scores on comparable scales. For $Q \in \{ C , R \}$

$$
\begin{array} { l } { \displaystyle \tau _ { Q } = \frac { 1 } { 5 } \sum _ { \pi \in \Pi _ { \mathrm { c a l } } } } \\ { \displaystyle \operatorname* { m a x } \bigr \{ \mathrm { S t d } _ { i } \bigl [ s _ { Q } ( \mathbf { a } _ { i } , \mathbf { b } _ { \pi ( i ) } ) \bigr ] , 1 0 ^ { - 6 } \bigr \} , } \\ { \displaystyle s _ { \mathrm { H S A } } ( \mathbf { a } , \mathbf { b } ) = \frac { s _ { C } ( \mathbf { a } , \mathbf { b } ) } { \tau _ { C } } + g _ { R } \frac { s _ { R } ( \mathbf { a } , \mathbf { b } ) } { \tau _ { R } } . } \end{array}\tag{15a}
$$

(15b)

Bidirectional Ranking. The calibrated score defines one match matrix: its rows rank B candidates for A queries, and its columns rank A candidates for B queries.

$$
\begin{array} { r } { [ S ^ { A  B } ] _ { i j } = s _ { \mathrm { H S A } } ( { \bf a } _ { i } , { \bf b } _ { j } ) , } \\ { S ^ { B  A } = ( S ^ { A  B } ) ^ { \top } . \qquad } \end{array}\tag{16}
$$

$$
\begin{array} { r } { \pi _ { i } ^ { A  B } = \mathrm { a r g s o r t } _ { j } ^ { \downarrow } [ S ^ { A  B } ] _ { i j } , } \\ { \pi _ { j } ^ { B  A } = \mathrm { a r g s o r t } _ { i } ^ { \downarrow } [ S ^ { A  B } ] _ { i j } . } \end{array}\tag{17}
$$

Prototype Classification. The same fixed pair score supports class prediction. Let $\mathbf { p } _ { c } ^ { A }$ be a normalized class prototype in modality A, formed from labeled A-side training features or from a fixed bank of class prompts when text provides the prototype modality. A test query b from modality B is classified by

$$
{ \widehat { y } } ( \mathbf { b } ) = { \underset { c \in { \mathcal { C } } } { \operatorname { a r g m a x } } } s _ { \mathrm { H S A } } ( \mathbf { p } _ { c } ^ { A } , \mathbf { b } ) .\tag{18}
$$

Within each relation, all compared methods use the same prototype bank. Class identities determine this bank; target test outcomes and A–B pair identities do not enter HSA fitting or calibration. All directions, ranks, weights, and scales are fixed before target evaluation. When encoder coordinates are compatible, the relation readout for the same target-modality pair can support both retrieval and prototype classification with all fitted parameters fixed. The main classification comparisons estimate the relation and construct its readout using the specified source features. Further validation of both task readouts under a single fitted state is provided in the appendix. Algorithm 1 summarizes the two readouts; details of the scoreblock implementation are provided in the appendix.

## D. Further Analysis

Knowledge, Recoverability, and Visibility. Definition 1 and Theorem 1 separate the full source-induced relation ${ { \kappa } _ { A B } }$ from the hub-readable component $\mathcal { K } _ { A B } ^ { H }$ . Recoverability depends on which source factors the hub resolves, while native visibility further depends on how the paired directions of ${ \kappa _ { A B } ^ { H } }$ align with the cosine axes. Nonzero singular values can therefore coexist with weak native retrieval when the left and right directions couple different axes.

Why Spectral Activation Exposes the Relation. HSA resolves the coordinate mismatch before scoring. Proposition 1 selects paired singular directions that retain maximal relation strength at a fixed dimension. Eq. (10) represents the relation as diagonal channels with strengths $\sigma _ { j }$ , while Eq. (12) converts channel-wise agreement into strength-aware sample evidence. Spectral activation thereby maps cross-coordinate coupling into matched carrier channels for direct scoring.

Testable Consequences. Three structural predictions follow. Shuffling either hub edge should weaken the relation; gains should persist with disjoint source instances, up to finitesample variation. Preserving cross-coordinate hub correlations should improve readout. Leading carriers should outperform subsequent or random carriers when counts and assigned reliability spectra are matched. The scoring rule predicts gains from adding informative carriers and increasing their reliability until estimation noise dominates. Gating and candidate resolution should help when residual geometry is reliable. Section V-C evaluates these predictions through correspondence, disjointsource, spectral, scaling, and component-ablation experiments. Scope. These claims are limited to second-order relations that can be linearly resolved through the frozen hub. The identifiable component ${ \cal { K } } _ { A B } ^ { H }$ may cover only part of $\kappa _ { A B } ,$ and the Gaussian model applies only to the standardized carrier coordinates. Relation formation during training remains outside the present scope.

## V. EXPERIMENTS

The experiments address five questions:

Q1 Effect: Does HSA improve held-out retrieval across relations, backbones, and directions, while also improving prototype classification across relations and backbones?

Q2 Source: Given the retrieval effect, does it require valid correspondence within both hub edges and persist without shared source instances?

Q3 Location: Given its source, which hub-covariance structure and spectral carriers contain the usable relation, and how concentrated is it?

Q4 Activation: Given the localized carriers, how do reliability, gating, and candidate resolution turn them into ranking gains?

Q5 Cost: Once the readout is specified, what fitting, retrieval-scoring, and classification-readout overhead does it introduce?

Each answer supplies the premise for the next; the following settings define their common protocol.

TABLE I  
HELD-OUT RETRIEVAL ON NINE IMAGEBIND RELATIONS. BOLDFACE AND UNDERLINING MARK THE BEST AND SECOND-BEST VALUES AMONG ALL METHODS EXCEPT FULL A–B. <sup>†</sup> MARKS METHODS THAT USE TARGET-PAIR IDENTITIES; BLUE PARENTHESES SHOW HSA GAINS OVER FROZEN COSINE.
<table><tr><td>Method</td><td colspan="10">Evaluation Relations: Bidirectional Recall@10 (%) ↑</td><td rowspan="2">Mean R@1 R@5</td><td rowspan="2">Mean</td><td rowspan="2">Mean R@10</td><td rowspan="2">R@10 Recovery</td></tr><tr><td></td><td>Tx-Au VGS</td><td>Tx-Th TR</td><td>Tx-IMU E4D</td><td>Au-D BV</td><td>Au-Th MAVD</td><td>Au-IMU E4D</td><td>D-Th TR</td><td>D-IMU UTD</td><td>Th-IMU CA</td></tr><tr><td>Full A-B†</td><td>84.57</td><td>74.71</td><td>4.01</td><td>7.13</td><td>3.09</td><td>7.15</td><td>15.76</td><td>6.01</td><td>4.15</td><td>16.28</td><td>20.34</td><td>22.95</td><td>100.0</td></tr><tr><td>Frozen Cosine</td><td>75.20</td><td>7.68</td><td>1.53</td><td>1.97</td><td>1.62</td><td>1.75</td><td>0.34</td><td>3.02</td><td>4.08</td><td>7.65</td><td>9.80</td><td>10.80</td><td>47.0</td></tr><tr><td>Hub-Relative [20]</td><td>36.85</td><td>12.67</td><td>1.46</td><td>1.88</td><td>2.20</td><td>2.26</td><td>0.27</td><td>2.56</td><td>4.08</td><td>2.88</td><td>4.70</td><td>7.14</td><td>31.1</td></tr><tr><td>Bi. Ridge [71]</td><td>82.05</td><td>70.52</td><td>6.85</td><td>6.85</td><td>4.63</td><td>6.57</td><td>11.71</td><td>7.21</td><td>13.88</td><td>15.24</td><td>20.04</td><td>23.36</td><td>101.8</td></tr><tr><td>Bi. Procrustes [72]</td><td>83.30</td><td>61.19</td><td>4.74</td><td>5.65</td><td>2.43</td><td>6.88</td><td>4.35</td><td>3.14</td><td>8.16</td><td>12.91</td><td>17.39</td><td>19.98</td><td>87.1</td></tr><tr><td>ReAlign [23]</td><td>75.60</td><td>9.10</td><td>1.82</td><td>2.91</td><td>2.31</td><td>1.95</td><td>0.33</td><td>1.74</td><td>4.29</td><td>7.77</td><td>9.63</td><td>11.12</td><td>48.4</td></tr><tr><td>ERM [75]</td><td>79.28</td><td>71.46</td><td>3.67</td><td>5.91</td><td>3.74</td><td>7.26</td><td>12.84</td><td>5.47</td><td>4.63</td><td>14.97</td><td>18.99</td><td>21.58</td><td>94.0</td></tr><tr><td>IRM [76]</td><td>79.75</td><td>73.08</td><td>1.85</td><td>5.28</td><td>3.09</td><td>3.22</td><td>14.85</td><td>2.36</td><td>4.01</td><td>15.21</td><td>18.86</td><td>20.83</td><td>90.8</td></tr><tr><td>VREx [77]</td><td>81.18</td><td>68.57</td><td>4.64</td><td>5.91</td><td>2.70</td><td>5.24</td><td>12.97</td><td>6.94</td><td>4.76</td><td>14.43</td><td>18.98</td><td>21.43</td><td>93.4</td></tr><tr><td>DANN [78]</td><td>77.90</td><td>70.14</td><td>4.20</td><td>5.74</td><td>3.43</td><td>6.54</td><td>13.10</td><td>5.19</td><td>3.88</td><td>14.34</td><td>18.62</td><td>21.12</td><td>92.0</td></tr><tr><td>CORAL [79]</td><td>79.28</td><td>71.46</td><td>3.57</td><td>5.91</td><td>3.70</td><td>6.26</td><td>12.84</td><td>5.62</td><td>4.97</td><td>15.00</td><td>18.96</td><td>21.51</td><td>93.7</td></tr><tr><td>ASIF† [19]</td><td>83.40</td><td>66.47</td><td>4.96</td><td>6.76</td><td>3.59</td><td>5.24</td><td>14.15</td><td>3.26</td><td>4.29</td><td>14.02</td><td>19.25</td><td>21.35</td><td>93.0</td></tr><tr><td>Paired-OP† [72]</td><td>82.95</td><td>59.07</td><td>3.86</td><td>5.39</td><td>2.43</td><td>4.52</td><td>4.37</td><td>3.84</td><td>6.94</td><td>12.35</td><td>17.10</td><td>19.26</td><td>83.9</td></tr><tr><td>HSA (Ours)</td><td>84.45</td><td>74.87</td><td>8.97</td><td>6.93</td><td>4.63</td><td>8.93</td><td>9.66</td><td>9.30</td><td>16.33</td><td>16.39</td><td>21.53</td><td>24.90</td><td>108.5</td></tr><tr><td></td><td>(+9.25)</td><td>(+67.20)</td><td>(+7.43)</td><td>(+4.97)</td><td>(+3.01)</td><td>(+7.19)</td><td>(+9.32)</td><td>(+6.28)</td><td>(+12.24)</td><td>(+8.73)</td><td>(+11.73)</td><td>(+14.10)</td><td></td></tr></table>

## A. Experimental Settings

Figure and table abbreviations are Tx (text), Im (image), Vi (video), Au (audio), D (depth), Th (thermal), and IMU (inertial measurement).

Scope and Data. We evaluate frozen ImageBind [16] and LanguageBind [17] on 19 nondegenerate held-out relations spanning seven modalities and ten datasets: VGGSound [80], NYUv2 [81], TartanRGBT distributed with AnyThermal [82], Ego4D [83], BatVision [84], MAVD [85], UTD-MHAD [86], Caltech Aerial RGBT [87], UCF101 [88], and MSR-VTT [89]. An evaluation relation is a backbone–dataset–target-pair configuration. For each of the 19 retrieval relations, all methods share the same training and test partitions, test queries, and complete test gallery. The classification study reports eleven class-labeled relations, with six under ImageBind and five under LanguageBind. Class definitions, splits, and counts are detailed in the appendix.

Comparisons and Information Boundary. The main tables distinguish source-only analytic, adapted trainable, targetpaired, and HSA groups. Analytic references are frozen cosine, hub-relative similarity [20], bidirectional ridge [71], bidirectional Procrustes [72], and ReAlign [23], which applies Anchor, Trace, and Centroid Alignment to unpaired A and B training marginals. ERM [75], IRM [76], VREx [77], DANN [78], and CORAL [79] use the same two-layer head architecture with the A–H and H–B edges as training environments; respectively, they minimize mean edge loss, enforce invariant optimality, penalize risk variance, adversarially suppress modality identity, and align covariances with the Deep CORAL penalty. Target-paired references are ASIF [19], Paired-OP [72], and Full A–B: ASIF builds a coupled dictionary, Paired-OP solves two orthogonal maps, and Full A– B trains heads with the same architecture. All compared methods keep the backbone frozen. Methods without targetpair supervision exclude A–B identities and target-informed model selection. For classification, every method scores the same A-side class prototypes against B-side test queries. ASIF and Paired-OP take zero gradient steps; Full A–B follows the fixed training protocol below.

Metrics and Reporting. Bidirectional Recall@k averages both retrieval directions for k ∈ {1, 5, 10}; Recall@10 is primary. Recovery@10 divides each method’s backbone-mean Recall@10 by Full A–B. Classification uses B →A prototype prediction, and its primary metric is macro Top-1, the mean of class-wise Top-1 accuracies. Retrieval aggregates weight the 19 relations equally, while the classification aggregate weights the eleven reported relations equally. ERM, IRM, VREx, DANN, CORAL, and Full A–B average seeds 40–42. Full A– B uses a single 200-step duration selected globally from trainonly pilot splits and fixed across relations and seeds. Complete retrieval and classification protocols, including datasets, class definitions, and classification eligibility, are provided in the appendix. Further retrieval results, seed variability, aggregation checks, diagnostics, and the excluded degenerate unit are documented there.

Across-Dataset Statistical Analysis. We assess cross-dataset stability by grouping the 19 retrieval relations into ten datasets and averaging seeds within each relation. Relation-equal mean differences and 95% percentile intervals use 100,000 datasetcluster bootstrap draws (seed 42), retaining all relations of each sampled dataset. One-sided exact sign-flip tests act on entire dataset groups, with Holm correction within predefined comparison families. These tests assume independent dataset groups and joint sign symmetry. The appendix reports datasetequal estimates, leave-one-dataset-out ranges, complete families, and the interpretation of within-relation intervals.

## B. Does HSA Improve Held-Out Performance? (Q1)

Q1 evaluates retrieval and prototype classification across relations and backbones, including both retrieval directions. Full A–B is the supervised reference, excluded from boldface and underlining. Tables I and III compare all methods on the nineteen evaluation relations. ReAlign raises crossbackbone mean Recall@10 from 18.27% with frozen cosine to 18.48%. HSA reaches 31.15%, exceeds ReAlign on all nineteen relations by 12.67 points on average, improves every relation, and attains the highest or tied-highest Recall@10 among all methods except Full A–B on fifteen. Across the ten dataset groups, HSA exceeds frozen cosine by 12.88 points (95% cluster interval [5.29, 18.68]; exact $p = 0 . 0 0 0 9 8 )$ . All nine comparisons with the other source-only methods remain positive after family-wise Holm correction $( p _ { \mathrm { H o l m } } \leq 0 . 0 0 8 7 9 )$ . The same closed-form readout therefore operates across heldout relations formed through both image and language hubs. The gain remains positive under dataset-balanced weighting, every single-dataset exclusion, and chance normalization; complete definitions and results are provided in the appendix. ImageBind. HSA reaches 24.90% mean Recall@10, improving frozen cosine by 14.10 points and bidirectional ridge by 1.53 points. Its smaller margin over ridge than on LanguageBind is consistent with Q4, where candidate resolution and residual orthogonalization mainly refine LanguageBind residual geometry. Under the image hub, HSA improves all nine held-out relations, spanning the natively visible text– audio relation and the near-chance text–thermal relation, and achieves the highest mean Recall@1, Recall@5, and Recall@10 among all methods except Full A–B. The largest increase is 67.20 points for text–thermal. For ImageBind prototype classification, HSA reaches 44.01% mean macro Top-1 accuracy across the six relations in Table II. This is 27.70 points above frozen cosine and 3.94 points above bidirectional ridge. The gains cover text-, audio-, depth-, thermal-, and inertial-modality readouts across VGGSound, TartanRGBT, Ego4D, BatVision, and UTD-MHAD.

TABLE II  
IMAGEBIND PROTOTYPE CLASSIFICATION. MACRO TOP-1 (%) ON SIX RELATIONS. BOLDFACE AND UNDERLINING MARK THE BEST AND SECOND-BEST VALUES AMONG ALL METHODS EXCEPT FULL A–B. <sup>†</sup> MARKS METHODS THAT USE TARGET-PAIR IDENTITIES; BLUE PARENTHESES SHOW HSA GAINS OVER FROZEN COSINE.
<table><tr><td rowspan="2">Method</td><td colspan="6">Prototype-Query Relations: Macro Top-1 (%) ↑</td></tr><tr><td>Tx-Au VGS</td><td>Tx-Th TR</td><td>Tx-IMU E4D</td><td>Au-D BV</td><td>Au-IMU E4D</td><td>D-IMU UTD</td></tr><tr><td>Full A-B†</td><td>33.57</td><td>78.13</td><td>19.57</td><td>53.64</td><td>19.21</td><td>27.00</td></tr><tr><td>Frozen Cosine</td><td>27.41</td><td>28.10</td><td>7.16</td><td>25.77</td><td>6.56</td><td>2.87</td></tr><tr><td>Hub-Relative [20]</td><td>11.07</td><td>27.81</td><td>5.73</td><td>18.41</td><td>5.88</td><td>2.78</td></tr><tr><td>Bi. Ridge [71]</td><td>31.25</td><td>84.43</td><td>19.86</td><td>58.78</td><td>20.48</td><td>25.63</td></tr><tr><td>Bi. Procrustes [72]</td><td>30.22</td><td>69.11</td><td>16.73</td><td>50.36</td><td>13.95</td><td>10.52</td></tr><tr><td>ReAlign [23]</td><td>27.08</td><td>30.45</td><td>10.70</td><td>34.18</td><td>10.23</td><td>4.91</td></tr><tr><td>ERM [75]</td><td>29.65</td><td>68.89</td><td>19.73</td><td>53.78</td><td>20.29</td><td>24.72</td></tr><tr><td>IRM [76]</td><td>31.00</td><td>72.55</td><td>15.52</td><td>52.31</td><td>10.23</td><td>5.36</td></tr><tr><td>VREx [77]</td><td>29.66</td><td>75.23</td><td>18.37</td><td>52.52</td><td>15.94</td><td>12.16</td></tr><tr><td>DANN [78]</td><td>29.33</td><td>62.80</td><td>20.90</td><td>49.51</td><td>20.50</td><td>23.94</td></tr><tr><td>CORAL [79]</td><td>29.65</td><td>69.13</td><td>19.83</td><td>53.75</td><td>20.42</td><td>27.29</td></tr><tr><td>ASIF† [19]</td><td>32.54</td><td>73.52</td><td>14.71</td><td>32.03</td><td>11.71</td><td>9.57</td></tr><tr><td>Paired-OP† [72]</td><td>31.81</td><td>56.02</td><td>14.12</td><td>47.97</td><td>9.76</td><td>13.12</td></tr><tr><td>HSA (Ours)</td><td>32.60 (+5.19)</td><td>90.92 (+62.82)</td><td>21.58 (+14.42)</td><td>61.40 (+35.63)</td><td>22.81 (+16.25)</td><td>34.75 (+31.88)</td></tr></table>

LanguageBind. HSA reaches 36.79% mean Recall@10, 11.78 points above frozen cosine and 4.26 points above Paired-OP, the strongest other method included in the ranking. HSA improves all ten held-out relations and leads on seven among all methods except Full A–B. The gains cover the nearly saturated image–video relation and the weaker video–audio and image–thermal relations, spanning a broad range of native retrieval strengths. For LanguageBind prototype classification, HSA reaches 62.53% mean macro Top-1 accuracy across the five relations in Table IV. This is 18.28 points above frozen cosine and 3.11 points above bidirectional ridge. The gains span image, video, audio, depth, and thermal observations from NYUv2, TartanRGBT, and MSR-VTT.

Classification Summary and Parameter Reuse. Across the eleven relations in Table II and Table IV, HSA raises mean macro Top-1 from 29.01% to 52.43%. These comparisons fit the relation using classification source features and evaluate a common prototype bank. We also test cross-task reuse by fixing every HSA parameter estimated from the source hub edges and scoring class prototypes as candidates. On eight relations with compatible coordinates and no overlap between fitting samples and classification queries, reuse reaches 48.01% macro Top-1. Frozen cosine and separately fitted HSA reach 26.46% and 48.79%, respectively, on the same relations. The reuse-minus-cosine gain is 21.56 points, with a datasetcluster interval of [15.16, 29.52]. Overlap and preprocessing boundaries are detailed in the appendix.

Target-Paired Reference. Tables I and III show that HSA reaches 31.15% mean Recall@10, compared with 26.78% for ASIF and 26.24% for Paired-OP, and outperforms each on 17 of 19 relations. Full A–B reaches 31.49 ± 0.42% overall, with HSA achieving higher Recall@10 on 10 of 19 relations. The comparison varies across backbones: HSA reaches 24.90% on ImageBind, compared with 22.95 ± 0.36% for Full A–B, and 36.79% on LanguageBind, compared with 39.18 ± 0.50% for Full A–B. The ± values denote sample standard deviations across the three runs. The overall HSA-minus-Full A–B difference is −0.34 points, with a descriptive cluster interval of [−5.43, 5.57]. This comparison has no predefined equivalence or non-inferiority margin.

Across-Rank Gains. In Tables I and III, the HSA gain over frozen cosine increases from Recall@1 to Recall@10: from 8.73 to 14.10 points on ImageBind and from 4.38 to 11.78 points on LanguageBind. These increases show that HSA recovers correct matches beyond rank 1.

Directional Consistency. HSA improves 37 of 38 directed relations (details in the appendix). Mean Recall@10 rises from 19.54% to 32.88% for A→B and from 17.01% to 29.43% for B →A. Forward and reverse gains are 15.43 and 12.77 points on ImageBind and 11.45 and 12.11 points on LanguageBind. Improvements thus extend across both query directions on both backbones.

Takeaway. Across both backbones, HSA improves retrieval across relations and query directions, and prototype classification across relations. These gains show that task performance can improve through a hub-based readout while the underlying representations remain frozen.

## C. Does the Signal Come from the Two Hub Edges? (Q2)

After Q1 establishes the effect, Q2 traces its information source by separating valid within-edge correspondence from the convenience of shared source rows. We test within-edge correspondence, disjoint-source estimation, and source-sample scaling in that order.

TABLE III  
HELD-OUT RETRIEVAL ON TEN LANGUAGEBIND RELATIONS. BOLDFACE AND UNDERLINING MARK THE BEST AND SECOND-BEST VALUES AMONG ALL METHODS EXCEPT FULL A-B. † MARKS METHODS THAT USE TARGET-PAIR IDENTITIES: BLUE PARENTHESES SHOW HSA GAINS OVER FROZEN COSINE.
<table><tr><td rowspan="2">Method</td><td colspan="10">Evaluation Relations: Bidirectional Recall@ 10 (%) ↑</td><td rowspan="2">Mean R@1</td><td rowspan="2">Mean R@5</td><td rowspan="2">Mean R@10</td><td rowspan="2"></td><td rowspan="2">R@10 Recovery</td></tr><tr><td>Im-Vi UCF</td><td>Im-Th TR</td><td>Im-D NYU</td><td>Im-Au VGS</td><td>Vi-Th TR</td><td>Vi-D TR</td><td>Vi-Au MSR</td><td>Th-D TR</td><td>Th-Au MAVD</td><td>D-Au BV</td></tr><tr><td>Full A-B†</td><td>99.76</td><td>60.36</td><td>44.19</td><td>41.02</td><td>54.43</td><td>25.84</td><td>26.34</td><td>31.31</td><td>2.73</td><td>5.82</td><td></td><td>15.28</td><td>30.85 39.18</td><td></td><td>100.0</td></tr><tr><td>Frozen Cosine</td><td>96.49</td><td>36.93</td><td>11.62</td><td>43.40</td><td>36.01</td><td>7.11</td><td>6.96</td><td></td><td>7.57</td><td>2.08</td><td>1.88</td><td>10.28</td><td>19.78</td><td>25.00</td><td>63.8</td></tr><tr><td>Hub-Relative [20]</td><td>35.66</td><td>14.45</td><td>5.66</td><td>18.15</td><td>18.23</td><td>3.44</td><td>6.28</td><td>3.56</td><td></td><td>2.42</td><td>1.63</td><td>1.93</td><td>6.93</td><td>10.95</td><td>27.9</td></tr><tr><td>Bi. Ridge [71]</td><td>75.68</td><td>25.92</td><td>10.09</td><td>55.80</td><td>23.17</td><td>11.24</td><td>20.31</td><td>14.91</td><td></td><td>3.46</td><td>4.79</td><td>5.51</td><td>16.27</td><td>24.54</td><td>62.6</td></tr><tr><td>Bi. Procrustes [72]</td><td>86.55</td><td>33.14</td><td>7.95</td><td>56.95</td><td>28.56</td><td>13.42</td><td>20.53</td><td>14.45</td><td></td><td>3.12</td><td>3.85</td><td>7.25</td><td>18.99</td><td>26.85</td><td>68.5</td></tr><tr><td>ReAlign [23]</td><td>97.84</td><td>43.23</td><td>7.87</td><td>43.60</td><td>38.88</td><td>4.24</td><td>5.20</td><td>6.08</td><td></td><td>2.89</td><td>1.28</td><td>11.12</td><td>20.13</td><td>25.11</td><td>64.1</td></tr><tr><td>ERM [75]</td><td>81.11</td><td>23.36</td><td>7.21</td><td>51.22</td><td>18.85</td><td>9.98</td><td>15.03</td><td>10.47</td><td></td><td>4.12</td><td>4.99</td><td>5.00</td><td>15.33</td><td>22.63</td><td>57.8</td></tr><tr><td>IRM [76]</td><td>81.31</td><td>21.41</td><td>8.15</td><td>51.93</td><td>18.23</td><td>9.17</td><td>18.25</td><td>10.70</td><td></td><td>4.08</td><td>4.05</td><td>5.11</td><td>15.60</td><td>22.73</td><td>58.0</td></tr><tr><td>VREx [77]</td><td>81.15</td><td>22.67</td><td>7.49</td><td>51.18</td><td>19.76</td><td>9.79</td><td>16.84</td><td>11.70</td><td></td><td>4.08</td><td>5.22</td><td>5.02</td><td>15.53</td><td>22.99</td><td>58.7</td></tr><tr><td>DANN [78]</td><td>81.17</td><td>22.40</td><td>6.22</td><td>49.98</td><td>16.55</td><td>9.59</td><td>16.16</td><td>11.96</td><td></td><td>2.89</td><td>4.14</td><td>4.91</td><td>15.00</td><td>22.11</td><td>56.4</td></tr><tr><td>CORAL [79]</td><td>81.11</td><td>22.44</td><td>7.21</td><td>51.22</td><td>19.57</td><td>10.02</td><td>15.03</td><td>11.28</td><td></td><td>4.00</td><td>4.79</td><td>5.02</td><td>15.39</td><td>22.67</td><td>57.9</td></tr><tr><td>ASIF† [19]</td><td>92.52</td><td>44.61</td><td>18.35</td><td>64.20</td><td>36.93</td><td>16.63</td><td>18.27</td><td>18.69</td><td></td><td>2.31</td><td>4.28</td><td>10.07</td><td>23.86</td><td>31.68</td><td>80.9</td></tr><tr><td>Paired-OP† [72]</td><td>98.45</td><td>49.31</td><td>15.37</td><td>54.85</td><td>44.27</td><td>17.78</td><td>19.29</td><td>19.95</td><td></td><td>2.31</td><td>3.68</td><td>12.66</td><td>25.32</td><td>32.53</td><td>83.0</td></tr><tr><td></td><td>99.72</td><td>60.89</td><td>13.53</td><td>67.15</td><td>53.33</td><td></td><td></td><td></td><td></td><td>3.93</td><td>5.39</td><td>14.65</td><td>28.27</td><td>36.79</td><td></td></tr><tr><td>HSA (Ours)</td><td>(+3.23)</td><td>(+23.97)</td><td>(+1.91)</td><td>(+23.75)</td><td>(+17.32)</td><td>18.00 (+10.89)</td><td>26.64 (+19.68)</td><td>19.27 (+11.70)</td><td></td><td>(+1.85)</td><td>(+3.51)</td><td>(+4.38)</td><td>(+8.48)</td><td>(+11.78)</td><td>93.9</td></tr></table>

TABLE IV

LANGUAGEBIND PROTOTYPE CLASSIFICATION. MACRO TOP-1 (%) ON FIVE RELATIONS. BOLDFACE AND UNDERLINING MARK THE BEST AND SECOND-BEST VALUES AMONG ALL METHODS EXCEPT FULL A–B. <sup>†</sup> MARKS METHODS THAT USE TARGET-PAIR IDENTITIES; BLUE PARENTHESES SHOW HSA GAINS OVER FROZEN COSINE.
<table><tr><td rowspan="2">Method</td><td colspan="5">Prototype-Query Relations: Macro Top-1 (%) ↑</td></tr><tr><td>Im-D NYU</td><td>Vi-Th TR</td><td>Vi-D TR</td><td>Vi-Au MSR</td><td>Th-D TR</td></tr><tr><td>Full A-B†</td><td>61.82</td><td>51.85</td><td>65.35</td><td>24.39</td><td>58.06</td></tr><tr><td>Frozen Cosine</td><td>51.96</td><td>74.03</td><td>43.68</td><td>13.77</td><td>37.80</td></tr><tr><td>Hub-Relative [20]</td><td>43.00</td><td>68.76</td><td>35.02</td><td>7.32</td><td>31.35</td></tr><tr><td>Bi. Ridge [71]</td><td>60.24</td><td>82.58</td><td>67.10</td><td>22.12</td><td>65.02</td></tr><tr><td>Bi. Procrustes [72]</td><td>60.85</td><td>79.19</td><td>61.69</td><td>20.44</td><td>62.38</td></tr><tr><td>ReAlign [23]</td><td>46.63</td><td>79.89</td><td>37.96</td><td>13.02</td><td>35.85</td></tr><tr><td>ERM [75]</td><td>55.69</td><td>77.52</td><td>66.76</td><td>22.92</td><td>65.38</td></tr><tr><td>IRM [76]</td><td>53.70</td><td>75.99</td><td>61.24</td><td>24.95</td><td>64.94</td></tr><tr><td>VREx [77]</td><td>53.58</td><td>75.39</td><td>64.60</td><td>22.55</td><td>65.89</td></tr><tr><td>DANN [78]</td><td>50.58</td><td>81.27</td><td>63.06</td><td>23.12</td><td>68.17</td></tr><tr><td>CORAL [79]</td><td>52.96</td><td>76.39</td><td>64.46</td><td>22.92</td><td>65.32</td></tr><tr><td>ASIF† [19]</td><td>51.20</td><td>80.17</td><td>55.60</td><td>17.01</td><td>52.12</td></tr><tr><td>Paired-OP† [72]</td><td>60.00</td><td>81.97</td><td>60.00</td><td>20.56</td><td>55.76</td></tr><tr><td>HSA (Ours)</td><td>62.89 (+10.92)</td><td>84.37 (+10.33)</td><td>69.99 (+26.32)</td><td>26.37 (+12.60)</td><td>69.01 (+31.21)</td></tr></table>

Within-Edge Source Correspondence. We test the sample correspondence along both observed hub edges by permuting the hub-side rows of the A–H edge, the H–B edge, or both edges independently. Target-side row order, embeddings, marginals, estimator, hyperparameters, and scoring formula remain fixed. Every condition refits the relation directions, selected ranks, source gate, and mismatch scales under the same source-only rules, and evaluates the complete HSA score on the full gallery. Each intervention averages five seeds; the intact fit reproduces the main HSA retrieval results. As shown in Fig. 3, permuting the A–H edge, the H–B edge, or both edges reduces mean Recall@10 from 31.15% to 4.64%, 4.58%, and 6.56%, respectively. The corresponding reductions are 26.52, 26.58, and 24.59 points, and all nineteen relationlevel differences are positive under each intervention. Their dataset-cluster intervals are [9.83, 40.64], [10.03, 41.01], and [9.22, 36.53] points; all three Holm-adjusted p values equal 0.00293. These interventions identify valid sample correspondence within both hub edges as a key source of HSA’s recovery of target-modality associations. Shuffling either edge substantially reduces retrieval performance even after the complete readout is re-estimated.

![](images/b7cfc3f72adcb3a5b0bfab3f0be117792b0d1898eefeddb62e4d3ea6125ed8c4.jpg)

![](images/8e79f5ad454fd13144f8657303653f035b97a0f8eb4a0d445980f8d47d3eba5f.jpg)  
Fig. 3. Hub-Edge Correspondence in Complete HSA. Left: Recall@10 under intact correspondence and three permutation controls. Right: paired decreases in percentage points. Each dot represents a relation; permutation results average five refits. Boxes span the interquartile range, schematic notches mark medians, and whiskers span observed extrema.

Disjoint Source Instances. HSA constructs its readout from two hub-edge moment systems, which predicts that its principal retrieval gain should persist when the edges are estimated from disjoint source instances. To test this prediction, we divide source rows into equal halves for each of ten split seeds and estimate the two edges either from the same half or from opposite halves. Per-edge sample budgets, sourceonly calibration, gate construction, scoring, and evaluation remain matched. While removing cross-edge row sharing, the disjoint condition preserves valid correspondence within each edge. Table V shows that, across the nineteen relations, the sample-size-matched shared-row condition gains 11.94 points over frozen cosine, of which strictly disjoint estimation retains 10.77 points, or 90.2%; every relation remains above frozen cosine. The 95% dataset-cluster interval for the mean disjoint gain is [3.83, 16.57]. Under equal per-edge budgets, shared-row estimation is 1.18 points above strictly disjoint estimation on average, with a 95% confidence interval of [0.53, 2.30]. Thus, HSA retains most of its retrieval gain when the two moment systems are estimated from mutually disjoint source-instance sets. Its principal signal comes from valid correspondence within each hub edge and the resulting composition of the two moment systems.

TABLE V  
RETRIEVAL WITH SHARED AND DISJOINT SOURCE INSTANCES. MEAN BIDIRECTIONAL RECALL@10 (%) UNDER MATCHED PER-EDGE BUDGETS; PARENTHESES SHOW GAINS OVER FROZEN COSINE.
<table><tr><td>Method</td><td>ImageBind</td><td>LanguageBind</td><td>Overall</td></tr><tr><td>Frozen Cosine</td><td>10.80</td><td>25.00</td><td>18.27</td></tr><tr><td rowspan="2">HSA, Shared Rows</td><td>24.22</td><td>35.62</td><td>30.22</td></tr><tr><td>(+13.42)</td><td>(+10.61)</td><td>(+11.94)</td></tr><tr><td rowspan="2">HSA, Disjoint Sources</td><td>22.63</td><td>34.82</td><td>29.04</td></tr><tr><td>(+11.83)</td><td>(+9.81)</td><td>(+10.77)</td></tr></table>

![](images/f2579b099b9ce99a194e7fc13a3c220f5f87297ea0e5e5309f1927ed34329a51.jpg)  
Fig. 4. Source-Sample Scaling under Shared and Disjoint Estimation. Bidirectional Recall@10 for each relation across ten nested per-edge budgets, normalized by the shared-row result at the full budget. Filled and hollow markers denote shared-row and disjoint-source estimation, respectively; shaded ribbons show their paired gaps. Numbers beside the relation labels report the shared-row Recall@10 (%) at the full budget.

Source-Sample Scaling. To determine how the remaining shared–disjoint difference changes with source sample size, we estimate both conditions at ten nested per-edge budgets, {10, 15, 20, 25, 35, 50, 65, 75, 85, 100}%, over split seeds 42– 46. For each relation, Fig. 4 normalizes both trajectories by the shared-row result at the full budget; the endpoint label gives that relation’s absolute full-budget Recall@10. As the per-edge source budget increases from 10% to 100%, the mean Recall@10 over all nineteen relations rises from 17.94% to 29.05% under disjoint estimation and from 22.68% to 30.21% under shared-row estimation. The difference between the two conditions contracts from 4.73 points at the 10% budget to 1.16 points at the full budget. At 50%, the disjoint estimator already reaches 26.59%, or 91.5% of its full-budget value. The gap narrows on both ImageBind and LanguageBind, showing a consistent sample-budget trend across the two backbones. Source-row sharing therefore primarily improves moment estimation at low sample budgets; as source data increase, HSA under disjoint estimation approaches the shared-row readout.

TABLE VI  
CONSTRUCTION OF THE HUB-READABLE RELATION. MEANBIDIRECTIONAL RECALL@10 (%); PARENTHESES SHOW CHANGES FROMFULL HSA.
<table><tr><td>Construction</td><td>ImageBind</td><td>LanguageBind</td><td>Overall</td></tr><tr><td>Diagonal Hub Covariance</td><td> $2 0 . 8 8 \left( - 4 . 0 2 \right)$ </td><td> $3 2 . 5 5 ( - 4 . 2 3 )$ </td><td> $2 7 . 0 2 \ : ( - 4 . 1 3 )$ </td></tr><tr><td>Trace-Matched Identity</td><td> $2 0 . 7 6 ( - 4 . 1 3 )$ </td><td> $3 2 . 5 1 ( - 4 . 2 7 )$ </td><td> $2 6 . 9 5 \ : ( - 4 . 2 1 )$ </td></tr><tr><td>HSA (Ours)</td><td>24.90</td><td>36.79</td><td>31.15</td></tr></table>

Takeaway. The main retrieval gain depends on valid correspondence within both hub edges and persists with disjoint source instances. The shrinking shared–disjoint gap with more data links the benefit of source sharing to estimation precision.

## D. Where Is Hub-Readable Knowledge Located? (Q3)

Having identified the source information in Q2, Q3 follows it from relation construction to spectral localization and capacity: we test the required hub covariance, the informativeness of leading carriers, and the rate at which their gain accumulates. Hub-Readable Relation Construction. We compare full hub covariance construction, diagonal hub covariance construction, and trace-matched identity construction, fixing target standardization, candidate-resolution coordinates, gate construction, gallery, and retrieval protocol. Full hub covariance construction preserves variances and cross-dimensional correlations, while diagonal hub covariance construction preserves variances alone. Trace-matched identity construction weights dimensions equally and matches the trace of the full regularized inverse covariance matrix. Table VI reports mean Recall@10 of 31.15%, 27.02%, and 26.95%, respectively. Full hub covariance construction improves 18 of 19 relations against each alternative, with mean gains of 4.13 and 4.21 points. The respective 95% intervals are [1.92, 5.76] and [2.50, 5.42], with Holm-adjusted $p = 0 . 0 0 3 9 1$ for both comparisons. Preserving cross-dimensional hub correlations thus improves relation recovery and retrieval.

Spectral Carrier Localization. We compare leading, subsequent, and random paired directions within the complete HSA score. Carrier count, candidate-resolution coordinates, the source gate, and the resolution scale remain fixed at the intact fit. All direction sets receive the same elementwise leading reliability spectrum, with carrier scales estimated separately using the same source-only mismatch rule. The subsequent condition uses the next singular directions, while the random condition averages five paired draws from the complement of the leading block. Leading carriers reproduce full HSA. Table VII reports 31.15% mean Recall@10 for leading carriers, compared with 14.72% for subsequent directions and 11.77% for random directions. Their mean advantages are 16.44 points [6.31, 25.41] and 19.38 points [7.06, 29.21], respectively. Both comparisons have Holm-adjusted datasetgroup $p = 0 . 0 0 1 9 5$ . Fig. 5 shows that leading carriers win all nineteen relations. This result localizes usable relation information to the leading paired directions under matched carrier counts and assigned reliability spectra.

![](images/515b06ae1cde655c33b756078b6b2d43951bb4affa11a8790c208d97fb3dea9f.jpg)  
Fig. 5. Spectral Carrier Localization in Complete HSA. From outer to inner, the three heatmap rings report bidirectional Recall@10 (%) for leading, subsequent, and random paired directions. Carrier counts and assigned reliability spectra are matched within each relation. Results for random directions are averaged over five seeds.

TABLE VII  
SPECTRAL CARRIER LOCALIZATION IN THE COMPLETE SCORE. ALLCONDITIONS RECEIVE THE SAME LEADING RELIABILITY SPECTRUM ANDUSE THE SAME SOURCE-ONLY RULE FOR DIRECTION-SPECIFIC CARRIERCALIBRATION. VALUES ARE MEAN BIDIRECTIONAL RECALL@10 (%);PARENTHESES SHOW CHANGES FROM THE LEADING CONDITION.
<table><tr><td>Carrier Block</td><td>ImageBind</td><td>LanguageBind</td><td>Overall</td></tr><tr><td>Subsequent</td><td> $6 . 8 9 ( - 1 8 . 0 1 )$ </td><td> $2 1 . 7 6 \left( - 1 5 . 0 3 \right)$ </td><td> $1 4 . 7 2 \ : ( - 1 6 . 4 4 )$ </td></tr><tr><td>Random</td><td> $7 . 1 6 ( - 1 7 . 7 4 )$ </td><td> $1 5 . 9 2 \left( - 2 0 . 8 6 \right)$ </td><td> $1 1 . 7 7 ( - 1 9 . 3 8 )$ </td></tr><tr><td>Leading (HSA)</td><td>24.90</td><td>36.79</td><td>31.15</td></tr></table>

Relation-Wise Carrier Capacity. To quantify carrier requirements, we retain nested leading prefixes from 0% to 100% in 5% steps, fixing the HSA fit, candidate-resolution branch, gate, and gallery. Each prefix receives source-only carrier-scale calibration; five equal-count random subsets (seeds 42–46) provide capacity-matched controls. Let $R ( c )$ denote bidirectional Recall@10 at retained fraction c, and R(0) the result with zero carriers and all remaining score components retained. For positive full gain $R ( 1 ) - R ( 0 )$ , C90 is the smallest evaluated fraction at which the score reaches $R ( 0 ) + 0 . 9 [ R ( 1 ) - R ( 0 ) ]$ and remains at or above this threshold at all subsequent checkpoints. Fractions are relative to the full carrier set selected for each relation. Fig. 6 shows median C90 values of 10% for ImageBind and 40% for LanguageBind. Eight of nine ImageBind relations and four of the nine defined LanguageBind relations reach C90 by 25% capacity. For LanguageBind Im– D, Recall@10 falls from 14.53% at zero carriers to 13.53% at full capacity, leaving C90 undefined. ImageBind typically concentrates gains in shorter leading prefixes, whereas LanguageBind requires broader carrier coverage.

![](images/5885e907b3165c77ef6f280b69486275693d5e31be17467e55daade5e8e86d81.jpg)  
Fig. 6. Relation-Wise Carrier Capacity in HSA. Curves show retrieval performance as progressively more leading carriers are retained. Background colors indicate Recall@10 differences from equal-count random carrier subsets, in percentage points. C90 denotes the minimum retained capacity that sustains at least 90% of the full-carrier gain over the zero-carrier baseline; N/A indicates a nonpositive full-carrier gain.

TABLE VIII  
CARRIER-PREFIX ACCUMULATION. MEAN BIDIRECTIONAL RECALL@10 (%) AND MEDIAN C90; SCORE PARENTHESES SHOW CHANGES FROM FULL HSA.
<table><tr><td>Retained Carriers</td><td>ImageBind</td><td>LanguageBind</td><td>Overall</td></tr><tr><td>Leading 25%</td><td> $2 4 . 8 9 ( - 0 . 0 1 )$ </td><td> $3 5 . 0 6 ( - 1 . 7 3 )$ </td><td> $3 0 . 2 4 ( - 0 . 9 1 )$ </td></tr><tr><td>Leading 50%</td><td> $2 4 . 9 4 \ : ( + 0 . 0 5 )$ </td><td> $3 6 . 3 8 ( - 0 . 4 1 )$ </td><td>30.96 (-0.19)</td></tr><tr><td>Random 50%</td><td> $2 1 . 9 5 ( - 2 . 9 5 )$ </td><td> $3 3 . 3 8 ( - 3 . 4 0 )$ </td><td>27.97 (-3.19)</td></tr><tr><td>Full HSA (Ours)</td><td>24.90</td><td>36.79</td><td>31.15</td></tr><tr><td>Median C90</td><td>10%</td><td>40%</td><td>20%</td></tr></table>

Backbone-Level Carrier Accumulation. Fig. 7 compares leading prefixes with the mean of five equal-count random subsets at 10%, 25%, 50%, and 100% retained capacity. Under the complete HSA score, Table VIII reports mean bidirectional Recall@10 of 30.24% and 30.96% for the leading 25% and 50% prefixes, respectively, compared with 31.15% for full HSA. Increasing capacity from 25% to 50% adds 0.72 percentage points, while including the remaining half adds a further 0.19 points. Thus, the aggregate performance approaches the full-capacity result with progressively smaller additional gains over these intervals. At each partial capacity shown, leading prefixes achieve higher mean Recall@10 than random subsets for both backbones. At 50% capacity, random subsets reach 27.97% overall, approximately 3.0 percentage points below the leading prefix. This comparison supports the value of spectral ordering when the number of retained carriers is constrained. The gap narrows as capacity increases and vanishes at 100%, where both conditions use the same full carrier set. The accumulation patterns differ between the backbones on their respective relation sets. ImageBind is already close to its fullcapacity mean at 25% capacity (24.89% versus 24.90%) and changes little at the larger reported capacities. LanguageBind shows a more gradual increase, from 35.06% at 25% capacity to 36.38% at 50% and 36.79% at full capacity. Together with the localization controls, these results support spectral ordering as a useful criterion for retaining retrieval performance with fewer carriers. The extent to which a short leading prefix preserves performance depends on the evaluated backbone and relation set.

![](images/4ed01428d5cfe2b78cf950f38b4935653bf6a1504f7a701438362e3edf531bc5.jpg)

![](images/28cfcc161ea5e0a648aaa2a84249247c6da9c443d31c6ffb948bcb7e2458d6b2.jpg)  
Fig. 7. Backbone-Level Carrier Accumulation in HSA. Bars show mean bidirectional Recall@10 for leading and random carriers; circles mark the leading result and five seed-specific random means. Diamonds show leadingminus-random gaps in percentage points; error bars span the seed range.

Takeaway. Cross-dimensional hub covariance improves recovery, and usable evidence concentrates in leading paired carriers. Their advantage under matched reliability spectra shows that spectral direction matters; differing accumulation rates reveal that this concentration varies across backbones.

## E. How Does HSA Turn Knowledge into Rankings? (Q4)

Q4 tests how carrier reliability and readout components convert the localized relation into ranking gains.

Reliability-Controlled Activation. To test how the channel reliabilities $c _ { j }$ control HSA’s use of the localized knowledge, we fix directions, retained rank, candidate-resolution coordinates, the gate, and both mismatch scales, then replace every retained reliability by $c _ { j } ( \alpha ) = \alpha c _ { j }$ for $\alpha \in \{ 0 , 0 . 0 5 , \ldots , 1 \}$

![](images/e70ec8aa9192f65f25386bcc61cac6631093d13dc036c7a90ec2a1fe34c617dc.jpg)  
Fig. 8. Reliability-Controlled Spectral Activation in HSA. Each curve shows the change in bidirectional Recall@10 relative to $\alpha = 0 ,$ normalized by the maximum observed increase for that relation across 21 channelreliability scales in [0, 1]. Panels (a) and (b) show results for ImageBind and LanguageBind, respectively.

Each displayed trajectory is centered at its zero-dose Recall@10 and divided by its largest observed increase; all reported statistics use the unnormalized values. As shown in Fig. 8, mean Recall@10 rises from 19.15% at $\alpha \ = \ 0$ to 31.15% at α = 1. Eighteen relations improve at full reliability, and the median within-relation Spearman correlation between α and Recall@10 is 0.913. As the reliability parameters are restored, carrier evidence progressively contributes to the score and improves retrieval on most relations.

Component Responsibilities. We assess carrier evidence with full-gallery Recall@10 and each remaining component with its function-specific metric. Reliability uses paired-versusmismatched discrimination (P-AUC); candidate resolution uses Top-1 accuracy in the fixed carrier top-ten candidate pool (C@1). The source gate uses relation-level benefit discrimination (G-AUC), and orthogonalization uses absolute correlation between the carrier and resolution scores (Red.). These additional metrics were specified before measurement and analyzed exploratorily. Table IX reports backbone-specific Recall@10 ablations and relation-equal function-metric changes; G-AUC uses all nineteen relations jointly. Removing the carrier score reduces mean Recall@10 by 12.00 points, the largest drop. Uniform channel reliability and removal of candidate resolution, the source gate, and orthogonalization reduce it by 2.48, 3.43, 3.86, and 0.99 points, respectively. The corresponding metrics show a 5.20-point improvement in paired-versus-mismatched discrimination AUC, a 4.43-point improvement in Top-1 accuracy within the fixed carrier top-ten candidate pool, and lower cross-branch correlation. The source gate discriminates whether candidate resolution improves retrieval with an AUC of 0.714. Candidate-resolution accuracy is conditional on the carrier top-ten pool containing a valid match. Complete metric definitions, backbone breakdowns, counts, and intervals are provided in the appendix.

Takeaway. Reliability weights carrier evidence, while the source gate controls candidate resolution, making spectral activation sensitive to both relation strength and residual geometry.

## F. What Does the Readout Cost? (Q5)

We profile 14 methods across 19 retrieval and 11 classification relations, reporting fit/train time, scoring cost, trainable parameters, and median final-fit steps. For trainable baselines, training durations are selected using pilot validation splits drawn only from the training partition. Full A–B uses a single 200-step budget, selected globally by minimizing aggregate validation loss on these pilot splits and then fixed across relations and seeds. Fitting uses method-specific software environments, limiting direct timing comparisons; classification readouts share a common CPU configuration. Timing protocols and reproduction checks are detailed in the appendix. Table X shows that HSA scores retrieval pairs at a cost comparable to the other zero-target-pair analytic readouts. HSA fits a retrieval relation in a median of 6.503 seconds and scores one million directional pairs in 0.031 seconds. Its higher one-time fitting cost is amortized over subsequent queries. When encoder coordinates are compatible, retrieval and prototype classification can share the relation readout estimated from the source hub edges, without re-estimating its parameters separately for the two tasks. The classification timing table measures fitting under the separately specified classification protocol. Table XI reports that, under the pooled workload across eleven classification relations, HSA has a CPU readout latency of 25.614 milliseconds per 1,000 queries and a throughput of 9.172 million query–prototype scores per second. Timing includes method-specific transforms, computation of the complete query–prototype score matrix, and Top-1 selection, quantifying the classification readout cost after fitting. Retrieval cost is normalized by directional pairs, while classification latency includes scoring the class-prototype bank and selecting a label. The two measurements therefore describe the work required by their respective task readouts.

TABLE IX  
COMPONENT ABLATIONS. BACKBONE-SPECIFIC RECALL@10 (%) AND OVERALL FUNCTION-METRIC CHANGES RELATIVE TO FULL HSA. CHANGES ARE IN PERCENTAGE POINTS EXCEPT RED., WHICH USES ABSOLUTE CORRELATION. ARROWS INDICATE PREFERRED DIRECTIONS; BLUE/RED MARK IMPROVEMENTS/DEGRADATIONS.
<table><tr><td>Ablation</td><td>ImageBind R@10 (∆)</td><td>LanguageBind R@10 (∆)</td><td>Metric</td><td> $\Delta$  Metric</td></tr><tr><td>w/o Carrier Score</td><td>10.47 (-14.43)</td><td>26.96 (-9.82)</td><td>R@10 ↑</td><td>-12.00</td></tr><tr><td>w/o Channel Reliability</td><td>23.60 (-1.30)</td><td>33.25 (-3.53)</td><td>P-AUC ↑</td><td>-5.20</td></tr><tr><td>w/o Candidate Resolution</td><td>25.33 (+0.44)</td><td>29.87 (-6.92)</td><td>C@1↑</td><td>-4.43</td></tr><tr><td>w/o Source Gate</td><td>19.29 (-5.61)</td><td>34.50 (-2.29)</td><td>G-AUC ↑</td><td>-21.43</td></tr><tr><td>w/o Orthogonalization</td><td>25.15 (+0.26)</td><td>34.67 (-2.12)</td><td>Red. ↓</td><td>+0.273</td></tr><tr><td>Full HSA (Ours)</td><td>24.90</td><td>36.79</td><td>–</td><td>一</td></tr></table>

TABLE X

RELATION FITTING AND RETRIEVAL COST. MEDIAN COST OVER 19 RELATIONS. FIT/TRAIN IS IN SECONDS. SCORING IS SECONDS PER MILLION DIRECTIONAL PAIRS; STEPS IS THE MEDIAN FINAL-FIT COUNT. <sup>†</sup> MARKS METHODS THAT USE TARGET-PAIR IDENTITIES.
<table><tr><td>Method</td><td>Fit/Train</td><td> $\mathrm { s / 1 0 ^ { 6 } }$  Pairs</td><td>Params.</td><td>Steps</td></tr><tr><td>Full A-B†</td><td>0.532</td><td>0.015</td><td>2.10–2.62M</td><td>200</td></tr><tr><td>Frozen Cosine</td><td></td><td>0.015</td><td>0</td><td>0</td></tr><tr><td>Hub-Relative [20]</td><td>0.014</td><td>0.035</td><td>0</td><td>0</td></tr><tr><td>Bi. Ridge [71]</td><td>1.076</td><td>0.032</td><td>0</td><td>0</td></tr><tr><td>Bi. Procrustes [72]</td><td>1.506</td><td>0.032</td><td>0</td><td>0</td></tr><tr><td>ReAlign [23]</td><td>0.042</td><td>0.029</td><td>0</td><td>0</td></tr><tr><td>ERM [75]</td><td>5.376</td><td>0.019</td><td>3.15–3.94M</td><td>400</td></tr><tr><td>IRM [76]</td><td>9.024</td><td>0.020</td><td>3.15-3.94M</td><td>400</td></tr><tr><td>VREx [77]</td><td>5.533</td><td>0.020</td><td>3.15–3.94M</td><td>400</td></tr><tr><td>DANN [78]</td><td>6.354</td><td>0.019</td><td>3.22-4.00M</td><td>400</td></tr><tr><td>CORAL [79]</td><td>7.296</td><td>0.019</td><td>3.15-3.94M</td><td>400</td></tr><tr><td>ASIF† [19]</td><td>0.015</td><td>0.730</td><td>0</td><td>0</td></tr><tr><td>Paired-OP† [72]</td><td>1.732</td><td>0.041</td><td>0</td><td>0</td></tr><tr><td>HSA (Ours)</td><td>6.503</td><td>0.031</td><td>0</td><td>0</td></tr></table>

TABLE XI  
PROTOTYPE-CLASSIFICATION COST. COST OVER ELEVEN RELATIONS. FIT/TRAIN IS IN SECONDS. READOUT REPORTS CPU MILLISECONDS PER 1,000 QUERIES AND MILLIONS OF QUERY–PROTOTYPE SCORES PER SECOND. STEPS IS THE MEDIAN FINAL-FIT COUNT; <sup>†</sup> MARKS METHODS THAT USE TARGET-PAIR IDENTITIES.
<table><tr><td>Method</td><td>Fit/Train</td><td> $\mathrm { m s / 1 0 ^ { 3 } \ { q } } .$ </td><td>M Scores/s</td><td>Params.</td><td>Steps</td></tr><tr><td>Full A-B†</td><td>1.028</td><td>7.535</td><td>31.181</td><td>2.10–2.62M</td><td>200</td></tr><tr><td>Frozen Cosine</td><td></td><td>1.139</td><td>206.181</td><td>0</td><td>0</td></tr><tr><td>Hub-Relative [20]</td><td>0.005</td><td>18.838</td><td>12.472</td><td>0</td><td>0</td></tr><tr><td>Bi. Ridge [71]</td><td>0.734</td><td>18.971</td><td>12.384</td><td>0</td><td>0</td></tr><tr><td>Bi. Procrustes [72]</td><td>0.684</td><td>20.070</td><td>11.706</td><td>0</td><td>0</td></tr><tr><td>ReAlign [23]</td><td>0.022</td><td>9.388</td><td>25.024</td><td>0</td><td>0</td></tr><tr><td>ERM [75]</td><td>4.958</td><td>10.894</td><td>21.566</td><td>3.15-3.94M</td><td>400</td></tr><tr><td>IRM [76]</td><td>8.638</td><td>10.813</td><td>21.727</td><td>3.15-3.94M</td><td>400</td></tr><tr><td>VREx [77]</td><td>5.272</td><td>10.590</td><td>22.184</td><td>3.15-3.94M</td><td>400</td></tr><tr><td>DANN [78]</td><td>5.934</td><td>10.726</td><td>21.903</td><td>3.22-4.00M</td><td>400</td></tr><tr><td>CORAL [79]</td><td>7.102</td><td>10.754</td><td>21.847</td><td>3.15-3.94M</td><td>400</td></tr><tr><td>ASIF† [19]</td><td>0.013</td><td>151.247</td><td>1.553</td><td>0</td><td>0</td></tr><tr><td>Paired-OP† [72]</td><td>0.698</td><td>21.868</td><td>10.743</td><td>0</td><td>0</td></tr><tr><td>HSA (Ours)</td><td>4.617</td><td>25.614</td><td>9.172</td><td>0</td><td>0</td></tr></table>

Takeaway. HSA incurs fitting overhead once, then supports lowcost retrieval and classification readouts, amortizing relation recovery over repeated queries.

## VI. CONCLUSION

This work characterizes the hub-readable component of latent multimodal knowledge in frozen binding models and shows how it can support cross-modal comparison. Our analysis establishes what can be recovered from the second-order statistics of two trained hub connections and clarifies the limits imposed by the hub. Building on this characterization, HSA expresses the recovered relation through paired spectral carriers and constructs a closed-form readout for retrieval and prototype classification, without target-pair fitting or backbone updates. Experiments across two frozen backbones, covering nineteen retrieval and eleven classification relations, show improved average performance over native cosine scores. Correspondence interventions and spectral controls further support the roles of valid within-edge correspondence and leading paired directions in the observed retrieval gains. Together, these findings connect the recoverability of latent multimodal structure with its practical utility, positioning HSA as both a relation readout and a diagnostic of the knowledge accessible through a frozen hub. Future work could extend relation recovery to nonlinear dependencies and integrate complementary information from multiple hubs. Tracking the recovered relations and their task utility during training could also help clarify how latent multimodal knowledge emerges, evolves, and becomes usable.

## REFERENCES

[1] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, “Multimodal ma-ˇ chine learning: A survey and taxonomy,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 2, pp. 423–443, Feb. 2019, doi: 10.1109/TPAMI.2018.2798607.

[2] P. Xu, X. Zhu, and D. A. Clifton, “Multimodal learning with transformers: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 10, pp. 12113–12132, Oct. 2023, doi: 10.1109/TPAMI.2023.3275156.

[3] Y. Zhu, Y. Wu, N. Sebe, and Y. Yan, “Vision+X: A survey on multimodal learning in the light of data,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 9102–9122, Dec. 2024, doi: 10.1109/TPAMI.2024.3420239.

[4] J. Liu, A. Shahroudy, M. Perez, G. Wang, L.-Y. Duan, and A. C. Kot, “NTU RGB+D 120: A large-scale benchmark for 3D human activity understanding,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 42, no. 10, pp. 2684–2701, Oct. 2020, doi: 10.1109/TPAMI.2019.2916873.

[5] D. Lahat, T. Adali, and C. Jutten, “Multimodal data fusion: An overview of methods, challenges, and prospects,” Proc. IEEE, vol. 103, no. 9, pp. 1449–1477, Sep. 2015, doi: 10.1109/JPROC.2015.2460697.

[6] Z. Lu, “A theory of multimodal learning,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 57244–57255, doi: 10.52202/075280-2501.

[7] Y. Zong, O. Mac Aodha, and T. M. Hospedales, “Self-supervised multimodal learning: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 7, pp. 5299–5318, Jul. 2025, doi: 10.1109/TPAMI.2024.3429301.

[8] R. Arandjelovic and A. Zisserman, “Look, listen and learn,” in´ Proc. IEEE Int. Conf. Comput. Vis., 2017, pp. 609–617, doi: 10.1109/ICCV.2017.73.

[9] Y. Tian, D. Krishnan, and P. Isola, “Contrastive multiview coding,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 776–794, doi: 10.1007/978-3- 030-58621-8 45.

[10] J.-B. Alayrac et al., “Self-supervised multimodal versatile networks,” in Adv. Neural Inf. Process. Syst., vol. 33, 2020, pp. 25–37.

[11] P. Morgado, I. Misra, and N. Vasconcelos, “Robust audio-visual instance discrimination,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 12929–12940, doi: 10.1109/CVPR46437.2021.01274.

[12] H. Akbari et al., “VATT: Transformers for multimodal self-supervised learning from raw video, audio and text,” in Adv. Neural Inf. Process. Syst., vol. 34, 2021, pp. 24206–24221.

[13] A. Guzhov, F. Raue, J. Hees, and A. Dengel, “AudioCLIP: Extending CLIP to image, text and audio,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2022, pp. 976–980, doi: 10.1109/ICASSP43922.2022.9747631.

[14] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. 38th Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 139, 2021, pp. 8748–8763.

[15] C. Jia et al., “Scaling up visual and vision-language representation learning with noisy text supervision,” in Proc. 38th Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 139, 2021, pp. 4904–4916.

[16] R. Girdhar et al., “ImageBind: One embedding space to bind them all,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 15180–15190, doi: 10.1109/CVPR52729.2023.01457.

[17] B. Zhu et al., “LanguageBind: Extending video-language pretraining to N-modality by language-based semantic alignment,” in Int. Conf. Learn. Represent., 2024, pp. 9588–9608.

[18] Y. Liu et al., “Multimodal medical image binding via shared text embeddings,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., 2026, pp. 1610–1620, doi: 10.1109/WACV61042.2026.00162.

[19] A. Norelli, M. Fumero, V. Maiorca, L. Moschella, E. Rodola, and\` F. Locatello, “ASIF: Coupled data turns unimodal models to multimodal without training,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 15303–15319, doi: 10.52202/075280-0673.

[20] L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, and E. Rodola, “Relative representations enable zero-shot latent space\` communication,” in Int. Conf. Learn. Represent., 2023. [Online]. Available: https://openreview.net/forum?id=SrC-nwieGJ

[21] Z. Wang et al., “Connecting multi-modal contrastive representations,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 22099–22114, doi: 10.52202/075280-0970.

[22] Z. Zhang et al., “Extending multi-modal contrastive representations,” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 91880–91903, doi: 10.52202/079017-2915.

[23] X. Yu et al., “Modality gap-driven subspace alignment training paradigm for multimodal large language models,” arXiv:2602.07026, 2026.

[24] J. Xie, X. Xiao, R. Liu, Z. Huang, Y. Zheng, and H. Huang, “EmergentBridge: Improving zero-shot cross-modal transfer in unified multimodal embedding models,” in Proc. 32nd ACM SIGKDD Conf. Knowl. Discovery Data Mining, 2026, pp. 5708–5719, doi: 10.1145/3770855.3818172.

[25] F. Groger, S. Wen, H. Le, and M. Brbi¨ c, “With limited data for´ multimodal alignment, let the STRUCTURE guide you,” in Adv. Neural Inf. Process. Syst., vol. 38, 2025, pp. 168473–168502, doi: 10.52202/085713-5075.

[26] M. Maniparambil, R. Akshulakov, Y. A. D. Djilali, S. Narayan, A. Singh, and N. E. O’Connor, “Harnessing frozen unimodal encoders for flexible multimodal alignment,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 29847–29857, doi: 10.1109/CVPR52734.2025.02778.

[27] H. Hotelling, “Relations between two sets of variates,” Biometrika, vol. 28, nos. 3–4, pp. 321–377, Dec. 1936, doi: 10.1093/biomet/28.3- 4.321.

[28] J. R. Kettenring, “Canonical analysis of several sets of variables,” Biometrika, vol. 58, no. 3, pp. 433–451, Dec. 1971, doi: 10.1093/biomet/58.3.433.

[29] G. Andrew, R. Arora, J. Bilmes, and K. Livescu, “Deep canonical correlation analysis,” in Proc. 30th Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 28, no. 3, 2013, pp. 1247–1255.

[30] W. Wang, R. Arora, K. Livescu, and J. Bilmes, “On deep multi-view representation learning,” in Proc. 32nd Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 37, 2015, pp. 1083–1092.

[31] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, and A. Y. Ng, “Multimodal deep learning,” in Proc. 28th Int. Conf. Mach. Learn., 2011, pp. 689–696.

[32] N. Srivastava and R. Salakhutdinov, “Multimodal learning with deep Boltzmann machines,” in Adv. Neural Inf. Process. Syst., vol. 25, 2012, pp. 2222–2230.

[33] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 8, pp. 5625–5644, Aug. 2024, doi: 10.1109/TPAMI.2024.3369699.

[34] J. Lu, D. Batra, D. Parikh, and S. Lee, “ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks,” in Adv. Neural Inf. Process. Syst., vol. 32, 2019, pp. 13–23.

[35] H. Tan and M. Bansal, “LXMERT: Learning cross-modality encoder representations from transformers,” in Proc. Conf. Empir. Methods Nat. Lang. Process. Int. Joint Conf. Nat. Lang. Process., 2019, pp. 5100– 5111, doi: 10.18653/v1/D19-1514.

[36] Y.-C. Chen et al., “UNITER: Universal image-text representation learning,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 104–120, doi: 10.1007/978-3-030-58577-8 7.

[37] A. Singh et al., “FLAVA: A foundational language and vision alignment model,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 15617–15629, doi: 10.1109/CVPR52688.2022.01519.

[38] J. Li, R. Selvaraju, A. Gotmare, S. Joty, C. Xiong, and S. C. H. Hoi, “Align before fuse: Vision and language representation learning with momentum distillation,” in Adv. Neural Inf. Process. Syst., vol. 34, 2021, pp. 9694–9705.

[39] J. Li, D. Li, C. Xiong, and S. Hoi, “BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation,” in Proc. 39th Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 162, 2022, pp. 12888–12900.

[40] X. Zhai et al., “LiT: Zero-shot transfer with locked-image text tuning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 18102–18112, doi: 10.1109/CVPR52688.2022.01759.

[41] J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in Proc. 40th Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 202, 2023, pp. 19730–19742.

[42] F. Shen and J. Tang, “IMAGPose: A unified conditional framework for pose-guided person generation,” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 6246–6266, doi: 10.52202/079017-0202.

[43] F. Shen et al., “IMAGDressing-v1: Customizable virtual dressing,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 7, 2025, pp. 6795–6804, doi: 10.1609/aaai.v39i7.32729.

[44] B. Zhou, L. Li, Y. Wang, H. Liu, Y. Yao, and W. Wang, “UniAlign: Scaling multimodal alignment within one unified model,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 29644– 29655, doi: 10.1109/CVPR52734.2025.02760.

[45] Y. Gao, S. Kim, D. E. Austin, and C. McIntosh, “MEDBind: Unifying language and multimodal medical data embeddings,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent., 2024, pp. 218–228, doi: 10.1007/978-3-031-72390-2 21.

[46] Y. Gao, S. Kim, J. You, and C. McIntosh, “ProbMED: A probabilistic framework for medical multimodal binding,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025, pp. 20157–20167, doi: 10.1109/ICCV51701.2025.01875.

[47] Y. Lyu, X. Zheng, J. Zhou, and L. Wang, “UniBind: LLM-augmented unified and balanced representation space to bind them all,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 26742– 26752, doi: 10.1109/CVPR52733.2024.02526.

[48] L. Xue et al., “ULIP: Learning a unified representation of language, images, and point clouds for 3D understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 1179–1189, doi: 10.1109/CVPR52729.2023.00120.

[49] X. Liu, X. Xia, S.-K. Ng, and T.-S. Chua, “Principled multimodal representation learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 8, pp. 9114–9128, Aug. 2026, doi: 10.1109/TPAMI.2026.3675685.

[50] X. Liu et al., “Calibrated multimodal representation learning with missing modalities,” in Proc. 43rd Int. Conf. Mach. Learn., 2026. [Online]. Available: https://icml.cc/virtual/2026/poster/66039

[51] Z. Wang et al., “FreeBind: Free lunch in unified multimodal space via knowledge fusion,” in Proc. 41st Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 235, 2024, pp. 52233–52246.

[52] Z. Wang et al., “OmniBind: Large-scale omni multimodal representation via binding spaces,” in Int. Conf. Learn. Represent., 2025, pp. 20831– 20851.

[53] Z. Huang, G. Niu, B. Han, M. Sugiyama, and T. Liu, “Towards out-ofmodal generalization without instance-level modal correspondence,” in Int. Conf. Learn. Represent., 2025, pp. 26989–27008.

[54] S. Hong, J. Kim, J. You, S. Choi, S. Kwak, and H. Cho, “TextME: Bridging unseen modalities through text descriptions,” in Proc. 43rd Int. Conf. Mach. Learn., 2026. [Online]. Available: https://icml.cc/virtual/2026/poster/63946

[55] M. Klabunde, T. Schumacher, M. Strohmaier, and F. Lemmerich, “Similarity of neural network models: A survey of functional and representational measures,” ACM Comput. Surv., vol. 57, no. 9, Sep. 2025, Art. no. 242, doi: 10.1145/3728458.

[56] Y. Li, J. Yosinski, J. Clune, H. Lipson, and J. Hopcroft, “Convergent learning: Do different neural networks learn the same representations?” in Proc. 1st Int. Workshop Feature Extraction: Modern Questions and Challenges at NIPS, ser. Proc. Mach. Learn. Res., vol. 44, 2015, pp. 196–212.

[57] E. Boix-Adsera, H. Lawrence, G. Stepaniants, and P. Rigollet, “GULP:\` A prediction-based metric between representations,” in Adv. Neural Inf. Process. Syst., vol. 35, 2022, pp. 7115–7127, doi: 10.52202/068431- 0516.

[58] Y. Bansal, P. Nakkiran, and B. Barak, “Revisiting model stitching to compare neural representations,” in Adv. Neural Inf. Process. Syst., vol. 34, 2021, pp. 225–236.

[59] K. Lenc and A. Vedaldi, “Understanding image representations by measuring their equivariance and equivalence,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2015, pp. 991–999, doi: 10.1109/CVPR.2015.7298701.

[60] N. Kriegeskorte, M. Mur, and P. Bandettini, “Representational similarity analysis–Connecting the branches of systems neuroscience,” Front. Syst. Neurosci., vol. 2, Nov. 2008, Art. no. 4, doi: 10.3389/neuro.06.004.2008.

[61] M. Raghu, J. Gilmer, J. Yosinski, and J. Sohl-Dickstein, “SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability,” in Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 6076–6085.

[62] A. S. Morcos, M. Raghu, and S. Bengio, “Insights on representational similarity in neural networks with canonical correlation,” in Adv. Neural Inf. Process. Syst., vol. 31, 2018, pp. 5732–5741.

[63] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in Proc. 36th Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 97, 2019, pp. 3519–3529.

[64] M. Maniparambil et al., “Do vision and language encoders represent the world similarly?” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 14334–14343, doi: 10.1109/CVPR52733.2024.01359.

[65] A. H. Williams, E. Kunz, S. Kornblith, and S. W. Linderman, “Generalized shape metrics on neural representations,” in Adv. Neural Inf. Process. Syst., vol. 34, 2021, pp. 4738–4750.

[66] F. Ding, J.-S. Denain, and J. Steinhardt, “Grounding representation similarity through statistical testing,” in Adv. Neural Inf. Process. Syst., vol. 34, 2021, pp. 1556–1568.

[67] V. W. Liang, Y. Zhang, Y. Kwon, S. Yeung, and J. Y. Zou, “Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning,” in Adv. Neural Inf. Process. Syst., vol. 35, 2022, pp. 17612–17625, doi: 10.52202/068431-1280.

[68] M. Huh, B. Cheung, T. Wang, and P. Isola, “Position: The Platonic Representation Hypothesis,” in Proc. 41st Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 235, 2024, pp. 20617–20642.

[69] D. Schnaus, N. Araslanov, and D. Cremers, “It’s a (blind) match! Towards vision–language correspondence without parallel data,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 24983– 24992, doi: 10.1109/CVPR52734.2025.02326.

[70] M. Tjandrasuwita, C. Ekbote, L. Ziyin, and P. P. Liang, “Understanding the emergence of multimodal representation alignment,” in Proc. 42nd Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 267, 2025, pp. 59723–59760.

[71] A. E. Hoerl and R. W. Kennard, “Ridge regression: Biased estimation for nonorthogonal problems,” Technometrics, vol. 12, no. 1, pp. 55–67, Feb. 1970, doi: 10.1080/00401706.1970.10488634.

[72] P. H. Schonemann, “A generalized solution of the orthogonal Procrustes¨ problem,” Psychometrika, vol. 31, no. 1, pp. 1–10, Mar. 1966, doi: 10.1007/BF02289451.

[73] V. Maiorca, L. Moschella, A. Norelli, M. Fumero, F. Locatello, and E. Rodola, “Latent space translation via semantic alignment,” in\` Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 55394–55414, doi: 10.52202/075280-2418.

[74] G. H. Golub and C. F. Van Loan, Matrix Computations, 4th ed. Baltimore, MD, USA: Johns Hopkins University Press, 2013.

[75] I. Gulrajani and D. Lopez-Paz, “In search of lost domain generalization,” in Int. Conf. Learn. Represent., 2021. [Online]. Available: https://openreview.net/forum?id=lQdXeXDoWtI

[76] M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz, “Invariant risk minimization,” arXiv:1907.02893, 2019.

[77] D. Krueger et al., “Out-of-distribution generalization via risk extrapolation (REx),” in Proc. 38th Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 139, 2021, pp. 5815–5826.

[78] Y. Ganin and V. Lempitsky, “Unsupervised domain adaptation by backpropagation,” in Proc. 32nd Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 37, 2015, pp. 1180–1189.

[79] B. Sun and K. Saenko, “Deep CORAL: Correlation alignment for deep domain adaptation,” in Proc. Eur. Conf. Comput. Vis. Workshops, 2016, pp. 443–450, doi: 10.1007/978-3-319-49409-8 35.

[80] H. Chen, W. Xie, A. Vedaldi, and A. Zisserman, “VGGSound: A large-scale audio-visual dataset,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2020, pp. 721–725, doi: 10.1109/ICASSP40776.2020.9053174.

[81] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, “Indoor segmentation and support inference from RGBD images,” in Proc. Eur. Conf. Comput. Vis., 2012, pp. 746–760, doi: 10.1007/978-3-642-33715-4 54.

[82] P. Maheshwari et al., “AnyThermal: Towards learning universal representations for thermal perception,” in Proc. IEEE Int. Conf. Robot. Autom., 2026. [Online]. Available: https://arxiv.org/abs/2602.06203

[83] K. Grauman et al., “Ego4D: Around the world in 3,000 hours of egocentric video,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 18973–18990, doi: 10.1109/CVPR52688.2022.01842.

[84] A. Brunetto, S. Hornauer, S. X. Yu, and F. Moutarde, “The audio-visual BatVision dataset for research on sight and sound,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., 2023, pp. 1–8, doi: 10.1109/IROS55552.2023.10341715.

[85] F. R. Valverde, J. V. Hurtado, and A. Valada, “There is more than meets the eye: Self-supervised multi-object detection and tracking with sound by distilling multimodal knowledge,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 11607–11616, doi: 10.1109/CVPR46437.2021.01144.

[86] C. Chen, R. Jafari, and N. Kehtarnavaz, “UTD-MHAD: A multimodal dataset for human action recognition utilizing a depth camera and a wearable inertial sensor,” in Proc. IEEE Int. Conf. Image Process., 2015, pp. 168–172, doi: 10.1109/ICIP.2015.7350781.

[87] C. Lee et al., “Caltech aerial RGB-thermal dataset in the wild,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 236–256, doi: 10.1007/978-3-031- 73036-8 14.

[88] K. Soomro, A. R. Zamir, and M. Shah, “UCF101: A dataset of 101 human actions classes from videos in the wild,” Center for Research in Computer Vision, Univ. Central Florida, Orlando, FL, USA, Tech. Rep. CRCV-TR-12-01, Nov. 2012.

[89] J. Xu, T. Mei, T. Yao, and Y. Rui, “MSR-VTT: A large video description dataset for bridging video and language,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 5288–5296, doi: 10.1109/CVPR.2016.571.

[90] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv:1807.03748, 2018.

[91] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Int. Conf. Learn. Represent., 2019.

## Appendix

A Source Model, Knowledge Definition, 17   
and Information Interface   
A.1 Latent-State Standardization and Factor 17   
Model   
A.2 Source-Induced Knowledge and Target 17   
Residuals   
A.3 Invariance and the Observable 17   
Information Interface   
B Hub Visibility and the Identifiable 17   
Relation   
B.1 Hub-Visible State and Relation 17   
B.2 Exact Recovery and Non-Identifiability 18   
B.3 Hub-Capacity Bound 18   
C Finite-Sample Estimation, 18   
Regularization, and Rank Roles   
C.1 Edge-Wise Moment Estimation 18   
C.2 Regularized Relation and Standardized 18   
Carrier   
C.3 Rank Selection and Hub-Capacity 19   
Safeguards   
D Carrier Readout and Gaussian Evidence 19   
D.1 Optimal Carrier Directions 19   
D.2 Gaussian Evidence Expansion 19   
E Candidate Resolution, Calibration, and 19   
Score Formation   
E.1 Orthogonal Candidate-Resolution Score 20   
E.2 Source-Derived Reliability Gate 20   
E.3 Mismatch Calibration and Final Score 20   
F Score Blocks, Rankings, Algorithm, and 20   
Complexity   
F.1 Blockwise Scoring and Bidirectional 20   
Rankings   
F.2 Execution Phases and Complexity 20   
G Classification Readout, Datasets, and 21   
Results   
G.1 Classification Protocol and Metrics 21   
G.2 Datasets, Splits, and Class Definitions 21   
G.3 Classification Eligibility 22   
G.4 Retrieval and Prototype Classification 22   
Under a Single Fitted State   
H Retrieval Protocol and Reproducibility 22   
H.1 Detailed Retrieval Protocol 23   
H.2 Efficiency Measurement and Outcome 24   
Checks   
H.3 Seed Variability of Trainable Baselines 25   
H.4 Excluded Degenerate Unit 25   
Additional Retrieval Results 25   
I.1 Direction-Wise Retrieval Consistency 25   
I.2 Recovery Relative to Target-Paired 25   
Training   
I.3 Robustness to Aggregation and Chance 25   
Level   
Mechanism and Diagnostic Analyses 26   
J.1 Complete Aggregate Intervention 26   
Statistics   
J.2 Complete Component-Ablation 26   
Visualization   
J.3 Matched Shared-Row and 26   
Disjoint-Source Control   
J.4 Correspondence Resolution under 27   
Group-Preserving Permutations   
Dataset-Cluster Statistics and 27   
Robustness   
K.1 Estimands, Resampling, and Complete 27   
Families   
K.2 Component Responsibilities: Definitions 28   
and Complete Results   
K.3 NYUv2 Evaluation Protocol and 30   
Cross-Dataset Robustness

## APPENDIX A SOURCE MODEL, KNOWLEDGE DEFINITION, AND INFORMATION INTERFACE

Recall. Definition 1 introduces ${ { \kappa } _ { A B } }$ through sourceconditioned means. Assumption 1 specifies the second-order model used to read this relation through the hub. This appendix derives the factor form, separates source-induced knowledge from target-residual dependence, and states the observable information interface.

## A.1 Latent-State Standardization and Factor Model

Let $\mathbf { L } ^ { ( 0 ) }$ have mean $\pmb { \mu } _ { L }$ and covariance $C _ { L }$ . After restricting $C _ { L }$ to its support when necessary, define the standardized state

$$
{ \bf L } = C _ { L } ^ { - 1 / 2 } ( { \bf L } ^ { ( 0 ) } - { \pmb \mu } _ { L } ) , \qquad G _ { m } = G _ { m } ^ { ( 0 ) } C _ { L } ^ { 1 / 2 }\tag{19}
$$

The standardized state satisfies $\mathbb { E } [ \mathbf { L } ] = 0$ and $\mathrm { C o v } ( \mathbf { L } ) = I _ { q }$ Under the conditional-mean model

$$
\begin{array} { r l } & { \mathbf { A } = \pmb { \mu } _ { A } + G _ { A } \mathbf { L } + \pmb { \varepsilon } _ { A } , } \\ & { \mathbf { H } = \pmb { \mu } _ { H } + G _ { H } \mathbf { L } + \pmb { \varepsilon } _ { H } , } \\ & { \mathbf { B } = \pmb { \mu } _ { B } + G _ { B } \mathbf { L } + \pmb { \varepsilon } _ { B } , } \end{array}\tag{20}
$$

the residual conditions used by HSA are

$$
\begin{array} { r } { \mathbb { E } [ \varepsilon _ { m } \mid \mathbf { L } ] = 0 , \qquad } \\ { \mathrm { C o v } ( \varepsilon _ { A } , \varepsilon _ { H } ) = 0 , \qquad \mathrm { C o v } ( \varepsilon _ { H } , \varepsilon _ { B } ) = 0 . } \end{array}\tag{21}
$$

## A.2 Source-Induced Knowledge and Target Residuals

The conditional means are $\mathbb { E } [ \mathbf { A } \ | \ \mathbf { L } ] = \mu _ { A } + G _ { A } \mathbf { L }$ and $\mathbb { E } [ \mathbf { B } \mid \mathbf { L } ] = \pmb { \mu } _ { B } + G _ { B } \mathbf { L }$ . Hence Definition 1 gives

$$
\begin{array} { r } { \mathcal { K } _ { A B } = \mathrm { C o v } ( G _ { A } \mathbf { L } , G _ { B } \mathbf { L } ) = G _ { A } \mathrm { C o v } ( \mathbf { L } ) G _ { B } ^ { \top } = G _ { A } G _ { B } ^ { \top } . } \end{array}\tag{22}
$$

The observed target cross-covariance expands separately as

$$
\begin{array} { r l } & { \Sigma _ { A B } = \mathbb { E } \big [ ( G _ { A } \mathbf { L } + \varepsilon _ { A } ) ( G _ { B } \mathbf { L } + \varepsilon _ { B } ) ^ { \top } \big ] } \\ & { \qquad = G _ { A } \mathbb { E } [ \mathbf { L } \mathbf { L } ^ { \top } ] G _ { B } ^ { \top } + G _ { A } \mathbb { E } [ \mathbf { L } \varepsilon _ { B } ^ { \top } ] } \\ & { \qquad + \mathbb { E } [ \varepsilon _ { A } \mathbf { L } ^ { \top } ] G _ { B } ^ { \top } + \mathbb { E } [ \varepsilon _ { A } \varepsilon _ { B } ^ { \top } ] } \\ & { \qquad = \mathcal { K } _ { A B } + \Omega _ { A B } , \qquad \Omega _ { A B } : = \mathrm { C o v } ( \varepsilon _ { A } , \varepsilon _ { B } ) . } \end{array}\tag{23}
$$

Conditional residual centering eliminates the two latent– residual terms. Assumption 1 leaves $\Omega _ { A B }$ unrestricted because the target edge is unavailable to HSA. The two observededge residual conditions then yield $\Sigma _ { A H } ~ = ~ G _ { A } G _ { H } ^ { \top }$ and $\Sigma _ { H B } = G _ { H } G _ { B } ^ { \top }$

## A.3 Invariance and the Observable Information Interface

For any orthogonal matrix $R \in \mathbb { R } ^ { q \times q }$ , define $\mathbf { L } ^ { \prime } = R \mathbf { L }$ and $G _ { m } ^ { \prime } = \dot { G _ { m } } R ^ { \intercal }$ . Then

$$
G _ { A } ^ { \prime } G _ { B } ^ { \prime \top } = { G _ { A } } R ^ { \top } ( G _ { B } R ^ { \top } ) ^ { \top } = { G _ { A } } R ^ { \top } R G _ { B } ^ { \top } = { G _ { A } } G _ { B } ^ { \top } .\tag{24}
$$

Hence ${ \kappa } _ { A B }$ depends on the relation expressed in the frozen target spaces and is invariant under orthogonal changes of basis in the latent state.

Fitting uses two paired hub-edge datasets, $\begin{array} { r l } { \mathcal { D } _ { A H } } & { { } = } \end{array}$ $\{ ( \mathbf { a } _ { i } , \mathbf { h } _ { i } ^ { \bar { A } H } ) \} _ { i = 1 } ^ { n _ { A H } }$ and $\mathcal { D } _ { H B } = \{ ( \mathbf { h } _ { j } ^ { H B } , \mathbf { b } _ { j } ) \} _ { j = 1 } ^ { n _ { H B } }$ . The datasets may use disjoint source instances and different sample sizes.

With synchronized observations, one source row may instead contribute one pair to each dataset. Both sampling designs induce the same population information interface:

$$
\mathfrak { S } _ { H } = \{ \pmb { \mu } _ { A } , \pmb { \mu } _ { H } , \pmb { \mu } _ { B } , \Sigma _ { A A } , \Sigma _ { A H } , \Sigma _ { H H } , \Sigma _ { H B } , \Sigma _ { B B } \} .\tag{25}
$$

Its empirical counterpart contains the two edge-moment systems and target marginals. Moment estimation does not use cross-dataset row identities. The interface excludes $n ^ { - 1 } \sum _ { i } \widetilde { \mathbf { a } } _ { i } \widetilde { \mathbf { b } } _ { i } ^ { \top }$ as well as any $A { - } B$ training loss or targetinformed rank selection. The population construction permits $n _ { A H }$ and $n _ { H B }$ to differ. The reported finite-sample protocol enforces equal per-edge counts so that the target marginals support the derangement calibration in Appendix E.

## APPENDIX B

## HUB VISIBILITY AND THE IDENTIFIABLE RELATION

Recall. Theorem 1 characterizes the latent multimodal knowledge identified by the two-hub-edge interface and states its exact-recovery boundary.

## B.1 Hub-Visible State and Relation

Proof of Theorem 1. From (20)–(21),

$$
\begin{array} { l l } { { \Sigma _ { A H } = G _ { A } G _ { H } ^ { \top } , } } & { { \Sigma _ { H B } = G _ { H } G _ { B } ^ { \top } , } } \\ { { \Sigma _ { H H } = G _ { H } G _ { H } ^ { \top } + \Psi _ { H } . } } & { { } } \end{array}\tag{26}
$$

Because $\mathrm { C o v } ( \mathbf { L } , \mathbf { H } ) = G _ { H } ^ { \top }$ , the best linear predictor of the centered state from the centered hub is

$$
\mathbf { L } ^ { H } = G _ { H } ^ { \top } \Sigma _ { H H } ^ { \dag } ( \mathbf { H } - \pmb { \mu } _ { H } ) .\tag{27}
$$

Symmetry of $\Sigma _ { H H } ^ { \dagger }$ and the identity $\Sigma _ { H H } ^ { \dagger } \Sigma _ { H H } \Sigma _ { H H } ^ { \dagger } = \Sigma _ { H H } ^ { \dagger }$ imply

$$
\begin{array} { r l } & { \mathrm { C o v } ( { \mathbf { L } ^ { H } } ) = G _ { H } ^ { \top } { \boldsymbol { \Sigma } } _ { H H } ^ { \dagger } { \boldsymbol { \Sigma } } _ { H H } { \boldsymbol { \Sigma } } _ { H H } ^ { \dagger } G _ { H } } \\ & { \qquad = G _ { H } ^ { \top } { \boldsymbol { \Sigma } } _ { H H } ^ { \dagger } G _ { H } = : { \mathcal { P } } _ { H } , } \\ & { \mathrm { C o v } ( { \mathbf { L } } , { \mathbf { L } ^ { H } } ) = \mathrm { C o v } ( { \mathbf { L } } , { \mathbf { H } } ) { \boldsymbol { \Sigma } } _ { H H } ^ { \dagger } G _ { H } = { \mathcal { P } } _ { H } . } \end{array}\tag{28}
$$

(29)

For ${ \bf L } ^ { \perp H } = { \bf L } - { \bf L } ^ { H }$

$$
\mathrm { C o v } ( \mathbf { L } ^ { \perp H } ) = I _ { q } - \mathcal { P } _ { H } - \mathcal { P } _ { H } + \mathcal { P } _ { H } = I _ { q } - \mathcal { P } _ { H } .\tag{30}
$$

Therefore $\mathcal { P } _ { H } \succeq 0$ and $I _ { q } - \mathcal { P } _ { H } \succeq 0$ , yielding $0 \preceq \mathcal { P } _ { H } \preceq I _ { q }$ The best linear target predictions are

$$
\begin{array} { r } { \mathbf { A } ^ { H } = \Sigma _ { A H } \Sigma _ { H H } ^ { \dagger } ( \mathbf { H } - \pmb { \mu } _ { H } ) = G _ { A } \mathbf { L } ^ { H } , } \\ { \mathbf { B } ^ { H } = \Sigma _ { B H } \Sigma _ { H H } ^ { \dagger } ( \mathbf { H } - \pmb { \mu } _ { H } ) = G _ { B } \mathbf { L } ^ { H } . } \end{array}\tag{31}
$$

Their cross-covariance has equivalent target-space and observable-moment forms:

$$
\begin{array} { r l } & { \mathrm { C o v } ( \mathbf { A } ^ { H } , \mathbf { B } ^ { H } ) = G _ { A } \mathrm { C o v } ( \mathbf { L } ^ { H } ) G _ { B } ^ { \top } = G _ { A } \mathcal { P } _ { H } G _ { B } ^ { \top } } \\ & { \quad \quad \quad = \boldsymbol { \Sigma } _ { A H } \boldsymbol { \Sigma } _ { H H } ^ { \dagger } \boldsymbol { \Sigma } _ { H H } \boldsymbol { \Sigma } _ { H H } ^ { \dagger } \boldsymbol { \Sigma } _ { H B } } \\ & { \quad \quad \quad = \boldsymbol { \Sigma } _ { A H } \boldsymbol { \Sigma } _ { H H } ^ { \dagger } \boldsymbol { \Sigma } _ { H B } = K _ { A B } ^ { H } . } \end{array}\tag{32}
$$

## B.2 Exact Recovery and Non-Identifiability

The population knowledge relation decomposes as

$$
\mathcal { K } _ { A B } = \underbrace { G _ { A } \mathcal { P } _ { H } G _ { B } ^ { \top } } _ { \mathcal { K } _ { A B } ^ { H } } + \underbrace { G _ { A } ( I _ { q } - \mathcal { P } _ { H } ) G _ { B } ^ { \top } } _ { \mathcal { K } _ { A B } ^ { \mathrm { r e m } } } .\tag{33}
$$

The second term equals $\operatorname { C o v } ( { G _ { A } } \mathbf { L } ^ { \perp H } , { G _ { B } } \mathbf { L } ^ { \perp H } )$ and represents the shared-source relation beyond the hub’s linear resolution. Consequently, ${ \mathcal { K } } _ { A B } ^ { H } = { \mathcal { K } } _ { A B }$ holds exactly when $G _ { A } ( I _ { q } - \mathcal { P } _ { H } ) G _ { B } ^ { \top } = 0$

The formula $K _ { A B } ^ { H } \ = \ \Sigma _ { A H } \Sigma _ { H H } ^ { \dagger } \Sigma _ { H B }$ is a deterministic functional of ${ \mathfrak { S } } _ { H }$ , which establishes its identifiability. To show the boundary for ${ { \kappa } _ { A B } }$ , fix $\alpha , \beta \in ( - 1 , 1 )$ and consider

$$
\Gamma ( t ) = \left[ \begin{array} { c c c } { { 1 } } & { { \alpha } } & { { t } } \\ { { \alpha } } & { { 1 } } & { { \beta } } \\ { { t } } & { { \beta } } & { { 1 } } \end{array} \right] , \qquad | t - \alpha \beta | \leq \sqrt { ( 1 - \alpha ^ { 2 } ) ( 1 - \beta ^ { 2 } ) } .\tag{34}
$$

The stated interval is exactly the positive-semidefinite condition obtained from the Schur complement because det $\Gamma ( t ) =$ $( 1 - \alpha ^ { 2 } ) ( 1 - \beta ^ { 2 } ) - ( t - \alpha \beta ) ^ { 2 }$ . For each admissible t, choose a square root $C _ { t } C _ { t } ^ { \top } ~ = ~ \Gamma ( t )$ , draw $\mathbf { L } _ { t } ~ \sim ~ \mathcal { N } ( 0 , I _ { 3 } )$ , and set $[ A , H , B ] ^ { \top } = \dot { C } _ { t } \mathbf { L } _ { t }$ with zero residuals. Every resulting model satisfies Assumption 1 and has the same means and unit marginals. Each also has $\Sigma _ { A H } = \alpha , \Sigma _ { H H } = 1$ , and $\begin{array} { r } { \Sigma _ { H B } = \beta _ { ; } } \end{array}$ so all models expose the same ${ \mathfrak { S } } _ { H }$ . Yet Definition 1 gives ${ \kappa } _ { A B } = t$ , which varies across the interval, while the identified component remains $K _ { A B } ^ { H } = \alpha \beta$ . Therefore the interface generally underidentifies the complete latent knowledge relation. □

When $\Psi _ { H } = 0 , \mathcal { P } _ { H } = G _ { H } ^ { \top } ( G _ { H } G _ { H } ^ { \top } ) ^ { \dagger } G _ { H }$ is the orthogonal projector onto the row space of $G _ { H } . \ { \mathrm { H } } \ G _ { H }$ has full column rank, this row space is R<sup>q</sup>. Consequently, $\mathcal { P } _ { H } ~ = ~ I _ { q }$ and ${ \ K } _ { A B } ^ { H } = { \ K } _ { A B }$ . With $\Psi _ { H } \succeq 0 .$ , the eigenvalues of $\mathcal { P } _ { H }$ lie in $[ 0 , 1 ]$ and quantify soft linear visibility.

## B.3 Hub-Capacity Bound

Proof of Corollary 1. The rank inequality for a matrix product gives

$$
\mathrm { r a n k } ( { K _ { A B } ^ { H } } ) \le \mathrm { r a n k } ( \Sigma _ { H H } ^ { \dag } ) = \mathrm { r a n k } ( \Sigma _ { H H } ) .\tag{35}
$$

If H has support on at most $N _ { H }$ distinct vectors, its centered support spans at most $N _ { H } - 1$ dimensions. The range of $\Sigma _ { H H }$ lies within this span. Therefore, rank $\left( \Sigma _ { H H } \right) \leq N _ { H } - 1$ and the result follows. □

## APPENDIX C FINITE-SAMPLE ESTIMATION, REGULARIZATION, AND RANK ROLES

Recall. Eq. (8) estimates the hub-readable relation and decomposes its standardized form. This appendix specifies the edge statistics, regularization, null rule, and empirical capacity rule used by HSA.

## C.1 Edge-Wise Moment Estimation

For one hub edge represented by matrices X, $H \in \mathbb { R } ^ { n \times d }$ let

$$
{ \widehat { \mu } } _ { X } = { \frac { 1 } { n } } \sum _ { i } \mathbf { x } _ { i } , \qquad { \widehat { \mu } } _ { H } = { \frac { 1 } { n } } \sum _ { i } \mathbf { h } _ { i } ,\tag{36}
$$

and define

$$
\begin{array} { l } { { \displaystyle \widehat { \Sigma } _ { X X } = \frac { 1 } { n } \sum _ { i } ( { \bf x } _ { i } - \widehat { \mu } _ { X } ) ( { \bf x } _ { i } - \widehat { \mu } _ { X } ) ^ { \top } } , } \\ { { \displaystyle \widehat { \Sigma } _ { X H } = \frac { 1 } { n } \sum _ { i } ( { \bf x } _ { i } - \widehat { \mu } _ { X } ) ( { \bf h } _ { i } - \widehat { \mu } _ { H } ) ^ { \top } } , } \\ { { \displaystyle \widehat { \Sigma } _ { H H } = \frac { 1 } { n } \sum _ { i } ( { \bf h } _ { i } - \widehat { \mu } _ { H } ) ( { \bf h } _ { i } - \widehat { \mu } _ { H } ) ^ { \top } } . } \end{array}\tag{37}
$$

The covariance marginals are symmetrized numerically. HSA estimates one such system for each observed hub edge and pools the two hub covariances as

$$
\begin{array} { r } { \overline { { \Sigma } } _ { H H } = \frac { 1 } { 2 } ( \widehat { \Sigma } _ { H H } ^ { A } + \widehat { \Sigma } _ { H H } ^ { B } ) . } \end{array}\tag{38}
$$

When both edges reuse identical hub rows, the two empirical hub covariances coincide. With disjoint source rows, they are estimated separately and may differ at finite sample size. Assumption 1 requires both to estimate compatible population hub statistics.

## C.2 Regularized Relation and Standardized Carrier

For any d-dimensional covariance C, the trace-scaled ridge is

$$
\rho ( C ) = \lambda { \frac { \mathrm { t r } ( C ) } { d } } , \qquad \lambda = 0 . 5 .\tag{39}
$$

HSA evaluates

$$
\widehat { \mathcal { K } } _ { A B } ^ { H } = \widehat { \Sigma } _ { A H } ( \overline { { \Sigma } } _ { H H } + \rho ( \overline { { \Sigma } } _ { H H } ) I ) ^ { - 1 } \widehat { \Sigma } _ { H B }\tag{40}
$$

using a linear solve. Its raw singular-value decomposition is

$$
\widehat { K } _ { A B } ^ { H } = U _ { R } \mathrm { d i a g } ( \eta _ { 1 } , \dots , \eta _ { r } ) V _ { R } ^ { \top } .\tag{41}
$$

Shared-row and disjoint-source estimation target the same regularized population composition under Assumption 1. Their finite-sample difference arises from cross-edge diagonal terms when both edges use the same source instances. These terms can transmit target-residual dependence through the hub quadratic form; disjoint estimation removes their cooccurrence. The matched-budget control in Appendix J.3 quantifies this effect on retrieval.

For $X \in \{ A , B \}$ , eigendecompose $ { \widehat { \Sigma } } _ { X X } ~ + ~ \rho _ { X } I ~ = ~$ $E _ { X } \mathrm { d i a g } ( \nu _ { X , j } ) E _ { X } ^ { \top }$ and set

$$
W _ { X } = E _ { X } \mathrm { d i a g } \Big ( \mathrm { m a x } ( \nu _ { X , j } , 1 0 ^ { - 8 } ) ^ { - 1 / 2 } \Big ) E _ { X } ^ { \top } .\tag{42}
$$

Then

$$
\begin{array} { r } { T _ { H } = W _ { A } \widehat { K } _ { A B } ^ { H } W _ { B } ^ { \top } = U _ { C } \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { r } ) V _ { C } ^ { \top } . } \end{array}\tag{43}
$$

## C.3 Rank Selection and Hub-Capacity Safeguards

To obtain $k _ { R } ,$ , seed 142 generates one random permutation $\pi _ { R }$ that reorders the left hub rows relative to A. Fixed points are allowed. The resulting null cross-covariance is

$$
\widehat { \boldsymbol { \Sigma } } _ { A H } ^ { \pi _ { R } } = \frac { 1 } { n } \sum _ { i } ( \mathbf { a } _ { i } - \widehat { \mu } _ { A } ) ( \mathbf { h } _ { \pi _ { R } ( i ) } - \widehat { \mu } _ { H } ) ^ { \top } .\tag{44}
$$

The observed target whiteners are retained to form

$$
\begin{array} { r } { T _ { H } ^ { \pi _ { R } } = W _ { A } \widehat { \Sigma } _ { A H } ^ { \pi _ { R } } ( \overline { { \Sigma } } _ { H H } + \rho ( \overline { { \Sigma } } _ { H H } ) I ) ^ { - 1 } \widehat { \Sigma } _ { H B } W _ { B } ^ { \top } . } \end{array}\tag{45}
$$

With $\sigma _ { 1 , \mathrm { n u l l } } = \sigma _ { 1 } ( T _ { H } ^ { \pi _ { R } } )$

$$
\begin{array} { r l r } {  { k _ { R } = \operatorname* { m a x } \{ 1 , \operatorname* { m i n } ( k _ { \operatorname* { m a x } } , r , \sum _ { j = 1 } ^ { r } { \bf 1 } [ \sigma _ { j } ^ { 2 } > \kappa \sigma _ { 1 , \mathrm { n u l l } } ^ { 2 } ] ) \} , } } \\ & { } & { \kappa = 2 , \qquad k _ { \mathrm { m a x } } = 2 5 6 . } \end{array}\tag{46}
$$

This rank determines how many columns of $U _ { R }$ and $V _ { R }$ define the raw subspace removed during candidate resolution.

The carrier coordinate count follows a separate rule motivated by Corollary 1. Quantize hub rows by $q ( \mathbf { h } ) \quad = \quad$ round(10<sup>3</sup>h), count distinct states $N _ { H } ^ { A } , N _ { H } ^ { B }$ , and define

$$
\begin{array} { c } { { g ( N ) = \operatorname* { m a x } \{ 1 , \operatorname* { m i n } ( k _ { \operatorname* { m a x } } , N - 1 , d ) \} , } } \\ { { k _ { H } = \operatorname* { m i n } \{ g ( N _ { H } ^ { A } ) , g ( N _ { H } ^ { B } ) , r \} . } } \end{array}\tag{47}
$$

For genuinely distinct hub states, Corollary 1 gives the exact $N - 1$ bound. Eq. (47) counts rounded states, whereas the covariance and singular directions use unrounded embeddings. Its state count is therefore an empirical proxy for structural capacity. The outer maximum retains one coordinate when $N = 1$ , and the remaining minima enforce the limits $k _ { \operatorname* { m a x } } , d ,$ and r. Thus $k _ { R }$ sets the null-resolved dimension of the raw projector, whereas $k _ { H }$ is a separately motivated heuristic cap on carrier coordinates.

## APPENDIX D

## CARRIER READOUT AND GAUSSIAN EVIDENCE

Recall. Proposition 1 selects the paired directions that retain the greatest standardized hub-readable relation strength. Eq. (11) defines the resulting sample coordinates. Eq. (12) converts these coordinates into carrier evidence. This appendix proves the variational result and derives the likelihood-ratio expansion.

## D.1 Optimal Carrier Directions

Proof of Proposition 1. Let $P \in \mathbb { R } ^ { d _ { A } \times k }$ and $Q \in \mathbb { R } ^ { d _ { B } \times k }$ satisfy $P ^ { \top } P = Q ^ { \top } Q = I _ { k }$ . The matrix $P Q ^ { \top }$ has exactly k nonzero singular values, all equal to one. Applying the von Neumann trace inequality [74] gives

$$
\mathrm { t r } ( P ^ { \top } T _ { H } Q ) = \langle T _ { H } , P Q ^ { \top } \rangle _ { F } \leq \sum _ { j = 1 } ^ { k } \sigma _ { j } ( T _ { H } ) = \sum _ { j = 1 } ^ { k } \sigma _ { j } .\tag{48}
$$

Choosing $P ~ = ~ U _ { C , k }$ and $Q \ = \ V _ { C , k }$ attains equality and proves (9). Mapping the maximizing directions back through the target whiteners gives

$$
\begin{array} { r l } & { ( \Phi _ { A } ^ { ( k ) } ) ^ { \top } \widehat { \mathcal { K } } _ { A B } ^ { H } \Phi _ { B } ^ { ( k ) } = U _ { C , k } ^ { \top } W _ { A } \widehat { \mathcal { K } } _ { A B } ^ { H } W _ { B } ^ { \top } V _ { C , k } } \\ & { \qquad = U _ { C , k } ^ { \top } T _ { H } V _ { C , k } = \mathrm { d i a g } ( \sigma _ { 1 } , \ldots , \sigma _ { k } ) , } \end{array}\tag{49}
$$

which proves (10).

HSA takes $k = k _ { H }$ and defines

$$
\begin{array} { l l } { { \Phi _ { A } = W _ { A } U _ { C , k _ { H } } , } } & { { \Phi _ { B } = W _ { B } V _ { C , k _ { H } } , } } \\ { { { \bf z } ^ { A } = \Phi _ { A } ^ { \top } ( { \bf a } - \widehat { \mu } _ { A } ) , } } & { { { \bf z } ^ { B } = \Phi _ { B } ^ { \top } ( { \bf b } - \widehat { \mu } _ { B } ) . } } \end{array}\tag{50}
$$

At population level,

$$
\mathbf { Z } ^ { A } = \Phi _ { A } ^ { \top } G _ { A } \mathbf { L } + \Phi _ { A } ^ { \top } \varepsilon _ { A } , \qquad \mathbf { Z } ^ { B } = \Phi _ { B } ^ { \top } G _ { B } \mathbf { L } + \Phi _ { B } ^ { \top } \varepsilon _ { B } .\tag{51}
$$

The Eckart–Young–Mirsky theorem [74] also gives

$$
U _ { C , k _ { H } } \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { k _ { H } } ) V _ { C , k _ { H } } ^ { \top } \in \underset { \mathrm { r a n k } ( R ) \leq k _ { H } } { \arg \operatorname* { m i n } } \| T _ { H } - R \| _ { F } .
$$

□

(52)

D.2 Gaussian Evidence Expansion

Set $c _ { j } \ = \ \mathrm { c l i p } ( \sigma _ { j } , 0 , 1 \ - - \ 1 0 ^ { - 6 } )$ . The working covariance matrices for matched and mismatched values of coordinate j are

$$
C _ { j } ^ { + } = \left[ \begin{array} { l l } { 1 } & { c _ { j } } \\ { c _ { j } } & { 1 } \end{array} \right] , \qquad C _ { j } ^ { - } = I _ { 2 } .\tag{53}
$$

The determinant and inverse required by the Gaussian density are

$$
| C _ { j } ^ { + } | = 1 - c _ { j } ^ { 2 } , \qquad ( C _ { j } ^ { + } ) ^ { - 1 } = \frac { 1 } { 1 - c _ { j } ^ { 2 } } \left[ \begin{array} { l l } { { 1 } } & { { - c _ { j } } } \\ { { - c _ { j } } } & { { 1 } } \end{array} \right] .\tag{54}
$$

For $\mathbf { u } = [ x , y ] ^ { \top }$ , the shared normalizing constants of the two zero-mean Gaussian densities cancel, leaving the determinant term:

$$
\begin{array} { l } { \displaystyle \ell _ { j } ( x , y ; c _ { j } ) = \log \frac { p ( { \bf u } \mid C _ { j } ^ { + } ) } { p ( { \bf u } \mid I _ { 2 } ) } } \\ { \displaystyle = - \frac { 1 } { 2 } \log ( 1 - c _ { j } ^ { 2 } ) - \frac { 1 } { 2 } { \bf u } ^ { \top } [ ( C _ { j } ^ { + } ) ^ { - 1 } - I _ { 2 } ] { \bf u } . } \end{array}\tag{55}
$$

Furthermore,

$$
( C _ { j } ^ { + } ) ^ { - 1 } - I _ { 2 } = \frac { 1 } { 1 - c _ { j } ^ { 2 } } \left[ { c _ { j } ^ { 2 } } - { c _ { j } } \right] ,\tag{56}
$$

so expansion of the quadratic form yields

$$
\ell _ { j } ( x , y ; c _ { j } ) = \frac { c _ { j } x y } { 1 - c _ { j } ^ { 2 } } - \frac { c _ { j } ^ { 2 } ( x ^ { 2 } + y ^ { 2 } ) } { 2 ( 1 - c _ { j } ^ { 2 } ) } - \frac { 1 } { 2 } \log ( 1 - c _ { j } ^ { 2 } ) .\tag{57}
$$

Summing (57) over $j = 1 , \dots , k _ { H }$ gives the carrier score in Eq. (12). The Gaussian working model provides a closed-form evidence score in the standardized readout coordinates, leaving the full embedding distribution unrestricted.

## APPENDIX E

Recall. Eq. (13) constructs candidate-resolution coordinates orthogonal to the leading raw relation subspace. Eq. (15b) combines their score with carrier evidence after mismatch calibration.

## E.1 Orthogonal Candidate-Resolution Score

From the raw decomposition (41), define

$$
\Pi _ { A } ^ { R } = U _ { R , k _ { R } } U _ { R , k _ { R } } ^ { \top } , \qquad \Pi _ { B } ^ { R } = V _ { R , k _ { R } } V _ { R , k _ { R } } ^ { \top } .\tag{58}
$$

The candidate-resolution coordinates and score are

$$
\begin{array} { r } { \mathbf { r } ^ { A } ( \mathbf { a } ) = \displaystyle \frac { ( I - \Pi _ { A } ^ { R } ) ( \mathbf { a } - \widehat { \mu } _ { A } ) } { \operatorname* { m a x } ( \| ( I - \Pi _ { A } ^ { R } ) ( \mathbf { a } - \widehat { \mu } _ { A } ) \| _ { 2 } , 1 0 ^ { - 1 2 } ) } , } \\ { \mathbf { r } ^ { B } ( \mathbf { b } ) = \displaystyle \frac { ( I - \Pi _ { B } ^ { R } ) ( \mathbf { b } - \widehat { \mu } _ { B } ) } { \operatorname* { m a x } ( \| ( I - \Pi _ { B } ^ { R } ) ( \mathbf { b } - \widehat { \mu } _ { B } ) \| _ { 2 } , 1 0 ^ { - 1 2 } ) } , } \\ { s _ { R } ( \mathbf { a } , \mathbf { b } ) = \mathbf { r } ^ { A } ( \mathbf { a } ) ^ { \top } \mathbf { r } ^ { B } ( \mathbf { b } ) . } \end{array}\tag{59}
$$

## E.2 Source-Derived Reliability Gate

For each observed hub edge, define the ridge-regularized covariance matrices as

$$
\begin{array} { r } { \widehat { \Omega } _ { A } = \widehat { \Sigma } _ { A A } - \widehat { \Sigma } _ { A H } ( \widehat { \Sigma } _ { H H } ^ { A } + \rho _ { H } ^ { A } I ) ^ { - 1 } \widehat { \Sigma } _ { H A } , } \\ { \widehat { \Omega } _ { B } = \widehat { \Sigma } _ { B B } - \widehat { \Sigma } _ { B H } ( \widehat { \Sigma } _ { H H } ^ { B } + \rho _ { H } ^ { B } I ) ^ { - 1 } \widehat { \Sigma } _ { H B } , } \end{array}\tag{60}
$$

where $\rho _ { H } ^ { A } ~ = ~ \rho ( \widehat \Sigma _ { H H } ^ { A } )$ and $\rho _ { H } ^ { B } ~ = ~ \rho ( \widehat \Sigma _ { H H } ^ { B } )$ . Let $\begin{array} { r l } { d _ { R } } & { { } = } \end{array}$ $\| \widehat { \Omega } _ { A } \| _ { F } \| \widehat { \Omega } _ { B } \| _ { F }$ . The specified rule is

$$
\alpha _ { R } = \left\{ \begin{array} { l l } { \mathrm { c l i p } \left( \frac { \langle \widehat { \Omega } _ { A } , \widehat { \Omega } _ { B } \rangle _ { F } } { d _ { R } } , - 1 , 1 \right) , } & { d _ { R } > 1 0 ^ { - 2 0 } , } \\ { 0 , } & { d _ { R } \leq 1 0 ^ { - 2 0 } , } \end{array} \right. .\tag{61}
$$

For each of the 256 coordinate permutations generated with seed $4 2 ,$ , let $P _ { \pi }$ be the corresponding permutation matrix and compute

$$
\begin{array} { r l } & { \alpha _ { R } ^ { \pi } = \left\{ \begin{array} { l l } { \mathrm { c l i p } \left( \frac { \langle \widehat { \Omega } _ { A } , P _ { \pi } \widehat { \Omega } _ { B } P _ { \pi } ^ { \intercal } \rangle _ { F } } { d _ { R } } , - 1 , 1 \right) , } & { d _ { R } > 1 0 ^ { - 2 0 } , } \\ { 0 } & { d _ { R } \leq 1 0 ^ { - 2 0 } , } \end{array} \right. } \\ & { q _ { R } = \mathrm { Q u a n t i l e } _ { 0 . 9 5 } \left( \left\{ \alpha _ { R } ^ { \pi } : \pi \in \Pi _ { G } \right\} \right) , } \\ & { g _ { R } = \mathrm { c l i p } \left( \frac { \alpha _ { R } - q _ { R } } { \operatorname* { m a x } \left( 1 - q _ { R } , 1 0 ^ { - 1 2 } \right) } , 0 , 1 \right) . } \end{array}\tag{2}
$$

Each coordinate permutation preserves the spectrum of the regularized covariance matrix while disrupting coordinate-wise agreement between the target spaces. The resolution-reliability gate is zero when the observed affinity does not exceed the 95th percentile of this null distribution. Otherwise, it rescales the excess over $q _ { R }$ from $[ q _ { R } , 1 ]$ to [0, 1]. Candidate-resolution evidence therefore enters the ranking according to cross-modal agreement estimated entirely from the hub edges.

## E.3 Mismatch Calibration and Final Score

The reported protocol supplies equal-size target-side training marginals, denoted by $\{ { \bf { a } } _ { i } \} _ { i = 1 } ^ { n }$ and $\{ { \bf b } _ { i } \} _ { i = 1 } ^ { n }$ . Synchronized runs inherit these counts from common source rows, while the disjoint-source control uses matched splits. The common index supplies rows for mismatch construction, while the two hub-edge moment systems remain separately estimated.

For calibration, each seed in {42, 43, 44, 45, 46} generates a derangement π, satisfying $\pi ( i ) \neq i$ for every row. For $Q \in$ {C, R},

$$
\begin{array} { l } { { \displaystyle \tau _ { Q } ^ { \pi } = \operatorname* { m a x } \bigl \{ \mathrm { S t d } _ { i = 1 } ^ { n } \bigl [ s _ { Q } \bigl ( { \bf a } _ { i } , { \bf b } _ { \pi ( i ) } \bigr ) \bigr ] , 1 0 ^ { - 6 } \bigr \} , } } \\ { { \displaystyle \tau _ { Q } = \frac { 1 } { 5 } \sum _ { \pi } \tau _ { Q } ^ { \pi } . } } \end{array}\tag{63}
$$

The final rule is

$$
s _ { \mathrm { H S A } } = \frac { s _ { C } } { \operatorname* { m a x } ( \tau _ { C } , 1 0 ^ { - 6 } ) } + g _ { R } \frac { s _ { R } } { \operatorname* { m a x } ( \tau _ { R } , 1 0 ^ { - 6 } ) } .\tag{64}
$$

Eq. (64) combines likelihood-ratio carrier evidence $s _ { C }$ with candidate-resolution evidence $s _ { R } .$ . The source-only gate g<sub>R</sub> weights the latter, and $\tau _ { C }$ and $\tau _ { R }$ calibrate the two score scales. This construction supplies the same fitted scoring rule for retrieval and prototype classification.

## APPENDIX F SCORE BLOCKS, RANKINGS, ALGORITHM, AND COMPLEXITY

Recall. Eqs. (16)–(17) turn the fitted readout into two retrieval directions. This appendix gives the block form used for efficient evaluation and summarizes the execution phases and computational complexity.

## F.1 Blockwise Scoring and Bidirectional Rankings

Let $Z _ { A } ~ \in ~ \mathbb { R } ^ { N _ { A } }$ <sup>A×kH</sup> and $Z _ { B } ~ \in ~ \mathbb { R } ^ { N _ { B } }$ <sup>×kH</sup> collect carrier coordinates. Define

$$
\begin{array} { l l } { { d _ { j } = { \displaystyle \frac { c _ { j } } { 1 - c _ { j } ^ { 2 } } } , } } & { { e _ { j } = { \displaystyle \frac { c _ { j } ^ { 2 } } { 2 ( 1 - c _ { j } ^ { 2 } ) } } , \hfill } } \\ { { \beta = - { \displaystyle \frac { 1 } { 2 } \sum _ { j = 1 } ^ { k _ { H } } } \log ( 1 - c _ { j } ^ { 2 } ) . } } & { { \hfill } } \end{array}\tag{65}
$$

Let $\mathbf { q } _ { A } = ( Z _ { A } \odot Z _ { A } ) \boldsymbol { \mathfrak { c } }$ e and $\mathbf q _ { B } = ( Z _ { B } \odot Z _ { B } ) \mathbf e$ , where ${ \mathbf { e } } =$ $[ e _ { 1 } , \ldots , e _ { k _ { H } } ] ^ { \top }$ . The full carrier block is

$$
\begin{array} { r } { S _ { C } = Z _ { A } \mathrm { d i a g } ( { \bf d } ) Z _ { B } ^ { \top } - { \bf q } _ { A } \mathbf { 1 } _ { N _ { B } } ^ { \top } - { \bf 1 } _ { N _ { A } } { \bf q } _ { B } ^ { \top } + \beta { \bf 1 } _ { N _ { A } } \mathbf { 1 } _ { N _ { B } } ^ { \top } . } \end{array}\tag{66}
$$

If $R _ { A }$ and $R _ { B }$ collect candidate-resolution coordinates, then

$$
\begin{array} { c } { { S _ { R } = R _ { A } R _ { B } ^ { \top } , } } \\ { { S ^ { A  B } = \displaystyle \frac { S _ { C } } { \tau _ { C } } + g _ { R } \displaystyle \frac { S _ { R } } { \tau _ { R } } , } } \\ { { S ^ { B  A } = ( S ^ { A  B } ) ^ { \top } . } } \end{array}\tag{67}
$$

The rankings are

$$
\begin{array} { r } { \pi _ { i } ^ { A  B } = \mathrm { a r g s o r t } _ { j } ^ { \downarrow } [ S ^ { A  B } ] _ { i j } , } \\ { \pi _ { j } ^ { B  A } = \mathrm { a r g s o r t } _ { i } ^ { \downarrow } [ S ^ { A  B } ] _ { i j } . } \end{array}\tag{68}
$$

## F.2 Execution Phases and Complexity

Algorithm 1 contains three phases. During fitting, HSA estimates edge and marginal statistics and composes $\widehat { \mathcal { K } } _ { A B } ^ { H }$ . It then performs both decompositions, selects $k _ { R }$ and $k _ { H } .$ , and fixes the carrier directions, raw projectors, regularized covariance matrices, and resolution-reliability gate. During calibration, HSA evaluates five deranged target-marginal pairings and fixes $\tau _ { C }$ and $\tau _ { R }$ . During task readout, it projects both task sets and evaluates their pair scores. Retrieval forms both score blocks and sorts both axes; prototype classification scores each query against the fixed class-prototype bank. Target test identities or labels never enter HSA fitting, calibration, or score construction. Retrieval labels are consulted only after scoring to determine hits, while classification labels are used to form prototypes and evaluate predictions under the protocol in Appendix G.1.

For n anchor rows and common dimension d, dense covariance estimation costs $O ( n d ^ { 2 } )$ time and $O ( d ^ { 2 } )$ memory. The dense solve, eigendecompositions, and singular-value decompositions cost $O ( d ^ { 3 } )$ time. Query projection costs $O ( ( N _ { A } +$ $N _ { B } ) d ( k _ { H } + k _ { R } ) )$ . The carrier and candidate-resolution score blocks cost $O ( N _ { A } N _ { B } k _ { H } )$ and $O ( N _ { A } N _ { B } d )$ , respectively. Query rows are evaluated in batches, so intermediate storage scales with batch size and avoids allocating the complete $N _ { A } N _ { B }$ score matrix.

## APPENDIX G

## CLASSIFICATION READOUT, DATASETS, AND RESULTS

This section documents the prototype-classification protocol underlying Tables II and IV, including the readout direction, class definitions, splits, eligibility criteria, and fixed-state reuse.

## G.1 Classification Protocol and Metrics

Prototype Direction. Each relation follows the modality order used in the main tables. The first modality is A and supplies one prototype $\mathbf { p } _ { c } ^ { A }$ for each class c; the second modality is B and supplies the test queries. Classification therefore evaluates the $B  A$ direction in Eq. (18). For every relation except ImageBind text–audio on VGGSound, each prototype is formed by averaging the normalized labeled training features in modality A and then normalizing the mean. The VGGSound ImageBind relation uses a fixed 80-template text-prompt bank to form its A-side class prototypes. Every compared method receives the same prototype bank and the same test queries within a relation.

Information Boundary. The main classification comparisons use the classification source features to estimate the targetmodality relation from the two hub-edge moment systems and construct the HSA readout, under the same zero-targetpair-identity boundary as retrieval. Both tasks use the same relation-estimation and readout method; Appendix G.4 further validates task readout with all parameters fixed. Class labels define the A-side prototype bank; target test outcomes do not enter carrier selection, rank selection, reliability gating, or mismatch calibration. Frozen cosine, hub-relative similarity, bidirectional ridge, bidirectional Procrustes, ReAlign, ERM, IRM, VREx, DANN, CORAL, and HSA use no A–B pair identities. ASIF, Paired-OP, and Full A–B use labeled targettraining pairs and retain the <sup>†</sup> marker. ASIF and Paired-OP participate in the boldface and underlining; only Full A–B is excluded. ERM, IRM, VREx, DANN, CORAL, and Full A–B report the mean over seeds 40–42; analytic methods use their fixed registered run.

Metric and Aggregation. Let $\mathcal { T } _ { c } ^ { \mathrm { t e } }$ be the test queries from class c. Macro Top-1 averages class-wise accuracy,

$$
\mathrm { m T 1 } = \frac { 1 } { | \mathcal { C } | } \sum _ { c \in \mathcal { C } } \frac { 1 } { | \mathcal { T } _ { c } ^ { \mathrm { t e } } | } \sum _ { i \in \mathcal { T } _ { c } ^ { \mathrm { t e } } } \mathbf { 1 } [ \widehat { y } _ { i } = c ] .\tag{69}
$$

Macro Top-1 is primary because several datasets are classimbalanced. The classification means weight the six Image-Bind relations or the five LanguageBind relations equally. Their combined mean weights all eleven relations equally. Diagnostic results do not enter these means.

## G.2 Datasets, Splits, and Class Definitions

VGGSound-309. Classification uses all 309 classes in the official VGGSound vocabulary. Its 9,258 training clips come from one cached source shard; all classes are present, with 2–232 clips per class and a median of 27. The 15,421 test clips are the intersection of the registered official test list and the available media snapshot, covering all 309 classes; 25 registered clips are absent from that snapshot. ImageBind forms fixed text-prompt prototypes. Retrieval uses the registered 100- class, 1,500/1,000 split described in Appendix H.1.

NYUv2. NYUv2 uses the official 795/654 scene split and ten scene classes: bathroom, bedroom, bookstore, classroom, dining room, home office, kitchen, living room, office, and others. Prototypes are formed from training images, and test queries are depth observations.

TartanRGBT. The four environment classes are indoor, offroad, outdoor, and urban. They are derived from the first component of each registered trajectory identifier. Training and test trajectories are disjoint. The 1,184 training samples contain 322, 458, 303, and 101 examples in these classes; the 436 test samples contain 252, 135, 21, and 28. The label park has no registered test trajectory and is excluded by the locked class rule. The same identities and split are used for all four ImageBind and LanguageBind relations.

Ego4D. Scenario metadata define seventeen fixed activity classes after requiring at least ten training and five test samples per relation and taking the class intersection across the text–inertial and audio–inertial relations: Bike mechanic; Carpenter; Cleaning/laundry; Cooking; Crafting/knitting/sewing/drawing/painting; Eating; Farmer; Gardening; Household management–caring for kids; Indoor navigation (walking); Playing board games; Playing games/video games; Playing with pets; Reading books; Watching TV; Working out at home; and jobs related to a construction or renovation company. Registered video-identity splits are applied before this fixed-class filtering.

BatVision. The seven classes are the collection locations 2nd Floor Luxembourg, 3rd Floor Luxembourg, Attic, Outdoor Cobblestone Path, Salle Chevalier, Salle des Colonnes, and V119 Cake Corridors. The registered training-plus-validation split provides 2,536 labeled samples for constructing seven class prototypes; the test split provides 584 queries. These labels support location recognition and may reflect locationspecific visual and acoustic structure.

UTD-MHAD. The 27 official actions are: right-arm swipe left;   
right-arm swipe right; right-hand wave; two-hand front clap;

TABLE XII  
CLASSIFICATION DATASET COVERAGE. PAIR ABBREVIATIONS FOLLOW THE MAIN TABLES. COUNTS GIVE PROTOTYPE-TRAINING AND TEST QUERIES. THE TWO EGO4D COUNTS FOLLOW THE DISPLAYED RELATION ORDER AND REFLECT THE EVALUATED MODALITY INTERSECTION.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Target Relations</td><td rowspan="2">Classes</td><td rowspan="2">Train</td><td rowspan="2">Test</td></tr><tr><td>ImageBind</td><td>LanguageBind</td></tr><tr><td>VGGSound-309</td><td>Tx-Au</td><td>一</td><td>309</td><td>9,258</td><td>15,421</td></tr><tr><td>NYUv2</td><td>一</td><td>Im-D</td><td>10</td><td>795</td><td>654</td></tr><tr><td>TartanRGBT</td><td>Tx-Th</td><td>Vi-Th, Vi–D Th-D</td><td>4</td><td>1,184</td><td>436</td></tr><tr><td>Ego4D</td><td>Tx-IMU</td><td></td><td>17</td><td>1,084</td><td>434</td></tr><tr><td>BatVision</td><td>Au-IMU Au-D</td><td>1</td><td>7</td><td>875 2,536</td><td>391 584</td></tr><tr><td>UTD-MHAD</td><td>D-IMU</td><td>-</td><td>27</td><td>431</td><td>430</td></tr><tr><td>MSR-VTT</td><td>一</td><td>一 Vi-Au</td><td>20</td><td>6,176</td><td>884</td></tr></table>

Total: 11 Classification Relations (6 ImageBind, 5 LanguageBind).

right-arm throw; cross arms at the chest; basketball shoot; draw X; draw a clockwise circle; draw a counter-clockwise circle; draw a triangle; bowling; front boxing; baseball swing; tennis forehand swing; two-arm curl; tennis serve; two-hand push; knock on a door; catch an object; pick up and throw; jog in place; walk in place; sit to stand; stand to sit; forward lunge; and squat. Subjects 1, 3, 5, and 7 form the 431-sample prototype split, while subjects 2, 4, 6, and 8 form the 430- sample test split.

MSR-VTT. The twenty official video categories are music, people, gaming, sports/actions, news/events/politics, education, TV shows, movie, animation, vehicles, how-to, travel, science/technology, animal, kids/family, documentary, food, cooking, beauty/fashion, and advertisement. All twenty categories occur in the registered 6,176/884 training/test split. Video prototypes classify audio queries.

## G.3 Classification Eligibility

MAVD Drive-Level Diagnostic. MAVD provides day-drive and night-drive labels, with one recording drive per class. These labels jointly encode time of day and drive identity. HSA reaches 100.00% on ImageBind audio–thermal and 76.65% on LanguageBind thermal–audio, compared with 50.00% for frozen cosine on both relations. These drive-level diagnostics are reported separately from the eleven-relation classification aggregate.

Caltech Label Availability. The Caltech Aerial RGBT cache provides timestamps and visual, thermal, and inertial features from one recording bag. Its annotations support paired retrieval but supply no semantic scene, terrain, or activity classes. Classification uses dataset-defined labels, so the thermal– inertial relation participates in retrieval evaluation only.

## G.4 Retrieval and Prototype Classification Under a Single Fitted State

The main classification tables estimate the HSA relation readout using the source features and splits specified for classification, with common prototypes and queries across methods. To test whether the same estimated relation can support both retrieval and prototype classification, we fix the complete HSA state estimated from the source hub edges: centers, whitening matrices, paired directions, ranks, reliability spectrum, source gate, and both mismatch scales. This state is identical to that used in the corresponding retrieval evaluation; classification performs only projection and scoring on the class prototypes and queries, without refitting or recalibration. Input checks cover encoder provenance, feature coordinates, and actual sample identities.

Eight of the eleven classification relations meet both fixedstate reuse criteria: compatible encoder coordinates and query identities disjoint from retrieval fitting samples. Table XIII reports the individual results and their relation-equal mean across these eight eligible relations and five datasets. NYUv2 is evaluated under its task-specific preprocessing protocols (Appendix K.3). ImageBind Tx–Th and Tx–Au provide interfacereuse diagnostics with overlapping queries; the two MAVD drive diagnostics are also reported separately.

Across the eight eligible relations, the unchanged retrieval state achieves 48.01% macro Top-1, compared with 26.46% for frozen cosine and 48.79% for HSA fitted separately for classification. The 21.56-point gain over cosine shows that the fitted relation supports class-prototype prediction as well as instance retrieval. All classification queries in this aggregate are disjoint from the retrieval fitting samples under the dataset splits in Appendix G.2; identity checks use the underlying samples across feature-extraction protocols. The separate overlap diagnostics contain 80/436 queries for ImageBind text– thermal on TartanRGBT and 89/15,421 for ImageBind text– audio on VGGSound.

## APPENDIX H

## RETRIEVAL PROTOCOL AND REPRODUCIBILITY

This section documents the datasets, information boundaries, training rules, implementation settings, seed variability, and excluded degenerate unit used by the main-paper retrieval comparisons.

TABLE XIII  
PROTOTYPE CLASSIFICATION WITH A FIXED RELATION READOUT. VALUES ARE MACRO TOP-1 (%). THE AGGREGATE USES EIGHT ELIGIBLE RELATIONS ACROSS FIVE DATASETS. OVERLAP AND DRIVE DIAGNOSTICS ARE GROUPED SEPARATELY AND EXCLUDED FROM THE MEAN. OVERLAP COUNTS USE ACTUAL SAMPLE IDENTITIES; BOLDFACE MARKS THE HIGHEST ACCURACY WITHIN EACH ROW.
<table><tr><td>Pair</td><td>Dataset</td><td>Scope</td><td>Reused HSA</td><td>Separate HSA</td><td>Frozen Cosine</td><td>Overlap</td></tr><tr><td colspan="7">Eligible Relations: Included in the Mean</td></tr><tr><td colspan="7">ImageBind</td></tr><tr><td>Au-D</td><td>BatVision</td><td>Eligible</td><td>61.40</td><td>61.40</td><td>25.77</td><td>0</td></tr><tr><td>Au-IMU</td><td>Ego4D</td><td>Eligible</td><td>16.46</td><td>22.81</td><td>6.56</td><td>0</td></tr><tr><td>D-IMU</td><td>UTD-MHAD</td><td>Eligible</td><td>34.75</td><td>34.75</td><td>2.87</td><td>0</td></tr><tr><td>Tx-IMU</td><td>Ego4D</td><td>Eligible</td><td>21.79</td><td>21.58</td><td>7.16</td><td>0</td></tr><tr><td colspan="7">LanguageBind</td></tr><tr><td>Th-D</td><td>TartanRGBT</td><td>Eligible</td><td>71.98</td><td>69.01</td><td>37.80</td><td>0</td></tr><tr><td>Vi-Au</td><td>MSR-VTT</td><td>Eligible</td><td>26.37</td><td>26.37</td><td>13.77</td><td>0</td></tr><tr><td>Vi-D</td><td>TartanRGBT</td><td>Eligible</td><td>73.76</td><td>69.99</td><td>43.68</td><td>0</td></tr><tr><td>Vi-Th</td><td>TartanRGBT</td><td>Eligible</td><td>77.61</td><td>84.37</td><td>74.03</td><td>0</td></tr><tr><td colspan="7">Mean over 8 Eligible Relations</td></tr><tr><td colspan="7">Diagnostics: Excluded from the Mean</td></tr><tr><td colspan="7">ImageBind</td></tr><tr><td>Au-Th</td><td>MAVD</td><td>Drive</td><td>100.00</td><td>100.00</td><td>50.00</td><td>0</td></tr><tr><td>Tx-Au</td><td>VGGSound</td><td>Overlap</td><td>27.99</td><td>32.60</td><td>27.41</td><td>89</td></tr><tr><td>Tx-Th</td><td>TartanRGBT</td><td>Overlap</td><td>91.90</td><td>90.92</td><td>28.10</td><td>80</td></tr><tr><td colspan="7">LanguageBind</td></tr><tr><td>Th-Au</td><td>MAVD</td><td>Drive</td><td>76.65</td><td>76.65</td><td>50.00</td><td>0</td></tr></table>

The aggregate covers five datasets. Reuse–cosine: 21.56 points [95% CI: 15.16, 29.52]; reuse–separate: −0.77 points [95% CI: −2.30, 0.00]. Intervals use dataset-cluster resampling; differences are computed before rounding.

## H.1 Detailed Retrieval Protocol

Datasets and Evaluation Relations. Each evaluation relation combines one backbone, one dataset, and one heldout target pair. ImageBind [16] uses image as its hub and contributes nine nondegenerate relations among text, audio, depth, thermal, and inertial modalities. LanguageBind [17] uses language as its hub and contributes all ten relations among image, video, audio, depth, and thermal. Across both backbones, the nineteen relations span seven modalities and ten datasets. The datasets are VGGSound [80], NYUv2 [81], TartanRGBT [82], Ego4D [83], and BatVision [84]. They also include MAVD [85], UTD-MHAD [86], Caltech Aerial RGBT [87], UCF101 [88], and MSR-VTT [89]. Table XIV reports modality coverage, positive definitions, sample counts, and relation counts.

VGGSound contributes one class-level and one pairedinstance relation. TartanRGBT contributes one trajectory-level and five paired-instance relations. Seventeen of the nineteen relations use paired-instance positives. Every query ranks the complete test gallery. The ImageBind text–depth configuration on NYUv2 contains one positive group and is excluded because every method obtains 100% Recall@10.

Data Splits. VGGSound uses class-stratified sampling, and Ego4D and Caltech Aerial RGB–Thermal use random samplelevel splits. TartanRGBT uses sample-level splits with shared trajectories for ImageBind and trajectory-disjoint splits for LanguageBind. UTD-MHAD uses subjects 1, 3, 5, and 7 for training and subjects 2, 4, 6, and 8 for testing. MAVD uses chronological splits within each drive. BatVision follows the dataset-provided splits within each location, combining the training and validation subsets for fitting. MSR-VTT follows the 7K/1K training/test annotations, retaining samples with available video and successfully decoded audio. UCF101 and NYUv2 use fixed training and test sample lists. These partitions remain unchanged across all compared methods for each relation.

Compared Methods and Information Boundary. All compared methods use all available training rows and keep the backbone frozen. For methods without target-pair supervision, A–B pair identities are masked throughout fitting, rank selection, gating, calibration, stopping, and validation. Frozen cosine compares target features directly. ReAlign [23] collects the A and B training marginals from the two hub edges, discards cross-modal row identities, and applies fixed A → B Anchor, Trace, and Centroid Alignment. Hubrelative similarity [20] compares target samples through their similarity vectors to hub samples. Bidirectional ridge [71] and Procrustes [72] independently map both targets into the hub space. ERM [75], IRM [76], VREx [77], DANN [78], and CORAL [79] use the same two-layer projection-head architecture. Each modality has a separate projection head with independent weights, and all heads share a learnable temperature parameter. These methods treat the two hub edges as training environments. HSA is the complete closed-form readout.

ASIF, Paired-OP, and Full A–B use A–B training-row identities. ASIF [19] builds a coupled dictionary from these paired rows. Paired-OP [72] uses every paired row to solve one orthogonal map per retrieval direction. Both methods take zero gradient steps. Full A–B trains the same projection architecture on target-paired data and averages seeds 40–42 under the globally selected 200-step duration. In Tables $\mathrm { I { - } I V , ~ } ^ { \dagger }$ marks ASIF, Paired-OP, and Full A–B because all three use targetpair identities. Boldface and underlining identify the best and second-best values among all methods except Full A–B. ASIF and Paired-OP participate in these markings.

TABLE XIV  
RETRIEVAL DATASET COVERAGE. TX, IM, VI, AU, D, TH, AND IMU DENOTE TEXT, IMAGE, VIDEO, AUDIO, DEPTH, THERMAL, AND INERTIAL MODALITIES.
<table><tr><td>Dataset</td><td>Modalities</td><td>Positive Definition</td><td>Train</td><td>Test</td><td>Relations</td></tr><tr><td>VGGSound (VGS)</td><td>Tx, Im, Au</td><td>Class (100), Instance</td><td>1,500</td><td>1,000</td><td>2</td></tr><tr><td>NYUv2 (NYU)</td><td>Tx, Im, D</td><td>Instance</td><td>47,584</td><td>654</td><td>1</td></tr><tr><td>TartanRGBT (TR)</td><td>Tx, Im, Vi, D, Th</td><td>Trajectory (46), Instance</td><td>1,733-8,602</td><td>436-8,601</td><td>6</td></tr><tr><td>Ego4D (E4D)</td><td>Tx, Im, Au, IMU</td><td>Instance</td><td>1,138-1,600</td><td>487-686</td><td>2</td></tr><tr><td>BatVision (BV)</td><td>Tx, Im, Au, D</td><td>Instance</td><td>2,536</td><td>584</td><td>2</td></tr><tr><td>MAVD</td><td>Tx, Im, Au, Th</td><td>Instance</td><td>647-648</td><td>432-433</td><td>2</td></tr><tr><td>UTD-MHAD (UTD)</td><td>Im, D, IMU</td><td>Instance</td><td>431</td><td>430</td><td>1</td></tr><tr><td>Caltech Aerial RGBT (CA)</td><td>Im, Th, IMU</td><td>Instance</td><td>245</td><td>245</td><td>1</td></tr><tr><td>UCF101 (UCF)</td><td>Tx, Im, Vi</td><td>Instance</td><td>7,070</td><td>3,030</td><td>1</td></tr><tr><td>MSR-VTT (MSR)</td><td>Tx, Vi, Au</td><td>Instance</td><td>6,176</td><td>884</td><td>1</td></tr><tr><td>Total</td><td></td><td>1</td><td></td><td></td><td>19</td></tr></table>

Baseline Training and Selection. For each trainable hub-edge baseline and seed, 90% of source rows fit a pilot model. The remaining 10% select the training duration using mean A– H and $H { - } B$ validation InfoNCE [90]. The loss is checked every 100 steps. After 400 steps, training stops following five checks without a 0.2% relative improvement, subject to a 4,000-step cap. The selected duration is the best validation step, with a minimum of 400 steps. Pilot weights are discarded before retraining on all source rows for the selected duration. Reported results average seeds 40, 41, and 42.

Full A–B uses one training duration across all relations and seeds. For each relation and seed in {40, 41, 42}, a deterministic 90/10 split of target-pair training rows fits a pilot. Each pilot records validation InfoNCE every 100 steps from 100 through 4,000. For candidate duration t, the selection score is the equal-relation mean of the seed-averaged log loss ratio,

$$
\mathcal { I } ( t ) = \frac { 1 } { 1 9 } \sum _ { e = 1 } ^ { 1 9 } \frac { 1 } { 3 } \sum _ { s \in \{ 4 0 , 4 1 , 4 2 \} } \log \frac { L _ { e , s } ( t ) } { L _ { e , s } ( 1 0 0 ) } .\tag{70}
$$

Minimizing $\mathcal { I } ( t )$ selects 200 steps, with the shorter duration breaking an exact tie. Pilot weights are then discarded. All $1 9 \times 3 = 5 7$ final models are retrained on every target-pair training row for exactly 200 steps. No target test sample enters duration selection. The target test set is accessed only after the protocol is frozen.

Metrics and Implementation. For relation e and $k \_ \in$ {1, 5, 10}, bidirectional Recall@k averages the two retrieval directions over the complete gallery. Recall@10 is the primary metric. Recovery@10 is defined in Eq. (71). Blue parentheses in the main tables show point gains over frozen cosine, while the final columns report ratios of backbone means to Full A–B. Following the source-to-anchor construction, the first listed modality on each evaluation edge is A, and the second is B. ReAlign uses a $1 0 ^ { - 1 0 }$ stabilizer and fixes the $A  B$ orientation without performance-based selection. ASIF uses $k = \mathrm { m i n } ( 8 0 0 , n _ { \mathrm { t r a i n } } )$ nonzero anchors and a similarity exponent of 8. Its anchor selection and sparse retrieval follow a fixed CPU path that determines the order of equalsimilarity anchors. Paired-OP row-normalizes inputs, applies no centering, and solves both retrieval directions independently by singular-value decomposition. Neither ASIF nor Paired-OP selects hyperparameters on the test set. HSA uses $\lambda =$ $0 . 5 , \ \kappa = 2 , \ k _ { \mathrm { m a x } } = 2 5 6$ , and rank-permutation seed 142. Score calibration uses mismatch seeds 42–46. The reliability gate uses 256 simultaneous row-and-column permutations of the regularized covariance matrix, the 0.95 quantile, and seed 42. Trainable baselines use a Linear(d, 1024)–GELU– Linear(1024, 256) head. They use AdamW [91] with learning rate $1 0 ^ { - 3 }$ and weight decay $1 0 ^ { - 4 }$ . All settings remain fixed across relations.

## H.2 Efficiency Measurement and Outcome Checks

Table X measures all fourteen methods over 19 nondegenerate relations and 494 independent runs. Analytic fits use one warm-up and five timed repeats, while trainable methods use seeds 40–42. Each timed score call reproduces the registered retrieval hits, yielding 2,964 passing checks. All GPU paths use the same NVIDIA RTX 4070 SUPER. ASIF retains its fixed CPU path, and each method retains its original software environment. Fit and training times quantify the realized workload in those environments. Steps is the median finalfit gradient-step count over relation–seed runs. The archived records retain the complete selection-search and total-gradientstep distributions.

Table XI repeats the measurement over the eleven classification relations. Analytic fits run on the CPU, while trainable methods replay model selection and final fitting for seeds 40– 42 on the same GPU. Relation values are seed medians. All post-fit readouts use a 12-thread BLAS and 6-thread PyTorch

RECALL@10 VARIABILITY ACROSS THREE TRAINING SEEDS. ENTRIES ARE SAMPLE STANDARD DEVIATIONS IN PERCENTAGE POINTS. “MEAN RELATION SD” AVERAGES THE NINETEEN RELATION-WISE STANDARD DEVIATIONS; “MAX RELATION SD” GIVES THEIR MAXIMUM.  
TABLE XV
<table><tr><td>Method</td><td>ImageBind Mean SD</td><td>LanguageBind Mean SD</td><td> $_ { \mathrm { A l l - } 1 9 }$  Mean SD</td><td>Mean Relation SD</td><td>Max Relation SD</td></tr><tr><td>ERM</td><td>0.25</td><td>0.17</td><td>0.12</td><td>0.58</td><td>1.33</td></tr><tr><td>IRM</td><td>0.03</td><td>0.18</td><td>0.09</td><td>0.52</td><td>1.30</td></tr><tr><td>VREx</td><td>0.28</td><td>0.17</td><td>0.22</td><td>0.71</td><td>1.55</td></tr><tr><td>DANN</td><td>0.31</td><td>0.37</td><td>0.25</td><td>0.71</td><td>1.69</td></tr><tr><td>CORAL</td><td>0.10</td><td>0.23</td><td>0.08</td><td>0.62</td><td>1.41</td></tr><tr><td>Full A-B</td><td>0.36</td><td>0.50</td><td>0.42</td><td>0.64</td><td>1.49</td></tr></table>

CPU configuration after one warm-up and five timed repeats. The timed boundary includes method-specific transforms, the complete query–prototype score matrix, and Top-1 selection; shared prototype construction is excluded. We report latency per 1,000 queries and score throughput because class counts vary. The 286 method-level runs reproduce every registered macro Top-1 value. All 198 trainable runs match the prescribed architectures, seeds, workloads, inputs, and final steps.

Classification fitting times are relation medians; readout latency and throughput pool time, queries, and scores across all eleven relations. On the shared CPU configuration, HSA processes 1,000 queries in 25.614 ms at 9.172 million query– prototype scores per second. Its latency is approximately 2.4 times that of the source-trained neural baselines (Table XI), measured after fitting on the same pooled workload.

## H.3 Seed Variability of Trainable Baselines

Table XV reports sample standard deviations across seeds 40–42. The first three columns give the standard deviation of each seed-level aggregate. The final two summarize relationwise standard deviations. All values were recomputed from the archived seed-level results using denominator n − 1.

## H.4 Excluded Degenerate Unit

The archived ImageBind text–depth evaluation on NYUv2 contains one positive group. Every evaluated readout obtains 100% Recall@10 because all candidates belong to the same positive set. This unit remains in the protocol record but is excluded from the nineteen-relation aggregate, method ranking, and recovery analysis.

## APPENDIX I

## ADDITIONAL RETRIEVAL RESULTS

This section reports direction-wise consistency, recovery relative to target-paired training, and robustness under alternative aggregate definitions. Unless stated otherwise, all aggregates use the same nineteen nondegenerate relations and the bidirectional full-gallery retrieval protocol.

## I.1 Direction-Wise Retrieval Consistency

Table XVI reports the direction-wise aggregates summarized in Q1. HSA improves both directions on each backbone, with all-relation gains of 13.33 and 12.43 points.

TABLE XVI  
DIRECTION-WISE RETRIEVAL CONSISTENCY. MEAN RECALL@10 (%) IN BOTH RETRIEVAL DIRECTIONS; PARENTHESES SHOW HSA GAINS OVER FROZEN COSINE.
<table><tr><td></td><td colspan="2">ImageBind</td><td colspan="2">LanguageBind</td><td colspan="2">Overall</td></tr><tr><td>Method</td><td> $A  B$ </td><td> $B  A$ </td><td> $A  B$ </td><td> $B  A$ </td><td> $A  B$ </td><td> $B  A$ </td></tr><tr><td>Frozen Cosine</td><td>13.74</td><td>7.86</td><td>24.77</td><td>25.24</td><td>19.54</td><td>17.01</td></tr><tr><td>HSA (Ours)</td><td>29.16 (+15.43)</td><td>20.63 (+12.77)</td><td>36.22 (+11.45)</td><td>37.35 (+12.11)</td><td>32.88 (+13.33)</td><td>29.43 (+12.43)</td></tr></table>

## I.2 Recovery Relative to Target-Paired Training

For each relation $e ,$ the reported ratio is

$$
\mathrm { R e c o v e r y @ 1 0 } ( e ) = 1 0 0 \times \frac { \mathrm { R @ 1 0 _ { H S A } } ( e ) } { \mathrm { R @ 1 0 _ { F u l l ~ \boldsymbol { A } \cdot \boldsymbol { B } } } ( e ) } .\tag{71}
$$

Table XVII gives the relation-wise values underlying the recovery columns in Tables I and III. The aggregate rows report ratios of backbone-mean Recall@10 values rather than averages of the relation-wise percentages.

HSA reaches 31.15% mean Recall@10 without target-pair fitting and exceeds Full A–B on 10 of 19 relation-level point estimates. With its specified projection head and objective, Full A–B uses a globally selected 200-step duration on the same data splits. It reaches $2 2 . 9 5 \pm 0 . 3 6 \%$ on ImageBind and $3 9 . 1 8 \pm 0 . 5 0 \%$ on LanguageBind, with an overall mean of $3 1 . 4 9 \pm 0 . 4 2 \%$ . Here ± denotes the sample standard deviation across seeds 40–42. Recovery quantifies HSA’s performance relative to this fitted target-paired reference; values above 100% indicate higher Recall@10 on the corresponding relation.

## I.3 Robustness to Aggregation and Chance Level

The main paper reports an unweighted mean over the nineteen configured relations. Table XVIII evaluates the HSA gain over frozen cosine under alternative summaries. The datasetbalanced value first averages relations within each dataset and then weights the ten datasets equally. The leave-one-datasetout range recomputes the relation mean after excluding each dataset in turn. For relation e, the chance-normalized gain is

$$
G _ { e } ^ { \mathrm { c h a n c e } } = 1 0 0 \times \frac { R _ { \mathrm { H S A } , e } - R _ { C 0 , e } } { 1 0 0 - C _ { e } } ,\tag{72}
$$

where $C _ { e }$ is the Recall@10 chance level implied by the complete gallery and the registered positive definition.

Relation-specific chance levels range from 0.12% to 19.96%. HSA exceeds chance on all nineteen relations, whereas frozen cosine does so on fourteen. Every alternative summary remains positive, showing that the aggregate gain persists across the observed mixture of datasets and relation difficulties. The 9.72-point dataset-balanced macro and the 8.02-point leave-one-dataset-out lower bound further demonstrate stable gains under equal dataset weighting and every single-dataset exclusion.

TABLE XVII  
RELATION-WISE RECALL@10 RELATIVE TO FULL A-B. PAIR ABBREVIATIONS FOLLOW THE MAIN TABLES. HSA AND FULL A-B ARE BIDIRECTIONAL PERCENTAGES. RECOVERY USES THE UNROUNDED VALUES IN EQ. (71); DISPLAYED RECALLS ARE INDEPENDENTLY ROUNDED TO TWO DECIMALS.
<table><tr><td>Backbone</td><td>Held-Out Pair</td><td>Dataset</td><td>HSA</td><td>Full A-B</td><td>Recovery (%)</td></tr><tr><td rowspan="10">ImageBind</td><td>Tx-Au</td><td>VGGSound</td><td>84.45</td><td>84.57</td><td>99.9</td></tr><tr><td>Tx-Th</td><td>TartanRGBT</td><td>74.87</td><td>74.71</td><td>100.2</td></tr><tr><td>Tx-IMU</td><td>Ego4D</td><td>8.97</td><td>4.01</td><td>223.6</td></tr><tr><td>Au-D</td><td>BatVision</td><td>6.93</td><td>7.13</td><td>97.2</td></tr><tr><td>Au-Th</td><td>MAVD</td><td>4.63</td><td>3.09</td><td>150.0</td></tr><tr><td>Au-IMU</td><td>Ego4D</td><td>8.93</td><td>7.15</td><td>124.9</td></tr><tr><td>D-Th</td><td>TartanRGBT</td><td>9.66</td><td>15.76</td><td>61.3</td></tr><tr><td>D-IMU</td><td>UTD-MHAD</td><td>9.30</td><td>6.01</td><td>154.8</td></tr><tr><td>Th-IMU</td><td>Caltech Aerial RGBT</td><td>16.33</td><td>4.15</td><td>393.4</td></tr><tr><td colspan="2">Ratio of Backbone Means</td><td>24.90</td><td>22.95</td><td>108.5</td></tr><tr><td rowspan="10">LanguageBind</td><td>Im-Vi Im-Th</td><td>UCF101</td><td>99.72</td><td>99.76</td><td>100.0</td></tr><tr><td>Im-D</td><td>TartanRGBT</td><td>60.89</td><td>60.36</td><td>100.9</td></tr><tr><td>Im-Au</td><td>NYUv2</td><td>13.53</td><td>44.19</td><td>30.6</td></tr><tr><td>Vi-Th</td><td>VGGSound</td><td>67.15</td><td>41.02</td><td>163.7</td></tr><tr><td>Vi-D</td><td>TartanRGBT</td><td>53.33</td><td>54.43</td><td>98.0</td></tr><tr><td></td><td>TartanRGBT</td><td>18.00</td><td>25.84</td><td>69.7</td></tr><tr><td>Vi-Au Th-D</td><td>MSR-VTT</td><td>26.64</td><td>26.34</td><td>101.1</td></tr><tr><td>Th-Au</td><td>TartanRGBT MAVD</td><td>19.27</td><td>31.31</td><td>61.5</td></tr><tr><td>D-Au</td><td>BatVision</td><td>3.93</td><td>2.73</td><td>143.7</td></tr><tr><td></td><td></td><td>5.39</td><td>5.82</td><td>92.6</td></tr><tr><td colspan="3">Ratio of Backbone Means</td><td>36.79</td><td>39.18</td><td>93.9</td></tr></table>

TABLE XVIII

ROBUSTNESS OF THE HSA–FROZEN-COSINE RECALL@10 GAIN. POINT GAINS ARE IN PERCENTAGE POINTS; THE FINAL ROW IS THE PERCENTAGE OF CHANCE-ADJUSTED HEADROOM RECOVERED.
<table><tr><td>Summary</td><td>Weighting Rule</td><td>Gain</td></tr><tr><td>Relation Mean</td><td>Equal weight over 19 relations</td><td>12.88</td></tr><tr><td>Relation Median</td><td>Median over 19 relations</td><td>9.25</td></tr><tr><td>Dataset-Balanced Macro</td><td>Equal weight over 10 datasets</td><td>9.72</td></tr><tr><td>Leave One Dataset Out</td><td>Relation Mean after each exclu- sion</td><td>8.02-14.11</td></tr><tr><td>Chance Normalized</td><td>Mean of Eq. (72)</td><td>13.98%</td></tr></table>

## APPENDIX J

## MECHANISM AND DIAGNOSTIC ANALYSES

This section collects the complete carrier interventions, component ablations, disjoint-source control, and correspondence-resolution diagnostics supporting the mainpaper mechanism analysis.

## J.1 Complete Aggregate Intervention Statistics

Source permutations evaluate complete HSA: each condition refits directions, ranks, the gate, and scales while preserving the algorithm, hyperparameters, and input marginals. Carrier localization holds the intact candidate-resolution branch, gate, and resolution scale fixed, assigns the same reliability spectrum to each direction set, and recalibrates its carrier scale by the same source-only rule. The reliability-dose endpoint comparison changes only α between 0 and 1 with all fitted quantities and scales fixed. Table XIX summarizes these three interventions; Appendix K lists the complete test families.

All relations from one dataset receive the same sign flip, so the smallest one-sided probability is determined by ten groups: $2 ^ { - 1 0 } = 1 / 1 0 2 4$ . Seeds are averaged within relations and do not add independent observations.

## J.2 Complete Component-Ablation Visualization

Fig. 9 visualizes the complete Recall@10 ablations, with dataset-cluster comparisons in Table XXII. Table IX in the main paper reports backbone-specific Recall@10 results together with the corresponding function-metric changes defined in Appendix K.2. Each comparison changes one scoring component while retaining the fitted directions, evaluation relations, and remaining scoring pipeline. Removing the carrier score produces the largest all-relation reduction, at 12.00 points. Removing the source-derived gate and using uniform channel reliabilities reduce the mean by 3.86 and 2.48 points. Candidate resolution contributes 3.43 points overall, concentrated on LanguageBind. Residual orthogonalization has the smallest overall effect and shows the same backbone dependence.

## J.3 Matched Shared-Row and Disjoint-Source Control

Theorem 1 expresses the population relation as a function of two hub-edge moment systems. We therefore test whether HSA retains its retrieval gain when the hub edges use mutually disjoint source instances. For each split seed in $\{ 4 2 , \ldots , 5 1 \}$ the source rows are randomly partitioned into equal halves $I _ { 1 }$ and $I _ { 2 } .$ . When the row count is odd, one predetermined row is omitted. Shared-row conditions estimate both edges from the same half. Disjoint conditions estimate A–H and

TABLE XIX  
AGGREGATE COMPLETE-SCORE INTERVENTIONS. DIFFERENCES ARE INTACT-MINUS-PERMUTED, LEADING-MINUS-CONTROL, OR α = 1 MINUS α = 0 BIDIRECTIONAL RECALL@10 IN PERCENTAGE POINTS. INTERVALS AND SIGN FLIPS USE DATASET GROUPS; HOLM CORRECTION USES EACH COMPLETE COMPARISON FAMILY.
<table><tr><td>Comparison</td><td>Positive Relations</td><td>Mean Difference</td><td>95% Cluster Interval</td><td>Holm p</td></tr><tr><td>Permute A-H</td><td>19/19</td><td>26.52</td><td>[9.83, 40.64]</td><td> $2 . 9 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Permute H-B</td><td>19/19</td><td>26.58</td><td>[10.03, 41.01]</td><td> $2 . 9 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Permute Both Edges</td><td>19/19</td><td>24.59</td><td>[9.22, 36.53]</td><td> $2 . 9 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Subsequent Carriers</td><td>19/19</td><td>16.44</td><td>[6.31, 25.41]</td><td> $1 . 9 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Random Carriers</td><td>19/19</td><td>19.38</td><td>[7.06, 29.21]</td><td> $1 . 9 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Reliability α = 0</td><td>18/19</td><td>12.00</td><td>[4.75, 16.96]</td><td>0.01172</td></tr></table>

![](images/8b85825e4e51c245d29eb184b15a87bed62d7f09e0aa411264c60de3db55a32e.jpg)  
Fig. 9. Complete Component Ablation of the HSA Readout. Mean bidirectional Recall@10 for full HSA and five matched single-component variants on ImageBind, LanguageBind, and all nineteen relations.

H–B from opposite halves. All conditions use equal peredge sample counts and matched cross-half calibration rows. The gate construction, scoring rule, and evaluation set remain unchanged.

For every metric, the two shared orientations and two disjoint orientations are first averaged within each split seed. Within-relation confidence intervals use 10,000 bootstrap resamples with seed 42. The cross-relation mean row instead uses the 100,000-draw dataset-cluster analysis in Appendix K. The individual relation intervals resample each relation’s natural independent unit: retrieval group, paired instance, scene, trajectory, subject, drive, or video. Table XX reports the independent-unit count and the primary Recall@10 comparison. Here C0 denotes frozen cosine, S the sample-sizematched shared-row condition, and D the strictly disjoint condition.

The clustered intervals reflect variation across the dataset’s sampling units: two drives for MAVD, four subjects for UTD-MHAD, and seven scenes for BatVision. For ImageBind audio–thermal, the drive-specific D − C0 gains are 25.79 points for the 19-row drive and 0.44 points for the 413-row drive. These two support points determine the [0.44, 25.79] interval and describe the observed between-drive heterogeneity.

Disjoint estimation retains a 10.77-point mean gain over frozen cosine, with dataset-cluster interval [3.83, 16.57], supporting relation recovery from independently sampled hub edges. Recall@10 gains are positive on all nineteen relations, with eighteen relation-wise intervals entirely above zero. LanguageBind image–depth gains 1.68 points, with interval [−0.80, 4.14]. Shared-row estimation adds 1.18 points on average, with interval [0.53, 2.30], extending beyond the prespecified ±1-point equivalence range. At Recall@1 and Recall@5, disjoint estimation improves eighteen relations; LanguageBind image–depth changes by −0.74 and −1.15 points, respectively.

All $1 9 \times 1 0 \times 4 = 7 6 0$ relation–seed–condition fits satisfy the prescribed hub-edge and calibration-row separation in the disjoint conditions. Numerical decomposition uses NumPy for 759 fits. For LanguageBind image–video, seed 50 and orientation D21, double-precision PyTorch CPU SVD supplies a converged solution after NumPy non-convergence. Its relative reconstruction error is $2 . 9 \times 1 0 ^ { - 1 5 }$

J.4 Correspondence Resolution under Group-Preserving Permutations

This section measures correspondence resolution with group-preserving permutations. The experiments use disjoint A–H and H–B training rows. Target-side permutations preserve the relevant frame, trajectory, or semantic-group structure. Table XXI records the finest resolution supported by the dataset and hub observations in each configuration.

## APPENDIX K

## DATASET-CLUSTER STATISTICS AND ROBUSTNESS

## K.1 Estimands, Resampling, and Complete Families

This post hoc statistical supplement averages seeds within each relation, then weights the nineteen relations equally. Its ten dataset groups are BatVision (2), Caltech (1), Ego4D (2), MAVD (2), MSR-VTT (1), NYUv2 (1), Tartan (6), UCF101 (1), UTD-MHAD (1), and VGGSound (2), with relation counts in parentheses. Relations sharing a dataset use one group identifier across backbones and modalities.

For relation differences $d _ { e } ,$ each cluster-bootstrap draw samples G datasets with replacement and carries every relation in each sampled group. The replicate estimate divides the sum of sampled differences by the actual number of sampled relations. We use seed 42, 100,000 draws, and the 2.5th and 97.5th percentiles. This preserves the relation-equal estimand.

TABLE XX  
RECALL@10 UNDER MATCHED SHARED-ROW AND DISJOINT-SOURCE ESTIMATION. VALUES ARE BIDIRECTIONAL PERCENTAGES. $n _ { \mathrm { u n i t } }$ GIVES THE NUMBER OF CLUSTERED BOOTSTRAP UNITS; THE FINAL COLUMNS REPORT D − C0 AND $S - D$ WITH NATURAL-UNIT 95% INTERVALS FOR EACH RELATION; THE FINAL AGGREGATE ROW USES A DATASET-CLUSTER INTERVAL.
<table><tr><td>Backbone</td><td colspan="2">Held-Out Pair Dataset</td><td> $n _ { \mathrm { u n i t } }$ </td><td>C0</td><td>S</td><td>D</td><td>D - C0 [95% CI]</td><td> $S - D$  [95% CI]</td></tr><tr><td rowspan="7">ImageBind</td><td>Tx-Au</td><td>VGGSound</td><td>100</td><td>75.20</td><td>83.80</td><td>81.06</td><td>5.86 [2.75,9.37]</td><td>2.74 [1.94,3.59]</td></tr><tr><td>Tx-Th</td><td>TartanRGBT</td><td>46</td><td>7.68</td><td>73.40</td><td>72.42</td><td>64.74 [58.02,71.09]</td><td>0.99 [0.48,1.52]</td></tr><tr><td>Tx-IMU</td><td>Ego4D</td><td>686</td><td>1.53</td><td>7.03</td><td>6.01</td><td>4.48 [3.35,5.67]</td><td>1.03 [0.44,1.63]</td></tr><tr><td>Au-D</td><td>BatVision</td><td>7</td><td>1.97</td><td>7.71</td><td>7.82</td><td>5.85 [4.17,7.42]</td><td>-0.11 [-0.54,0.42]</td></tr><tr><td>Au-Th</td><td>MAVD</td><td>2</td><td>1.62</td><td>4.11</td><td>3.18</td><td>1.56 [0.44,25.79]</td><td>0.93 [0.63,7.50]</td></tr><tr><td>Au-IMU</td><td>Ego4D</td><td>487</td><td>1.75</td><td>7.84</td><td>6.42</td><td>4.67 [3.27,6.08]</td><td>1.42 [0.60,2.24]</td></tr><tr><td>D-Th</td><td>TartanRGBT</td><td>46</td><td>0.34</td><td>9.71</td><td>9.50</td><td>9.16 [7.98,10.40]</td><td>0.21 [0.01,0.43]</td></tr><tr><td></td><td>D-IMU</td><td>UTD-MHAD</td><td>4</td><td>3.02</td><td>6.87</td><td>6.64</td><td>3.62 [1.23,6.00]</td><td>0.23 [-0.19,0.69]</td></tr><tr><td rowspan="7">LanguageBind</td><td>Th-IMU</td><td>Caltech Aerial RGBT</td><td>245</td><td>4.08</td><td>17.53</td><td>10.64</td><td>6.56 [4.63,8.43]</td><td>6.89 [4.52,9.34]</td></tr><tr><td>Im-Vi</td><td>UCF101</td><td>3030</td><td>96.49</td><td>99.61</td><td>99.60</td><td>3.11 [2.65,3.61]</td><td>0.02 [-0.00,0.04]</td></tr><tr><td>Im-Th</td><td>TartanRGBT</td><td>9</td><td>36.93</td><td>58.54</td><td>58.15</td><td>21.22 [16.77,25.79]</td><td>0.40 [-0.14,0.81]</td></tr><tr><td>Im-D</td><td>NYUv2</td><td>654</td><td>11.62</td><td>13.47</td><td>13.30</td><td>1.68 [-0.80,4.14]</td><td>0.17 [0.01,0.34]</td></tr><tr><td>Im-Au</td><td>VGGSound</td><td>1000</td><td>43.40</td><td>62.92</td><td>61.43</td><td>18.04 [15.87,20.19]</td><td>1.48 [1.12,1.85]</td></tr><tr><td>Vi-Th</td><td>TartanRGBT</td><td>9</td><td>36.01</td><td>52.99</td><td>52.57</td><td>16.56 [12.86,22.58]</td><td>0.42 [0.10,0.78]</td></tr><tr><td>Vi-D</td><td>TartanRGBT</td><td>9</td><td>7.11</td><td>16.99</td><td>16.25</td><td>9.14 [5.09,18.06]</td><td>0.74 [0.07,1.80]</td></tr><tr><td></td><td>Vi-Au</td><td>MSR-VTT</td><td>884</td><td>6.96</td><td>24.13</td><td>20.91</td><td>13.95 [11.83,16.06]</td><td>3.22 [2.64,3.81]</td></tr><tr><td></td><td>Th-D</td><td>TartanRGBT</td><td>9</td><td>7.57</td><td>18.57</td><td>16.97</td><td>9.40 [3.50,17.44]</td><td>1.59 [0.93,2.11]</td></tr><tr><td></td><td>Th-Au MAVD</td><td></td><td>2</td><td>2.08</td><td>3.45</td><td>3.53</td><td>1.45 [1.35,3.68]</td><td>-0.08 [-1.71,-0.00]</td></tr><tr><td></td><td>D-Au BatVision</td><td></td><td>7</td><td>1.88</td><td>5.48</td><td>5.44</td><td>3.56 [1.07,6.60]</td><td>0.04 [-0.25,0.38]</td></tr><tr><td colspan="2">Mean over 19 Relations</td><td></td><td></td><td>18.27</td><td>30.22</td><td>29.04</td><td>10.77 [3.83,16.57]</td><td>1.18 [0.53,2.30]</td></tr></table>

TABLE XXI

CORRESPONDENCE-RESOLUTION DIAGNOSTICS. EFFECTS ARE BIDIRECTIONAL RECALL@10 DIFFERENCES IN PERCENTAGE POINTS WITH 95% INTERVALS WHEN AVAILABLE. AN INTERVAL CONTAINING ZERO LEAVES THE CORRESPONDING RESOLUTION UNRESOLVED.
<table><tr><td>Dataset, Pair, and Backbone</td><td>Target-Level Evidence</td><td>Group-Level Evidence</td><td>Supported Resolution</td></tr><tr><td>TartanRGBT D-Th (ImageBind)</td><td>3.13 [2.26,4.34]</td><td></td><td>Frame</td></tr><tr><td>MSR-VTT Vi–Au (LanguageBind)</td><td>interval contains zero</td><td>28.54 [26.47,30.58]</td><td>Semantic Group</td></tr><tr><td>BatVision Au-D (ImageBind)</td><td>0.24 [-0.16,0.79]</td><td></td><td>Unresolved</td></tr></table>

We additionally report dataset-equal means and leave-onedataset-out ranges. One-sided exact sign flips enumerate $2 ^ { G }$ whole-group signs, using the relation mean as the statistic and including ties in the tail. The assumptions are between-group independence and joint sign symmetry of group differences.

The seven families are F1, HSA versus frozen cosine; F2, the other nine source-only comparators; F3, the three source permutations; F4, the two carrier-position controls; F5, all five original Recall@10 component ablations; F6, the two relation-construction controls; and F7, disjoint-minus-cosine and shared-minus-disjoint estimation. Holm correction is applied within each complete family. ASIF, Paired-OP, and Full A–B use target-pair information and are reported separately as descriptive references. The responsibility metrics, fixedstate classification, and NYUv2 exclusion analysis receive descriptive intervals only.

Candidate resolution, source gating, and residual orthogonalization contribute positive mean Recall@10 gains, each with $\begin{array} { r c l } { p _ { \mathrm { H o l m } } } & { = } & { 0 . 0 8 7 8 9 } \end{array}$ . Function-specific metrics assess discrimination and redundancy exploratorily. Full A–B is a descriptive reference: its difference interval spans zero, with no prespecified equivalence or non-inferiority margin. Datasetcluster intervals describe the evaluated benchmark collection across two shared backbones.

## K.2 Component Responsibilities: Definitions and Complete Results

The responsibility metrics were specified before their additional measurement and use the original complete score and registered test inputs. Fitted quantities remain fixed except for the removed component and its prescribed sourcescale calibration. Carrier evidence is evaluated by the original bidirectional full-gallery Recall@10. All changes are ablation minus full HSA; lower Red. and higher values of the other metrics are preferred.

TABLE XXII  
COMPLETE DATASET-CLUSTER COMPARISONS. ALL DIFFERENCES AND INTERVALS ARE IN PERCENTAGE POINTS. RELATION-EQUAL MEANS, DATASET-EQUAL MEANS, AND LEAVE-ONE-DATASET-OUT RANGES HAVE DISTINCT WEIGHTING INTERPRETATIONS. EVERY COMPARISON COVERS NINETEEN RELATIONS AND TEN DATASETS.
<table><tr><td>Family</td><td>Control/Comparison</td><td>Relation Mean</td><td>Dataset Mean</td><td>95% Interval</td><td>Leave-One-Out Range</td><td>Raw p</td><td colspan="2">Holm p</td></tr><tr><td>F1</td><td>Frozen Cosine</td><td>12.88</td><td>9.72</td><td>[5.29, 18.68]</td><td>[8.02, 14.11]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>F2</td><td>Hub-Relative</td><td>22.01</td><td>20.36</td><td>[8.74, 32.28]</td><td>[18.06, 24.37]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $8 . 7 9 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F2</td><td>Bi. Ridge</td><td>7.17</td><td>6.11</td><td>[2.14, 11.07]</td><td>[4.44, 7.99]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $8 . 7 9 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F2</td><td>Bi. Procrustes</td><td>7.56</td><td>6.44</td><td>[3.49, 10.95]</td><td>[4.82, 8.28]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $8 . 7 9 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F2</td><td>ReAlign</td><td>12.67</td><td>9.99</td><td>[5.41, 18.16]</td><td>[8.20, 13.96]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $8 . 7 9 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F2</td><td>ERM</td><td>9.02</td><td>8.20</td><td>[3.92, 12.77]</td><td>[6.33, 10.04]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $8 . 7 9 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F2</td><td>IRM</td><td>9.32</td><td>8.48</td><td>[4.60, 12.76]</td><td>[6.81, 10.34]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $8 . 7 9 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F2</td><td>VREx</td><td>8.90</td><td>7.86</td><td>[3.79, 12.73]</td><td>[6.04, 9.88]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $8 . 7 9 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F2</td><td>DANN</td><td>9.51</td><td>8.61</td><td>[4.40, 13.27]</td><td>[6.81, 10.50]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $8 . 7 9 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F2</td><td>CORAL</td><td>9.03</td><td>8.21</td><td>[4.03, 12.70]</td><td>[6.40, 10.05]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $8 . 7 9 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F3</td><td>Permuted A-H</td><td>26.52</td><td>24.60</td><td>[9.83, 40.64]</td><td>[21.44, 29.35]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $2 . 9 3 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F3</td><td>Permuted H-B</td><td>26.58</td><td>25.25</td><td>[10.03, 41.01]</td><td>[21.67, 29.44]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $2 . 9 3 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F3</td><td>Both Edges Permuted</td><td>24.59</td><td>21.98</td><td>[9.22, 36.53]</td><td>[20.04, 27.26]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $2 . 9 3 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F4</td><td>Subsequent Carriers</td><td>16.44</td><td>14.15</td><td>[6.31, 25.41]</td><td>[12.63, 18.15]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $1 . 9 5 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F4</td><td>Random Carriers</td><td>19.38</td><td>16.13</td><td>[7.06, 29.21]</td><td>[15.33, 21.41]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $1 . 9 5 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F5</td><td>w/o Carrier Score</td><td>12.00</td><td>9.72</td><td>[4.75, 16.96]</td><td>[9.00, 13.20]</td><td> $2 . 9 3 \times 1 0 ^ { - 3 }$ </td><td>0.01172</td><td></td></tr><tr><td>F5</td><td>w/o Channel Reliability</td><td>2.48</td><td>2.69</td><td>[1.49, 3.55]</td><td>[2.13, 2.69]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $4 . 8 8 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F5</td><td>w/o Candidate Resolution</td><td>3.43</td><td>2.53</td><td>[0.26, 5.97]</td><td>[1.52, 3.93]</td><td>0.03906</td><td>0.08789</td><td></td></tr><tr><td>F5</td><td>w/o Source Gate</td><td>3.86</td><td>2.61</td><td>[0.55, 5.89]</td><td>[2.29, 4.58]</td><td>0.0293</td><td>0.08789</td><td></td></tr><tr><td>F5</td><td>w/o Orthogonalization</td><td>0.99</td><td>0.59</td><td>[-0.02, 1.84]</td><td>[0.28, 1.18]</td><td>0.07422</td><td>0.08789</td><td></td></tr><tr><td>F6</td><td>Diagonal Hub Covariance</td><td>4.13</td><td>3.66</td><td>[1.92, 5.76]</td><td>[3.03, 4.57]</td><td> $1 . 9 5 \times { { 1 0 } ^ { - 3 } }$ </td><td> $3 . 9 1 \times { 1 0 } ^ { - 3 }$ </td><td></td></tr><tr><td>F6</td><td>Trace-Matched Identity</td><td>4.21</td><td>4.01</td><td>[2.50, 5.42]</td><td>[3.55, 4.57]</td><td> $1 . 9 5 \times 1 0 ^ { - 3 }$ </td><td> $3 . 9 1 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F7</td><td>Disjoint Minus Cosine</td><td>10.77</td><td>7.34</td><td>[3.83, 16.57]</td><td>[5.72, 11.86]</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $1 . 9 5 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>F7</td><td>Shared Minus Disjoint</td><td>1.18</td><td>1.50</td><td>[0.53, 2.30]</td><td>[0.86, 1.38]</td><td> $2 . 9 3 \times 1 0 ^ { - 3 }$ </td><td> $2 . 9 3 \times 1 0 ^ { - 3 }$ </td><td></td></tr></table>

TABLE XXIII

DESCRIPTIVE TARGET-PAIRED REFERENCES. DIFFERENCES ARE HSA MINUS REFERENCE IN PERCENTAGE POINTS, ACROSS NINETEEN RELATIONS AND TEN DATASETS.
<table><tr><td>Reference</td><td>Relation Mean Difference</td><td>Dataset Mean Difference</td><td>95% Interval</td><td>Leave-One-Out Range</td></tr><tr><td>ASIF</td><td>4.37</td><td>4.31</td><td>[1.84, 6.30]</td><td>[3.42, 4.88]</td></tr><tr><td>Paired-OP</td><td>4.91</td><td>4.37</td><td>[2.64, 6.41]</td><td>[4.00, 5.30]</td></tr><tr><td>Full A-B</td><td>-0.34</td><td>-0.19</td><td>[-5.43, 5.57]</td><td>[-1.91, 1.53]</td></tr></table>

Reliability Discrimination, P-AUC. Positives are all registered $( A _ { i } , B _ { i } )$ pairs, once per row. Negatives use five fixed seeds, 47–51, excluding same-class candidates under classpositive protocols. Calibrated carrier scores distinguish these positives from mismatched pairs. We compare the original channel reliabilities with a constant spectrum having the same mean; the uniform condition recalibrates its carrier scale by the same source-only rule. P-AUC measures discrimination for these registered pairs, while full-gallery retrieval preserves the original multiple-positive definitions.

Candidate Resolution, C@1. In each direction, carrier scores define a fixed top-ten candidate pool. Full HSA and carrieronly scores rank the same pool. C@1 is Top-1 accuracy conditional on that pool containing at least one valid positive. Pool formation and Top-1 selection share the same candidate order and tie handling, with no label-based tie breaking. Coverage is the proportion of queries meeting the condition.

Counts are combined over both directions within each relation before relation-equal aggregation.

Gate Discrimination, G-AUC. The source gate g<sub>R</sub> is the predictor for each relation. The evaluation label records whether adding the ungated candidate-resolution term improves Recall@10 over carrier-only scoring. This label evaluates gate ordering after scoring; gate estimation continues to use source statistics alone. A constant-one gate has AUC 0.5. The pooled comparison uses the complete nineteen-relation set. The descriptive backbone breakdown in Table XXIV recomputes G-AUC from the same predictions and labels within each backbone, giving 0.125 for ImageBind and 0.917 for LanguageBind. These within-backbone values measure a different ordering from the pooled G-AUC of 0.714.

Score Redundancy, Red. On the same registered positive and negative pairs used for reliability, Red. is the absolute Pearson correlation between carrier and candidate-resolution scores. Orthogonal residual coordinates and raw centered, rownormalized residual coordinates use their respective resolution scales obtained by the same source-only rule. This metric measures linear score redundancy between the two branches. Candidate coverage is defined for all nineteen relations. Its relation-equal mean is 27.72%, with 95% dataset-cluster interval [12.04, 43.41]%. The conditional set contains 15,282 of 45,170 bidirectional queries, giving pooled coverage of 33.83%. These summaries weight relations and queries differently; the main analysis uses relation-equal aggregation. C@1 improvement is conditional on the fixed carrier pool containing a positive. Candidate-resolution benefit varies across relations: ungated resolution benefits 5 of nineteen, with beneficial/nonbeneficial counts of 1/8 for ImageBind and 4/6 for LanguageBind. Pooled G-AUC is 0.714, with 95% cluster interval [0.178, 1.000]. The interval uses 99,438 bootstrap draws containing both labels; 562 single-label draws are undefined. P-AUC and redundancy intervals are narrower; the candidateresolution lower bound approaches zero. These exploratory metrics characterize discrimination, candidate ranking, and score redundancy.

TABLE XXIV  
COMPLETE RESPONSIBILITY RESULTS AND EXPLORATORY INTERVALS. THE FIRST FOUR METRICS ARE DISPLAYED AS PERCENTAGES, WITH CHANGES AND INTERVALS IN PERCENTAGE POINTS; RED. USES ABSOLUTE CORRELATION UNITS. BACKBONE COLUMNS GIVE WITHIN-BACKBONE CHANGES; INTERVALS APPLY TO THE OVERALL CHANGE. THE GATE CHANGE IS CONSTANT-ONE MINUS SOURCE GATE, WITH ITS INTERVAL OBTAINED BY REVERSING THE G-AUC INTERVAL.
<table><tr><td>Metric</td><td>HSA (Ours)</td><td>Ablation</td><td>Overall ∆</td><td>ImageBind Δ</td><td>LanguageBind Δ</td><td>95% Cluster Interval</td></tr><tr><td>R@10 ↑</td><td>31.15</td><td>19.15</td><td>-12.00</td><td>-14.43</td><td>-9.82</td><td>[-16.96, -4.75]</td></tr><tr><td>P-AUC ↑</td><td>81.17</td><td>75.97</td><td>-5.20</td><td>-6.79</td><td>-3.76</td><td>[-7.90, -3.37]</td></tr><tr><td>C@1↑</td><td>28.80</td><td>24.37</td><td>-4.43</td><td>+1.22</td><td>-9.51</td><td>[-8.83, -0.00]</td></tr><tr><td>G-AUC ↑</td><td>71.43</td><td>50.00</td><td>-21.43</td><td>+37.50</td><td>-41.67</td><td>[-50.00, 32.22]</td></tr><tr><td>Red. ↓</td><td>0.034</td><td>0.308</td><td>+0.273</td><td>+0.148</td><td>+0.386</td><td>[0.098, 0.403]</td></tr></table>

TABLE XXV

SENSITIVITY EXCLUDING NYUV2. MEANS USE PERCENTAGES; DIFFERENCES AND INTERVALS USE PERCENTAGE POINTS. EVERY ROW COVERSEIGHTEEN RELATIONS AND NINE DATASETS.
<table><tr><td>Quantity</td><td>Relation Mean</td><td>Dataset Mean</td><td>95% Interval</td><td>Leave-One-Out Range</td></tr><tr><td>HSA Mean</td><td>32.13</td><td>31.84</td><td>[13.27, 49.78]</td><td>[26.67, 35.62]</td></tr><tr><td>Frozen Cosine Mean</td><td>18.64</td><td>21.24</td><td>[3.96, 37.38]</td><td>[13.56, 20.77]</td></tr><tr><td>HSA Minus Cosine</td><td>13.49</td><td>10.59</td><td>[5.73, 19.30]</td><td>[8.53, 14.87]</td></tr><tr><td>HSA Minus Full A-B</td><td>1.34</td><td>3.20</td><td>[-2.16, 6.98]</td><td>[-0.11, 4.22]</td></tr><tr><td>Intact Minus Permuted A-H</td><td>27.34</td><td>26.04</td><td>[10.05, 42.37]</td><td>[22.05, 30.46]</td></tr><tr><td>Intact Minus Permuted H-B</td><td>27.40</td><td>26.74</td><td>[10.38, 42.80]</td><td>[22.29, 30.54]</td></tr><tr><td>Intact Minus Both Permuted</td><td>25.29</td><td>23.09</td><td>[9.21, 37.89]</td><td>[20.70, 28.22]</td></tr><tr><td>Leading Minus Subsequent</td><td>16.67</td><td>14.35</td><td>[5.84, 26.20]</td><td>[12.65, 18.51]</td></tr><tr><td>Leading Minus Random</td><td>19.81</td><td>16.63</td><td>[6.72, 30.16]</td><td>[15.56, 22.01]</td></tr></table>

The auxiliary metrics follow distinct stages of retrieval: P-AUC assesses pair discrimination, C@1 assesses selection within carrier-localized candidates, G-AUC assesses gate ordering across relations, and Red. measures score overlap. C@1 should be interpreted alongside candidate coverage because its denominator includes only covered queries. Full-gallery Recall@10 evaluates all queries, connecting these conditional diagnostics to the complete retrieval task.

K.3 NYUv2 Evaluation Protocol and Cross-Dataset Robustness

NYUv2 uses shared frozen features across methods and controls within each task. Retrieval min–max normalizes depth per image and quantizes it to 8-bit values (0–255), subsequently interpreted as millimeters. Classification converts filled metric depth from meters to millimeters before encoding. RGB resize and tensor-conversion orders also differ across tasks. Each task is evaluated under its own protocol; cross-task fixed-state reuse covers the eight compatible relations in Appendix G.4. Excluding NYUv2 leaves eighteen relations across nine datasets (Table XXV). HSA reaches 32.13% mean Recall@10, exceeding frozen cosine by 13.49 points, with descriptive 95% datasetcluster interval [5.73, 19.30]. All intact-minus-permuted and leading-minus-control intervals remain positive. This subset preserves both the retrieval advantage and the key mechanism results, supporting cross-dataset robustness.

The two averaging schemes in Table XXV address complementary aspects of benchmark composition. Relation-equal averaging gives each retained target-pair configuration the same weight, whereas dataset-equal averaging first combines configurations within a dataset. TartanRGBT contributes one mean for six relations. The corresponding HSA means are 32.13% and 31.84%, with gains over cosine of 13.49 and 10.59 points. Every leave-one-dataset-out gain within this subset is positive, ranging from 8.53 to 14.87 points. Thus the improvement persists when datasets receive equal weight and when each remaining dataset is omitted in turn.