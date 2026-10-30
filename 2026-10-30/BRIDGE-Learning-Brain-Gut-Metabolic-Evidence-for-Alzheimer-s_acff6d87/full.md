![](images/c0b0a9dbeef5db946818d4dedfd6bbd6fd61eb7d711c0a223a19fe890bb740d9.jpg)  
b

# BRIDGE: Learning Brain-Gut Metabolic Evidence for Alzheimer’s Disease Diagnosis

Pengli Zhu, Member, IEEE, Ruxi Chang, Tianliang Yao, Yu Jiang, Shengyuan Liu, Yiyang Lin, Xiaoyi Duan, Shuang Wu, Yixuan Yuan, Senior Member, IEEE, and the Alzheimer’s Disease Neuroimaging Initiative

Abstract— Alzheimer’s disease (AD) diagnosis may benefit from gut metabolic information associated with brain– gut interactions, yet existing models largely analyze cerebral glucose metabolism alone. Although whole-body fluorodeoxyglucose positron emission tomography (FDG-PET) can provide paired brain–gut observations, routine brain examinations rarely include gut imaging, limiting direct use of this information for diagnosis. In this paper, we present BRIDGE (Brain and Gut Representation Integration via Distribution-Guided Enhancement), a two-stage framework for transferring brain–gut metabolic knowledge to brain-only diagnosis. Specifically, BRIDGE integrates a conditional brain-to-gut latent transport module and a disease-evidence-guided cerebral pathway to capture subject-specific cross-organ correspondence and diseaserelevant cerebral representations. For the cerebral pathway, a large language model (LLM) helps organize literaturederived descriptors into human-approved semantic anchors that guide feature learning. Additionally, we introduce a counterfactual-guided residual fusion mechanism, which uses matched and non-matched virtual-gut evidence to supervise a confidence gate and dynamically regulate the contribution of transported cross-organ information. At inference, BRIDGE exploits complementary gut-related context using brain PET alone, without measured gut PET. Extensive experiments across public and private cohorts demonstrate that BRIDGE outperforms brain-only and cross-organ baselines, achieving AUC gains of 6.0% and 5.2% over its brain-only counterpart for CN/MCI and CN/AD diagnosis, respectively. Notably, its ability to transfer complementary brain–gut metabolic context while preserving brain PET as the sole inference input highlights its potential for clinically practical diagnosis. Our code is available at https://github.com/CUHK-AIM-Group/BRIDGE.

Index Terms— FDG-PET, AD Diagnosis, Brain–Gut Representation Learning, Cross-organ Latent Transport.

![](images/af26e8d6bfb5128364740127f1083b3d1d917b144baf5a7212096c102f47491f.jpg)

![](images/74eeb3d8e90ccbec60600f278151d81d620678a41c09d7f672f615fb607f5d0b.jpg)  
Fig. 1. Conceptual illustration and performance of BRIDGE. (a) Brain–gut representation learning and virtual-gut transfer for brain-only inference. (b) AUC comparison between the brain-only and +gut models for AD and MCI diagnosis on the public and private cohorts, respectively.

## I. INTRODUCTION

Alzheimer’s disease (AD) is a progressive neurodegenerative disorder associated with cognitive decline and altered cerebral glucose metabolism [1]. Within this clinical spectrum, mild cognitive impairment (MCI) represents a clinically heterogeneous stage of cognitive decline and, in some individuals, may precede progression to AD. To characterize these metabolic changes, fluorodeoxyglucose positron emission tomography (FDG-PET) non-invasively assesses cerebral glucose metabolism and is widely used to evaluate ADrelated cognitive impairment [2], [3]. Recent computational methods have advanced FDG-PET-based diagnosis through convolutional, multiview, and radiomics-guided representation learning [4]–[6]. However, these approaches largely analyze cerebral metabolism in isolation, leaving complementary peripheral metabolic information underexplored.

This limitation motivates consideration of systemic metabolic interactions beyond the brain [7]. The microbiota– gut–brain axis provides a physiological basis for investigating such associations in AD-related cognitive impairment [8]–[11], while whole-body FDG-PET provides an imaging platform that simultaneously captures cerebral and peripheral glucose metabolism [12]. Building on this opportunity, recent studies have explored multi-organ representation learning for neurological assessment. BHGCL-guided B-GAT models brain– heart–gut interactions, while multi-constraint knowledge distillation transfers multi-organ information to a deployable brain-only model [13], [14]. MOGAD incorporates inter-organ alignment and hierarchical knowledge transfer, whereas the BHG/Positional framework leverages anatomical prompts and hierarchical multi-organ representations [15], [16]. Collectively, these studies establish the feasibility of using peripheral metabolic information to support neurological diagnosis while retaining brain PET as the sole input at deployment.

Nevertheless, developing a brain-only diagnostic framework that benefits from brain–gut metabolic information at the individual level remains challenging. First, brain and gut PET represent anatomically distinct organs without direct voxel-wise correspondence. Direct brain-to-gut image synthesis is therefore inherently underdetermined, while conventional feature transfer does not explicitly preserve subject-specific correspondence between inferred and measured gut representations. Second, disease-relevant cerebral evidence is not readily available as structured guidance for feature learning. Descriptions of regional uptake patterns and disease stages appear across the literature, while diagnosis labels alone do not explicitly align brain PET features with that evidence. Turning these descriptions into reliable semantic guidance requires literature-grounded organization and human review. Third, transported cross-organ information may not be equally reliable across subjects. Inaccurate inferred representations can introduce misleading evidence and compromise an otherwise reliable cerebral prediction. Therefore, an effective cross-organ framework requires mechanisms to learn subjectspecific brain–gut correspondence, organize cerebral features using disease evidence, and assess and regulate the diagnostic contribution of transported information.

To address these challenges, we propose Brain and Gut Representation Integration via Distribution-Guided Enhancement (BRIDGE), a two-stage framework for transferring cross-organ metabolic information to brain-only diagnosis. As illustrated in Fig. 1, BRIDGE separates brain– gut representation learning from diagnostic learning, allowing paired whole-body PET and labeled brain PET to be utilized across different cohorts. To capture subject-specific crossorgan correspondence without reconstructing gut anatomy, BRIDGE employs conditional brain-to-gut latent transport. For cerebral feature learning, a disease-evidence-guided pathway uses human-approved semantic anchors, organized from the literature with assistance from a large language model (LLM), to structure disease-relevant brain features. Finally, counterfactual-guided residual fusion contrasts matched and non-matched virtual-gut evidence and uses a confidence gate to regulate the contribution of transported information.

The key contributions are summarized as follows:

1) We formulate a two-stage cross-organ learning framework that transfers brain–gut metabolic knowledge across disjoint paired whole-body and labeled brain PET cohorts while requiring only brain PET at inference;

2) We introduce conditional latent transport for subjectspecific brain–gut correspondence in latent space, together with disease-evidence-guided cerebral learning based on human-approved semantic anchors organized with large language model assistance;

3) We develop a counterfactual-guided residual fusion mechanism that adaptively regulates virtual-gut evidence at inference, and show that the resulting framework improves diagnostic performance over brain-only and cross-organ methods across public and private cohorts.

The paper is organized as follows. Section II reviews related work on FDG-PET diagnosis, cross-organ learning, and missing-modality transfer. Section III presents BRIDGE and its optimization. Section IV describes the experimental settings and evaluation results. Finally, Section V concludes this work.

## II. RELATED WORK

## A. Brain FDG-PETDiagnosis

Deep learning has been widely applied to FDG-PET-based diagnosis for characterizing disease-related cerebral metabolic patterns [17]. Early studies primarily used convolutional networks to extract discriminative PET features [4], while subsequent work incorporated multiview learning and radiomicsguided representations [5], [6]. More recent architectures, including ConvNeXt, Vision Transformers, and Swin Transformers, further improve local and global feature modeling [18]– [20]. Semantic guidance has also been explored [21], [22], MedCLIP learns from unpaired image–text data, whereas MedKLIP injects entity-level descriptions into local visual representations [23], [24]. Despite these advances, existing approaches predominantly construct diagnostic representations from cerebral imaging alone and do not explicitly incorporate complementary metabolic information from peripheral organs.

## B. Cross-Organ Metabolic Learning

Whole-body PET enables systemic analysis of metabolic relationships across organs and has motivated recent work on multi-organ learning for neurological assessment [25]. Brain– heart–gut relationships have been modeled through graphbased learning and knowledge distillation [13], [14]. MO-GAD introduces inter-organ alignment and hierarchical knowledge transfer [15], while BHG/Positional leverages anatomical prompts for multi-organ learning [16]. These methods demonstrate that peripheral metabolic information can improve neurological assessment while preserving brain-only deployment.

Nevertheless, existing approaches largely rely on joint multi-organ modeling or teacher–student transfer and do not explicitly model subject-specific brain–gut correspondence when paired whole-body PET and labeled brain PET originate from disjoint cohorts. This setting motivates cross-organ knowledge transfer that can learn correspondence independently of downstream diagnostic supervision.

## C. Missing-Modality Representation Transfer

Missing-modality learning exploits complementary information unavailable at inference through shared representations, image synthesis, or knowledge distillation [26]–[28]. HeMIS and shared-specific modeling accommodate incomplete inputs [29], [30]. Image translation methods synthesize missing contrasts from available imaging modalities [31]–[33]. Knowledge distillation transfers information across modality combinations for reduced-input inference [34], [35].

Most existing methods consider modalities or contrasts of the same anatomy, where correspondence is relatively well defined [36]–[38]. Brain and gut PET instead represent anatomically distinct organs without direct voxel-wise correspondence. BRIDGE addresses this setting through representation-level transport that learns subject-specific cross-organ correspondence without anatomical reconstruction and adaptively regulates transported information during diagnosis.

![](images/0840f26704d88d12e5522fe98863696a8975f06405799af75d9a4d587a976500.jpg)  
Fig. 2. Overview of BRIDGE framework. Stage 1 uses paired whole-body PET to learn subject-specific brain-to-gut latent transport through organ-specific encoders and endpoint constraints. Stage 2 uses labeled brain PET to train a cerebral pathway guided by literature-derived, human approved semantic anchors and a virtual-gut residual pathway. An offline LLM assists in organizing the anchors, while matched and nonmatched gains supervise the confidence gate. At inference, a brain-only PET drives both pathways, and the gate regulates the virtual-gut contribution.

## III. METHODOLOGY

## A. Overview

In this paper, we formulate BRIDGE as a two-stage framework for learning subject-specific brain–gut correspondence from paired whole-body PET and transferring it to brain-only diagnosis. The two training cohorts are defined as follows:

$$
\begin{array} { r l r } { \mathcal { D } _ { \mathrm { W B } } = \{ ( \mathbf { x } _ { b } ^ { i } , \mathbf { x } _ { g } ^ { i } ) \} _ { i = 1 } ^ { N _ { \mathrm { W B } } } , } & { } & \\ { \mathcal { D } _ { \mathrm { B } } = \{ ( \mathbf { x } _ { b } ^ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { \mathrm { B } } } , } & { } & { y _ { i } \in \{ 1 , \dots , C \} , } \end{array}\tag{1}
$$

Here, $\mathbf { x } _ { b } ^ { i }$ and $\mathbf { x } _ { g } ^ { i }$ denote paired brain and gut PET crops, $y _ { i }$ is a diagnostic label among C classes, and i indexes the two cohorts independently. The whole-body cohort D<sub>WB</sub> provides paired observations without requiring diagnostic labels, whereas $\mathcal { D } _ { \mathrm { B } }$ provides labeled brain PET without requiring measured gut PET. With these cohorts, our goal is to estimate $p ( y \mid \mathbf { x } _ { b } )$ by transferring subject-specific brain–gut correspondence learned from $\mathcal { D } _ { \mathrm { W B } }$ to brain-only diagnosis on $\mathcal { D } _ { \mathrm { B } }$

As illustrated in Fig. 2, Stage 1 uses paired whole-body PET to learn organ-specific representations and a brain-togut transport operator $\mathcal { T } _ { \psi }$ , with paired endpoint regularization encouraging subject-specific latent correspondence. Stage 2 freezes this operator and combines a cerebral pathway guided by human-approved, LLM-assisted semantic anchors with a virtual-gut residual pathway through a confidence gate supervised during training by the relative diagnostic gains of matched and nonmatched virtual-gut evidence. For a brain scan, the cerebral pathway produces logits $\ell _ { b } \in \mathbb { R } ^ { C }$ , while the transported pathway produces residual logits $\Delta \ell _ { g } \in \mathbb { R } ^ { C }$ The gate $\alpha \in [ 0 , 1 ]$ controls the contribution of the residual:

$$
\ell _ { \mathrm { f i n a l } } = \ell _ { b } + \alpha \Delta \ell _ { g } , \quad { \bf p } _ { \mathrm { f i n a l } } = \mathrm { s o f t m a x } ( \ell _ { \mathrm { f i n a l } } ) .\tag{2}
$$

At inference, both pathways use the same brain scan, and the gate regulates the virtual-gut contribution rather than estimating a calibrated probability of diagnostic correctness.

## B. Stage 1: Brain-to-Gut Latent Transport

To learn subject-specific correspondence between anatomically distinct organs without reconstructing gut anatomy, Stage 1 transports brain representations toward their paired gut representations in a common latent space. To this end, the stage consists of organ-specific encoding, conditional latent transport, and paired endpoint regularization.

1) Organ-Specific Latent Encoding: To avoid imposing a shared anatomical representation on the brain and gut, we use separate three-dimensional encoders $E _ { b }$ and $E _ { g }$ followed by projection heads $P _ { b }$ and $P _ { g }$ . They map paired observations into a common d-dimensional transport space: $\mathbf { z } _ { b } =$

$P _ { b } ( E _ { b } ( \mathbf { x } _ { b } ) ) , \mathbf { z } _ { g } = P _ { g } ( E _ { g } ( \mathbf { x } _ { g } ) ) , \mathbf { z } _ { b } , \mathbf { z } _ { g } \in \mathbb { R } ^ { d }$ . The pairing provides subject identity rather than voxelwise correspondence. The gut encoder and projection define the target representation used to learn transport and are discarded after Stage 1.

2) Conditional Cross-Organ Flow: To model the cross-organ transformation without assuming a direct voxel-level mapping, we learn a continuous latent transport with conditional flow matching [39]–[41]. For subject i and $t \sim \mathcal { U } ( 0 , 1 )$ , a straight path connects the paired latents, and the vector field $v _ { \psi }$ is trained to recover its target velocity:

$$
\begin{array} { r l } & { \mathbf { z } _ { i , t } = ( 1 - t ) \mathbf { z } _ { b } ^ { i } + t \mathbf { z } _ { g } ^ { i } , \qquad \mathbf { u } _ { i } = \mathbf { z } _ { g } ^ { i } - \mathbf { z } _ { b } ^ { i } , } \\ & { \mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { i , t } \left\| v _ { \psi } ( \mathbf { z } _ { i , t } , t \mid \mathbf { z } _ { b } ^ { i } ) - \mathbf { u } _ { i } \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{3}
$$

Conditioning on the source latent $\mathbf { z } _ { b } ^ { i }$ keeps the transported trajectory tied to the input subject. At inference of the transport branch, integrating the learned field from the brain latent yields a deterministic virtual-gut endpoint:

$$
\frac { d \mathbf { z } ( t ) } { d t } = v _ { \psi } ( \mathbf { z } ( t ) , t \mid \mathbf { z } _ { b } ) , \quad \mathbf { z } ( 0 ) = \mathbf { z } _ { b } , \quad \hat { \mathbf { z } } _ { g } = \mathcal { T } _ { \psi } ( \mathbf { z } _ { b } ) = \mathbf { z } ( 1 ) .\tag{4}
$$

This endpoint is a representation-level estimate for downstream diagnosis; it is not intended to reconstruct gut anatomy or model the full conditional distribution of gut states.

3) Paired Endpoint Regularization: To encourage locally learned flow dynamics to reach a subject-consistent gut representation, we explicitly constrain the transported endpoint. Specifically, endpoint fidelity is measured by Euclidean and angular agreement, while a minibatch contrastive term [42] favors the paired gut latent over nonmatched subjects:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { e n d } } = \mathbb { E } _ { i } \left[ \| \hat { \mathbf { z } } _ { g } ^ { i } - \mathbf { z } _ { g } ^ { i } \| _ { 2 } ^ { 2 } + 1 - \cos ( \hat { \mathbf { z } } _ { g } ^ { i } , \mathbf { z } _ { g } ^ { i } ) \right] , } \\ & { \mathcal { L } _ { \mathrm { c t r } } = - \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( \sin ( \hat { \mathbf { z } } _ { g } ^ { i } , \mathbf { z } _ { g } ^ { i } ) / \tau ) } { \sum _ { j = 1 } ^ { B } \exp ( \sin ( \hat { \mathbf { z } } _ { g } ^ { i } , \mathbf { z } _ { g } ^ { j } ) / \tau ) } , } \end{array}\tag{5}
$$

where B is the minibatch size, $\cos ( \cdot , \cdot )$ and sim(·, ·) denote cosine similarity, and $\tau > 0$ is the contrastive temperature. We additionally use the trajectory-energy penalty $\begin{array} { r l } { \mathcal { L } _ { \mathrm { e n g } } } & { { } = } \end{array}$ <sup>E</sup><sub>i</sub> $\begin{array} { r } { \int _ { 0 } ^ { 1 } \| v _ { \psi } ( \mathbf { z } _ { i } ( t ) , t \mid \mathbf { z } _ { b } ^ { i } ) \| _ { 2 } ^ { 2 } } \end{array}$ dt to discourage unnecessarily large integrated velocities. Therefore, the complete objective for Stage 1 training is formulated as:

$$
\mathcal { L } _ { \mathrm { S 1 } } = \mathcal { L } _ { \mathrm { F M } } + \lambda _ { \mathrm { e n d } } \mathcal { L } _ { \mathrm { e n d } } + \lambda _ { \mathrm { c t r } } \mathcal { L } _ { \mathrm { c t r } } + \lambda _ { \mathrm { e n g } } \mathcal { L } _ { \mathrm { e n g } } .\tag{6}
$$

All Stage 1 terms use paired imaging observations only and require no diagnostic labels.

## C. Stage 2: Evidence-Guided Virtual-Gut Fusion

To exploit transported cross-organ context without allowing unreliable virtual-gut evidence to dominate diagnosis, Stage 2 combines an evidence-guided cerebral pathway with a gated virtual-gut residual. The Stage 1 brain encoder, projection, and transport operator remain fixed throughout diagnostic learning.

1) Disease-Evidence Anchor Feature Modulation: To organize cerebral features around disease-relevant metabolic evidence, we use literature-derived descriptors as semantic queries over brain tokens. The diagnostic encoder produces $\dot { \textbf { T } } = \mathbf { \nabla } E _ { d } ( \mathbf { x } _ { b } ) = [ \mathbf { t } _ { 1 } , \dots , \mathbf { t } _ { N } ] \in \mathbf { \nabla } \mathbb { R } ^ { N \times d }$ , where N is the number of spatial tokens. To provide this guidance, the fixed descriptor bank contains $K = 8$ human-approved statements organized offline from literature passages by metabolic pattern, anatomical location, uptake direction, and disease stage. No patient images were provided to the offline LLM. For reproducibility, the archived manifest retains the retrieval date (August 23, 2026), the fixed prompt, four PubMed/PubMed Central queries, seven source records with persistent identifiers and verification status, and the recorded LLM provenance.

Let $\{ q _ { k } \} _ { k = 1 } ^ { K }$ denote these descriptors. Each descriptor is encoded with the BiomedBERT base checkpoint [43] using attention-mask mean pooling, projected from 768 dimensions to $d = 2 5 6$ by a fixed Gaussian projection with seed 2026, and $\ell _ { 2 }$ normalized. The complete checkpoint identifier and revision hash are retained in the anchor-embedding artifact. This produces the fixed initialization $\mathbf { A } ^ { 0 } = [ \mathbf { a } _ { 1 } ^ { 0 } , \dots , \mathbf { \bar { a } } _ { K } ^ { 0 } ] \in \mathbb { R } ^ { K \times d } ;$ a learnable residual gives $\mathbf { A } = \mathbf { A } ^ { 0 } + \Delta \mathbf { A }$ . We refine these anchors with scan-specific brain evidence and use the refined anchors to modulate brain tokens through cross-attention:

$$
\begin{array} { r l } & { \mathbf H = \mathbf A + \mathrm { A t t n } ( \mathbf A , \mathbf T , \mathbf T ) , } \\ & { { \widetilde \mathbf T } = \mathbf T + \mathrm { A t t n } ( \mathbf T , \mathbf H , \mathbf H ) , } \\ & { { \widetilde \mathbf f } _ { b } = \Phi _ { b } \Big ( [ \mathrm { P o o l } ( \widetilde \mathbf T ) , \mathrm { P o o l } ( \mathbf H ) ] \Big ) , \quad \ell _ { b } = C _ { b } ( \mathbf f _ { b } ) . } \end{array}\tag{7}
$$

The scan-conditioned anchors H act as semantic queries rather than anatomical masks, while Te contains the modulated brain tokens. In Eq. (7), Pool denotes token pooling, $\Phi _ { b }$ maps the pooled representations to cerebral features, and $C _ { b }$ produces the cerebral logits. To preserve the anchors’ literature-derived directions while allowing task adaptation, the cerebral objective augments the cross-entropy classification loss $\mathcal { L } _ { \mathrm { C E } }$ with semantic-preservation and diversity regularization:

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { b r a i n } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \mathcal { L } _ { \mathrm { C E } } ( \ell _ { b , i } , y _ { i } ) + \lambda _ { \mathrm { s e m } } \mathcal { L } _ { \mathrm { s e m } } + \lambda _ { \mathrm { d i v } } \mathcal { L } _ { \mathrm { d i v } } , } } \\ & { \mathcal { L } _ { \mathrm { s e m } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \big [ 1 - \cos ( { \bf a } _ { k } , { \bf a } _ { k } ^ { 0 } ) \big ] , } & \\ & { \mathcal { L } _ { \mathrm { d i v } } = \| \bar { \bf A } \bar { \bf A } ^ { \top } - { \bf I } \| _ { F } ^ { 2 } , } & \end{array}\tag{8}
$$

where each row of A<sup>¯</sup> is the $\ell _ { 2 } \cdot$ -normalized version of the corresponding row of A, and I is the $K \times K$ identity matrix. The regularizers encourage semantic preservation and nonredundant anchor directions but do not by themselves guarantee interpretable specialization.

2) Fixed Virtual-Gut Evidence Stream: To expose complementary gut-related context from the same brain scan, the frozen Stage 1 branch maps the input brain PET to a virtual-gut latent, which is converted to a residual correction by a trainable head $C _ { g } \mathbf { : } \mathbf { z } _ { b } = P _ { b } ( E _ { b } ( \mathbf { x } _ { b } ) ) , \hat { \mathbf { z } } _ { g } = \mathcal { T } _ { \psi } ( \mathbf { z } _ { b } ) , \Delta \ell _ { g } = C _ { g } ( \hat { \mathbf { z } } _ { g } ) \in$ $\mathbb { R } ^ { C }$ . Accordingly, only $C _ { g }$ is optimized in this stream during Stage 2; the transported representation itself remains fixed. The residual logits are not hard-bounded, and their magnitude is controlled by the fusion objective below.

3) Counterfactual-Guided Residual Fusion: To guide residual admission, we compare matched and nonmatched virtualgut evidence within each diagnostic class and use their diagnostic gains to construct a soft target for the confidence gate. For subject i, the cerebral logits are held fixed while its matched virtual-gut latent is replaced by the latent of another record $j ( i ) \neq i$ from the same diagnostic class:

$$
\begin{array} { r l } & { \ell _ { i } ^ { \mathrm { m } } = \ell _ { b , i } + C _ { g } ( \hat { \mathbf { z } } _ { g , i } ) , } \\ & { \ell _ { i } ^ { \mathrm { n m } } = \ell _ { b , i } + C _ { g } ( \hat { \mathbf { z } } _ { g , j ( i ) } ) . } \end{array}\tag{9}
$$

Within each minibatch, same-class members are randomly permuted and circularly shifted to construct $j ( i )$ . Samples without a same-class partner are excluded from gate supervision and the matched-gain penalty but remain in the classification and residual-regularization objectives.

Let $S ( \ell , y ) ~ = ~ \mathrm { { l o g } [ s o f t m a x ( \ell ) ] } _ { y }$ denote the correct-class log probability. We measure how much the matched and nonmatched residuals improve this evidence relative to the cerebral prediction and convert their difference into a soft admission target as follows:

$$
\begin{array} { r } { g _ { i } ^ { \mathrm { m } } = S ( \ell _ { i } ^ { \mathrm { m } } , y _ { i } ) - S ( \ell _ { b , i } , y _ { i } ) , ~ } \\ { g _ { i } ^ { \mathrm { n m } } = S ( \ell _ { i } ^ { \mathrm { n m } } , y _ { i } ) - S ( \ell _ { b , i } , y _ { i } ) , ~ } \\ { c _ { i } ^ { * } = \sigma \Big ( \frac { g _ { i } ^ { \mathrm { m } } - \operatorname* { m a x } ( g _ { i } ^ { \mathrm { n m } } , 0 ) } { \tau _ { g } } \Big ) . ~ } \end{array}\tag{10}
$$

Thus, the admission target $c _ { i } ^ { * } .$ , whose softness is controlled by the positive temperature $\tau _ { g } ,$ exceeds 0.5 only when the matched residual provides positive gain and exceeds the corresponding nonmatched gain. This target guides gate learning with diagnostic labels during training, enabling confidence estimation from brain-derived features at inference.

To adaptively determine whether the transported evidence should contribute to the final prediction, we predict a confidence score from the cerebral feature, virtual-gut latent, residual logits, and cerebral predictive entropy. Let $\begin{array} { r l } { h _ { i } } & { { } = } \end{array}$ $\mathcal { H } ( \operatorname { s o f t m a x } ( \ell _ { b , i } ) )$ denote the predictive entropy. The gate and final logits are then computed as:

$$
\begin{array} { r l } & { \mathbf { r } _ { i } = [ \mathbf { f } _ { b , i } , \hat { \mathbf { z } } _ { g , i } , \Delta \ell _ { g , i } , h _ { i } ] , } \\ & { \qquad \alpha _ { i } = \sigma ( G _ { \omega } ( \mathbf { r } _ { i } ) ) , } \\ & { \qquad \ell _ { \mathrm { f i n a l } , i } = \ell _ { b , i } + \alpha _ { i } \Delta \ell _ { g , i } . } \end{array}\tag{11}
$$

where $G _ { \omega }$ is a trainable scalar-output gate network and σ is the logistic sigmoid. To supervise residual admission, we detach the confidence target as $\bar { c } _ { i } = \mathrm { s t o p g r a d } ( c _ { i } ^ { * } )$ and combine gate supervision with a matched-gain penalty over the valid withinbatch replacements V:

$$
\begin{array} { c } { { \mathcal { L } _ { \mathrm { g a t e } } = \displaystyle \frac { 1 } { B } \sum _ { i \in \mathcal { V } } \mathrm { B C E } ( \alpha _ { i } , \bar { c } _ { i } ) , \hfill } } \\ { { \mathcal { L } _ { \mathrm { c f } } = \displaystyle \frac { 1 } { B } \sum _ { i \in \mathcal { V } } [ - g _ { i } ^ { \mathrm { m } } ] _ { + } . \hfill } } \end{array}\tag{12}
$$

where BCE denotes binary cross-entropy and $\begin{array} { r l } { [ u ] _ { + } } & { { } = } \end{array}$ $\operatorname* { m a x } ( u , 0 )$ , with both losses set to zero when V is empty. The matched-gain penalty ${ \mathcal { L } } _ { \mathrm { c f } }$ discourages reductions in correctclass evidence relative to the cerebral prediction. Gradients propagate through the predicted gate and matched gain, while the confidence target remains detached. These terms are complemented by final-prediction classification and residual

magnitude regularization:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { c l s } } = \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \mathcal { L } _ { \mathrm { C E } } ( \ell _ { \mathrm { f i n a l } , i } , y _ { i } ) , } \\ & { \mathcal { L } _ { \mathrm { r e s } } = \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \| \Delta \ell _ { g , i } \| _ { 2 } ^ { 2 } . } \end{array}\tag{13}
$$

These terms jointly supervise the fused prediction, guide residual admission, and control residual magnitude through the overall fusion objective:

$$
{ \mathcal { L } } _ { \mathrm { f u s i o n } } = { \mathcal { L } } _ { \mathrm { c l s } } + \lambda _ { \mathrm { c f } } { \mathcal { L } } _ { \mathrm { c f } } + \lambda _ { \mathrm { g a t e } } { \mathcal { L } } _ { \mathrm { g a t e } } + \lambda _ { \mathrm { r e s } } { \mathcal { L } } _ { \mathrm { r e s } } .\tag{14}
$$

where the weights $\lambda _ { \mathrm { c f } } , \lambda _ { \mathrm { g a t e } } ,$ and $\lambda _ { \mathrm { { r e s } } }$ balance the contributions of the matched-gain penalty, gate-supervision loss, and residual regularization, respectively.

## D. Optimization and Brain-Only Inference

To preserve the separation between cross-organ correspondence learning and disease supervision, we optimize the two stages sequentially. Stage 1 minimizes ${ \mathcal { L } } _ { \mathrm { S 1 } }$ in Eq. (6) using paired whole-body PET. We then freeze $E _ { b } , P _ { b } ,$ , and $\mathcal { T } _ { \psi }$ and discard $E _ { g }$ and $P _ { g } .$ . Before joint Stage 2 training, the cerebral branch is warmed up with $\mathcal { L } _ { \mathrm { b r a i n } }$ . The diagnostic components are subsequently optimized with $\mathcal { L } _ { \mathrm { S 2 } } = \mathcal { L } _ { \mathrm { b r a i n } } + \lambda _ { \mathrm { f u s i o n } } \mathcal { L } _ { \mathrm { f u s i o n } }$

At inference, a single brain PET scan is processed by the diagnostic branch and the frozen transport branch. Their outputs are combined according to Eq. (2). Accordingly, measured gut PET, diagnostic labels, nonmatched replacement samples, and the offline LLM are not required.

TABLE I  
COHORT COMPOSITION AND EXPERIMENTAL ROLES. BG AND B DENOTE PAIRED BRAIN–GUT PET AND BRAIN PET, RESPECTIVELY. N DENOTES THE NUMBER OF EXAMINATIONS.
<table><tr><td>Cohort</td><td>Modality</td><td>Labels</td><td>N</td><td>Role</td></tr><tr><td>AutoPET [44]</td><td>BG</td><td>Unlabeled</td><td>188</td><td>Transport pretrain</td></tr><tr><td>PETWB [45]</td><td>BG</td><td>Unlabeled</td><td>484</td><td>Transport pretrain</td></tr><tr><td>QUADRA [46]</td><td>BG</td><td>Unlabeled</td><td>96</td><td>Transport pretrain</td></tr><tr><td>ENHANCE [47]</td><td>BG</td><td>Unlabeled</td><td>12</td><td>Transport pretrain</td></tr><tr><td>ADNI [48]</td><td>B</td><td>CN/MCI/AD</td><td>1,004</td><td>Source training validation</td></tr><tr><td>Private cohort</td><td>BG</td><td>CN/MCI</td><td>278</td><td>Primary evaluation</td></tr></table>

## IV. EXPERIMENTS

## A. Setup and Details

1) Study Cohorts: Table I summarizes the three complementary data resources used in our experiments. Four public whole-body FDG-PET/CT cohorts, including QUADRA [46], AutoPET [44], PETWB [45], and ENHANCE [47], provide 780 unlabeled paired brain–gut examinations for transport pretraining. ADNI [48] provides 1,004 labeled brain examinations spanning CN, MCI, and AD for source training and evaluation. The private cohort contains 278 paired examinations (138 CN and 140 MCI) for adaptation and held-out CN/MCI evaluation.

Private-cohort PET/CT acquisition followed a 6-h fast and confirmation of blood glucose below 11.1 mmol/L. After intravenous $^ { 1 8 } \mathrm { F - F D G }$ administration (3.7 MBq/kg) and 60 min of rest, PET was acquired at eight bed positions (1.5 min each). PET and CT used 5-mm slices and $1 2 8 \times 1 2 8$ and $5 1 2 \times 5 1 2$ matrices, respectively, with three-iteration time-of-flight PET reconstruction followed by PET/CT fusion. Subsequent cohort selection required cerebral and abdominopelvic coverage, valid PET/CT pairing, and usable CT-derived gut masks, excluding patients with confirmed brain or gastrointestinal tumors. The MCI group additionally required clear signs of age-related cerebral changes. Quality control excluded examinations with inadequate coverage, artifacts, invalid spatial metadata, or failed registration or mask checks. These criteria defined the examination-level cohort used for subsequent model adaptation and held-out CN/MCI evaluation.

TABLE II  
PRIVATE-COHORT CN/MCI CLASSIFICATION PERFORMANCE (MEAN±SD OVER FIVE FOLDS). B AND BG DENOTE BRAIN AND BRAIN–GUT PET. <sup>∗</sup> INDICATES SIGNIFICANCE BY TWO-SIDED PAIRED t-TEST WITH HOLM CORRECTION $\mathbf { \Sigma } ( p < \mathbf { 0 . 0 5 } )$ . BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td>Method</td><td>Category</td><td>AUC↑</td><td>ACC↑</td><td>BACC↑</td><td>SEN↑</td><td>SPE↑</td><td>Macro-F1↑</td></tr><tr><td colspan="8">Brain-only baselines</td></tr><tr><td>ConvNeXt [18]</td><td>B/B</td><td> $\underline { { 0 . 8 3 7 } } \pm 0 . 0 5 0$ </td><td> $0 . 7 2 8 { \pm } 0 . 0 4 0 ^ { \ast }$ </td><td> $0 . 7 2 7 { \scriptstyle \pm 0 . 0 3 9 ^ { * } }$ </td><td> $\underline { { 0 . 8 1 6 \pm 0 . 1 4 9 } }$ </td><td> $0 . 6 3 8 { \scriptstyle \pm 0 . 1 0 1 } ^ { * }$ </td><td> $0 . 7 2 6 { \pm } 0 . 0 4 1 ^ { \ast }$ </td></tr><tr><td>3D ViT [19]</td><td>B/B</td><td> $0 . 6 8 4 { \pm } 0 . 0 7 0 ^ { \ast }$ </td><td> $0 . 6 2 6 { \pm } 0 . 0 6 8 ^ { \ast }$ </td><td> $0 . 6 2 6 { \pm } 0 . 0 6 8 ^ { \ast }$ </td><td> $0 . 6 2 9 { \scriptstyle \pm 0 . 0 9 0 } ^ { * }$ </td><td> $0 . 6 2 4 { \scriptstyle \pm 0 . 0 9 1 } ^ { * }$ </td><td> $0 . 6 2 5 { \scriptstyle \pm 0 . 0 6 9 ^ { * } }$ </td></tr><tr><td>Swin-T [20]</td><td>B/B</td><td> $\underline { { 0 . 8 3 7 } } \underline { { \pm 0 . 0 6 0 } }$ </td><td> $0 . 7 5 6 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $0 . 7 5 6 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $0 . 7 4 3 { \scriptstyle \pm 0 . 0 9 2 }$ </td><td> $0 . 7 6 9 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $0 . 7 5 5 { \scriptstyle \pm 0 . 0 6 0 }$ </td></tr><tr><td>MiSePyNet [5]</td><td>B/B</td><td> $0 . 7 3 6 { \pm } 0 . 0 5 5 ^ { \ast }$ </td><td> $0 . 6 5 1 { \scriptstyle \pm 0 . 0 7 5 ^ { \ast } }$ </td><td> $0 . 6 5 3 { \scriptstyle \pm 0 . 0 7 5 ^ { * } }$ </td><td> $0 . 4 9 3 { \pm } 0 . 2 0 3 ^ { \ast }$ </td><td> $0 . 8 1 2 { \pm } 0 . 1 3 2 $ </td><td> $0 . 6 3 2 { \scriptstyle \pm 0 . 0 9 5 ^ { \ast } }$ </td></tr><tr><td colspan="8">Privileged-training comparators</td></tr><tr><td>MOGAD [15]</td><td>BG/B</td><td> $0 . 7 9 2 { \scriptstyle \pm 0 . 0 8 4 } ^ { * }$ </td><td> $0 . 7 2 4 { \pm } 0 . 1 0 0 ^ { \ast }$ </td><td> $0 . 7 2 4 { \scriptstyle \pm 0 . 1 0 1 ^ { * } }$ </td><td> $0 . 7 2 1 { \pm } 0 . 1 5 9$ </td><td> $0 . 7 2 7 { \scriptstyle \pm 0 . 1 8 8 }$ </td><td> $0 . 7 1 9 { \scriptstyle \pm 0 . 1 0 4 ^ { * } }$ </td></tr><tr><td>BHG/Positional [16]</td><td>BG/B</td><td> $0 . 8 1 4 { \scriptstyle \pm 0 . 0 6 9 ^ { \ast } }$ </td><td> $0 . 7 2 3 { \scriptstyle \pm 0 . 0 6 7 \scriptscriptstyle * }$ </td><td> $0 . 7 2 2 { \scriptstyle \pm 0 . 0 6 6 } ^ { * }$ </td><td> $0 . 7 8 6 { \pm } 0 . 1 4 9$ </td><td> $0 . 6 5 9 { \scriptstyle \pm 0 . 0 8 6 } ^ { * }$ </td><td> $0 . 7 2 0 { \scriptstyle \pm 0 . 0 6 7 \scriptscriptstyle * }$ </td></tr><tr><td colspan="8">Proposed method</td></tr><tr><td>Brain-only</td><td>B/B</td><td> $0 . 8 3 1 { \pm } 0 . 0 3 0$ </td><td> $\underline { { 0 . 7 6 6 \pm 0 . 0 3 2 } }$ </td><td> $\underline { { 0 . 7 6 7 } } \pm 0 . 0 3 2$ </td><td> $0 . 7 6 4 { \scriptstyle \pm 0 . 0 8 6 } ^ { \ast }$ </td><td> $0 . 7 6 9 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $\underline { { 0 . 7 6 6 \pm 0 . 0 3 1 } }$ </td></tr><tr><td>BRIDGE</td><td>BG/B</td><td> $\mathbf { 0 . 8 8 1 { \pm } 0 . 0 5 0 }$ </td><td> $\mathbf { 0 . 8 1 8 { \pm } 0 . 0 2 8 }$ </td><td> $\mathbf { 0 . 8 1 9 \pm 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 8 2 1 { \scriptstyle \pm 0 . 0 6 7 } }$ </td><td> $\mathbf { 0 . 8 1 6 { \overset { . } { = } } 0 . 0 2 9 }$ </td><td> $\mathbf { 0 . 8 1 5 { \scriptstyle \pm 0 . 0 2 7 } }$ </td></tr></table>

2) Data Preprocessing: Brain preprocessing included spatial harmonization, intensity normalization, and quality control. T1-weighted MRI, when available, underwent N4 biasfield correction, affine registration to the MNI template with ANTs, and skull stripping with SynthStrip. Cerebral PET was rigidly registered to the corresponding MRI, transformed to MNI space using the MRI-derived transformation, normalized by pons-referenced uptake, and masked to remove extracranial signal. Registered volumes were visually inspected for anatomical coverage, alignment, and orientation.

Gut preprocessing included CT-based segmentation, PETspace mapping, resampling, and intensity normalization. Gut masks were generated from whole-body CT using TotalSegmentator [49] by combining the duodenum, small intestine, colon, sigmoid colon, and rectum, and were mapped to PET space after rigid CT–PET registration. Gut PET was resampled to $1 \times 1 \times 1 , \mathrm { { m m ^ { 3 } } }$ , while cerebral PET from all cohorts was affinely harmonized to a reference with ANTsPyX and visually checked. Cerebral and gut volumes were resized to $9 6 \times 1 1 2 \times 9 6$ and $9 6 \times 6 4 \times 1 2 8$ voxels, respectively, and normalized using robust scan-wise intensity normalization.

3) Implementation Details: BRIDGE was implemented in Python 3.10 and PyTorch 2.6 on NVIDIA H200 NVL GPUs. Source initialization used 60 epochs, followed by privatecohort fine-tuning for up to 70 epochs with patience 15. The AdamW optimizer used learning rates of $2 \times 1 0 ^ { - 4 }$ and $5 \times 1 0 ^ { - 5 }$ for the two stages, respectively, with weight decay $1 0 ^ { - 4 }$ and batch size eight. Mixed precision and gradient clipping at 5.0 were used throughout. The transport solver used eight explicit Euler steps with midpoint vector-field evaluation. For the source cerebral training, $\lambda _ { \mathrm { s e m } } = 0 . 0 5$ and $\lambda _ { \mathrm { d i v } } = 0 . 0 1$ , while both weights were zero during private brain-head adaptation. The reported fusion runs used $\lambda _ { \mathrm { c f } } ~ = ~ 0 . 2 , ~ \lambda _ { \mathrm { g a t e } } ~ = ~ 0 . 2 ,$ $\lambda _ { \mathrm { r e s } } = 0 . 0 0 5 ,$ , and $\tau _ { g } = 0 . 2$ with the counterfactual objective. 4) Evaluation Metrics: The primary evaluation task was CN/MCI classification on the private cohort, with MCI treated as the positive class. Following a five-fold diagnosis-stratified protocol, approximately 70%, 10%, and 20% of the data were assigned to training, validation, and testing, respectively. All methods used identical disjoint partitions, with model selection and classification thresholds determined on the validation set. For all methods, inference was performed using brain PET alone. We report the area under the receiver operating characteristic curve (AUC), accuracy (ACC), balanced accuracy (BACC), sensitivity (SEN), specificity (SPE), and macro-F1, expressed as mean±SD across the five test folds. For exploratory ablation analyses, statistical uncertainty was assessed using 5,000 fold- and diagnosis-stratified bootstrap resamples, with Holm correction for multiple comparisons.

## B. Comparison With State-of-the-Arts

1) Primary Private-Cohort Evaluation: To determine whether transferred cross-organ information improves brain-only diagnosis, we first evaluate BRIDGE on the primary privatecohort CN/MCI task. As shown in Table II, BRIDGE reaches an AUC of 0.881 and BACC of 0.819, representing relative gains of 6.0% and 6.8% over its protocol-matched brain-only counterpart. Statistical significance is assessed using two-sided paired student t-tests [50] across the five matched outer folds with Holm correction for multiple comparisons. A superscript ∗ denotes adjusted $p \ < \ 0 . 0 5$ . Among competing methods, ConvNeXt and Swin-T share the highest AUC at 0.837. Importantly, BRIDGE obtains these gains while using brain PET alone during private-cohort adaptation and testing, whereas the privileged-training comparators additionally require paired private brain–gut PET during their teacher stages.

TABLE III  
ADNI CN/AD CLASSIFICATION PERFORMANCE (MEAN±SD OVER FIVE FOLDS). B AND BG DENOTE BRAIN AND BRAIN–GUT PET. <sup>∗</sup> INDICATES SIGNIFICANCE BY TWO-SIDED PAIRED t-TEST WITH HOLM CORRECTION $\mathbf { \left( \right)} p < \mathbf { 0 . 0 5 } $ . BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td>Method</td><td>Category</td><td>AUC↑</td><td>ACC↑</td><td>BACC↑</td><td>SEN↑</td><td>SPE↑</td><td> $\mathbf { M a c r o - F 1 } \uparrow$ </td></tr><tr><td colspan="8">Brain-only baselines</td></tr><tr><td>ConvNeXt [18]</td><td>B/B</td><td> $0 . 7 4 0 { \scriptstyle \pm 0 . 0 3 9 } ^ { * }$ </td><td> $0 . 6 6 1 { \scriptstyle \pm 0 . 0 5 1 } ^ { * }$ </td><td> $0 . 6 6 0 { \scriptstyle \pm 0 . 0 4 9 } ^ { * }$ </td><td> $0 . 6 3 1 { \scriptstyle \pm 0 . 1 0 2 ^ { * } }$ </td><td> $0 . 6 9 0 { \scriptstyle \pm 0 . 1 6 5 ^ { * } }$ </td><td> $0 . 6 5 6 { \scriptstyle \pm 0 . 0 5 0 ^ { * } }$ </td></tr><tr><td>3D ViT [19]</td><td>B/B</td><td> $0 . 6 9 1 { \scriptstyle \pm 0 . 0 6 3 ^ { * } }$ </td><td> $0 . 5 9 9 { \scriptstyle \pm 0 . 0 6 1 } ^ { * }$ </td><td> $0 . 5 9 6 { \scriptstyle \pm 0 . 0 6 4 } ^ { * }$ </td><td> $0 . 4 4 3 { \scriptstyle \pm 0 . 2 6 7 \cdots }$ </td><td> $0 . 7 5 0 { \scriptstyle \pm 0 . 1 5 5 ^ { \ast } }$ </td><td> $0 . 5 6 1 { \scriptstyle \pm 0 . 1 3 0 ^ { * } }$ </td></tr><tr><td>Swin-T [20]</td><td>B/B</td><td> $0 . 6 9 3 { \scriptstyle \pm 0 . 0 7 2 ^ { * } }$ </td><td> $0 . 6 1 9 { \scriptstyle \pm 0 . 0 7 7 ^ { * } }$ </td><td> $0 . 6 1 8 { \pm } 0 . 0 8 0 \ ^ { \ast }$ </td><td> $0 . 5 5 1 { \scriptstyle \pm 0 . 2 5 1 } ^ { * }$ </td><td> $0 . 6 8 4 { \pm } 0 . 1 6 8 ^ { \ast }$ </td><td> $0 . 6 0 1 { \scriptstyle \pm 0 . 1 0 2 ^ { * } }$ </td></tr><tr><td>MiSePyNet [5]</td><td>B/B</td><td> $0 . 9 3 5 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 8 6 9 { \pm } 0 . 0 2 1 $ </td><td> $0 . 8 6 9 { \pm } 0 . 0 2 0 $ </td><td> $0 . 8 8 3 { \pm } 0 . 0 6 9$ </td><td> $0 . 8 5 5 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $0 . 8 6 9 { \pm } 0 . 0 2 1 $ </td></tr><tr><td colspan="8">Privileged-training comparators</td></tr><tr><td>MOGAD [15]</td><td>BG/B</td><td> $\underline { { 0 . 9 3 6 { \pm } 0 . 0 2 0 } }$ </td><td> $0 . 8 3 5 { \scriptstyle \pm 0 . 0 5 1 { } ^ { * } }$ </td><td> $0 . 8 3 6 { \pm } 0 . 0 4 9 ^ { \ast }$ </td><td> $0 . 8 6 3 { \scriptstyle \pm 0 . 1 2 9 }$ </td><td> $0 . 8 1 0 { \scriptstyle \pm 0 . 1 7 4 ^ { * } }$ </td><td> $0 . 8 3 2 { \scriptstyle \pm 0 . 0 5 5 ^ { \ast } }$ </td></tr><tr><td>BHG/Positional [16]</td><td>BG/B</td><td> $0 . 8 8 3 { \pm } 0 . 1 9 3$ </td><td> $0 . 8 2 8 { \scriptstyle \pm 0 . 1 5 6 } ^ { * }$ </td><td> $0 . 8 2 5 { \scriptstyle \pm 0 . 1 5 1 } ^ { * }$ </td><td> $0 . 8 7 8 { \scriptstyle \pm 0 . 1 4 1 }$ </td><td> $0 . 7 7 1 { \scriptstyle \pm 0 . 3 8 7 \cdots }$ </td><td> $0 . 7 9 5 { \scriptstyle \pm 0 . 2 2 4 } ^ { * }$ </td></tr><tr><td colspan="8">Proposed method</td></tr><tr><td>Brain-only</td><td>B/B</td><td> $0 . 9 0 9 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 8 6 3 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $0 . 8 6 7 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $0 . 8 5 3 { \scriptstyle \pm 0 . 1 1 2 ^ { * } }$ </td><td> $0 . 8 8 1 { \pm } 0 . 1 1 4$ </td><td> $0 . 8 6 1 { \scriptstyle \pm 0 . 0 4 4 }$ </td></tr><tr><td>BRIDGE</td><td>BG/B</td><td> $\mathbf { 0 . 9 5 6 { \scriptstyle \pm 0 . 0 2 2 } }$ </td><td> $\mathbf { 0 . 9 0 8 { \scriptstyle \pm 0 . 0 3 9 } }$ </td><td> $\mathbf { 0 . 9 0 7 } \pm \mathbf { 0 . 0 3 9 }$ </td><td> $\mathbf { 0 . 9 0 3 \pm 0 . 1 0 5 }$ </td><td> $\mathbf { 0 . 9 1 1 { \pm } 0 . 0 9 7 }$ </td><td> $\mathbf { 0 . 9 0 3 { \scriptstyle \pm 0 . 0 4 1 } }$ </td></tr></table>

(a) Sensitivity--specificity operating profile  
![](images/bdffe2c55d650f4762983db8c38047f44a7783b1dbdd707e028a9f13449f56b9.jpg)

(b) Error cost as missed-MCI penalty varies  
![](images/542e1d2cc3019c46c8da00e749d09895a9377e7341cede0063eef48c0f79b6d6.jpg)  
Fig. 3. Private-cohort operating-point analysis. (a) Sensitivity and specificity across methods. Markers show five-fold means, error bars show SD, and marker size denotes AUC. The diagonal indicates equal sensitivity and specificity. (b) Expected error cost per 100 scans from five-fold mean sensitivity and specificity and cohort proportions, plotted against the missed-MCI penalty relative to false alarms. Red denotes BRIDGE, gray the comparators, and the navy dashed curve their lowest cost. Both axes are logarithmic, with lower error costs indicating better performance and the dotted line at 1 marking equal penalties.

2) Source-Domain Evaluation: As a complementary sourcedomain evaluation, we compare the same method families on

ADNI CN/AD diagnosis. As shown in Table III, BRIDGE achieves an AUC of 0.956, corresponding to a 5.2% relative gain over its matched brain-only model. Its BACC reaches 0.907, compared with 0.869 for MiSePyNet, while the leading privileged-training comparator attains an AUC of 0.936. Statistical significance is assessed using the same paired testing protocol and significance criterion described above. Together with the private-cohort results, these findings support the applicability of the proposed framework to both CN/AD and CN/MCI diagnosis across public and private cohorts.

3) Operating-Point Analysis: To further examine operating behavior, Fig. 3(a) shows the sensitivity–specificity profile of each method. Consistent with Table II, BRIDGE is the only method that improves all six metrics over its protocol-matched brain-only model, reaching sensitivity of 0.821 and specificity of 0.816. ConvNeXt approaches the same sensitivity but has substantially lower specificity, whereas MiSePyNet shows the opposite imbalance. Due to the potentially unequal costs of missed MCI and false alarms, Fig. 3(b) tests this advantage across relative penalties. BRIDGE retains the lowest expected cost, yielding 5.2 fewer expected errors per 100 scans than the best comparator at equal penalties. At unchanged validationselected thresholds, this pattern suggests that brain–gut transfer improves the balance between MCI detection and false alarms.

## C. Ablation and Transport Analysis

1) Component-Level Ablation: To assess the roles of the major components, we compare removal and replacement variants under the same private-cohort protocol, with the brainonly variant providing the matched reference for assessing the added virtual-gut pathway. As shown in Table IV, removing latent transport produces the largest AUC decline among single-component variants, from 0.881 to 0.845, and lowers sensitivity from 0.821 to 0.757. Removing counterfactualconfidence supervision lowers BACC from 0.819 to 0.781, whereas fixed residual admission reduces specificity from 0.816 to 0.783. Evidence-anchor removal has a more modest effect, lowering BACC from 0.819 to 0.802. Overall, the results suggest complementary contributions from transport, semantic guidance, and adaptive residual fusion.

t-SNE Component 1  
TABLE IV  
ABLATION STUDY ON PRIVATE-COHORT CN/MCI CLASSIFICATION. RESULTS ARE $\mathsf { M E A N \pm S D }$ OVER FIVE HELD-OUT FOLDS. COMPONENT-LEVEL VARIANTS MODIFY INDIVIDUAL MODULES, WHEREAS OBJECTIVE-LEVEL VARIANTS MODIFY INDIVIDUAL LOSS TERMS.
<table><tr><td>Variant</td><td>AUC↑</td><td>ACC↑</td><td>BACC↑</td><td>SEN↑</td><td>SPE↑</td><td>Macro-F1↑</td></tr><tr><td>Full BRIDGE</td><td> $\mathbf { 0 . 8 8 1 \pm 0 . 0 5 0 }$ </td><td>0.818±0.028</td><td> $\mathbf { 0 . 8 1 9 \pm 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 8 2 1 { \scriptstyle \pm 0 . 0 6 7 } }$ </td><td> $\mathbf { 0 . 8 1 6 { \overset { . } { = } } 0 . 0 2 9 }$ </td><td> $\mathbf { 0 . 8 1 5 { \scriptstyle \pm 0 . 0 2 7 } }$ </td></tr><tr><td>Component-level ablations</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Brain-only</td><td> $0 . 8 3 1 { \pm } 0 . 0 3 0$ </td><td> $0 . 7 6 6 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 7 6 7 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 7 6 4 { \scriptstyle \pm 0 . 0 8 6 }$ </td><td> $0 . 7 6 9 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $0 . 7 6 6 { \scriptstyle \pm 0 . 0 3 1 }$ </td></tr><tr><td>Without evidence anchors</td><td> $0 . 8 7 5 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $0 . 8 0 2 { \pm } 0 . 0 2 8$ </td><td> $0 . 8 0 2 { \pm } 0 . 0 2 8$ </td><td> $0 . 8 1 4 { \scriptstyle \pm 0 . 0 7 3 }$ </td><td> $0 . 7 9 0 { \scriptstyle \pm 0 . 0 5 7 }$ </td><td> $0 . 8 0 2 { \scriptstyle \pm 0 . 0 2 7 }$ </td></tr><tr><td>Without latent transport</td><td> $0 . 8 4 5 { \scriptstyle \pm 0 . 0 5 4 }$ </td><td> $0 . 7 7 4 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 7 7 4 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 7 5 7 { \pm } 0 . 0 3 0$ </td><td> $0 . 7 9 0 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td> $0 . 7 7 4 { \scriptstyle \pm 0 . 0 3 1 }$ </td></tr><tr><td>Fixed residual admission  $( \alpha = 1 )$ </td><td> $0 . 8 7 6 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 7 9 1 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 7 9 1 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 8 0 0 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 7 8 3 { \scriptstyle \pm 0 . 0 3 7 }$ </td><td> $0 . 7 9 1 { \scriptstyle \pm 0 . 0 0 8 }$ </td></tr><tr><td>Without counterfactual-confidence supervision</td><td> $0 . 8 6 5 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 7 8 1 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 7 8 1 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 7 7 1 { \scriptstyle \pm 0 . 0 8 2 }$ </td><td> $0 . 7 9 0 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $0 . 7 8 0 { \scriptstyle \pm 0 . 0 3 1 }$ </td></tr><tr><td>Objective-level ablations</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Without matched-gain penalty</td><td> $0 . 8 6 1 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 7 8 8 { \pm } 0 . 0 2 2$ </td><td> $0 . 7 8 3 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 8 1 1 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $0 . 7 5 4 { \pm } 0 . 0 3 7$ </td><td> $0 . 7 8 7 { \scriptstyle \pm 0 . 0 2 2 }$ </td></tr><tr><td>Without gate-supervision loss</td><td> $0 . 8 7 2 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 7 8 4 { \pm } 0 . 0 2 4$ </td><td> $0 . 7 8 4 { \pm } 0 . 0 2 4$ </td><td> $0 . 7 9 3 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $0 . 7 7 5 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $0 . 7 8 4 { \pm } 0 . 0 2 4$ </td></tr><tr><td>Without residual regularization</td><td> $0 . 8 5 4 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 7 7 3 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 7 7 4 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 7 9 3 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $0 . 7 5 4 { \pm } 0 . 0 5 0$ </td><td>0.773±0.027</td></tr><tr><td>Focal classification loss  $( \gamma = 2 )$ </td><td> $0 . 8 7 1 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $0 . 7 8 1 { \pm } 0 . 0 3 8$ </td><td> $0 . 7 8 1 { \pm } 0 . 0 3 8$ </td><td> $0 . 7 6 4 { \scriptstyle \pm 0 . 0 9 3 }$ </td><td> $0 . 7 9 8 { \pm } 0 . 0 8 5$ </td><td> $0 . 7 8 0 { \pm } 0 . 0 3 8$ </td></tr></table>

(a) Transport alignment  
![](images/38fcd54b133da356ae28d41cebd4157a2f05883af04136a6cab9496e1976ef79.jpg)  
(c) ADNI: CN/AD

(b) Paired gain  
![](images/d9ad5e849502b822acff1024a43c929995fbf463af17e8eb45ecfc21701f457c.jpg)  
(d) Private: CN/MCI

![](images/8dc8a6749e0cb4b97942fe1543d0497daae3aa7756641bd398bcefdcbe25fb50.jpg)

![](images/bf0e4318e669bb040b4586ed9299db191df031ac63ee8c1dd3dd32693e62db0b.jpg)  
Fig. 4. Cross-organ transport alignment and task-specific feature geometry. (a) Cosine similarities of cerebral and transported virtualgut latents to matched measured-gut latents $\mathbf { \Sigma } ( n = \mathbf { 2 7 8 } )$ . The dashed line marks equality. (b) Paired gains after transport, with the point and bar indicating the mean and stratified-bootstrap 95% CI, respectively. (c) ADNI CN/AD fold-1 training features (182 CN, 174 AD). (d) Privatecohort CN/MCI fold-1 training features (96 CN, 98 MCI).

2) Objective-Level Ablation: To assess the contribution of the optimization objective, we examine objective-level variants while keeping the full architecture unchanged. Table IV shows that removing residual regularization lowers BACC from 0.819 to 0.774 and specificity from 0.816 to 0.754. Removing gate-supervision loss reduces BACC by 4.3% relative to the full objective, while focal loss yields an AUC of 0.871 but sensitivity of only 0.764. The complete objective therefore leads across all six mean metrics under the current protocol. Together, these ablations suggest that matched-gain penalization, gate supervision, and residual regularization contribute in complementary ways to a balanced operating profile.

3) Cross-Organ Transport Analysis: Beyond classification performance, we assess whether the frozen transport moves cerebral representations toward subject-matched measured-gut counterparts. As shown in Fig. 4(a) and (b), mean cosine similarity to the measured-gut latent increases from 0.8467 to 0.8491 after transport. All 278 paired examinations show positive gains (mean $\Delta = 0 . 0 0 2 3 3 ;$ stratified-bootstrap 95% CI, $0 . 0 0 2 2 4 - 0 . 0 0 2 4 3 )$ , corresponding to a 0.276% relative increase in mean cosine similarity. The modest magnitude is consistent with a subtle, subject-specific representationlevel adjustment rather than a large transformation or explicit anatomical reconstruction. Measured gut PET provides a subject-matched reference for transport alignment, while adaptation and inference rely on brain PET alone.

## D. Representation and Interpretability Analysis

1) Task-Specific Feature Geometry: To characterize the organization of the learned diagnostic features, we visualize taskspecific t-SNE projections of gated BRIDGE representations from the ADNI CN/AD and private CN/MCI training subsets in Fig. 4(c) and (d). The embeddings are generated independently for the two diagnostic tasks and are used only to examine within-task representation geometry and class organization. The ADNI CN/AD representation shows comparatively clear group separation, whereas the private CN/MCI representation exhibits a broader transition between diagnostic categories. Neighborhood label agreement remains high $( > 0 . 9 0 )$ in both tasks, indicating diagnosis-related local organization despite the subtler CN/MCI distinction. Together, these visualizations illustrate diagnostic group structure in both training cohorts, complementing the quantitative classification results.

![](images/40f3fa5b33eace90d64846138bcd336b516675c22eba43cd3fc6e399aa785578.jpg)  
Fig. 5. Out-of-fold regional attribution on the private cohort. Circular bars show the ten AAL3 regions with the highest mean normalized positive integrated-gradient attribution to the final MCI logit, with scores rescaled to the top-ranked region. Surrounding surfaces localize the corresponding atlas parcels. Results aggregate all 278 examinations and reflect model-attributed diagnostic evidence rather than causal anatomy or imaging biomarkers.

2) Regional Attribution Analysis: To localize the cerebral evidence underlying private-cohort predictions, we aggregate out-of-fold integrated-gradient attribution for the final MCI logit within AAL3 regions and summarize the most influential regions in Fig. 5. The attribution is concentrated in posterior cerebral regions, with the left calcarine and inferior occipital cortices showing the strongest contributions and additional involvement of lingual, cuneal, and bilateral occipital regions. Bilateral calcarine, inferior occipital, and middle occipital parcels, together with lingual and cuneal regions, form a coherent posterior visual-cortical grouping, while left cerebellar Crus II contributes outside this cortical cluster. This spatial organization links model attribution to identifiable anatomical regions, clarifying how regional cerebral responses contribute to the model’s MCI classification decisions.

3) Brain–Gut Grad-CAM Visualization: To examine the spatial responses underlying individual predictions, Fig. 6 presents an ADNI AD brain examination and a paired private MCI brain–gut examination in three orthogonal views. Both brain maps highlight posterior responses, while the MCI example also shows an elongated central pattern in the axial view. These distributions illustrate the cerebral regions contributing to the model’s predictions. Complementing the brain maps, the measured-gut map from the same private case highlights localized responses of the trained residual pathway, providing a post-hoc anatomical reference for interpreting gut-related evidence. Together, these maps complement the quantitative results with localized cerebral and gut-residual responses, clarifying BRIDGE’s brain-only diagnostic framework.

![](images/323798f9de4cdb597e2b84be6916a301853ac1a910ce6048174b91b02f42ad6d.jpg)  
Fig. 6. Illustrative brain–gut Grad-CAM maps. (a) ADNI AD brain PET, with attribution to the AD-class diagnostic logit. (b) Private-cohort MCI brain PET, with attribution to the MCI-class diagnostic logit. (c) Matched measured-gut PET from the same private examination, probed post hoc through the trained virtual-gut residual head. Columns show axial, sagittal, and coronal views.

## V. CONCLUSION

This study presents BRIDGE, a framework that transfers metabolic knowledge from paired whole-body PET to cognitive impairment diagnosis using brain PET alone. BRIDGE integrates conditional brain-to-gut latent transport, cerebral representation learning guided by disease evidence, and counterfactual-guided residual fusion to model subject-specific correspondence between brain and gut representations and adaptively regulate the diagnostic contribution of virtual gut evidence without requiring measured gut PET at inference. By separating cross-organ representation learning from downstream diagnostic supervision, the framework enables paired whole-body PET and labeled brain PET from distinct cohorts to contribute to a unified diagnostic model. Experiments across public and private cohorts demonstrate that BRIDGE achieves AUCs of 0.881 and 0.956 for CN/MCI and CN/AD diagnosis, representing relative improvements of 6.0% and 5.2%, respectively, over its corresponding brain PET models evaluated under the same protocol, while outperforming existing cross-organ approaches. Ablation and representation analyses support the complementary roles of latent transport, semantic guidance, and adaptive residual fusion. These findings demonstrate the feasibility of learning transferable brain–gut metabolic correspondence from heterogeneous data resources and incorporating it into brain-only diagnosis. Future work will assess the robustness and generalizability of BRIDGE across independent cohorts and imaging protocols.

[1] P. Scheltens, B. De Strooper et al., “Alzheimer’s disease,” Lancet, vol. 397, no. 10284, pp. 1577–1590, 2021.

[2] W. Zhang, S. Raghavan et al., “Divergent white matter metabolic signature patterns indicate impending cognitive decline in aging and dementia,” Nature Commun., vol. 17, no. 1, p. 4152, 2026.

[3] L. Barnard, H. Botha et al., “An FDG-PET-based machine learning framework to support neurologic decision-making in alzheimer disease and related disorders,” Neurology, vol. 105, no. 2, p. e213831, 2025.

[4] Y. Ding, J. H. Sohn et al., “A deep learning model to predict a diagnosis of alzheimer disease by using 18f-fdg pet of the brain,” Radiology, vol. 290, no. 2, pp. 456–464, 2019.

[5] X. Pan, T.-L. Phan et al., “Multi-view separable pyramid network for AD prediction at MCI stage by 18f-FDG brain PET imaging,” IEEE Trans. Med. Imag., vol. 40, no. 1, pp. 81–92, 2021.

[6] J. Jiang, C. Li et al., “Using interpretable deep learning radiomics model to diagnose and predict progression of early AD disease spectrum: A preliminary [18f]FDG PET study,” Eur. Radiol., vol. 35, no. 5, pp. 2620– 2633, 2025.

[7] M. Arnold, M. Buyukozkan et al., “Individual bioenergetic capacity as a potential source of resilience to alzheimer’s disease,” Nature Commun., vol. 16, no. 1, p. 1910, 2025.

[8] J. S. Loh, W. Q. Mak et al., “Microbiota-gut-brain axis and its therapeutic applications in neurodegenerative diseases,” Signal Transduct. Target. Ther., vol. 9, no. 1, p. 37, 2024.

[9] T. E. Ohara and E. Y. Hsiao, “Microbiota–neuroepithelial signalling across the gut–brain axis,” Nat. Rev. Microbiol., vol. 23, no. 6, pp. 371– 384, 2025.

[10] L. Jia, Y. Ke et al., “Metagenomic analysis characterizes stage-specific gut microbiota in alzheimer’s disease,” Mol. Psychiatry, vol. 30, no. 9, pp. 3951–3962, 2025.

[11] V. Vemuganti, J. W. Kang et al., “Gut bacterial metabolite imidazole propionate potentiates alzheimer’s disease pathology,” Nature Commun., vol. 17, no. 1, p. 8017, 2026.

[12] B. K. Geist, J. Guthrie et al., “The metabolic organ connectome: A novel approach to measure allostatic load during health-to-disease transition,” Med, vol. 6, no. 12, p. 100881, 2025.

[13] F. Li, S. Bai et al., “Enhancing diagnosis of mild cognitive impairment through brain-heart-gut metabolic networks in whole-body pet imaging,” Cell Rep. Med., vol. 7, no. 2, p. 102629, 2026.

[14] F. Li, S. Zhao et al., “Brain-heart-gut guided multi-constraint knowledge distillation for early alzheimer’s disease diagnosis,” in Proc. MICCAI, 2025, pp. 45–54.

[15] S. Zhao, F. Li et al., “Multi-organ guided diagnosis of mild cognitive impairment via hierarchical alignment and knowledge distillation,” Med. Image Anal., vol. 113, p. 104176, 2026.

[16] F. Li, S. Zhao et al., “Positional prompts-enhanced brain-heart-gut interactions for mild cognitive impairment diagnosis,” IEEE Trans. Med. Imag., vol. 45, no. 7, pp. 3598–3609, 2026.

[17] Y. Sun, Y. Chen et al., “Diagnostic performance of deep learning-assisted [18f]fdg pet imaging for alzheimer’s disease: a systematic review and meta-analysis,” Eur. J. Nucl. Med. Mol. Imaging, vol. 52, no. 10, pp. 3600–3612, 2025.

[18] S. Woo, S. Debnath et al., “ConvNeXt V2: Co-designing and scaling convnets with masked autoencoders,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 16 133–16 142.

[19] A. Dosovitskiy, L. Beyer et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.

[20] Z. Liu, Y. Lin et al., “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 9992–10 002.

[21] V. Nath, W. Li et al., “VILA-M3: Enhancing vision-language models with medical expert knowledge,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 14 788–14 798.

[22] Q. Chen, J. Wang et al., “HoloDx: Knowledge- and data-driven multimodal diagnosis of alzheimer’s disease,” IEEE Trans. Med. Imag., vol. 45, no. 1, pp. 206–219, 2026.

[23] Z. Wang, Z. Wu et al., “MedCLIP: Contrastive learning from unpaired medical images and text,” in Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP), 2022, pp. 3876–3887.

[24] C. Wu, X. Zhang et al., “MedKLIP: Medical knowledge enhanced language-image pre-training for x-ray diagnosis,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 21 315–21 326.

[25] A. Labarthe, S. Varet et al., “Personalized mapping of body homeostasis using whole-body pet connectomics and routine fdg pet imaging,” Commun. Med., vol. 6, no. 1, p. 294, 2026.

[26] K. Han, D. Hu et al., “Incomplete multi-modal disentanglement learning with application to alzheimer’s disease diagnosis,” IEEE Trans. Med. Imag., vol. 45, no. 2, pp. 632–645, 2026.

[27] C. Liu et al., “Multi-view hilbert curve-based hierarchical information aggregation for incomplete multimodal alzheimer’s disease diagnosis,” IEEE Trans. Med. Imag., vol. 45, no. 7, pp. 3948–3961, 2026.

[28] G. Yang, Y. Du et al., “Taming arbitrary modality missingness and imbalance: A unified graph-MoE framework for alzheimer’s disease diagnosis,” Med. Image Anal., vol. 113, p. 104222, 2026.

[29] M. Havaei, N. Guizard et al., “HeMIS: Hetero-modal image segmentation,” in Proc. MICCAI, 2016, pp. 469–477.

[30] H. Wang, Y. Chen et al., “Multi-modal learning with missing modality via shared-specific feature modelling,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 15 878–15 887.

[31] K. Chen, Y. Weng et al., “A multi-view learning approach with diffusion model to synthesize FDG PET from MRI T1WI for diagnosis of alzheimer’s disease,” Alzheimer’s Dement., vol. 21, no. 2, p. e14421, 2025.

[32] P. Zhu, Y. Fu et al., “Q-space guided multi-modal translation network for diffusion-weighted image synthesis,” IEEE Trans. Med. Imag., vol. 45, no. 3, pp. 1167–1178, 2026.

[33] Y. Zhang, C. Peng et al., “Unified multi-modal image synthesis for missing modality imputation,” IEEE Trans. Med. Imag., vol. 44, no. 1, pp. 4–18, 2025.

[34] M. G. Kwak, L. Mao et al., “A cross-modal mutual knowledge distillation framework for alzheimer’s disease diagnosis: Addressing incomplete modalities,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 14 218–14 233, 2025.

[35] C. Wang et al., “E<sup>2</sup>AD: Enhanced and explainable alzheimer’s disease detection framework via anatomy- and relation-aware cross-modal knowledge distillation,” Med. Image Anal., vol. 112, p. 104099, 2026.

[36] X. Zhang, J. Liang et al., “Structure-aware MRI translation: Multimodal latent diffusion model with arbitrary missing modalities,” in Proc. MICCAI, 2025, pp. 508–518.

[37] Y. Wang, H. Xiong et al., “UniSyn: A generative foundation model for universal medical image synthesis across MRI, CT and PET,” in Proc. MICCAI, 2025, pp. 673–682.

[38] P. Zhu, C. Liu et al., “Cycle-conditional diffusion model for noise correction of diffusion-weighted images using unpaired data,” Med. Image Anal., vol. 103, p. 103579, 2025.

[39] Y. Lipman, R. T. Q. Chen et al., “Flow matching for generative modeling,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[40] M. Yazdani, Y. Medghalchi et al., “Flow matching for medical image synthesis: Bridging the gap between speed and quality,” in Proc. MICCAI, 2025, pp. 216–226.

[41] J. Chemseddine, P. Hagemann et al., “Conditional wasserstein distances with applications in bayesian OT flow matching,” J. Mach. Learn. Res., vol. 26, no. 141, pp. 1–47, 2025.

[42] A. van den Oord, Y. Li et al., “Representation learning with contrastive predictive coding,” 2018, arXiv:1807.03748.

[43] Y. Gu, R. Tinn et al., “Domain-specific language model pretraining for biomedical natural language processing,” ACM Trans. Comput. Healthcare, vol. 3, no. 1, pp. 2:1–2:23, 2022.

[44] S. Gatidis, T. Hepp et al., “A whole-body FDG-PET/CT dataset with manually annotated tumor lesions,” Sci. Data, vol. 9, no. 1, p. 601, 2022.

[45] L. Xue, G. Feng et al., “PETWB-REP: A multi-cancer whole-body FDG PET/CT dataset with corresponding radiology reports,” Sci. Data, vol. 13, no. 1, p. 675, 2026.

[46] S. Gutschmayer, J. Yu et al., “Whole-body [18f]FDG-PET/CT imaging of healthy controls: Test/retest data for systemic, multi-organ analysis,” Sci. Data, vol. 12, no. 1, p. 1707, 2025.

[47] D. Ferrara, M. Pires et al., “Sharing a whole-/total-body [18f]FDG-PET/CT dataset with CT-derived segmentations: An ENHANCE.PET initiative,” Sci. Data, vol. 13, no. 1, p. 869, 2026.

[48] C. R. Jack, Jr., M. A. Bernstein et al., “The alzheimer’s disease neuroimaging initiative (adni): Mri methods,” J. Magn. Reson. Imaging, vol. 27, no. 4, pp. 685–691, 2008.

[49] J. Wasserthal, H.-C. Breit et al., “Totalsegmentator: robust segmentation of 104 anatomic structures in ct images,” Radiol. Artif. Intell., vol. 5, no. 5, p. e230024, 2023.

[50] B. Efron and T. Hastie, Computer age statistical inference, student edition: algorithms, evidence, and data science. Cambridge University Press, 2021, vol. 6.