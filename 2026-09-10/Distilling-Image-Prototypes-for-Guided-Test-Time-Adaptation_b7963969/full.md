# Distilling Image Prototypes for Guided Test-Time Adaptation

Liwen Wang, Xingbo Dong, Iman Yi Liao, Deyin Liu, Member, IEEE,

Massimo Tistarelli, Lin Yuanbo Wu, Senior Member, IEEE, Zhe Jin, Member, IEEE

Abstract—Test-Time Adaptation (TTA) enhances the robustness of models against distribution shifts but faces two critical challenges: error accumulation from noisy pseudo-labels and catastrophic forgetting of source knowledge. Uncertaintybased approaches designed to mitigate error accumulation often yield overconfident or computationally expensive estimates, while strategies intended to prevent forgetting via prototype replay rely on static representations that easily become misaligned as the model adapts. To address these issues, this paper proposes a novel framework, Distilling Image Prototype for Guided Test-Time Adaptation (DIPTTA). The core of the proposed approach is the introduction of a Distill Image Prototype (DIP), a compact set of synthetic images that serves as a dynamic and regenerative anchor of source knowledge. This prototype enables a dynamic feature replay mechanism that continuously generates feature prototypes aligned with the current state of the model, thus effectively preventing catastrophic forgetting. Furthermore, the DIP anchors a source-calibrated uncertainty estimation method, which provides a less biased measure of sample reliability by leveraging stable source knowledge, thereby robustly suppressing error accumulation. Extensive experiments on multiple benchmarks demonstrate that DIPTTA significantly outperforms state-of-theart methods, particularly under severe domain shifts. The source code is available at https://github.com/LiwenWang919/DIPTTA.

Index Terms—Test-time adaptation, distilling image prototype, dynamic replay, source-calibrated uncertainty estimation.

## I. INTRODUCTION

D <sup>EEP</sup> <sup>learning</sup> <sup>models</sup> <sup>struggle</sup> <sup>when</sup> <sup>test</sup> <sup>data</sup> <sup>shifts</sup> <sup>from</sup>their training distribution. Test-Time Adaptation (TTA) their training distribution. Test-Time Adaptation (TTA) addresses this by enabling online adaptation using unlabeled data [1]–[4]. However, TTA faces two major hurdles: error accumulation from noisy pseudo-labels and the catastrophic forgetting of original source knowledge. Current strategies to combat catastrophic forgetting often involve replaying information from the source domain. A prominent line of work [5], [6] advocates for replaying pre-computed feature prototypes, which are class-wise average feature vectors derived from the source data. This line of research also includes methods that maintain a source-like feature bank or use nearest-neighbor retrieval [7]. However, these approaches have a critical flaw: these prototypes are computed once using the pretrained source model and remain static throughout the adaptation process. As the model updates its parameters to the target domain, its internal feature representation evolves, causing a growing misalignment between the static prototypes and the current feature space of the model. Consequently, these outdated pro totypes provide a progressively potentially misleading anchor, thereby failing to prevent forgetting effectively. This fundamental limitation motivates the first research question: How can a replay-capable knowledge anchor be established that dynamically evolves with the model to alleviate catastrophic forgetting in a continually shifting environment?

Conversely, mitigating error accumulation typically relies on uncertainty estimation to identify and down-weight unreliable predictions. However, existing methods face a significant dilemma. Entropy-based techniques [8]–[10] are prone to overconfidence, especially on out-of-distribution data, as they only capture the sharpness of the output distribution, not the true epistemic uncertainty of the model [1], [2], [11], [12]. While more principled Bayesian methods [13], [14] can capture this uncertainty, they are frequently computationally prohibitive for online TTA. Recent efforts to improve their efficiency, such as through last-layer variational inference [15] or simplified Laplace approximations [16], still face challenges in non-stationary TTA settings. More critically, both types of estimates are inherently biased in the TTA context because they are computed solely based on the reaction of the model to the target data, which is itself non-stationary and potentially noisy. There is no stable reference point from the source domain to ground these estimates. This leads to the second research question: How can a stable source-domain anchor be leveraged to achieve unbiased uncertainty estimation, thereby reliably suppressing error accumulation?

It is identified that the root cause of both limitations is the lack of a dynamic, reliable, and efficient source of knowledge during adaptation. To bridge this gap, this paper introduces a concept dubbed Distilling Image Prototype (DIP). A DIP is not a static feature vector but a compact set of synthetic images distilled from the entire source dataset. This distillation process refines synthetic images so that a model trained on them closely resembles the performance of a model trained on the complete source data [17]. The key advantage of the DIP is its regenerability: it exists in the image space, allowing it to be continuously processed through the updated feature extractor during adaptation. This enables the on-demand generation of feature prototypes that are inherently aligned with the current feature space of the model, thereby solving the misalignment issue of static prototypes.

Guided by this concept, the Distilling Image Prototype for Guided Test-Time Adaptation (DIPTTA) framework is proposed. The DIP acts as a unified anchor, enabling two key innovations. First, to establish a replay-capable knowledge anchor that dynamically evolves with the model to prevent catastrophic forgetting in a continually shifting environment, the feature prototypes are regenerated for each incoming batch of target data by forwarding the DIP through the current student model. These dynamic prototypes are then used in a contrastive loss to pull the features of pseudo-labeled target samples towards their corresponding class anchors, ensuring that the model retains source knowledge without feature space distortion. Second, to leverage a stable source-domain anchor to achieve unbiased uncertainty estimation, thereby reliably suppressing error accumulation, the predictive uncertainty is computed by observing the response of the model (specifically, the classifier head) to the feature prototypes generated from the DIP. This measures the consistency between the target sample and the stable source knowledge, rather than the selfconfidence of the model regarding the target sample alone. The resulting uncertainty measure is inherently unbiased and serves as a robust weight to modulate the contribution of each sample to the adaptation loss, effectively curating the learning signal. The key contributions can be summarized as follows:

We introduce the novel concept of a Distilling Image Prototype (DIP) for test-time adaptation. A DIP is a compact set of synthetic images that serve as a dynamic and regenerative knowledge anchor, overcoming the rigidity of static source representations and providing a principled solution to knowledge preservation in nonstationary environments.

• We develop a Dynamic Feature Prototype Replay (DFPR) mechanism based on the DIP. This mechanism continuously regenerates feature prototypes aligned with the model’s current state, effectively addressing catastrophic forgetting in continual test-time adaptation without access to the data of the source domain.

• We propose a Source-Calibrated Uncertainty Estimation (SCUE) method that leverages the DIP to debias uncertainty quantification. By anchoring the uncertainty measure in the stable source knowledge, our method provides a more reliable signal for weighting target samples, leading to robust adaptation against error accumulation.

• Extensive experiments on multiple benchmarks demonstrate that our proposed DIP framework achieves state-ofthe-art performance. The consistent improvements across various scenarios, especially under strong domain shifts, validate the effectiveness and robustness of our approach.

## II. RELATED WORK

## A. Test Time Adaptation

Test-Time Adaptation (TTA) methods aim to adapt models via objectives applied directly to the test data. A dominant strategy has been output-level adaptation, which includes entropy minimization [8], [9], [18] to encourage confident predictions, and self-training [7], [19]–[21], which uses highconfidence pseudo-labels for supervision. These methods are highly susceptible to error accumulation from noisy pseudolabels, especially under significant domain shifts.

To enhance robustness, another line of work focuses on feature-level adaptation [22]–[25]. This includes methods based on consistency regularization, such as the Mean Teacher paradigm [24], [26]–[28], which stabilizes predictions by enforcing agreement between a teacher model and a student model under different augmentations. Furthermore, contrastive learning techniques [29], [30] have been adopted to learn a structured feature space with compact class clusters, improving feature discrimination in the target domain. While these methods promote feature invariance, they still fundamentally rely on the quality of pseudo-labels or the predictions of the teacher model, which can be unreliable for OOD samples [5].

A critical challenge in TTA, particularly in the continual setting (CTTA) [31], [32], is catastrophic forgetting. To address this without accessing source data, recent methods have explored replaying source knowledge [33]. Notably, several approaches [5], [6] replay pre-computed, static feature prototypes of the source classes. However, a fundamental limitation persists: as the model adapts, its feature representation evolves, causing these statically computed prototypes to become progressively misaligned with the current feature space [34]. This misalignment undermines their effectiveness as a stable knowledge anchor. Dataset distillation is rapidly becoming a research hotspot in machine learning, motivated primarily by the increasing scale of data and substantial model training overhead. By generating small yet potent synthetic datasets, this technology paves the way for more efficient learning and has demonstrated considerable value in numerous scenarios [35], including its application in continual learning [36], [37]. In this paper, dataset distillation is introduced into TTA to alleviate the problem of catastrophic forgetting; a replay mechanism for latent prototypes driven by dataset distillation is established, and its feasibility is verified through experiments. The proposed DIPTTA framework addresses the above limitations through a novel paradigm centered on a Distilling Image Prototype (DIP). Unlike static feature prototypes, the DIP is a set of synthetic images that allows for the dynamic regeneration of feature prototypes aligned with the evolving model, effectively overcoming the misalignment issue. This dynamic replay mechanism is coupled with a source-calibrated uncertainty estimation method that leverages the DIP to debias uncertainty quantification, offering a more robust solution to both catastrophic forgetting and error accumulation.

## B. Uncertainty Quantification in Test Time Adaptation

Uncertainty estimation is crucial in TTA to identify unreliable predictions under domain shifts and to prevent error accumulation [38]–[40]. Existing methods can be broadly categorized by their underlying principles. Simple predictive metrics, such as the prediction entropy utilized in methods like Tent [8] and SHOT [41], are computationally lightweight.

![](images/bc7e785605a0088eb8e75d4638ee2455bf53d74beb095167119de3db84ad76ac.jpg)  
Fig. 1. Overall framework of our proposed DIPTTA. The top part visualizes the Distilling Image Prototype (DIP) module, which generates high-fidelity synthetic prototypes via a two-loop distillation strategy. The bottom part depicts the core online adaptation pipeline, where a student model is continuousl updated with the guidance of a teacher model. Specifically, a source-calibrated uncertainty estimation and reweighting mechanism is embedded to dynamically calibrate the gradient contribution from target data, effectively mitigating distribution shift issues in open-world adaptation scenarios.

However, these approaches primarily reflect the sharpness of the output distribution and often fail to account for the epistemic uncertainty of the model. This leads to overconfident and miscalibrated predictions on out-of-distribution data. Principled Bayesian methods offer a more formal framework for capturing epistemic uncertainty. Techniques such as MC Dropout [13], model ensembles [42], [43], and Bayesian neural networks [14], [44] model distributions over parameters or predictions. While these methods can provide bettercalibrated uncertainty estimates, their computational demands often hinder their application in online TTA scenarios. Other lightweight strategies include using the maximum softmax probability [45] or energy scores [46] as uncertainty proxies.

A more fundamental limitation common to both categories is that they estimate uncertainty based solely on the target domain and the adapting model. The model itself is changing, and the target data is non-stationary. This lack of a stable reference point means that the uncertainty estimates can be inherently biased by the shifting target distribution, lacking calibration to the source knowledge [47], [48]. This limitation is addressed by introducing a source-calibrated uncertainty framework. Instead of relying exclusively on the output of the adapting model for a target sample, the Distilling Image Prototype (DIP) is leveraged as a stable anchor representing source knowledge. This enables the estimation of the consistency of a target sample with the source domain, resulting in a less biased measure of reliability. By grounding the uncertainty quantification in the stable source representation, the proposed method provides a more robust mechanism for identifying unreliable predictions during adaptation.

## III. METHOD

In this section, we detail our method, Distill Image Prototype-guided Test-Time Adaptation (DIPTTA).

## A. Notation and Model Architecture

Let $\mathcal { D } _ { S } ~ = ~ \{ ( x _ { i } ^ { S } , y _ { i } ^ { S } ) \}$ be the source dataset, with samples from a source domain distribution $P _ { \cal { S } } ( x , y )$ . Similarly, $\mathcal { D } _ { T } = \{ ( x _ { i } ^ { T } ) \}$ is the unlabeled target dataset from a target domain distribution $P _ { \cal T } ( x )$ . The fundamental challenge in this setting is that the source distribution is different from the target distribution, i.e., $P s ( x , y ) \neq P \tau ( x ) P ( y | x )$ , although they share a common label space Y. Given a well-trained model $M _ { S }$ pretrained on the full source data $\mathcal { D } _ { \mathcal { S } }$ , the goal is to adapt $M _ { S }$ to the target distribution under the test-time adaptation setting, which adapts the model using sequential unlabeled target batches without access to source data.

1) Model Architecture: To achieve this, a mean teacher framework is employed. At the start $( k = 0 )$ , both the student model $( M _ { s t } = h _ { s t } ( f _ { s t } ( \cdot ) ) )$ and the teacher model $( M _ { t e } \ =$ $h _ { t e } ( f _ { t e } ( \cdot ) ) )$ are initialized with the parameters pretrained in the source domain, such that $M _ { s t , k = 0 } = M _ { t e , k = 0 } = M _ { S }$ . For each incoming batch $B _ { k }$ , the parameters of the student model $\theta _ { s t }$ are optimized. Subsequently, the parameters of the teacher model $\theta _ { t e }$ are updated via an Exponential Moving Average (EMA) of the parameters of the student model:

$$
M _ { t e } \gets \alpha M _ { t e } + ( 1 - \alpha ) M _ { s t }\tag{1}
$$

This framework enables the model to dynamically adapt to non-stationary domains by utilizing a stream of unlabeled data to track distributional changes, while concurrently receiving stable supervisory signals from the teacher component.

## B. Dynamic Feature Prototype Replay

To overcome the inherent limitations of static feature prototypes, the knowledge anchor is shifted from the feature space to the image space. Unlike existing methods that rely on frozen feature vectors, the DIP framework optimizes a set of synthetic images ${ \mathcal { P } } _ { V }$ to encapsulate the essential geometry of the source distribution. This design choice is grounded in the principle of representation invariance: while the feature extractor $f _ { s t }$ undergoes significant drift during target adaptation, the semantic content within the image space remains a constant reference point. By regenerating features through the latest model state (Eq. 6), it is ensured that the source knowledge anchor is consistently aligned with the current model, thus theoretically eliminating the misalignment error that plagues conventional replay mechanisms.

1) Distilling Image Prototype Initialization and Optimization: Before optimization, a small set of synthetic images $\mathcal { P } _ { \mathcal { V } } = \{ ( s _ { 1 } , y _ { 1 } ) , \dots , ( s _ { n } , y _ { n } ) \}$ with $n \ll | \mathcal { D } _ { S } |$ is randomly initialized and subsequently optimized for each class. This procedure is a bi-level optimization process.

In the Simulated Training (Inner Loop), a virtual model $M _ { \nu }$ (a copy of $M _ { S } )$ is updated using a mini-batch from $\mathcal { P } _ { \mathcal { V } }$ to minimize a simulated training loss, typically on the classifier head or a few top layers, thereby mimicking lightweight adaptation. The inner loop optimization can be formulated as:

$$
\begin{array} { r } { \theta ^ { ( t + 1 ) } = \theta ^ { ( t ) } - \alpha \nabla _ { \theta } \mathcal { L } _ { \mathrm { i n n e r } } ( B _ { \nu } ; \theta ^ { ( t ) } ) , } \end{array}\tag{2}
$$

where $\boldsymbol { \theta } ^ { ( t ) }$ represents the model parameters at inner loop step $t , \alpha$ is the learning rate of the inner loop, $B _ { \nu }$ is a mini-batch sampled from the synthetic dataset $\mathcal { P } _ { \mathcal { V } }$ , and $\mathcal { L } _ { \mathrm { i n n e r } }$ is typically the cross-entropy loss defined as

$$
\mathcal { L } _ { \mathrm { i n n e r } } = \frac { 1 } { | B _ { \nu } | } \sum _ { ( s _ { i } , y _ { i } ) \in B _ { \nu } } \mathbf { C E } ( M _ { \mathcal { V } } ( s _ { i } ) , y _ { i } ) .\tag{3}
$$

In the Meta-Optimization (Outer Loop), the synthetic images $s _ { i }$ in $\mathcal { P } _ { \mathcal { V } }$ are optimized based on the performance of the virtually trained $M _ { \nu }$ on a mini-batch from the source dataset $\mathcal { D } _ { \mathcal { S } }$ The meta-loss $\mathcal { L } _ { \mathrm { m e t a } }$ drives the synthetic images to encapsulate the essential knowledge of $\mathcal { D } _ { \mathcal { S } }$ . The outer loop optimization can be formulated as:

$$
s _ { i } \gets s _ { i } - \beta \nabla _ { s _ { i } } \mathcal { L } _ { \mathrm { m e t a } } ( B _ { \mathcal { S } } ; \boldsymbol { \theta } ^ { \ast } ( s _ { i } ) ) ,\tag{4}
$$

where $s _ { i }$ denotes the synthetic images being optimized, $\beta$ is the outer loop learning rate, $\theta ^ { * } ( s _ { i } )$ represents the model parameters after inner loop training on $\mathcal { P } _ { \mathcal { V } } , B _ { \mathcal { S } }$ is a mini-batch sampled from the source dataset $\mathcal { D } _ { \mathcal { S } }$ , and $\mathcal { L } _ { \mathrm { m e t a } }$ is generally the cross-entropy loss defined as

$$
\mathcal { L } _ { \mathrm { m e t a } } = \frac { 1 } { | B s | } \sum _ { ( x _ { j } , y _ { j } ) \in B _ { S } } \mathbf { C E } ( M _ { \mathcal { V } } ( x _ { j } ) , y _ { j } ) .\tag{5}
$$

This optimization can include matching feature distributions between $M _ { \mathcal { V } }$ on $\mathcal { P } _ { \mathcal { V } }$ and $M _ { S }$ on $\mathcal { D } _ { \mathcal { S } }$ , and ensuring classifier consistency. Optional privacy-friendly regularization (pixel value constraints and smoothness regularization) can be applied during optimization to avoid generating images that are highly similar to the original samples.

2) Dynamic Feature Prototype Generation: After optimization, the final synthetic images $\mathcal { P } _ { \nu } ^ { * }$ are obtained. During testtime adaptation, feature prototypes are dynamically generated by passing $\mathcal { P } _ { \nu } ^ { * }$ through the feature extractor $f _ { s t } ^ { ( k ) }$ of the current student model at each adaptation step k. For each class $c ,$ the feature prototype $\mathcal { P } _ { f } ^ { ( k ) } [ c ]$ is computed as:

$$
\mathcal { P } _ { f } ^ { ( k ) } [ c ] = M e a n ( \{ f _ { s t } ^ { ( k ) } ( s ) ~ | ~ ( s , y ) \in \mathcal { P } _ { \mathcal { V } } ^ { * } , y = c \} )\tag{6}
$$

These dynamically generated feature prototypes ${ \mathcal P } _ { f } ^ { ( k ) }$ are utilized in the proposed contrastive learning strategy. Given the features $f _ { s t } ( x _ { i } )$ extracted by the student model for a target sample $x _ { i } ,$ the similarity between $f _ { s t } ( x _ { i } )$ and all feature prototypes is computed. The contrastive loss for a single sample $x _ { i }$ is defined as:

$$
\mathcal { L } _ { \mathrm { c o n t r a s t i v e } } ( x _ { i } ) = - \log \frac { \exp ( s i m ( f _ { s t } ( x _ { i } ) , \mathcal { P } _ { f } ^ { ( k ) } [ \hat { y } _ { i } ] ) / \tau ) } { \sum _ { c \in C } \exp ( s i m ( f _ { s t } ( x _ { i } ) , \mathcal { P } _ { f } ^ { ( k ) } [ c ] ) / \tau ) } ,\tag{7}
$$

where $\hat { y } _ { i }$ is the pseudo-label predicted by the teacher model for $x _ { i } ,$ , sim $( \cdot , \cdot )$ denotes cosine similarity, and $\tau$ is a temperature parameter.

This contrastive learning approach alleviates feature uncertainty in the target domain by utilizing dynamically updated feature prototypes ${ \mathcal P } _ { f } ^ { ( k ) }$ . The prototypes are driven by the DIP $\mathcal { P } _ { \nu } ^ { \ast }$ , which serves as a compact substitute for source data. The contrastive loss guides the model by drawing the features of target samples closer to their assigned class prototypes while pushing them away from other prototypes, effectively anchoring the target feature space to robust source knowledge.

## C. Source-Calibrated Uncertainty Estimation

To mitigate error accumulation from noisy pseudo-labels, a source-calibrated uncertainty estimation method is proposed. Unlike traditional approaches that rely solely on target data, the proposed method uses the $\mathrm { D I P }$ as a stable anchor from the source domain to provide unbiased uncertainty quantification.

The source-calibrated uncertainty stems from the modeling of uncertainty sources. Traditional methods estimate uncertainty based solely on $p ( \boldsymbol { y } | \boldsymbol { x } _ { t } , \boldsymbol { \theta } )$ , which conflates aleatoric uncertainty (from target data noise) and epistemic uncertainty (from model ignorance due to domain shift). During Test-Time Adaptation (TTA), parameters θ evolve rapidly, making epistemic uncertainty difficult to isolate.

The proposed method introduces a source anchor $z _ { s }$ (from DIP) to disentangle this conflation by evaluating the conditional distribution $p ( \boldsymbol { y } | \boldsymbol { x } _ { t } , \boldsymbol { z } _ { s } , \boldsymbol { \theta } )$ . Within a Bayesian framework:

$$
p ( y | x _ { t } , z _ { s } , \theta ) \propto p ( y | x _ { t } , \theta ) \cdot p ( z _ { s } | y , x _ { t } , \theta )\tag{8}
$$

Here, $p ( z _ { s } | y , x _ { t } , \theta )$ serves as a calibration factor, measuring the plausibility of the source anchor $z _ { s }$ given the target sample and its prediction. If a target sample has high prediction confidence $p ( \boldsymbol { y } | \boldsymbol { x } _ { t } , \boldsymbol { \theta } )$ but its features diverge from the source anchor (low $p ( z _ { s } | y , x _ { t } , \theta ) )$ , the calibrated probability $p ( \boldsymbol { y } | \boldsymbol { x } _ { t } , \boldsymbol { z } _ { s } , \boldsymbol { \theta } )$ decreases, indicating potential miscalibration. Conversely, good alignment with the source anchor increases the calibrated confidence [49].

Thus, the source-calibrated uncertainty, $\begin{array} { r l r l } { \mathcal { U } ( x _ { t } ) } & { { } = } & { } & { { } } \end{array}$ $\mathbb { H } [ p ( \boldsymbol { y } | \boldsymbol { x } _ { t } , \boldsymbol { z } _ { s } , \boldsymbol { \theta } ) ]$ , corrects the biased distribution from target data alone, providing a more reliable measure of predictive uncertainty that is robust to domain shift.

While $\operatorname { E q } .$ . 8 defines the ideal source-calibrated distribution, directly computing the integral over the entire parameter space is computationally intractable for online TTA. To bridge this gap, the response of the model to the DIP-based prototypes is treated as a proxy for epistemic uncertainty. Specifically, the focus is placed on the uncertainty of the classification head $\theta _ { h } .$ , as the feature extractor $f _ { s t }$ is assumed to be a deterministic mapping. By applying a Laplace Approximation $\mathrm { ( L A ) }$ centered at the MAP estimate $\theta _ { \mathrm { M A P } }$ , the abstract calibration problem is transformed into a practical weight estimation task (Eq. 11), thereby anchoring the target prediction reliability in the stability of the pre-acquired source knowledge.

A direct implementation of the aforementioned theory is difficult. Therefore, the paper adopts an approximation method based on Bayesian inference, which focuses on modeling the uncertainty of the model parameters and thereby indirectly achieves source domain calibration. The posterior distribution over the network parameters $\theta ,$ conditioned on the source knowledge represented by the DIP-based prototypes $\mathcal { P } _ { f }$ , is given by $p ( \theta | \mathcal { P } _ { f } ) \propto p ( \theta ) p ( \mathcal { P } _ { f } | \theta )$ . As computing this posterior is intractable, a Laplace Approximation (LA) is employed, which forms a Gaussian distribution around a mode of the posterior, $\theta _ { \mathrm { { M A P } : } }$

$$
p ( \theta | \mathcal { P } _ { f } ) \approx \mathcal { N } ( \theta | \theta _ { \mathrm { M A P } } , H ^ { - 1 } )\tag{9}
$$

Here, H is the negative Hessian of the log-posterior evaluated at $\theta _ { \mathrm { M A P } }$ . To maintain computational feasibility, this Bayesian treatment is applied only to the final classification layer of the student model, $h _ { s t }$ , using the last-layer Laplace approximation. The feature extractor $f _ { s t }$ remains deterministic. For efficiency, the Hessian is approximated using Kronecker-factored Laplace Approximation (KFLA) [50] where $H \approx V \otimes U$ . This enables the efficient estimation of the predictive posterior distribution for a given feature representation z:

$$
p ( y | z , \mathcal { P } _ { f } ) \approx \int s o f t m a x ( h _ { s t } ( z ; \theta ) ) p ( \theta | \mathcal { P } _ { f } ) d \theta\tag{10}
$$

To derive a source-calibrated uncertainty weight $w _ { i }$ for each target sample $x _ { i } ,$ , the predictive posterior is approximated using Monte Carlo $\mathrm { ( M C ) }$ integration. Specifically, α parameter samples $\theta _ { j } \sim \mathcal { N } ( \theta | \theta _ { \mathrm { M A P } } , H ^ { - 1 } )$ are drawn to compute the mean predictive probability $\bar { P } ( x _ { i } )$ . The final weight is defined as the exponentiated negative entropy of this mean prediction:

$$
w _ { i } = \exp ( - E n t r o p y ( \bar { P } ( x _ { i } ) ) )\tag{11}
$$

This weighting mechanism down-weights target samples that exhibit high predictive uncertainty, thereby mitigating negative adaptation on out-of-distribution samples.

The uncertainty weights $w _ { i }$ are applied to two loss components. First, the Consistency Loss $( \mathcal { L } _ { \mathrm { C o n s i s t e n c y } } )$ quantifies the agreement between the outputs of the student and teacher models:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { C o n s i s t e n c y } } = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { i } \cdot S C E ( M _ { s t } ( x _ { i } ) , M _ { t e } ( x _ { i } ) ) } \\ & { \quad \quad \quad \quad + \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { i } \cdot S C E ( M _ { s t } ( \tilde { x } _ { i } ) , M _ { t e } ( x _ { i } ) ) , } \end{array}\tag{12}
$$

where ${ \tilde { x } } _ { i }$ is an augmented version of $x _ { i } ,$ and $S C E ( p , q ) \ =$ $\begin{array} { r } { \frac { 1 } { 2 } ( C E ( p , q ) + C E ( \hat { q } , p ) ) } \end{array}$ is the symmetric cross-entropy.

Second, the Contrastive Loss $( \mathcal { L } _ { \mathrm { C o n t r a s t i v e } } )$ incorporates uncertainty weighting:

$$
{ \mathcal { L } } _ { \mathrm { C o n t r a s t i v e } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } w _ { i } \cdot { \mathcal { L } } _ { \mathrm { c o n t r a s t i v e } } ( x _ { i } )\tag{13}
$$

## D. Overall Optimization Objective

The total loss for DIPTTA comprises three components: the Consistency Loss $( \mathcal { L } _ { C o n s i s t e n c y } )$ , the Contrastive Loss $( \mathcal { L } _ { C o n t r a s t i v e } )$ , and the Replay Loss $( \mathcal { L } _ { R e p l a y } )$ . The Replay Loss is computed on the feature prototypes driven by the latent prototypes of the source domain to prevent catastrophic forgetting via the replay mechanism:

$$
\mathcal { L } _ { R e p l a y } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } C E ( h _ { s t } ( \mathcal { P } _ { f } [ y ] ) , y )\tag{14}
$$

The final optimization objective is to minimize the total loss $( \mathcal { L } _ { t o t a l } )$ , a weighted sum of these components:

$$
\mathcal { L } _ { t o t a l } = \lambda _ { 1 } \mathcal { L } _ { R e p l a y } + \lambda _ { 2 } \mathcal { L } _ { C o n t r a s t i v e } + \lambda _ { 3 } \mathcal { L } _ { C o n s i s t e n c y }\tag{15}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } .$ , and $\lambda _ { 3 }$ are hyperparameters controlling the contribution of each loss term. The losses are weighted using $\lambda _ { 1 } = 0 . 5 , \lambda _ { 2 } = 0 . 2 5$ , and $\lambda _ { 3 } = 0 . 1 5$ across all experiments (refer to Section IV-F3 and Figure 3 for details on the hyperparameter sensitivity analysis).

## IV. EXPERIMENTS

## A. Datasets and Experiment Details

Extensive experiments are conducted to demonstrate the effectiveness of the proposed approach. DIPTTA is evaluated on five benchmark tasks for continual test-time adaptation in image processing: CIFAR-10-C, CIFAR-100-C, ImageNet-C [51], ImageNet-R [52], and the CCC benchmark. These tasks are designed to assess the robustness of machine learning models to corruptions and disturbances in the input data.

In all experiments, the TTA setup is strictly adhered to, wherein no source data is accessed. All models are evaluated online, based on a maximum corruption severity level of five. Similar to CoTTA, standard pre-trained WideRes-Net [56], ResNeXt-29 [57], and ResNet-50 [58] are employed as the source models on a single RTX3090 (24GB) GPU for CIFAR10-C, CIFAR100-C, ImageNet-C, TinyImageNet-C, ImageNet-R, and the CCC benchmark. All results are evaluated with a corruption severity level of 5 in an online manner. For test time adaptation, the learning rate is set to 0.00025/0.001 for ResNet50/ViT-Base experiments. SGD [59] is utilized as the optimizer, with a momentum of 0.9 and a batch size of 64.

## B. Comparison with SOTA methods for TTA

Table I presents the comparative results of DIPTTA with state-of-the-art methods, including entropy minimization and test-time batch normalization under the setting of TTA.

TABLE I  
CLASSIFICATION ERROR RATE (%) ON CIFAR10-C, CIFAR100-C, AND TINYIMAGENET-C UNDER TTA. BOLD TEXT INDICATES THE BEST.
<table><tr><td>Method</td><td>Gauss.</td><td>Shot</td><td>Imp.</td><td>Def.</td><td>Glass</td><td>Mot.</td><td>Zoom</td><td>Snow</td><td>Fro.</td><td>Fog Bri.</td><td>Con.</td><td>Ela.</td><td>Pix.</td><td>JPEG</td><td>Mean</td></tr><tr><td colspan="10">CIFAR10 to CIFAR10-C</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Source Only</td><td>72.3</td><td>65.7</td><td>72.9</td><td>46.9</td><td>54.3</td><td>34.8</td><td>42.0</td><td>25.1</td><td>41.3</td><td>26.0</td><td>9.3</td><td>46.7</td><td>26.6</td><td>58.5</td><td>30.3</td><td>43.5</td></tr><tr><td>Tent [8] [ICLR 2021]</td><td>24.7</td><td>22.3</td><td>32.4</td><td>11.6</td><td>32.2</td><td>13.0</td><td>11.0</td><td>16.0</td><td>16.1</td><td>13.1</td><td>7.7</td><td>11.2</td><td>22.0</td><td>17.2</td><td>23.6</td><td>18.3</td></tr><tr><td>EATA [39] ] [ICML 2022]</td><td>24.3</td><td>22.3</td><td>32.2</td><td>11.3</td><td>31.7</td><td>12.9</td><td>10.8</td><td>16.0</td><td>16.2</td><td>13.4</td><td>7.9</td><td>11.3</td><td>21.5</td><td>17.0</td><td>23.4</td><td>18.1</td></tr><tr><td>MEMO [53]</td><td>57.2</td><td>51.2</td><td>56.3</td><td>32.3</td><td>48.7</td><td>27.6</td><td>29.6</td><td>20.7</td><td>29.3</td><td>20.7</td><td>8.6</td><td>30.5</td><td>25.1</td><td>52.4</td><td>27.4</td><td>34.4</td></tr><tr><td>SAR [9] [ICLR 2023]</td><td>28.3</td><td>26.0</td><td>35.7</td><td>12.7</td><td>34.8</td><td>13.9</td><td>12.0</td><td>17.5</td><td>17.6</td><td>14.9</td><td>8.2</td><td>13.0</td><td>23.5</td><td>19.5</td><td>27.2</td><td>20.3</td></tr><tr><td>COME [54]</td><td>24.9</td><td>22.1</td><td>29.0</td><td>16.9</td><td>33.1</td><td>17.9</td><td>15.0</td><td>15.6</td><td>12.4</td><td>12.8</td><td>7.9</td><td>9.4</td><td>21.7</td><td>16.0</td><td>19.8</td><td>18.3</td></tr><tr><td>AEA [55]</td><td>25.8</td><td>22.6</td><td>31.6</td><td>11.8</td><td>30.2</td><td>12.5</td><td>10.2</td><td>14.5</td><td>14.1</td><td>12.4</td><td>7.6</td><td>10.8</td><td>19.1</td><td>14.6</td><td>20.1</td><td>17.2</td></tr><tr><td>DIPTTA (Ours)</td><td>23.9</td><td>20.9</td><td>28.0</td><td>11.1</td><td>27.3</td><td>12.2</td><td>10.6</td><td>14.7</td><td>14.3</td><td>11.8</td><td>8.4</td><td>10.6</td><td>18.9</td><td>14.0</td><td>19.5</td><td>16.4</td></tr><tr><td colspan="10">CIFAR100 to CIFAR100-C</td><td colspan="7"></td></tr><tr><td>Source Only</td><td>73.0</td><td>68.0</td><td>39.4</td><td>29.3</td><td>54.1</td><td>30.8</td><td>28.8</td><td>39.5</td><td>45.8</td><td>50.3</td><td>29.5</td><td>55.1</td><td>37.2</td><td>74.7</td><td>41.2</td><td>46.4</td></tr><tr><td>Tent [8] [ICLR 2021]</td><td>37.3</td><td>34.8</td><td>34.5</td><td>25.0</td><td>37.4</td><td>27.5</td><td>25.1</td><td>30.4</td><td>32.0</td><td>33.8</td><td>24.1</td><td>28.1</td><td>32.9</td><td>28.4</td><td>36.9</td><td>31.2</td></tr><tr><td>EATA [39] ] [ICML 2022]</td><td>37.2</td><td>35.1</td><td>34.5</td><td>25.0</td><td>37.0</td><td>27.5</td><td>25.2</td><td>30.4</td><td>31.1</td><td>34.6</td><td>23.9</td><td>27.7</td><td>32.7</td><td>28.4</td><td>36.5</td><td>31.1</td></tr><tr><td>MEMO [53]</td><td>57.5</td><td>53.5</td><td>35.9</td><td>28.7</td><td>45.5</td><td>29.5</td><td>29.3</td><td>33.9</td><td>35.7</td><td>46.5</td><td>26.6</td><td>38.3</td><td>36.9</td><td>52.2</td><td>38.8</td><td>39.2</td></tr><tr><td>SAR [9] ] [ICLR 2023]</td><td>40.5</td><td>37.9</td><td>38.7</td><td>26.6</td><td>39.9</td><td>28.9</td><td>27.0</td><td>33.4</td><td>33.2</td><td>38.7</td><td>25.5</td><td>29.4</td><td>34.2</td><td>31.4</td><td>39.4</td><td>33.6</td></tr><tr><td>COME [54] [ICLR 2025]</td><td>37.3</td><td>34.8</td><td>34.4</td><td>25.0</td><td>37.4</td><td>27.6</td><td>25.0</td><td>30.4</td><td>31.9</td><td>33.7</td><td>24.0</td><td>28.1</td><td>32.9</td><td>28.3</td><td>36.9</td><td>31.1</td></tr><tr><td>AEA [55] ] [ICLR 2025] DIPTTA (Ours)</td><td>36.9</td><td>34.4</td><td>33.8 37.5</td><td>24.8 25.0</td><td>37.4</td><td>27.2</td><td>24.6</td><td>30.0</td><td>31.1</td><td>33.7</td><td>23.5</td><td>28.1</td><td>32.7</td><td>28.3</td><td>36.4</td><td>30.9</td></tr><tr><td></td><td>38.8</td><td>32.5</td><td></td><td></td><td>40.9</td><td>24.4</td><td>23.7</td><td>30.3</td><td>33.5</td><td>32.8</td><td>19.1</td><td>21.6</td><td>24.7</td><td>31.9</td><td>29.2</td><td>30.4</td></tr><tr><td colspan="10">ImageNet to TinyImageNet-C</td><td colspan="7"></td></tr><tr><td>Source Only</td><td>96.6</td><td>95.1</td><td>97.2</td><td>92.5</td><td>92.2</td><td>77.8</td><td>78.5</td><td>81.9</td><td>78.1</td><td>89.5</td><td>77.8</td><td>98.3</td><td>69.5</td><td>71.9</td><td>55.6</td><td>83.5</td></tr><tr><td>Tent [8] [ICLR 2021]</td><td>66.5</td><td>64.3</td><td>72.9</td><td>63.2</td><td>75.1</td><td>55.5</td><td>54.8</td><td>63.6</td><td>61.9</td><td>67.6</td><td>57.6</td><td>86.2</td><td>56.6</td><td>51.5</td><td>52.5</td><td>63.3</td></tr><tr><td>EATA [39] [ICML 2022]</td><td>65.1</td><td>63.6</td><td>69.7</td><td>62.3</td><td>74.2</td><td>55.3</td><td>54.1</td><td>61.3</td><td>61.0</td><td>63.9</td><td>55.4</td><td>90.8</td><td>56.1</td><td>51.1</td><td>52.2</td><td>62.4</td></tr><tr><td>SAR [9] [ICLR 2023]</td><td>67.2</td><td>65.2</td><td>73.6</td><td>63.9</td><td>75.9</td><td>55.8</td><td>55.2</td><td>64.1</td><td>62.6</td><td>68.4</td><td>58.0</td><td>86.0</td><td>57.1</td><td>51.8</td><td>52.9</td><td>63.8</td></tr><tr><td>COME [54] [ICLR 2025]</td><td>66.1</td><td>64.6</td><td>72.3</td><td>63.2</td><td>75.1</td><td>55.8</td><td>54.9</td><td>63.1</td><td>61.2</td><td>65.9</td><td>56.9</td><td>85.3</td><td>56.9</td><td>51.8</td><td>52.9</td><td>63.1</td></tr><tr><td>AEA [55] [ICLR 2025]</td><td>63.7</td><td>62.1</td><td>67.9</td><td>60.9</td><td>72.8</td><td>53.8</td><td>52.9</td><td>60.1</td><td>58.9</td><td>61.9</td><td>54.2</td><td>90.3</td><td>54.5</td><td>49.8</td><td>51.1</td><td>61.0</td></tr><tr><td>DIPTTA (Ours)</td><td>63.4</td><td>62.8</td><td>66.8</td><td>65.4</td><td>68.8</td><td>53.3</td><td>51.5</td><td>58.3</td><td>58.2</td><td>58.6</td><td>53.9</td><td>90.1</td><td>56.2</td><td>47.9</td><td>50.1</td><td>60.1</td></tr></table>

1) Comparison on CIFAR10 to CIFAR10-C: The experimental results on the CIFAR10-C benchmark demonstrate the superiority of the proposed DIPTTA method. As shown in the table, DIPTTA achieves the lowest mean classification error of 16.4%, outperforming the runner-up method AEA (17.2%) by 0.8% and significantly reducing the error compared to the Source Only baseline (43.5%). Specifically, DIPTTA attains the best performance in 10 out of 15 corruption types, showing particular robustness in categories such as Gaussian noise, Shot noise, and Impulse noise, which clearly demonstrates the effectiveness of DIPTTA in addressing continual domain shifts and its advantage over the state-of-the-art methods.

2) Comparison on CIFAR100 to CIFAR100-C: On the more challenging CIFAR100-C dataset, DIPTTA continues to demonstrate robust adaptation capabilities. It achieves a stateof-the-art mean error rate of 30.4%, surpassing competitive baselines such as AEA (30.9%) and EATA (31.1%). The proposed method shows significant improvements in severe corruption scenarios, particularly excelling in Contrast (21.6%) and Elastic transform (24.7%), where it outperforms the second-best results by substantial margins.

3) Comparison on ImageNet to TinyImageNet-C: For the large-scale TinyImageNet-C benchmark, DIPTTA maintains its leadership with a mean error of 60.1%. This represents a substantial improvement over the Source Only baseline of 83.5% and outperforms the previous method, AEA (61.0%). The results indicate that DIPTTA scales effectively to complex datasets, achieving the lowest error rates across a diverse range of corruptions, including Glass blur, Snow, and Pixelate.

## C. Comparison with SOTA methods for CTTA

The proposed DIPTTA is compared with various CTTA benchmark methods, including entropy regularization, unsupervised contrastive learning, self-training, and pseudo-label filtering. The results are shown in Table II for CIFAR10- C, CIFAR100-C, ImageNet-C, and in Table III for the CCC benchmark, respectively.

1) Comparison on CIFAR10 to CIFAR10-C: On the CIFAR10-C benchmark, directly testing the Source-only model on the target domains yields a high average error of 43.5%. While subsequent methods like Tent, which aids in sequential adaptation, and CoTTA, which enhances pseudo-label quality, improve upon this baseline, the proposed DIPTTA method establishes a new state-of-the-art. It achieves the lowest mean error rate of 13.9%, significantly outperforming all competing methods, including RMT (16.7%), BeCoTTA (16.3%), and TCA (14.7%). This demonstrates its superior ability to adapt to continual domain shifts in this setting.

2) Comparison on CIFAR100 to CIFAR100-C: For the more challenging CIFAR100-C dataset, the Source-only model starts with a 46.4% error rate. It clearly exposes the critical issue of error accumulation in continual test-time adaptation. For example, the Tent-based method suffers from this problem, resulting in a substantially higher error rate of 60.9% in the long run. Methods like RMT, which employs a symmetric cross-entropy optimization, DSS, which notes the existence of high- and low-quality samples, and BeCoTTA, which improves computational efficiency, all offer incremental progress. On the other hand, the DIPTTA method once again demonstrates dominant performance, reducing the average error to just 27.8%. This result is a significant improvement over all prior art and underscores its effectiveness in preventing model degradation over long, sequential adaptations.

3) Comparison on ImageNet to ImageNet-C: On the largescale ImageNet-C dataset, the source-only model struggles immensely, showing an initial error rate of 82.0%. This benchmark tests the scalability and robustness of adaptation methods under severe corruptions. Although advanced approaches such as CoTTA (62.7%) provide substantial gains, they still fall short of the top performance. The DIPTTA method again proves its robustness, achieving a leading mean error rate of 58.4%. It outperforms the latest methods like TCA (59.3%) and BeCoTTA (60.9%), confirming its effectiveness on complex, large-scale data.

TABLE II  
CLASSIFICATION ERROR RATE (%) ON CIFAR10-C, CIFAR100-C, AND IMAGENET-C UNDER CTTA. BOLD TEXT INDICATES THE BEST.
<table><tr><td>Method</td><td>Gauss.</td><td>Shot</td><td>Imp.</td><td>Def.</td><td>Glass</td><td>Mot.</td><td>Zoom</td><td>Snow</td><td>Fro.</td><td>Fog</td><td>Bri.</td><td>Con.</td><td>Ela.</td><td>Pix.</td><td>JPEG Mean</td></tr><tr><td colspan="10">CIFAR10 to CIFAR10-C</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Source Only</td><td>72.3</td><td>65.7</td><td>72.9</td><td>46.9</td><td>54.3</td><td>34.8</td><td>42.0</td><td>25.1</td><td>41.3</td><td>26.0</td><td>9.3</td><td>46.7</td><td>26.6</td><td>58.5</td><td>30.3</td><td>43.5</td></tr><tr><td>Tent [8] [ICLR 2021]</td><td>24.8</td><td>20.6</td><td>28.6</td><td>14.4</td><td>31.1</td><td>16.5</td><td>14.1</td><td>19.1</td><td>18.6</td><td>18.6</td><td>12.2</td><td>20.3</td><td>25.7</td><td>20.8</td><td>24.9</td><td>20.7</td></tr><tr><td>CoTTA [10] [CVPR 2022]</td><td>24.3</td><td>21.3</td><td>26.6</td><td>11.6</td><td>27.6</td><td>12.2</td><td>10.3</td><td>14.8</td><td>14.1</td><td>12.4</td><td>7.6</td><td>10.6</td><td>18.3</td><td>13.4</td><td>17.3</td><td>16.2</td></tr><tr><td>RMT [5] [CVPR 2023]</td><td>24.0</td><td>20.4</td><td>25.6</td><td>12.6</td><td>25.4</td><td>14.2</td><td>12.2</td><td>15.4</td><td>15.1</td><td>14.1</td><td>10.3</td><td>13.7</td><td>17.1</td><td>13.5</td><td>16.0</td><td>16.7</td></tr><tr><td>DSS [60] 1 [WACV 2024]</td><td>24.1</td><td>21.3</td><td>25.4</td><td>11.7</td><td>26.9</td><td>12.2</td><td>10.5</td><td>14.5</td><td>14.1</td><td>12.5</td><td>7.8</td><td>10.8</td><td>18.0</td><td>13.1</td><td>17.3</td><td>16.0</td></tr><tr><td>BeCoTTA [61] 1 [ICML 2024]</td><td>22.9</td><td>19.1</td><td>26.9</td><td>10.2</td><td>27.5</td><td>12.7</td><td>10.4</td><td>14.7</td><td>14.3</td><td>12.4</td><td>7.2</td><td>9.4</td><td>20.9</td><td>15.2</td><td>20.2</td><td>16.3</td></tr><tr><td>BeCoTTA [61] [CVPR 2025] DIPTTA (Ours)</td><td>22.4</td><td>19.2</td><td>23.0</td><td>10.8</td><td>23.2</td><td>11.6</td><td>9.9</td><td>13.1</td><td>13.1</td><td>11.8</td><td>7.6</td><td>10.7</td><td>16.4</td><td>12.0</td><td>15.5</td><td>14.7</td></tr><tr><td></td><td>21.9</td><td>18.2</td><td>24.7</td><td>10.4</td><td>23.8</td><td>11.2</td><td>9.5</td><td>12.1</td><td>11.8</td><td>10.3</td><td>7.1</td><td>8.6</td><td>14.7</td><td>10.4</td><td>14.2</td><td>13.9</td></tr><tr><td colspan="10">CIFAR100 to CIFAR100-C</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Source Only</td><td>73.0</td><td>68.0</td><td>39.4</td><td>29.3</td><td>54.1</td><td>30.8</td><td>28.8</td><td>39.5</td><td>45.8</td><td>50.3</td><td>29.5</td><td>55.1</td><td>37.2</td><td>74.7</td><td>41.2</td><td>46.4</td></tr><tr><td>Tent [8] [ICLR 2021]</td><td>37.2</td><td>35.8</td><td>41.7</td><td>37.9</td><td>51.2</td><td>48.3</td><td>48.5</td><td>58.4</td><td>63.7</td><td>71.1</td><td>70.4</td><td>82.3</td><td>88.0</td><td>88.5</td><td>90.4</td><td>60.9</td></tr><tr><td>CoTTA [10] [CVPR 2022]</td><td>40.1</td><td>37.7</td><td>39.7</td><td>26.9</td><td>38.0</td><td>27.9</td><td>26.4</td><td>32.8</td><td>31.8</td><td>40.3</td><td>24.7</td><td>26.9</td><td>32.5</td><td>28.3</td><td>33.5</td><td>32.5</td></tr><tr><td>RMT [5] [CVPR 2023] DSS [60]</td><td>40.5</td><td>36.1</td><td>36.3 37.2</td><td>27.7</td><td>33.9</td><td>28.5</td><td>26.4</td><td>29.0</td><td>29.0</td><td>32.5</td><td>25.1</td><td>27.4</td><td>28.2</td><td>26.3</td><td>29.3</td><td>30.4</td></tr><tr><td>1 [WACV 2024] BeCoTTA [61] [ICML 2024]</td><td>39.7 42.1</td><td>36.0 38.0</td><td>42.2</td><td>26.3 30.2</td><td>35.6 42.9</td><td>27.5</td><td>25.2</td><td>31.4</td><td>30.0</td><td>37.8</td><td>24.2</td><td>26.0</td><td>30.0</td><td>26.3</td><td>31.3</td><td>30.9</td></tr><tr><td>BeCoTTA [61] ] [CVPR 2025]</td><td>38.5</td><td>36.0</td><td>36.6</td><td>25.8</td><td>34.6</td><td>31.7</td><td>29.8</td><td>35.1</td><td>33.9</td><td>38.5</td><td>27.9</td><td>32.0</td><td>36.7</td><td>31.6</td><td>39.9</td><td>35.5</td></tr><tr><td>DIPTTA (Ours)</td><td>37.4</td><td>34.0</td><td>34.6</td><td>25.1</td><td>32.4</td><td>27.2 25.7</td><td>25.1</td><td>30.5</td><td>27.0 25.9</td><td>30.1 29.1</td><td>24.1</td><td>25.7 23.5</td><td>27.3</td><td>26.6</td><td>30.3</td><td>29.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>ImageNet to ImageNet-C</td><td></td><td>23.5</td><td>26.3</td><td></td><td></td><td>22.4</td><td></td><td>25.7</td><td>23.2</td><td>27.6</td><td>27.8</td></tr><tr><td colspan="10"></td><td colspan="7"></td></tr><tr><td>Source Only Tent [8] [ICLR 2021]</td><td>97.8</td><td>97.1</td><td>98.2</td><td>81.7</td><td>89.8</td><td>85.2</td><td>78.0</td><td>83.5</td><td>77.1</td><td>75.9</td><td>41.3</td><td>94.5</td><td>82.5</td><td>79.3</td><td>68.6</td><td>82.0</td></tr><tr><td></td><td>81.6</td><td>74.6</td><td>72.7</td><td>77.6</td><td>73.8</td><td>65.5</td><td>55.3</td><td>61.6</td><td>63.0</td><td>51.7</td><td>38.2</td><td>72.1</td><td>50.8</td><td>47.4</td><td>53.3</td><td>62.6</td></tr><tr><td>CoTTA [10] [CVPR 2022]</td><td>84.7</td><td>82.1</td><td>80.6</td><td>81.3</td><td>79.0</td><td>68.6</td><td>57.5</td><td>60.3</td><td>60.5</td><td>48.3</td><td>36.6</td><td>66.1</td><td>47.3</td><td>41.2</td><td>46.0</td><td>62.7</td></tr><tr><td>RMT [5] [CVPR 2023]</td><td>80.2</td><td>76.4</td><td>74.5</td><td>77.1</td><td>74.4</td><td>66.2</td><td>57.6</td><td>57.0</td><td>59.1</td><td>48.0</td><td>39.1</td><td>60.6</td><td>47.3</td><td>42.5</td><td>43.4</td><td>60.2</td></tr><tr><td>DSS [60] [WACV 2024]</td><td>82.3</td><td>78.4</td><td>76.7</td><td>81.9</td><td>77.8</td><td>66.9</td><td>60.9</td><td>50.8</td><td>60.9</td><td>47.7</td><td>35.4</td><td>69.0</td><td>47.5</td><td>40.9</td><td>46.2</td><td>62.2</td></tr><tr><td>BeCoTTA [61] [ICML 2024]</td><td>84.1</td><td>74.3</td><td>72.2</td><td>77.4</td><td>71.9</td><td>63.4</td><td>55.1</td><td>57.2</td><td>61.2</td><td>50.7</td><td>36.4</td><td>66.1</td><td>49.2</td><td>45.6</td><td>48.4</td><td>60.9</td></tr><tr><td>BeCoTTA [61] [CVPR 2025]</td><td>78.3</td><td>71.8</td><td>73.5</td><td>74.4</td><td>73.5</td><td>63.3 62.2</td><td>56.5 56.0</td><td>56.9 56.1</td><td>59.4 58.3</td><td>48.1 47.4</td><td>39.6 38.8</td><td>59.6 58.4</td><td>47.2 46.5</td><td>42.9 42.0</td><td>44.7 43.9</td><td>59.3 58.4</td></tr><tr><td>DIPTTA (Ours)</td><td>76.9</td><td>70.3</td><td>72.5</td><td>73.8</td><td>72.7</td></table>

TABLE III
<table><tr><td colspan="4">CLASSIFICATION ACCURACY (%) ON CCC-BENCHMARK, WHICH IS A LONG-SEQUENCE TASK. BOLD TEXT INDICATES THE BEST.</td></tr><tr><td>Method</td><td>CCC-Easy</td><td>CCC-Medium</td><td>CCC-Hard</td><td>Average</td></tr><tr><td>CoTTA [10] [CVPR 2022]</td><td> $1 4 . 9 \pm 0 . 8 8$ </td><td> $7 . 7 \pm 0 . 4 3$ </td><td> $1 . 1 \pm 0 . 1 6$ </td><td>7.9</td></tr><tr><td>ETA [39] [ICML 2022]</td><td> $4 1 . 4 \pm 0 . 9 5$ </td><td> $1 . 1 \pm 0 . 4 3$ </td><td> $0 . 2 \pm 0 . 0 5$ </td><td>14.2</td></tr><tr><td>EATA [39] [ICML 2022]</td><td> $4 8 . 2 \pm 0 . 6 0$ </td><td> $3 5 . 4 \pm 1 . 0 2$ </td><td> $8 . 7 \pm 0 . 8 0$ </td><td>30.8</td></tr><tr><td>RDumb [62] [NeurIPS 2023]</td><td> $4 9 . 3 \pm 0 . 8 8$ </td><td> $3 8 . 9 \pm 1 . 4 0 $ </td><td> $9 . 6 \pm 1 . 6 0$ </td><td>32.6</td></tr><tr><td>BeCoTTA [61][CVPR 2025]</td><td> $4 9 . 1 \pm 0 . 3 5$ </td><td> $3 9 . 5 \pm 0 . 5 3$ </td><td> ${ \bf 1 0 . 1 \pm 0 . 2 2 }$ </td><td>32.9</td></tr><tr><td>DIPTTA (Ours)</td><td> ${ \bf 5 1 . 2 \pm 0 . 7 0 }$ </td><td> ${ \bf 4 0 . 7 \pm 0 . 7 2 }$ </td><td> $9 . 4 \pm 0 . 5 7$ </td><td>33.8</td></tr></table>

4) Comparison on ImageNet to CCC: To further assess the robustness of the DIPTTA model in long-sequence continual adaptation tasks, an evaluation using the ImageNet to CCC benchmark was conducted. As detailed in Table III, the DIPTTA method shows highly competitive results. Specifically, DIPTTA achieves leading performance on both CCC-Easy and CCC-Medium difficulty levels, with accuracies of 51.2% and 40.7%, respectively. Although TCA performs the best on the CCC-Hard setting, DIPTTA obtains the best overall performance when considering the three difficulties, registering an average accuracy of 33.8%.

## D. Comparison with SOTA under Natural Domain Shifts

As shown in Table IV, experimental evaluation on the ImageNet-R dataset substantiates the superior and consistent performance of the proposed DIPTTA method. On the ResNet-50 architecture, DIPTTA achieves a state-of-the-art error rate of 52.7%, and its advantage is even more pronounced on the Vision Transformer (ViT) architecture, where it reduces the error rate to 35.3%. This consistent top-tier performance validates the potent effectiveness and generalizability of DIPTTA. Its success is attributed to the distilling image prototype, which captures essential and abstract class knowledge rather than just superficial features. This allows the model to effectively distinguish between the core content of an image and its changing artistic style, maintaining strong recognition capabilities while adapting. In contrast, other methods designed primarily for specific algorithmic corruptions struggle to generalize to such complex and varied natural domain shifts.

## E. Ablation Studies

To evaluate the contribution of the two key innovations, a detailed ablation study was performed. Their effectiveness is analyzed by comparing the results with existing methods and by evaluating the standalone impact of each component.

1) Effectiveness of Dynamic Feature Prototype Generation: The results from the CTTA Setting highlight the critical need for dynamic prototypes to prevent catastrophic forgetting. For instance, the Tent method, which lacks a knowledge replay mechanism, sees its error rate on CIFAR100-C increase dramatically from 31.2% in the TTA setting to 60.9% under CTTA. More importantly, the DIPTTA method, which uses dynamic prototypes, significantly outperforms RMT, a method that relies on static prototypes, across all datasets. On CIFAR10-C, DIPTTA achieves a 13.9% error rate compared to an error rate of 16.7% for RMT. This confirms that dynamically regenerating prototypes that stay aligned with the evolving model is crucial for preventing the knowledge anchor from becoming obsolete.

TABLE IV  
CLASSIFICATION ERROR RATE (%) ON IMAGENET-R. RESULTS ARE EVALUATED IN THE SINGLE-DOMAIN ADAPTATION SCENARIO. BOLD TEXT INDICATES THE BEST. WE USE \* TO DENOTE EPISODIC ADAPTATION AND USE <sup>′</sup> TO DENOTE THE RE-IMPLEMENTED
<table><tr><td>Model</td><td>Source Only</td><td>Tent</td><td>EATA</td><td>MEMO*</td><td>CoTTA</td><td>RMT</td><td>TEA</td><td>SAR</td><td>COME</td><td>AEA</td><td>EATA-C&#x27;</td><td>TCA</td><td>DIPTTA(Ours)</td></tr><tr><td>ResNet-50(BN)</td><td>62.0</td><td>57.7</td><td>55.1</td><td>58.1</td><td>57.6</td><td>57.1</td><td>57.2</td><td>57.3</td><td>54.6</td><td>54.2</td><td>52.9</td><td>53.4</td><td>52.7</td></tr><tr><td>ViT(LN)</td><td>47.5</td><td>45.8</td><td>41.8</td><td>42.5</td><td>43.6</td><td>45.8</td><td>39.9</td><td>45.0</td><td>38.5</td><td>37.3</td><td>35.8</td><td>36.2</td><td>35.3</td></tr></table>

TABLE V

AN ABLATION STUDY OF THE DFPR MODULE AND THE SCUE MODULE ACROSS FOUR DATASETS, WHERE (C) INDICATES THE CTTA, (BN) INDICATES EMPLOYING THE RESNET50 AS BACKBONE, AND (LN) INDICATES EMPLOYING VIT AS BACKBONE.
<table><tr><td>Prototype</td><td>Uncertainty</td><td>Example Method</td><td>CIFAR 10-C</td><td>CIFAR 100-C</td><td>Tiny ImageNet-C</td><td>ImageNet -R(BN)</td><td>ImageNet -R(LN)</td><td>CIFAR 10-C(C)</td><td>CIFAR 100-C(C)</td><td>ImageNet -C(C)</td></tr><tr><td>None</td><td>Entropy-based</td><td>Tent</td><td>18.3</td><td>31.2</td><td>63.3</td><td>57.7</td><td>43.8</td><td>20.7</td><td>60.9</td><td>62.6</td></tr><tr><td>None</td><td>Entropy-based</td><td>COME</td><td>18.3</td><td>31.1</td><td>63.1</td><td>54.6</td><td>38.5</td><td>16.7</td><td>29.0</td><td>58.9</td></tr><tr><td>Static</td><td>Entropy-based</td><td>RMT(baseline)</td><td>18.2</td><td>31.0</td><td>62.8</td><td>57.1</td><td>45.8</td><td>16.7</td><td>30.4</td><td>60.2</td></tr><tr><td>Static</td><td>Source-Calibrated</td><td>Ours(w/o DFPR)</td><td>17.2</td><td>30.8</td><td>61.7</td><td>54.0</td><td>37.9</td><td>15.2</td><td>28.1</td><td>59.0</td></tr><tr><td>Dynamic(DFPR)</td><td>Entropy-based</td><td>Ours(w/o SCUE)</td><td>16.9</td><td>30.9</td><td>61.3</td><td>53.4</td><td>38.7</td><td>14.5</td><td>28.7</td><td>59.6</td></tr><tr><td>Dynamic(DFPR)</td><td>Source-Calibrated</td><td>Ours</td><td>16.4</td><td>30.4</td><td>60.1</td><td>52.7</td><td>35.3</td><td>13.9</td><td>27.8</td><td>58.4</td></tr></table>

TABLE VI

TIME COST(S) PER BATCH ON FIVE DATASETS. WE USE A BATCH SIZE OF 200 FOR ALL RESULTS. WE USE <sup>′</sup> TO DENOTE THE RE-IMPLEMENTED.
<table><tr><td>Dataset</td><td>Source Only</td><td>Tent</td><td>EATA</td><td>MEMO</td><td>CoTTA</td><td>RMT</td><td>SAR</td><td>COME</td><td>AEA</td><td>EATA-C&#x27;</td><td>TCA</td><td>DIPTTA(Ours)</td></tr><tr><td>CIFAR10-C</td><td>0.718</td><td>1.809</td><td>2.643</td><td>62.350</td><td>3.893</td><td>2.635</td><td>3.192</td><td>2.930</td><td>31.269</td><td>2.973</td><td>6.186</td><td>2.749</td></tr><tr><td>CIFAR100-C</td><td>0.827</td><td>1.992</td><td>2.862</td><td>73.403</td><td>4.127</td><td>2.965</td><td>3.428</td><td>3.357</td><td>36.707</td><td>3.505</td><td>6.539</td><td>2.979</td></tr><tr><td>ImageNet-C</td><td>3.157</td><td>7.093</td><td>9.561</td><td>129.139</td><td>17.285</td><td>11.773</td><td>15.021</td><td>16.153</td><td>116.250</td><td>12.379</td><td>26.154</td><td>12.235</td></tr><tr><td>ImageNet-R(BN)</td><td>3.553</td><td>7.651</td><td>9.664</td><td>126.38</td><td>18.015</td><td>12.036</td><td>15.326</td><td>16.277</td><td>123.135</td><td>13.022</td><td>27.003</td><td>13.173</td></tr><tr><td>ImageNet-R(LN)</td><td>10.052</td><td>22.391</td><td>28.095</td><td>376.748</td><td>73.571</td><td>37.268</td><td>47.350</td><td>50.189</td><td>392.267</td><td>43.174</td><td>98.150</td><td>39.258</td></tr></table>

2) Advantage of Source-Calibrated Uncertainty: The source-calibrated uncertainty mechanism demonstrates clear advantages over conventional entropy-based methods. DIPTTA consistently outperforms methods like Tent and EATA, which are susceptible to overconfident predictions on distributionshifted data. On the challenging ImageNet-R benchmark, which features natural domain shifts, DIPTTA achieves stateof-the-art results. Furthermore, it accomplishes this while maintaining high computational efficiency, offering a superior trade-off compared to other methods. This performance validates that calibrating uncertainty against stable source knowledge leads to more reliable and robust adaptation.

3) Component-wise Analysis: A direct component-wise ablation, detailed in Table V, confirms the individual and combined value of the DFPR and SCUE modules. The baseline model (equivalent to Tent), without either component, records a reference error rate of 18.3% on CIFAR10-C. Integrating only the SCUE module (DIPTTA w/o DFPR) lowers the error rate to 17.2%, confirming its standalone value in mitigating error accumulation. Similarly, incorporating only the DFPR module (DIPTTA w/o SCUE) provides an even greater boost by reducing the error rate to 16.9%. Most importantly, the combination of both DFPR and SCUE in the full DIPTTA model achieves the best result, dropping the CIFAR10-C error rate to 16.4%, which represents a total improvement of 1.9 percentage points over the baseline. This trend remains consistent across other datasets, such as TinyImageNet-C, where the error rate falls from a baseline of 63.3% to 60.1% when both modules are used. The superior performance of the combined system strongly suggests that the two components are complementary and produce a synergistic effect, justifying the inclusion of both in the final model architecture.

## F. Further Empirical Analysis

1) Time Cost Analysis: The time cost analysis in Table VI confirms the exceptional computational efficiency of DIPTTA. In sharp contrast to computationally intensive methods like MEMO and AEA, whose memory banks and meta-learning render them impractical for time-sensitive applications, DIPTTA maintains a lean profile comparable to its efficient baseline, RMT. The minimal overhead from the proposed DFPR and SCUE modules is the key takeaway; on ImageNet-C, DIPTTA (12.235s) is only marginally slower than RMT (11.773s). This result validates that DIPTTA successfully achieves a superior trade-off, advancing adaptation performance without the cost of inefficiency.

![](images/4b75b12aa20913c050f599a667cdf2f9fdf41714ada4e30f38711cf64aec8ddd.jpg)  
Fig. 2. Sensitivity analysis of batch size, where C indicates the CTTA.

2) Sensitivity Analysis of Batch Size: As illustrated in Figure 2, DIPTTA exhibits robustness to batch size, consistently outperforming competitors even at a batch size of 1. On CIFAR benchmarks, it achieves optimal error rates (13.9% and 27.8%) at batch size 200 while maintaining stability across the 50–300 range, contrasting with the volatility of methods like TCA. This stability extends to ImageNet-C, confirming the low sensitivity and practical reliability of DIPTTA.

3) Hyperparameter Sensitivity Analysis: Figure 3 illustrates the impact of three hyperparameters $( \lambda _ { 1 } \lambda _ { 2 } , \lambda _ { 3 } )$ on the performance of a TTA method across the CIFAR10-C and CIFAR100-C datasets. The color of each point represents the mean error rate (%), where brighter yellow indicates lower error. The optimal points for the datasets were found to be (0.5, 0.25, 0.15) and (0.45, 0.2, 0.2), respectively. Given the proximity of these optima, the single parameter set of (0.5, 0.25, 0.15), marked by the red star, is selected for all experiments.

![](images/bc60ff24741a69ddd915ec84d0bd2412c9762de1c96fbb695d23115f96d835cb.jpg)

![](images/4abee7abfe6af2746fa5a88e6bf280aacf8a31f83de91f3ba5454001f9da0417.jpg)  
Fig. 3. Loss function hyperparameter sensitivity analysis of the TTA method on three benchmark datasets.

![](images/9cbae314d9163ae372432b25a0971b83bbd419ac528a1bf254275ba99a994dfd.jpg)  
Fig. 4. Visualization of t-SNE for two methods and our proposed DIPTTA on CIFAR10-C(TTA Scenarios). We select a random batch of features from the Gaussian and Fog domains for comparative visualization.

4) t-SNE Analysis: t-SNE [63] is utilized to visualize the feature representations of the model. As shown in Fig. 4, the feature vectors (dots) from the DIPTTA model exhibit significantly tighter clustering around the source domain prototypes (stars) compared to other methods on CIFAR10-C. This improved performance stems from two key factors: 1) the employed data distillation technique refines source domain knowledge to create more accurate class prototypes that better represent true feature centroids; and 2) the combined uncertainty-weighted consistency and contrastive losses form a robust constraint that aligns target feature vectors with these optimized prototypes. Consequently, the model learns more compact and separable features, creating clearer decision boundaries.

## V. CONCLUSION

In this paper, the critical challenges of error accumulation from noisy pseudo-labels and catastrophic forgetting of source knowledge during test-time adaptation are analyzed. To this end, the DIPTTA framework is proposed, which introduces a Distilling Image Prototype (DIP) as a compact and regenerative source knowledge anchor. The core contribution is twofold. First, the DIP enables dynamic feature prototype replay, which continuously regenerates feature prototypes aligned with the evolving feature space of the model, effectively overcoming the misalignment issue of static prototypes. Second, it facilitates source-calibrated uncertainty estimation, which leverages the stable DIP to debias uncertainty quantification and reliably suppress error propagation. Extensive experiments on multiple benchmarks validate that DIPTTA significantly outperforms state-of-the-art methods. The results demonstrate that the proposed approach provides a robust and effective solution for model adaptation in non-stationary test environments.

## REFERENCES

[1] J. Liang, R. He, and T. Tan, “A comprehensive survey on test-time adaptation under distribution shifts,” International Journal of Computer Vision, vol. 133, no. 1, pp. 31–64, 2025.

[2] Z. Wang, Y. Luo, L. Zheng, Z. Chen, S. Wang, and Z. Huang, “In search of lost online test-time adaptation: A survey,” International Journal of Computer Vision, vol. 133, no. 3, pp. 1106–1139, 2025.

[3] L. Chen, Y. Zhang, Y. Song, Y. Shan, and L. Liu, “Improved test-time adaptation for domain generalization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 24 172–24 182.

[4] B. Hu, J. Liu, K. Zheng, and Z.-J. Zha, “Unleashing knowledge potential of source hypothesis for source-free domain adaptation,” IEEE Transactions on Multimedia, vol. 26, pp. 5422–5434, 2023.

[5] M. Dobler, R. A. Marsden, and B. Yang, “Robust mean teacher for¨ continual and gradual test-time adaptation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 7704–7714.

[6] G. Chakrabarty, M. Sreenivas, and S. Biswas, “Santa: Source anchoring network and target alignment for continual test time adaptation,” Transactions on Machine Learning Research, 2023.

[7] M. Jang, S.-Y. Chung, and H. W. Chung, “Test-time adaptation via selftraining with nearest neighbor information,” in The Eleventh International Conference on Learning Representations, 2022.

[8] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, “Tent: Fully test-time adaptation by entropy minimization,” in International Conference on Learning Representations, 2020.

[9] S. Niu, J. Wu, Y. Zhang, Z. Wen, Y. Chen, P. Zhao, and M. Tan, “Towards stable test-time adaptation in dynamic wild world,” in The Eleventh International Conference on Learning Representations, 2023.

[10] Q. Wang, O. Fink, L. Van Gool, and D. Dai, “Continual test-time domain adaptation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 7201–7211.

[11] Z. Xiao and C. G. Snoek, “Beyond model adaptation at test time: A survey,” arXiv preprint arXiv:2411.03687, 2024.

[12] P. Tian, X. Zhou, H. Yu, and S. Xie, “High specificity guided crossdomain few-shot segmentation,” IEEE Transactions on Multimedia, 2025.

[13] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,” in Proceedings of the International Conference on Machine Learning (ICML), 2016, pp. 1050–1059.

[14] D. J. C. Mackay, Bayesian methods for adaptive models. California Institute of Technology, 1992.

[15] A. Kristiadi, M. Hein, and P. Hennig, “Being bayesian, even just a bit, fixes overconfidence in relu networks,” in International conference on machine learning. PMLR, 2020, pp. 5436–5446.

[16] E. Daxberger, A. Kristiadi, A. Immer, R. Eschenhagen, M. Bauer, and P. Hennig, “Laplace redux-effortless bayesian deep learning,” Advances in neural information processing systems, vol. 34, pp. 20 089–20 103, 2021.

[17] B. Zhao and H. Bilen, “Dataset condensation with differentiable siamese augmentation,” in International Conference on Machine Learning. PMLR, 2021, pp. 12 674–12 685.

[18] L. Yuan, B. Xie, and S. Li, “Robust test-time adaptation in dynamic scenarios,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 15 922–15 932.

[19] J. Ma, “Improved self-training for test-time adaptation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 23 701–23 710.

[20] S. Goyal, M. Sun, A. Raghunathan, and J. Z. Kolter, “Test time adaptation via conjugate pseudo-labels,” Advances in Neural Information Processing Systems, vol. 35, pp. 6204–6218, 2022.

[21] S. Sinha, P. Gehler, F. Locatello, and B. Schiele, “Test: Test-time selftraining under distribution shift,” in Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 2759–2769.

[22] Y. Iwasawa and Y. Matsuo, “Test-time classifier adjustment module for model-agnostic domain generalization,” Advances in Neural Information Processing Systems, vol. 34, pp. 2427–2440, 2021.

[23] M. Boudiaf, R. Mueller, I. Ben Ayed, and L. Bertinetto, “Parameterfree online test-time adaptation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 8344–8353.

[24] S. Wang, D. Zhang, Z. Yan, J. Zhang, and R. Li, “Feature alignment and uniformity for test time adaptation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 20 050–20 060.

[25] S. Jung, J. Lee, N. Kim, A. Shaban, B. Boots, and J. Choo, “Cafa: Class-aware feature alignment for test-time adaptation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19 060–19 071.

[26] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” Advances in neural information processing systems, vol. 30, 2017.

[27] S. Zhang, L. Zhang, and Z. Liu, “Test-time adaptation for object detection via dynamic dual teaching,” Image and Vision Computing, p. 105740, 2025.

[28] Y. Ye, W. Wei, L. Zhang, C. Ding, and Y. Zhang, “Domain consistency learning for continual test-time adaptation in image semantic segmentation,” Pattern Recognition, vol. 165, p. 111585, 2025.

[29] D. Chen, D. Wang, T. Darrell, and S. Ebrahimi, “Contrastive test-time adaptation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 295–305.

[30] Z. Ma, Y. Li, Y. Luo, X. Luo, J. Li, C. Chen, X.-S. Hua, and G. Lu, “Discrepancy and structure-based contrast for test-time adaptive retrieval,” IEEE Transactions on Multimedia, vol. 26, pp. 8665–8677, 2024.

[31] T. Gong, J. Jeong, T. Kim, Y. Kim, J. Shin, and S.-J. Lee, “Note: Robust continual test-time adaptation against temporal correlation,” Advances in Neural Information Processing Systems, vol. 35, pp. 27 253–27 266, 2022.

[32] D. Sojka, S. Cygert, B. Twardowski, and T. Trzci´ nski, “Ar-tta: A simple´ method for real-world continual test-time adaptation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3491–3495.

[33] F. F. Niloy, S. M. Ahmed, D. S. Raychaudhuri, S. Oymak, and A. K. Roy-Chowdhury, “Effective restoration of source knowledge in continual test time adaptation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 2091–2100.

[34] G. Wang, C. Ding, W. Tan, and M. Tan, “Decoupled prototype learning for reliable test-time adaptation,” IEEE Transactions on Multimedia, 2025.

[35] J. Geng, Z. Chen, Y. Wang, H. Woisetschlaeger, S. Schimmler, R. Mayer, Z. Zhao, and C. Rong, “A survey on dataset distillation: Approaches, applications and future directions,” in IJCAI, 2023.

[36] A. Carta, A. Cossu, V. Lomonaco, and D. Bacciu, “Distilled replay: Overcoming forgetting through synthetic samples,” in Continual Semi-Supervised Learning: First International Workshop, CSSL 2021, Virtual Event, August 19-20, 2021, Revised Selected Papers, vol. 13418. Springer Nature, 2022, p. 104.

[37] H. Jin, S. Liu, C. Cong, Q. Feng, Y. Liu, L. Huang, and Y. Hu, “Fedwsidd: Federated whole slide image classification via dataset distillation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2025, pp. 178–188.

[38] X. Dong, L. Wang, X. Lv, X. Zhang, H. Zhang, B. Pu, Z. Gao, I. Y. Liao, and Z. Jin, “Certaintta: Estimating uncertainty for test-time adaptation on medical image segmentation,” Information Fusion, p. 103300, 2025.

[39] S. Niu, J. Wu, Y. Zhang, Y. Chen, S. Zheng, P. Zhao, and M. Tan, “Efficient test-time model adaptation without forgetting,” in International conference on machine learning. PMLR, 2022, pp. 16 888–16 905.

[40] M. Tan, G. Chen, J. Wu, Y. Zhang, Y. Chen, P. Zhao, and S. Niu, “Uncertainty-calibrated test-time model adaptation without forgetting,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[41] J. Liang, D. Hu, and J. Feng, “Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation,” in Proceedings of the International Conference on Machine Learning (ICML), 2020, pp. 6028–6039.

[42] R. Rahaman et al., “Uncertainty quantification and deep ensembles,” Advances in neural information processing systems, vol. 34, pp. 20 063– 20 075, 2021.

[43] T. Abe, E. K. Buchanan, G. Pleiss, R. Zemel, and J. P. Cunningham, “Deep ensembles work, but are they necessary?” Advances in Neural Information Processing Systems, vol. 35, pp. 33 646–33 660, 2022.

[44] J. Denker and Y. LeCun, “Transforming neural-net output levels to probability distributions,” Advances in neural information processing systems, vol. 3, 1990.

[45] D. Hendrycks and K. Gimpel, “A baseline for detecting misclassified and out-of-distribution examples in neural networks,” in International Conference on Learning Representations, 2017.

[46] W. Liu, X. Wang, J. Owens, and Y. Li, “Energy-based out-of-distribution detection,” Advances in neural information processing systems, vol. 33, pp. 21 464–21 475, 2020.

[47] P. C. Trimmer, A. I. Houston, J. A. Marshall, M. T. Mendl, E. S. Paul, and J. M. McNamara, “Decision-making under uncertainty: biases and bayesians,” Animal cognition, vol. 14, no. 4, pp. 465–476, 2011.

[48] M. Tan, P. Chen, H. Zhi, J. Mai, B. Rosman, D. Ji, and R. Zeng, “Sourcefree elastic model adaptation for vision-and-language navigation,” IEEE Transactions on Multimedia, 2025.

[49] A. Kumar, P. S. Liang, and T. Ma, “Verified uncertainty calibration,” Advances in neural information processing systems, vol. 32, 2019.

[50] H. Ritter, A. Botev, and D. Barber, “A scalable laplace approximation for neural networks,” in 6th international conference on learning representations, ICLR 2018-conference track proceedings, vol. 6. International Conference on Representation Learning, 2018.

[51] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.

[52] D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo et al., “The many faces of robustness: A critical analysis of out-of-distribution generalization,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 8340–8349.

[53] M. Zhang, S. Levine, and C. Finn, “Memo: Test time robustness via adaptation and augmentation,” Advances in neural information processing systems, vol. 35, pp. 38 629–38 642, 2022.

[54] Q. Zhang, Y. Bian, X. Kong, P. Zhao, and C. Zhang, “Come: Testtime adaptation by conservatively minimizing entropy,” in International Conference on Learning Representations, 2025.

[55] W. Choi, D.-Y. Kim, J. Park, J. Lee, Y. Park, D.-J. Han, and J. Moon, “Adaptive energy alignment for accelerating test-time adaptation,” in The Thirteenth International Conference on Learning Representations, 2025.

[56] S. Zagoruyko and N. Komodakis, “Wide residual networks,” in British Machine Vision Conference 2016. British Machine Vision Association, 2016.

[57] D. Yin, R. Gontijo Lopes, J. Shlens, E. D. Cubuk, and J. Gilmer, “A fourier perspective on model robustness in computer vision,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[58] F. Croce, M. Andriushchenko, V. Sehwag, E. Debenedetti, N. Flammarion, M. Chiang, P. Mittal, and M. Hein, “Robustbench: a standardized adversarial robustness benchmark,” in Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2020.

[59] L. Bottou, “Large-scale machine learning with stochastic gradient descent,” in Proceedings of COMPSTAT. Springer, 2010, pp. 177–186.

[60] Y. Wang, J. Hong, A. Cheraghian, S. Rahman, D. Ahmedt-Aristizabal, L. Petersson, and M. Harandi, “Continual test-time domain adaptation via dynamic sample selection,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 1701–1710.

[61] D. Lee, J. Yoon, and S. J. Hwang, “Becotta: Input-dependent online blending of experts for continual test-time adaptation,” in International Conference on Machine Learning. PMLR, 2024, pp. 27 072–27 093.

[62] O. Press, S. Schneider, M. Kummerer, and M. Bethge, “Rdumb: A¨ simple approach that questions our progress in continual test-time adaptation,” Advances in Neural Information Processing Systems, vol. 36, pp. 39 915–39 935, 2023.

[63] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne.” Journal of machine learning research, vol. 9, no. 11, 2008.