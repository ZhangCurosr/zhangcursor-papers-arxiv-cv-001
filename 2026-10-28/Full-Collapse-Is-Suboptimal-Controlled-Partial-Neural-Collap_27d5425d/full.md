# Full Collapse Is Suboptimal: Controlled Partial Neural Collapse for Long-Tailed Recognition

Hyeontaek Hwang<sup>1</sup> Cheonghwan Hur<sup>2</sup> Daeyoung Kim<sup>1</sup>

<sup>1</sup>KAIST <sup>2</sup>RTST Inc.

<sup>1</sup>{hwanght, kimd}@kaist.ac.kr <sup>2</sup>princehur@rtst.co.kr

## Abstract

Recent long-tailed recognition (LTR) methods incorporate neural collapse (NC) and align features to a fixed Equiangular Tight Frame (ETF) as a uniform target for all classes. However, forcing tail classes to fully collapse over-contracts their samples and weakens generalization. To address this, we propose a controlled partial neural collapse, realized through two coordinated controls on a backbone: the Class-Adaptive Sparse Activation (CASA) classifier and the ETF-constrained adaptive margin (ETF-AM) loss. Rather thanforcing a uniformfull collapse, this regime allows head classes to collapse along the ETF while tail classes retain within-class diversity. This prevents the tail from being forced into full collapse, which is suboptimal under imbalance. To set the dimensionality ofcollapse, CASA induces class-related node specialization through featureconditioned gating so that tail classes occupy their own feature directions. For the clearance axis, ETF-AM applies afrequency-adaptive margin, with smaller marginsfor head classes and largerfor tail classes. On CIFAR100-LT, ImageNet-LT, Places-LT, and iNaturalist-2018, our method achieves the best overall accuracy over prior methods. In the empirical analysis, within-class spread increases monotonically from head to tail classes, consistent with tail classes retaining variability rather than fully collapsing.

## 1. Introduction

Real-world data often follows a long-tailed distribution, where head (Many) classes hold most samples, and tail (Few) classes hold only a handful [38, 56]. Models trained on such data are biased toward head classes and lose accuracy on tail classes, which limits their reliable deployment in safety-critical domains [13, 46, 60]. Long-tailed recognition (LTR) studies how to train accurate models despite this imbalance [4, 5, 17, 26, 27, 31, 34, 40, 41, 52, 53, 55, 59].

A recent line of LTR research draws on neural collapse (NC), which describes the terminal-phase geometry of a network trained on balanced data [35]. Under NC, withinclass features collapse to their class mean, and the class means form a simplex Equiangular Tight Frame (ETF) that maximizes inter-class separation. Several methods import this geometry into LTR by fixing the classifier to an ETF and training the features to match it [29, 51] or by regularizing feature directions toward it [15]. These methods pursuefull collapse for every class.

Full collapse, however, is the optimal geometry only for balanced data. Under imbalance, the global-optimal last-layer geometry deviates significantly from the balanced ETF. For instance, minority classes may degenerate into a single shared direction [8] or, depending on the loss, scatter across multiple directions [50]. Additionally, finite regularization further distorts this structure away from the ideal ETF [6, 44].

Forcing tail classes to collapse fully therefore overcontracts their samples and removes the within-class variance that tail generalization needs [11, 18]. Prior NC-based LTR methods inherit this problem because they target the same full collapse geometry across all classes.

We reframe LTR around controlled partial neural collapse: head classes effectively collapse along the ETF while tail classes retain within-class diversity. The two requirements act on distinct NC properties, within-class collapse (NC1) and inter-class separation (NC2), which the standard NC framework quantifies separately [35]. The two interact through a trade-off that favors retaining a positive tail spread over full collapse.

Based on this motivation, we prove that a positive within-class spread, rather than full collapse, lowers balanced test risk under a finite-sample class-conditional model. We realize controlled partial collapse with two coordinated controls on a backbone (Fig. 1). One control sets the dimensionality of collapse. CASA uses featureconditioned node-level gating, allowing different classes to develop distinct activation patterns. Different classes then occupy distinct feature directions instead of a single shared subspace. The other control targets the training-to-inference clearance. Our ETF-constraint Adaptive Margin (ETF-AM)

![](images/a0acfba0260d69c5cf543ff25eef29d52aeef60cd0d67c6adc41fec6f17b7dba.jpg)  
Figure 1. Controlled partial neural collapse via two coordinated mechanisms. Full collapse over-concentrates tail classes at their ETF vertices (center); our method addresses this by controlling the dimensionality of collapse via instance-conditioned node selection (CASA, right; Sec. 3.1) and the training-to-inference clearance per class (ETF-AM, left; Sec. 3.2).

loss fixes the classifier to an ETF for inter-class separation and assigns larger margins to tail classes, widening their training-to-inference clearance.

In summary, our contributions are as follows:

• Theoretical suboptimality of full collapse: Under a finite-sample class-conditional model, we show that full collapse is not risk-optimal for classes below an explicit sample-count threshold, and that the optimal within-class spread is ordered by class frequency among those classes.

• Controlled Partial Collapse: To effectively induce partial collapse, we propose two coordinated mechanisms: the CASA classifier gates feature nodes to dictate the dimensionality of collapse, while the ETF-AM loss imposes an adaptive margin to set its clearance.

• Empirical validation: Our method achieves the best overall accuracy over prior methods on CIFAR100-LT, ImageNet-LT, Places-LT, and iNaturalist-2018. The empirical analysis shows that further tail compression degrades generalization on tail classes.

## 2. Background and Related Work

Existing approaches to LTR have primarily relied on reweighting [14, 25, 39, 40, 43, 63], loss design [5, 48], logit adjustment [25, 32, 53, 59], and margin adjustment [4, 21]. These methods apply margins within a learnable, unconstrained classifier [1, 4] and therefore lack an explicit geometric constraint on the final embedding structure [51].

Several studies induce NC to improve long-tailed recognition [15, 29, 51]. NC [35] emerges at the end of training on balanced data: within-class features collapse to one center (NC1), class centers form a simplex ETF (NC2), classifier weights align with the centers (NC3), and decisions follow the nearest center (NC4). These methods fix the classifier to an ETF, or regularize features toward it, and pursue full collapse for every class.

Theoretical analyses [6, 8, 44, 50] show that the optimal geometry deviates from that of the balanced ETF when the data is imbalanced. The layer-peeled model exhibits minority collapse, in which minority-class weights degenerate into a single shared direction beyond an imbalance threshold [8]. Vanishing-regularization analysis shows the optimum follows a simplex-encoded-label geometry rather than the ETF [44]. Similarly, an unconstrained ReLU analysis confirms that class means become orthogonal with classdependent lengths rather than equiangular [6]. Recent work argues that minority classes are best represented by multiple centers rather than a single collapsed mean [50].

A complementary line of work studies inter-class geometry when the number of classes C exceeds the feature dimension d + 1. In this regime, a simplex ETF is geometrically impossible; prior work establishes that the optimal inter-class target is instead a generalized maximalseparation frame with coherence approaching the Welch bound p(C − d)/(d(C − 1)) [12, 19, 28]. These works target the inter-class frame under full collapse. Our analysis is orthogonal to this choice, since it acts only on the within-class spread and holds for any maximal-separation target.

Recent methods continue to pursue full NC recovery: All-around Neural Collapse (AllNC) [54] restores collapse across activations, means, and classifier vectors. In parallel, Space Alignment [47] corrects feature-classifier misalignment to better reach the ETF. Both treat the balanced ETF as the target for every class. We take the complementary view: for tail classes, this target benefits from relaxation rather than full collapse. Moreover, achieving such full NC recovery with these methods relies on heavier training machinery, such as additional branches, contrastive objectives, or alignment modules, rather than on a single backbone.

## 3. Methodology

Prior NC-based LTR methods operate on the inter-class axis, refining the mean geometry while keeping every class fully collapsed. In contrast, we achieve controlled partial collapse by intervening on the within-class geometry along two distinct axes. On the dimensionality axis, the CASA classifier (Sec. 3.1) uses node-level gating to give each class its own feature subspace rather than forcing all classes to share one. On the clearance axis, the ETF-AM loss (Sec. 3.2) applies a per-class adaptive margin that sets a frequency-dependent training-to-inference clearance. In this section, we first present the theoretical motivation for this design, then describe the main components, as illustrated in Fig. 1.

Theoretical Motivation. We utilize the unconstrainedfeatures model (UFM), which treats last-layer features as free variables and is standard for NC analysis [6, 51]. We consider C classes with sample counts $n _ { 1 } \geq \cdots \geq n _ { C }$ . The class means align with a maximal-separation inter-class target, yielding a base separation margin $\gamma > 0$ from the nearest decision boundary.

The within-class spread $\begin{array} { r } { s _ { c } = \frac { 1 } { n _ { c } } \sum _ { i \in c } \| h _ { i } - \mu _ { c } \| ^ { 2 } } \end{array}$ measures how tightly features $h _ { i }$ cluster around the class mean $\mu _ { c } ;$ full collapse drives every $s _ { c }$ to zero. To reason about generalization, we adopt a finite-sample class-conditional model, defining the signed test margin of a class-c sample as

$$
\mathcal { M } _ { c } ( s _ { c } ) = ( \gamma - a s _ { c } ) + Z _ { \mathrm { t e s t } } + Z _ { \mathrm { e s t } } ( s _ { c } , n _ { c } ) ,\tag{1}
$$

with separation cost $- a s _ { c } ,$ where $a \ > \ 0$ is the signed distance lost per unit of spread, irreducible noise $Z _ { \mathrm { t e s t } } \sim$ ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ with $\sigma ^ { 2 } > 0$ , and Gaussian finite-sample term $Z _ { \mathrm { e s t } }$ . A sample is correctly classified when $\mathcal { M } _ { c } ( s _ { c } ) > 0$ and $\mathcal { R } _ { c } ( s _ { c } ) = \mathrm { P r } [ \mathcal { M } _ { c } ( s _ { c } ) \leq 0 ]$ ] is the per-class risk.

On free features, the UFM optimum is full collapse. But the UFM treats features as free variables with no sample count, so its optimum is independent of $n _ { c }$ and cannot reflect how Few-sample classes generalize.

To handle this, we draw on recent transfer learning and generalization analyses [10, 11], which establish withinclass variability as a key determinant of finite-sample error, with estimates that degrade as the number of available samples shrinks. We formalize this via the variance of the finite-sample term, which grows as $n _ { c }$ decreases: raising $s _ { c }$ from zero trades off separation against estimation variance.

Assumption 1 (Finite-sample variance model). Let $V ( s _ { c } , n _ { c } ) \triangleq \operatorname { V a r } ( Z _ { \mathrm { e s t } } )$ , then

$$
V ( s _ { c } , n _ { c } ) = \phi ( s _ { c } ) / n _ { c } ,\tag{2}
$$

where $\phi \geq 0$ is differentiable on $s _ { c } \geq 0$ with $0 < \phi ( 0 ) < \infty$ and $\phi ^ { \prime } ( 0 ) < 0 .$ . For the cross-class statement below, $a , \gamma ,$ $\sigma ,$ and ϕ are shared across classes; the class counts $n _ { c }$ are the varying model parameters.

Since M is Gaussian, $\mathcal { R } _ { c } ( s _ { c } ) = \Phi ( - g ( s _ { c } ) )$ with $g =$ $( \gamma - a s _ { c } ) / \sqrt { \sigma ^ { 2 } + V ( s _ { c } , n _ { c } ) }$ . The condition $\mathcal { R } _ { c } ^ { \prime } ( 0 ) < 0 .$ under which a small positive spread lowers the risk, holds exactly when

$$
- V _ { s } ( 0 , n _ { c } ) > \frac { 2 a \left[ \sigma ^ { 2 } + V ( 0 , n _ { c } ) \right] } { \gamma } ,\tag{3}
$$

where $\begin{array} { r } { { \cal V } _ { s } ~ = ~ { \frac { \partial V } { \partial s _ { c } } } } \end{array}$ . We call this the sensitivity condition $( { \mathrm { S e c . ~ A } } )$

Theorem 1 (Suboptimality of full collapse). Let V be as in Assumption 1. For any class c satisfying Eq. (3), the perclass risk on $[ 0 , \gamma / a ]$ attains its global minimum in the interior. Writing $s _ { c } ^ { \star }$ for the smallest global minimizer,

$$
s _ { c } ^ { \star } > 0 \quad a n d \quad \mathcal { R } _ { c } ( 0 ) > \mathcal { R } _ { c } ( s _ { c } ^ { \star } ) .\tag{4}
$$

Because the balanced risk is the average of the separable class-wise risks, if at least one class satisfies Eq. (3), the all-class zero-spread solution is not globally optimal within this model. Theorem 1 leaves open which classes satisfy the condition; the following Corollary 1 resolves it into a threshold on the training count.

Corollary 1 (Scarcity-dependent spread). Let $n ^ { \star }$ be the sample count at which Eq. (3) changes sign. Then $E q . \ ( 3 )$ holds if and only $i f 0 < n _ { c } < n ^ { \star }$ . If, in addition, $\phi ^ { \prime } ( s ) < 0$ for every $s \in ( 0 , \gamma / a )$ , then for any two classes $c , c ^ { \prime }$ with $n _ { c ^ { \prime } } < n _ { c } < n ^ { \star }$

$$
s _ { c } ^ { \star } < s _ { c ^ { \prime } } ^ { \star } .\tag{5}
$$

The threshold is what Eq. (3) becomes once $V = \phi / n _ { c }$ is substituted, and it is non-vacuous when $- \gamma \phi ^ { \prime } ( 0 ) > 2 a \phi ( 0 )$ (Sec. A.3). The ordering needs ϕ to decrease on the whole interval not only at zero. Within the shared-parameter model, both the threshold and the ordering are governed by $n _ { c }$

## 3.1. Class-Adaptive Sparse Activation Classifier

## 3.1.1. Detailed Architecture

The CASA classifier controls the dimensionality of collapse. Forcing every class to collapse within the same shared feature subspace over-contracts tail classes, since their features are squeezed into a subspace shared across classes rather than their own dedicated direction [50]. As shown in Fig. 2, CASA instead assigns each class its own subset of feature dimensions through a node-level gating layer, so head and tail classes share the fixed ETF target for their means but operate in distinct subspaces around it.

CASA incorporates a node-level gating layer, a single fully connected layer integrated within the main classification layer, which selectively activates a subset of specialized nodes for each instance (Eq. (6)):

![](images/97a5f84f246324c926b355692a343a5da22fbc9c18991e66f8826044c3650f5d.jpg)  
Figure 2. The overall architecture of the CASA classifier, which selects a subset of nodes through instance-conditioned node-level gating.

$$
M _ { \mathrm { s o f t } } = \mathrm { S i g m o i d } ( G ( h ; \omega _ { \mathrm { g a t e } } ) ) ,\tag{6}
$$

where $M _ { \mathrm { s o f t } } \ \in \ \mathbb { R } ^ { d }$ represents the soft mask vector, $G ( \cdot )$ denotes the gating network, $h = f ( x ; \omega )$ is the feature produced by the feature extractor for input $x ,$ and $\omega _ { \mathrm { g a t e } }$ are the trainable parameters of the gating layer.

During training, $M _ { \mathrm { s o f t } }$ is discretized using a thresholding operation to produce the binary mask M for class-related node selection:

$$
M = \left\{ \begin{array} { l l } { \mathbb { I } _ { \{ m _ { i } > 0 . 5 \} } ( M _ { \mathrm { s o f t } } ) } & { \mathrm { i f ~ t r a i n i n g } } \\ { M _ { \mathrm { s o f t } } } & { \mathrm { o t h e r w i s e } } \end{array} , \right.\tag{7}
$$

where $m _ { i }$ is the i-th element of $M _ { \mathrm { s o f t } }$ and I is the indicator function. The hard threshold during training drives classspecific node selection; the soft mask at test aggregates node contributions smoothly for stable generalization.

The mask is applied along the node dimension, $W _ { m } =$ $M \odot W .$ , and the forward pass is $\hat { y } = W _ { m } ^ { \top } h + b$ . Since M acts on the node dimension shared by all classes, this is equivalent to masking the feature:

$$
W _ { m } ^ { \top } h = W ^ { \top } ( M \odot h ) ,\tag{8}
$$

where the classifier weights are fixed to the ETF structure. CASA selects the subspace in which each class’s features collapse.

## 3.1.2. Instance-Conditioned Gating and Its Class Structure

The gate is conditioned on the instance feature h and trained jointly with the classifier. When the masked weights on the classifier are $W _ { m } = M \odot W$ , the prediction is

$$
\hat { y } = f _ { \mathrm { c l s } } ( h ; W _ { m } ) ,\tag{9}
$$

where $\hat { y }$ is the prediction and $f _ { \mathrm { c l s } } ( \cdot )$ is that classifier. Optimizing it with a classification loss such as cross-entropy gives:

$$
\mathcal { L } _ { \mathrm { c l s } } = - \sum _ { i = 1 } ^ { N } { y _ { i } ^ { \top } \log f _ { \mathrm { c l s } } ( h _ { i } ; W _ { m } ) } ,\tag{10}
$$

where $y _ { i }$ is the one-hot ground-truth label and $h _ { i }$ is the feature of sample $x _ { i }$ . Because the mask is produced from the feature and the loss is minimized class-wise, samples of the same class receive correlated gate inputs, which empirically yields similar masks within a class.

During the backward pass, we use a straight-through estimator [3, 42]. The hard threshold is bypassed on the backward path, and gradients flow only through the weights of activated nodes (Eq. (11)):

$$
\frac { \partial \mathcal { L } } { \partial \omega _ { \mathrm { g a t e } } } = \mathbb { I } _ { \{ m _ { i } > 0 . 5 \} } \cdot \frac { \partial \mathcal { L } } { \partial m _ { i } } \cdot \frac { \partial m _ { i } } { \partial \omega _ { \mathrm { g a t e } } } .\tag{11}
$$

This selective gradient flow restricts updates to the activated nodes. Furthermore, we employ a spike-and-slab prior [42] on the gating mechanism to encourage sparse masks. The prior term is formulated as:

$$
\mathcal { L } _ { g } = \vartheta ( 1 - m ) + ( 1 - \vartheta ) \frac { m ^ { 2 } } { 2 \varpi ^ { 2 } } .\tag{12}
$$

Here $\vartheta$ weighs the spike-and-slab term and $\varpi ^ { 2 }$ is the slab variance: the spike term pushes nodes toward activation while the slab term prevents the gate from saturating, balancing sparsity against signal. Unlike attention-based or feature reweighting methods, the hard threshold and sparse gradient flow activate a sparse subset of nodes rather than reweighting all nodes.

## 3.2. ETF-Constrained Adaptive Margin Loss

Under the additional monotonicity condition, Corollary 1 gives a wider risk-optimal spread to scarcer classes. Following this ordinal prediction, the ETF-AM loss combines a fixed ETF classifier with a per-class adaptive margin, giving scarcer classes a wider training-to-inference clearance.

## 3.2.1. ETF-Constrained Margin Formulation

Consider a multi-class classification problem with C classes, where the number of samples in class c is denoted by $n _ { c } .$ . We define an adaptive margin for each class as:

$$
\varrho _ { c } = \frac { \Gamma } { n _ { c } ^ { \varsigma } } ,\tag{13}
$$

where $\Gamma > 0$ is the margin scale fixed by the ETF geometry below. We set $\varsigma = 0 . 2 5$ following the frequency scaling of LDAM [4]. The exponent is a design choice rather than a rate derived from Theorem 1 and Corollary 1. Under the ETF constraint, the Euclidean distance between any two distinct class weight vectors is

$$
\| w _ { c } - w _ { c ^ { \prime } } \| = \sqrt { 2 + \textstyle { \frac { 2 } { C - 1 } } } , \quad c \neq c ^ { \prime } .\tag{14}
$$

Placing the decision boundary at the midpoint yields

$$
\begin{array} { r } { \Gamma = \frac 1 2 \| w _ { c } - w _ { c ^ { \prime } } \| \to \frac { \sqrt 2 } { 2 } \mathrm { ~ a s ~ } C \to \infty , } \end{array}\tag{15}
$$

with the full derivation in Sec. B of the supplementary material. Tail classes receive a larger $\varrho _ { c }$ and correspondingly larger clearance, while head classes keep a small margin.

We adopt an ETF classifier that enforces the constraints:

$$
\| w _ { c } \| = 1 \quad \mathrm { a n d } \quad \langle w _ { c } , w _ { c ^ { \prime } } \rangle = - { \frac { 1 } { C - 1 } } \quad \mathrm { f o r } c \neq c ^ { \prime } ,\tag{16}
$$

which guarantees that all class weight vectors $w _ { c }$ are uniformly distributed on the unit hypersphere. Given a feature embedding $h \in \mathbb { R } ^ { d }$ extracted by a feature extractor $f ( x ; \omega )$ the logit for class c is computed as:

$$
z _ { c } = w _ { c } ^ { \top } u + b _ { c } , \quad u = M \odot h ,\tag{17}
$$

where u is the gated feature of Eq. (8). We then modify the logit for the true class $y$ by subtracting the corresponding margin:

$$
\hat { z } _ { c } = \left\{ \begin{array} { l l } { z _ { c } - \varrho _ { c } , } & { \mathrm { i f ~ } c = y , } \\ { z _ { c } , } & { \mathrm { o t h e r w i s e } } \end{array} \right. .\tag{18}
$$

Subtracting $\varrho _ { y }$ makes the training condition $z _ { y } - z _ { k } > \varrho _ { y }$ stricter than the inference rule $z _ { y } - z _ { k } > 0$ . The two hyperplanes are parallel and separated by $\Delta _ { y k } = \varrho _ { y } / \| w _ { y } - w _ { k } \|$ By Eq. (14), this distance is independent of the competitor k, giving a single per-class clearance $\Delta _ { y } = \varrho _ { y } / ( 2 \Gamma )$ A feature that meets the training condition therefore retains at least $\Delta _ { y }$ in signed distance once the margin is removed. The margin is imposed on the gated feature, and the ETF fixes $\| w _ { y } - w _ { k } \|$ irrespective of the mask, so gating leaves $\Delta _ { y }$ unchanged.

While the frequency scaling follows LDAM [4], the fixed frame changes what the margin means geometrically. Under a learnable head the same $\varrho _ { y }$ maps to a different distance for each competitor $k ,$ whereas here it maps to a single per-class clearance $\Delta _ { y } .$ . The margin-adjusted loss is defined via the standard cross-entropy formulation:

$$
\mathcal { L } _ { \mathrm { E T F - A M } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( z _ { y _ { i } } - \varrho _ { y _ { i } } ) } { \exp ( z _ { y _ { i } } - \varrho _ { y _ { i } } ) + \sum _ { c \neq y _ { i } } \exp ( z _ { c } ) } ,\tag{19}
$$

where N is the number of samples, $y _ { i }$ denotes the groundtruth label for sample i and $\varrho _ { y _ { i } }$ is the adaptive margin for that class. Because the ETF weights are fixed at maximal separation (Eq. (16)), the network can reduce $\mathcal { L } _ { \mathrm { E T F - A M } }$ only through the backbone and the gate, not by rotating the classifier. A further discussion and clearance derivation are in Sec. C.

The overall loss function integrates the classification loss and the sparse regularization term induced by the spike-andslab prior:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { E T F - A M } } + \mathcal { L } _ { g } + \beta \cdot \| h \| ^ { 2 } , } \end{array}\tag{20}
$$

where $\mathcal { L } _ { g }$ corresponds to the sparse regularization in Eq. $( 1 2 ) , \beta$ is the coefficient of the feature-norm regularizer.

Together, the $\mathcal { L } _ { \mathrm { E T F - A M } }$ and $\mathcal { L } _ { g }$ terms act on the dimensionality of collapse and clearance, thereby promoting partial collapse.

## 4. Experiments

## 4.1. Experiment Setup

## 4.1.1. Metric

We use top-1 accuracy [1] and, for detailed analysis, the standard split into Many (> 100 samples), Medium (> 20 and ≤ 100), and Few (≤ 20) subsets, following previous works [1, 4, 30].

Following Hasegawa and Sato [15], we also use the FDR score [9], the ratio of between-class scatter to withinclass scatter, which quantifies the emergence of NC; maskconsistency metrics (cosine similarity [33], symmetric KL [24]) are defined in Sec. D of the supplementary material. In our partial collapse setting, FDR primarily tracks inter-class separation (NC2/NC3); within-class relaxation is measured separately via the per-class spread of Sec. 4.4.

## 4.1.2. Dataset

We use four datasets to evaluate the proposed method on long-tailed classification tasks.

CIFAR100-LT. CIFAR100-LT down-samples CI-FAR100 [23] with imbalance factor $\rho = \mathrm { m a x } ( n _ { c } ) / \mathrm { m i n } ( n _ { c } )$ the ratio between the largest and smallest class sizes [1].

ImageNet-LT. Following [30], we construct ImageNet-LT by subsampling the original ImageNet [7] using a Pareto distribution with a power value $\alpha = 6 [ 2 ]$ , resulting in 5 to 1,280 images per class for training.

Places-LT. Following the same procedure, Places-LT applies a Pareto distribution $( \alpha ~ = ~ 6 )$ to the 365 classes of Places [62], with class sizes from 5 to 4,980.

iNaturalist-2018. iNaturalist-2018 [45] is a naturally long-tailed fine-grained dataset (imbalance factor $\rho$ of 500) with $C = 8$ ,142 classes, 437K training and 24K validation images.

## 4.1.3. Implementation Details

We use ResNet34 [16] for CIFAR100-LT, ResNeXt50 [49] for ImageNet-LT, and ResNet152 [16] for Places-LT, following the implementation of Alshammari et al. [1], Hasegawa and Sato [15], Zhang et al. [57]. For iNaturalist-2018, we use ResNet50 [16] (d = 2,048); since ${ \cal C } =$ $8 , 1 4 2 > d + 1$ , no simplex ETF exists, so we train the classifier directly on the features while keeping CASA gating and the ETF-AM loss with tuned Γ. Models are trained on long-tailed sets and evaluated on balanced test sets. Following prior works [30, 57, 58], Places-LT uses ImageNet-1k pre-trained weights from PyTorch [36]; CIFAR100-LT and ImageNet-LT train from scratch. We apply post-training logit adjustment (LA) [15, 22]. Main results (Tabs. 1 to 4)

Table 1. Top-1 accuracy on the test set of CIFAR100-LT (ρ = 100). Baseline numbers are taken from the original papers; “-” denotes results not reported in the original papers. Bold marks the best and underline the second-best result. † indicates result with ResNet34
<table><tr><td>Method</td><td>Many</td><td>Medium</td><td>Few</td><td>All</td></tr><tr><td>CE (Baseline)</td><td>64.60</td><td>36.90</td><td>11.90</td><td>38.50</td></tr><tr><td>CB [5]</td><td></td><td>一</td><td></td><td>39.60</td></tr><tr><td>LDAM [4]</td><td>一</td><td>一</td><td>一</td><td>42.04</td></tr><tr><td>BALMS [40]</td><td></td><td>一</td><td></td><td>50.80</td></tr><tr><td>VS+SAM [39]</td><td>56.40</td><td>48.80</td><td>31.70</td><td>46.60</td></tr><tr><td>GCL [26]</td><td></td><td></td><td></td><td>48.70</td></tr><tr><td>WB [1] †</td><td>72.60</td><td>51.86</td><td>32.63</td><td>53.35</td></tr><tr><td>RBL [37]</td><td>一</td><td>一</td><td></td><td>53.10</td></tr><tr><td>NC-DRW-cRT [29]</td><td>一</td><td>一</td><td></td><td>48.70</td></tr><tr><td>DLLA [53]</td><td></td><td></td><td></td><td>50.08</td></tr><tr><td>ETF [15] †</td><td>74.20</td><td>52.90</td><td>29.90</td><td>53.00</td></tr><tr><td>BIB [25]</td><td></td><td>一</td><td></td><td>44.90</td></tr><tr><td>MBIB [25]</td><td></td><td></td><td></td><td>47.50</td></tr><tr><td>Ours †</td><td>74.71</td><td>55.77</td><td>31.65</td><td>54.73</td></tr></table>

Table 2. Top-1 accuracy on the test set of ImageNet-LT with ResNeXt50. Baseline numbers are taken from the original papers; “–” denotes results not reported in the original papers. Bold marks the best and underline the second-best result.
<table><tr><td>Method</td><td>Many</td><td>Medium</td><td>Few</td><td>All</td></tr><tr><td>CE (Baseline)</td><td>65.90</td><td>37.50</td><td>7.70</td><td>44.40</td></tr><tr><td>LDAM [4]</td><td></td><td></td><td></td><td>51.30</td></tr><tr><td>ALA [59]</td><td>64.10</td><td>49.90</td><td>34.70</td><td>53.30</td></tr><tr><td>WB [1]</td><td>62.50</td><td>50.40</td><td>41.50</td><td>53.90</td></tr><tr><td>RBL [37]</td><td>64.80</td><td>49.60</td><td>34.20</td><td>53.30</td></tr><tr><td>NC-DRW-cRT [29]</td><td>65.60</td><td>51.20</td><td>35.40</td><td>54.20</td></tr><tr><td>DLLA [53]</td><td></td><td></td><td></td><td>54.50</td></tr><tr><td>ETF [15]</td><td>62.40</td><td>49.30</td><td>36.00</td><td>52.60</td></tr><tr><td>BIB [25]</td><td>64.70</td><td>51.20</td><td>32.70</td><td>53.90</td></tr><tr><td>MBIB [25]</td><td>67.00</td><td>52.80</td><td>33.50</td><td>55.70</td></tr><tr><td>Ours</td><td>65.20</td><td>55.35</td><td>38.87</td><td>56.90</td></tr></table>

use a fixed seed, following the protocol of the cited baselines. Further training settings are in Sec. E of the supplementary material.

## 4.2. SOTA Benchmark

Tabs. 1 to 3 show that our method achieves the best overall accuracy over prior approaches on all three benchmarks: 54.73% (+1.38%p) on CIFAR100-LT, 56.90% (+1.20%p) on ImageNet-LT, and 41.20% (+0.56%p) on Places-LT. The gains are concentrated in the Medium and Few splits, while Many-class accuracy is preserved. These results are consistent with the frequency-adaptive margin, which assigns larger margins to underrepresented classes.

iNaturalist-2018. Tab. 4 reports results on iNaturalist-2018, the natural fine-grained long-tail benchmark. Our method achieves 73.50% overall top-1 accuracy, outperforming prior methods. We utilize a learnable classifier because no simplex ETF exists. The partial-collapse still applies since Theorem 1 is agnostic to the inter-class target, and tail classes retain larger within-class spread than head classes here as well (Tab. 7). This will be discussed more in Sec. 4.4.

Table 3. Top-1 accuracy on the test set of Places-LT with ResNet152. Baseline numbers are taken from the original papers; “–” denotes results not reported in the original papers. Bold marks the best and underline the second-best result.
<table><tr><td>Method</td><td>Many</td><td>Medium</td><td>Few</td><td>All</td></tr><tr><td>CE (Baseline)</td><td>45.70</td><td>27.30</td><td>8.20</td><td>30.20</td></tr><tr><td>OLTR [30]</td><td>44.70</td><td>37.00</td><td>25.30</td><td>35.90</td></tr><tr><td>D-LWS [20]</td><td>40.60</td><td>39.10</td><td>28.60</td><td>37.60</td></tr><tr><td>BALMS [40]</td><td>41.20</td><td>39.80</td><td>31.60</td><td>38.70</td></tr><tr><td>DisAlign [56]</td><td>40.40</td><td>42.40</td><td>30.10</td><td>39.30</td></tr><tr><td>LADE[17]</td><td>42.80</td><td>39.00</td><td>31.20</td><td>38.80</td></tr><tr><td>GCL [26]</td><td></td><td></td><td></td><td>40.64</td></tr><tr><td>RBL [37]</td><td>44.10</td><td>40.70</td><td>24.40</td><td>38.80</td></tr><tr><td>DLLA [53]</td><td>43.90</td><td>40.10</td><td>32.90</td><td>40.10</td></tr><tr><td>Ours</td><td>44.10</td><td>41.30</td><td>35.50</td><td>41.20</td></tr></table>

Table 4. Top-1 accuracy on iNaturalist-2018 with ResNet50. Baseline numbers are taken from the original papers; “–” denotes results not reported in the original papers. Bold marks the best and underline the second-best result.
<table><tr><td>Method</td><td>Many</td><td>Medium</td><td>Few</td><td>All</td></tr><tr><td>CE (Baseline)</td><td>75.70</td><td>66.90</td><td>61.70</td><td>65.80</td></tr><tr><td>cRT [20]</td><td></td><td></td><td></td><td>65.20</td></tr><tr><td>LWS [20]</td><td></td><td></td><td></td><td>65.90</td></tr><tr><td>MiSLAS [61]</td><td>一</td><td>一</td><td></td><td>71.60</td></tr><tr><td>GCL [26]</td><td></td><td></td><td></td><td>72.00</td></tr><tr><td>WB [1]</td><td>71.00</td><td>70.30</td><td>69.40</td><td>70.00</td></tr><tr><td>DLLA [53]</td><td></td><td></td><td></td><td>71.30</td></tr><tr><td>BIB [25]</td><td>73.90</td><td>72.90</td><td>72.10</td><td>72.70</td></tr><tr><td>MBIB [25]</td><td>72.60</td><td>73.60</td><td>73.10</td><td>73.30</td></tr><tr><td>Ours (learnable head)</td><td>74.00</td><td>73.90</td><td>73.30</td><td>73.50</td></tr></table>

Footprint. Our method adds only a node-level gating layer. The gating layer and classifier scale with the embedding dimension d, adding 0.3M parameters at d=512 (CIFAR100-LT) and up to 4.2M at d=2,048 (ImageNet-LT, Places-LT, iNaturalist), while FLOPs increase by under 0.1% across all settings as shown in Tab. 5.

## 4.3. Ablation Study

We conduct various ablation studies on the ETF-AM loss and the gating layer, as summarized in Tab. 6.

Ablation of key components. As shown in Tab. 6, omitting ETF-AM leaves Many nearly unchanged (74.88%) but drops Medium and Few by 2.4 and 4.9 points (overal 52.42%). Removing the gating layer reduces accuracy to

Table 5. Parameter count and FLOPs of the proposed method, and the Vanilla model. (The number of parameters and FLOPs are measured with ptflops)
<table><tr><td></td><td></td><td colspan="2">Vanilla</td><td colspan="2">Ours</td></tr><tr><td>Dataset</td><td>Backbone</td><td>#param</td><td>FLOPs</td><td>#param</td><td>FLOPs</td></tr><tr><td>CIFAR100-LT</td><td>ResNet34</td><td>21.3M</td><td>2.3G</td><td>21.6M</td><td>2.3G</td></tr><tr><td>ImageNet-LT</td><td>ResNeXt50</td><td>25.0M</td><td>8.6G</td><td>29.2M</td><td>8.6G</td></tr><tr><td>Places-LT</td><td>ResNet152</td><td>60.2M</td><td>23.2G</td><td>64.4M</td><td>23.2G</td></tr><tr><td>iNaturalist</td><td>ResNet50</td><td>40.2M</td><td>8.2G</td><td>44.4M</td><td>8.2G</td></tr></table>

Table 6. Component ablation on CIFAR100-LT: FDRs and Top-1 accuracy. Bold marks the best.
<table><tr><td>Method</td><td>|FDR (Train) FDR (Test) |Many Medium Few Avg.</td><td></td><td></td><td></td></tr><tr><td>Full model</td><td> $\mid 9 . 9 4 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 8 \times 1 0 ^ { 2 }$ </td><td></td><td>|74.71 55.77 31.65 54.73</td></tr><tr><td colspan="5">Ablation on key components</td></tr><tr><td>w/o ETF-AM</td><td> $3 . 3 6 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 3 \times 1 0 ^ { 2 }$  |74.88</td><td>53.34</td><td>26.74 52.42</td></tr><tr><td>w/o Gating layer</td><td> $9 . 4 6 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 4 \times 1 0 ^ { 2 }$  74.20</td><td>54.10</td><td>30.60 53.60</td></tr><tr><td colspan="5">Capacity-matched gating and margin baselines</td></tr><tr><td>2-layer MLP</td><td> $\vert 3 . 8 4 \times 1 0 ^ { 4 }$   $1 . 0 5 \times 1 0 ^ { 2 }$ </td><td>70.30</td><td>45.00</td><td>29.50 48.80</td></tr><tr><td>Soft Attn.</td><td> $7 . 0 9 \times 1 0 ^ { 4 }$   $1 . 2 3 \times 1 0 ^ { 2 }$ </td><td>74.40</td><td>53.40</td><td>29.20 52.90</td></tr><tr><td>Fixed margin  $( \Gamma = 0 . 7 1 )$ </td><td> $6 . 2 3 \times 1 0 ^ { 4 }$   $1 . 2 2 \times 1 0 ^ { 2 }$ </td><td>73.70</td><td>52.90</td><td>29.60 52.80</td></tr></table>

53.60%, confirming that the masks aid inter-class separation.

These drops reflect that each component aligns with its role. Removing ETF-AM hurts the tail the most, consistent with keeping tail classes away from full collapse. Removing the gate yields a smaller, more uniform drop, consistent with the dimensionality of the axis.

Capacity-matched gating and margin baselines. To isolate whether CASA’s gain comes from added parameters or per-class node selection, we compare parameter-matched variants on CIFAR100-LT in Tab. 6. CASA (54.73%) outperforms the soft-attention gate (52.90%) and the two-layer MLP head (48.80%), demonstrating the effectiveness of the gating layer. Replacing the adaptive margin with a fixed margin Γ = 0.71 drops accuracy to 52.80%. The tail gain therefore comes from scaling the margin with class frequency, not from the margin alone. Extended experimental analysis, including an ablation study on a larger dataset and qualitative analysis, is provided in Sec. F of the supplementary materials.

## 4.4. Empirical Support for Partial Collapse

Theorem 1 shows that the risk is strictly worse at zero spread for classes below the threshold, and Corollary 1 orders their optimal spread by $n _ { c }$

We measure the within-class spread $s _ { c }$ (test set for CIFAR100-LT and validation set for iNaturalist-2018) of the trained models, as shown in Tab. 7. On CIFAR100-LT, the spread increases monotonically from Many to Few: the Few-class spread is 1.76× the Many-class spread. Compared with the cross-entropy baseline (without ETF-AM), ETF-AM retains a larger relative within-class spread on the Few classes (1.76 vs. 1.33), and this larger spread coexists with higher Few-class accuracy. We report relative spreads because the raw values are scale-dependent: on the Manyclasses, ETF-AM and CE differ in $s _ { c }$ (18.4 vs. 172.2) as their feature norm (8.6 vs. 26.1). This is consistent with the partial-collapse view, in which tail classes benefit from within-class diversity rather than from full collapse. The same monotonic pattern holds on iNaturalist-2018, where no simplex ETF exists, and the classifier is learned: the Few-class spread is 1.40× the Many-class spread under ETF-AM, against 1.01× for the CE baseline.

Table 7. Within-class spread $s _ { c } ,$ averaged within each frequency split and normalized to the Many-class mean. In both datasets, the spread increases monotonically from head to tail, and ETF-AM maintains a larger tail spread than the CE baseline.
<table><tr><td>Dataset</td><td>Loss</td><td>Many</td><td>Medium</td><td>Few</td></tr><tr><td>CIFAR100-LT</td><td>ETF-AM CE</td><td>1.00 1.00</td><td>1.40 1.18</td><td>1.76</td></tr><tr><td rowspan="2">iNaturalist-2018</td><td>ETF-AM</td><td></td><td></td><td>1.33</td></tr><tr><td>CE</td><td>1.00 1.00</td><td>1.38 1.04</td><td>1.40 1.01</td></tr></table>

This is the partial-collapse geometry: inter-class separation (NC2) is preserved while within-class collapse (NC1) is relaxed for tail classes. Theorem 1 acts only on the within-class spread and is agnostic to the inter-class target. The retained tail spread on both datasets, including the no-ETF iNaturalist regime, is consistent with it. Section F tests the zero-spread side with a controlled collapse experiment, and reports the frequency dependence of the spread.

## 4.5. Mask Consistency

To support the dimensionality of collapse controlled by CASA, we evaluate the consistency of the mask generation process by analyzing the symmetric KL divergence and cosine similarity between mask distributions computed from sample pairs. Fig. 3 illustrates the epoch-wise evolution of these metrics. The gating layer progressively decreases cosine similarity and increases KL divergence between samples of different classes, while showing the opposite trend within the same class. Same-class instances activate more similar subsets of nodes than different-class instances.

Tab. 8 quantifies the consistency of generated masks. On the train set, intra-class pairs reach cosine similarity 0.96 (KL 0.001) versus inter-class 0.48 (KL 1.115). On the test set, intra-class similarity drops to 0.81 (KL 0.046), while inter-class similarity 0.72 (KL 0.430) remains distinct. The gating layer therefore produces masks with class-related structure on both splits. The KL-divergence histograms in Fig. 4 further validate this: on both the train and test sets, same-class pairs concentrate at lower divergence than inter-class pairs on both splits. The intra-class divergence is slightly higher on the test set, reflecting the distribution shift between the long-tailed training set and the class-balanced test set rather than a loss of mask consistency. Further discussion about masks and success/failure case analysis is in Sec. F of the supplementary material.

![](images/2de1744dceed4a49a2df9e079e992de5c90272245a3187cd61981e6b441aad85.jpg)

![](images/a8b950fea78a7239791094bab3f0e12384e4c29748bf25abc7030943f11b5450.jpg)  
Figure 3. Epoch-wise cosine similarity (a) and symmetric KL divergence (b) of generated masks on CIFAR100-LT. Within-class masks grow more similar over training, whereas between-class masks separate.  
Figure 4. Histograms of symmetric KL divergence of generated mask (before thresholding on the training time) between sample pairs. (a) shows divergences computed from training set samples, and (b) displays divergences from test set samples. All histograms are plotted on a logarithmic scale.

Table 8. Cosine similarity and average KL divergence of generated masks between inter-class and intra-class on CIFAR100-LT.
<table><tr><td></td><td colspan="2">Cosine Similarity</td><td colspan="2">Avg. KL Divergence</td></tr><tr><td></td><td>Inter</td><td>Intra</td><td>Inter</td><td>Intra</td></tr><tr><td>Train</td><td>0.48</td><td>0.96</td><td>1.115</td><td>0.001</td></tr><tr><td>Test</td><td>0.72</td><td>0.81</td><td>0.430</td><td>0.046</td></tr></table>

## 5. Limitations

Our method has some limitations. (i) When $C > d + 1$ , a simplex ETF does not exist, so we utilize a learnable classifier head. We keep the same procedure, but $\| w _ { y } - w _ { k } \|$ is no longer constant across pairs: Γ must be tuned, and the clearance holds per pair rather than per class. (ii) Theorem 1 is established under a finite-sample class-conditional model (Sec. A). The positivity of the optimal spread follows from the qualitative form of the assumption alone, while the sample-count dependence is grounded in prior analyses [10, 11]. The cross-class ordering additionally requires ϕ to decrease throughout the feasible interval. We test the full-collapse-suboptimal side empirically and treat the interior optimum as a model-level prediction; a proof for the trained network remains open. (iii) We verify the claim for long-tailed image classification; transfer to other regimes, such as long-tailed detection or segmentation, remains open.

## 6. Conclusion

We reframed long-tailed recognition around controlled partial neural collapse: head classes effectively collapse along a fixed ETF while tail classes retain within-class diversity. Under a finite-sample class-conditional model, we showed that full collapse fails to minimize risk for sample-scarce classes and that the optimal spread is ordered by class frequency; our measurements are consistent with that ordering. Two coordinated controls on a backbone reach this regime: ETF-AM acts on the training-to-inference clearance, and CASA on its dimensionality. Our method achieves the best performance on CIFAR100-LT, ImageNet-LT, Places-LT, and iNaturalist-2018.

## Acknowledgement

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-00573160), the Institute of Information & Communications Technology Planning & Evaluation (IITP) - Information Technology Research Center (ITRC) grant funded by the Korea government (MSIT) (IITP-2026-RS-2023-00259703), and the “Advanced GPU Utilization Support Program” funded by the Government of the Republic of Korea (Ministry of Science and ICT) (for H. Hwang and D. Kim); and by the Electronics and Telecommunications Research Institute (ETRI) grant funded by ICT R&D program of MSIT/IITP [MSIT, No. RS-2025-02304183, Development of Optimization Code Conversion Technology for Heterogeneous AI Semiconductor-Based Large-Scale Models] (for C. Hur).

## References

[1] Shaden Alshammari, Yu-Xiong Wang, Deva Ramanan, and Shu Kong. Long-tailed recognition via weight balancing. In CVPR, pages 6897–6907, 2022. 2, 5, 6

[2] Barry C. Arnold. Pareto distribution. Wiley StatsRef: Statistics Reference Online, pages 1–10, 2014. 5

[3] Yoshua Bengio, Nicholas Leonard, and Aaron Courville.´ Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013. 4

[4] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced datasets with labeldistribution-aware margin loss. NeurIPS, 32, 2019. 1, 2, 4, 5, 6

[5] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. Class-balanced loss based on effective number of samples. In CVPR, pages 9268–9277, 2019. 1, 2, 6

[6] Hien Dang, Tho Tran, Tan Minh Nguyen, and Nhat Ho. Neural collapse for cross-entropy class-imbalanced learning with unconstrained ReLU features model. In ICML, pages 10017– 10040, 2024. 1, 2, 3

[7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, pages 248–255, 2009. 5

[8] Cong Fang, Hangfeng He, Qi Long, and Weijie J. Su. Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training. PNAS, 118(43): e2103091118, 2021. 1, 2

[9] Ronald A. Fisher. The use of multiple measurements in taxonomic problems. Annals ofEugenics, 7(2):179–188, 1936. 5

[10] Tomer Galanti, Andras Gy ´ orgy, and Marcus Hutter. General-¨ ization bounds for few-shot transfer learning with pretrained classifiers. arXiv preprint arXiv:2212.12532, 2022. 3, 8

[11] Tomer Galanti, Andras Gy´ orgy, and Marcus Hutter. On the¨ role of neural collapse in transfer learning. In International Conference on Learning Representations, 2022. 1, 3, 8

[12] Peifeng Gao, Qianqian Xu, Peisong Wen, Huiyang Shao, Zhiyong Yang, and Qingming Huang. A study of neural collapse phenomenon: Grassmannian frame, symmetry and generalization. arXiv preprint arXiv:2304.08914, 2023. 2

[13] Zahra Gharaee, ZeMing Gong, Nicholas Pellegrino, et al. A step towards worldwide biodiversity assessment: The BIOSCAN-1m insect dataset. NeurIPS, 36:43593–43619, 2023. 1

[14] Dandan Guo, Zhuo Li, He Zhao, Mingyuan Zhou, and Hongyuan Zha. Learning to re-weight examples with optimal transport for imbalanced classification. NeurIPS, 35: 25517–25530, 2022. 2

[15] Naoya Hasegawa and Issei Sato. Exploring weight balancing on long-tailed recognition problem. In ICLR, 2024. 1, 2, 5, 6

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016. 5

[17] Youngkyu Hong, Seungju Han, Kwanghee Choi, Seokjun Seo, Beomsu Kim, and Buru Chang. Disentangling label distribution for long-tailed visual recognition. In CVPR, pages 6626–6636, 2021. 1, 6

[18] Like Hui, Mikhail Belkin, and Preetum Nakkiran. Limitations of neural collapse for understanding generalization in deep learning. arXiv preprint arXiv:2202.08384, 2022. 1

[19] Jiachen Jiang, Jinxin Zhou, Peng Wang, Qing Qu, Dustin Mixon, Chong You, and Zhihui Zhu. Generalized neural collapse for a large number of classes. In International Conference on Machine Learning (ICML), 2024. 2

[20] Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, and Yannis Kalantidis. Decou pling representation and classifier for long-tailed recogni tion. In ICLR, 2020. 6

[21] Haeyong Kang, Thang Vu, and Chang D. Yoo. Learning imbalanced datasets with maximum margin loss. In ICIP, pages 1269–1273, 2021. 2

[22] Byungju Kim and Junmo Kim. Adjusting decision boundary for class imbalanced learning. IEEE Access, 8:81674–81685, 2020. 5

[23] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. 5

[24] Solomon Kullback and Richard A. Leibler. On information and sufficiency. The Annals of Mathematical Statistics, 22 (1):79–86, 1951. 5, 4

[25] Yifan Lan, Cai xin, Jun Cheng, and Shan Tan. Mixture of balanced information bottlenecks for long-tailed visual recognition. Transactions on Machine Learning Research, 2025. 2, 6

[26] Mengke Li, Yiu-ming Cheung, and Yang Lu. Long-tailed visual recognition via Gaussian clouded logit adjustment. In CVPR, pages 6929–6938, 2022. 1, 6

[27] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In´ ICCV, pages 2980–2988, 2017. 1

[28] Weiyang Liu, Longhui Yu, Adrian Weller, and Bernhard Scholkopf. Generalizing and decoupling neural collapse via¨

hyperspherical uniformity gap. In International Conference on Learning Representations (ICLR), 2023. 2

[29] Xuantong Liu, Jianfeng Zhang, Tianyang Hu, He Cao, Yuan Yao, and Lujia Pan. Inducing neural collapse in deep longtailed learning. In AISTATS, pages 11534–11544, 2023. 1, 2, 6

[30] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X. Yu. Large-scale long-tailed recognition in an open world. In CVPR, pages 2537–2546, 2019. 5, 6

[31] Alexander Long, Wei Yin, Thalaiyasingam Ajanthan, et al. Retrieval augmented classification for long-tail visual recognition, 2022. 1

[32] Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar. Long-tail learning via logit adjustment. In ICLR, 2021. 2

[33] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. In ICLR Workshop, 2013. 5

[34] Tai-Yu Pan, Cheng Zhang, Yandong Li, et al. On model calibration for long-tailed object detection and instance segmentation, 2021. 1

[35] Vardan Papyan, X. Y. Han, and David L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. CoRR, abs/2008.08186, 2020. 1, 2

[36] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 5

[37] Gao Peifeng, Qianqian Xu, Peisong Wen, Zhiyong Yang, Huiyang Shao, and Qingming Huang. Feature directions matter: Long-tailed learning via rotated balanced representation. In ICML, 2023. 6

[38] Wujian Peng, Zejia Weng, Hengduo Li, Zuxuan Wu, and Yu-Gang Jiang. BMB: Balanced memory bank for long-tailed semi-supervised learning. IEEE TMM, 2025. 1

[39] Harsh Rangwani, Sumukh K. Aithal, Mayank Mishra, et al. Escaping saddle points for effective generalization on classimbalanced data. NeurIPS, 35:22791–22805, 2022. 2, 6

[40] Jiawei Ren, Cunjun Yu, Xiao Ma, Haiyu Zhao, and Shuai Yi. Balanced meta-softmax for long-tailed visual recognition. NeurIPS, 33:4175–4186, 2020. 1, 2, 6

[41] Jiang-Xin Shi, Tong Wei, Yuke Xiang, and Yu-Feng Li. How re-sampling helps for long-tail learning?, 2023. 1

[42] Suraj Srinivas, Akshayvarun Subramanya, and R. Venkatesh Babu. Training sparse neural networks. In CVPR Workshops, pages 138–145, 2017. 4

[43] Jingru Tan, Xin Lu, Gang Zhang, Changqing Yin, and Quanquan Li. Equalization loss v2: A new gradient balance approach for long-tailed object detection. In CVPR, pages 1685–1694, 2021. 2

[44] Christos Thrampoulidis, Ganesh Ramachandra Kini, Vala Vakilian, and Tina Behnia. Imbalance trouble: Revisiting neural-collapse geometry. NeurIPS, 35, 2022. 1, 2

[45] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and

Serge Belongie. The inaturalist species classification and detection dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8769–8778, 2018. 5

[46] Jiangong Wang, Xiao Wang, Tianyu Shen, et al. Parallel vision for long-tail regularization: Initial results from IVFC autonomous driving testing. IEEE T-IV, 7(2):286–299, 2022. 1

[47] Jinping Wang, Zhiqiang Gao, and Zhiwu Xie. Space align ment matters: The missing piece for inducing neural collapse in long-tailed learning. arXiv preprint arXiv:2512.07844, 2025. 2

[48] Tong Wu, Qingqiu Huang, Ziwei Liu, Yu Wang, and Dahua Lin. Distribution-balanced loss for multi-label classification in long-tailed datasets. In ECCV, 2020. 2

[49] Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and´ Kaiming He. Aggregated residual transformations for deep neural networks. In CVPR, pages 1492–1500, 2017. 5

[50] Hongren Yan, Yuhua Qian, Furong Peng, Jiachen Luo, Zheqing Zhu, and Feijiang Li. Neural collapse to multiple cen ters for imbalanced data. NeurIPS, 37, 2024. 1, 2, 3

[51] Yibo Yang, Shixiang Chen, Xiangtai Li, Liang Xie, Zhouchen Lin, and Dacheng Tao. Inducing neural collapse in imbalanced learning: Do we really need a learnable classifier at the end of deep neural network? NeurIPS, 35:37991– 38002, 2022. 1, 2, 3

[52] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. CutMix: Regularization strategy to train strong classifiers with localizable features, 2019. 1

[53] Enhao Zhang, Chuanxing Geng, Chaohua Li, and Songcan Chen. Dynamic learnable logit adjustment for long-tailed visual recognition. IEEE TCSVT, 2024. 1, 2, 6

[54] Enhao Zhang, Chaohua Li, Chuanxing Geng, and Songcan Chen. All-around neural collapse for imbalanced classification. arXiv preprint arXiv:2408.07253, 2024. 2

[55] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. Mixup: Beyond empirical risk minimiza tion, 2018. 1

[56] Songyang Zhang, Zeming Li, Shipeng Yan, Xuming He, and Jian Sun. Distribution alignment: A unified framework for long-tail visual recognition. In CVPR, pages 2361–2370, 2021. 1, 6

[57] Yifan Zhang, Bryan Hooi, Lanqing Hong, and Jiashi Feng. Self-supervised aggregation of diverse experts for testagnostic long-tailed recognition. NeurIPS, 35:34077–34090, 2022. 5

[58] Qihao Zhao, Chen Jiang, Wei Hu, Fan Zhang, and Jun Liu. MDCS: More diverse experts with consistency selfdistillation for long-tailed recognition. In ICCV, pages 11597–11608, 2023. 5

[59] Yan Zhao, Weicong Chen, Xu Tan, Kai Huang, and Jihong Zhu. Adaptive logit adjustment loss for long-tailed visual recognition. In AAAI, pages 3472–3480, 2022. 1, 2, 6

[60] Qiaoyu Zheng, Weike Zhao, Chaoyi Wu, Xiaoman Zhang, Lisong Dai, Hengyu Guan, Yuehua Li, Ya Zhang, Yanfeng

Wang, and Weidi Xie. Large-scale long-tailed disease diagnosis on radiology images. Nature Communications, 15(1): 10147, 2024. 1

[61] Zhisheng Zhong, Jiequan Cui, Shu Liu, and Jiaya Jia. Improving calibration for long-tailed recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 6

[62] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE TPAMI, 40(6):1452–1464, 2017. 5

[63] Jianggang Zhu, Zheng Wang, Jingjing Chen, Yi-Ping Phoebe Chen, and Yu-Gang Jiang. Balanced contrastive learning for long-tailed visual recognition. In CVPR, pages 6908–6917, 2022. 2

# Full Collapse Is Suboptimal: Controlled Partial Neural Collapse for Long-Tailed Recognition

Supplementary Material

## A. Finite-Sample Model for the Optimality of Partial Collapse

This section supports Theorem 1 and Corollary 1 of the main paper. We first show that the UFM, the standard abstraction for neural collapse, drives every class to full collapse at its training optimum (Sec. A.1). The UFM contains no sampling process and no learned map from inputs to features, so it has no training-to-test gap and cannot express how the risk-optimal geometry depends on the class size. We therefore analyze the finite-sample classconditional model introduced in the main paper (Sec. A.2) and prove that, for any class satisfying the sensitivity condition of $\operatorname { E q . } \left( 3 \right)$ , the within-class spread that minimizes perclass risk is strictly positive $( \mathrm { S e c . } \mathrm { A . } 3 )$ . The same section resolves that condition into a threshold on the training count and, under an additional monotonicity condition on ϕ, orders the optimal spread across classes.

## A.1. The Training Optimum Under the UFM Is Full Collapse

We use the UFM, which treats last-layer features as free variables [6, 51]. The features are $h _ { c , i } = \mu _ { c } + \varepsilon _ { c , i }$ for class $c = 1 , \ldots , C$ and sample $i = 1 , \dots , n _ { c } ,$ with counts $n _ { 1 } \geq \cdots \geq n _ { C }$ , class mean $\mu _ { c } .$ , zero-mean deviation $\varepsilon _ { c , i } ,$ and within-class spread $\begin{array} { r } { s _ { c } ~ = ~ \frac { 1 } { n _ { c } } \sum _ { i = 1 } ^ { n _ { c } } \| \varepsilon _ { c , i } \| ^ { 2 } } \end{array}$ as in the main paper. The classifier provides a maximal-separation inter-class geometry: a simplex ETF (with $\lVert \boldsymbol { w } _ { c } \rVert = 1$ and $\langle w _ { c } , w _ { c ^ { \prime } } \rangle = - \frac { 1 } { C - 1 }$ for $c \neq c ^ { \prime } )$ when $C \leq d + 1$ , which gives a positive separation margin $\gamma > 0$ between distinct class means. The proof uses this margin, not a specific frame. A per-class budget with some finite radius R couples the mean norm and the spread, $\| \mu _ { c } \| ^ { 2 } + s _ { c } \leq R ^ { 2 }$ . With the margin cross-entropy of the main paper, the class-c loss decreases as $\mu _ { c }$ aligns with $w _ { c }$ and as the true-class logit $w _ { c } ^ { \top } \mu _ { c } = \| \mu _ { c } \|$ grows. The budget then forces $s _ { c } $ 0: every unit of spread is spent at the cost of mean norm, and the loss prefers mean norm. The adaptive margin $\varrho _ { c }$ shifts the loss scale but not this conclusion. Hence, the UFM training optimum is full collapse, $s _ { c } = 0$ , for every class and any imbalance profile. The result is imbalance-agnostic because the inter-class target is held fixed and the argument acts only on the within-class spread $s _ { c } .$ The imbalance reshapes the inter-class geometry studied elsewhere [6, 8, 44], but enters our analysis only through the test-risk dependence on $n _ { c } ,$ not the training optimum.

## A.2. The Finite-Sample Class-Conditional Model

The optimum above concerns the training objective on free features; it says nothing about test performance, because the UFM has no data-generating distribution. To reason about generalization, the main paper adopts a minimal population model for one class c with training count $n _ { c } ,$ in which the signed test margin of a random class-c sample is

$$
\mathcal { M } _ { c } ( s _ { c } ) = ( \gamma - a s _ { c } ) + Z _ { \mathrm { t e s t } } + Z _ { \mathrm { e s t } } ( s _ { c } , n _ { c } ) ,\tag{21}
$$

with $a > 0$ , irreducible test noise $Z _ { \mathrm { t e s t } } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ with $\sigma ^ { 2 } > 0 ,$ , and an independent Gaussian finite-sample term $Z _ { \mathrm { e s t } }$ . The sample is correctly classified when $\mathcal { M } _ { c } ( s _ { c } ) > 0$ The term $- a s _ { c }$ is the separation cost: a larger within-class spread moves features closer to the boundary. The term $Z _ { \mathrm { e s t } }$ is the finite-sample cost of the learned feature map, whose variance is the object of Assumption 1 and Eq. (3). Relaxing the spread reduces variance at rate $- V _ { s } ( 0 , n _ { c } )$ while paying separation at rate $^ { a , }$ and the trade favors a positive spread when the former dominates. The modeling content of Assumption 1 lies in the dependence of V on the training count $n _ { c } .$ The finite-sample cost of estimating a class-c feature map grows as $n _ { c }$ shrinks. This dependence is what the UFM lacks and what makes the analysis sensitive to imbalance.

Galanti et al. [11] establish within-class variability as a key determinant of finite-sample error, with estimates that degrade as the number of available samples shrinks. Hui et al. [18] observe that train-time variability collapse and test accuracy can be anti-correlated, indicating that driving the spread to zero degrades the learned feature map rather than improving it; this is what $\phi ^ { \prime } ( 0 ) < 0$ encodes.

Two simplifications are built into Eq. (21). First, the model identifies the test-time spread with the training spread: the same $s _ { c }$ enters both the separation cost and the estimation variance, which assumes the learned feature map reproduces its training spread at test time. Second, classes are decoupled, and $a , \gamma , \sigma ,$ , and ϕ are shared across classes; the class count $n _ { c }$ is the varying model parameter in the threshold result. The separation cost $- a s _ { c }$ therefore prices only a class’s own interference toward the boundary, not that of its neighbors. Both simplifications keep the risk separable by class, which makes the closed-form analysis below possible.

## A.3. Proofs of Theorem 1 and Corollary 1

The balanced test risk weights each class equally, so it suffices to minimize the per-class error $\begin{array} { r l } { \mathcal { R } _ { c } ( s _ { c } ) } & { { } = } \end{array}$ $\mathrm { P r } [ \mathcal { M } _ { c } ( s _ { c } ) \leq 0 ]$ . We prove Theorem 1 of the main paper, then Corollary 1, which resolves the sensitivity condition into a threshold on the training count.

Theorem A.1 (Theorem 1, suboptimality of full collapse). Let V be as in Assumption 1. For any class c satisfying $E q . \ ( 3 )$ , the per-class risk on $[ 0 , \gamma / a ]$ attains its global minimum in the interior. Writing $s _ { c } ^ { \star } f o r$ the smallest global minimizer,

$$
s _ { c } ^ { \star } > 0 \quad a n d \quad \mathcal { R } _ { c } ( 0 ) > \mathcal { R } _ { c } ( s _ { c } ^ { \star } ) .\tag{22}
$$

Proof of Theorem 1. Since $Z _ { \mathrm { t e s t } }$ and $Z _ { \mathrm { e s t } }$ are independent zero-mean Gaussians, $\mathcal { M } _ { c } ( s )$ is Gaussian with mean $\gamma - a s$ and variance $D ( s ) = \sigma ^ { 2 } + V ( s , n _ { c } )$ . With Φ the standard Gaussian CDF,

$$
\mathcal { R } _ { c } ( s ) = \Phi ( - g ( s ) ) , \qquad g ( s ) = \frac { \gamma - a s } { \sqrt { D ( s ) } } ,\tag{23}
$$

and since Φ is strictly increasing, minimizing $\mathcal { R } _ { c }$ on $[ 0 , \gamma / a ]$ is equivalent to maximizing $g .$

By $\sigma ^ { 2 } > 0$ , Assumption 1 gives $0 < D ( s ) = \sigma ^ { 2 } +$ $V ( s , n _ { c } ) ~ < ~ \infty$ throughout the compact interval $[ 0 , \gamma / a ]$ Hence g is continuous on that interval and differentiable in its interior. Differentiating,

$$
g ^ { \prime } ( s ) = - \frac { 2 a D ( s ) + ( \gamma - a s ) V _ { s } ( s , n _ { c } ) } { 2 D ( s ) ^ { 3 / 2 } } .\tag{24}
$$

The denominator is positive, so at $s = 0 ,$

$$
g ^ { \prime } ( 0 ) > 0 \Longleftrightarrow - V _ { s } ( 0 , n _ { c } ) > \frac { 2 a \left[ \sigma ^ { 2 } + V ( 0 , n _ { c } ) \right] } { \gamma } ,\tag{25}
$$

which is exactly Eq. (3). Let class c satisfy it.

At the two endpoints, $g ( 0 ) \ = \ \gamma / \sqrt { D ( 0 ) } \ > \ 0$ , while $g ( \gamma / a ) ~ = ~ 0$ Since $g ^ { \prime } ( 0 ) ~ > ~ 0 .$ , there is $\delta \ > \ 0$ with $g ( \delta ) \ > \ g ( 0 )$ Since g is continuous on the compact interval, it attains a maximum, and that value exceeds both endpoints. The maximum therefore lies in $( 0 , \gamma / a )$ . Hence g has an interior global maximizer, equivalently $\mathcal { R } _ { c }$ has an interior global minimizer.

Let $\begin{array} { r } { \mathcal { S } ( n _ { c } ) = \arg \operatorname* { m a x } _ { s \in [ 0 , \gamma / a ] } g ( s ) } \end{array}$ . The preceding argument shows that $S ( n _ { c } )$ is nonempty and contained in $( 0 , \gamma / a )$ . It is a closed subset of a compact interval, so it has a smallest element; define $s _ { c } ^ { \star }$ to be that element. Since max<sub>s</sub> $g ( s ) > g ( 0 )$

$$
\begin{array} { c } { s _ { c } ^ { \star } > 0 \quad \mathrm { a n d } } \\ { \mathcal { R } _ { c } ( s _ { c } ^ { \star } ) = \Phi ( - g ( s _ { c } ^ { \star } ) ) < \Phi ( - g ( 0 ) ) = \mathcal { R } _ { c } ( 0 ) . } \end{array}\tag{26}
$$

Full collapse is therefore strictly suboptimal for any class satisfying Eq. (3). The proof of Theorem 1 is finished.

Theorem 1 is a per-class existence statement. It applies to whichever classes satisfy Eq. (3) and is silent about the rest. The assumed $1 / n _ { c }$ scaling turns this condition into an explicit threshold on the training count.

Corollary A.1 (Corollary 1, scarcity-dependent spread). Let $n ^ { \star }$ be the sample count at which Eq. (3) changes sign. Then Eq. (3) holds if and only $i f 0 < n _ { c } < n ^ { \star }$ . If, in addition, $\phi ^ { \prime } ( s ) < 0$ for every $s \in ( 0 , \gamma / a )$ , then for any two classes $c , c ^ { \prime }$ with $n _ { c ^ { \prime } } < n _ { c } < n ^ { \star }$

$$
s _ { c } ^ { \star } < s _ { c ^ { \prime } } ^ { \star } .\tag{27}
$$

Proof of Corollary 1. Threshold. By Assumption 1, $V ( 0 , n _ { c } ) = \phi ( 0 ) / n _ { c }$ and $V _ { s } ( 0 , n _ { c } ) = \phi ^ { \prime } ( 0 ) / n _ { c } ,$ so Eq. (3) reads

$$
- \frac { \phi ^ { \prime } ( 0 ) } { n _ { c } } > \frac { 2 a } { \gamma } \left[ \sigma ^ { 2 } + \frac { \phi ( 0 ) } { n _ { c } } \right] .\tag{28}
$$

Multiplying by $\gamma n _ { c } > 0$ and rearranging gives 2 $a \sigma ^ { 2 } n _ { c } <$ $- \gamma \phi ^ { \prime } ( 0 ) - 2 a \phi ( 0 )$ , i.e. $0 < n _ { c } < n ^ { \star }$ with

$$
n ^ { \star } = \frac { - \gamma \phi ^ { \prime } ( 0 ) - 2 a \phi ( 0 ) } { 2 a \sigma ^ { 2 } } .\tag{29}
$$

Ordering. Assume in addition $\phi ^ { \prime } ( s ) < 0$ on $( 0 , \gamma / a )$ Since $g \geq 0 ;$ , maximizing g is equivalent to maximizing

$$
F ( s , n ) = g ( s ) ^ { 2 } = \frac { n ( \gamma - a s ) ^ { 2 } } { n \sigma ^ { 2 } + \phi ( s ) } .\tag{30}
$$

For n $< n ^ { \star }$ , Theorem 1 places every maximizer in $( 0 , \gamma / a )$ where $F \mathrm { ~  ~ { ~ > ~ } ~ } 0 ;$ on this open interval set $\begin{array} { r l } { \Lambda ( s , n ) } & { { } = } \end{array}$ log $F ( s , n )$ . For $s > s ^ { \prime }$

$$
\begin{array} { c } { { \displaystyle \displaystyle \frac { \partial } { \partial n } \big [ \Lambda ( s , n ) - \Lambda ( s ^ { \prime } , n ) \big ] = } } \\ { { \displaystyle \sigma ^ { 2 } \bigg [ \frac { 1 } { n \sigma ^ { 2 } + \phi ( s ^ { \prime } ) } - \frac { 1 } { n \sigma ^ { 2 } + \phi ( s ) } \bigg ] < 0 , } } \end{array}\tag{31}
$$

since $\phi ( s ) < \phi ( s ^ { \prime } )$ and $\sigma ^ { 2 } > 0$ . Thus Λ has strictly decreasing differences in $( s , n )$

Let $n _ { c ^ { \prime } } < n _ { c } < n ^ { \star }$ and take any $s \in \mathcal { S } ( n _ { c } )$ and $s ^ { \prime } \in$ $\boldsymbol { S } ( \boldsymbol { n } _ { c ^ { \prime } } )$ . If $s > s ^ { \prime }$ , optimality gives $\Lambda ( s , n _ { c } ) - \Lambda ( s ^ { \prime } , n _ { c } ) \geq 0$ and $\Lambda ( s , n _ { c ^ { \prime } } ) - \Lambda ( s ^ { \prime } , n _ { c ^ { \prime } } ) \ : \leq \ : 0 .$ , contradicting strictly decreasing differences. Hence $s \leq s ^ { \prime }$ for every such pair.

Both maximizers are interior, so each satisfies $- 2 a / ( \gamma -$ $~ a s ) = \phi ^ { \prime } ( s ) / ( n \sigma ^ { 2 } + \phi ( s ) )$ . The left side does not depend on n while the right side is strictly monotone in n whenever $\phi ^ { \prime } ( s ) < 0$ , so no point is optimal for both counts. Therefore $s _ { c } ^ { \star } < s _ { c ^ { \prime } } ^ { \star }$ . The proof of Corollary 1 is finished. □

Regarding the condition. The threshold and ordering parts have different roles. The threshold identifies the classes for which full collapse is suboptimal. The ordering is the part the design uses, since ETF-AM follows the same monotone ranking in $n _ { c }$ . It is also the part we measure (Sec. F.1), while $n ^ { \star }$ depends on constants the model leaves unspecified. The threshold uses the sign of $\phi ^ { \prime }$ at zero, whereas the ordering requires it on the whole interval $( 0 , \gamma / a )$

Scope of the inter-class target. The inter-class target is the simplex ETF of Sec. 3 when $C \leq d + 1$ . For $C > d + 1$ (e.g., iNaturalist-2018 with $C \ = \ 8 , 1 4 2 \ > \ d \ = \ 2 , 0 4 8 )$ no simplex ETF exists, and we learn the classifier instead. The proof concerns the within-class spread $s _ { c }$ and holds unchanged for either inter-class target, since it uses only the base margin $\gamma > 0$

## B. Reference Scale Determination Under ETF

In this section, we elaborate on the derivation of the reference scale under the ETF constraint as mentioned in Sec. 3.2.1 of the main paper.

Under the ETF constraint, all class weight vectors $w _ { c }$ lie on the unit hypersphere, and their pairwise inner products satisfy:

$$
\langle w _ { c } , w _ { c ^ { \prime } } \rangle = - \frac { 1 } { C - 1 } , \quad \mathrm { f o r } c \neq c ^ { \prime } ,\tag{32}
$$

where C denotes the total number of classes.

Based on this property, the squared Euclidean distance between any two unit-norm vectors can be derived as follows:

$$
\| w _ { c } - w _ { c ^ { \prime } } \| ^ { 2 } = 1 + 1 - 2 \Bigl ( - \frac { 1 } { C - 1 } \Bigr ) = 2 + \frac { 2 } { C - 1 } .\tag{33}
$$

Thus, the Euclidean distance between any two distinct class weight vectors is given by:

$$
\| w _ { c } - w _ { c ^ { \prime } } \| = \sqrt { 2 + \frac { 2 } { C - 1 } } .\tag{34}
$$

Assuming that the decision boundary between two class centers is placed at the midpoint, the effective margin (the distance from a class center to the boundary) is:

$$
\begin{array} { r } { \mathrm { { M a r g i n } } = \frac { 1 } { 2 } \sqrt { 2 + \frac { 2 } { C - 1 } } . } \end{array}\tag{35}
$$

For a sufficiently large C, the term $\frac { 2 } { C - 1 }$ becomes negligible, and the margin converges to:

$$
\begin{array} { r } { \mathrm { M a r g i n } = { \frac { 1 } { 2 } } \sqrt { 2 + { \frac { 2 } { C - 1 } } }  { \frac { 1 } { 2 } } \sqrt { 2 } \approx 0 . 7 0 7 1 . } \end{array}\tag{36}
$$

This derivation shows that Γ asymptotically approaches $\textstyle { \frac { \sqrt { 2 } } { 2 } }$ , ensuring that the margin adheres to the intrinsic geometric constraints imposed by the ETF configuration.

## C. Clearance Derivation Under the ETF Constraint

This section derives the training-to-inference clearance induced by the adaptive margin $\varrho _ { y } .$ , as used in Sec. 3.2 of the main paper.

Two parallel hyperplanes. With the logit $z _ { c } = w _ { c } ^ { \top } u + b _ { c }$ and the masked feature $u = M \odot h$ , the inference rule for a true class y against a class k is $z _ { y } - z _ { k } > 0$ as:

$$
( w _ { y } - w _ { k } ) ^ { \top } u + ( b _ { y } - b _ { k } ) > 0 ,\tag{37}
$$

whereas subtracting $\varrho _ { y }$ from the true-class logit makes the training condition $z _ { y } - \varrho _ { y } > z _ { k }$ , i.e.

$$
( w _ { y } - w _ { k } ) ^ { \top } u + ( b _ { y } - b _ { k } ) - \varrho _ { y } > 0 .\tag{38}
$$

The two hyperplanes share the normal vector $w _ { y } - w _ { k }$ and are therefore parallel; they differ only in their offsets. The Euclidean distance between them is

$$
\Delta _ { y k } = \frac { \varrho _ { y } } { \| w _ { y } - w _ { k } \| } .\tag{39}
$$

We call $\Delta _ { y k }$ the clearance: a feature meeting Eq. (38) sits at least $\Delta _ { y k }$ from the inference boundary of Eq. (37) once the margin is removed.

Independence from the competitor. Under the ETF constraint, $\| w _ { y } - w _ { k } \|$ is the same for every pair (Eq. (14)), and by Eq. (15) it equals 2Γ. Hence

$$
\Delta _ { y k } = \Delta _ { y } = \frac { \varrho _ { y } } { 2 \Gamma } ,\tag{40}
$$

which depends solely on the true class. On a learnable classifier head, $\| w _ { y } - w _ { k } \|$ varies by pair and over training, so the same logit margin $\varrho _ { y }$ maps to a clearance that is neither constant across competitors nor stable during optimization.

Clearance under gating. Both hyperplanes in Eqs. (37) and (38) are defined in u-space, and the mask enters only through $u = M \odot h$ . The normal is $w _ { y } - w _ { k }$ , whose length the ETF fixes at 2Γ irrespective of M, so the clearance of Eq. (40) is unchanged by gating: the mask selects which coordinates carry the feature, not the classifier geometry from which $\Delta _ { y }$ is derived.

Scope of the clearance. $\Delta _ { y }$ is a nominal training-toinference buffer. It bounds the boundary-normal variation a feature can absorb between the two rules, but it does not set the learned cloud radius or guarantee a particular withinclass spread.

## D. Detailed Formulation of Evaluation Metrics

## D.1. Fisher Discriminant Ratio

Given N samples $x _ { i } \in \mathbb { R } ^ { D }$ and C classes, where $n _ { c }$ is the number of samples in class c and $\mu _ { c }$ is the mean feature of

Table 9. Hyperparameters used in experiments. Γ is a fixed value, except for iNaturalist2018, which cannot have an ETF structure.
<table><tr><td></td><td>CIFAR100-LT</td><td>ImageNet-LT</td><td>Places-LT</td><td>iNaturalist2018</td></tr><tr><td colspan="5">Data Settings</td></tr><tr><td># Classes</td><td>100</td><td>1000</td><td>365</td><td>8142</td></tr><tr><td>Batch Size</td><td>64</td><td>128</td><td>128</td><td>512</td></tr><tr><td colspan="5">Optimizer Parameters</td></tr><tr><td>Total Epochs</td><td>320</td><td>200</td><td>30</td><td>200</td></tr><tr><td>Optimizer</td><td>SGD</td><td>SGD</td><td>SGD</td><td>SGD</td></tr><tr><td>Momentum</td><td>0.9</td><td>0.9</td><td>0.9</td><td>0.9</td></tr><tr><td>Scheduler</td><td>Cos</td><td>MultiStep</td><td>ReduceLROnPlateau</td><td>Cos</td></tr><tr><td>Base LR</td><td>0.01</td><td>0.05</td><td>0.3</td><td>0.05</td></tr><tr><td>Weight Decay</td><td>0.005</td><td>0.00024</td><td>0.00024</td><td>0.0005</td></tr><tr><td> $\beta$  (Feature Reg.)</td><td>0.01</td><td>0.0001</td><td>0.00003</td><td>0.0001</td></tr><tr><td>LA Coef.</td><td>0.65</td><td>0.2</td><td>0.35</td><td>0.15</td></tr><tr><td>Γ (Reference Scale)</td><td>0.71</td><td>0.71</td><td>0.71</td><td>0.9</td></tr><tr><td colspan="5">Loss Hyperparameters</td></tr><tr><td>v (spike)</td><td>0.5</td><td>0.5</td><td>0.3</td><td>0.5</td></tr><tr><td> $\varpi \left( \mathrm { s l a b } \right)$ </td><td>0.1</td><td>0.1</td><td>0.3</td><td>0.1</td></tr></table>

Table 10. Extended ablation study on CIFAR100-LT: FDR on train and test sets, and Top-1 Accuracy for Many, Medium, Few, and overal Average classes.
<table><tr><td>Method</td><td>FDR (Tr)</td><td>FDR (Te)</td><td>Many</td><td>Med</td><td>Few</td><td> $\operatorname { A v g } .$ </td></tr><tr><td> $\vartheta = 0 . 5 , \varpi = 0 . 1$ </td><td> $9 . 9 4 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 8 \times 1 0 ^ { 2 }$ </td><td>74.71</td><td>55.77</td><td>31.65</td><td>54.73</td></tr><tr><td>w/o ETF-AM</td><td> $3 . 3 6 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 3 \times 1 0 ^ { 2 }$ </td><td>74.88</td><td>53.34</td><td>26.74</td><td>52.42</td></tr><tr><td>w/o Gating</td><td> $9 . 4 6 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 4 \times 1 0 ^ { 2 }$ </td><td>74.20</td><td>54.10</td><td>30.60</td><td>53.60</td></tr><tr><td> $\vartheta = 0 . 9 , \varpi = 0 . 1$ </td><td> $9 . 6 5 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 4 \times 1 0 ^ { 2 }$ </td><td>73.00</td><td>55.48</td><td>31.29</td><td>53.56</td></tr><tr><td> $\vartheta { = } 0 . 7 , \varpi { = } 0 . 1$ </td><td> $9 . 7 6 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 4 \times 1 0 ^ { 2 }$ </td><td>73.52</td><td>55.26</td><td>31.48</td><td>54.10</td></tr><tr><td> $\vartheta = 0 . 3 , \varpi = 0 . 1$ </td><td> $9 . 8 0 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 5 \times 1 0 ^ { 2 }$ </td><td>73.85</td><td>54.22</td><td>31.73</td><td>54.22</td></tr><tr><td> $\vartheta = 0 . 1 , \varpi = 0 . 1$ </td><td> $9 . 8 4 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 5 \times 1 0 ^ { 2 }$ </td><td>74.08</td><td>54.59</td><td>32.12</td><td>54.30</td></tr><tr><td> $\vartheta = 0 . 5 , \varpi = 0 . 9$ </td><td> $9 . 2 9 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 4 \times 1 0 ^ { 2 }$ </td><td>71.43</td><td>52.40</td><td>28.97</td><td>51.22</td></tr><tr><td> $\vartheta = 0 . 5 , \varpi = 0 . 7$ </td><td> $9 . 3 2 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 4 \times 1 0 ^ { 2 }$   $1 . 2 5 \times 1 0 ^ { 2 }$ </td><td>72.61</td><td>53.40</td><td>29.87</td><td>52.19</td></tr><tr><td> $\vartheta = 0 . 5 , \varpi = 0 . 5$ </td><td> $9 . 4 8 \times 1 0 ^ { 4 }$ </td><td></td><td>73.32</td><td>54.21</td><td>30.82</td><td>53.24</td></tr><tr><td> $\vartheta = 0 . 5 , \varpi = 0 . 3$ </td><td> $9 . 4 7 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 7 \times 1 0 ^ { 2 }$ </td><td>73.55</td><td>55.60</td><td>31.90</td><td>53.72</td></tr></table>

class $c ,$ we define the following:

$$
S _ { B } = \sum _ { c = 1 } ^ { C } n _ { c } ( \mu _ { c } - \mu ) ( \mu _ { c } - \mu ) ^ { \top } ,\tag{41}
$$

$$
S _ { W } = \sum _ { c = 1 } ^ { C } \sum _ { i \in I _ { c } } ( x _ { i } - \mu _ { c } ) ( x _ { i } - \mu _ { c } ) ^ { \top } ,\tag{42}
$$

where $\mu$ is the overall mean of all samples.

FDR is defined as the ratio of the trace of the betweenclass scatter to the trace of the within-class scatter:

$$
\mathrm { F D R } = \frac { \mathrm { t r } ( S _ { B } ) } { \mathrm { t r } ( S _ { W } ) } .\tag{43}
$$

A high FDR indicates that the features are more compact within each class (lower $S _ { W } )$ and more separated between classes (higher $S _ { B } ) _ { \ l }$ , which is a characteristic of effective NC. We use FDR to verify that inter-class separation is preserved, not to argue for uniform within-class compactness. Because tail classes are sample-scarce, the tail spread we deliberately retain contributes little to the overall $\mathrm { t r } ( S _ { W } )$

## D.2. Metrics for Mask Consistency

Given two mask distribution vectors $m _ { i }$ and $m _ { j } .$ , the cosine similarity is computed as:

$$
\mathrm { C o s S i m } ( m _ { i } , m _ { j } ) = \frac { m _ { i } \cdot m _ { j } } { \| m _ { i } \| \| m _ { j } \| } .\tag{44}
$$

The symmetric KL divergence is defined as the sum of the Kullback-Leibler divergences [24] in both directions:

$$
\mathrm { K L } _ { \mathrm { s y m } } ( m _ { i } , m _ { j } ) = \mathrm { K L } ( m _ { i } \| m _ { j } ) + \mathrm { K L } ( m _ { j } \| m _ { i } ) ,\tag{45}
$$

Table 11. Extended Ablation study on ImageNet-LT with ResNeXt50: FDR on train and test sets, and Top-1 Accuracy for Many, Medium, Few, and overall Average classes. Bold marks the best.
<table><tr><td>Method</td><td>FDR (Tr)</td><td>FDR (Te)</td><td>Many</td><td>Med</td><td>Few</td><td> $\operatorname { A v g } .$ </td></tr><tr><td> $\vartheta = 0 . 5 , \varpi = 0 . 1$  一</td><td> $9 . 1 9 \times 1 0 ^ { 2 }$ </td><td> $7 . 2 5 \times 1 0 ^ { 2 }$ </td><td>65.20</td><td>55.35</td><td>38.87</td><td>56.90</td></tr><tr><td>w/o ETF-AM</td><td> $5 . 2 0 \times 1 0 ^ { 2 }$ </td><td> $6 . 1 6 \times 1 0 ^ { 2 }$ </td><td>65.17</td><td>52.19</td><td>27.52</td><td>53.83</td></tr><tr><td>w/o Gating</td><td> $8 . 8 9 \times 1 0 ^ { 2 }$ </td><td> $7 . 0 2 \times 1 0 ^ { 2 }$ </td><td>64.54</td><td>52.87</td><td>30.73</td><td>54.34</td></tr><tr><td> $\vartheta = 0 . 9 , \varpi = 0 . 1$ </td><td> $8 . 4 2 \times 1 0 ^ { 2 }$ </td><td> $6 . 1 1 \times 1 0 ^ { 2 }$ </td><td>64.05</td><td>54.81</td><td>36.76</td><td>55.91</td></tr><tr><td> $\vartheta { = } 0 . 7 , \varpi { = } 0 . 1$ </td><td> $8 . 5 4 \times 1 0 ^ { 2 }$ </td><td> $6 . 4 2 \times 1 0 ^ { 2 }$ </td><td>62.25</td><td>53.63</td><td>34.33</td><td>54.32</td></tr><tr><td> $\vartheta = 0 . 3 , \varpi = 0 . 1$ </td><td> $8 . 6 4 \times 1 0 ^ { 2 }$ </td><td> $6 . 6 2 \times 1 0 ^ { 2 }$ </td><td>63.24</td><td>54.73</td><td>32.91</td><td>55.03</td></tr><tr><td>v=0.1, α=0.1</td><td> $8 . 9 7 \times 1 0 ^ { 2 }$ </td><td> $6 . 8 5 \times 1 0 ^ { 2 }$ </td><td>64.98</td><td>55.12</td><td>38.71</td><td>56.68</td></tr><tr><td>v=0.5,α=0.9</td><td> $8 . 2 1 \times 1 0 ^ { 2 }$ </td><td> $6 . 5 3 \times 1 0 ^ { 2 }$ </td><td>60.33</td><td>53.29</td><td>32.35</td><td>53.15</td></tr><tr><td>v=0.5,α=0.7</td><td> $8 . 3 5 \times 1 0 ^ { 2 }$ </td><td> $6 . 6 2 \times 1 0 ^ { 2 }$ </td><td>62.48</td><td>53.78</td><td>33.01</td><td>54.30</td></tr><tr><td> $\vartheta = 0 . 5 , \varpi = 0 . 5$ </td><td> $8 . 5 0 \times 1 0 ^ { 2 }$ </td><td> $6 . 7 5 \times 1 0 ^ { 2 }$ </td><td>63.27</td><td>54.18</td><td>35.27</td><td>55.10</td></tr><tr><td> $\vartheta = 0 . 5 , \varpi = 0 . 3$ </td><td> $9 . 1 0 \times 1 0 ^ { 2 }$ </td><td> $7 . 1 8 \times 1 0 ^ { 2 }$ </td><td>64.52</td><td>55.20</td><td>38.96</td><td>56.52</td></tr></table>

![](images/81ad954b17a91cece9dd70e25681a14a496d7f60419e9c39db45a2c0c7ff2814.jpg)

![](images/d2b8146f6da7ef5cf93a767b3d3e53812cf72ad68fd63fab981a8ff9a63b9932.jpg)  
Figure 5. Hyperparameter (ϑ and ϖ) heatmap on CIFAR100-LT. (a) Average accuracy on the test set, (b) FDR score on the test set.

where, for two probability distributions P and $Q ,$

$$
\mathrm { K L } ( P \| Q ) = \sum _ { k } P ( k ) \log { \frac { P ( k ) } { Q ( k ) } } .\tag{46}
$$

These metrics are computed for pairs of samples within the same class (intra-class) and across different classes (inter-class) to evaluate the degree of similarity in the generated mask distributions.

## E. Training Settings

For our experimental environment, we use an Ubuntu 22.04 server with an Intel i9-10900X CPU, 64GB of RAM, and an NVIDIA 5090 GPU with CUDA 13.0, and an Ubuntu 22.04 server with two AMD EPYC 9654 CPUs, 512GB of RAM, and four NVIDIA PRO6000 Workstation Edition GPUs with CUDA 13.0. All experiments are implemented using PyTorch 2.10.0 with Python 3.13.

Detailed hyperparameters are provided in Tab. 9. For Places-LT $( \rho \ = \ 9 9 6 )$ , we set $( \vartheta , \varpi ) \ : = \ : ( 0 . 3 , 0 . 3 )$ . This setting activates more weights for the Few classes, thereby enhancing their feature representation and overall performance under extreme imbalance.

## F. Extended Experiments and Analysis

## F.1. Frequency Dependence of the Within-Class Spread

Corollary 1 gives a model-level threshold and, under the additional monotonicity condition, an ordering of the riskoptimal spreads by class frequency. The ordering predicts that the learned spread decreases as $n _ { c }$ increases. We test this descriptive implication using the backbone features. For a split S, let $N _ { c } ^ { S }$ denote the number of its class-c samples. We compute

$$
\bar { h } _ { c } ^ { S } = \frac { 1 } { N _ { c } ^ { S } } \sum _ { i \in { S : y _ { i } = c } } h _ { i } ,\tag{47}
$$

$$
s _ { c } ^ { S } = \frac { 1 } { N _ { c } ^ { S } } \sum _ { i \in { \cal S } : y _ { i } = c } \| h _ { i } - \bar { h } _ { c } ^ { S } \| _ { 2 } ^ { 2 } .\tag{48}
$$

For both the training and Test/Val. set, the predictor is the training count $n _ { c } ,$ not the number of samples in the evalua-

Table 12. Log–log OLS slope of class-wise within-class spread $s _ { c }$ against the training count $n _ { c } .$ All slopes are negative on both splits.
<table><tr><td>Dataset</td><td>Train</td><td> $\mathrm { T e s t } / \mathrm { V a l } .$ </td></tr><tr><td>CIFAR100-LT</td><td>-0.602</td><td>-0.249</td></tr><tr><td>iNaturalist-2018</td><td>-0.120</td><td> $- 0 . 1 2 4$ </td></tr></table>

tion split. Specifically, with $x _ { c } = \log n _ { c }$ and $y _ { c } ^ { S } = \log s _ { c } ^ { S } .$ the reported slope is the unweighted Ordinary Least Squares (OLS) coefficient

$$
\widehat { \beta } _ { S } = \frac { \sum _ { c \in \mathcal { C } _ { S } } ( x _ { c } - \bar { x } _ { S } ) ( y _ { c } ^ { S } - \bar { y } _ { S } ) } { \sum _ { c \in \mathcal { C } _ { S } } ( x _ { c } - \bar { x } _ { S } ) ^ { 2 } } ,\tag{49}
$$

where $\bar { x } _ { \mathcal { S } }$ and $\bar { y } _ { S }$ are the corresponding means over the set of classes $\mathcal { C } _ { S }$ . Thus, each class contributes one equally weighted observation. We use natural logarithms; the fit includes 100 classes on CIFAR100-LT and 8,142 classes on iNaturalist. As Tab. 12 shows, all four slopes are negative, and every split retains Many < Medium < Few.

We compare signs and orderings rather than magnitudes. The training and test splits are differently distributed. The model also identifies the training and test spread (Sec. A.2), a simplification we adopt for tractability. The iNaturalist slopes agree closely across splits, though they are smaller in magnitude than on CIFAR.

## F.2. Controlled Collapse Intervention

To examine the effect of stronger within-class compression, we add a penalty that contracts Few-class backbone features toward their running class means,

$$
\mathcal { L } _ { \mathrm { c o l } } = \frac { 1 } { \vert \mathcal { B } \vert } \sum _ { i \in \mathcal { B } } \mathbb { I } [ n _ { y _ { i } } \le 2 0 ] \Vert h _ { i } - \widetilde { \mu } _ { y _ { i } } \Vert _ { 2 } ^ { 2 } ,\tag{50}
$$

and train with $\mathcal { L } + \lambda \mathcal { L } _ { \mathrm { c o l } }$ , varying λ. Here B is the minibatch, and $\widetilde { \mu } _ { c }$ is a running mean of class c with momentum 0.9 initialized at the first batch mean containing the class. The penalty acts on the raw backbone feature h.

For every seed, each $\lambda > 0$ setting yields substantially lower normalized Few-class spread on the clean training set than $\lambda \ = \ 0 ,$ , although the training-set spread is not monotonic across the three nonzero values. On the test set, both normalized spread and post-LA Few-class accuracy decrease monotonically with λ for all three seeds. From $\lambda = 0 \mathrm { \ t o \ 0 . 3 } \mathrm { } .$ , normalized Few-class spread decreases by 64.5±17.8% on the clean training set and $1 9 . 7 { \pm } 1 . 1 \%$ on the test set. Over the same change, post-LA Few-class accuracy decreases by $3 . 6 9 { \pm } 1 . 3 5 $ points, while training balanced accuracy remains at least 99.6%. We therefore view this sweep as a one-sided stress test: increasing the contraction penalty from the λ = 0 baseline reduces test spread but degrades post-LA Few-class performance. The $\lambda = 0 . 3$ setting does not enforce exact $s _ { c } = 0$ , and the sweep does not identify an optimum or show that spread alone causes the accuracy change.

Table 13. Three-seed sweep of the Few-class contraction penalty with ResNet-34 on CIFAR100-LT. Each entry reports the mean±sample SD.
<table><tr><td> $\lambda$ </td><td>Train  $s _ { c } ( \times 1 0 ^ { - 3 } )$ </td><td> $\mathrm { T e s t } s _ { c }$ </td><td>Few Acc. (%)</td></tr><tr><td>0.0</td><td> $4 . 0 4 5 { \pm } 0 . 3 3$ </td><td> $0 . 3 3 2 5 { \pm } 0 . 0 0 3 3$ </td><td> $3 3 . 3 4 \pm 1 . 2 9$ </td></tr><tr><td>0.1</td><td> $0 . 9 9 1 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 2 9 2 5 { \pm } 0 . 0 0 4 8$ </td><td> $3 2 . 1 5 { \pm } 0 . 7 6$ </td></tr><tr><td>0.2</td><td> $0 . 8 5 5 { \pm } 0 . 0 3$ </td><td> $0 . 2 7 4 8 { \pm } 0 . 0 0 4 5$ </td><td> $3 0 . 6 2 { \pm } 1 . 4 5 $ </td></tr><tr><td>0.3</td><td> $1 . 4 4 0 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $0 . 2 6 7 0 { \scriptstyle \pm 0 . 0 0 1 4 }$ </td><td> $2 9 . 6 6 { \pm } 1 . 6 8 $ </td></tr></table>

## F.3. Extended Ablation Study

To further validate this trend on a larger-scale benchmark, we provide extended ablation results on CIFAR100-LT and ImageNet-LT. Tabs. 10 and 11 report FDR scores and top-1 accuracy for Many, Medium, and Few classes across various settings.

In both datasets, increasing either ϑ or ϖ reduces mask sparsity, resulting in denser mask outputs and reduced node specialization. This, in turn, leads to lower FDR scores and decreased overall classification accuracy, with the model’s behavior approaching performance without a gating layer.

For example, on ImageNet-LT, as ϑ increases from 0.5 to 0.9, FDR drops from $9 . 1 9 \times 1 0 ^ { 2 }$ to $8 . 4 2 \times 1 0 ^ { 2 }$ , and average accuracy decreases from 56.90% to 55.91%. Increasing ϖ from 0.1 to 0.9, while fixing ϑ, further decreases FDR to $8 . 2 1 \times 1 0 ^ { 2 }$ and lowers accuracy to 53.15%.

Fig. 5 illustrates the hyperparameter sweep for $\vartheta$ and $\varpi$ on the CIFAR100-LT dataset. The results are consistent with the extended ablation study. Based on our experiments, we recommend the following approach for tuning ϑ and ϖ on a new dataset. First, fix the slab variance $\varpi$ to a small value (e.g., 0.1). Second, search for the optimal ϑ in a range like [0.1, 0.3, 0.5, 0.7]. Finally, with the best ϑ fixed, perform a fine-grained search for ϖ around the initial value (e.g., [0.05, 0.1, 0.3]).

## F.4. Mask Mode at Training and Inference

Training uses hard masks while inference uses the soft sigmoid gate. To check the impact of gate design, we conduct all four combinations on CIFAR100-LT (Tab. 14). The learned geometry is set by the training mode alone: hard training raises train FDR from $6 . 9 7 \times 1 0 ^ { 4 } \mathrm { t o } 9 . 9 4 \times 1 0 ^ { 4 }$ and test FDR from $1 . 2 1 \times 1 0 ^ { 2 } \mathrm { t o } 1 . 2 8 \times 1 0 ^ { 2 }$ , while the inference mode leaves both unchanged. Accuracy depends on both, and our pairing is best overall; soft training costs more than two points, with the loss concentrated on Medium and Few.

Table 14. Mask mode at training and inference on CIFAR100-LT. FDR is determined by the training mode alone. Bold marks the best; underline the second-best.
<table><tr><td>Method</td><td>FDR (Tr)</td><td>FDR (Te)</td><td>Many</td><td>Med.</td><td>Few</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Soft / Soft</td><td> $6 . 9 7 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 1 \times 1 0 ^ { 2 }$ </td><td>77.53</td><td>52.03</td><td>25.10</td><td>52.35</td></tr><tr><td>Soft / Hard</td><td> $6 . 9 7 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 1 \times 1 0 ^ { 2 }$ </td><td>76.18</td><td>50.06</td><td>24.55</td><td>51.03</td></tr><tr><td>Hard / Hard</td><td> $9 . 9 4 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 8 \times 1 0 ^ { 2 }$ </td><td>75.34</td><td>52.01</td><td>29.45</td><td>53.41</td></tr><tr><td>Hard / Soft (Ours)</td><td> $9 . 9 4 \times 1 0 ^ { 4 }$ </td><td> $1 . 2 8 \times 1 0 ^ { 2 }$ </td><td>74.71</td><td>55.77</td><td>31.65</td><td>54.73</td></tr></table>

Table 15. Residual cosine similarity after removing the datasetmean mask, CIFAR100-LT.
<table><tr><td></td><td>Intra-class</td><td>Inter-class</td></tr><tr><td>Train</td><td>0.986</td><td>0.254</td></tr><tr><td>Test</td><td>0.383</td><td>0.015</td></tr></table>

## F.5. Residual Mask Structure

The mask similarities in Tab. 8 are inflated by a component shared across the entire dataset: the gate learns a common activation pattern before differentiating classes. To isolate the class-related part, we subtract the dataset-mean mask $\begin{array} { r } { \bar { m } \ = \ \frac { 1 } { N } \sum _ { i } m _ { i } } \end{array}$ and recompute cosine similarity on the residuals $m _ { i } - { \bar { m } }$

As Tab. 15 shows, the intra/inter gap is on both splits. The test-set values are much smaller: the gate is conditioned on the feature, so unseen samples inherit class structure only to the extent the feature map transfers it. The structure is thus a statistical property of the learned map, not a constraint enforced by the objective, and we claim only an empirically class-related residual structure.

## F.6. Qualitative Result

Fig. 6 shows a confusion matrix on the test set of CIFAR100-LT $( \rho = 1 0 0 )$ . Classes are sorted in descending order by the number of samples per class. In the head classes, it shows almost perfect results. In the tail classes, there are some notable misclassifications, such as ‘streetcar’ vs. ‘bus’ and ‘whale’ vs. ‘dolphin’, where the former belong to tail classes and the latter to head classes. This error is more likely due to semantic overlap within the class itself or in the background, rather than to the long-tailed property.

Failure Case Analysis. Fig. 7 analyzes two visually similar Few/Many pairs (‘streetcar’/‘bus’, 11/262 training samples; ‘whale’/‘dolphin’, 6/119), split into success and failure cases. For correctly classified samples, the Few-class masks are distinct from the Many counterpart (Fig. 7a), so the gate assigns them distinguishable node patterns; misclassified samples show high mask similarity to the Many class, indicating failure from visual overlap rather than long-tailed bias. Removing ETF-AM increases this similarity (Fig. 7b), so the adaptive margin supplies the separation the gate needs, consistent with the Few-class drop when ETF-AM is removed as in Sec. 4.3.

![](images/51082369f8d5ddb59334b5c27e2a07b58478f2e3b67a94d8117bcf59439efea3.jpg)  
Figure 6. Confusion matrix for CIFAR100-LT (ρ = 100). Classes are sorted in descending order by the number of training samples per class.

![](images/0fc48591468756f5979088e9741011b7aa4bc4acb9be7bf74094eda45bce8eb0.jpg)

![](images/ddf4f3498e382e3f9d1b89af51fc2774aff6ecc7bf51a17034c32bdef6f1c670.jpg)  
Figure 7. Generated mask similarity on the CIFAR100-LT test set for success and failure cases (‘streetcar’ (11 training samples) vs. ‘bus (262) and ‘whale’ (6) vs. ‘dolphin’ (119)): (a) the proposed method, (b) the proposed method w/o ETF-AM loss.