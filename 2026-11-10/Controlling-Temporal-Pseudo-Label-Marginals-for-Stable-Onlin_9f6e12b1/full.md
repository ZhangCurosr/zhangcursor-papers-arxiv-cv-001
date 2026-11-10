# Controlling Temporal Pseudo-Label Marginals for Stable Online Test-Time Adaptation

Rani Amar   
Faculty of Engineering   
Bar-Ilan University   
Ramat Gan, Israel   
rani.amar@biu.ac.il   
Ofir Lindenbaum   
Faculty of Engineering   
Bar-Ilan University   
Ramat Gan, Israel   
ofir.lindenbaum@biu.ac.il

## Abstract

Continual Test-Time Adaptation (CTTA) methods update models online using a stream of unlabeled data to overcome distribution shifts encountered at test time. However, CTTA is often unstable, as it can reinforce biases in early predictions and increasingly predict only a few classes, resulting in drift or even collapse. In this paper, we identify narrowing pseudo-label class usage as an indicator of CTTA instability and quantify this behavior. Most existing methods have attempted to address this issue by either selecting reliable adaptation samples or re-weighting the loss to encourage diverse pseudo-labels, but they do not explicitly enforce a class marginal over a temporal window. Instead, we propose Sinkhorn Class-Marginal Regularization (SCALAR), a simple plug-and-play module that controls the pseudolabel distribution over a short temporal window. SCALAR works by maintaining recent predictions in a buffer, balancing their class distribution via the Sinkhorn algorithm to match a specified target distribution, and using the balanced soft targets for adaptation. Without changing the model architecture, SCALAR maintains the per-step runtime and increases the peak allocated GPU memory by only 10 MB on ImageNet-C. Our method is particularly effective on skewed, non-stationary data with severe distribution shifts, where SCALAR significantly alleviates drift and collapse when combined with existing stabilization methods based on entropy or diversity. Code is available at https://github.com/r-isr/SCALAR.

## 1 Introduction

Continual test-time adaptation (CTTA) assumes a deployment setting where a model must continuously adapt at test time under non-stationary conditions.

In CTTA, models must adapt online to a stream of unlabeled test samples that may differ in distribution from the training data and may change over time, without access to the source data [32]. Typically, CTTA methods adopt a teacher–student framework where an exponential moving average (EMA) of the student parameters defines the teacher model, which generates pseudo-labels for each batch. The student is updated online using these pseudo-labels as supervision [32, 10, 34, 23, 22]. Other methods such as TENT [31] minimize an entropy objective to directly update the BatchNorm affine parameters of a single model.

In both settings, the adaptation signal is driven by the model’s own predictions. Small biases can cause the adaptation signal to focus on fewer classes even in approximately balanced data streams due to the self-training effect. This can cause the pseudo-label distribution to become less diverse and eventually collapse to a small subset of classes. The CTTA community has observed two failure modes: (1) error accumulation during test-time adaptation, ranging from drift (gradual accumulation of errors) to collapse (rapid accumulation of errors), and (2)forgetting of the model’s accuracy on clean data after adapting on corrupted data [32, 34, 15]. Existing CTTA methods attempt to mitigate temporal instabilities by modifying local model updates. These include filtering out potentially unreliable predictions [25, 26], promoting pseudo-label diversity [33], adding regularization to prevent the test-time model from deviating too far from the source model [25, 24], and reducing temporal correlation in adaptation batches using a memory bank [15, 34]. However, they do not explicitly project window-level pseudo-label class marginals toward a specified target.

We aim to control the temporal class marginals of the pseudo-labels, which form the supervision signal in CTTA. We propose Sinkhorn Class-Marginal Regularization (SCALAR), a lightweight intervention to mitigate the marginal contraction that can contribute to test-time collapse.

Concretely, SCALAR maintains a short first-in-first-out (FIFO) queue of recent teacher logits and balances their class marginals over this window using the Sinkhorn–Knopp algorithm. This generates soft labels by rescaling teacher predictions so their class marginals over the window are close to a user-specified target prior (uniform by default). This procedure approximates a constrained KL projection of teacher predictions to satisfy a desired temporal class marginal [28, 9].

SCALAR is not intended to recover the test-time class prior but to stabilize the supervision signal observed by the student. While providing the correct class prior can reduce corruption errors, SCALAR with a uniform prior still provides substantial stabilization even in skewed settings. However, SCALAR with a uniform prior can slightly hinder the model’s ability to specialize to target classes if the target distribution consists of only a subset of source classes and the model is already stably adapting to them.

We find SCALAR’s stabilization benefit is highly state-dependent: when class usage is stable, SCALAR often has a smaller effect. But as class usage contracts and errors increase, SCALAR can counteract this contraction. We evaluate this behavior across balanced and skewed streams, fixed and dynamic skew schedules, different adaptation speeds, both teacher–student and entropy minimization settings, and backbone architectures like ViT-B and ResNet-50.

Since SCALAR is a window-level intervention, it complements other methods that stabilize updates on a per-sample (e.g., filtering, loss weighting) or per-parameter level (regularization). We demonstrate SCALAR’s complementarity to the EATA method [25], which uses an entropy- and diversity-aware, history-dependent approach to select samples for updating. Despite adaptive sample selection, EATA does not explicitly constrain class marginals of the supervision signal. Applying SCALAR on top of EATA further reduces drift and mitigates collapse.

Contributions. Our contributions are threefold:

• Temporal control of pseudo-labels. We propose SCALAR, which uses Sinkhorn balancing over a short queue of recent predictions to bring their class marginals close to a specified target prior.

• When SCALAR helps. We track the effective number of pseudo-label classes to study when marginal balancing reduces drift and collapse. We also show that a uniform prior can hinder specialization when only a subset of source classes is present, and adaptation is already stable.

• Stabilization across adaptation methods. We show that SCALAR reduces drift and collapse under skewed streams and aggressive adaptation in both teacher–student and entropy-minimization settings, including EATA. Our ImageNet-C measurements show no measurable increase in runtime and a 10 MB increase in peak allocated GPU memory.

## 2 Related Work

Test-time and continual test-time adaptation. Test-time adaptation (TTA) updates a model at inference using unlabeled test samples [30, 21]. TENT [31] does this by minimizing prediction entropy with respect to the BatchNorm affine parameters. Continual TTA (CTTA) considers streams whose distribution changes over time [32]. Many CTTA methods use a teacher–student framework, where an EMA teacher generates pseudo-labels, and the student learns from them online [32, 10, 34, 23, 22]. Related source-free domain adaptation work constructs prototype-based pseudo-labels through probabilistic reasoning in concept space for concept bottleneck models [13].

![](images/c4656cf8da881ccface10fd42f6bf269eddbb19c4115ec31a91ef6c2ba54a4c3.jpg)  
Figure 1: Overview of the SCALAR framework. White blocks denote the baseline teacher–student CTTA pipeline; green blocks denote the added SCALAR module. Given an input view $x _ { u }$ for the teacher branch and an augmented view $x _ { v }$ for the student branch, the teacher produces logits $z _ { u }$ and probabilities $q _ { u } = \mathrm { s o f t m a x } ( z _ { u } )$ . Recent teacher logits are stored in a FIFO window and converted to probabilities $P ,$ which are balanced toward a target prior π via Sinkhorn to produce balanced targets $\tilde { q } _ { u } .$ . The student produces logits $s _ { v }$ and probabilities $p _ { v } .$ and is optimized using a mixture of the baseline consistency loss and the SCALAR loss based on $\tilde { q } _ { u }$ . Gradients are stopped through teacher-derived targets, marked by sg, and teacher parameters are updated via EMA. Here $\mathcal { L } _ { \mathrm { c o n s } } \equiv \mathcal { L } _ { \mathrm { c o n s i s t e n c y } }$ and $\bar { \mathcal { L } _ { \mathrm { b a l } } } = \mathrm { C E } ( \mathrm { s g } [ \tilde { q } _ { u } ] , \bar { p } _ { v } )$

Self-supervised learning methods, including contrastive learning, teacher–student learning, and masked image modeling [8, 18, 16, 6, 19, 2], have also informed auxiliary objectives for adaptation without labels [7, 22].

In multi-view clustering, COPER [12] learns cluster assignments from pseudo-labels that are consistent across views.

SeLa [1] and SwAV [5] use Sinkhorn balancing to avoid degenerate cluster assignments. For unsupervised permutation learning, the work in [11] introduced entropy-adaptive Gumbel–Sinkhorn normalization to improve assignment stability. ReMixMatch [3] aligns pseudo-labels with the labeleddata class distribution using a moving average of predictions over recent batches. Rozner et al. [27] use prediction-based label smoothing to discourage collapse into frequently predicted clusters when learning from unlabeled source domains for generalization to unseen domains. SCALAR instead applies a joint Sinkhorn projection over a short prediction window during CTTA, using a specified target prior without source data.

Class-prior shift and temporally correlated streams. Temporally correlated inputs and classordered streams can increase error accumulation and cause prediction collapse in CTTA [34, 10, 24, 15]. Universal TTA [24] examines these failures under class imbalance and class-ordered streams, while NOTE [15] studies adaptation to temporally correlated samples.

Existing methods address these failures by changing either the predictions or the samples used for adaptation. Universal TTA [24] reweights predictions using a class-prior estimate from the current batch. LAME [4] adjusts predictions using similarities between samples in feature space, without updating the model weights. EATA [25] updates the model only on samples selected for reliability and diversity. DELTA [35] addresses class bias through online loss reweighting, while TRIBE [29] uses balanced normalization and an anchored self-training loss.

SCALAR acts on the pseudo-labels used to update the model. It collects recent predictions in a short queue and balances their class marginals toward a specified prior. The balanced targets then provide supervision for adaptation, limiting the concentration of pseudo-label mass on a few classes.

## 3 Method

We first describe the teacher–student baseline, then introduce SCALAR as a step that balances the pseudo-labels used to update the student. SCALAR collects recent predictions in a short queue so that class usage is controlled across several batches. The architecture, optimizer, and update schedule remain the same. Fig. 1 shows the full procedure.

Let $f _ { \theta _ { 0 } } : \mathcal { X }  \mathbb { R } ^ { C }$ be a C-class classifier. In the CTTA setting, the model adapts online on an unlabeled stream $\{ x _ { t } \} _ { t \ge 1 }$ drawn from a potentially shifting distribution, without access to source data. We denote logits by $f _ { \theta } ( x ) \in \mathbb { R } ^ { C }$ and probabilities by softmax $( f _ { \theta } ( x ) )$ ). At each step, the model makes predictions and may use them to perform one adaptation update.

## 3.1 Teacher–Student CTTA Baseline

Our main experiments compare the same teacher–student baseline with and without SCALAR. The student $f _ { \theta }$ learns from the teacher’s predictions through gradient descent. The teacher $f _ { \phi }$ follows the student through an exponential moving average (EMA) of its parameters. We describe these two updates below; augmentations and hyperparameters are given in App. A.

Two-view consistency. For each unlabeled input $x _ { u } ,$ , we form an augmented view $x _ { v } = \mathcal { T } ( x _ { u } )$ where $\tau$ denotes a stochastic data augmentation operator. The teacher predicts from the input without further augmentation, while the student predicts from the augmented view:

$$
\begin{array} { r l r } & { z _ { u } = f _ { \phi } ( x _ { u } ) , \quad } & { q _ { u } = \mathrm { s o f t m a x } ( z _ { u } ) , } \\ & { s _ { v } = f _ { \theta } ( x _ { v } ) , \quad } & { p _ { v } = \mathrm { s o f t m a x } ( s _ { v } ) . } \end{array}
$$

The student minimizes a symmetric consistency loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n s i s t e n c y } } = \frac { 1 } { 2 } [ \mathrm { C E } ( \mathrm { s g } [ q _ { u } ] , p _ { v } ) + \mathrm { C E } ( p _ { v } , \mathrm { s g } [ q _ { u } ] ) ] , } \end{array}
$$

where $\mathrm { s g } [ \cdot ]$ denotes stop-gradient, and $\begin{array} { r } { \mathrm { C E } ( a , b ) \ = \ - \sum _ { c = 1 } ^ { C } a _ { c } \log b _ { c } } \end{array}$ denotes the cross-entropy between distributions a and b. This loss encourages the student to agree with the teacher despite the augmentation. The teacher predictions are held fixed during the student update.

EMA teacher. After each student update, the teacher’s parameters are updated as an exponential moving average with momentum coefficient $\alpha \in [ 0 , 1 )$ :

$$
\phi  \alpha \phi + ( 1 - \alpha ) \theta .
$$

A larger α makes the teacher follow the student more slowly, reducing the effect of any single update. However, averaging the parameters does not directly control which classes receive pseudo-label mass. SCALAR adds this control over a window of recent predictions. In our ViT baseline, attention and the remaining parameters use separate EMA rates (App. E.1); α denotes the rate for the corresponding parameters.

## 3.2 SCALAR: Class-Marginal Balancing via a Logit Queue

SCALAR balances predictions over a window of N samples, including the current batch of size $B .$ We use a temperature $\tau > 0$ to convert logits to probabilities and K Sinkhorn iterations to balance them toward a target prior $\pi \in \Delta ^ { C - 1 }$ , which is uniform by default. The weight $\lambda \in [ 0 , 1 ]$ controls the contribution of the balanced targets to the student loss.

Logit queue. We maintain a FIFO queue $Q _ { \mathrm { m e m } } \in \mathbb { R } ^ { N \times C }$ containing the most recent teacher logits. At each step, the oldest B logits are removed, and the current logits $z _ { u }$ are appended. The queue is converted to a probability matrix via a row-wise softmax:

$$
P = \mathrm { s o f t m a x } ( Q _ { \mathrm { m e m } } / \tau ) \in \mathbb { R } ^ { N \times C } , \qquad P _ { i , j } = \frac { \exp ( ( Q _ { \mathrm { m e m } } ) _ { i , j } / \tau ) } { \sum _ { c = 1 } ^ { C } \exp ( ( Q _ { \mathrm { m e m } } ) _ { i , c } / \tau ) } .
$$

The column sums $P ^ { \top } \mathbf { 1 } _ { N }$ give the total probability mass assigned to each class in the window, where ${ \bf 1 } _ { N }$ is a vector of ones. Dividing by $\dot { N }$ gives the average class distribution. Using several recent batches allows us to balance this distribution without requiring each small batch to represent all classes.

Algorithm 1 SCALAR Online Update.   
1: Input: batch $x _ { u } ,$ teacher $f _ { \phi } ,$ student $f _ { \theta } ,$ , queue $Q _ { \mathrm { m e m } }$ (capacity N), temperature $\tau ,$ iterations $K ,$ prior π,   
weight λ   
2: x<sub>v</sub> ← T (x<sub>u</sub>); z<sub>u</sub> ← f<sub>ϕ</sub>(x<sub>u</sub>); q<sub>u</sub> ← softmax(z<sub>u</sub>); p<sub>v</sub> ← softmax $\left( f _ { \theta } ( x _ { v } ) \right)$   
3: $\mathbf { i f } \left| Q _ { \mathrm { m e m } } \right| = N$ then   
4: Update queue (dequeue oldest B, enqueue $z _ { u } )$   
5: $\hat { P ^ { ' } } { \gets } \operatorname { s o f t m a x } ( \hat { Q _ { \mathrm { m e m } } } / \tau )$   
6: Q ← SinkhornBalance $( P , \pi , K )$   
7: q˜ ← last B rows of $Q$   
8: ${ \bf \dot { \mathcal { L } } } \gets ( 1 - \lambda ) { \mathcal { L } }$ <sub>consistency</sub> + λ CE(sg[˜q<sub>u</sub>], p<sub>v</sub>)   
9: else   
10: Q<sub>mem</sub> ← Enqueue(Q<sub>mem</sub>, z<sub>u</sub>)   
11: $\dot { \mathcal { L } }  \mathcal { L } .$ consistency \triangleright warm-up   
12: θ ← θ − η∇<sub>θ</sub>L; ϕ ← α ϕ + (1 − α) θ

Sinkhorn–Knopp balancing. Given the target prior π, we seek a matrix $Q$ whose rows sum to one and whose column sums match $N \pi$ . Starting from $Q ^ { ( 0 ) } = P ;$ , we iterate:

$$
\tilde { Q } ^ { ( k ) } = \mathrm { D i a g } ( Q ^ { ( k ) } \mathbf { 1 } _ { C } ) ^ { - 1 } Q ^ { ( k ) } , \qquad Q ^ { ( k + 1 ) } = \tilde { Q } ^ { ( k ) } \mathrm { D i a g } \left( \frac { N \pi } { \tilde { Q } ^ { ( k ) \top } \mathbf { 1 } _ { N } } \right) ,\tag{1}
$$

followed by a final row normalization. Here Diag(·) forms a diagonal matrix from a vector. Row normalization makes each row a probability distribution, while column scaling adjusts the total mass assigned to each class. After a finite number of iterations, the column sums approximately match $N \pi$ This procedure approximates the constrained KL projection of $P \left[ 2 8 , 9 \right]$ . Since classes are rescaled by different factors, their ordering within a sample can change. We use the last $B$ rows of $Q$ as the balanced targets $\tilde { q } _ { u }$ for the current batch; past predictions provide context for balancing.

Combined objective. We update the student using a mixture of the consistency loss and the loss on balanced targets $\tilde { q } _ { u } { : }$

$$
\mathcal { L } = ( 1 - \lambda ) \mathcal { L } _ { \mathrm { c o n s i s t e n c y } } + \lambda \mathrm { C E } ( \mathrm { s g } [ \tilde { q } _ { u } ] , p _ { v } ) ,\tag{2}
$$

where λ determines how strongly the balanced targets influence the student relative to the original consistency loss. As above, we stop gradients through the targets and update the teacher only via EMA. While the queue fills, we train with $\mathcal { L } _ { \mathrm { c o n s i s t e n c y } }$ alone.

Algorithm 1 summarizes the online update procedure.

Computational cost and wall-clock overhead. SCALAR adds a Sinkhorn balancing step over the logit queue. Each update costs $\mathcal { O } ( K N C )$ time and O(NC) memory, where N is the queue length, C the number of classes, and K the number of Sinkhorn iterations. For ImageNet-C $( C { = } 1 0 0 0$ $N { = } 1 0 2 4 , K { = } 5 )$ , this is about 0.02 GFLOPs and a ∼4 MB fp32 queue. Measured per-step time is unchanged within noise $( 0 . 1 0 6 3 \mathrm { s } \to 0 . 1 0 6 2 \mathrm { s } )$ ), and peak GPU memory increases by 10 MB allocated / 20 MB reserved. Full overhead details are provided in App. F.

## 4 Experiments

Our experiments address three questions:

• (Q1) State-dependent stabilization. When does SCALAR reduce drift and collapse? How does its benefit change as pseudo-label class usage contracts? Does it also improve predictive calibration?

• (Q2) Choice of target prior. When the test-time class prior is unknown, does a uniform target stabilize adaptation? How much does an accurate prior help, and when does a uniform target hinder specialization?

• (Q3) Use with other adaptation methods. Does SCALAR improve methods that already select samples for reliability and diversity? Does it also help TENT with a ResNet backbone?

![](images/807d05f2a0cf3e4a48df9ec1001f5914323294c58f85ad6f61b8d749583614f2.jpg)

![](images/be2e89203f460e47cd5ed688b80d9d315aca042370ea896827359a6e0638aa6f.jpg)

![](images/08eaef7a230227719c70487df42179990fc4e482f3cc4370d66d2202dbb627bd.jpg)  
Figure 2: Temporal contraction across adaptation regimes. Temporal evolution of error (top) and effective pseudo-label classes $C _ { \mathrm { e f f } }$ (bottom) in a teacher–student continual test-time adaptation setup with a ViT-B/16 backbone (ImageNet-C, severity 5, single-pass protocol). Here, $C _ { \mathrm { e f f } }$ denotes the effective number of classes in the teacher pseudo-label distribution. Left: Balanced stream (no skew) exhibits gradual contraction and late-stage drift. Middle: Fixed skew with rotating dominant classes (medium adaptation) results in stronger drift. Right: Dynamic skew with fast adaptation induces rapid contraction leading to collapse. Across regimes, SCALAR stabilizes class usage and mitigates the associated error growth.

Table 1: Balanced corruption benchmarks (severity 5). Average top-1 error (%) over all 15 corruptions on CIFAR-10-C, CIFAR-100-C, and ImageNet-C with ViT-B/16 backbones. SCALAR denotes our teacher–student CTTA pipeline with the FIFO logit queue and Sinkhorn balancing; Baseline w/o SCALAR is the same pipeline without this module. SCALAR uses a uniform class prior. For SCALAR and the teacher–student baseline, results are averaged over 5 seeds (std. ≈ 0.2); prior-method numbers use their published settings, which can differ in initialization and batch size.
<table><tr><td>Method</td><td>CIFAR-10-C</td><td>CIFAR-100-C</td><td>ImageNet-C</td></tr><tr><td>Source-only</td><td>28.1</td><td>35.4</td><td>55.8</td></tr><tr><td>TENT [31]</td><td>23.5</td><td>32.1</td><td>51.0</td></tr><tr><td>CoTTA [32]</td><td>24.6</td><td>34.8</td><td>54.8</td></tr><tr><td>VDP [14]</td><td>24.1</td><td>32.0</td><td>50.0</td></tr><tr><td>VIDA [23]</td><td>20.7</td><td>27.3</td><td>43.4</td></tr><tr><td>Versatile [33]</td><td>8.9</td><td>24.0</td><td>42.7</td></tr><tr><td>Universal [24]</td><td></td><td></td><td>45.0</td></tr><tr><td>ADMA [22]</td><td>12.6</td><td>26.4</td><td>42.5</td></tr><tr><td>REM [17]</td><td>9.4</td><td>23.4</td><td>39.2</td></tr><tr><td>SCALAR (ours)</td><td>8.1</td><td>23.2</td><td>36.1</td></tr><tr><td>Baseline (w/o SCALAR)</td><td>10.3</td><td>24.7</td><td>38.1</td></tr></table>

## 4.1 Setup

Protocol. Unless otherwise stated, experiments follow the standard continual test-time adaptation (CTTA) protocol [32]: a pretrained model adapts online to a single-pass stream of unlabeled test data, with no access to source data or ground-truth labels during adaptation. We apply this protocol to both standard corruption streams and controlled label-skewed streams constructed from ImageNet-C.

Datasets and streams. We use ImageNet-C and CIFAR-10/100-C [20] at severity 5. These benchmarks contain corrupted versions of the original validation images, grouped into 15 corruption types such as noise, blur, weather, and digital artifacts, which are processed sequentially to form a continual test stream. For the standard balanced benchmarks, we use the original corruption streams and report average top-1 error across all corruptions.

Table 2: Fixed-skew continual streams across adaptation speeds (ImageNet-C, severity 5). Each cell reports average corruption error and post-stream clean error (%), mean ± SD over three runs. Adaptation speed is controlled by the learning rate and EMA momentum (App. A.3). SCALAR-U uses a uniform class prior; SCALAR-C uses the class prior of each corruption segment (oracle reference). At high speed, the baseline collapsed in 2/3 runs; all SCALAR runs remained stable.
<table><tr><td></td><td colspan="2">High speed</td><td colspan="2">Medium speed</td><td colspan="2">Low speed</td></tr><tr><td>Method</td><td></td><td>Avg Err Clean Err</td><td></td><td>Avg Err Clean Err</td><td> $\operatorname { A v g }$ </td><td>Err Clean Err</td></tr><tr><td>Baseline (w/o SCALAR)</td><td> $8 6 . 8 \pm 1 4 . 4$ </td><td> $7 6 . 4 \pm 4 0 . 6$ </td><td> $4 8 . 3 \pm 4 . 4$ </td><td> $2 5 . 2 \pm 5 . 9$ </td><td> $4 7 . 5 \pm 0 . 6$ </td><td> $1 7 . 4 \pm 0 . 3$ </td></tr><tr><td>SCALAR-U</td><td> $4 8 . 1 \pm 1 . 1$ </td><td> $2 1 . 1 \pm 0 . 5$ </td><td> $4 2 . 9 \pm 0 . 7$ </td><td> $1 8 . 2 \pm 0 . 4$ </td><td> $4 9 . 1 \pm 0 . 4$ </td><td> $1 7 . 3 \pm 0 . 1$ </td></tr><tr><td> $\mathbf { S C A L A R - C }$ </td><td> ${ \bf 4 6 . 1 \pm 1 . 0 }$ </td><td> $2 2 . 2 \pm 0 . 4$ </td><td> $4 2 . 5 \pm 0 . 4$ </td><td> $1 7 . 7 \pm 0 . 3$ </td><td> $4 8 . 2 \pm 0 . 8$ </td><td>17.1 ±0.1</td></tr></table>

To study sustained prior shift, we construct fixed- and dynamic-skew ImageNet-C streams by resampling the class composition within each corruption segment; full details are provided in App. G. In both cases, classes are divided into head and tail groups, and skew is induced by oversampling head-class examples while subsampling tail-class examples. In the fixed-skew setting, each segment contains 50 head classes and 950 tail classes; each head class is sampled to 100 examples, while each tail class contributes one example. Thus, the per-class head–tail ratio is 100:1, with roughly 84% of the segment mass assigned to head classes. The skew strength is fixed across the stream, while the dominant class identities are deterministically reshuffled across corruption segments. In the dynamic-skew setting, the head-set size and per-class oversampling strength both vary across segments, producing head–tail ratios ranging from 25:1 to 250:1 while keeping the aggregate head mass roughly comparable. All methods are evaluated on streams with the same skew configuration. Long-horizon persistent-skew and true-subset variants are defined with the corresponding experiments; full construction details are provided in App. G.

Models. Our primary experiments use a ViT-B/16 backbone within a teacher–student framework. We also add SCALAR to EATA [25] with ViT-B/16 and to TENT [31] with a ResNet-50 backbone.

SCALAR settings. Unless otherwise stated, we use $N = 1 0 2 4 , \tau = 0 . 1$ , and $\lambda = 0 . 5 ;$ for EATA, whose entropy filter retains only sharply peaked predictions, we use $\tau = 0 . 7$ and $N = 4 0 9 6$ . Learning rates and EMA momenta are listed in App. A.3.

Metrics. We report: (i) top-1 corruption error (lower is better), (ii) post-stream clean error, measured on clean data after adaptation to quantify forgetting, and (iii) the effective number of pseudo-label classes, $C _ { \mathrm { e f f } } ( t )$ , which we use only as a label-free diagnostic of temporal class-marginal contraction, not as an accuracy measure:

$$
C _ { \mathrm { e f f } } ( t ) = \exp \Big ( - \sum _ { y = 1 } ^ { C } \bar { p } _ { t } ( y ) \log \bar { p } _ { t } ( y ) \Big ) ,
$$

where $\bar { p } _ { t }$ is the window-averaged pseudo-label distribution (before Sinkhorn balancing). $C _ { \mathrm { e f f } } ( t ) =$ C under uniform class usage and decreases as probability mass concentrates on fewer classes. Concentration can be appropriate when the stream contains only a few classes, and diverse predictions can still be wrong. We therefore examine $C _ { \mathrm { e f f } } ( t )$ alongside error when studying drift and collapse.

Baselines and variants. Baseline denotes our teacher–student CTTA pipeline without SCALAR’s Sinkhorn marginal-balancing module. SCALAR or SCALAR full framework denotes the same teacher–student pipeline with the SCALAR module. When SCALAR is added to an existing method, we write Method+SCALAR, e.g., EATA+SCALAR or TENT+SCALAR, meaning that only the SCALAR marginal-regulation module is added to that method. Unless otherwise stated, SCALAR uses a uniform class prior. We write SCALAR-U for this variant and SCALAR-C for the oracle-prior variant; the target is the segment prior or aggregate stream prior, as specified for each experiment.

## 4.2 State-Dependent Stabilization

We first examine how pseudo-label class usage changes during adaptation. We track $C _ { \mathrm { e f f } } ( t )$ , defined in Sec. 4.1, together with error to see when class contraction accompanies a loss of accuracy.

Table 3: Calibration under aggressive adaptation (ViT-B/16, fixed-skew ImageNet-C, high speed as in Tab. 2, single pass, averaged over three runs). The baseline collapsed in 2/3 runs.
<table><tr><td>Method</td><td>ECE↓</td><td>NLL↓</td></tr><tr><td>Baseline</td><td>0.73</td><td>13.37</td></tr><tr><td>SCALAR</td><td>0.07</td><td>2.15</td></tr></table>

Table 4: EATA under dynamic label skew (ImageNet-C, severity 5, ViT-B/16). Corruption / post-stream clean error (%), mean ± SD over three runs (stable) and eight runs (aggressive). Under aggressive adaptation, EATA collapses in 4/8 runs, whereas all SCALAR runs remain stable.
<table><tr><td>Method</td><td></td><td>Avg Err Clean Err Coll.</td><td></td></tr><tr><td>Stable</td><td></td><td></td><td></td></tr><tr><td>EATA</td><td> $5 0 . 8 { \pm } 2 . 4 $ </td><td> $1 8 . 8 { \pm } 0 . 6 $ </td><td>0/3</td></tr><tr><td>+SCALAR-U</td><td> ${ \bf 4 9 . 6 } \pm 2 . 1$ </td><td> ${ \bf 1 8 . 0 { \pm } } 0 . 4$ </td><td>0/3</td></tr><tr><td>Aggressive</td><td></td><td></td><td></td></tr><tr><td>EATA</td><td> $7 7 . 0 { \pm } 1 1 . 3 $ </td><td> $6 7 . 0 { \pm } 3 7 . 0 $ </td><td>4/8</td></tr><tr><td>+SCALAR-U</td><td> ${ \pm } 7 . 0 { \pm } 4 . 6 $ </td><td>一  $2 1 . 1 { \pm } 1 . 0$ </td><td>0/8</td></tr></table>

Fig. 2 shows the evolution of corruption error and $C _ { \mathrm { e f f } } ( t )$ across balanced, fixed-skew, and dynamicskew regimes. Error tends to increase as $C _ { \mathrm { e f f } } ( t )$ decreases. Balanced streams show mild gradual contraction, fixed skew strengthens this effect, and dynamic skew with rapid adaptation leads to sharp contraction and collapse.

On standard balanced corruption benchmarks (Tab. 1), where contraction is mild, SCALAR gives moderate improvements $( \mathrm { e . g . , 3 8 . 1 \%  3 6 . 1 \% }$ on ImageNet-C). Under stronger drift or collapse, SCALAR reduces the decline in $C _ { \mathrm { e f f } } ( t )$ and the associated rise in error. Its benefit is larger in these unstable regimes.

## 4.3 Uniform Targets under Class Skew

A uniform target may differ substantially from the class distribution in a skewed stream. We test whether it can still stabilize adaptation, then compare it with the correct prior to measure the cost of this mismatch.

Uniform prior in skewed streams. Tab. 2 reports results on fixed-skew ImageNet-C streams across adaptation speeds, controlled by the learning rate and EMA momentum. In high-speed regimes, where the baseline collapses in 2/3 runs, SCALAR-U reduces the corruption error from 86.8% to 48.1% and the post-stream clean error from 76.4% to 21.1%. It also improves corruption and clean performance at medium speed. At low speed, however, corruption error increases from 47.5% to 49.1%, while clean error changes from 17.4% to 17.3%, consistent with over-regularization when adaptation is already stable. Thus, the benefit depends on whether adaptation is becoming unstable, even though the class skew is the same at all three speeds. A target-sharpening baseline also collapses in 2/3 runs at high speed (App. E.4).

Benefit of an accurate prior. To quantify the benefit of accurate prior knowledge, we compare against SCALAR-C, which uses the empirical prior of each corruption segment as an oracle reference. Across the fixed-skew regimes, the correct prior further reduces corruption error relative to the uniform prior. At high speed, both targets prevent collapse in these runs. Knowing the prior improves accuracy further, but is not required for the observed stabilization.

## 4.4 Calibration

We also examine whether SCALAR improves the probabilities assigned to predictions. In a separate single-pass evaluation under aggressive adaptation, SCALAR reduces expected calibration error (ECE) from 0.73 to 0.07 and negative log-likelihood (NLL) from 13.37 to 2.15 (Tab. 3). These results show improved calibration in this setting, in addition to the reductions in top-1 error reported above.

## 4.5 Adding SCALAR to EATA

We next test whether SCALAR helps when the adaptation method already selects samples for reliability and diversity. EATA uses prediction entropy and a running estimate of past predictions to select samples for updating. We add SCALAR to control the class marginals of the supervision signal and evaluate both methods under dynamic skew (Tab. 4). In all EATA experiments, with or without SCALAR, the Fisher regularizer is disabled.

Table 5: Long-horizon persistent skew (ImageNet-C, five passes, >400K images, three runs). The same 50 classes dominate throughout (100 samples each, one per remaining class, per corruption). Error (%), $\mathrm { m e a n } \pm \mathrm { S D }$ . SCALAR-C uses the aggregate class prior of the stream (gray, oracle reference). Bold: best non-oracle result.
<table><tr><td>Method</td><td>Pass 1 Pass 5</td><td></td><td>Overall</td><td>Final clean Coll.</td><td></td></tr><tr><td>EATA</td><td>58.1</td><td>78.6</td><td> $6 6 . 5 { \pm } 1 2 . 9$ </td><td> $5 2 . 3 { \pm } 4 1 . 3 $ </td><td>1/3</td></tr><tr><td>+SCALAR-U</td><td>49.3</td><td>48.7</td><td> ${ \bf 5 0 . 0 \pm } 2 . 7$ </td><td> $2 1 . 5 { \pm } 1 . 4$ </td><td>0/3</td></tr><tr><td>+SCALAR-C</td><td>47.9</td><td>43.3</td><td> $4 3 . 9 2 1 . 7 $ </td><td> $2 1 . 4 \pm 0 . 8$ </td><td>0/3</td></tr></table>

Table 6: Long-horizon class absence (ImageNet-C, five passes, two runs). Only the same 50 classes appear; the remaining 950 are absent. Error (%), mean over runs; SCALAR-C in gray (oracle reference). Bold: best non-oracle result.
<table><tr><td>Method</td><td></td><td></td><td>Pass 1 Pass 5 Final clean Coll.</td><td></td></tr><tr><td>Conservative</td><td></td><td></td><td></td><td></td></tr><tr><td>EATA</td><td>51.2 51.6</td><td>44.2 48.7</td><td>18.3</td><td>0/2</td></tr><tr><td>+SCALAR-U</td><td>51.8</td><td>42.6</td><td>17.6 17.4</td><td>0/2 0/2</td></tr><tr><td>+SCALAR-C</td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive EATA</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>70.7</td><td>76.9</td><td>64.1</td><td>1/2</td></tr><tr><td>+SCALAR-U</td><td>50.0</td><td>52.9</td><td>22.2</td><td>0/2</td></tr><tr><td> $+ \mathrm { S C A L A R - C }$ </td><td>54.6</td><td>37.3</td><td>20.6</td><td>0/2</td></tr></table>

Table 7: Transfer to TENT (ResNet-50, standard ImageNet-C stream, severity 5). Corruption / post-stream clean error (%), mean ± SD over three random stream orderings, paired across methods. At medium and high speeds, TENT collapses in 3/3 runs; all TENT+SCALAR runs remain stable.
<table><tr><td></td><td colspan="2">Optimal speed</td><td colspan="2">Medium speed</td><td colspan="2">High speed</td></tr><tr><td>Method</td><td></td><td>Avg Err Clean Err</td><td></td><td>Avg Err Clean Err</td><td> $\operatorname { A v g } \operatorname { E r r }$ </td><td>Clean Err</td></tr><tr><td>TENT-R50 [31]</td><td> $6 7 . 5 \pm 0 . 4$ </td><td> $3 2 . 9 \pm 0 . 3$ </td><td> $8 8 . 3 \pm 1 . 3$ </td><td> $9 7 . 1 \pm 1 . 1 $ </td><td> $9 5 . 9 \pm 0 . 4 $ </td><td> $9 9 . 6 \pm 0 . 2 $ </td></tr><tr><td> $\mathbf { T E N T } + \mathbf { S C A L A R }$ </td><td> $6 6 . 5 \pm 0 . 3$ </td><td> $3 0 . 9 \pm 0 . 2 $ </td><td> ${ \bf 6 7 . 4 \pm 0 . 5 }$ </td><td> $3 7 . 1 \pm 0 . 8 $ </td><td> ${ \bf 6 9 . 8 \pm 1 . 1 }$ </td><td> $4 3 . 3 \pm 1 . 2$ </td></tr></table>

At EATA’s stable operating point, adding SCALAR reduces average error from 50.8% to 49.6%. However, EATA is sensitive to larger learning rates: under aggressive adaptation its average error rises to 77.0% and it collapses in four of eight runs. SCALAR prevents collapse in all runs, reducing average error to 57.0% and keeping clean error at 21.1%, close to its stable-regime level. Sample selection alone does not prevent collapse at this learning rate; balancing the targets provides additional stabilization.

Long-horizon persistent skew. The dynamic-skew stream changes which classes dominate. To test whether the benefit persists when the imbalance lasts longer, we keep the same 50 dominant classes throughout five passes (Tab. 5). EATA progressively degrades, from 58.1% error in the first pass to 78.6% in the fifth, and collapses in one of three runs. EATA+SCALAR-U remains stable in all runs, reaching 48.7% in the fifth pass and reducing final clean error from 52.3% to 21.5%. Using the aggregate stream prior further reduces overall error from 50.0% to 43.9%, whereas the stability benefit is already obtained with a uniform target.

Boundary case: true class absence. In the persistent-skew experiment, all classes still appear. We now remove the tail classes entirely: only the same 50 classes appear over five passes, and the remaining 950 are absent (Tab. 6). A uniform target assigns mass to these absent classes, so it may hinder specialization to the observed subset. Under conservative adaptation, EATA specializes to the observed classes, whereas EATA+SCALAR-U improves less, indicating mild over-regularization; the aggregate stream prior (SCALAR-C) recovers this specialization. Under aggressive adaptation, EATA collapses in one of two runs, whereas EATA+SCALAR-U remains stable but does not improve across passes. SCALAR-C again improves steadily. The uniform target prevents collapse in this setting, but limits specialization to the classes that are present.

## 4.6 Transfer Beyond ViT Teacher–Student Adaptation

We also add SCALAR to TENT with a ResNet-50 backbone (Tab. 7). This tests whether balancing helps a single model trained by entropy minimization, without an EMA teacher. SCALAR gives a small improvement at the stable operating point and larger gains when faster adaptation causes drift

and forgetting. At high adaptation speed, where TENT collapses in all three runs, SCALAR reduces corruption error from 95.9% to 69.8% and clean error from 99.6% to 43.3%.

## 4.7 Ablation: Temporal Window and Hyperparameters

We examine the effect of the temporal window size N, temperature τ , loss weight λ, and the number of Sinkhorn iterations K on ImageNet-C. A longer window includes more predictions when balancing class usage, but also retains older predictions for longer.

We evaluate N ∈ {256, 512, 1024, 4096}, $\tau \in [ 0 . 0 1 , 1 . 0 ]$ , and $\lambda \in [ 0 . 1 , 0 . 9 ]$ . Corruption error is 36.1% at both $N = 1 0 2 4$ and 4096, compared with 38.2% at $N = 5 \mathrm { { 1 \dot { 2 } } }$ . It varies by 1.1 percentage points over $\tau \in [ 0 . 0 1 , 1 . 0 ]$ and by 0.5 percentage points over $\lambda \in [ 0 . 3 , 0 . 7 ]$ . Additional Sinkhorn iterations beyond $K = 5$ did not yield measurable improvements. Detailed sensitivity curves are provided in App. C.

## 5 Conclusion

We proposed SCALAR to control the class marginals of the pseudo-labels used for continual test-time adaptation. SCALAR balances recent predictions over a short window before they are used to update the model. Its benefit is largest in our experiments when class usage contracts and adaptation becomes unstable. It also reduces drift and collapse when added to EATA and TENT, and improves calibration under aggressive adaptation.

A uniform target provides stabilization without knowledge of the test-time prior. The correct prior can improve accuracy further, but the choice of target also matters when adaptation is already stable. Under persistent class absence, uniform balancing can limit useful specialization to the classes that remain.

Limitations and future work. Our experiments focus on image classification and controlled skewed streams constructed from corruption benchmarks. Further work is needed to test SCALAR on naturally occurring non-i.i.d. streams and on tasks such as detection and segmentation. A further question is how to adapt the target prior and balancing strength so that they allow specialization while still preventing collapse.

Broader impact. Reducing adaptation-induced errors could improve the reliability of deployed image classifiers under changing conditions. However, forcing a uniform target can worsen performance under genuine class imbalance, so deployment requires validation on the intended population and monitoring for harmful adaptation.

## Acknowledgments and Disclosure of Funding

Ofir Lindenbaum was supported by the MOST grant No. 0007341. The authors declare no competing interests.

## References

[1] Yuki Markus Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous clustering and representation learning. In ICLR, 2020.

[2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a jointembedding predictive architecture. In CVPR, 2023.

[3] David Berthelot, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, and Colin Raffel. ReMixMatch: Semi-supervised learning with distribution alignment and augmentation anchoring. In ICLR, 2020.

[4] Malik Boudiaf, Romain Mueller, Ismail Ben Ayed, and Luca Bertinetto. Parameter-free online test-time adaptation. In CVPR, 2022.

[5] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020.

[6] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.

[7] Dian Chen, Dequan Wang, Trevor Darrell, and Sayna Ebrahimi. Contrastive test-time adaptation. In CVPR, 2022.

[8] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

[9] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In NeurIPS, 2013.

[10] Mario Döbler, Robert A. Marsden, and Bin Yang. Robust mean teacher for continual and gradual test-time adaptation. In CVPR, 2023.

[11] Ran Eisenberg and Ofir Lindenbaum. Learning permutation from structure without supervision. In ICML, 2026. URL https://arxiv.org/abs/2605.25551.

[12] Ran Eisenberg, Jonathan Svirsky, and Ofir Lindenbaum. COPER: Correlation-based permutations for multi-view clustering. In ICLR, 2025. URL https://openreview.net/forum? id=5ZEbpBYGwH.

[13] Ran Eisenberg, Ofir Lindenbaum, and Jacob Goldberger. A probabilistic source-free domain adaptation method for concept bottleneck models. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2026 Workshops and Challenges. Springer Nature Switzerland, 2026. URL https://papers.miccai.org/miccai-2026-sat/iMIMIC\_005.html.

[14] Yulu Gan, Yan Bai, Yihang Lou, Xianzheng Ma, Renrui Zhang, Nian Shi, and Lin Luo. Decorate the newcomers: Visual domain prompt for continual test time adaptation. In AAAI, 2023.

[15] Taesik Gong, Jongheon Jeong, Taewon Kim, Yewon Kim, Jinwoo Shin, and Sung-Ju Lee. NOTE: Robust continual test-time adaptation against temporal correlation. In NeurIPS, 2022.

[16] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Ávila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In NeurIPS, 2020.

[17] Jisu Han, Jaemin Na, and Wonjun Hwang. Ranked entropy minimization for continual test-time adaptation. In ICML, 2025.

[18] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

[19] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022.

[20] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In ICLR, 2019.

[21] Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, and Xiaodi Hou. Adaptive batch normalization for practical domain adaptation. Pattern Recognition, 80:109–117, 2018. doi: 10.1016/j.patcog.2018.03.005.

[22] Jiaming Liu, Ran Xu, Senqiao Yang, Renrui Zhang, Qizhe Zhang, Zehui Chen, Yandong Guo, and Shanghang Zhang. Continual-MAE: Adaptive distribution masked autoencoders for continual test-time adaptation. In CVPR, 2024.

[23] Jiaming Liu, Senqiao Yang, Peidong Jia, Renrui Zhang, Ming Lu, Yandong Guo, Wei Xue, and Shanghang Zhang. ViDA: Homeostatic visual domain adapter for continual test time adaptation. In ICLR, 2024.

[24] Robert A. Marsden, Mario Döbler, and Bin Yang. Universal test-time adaptation through weight ensembling, diversity weighting, and prior correction. In WACV, 2024.

[25] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-time model adaptation without forgetting. In ICML, 2022.

[26] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. In ICLR, 2023.

[27] Amit Rozner, Barak Battash, Lior Wolf, and Ofir Lindenbaum. Domain-generalizable multipledomain clustering. Transactions on Machine Learning Research, 2024. URL https:// openreview.net/forum?id=O9RUANpPmb.

[28] Richard Sinkhorn. A relationship between arbitrary positive matrices and doubly stochastic matrices. The Annals of Mathematical Statistics, 35(2):876–879, 1964. doi: 10.1214/aoms/ 1177703591.

[29] Yongyi Su, Xun Xu, and Kui Jia. Towards real-world test-time adaptation: Tri-net self-training with balanced normalization. In AAAI, 2024.

[30] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, and Moritz Hardt. Test-time training with self-supervision for generalization under distribution shifts. In ICML, 2020.

[31] Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. In ICLR, 2021.

[32] Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation. In CVPR, 2022.

[33] Xu Yang, Xuan Chen, Moqi Li, Kun Wei, and Cheng Deng. A versatile framework for continual test-time domain adaptation: Balancing discriminability and generalizability. In CVPR, 2024.

[34] Longhui Yuan, Binhui Xie, and Shuang Li. Robust test-time adaptation in dynamic scenarios. In CVPR, 2023.

[35] Bowen Zhao, Chen Chen, and Shu-Tao Xia. DELTA: Degradation-free fully test-time adaptation. In ICLR, 2023.

Table 8: Reproduction checks against published benchmark results. All values are average top-1 corruption error (%). Published results are retained in Table 1.
<table><tr><td>Method / Dataset</td><td>Reported</td><td>Reproduced</td></tr><tr><td>Source-only / CIFAR-10-C</td><td>28.1</td><td>27.9</td></tr><tr><td>Source-only / CIFAR-100-C</td><td>35.4</td><td>36.0</td></tr><tr><td>Source-only / ImageNet-C</td><td>55.8</td><td>56.2</td></tr><tr><td>VIDA / ImageNet-C</td><td>43.4</td><td>43.35</td></tr><tr><td>TENT / ImageNet-C</td><td>51.0</td><td>50.48</td></tr></table>

This appendix gives implementation details, hyperparameter analyses, per-corruption results, additional ablations, computational costs, and the construction of skewed streams.

## A Implementation Details

## A.1 Experimental Setup

Baselines and reproduction. For methods whose results are taken from prior CTTA papers, we use the reported numbers under the corresponding benchmark setting. As a check on our evaluation, we reproduce the source-only baseline using the ViT-B/16 checkpoint from our experiments. The reproduced source-only errors differ from the published values by at most 0.6 percentage points. Table 8 lists these reproduction checks. Our main controlled comparisons use paired runs with and without SCALAR under the same initialization.

Hyperparameters (balanced benchmarks). For CIFAR-10-C experiments, we use a batch size of 4, dual-rate EMA parameters $( \alpha _ { \mathrm { a t t } } , \alpha _ { \mathrm { r e s t } } ) = ( 0 . 9 0 , 0 . 9 9 )$ , and learning rate $\eta = 5 \times 1 0 ^ { - 5 }$ . CIFAR-100-C experiments use batch size 4, learning rate $\eta = 5 \times 1 0 ^ { - 5 }$ , and dual-rate EMA parameters $( \alpha _ { \mathrm { a t t } } , \alpha _ { \mathrm { r e s t } } ) = ( 0 . 9 9 9 9 , 0 . 9 9 9 )$ . ImageNet-C experiments use batch size $^ { 8 , }$ learning rate $\eta = 1 0 ^ { - 5 }$ and dual-rate EMA parameters $( \alpha _ { \mathrm { a t t } } , \alpha _ { \mathrm { r e s t } } ) = ( 0 . 9 0 , 0 . 9 9 )$ . We chose EMA rates for stable adaptation of the teacher–student baseline without SCALAR and fixed them for each dataset. The ablation in App. E.1 compares single- and dual-rate EMA settings with SCALAR; it does not compare EMA sensitivity with and without SCALAR.

## A.2 Augmentation Pipeline

The teacher processes the incoming images without further augmentation, while the student receives strongly augmented views for the consistency objective. Table 9 lists the student augmentations. We use transformations intended to preserve the class label while changing appearance, such as color changes and small geometric distortions. These augmentations are used for the adaptation loss; predictions are made on inputs without these additional transformations.

## A.3 Experiment-Specific Settings

Unless otherwise stated, all experiments use the SCALAR settings of Sec. 4.1 and the batch sizes of App. A.1, and collapse is defined as in $\mathbf { A p p }$ . G. The adaptation regimes in the main paper differ mainly in learning rate and, for the teacher–student pipeline, EMA momentum; we list these values below for each experiment.

Fixed-skew adaptation speeds (Tab. 2). High speed uses $\eta = 1 0 ^ { - 5 }$ with $\alpha _ { \mathrm { a t t } } = \alpha _ { \mathrm { r e s t } } = 0 . 9 ;$ medium speed uses $\eta = 5 \times 1 0 ^ { - 6 }$ with $\alpha _ { \mathrm { a t t } } = \alpha _ { \mathrm { r e s t } } = 0 . 9 9$ ; and low speed uses $\eta = 5 { \times } 1 0 ^ { - 7 }$ with $\alpha _ { \mathrm { a t t } } = \alpha _ { \mathrm { r e s t } } = 0 . 9 9 9$

EATA under dynamic skew (Tab. 4). When SCALAR is added to EATA, the queue stores the logits of the samples that EATA selects for the update, and the loss combines EATA’s weighted entropy with the cross-entropy to the balanced targets, with weights 1 − λ and λ. The queue retains the logits of the latest N selected samples; steps with no selected samples skip the update and leave the queue unchanged. In all EATA experiments, with or without SCALAR, the Fisher regularizer is disabled. The stable operating point uses $\eta = 1 0 ^ { - 5 }$ , chosen as the learning rate at which EATA alone remains stable across runs; the aggressive regime uses $\eta = 6 \times 1 0 ^ { - 4 }$ . EATA uses batch size 8, entropy margin $E _ { 0 } = \ln ( 1 0 0 0 ) / 2 - 1$ , and redundancy margin $\epsilon = 0 . 0 5$

Long-horizon experiments (Tabs. 5–6). The persistent-skew stream uses $\eta = 3 { \times } 1 0 ^ { - 4 }$ . On the class-absence stream, the conservative regime uses $\eta = 5 { \times } 1 0 ^ { - 5 }$ and the aggressive regime uses $\eta = 3 { \times } 1 0 ^ { - 4 }$

TENT (Tab. 7). Since TENT has no teacher, SCALAR stores the model’s own logits in the queue, and the loss combines the entropy with the cross-entropy to the balanced targets, with weights $1 - \lambda$ and λ. TENT uses batch size 64; optimal, medium, and high speeds use $\bar { \eta = 2 . 5 { \times } 1 0 ^ { - 4 } , 8 { \times } 1 0 ^ { - 2 } }$ and $1 . 5 { \times } 1 0 ^ { - 1 }$ , respectively.

Temporal contraction (Fig. 2). The balanced stream uses the ImageNet-C settings of App. A; the fixed- and dynamic-skew streams use the medium and high speeds of Tab. 2, respectively.

Table 9: Student augmentation pipeline for consistency training. Strong augmentations are applied only to student inputs; teacher inputs are not augmented.
<table><tr><td>Augmentation</td><td>Parameters + note</td></tr><tr><td rowspan="5"></td><td>degrees 0°</td></tr><tr><td>translate (32/W, 32/H)</td></tr><tr><td>scale 0.98–1.02, bilinear, fill=0</td></tr><tr><td>brightness 0.6–1.4, contrast 0.7–1.3,</td></tr><tr><td>saturation 0.5–1.5, hue ±0.06,</td></tr><tr><td>ColorJitterPro</td><td>gamma 0.7–1.3</td></tr><tr><td>RandomGrayscale</td><td> $p = 0 . 2$ </td></tr></table>

## B Additional Details on Sinkhorn Balancing

## B.1 KL-Projection Interpretation

The balancing step seeks targets that satisfy the desired class marginals while staying close to the teacher predictions in KL divergence [28, 9]. The constraints apply to the whole window: individual samples can still have different class distributions. Below, we give the window construction and normalization steps used to approximate this projection.

Window of teacher predictions. Let B be the current batch size and $M = N - B$ the number of past logits retained in the queue, so that the window has N rows as in Sec. 3. We stack the teacher logits from the current batch and the queue:

$$
\mathrm { l o g i t s } _ { \mathrm { e f f } } \in \mathbb { R } ^ { ( B + M ) \times C } .
$$

We convert logits to probabilities with a temperature $\tau > 0 \mathrm { : }$

$$
P = \mathrm { s o f t m a x } \big ( \mathrm { l o g i t s } _ { \mathrm { e f f } } / \tau \big ) \in \mathbb { R } ^ { N \times C } , \quad N = B + M .
$$

Each row of P is a class distribution for one sample. The column sums give the total predicted mass per class; dividing them by N gives the average distribution over the window.

Sinkhorn balancing to a prior. We seek $Q \in \mathbb { R } ^ { N \times C }$ with row sums 1 and column marginals $N \pi \in \mathbb { R } ^ { C }$

$$
Q { \bf 1 } = { \bf 1 } , \qquad { \bf 1 } ^ { \top } Q = N \pi ^ { \top } ,
$$

where $\pi \in \Delta ^ { C - 1 }$ encodes the desired class usage (uniform $\pi _ { c } = 1 / C$ when no test-time prior is known, or a given prior when class skew is specified). We obtain Q by applying K rounds of Sinkhorn scaling. Starting from $Q ^ { ( 0 ) } = P ;$ , each Sinkhorn iteration consists of:

$$
{ \tilde { Q } } ^ { ( k ) }  Q ^ { ( k ) } \oslash \big ( Q ^ { ( k ) } \mathbf { 1 } \big ) ,
$$

![](images/2558c0a21bf5c1f98fc2cbbc14ed9e9f74b0bca68cc16f33226bf5dd436e93d3.jpg)  
Window size

![](images/a9c86e0bc7cf3207966abbc23f4478ce8acc058b0acaaf2c84806920a7d0b575.jpg)  
Temperature τ

![](images/9727b929129a5ea8c1b2cf451eeea8f9ef47991218553d48eac09c8100cd3779.jpg)  
Figure 3: SCALAR hyperparameter sensitivity on ImageNet-C (class-balanced corruption setting, severity 5; error $\%$ ).

$$
Q ^ { ( k + 1 ) } \gets \tilde { Q } ^ { ( k ) } \mathrm { D i a g } \Big ( \frac { N \pi } { \tilde { Q } ^ { ( k ) \top } \mathbf { 1 } _ { N } } \Big ) , \qquad k = 0 , \dots , K - 1 ,
$$

and a final row normalization

$$
Q  Q ^ { ( K ) } \oslash \big ( Q ^ { ( K ) } \mathbf { 1 } \big ) .
$$

Here ⊘ denotes row-wise division by the current row sums and Diag(·) forms a diagonal matrix, as in Eq. (1). The final normalization ensures that each row sums to one. Since we use a finite number of iterations, the column sums need only approximately match $N \pi ; Q$ therefore approximates the KL projection onto the specified marginals. In the implementation, target column masses are clamped to at least $1 0 ^ { - 1 2 }$ and $1 0 ^ { - 1 2 }$ is added to all row and column sums, which avoids division by zero for classes with zero prior mass.

Objective and update. Let $q _ { u }$ denote the original teacher probabilities and $\tilde { q } _ { u }$ the balanced targets for the current batch obtained by Sinkhorn projection toward the target prior π. The SCALAR-specific supervision term is

$$
\mathcal { L } _ { \mathrm { b a l } } = \mathrm { C E } ( \mathrm { s g } [ \tilde { q } _ { u } ] , p _ { v } ) .
$$

The combined objective is

$$
\mathcal { L } = \left( 1 - \lambda \right) \mathcal { L } _ { \mathrm { c o n s i s t e n c y } } + \lambda \mathcal { L } _ { \mathrm { b a l } } ,
$$

which is identical to Eq. (2). During warm-up, before marginal balancing is activated, the method reduces to $\mathcal { L } _ { \mathrm { c o n s i s t e n c y } } .$

We apply a single gradient step on the student parameters θ and then update the teacher via the dual-rate EMA. SCALAR modifies the teacher targets used in the loss without changing the backbone architecture.

## C Hyperparameter Sensitivity

## C.1 SCALAR module ablations (queue + Sinkhorn)

We vary the window size, temperature, and loss weight to examine how each affects SCALAR. All experiments in this section use a uniform target and K=5 Sinkhorn iterations.

Figure 3 summarizes the sensitivity of SCALAR to its three primary hyperparameters on ImageNet-C in the standard corruption setting: (i) the mixing weight λ controlling the strength of the SCALAR term, (ii) the temperature τ applied in the softmax over the queue window, and (iii) the window size N (queue length).

Windows of 256–512 samples give higher error than windows of 1024–4096 samples. Increasing the window from $N = 1 0 2 4$ to 4096 gives no further reduction in this sweep. Thus, using more recent predictions helps up to a point, but the largest tested window offers no advantage over our default.

Error is lowest at $\tau = 0 . 1$ and $\lambda = 0 . 5$ , with slightly higher values at either end of the tested ranges. The mixing-weight sweep under class skew (App. E.3) also gives lower corruption errors at $\lambda \stackrel { - } { = } 0 . 5 \substack { - 0 . 7 5 }$ than at 0.25.

Table 10: CIFAR-10-C (severity 5) corruption results (error %, lower is better). Per-corruption test error and average over all 15 corruptions. Gain is the improvement in average error relative to the original Source model, in percentage points (higher is better). Boldface indicates the best method for each corruption. SCALAR results are averaged over 5 seeds; the standard deviation of its average error is below 0.2 percentage points. Prior-method values are taken from the cited publications.
<table><tr><td rowspan="2">Time Method</td><td rowspan="2">Ref</td><td colspan="10">Impulse</td><td colspan="10">Contrast Elastic</td></tr><tr><td>Gauss.</td><td>Shot</td><td></td><td></td><td>Defocus</td><td>Glass</td><td>Motion</td><td>Zoom</td><td>Snow</td><td>Frost</td><td>Fog</td><td>Bright.</td><td></td><td></td><td>Pixelate</td><td>JPEG</td><td>Avg.</td><td>Gain</td></tr><tr><td>Source</td><td></td><td>60.1</td><td>53.2</td><td>38.3</td><td>19.9</td><td>35.5</td><td>22.6</td><td>18.6</td><td>12.1</td><td>12.7</td><td>22.8</td><td>5.3</td><td></td><td>49.7</td><td>23.6</td><td>24.7</td><td>23.1</td><td>28.1</td><td>0.0</td></tr><tr><td>TENT</td><td>[31]</td><td>57.7</td><td>56.3</td><td>29.4</td><td>16.2</td><td>35.3</td><td>16.2</td><td>12.4</td><td></td><td>11.0</td><td>11.6</td><td>14.9</td><td>4.7</td><td>22.5</td><td>15.9</td><td>29.1</td><td>19.5</td><td>23.5</td><td>4.6</td></tr><tr><td>CoTTA</td><td>[32]</td><td>58.7</td><td>51.3</td><td>33.0</td><td>20.1</td><td>34.8</td><td>20.0</td><td>15.2</td><td>11.1</td><td>11.3</td><td>18.5</td><td>4.0</td><td></td><td>34.7</td><td>18.8</td><td>19.0</td><td>17.9</td><td>24.6</td><td>3.5</td></tr><tr><td>VDP</td><td>[14]</td><td>57.5</td><td>49.5</td><td>31.7</td><td>21.3</td><td>35.1</td><td>19.6</td><td>15.1</td><td>10.8</td><td>10.3</td><td>18.1</td><td>4.0</td><td></td><td>27.5</td><td>18.4</td><td>22.5</td><td>19.9</td><td>24.1</td><td>4.0</td></tr><tr><td>VIDA</td><td>[23]</td><td>52.9</td><td>47.9</td><td>19.4</td><td>11.4</td><td>31.3</td><td>13.3</td><td>7.6</td><td>7.6</td><td>9.9</td><td>12.5</td><td>3.8</td><td></td><td>26.3</td><td>14.4</td><td>33.9</td><td>18.2</td><td>20.7</td><td>7.4</td></tr><tr><td>Versatile</td><td>[33]</td><td>16.3</td><td>11.1</td><td>9.6</td><td>8.4</td><td>14.6</td><td>8.6</td><td>5.5</td><td>6.3</td><td>5.7</td><td>7.1</td><td>3.3</td><td></td><td>5.4</td><td>10.9</td><td>7.7</td><td>12.8</td><td>8.9</td><td>19.2</td></tr><tr><td>REM SCALAR</td><td>[17]</td><td>17.3</td><td>12.5</td><td>10.3</td><td>8.4 7.68</td><td>17.7 12.14</td><td>8.4 7.76</td><td>5.5 5.42</td><td>6.6 5.94</td><td>5.6 5.06</td><td>7.2 6.12</td><td>3.7 3.48</td><td></td><td>6.4 3.82</td><td>11.0 8.64</td><td>7.3</td><td>13.0 9.62</td><td>9.4 8.1</td><td>18.7 20.0</td></tr><tr><td></td><td>Proposed</td><td>20.32</td><td>10.32</td><td>9.52</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.58</td><td></td><td></td><td></td></tr></table>

Table 11: CIFAR-100-C (severity 5) corruption results (error %, lower is better). Per-corruption test error and average over all 15 corruptions. Gain is the improvement in average error relative to the original Source model, in percentage points (higher is better). Boldface indicates the best method for each corruption. SCALAR results are averaged over 5 seeds; the standard deviation of its average error is below 0.2 percentage points. Prior-method values are taken from the cited publications.
<table><tr><td rowspan="2">Time Method</td><td rowspan="2">Ref</td><td rowspan="2">t Gauss.</td><td rowspan="2">Shot</td><td rowspan="2">Impulse</td><td rowspan="2">Defocus</td><td rowspan="2">Glass</td><td rowspan="2">Motion</td><td rowspan="2">Zoom</td><td rowspan="2">Snow</td><td rowspan="2">Frost</td><td rowspan="2">Fog</td><td rowspan="2"></td><td rowspan="2">Contrast</td><td rowspan="2">Elastic</td><td rowspan="2">Pixelate</td><td rowspan="2">→ JPEG</td><td rowspan="2">Avg. Gain</td></tr><tr><td>Bright.</td></tr><tr><td>Source</td><td></td><td>55.0</td><td>51.5</td><td>26.9</td><td>24.0</td><td>60.5</td><td>29.0</td><td>21.4</td><td>21.1</td><td>25.0</td><td>35.2</td><td>11.8</td><td>34.8</td><td>43.2</td><td>56.0</td><td>35.9</td><td>35.4 0</td></tr><tr><td>TENT</td><td>[31]</td><td>53.0</td><td>47.0</td><td>24.6</td><td>22.3</td><td>58.5</td><td>26.5</td><td>19.0</td><td>21.0</td><td>23.0</td><td>30.1</td><td>11.8</td><td>25.2</td><td>39.0</td><td>47.1</td><td>33.3 32.1</td><td>3.3</td></tr><tr><td>CoTTA</td><td>[32]</td><td>55.0</td><td>51.3</td><td>25.8</td><td>24.1</td><td>59.2</td><td>28.9</td><td>21.4</td><td>21.0</td><td>24.7 34.9</td><td>11.7</td><td></td><td>31.7</td><td>40.4 55.7</td><td>35.6</td><td>34.8</td><td>0.6</td></tr><tr><td>VDP</td><td>[14]</td><td>54.8</td><td>51.2</td><td>25.6</td><td>24.2</td><td>59.1</td><td>28.8</td><td>21.2</td><td>20.5</td><td>23.3 33.8</td><td>7.5</td><td>11.7</td><td>32.0</td><td>51.7</td><td>35.2</td><td>32.0</td><td>3.4</td></tr><tr><td>VIDA</td><td>[23]</td><td>50.1</td><td>40.7</td><td>22.0</td><td>21.2</td><td>45.2</td><td>21.6</td><td>16.5</td><td>17.9</td><td>16.6 25.6</td><td>11.5</td><td>29.0</td><td>29.6</td><td>34.7</td><td>27.1</td><td>27.3</td><td>8.1</td></tr><tr><td>Versatile</td><td>[33]</td><td>38.2</td><td>31.8</td><td>18.2</td><td>20.8</td><td>34.3</td><td>20.3</td><td>17.5</td><td>14.9</td><td>16.2 22.9</td><td>11.5</td><td>27.5</td><td>28.2</td><td>32.5</td><td>25.3</td><td>24.0</td><td>11.4</td></tr><tr><td>ADMA</td><td>[22]</td><td>48.6</td><td>30.7</td><td>18.5</td><td>21.3</td><td>38.4</td><td>22.2</td><td>17.5</td><td>19.3</td><td>18.0 24.8</td><td>13.1</td><td>27.8</td><td>31.4</td><td>35.5</td><td>29.5</td><td>26.4</td><td>9.0</td></tr><tr><td>REM</td><td>[17]</td><td>29.2</td><td>25.5</td><td>17.0</td><td>19.1</td><td>35.2</td><td>21.2</td><td>18.3</td><td>19.5</td><td>18.7 22.8</td><td>15.5</td><td>17.6</td><td>31.6</td><td>26.2</td><td>33.0</td><td>23.4</td><td>12.0</td></tr><tr><td>SCALAR</td><td>Proposed</td><td>38.26</td><td>26.98</td><td>21.6</td><td>23.5</td><td>28.78</td><td>22.66</td><td>20.16</td><td>20.48</td><td>19.02 21.2</td><td>16.5</td><td>16.82</td><td>25.6</td><td>21.06</td><td>25.44</td><td>23.2</td><td>12.2</td></tr></table>

## D Per-Corruption Results

The following tables report results for all 15 corruption types under the default configurations used in the main paper.

In Tables 10–12, boldface indicates the best method for each corruption. On CIFAR-10-C, SCALAR attains the lowest error for 13 of 15 corruptions, and on ImageNet-C, it is best on 13 of 15 as well, while also achieving the lowest average error and the largest gain. On CIFAR-100-C, SCALAR is best on 4 out of 15 corruptions, but still attains the best average error and gain.

## E Additional Ablation Studies

## E.1 Dual vs. Single EMA on CIFAR-100-C

Dual-rate EMA teacher. We use an EMA teacher with separate decay rates for attention blocks and the remaining layers. Let $\theta ^ { \mathrm { a t t } }$ and $\theta ^ { \mathrm { r e s t } }$ denote the student parameters for attention and for the rest of the network (MLP, convolution, embeddings), and $\phi ^ { \mathrm { a t t } } , \phi ^ { \mathrm { r e s t } }$ the corresponding teacher parameters. After each student update, we apply

$$
\begin{array} { r l } & { \phi ^ { \mathrm { a t t } }  \alpha _ { \mathrm { a t t } } \phi ^ { \mathrm { a t t } } + ( 1 - \alpha _ { \mathrm { a t t } } ) \theta ^ { \mathrm { a t t } } , } \\ & { \phi ^ { \mathrm { r e s t } }  \alpha _ { \mathrm { r e s t } } \phi ^ { \mathrm { r e s t } } + ( 1 - \alpha _ { \mathrm { r e s t } } ) \theta ^ { \mathrm { r e s t } } , } \end{array}
$$

with $\left( \alpha _ { \mathrm { a t t } } , \alpha _ { \mathrm { r e s t } } \right)$ fixed by the teacher–student baseline configuration, independently of SCALAR. Separate rates allow attention and the remaining parameters to follow student updates at different speeds. We use dual-rate EMA in our ViT teacher–student experiments; SCALAR can also be used with other teacher–student designs or without an EMA teacher.

To study the effect of using separate EMA rates for different layers, we run a controlled ablation on CIFAR-100-C (severity 5) with SCALAR. We fix the learning rate and SCALAR hyperparameters (uniform prior, temperature $\tau = 0 . 1$ , window size $N = 1 0 2 4$ , Sinkhorn iterations $\bar { K } = \bar { 5 } )$ and vary only the EMA configuration.

Table 12: ImageNet-C (severity 5) corruption results (error %, lower is better). Per-corruption test error and average over all 15 corruptions. Gain is the improvement in average error relative to the original Source model, in percentage points (higher is better). Boldface indicates the best method for each corruption. SCALAR results are averaged over 5 seeds; the standard deviation of its average error is below 0.2 percentage points. Prior-method values are taken from the cited publications. Versatile’s reported mean of 42.7% is retained; its displayed entries average to 42.77%.
<table><tr><td rowspan="2">Time Method</td><td rowspan="2">Ref</td><td rowspan="2">t Gauss.</td><td rowspan="2">Shot</td><td rowspan="2">Impulse</td><td rowspan="2">Defocus</td><td rowspan="2">Glass</td><td rowspan="2">Motion</td><td rowspan="2">Zoom</td><td rowspan="2">Snow</td><td rowspan="2">Frost</td><td rowspan="2">Fog</td><td rowspan="2">Bright.</td><td rowspan="2">Contrast</td><td rowspan="2"></td><td colspan="2">JPEG</td><td rowspan="2">Avg.</td><td rowspan="2">Gain</td></tr><tr><td>Elastic</td><td>Pixelate</td></tr><tr><td>Source</td><td></td><td>53.0</td><td>51.8</td><td>52.1</td><td>68.5</td><td>78.8</td><td>58.5</td><td>63.3</td><td>49.9</td><td>54.2</td><td>57.7</td><td>26.4</td><td>91.4</td><td>57.5</td><td>38.0</td><td>36.2</td><td>55.8</td><td>0.0</td></tr><tr><td>TENT</td><td>[31]</td><td>52.2</td><td>48.9</td><td>49.2</td><td>65.8</td><td>73.0</td><td>54.5</td><td>58.4</td><td>44.0</td><td>47.7</td><td>50.3</td><td>23.9</td><td>72.8</td><td>55.7</td><td>34.4</td><td>33.9</td><td>51.0</td><td>4.8</td></tr><tr><td>CoTTA</td><td>[32]</td><td>52.9</td><td>51.6</td><td>51.4</td><td>68.3</td><td>78.1</td><td>57.1</td><td>62.0</td><td>48.2</td><td>52.7</td><td>55.3</td><td>25.9</td><td>90.0</td><td>56.4</td><td>36.4</td><td>35.2</td><td>54.8</td><td>1.0</td></tr><tr><td>VDP</td><td>[14]</td><td>52.7</td><td>51.6</td><td>50.1</td><td>58.1</td><td>70.2</td><td>56.1</td><td>58.1</td><td>42.1</td><td>46.1</td><td>45.8</td><td>23.6</td><td>70.4</td><td>54.9</td><td>34.5</td><td>36.1</td><td>50.0</td><td>5.8</td></tr><tr><td>VIDA</td><td>[23]</td><td>47.7</td><td>42.5</td><td>42.9</td><td>52.2</td><td>56.9</td><td>45.5</td><td>48.9</td><td>38.9</td><td>42.7</td><td>40.7</td><td>24.3</td><td>52.8</td><td>49.1</td><td>33.5</td><td>33.1</td><td>43.4</td><td>12.4</td></tr><tr><td>Versatile</td><td>[33]</td><td>47.5</td><td>42.1</td><td>41.6</td><td>55.5</td><td>55.4</td><td>44.5</td><td>47.9</td><td>38.8</td><td>37.8</td><td>39.6</td><td>23.6</td><td>57.0</td><td>44.4</td><td>33.5</td><td>32.3</td><td>42.7</td><td>13.1</td></tr><tr><td>ADMA</td><td>[22]</td><td>46.3</td><td>41.9</td><td>42.5</td><td>51.4</td><td>54.9</td><td>43.3</td><td>40.7</td><td>34.2</td><td>35.8</td><td>64.3</td><td>23.4</td><td>60.3</td><td>37.5</td><td>29.2</td><td>31.4</td><td>42.5</td><td>13.3</td></tr><tr><td>REM</td><td>[17]</td><td>43.5</td><td>38.1</td><td>39.2</td><td>53.2</td><td>49.0</td><td>43.5</td><td>42.8</td><td>37.5</td><td>35.2</td><td>35.4</td><td>23.2</td><td>46.8</td><td>41.6</td><td>28.9</td><td>30.2</td><td>39.2</td><td>16.6</td></tr><tr><td>SCALAR</td><td>Proposed</td><td>46.4</td><td>39.78</td><td>38.88</td><td>46.24</td><td>44.28</td><td>38.2</td><td>37.02</td><td>33.28</td><td>33.22</td><td>32.42</td><td>21.74</td><td>40.9</td><td>34.36</td><td>26.76</td><td>28.54</td><td>36.1</td><td>19.7</td></tr></table>

Table 13: Single vs. dual EMA on CIFAR-100-C (severity 5). Each row reports per-corruption error (%) on the CIFAR-100-C severity-5 stream, along with the average corruption error (Avg.) and the post-stream clean error on CIFAR-100 (Clean). Single-EMA rows use a shared rate for all parameters $( \alpha _ { \mathrm { a t t } } = \alpha _ { \mathrm { r e s t } } ) ,$ whereas the dual-EMA configuration decouples attention and the remaining weights.
<table><tr><td>LR</td><td>αrest</td><td>αatt</td><td>Gauss.</td><td>Shot</td><td>Impulse</td><td>Defocus</td><td>Glass</td><td>Motion</td><td>Zoom</td><td>Snow</td><td>Frost</td><td>Fog</td><td>Bright.</td><td>Contrast</td><td>Elastic</td><td>Pixelate</td><td>JPEG</td><td>Avg.</td><td>Clean</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9990</td><td>0.9990</td><td>40.61</td><td>30.07</td><td>25.03</td><td>26.99</td><td>32.66</td><td>24.94</td><td>22.76</td><td>23.75</td><td>22.07</td><td>25.28</td><td>19.25</td><td>19.57</td><td>31.42</td><td>29.34</td><td>32.71</td><td>27.10</td><td>25.30</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9991</td><td>0.9991</td><td>40.00</td><td>28.84</td><td>24.43</td><td>26.45</td><td>33.75</td><td>25.66</td><td>23.52</td><td>25.63</td><td>23.19</td><td>26.36</td><td>21.08</td><td>20.98</td><td>32.70</td><td>29.66</td><td>32.57</td><td>27.70</td><td>26.66</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9992</td><td>0.9992</td><td>41.08</td><td>29.82</td><td>23.95</td><td>26.38</td><td>31.99</td><td>24.83</td><td>22.23</td><td>23.64</td><td>21.46</td><td>24.87</td><td>19.06</td><td>19.43</td><td>31.31</td><td>28.88</td><td></td><td>32.87 26.80</td><td>25.38</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9993</td><td>0.9993</td><td>41.84</td><td>30.24</td><td>25.31</td><td>27.02</td><td>34.87</td><td>25.77</td><td></td><td>22.24 24.21 22.60 27.14</td><td></td><td></td><td>19.88</td><td>19.97</td><td>31.95</td><td>28.05</td><td></td><td>31.2027.50</td><td>24.66</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9994</td><td>0.9994</td><td>42.22</td><td>30.69</td><td>24.05</td><td>25.94</td><td>32.54</td><td>24.36</td><td>21.43</td><td>23.27</td><td>20.91</td><td>24.35</td><td>18.22</td><td>18.42</td><td>29.80</td><td>27.57</td><td></td><td>29.88 26.20</td><td>22.56</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9995</td><td>0.9995</td><td>43.48</td><td>31.34</td><td>24.91</td><td>27.29</td><td>33.43</td><td>24.87</td><td></td><td>21.60 23.17 20.85</td><td></td><td>24.69</td><td>18.13</td><td>18.71</td><td>30.59</td><td>31.42</td><td></td><td>29.85 27.00</td><td>22.63</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9996</td><td>0.9996</td><td>44.44</td><td>31.92</td><td>23.41</td><td>25.95</td><td>33.07</td><td>24.69</td><td></td><td>21.80 22.93</td><td>20.52</td><td>25.18</td><td>17.59</td><td>18.45</td><td>29.21</td><td>27.75</td><td>28.52</td><td>26.40</td><td>21.20</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9997</td><td>0.9997</td><td>47.44</td><td>36.02</td><td>23.51</td><td>25.30</td><td>33.43</td><td>23.65</td><td>19.61</td><td>20.66</td><td>18.99</td><td>23.34</td><td>15.56</td><td>16.65</td><td>27.82</td><td>29.03</td><td>27.84</td><td>25.90</td><td>19.49</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9998</td><td>0.9998</td><td>49.20</td><td>38.28</td><td>22.70</td><td>24.75</td><td>36.84</td><td>23.36</td><td>19.27</td><td>20.01</td><td>18.39</td><td>23.00</td><td>15.04</td><td>16.02</td><td>27.40</td><td>28.78</td><td>26.91</td><td>26.00</td><td>18.35</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.99985</td><td>0.99985</td><td>50.17</td><td>40.31</td><td>22.52</td><td>24.94</td><td>36.12</td><td>23.69</td><td>18.94</td><td>19.26</td><td>17.72</td><td>22.66</td><td>13.70</td><td>15.84</td><td>26.41</td><td>28.27</td><td>25.67</td><td>25.70</td><td>17.14</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9999</td><td>0.9999</td><td>51.47</td><td>42.70</td><td>22.45</td><td>23.59</td><td>37.90</td><td>22.53</td><td></td><td>17.81 17.73</td><td>16.36 21.80</td><td></td><td>12.34</td><td>14.78</td><td>24.71</td><td>31.79</td><td></td><td>25.71 25.60</td><td>15.92</td></tr><tr><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.9990</td><td>0.9999</td><td>43.56</td><td>32.51</td><td>20.39</td><td>22.05</td><td>31.25</td><td>20.69</td><td>17.53</td><td>18.32</td><td>16.67</td><td>20.55</td><td>13.66</td><td>15.43</td><td>24.12</td><td>23.76</td><td>23.64</td><td>22.90</td><td>16.73</td></tr></table>

Tab. 13 compares a single-EMA scheme, where a shared rate α is used for all parameters, and a dual-EMA scheme, where attention blocks and the remaining weights use different EMA rates $\left( \alpha _ { \mathrm { r e s t } } , \alpha _ { \mathrm { a t t } } \right)$ . For the single-EMA setting we sweep α in the range [0.9990, 0.9999] in steps of $1 0 ^ { - 4 }$ (with an additional point at $\alpha = 0 . 9 9 9 8 5 )$ , while the dual-EMA configuration uses $( \alpha _ { \mathrm { r e s t } } , \alpha _ { \mathrm { a t t } } ) =$ (0.999, 0.9999).

The average corruption error is measured on the CIFAR-100-C severity-5 stream, and the clean-post error is measured on CIFAR-100 after completing the stream. Within this sweep, the dual-EMA configuration achieves the lowest observed corruption error (22.9%) and reduces the clean-post error below the initial clean error (from ≈ 17.3% to 16.7%). The slowest single-EMA setting $( \alpha = 0 . 9 9 9 9 )$ achieves the best clean-post error in this grid (15.9%), but at the cost of a higher average corruption error (25.6%), roughly 2–3 percentage points above the dual-EMA value. Even when sampling single-EMA rates as fine as $\alpha = 0 . 9 9 9 8 5$ , we do not find a single rate that matches the dual-EMA corruption–clean trade-off in this configuration. Separate EMA rates help in this CIFAR-100-C experiment. More datasets and hyperparameter settings are needed to test whether the benefit holds more generally.

## E.2 Batch-Size Robustness

We examine the effect of batch size on EATA+SCALAR-U under persistent class skew over five consecutive passes (Tab. 14). The same 50 classes remain dominant throughout the stream. We vary the batch size while keeping the window size fixed at $N = 1 0 2 4$ , with learning rate $\eta = 1 0 ^ { - 5 }$ and temperature $\tau = 0 . 3$ . This is a separate configuration from the persistent-skew experiment in Tab. 5, which uses N = 4096, τ = 0.7, and $\eta = 3 \times 1 0 ^ { - 4 }$ ; the absolute errors are therefore not directly comparable.

SCALAR remains stable across the tested batch sizes, including batches of two samples. Larger batches show modestly lower corruption and post-stream clean errors in this setting.

Table 14: Batch-size sensitivity on ImageNet-C (severity 5) under five-pass persistent skew. Error (%), mean ± standard deviation over three runs.
<table><tr><td>Batch size</td><td> $\mathbf { A v g . \ e r r o r }$ </td><td>Final clean</td><td> $\mathrm { C o l l . }$ </td></tr><tr><td>2</td><td> $4 9 . 2 \pm 1 . 2$ </td><td> $2 0 . 3 \pm 0 . 3$ </td><td>0/3</td></tr><tr><td>4</td><td> $4 7 . 7 \pm 0 . 6$ </td><td> $1 9 . 9 \pm 0 . 5$ </td><td>0/3</td></tr><tr><td>8</td><td> $4 6 . 0 \pm 0 . 8$ </td><td> $1 9 . 3 \pm 0 . 1$ </td><td>0/3</td></tr></table>

Table 15: Sensitivity to the mixing weight under class skew on ImageNet-C (severity 5). Each entry reports average corruption error / post-stream clean error (%).
<table><tr><td>Dominant classes</td><td> $\lambda = 0 . 2 5$ </td><td> $\lambda = 0 . 5$ </td><td> $\lambda = 0 . 7 5$ </td></tr><tr><td>25</td><td>50.7 / 21.6</td><td>43.9 / 18.5</td><td>45.8 / 18.0</td></tr><tr><td>50</td><td>49.6 / 19.4</td><td>48.4 / 18.8</td><td>48.7 / 18.1</td></tr><tr><td>200</td><td>48.9 / 20.3</td><td>46.7 / 19.2</td><td>46.6 /18.3</td></tr><tr><td>Mean</td><td>49.7 / 20.4</td><td>46.3 / 18.9</td><td>47.0/ 18.1</td></tr></table>

## E.3 Sensitivity to the Mixing Weight under Class Skew

We also examine the sensitivity to the SCALAR mixing weight λ under three levels of class skew, with 25, 50, and 200 dominant classes (Tab. 15). We use a window size of $N = 1 0 2 4$ and report a single run per configuration.

The choice of mixing weight affects corruption and clean error differently. Averaged across the tested skew levels, $\lambda = 0 . 5$ gives the lowest corruption error, while $\lambda \stackrel { } { = } 0 . 7 5$ gives the lowest post-stream clean error. Each configuration uses one run, so these comparisons do not establish a precise optimum.

## E.4 Comparison with Target Sharpening

We compare SCALAR with a target-sharpening baseline in the fixed-skew, high-speed ImageNet-C setting of Tab. 2. The sharpened variant replaces the balanced targets $\tilde { q } _ { u }$ in Eq. (2) with softmax $ { \left( z _ { u } / \tau \right) }$ , keeping the same warm-up length, temperature $\tau = 0 . 1$ , and mixing weight $\lambda = 0 . 5$ The sharpening baseline is evaluated over three independent runs under matched adaptation conditions. The results are shown in Tab. 16.

Sharpening does not give a consistent improvement over the baseline, with both methods collapsing in two of the three runs. SCALAR-U gives lower corruption and final clean errors than sharpening in this comparison, suggesting that sharpening targets alone does not account for the benefit of balancing across the window.

## F Computational Overhead

SCALAR converts an $N \times C$ FIFO queue of teacher logits to probabilities and applies K iterations of Sinkhorn row/column scaling. Ignoring the lower-order cost of the per-step softmax on the current mini-batch (typically negligible when $\bar { N } \gg B )$ , each Sinkhorn iteration consists of two reductions and two elementwise normalization passes: row normalization $r = Q \mathbf { 1 } _ { C } , Q \gets Q \oslash r ,$ followed by column scaling $c = \mathbf { 1 } _ { N } ^ { \top } Q , s = t \oslash \bar { c , Q } \gets Q \odot s .$ where $\mathbf { 1 } _ { C }$ and $\mathbf { 1 } _ { N }$ are all-ones vectors, t encodes the target column marginals $( { \bf e . g . } , t = N \pi$ for a prior π), and $\oslash / \odot$ denote broadcast division / multiplication.

Each reduction and elementwise pass touches all $N C$ entries, so one iteration costs $\Theta ( N C )$ scalar operations. Over K iterations, the total complexity is

$$
\mathcal { O } ( K N C ) .
$$

These operations act on the prediction matrix and do not require another forward or backward pass through the model.

Table 16: Comparison with target sharpening on fixed-skew ImageNet-C (severity 5, high speed). Average corruption error and final clean error (%), mean $\pm$ standard deviation over three runs. Coll. indicates collapsed runs. Baseline and SCALAR-U results are from Tab. 2.
<table><tr><td>Method</td><td>Avg. error</td><td>Final clean</td><td>Coll.</td></tr><tr><td>Baseline (w/o SCALAR)</td><td> $8 6 . 8 \pm 1 4 . 4$ </td><td> $7 6 . 4 \pm 4 0 . 6$ </td><td>2/3</td></tr><tr><td>Sharpening</td><td> $9 0 . 7 \pm 6 . 5$ </td><td> $7 4 . 7 \pm 4 3 . 3$ </td><td>2/3</td></tr><tr><td>SCALAR-U</td><td> $4 8 . 1 \pm 1 . 1$ </td><td> $2 1 . 1 \pm 0 . 5$ </td><td>0/3</td></tr></table>

Table 17: Backbone vs. SCALAR module overhead. Compute is per forward pass for ViT-B/16 and per Sinkhorn-based CTTA update for SCALAR. SCALAR assumes C = 1000 classes (ImageNet), window size $N = 1 0 2 4$ , and $K = 5$ Sinkhorn iterations, as used in our experiments. For CIFAR-10/100 $( C \in \{ 1 0 , 1 0 0 \} )$ ) the overhead is another 10–100× smaller.
<table><tr><td>Component</td><td>Compute (GFLOPs)</td><td>Memory (MB)</td></tr><tr><td>ViT-B/16 backbone (forward)</td><td>≈ 17.5</td><td>≈ 330.3</td></tr><tr><td>SCALAR (ImageNet, FP32)</td><td>≈ 0.02</td><td>≈ 4 (fp32 queue)</td></tr></table>

Memory footprint. The memory cost is dominated by storing the logit queue $Q _ { \mathrm { m e m } } , \mathrm { i . e . , } N C$ floating-point entries, giving O(NC) memory. In our ImageNet experiments we use $C { = } 1 0 0 0$ classes and a window N≈1000 (e.g., N=1024), so $Q _ { \mathrm { m e m } }$ contains about $1 0 ^ { 6 }$ entries: ≈ 2 MB in fp16 (or ≈ 4 MB in fp32), plus negligible row/column buffers. For CIFAR-10/100, where $C \in \{ 1 \bar { 0 } , 1 0 0 \}$ both computation and memory scale linearly with C and are correspondingly smaller.

Compute relative to the backbone. For the window size and iteration count in Tab. 17, the estimated compute cost is below 1% of a ViT-B/16 forward pass. We report measured memory overhead below.

Measured overhead. On ImageNet-C (C=1000, window=1024) with micro-batch size $B { = } 8$ on an NVIDIA RTX 3080, per-step runtime is unchanged within measurement noise (0.1063s → 0.1062s) and peak GPU memory increases by 10MB (peak\_alloc) / 20MB (peak\_reserved). These measurements use our FP32 implementation, which includes redundant softmax operations. Removing them may reduce the overhead further.

## G Construction of Skewed Streams

We construct label-skewed ImageNet-C streams by controlling the class composition of each corruption segment. The stream stays skewed over extended intervals, allowing prediction biases to accumulate during adaptation.

For each corruption segment, we form a head–tail class split and resample examples according to this split. Head-class examples are oversampled with replacement to a fixed number of samples per head class, while tail classes are subsampled by keeping only a small number of examples per tail class. In our experiments, tail classes are subsampled to one example per class. This preserves the corruption sequence while making head classes more frequent. We generate the sample indices deterministically and reuse them across methods.

Fixed vs. dynamic label skew. In the fixed-skew setting, the skew strength and head-set size are fixed across corruption segments. Specifically, we use 50 head classes and 950 tail classes, oversampling each head class to 100 samples while keeping one example per tail class. The dominant class identities are selected deterministically for each segment, so all methods are evaluated on the same generated stream.

In the dynamic-skew setting, the number of head classes and samples per head class vary across corruption segments. We use a cyclic schedule in which the number of head classes follows

$$
( 3 0 , 1 5 0 , 2 0 , 2 0 0 , 4 0 , 1 0 0 ) ,
$$

and the corresponding number of samples per head class follows

$$
( 1 7 0 , 3 3 , 2 5 0 , 2 5 , 1 2 5 , 5 0 ) .
$$

Tail classes are again subsampled to one example per class. This changes the per-class head–tail ratio over time, from 25:1 up to 250:1, while keeping the aggregate head mass roughly comparable across segments. Class identities are deterministically reshuffled across segments, and once the schedule is exhausted, it is repeated from the beginning.

Both streams change which classes dominate. The dynamic stream also changes how many classes dominate and how much more often they appear.

Collapse criterion. We count a run as collapsed when its post-stream clean error exceeds 95%.

Oracle prior. SCALAR-C uses the empirical class distribution of each corruption segment as its target prior π, recomputed for every segment from the generated stream. This oracle is used only as a reference; no sample-level labels enter the adaptation procedure.

Long-horizon persistent skew. For the five-pass experiment in Tab. 5, we use the same 50 dominant classes throughout the entire stream. Within each corruption segment, each dominant class contributes 100 samples and each of the remaining 950 classes contributes one sample. A pass consists of all 15 ImageNet-C corruptions processed sequentially, with no model reset between corruptions or passes; each pass contains approximately 89K samples and the full stream approximately 446K samples. Unlike the fixed-skew setting, where the dominant class identities change across segments, here the same dominant set persists throughout. The oracle prior in this experiment is the aggregate class distribution of the stream.

True-subset stream. To isolate persistent class absence, we construct a stricter variant in which only the same 50 ImageNet classes appear throughout the stream, and the remaining 950 classes are completely absent. The subset is held fixed across all corruption segments and five consecutive passes, again without model reset. The mismatch between the uniform target and the stream prior therefore persists across all five passes.