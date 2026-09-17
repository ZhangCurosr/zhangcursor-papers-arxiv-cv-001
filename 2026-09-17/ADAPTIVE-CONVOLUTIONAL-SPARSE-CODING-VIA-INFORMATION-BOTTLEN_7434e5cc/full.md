# ADAPTIVE CONVOLUTIONAL SPARSE CODING VIA INFORMATION BOTTLENECK FOR ROBUST VISUAL SIGNAL REPRESENTATION

Meng’en Qin<sup>1</sup>, Yinchen Liu<sup>2</sup>, Mingxuan Cui<sup>3</sup>, Youlu Xing<sup>1,∗</sup>

<sup>1</sup>Faculty of Computer Science and Artificial Intelligence, Shenzhen University of Advanced Technology, Shenzhen, China <sup>2</sup>School of Mathematical Sciences,

University of Electronic Science and Technology of China, Chengdu, China <sup>3</sup>School of Mathematics, Shandong University, Jinan, China

## ABSTRACT

Visual signals require compact yet sufficient representations for robust downstream prediction. Convolutional sparse coding (CSC) provides an explicit mechanism for suppressing redundant components while preserving signal content, but its sparsity coefficient is typically fixed and manually selected. We propose an adaptive convolutional sparse coding framework for robust visual signal representation. Specifically, we unfold the CSC optimization with the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) and treat the sparsity coefficient as a differentiable variable jointly learned with the network parameters. From the information bottleneck perspective, this coefficient controls the trade-off between information retention and compression: the sparsity term promotes compact representations, while the reconstruction term together with task loss preserves task-relevant signal content. We further introduce a label-free post-training strategy that adjusts the compression strength for corrupted inputs with the main network parameters fixed. Experiments on CIFAR and ImageNet demonstrate competitive clean-data recognition and greatly improved robustness under different input perturbations.

Index Terms— convolutional sparse coding, information bottleneck, visual signal representation

## 1. INTRODUCTION

Robust visual recognition relies on learning representations that are both sufficient for downstream tasks and compact with respect to the input signal [1, 2, 3]. The information bottleneck (IB) principle [1, 2, 4] provides a theoretical perspective on this objective: given an input X and a downstream variable Y, a great output representation T should retain information relevant to Y while discarding information in X that is irrelevant to Y as much as possible. The corresponding IB objective can be formulated as

$$
\mathcal { F } _ { \operatorname* { m i n } } [ p ( t | x ) ] = \underbrace { I ( T ; X ) \downarrow } _ { \mathrm { e n s u r e ~ c o m p a c t n e s s } } - \underbrace { \beta I ( T ; Y ) \uparrow } _ { \mathrm { e n s u r e ~ s u f f i c i e n c y } } ,\tag{1}
$$

where $I ( \cdot ; \cdot )$ denotes mutual information, I(T; X) characterizes the amount of the retained input information, I(T; Y ) measures its task-relevant information, and $\beta > 0$ controls the trade-off between compression and preservation. From the IB perspective, the forward propagation of deep networks (e.g., ResNet [5], Swin Transformer [6], VMamba [7]) can be viewed as a progressive transformation of the input into increasingly task-oriented representations [8, 9]. Ideally, this transformation should remove task-irrelevant information without discarding information necessary for the task. However, modern deep networks typically optimize the final task loss without explicitly controlling the information trade-off of intermediate representations. As a result, the model may suffer from information degradation [10, 11, 12] when

$$
I ( Y ; X ) \geq I { \bigl ( } Y ; f _ { \theta _ { 1 } } ^ { 1 } ( X ) { \bigr ) } \cdots > I ( Y ; f _ { \theta _ { n } } ^ { n } ( X ) ) \cdots \geq I ( Y ; T ) ,\tag{2}
$$

where $f _ { \theta _ { n } } ^ { n } ( X )$ denotes the representation obtained after the first n layers; or information redundancy [13] if

$$
I ( X ; f _ { \theta _ { 1 } } ^ { 1 } ( X ) ) \geq \cdots I ( X ; f _ { \theta _ { n } } ^ { n } ( X ) ) \cdots \geq I ( X ; T ) > I ( Y ; T ) ,\tag{3}
$$

Such an imbalance between compactness and sufficiency can hinder the model’s robustness under input perturbations.

Convolutional sparse coding (CSC [14, 15, 16]) provides an explicit mechanism for controlling the complexity of visual represen tations while preserving task-relevant information together with the final task loss. CSC models $\boldsymbol { X } = ( \boldsymbol { x } ) _ { M } \in \mathbb { R } ^ { M \times H \times \grave { W } }$ as

$$
X = D * T \doteq ( \sum _ { i = 1 } ^ { C } ( d ) _ { i } * ( t ) _ { i } ) _ { M } ,\tag{4}
$$

where ∗ is the convolutional operator, $D = ( d ) _ { M \times C } \in \mathbb { R } ^ { M \times C }$ ×k×k denotes a convolutional dictionary, and $\overset { \cdot } { T } \overset { ^ { \prime } } { = } \mathit { \Pi } ( t ) _ { C } \in \mathbb { R } ^ { C \times H \times W }$ stands for a sparse representation. The sparse code is obtained by balancing reconstruction fidelity and representation complexity:

$$
T ^ { * } = \arg \operatorname* { m i n } _ { T } \underbrace { \frac { 1 } { 2 } \| X - D * T \| _ { 2 } ^ { 2 } } _ { \mathrm { i n f o r m a t i o n } \mathrm { p r e s e r v a t i o n } } + \underbrace { \lambda \| T \| _ { 1 } \downarrow } _ { \mathrm { r e p r e s e n t a t i o n } \mathrm { c o m p r e s s i o n } } .\tag{5}
$$

The reconstruction term encourages preserving the input signal, whereas the $\ell _ { 1 }$ term suppresses unnecessary parameters. $\lambda \geq 0$ provides an explicit control of the compression strength. Although λ in CSC and $\beta$ in the IB objective arise from different optimization formulations, they play analogous roles in adjusting the balance between representation compression and task-relevant information preservation. This property makes CSC with the task loss a natural signal processing mechanism for realizing the compact-sufficient trade-off advocated by the IB principle.

Some studies have tried to integrate CSC into deep networks, achieving competitive visual performance and improved robustness.

ML-CSC [14] pioneers the connection between convolutional networks and sparse coding. Res-CSC and MSD-CSC [15] explain the relation between the multi-layer convolutional sparse coding network and residual network. CSC-CTRL [17] uses CSC layers to build in vertible deep autoencoding models whose performance can compete with tried-and-tested deep generative models. SCN [16] trains a deep and end-to-end sparse coding network with a supervised task-driven algorithm via loss backpropagation. SDNet [18] shows that convolutional sparse coding can be integrated with deep networks through differentiable optimization layers, improving model robustness while maintaining computational efficiency. However, these approaches still typically treat λ as a pre-selected hyperparameter in training, which may lead to suboptimal compression across different layers and obstruct learning the IB trade-off. This inspires us to model λ as a learnable and adaptive variable whose value is determined jointly by the representation and downstream task.

Motivated by this, we propose an information bottleneck-driven adaptive convolutional sparse coding framework (ACSC). Rather than fixing the sparse coefficient, we make λ differentiable within the FISTA [19] iterations and learn it jointly with the convolutional dictionary and network parameters. The resulting layer provides an clear, layer-wise control of information compression. This design turns λ into a learnable variable and further enables its post-training adaptation using few unlabelled samples under various corruption. Our contributions are as follows:

• We establish an explicit connection between CSC and the IB principle and provide an interpretable lens for learning compact and sufficient visual representations.

• We propose an adaptive convolutional sparse coding framework that makes λ an update variable within the unfolded FISTA iterations.

• We further design an unsupervised post-training loss for λ adaptation to improve model robustness under corrupted inputs, while keeping other parameters fixed.

## 2. METHOD

## 2.1. Sparse Inference via FISTA

As shown in Fig. 1, considering the k-th iteration of n-th CSC layer, we start the optimization from $\begin{array} { r } { T _ { n } ^ { ( 0 ) } = 0 , t _ { n } ^ { ( 1 ) } = 1 } \end{array}$ and $Y _ { n } ^ { ( 0 ) } = { \dot { T _ { n } } } ^ { ( 0 ) }$ [20]. In Eq. (5), let

$$
f ( T _ { n } ^ { ( k ) } ) = \frac { 1 } { 2 } \left\| T _ { n - 1 } - D _ { n } * T _ { n } ^ { ( k ) } \right\| _ { 2 } ^ { 2 } ,\tag{6}
$$

whose gradient is Lipschitz continuous with constant $L _ { n }$ . In the k-th FISTA iteration, we have

$$
\begin{array} { c } { { U _ { n } ^ { ( k - 1 ) } = Y _ { n } ^ { ( k - 1 ) } - \displaystyle \frac { 1 } { L _ { n } } \nabla f \Big ( Y _ { n } ^ { ( k - 1 ) } \Big ) , } } \\ { { \nabla f \Big ( Y _ { n } ^ { ( k - 1 ) } \Big ) = ( D _ { n } ) ^ { T } * ( D _ { n } * Y _ { n } ^ { ( k - 1 ) } - T _ { n - 1 } ) , } } \end{array}\tag{7}
$$

followed by

$$
\begin{array} { r } { T _ { n } ^ { ( k ) } = S _ { \lambda _ { n } / L _ { n } } \Big ( U _ { n } ^ { ( k - 1 ) } \Big ) , } \end{array}\tag{8}
$$

where the element-wise soft-thresholding operator is defined as

$$
\begin{array} { r } { S _ { \tau } ( u ) = \mathrm { s i g n } ( u ) \operatorname* { m a x } \left( \left| u \right| - \tau , 0 \right) . } \end{array}\tag{9}
$$

The Nesterov acceleration [19] is given by

$$
\begin{array} { l } { { \displaystyle Y _ { n } ^ { ( k ) } = T _ { n } ^ { ( k ) } + \frac { t _ { n } ^ { ( k ) } - 1 } { t _ { n } ^ { ( k + 1 ) } } \left( T _ { n } ^ { ( k ) } - T _ { n } ^ { ( k - 1 ) } \right) , } } \\ { { \displaystyle t _ { n } ^ { ( k + 1 ) } = \frac { 1 + \sqrt { 1 + 4 ( t _ { n } ^ { ( k ) } ) ^ { 2 } } } { 2 } . } } \end{array}\tag{10}
$$

After a fixed number K of iterations, the sparse representation output of n-th CSC layer is $T _ { n } = F _ { \lambda _ { n } } ( T _ { n - 1 } ; \bar { D _ { n } } )$ .

## 2.2. Adaptive Convolutional Sparse Coding

As discussed in Sec. 1, the sparse coding objective is a favorable choice for the compactness-sufficiency balance advocated by the IB principle. Since $\lambda _ { n }$ directly determines the threshold in (8), it can be treated as a learnable compression variable. The key is to differentiate the unfolded FISTA iterations with respect to $\lambda _ { n }$

For the k-th iteration, we have

$$
\frac { \partial T _ { n } ^ { ( k ) } } { \partial \lambda _ { n } } = \left\{ \begin{array} { l l } { - \frac { 1 } { L _ { n } } \operatorname { s i g n } ( U _ { n } ^ { ( k - 1 ) } ) + \frac { \partial U _ { n } ^ { ( k - 1 ) } } { \partial \lambda _ { n } } , } & { | U _ { n } ^ { ( k - 1 ) } | > \frac { \lambda _ { n } } { L _ { n } } , } \\ { 0 , } & { | U _ { n } ^ { ( k - 1 ) } | \le \frac { \lambda _ { n } } { L _ { n } } . } \end{array} \right.\tag{11}
$$

Because $T _ { n - 1 }$ and $D _ { n }$ are fixed with respect to $\lambda _ { n }$ during the sparse inference of the current layer, differentiating (7) gives

$$
\frac { \partial U _ { n } ^ { ( k - 1 ) } } { \partial \lambda _ { n } } = \left[ I - \frac { 1 } { L _ { n } } ( D _ { n } ) ^ { T } * D _ { n } \right] * \frac { \partial Y _ { n } ^ { ( k - 1 ) } } { \partial \lambda _ { n } } .\tag{12}
$$

The dependence of the extrapolated variable on $\lambda _ { n }$ is obtained by differentiating (10). Since $t _ { n } ^ { ( k ) }$ is independent of $\lambda _ { n } ,$ we have

$$
\frac { \partial Y _ { n } ^ { ( k - 1 ) } } { \partial \lambda _ { n } } = \frac { \partial T _ { n } ^ { ( k - 1 ) } } { \partial \lambda _ { n } } + \frac { t _ { n } ^ { ( k - 1 ) } - 1 } { t _ { n } ^ { ( k ) } } \left( \frac { \partial T _ { n } ^ { ( k - 1 ) } } { \partial \lambda _ { n } } - \frac { \partial T _ { n } ^ { ( k - 2 ) } } { \partial \lambda _ { n } } \right) .\tag{13}
$$

Therefore, with the initialization $\begin{array} { r } { \frac { \partial T _ { n } ^ { ( 0 ) } } { \partial \lambda _ { n } } = \frac { \partial Y _ { n } ^ { ( 0 ) } } { \partial \lambda _ { n } } = 0 } \end{array}$ , Eqs. (11)-(13) provide a complete recursive computation of $\partial { T _ { n } } / \partial { \lambda _ { n } }$ . For a downstream training loss $\begin{array} { r } { \mathcal { L } _ { \mathrm { t r a i n } } , \frac { \partial \mathcal { L } _ { \mathrm { t r a i n } } } { \partial \lambda _ { n } } = \frac { \partial \mathcal { L } _ { \mathrm { t r a i n } } } { \partial T _ { n } } \frac { \partial \dot { T } _ { n } } { \partial \lambda _ { n } } } \end{array}$ . Consequently, $\lambda _ { n }$ can be optimized jointly with the network parameters by standard backpropagation. Additionally, to enforce non-negativity of $\lambda _ { n } ,$ we parameterize their updates as

$$
\lambda _ { n } \gets \mathrm { S o f t p l u s } ( \lambda _ { n } - \eta _ { \lambda _ { n } } \cdot \nabla _ { \lambda _ { n } } \mathcal { L } _ { \mathrm { t r a i n } } ) .\tag{14}
$$

With the hypergradient flow above, we optimize $\lambda _ { n }$ using the following IB-guided training objective:

$$
\mathcal { L } _ { \mathrm { t r a i n } } = \underbrace { \mathcal { L } _ { \mathrm { t a s k } } \downarrow } _ { \mathrm { e n c o u r a g e ~ s u f f i c i e n c y } } + \gamma \underbrace { \sum _ { n = 1 } ^ { L } \frac { \Vert F _ { \lambda _ { n } } ( T _ { n - 1 } ; D _ { n } ) \Vert _ { 1 } } { \vert T _ { n } \vert } \downarrow } _ { \mathrm { e n c o u r a g e ~ c o m p a c t n e s s } } ,\tag{15}
$$

$$
\frac { \partial \mathcal { L } _ { \mathrm { t r a i n } } } { \partial \lambda _ { n } } = \underbrace { \Bigg \langle \frac { \partial \mathcal { L } _ { \mathrm { t a s k } } } { \partial T _ { L } } , \frac { \partial T _ { L } } { \partial \lambda _ { n } } \Bigg \rangle } _ { \mathrm { t a s k \ ' g r a d i e n t } } + \underbrace { \gamma \sum _ { m = n } ^ { L } \frac { 1 } { | T _ { m } | } \left. \mathrm { s i g n } ( T _ { m } ) , \frac { \partial T _ { m } } { \partial \lambda _ { n } } \right. } _ { \mathrm { c o m p r e s s i o n  { g r a d i e n t } } } ,\tag{16}
$$

where $\langle , \rangle$ is the inner product, $| T _ { n } |$ denotes the number of elements in $T _ { n } , \mathcal { L } _ { \mathrm { t a s k } }$ is the task loss and $\gamma > 0$ controls the compression incentive. The task loss encourages preserving task-relevant information, while the second term encourages larger sparsity and stronger suppression of redundant components. Their competition implements the compression-retention trade-off motivated by the IB principle.

## 2.3. Label-free Post-training Adaptation for Corrupted Data

After source-domain training, the network parameters and the sparsity coefficients are denoted by (θ, λ). When the input distribution is corrupted or shifted, the learned λ from clean data may no longer provide an appropriate compression between redundancy removal and task-relevant preservation. We therefore adapt the compression coefficients using a small set of unlabeled corrupted or shifted samples while keeping θ fixed. We define the relative reconstruction error as a label-free fidelity measure:

![](images/2bcb50846cd69dd82d21fde8c51fc548bb60c59ed1cf5d7abccb0f99f2225f29.jpg)  
Fig. 1: Illustration of the proposed adaptive convolutional sparse coding framework (ACSC).

$$
\mathcal { R } _ { n } = \frac { \left\| T _ { n - 1 } ^ { \hat { t } } - D _ { n } * \hat { T } _ { n } ^ { \hat { t } } \right\| _ { 2 } ^ { 2 } } { \left\| T _ { n - 1 } - D _ { n } * T _ { n } \right\| _ { 2 } ^ { 2 } + \epsilon } , \qquad \hat { T } _ { n } = F _ { \lambda _ { n } } \hat { ( T _ { n - 1 } ; D _ { n } ) } ,\tag{17}
$$

where $\epsilon > 0$ is a small constant for numerical stability, t means t-th $\lambda _ { n }$ update. We then optimize λ using the following label-free loss:

$$
\mathcal { L } _ { \mathrm { a d a p t } } = \sum _ { n = 1 } ^ { L } { \frac { \mathcal { R } _ { n } } { \lambda _ { n } + \epsilon } }\tag{18}
$$

Minimizing (18) encourages larger λ to suppress redundant components, while the relative reconstruction term penalizes excessive compression that would distort the observed signal. In contrast to the source-domain training objective (15), (18) depends only on the observed corrupted signal and the sparse reconstruction, enabling unsupervised post-training adaptation. During adaptation, the main network parameters are frozen, and only the coefficients λ are up dated: $\lambda _ { n } ^ { t + 1 } = \mathrm { S o f t p l u s } ( \lambda _ { n } ^ { t } - \eta _ { \lambda _ { n } ^ { t } } \cdot \nabla _ { \lambda _ { n } ^ { t } } \mathcal { L } _ { \mathrm { a d a p t } } )$ . When batch normalization [21] is employed, its statistics can be updated using the corrupted batches to adapt to the distribution shift.

## 3. EXPERIMENTS

We evaluate the proposed ACSC on the ImageNet-1K [22], CIFAR-10 and CIFAR-100 [23] datasets. We use ResNet-18 as the baseline backbone and construct two variants: ACSC-18 and ACSC-18-all, which replace the first and all convolutional layers with ACSC layers, respectively. Two FISTA iterations are unrolled to perform the forward pass of each CSC layer, and γ is set to 0.001 in Eq. (15) across all experiments. For post-training adaptation, a small unlabeled subset (100 by default) of corrupted samples is used to update the compression coefficients λ while keeping θ frozen. To train models, we used a single NVIDIA RTX 2080Ti with batch size 128 for CIFAR-10/100, and 4 NVIDIA RTX 3090 GPUs with batch size 512 for ImageNet. We compare against ResNet-18 [5] and other CSC methods under the same training protocol.

## 3.1. Classification Performance on Clean Data

Table 1 shows that the proposed ACSC achieves great performance on clean data. When all convolutional layers are replaced with ACSC layers, our method reaches 97.65%, 80.76% and 72.53% on CIFAR-10, CIFAR-100 and ImageNet, outperforming other methods.

![](images/8f6a47d184936ece7d39500a1c82cf503c819ab5191cf97a4632b1fe551bf176.jpg)  
Fig. 2: The adapted λ value under different noise types and severities.

Table 1: Performance of different methods on clean test data, including CIFAR-10, CIFAR-100, and ImageNet datasets.
<table><tr><td rowspan=1 colspan=5>Dataset          Method     Top-1 Acc MemroyTraining Speed</td></tr><tr><td rowspan=4 colspan=1>CIFAR-10</td><td rowspan=3 colspan=1>ResNet-18 [5]SCN-18 [16]SDNet-18 [18]</td><td rowspan=1 colspan=1>95.54%</td><td rowspan=1 colspan=1>1.0 GB</td><td rowspan=1 colspan=1>1600 n/s</td></tr><tr><td rowspan=1 colspan=1>95.12%</td><td rowspan=1 colspan=1>3.5 GB</td><td rowspan=1 colspan=1>158 n/s</td></tr><tr><td rowspan=1 colspan=1>95.20%</td><td rowspan=1 colspan=1>1.2 GB</td><td rowspan=1 colspan=1>1500 n/s</td></tr><tr><td rowspan=1 colspan=1>ACSC-18 (ours)ACSC-18-all (ours)</td><td rowspan=1 colspan=1>96.18%97.65%</td><td rowspan=1 colspan=1>1.4 GB3.8 GB</td><td rowspan=1 colspan=1>1324 n/s463 n/s</td></tr><tr><td rowspan=4 colspan=1>CIFAR-100</td><td rowspan=3 colspan=1>ResNet-18 [5]SCN-18 [16]SDNet-18 [18]</td><td rowspan=3 colspan=1>77.82%78.59%78.31%</td><td rowspan=1 colspan=1>1.0 GB</td><td rowspan=1 colspan=1>1600 n/s</td></tr><tr><td rowspan=1 colspan=1>3.5 GB</td><td rowspan=1 colspan=1>158 n/s</td></tr><tr><td rowspan=1 colspan=1>1.2 GB</td><td rowspan=1 colspan=1>1500 n/s</td></tr><tr><td rowspan=1 colspan=1>ACSC-18 (ours)ACSC-18-all (ours)</td><td rowspan=1 colspan=1>79.63%80.76%</td><td rowspan=1 colspan=1>1.4 GB3.8 GB</td><td rowspan=1 colspan=1>1324 n/s463 n/s</td></tr><tr><td rowspan=4 colspan=1>ImageNet-1K</td><td rowspan=3 colspan=1>ResNet-18 [5]SCN [16]SDNet-18 [18]</td><td rowspan=3 colspan=1>68.98%70.42%69.47%</td><td rowspan=1 colspan=1>24.1 GB</td><td rowspan=3 colspan=1>2100 n/s51 n/s1800 n/s</td></tr><tr><td rowspan=1 colspan=1>95.1 GB</td></tr><tr><td rowspan=1 colspan=1>37.6 GB</td></tr><tr><td rowspan=1 colspan=1>ACSC-18 (ours)ACSC-18-all (ours)</td><td rowspan=1 colspan=1>71.12%72.53%</td><td rowspan=1 colspan=1>39.7GB88.6 GB</td><td rowspan=1 colspan=1>1689 n/s153 n/s</td></tr></table>

## 3.2. Robustness Analysis on Corrupted Data

Table 2 shows that, without post-training adaptation, ACSC-18 already consistently outperforms both ResNet-18 and other CSC baselines. More importantly, label-free post-training adaptation of λ further improves the robustness of the proposed models across all corruption types. The adapted ACSC-18 also surpasses the corresponding SDNet-18 model with per-sample λ tuning, demonstrating that the proposed post-training compression adaptation is more effective than directly tuning the sparsity coefficient of a fixed sparsecoding model. These results support the view that robustness can be improved by re-estimating the compression strength according to the corrupted data distribution rather than keeping a fixed compression level learned from clean data. In Fig. 2, a monotonic trend can be observed across all noise types: λ increases as the corruption severity becomes stronger, indicating that the proposed ACSC-18 automatically imposes stronger sparsity constraints when the input contains more redundancy. This behavior is consistent with the information bottleneck interpretation, where λ serves as a controllable compression variable that increases the suppression of task-irrelevant components as the amount of nuisance information grows.

![](images/dde78dc6c0bf05474c1b0301b0f89840e087b180986bdb95d30a3a360db3d0fd.jpg)

![](images/9e05a094c8350aca467673f1fa23a4d82a36066d4b3303f4944f6c9368a9b6f4.jpg)

![](images/e8be25ab9c85d15b4d631c7ee5016dce333a88b78be7e364b44fb4c12afab883.jpg)

![](images/a6fcc9b60df3c0d49701e38bf33f9a18b61ea9fc8846539b23d81ca10a84f198.jpg)

![](images/84cbcf894bd837a98f9f1ddf897af9cd7375bee03b7c7fefde1b890a5e912762.jpg)  
Fig. 3: λ dynamic behavior when ACSC-18-all is trained on CIFAR-10. We visualize λ values of all ACSC layers at the T-th iteration

Table 2: Test accuracy of different models trained on clean data and evaluated on different noise types from CIFAR-10-C and ImageNet-C [24]. The results are averaged over 5 severity levels for each type.
<table><tr><td colspan="2">Noise Type</td><td colspan="4">CIFAR-10-C</td><td colspan="3">ImageNet-C</td></tr><tr><td colspan="2">Method</td><td>Gaussian</td><td>Shot</td><td></td><td>Speckle Impulse</td><td>Gaussian</td><td>Shot</td><td>Impulse</td></tr><tr><td colspan="2">ResNet-18 [5]</td><td>44.43%</td><td>57.88%</td><td>62.16%</td><td>51.72%</td><td>22.73%</td><td></td><td>21.78% 17.38%</td></tr><tr><td colspan="2">SCN [16]</td><td>50.79%</td><td>62.97%</td><td>67.45%</td><td>54.19%</td><td></td><td></td><td></td></tr><tr><td colspan="2">SDNet-18 [18]</td><td>50.58%</td><td>63.29%</td><td>67.11%</td><td>54.13%</td><td>24.98%</td><td>23.97%</td><td>19.12%</td></tr><tr><td colspan="2">ACSC-18 (ours)</td><td>51.62%</td><td>63.95%</td><td>68.23%</td><td>54.26%</td><td>25.63%</td><td>24.49%</td><td>20.76%</td></tr><tr><td colspan="2">ACSC-18-all (ours)</td><td>53.98%</td><td>65.12%</td><td>69.87%</td><td>55.39%</td><td>27.22%</td><td>26.33%</td><td>22.09%</td></tr><tr><td colspan="2">SDNet-18 + per-sample λ tuning ACSC-18 (ours)</td><td>64.92%</td><td>71.13%</td><td>71.42%</td><td>57.48%</td><td>29.16%</td><td>27.59%</td><td>22.01%</td></tr></table>

Table 3: Ablation on the number of FISTA iterations and number of corrupted samples used in post-training adaptation. We report ACSC-18 performance on CIFAR-10 [23] and CIFAR-10-C [24].
<table><tr><td></td><td></td><td>iteration number Top-1 Acc, sample number Gaussian</td><td></td><td>Shot</td><td></td><td>Speckle Impulse</td></tr><tr><td>2</td><td>96.18%</td><td>50</td><td>66.04%</td><td>76.93%</td><td>71.12%58.67%</td><td></td></tr><tr><td>4</td><td>96.54%</td><td>100</td><td>66.63%</td><td></td><td>72.21% 71.89%59.03%</td><td></td></tr><tr><td>8</td><td>96.93%</td><td>500</td><td>67.54%</td><td></td><td>73.13%72.46%</td><td>60.58%</td></tr></table>

## 3.3. Ablation Analysis

Table 3 studies the effects of the number of FISTA iterations and corrupted samples used for post-training adaptation. Increasing FISTA iterations gradually improves the clean Top-1 accuracy, while increasing the adaptation samples from 50 to 500 also promotes robustness across all corruption types. However, the gains are relatively limited compared with the additional computational costs. We therefore use 2 FISTA iterations and 100 corrupted samples as the default setting.

## 3.4. λ Dynamic Behavior Analysis in Training

To investigate how the proposed adaptive compression mechanism evolves during training, we visualize the layer-wise dynamics of λ in ACSC-18-all, as is shown in Fig. 3. At the early stage of training, the learned λ remain relatively small, allowing the network to preserve more information from the input while primarily optimizing the downstream task. As the training accuracy approaches saturation, the compression coefficients increase rapidly and subsequently converge. This behavior suggests a two-stage learning process: the early training is dominated by task fitting, whereas the later stage increasingly favors the removal of redundant representation components. Such a fitting-compression transition is consistent with the information bottleneck interpretation, which emphasizes retaining task-relevant information while progressively suppressing information that is less useful for the downstream task. The layer-wise distribution of λ further reveals a clear depth-dependent compression pattern. The coefficients in earlier layers are generally smaller, whereas larger values are observed in layers closer to the downstream task. This observation is consistent with the IB view that early layers should preserve a broader range of input information, while representations closer to the prediction objective can impose stronger compression once task-relevant information has been extracted. Therefore, the learned λ profile provides an explicit, interpretable indicator of how compression is distributed across the network hierarchy.

In addition, the learned λ exhibits four pronounced compression cycles during training, manifested as four major peaks in the profile. We attribute this behavior to the architecture of ResNet, where the feature representation width is expanded at four stages. Each expansion increases the representational capacity and may consequently introduce additional redundant components. The model responds by assigning stronger compression coefficients around these expansion stages, resulting in the four observed peaks. This architecturedependent pattern further indicates that the adaptive sparsity coefficients are not merely free parameters, but reflect the representationcompression behavior of different stages of the network.

## 4. CONCLUSION

We presented an information bottleneck-driven adaptive convolutional sparse coding framework for robust visual signal representation. By unfolding FISTA, λ becomes a trainable compression variable, enabling the network to jointly learn task-related representations and adaptive compression. We further introduced a label-free post-training adaptation strategy that re-estimates the compression strength for corrupted inputs. Despite these brilliant results, the study is mainly evaluated on the ResNet architecture, and the relationship between the learned λ and information compression is supported primarily by empirical evidence. Future work will investigate more diverse visual datasets and distribution shifts, establish a more rigorous theoretical connection between sparse coding and the information bottleneck objective, and explore more general adaptive compression mechanisms beyond the current CSC architecture.

## 5. REFERENCES

[1] Naftali Tishby, Fernando C Pereira, and William Bialek, “The information bottleneck method,” arXiv preprint physics/0004057, 2000. 1

[2] Naftali Tishby and Noga Zaslavsky, “Deep learning and the information bottleneck principle,” in 2015 IEEE Information Theory Workshop, 2015, pp. 1–5. 1

[3] Raef Bassily, Shay Moran, Ido Nachum, Jonathan Shafer, and Amir Yehudayoff, “Learners that use little information,” in Proceedings ofAlgorithmic Learning Theory. PMLR, 2018, pp. 25–55. 1

[4] Shizhe Hu, Zhengzheng Lou, Xiaoqiang Yan, and Yangdong Ye, “A survey on information bottleneck,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5325–5344, 2024. 1

[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778. 1, 3, 4

[6] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision. IEEE, 2021, pp. 9992–10002. 1

[7] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, and Yunfan Liu, “Vmamba: Visual state space model,” Advances in Neural Information Processing Systems, vol. 37, pp. 103031–103063, 2024. 1

[8] Ivan Butakov, Aleksandr Tolmachev, Sofia Malanchuk, Anna Neopryatnaya, Alexey Frolov, and Kirill Andreev, “Information bottleneck analysis of deep neural networks via lossy compression,” in International Conference on Learning Representations, 2024, vol. 2024, pp. 40868–40890. 1

[9] Kenji Kawaguchi, Zhun Deng, Xu Ji, and Jiaoyang Huang, “How does information bottleneck help deep learning?,” in International Conference on Machine Learning. PMLR, 2023, pp. 16049–16096. 1

[10] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao, “Yolov9: Learning what you want to learn using programmable gradient information,” in European Conference on Computer Vision, 2024, pp. 1–21. 1

[11] Yuxuan Cai, Yizhuang Zhou, Qi Han, Jianjian Sun, Xiangwen Kong, Jun Li, and Xiangyu Zhang, “Reversible column networks,” in The Eleventh International Conference on Learning Representations, 2023. 1

[12] Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, and Zhuowen Tu, “Deeply-supervised nets,” in Artificial Intel ligence and Statistics. PMLR, 2015, pp. 562–570. 1

[13] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese, “Taskonomy: Disentangling task transfer learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 3712–3722. 1

[14] Vardan Papyan, Yaniv Romano, and Michael Elad, “Convolutional neural networks analyzed via convolutional sparse coding,” Journal of Machine Learning Research, vol. 18, no. 83, pp. 1–52, 2017. 1, 2

[15] Zhiyang Zhang and Shihua Zhang, “Towards understanding residual and dilated dense neural networks via convolutional sparse coding,” National Science Review, vol. 8, no. 3, pp. nwaa159, 2021. 1, 2

[16] Xiaoxia Sun, Nasser M Nasrabadi, and Trac D Tran, “Supervised deep sparse coding networks for image classification,” IEEE Transactions on Image Processing, vol. 29, pp. 405–418, 2019. 1, 2, 3, 4

[17] Xili Dai, Ke Chen, Shengbang Tong, Jingyuan Zhang, Xingjian Gao, Mingyang Li, Druv Pai, Yuexiang Zhai, Xiaojun Yuan, Heung-Yeung Shum, Lionel Ni, and Yi Ma, “Closed-loop transcription via convolutional sparse coding,” in Conference on Parsimony and Learning. 2024, vol. 234 of Proceedings of Machine Learning Research, pp. 570–589, PMLR. 2

[18] Mingyang Li, Pengyuan Zhai, Shengbang Tong, Xingjian Gao, Shao-Lun Huang, Zhihui Zhu, Chong You, Yi Ma, et al., “Revisiting sparse convolutional model for visual recognition,” Advances in Neural Information Processing Systems, vol. 35, pp. 10492–10504, 2022. 2, 3, 4

[19] Amir Beck and Marc Teboulle, “A fast iterative shrinkagethresholding algorithm for linear inverse problems,” SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183–202, 2009. 2

[20] Neal Parikh and Stephen Boyd, “Proximal algorithms,” Foundations and Trends in Optimization, vol. 1, no. 3, pp. 127–239, 2014. 2

[21] Sergey Ioffe and Christian Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in International Conference on Machine Learning. PMLR, 2015, pp. 448–456. 3

[22] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2009, pp. 248–255. 3

[23] Alex Krizhevsky, Geoffrey Hinton, et al., “Learning multiple layers of features from tiny images,” 2009. 3, 4

[24] Dan Hendrycks and Thomas Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” in International Conference on Learning Representations, 2019. 4