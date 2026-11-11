# Evidential Deep Learning with conditional entropy for glaucoma detection

<sub>Hai</sub> <sub>Siong</sub> <sub>Tan</sub>1[0000-0002-6926-1239] <sub>and</sub> <sub>Kuancheng</sub> <sub>Wang</sub>2

<sup>1</sup> Gryphon Center for Artificial Intelligence and Theoretical Sciences, Singapore haisiong.tan@gryphonai.com.sg

2 Electrical and Computer Engineering Department, University of California San Diego, CA, USA

Abstract. We propose a novel conditional entropy function as a basis for quantifying predictive uncertainty in Evidential Deep Learning (EDL) for glaucoma detection from retinal fundus images. Using the EyePACS-AIROGS-light-V2 and RIM-ONE-DL datasets for validation, we found that our proposed EDL model regularized by conditional entropy function produced uncertainty estimates with substantially lower calibration errors than Deep Ensembles, MC Dropout, post-hoc calibration and the conventional EDL framework based on Dempster uncertainty. To further probe our models’ interpretability, we analyzed Grad-CAM heatmaps using a focus index representing spatial concentration. Compared with the other uncertainty quantification methods, EDL models exhibited greater separation between the index distributions of correctly and incorrectly classified fundus images. In particular, false negatives were associated with localized Grad-CAM activations that were displaced from the optic disc, consistent with its established clinical importance in glaucoma assessment. Overall, our findings suggest that our proposed EDL framework for glaucoma detection provides not only enhanced uncertainty calibration but also admits a more physiologically consistent interpretability.

Keywords: Uncertainty quantification · glaucoma detection

## 1 Introduction

This manuscript is devoted to the subject of using neural network models to recognize glaucoma – a group of ocular conditions characterized by progressive damage to the optic nerve – from retinal fundus images [3]. The efectiveness of using deep learning techniques for glaucoma detection has been explored primarily through the use of transfer learning [6, 24], where established models trained on large database such as ImageNet are used and their weights adjusted to the smaller fundus image datasets during model training. Past works in these areas indicated that using neural network models for these purposes is efective as a complementary method for glaucoma diagnosis, and more generally other eye diseases such as age-related macular degeneration [22] and diabetic retinopathy [2].

A major theme of our work here lies in uncertainty quantification (UQ). To our knowledge, the majority of literature on neural network-assisted glaucoma detection has been focused on the comparative performances of various models defined with regards to accuracy-related metrics such as AUROC, etc. [6, 24], but lacking analysis of the reliability of model predictions. In a safety-critical application like glaucoma screening, it would be ideal for model’s predictions to be accompanied by some measure of their uncertainties or level of model confidence, so that users are aware of some measure of the statistical reliability behind each image classification. For any model prediction associated with a relatively large uncertainty, physicians would be motivated to trust it to a lesser extent and rely more on their own manual assessment. Having a sense of how confident the model is in predicting glaucoma from the fundus image thus enables a healthier reliance on the model. Ideally, the model-generated probability of the image indicating glaucoma should also be consistent empirically with its mean accuracy, so that we can trust the probabilistic output and potentially leverage it for tracking rate of disease progression. Thus, in this work, our primary metric is the expected calibration error [18] which has been an essential measure of UQ in various areas of machine learning [16].

Within the landscape of UQ methods, our study compares among four main frameworks: Evidential Deep Learning (EDL) [28], Monte Carlo Dropout [12], post-hoc temperature scaling and Deep Ensemble [21], leaving a more extensive investigation for future work. For EDL, we propose a novel conditional entropy function that can be used to quantify predictive uncertainty instead of Dempster uncertainty in standard EDL formalism. This is essentially derived in the spirit of Bayesian statistics by marginalizing the Shannon entropy with respect to the prior density function. We found that when EDL model training was regularized by our entropy function, it led to superior performance compared to the standard formalism and other UQ frameworks. Theoretically, it amounts to a reinterpretation of EDL as an UQ technique that can be solely understood using conventional statistical and Bayesian principles without alluding to Dempster-Shafer theory of evidence [8, 29]. Our paper is organized as follows. In Sec. 2, we describe our novel entropy function, its associated regularizer term and various aspects of model implementation. Sec. 3 presents the main results of our experiments which are summarized in Table 1. We discuss clinical implications of our results in Sec. 4 before concluding in Sec. 5. The Appendix provides a brief analysis of the asymptotic behavior of the entropy function. We hope that our work here contributes towards robust clinical deployment of A.I.-enabled glaucoma detection from fundus images.

## 2 Methodology

## 2.1 On datasets

The two retinal image datasets that we have chosen for model training for this work are representative of two major classes of fundus images - (i)full fundus that presents the complete, wide-angle picture of the retina, macula and blood vessels, and (ii)cropped fundus that is zoomed onto the neighborhood of the optic nerve head, including the optic disc region.

1. EyePACS-AIROGS-light-v2 set: This is a machine-learning-ready fullfundus glaucoma dataset curated in [20, 30] using a balanced subset of standardized fundus images from the Rotterdam EyePACS AIROGS set [31]. Image sampling and standardized preprocessing techniques were applied to the original Rotterdam dataset for more convenient benchmarking with reduced dataset size. This dataset is split into training, validation, and test folders which contain 4000 (84%), 385 (8%), and 385 (8%) fundus images in each class respectively.

2. The RIM-ONE-DL image dataset [11] consists of 313 fundus images from normal subjects and 172 images from patients with glaucoma. These cropped fundus pictures were collected in three Spanish hospitals and cropped squarely around the head of the optic nerve using the same proportionality criterion. For benchmarking purposes, the authors of [26] have performed a specific train/test split that contain 339 (70%) and 146 (30%) fundus images. For the validation dataset used in our hyperparameter ablation experiments, we further split the testing dataset into two.

Apart from these two main datasets, we also leveraged two well-known retinal fundus dataset: (i)the CRFO dataset of [14] and (ii)the ACRIMA dataset of [10] to assess the efectiveness of using uncertainty estimates for identifying out-of-distribution (OOD) images. The CRFO dataset contains 79 full-fundus images (containing 48 glaucoma cases) while the ACRIMA dataset contains 705 cropped images (containing 396 glaucoma cases). These datasets exhibit systematic distributional shifts in the LAB luminance channel relative to EyePACS and RIM-ONE, respectively. We leveraged these distributional shifts to assess OOD detection performance using the AUROC computed from the uncertainty estimates of diferent models.

## 2.2 On Evidential Deep Learning

A fundamental idea behind Evidential Deep Learning (EDL) as proposed in the seminal work [28] is the specification of a prior distribution for the discrete probability distribution assumed to underlie the classification task. For a binary class problem, this is the Bernoulli distribution and in EDL, it is assumed to have Beta distribution as its prior, of which density function reads

$$
\pi \left( p _ { 1 } , p _ { 2 } | \alpha , \beta \right) = \frac { p _ { 1 } ^ { \alpha - 1 } p _ { 2 } ^ { \beta - 1 } } { B ( \alpha , \beta ) } , B ( \alpha , \beta ) \equiv \frac { { \cal T } ( \alpha ) { \cal T } ( \beta ) } { { \cal T } ( \alpha + \beta ) } ,\tag{1}
$$

where $p _ { i }$ denotes the class probability with the constraint $p _ { 1 } + p _ { 2 } = 1$ $\Gamma ( \alpha )$ denotes the Gamma function and $\alpha , \beta \geq 1$ are prior parameters that are used to express various quantities of interest. For example, the mean probability is obtained after integrating over the prior density

$$
\langle p _ { j } \rangle = \int _ { 0 } ^ { 1 } d p _ { j } p _ { j } \frac { p _ { 1 } ^ { \alpha - 1 } p _ { 2 } ^ { \beta - 1 } } { B ( \alpha , \beta ) } = \frac { \alpha _ { j } } { \alpha + \beta } , j = 1 , 2 .\tag{2}
$$

A similar approach is used for deriving a suitable loss function, for example integrating the standard cross-entropy loss or the mean-squared-error loss over the Beta prior yields eqns. 4 and 5 of [28]. As for the predictive uncertainty, authors of the original EDL framework [28] alluded to Dempster-Shafer theory [8, 29] to assert that the quantity $u _ { d } = 2 / ( \alpha + \beta )$ represents uncertainty – in the sense that we interpret each of the two bracketed terms in the following equality as measures of the amount of data support (or ‘belief mass’ in the language used in [28]) in favor of the classification.

$$
u _ { d } + \left( { \frac { \alpha - 1 } { \alpha + \beta } } \right) + \left( { \frac { \beta - 1 } { \alpha + \beta } } \right) = 1 .\tag{3}
$$

Although Dempster-Shafer Theory [8, 29] was cited for the interpretational postulate underlying eqn. (3), apart from semantics and literal definitions, we are unaware of any conventional information-theoretic justification for taking $u _ { d }$ as a pure measure of uncertainty. In this work, we allude to an entropy-based definition of uncertainty instead. Similar to how the mean probabilities and loss functions formulated in [28] were derived from integrating functions of the class probabilities over their Beta priors, a natural notion of uncertainty can be derived in the Bayesian sense by marginalizing the uncertainty measure with respect to the prior density.

## 2.3 Deriving a coarse-grained entropy for EDL models

One common measure of uncertainty is the Shannon entropy. Denoting the probability of the $k ^ { \mathrm { t h } }$ image being classified as glaucomatous by $p _ { k }$ , the Shannon entropy representing probabilistic uncertainty associated with the binary classification is the binary entropy function defined as

$$
S = - \sum _ { k } \left( p _ { k } \log p _ { k } + \left( 1 - p _ { k } \right) \log \left( 1 - p _ { k } \right) \right) .\tag{4}
$$

In the original EDL framework [28], the loss function was derived by averaging the mean-squared-error term over p taking the prior for $p$ to be the Beta distribution parametrized by $\alpha , \beta$ . In a similar spirit, we consider the average binary entropy $S _ { c }$ obtained by marginalizing (4) with respect to the prior density.

$$
S _ { c } = - \sum _ { k } \int _ { 0 } ^ { 1 } d p _ { k } \left( p _ { k } \log p _ { k } + ( 1 - p _ { k } ) \log ( 1 - p _ { k } ) \right) \frac { p _ { k } ^ { \alpha - 1 } ( 1 - p _ { k } ) ^ { \beta - 1 } } { B ( \alpha , \beta ) } ,\tag{5}
$$

where $\begin{array} { r } { B ( \alpha , \beta ) \equiv \frac { \Gamma ( \alpha ) \Gamma ( \beta ) } { \Gamma ( \alpha + \beta ) } } \end{array}$ is the Beta function. Using the integration formula [1]

$$
\int _ { 0 } ^ { 1 } d x \log ( x ) x ^ { a } ( 1 - x ) ^ { b } = \frac { T ( 1 + a ) T ( 1 + b ) ( H _ { a } - H _ { 1 + a + b } ) } { T ( 2 + a + b ) } , a > - 1 , b > - 1 ,
$$

where $H _ { a }$ denotes harmonic numbers analytically continued to the real plane, we obtain

$$
S _ { c } = \sum _ { k } \psi ( \alpha _ { k } + \beta _ { k } + 1 ) - \frac { \alpha _ { k } \psi ( \alpha _ { k } + 1 ) + \beta _ { k } \psi ( \beta _ { k } + 1 ) } { \alpha _ { k } + \beta _ { k } } , \psi ( x ) \equiv T ^ { \prime } ( x ) / { T ( x ) } ,\tag{6}
$$

where $\psi$ is the digamma function.<sup>3</sup> Our coarse-grained entropy function admits a natural interpretation in classical information theory – that of conditional entropy which is fundamentally defined [7] as the expected value of entropies of distributions, averaged over the conditioning random variable. In our binary classification context, the parameter $p$ of the Bernoulli distribution is lifted to be a continuous, bounded variable with the Beta distribution as the prior density. Taking this to be the conditioning random variable, the form of eqn. (5) can be identified with that of conditional entropy $H ( Y | \mathcal { P } )$ with Y being the discrete binary-valued random variable with a Bernoulli distribution conditioned on a continuous, bounded variable $\mathcal { P }$ equipped with a Beta distribution. Each term of the sum in (5) can be written as

$$
H ( Y | \mathcal { P } ) = \int _ { 0 } ^ { 1 } d p \pi ( p ) H ( Y | \mathcal { P } = p ) = \int _ { 0 } ^ { 1 } d p \pi ( p ) \left[ p \log p + ( 1 - p ) \log ( 1 - p ) \right] ,\tag{7}
$$

where $\begin{array} { r } { \pi ( p ) = \frac { p ^ { \alpha - 1 } ( 1 - p ) ^ { \beta - 1 } } { B ( \alpha , \beta ) } } \end{array}$ is the Beta distribution. Henceforth, we call $S _ { c }$ the conditional entropy. By inspection, it is positive-definite since the binary entropy function $H ( Y | p )$ is non-negative and $S _ { c }$ is defined by integrating $H ( Y | p )$ with respect to another normalizable, non-negative density function $\pi ( p )$

Following [28], we take $\alpha , \beta \geq 1$ and in this domain, the conditional entropy $S _ { c }$ satisfies

$$
0 < S _ { c } ( \alpha , \beta ) < \log 2 , \operatorname * { l i m } _ { \alpha = \beta \atop \alpha  \infty } S _ { c } ( \alpha , \beta ) = \log 2 .\tag{8}
$$

This supremum is approached along $\alpha = \beta \to \infty$ . To interpret this maximal value, we note that the binary entropy function is maximized by $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ . The limit $\alpha = \beta \to \infty$ corresponds to the Beta prior collapsing to a singular δ- function at $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ , yielding the maximum value of the average binary entropy $S _ { c } .$ . Away from this singular limit, other finite values of $( \alpha , \beta )$ yield lower average entropies since they characterize a probabilistic spread of p instead of just taking it to be $\frac { 1 } { 2 }$ . Fig. 1 is a contour plot of $S _ { c }$ in the domain $( \alpha , \beta ) \in [ 1 , \infty ) \times [ 1 , \infty )$ In the Appendix, we describe the asymptotic region in more detail.

For $\begin{array} { r } { \alpha = \beta = 1 , S _ { c } = \frac { 1 } { 2 } } \end{array}$ – the value of the average binary entropy when the Beta prior for p is equivalently the uniform distribution $\mathcal { U } ( 0 , 1 )$ for these values of $\alpha , \beta .$ . The infimum of 0 is attained along $\beta = 1 , \alpha \to \infty { \mathrm { ~ o r ~ } } \alpha = 1 , \beta \to \infty$ which pertain to the Beta prior reduced to singular δ-function localized at $p = 1$ or $0$ respectively. In summary, the conditional entropy $S _ { c }$ generalizes the binary entropy function by computing the average entropy of the stochastic distribution of $p$ as defined by the Beta prior.

## 2.4 A conditional entropy-based regularizer for model training

To integrate conditional entropy as a measure of predictive uncertainty in model training, a common approach is to regularize model training such that the uncertainty associated with the outputs are consistent with eqn. (6). Similar in spirit to the original seminal works of EDL [4, 28], we use a loss term that increases our measure of uncertainty for erroneous classifications. Since we would like the conditional entropy to be larger for erroneous predictions, we consider the regularizer

![](images/e3eb9328916a1e0bee9ef5b6964e8186a3a03e1a400e9896baacfd4821b9ef06.jpg)  
Fig. 1. Contour plot of the conditional entropy function $S _ { c } ,$ , with darker regions representing lower values. The red dashed line is $\alpha ~ = ~ \beta ,$ along which the global supremum of log 2 is approached at infinity. In the Appendix, we describe the asymptotic region in more detail.

$$
\mathcal { L } _ { c } = \delta ^ { 2 } \left( 1 - \frac { 1 } { \log 2 } S _ { c } ( \alpha , \beta ) \right) , \delta \equiv y _ { t } ( 1 - y _ { p } ) + ( 1 - y _ { t } ) y _ { p } .\tag{9}
$$

where δ represents a diferentiable error term with $y _ { t } , y _ { p }$ being ground truth and predicted classes respectively. We choose the loss term to be quadratic in δ so that it is more sensitive to the wrongly predicted samples. In eqn. (9), the log 2 factor normalizes the entropy term so that it is bounded within (0,1). Critically, using the regularizer (9) implements our use of conditional entropy as predictive uncertainty, since $S _ { c } ( \alpha , \beta )$ is now invoked directly in model training.

## 2.5 On the implementation of EDL models and other UQ frameworks

MobileNetV3-Small [17] from the Torchvision library [25] was used as the backbone model. The network was initialized with ImageNet-pretrained weights for transfer learning. As an adjustment module for our glaucoma classification task, the classifier head was replaced with a fully connected multi-layer perceptron (MLP) producing a single output logit which was subsequently passed through a sigmoid activation function to generate probabilities in the baseline model. The MLP was picked to be 3-layer with the number of neurons being (256, 128, 1) for the EyePACS dataset and (64, 32, 1) for the RIM-ONE dataset and model was trained using the binary cross entropy loss. For EDL models, a slightly diferent classification head was used, consisting of a 4-layer MLP with the number of neurons being (256, 128, 64, 2) for the EyePACS dataset and (128, 64, 32, 2) for the RIM-ONE dataset. The final two-unit output layer were taken to be logits which were then passed through a softplus function to generate the evidence vectors $e _ { j }$ defined as $\alpha _ { j } = 1 + e _ { j }$ . The loss function in our experiments can be generally expressed as

$$
\mathcal { L } _ { E D L } = \lambda _ { c e } \mathcal { L } _ { c e } + \lambda _ { c } \mathcal { L } _ { c } + \lambda _ { k l } \mathcal { L } _ { k l } ,\tag{10}
$$

where $\mathcal { L } _ { c }$ is defined in eqn. (9), and $\mathcal { L } _ { c e } , \mathcal { L } _ { k l }$ are the cross-entropy and KLdivergence regularizer of [28]<sup>4</sup>. Models in the standard EDL model are defined as setting $\mathcal { L } _ { c } = 0$ , whereas our proposed model is associated with setting $\mathcal { L } _ { K L } = 0$ since this regularizer is defined with respect to Dempster uncertainty instead. The coeficients $\left( \left\{ \lambda _ { c e } , \lambda _ { c } , \lambda _ { k l } \right\} \right)$ , like the number of hidden neurons per layer of the MLP adjustment heads, were picked after performing ablation experiments. After a grid search involving four orders of magnitude (0.01 to 100) for each of $\left( \lambda _ { C E } , \lambda _ { c } , \lambda _ { k l } \right)$ , the optimal values were $\lambda _ { c e } = \{ 1 . 0 , 0 . 1 \} , \lambda _ { c } = \{ 0 . 1 5 , 0 . 1 \} , \lambda _ { k l } =$ {1.0, 0.1} for EyePACS-and RIM-ONE-trained models respectively. All models were trained with Adam optimizer and a learning rate of $\bar { 1 0 } ^ { - 4 }$ for 50 epochs at which a relative tolerance $\lesssim 0 . 0 1$ was attained.

In this work, we compared the EDL methodology against other standard UQ frameworks – post-hoc temperature scaling [13], Deep Ensemble [21] and MC Dropout [12]. Each of the 10 neural networks in the ensemble shared the same MobileNet architecture (equipped with the $\mathrm { M L P }$ adjustment module for classifier head described earlier) with random weight initialization performed via the He-uniform initializer [15] to induce diferent initial weights. For MC Dropout model, we added a dropout layer after each hidden layer of the MLP head attached to the baseline model, and 20 forward passes were used for the Monte Carlo simulation. For post-hoc temperature scaling of the baseline models, the output logits were first cached on the validation data, after which a scalar temperature was optimized by minimizing the Brier score of the resulting sigmoid probabilities. The optimized temperature<sup>5</sup> was used to rescale the logits prior to computing the calibrated probabilities. Trained on an A100 GPU, we found that generally, EyePACS-based EDL models required 50 minutes for training which was about twice the duration required for RIM-ONE-based models.

## 2.6 Some comments on evaluation metrics

The expected calibration error (ECE) is a measure of how well the model’s probabilistic predictions match empirical ones. Splitting the dataset into a number of bins, it is defined as a weighted mean of the absolute diference between the bin-averaged accuracy and confidence [18], i.e.

$$
E C E \equiv \sum _ { \mu = 1 } ^ { M } \frac { | B _ { \mu } | } { N } \left| f _ { a c c } ( B _ { \mu } ) - f _ { p r e d } ( B _ { \mu } ) \right| ,
$$

where N is the number of datapoints, $B _ { \mu }$ are the M bins, $f _ { a c c } , f _ { p r e d }$ are the average accuracies and predicted probabilities in each bin. For classification tasks, these quantities are $\begin{array} { r } { f _ { a c c } = \frac { 1 } { | B _ { \mu } | } \sum _ { i \in B _ { \mu } } \delta _ { \hat { y } _ { i } , y _ { i } } , f _ { p r e d } = \frac { 1 } { | B _ { \mu } | } \sum _ { i \in B _ { \mu } } c _ { i } , } \end{array}$ , where $y _ { i } , \hat { y } _ { i }$ are the predicted and ground-truth class labels respectively, with $c _ { i }$ being the probabilistic confidence corresponding to the predicted class, $\ \mathrm { i } . \mathrm { e } . \ c _ { i } =$ arg max<sub>k</sub> $p _ { i } ^ { k }$ where $p _ { i } ^ { k }$ are the softmax class probabilities for category k predicted for the $i ^ { \mathrm { t h } }$ datapoint. We evaluated ECE using both equal-width $\left( \equiv \mathrm { E C E } _ { w } \right)$ and equal-quantile binning $\left( \equiv \mathrm { E C E } _ { q } \right)$ schemes with 10 bins.

To investigate the spatial regions that influenced the model’s predictions, we employed Gradient-weighted Class Activation Mapping (Grad-CAM) [27] to generate class-discriminative activation maps. To describe the concentration of these activations, we computed a focus index, defined as the normalized complement of the efective number of activated pixels, with larger values indicating more localized attention. We first normalize the Grad-CAM values, converting them to an empirical probability distribution $g _ { i }$ over all pixels of each fundus image. This is used to define an efective number of activated pixels (see e.g. [5]) $N _ { e f f } \equiv 1 / \sum _ { i } g _ { i } ^ { 2 }$ , which is commonly used to quantify localization and sometimes known as the inverse participation ratio [5]. To obtain an intuitive measure of spatial concentration, we define an index $\begin{array} { r } { \dot { \boldsymbol { F } } \equiv 1 - \frac { N _ { e f f } } { N _ { t } } } \end{array}$ , where $N _ { t }$ is the total number of pixels. Larger values of F indicate more spatially concentrated Grad-CAM activations, while smaller values indicate a more uniformly spread distribution.

## 3 Results

In Table 1, we collect various metrics characterizing model performances. In the aspect of AUC, various models achieved an average value of 0.96 (EyePACS) and 0.97 (RIM-ONE). For accuracy, all models attained values $\gtrsim 0 . 9 0$ . The EDL models trained with $\mathcal { L } _ { c }$ attained the highest accuracy for both datasets, and highest AUC for the RIM-ONE dataset. These diferences among them were rather minor (≲ 2%) and we regard various models as achieving similar accuracy performance for both datasets. While the models exhibited similar accuracy, the ECE metrics revealed more pronounced diferences among them, with the baseline model, temperature scaling and MC Dropout being clearly inferior to other models for both datasets. Reliability diagrams for both datasets are shown in Fig. 2. EDL models trained with $\mathcal { L } _ { c }$ attained the lowest ECE for both datasets, and the lowest Brier scores. Deep Ensemble method attained a similar degree of calibration for the RIM-ONE dataset and from Fig. 2, it was relatively overconfident compared to EDL models, whereas MC Dropout models appeared to be the most overconfident ones. In the aspect of uncertaintybased OOD detection, we found the model ranking by this metric to be datasetdependent. Nevertheless, our proposed EDL model achieved the highest OOD-AUROC for EyePACS and, together with Deep Ensemble, for RIM-ONE.

While small values of the ECE metrics formally quantify how robust the probabilistic outputs are, another useful feature of the uncertainty distribution is that it may provide some degree of discrimination between correct and wrong predictions. In Fig. 2, we plot the empirical cumulative distribution functions (ECDF) versus the uncertainty estimates for various models trained on each dataset. They depict how uncertainty distributions for the sets of wrong predictions were all markedly separate from those for the correct sets, with the separation stronger for experiments involving the larger EyePACS dataset.

Table 1. Abbrev – Acc: accuracy; AUC: area under ROC curve; $\operatorname { E C E } _ { q } ,$ E $\mathrm { C E } _ { w } \colon$ mean ECE metrics that are quantile-based and fixed width-based; $B _ { s } \colon$ Brier score; $W _ { h } , W _ { g } \ -$ Wasserstein distance between Grad-CAM indices of correct and wrong predictions for healthy and glaucomatous eye images; OOD: the AUROC score for identifying OOD images based on uncertainty; EDL(Dempster): EDL models trained with $\mathcal { L } _ { k l }$ of [28]; EDL(w/o reg.): models trained with $\mathcal { L } _ { c } = \mathcal { L } _ { k l } = 0$ in the loss function.  
EyePACS dataset-trained model performance
<table><tr><td rowspan=1 colspan=1>Models</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>ECEq</td><td rowspan=1 colspan=1>ECEw</td><td rowspan=1 colspan=1> $\overline { { B _ { s } } }$ </td><td rowspan=1 colspan=1> $\overline { { W _ { g } } }$ </td><td rowspan=1 colspan=1> $\overline { { W _ { h } } }$ </td><td rowspan=1 colspan=1>OOD</td></tr><tr><td rowspan=1 colspan=1>(Baseline) MNetV3</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.054</td><td rowspan=1 colspan=1>0.063</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.038</td><td rowspan=1 colspan=1>0.050</td><td rowspan=1 colspan=1>0.82</td></tr><tr><td rowspan=1 colspan=1>Temp. Scaling</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.047</td><td rowspan=1 colspan=1>0.050</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.050</td><td rowspan=1 colspan=1>0.038</td><td rowspan=1 colspan=1>0.82</td></tr><tr><td rowspan=1 colspan=1>MC Dropout</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.038</td><td rowspan=1 colspan=1>0.044</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.103</td><td rowspan=1 colspan=1>0.043</td><td rowspan=1 colspan=1>0.79</td></tr><tr><td rowspan=1 colspan=1>Deep Ensemble</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>0.030</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.105</td><td rowspan=1 colspan=1>0.138</td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=1>EDL(w/o reg.)</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.029</td><td rowspan=1 colspan=1>0.033</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.73</td></tr><tr><td rowspan=1 colspan=1>EDL(conditional)</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.87</td></tr><tr><td rowspan=1 colspan=1>EDL(Dempster)</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>0.026</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.51</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.75</td></tr></table>

RIM-ONE dataset-trained model performance
<table><tr><td rowspan=1 colspan=1>Models</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>ECEq</td><td rowspan=1 colspan=1>ECEw</td><td rowspan=1 colspan=1> $\overline { { B _ { s } } }$ </td><td rowspan=1 colspan=1> $\overline { { W _ { g } } }$ </td><td rowspan=1 colspan=1> $\overline { { W _ { h } } }$ </td><td rowspan=1 colspan=1>OOD</td></tr><tr><td rowspan=1 colspan=1>(Baseline) MNetV3</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.048</td><td rowspan=1 colspan=1>0.062</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.60</td></tr><tr><td rowspan=1 colspan=1>Temp. Scaling</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.056</td><td rowspan=1 colspan=1>0.056</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.66</td></tr><tr><td rowspan=1 colspan=1>MC Dropout</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.045</td><td rowspan=1 colspan=1>0.057</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.68</td></tr><tr><td rowspan=1 colspan=1>Deep Ensemble</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.027</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.70</td></tr><tr><td rowspan=1 colspan=1>EDL (w/o reg.)</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.040</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.68</td></tr><tr><td rowspan=1 colspan=1>EDL(conditional)</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.021</td><td rowspan=1 colspan=1>0.033</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.70</td></tr><tr><td rowspan=1 colspan=1>EDL(Dempster)</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.032</td><td rowspan=1 colspan=1>0.036</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.69</td></tr></table>

Beyond predictive accuracy and uncertainty-based metrics, we examined the Grad-CAM focus indices to characterize the spatial behavior of various models. Evidential models consistently exhibited greater separation between correct and incorrect predictions in terms of the focus index distributions than the other UQ approaches, as reflected by the Wasserstein distances between the corresponding distributions tabulated in Table 1. Fig. 3 shows the violin plots of focus indices for three classes of models. While Deep Ensemble and MC Dropout produced substantially overlapping focus index distributions – a trend also displayed by the baseline and post-hoc temperature scaling, the EDL models demonstrated clearly more pronounced index separations between correct and incorrect predictions. This feature was found to characterize general EDL models, regardless of whether they were trained with $\mathcal { L } _ { c }$ or $\mathcal { L } _ { k l }$

![](images/2b7c87021df0226a3bce8ffddd38e95d7c4542296d81ebfdaa68ec543f2cb1bf.jpg)  
Fig. 2. top: ECDF of uncertainty estimates for correct (solid) and wrong (dashed) classifications; bottom: Reliability diagrams using equal-quantile binning (10 bins).

## 4 Discussion

The EDL models regularized by conditional entropy appeared to demonstrate consistent superiority in uncertainty calibration (as measured by $\operatorname { E C E } _ { q }$ and $\mathrm { E C E } _ { w } )$ . Other models were weaker in their degrees of uncertainty calibration being mostly overconfident as apparent in Fig. 2. In the absence of any regularizer, the EDL models did not exhibit advantages over standard ones trained using a regularizer $\mathcal { L } _ { k l }$ of [28] that is based on Dempster uncertainty. The ECDF plots in Fig. 2 show the uncertainty-aware models to display various degrees of separation in the uncertainty distributions between correct and wrong predictions. In the aspect of Grad-CAM indices, we found that EDL models exhibited a property quite distinct from Deep Ensemble, temperature scaling and Monte Carlo Dropout – the focus index distributions pertaining to correct and wrong predictions were clearly separable for both datasets. In particular, within the set of glaucomatous images, the indices for false negatives spread around higher medians relative to true positive images.

In Fig. 4, we show two representative samples of false negative classification. The heatmaps appeared to show that for the EDL model (for illustration, we chose the one trained with $\mathcal { L } _ { c } )$ , the activation foci was visibly misplaced from the critical optic disc region yet much more localized compared to those of Deep Ensemble and MC Dropout. For the RIM-ONE image in ${ \mathrm { F i g . } }$ 4 where the fundus image was already cropped around the optic disc region (conventionally an essential area to examine for glaucoma), the misplaced attention was still contained within the optic disc and overlapping with the neuro-retinal rim in contrast to the much more difuse model attention generated by the other two models. For the EyePACS image in Fig. 4, the misplaced attention of the EDL model was relatively more localized and the heatmap’s misplaced attention was consistent with the spatial deviation from the optic disc neighborhood. This feature can be used by the clinician as one mechanism to help flag potential false negatives by leveraging on our understanding of glaucoma symptoms being mainly localized within the optic disc. In contrast, we noted that the attention areas of all other models other than EDL ones generally overlapped with the optic disc region. We found this distinguishing feature of misplaced attention mirroring false negative cases to characterize EDL models, regardless of whether they were trained with $\mathcal { L } _ { c }$ or $\mathcal { L } _ { k l }$

![](images/9f5062d581a32f9e3333cadaac3f4050ce96ba808ae0c6615a802b955c13184b.jpg)

![](images/41ceb58fc04a13782a1bcb8e5075e3c20eb1572f2015dd14fe10ec9eb5a7ddda.jpg)

![](images/835c1e0ce1cd460752e4041d4616029ce636bfe51c90d5a2af583a5386cad6ec.jpg)  
Fig. 3. Violin plots of the focus index characterizing Grad-CAM heatmaps of EDL, Deep Ensemble, and MC Dropout models trained on the EyePACS dataset. The EDL model here was trained with $\mathcal { L } _ { c } ,$ although the distinct separation between correct and wrong predictions also characterized other EDL models (see Wasserstein distances of Table 1). Similar relative trends were also observed for RIM-ONE-trained models.

In this work, we have only used MobileNet as the backbone architecture. It would be important to examine other models, such as vision transformers, to study how universal the efectiveness of conditional entropy is. In the clinical operation setting, an uncertainty threshold associated with the minimal levels of sensitivity and precision can potentially be used to characterize model reliability. Thus, various types of accuracy-uncertainty threshold plots would be useful indicators to study prior to clinical implementation<sup>6</sup>. Another future direction would be to study if it is possible to characterize the nature of conditional entropy more precisely, for example whether it is representative of aleatoric or epistemic uncertainty [9, 19]. In [23], a class of neural networks called Prior Networks were proposed which also leveraged the conditional entropy we derived here. We note that these models were trained using loss and regularization terms significantly diferent from ours, and that although mentioned in their work (see Appendix C), the conditional entropy was not examined directly in their experiments. It would be interesting to study the other notions of entropy presented in [23] in our context.

![](images/0f6ebc7300d350ebd332b5b72a66044dcd19b5c6c38d7b241c965d141e83ff14.jpg)  
Fig. 4. Grad-CAM visualizations for representative false negative fundus (leftmost) images of each dataset (top: RIM-ONE-trained models, bottom: EyePACS-trained models). In the top row, a dashed white contour delineates the optic disc.

## 5 Conclusion

Our study of uncertainty-aware models for glaucoma classification has shown that, when built upon a pre-trained MobileNet backbone, EDL models incorporating our conditional entropy function (Eq. (6)), as implemented through the corresponding regularizer (Eq. (9)), produced uncertainty estimates with substantially lower calibration errors than those obtained using Deep Ensembles, post-hoc temperature scaling, MC Dropout, and traditional EDL models based on Dempster uncertainty. Towards elucidating the interpretability of our models, we have also examined their Grad-CAM heatmaps. Distributions of Grad-CAM focus indices were relatively more discriminatory between correctly and wrongly classified fundus images for EDL models compared to the other UQ methods. In particular, false negatives were associated with localized Grad-CAM activations that were displaced from the optic disc, consistent with its established physiological importance in glaucoma assessment. Overall, our findings suggest that our proposed EDL model based on conditional entropy provides not only enhanced uncertainty calibration, but also admits a more physiologically consistent interpretability, making it a potentially efective model for uncertainty-aware glaucoma detection from fundus images.

Acknowledgments. We express our gratitude to ophthalmologist Dr Claudine Pang of Asia Retina Eye Centre for various medical physiology-related explanations. We also thank the anonymous reviewers of TAAI 2026 for various feedback.

Disclosure of Interests. The author has no competing interests to declare that are relevant to the content of this article.

## A On the asymptotic behavior of conditional entropy $S _ { c }$

In this Appendix, we show that in the asymptotic region $\alpha , \beta \to \infty ,$ , the conditional entropy function can be mapped to the standard binary entropy function. We begin by noting that the domain $( ( \alpha , \beta ) \in [ 1 , \infty ) \times [ 1 , \infty ) )$ can be decomposed into rays of fixed ratio described by $\beta = r \alpha , r > 0$ . We use this decomposition to examine the asymptotic behavior of the conditional entropy function $S _ { c }$ along diferent directions in the $( \alpha , \beta )$ -plane. Along each line with a fixed $r ,$

$$
f _ { r } ( \alpha ) \equiv S _ { c } ( \alpha , r \alpha ) = \psi \left( ( r + 1 ) \alpha + 1 \right) - \frac { 1 } { 1 + r } \left( \psi ( \alpha + 1 ) + r \psi ( r \alpha + 1 ) \right) .\tag{11}
$$

Using lim $. z \to \infty \psi ( z ) \sim \log ( z ) - \frac { 1 } { 2 z }$ , we obtain

$$
\operatorname* { l i m } _ { \alpha \to \infty } f _ { r } ( \alpha ) \sim \log { ( ( r + 1 ) \alpha + 1 ) } - \frac { 1 } { 2 ( ( r + 1 ) \alpha + 1 ) } \sim \overline { { p } } \log { \overline { { p } } } + ( 1 - \overline { { p } } ) \log ( 1 - \overline { { p } } ) ,\tag{12}
$$

where we have defined $\begin{array} { r } { \overline { { p } } = \frac { \alpha } { \alpha + \beta } } \end{array}$ . This is the binary entropy function evaluated at $p = { \overline { { p } } } .$ . Another way of understanding this mapping is to note that the asymptotic region $\alpha , \beta \to \infty$ is where the Beta prior distributions reduce to singular δ- functions centered at various values of ${ \overline { { p } } } .$ To see this, we introduce a scaling factor s to write $\alpha = \tilde { \alpha } s , \beta = \tilde { \beta } s ,$ , and invoke the following known relation [1] between Beta and normal distributions which reads

$$
\operatorname* { l i m } _ { s \to \infty } \mathrm { B e t a } ( \tilde { \alpha } s , \tilde { \beta } s ) \to { \cal N } \left( \frac { \tilde { \alpha } } { \tilde { \alpha } + \tilde { \beta } } , \frac { \tilde { \alpha } \tilde { \beta } } { ( \tilde { \alpha } + \tilde { \beta } ) ^ { 3 } s } \right) .\tag{13}
$$

At any fixed r (and thus $\overline { { p } } )$

$$
\operatorname* { l i m } _ { \alpha = r \beta } \operatorname { B e t a } ( \alpha , \beta ) \to \operatorname* { l i m } _ { \alpha \to \infty } \mathcal { N } \left( \bar { p } , ( 1 - \bar { p } ) \frac { 1 } { \alpha } \right) .
$$

In this limit, there is no stochasticity in $p ,$ and thus the averaged entropy is simply the entropy evaluated at ${ \overline { { p } } } ,$ consistent with what we have shown in eqn. (12).

## References

1. Abramowitz, M., Stegun, I.A.: Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables. Dover Publications, New York, dover reprint edn. (1965)

2. Akhtar, S., Aftab, S., Ali, O., Ahmad, M., Khan, M.A., Abbas, S., Ghazal, T.M.: A deep learning based model for diabetic retinopathy grading. Scientific Reports 15(1), 3763 (2025)

3. Allingham, R.R., Shields, M.B.: Shields’ textbook of glaucoma. Wolters Kluwer/Lippincott Williams and Wilkins Health, Philadelphia, 6th edition / r. rand allingham et al. edn. (2011)

4. Amini, A., Schwarting, W., Soleimany, A., Rus, D.: Deep evidential regression. In: Advances in Neural Information Processing Systems. vol. 33, pp. 14927–14937 (2020)

5. Anderson, P.W.: Absence of difusion in certain random lattices. Physical Review 109(5), 1492–1505 (March 1958). https://doi.org/10.1103/PhysRev.109.1492

6. Camara, J., Neto, A., Pires, I.M., Villasana, M.V., Zdravevski, E., Cunha, A.: Literature review on artificial intelligence methods for glaucoma screening, segmentation, and classification. J Imaging 8(2) (Jan 2022)

7. Cover, T.M., Thomas, J.A.: Elements of Information Theory. Wiley-Interscience, Hoboken, N.J., 2nd edn. (2006)

8. Dempster, A.P.: A generalization of bayesian inference. Journal of the Royal Statistical Society B(10), 205–247 (1968)

9. Depeweg, S., Hernández-Lobato, J.M., Doshi-Velez, F., Udluft, S.: Decomposition of uncertainty in bayesian deep learning for eficient and risk-sensitive learning (2018), https://arxiv.org/abs/1710.07283

10. Diaz-Pinto, A., Morales, S., Naranjo, V., Köhler, T., Mossi, J.M., Navea, A.: Cnns for automatic glaucoma assessment using fundus images: an extensive validation. BioMedical Engineering OnLine 18(1), 29 (2019)

11. Fumero Batista, F.J., Diaz-Aleman, T., Sigut, J., Alayon, S., Arnay, R., Angel-Pereira, D.: Rim-one dl: A unified retinal image database for assessing glaucoma using deep learning. Image Analysis and Stereology 39(3), 161–167 (Nov 2020). https://doi.org/10.5566/ias.2346, https://www.iasiss.org/ojs/IAS/article/view/2346

12. Gal, Y., Ghahramani, Z.: Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In: Balcan, M.F., Weinberger, K.Q. (eds.) Proceedings of The 33rd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 48, pp. 1050–1059. PMLR, New York, New York, USA (20–22 Jun 2016)

13. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: Precup, D., Teh, Y.W. (eds.) Proceedings of the 34th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 70, pp. 1321–1330. PMLR (2017)

14. Hassan, T., Akram, M.U., Nazir, M.N.: A composite retinal fundus and oct dataset with detailed clinical markings of retinal layers and retinal lesions to grade macular and glaucomatous disorders (2021). https://doi.org/10.17632/trghs22fpg.4

15. He, K., Zhang, X., Ren, S., Sun, J.: Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV). pp. 1026–1034 (2015). https://doi.org/10.1109/ICCV.2015.123

16. He, W., Jiang, Z., Xiao, T., Xu, Z., Li, Y.: A survey on uncertainty quantification methods for deep learning (2025), https://arxiv.org/abs/2302.13425

17. Howard, A., Sandler, M., Chu, G., Chen, L.C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q.V., Adam, H.: Searching for mobilenetv3. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) pp. 1314–1324 (2019)

18. Jungo, A., Reyes, M.: Assessing reliability and challenges of uncertainty estimations for medical image segmentation (2019), https://arxiv.org/abs/1907.03338

19. Kendall, A., Gal, Y.: What uncertainties do we need in bayesian deep learning for computer vision? (2017), https://arxiv.org/abs/1703.04977

20. Kiefer, R.: Glaucoma dataset: Eyepacs-airogs-light-v2 (2024), https://www.kaggle.com/dsv/7802508

21. Lakshminarayanan, B., Pritzel, A., Blundell, C.: Simple and scalable predictive uncertainty estimation using deep ensembles. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. p. 6405–6416. NIPS’17, Curran Associates Inc. (2017)

22. Leng, X., Shi, R., Wu, Y., Zhu, S., Cai, X., Lu, X., Liu, R.: Deep learning for detection of age-related macular degeneration: A systematic review and meta-analysis of diagnostic test accuracy studies. PLoS One 18(4), e0284060 (2023)

23. Malinin, A., Gales, M.: Predictive uncertainty estimation via prior networks (2018), https://arxiv.org/abs/1802.10501

24. Meedeniya, D., Shyamalee, T., Lim, G., Yogarajah, P.: Glaucoma identification with retinal fundus images using deep learning: Systematic review. Informatics in Medicine Unlocked 56, 101644 (2025)

25. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S.: Pytorch: An imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 32 (2019)

26. Sanchez, J.L., Sigut, J., Gonzalez-Hernandez, M., Alayon, S., Fumero, F.: RIM-ONE: An open retinal image database for optic nerve evaluation . In: Proceedings of the 26th IEEE International Symposium on Computer-Based Medical Systems. pp. 1–6. IEEE Computer Society, Los Alamitos, CA, USA (Jun 2011). https://doi.org/10.1109/CBMS.2011.5999143, https://doi.ieeecomputersociety.org/10.1109/CBMS.2011.5999143

27. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Gradcam: Visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV). pp. 618–626 (2017)

28. Sensoy, M., Kaplan, L., Kandemir, M.: Evidential deep learning to quantify classification uncertainty (2018), https://arxiv.org/abs/1806.01768

29. Shafer, G.: A Mathematical Theory of Evidence. Princeton University Press, Princeton, NJ (1976)

30. Steen, J., Kiefer, R., Ardali, M., Abid, M., Amjadian, E.: Standardized and openaccess glaucoma dataset for artificial intelligence applications. Investigative Ophthalmology & Visual Science 64(8), 384 (2023)

31. de Vente, C., Vermeer, K.A., Jaccard, N., Wang, H., Sun, H., Khader, F., et al.: Airogs: Artificial intelligence for robust glaucoma screening challenge. arXiv preprint arXiv:2302.01738 (2023)