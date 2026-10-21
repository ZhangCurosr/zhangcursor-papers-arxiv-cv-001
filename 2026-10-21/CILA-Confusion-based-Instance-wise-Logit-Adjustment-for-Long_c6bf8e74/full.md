# CILA: Confusion-based Instance-wise Logit Adjustment for Long-Tailed Recognition

Jiyoung Lim Sungkyunkwan University maya0707@skku.edu

Jee-Hyong Lee Sungkyunkwan University john@skku.edu

## Abstract

Real-world datasets are inherently class-imbalanced, leading models to favor majority classes and struggle with minority ones. Existing logit adjustment methods address this by modifying logits based on class frequencies, but their class-wise adjustments ignore instance-wise confusion and often suppress informative signalsfrom confusing nontarget classes. This results in insufficient learning of such classes, increased prediction uncertainty, and limited generalization.

To address this, we propose two complementary methods. First, Confusion-based Instance-wise Logit Adjustment (CILA) adaptively adjusts logits based on instance-wise confusion between the target and non-target classes, restoring informative gradients that class-wise adjustment suppresses. Second, Batch-adaptation Noise-Injection (BNI) perturbs the target logit during training, extending the confusion-aware update to correctly classified but near-boundary samples that the penalty alone cannot reach. CILA is a loss-agnostic plug-in module that can be seamlessly integrated into existing long-tailed learning methods, and consistently improves performance across multiple benchmarks. Our code will be available at https://github.com/maya0395/CILA.

## 1. Introduction

Deep Neural Networks (DNNs) have achieved remarkable success, largely driven by the availability of well-curated and balanced datasets [17, 26]. However, real-world data often exhibit naturally imbalanced class distributions [9, 21, 32], where a few head classes contain abundant samples, while most tail classes have only limited data. Training models on such long-tailed datasets poses significant challenges, as models tend to focus heavily on head classes while overlooking tail classes. Consequently, they become overly biased toward head classes, struggle to generalize to tail classes, and suffer from degraded overall performance.

To address the challenges of long-tailed distributions, various strategies have been proposed. Data re-balancing techniques [2, 3, 5, 7, 11, 12, 18, 22, 27, 28, 31] mitigate class imbalance through resampling or reweighting. However, such methods often introduce instability and overfitting, especially under severe imbalance conditions[7, 14, 34]. In contrast, logit adjustment techniques [4, 16, 23, 25, 29, 31, 35] address the problem from a logit imbalance perspective by adding class-dependent adjustment terms to the logits. This suppresses head class logits and amplifies tail classes, alleviating logit imbalance.

While effective, existing LA methods have a fundamental limitation: they apply class-dependent adjustments uniformly to all logits. Such uniform adjustment reduces the logits of all classes, with tail classes experiencing a significantly larger reduction than head classes due to their lower frequency. For target logits, this relative difference helps mitigate logit imbalance by emphasizing the contribution of tail classes during training. However, when applied to non-target logits, the same adjustment uniformly suppresses their magnitudes according to class frequency, distorting the learning signals from confusing non-target classes. As a result, the model fails to capture meaningful distinctions among these confusing classes, making it difficult to learn precise and discriminative decision boundaries.

To further investigate this phenomenon, we analyze epoch-wise changes in the target and non-target probabilities of head, middle, and tail classes, as illustrated in Figure 1. The gray and orange lines represent values before and after logit adjustment, respectively. As shown in the left column, the adjustment increases target probabilities of head classes while decreasing those of tail classes, appropriately shifting the learning emphasis toward tail samples. However, in the non-target case (right column), the same adjustment uniformly suppresses the non-target probabilities of middle and tail classes throughout training, weakening the gradient signal from confusing non-target categories. We define this as the Non-Target Logit Over-Suppression (NTLOS) problem, where uniform logit adjustment excessively suppresses confusing non-target logits, reducing informative gradients and hindering fine-grained discrimination. Consequently, models trained under NTLOS exhibit higher prediction entropy on test data, as suppressed nontarget representations fail to generalize to semantically similar classes.

![](images/840bb177099ee81269e78f1f940563dffbb367aa7273eda949200964d1c02a4a.jpg)  
Figure 1. Visualization of the Non-Target Logit Over-Suppression Problem on CIFAR100-IF100. Logit adjustment increases head class target probabilities while reducing those of tail classes. However, it also excessively suppresses non-target probabilities of middle and tail classes, weakening informative gradients from confusing classes and hindering discrimination between confusing classes.

To mitigate this issue, we propose two complementary methods. Confusion-based Instance-wise Logit Adjustment (CILA) dynamically adjusts each instance’s logits through a confusion-based penalty, where confusion refers to cases in which one or more non-target logits exceed the target logit. This penalty preserves informative gradients from confusing non-target classes that uniform adjustment suppresses. However, the confusion penalty activates only for misclassified instances, leaving correctly classified but near-boundary samples unaddressed. To complement this, we propose Batch-adaptation Noise-Injection (BNI), which perturbs the target logit during training so that near-boundary correct samples become confusing after noise injection, making them eligible for the confusionaware learning process. By simulating uncertainty in correct predictions, BNI encourages the model to maintain a wider decision margin, leading to more stable and flexible decision boundaries.

CILA is loss-agnostic and can be seamlessly integrated into various long-tailed learning frameworks. Extensive experiments demonstrate that CILA consistently improves classification accuracy and training stability across multiple benchmarks, including CIFAR-100-LT, ImageNet-LT, and iNaturalist 2018.

The key contributions of this paper are summarized as follows:

• We identify the Non-Target Logit Over-Suppression as a key limitation of existing logit adjustment methods.

• We propose CILA, which applies a confusion-based penalty to restore informative gradients from confusing non-target classes.

• We propose BNI, which extends the confusion-aware update to near-boundary samples via batch-adaptive noise injection.

## 2. Related Work

Data Re-Balancing. Traditional approaches to address the class imbalance problem include Re-Sampling [2–5, 11, 12, 18, 27, 28] or Re-Weighting [4, 7]. Re-sampling balances class distribution by increasing or decreasing the number of samples per class, while re-weighting assigns loss weights based on class frequencies to emphasize tail classes. Although effective, re-sampling can cause overfitting in tail classes or under-represent head classes, while re-weighting may cause unstable training due to excessively large gradients in tail classes.

Logit Adjustment. To overcome data-level limitations, logit adjustment methods modify the classifier’s output logits. Margin-based approaches [4, 29, 37] enhance class separability by imposing instance-adaptive margins on target logits, yet still modify only target logits and thus fail to address the learning signal from confusing non-target classes. Logit Adjustment (LA) [23, 25] applies class-frequencyderived terms to all logits, suppressing head-class dominance and emphasizing tail classes, and has been widely adopted across long-tailed recognition [8, 20, 39], zeroshot learning [6], and generalized classification [38]. However, both margin-based methods and LA share a fundamental limitation: they apply class-level adjustments uniformly across all instances, ignoring the instance-wise degree of confusion between the target and non-target classes. This causes NTLOS, suppressing informative gradients from confusing non-target classes regardless of how confusing each instance is.

Confusion-aware Learning. CP-Mix [36] estimates classpair confusion from the model’s confusion matrix and synthesizes augmented samples between confusing class pairs, while OTMix [10] uses optimal transport to generate semantically meaningful minority samples guided by classlevel confusion. Both methods address confusion at the class level through data synthesis, relying on accumulated confusion statistics rather than the confusion of each individual instance.

In contrast, CILA operates directly at the logit level, applying a penalty to each non-target logit based on its instance-wise degree of confusion with the target at every training step, without requiring data synthesis or a precomputed confusion matrix. Furthermore, BNI extends this update to correctly classified but near-boundary samples, which confusion-based augmentation methods do not address.

![](images/4ee4631f90bbdcae29087e5d643e870a1d66a1f7589fce8e96b0841994b2aef6.jpg)  
Figure 2. Illustration of class-wise logit adjustment in non-confusing and confusing cases. (a) shows a non-confusing case that is easily distinguished. However, (b) represents a confusing case that is easily confused with another class (Dog). Class-wise logit adjustmen increases a positive sample loss for the correct class while simultaneously reducing negative sample losses for confusing classes.

## 3. Problem Setup

## 3.1. Logit Imbalance in Long-tailed Recognition

In long-tailed datasets, the class imbalance causes headclass logits to grow significantly larger during training, resulting in severe logit imbalance. To alleviate this, logit adjustment methods apply class-dependent terms derived from class frequencies to all logits:

$$
\pi _ { k } ^ { L A } = \log { \left( \frac { n _ { k } } { \sum n _ { j } } \right) } ,\tag{1}
$$

$$
L _ { L A } = - \log \left( \frac { e ^ { z _ { y } + \pi _ { y } ^ { L A } } } { \sum _ { k = 1 } ^ { N } e ^ { z _ { k } + \pi _ { k } ^ { L A } } } \right)\tag{2}
$$

where $y$ denotes the ground truth (target) class for the given sample $x , z _ { k }$ is the logit of class $k ,$ and $n _ { k }$ is the number of training samples in class $k .$

Figure 2 illustrates how logit adjustment operates in a four-class scenario (Apple, Bed, Cat, Dog), where Cat and Dog are tail classes. As shown in Figure 2(a), the larger adjustment terms assigned to tail classes lower their logits (blue dashed → solid), helping the model focus more on tail classes and mitigating the logit imbalance between head and tail classes.

## 3.2. Non-Target Logit Over-Suppression Problem

While class-wise logit adjustment effectively mitigates class imbalance for target logits, it carries a critical limitation on non-target logits. Figure 2(b) illustrates this in a confusing case, where the input is a Cat-like-Dog image easily mistaken for Dog. The same adjustment is applied uniformly to all non-target logits, reducing the logit for Dog (red dashed → solid box) and lowering its loss contribution, weakening the learning signal needed to differentiate semantically similar categories. We term this the Non-Target Logit Over-Suppression (NTLOS) problem, where uniform logit adjustment excessively suppresses confusing non-target logits, reducing informative gradients and hindering fine-grained discrimination. We formally characterize NTLOS at the gradient level in Section 4.3.

## 4. Proposed Method

As shown in Section 3.2, uniform logit adjustment suppresses the gradient from confusing non-target classes, hindering fine-grained discrimination. To address this, we propose two complementary methods. First, Confusionbased Instance-wise Logit Adjustment (CILA) applies a confusion-based penalty proportional to the degree to which each non-target class challenges the target, preventing the over-suppression of informative non-target logits. Second, Batch-adaptation Noise-Injection (BNI) extends the confusion-aware update to correctly classified but nearboundary samples, which the penalty alone cannot reach.

## 4.1. Confusion-based Instance-wise LA (CILA)

We introduce Confusion-based Instance-wise Logit Adjustment, a method that adaptively adjusts each instance’s logits based on the degree of confusion, quantified by how much a non-target logit $z _ { k }$ exceeds the target logit $z _ { y }$ . The adjustment is computed in three steps: (1) measuring a confusionbased penalty term, (2) weighting classes to reflect class imbalance, and (3) combining these to produce the final adjustment term.

Step 1. Confusion-based Penalty Term. We define class k as confusing for a given instance when its logit exceeds the target logit, $z _ { k } > z _ { y }$ . To quantify the degree of confusion, we measure the relative logit magnitude anchored at $z _ { \mathrm { m i n } } \colon$

$$
\mathcal { C } _ { k } = \frac { z _ { k } - z _ { \operatorname* { m i n } } } { z _ { y } - z _ { \operatorname* { m i n } } + \epsilon } ,\tag{3}
$$

where ϵ is a small constant for numerical stability. This formulation is motivated by the shift-invariance of the softmax, where logits can be expressed relative to $z _ { m i n }$ . Thus, $C _ { k }$ provides a scale-normalized measure of how closely a nontarget class approaches the target. In particular, $C _ { k } > 1$ if and only if $z _ { k } > z _ { y }$ , which is precisely our confusion criterion.

Based on $\mathcal { C } _ { k } .$ we define the confusion-based penalty as:

$$
P _ { k } = \mathrm { c l i p } \left( \mathcal { C } _ { k } - 1 , 0 , 2 \right) .\tag{4}
$$

Subtracting 1 activates the penalty only for genuinely confusing classes, while its magnitude increases with the degree of confusion to provide stronger adjustment to more strongly confused classes. To prevent excessively large corrections from dominating optimization, we set the clipping bound to 2 based on the ablation in Table 5: smaller bounds discard informative signals for tail samples, whereas a bound of 2 yields the largest tail-class gains.

Step 2. Class Weighting Term. In long-tailed distributions, tail classes rarely encounter confusing samples, though such instances are crucial for refining decision boundaries. To address this, we modulate the confusion penalty using classwise weights, which amplify the penalty when the target class is a tail class. Following prior studies [24], the classwise weight is defined as follows:

$$
D _ { k } \ = \ { \frac { 1 / n _ { k } } { \sum _ { j } 1 / n _ { j } } } \cdot C\tag{5}
$$

where $n _ { k }$ denotes the number of training instances in class $k ,$ , and $C$ is the total number of classes. The normalized inverse-frequency weight ensures stronger penalties for tail classes $( n _ { k }$ small).

Step 3. Confusion-based Logit Adjustment. The final logit adjustment term for each class k is computed by combining the confusion penalty and the Class Weighting term:

$$
\begin{array} { r } { \pi _ { k } ^ { C } = P _ { k } \cdot D _ { y } , } \end{array}\tag{6}
$$

where $D _ { y }$ is the weight of the target class $y .$ This formulation amplifies the penalty for confusing non-targets when the target is a tail class, encouraging the model to resolve instance-level ambiguity.

The confusion-aware logit adjustment is incorporated into the training loss as follows:

$$
\begin{array} { r } { L _ { C I L A } = - \log \left( \frac { e ^ { z _ { y } + \pi _ { y } ^ { C } } } { \sum _ { k = 1 } ^ { N } e ^ { z _ { k } + \pi _ { k } ^ { C } } } \right) . } \end{array}\tag{7}
$$

## 4.2. Batch-adaptation Noise-Injection (BNI)

The confusion penalty activates only for misclassified instances $( z _ { k } \ > \ z _ { y } ) .$ , leaving correctly classified but nearboundary samples unaddressed. To complement this, we introduce Batch-adaptation Noise-Injection (BNI), which perturbs the target logit so that $z _ { y } ^ { N }$ replaces $z _ { y }$ in Eq. 7 before computing $\mathcal { C } _ { k }$ , enabling CILA to handle near-boundary correct samples.

Noise Injection. To expose near-boundary correct samples to the confusion-aware penalty, BNI reduces the target logit by injecting a dynamically scaled noise term before computing $C _ { k }$ , so that the confusion penalty is evaluated on the perturbed target. The target logit $z _ { y }$ is perturbed as:

$$
z _ { y } ^ { N } = z _ { y } - N _ { y } , \qquad N _ { y } = D _ { y } { \cdot } \xi , \quad \xi \sim \mathrm { R a y l e i g h } ( \sigma ) ,\tag{8}
$$

![](images/8c8bed1caed4e95fe68dce319b9e3c94cd860ed3635e3e8c5652390971b55205.jpg)  
Figure 3. Overview of the proposed method.

where $D _ { y }$ is the class-dependent weight defined in Eq. 5, with larger $D _ { y }$ for tail classes concentrating the regularization where decision boundaries are most uncertain.

We adopt the Rayleigh distribution for ξ because, unlike Gaussian whose density peaks at zero, Rayleigh attains its mode at $\sigma > 0$ , ensuring non-trivial perturbation magnitude. Formally, its expected value satisfies:

$$
\mathbb { E } [ \xi ] = \sigma \cdot \sqrt { \frac { \pi } { 2 } } > 0 ,\tag{9}
$$

guaranteeing that the noise consistently induces a meaningful reduction in the target logit rather than being dominated by near-zero perturbations. The scale parameter $\sigma$ is not fixed but adapts dynamically to the current training state, as described next.

Batch-Adaptive Noise Scale. Since logit magnitudes grow substantially during training, a fixed σ is either too small in late epochs or too large in early ones. We therefore set σ proportional to the batch mean target logit $\begin{array} { r l } { \bar { z } _ { y } } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { | B | } \sum _ { ( x , y ) \in B } z _ { y } \colon } \end{array}$

$$
\sigma = \hat { z } _ { y } \cdot \rho ,\tag{10}
$$

where $\rho < 1$ is a scaling factor derived from the stability condition detailed below. Since $\bar { z } _ { y }$ can be near zero or negative during early training, we apply a minimum value of 0.5 in practice to ensure $\sigma > 0$ as required by the Rayleigh distribution.

Using $\bar { z } _ { y }$ directly as $\sigma$ would collapse the target logit even for well-classified samples. We therefore impose the stability condition $\mathbb { E } [ N _ { y } ] < \bar { z } _ { y } ,$ , which gives:

$$
\mathbb { E } [ N _ { y } ] = D _ { y } \cdot \bar { z } _ { y } \cdot \rho \cdot \sqrt { \frac { \pi } { 2 } } < \bar { z } _ { y } ,\tag{11}
$$

Table 1. Performance comparison with loss function-based on the CIFAR-100-LT, re-implemented from official code (CUDA/DODA/SHIKE: respective repositories; others: DBM [29])
<table><tr><td colspan="2"></td><td colspan="4">IF 10</td><td colspan="4">IF 50</td><td colspan="4">IF 100</td></tr><tr><td>Method</td><td>Loss</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td><td>Head</td><td>Med</td><td>Tail</td><td>All 43.4±0.3</td></tr><tr><td rowspan="9">Loss</td><td>CE</td><td>67.4±0.5</td><td>45.7±0.5</td><td></td><td>60.7±0.5</td><td>69.7±0.4</td><td>40.2±0.6</td><td>18.0±0.7</td><td>48.3±0.3 47.7±0.0</td><td>72.0±0.3</td><td>43.5±0.6</td><td>10.0±0.4</td><td></td></tr><tr><td>LDAM</td><td>67.1±0.2</td><td>46.5±0.8</td><td></td><td>60.8±0.4</td><td>69.8±0.3</td><td>39.3±0.5</td><td>16.3±0.5</td><td></td><td>71.4±0.4</td><td>42.4±1.1</td><td>10.5±0.5</td><td>43.0±0.4</td></tr><tr><td>DBM</td><td>67.7±0.5</td><td>46.8±0.7</td><td></td><td>61.2±0.4</td><td>69.8±0.2</td><td>40.0±0.5</td><td>17.7±0.5</td><td>48.2±0.2</td><td>71.7±0.8</td><td>42.3±0.5</td><td>10.2±0.4</td><td>43.0±0.2</td></tr><tr><td>CILA</td><td>66.8±0.1</td><td>54.1±0.3</td><td></td><td>62.7±0.3</td><td>68.0±0.2</td><td>44.6±0.5</td><td>29.0±0.4</td><td>51.2±0.3</td><td>70.3±0.5</td><td>46.1±0.2</td><td>19.2±0.8</td><td>46.1±0.1</td></tr><tr><td>CE-DRW</td><td>65.1±0.7</td><td>54.2±1.6</td><td></td><td>61.7±0.2</td><td>65.9±0.7</td><td>44.9±1.0</td><td>28.4±0.5</td><td>50.5±0.1</td><td>67.6±0.9</td><td>47.3±0.8</td><td>19.0±0.3</td><td>45.9±0.2</td></tr><tr><td>LDAM-DRW</td><td>61.7±1.9</td><td>59.9±3.0</td><td></td><td>61.2±0.3</td><td>56.8±0.9</td><td>49.2±0.1</td><td>39.4±0.3</td><td>50.5±0.3</td><td>55.9±0.7</td><td>49.6±0.2</td><td>31.8±0.4</td><td>46.5±0.1</td></tr><tr><td>DBM-DRW</td><td>63.6±0.1</td><td>55.2±0.4</td><td></td><td>61.0±0.1</td><td>66.1±0.2</td><td>46.3±0.2</td><td>29.0±0.6</td><td>51.3±0.1</td><td>67.3±0.6</td><td>47.8±0.4</td><td>19.9±0.7</td><td>46.2±0.2</td></tr><tr><td>CILA-DRW</td><td>66.5±0.1</td><td>55.0±0.3</td><td></td><td>62.7±0.2</td><td>67.0±0.6</td><td>46.2±0.1</td><td>29.6±0.6</td><td>51.5±0.2</td><td>68.1±1.1</td><td>48.0±0.5</td><td>20.4±0.4</td><td>46.5±0.2</td></tr><tr><td>LA</td><td>65.6±0.3</td><td>55.2±0.4</td><td></td><td>62.4±0.3</td><td>65.2±0.3</td><td>46.9±0.3</td><td>37.5±1.4</td><td>52.5±0.2</td><td>65.1±0.3</td><td>47.9±0.7</td><td>27.5±0.5</td><td>47.8±0.3</td></tr><tr><td></td><td>LA + DBM</td><td>65.4±0.4 65.9±0.1</td><td>54.8±0.4 57.1±0.1</td><td></td><td>62.1±0.2 63.2±0.1</td><td>65.2±0.3 63.5±0.1</td><td>47.0±0.4 48.5±0.5</td><td>34.8±1.3</td><td>52.3±0.4</td><td>66.6±0.1</td><td>48.1±0.6</td><td>25.4±0.73</td><td>47.8±0.3 48.5±0.5</td></tr><tr><td></td><td>LA+CILA</td><td>64.8±0.1</td><td>58.1±0.8</td><td></td><td></td><td>60.7±0.6</td><td>48.2±1.3</td><td>41.3±0.9</td><td>52.6±0.3</td><td>64.4±0.2</td><td>49.1±0.8</td><td>30.0±0.6</td><td></td></tr><tr><td rowspan="3">CMO</td><td>LA</td><td></td><td></td><td></td><td>62.7±0.2</td><td></td><td></td><td>41.0±0.8</td><td>52.0±0.7</td><td>59.5±0.8</td><td>49.6±0.3</td><td>32.2±0.2</td><td>47.8±0.1</td></tr><tr><td>LA+CILA</td><td>66.5±0.2</td><td>61.3±0.1</td><td></td><td>64.9±0.2</td><td>62.3±1.2</td><td>50.7±1.3</td><td>46.5±0.4</td><td>54.7±0.1</td><td>61.0±0.4</td><td>50.1±0.8</td><td>36.4±1.1</td><td>49.8±0.2</td></tr><tr><td>LA</td><td>66.3±0.1</td><td>55.9±0.1</td><td>63.1±0.1</td><td></td><td>66.2±1.3</td><td>47.5±0.5</td><td>35.9±0.3</td><td>53.1±0.7</td><td>66.8±0.5</td><td>48.8±0.9</td><td>27.2±0.3</td><td>48.6±0.5</td></tr><tr><td rowspan="2">BCL GML</td><td>LA+CILA</td><td>68.2±0.3</td><td>59.2±0.1</td><td></td><td>65.4±0.3</td><td>67.0±0.3</td><td>52.9±0.2</td><td>43.9±0.2</td><td>57.1±0.0</td><td>68.6±1.4</td><td>53.2±0.1</td><td>33.7±0.3</td><td>52.8±0.5</td></tr><tr><td>LA</td><td>68.0±0.2</td><td>59.2±0.9</td><td></td><td>65.6±0.2</td><td>68.8±0.2</td><td>52.1±0.3</td><td>42.1±0.9</td><td>57.3±0.3</td><td>69.4±0.2</td><td>54.6±0.3</td><td>31.8±0.1</td><td>53.1±0.0</td></tr><tr><td rowspan="2"></td><td>LA+CILA</td><td>69.4±0.2</td><td>56.8±1.5</td><td></td><td>65.5±0.4</td><td>69.1±0.2</td><td>56.0±0.3</td><td>42.1±1.9</td><td>59.0±0.1</td><td>70.4±0.5</td><td>57.3±0.6</td><td>31.4±0.1</td><td>54.2±0.4</td></tr><tr><td>LA</td><td>67.6±0.3</td><td>60.2±0.5</td><td></td><td>65.3±0.2</td><td>68.4±0.4</td><td>53.8±0.0</td><td>46.9±1.0</td><td>58.6±0.3</td><td>69.3±0.4</td><td>55.9±0.3</td><td>34.8±0.6</td><td>54.3±0.2</td></tr><tr><td rowspan="2">NCL</td><td>LA+CILA</td><td>68.2±0.4</td><td>58.5±1.2</td><td></td><td>65.2±0.2</td><td>68.5±0.1</td><td>55.2±0.4</td><td>45.0±1.8</td><td>58.8±0.2</td><td>69.4±1.9</td><td>57.7±1.4</td><td>34.9±3.1</td><td>55.0±0.2</td></tr><tr><td>LA</td><td>64.1±0.2</td><td>56.6±0.4</td><td></td><td>61.7±0.2</td><td>62.5±0.8</td><td>47.7±0.8</td><td>37.6±0.4</td><td>51.9±0.4</td><td>62.6±0.1</td><td>48.3±0.2</td><td>28.6±0.7</td><td>47.4±0.3</td></tr><tr><td rowspan="2">CUDA</td><td>LA+CILA</td><td>67.4±0.3</td><td>59.0±0.2</td><td></td><td>64.8±0.2</td><td>64.8±0.6</td><td>51.3±0.5</td><td>42.0±0.1</td><td>55.2±0.2</td><td>64.9±0.6</td><td>51.0±0.6</td><td>33.0±0.9</td><td>50.5±0.3</td></tr><tr><td>LA</td><td>62.2±0.2</td><td>55.3±0.3</td><td></td><td>60.1±0.2</td><td>61.3±0.6</td><td>47.9±0.5</td><td>35.2±1.3</td><td>51.1±0.4</td><td>62.0±1.3</td><td>48.9±0.6</td><td>25.7±0.2</td><td>46.5±0.5</td></tr><tr><td rowspan="2">DODA</td><td>LA+CILA</td><td>65.5±0.3</td><td>58.1±1.5</td><td></td><td>63.2±0.3</td><td>63.2±0.2</td><td>52.2±0.5</td><td>41.0±1.0</td><td>54.7±0.3</td><td>64.2±1.3</td><td>51.5±1.2</td><td>32.0±0.8</td><td>50.1±0.2</td></tr><tr><td>LA</td><td>70.0±0.2</td><td>61.9±0.4</td><td></td><td>67.5±0.3</td><td>70.8±0.1</td><td>54.2±0.3</td><td>42.9±0.4</td><td>59.0±0.2</td><td>71.5±0.3</td><td>56.7±0.2</td><td>35.8±0.2</td><td>55.6±0.1</td></tr><tr><td rowspan="2">SHIKE</td><td>LA+CILA</td><td>70.5±0.8</td><td>62.6±0.7</td><td></td><td>68.2±0.2</td><td>70.0±0.1</td><td>55.9±0.3</td><td>45.9±0.3</td><td>60.6±0.1</td><td>67.6±0.8</td><td>57.6±0.5</td><td>41.4±0.3</td><td>57.0±0.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

and yields the upper bound:

$$
\rho < \sqrt { \frac { 2 } { \pi } } \frac { 1 } { D _ { \mathrm { t a i l } } } ,\tag{12}
$$

where $D _ { \mathrm { t a i l } }$ is the weight of the most underrepresented class. For CIFAR-100-LT (IF=100), $D _ { \mathrm { t a i l } } \ \approx \ 3 . 9 7$ gives $\rho \lesssim 0 . 2 0 1$ , so we set $\rho = 0 . 2$ . For other datasets, $\rho$ is determined analogously from Eq. 12 using their respective $D _ { \mathrm { t a i l } }$ values.

Unlike existing noise regularization methods that primarily suppress overconfidence, BNI serves a fundamentally different purpose: it intentionally simulates uncertainty in correctly predicted but near-boundary samples, making them eligible for the confusion-aware penalty. Rather than smoothing confidence globally, BNI concentrates its effect on tail classes via the class-dependent scale $D _ { y } ,$ adaptively expanding the set of informative instances that refine decision boundaries.

## 4.3. Gradient Analysis

We formally compare the gradient flow under LA and CILA to characterize NTLOS and show that CILA provably restores the suppressed gradient.

Gradient under LA. The gradient of $\mathcal { L } _ { \mathrm { L A } }$ with respect to a non-target logit $z _ { k } ~ ( k ~ \neq ~ y )$ is $\tilde { p } _ { k }$ , where $\begin{array} { r l } { \tilde { p } _ { k } } & { { } = } \end{array}$ $\frac { \exp ( z _ { k } + \pi _ { k } ^ { \mathrm { L A } } ) } { \sum _ { j } \exp ( z _ { j } + \pi _ { j } ^ { \mathrm { L A } } ) }$ . For a confusing tail class $k , \pi _ { k } ^ { \mathrm { L A } } \ll 0$ substantially reduces $\tilde { p } _ { k }$ regardless of instance-level confusion, confirming NTLOS as a gradient-level phenomenon:

$\frac { \partial \mathcal { L } _ { \mathrm { L A } } } { \partial z _ { k } } = \tilde { p } _ { k } \ll p _ { k } ,$ for confusing tail class k, (13)

where $\begin{array} { r } { p _ { k } \ = \ \frac { \exp ( z _ { k } ) } { \sum _ { i } \exp ( z _ { j } ) } } \end{array}$ is the standard softmax probability without adjustment. Thus, the gradient signal from confusing non-target classes is significantly suppressed by LA, regardless of how confusing the instance actually is.

Gradient Restoration by CILA. Under CILA, the gradient takes the same form with $\begin{array} { r } { \tilde { q } _ { k } = \frac { \exp ( z _ { k } + \pi _ { k } ^ { C } ) } { \sum _ { j } \exp ( z _ { j } + \pi _ { j } ^ { C } ) } } \end{array}$ . For a confusing class k satisfying $z _ { k } \ > \ z _ { y } ,$ , the confusion penalty satisfies $\pi _ { k } ^ { C } ~ > ~ 0 ~ > ~ \pi _ { k } ^ { \mathrm { L A } }$ , so the adjusted logit of class k is strictly larger under CILA than under LA. For nonconfusing classes j where $z _ { j } ~ \le ~ z _ { y } , ~ P _ { j } ~ = ~ 0$ and thus $\pi _ { j } ^ { C } = 0$ , so their adjusted logits remain unchanged. Since the softmax probability is monotonically increasing in its own logit, it follows that $\tilde { q } _ { k } > \tilde { p } _ { k }$ , and therefore:

$$
\frac { \partial \mathcal { L } _ { \mathrm { C I L A } } } { \partial z _ { k } } = \tilde { q } _ { k } > \tilde { p } _ { k } = \frac { \partial \mathcal { L } _ { \mathrm { L A } } } { \partial z _ { k } } , \quad \forall k \mathrm { s . t . } z _ { k } > z _ { y } .\tag{14}
$$

As $\pi _ { k } ^ { C }$ increases with $z _ { k } - z _ { y }$ , the gradient correction grows with the degree of confusion, meaning the most confusing instances receive the strongest learning signal.

Furthermore, BNI perturbs the target logit before computing the confusion penalty:

$$
z _ { y } ^ { N } = z _ { y } - N _ { y } , \qquad N _ { y } = D _ { y } \xi , \quad \xi \sim \mathrm { R a y l e i g h } ( \sigma ) .\tag{15}
$$

Table 2. Accuracy (%) on ImageNet-LT and iNaturalist 2018. † and ‡ denote results from DODA [33] and DBM [29], respectively.
<table><tr><td rowspan="2">Method</td><td colspan="4">ImageNet-LT</td><td colspan="4">iNaturalist 2018</td></tr><tr><td>Head</td><td>Med</td><td>Tail</td><td>All</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td>CE†</td><td>64.0</td><td>33.8</td><td>5.8</td><td>41.6</td><td>73.9</td><td>63.5</td><td>55.5</td><td>61.0</td></tr><tr><td>cRT ↑</td><td>58.8</td><td>44.0</td><td>26.1</td><td>47.3</td><td>69.0</td><td>66.0</td><td>63.2</td><td>65.2</td></tr><tr><td>LWS†</td><td>57.1</td><td>45.2</td><td>29.3</td><td>47.7</td><td>65.0</td><td>66.3</td><td>65.5</td><td>65.9</td></tr><tr><td>CE-DRW †</td><td>61.7</td><td>47.3</td><td>28.8</td><td>50.1</td><td>68.2</td><td>67.3</td><td>66.4</td><td>67.0</td></tr><tr><td>LDAM-DRW †</td><td>60.4</td><td>46.9</td><td>30.7</td><td>49.8</td><td></td><td></td><td></td><td>66.1</td></tr><tr><td>LA†</td><td>60.9</td><td>48.8</td><td>32.1</td><td>51.0</td><td>65.5</td><td>67.5</td><td>67.5</td><td>67.2</td></tr><tr><td>ALA</td><td>62.4</td><td>49.1</td><td>35.7</td><td>52.4</td><td>71.3</td><td>70.8</td><td>70.4</td><td>70.7</td></tr><tr><td>LA + OTmix</td><td>64.0</td><td>52.4</td><td>42.7</td><td>55.6</td><td>71.1</td><td>72.0</td><td>70.8</td><td>71.5</td></tr><tr><td>LA + CILA</td><td>62.9±0.1</td><td>53.7±0.3</td><td>40.7±0.4</td><td>55.5±0.1</td><td>66.1±0.2</td><td>71.3±0.1</td><td>73.8±0.3</td><td>71.7±0.2</td></tr><tr><td>BCL †</td><td>65.3</td><td>53.5</td><td>36.3</td><td>55.6</td><td>68.2</td><td>71.3</td><td>71.3</td><td>71.0</td></tr><tr><td>BCL-DBM </td><td></td><td></td><td></td><td>56.3</td><td>65.6</td><td>71.8</td><td>73.8</td><td>71.9</td></tr><tr><td>BCL + CILA</td><td>65.6±0.1</td><td>55.4±0.3</td><td>44.3±0.4</td><td>57.8±0.1</td><td>67.1±1.0</td><td>71.8±0.4</td><td>74.6±0.5</td><td>72.4±0.1</td></tr></table>

Since the noise scale is adapted to the current batch logit magnitude, the resulting perturbation follows the training state and exposes correctly classified but near-boundary samples to the same confusion-aware gradient mechanism.

## 4.4. Stability of the Combined Objective.

We analyze the stability of the combined CILA-BNI objective by separately controlling the magnitudes of the confusion-based correction and the injected perturbation.

First, the CILA penalty is explicitly bounded by the clipping operation in Eq. (4), such that $0 \leq P _ { k } \leq 2$ . Therefore, the additional logit correction satisfies $0 \leq \pi _ { k } ^ { C } \leq 2 D _ { y } ,$ where $D _ { y }$ is the class-dependent weight in Eq. (5). This bound prevents the confusion-aware correction from growing arbitrarily large, even for highly confusing instances. Second, BNI controls the magnitude of the target-logit perturbation through the batch-adaptive scale in Eq. (10). Specifically, imposing E ${ [ N _ { y } ] < \bar { z } _ { y } }$ yields the upper bound on ρ in Eq. (12). Thus, the expected perturbation remains smaller than the batch-mean target logit, preventing the injected noise from dominating the target signal.

Together, these constraints bound the CILA and BNI corrections, enabling confusion-aware gradient correction while preventing unbounded logit modifications and supporting stable optimization.

## 5. Experiments

Datasets. To ensure fair comparisons, we conduct experiments on three widely used long-tailed image classification benchmarks: CIFAR-100-LT [4], ImageNet-LT [22], and iNaturalist 2018 [32].

Comparison Baselines. We evaluate our proposed method against a diverse set of baseline methods to ensure fair and comprehensive comparisons across different learning strategies for long-tailed recognition. The baselines include:

• Loss Function-Based Method: Cross-Entropy (CE) [13]

• Re-weighting Method: DRW (NeurIPS’19) [4] • Logit Adjustment Method: LDAM (NeurIPS’19) [4], LA (ICLR’21) [23], ALA (AAAI’22) [37], and DBM (AAAI’25) [29]

• Augmentation Method: CMO (CVPR’22) [24], OTmix (NeurIPS’23) [10]

• Curriculum Learning Method: CUDA (ICLR’23) [1], DODA (ICLR’24) [33]

• Contrastive Learning Method: BCL (CVPR’22) [39]

• Mutual Information Maximization Method: GML (ICML’23) [30]

• Ensemble-based method: NCL (CVPR’22) [19], SHIKE (CVPR’23) [15]

In the tables, CILA denotes our full method comprising Confusion-based Logit Adjustment and Batch-adaptation Noise-Injection (BNI). As a plug-in module, CILA can be integrated into existing methods and is evaluated both as a standalone method and in combination with representative baselines.

Implementation Details. Following the experimental setup of prior work [4, 23, 29], all implementation details are provided in the supp. 3.

## 5.1. Experimental Results

CIFAR-100-LT. Table 1 presents the performance comparison on CIFAR-100-LT under varying imbalance factors (IF=10, 50, 100). CILA consistently improves over loss function-based, re-weighting, and logit adjustment baselines, and further generalizes as a plug-in module across a range of representative long-tailed learning methods spanning diverse learning paradigms.

ImageNet-LT and iNaturalist 2018. Table 2 presents the performance of the proposed method on ImageNet-LT and iNaturalist 2018 datasets. BCL+CILA achieves the best performance on both datasets, with particularly notable gains in tail classes, demonstrating that the confusion-aware penalty and batch-adaptive noise injection effectively improve discrimination in rare categories.

![](images/62562149aca2d992b713cd8d42f434df42022442ed872e935f745683e87cd372.jpg)  
Figure 4. Confusion penalty and accuracy across training epochs under three noise settings: no noise (zero), fixed-scale noise (fix), and batch-adaptive noise (mu) (CIFAR-100-LT, IF=100).

## 5.2. Analysis of Proposed Method

We analyze the effectiveness of CILA and BNI through four complementary evaluations: feature discriminability, prediction uncertainty, confusion between semantically similar classes, and the effect of batch-adaptive noise scaling.

Feature Discriminability. We measure intra-class compactness and inter-class separability following [29]. As shown in Table 3, CILA-LA achieves smaller intra-class distances and higher inter-class separability than LA and DBM-LA across all class groups. Unlike DBM-LA, which improves separability at the cost of increased intra-class distances, CILA-LA improves both metrics simultaneously, demonstrating more discriminative representations.

Table 3. Analysis of inter-class separability and intra-class compactness on CIFAR100 (IF 100).
<table><tr><td rowspan="2">Method</td><td colspan="4">Inter-class Separability ↑</td><td colspan="4">Intra-class Compactness ↓</td></tr><tr><td>Head</td><td>Med</td><td>Tail</td><td>All</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td>LA [23]</td><td>6.0</td><td>6.0</td><td>5.7</td><td>5.9</td><td>28.6</td><td>28.8</td><td>29.6</td><td>29.0</td></tr><tr><td>DBM-LA [29]</td><td>7.0</td><td>6.3</td><td>6.0</td><td>6.2</td><td>31.0</td><td>31.9</td><td>33.1</td><td>32.0</td></tr><tr><td>CILA-LA (Ours)</td><td>6.8</td><td>6.9</td><td>6.1</td><td>6.6</td><td>28.1</td><td>28.3</td><td>29.3</td><td>28.6</td></tr></table>

Prediction Uncertainty. To verify that CILA mitigates NT-LOS in terms of prediction confidence, we measure mean prediction entropy on test data across head, medium, and tail classes. As shown in Figure 5, LA consistently exhibits higher entropy across all class groups and imbalance factors, with particularly large values for tail classes, directly reflecting the suppressed non-target gradients identified in Section 3.2. In contrast, CILA substantially reduces entropy across all settings, confirming that the confusion-aware adjustment restores informative gradients and yields more stable, confident predictions.

Effect of Batch-Adaptive Noise Scale. To evaluate the effect of adaptive noise scaling, we track the confusion penalty and accuracy during training under three settings: no noise (zero), fixed-scale noise (fix), and batch-adaptive noise (mu), as shown in Figure 4.

![](images/2c681a5d9c5d2866716506da8d1c0167fbb07ff585bc851dde41a68eeb0537ad.jpg)  
Figure 5. Mean prediction entropy on test data.

Without noise, the tail penalty nearly vanishes after 100 epochs despite persistently low tail accuracy, confirming that many tail instances are correctly classified yet remain near the decision boundary. With batch-adaptive noise, these instances continue to receive meaningful penalties throughout training, leading to sustained improvement in tail accuracy. Compared to fixed-scale noise, batch-adaptive scaling maintains larger penalties on tail classes, as the noise scale is governed by the batch mean target logit, which is inherently smaller for tail classes. Consequently, batchadaptive noise improves tail performance with minimal impact on head accuracy, demonstrating that the adaptivity of the scaling, not merely the presence of noise, is critical for tail generalization.

Confusion between Semantically Similar Classes. To directly evaluate whether CILA improves discrimination between confusing classes, we analyze the confusion matrix on semantically similar class pairs. As shown in Figure 6, LA frequently misclassifies samples among visually similar categories, whereas CILA reduces such confusion, confirming that the instance-wise penalty effectively restores the learning signal needed to separate fine-grained classes.

## 5.3. Ablation Study

We conduct ablation studies to validate the contribution of each component of CILA and the design choices of BNI. Additional results are provided in the supplementary.

![](images/0ca6201333e4b77daeebcbc68975d7845f173c41dc0df8b4e53cdb755080ca5e.jpg)

![](images/9fa1a2fd02607301fca4e2608adf9b26dff480f0f6088ecd82f737298b919680.jpg)

![](images/659f2fd2dcc330e0ccaa9c3f832cda48f8f976c986c1fd668a0496df419c1029.jpg)  
Figure 6. Difference in confusion matrices (CILA − LA) for semantically similar class pairs on CIFAR-100-LT (IF=100). Positive values (green) indicate improved correct predictions by CILA, confirming reduced misclassification between fine-grained classes.

Component Ablation. Table 4 summarizes the performance improvements obtained by progressively enabling CILA, BNI, and the class weighting term $( D _ { y }$ in Eq. 5). Both CILA and BNI yield improvements over the baseline LA accuracy. However, when $D _ { y }$ is omitted, accuracy improves for head classes but degrades for medium and tail classes, demonstrating that $D _ { y }$ effectively balances learning across class groups and mitigates head-class dominance.

Table 4. Ablation study on the three components under different imbalance factors (IF = 50, 100). ✓indicates the corresponding component is enabled.
<table><tr><td>Components</td><td colspan="3"></td><td colspan="5"></td></tr><tr><td>CILA BNI</td><td> $D _ { y }$ </td><td>Head Med</td><td>Tail</td><td>All</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td></td><td></td><td>62.7 47.5</td><td>36.8</td><td>51.8</td><td>62.3</td><td>47.1</td><td>25.3</td><td>45.9</td></tr><tr><td> $\checkmark$   $\checkmark$ </td><td> $\checkmark$ </td><td>64.6</td><td>46.8 36.1</td><td>52.1</td><td>64.9</td><td>47.0</td><td>25.8</td><td>46.9</td></tr><tr><td></td><td></td><td>64.3</td><td>47.0 37.3</td><td>52.3</td><td>64.6</td><td>47.6</td><td>28.0</td><td>47.6</td></tr><tr><td></td><td> $\checkmark$ </td><td>69.8</td><td>42.8 22.1</td><td>50.1</td><td>70.3</td><td>45.4</td><td>15.2</td><td>45.1</td></tr><tr><td></td><td> $\checkmark$ </td><td>63.3</td><td>47.9 37.7</td><td>52.4</td><td>62.2</td><td>47.1</td><td>29.1</td><td>47.0</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>70.0</td><td>42.4</td><td>21.3</td><td>49.9</td><td>71.4 44.8</td><td>13.5</td><td>44.7</td></tr><tr><td> $\checkmark$ </td><td></td><td></td><td>61.3 49.3</td><td>42.2</td><td>53.0</td><td>64.7</td><td>47.6 30.1</td><td>48.3</td></tr></table>

Effect of Penalty Upper Bound. We analyze the sensitivity to the clipping upper bound of the confusion-based penalty. As shown in Table 5, smaller bounds such as 0.5 or 1 degrade performance by truncating informative penalty signals from strongly confusing instances, whereas larger bounds of 1.5 and 2 retain stronger correction signals. In particular, an upper bound of 2 yields the largest gains on tail-class accuracy, indicating that confusion signals for tail classes are preserved rather than truncated. We therefore adopt 2 as a robust operating point, which preserves informative confusion signals while preventing excessively large corrections from dominating the optimization.

Table 5. Performance comparison across penalty upper bound.
<table><tr><td rowspan="2">Upper Bound</td><td colspan="4">IF 50</td><td colspan="4">IF 100</td></tr><tr><td>Head</td><td>Med</td><td>Tail</td><td>All</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td>0.5</td><td>62.3</td><td>48.0</td><td>37.2</td><td>51.9</td><td>62.7</td><td>47.4</td><td>29.3</td><td>47.4</td></tr><tr><td>1.0</td><td>63.2</td><td>47.0</td><td>39.1</td><td>52.2</td><td>64.3</td><td>48.4</td><td>28.1</td><td>47.9</td></tr><tr><td>1.5</td><td>64.3</td><td>47.4</td><td>37.3</td><td>52.5</td><td>64.8</td><td>48.2</td><td>28.8</td><td>48.2</td></tr><tr><td>2.0</td><td>61.3</td><td>49.3</td><td>42.2</td><td>53.0</td><td>64.7</td><td>47.6</td><td>30.1</td><td>48.3</td></tr></table>

Effect of Noise Type.

We further examine the choice of noise distribution in BNI. As shown in Table 6, Rayleigh noise consistently achieves the best overall performance, with notable gains on tail classes. This aligns with our design criterion that the injected noise should provide a non-trivial positive perturbation rather than concentrate near zero (Supp. Fig.1). Gaussian and Chi noise place substantial probability mass near zero, whereas Rayleigh has its mode at a positive value, more effectively exposing near-boundary samples to the confusion-aware penalty. These results support Rayleigh noise as a suitable choice for BNI.

Table 6. Performance comparison across noise types.
<table><tr><td rowspan="2">Noise Type</td><td colspan="4">IF 50</td><td colspan="4">IF 100</td></tr><tr><td>Head</td><td>Med</td><td>Tail</td><td>All</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td>Chi</td><td>63.3</td><td>48.4</td><td>35.5</td><td>52.2</td><td>63.4</td><td>48.5</td><td>28.5</td><td>47.7</td></tr><tr><td>Gaussian</td><td>65.0</td><td>47.0</td><td>39.0</td><td>52.9</td><td>64.3</td><td>48.7</td><td>28.9</td><td>48.2</td></tr><tr><td>Rayleigh</td><td>61.3</td><td>49.3</td><td>42.2</td><td>53.0</td><td>64.7</td><td>47.6</td><td>30.1</td><td>48.3</td></tr></table>

Effect of noise ratio $\rho _ { \bullet }$ We further analyze the sensitivity of the noise ratio $\rho ,$ which controls the proportion of the mean target logit magnitude used to determine the injected noise scale (Eq. 10). Based on our theoretical derivation, $\rho = 0 . 2$ is prescribed for CIFAR-100-LT. As confirmed in Table 7, this value consistently achieves the best overall accuracy across all imbalance factors, with particularly notable gains in tail classes, validating the theoretical grounding of our noise design.

Table 7. Effect of noise ratio ρ on accuracy.
<table><tr><td rowspan="2">ρ in Eq. 10</td><td colspan="4">IF 50</td><td colspan="4">IF 100</td></tr><tr><td>Head</td><td>Med</td><td>Tail</td><td>All</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td>0.1</td><td>64.0</td><td>48.3</td><td>38.0</td><td>52.9</td><td>64.7</td><td>48.0</td><td>27.9</td><td>47.8</td></tr><tr><td>0.2</td><td>61.3</td><td>49.3</td><td>42.2</td><td>53.0</td><td>64.7</td><td>47.6</td><td>30.1</td><td>48.3</td></tr><tr><td>0.3</td><td>63.2</td><td>47.9</td><td>37.7</td><td>52.4</td><td>62.5</td><td>50.0</td><td>27.5</td><td>47.6</td></tr></table>

## 5.4. Computational Cost Analysis

On ImageNet-LT, using four RTX 4090 GPUs (batch:256), CILA (with BNI) takes 160 seconds per epoch compared to 153 seconds for LA, incurring only a 4.6% overhead while achieving an 8.8% absolute accuracy gain. The added computation consists of lightweight element-wise operations on output logits, requiring no extra forward passes, gradients, or parameters. Thus, CILA achieves substantial performance gains at nearly the same training cost as LA.

## 6. Conclusion

We identified the Non-Target Logit Over-Suppression (NT-LOS) problem in class-wise logit adjustment and formally characterized it at the gradient level. To address this, we proposed CILA, which restores informative gradients via a confusion-based penalty, and BNI, which extends the update to near-boundary samples via batch-adaptive noise injection, consistently improving performance across longtailed benchmarks.

## References

[1] Sumyeong Ahn, Jongwoo Ko, and Se-Young Yun. Cuda: Curriculum of data augmentation for long-tailed recognition. arXiv preprint arXiv:2302.05499, 2023. 6

[2] Ricardo Barandela, E Rangel, Jose Salvador S´ anchez, and´ Francesc J Ferri. Restricted decontamination for the imbalanced training sample problem. In Progress in Pattern Recognition, Speech and Image Analysis: 8th Iberoamerican Congress on Pattern Recognition, CIARP 2003, Havana, Cuba, November 26-29, 2003 Proceedings 8, pages 424– 431. Springer, 2003. 1, 2

[3] Mateusz Buda, Atsuto Maki, and Maciej A Mazurowski. A systematic study of the class imbalance problem in convolutional neural networks. Neural networks, 106:249–259, 2018. 1

[4] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced datasets with labeldistribution-aware margin loss. Advances in neural information processing systems, 32, 2019. 1, 2, 6

[5] Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic minority oversampling technique. Journal of artificial intelligence research, 16:321–357, 2002. 1, 2

[6] Dubing Chen, Yuming Shen, Haofeng Zhang, and Philip HS Torr. Zero-shot logit adjustment. arXiv preprint arXiv:2204.11822, 2022. 2

[7] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. Class-balanced loss based on effective number of samples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9268–9277, 2019. 1, 2

[8] Fei Du, Peng Yang, Qi Jia, Fengtao Nan, Xiaoting Chen, and Yun Yang. Global and local mixture consistency cumulative learning for long-tailed visual recognitions. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 15814–15823, 2023. 2

[9] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88:303–338, 2010. 1

[10] Jintong Gao, He Zhao, Zhuo Li, and Dandan Guo. Enhancing minority classes by mixing: An adaptative optimal transport approach for long-tailed classification. Advances in neural information processing systems, 36:60329–60348, 2023. 2, 6

[11] Hui Han, Wen-Yuan Wang, and Bing-Huan Mao. Borderlinesmote: a new over-sampling method in imbalanced data sets learning. In International conference on intelligent computing, pages 878–887. Springer, 2005. 1, 2

[12] Haibo He and Edwardo A Garcia. Learning from imbalanced data. IEEE Transactions on knowledge and data engineering, 21(9):1263–1284, 2009. 1, 2

[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 6

[14] Chen Huang, Yining Li, Chen Change Loy, and Xiaoou Tang. Learning deep representation for imbalanced classification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5375–5384, 2016. 1

[15] Yan Jin, Mengke Li, Yang Lu, Yiu-ming Cheung, and Hanzi Wang. Long-tailed visual recognition via self-heterogeneous integration with knowledge excavation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 23695–23704, 2023. 6

[16] Ganesh Ramachandra Kini, Orestis Paraskevas, Samet Oymak, and Christos Thrampoulidis. Label-imbalanced and group-sensitive classification under overparameterization. Advances in Neural Information Processing Systems, 34: 18970–18983, 2021. 1

[17] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.(2009), 2009. 1

[18] Miroslav Kubat, Stan Matwin, et al. Addressing the curse of imbalanced training sets: one-sided selection. In Icml, page 179. Citeseer, 1997. 1, 2

[19] Jun Li, Zichang Tan, Jun Wan, Zhen Lei, and Guodong Guo. Nested collaborative learning for long-tailed visual recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6949–6958, 2022. 6

[20] Mengke Li, Yiu-ming Cheung, and Yang Lu. Long-tailed visual recognition via gaussian clouded logit adjustment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6929–6938, 2022. 2

[21] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014. 1

[22] Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou. Exploratory undersampling for class-imbalance learning. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernet ics), 39(2):539–550, 2008. 1, 6

[23] Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar. Long-tail learning via logit adjustment. arXiv preprint arXiv:2007.07314, 2020. 1, 2, 6, 7

[24] Seulki Park, Youngkyu Hong, Byeongho Heo, Sangdoo Yun, and Jin Young Choi. The majority can help the minority: Context-rich minority oversampling for long-tailed classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6887–6896, 2022. 4, 6

[25] Jiawei Ren, Cunjun Yu, Xiao Ma, Haiyu Zhao, Shuai Yi, et al. Balanced meta-softmax for long-tailed visual recog nition. Advances in neural information processing systems, 33:4175–4186, 2020. 1, 2

[26] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, San jeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of com puter vision, 115:211–252, 2015. 1

[27] Li Shen, Zhouchen Lin, and Qingming Huang. Relay backpropagation for effective learning of deep convolutional neural networks. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part VII 14, pages 467–482. Springer, 2016. 1, 2

[28] Jiang-Xin Shi, Tong Wei, Yuke Xiang, and Yu-Feng Li. How re-sampling helps for long-tail learning? Advances in Neural Information Processing Systems, 36:75669–75687, 2023. 1, 2

[29] Minseok Son, Inyong Koo, Jinyoung Park, and Changick Kim. Difficulty-aware balancing margin loss for long-tailed recognition. arXiv preprint arXiv:2412.15477, 2024. 1, 2, 5, 6, 7

[30] Min-Kook Suh and Seung-Woo Seo. Long-tailed recognition by mutual information maximization between latent features and ground-truth labels. In International conference on machine learning, pages 32770–32782. PMLR, 2023. 6

[31] Jingru Tan, Changbao Wang, Buyu Li, Quanquan Li, Wanli Ouyang, Changqing Yin, and Junjie Yan. Equalization loss for long-tailed object recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11662–11671, 2020. 1

[32] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset-supplementary material. Reptilia, 32(400): 1–3, 2017. 1, 6

[33] Binwu Wang, Pengkun Wang, Wei Xu, Xu Wang, Yudong Zhang, Kun Wang, and Yang Wang. Kill two birds with one stone: Rethinking data augmentation for deep long-tailed learning. In The Twelfth International Conference on Learn ing Representations, 2024. 6

[34] Zitai Wang, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, and Qingming Huang. A unified generalization analysis of re-weighting and logit-adjustment for imbalanced learning. Advances in Neural Information Processing Systems, 36:48417–48430, 2023. 1

[35] Han-Jia Ye, Hong-You Chen, De-Chuan Zhan, and Wei-Lun Chao. Identifying and compensating for feature deviation in imbalanced deep learning. arXiv preprint arXiv:2001.01385, 2020. 1

[36] Youngseok Yoon, Sangwoo Hong, Hyungjun Joo, Yao Qin, Haewon Jeong, and Jungwoo Lee. Mix from failure: Confusion-pairing mixup for long-tailed recognition. arXiv preprint arXiv:2411.07621, 2024. 2

[37] Yan Zhao, Weicong Chen, Xu Tan, Kai Huang, and Jihong Zhu. Adaptive logit adjustment loss for long-tailed visual recognition. In Proceedings of the AAAI conference on artificial intelligence, pages 3472–3480, 2022. 2, 6

[38] Beier Zhu, Kaihua Tang, Qianru Sun, and Hanwang Zhang. Generalized logit adjustment: Calibrating fine-tuned models by removing label bias in foundation models. Advances in Neural Information Processing Systems, 36:64663–64680, 2023. 2

[39] Jianggang Zhu, Zheng Wang, Jingjing Chen, Yi-Ping Phoebe Chen, and Yu-Gang Jiang. Balanced contrastive learning for long-tailed visual recognition. In Proceedings of