# DLC: Dynamic Loss Controller for Multi-Objective Image Restoration

Jaewan Ko and Janghoon Choi<sup>⋆</sup>

Major in Data Science Convergence, Graduate School of Data Science Kyungpook National University, Daegu, Republic of Korea {asdf8800, jhchoi09}@knu.ac.kr

Abstract. In this paper, we introduce a metric-guided dynamic loss controller (DLC) for multi-objective image restoration. Conventional image restoration pipelines usually train with a fixed weighted combination of multiple losses, without changing the relative importance of fidelity, perceptual similarity, and no-reference quality during optimization. DLC is an architecture- and loss-term-agnostic training-time controller: it does not modify the restoration architecture or introduce new diferentiable loss terms, but dynamically reweights the existing training losses. During training, DLC periodically evaluates the current model on a small fixed feedback subset and uses the resulting quality metrics to update the loss-weight vector through an LLM-based controller. Because DLC operates on existing loss terms rather than task-specific architectures, the same controller formulation can be instantiated across diverse image restoration training pipelines. We evaluate DLC on three restoration domains: low-light image enhancement, deraining, and real-world superresolution, using both reference-based and no-reference quality metrics. Across these settings, DLC considers metric-dependent trade-ofs during optimization and guides training toward balanced operating points across fidelity and perceptual quality. The results show that DLC can move models toward more favorable operating points across diferent restoration domains, supporting its role as a practical plug-in controller for multi-objective image restoration.

Keywords: Image restoration · Dynamic loss weighting · Multi-objective optimization · LLM-based optimization

## 1 Introduction

Deep image restoration models are commonly optimized with a weighted combination of multiple loss terms [3, 13, 16, 27, 30, 40–42]. Various loss functions have been used: pixel-level losses encourage fidelity to a reference image, structural losses preserve local layout and contrast, perceptual losses promote realistic output through feature-level similarity, and task-specific objectives drive the behavior required by a particular degradation model. In most training pipelines, however, the relative weights of these losses are selected before training and then kept fixed until convergence. This design is convenient and reproducible, but it assumes that one static weighting scheme remains appropriate throughout the entire optimization process.

![](images/a264103c110ecc912742e66af33280b0536d4c27a3b39d605afab43da5eedb20.jpg)  
Fig. 1: Conceptual illustration of the fidelity–perceptual trade-of in image restoration. Fidelity-oriented optimization tends to preserve reference consistency, while perceptualoriented optimization emphasizes visual sharpness and detail. DLC aims to guide training toward a balanced operating point through metric-guided loss reweighting.

This assumption is restrictive because restoration quality is inherently multiobjective. Fidelity-oriented metrics such as PSNR and SSIM [29] reward accurate reconstruction of the reference signal, while perceptual and no-reference metrics may favor sharper textures, stronger contrast, or more natural image statistics [5, 20, 21, 43]. Improving one objective can therefore weaken another: conservative outputs often obtain high PSNR but appear over-smoothed, whereas stronger perceptual emphasis may recover fine details while introducing artifacts or reducing pixel-level fidelity [1]. This perception–distortion tension has been widely discussed in image restoration and motivates the need to consider balanced operating points, which here refer to favorable fidelity–perception tradeofs rather than formally defined Pareto optima [1].

Several adaptive loss-weighting methods have been proposed for multi-task or multi-objective learning, often using task uncertainty, training-loss changes, gradient magnitudes, validation feedback, learned schedules, or multi-objective gradient information as the adjustment signal [4,15,18,22,23,32,38]. These sig nals are useful, but they do not directly represent the evaluation criteria used in restoration benchmarks. Restoration papers are usually judged by a heterogeneous metric set that may include reference fidelity, reference-based perceptual similarity, no-reference quality, and distributional realism. This creates a practical gap: the quantities used to tune the training objective are often diferent from the quantities used to evaluate the final model.

Herein, we propose DLC, a metric-guided dynamic loss controller for multiobjective training. Rather than designing a new backbone or introducing a new diferentiable metric loss, DLC treats loss weighting itself as a feedback-driven training decision. During training, DLC periodically evaluates the current model on a fixed feedback subset sampled from the training distribution. It then constructs a structured feedback state from the current metric profile, recent metric changes, and loss-weight history. An LLM-based controller maps this state to the next loss-weight vector for the following training interval [11, 19, 34]. In this way, DLC converts heterogeneous quality-metric feedback into adaptive loss reweighting while preserving the original model, optimizer, data pipeline, and set of training losses. As illustrated in Fig. 2, DLC adds a metric-guided loss-control branch to the original restoration training loop.

![](images/5960ebd3571a4b6c4df6d9d8b9e0d99d6484e0be1c35439a4edc8407682603e1.jpg)  
Fig. 2: Overview of DLC. At intervention step $k ,$ the current restored output is evaluated by task-relevant metrics to construct a feedback state. The loss controller then produces the next loss-weight vector $\lambda _ { k + 1 }$ for the following training interval.

To validate that our proposed DLC operates over the existing loss space of each restoration model, we evaluate DLC on three restoration domains. In lowlight enhancement [44], it adjusts pixel, perceptual, and SSIM loss weights. In deraining [6], it adjusts reconstruction, consistency, perceptual, and adversarialrelated loss multipliers. In real-world super-resolution [24], it adjusts L2, LPIPS, and CSD objective weights. These instantiations difer in architecture, supervision, and loss design, but share the same feedback-to-action formulation. This makes DLC a training-time plug-in rather than a task-specific architectural modification.

For quantitative evaluation, we evaluate DLC on three restoration domains and seven benchmark settings. In low-light enhancement, DLC improves multiple fidelity, perceptual, and no-reference metrics depending on the dataset. In deraining, DLC shifts the balance between fidelity-oriented and perceptual quality measures. In real-world super-resolution, DLC improves reference fidelity and reference-based perceptual metrics on DRealSR and RealSR. These results do not show that one method improves every metric simultaneously. Instead, they show that metric-guided loss control can steer training toward useful operating points while exposing the remaining trade-ofs among metrics. Contributions.

– We formulate dynamic loss reweighting as a metric-guided training decision for multi-objective restoration, connecting training-time loss control to quality-metric feedback measured on a fixed feedback subset.

– We introduce DLC, an LLM-driven plug-in controller that converts metric feedback and recent training history into adaptive loss-weight updates.

– We instantiate DLC in three heterogeneous pipelines: low-light enhancement, deraining, and real-world super-resolution.

– We provide quantitative and trade-of-oriented evidence across seven benchmarks, showing that DLC can improve key fidelity or perceptual metrics without modifying the underlying backbone architectures.

## 2 Related Work

## 2.1 Composite Losses in Image Restoration

Modern image restoration models are commonly trained with a weighted combination of multiple loss terms. Pixel-wise losses encourage signal fidelity, structural losses preserve local layout and contrast, perceptual losses promote featurelevel similarity, and adversarial or task-specific objectives are often introduced to improve realism or enforce degradation-specific constraints [13,16,27,30]. This composite-loss design is widely used across restoration tasks, including low-light image enhancement, deraining, and real-world super-resolution [6, 9, 24, 26, 28, 44, 45]. However, the relative coeficients of these losses are typically selected before training and then kept fixed throughout optimization. Such a fixed configuration implicitly assumes that the same trade-of among fidelity, perceptual quality, and task-specific behavior remains appropriate during the entire training process. DLC addresses this fixed-weight assumption by using the same set of training losses as the baseline while dynamically adjusting their weights according to metric feedback.

## 2.2 Dynamic Loss Weighting for Multi-Objective Training

Dynamic loss weighting has been studied in multi-task and multi-objective learning [4, 15, 18, 22, 23, 32, 38]. Existing approaches adjust loss weights using signals such as task uncertainty [15], changes in training loss [18], gradient magnitudes [4], or multi-objective gradient information [23, 38]. These methods show that static loss coeficients can be suboptimal when multiple objectives are optimized jointly. However, most existing weighting strategies rely on internal training signals, such as losses or gradients, rather than on the heterogeneous quality metrics used to evaluate image restoration models. In restoration, the final model is often judged by a mixture of fidelity, reference-based perceptual, no-reference, and distributional metrics. DLC difers from conventional loss-balancing methods by treating these quality metrics as training-time feedback for loss-weight control.

## 2.3 Quality Metrics for Image Restoration Feedback

Image restoration methods are evaluated with diverse metrics because no single score captures all aspects of visual quality. PSNR and SSIM [29] are commonly used to measure pixel-level fidelity and structural agreement. Reference-based perceptual metrics such as LPIPS [43] and DISTS [5] estimate perceptual similarity in learned or feature-based spaces. No-reference metrics such as NIQE [21], BRISQUE [20], CLIP-IQA [25], MUSIQ [14], and MANIQA [35] estimate image quality without a ground-truth reference, while distributional metrics such as FID [10] are often used to assess realism. These metrics do not always agree with one another. Improving distortion-oriented fidelity can lead to over-smoothed outputs, whereas emphasizing perceptual quality may introduce artifacts or reduce reference fidelity. DLC is motivated by this metric diversity. Instead of collapsing all metrics into a single scalar reward, DLC preserves the multi-metric profile and uses it as structured feedback for adaptive loss reweighting.

## 2.4 LLM-Based Optimization and Loss Control

Large language models (LLMs) have recently been explored as optimization and decision-making modules beyond text generation. Optimization by PROmpting (OPRO) formulates optimization as a prompting process, where an LLM proposes new candidate solutions from previous solutions and their scores [34]. AgentHPO uses an LLM agent to automate hyperparameter optimization by processing task information and iteratively refining configurations based on historical trials [19]. Recent work also investigates whether LLMs can act as incontext meta-learners for model and hyperparameter selection from dataset metadata [11]. TextGrad further explores text-based feedback as an optimization signal for improving components of compound AI systems [39].

DLC uses an LLM in a narrow and operational role. The LLM does not generate restored images, synthesize labels, or design a new architecture. Instead, it receives a structured feedback state consisting of metric values, metric directions, recent metric changes, current loss weights, and adjustable loss terms. The controller then returns the next loss-weight configuration within the existing set of training losses. This framing makes the LLM a training-time loss controller rather than a generative component. The main contribution of DLC is therefore a metric-guided feedback loop that connects restoration quality evaluation to adaptive loss reweighting.

## 3 Method

## 3.1 Problem Formulation

Let $f _ { \theta }$ denote a restoration model with parameters θ. Most restoration pipelines train the model with a weighted combination of multiple loss terms,

$$
\mathcal { L } ( \boldsymbol { \theta } ; \lambda ) = \sum _ { i = 1 } ^ { N } \lambda _ { i } \mathcal { L } _ { i } ( \boldsymbol { \theta } ) ,\tag{1}
$$

where $\mathcal { L } _ { i }$ is an existing loss term in the original training pipeline and $\lambda _ { i }$ is its corresponding weight. The loss terms may represent diferent objectives, such as pixel fidelity, structural consistency, perceptual similarity, adversarial realism, or task-specific constraints. Standard training uses a fixed weight vector λ throughout optimization. This means that the same relative importance among all losses is imposed from the beginning to the end of training.

DLC changes only this fixed-weight assumption. We divide training into a sequence of intervals. During interval $k ,$ the model is optimized with the current loss-weight vector $\lambda _ { k }$ . Within this interval, the original optimizer updates the model parameters using the weighted objective:

$$
\begin{array} { r } { \theta _ { t + 1 } = \theta _ { t } - \eta _ { t } \nabla _ { \theta } \mathcal { L } ( \theta _ { t } ; \lambda _ { k } ) , \qquad t \in [ k T , ( k + 1 ) T ) , } \end{array}\tag{2}
$$

where $T$ denotes the intervention interval and $\eta _ { t }$ is the learning rate at training step t. Thus, $\lambda _ { k }$ is kept fixed within each interval and updated only at discrete intervention steps.

After the interval, the current model is evaluated on a fixed feedback subset, producing a set of feedback metrics. For each metric $M _ { j }$ , DLC records both its value $m _ { k } ^ { ( j ) }$ and its optimization direction $d _ { j } \in \{ \uparrow , \downarrow \}$ , where ↑ denotes a higheris-better metric and ↓ denotes a lower-is-better metric. A controller then maps the metric state and recent training history to the next loss-weight vector:

$$
\begin{array} { r } { \mathbf { s } _ { k } = \left( \{ ( m _ { k } ^ { ( j ) } , d _ { j } ) \} _ { j = 1 } ^ { J } , \varDelta \mathbf { m } _ { k } , \lambda _ { k } , \mathcal { H } _ { k } , \mathcal { A } \right) , \quad \lambda _ { k + 1 } = \pi _ { \mathrm { L L M } } ( \mathbf { s } _ { k } ) , } \end{array}\tag{3}
$$

where J is the number of feedback metrics, $\varDelta \mathbf { m } _ { k }$ denotes recent metric changes, $\mathcal { H } _ { k }$ is the history of previous metric and weight states, and A is the set of adjustable loss terms. The model architecture, optimizer, data pipeline, and set of training losses are not modified. DLC only changes the loss-weight vector used in the next training interval.

## 3.2 DLC Feedback Loop

DLC wraps an existing training pipeline with a periodic metric-guided feedback loop. At the beginning of training, we select a small fixed feedback subset from the training data and keep it unchanged throughout training to measure metric changes under a consistent condition. Using the same subset at every intervention reduces variation caused by changing sample composition, so metric changes more directly reflect the efect of model updates. This subset is used only for training-time controller feedback, while final performance is reported on the standard benchmark test sets.

At each intervention step, DLC evaluates the current checkpoint on the fixed feedback subset and computes task-relevant quality metrics. These metrics may include fidelity-oriented scores, such as PSNR and SSIM [29]; reference-based perceptual metrics, such as LPIPS [43] or DISTS [5]; and no-reference quality metrics, such as NIQE [21], BRISQUE [20], CLIP-IQA [25], MUSIQ [14], or MANIQA [35], depending on the task. The resulting metric values, recent changes, and current loss weights are summarized into a structured feedback state. This state is then provided to an LLM controller, which returns the loss weights for the next training interval.

DLC does not optimize the feedback metrics directly as diferentiable losses. Instead, it uses them as non-diferentiable feedback to select the next loss-weight configuration, while gradient-based optimization still relies on the original differentiable losses of each pipeline. This is important because many restoration metrics are non-diferentiable, unstable as direct objectives, or not designed as training losses.

## 3.3 Feedback State and Controller Output

The controller input is a structured state rather than raw images. For each intervention step, it contains metric values, metric directions, recent metric changes, current loss weights, and adjustable loss terms. The metric direction specifies whether each metric should be maximized or minimized; for example, higher PSNR, SSIM, CLIP-IQA, MUSIQ, and MANIQA indicate better quality, while lower LPIPS, DISTS, NIQE, BRISQUE, and FID indicate better quality. Rather than collapsing all metrics into a single scalar score, DLC preserves their values and directions so that the controller can interpret each recent change as improvement or degradation.

The feedback state also includes a short task instruction that describes the overall restoration goal, such as improving visual quality according to the feedback metrics while avoiding severe degradation of already strong metrics. The controller uses this information together with recent metric and weight history to determine how the loss weights should change. The output is a structured dictionary of numeric loss weights:

$$
\begin{array} { r } { \lambda _ { k + 1 } = \left\{ \lambda _ { k + 1 } ^ { ( 1 ) } , \lambda _ { k + 1 } ^ { ( 2 ) } , \dots , \lambda _ { k + 1 } ^ { ( N ) } \right\} . } \end{array}\tag{4}
$$

Only these numeric weights are used by the training loop. The controller may also produce a short textual rationale for logging and analysis, but this rationale is not used as an optimization signal.

This design allows DLC to handle heterogeneous metric profiles without requiring a manually designed analytic rule for every task. Instead of assuming a universal update formula, DLC uses the controller to interpret the metric state in the context of the available loss terms and recent training behavior. The role of the LLM is therefore narrow and operational: it maps a structured metric summary to the next loss-weight configuration. In our implementation, we instantiate the controller with LLaMA-3 8B [7] and keep the prompt template, decoding parameters, and structured output format fixed within each experiment. This fixed interface improves reproducibility by ensuring that the optimization process depends on structured numeric outputs rather than free-form textual responses.

## 3.4 Domain Instantiations

DLC is instantiated over the existing loss space of each restoration domain. The controller formulation remains the same, but the adjustable losses and feedback metrics are task-dependent. This reflects the fact that diferent restoration tasks use diferent training objectives and evaluation protocols. Table 1 summarizes the controlled losses for low-light enhancement, deraining, and real-world superresolution. In all cases, DLC reweights only the existing loss terms of each original pipeline and does not introduce task-specific architectural changes. The controlled losses are inherited from the corresponding baseline objectives [6, 24, 44] rather than newly designed for DLC. Thus, the comparison isolates the efect of the loss-weight decision policy.

Table 1: Domain-specific DLC instantiations.
<table><tr><td>Domain</td><td>Controlled losses</td></tr><tr><td>Low-light Enhancement</td><td>pixel, perceptual, SSIM</td></tr><tr><td>Deraining</td><td>reconstruction, consistency, perceptual, adversarial</td></tr><tr><td>Real-world SR</td><td>L2, LPIPS, CSD</td></tr></table>

## 3.5 Training Procedure

The overall DLC training procedure is summarized in Algorithm 1. The procedure only changes the loss weights at discrete intervention steps, while keeping the original model, optimizer, data pipeline, and training budget unchanged. Only the parsed numeric weights are passed to the training loop, while any textual rationale is used only for logging and analysis.

Algorithm 1 DLC Training Procedure   
Require: Model $f _ { \theta } ,$ training data $\mathcal { D } _ { t r a i n } ,$ feedback subset $\mathcal { D } _ { f b } ,$ initial weights $\lambda _ { 0 } ,$   
interval $T$   
Ensure: Trained restoration model $f _ { \theta }$   
1: Initialize intervention index $k  0$   
2: while training budget is not exhausted do   
3: Train $f _ { \theta }$ for one interval using $\lambda _ { k }$   
4: Evaluate the current checkpoint on $\mathcal { D } _ { f b }$   
5: Compute feedback metrics m<sub>k</sub>   
6: Build feedback state $\mathbf { s } _ { k } = \left( \mathbf { m } _ { k } , \varDelta \mathbf { m } _ { k } , \lambda _ { k } , \mathcal { H } _ { k } \right)$   
7: Query the LLM controller with s   
8: Parse the output into $\lambda _ { k + 1 }$   
9: Set $k \gets k + 1$   
10: end while

## 4 Experiments

## 4.1 Experimental Setup

We evaluate DLC on three restoration domains: low-light image enhancement [44], deraining [6], and real-world super-resolution [24]. For each domain, the baseline is the corresponding original pipeline trained with fixed loss weights. DLC uses the same backbone, training data, optimizer, and set of training losses as the baseline, and changes only the loss-weight decision policy during training. For DLC, we use a fixed feedback subset of 10 training samples in each experiment. The same subset is evaluated at every intervention step to provide consistent metric feedback to the controller. All reported results are computed on the standard test sets.

For low-light image enhancement and deraining, we report PSNR, SSIM, LPIPS, NIQE, CLIP-IQA, and BRISQUE. For real-world super-resolution, we follow the broader evaluation protocol used in this domain and report PSNR, SSIM, LPIPS, DISTS, CLIP-IQA, NIQE, MUSIQ, MANIQA, and FID. Higher values are better for PSNR, SSIM, CLIP-IQA, MUSIQ, and MANIQA, while lower values are better for LPIPS, DISTS, NIQE, BRISQUE, and FID. We bold a DLC result only when it improves over the baseline according to the corresponding metric direction.

## 4.2 Main Quantitative Results

Low-light image enhancement. Table 2 reports results on LOL-v1 [30], LOLv2-Syn, and LOL-v2-Real [37]. DLC improves most perceptual and no-reference quality metrics on LOL-v1, with only a minor PSNR drop. On LOL-v2-Syn, DLC provides clear gains in reference fidelity and perceptual similarity, improving PSNR, SSIM, LPIPS, and NIQE. The strongest improvement is observed on LOL-v2-Real, where DLC substantially increases PSNR from 27.39 to 30.74 while also improving structural and perceptual metrics. These results indicate that metric-guided loss reweighting can find a better fidelity–perceptual operating point than the baseline.

Table 2: Low-light image enhancement results. DLC values are bolded when they improve over the baseline.
<table><tr><td>Dataset</td><td>Method PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>CLIP-IQA↑</td><td>BRISQUE↓</td></tr><tr><td>LOL-v1</td><td>Base</td><td>23.60 0.8482</td><td>0.1231</td><td>4.5780</td><td>0.3920</td><td>23.0670</td></tr><tr><td>LOL-v1</td><td>DLC</td><td>23.50 0.8519</td><td>0.0974</td><td>4.4457</td><td>0.4630</td><td>19.9560</td></tr><tr><td>LOL-v2-Syn</td><td>Base</td><td>25.50 0.9362</td><td>0.1195</td><td>4.2735</td><td>0.5261</td><td>13.3023</td></tr><tr><td>LOL-v2-Syn</td><td>DLC</td><td>26.07 0.9450</td><td>0.0395</td><td>4.1369</td><td>0.5025</td><td>13.3490</td></tr><tr><td>LOL-v2-Real</td><td>Base</td><td>27.39 0.9007</td><td>0.0891</td><td>4.5415</td><td>0.4128</td><td>24.7530</td></tr><tr><td>LOL-v2-Real DLC</td><td></td><td>30.74 0.9105</td><td>0.0622</td><td>4.5441</td><td>0.4711</td><td>24.2293</td></tr></table>

Deraining. Table 3 summarizes the deraining results on Rain100L [36] and RealRain1K-H [17]. On Rain100L, DLC improves PSNR, LPIPS, and BRISQUE, showing better reconstruction accuracy and perceptual quality. On RealRain1K-H, DLC improves SSIM, LPIPS, and CLIP-IQA, although PSNR and some noreference metrics decrease. This suggests that DLC does not simply optimize a single metric, but shifts the model toward diferent operating points depending on the dataset and feedback signal.

Table 3: Deraining results. DLC values are bolded when they improve over the baseline.
<table><tr><td>Dataset</td><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>CLIP-IQA↑ BRISQUE</td><td></td></tr><tr><td>Rain100L</td><td>Base</td><td>33.2795</td><td>0.9537</td><td>0.0373</td><td>3.3761</td><td>0.7588</td><td>8.1748</td></tr><tr><td>Rain100L</td><td>DLC</td><td>33.5887</td><td>0.9526</td><td>0.0345</td><td>3.4173</td><td>0.7396</td><td>6.6552</td></tr><tr><td>RealRain1K-H</td><td>Base</td><td>29.6702</td><td>0.9280</td><td>0.1720</td><td>10.4063</td><td>0.2349</td><td>51.2430</td></tr><tr><td>RealRain1K-H</td><td>DLC</td><td>29.1054</td><td>0.9318</td><td>0.1513</td><td>10.6044</td><td>0.2746</td><td>51.6570</td></tr></table>

Real-world super-resolution. Table 4 reports real-world super-resolution results on DRealSR [31] and RealSR [2]. Across both datasets, DLC consistently improves PSNR, SSIM, LPIPS, and DISTS, indicating better reference fidelity and reference-based perceptual alignment under the same super-resolution backbone. DLC also improves MANIQA on both datasets and FID on DRealSR, while CLIP-IQA, NIQE, and MUSIQ show trade-ofs, suggesting that DLC shifts the reconstruction–perception balance rather than uniformly improving all metrics. These results demonstrate that DLC can improve key fidelity and perceptual metrics without modifying the super-resolution architecture.

Table 4: Real-world super-resolution results. DLC values are bolded when they improve over the baseline.
<table><tr><td>Dataset</td><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DISTS↓</td><td>CLIP-IQA↑ NIQE↓ MUSIQ↑ MANIQA↑</td><td></td><td></td><td></td><td>FID↓</td></tr><tr><td>DRealSR</td><td>Base</td><td>28.3197</td><td>0.7804</td><td>0.2960</td><td>0.2169</td><td>0.6970</td><td>6.1792</td><td>66.1088</td><td>0.6161</td><td>130.4879</td></tr><tr><td>DRealSR</td><td>DLC</td><td>29.1599</td><td>0.7974</td><td>0.2806</td><td>0.2124</td><td>0.6847</td><td>6.3306</td><td>64.7712</td><td>0.6166</td><td>130.1198</td></tr><tr><td>RealSR</td><td>Base</td><td>25.5031</td><td>0.7418</td><td>0.2672</td><td>0.2044</td><td>0.6699</td><td>5.5009</td><td>70.1478</td><td>0.6551</td><td>124.1799</td></tr><tr><td>RealSR</td><td>DLC</td><td>26.1454</td><td>0.7490</td><td>0.2582</td><td>0.2003</td><td>0.6652</td><td>5.8489</td><td>68.4462</td><td>0.6627</td><td>124.5460</td></tr></table>

## 4.3 Qualitative Results

Fig. 3 compares the fixed-loss baseline and DLC across restoration domains using the same backbones. The zoomed crops highlight local diferences in texture, structure, and perceptual quality.

Input  
![](images/f51a5557c3edf795ca60d8367aa7b3f7e3cbcda79a5d397a10a92e555b65a4aa.jpg)  
GT  
Fig. 3: Qualitative comparison between the fixed-loss baseline and DLC across restoration domains. Zoomed crops highlight texture, structure, and perceptual diferences.

The qualitative results provide visual examples of the metric-level trends observed in the quantitative evaluation. Compared with the baseline, DLC often yields clearer local structures or more favorable perceptual details in the highlighted crops. The improvements are not uniform across all regions, but they illustrate how metric-guided loss reweighting can shift the visual operating point without changing the underlying restoration architecture.

## 4.4 Comparison of Loss-Weighting Policies

To isolate the efect of the loss-weight policy, we compare DLC on LOL-v1 with fixed-weight and rule-based baselines as well as GradNorm [4], Uncertainty Weighting [15], and CoV Weighting [8]. The fixed-weight variant uses DLC’s final loss-weight vector from the start of an independent training run, while the rule-based variant uses a hand-designed metric-trend heuristic. All methods otherwise use the same backbone, dataset, loss components, optimizer, and training budget.

Table 5 shows that diferent weighting policies lead to diferent fidelity– perception trade-ofs. DLC achieves the highest PSNR while remaining competitive on SSIM and perceptual quality, whereas other methods perform better on selected perceptual or no-reference metrics. These results indicate that the resulting operating point depends on the feedback signal and weight-update policy rather than on adaptive weighting alone. Unlike the other adaptive methods, DLC determines weight updates from heterogeneous quality-metric feedback together with recent loss-weight history.

Table 5: Comparison of loss-weighting policies on LOL-v1. All methods use the same backbone, dataset, loss components, optimizer, and training budget, and difer only in the loss-weight decision policy. The best result for each metric is bolded.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>CLIP-IQA↑</td><td>BRISQUE↓</td></tr><tr><td>Fixed weight</td><td>23.11</td><td>0.8548</td><td>0.0926</td><td>4.4830</td><td>0.4580</td><td>21.0060</td></tr><tr><td>Rule-based</td><td>23.28</td><td>0.8485</td><td>0.0996</td><td>4.4940</td><td>0.4540</td><td>20.8500</td></tr><tr><td>GradNorm</td><td>22.81</td><td>0.8473</td><td>0.0934</td><td>4.3496</td><td>0.4720</td><td>19.6089</td></tr><tr><td>Uncertainty Weighting</td><td>23.02</td><td>0.8480</td><td>0.0946</td><td>4.3853</td><td>0.4548</td><td>19.9532</td></tr><tr><td>CoV Weighting</td><td>22.77</td><td>0.8482</td><td>0.0938</td><td>4.3791</td><td>0.4845</td><td>19.2890</td></tr><tr><td>DLC (Ours)</td><td>23.50</td><td>0.8519</td><td>0.0974</td><td>4.4457</td><td>0.4625</td><td>19.9560</td></tr></table>

Fig. 4 qualitatively compares all loss-weighting policies and illustrates the visual diferences induced by each strategy.

![](images/3ebe6b2bcba3e8dbc887decc3e4b340c0c6efa3bbdaaaa0e402e5fc78d36025d.jpg)  
Fig. 4: Qualitative comparison of loss-weighting policies on LOL-v1. The results illustrate the diferent visual characteristics induced by fixed, rule-based, and adaptive weighting strategies.

## 4.5 Controller LLM Sensitivity

We evaluate Qwen2.5-7B [33], Mistral-7B [12], and LLaMA-3 8B [7] under the same DLC setting, and observe comparable performance across all three controllers, indicating that DLC is not tied to a specific LLM backbone.

Table 6: Controller LLM sensitivity on LOL-v1. Best values are bolded.
<table><tr><td>Controller</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>CLIP-IQA↑</td><td>BRISQUE↓</td></tr><tr><td>Qwen2.5-7B</td><td>23.2304</td><td>0.8505</td><td>0.0978</td><td>4.3938</td><td>0.4600</td><td>19.4955</td></tr><tr><td>Mistral-7B</td><td>23.2334</td><td>0.8505</td><td>0.0979</td><td>4.4092</td><td>0.4599</td><td>19.5009</td></tr><tr><td>LLaMA-3 8B</td><td>23.5045</td><td>0.8519</td><td>0.0974</td><td>4.4457</td><td>0.4625</td><td>19.9560</td></tr></table>

## 4.6 Loss-Weight Trajectory Analysis

To analyze how DLC changes the training objective over time, Fig. 5 shows the loss-weight trajectory and a representative controller decision on LOL-v1. The pixel and perceptual weights gradually increase, while the structural weight remains relatively stable, indicating that DLC mainly adjusts the reconstruction– perceptual balance. When fidelity metrics improve but perceptual feedback degrades, the controller increases the perceptual weight for the next interval. These results show that DLC dynamically reweights existing losses according to observed metric trade-ofs rather than following a fixed update heuristic.

![](images/5c52fcff25f54c5262ac8e9676bf6117d4c5f515fc6aad1dded26d920fc74c15.jpg)  
Fig. 5: Loss-weight trajectory and controller decision example on LOL-v1. Left: representative trajectories of the pixel, SSIM, and perceptual loss weights over intervention steps. Right: an example controller decision at a single intervention step, where fidelity-oriented metrics improve while perceptual or no-reference feedback degrades. DLC adjusts the next loss weights according to the observed metric trade-of. <sup>†</sup>The loss-weight vector is ordered as $( \lambda _ { \mathrm { p i x } } , \lambda _ { \mathrm { s s i m } } , \lambda _ { \mathrm { p e r c } } )$

## 4.7 Training-Time and Test-Time Cost

DLC introduces additional cost only during training, due to periodic metric evaluation on the fixed feedback subset and controller inference. Table 7 summarizes the measured overhead from representative runs across the three restoration domains. In low-light enhancement, each intervention takes approximately 7–8 seconds, resulting in a total overhead of 2.12–6.36 minutes. In deraining, each intervention takes approximately 7–9 seconds, with a total overhead of 1.49–2.17 minutes. Since the controller is invoked only at discrete intervals, the relative overhead remains below 0.2% in these two domains.

For real-world super-resolution, DLC introduces a per-call overhead of 7.07 seconds and a total overhead of 0.68 minutes, corresponding to 1.69% of the measured training time. Although the relative overhead is higher than in the other two domains, it is still incurred only during training. Importantly, DLC does not add any module to the restoration architecture and does not change the inference pipeline. Therefore, the final trained model has the same number of parameters and the same test-time cost as the baseline.

Table 7: Training-time overhead of DLC measured from representative runs. The reported overhead is incurred only during training, and the final model has no additional test-time cost.
<table><tr><td>Domain</td><td>Per-call overhead</td><td>Total overhead</td><td>Relative overhead</td></tr><tr><td>Low-light enhancement</td><td>7.49–7.71 s</td><td>2.12–6.36 min</td><td>&lt; 0.2%</td></tr><tr><td>Deraining</td><td>6.94–9.29 s</td><td>1.49–2.17 min</td><td>0.16–0.18%</td></tr><tr><td>Real-world SR</td><td>7.07 s</td><td>0.68 min</td><td>1.69%</td></tr></table>

## 5 Conclusion

We presented DLC, a metric-guided dynamic loss controller for multi-objective restoration training. DLC preserves the existing model, optimizer, data pipeline, and training losses, and updates only the loss-weight vector at discrete training intervals using metric feedback from a fixed feedback subset. Through an LLM-based controller, DLC maps heterogeneous quality metrics, recent metric changes, and loss-weight history to adaptive loss reweighting.

Across low-light image enhancement, deraining, and real-world super-resolution, DLC improves key fidelity or perceptual metrics while making the remaining trade-ofs explicit. The ablation on loss-weight update policies further indicates that the benefit comes not merely from changing weights, but from using metric feedback to guide the update direction.

DLC is a training-time plug-in, and the final model incurs no additional testtime cost. These results suggest that metric-guided loss control is a practical way to steer existing restoration pipelines toward more favorable operating points. Future work includes more systematic analysis of feedback subset selection, intervention frequency, and controller design.

Acknowledgements. This research was supported by the Republic of Korea Government, funded by the Ministry of Science and ICT (National Research Foundation of Korea (NRF) and the Institute for Information & Communications Technology Planning & Evaluation (IITP) [RS-2023-00242528(10%), RS-2024-00437756(50%)]), the Ministry of Education [BK21 FOUR:2120241015413(20%), Glocal 30 Program(20%)].

## References

1. Blau, Y., Michaeli, T.: The perception-distortion tradeof. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6228–6237 (2018)

2. Cai, J., Zeng, H., Yong, H., Cao, Z., Zhang, L.: Toward real-world single image super-resolution: A new benchmark and a new model. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3086–3095 (2019)

3. Chen, L., Chu, X., Zhang, X., Sun, J.: Simple baselines for image restoration. In: European conference on computer vision. pp. 17–33. Springer (2022)

4. Chen, Z., Badrinarayanan, V., Lee, C.Y., Rabinovich, A.: Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In: International conference on machine learning. pp. 794–803. PMLR (2018)

5. Ding, K., Ma, K., Wang, S., Simoncelli, E.P.: Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence 44(5), 2567–2581 (2020)

6. Dong, G., Zheng, T., Cao, Y., Qing, L., Ren, C.: Channel consistency prior and self-reconstruction strategy based unsupervised image deraining. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 7469–7479 (2025)

7. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al.: The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024)

8. Groenendijk, R., Karaoglu, S., Gevers, T., Mensink, T.: Multi-loss weighting with coeficient of variations. In: 2021 IEEE winter conference on applications of computer vision (WACV). pp. 1468–1477. IEEE (2021)

9. Guo, C., Li, C., Guo, J., Loy, C.C., Hou, J., Kwong, S., Cong, R.: Zero-reference deep curve estimation for low-light image enhancement. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1780–1789 (2020)

10. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)

11. Hili, Y.A.E., Thomas, A., Tiomoko, M., Benechehab, A., Léger, C., Ancourt, C., Kégl, B.: Llms as in-context meta-learners for model and hyperparameter selection. arXiv preprint arXiv:2510.26510 (2025)

12. Jiang, A.Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D.S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L.R., Lachaux, M.A., Stock, P., Scao, T.L., Lavril, T., Wang, T., Lacroix, T., Sayed, W.E.: Mistral 7b (2023), https://arxiv.org/abs/2310.06825

13. Johnson, J., Alahi, A., Fei-Fei, L.: Perceptual losses for real-time style transfer and super-resolution. In: European conference on computer vision. pp. 694–711. Springer (2016)

14. Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5148–5157 (2021)

15. Kendall, A., Gal, Y., Cipolla, R.: Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7482–7491 (2018)

16. Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z., et al.: Photo-realistic single image superresolution using a generative adversarial network. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4681–4690 (2017)

17. Li, W., Zhang, Q., Zhang, J., Huang, Z., Tian, X., Tao, D.: Toward real-world single image deraining: A new benchmark and beyond. arXiv preprint arXiv:2206.05514 (2022)

18. Liu, S., Johns, E., Davison, A.J.: End-to-end multi-task learning with attention. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1871–1880 (2019)

19. Liu, S., Gao, C., Li, Y.: Large language model agent for hyper-parameter optimization. arXiv preprint arXiv:2402.01881 (2024)

20. Mittal, A., Moorthy, A.K., Bovik, A.C.: No-reference image quality assessment in the spatial domain. IEEE Transactions on image processing 21(12), 4695–4708 (2012)

21. Mittal, A., Soundararajan, R., Bovik, A.C.: Making a “completely blind” image quality analyzer. IEEE Signal processing letters 20(3), 209–212 (2013)

22. Ren, M., Zeng, W., Yang, B., Urtasun, R.: Learning to reweight examples for robust deep learning. In: International conference on machine learning. pp. 4334– 4343. PMLR (2018)

23. Sener, O., Koltun, V.: Multi-task learning as multi-objective optimization. Advances in neural information processing systems 31 (2018)

24. Sun, L., Wu, R., Ma, Z., Liu, S., Yi, Q., Zhang, L.: Pixel-level and semantic-level adjustable super-resolution: A dual-lora approach. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2333–2343 (2025)

25. Wang, J., Chan, K.C., Loy, C.C.: Exploring clip for assessing the look and feel of images. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 2555–2563 (2023)

26. Wang, X., Xie, L., Dong, C., Shan, Y.: Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1905–1914 (2021)

27. Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., Change Loy, C.: Esrgan: Enhanced super-resolution generative adversarial networks. In: Proceedings of the European conference on computer vision (ECCV) workshops. pp. 0–0 (2018)

28. Wang, Y., Wan, R., Yang, W., Li, H., Chau, L.P., Kot, A.: Low-light image enhancement with normalizing flow. In: Proceedings of the AAAI conference on artificial intelligence. vol. 36, pp. 2604–2612 (2022)

29. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)

30. Wei, C., Wang, W., Yang, W., Liu, J.: Deep retinex decomposition for low-light enhancement. arXiv preprint arXiv:1808.04560 (2018)

31. Wei, P., Xie, Z., Lu, H., Zhan, Z., Ye, Q., Zuo, W., Lin, L.: Component divideand-conquer for real-world image super-resolution. In: European conference on computer vision. pp. 101–117. Springer (2020)

32. Xu, H., Zhang, H., Hu, Z., Liang, X., Salakhutdinov, R., Xing, E.: Autoloss: Learning discrete schedules for alternate optimization (2018), https://arxiv.org/abs/ 1810.02442

33. Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., Qiu, Z.: Qwen2.5 technical report (2025), https://arxiv.org/abs/2412.15115

34. Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q.V., Zhou, D., Chen, X.: Large language models as optimizers. In: International Conference on Learning Representations. vol. 2024, pp. 12028–12068 (2024)

35. Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J., Yang, Y.: Maniqa: Multi-dimension attention network for no-reference image quality assessment. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1191–1200 (2022)

36. Yang, W., Tan, R.T., Feng, J., Liu, J., Guo, Z., Yan, S.: Deep joint rain detection and removal from a single image. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1357–1366 (2017)

37. Yang, W., Wang, S., Fang, Y., Wang, Y., Liu, J.: From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3063–3072 (2020)

38. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C.: Gradient surgery for multi-task learning. Advances in neural information processing systems 33, 5824–5836 (2020)

39. Yuksekgonul, M., Bianchi, F., Boen, J., Liu, S., Huang, Z., Guestrin, C., Zou, J.: Textgrad: Automatic "diferentiation" via text. arXiv preprint arXiv:2406.07496 (2024)

40. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer: Eficient transformer for high-resolution image restoration. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5728–5739 (2022)

41. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H., Shao, L.: Learning enriched features for real image restoration and enhancement. In: European conference on computer vision. pp. 492–511. Springer (2020)

42. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H., Shao, L.: Multi-stage progressive image restoration. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14821–14831 (2021)

43. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)

44. Zhang, T., Liu, P., Lu, Y., Cai, M., Zhang, Z., Zhang, Z., Zhou, Q.: Cwnet: Causal wavelet network for low-light image enhancement. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8789–8799 (2025)

45. Zhang, Y., Zhang, J., Guo, X.: Kindling the darkness: A practical low-light image enhancer. In: Proceedings of the 27th ACM international conference on multimedia. pp. 1632–1640 (2019)