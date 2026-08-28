# FAN-LoRA: A Fourier-Adaptive Nonlinear Low-Rank Adaptor for Medical Foundation Model Domain Adaptation

Ziquan Liu<sup>1</sup>, Zhewei Zhu<sup>1</sup>, and Xuyang Shi<sup>1∗</sup>

<sup>1</sup>School of Information and Control Engineering, Southwest University of Science and Technology Mianyang, China

Abstract—The advent of vision foundation models, notably the Segment Anything Model (SAM), has catalyzed significant advancements in natural image segmentation. However, their direct transfer to medical imaging remains severely bottlenecked by profound domain gaps, such as cross-modality and crosscenter shifts. Existing Parameter-Efficient Fine-Tuning (PEFT) methods facilitate the adaptation of SAM to medical domains; nevertheless, they frequently suffer from performance degradation under severe distribution shifts. This vulnerability primarily stems from the implicit entanglement of heterogeneous frequency components within a shared low-rank subspace, which directly exacerbates sub-optimal structural alignment and localized boundary blurring. To overcome this representational bottleneck, we propose the Fourier-Adaptive Nonlinear Low-Rank Adaptor (FAN-LoRA), a novel frequency-decoupled finetuning architecture. FAN-LoRA explicitly separates the optimization space by employing a B-spline-driven low-pass branch for global structural alignment, synergistically coupled with a discrete Fourier high-pass branch for local textural compensation. Extensive experiments across three challenging crossmodality and cross-center benchmarks demonstrate that FAN-LoRA consistently outperforms state-of-the-art PEFT baselines. Compared to the strongest competitors, our method achieves consistent improvements in average Dice scores and notable reductions in boundary errors, while maintaining a compact module size without compromising computational efficiency.

Index Terms—Medical Image Segmentation; Parameter-Efficient Fine-Tuning (PEFT); Domain Adaptation; Frequency Decoupling; Foundation Models

## I. INTRODUCTION

The advent of foundation models has catalyzed a paradigm shift in computer vision. Notably, the Segment Anything Model (SAM) [1], pre-trained on the SA-1B dataset, has demonstrated remarkable zero-shot generalization in natural image segmentation. However, the direct transfer of SAM to medical image analysis remains severely bottlenecked by profound domain gaps [2], [3]. Unlike natural images, medical modalities exhibit complex underlying physical mechanisms, subtle intensity variations, and highly heterogeneous lesion morphologies [4], [5]. Consequently, applying pre-trained SAM directly to medical images frequently yields sub-optimal boundary delineation and significant performance degradation in low-contrast regions [3], [6], [7].

To bridge this divide, contemporary research has pivoted to adapt SAM for specific medical domains. While large-scale full fine-tuning efforts, such as MedSAM [8], have established robust domain-specific baselines, their prohibitive computational costs hinder rapid deployment in specialized clinical settings. Consequently, Parameter-Efficient Fine-Tuning (PEFT) has emerged as the predominant approach [9]–[13]. Methods leveraging Low-Rank Adaptation (LoRA) [14], [15] or domain-specific adapters [16] successfully inject task-specific knowledge with minimal parameter overhead. Nevertheless, current PEFT-adapted medical SAMs still suffer from severe performance degradation when confronting cross-center data or cross-modality domain shifts.

Existing PEFT methods implicitly entangle low-frequency structural drifts and high-frequency textural perturbations within a single low-rank subspace. This entanglement arises because domain discrepancies in medical imaging are inherently frequency-heterogeneous: low-frequency components dominate global intensity and anatomical shifts, while highfrequency components capture sensor-specific artifacts and lesion boundary details [17]–[19]. Consequently, gradient conflicts during optimization often lead to either over-smoothed representations or noise overfitting. This entanglement directly exacerbates the aforementioned sub-optimal boundary delineation and performance degradation in low-contrast regions.

To resolve this fundamental bottleneck, we propose the Fourier-Adaptive Nonlinear Low-Rank Adaptor (FAN-LoRA), a novel frequency-decoupled fine-tuning architecture. Diverging from conventional spatial-domain PEFT methods, we fundamentally re-model medical domain adaptation as a synergistic optimization of “low-frequency structural alignment” and “high-frequency residual compensation.”

Specifically, FAN-LoRA features a dual-branch architecture. For the low-pass branch, which governs global anatomical structure alignment, we draw inspiration from the continuous B-spline functions utilized in recent non-linear adapters [20], [21]. Recognizing that the Fourier transform of B-spline basis functions exhibits a smooth low-pass response with polynomial decay in the frequency domain, we reconstruct the Adaptive Nonlinear Layer (ANL) with custom normalization and enhanced non-linear activations. This transforms the module into a highly efficient structural smoother, capturing the global intensity mapping between source and target domains on a low-frequency manifold. Concurrently, to compensate for the inevitable loss of local texture inherent in low-pass filtering, we introduce a complementary, explicit high-frequency branch. This branch constructs a sinusoidal projection using a fixed sinusoidal projection matrix, which approximates a subset of Fourier basis functions in the real space, effectively mapping spatial features into a discrete spectral subspace [22]. Here, the model optimizes a highly parameter-efficient set of spectral parameters to capture target-specific highfrequency shifts before projecting back to the spatial domain. This cohesive design achieves explicit feature-level frequency decoupling, ensuring structural fidelity without compromising local detail.

The main contributions of this paper are summarized as follows:

• Analytical Insight: We identify and empirically analyze the performance bottleneck of frequency coupling in PEFT for cross-domain medical segmentation. We demonstrate that decoupling spatial gradients into Bspline-driven low-pass and Fourier-driven high-pass subspaces effectively mitigates optimization conflicts.

• Architectural Design: We propose the FAN-LoRA architecture, which explicitly separates adaptation into a Bspline-driven low-pass branch for global structural alignment and a discrete Fourier high-pass branch for local textural compensation, rather than implicitly relying on shared parameterization.

• Empirical Effectiveness: Extensive experiments across three challenging domain adaptation scenarios, including cross-modality and cross-center, demonstrate that FAN-LoRA achieves highly competitive performance compared to existing state-of-the-art PEFT methods in both accuracy and robustness.

## II. RELATED WORK

## A. Foundation Models in Medical Image Segmentation

Prior to the advent of foundation models, the landscape of medical image segmentation was dominated by heavily supervised, task-specific architectures, most notably the nnU-Net framework [23], which excelled through empirical pipeline optimization. The introduction of SAM [1] shifted the paradigm towards promptable, universal segmentation. However, the zero-shot application of SAM to medical modalities highlighted significant domain gaps, with recent comprehensive evaluations systematically characterizing its performance limitations on multi-modal medical data [3].

To contextualize SAM for clinical use, comprehensive finetuning strategies were initially adopted. MedSAM [8] aggregated a large-scale multi-modal dataset to train a universal medical baseline, demonstrating that domain-specific finetuning substantially improves segmentation accuracy. While these fully fine-tuned models possess strong generalized representations, their massive computational requirements render them inflexible for continuous adaptation to specific clinical centers or novel imaging protocols, highlighting the necessity for parameter-efficient alternatives.

## B. Parameter-Efficient Fine-Tuning and Adapters

Parameter-efficient fine-tuning (PEFT) techniques aim to adapt large-scale pre-trained models to downstream tasks by updating only a minimal subset of parameters [9]. Within this paradigm, adapter-based methods inject lightweight trainable modules into frozen backbones. [24] first introduced this concept in NLP, which was subsequently adapted to general vision tasks [10], and specifically tailored for medical SAM architectures via domain-specific adapters [16]. LoRA [14] offers an alternative approach by optimizing incremental lowrank weight matrices without modifying the base model’s architecture. SAMed [15] pioneered the application of LoRA to SAM’s image encoder for medical image segmentation, demonstrating that low-rank updates effectively inject domainspecific knowledge with minimal parameter overhead. Recent advancements further optimize this paradigm; for instance, DeLoRA [25] decouples magnitude and direction updates, while SHiRA [26] utilizes sparse high-rank adaptations for rapid fusion.

Despite their efficiency and structural improvements, standard linear PEFT methods face representational bottlenecks when modeling highly non-linear medical domain shifts. To address this, non-linear PEFT variants have been explored. Notably, AuroRA [20] introduces an Adaptive Nonlinear Layer utilizing learnable B-spline basis functions, effectively expanding the approximation capacity of low-rank spaces. However, while AuroRA improves general gradient stability, it continues to operate within a unified parametric space. Consequently, it lacks an explicit mechanism to disentangle the conflicting high- and low-frequency gradients inherent in complex medical cross-modality shifts.

## C. Frequency-Domain Learning for Domain Adaptation

Frequency-domain analysis has a rich history in mitigating domain shift in medical imaging. Classical unsupervised domain adaptation (UDA) methods, such as FDA [18], demonstrated that swapping low-frequency Fourier amplitude spectra effectively transfers visual styles while preserving highfrequency semantic structures.

More recently, frequency-aware mechanisms have been integrated into the PEFT paradigm. FourierFT [22] treats spatial weight updates as spectral coefficients, optimizing a sparse set of Fourier parameters to maximize efficiency in general vision and language tasks. In the medical domain, AdaptFR-CNet [27] incorporates frequency-domain consistency regularization within a semi-supervised PEFT framework, leveraging frequency cues to enhance segmentation performance with limited labeled data. Similarly, FreqFiT [28] introduces a frequency-based fine-tuning module inserted between ViT blocks, capturing global anatomical dependencies that spatialdomain PEFT methods often overlook. However, these existing frequency-based adapters apply spectral operations without structurally separating the optimization of low-frequency structural priors from high-frequency textural details.

![](images/1a106f5b7a2d4994c0fa2177e071f58b4b28a54b2127d95b99eef025743bdedd.jpg)  
Fig. 1. Overall architecture of the proposed Fourier-Adaptive Nonlinear Low-Rank Adaptor (FAN-LoRA). The module fundamentally decouples spatial gradients by projecting input features into two parallel branches: a B-spline-driven Adaptive Nonlinear Layer (ANL) for low-frequency global structural alignment, and an explicit discrete Fourier branch for high-frequency local detail compensation.

Our proposed FAN-LoRA departs from this paradigm by enforcing explicit feature-level frequency decoupling—utilizing continuous splines strictly for low-frequency structural alignment and discrete Fourier transforms strictly for highfrequency residual compensation.

## III. METHODOLOGY

To address the domain shift issues encountered by medical foundation models when applied to varying modalities and clinical center data, we propose a parameter-efficient finetuning module based on frequency decoupling, the Fourier-Adaptive Nonlinear Low-Rank Adaptor (FAN-LoRA), as shown in Fig. 1. The core idea is that domain shifts in medical images manifest in the frequency domain as a mixture of lowfrequency structure changes and high-frequency detail variations. Traditional linear adapters tend to lose high-frequency details or overfit low-frequency noise during optimization. Therefore, FAN-LoRA introduces parallel branches comprising a B-spline Adaptive Nonlinear Layer (ANL) with low-pass characteristics and an explicit lightweight Fourier high-pass filter.

## A. Overall Architecture

Given the frozen weight matrix $\mathbf { W } _ { 0 } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ of the pretrained model and the input feature $x \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ , the forward propagation of FAN-LoRA consists of the base linear mapping, a low-pass adaptation branch and a high-pass adaptation branch:

$$
y = \mathbf { W } _ { 0 } x + H _ { \mathrm { l o w } } ( x ) + H _ { \mathrm { h i g h } } ( x )\tag{1}
$$

where $H _ { \mathrm { l o w } } ( x )$ and $H _ { \mathrm { h i g h } } ( x )$ denote the low-frequency structural adaptation and high-frequency residual compensation, respectively.

## B. Low-Pass Adaptation via B-Spline ANL

To capture global, smooth domain distribution changes, we introduce the ANL in the low-rank space. We first project the features to a low-rank dimension r using the down-projection matrix $\mathbf { A } \in \mathbb { R } ^ { r \times d _ { \mathrm { i n } } }$ , process them through the ANL module, and then map them back to the original space via the upprojection matrix $\mathbf { B } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$

$$
H _ { \mathrm { l o w } } ( x ) = \frac { \alpha } { r } \mathbf { B } \cdot \Phi _ { \mathrm { A N L } } ( \mathbf { A } \cdot \mathcal { D } ( x ) )\tag{2}
$$

where r is the intrinsic rank, and α is a scaling factor controlling the adaptation magnitude. The stochastic dropout $\mathcal { D } ( \cdot )$ , inherited from AuroRA [20], is applied before the lowrank projection to regularize the bottleneck subspace and mitigate overfitting of the B-spline basis functions to sourcedomain noise patterns. $\Phi _ { \mathrm { A N L } } ( \cdot )$ represents the ANL composed of a SiLU activation and a B-spline residual branch.

The ANL module contains a basic non-linear transformation and a residual branch based on B-splines. For the input feature $z = \mathbf { A } x$ , we apply LayerNorm followed by a nonlinear transformation and a spline-based residual mapping. In practice, we adopt a spline order $p = 3$ with a fixed grid size of $G = 5$ , uniformly distributed over the normalized input range [20]. Its core lies in the aggregated output of the B-splines:

$$
\mathrm { S p l i n e } ( z ) = \sum _ { i = 1 } ^ { G } w _ { i } \mathcal { B } _ { i , p } ( \mathrm { L a y e r N o r m } ( z ) )\tag{3}
$$

where $\begin{array} { r } { B _ { i , p } ( \cdot ) } \end{array}$ is the B-spline basis of degree $p ,$ and $w _ { i }$ denotes the learnable spline weights.

From a signal processing perspective, the Fourier transform of a p-degree B-spline basis function is proportional to $\mathrm { s i n c } ^ { p + 1 } ( f )$ , whose magnitude exhibits increasingly rapid decay at high frequencies as the spline order increases [29]. This property implies that B-spline basis functions act as smooth low-pass filters with a polynomially decaying frequency response [30]. Consequently, B-spline-based interpolation can be interpreted as a low-pass filtering process that suppresses highfrequency oscillations and noise while preserving the underlying low-frequency structure [31]. Such smoothing behavior has been widely exploited in signal and image processing to achieve stable approximation of noisy data [31]. In the context of medical imaging, this property is particularly beneficial, as it promotes smooth structural alignment across domains and facilitates the learning of consistent global anatomical representations.

## C. High-Pass Compensation via Lightweight FourierFT

While the low-pass branch effectively aligns the global domain distribution, it tends to smooth out minute textures and sharp boundaries of medical lesions $( \mathrm { e . g . }$ , spiculations on tumor margins). To address this, we design a lightweight highpass Fourier branch $H _ { \mathrm { h i g h } } ( x )$ for frequency compensation.

We explicitly define the high-frequency basis as sine waves in the discrete space. Assuming the number of sampled high frequencies is $n = 1 6$ (default; see ablation in Table X), we uniformly sample the high-frequency band $\omega \in \left[ \frac { \pi } { 2 } , \pi \right)$ . By interpreting the permutation-invariant channel dimensions as a discrete sequential signal, we project the features into an implicit spectral subspace—a technique conceptually aligned with implicit neural representations. Since the Nyquist frequency of discrete signals is π, this half-open interval heuristically emphasizes higher-frequency components while strictly avoiding the singularity at $\pi ,$ , ensuring no spectral parameters are wasted. The high-frequency projection matrices $\mathbf { U } _ { \mathrm { h i g h } } \in$ $\mathbb { R } ^ { d _ { \mathrm { i n } } \times n }$ and $\mathbf { V } _ { \mathrm { h i g h } } \in \mathbf { \bar { \mathbb { R } } } ^ { d _ { \mathrm { o u t } } \times \bar { n } }$ are defined as:

$$
\mathbf { U } _ { \mathrm { h i g h } } ^ { ( i , j ) } = \frac { 1 } { \sqrt { d _ { \mathrm { i n } } } } \sin ( i \cdot \omega _ { j } ) , \quad \mathbf { V } _ { \mathrm { h i g h } } ^ { ( k , j ) } = \frac { 1 } { \sqrt { d _ { \mathrm { o u t } } } } \sin ( k \cdot \omega _ { j } )\tag{4}
$$

where $i \in \{ 1 , \ldots , d _ { \mathrm { i n } } \}$ and $k \in \{ 1 , \dots , d _ { \mathrm { o u t } } \}$ are feature dimension indices, and $\omega _ { j }$ is the j-th sampled high frequency. These two matrices are generated dynamically and require no gradient calculations.

The forward process of the high-pass branch is implemented via a learnable high-frequency spectral modulation vector $S _ { \mathrm { f r e q } } \in \mathbb { R } ^ { n }$

$$
H _ { \mathrm { h i g h } } ( x ) = \mathbf { V } _ { \mathrm { h i g h } } ( S _ { \mathrm { f r e q } } \odot ( \mathbf { U } _ { \mathrm { h i g h } } ^ { \top } x ) )\tag{5}
$$

This branch projects features into a high-frequency subspace via fixed sinusoidal bases, modulates them with a learnable spectral vector, and reconstructs the residual in the original feature space. Compared to conventional approaches, this design avoids explicitly constructing large dense weight matrices in the high-frequency space, achieving extreme parameter efficiency (requiring only n parameters).

![](images/69df5f7ada1a4cb80b73128d69283eba543a2c6387e8903c0f5fc5889878f3d3.jpg)

![](images/f31a6991c8ddc2a22baf34b51429fcefd422ea125a4a0281d70bab165fa7e5f3.jpg)  
Fig. 2. Visualization of the discrete Fourier projection basis matrix $\mathbf { U } _ { \mathrm { h i g h } } .$ The columns are generated via sinusoidal sampling along discrete integer dimensions, capturing distinct high-frequency components to effectively extract texture residuals without excessive parameter overhead.

The column vectors of the Fourier projection basis $\mathbf { U } _ { \mathrm { h i g h } }$ are generated by sinusoidal sampling along discrete integer dimensions, as shown in Fig. 2. The lowest frequency component (0.5π) exhibits a regular sinusoidal pattern due to its integer period; the mid-frequency components suffer from discrete sampling mismatch due to their non-integer periods, resulting in natural waveform distortion.

The effectiveness of FAN-LoRA in MedSAM domain adaptation stems from its gradient decoupling in the frequency domain. In traditional LoRA fine-tuning, the updating of the weight matrix must simultaneously accommodate largescale pixel drifts (low frequency) and local texture mutations (high frequency). This often leads to optimization dilemmas in medical multi-modality data (e.g., transferring from CT to MRI). FAN-LoRA explicitly resolves this dilemma: the B-spline branch smoothly absorbs large-scale low-frequency domain shifts, while the Fourier branch precisely compensates for high-frequency structural boundaries, achieving robust and decoupled domain adaptation.

## IV. EXPERIMENTS

## A. Datasets

We evaluate FAN-LoRA on three representative medical image segmentation benchmarks under cross-modality and crosscenter settings, as shown in Table I. Specifically, MM-WHS 2017 [32] is used for cross-modality cardiac segmentation, with MR as the source domain and CT as the target domain. Promise 12 [33] and NCI-ISBI [34] are employed for crosscenter prostate segmentation under different scanner conditions. FLARE 22 [35] and CHAOS [36] are adopted for crossmodality abdominal organ segmentation, where CT serves as the source domain and MRI as the target domain. These datasets cover diverse anatomical structures and domain shift patterns, enabling a comprehensive evaluation of adaptation performance.

## B. Implementation Details

All experiments were conducted on a single NVIDIA Tesla V100 GPU (32GB). We adopted the officially pre-trained MedSAM model as our backbone architecture. During finetuning, all parameters of the pre-trained backbone were kept frozen, and gradients were exclusively computed for the newly inserted FAN-LoRA modules. For data preprocessing, input images and their corresponding segmentation masks were resized to a uniform spatial resolution of 1024×1024 pixels to meet the model’s input constraints. The network was optimized for 100 epochs using the AdamW optimizer with a batch size of 2. The initial learning rate was set to $1 \times 1 0 ^ { - 4 }$ for the LoRA parameters, coupled with a mild weight decay of $1 \times 1 0 ^ { - 3 }$ to regularize the newly added adapters. The training objective was formulated as a linear combination of Dice loss and Binary Cross-Entropy (BCE) loss.

TABLE I  
SUMMARY OF DATASETS AND DOMAIN ADAPTATION SETTINGS.
<table><tr><td>Source</td><td>Target</td><td>Modality</td><td>Anatomy</td></tr><tr><td>MM-WHS 2017 MR</td><td>MM-WHS 2017 CT</td><td> $\mathbf { M } \mathbf { R }  \mathbf { C } \mathbf { T }$ </td><td>Heart</td></tr><tr><td>Promise 12</td><td>NCI-ISBI</td><td> $\mathrm { M R I } \ ( 1 . 5 \mathrm { T } / 3 \mathrm { T } )$ </td><td>Prostate</td></tr><tr><td>FLARE 22</td><td>CHAOS</td><td> $\mathbf { C T } \to \mathbf { M R I }$ </td><td>Abdomen</td></tr></table>

Supervised fine-tuning was performed exclusively on the source domain using identical data splits across all methods to ensure fair comparison. All experiments used the official MedSAM ViT-B/16 pre-trained weights with bounding box prompts derived from the masks (expanded by a random margin of 0-50 pixels to simulate human interactions), following standard evaluation protocols.

## C. Experimental Results

Comparison with Existing Adapters: We quantitatively compared FAN-LoRA with mainstream parameter-efficient fine-tuning methods on the four target domain datasets. These methods include LoRA [14], its variants such as DeLoRA [25] and SHiRA [26], nonlinear adapter AuroRA [20], and the frequency-based method FourierFT [22] and FreqFiT [28], covering a variety of design paradigms, including linear, nonlinear, and frequency-domain adaptive. The evaluation metrics primarily include the Dice Similarity Coefficient (DSC) and the 95% Hausdorff Distance (HD95). The quantitative results are presented in Tables II, III, and IV. FAN-LoRA achieves the best performance among all compared PEFT methods on the CHAOS, NCI-ISBI, and MM-WHS target domains, delivering consistent DSC gains (e.g., an absolute +1.0% over the strongest spectral baseline on MM-WHS) and achieving the lowest HD95 across all examined organs.

TABLE II  
QUANTITATIVE COMPARISON OF DIFFERENT PEFT METHODS ON THE CHAOS TARGET DOMAIN (MRI)
<table><tr><td>Method</td><td colspan="6">CHAOS</td></tr><tr><td></td><td>Liver</td><td>R.kidney</td><td>L.kidney</td><td>Spleen</td><td>Avg.</td><td>HD95↓</td></tr><tr><td>Zero-shot</td><td>90.9</td><td>90.5</td><td>89.8</td><td>93.6</td><td>91.2</td><td>5.03</td></tr><tr><td>Fine-tuning</td><td>72.6</td><td>87.5</td><td>84.4</td><td>88</td><td>83.1</td><td>11.21</td></tr><tr><td>LoRA</td><td>94.3</td><td>91.6</td><td>91.5</td><td>94.6</td><td>93</td><td>2.84</td></tr><tr><td>FourierFT</td><td>94.7</td><td>92.5</td><td>91.4</td><td>94.5</td><td>93.3</td><td>2.71</td></tr><tr><td>DeLoRA</td><td>94.4</td><td>92.9</td><td>91.6</td><td>94.2</td><td>93.3</td><td>2.66</td></tr><tr><td>SHiRA</td><td>94.5</td><td>92.3</td><td>91</td><td>94.3</td><td>93</td><td>2.95</td></tr><tr><td>FreqFiT</td><td>94.8</td><td>92.6</td><td>90.2</td><td>93.9</td><td>92.9</td><td>3.28</td></tr><tr><td>AuroRA</td><td>94.1</td><td>93</td><td>92.1</td><td>94.4</td><td>93.4</td><td>3.04</td></tr><tr><td>FAN-LoRA (Ours)</td><td>94.5</td><td>93.4</td><td>92.7</td><td>94.8</td><td>93.9</td><td>2.32</td></tr></table>

As shown in Table II, compared with AuroRA, which relies solely on non-linear mapping, FAN-LoRA achieves better boundary accuracy, indicating that non-linearity alone is insufficient to recover high-frequency details. Similarly, although FourierFT improves over LoRA by introducing spectral modeling, its lack of structural constraints leads to suboptimal global consistency. The superior performance of FAN-LoRA suggests that explicitly decoupling low-frequency structural alignment and high-frequency detail compensation is crucial for crossmodality adaptation from CT to MRI, where both intensity distribution and boundary characteristics differ consistently.

Note that the zero-shot and full fine-tuning baselines exhibit notably lower Dice on CHAOS compared to LoRA; this is attributed to the severe CT→MRI intensity gap causing overfitting in full-parameter tuning, which our frequencydecoupled design effectively mitigates.

TABLE III  
QUANTITATIVE COMPARISON OF DIFFERENT PEFT METHODS ON THE NCI-ISBI TARGET DOMAIN ACROSS 1.5T AND 3T SCANNERS
<table><tr><td rowspan="2">Method</td><td colspan="4">NCI-ISBI 1.5T</td><td colspan="4">NCI-ISBI 3T</td></tr><tr><td>PZ</td><td>CG</td><td>Avg.</td><td>HD95↓</td><td>PZ</td><td>CG</td><td>Avg.</td><td>HD95↓</td></tr><tr><td>Zero-shot</td><td>67.2</td><td>87.7</td><td>77.5</td><td>37.68</td><td>66.4</td><td>89.9</td><td>78.2</td><td>31.52</td></tr><tr><td>Fine-tuning</td><td>65.8</td><td>90.2</td><td>78</td><td>34.3</td><td>58.9</td><td>91.7</td><td>75.3</td><td>33.69</td></tr><tr><td>LoRA</td><td>66.3</td><td>93.5</td><td>79.9</td><td>30.04</td><td>60.2</td><td>93.1</td><td>76.7</td><td>31.99</td></tr><tr><td>FourierFT</td><td>68.2</td><td>92.7</td><td>80.5</td><td>27.87</td><td>62.8</td><td>93.1</td><td>78</td><td>27.51</td></tr><tr><td>DeLoRA</td><td>65.7</td><td>93.2</td><td>79.5</td><td>30.4</td><td>65.5</td><td>92.8</td><td>79.2</td><td>27.12</td></tr><tr><td>SHiRA</td><td>67.4</td><td>91.9</td><td>79.7</td><td>28.91</td><td>59.1</td><td>92.9</td><td>76</td><td>32.93</td></tr><tr><td>FreqFiT</td><td>65.1</td><td>92.6</td><td>78.9</td><td>27.99</td><td>61.8</td><td>92</td><td>76.9</td><td>28.34</td></tr><tr><td>AuroRA</td><td>66.8</td><td>91.4</td><td>79.1</td><td>29.29</td><td>62.6</td><td>91.8</td><td>77.2</td><td>29.68</td></tr><tr><td>FAN-LoRA (Ours)</td><td>68.4</td><td>93.3</td><td>80.9</td><td>26.96</td><td>66.4</td><td>92.8</td><td>79.6</td><td>26.48</td></tr></table>

Table III evaluates robustness under cross-center variations, where domain shifts mainly manifest as intensity distribution changes and acquisition noise. FAN-LoRA consistently achieves the highest Dice scores and the lowest HD95 on both 1.5T and 3T settings. Notably, its improvement over LoRA is more pronounced in HD95 than in Dice, indicating that FAN-LoRA primarily enhances boundary precision rather than coarse region overlap. This aligns with our design motivation: the high-frequency branch explicitly targets boundary refinement, which is critical under scanner-induced noise. Compared with FourierFT, FAN-LoRA also exhibits better stability across different magnetic field strengths, suggesting that simply introducing frequency modeling is insufficient without the structural guidance of the low-pass branch. Similarly, FreqFiT, another frequency-oriented adapter, lags behind FAN-LoRA, particularly in boundary accuracy, further confirming that high-frequency optimization becomes unstable in the absence of explicit frequency decoupling.

In the challenging MM-WHS cross-modality setting, FAN-LoRA demonstrates a substantial performance gain over all baselines. Unlike CHAOS, this task involves more complex anatomical structures and stronger modality gaps, making it particularly sensitive to structural misalignment, as shown in Table IV. Methods such as SHiRA and AuroRA improve over LoRA but still struggle with consistent multi-organ alignment, as reflected by their lower average Dice scores. Among them, SHiRA achieves a competitive Dice of 88.2 yet exhibits a relatively high HD95 of 20.83. This suggests that while its sparse high-rank adapter updates are sufficient for coarse region coverage, they provide insufficient gradient density along organ boundaries, especially at the myocardial-blood pool interface, which leads to degraded boundary precision. FourierFT consistently boosts performance by capturing spectral information; however, its lack of explicit structural modeling limits further gains. FAN-LoRA achieves the best performance by jointly optimizing global structure and local details. The low-pass branch ensures stable anatomical alignment across modalities, while the high-pass branch refines organ boundaries. This synergy is especially beneficial for complex cardiac structures, where both global topology and local boundary precision are critical.

TABLE IV  
QUANTITATIVE COMPARISON OF DIFFERENT PEFT METHODS ON THE MM-WHS 2017 CT TARGET DOMAIN
<table><tr><td>Method</td><td colspan="6">MM-WHS 17 CT</td></tr><tr><td></td><td>LAC</td><td>LVC</td><td>MYO</td><td>AA</td><td>Avg.</td><td>HD95↓</td></tr><tr><td>Zero-shot</td><td>68.5</td><td>91.3</td><td>87.2</td><td>90.7</td><td>84.4</td><td>30.07</td></tr><tr><td>Fine-tuning</td><td>70.2</td><td>91.9</td><td>87.3</td><td>91.3</td><td>85.2</td><td>28.79</td></tr><tr><td>LoRA</td><td>74.1</td><td>91.8</td><td>88.5</td><td>91</td><td>86.4</td><td>27.84</td></tr><tr><td>FourierFT</td><td>86.3</td><td>91.9</td><td>89.3</td><td>91.4</td><td>89.7</td><td>12.97</td></tr><tr><td>DeLoRA</td><td>73.8</td><td>91.6</td><td>88.3</td><td>91.2</td><td>86.5</td><td>26.95</td></tr><tr><td>SHiRA</td><td>81.7</td><td>91.6</td><td>88.3</td><td>91.2</td><td>88.2</td><td>20.83</td></tr><tr><td>FreqFiT</td><td>85.3</td><td>92.4</td><td>87.8</td><td>89.6</td><td>88.8</td><td>19.44</td></tr><tr><td>AuroRA</td><td>79.9</td><td>91.8</td><td>85.7</td><td>90.7</td><td>87</td><td>20.78</td></tr><tr><td>FAN-LoRA (Ours)</td><td>86.4</td><td>93</td><td>90.4</td><td>92.8</td><td>90.7</td><td>12.47</td></tr></table>

Performance Retention on Source Domains: Tables V and VI demonstrate that FAN-LoRA not only adapts effectively to target domains but also preserves performance on source domains, avoiding catastrophic forgetting. Interestingly, FAN-LoRA even slightly improves performance over full finetuning in some cases (e.g., FLARE 22), suggesting that the frequency-decoupled design acts as an implicit regularizer. By restricting low-frequency and high-frequency updates into separate subspaces, the model avoids overfitting to domainspecific noise while preserving general anatomical priors. This property is particularly desirable in clinical applications, where maintaining robustness across multiple domains is often more critical than optimizing for a single dataset.

Parameter Efficiency Analysis: As shown in Table VII, LoRA-based methods introduce over 440K trainable parameters, while FourierFT is highly efficient but limited in structural modeling. AuroRA reduces the parameter count to 111K, and FAN-LoRA maintains a nearly identical footprint of approximately 111.5K parameters despite its additional lightweight Fourier branch. This makes FAN-LoRA roughly four times more parameter-efficient than standard LoRA (over 450K parameters) while delivering superior segmentation accuracy and boundary precision, demonstrating a clearly favorable efficiency–performance trade-off. In contrast, FreqFiT incurs an even larger parameter budget (592K) than LoRA, yet fails to match FAN-LoRA in either metric, further confirming that a larger parameter count without explicit frequency decoupling does not translate into better domain adaptation.

Training Dynamics and Loss Curve Analysis: As illustrated in Fig. 3, FAN-LoRA exhibits a smoother and more stable convergence compared to AuroRA and FourierFT. Notably, the validation loss of FAN-LoRA shows fewer oscillations, indicating reduced overfitting to high-frequency noise. This behavior can be attributed to the explicit frequency decoupling mechanism. In contrast, AuroRA operates in a unified non-linear space, where gradients from different frequency components may interfere, while FourierFT lacks sufficient constraints on structural consistency. By separating optimization into low- and high-frequency subspaces, FAN-LoRA effectively stabilizes gradient updates, leading to improved generalization.

![](images/8fae4ead6331bc4227ed7b2621407fdd6af99062514d5747c346073532568fee.jpg)  
Fig. 3. Training and validation loss curves on the FLARE 22 and Promise 12 datasets, initialized with SAM pre-trained weights. Four methods are compared: Fine-tuning, AuroRA, FourierFT, and FAN-LoRA. On both datasets, Fine-tune exhibits signs of overfitting. In contrast, compared with AuroRA and FourierFT, FAN-LoRA demonstrates a smoother convergence trajectory and exhibits less overfitting.

Qualitative Comparison: Fig. 4 presents representative visual comparisons across multiple target domains. Compared with existing PEFT methods, FAN-LoRA consistently produces segmentation maps with sharper boundaries and more anatomically consistent structures.

Specifically, linear adapters such as LoRA and DeLoRA tend to generate over-smoothed predictions, exhibiting limitations in consistently recovering fine boundary details under strong domain shifts. FourierFT partially improves highfrequency detail preservation but often introduces spurious noise or oscillatory artifacts due to the lack of structural constraints. AuroRA improves global smoothness via nonlinear mapping; however, it still exhibits blurred boundaries in challenging regions, indicating insufficient high-frequency compensation.

In contrast, FAN-LoRA achieves a superior balance between structural coherence and detail preservation. The lowpass ANL branch effectively aligns global anatomical structures, while the high-pass Fourier branch selectively enhances boundary sharpness and local textures. This frequencydecoupled design leads to more precise delineation of organ boundaries and better recovery of thin or low-contrast structures, validating the effectiveness of the proposed approach. Quantitative boundary precision is further validated by the lowest HD95 across all target domains (see Tables II, III and IV).

TABLE V  
PERFORMANCE RETENTION EVALUATION OF DIFFERENT PEFT METHODS ON THE FLARE 22 SOURCE DOMAIN (CT)
<table><tr><td>Method</td><td colspan="9"></td><td colspan="5"></td></tr><tr><td></td><td>Liver</td><td>R.kidney</td><td>Spleen</td><td>Pancreas</td><td>Aorta</td><td>IVC</td><td>RAG</td><td>LAG</td><td>Gallbladder</td><td>Esophagus</td><td>Stomach</td><td>L.kidney</td><td>Avg.</td><td>HD95↓</td></tr><tr><td>Zero-shot</td><td>96</td><td>95.8</td><td>96.2</td><td>83.2</td><td>94.3</td><td>92</td><td>56.2</td><td>67.2</td><td>88.8</td><td>78.3</td><td>90.7</td><td>68.9</td><td>83.97</td><td>11.02</td></tr><tr><td>Fine-tuning</td><td>96.2</td><td>96.4</td><td>97.4</td><td>83.5</td><td>95.9</td><td>93.4</td><td>77.2</td><td>77.3</td><td>90.3</td><td>88.7</td><td>92.9</td><td>80.3</td><td>89.13</td><td>4.88</td></tr><tr><td>LoRA</td><td>96.4</td><td>96.6</td><td>96.7</td><td>84.3</td><td>94.7</td><td>94.2</td><td>68.8</td><td>67.6</td><td>91</td><td>89.1</td><td>93.4</td><td>74</td><td>87.23</td><td>7.1</td></tr><tr><td>FourierFT</td><td>95.9</td><td>95.8</td><td>95.5</td><td>87.4</td><td>93.6</td><td>93.9</td><td>71.7</td><td>74.4</td><td>90.6</td><td>88.7</td><td>92.3</td><td>78.9</td><td>88.23</td><td>5.32</td></tr><tr><td>DeLoRA</td><td>96.3</td><td>96.4</td><td>96.4</td><td>86.4</td><td>94.7</td><td>94.4</td><td>75.2</td><td>76</td><td>90.9</td><td>88.5</td><td>92.9</td><td>81</td><td>89.09</td><td>4.34</td></tr><tr><td>SHiRA</td><td>95.8</td><td>96.1</td><td>96.6</td><td>84.3</td><td>94.3</td><td>93</td><td>64.9</td><td>70.8</td><td>89.9</td><td>86.7</td><td>91</td><td>77</td><td>86.7</td><td>6.54</td></tr><tr><td>FreqFiT</td><td>94.1</td><td>95.8</td><td>96.9</td><td>84.5</td><td>95.2</td><td>92.7</td><td>65.1</td><td>71.6</td><td>88.3</td><td>86.6</td><td>91.2</td><td>78.6</td><td>86.7</td><td>6.98</td></tr><tr><td>AuroRA</td><td>95.2</td><td>96.0</td><td>95.8</td><td>83.4</td><td>94.1</td><td>92.9</td><td>72.9</td><td>74.8</td><td>90.4</td><td>89.4</td><td>92.3</td><td>77.5</td><td>87.89</td><td>5.92</td></tr><tr><td>FAN-LoRA (Ours)</td><td>97.1</td><td>96.8</td><td>97.5</td><td>85.7</td><td>96.0</td><td>94.3</td><td>78.2</td><td>81.1</td><td>92.1</td><td>90.1</td><td>94.0</td><td>83.1</td><td>90.50</td><td>5.56</td></tr></table>

TABLE VI

PERFORMANCE RETENTION EVALUATION OF DIFFERENT PEFT METHODS ON THE PROMISE 12 AND MM-WHS 2017 MR SOURCE DOMAINS
<table><tr><td rowspan="2">Method</td><td colspan="2">Promise 12</td><td colspan="6">MM-WHS 17 MR</td></tr><tr><td>prostate</td><td>HD95 ↓</td><td>LAC</td><td>LVC</td><td>MYO</td><td>AA</td><td> $\operatorname { A v g } .$ </td><td>HD95 ↓</td></tr><tr><td>Zero-shot</td><td>92.3</td><td>4.58</td><td>52.5</td><td>80.8</td><td>82.9</td><td>73.3</td><td>72.38</td><td>42.15</td></tr><tr><td>Fine-tuning</td><td>94.2</td><td>3.12</td><td>84.5</td><td>83.2</td><td>88.7</td><td>85.2</td><td>85.4</td><td>20.03</td></tr><tr><td>LoRA</td><td>94</td><td>3.45</td><td>80.2</td><td>83.2</td><td>87.9</td><td>82.9</td><td>83.6</td><td>20.55</td></tr><tr><td>FourierFT</td><td>94.1</td><td>2.85</td><td>81.1</td><td>82.8</td><td>87.2</td><td>82.1</td><td>83.37</td><td>23.46</td></tr><tr><td>DeLoRA</td><td>94.1</td><td>3.11</td><td>85.2</td><td>82.8</td><td>88.4</td><td>79.3</td><td>83.9</td><td>21.28</td></tr><tr><td>SHiRA</td><td>93.9</td><td>3.07</td><td>80.6</td><td>82.9</td><td>87.6</td><td>78.7</td><td>82.5</td><td>22.64</td></tr><tr><td>FreqFiT</td><td>94.92</td><td>2.79</td><td>82.8</td><td>83.1</td><td>86.9</td><td>79.7</td><td>83.1</td><td>23.23</td></tr><tr><td>AuroRA</td><td>93.8</td><td>3.16</td><td>81.7</td><td>82.9</td><td>87.5</td><td>81</td><td>83.3</td><td>28.17</td></tr><tr><td>FAN-LoRA (Ours)</td><td>95</td><td>1.89</td><td>84.7</td><td>83.6</td><td>88.5</td><td>82.8</td><td>84.9</td><td>19.63</td></tr></table>

TABLE VII

COMPARISON OF TRAINABLE PARAMETERS FOR DIFFERENT PEFT ADAPTERS WHEN INTEGRATED INTO THE MEDSAM IMAGE ENCODER.
<table><tr><td>Method</td><td>Trainable Parameters</td></tr><tr><td>LoRA</td><td>454,656</td></tr><tr><td>FourierFT</td><td>24,000</td></tr><tr><td>DeLoRA</td><td>442,392</td></tr><tr><td>SHiRA</td><td>442,368</td></tr><tr><td>FreqFiT</td><td>592128</td></tr><tr><td>AuroRA</td><td>111,072</td></tr><tr><td>FAN-LoRA (Ours)</td><td>111,552</td></tr></table>

## D. Ablation Study

Ablation Study on Core Modules: Table VIII provides empirical evidence supporting our core hypothesis. The independent introduction of either the low-pass or high-pass branch yields only marginal improvements in average Dice; however, the high-pass branch alone already reduces HD95 from 30.51 to 30.19 on the 1.5T domain and from 28.41 to 27.13 on the 3T domain, demonstrating that explicit high-frequency modeling provides independent boundary-refinement benefits even in the absence of structural guidance, indicating that neither structural alignment nor detail compensation alone is sufficient. The full FAN-LoRA model synergizes both branches to achieve the highest performance. This confirms our hypothesis that global structural alignment (low-pass) and local detail compensation (high-pass) are highly complementary and indispensable for robust domain adaptation.

TABLE VIII  
ABLATION STUDY OF THE CORE LOW-PASS AND HIGH-PASS MODULES INFAN-LORA ON THE NCI-ISBI TARGET DOMAIN
<table><tr><td>Base LoRA</td><td>Low-pass</td><td>High-pass</td><td colspan="2">NCI-ISBI 1.5T Avg. HD95 ↓</td><td colspan="2">NCI-ISBI 3T Avg. HD95↓</td></tr><tr><td>√</td><td></td><td></td><td>78.6</td><td>30.51</td><td>78.1</td><td>28.41</td></tr><tr><td>√</td><td>√</td><td></td><td>79.1</td><td>29.08</td><td>78.6</td><td>28.19</td></tr><tr><td>√</td><td></td><td>√</td><td>79.5</td><td>30.19</td><td>78.7</td><td>27.13</td></tr><tr><td>√</td><td>√</td><td>√</td><td>80.9</td><td>26.96</td><td>79.6</td><td>26.48</td></tr></table>

TABLE IX

ABLATION STUDY ON HIGH-PASS BRANCH COMPONENTS ON THENCI-ISBI TARGET DOMAIN.
<table><tr><td> $U _ { \mathrm { h i g h } } , V _ { \mathrm { h i g h } }$ </td><td> $S _ { \mathrm { f r e q } }$ </td><td>NCI-ISBI 1.5T Avg.</td><td>HD95↓ Avg.</td><td>NCI-ISBI 3T</td><td>HD95↓</td></tr><tr><td>Random &amp; Frozen</td><td></td><td>75.9</td><td>36.09</td><td>77.3</td><td>30.63</td></tr><tr><td>Learnable</td><td></td><td>78.9</td><td>35.25</td><td>78.05</td><td>28.75</td></tr><tr><td>Fixed Sinusoidal</td><td></td><td>79.6</td><td>28.63</td><td>78.9</td><td>27.11</td></tr><tr><td>Random &amp; Frozen</td><td>√</td><td>77.3</td><td>37.24</td><td>78.3</td><td>29.36</td></tr><tr><td>Learnable</td><td>√</td><td>79.2</td><td>30.58</td><td>78.6</td><td>28.97</td></tr><tr><td>Fixed Sinusoidal</td><td>√</td><td>80.9</td><td>26.96</td><td>79.6</td><td>26.48</td></tr></table>

We further dissect the high-pass branch by ablating its two key components: the projection matrices $U _ { \mathrm { h i g h } } , V _ { \mathrm { h i g h } }$ and the frequency modulation scalar $S _ { \mathrm { f r e q } }$ . Table IX reports the results under different configurations. Without the modulation scalar, fixed sinusoidal projections substantially outperform randomly frozen and learnable matrices, reducing HD95 from over 35 to 28.63 on the 1.5T domain, which confirms that a structured frequency basis is essential for boundary refinement. The learnable scalar $S _ { \mathrm { f r e q } }$ degrades performance when paired with random projections, yet it brings a clear gain when combined with the sinusoidal basis, achieving the best HD95 of 26.96 on 1.5T and 26.48 on 3T. We therefore adopt fixed sinusoidal projections with a learnable $S _ { \mathrm { f r e q } }$ as the default high-pass design.

Hyperparameter Sensitivity: Table X details the sensitivity analysis regarding the intrinsic rank of the low-pass space and the number of sampled frequencies n in the high-pass branch. Our results reveal that an optimal balance is struck at a lower rank dimension combined with a concise frequency sampling, achieving peak Dice scores on both 1.5T and 3T targets. Increasing the rank or the frequency dimensions excessively leads to marginal performance degradation, likely

Label

![](images/8745f3dc36a83151dad249f04ee67effac36f78dbe1b0055430bafc3d0f70687.jpg)  
Fig. 4. Qualitative comparison of different domain adaptation methods. From left to right: input image, ground truth, FAN-LoRA (ours), AuroRA, DeLoRA FourierFT, LoRA, and SHiRA. FAN-LoRA delivers sharper boundaries and better-preserved fine structures under domain shift, while competing methods tend to over-smooth or produce noisy artifacts. Red boxes highlight representative errors, such as missed structures and false positives.

due to the over-parameterization of the low-rank subspace which inadvertently re-introduces high-frequency noise fitting. Furthermore, this phenomenon suggests that within the pretrained MedSAM feature space, the intrinsic rank of the global anatomical transformations between medical modalities is inherently low, making higher rank dimensions redundant and prone to overfitting.

Regarding the high-frequency sampling count n, the optimal value of n = 16 emerges from a trade-off between representation capacity and over-parameterization. The highfrequency projection matrices $\mathbf { U } _ { \mathrm { h i g h } }$ and $\mathbf { V } _ { \mathrm { h i g h } }$ are constructed via discrete sinusoidal sampling along integer feature dimensions. As the sampling density increases, the column vectors of these projection matrices become increasingly correlated, reducing the effective rank of the projected spectral subspace. Selecting $n \ll d _ { \mathrm { i n } }$ ensures a well-conditioned basis for highfrequency extraction. Empirically, n = 16 strikes an optimal balance: smaller values (e.g., n = 8) under-represent the high-frequency subspace, limiting texture compensation; larger values $( \mathbf { e } . \mathbf { g } . , n \geq 6 4 )$ introduce redundant spectral components that increase overfitting risk without measurable performance gains. We recommend n as a lightweight hyperparameter amenable to minimal grid search in the range [8, 64], with the understanding that cross-modality shifts may benefit from slightly larger n than cross-center shifts due to stronger highfrequency perturbations.

TABLE X  
HYPERPARAMETER SENSITIVITY ANALYSIS OF RANK DIMENSION AND FREQUENCY SAMPLING ON THE NCI-ISBI TARGET DOMAIN
<table><tr><td rowspan="2">Rank</td><td rowspan="2">Frequency</td><td colspan="4">NCI-ISBI 1.5T</td><td colspan="4">NCI-ISBI 3T</td></tr><tr><td>PZ</td><td>CG</td><td>Avg.</td><td>HD95↓</td><td>PZ</td><td>CG</td><td>Avg.</td><td>HD95↓</td></tr><tr><td>4</td><td></td><td>66.2</td><td>92.2</td><td>79.2</td><td>30.38</td><td>65.1</td><td>92.9</td><td>79</td><td>27.71</td></tr><tr><td>8</td><td>16</td><td>66.1</td><td>93.8</td><td>80</td><td>30.18</td><td>59.6</td><td>93.2</td><td>76.4</td><td>32.77</td></tr><tr><td>16</td><td></td><td>65.7</td><td>92.9</td><td>79.3</td><td>30.68</td><td>61.2</td><td>93.4</td><td>77.3</td><td>31.42</td></tr><tr><td>32</td><td></td><td>65.8</td><td>93.2</td><td>79.5</td><td>33.44</td><td>59</td><td>93.3</td><td>76.15</td><td>33.47</td></tr><tr><td></td><td>32</td><td>66.7</td><td>93.2</td><td>80</td><td>29.57</td><td>59.3</td><td>92.6</td><td>76</td><td>32.15</td></tr><tr><td>2</td><td>64</td><td>66.6</td><td>92.6</td><td>79.6</td><td>29.7</td><td>63.9</td><td>92.4</td><td>78.2</td><td>28.12</td></tr><tr><td></td><td>128</td><td>66.6</td><td>93.5</td><td>80.1</td><td>29.32</td><td>60.9</td><td>92.4</td><td>76.7</td><td>29.72</td></tr><tr><td></td><td>256</td><td>67.1</td><td>92.9</td><td>80</td><td>29.35</td><td>60.1</td><td>92.5</td><td>76.3</td><td>31.05</td></tr><tr><td>2</td><td>16</td><td>68.4</td><td>93.3</td><td>80.9</td><td>26.96</td><td>66.4</td><td>92.8</td><td>79.6</td><td>26.48</td></tr></table>

Error Analysis: Despite its competitive performance, FAN-LoRA may still struggle in extremely low-contrast regions where both structural and textural cues are weak. In such cases, the high-frequency branch may have limited signal to enhance, while the low-pass branch may over-smooth ambiguous boundaries.

## V. CONCLUSION

In this paper, we presented FAN-LoRA, a frequencydecoupled PEFT framework that explicitly separates low- and high-frequency optimization for medical foundation model adaptation. Extensive experiments demonstrate competitive advantages over existing PEFT baselines in both accuracy and boundary precision. While highly effective, we note that the performance gains may diminish in extremely lowcontrast regions where frequency cues are intrinsically weak. Future work will explore adaptive frequency-band selection and theoretically grounded gradient decomposition for broader multi-domain generalization.

## REFERENCES

[1] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.

[2] M. A. Mazurowski, H. Dong, H. Gu, J. Yang, N. Konz, and Y. Zhang, “Segment anything model for medical image analysis: an experimental study,” Medical Image Analysis, vol. 89, p. 102918, 2023.

[3] Y. Huang, X. Yang, L. Liu, H. Zhou, A. Chang, X. Zhou, R. Chen, J. Yu, J. Chen, C. Chen et al., “Segment anything model for medical images?” Medical Image Analysis, vol. 92, p. 103061, 2024.

[4] F. Bougourzi and A. Hadid, “Recent advances in medical imaging segmentation: A survey,” arXiv preprint arXiv:2505.09274, 2025.

[5] R. Wang, T. Lei, R. Cui, B. Zhang, H. Meng, and A. K. Nandi, “Medical image segmentation using deep learning: A survey,” IET image processing, vol. 16, no. 5, pp. 1243–1267, 2022.

[6] W. Ji, J. Li, Q. Bi, T. Liu, W. Li, and L. Cheng, “Segment anything is not always perfect: An investigation of sam on different real-world applications,” 2024.

[7] X. Yao, H. Liu, D. Hu, D. Lu, A. Lou, H. Li, R. Deng, G. Arenas, B. Oguz, N. Schwartz et al., “Fnpc-sam: uncertainty-guided false negative/positive control for sam on noisy medical images,” in Medical Imaging 2024: Image Processing, vol. 12926. SPIE, 2024, p. 1292602.

[8] J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, “Segment anything in medical images,” Nature Communications, vol. 15, p. 654, 2024.

[9] V. Lialin, V. Deshpande, X. Yao, and A. Rumshisky, “Scaling down to scale up: A guide to parameter-efficient fine-tuning,” arXiv preprint arXiv:2303.15647, 2023.

[10] S. Chen, C. Ge, Z. Tong, J. Wang, Y. Song, J. Wang, and P. Luo, “Adaptformer: Adapting vision transformers for scalable visual recognition,” Advances in Neural Information Processing Systems, vol. 35, pp. 16 664–16 678, 2022.

[11] K. Phuntsho, K. Lee, I. Lee, E. Ahn et al., “Adaptation of foundation models for medical image analysis: Strategies, challenges, and future directions,” arXiv preprint arXiv:2511.01284, 2025.

[12] J. Silva-Rodr´ıguez, J. Dolz, and I. B. Ayed, “Towards foundation models and few-shot parameter-efficient fine-tuning for volumetric organ segmentation,” Medical Image Analysis, vol. 103, p. 103596, 2025.

[13] X. Jiang, C. Yang, L. Zhang, T. K.-T. Cheng, and X. Yang, “Sr-sam: Subspace regularization for domain generalization of segment anything model,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2025, pp. 510–520.

[14] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” Iclr, vol. 1, no. 2, p. 3, 2022.

[15] K. Zhang and D. Liu, “Customized segment anything model for medical image segmentation,” arXiv preprint arXiv:2304.13785, 2023.

[16] J. Wu, Z. Wang, M. Hong, W. Ji, H. Fu, Y. Xu, M. Xu, and Y. Jin, “Medical sam adapter: Adapting segment anything model for medical image segmentation,” Medical image analysis, vol. 102, p. 103547, 2025.

[17] Q. Liu, C. Chen, J. Qin, Q. Dou, and P.-A. Heng, “Feddg: Federated domain generalization on medical image segmentation via episodic learning in continuous frequency space,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 1013– 1023.

[18] Y. Yang and S. Soatto, “Fda: Fourier domain adaptation for semantic segmentation,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 4085–4095.

[19] Z. Li, L. Li, and J. Zhang, “Adaptive frequency domain alignment network for medical image segmentation,” arXiv preprint arXiv:2512.16393, 2025.

[20] H. Dong, W. Zhu, G. Song, and L. Wang, “Aurora: Breaking lowrank bottleneck of lora with nonlinear mapping,” arXiv preprint arXiv:2505.18738, 2025.

[21] X. Cai, R. Pan, and H. Yang, “Loki: Low-dimensional kan for efficient fine-tuning image models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 14 869–14 880.

[22] Z. Gao, Q. Wang, A. Chen, Z. Liu, B. Wu, L. Chen, and J. Li, “Parameter-efficient fine-tuning with discrete fourier transform,” in International Conference on Machine Learning. PMLR, 2024, pp. 14 884– 14 901.

[23] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnu-net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature methods, vol. 18, no. 2, pp. 203–211, 2021.

[24] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for nlp,” in International conference on machine learning. PMLR, 2019, pp. 2790–2799.

[25] M. Bini, L. Girrbach, and Z. Akata, “Delora: Decoupling angles and strength in low-rank adaptation,” arXiv preprint arXiv:2503.18225, 2025.

[26] K. Bhardwaj, N. P. Pandey, S. Priyadarshi, V. Ganapathy, R. Esteves, S. Kadambi, S. Borse, P. Whatmough, R. Garrepalli, M. Van Baalen et al., “Rapid switching and multi-adapter fusion via sparse high rank adapters,” arXiv preprint arXiv:2407.16712, 2024.

[27] A. He, Y. Wu, Z. Wang, T. Li, and H. Fu, “Adaptfrcnet: Semi-supervised adaptation of pre-trained model with frequency and region consistency for medical image segmentation,” Medical Image Analysis, vol. 103, p. 103626, 2025.

[28] S. T. Ly and H. V. Nguyen, “Frequency strikes back: Boosting parameterefficient foundation model adaptation for medical imaging,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2025, pp. 258–267.

[29] E. Cordero, M. de Gosson, M. Dorfler, and F. Nicola, “Generalized¨ born-jordan distributions and applications,” Advances in Computational Mathematics, vol. 46, no. 4, p. 51, 2020.

[30] M. Unser, A. Aldroubi, and M. Eden, “A family of polynomial spline wavelet transforms,” Signal processing, vol. 30, no. 2, pp. 141–162, 1993.

[31] R. Panda and B. Chatterji, “Least squares generalized b-spline signal and image processing,” Signal Processing, vol. 81, no. 10, pp. 2005–2017, 2001.

[32] X. Zhuang, L. Li, C. Payer, D. Stern, M. Urschler, M. P. Heinrich,<sup>ˇ</sup> J. Oster, C. Wang, O. Smedby, C. Bian <sup>¨</sup> et al., “Evaluation of algorithms for multi-modality whole heart segmentation: an open-access grand challenge,” Medical image analysis, vol. 58, p. 101537, 2019.

[33] G. Litjens, R. Toth, W. Van De Ven, C. Hoeks, S. Kerkstra, B. Van Ginneken, G. Vincent, G. Guillard, N. Birbeck, J. Zhang et al., “Evaluation of prostate segmentation algorithms for mri: the promise12 challenge,” Medical image analysis, vol. 18, no. 2, pp. 359–373, 2014.

[34] N. Bloch, A. Madabhushi, H. Huisman, J. Freymann, J. Kirby, M. Grauer, A. Enquobahrie, C. Jaffe, L. Clarke, and K. Farahani, “NCI-ISBI 2013 challenge: Automated segmentation of prostate structures,” 2015.

[35] J. Ma and B. Wang, Fast and Low-Resource Semi-supervised Abdominal Organ Segmentation: MICCAI 2022 Challenge, FLARE 2022, Held in Conjunction with MICCAI 2022, Singapore, September 22, 2022, Proceedings. Springer Nature, 2023.

[36] A. E. Kavur, N. S. Gezer, M. Barıs¸, S. Aslan, P.-H. Conze, V. Groza, D. D. Pham, S. Chatterjee, P. Ernst, S. Ozkan<sup>¨</sup> et al., “Chaos challengecombined (ct-mr) healthy abdominal organ segmentation,” Medical image analysis, vol. 69, p. 101950, 2021.