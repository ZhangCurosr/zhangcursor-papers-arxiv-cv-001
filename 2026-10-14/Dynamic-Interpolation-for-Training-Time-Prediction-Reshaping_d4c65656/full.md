# Dynamic Interpolation for Training-Time Prediction Reshaping in

Low-Light Image Enhancement

Peirui Zhu, Shijie Hao, Jingxi Liao, Richang Hong, Senior Member, IEEE, and Meng Wang, Fellow, IEEE

Abstract—Low-light image enhancement (LLIE) is an important image/video processing problem for reliable visual systems under poor illumination, yet it remains challenging due to the strong coupling among insufficient illumination, noise amplification, and structural degradation. Existing methods typically improve enhancement performance through either increasingly sophisticated network architectures or carefully engineered objective functions. While effective, the former often increases model complexity and deployment burden, whereas the latter usually relies on empirically tuned loss weights. In this paper, we revisit supervised LLIE from a training-dynamics perspective and observe that brightness-related errors tend to converge early, while residual noise and structural artifacts often remain insufficiently optimized in later stages. To address this issue, we propose a Dynamic Interpolation Training-only Module (DITM), a lightweight training-only module that introduces no additional learnable parameters and adaptively reshapes the model prediction for supervised feed-forward LLIE training. Specifically, DITM employs a brightness-aware interpolation scheme, in which the model prediction is combined with a mean brightness-aligned image constructed by rescaling the model prediction according to the mean brightness of the ground-truth image. By reducing the relative dominance of global brightness discrepancy, DITM makes residual local errors more influential during backpropagation after global luminance has largely stabilized, thereby facilitating artifact suppression and fine-detail restoration in later training stages. DITM is used only during training and is fully removed at inference, introducing no additional inference cost. Extensive experiments on multiple benchmark datasets and CNN-, Transformer-, and Mamba-based LLIE backbones show that DITM consistently improves fidelity and perceptual quality while preserving the inference-time architecture and introducing no additional inference-time computation.

Index Terms—Low-Light Image Enhancement, Dynamic Interpolation, Training-only Module, Lightweight, Zero Inference Overhead.

## I. INTRODUCTION

L <sup>OW-LIGHT</sup> <sup>image</sup> <sup>enhancement</sup> <sup>(LLIE)</sup> <sup>is</sup> <sup>a</sup> <sup>fundamental</sup> image/video processing task for robust visual perception in poorly illuminated environments [1]. Unlike singledegradation problems such as denoising or super-resolution, LLIE must handle complex, coupled degradations [2], [3]. In low-light environments, images typically exhibit both low dynamic range and elevated noise levels due to the low signal-to-noise ratio, which makes enhancement particularly challenging [4]. This commonly leads to an “amplificationsuppression dilemma”: increasing illumination to restore visibility amplifies latent noise, whereas aggressive denoising risks erasing high-frequency structural details [5], [6]. The main challenge lies in balancing global luminance restoration, local noise suppression, and structural fidelity under practical visual-system constraints. Existing LLIE methods often improve enhancement performance through network architecture design or specialized objective design. Network design typically focuses on enhancing the model’s ability to handle complex degradations and capture global contextual relationships [7]–[9]. While effective, these methods may introduce additional computational overhead and inference latency, which is undesirable for resource-constrained visual processing systems. Another direction employs specialized loss terms to balance brightness enhancement, noise suppression, and detail preservation [10]–[13]. Although effective, these methods remain highly sensitive to loss weight configuration, requiring extensive empirical tuning.

Although recent advances in network architectures and objective functions have substantially improved low-light image enhancement, we observe a recurring asynchronous convergence pattern during training. As shown in Fig. 1, the global brightness discrepancy is reduced mainly in the early stage, whereas residual noise, blurred structures, and local artifacts may persist until considerably later stages. To quantitatively characterize this optimization behavior, we track the evolution of the global brightness discrepancy throughout training across four representative LLIE models on LOLv1 and LOLv2- synthetic using two brightness representations. We evaluate this evolution using BMAE and the early reduction ratio η<sub>c</sub>(r). Across the evaluated settings, 72.3%–96.2% of the total BMAE reduction is achieved within the first 20% of training (Fig. 1(a)), while noticeable local restoration errors may persist in late-stage predictions (Fig. 1(b)). These observations indicate an asynchronous optimization pattern: the image-level brightness discrepancy is largely reduced early, whereas local artifact suppression and structural-detail restoration continue to require further optimization. To address this issue, we propose reshaping training-time predictions to promote further optimization of the remaining restoration errors.

We propose the Dynamic Interpolation Training-only Module (DITM), a lightweight training-only module that introduces no additional learnable parameters for supervised feed-forward LLIE training. Guided by an adaptive weight (interpolation factor), DITM generates a dynamically interpolated prediction by fusing the model prediction with a mean brightnessaligned image, which is constructed by rescaling the model prediction according to the mean brightness of the groundtruth image. This factor progressively adjusts the training signal by reducing the relative dominance of global luminance discrepancy in the reconstruction loss. As a result, after global brightness has largely stabilized, residual local errors receive relatively stronger influence during backpropagation, which helps the model further suppress artifacts and restore details. Importantly, DITM is applied solely during training and fully removed at inference, incurring zero additional inference overhead. Instead of redesigning the enhancement backbone or increasing inference-time complexity, DITM improves supervised feed-forward LLIE models by reshaping the model prediction during training in a lightweight, plug-and-play, and architecture-agnostic manner. This design is particularly suitable for efficient visual enhancement systems, since it improves the training behavior of existing LLIE backbones without modifying their deployed inference architectures. As a result, the enhanced models retain the same number of inference parameters, operations, and latency as the original baselines, which is desirable for practical low-light image/video processing applications. Of note, instead of redesigning the enhancement backbone or increasing inference-time complexity, DITM introduces a training-only interpolation process between the raw model prediction and its brightness-aligned image. The same reconstruction loss function is retained, but it is evaluated on the interpolated prediction rather than directly on the raw prediction. Therefore, DITM does not explicitly multiply samples, loss terms, or gradients by adaptive weights, nor does it introduce auxiliary prediction heads, intermediatelayer supervision objectives, or a sample-difficulty schedule. Its key distinction lies in changing the prediction involved in supervision through brightness-aware interpolation, rather than changing the loss formulation or the training schedule. Since DITM is removed after training, the enhanced models preserve the same deployed architecture, inference parameters, operations, and latency as the original baselines, making it suitable for efficient low-light image/video enhancement systems.

(b)  
![](images/647efa2e5b475688ac2e5e33c38771011ae93850f10857ddde61686edd195402.jpg)  
Fig. 1. Empirical motivation for DITM. (a) Global brightness discrepancy is reduced primarily during the early stage of baseline training. We present the evolution of the global brightness discrepancy throughout the training of four representative baseline LLIE models on LOLv1 and $\mathrm { L O L v } 2 \mathrm { \cdot }$ -synthetic, evaluated using the Y channel of YCbCr and the V channel of HSV. Let A denote the baseline prediction at training stage t, and let $Y$ denote the corresponding ground-truth image. For brightness representation c, BMAE is defined as $\mathrm { B M A E } _ { c } ( t ) \bar { \mathbf { \Psi } } = \mathbf { \Psi } | \mathrm { M e a n } [ \Phi _ { c } ( Y ) ] \bar { \mathbf { \Psi } } - \mathbf { M } \bar { \mathbf { e } } \mathbf { a } \mathbf { n } [ \Phi _ { c } ( A _ { t } ) ] |$ |, where $\Phi _ { c } ( \cdot )$ extracts the specified brightness component, Mean[·] computes its spatial mean over all pixels, and $\dot { c } \in \lbrace \mathrm { Y C b } \dot { \mathrm { C r } } \mathrm { - Y } , \mathrm { H } \dot { \mathrm { S } } \mathrm { V } \mathrm { - } \mathrm { V } \rbrace$ . At each training iteration, BMAE is computed by averaging the brightness discrepancy over the samples used for the current update. The early brightness-error reduction ratio is defined as $\begin{array} { r } { \eta _ { c } ( r ) = \frac { \hat { \mathrm { B } } \mathrm { M A E } _ { \mathrm { i n i t } } ^ { c ^ { * } } - \mathrm { B M } \mathrm { \bar { A } E } _ { r } ^ { \frac { * } { c } } } { \mathrm { B M A E } _ { \mathrm { i n i t } } ^ { c } - \mathrm { B M A E } _ { \mathrm { f i n a l } } ^ { c } } \times } \end{array}$ 100%, where the three BMAE terms denote the averaged BMAE values around the initial stage, training progress r, and the final stage, respectively. The annotated $\eta _ { c } ( 2 0 \% )$ values show that 72.3%–96.2% of the overall BMAE reduction is already achieved within the first 20% of training. (b) The baseline models still struggle to recover local structures and fine details as training approaches its final stage. The middle-stage baseline predictions contain noticeable residual noise, blurred structures, and local artifacts, some of which persist into the late stage. Motivated by these observations, we propose reshaping training-time predictions so that, after global brightness has largely stabilized, residual local restoration errors become relatively more influential during subsequent optimization, thereby facilitating late-stage artifact suppression and structural-detail restoration. (c) Visual examples of our DITM-based models. With DITM, the same backbones produce cleaner local structures and textures with fewer residual artifacts. In this figure, red and blue boxes indicate enlarged regions. Best viewed enlarged.

Our main contributions are summarized as follows:

• We identify a training-stage optimization imbalance in supervised LLIE, where global brightness discrepancy tends to stabilize earlier than residual artifact suppression and fine-detail restoration.

• We propose DITM, a lightweight training-only module for supervised feed-forward LLIE training that adaptively reshapes the model prediction through brightness-aware interpolation, without modifying the backbone architecture or introducing additional learnable parameters.

• DITM is entirely removed at inference and improves fidelity, perceptual quality, and no-reference assessment scores across CNN-, Transformer-, and Mamba-based supervised feed-forward LLIE backbones, with zero ad-

## II. RELATED WORK

## A. Learning-based Low-Light Image Enhancement

Existing learning-based LLIE methods mainly improve enhancement performance from two directions: network design and loss function design. The former focuses on strengthening the restoration capacity of the model itself, while the latter improves optimization through carefully designed training constraints.

From the perspective of model architecture, one major direction is to improve representation capacity. Early methods mainly relied on CNNs to address low-light degradation [14]– [16], but their locality-driven design limits the modeling of long-range dependencies and global contextual relationships. To enhance long-range dependency capture, transformer-based architectures introduced self-attention for stronger non-local interaction and contextual representation [9], [17], [18]. To reduce the computational burden of self-attention while preserving global dependency modeling, state-space models (SSMs) have recently been introduced as a more efficient alternative with linear complexity [19]–[22]. In parallel, diffusion-based models have also been explored for LLIE, leveraging iterative denoising to improve restoration quality under severe low-light degradation [23]–[27]. Related diffusion-based restoration studies have also investigated pretrained text-to-image models for generic image restoration [28]. Beyond architectural advancement, another important direction is explicit degradation modeling. Unlike generic restoration networks, these methods explicitly account for the complex degradation patterns in low-light images, such as noise amplification and structural corruption. By incorporating designs tailored to degradation characteristics, these approaches aim to enable more robust restoration under challenging illumination conditions [7], [8], [29]–[32]. Another important direction explores frequency representations for LLIE. By complementing spatial features with frequency information, these methods better preserve edges, textures, and structural consistency during restoration [33]–[36]. Despite their effectiveness, these architecture-driven approaches may increase computational cost or inference latency, which motivates training-side enhancement strategies that preserve the deployed inference pipeline.

Loss function design provides an efficient way to improve enhancement performance during training. For LLIE, the loss functions typically combine multiple complementary constraints to encourage structure preservation and perceptual fidelity, thereby implicitly coordinating multiple degradation factors during restoration [10], [12], [37]–[40]. While these methods can improve optimization stability, the relative influence of different objectives is usually statically predefined and often requires extensive empirical tuning, which limits adaptability to dynamically varying degradation characteristics during training.

Beyond low-light image enhancement, low-light video enhancement has also attracted increasing attention due to its importance in real-world dynamic scenes [2]. Different from image-based methods that mainly focus on spatial restoration within individual frames, video enhancement further requires temporal consistency across consecutive frames to suppress motion-induced artifacts [41], [42]. More recently, this research direction has been extended into event-guided lowlight video enhancement [43], where event cameras provide complementary high-dynamic-range temporal cues for robust enhancement in challenging low-light scenarios.

## B. Training-Side Strategies for Efficient Vision Models

Training-side strategies have become a practical direction for efficient vision models, since they can improve representation learning, optimization behavior, or deployment efficiency without fully redesigning the backbone network. In the broader context of efficient vision model design, one related line of studies introduces lightweight structural modules into existing architectures. For example, SENet recalibrates channel-wise feature responses through a compact squeeze-and-excitation operation [44], while CBAM and ECA further exploit spatialchannel attention or efficient local cross-channel interaction to strengthen feature representation with limited extra complexity [45], [46]. Similar modular designs have been widely used in recognition, detection, segmentation, and restoration tasks to enhance multi-scale perception, spatial interaction, or taskspecific feature adaptation [47]–[50]. Although these structural modules are effective, many of them still become part of the deployed inference model, which is different from trainingonly strategies that aim to improve learning while preserving the original inference pipeline.

Beyond structural plug-ins, training-time auxiliary modules have been increasingly adopted to improve learning dynamics without adding inference cost. For example, YOLOv7 [51] introduces auxiliary supervision branches and optimization regularizers that are active only during training and removed at inference. Similar training-time strategies have also been explored in video understanding tasks such as action recognition, where temporary training branches enhance motion representation while preserving single-stream inference [52]. In the LLIE domain, some methods also employ trainingtime auxiliary modules to accelerate convergence and enhance feature learning, while introducing no additional computational overhead during inference. For example, FDANet [53] introduces a raw supervision branch during training to guide feature representation, which is discarded at test time. These methods demonstrate that auxiliary pathways can provide optimization guidance while maintaining a lightweight inference pipeline, achieving a balance between improved learning and computational efficiency.

DITM is also related to training-time optimization strategies such as auxiliary supervision, curriculum learning, selfpaced learning, and adaptive loss reweighting. Deeply supervised networks introduce companion objectives at intermediate layers to improve gradient propagation and feature learning [54], while curriculum learning adjusts the training order or difficulty of samples to guide optimization [55]. In LLIE, Luo et al. [56] introduced self-paced learning into unsupervised enhancement by progressively selecting reliable samples for optimization, and Kim et al. [57] designed an intrinsic decomposition-based curriculum to guide LLIE training from easier to harder degradation patterns. Adaptive reweighting methods, such as focal loss and GradNorm, modify the relative contribution of samples, tasks, or gradients during training [58], [59]. In image reconstruction, focal frequency loss further encourages models to focus on hard-to-reconstruct frequency components through dynamic frequency-domain weighting [60]. Different from these methods, DITM does not introduce auxiliary objectives, alter the sample curriculum, select training samples, or explicitly assign weights to loss terms. Instead, it reshapes the supervised prediction itself through brightness-aware interpolation before loss computation, thereby changing the effective training signal while keeping the original loss form and inference model unchanged.

Overall, existing training-side strategies improve optimization mainly by adding auxiliary branches or objectives, scheduling the learning difficulty, selecting samples progressively, or dynamically reweighting samples, tasks, gradients, or loss components. In contrast, DITM follows a different route: it reshapes the supervised prediction without introducing learnable parameters by interpolating the model prediction with a mean brightness-aligned counterpart. This design is specifically motivated by the training dynamics of supervised LLIE, where global brightness discrepancy often stabilizes earlier than residual noise and structural artifacts. Therefore, DITM is not a general loss-weighting or curriculum strategy, but a task-oriented prediction reshaping mechanism that weakens the dominance of global luminance mismatch and makes residual local degradation more influential during later optimization. Since DITM introduces no auxiliary branch, trainable parameters, auxiliary prediction, sample-selection schedule, or inference-time computation, it keeps the deployed enhancement model unchanged.

## III. METHOD

This section presents the proposed Dynamic Interpolation Training-only Module (DITM) for efficient supervised LLIE training. Section III-A first gives an overview of the overall training framework and illustrates how DITM is integrated into the baseline network. Section III-B introduces the construction of the mean brightness-aligned image and the dynamically interpolated prediction. Section III-C introduces the design of the adaptive interpolation factor s under a Weighted Least Squares (WLS) perspective. Section III-D offers a qualitative analysis of the training response induced by DITM. Section III-E further discusses training–inference consistency. Section III-F discusses the computational overhead introduced by DITM.

## A. Overview

To improve supervised LLIE models without increasing inference complexity, we propose a plug-and-play trainingonly module, termed the Dynamic Interpolation Trainingonly Module (DITM). As illustrated in Fig. 2(a), the overall training framework consists of two components: the baseline enhancement network $f _ { \theta } ( \cdot )$ and DITM. Instead of modifying the internal architecture of the backbone, DITM operates on the model prediction A and regulates the optimization process during training. Owing to its architecture-agnostic design, DITM can be integrated into diverse supervised LLIE backbones while leaving the inference architecture unchanged.

Given a low-light input image X, the enhancement network $f _ { \theta } ( \cdot )$ produces a prediction $A = f _ { \theta } ( X )$ . Instead of directly using A for loss computation, DITM leverages the global mean brightness of the ground-truth image Y and transforms A into a dynamically interpolated prediction C via:

$$
C = { \mathrm { D I T M } } ( A , Y ) .\tag{1}
$$

In essence, $C$ adaptively reshapes the model prediction A. The internal operation of DITM is illustrated in Fig. 2(c). C serves as a dynamically interpolated prediction that encourages the optimization process to better balance global luminance restoration, local artifact suppression, and detail restoration during training. The network parameters θ are then optimized by minimizing the loss between the interpolated prediction C and the ground-truth image Y:

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \mathcal { L } ( C , Y ) .\tag{2}
$$

It is worth emphasizing that DITM is used only during training and is removed during inference. As illustrated in Fig. 2(b), the baseline enhancement network directly produces the final enhanced image $A ^ { * } = f _ { \theta ^ { * } } ( X )$ . Although supervision is applied to the interpolated prediction C, DITM introduces no learnable parameters, and all gradients propagate through A to update only the baseline enhancement network $f _ { \theta } ( X )$ . Consequently, the trained baseline enhancement network $f _ { \theta ^ { \ast } } ( X )$ achieves improved restoration performance during inference without introducing additional parameters, operations, or latency at inference.

## B. Dynamic Construction of Interpolated Prediction

Mean Brightness-Aligned Prediction. Using the meanbrightness ratio between $Y$ and $A ,$ , we construct B by linearly rescaling A:

$$
B = \beta A , \quad { \mathrm { w h e r e ~ } } \beta = { \frac { \mu _ { Y } } { \mu _ { A } } } = { \frac { \mathbf { M e a n } ( \operatorname { G r a y } ( Y ) ) } { \mathbf { M e a n } ( \operatorname { G r a y } ( A ) ) } } .\tag{3}
$$

Here, $\mu _ { Y }$ and $\mu _ { A }$ denote the global mean brightness values of the ground-truth image $Y$ and the model prediction A, respectively. They are obtained by spatially averaging the grayscale images converted from RGB using the weights (0.2989, 0.5870, 0.1140). Since B is constructed by uniformly scaling all pixels of A with the same positive factor $\beta ,$ this operation changes only its global intensity scale without altering pixel locations or spatial relationships. Consequently, B matches the global mean brightness of $Y$ while retaining the structural layout, edges, and local content inherited from A. It should be noted that this operation provides global meanbrightness alignment rather than explicit spatial illumination modeling.

Dynamic Prediction Interpolation. With the mean brightness-aligned image B established, the dynamically interpolated prediction $C$ is constructed through adaptive interpolation between A and $B \colon$

$$
C = s A + ( 1 - s ) B , \quad s \in ( 0 , 1 ) ,\tag{4}
$$

![](images/eeda007712ebe0ead726947c0c1db710e5fdb8d511567daa22342f6bc14604fa.jpg)

![](images/966645e985f9721201e50815e2a8e6866e42276bf5c92501a75d574d93c0e8ab.jpg)  
Fig. 2. Training and inference pipeline of the proposed DITM. (a) Training Stage: DITM adjusts the model prediction A of the baseline model $f _ { \theta } ( \cdot )$ by leveraging the global mean brightness values of the ground-truth image Y, generating an interpolated prediction C for loss computation. (b) Inference Stage: DITM is discarded, and the trained baseline $f _ { \theta ^ { * } } \left( \cdot \right)$ produces the final enhanced image $A ^ { * }$ with no additional inference-time computation. (c) DITM: This module adaptively computes an interpolation factor s based on the brightness discrepancy between A and Y to yield the adjusted prediction C.

Here, s is a sample-wise adaptive interpolation factor determined by the current brightness discrepancy, which dynamically controls the balance between A and B. In our implementation, s is detached before being used to construct C. Therefore, its value varies across samples and training iterations but is treated as a constant during backpropagation to avoid introducing an additional optimization path through the factor itself.

Instead of directly supervising the model prediction A, the proposed module applies supervision to the interpolated prediction C. During training, network parameters are updated by minimizing the discrepancy between C and the groundtruth image Y. This interpolation introduces brightness-aware prediction reshaping into the supervised training objective. Instead of directly imposing local structural constraints, DITM weakens the dominance of global brightness mismatch in the reconstruction loss. Consequently, when global luminance has been largely corrected, residual local errors, such as noiserelated deviations and structural artifacts, become relatively more influential during backpropagation. The effectiveness of DITM therefore depends on how this interpolation strength is adaptively controlled during training. Because different training stages and image states may require different interpolation strengths, the factor s is designed to provide persample adjustment and maintain a suitable balance throughout optimization.

## C. Adaptive Interpolation Factor Design

As a key component of DITM, the adaptive interpolation factor s controls the balance between A and B during training.

To achieve effective adjustment of $s ,$ its design follows two core principles:

• Brightness-Discrepancy Awareness: s should reflect the brightness discrepancy between the model prediction A and the ground-truth image Y , so as to provide an adaptive signal that guides the balance between A and B during training.

• Smooth Optimization-Focus Shift: s should decay smoothly as the brightness discrepancy decreases, enabling the optimization to gradually shift focus toward refining image degradations beyond global luminance.

Weighted Optimization Framework. Adaptive interpolation essentially aims to strike a reasonable balance between A and B. To provide an analytical interpretation of this balance, we describe the interpolation process from a Weighted Least Squares perspective. In this formulation, the interpolated prediction C is treated as the weighted centroid between A and B, obtained by minimizing the following weighted interpolation objective:

$$
\mathcal { I } _ { C } ( C ; w _ { A } , w _ { B } ) = \frac { 1 } { 2 } w _ { A } \| C - A \| ^ { 2 } + \frac { 1 } { 2 } w _ { B } \| C - B \| ^ { 2 } ,\tag{5}
$$

where $w _ { A }$ and $w _ { B }$ are non-negative weights that control the relative contributions of A and B in the construction of C during training. This formulation is used to interpret the interpolated prediction C as a weighted centroid between A and B. Under this framework, the resulting solution naturally balances the influence of A and B, such that C approaches A when $w _ { A }$ dominates and approaches B when $w _ { B }$ dominates. Under this view, minimizing the weighted interpolation objective yields a weighted combination of A and B, which is consistent with the interpolation form in $\mathrm { E q . ~ } ( 4 )$

By setting the derivative $\begin{array} { r } { \frac { \partial \mathcal { J } _ { C } } { \partial C } = 0 } \end{array}$ , we obtain the closedform solution to this weighted interpolation problem:

$$
C _ { \mathrm { o p t } } = \frac { w _ { A } A + w _ { B } B } { w _ { A } + w _ { B } } .\tag{6}
$$

Equivalently, according to Eq. (4), we write $C _ { \mathrm { o p t } } = s A + ( 1 -$ $s ) B$ with $\begin{array} { r } { s = \frac { w _ { A } } { w _ { A } + w _ { B } } } \end{array}$

Brightness Discrepancy Design of Weight $w _ { A }$ . To satisfy the Brightness-Discrepancy Awareness, the weight $w _ { A }$ is designed to quantify the brightness discrepancy between A and $Y .$ . To this end, the global brightness discrepancy between A and $Y$ is measured based on the mean intensities $\mu _ { A }$ and $\mu _ { Y }$ Accordingly, the weight $w _ { A }$ is defined as

$$
w _ { A } = ( \mu _ { A } - \mu _ { Y } ) ^ { 2 } .\tag{7}
$$

The quadratic form emphasizes larger brightness deviations while providing a smooth adaptive weighting signal. In our implementation, both A and Y are normalized to [0, 1] before computing $\mu _ { A } , \mu _ { Y }$ , and $w _ { A }$ . Therefore, $w _ { A }$ is defined on a bounded normalized brightness-discrepancy scale. Since the adaptive factor s is detached before backpropagation, $w _ { A }$ is used to determine the interpolation strength in the forward pass rather than to introduce an additional optimizable gradient pathway.

Complementary Design of Weight $w _ { B }$ . To implement the Smooth Optimization-Focus Shift, the weight $w _ { B }$ is designed as a non-linear complementary function of $w _ { A }$ . To achieve this, we introduce an exponential form with a threshold mechanism to control the transition speed:

$$
w _ { B } = \exp ( - \lambda ( w _ { A } - \delta ) ) , \quad \lambda > 0 , \delta > 0 ,\tag{8}
$$

where λ and δ are preset hyperparameters controlling the transition sharpness and the brightness-discrepancy transition point of the complementary weighting, respectively. Since $w _ { A }$ is defined on the normalized squared mean-brightness discrepancy scale, δ is specified on the same scale and is thus comparable across different datasets and backbone networks. In this paper, λ and δ are empirically fixed across all datasets and backbone networks, and their influence is further analyzed in the hyperparameter study. This exponential formulation provides a smooth monotonic response to the brightnessdiscrepancy weight, enabling the interpolation factor to change more sensitively when $w _ { A }$ approaches the transition region. The interpolation factor s is then defined as

$$
s = \frac { w _ { A } } { w _ { A } + \exp ( - \lambda ( w _ { A } - \delta ) ) } ,\tag{9}
$$

the brightness statistics and s are computed independently for each sample, without batch-wise aggregation.

Overall, this design defines a brightness-aware adaptive interpolation mechanism. Empirically, it allows the interpolation factor s to vary with the brightness discrepancy and to maintain a suitable balance throughout training. The weight $w _ { A }$ quantifies the brightness discrepancy between A and $Y$ and diminishes as A gets closer in brightness to Y. The exponential complementary function defined in Eq. (8) transforms the decreasing discrepancy in $w _ { A }$ into a smoothly increasing $w _ { B } ,$ with λ controlling the transition rate and δ specifying the onset of accelerated change. Importantly, this function induces a gradual adjustment of the interpolation factor s. When $w _ { A }$ is above the transition point δ, s changes relatively mildly and retains a larger contribution from the original prediction A. When $w _ { A }$ falls below this transition point, s decreases more sensitively, thereby increasing the relative contribution of the brightness-aligned image B. As a result, this design enables s to be adjusted according to different levels of brightness discrepancy, maintaining an appropriate interpolation balance throughout training.

In practice, during the early training stage, the model prediction A usually presents a relatively large brightness discrepancy from the ground-truth image Y . Accordingly, $w _ { A }$ tends to be larger on the normalized brightness-discrepancy scale, which preserves a relatively large interpolation factor s and retains a stronger contribution from the original prediction A at this stage. As training progresses and the brightness discrepancy gradually decreases, $w _ { A }$ correspondingly declines, causing s to decrease and increasing the relative contribution of the brightness-aligned image B. In this stage, the global brightness discrepancy contributes less dominantly to the reconstruction loss, so residual local discrepancies, including noise-related deviations and structural artifacts, exert relatively stronger influence on the gradient update.

## D. Analysis of Training Response Induced by DITM

To examine the training behavior induced by DITM, we analyze the response of the training objective in the input space. As defined in Eq. (4), C is obtained by adaptively interpolating A and B with the factor s. Although s is dynamically recomputed for each sample at every training iteration, its gradient is stopped before backpropagation. Therefore, within each optimization step, s is treated as a constant and does not introduce an additional gradient path through its own computation. Accordingly, in the following derivation, s denotes the value computed during the forward pass, with its gradient stopped during backpropagation.

Let L denote the reconstruction loss imposed on the interpolated prediction C. Although the input image X is not an optimization variable, the derivative of the loss with respect to X can serve as a local response indicator of how the objective behaves across different spatial regions. The corresponding response can be written as

$$
{ \frac { \partial { \mathcal { L } } } { \partial X } } = { \frac { \partial { \mathcal { L } } } { \partial C } } \left( s { \frac { \partial A } { \partial X } } + \left( 1 - s \right) { \frac { \partial B } { \partial X } } \right) .\tag{10}
$$

Since $B = \beta A$ , its derivative with respect to X can be further decomposed as

$$
\frac { \partial B } { \partial X } = \beta \frac { \partial A } { \partial X } + A \frac { \partial \beta } { \partial X } .\tag{11}
$$

Substituting this into the previous equation gives

$$
{ \frac { \partial { \mathcal { L } } } { \partial X } } = { \frac { \partial { \mathcal { L } } } { \partial C } } \left[ ( s + ( 1 - s ) \beta ) { \frac { \partial A } { \partial X } } + ( 1 - s ) A { \frac { \partial \beta } { \partial X } } \right] .\tag{12}
$$

The above formulation reveals that the training response consists of two components. The first term is proportional to $\textstyle { \frac { \partial A } { \partial X } }$ and preserves the spatially varying gradients produced by the backbone network. The second term arises from $\frac { \partial \beta } { \partial X }$ where $\beta$ depends on the global mean brightness $\mu _ { A }$ . Since $\mu _ { A }$ aggregates information over the entire prediction, this term introduces a luminance-related global coupling effect into the response. Nevertheless, this term does not impose an explicit local structural constraint; rather, its spatial effect is indirectly shaped by the prediction A and the backbone Jacobian.

At the early training stage, s remains relatively large, so the training objective places greater weight on the prediction A. In this regime, the response is mainly associated with correcting large global luminance errors and therefore tends to be broadly distributed rather than clearly localized to specific residual structures. As training proceeds and the brightness discrepancy decreases, s gradually decays according to the adaptive interpolation rule in Section III-C, increasing the relative influence of the brightness-aligned image B. Because B reduces the relative dominance of global brightness mismatch in the reconstruction loss, residual local discrepancies become more influential in the resulting gradient response.

Consequently, under DITM, the response tends to become less dominated by global luminance correction and more aligned with spatially non-uniform residual errors, including structural boundaries, texture regions, and noise-contaminated areas. This pattern is consistent with the view that DITM makes noise-related and structure-related residuals relatively more visible to the training objective during the middle and late training stages. The corresponding experimental observations in Section IV-D further provide empirical support for this training behavior.

## E. Further Discussion on Training–Inference Consistency

To further clarify the training–inference consistency of DITM, we discuss the role of the ground-truth brightness statistic during training and its relation to the deployed inference model. During training, the backbone receives only the low-light image X and produces the raw model prediction $\begin{array} { r } { A \ = \ f _ { \theta } ( X ) } \end{array}$ . The ground-truth image $Y$ is used only on the loss side to construct the interpolated prediction $C = { \mathrm { D I T M } } ( A , Y )$ , and is not provided as an input, auxiliary condition, or additional feature to the enhancement network. The corresponding training objective is ${ \mathcal { L } } ( C , Y )$

DITM introduces neither learnable parameters nor an additional prediction branch. Although the loss is imposed on $C ,$ the backbone parameters are updated through the dependency of C on the raw prediction A:

$$
\frac { \partial \mathcal { L } ( C , Y ) } { \partial \theta } = \frac { \partial \mathcal { L } } { \partial C } \left[ s I + ( 1 - s ) \frac { \partial B } { \partial A } \right] \frac { \partial A } { \partial \theta } .\tag{13}
$$

Here, I denotes the identity operator. In our implementation, the adaptive interpolation factor s is detached before backpropagation and is therefore treated as a constant for each sample within the current optimization step. Consequently, the ground-truth brightness statistic affects the optimization signal only through the deterministic loss-side transformation that reshapes A into $C ,$ rather than through a learnable GTdependent control pathway.

Importantly, DITM uses only the scalar mean-brightness statistic $\mu _ { Y }$ from the ground-truth image. Since $B \ = \ \beta A$ the interpolated prediction can be equivalently written as $C = [ s + ( 1 - s ) \beta ] A$ . Therefore, $C$ does not borrow any spatial content, such as structures or textures, from $Y ;$ all its spatial information is inherited from the raw prediction A. Consequently, improvements in local detail restoration and artifact suppression are still learned by the backbone and reflected in A.

After training, DITM is completely removed, and the deployed model directly produces $A ^ { * } ~ = ~ f _ { \theta ^ { * } } ( X )$ . Therefore, the inference dependency remains $X \to A ^ { * }$ , and neither the ground-truth image nor its brightness statistics are required at test time. DITM changes how the supervised training signal is presented to the backbone while preserving the original feedforward inference mapping. All reported validation and test results are computed from the raw output $A ^ { * }$ , rather than from the GT-dependent interpolated prediction C; when available, the ground-truth image is used only for metric computation.

## F. Computational Complexity of DITM

Based on Fig. 2, we analyze the additional computational and memory overhead introduced by DITM. The main computations involve constructing the mean brightness-aligned image B and the interpolated prediction C. Computing B requires evaluating the mean brightness of the model prediction $A ,$ which has linear complexity $\mathcal { O } ( H \times W )$ for an image of size $H \times W$ . Generating C involves element-wise interpolation between A and $B ,$ which is also linear in $H \times W$ The adaptive interpolation weight s is computed from scalar statistics and incurs negligible cost. Consequently, the overall additional computational complexity introduced by DITM remains $\mathcal { O } ( H \times W )$

In terms of memory, DITM requires storing B and $C ,$ each of size $H \times W$ . Compared to the backbone, whose dominant cost arises from multi-channel feature maps, both the extra computational and memory overheads introduced by DITM are negligible. Moreover, no additional trainable parameters are introduced, and the inference-time architecture and complexity remain identical to those of the original backbone.

## IV. EXPERIMENTS

This section experimentally evaluates DITM. Section IV-A introduces the datasets, evaluation metrics, baseline models, and implementation details. Sections IV-B and IV-C present quantitative and qualitative comparisons, respectively. Section IV-D further analyzes alternative interpolation strategies, hyperparameters, training behavior and overhead, numerical robustness under extremely low-light conditions, and performance under spatially non-uniform illumination. Finally, Section IV-E discusses the applicability scope and limitations of DITM.

Datasets. We evaluated our proposed DITM on eight widely used low-light image enhancement datasets. The paired datasets include LOLv1 [14], LOLv2-real [61], and LOLv2- synthetic [61], which provide aligned low-light and normallight images for supervised evaluation. LOLv1 contains 485 training images and 15 testing images, LOLv2-real includes 689 training images and 100 testing images, and LOLv2- synthetic provides 900 training images and 100 testing images. For unpaired datasets, we used LIME [6], VV [62], NPE [63], MEF [64], and DICM [65].

Baseline Methods. To evaluate the generality of DITM across different enhancement architectures, we integrated it into seven models covering three major architectural paradigms: CNN-based, Transformer-based, and Mambabased approaches. The CNN-based baselines include RetinexNet [14], MIRNet [66], and CID-Net [29]; the Transformer-based baselines include LLFormer [18] and RetinexFormer [17]; and the Mamba-based baselines include RetinexMamba [20] and MambaLLIE [22].

Evaluation Metrics. To evaluate both distortion fidelity and perceptual quality, we adopted a combination of fullreference and no-reference metrics. For paired datasets, we employed standard full-reference metrics including PSNR, SSIM, DeepSSIM [67], and LPIPS [68] to evaluate pixel-wise fidelity and perceptual similarity. For unpaired datasets without ground-truth references, we followed the evaluation protocol in [69] and adopted perceptually-aligned no-reference metrics that better reflect human visual perception. Specifically, we used MUSIQ [70] and Q-Align [71], the latter of which provides two complementary evaluation scores: image quality assessment (IQA) and image aesthetic assessment (IAA). These perceptually aligned no-reference metrics, including the LMM-based Q-Align, provide complementary evaluation of perceptual quality and aesthetic appeal for unpaired low-light images. It is worth noting that we also reported Q-Align (IQA and IAA) on paired datasets for complementary perceptual evaluation.

Implementation Details. We integrated DITM into each baseline and retrained both the original and DITM-augmented models from scratch following the official configurations. For each paired comparison, all training and evaluation settings were kept identical, including the dataset, preprocessing, patch size, batch size, optimizer, learning rate and its schedule, training length, data augmentation, loss function, and evaluation protocol. The only difference was whether DITM was enabled during training, while the inference procedure remained unchanged. Each baseline and its DITM-augmented counterpart were independently trained three times, and the results are reported as mean and standard deviation. The supervised feed-forward LLIE experiments were conducted using NVIDIA RTX 4090 GPUs. The hyperparameters λ and δ were fixed to 10 and 0.3, respectively, across all backbones. All brightness statistics were computed on images normalized to [0, 1], and δ is therefore defined on the normalized squared mean-brightness discrepancy scale.

Performance on Paired Datasets. Table I shows that DITM improves the mean performance in 124 of the 126 paired comparisons across three LOL datasets, seven LLIE backbones, and six evaluation metrics. Consistent gains are observed in the main full-reference metrics, including PSNR, SSIM, LPIPS, and DeepSSIM, while only minor degradations occur in IAA for MIRNet on LOLv1 and MambaLLIE on LOLv2-real. The three-run results further show that the mean improvements are consistently observed across most paired comparisons. The evaluated models cover CNN-, Transformer-, and Mamba-based architectures, demonstrating that DITM generalizes well across diverse supervised feed-forward LLIE backbones. The improvements obtained with CID-Net further indicate that DITM is complementary to existing denoisingoriented enhancement designs.

Performance on Unpaired Datasets. We also conducted quantitative evaluation on five unpaired datasets. From the weighted average performance across the five datasets shown in Table II, all seven methods equipped with DITM improve the weighted average scores on the three metrics, except for the MUSIQ score of RetinexNet and CID-Net with DITM. These results suggest that DITM generally improves or maintains cross-dataset generalization on unpaired real-world low-light images, especially in terms of Q-Align IQA and IAA.

## C. Visual Quality Evaluation

Visual Comparison on the Paired Datasets. Fig. 3 provides visual comparisons on paired benchmarks. The baseline models improve visibility but still suffer from visible degradations in challenging low-light regions, including noise and structural artifacts. After integrating DITM, the enhanced results are closer to the ground truth, with reduced noise, fewer artifacts, and improved structural consistency. The zoomed-in regions in Fig. 3 provide a more detailed comparison. For CNN-based methods, DITM particularly aids in stabilizing color restoration while simultaneously suppressing chroma noise. In Transformer-based models, the improvement is particularly evident in reducing structural artifacts and recovering sharper edges. For Mamba-based methods, DITM further improves texture representation while reducing visual artifacts.

Visual Comparison on the Unpaired Datasets. As shown in Fig. 4, DITM improves the robustness of different supervised LLIE models under real-world low-light conditions, producing more visually consistent results with fewer artifacts and better brightness balance. Baseline methods such as CID-Net introduce obvious halo artifacts around strongly illuminated regions in their enhanced results. For RetinexFormer, Retinex-Mamba and MIRNet, their results show varying degrees of local artifacts or exposure instability. In addition, LLFormer shows large-area stripe artifacts, while RetinexNet exhibits pronounced color blotches. These issues are alleviated after integrating DITM, as evidenced by the examples in Fig. 4, indicating improved robustness.

![](images/e836655bed960a7899405f724746c75ef7229ec9a32c2abf391fd66366c8546d.jpg)  
Fig. 3. Visual comparison of different low-light image enhancement methods. From left to right: (a) Low-light Input, (b) Baseline model, (c) Ours (Baselin + DITM), and (d) Ground Truth. By integrating DITM into these existing supervised LLIE frameworks, the visual quality of the shown examples is improved in terms of artifact suppression, noise reduction, and preservation of structural details. Zoomed-in regions are provided for better visualization of local details

TABLE I  
QUANTITATIVE COMPARISON ON LOL DATASETS. EACH SETTING WAS INDEPENDENTLY REPEATED THREE TIMES, AND THE MEAN AND STANDARD DEVIATION ARE REPORTED. VALUES IN PARENTHESES DENOTE THE MEAN CHANGE OF DITM OVER THE CORRESPONDING BASELINE. RED INDICATES IMPROVEMENT, WHILE GREEN INDICATES DEGRADATION.
<table><tr><td rowspan="2">Datasets</td><td rowspan="2">Methods</td><td colspan="5">Evaluation Metrics</td></tr><tr><td>PSNR↑ SSIM↑</td><td>LPIPS↓ 0.424±0.004</td><td>DeepSSIM↑ 0.3834±0.0028</td><td>IQA↑ 2.032±0.019</td><td>IAA↑ 1.204±0.007</td></tr><tr><td rowspan="8">LOLv1</td><td>RetinexNet RetinexNet with DITM</td><td>17.076±0.080 17.546±0.111 (+0.470)</td><td>0.7095±0.0020 0.7190±0.0013 (+0.0095)</td><td>0.403±0.015 (-0.021)</td><td>)0.4002±0.0208 (+0.0168)</td><td>2.144±0.036 (+0.112)</td><td>1.279±0.047 (+0.075)</td></tr><tr><td>MIRNet MIRNet with DITM</td><td>21.447±0.107 21.703±0.101 (+0.256)</td><td>0.7849±0.0030 0.7963±0.0011(+0.0114)</td><td>0.225±0.001 0.217±0.004 (-0.008)</td><td>0.6072±0.0032 0.6265±0.0043 (+0.0193)</td><td>2.937±0.016 2.973±0.030 (+0.036)</td><td>1.759±0.020 1.750±0.019 (-0.009)</td></tr><tr><td>CID-Net CID-Net with DITM</td><td>24.102±0.184 24.447±0.372 (+0.345)</td><td>0.8563±0.0016 0.8628±0.0009 (+0.0065)</td><td>0.098±0.003 0.087±0.001 (-0.011)</td><td>0.8193±0.0125 0.8494±0.0042 (+0.0301)</td><td>3.889±0.048 3.969±0.013 (+0.080)</td><td>2.097±0.009 2.125±0.003 (+0.028)</td></tr><tr><td>LLFormer LLFormer with DITM</td><td>23.236±0.161 23.356±0.191(+0.120)</td><td>0.8129±0.0011 0.8215±0.0018 (+0.0086)</td><td>0.171±0.002 0.162±0.003 (-0.009)</td><td>0.6632±0.0019 0.6763±0.0038 (+0.0131)</td><td>3.125±0.025 3.176±0.019 (+0.051)</td><td>1.839±0.019 1.867±0.005 (+0.028)</td></tr><tr><td>RetinexFormer RetinexFormer with DITM</td><td>23.478±0.331 23.946±0.152 (+0.468)</td><td>0.8266±0.0015 0.8437±0.0014 (+0.0171)</td><td>0.152±0.003 0.121±0.002 (-0.031)</td><td>0.6739±0.0072 0.7470±0.0092 (+0.0731)</td><td>3.265±0.016 3.611±0.056 (+0.346)</td><td>1.932±0.007 2.036±0.006 (+0.104)</td></tr><tr><td>RetinexMamba RetinexMamba with DITM</td><td>23.345±0.117 23.729±0.176 (+0.384)</td><td>0.8311±0.0027 0.8458±0.0008 (+0.0147)</td><td>0.142±0.007</td><td>0.7001±0.0124 0.7536±0.0109 (+0.0535)</td><td>3.364±0.047</td><td>1.986±0.014</td></tr><tr><td>MambaLLIE MambaLLIE with DITM</td><td>22.985±0.193 23.291±0.259 (+0.306)</td><td>0.8232±0.0019 0.8436±0.0028 (+0.0204)</td><td>0.123±0.004 (-0.019) 0.151±0.004 0.128±0.002 (-0.023)</td><td>0.6916±0.0027 0.7418±0.0021 (+0.0502)</td><td>3.644±0.067 (+0.280) 3.369±0.026</td><td>2.038±0.011 (+0.052) 2.037±0.023</td></tr><tr><td>RetinexNet MIRNet</td><td>16.494±0.437</td><td>0.6705±0.0123</td><td>0.519±0.021</td><td>0.3049±0.0272</td><td>3.574±0.047 (+0.205) 1.826±0.013</td><td>2.055±0.021 (+0.018) 1.223±0.008</td></tr><tr><td rowspan="7">LOLv2-real</td><td>RetinexNet with DITM</td><td>17.206±0.354 (+0.712)</td><td>0.7015±0.0081 (+0.0310)</td><td>0.501±0.032 (-0.018)</td><td>0.3437±0.0140 (+0.0388)</td><td>2.016±0.171 (+0.190)</td><td>1.338±0.104 (+0.115)</td></tr><tr><td>MIRNet with DITM CID-Net</td><td>21.504±0.173 21.863±0.147 (+0.359)</td><td>0.8066±0.0061 0.8210±0.0033 (+0.0144)</td><td>0.324±0.035 0.263±0.017 (-0.061)</td><td>0.4913±0.0411 0.5436±0.0039 (+0.0523)</td><td>2.448±0.161 2.698±0.052 (+0.250)</td><td>1.439±0.092 1.597±0.015 (+0.158)</td></tr><tr><td>CID-Net with DITM LLFormer</td><td>22.711±0.192 23.010±0.187 (+0.299) 0.8613±0.0010 (+0.0087)</td><td>0.8526±0.0029</td><td>0.138±0.002 0.127±0.004 (-0.011)</td><td>0.7821±0.0051 0.7981±0.0022 (+0.0160)</td><td>3.551±0.012 )3.616±0.018 (+0.065)</td><td>1.996±0.015 2.001±0.013 (+0.005)</td></tr><tr><td>LLFormer with DITM RetinexFormer</td><td>21.178±0.035 21.482±0.161 (+0.304) 0.8230±0.0064 (+0.0138)</td><td>0.8092±0.0053</td><td>0.221±0.021</td><td>0.5889±0.0299 0.193±0.009 (-0.028) 0.6289±0.0175 (+0.0400)</td><td>2.802±0.099 2.920±0.081 (+0.118)</td><td>1.666±0.069 1.751±0.042 (+0.085)</td></tr><tr><td>RetinexFormer with DITM RetinexMamba</td><td>21.977±0.083 22.641±0.254 (+0.664) 0.8544±0.0033 (+0.0149)</td><td>0.8395±0.0022</td><td>0.171±0.006</td><td>0.6645±0.0030 0.145±0.006 (-0.026) 0.7196±0.0034 (+0.0551) 3.430±0.017 (+0.247)</td><td>3.183±0.016</td><td>1.874±0.007 1.970±0.009 (+0.096)</td></tr><tr><td>RetinexMamba with DITM MambaLLIE</td><td>21.960±0.367 22.394±0.234 (+0.434) 0.8600±0.0022 (+0.0175) 21.586±0.129</td><td>0.8425±0.0069</td><td>0.165±0.005 0.146±0.005 (-0.019)</td><td>0.6929±0.0079 0.7461±0.0193 (+0.0532)</td><td>3.295±0.073 ) 3.576±0.108 (+0.281) 2.011±0.023 (+0.053)</td><td>1.958±0.031</td></tr><tr><td>MambaLLIE with DITM</td><td>22.346±0.261 (+0.760) 0.8562±0.0050 (+0.0076)</td><td>0.8486±0.0020</td><td>0.162±0.004 0.152±0.007 (-0.010)</td><td>0.7179±0.0035 )0.7298±0.0036 (+0.0119) 3.408±0.017 (+0.009)</td><td>3.399±0.042</td><td>2.005±0.014 1.993±0.008 (-0.012)</td></tr><tr><td rowspan="7">LOLv2-synthetic</td><td>RetinexNet RetinexNet with DITM</td><td>18.965±0.098 19.276±0.068 (+0.311) 0.8086±0.0014 (+0.0139)</td><td>0.7947±0.0005</td><td>0.252±0.002 0.238±0.005 (-0.014)</td><td>0.6126±0.0016 0.6382±0.0094 (+0.0256)</td><td>2.449±0.013 2.590±0.014 (+0.141)</td><td>1.756±0.014 1.814±0.008 (+0.058)</td></tr><tr><td>MIRNet MIRNet with DITM</td><td>21.972±0.077 22.098±0.092 (+0.126)</td><td>0.8935±0.0006 0.8990±0.0008 (+0.0055)</td><td>0.117±0.002</td><td>0.8196±0.0020 0.8408±0.0012 (+0.0212)</td><td>2.970±0.024</td><td>2.200±0.007</td></tr><tr><td>CID-Net |CID-Net with DITM</td><td>25.519±0.088 25.719±0.250 (+0.200)</td><td>0.9339±0.0011</td><td>0.105±0.001 (-0.012) 0.049±0.000</td><td>0.9243±0.0020</td><td>3.051±0.007 (+0.081) 3.283±0.004</td><td>2.201±0.010 (+0.001) 2.163±0.004 2.176±0.005 (+0.013)</td></tr><tr><td>LLFormer |LLFormer with DITM</td><td>24.860±0.083 25.204±0.136 (+0.344)</td><td>0.9384±0.0015 (+0.0045) 0.9268±0.0006 0.9327±0.0005 (+0.0059)</td><td>0.045±0.001 (-0.004) 0.059±0.001 0.054±0.001 (-0.005)</td><td>0.9296±0.0008 (+0.0053) 0.9030±0.0011 0.9134±0.0007 (+0.0104)</td><td>3.304±0.010 (+0.021) 3.164±0.003 3.192±0.003 (+0.028)</td><td>2.117±0.005 2.128±0.003 (+0.011)</td></tr><tr><td>RetinexFormer RetinexFormer with DITM</td><td>25.308±0.173 25.657±0.101 (+0.349)</td><td>0.9282±0.0013 0.9367±0.0025 (+0.0085)</td><td>0.060±0.001 0.052±0.002 (-0.008)</td><td>0.9061±0.0010 0.9197±0.0034 (+0.0136)</td><td>3.182±0.007 3.246±0.034 (+0.064)</td><td>2.130±0.002 2.153±0.015 (+0.023)</td></tr><tr><td>RetinexMamba RetinexMamba with DITM</td><td>25.705±0.092 25.920±0.157 (+0.215)</td><td>0.9345±0.0006 0.9417±0.0007 (+0.0072)</td><td>0.056±0.001 0.047±0.001 (-0.009)</td><td>0.9123±0.0012 0.9286±0.0016 (+0.0163)</td><td>3.224±0.019 3.329±0.017 (+0.105)</td><td>2.160±0.007 2.197±0.009 (+0.037)</td></tr><tr><td>MambaLLIE MambaLLIE with DITM</td><td>25.229±0.090 25.834±0.172 (+0.605) 0.9411±0.0013 (+0.0061)</td><td>0.9350±0.0018</td><td>0.054±0.002</td><td>0.9160±0.0061 0.047±0.002 (-0.007) 0.9280±0.0012 (+0.0120)</td><td>3.243±0.038 3.317±0.009 (+0.074) 2.203±0.003 (+0.017)</td><td>2.186±0.009</td></tr></table>

## D. Further Analysis of DITM

1) Comparison with Alternative Interpolation Strategies: To verify that the gain comes from the proposed adaptive interpolation rather than from generic prediction interpolation, we compared DITM with several heuristic interpolation strategies on three paired datasets, using RetinexFormer as baseline. The interpolation strategies for comparison include:

• Baseline: The original backbone network trained without interpolation.

• Directly Using B (B-only): The model prediction A is completely replaced by the mean brightness-aligned image B during loss computation.

• Interpolation with Static Weights (ISW): A fixed fusion strategy with constant weights (0.8 for the model prediction A, 0.2 for the mean brightness-aligned image B) throughout training.

• Heuristic Dynamic Interpolation (HDI): A three-stage training schedule that first optimizes exclusively with A (0–50k iterations), then linearly interpolates between the model prediction A and the mean brightness-aligned image B from 50k to 100k iterations using a linearly varying weight decreasing from 1 to 0, and finally transitions entirely to B (100k–150k).

Experimental results based on different strategies are shown in Table III. Across the three paired datasets (LOLv1, LOLv2- real, and LOLv2-synthetic), the interpolation strategy used in DITM consistently outperforms the others under different evaluation metrics, demonstrating the effectiveness of our dynamic image interpolation mechanism.

We further analyze how different strategies affect the optimization process. We conducted an experiment on the LOLv1 training dataset, and recorded the training loss, SSIM, and LPIPS over iterations during the training process, as shown in Fig. 5. Among these strategies, B-only completely replaces the raw prediction A with the brightness-aligned image B, thereby removing the direct supervision signal associated with the original brightness discrepancy of A. This weakens effective optimization and leads to degraded performance. Fig. 5 shows that this strategy substantially degrades training stability and final performance. HDI adopts a manually designed interpolation strategy, whose heuristic scheduling limits adaptability across datasets. As ISW introduces a fixed interpolation weight, its static nature restricts flexibility under varying image conditions. In contrast, DITM determines the interpolation factor on a per-sample basis, where the weight s is adaptively estimated for each image in the training set, leading to consistently better performance than other strategies.

TABLE II  
QUANTITATIVE COMPARISON ON FIVE UNPAIRED DATASETS. THE WEIGHTED AVERAGE PERFORMANCE IS OBTAINED ACCORDING TO THE IMAGE NUMBER OF THE FIVE DATASETS.
<table><tr><td rowspan="2">Methods</td><td colspan="3">DICM</td><td colspan="3">LIME</td><td colspan="3">MEF</td><td colspan="3"></td><td colspan="3"></td><td colspan="3">Weighted Average</td></tr><tr><td>|MUSIQ↑</td><td></td><td>IQA↑ IAA↑</td><td>|MUSIQ↑</td><td>IQA↑ IAA↑</td><td></td><td>|MUSIQ↑</td><td>IQA↑ IAA↑</td><td></td><td>MUSIQ↑</td><td></td><td>IQA↑ IAA↑</td><td>MUSIQ↑</td><td></td><td>IQA↑ IAA↑</td><td>MUSIQ↑</td><td>IQA↑</td><td>IAA↑</td></tr><tr><td>RetinexNet</td><td>61.058</td><td>3.165</td><td>2.349</td><td>62.558</td><td>2.555</td><td>2.236</td><td>67.494</td><td>2.877</td><td>2.395</td><td>69.172</td><td>2.486</td><td>1.896</td><td>39.267</td><td>2.667</td><td>2.105</td><td>58.452</td><td>2.944</td><td>2.272</td></tr><tr><td>RetinexNet with DITM</td><td>61.112</td><td>3.225</td><td>2.368</td><td>61.729</td><td>2.628</td><td>2.269</td><td>67.111</td><td>2.952</td><td>2.459</td><td>68.542</td><td>2.617</td><td>2.007</td><td>37.892</td><td>2.779</td><td>2.158</td><td>58.067 (↓)</td><td>3.021 (↑)</td><td>2.310 (↑)</td></tr><tr><td>MIRNet</td><td>57.611</td><td>3.551</td><td>2.473</td><td>59.912</td><td>2.816 2.292</td><td></td><td>61.546</td><td>3.251</td><td>2.609</td><td>64.569</td><td>3.156</td><td>2.329</td><td>36.278</td><td>3.127</td><td>2.084</td><td>54.748</td><td>3.350</td><td>2.395</td></tr><tr><td>MIRNet with DITM</td><td>58.039</td><td>3.635</td><td>2.476</td><td>59.621</td><td>2.924</td><td>2.357</td><td>61.676</td><td>3.281</td><td>2.635</td><td>64.708</td><td>3.282</td><td>2.353</td><td>37.796</td><td>3.281</td><td>2.131</td><td>55.267 (↑)</td><td>3.444 (↑)</td><td>2.414 (↑)</td></tr><tr><td>CID-Net</td><td>58.324</td><td>3.619 2.399</td><td></td><td>59.359</td><td>2.939</td><td>2.397</td><td>62.508</td><td>3.463 2.598</td><td></td><td>66.436</td><td></td><td>3.475 2.436</td><td>40.264</td><td>3.606</td><td>2.230</td><td>56.081</td><td>3.534</td><td>2.396</td></tr><tr><td>CID-Net with DITM</td><td>58.165</td><td>3.659</td><td>2.439</td><td>59.672</td><td>3.022</td><td>2.407</td><td>62.377</td><td>3.459</td><td>2.626</td><td>66.986</td><td></td><td>3.516 2.469</td><td>39.925</td><td>3.625</td><td>2.241</td><td>55.973 (↓)</td><td>3.567 (↑)</td><td>2.426 (↑)</td></tr><tr><td>LLFormer</td><td>58.983</td><td>3.743</td><td>2.527</td><td>57.597</td><td>3.320</td><td>2.450</td><td>63.144</td><td>3.485 2.603</td><td></td><td>65.904</td><td></td><td>3.598 2.593</td><td>32.546</td><td></td><td>3.568 2.161</td><td>54.903</td><td>3.634</td><td>2.466</td></tr><tr><td>LLFormer with DITM</td><td>59.316</td><td>3.775</td><td>2.539</td><td>58.187</td><td>3.275</td><td>2.450</td><td>63.032</td><td>3.497</td><td>2.623</td><td>66.480</td><td></td><td>3.616 2.560</td><td>32.983</td><td>3.579</td><td>2.175</td><td>55.232 (↑)</td><td>3.652 (↑)</td><td>2.476 (↑)</td></tr><tr><td>RetinexFormer</td><td>58.662</td><td>3.750 2.537</td><td></td><td>59.518</td><td>3.327</td><td>2.487</td><td>62.418</td><td>3.475 2.567</td><td></td><td>66.484</td><td></td><td>3.418 2.487</td><td>40.968</td><td>3.472</td><td>2.148</td><td>56.399</td><td>3.608</td><td>2.461</td></tr><tr><td>RetinexFormer with DITM</td><td>58.844</td><td>3.7562.525</td><td></td><td>59.098</td><td>3.341</td><td>2.449</td><td>62.729</td><td>3.521</td><td>2.629</td><td>67.318</td><td>3.702</td><td>2.626</td><td>42.046</td><td>3.520</td><td>2.177</td><td>56.760 (↑)</td><td>3.645 (↑)</td><td>2.474 (↑)</td></tr><tr><td>RetinexMamba</td><td>58.894</td><td>3.780 2.548</td><td></td><td>60.734</td><td>3.325</td><td>2.466</td><td>61.832</td><td>3.405</td><td>2.530</td><td>66.212</td><td></td><td>3.486 2.507</td><td>59.349</td><td>3.583</td><td>2.200</td><td>59.971</td><td>3.639</td><td>2.469</td></tr><tr><td>RetinexMamba with DITM</td><td>58.883</td><td>3.8062.558</td><td></td><td>60.575</td><td>3.285</td><td>2.442</td><td>62.684</td><td>3.525</td><td>2.649</td><td>66.845</td><td>3.496</td><td>2.523</td><td>59.979</td><td>3.683</td><td>2.233</td><td>60.223 (↑)</td><td>3.686 (↑)</td><td>2.498 (↑)</td></tr><tr><td>MambaLLIE</td><td>59.152</td><td>3.801</td><td>2.555</td><td>60.836</td><td>3.261</td><td>2.468</td><td>62.306</td><td>3.455</td><td>2.625</td><td>65.968</td><td>3.508</td><td>2.513</td><td>58.926</td><td>3.557</td><td>2.187</td><td>60.086</td><td>3.649</td><td>2.486</td></tr><tr><td>MambaLLIE with DITM</td><td>59.058</td><td>3.854 2.577</td><td></td><td>61.034</td><td>3.340 2.533</td><td></td><td>62.826</td><td>3.535 2.649</td><td></td><td>66.120</td><td></td><td>3.384 2.450</td><td>59.789</td><td>3.6642.241</td><td></td><td>60.291 (↑)</td><td>3.707 (↑)</td><td>2.512 (↑)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE III  
ABLATION STUDY OF ALTERNATIVE INTERPOLATION STRATEGIES USING RETINEXFORMER AS THE BASELINE.
<table><tr><td rowspan="2">Methods</td><td colspan="3">LOLv1</td><td colspan="3">LOLv2-real</td><td colspan="3">LOLv2-synthetic</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Baseline</td><td>23.498</td><td>0.8264</td><td>0.153</td><td>22.066</td><td>0.8378</td><td>0.178</td><td>25.149</td><td>0.9267</td><td>0.061</td></tr><tr><td>B-only</td><td>20.842</td><td>0.7966</td><td>0.165</td><td>20.370</td><td>0.8005</td><td>0.334</td><td>20.462</td><td>0.8766</td><td>0.125</td></tr><tr><td>ISW</td><td>23.381</td><td>0.8323</td><td>0.136</td><td>22.122</td><td>0.8447</td><td>0.154</td><td>25.497</td><td>0.9278</td><td>0.059</td></tr><tr><td>HDI</td><td>22.413</td><td>0.8333</td><td>0.127</td><td>21.903</td><td>0.8388</td><td>0.167</td><td>25.713</td><td>0.9346</td><td>0.053</td></tr><tr><td>DITM (Ours)</td><td>24.050</td><td>0.8444</td><td>0.120</td><td>22.934</td><td>0.8532</td><td>0.139</td><td>25.717</td><td>0.9394</td><td>0.050</td></tr></table>

2) Hyperparameter Study: To investigate the influence of hyperparameters λ and δ on the overall performance, we conducted sensitivity experiments using RetinexFormer and CID-Net on validation subsets of LOLv1 and LOLv2-synthetic. For LOLv1, the original 485 training pairs were further split into 445 pairs for model training and 40 pairs for validating different hyperparameter settings. For LOLv2-synthetic, 800 of the original 900 training pairs were used for model training, and the remaining 100 pairs were used for validation. We evaluated $\lambda \in \{ 6 , 1 0 , 1 4 , 1 8 \}$ and $\delta \in \{ 0 . 2 , 0 . 2 5 , 0 . 3 , 0 . 3 5 \}$ and report the corresponding SSIM and LPIPS values in Fig. 6.

The figure shows that model performance remains relatively stable across different hyperparameter combinations, with several settings achieving strong results, indicating that DITM is not overly sensitive to the choices of λ and δ. These results are also generally consistent with the roles of the two hyperparameters defined in Eq. (9). Specifically, λ controls the variation rate of s, while δ determines the transition region of the interpolation factor, around which s changes more rapidly and the contribution of global brightness correction is gradually adjusted during training. Based on its consistently competitive validation performance across the four model–dataset settings, we select $( \lambda , \delta ) \ : = \ : ( 1 0 , 0 . 3 )$ as the unified configuration and use it in all other experiments without model- or dataset-specific retuning.

3) Analysis of Training Process with DITM: We further investigated the training behavior associated with DITM. In particular, we aimed to provide empirical evidence consistent with the analysis in Section III-D, and to examine how DITM affects model training across different stages. The experiment was conducted on the LOLv1 training dataset, using RetinexFormer as the baseline method. Fig. 7 illustrates the enhancement results obtained from different training stages, while Fig. 8 presents input gradient and reconstruction error maps, jointly providing complementary insights into the training behavior associated with DITM.

At a typical early stage of training (10K iterations), the gradient maps of both models are widely distributed across the image (Fig. 8(a)), with gradient responses appearing in both structural and background regions. This indicates that the network is still in an exploratory phase without a clearly localized training response. Correspondingly, the reconstruction error maps (Fig. 8(b)) exhibit large errors across the entire image. As shown in Fig. 7, both models yield visibly underenhanced results with less-suppressed noise.

![](images/7d8d11c60199b2e94cb67b426e37a6c06270d291f1a9a4d94c46cec6bd6ef80b.jpg)  
Fig. 4. Visual comparison on unpaired real-world low-light datasets. From left to right: (a) Input, (b) Baseline, and (c) Ours (Baseline + DITM). The proposed DITM generally improves the generalization capability of existing supervised LLIE frameworks, resulting in improved visibility, reduced noise, and more robust restoration performance. Zoomed-in views are provided for a subset of regions for detailed comparison, while some regions are additionally highlighted using red dashed boxes to indicate noticeable artifacts.  
Baseline Directly Using B (B-only) Interpolation with Static Weights (ISW) Heuristic Dynamic Interpolation (HDI) DITM (Ours)

![](images/a86fd6ed6433307929212f3815a8fa6936b1f05815a12ed019cfb82a4ed82d23.jpg)

![](images/425845cbebc0558b6cf1e0412ee43759151fc6b8aeabd6bcdad6d1308048ea37.jpg)

![](images/8394bbeac44e57ec8911f55a6f5cce31b7ee378f601ae267aaace1dbb6361479.jpg)  
Fig. 5. Comparison of training dynamics across different strategies on the LOLv1 training dataset. Curves of training loss, SSIM, and LPIPS are shown over 150K iterations, with values recorded every 1K iterations. Compared with the other strategies, DITM shows better performance over most of the training process.

At a typical middle stage of training (80K iterations), noticeable differences begin to emerge. As shown in Fig. 8(a), the baseline continues to exhibit strong gradient responses around structural regions, while its responses in background areas remain scattered and relatively weak. In contrast, DITM produces more concentrated gradients on structurally relevant regions and suppresses irrelevant responses in the background. Correspondingly, the reconstruction error maps in Fig. 8(b) show that the baseline still contains obvious residual errors, whereas DITM achieves more localized and reduced errors. These observations are further supported by the visual comparison in Fig. 7, where DITM exhibits clearer structures and less noise compared to the baseline. This behavior is consistent with the analysis in Section III-D: as the adaptive factor s evolves, the contribution of the mean brightnessaligned image B branch increases, reducing the influence of global brightness discrepancies and making structural and texture-related residuals relatively more visible to the training objective.

At a typical late stage of training (150K iterations), the differences become more pronounced. The gradient responses of DITM are largely suppressed in non-structural regions (Fig. 8(a)), indicating that the optimization has become more focused and stable. Meanwhile, the corresponding error maps (Fig. 8(b)) are significantly cleaner and more uniform. In contrast, the baseline still shows residual errors and noticeable gradient responses in background areas, suggesting that the baseline still retains more residual optimization responses in background regions. As illustrated in Fig. 7, DITM produces more visually pleasing results with less noise and more consistent structural details.

Overall, these observations are consistent with the analysis in Section III-D. The design of adaptive interpolation tries to reduce the influence of global brightness discrepancy as training progresses, and encourages the model to focus more on appearance-related details, such as structures and textures, at middle and late training stages.

To quantitatively examine whether the adaptive interpolation factor s evolves consistently with its intended design, we record its per-sample distribution during the training of RetinexFormer at 10K, 80K, and 150K iterations on LOLv1, LOLv2-real, and LOLv2-synthetic. In addition to the violin distributions, we report the mean value $\mu _ { s }$ and the low-factor ratio.

![](images/68e7d9ba53409db765fa0f8cace5b3f69f0b65f13fe011910cc14c12131470b3.jpg)

Fig. 6. SSIM and LPIPS of RetinexFormer and CID-Net under different λ and δ settings on the held-out validation subsets of LOLv1 and LOLv2-synthetic.  
![](images/b3ad2dd4992dbc7f7ff37c0f8f56cc181f65c79439d52202371c68bf6fffd057.jpg)  
Fig. 7. Qualitative comparison of the enhancement performance at different training stages (10K, 80K, and 150K iterations) on the LOLv1 training dataset. Compared to the baseline, the model with DITM begins to exhibit superior detail restoration and noise suppression from the middle stage of training. Zoomedin views are provided to better illustrate the progressive improvements across different training stages.

(a) Input Gradient Maps <sup>∂</sup>L  
![](images/2c39eacc4b7771e1dd1abee2ec9c508a74c5a303b206c7f905f5501408a980a1.jpg)

(b) Reconstruction Error Maps  
![](images/63c92cf8cceb397132c53da5ec608100eaa2101710a4bd6ff5402199bbfe9823.jpg)  
Fig. 8. Visualization of input gradient maps $\textstyle { \frac { \partial { \mathcal { L } } } { \partial X } }$ (gradients of the loss with respect to the input) and reconstruction error maps at different training stages (10K, 80K, and 150K iterations) for the Baseline (top row) and DITM (bottom row). (a) Input gradient maps suggest that DITM progressively yields more concentrated responses on structural regions, while the baseline exhibits more diffuse responses throughout training. (b) Reconstruction error maps illustrate that the errors gradually diminish during training for both methods. The model with DITM exhibits cleaner and more spatially uniform residual patterns in the later stage, suggesting a more focused training response in later stages.

$$
R _ { s < 0 . 1 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } ( s _ { i } < 0 . 1 ) \times 1 0 0 \% ,\tag{14}
$$

where N denotes the number of recorded samples and $\mathbb { I } ( \cdot )$ is the indicator function. A larger $R _ { s < 0 . 1 }$ indicates that a greater proportion of samples assign a stronger contribution to the mean brightness-aligned prediction B.

![](images/5b545b1fc3cf67fb4fbfdf01678460f877685f10fb7af01fb5701adaf3c159c4.jpg)

![](images/1fd3d06e3c35bcc7b096bb7cca6d2ea77c26688287aa3ef54046d89192b8aa62.jpg)

![](images/bafb6280935ecf9b53c449222069e2030c99e7da7c11359c7e9ca6a0cedc8282.jpg)  
Fig. 9. Distribution evolution of the adaptive interpolation factor s during the training of RetinexFormer on LOLv1, LOLv2-real, and LOLv2-synthetic The distributions are recorded at 10K, 80K, and 150K training iterations. The width of each violin represents the distribution density of the per-sample interpolation factors, while the diamond marker denotes the corresponding mean value $\mu _ { s } .$ We additionally report $R _ { s < 0 . 1 } .$ , which denotes the proportion of recorded samples whose interpolation factors are smaller than 0.1 at each training stage. Across all three datasets, the dominant distribution of s progressively shifts toward smaller values, accompanied by decreasing $\mu _ { s }$ and increasing $R _ { s < 0 . 1 }$ . The remaining thin upper tails indicate that a small number of samples retain relatively large s values at later stages, reflecting the per-sample adaptive behavior of DITM rather than a fixed iteration-dependent schedule.

As shown in Fig. 9, the distribution mass of s progressively shifts toward smaller values on all three datasets. On LOLv1, $\mu _ { s }$ decreases from 0.381 at 10K iterations to 0.149 at 80K and 0.082 at 150K, while $R _ { s < 0 . 1 }$ increases from 49.1% to 74.8% and 85.4%. On LOLv2-real, $\mu _ { s }$ decreases from 0.335 to 0.173 and 0.072, while $R _ { s < 0 . 1 }$ increases from 54.3% to 72.3% and 86.1%. On LOLv2-synthetic, $\mu _ { s }$ decreases from 0.274 to 0.073 and 0.031, while $R _ { s < 0 . }$ increases from 57.8% to 87.0% and 93.6%.

These statistics verify the intended stage-dependent behavior of DITM. At the earlier training stage, relatively larger values of s keep the interpolated prediction C closer to the raw prediction A, preserving stronger direct supervision for correcting substantial brightness discrepancy. As training progresses, the dominant distribution of s shifts toward smaller values, moving C closer to the mean brightness-aligned prediction B. This reduces the relative influence of residual image-level luminance mismatch in the loss-side prediction and makes non-global residual errors, including noise, texture degradation, structural artifacts, and local exposure mismatch, relatively more involved in subsequent optimization.

The thin upper tails observed at later stages indicate that a small number of samples still retain relatively large s values. This behavior is consistent with the per-sample design of DITM: samples with larger remaining brightness discrepancies preserve a stronger contribution from the raw prediction A, rather than following a fixed iteration-dependent schedule. Together with the gradient-response analysis, these statistics provide complementary evidence that DITM progressively reshapes the composition of the supervised error signal as intended.

4) Robustness under Extremely Low-Light Conditions: To further examine the numerical behavior of DITM under challenging low-light conditions, we selected the darkest 10% of the LOLv1 training set according to the mean brightness of the low-light inputs, resulting in 49 samples. The same fixed subset was evaluated using early-, middle-, and late-stage checkpoints obtained from RetinexFormer and RetinexMamba trained with DITM. As shown in Table IV, $\mu _ { A }$ remained numerically well behaved across all settings. The minimum observed $\mu _ { A }$ ranged from 0.1694 to 0.2064, while the maximum observed $\beta = \frac { \mu _ { Y } } { \mu _ { A } }$ remained below 1.7028. Moreover, the mean $\beta$ progressively decreased from 1.1964 to 1.0510 for RetinexFormer and from 1.2250 to 1.0603 for RetinexMamba as training proceeded. No NaN or Inf values were observed. These results indicate that the luminance statistics and the resulting scaling ratio remain numerically stable within the evaluated challenging low-light subset. Meanwhile, the mean value of s decreased from 0.5739 to 0.1163 for RetinexFormer and from 0.6324 to 0.1354 for RetinexMamba, consistent with the intended transition from preserving the raw prediction at the early stage to increasingly incorporating the brightnessaligned prediction as the brightness discrepancy is reduced.

TABLE IV  
NUMERICAL BEHAVIOR OF DITM ON THE DARKEST 10% SAMPLES OF THE LOLV1 TRAINING SET AT DIFFERENT TRAINING STAGES.
<table><tr><td>Model</td><td>Stage</td><td> $\mathbf { m i n } ( \mu _ { A } )$ </td><td> $\mathbf { M e a n } ( \mu _ { A } )$ </td><td>Mean(β)</td><td>max(β)</td><td>Mean(s)</td></tr><tr><td rowspan="3">RetinexFormer with DITM</td><td>Early</td><td>0.1794</td><td>0.3634</td><td>1.1964</td><td>1.7028</td><td>0.5739</td></tr><tr><td>Middle</td><td>0.1896</td><td>0.4044</td><td>1.0787</td><td>1.4301</td><td>0.2617</td></tr><tr><td>Late</td><td>0.1990</td><td>0.4128</td><td>1.0510</td><td>1.2256</td><td>0.1163</td></tr><tr><td rowspan="3">RetinexMamba with DITM</td><td>Early</td><td>0.1694</td><td>0.3542</td><td>1.2250</td><td>1.6734</td><td>0.6324</td></tr><tr><td>Middle</td><td>0.1902</td><td>0.3923</td><td>1.1027</td><td>1.4034</td><td>0.3161</td></tr><tr><td>Late</td><td>0.2064</td><td>0.4103</td><td>1.0603</td><td>1.3587</td><td>0.1354</td></tr></table>

5) Performance under Spatially Non-Uniform Illumination: We further evaluate DITM on samples with pronounced spatially non-uniform illumination. To construct this evaluation subset, each low-light input X is first converted into a grayscale brightness map and then smoothed to capture its low-frequency illumination distribution. The illumination nonuniformity score is defined as

$$
U ( X ) = \frac { \mathrm { S t d } ( M ( X ) ) } { \mathrm { M e a n } ( M ( X ) ) } ,\tag{15}
$$

TABLE V  
PERFORMANCE COMPARISON ON THE TOP 20% SPATIALLY NON-UNIFORM SAMPLES FROM THE LOLV2-SYNTHETIC TEST SET. EACH SETTING WAS INDEPENDENTLY REPEATED THREE TIMES, AND THE MEAN AND STANDARD DEVIATION ARE REPORTED. VALUES IN PARENTHESES DENOTE THE MEAN CHANGE OF DITM OVER THE CORRESPONDING BASELINE. RED INDICATES IMPROVEMENT, WHILE GREEN INDICATES DEGRADATION.
<table><tr><td rowspan="2">Methods</td><td colspan="6">Evaluation Metrics</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DeepSSIM↑</td><td>IQA↑</td><td>IAA↑</td></tr><tr><td>RetinexFormer</td><td>27.012±0.607</td><td>0.9035±0.0025</td><td>0.073±0.002</td><td>0.8823±0.0005</td><td>3.014±0.009</td><td>2.176±0.004</td></tr><tr><td>RetinexFormer with DITM</td><td>27.673±0.462 (+0.661)</td><td>0.9121±0.0008 (+0.0086)</td><td>0.068±0.002 (-0.005)</td><td>0.8928±0.0015 (+0.0105)</td><td>3.070±0.040 (+0.056)</td><td>2.205±0.015 (+0.029)</td></tr><tr><td>RetinexMamba</td><td>26.264±0.577</td><td>0.9036±0.0003</td><td>0.072±0.001</td><td>0.8845±0.0024</td><td>3.108±0.016</td><td>2.256±0.009</td></tr><tr><td>RetinexMamba with DITM</td><td>26.596±0.051 (+0.332)</td><td>0.9124±0.0003 (+0.0088)</td><td>0.065±0.001 (-0.007)</td><td>0.8956±0.0018 (+0.0111)</td><td>3.231±0.022 (+0.123)</td><td>2.311±0.011 (+0.055)</td></tr></table>

TABLE VI  
EVALUATION OF DITM ON DIFFUSION-BASED LOW-LIGHT IMAGE ENHANCEMENT METHODS ON THE LOLV1 DATASET.
<table><tr><td>Methods</td><td>PSNR↑</td><td>SSIM↑</td><td>DeepSSIM↑</td><td>LPIPS↓</td><td>IQA↑</td><td>IAA↑</td></tr><tr><td>PyDiff PyDiff with DITM</td><td>21.958 22.004 (+0.046)</td><td>0.8969 0.8976 (+0.0007)</td><td>0.8283 0.8289 (+0.0006)</td><td>0.110 0.110 (+0.000)</td><td>4.158 4.160 (+0.002)</td><td>2.245 2.241 (-0.004)</td></tr><tr><td>Diff-Retinex++ Diff-Retinex++ with DITM</td><td>24.344 24.108 (-0.236)</td><td>0.8655 0.8672 (+0.0017)</td><td>0.8308 0.8326 (+0.0018)</td><td>0.105 0.103 (-0.002)</td><td>4.201 4.213 (+0.012)</td><td>2.282 2.281 (-0.001)</td></tr></table>

where M(X) denotes the Gaussian-smoothed brightness map of X obtained with a 51 × 51 Gaussian kernel. A larger U(X) indicates stronger relative spatial variation in the low-frequency brightness distribution. We rank the LOLv2- synthetic test images according to U(X) and select the top 20% as the spatially non-uniform subset. This score is used only for subset construction rather than as an image-quality metric.

As shown in Table V, the selected spatially non-uniform subset exhibits different performance characteristics compared with the full LOLv2-synthetic test set, with relatively lower SSIM and higher LPIPS values, indicating the influence of spatial illumination variation on image restoration. Nevertheless, DITM consistently improves both RetinexFormer and RetinexMamba on this subset. Across three independent runs, DITM achieves consistent improvements in SSIM, LPIPS, DeepSSIM, IQA, and IAA, while also providing positive average PSNR gains. These results suggest that although DITM is guided by global luminance statistics and does not explicitly model spatial illumination distributions, the potential limitation does not significantly affect its effectiveness under the evaluated spatially non-uniform illumination conditions.

6) Training-Time Overhead: We further evaluate the efficiency of DITM on seven LLIE backbones, covering CNN-, Transformer-, and Mamba-based architectures. As reported in Table VII, DITM only brings marginal training-time overhead, with the largest increase being 2.64%. The peak GPU memory remains almost unchanged. Since DITM is removed during inference, the deployed models have the same parameters, FLOPs, and latency as the original backbones.

## E. Applicability Scope and Limitations of DITM

DITM is designed for paired supervised LLIE models that produce an explicit image-space prediction and directly optimize it against a corresponding normal-light reference. Within this setting, DITM is independent of the specific backbone architecture, as demonstrated by its consistent effectiveness across CNN-, Transformer-, and Mamba-based feed-forward models. Its applicability, however, depends on both the availability of paired supervision and the role of image-space reconstruction in the underlying optimization formulation.

TABLE VII  
TRAINING-TIME OVERHEAD OF DITM ON THE ADOPTED LLIE BACKBONES IN OUR EXPERIMENTS.
<table><tr><td>Model</td><td>Training Time / Iter Peak GPU Memory</td><td></td></tr><tr><td>RetinexNet</td><td>326.458 ms</td><td>2.103 GB</td></tr><tr><td>RetinexNet with DITM</td><td>328.990 ms (+0.78%)</td><td>2.104 GB</td></tr><tr><td>MIRNet</td><td>519.327 ms</td><td>20.040 GB</td></tr><tr><td>MIRNet with DITM</td><td>520.944 ms (+0.31%)</td><td>20.041 GB</td></tr><tr><td>CID-Net</td><td>210.626 ms</td><td>14.350 GB</td></tr><tr><td>CID-Net with DITM</td><td>213.715 ms (+1.47%)</td><td>14.375 GB</td></tr><tr><td>LLFormer</td><td>425.855 ms</td><td>20.060 GB</td></tr><tr><td>LLFormer with DITM</td><td>427.666 ms (+0.43%)</td><td>20.063 GB</td></tr><tr><td>RetinexFormer</td><td>48.034 ms</td><td>3.008 GB</td></tr><tr><td>RetinexFormer with DITM</td><td>49.301 ms (+2.64%)</td><td>3.008 GB</td></tr><tr><td>MambaLLIE</td><td>134.341 ms</td><td>4.638 GB</td></tr><tr><td>MambaLLIE with DITM</td><td>135.370 ms (+0.77%)</td><td>4.641 GB</td></tr><tr><td>RetinexMamba</td><td>266.436 ms</td><td></td></tr><tr><td>RetinexMamba with DITM 268.306 ms (+0.70%)</td><td></td><td>9.871 GB 9.873 GB</td></tr></table>

Requirement for Paired Supervision. Both the mean brightness-aligned prediction B and the adaptive interpolation factor s require the corresponding normal-light ground-truth image Y. Therefore, DITM in its current formulation is not directly applicable to unpaired, unsupervised, or zero-reference LLIE training, where such a reference is unavailable during optimization. This limitation concerns the training regime rather than the testing data. A model trained with DITM under paired supervision can still be directly evaluated on unpaired real-world test sets, as demonstrated by our crossdataset evaluation.

Dependence on the Restoration Formulation. DITM is most effective when the reshaped image prediction directly participates in optimizing the parameters responsible for the final restoration. To examine this condition, we further evaluated DITM on two diffusion-based LLIE methods, PyDiff [23] and Diff-Retinex++ [72], on the LOLv1 dataset. For each method, DITM was incorporated into the image-space reconstruction branch as a training-only prediction reshaping operation, while all other loss terms, loss weights, and training settings were kept unchanged. The quantitative results are reported in Table VI.

TABLE VIII  
GRADIENT RESPONSE ANALYSIS OF DITM ON DIFFUSION-BASED LLIE MODELS. THE RESULTS ARE AVERAGED OVER SAMPLED IMAGES. $\rho _ { g }$ AND cos DENOTE THE GRADIENT NORM RATIO AND COSINE SIMILARITY BETWEEN BASELINE AND DITM OPTIMIZATION, RESPECTIVELY.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Parameter Group</td><td rowspan=1 colspan=1>Early Stage $\rho _ { g }$   COS</td><td rowspan=1 colspan=1>Late Stage $\rho _ { g }$   COS</td></tr><tr><td rowspan=1 colspan=1>PyDiff</td><td rowspan=1 colspan=1>|Full ModelDiffusion Denoising ModuleImage Restoration Module</td><td rowspan=1 colspan=1>0.8700.5431.0001.0000.8600.516</td><td rowspan=1 colspan=1>0.8620.6161.0001.0000.8410.585</td></tr><tr><td rowspan=1 colspan=1>Diff-Retinex++</td><td rowspan=1 colspan=1>Full ModelDiffusion Denoising ModuleImage Restoration Module</td><td rowspan=1 colspan=1>0.9870.9811.0001.0000.976 0.962</td><td rowspan=1 colspan=1>|0.9780.9761.0001.0000.974 0.971</td></tr></table>

As shown in Table VI, unlike the consistent improvements observed on supervised feed-forward LLIE backbones, DITM provides only marginal and inconsistent gains on the two investigated diffusion-based models. To further analyze this phenomenon, we compare the gradient responses induced by the original and DITM-modified objectives under the same model state. Specifically, for a parameter subset $\theta _ { m }$ , we compute the gradient norm ratio

$$
\rho _ { g } ( \theta _ { m } ) = \frac { \| \nabla _ { \theta _ { m } } \mathcal { L } _ { \mathrm { D I T M } } \| _ { 2 } } { \| \nabla _ { \theta _ { m } } \mathcal { L } _ { \mathrm { b a s e } } \| _ { 2 } } ,\tag{16}
$$

and the cosine similarity

$$
\cos ( \theta _ { m } ) = \frac { \nabla _ { \theta _ { m } } \mathcal { L } _ { \mathrm { b a s e } } \cdot \nabla _ { \theta _ { m } } \mathcal { L } _ { \mathrm { D I T M } } } { \Vert \nabla _ { \theta _ { m } } \mathcal { L } _ { \mathrm { b a s e } } \Vert _ { 2 } \Vert \nabla _ { \theta _ { m } } \mathcal { L } _ { \mathrm { D I T M } } \Vert _ { 2 } } ,\tag{17}
$$

where $\mathcal { L } _ { \mathrm { b a s e } }$ and $\mathcal { L } _ { \mathrm { D I T M } }$ denote the original and DITMmodified objectives, respectively. Here, $\theta _ { m }$ denotes a specific parameter group, including the full model, diffusion denoising module, or image restoration module. The results are reported in Table VIII.

These gradient observations provide a quantitative explanation for the limited improvements observed on the two diffusion-based LLIE models. Unlike supervised feed-forward models, where the reshaped prediction directly supervises the backbone responsible for producing the restored image, the investigated diffusion models employ separate optimization pathways for diffusion denoising and image-space reconstruction. As shown in Table VIII, the gradients of the diffusion denoising modules remain nearly unchanged after introducing DITM, indicating that the reshaped reconstruction supervision does not establish an effective optimization connection with the parameters responsible for the diffusion denoising process. Therefore, the effectiveness of DITM depends on whether the reshaped reconstruction supervision can influence the parameters governing the final restoration process.

These observations indicate that DITM should be regarded as a backbone-agnostic prediction reshaping strategy within paired supervised feed-forward LLIE, rather than a universally applicable mechanism for all LLIE paradigms. Diffusion models without explicit image-space reconstruction objectives [24], [27] cannot directly adopt the current formulation. For diffusion-based models containing image-space reconstruction branches [23], [72], the benefit of DITM depends on the optimization connection between the reshaped supervision and the diffusion restoration process.

## V. CONCLUSION

We presented a lightweight prediction-reshaping strategy called DITM for supervised feed-forward LLIE. Without modifying the backbone architecture or introducing additional inference-time cost, DITM dynamically interpolates the raw model prediction with a brightness-aligned prediction during training. This operation reduces the relative dominance of global luminance discrepancy in the supervised prediction, allowing residual artifact-, structure-, and detail-related errors to become more involved in subsequent optimization. Extensive experiments on multiple benchmark datasets demonstrate that DITM generally provides stable quantitative and qualitative improvements across CNN-, Transformer-, and Mamba-based LLIE backbones. With its lightweight training-time design and zero-overhead inference property, DITM offers a simple way to improve supervised feed-forward LLIE models.

DITM is currently designed for paired supervised LLIE, as it relies on the ground-truth mean brightness during training. Future work will explore reliable pseudo brightness targets or learned brightness references to extend the predictionreshaping strategy to unpaired, self-supervised, and zeroreference LLIE. Since the current global mean-brightness alignment does not explicitly model spatially varying illumination, future work will explore spatially adaptive illumination guidance to better handle non-uniform illumination and local lighting variations. In addition, it is important to investigate prediction reshaping in diffusion-based restoration frameworks where image-space reconstruction supervision remains gradient-connected to the denoising network, particularly its appropriate placement along the denoising trajectory.

Acknowledgments. This work was supported by the National Natural Science Foundation of China under Grant XXXXXXX.

## REFERENCES

[1] Q. Zhao, G. Li, B. He, and R. Shen, “Deep learning for low-light vision: A comprehensive survey,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 9, pp. 15 685–15 705, 2025.

[2] C. Li, C. Guo, L. Han, J. Jiang, M.-M. Cheng, J. Gu, and C. C. Loy, “Low-light image and video enhancement using deep learning: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 9396–9416, 2021.

[3] J. Liu, D. Xu, W. Yang, M. Fan, and H. Huang, “Benchmarking low-light image enhancement and beyond,” International Journal of Computer Vision, pp. 1153–1184, 2021.

[4] H. Su, L. Yu, and C. Jung, “Joint contrast enhancement and noise reduction of low light images via jnd transform,” IEEE Transactions on Multimedia, vol. 24, pp. 17–32, 2022.

[5] X. Fu, D. Zeng, Y. Huang, X.-P. Zhang, and X. Ding, “A weighted variational model for simultaneous reflectance and illumination estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2016, pp. 2782–2790.

[6] X. Guo, Y. Li, and H. Ling, “Lime: Low-light image enhancement via illumination map estimation,” IEEE Transactions on Image Processing, pp. 982–993, 2017.

[7] D. Feijoo, J. C. Benito, A. Garcia, and M. V. Conde, “Darkir: Robust low-light image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 10 879– 10 889.

[8] X. Xu, R. Wang, C.-W. Fu, and J. Jia, “Snr-aware low-light image enhancement,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 17 714–17 724.

[9] J. Pan, X. Liu, Y. Bai, D. Zhai, J. Jiang, and D. Zhao, “Illuminationaware low-light image enhancement with transformer and auto-knee curve,” ACM Transactions on Multimedia Computing, Communications and Applications, pp. 1–23, 2024.

[10] Y. Jiang, X. Gong, D. Liu, Y. Cheng, C. Fang, X. Shen, J. Yang, P. Zhou, and Z. Wang, “Enlightengan: Deep light enhancement without paired supervision,” IEEE Transactions on Image Processing, pp. 2340–2349, 2021.

[11] L. Ma, T. Ma, R. Liu, X. Fan, and Z. Luo, “Toward fast, flexible, and robust low-light image enhancement,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5637–5646.

[12] C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, and R. Cong, “Zeroreference deep curve estimation for low-light image enhancement,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 1780–1789.

[13] F. Lv, Y. Li, and F. Lu, “Attention guided low-light image enhancement with a large scale low-light simulation dataset,” International Journal of Computer Vision, pp. 2175–2193, 2021.

[14] C. Wei, W. Wang, W. Yang, and J. Liu, “Deep retinex decomposition for low-light enhancement,” in Proceedings of British Machine Vision Conference, 2018.

[15] R. Liu, L. Ma, J. Zhang, X. Fan, and Z. Luo, “Retinex-inspired unrolling with cooperative prior architecture search for low-light image enhancement,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 10 561–10 570.

[16] Y. Zhang, J. Zhang, and X. Guo, “Kindling the darkness: A practical low-light image enhancer,” in Proceedings of the 27th ACM International Conference on Multimedia, 2019, pp. 1632–1640.

[17] Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, and Y. Zhang, “Retinexformer: One-stage retinex-based transformer for low-light image enhancement,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 12 504–12 513.

[18] T. Wang, K. Zhang, T. Shen, W. Luo, B. Stenger, and T. Lu, “Ultra-highdefinition low-light image enhancement: A benchmark and transformerbased method,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2023, pp. 2654–2662.

[19] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in Proceedings of the First Conference on Language Modeling, 2024.

[20] J. Bai, Y. Yin, Q. He, Y. Li, and X. Zhang, “Retinexmamba: Retinexbased mamba for low-light image enhancement,” in Proceedings of International Conference on Neural Information Processing, 2024, pp. 427–442.

[21] W. Zou, H. Gao, W. Yang, and T. Liu, “Wave-mamba: Wavelet state space model for ultra-high-definition low-light image enhancement,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 1534–1543.

[22] J. Weng, Z. Yan, Y. Tai, J. Qian, J. Yang, and J. Li, “Mamballie: Implicit retinex-aware low light enhancement with global-then-local state space,” in Proceedings of Advances in Neural Information Processing Systems, 2024, pp. 27 440–27 462.

[23] D. Zhou, Z. Yang, and Y. Yang, “Pyramid diffusion models for low-light image enhancement,” in Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 2023, pp. 1795–1803.

[24] J. Hou, Z. Zhu, J. Hou, H. Liu, H. Zeng, and H. Yuan, “Global structure-aware diffusion process for low-light image enhancement,” in Proceedings of Advances in Neural Information Processing Systems, 2023, pp. 79 734–79 747.

[25] H. Jiang, A. Luo, H. Fan, S. Han, and S. Liu, “Low-light image enhancement with wavelet-based diffusion models,” ACM Transactions on Graphics, pp. 1–14, 2023.

[26] X. Zeng, L. Zhu, W. Yang, H. Leung, S. Wang, and S. Kwong, “Lowlight image enhancement via diffusion models with semantic priors of any region,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 36, no. 3, pp. 3754–3767, 2026.

[27] K. Shang, M. Shao, C. Wang, Y. Cheng, and S. Wang, “Multi-domain multi-scale diffusion model for low-light image enhancement,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 38, no. 5, 2024, pp. 4722–4730.

[28] C. Wang, H. Fan, H. Yang, S. Karimi, L. Yao, and Y. Yang, “Adapting text-to-image generation with feature difference instruction for generic image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 23 539–23 550.

[29] Q. Yan, Y. Feng, C. Zhang, G. Pang, K. Shi, P. Wu, W. Dong, J. Sun, and Y. Zhang, “Hvi: A new color space for low-light image enhancement,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 5678–5687.

[30] X. Xu, R. Wang, and J. Lu, “Low-light image enhancement via structure modeling and guidance,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 9893–9903.

[31] G. Liang, K. Chen, H. Li, Y. Lu, and L. Wang, “Towards robust eventguided low-light image enhancement: a large-scale real-world eventimage dataset and novel approach,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 23– 33.

[32] Z. Zhao, L. Gao, P. Gao, and J. Qin, “Dp-retinex: Dual-prior guided low-light image enhancement with yuv-domain reflectance-illumination decomposition,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 36, no. 5, pp. 6758–6769, 2026.

[33] C. Wang, H. Wu, and Z. Jin, “Fourllie: Boosting low-light image enhancement by fourier frequency information,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 7459– 7469.

[34] J. Zhang, X. Wang, Y. Li, and W. Wang, “Prnet: low-light image enhancement based on fourier transform,” IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1–14, 2025.

[35] T. Zhang, P. Liu, M. Zhao, and H. Lv, “Dmfourllie: Dual-stage and multi-branch fourier network for low-light image enhancement,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 7434–7443.

[36] Z. He, W. Ran, S. Liu, K. Li, J. Lu, C. Xie, Y. Liu, and H. Lu, “Low-light image enhancement with multi-scale attention and frequency-domain optimization,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 4, pp. 2861–2875, 2024.

[37] W. Dong, Y. Min, H. Zhou, and J. Chen, “Towards scale-aware low-light enhancement via structure-guided transformer design,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 1460–1470.

[38] H. Jiang, A. Luo, X. Liu, S. Han, and S. Liu, “Lightendiffusion: Unsupervised low-light image enhancement with latent-retinex diffusion models,” in Proceedings of European Conference on Computer Vision, 2024, pp. 161–179.

[39] W. Wang, H. Yang, J. Fu, and J. Liu, “Zero-reference low-light enhancement via physical quadruple priors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 057–26 066.

[40] X. Li, M. Liu, and Q. Ling, “Pixel-wise gamma correction mapping for low-light image enhancement,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 2, pp. 681–694, 2024.

[41] L. Zhu, W. Yang, B. Chen, H. Zhu, X. Meng, and S. Wang, “Temporally consistent enhancement of low-light videos via spatial-temporal compatible learning,” International Journal of Computer Vision, vol. 132, no. 10, pp. 4703–4723, 2024.

[42] J. Ye, C. Qiu, and Z. Zhang, “Snr-prior guided trajectory-aware transformer for low-light video enhancement,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 3, pp. 1873–1885, 2024.

[43] K. Chen, G. Liang, Y. Lu, H. Li, and L. Wang, “Evlight++: Lowlight video enhancement with an event camera: A large-scale realworld dataset, novel method, and more,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 2, pp. 1608–1625, 2026.

[44] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proceedings ofthe Conference on Computer Vision and Pattern Recognition, 2018, pp. 7132–7141.

[45] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “Cbam: Convolutional block attention module,” in Proceedings ofEuropean Conference on Computer Vision, 2018, pp. 3–19.

[46] Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu, “Eca-net: Efficient channel attention for deep convolutional neural networks,” in

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11 534–11 542.

[47] Y. Rao, W. Zhao, Y. Tang, J. Zhou, S. N. Lim, and J. Lu, “Hornet: Efficient high-order spatial interactions with recursive gated convolutions,” in Advances in Neural Information Processing Systems, 2022, pp. 10 353–10 366.

[48] K. A. Hashmi, G. Kallempudi, D. Stricker, and M. Z. Afzal, “Featenhancer: Enhancing hierarchical features for object detection and beyond under low-light vision,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 6725–6735.

[49] Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, and L. Van Gool, “Denoising diffusion models for plug-and-play image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1219–1229.

[50] Y. Zhang, B. Teng, D. Yang, Z. Chen, H. Ma, G. Li, and W. Ding, “Learning a single convolutional layer model for low light image enhancement,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 7, pp. 5995–6008, 2024.

[51] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 7464–7475.

[52] M. A. Nugroho, S. Woo, S. Lee, J. Park, Y. Wang, D. Kim, and C. Kim, “Flow-assisted motion learning network for weakly-supervised group activity recognition,” in Proceedings of European Conference on Computer Vision, 2024, pp. 73–91.

[53] Q. Yang, Q. Cheng, H. Yue, L. Zhang, Y. Liu, and J. Yang, “Learning to see low-light images via feature domain adaptation,” IEEE Transactions on Image Processing, vol. 34, pp. 2680–2693, 2025.

[54] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu, “Deeply-Supervised Nets,” in Proceedings of the International Conference on Artificial Intelligence and Statistics, 2015, pp. 562–570.

[55] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum learning,” in Proceedings of the International Conference on Machine Learning (ICML), 2009.

[56] Y. Luo, X. Chen, J. Ling, C. Huang, W. Zhou, and G. Yue, “Unsupervised low-light image enhancement with self-paced learning,” IEEE Transactions on Multimedia, vol. 27, pp. 1808–1820, 2025.

[57] J. Kim, G. Park, J. Bae, and S. Jung, “Intrinsic decomposition-based curriculum learning for low-light image enhancement,” IEEE Signal Processing Letters, vol. 32, pp. 4019–4023, 2025.

[58] T. Lin, P. Goyal, R. B. Girshick, K. He, and P. Dollar, “Focal loss for´ dense object detection,” in Proceedings of the International Conference on Computer Vision (ICCV), 2017.

[59] Z. Chen, V. Badrinarayanan, C. Lee, and A. Rabinovich, “Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks,” in Proceedings of the International Conference on Machine Learning (ICML), 2018, pp. 793–802.

[60] L. Jiang, B. Dai, W. Wu, and C. C. Loy, “Focal frequency loss for image reconstruction and synthesis,” in Proceedings of the International Conference on Computer Vision (ICCV), 2021.

[61] W. Yang, W. Wang, H. Huang, S. Wang, and J. Liu, “Sparse gradient regularized deep retinex network for robust low-light image enhancement,” IEEE Transactions on Image Processing, pp. 2072–2086, 2021.

[62] V. Vonikakis, R. Kouskouridas, and A. Gasteratos, “On the evaluation of illumination compensation algorithms,” Multimedia Tools and Applications, pp. 9211–9231, 2018.

[63] S. Wang, J. Zheng, H.-M. Hu, and B. Li, “Naturalness preserved enhancement algorithm for non-uniform illumination images,” IEEE Transactions on Image Processing, pp. 3538–3548, 2013.

[64] K. Ma, K. Zeng, and Z. Wang, “Perceptual quality assessment for multiexposure image fusion,” IEEE Transactions on Image Processing, pp. 3345–3356, 2015.

[65] C. Lee, C. Lee, and C.-S. Kim, “Contrast enhancement based on layered difference representation of 2d histograms,” IEEE Transactions on Image Processing, pp. 5372–5384, 2013.

[66] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, “Learning enriched features for real image restoration and enhancement,” in Proceedings of European Conference on Computer Vision, 2020, pp. 492–511.

[67] K. Zhang, W. Chen, T. Zhao, and Z. Wang, “Structural similarity in deep features: Unified image quality assessment robust to geometrically disparate reference,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 3, pp. 2581–2595, 2026.

[68] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,”

in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 586–595.

[69] F. Zhang, S. B. Rangrej, T. Aumentado-Armstrong, A. Fazly, and A. Levinshtein, “Augmenting perceptual super-resolution via image quality predictors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 2311–2322.

[70] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, “Musiq: Multiscale image quality transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5148–5157.

[71] H. Wu, Z. Zhang, W. Zhang, C. Chen, L. Liao, C. Li, Y. Gao, A. Wang, E. Zhang, W. Sun, Q. Yan, X. Min, G.-t. Zhai, and W. Lin, “Qalign: Teaching lmms for visual scoring via discrete text-defined levels,” in Proceedings of the International Conference on Machine Learning, 2024, pp. 54 015–54 029.

[72] X. Yi, H. Xu, H. Zhang, L. Tang, and J. Ma, “Diff-retinex++: Retinexdriven reinforced diffusion model for low-light image enhancement,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 8, pp. 6823–6841, 2025.

![](images/7ab1422cd48b7ae957ebf2852ec630e3839f0fd59a2b1aefe5e401ecbe625378.jpg)  
Peirui Zhu received the B.S. degree from Hefei University of Technology (HFUT) in 2024, where he is currently pursuing the M.S. degree with the School of Computer Science and Information Engineering. His research interests include low-light image enhancement.

![](images/bde9e2db0c600a83ee567114e4078500f18cb098e7736d06091a8630e66307d9.jpg)

Shijie Hao is a Professor with the School of Computer Science and Information Engineering, Hefei University of Technology (HFUT). He received the Ph.D. degree from HFUT in 2012. He was a visiting scholar with the University of North Carolina at Chapel Hill (UNC) from 2015 to 2016. His research interests include low-level image processing.

![](images/db303db4d66e68d6c86c86b4d221ddc5c2450b44afc6e6c8c343f20494fcb969.jpg)

Jingxi Liao received the B.S. and M.S. degrees from the School of Computer Science and Information Engineering, Hefei University of Technology (HFUT). His research interests include low-light image enhancement.

![](images/6d6e50fc8dfcf5e1bc29a6b4e1bc6feb57751178384045ba846710ed2848b545.jpg)

Richang Hong received the Ph.D. degree from the University of Science and Technology of China (USTC) in 2008. From 2008 to 2010, he was a Research Fellow with the National University of Singapore. He is currently a Professor with HFUT. He has authored over 200 publications in multimedia content analysis and related areas. He serves as an Associate Editor for IEEE TCSVT, IEEE TBD, IEEE TMM, and ACM TOMM.

![](images/5cf03a3bc8bfdfb5614eb0c04afa686534afc05ed224948a48daa7621bee2cbf.jpg)

Meng Wang received the Ph.D. degree from the University of Science and Technology of China (USTC) in 2008. He is currently a Professor with HFUT. His research interests include multimedia, computer vision, and pattern recognition. He has authored over 200 papers in these areas. He is a Fellow of the IEEE and serves/served as an AE for IEEE TPAMI, IEEE TCSVT, IEEE TKDE, IEEE TNNLS, and IEEE TMM.