# Diffusion Meta-Prompting and Steering for Generalizable Foundation Model Adaptation

Deepak Sridhar<sup>1</sup> Yi Li<sup>2</sup> Kartikeya Bhardwaj<sup>2</sup> Shuangjun Liu<sup>2</sup> Taotao Jing<sup>2</sup>

Yuan Li<sup>2</sup> Shuai Zhang<sup>2</sup> Jiancheng Lyu<sup>2</sup> Dashan Gao<sup>2</sup> Nuno Vasconcelos<sup>1</sup>

<sup>1</sup>University of California, San Diego <sup>2</sup>Qualcomm AI Research<sup>∗</sup> desridha@ucsd.edu, nvasconcelos@ucsd.edu

## Abstract

Prompt learning is a popular method for adapting foundation models, but learned prompts are typically task-specific and fail to generalize to new classes, domains, or compositions of tasks. In this paper, we introduce a Diffusion Meta-Prompt (DMP) model , a framework that models the distribution of learned prompts using diffusion models. Given a repository of previously learned prompts, DMP is trained and sampled without access to the original task examples or task losses, and synthesizes new prompts conditioned on natural language task descriptions. To improve the sampling stability, we introduce a test-time steering strategy for DMP, which uses the best training-selected prompt in the repository as a latent anchor during diffusion sampling, without retraining the DMP or accessing test classes. DMP improves generalization across classification, retrieval and text-to-image generation tasks, supports concept composition and negative prompting without explicit training. It reduces storage and inference costs by over 90% compared to prompt retrieval methods. For composite classification, DMP achieves upto 2.0% average gain over prior meta-learning methods across 55 pairs of datasets with gains as high as 8.5% on specific pairs such as Eurosat and Flowers. DMP also enhances cross-task generalization with ∼2-9% improvement for hierarchical classification task. We further provide a theoretical guarantee bounding the expected task loss of prompts sampled from a DMP. Code is available here: DMP

## 1 Introduction

Foundation models (Rombach et al., 2022; Radford et al., 2021) generalize to diverse tasks owing to their scale, and can be customized to downstream tasks with parameter-efficient methods (Hu et al., 2022; Zhang and Agrawala, 2023) such as prompt learning (PL) (Zhou et al., 2022b). In vision-language models, prompts are learned in the visual (Jia et al., 2022), textual (Zhou et al., 2022a), or both spaces (Roy and Etemad, 2024; Hao et al., 2025) to improve fine-grained (Helber et al., 2018) and taxonomic classification (Wu et al., 2024), class discrimination (Zhou et al., 2022b,a), and robustness to domain shift (Ge et al., 2022). In generative models, prompts customize synthesis to specific concepts, people, or objects (Gal et al., 2022; Ruiz et al., 2023), or control the strength of attributes such as age, emotion, or style (Sridhar and Vasconcelos, 2024).

Despite their power, learned prompts have key limitations (Figure 1, left). They are inherently taskspecific, requiring task-specific data and losses: classification prompts generalize poorly beyond the few-shot base classes and must be retuned for each label set, while generative prompts for attributes such as age or smiling must be learned separately per entity or concept. Since different applications could share the same prompts, practical systems rely on large repositories of prompts or adaptation weights, such as HuggingFace or Civitai (Civitai, 2023), where prompts must be stored, searched, and manually composed (Luo et al., 2024), limiting reuse.

![](images/b630da4d65fe0c953d707938893ce7203978e0ce0563eb6f9c9c0217a1627021.jpg)  
Figure 1: DMP versus existing approaches: DMP models provide a natural language interface for the adaptation of foundation models, while eliminating the complexities of large prompt repositories, offering better generalization and composition with a minimal runtime overhead due to the feasibility of using small DMP models and fast sampling.

Table 1: Conceptual differences between DMP and prior meta-learning methods.
<table><tr><td>Method</td><td>Task Data/</td><td>Supports Loss Free Multiple Tasks across Tasks</td><td>Learns</td><td>Core Idea</td></tr><tr><td>Bayesian Prompt Learning (BPL)</td><td>x</td><td>X</td><td>X</td><td>Bayesian uncertainty modeling over task-specific prompts.</td></tr><tr><td>Prompt Learning via Meta-Regularization (ProMetaR)</td><td>X</td><td>V</td><td>V</td><td>Meta-regularization to improve prompt generalization across tasks; requires training data.</td></tr><tr><td>Gradient-Regulated Meta-Prompt (GRAM)</td><td>X</td><td>L</td><td>X</td><td>Meta-learned prompt init + gradient regulator for few-shot cross-domain generalization.</td></tr><tr><td>PRewrite (Prompt Rewriting w/ RL)</td><td>X</td><td>7</td><td>X</td><td>LLM-based prompt rewriter trained with RL to improve downstream task performance.</td></tr><tr><td>DMP (Ours)</td><td>L</td><td>L</td><td>L</td><td>Learns a prompt distribution; one-shot guided sampling, composition, negative prompts.</td></tr></table>

In this work, we propose a different perspective: instead of treating prompts as isolated optimization outcomes, they should be considered as samples from a structured distribution. Across tasks and domains, learned prompts exhibit regularities that admit smooth interpolation, and support meaningful composition and negation. Capturing this structure would enable the reuse of prior prompt learning effort and improve generalization to new tasks. Rather than learning one set of prompts at a time, there is a need for meta-prompting techniques capable of learning to generate prompts across many tasks and models. However, to be scalable, the learning should be feasible without revisiting the original task data or losses. Instead, one would like to rely on the large prompt repositories that are already available, whose prompts were themselves obtained by standard prompt optimization on their respective source tasks. This rules out classical meta-learning techniques (Finn et al., 2017; Nichol et al., 2018; Li et al., 2017; Andrychowicz et al., 2016), based on a double optimization loop, where a prompt generator is trained jointly with the prompted models across several tasks.

To accomplish this goal we propose an approach, Diffusion Meta-Prompting (DMP), where a generative model is learned over prompt embeddings, using diffusion models. A DMP is trained on a repository of previously learned prompts, and learns to synthesize new prompts conditioned on natural language task descriptions. As illustrated on the right of Figure 1, at inference, prompts are simply sampled with the DMP model, enabling one-shot adaptation of foundation models to unseen classes, composite tasks, and novel concept combinations. Table 1 compares DMP against prior meta-learning approaches, showing its uniqueness in terms of not requiring access to the original task data or losses once a prompt repository is available, while learning joint prompt structure over tasks, and requiring a minimal amount of prompt storage. The combination of these properties makes DMP much more scalable in the number of tasks and capable of prompt interpolation and generalization.

Although the ability of diffusion models to synthesize prompts is not surprising, different diffusion trajectories can produce prompts with different downstream generalization behavior. To mitigate these effects, we introduce a simple test-time steering mechanism, which allows DMP to reuse the prompt repository at inference time without any additional training. For each dataset, we select the prompt seed with the highest training accuracy, and use this as an anchor during DDIM sampling. At each denoising step, the predicted clean prompt is interpolated with this anchor. This biases the sampling trajectory toward a known low-risk region of the learned prompt manifold while still allowing the diffusion model to synthesize new prompts conditioned on the task specification. Importantly, the anchor is selected once, using the source-task training performance already associated with the repository, and does not use target-task examples or novel test classes. This allows task generalization.

As shown in the rightmost panel of Figure 1, DMPs eliminate the complexities of dealing with prompt repositories, offering better generalization and compositionality with a minimal runtime overhead. For example, given a subject name (“Jennifer Aniston”) and attributes (“hair” and “age”), a DMP can generate a set of prompts to personalize a foundation diffusion model like Stable Diffusion XL (SDXL) to the generation of images with this combination of concepts.

We choose diffusion as the generative model for prompts since it offers better generalization than alternatives like autoregressive methods (Radford et al., 2019) (shown later in Table 19, Appendix A.6.1). This is demonstrated in two ways. First, we show that DMPs learn to produce a range of sophisticated prompt operations, such as concept composition, guided sampling, novel subject generation, editing and negative prompting without explicit training. Second, we show that DMPs can learn multiple tasks, by showing that a single model can be trained to produce prompts for both subject personalization and slider attributes. This replaces the combinatorial complexities of searching separate prompt repositories for the two (and potentially more) tasks into a single DMP model. DMPs are also shown applicable to a diversity of fundamentally different tasks, ranging from image synthesis to composite classification. For the latter, we show that learning the distribution of prompts across datasets and label sets generalizes to unseen classes better than prior meta-learning approaches and the standard approach of individual prompt learning per dataset. The fact that this happens even though the DMP model is trained on the individual prompts ofthe baseline approach shows that there is structure in prompt space, which DMPs learn for improved generalization.

Beyond prompt embeddings, the same principle can be applied to foundation-model representations themselves. We therefore evaluate DMP in a sparse multi-view retrieval setting, where a query view of an object must retrieve the remaining views of the same object from a gallery in which many views are missing. In this setting, DMP is trained on objects disjoint from the gallery and synthesizes embeddings for missing views. Retrieval is then performed with an aggregated gallery containing both observed and synthesized embeddings. We demonstrate that DMP can also model such structured distributions beyond learned prompts, and can improve retrieval for a different representation space and downstream objective without accessing the gallery objects during training (Table 7).

## Overall, the paper makes the following key contributions

• We introduce Diffusion Meta-Prompting, a meta-learning framework that treats learned prompts as samples from a shared distribution, which is learned with diffusion models. Unlike prompt learning, retrieval, or refinement methods, DMP is trained once on a repository of previously learned prompts and synthesizes new task-specific prompts at inference without revisiting the original task data or task losses, and without additional prompt optimization.

• We introduce a test-time steering mechanism at inference without any additional training that produces prompts with better generalization (upto 2.0%) over standard sampling.

• We provide a theoretical guarantee bounding the expected task loss of prompts sampled from a DMP in terms of the empirical loss of the prompt repository and the denoising error of the diffusion model.

• Through extensive experiments, we show that DMP improves generalization in the most challenging regimes where prompt learning typically fails, including cross-task, and composite classification settings (up to 2.0% accuracy gains), while also reducing storage and inference costs by over 90% compared to prompt retrieval approaches.

• We also demonstrate that DMP is representation-agnostic by applying it to sparse multi-view retrieval experiment which improves retrieval by +3.44 mAP over direct DINOv2 retrieval.

## 2 Related work

Foundation models. We focus on two classes of foundation models: vision-language representation models (Radford et al., 2021) and text-to-image (T2I) generation models (Rombach et al., 2022). Contrastively trained vision-language models, such as CLIP, are widely used for open-set classification, detection, and segmentation, with prompting techniques enabling adaptation to fine-grained tasks (Zhou et al., 2022b). In T2I models, personalization methods like Textual Inversion (Gal et al., 2022) allow generation of images of custom concepts using only a few examples. Prompt Sliders (Sridhar and Vasconcelos, 2024) and (Baumann et al., 2024) learn concepts or attributes in textual space, either globally or locally. In this work, we propose to unify the prompt learning for foundation models by training a diffusion model to synthesize the prompts conditioned on natural language across tasks.

Prompt learning. Prompt tuning learns textual (Zhou et al., 2022a), visual (Jia et al., 2022), or multimodal (Khattak et al., 2023; Yang et al., 2024; Li et al., 2025b,a) prompts for CLIP (Radford et al., 2021). Textual prompt learning, pioneered by CoOp (Zhou et al., 2022b) and CoCoOp (Zhou et al., 2022a), fine-tunes a CLIP model for few-shot transfer by optimizing continuous prompt vectors in the language branch. Visual prompt tuning (Jia et al., 2022) introduces task-specific learnable prompts in the visual encoder while keeping the backbone fixed. Bayesian prompt learning (Derakhshani et al., 2023) formulated prompt learning as a variational inference problem and demonstrated its ability to generalize to unseen classes at the expense of base class accuracy. Multi-modal prompt learning (Roy and Etemad, 2024; Hao et al., 2025) optimizes prompts in both vision and language encoders to improve cross-modal alignment. Table 1 and 10 in Appendix shows how DMP differs from prior prompt-learning approaches across key axes. Unlike prior approaches, which refine prompts for a single task or single example, DMP uniquely enables multi-task prompt learning and generation, without access to task specific data or model training, simply requiring access to a repository of pre-computed prompts.

Meta learning. Meta- learning methods aim to amortize learning across tasks, enabling rapid adaptation to new problems (Ha et al., 2017; Hospedales et al., 2021). These methods typically have a double-loop optimization, e,g, iterating between leaning of the prompt generator and of the downstream model. Unlike DMP, this is usually not scalable to large numbers of tasks. Recently, diffusion models have been explored as generative priors for weights and representations (Zhang et al., 2024; Du et al., 2024). (Du et al., 2024) proposed a diffusion model for refining CLIP prompts for classification. It is trained on a specific example and classification task to improve CLIP prompts for that task. Unlike this type of refinement approach, DMP is a generalist method, trained to sample prompts across tasks and downstream functionalities.

## 3 Diffusion Meta-Prompting

We introduce the Diffusion Meta-Prompting (DMP) framework for synthesizing prompts conditioned on task descriptions. A community of users first learn a set of prompts $x ( c )$ for each task c in a set ${ \mathcal { C } } ,$ using standard prompt learning techniques. The prompts $x ( c )$ may cover multiple tasks, e.g. SDXL image synthesis or CLIP-based image classification. The learned prompts $\boldsymbol { x } ( c ) \in \mathbb { R } ^ { d }$ are stored in a prompt repository ${ \mathcal { R } } ,$ , together with textual descriptions $y ( c )$ of the task, e.g. “SDXL prompt for tattoo synthesis". The repository prompts $x ( c )$ , are then used to train the DMP model. This is a conditional prompt generator $p _ { \theta } ( x \mid y )$ , parameterized by a diffusion model. Since the repository prompts $x ( c )$ are already learned with a downstream task loss $\mathcal { L } _ { \mathrm { t a s k } } ( S ; c )$ , a DMP trained with the standard MSE loss suffices to synthesize prompts of low downstream risk for task c when conditioned on $y ( c )$ , as shown theoretically in Proposition 1.

Training. DMPs are diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020a) that synthesize prompts by iteratively denoising a noise seed. DMP training is based on a pair of forward and backward Markov chains. Given a task or concept $c ,$ an associated prompt $x ( c )$ is retrieved from R and noised according to a forward process that progressively adds noise to $x _ { 0 } = x ( c )$ , according to

$$
x _ { t } = \sqrt { \alpha _ { t } } x _ { 0 } + \sqrt { 1 - \alpha _ { t } } \epsilon _ { t } ,\tag{1}
$$

where t is a timestep, $\begin{array} { r } { \alpha _ { t } : = \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) , \{ \beta _ { t } \} _ { t = 1 } ^ { T } } \end{array}$ is a variance schedule, $\epsilon _ { t } \sim N ( 0 , \bf { I } )$ and N is a Gaussian distribution. In the reverse process, a neural network $\epsilon _ { \theta }$ recurrently denoises $x _ { t }$ to recover $x _ { 0 }$ . The network $\epsilon _ { \theta } ( x _ { t } , t )$ is a U-Net (Ronneberger et al., 2015) with self and cross-attention layers (Vaswani et al., 2017). The latter are conditioned by a text prompt $y ( c )$ that specifies the task $c , \mathrm { e . g . \ ^ { 6 6 } a }$ personalization prompt for Jennifer Anniston”, in the example of Figure 1. A text embedding $\tau _ { \theta }$ maps $y ( c )$ into a conditioning vector $\tau _ { \theta } ( y )$ , where we omit the argument c for brevity. The denoising network $\epsilon _ { \theta } ( x _ { t } , \tau _ { \theta } ( y ) , t )$ is trained to predict noise $\epsilon _ { t }$ , by minimizing the risk

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d e n o i s e } } ( \theta ) = \mathbb { E } _ { t , x _ { 0 } , y , \epsilon } \left[ \| \epsilon _ { t } - \epsilon _ { \theta } ( x _ { t } , \tau _ { \theta } ( y ) , t ) \| ^ { 2 } \right] . } \end{array}\tag{2}
$$

In our implementation, the DMP model is trained with classifier-free guidance, where an empty text or null prompt is used 20% of the time, to allow for better guidance during sampling. After training, a user simply specifies the task text prompt $y ( c )$ . The diffusion model samples a prompt $x ( c )$ with

$$
x _ { t - 1 } = \sqrt { \alpha _ { t - 1 } } \hat { x } _ { 0 } + \sqrt { 1 - \alpha _ { t - 1 } - \sigma _ { t } ^ { 2 } } \epsilon _ { \theta } ( x _ { t } , \tau _ { \theta } ( y ) , t ) + \sigma _ { t } \epsilon _ { t }\tag{3}
$$

where $\hat { x } _ { 0 } = ( x _ { t } - \sqrt { 1 - \alpha _ { t } } \epsilon _ { \theta } ( x _ { t } , \tau _ { \theta } ( y ) , t ) ) / \sqrt { \alpha _ { t } }$ is a prediction of the denoised prompt according to (1), $x _ { T } \sim N ( 0 , { \bf I } )$ is a noise seed, and $x ( c ) = x _ { 0 }$ . From a meta-learning perspective, θ are meta-parameters amortizing the production of prompts across tasks. Unlike per-task optimization, DMP enables one-shot prompt synthesis by sampling from the learned generator.

Theoretical Guarantee. The following result provides a theoretical guarantee for DMP performance, by bounding the distance between the task loss of the prompts learned by traditional prompt learning and those sampled with a DMP. The proof is given in Appendix A.1.

Proposition 1 (DMP performance guarantee). Let θ be a DMP such that the the empirical version of the denoising loss $o f ( 2 )$ over a repository ofn i.i.d. prompts satisfies $\qquad \mathcal { L } _ { \mathrm { d e n o i s e } } ( \theta ) \overset { \cdot } { \leq } \tau$ for condition y. Then with probability at least $\dot { 1 } - \delta ,$

$$
\big | \mathbb { E } _ { \boldsymbol { x } \sim p _ { \theta } ( \boldsymbol { x } | \boldsymbol { y } ) } [ \mathcal { E } _ { \mathrm { t a s k } } ( \boldsymbol { x } ) ] - \mathbb { E } _ { \boldsymbol { x } \sim \hat { p } _ { \mathrm { d a t a } } ( \boldsymbol { x } | \boldsymbol { y } ) } [ \mathcal { E } _ { \mathrm { t a s k } } ( \boldsymbol { x } ) ] \big | \le L _ { \operatorname* { m a x } } \sqrt { 2 C \tau } + L _ { \operatorname* { m a x } } \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } + o ( 1 ) .\tag{4}
$$

where $p _ { \theta } ( x \mid y )$ is probability distribution of the DMP model conditioned on y, $\hat { p } _ { d a t a } ( x \mid y )$ is the empirical distribution of the data conditioned on $y , \mathcal { L } _ { \mathrm { t a s k } } ( x )$ is a task loss bounded $b y \left[ 0 , L _ { \operatorname* { m a x } } \right]$ , and $L _ { m a x } ^ { \phantom { \dagger } } , C$ are constants.

Hence, given large enough n and small enough τ, DMP is guaranteed to meet the performance of prompt learning.

Architecture. To implement the DMP model, we develop a 1D variant of the popular Stable Diffusion model, where 2D operations are replaced by their 1D counterpart $( \mathrm { e . g . }$ ., 1D-conv). Since the size of the prompt embeddings (d) is relatively small, we have found that, for most applications, the model can operate directly in the space $\mathcal { P }$ of prompts $\boldsymbol { x } ( c ) \in \mathbb { R } ^ { d }$ . However, for applications involving unusually long prompts $( d \geq 2 0 \hat { 4 } 8 )$ , we have also trained a variational autoencoder that maps prompts from $\mathcal { P }$ into a space of lower dimensionality, for faster training and convergence. This is a 1D variant of stable diffusion autoencoder with less than 1M parameters and a latent dimension of 128. See appendix section A.10 for more details.

Concept Composition. By framing diffusion models as Energy Based Models, (Liu et al., 2022b) showed that it is possible to compose multiple tasks by conjunction or negation. Given a diffusion model $\epsilon _ { \theta } ( x _ { t } , t )$ , n tasks $c _ { i }$ are combined by implementing the denoising chain with

$$
\hat { \epsilon } ( x _ { t } , t ) = \epsilon _ { \theta } ( x _ { t } , t ) + \sum _ { i = 1 } ^ { n } \eta _ { i } \left( \epsilon _ { \theta } ( x _ { t } , \tau _ { \theta } ( y ( c _ { i } ) ) , t ) - \epsilon _ { \theta } ( x _ { t } , t ) \right)\tag{5}
$$

where $\eta _ { i }$ is a hyperparameter corresponding to a temperature scaling of task $c _ { i }$ . If the conditioning is just empty text, this reduces to classifier-free guidance. The standard implementation of task composition is to run the downstream diffusion model n times (once per task) and average noise predictions with (5) (Liu et al., 2022b). This is significantly more complex than performing the task composition of (5) in the (much less complex) DMP, which enables the sampling of single prompt $x _ { 0 }$ for the downstream model that combine all tasks $c _ { 1 } , \ldots , c _ { n }$

Test-Time Steering. While DMP can synthesize prompts in one shot, different samples from $p _ { \theta } ( x \mid y ( i ) )$ may lie in prompt regions with different downstream generalization behavior. To address this, we introduce a simple test-time steering mechanism that biases diffusion sampling toward a high-performing prompt already available in the repository, without retraining the DMP model.

For each task dataset i, let $\{ x _ { i , m } \} _ { m = 1 } ^ { M }$ denote the prompts obtained from M different prompt-learning initializations. We evaluate prompt x, by measuring the accuracy $\mathrm { A c c } _ { \mathrm { t r a i n } } ( x )$ of the downstream model prompted with x on the training split, and select the best

$$
x _ { i } ^ { \star } = \arg \operatorname* { m a x } _ { m } \mathrm { ~ A c c _ { t r a i n } ( } x _ { i , m } \mathrm { ) . }\tag{6}
$$

This prompt is then used as an anchor for DMP sampling by replacing, in eq. $( 3 ) , { \hat { x } } _ { 0 }$ with an estimate of the denoised prompt steered by $\boldsymbol { x } _ { i } ^ { \star }$

$$
\begin{array} { r } { \tilde { x } _ { 0 } ( x _ { t } , y ( i ) , t ) = ( 1 - \lambda _ { t } ) \hat { x } _ { 0 } ( x _ { t } , y ( i ) , t ) + \lambda _ { t } x _ { i } ^ { \star } , } \end{array}\tag{7}
$$

where $\lambda _ { t } \in [ 0 , 1 ]$ controls the steering strength. In practice, we use a constant value $\lambda _ { t } = \lambda$ for all denoising steps. The DDIM transition is then computed with $\tilde { x } _ { 0 }$ in place of $\scriptstyle { \hat { x } } _ { 0 }$ in (3) with $\sigma _ { t } = 0$

This procedure can be interpreted as imposing a test-time prior over the prompt manifold. The diffusion model still conditions on the dataset text $y ( i )$ and samples from a random noise seed, but the synthesized prompt is gently pulled toward a known low-risk region of the repository. Note that the generated prompt is not a copy of $\boldsymbol { x } _ { i } ^ { \star }$ . Instead, $\boldsymbol { x } _ { i } ^ { \star }$ acts as an anchor that stabilizes sampling around a prompt basin of empirically strong performance on the training data.

For composite tasks, e.g. classification problems where the conditioning string contains class names from $K$ datasets of anchors $\{ x _ { i _ { j } } ^ { \star } \} _ { j = 1 } ^ { K } .$ , we use the average anchor $\begin{array} { r } { x _ { \mathrm { c o m p } } ^ { \star } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } x _ { i j } ^ { \star } } \end{array}$ , as $x _ { i } ^ { \star }$ in Eq. 7. Importantly, the prompt anchor is selected only from training-set performance and does not require access to novel test data. The method is therefore a lightweight test-time scaling strategy: it reuses the repository to identify a reliable anchor and improves the stability of DMP sampling while preserving the ability to synthesize new prompts conditioned on task specifications.

## 4 Experiments

Implementation Details. We conduct all experiments on 24GB (NVIDIA-A10 or 3090-RTX) GPUs using pytorch. DMP models are trained with the standard hyperparameters of (Rombach et al., 2022), a learning rate of $1 e ^ { - 6 }$ , and batch size of 320, for 2, 000 epochs. This takes about a day to train on 4 GPUs for 3,000 SDXL personalization prompts. For classification prompts, the DMP autoencoder and diffusion is trained for 10, 000 epochs, which takes about 10 hours<sup>2</sup>. DMP uses 50 DDIM timesteps to sample one prompt which takes about 1 sec and it can be sped up with faster sampling methods (Liu et al., 2022a). For fair comparisons, we use the codebase of respective prompt learning methods and run all our experiments under this setup. For test-time steering, we use $\lambda = 0 . 2$ . More details on the architecture and models are in the appendix A.10.

## 4.1 DMP for classification

We trained a DMP model, DMPClass, for prompting a CLIP-style classifier. The goal is not to beat the SOTA in prompt based classifiers, but to show that DMP can synthesize prompts that generalize to multiple tasks, outperforming other meta-prompting approaches. We utilize CoOp prompts to build the prompt repository with imagenet and the 10 fine-grained datasets commonly used in the prompt learning literature (Zhou et al., 2022b,a). We note that DMP is agnostic to the prompt repository and can be applied to other prompting methods. However, all methods other than CoOp use additional weights, projection matrices, or architectural enhancements that involve high dimensional parameter matrices. Modeling these parameters requires much larger diffusion models and compute than we used, which is sufficient to learn prompts. More importantly, most of the meta-prompting baselines that we compare to simply do not scale to the generation of such parameters. Appendix section A.3 discusses the application of DMP to other prompting methods, where it is only used to sample prompts, maintaining the additional parameters from the original methods. The appendix shows that DMP still achieves consistent gains over the original prompts, although these are smaller than for CoOp, which has no additional parameters.

Prompt Repository. Given a classification problem, a prompt set is trained for N epochs (where N follows the original paper settings), using a context vector of size K. The CoOp paper reports full results only for context length 16. We use the context length of 4 $( K = 4 )$ suggested in CoCoOp (Zhou et al., 2022a). All prompt learning methods are trained under the Base2New setting, which assesses generalization to unseen (New) classes by only using half of dataset (Base) classes for training. DMP is trained with the prompts originally learned on the Base set. We evaluate the DMP-sampled prompts on training (Base), remaining (New), and all (All) classes. We use $M = 4 0$ different initializations per dataset and save the corresponding prompt embeddings to produce a training repository $\mathcal { R } \in \mathbb { R } ^ { 1 1 \times 4 0 \times K \times d }$ . The best performing prompt of $( 6 ) \left( x _ { \mathrm { c o m p } } ^ { \star } \right.$ for composite tasks) is used as anchor for DMP sampling and as the baseline CoOp prompt, against which DMP prompts are compared.

Training. Due to the high dimensionality of the prompt embeddings $( K \times d = 2 0 4 8 )$ , DMPClass models are trained with an autoencoder. To obtain the text inputs, we concatenate all the class labels $c _ { k } , k \in \{ 1 , . . . C _ { i } \}$ , of dataset i into a text string $y ( i ) = \{ c _ { 1 } , \ldots , c _ { C _ { i } } \}$ whose text embedding $\tau ( y ( i ) )$ is used to condition the DMPClass model. This setup allows us to flexibly mix and match class names across datasets to enable straightforward compositionality (Table 3) without requiring handcrafted or semantically enriched descriptions. Once the model is trained, it suffices to sample conditioned on a string of mixed class labels from any of the datasets. Since the CLIP text encoder has a limit

Table 2: Zero-shot accu- Table 3: Accuracy comparison racy comparison on com- on combined EuroSAT and bined dataset classes. Flowers dataset pair.
<table><tr><td>Method</td><td>All</td><td>Base New</td></tr><tr><td>CoOp</td><td>69.6</td><td>73.6 76.7</td></tr><tr><td>BPL</td><td>72.0</td><td>77.5 78.4</td></tr><tr><td>ProMetaR</td><td>71.2</td><td>75.8 77.6</td></tr><tr><td>DMP</td><td>73.6</td><td>76.0 78.8</td></tr><tr><td> $\Delta _ { \mathrm { C o O p } }$ </td><td>+4.0 +2.4</td><td>+2.1</td></tr></table>

<table><tr><td>Method</td><td>All Base New</td></tr><tr><td>CoOp (ZS) BPL (ZS)</td><td>35.5 50.2 59.4 52.6 61.3 73.4</td></tr><tr><td>ProMetaR (ZS) DMP (ZS)</td><td>48.0 59.9 67.1 61.1 74.5 77.8</td></tr><tr><td>BPL (Trained)</td><td>56.2 76.1 75.1</td></tr></table>

Table 4: Ablation of test-time steering on 55 composite datasetpairs classification. ∆<sub>steer</sub>, $\Delta _ { \mathrm { C o O p } }$ denotes the absolute accuracy gain over respective baselines.
<table><tr><td>Split CoOp BPL ProMetaR</td><td></td><td>DMP w/o Steering w/ Steering</td><td>DMP</td><td></td><td> $\Delta _ { \mathrm { s t e e r } } \Delta _ { \mathrm { C o O p } }$ </td></tr><tr><td>All 58.7 60.8</td><td>61.2</td><td>61.4</td><td>63.1</td><td>+1.7</td><td>+4.4</td></tr><tr><td>Base 65.1 66.3</td><td>67.4</td><td>67.3</td><td>69.4</td><td>+2.1</td><td>+4.3</td></tr><tr><td>New 65.1 68.7</td><td>68.9</td><td>68.6</td><td>70.8</td><td>+2.2</td><td>+5.7</td></tr><tr><td>Avg. 63.065.3</td><td>65.8</td><td>65.8</td><td>67.8</td><td>+2.0</td><td>+4.8</td></tr></table>

Table 5: Memory and complexity requirements for DMP against retrieval.

Table 6: Comparison of DMP-Variation with Textual Inversion (TI) for variation synthesis.
<table><tr><td>Method|Mem</td><td>(GB) ↓</td><td>|Time |(s) ↓</td></tr><tr><td>Stylus DMP ∆</td><td>|1.30 0.12  $\big | ( 9 1 \% \downarrow ) \big | ( 9 2 \% \downarrow )$ </td><td>|12.1 1</td></tr></table>

Table 7: Sparse multi-view retrieval on MVImgNet. The gallery contains 5,447 objects, with only one to four available views per object. DMP is trained on objects disjoint from the gallery and synthesizes embeddings for missing views.
<table><tr><td>Method Face-ID CLIP-T CSD (SD-RV) ↓</td><td>↑</td><td>↑</td></tr><tr><td>TI 0.428</td><td>27.0</td><td>50.4</td></tr><tr><td>DMP 0.299</td><td>28.4</td><td>61.8</td></tr><tr><td>∆ -0.13</td><td>+1.4</td><td>+11.4</td></tr></table>

<table><tr><td>Method</td><td>NDCG</td><td>MRR</td><td>Top-1</td><td>mAP</td></tr><tr><td>DINOv2</td><td>91.64</td><td>94.36</td><td>91.44</td><td>86.20</td></tr><tr><td>DMP</td><td>93.47</td><td>94.42</td><td>91.74</td><td>89.64</td></tr><tr><td>∆</td><td>+1.83</td><td>+0.06</td><td>+0.30</td><td>+3.44</td></tr></table>

of 77 tokens (around 50 words), these strings can be too long for many classes (e.g. the 1,000 Imagenet classes). To overcome this, we consider each class name independently and obtain the CLIP embedding for each class, resulting in C embeddings for C classes, which are concatenated into a vector that is used to prompt the DMPClass model. See A.6.3 for ablation on text inputs.

Composite classification. To evaluate the generalization ability of DMP, we introduce a composite classification task. We consider two settings under this task: (1) classify over the set of all classes of all datasets and (2) classify over pairs of class label sets from the fine-grained datasets we considered. For setting (2), we created 55 datasets $\mathcal { T } _ { i } , i \in \{ 1 , . . . , 5 5 \}$ containing pairs of All, Base and New classes from the original 11 datasets. These settings test out-of-domain generalization, since prompts trained on one dataset are used to classify a mix of their classes and classes from other datasets. For example, while Oxford Flowers classes require modeling flower attributes, the composition with FGVC Aircrafts requires the modeling of both flower and airplane attributes, on which the original prompts were never trained simultaneously. For each dataset T<sub>i</sub>, we performed classification with (1) the prompt sampled from DMP (trained on the repository of individual dataset prompts), and (2) the average of two anchor prompts x<sup>⋆</sup> learned from the two datasets that compose T<sub>i</sub>.

For the first setting, we compare DMP against the baseline CoOp prompts as well as prior metaprompt learning methods such as BPL (Derakhshani et al., 2023) and ProMetaR (Park et al., 2023). Table 2 shows the results under All, Base and New Classes. For evaluation over all classes, DMP improves over CoOp by 4% and outperforms the best meta-prompting baseline (BPL) by 1.6%. Table 3 shows, that, in setting (2), the DMP gains can be more even pronounced for specific dataset class pairs, such as EuroSAT and Flowers. For this pair, DMP exceeds BPL’s zero-shot baseline by +8.5/+4.4% and CoOp by +25.6/18.4%for all/new classes, respectively. We further compare DMP with BPL (Trained), which is trained on the entire dataset pair. As shown in Table 3, DMP without any task-specific training even outperforms this approach by +2.7% on new classes. We also note that BPL is computationally expensive, requiring approximately 10 hours on a 40GB GPU even for a single dataset pair, which limits its scalability to composite tasks as compared to DMP.

Table 4 (left) compares the performance of the DMP prompts to others, showing the average accuracy across all 55 dataset combination pairs. DMP obtains gains of +2.0% for Base,+1.9% for New, and +1.9% for All classes over ProMetaR, the best baseline for the pairwise setting. DMP further outperforms CoOp by +4.3% for Base,+5.7% for New, and +4.4% for All classes.

Effect of test-time steering. Table 4 also isolates the contribution of the test-time steering mechanism of (7). Compared with DMP sampling by the same model without steering, test-time steering consistently improves accuracy across all evaluation splits. The gain is +1.7 points on the All split, +2.1 on the Base split, and +2.2 on the New split, yielding an average improvement of +2.0 points. This suggests that steering the DDIM sampling trajectory toward a training-selected high-performing prompt is a useful test-time regularization towards high-performing regions of the prompt manifold. Importantly, the steering target is selected using only training-split performance and does not require access to base or novel test classes, so the improvement is obtained without additional prompt training or test-label supervision. Ablation on the steering strength is shown in Appendix A.3

![](images/35533f19c0dc25f87733aad7cbaa3fe615031cecaee6287e0d2c9aa5edc57041.jpg)  
Figure 2: Taxonomic classification generaliza- Figure 3: Generalization of DMPVariation model. Images generated tion: Comparison of DMP against CoOp and using prompts synthesized by TI (top) and DMPVariation (bottom) for BPL prompts for hierarchical classification. DMP various downstream model text prompts (shown on top). DMPVariation prompts generalize better than BPL prompts with prompts are more robust than TI prompts, which impair the ability of the ≈ 2-9% accuracy gains. downstream model to generalize. See Fig. 16 for additional results.

Taxonomic. We next consider the more challenging cross-task generalization setting of taxonomic classification (Wu et al., 2024). This tests the ability of the downstream models to classify images with respect to different class subsets in a class hierarchy, using the metric of Mean Treecut Accuracy (MTA) over 25 treecuts. Since the training classes are the leaves of taxonomy, hierarchical classification requires generalization from finer to coarser classes. For example, while ‘cats’ and ‘dolphins’ are two very different classes, they both belong to ‘mammals’. (Wu et al., 2024) showed that standard prompt learning methods perform poorly on this task, where effective prompt learning requires hierarchical training and sampling of label sets across the entire taxonomy.

Figure 2 shows the MTA performance of CoOp, BPL, and DMP prompts. Since only ImageNet and SUN provide labels with respect to the entire class taxonomy, we used these two datasets, plus the OOD variants of ImageNet on this experiment. Table 21 presents the full results of the experiment. DMP prompts obtain an average gain between 2.4 and 9.4% over BPL and a larger gain between 5.8 and 11.7% over CoOp, even though no prompts are ever trained for hierarchical classification. These results are consistent with the previous findings and show that DMP excels as the generalization challenge increases, learning to produce prompts that are much more robust than those of the CoOp prompt learning method in which it was trained.

## 4.2 DMPVariation

Prompt learning is an approach to personalize diffusion models to the synthesis of images of specific concepts. DMPVariation is a DMP model that synthesizes personalization prompts learned via textual inversion (TI) (Gal et al., 2022), to prompt a diffusion model to synthesize variations of a subject.

Prompt Repository. To produce a prompt repository R, we split the CelebA dataset into unique identities using the groundtruth labels, and use 3,000 of these as concepts c, for which we train personalized prompts x(c) with Textual Inversion using 1,000 steps of gradient descent. To obtain prompts for variations, we note that earlier optimization steps do not fully encode face attributes, as compared to steps later in the optimization. We save the prompts x(c) from 40 gradient steps (between 200 − 400 steps in intervals of 5), per identity. This produces a repository R of 40 prompt variations per identity to obtain a training repository R ∈ R<sup>3000×40×d</sup>. The text string y(c) associated with task c is “a photo of <identity-c>".

Training. The DMPVariation model was trained directly in prompt space P, i.e. without autoencoder. Each personalized prompt produced by TI is used as conditional input to DMPVariation, by concatenating it with the noise vector. During training, the model learns to denoise the 40 different variations of the conditional prompt input. Hence, sampling from the DMP produces a diversity of prompts that induce the downstream model to synthesize variations of the subject face (see Fig. 15 for an example). At inference, prompts that induce variations of novel subjects can be sampled with DMP by simply conditioning on the TI prompt for the new subject.

DMPVariation demonstrates the effectiveness of DMP for generative tasks, namely the task of synthesizing images with variations of a subject or concept. To evaluate prompt generalization in this context, we compare DMPVariation prompts to the prompts originally learned by TI on the CelebA dataset. Performance is measured with Face-ID (VGGFace2), CLIP-Score across 28 diverse prompts (listed on Table 25), and the HPSv2 metric on 100 prompt concepts downloaded from the Huggingface repository. Note that none of these concepts are on the CelebA dataset. Table 6 shows that DMPVariation achieves lower Face-ID similarity, indicating diverse yet semantically similar prompts, and improves CLIP-Text similarity by 1.4% and the CSD style metric by +11.4%, demonstrating better generalization to novel prompts compared to TI embeddings. However, as is typical for generative modeling, these metrics do not capture the substantial qualitative gap between the two approaches. While TI prompts produce images that do not fully comply with the instruction, DMP prompts consistently produce images that satisfy the instructions.

Figure 3 illustrates this gap by presenting images synthesized with DMPVariation and Textual Inversion (TI) prompts. The downstream model, personalized to a subject with prompts produced by TI or DMPVariation, is asked to generate an image of the subject in a new context, specified as a text prompt atop each image. The personalization prompts produced by DMPVariation demonstrate greater robustness and generalization, effectively adapting to different contexts. This is unlike TI prompts, which tend to overfit to the subject, leading to poor generalization across contexts. In result, TI prompts fail to generate suitable images of the dog for all contexts other than “in front of a house". The figure also shows that DMPVariation produces successful variation prompts for general objects like dogs despite being trained only on CelebA-face identities. Fig. 13 presents images for other objects, such as bird and statue. Fig. 16 shows additional results where TI completely fails as compared to DMPVariation.

DMP for sparse multi-view retrieval. We next evaluate whether DMP can generalize beyond prompt synthesis by applying it to a representation-level retrieval task. We consider sparse multi-view retrieval, where the query is one view of an object and the goal is to retrieve all other views of the same object from a gallery. Unlike densely populated multi-view retrieval settings, we assume that gallery coverage is incomplete: each object has only between one and four available gallery views. This creates a missing-view problem, where direct retrieval from the observed gallery may fail when the query view differs substantially from the available views of the same object.

We construct this task from a subset of MVImgNet (Yu et al., 2023) containing 5,447 gallery objects. The baseline is a retrieval operation using DINOv2 embeddings of the available gallery views. DMP is trained to synthesize missing views on a set of objects disjoint from those in the gallery, to guarantee that all objects are novel at test time. DMP-based retrieval complements the observed DINOv2 embeddings with DMP-synthesized embeddings of the missing views of each object. Retrieval is then performed against an aggregated gallery containing both the observed and DMP-synthesized embeddings.

Table 7 shows that DMP consistently improves retrieval performance over direct DINOv2 embeddings. The largest gain is observed for mAP (+3.44), indicating that synthesized missing-view embeddings improve the ranking of all relevant views, rather than only the top prediction. DMP also improves NDCG by +1.83 and Top-1 accuracy by +0.30, while maintaining comparable MRR. These results show that DMP learns useful structure in the space of foundation-model embeddings and can synthesize representations that compensate for sparse gallery coverage. Importantly, this experiment demonstrates that the benefits of DMP are not limited to learned prompt spaces: the same metagenerative framework can provide gains for different representations and downstream tasks.

Storage and Runtime Efficiency. One of the benefits of the DMP framework is its high efficiency. DMP eliminates the need to manage a repository of prompts and associated metadata, such as text descriptions. This contrasts with methods like Stylus (Luo et al., 2024), that automatically search, retrieve, and compose LoRAs from a repository. Table 5 compares the storage and processing requirements of DMP and Stylus, when used for the tasks that we consider in this work. DMP is significantly more efficient, reducing storage needs by 91% and improving inference speed by 92%.

## 5 Ablation Studies

Robustness to anchor perturbation. To assess whether the gains of test-time steering depend on the exact anchor $x _ { i } ^ { \star }$ of (6), we perturb it in random directions at increasing $L _ { 2 }$ distances before using it in (7). Table 8 shows that accuracy remains essentially unchanged under moderate to large perturbations, varying by at most 0.27 points across splits. This suggests that steering does not rely on a single privileged prompt, but instead guides sampling toward a broad high-performing neighborhood of the prompt manifold. Since the anchor is fixed throughout sampling, however, these results alone do not distinguish task-specific guidance from the stabilizing effect of a fixed target.

To separate the two, we replace the training-selected anchor with a fixed, randomly selected training prompt. Accuracy decreases from 63.1/69.4/70.8 to 62.4/69.1/70.2 on the All/Base/New splits, i.e., by 0.7/0.3/0.6 points. Since this alternative anchor also remains fixed during sampling, the degradation cannot be attributed to a loss of stabilization alone, indicating that the training-selected anchor provides task-relevant guidance beyond the text condition.

Table 8: Robustness of test-time steering to perturbations of the anchor prompt. Results are averaged over dataset pairs, and over repeated noise samples within each pair. ∆ is relative to the unperturbed anchor. Fixed anchor: a randomly selected training prompt is used as the anchor. Resampled: the perturbation direction is resampled at every denoising step.
<table><tr><td rowspan="2">Setting</td><td rowspan="2"></td><td colspan="2">All</td><td colspan="2">Base</td><td colspan="2">New</td></tr><tr><td>Acc.</td><td>∆</td><td>Acc.</td><td> $\Delta$ </td><td>Acc.</td><td> $\Delta$ </td></tr><tr><td rowspan="2"> $L _ { 2 }$   $L _ { 2 }$ </td><td>0.00</td><td>63.10</td><td></td><td>69.40</td><td></td><td>70.80</td><td></td></tr><tr><td>0.25</td><td>63.01</td><td>-0.09</td><td>69.13</td><td>-0.27</td><td>70.77</td><td>-0.03</td></tr><tr><td rowspan="2"> $L _ { 2 }$   $L _ { 2 }$ </td><td>0.50</td><td>62.94</td><td>-0.16</td><td>69.13</td><td>-0.27</td><td>70.73</td><td>-0.07</td></tr><tr><td>1.00</td><td>63.09</td><td>-0.01</td><td>69.28</td><td>-0.12</td><td>70.90</td><td>+0.10</td></tr><tr><td colspan="2">Fixed anchor</td><td>62.36</td><td>-0.74</td><td>69.12</td><td>-0.28</td><td>70.21</td><td>-0.59</td></tr><tr><td colspan="2">Resampled</td><td>61.27</td><td>-1.83</td><td>67.47</td><td>-1.93</td><td>68.56</td><td>-2.24</td></tr></table>

Table 9: Sensitivity of DMP to the class-name condition, with and without test-time steering. A fraction of the Caltech101 class names in the condition is replaced with ImageNet class names. $L _ { 2 }$ distances of the condition embedding and of the sampled prompt are measured with respect to the original (0%) condition.
<table><tr><td>Replaced names</td><td>Condition  $L _ { 2 }$ </td><td colspan="3">w/o Steering</td><td colspan="3">w/ Steering</td></tr><tr><td></td><td></td><td>Prompt  $L _ { 2 }$ </td><td>Caltech101</td><td>ImageNet</td><td>Prompt  $L _ { 2 }$ </td><td>Caltech101</td><td>ImageNet</td></tr><tr><td>0%</td><td>0.0</td><td>0.000</td><td>92.0</td><td>64.7</td><td>0.000</td><td>93.0</td><td>64.9</td></tr><tr><td>40%</td><td>24.2</td><td>1.623</td><td>92.1</td><td>64.4</td><td>0.025</td><td>93.0</td><td>64.8</td></tr><tr><td>60%</td><td>29.8</td><td>1.606</td><td>92.6</td><td>64.3</td><td>0.049</td><td>93.1</td><td>64.5</td></tr><tr><td>80%</td><td>34.0</td><td>1.600</td><td>92.5</td><td>64.5</td><td>0.049</td><td>93.2</td><td>64.5</td></tr></table>

Finally, we resample the perturbation direction at every denoising step, which removes the consistency of the steering target across the trajectory. This leads to a larger reduction, 1.8/1.9/2.2 points below the unperturbed anchor, showing that a consistent steering target is also important. Overall, the anchor plays two complementary roles: its location in prompt space guides sampling toward a task-relevant region, and its consistency across denoising steps stabilizes the sampling trajectory.

Sensitivity to the class-name condition. We next examine how DMP responds to changes in its text condition, with and without steering. Starting from the Caltech101 class-name condition, we progressively replace 40–80% of the class names with distinct ImageNet class names. Table 9 reports the $L _ { 2 }$ distance of the resulting condition embedding and of the sampled prompt from those of the original condition, together with the downstream accuracy. As more names are replaced, the condition embedding moves away from the original, with its $L _ { 2 }$ distance increasing from 17.4 to 34.0.

Without steering, these changes propagate to the sampled prompts, whose $L _ { 2 }$ distance from the original prompt reaches 1.42–1.62, and replacing 40% of the names changes ImageNet accuracy by 0.3 points. Hence, neither the text encoder nor DMP collapses related class vocabularies into the same representation or prompt. With steering, the prompt distances decrease to 0.025–0.049, and accuracy remains stable at 93.0–93.2% on Caltech101 and 64.5–64.9% on ImageNet. These results separate the two effects: DMP remains sensitive to the class-name condition, while the training-selected anchor limits how strongly this variation changes the sampled prompt and its downstream behavior, stabilizing generation within a locally high-performing region of prompt space.

See Appendix A.4 for application of DMP for personalization and image editing, Sec. A.6 for additional ablations and Sec. A.9 for a discussion on the limitations and scope for future works.

## 6 Conclusion

We introduced a Diffusion Meta-Prompting (DMP) framework that learns a distribution over prompts and enables synthesis of task-specific prompts directly from natural language. Across recognition, retrieval, and generation settings, DMP produces representations and prompts that generalize beyond the tasks, label sets, views, and concepts seen during training, while avoiding the overhead of storing and searching large repositories. Empirically, DMP yields consistent gains in the most challenging generalization regimes improving cross-task and composite classification, supports compositional control and semantic negation, and improves prompt compliance for subject variations. Finally, we provided a theoretical bound linking the expected downstream task loss of DMP-sampled prompts to repository quality and diffusion denoising error, offering a principled perspective on when metaprompt generation succeeds.

## Acknowledgements

This work was partially funded by Qualcomm Innovation Fellowship 2025, NSF awards IIS-2303153 and NAIRR-240300. We also acknowledge and thank the use of the Nautilus platform for some of the experiments.

## Broader Impact

We introduce a new meta-learning framework for generating prompts for foundation models. It carries the risks associated with generative foundation models. While the proposed method offers benefits of better generalization with storage and runtime efficiency, it uses existing pretrained models which are shown to contain harmful biases that maybe elicited by the prompts. It can also be potentially misused to propagate harmful, unlawful or unethical information with the personalization of celebrities. Since, the framework is meta-learning, any harmful prompts can be identified before the image generation step where the embeddings can be inspected with nearest neighbor tokens in the text space. Additionally, advancements in image watermarking (Luo et al., 2022; Cao et al., 2025; Kohli and Gowal, 2023; Gowal et al., 2025) can help to identify generated image contents to protect against these risks.

## References

Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W. Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando de Freitas. Learning to learn by gradient descent by gradient descent. In Advances in Neural Information Processing Systems, 2016.

Stefan Andreas Baumann, Felix Krause, Michael Neumayr, Nick Stracke, Vincent Tao Hu, and Björn Ommer. Continuous, Subject-Specific Attribute Control in T2I Models by Identifying Semantic Directions, 2024.

Jie Cao, Qi Li, Zelin Zhang, and Jianbing Ni. Secure and robust watermarking for ai-generated images: A comprehensive survey, 2025. URL https://arxiv.org/abs/2510.02384. A broad survey on watermarking for AI-generated visual content.

Civitai. https://civitai.com/, 2023. Webpage.

Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. Wiley-Interscience, 2 edition, 2006.

Mohammad Mahdi Derakhshani, Enrique Sanchez, Adrian Bulat, Victor Guilherme Turrisi da Costa, Cees GM Snoek, Georgios Tzimiropoulos, and Brais Martinez. Bayesian prompt learning for image-language model generalization. ICCV, 2023.

Yingjun Du, Gaowen Liu, Yuzhang Shang, Yuguang Yao, Ramana Kompella, and Cees G. M. Snoek. Prompt diffusion robustifies any-modality prompt learning, 2024. URL https://arxiv.org/ abs/2410.20164.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning, pages 1126–1135. PMLR, 2017.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv, 2022. doi: 10.48550/ARXIV.2208.01618. URL https://arxiv.org/abs/ 2208.01618.

Chunjiang Ge, Rui Huang, Mixue Xie, Zihang Lai, Shiji Song, Shuang Li, and Gao Huang. Domain adaptation via prompt learning. arXiv preprint arXiv:2202.06687, 2022.

Sven Gowal, Rudy Bunel, Florian Stimberg, David Stutz, Guillermo Ortiz-Jimenez, Christina Kouridi, Mel Vecerik, Jamie Hayes, Sylvestre-Alvise Rebuffi, Paul Bernard, Chris Gamble, Miklós Z. Horváth, Fabian Kaczmarczyck, Alex Kaskasoli, Aleksandar Petrov, Ilia Shumailov, Meghana Thotakuri, Olivia Wiles, Jessica Yung, Zahra Ahmed, Victor Martin, Simon Rosen, Christopher

Savcak, Armin Senoner, Nidhi Vyas, and Pushmeet Kohli. Synthid-image: Image watermarkingˇ at internet scale, 2025. URL https://arxiv.org/abs/2510.09263. A deep learning–based system for invisibly watermarking AI-generated imagery at internet scale.

David Ha, Andrew M. Dai, and Quoc V. Le. Hypernetworks. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=rkpACe1lx.

Fusheng Hao, Fengxiang He, Fuxiang Wu, Tichao Wang, Chengqun Song, and Jun Cheng. Task-aware clustering for prompting vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025.

Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Introducing eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. In IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, pages 204–207, 2018. doi: 10.1109/IGARSS.2018.8519248.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran Associates, Inc., 2020a. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/ 4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020b.

Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal ofthe American Statistical Association, 58(301):13–30, 1963.

Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9): 5149–5169, 2021. doi: 10.1109/TPAMI.2021.3069109.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= nZeVKeeFYf9.

HuggingFace. https://huggingface.co/blog/lora-adapters-dynamic-loading, 2023. Webpage.

Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European Conference on Computer Vision (ECCV), 2022.

Muhammad Uzair khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Maple: Multi-modal prompt learning. In The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Muhammad Uzair Khattak, Syed Talal Wasim, Muzammal Naseer, Salman Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Self-regulating prompts: Foundational model adaptation without forgetting. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15190–15200, October 2023.

Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014.

Pushmeet Kohli and Sven Gowal. Identifying ai-generated images with synthid. Google DeepMind Blog, 2023. URL https://deepmind.google/blog/ identifying-ai-generated-images-with-synthid. SynthID watermarking technology for AI-generated images.

Haoyang Li, Liang Wang, Chao Wang, Jing Jiang, Yan Peng, and Guodong Long. Dpc: Dual-prompt collaboration for tuning vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025a. URL https://arxiv.org/ abs/2503.13443. arXiv preprint arXiv:2503.13443.

Zheng Li, Yibing Song, Ming-Ming Cheng, Xiang Li, and Jian Yang. Advancing textual prompt learning with anchored attributes. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025b. URL https://iccv.thecvf.com/Conferences/2025/ AcceptedPapers. Accepted (Poster).

Zhenguo Li, Fengwei Zhou, Fei Chen, and Hang Li. Meta-sgd: Learning to learn quickly for few-shot learning. arXiv preprint arXiv:1707.09835, 2017.

Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In International Conference on Learning Representations, 2022a. URL https: //openreview.net/forum?id=PlKWVd2yBkY.

Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XVII, pages 423–439. Springer, 2022b.

Michael Luo, Justin Wong, Brandon Trabucco, Yanping Huang, Joseph E. Gonzalez, Zhifeng Chen, Ruslan Salakhutdinov, and Ion Stoica. Stylus: Automatic adapter selection for diffusion models, 2024.

Xiyang Luo, Michael Goebel, Elnaz Barshan, and Feng Yang. Leca: A learned approach for efficient cover-agnostic watermarking, 2022.

Colin McDiarmid. On the method of bounded differences. In Johannes Siemons, editor, Surveys in Combinatorics, volume 141, pages 148–188. Cambridge University Press, 1989.

Alex Nichol, Joshua Achiam, and John Schulman. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999, 2018.

Jinyoung Park, Juyeon Ko, and Hyunwoo J. Kim. Prompt learning via meta-regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023.

A. Radford, J. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763, 2021.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI, 2019.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.

Shuvendu Roy and Ali Etemad. Consistency-guided prompt learning for vision-language models. In ICLR, 2024.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. CVPR, 2023.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265, Lille, France, 07–09 Jul 2015. PMLR.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Deepak Sridhar and Nuno Vasconcelos. Prompt sliders for fine-grained control, editing and erasing of concepts in diffusion models. In In Proceedings of the IEEE/CVF European Conference on Computer Vision Workshops, 2024.

Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, 2008.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661–1674, 2011.

Tz-Ying Wu, Chih-Hui Ho, and Nuno Vasconcelos. ProTeCt: Prompt Tuning for Taxonomic Open Set Classification . In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16531–16540, Los Alamitos, CA, USA, June 2024. IEEE Computer Society. doi: 10.1109/CVPR52733.2024.01564. URL https://doi.ieeecomputersociety.org/10. 1109/CVPR52733.2024.01564.

Lingxiao Yang, Ruyuan Zhang, Yanchen Wang, and Xiaohua Xie. Mma: Multi-modal adapter for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23826–23837. IEEE/CVF, 2024.

Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. Mvimgnet: A large-scale dataset of multi-view images. In CVPR, 2023.

Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, and Huicheng Zheng. Inserting anybody in diffusion models via celeb basis. arXiv preprint arXiv:2306.00926, 2023.

Baoquan Zhang, Chuyao Luo, Demin Yu, Huiwei Lin, Xutao Li, Yunming Ye, and Bowen Zhang. Metadiff: Meta-learning with conditional diffusion for few-shot learning, 2024. URL https: //arxiv.org/abs/2307.16424.

Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. ICCV, 2023.

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022a.

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for visionlanguage models. International Journal ofComputer Vision (IJCV), 2022b.

## A Appendix

## A.1 Theoretical Analysis

Let C denote a distribution over tasks or concepts $c \sim { \mathcal { C } } .$ . For each concept c, we assume a textual description $y ( c ) \left( \mathbf { e . g . , \vec { \nu } \vec { a } } \right.$ personalization prompt for Jennifer Aniston”), a downstream task with loss $\mathcal { L } _ { \mathrm { t a s k } } ( \bar { S } ; c )$ that evaluates the performance of a candidate prompt S in prompt space $\mathcal { P }$ when applied to a frozen foundation model ${ \bar { \boldsymbol { F } } } ,$ and a repository R of exemplar prompts $x ( c )$ obtained from existing prompt-learning techniques.

Our objective in this section is to bound the expected downstream loss when sampling prompts from the pretrained generator:

$$
\begin{array} { r } { \mathcal { I } _ { \mathrm { p r e } } ( \theta ) : = \mathbb { E } _ { c \sim \mathcal { C } } \Big [ \mathbb { E } _ { S \sim p _ { \theta } ( \cdot \vert y ( c ) ) } \big [ \mathcal { L } _ { \mathrm { t a s k } } ( S ; c ) \big ] \Big ] . } \end{array}
$$

High-level intuition. When the trained diffusion model $p _ { \theta } ( \cdot \mid y )$ closely matches the repository/data distribution $p _ { \mathrm { d a t a } } ( \cdot \mid y )$ in distributional distance, the expected task loss under $p _ { \theta }$ is close to the expected repository task loss under $p _ { \mathrm { d a t a } }$ . If the repository was constructed to contain useful prompts for downstream tasks $( \mathrm { i . e . , } p _ { \mathrm { d a t a } }$ has low expected task loss), then a small distributional discrepancy implies low expected task loss for $p _ { \theta }$ as well. We make this statement precise with the following bounds.

## A.1.1 Distributional discrepancy bound

We begin with a straightforward decomposition and use standard total-variation and Pinsker inequalities (Cover and Thomas, 2006; Tsybakov, 2008).

Proposition 1 (Distributional discrepancy bound). Let $p _ { \mathrm { d a t a } } ( \cdot \mid y )$ and $p _ { \theta } ( \cdot \mid y )$ be two distributions on prompt space P for a fixed condition y. Assume the task loss $\mathcal { L } _ { \mathrm { t a s k } } ( x )$ is bounded in $[ 0 , L _ { \mathrm { m a x } } ]$ Then

$$
\begin{array} { r l } { \Big | \mathbb { E } _ { S \sim p _ { \theta } } \big [ \mathcal { L } _ { \mathrm { t a s k } } ( x ) \big ] - \mathbb { E } _ { S \sim p _ { \mathrm { d a t a } } } \big [ \mathcal { L } _ { \mathrm { t a s k } } ( x ) \big ] \Big | } & { } \\ { \leq 2 L _ { \operatorname* { m a x } } \mathrm { T V } \big ( p _ { \theta } , p _ { \mathrm { d a t a } } \big ) . } \end{array}\tag{8}
$$

and by Pinsker’s inequality,

$$
\begin{array} { r } { \mathrm { T V } ( p _ { \theta } , p _ { \mathrm { d a t a } } ) \leq \sqrt { \frac { 1 } { 2 } \operatorname { K L } ( p _ { \mathrm { d a t a } } \| p _ { \theta } ) } . } \end{array}\tag{9}
$$

Consequently,

$$
\begin{array} { r l } & { \mathbb { E } _ { S \sim p _ { \theta } } \left[ \mathcal { L } _ { \mathrm { t a s k } } ( x ) \right] \le \mathbb { E } _ { S \sim p _ { \mathrm { d a t a } } } \left[ \mathcal { L } _ { \mathrm { t a s k } } ( x ) \right] } \\ & { ~ + ~ { \cal L } _ { \mathrm { m a x } } \sqrt { 2 \mathrm { K L } ( p _ { \mathrm { d a t a } } \| p _ { \theta } ) } . } \end{array}\tag{10}
$$

Proof. Let $\ell ( x ) = \mathcal { L } _ { \mathrm { t a s k } } ( x )$ and denote $\Delta : = \mathbb { E } _ { p _ { \theta } } [ \ell ] - \mathbb { E } _ { p _ { \mathrm { d a t a } } } [ \ell ]$ . By the definition of total variation (and the fact that $0 \leq \ell \leq L _ { \mathrm { m a x } } )$

$$
\begin{array} { r l r } {  { | \Delta | = \Big | \int \ell ( x ) ( p _ { \theta } - p _ { \mathrm { d a t a } } ) ( d x ) \Big | } } \\ & { } & { \leq \int | \ell ( x ) | | p _ { \theta } - p _ { \mathrm { d a t a } } | ( d x ) } \\ & { } & { \leq L _ { \operatorname* { m a x } } \int | p _ { \theta } - p _ { \mathrm { d a t a } } | ( d x ) . } \end{array}\tag{11}
$$

By definition $\begin{array} { r } { \mathrm { T V } ( p _ { \theta } , p _ { \mathrm { d a t a } } ) = \frac { 1 } { 2 } \int | p _ { \theta } - p _ { \mathrm { d a t a } } | } \end{array}$ , hence (8) holds.

Pinsker’s inequality (see e.g. (Cover and Thomas, 2006)) gives (9). Combining the two inequalities yields (10). □

Remarks. Inequality (10) reduces expected downstream loss under the model to two terms: the expected loss of the repository distribution (which is a function of data collection quality) and the KL divergence between the dataset distribution and the pretrained generator. The latter is controlled by how well the denoiser is trained (see next subsection).

## A.1.2 Relating denoising loss to model-data KL

We now recall a standard link between the denoising objective used in DDPMs and a divergence between the model and data distributions (see, e.g., (Vincent, 2011; Song et al., 2020)). Under standard DDPM assumptions and appropriate variance schedule, minimizing the simplified denoising loss (2) is equivalent (up to constants and time discretization effects) to score matching / denoising score-matching which estimates the score function $\nabla _ { x } \log p _ { \mathrm { d a t a } } ( x )$ . A well-trained denoiser implies an accurate score estimator, which in turn implies a small $\mathrm { K L }$ divergence between the model and data distributions in x -space (the space of clean prompts). Formally, one can show:

Lemma 1 (Denoising loss controls KL (informal)). Under the standard DDPM/score-matching correspondence and mild regularity conditions, ifthe denoising risk satisfies

$$
{ \mathcal { L } } _ { \mathrm { d e n o i s e } } ( \theta ) \leq \varepsilon ,
$$

then the KL divergence between $p _ { \mathrm { d a t a } } ( x _ { 0 } \mid y )$ and $p _ { \theta } ( x _ { 0 } \mid y )$ admits the bound

$$
\begin{array} { r } { \mathrm { K L } \bigl ( p _ { \mathrm { d a t a } } ( \cdot \mid y ) \bigr | \bigl | p _ { \theta } ( \cdot \mid y ) \bigr ) \le C \varepsilon + o ( 1 ) , } \end{array}
$$

where $C > 0$ is a constant that depends on the variance schedule, the discretization, and model parametrization; the $o ( 1 )$ term vanishes as the diffusion discretization becomesfiner.

Remarks. The lemma is qualitative: precise constants follow from score-matching and likelihood bounds in the DDPM literature $( \mathrm { e . g . }$ , (Ho et al., 2020b; Song et al., 2020)). The key message is that a small denoising loss implies a small divergence between the learned and data distributions.

Combining Lemma 1 with Proposition 1 yields a bound of the form

$$
\begin{array} { r l } {  { \mathbb { E } _ { S \sim p _ { \theta } } \bigl [ \mathcal { L } _ { \mathrm { t a s k } } ( x ) \bigr ] \le \mathbb { E } _ { S \sim p _ { \mathrm { d a t a } } } \bigl [ \mathcal { L } _ { \mathrm { t a s k } } ( x ) \bigr ] } \quad } & { } \\ & { \phantom { = } + L _ { \mathrm { m a x } } \sqrt { 2 C \varepsilon } + o ( 1 ) . } \end{array}\tag{12}
$$

## A.1.3 Concentration from finite repository

So far we have related the model expectation to the (population) data distribution. In practice we only train on finite R with n samples per condition y. Let $\widehat { p } _ { \mathrm { d a t a } }$ denote the empirical distribution formed by the repository. By Hoeffding’s inequality (or McDiarmid) (Hoeffding, 1963; McDiarmid, 1989), with probability at least $1 - \delta$ over the draw of the repository,

$$
\begin{array} { r l } & { \left| \mathbb { E } _ { S \sim \widehat { p } _ { \mathrm { d a t a } } } \big [ \mathcal { L } _ { \mathrm { t a s k } } ( x ) \big ] - \mathbb { E } _ { S \sim p _ { \mathrm { d a t a } } } \big [ \mathcal { L } _ { \mathrm { t a s k } } ( x ) \big ] \right| } \\ & { \phantom { \left| \sum _ { \ell = { \mathrm { d a t a } } } \widehat { p } _ { \mathrm { d a t a } } \right| } \leq L _ { \mathrm { m a x } } \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } . } \end{array}\tag{13}
$$

## A.1.4 DMP performance guarantee

Combining the above pieces yields the following performance guarantee for DMP.

Proposition 2 (DMP performance guarantee). Assume the denoising loss satisfies ${ \mathcal { L } } _ { \mathrm { d e n o i s e } } ( \theta ) \leq \varepsilon$ and the repository contains n i.i.d. prompts for the condition y. Then with probability at least $1 - \delta$ over the repository sample,

$$
\begin{array} { r l } { \mathbb { E } _ { S \sim p _ { \theta } ( \cdot | y ) } \big [ \mathcal { L } _ { \mathrm { t a s k } } ( x ) \big ] \le \mathbb { E } _ { S \sim \widehat { p } _ { \mathrm { d a t a } } ( \cdot | y ) } \big [ \mathcal { L } _ { \mathrm { t a s k } } ( x ) \big ] } & { } \\ { + L _ { \mathrm { m a x } } \sqrt { 2 C \varepsilon } } & { } \\ { + L _ { \mathrm { m a x } } \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } + o ( 1 ) , } \end{array}\tag{14}
$$

where $C$ is the constantfrom Lemma 1 and the $o ( 1 )$ term accounts for discretization error in the diffusion approximation.

Proof. Start from Proposition 1 applied to $p _ { \theta }$ and $p _ { \mathrm { d a t a } } .$

$$
\begin{array} { r } { \mathbb { E } _ { p _ { \theta } } [ \ell ] \leq \mathbb { E } _ { p _ { \mathrm { d a t a } } } [ \ell ] + 2 L _ { \operatorname* { m a x } } \sqrt { \frac { 1 } { 2 } \operatorname { K L } ( p _ { \mathrm { d a t a } } \| p _ { \theta } ) } . } \end{array}
$$

Apply Lemma 1 to bound the KL by $C \varepsilon + o ( 1 )$ ; hence

$$
\begin{array} { r } { \mathbb { E } _ { p _ { \theta } } [ \ell ] \leq \mathbb { E } _ { p _ { \mathrm { d a t a } } } [ \ell ] + L _ { \operatorname* { m a x } } \sqrt { 2 C \varepsilon } + o ( 1 ) . } \end{array}
$$

Now replace the population expectation $\mathbb { E } _ { p _ { \mathrm { d a t a } } } [ \ell ]$ by the empirical expectation $\mathbb { E } _ { \widehat { p } _ { \mathrm { d a t a } } } [ \ell ]$ and apply Proposition 13 (Hoeffding) which with probability at least 1 − δ yields the stated sampling error term $L _ { \operatorname* { m a x } } { \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } }$ . Combining these terms gives (14). □

Interpretation. Bound (14) decomposes the expected task loss under the pretrained generator into (i) the empirical repository loss (quality of collected prompts), (ii) an approximation term controlled by the denoising risk (how well DMP models the repository), and (iii) a sampling term that vanishes as the repository size n grows.

![](images/647bd2ddbe7a124d17b2ab329cfc76e28c94aac593ccbab7fb969e438570b437.jpg)  
Figure 4: Left: Diffusion Meta-Prompt framework for Text-to-Prompt synthesis. Right: Prompt Variation synthesis conditioned on learned Textual Inversion Prompts.

Figure 4 shows the DMP framework where the left part denotes the training of text-to-prompt diffusion model and right part denotes the diffusion model for generating prompt variations conditioned on the original prompts.

## A.2 Relation to Prior Methods

Table 10 illustrates how DMP differs from prior prompt-learning approaches across key axes such as requiring task-specific data or loss, supporting multiple tasks in a zero-shot manner, learning across multiple tasks and the need for storage of prompts. Classical prompt learning techniques are task and model specific, and require task specific data and losses. DMP instead learns the distribution of prompts from a prompt repository (produced by these prompt learning techniques). It can then be used to sample prompts for many tasks. Note that $D M P$ is not task specific and does not require task specific losses or data, just a prompt repository. This makes DMP as a general prompt generator, free from the data and optimization constraints that characterize prior prompt-learning techniques. We show that, in many cases, DMP sampled prompts even outperform prompt learning prompts. For example, the classification results of Table 19 shows that DMP outperforms the base method on the unseen classes of the very same dataset, despite never accessing a single image.

## A.3 DMP for Classification

Ablation on Steering Strength Table 11 shows the effect of the steering strength on composite classification accuracy. The trend suggests that moderate steering provides the best trade-off between stability and diversity during diffusion sampling.

Table 10: Conceptual differences between DMP and prior prompt learning methods. Unlike prior works that refine prompts using data for a specific task or example, DMP treats prompts as a distribution and trains a diffusion meta-model that amortizes prompt generation across many tasks and repositories. DMP does not require access to the original data used to train the prompts (L159–L161) and supports one-shot sampling, composition, and negative prompting without per-task optimization.
<table><tr><td>Method</td><td>Task Data/</td><td>Supports Loss Free Multiple Tasks across Tasks Storage</td><td>Learns</td><td>Minimal Core Idea</td><td></td></tr><tr><td colspan="6">Stylus / Retrieval-based 7</td></tr><tr><td colspan="6">Diffusion-based Methods</td></tr><tr><td>Prompt Diffusion</td><td>x</td><td>X</td><td>x</td><td>x</td><td>Refines prompts for a specific task using data.</td></tr><tr><td>Diff-Prompt Neural Network Diffusion</td><td>X x</td><td>X X</td><td>X X</td><td>X V</td><td>Mask-supervised diffusion for a single task. Diffusion to generate model weights</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>for a particular task/dataset (weight-generation).</td></tr><tr><td>Conditional LoRA Param Gen.</td><td>X</td><td>7</td><td>V</td><td>L</td><td>Diffusion/hypernetwork that generates LoRA adapters conditioned on task (Cond P-Diff / CondLoRA).</td></tr><tr><td>DiffLoRA</td><td>X</td><td>V</td><td>了</td><td>V</td><td>Diffusion model predicts personalized low-rank (LoRA) weights at inference (zero-shot personalization).</td></tr><tr><td colspan="6">Prompt Learning Methods</td></tr><tr><td>Hierarchical Variational TTP</td><td>X</td><td>x</td><td>X</td><td>X</td><td>Test-time variational prompt generator relying on task-specific data features.</td></tr><tr><td>Language-Aware Soft Prompting (LASP)</td><td>X</td><td>X</td><td>X</td><td>X</td><td>Text-conditioned optimization for soft prompts, task-specific.</td></tr><tr><td>Consistency-guided PL (CoPrompt)</td><td>x</td><td>x</td><td>X</td><td>X</td><td>Consistency loss improves prompt robustness; task-specific.</td></tr><tr><td>PromptKD</td><td>X</td><td>X</td><td>X</td><td>X</td><td>Distills prompts from teacher to student in unsupervised setting; task-specific.</td></tr><tr><td>Dual-Prompt Collaboration (DPC)</td><td>X</td><td>X</td><td>X</td><td>X</td><td>Dual-prompt collaboration requiring tuned prompt parameters; task-specific.</td></tr><tr><td>Bayesian Prompt Learning (BPL)</td><td>x</td><td>X</td><td>X</td><td>X</td><td>Bayesian uncertainty modeling over task-specific prompts.</td></tr><tr><td>Patch-Prompt Aligned Bayesian Prompt Tuning</td><td>X</td><td>X</td><td>X</td><td>X</td><td>Bayesian hierarchical prompt generation (label-specific stochastic prompts); per-task tuning.</td></tr><tr><td colspan="6">Meta-Learning Methods</td></tr><tr><td>Prompt Learning via Meta-Regularization (ProMetaR)</td><td>x</td><td>L</td><td>7</td><td>X</td><td>Meta-regularization to improve prompt generalization across tasks; requires training data.</td></tr><tr><td>Gradient-Regulated Meta-Prompt (GRAM)</td><td>X</td><td>L</td><td>V</td><td>X</td><td>Meta-learned prompt init + gradient regulator for few-shot cross-domain generalization.</td></tr><tr><td>PRewrite (Prompt Rewriting w/ RL)</td><td>X</td><td>L</td><td>x</td><td>X</td><td>LLM-based prompt rewriter trained with RL to improve downstream task performance.</td></tr><tr><td>DMP (Ours)</td><td>L</td><td>V</td><td>L</td><td>L</td><td>Learns a prompt distribution; one-shot sampling, composition, negative prompts.</td></tr></table>

Table 11: Ablation of steering strength λ for DMP on 55 composite dataset-pairs classification. Moderate steering gives the strongest average accuracy, while overly aggressive steering begins to over-constrain the sampling trajectory.
<table><tr><td>λ</td><td>All</td><td>Base</td><td>New</td><td>Avg.</td></tr><tr><td>0.1</td><td>62.5</td><td>68.5</td><td>69.8</td><td>66.9</td></tr><tr><td>0.2</td><td>63.1</td><td>69.4</td><td>70.8</td><td>67.8</td></tr><tr><td>0.3</td><td>62.8</td><td>69.1</td><td>70.4</td><td>67.4</td></tr><tr><td>0.4</td><td>62.2</td><td>68.4</td><td>69.6</td><td>66.7</td></tr><tr><td>0.5</td><td>61.7</td><td>67.8</td><td>68.9</td><td>66.1</td></tr><tr><td>0.6</td><td>61.6</td><td>67.9</td><td>69.1</td><td>66.2</td></tr><tr><td>0.7</td><td>61.5</td><td>67.6</td><td>68.7</td><td>65.9</td></tr><tr><td>0.8</td><td>61.3</td><td>67.2</td><td>68.3</td><td>65.6</td></tr><tr><td>0.9</td><td>61.0</td><td>66.8</td><td>67.9</td><td>65.2</td></tr><tr><td>1.0</td><td>60.7</td><td>66.4</td><td>67.5</td><td>64.9</td></tr></table>

Composite classification with other prompt-learning methods. We further evaluate DMP on the composite classification task using recent prompt-learning methods, including MaPLe (khattak et al., 2023) and TAC (Hao et al., 2025). The results are reported as the average accuracy over the 55 pairwise dataset compositions defined in Sec. 4.1.

Table 18 shows that DMP yields consistent improvements across prompt-learning backbones. The gains are largest for CoOp since we learn the full prompt distribution using a diffusion model. For other prompt learning methods such as MaPLe and TAC, the gains are smaller but remain consistent, reaching up $\mathrm { 1 0 + 1 . 2 / + 0 . 6 }$ points for unseen classes for MaPLe/TAC respectively. This is expected because these methods learn additional parameters beyond the soft prompts, such as projection matrices or task-aware pre-context modules, which are too high-dimensional to synthesize with the current DMP implementation and available training resources. In these cases, DMP only synthesizes the prompt parameters while keeping the method-specific auxiliary parameters fixed. Extending

Table 12: Domain generalization accuracy (%) for DMPCoOp prompts sampled on ImageNet.
<table><tr><td colspan="3">Source</td><td colspan="3">Target</td></tr><tr><td>Method</td><td>ImageNet -V2</td><td>-S</td><td>-A</td><td>-R</td><td>Average</td></tr><tr><td>CoOp</td><td>68.5</td><td>61.2 45.2 48.2 74.1</td><td></td><td></td><td>59.4</td></tr><tr><td>DMPCoOp</td><td>68.7</td><td>62.1 46.5 50.6 75.1</td><td></td><td></td><td>60.6</td></tr></table>

Table 13: Domain generalization accuracy (%) for DMPTAC prompts sampled on ImageNet.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Source ImageNet -V2</td><td colspan="3">Target</td></tr><tr><td>-S</td><td>-A -R</td><td>Average</td></tr><tr><td>TAC</td><td>71.3</td><td>64.6 48.3 48.6 76.6</td><td></td><td>61.9</td></tr><tr><td>DMPTAC</td><td>71.4</td><td></td><td>64.7 48.5 48.8 76.7</td><td>62.0</td></tr></table>

Table 14: Cross-task generalization per dataset: performance for Mean Treecut Accuracy (MTA).
<table><tr><td>Method</td><td>ImageNet</td><td>-V2</td><td>-S</td><td>-R</td><td>-A</td><td>SUN397</td></tr><tr><td>TAC</td><td>71.3</td><td>34.1</td><td>28.3</td><td>5.5</td><td>36.5</td><td>30.0</td></tr><tr><td>DMPTAC</td><td>71.4</td><td>39.6</td><td>37.3</td><td>5.7</td><td>38.9</td><td>40.0</td></tr><tr><td>∆</td><td>+0.1</td><td>+5.5</td><td>+9.0</td><td>+0.2</td><td>+2.4</td><td>+10.0</td></tr></table>

Table 15: Comparison of DMPMulti with TI prompts.

<table><tr><td>Method (SD-RV) Face-ID↑ DINO↓ CLIP-I↓ CLIP-T ↑</td></tr><tr><td>Textual Inversion 0.428 0.627</td></tr><tr><td>0.696 0.244 Stylus (Top-1) 0.434 0.645 0.706 0.246</td></tr><tr><td>DMPMulti 0.434 0.558 0.653 0.245</td></tr><tr><td>DMPMulti(20) 0.429 0.595 0.599 0.285</td></tr></table>

Table 16: Comparison of slider prompts generated by a separate DMP (DMPSlider) against DMPMulti.
<table><tr><td>Method</td><td>|CLIP-s ↑ |LPIPS↓</td></tr><tr><td>Prompt Slider</td><td>30.00 0.219</td></tr><tr><td>DMPMulti 29.86</td><td>0.126</td></tr><tr><td>DMPSlider (separate)</td><td>29.88 0.121</td></tr></table>

Table 17: Comparison of separately trained DMP (DMPIdentity) with Textual Inversion and DMPMulti for identity synthesis
<table><tr><td>Method (SD-RV)</td><td>Face-ID↑ DINO ↓</td><td></td><td>CLIP-I↓</td><td>CLIP-T ↑</td></tr><tr><td>Textual Inversion</td><td>0.428</td><td>0.627</td><td>0.696</td><td>0.244</td></tr><tr><td>DMPMulti</td><td>0.434</td><td>0.558</td><td>0.653</td><td>0.245</td></tr><tr><td>DMPIdentity (separate)</td><td>0.435</td><td>0.550</td><td>0.659</td><td>0.246</td></tr></table>

DMP to jointly synthesize these higher-dimensional components would likely require compression or low-rank parameterization strategies, which we leave for future work.

Same dataset (Base to New generalization). In this setting, prompts are trained in the Base classes of each dataset and evaluated on all classes of the same dataset. This is the least demanding setting, since the prompts have to generalize only to unseen classes of the dataset where they were trained.

Table 19 summarizes the performance of PL and DMP prompts for four prompt learning methods, CoOp, CoPrompt, Maple, and TAC over the baselines across 11 datasets for Base-to-novel class generalization. The reported results are the average over three seed runs. As expected, DMP prompts have slightly lower performance for the base classes, since they are inherently bounded by the accuracy of the PL prompts used to train the DMP. Further, Proposition 1 suggests that the gap could be bridge by using more than 40 prompts per dataset to train the DMP model, we show that this is true by ablating the number of prompts used for training the DMP in Table 27. The table shows that, even under these conditions, the DMP prompts generalize better than the baseline, achieving an average accuracy gain between +0.4% and 4.8% for New (unseen) classes, across prompting methods. For new classes, DMP outperforms CoOp on 11/11 datasets for all methods listed on the table. The table details the results for three datasets for brevity.

With regards to prompting methods, the table shows that the DMP gains are larger for CoOp, then Co Prompt, MaPLe and finally TAC. This is expected because DMP only synthesizes prompts. However, all methods other than CoOp, complement these prompts with additional learned weights or projection matrices. For these methods, we used the original learned weights together with the prompts synthesized by DMP. Despite only modifying prompts, DMP achieves gains of +0.4-0.8% over these methods for unseen classes. While a DMP-type technique could be used to sample the additional weights, this is a more complex endeavor due to the high dimensionality of the latter and compute requirements. We leave the synthesis of large parameter matrices for future work.

In any case, it can be concluded that, even on the same dataset setting, meta-prompting with a diffusion model outperforms the existing prompting techniques that are used to train it.

Table 18: Composite classification accuracy (%) averaged over 55 dataset pairs. DMP yields consistent improvements across prompt-learning methods. Gains are largest for CoOp, which only learns soft prompts, and smaller for MaPLe and TAC, which include additional high-dimensional learned components that are kept fixed in our current implementation.
<table><tr><td>Method</td><td>All</td><td>Base</td><td>New</td></tr><tr><td>CoOp</td><td>58.7</td><td>65.1</td><td>65.1</td></tr><tr><td>DMPCoOp</td><td>63.1</td><td>69.4</td><td>70.8</td></tr><tr><td>∆</td><td>+4.4</td><td>+4.3</td><td>+5.7</td></tr><tr><td>MaPLe DMPMaPLe</td><td>57.2</td><td>64.3 64.7</td><td>63.0 64.2</td></tr><tr><td>∆</td><td>57.9 +0.7</td><td>+0.4</td><td>+1.2</td></tr><tr><td>TAC</td><td></td><td></td><td></td></tr><tr><td>DMPTAC</td><td>63.2</td><td>69.6</td><td>69.7</td></tr><tr><td></td><td>63.5</td><td>70.0</td><td>70.4</td></tr><tr><td>∆</td><td>+0.3</td><td>+0.4</td><td>+0.6</td></tr></table>

These findings suggest that the original learned prompts may be somewhat overfitted to the base classes, whereas DMP prompts induce a classifier of stronger generalization. Over all classes (both seen and unseen) and datasets, DMP has an accuracy gain of upto +3.0%. It can be concluded that meta-prompting with a diffusion model outperforms the existing prompting techniques that are used to train it.

Table 19: Base2new generalization per dataset: performance for All, Base, and New classes, and HM (Harmonic Mean). The results reported are the average over three seed runs. The Average column reports the average over 11 datasets. \* denotes our implementation as the code is not publicly available.
<table><tr><td rowspan="2">Method</td><td colspan="4">(a) Average</td><td colspan="4">(b) ImageNet</td><td colspan="4">(c) Caltech101</td><td colspan="4">(d) OxfordPets</td></tr><tr><td>All</td><td>Base</td><td>New</td><td>HM</td><td>All</td><td>Base</td><td>New</td><td>HM</td><td>All</td><td>Base</td><td>New</td><td>HM</td><td>All</td><td>Base</td><td>New</td><td>HM</td></tr><tr><td>VAE (Kingma and Welling, 2014)</td><td>58.6</td><td>62.7</td><td>68.3</td><td>65.0</td><td>57.0</td><td>64.1</td><td>57.8</td><td>60.8</td><td>89.1</td><td>91.7</td><td>94.3</td><td>92.9</td><td>90.1</td><td>92.0</td><td>97.1</td><td>94.5</td></tr><tr><td>Top-1 Retrieval (Luo et al., 2024)</td><td>67.5</td><td>80.7</td><td>72.8</td><td>76.5</td><td>50.5</td><td>78.4</td><td>47.9</td><td>59.5</td><td>37.3</td><td>76.9</td><td>46.4</td><td>57.9</td><td>69.9</td><td>85.8</td><td>66.8</td><td>75.1</td></tr><tr><td>Transformer (Vaswani et al., 2017)</td><td>68.3</td><td>82.3</td><td>69.4</td><td>75.3</td><td>68.1</td><td>76.4</td><td>66.8</td><td>71.3</td><td>94.8</td><td>98.2</td><td>94.8</td><td>96.5</td><td>89.9</td><td>94.6</td><td>95.4</td><td>95.0</td></tr><tr><td>GPT-2 (Radford et al., 2019)</td><td>68.1</td><td>81.8</td><td>69.1</td><td>74.9</td><td>68.0</td><td>76.2</td><td>66.8</td><td>71.2</td><td>94.8</td><td>98.3</td><td>95.0</td><td>96.6</td><td>90.1</td><td>94.7</td><td>95.9</td><td>95.3</td></tr><tr><td>Prompt Diffusion* (Du et al., 2024)</td><td>67.0</td><td>69.9</td><td>73.0</td><td>71.4</td><td>54.7</td><td>60.6</td><td>56.4</td><td>58.4</td><td>91.3</td><td>94.4</td><td>93.2</td><td>93.8</td><td>92.3</td><td>94.3</td><td>97.5</td><td>95.9</td></tr><tr><td>CoOp (Zhou et al., 2022b)</td><td>67.1</td><td>80.9</td><td>68.6</td><td>74.2</td><td>68.0</td><td>76.2</td><td>66.8</td><td>371.2</td><td>93.8</td><td>98.1</td><td>93.0</td><td>95.5</td><td>90.6</td><td>95.2</td><td>96.5</td><td>95.8</td></tr><tr><td>DMPCoOp ∆</td><td>70.1</td><td>80.3</td><td>73.4</td><td>76.5</td><td>68.7</td><td>75.4</td><td>68.8</td><td>72.0</td><td>94.8</td><td>98.3</td><td>95.3</td><td>96.8</td><td>91.7</td><td>95.4</td><td>97.3</td><td>96.3</td></tr><tr><td></td><td>+3.0</td><td>-0.6</td><td>+4.8</td><td>+2.3</td><td>+0.7</td><td>-0.8</td><td>+2.0</td><td>+0.8</td><td>+1.0</td><td>+0.2</td><td>+2.3</td><td>+1.3</td><td>+1.1</td><td>+0.2</td><td>+0.8</td><td>+0.5</td></tr><tr><td>CoPrompt (Roy and Etemad, 2024)</td><td>72.3</td><td>83.1</td><td>74.6</td><td>78.3</td><td>70.7</td><td>76.7</td><td>71.4</td><td>73.9</td><td>95.8</td><td>98.7</td><td>95.3</td><td>97.0</td><td>91.1</td><td>95.3</td><td>97.0</td><td>96.1</td></tr><tr><td>DMPCoPrompt ∆</td><td>72.9</td><td>82.5</td><td>75.4</td><td>78.5</td><td>70.7</td><td>76.6</td><td>71.5</td><td>73.9</td><td>95.7</td><td>98.7</td><td>95.4</td><td>97.0</td><td>91.6</td><td>95.2</td><td>97.2</td><td>96.2</td></tr><tr><td></td><td>+0.6</td><td>-0.6</td><td>+0.8</td><td>+0.2</td><td>0.0</td><td>-0.1</td><td>+0.1</td><td>0.0</td><td>-0.1</td><td>0.0</td><td>+0.1</td><td>0.0</td><td>+0.5</td><td>-0.1</td><td>+0.2</td><td>+0.1</td></tr><tr><td>Maple (khattak et al., 2023)</td><td>72.0</td><td>82.2</td><td>75.1</td><td>78.2</td><td>70.2</td><td>76.7</td><td>70.5</td><td>73.5</td><td>94.5</td><td>98.0</td><td>94.3</td><td>96.1</td><td>92.3</td><td>95.4</td><td>97.8</td><td>96.6</td></tr><tr><td>DMPMaple</td><td>72.4</td><td>82.0</td><td>75.9</td><td>78.6</td><td>70.3</td><td>76.8</td><td>70.6</td><td>73.6</td><td>94.8</td><td>98.0</td><td>95.5</td><td>96.7</td><td>92.3</td><td>95.4</td><td>97.6</td><td>96.5</td></tr><tr><td>∆</td><td>+0.4</td><td>-0.2</td><td>+0.8</td><td>+0.4</td><td>+0.1</td><td>+0.1</td><td>+0.1</td><td>+0.1</td><td>+0.3</td><td>0.0</td><td>+1.2</td><td>+0.6</td><td>0.0</td><td>0.0</td><td>-0.2</td><td>-0.1</td></tr><tr><td>TAC (Hao et al., 2025)</td><td>74.6</td><td>85.2</td><td>77.1</td><td>80.8</td><td>71.3</td><td>78.5</td><td>71.0</td><td>74.6</td><td>95.2</td><td>98.6</td><td>95.0</td><td>96.7</td><td>93.1</td><td>96.0</td><td>98.0</td><td>97.0</td></tr><tr><td>DMPTAC</td><td>75.0</td><td>85.1</td><td>77.5</td><td>80.9</td><td>71.4</td><td>78.5</td><td>71.2</td><td>74.6</td><td>95.2</td><td>98.6</td><td>95.0</td><td>96.8</td><td>93.1</td><td>95.9</td><td>98.2</td><td>97.0</td></tr><tr><td>∆</td><td>+0.4</td><td>-0.1</td><td>+0.4</td><td>+0.1</td><td>+0.1</td><td>0.0</td><td>+0.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>+0.1</td><td>0.0</td><td>-0.1</td><td>+0.2</td><td>0.0</td></tr></table>

Table 21 shows the full results of cross-task generalization experiment described in section 4.1. We note that DMPCoOp obtains higher accuracies consistently across both the Mean Treecut Accuracy and Hierarchical Consistency Accuracy metrics on all the datasets considered. Notably, DMPCoOp obtains +11.4% and +5% improvement on SUN dataset for MTA and HCA respectively. This shows that DMPCoOp prompts are more robust and generalize well beyond the task on which it was trained.

We further provide the results of cross-domain experiments for TAC method in Table 13, and crosstask generalization in Table 14. The gains over baseline TAC are larger for cross-task generalization where DMPTAC prompts obtains +10% on SUN and +9% on Imagenet-Sketch datasets for hierarchical classification.

t-SNE visualization. To assess the fidelity and diversity of prompts synthesized by the DMP framework, we conducted an embedding space analysis comparing real prompts from the repository R with DMP-generated prompts sampled using different random noise seeds. Figure 5 presents a twodimensional t-SNE projection of both sets of embeddings, with real prompts in blue and generated prompts in orange for SUN397, FGVC, Oxford Flowers and Stanford Cars datasets respectively from left to right.

The visualization reveals that generated prompts broadly overlap with the real prompt manifold, while also exhibiting a greater spread, indicative of higher variability. This suggests that the model captures the underlying structure of prompt space without resorting to memorization, while also producing novel variations.

![](images/f201a7bb61c1c3acf507cfdddb7912601eda335858c25b4c81f769bf734f85d8.jpg)

![](images/088efbc35d27ff3f68149c1bb484b1d074b9995f7c225dcbc1565fcb45326d7a.jpg)

![](images/d6e639a7ef7a37def401a7d7df0ec9ba1f10df468b56b7ccda2d0649ae588fb3.jpg)

![](images/826e98d0960acc374a2d2abfa5eb16851b3996d09693aab5506364bd89957899.jpg)  
Figure 5: t-SNE visualizations of prompt embeddings. Each figure shows real prompts (blue) and DMP-generated prompts (orange) with different noise seeds for different datasets: (a) SUN397, (b) FGVC-Aircraft, (c) Oxford Flowers, (d) Stanford Cars. Generated prompts broadly overlap with the real prompt manifolds while exhibiting greater spread, demonstrating both fidelity and diversity across domains.

To quantify these observations, we computed the fraction of a prompt’s 5 nearest neighbors (in embedding space) that share the same class label between real and generated prompts across 11 datasets. On average, 70.7% of generated prompts share the same nearest-neighbor labels as their real counterparts, showing strong semantic alignment between real and generated embeddings while ensuring diversity.

The clusters that are seen in the plots show that the prompts sampled by DMP are more consistent than those originally learned by the prompt learning method. This could explain the improved generalization of these prompts and the lower variance of the performance of the prompted model to different noise seeds. We observed a standard deviation of 0.7% from the mean accuracy for prompts sampled by DMPCoOp with 200 different noise seeds. In contrast, the standard deviation of the baseline prompt learning method used to build the repository from different initialization seeds show a higher variation of 4.7%.

We additionally evaluated five prompts generated from different random seeds on the combined Caltech101–ImageNet task and measured prediction disagreement, per-class accuracy correlation, and error-set overlap. Table 20 shows that, across the five prompts, the standard deviation of accuracy is only 0.09 percentage points (range: 0.2 points). The per-class accuracy correlation of 0.999, and the error set overlap of 98.66% indicate highly consistent decision boundaries across random seeds. On the other hand, the fact that different sampled prompts disagree on approximately 1.08% of individual predictions shows that they are not identical. Overall, these results indicate that diffusion sampling yields slight local variations while preserving essentially the same functional behavior and generalization performance.

Table 20: Agreement statistics across different random seeds.
<table><tr><td>Prompt pair</td><td>Accuracy (%)</td><td>Prediction disagreement (%)</td><td>Per-class accuracy correlation</td><td>Error-set overlap (%)</td></tr><tr><td>Seed 1 vs. Seed 2</td><td>59.4 / 59.5</td><td>1.01</td><td>0.9992</td><td>98.81</td></tr><tr><td>Seed 1 vs. Seed 3</td><td>59.4 / 59.5</td><td>1.12</td><td>0.9991</td><td>98.68</td></tr><tr><td>Seed 1 vs. Seed 4</td><td>59.4 / 59.3</td><td>1.52</td><td>0.9982</td><td>97.97</td></tr><tr><td>Seed 1 vs. Seed 5</td><td>59.4 / 59.5</td><td>0.97</td><td>0.9992</td><td>98.78</td></tr><tr><td>Seed 2 vs. Seed 3</td><td>59.5 / 59.5</td><td>0.79</td><td>0.9994</td><td>99.07</td></tr><tr><td>Seed 2 vs. Seed 4</td><td>59.5 / 59.3</td><td>1.43</td><td>0.9985</td><td>98.19</td></tr><tr><td>Seed 2 vs. Seed 5</td><td>59.5 / 59.5</td><td>0.71</td><td>0.9995</td><td>99.18</td></tr><tr><td>Seed 3 vs. Seed 4</td><td>59.5 / 59.3</td><td>1.21</td><td>0.9986</td><td>98.44</td></tr><tr><td>Seed 3 vs. Seed 5</td><td>59.5 / 59.5</td><td>0.79</td><td>0.9994</td><td>99.05</td></tr><tr><td>Seed 4 vs. Seed 5</td><td>59.3 / 59.5</td><td>1.19</td><td>0.9987</td><td>98.43</td></tr><tr><td>Mean</td><td>59.44</td><td>1.08</td><td>0.9990</td><td>98.66</td></tr></table>

Impact of Classifier-Free Guidance (CFG) scales. The performance of DMP with different guidance values is shown in Table 22. The trends show that the average accuracy across all datasets decreases with increasing guidance scales.

Variational Autoencoder Reconstruction. Table 23 shows the reconstruction accuracy of the CoOp prompts for the trained Variational Autoencoder (VAE) across different datasets. The autoencoder reconstructs the prompts almost perfectly with only 0.1% difference on average across all the datasets.

Table 21: Cross-task generalization per dataset: performance for Mean Treecut Accuracy (MTA) (Wu et al., 2024), and Hierarchical Consistency Accuracy (HCA) (Wu et al., 2024).
<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet</td><td colspan="2"> $\overline { { - \mathbf { V } 2 } }$ </td><td colspan="2"> $\overline { { \mathbf { \nabla } - \mathbf { S } } }$ </td><td colspan="2">-R</td><td colspan="2">-A</td><td colspan="2">SUN397</td></tr><tr><td>MTA</td><td>HCA</td><td>MTA</td><td>HCA</td><td>MTA</td><td>HCA</td><td>MTA</td><td>HCA</td><td>MTA</td><td>HCA</td><td>MTA</td><td>HCA</td></tr><tr><td>CoOp</td><td>40.7</td><td>0.8</td><td>38.9</td><td>0.8</td><td>34.4</td><td>0.5</td><td>57.7</td><td>9.9</td><td>43.8</td><td>4.1</td><td>19.0</td><td>31.3</td></tr><tr><td>DMPCoOp</td><td>47.4</td><td>2.6</td><td>45.0</td><td>2.1</td><td>40.2</td><td>2.0</td><td>57.7</td><td>18.7</td><td>55.5</td><td>5.5</td><td>30.4</td><td>36.3</td></tr><tr><td>∆</td><td>+6.7</td><td>+1.8</td><td>+6.1</td><td>+1.3</td><td>+5.8</td><td>+1.5</td><td>0.0</td><td>+8.8</td><td>+11.7</td><td>+1.3</td><td>+11.4</td><td>+5.0</td></tr></table>

Table 22: Comparison with different classifier free guidance scales. Base2new generalization per dataset: performance for All, Base, and New classes, and HM (Harmonic Mean).
<table><tr><td></td><td colspan="4">(a) Average (scale 4.5)</td><td colspan="4">(b) Average (scale 5.5)</td><td colspan="4">(c) Average (scale 6.5)</td><td colspan="4">(d) Average (scale 7.5)</td></tr><tr><td>Method</td><td>All</td><td>Base</td><td>New</td><td>HM</td><td>All</td><td>Base</td><td>New</td><td>HM</td><td>All</td><td>Base</td><td>New</td><td>HM</td><td>All</td><td>Base</td><td>New</td><td>HM</td></tr><tr><td>DMPCoOp</td><td>70.1</td><td>80.3</td><td>73.4</td><td>76.5</td><td></td><td>69.6</td><td>80.3 70.6</td><td>76.3</td><td></td><td>69.6 79.3</td><td>70.4</td><td>75.0</td><td></td><td>68.1</td><td>79.4 70.1</td><td>75.0</td></tr></table>

## A.4 Multi-task DMP

DMPMulti is a multi-task DMP model trained to synthesize prompts for personalized subjects and prompt sliders. Both tasks are unified within a single framework through the shared CLIP-ViT L/16 text encoder.

Prompt Slider Repository. Prompt Sliders is a technique to learn text prompts in the CLIP text embedding space that allow control of particular attributes of a designated concept. The prompt is learned through the optimization

$$
S ^ { * } = \arg \operatorname* { m i n } _ { S } \mathbb { E } _ { x \sim \mathcal { E } ( x ) , y , \epsilon \sim \mathcal { N } ( 0 , 1 ) , t } \left\| \epsilon _ { t } - \epsilon _ { \theta } ^ { d } ( x _ { t } , \tau _ { \theta } ( y , S ) , t ) \right\| _ { 2 } ^ { 2 }\tag{15}
$$

where $x _ { t }$ is an image, and $y$ is any additional prompt text, such as $\mathbf { \ddot { a } }$ photo of $\mathbf { a } "$ . Both $\tau _ { \theta }$ and $\epsilon _ { \theta }$ are fixed during the optimization. Given a target concept $c _ { t } .$ , a prompt slider $S ^ { * }$ is learned to encourage the distribution of images of $c _ { t }$ to exhibit more positive attributes $c ^ { + }$ and fewer negative attributes $c ^ { - }$ This is implemented by replacing $\epsilon _ { t }$ with

$$
\epsilon _ { t } ( \alpha ) = \epsilon _ { \theta } ^ { d } ( x _ { t } , \tau _ { \theta } ( y ( c _ { t } ) ) , t ) + \alpha \eta \sum _ { p \in P } ( \epsilon _ { \theta } ^ { d } ( x _ { t } , \tau _ { \theta } ( y ( c ^ { + } , p ) ) , t ) - \epsilon _ { \theta } ^ { d } ( x _ { t } , \tau _ { \theta } ( y ( c ^ { - } , p ) ) , t ) )\tag{16}
$$

and S by $\alpha S$ in (15), where $\eta$ is a guidance scale, α a scaling parameter, $y ( c _ { t } )$ the concept name, and $P$ a set of concepts that the attribute manipulation should preserve (for example, race or gender). The positive $c ^ { + }$ , and negative $c ^ { - }$ attributes are sampled from a template predefined for concept $c _ { t }$ . To create a slider repository $\mathcal { R } _ { S }$ , we trained prompt sliders for 20 different concepts using the SD-XL model and 3, 000 backpropagation steps. For each concept $c ,$ we save 40 prompts $S ( c )$ (from the last 200 steps in intervals of 5) to obtain a training tensor $\mathcal { R } _ { S } \in \mathbb { R } ^ { 2 0 \times 4 0 \times d }$

Prompt Identity Repository. We use a random subset of 20 identities from the 3000 celebrity faces in the DMPVariation prompt repository to create a prompt identity repository $\mathcal { R } _ { I }$ with each identity containing 40 prompts obtained from the last 200 steps in intervals of 5 to obtain a training tensor $\mathcal { R } _ { I } \in \mathbb { R } ^ { 2 \breve { 0 } \times 4 0 \times d }$

Training. The DMPMulti model was trained directly in prompt space P, i.e. without autoencoder, using the full set of prompts from both repositories, i.e. $\mathcal { R } = \mathcal { R } _ { S } \cup \mathcal { R } _ { I }$ . For each slider concept c (e.g., age, smiling etc.) we use the concept name as the text condition for DMPMulti. For identities, we use “identity $- \bar { c } "$ as the text condition where $c = 1 , \ldots , 2 0$ is associated with each identity c in $\mathcal { R }$

Inference. At inference, DMPMulti can be conditioned with the slider concept name, to generate prompt sliders, or with “identity-c" to generate identity prompts. Novel identities can be sampled by specifying a new identity "identity $- c "$ with $c > 2 0$ . Slider and identity prompts can then be fed to the downstream model in isolation or together.

Table 15 compares Identity prompts from DMPMulti with Stylus and Textual Inversion across face recognition accuracy, image-to-image similarity, and prompt fidelity. DMPMulti achieves higher identity scores, maintains prompt compliance comparable to TI/Stylus, and shows lower similarity to training images, indicating stronger generalization. In contrast, TI tends to overfit, consistent with prior findings (Yuan et al., 2023). The last row of Table 15 shows an ablation study of using fewer samples (20 vs 40) per identity for training DMP. Notably, DMPMulti (20) achieves similar identity fidelity and even higher prompt compliance (↑17%) than the 40-sample variant.

Figure 7 shows qualitative results of slider prompts generated by DMP for the attributes "long hair" and "chubby".

Table 23: Quantitative results of VAE reconstruction: Comparison against CoOp prompts across various datasets. The VAE reconstructs the CoOp prompts baseline almost perfectly with only 0.1% difference on average.
<table><tr><td></td><td>ImageNet</td><td>Oxford flowers</td><td>Oxford pets Stanford cars</td><td>Caltech 101</td><td>Food101</td><td>FGVC Aircraft</td><td>SUN397</td><td>DTD</td><td>EuroSAT</td><td>UCF101</td><td>Average</td></tr><tr><td>Model CoOp Autoencoder</td><td>68.0 68.0</td><td>72.6 90.1 72.0 89.6</td><td>68.7 68.6</td><td>94.8 94.6</td><td>84.9 85.1</td><td>25.1 25.0</td><td>67.6 67.0</td><td>51.9 52.0</td><td>58.9 59.5</td><td>66.2 66.1</td><td>68.1 68.0</td></tr><tr><td>Δ</td><td>0.0 -0.6</td><td>-0.5</td><td>-0.1</td><td>-0.2</td><td>+0.2</td><td>-0.1</td><td>-0.6</td><td>+0.1</td><td>+0.6</td><td>-0.1</td><td>-0.1</td></tr><tr><td colspan="10">Table 24: Ablation study on Cross-dataset generalization of DMPCoOp Imagenet prompts: Comparison of classnames against dataset names as prompt inputs to the DMPCoOp model.</td><td></td></tr><tr><td>Model</td><td></td><td>ImageNet</td><td>Oxford flowers Oxford pets</td><td>Stanford cars</td><td>Caltech 101</td><td>Food101</td><td>FGVC Aircraft</td><td>DTD SUN397</td><td>EuroSAT</td><td>UCF101</td><td>Average</td></tr><tr><td>CoOp</td><td>68.0 67.8</td><td>66.5 65.2</td><td>88.5 88.0</td><td>61.9 63.4</td><td>92.7 92.7</td><td>84.8 83.9</td><td>15.2 16.8</td><td>60.7 40.9 62.6 41.0</td><td>46.9 46.3</td><td>65.3 66.6</td><td>62.9</td></tr><tr><td>DMPCoOp (Dataset Names) ∆ (Dataset)</td><td>-0.2</td><td>-1.3</td><td>-0.5</td><td>+1.5</td><td>0.0</td><td>-0.9</td><td>+1.6</td><td>+1.9</td><td>+0.1 -0.6</td><td>+1.3</td><td>63.2 +0.3</td></tr><tr><td></td><td>68.7</td><td></td><td>89.2</td><td>62.4</td><td>91.4</td><td>85.7</td><td>20.5</td><td>62.9</td><td>40.4</td><td></td><td></td></tr><tr><td>DMPCoOp (Class Names)</td><td></td><td>69.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.2</td><td>66.7</td><td>63.9</td></tr><tr><td>∆ (Class)</td><td>+0.7</td><td>+2.5</td><td>+0.7</td><td>+0.5</td><td>-1.3</td><td>+0.9</td><td>+5.3</td><td>+2.2 -0.5</td><td>-0.7</td><td>+1.4</td><td>+1.0</td></tr></table>

Figure 10 presents a qualitative comparison of images generated by Concept Sliders (using the authorprovided models), Prompt Sliders, and DMP. The results show that Concept Sliders struggle to induce the intended attributes, even at higher scales, due to their sensitivity to training hyperparameters, which requires careful tuning as noted in (Sridhar and Vasconcelos, 2024). We observe that DMP-generated prompts produce images that are qualitatively similar to those from Prompt Sliders. However, the original Prompt Sliders method has limitations in maintaining subject identity at higher scales, as discussed in (Sridhar and Vasconcelos, 2024). In contrast, DMP—despite being trained with the Prompt Sliders embeddings—demonstrates greater robustness, effectively preserving subject identity even at relatively higher scales. These result corroborate the results observed in Table 2 of the paper.

Figure 18 illustrates the results of pure negative prompts, where the sliders yield images with attributes opposite to the specified concepts, such as shorter hair or a neutral (non-smiling) expression.

## A.5 DMP for Variations

Generalization. Figure 12 shows images synthesized for the variation prompts generated by DMP-Variation model, for a common seed. Note that the variation prompts are conditioned by the Textual inversion embedding for the target concept, which is illustrated by a GT image in the figure. The figure shows that DMPVariation produces successful variation prompts for general objects as diverse as birds and statues despite being trained only on CelebA-face identities.

Qualitative Results. Figure 13 shows additional qualitative results for variations generated by DMPVariation model.

Figure 15 shows the synthesized prompts that produce variations of a person, conditioned on the textual inversion embedding of the person. Note that, while all images are synthesized with the same SDXL seed, they exhibit a diversity of background scenes, hair patterns, clothing, etc. This is an additional benefit of the natural prompt variability of DMP-based personalization: to increase the diversity of the synthesized images.

Figure 16 presents qualitative results comparing DMP and TI in generating personalized subject images across diverse contexts. The variation prompts produced by DMP demonstrate greater robustness and generalization, effectively adapting to different contexts while preserving subject identity. In contrast, Textual Inversion tends to overfit to the subject, leading to poor generalization. For instance, it completely fails to generate correct images for the Buddha statue and succeeds in only a single scenario for the duck and dog subjects. Table 29 shows the detailed HPSv2 scores for the results presented in Table 6 of the paper.

Identity Composition. Figure 17 demonstrates results of combining two celebrities, displayed on the left, using DMPVariation model to synthesize identity prompts with Eq. 8. The synthesized identity clearly incorporates prominent features from both original faces, such as the nose and chin, resulting in a cohesive blend of attributes.

![](images/69c5daf3a7bb8cc6e852e08510697485b985248a76d86c15761f085ceec652d7.jpg)

![](images/e237b6b656630fb99515ee923241ffff02a8bc4cbf0c9ffeb899e0f6d3a78db1.jpg)

![](images/af183c6bfd6206766f6141a110a5d9d0648f760de95c9bb19ad863f5e6c43799.jpg)

![](images/81295f63b2f43f18f986a1101e349ae43182aab5efbba14956160513685a326c.jpg)

![](images/1633dd30870e6d33a2040119f59589548b7c86be573d8a337f9ee9d1af24a767.jpg)

![](images/4c06b1aaec8a5cb70b4e8e9e39e462dddf822e3455922d9b2ea6ecf18100f316.jpg)  
smiling

![](images/afc75a4aee5abc658c5f747adff55ddec9c4bfbd7d7b1337ab8977bf9e2a1e7d.jpg)

![](images/9fda6db975dc316eaa737cc38825d874ab15fe3931b567358ae271af3522b481.jpg)

Figure 6: Prompt Sliders images synthesized with sliders sampled by DMPSlider when prompted for age and smiling. The prompt used for the SD-XL model is "A photo ofa beautiful man".  
![](images/14a236e0e3e0f333eb42a96bc8f08ac103096b595ad3f82bc78818aefb3404de.jpg)

![](images/b7515c18dc08bb9b4a481288c229c6cdbe49a204afd40cf22dd4be5c19c6d3b8.jpg)  
Long hair

![](images/865dbedf3d4b40c618a62cd824bc433031441b1be1ca856e4c43863e6b2e0d31.jpg)

![](images/74b1887523fe9df07a711597fde3aecfc9980c73e4305240c0712a04036e38d7.jpg)

![](images/1ad29f4bbcd5ff0edd1381261eb2e6c895eef194a452fb1275297c1869613642.jpg)  
Chubby

![](images/dee12b0aea75a6fc9647296842f7f5d9b0a53c0b4b39ca089b05efadb81b3b20.jpg)  
Figure 7: Qualitative results of DMPSlider prompts depicting the concepts “long hair" and “chubby". The prompts used for the SD-XL model for the images shown from left to right are as follows. "A closeup photo ofa person", "Professional headshot ofa person".

Subject Composition. Figure 19 illustrates the ability of DMPVariation to generate prompts for combined concepts. In this examples, the prompts elicit the downstream model to produce images that combine the two subjects displayed on the left. This is done by sampling subject prompts using the DMPVariation model and Eq. (5). These are then fed to the SDXL model to produce the images on the right, for increasing guidance scale η. The synthesized subject clearly incorporates prominent features from both, such as the nose and chin, resulting in a cohesive blend of attributes.

Interpreting Variation Prompts. We computed the top-5 nearest-neighbor tokens in the CLIP vocabulary for the prompts sampled by DMPVariation model conditioned on a Textual Inversion embedding of a subject. For a random subject not in the training dataset, the TI prompt embedding returns <w>karanjohar</w>, <w>conclude</w>, <w>leaked</w>, <w>prohibition</w>, <w>vijaysethu</w>. The DMP-Variation prompt returns <w>karanjohar</w>, <w>pandoramusic</w>, episo, <w>leaked</w>, <w>refriger</w>. It overlaps with two words out of five showing that the prompt is indeed a variation of the conditioned Textual Inversion prompt. Extending this analysis over 66 unseen identities, we found an average of 2.7 common words in the top-5 and 5.6 in the top-10 nearest-neighbor tokens, demonstrating that the DMP model effectively generalizes while retaining some of the subject-specific characteristics.

Table 25: Prompts used for evaluating generalization. Prompts were designed to explore style and concept variations of the subject sks and borrowed from DreamBooth (Ruiz et al., 2023).
<table><tr><td colspan="4">Prompts</td></tr><tr><td>a sks on the beach</td><td>sks flower arrangement</td><td>sks stained glass window</td><td>sks as a witcher</td></tr><tr><td>A photo of two sks on a boat</td><td>sks Funko Pop</td><td>sks latte art</td><td>A cubism painting of sks person</td></tr><tr><td>Manga drawing of sks</td><td>Pointillism painting of sks</td><td>Ukiyo-e painting of sks</td><td>A sks as a knight in plate armor</td></tr><tr><td>sks as a knight in plate</td><td>Banksy art of sks</td><td>sks piloting a fighter jet</td><td>Greek sculpture of sks</td></tr><tr><td>Fauvism painting of sks</td><td>Cave mural depicting sks</td><td>sks by Andy Warhol</td><td>sks in the style of Archer</td></tr><tr><td>Colorful graffiti of sks</td><td>sks as Ziggy Stardust</td><td>sks in a comic book</td><td>Watercolor painting of sks</td></tr><tr><td>a sand sculpture of sks</td><td>sks in a Santa hat</td><td>sks as a wizard</td><td>a photo of sks</td></tr></table>

GT Images

![](images/b7a24cebb9910b40756638ad6a706b9e50668da9e9b93c4ae957af73a58d3fda.jpg)  
Figure 8: Qualitative comparison of image synthesis with TI and DMPMulti prompts. Left: groundtruth images for Identity-38 and Identity-96 in the training set. Right: images synthesized by SD for the original TI prompts (first row) and prompts sampled from the DMPMulti model (second row):

## A.6 Ablation Studies

The following section lists the setup for the ablation study on sensitivity of DMP to class-name conditions in Table 9. Table 26 shows the corresponding class names provided to DMP at each replacement level.

Original 0% Condition. The original Caltech101 condition is:

accordion; bonsai; chandelier; euphonium; menorah;

saxophone; stapler; yin\_yang; anchor; camera

It contains objects, instruments, and symbols drawn from Caltech101.

Replacement Pool. The replacement names are different ImageNet classes:

Positive  
Negative  
![](images/cff0082dba6502fc0fc5c2214f9291fecea4e72cf85cb3eceda18c8a2b147171.jpg)  
DMP-Negative Prompting (SD-RV)  
Figure 9: Negative prompting samples from the SD Realistic Vision model when prompted by the DMPMulti, itself prompted with id-21 (leftmost) as positive and id-24 (second from left) as negative prompt. The generated identity has features opposing to id-24 (chubby cheeks, small eyes, wide nose, etc.)

brambling; goldfinch; house finch; junco;

indigo bunting; American robin; bulbul; jay

These names are inserted cumulatively and in a fixed order. This controls which names change and ensures that the only systematic variable is the replacement percentage.

Table 26: Class-name conditions supplied to DMP at each replacement level.
<table><tr><td>Replacement</td><td>Class-name condition supplied to DMP</td></tr><tr><td>0%</td><td>accordion, bonsai, chandelier, euphonium, menorah, saxophone, stapler, yin_yang, anchor, camera</td></tr><tr><td>20%</td><td>brambling, goldfinch, chandelier, euphonium, menorah, saxophone, stapler, yin_yang, anchor, camera</td></tr><tr><td>40%</td><td>brambling, goldfinch, house finch, junco, menorah, saxophone, stapler, yin_yang, anchor, camera</td></tr><tr><td>60%</td><td>brambling, goldfinch, house finch, junco, indigo bunting, American robin, stapler, yin_yang, anchor, camera</td></tr><tr><td>80%</td><td>brambling, goldfinch, house finch, junco, indigo bunting, American robin, bulbul, jay, anchor, camera</td></tr></table>

## A.6.1 Ablation Study on Alternative methods for Meta-Prompting.

The first two rows of Table 19 show the ablation study of using Transformer and GPT-2 for modeling the distribution of CoOp prompts across the 11 datasets. We include the Transformer as a nongenerative baseline and GPT-2 as an autoregressive generative model. For the Transformer, we use a pretrained RoBERTa-Base model, which is finetuned using LoRA to predict prompt embeddings from text conditions. A linear layer is added on top of the final layer to produce output embeddings of the target dimension (2048 for CoOp/CoPrompt), and training is done with MSE loss. The table shows that transformer tends to overfit to the base classes as it is able to closely match the performance of the baseline CoOp on the base classes while under-performing on the novel classes leading to a decrease in the overall performance. For GPT-2, we discretize the continuous prompt embeddings by identifying their top-5 nearest tokens in the CLIP embedding space. These tokens are then converted back to text and used to finetune a pretrained GPT-2 model with LoRA, trained to predict the top-5 tokens using standard cross-entropy loss. The model is conditioned on the same text inputs as used in the diffusion counterpart. Although each text condition has 40 associated prompts from different initializations (as described in Section 4.1), the resulting tokens after discretization are almost identical across seeds. This indicates that the variation captured in the continuous embedding space is lost during the discretization process-a known limitation, as discretization inherently reduces information. The table reflects this observation and shows that GPT-2 based modeling is inferior to diffusion since diffusion is much better for modeling continuous distribution of prompts. Moreover, unlike autoregressive methods, diffusion offers multiple benefits such as classifier-free guidance, negative prompting, inversion, editing and composition that are challenging or infeasible with models like GPT-2. These results show that modeling the prompt distribution with diffusion is more effective and flexible than using autoregressive methods.

![](images/63e4be102986e7730c337d4ba6aec0b0a0184b7383996b847aaeeffd7cf9e6db.jpg)  
Figure 10: Qualitative comparison of images synthesized with baseline prompt sliders and concept sliders against DMPSlider. The prompt used for the SD-XL model is “A photo ofa $g i r l ^ { \prime \prime }$ and the slider trained to control the concept “curlyhair".

## A.6.2 Ablation on the number of prompts per concept

We already include the ablation study when using only 20 prompts per concept for identity synthesis in Table 15. Here, we include an additional ablation study on the number of prompts for classification tasks in Table 27. It shows that DMP works well even with as few as 5 or 10 prompts per task/concept. For 5 prompts per class, DMPCoOp obtains +1.2% gain on new classes and it increases as n increases (+1.6/+2.5/+3.0 for n =10/20/40 prompts respectively). The results are consistent with our theoretical guarantee discussed in Appendix A.1 where higher n corresponds to lower downstream risk and better generalization.

## A.6.3 Ablation on the text inputs to DMPCoOp model

We conducted an ablation study using the dataset names as the prompt or text condition to the DMPCoOp model instead of the classnames. Table 24 shows that the performance of the model using dataset names is better than the baseline CoOp prompts by 0.3% on average while it is 0.7% lower as compared to the model using classnames. This shows that using class-specific names generates prompts with robust generalization than just using a single dataset name as the text condition.

## A.6.4 Ablation on the quality of training data

To investigate this, we assessed the impact of the quality of training data by training DMP with noisy prompts. We apply additive Gaussian noise $( n _ { i } )$ with a standard deviation of $\sigma = 0 . 0 1$ to the $i ^ { \mathrm { { \acute { h } } } }$

Closest training image

![](images/ef02847ea3a62a2e7cc98028da077ebe5030c29170aa389ed38cb2d6ad351c19.jpg)

Novel id

![](images/76adbeb515d55c452699763b5f18d50166d3bfd01dd4cd8ef443b4632b82b623.jpg)

Closest training image

![](images/f944a3ab6474d6978987ec52b1655879ced3157f81f01a3885e93539c1a9245b.jpg)

Novel id  
![](images/1b1911d185f0c3949429d9f798e91af52d267ec507cea06ceb3ad5ff6d2912df.jpg)  
Figure 11: Additional Qualitative results: Nearest neighbors in the training set shown on the left for novel ids sampled by DMPMulti model to the right.

GT Image

![](images/6834b2fd0c6284177ae81e1d98ce06ac3219e61495b9da3105dd78c102f3d90c.jpg)

TI  
![](images/e4f0e9c3fb41d87e8f0417cf82b3f94c31dc032ab231e815acdc2858e4486ce8.jpg)

DMPVariations  
![](images/d317a491f9af112a70d1dd30743d5a4a50010c850a2521482a232c45f97bbb52.jpg)

![](images/2b32f77e6b236d77ea626a87858f28f7649f0baf5937c285b0c25ccad8a30a7a.jpg)  
Figure 12: Generalization of DMPVariation model. Left: groundtruth images. Second: images generated by SD for the original TI prompts and the last two columns are the images for variation prompts sampled by DMP. See Fig. 13 for additional results.

prompt $x ( c ) _ { i }$ as

$$
x ( c ) _ { i } = ( 1 - \sigma ) x ( c ) _ { i } + \sigma n _ { i } .
$$

The noisy prompt is applied to $x \%$ of the training dataset, where $x \in \{ 1 0 , 4 0 \}$

Table 28 summarizes the experiment where random noise is added to the prompts in the training repository. Meta-prompting is observed to be robust to noise levels ranging from 10% to 40% of the training prompts. DMPCoOp trained with 10% noisy prompts still obtains +1.9% improvement over the baseline. Further, the average H.M for the DMP model with 10% noisy prompts is slightly better (76.7 vs 76.5 for DMPCoOp) than the DMPCoOp model trained on clean prompts suggesting that a small amount of noise can also help with generalization. This is similar to image diffusion models, which are also known to be robust to noise added during training.

TI  
GT Image  
DMPVariations  
![](images/e337e922f6088dc83806261841e1969250929d9c5f989f86cdc81750b8bdd017.jpg)  
Figure 13: Additional Qualitative results of generalization of DMPVariation model. Left: groundtruth images. Second: images generated by SD for the original TI prompts and the last two columns are the images for variation prompts sampled by DMP.

## A.7 Ablation on DMPMult

We trained separate DMP models for identity synthesis and slider synthesis to compare their performance with the DMPMulti model, which was trained to generate both prompt types simultaneously. Table 16 presents the results of the DMPSlider model, trained solely for slider prompt generation. Table 17 presents the results of the DMPIdentity model, trained solely for identity prompt generation. The results indicate that its performance is comparable to that of the DMPMulti model, demonstrating that multi-task training in DMPMulti does not compromise its effectiveness.

## A.7.1 Ablation on Identity Composition with Textual Inversion

For identity composition, we perform an ablation study using Stable Diffusion with Equation 5, generating new identities by combining prompts such as "a photo of id-1" and "a photo of id-2." Figure 14 illustrates the results of this process using Textual Inversion prompts with the Stable Diffusion v1.5 model. The generated images are often noisy, distorted, and tend to replicate the input identities rather than effectively merging their attributes. Additionally, running a full forward diffusion process with multiple identities doubles the inference time from 4 to 8 seconds per image.

GT Images  
Composed Identity with TI prompts  
![](images/14fbf2495ac0de0e6c5e20f503e8de52d91f846f3ec87b2127248b23e3d58c66.jpg)

Figure 14: Ablation study on Identity composition with Textual Inversion: images generated with TI prompts by composing the Stable Diffusion outputs for the composition of the two identities shown on the left for each row. TI composition produce distorted identities or repeats the same identities.  
![](images/46e320a3bb4cb095460c7e12015f02909195e83eb258b9ef37f2cd2165b82001.jpg)  
Figure 15: DMP Variations of Textual Inversion (TI) Prompts: The leftmost image is the real groundtruth, the second is generated by TI, and the rest are DMP variations (each image represents a new identity) conditioned on the TI embedding. All images are generated with a fixed seed to the stable diffusion model. The FaceID similarity to the groundtruth image is listed on top of each image. Lower is better for diverse variations.  
In contrast, our DMPMulti achieves high-quality identity compositions in approximately 5 seconds, introducing only a 1-second overhead compared to standard Stable Diffusion.

## A.8 DMP for Personalization

The DMPVariation model enables the generation of prompt variations conditioned on a given Textual Inversion prompt, making it possible to train a text-to-prompt meta-diffusion model as a replacement for personalization prompt repositories. Users only require access to existing prompt repositories, as the DMPVariation model can directly generate the necessary intermediate embeddings. We choose the top-k (k = 40) embeddings generated from the DMPVariation model based on the cosine similarity with the available TI prompts. These embeddings can serve as training data for generating personalized prompts based on textual input. Once trained, this approach eliminates the need to search and retrieve prompts from a database, allowing for on-the-fly prompt generation.

Figure 8 presents a comparison of the images generated by DMPMulti model for the identity labeled "id-38" and "id-96" respectively. The figure presents three classes of images: groundtruth on the left, synthesized by SD prompted by the original TI prompts in the first and third row of the right side, and synthesized by SD prompted by the DMPMulti model, itself prompted for“identity-c." The images synthesized using DMPMulti model have quality comparable to those synthesized with TI prompts.

Negative Text Guidance. Figure 9 illustrates the effect of negative prompting by displaying images synthesized when DMPMulti model is prompted with identity-21 (leftmost) as positive and identity-24 (second from left) as negative prompt. The generated identity exhibits contrasting characteristics, such as fuller cheeks, smaller eyes, and a broader nose—features to those of the negative identity (id-24).

Novel Identities. Figure 11 shows qualitative results of novel identities sampled by DMP and their closest training images. Figure 24 shows additional qualitative results of novel identities sampled by DMPMulti.

Identity Prompt Diffusion. Figure 20 presents the qualitative results of various identities sampled by DMPMulti model. The figure contains the groundtruth on the left, and synthesized by SD prompted by the DMPMulti, itself prompted for“identity-c" on the right. The images synthesized using both models reflect the original identities in the groundtruth images.

A photo of S\*  
as a drawing  
![](images/3c7a977d62eaa3b3e35159c09ec71b50eaf04934187256745683aa432c3fa789.jpg)  
Figure 16: Qualitative results of images generated from prompts synthesized by DMPVariation. Our Diffusion Meta-Prompts are more robust and less overfitted than the baseline textual inversion which fails to generalize  
Scale 9.0 Scale 9.1 Scale 9.2 Scale 9.3 Scale 9.4 Scale 9.5 Scale 9.6 Scale 9.7 Scale 9.8 Scale 9.9 Scale 10

![](images/150e876cb85c8fafe12741f60db834f8477fc3b37cf512e8a1679996620b7a02.jpg)  
Figure 17: Identity composition: images generated with DMPVariation model prompts (right) for increasing guidance scales from 9 to 10 for the composition of the two identities shown on the left.

Identity Composition. Figure 21 demonstrates additional results of combining two identities, displayed on the left, using DMPMulti to synthesize identity prompts with (5). The synthesized identity clearly incorporates prominent features from both original faces, such as the nose and chin, resulting in a cohesive blend of attributes.

Figure 22 shows the results of identity composition using SDv1.5 checkpoint that uses the same CLIP text encoder as SD-Realistic Vision checkpoint. It shows that DMP performs effectively without requiring retraining for this version. Since DMP was trained in CLIP text space, it generalizes to all models sharing the CLIP text encoder, eliminating the need for retraining on specific model versions.

Interpolation. Figure 23 shows the qualitative results of interpolating between two faces using the DMPMulti model with classifier-free guidance scale between 0 to 5. The results show that DMPMulti enables fine-grained interpolation by simply manipulating the guidance scale.

## A.9 Limitations and Future Work

While the DMP framework unifies and improves prompt generation and generalization, simplifying deployment, the effectiveness of DMP is fundamentally constrained by the quality and expressiveness of the underlying prompt learning method used to construct the training repository. Second, DMP inherits the limitations of text prompts such as lack of fine-grained control and is sensitive to noise seeds similar to image diffusion models with a small variance between the sampled prompts. In our experiments, DMP maintained high accuracy even at different noise seeds with only a standard deviation of 0.7 % from the mean accuracy for classification tasks.

![](images/341256cf116120dce741cc1a3469d0b19a865b01abd6b4f88136b071b6731119.jpg)

![](images/6b46830c5d4ef32a47c2202193a89deacb4f690a5ee6801d3ba2c1426021c04f.jpg)  
Figure 18: Negative Prompt Sliders images synthesized with sliders sampled by DMPSlider when nega tively prompted for different concepts. The prompt used for the SD-XL model is "A photo ofa girl".  
Figure 19: Subject composition: images generated with DMPVariation model prompts (right) for the composition of the two identities shown on the left with the guidance scale denoted at the top. See Fig. 17 for additional results at finer scale.

<sup>‘-’</sup> <sup>chubby</sup> <sup>‘-’</sup> <sup>surprised</sup>Table 27: Ablation study with different number of prompts per concept. Base2new generalization per dataset: performance for All, Base, and New classes, and HM (Harmonic Mean). The results reported are the average over three seed runs.
<table><tr><td></td><td colspan="4">(a) Average</td><td colspan="3">(b) ImageNet</td><td colspan="4">(c) Caltech101</td><td colspan="4">(d) OxfordPets</td></tr><tr><td>Method</td><td>All</td><td>Base</td><td>New</td><td>HM</td><td>All</td><td>Base</td><td>New HM</td><td>All</td><td>Base</td><td>New</td><td>HM</td><td>All</td><td>Base</td><td>New</td><td>HM</td></tr><tr><td>CoOp (Zhou et al., 2022b)</td><td>67.1</td><td>80.9</td><td>68.6</td><td>74.2</td><td>68.0</td><td>76.2</td><td>66.8</td><td>71.2 93.8</td><td>98.1</td><td>93.0</td><td>95.5</td><td>90.6</td><td></td><td>95.2 96.5</td><td>95.8</td></tr><tr><td>DMPCoOp</td><td>70.1</td><td>80.3</td><td>73.4</td><td>76.5</td><td>68.7</td><td>75.4</td><td>68.8 72.0</td><td>94.8</td><td>98.3</td><td>95.3</td><td>96.8</td><td>91.7</td><td>95.4</td><td>97.3</td><td>96.3</td></tr><tr><td>DMPCoOp (5 prompts)</td><td>69.1</td><td>81.4</td><td>71.6</td><td>75.9</td><td>68.9</td><td>76.6 68.0</td><td>72.0</td><td>94.8</td><td>97.9</td><td>95.2</td><td>96.5</td><td>91.1</td><td>95.4</td><td>97.1 96.2</td><td></td></tr><tr><td>DMPCoOp (10 prompts) DMPCoOp (20 prompts)</td><td>69.3</td><td>81.8</td><td>72.0</td><td>76.3</td><td>69.0</td><td>76.6</td><td>68.5 72.3</td><td>94.5</td><td>98.2</td><td>95.2</td><td>96.7</td><td>92.9</td><td>96.0</td><td>97.5 96.7</td><td></td></tr><tr><td></td><td>69.1</td><td>81.2</td><td>72.9</td><td>76.6</td><td>69.3</td><td>76.4</td><td>68.8 72.4</td><td>94.3</td><td>98.2</td><td>95.1</td><td>96.6</td><td>93.0</td><td>96.0</td><td>97.5 96.7</td><td></td></tr><tr><td>Method</td><td colspan="4">(e) StanfordCars</td><td colspan="4">(f) Flowers102</td><td colspan="4">(g) Food101</td><td colspan="4">(h) FGVC Aircraft</td></tr><tr><td></td><td>All</td><td>Base</td><td>New</td><td>HM</td><td>All</td><td>Base</td><td>New HM</td><td>All</td><td>Base</td><td>New</td><td></td><td>HM</td><td>All</td><td>Base New</td><td>HM</td></tr><tr><td>CoOp (Zhou et al., 2022b)</td><td>68.3</td><td>75.8</td><td>68.2</td><td>71.8</td><td>72.2 96.2</td><td>66.7</td><td>78.8</td><td>83.8</td><td>89.8</td><td>88.0</td><td>88.9</td><td>25.9</td><td>37.0</td><td>30.4</td><td>33.4</td></tr><tr><td>DMPCoOp</td><td>69.3</td><td>74.5</td><td>72.0</td><td>73.2</td><td>75.6 93.4</td><td>72.2</td><td>81.4</td><td>86.4</td><td>89.6</td><td>89.9</td><td>89.7</td><td>26.3</td><td>35.7</td><td></td><td>31.633.5</td></tr><tr><td>DMPCoOp (5 prompts)</td><td>68.8</td><td>75.2</td><td>70.8</td><td>72.9</td><td>74.5 96.0</td><td>71.1</td><td>81.7</td><td>84.5</td><td>89.5</td><td>89.1</td><td>89.3</td><td>27.9</td><td>37.5</td><td></td><td>33.335.3</td></tr><tr><td>DMPCoOp (10 prompts)</td><td>68.5 75.8</td><td></td><td>69.9</td><td>72.7</td><td>73.7 95.7</td><td>69.8</td><td>80.7</td><td>85.5</td><td>89.8</td><td>91.2</td><td>90.5</td><td>27.8</td><td>37.0</td><td></td><td>32.0 34.3</td></tr><tr><td>DMPCoOp (20 prompts)</td><td>68.7 75.0</td><td></td><td>70.8</td><td>72.8</td><td>75.9 96.4</td><td>72.1</td><td>82.5</td><td>85.4</td><td>90.0</td><td>91.1</td><td>90.5</td><td>27.9</td><td>37.3</td><td></td><td>31.6 34.2</td></tr><tr><td></td><td colspan="4">(i) SUN397</td><td colspan="4">(j) DTD</td><td colspan="4">(k) EuroSAT</td><td colspan="4">(1) UCF101</td></tr><tr><td>Method</td><td>All Base</td><td></td><td>New</td><td>HM</td><td>All</td><td></td><td></td><td>HM</td><td></td><td></td><td></td><td>HM</td><td>All</td><td></td><td>HM</td></tr><tr><td>CoOp (Zhou et al., 2022b)</td><td></td><td></td><td></td><td></td><td></td><td>Base New</td><td></td><td>All</td><td>Base</td><td>New</td><td></td><td></td><td>Base</td><td>New</td><td></td></tr><tr><td>DMPCoOp</td><td>67.2 67.3</td><td>80.7</td><td>71.8</td><td>76.0 76.5</td><td>51.0</td><td>78.5 50.5</td><td>61.5</td><td>49.7</td><td>78.3</td><td>58.3</td><td>66.8 83.1</td><td>68.1 71.7</td><td>84.0 79.6</td><td>64.2</td><td>72.8 70.7 74.9</td></tr><tr><td>DMPCoOp (5 prompts)</td><td>66.8</td><td>80.4 79.8</td><td>72.9 71.7</td><td>75.5</td><td>53.7 52.1</td><td>75.1 55.8</td><td>64.0</td><td>65.9 59.1</td><td>85.6</td><td>80.7</td><td>75.3</td><td>71.1</td><td>85.6</td><td></td><td>69.6 76.8</td></tr><tr><td>DMPCoOp (10 prompts)</td><td>67.6</td><td>80.3</td><td>73.4</td><td>76.7</td><td>52.1</td><td>76.4 54.3</td><td>63.5 53.4 62.6</td><td>60.1</td><td>85.7 89.0</td><td>67.2 71.7</td><td>79.4</td><td>70.7</td><td>85.3</td><td></td><td>69.2 76.4</td></tr><tr><td>DMPCoOp (20 prompts)</td><td>67.6</td><td>80.4</td><td>73.3 76.7</td><td></td><td>51.475.6</td><td>75.6</td><td>54.1 63.1</td><td>54.6</td><td>84.8</td><td>74.4</td><td>79.3</td><td>71.8</td><td>82.9</td><td></td><td>72.6 77.4</td></tr></table>

Future research directions can address these limitations by exploring joint training of the meta-model and the downstream foundation models, potentially overcoming the performance ceiling imposed by existing prompt learning techniques. Other directions for future work can explore ways for distilling LoRA (Hu et al., 2022) adapters into prompts and the design of a Meta-LoRA model, which synthesizes weight matrices instead of prompts (a more complex problem due to the large parameter cardinality of LoRA weights).

## A.10 Implementation details

In this section, we describe the evaluation setup followed in our experiments and the rationale behind choosing the setup. We then describe the hyperparameter settings used to train all DMP models and finally the prompt format used in DMPMulti model.

Evaluation Setup. For downstream model, we use stable diffusion (Rombach et al., 2022) Realistic-Vision-v4 checkpoint using classifier-free guidance with a scale of 4.5 and 30 DDIM steps for the image synthesis. For SD-XL (HuggingFace, 2023) model, we use a scale of 7.5 with 20 DDIM steps. Note that, because DMP prompts are introduced in the CLIP text encoder, they can be interchangeably used with any diffusion model using this encoder. Our choice of downstream diffusion model follows the original prompting methods.

For models other than CoOp, additional weights or head layers are optimized to prompt the deeper layers of the CLIP encoders. Due to the large number of parameters associated with these weights, it is infeasible to train a diffusion model to synthesize these parameters. For example, the projection matrices of MaPLe have 3.55 million parameters while CoPrompt and TAC have 4.65 million parameters each. This is much larger than even the images produced by Stable diffusion (65536 parameter latent). In contrast, all text prompts have only 2048 or fewer parameters (only 256 parameters in the latent space). So, we only synthesize the text prompts attached to the input of CLIP model with DMP. During inference, we replace the learned textual prompt from the baseline method with the prompt sampled with DMP while keeping the other weights of the baseline method fixed. As followed in prompt learning literature, we pick three random seeds and report the average results.

Table 28: Base2new generalization per dataset: performance for All, Base, and New classes, and HM (Harmonic Mean). The results reported are the average over three seed runs.
<table><tr><td></td><td colspan="4">(a) Average</td><td colspan="3">(b) ImageNet</td><td colspan="3">(c) Caltech101</td><td colspan="4">(d) OxfordPets</td></tr><tr><td>Method</td><td>All</td><td>Base</td><td>New</td><td>HM</td><td>All Base</td><td>New</td><td>HM</td><td>All</td><td>Base New</td><td>HM</td><td>All</td><td>Base</td><td>New</td><td>HM</td></tr><tr><td>CoOp (Zhou et al., 2022b)</td><td>67.1</td><td>80.9</td><td>68.6</td><td>74.2</td><td>68.0 76.2</td><td>66.8</td><td>71.2</td><td>93.8</td><td>98.1</td><td>93.0</td><td>95.5</td><td>90.6</td><td>95.2 96.5</td><td>95.8</td></tr><tr><td>DMPCoOp</td><td>70.1</td><td>80.3</td><td>73.4</td><td>76.5</td><td>68.7 75.4</td><td>68.8</td><td>72.0</td><td>94.8</td><td>98.3 95.3</td><td>96.8</td><td>91.7</td><td>95.4</td><td>97.3</td><td>96.3</td></tr><tr><td>DMPCoOp (10% noise)</td><td>69.6</td><td>81.7</td><td>72.3</td><td>76.7</td><td>69.2 76.5</td><td>69.0</td><td>72.6</td><td>94.0 98.1</td><td>94.2</td><td>96.1</td><td>91.5</td><td>95.0</td><td>96.9</td><td>95.9</td></tr><tr><td>DMPCoOp (40% noise)</td><td>68.9 81.2</td><td></td><td>70.3</td><td>75.0</td><td>68.5 76.6</td><td>67.3</td><td>71.6</td><td>94.1 97.9</td><td>95.2</td><td>96.5</td><td>92.8</td><td>95.7</td><td>97.9</td><td>96.8</td></tr><tr><td></td><td colspan="4">(e) StanfordCars</td><td colspan="3">(f) Flowers102</td><td colspan="4">(g) Food101</td><td colspan="4">(h) FGVC Aircraft</td></tr><tr><td>Method</td><td>All Base</td><td></td><td>New</td><td>HM|</td><td>All</td><td>Base</td><td>New HM</td><td>All</td><td>Base</td><td>New</td><td>HM|</td><td>All</td><td>Base</td><td>New HM</td></tr><tr><td>CoOp (Zhou et al., 2022b)</td><td>68.3</td><td>75.8</td><td>68.2</td><td>71.8</td><td>72.2 96.2</td><td></td><td>78.8</td><td>83.8</td><td>89.8</td><td></td><td></td><td>37.0</td><td></td><td>33.4</td></tr><tr><td>DMPCoOp</td><td>69.3 74.5</td><td></td><td>72.0</td><td>73.2</td><td>75.6 93.4</td><td>66.7 72.2</td><td>81.4</td><td>86.4 89.6</td><td>88.0</td><td>88.9 89.7</td><td>25.9 26.3</td><td>35.7</td><td>30.4 31.6</td><td>33.5</td></tr><tr><td>DMPCoOp (10% noise)</td><td>69.1 76.7</td><td>69.4</td><td>72.9</td><td></td><td>77.1 96.4</td><td>72.8</td><td>83.0</td><td>85.6 89.9</td><td>89.9 91.1</td><td>90.5</td><td>27.7</td><td>37.6</td><td>33.8</td><td>35.6</td></tr><tr><td>DMPCoOp (40% noise) 68.3</td><td>76.6</td><td>69.1</td><td></td><td>72.7</td><td>79.4 95.9</td><td>75.0</td><td>84.2</td><td>84.6 89.4</td><td>90.3</td><td>89.8</td><td>26.4</td><td>36.6</td><td>32.8</td><td>34.6</td></tr><tr><td></td><td colspan="4">(i) SUN397</td><td colspan="4">(j) DTD</td><td colspan="4">(k) EuroSAT</td><td colspan="4">(1) UCF101</td></tr><tr><td>Method</td><td>All Base</td><td></td><td>New</td><td>I HM</td><td>All</td><td>Base</td><td>New HM</td><td>一 All</td><td>Base</td><td>New</td><td>HM</td><td>All</td><td>Base</td><td></td><td>HM</td></tr><tr><td>CoOp (Zhou et al., 2022b)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>New</td><td></td></tr><tr><td>DMPCoOp</td><td>67.2 67.3</td><td>80.7 80.4</td><td>71.8 72.9</td><td>76.0 76.5</td><td>51.0 53.7</td><td>78.5</td><td>50.5 61.5 55.8 64.0</td><td>49.7 65.9</td><td>78.3</td><td>58.3</td><td>66.8</td><td>68.1</td><td>84.0 79.6</td><td>64.2</td><td>72.8 74.9</td></tr><tr><td>DMPCoOp (10% noise)</td><td>66.2</td><td>80.7</td><td>70.5</td><td>75.3</td><td>53.5</td><td>75.1 79.9</td><td>51.9 62.9</td><td>59.3</td><td>85.6 84.5</td><td>80.7 71.6</td><td>83.1 77.5</td><td>71.7 72.5</td><td>83.0</td><td>70.7 74.0</td><td>78.2</td></tr><tr><td>DMPCoOp (40% noise)</td><td>65.9</td><td>79.7</td><td>69.8 74.4</td><td></td><td>49.8 78.4</td><td></td><td>48.6 60.0</td><td>59.6</td><td>84.0</td><td>62.8 71.9</td><td></td><td>68.4</td><td>82.3</td><td></td><td>64.1 72.1</td></tr></table>

We note that different initialization seeds result in minor performance differences which explains the performance difference from the original paper reported results.

## A.10.1 Hyperparameter Settings

Table 30 summarizes the detailed hyperparameter settings of the DMP models trained from scratch reported in the main paper.

## A.10.2 Evaluation Prompts for DMPMulti

The example format of the prompts used for evaluation is shown below for the concept "Age",

• A portrait of a woman with a warm smile, {}

• A person’s face, {}

• A man sitting on a park bench, reminiscing about his youth, {}

• A couple of friends enjoying a picnic together, {}

• A photo of a person, {}

![](images/92678f109fdc32db55baeb80f5801bef9cc541114048d24302de9aaa1e7e5804.jpg)  
Figure 20: Qualitative results of image synthesis with DMPMulti prompts. Left: groundtruth images. Right: images synthesized by SD with prompts sampled from the DMPMulti model.

<table><tr><td>Concept</td><td>DMP</td><td>TI</td><td>Delta</td></tr><tr><td>ettblackteapot</td><td>0.2905</td><td>0.2778</td><td>+0.0127</td></tr><tr><td>dicoo</td><td>0.2837</td><td>0.2700</td><td>+0.0137</td></tr><tr><td>goku</td><td>0.2830</td><td>0.2800</td><td>+0.0030</td></tr><tr><td></td><td>0.2827</td><td>0.2754</td><td>+0.0073</td></tr><tr><td>degodsheavy</td><td>0.2825</td><td>0.2810</td><td>+0.0015</td></tr><tr><td>spider-gwen</td><td></td><td></td><td></td></tr><tr><td>johnny-silverhand</td><td>0.2812</td><td>0.2760</td><td>+0.0052</td></tr><tr><td>blue-haired-boy</td><td>0.2812</td><td>0.2800</td><td>+0.0012</td></tr><tr><td>bullvbear black-waifu</td><td>0.2798 0.2790</td><td>0.2683</td><td>+0.0115</td></tr><tr><td></td><td></td><td>0.2715</td><td>+0.0075</td></tr><tr><td>a-female-hero-from-the-legend-of-mir</td><td>0.2788</td><td>0.2710</td><td>+0.0078</td></tr><tr><td>freddy-fazbear</td><td>0.2786</td><td>0.2750</td><td>+0.0036</td></tr><tr><td>ldrs</td><td>0.2783</td><td>0.2642</td><td>+0.0141</td></tr><tr><td>chonkfrog</td><td>0.2778</td><td>0.2756</td><td>+0.0022</td></tr><tr><td>concept-art</td><td>0.2770</td><td>0.2715</td><td>+0.0055</td></tr><tr><td>degods</td><td>0.2769</td><td>0.2686</td><td>+0.0083</td></tr><tr><td>hanfu-anime-style</td><td>0.2766</td><td>0.2703</td><td>+0.0063</td></tr><tr><td>joemad</td><td>0.2764</td><td>0.2673</td><td>+0.0091</td></tr><tr><td>dog</td><td>0.2761</td><td>0.2780</td><td>-0.0019</td></tr><tr><td>stuffed-penguin-toy</td><td>0.2760</td><td>0.2756</td><td>+0.0004</td></tr><tr><td>fox-purple</td><td>0.2760</td><td>0.2734</td><td>+0.0026</td></tr><tr><td>colossus</td><td>0.2756</td><td>0.2634</td><td>+0.0122</td></tr><tr><td>chungus-poodl-pet</td><td>0.2754</td><td>0.2637</td><td>+0.0117</td></tr><tr><td>anya-forger</td><td>0.2754</td><td>0.2737</td><td>+0.0017</td></tr><tr><td>furrpopasthetic</td><td>0.2751</td><td>0.2737</td><td>+0.0014</td></tr><tr><td>bob-dobbs</td><td>0.2751</td><td>0.2664</td><td>+0.0087</td></tr><tr><td>eddie</td><td>0.2751</td><td>0.2607</td><td>+0.0144</td></tr><tr><td>arthur1</td><td>0.2750</td><td>0.2666</td><td>+0.0084</td></tr><tr><td>dragonborn</td><td>0.2750</td><td>0.2556</td><td>+0.0194</td></tr><tr><td>kay</td><td>0.2747</td><td>0.2605</td><td>+0.0142</td></tr><tr><td>tesla-bot</td><td>0.2747</td><td>0.2634</td><td>+0.0113</td></tr><tr><td>borderlands</td><td>0.2747</td><td>0.2637</td><td>+0.0110</td></tr><tr><td>lavko</td><td>0.2744</td><td>0.2588</td><td>+0.0156</td></tr><tr><td>gim</td><td>0.2740</td><td>0.2659</td><td>+0.0081</td></tr><tr><td>hubris-oshri</td><td>0.2740</td><td>0.2659</td><td>+0.0081</td></tr><tr><td>tubby</td><td>0.2737</td><td>0.2673</td><td>+0.0064</td></tr><tr><td>finn-token</td><td>0.2737</td><td>0.2734</td><td>+0.0003</td></tr><tr><td>moxxi</td><td>0.2730</td><td>0.2722</td><td>+0.0008</td></tr><tr><td>altvent</td><td>0.2730</td><td>0.2676</td><td>+0.0054</td></tr><tr><td>omlettehaai</td><td>0.2730</td><td>0.2600</td><td>+0.0130</td></tr><tr><td>jos-de-kat</td><td>0.2730</td><td>0.2651</td><td>+0.0079</td></tr><tr><td>loab-style</td><td>0.2730</td><td>0.2573</td><td>+0.0157</td></tr><tr><td>crinos-form-garou</td><td>0.2727</td><td>0.2693</td><td>+0.0034</td></tr><tr><td>blue-zombie</td><td>0.2727</td><td>0.2693</td><td>+0.0034</td></tr><tr><td>cgdonny1</td><td>0.2725</td><td>0.2610</td><td>+0.0115</td></tr><tr><td>amogus</td><td>0.2725</td><td>0.2551</td><td>+0.0174</td></tr><tr><td>lucky-luke</td><td>0.2725</td><td>0.2751</td><td>-0.0026</td></tr><tr><td>captain-haddock</td><td>0.2725</td><td>0.2725</td><td>+0.0000</td></tr><tr><td>manga-nov-23</td><td>0.2725</td><td>0.2551</td><td>+0.0174</td></tr><tr><td>button-eyes</td><td>0.2722</td><td>0.2683</td><td>+0.0039</td></tr><tr><td>bruma</td><td>0.2722</td><td>0.2632</td><td>+0.0090</td></tr><tr><td>ouroboros</td><td>0.2720</td><td>0.2734</td><td>-0.0014</td></tr><tr><td>fursona</td><td>0.2720</td><td>0.2646</td><td>+0.0074</td></tr><tr><td>kanovt</td><td>0.2720</td><td>0.2522</td><td>+0.0198</td></tr><tr><td>doc</td><td>0.2717</td><td>0.2693</td><td>+0.0024</td></tr><tr><td>warhammer-40k-drawing-style</td><td>0.2715</td><td>0.2727</td><td>-0.0012</td></tr><tr><td>nard-style</td><td>0.2715</td><td>0.2698</td><td>+0.0017</td></tr><tr><td>baluchitherian</td><td>0.2715</td><td>0.2617</td><td>+0.0098</td></tr><tr><td>insidewhale</td><td>0.2715</td><td>0.2630</td><td>+0.0085</td></tr><tr><td>irasutoya</td><td>0.2715</td><td>0.2598</td><td>+0.0117</td></tr><tr><td>devonm</td><td>0.2712</td><td>0.2617</td><td>+0.0095</td></tr><tr><td>edgerunners-style-v2</td><td>0.2712</td><td>0.2660</td><td>+0.0052</td></tr><tr><td>nixeu</td><td>0.2710</td><td>0.2660</td><td>+0.0050</td></tr><tr><td>shek-9-12-opening</td><td>0.2710</td><td>0.2708</td><td>+0.0002</td></tr><tr><td>loab-character</td><td>0.2710</td><td>0.2656</td><td>+0.0054</td></tr><tr><td>ldr</td><td>0.2710</td><td>0.2666</td><td>+0.0044</td></tr><tr><td>malika-favre-art-style</td><td>0.2710</td><td>0.2632</td><td>+0.0078</td></tr><tr><td>drive-scorpion-jacket</td><td>0.2710</td><td>0.2686</td><td>+0.0024</td></tr><tr><td>apulian-rooster-v0-1</td><td>0.2710</td><td>0.2351</td><td>+0.0359</td></tr><tr><td>fftstyle</td><td>0.2708</td><td>0.2593</td><td>+0.0115</td></tr><tr><td>alf</td><td>0.2708</td><td></td><td>+0.0044</td></tr><tr><td>wheelchair</td><td>0.2708</td><td>0.2664 0.2730</td><td>-0.0022</td></tr><tr><td>obama-self-2</td><td></td><td></td><td>+0.0000</td></tr><tr><td></td><td>0.2705</td><td>0.2705</td><td>+0.0027</td></tr><tr><td>dog-chip</td><td>0.2703</td><td>0.2676</td><td></td></tr><tr><td>cheburashka</td><td>0.2700</td><td>0.2705</td><td>-0.0005</td></tr><tr><td>ihylc</td><td>0.2700</td><td>0.2580</td><td>+0.0120</td></tr><tr><td>cat-toy Average</td><td>0.2695 0.2731</td><td>0.2600 0.2663</td><td>+0.0095 +0.0068</td></tr></table>

Table 29: HPSv2 comparison for 75 random concepts where each concept is evaluated for 6 different prompts.

GT Images  
![](images/a84ac183d53274521b355b56ff256fef21f740779cc4e56aef49a00ec829f8bd.jpg)

![](images/18d011fc8af7220cc9ccd06afaf9981c1a4cb9e9fd9f6c2879ee3b4213ce1e90.jpg)  
Composed Identity images with DMPMulti mode

![](images/94f0a6ef9e12945aec64acde5fed66bb3ccdaf3f8b0d276b39ab4b6f46e3bbfc.jpg)

![](images/fe65ef86262a77863eb627821c3bf126cec86b86032b9bb15c5e07b9d2482cd1.jpg)

Figure 21: DMPMulti composition: Images generated with DMPMulti model prompts (right) for the composition of the two identities shown on the left.  
![](images/533f74a9a0416484eb861b7b905a91d1a6c0b8c9b452f9f4cbb26652ac5bb24d.jpg)  
Figure 22: DMPMulti composition with SDv1.5 model: Prompts synthesized by DMP generalize well to different downstream models sharing the same CLIP text encoder without any re-training.

![](images/870faf45c4dbbfcac36b42d3bbe05fc465a83da114e4fa31aefe78a2e0f0735c.jpg)  
Figure 23: Interpolating between two faces using DMPMulti with concept composition.

Table 30: Hyperparameter Settings for the DMP models trained across three different tasks namely personalization, concepts and classification
<table><tr><td></td><td>DMPVariation</td><td>DMPMulti</td><td>DMPCoOp</td><td>Autoencoder</td></tr><tr><td>z-shape x-shape |Z|</td><td>768 x 1</td><td>一  $7 6 8 \mathrm { ~ x ~ } 1$ </td><td> $1 6 \times 8$   $2 0 4 8 \mathrm { ~ x ~ } 1$ </td><td> $1 6 \times 8$  2048 x 1</td></tr><tr><td></td><td></td><td>768</td><td>128</td><td>128</td></tr><tr><td>|x| Diffusion steps</td><td>768 1000</td><td>1000</td><td>2048 1000</td><td>768 1000</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Noise Schedule</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td></tr><tr><td>Nparams</td><td>33M</td><td>33M</td><td>33M</td><td>1M</td></tr><tr><td>Channels</td><td>128</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Depth</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Channel Multiplier</td><td>1,2,2,2</td><td>1,2,2,2</td><td>1,2,2,2</td><td>1,1,1,1,2,2,2,2</td></tr><tr><td>Attention resolutions</td><td>16,8,4</td><td>16,8,4</td><td>8,4,2</td><td>–</td></tr><tr><td>Head Channels</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Batch Size</td><td>320</td><td>320</td><td>256</td><td>128</td></tr><tr><td>Iterations</td><td>100k</td><td>100k</td><td></td><td></td></tr><tr><td>Learning Rate</td><td>1e-6</td><td>1e-6</td><td>100k 2.0e-7</td><td>100k 4.5e-6</td></tr></table>

![](images/dcceab858c807a2d5a9ae41dd68e25a472e263ad15e7afa8d89c05f424ca61ea.jpg)  
Figure 24: Qualitative results of generating novel identities during inference using random identity conditioning with DMPMulti model. The prompt used for the Stable Diffusion Realistic Vision model is "A photo of a id-x" where x is the id not present in the training set. The identities do not overfit (mean face ID similarity of 0.0102 across training images). Note that all the images use a fixed seed to the diffusion model.