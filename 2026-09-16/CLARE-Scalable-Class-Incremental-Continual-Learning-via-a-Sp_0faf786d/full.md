# CLARE: Scalable Class-Incremental Continual Learning via a Sparsity-Based Framework

Yunxiang Fu<sup>1</sup> Meng Lou<sup>1</sup> Zicheng Liao<sup>2</sup> Yizhou Yu<sup>1</sup> <sup>1</sup>School of Computing and Data Science, The University of Hong Kong <sup>2</sup>Hong Kong Generative AI Research and Development Center

yunxiang@connect.hku.hk, loumeng@connect.hku.hk

zichengliao@gmail.com, yizhouy@acm.org

## Abstract

Continual learning must balance the learning ofnew knowledge with the retention of previously learned knowledge to incrementally learn tasksfrom a data stream without catastrophic forgetting. While leveraging pretrained models has significantly advanced continual learning, existing meth ods exhibit a scalability bottleneck when trained sequentially on many tasks, suffering from performance degradation due to inter-task interference and loss of plasticity. Inspired by evidence that sparse fine-tuning achieves performance comparable to full fine-tuning, this paper presents a novel sparsity-driven continual learning framework. Our continual learning method, termed CLARE, operates in two stages: it first identifies a sparse, taskcritical parameter mask via a sparsity-inducing objective, then performs mask-constrained fine-tuning by only optimizing parameters selected by the mask. This two-stage sparse adapter mechanism enables all tasks to be accumulated within a shared adapter space while reducing destructive interference across tasks. Extensive experiments demonstrate the scalability of CLARE. On the long tasksequence benchmark Omnibenchmark-1k, CLARE outperforms strong baselines in final accuracy by a large margin, e.g, improving EASE by 4.64% and 13.34% after learning 100 tasks, respectively.

## 1. Introduction

The core challenge of continual learning (CL) lies in achieving a balance between the capacity to learn new diverse tasks (learning plasticity) and the ability to retain previously learned knowledge without catastrophic forgetting (memory stability). Class-incremental learning (CIL) stands as one of the most challenging settings in CL, requiring a model to incrementally learn new classes over time without accessing previous task data, while maintaining recognition capacity on all seen classes. Traditional CIL methods can often be categorized into three main paradigms: regularization-based methods [15, 18], replay-based methods [21], and optimization-based methods [5]. Recent advancements leverage strong pretrained models (PTM) to further improve performance instead of training models from scratch, as pretrained models contain rich prior knowledge learned from large-scale datasets [4, 41, 48]. In particular, two prominent directions include leveraging taskspecific parameters with a routing mechanism to use the most relevant parameters during inference [8, 35, 41, 46, 48] and merging task-specific parameters into a single set of parameters for all tasks [19, 31, 43]. While the former achieves stronger performance, it usually requires the number of stored task-specific parameters to increase linearly with the number of tasks and rely on an accurate routing mechanism to predict the task identity of inputs during inference [25]. In this work, we focus on the latter paradigm since it is efficient when there are many tasks to be learned, as it does not require storing task-specific parameters.

Although recent PTM-based CIL methods (InfLoRA [19], SD-LoRA [43]) that do not store task-specific parameters have shown promising performance on short task sequences (e.g., 10 or 20 tasks), scaling these methods to longer task sequences typically means substantial performance sacrifice [25]. This decline stems primarily from an imbalance between interference and plasticity. As the task sequence lengthens, effective new task learning causes catastrophic forgetting of earlier knowledge as new updates overwrite or conflict with parameters crucial for previous tasks. However, effective earlier knowledge preservation restricts a model’s capacity to integrate new information, resulting in progressively poorer performance on new tasks.

In this paper, we hypothesize that strategically learning a small number of parameters for each task can already maintain sufficient plasticity while dramatically reducing the likelihood of destructive interference across tasks. This hypothesis is supported by existing literature on learning sparse neural networks [27, 28, 42] as well as empirical evidence showing that the magnitude of parameter updates during fine-tuning follows a long-tailed distribution (Figure 1 (left)), with substantial updates being confined to a tiny subset of parameters. More importantly, it is only necessary to update a small proportion of the model parameters to achieve competitive task-specific performance, as illustrated in Figure 1 (right). On the basis of this insight, we propose a sparsity-driven continual learning framework that learns a sparse subset of parameters for each task, enabling effective scaling to extended sequences while balancing the plasticity-stability trade-off in continual learning models.

![](images/05299965c95e7a41d8d11e51625bca60bcfe5c6fcdebcf75edb5634f5ca88eb1.jpg)

![](images/f4a3888aaa258b96c222e106e8ff7191fcf5fa7ee5a395cf62bb6c08147b9785.jpg)  
Figure 1. Sparse parameter update analysis. Left: Long-tail distribution of parameter update magnitudes shows most parameters experience tiny updates (< 0.01), while only 8.5K parameters have updates ≥ 0.05. Right: Sparse parameter updates achieve performance close to full fine-tuning on ImageNet-R using ImageNet-1K pretrained ViT-B/16.

Our sparsity-driven framework manages parameter allocation across task sequences through a two-stage learning process. Starting from a pretrained base model, we first identify task-critical parameters by optimizing a sparsityinducing objective, which produces a binary mask identifying the most relevant parameters for the task. We then perform mask-constrained fine-tuning, updating only these relevant parameters while keeping the remainder frozen. This enables the model to achieve promising performance by updating only a sparse subset of the total parameters, thereby facilitating targeted knowledge acquisition with minimal interference and preserved plasticity. In practice, the parameters learned for new tasks are incrementally fused into the base model via simple accumulation for computational efficiency. Distinct from InfLoRA [19] and SD-LoRA [43], our method explicitly learns the subset of parameters that are most important to each task, instead of choosing them based on heuristics such as orthogonal constraints.

We evaluate CLARE through extensive experiments spanning both long and standard task-sequence settings. On long-sequence benchmarks, CLARE achieves the highest final accuracy of 66.88% after 100 tasks on

OmniBenchmark-1k [25], surpassing the strongest adapterbased baseline in final accuracy by 4.64% and the LoRAbased SD-LoRA [43] by over 38%. On 50-task splits of ImageNet-R and ImageNet-A, CLARE outperforms InfLoRA[19] by 13.9% and 19.28% points in final accuracy, respectively. On standard class-incremental benchmarks with 10 and 20 tasks, CLARE attains the highest final accuracy across all six evaluated settings, with particularly large margins on datasets with distribution shift: on ImageNet-A and ObjectNet, it improves final accuracy by 5.40% and 8.40% points over the strongest baseline, respectively. These results demonstrate that capacity-aware sparse adapter learning scales effectively from short to long task sequences and provides a unified solution that does not require task-specific routing or task identity at inference time.

In summary, our contributions are:

• We propose CLARE, a sparsity-driven adapter-based continual learning framework that learns task-critical sparse adapter masks before mask-constrained task learning.

• We explicitly learn the task-critical mask using an L<sub>1</sub>- regularized objective to induce natural sparsity in taskspecific parameter updates, allowing each task to discover a compact set of task-critical parameters.

• We provide extensive evaluations and ablations showing how learned sparsity, two-stage optimization, and adapter capacity affect performance across short, medium, and long task sequences.

## 2. Related work

Pretrained Model-Based CIL Traditional CIL methods address catastrophic forgetting through regularization constraints [15, 18], rehearsal strategies that retain exemplars from previous tasks [33], gradient constraints [21], and parameter-isolation mechanisms [1, 29]. Inspired by the development of strong representations learned by pretrained models (PTM) for vision [4, 7, 22–24, 32, 34] and their applicability to different tasks [2, 6, 11, 17, 26], recent CIL methods increasingly build on frozen or partially tuned pretrained vision transformers [25, 41, 48]. Prompt-based methods such as L2P [41], DualPrompt [40], and CODA-Prompt [35] learn task-related prompt tokens and retrieve them at inference. Adapter-based methods further improve plasticity by learning lightweight task-specific modules, as in EASE [48] and SEMA [38]. However, these methods often store task-specific parameters and depend on accurate retrieval or routing during inference, which becomes increasingly challenging as the task sequence grows. In contrast, our work focuses on learning sparse task updates that can be merged into a single task-agnostic model.

Parameter-Efficient CIL Another line of PTM-based CIL seeks to avoid task-specific routing by combining taskspecific updates into one model for inference. Works like InfLoRA [19], SD-LoRA [43], and LoDA [10] learn taskspecific LoRA modules and impose constraints to reduce interference among tasks. MagMax [31] merges task-specific parameter updates into a shared model, using predefined or post-hoc rules such as random pruning or magnitudebased selection. There are also conceptually works that focus on LLM tasks, including OA-Adapter [37], Share [14], CSBoRA [20], OPLoRA [44]. These methods are free of inference-time routing, but do not explicitly learn which parameters should be updated for each task before task learning, leading to suboptimal learning plasticity. In contrast, our method uses an $L _ { 1 }$ -regularized objective to induce naturally sparse task updates, then uses the discovered mask for constrained training and merging.

Sparse and Mask-Based Continual Learning Sparsity has also been widely studied in continual learning through subnetwork selection, pruning, and mask learning. Piggyback [30] learns task-specific binary masks over a fixed backbone, while PackNet [29] progressively prunes and allocates parameters for new tasks. Other sparse or subnetwork-based methods similarly reduce forgetting by assigning different parameter subsets to different tasks [1, 36, 39, 45]. These methods usually maintain task-specific masks or sparse subnetworks and often require task identity or task-specific selection during inference. For example, PackNet [29] and Piggyback [30] keep a per-task mask/subnetwork and, at test time, select the right one using the task identity or an arg-max over stored masks. They also rely on knowing the number of tasks in advance.

Distinctively, CLARE’s core novelty is that sparsity is a training-time allocation mechanism, not an inference-time selection mechanism. CLARE keeps no per-task mask at inference. After each task its sparse update is merged into a single shared adapter $\begin{array} { r } { \alpha _ { \mathrm { s h a r e d } } = \alpha _ { 0 } + \sum _ { t } M _ { t } \odot ( \alpha _ { t } - \alpha _ { 0 } ) } \end{array}$ every input uses that same adapter with all coordinates active, and no task identity or routing is used. The mask $M _ { t }$ only decides which currently-unused coordinates a new task t may modify. A direct consequence is that earlier tasks’ coordinates remain active for every later input, so prior knowledge is reused rather than gated. This gives CLARE high learning plasticity (average incremental accuracy), not merely forgetting mitigation. This also makes sparse parameter selection part of the learning objective rather than a hand-designed or purely post-hoc pruning rule.

## 3. Method

## 3.1. Problem Definition

We study exemplar-free class-incremental learning. The model receives a sequence of image classification tasks $\{ \mathcal { D } _ { 1 } , \hdots , \mathcal { D } _ { T } \}$ . Task t contains training samples $\begin{array} { r l } { \mathcal { D } _ { t } } & { { } = } \end{array}$ $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { t } }$ whose labels belong to a new class set $\mathcal { C } _ { t }$ . The class sets are disjoint for different tasks, so ${ \mathcal { C } } _ { i } \cap { \mathcal { C } } _ { j } = \emptyset$ for $i \neq j$ . During task $t ,$ the model can only access $\mathcal { D } _ { t }$ and cannot replay samples from previous tasks. After learning task $t ,$ the model must classify test samples from all seen classes $\textstyle { \mathcal { V } } _ { t } = \bigcup _ { i = 1 } ^ { t } { \mathcal { C } } _ { i }$

We build CLARE as a parameter-efficient CIL method on top of a pretrained vision transformer (ViT-B/16 [4]). The pretrained backbone parameters are denoted by $\theta _ { 0 }$ and kept frozen. We insert lightweight adapters [3] into each transformer block and denote all adapter parameters by α. The classifier after task t is denoted by $\phi _ { t } ,$ , and the prediction function is written as $f ( x ; \theta _ { 0 } , \alpha , \phi _ { t } )$ . CLARE only updates the adapters and classifier. It does not fine-tune the full backbone.

Figure 2 gives an overview. CLARE keeps one shared adapter across the whole task sequence. For each task, it first explicitly learns a parameter-importance sparse mask for the current task using an $L _ { 1 }$ -regularized objective. Subsequently, it trains only the selected adapter parameters. After task learning, the masked task update is added to the shared adapter, and the next task starts from this updated adapter.

## 3.2. Capacity-Aware Sparse Adapter Learning

CLARE uses one shared adapter to learn all tasks. This design is parameter-efficient, but it may create a clear source of forgetting. Specifically, if a new task changes adapter parameters that are important for old tasks, the feature representation of historical classes can shift. The classifier was trained on the historical representations, so this shift can damage old-task predictions and cause catastrophic forgetting. The problem becomes more severe in long task sequences because more tasks compete for the same limited trainable parameters.

![](images/88abde1972e87cf7de9440fe1e4d685cd61fc89f50b5bdfeee78315e2eae8d47.jpg)  
Figure 2. (a) A shared adapter contains parameters that have been frozen after previous tasks (blue) and parameters that remain available for future tasks (orange). (b) For a new task, CLARE first selects a sparse set of parameters from the available pool (Stage 1), then resets them to their original values and optimizes only those selected parameters (Stage 2). After training, the updated parameters are frozen and added to the frozen set.

To control this interference, CLARE explicitly manages which scalar entries of the adapter parameters can be updated by each task. We call each scalar entry an adapter coordinate. Let $\Omega = \{ 1 , \dots , N \}$ index all adapter coordinates, where N is the total number of adapter parameters. After several tasks have been learned, some coordinates have already been selected by previous task masks. For task t, let $\begin{array} { r } { \Omega _ { < t } = \bigcup _ { i = 1 } ^ { t - 1 } \Omega _ { i } } \end{array}$ <sub>i</sub> be the set of coordinates used by previous tasks, and let

$$
\mathcal { A } _ { t } = \Omega \backslash \Omega _ { < t }\tag{1}
$$

be the set of coordinates still available for task t. CLARE selects a sparse set $\Omega _ { t } \subseteq A _ { t }$ for the current task. In this way, new tasks are guided toward unused adapter parameters instead of freely overwriting parameters already assigned to earlier tasks.

Importantly, since knowledge of all historical tasks are learned in $\Omega _ { < t } ,$ , we encourage $\Omega _ { t }$ to build on the accumulated adapter state from previous tasks and use minimal sparse updates for the new task t. Specifically, the percentage of $\boldsymbol { A } _ { t }$ that can be updated for task t (sparsity ratio) is set to a fixed value. For example, $\rho = 0 . 9 5$ means that task t only uses $1 - \rho = 5 \%$ of $\boldsymbol { A } _ { t }$ to learn, not 5% of the original adapter capacity for every task. If $F _ { t } = | A _ { t } |$ is the number of available coordinates before task t, then

$$
F _ { t + 1 } \approx \rho F _ { t } , \qquad F _ { t } \approx \rho ^ { t - 1 } N .\tag{2}
$$

After T tasks, the used capacity is approximately $1 - \rho ^ { T }$ For $\rho = 0 . 9 5$ and $T = 1 0 0$ , this value is $1 - 0 . 9 5 ^ { 1 0 0 } \approx$ 0.994. The adapter is nearly saturated after 100 tasks, but the method does not suffer from interference where each parameter coordinate is used by $5 \% \times 1 0 0 = 5$ different tasks on average when assigning a fixed 5% budget to every task. This geometric schedule slows capacity consumption and explains why a single adapter can support a 100-task sequence.

Let $\alpha _ { 0 }$ denote the initial adapter parameters before any continual task is learned. For task t, CLARE learns an adapter state $\alpha _ { t }$ and defines the task-specific adapter update as

$$
\Delta \alpha _ { t } = \alpha _ { t } - \alpha _ { 0 } .\tag{3}
$$

Sparsity is imposed on $\Delta \alpha _ { t }$ , not on the full pretrained backbone. Since a task modifies only free coordinates $\boldsymbol { A } _ { t }$ in Equation 1 while used ones are frozen. In this trainable subspace, the previous shared adapter equals $\alpha _ { 0 } .$ , so defining $\Delta \alpha _ { t }$ w.r.t. $\alpha _ { 0 }$ or the previous shared state is equivalent.

After each task, the sparse update is immediately added to the shared adapter. For tasks after the first one, CLARE starts from this learned shared adapter instead of returning to an independent adapter. The next task can therefore use the representation accumulated so far, while the freecoordinate mask limits direct overwriting of earlier task updates.

## 3.3. Two-Stage Sparse Update Optimization

We now describe how the sparse adapter coordinate set $\Omega _ { t }$ and the adapter update $\Delta \alpha _ { t }$ are learned for each task t. CLARE uses a two-stage approach: the first stage selects $\Omega _ { t }$ from the currently available coordinates, and the second stage optimizes $\Delta \alpha _ { t }$ while restricting adapter updates to $\Omega _ { t }$

Stage 1: $L _ { 1 }$ -induced mask discovery. For task t, the first stage decides which available adapter coordinates should be assigned to the current task. CLARE starts from the current shared adapter and updates only the coordinates in $\boldsymbol { A } _ { t }$ for a few epochs using the current task data. The objective is

$$
\operatorname* { m i n } _ { \alpha } \ \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { t } } \left[ \ell \left( f ( x ; \theta _ { 0 } , \alpha , \phi _ { t } ) , y \right) \right] + \lambda \left\| P _ { { A } _ { t } } ( \alpha - \alpha _ { 0 } ) \right\| _ { 1 } ,\tag{4}
$$

where $P _ { \boldsymbol { A } _ { t } } ( \cdot )$ keeps only currently free adapter coordinates and λ controls the strength of the sparsity term. The classifier provides the task loss, while the purpose of this stage is to reveal which adapter coordinates respond to the new task. The $L _ { 1 }$ penalty encourages most free-coordinate updates to stay close to zero. As a result, a coordinate that still changes by a large amount is likely to be important for learning task t, while coordinates with near-zero changes can be left unused.

Using the adapter values reached at the end of this short optimization, CLARE scores each available coordinate by the magnitude of this movement:

$$
s _ { t } ^ { ( j ) } = \left| \alpha ^ { ( j ) } - \alpha _ { 0 } ^ { ( j ) } \right| , \qquad j \in \mathcal { A } _ { t } .\tag{5}
$$

Larger scores mean that the coordinate moved more even under the $L _ { 1 }$ penalty, so CLARE treats them as more useful for the current task.

The number of coordinates selected for task t follows the remaining-capacity schedule in the previous subsection. The target budget is

$$
\bar { k } _ { t } = \mathrm { r o u n d } \left( ( 1 - \rho ) | \boldsymbol { \mathcal { A } } _ { t } | \right) .\tag{6}
$$

Since ${ { \bar { k } } _ { t } }$ may be zero after rounding when few coordinates remain, CLARE clips it to a valid integer budget:

$$
k _ { t } = \operatorname* { m i n } \left\{ | A _ { t } | , \operatorname* { m a x } \left\{ 1 , \bar { k } _ { t } \right\} \right\} , \qquad \mathrm { w h e n } \ | A _ { t } | > 0 .\tag{7}
$$

If no available coordinate remains, CLARE assigns an empty mask. Otherwise, CLARE ranks all scores $\{ s _ { t } ^ { ( j ) }$ $j \in \mathcal { A } _ { t } \}$ globally and sets $\tau _ { t }$ to the k -th largest score. The selected coordinate set is

$$
\Omega _ { t } = \left\{ j \in \mathcal { A } _ { t } : s _ { t } ^ { ( j ) } \geq \tau _ { t } \right\} .\tag{8}
$$

Equivalently, the binary mask is $M _ { t } ^ { ( j ) } = \mathbb { I } [ j \in \Omega _ { t } ]$ . If several coordinates have exactly the same score as $\tau _ { t }$ , they are all included, so the mask size can differ slightly from $k _ { t }$ in the rare case of ties. This global top-k<sub>t</sub> rule has two purposes. First, it gives each task only a fixed fraction of the coordinates still available, which preserves capacity for future tasks. Second, it lets the task use that capacity wherever the learned update indicates it is most useful, instead of forcing every adapter layer to receive the same budget.

Excluding $\Omega _ { < t }$ further prevents direct reuse of coordinates already assigned to previous tasks.

Stage 2: mask-constrained learning. The adapter values obtained in Stage 1 are not used as the final task parameters. CLARE restores the adapter to the state used at the beginning of Stage 1: the initial adapter for the first task and the shared adapter learned so far for later tasks. It then learns task t while updating only the coordinates in $\Omega _ { t } .$ . Equivalently, for any adapter coordinate j not selected in Stage 1, CLARE sets the gradient of $\alpha _ { t } ^ { ( j ) }$ to zero before each optimizer step:

$$
\begin{array} { r l } & { \underset { \alpha _ { t } , \phi _ { t } } { \mathrm { m i n } } \ \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) \sim \mathcal { D } _ { t } } \left[ \ell _ { \mathrm { c l s } } \left( f ( \boldsymbol { x } ; \boldsymbol { \theta } _ { 0 } , \alpha _ { t } , \boldsymbol { \phi } _ { t } ) , \boldsymbol { y } \right) \right] , } \\ & { \nabla _ { \alpha _ { t } ^ { ( j ) } } = 0 \quad \mathrm { i f } j \not \in \Omega _ { t } . } \end{array}\tag{9}
$$

Here $\nabla _ { \alpha _ { t } ^ { ( j ) } }$ denotes the gradient with respect to the $j -$ th scalar adapter coordinate after the task loss is backpropagated. Only coordinates in $\Omega _ { t }$ receive adapter updates, while all other adapter coordinates remain fixed. This reset is important: Stage 1 is designed to select coordinates under an $L _ { 1 }$ pressure, whereas Stage 2 is designed to learn the task well once the coordinate budget has been fixed. Removing the $L _ { 1 }$ term in Stage 2 prevents the sparsity objective from weakening classification learning. In practice, the classifier is expanded for the new classes and trained together with the active adapter coordinates.

## 3.4. Sequential Sparse Update and Inference

After task learning, CLARE stores the masked adapter delta

$$
\Delta \alpha _ { t } ^ { \mathrm { s p a r s e } } = M _ { t } \odot ( \alpha _ { t } - \alpha _ { 0 } ) .\tag{10}
$$

where $M _ { t } ^ { ( j ) } = \mathbb { I } [ j \in \Omega _ { t } ]$ . Thus, only the coordinates selected for task t are written into the task update. All other coordinates contribute zero. The shared adapter used after task t is obtained by adding the sparse task deltas to the initial adapter state:

$$
\alpha _ { \mathrm { s h a r e d } } ^ { ( t ) } = \alpha _ { 0 } + \sum _ { i = 1 } ^ { t } \Delta \alpha _ { i } ^ { \mathrm { s p a r s e } } .\tag{11}
$$

Because each task selects coordinates from the remaining free capacity, the shared adapter accumulates task knowledge while reducing direct overlap among task updates. The update is performed sequentially after each task, so the next task always starts from the adapter learned so far.

At inference time, CLARE uses the frozen backbone $\theta _ { 0 }$ the shared adapter $\alpha _ { \mathrm { s h a r e d } } ^ { ( t ) }$ , and a single classifier $\phi _ { t }$ over all seen classes:

$$
\hat { y } = \arg \operatorname* { m a x } _ { c \in \mathcal { V } _ { t } } f _ { c } ( x ; \theta _ { 0 } , \alpha _ { \mathrm { s h a r e d } } ^ { ( t ) } , \phi _ { t } ) .\tag{12}
$$

Throughout continual learning, we maintain only one set of adapter weights and do not require task identity for inference. We use a cosine classifier whose class weights grow as new classes arrive following standard implementations [48].

CLARE differs from existing sparse and parameterefficient continual learning methods in two ways: (1) Its sparsity is induced during learning through a $L _ { 1 ^ { - } }$ regularization. (2) The mask is capacity-aware since each task selects from the remaining free coordinates. These properties allow CLARE to use a single adapter to scale to long task sequences.

## 4. Experiments

We evaluate CLARE in exemplar-free class-incremental learning (CIL) under two settings ranging from 10 tasks to 100 tasks. The main setting studies the challenging long task sequence setup, where methods must preserve performance as the number of incremental tasks grows to 100. The second setting leverages standard CIL protocols on established datasets to verify that the same sparse shared-adapter design in CLARE remains competitive under shorter sequences. We then comprehensively investigate the effect of each component of CLARE through controlled ablation studies.

## 4.1. Experimental Setup

Datasets. Long-sequence evaluation uses ImageNet-R [12] and ImageNet-A [13] split into 50 tasks with 4 classes per task, and OmniBenchmark-1k [25] split into 100 tasks with 10 classes per task. We denote IncN as the number of novel classes learned per task. Following previous works [19, 43, 48, 49], standard benchmark evaluation uses ImageNet-R [12] with 10 tasks (Inc20) and 20 tasks (Inc10), CIFAR-100 [16] with 10 tasks (Inc10) and 20 tasks (Inc5), ImageNet-A [13] with 10 tasks (Inc20), and ObjectNet with 10 tasks (Inc20).

Baselines. We compare with representative pretrainedmodel CIL methods from different families, including prompt-based (L2P [41], DualPrompt [40], CODA-Prompt [35]), adapter-based [8, 38, 48, 49], classifier-based [9, 47] and LoRA-based [19, 43]. These baselines cover methods that use task-specific parameters or prompts, and methods that maintain a single model at inference time. All adapterbased and LoRA-based baselines are run from their official implementations.

Implementation details. CLARE follows prior works [19, 43, 48, 49] to use a ViT-B/16 pretrained on ImageNet-21K as the frozen backbone and trains lightweight adapters together with the classifier. During training, we use a learning rate of 0.02 with cosine learning rate decay, a batch size of 32, and a SGD optimizer with weight decay of 0.0005. The adapter bottleneck dimension is 64, the first-stage mask discovery is trained for 5 epochs with an $L _ { 1 }$ coefficient of $1 0 ^ { - 4 }$ , and the second-stage masked training is run for 20 epochs. The sparsity ratio is set to 95%.

Metrics. We report the average incremental accuracy A<sup>¯</sup> and the final accuracy $A _ { T }$ . Here, $A _ { T }$ is the accuracy over all seen classes after training the final task, and A<sup>¯</sup> is the mean of the accuracies measured after each incremental task. Higher values are better for both metrics.

## 4.2. Long Task Sequence Evaluation

Table 1 evaluates long task sequences, a setting where single adapter-based models that do not leverage task identity at inference struggle (e.g., InfLoRA, SD-LoRA). CLARE consistently achieves the highest final accuracy and average incremental accuracy. On the longest OmniBenchmark-1k benchmark comprising 100 tasks, CLARE attains a final accuracy of 66.88%, which represents a substantial 137% relative improvement over SD-LoRA [43]. This gain suggests that the capacity-aware sparsity introduced by CLARE can mitigate catastrophic forgetting even for 100 tasks. Compared to Aper [49], which trains the adapter only on the first task, freezes it for subsequent tasks, and relies on class prototypes for prediction, CLARE yields significant improvements of 11.69% on 50-task ImageNet-R and 9.4% on 50- task ImageNet-A, respectively. This demonstrates stronger learning plasticity of CLARE. These results indicate that CLARE can maintain a balance between learning plasticity and knowledge retention even over long task sequences.

## 4.3. Standard Benchmark Evaluation

Table 2 evaluates CLARE under the commonly used CIL settings with 10 and 20 tasks [19, 35, 38, 41, 43, 48]. Across all six reported settings over four datasets, CLARE achieves the highest final accuracy among the compared methods. The gains are especially clear on the datasets with larger distribution shifts: on ImageNet-A Inc20, CLARE improves $A _ { T }$ from 55.96% for the strongest baseline (SD-LoRA) to 61.36%, and on ObjectNet Inc20, it improves $A _ { T }$ from 59.37 to 67.77. The results on ImageNet-R and CIFAR-100 show a similar pattern under both 10 and 20 task splits. On ImageNet-R, CLARE improves final accuracy over the strongest baseline by 2.47% in the 20-task Inc10 setting. On CIFAR-100 with 10 tasks, the improvement is 3.91%. Together with Table 1, these consistent gains across different number of tasks indicate that constraining adapter updates to sparse, selected coordinates in CLARE is a scalable and can generalize to various ranges of tasks.

## 4.4. Ablation Studies

We investigate each design component of CLARE, which are the capacity-aware sparse adapter learning for determining how many parameters to use for the current task and two-stage sparse update optimization for maximizing learning plasticity using the amount of parameters available. Table 3 shows results on ImageNet-R with 10 tasks and Omnibenchmark-1k with 100 tasks. In the second row of Table 3, we enforce a constant predefined portion of parameters for each task. We report the best results from over four different sparsity ratios {10%, 5%, 2.5%, 1%} for both datasets. It can be observed that on the 100 task setting, the final accuracy drops significantly by 13.71% to 53.17%, demonstrating the importance of capacity-aware sparsity. In the third row (w/o two-stage), we remove the two-stage learning process and directly train the adapter, and select the parameters for each task based on its magnitude. It can be seen that final accuracy decreased for both settings, suggesting that the two stage optimization is beneficial.

Table 1. Performance comparison on benchmarks with long task sequences. We report average incremental accuracy A<sup>¯</sup> and final accuracy $A _ { T }$
<table><tr><td>Method</td><td>ImageNet-R 50 Tasks (Inc4) À  ${ \bf A } _ { T }$ </td><td>À</td><td>ImageNet-A 50 Tasks (Inc4)  ${ \bf A } _ { T }$ </td><td></td><td>OmniBenchmark-1k 100 Tasks (Inc10) À  ${ \bf A } _ { T }$ </td></tr><tr><td>L2P(CVPR&#x27;22)</td><td>69.16</td><td>63.45</td><td>49.89</td><td>36.41</td><td>60.91 48.87</td></tr><tr><td>DualPrompt(ECCV&#x27;22)</td><td>64.00</td><td>56.33</td><td>43.85</td><td>29.95 62.18</td><td>49.45</td></tr><tr><td>CODA-Prompt(CVPR&#x27;23)</td><td>62.43</td><td>57.57</td><td>38.24</td><td>26.60 64.16</td><td>51.75</td></tr><tr><td>EASE(CVPR&#x27;24)</td><td>78.11</td><td>70.63</td><td>59.86</td><td>47.53 65.00</td><td>53.54</td></tr><tr><td> $\mathbf { S E M A } _ { ( \mathbf { C V P R } ^ { \prime } 2 5 ) }$ </td><td>67.80</td><td>59.32</td><td>52.99</td><td>40.68 56.55</td><td>33.96</td></tr><tr><td>APER-Adapter(IJCV&#x27;25)</td><td>72.43</td><td>64.83</td><td>61.29</td><td>48.58 73.23</td><td>62.24</td></tr><tr><td>InfLoRA(CVPR&#x27;24)</td><td>71.68</td><td>62.62</td><td>50.17</td><td>38.70 51.53</td><td>27.01</td></tr><tr><td> $\mathrm { S D - L o R A _ { ( I C L R ^ { \prime } 2 5 ) } }$ </td><td>68.40</td><td>63.28</td><td>54.72</td><td>41.28 53.97</td><td>28.15</td></tr><tr><td>CLARE (Ours)</td><td>83.00</td><td>76.52</td><td>68.83</td><td>57.98 78.32</td><td>66.88</td></tr></table>

Table 2. Performance comparison with state-of-the-art methods on standard CIL benchmarks. We report average incremental accuracy A<sup>¯</sup> and final accuracy $A _ { T }$

<table><tr><td rowspan="3">Method</td><td colspan="4">ImageNet-R</td><td colspan="4">CIFAR-100</td><td rowspan="2" colspan="2">ImageNet-A 10 Tasks (Inc20)</td><td rowspan="2" colspan="2">ObjectNet 10 Tasks (Inc20)</td></tr><tr><td colspan="2">10 Tasks (Inc20)</td><td colspan="2">20 Tasks (Inc10)</td><td colspan="2">10 Tasks (Inc10)</td><td colspan="2">20 Tasks (Inc5)</td></tr><tr><td>À</td><td> ${ \bf A } _ { T }$ </td><td>Á</td><td> ${ \bf A } _ { T }$ </td><td>À</td><td> ${ \bf A } _ { T }$ </td><td>À</td><td> ${ \bf A } _ { T }$ </td><td>Á</td><td> ${ \bf A } _ { T }$ </td><td>Ã</td><td> ${ \bf A } _ { T }$ </td></tr><tr><td>L2P(CVPR&#x27;22)</td><td>75.46</td><td>69.77</td><td>63.75</td><td>55.78</td><td>85.92</td><td>79.19</td><td>85.94</td><td>79.93</td><td>49.39</td><td>41.71</td><td>66.77</td><td>55.16</td></tr><tr><td>DualPrompt(ECCV&#x27;22)</td><td>73.10</td><td>67.18</td><td>66.52</td><td>61.77</td><td>89.65</td><td>84.89</td><td>87.87</td><td>81.15</td><td>53.71</td><td>41.67</td><td>64.31</td><td>52.99</td></tr><tr><td>CODA-Prompt(CVPR&#x27;23)</td><td>77.97</td><td>72.27</td><td>70.45</td><td>64.68</td><td>91.05</td><td>86.44</td><td>89.11</td><td>81.96</td><td>53.54</td><td>42.73</td><td>66.53</td><td>56.80</td></tr><tr><td>EASE(CVPR&#x27;24)</td><td>81.74</td><td>76.17</td><td>81.18</td><td>74.62</td><td>92.11</td><td>87.72</td><td>91.51</td><td>85.80</td><td>65.34</td><td>55.04</td><td>71.04</td><td>59.37</td></tr><tr><td>SEMA(CVPR&#x27;25)</td><td>81.39</td><td>77.84</td><td>77.84</td><td>69.60</td><td>91.60</td><td>86.75</td><td>92.23</td><td>87.84</td><td>63.83</td><td>52.21</td><td>67.95</td><td>54.92</td></tr><tr><td>APER-Adapter(IJCV&#x27;25)</td><td>75.82</td><td>67.95</td><td>72.35</td><td>64.33</td><td>92.22</td><td>87.45</td><td>90.65</td><td>85.15</td><td>60.53</td><td>49.57</td><td>69.24</td><td>57.41</td></tr><tr><td>InfLoRA(CVPR&#x27;24)</td><td>80.82</td><td>75.65</td><td>77.28</td><td>71.01</td><td>91.70</td><td>86.51</td><td>89.13</td><td>81.46</td><td>58.50</td><td>46.28</td><td>70.67</td><td>58.04</td></tr><tr><td>SD-LoRA(ICLR*25)</td><td>82.04</td><td>77.34</td><td>80.22</td><td>75.26</td><td>92.54</td><td>88.01</td><td>90.90</td><td>85.18</td><td>64.95</td><td>55.96</td><td>70.37</td><td>58.54</td></tr><tr><td>CLARE (Ours)</td><td>83.88</td><td>79.73</td><td>82.67</td><td>77.73</td><td>94.73</td><td>91.92</td><td>94.96</td><td>91.73</td><td>69.46</td><td>61.36</td><td>76.67</td><td>67.77</td></tr></table>

Impact of the Sparsity Ratio . Table 4 shows the impact of the sparsity ratio for 10 tasks (ImageNet-R) and 100 tasks (Omnibenchmark-1k). We observe that CLARE is robust to the sparsity ratio when choosing from 95% to 97.5%, where the declines in final accuracy are not substantial (-0.38% for 10 tasks and -0.41% for 100 tasks). We note that choosing

We further investigate design choices and important hyperparameters within each component. Specifically, we study the effect of the sparsity ratio, the training strategies for the two-stage sparse update optimization, and the effect of the $L _ { 1 }$ normalization coefficient λ.

the sparsity ratio to be smaller (90%) can improve performance on the short 10 task setting, but lead to substantial decline in performance in the 100 task setting. This decline is due to insufficient number of parameters to learn new task when the number of tasks increases to 100.

Training Strategies for Two-Stage Learning . Table 5 examines the effect of different mask selection and optimization strategies, as well as the sensitivity to the sparsity coefficient λ. Removing the sparse mask entirely (no sparse mask) causes severe forgetting, with final accuracy collapsing to near zero on both benchmarks, confirming that restricting adapter updates is essential for retaining previous knowledge. Replacing the learned mask with a random selection of the same number of available coordinates (random mask) reduces final accuracy by 0.51% on ImageNet-R and 5.65% on OmniBenchmark-1k, indicating that the $L _ { 1 } -$ guided mask identifies task-relevant coordinates are beneficial. When each task is trained independently from the initial pretrained adapter, without leveraging prior accumulated knowledge in the shared adapter (independent tuning), performance drops drastically to 32.33% and 11.03% final accuracy, respectively. This collapse shows that knowledge accumulation across tasks is critical and that sparse masking alone is insufficient without a shared representation.

Table 3. Impact of capacity-aware sparsity and two stage learning on ImageNet-R with 10 tasks and Omnibenchmark-1K with 100 tasks.
<table><tr><td></td><td colspan="3">ImageNet-R</td><td colspan="3">Omnibenchmark-1K</td></tr><tr><td>Variant</td><td>A</td><td> $A _ { T }$ </td><td> $\Delta A _ { T }$ </td><td> $\bar { A }$ </td><td> $A _ { T }$ </td><td> $\Delta A _ { T }$ </td></tr><tr><td>CLARE</td><td>83.88</td><td>79.73</td><td>0.00</td><td>78.32</td><td>66.88</td><td>0.00</td></tr><tr><td>w/o capacity-aware sparsity</td><td>83.94</td><td>79.60</td><td>-0.13</td><td>64.20</td><td>53.17</td><td>-13.71</td></tr><tr><td>w/o two-stage learning</td><td>83.25</td><td>78.96</td><td>-1.08</td><td>75.19</td><td>63.09</td><td>-3.79</td></tr></table>

Table 4. Impact of the sparsity ratio.

<table><tr><td colspan="4">ImageNet-R</td><td colspan="3">Omnibenchmark-1K</td></tr><tr><td>Variant</td><td>À</td><td> $A _ { T }$ </td><td> $\Delta A _ { T }$ </td><td>À</td><td> $A _ { T }$ </td><td> $\Delta A _ { T }$ </td></tr><tr><td>90%</td><td>83.88</td><td>84.04</td><td>+0.45</td><td>73.49</td><td>59.25</td><td>-7.63</td></tr><tr><td>95%</td><td>83.88</td><td>79.73</td><td>0.00</td><td>78.32</td><td>66.88</td><td>0.00</td></tr><tr><td>97.5%</td><td>82.38</td><td>79.35</td><td>-0.38</td><td>79.03</td><td>66.47</td><td>-0.41</td></tr><tr><td>99%</td><td>79.62</td><td>73.14</td><td>-6.59</td><td>77.62</td><td>65.59</td><td>-1.29</td></tr></table>

We further study alternatives for the first stage of the two-stage optimization. Using $L _ { 2 }$ regularization instead of $L _ { 1 } ~ ( \mathrm { S t a g e } { \cdot } 1 ~ L _ { 2 } )$ yields a modest decline, notably a 1.52% lower final accuracy on the 100-task setting, suggesting that the sparsity-inducing property of $L _ { 1 }$ is better suited for selecting a compact set of coordinates. Removing the sparsity term entirely $( \lambda = 0 )$ and selecting coordinates solely by update magnitude after unregularized probing reduces final accuracy by 0.88% and 1.91%, respectively, confirming that the $L _ { 1 }$ penalty helps identify the most essential adapter parameters for each task. Varying λ around the default value of $0 . 0 0 0 1 ( \lambda = 0 . 0 0 0 0 1$ and $\lambda = 0 . 0 0 1 )$ yields stable final accuracies, demonstrating that the method is robust to the choice of this hyperparameter.

Robustness to Seeds In the main experiments and ablations of the paper, we partition each dataset by randomly shuffling the class order using the seed 1993, following EASE. In Table 6, we evaluate the performance of CLARE and a representative baseline SD-LoRA on three random seed and report the mean±std. We observe that variance is small and final accuracy is close to that of the seed 1993 so CLARE is robust to different orders across different benchmarks.

Impact of Adapter Capacity . The adapter bottleneck dimension d controls the total number of trainable adapter coordinates, i.e., the size of the pool from which each task selects its sparse mask. We investigate the impact of d on CLARE’s gains on long task sequences. Specifically, we vary $d \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 \}$ on ImageNet-R with 50 tasks.

Table 7 shows that final accuracy is nearly insensitive to adapter width. Reducing the bottleneck to d = 16 (one quarter of the default width) changes $A _ { T }$ by only −0.11 points, while doubling it to d = 128 improves $A _ { T }$ by only +0.13 points. Across this eight-fold range, the spread in final accuracy is 0.27 points, and even $d \ = \ 1 6$ remains well above the strongest baselines on the same split in Table 1 (EASE 70.63, InfLoRA 62.62). This 50-task setting is a stringent test of capacity: with $\rho \ : = \ : 0 . 9 5$ , about $1 - 0 . 9 5 ^ { 5 0 } \approx 9 2 \%$ of the adapter has already been assigned, so extra width would help if the method were limited by the number of parameters. The nearly unchanged $A _ { T }$ indicates that it is not. CLARE’s improvements instead come from assigning each task a sparse subset of the remaining coordinates, which reduces destructive interference even when the adapter itself is small.

## 4.5. Efficiency

CLARE keeps a single shared adapter and does not use task identity or routing at inference, so it adds no inference-time overhead relative to a standard adapter model. Although Stage 1 spends a few epochs on mask discovery, Stage 2 updates only the selected sparse coordinates, which keeps training efficient. On 50-task ImageNet-A, the total training time of CLARE is 4.37 hours on a single NVIDIA L40 GPU, slightly faster than SD-LoRA (4.41 hours) under the same setting.

## 5. Limitations

CLARE is designed so that the sequence length T need not be known in advance. Rather than splitting the adapter into a predefined number of task slots, each new task selects a sparse subset of the coordinates that have not yet been assigned. The consumed capacity therefore grows as $1 - \rho ^ { T }$ . With the default $\rho = 0 . 9 5$ , a single adapter is only about 99.4% full after 100 tasks, which is enough to remain accurate from the standard 10–20 task setting through the 100-task OmniBenchmark-1k protocol. The limitation is that this pool is finite and cannot scale indefinitely to very large number of tasks like 10k tasks. On substantially longer streams the remaining free coordinates will run out, and new tasks would then have little unused capacity left to learn. Expanding capacity at that point does not require changing the allocation principle. One can freeze a saturated adapter and continue with a fresh one, or increase the bottleneck when free coordinates become scarce, then apply the same remaining-capacity schedule. We leave such expansion to future work. Notably, prior works like SD-LoRA mostly focus on 10 to 20 tasks, while our proposed CLARE can learn effectively for 100 tasks.

Table 5. Impact of mask selection strategies, two-stage training, and $L _ { 1 }$ coefficient λ on ImageNet-R with 10 tasks and Omnibenchmark 1K with 100 tasks. independent tuning trains each task from the initial (not shared) adapter, then merges.
<table><tr><td></td><td colspan="3">ImageNet-R</td><td colspan="3">Omnibenchmark-1K</td></tr><tr><td>Variant</td><td> $\bar { A }$ </td><td> $A _ { T }$ </td><td> $\Delta A _ { T }$ </td><td>A</td><td> $A _ { T }$ </td><td> $\Delta A _ { T }$ </td></tr><tr><td>CLARE  $( \lambda = 0 . 0 0 0 1 )$ </td><td>83.88</td><td>79.73</td><td>0.00</td><td>78.32</td><td>66.88</td><td>0.00</td></tr><tr><td>No sparse mask</td><td>24.36</td><td>0.23</td><td>-79.50</td><td>8.73</td><td>0.04</td><td>-66.84</td></tr><tr><td>Random mask</td><td>83.16</td><td>79.22</td><td>-0.51</td><td>70.10</td><td>61.23</td><td>-5.65</td></tr><tr><td>Independent tuning</td><td>58.32</td><td>32.33</td><td>-47.40</td><td>23.95</td><td>11.03</td><td>-55.85</td></tr><tr><td>Stage-1 L2 regularization</td><td>83.96</td><td>79.59</td><td>-0.20</td><td>76.17</td><td>65.36</td><td>-1.52</td></tr><tr><td>Stage-1  $\lambda = 0$ </td><td>83.05</td><td>78.85</td><td>-0.88</td><td>75.84</td><td>64.97</td><td>-1.91</td></tr><tr><td>Stage-1  $\lambda = 0 . 0 0 0 0 1$ </td><td>84.08</td><td>79.92</td><td>+0.19</td><td>77.34</td><td>65.69</td><td>-1.19</td></tr><tr><td> $\operatorname { S t a g e - 1 } \lambda = 0 . 0 0 1$ </td><td>84.29</td><td>79.61</td><td>-0.12</td><td>76.92</td><td>66.16</td><td>-0.58</td></tr></table>

Table 6. Final accuracy $A _ { T }$ (mean±std over 3 random seeds).

<table><tr><td>Method</td><td>ImageNet-R 10 Task</td><td>ImageNet-A 50 Task</td><td>Omnibenchmark-1k 100 Task</td></tr><tr><td>CLARE</td><td> $7 9 . 4 8 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $5 7 . 4 9 { \scriptstyle \pm 0 . 6 2 }$ </td><td>67.10±0.46</td></tr><tr><td>SD-LoRA</td><td> $7 7 . 5 0 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $4 1 . 3 3 { \pm } 1 . 9 1 $ </td><td> $1 3 . 2 7 { \scriptstyle \pm 1 2 . 8 3 }$ </td></tr></table>

Table 7. Impact of adapter bottleneck dimension d on ImageNet-R with 50 tasks (Inc4). Relative width is measured with respect to the default $d = 6 4$

<table><tr><td>Variant</td><td>Relative width</td><td> $A _ { T }$ </td><td> $\Delta A _ { T }$ </td></tr><tr><td> $d = 1 6$ </td><td>×0.25</td><td>76.41</td><td>-0.11</td></tr><tr><td> $d = 3 2$ </td><td>×0.50</td><td>76.38</td><td>-0.14</td></tr><tr><td> $d = 6 4$ </td><td>×1.00</td><td>76.52</td><td>0.00</td></tr><tr><td> $d = 1 2 8$ </td><td>×2.00</td><td>76.65</td><td>+0.13</td></tr></table>

## 6. Conclusion

In this paper, we have addressed the challenge of scaling class-incremental continual learning to long task sequences by proposing CLARE, a sparsity-driven adapterbased framework. CLARE is built on a simple yet effective principle, where each task updates only a sparse subset of parameters, selected from parameters not already assigned to earlier tasks. The sparse updates are accumulated as it learns new tasks. This design eliminates the need for taskspecific routing or stored task-specific parameters at inference, making it particularly suitable for long task sequences where storing per-task modules becomes impractical. Extensive experiments on both long-sequence and standard benchmarks demonstrate that CLARE consistently outperforms representative continual learning methods.

Several limitations point to directions for future work. For instance, the geometric capacity schedule inevitably saturates the adapter as the number of tasks increases, limiting the maximum sequence length that the current method can support without expanding the adapter capacity. Developing dynamic capacity allocation strategies that adjust the sparsity budget based on task difficulty or inter-task similarity could extend the method to substantially longer sequences. Additionally, while CLARE is evaluated on vision tasks with a ViT backbone, extending the capacity-aware sparse allocation principle to other modalities, architecture, and domains remains an open question.

## References

[1] Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a network of experts. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3366–3375, 2017. 2, 3

[2] Chaoqi Chen, Jiongcheng Li, Xiaoguang Han, Xiaoqing Liu, and Yizhou Yu. Compound domain generalization via metaknowledge encoding. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7109–7119. IEEE, 2022. 3

[3] Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, and Ping Luo. Adaptformer: Adapting vision transformers for scalable visual recogni tion. Advances in Neural Information Processing Systems, 35:16664–16678, 2022. 3

[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl vain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 1, 3

[5] Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for continual learning. In Inter

national conference on artificial intelligence and statistics, pages 3762–3773. PMLR, 2020. 1

[6] Yunxiang Fu, Chaoqi Chen, Yu Qiao, and Yizhou Yu. Dreamda: Generative data augmentation with diffusion models. arXiv preprint arXiv:2403.12803, 2024. 3

[7] Yunxiang Fu, Meng Lou, and Yizhou Yu. Segman: Omniscale context modeling with state space models and local attention for semantic segmentation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19077–19087. IEEE, 2025. 3

[8] Zijian Gao, Wangwang Jia, Xingxing Zhang, Dulan Zhou, Kele Xu, Feng Dawei, Yong Dou, Xinjun Mao, and Huaimin Wang. Knowledge memorization and rumination for pretrained model-based class-incremental learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 20523–20533, 2025. 1, 6

[9] Dipam Goswami, Yuyang Liu, Bartłomiej Twardowski, and Joost Van De Weijer. Fecam: Exploiting the heterogeneity of class distributions in exemplar-free continual learning. Advances in Neural Information Processing Systems, 36:6582– 6595, 2023. 6

[10] Lingfeng He, De Cheng, Huaijie Wang, Xi Yang, Nannan Wang, and Xinbo Gao. Task-driven subspace decomposition for knowledge sharing and isolation in lora-based continual learning. arXiv preprint arXiv:2603.00191, 2026. 3

[11] Xiang He, Sibei Yang, Guanbin Li, Haofeng Li, Huiyou Chang, and Yizhou Yu. Non-local context encoder: Robust biomedical image segmentation against adversarial attacks. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 8417–8424, 2019. 3

[12] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 8340–8349, 2021. 6

[13] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15262–15271, 2021. 6

[14] Prakhar Kaushik, Ankit Vaidya, Shravan Chaudhari, Rama Chellappa, and Alan Yuille. Shared lora subspaces for almost strict continual learning. arXiv preprint arXiv:2602.06043, 2026. 3

[15] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017. 1, 2

[16] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.(2009), 2009. 6

[17] Jiayuan Li, Zhen Wang, Nan Xu, and Zhuhong You. Semantic segmentation with scale alignment and contextual information fusion for multimodal remote sensing images. Information Fusion, page 103671, 2025. 3

[18] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelli gence, 40(12):2935–2947, 2017. 1, 2

[19] Yan-Shuo Liang and Wu-Jun Li. Inflora: Interference-free low-rank adaptation for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23638–23647, 2024. 1, 2, 3, 6

[20] Yuyang Liu, Lai-Man Po, Farrell Hung, Zhuohan Wang, Haoxuan Wu, Zeyu Jiang, Kun Li, Xuyuan Xu, and Kwok Wai Cheung. Csbora: A continual learning method for large language models with true orthogonality and reduced forget ting. Pattern Recognition, 179:113782, 2026. 3

[21] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neu ral information processing systems, 30, 2017. 1, 2

[22] Meng Lou and Yizhou Yu. Overlock: An overview-firstlook-closely-next convnet with context-mixing dynamic kernels. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 128–138. IEEE, 2025. 3

[23] Meng Lou, Yunxiang Fu, and Yizhou Yu. Sparx: A sparse cross-layer connection mechanism for hierarchical vision mamba and transformer networks. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 19104– 19114, 2025.

[24] Meng Lou, Shu Zhang, Hong-Yu Zhou, Sibei Yang, Chuan Wu, and Yizhou Yu. Transxnet: learning both global and local dynamics with a dual dynamic token mixer for visual recognition. IEEE Transactions on Neural Networks and Learning Systems, 36(6):11534–11547, 2025. 3

[25] Meng Lou, Yunxiang Fu, and Yizhou Yu. Scaling continual learning to 300+ tasks with bi-level routing mixture-ofexperts. In Forty-third International Conference on Machine Learning, 2026. 1, 2, 3, 6

[26] Meng Lou, Hanzhong Guo, Linwei Chen, and Yizhou Yu. Overcoming catastrophic forgetting in visual continual learning with reinforcement fine-tuning. arXiv preprint arXiv:2605.09640, 2026. 3

[27] Christos Louizos, Max Welling, and Diederik P Kingma. Learning sparse neural networks through l 0 regularization. International Conference on Learning Representations, 2018. 2

[28] Rongrong Ma, Jianyu Miao, Lingfeng Niu, and Peng Zhang. Transformed l 1 regularization for learning sparse deep neural networks. Neural Networks, 119:286–298, 2019. 2

[29] Arun Mallya and Svetlana Lazebnik. Packnet: Adding mul tiple tasks to a single network by iterative pruning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 7765–7773, 2018. 2, 3

[30] Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggy back: Adapting a single network to multiple tasks by learn ing to mask weights. In Proceedings of the European conference on computer vision (ECCV), pages 67–82, 2018. 3

[31] Daniel Marczak, Bartłomiej Twardowski, Tomasz Trzcinski,´ and Sebastian Cygert. Magmax: Leveraging model merging for seamless continual learning. In European Conference on Computer Vision, pages 379–395. Springer, 2024. 1, 3

[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3

[33] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings ofthe IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017. 2

[34] Cheng Shi, Yizhou Yu, and Sibei Yang. Vision transformers need more than registers. In 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26328–26337. IEEE, 2026. 3

[35] James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, and Zsolt Kira. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11909–11919, 2023. 1, 3, 6

[36] Fengqiang Wan and Yang Yang. Probabilistic group mask guided discrete optimization for incremental learning. In Forty-second International Conference on Machine Learning, 2025. 3

[37] Zhiyi Wan, Wanrou Du, Liang Li, Miao Pan, and Xiaoqi Qin. Budget-adaptive adapter tuning in orthogonal subspaces for continual learning in llms. arXiv e-prints, pages arXiv–2505, 2025. 3

[38] Huiyi Wang, Haodong Lu, Lina Yao, and Dong Gong. Selfexpansion of pre-trained models with mixture of adapters for continual learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10087–10098, 2025. 3, 6

[39] Zifeng Wang, Zheng Zhan, Yifan Gong, Geng Yuan, Wei Niu, Tong Jian, Bin Ren, Stratis Ioannidis, Yanzhi Wang, and Jennifer Dy. Sparcl: Sparse continual learning on the edge. Advances in Neural Information Processing Systems, 35:20366–20380, 2022. 3

[40] Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, et al. Dualprompt: Complementary prompting for rehearsal-free continual learning. In European conference on computer vision, pages 631–648. Springer, 2022. 3, 6

[41] Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 139–149, 2022. 1, 3, 6

[42] Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Learning structured sparsity in deep neural networks. Advances in neural information processing systems, 29, 2016. 2

[43] Yichen Wu, Hongming Piao, Long-Kai Huang, Renzhen Wang, Wanhua Li, Hanspeter Pfister, Deyu Meng, Kede Ma,

and Ying Wei. Sd-lora: Scalable decoupled low-rank adap tation for class incremental learning. In International Conference on Learning Representations, 2025. 1, 2, 3, 6

[44] Yifeng Xiong and Xiaohui Xie. Oplora: Orthogonal projec tion lora prevents catastrophic forgetting during parameterefficient fine-tuning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 34088–34096, 2026. 3

[45] Murat Onur Yildirim, Elif Ceren Gok, Ghada Sokar, Decebal Constantin Mocanu, and Joaquin Vanschoren. Continual learning with dynamic sparse training: Exploring algorithms for effective model updates. In Conference on parsimony and learning, pages 94–107. PMLR, 2024. 3

[46] Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Ping Hu, Dong Wang, Huchuan Lu, and You He. Boosting continual learning of vision-language models via mixture-of-experts adapters. In Proceedings of the IEEE/CVF Conference on Computer Vi sion and Pattern Recognition, pages 23219–23230, 2024. 1

[47] Gengwei Zhang, Liyuan Wang, Guoliang Kang, Ling Chen, and Yunchao Wei. Slca: Slow learner with classifier align ment for continual learning on a pre-trained model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19148–19158, 2023. 6

[48] Da-Wei Zhou, Hai-Long Sun, Han-Jia Ye, and De-Chuan Zhan. Expandable subspace ensemble for pre-trained modelbased class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23554–23564, 2024. 1, 3, 6

[49] Da-Wei Zhou, Zi-Wen Cai, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. Revisiting class-incremental learning with pre-trained models: Generalizability and adaptivity are all you need. International Journal of Computer Vision, 133(3): 1012–1032, 2025. 6