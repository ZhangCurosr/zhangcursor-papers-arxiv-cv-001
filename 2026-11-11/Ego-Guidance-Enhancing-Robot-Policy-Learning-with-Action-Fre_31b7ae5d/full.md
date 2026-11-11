# Ego Guidance: Enhancing Robot Policy Learning with Action-Free Human Egocentric Videos

Lin Long<sup>∗1,5</sup> Tianren Ma<sup>∗2,5</sup> Zhong Chu<sup>∗3,5</sup> Di Wu<sup>∗4,5</sup> Maosong Cao<sup>5</sup> Qixiang Ye<sup>2</sup> Junbo Zhao<sup>1†</sup> Shanghang Zhang<sup>4‡</sup>

<sup>1</sup>Zhejiang University <sup>2</sup>University of Chinese Academy of Sciences <sup>3</sup>Nanyang Technology University <sup>4</sup>Peking University <sup>5</sup>5Ant Group <sup>∗</sup>Equal contribution <sup>†</sup>Project lead <sup>‡</sup>Corresponding author

 <sub>https://llong-cs.github.io/EgoGuidance</sub>

## Abstract

In this paper, we investigate how human egocentric videos can benefit robot policy learning without action supervision. Building on a pretrained VLM, we propose Shared Prediction on Action and Representation (SPAR), an MoT architecture in which future visual state prediction can provide independent and complementary learning signals alongside action prediction. We find that representations learned from egocentric videos improve downstream performance on out-of-distribution robot tasks, with further gains as video data scales. We also study the effects of objective choices and data curation, providing insights into how to leverage action-free videos for scalable robot learning.

## 1 Introduction

Embodied AI models are still in their infancy, eagerly absorbing every valuable source of world knowledge. Among these sources, human egocentric videos are particularly promising. Their largescale availability, ease of acquisition, and close connection to the open-world environment make them an important potential source of data for embodied intelligence [1–4]. However, precise action annotation is expensive to acquire, making most of the resources available only in video form (which we call action-free). In this work, we investigate whether such action-free data can provide benefits to robot policy learning without action restoration.

Guided by first-principled considerations, rather than adopting a World Action Model (WAM) architecture [5–7] that performs diffusion in a generative latent space, we start from a simpler and well-optimized Vision Language Model (VLM) backbone, expand it into a Mixture-of-Transformers (MoT), and jointly optimize the model with action and visual state prediction. Building on this Shared Prediction on Action and Representation (SPAR), we investigate how different encoders for representing the predicted state affect performance.

We further conduct preliminary studies on several important factors, including data cleaning, diversity trade-off, and the scalability potential of pretraining with action-free egocentric videos.

## 2 Method

We adopt an MoT architecture [10, 11] that leverages a pretrained VLM [12] to jointly model robot actions and future world states. As shown in Fig. 1, two independently parameterized transformer stacks, i.e., an action expert and a generation expert are extended on the shared VLM backbone.

Action Expert. Given the current observation and task instruction at time t and horizon length H, the VLM’s language model produces per-layer hidden states $\mathscr { H } = \{ \mathbf { h } ^ { ( l ) } \} _ { l = 1 } ^ { L }$ . An action expert $f _ { \psi }$ cross-attends to these states and predicts an action chunk $\mathbf { A } _ { t } = a _ { t : t + H } \in \mathbb { R } ^ { H \times d _ { a } }$ with flow matching [13]: For a clean sample $\mathbf { x } ,$ Gaussian noise $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , and flow time $s \in [ 0 , 1 ]$ , we define $\mathbf { x } ^ { s } = ( 1 - s ) { \boldsymbol { \epsilon } } + s \mathbf { x }$ and predict the velocity x − ϵ. Applied to action chunks, this gives a common vision-language-action (VLA) [8, 14] loss:

![](images/043387c03108c8933bcddec3f77573be2197785b6db813d366d3924835d34810.jpg)  
Figure 1: Architecture Overview of SPAR. Starting from a common VLA [8, 9], SPAR extends it with a generation expert that mirrors the action expert design, and predicts the representation of the future visual state simultaneously. Robot data supervises both experts, while the egocentric videos supervise only the generation expert. The VLM backbone can therefore be optimized with heterogeneous data composition. The generation expert is removed at inference.

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a c t i o n } } = \mathbb { E } _ { s , \epsilon } \frac { 1 } { H d _ { a } } \big \| f _ { \psi } ( \mathbf { A } _ { t } ^ { s } , s ; \mathcal { H } ) - ( \mathbf { A } _ { t } - \epsilon ) \big \| _ { 2 } ^ { 2 } . } \end{array}\tag{1}
$$

Generation Expert. Action labels provide a relatively low-dimensional, embodiment-specific training signal. To provide high-capacity and generalizable supervision for the shared VLM, we add a generation expert $g _ { \omega }$ that predicts a future visual representation from the same hidden states H through layer-wise cross-attention. SPAR’s two experts do not attend to each other, making the action and visual state prediction two independent proxy tasks for backbone optimization, i.e., the generation loss can improve the action pathway only by reshaping the shared hidden states.

Given the step description, the generation expert predicts the future visual state $\mathbf { Z } _ { t } = \phi \big ( \mathbf { o } _ { t + \Delta } \big ) \in$ $\mathbb { R } ^ { N \times d }$ , that is, the features of the primary camera’s future observation $\mathbf { o } _ { t + \Delta }$ extracted by visual encoder ϕ. Here, we align $\mathbf { o } _ { t + \Delta }$ with $\boldsymbol { a } _ { t + H }$ to ensure that they are sampled in the same timestep.

The prediction of $\mathbf { Z } _ { t }$ can be supervised by either flow matching or regression: For the flow-matching version, we apply a similar velocity target used for actions to the future visual features, with noising schedules modified following RAE [15]. The noised target $\mathbf { Z } _ { t } ^ { s }$ is linearly projected to the expert’s width and given learned positional embeddings, the blocks are conditioned on s through AdaLN, and an output projection predicts the velocity, optimized by $\mathcal { L } _ { \mathrm { g e n } } ^ { \mathrm { f l o w } }$ . For the regression version, we set the timestep to constant $s = 0$ , and use a three-layer MLP to map the expert’s output to the prediction. The expert is therefore optimized by $\mathcal { L } _ { \mathrm { g e n } } ^ { \mathrm { r e g } ^ { \bullet } }$ to directly predict the standardized target $\bar { \mathbf Z } _ { t } = ( \mathbf Z _ { t } - \mu _ { \phi } ) / \sigma _ { \phi }$ , where $\mu _ { \phi }$ and $\sigma _ { \phi }$ are scalar statistics of encoder $\phi \left( { \mathrm { A p p . ~ A } } \right)$

The losses are written as follows, and can both be denoted by $\mathcal { L } _ { \mathrm { g e n } }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g e n } } ^ { \mathrm { f l o w } } = \mathbb { E } _ { s , \epsilon , t } \frac { 1 } { N d } \big \| g _ { \omega } ( \mathbf { Z } _ { t } ^ { s } , s ; \mathcal { H } ) - ( \mathbf { Z } _ { t } - \epsilon ) \big \| _ { 2 } ^ { 2 } , \quad \mathcal { L } _ { \mathrm { g e n } } ^ { \mathrm { r e g } } = \mathbb { E } _ { t } \frac { 1 } { N d } \big \| g _ { \omega } ( \mathcal { H } ) - \bar { \mathbf { Z } } _ { t } \big \| _ { 2 } ^ { 2 } . } \end{array}\tag{2}
$$

Co-training with Action-free Videos. Each data source supervises the losses it can provide:

$$
{ \mathcal { L } } _ { \mathrm { r o b o t } } = { \mathcal { L } } _ { \mathrm { a c t i o n } } + \lambda { \mathcal { L } } _ { \mathrm { g e n } } , \qquad { \mathcal { L } } _ { \mathrm { e g o } } = \lambda { \mathcal { L } } _ { \mathrm { g e n } } ,\tag{3}
$$

with $\lambda = 1$ . Each training step combines a robot and an egocentric mini-batch. Both losses backpropagate through the cross-attention into the LLM, while the action expert is updated only by robot data. The effect of egocentric videos on the policy is therefore entirely mediated by the backbone representation they help shape.

Inference. At inference, the generation expert is removed following Fast-WAM [16], and actions are obtained by integrating the action expert’s velocity field from Gaussian noise with a few Euler steps. The deployed policy thus has the same architecture and cost as a standard VLA. All reported numbers are averaged across three runs with different random seeds.

## 3 Experiment

We describe the datasets and training settings in this section. See Appendix A for more details.

## 3.1 Datasets

Robot Data. We curate the dataset from DROID [17] by retaining successful episodes and filtering out episodes with no motion or without valid instruction or action annotations. This yields approximately 38,000 episodes totaling 230 hours of robot data (66% retention), which we co-train with egocentric video data. For downstream fine-tuning, we use LIBERO [18] demonstrations following the standard protocol [19] for simulated evaluation. For the final real robot scenario, we extend the robot data with more datasets [18, 20–24], totaling about 8, 000 hours, and internal demonstrations collected via tele-operation for downstream fine-tuning.

Egocentric Data. We use Ego4D [1] for our preliminary experiments. As it does not provide usable action labels, we treat it as action-free egocentric video data. Its raw videos comprise 3,976 hours, and we describe our curation in Section 4. To increase scale, we also use Open-AoE [3], which provides approximately 1M annotated video clip–caption pairs on 8,000 tasks, spanning 2,000 hours. For Open-AoE, we use the released clips and language descriptions following the dataset’s original curation, without applying the Ego4D-specific gates.

## 3.2 Training

We compared two training strategies in the primary experiments. In single-stage training, we jointly train on the robot demonstrations that are aligned with the downstream evaluation tasks and egocentric videos. In two-stage training, we first co-train on a broader collection of robot demonstrations and egocentric videos, then fine-tune on downstream-aligned demonstrations alone. The single-stage strategy consistently yields poorer generalization, with little variation across experimental settings. This behavior suggests overfitting to the limited robot data and provides little insight into the effect of egocentric pretraining. We therefore use the two-stage strategy for SPAR as default.

Pretraining. We co-train on robot and egocentric data. Both robot and egocentric data supervise the prediction on future visual-state representation. Robot data also supervise the prediction of the subsequent action chunk as common VLA models.

Downstream Fine-tuning. We then fine-tune on robot demonstrations with ${ \mathcal { L } } _ { \mathrm { r o b o t } }$ . Because pretraining is intended to shape the shared VLM representation, the action expert can either be reinitialized for the downstream task or trained further from its pretrained weights. In our experiments, reinitializing the action expert yields a modest improvement.

## 3.3 Preliminary Findings

Representational target helps generalization. Figure 2(a) compares the per-dimension success rates on Libero-Plus across states encoded by six different encoders [12, 25–29]. The target representation substantially affects performance under distribution shift. SigLIP2 achieves the highest reported average success rate (76.2%), followed by Qwen-ViT (75.6%) and DINOv2 (74.9%). All five representation targets outperform the reconstruction-oriented Wan-VAE latent (60.0%) in this comparison, consistent with the view that predictive supervision in a learned visual feature space is more useful for policy learning than reconstructive latents. No target dominates every shift: DINOv2 obtains the highest diffusion scores for sensor noise and object layout, whereas Qwen-ViT leads under lighting and background-texture changes. The choice of visual target affects both overall performance and the types of perturbations to which the policy generalizes.

Regression supervision also works. Interestingly, figure 2(b) shows that direct regression of future visual features also provides useful supervision. Its per-dimension results are broadly comparable to diffusion for several targets, particularly SigLIP2 and DINOv2, suggesting that the benefits of visual prediction in our model do not depend on a diffusion objective.

![](images/dca93a4012ca3b859e9e372c2dda71366537e1407b5e87b43240b42859d0be18.jpg)  
Figure 2: Success rates across seven distribution shifts. (a) Diffusion and (b) regression prediction of one future state frame with six visual targets; the two panels share a target legend. (c) Diffusion prediction of one versus four future frames with SigLIP2 in a separate frame-count experiment.

State prediction can be single framed. During training, the timestep t is already randomly sampled and reflects the request for dynamic modeling. We then investigate whether predicting a single frame works, as it significantly reduce the training cost. For the high-dimensional SigLIP2 target (256 tokens of width 1,152; App. A), Fig. 2(c) compares predicting one future frame with predicting four using the same diffusion objective. Their reported average success rates are 74.32% and 74.62%, respectively, a difference of only 0.30 percentage points. Together with the target comparison, we set the default design as predicting a single future frame in SigLIP2’s visual representation space.

## 4 What Matters in Action-Free Egocentric Videos?

Data Processing. We divide Ego4D videos into uniform 5-second clips and apply two cumulative quality gates based on EgoSmith [30]. Gate 1 uses corner and optical-flow checks to retain clips with limited camera motion and reliable flow estimates; Gate 2 retains clips with at least one sufficiently visible hand in a prescribed image region. The first gate retains approximately 1,893 hours (48% of the raw duration), and both gates together retain 975 hours (approximately 25%; Fig. 3). We tested further fine-grained EgoSmith-based quality filters but observed no additional benefit, so they are not part of the final pipeline. We then use a locally deployed Qwen3.6-35B-A3B model to generate a short English step description from 15 frames sampled at 10-frame intervals per clip. This description serves as the language instruction for future-state prediction. Appendix B gives the exact gate criteria and captioning instructions.

The curation is consequential for downstream learning (Fig. 3). Training with raw Ego4D clips underperforms the robot-only baseline, whereas the clips retained after both gates surpass it at 16k and 20k fine-tuning steps. Gate 1 alone provides at most a small improvement over robot-only training; requiring visible hands in addition to stable, reliable optical flow gives the strongest result. Thus, selecting clips with visible hand activity makes action-free video a more useful source of supervision than simply adding unfiltered video.

Data Scaling. We then conduct a data scaling experiment with ego videos from 0 to 1,500 hours. We use the off-the-shelf Open-AoE, and its step description as the dataset source. As shown in Fig. 4a, In-domain LIBERO success changes little across these settings, as expected because the added human videos do not depict the downstream tasks. By contrast, LIBERO-Plus success rises steadily as video hours increase. Since these videos provide no action labels, the OOD trend is consistent with their visual prediction objective improving the shared VLM backbone and, in turn, the generalization of the downstream policy.

![](images/0ec57a23fc3d9b26868752b348e31f2daa640db6cd7390f11297690dced33f81.jpg)

![](images/d52f51f74afc63cc375b947539b4e58a48d4c73359b2a8dfa686b3250e79cd10.jpg)  
Figure 3: Ego4D curation and corresponding OOD task success rates. Left: hours retained after the motion-quality and visible-hand gates. Right: robot-only training compared with co-training on raw Ego4D, Gate-1 clips, or clips passing both gates, across fine-tuning steps.

![](images/0897d5712fce6c5a7a3d6024e1df0169fed68de56a104565a270a716690a5964.jpg)  
(a) Video scale.

![](images/ce5deb4bde5bd477faa07c7460dbbc3f9a5bb5a665732b95872e4820ed42663a.jpg)  
(b) Diversity at a fixed budget.  
Figure 4: Egocentric video data ablations. (a) In-domain and OOD success with 0, 150, 450, and 1,500 hours of Open-AoE video co-training. (b) LIBERO-Plus success as scene, object, or interaction diversity varies within a fixed budget of 200K clip–description pairs.

Data Diversity. We also study whether we can trade off some valid video duration for diversity. However, as shown in Fig. 4b, at a fixed egocentric video budget (200K clip-description pairs sampled from Open-AoE), increasing diversity along scene, object, or interaction dimensions does not yield a consistent gain in downstream success. Neither scene nor interaction diversity improves monotonically, while the highest object-diversity setting performs worse in this preliminary comparison. Together with the continuing gains from adding more hours, this suggests that egocentric video has not saturated the model over the tested range: expanding the amount of data across scenes, objects, and interactions appears more promising than maximizing any one diversity measure within a fixed video budget.

## 5 Real-World Robot Evaluation

We evaluate our model on a physical bimanual platform and verify: (1) Can egocentric pretraining enable our model to perform comparably to a strong VLA baseline on real-world manipulation tasks? (2) Does it retain this performance under visual perturbations unseen during fine-tuning?

Hardware. As shown in Fig. 5, we use an AgileX Cobot Magic V2.0 with two 6-DoF arms and parallel-jaw grippers. We replace the stock wrist cameras with RealSense D405 cameras and add a top-mounted RealSense D455f. The policy takes images from all three cameras as input. The action space has 14 dimensions: six joint positions and one gripper command per arm.

![](images/9de31c409901fc377506cc476c4e70160148275a9d0ea5815adbe163f43e8868.jpg)

![](images/aefbe2dc2a327e2265e321c2563e37c286d7102a5d224ad54572aed5ef3a9812.jpg)  
Figure 5: Real-world evaluation setup. Left: The hardware setup; objects from several tasks are placed together for illustration only. Right: Five sequential snapshots per task.

Table 1: Real-world Results. # Demos denotes the number of tele-operation demonstration collected for each task. We report the success rate out of 10 trials for each task.
<table><tr><td></td><td>Cucumber Transfer</td><td>Flower Handover</td><td>Toaster Loading</td><td>Cup Hanging</td><td>Bottle Uncapping</td><td>Total</td></tr><tr><td># Demos</td><td>220</td><td>160</td><td>150</td><td>139</td><td>57</td><td>一</td></tr><tr><td>π0.5</td><td>9/10</td><td>9/10</td><td>2/10</td><td>1/10</td><td>1/10</td><td>22/50</td></tr><tr><td>Ours</td><td>9/10</td><td>8/10</td><td>2/10</td><td>0/10</td><td>1/10</td><td>20/50</td></tr></table>

Tasks. We design five tasks that vary in precision, inter-arm coordination, and horizon. Cucumber Transfer: The right arm moves a cucumber from the right plate to the left plate. This task serves as a basic pick-and-place test. Flower Handover: The left arm picks up a flower and hands it to the right arm in midair. The right arm then inserts the flower into a vase. This task tests coordinated positioning and grasp timing between the arms. Cup Hanging: The right arm grasps a cup by its handle, turns it upside down, and hangs it on a rack. The small clearance between the gripper and the handle opening requires precise positioning. Bottle Uncapping: The right arm lifts and holds a bottle. The left arm unscrews the cap and places it on the upper-left area of the table, and the right arm returns the bottle to the table. This task tests rotational manipulation and bimanual coordination. Toaster Loading: The left arm inserts two slices of bread into the toaster one at a time. Each insertion requires precise alignment, and the repeated pick-and-insert sequence extends the horizon.

Training and Evaluation. We collect demonstrations by tele-operation (Table 1). Bottle Uncapping has only 57 demonstrations and serves as a few-shot setting. We compare our model against $\pi _ { 0 . 5 }$ [9], a VLA pretrained on large-scale robot data and finetune it using the official pi05\_base weight. For each method and task, we fine-tune a separate checkpoint for 20k steps. Both models predict 50-step action chunks and execute 30 steps per chunk with asynchronous inference at 30 Hz.

We evaluate each checkpoint for 10 trials per task. Initial object positions are randomized within a 4 cm × 4 cm region. A trial is successful only if the robot completes the entire task without failure, as judged by a human evaluator. We report the per-task success rate and its average over the five tasks.

Results. As shown in Table 1, our model achieves an average success rate of 40%, compared with 44% for $\pi _ { 0 . 5 }$ . The two methods tie on three tasks and differ by one trial on the other two. Notably, our model performs comparably to $\pi _ { 0 . 5 }$ despite using a smaller robot pretraining corpus and training its action expert from scratch. This suggests that the visual-state prediction on human videos provides transferable priors that partly compensate for both. Both methods achieve high success on Cucumber Transfer and Flower Handover but rarely complete Cup Hanging, Bottle Uncapping, or Toaster Loading. Precision grasping, bimanual rotational manipulation, and long-horizon execution remain challenging for both methods. Most failures of our model occur during grasping, which is likely related to its smaller robot pretraining corpus.

![](images/47e2695fba3ee068914af68c7d9212f3fd793bd8b3517639199446eb7a47f33f.jpg)  
Figure 6: Real-world generalization evaluation setting under visual perturbations.

Table 2: Success under visual perturbations out of 10 trials for each task. Default: unperturbed setting; Illum.: illumination changes; Backgr.: background variations; Distr.: static distractors.
<table><tr><td></td><td colspan="4">Cucumber Transfer</td><td colspan="4">Flower Handover</td></tr><tr><td>Method</td><td>Default</td><td>Illum.</td><td>Backgr.</td><td>Distr.</td><td>Default</td><td>Illum.</td><td>Backgr.</td><td>Distr.</td></tr><tr><td>π0.5</td><td>9</td><td>7</td><td>7</td><td>4</td><td>9</td><td>5</td><td>2</td><td>3</td></tr><tr><td>Ours</td><td>9</td><td>9</td><td>8</td><td>5</td><td>8</td><td>2</td><td>2</td><td>0</td></tr></table>

Visual perturbations. We apply three perturbations to Cucumber Transfer and Flower Handover (Fig. 6). For illumination changes, we dim the white light and turn on a colored light that cycles at 1.5 Hz. For background variations, we cover the tabletop with a checkered tablecloth. For static distractors, we place a similar object next to the target: a carrot beside the cucumber and additional flowers beside the flower. We evaluate each perturbation separately for 10 trials per task.

Results under Perturbations. Table 2 reports the results. Averaged over both tasks, static distractors cause the largest drop for both methods, from 90% to 35% for $\pi _ { 0 . 5 }$ and from 85% to 25% for ours. On Cucumber Transfer, both methods achieve 9/10 in the default setting. Our model outperforms $\pi _ { 0 . 5 }$ under all three perturbations. Under illumination changes, our model maintains 9/10, whereas $\pi _ { 0 . 5 }$ drops to 7/10. Under background variations and static distractors, our model achieves 8/10 and 5/10, compared with 7/10 and 4/10 for $\pi _ { 0 . 5 }$ . Consistent with the LIBERO-Plus results (Sec. 4), this suggests that visual-state prediction on diverse human videos yields representations that are less sensitive to appearance changes. On Flower Handover, our model matches $\pi _ { 0 . 5 }$ under background variations (2/10) but underperforms it under illumination changes (2/10 vs. 5/10) and static distractors (0/10 vs. 3/10). We hypothesize that $\pi _ { 0 . 5 }$ benefits here from action priors learned from large-scale bimanual robot data.

## 6 Conclusion

We investigated whether human egocentric videos without action labels can benefit robot policy learning through future visual-state prediction. During model design exploration, we find that learned visual representations are more effective prediction targets than reconstruction latents under distribution shift, and regression is a viable alternative to flow matching under the Shared Prediction on Action and Representation (SPAR) design. Data quality also matters: filtering Ego4D for stable motion and visible hands improves on both raw-video co-training and the robot-only baseline. As the amount of action-free video increases, OOD success rises consistently. On real-world manipulation tasks, our policy performs comparably to the powerful VLA baseline, with task-dependent results under visual perturbations. Together, these findings provide fresh insight for the utilization of action free egocentric video, showing its potential as a useful source of transferable visual supervision.

## References

[1] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhongcong Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Cartillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli, Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Abrham Gebreselasie, Cristina Gonzalez, James Hillis, Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, Jachym Kolar, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li, Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Modhugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will Price, Paola Ruiz Puentes, Merey Ramazanova, Leda Sari, Kiran Somasundaram, Audrey Southerland, Yusuke Sugano, Ruijie Tao, Minh Vo, Yuchen Wang, Xindi Wu, Takuma Yagi, Ziwei Zhao, Yunyi Zhu, Pablo Arbelaez, David Crandall, Dima Damen, Giovanni Maria Farinella, Christian Fuegen, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar, Hanbyul Joo, Kris Kitani, Haizhou Li, Richard Newcombe, Aude Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi, Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei Yan, and Jitendra Malik. Ego4D: Around the World in 3,000 Hours of Egocentric Video. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18973–18990. IEEE, 2022. ISBN 978-1-6654-6946-3.

[2] Jiangran Lyu, Kai Liu, Xuheng Zhang, Haoran Liao, Yusen Feng, Wenxuan Zhu, Tingrui Shen, Jiayi Chen, Jiazhao Zhang, Yifei Dong, Cui Wenbo, Senmao Qi, Shuo Wang, Yixin Zheng, Mi Yan, Xuesong Shi, Haoran Li, Dongbin Zhao, Ming-Yu Liu, Zhizheng Zhang, Li Yi, Yizhou Wang, and He Wang. LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion. In Robotics: Science and Systems XXII. Robotics: Science and Systems Foundation, 2026. ISBN 979-8-9902848-2-1.

[3] Zishuo Li, Bowen Yang, Changtao Miao, Kai Zhu, Hao Chen, Qingze Guan, Zhengxing Wu, Wanke Zhan, Yang Sun, Zhiyi Huang, Zitong Shan, Zhenchao Jin, Jiadong Hong, Taowen Wang, Yushi Feng, You Liu, Yibo Wang, Yifan Yang, Zhaowen Zhou, Man Luo, Hao Cheng, Bo Zhang, Jianshu Li, Jiansheng Cai, Guocai Yao, Jize Zhang, Chenhao Lin, Renjing Xu, Lequan Yu, Chao Shen, Chunhua Shen, and Zhe Li. Open-AoE: An Open Egocentric Manipulation Dataset and Toolchain for Embodied Learning. arXiv preprint arXiv:2607.14183, 2026.

[4] Zhihao Sun, Liu Liu, Xinjiang Wang, Haoyi Jiang, Wei Feng, Huiqiang Zhang, Xiaosong Jia, Zhizhong Su, and Zuxuan Wu. Ego4WAM: What Matters When Scaling Egocentric Human Data for Robot Learning? arXiv preprint arXiv:2609.40341, 2026. doi: 10.48550/arXiv.2609.40341.

[5] Junbang Liang, Pavel Tokmakov, Ruoshi Liu, Sruthi Sudhakar, Paarth Shah, Rares Ambrus, and Carl Vondrick. Video Generators are Robot Policies. arXiv preprint arXiv:2508.00795, 2025. doi: 10.48550/ arXiv.2508.00795.

[6] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Causal World Modeling for Robot Control. arXiv preprint arXiv:2601.21998, 2026. doi: 10.48550/arXiv.2601.21998.

[7] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi "Jim" Fan, and Joel Jang. World Action Models are Zero-shot Policies. arXiv preprint arXiv:2602.15922, 2026.

[8] StarVLA Community. StarVLA: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.

[9] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[10] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π : A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164, 2024.

[11] Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.

[12] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/ blog?id=qwen3.5.

[13] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

[14] NVIDIA, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[15] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

[16] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do World Action Models Need Test-time Future Imagination? arXiv preprint arXiv:2603.16666, 2026.

[17] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Srirama, Lawrence Chen, Kirsty Ellis, Peter Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Ma, Patrick Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Walke, Blake Wulfe, Ted Xiao, Jonathan Yang, Arefeh Yavary, Tony Zhao, Christopher Agia, Rohan Baijal, Mateo Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Foster, Jensen Gao, David Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Donovon Jackson, Charlotte Le, Yunshuang Li, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph Lim, Jitendra Malik, Roberto Martín-Martín, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. In Robotics: Science and Systems XX. Robotics: Science and Systems Foundation, 2024. ISBN 979-8-9902848-0-7. URL http://www.roboticsproceedings.org/rss20/p120.pdf.

[18] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

[19] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. Libero-plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

[20] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

[21] AgiBot World Colosseum contributors. Agibot world colosseum. https://github.com/ OpenDriveLab/AgiBot-World, 2024.

[22] InternData-A1 contributors. Interndata-a1. https://github.com/InternRobotics/InternManip, 2025.

[23] Arthur Allshire, Himanshu Gaurav Singh, Ritvik Singh, Adam Rashid, Hongsuk Choi, David McAllister, Justin Yu, Yiyuan Chen, Huang Huang, Pieter Abbeel, Xi Chen, Rocky Duan, Phillip Isola, Jitendra Malik, Fred Shentu, Guanya Shi, Philipp Wu, and Angjoo Kanazawa. Scalable behavior cloning with open data, training, and evaluation, 2026. URL https://arxiv.org/abs/2606.27375.

[24] Simple AI, Yuteng Wei, Jinming Ma, Jiawei Wang, Weitao Zhou, Yushen Zuo, Ke Rui, Minglei Li, Jinhao Zhang, Zhikang Pan, Xiang Wang, Haoran Jia, Huan Du, Zicheng Zeng, Jun Ma, Guiyu Qin, Di Zhang, and Xiaofei Li. Hifi-umi: Learning deployable manipulation policies from high-fidelity umi data alone. arXiv preprint arXiv:2607.25895, 2026.

[25] Michael Tschannen, Alexey A. Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier J. Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. CoRR, abs/2502.14786, 2025.

[26] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[27] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2024, 2024.

[28] Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[29] Zelin Fu, Bin Tan, Changjiang Sun, Shaohui Liu, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, and Nan Xue. Vision pretraining for dense spatial perception. arXiv preprint arXiv:2607.05247, 2026.

[30] Yifan Zhong, Zhang Chen, Tianrui Guan, Fanlian Zeng, Ka Nam Lui, Yuyao Ye, Tingrui Zhang, Jiayi Li, Tianjia He, Wenjie Lou, Ruilin Yan, Xinhao Ji, Guangyu Zhao, Jiayuan Zhang, Wenxi Xu, Chengdong Ma, Yuanpei Chen, and Yaodong Yang. Egosteer: An open-source full-stack system towards steerable dexterous manipulation from egocentric videos, 2026. URL https://arxiv.org/abs/2607.09701.

[31] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[32] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

[33] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR), 2024.

[34] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning (ICML), 2024.

## A Implementation Details

Backbone and inputs. We use Qwen3.5-2B [12], whose LLM has L=24 layers of width $D { = } 2 0 4 8$ Observations are processed at native resolution with a budget of 256×256 pixels per image. Each image is preceded by a description of its camera role: “primary global view from a fixed exterior camera” and “local effector view from a wrist camera” for DROID, “primary global view from a fixed front camera” and “local effector view from a wrist camera” for LIBERO, and “primary global view from a moving human head camera” for egocentric video. For egocentric clips, the automatically generated step description serves as the instruction (App. B). The generation target is always the future frame of the primary global view, i.e., the third-person camera for robots and the head camera for humans. The vision encoder, including its token-merging projector, is frozen; the LLM and both experts are trained.

Expert architecture. Both experts have L blocks of width D (32 heads of dimension 64, dropout 0.2), and block l is paired with LLM layer l. Blocks alternate between cross-attention to the paired hidden states $\mathbf { h } ^ { ( l ) }$ and self-attention among the expert tokens, each modulated by AdaLN [31] on a conditioning embedding c:

$$
\tilde { \mathbf { x } } = \mathbf { x } + \mathrm { A t t n } \big ( \mathrm { A d a L N } ( \mathbf { x } ; \mathbf { c } ) , \mathbf { m } ^ { ( l ) } \big ) , \qquad \mathbf { x } \gets \tilde { \mathbf { x } } + \mathrm { F F N } \big ( \mathrm { L N } ( \tilde { \mathbf { x } } ) \big ) ,\tag{4}
$$

where $\mathbf { m } ^ { ( l ) } = \mathbf { h } ^ { ( l ) }$ in cross-attention blocks and the normalized expert tokens otherwise. The two experts never attend to each other.

• Action expert. The input is 32 learnable tokens followed by H action tokens, each embedding one row of the noised chunk together with a sinusoidal encoding of the flow time and a learned positional embedding; a proprioceptive-state token can optionally be prepended (not used in our experiments). The blocks are conditioned on $\mathbf { c } = \operatorname { e m b } ( s )$ , and an MLP maps the outputs at the action positions to a velocity.

• Regression generation expert. The input is N learnable query tokens with positional embeddings, one per target token. The conditioning is constant, $\mathbf { c } = \mathrm { e m b } ( 0 )$ , and a three-layer MLP with an output LayerNorm maps the outputs to $\hat { \mathbf { Z } } _ { t } \in \mathbb { R } ^ { N \times d }$

• Diffusion generation expert. The noised target tokens are linearly projected to width D and given learned positional embeddings; the blocks are conditioned on $\mathbf { c } = \operatorname { e m b } ( s )$ , and an AdaLN-modulated output projection predicts the velocity in $\mathbb { R } ^ { N \times d }$

We attach the generation branch as a separate expert rather than appending learnable generation queries to the LLM sequence: the latter changes the input on which the action pathway conditions, and in preliminary experiments it consistently yielded lower success rates.

Temporal alignment. The future frame $\mathbf { o } _ { t + \Delta }$ is aligned with the end of the action chunk, $\boldsymbol { a } _ { t + H }$ , so the visual prediction horizon in physical time is $\tau = H / f _ { \mathrm { R } }$ , where $f _ { \mathrm { R } }$ is the frame rate of the robot data. It is converted to each stream’s frame rate so that all streams predict the same instant: a stream recorded at $f _ { s }$ frames per second uses an offset of $\begin{array} { r } { \Delta _ { s } = \lfloor \tau f _ { s } + \frac { 1 } { 2 } \rfloor } \end{array}$ frames. During pretraining on DROID (15 Hz), τ=8/15≈0.53 s, i.e., 8 DROID frames and 16 frames of a 30 fps egocentric video; during fine-tuning on LIBERO (20 Hz), τ=0.4 s, i.e., 8 frames.

Generation targets. Tab. 3 lists the target encoders. All are frozen and applied online to the raw future frame, so switching targets requires no data re-processing. For the regression loss, each target is standardized as $\bar { \mathbf Z } _ { t } = ( \mathbf T _ { t } - \mu _ { \phi } ) / \bar { \sigma } _ { \phi }$ with scalar statistics estimated once on future frames of the training mixture.

Flow-matching schedules. For actions, following [10], we sample $s = 1 - u / 0 . 9 9 9$ with $u \sim$ Beta(1.5, 1) and draw two (s, ϵ) pairs per example while reusing the VLM features. For the diffusion generation expert, the schedule depends on the target type. For representation targets, we follow RAE [15]: the noise level $\sigma = 1 - s$ is drawn from a logit-normal distribution [34] and shifted as $\sigma \gets \gamma \sigma / ( 1 + ( \gamma - 1 ) \sigma )$ with γ = max $( 1 , \sqrt { N d / 4 0 9 6 } )$ . For VAE latents, we follow common practice in video latent diffusion [28]: σ is drawn uniformly and shifted with $\gamma { = } 5$ , and the per-sample loss is weighted by $w ( s ) \propto \exp \big ( - 2 ( s - { \textstyle { \frac { 1 } { 2 } } } ) ^ { 2 } \big )$ , normalized to unit mean within a batch. Diffusion targets are not standardized.

Table 3: Target encoders. N and d are the number and dimension of target tokens per frame.
<table><tr><td>Target</td><td>Encoder</td><td>Pretraining objective</td><td>Res.</td><td>N</td><td> $d$ </td></tr><tr><td>Qwen-ViT</td><td>Qwen3.5-2B vision encoder [12]</td><td>vision–language (VLM)</td><td>224</td><td>49</td><td>2048</td></tr><tr><td>SigLIP2</td><td>So400m/16 [32]</td><td>language-image</td><td>256</td><td>256</td><td>1152</td></tr><tr><td>DINOv2</td><td>ViT-g/14 [33]</td><td>self-distillation</td><td>224</td><td>257</td><td>1536</td></tr><tr><td>LingBot</td><td>LingBot-Vision ViT-g/16 [29]</td><td>masked boundary modeling</td><td>512</td><td>1024</td><td>1536</td></tr><tr><td>V-JEPA2†</td><td>ViT-g/16 [26]</td><td>latent video prediction</td><td>256</td><td>256</td><td>1408</td></tr><tr><td>Wan-VAE</td><td>Wan2.2 VAE [28]</td><td>pixel reconstruction</td><td>256</td><td>256</td><td>48</td></tr></table>

$^ \dag \mathrm { A }$ video encoder; the single future frame is repeated to fill one temporal tubelet.

Co-training. The per-GPU batch is split between the robot and egocentric streams by a sampling ratio (1:1 by default). The two mini-batches are forwarded separately, each computing only its bound losses, and their gradients are accumulated into a single optimizer step (Alg. 1). Because each loss is averaged within its own mini-batch, the ratio changes how many samples of each source are seen per step but not the relative weight of the two objectives.

Algorithm 1 One co-training step   
Require: robot mini-batch $\begin{array} { r } { B _ { \mathrm { R } } ; } \end{array}$ egocentric mini-batch $\boldsymbol { B _ { \mathrm { E } } }$   
1: for $B \in \{ B _ { \mathrm { R } } , B _ { \mathrm { E } } \}$ do   
2: $\mathcal { H }  \mathrm { L L M } _ { \theta } ($ images, role prompts, ℓ) ▷ per-layer hidden states   
3: $\mathbf { Z } _ { t } \gets \phi ( \mathbf { o } _ { t + \Delta } )$ ▷ frozen encoder, no gradient   
4: $\mathcal { L } _ { B } \gets \lambda \mathcal { L } _ { \mathrm { g e n } } ( g _ { \omega } ; \mathcal { H } , \mathbf { Z } _ { t } )$   
5: if $\boldsymbol { B } = \boldsymbol { B } _ { \mathrm { R } }$ then   
6: $\mathcal { L } _ { B } \gets \mathcal { L } _ { B } + \mathcal { L } _ { \mathrm { a c t i o n } } ( f _ { \psi } ; \mathcal { H } , \mathbf { A } _ { t } )$   
7: end if   
8: accumulate $\nabla _ { \boldsymbol { \theta } , \psi , \omega } \mathcal { L } _ { B }$   
9: end for   
10: update $( \theta , \psi , \omega )$ with AdamW

Optimization. Tab. 4 lists the training hyperparameters. We train with bfloat16 mixed precision, DeepSpeed ZeRO-2, and gradient checkpointing.

Table 4: Training hyperparameters.  
Optimizer AdamW, $\beta = ( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 } ,$ , weight decay $1 0 ^ { - 8 }$   
Learning rate LLM $2 . 8 3 \times 1 0 ^ { - 5 } ;$ experts $2 . 8 3 \times 1 0 ^ { - 4 } ;$ other $\mathrm { 1 . 4 1 \times 1 0 ^ { - 4 } }$   
Schedule 500 warmup steps, cosine decay to $2 . 8 3 \times 1 0 ^ { - 6 }$   
Pretraining steps 20,000   
Downstream fine-tuning steps 20,000   
Pretraining global batch size 1,024 (1:1 robot/egocentric)   
Gradient clipping 1.0   
Action chunk $H { = } 8 , d _ { a } { = } 7$   
Loss weights $\lambda _ { \mathrm { a c t i o n } } = \lambda _ { \mathrm { g e n } } = 1$   
Future frames 1   
Inference 4 Euler steps

Inference. Starting from Gaussian noise, the action chunk is obtained with M=4 Euler steps, $\begin{array} { r } { \mathbf { A } _ { t } ^ { s _ { k + 1 } } = \mathbf { A } _ { t } ^ { s _ { k } } + \frac { 1 } { M } \vartriangle \mathbf { \breve { f } } _ { \psi } ( \mathbf { A } _ { t } ^ { s _ { k } } , s _ { k } ; \mathcal { H } ) } \end{array}$ with $s _ { k } = k / M$ . The generation expert is not executed.

## B Ego4D Data Curation

Clip construction. We partition each source video into uniform, non-overlapping 5-second clips. The two quality gates are applied in sequence to sampled points within each clip, following the EgoSmith procedure [30].

Gate 1: motion quality. Corner detection and optical-flow fitting provide a point-wise motion check. Let $N _ { F }$ (flow\_evaluable\_count) be the number of flow-evaluable sampled points in a clip, excluding the first sampled point of a source video when no preceding frame is available. Let $n _ { F }$ (flow\_pass\_count) count those points for which the optical-flow displacement check passes (motion\_ok=True), and let $L _ { F }$ (flow\_max\_invalid\_run) be the longest consecutive run of failed checks (motion\_ok=False) among flow-evaluable points. Define $r _ { F } = n _ { F } / N _ { F }$ (flow\_pass\_ratio) when $N _ { F } > 0$ . The clip passes Gate 1 only if all three conditions hold:

$$
r _ { F } \geq 0 . 8 5 , \qquad n _ { F } \geq 3 , \qquad L _ { F } \leq 2 .\tag{5}
$$

A clip with $N _ { F } = 0$ has an undefined pass ratio and fails the gate. This gate favors limited camera motion with reliable optical-flow estimates.

Gate 2: visible hands. A YOLO hand detector is applied to every sampled frame. A detected hand bounding box is valid only if its confidence is at least 0.30, its area is between 2% and 50% of the image area, and it intersects the normalized image region [0.075, 0.075, 0.925, 1.0] (left, top, right, bottom). Let $h _ { i }$ (hand\_count) be the number of valid hand boxes in sampled frame i. For a clip with $N$ sampled frames, let $\begin{array} { r } { n _ { H } = \sum _ { i = 1 } ^ { N } \mathbf { 1 } [ h _ { i } \geq 1 ] } \end{array}$ (hand1\_pass\_count), let $r _ { H } = n _ { H } / N$ (hand1\_pass\_ratio), and let $L _ { H }$ (hand1\_max\_invalid\_run) be the longest consecutive run of frames with $h _ { i } < 1$ . The clip passes Gate 2 only if

$$
r _ { H } \geq 0 . 1 0 , \qquad n _ { H } \geq 1 , \qquad L _ { H } \leq 5 .\tag{6}
$$

Clips with no sampled frames fail this gate. This criterion requires at least one sufficiently visible hand; it does not require two hands.

Caption generation. For each 5-second clip, we sample every tenth frame to form a sequence of 15 frames and provide it to a locally deployed Qwen3.6-35B-A3B model. We require a non-empty English step\_description: one imperative sentence beginning with a verb and describing the principal visible activity. The field must remain non-empty even if the clip is marked Invalid; in that case, the description should conservatively state the visible non-manipulation activity without inventing an object interaction. The resulting description is used as the text instruction when predicting future visual states from the clip.