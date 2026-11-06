# Aligning Policy Encoders with a Shape Teacher

Madelon Bernardy ∗ University of Amsterdam

Mohammad Mahdi Derakhshani University of Amsterdam

David M Knigge New Theory AI

Riccardo Valperga University of Amsterdam

Artem Moskalev New Theory AI

## Abstract

3D-aware representations can improve robotic manipulation by providing geometric information that is robust to changes in viewpoint, object position, and occlusion. However, using 3D inputs at deployment requires additional sensing and processing, making image-based policies more practical. We ask whether an image-based policy can acquire 3D shape information during training while retaining a purely image-based interface at deployment. We propose shape-teacher alignment, a training-time mechanism that uses a pretrained 3D shape representation as an alignment target for an image-based policy encoder. Specifically, we use Point-MAE features from individual objects as targets for corresponding image patches, transferring geometric structure to the policy without changing its deployment inputs or architecture. On LIBERO, shape-teacher alignment produces image representations that are more structured around object shape and corresponding parts, while substantially accelerating policy convergence and improving success rates. For pretrained policies, convergence is accelerated by 4.7× and 5.6× on the Object and Spatial suites, respectively, without requiring the shape teacher or depth input at deployment. The project is available at: https://miepjemadelon. github.io/aligningpolicieswithashapeteacher/.

## 1 Introduction

Geometric information is important for robotic manipulation, with 3D-aware policies demonstrating improved data efficiency and robustness to variations such as viewpoint changes, object displacement, and occlusion [Zhen et al., 2024, Sun et al., 2025, Li et al., 2026a, Qu et al., 2025, Ze et al., 2024, Chen et al., 2023]. However, incorporating 3D observations into a robotic policy requires additional sensing and processing, including depth cameras, calibration, and depth estimation. These costs make purely image-based policies attractive in practice, but such policies can lack the geometric information needed for robust manipulation [Sun et al., 2025, Garcia et al., 2025]. We therefore ask: can a 2D robotic policy acquire shape-aware representations during training without requiring 3D inputs at deployment?

Recent work on diffusion models has shown that pretrained representations can guide a target model through Representation Alignment (REPA). REPA [Yu et al., 2025] aligns intermediate representations with features from a frozen pretrained encoder, improving convergence and generation quality, and has since been applied to robotic policies using pretrained vision encoders as alignment targets [Li et al., 2025, Qiu et al., 2026]. However, these approaches do not specifically target 3D geometric structure, leaving open whether alignment with a representation that explicitly captures 3D shape can introduce geometric structure into an image-based policy. A shape teacher can encourage the policy encoder to represent corresponding geometric structure across objects, while the policy itself remains image-based at deployment.

We use a pretrained 3D shape teacher, Point-MAE [Pang et al., 2022], to align the image encoder of a 2D diffusion policy during training. Point-MAE features from individual objects are aligned with their corresponding image patches, encouraging the policy encoder to incorporate geometric structure while retaining the original image-based policy interface. We evaluate the resulting representations and policies on LIBERO, finding that shape-teacher alignment produces image representations that are more structured around object shape and corresponding object parts, while the resulting policies converge substantially faster and achieve higher success rates. For pretrained policies, convergence is accelerated by 4.7× and 5.6× on the Object and Spatial suites, respectively, with no need for the shape teacher or depth input at deployment. A comparison with a semantic teacher shows that shape and semantic alignment induce different representations, demonstrating that alignment can be used to impose distinct structural biases on the policy encoder.

To summarize, we make the following contributions:

• We introduce shape-teacher alignment, a training-time mechanism for transferring 3D shape structure into an image-based robotic policy encoder without requiring 3D inputs at deployment.

• We instantiate shape-teacher alignment by matching per-object features from a pretrained 3D shape encoder to corresponding image patches, while keeping the deployed policy unchanged.

• On LIBERO, shape-teacher alignment produces more shape-structured representations and substantially faster policy convergence, with 4.7× and 5.6× speedups for pretrained policies on the Object and Spatial suites, respectively.

## 2 Method

Baseline Diffusion Policy We use the diffusion policy of Chi et al. [2025] as our behavioral cloning baseline. The policy conditions on two consecutive observations, t − 1 and $t ,$ and predicts a 10-step action trajectory, of which 8 steps are executed. We keep the action diffusion process unchanged, while modifying the observation encoder and transformer decoder. The observation encoder is a ViT/S-14 with the same architectural modifications as DINOv2: the CLS token is discarded, and the remaining patch tokens are used as observation features. The language token is produced by a frozen CLIP language model [Radford et al., 2021], and the diffusion timestep is encoded using a sinusoidal embedding. The transformer decoder follows the cross-attention architecture of Chi et al. [2025] and consists of a 12-layer decoder and a 2-layer encoder for the conditioning tokens. Inspired by Dasari et al. [2025], the output projections of the cross-attention blocks are zero-initialized to improve stability.

Shape-Teacher Alignment We align image-patch representations with features from a frozen pretrained 3D shape encoder. We use Point-MAE [Pang et al., 2022] as the shape teacher, which encodes individual 3D objects into local shape features. For each object, we construct a point cloud from its mask and depth map, resample it to 1024 points, and partition the points into local groups of M = 32 points. Point-MAE then produces a feature for each group.

The key challenge is to associate these 3D features with the corresponding image patches. We therefore back-project each 3D point to its corresponding pixel in the image and use these correspondences to determine how strongly each local 3D group contributes to each image patch (Figure 6). Let B denote the set of object masks in an image, $P$ the set of image patches, and $\hat { A ^ { j } }$ the set of point groups belonging to object $j \in B ,$ . For a point group $a \in A ^ { j }$ and image patch $p \in P ,$ , let $\pi ( \bar { d , } p ) \in \mathbf { \bar { \{ 0 , 1 \} } }$ indicate whether point $d \in a$ falls into patch $p .$ The fraction of points in group a that fall into patch p is then

$$
\phi ( a , p ) = { \frac { 1 } { M } } \sum _ { d \in a } \pi ( d , p ) ,\tag{1}
$$

which we use to weight the contribution of each 3D group to the patch. The total coverage of patch p by object j is

$$
\omega _ { p } ^ { j } = \sum _ { a \in A ^ { j } } \phi ( a , p ) .\tag{2}
$$

Let $z _ { a } ^ { j }$ denote the Point-MAE feature of group a from object $j .$ . We use the coverage weights to aggregate the corresponding 3D features into a single alignment target for each image patch:

$$
t _ { p } ^ { j } = { \frac { 1 } { \omega _ { p } ^ { j } } } \sum _ { a \in A ^ { j } } \phi ( a , p ) z _ { a } ^ { j } .\tag{3}
$$

This target represents the 3D shape information associated with the portion of object j visible in patch p.

We then align the corresponding image and 3D representations using a coverage-weighted cosine loss. Let $f _ { p }$ denote the image encoder feature for patch $p ,$ and let h be an alignment head with the same architecture as Li et al. [2025]. The alignment loss is

$$
\mathcal { L } _ { a l i g n } ( f ) = \frac { \sum _ { j \in B } \sum _ { p \in P } \omega _ { p } ^ { j } \left( 1 - \cos _ { - } \sin ( h ( f _ { p } ) , t _ { p } ^ { j } ) \right) } { \sum _ { j \in B } \sum _ { p \in P } \omega _ { p } ^ { j } } .\tag{4}
$$

Following REPA [Yu et al., 2025], we add the alignment loss to the behavioral cloning objective:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { b c } + \lambda \mathcal { L } _ { a l i g n } , } \end{array}\tag{5}
$$

where λ controls the strength of the alignment objective. The shape teacher is kept frozen and used only during training, so the deployed policy remains unchanged and incurs no additional inference cost.

## 3 Experiments

We evaluate whether shape-teacher alignment changes the learned image representations and improves policy learning, while preserving the image-only deployment interface. We consider two common regimes for robotic policy learning: training a task-specific policy from scratch on the target tasks, and fine-tuning a policy from a pretrained checkpoint. This allows us to evaluate shape-teacher alignment both when the policy learns its visual representation from scratch and when it starts from an existing pretrained representation.

Dataset We conduct all experiments on the LIBERO benchmark [Liu et al., 2023], using the LIBERO-Object and LIBERO-Spatial suites. For experiments with pretrained policies, we use LIBERO-90 to pretrain the policy. To evaluate robustness to changes in object placement, we use Object-Displacement and Spatial-Displacement datasets constructed by placing targets on two rings around their original locations (Figure 9). We additionally evaluate zero-shot robustness to occlusion using LIBERO-Occ [Li et al., 2026b].

Evaluation We evaluate whether shape-teacher alignment changes the structure of the learned image representations and improves downstream policy learning. We examine whether aligned features are organized more strongly by object identity, object category, shape, and corresponding object parts. Policy performance is measured by task success rate throughout training, allowing us to compare both final performance and convergence speed. For robustness experiments, we measure the change in success rate relative to the standard evaluation setting under target displacement and occlusion.

![](images/1cdb0e0c841c270699eef54a24db62e3be38911fe510a0bba81fd7e855bf7f19.jpg)  
Figure 1: Patch similarity visualization. The red square marks the query patch and the green square its most similar patch.

![](images/69862772314ed96fe08c6fda878405f1e9e819ce15e2e51939c52af04895c1ce.jpg)  
Figure 2: Object-level image representations on LIBERO-Object for the from-scratch policy. Colors correspond to object categories. Shape-teacher alignment produces clearer object-level structure than the baseline.

Table 1: Percentage of query patches whose most similar patch falls on the same object, another object with similar shape, or an unrelated region. For each query patch, the match is the patch with the highest cosine similarity among the evaluated images.
<table><tr><td></td><td>Same object</td><td>Similar-shape object</td><td>Unrelated</td></tr><tr><td>From pre-trained</td><td></td><td></td><td></td></tr><tr><td>Baseline</td><td>10.2</td><td>8.2</td><td>81.6</td></tr><tr><td>Shape teacher aligned</td><td>8.9</td><td>35.1</td><td>56.0</td></tr><tr><td>From scratch</td><td></td><td></td><td></td></tr><tr><td>Baseline</td><td>17.8</td><td>10.7</td><td>71.5</td></tr><tr><td>Shape teacher aligned</td><td>1.3</td><td>36.4</td><td>62.2</td></tr></table>

Baselines We compare shape-teacher alignment against the corresponding no-alignment diffusion policy in both training regimes. In the from pretrained setting, the policy is initialized from a checkpoint trained on LIBERO-90 and then fine-tuned on the target tasks. In thefrom scratch setting, the same architecture is trained from random initialization on the target tasks. Unless otherwise stated, the aligned model uses Point-MAE as the shape teacher with alignment weight λ = 3.

## 3.1 Shape- and Object-centric Representation

We examine whether shape-teacher alignment transfers geometric structure into the image encoder, using patch-level similarity to assess correspondence across object parts and object-level clustering to assess object structure.

Shape Grouping We first examine whether image patches representing similar object parts and shapes become more similar after alignment. For each selected patch, we identify the most similar patch among the image features and record whether it corresponds to the same object, another object with similar shape, or an unrelated region. Table 1 shows the resulting distribution on LIBERO-Object. For pretrained policies, alignment with the shape teacher increases the fraction of matches to another similar-shape object from 8.2% to 35.1%, while the fraction of matches falling into the “unrelated” category decreases from 81.6% to 56.0%. The same trend is observed for policies trained from scratch, where similar-shape matches increase from 10.7% to 36.4%. Figure 1 provides qualitative examples of these matches: aligned features can match corresponding parts across similarly shaped objects, such as the neck of a bottle or the side of a bowl, rather than only nearby patches of the same object. These results indicate that alignment introduces shape-related structure into the patch representations.

occluded  
![](images/d4c194f7eff8f148618836bc7d6fe2da466d82c4d33d1f69bbe18135f309314d.jpg)  
LIBERO-Object,  
pretrained

![](images/c06c7c7bfe24975a4f6b7f012580aa3bfb35c877cde8b5d9753cca5fd05955f0.jpg)  
LIBERO-Spatial, pretrained

![](images/7faf165421c094f9f87b033b06915cc3e6c3b070c27097d6ce0f894fd7f6525c.jpg)  
LIBERO-Object, from scratch

![](images/4ec22d4d54920bd7b414a9d9638fe03972f5582232c886c35073757470be2a9c.jpg)  
LIBERO-Spatial, from scratch

Figure 3: Policy learning with and without shape-teacher alignment across LIBERO-Object and LIBERO-Spatial, for pretrained and from-scratch policies. Shape-teacher alignment accelerates convergence in both training regimes. Shaded regions show ±1 standard deviation over three seeds.  
![](images/a93021dcd1893edb00e70deabd6d971eb15b9ce2c2ab72b8d47458cc66c30075.jpg)  
occluded

![](images/9d8019ab530b203d070bad11cdb4311afc8583328f6ab5880470edd488df772d.jpg)

![](images/ae45b0c41d0f5bbd5bdd489ea97978c070e392918a03d6dbfea609baf1ca75ba.jpg)  
LIBERO-Object, displaced

![](images/0f97c6bc649d16252c4c5e2478bb19152ded560f21168a37ff8af2faeb62767f.jpg)  
LIBERO-Spatial, displaced  
Figure 4: Zero-shot robustness to target occlusion and displacement for the pretrained policies. Annotations show the relative change in success rate from standard evaluation. Error bars show ±1 standard deviation over three seeds.

Object Grouping We next examine whether alignment also organizes representations at the object level. We average the patch features corresponding to each object and visualize the resulting representations with t-SNE in Figure 2. Compared with the baseline, the aligned representations form clearer clusters of objects, indicating that alignment also produces stronger object-level structure. Together with the patch-level analysis above, this suggests that shape-teacher alignment organizes the image representation both across corresponding object parts and at the level of whole objects.

## 3.2 Improved Convergence

We next examine whether the representational changes induced by shape-teacher alignment translate into faster policy learning. Figure 3 compares the success rate throughout training for the baseline and Point-MAE-aligned policies in both training regimes. For pretrained policies, alignment reaches the same success rate as the baseline 4.7× faster on LIBERO-Object and 5.6× faster on LIBERO-Spatial. The effect is also present when training from scratch, with convergence accelerated by 2.3× and 3.9× on the Object and Spatial suites, respectively.

The consistent acceleration across both training regimes shows that the shape-teacher objective improves the optimization trajectory of the policy rather than only changing its final representation. The larger gains for pretrained policies suggest that shape-teacher alignment is particularly effective when adapting an existing visual representation to the target manipulation tasks.

## 3.3 Robustness

We next examine whether the representational changes induced by shape-teacher alignment improve robustness to changes in the deployment environment. We evaluate performance under target displacement and object occlusion by measuring the change in success rate relative to the standard evaluation setting. Figure 4 shows that the effect of alignment depends on both the type of distribution shift and the task suite. Under occlusion, the aligned policy has a similar performance drop to the baseline on LIBERO-Object but a smaller drop on LIBERO-Spatial. Under target displacement, alignment reduces the performance drop on LIBERO-Object, while the difference on LIBERO-Spatial is smaller. Thus, shape-teacher alignment does not provide a uniform improvement in robustness across all distribution shifts, although it performs similarly or better than the baseline across the tested conditions.

Table 2: Success rate at epoch 100 and convergence speed-up for alignment with PointMAE and DINOv2. Convergence speed-up is measured relative to the corresponding no-alignment baseline. Standard deviation over three seeds are reported for Baseline and PointMAE alignment.
<table><tr><td colspan="2"></td><td>Baseline</td><td>PointMAE</td><td>DINOv2</td></tr><tr><td rowspan="4">Success rate (ep. 100)</td><td>Object, from pretrained</td><td> $0 . 8 5 \pm 0 . 0 0$ </td><td> $0 . 9 0 \pm 0 . 0 4$ </td><td>0.91</td></tr><tr><td>Object, from scratch</td><td> $0 . 1 0 \pm 0 . 0 5$ </td><td>0.53</td><td>0.75</td></tr><tr><td>Spatial, from pretrained</td><td> $0 . 7 1 \pm 0 . 0 2$ </td><td>0.82 ± 0.05</td><td>0.85</td></tr><tr><td>Spatial, from scratch</td><td> $0 . 2 0 \pm 0 . 0 6$ </td><td>0.62</td><td>0.35</td></tr><tr><td rowspan="4">Convergence speed-up</td><td>Object, from pretrained</td><td></td><td>4.7×</td><td>8.6×</td></tr><tr><td>Object, from scratch</td><td></td><td>2.3×</td><td>≥ 11.1×</td></tr><tr><td>Spatial, from pretrained</td><td></td><td>5.6×</td><td>8.9×</td></tr><tr><td>Spatial, from scratch</td><td></td><td>3.9×</td><td>3.7×</td></tr></table>

![](images/10afb73f7dfe8d7a43b5566131da731fbec45ed2b67ffbf5a77eb6e976adee65.jpg)  
Figure 5: Patch-level similarity after alignment with DINOv2 and PointMAE. DINOv2 produces more object-oriented similarity patterns, whereas PointMAE produces greater similarity across corresponding parts of similarly shaped objects.

## 3.4 Alignment Target Ablation

We finally examine whether the structure learned by the policy encoder depends on the choice of alignment target. We replace the Point-MAE shape teacher with DINOv2 [Oquab et al., 2024], a pretrained semantic vision encoder, while keeping the alignment procedure and policy architecture unchanged. DINOv2-based runs use an alignment weight of λ = 1. Table 2 compares the resulting policy performance and convergence speed. DINOv2 provides a strong alignment target and achieves higher success rates in several settings, while Point-MAE remains competitive on pretrained policies.

More importantly, the two alignment targets produce qualitatively different representation structures. Figure 5 shows patch-level similarity for the two aligned encoders. DINOv2 produces more semanticoriented similarity patterns, whereas Point-MAE produces greater similarity across similarly shaped objects and corresponding object parts. Thus, the effect of alignment is not determined only by the alignment procedure itself: the choice of teacher influences the structural bias learned by the policy encoder.

## 4 Conclusion

We introduced shape-teacher alignment as a training-time mechanism for transferring 3D shape structure into an image-based robotic policy without requiring 3D inputs at deployment. On LIBERO, alignment produces image representations that are structured around both object shape and object identity, and substantially accelerates policy convergence, with the largest gains observed for policies initialized from pretrained checkpoints. Robustness to distribution shifts is more conditional, varying across perturbation types and task suites. Comparisons with a semantic teacher further show that the choice of alignment target determines the structure induced in the policy representation. These results suggest that representation alignment can be used not only to improve policy optimization, but also to deliberately shape the geometric structure learned by an image-based robotic policy while preserving its deployment interface.

Limitations and Future Work Our evaluation is limited to a single policy architecture and the LIBERO benchmark, and uses Point-MAE as a single 3D shape teacher. The effect of shape-teacher alignment across other policy architectures, robotic manipulation benchmarks, and pretrained 3D representations therefore remains to be established. The robustness experiments also show that alignment does not uniformly improve performance under all distribution shifts. Future work wil investigate whether stronger or more diverse 3D teachers, including representations that combine geometric and semantic information, can transfer complementary structure to the policy encoder, and whether alignment can improve robustness under more challenging forms of occlusion and distribution shift. Extending the evaluation to additional architectures and real-robot settings will further clarify the generality of training-time geometric alignment.

## References

Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. Lof: identifying densitybased local outliers. In Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data, SIGMOD ’00, page 93–104, New York, NY, USA, 2000. Association for Computing Machinery. ISBN 1581132174. doi: 10.1145/342009.335388. URL https: //doi.org/10.1145/342009.335388.

Shizhe Chen, Ricardo Garcia, Cordelia Schmid, and Ivan Laptev. PolarNet: 3D Point Clouds for Language-Guided Robotic Manipulation, November 2023.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion, 2025. URL https://doi.org/10.1177/02783649241273668.

Sudeep Dasari, Oier Mees, Sebastian Zhao, Mohan Kumar Srirama, and Sergey Levine. The ingredients for robotic diffusion transformers, 2025. URL https://doi.org/10.1109/ICRA55743. 2025.11127493.

Ricardo Garcia, Shizhe Chen, and Cordelia Schmid. Towards generalizable vision-language robotic manipulation: A benchmark and llm-guided 3d policy. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 8996–9002, 2025. doi: 10.1109/ICRA55743.2025. 11127315.

Chengmeng Li, Junjie Wen, Yaxin Peng, Yan Peng, and Yichen Zhu. Pointvla: Injecting the 3d world into vision-language-action models, 2026a.

Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial forcing: Implicit spatial representation alignment for vision-language-action model, 2025. URL https://arxiv.org/abs/2510.12276.

Taishan Li, Jiwen Zhang, Siyuan Wang, Xuanjing Huang, and Zhongyu Wei. Libero-occ: Evaluating and improving vision-language-action models under scene-induced occlusion via viewpoint imagination, 2026b. URL https://arxiv.org/abs/2606.10862.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: benchmarking knowledge transfer for lifelong robot learning, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets\_and\_Benchmarks.html.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. URL https://arxiv.org/abs/2304.07193.

Yatian Pang, Wenxiao Wang, Francis E. H. Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked Autoencoders for Point Cloud Self-supervised Learning. In Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision – ECCV 2022, volume 13662, pages 604–621. Springer Nature Switzerland, Cham, 2022. ISBN 978-3-031-20085-4 978-3-031-20086-1. doi: 10.1007/978-3-031-20086-1\_35. URL https: //link.springer.com/10.1007/978-3-031-20086-1\_35. Series Title: Lecture Notes in Computer Science.

Lu Qiu, Yizhuo Li, Yi Chen, Yuying Ge, Yixiao Ge, and Xihui Liu. Making foresight actionable: Repurposing representation alignment in world action models, 2026. URL https://arxiv.org/ abs/2606.12217.

Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. Spatialvla: Exploring spatial representations for visual-language-action model, 2025. URL https://arxiv.org/abs/2501.15830.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. URL http:// proceedings.mlr.press/v139/radford21a.html.

Lin Sun, Bin Xie, Yingfei Liu, Hao Shi, Tiancai Wang, and Jiale Cao. Geovla: Empowering 3d representations in vision-language-action models, 2025. URL https://arxiv.org/abs/2508. 09071.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think, 2025.

Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. In Proceedings of Robotics: Science and Systems (RSS), 2024.

Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3D-VLA: A 3D vision-language-action generative world model. In Ruslan Salakhut dinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 61229–61245. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/zhen24a.html.

Table 3: Similarity groups used for cluster analysis
<table><tr><td>Shape</td><td>Objects</td></tr><tr><td>Drink Carton</td><td>Milk, Orange Juice</td></tr><tr><td>Box</td><td>Cream Cheese, Butter, Chocolate Pudding</td></tr><tr><td>Tin</td><td>Tomato Sauce, Alphabet Soup</td></tr><tr><td>Bottle</td><td>BBQ Sauce, Salad Dressing, Ketchup</td></tr></table>

## A Model Architecture

Figure 6 shows a schematic overview of the method. The left side shows how the alignment target is determined, using back-projection to match point clusters to image patches. The right side depicts the action prediction pipeline.

![](images/c89a8012fd1d4b9f673aabf4eeb3c9217ba039321f3008b7589c74105794aa74.jpg)  
Figure 6: Overview of the methodology. On the left, the pipeline to retrieve Point-MAE targets is visualized. On the right, the action-prediction pipeline is visualized.

## B Shape grouping Experimental Details

For the shape grouping experiment, similar patches are grouped based on the object mask they land on. See Table 3 for the clustering groups. To get the final result, per encoded image, we select all patches that overlap by at least 50% with object masks in Table 3. For each of these patches in the object, the most similar patch in terms of cosine similarity is selected. Examples of this can be found in Figure 1 and in Figure 5. We record whether this patch is inside the same object instance, inside an object within the same group (see Table 3), or somewhere else in the image. We average these results over 1000 images to get the final results shown in Table 1.

## C Iterations over Lambda

In addition to testing the effect of alignment for a fixed alignment weight, we also examined the effect of different alignment weights λ on the model’s convergence speed. The results for the from-scratch and pre-trained models can be found in Figure 7. These results suggest that the alignment weight λ has a sweet spot, and that exceeding it can hurt performance, especially when training from scratch, as shown on the LIBERO-Object set. However, the convergence results differ greatly per configuration. For instance, on the pre-trained models, a higher alignment weight does not seem to strongly affect convergence, nor does it clearly degrade convergence. Still, more configurations and more seeds are needed before we can draw a clear conclusion from these results.

## D Additional Target Ablation Results

We additionally compared PointMAE alignment against DINOv2 alignment for convergence speed, shown in Figure 8, and for quantitative patch similarity, shown in Table 4. For convergence speed, our findings are similar to those discussed in Section C: there appears to be an optimum value for the alignment weight, after which convergence speed drops or stays similar, depending on the dataset and on the pre-training. When considering the shape groupings of DINOv2, our findings again are similar to those in Section 3.4. Where Point-MAE aligned models group patches with similar shapes, more so than patches within the same object, DINOv2 aligned models often select patches from the same object, rather than patches with the same shape. Overall, the fraction of patches whose most similar patch is on an unrelated item is similar for both models.

![](images/4d914faf225060522bde17e5d57345bb27051ffcc6208e2e0bbc03e47ef2b793.jpg)  
(a) LIBERO-Object, from scratch.

![](images/65e10271ea94d6b84bdcc116271fff08dfaa43534babddc8e151ec45bae503d9.jpg)  
(b) LIBERO-Spatial, from scratch.

![](images/a28597577fe2777724eb143a6ef60352f9c01364594b5ee59c39c350a17844c5.jpg)  
(c) LIBERO-Object, pretrained.

![](images/71da35c1660e0b23bf4680b7b36fcb555fd8b9e33a7c3d929982c2f65e9ac875.jpg)  
(d) LIBERO-Spatial, pretrained.  
Figure 7: The convergence of each model on the LIBERO-Object and LIBERO-Spatial datasets for different alignment weights λ, for both pre-trained and from scratch configurations. Shaded regions show ±1 standard deviation over three seeds.

Table 4: Percentage of query patches whose most similar patch falls on the same object, another object with similar shape, or an unrelated region, including DINOv2 [Oquab et al., 2024] results. For each query patch, the match is the patch with the highest cosine similarity among the evaluated images.
<table><tr><td></td><td>Same object</td><td>Similar-shape object</td><td>Unrelated</td></tr><tr><td>From pre-trained</td><td></td><td></td><td></td></tr><tr><td>Baseline</td><td>10.2</td><td>8.2</td><td>81.6</td></tr><tr><td>Shape teacher aligned</td><td>8.9</td><td>35.1</td><td>56.0</td></tr><tr><td>DINOv2 aligned</td><td>31.7</td><td>5.7</td><td>62.6</td></tr><tr><td>From scratch</td><td></td><td></td><td></td></tr><tr><td>Baseline</td><td>17.8</td><td>10.7</td><td>71.5</td></tr><tr><td>Shape teacher aligned</td><td>1.3</td><td>36.4</td><td>62.2</td></tr><tr><td>DINOv2 aligned</td><td>30.1</td><td>11.7</td><td>58.2</td></tr></table>

## E Datasets

The out-of-distribution placements of the rings are shown in Figure 9a and in Figure 9b for LIBERO Object displacements and LIBERO-Spatial displacements, respectively.

![](images/95892c7e1ff55f9637da7785d10066bd9e00530dc5a11dcacaa57a34fa016a76.jpg)  
(a) LIBERO-Object, from scratch.

![](images/4af4fb71fc3ed0f25d915d3b8fa3ba7bd66084228c3817af2f56e3d9828b2975.jpg)  
(b) LIBERO-Spatial, from scratch.

![](images/d85091bd6f6a8a2d7d5ffc8cb1c1465db609fb7b3a1b0663b86f6dab250f96b8.jpg)  
(c) LIBERO-Object, pretrained.

![](images/6132300c327dee7b69487bdd972cc6ea74f14c3394f225ade1afc55f51635514.jpg)  
(d) LIBERO-Spatial, pretrained.  
Figure 8: The convergence of each model on the LIBERO-Object and LIBERO-Spatial datasets for different alignment weights λ, for both pre-trained and from scratch configurations, for alignment with DINOv2. Shaded regions show ±1 standard deviation over three seeds.

To generate the rings, first, the object bounding box over the original dataset target displacements is computed. To generate LIBERO-Spatial displacements, outlier filtering was performed to generate the bounding box, as the spread is much larger than for LIBERO-Object. Not filtering out outliers caused performance on the outermost rings to drop to near zero. The Local Outlier Factor [Breunig et al., 2000] was used to identify outliers.

From the bounding box, the rings are computed as fractions of the diameter of the bounding box. Ring 1 starts at 0.1 · diameter from the edge and ends at 0.25 · diameter. Ring 2 starts at 0.35 · diameter from the edge and ends at 0.50 · diameter. After the areas were created, 15 points were sampled from these areas using rejection sampling.

## F Compute

All experiments were run on the <redacted for anonymity> compute cluster. All experiments were run on an A100 GPU. Each training run took about 5 hours, and each checkpoint validation took about 15 minutes, for both standard and out-of-distribution validation. Experiments that only used the image encoder, such as patch similarities and t-SNE clustering, were negligible in compute, with each task taking less than 10 minutes. In total, training and evaluating the Point-MAE models, baseline models, and DINOv2 models used in the main paper, as well as the other experiments in the main paper, took around 150 hours on an A100 GPU. The hyperparameter search over λ for both Point-MAE and DINOv2 took approximately 100 additional hours on an A100 GPU. However, the full research project required more resources, including additional configurations that did not make it into the paper and testing other properties of those models, such as different variants of alignment heads.

![](images/b79504eaeb34143e18ab06370fe8556ce9aca8f77618464682c67e2a39aa7578.jpg)

![](images/02ac96ba3f2a391256e9027e933de2d24f4b3fdcbb83edba8f920a6746537668.jpg)

![](images/9ae5868996362c841d198892abd202c58884e3536e5b587480a887ed0b4ff94e.jpg)

![](images/5054fa2430d91b0e00a5bd719e3aec294cee8984e741dc1f55b10d38856bdfb5.jpg)

![](images/012cdf34ac00c095fa01109aece8cdb77bce1a9b648e326025bb31995a2366c2.jpg)

![](images/95ec7759b3ce3d0c7c4fa869da4ee1847f96ef3edbe440ffb892d6f181ddda3a.jpg)

![](images/b6f1d7c4eb6f68a6d30025fdf62eba7e2c03232b77aad53a962b92de2862cc6c.jpg)

(a) LIBERO-Object.  
![](images/44d53d3e0cb96ff16138b699ebfcc9e99f04e6498864beb79ad913183397c1a0.jpg)

![](images/508bdb8778d9936a87b94c6b519694e7d4ebc709068f6c4db6263c4d40fb3480.jpg)

![](images/fc8e7e73c61120f0213891cb29d452f202043750ebf0e074e7b8b3c1211d81dd.jpg)

![](images/57319b4dd89f8a3aae9fc34a524d7192eadaac7943aa8cd242cfc4869c5b8c52.jpg)  
(b) LIBERO-Spatial.  
Figure 9: Displacements of the target object per task, for both suites. Orange marks the placements of the original dataset; blue and green mark the two out-of-distribution rings, ring 1 and ring 2, respectively.