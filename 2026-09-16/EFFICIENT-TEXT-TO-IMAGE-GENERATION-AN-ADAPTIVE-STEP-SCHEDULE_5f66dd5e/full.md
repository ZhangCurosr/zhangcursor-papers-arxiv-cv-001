# EFFICIENT TEXT-TO-IMAGE GENERATION: AN ADAPTIVE STEP SCHEDULE CONTROLLER FOR DIFFUSION MODELS

Kuluhan Binici<sup>1,</sup> <sup>2</sup>, Cihan Acar<sup>3</sup>, Shivam Aggarwal<sup>2</sup>, Siying Liu<sup>3</sup>, Tulika Mitra<sup>2</sup>

<sup>1</sup>SAP, <sup>2</sup>National University of Singapore, <sup>3</sup>Institute for Infocomm Research (I2R), A\*STAR, Singapore

## ABSTRACT

Text-to-image diffusion models often use a fixed number of denoising steps, balancing time costs and image quality. However, the optimal number of steps depends on the complexity of the input text prompt. We propose an adaptive diffusion controller that dynamically adjusts the number of steps to generate high-quality images efficiently, without additional model training. By leveraging a mixture of step schedules with varying step sizes and evaluating the error term discrepancy at each timestep, our method transitions between schedules to optimize performance. Experiments on COCO and DiffusionDB show that our approach reduces inference time while maintaining visual fidelity, offering a more efficient alternative for text-to-image diffusion models.

Index Terms— Diffusion models, efficient AI

## 1. INTRODUCTION

Diffusion models have showcased an impressive ability to create high-diversity and high-quality images from textual descriptions [1, 2]. One of the primary challenges with diffusion models is the high computational time incurred during the inference process. Knowledge distillation and neural architecture search techniques have been implemented to decrease the required number of time steps in the generation process [3, 4]. However, these require supplementary training of models. In addition, these methods typically use a fixed number of diffusion steps without considering the specific needs or complexity of the task. This one-size-fits-all approach to the denoising schedule overlooks the nuanced dynamics of the diffusion process, which can lead to either redundant computation or potentially immature results. As illustrated in Figure 1, the optimal number of steps is affected by the initialization of the noisy image, i.e. the random seed, and the text prompt. Notably, the dependence of the optimal time step schedule for the specific generation task has also been acknowledged by previous works [5]. In this paper, we address this inefficiency by proposing an adaptive controller designed to allocate a minimal number of steps required to achieve visually satisfactory results for each text prompt. This allows better visual quality retention while improving time efficiency, rather than simply reducing the fixed schedule for all images. We named our approach as Adaptive Diffusion Step Controller (ADSC). ADSC does not require additional model training and functions by monitoring the convergence of the generation process during runtime to adjust the schedule accordingly. We postulate that the cosine distance between the conditional and unconditional update directions can serve as a heuristic for estimating convergence. When the cosine distance is large, indicating that the content described by the text prompt has not been generated in the denoised image yet, our scheduler opts for taking smaller step sizes to refine the generative process with finer granularity. Conversely, when the distance narrows, we infer that such content information is already contained in the image; thus taking larger step sizes are employed, expediting the overall process without compromising the image quality. Our observations reveal a discernible pattern where the cosine distance between update directions is initially substantial, reflecting the high entropy in the early stages that the diffusion process has not converged and the denoised image has not aligned with the text prompt. This distance typically diminishes as the process unfolds, permitting the scheduler to incrementally increase the step size.

![](images/611605ef18b05c45a386533fbfd582cb263749a68c7bc6e268d0538dbc709603.jpg)  
Fig. 1: Effect of text prompt and seed combinations on diffusion steps for generating visually sufficient images. Step counts shown are examples; optimal steps may vary.

To validate the efficacy of our proposed method, we conducted extensive experiments on two well-established datasets: COCO [6] and DiffusionDB [7]. For evaluation metric, we utilize a variant of the CLIP score [8]; namely CLIP-I [9], and also the DINO metric [10, 9]. These metrics mainly compare the feature-space projections of the generated and reference images. The experimental results reveal that our adaptive approach can effectively reduce the average number of timesteps required to generate images from a given sequence of text prompts, in some cases by more than 50%, all while maintaining the visual quality.

## 2. RELATED WORK

To direct the generative process towards predetermined attributes or categories, external classifiers were incorporated to diffusion models but this requires the training and integration of distinct classifier models. Classifier-free diffusion guidance [11] effectively tackles this by embedding the guidance mechanism within the diffusion model itself, eliminating the need for an external classifier. Subsequent studies have enhanced the image generation process more precisely by guiding the internal representations within the diffusion models to control attributes like shape, location, and appearance of objects [12] and also by incorporating additional inputs such segmentation maps, keypoints and bounding boxes to introduce spatial conditioning controls [13, 14]. Due to the iterative nature of the denoising process, diffusion models face a significant issue in that they require a considerable amount of computational time during the inference process, as multiple forward passes are needed for each image generation. To tackle this challenge, efficient solvers such as DDIM [15], PNDM [16] and LMS [17] has been introduced to accelerate the sampling process. In addition, architectural compression [4], quantization [18], and knowledge distillation techniques have been implemented to decrease the required number of time steps in the generation process [3, 19]. Recently, Denoising Diffusion Step-aware Models (DDSMs) [20] have been developed, utilizing neural networks of varying sizes tailored to the specific demands of different stages of the generation process. The main drawback of these techniques is the need for supplementary training of models specifically for distillation, as well as the requirement to undertake searches for the most effective model architectures [5]. Furthermore, these methods employ a fixed number of diffusion steps, overlooking the need to adapt to the distinct challenges posed by different prompts.

## 3. ADAPTIVE DIFFUSION STEP CONTROLLER

Our Adaptive Diffusion Step Controller (ADSC) encompasses two key technical components. First, we establish a diffusion step schedule control space within which our ADSC operates. This space is comprised of various schedules with distinct step lengths, enabling the controller to determine an optimal configuration by intelligently blending these schedules. Importantly, the step schedule control space allows each generation task to be completed in a varying number of steps, depending on the requirements of the generation task. Secondly, we introduce a heuristic criterion that guides the ADSC controller in crafting schedules tailored to each text prompt. This criterion strives to estimate the convergence of the diffusion process by monitoring the cosine distances between the conditional and unconditional noise estimations of the diffusion model.

## 3.1. Heterogenous schedule control space

![](images/8f21b4612deb646af5ace498f62017b697def618834851798ad7ba80b85cb02a.jpg)  
Fig. 2: The denoising process begins with the longest schedule (e.g., 40 steps in the example) and transitions to shorter schedules (20 followed by 10) upon detecting convergence.

The control space consists of K schedules, each with a varying number of steps, denoted as $S ^ { k } = \{ t _ { 1 } ^ { k } , t _ { 2 } ^ { k } , . . . t _ { N ^ { k } } ^ { k } \}$ where $\vert S ^ { k } \vert = N ^ { k }$ represents the number of steps in schedule $k \in [ 1 , K ]$ The difference in time units between the steps in each schedule is distinct, resulting in diverse step sizes across the schedules. The controller initiates the execution by following the schedule with the most number of steps. It transitions to the corresponding step in the subsequent schedule whenever convergence is flagged. Each schedule $S ^ { k }$ has fewer steps and consequently larger step sizes compared to the previous one :

$$
N ^ { k } > N ^ { k + 1 }\tag{1}
$$

$$
t _ { i + 1 } ^ { k + 1 } - t _ { i } ^ { k + 1 } > t _ { j + 1 } ^ { k } - t _ { j } ^ { k } > 0 ; \forall i \in [ 1 , N ^ { k + 1 } ) , \forall j \in [ 1 , N ^ { k } )\tag{2}
$$

This is motivated by the presumption that as the diffusion process converges and a major portion of the content information has been generated, the estimation error incurred by taking larger steps could have less impact on the final image quality.Moreover, the control space can be represented by $\bar { C } = \bar { \{ S ^ { 1 } , S ^ { 2 } , . . . , S ^ { K } \} }$ . When the transition criterion is met during the execution of the $t _ { i } ^ { k }$ timestep of a certain schedule $S ^ { k }$ , the controller transitions to the subsequent schedule and resumes diffusion starting from the timestep $t _ { j } ^ { k + 1 }$ that is closest to $t _ { i } ^ { k } ,$ , i.e.

$$
j = \arg \operatorname* { m i n } _ { j ^ { \prime } } | t _ { j ^ { \prime } } ^ { k + 1 } - t _ { i } ^ { k } |\tag{3}
$$

Finally, the set of all possible mixtures of schedules that can be generated by the controller through its exploration on the control space, can be formulated as:

$$
M = \bigcup _ { k = 1 } ^ { K } \{ t _ { s t } ^ { k } , \dotsc , t _ { e n d } ^ { k } \} ; t _ { 1 } ^ { k } \leq t _ { s t } \leq t _ { e n d } \leq t _ { N ^ { k } } ^ { k }\tag{4}
$$

where $t _ { s t } ^ { k }$ and $t _ { e n d } ^ { k }$ denote the starting and ending timesteps, respectively, of the selected segment from schedule k. The overall process is explained in pseudo-code format in Alg. 1.

Algorithm 1 Control Process of the ADSC Controller   
1: Input: Control space $\boldsymbol { C } = \{ S ^ { 1 } , S ^ { 2 } , . . . , S ^ { K } \}$   
2: Output: Mixed step schedule m   
3: m ← ∅, k ← 1, i ← 1 ▷ Start with the longest schedule   
4: while $t _ { i } ^ { k } < t _ { N ^ { k } } ^ { k }$ do   
5: if convergence detected then   
6: k ← k + 1 ▷ Transition to the next schedule   
7: i ← arg min |t<sup>k</sup><sub>′</sub> −   
j   
<sub>j</sub><sup>′</sup>   
8: else   
9: m ← m ∪ {t<sup>k</sup>} ▷ Add the timestep to the mixed schedule   
10: $i \gets i + 1$   
11: end if   
12: end while

## 3.2. Convergence criterion

In this work we consider text-to-image generation using classifier-free guidance [11]. Classifier-Free Guidance generates both conditional and unconditional noise estimates (denoted by $\epsilon _ { c }$ and $\epsilon _ { u } )$ at each denoising step, steering model outputs to match specific conditions. The conditional estimate targets the given condition (e.g., a text description), while the unconditional estimate is produced without any guiding condition. The noisy image $\boldsymbol { x } _ { t _ { i } }$ is updated as

$$
x _ { t _ { i + 1 } } = f ( x _ { t _ { i } } , \underbrace { \epsilon _ { u } + \gamma ( \epsilon _ { c } - \epsilon _ { u } ) } _ { \mathrm { e s t i m a t e d n o i s e } \left( \epsilon \right) } )\tag{5}
$$

where $f$ is the function used to update the noisy image, and ϵ is the total noise estimate. γ is a coefficient called “guidance scale”, that adjusts the contribution of $\epsilon _ { c }$ and $\epsilon _ { u }$ while updating the image. To determine the convergence of the diffusion process, we track the cosine distance between the unconditional noise estimate and the total estimated noise ϵ, which is influenced by the conditional estimate $\epsilon _ { c }$ . The cosine distance at each time step t is denoted as $d ^ { ( t ) } { \cos ( \epsilon _ { u } , \epsilon ) } = d _ { \mathrm { { c o s } } } ( t )$ . As the diffusion progresses, $d _ { \mathrm { c o s } } ( t )$ exhibits a decline that resembles an inverse exponential function, as exemplified in Figure 3 for a sample drawn from DiffusionDB dataset.

![](images/32159055a92decbad5036f5390c7c0bcfb19374c5dd578885f8dd297f636db9f.jpg)  
Fig. 3: Plot of cosine distance between total and unconditional noise estimates across diffusion steps, along with first and second derivative curves, for a sample from the DiffusionDB dataset. The X-axis represents the diffusion steps.

The elbow point on this curve is the point after which $d _ { \mathrm { c o s } } ( t )$ does not change significantly, indicating that the denoised image has sufficiently converged to the conditional, i.e. the text prompt. At this stage, the quality of the denoised image becomes less sensitive to estimation errors resulting from large step sizes, which allows us to safely transition to the next schedule in the control space. The cosine distance curve $D _ { \mathrm { c o s } } ( t ) = \{ d _ { \mathrm { c o s } } ( t _ { 1 } ) , d _ { \mathrm { c o s } } ( t _ { 2 } ) , \dots , d _ { \mathrm { c o s } } ( t _ { T } ) \}$ is subject to fluctuations, and approaches that make use of the derivative values to locate the elbow can erroneously detect local minima instead. To address this issue, we first smoothen the curve by applying exponential averaging. Let $\tilde { d } _ { \mathrm { c o s } } ( t _ { i } )$ denote the smoothed cosine distance at timestep $t _ { i \cdot }$ which can be computed using exponential averaging:

$$
\tilde { d } _ { \mathrm { c o s } } ( t _ { i } ) = \alpha d _ { \mathrm { c o s } } ( t _ { i } ) + ( 1 - \alpha ) \tilde { d } _ { \mathrm { c o s } } ( t _ { i - 1 } )\tag{6}
$$

where α is the smoothing factor, which determines the degree of smoothing applied to the curve. Next, we examine the first and second derivatives of the smoothed cosine distance curve, denoted by $\tilde { d } _ { c o s } ( t _ { i } ) ^ { \prime }$ and $\tilde { d } _ { c o s } ( t _ { i } ) ^ { \prime \prime }$ respectively:

$$
\tilde { d } _ { c o s } ( t _ { i } ) ^ { \prime } = \tilde { d } _ { c o s } ( t _ { i } ) - \tilde { d } _ { c o s } ( t _ { i - 1 } )\tag{7}
$$

$$
\tilde { d } _ { c o s } ( t _ { i } ) ^ { \prime \prime } = \tilde { d } _ { c o s } ( t _ { i + 1 } ) + \tilde { d } _ { c o s } ( t _ { i - 1 } ) - 2 \times \tilde { d } _ { c o s } ( t _ { i } )\tag{8}
$$

If the absolute values of both derivatives fall below predefined thresholds, we signal that the elbow point has been reached. Finally, we define the convergence criterion in terms of the first and second derivatives:

$$
C = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } | \tilde { d } _ { \mathrm { c o s } } ( t _ { i } ) ^ { \prime } | < \delta _ { 1 } \ \mathrm { a n d ~ } | \tilde { d } _ { \mathrm { c o s } } ( t _ { i } ) ^ { \prime \prime } | < \delta _ { 2 } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{9}
$$

As expressed in Eq. 9 the convergence criterion is set to 1 (indicating convergence) if the absolute values of both the first derivative, and the second derivative, fall below their respective threshold values, $\delta _ { 1 }$ and $\delta _ { 2 }$

## 4. EXPERIMENTS

We integrate ADSC into the Stable Diffusion pipeline [21] and compare the quality of the generated images with those produced using a fixed number of step size. Our experiments involve utilizing PNDM [16] and DDIM [15] schedulers, while attaching our ADSC to these schedulers to control the generation process based on text prompts. In our experiments we chose the number of schedules in our control space, K, as 4, with each schedule containing half the number of steps as the previous one. This way we ensure one of every two time steps to be coinciding among consecutive schedules, allowing flexible transitions. For the guidance scale γ, we use the default value used in Stable Diffusion, that is 7.5. We evaluate the performance of our approach on two text-to-image generation datasets, namely DiffusionDB [7] and COCO [6]. To correct spelling errors, we pre-process DiffusionDB following the procedure outlined in Wu et al. [22]. For evaulation metrics, we use the CLIP-T variant of the CLIP score [8, 23], that measures the alignment between the text prompt describing the image and the generated image itself. Later, to also assess the visual quality, we used the CLIP-I [8, 9] and DINO [10] scores which compares the CLIP and DINO image embeddings of the generated samples with the reference images. Here we assume that images produced with many diffusion steps - mostly 50 steps - are near-optimal and use them as reference points.

## 4.1. COCO results

For our evaluation on the COCO dataset, we first processed the prompts with our ADSC attached to the PNDM [16] and DDIM [15] schedulers within the Stable Diffusion v1- 5 pipeline. In contrast to these fixed-step schedulers, our method tailors the number of steps specifically for each unique image sampling task. Therefore, to ensure a fair comparison with fixed-step schedulers, we determine their step counts based on the minimum, maximum, and average number of steps utilized by our adaptive controller. These values are displayed along with the visualization of the step distributions in Figure 4. Results indicate that the distribution of steps for both schedulers exhibits similarities to the Gaussian distribution. Moreover, the distribution of steps differs for each schedule, and among the two, DDIM appears to require a smaller number of steps when controlled by our ADSC. The ADSC-controlled DDIM resulted in 18.89

![](images/c59eaa451ae02f46190767948c75d2e5248139cb01ec24d71a11d9d93189c545.jpg)  
(a) DDIM

![](images/bf2396d47da5f268a55c820f05679af585b8a7a64328f46939f42859549c02d6.jpg)  
(b) PNDM  
Fig. 4: Distribution of steps upon controlling DDIM and PNDM schedulers using our ADSC in COCO dataset.

steps taken on average per prompt while PNDM led to 24.48 steps on average. These step values are rounded and used for DDIM and PNDM schedulers as baselines. After establishing the number of steps for our baselines, we proceed to compare our method using CLIP-I, CLIP-T, and DINO scores. For calculating reference-based scores, namely CLIP-I and DINO, we employ images generated by fixed 50-step schedules as 50 is the default number of step used in Stable Diffusion pipeline [24]. As presented in Table 1, the results demonstrate that, on average, ADSC attains enhanced visual quality retention for the same number of reverse diffusion steps. When comparing the ADSC-controlled DDIM scheduler to its 19-step fixed counterpart, we observe a 3.3% (95.70 vs 92.40) increase in CLIP-I score and an 8.30% (92.42 vs 84.12) improvement in DINO score, with an average step count of 18.89 steps. Likewise, the ADSC-controlled PNDM scheduler outperforms its fixed-step counterpart, with a CLIP-I score improvement of 1.50% (94.10 vs 92.60) and a DINO score increase of 4.28% (89.24 vs 84.96). Notably, the ADSC-controlled DDIM also surpasses the fixed 31-step schedule. This can be attributed to the heterogeneous schedules offered by our adaptive controller, which are not directly comparable to homogeneous schedules containing the same number of steps. In the case of CLIP-T, the scores remain relatively consistent across various baselines and our method, primarily because it evaluates text-image alignment.

<table><tr><td colspan="5">DDIM</td><td colspan="4">PNDM</td></tr><tr><td>Strategy</td><td>ADSC - ours* (18.89 steps)</td><td>fixed (12 steps)</td><td>fixed* (19 steps)</td><td>fixed (31 steps)</td><td> $\overline { { \mathbf { A D S C - o u r s ^ { * } } } }$  (24.48 steps)</td><td>fixed (16 steps)</td><td>fixed* (24 steps)</td><td>fixed (34 steps)</td></tr><tr><td>CLIP-I (↑)</td><td>95.70</td><td>90.00</td><td>92.40</td><td>95.20</td><td>94.10</td><td>91.30</td><td>92.60</td><td>94.50</td></tr><tr><td>CLIP-T (↑)</td><td>31.40</td><td>31.40</td><td>31.40</td><td>31.40</td><td>31.40</td><td>31.30</td><td>31.40</td><td>31.30</td></tr><tr><td>DINO (↑)</td><td>92.42</td><td>78.78</td><td>84.12</td><td>90.72</td><td>89.24</td><td>81.06</td><td>84.96</td><td>89.20</td></tr><tr><td>Avg wallclock time (sec/img)(↓)</td><td>0.79</td><td>0.55</td><td>0.78</td><td>1.23</td><td>1.06</td><td>0.71</td><td>1.06</td><td>1.40</td></tr></table>

Table 1: Comparison between ADSC and baseline fixed-step schedulers using various metrics on the COCO dataset. (\*) highlights our method and the main baselines.

To ensure a comprehensive assessment, we also measure the average wall-clock times (latencies) per generated image. The results reveal that the latency of ADSC is nearly identical to that of fixed schedulers, with only minor differences up to two decimal places. For instance, the ADSC-controlled DDIM scheduler took an average of 0.79 seconds for 18.89 steps, which is equal to the runtime of the 19-step DDIM schedule. Lastly, we provide example images generated by our ADSC and fixed baselines in Figure 5. The first and last two rows are obtained by DDIM and PNDM schedulers, respectively. An initial observation reveals that, in general, ADSC resulted in higher-quality images within a smaller number of timesteps compared to using a fixed step scheduler. Moreover, the prompts given in the first and last rows required more steps to generate images of sufficient quality compared to others. This can be attributed to the fact that the objects described in these prompts inherently contain intricate details, such as the keyboard and home screen of a laptop or street names written on a sign. Conversely, cats, cars, and tables are relatively more plain objects. The commonality of the objects and their presence in the training dataset of the diffusion model can be another contributing factor.

![](images/3f6d4e739b464b179f9ac1fe37f56f1173595a187eccb4963749cf74fae7fa7e.jpg)  
Fig. 5: Text prompts from COCO and images generated by our ADSC, and schedulers fixed number of steps.

## 4.2. DiffusionDB results

We followed the same procedures described in Section 4.1 to evaluate our method on the DiffusionDB dataset. The distribution of steps exhibits differences compared to COCO experiments, as seen in Figure 6. These differences are also reflected in the avg, min, and max statistics. Overall, both adap-

![](images/466c04fb7d47676f5fea719e02524947c2be599a282d448e998b0ee7f299ecdf.jpg)  
(a) DDIM

![](images/1c0dce50bd0138692825b42d6abc693cd0ca1331b33c307aae31fbc9078183aa.jpg)  
(b) PNDM  
Fig. 6: Distribution of steps upon controlling DDIM and PNDM schedulers using our ADSC in DiffusionDB dataset.

tive and fixed scheduling approaches saw an improvement in CLIP-T scores beyond those observed in the COCO experiments. Upon coupling ADSC with DDIM, we observed that the improvement margins of CLIP-I and DINO remain significant, which are 3.2% (95.10 vs. 91.90) and 6.4% (89.36 vs. 82.96) respectively. As for the PNDM scheduler, promptadaptive control did not lead to any noticeable improvement in CLIP-I and caused the DINO score to drop by 1.59%. This suggests that ADSC is more compatible with DDIM and more consistently improves performance when coupled with it. However, it is also worth noting that ADSC did not degrade PNDM performance significantly either. Therefore, considering the improvements seen in benchmarks like COCO, it remains beneficial to employ ADSC to enhance PNDM as well. Figure 7 contains images generated based on DiffusionDB

<table><tr><td colspan="5">DDIM</td><td colspan="4">PNDM</td></tr><tr><td rowspan="2">Strategy</td><td>ADSC - ours* (18.06 steps)</td><td>fixed (12 steps)</td><td>fixed* (18 steps)</td><td>fixed (33 steps)</td><td>ADSC - ours* (24.75 steps)</td><td>fixed (16 steps)</td><td>fixed* (25 steps)</td><td>fixed (35 steps)</td></tr><tr><td>95.10</td><td>89.40</td><td>91.90</td><td>95.20</td><td>93.70</td><td>89.90</td><td>93.60</td><td>95.70</td></tr><tr><td>CLIP-I (↑) CLIP-T (↑)</td><td>33.70</td><td>33.70</td><td>33.90</td><td>34.00</td><td>33.80</td><td>33.30</td><td>33.70</td><td>33.80</td></tr><tr><td>DINO (↑)</td><td>89.36</td><td>77.01</td><td>82.96</td><td>90.90</td><td>85.50</td><td>79.02</td><td>87.09</td><td>87.21</td></tr><tr><td>Avg wallclock time (sec/img)(↓)</td><td>0.76</td><td>0.55</td><td>0.75</td><td>1.32</td><td>1.06</td><td>0.71</td><td>1.06</td><td>1.42</td></tr></table>

Table 2: Comparison between ADSC and baseline fixed-step schedulers using various metrics on the DifusionDB dataset. (\*) highlights our method and the main baselines.

prompts. The first two and last two rows are generated using DDIM and PNDM, respectively. For all four prompts, ADSC produced high-quality images, while several problems can be noticed in the images generated by fixed schedules. The first one is the misalignment with the text prompt, which can be observed in the last two rows: The 16 and 24-step generated images from the prompt "A programmer in a cyberpunk..." do not contain distinctive features of a programmer, and the 16, 24-step generated images from the prompt "A young man with black..." miss the color information of facial features described in the prompt as they are in grayscale. Although the 34-step generated image contains blue eyes, it is inconsistent with the grayscale theme. The second problem is the visual defects which can be seen in the first, second, and third rows. Finally, we expand our evaluation by comparing our method against fixed step schedules from two other schedulers, DDPM [25] and LMS [17]. We employ our ADSC combined with the DDIM scheduler for adaptive scheduling, as it yielded the best results. Table 3 presents the findings.

![](images/a0c5e51ec4f98efc1b7db99dc02a0c3ceb99a3ee7681d6d8551b679f370b3c72.jpg)  
Fig. 7: Text prompts from DiffusionDB and images generated by our ADSC, and schedulers with fixed numbers of steps.

<table><tr><td>Dataset</td><td colspan="3">DiffusionDB (50 → 18 steps)</td><td colspan="3">COCO (50 → 19 steps)</td></tr><tr><td>Metric</td><td>CLIP-I</td><td>CLIP-T</td><td>DINO</td><td>CLIP-I</td><td>CLIP-T</td><td>DINO</td></tr><tr><td>DDPM [25]</td><td>86.7</td><td>33.5</td><td>69.68</td><td>88.5</td><td>31.3</td><td>74.8</td></tr><tr><td>PNDM [16]</td><td>90.8</td><td>33.5</td><td>80.69</td><td>91.7</td><td>31.3</td><td>82.3</td></tr><tr><td>DDIM [15]</td><td>91.9</td><td>33.9</td><td>83.0</td><td>92.4</td><td>31.4</td><td>84.1</td></tr><tr><td>LMS [17]</td><td>94.1</td><td>33.7</td><td>88.2</td><td>94.8</td><td>31.3</td><td>90.5</td></tr><tr><td>DDIM + ADSC (ours)</td><td>95.1</td><td>33.7</td><td>89.4</td><td>95.7</td><td>31.4</td><td>92.4</td></tr></table>

Table 3: Comparison with various schedulers

## 5. CONCLUSION

We introduced ADSC to address the inefficiency of fixed denoising schedules in text-to-image diffusion generative models. Our method dynamically adjusts the number of denoising steps required to generate high-quality images based on the specific characteristics of each text prompt, thereby optimizing computational resources and time efficiency without compromising significant visual quality. Experiments have demonstrated the effectiveness of our approach in reducing the average number of diffusion time steps, while still preserving the visual fidelity. Importantly, our method does not necessitate additional model training. One limitation is the need for manual tuning of hyper-parameters, such as gradient thresholds, which can be time-consuming process. Future work can explore adaptive thresholding and learned convergence criteria using reinforcement learning methods.

## 6. REFERENCES

[1] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” arXiv preprint arXiv:2112.10741, 2021. 1

[2] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, vol. 1, no. 2, pp. 3, 2022. 1

[3] E. Luhman and T. Luhman, “Knowledge distillation in iterative generative models for improved sampling speed,” arXiv preprint arXiv:2101.02388, 2021. 1, 2

[4] B. Kim, H. Song, T. Castells, and S. Choi, “On architectural compression of text-to-image diffusion models,” arXiv preprint arXiv:2305.15798, 2023. 1, 2

[5] L. Li, H. Li, X. Zheng, J. Wu, X. Xiao, R. Wang, M. Zheng, X. Pan, F. Chao, and R. Ji, “Autodiffusion: Training-free optimization of time steps and architectures for automated diffusion model acceleration,” in ICCV, 2023, pp. 7105–7114. 1, 2

[6] T. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C L. Zitnick, “Microsoft coco: Common objects in context,” in ECCV. Springer, 2014, pp. 740–755. 1, 3

[7] Z. J Wang, E. Montoya, D. Munechika, H. Yang, B. Hoover, and D. H. Chau, “Diffusiondb: A largescale prompt gallery dataset for text-to-image generative models,” arXiv preprint arXiv:2210.14896, 2022. 1, 3

[8] J. Hessel, A. Holtzman, M. Forbes, R. L. Bras, and Y. Choi, “Clipscore: A reference-free evaluation metric for image captioning,” arXiv preprint arXiv:2104.08718, 2021. 1, 3

[9] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-toimage diffusion models for subject-driven generation,” in ICCV, 2023, pp. 22500–22510. 1, 3

[10] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin, “Emerging properties in self-supervised vision transformers,” in ICCV, 2021, pp. 9650–9660. 1, 3

[11] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022. 2, 3

[12] D. Epstein, A. Jabri, B. Poole, A. Efros, and A. Holynski, “Diffusion self-guidance for controllable image generation,” NeurIPS, vol. 36, 2024. 2

[13] Y. Li, H. Liu, Q. Wu, F. Mu, J. Yang, J. Gao, C. Li, and Y. J. Lee, “Gligen: Open-set grounded text-to-image generation,” in ICCV, 2023, pp. 22511–22521. 2

[14] M. Chen, I. Laina, and A. Vedaldi, “Training-free layout control with cross-attention guidance,” in WACV, 2024, pp. 5343–5353. 2

[15] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020. 2, 3, 4, 5

[16] L. Liu, Y. Ren, Z. Lin, and Z. Zhao, “Pseudo numerical methods for diffusion models on manifolds,” arXiv preprint arXiv:2202.09778, 2022. 2, 3, 4, 5

[17] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” NeurIPS, vol. 35, pp. 26565–26577, 2022. 2, 5

[18] X. Li, Y. Liu, L. Lian, H. Yang, Z. Dong, D. Kang, S. Zhang, and K. Keutzer, “Q-diffusion: Quantizing diffusion models,” in ICCV, 2023, pp. 17535–17545. 2

[19] C. Meng, R. Rombach, R. Gao, D. Kingma, S. Ermon, J. Ho, and T. Salimans, “On distillation of guided diffusion models,” in CVPR, 2023, pp. 14297–14306. 2

[20] S. Yang, Y. Chen, L. Wang, S. Liu, and Y. Chen, “Denoising diffusion step-aware models,” arXiv preprint arXiv:2310.03337, 2023. 2

[21] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022, pp. 10684–10695. 3

[22] X. Wu, Y. Hao, K. Sun, Y. Chen, F. Zhu, R. Zhao, and H. Li, “Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis,” arXiv preprint arXiv:2306.09341, 2023. 3

[23] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., “Learning transferable visual models from natural language supervision,” in ICML. PMLR, 2021, pp. 8748–8763. 3

[24] P. von Platen, S. Patil, A. Lozhkov, P. Cuenca, N. Lambert, K. Rasul, M. Davaadorj, and T. Wolf, “Diffusers: State-of-the-art diffusion models,” https:// github.com/huggingface/diffusers. 4

[25] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” NeurIPS, vol. 33, pp. 6840–6851, 2020. 5