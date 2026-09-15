# Can a Neural Encoding Model Replicate an fMRI Visualization Study?

Erfan Nasirzadeh Orang\*

Zack While<sup>†</sup>

Department of Computer Science and Information Systems Youngstown State University

## ABSTRACT

Most knowledge of graphical perception comes from behavioral studies. Understanding from a neural perspective is much more limited due in part to neuroimaging studies’ expensiveness and difficulty to conduct. In this paper, we evaluate whether Meta’s TRIBE V2 neural encoding model can recover neural contrasts from a visualization fMRI study. Specifically, we evaluate TRIBE V2 through a conceptual replication of the visualization-viewing component of a prior comparison of Bubble charts and three-dimensional Surface charts in color and grayscale. We generate TRIBE-predicted cortical responses for the original stimuli and compare the resulting contrasts with those reported in the human study. The model reproduced the direction of 11 of 14 reported cortical effects, with agreement concentrated in visual-processing regions. This agreement characterizes the model’s alignment with the prior human-generated fMRI results rather than independently confirming them. We discuss the limitations encountered when working with this model for in-silico replication and hope to encourage future work exploring this new avenue for neuroimaging studies in visualization. Supplemental materials are available at https://osf.io/8a96x/.

Index Terms: Graphical perception, fMRI replication, neural encoding models, TRIBE, visualization evaluation.

## 1 INTRODUCTION

Data encoding can noticeably affect how people interpret visualizations. Graphical perception studies have shown that different visual encodings vary in how accurately and efficiently viewers extract information from data, motivating decades of behavioral evaluation in visualization research [6, 15]. Building on that body of work, some visualization research has sought to explain such differences by grounding their work in the perceptual and cognitive processes underlying visual analysis [5, 33]. Nevertheless, empirical visual ization research has primarily examined design through the lens of user performance; while such studies can indicate which encodings are effective, they provide only indirect insight into the underlying neural processes. This additional perspective would provide a more comprehensive view of how design choices impact interpretation and insight generation that cannot be achieved solely by metrics such as accuracy and response time [2]. Studying those processes directly requires neuroimaging techniques such as fMRI, which depend on expensive facilities and controlled laboratory experiments [9].

Walden et al. [31] demonstrated the value of direct neural measurement by comparing responses to two different visual encodings of the same multidimensional dataset. Bubble charts can encode values through horizontal and vertical position, mark size, and color, representing each observation as a discrete mark that must be read individually. Three-dimensional Surface charts can encode the same values as a continuous shaded terrain in space, presenting the data as a single object that may be processed more holistically. They posited that the distinction could theoretically matter since terrain-like structure may invoke perceptual machinery that discrete symbolic marks do not, which the authors framed through evolutionary theory in their study. They found differences in both behavioral performance and fMRI activations between the two, with stronger activation for Surface charts in ventral regions of the brain; this provided neural evidence consistent with the authors evolution-based hypothesis beyond observed behavioral differences.

Recent advances in neural encoding models provide a potential alternative to arranging and funding traditional fMRI studies. Meta’s TRIBE V2 neural encoding model was trained on more than 1,000 hours of paired fMRI and multimodal stimuli (video, text, and audio) collected across 720 subjects. It predicts whole-brain surface activation without requiring subject-specific fMRI scans at inference time. However, it remains unclear whether the model generalizes to static stimuli and highly artificial images such as computer-generated data visualizations. Evaluating an alternative to direct neural measurement requires previously reported human fMRI results for comparison. Thus, we evaluated TRIBE V2 through a conceptual replication of the visualization-viewing component of Walden et al.’s study [31], investigating whether the model’s predicted activations recover the neural contrasts reported for Bubble and three-dimensional Surface charts. This aligns with work promoting replication studies [18, 34] and critiquing the validity of the field’s empirical underpinnings due to a lack of replication [17].

We used TRIBE V2 to simulate responses to the stimuli from both of their experiments, conducting the same subtractive analysis and comparing those results to the originals, grouped by brain region. We observed general directional agreement with the prior study, reproducing the direction of most reported cortical effects despite methodological compromises imposed by the model’s limitations. The primary contributions of this work are: (1) the first known evaluation of a predictive neural encoding model through a conceptual replication of an fMRI-based visualization study, (2) an exploratory analysis of the results and their implications for TRIBE V2’s current utility in visualization research, and (3) a discussion of methodological considerations and practical guidance for visualization researchers interested in using TRIBE V2.

## 2 RELATED WORK

Behavioral graphical perception. Cleveland and McGill [6] provided one of the earliest studies ranking perceptual tasks primarily by reading accuracy while also considering response time; Heer and Bostock [13] later reproduced that work at scale using crowdsourcing. Szafir expanded on the concept of evaluating participants by considering both user tasks and chart types [30]. A growing body of empirical work has diversified the ways that the field evaluates user performance and effective design, including memorability [4], interactivity [32], and trustworthiness [11].

Grounding visualization in vision science. Others argue for greater consideration of how the visual system works instead of solely focusing on behavioral measures [25, 29]. Scaife and Rogers [28] proposed the term external cognition, emphasizing the interaction between external graphical representations (e. g., visualizations) and internal cognitive representations (e. g., mental models), and Anderson [2] argued for evaluating visualizations with cognitive and physiological measures rather than performance alone. Neural responses provide a direct means of studying the brain processes engaged during visualization, yet they are uncommon in the literature, since collecting such information requires access to expensive imaging facilities and controlled laboratory studies [9].

Neural measurement of visualization. Walden et al. [31] and the earlier Safi et al. [26] placed participants in an fMRI scanner while they answered multiple-choice questions about Bubble charts and three-dimensional Surface charts, using the resulting fMRI responses to characterize how the two types of visualizations are neurally processed. Li et al. [20] also utilized an fMRI scanner to compare responses to bar charts and line charts, reporting greater involvement of ventral-stream regions associated with recognition and interpretation than dorsal-stream regions associated with spatial and action-oriented processing. Meanwhile, both Anderson [2] and Idesis et al. [16] recently reviewed methods for monitoring neural signals, with the latter considering their future use in generating data visualizations that adapt to a user’s needs in real time. Unlike these studies, which directly measured participants with fMRI, we evaluate whether a predictive neural encoding model can approximate the neural responses observed in such experiments.

Models as proxies. A growing direction of research involves tasking deep learning models such as convolutional neural networks (CNNs) [12] and large language models (LLMs) [19, 21, 23] with understanding data visualizations, often comparing their responses to human participants. The field of human computer interaction has recently begun efforts to consider its approach toward the use of LLMs to simulate participants in human subjects studies [1]; data visualization has also begun seeing studies investigating this style of synthetic participation [27]. Rather than simulating subjective judgments or task performance, we use a deep learning model to predict cortical responses to specific types of visualizations.

Replication in visualization and HCI. Extant work in visualization has provided methods of classifying replication studies based on various aspects of their design or their novelty. Hornbæk et al. [14] distinguished strict, partial, and conceptual replications according to the degree to which the original measures, manipulations, and setting were retained. Quadri and Rosen [24] complemented this with reevaluation, expansion, and specialization, sorting studies by the objective of their novel contribution. We use a conceptual replication as an evaluation setting for a new data-generating method.

## 3 METHODOLOGY

This section outlines major details regarding the study’s design.

## 3.1 Conceptual Replication as Model Evaluation

Walden et al. [31], which will be referred to as the prior study, ran two fMRI experiments comparing how people read Bubble and three-dimensional Surface charts. Participants viewed each chart alongside a multiple-choice question and answered at their own pace while neural activations were recorded. They defined the questions as being extraction (answers based on visible information shown on the chart) and integration (answers requiring reasoning about information on the chart) tasks. The authors computed a subtractive contrast between neural responses to the two chart types and reported the resulting differences as clusters, each labeled by brain region. Their first experiment used color charts encoding four data dimensions (Figure 1a and Figure 1b), with each Bubble chart showing all four while the corresponding Surface chart required showing two charts with three dimensions depicted for the same data. Following this design, we simulated responses to the same stimuli, applied the same subtractive contrast, and compared the direction of each effect with the corresponding reported cluster.

With that in mind, our objective was not to independently confirm or challenge the validity of the prior study’s human fMRI findings. We instead used its published results as an empirical reference for evaluating whether TRIBE V2 can reproduce known cortical contrasts elicited by data visualizations. The model, rather than the original findings, was therefore the primary object of evaluation. Even so, our approach still reproduced several elements of the prior study’s design, including its stimuli, chart comparison, subtractive contrast, and region-level results. Using Hornbæk et al.’s classification, we therefore characterize this work as a conceptual replication, since it investigates an existing effect using a substantially different measurement system and experimental setting [14]. Under Quadri and Rosen’s taxonomy, the study would be labeled as a re-evaluation: the novelty here lies in assessing whether a neural encoding model can recover results previously obtained through human fMRI [24].

Our results thus do not independently confirm the prior study, as the replicated comparison instead provides a benchmark for evaluating TRIBE V2’s behavior on visualization stimuli, similar to prior work that reproduced established findings to evaluate a new datacollection setting [13]. A direct replication would need to closely match the data collection conditions of the original, which is not possible due to a lack of human participants. Additionally, the behav ioral task-completion portion required responses to multiple-choice questions, which TRIBE V2 cannot provide; thus, we only replicated the visualization-viewing portion of the study, striving to otherwise align with the prior study’s intent within the model’s limitations.

## 3.2 Stimuli

We evaluated TRIBE V2 using the visualization-viewing components of both experiments from the prior study after obtaining the original stimuli from the authors [31]. Experiment 1 used colored data visual izations encoding four data dimensions, consisting of three numeric features and one categorical feature. Because a Surface encodes only three dimensions, its Surface condition presents two charts side by side that each represent one category, which we stitched into one image (Figure 1b). Out of concern for the impact of comparing two images instead of simply looking at one for the Surface condition, Experiment 2 used grayscale three-dimensional graphs, entirely removing the categorical feature from Surface . Each experiment consisted of 60 matched Bubble and Surface pairs (120 images total), with each pair accompanied by the same question. Following the advice of Benchetrit et al. [3] for working with static images, we converted each image to a 3-second silent video with the frame held constant. Images kept their native resolution (even-dimension crop only, no scaling or padding), and no audio track was added; notably, video-only input is supported by the model. Notably, this differs from the prior study by omitting the task question, as TRIBE V2 does not currently provide a straightforward mechanism for reproducing the original question-answering task. The model thus responds only to the visualization stimulus.

We ran TRIBE V2 in its default unseen-subject mode, which predicts a single population-level response on the fsaverage5 cortical Surface of the brain (20,484 vertices), generating samples at a rate of one per second (1 Hz). The model’s predictions already incorporate the roughly 5-second hemodynamic delay that requires additional exposure to get the strongest Blood-Oxygenation-Level-Dependent (BOLD) response. Although TRIBE V2 emits one prediction per second, we only report responses to the first timestep, following the single-timestep truncation described by Benchetrit et al. [3]. To check for robustness, we arrived at similar results for both the color and grayscale models when truncating at time steps t = {0,1,2}, which are provided in the supplemental materials.

![](images/71b05ae6d226b31e12754b31da9e26456182f3d278e06423d9af1a6de9e9c6e6.jpg)  
(a) Bubble , Color

![](images/c0d5f7894ef014ca79941e0bc068086974c27ee15a0fbf7284c18e5deacdc066.jpg)  
(b) Surface , Color (Blue, Red), Stitched

![](images/ee203a37b5931e5b145ce80cf27a97fc25f2ed5cba479bb36eea41e5fb8d5d1a.jpg)  
(c) Bubble , Grayscale

![](images/2af6a4aad746edd058e8e40253c8e34b53271da95ef8f412e9b3bf5644da19a3.jpg)  
(d) Surface , Grayscale  
Figure 1: Example stimuli from Experiment 1 (a-b) and Experiment 2 (c-d), which were provided by the authors of the prior study [31].

## 3.3 Procedure

After converting the stimuli to three-second static videos with no audio, we processed each one with the model, resulting in a set of vertices representing activations of an fsaverage5<sup>1</sup> cortical surface of the brain for each stimulus, stored as a NumPy array for analysis. For each experiment’s 60 matched pairs, comprising 120 images per experiment (color and grayscale), inference (prediction generation) took approximately one hour on a 32 GB NVIDIA Tesla V100 SXM2 GPU [22]. The model’s memory (VRAM) usage peaked at approximately 1.07 GB and remained near that level throughout.

## 4 DATA ANALYSIS AND RESULTS

Here we summarize our method of analysis as well as the results.

## 4.1 Analysis Method

For each pair, we computed a cognitive-subtraction contrast, Surface minus Bubble , at every vertex. This was the same contrast Walden et al. ran and the standard approach in prior fMRI graph studies [20, 26]. We averaged across the 60 pairs and assigned each vertex to a region using the Destrieux atlas [8], excluding noncortical labels. Because the model is deterministic, the 60 pairs are 60 distinct stimuli instead of independent measurements of one effect. Averaging across them reduced variation from individual images (differences in visual density or value range) rather than noise and isolated the part of the contrast attributed to graph type.

For each of the prior study’s reported clusters, we checked whether the predicted contrast had the same direction, i. e., whether the same chart type (Surface or Bubble ) had stronger activations. Comparison was restricted to cortical clusters, since TRIBE V2’s surface output does not cover the cerebellar and subcortical clusters that the prior study also reported. Furthermore, as an additional sanity check for these results, we repeated the analysis after removing each pair’s whole-brain mean, which is available in the supplemental materials. This step removed global differences in predicted activation magnitude and tested whether the observed effects persisted as regional deviations from the brain-wide average. Lastly, to measure uncertainty, we applied BCa bootstrapping [10] to the contrasts from the stimulus pairs, computing a 95% confidence interval (CI) of the mean difference across the stimuli.

## 4.2 Directional Agreement as a Coarse Comparison

We note here that direct numeric comparison between our results and the prior study is not well-defined. The prior study’s z-scores come from variance across 20 human participants, each with measurement and physiological noise. TRIBE V2 is deterministic and, in unseen-subject mode, produces a single population-level prediction [7], so there is no measurement variance to standardize against. We therefore report each contrast’s direction (i. e., its sign) and raw magnitude, evaluating directional agreement only within the brain regions highlighted by the prior study. Directional agreement indi cates whether TRIBE V2 predicts the same chart type (Surface or Bubble ) to produce the stronger response within a reported region. It thus provides a coarse measure of whether the model preserves the relative ordering of the two conditions observed in the prior results. This agreement provides evidence about TRIBE V2’s correspondence with the prior results. It does not independently confirm the existence or reliability of the original human effects, nor does it demonstrate that the model recaptured the effects’ magnitude, spatial distribution, statistical reliability, or functional significance.

## 4.3 Color Stimuli (Experiment 1)

TRIBE V2 matched 7 of the prior study’s 9 cortical clusters (Table 1). Both fusiform clusters, left superior lateral occipital, and all four Bubble -dominant<sup>2</sup> clusters matched. On the other hand, the left inferior lateral occipital and left superior parietal lobule disagreed with the prior study’s direction. Some predicted differences were close to zero, including those in the left middle frontal gyrus (−0.011) and right anterior cingulate (−0.017). The largest absolute mean contrast was 0.087 in the left occipital pole.

Table 1: Comparison of the model’s predicted contrasts with the prior study’s results for Experiment 1 (color). Positive values indicate higher predicted fMRI response for Surface . A ✓ symbol indicates alignment with the corresponding result of the prior study, while a ✗ symbol indicates disagreement. L stands for left and R stands for right. In the TRIBE V2 repository demo, responses to a 52- second within-distribution color video with audio had a middle 95% range of approximately [−0.35,0.54] across time steps (µ = 0.07).

<table><tr><td>Cluster (Walden)</td><td>Prior Result</td><td>Ours (Mean)</td><td>Ours (95% CI)</td></tr><tr><td>L Superior Lateral Occipital</td><td>Surface</td><td>+0.056√</td><td>+0.044,+0.071</td></tr><tr><td>R Temporal Occipital Fusiform</td><td>Surface 4</td><td>+0.041√</td><td>+0.033,+0.049]</td></tr><tr><td>L Temporal Occipital Fusiform</td><td>Surface 4</td><td>+0.019√</td><td>[+0.011,+0.027]</td></tr><tr><td>L Inferior Lateral Occipital</td><td>Surface</td><td>-0.030X</td><td>[−0.042, −0.016]</td></tr><tr><td>L Superior Parietal Lobule</td><td>Bubble</td><td>+0.038X</td><td>[+0.031,+0.047]</td></tr><tr><td>L Middle Frontal Gyrus</td><td>Bubble</td><td>-0.011√</td><td>[−0.012,−0.010]</td></tr><tr><td>R Anterior Cingulate</td><td>Bubble</td><td>-0.017√</td><td>−0.020,−0.015]</td></tr><tr><td>R Middle Precentral Gyrus</td><td>Bubble</td><td>-0.021√</td><td>[−0.023, −0.020]</td></tr><tr><td>L Occipital Pole</td><td>Bubble</td><td>-0.087√</td><td>[−0.099, −0.072]</td></tr></table>

## 4.4 Grayscale Stimuli (Experiment 2)

TRIBE V2 matched the direction of 4 of the prior study’s 5 cortical clusters (Table 2): left superior lateral occipital (+0.142), right fusiform (+0.095), right cuneus (−0.049), and left middle frontal gyrus matched weakly (+0.012). The right superior frontal did not match the prior study (−0.011). Here two clusters were close to zero (left middle frontal gyrus and right superior frontal gyrus) and the largest absolute value was the left superior lateral occipital (0.142).

## 4.5 Consistency Across Experiments

Left superior parietal was Surface -dominant in both experiments and both analyses. It is not among the prior study’s Experiment 2 clusters, but the corresponding parcel in our grayscale data was also Surface -dominant (+0.146), matching the color result (+0.038) and running opposite of the prior study’s Bubble direction. In total, TRIBE V2 matched the direction of 11 of 14 cortical clusters.

Table 2: Grayscale (Experiment 2). Predicted contrast at the prior study’s reported cortical clusters. Positive values indicate Surface -dominance. A ✓ shows matching the results of the prior study, while ✗ shows disagreement. L stands for left and R stands for right. In the TRIBE V2 repository demo, responses to a 52- second within-distribution color video with audio had a middle 95% range of approximately [−0.35, 0.54] across time steps (µ = 0.07).
<table><tr><td>Cluster (Walden)</td><td>Prior Result</td><td>Ours (Mean)</td><td>Ours (95% CI)</td></tr><tr><td>L Superior Lateral Occipital</td><td>Surface 4</td><td>+0.142√</td><td>+0.135,+0.149</td></tr><tr><td>R Fusiform Cortex</td><td>Surface 4</td><td>+0.095√</td><td>+0.089,+0.100]</td></tr><tr><td>L Middle Frontal Gyrus</td><td>Surface 4</td><td>+0.012√</td><td>[+0.011,+0.013]</td></tr><tr><td>R Superior Frontal Gyrus</td><td>Surface 4</td><td>-0.011X</td><td>-0.013,-0.010]</td></tr><tr><td>R Cuneus</td><td>Bubble</td><td>-0.049√</td><td>-0.052,−0.045</td></tr></table>

## 5 DISCUSSION

Our study evaluated TRIBE V2 through a conceptual replication of selected components of Walden et al.’s experiments [31]; it did not reproduce the complete human experimental procedure (see Section 3.1). Participants in their study viewed each stimulus while answering a multiple-choice question based on an extraction or integration task, whereas TRIBE V2 only received a visualization with no task. Our process thus lacked the prior study’s task demands, and the results must be interpreted with this distinction in mind.

Representing task instructions is an open question. TRIBE V2 is not designed to extract written questions from images and instead processes language through audio-derived text [7]. However, doing so would instead be simulating a participant hearing a question as opposed to reading it. The proper way to provide questions to the model is still an open question, and as of now the recommendation of TRIBE V2’s authors is for audio-free videos [3]. More work is needed to investigate how task instructions can be provided to neural encoding models while still aligning with human studies.

Given only visual stimuli, TRIBE V2’s agreement was concentrated in perceptual regions. The clusters that agreed with the prior study were primarily visual and occipital regions, including middle and lateral occipital cortex, lingual gyrus, fusiform, and cuneus. In contrast, the frontal regions examined in the prior study produced some of the smallest differences in our analysis. Because the prior study required participants to view the visualizations and answer questions, these regions’ lower agreement may partially reflect task demands such as working memory, response selection, or comparison processes that were absent from our image-only stimuli. We caution that our results should not be interpreted as demonstrating that TRIBE V2 reproduces human visualization perception broadly. The model aligned with the direction of several reported contrasts, suggesting that it may capture some perceptual distinctions between the chart types reflected in the prior study.

TRIBE V2 shows consistent behavior across varying stimuli. The perceptual match held across a change in rendering (color vs. grayscale), dimensionality (data with four features vs. data with three), and stimulus style (single image vs. stitched images). The agreement persisted across both experiments despite differences in color, data dimensionality, and presentation, providing some evidence that the observed agreement is robust to such changes.

Not all disagreements are explained by task omission. One notable disagreement between our results and the prior study was the left superior parietal region, which was consistently Surface -dominant across both experiments despite showing the opposite direction in the prior study. Because parietal regions are often associated with visuospatial processing, this is not as readily explained by the absence of the question-answering task. It may, instead, reflect a limitation of the model for visualization stimuli or sensitivity to aspects of visual inputs that differs from humans.

TRIBE V2 predictions introduce methodological limitations. The model’s predictions are deterministic and generate a single population response, compared to the prior study’s use of z-scores to illustrate response magnitude. This motivated our focus on the direction of effect rather than specific numerical outcomes in the analysis. As a result, our evaluation should instead be viewed as a qualitative comparison of agreement, as matching effect direction alone does not demonstrate that the underlying neural responses are identical to those reported in the prior study. Mapping the prior study’s voxel-based clusters onto the parcels used by TRIBE V2 required name-based matching, introducing the possibility of small misalignments due to differing representations. The stitched color Surface charts differed from Bubble stimuli in aspect ratio, which the model’s fixed input resizing compresses unequally. Static, silent images are out-of-distribution for a video-trained model, so as of now visualization stimuli may lead to less consistent results. This may have contributed to the relatively small differences observed in some regions (e. g., right superior frontal gyrus). Notably, because the model only predicts neural activations on the cortical surface, it cannot capture cerebellar or subcortical responses.

Visualization researchers should work with TRIBE V2’s strengths. Given the lack of a natural mechanism in the model for providing task instructions with the prior study’s stimuli as-is, our replication focused on comparing perceptual effects and observed the strongest agreement in visual and occipital regions. As such, we recommend using TRIBE V2 as a supplementary tool for comparing perceptual impacts of visualization design choices. Additionally, researchers should be prepared and able to convert their visualizations into static video clips. From our study, it is unclear how well the model is suited for non-static visualization clips (e. g., videos of interactive, animated visualizations), and this is an open question for future work to consider. Because the model’s output values represent a population response rather than participant-level measurements, comparisons to real-world fMRI studies should generally be interpreted qualitatively. However, the outputs can still be compared numerically against other predictions generated by the model. One must also consider TRIBE V2’s computational requirements. Based on our observed peak usage (∼1.07 GB), inference appears feasible on GPUs with substantially less memory than the 32 GB used in our experiments. Furthermore, since the model only predicts activations on the cortical surface, it is most appropriate for studying cortical processes. Those studying cerebellar or subcortical activity will likely require the use of more traditional methods.

## 6 FUTURE WORK AND CONCLUSION

Models like this could provide a low-cost way to screen visualization designs against a neural benchmark, although they currently appear better suited to modeling perceptual responses than task engagement. Future work could test different methods of adding the question text to see whether the frontal clusters return. Furthermore, researchers could replicate additional fMRI studies such as one by Li et al. [20].

In this paper, we evaluated TRIBE V2 through a conceptual replication of the visualization-viewing component of Walden et al.’s fMRI experiments [31]. The model reproduced the effect direction of 11 of 14 cortical clusters from Walden et al.’s two fMRI experiments [31]. While these results do not independently confirm the original findings, they suggest that neural encoding models may be capable of recovering some perceptual distinctions observed in visualization neuroimaging studies. However, additional work is needed to fully understand their utility. While in its infancy as a research tool, this approach shows considerable promise and warrants some consideration from the visualization field as it continues to develop.

## REFERENCES

[1] W. Agnew, S. Kapania, H. Schroeder, M. Aubin Le Quer´ e, S. E. Fox,´ and H. Heidari. Workshop on developing standards and documentation for llm use as simulated research participants. In Proceedings of the Extended Abstracts ofthe 2026 CHI Conference on Human Factors in Computing Systems, pp. 1–4, 2026. 2

[2] E. W. Anderson. Evaluating visualization using cognitive measures. In Proceedings of the 2012 BELIV Workshop: Beyond Time and Errors - Novel Evaluation Methodsfor Visualization, BELIV ’12. Association for Computing Machinery, New York, NY, USA, 2012. doi: 10.1145/ 2442576.2442581 1, 2

[3] Y. Benchetrit, M. Careil, S. Dahan, H. Banville, S. d’Ascoli, and J.-R. King. Boosting brain-to-image decoding with TRIBE v2 data augmentation. arXiv preprint arXiv:2606.06345, 2026. 2, 4

[4] M. A. Borkin, Z. Bylinskii, N. W. Kim, C. M. Bainbridge, C. S. Yeh, D. Borkin, H. Pfister, and A. Oliva. Beyond memorability: Visualization recognition and recall. IEEE Transactions on Visualization and Computer Graphics, 22(1):519–528, 2016. doi: 10.1109/TVCG.2015. 2467732 1

[5] S. Card, J. Mackinlay, and B. Shneiderman. Readings in Information Visualization: Using Vision to Think. Interactive Technologies. Elsevier Science, 1999. 1

[6] W. S. Cleveland and R. McGill. Graphical perception: Theory, experi mentation, and application to the development of graphical methods. Journal of the American Statistical Association, 79(387):531–554, 1984. 1

[7] S. d’Ascoli, J. Rapin, Y. Benchetrit, T. Brooks, K. Begany, J. Raugel, H. Banville, and J.-R. King. A foundation model of vision, audition, and language for in-silico neuroscience. arXiv preprint arXiv:2605.04326, 2026. 3, 4

[8] C. Destrieux, B. Fischl, A. Dale, and E. Halgren. Automatic parcellation of human cortical gyri and sulci using standard anatomical nomenclature. NeuroImage, 53(1):1–15, 2010. 3

[9] A. Dimoka. How to conduct a functional magnetic resonance (fmri) study in social science research1. Management Information Systems Quarterly, 36(3):811–840, 09 2012. doi: 10.2307/41703482 1, 2

[10] B. Efron. Better bootstrap confidence intervals. Journal ofthe American statistical Association, 82(397):171–185, 1987. 3

[11] H. Elhamdadi, A. Stefkovics, J. Beyer, E. Moerth, H. Pfister, C. X. Bearfield, and C. Nobre. Vistrust: a multidimensional framework and empirical study of trust in data visualizations. IEEE Transactions on Visualization and Computer Graphics, 30(1):348–358, 2024. doi: 10. 1109/TVCG.2023.3326579 1

[12] D. Haehn, J. Tompkin, and H. Pfister. Evaluating ‘graphical perception with cnns. IEEE Transactions on Visualization and Computer Graphics, 25(1):641–650, 2019. doi: 10.1109/TVCG.2018.2865138 2

[13] J. Heer and M. Bostock. Crowdsourcing graphical perception: Using mechanical turk to assess visualization design. In Proceedings ofthe ACM CHI Conference on Human Factors in Computing Systems, pp. 203–212, 2010. 1, 2

[14] K. Hornbæk, S. S. Sander, J. A. Bargas-Avila, and J. Grue Simonsen. Is once enough? on the extent and content of replications in human computer interaction. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, CHI ’14, p. 3523–3532. Asso ciation for Computing Machinery, New York, NY, USA, 2014. doi: 10. 1145/2556288.2557004 2

[15] S. Hu, O. Jiang, J. Riedmiller, and C. X. Bearfield. Motion-based visual encoding can improve performance on perceptual tasks with dynamic time series. IEEE Transactions on Visualization and Computer Graphics, 31(1):163–173, 2025. doi: 10.1109/TVCG.2024.3456405 1

[16] S. Idesis, M. Kassinopoulos, M. Barreda-Angeles, L. A. Leiva, and<sup>´</sup> I. Arapakis. Cognitive state monitoring for neuroadaptive information visualization. Frontiers in Human Neuroscience, 2026. 2

[17] R. Kosara. An empire built on sand: Reexamining what we think we know about visualization. In Proceedings of the Sixth Workshop on

Beyond Time and Errors on Novel Evaluation Methodsfor Visualization, BELIV ’16, p. 162–168. Association for Computing Machinery, New York, NY, USA, 2016. doi: 10.1145/2993901.2993909 1

[18] R. Kosara and S. Haroz. Skipping the replication crisis in visualization: Threats to study validity and how to address them, 2018. doi: 10. 31219/osf.io/f8qey 1

[19] H. Li, G. Appleby, and A. Suh. Large language models as visual data analysis assistants. IEEE Transactions on Visualization and Computer Graphics, 30(1), 2024. 2

[20] M. Li, S. Lu, J. Wang, L. Ma, M. Zhang, and N. Zhong. Ventral stream plays an important role in statistical graph comprehension: An fMRI study. In Brain Informatics and Health, pp. 12–20, 2014. 2, 3, 4

[21] R. H. Nguyen, K. Maeda, M. Geshvadi, and D. Haehn. Evaluating ‘graphical perception’ with multimodal llms. In 2025 IEEE 18th Pacific Visualization Conference (PacificVis), pp. 290–295, 2025. doi: 10. 1109/PacificVis64226.2025.00035 2

[22] Ohio Supercomputer Center. Ohio supercomputer center, 1987. 3, 5

[23] P. Poonam, P.-P. Vazquez, and T. Ropinski. Evaluating graphi-´ cal perception capabilities of vision transformers. arXiv preprint arXiv:2602.18178, 2026. 2

[24] G. J. Quadri and P. Rosen. You can’t publish replication studies (and how to anyways). arXiv preprint arXiv:1908.08893, 2019. 2

[25] R. A. Rensink. On the prospects for a science of visualization. In Handbook of Human Centric Visualization, pp. 147–175. Springer, 2014. 1

[26] R. Safi, E. Walden, G. Cogo, D. Lucus, and E. Moradiabadi. An evolutionary explanation of graph comprehension using fMRI. 2015. 2, 3

[27] J. Satkunarajan, M. Abdelaal, S. Koch, K. Kurzhals, and D. Weiskopf. Can llms simulate target users in visualization case studies? Computer Graphics Forum, n/a(n/a):e70446. doi: 10.1111/cgf.70446 2

[28] M. Scaife and Y. Rogers. External cognition: How do graphical representations work? International Journal ofHuman-Computer Studies, 45(2):185–213, 1996. 1

[29] K. Schonborn and L. Besan¨ c¸on. Toward a cognitive-science grounding for visualization evaluation. In IEEE VIS Workshop on Visualization Education, Literacy, and Activities (EduVis), 2024. 1

[30] D. A. Szafir. The good, the bad, and the biased: Five ways visualizations can mislead (and how to fix them). Interactions, 25(4):26–33, 2018. 1

[31] E. Walden, G. S. Cogo, D. J. Lucus, E. Moradiabadi, and R. Safi. Neural correlates of multidimensional visualizations: An fMRI comparison of bubble and three-dimensional surface graphs using evolutionary theory. MIS Quarterly, 42(4):1097–1116, 2018. 1, 2, 3, 4

[32] E. Wall, A. Arcalgud, K. Gupta, and A. Jo. A markov model of users’ interactive behavior in scatterplots. In 2019 IEEE Visualization Conference (VIS), pp. 81–85, 2019. doi: 10.1109/VISUAL.2019.8933779 1

[33] C. Ware. Information Visualization: Perception For Design. Elsevier, 2012. doi: 10.1016/C2009-0-62432-6 1

[34] M. L. Wilson, E. H. Chi, S. Reeves, and D. Coyle. Replichi: the workshop ii. In CHI ’14 Extended Abstracts on Human Factors in Computing Systems, CHI EA ’14, p. 33–36. Association for Computing Machinery, New York, NY, USA, 2014. doi: 10.1145/2559206. 2559233 1

## ACKNOWLEDGMENTS

This work used high performance computing (HPC) resources provided by the Ohio Supercomputer Center [22]. Additionally, the authors acknowledge that Claude’s Opus 4.8 and Opus 5 large language models (LLMs) were used during the process of adjusting the TRIBE V2 model’s provided code examples to run on our specific dataset and analyze the results for our two experiments. We then carefully verified that the code worked as intended, and we take full responsibility for any results that are derived from such code.