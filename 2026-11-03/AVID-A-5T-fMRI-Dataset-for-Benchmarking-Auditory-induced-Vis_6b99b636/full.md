# AVID: A 5T fMRI Dataset for Benchmarking Auditory-induced Visual Mental Imagery Decoding

Shiqi Shen<sup>1,2,3,∗</sup> Shurui Li<sup>1,∗</sup> Xilin Zhang<sup>2,†</sup> Yuanning Li<sup>1,3,†</sup>

<sup>∗</sup>Equal contribution <sup>†</sup>Corresponding authors

<sup>1</sup>ShanghaiTech University <sup>2</sup>South China Normal University <sup>3</sup>XS Vision 2023010281@m.scnu.edu.cn, lishr2022@shanghaitech.edu.cn xlzhang@m.scnu.edu.cn, liyn2@shanghaitech.edu.cn

## Abstract

Visual mental imagery (VMI) decoding uses brain activity to reconstruct internally generated visual scenes, providing a potential pathway for externalizing internal experiences and developing neuroprosthetic communication interfaces. However, progress in this field is limited by the lack of dedicated, training-scale VMI datasets. Current VMI paradigms often rely on visual prompts or short-term memory images, introducing visual and mnemonic confounds that make it difficult to disentangle neural activity associated with endogenous image construction from signals driven by prior or concurrent visual input. Furthermore, existing VMI datasets offer limited descriptive control over individual visual attributes, making it difficult to systematically assess fine grained visual details. To address these gaps, we introduce AVID, the first dedicated training-scale 5.0 Tesla functional magnetic resonance imaging (fMRI) resource for auditory-induced VMI decoding. Acquired with high-field 5T fMRI, AVID comprises approximately 67 hours of dense recordings from nine participants. The dataset is uniquely structured into two complementary splits: a scene-level complex split using naturalistic Mandarin descriptions and a controlled simple split with explicitly specified visual attributes. We further formalize a standardized evaluation protocol that decouples source-image consistency from semantic alignment, allowing for a more nuanced assessment of imagery reconstruction. Baseline qualitative and quantitative results show that current reconstruction pipelines remain limited on AVID, although the most successful reconstructions preserve recognizable scene level semantics from spoken descriptions. By combining dedicated training-scale neural recordings and standardized evaluation resources, AVID provides both a data foundation for imagery-specific model development and a benchmark for evaluating VMI decoding with substantially reduced sensory contamination.

## 1 Introduction

Visual mental imagery (VMI) refers to the ability to internally construct visual scenes in the absence of sensory input. This capability supports core cognitive functions such as planning, memory, and language grounding. Decoding the neural representations underlying VMI offers a potential pathway toward externalizing internal visual experiences, with implications for neuroprosthetic systems and communication interfaces.

Despite recent advances in neural decoding and brain-to-image reconstruction [Takagi and Nishimoto, 2023, Ozcelik and VanRullen, 2023, Scotti et al., 2023, Lu et al., 2023, Scotti et al., 2024, Wang et al., 2024], most progress has been driven by stimulus-based visual perception. In contrast, reconstructing imagined content remains substantially more challenging [Kneeland et al., 2025]. Mental imagery is not time-locked to an external stimulus and typically produces weaker and more variable neural responses, with substantial inter-subject variability in imagery strategies and vividness. Moreover, VMI is an endogenous process that can be elicited through non-visual modalities such as language or auditory input, further complicating both modeling and evaluation.

Existing VMI studies remain limited in both scale and experimental design. Many paradigms rely on memorized images or visually presented prompts [Miyawaki et al., 2008, Dijkstra et al., 2017, Naselaris et al., 2015, Shen et al., 2019], introducing confounds such as residual perceptual activity and short-term or working-memory replay. Although some work has explored auditory cues [Ganis et al., 2004, Senden et al., 2019], these approaches are typically restricted to isolated words or low-dimensional categories and do not capture the compositional structure of natural scenes. As a result, current resources provide limited traction for studying scene-level mental imagery under reduced sensory contamination.

More fundamentally, evaluating VMI decoding raises challenges that are not present in perceptionbased reconstruction. Unlike perception, imagined content does not have a unique pixel-level ground truth, making direct reconstruction evaluation ill-posed. Standard evaluation protocols, which assume a well-defined visual target, therefore introduce a mismatch between what is measured and what the task requires. In addition, existing benchmark designs often conflate perception, memory, and imagery, making it difficult to isolate endogenous image construction from residual sensory or mnemonic processes.

We therefore treat auditory-induced VMI decoding explicitly as an evaluation problem rather than only a reconstruction problem. To this end, we introduce a visual-free, auditory-induced paradigm and collect the Auditory-induced Visual Imagery Dataset (AVID), a 5.0 Tesla fMRI dataset acquired under two complementary stimulus regimes. The complex split uses natural-scene descriptions to probe scene-level mental imagery, while the simple split uses controlled descriptions with explicitly specified attributes such as identity, color, count, alphanumeric content, and position. Together, these splits make it possible to distinguish variation due to open-ended scene ambiguity from incomplete recovery of verbally specified information.

We further define a standardized evaluation protocol that jointly considers source image consistency and semantic alignment with the spoken description. The protocol evaluates whether reconstructions remain anchored to the source image used to derive the spoken description, while also capturing the verbally specified content conveyed to the participant. We report baseline results using four representative visual decoding pipelines, LDM [Takagi and Nishimoto, 2023], MindEye1 [Scotti et al., 2023], MindEye2 [Scotti et al., 2024], and MindBridge [Wang et al., 2024], treating them as probes of the benchmark rather than as claims of model novelty.

Our contributions are as follows.

• We introduce a visual-free, auditory-induced VMI paradigm designed to reduce visual contamination and short-term or working-memory replay confounds.

• We release AVID, the first training-scale 5T VMI fMRI dataset with a scene-level complex split and a controlled simple split with explicitly specified attributes.

• We define a standardized evaluation protocol for auditory-induced VMI decoding and provide baseline evaluation results under this protocol.

## 2 Related work

## 2.1 fMRI image reconstruction from visual perception

The development of deep learning in computer vision [Krizhevsky et al., 2012, Dosovitskiy et al., 2020, Liang et al., 2021, Wu et al., 2025] has changed the scope of neural decoding. Earlier fMRI decoding methods relied mainly on linear mappings, sparse priors, or Bayesian estimation [Kay et al., 2008, Naselaris et al., 2009, Miyawaki et al., 2008, Kamitani and Tong, 2005]. These methods established important links between brain activity and low-level visual features or discrete semantic categories, but they were not intended to reconstruct the texture, layout, and spatial structure of natural images.

Recent fMRI based image reconstruction has been driven largely by visual perception datasets such as NSD [Allen et al., 2022] and by decoders that map brain activity into the latent spaces of modern generative models [Takagi and Nishimoto, 2023, Scotti et al., 2023, 2024]. These methods provide strong baselines and useful starting points for our benchmark, but they also highlight a key distinction. In visual perception, the reconstruction target is an externally presented image. In auditory induced mental imagery, the target is internally generated and has no unique pixel level reference. This shift changes not only the decoding problem but also the assumptions under which reconstruction quality is measured.

Cognitive neuroscience also cautions against treating mental imagery as a direct replay of perception [Wadia et al., 2026, Pearson et al., 2015]. VMI recruits a distributed network that partly overlaps with perception, while differing in spatial precision, response variability, and the relative involvement of early visual and higher order regions [Pearson, 2019, Spagna et al., 2021, Li et al., 2024]. These differences are important for benchmark design. If imagery responses are weaker, less spatially stable, and more strongly mediated by semantic content than perception responses, then image similarity alone may fail to capture meaningful recovery of imagined content.

## 2.2 Mental imagery benchmarks

Existing VMI datasets are often constrained by the way imagery is elicited. Memory-based paradigms require participants to view or memorize specific images before scanning [Miyawaki et al., 2008, Dijkstra et al., 2017, Naselaris et al., 2015, Kneeland et al., 2025]. Such designs are valuable for studying the overlap between perception, memory, and imagery, but they make it difficult to separate endogenous image construction from residual visual traces or short-term/working-memory replay. Text-cued paradigms reduce reliance on memorized images, but on-screen instructions can still introduce visual input during the task [Hayashi et al., 2014, Anderson et al., 2015, Albers et al., 2013]. These constraints motivate benchmark designs in which imagery is specified without concurrent visual presentation.

The closest existing benchmark is NSD-Imagery, which studies mental imagery in NSD participants using a relatively small imagery sample built on the much larger NSD resource [Kneeland et al., 2025]. This setting is well suited for testing how decoders trained on visual perception data transfer to imagery, including cases cued by visually presented words or text. Its results also suggest that such transfer remains limited.

## 3 Acquisition Paradigm and Dataset Construction

## 3.1 Overview

AVID [Shen et al., 2026] was designed to study visual mental imagery elicited without concurrent visual input. Its acquisition protocol therefore focuses on two requirements. First, the imagery cue should specify sufficiently rich visual content while avoiding visual presentation during the imagery period. Second, the resulting fMRI recordings should support controlled analyses of both natural scene imagery and explicitly specified visual attributes. To meet these requirements, we used spoken descriptions, eyes closed imagery, dense within participant sampling, and two complementary stimulus splits.

## 3.2 Participants

Nine healthy adults aged 18 to 25 were recruited from the student population. All participants were healthy native Mandarin speakers with normal or corrected-to-normal vision and normal hearing. The study was approved by the Institutional Review Board of ShanghaiTech University (IRB protocol number: Q2023-007), and all participants provided written informed consent in accordance with institutional guidelines. Participants were also informed of the long-term nature of the study prior to enrollment and were naive to the specific experimental hypotheses. Participants were financially compensated at a rate of 100 RMB per hour. They were informed of their right to withdraw from the study at any time without penalty and were also made aware that their data would be used for scientific research and, following anonymization, would be shared publicly.

![](images/b803073a671ef47e905fb92b5f20d097d6d2313c9ac65462a5d5b43f73bd8625.jpg)  
Figure 1: Overview of our benchmark. (A) Source images are converted into spoken Mandarin descriptions, which participants hear during eyes-closed imagery in the 5T scanner. The fMRI data are preprocessed to estimate trial-wise responses and define ROI-based neural inputs for decoding. (B) AVID includes approximately 67 hours of recordings from 9 participants across complex scene-level and controlled attribute splits, using a trial structure with an auditory cue, imagery period, vividness rating, and reset interval. (C) Decoders are evaluated on fixed AVID splits, with reconstructions scored by source-image consistency and spoken-description alignment.

Participants were pre-screened using the Vividness of Visual Imagery Questionnaire (VVIQ) [Marks, 1973], with scores ranging from 52 to 81, indicating relatively vivid self-reported imagery. All participants completed the main mental imagery task and a functional localization task described in Section 3.5. The simple split was collected from seven of these participants under the same recruitment criteria and task instructions. More details are provided in Appendix B.2.

## 3.3 Visual-free auditory-induced paradigm

In AVID, participants performed an event-related mental imagery task (Figure 1), in which each trial began with a spoken description. The auditory cues are spoken Mandarin descriptions that specify visual content without presenting visual stimuli. The description specified the content to be imagined without presenting a visual prompt during the imagery period. Participants kept their eyes closed and formed a static visual scene from an observer perspective, rather than imagining themselves acting within the scene. This instruction was intended to reduce variability associated with self representation, motor planning, and egocentric spatial updating.

Each trial lasted 12 s. The first 8 s served as the imagery window. The spoken description was presented at trial onset, and participants maintained the elicited image throughout this period. A 250 ms auditory beep marked the transition to a 4 s reset interval before the next trial. Participants rated imagery vividness on a 4 point scale using an MR compatible button box. Additional implementation details on run and session organization are provided in Appendix B.2.

## 3.4 Stimulus splits

Complex split. The complex split uses scene-level spoken descriptions derived from processed image–caption stimulus pairs from the Caption Scene Dataset [Li et al., 2026], which were originally based on COCO-CN, a Mandarin image caption collection [Li et al., 2019]. Captions were synthesized into 4 to 6 s audio clips using Google Text-to-Speech, with each caption containing 12 to 20 Mandarin characters. We curated 800 unique complex stimuli, and each stimulus was presented twice to each participant in pseudo-randomized order. More details about stimuli preparation are provided in Appendix B.1

Table 1: Summary of AVID dataset statistics.
<table><tr><td>Split</td><td>Participants</td><td>Stimuli</td><td>Total trials</td><td>Task time (h)</td><td>Repetitions</td></tr><tr><td>Complex</td><td>9</td><td>800</td><td>14,400</td><td>48.0</td><td>2</td></tr><tr><td>Simple</td><td>7</td><td>400</td><td>5,600</td><td>18.7</td><td>2</td></tr><tr><td>Total</td><td>9</td><td>1,200</td><td>20,000</td><td>66.7</td><td>2</td></tr></table>

Simple split. The simple split contains 400 in-house image-caption stimulus pairs designed to make a small set of visual attributes explicit, including object identity, alphanumeric content, color, count, and relative position. The spoken descriptions describe sparse configurations on a neutral background, reducing semantic ambiguity while preserving the same spoken description imagery procedure used in the complex split. Each stimulus was repeated twice for each participant.

The two splits are complementary rather than hierarchical. The complex split probes whether decoders can recover scene level semantic structure from natural descriptions, where multiple visual realizations are plausible. The simple split narrows this ambiguity by explicitly specifying a small number of inspectable attributes. It therefore provides a controlled setting for diagnosing whether reconstruction errors reflect failures to recover information that was unambiguously given to the participant. Table 1 summarizes the stimulus and trial counts for the two splits.

## 3.5 Functional localizer and ROI definitions

The auditory-induced imagery paradigm engages speech perception, linguistic comprehension, and mental image construction within the same task. AVID therefore includes a functional localizer to dissociate these components and define corresponding ROI sets for benchmark inputs. The localizer follows the logic of individual-subject language localizers [Fedorenko et al., 2024] and includes three auditory conditions: Mandarin sentences, non-word sequences, and non-linguistic acoustic controls.

AVID provides ROI definitions at three levels. A noise-ceiling analysis [Allen et al., 2022] over repeated mental imagery trials estimates response reliability and identifies task-responsive vertices, denoted as $R O I _ { m }$ . The contrast between non-word sequences and non-linguistic acoustic controls identifies speech-responsive vertices, denoted as $R O I _ { s }$ . The contrast between sentences and nonword sequences identifies language-selective vertices, denoted as $R O I _ { l }$ . These masks make the neural input definition part of the benchmark specification, allowing all baselines to be evaluated on the same pre-defined vertex sets rather than selecting inputs in a model-dependent way.

## 3.6 MRI data acquisition and preprocessing

Functional MRI data were acquired on a 5T MRI scanner. Functional images were collected with TR = 1.5 s, 2.0 mm isotropic voxels, and an anterior to posterior phase-encoding direction, with matched posterior to anterior images acquired for distortion correction. High-resolution T1-weighted anatomical images were obtained before fMRI acquisition at 0.7 mm isotropic resolution. Standard preprocessing included slice-timing correction, distortion correction, motion correction, anatomical alignment and one-step resampling. Additional details are provided in the Appendix B.3 and B.5.

## 3.7 Quality control and neural signal validation

We performed quality control at multiple stages to establish the quality and reliability of AVID. Temporal signal-to-noise ratio (tSNR) was computed from raw fMRI time series to assess acquisition quality (Appendix B.4). Repeat-based noise-ceiling analyses assessed the reliability of trial-wise neural responses (Appendix B.7). We also examined how neural reliability varied with self-reported imagery vividness (Appendix B.8). Neural signal validation further assessed decodable task-related information using complementary decoding analyses with shuffled-fMRI and matched-Gaussian controls (Appendix B.9). Together, these assessments support the signal quality, repeatability, and neural informativeness of AVID, providing a reliable foundation for auditory-induced visual imagery research.

## 4 Evaluation protocol

## 4.1 Problem formulation

Given fMRI activity recorded during an auditory induced VMI trial, a reconstruction pipeline generates multiple candidate images intended to reflect the participant’s imagined content. Unlike visual perception reconstruction, the imagined content does not have a unique pixel level target. The source image associated with the spoken description provides a reference for constructing the spoken description, supervising the reconstruction model, and computing source-image-based metrics. However, it should not be interpreted as the participant’s exact mental image.

Therefore, the protocol evaluates reconstructions along two complementary axes. Source image consistency measures whether a reconstruction remains visually anchored to the source image from which the spoken description was derived. Spoken-description semantic alignment measures whether a reconstruction preserves the content that was verbally specified to the participant. Reporting these axes separately makes it possible to distinguish visual resemblance to the original source image from semantic recovery of the spoken description.

## 4.2 Evaluation splits and sampling policy

For the baseline evaluations, we use pre-specified stimulus-level splits within each stimulus split. The released split files define which stimuli are used for training and which are held out for testing, ensuring that fMRI responses to the same stimulus do not appear in both sets. This setting evaluates reconstruction from responses to held-out stimuli within the AVID stimulus distribution. However, the split is not designed to enforce strict semantic disjointness between training and test sets; semantically related descriptions or source images may still appear across the two sets.

In addition to the primary benchmark setting, we provide three baseline training settings to characterize common use cases of AVID: single-participant training, leave-one-participant-out transfer training, and multi-participant joint training. These settings are intended as reference baselines rather than mandatory components of the evaluation protocol. They cover subject-specific decoding from an individual’s own data, transfer to a held-out participant using data from other participants, and shared model training across participants. Across these baselines, we use the same released data partitions, neural input definitions, and metric suite. Each reconstruction pipeline retains its own generation and candidate-selection procedure, and the resulting outputs are evaluated with the same source-image consistency and spoken-description alignment metrics.

## 4.3 Neural input settings

To keep model comparisons tied to the same neural evidence, all reconstruction pipelines operate on shared AVID fMRI inputs prepared by the benchmark pipeline. The inputs consist of trial wise beta responses after the preprocessing, repeat reliability analysis, and ROI definition steps described in Section 3.5. The primary neural input uses vertices in $\bar { R O I _ { m } } \backslash R O I _ { s }$ , removing speech responsive vertices while retaining task responsive activity that may support VMI reconstruction.

Unlike the inputs commonly used for visual perception decoding, which provide broad coverage of visual cortex, AVID’s primary input is distributed across frontoparietal, sensorimotor, temporal, and language-selective regions. Early visual cortex (EVC) is sparsely represented: EVC account for approximately 4.5% and 5.8% of the selected vertices in the complex and simple splits, respectively.

## 4.4 Metrics

The metric suite is designed to remain comparable with perception-based fMRI reconstruction work while reflecting the language-specified nature of auditory-induced VMI. We use spoken-description semantic alignment as the primary criterion for semantic recovery, and source-image consistency as a secondary diagnostic of source-compatible visual structure. The two groups are reported separately rather than combined into a single aggregate score.

Source-image consistency. Source-image consistency metrics evaluate whether reconstructions remain visually anchored to the source image associated with the spoken description. We report PixCorr and SSIM as direct image-comparison metrics, and Inception and CLIP-I as feature-based two-way identification accuracies. These metrics enable comparison with perception-based reconstruction work, but are not interpreted as direct measurements of the participant’s exact imagery.

Spoken-description semantic alignment. Spoken-description semantic alignment metrics evaluate whether reconstructions preserve the content conveyed by the spoken description. We report CLIP-T [Radford et al., 2021], SigLIP [Zhai et al., 2023], and AltCLIP [Chen et al., 2023] as image-text two-way identification accuracies, and TIFA [Hu et al., 2023] as a question-based evaluation of entities, attributes, and relations. Detailed metric definitions are provided in Appendix B.10.

Attribute-level recovery. For the simple split, the controlled experimental design allows us to additionally evaluate reconstruction accuracy for three attributes: color, spatial position, and visual identity. We use a frozen Qwen3-VL-8B model as an evaluator to answer closed-set questions about the reconstructed images. We report class-balanced accuracy for each attribute to account for uneven reference-label distributions. Evaluation details and evaluator validation are provided in Appendix B.11.

## 5 Baseline Methods

We evaluate four representative fMRI-to-image reconstruction methods, LDM, MindEye1, MindEye2, and MindBridge, as baseline references for AVID. These methods cover three common training settings: LDM and MindEye1 are evaluated as single-participant models, MindEye2 is evaluated in a leave-one-participant-out transfer setting, and MindBridge is evaluated as a multi-participant joint model. Together, these baselines characterize how AVID supports subject-specific decoding, cross-participant transfer, and shared model training.

Our goal is not to introduce a new decoding architecture, but to examine how established reconstruction methods behave under auditory-induced VMI. To make the comparison faithful to prior work, we preserve each method’s original training and reconstruction setting whenever possible, including its model-selection, generation, and candidate-selection procedures. All methods are evaluated on the released AVID splits, predefined neural input definitions, and shared metric suite.

For AVID, the required adaptations are limited to data loading, neural input formatting, split handling, and evaluation with the released metrics. We do not introduce additional architectural changes or tune methods separately for individual metrics. Implementation details that are needed for reproducibility are provided in Appendix C.

## 6 Results

We evaluate LDM, MindEye1, MindEye2, and MindBridge on the complex and simple splits using the primary neural input $\dot { R O I _ { m } } \backslash \bar { R O I _ { s } }$ This setting removes speech responsive vertices while retaining task responsive activity used for reconstruction. We report qualitative reconstructions together with two groups of metrics: source-image consistency and spoken-description alignment. As a stricter auditory-language control, we also evaluate $R O I _ { m } \dot { \backslash } ( R O I _ { s } \cup R O I _ { l } ) .$ ; retained performance suggests decoding beyond speech- and language-selective regions (Appendix B.13).

## 6.1 Qualitative reconstructions

Figure 2 shows representative reconstructions from the complex and simple splits. MindEye1 gives comparatively stable reconstructions in terms of semantic coverage and overall scene completeness. Its outputs often preserve the dominant object category and coarse scene context, even when the precise layout or local details differ from the source image. LDM sometimes produces visually rich images with strong texture and local detail, but this texture can dominate the reconstruction and does not always correspond to the described content. MindBridge tends to introduce additional semantic elements that are not clearly supported by the description, and its reconstructions are visually less stable in several examples. MindEye2 is less consistent in semantic alignment on this split, often producing images that preserve only weak or partial correspondence to the spoken description. When a description contains several semantic elements, some methods capture only a subset of the described content or shift toward related but mismatched content. This behavior is consistent with the design of the complex split, where the source image anchors stimulus construction and source-image-based evaluation without serving as the participant’s exact imagined image.

![](images/7903607185de6c171fdb6bfefed66ad3655ed3f535996fa93e9a2270418674a3.jpg)  
A  
B  
Figure 2: Qualitative comparison of mental imagery generation results. (A) Examples from the complex stimuli split. (B) Examples from the simple stimuli split. For each stimulus, we compare the source image with reconstructions generated by LDM, MindEye1, MindEye2, and MindBridge. The results illustrate how different methods recover semantic content and visual structure across natural images and simple shape stimuli.

The simple split complements this view. MindBridge appears comparatively stronger in the shown examples, often preserving the semantic identity of the target object and its approximate position. LDM is less reliable in semantic alignment for these sparse stimuli, but it sometimes predicts content at multiple plausible locations and can recover the correct color. Although visually sparse, it also contains multiple semantic attributes specified in the descriptions.

The reconstructions are not uniformly better than those in the complex split. Since these attributes are explicitly specified, such deviations are less plausibly attributed to open-ended variation between an imagined scene and its source image, and more directly suggest incomplete attribute recovery. Across both splits, the examples show that current reconstruction methods can recover recognizable content in favorable cases, but do not yet recover all specified semantics and attributes reliably.

## 6.2 Complex split

Table 2 reports the quantitative results on the complex split, which is the primary setting for evaluating scene-level visual imagery elicited from spoken natural descriptions. MindEye1 achieves the highest mean scores on CLIP-I, CLIP-T, SigLIP, and AltCLIP, whereas MindBridge obtains the highest mean Inception score. LDM also remains above chance on several semantic metrics. These results indicate that single-participant models can extract stable visual and semantic information from auditory-induced imagery responses.

Table 2: Complex split baseline results. Metrics are grouped by source image consistency and spoken description alignment.
<table><tr><td></td><td colspan="4">Image-image</td><td colspan="4">Image-text</td></tr><tr><td>Method</td><td>PixCorr</td><td>SSIM</td><td>Inception</td><td>CLIP-I</td><td>CLIP-T</td><td>SigLIP</td><td>AltCLIP</td><td>TIFA</td></tr><tr><td>LDM</td><td>0.038</td><td>0.185</td><td>59.09%</td><td>59.61%</td><td>66.81%</td><td>68.54%</td><td>67.34%</td><td>32.56%</td></tr><tr><td>MindEye1</td><td>0.056</td><td>0.291</td><td>65.14%</td><td>65.94%</td><td>67.83%</td><td>70.63%</td><td>69.83%</td><td>30.27%</td></tr><tr><td>MindEye2</td><td>0.034</td><td>0.262</td><td>55.75%</td><td>57.81%</td><td>57.95%</td><td>59.08%</td><td>58.82%</td><td>24.34%</td></tr><tr><td>MindBridge</td><td>0.040</td><td>0.170</td><td>65.22%</td><td>65.80%</td><td>67.66%</td><td>69.20%</td><td>69.61%</td><td>32.25%</td></tr></table>

Table 3: Simple split baseline results. Metrics are grouped by source image consistency and spoken description alignment.
<table><tr><td></td><td colspan="4">Image-image</td><td colspan="4">Image-text</td></tr><tr><td>Method</td><td>PixCorr</td><td>SSIM</td><td>Inception</td><td>CLIP-I</td><td>CLIP-T</td><td>SigLIP</td><td>AltCLIP</td><td>TIFA</td></tr><tr><td>LDM</td><td>-0.001</td><td>0.636</td><td>54.07%</td><td>51.75%</td><td>55.59%</td><td>57.16%</td><td>57.25%</td><td>48.92%</td></tr><tr><td>MindEye1</td><td>-0.028</td><td>0.962</td><td>53.19%</td><td>52.75%</td><td>53.19%</td><td>54.33%</td><td>52.83%</td><td>44.34%</td></tr><tr><td>MindEye2</td><td>0.004</td><td>0.684</td><td>53.54%</td><td>51.41%</td><td>52.09%</td><td>52.81%</td><td>51.53%</td><td>39.34%</td></tr><tr><td>MindBridge</td><td>-0.006</td><td>0.906</td><td>57.49%</td><td>55.69%</td><td>55.69%</td><td>57.13%</td><td>56.69%</td><td>40.32%</td></tr></table>

MindBridge achieves mean performance close to MindEye1 on several feature- and text-based identification metrics, indicating that joint training across participants can support reconstruction within the training cohort. However, MindEye2, evaluated using leave-one-participant-out pretraining followed by target-participant fine-tuning, shows lower performance on these metrics. The remaining gap is therefore concentrated in the current transfer baseline, highlighting cross-participant transfer and adaptation as important directions for improving auditory-induced imagery decoding on AVID.

The low PixCorr values across all methods further show that pixel-level recovery of the source image is not an appropriate expectation for auditory-induced VMI. This is consistent with the design of the complex split, where the source image serves as a visual reference for constructing the spoken description and anchoring part of the evaluation, but not as the participant’s unique imagined image. A reconstruction may therefore differ from the exact source-image composition while still preserving the core entities or relations conveyed by the spoken description. These results support reporting source-image consistency and spoken-description alignment separately, rather than treating VMI reconstruction as a standard pixel-level image reconstruction problem.

## 6.3 Simple split

Table 3 reports the results on the simple split. This split is designed as a controlled diagnostic setting rather than an easier version of the complex split. It further supports the conclusion from the complex split that current baselines can recover partial information from AVID, but their success depends strongly on training setting and metric type.

The image-based metrics reveal an important limitation. SSIM is much higher than in the complex split, with two methods above 0.90, whereas PixCorr remains close to zero. This discrepancy may partly reflect the sparse structure of the simple stimuli, as large low-texture backgrounds can inflate structural similarity, while pixel correlation remains sensitive to exact foreground position, scale, and contrast. Thus, high SSIM should not be interpreted as accurate recovery of the specified content, and the near-zero PixCorr again suggests that pixel-level reconstruction is not an appropriate primary objective for auditory-induced VMI.

The description-based metrics provide a more direct diagnostic of attribute recovery. TIFA is higher on the simple split than on the complex split, consistent with the fact that explicitly specified attributes such as identity, color, count, text, and position can be expressed as visual questions. However, the scores remain far from ceiling, indicating that current baselines still do not reliably recover all specified attributes. Together, the simple split complements the complex split by showing that semantic recovery remains incomplete even when the intended visual content is tightly constrained.

Table 4: Attribute-level evaluation on the simple split. Reconstruction scores are class-balanced accuracies, reported as mean ± sample standard deviation across seven participants. The chance row gives the analytical expected score under uniform selection from the corresponding answer options. Differences between model configurations are descriptive.
<table><tr><td>Method</td><td>Color</td><td>Position</td><td>Visual identity</td></tr><tr><td>LDM</td><td> $3 0 . 1 3 \pm 5 . 5 8 \%$ </td><td> $2 4 . 0 7 \pm 5 . 4 6 \%$ </td><td> $2 . 3 6 \pm 1 . 7 9 \%$ </td></tr><tr><td>MindEye1</td><td> $3 1 . 6 5 \pm 7 . 8 3 \%$ </td><td> $3 5 . 3 4 \pm 8 . 5 6 \%$ </td><td> $3 . 3 8 \pm 1 . 9 6 \%$ </td></tr><tr><td>MindEye2</td><td> $3 2 . 5 9 \pm 1 0 . 4 7 \%$ </td><td> $2 9 . 3 9 \pm 4 . 7 3 \%$ </td><td> $2 . 0 7 \pm 1 . 2 5 \%$ </td></tr><tr><td>MindBridge</td><td> $3 5 . 5 5 \pm 4 . 0 4 \%$ </td><td> $2 6 . 9 9 \pm 6 . 8 1 \%$ </td><td> $2 . 7 0 \pm 1 . 3 0 \%$ </td></tr><tr><td>Chance</td><td>25.76%</td><td>20.00%</td><td>1.05%</td></tr></table>

Moreover, the controlled design of the simple split enables a finer-grained evaluation of reconstruction quality beyond aggregate image–text scores. Because the descriptions explicitly specify visual attributes, we can assess whether reconstructions preserve individual components of the intended content, including color, spatial position, and visual identity. Table 4 reports class-balanced accuracy for these attributes, summarized as the mean and sample standard deviation across seven participants.

The results reveal different patterns of attribute recovery across methods. The analytical chance level for color is 25.76% because in test set some stimuli accept two of the four answer options as correct, increasing the expected class-balanced accuracy under uniform random guessing. MindBridge obtained the highest mean color score, whereas MindEye1 obtained the highest mean position and visual-identity scores. Across methods, mean color and position scores exceeded their analytical chance expectations, although recovery remained incomplete and varied across participants. Visualidentity accuracy ranged from 2.07% to 3.38%, compared with a chance expectation of 1.05%, indicating particularly limited recovery of the intended identity. These findings illustrate the diagnostic value of the controlled stimuli: aggregate semantic alignment can coexist with failures to reconstruct specific attributes, which become apparent when those attributes are evaluated separately.

## 7 Data and Code Availability

We release the de-identified AVID dataset, an example tutorial dataset, and accompanying evaluation code. The full dataset is hosted at https://doi.org/10.57760/sciencedb.33666, and the code repository at https://github.com/lishurui0612/AVID-benchmark. A validated Croissant metadata file is provided with the release.

All released imaging data are de-identified before distribution. Additional details on file organization, release terms, and related documentation are provided in Appendix A.

## 8 Conclusion and Limitation

We introduced AVID, a dedicated 5T fMRI dataset for auditory-induced VMI decoding collected under a visual-free imagery paradigm. Based on AVID, we define a benchmark for this task with fixed splits, predefined neural inputs, and a tailored evaluation protocol, and provide baseline evaluations using representative reconstruction methods. The resulting reconstructions represent an initial step toward auditory-induced imagery decoding, preserving much of the recognizable semantic content in the strongest cases and showing trends that are broadly consistent with the quantitative metrics. At the same time, there remains a gap compared with perception-oriented decoding, indicating that future progress will likely require models designed more directly for visual mental imagery.

AVID has several scope boundaries. The current release is limited to Mandarin-speaking participants, and the simple split was collected from a subset of the complex split participants. Because AVID contains human neuroimaging data collected during mental imagery, it is intended for research on auditory-induced visual imagery, neural decoding, and evaluation methodology, with additional limitations and responsible-use considerations provided in Appendix A.3.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Science and Technology Major Project of China (Grant No. 2025ZD0217000 to Y.L.); the National Natural Science Foundation of China (Grant No. 32271099 to X.Z.; Grant No. 32371154 to Y.L.); the Research Center for Brain Cognition and Human Development of Guangdong Province (Grant No. 2024B0303390003 to X.Z.); the Shanghai Rising-Star Program (Grant No. 24QA2705500 to Y.L.); Lin Gang Laboratory (Grant Nos. LG-GG-202402-06 and LGL-1987-18 to Y.L.); and the State Key Laboratory of Advanced Medical Materials and Devices (Grant No. YGSKL2026ZZ01 to Y.L.).

The authors acknowledge the Advanced MR Imaging Research Laboratory in the School of Biomedical Engineering at ShanghaiTech University for providing the imaging platform and support for this work.

Competing interests: Yuanning Li is a cofounder of XS Vision, and Shiqi Shen is an intern at XS Vision. The other authors declare no competing interests.

## References

Anke Marit Albers, Peter Kok, Ivan Toni, H Chris Dijkerman, and Floris P De Lange. Shared representations for working memory and mental imagery in early visual cortex. Current Biology, 23(15):1427–1431, 2013.

Emily J Allen, Ghislain St-Yves, Yihan Wu, Jesse L Breedlove, Jacob S Prince, Logan T Dowdle, Matthias Nau, Brad Caron, Franco Pestilli, Ian Charest, et al. A massive 7t fmri dataset to bridge cognitive neuroscience and artificial intelligence. Nature neuroscience, 25(1):116–126, 2022.

Andrew James Anderson, Elia Bruni, Alessandro Lopopolo, Massimo Poesio, and Marco Baroni. Reading visually embodied meaning from the brain: Visually grounded computational models decode visual-object mental imagery induced by written text. NeuroImage, 120:309–322, 2015.

Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Qinghong Yang, and Ledell Wu. Altclip: Altering the language encoder in clip for extended language capabilities. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8666–8682, 2023.

Nadine Dijkstra, Sander E Bosch, and Marcel AJ van Gerven. Vividness of visual imagery depends on the neural overlap with perception in visual areas. Journal ofNeuroscience, 37(5):1367–1373, 2017.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Evelina Fedorenko, Anna A Ivanova, and Tamar I Regev. The language network as a natural kind within the broader landscape of the human brain. Nature Reviews Neuroscience, 25(5):289–312, 2024.

Giorgio Ganis, William L Thompson, and Stephen M Kosslyn. Brain areas underlying visual mental imagery and visual perception: an fmri study. Cognitive brain research, 20(2):226–241, 2004.

Atsuko Hayashi, Yasumasa Okamoto, Shinpei Yoshimura, Atsuo Yoshino, Shigeru Toki, Hidehisa Yamashita, Fumio Matsuda, and Shigeto Yamawaki. Visual imagery while reading concrete and abstract japanese kanji words: An fmri study. Neuroscience Research, 79:61–66, 2014.

Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20406–20417, 2023.

Yukiyasu Kamitani and Frank Tong. Decoding the visual and subjective contents of the human brain. Nature neuroscience, 8(5):679–685, 2005.

Kendrick N Kay, Thomas Naselaris, Ryan J Prenger, and Jack L Gallant. Identifying natural images from human brain activity. Nature, 452(7185):352–355, 2008.

Reese Kneeland, Paul S Scotti, Ghislain St-Yves, Jesse Breedlove, Kendrick Kay, and Thomas Naselaris. Nsd-imagery: A benchmark dataset for extending fmri vision decoding methods to mental imagery. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28852–28862, 2025.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

Shurui Li, Zheyu Jin, Shi Gu, Ru-Yuan Zhang, and Yuanning Li. A large-scale fmri dataset for vision-language semantic association. Scientific Data, 2026.

Xirong Li, Chaoxi Xu, Xiaoxu Wang, Weiyu Lan, Zhengxiong Jia, Gang Yang, and Jieping Xu. Cococn for cross-lingual image tagging, captioning, and retrieval. IEEE Transactions on Multimedia, 21(9):2347–2360, 2019.

Yuanning Li, Huzheng Yang, and Shi Gu. Enhancing neural encoding models for naturalistic perception with a multi-level integration of deep neural networks and cortical networks. Science Bulletin, 69(11):1738–1747, 2024.

Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1833–1844, 2021.

Yizhuo Lu, Changde Du, Qiongyi Zhou, Dianpeng Wang, and Huiguang He. Minddiffuser: Controlled image reconstruction from human brain activity with semantic and structural diffusion. In Proceedings of the 31st ACM international conference on multimedia, pages 5899–5908, 2023.

David F Marks. Visual imagery differences in the recall of pictures. British journal of Psychology, 64(1):17–24, 1973.

Yoichi Miyawaki, Hajime Uchida, Okito Yamashita, Masa-aki Sato, Yusuke Morito, Hiroki C Tanabe, Norihiro Sadato, and Yukiyasu Kamitani. Visual image reconstruction from human brain activity using a combination of multiscale local image decoders. Neuron, 60(5):915–929, 2008.

Thomas Naselaris, Ryan J Prenger, Kendrick N Kay, Michael Oliver, and Jack L Gallant. Bayesian reconstruction of natural images from human brain activity. Neuron, 63(6):902–915, 2009.

Thomas Naselaris, Cheryl A Olman, Dustin E Stansbury, Kamil Ugurbil, and Jack L Gallant. A voxel-wise encoding model for early visual areas decodes mental images of remembered scenes. Neuroimage, 105:215–228, 2015.

Furkan Ozcelik and Rufin VanRullen. Natural scene reconstruction from fmri signals using generative latent diffusion. Scientific Reports, 13(1):15666, 2023.

Joel Pearson. The human imagination: the cognitive neuroscience of visual mental imagery. Nature reviews neuroscience, 20(10):624–634, 2019.

Joel Pearson, Thomas Naselaris, Emily A Holmes, and Stephen M Kosslyn. Mental imagery: functional mechanisms and clinical applications. Trends in cognitive sciences, 19(10):590–602, 2015.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Paul S. Scotti, Atmadeep Banerjee, Jimmie Goode, Stepan Shabalin, Alex Nguyen, Ethan Cohen, Aidan J. Dempster, Nathalie Verlinde, Elad Yundler, David Weisberg, Kenneth A. Norman, and Tanishq Mathew Abraham. Reconstructing the mind’s eye: fmri-to-image with contrastive learning and diffusion priors, 2023. URL https://arxiv.org/abs/2305.18274.

Paul S Scotti, Mihir Tripathy, Cesar Kadir Torrico Villanueva, Reese Kneeland, Tong Chen, Ashutosh Narang, Charan Santhirasegaran, Jonathan Xu, Thomas Naselaris, Kenneth A Norman, et al. Mindeye2: Shared-subject models enable fmri-to-image with 1 hour of data. arXiv preprint arXiv:2403.11207, 2024.

Mario Senden, Thomas C Emmerling, Rick Van Hoof, Martin A Frost, and Rainer Goebel. Reconstructing imagined letters from early visual cortex reveals tight topographic correspondence between visual mental imagery and perception. Brain Structure and Function, 224(3):1167–1183, 2019.

Guohua Shen, Tomoyasu Horikawa, Kei Majima, and Yukiyasu Kamitani. Deep image reconstruction from human brain activity. PLoS computational biology, 15(1):e1006633, 2019.

Shiqi Shen, Shurui Li, Xilin Zhang, and Yuanning Li. AVID: Auditory-induced Visual Mental Imagery Dataset, 2026. URL https://doi.org/10.57760/sciencedb.33666.

Alfredo Spagna, Dounia Hajhajate, Jianghao Liu, and Paolo Bartolomeo. Visual mental imagery engages the left fusiform gyrus, but not the early visual cortex: A meta-analysis of neuroimaging evidence. Neuroscience & Biobehavioral Reviews, 122:201–217, 2021.

Yu Takagi and Shinji Nishimoto. High-resolution image reconstruction with latent diffusion models from human brain activity. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 14453–14463, 2023.

Varun S Wadia, Chrystal M Reed, Jeffrey M Chung, Lisa M Bateman, Adam N Mamelak, Ueli Rutishauser, and Doris Y Tsao. A shared code for perceiving and imagining objects in human ventral temporal cortex. Science, 392(6794):207–215, 2026.

Shizun Wang, Songhua Liu, Zhenxiong Tan, and Xinchao Wang. Mindbridge: A cross-subject brain decoding framework. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11333–11342, 2024.

Xu Wu, Zhihui Lai, Xianxu Hou, Jie Zhou, Ya-nan Zhang, and Linlin Shen. Lightqanet: Quantized and adaptive feature learning for low-light image enhancement. arXiv preprint arXiv:2510.14753, 2025.

Bright Xu. Nlp chinese corpus: Large scale chinese corpus for nlp. Zenodo, 2019.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

## A Dataset documentation and release

## A.1 Dataset contents and structure

The AVID release follows the Brain Imaging Data Structure (BIDS) format and includes the data and metadata needed to reproduce the neural inputs and evaluation protocol used in the main paper. The released materials include raw fMRI data, preprocessed time-series data, GLMSingle beta estimates, anatomical reconstruction outputs, stimulus metadata, ROI masks, train-test split files, and presentation files containing stimulus order and onset information. The stimulus metadata link each stimulus to its spoken description, audio file, source-image information, task split, and presentation index. The dataset description on Science Data Bank summarizes the released components and access procedure.

Directory structure At a high level, the released dataset is organized as follows:

AVID/   
sub-XX/   
anat/   
sub-XX\_run-\*\_T1w.nii.gz   
sub-XX\_run-\*\_T1w.json   
func/   
session-YY/   
sub-XX\_ses-YY\_task-\*\_run-\*\_AP\_bold.nii.gz   
sub-XX\_ses-YY\_task-\*\_run-\*\_AP\_bold.json   
sub-XX\_ses-YY\_task-\*\_run-\*\_AP\_ref.nii.gz   
sub-XX\_ses-YY\_task-\*\_run-\*\_AP\_ref.json   
sub-XX\_ses-YY\_task-\*\_run-\*\_PA\_bold.nii.gz   
sub-XX\_ses-YY\_task-\*\_run-\*\_PA\_bold.json   
sub-XX\_ses-YY\_task-\*\_run-\*\_PA\_ref.nii.gz   
sub-XX\_ses-YY\_task-\*\_run-\*\_PA\_ref.json   
source\_data/   
used\_stimuli.tsv   
experiments\_scripts/   
sub-XX\_scripts.tar   
stimuli/   
Audio stimuli, source images, and localizer stimuli   
derivatives/   
pp\_data/   
sub-XX/   
anat/   
sub-XX\_avg\_T1w.nii.gz   
func/   
minimally\_preprocessed time-series files   
pp\_betas\_surf/   
sub-XX/   
GLMSingle beta estimates in surface space   
pp\_betas\_vol/   
sub-XX/   
GLMSingle beta estimates in volume space   
freesurfer\_results/   
sub-XX\_fsresults.tar

```shell
ROIs/
sub-XX/
ROI masks of speech perception and linguistic comprehension
noise_ceiling/
surf/sub-XX/
noise-ceiling estimates in surface space
vol/sub-XX/
noise-ceiling estimates in volume space
dataset_official_split/
surface/
sub-XX_complex_train.tar
sub-XX_complex_val.tar
sub-XX_simple_train.tar
sub-XX_simple_val.tar
volume/
sub-XX_complex_train.tar
sub-XX_complex_val.tar
sub-XX_simple_train.tar
sub-XX_simple_val.tar
```

## A.2 De-identification and privacy protection

All released imaging data are de-identified before distribution. Participant identifiers are replaced with random study codes, and the mapping between study codes and participant identities is not released. Direct identifiers, including participant names, dates of birth, height and weight, scan dates, are removed. Anatomical images are defaced before release.

## A.3 Limitations and responsible use

AVID should be interpreted within the assumptions of its acquisition and evaluation protocol. The source image associated with each spoken description is a visual reference used for stimulus construction, model supervision, and evaluation. It is not a unique ground truth image for the participant’s mental imagery. Source image metrics and spoken description metrics should therefore be treated as complementary evaluation proxies rather than direct measurements of an exact imagined image.

The current release is limited in population, language, and sampling scope. All participants were young, healthy Mandarin-speaking adults and were pre-screened for relatively vivid self-reported imagery. This selection may bias the benchmark toward individuals with stronger imagery abilities and limits its representativeness across the broader population. Results should not be assumed to generalize to individuals with lower imagery vividness, other languages, age groups, or clinical populations without further validation. The simple split was collected from a subset of the complexsplit participants and contains less scan time, so comparisons between the two splits should be interpreted as diagnostic rather than as matched estimates of task difficulty.

AVID contains de-identified human neuroimaging data collected during a mental imagery task and should be handled as sensitive research data. The dataset is intended for research on auditory induced visual imagery, neural decoding, and evaluation methodology. It is not intended for participant re identification, clinical diagnosis, individual level profiling or decision making, or claims about private mental content beyond the task conditions represented in the dataset.

## A.4 AVID license

CC BY 4.0

## A.5 Existing assets and licenses

AVID uses existing assets for stimulus construction, MRI preprocessing, baseline implementation, and evaluation. The complex split is built from COCO-CN captions and MS-COCO source images and annotations, following their original licenses and terms. Mandarin auditory stimuli were synthesized using Google Text-to-Speech under the applicable Google Cloud terms. Localizer sentences were selected from the NLP Chinese Corpus. MRI preprocessing and response estimation use established tools including FSL, FreeSurfer, GLMSingle, MATLAB/Psychtoolbox, and GStreamer, and baseline evaluations use released methods and components from LDM, MindEye, MindEye2, MindBridge, CLIP, SigLIP, AltCLIP, TIFA, Inception, and SSIM under their respective licenses. The AVID license applies only to the newly collected fMRI data, dataset metadata, and materials created by the authors, and does not override the licenses or terms of third-party assets.

## A.6 Participant facing materials

Pre-scan preparation and safety instructions. Before scanning, participants reviewed the study information and provided written informed consent. They were then familiarized with the MRI environment, the response button boxes, and the safety squeeze ball. Participants listened to test audio through the MRI-compatible headphones, and the volume was adjusted to ensure that the stimuli were clearly audible over scanner noise while remaining comfortable. They were instructed to keep their head still during scanning, to use the response buttons for vividness ratings during the main task, and to press the safety squeeze ball if they experienced discomfort or wished to stop the experiment. Standard MRI safety instructions were provided before entering the scanner.

Main imagery task instructions. Participants were instructed to imagine only the content described in the audio. If an attribute was not mentioned, they were asked not to add extra semantic content, although surface texture was not constrained. They were instructed to form the image as clearly as possible within the available time. When no spatial position was specified, they were asked to place the imagined content near the center of the scene. Participants kept their eyes closed during imagery trials and were allowed to open their eyes only during non-task periods. They were instructed to keep their head still throughout the scan, to imagine a static image rather than a dynamic event, and to imagine the described visual content rather than recalling a personal experience. Participants were also told that they could stop the experiment at any time and should press the safety squeeze ball in case of discomfort or emergency.

Each main imagery session contained eight auditory-imagery runs and one PA reference scan. Each auditory-imagery run lasted approximately 5 minutes. At the beginning of each auditory-imagery run, an 8 s countdown instructed participants to close their eyes before the first auditory prompt, as shown in Figure 3. In each trial, participants listened to a spoken description lasting approximately 4–6 s and formed the corresponding visual image. They then rated the vividness and richness of the imagined image using the response buttons. The rating was completed before the auditory beep, and participants were instructed to maintain the imagined image until the beep occurred. After the beep, participants stopped imagining the previous content and prepared for the next trial after a reset interval. Participants were allowed to take a brief rest every one or two runs.

Functional localizer instructions. In the localizer task, participants listened to auditory stimuli while keeping their eyes closed. The stimuli included narrative Mandarin sentences, scrambled Mandarin speech, and natural sounds. For narrative Mandarin sentences, participants were instructed to understand the semantic content. For scrambled Mandarin speech, they were instructed to attend to the pronunciation and acoustic properties without trying to derive semantic meaning. For natural sounds, they were instructed to listen carefully to the sounds. Participants were explicitly instructed not to form visual mental images for any localizer stimuli. In each localizer run, auditory stimuli were presented in repeated blocks. Each block contained four consecutive audio clips, followed by a 14 s interval, and this block structure was repeated until the run ended. Narrative Mandarin and scrambled Mandarin speech were paired within one set of runs, while scrambled Mandarin speech and natural sounds were paired within another set. Each set was repeated twice, resulting in four localizer runs in total. Each localizer run lasted approximately 7 minutes.

![](images/9185b19f38f160f8c207c4322243b696eb32ebc174584ec23f547bd7b6a77454.jpg)  
Figure 3: Countdown screenshot.

## B Additional experimental results

## B.1 Stimuli preparation

We prepared three sets of auditory stimuli: the complex stimulus split, the simple stimulus split, and the functional localizer stimuli. All task stimuli were presented as spoken Mandarin audio during eyes-closed fMRI scanning.

Complex stimulus split. We used the processed image–caption pairs from the Caption Scene Dataset (CSD). From these pairs, we first selected captions containing 12–20 Mandarin characters, resulting in 8,903 candidate pairs. We then manually screened these candidates and retained 1,767 stimulus pairs with clear descriptions and vivid, easily imaginable visual content. The selected captions were converted into audio using Google Text-to-Speech and post-processed to 48 kHz. Finally, 800 stimuli were randomly selected from this curated set. These 800 stimuli were shared across all participants; that is, all participants received the same stimulus set, with only the presentation order varying across participants.

Simple stimulus split. The simple split was constructed using rule-based Mandarin caption templates to explicitly control a small set of visual attributes. The actual stimuli presented to participants were all spoken Mandarin descriptions. We used five spatial positions (top, bottom, left, right, and center) and three colors (red, yellow, and blue). The rule-based stimuli included centered uppercase and lowercase letters, centered single digits, centered three-digit number strings, and colored geometric shapes at specified spatial positions. The shape set included triangles, rings, circles, crosses, diamonds, five-pointed stars, hearts, squares, 3-by-3 dot arrays, upward-, downward-, leftward-, and rightward-facing semicircles, vertical rectangles, and horizontal rectangles. This rule-based procedure generated 351 controlled stimuli.

To complement both these rule-based stimuli and the complex scene-level split, we additionally selected 49 high-frequency object categories from ImageNet. These stimuli were designed as singlecategory imagery trials, in which participants imagined one object category at the center of the scene. Together, the simple split contained 400 stimuli and provided a controlled setting for testing whether reconstructions could recover explicitly specified attributes, including object identity, color, alphanumeric content, shape, count, and position.

English caption preparation. All original stimuli in AVID were prepared and presented in Mandarin. Although the source images were derived from COCO-related resources, not every selected image–caption pair retained a directly matched original English annotation. We therefore prepared English captions by translating the Mandarin captions with LLM assistance. Specifically, we used GPT-5.4 to generate structure-preserving English translations. The prompt was:

Table 5: Example complex stimuli.
<table><tr><td>Image</td><td>Chinese caption</td><td>English caption</td></tr><tr><td>新</td><td>带、戴着眼镜。</td><td>这个男人穿着西装、系着领 This man is wearing a suit, a tie, and glasses.</td></tr><tr><td>上國</td><td>了他的球拍</td><td>一个网球运动员在草地上丢下 A tennis player has dropped his racket on the grass.</td></tr><tr><td></td><td>饼</td><td>盘子上有一个涂有奶酪的比萨 There is a cheese pizza on the plate.</td></tr><tr><td><img src="images/7986694c6e1ffb5f75902d528e658fc046918045e1e0e1dcc2a65b2ac3c5b81c.jpg"/></td><td>一辆公交车正在道路上行驶。</td><td>A bus is driving on the road.</td></tr><tr><td><img src="images/508e8ef19e47d3c37372c56a2bb9372490f82b5c9973dc8c453eab44a44d30a9.jpg"/></td><td>一位男子在雪地里玩滑雪板。</td><td>A man is snowboarding in the snow.</td></tr></table>

Add a new field named translated\_caption to the TSV file. Translate the Chinese text in the caption field into English, making the translation as fluent and literal as possible. Do not add extra information, and preserve the original structural content.

Two authors manually reviewed and cross-checked the English translations against the original Mandarin descriptions and revised them where necessary to preserve semantic accuracy. The translated captions were used as auxiliary annotations for documentation, model input, and text-based evaluation. They were never presented to participants during scanning. For the simple split, this procedure nearly fully preserved the predefined template structure of the original Mandarin captions, including position, color, alphanumeric content, shape identity, and object category (Table 5 and 6).

Functional localizer stimuli. The functional localizer was designed to separate auditory, speechrelated, and language-related components elicited by the spoken-description imagery paradigm. It included three auditory conditions: Mandarin sentences, non-word speech-like sequences, and non-linguistic acoustic controls.

For the sentence condition, we selected sixteen Mandarin sentences with low imagery-evoking potential from a Chinese NLP corpus [Xu, 2019]. These sentences were chosen to contain meaningful linguistic content while minimizing explicit visual imagery demands, so that the localizer primarily targeted speech perception and language comprehension rather than visual scene construction.

Table 6: Example simple stimuli.
<table><tr><td>Image</td><td>Chinese caption</td><td>English caption</td></tr><tr><td></td><td>母C</td><td>在中间，有一个红色的大写字 In the center, there is a red uppercase letter C.</td></tr><tr><td></td><td>在中间，有一个红色的数字0</td><td>In the center, there is a red digit 0.</td></tr><tr><td></td><td>圆</td><td>在上方,有一个黄色的朝上的半 At the top, there is a yellow upward- facing semicircle.</td></tr><tr><td></td><td>在左边,有一个蓝色的菱形</td><td>On the left, there is a blue diamond.</td></tr><tr><td></td><td>在中间，有一架飞机</td><td>In the center, there is an airplane.</td></tr></table>

For the non-word condition, we generated speech-like control sequences from the sentence recordings using an audio-quilting procedure. This procedure rearranged short temporal segments of the original speech recordings to disrupt lexical and semantic content while approximately preserving low-level acoustic properties, including the temporal amplitude envelope and speech-like acoustic structure. The resulting stimuli therefore retained many auditory properties of spoken Mandarin but no longer formed intelligible sentences.

For the non-linguistic acoustic control condition, we used natural sound clips that did not contain speech. Together, these three conditions allowed us to estimate speech-responsive vertices by contrasting speech-like stimuli with non-linguistic sounds, and language-selective vertices by contrasting meaningful Mandarin sentences with non-word speech-like sequences.

## B.2 Experiment procedure

Participant screening and pilot experiments. Before MRI data collection, participants completed the Vividness of Visual Imagery Questionnaire (VVIQ), as described in Section 3.2. We selected participants with relatively high self-reported imagery vividness for the MRI experiment. Before finalizing the task protocol, we also conducted pilot experiments using the same auditory-induced imagery paradigm to compare eyes-open fixation and eyes-closed imagery. In the eyes-open variant, participants fixated on a central cross while performing the imagery task. In the eyes-closed variant, participants performed the task with their eyes closed. Based on the pilot results, we adopted the eyes-closed paradigm for the final experiment.

Run and session organization. In the final task design, each run contained 25 trials. Each trial lasted 12 s, comprising an 8 s imagery window followed by a 4 s reset interval. Participants rated the vividness of their imagery before the auditory beep at 8 s and maintained the imagined scene until the beep. After the beep, they stopped imagining and rested until the next trial. Each run began with an 8 s blank period before the first trial and ended with a 4 s blank period after the final trial, resulting in a total run duration of 5 min 12 s. Each scanning session consisted of eight task runs. A posterior-to-anterior phase-encoding reference scan was acquired in the middle of each session, after the first four task runs, to support distortion correction.

Scan schedule. During the formal MRI experiment, participants completed both structural MRI and task-fMRI scanning. For participants 1–4, structural scans were acquired in separate sessions, interleaved with an unpublished retinotopic localizer that is not included in the AVID release; each of these participants completed four T1-weighted scans. For the remaining participants, two T1- weighted scans were acquired during the first functional MRI session, one at the beginning and one at the end of the session. For the complex split, each participant completed eight task sessions, corresponding to 64 task runs in total. For the simple split, each participant completed four task sessions, corresponding to 32 task runs in total. Participants also completed functional localizer scans, including two runs for the speech perception localizer and two runs for the linguistic comprehension localizer.

## B.3 MRI data acquisition

MRI data were acquired on a 5T United Imaging uMR Jupiter system using a 48-channel head coil.   
The MRI acquisition included high-resolution T1-weighted anatomical imaging and task-based fMRI.

During MRI scanning, all participants wore earplugs to reduce scanner noise. During task-fMRI runs, participants additionally wore MRI-compatible headphones to listen to the auditory stimuli. Before scanning, the audio volume was adjusted for each participant to ensure that the stimuli were clearly audible over scanner noise while remaining comfortable. Stimulus presentation was synchronized with scanner triggers to ensure accurate alignment between auditory stimulus onset and fMRI acquisition timing. All auditory stimuli were presented as 48-kHz audio files. Stimulus presentation was implemented in MATLAB using Psychtoolbox 3.0.18 and GStreamer 1.18.6.

Responses were collected using two button boxes, one for each hand, with two buttons on each box. Button responses and response times were recorded throughout the task. A safety squeeze ball was placed next to the participant throughout the experiment, allowing participants to report unexpected discomfort or emergency situations at any time.

Structural T1-weighted MRI images were acquired using a GRE-FSP sequence. The imaging parameters were as follows: voxel size = 0.7 mm isotropic, number of sagittal slices = 270, field of view (FOV) = 221.2 × 257.6 mm, repetition time (TR) = 7.8 ms, echo time (TE) = 2.6 ms, inversion time (TI) = 950 ms, flip angle = 9 degrees, and bandwidth = 220 Hz per pixel.

Whole-brain functional MRI data were acquired using a T2\*-weighted gradient-echo, multi-band accelerated echo-planar imaging (EPI) sequence. The imaging parameters for task-based functional MRI were as follows: number of axial slices = 75, voxel size = 2.0 mm isotropic, matrix size = 100 × 100, field of view (FOV) = 200 × 200 mm, slice thickness = 2.0 mm, no slice gap, repetition time (TR) = 1,500 ms, echo time (TE) = 22 ms, flip angle = 62 degrees, echo spacing = 0.335 ms, bandwidth = 1,740 Hz per pixel, multiband factor = 3, and in-plane acceleration factor = 2. Task fMRI runs were acquired with an anterior-to-posterior (AP) phase-encoding direction, and matched posterior-to-anterior (PA) reference images with the same imaging parameters were acquired for distortion correction.

![](images/f2d2874d16f54a0a21ecea359dde8cfe8dc852216fdc5567b9f245a5d40051d8.jpg)

![](images/c27548859619ec0506a9573da128a23d3ab8746524eef49df4983c8e9759e38c.jpg)  
Figure 4: Temporal signal-to-noise ratio (tSNR) across subjects and runs in AVID. (A) Distribution of run-wise median tSNR values for each subject, where tSNR was computed voxel-wise after quadratic detrending and summarized within the top 10% highest-tSNR voxels in each run. (B) Run-by-run median tSNR trajectories for all subjects, with colored lines indicating individual subjects and the black line indicating the across-subject median. The dashed vertical line separates complex runs from simple runs.

## B.4 tSNR

We assessed signal quality by computing the temporal signal-to-noise ratio (tSNR) on the raw fMRI time series from all participants. For each run, tSNR was computed voxel-wise as the temporal mean divided by the temporal standard deviation. To obtain a simple brain mask without relying on preprocessing-derived masks, we first computed the mean functional volume and selected the top 10% of voxels by mean signal intensity. This threshold was chosen as a coarse high-signal brain mask, considering that the effective cortical voxels occupied approximately 7% of the full imaging volume.

For each run, we summarized signal quality by taking the median tSNR across voxels within this mask. As shown in Figure 4, all participants showed run-wise median tSNR values above 50. Given the relatively short TR and high spatial resolution used in our 5T acquisition, this level of tSNR indicates acceptable raw signal quality. Across runs, tSNR was generally stable for most participants, although participant S6 showed relatively larger run-to-run variability.

## B.5 fMRI preprocessing and data censoring

For participants with multiple T1-weighted scans, individual T1 images were first aligned across scans using FLIRT and then averaged to obtain a high signal-to-noise anatomical reference image. The averaged T1-weighted image was used for cortical surface reconstruction with FreeSurfer’s recon-all. The resulting cortical reconstructions were manually inspected and edited when necessary to ensure accurate gray–white matter segmentation. Because the functional images contained 75 slices, one zero-filled slice was appended to each fMRI volume, resulting in 76 slices before distortion correction and subsequent preprocessing.

Functional MRI preprocessing was performed in several steps. First, slice-timing correction was applied using FSL’s slicetimer. For each run, the acquired multiband reference image was used as the functional reference image, rather than the temporal mean or median of the fMRI time series. Susceptibility-induced distortion was corrected using FSL’s topup, based on the anterior-to-posterior (AP) and posterior-to-anterior (PA) images acquired within the same session. Motion correction was then performed, and all runs from all sessions of the same participant were aligned to a common crosssession functional reference image to improve inter-session stability. The functional reference image was subsequently aligned to the participant’s averaged T1-weighted anatomical image using boundarybased registration (BBR). Finally, the estimated transformation matrices from the preprocessing steps were concatenated and applied in a single resampling step to reduce interpolation-related errors. After the minimal preprocessing steps were completed in volume space, the functional signals were projected onto the cortical surface. Following preprocessing, the results of distortion correction and motion correction were manually inspected to ensure data quality.

![](images/d258186cd9ea5bb799698fa148d5d030068c0f2728f45ba8d08017bf3459ee08.jpg)  
Figure 5: Cortical distribution of noise ceilings, shown in left lateral, ventral, and right lateral views. Colors indicate the estimated percentage of explainable variance in the mean response across two stimulus repetitions. Gray shading depicts cortical anatomy.

Notably, we did not perform volume-level censoring or remove individual fMRI volumes during offline preprocessing. Instead, data quality was monitored during acquisition using the real-time head-motion monitoring system provided by United Imaging. If head motion exceeded the predefined warning threshold during a scan, the corresponding run was excluded from further analysis.

## B.6 Beta estimation

Single-trial beta estimates were obtained using GLMSingle. Each scanning session consisted of eight runs, and each run contained 25 trials. Each trial lasted 12 s, with an 8 s blank period at the beginning of each run and a 4 s blank period at the end. During stimulus preparation, we followed the sampling strategy used in the NSD, in which stimulus repetitions were generated from a probability distribution formed by mixing a relatively narrow von Mises distribution with a uniform distribution. This design ensured that repeated stimuli were present within each session, which supports the estimation procedure in GLMSingle.

For beta estimation, data from each session were entered into GLMSingle independently. We used the Type-D beta estimates as the released single-trial response estimates in the dataset. The modeled stimulus duration was set to 4 s. Although the actual auditory stimulus durations were not strictly identical across trials, they were all approximately 4 s, within a range that can be reasonably modeled by GLMSingle. In the released data, we provide both the original GLMSingle beta estimates and beta estimates that were z-scored within each session.

## B.7 Noise ceiling

We estimated vertex-wise noise ceilings from the two responses to each stimulus. Noise variance was estimated as the mean within-stimulus variance, and signal variance as the total response variance minus noise variance, truncated at zero. The noise ceiling for the two-repeat mean was

$$
\mathrm { N C } _ { v } = 1 0 0 \frac { S _ { v } } { S _ { v } + E _ { v } / 2 } ,
$$

where $S _ { v }$ and $E _ { v }$ denote the estimated signal and noise variances at vertex $v ,$ respectively.

Individual maps were resampled to fsaverage and averaged across participants separately for the complex $( n = 9 )$ and simple $( n = 7 )$ splits, excluding the medial wall. Figure 5 shows the cortical distribution of noise ceilings. With only two repetitions, these estimates provide coarse repeatability diagnostics rather than precise performance upper bounds.

## B.8 Relationship between behavior and neural responses

We examined whether subjective imagery vividness was associated with the reliability of neural responses. Participants reported vividness using a four-point scale during the imagery task. Response frequencies were highly imbalanced across participants, with several participants predominantly selecting a single rating. To avoid sparse estimates from separate rating bins, we ranked stimuli by each participant’s mean vividness rating within each split and compared the 100 highest-rated stimuli with the 100 lowest-rated stimuli. We estimated noise ceilings (NCs) separately for these two stimulus groups in four bilateral FreeSurfer-defined cortical regions: the postcentral gyrus, superior temporal gyrus, fusiform gyrus, and pars orbitalis of the inferior frontal gyrus. This analysis included nine participants for the complex split and seven for the simple split. Comparisons between the high and low-vividness groups used two-sided paired Wilcoxon signed-rank tests across participants.

Table 7: Neural response reliability for the 100 lowest- and 100 highest-vividness stimuli selected separately within each participant and split. Values are across-participant mean ± sample standard deviation of noise-ceiling estimates. The complex and simple splits included nine and seven participants, respectively. p values were obtained using two-sided paired Wilcoxon signed-rank tests.
<table><tr><td>Split</td><td>Cortical region</td><td>Low-vividness NC</td><td>High-vividness NC</td><td>p</td></tr><tr><td>Complex</td><td>Postcentral gyrus</td><td> $8 . 2 1 \pm 2 . 4 6 \%$ </td><td> $1 1 . 5 5 \pm 2 . 6 4 \%$ </td><td>0.0039</td></tr><tr><td></td><td>Superior temporal gyrus</td><td> $1 2 . 3 8 \pm 1 . 2 0 \%$ </td><td> $1 3 . 8 9 \pm 2 . 1 6 \%$ </td><td>0.0273</td></tr><tr><td></td><td>Fusiform gyrus</td><td> $1 0 . 5 7 \pm 3 . 4 3 \%$ </td><td> $1 0 . 4 2 \pm 2 . 8 4 \%$ </td><td>0.8203</td></tr><tr><td></td><td>Pars orbitalis</td><td> $8 . 7 5 \pm 1 . 6 8 \%$ </td><td> $9 . 2 2 \pm 1 . 8 8 \%$ </td><td>0.3594</td></tr><tr><td>Simple</td><td>Postcentral gyrus</td><td> $8 . 9 4 \pm 2 . 4 0 \%$ </td><td> $9 . 6 9 \pm 2 . 0 0 \%$ </td><td>0.0781</td></tr><tr><td></td><td>Superior temporal gyrus</td><td> $1 2 . 0 8 \pm 1 . 4 2 \%$ </td><td> $1 2 . 0 6 \pm 2 . 4 2 \%$ </td><td>1.0000</td></tr><tr><td></td><td>Fusiform gyrus</td><td> $7 . 7 4 \pm 1 . 6 3 \%$ </td><td> $9 . 4 4 \pm 2 . 2 2 \%$ </td><td>0.0156</td></tr><tr><td></td><td>Pars orbitalis</td><td> $7 . 1 5 \pm 1 . 8 8 \%$ </td><td> $8 . 7 2 \pm 2 . 0 5 \%$ </td><td>0.0156</td></tr></table>

Table 7 summarizes the results. In the complex split, the high-vividness group had higher NCs in the postcentral gyrus (11.55 versus 8.21, $p = 0 . 0 0 3 9 )$ and superior temporal gyrus (13.89 versus $1 2 . 3 8 , p = 0 . 0 2 7 3 )$ . The corresponding comparisons in the fusiform gyrus and pars orbitalis yielded $p = 0 . 8 2 0 3$ and $p = 0 . 3 5 9 4$ , respectively. In the simple split, higher NCs were observed for the highvividness group in the fusiform gyrus (9.44 versus $7 . 7 4 , p = 0 . 0 1 5 6 )$ and pars orbitalis (8.72 versus 7.15, $p = 0 . 0 1 5 6 )$ , whereas the postcentral and superior temporal comparisons yielded $p = 0 . 0 7 8 1$ and $p = 1 . 0 0 0 0$

These results demonstrate a regionally selective association between subjective imagery vividness and neural response reliability in AVID. Higher-vividness stimuli were associated with more reliable responses in the postcentral and superior temporal gyri for the complex split, and in the fusiform gyrus and pars orbitalis for the simple split. This correspondence links the recorded neural responses to participants’ reported imagery experience, with the association expressed in specific cortical regions.

## B.9 Neural signal validation

Direct decoding of color and spatial position. To assess whether AVID contains decodable neural information without relying on an image-generation model, we decoded color and spatial position directly from surface-space fMRI beta responses in the simple split. Repeated responses to the same stimulus were averaged, and high-reliability vertices were selected using the simple-split noise-ceiling map. Decoding used five-fold stimulus-level grouped cross-validation, so responses to the same stimulus were not shared between training and test folds. For each of the seven participants, we calculated the cross-validated accuracy among the evaluated ridge-classifier and SVM settings. As a random-feature control, we replaced fMRI inputs with Gaussian features while keeping the labels, folds, feature dimensionality, and decoding procedure unchanged.

Table 8 summarizes the results. Mean decoding accuracy was $0 . 3 9 5 \pm 0 . 0 2 1$ for color and 0.440 ± 0.144 for position, compared with nominal chance levels of 1/3 and 1/5, respectively. One-sided one-sample t-tests against nominal chance yielded t = 7.71, p = 0.00012 for color and $t = 4 . 4 2$ $p = 0 . 0 0 2 2$ for position; Wilcoxon signed-rank tests yielded $p = 0 . 0 0 7 8$ for both attributes. The Gaussian-feature controls remained close to nominal chance.

fMRI-to-text decoding. We decoded captions from fMRI responses to 160 held-out complex stimuli per participant across nine participants using a fixed Qwen caption decoder. We evaluated the decoded captions using BLEU-1, BLEU-4, ROUGE-L F1, and Token-F1, commonly used metrics for evaluating reconstructed or generated text against reference captions. BLEU-1 and BLEU-4 measure n-gram precision up to unigrams and four-grams, respectively; ROUGE-L F1 measures longestcommon-subsequence overlap; and Token-F1 combines token-level precision and recall. These metrics do not have a universal analytical chance level. We therefore estimated empirical chance performance using within-participant shuffled fMRI and independent Gaussian inputs matched to the training-response mean and standard deviation of each input feature. We generated 50 repetitions of each control per participant while keeping the decoder and generation and evaluation procedures fixed. As shown in Table 9, real fMRI outperformed both controls on all four metrics (all Holm-adjusted $p = 0 . 0 3 1 2 5 )$ , supporting a contribution from stimulus-specific neural information beyond the fixed decoder and uninformative inputs.

Table 8: Direct attribute decoding from simple-split fMRI responses and Gaussian-feature controls. fMRI accuracies are means ± sample SD across seven participants; Gaussian values summarize the random-feature control. The p values compare participant-level fMRI accuracies with nominal chance and are uncorrected.
<table><tr><td>Attribute</td><td>fMRI accuracy</td><td>Chance</td><td>Gaussian control</td><td>t-test p</td><td>Wilcoxon p</td></tr><tr><td>Color</td><td> $0 . 3 9 5 \pm 0 . 0 2 1$ </td><td>0.333</td><td> $0 . 3 3 2 \pm 0 . 0 3 1$ </td><td>0.00012</td><td>0.0078</td></tr><tr><td>Position</td><td> $0 . 4 4 0 \pm 0 . 1 4 4$ </td><td>0.200</td><td> $0 . 2 0 0 \pm 0 . 0 3 1$ </td><td>0.0022</td><td>0.0078</td></tr></table>

Table 9: fMRI-to-text decoding with neural-input controls. Values are means ± sample SD across nine participants. Bold indicates significantly higher performance than both controls after Holm correction.
<table><tr><td>Input</td><td>BLEU-1</td><td>BLEU-4</td><td>ROUGE-L F1</td><td>Token-F1</td></tr><tr><td>Real fMRI</td><td> $\mathbf { 0 . 2 2 3 6 \pm 0 . 0 1 9 9 }$ </td><td> $\mathbf { 0 . 0 7 5 9 \pm 0 . 0 0 7 9 }$ </td><td> $\mathbf { 0 . 2 5 2 2 \pm 0 . 0 1 3 7 }$ </td><td> $\mathbf { 0 . 2 9 6 1 \pm 0 . 0 1 9 3 }$ </td></tr><tr><td>Gaussian fMRI</td><td> $0 . 2 0 4 9 \pm 0 . 0 0 7 5$ </td><td> $0 . 0 6 0 9 \pm 0 . 0 0 2 5$ </td><td> $0 . 2 2 9 8 \pm 0 . 0 0 5 0$ </td><td> $0 . 2 7 5 6 \pm 0 . 0 0 3 3$ </td></tr><tr><td>Shuffled fMRI</td><td> $0 . 1 9 2 4 \pm 0 . 0 0 3 1$ </td><td> $0 . 0 6 0 8 \pm 0 . 0 0 1 4$ </td><td> $0 . 2 2 6 6 \pm 0 . 0 0 3 8$ </td><td> $0 . 2 6 4 5 \pm 0 . 0 0 3 2$ </td></tr></table>

Shuffled-fMRI image reconstruction. We additionally tested whether image reconstruction depended on the correspondence between neural responses and target stimuli. This control was conducted only for MindEye1 on the complex split, across nine participants. For each participant, we generated 50 shuffled-fMRI repetitions by randomly permuting stimulus–response correspondences within that participant while keeping the trained model and reconstruction and evaluation procedures fixed. For each repetition, scores were first averaged across participants; the empirical chance distribution was then summarized across the 50 group-level shuffled scores. Real-fMRI performance was compared with this null distribution using a one-sided empirical permutation test. As shown in Table 10, the observed group mean exceeded all 50 shuffled group means for every reported metric, with larger differences for semantic identification than for PixCorr and SSIM.

## B.10 Metric definitions

PixCorr. PixCorr measures low-level pixel agreement between each reconstruction and its corresponding source image. We compute the Pearson correlation between the flattened pixel values of the reconstructed image and the source image after matching their spatial resolution. The final PixCorr score is averaged over images in the evaluation set. Because auditory-induced imagery does not define a unique pixel-level target, PixCorr is used only as a low-level reference metric rather than as a primary measure of VMI reconstruction quality.

SSIM. SSIM measures structural similarity between each reconstruction and its corresponding source image. Unlike PixCorr, SSIM is less sensitive to exact pixel-wise intensity matching and instead captures local luminance, contrast, and structural agreement. The final SSIM score is averaged over images in the evaluation set. As with PixCorr, SSIM is interpreted as a source-image consistency measure, not as direct evidence of the participant’s exact imagined image.

Inception 2-Way Comparison (Inception 2WC). We report Inception 2WC as a feature-based source-image consistency metric. For each reconstructed image i, we compare its Inception feature similarity to the matched source image, $S _ { i i }$ , against its similarity to each mismatched source image, $S _ { i j }$ for all $j \neq i$ . The score is the fraction of pairwise comparisons in which the matched source image is more similar:

Table 10: MindEye1 reconstruction with real and shuffled fMRI on the complex split. Real-fMRI values are means ± sample SD across nine participants; shuffled-fMRI values are means ± SD across 50 group-level shuffle repetitions. Two-way identification accuracies are reported as percentages. Bold indicates significant real-fMRI performance under a one-sided empirical permutation test.
<table><tr><td>Metric</td><td>Real fMRI</td><td>Shuffled-fMRI chance</td></tr><tr><td>PixCorr</td><td> $\mathbf { 0 . 0 5 6 2 \pm 0 . 0 2 4 5 }$ </td><td> $0 . 0 2 7 2 \pm 0 . 0 0 7 2$ </td></tr><tr><td>SSIM</td><td> $\mathbf { 0 . 2 9 1 0 \pm 0 . 0 0 7 4 }$ </td><td> $0 . 2 8 2 8 \pm 0 . 0 0 1 3$ </td></tr><tr><td>CLIP-I 2WC</td><td> $\mathbf { 6 5 . 9 4 \pm 8 . 2 0 \% }$ </td><td> $5 0 . 0 2 \pm 0 . 9 1 \%$ </td></tr><tr><td>Inception 2WC</td><td> ${ \bf 6 5 . 1 4 \pm 8 . 8 6 \% }$ </td><td> $4 9 . 8 2 \pm 1 . 0 3 \%$ </td></tr><tr><td>CLIP-T 2WC</td><td> $\mathbf { 6 7 . 8 3 \pm 8 . 8 1 \% }$ </td><td> $4 9 . 9 0 \pm 0 . 9 6 \%$ </td></tr><tr><td>SigLIP 2WC</td><td> $\mathbf { 7 0 . 6 3 \pm 1 0 . 5 9 \% }$ </td><td> $4 9 . 9 9 \pm 0 . 2 2 \%$ </td></tr><tr><td>AltCLIP 2WC</td><td> $\mathbf { 6 9 . 8 3 \pm 9 . 5 9 \% }$ </td><td> $4 9 . 9 7 \pm 1 . 0 2 \%$ </td></tr></table>

$$
\mathrm { 2 W C } = \frac { 1 } { N ( N - 1 ) } \sum _ { i = 1 } ^ { N } \sum _ { j \neq i } \mathbf { 1 } [ S _ { i i } > S _ { i j } ] .
$$

This score can be interpreted as an average pairwise two-way identification accuracy, with a chance level of 50%.

CLIP-Image 2-Way Comparison (CLIP-I 2WC). CLIP-I 2WC uses the same two-way comparison procedure as Inception 2WC, but computes similarities in the CLIP image-embedding space. It evaluates whether each reconstruction is closer to its matched source image than to mismatched source images in a high-level visual feature space. Its chance level is 50%.

CLIP-Text 2-Way Comparison (CLIP-T 2WC). CLIP-T 2WC evaluates spoken-description alignment using the CLIP image–text embedding space. For each reconstructed image i, we compare its similarity to the matched caption, $S _ { i i }$ , against its similarity to each mismatched caption, $S _ { i j }$ for all $j \neq i ,$ , using the same 2WC formula above. The metric measures whether the reconstruction is more aligned with the correct spoken-description caption than with distractor captions. Its chance level is 50%.

SigLIP-Text 2-Way Comparison (SigLIP 2WC). SigLIP 2WC uses the same image–text two-way comparison procedure as CLIP-T, but computes similarities using SigLIP embeddings. It provides an additional vision-language embedding-based measure of spoken-description alignment. Its chance level is 50%.

AltCLIP-Text 2-Way Comparison (AltCLIP 2WC). AltCLIP 2WC also follows the same image– text two-way comparison procedure, but uses AltCLIP embeddings to construct the image–text similarity matrix. Because AltCLIP is a multilingual vision-language model, it provides a complementary check for text-based semantic alignment in our Mandarin-stimulus setting. Its chance level is 50%.

TIFA. We additionally report TIFA as a question-answering-based image–text consistency metric. For each caption, a frozen set of multiple-choice questions is derived from the described content, and a VQA model answers these questions based on the reconstructed image. The per-image TIFA score is computed by the TIFA scorer, and the final reported value is averaged over images in the evaluation set. The number of questions is not fixed across images: in our frozen benchmark, the complex split contains 1,302 questions over 160 images, with an average of 8.14 questions per image and a range of 3 to 18; the simple split contains 418 questions over 80 images, with an average of 5.23 questions per image and a range of 3 to 9. TIFA does not have a single universal chance level because it depends on the number of answer choices per question. In our question banks, all questions are either 2-way or 4-way multiple choice, corresponding to random-guessing baselines of 50% and 25%, respectively. Under uniform random guessing over answer choices, the expected average score is approximately 36.7% for the complex split and 35.1% for the simple split. This value is a nominal random-answer baseline, not an empirical chance level of the full TIFA pipeline. Because the actual score is produced by a VQA model conditioned on generated images, the effective empirical baseline can be lower when reconstructions do not contain sufficient visual evidence for the queried content.

## B.11 Evaluation of attribute-level recovery

Evaluator and prompting protocol. We use a frozen Qwen3-VL-8B model to evaluate foreground color, spatial position, and visual identity in simple-split reconstructions. Each image is paired with three fixed attribute-specific prompts, without access to the target caption or reference answer. The task instructions are:

• Color: “Determine the dominant visible color of the main foreground content in the image.” Options are red, yellow, blue, and other or multicolored.

• Position: “Determine the location of the main foreground content relative to the entire image canvas.” Options are top, bottom, left, right, and center.

• Visual identity: “Identify the main foreground content in the image.” The full prompt includes 95 options covering letters, digit sequences, geometric shapes, and objects, such as C, 194, triangle, and coffee cup.

Each prompt also provides attribute-specific classification rules and the complete option list. For example, color is judged independently of object identity, and position is determined from the center of the foreground content. All prompts require: “Output only the exact option text.” Inference uses greedy decoding (do\_sample=False) with a maximum of 24 generated tokens. After trimming leading and trailing whitespace, responses are scored by exact matching against accepted reference answers. Responses outside the option list are incorrect, without semantic matching or post-hoc correction.

Metric calculation. We report class-balanced accuracy separately for color, spatial position, and visual identity:

$$
\mathrm { B A } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathrm { A c c } _ { c } ,
$$

where $C$ is the number of reference classes represented in the evaluated images and $\operatorname { A c c } _ { c }$ is the exactmatch accuracy within class c. Each class therefore receives equal weight regardless of its frequency. Responses must exactly match a predefined accepted answer; invalid responses are scored as incorrect. For 11 natural-object stimuli, both the annotated primary color and other or multicolored are accepted; all other questions have a single accepted answer. Across the full 400-image question bank, uniform random guessing yields class-balanced chance levels of 25.57%, 20%, and 1.05% for color, position, and visual identity, respectively.

Evaluator ceiling and failure cases. We evaluated the frozen evaluator on all 400 original reference images using the same prompts and scoring rules. Class-balanced accuracy was 100% for color and position and 98.95% for visual identity (96.25% ordinary accuracy). All 15 identity errors involved semicircle orientation. In 13 cases, the predicted straight-edge direction coincided with the foreground’s position on the canvas, suggesting confusion between global position and local orientation. Table 11 presents representative cases. Each prediction was a valid closed-set response specifying an incorrect straight-edge direction: for example, “top” in the table denotes the full response “semicircle (straight edge at top)”. These results characterize the evaluator’s residual errors on reference images and serve as empirical ceilings.

## B.12 ROI profiles and ROI-restricted reconstruction

To assess whether reconstruction performance could be explained solely by responses in speechand language-responsive cortex, we compared three functionally defined inputs: the speech ROI, the language ROI, and the remaining cortical ROI outside the union of the speech and language localizer masks. The localizer was conducted under eyes-closed auditory conditions, during which participants were instructed not to form visual images. The remaining ROI therefore represents a speech/language-excluded cortical input rather than an anatomically restricted visual-only ROI.

Table 11: Representative evaluator errors on original reference images. The evaluator correctly identifies color and spatial position but misclassifies semicircle orientation. Identity answers are abbreviated to the straight-edge direction; the full answer format is “semicircle (straight edge at [direction])”. Reference descriptions are shown for illustration and are not provided to the evaluator.
<table><tr><td>Reference image Reference description</td><td></td><td>Correct answer</td><td>Evaluator answer</td></tr><tr><td></td><td>At the top, there is a red upward-facing semicircle.</td><td>bottom</td><td>top</td></tr><tr><td></td><td>At the top, there is a yellow left-facing semicircle.</td><td>right</td><td>top</td></tr><tr><td></td><td>On the left, there is a red left-facing semicircle.</td><td>right</td><td>left</td></tr></table>

Table 12: ROI sizes after NC-based vertex selection. Values are the mean ± sample standard deviation of vertex counts across participants (complex: $n = 9$ ; simple: $n = 7 )$ . The top 15% were selected separately within each candidate region.
<table><tr><td>Split</td><td></td><td></td><td>Speech ROI Language ROI Remaining ROI</td></tr><tr><td>Complex</td><td> $1 , 4 2 6 \pm 9 3 3$ </td><td> $^ { 2 , 1 9 1 \pm 2 , 1 5 3 }$ </td><td> $^ { 3 8 , 5 3 8 \pm 5 , 9 0 0 }$ </td></tr><tr><td>Simple</td><td> $1 , 5 6 1 \pm 1 , 0 2 4$ </td><td> $2 , 3 4 8 \pm 2 , 4 6 0$ </td><td> $^ { 3 7 , 5 3 7 \pm 6 , 3 4 4 }$ </td></tr></table>

Vertex selection was performed separately for each participant and task. Within each candidate region, vertices were ranked by noise ceiling, and the top 15% were retained, with the number of selected vertices rounded upward. Table 12 reports the resulting vertex counts. The speech ROI was concentrated in superior temporal regions, particularly the superior temporal gyrus (STG). The language ROI covered frontal and temporal language regions, including the middle frontal gyrus (MFG), inferior frontal gyrus (IFG), anterior temporal cortex, and posterior temporal cortex. The remaining ROI predominantly covered higher-order visual and associative regions along the dorsal and ventral visual pathways, with comparatively limited involvement of primary visual cortex.

We evaluated ROI-restricted reconstruction using MindEye1 on the complex split. A separate model was trained for each participant and ROI for 240 epochs, with the input dimensionality determined by the selected vertex count. Reconstruction used an image-to-image strength of 0.7 and five candidates per stimulus. Performance was evaluated on the 160 held-out stimuli for each of the nine participants and summarized as the participant-level mean ± sample standard deviation. For each of the ten metrics, we tested whether the remaining ROI outperformed the speech ROI and the language ROI using one-sided paired t-tests across participants. Benjamini–Hochberg correction was applied jointly across all 20 comparisons, with significance defined as adjusted $q < 0 . 0 5$

As shown in Table 13, the remaining ROI significantly outperformed both comparison ROIs on seven of the ten metrics: CLIP-I 2WC, Inception 2WC, CLIP-T R@5, CLIP-T 2WC, SigLIP-T 2WC, AltCLIP-T 2WC, and TIFA. For CLIP-T R@1, it significantly outperformed the speech ROI but not the language ROI. Neither comparison established a significant advantage for the remaining ROI on PixCorr or SSIM. These findings indicate that information supporting reconstruction was retained outside the regions identified by the no-imagery speech and language localizer, particularly for high-level visual and semantic measures. However, differences in vertex count cannot be entirely ruled out as a contributing factor to the remaining ROI’s performance advantage. Although this ROI includes higher-order visual and associative regions, the present analysis cannot determine which subregions contribute most to reconstruction. Future work could use parcel-level decoding and region-wise ablation to quantify their contributions.

Table 13: ROI-restricted MindEye1 reconstruction performance on the complex split $( n \ : = \ : 9 )$ Values are the mean ± sample standard deviation across participants. The final two columns report Benjamini–Hochberg-adjusted p-values from one-sided paired t-tests assessing whether the remaining ROI outperformed the respective comparison ROI. Correction was applied jointly across all 20 comparisons; adjusted $p < 0 . 0 5$ indicates significant superiority.
<table><tr><td>Metric</td><td>Speech ROI</td><td>Language ROI</td><td>Remaining ROI</td><td>vs. Speech</td><td>vs. Language</td></tr><tr><td>PixCorr</td><td> $0 . 0 4 6 \pm 0 . 0 1 9$ </td><td> $0 . 0 4 9 \pm 0 . 0 2 6$ </td><td> $0 . 0 6 1 \pm 0 . 0 2 4$ </td><td>0.0747</td><td>0.0915</td></tr><tr><td>SSIM</td><td> $0 . 3 2 7 \pm 0 . 0 2 8$ </td><td> $0 . 2 9 6 \pm 0 . 0 1 1$ </td><td> $0 . 2 9 1 \pm 0 . 0 1 1$ </td><td>0.9976</td><td>0.8892</td></tr><tr><td>CLIP-I 2WC</td><td> $0 . 5 3 8 \pm 0 . 0 6 4$ </td><td> $0 . 6 1 3 \pm 0 . 0 6 9$ </td><td> $0 . 6 5 7 \pm 0 . 0 8 3$ </td><td>0.0011</td><td>0.0136</td></tr><tr><td>Inception 2WC</td><td> $0 . 5 3 7 \pm 0 . 0 6 2$ </td><td> $0 . 6 0 2 \pm 0 . 0 8 1$ </td><td> $0 . 6 4 8 \pm 0 . 0 9 2$ </td><td>0.0030</td><td>0.0257</td></tr><tr><td>CLIP-T R@1</td><td> $0 . 0 1 3 \pm 0 . 0 1 3$ </td><td> $0 . 0 2 6 \pm 0 . 0 3 2$ </td><td> $0 . 0 3 5 \pm 0 . 0 3 9$ </td><td>0.0488</td><td>0.1294</td></tr><tr><td>CLIP-T R@5</td><td> $0 . 0 5 7 \pm 0 . 0 5 5$ </td><td> $0 . 0 9 2 \pm 0 . 0 8 3$ </td><td> $0 . 1 4 7 \pm 0 . 1 0 5$ </td><td>0.0056</td><td>0.0070</td></tr><tr><td>CLIP-T 2WC</td><td> $0 . 5 5 4 \pm 0 . 0 7 1$ </td><td> $0 . 6 1 2 \pm 0 . 0 9 0$ </td><td> $0 . 6 8 1 \pm 0 . 0 9 4$ </td><td>0.0006</td><td>0.0039</td></tr><tr><td>SigLIP-T 2WC</td><td> $0 . 5 5 8 \pm 0 . 0 8 8$ </td><td> $0 . 6 3 3 \pm 0 . 0 8 7$ </td><td> $0 . 7 0 0 \pm 0 . 0 9 5$ </td><td>0.0006</td><td>0.0030</td></tr><tr><td>AltCLIP-T 2WC</td><td> $0 . 5 6 3 \pm 0 . 0 7 9$ </td><td> $0 . 6 2 5 \pm 0 . 0 9 4$ </td><td> $0 . 7 0 3 \pm 0 . 0 9 6$ </td><td>0.0006</td><td>0.0026</td></tr><tr><td>TIFA</td><td> $0 . 2 5 4 \pm 0 . 0 4 0$ </td><td> $0 . 2 7 7 \pm 0 . 0 4 3$ </td><td> $0 . 3 0 9 \pm 0 . 0 5 4$ </td><td>0.0010</td><td>0.0035</td></tr></table>

## B.13 ROI sensitivity analysis

We further evaluate the stricter neural input $\mathrm { R O I } _ { m } \setminus \left( \mathrm { R O I } _ { s } \cup \mathrm { R O I } _ { l } \right)$ as a sensitivity analysis. This input removes both speech-responsive and language-selective vertices from the task-responsive mask. The goal is not to define a purer VMI region, since language-selective cortex may still carry conceptual information used to construct the imagined scene. Instead, this analysis tests whether the main trends remain stable when auditory and language-responsive components are more aggressively excluded.

Figure 6 shows qualitative reconstructions under the stricter input. Compared with the primary setting, the complex examples show similar qualitative behavior. The degree of semantic recovery is broadly comparable, with several reconstructions still preserving recognizable scene content while remaining variable in layout and local detail. The simple examples show more visible loss of some specified attributes in the selected cases, although these examples should be interpreted together with the aggregate metrics rather than as a standalone ranking. LDM remains qualitatively similar across the two ROI settings, partly because its outputs often contain strong texture and predictions at multiple plausible locations. MindBridge more often preserves semantic identity, but color recovery appears less stable than in some other methods. MindEye2 shows weaker color recovery in these examples, while retaining position information relatively well.

Table 14 reports the complex split results. The stricter input produces only modest changes for LDM and MindEye1 across most metrics. MindEye1 continues to achieve relatively strong performance across Inception, CLIP-I, CLIP-T, SigLIP, and AltCLIP under the stricter ROI definition. MindEye2 shows mixed changes across metrics, with no consistent directional shift relative to the primary input. MindBridge shows modest decreases across feature- and text-based identification metrics under the stricter input, while retaining substantial correspondence with the source images and spoken descriptions. This suggests that the multi-participant model is sensitive to the ROI definition, but can still recover decodable information after language-selective vertices are removed.

Table 15 reports the simple split results. PixCorr remains near zero across methods, and SSIM remains high for MindEye1 and MindBridge. The text-based metrics also show method-specific changes. LDM and MindEye1 remain close to their primary-input results, while MindEye2 shows mixed changes across metrics, with no consistent directional shift. MindBridge retains the strongest CLIP-I score, but its AltCLIP and TIFA scores remain lower than those of the strongest simple-split baselines. These results are consistent with the main analysis in showing that the simple split is not uniformly easier than the complex split, and that recovering explicitly specified attributes remains difficult under the stricter ROI definition.

A  
B  
![](images/78b9f0dc40a690aec7bb1ba5fed03b7cca549712bb63f1d8041db144768b121a.jpg)  
Figure 6: Qualitative comparison of mental imagery generation results for ROI sensitivity analysis. (A) Examples from the complex stimuli split. (B) Examples from the simple stimuli split. For each stimulus, we compare the source image with reconstructions generated by LDM, MindEye1, MindEye2, and MindBridge. The results illustrate how different methods recover semantic content and visual structure across natural images and simple shape stimuli.

Table 14: Complex split baseline results under NSL. Metrics are grouped by source image consistency and spoken description alignment.
<table><tr><td></td><td colspan="4">Image-image</td><td colspan="4">Image-text</td></tr><tr><td>Method</td><td>PixCorr</td><td>SSIM</td><td>Inception</td><td>CLIP-I</td><td>CLIP-T</td><td>SigLIP</td><td>AltCLIP</td><td>TIFA</td></tr><tr><td>LDM</td><td>0.040</td><td>0.187</td><td>58.30%</td><td>59.36%</td><td>67.42%</td><td>68.30%</td><td>68.09%</td><td>31.99%</td></tr><tr><td>MindEye1</td><td>0.061</td><td>0.291</td><td>64.85%</td><td>65.71%</td><td>68.06%</td><td>70.04%</td><td>70.33%</td><td>30.91%</td></tr><tr><td>MindEye2</td><td>0.031</td><td>0.265</td><td>55.06%</td><td>56.75%</td><td>55.77%</td><td>57.85%</td><td>57.06%</td><td>24.62%</td></tr><tr><td>MindBridge</td><td>0.040</td><td>0.170</td><td>63.69%</td><td>64.78%</td><td>66.70%</td><td>67.82%</td><td>68.50%</td><td>32.06%</td></tr></table>

Overall, the stricter ROI analysis supports the main findings. The baselines do not rely only on vertices identified as speech-responsive or language-selective, since recognizable visual and semantic information remains recoverable after these vertices are removed. At the same time, the methodspecific changes show that neural input definition is an important part of the evaluation setting. We therefore report both ROI settings to characterize how the baseline evaluations vary with the neura input definition, rather than treating either setting as a definitive isolation of VMI-related activity.

## B.14 Participant-level variability

Tables 16 and 17 report participant-level means and sample standard deviations across both stimulus splits and neural input settings. Each participant contributes one score per metric, and the reported standard deviations quantify between-participant variability.

Table 15: Simple split baseline results under NSL. Metrics are grouped by source image consistency and spoken description alignment.
<table><tr><td></td><td colspan="4">Image-image</td><td colspan="4">Image-text</td></tr><tr><td>Method</td><td>PixCorr</td><td>SSIM</td><td>Inception</td><td>CLIP-I</td><td>CLIP-T</td><td>SigLIP</td><td>AltCLIP</td><td>TIFA</td></tr><tr><td>LDM</td><td>-0.006</td><td>0.636</td><td>51.90%</td><td>52.31%</td><td>55.69%</td><td>57.12%</td><td>56.87%</td><td>48.42%</td></tr><tr><td>MindEye1</td><td>-0.031</td><td>0.963</td><td>52.35%</td><td>51.10%</td><td>51.45%</td><td>51.15%</td><td>51.84%</td><td>44.82%</td></tr><tr><td>MindEye2</td><td>0.003</td><td>0.679</td><td>54.39%</td><td>53.93%</td><td>52.04%</td><td>53.07%</td><td>52.85%</td><td>41.15%</td></tr><tr><td>MindBridge</td><td>-0.008</td><td>0.916</td><td>56.84%</td><td>55.53%</td><td>54.72%</td><td>57.58%</td><td>54.59%</td><td>40.61%</td></tr></table>

Table 16: Participant-level variability in source-image consistency metrics. Values are mean ± sample SD across nine participants for the complex split and seven for the simple split. Inception and CLIP-I scores are reported as percentages. Bold marks the highest mean within each split and input setting. NS denotes $\mathrm { \hat { R } O I } _ { m } \setminus \mathrm { \hat { R } O I } _ { s } ,$ and NSL denotes $\mathrm { R O I } _ { m } \setminus \left( \mathrm { R O I } _ { s } \cup \mathrm { R O I } _ { l } \right)$
<table><tr><td>Method</td><td>PixCorr</td><td>SSIM</td><td>Inception</td><td>CLIP-I</td></tr><tr><td colspan="5">Complex split: NS input</td></tr><tr><td>LDM</td><td> $0 . 0 3 8 \pm 0 . 0 1 4$ </td><td> $0 . 1 8 5 \pm 0 . 0 0 5$ </td><td> $5 9 . 1 0 \pm 5 . 0 0 \%$ </td><td> $5 9 . 6 0 \pm 4 . 9 0 \%$ </td></tr><tr><td>MindEye1</td><td> $\mathbf { 0 . 0 5 6 \pm 0 . 0 2 5 }$ </td><td> $\mathbf { 0 . 2 9 1 \pm 0 . 0 0 7 }$ </td><td> $6 5 . 1 0 \pm 8 . 9 0 \%$ </td><td> $\mathbf { 6 5 . 9 0 \pm 8 . 2 0 \% }$ </td></tr><tr><td>MindEye2</td><td> $0 . 0 3 4 \pm 0 . 0 1 7$ </td><td> $0 . 2 6 2 \pm 0 . 0 1 8$ </td><td> $5 5 . 7 0 \pm 4 . 7 0 \%$ </td><td> $5 7 . 8 0 \pm 4 . 3 0 \%$ </td></tr><tr><td>MindBridge</td><td> $0 . 0 4 0 \pm 0 . 0 0 6$ </td><td> $0 . 1 7 0 \pm 0 . 0 0 6$ </td><td> $\mathbf { 6 5 . 2 0 \pm 6 . 5 0 \% }$ </td><td> $6 5 . 8 0 \pm 7 . 0 0 \%$ </td></tr><tr><td colspan="5">Simple split: NS input</td></tr><tr><td>LDM</td><td> $- 0 . 0 0 1 \pm 0 . 0 1 2$ </td><td> $0 . 6 3 6 \pm 0 . 0 1 7$ </td><td> $5 4 . 1 0 \pm 3 . 2 0 \%$ </td><td> $5 1 . 7 0 \pm 1 . 4 0 \%$ </td></tr><tr><td>MindEye1</td><td> $- 0 . 0 2 8 \pm 0 . 0 1 8$ </td><td> $\mathbf { 0 . 9 6 2 \pm 0 . 0 0 3 }$ </td><td> $5 3 . 2 0 \pm 3 . 4 0 \%$ </td><td> $5 2 . 8 0 \pm 3 . 2 0 \%$ </td></tr><tr><td>MindEye2</td><td> $\mathbf { 0 . 0 0 4 \pm 0 . 0 3 0 }$ </td><td> $0 . 6 8 4 \pm 0 . 0 2 9$ </td><td> $5 3 . 5 0 \pm 3 . 4 0 \%$ </td><td> $5 1 . 4 0 \pm 1 . 4 0 \%$ </td></tr><tr><td>MindBridge</td><td> $- 0 . 0 0 6 \pm 0 . 0 1 4$ </td><td> $0 . 9 0 6 \pm 0 . 0 0 6$ </td><td> $\mathbf { 5 7 . 5 0 \pm 3 . 7 0 \% }$ </td><td> $\mathbf { 5 5 . 7 0 \pm 3 . 6 0 \% }$ </td></tr><tr><td colspan="5">Complex split: NSL input</td></tr><tr><td>LDM</td><td> $0 . 0 4 0 \pm 0 . 0 1 3$ </td><td> $0 . 1 8 7 \pm 0 . 0 0 5$ </td><td> $5 8 . 3 0 \pm 5 . 7 0 \%$ </td><td> $5 9 . 4 0 \pm 5 . 0 0 \%$ </td></tr><tr><td>MindEye1</td><td> $\mathbf { 0 . 0 6 1 \pm 0 . 0 2 4 }$ </td><td> $\mathbf { 0 . 2 9 1 \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 6 4 . 8 0 \pm 9 . 2 0 \% }$ </td><td> $\mathbf { 6 5 . 7 0 \pm 8 . 3 0 \% }$ </td></tr><tr><td>MindEye2</td><td> $0 . 0 3 1 \pm 0 . 0 2 1$ </td><td> $0 . 2 6 5 \pm 0 . 0 1 4$ </td><td> $5 5 . 1 0 \pm 4 . 8 0 \%$ </td><td> $5 6 . 8 0 \pm 4 . 4 0 \%$ </td></tr><tr><td>MindBridge</td><td> $0 . 0 4 0 \pm 0 . 0 0 7$ </td><td> $0 . 1 7 0 \pm 0 . 0 0 5$ </td><td> $6 3 . 7 0 \pm 6 . 5 0 \%$ </td><td> $6 4 . 8 0 \pm 6 . 8 0 \%$ </td></tr><tr><td colspan="5">Simple split: NSL input</td></tr><tr><td>LDM</td><td> $- 0 . 0 0 6 \pm 0 . 0 1 1$ </td><td> $0 . 6 3 6 \pm 0 . 0 2 8$ </td><td> $5 1 . 9 0 \pm 3 . 4 0 \%$ </td><td> $5 2 . 3 0 \pm 0 . 7 0 \%$ </td></tr><tr><td>MindEye1</td><td> $- 0 . 0 3 1 \pm 0 . 0 1 6$ </td><td> $\mathbf { 0 . 9 6 3 \pm 0 . 0 0 3 }$ </td><td> $5 2 . 3 0 \pm 1 . 4 0 \%$ </td><td> $5 1 . 1 0 \pm 1 . 8 0 \%$ </td></tr><tr><td>MindEye2</td><td> $\mathbf { 0 . 0 0 3 \pm 0 . 0 2 5 }$ </td><td> $0 . 6 7 9 \pm 0 . 0 2 3$ </td><td> $5 4 . 4 0 \pm 2 . 1 0 \%$ </td><td> $5 3 . 9 0 \pm 3 . 9 0 \%$ </td></tr><tr><td>MindBridge</td><td> $- 0 . 0 0 8 \pm 0 . 0 1 4$ </td><td> $0 . 9 1 6 \pm 0 . 0 0 5$ </td><td> $\mathbf { 5 6 . 8 0 \pm 3 . 6 0 \% }$ </td><td> $\mathbf { 5 5 . 5 0 \pm 2 . 6 0 \% }$ </td></tr></table>

## C Implementation details

LDM. For LDM, we used a single-subject training setup. Unlike the neural network baselines, LDM does not use epoch-based stochastic optimization. Instead, it fits linear ridge-regression models from fMRI responses to text embeddings and VAE latents using the himalaya kernel ridge implementation. The default configuration used 5-fold cross-validation over a predefined alpha grid, with fit\_intercept=false. Accordingly, the notions of optimizer, learning rate, weight decay, and number of training epochs are not applicable in the usual deep-learning sense. The default batch size for feature extraction was 4, while inference used Stable Diffusion with 50 denoising steps, guidance scale 7.5, and img2img strength 0.8. Experiments were run on NVIDIA A100 80GB GPUs. LDM used one GPU and required approximately 15 minutes per participant.

MindEye1. For MindEye1, we used a single-subject training setup. The model was trained for 240 epochs with a batch size of 64 and a maximum learning rate of $3 \times 1 0 ^ { - 4 } .$ . Optimization was performed with AdamW. Following the original implementation, weight decay was set to $1 0 ^ { - 2 }$ for regular parameters and to 0 for bias and normalization parameters. We used a cycle learning-rate schedule by default. MixCo-based voxel mixup was enabled during the early phase of training with mixup\_pct=0.33, i.e., approximately the first 33% of epochs, after which training switched to the corresponding soft contrastive objective without MixCo. The unified configuration further enabled image augmentation and used CLIP ViT-L/14 as the target embedding space. Experiments were run on NVIDIA A100 80GB GPUs. MindEye1 used one GPU and required approximately 1.5 hours per participant.

Table 17: Participant-level variability in spoken-description alignment metrics. All metrics are reported as percentages, with values given as mean ± sample SD across nine participants for the complex split and seven for the simple split. Bold marks the highest mean within each split and input setting. NS and NSL follow Table 16.
<table><tr><td>Method</td><td>CLIP-T</td><td>SigLIP</td><td>AltCLIP</td><td>TIFA</td></tr><tr><td colspan="5">Complex split: NS input</td></tr><tr><td>LDM</td><td> $6 6 . 8 0 \pm 8 . 3 0 \%$ </td><td> $6 8 . 5 0 \pm 7 . 6 0 \%$ </td><td> $6 7 . 3 0 \pm 8 . 4 0 \%$ </td><td> $\mathbf { 3 2 . 6 0 \pm 4 . 4 0 \% }$ </td></tr><tr><td>MindEye1</td><td> $6 7 . 8 0 \pm 8 . 8 0 \%$ </td><td> $\mathbf { 7 0 . 6 3 \pm 1 0 . 5 9 \% }$ </td><td> $6 9 . 8 0 \pm 9 . 6 0 \%$ </td><td> $3 0 . 3 0 \pm 4 . 9 0 \%$ </td></tr><tr><td>MindEye2</td><td> $5 7 . 9 0 \pm 5 . 3 0 \%$ </td><td> $5 9 . 1 0 \pm 5 . 9 0 \%$ </td><td> $5 8 . 8 0 \pm 6 . 5 0 \%$ </td><td> $2 4 . 3 0 \pm 1 . 9 0 \%$ </td></tr><tr><td>MindBridge</td><td> $6 7 . 7 0 \pm 7 . 0 0 \%$ </td><td> $6 9 . 2 0 \pm 7 . 2 0 \%$ </td><td> $6 9 . 6 0 \pm 7 . 5 0 \%$ </td><td> $3 2 . 3 0 \pm 3 . 3 0 \%$ </td></tr><tr><td colspan="5">Simple split: NS input</td></tr><tr><td>LDM</td><td> $5 5 . 6 0 \pm 3 . 8 0 \%$ </td><td> $\mathbf { 5 7 . 2 0 \pm 1 . 7 0 \% }$ </td><td> $\mathbf { 5 7 . 2 0 \pm 3 . 8 0 \% }$ </td><td> $\mathbf { 4 8 . 9 0 \pm 2 . 0 0 \% }$ </td></tr><tr><td>MindEye1</td><td> $5 3 . 2 0 \pm 4 . 2 0 \%$ </td><td> $5 4 . 3 0 \pm 5 . 2 0 \%$ </td><td> $5 2 . 8 0 \pm 4 . 1 0 \%$ </td><td> $4 4 . 3 0 \pm 1 . 5 0 \%$ </td></tr><tr><td>MindEye2</td><td> $5 2 . 1 0 \pm 2 . 5 0 \%$ </td><td> $5 2 . 8 0 \pm 2 . 6 0 \%$ </td><td> $5 1 . 5 0 \pm 2 . 1 0 \%$ </td><td> $3 9 . 3 0 \pm 2 . 7 0 \%$ </td></tr><tr><td>MindBridge</td><td> $\mathbf { 5 5 . 7 0 \pm 4 . 0 0 \% }$ </td><td> $5 7 . 1 0 \pm 4 . 2 0 \%$ </td><td> $5 6 . 7 0 \pm 2 . 6 0 \%$ </td><td> $4 0 . 3 0 \pm 1 . 5 0 \%$ </td></tr><tr><td colspan="5">Complex split: NSL input</td></tr><tr><td>LDM</td><td> $6 7 . 4 0 \pm 8 . 4 0 \%$ </td><td> $6 8 . 3 0 \pm 8 . 4 0 \%$ </td><td> $6 8 . 1 0 \pm 8 . 4 0 \%$ </td><td> $3 2 . 0 0 \pm 3 . 6 0 \%$ </td></tr><tr><td>MindEye1</td><td> ${ \bf 6 8 . 1 0 \pm 9 . 4 0 \% }$ </td><td> $\mathbf { 7 0 . 0 0 \pm 9 . 5 0 \% }$ </td><td> $\mathbf { 7 0 . 3 0 \pm 9 . 6 0 \% }$ </td><td> $3 0 . 9 0 \pm 5 . 4 0 \%$ </td></tr><tr><td>MindEye2</td><td> $5 5 . 8 0 \pm 4 . 6 0 \%$ </td><td> $5 7 . 8 0 \pm 6 . 5 0 \%$ </td><td> $5 7 . 1 0 \pm 5 . 7 0 \%$ </td><td> $2 4 . 6 0 \pm 2 . 5 0 \%$ </td></tr><tr><td>MindBridge</td><td> $6 6 . 7 0 \pm 8 . 3 0 \%$ </td><td> $6 7 . 8 0 \pm 8 . 6 0 \%$ </td><td> $6 8 . 5 0 \pm 8 . 8 0 \%$ </td><td> $\mathbf { 3 2 . 1 0 \pm 3 . 4 0 \% }$ </td></tr><tr><td colspan="5">Simple split: NSL input</td></tr><tr><td>LDM</td><td> $\mathbf { 5 5 . 7 0 \pm 3 . 5 0 \% }$ </td><td> $5 7 . 1 0 \pm 3 . 0 0 \%$ </td><td> $\mathbf { 5 6 . 9 0 \pm 3 . 4 0 \% }$ </td><td> $\mathbf { 4 8 . 4 0 \pm 2 . 2 0 \% }$ </td></tr><tr><td>MindEye1</td><td> $5 1 . 4 0 \pm 1 . 9 0 \%$ </td><td> $5 1 . 2 0 \pm 3 . 1 0 \%$ </td><td> $5 1 . 8 0 \pm 3 . 8 0 \%$ </td><td> $4 4 . 8 0 \pm 1 . 6 0 \%$ </td></tr><tr><td>MindEye2</td><td> $5 2 . 0 0 \pm 3 . 2 0 \%$ </td><td> $5 3 . 1 0 \pm 3 . 7 0 \%$ </td><td> $5 2 . 8 0 \pm 2 . 3 0 \%$ </td><td> $4 1 . 1 0 \pm 1 . 4 0 \%$ </td></tr><tr><td>MindBridge</td><td> $5 4 . 7 0 \pm 3 . 2 0 \%$ </td><td> $\mathbf { 5 7 . 6 0 \pm 3 . 6 0 \% }$ </td><td> $5 4 . 6 0 \pm 2 . 5 0 \%$ </td><td> $4 0 . 6 0 \pm 2 . 4 0 \%$ </td></tr></table>

MindEye2. For MindEye2, we used the multi-subject leave-one-out (LOO) setting as the default benchmark configuration. The main training stage used all non-held-out subjects jointly and was run for 150 epochs with batch size 64 and maximum learning rate $1 0 ^ { - 3 }$ . Optimization was performed with AdamW with weight decay $1 0 ^ { - 2 }$ , together with a cycle learning-rate schedule by default. MixCo was used in the main training stage with mixup\_pct=0.33, meaning that voxel-space MixCo augmentation was active for approximately the first 33% of epochs. After the main LOO training stage, the held-out subject was further fine-tuned for 15 epochs with learning rate $3 \times 1 0 ^ { - 4 }$ , no learning-rate schedule, and MixCo disabled (loo\_finetune\_mixup\_pct=0.0). The default MindEye2 configuration used a 4-block backbone with hidden dimension 4096, OpenCLIP token targets, and both the diffusion prior and blurry reconstruction branches. Experiments were run on NVIDIA A100 80GB GPUs. MindEye2 used 16 GPUs and required approximately 1 hour per held-out participant, including the LOO training and fine-tuning stages.

MindBridge. For MindBridge, we used the multi-subject joint training setup. The default configuration trained the model for 600 epochs with batch size 4 and validation batch size 4, using a maximum learning rate of $1 . 5 \times 1 0 ^ { - 4 }$ Optimization was performed with AdamW. As in the official implementation, weight decay was set to $1 0 ^ { - 2 }$ for regular parameters and to 0 for bias and normalization parameters. A cycle learning-rate schedule was used by default, with evaluation and checkpointing performed every 10 epochs. In contrast to MindEye1 and MindEye2, MixCo was not used in the default MindBridge setup. The model used a 4-block architecture with hidden size 2048 and incorporated the standard joint reconstruction objective together with the cycle-consistency term from the MindBridge framework. Experiments were run on NVIDIA A100 80GB GPUs. MindBridge used one GPU and required approximately 4 hours per participant.

Table 18: Summary of baseline configurations. The primary input ROI is $\mathrm { R O I } _ { m } \ \backslash$ ROI<sub>s</sub> for all four methods. Detailed hyperparameters are provided in the model-specific descriptions above; the additional speech/language-excluded input is evaluated in the ROI sensitivity analysis.
<table><tr><td>Baseline</td><td>Training setting</td><td>Primary ROI</td><td>input Model fitting and reconstruction</td></tr><tr><td>LDM</td><td>Single-subject</td><td> $\mathrm { R O I } _ { m } \setminus \mathrm { R O I } _ { s }$ </td><td>Ridge regression with 5-fold cross-validation over the regularization grid; Stable Diffusion reconstruction with 50 denoising steps, guid- ance scale 7.5, and img2img strength 0.8.</td></tr><tr><td>MindEye1</td><td>Single-subject</td><td> $\mathrm { R O I } _ { m } \setminus \mathrm { R O I } _ { s }$ </td><td>Subject-specific training for 240 epochs with CLIP ViT-L/14 targets; MindEye1 reconstruc- tion and candidate-selection pipeline.</td></tr><tr><td>MindEye2</td><td>Leave-one- participant-out transfer</td><td> $\mathrm { R O I } _ { m } \setminus \mathrm { R O I } _ { s }$ </td><td>Joint training on non-target participants for 150 epochs, followed by 15 epochs of target- participant fine-tuning; diffusion-prior and blurry reconstruction branches.</td></tr><tr><td>MindBridge</td><td>Multi-participant joint training</td><td> $\mathrm { R O I } _ { m } \setminus \mathrm { R O I } _ { s }$ </td><td>Joint training for 600 epochs with reconstruc- tion and cycle-consistency objectives.</td></tr></table>

Table 19: Identity membership of the additional visual-identity-disjoint split. Counts in parentheses indicate the number of identities. Semicircles are described by the location of their straight edge.
<table><tr><td>Family</td><td>Training identities</td><td>Validation identities</td></tr><tr><td>Letter</td><td>F, G, H, J, K, Q, W, X, Y (9)</td><td>C, S (2)</td></tr><tr><td>Single digit</td><td>0, 1, 2, 3, 4, 6, 7, 9 (8)</td><td>5,8 (2)</td></tr><tr><td>Digits</td><td>103, 194, 372, 431, 566, 653, 740, 918 (8)</td><td>286, 829 (2)</td></tr><tr><td>Shape</td><td>Circle; cross; diamond; five-pointed star; ring; square; three-by-three dot array; triangle; vertical rectangle; semicircles with the straight edge at the bottom, left, or right (12)</td><td>Heart; horizontal rectangle; semicircle with the straight edge at the top (3)</td></tr><tr><td>Object</td><td>T-shirt; backpack; bicycle; book; bookcase; bottle; bottle cap; bow; bus; car; cat; chair; coffee cup; desk lamp; dog; duck; electric fan; forklift; fountain; hair dryer; hammer; laptop; microwave; mirror; mobile phone; motorcycle; mouse; oven; refrigerator; remote control; scissors; sofa; spoon; table; television; toilet; umbrella; watch (38)</td><td>Airplane; bed; chicken; goldfish; golf ball; hatchet; jeans; keyboard; knife; sink; toothbrush (11)</td></tr></table>

Baseline summary. Table 18 summarizes the training settings, primary neural inputs, and implementation procedures of the four baselines. All methods used the same released AVID stimulus-level splits and evaluation metrics.

## D Additional visual-identity-disjoint split

The original simple-split partitions separate individual stimuli but allow different variants of the same visual identity to appear in both training and validation. Specifically, 35 identities are shared between the original partitions, and 66 of the 80 validation stimuli have an identity also represented in training. We therefore provide an additional visual-identity-disjoint split of the same 400 stimuli, comprising 320 training stimuli covering 75 identities and 80 validation stimuli covering 20 identities, with zero identity overlap.

Table 19 lists the identities assigned to each partition. All color, position, and letter-case variants of a given identity belong to the same partition. The validation set contains letters C and S, digits 5 and 8, digit sequences 286 and 829, three geometric shapes, and eleven object categories. Color and position values remain represented in both partitions, so this protocol evaluates generalization to held-out visual identities while retaining shared attribute values.

The additional split can be reproduced by running the following command from the root of the accompanying GitHub repository:

python scripts/generate\_simple\_identity\_disjoint\_split.py

All reconstruction results reported in this paper use the original stimulus-level splits. The additional identity-disjoint split is provided as an optional evaluation protocol; baseline performance under this protocol has not been evaluated in the present study.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state the paper’s contributions as a visual-free auditory-induced VMI paradigm, the AVID 5T fMRI dataset, a benchmark defined based on AVID, and baseline evaluations under the proposed protocol. The claims are limited to dataset construction, evaluation, and baseline characterization rather than model novelty.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses limitations in the Conclusion and Limitations section and Appendix A, including the Mandarin-speaking participant population, split imbalance, and the sensitivity of human neuroimaging data.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not present theoretical results, theorems, or formal proofs. Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper specifies the acquisition paradigm, stimulus splits, preprocessing, beta estimation, neural input definitions, metric suite, and baseline evaluation protocol. The dataset link provides detailed documentation on data organization and usage, and the accompanying code link provides scripts for data loading, metric computation, and baseline evaluation to support reasonable reproduction of the reported results.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: The paper provides access to the de-identified AVID release, Croissant metadata, and code for data loading, metric computation, and baseline evaluation. The dataset and code links provide reviewer-accessible documentation and scripts for using the released data and reasonably reproducing the reported baseline evaluations.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Sections 3–5 describe the dataset construction, participant task, stimulus splits, neural input settings, evaluation metrics, and baseline methods needed to understand the results. Appendix C provides additional baseline implementation details.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The paper reports participant-level means and sample standard deviations for reconstruction and attribute-level evaluation results (Tables 4, 16 and 17). We also report paired Wilcoxon signed-rank tests for vividness-related differences in neural reliability (Table 7), statistical comparisons of fMRI-to-text decoding against shuffled-fMRI and matched-Gaussian controls (Table 9), and one-sided empirical permutation tests for MindEye1 reconstruction against shuffled-fMRI controls (Table 10). The reported variability measures and statistical conclusions are specific to the corresponding experimental settings.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper reports the hardware used for the experiments and provides additional compute details in Appendix C, including GPU type, memory, and resources needed for the baseline evaluations.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The study was conducted in accordance with institutional ethical guidelines, with written informed consent from all participants. The released data are de-identified, and the paper documents intended uses and disallowed uses for AVID.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses potential positive impacts for VMI decoding and neuroprosthetic communication, as well as privacy and misuse risks associated with human neuroimaging data. Responsible-use constraints and disallowed uses are described in Appendix A.

## Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [Yes]

Justification: The paper describes safeguards for releasing sensitive human neuroimaging data, including de-identification, random participant codes, removal of direct identifiers, defacing of anatomical images, and documentation of intended and disallowed uses.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper cites the existing datasets, models, software tools, and evaluation components used in the study. Appendix A.5 summarizes their roles and licensing or usage terms, and clarifies that the AVID license applies only to newly collected fMRI data, dataset metadata, and author-created materials.

## Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The paper introduces AVID as a new dataset and provides documentation on dataset contents, structure, access, license, de-identification, limitations, and responsible use. The release also includes Croissant metadata and accompanying code for evaluation.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [Yes]

Justification: Sections 3.2 and 3.3 describe the participant task procedure, consent, compensation, withdrawal rights, and public data sharing after anonymization. Appendix A.6 provides the full participant-facing instructions.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [Yes]

Justification: The paper states that the study received institutional ethics approval and that all participants provided written informed consent in accordance with institutional guidelines. The participant procedure also describes withdrawal rights, compensation, anonymized data sharing, and MRI safety procedures.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: We used GPT-5.4 to translate Mandarin stimulus descriptions into English for documentation, model input, and text-based evaluation. Two authors manually reviewed and cross-checked the translations against the original Mandarin descriptions and revised them where necessary to preserve semantic accuracy. We also used Qwen3-VL-8B for attribute-level evaluation of reconstructed images.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.