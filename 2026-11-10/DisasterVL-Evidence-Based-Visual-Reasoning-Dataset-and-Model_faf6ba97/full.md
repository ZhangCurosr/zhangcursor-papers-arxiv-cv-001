# DisasterVL: Evidence-Based Visual Reasoning Dataset and Model for Disaster Scene Understanding

Ali Diba<sup>1</sup>, Aisha Al-Mohannadi<sup>1,2</sup>, Yin Yang<sup>2</sup>, Muhammad Imran<sup>1</sup>, Ferda Ofli<sup>1</sup> <sup>1</sup>Qatar Computing Research Institute, Hamad Bin Khalifa University, Doha, Qatar <sup>2</sup>College of Science & Engineering, Hamad Bin Khalifa University, Doha, Qatar

{adiba,aalmohannadi,yyang,mimran,fofli}@hbku.edu.qa

## Abstract

General-purpose vision-language models struggle with crisis imagery due to its inherent noise, ambiguity, and high variability across disaster events — yet timely, accurate scene understanding is critical for humanitarian response. To address this, we introduce DisasterVL, a large-scale multitask dataset containing 1.2 million disaster images from 142 countries paired with 25 million structured annotations spanning dense captioning, incident and location assessment, relational scene descriptions, and crisis-grounded visual question answering (VQA) across diverse hazard types and scene contexts worldwide. Unlike prior disaster benchmarks that focus narrowly on classification, DisasterVL unifies recognition, grounded reasoning, and uncertaintyaware interpretation. A distinguishingfeature is its evidencegrounded annotation design: each VQA instance includes supporting evidence text that links visual observations to answers, alongside positive and negative question pairs, multiple answer formats, and difficulty labels. Visual grounding layer links caption phrases to image regions and relationship statements to subject-object boxes, with explicit labels for visual supports and clues. To support rigorous evaluation, we construct a human-verified test set of38K images (1M annotations) spanning broad geographic regions and disaster types. Experiments with domain-adapted vision-language models demonstrate strong performance across disaster captioning, incident recognition, and grounded VQA, while also exposing the limitations ofgeneral-purpose models in crisisspecific reasoning. DisasterVL establishes a scalable, reproducible benchmarkfor vision-language models that must reason reliably under real-world uncertainty. We will publicly release the dataset, trained models, and code.

## 1. Introduction

Visual evidence is critical to disaster response and humanitarian coordination, as it offers rapid insight into impacts that are often difficult to capture through textual reports alone [3, 32, 36]. Images shared during crisis incidents, such as floods, earthquakes, can reveal damaged buildings, inundated roads, blocked infrastructure, displaced populations, and other critical conditions on the ground [5, 7]. As the volume of such imagery continues to grow across online social platforms, news media, and field reports, the ability to automatically interpret disaster images at scale has become increasingly important. Reliable visual understanding can support faster situational awareness, improve prioritization of response actions, and help humanitarian organizations make more informed decisions under time pressure and uncertainty [9, 12, 30, 38].

Recent advances in vision-language models offer a promising path toward automated disaster image understanding, enabling systems to describe images, answer questions, and support flexible interaction with visual data [17, 34, 40]. Yet crisis imagery remains challenging for general-purpose VLMs because such images are often noisy, ambiguous, occluded, and shaped by complex cues such as smoke, debris, and floodwater [6, 24]. In these settings, models may rely on superficial patterns rather than grounded visual evidence, producing hallucinated objects and overconfident but incorrect answers to operationally relevant questions [21, 22]. Such errors are especially consequential in humanitarian contexts, where visual misinterpretation can distort situational awareness and downstream decisions. Existing disaster vision datasets, such as CrisisMMD [4], Incidents1M [52], xBD [23], have enabled progress in incident classification, damage assessment, and humanitarian information extraction. However, most remain designed around narrow recognition tasks or specific data modalities, offering limited support for evaluating broader visual understanding, grounded reasoning, and generalization across diverse disaster events.

To address this gap, we introduce DisasterVL, a largescale dataset for grounded vision-language learning in disaster scenarios. DisasterVL contains approximately 1.2 million crisis-related images from 142 countries organized under a canonical taxonomy of 29 disaster/hazard categories, covering more than 12K incident types, such as wildfires, earthquakes, urban destruction, and floods. More importantly, DisasterVL provides structured annotations for complementary tasks, including (i) image captioning (1.2 million), (ii) incident classification and reasoning (1.2 million), (iii) grounded visual question answering (20.5 million), and (iv) multiple-choice question answering (2.7 million). These annotations are generated through a controlled VLM-assisted pipeline that emphasizes observable visual evidence, uncertainty marking, and humanitarian information needs such as hazard recognition, damage indicators, spatial exposure, infrastructure conditions, response presence, and hallucination resistance. This design enables the same visual content to be studied from multiple perspectives, ranging from descriptive understanding to grounded and operationally meaningful crisis reasoning.

![](images/fc38dac191be60ca4de7a726f9225d88d35322ec782915249e78f8de62f5aa8c.jpg)  
Figure 1. A DisasterVL sample and its multi-layer annotations: caption, disaster type, evidence, spatial relationships, question-answer pairs, multiple-choice questions, and uncertainty statements.

We extend the annotation with region-level visual grounding: visual clues from caption text are linked to bounding boxes, and relationship statements are linked to subject and object regions with supported, unclear, or contradicted labels. Phrases that cannot be localized are retained with a reason. This adds an explicit localization task to the existing captioning and reasoning tasks, enabling inspection of the spatial evidence associated with a textual claim.

To assess different levels of disaster image understanding, we evaluate DisasterVL by training VLMs on the dataset and testing their performance across image captioning, VQA, and reasoning tasks. We fine-tune Qwen3.5 models on DisasterVL and compare them against general-purpose opensource and proprietary VLMs using task-appropriate evaluation metrics. The results show that DisasterVL fine-tuning improves disaster-specific visual understanding: the adapted

9B model outperforms all baselines, including GPT-5, on every generative and grounding task, with the largest gains on reference-overlap metrics for reasoning tasks. Gains are smaller on GES, which scores factual agreement rather than wording, and on multiple-choice QA, where adapted and strong general-purpose models perform similarly.

Overall, DisasterVL provides a large-scale foundation for disaster-focused vision-language learning. To the best of our knowledge, it is the first multimodal disaster dataset to combine million-scale captions, tens of millions of VQA pairs, broad geographic coverage, and diverse disaster types within a unified benchmark. Models trained on DisasterVL show substantial gains over general-purpose baselines, demonstrating the value of domain-specific, evidence-oriented supervision for reliable disaster image understanding.

## 2. Related Work

Disaster Image Analysis Datasets. Prior disaster image understanding broadly splits into aerial/satellite and groundlevel analysis. Remote sensing (RS) datasets have targeted specific disaster types, such as floods (Sen1Floods11 [13]) and landslides (Sen12Landslides [26]), or specific tasks such as building damage assessment (xBD [23]). More recent benchmarks like FloodNet [41] and DisasterM3 [48] have incorporated VQA for overhead visual reasoning, yet remain inherently limited by a bird’s-eye perspective that misses proximal humanitarian signals. Ground-level datasets have similarly evolved from specialized damage assessment toward broader multimodal categorization: MEDIC [7] and CrisisMMD [4] established foundations for humanitarian classification, while Incidents1M [52] provided scale for robust detection. However, these resources remain predominantly classification-centric. Nascent reasoning-oriented benchmarks like DisasterVQA [2] show promise but lack the scale required to fine-tune modern VLMs (see Table 1). DisasterVL bridges this gap by unifying large-scale incident detection with generative reasoning — integrating captioning, incident labeling, and open-ended and multiple-choice VQA to enable grounded, reliable crisis reasoning.

Table 1. Comparison of disaster-oriented datasets by view, scale and task coverage.
<table><tr><td>View</td><td>Dataset</td><td>Images</td><td>Ann.</td><td>Classification</td><td>Segmentation</td><td>Captioning</td><td>VQA</td><td>MCQ</td><td>Visual Grounding</td><td>Reasoning</td></tr><tr><td rowspan="5">Aerial</td><td>Sen1Floods11 [13]</td><td>4,831</td><td></td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Sen12Landslides [26]</td><td>39K</td><td></td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>xBD [23]</td><td>22K</td><td></td><td>×</td><td>√</td><td>X</td><td>X</td><td>×</td><td>X</td><td>X</td></tr><tr><td>FloodNet [41]</td><td>2,348</td><td>7,345</td><td>√</td><td>√</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>DisasterM3 [48]</td><td>27K</td><td>123K</td><td>√</td><td>√</td><td>√</td><td>X</td><td>√</td><td>X</td><td>√</td></tr><tr><td rowspan="5">Ground</td><td>MEDIC [7]</td><td>72K</td><td></td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Incidents1M [52]</td><td>977K</td><td></td><td>√</td><td>X</td><td>X</td><td>X</td><td>×</td><td>X</td><td>X</td></tr><tr><td>CrisisMMD [4]</td><td>18K</td><td>16K</td><td>V</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>DisasterVQA [2]</td><td>1,395</td><td>4,405</td><td>×</td><td>X</td><td>X</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td>DisasterVL (Ours)</td><td>1.2M</td><td>25M</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Visual Question Answering and Grounded Reasoning. VQA has evolved from basic recognition toward compositional reasoning and visual grounding. Early benchmarks exposed a critical flaw: models could exploit answer priors without genuinely attending to images [1, 8, 21], prompting a shift toward balanced distributions, polarity-aware evaluation, and evidence-based supervision [14, 25, 29, 39, 42, 49]. These principles are particularly salient for disaster analysis, where negative evidence and uncertainty-aware reasoning are as critical as positive identification. DisasterVL incorporates these diagnostic principles through reasoning-oriented and multiple-choice VQA across open-ended and constrained answer spaces. In the RS domain, models have progressively advanced from basic captioning (RS-GPT [28]) to region-level grounding (GeoChat [31], LHRS-Bot [35], SkyEyeGPT [53]), and more recently to multi-sensor, multitemporal reasoning (EarthGPT [54], GeoBench-VLM [18], EarthDial [46], EarthMind [44]). Yet these remain generalpurpose, with disaster coverage limited to isolated tasks like flood detection or damaged building counting. DisasterVL differentiates itself through comprehensive vertical support for the humanitarian domain, bridging general-purpose RS reasoning and the specialized demands of on-the-ground crisis response. The region-level annotation complements answer-level evidence by identifying the image regions associated with caption phrases and the entities participating in a stated relationship. These links provide a spatial target for grounding, separate from the quality of the generated answer.

## 3. DisasterVL Dataset

Disaster image understanding requires models to recognize hazards, damaged infrastructure, affected environments, response activities, and uncertainty under visually noisy and geographically diverse conditions. DisasterVL is designed to address these challenges by providing large-scale, evidencebased supervision for disaster-specific vision-language learning, with broad coverage of disaster types, geographic contexts, visual scenes, and humanitarian-relevant information needs. Next, we describe the dataset collection, filtering, and annotation steps.

## 3.1. Data Collection and Filtering

We collected over 7 million images from Twitter/X between 2018 and 2023 using hundreds of disaster-specific keywords, hashtags, event names, and multilingual query terms spanning floods, earthquakes, wildfires, hurricanes, landslides, explosions, and severe storms. After removing approximately 2 million exact and near-duplicate images via hashing and color histogram filtering, the remaining 5 million images were screened with Qwen3-VL-32B [10] for visually observable disaster-related content, yielding the 1.2 million potentially relevant images that constitute DisasterVL.

## 3.2. Multidimensional Scene Captioning and Evidence Extraction

As shown in Figure 2, in the first step, each filtered image was annotated with GPT-5 [45] to generate: (i) a dense factual caption, (ii) an incident assessment, (iii) visual evidence, (iv) a location assessment, (v) visible spatial and interaction relationships, and (vi) uncertainties. The model was instructed to base all outputs only on visible evidence and to avoid unsupported assumptions about causes, locations, identities, injuries, or intent.

The caption describes the main visible content of the image, including damaged infrastructure, environmental conditions, objects, property conditions, human presence, and spatial layout. The incident assessment assigns a likely incident type, together with a confidence level and provides visual evidence. For example, collapsed walls, exposed interiors, rubble, and heavy machinery near a damaged building may support a structural-damage label, while the exact cause may remain unknown. The likely types produced a longtail distribution of more than 12K incident variants, which we mapped to 29 canonical disaster/hazard types, such as flood, wildfire, road traffic accident. The location assessment records possible geographic cues only when visible evidence is available, such as signs, landmarks, language, or architecture; otherwise, the location is marked as unknown.

![](images/69c87c40980f9f28d58480653df8645f525dad3ac69a36f5596cca0454799256.jpg)  
Figure 2. DisasterVL data and annotation pipeline: A curated disaster image dataset is first collected. GPT-5 [45] then generates imagegrounded captions and supporting evidence, followed by tailored VQA annotations conditioned on the caption, incident assessment, supporting evidence, relationships, and uncertainties. Finally, a human and Gemini 3.1 Pro [19] work together to create a verified test set.

The annotation also captures relationships among visible entities, such as heavy machinery positioned next to a damaged building, rubble accumulated along a sidewalk, people standing near equipment, or adjacent buildings appearing intact. Readable text is transcribed when present. Finally, uncertainty notes identify ambiguous elements that cannot be confirmed visually, such as whether heavy machinery is being used for demolition or debris removal, the extent of unseen damage, or the absence of reliable geographic cues. Prompt templates for this step can be found in Appendix E.1.

## 3.3. Visual Question-Answer Generation

We generate visual question-answer pairs to help VLMs learn disaster-scene understanding beyond captioning. We aim for diverse VQAs (binary, MCQs, and open-ended) that cover hazard recognition, damage interpretation, spatial reasoning, critical infrastructure, response presence, difficulty level, and uncertainty handling. For each image, GPT-5 is provided with annotations from the last stage: generated caption, incident assessment, supporting evidence, relationships, and uncertainties. The model is then asked to generate 12–15 positive QA pairs and up to five negative yes/no questions.

Positive questions focus on visible, crisis-relevant details, such as hazard type, structural damage, rubble location, proximity to machinery, traffic infrastructure, or possible responder presence. Answers are required to remain imagebased, use visible cues, and express uncertainty when needed. Negative questions serve as hallucination checks by asking about absent conditions, such as whether flooding or water damage is visible in a structurally damaged building scene, whether fire or smoke is shown when no burn or smoke cues are present, or whether emergency vehicles such as fire trucks or ambulances appear when none are visible. Prompt templates for VQA and MCQ generation can be found in Appendix E.2 and Appendix E.3, respectively.

## 3.4. Region-Level Visual Grounding Annotations

DisasterVL has a visual grounding layer for the existing captions and relationship statements. Each annotated region consists of a bounding box and a short description of its visible content, such as a tree trunk engulfed in flames. Grounded caption phrases are linked to one or more corresponding regions. These links retain the connection between the original description and its spatial evidence rather than treating the boxes as independent object labels.

Phrases that cannot be assigned a box are retained as unresolved, with a reason indicating an absence statement, non-visual content, an not-visible entity, or an ambiguous reference. For example, “no people are visible” is recorded as an absence statement rather than assigned a person box.

## 3.5. Test Set Construction and Quality Control

Next, we design a test set that is reliable, geographically broad, and representative of diverse disaster types. As most images in DisasterVL were collected during real-world disaster events across various geographies, such as the 2019 Hurricane Dorian and the 2023 Turkey earthquake, the dataset carries natural geographic diversity that we leverage during test set construction. For each canonical disaster category, we set a sampling target of up to 10% of available instances, capped at 4,000 images per category. To ensure intra-category diversity rather than redundant near-duplicate sampling, we apply k-means clustering within each category prior to sampling, setting the number of clusters equal to the per-category target. One representative image is then drawn from each cluster, with priority given to samples that carry event-based location information, reinforcing geographic breadth in the final selection. As k-means does not always yield the full target number of clusters, the final per-category count may be slightly below the target, resulting in a test set of 38K images. This design enables evaluation of both geographic generalization, such as model performance across regions, and disaster-type generalization, such as robustness across different hazard categories, spanning 75 countries and 28 disaster/hazard types.

To improve test set reliability, we used Gemini 3.1 Pro [19] with human verification to audit the test set annotations, including captions, incident evidence, location assessments, and VQAs. First, given the image and its full annotation record, Gemini checked whether each claim was visually supported, internally consistent, and relevant to disaster assessment. It was instructed to preserve the original annotation by default and make only corrections when an error was clear. Next, we grouped Gemini’s proposed edits by edit magnitude. Edits with less than 80% similarity between the original and revised text were treated as major; edits above this threshold were treated as moderate, while small wording or stylistic suggestions were classified as minor. Human reviewers inspected 2–5% of edits from all three groups. About 70% of major edits were valid corrections, compared with fewer than 20% of moderate edits; minor edits were generally unnecessary. We therefore accepted only major Gemini edits and retained the original GPT-5 annotations for moderate and minor cases. Prompt template for test set verification can be found in Appendix E.4.

Human verification. We human-validated a stratified sample of 1,500 test images after the Gemini audit, with 51–52 images per canonical type. Two annotators independently judged each canonical label, caption claim, and VQA answer, and a third expert adjudicated disagreements. Interannotator agreement was 98.0% for canonical labels (Cohen’s κ = 0.98), 96% for caption claims, and 95.0% for VQA correctness. After adjudication, 1% of canonical labels (14/1,500), 1.3% of caption claims (39/3,000), and 2.1% of answerable VQA answers (65/3000) were incorrect, which estimates the residual error of the released test set.

## 3.6. Dataset Statistics and Coverage

Table 2 summarizes the scale, coverage, and annotation stats of DisasterVL. The dataset contains 1,235,605 images across 29 canonical hazard types, with 1.2M detailed evidencegrounded captions and more than 23M question annotations (20.57M QA pairs; 2.71M MCQs). Each image has 16 QA pairs on average. Beyond the canonical taxonomy, incident labels form a much richer long tail of about 12K likely incident types, capturing the fine-grained diversity of real disaster imagery. The test set preserves all 28 canonical types and includes 726,869 question annotations (641,167 QA pairs; 85,702 MCQs) spanning 1,498 likely incident types.

DisasterVL also provides broad geographic and semantic coverage. Location metadata is available for about 70% of the full dataset (covering 142 countries) and 60% of the test set. Captions are dense, averaging 88.63 words, indicating detailed scene descriptions rather than short labels. Most examples include explicit spatial relationships and uncertainty statements (97.5% and 98.1% in the full set), supporting grounded reasoning and uncertainty-aware evaluation. The VQA distribution remains consistent across splits, with roughly 80/20 positive-negative polarity, about 60% freeform answers, and a full-set difficulty mix of 57.4/38.5/4.1% easy, medium, and hard questions. Additional dataset statistics can be found in Appendix A.

## 4. DisasterVL Tasks and Models

DisasterVL Tasks. We define five complementary tasks, used for both training and evaluation: factual scene captioning, visual question answering, incident reasoning, multiplechoice VQA, and region-level visual grounding. Across all tasks, models are expected to ground their outputs in visible image evidence and avoid unsupported inference.

T1: Disaster scene captioning: Given an image, the model generates a factual description grounded in visible content. The task uses the caption for training, supported by incident and uncertainty annotations. Captions emphasize disaster-relevant evidence such as infrastructure damage, environmental hazard, and response activity.

T2: Grounded VQA reasoning: Given an image and an open-ended question, the model generates an answer grounded in visible evidence. Each training QA example includes the question, answer, answer type, polarity, and supporting evidence, including checks for absent cues. This task tests whether models can answer disaster-relevant questions, reason from visual cues, and acknowledge when the image does not provide enough evidence.

T3: Grounded incident reasoning: Given an image, the model infers the likely disaster or incident type and justifies it using visual evidence. For training, we use incident and canonical labels, evidence statements, and uncertainty annotations. The goal is not only to classify the incident, but to explain why the label is supported and when the image does not justify a stronger claim.

Table 2. Summary statistics of DisasterVL for the full dataset, training set, and test set.
<table><tr><td></td><td colspan="7">Scale</td><td colspan="7">Annotation &amp; VQA Structure</td></tr><tr><td>Split</td><td>Images</td><td>CT</td><td>IT</td><td>QA</td><td>Q/I</td><td>MCQ</td><td>VG</td><td>Rel/Unc/Loc</td><td>CapLen</td><td>Pos/Neg</td><td>FF</td><td>E/M/H</td><td></td><td>Supp/Abs</td></tr><tr><td>Train</td><td>#1,197,351</td><td>29</td><td>12K</td><td>19.9M</td><td>16.64</td><td>2.6M</td><td></td><td>7.5M</td><td>97.5 / 98.0 / 69.04</td><td>88.78</td><td>79.8 / 20.2</td><td>59.6</td><td>57.6 / 38.4 / 4.0</td><td>1.56 / 0.34</td></tr><tr><td>Test</td><td>38,254</td><td>28</td><td>1,498</td><td>641K</td><td>16.76</td><td></td><td>85K</td><td>350K</td><td>99.0 / 99.7 / 59.70</td><td>83.78</td><td>79.1 / 20.9</td><td>62.7</td><td>51.9 / 42.4 / 5.8</td><td>1.55 / 0.35</td></tr><tr><td>Full</td><td>1.2M</td><td>29</td><td>12K</td><td>20.5M</td><td>16.65</td><td></td><td>2.7M</td><td>7.85M</td><td>|97.5 / 98.1 / 69.12</td><td>88.63</td><td>79.8 /20.2</td><td>59.7</td><td>57.4 / 38.5 / 4.1</td><td>1.56 / 0.34</td></tr></table>

CT = canonical type; IT = incident type (i.e., GPT-5 assigned “likely type"); QA = question-answer pairs; Q/I = questions per image; VG = visual grounding; Rel/Unc/Loc = relationships, uncertainties, and location assessment (%); CapLen = caption length; Pos/Neg = polarity (%); FF = free-form (%); E/M/H = Easy/Medium/Hard (%); Supp/Abs = support-evidence and absence-check items.

T4: Multiple-choice VQA: Given an image, a question, and candidate answers, the model selects the correct option. The task is derived from grounded QA and incident annotations, using plausible distractors to test whether models can distinguish visually supported answers from similar but unsupported alternatives. It provides a controlled evaluation of disaster-specific visual reasoning.

T5: Region-level visual grounding: Given an image, the model localizes disaster-relevant evidence with bounding boxes on a 0–1000 normalized grid, in two settings. In detection, it receives only the image and returns a labelled box for every disaster-relevant object and region; the targets are the annotated regions. In grounded captioning, it writes a detailed description and grounds its key phrases with boxes in the same output; the targets are the regions linked to phrases of the reference caption. Detection is scored on box placement only; grounded captioning additionally requires each box to carry the matching phrase (Sec. 5.2).

DisasterVL Models. We develop a family of disasterspecialized vision-language models, DisasterVLMs, built on the Qwen3.5 [47] model family. Qwen3.5 serves as a strong foundation due to its competitive multimodal reasoning capabilities, open availability, and support for high-resolution visual inputs — properties particularly relevant for the finegrained scene understanding required in crisis imagery. We train 2B, 4B, and 9B variants; the main results (Table 3) report the 4B and 9B models, and the 2B model is included in the scaling analysis (Appendix B).

Rather than full fine-tuning, we adopt parameter-efficient adaptation via LoRA [27], applying low-rank updates to both the language model and the multimodal projector while keeping the vision backbone frozen. This concentrates trainable capacity on cross-modal alignment and task-specific reasoning, mitigates the risk of catastrophic forgetting of pretrained visual representations, and substantially reduces memory requirements. We use LoRA rank 32, alpha 64, and dropout 0.05, with 16-bit mixed-precision training.

All models are trained jointly across the five DisasterVL tasks using instruction-formatted inputs, with task identity encoded via prompt templates. Training is conducted on 8 H200 GPUs for 2 epochs using a learning rate of $3 \times 1 0 ^ { - 5 }$ 100 warmup steps, and a maximum sequence length of 1024, with an effective batch size of 640 via gradient accumulation. Gradient checkpointing and FlashAttention-2 are enabled throughout for memory-efficient, stable training. At inference time, models are evaluated in a zero-shot prompted setting using the same instruction templates as training, without any task-specific decoding adjustments.

## 5. Experiments

## 5.1. Baselines and Metrics

We compare DisasterVLMs with ten zero-shot baselines from six model families: Molmo2 [16], InternVL3.5- 8B [51], Qwen3-VL (4B, 8B, 32B) [10], Gemma4-4B [20], the Qwen3.5 base models (2B, 4B, 9B) [47] from which DisasterVLMs are adapted, and the proprietary GPT-5 [45] as a strong closed-model reference. All baselines use the same inference prompts as DisasterVLM, without task-specific fine-tuning. Table 3 reports the 4B and 9B DisasterVLM variants.

For the generative tasks, we report ROUGE-L [33], which measures lexical overlap with the reference, and Sentence-T5 Sharpened Cosine Similarity (ST5-SCS) [15], which measures semantic similarity in embedding space and therefore credits valid answers that differ lexically from the reference. We also report GES, described below. Visual grounding (T5) is scored with box-level F1 (Sec. 5.2). For incident reasoning, we additionally report incident-type accuracy, and for multiple-choice VQA, exact answer accuracy.

Reference-grounded evaluation. For captioning, we additionally use Grounded Evidence Score (GES), which measures factual agreement with the reference caption. An LLM-based aligner labels each extracted response claim as matched, contradicted, or not in the reference. Reference precision is the fraction of claims matched to reference facts;

recall is the fraction of distinct reference facts covered by matched claims. Their harmonic mean gives GES. The strict score penalizes all unmatched claims, including visually correct details omitted from the reference, and therefore measures reference alignment rather than visual correctness alone. Appendix G gives the full definition, scoring variants, and aggregation protocol.

## 5.2. Experimental Results

Table 3 summarizes results on all five tasks (T1–T5). DisasterVLM-9B obtains the best score in every column except MCQ accuracy, where DisasterVLM-4B is best and the 9B model is on par with GPT-5. The margin over the strongest baseline depends on the metric: it is large for ROUGE-L on all open-ended tasks and for ST5-SCS on VQA and incident reasoning, and much smaller for GES and the grounding F1 scores. The 4B variant does not consistently outperform GPT-5: it trails on GES for captioning and VQA, on both grounding metrics, and on incident accuracy.

Disaster Scene Captioning. DisasterVLM-9B achieves the best captioning scores on all three metrics (40.51 ROUGE-L, 89.25 ST5-SCS, 69.4 GES), compared with 31.34, 87.67, and 64.8 for GPT-5, the strongest baseline on each. DisasterVLM-4B also exceeds every baseline on ROUGE-L (37.78) and ST5-SCS (88.53), but trails GPT-5 on GES (62.1 vs. 64.8), so its lexical advantage does not translate into higher factual agreement with the reference. More generally, the ROUGE-L gain of the 9B model (+9.17) is about twice its GES gain (+4.6), which suggests that part of the improvement reflects alignment with the reference caption style.

Grounded VQA-Reasoning. This task shows the largest gains on the reference-overlap metrics. DisasterVLM-9B reaches 44.18 ROUGE-L and 93.16 ST5-SCS, against 26.45 (GPT-5) and 81.20 (Qwen3-VL-32B) for the strongest baselines, and DisasterVLM-4B is close behind (42.72 / 92.07). On GES the gap narrows: the 9B model exceeds GPT-5 by 2.3 points (65.7 vs. 63.4), and the 4B model is below it (59.6). The baselines’ answers therefore differ from the references much more in wording than in GES, which scores claim support and answer correctness rather than surface overlap.

Grounded Incident Reasoning. This task requires both correct incident identification and a grounded visual justification. DisasterVLM-9B is best on all four metrics (51.50 ROUGE-L, 93.70 ST5-SCS, 73.6 GES, 81.52 accuracy), against 29.65, 82.43, 69.2, and 78.85 for GPT-5. DisasterVLM-4B exceeds GPT-5 on the three generative metrics, including GES (70.9 vs. 69.2), but not on incident accuracy (77.93 vs. 78.85). The accuracy gain of the 9B model (+2.67) is much smaller than its gains in explanation overlap, which suggests that adaptation mainly changes how incidents are justified rather than whether they are identified.

Multiple-Choice VQA. MCQ shows the smallest differences between adapted and general-purpose models. DisasterVLM-4B is best (78.22), 0.49 points above GPT-5 (77.73), and DisasterVLM-9B (77.70) is on par with GPT-5. Both adapted models improve substantially over their Qwen3.5 base models (64.32 and 67.37, gains of 13.90 and 10.33 points), but Qwen3-VL-32B (76.12) and Gemma4- 4B (75.79) are also within 2.5 points of the best result. The constrained answer format therefore appears to reduce the advantage of domain-specific supervision relative to openended generation. The 0.52-point difference between the 4B and 9B variants is too small to rank them without confidence intervals.

Visual Grounding. Both T5 settings are scored with boxlevel F1 under greedy one-to-one matching at a minimum IoU of 0.5. True positives, predicted boxes and reference boxes are summed over the test set before precision, recall and F1 are computed, so an image without predictions contributes all of its references as misses. Detection F1 (D-F1) matches predictions to the annotated regions in order of descending IoU. Box labels are not used, so D-F1 measures where the model places boxes, not what it calls them. Grounded-captioning F1 (GC-F1) uses one reference entry per phrase–region link of the reference caption. A prediction can match an entry only if its phrase also has a cosine similarity of at least 0.5 to the entry’s phrase, using all-MiniLM-L6- v2 sentence embeddings [43, 50]; candidates are accepted in order of phrase similarity, with IoU breaking ties. A correct box under the wrong phrase is therefore not credited, and a region described by several phrases must be grounded once under each of them. GC-F1 thus measures phrase grounding rather than localization alone. DisasterVLM-9B achieves the best grounding scores (64.1 D-F1, 59.2 GC-F1), ahead of GPT-5 (62.8 / 55.9), while DisasterVLM-4B is slightly below GPT-5 on both (61.6 / 54.7). GC-F1 is lower than D-F1 for every model, as expected from its additional phraseagreement requirement. Gemma4-4B grounding scores are not yet available.

## 5.3. Additional Analyses

Beyond aggregate task performance, we conduct a series of targeted analyses to better understand model behavior along several dimensions. In the main paper, we examine objectand scene-based term recognition; incident-level classification behavior and geographic generalization across regions are analyzed in Appendix C. Term recognition measures agreement with annotated vocabulary; it does not measure the region localization introduced in T5. We additionally provide scaling and data efficiency experiments as well as a deeper grounding analysis of object- and scene-level evidence across disaster types in Appendix B and Appendix C, respectively.

Table 3. Results on captioning, visual grounding, VQA reasoning, incident reasoning, and multiple-choice QA (MCQ). Column leaders are in bold. Dashes indicate missing scores.
<table><tr><td></td><td colspan="3">Captioning</td><td colspan="2">Visual Grounding</td><td colspan="3">VQA Reasoning</td><td colspan="4">Incident Reasoning</td><td>MCQ</td></tr><tr><td>Model</td><td>R-L</td><td>ST5</td><td>GES</td><td>D-F1 GC-F1</td><td>R-L</td><td>ST5</td><td>GES</td><td>R-L</td><td></td><td>ST5</td><td>GES</td><td>Acc</td><td>Acc</td></tr><tr><td>Molmo2 [16]</td><td>23.89</td><td>81.90</td><td>52.4</td><td>46.4</td><td>39.6 17.96</td><td>75.24</td><td>53.9</td><td></td><td>21.25</td><td>73.64</td><td>55.2</td><td>70.84</td><td>63.57</td></tr><tr><td>InternVL3.5-8B [51]</td><td>25.12</td><td>83.45</td><td>56.7</td><td>53.6</td><td>47.5</td><td>19.09</td><td>77.89</td><td>56.8</td><td>22.87</td><td>76.59</td><td>56.9</td><td>72.60</td><td>65.83</td></tr><tr><td>Qwen3-VL-4B [10]</td><td>23.98</td><td>83.20</td><td>54.5</td><td>47.2</td><td>42.5 17.80</td><td>75.83</td><td>54.1</td><td></td><td>19.78</td><td>75.59</td><td>55.6</td><td>69.92</td><td>62.44</td></tr><tr><td>Qwen3-VL-8B [10]</td><td>24.29</td><td>83.15</td><td>56.2</td><td>52.5</td><td>45.1 18.73</td><td>77.25</td><td></td><td>53.6</td><td>19.16</td><td>73.78</td><td>55.7</td><td>70.61</td><td>63.14</td></tr><tr><td>Qwen3.5-2B [47]</td><td>24.26</td><td>81.40</td><td>49.8</td><td>43.5</td><td>37.9 17.61</td><td>75.78</td><td></td><td>51.7</td><td>18.05</td><td>72.22</td><td>53.5</td><td>68.85</td><td>61.73</td></tr><tr><td>Qwen3.5-4B [47]</td><td>24.30</td><td>82.71</td><td>55.7</td><td>49.6</td><td>44.4 18.10</td><td>76.71</td><td></td><td>55.6</td><td>23.03</td><td>75.24</td><td>55.5</td><td>71.40</td><td>64.32</td></tr><tr><td>Qwen3.5-9B [47]</td><td>25.25</td><td>82.56</td><td>57.8</td><td>55.3</td><td>48.8 19.17</td><td>78.08</td><td></td><td>57.8</td><td>23.13</td><td>77.44</td><td>57.3</td><td>74.20</td><td>67.37</td></tr><tr><td>Gemma4-4B [20]</td><td>24.65</td><td>82.90</td><td>57.6</td><td></td><td>19.05</td><td>77.24</td><td></td><td>55.3</td><td>23.10</td><td>76.45</td><td>54.2</td><td>73.28</td><td>75.79</td></tr><tr><td>Qwen3-VL-32B [10]</td><td>25.82</td><td>87.04</td><td>61.5</td><td>58.5</td><td>52.7 25.45</td><td>81.20</td><td></td><td>61.7</td><td>27.21</td><td>80.10</td><td>62.4</td><td>75.24</td><td>76.12</td></tr><tr><td>GPT-5 [45]</td><td>31.34</td><td>87.67</td><td>64.8</td><td>62.8</td><td>55.9</td><td>26.45</td><td>80.76</td><td>63.4</td><td>29.65</td><td>82.43</td><td></td><td>69.2 78.85</td><td>77.73</td></tr><tr><td>DisasterVLM-4B</td><td>37.78</td><td>88.53</td><td>62.1</td><td>61.6</td><td>54.7</td><td>42.72</td><td>92.07</td><td>59.6</td><td>45.23</td><td>89.84</td><td>70.9 77.93</td><td></td><td>78.22</td></tr><tr><td>DisasterVLM-9B</td><td>40.51</td><td>89.25</td><td>69.4</td><td>64.1</td><td>59.2</td><td>44.18</td><td>93.16</td><td>65.7</td><td>51.50</td><td>93.70</td><td></td><td>73.681.52</td><td>77.70</td></tr><tr><td>∆ vs. best baseline</td><td></td><td>+9.17 +1.58</td><td>+4.6</td><td>+1.3</td><td>+3.3</td><td>+17.73 +11.96</td><td></td><td>+2.3</td><td>+21.85</td><td>+11.27</td><td></td><td>+4.4 +2.67</td><td>+0.49</td></tr></table>

R-L: ROUGE-L; ST5: ST5-SCS; GES: Grounded Evidence Score (App. G); Acc: accuracy; D-F1: detection box F1 at IoU ≥ 0.5; GC-F1: grounded-captioning F1, requiring IoU ≥ 0.5 and phrase agreement (Sec. 5.2). Where shown, cell shading is normalized per column; darker blue indicates higher performance. ∆: per-column difference between the best DisasterVLM variant and the best baseline.

Table 4. Results on object and scene classification
<table><tr><td>Model</td><td>Object Macro-F1</td><td>Scene Macro-F1</td></tr><tr><td>Qwen3.5-9B</td><td>73.6</td><td>76.1</td></tr><tr><td>DisasterVLM-4B</td><td>77.9</td><td>78.2</td></tr><tr><td>DisasterVLM-9B</td><td>81.5</td><td>82.85</td></tr></table>

Object- and Scene-Based Analysis. We evaluate object and scene term recognition by extracting disaster-relevant terms from reference captions and model predictions, then comparing the predicted terms against annotated references. This yields separate object-level and scene-level macro-F1 scores measuring the model’s ability to recognize relevant object and scene terms. As shown in Table 4, DisasterVLM-9B achieves the strongest performance on both metrics, outperforming both DisasterVLM-4B and the Qwen3.5-9B baseline. These results indicate improved recognition of disaster-relevant object and scene terms.

## 5.4. Discussion

DisasterVLM-9B outperforms all general-purpose baselines, including GPT-5, on every metric in T1–T5 except MCQ accuracy, where it is on par with GPT-5 (77.70 vs. 77.73) and DisasterVLM-4B is best (78.22). Margins are largest on lexical metrics for open-ended and reasoning tasks (up to +21.9 ROUGE-L) and much smaller on GES (+2.3 to +4.6), so part of the lexical gain likely reflects alignment with the reference style rather than additional visual evidence. Scale does not fully determine the ranking: the 4B variant slightly exceeds the 9B model on MCQ, although the gap is small. Incident-level and cross-continent analyses (Appendix C) show that errors are concentrated among visually similar damage types (earthquake, urban structural damage, road traffic accident) and that transferability is uneven across regions, indicating where additional data collection is most needed.

## 6. Conclusion

We introduced DisasterVL, a dataset and benchmark for disaster scene understanding comprising more than 1.2M images from 142 countries across 29 hazard types, with evidence-grounded captions, more than 23M question annotations, and a visual grounding layer that links caption phrases and relationship endpoints to image regions. Beyond identifying the visible hazard, the annotations record the supporting evidence, what remains uncertain, and what cannot be inferred from the image alone. DisasterVLM-9B outperforms all general-purpose baselines, including GPT-5, on captioning, visual grounding, grounded VQA, and incident reasoning, with the largest margins on reference-overlap metrics for reasoning tasks; on multiple-choice QA, adapted and strong general-purpose models perform similarly. Visually overlapping incident types, regional variation, and uneven cross-continent generalization remain open challenges for deployment. Future work will broaden geographic and rarehazard coverage, strengthen human validation, and extend DisasterVL toward geospatial grounding and event-level reasoning.

## References

[1] Aishwarya Agrawal, Dhruv Batra, Devi Parikh, and Aniruddha Kembhavi. Don’t just assume; look and answer: Overcoming priors for visual question answering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition (CVPR), pages 4971–4980, 2018. 3

[2] Aisha Al-Mohannadi, Ayisha Firoz, Yin Yang, Muhammad Imran, and Ferda Ofli. DisasterVQA: A visual question answering benchmark dataset for disaster scenes. In International AAAI Conference on Web and Social Media (ICWSM), pages 1–10, 2026. 3

[3] Firoj Alam, Muhammad Imran, and Ferda Ofli. Image4act: Online social media image processing for disaster response. In Proceedings of the 2017 IEEE/ACM international conference on advances in social networks analysis and mining (ASONAM), pages 601–604, 2017. 1

[4] Firoj Alam, Ferda Ofli, and Muhammad Imran. CrisisMMD: Multimodal twitter datasets from natural disasters. In International AAAI Conference on Web and Social Media (ICWSM), 2018. 1, 2, 3

[5] Firoj Alam, Umair Qazi, Muhammad Imran, and Ferda Ofli. HumAID: Human-annotated disaster incidents data from twitter with deep learning benchmarks. In Proceedings of the International AAAI Conference on Web and Social Media (ICWSM), pages 933–942, 2021. 1

[6] Firoj Alam, Tanvirul Alam, Ferda Ofli, and Muhammad Imran. Robust training of social media image classification models. IEEE Transactions on Computational Social Systems, 11(1):546–565, 2022. 1

[7] Firoj Alam, Tanvirul Alam, Md Arid Hasan, Abul Hasnat, Muhammad Imran, and Ferda Ofli. MEDIC: a multi-task learning dataset for disaster image classification. Neural Computing and Applications, 35(3):2609–2632, 2023. 1, 2, 3

[8] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. VQA: Visual question answering. In Proceedings of the IEEE international conference on computer vision (ICCV), pages 2425–2433, 2015. 3

[9] Amna Asif, Shaheen Khatoon, Md Maruf Hasan, Majed A Alshamari, Sherif Abdou, Khaled Mostafa Elsayed, and Mohsen Rashwan. Automatic analysis of social media images to identify disaster type and infer appropriate emergency response. Journal ofBig Data, 8(1):83, 2021. 1

[10] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 3, 6, 8

[11] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pages 65–72, 2005.

[12] Munzir Hubiba Bashir, Musheer Ahmad, Danish Raza Rizvi, and Ahmed A Abd El-Latif. Efficient cnn-based disaster events classification using uav-aided images for emergency response application. Neural Computing and Applications, 36(18):10599–10612, 2024. 1

[13] Derrick Bonafilia, Beth Tellman, Tyler Anderson, and Erica Issenberg. Sen1floods11: A georeferenced dataset to train and test deep learning flood algorithms for sentinel-1. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 210–211, 2020. 2, 3

[14] Chongyan Chen, Samreen Anjum, and Danna Gurari. VQA Therapy: Exploring answer differences by visually grounding answers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15315–15325, 2023. 3

[15] Zhenyuan Chen, Chenxi Wang, Ningyu Zhang, and Feng Zhang. RSCC: A large-scale remote sensing change caption dataset for disaster events. In The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2025. 6

[16] Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Mohammadreza Salehi, Rohun Tripathi, Sangho Lee, Zhongzheng Ren, Chris Dongjoo Kim, Yinuo Yang, et al. Molmo2: Open weights and data for vision-language models with video understanding and grounding. arXiv preprint arXiv:2601.10611, 2026. 6, 8

[17] Wenliang Dai, Junnan Li, Dongxu Li, et al. Instructblip: Towards general-purpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 1

[18] Muhammad Danish, Muhammad Akhtar Munir, Syed Roshaan Ali Shah, Kartik Kuckreja, Fahad Shahbaz Khan, Paolo Fraccaro, Alexandre Lacoste, and Salman Khan. GEOBench-VLM: Benchmarking vision-language models for geospatial tasks. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7132–7142, 2025. 3

[19] Google DeepMind. Gemini 3.1 Pro Model Card, 2026. 4, 5 [20] Google DeepMind. Gemma 4 model overview, 2026. 6, 8

[21] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pages 6904–6913, 2017. 1, 3

[22] Anisha Gunjal, Jihan Yin, and Erhan Bas. Detecting and preventing hallucinations in large vision language models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 18135–18143, 2024. 1

[23] Ritwik Gupta, Bryce Goodman, Nirav Patel, Ricky Hosfelt, Sandra Sajeev, Eric Heim, Jigar Doshi, Keane Lucas, Howie Choset, and Matthew Gaston. Creating xBD: A dataset for assessing building damage from satellite imagery. In Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 10–17, 2019. 1, 2, 3

[24] Tanu Gupta and Sudip Roy. DisasterRes-Net: A framework for analyzing social media images in disaster response. In-

ternational Journal ofDisaster Risk Reduction, 116:105119, 2025. 1

[25] Wei Han, Hantao Huang, and Tao Han. Finding the evidence: Localization-aware answer prediction for text visual question answering. In Proceedings of the 28th international conference on computational linguistics (COLING), pages 3118–3131, 2020. 3

[26] Paul Höhn, Konrad Heidler, Robert Behling, and Xiao Xiang Zhu. A spatio-temporal dataset for satellite-based landslide detection. Scientific Data, 12(1):1772, 2025. 2, 3

[27] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022. 6

[28] Yuan Hu, Jianlong Yuan, Congcong Wen, Xiaonan Lu, Yu Liu, and Xiang Li. RSGPT: A remote sensing vision language model and benchmark. ISPRS Journal of Photogrammetry and Remote Sensing, 224:272–286, 2025. 3

[29] Drew A Hudson and Christopher D Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 6700–6709, 2019. 3

[30] Muhammad Imran, Umair Qazi, Ferda Ofli, Steve Peterson, and Firoj Alam. AI for disaster rapid damage assessment from microblogs. In Proceedings ofthe AAAI conference on artificial intelligence (AAAI), pages 12517–12523, 2022. 1

[31] Kartik Kuckreja, Muhammad Sohail Danish, Muzammal Naseer, Abhijit Das, Salman Khan, and Fahad Shahbaz Khan. GeoChat: Grounded large vision-language model for remote sensing. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 27831– 27840, 2024. 3

[32] Ryan Lagerstrom, Yulia Arzhaeva, Piotr Szul, Oliver Obst, Robert Power, Bella Robinson, and Tomasz Bednarz. Image classification to support emergency situation awareness. Frontiers in Robotics and AI, 3:54, 2016. 1

[33] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004. 6

[34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 1

[35] Dilxat Muhtar, Zhenshi Li, Feng Gu, Xueliang Zhang, and Pengfeng Xiao. Lhrs-bot: Empowering remote sensing with vgi-enhanced large multimodal language model. In European Conference on Computer Vision (ECCV), pages 440–457, 2024. 3

[36] Dat T Nguyen, Ferda Ofli, Muhammad Imran, and Prasenjit Mitra. Damage assessment from social media imagery data during disasters. In IEEE/ACM international conference on advances in social networks analysis and mining (ASONAM), pages 569–576, 2017. 1

[37] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of

the Association for Computational Linguistics (ACL), pages 311–318, 2002.

[38] Catherine VL Pennington, Rémy Bossu, Ferda Ofli, Muhammad Imran, Umair Qazi, Julien Roch, and Vanessa J Banks. A near-real-time global landslide incident reporting tool demonstrator using social media and artificial intelligence. International Journal ofDisaster Risk Reduction, 77:103089, 2022. 1

[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning (ICML), pages 8748–8763, 2021. 3

[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. 1

[41] Maryam Rahnemoonfar, Tashnim Chowdhury, Argho Sarkar, Debvrat Varshney, Masoud Yari, and Robin Roberson Murphy. FloodNet: A high resolution aerial imagery dataset for post flood scene understanding. IEEE Access, 9:89644–89654, 2021. 2, 3

[42] Sainandan Ramakrishnan, Aishwarya Agrawal, and Stefan Lee. Overcoming language priors in visual question answering with adversarial regularization. In Advances in neural information processing systems (NeurIPS), 2018. 3

[43] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019. 7

[44] Yan Shu, Bin Ren, Zhitong Xiong, Danda Pani Paudel, Luc Van Gool, Begum Demir, Nicu Sebe, and Paolo Rota. Earth-Mind: Towards multi-granular and multi-sensor earth observation with large multimodal models. arXiv preprint arXiv:2506.01667, 2025. 3

[45] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. OpenAI GPT-5 system card. arXiv preprint arXiv:2601.03267, 2025. 3, 4, 6, 8

[46] Sagar Soni, Akshay Dudhane, Hiyam Debary, Mustansar Fiaz, Muhammad Akhtar Munir, Muhammad Sohail Danish, Paolo Fraccaro, Campbell D Watson, Levente J Klein, Fahad Shahbaz Khan, et al. EarthDial: Turning multi-sensory earth observations to interactive dialogues. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 14303–14313, 2025. 3

[47] Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, 2026. 6, 8

[48] Junjue Wang, Weihao Xuan, Heli Qi, Zhihao Liu, Kunyi Liu, Yuhan Wu, Hongruixuan Chen, Jian Song, Junshi Xia, Zhuo Zheng, and Naoto Yokoya. DisasterM3: A remote sensing vision-language dataset for disaster damage assessment and response. In The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2025. 2, 3

[49] Jiacong Wang, Zijian Kang, Haochen Wang, LiangXiao, Ya Wang, Jiawen Li, Bohong Wu, Ran Jiao, Haiyong Jiang, ChaoFeng, and Jun Xiao. VGR: Visual grounded reasoning. In The Fourteenth International Conference on Learning Representations, 2026. 3

[50] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 7

[51] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 6, 8

[52] Ethan Weber, Dim P. Papadopoulos, Agata Lapedriza, Ferda Ofli, Muhammad Imran, and Antonio Torralba. Incidents1M: A Large-Scale Dataset of Images With Natural Disasters, Damage, and Incidents. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4768–4781, 2023. 1, 2, 3

[53] Yang Zhan, Zhitong Xiong, and Yuan Yuan. SkyEyeGPT: Unifying remote sensing vision-language tasks via instruction tuning with large language model. ISPRS Journal of Photogrammetry and Remote Sensing, 221:64–77, 2025. 3

[54] Wei Zhang, Miaoxin Cai, Tong Zhang, Yin Zhuang, and Xuerui Mao. EarthGPT: A universal multimodal large language model for multisensor image comprehension in remote sensing domain. IEEE Transactions on Geoscience and Remote Sensing, 62:1–20, 2024. 3

## Supplementary Material

## A. Additional Dataset Statistics

Table 5 and Fig 3 provide a more detailed statistical summary of DisasterVL for the full dataset, train set, and the final diversity-aware test set. Beyond the dataset’s overall scale, the table highlights the richness of the annotation structure, including relationships, uncertainty fields, visible text, location assessment, and evidence-grounded VQA support. It also summarizes the distribution of QA types and difficulty levels, showing that the test set remains broadly aligned with the full dataset while being slightly more challenging overall. These statistics complement the main paper summary by documenting the scope, density, and structural properties of the released benchmark.

## A.1. Object and Scene Term Distributions

Figure 4 provides a lexical summary of the most frequent object and scene terms appearing in DisasterVL annotations. The left panel shows the top 100 disaster-related object terms, while the right panel shows the top 100 scene terms, both ranked by occurrence on a logarithmic scale. These distributions reflect the dense and operationally oriented nature of the dataset: common object/attribute terms such as damage, emergency,fire, water, smoke, and debris indicate that annotations consistently capture visible hazards, affected infrastructure, response-related entities, and physical evidence of disaster impact. Likewise, frequent scene terms such as flooding, street, structural collapse, daylight, terrain, and landslide show that the dataset also encodes broader environmental and situational context rather than isolated object mentions alone.

The long-tailed shape of both distributions is also informative. A relatively small number of highly recurrent terms accounts for a large portion of the corpus, while many lower-frequency terms capture more specific disaster cues, terrain conditions, and operational contexts. This pattern is expected in a large-scale real-world disaster dataset and supports the claim that DisasterVL contains both broad coverage of common hazards and finer-grained vocabulary for more specialized scenarios. Together, these statistics provide an additional view of annotation richness beyond image counts and QA totals, showing that the dataset spans a wide spectrum of disaster objects, environments, and scene-level descriptors.

## A.2. Question Category Distribution

Figure 5 summarizes the distribution of VQA categories in DisasterVL on a logarithmic scale. The most frequent categories are damage assessment, visual evidence recognition, and uncertainty and absence reasoning, reflecting the dataset’s emphasis on operational disaster understanding rather than generic image QA. These categories directly support tasks that require models to identify visible damage, ground answers in scene evidence, and distinguish observable facts from unsupported inference.

The distribution also shows broad coverage across incident identification, human activity and response, location and geography, environmental hazards, infrastructure access, and spatial relationships. Lower-frequency categories such as text and signage, counting and quantities, temporal conditions, and resource needs and relief capture more specialized dimensions of reasoning.

## B. Scaling Behavior and Data Efficiency

Figure 6 analyzes how performance varies with both model scale and training-data scale. The left panel compares ROUGE-L across DisasterVL variants. Overall, performance improves with model size, but the trend is not strictly monotonic at smaller scales. Captioning performance increases from 37.63 for the 2B model to 38.12 for the 4B model, and rises to 39.16 for the 9B model. This suggests that larger models are generally beneficial, although gains are shaped by optimization and model-family effects rather than size alone. Notably, the 9B model gives the strongest captioning result, indicating that the full-capacity variant benefits most from adaptation to the disaster domain.

The right panel examines data efficiency by varying the amount of training data used for adaptation. For both captioning and reasoning VQA, performance improves steadily as more data is introduced. Captioning ROUGE-L increases from 38.23 at 10% of training data to 40.63 at 80%, while reasoning VQA rises from 41.28 to 44.95 over the same range. The gains are especially pronounced in the low-data regime, with substantial improvement between 10% and 40%, after which the curves continue to improve more gradually. This pattern indicates that DisasterVL provides useful supervision even at relatively small fractions of the full training set, while still offering clear benefits at larger scale.

Taken together, these results highlight two main points. First, adapting general-purpose VLMs on DisasterVL yields consistent gains in disaster-oriented generation and reasoning. Second, the benchmark exhibits meaningful scaling behavior with respect to both model capacity and trainingdata size, making it suitable not only for evaluation but also for studying data efficiency and adaptation dynamics in crisis-oriented multimodal learning.

## C. Additional Object and Scene Term Analysis

The analyses in this section concern recovery of object and scene terms and agreement with their annotated states. They do not evaluate region boxes, phrase–region links, or relationship endpoint localization. The evaluation protocol for T5 is given in Appendix F.

To further characterize the grounding behavior of DisasterVL, we analyze object- and scene-level evidence across disaster types. Rather than relying only on aggregate task metrics, these results examine which terms are most reliably present for each disaster category and how grounding quality differs between object-centric and scene-centric descriptions. This analysis is useful because many disaster types are defined not by a single salient object, but by combinations of local evidence (e.g., debris, smoke, emergency equipment)

Table 5. Detailed statistics of DisasterVL for the full dataset, training set, and test set.
<table><tr><td>Statistic</td><td>Full dataset</td><td>Train set</td><td>Test set</td></tr><tr><td>Total images</td><td>1,235,605</td><td>1,197,351</td><td>38,254</td></tr><tr><td>Canonical categories</td><td>29</td><td>29</td><td>28</td></tr><tr><td>Total QA pairs</td><td>20,568,496</td><td>19,927,329</td><td>641,167</td></tr><tr><td>Average QA pairs / image</td><td>16.65</td><td>16.64</td><td>16.76</td></tr><tr><td>Median QA pairs / image</td><td>17</td><td>17</td><td>17</td></tr><tr><td>Total MCQs</td><td>2,712,660</td><td>2,626,958</td><td>85,702</td></tr><tr><td>Average MCQs / image</td><td>2.20</td><td>2.19</td><td>2.24</td></tr><tr><td>Images with relationships (%)</td><td>97.5</td><td>97.5</td><td>99.0</td></tr><tr><td>Images with uncertainties (%)</td><td>98.1</td><td>98.0</td><td>99.7</td></tr><tr><td>Images with visible text (%)</td><td>51.6</td><td>51.8</td><td>45.4</td></tr><tr><td>Images with known location assessment (%)</td><td>69.12</td><td>69.42</td><td>59.7</td></tr><tr><td>Images with collection metadata (%)</td><td>99.7</td><td>99.7</td><td>100.0</td></tr><tr><td>Average caption length (words)</td><td>88.63</td><td>88.78</td><td>83.78</td></tr><tr><td>Positive QA pairs (%)</td><td>79.8</td><td>79.8</td><td>79.1</td></tr><tr><td>Negative QA pairs (%)</td><td>20.2</td><td>20.2</td><td>20.9</td></tr><tr><td>Free-form QA pairs (%)</td><td>59.7</td><td>59.6</td><td>62.7</td></tr><tr><td>Yes/No QA pairs (%)</td><td>38.6</td><td>38.7</td><td>35.6</td></tr><tr><td>Easy QA pairs (%)</td><td>57.4</td><td>57.6</td><td>51.9</td></tr><tr><td>Medium QA pairs (%)</td><td>38.5</td><td>38.4</td><td>42.4</td></tr><tr><td>Hard QA pairs (%)</td><td>4.1</td><td>4.0</td><td>5.8</td></tr><tr><td>Avg. support evidence items / QA</td><td>1.56</td><td>1.56</td><td>1.55</td></tr><tr><td>Avg. absence-check items / QA</td><td>0.34</td><td>0.34</td><td>0.35</td></tr></table>

![](images/6d55f2afb081f7cc325fb06a99fe69f67ec5cf01238bf36f460c38f7f1951d9d.jpg)

![](images/e23c75d5b65790fa7cf416e140978f2172adba41cf0aaf996db65115576827c2.jpg)  
Figure 3. Category distribution of DisasterVL across the full set and test set. The full set follows the naturally imbalanced distribution of collected crisis imagery, with categories such as Urban / Structural Damage and Flooding appearing most frequently. In contrast, the test split is constructed to be more balanced across major disaster types, improving the reliability of evaluation across categories. For readability, low-frequency categories are grouped under Others. For better insight and comparison, the Unknown and Not-Crisis categories are removed from the full set distribution.

![](images/540af00b359cc323fed9183279633c05cb316bb0d363f0ed7bc7e432cbc438c4.jpg)  
Figure 4. Left: Top 100 disaster-related object terms by occurrence in DisasterVL records. Frequent terms such as damage, emergency, fire, water, smoke, and debris reflect the strong presence of hazard cues, impacted infrastructure, and response-related entities in the annotations. Right: Top 100 disaster scene terms by occurrence. Common terms such as flooding, street, structural collapse, daylight, and terrain highlight the dataset’s emphasis on broader environmental and situational context. Both panels use a logarithmic x-axis to illustrate the naturally long-tailed distribution of the annotation vocabulary in DisasterVL.

and broader scene context (e.g., flooding, structural collapse, terrain, or weather conditions).

Figures 7 and 8 show the true-positive rate of the top

50 object and scene terms for each disaster type. Several patterns are apparent. First, frequent and visually distinctive hazards such as wildfire, flood, storm/severe weather, and hurricane/cyclone/typhoon tend to have broad bands of high true-positive rates, indicating that the corresponding terms are consistently recognized across many annotations. Second, some disaster types are grounded more selectively, with only a smaller subset of terms achieving high recall. This is especially visible for categories such as medical emergency, protest/riot/civil unrest, and relief aid distribution, where fewer highly reliable terms dominate the representation. Third, the heatmaps also show that grounding is distributed across both concrete object cues and more global scene descriptors. For example, disaster categories such as urban structural damage, landslide/mudslide/avalanche, and winter storm depend not only on isolated object mentions, but also on scene-level environmental context.

QA Category Distribution  
![](images/176ea46dd5e57211adb31638a1df3db94dbf4b03beda41bc2b92df9ba3a5b5d7.jpg)  
Figure 5. Distribution of VQA categories in DisasterVL. Counts are shown on a logarithmic scale. The most frequent categories focus on damage assessment, visual evidence recognition, and reasoning about uncertainty or absence, while lower-frequency categories cover more specialized reasoning dimensions such as text recognition, counting, temporal conditions, and resource needs.

![](images/4365ba5d59e6603e0be716b92c2269c4c7ccc9fe23b71b6ac0b7a5660aec0b71.jpg)  
(a)

![](images/6e09d3dd802eb978a5a5efd5dfab958c54c19b03da41fcb6f1e19c5720dd7a56.jpg)  
(b)  
Figure 6. Left: ROUGE-L across DisasterVL model variants, showing the effect of model scaling on captioning performance. Right: ROUGE-L as a function of training-data scale for captioning and reasoning VQA, illustrating that both tasks benefit consistently from larger portions of DisasterVL.

Figure 9 summarizes this contrast more directly by comparing object and scene grounding performance for the top disaster types using present F1 and state-agreement scores. A consistent pattern emerges: scene grounding is often slightly stronger than object grounding for disaster types defined primarily by environmental or contextual conditions, whereas object grounding remains competitive or stronger for categories with distinctive physical entities or localized evidence. For instance, wildfire, winter storm, and storm/severe weather show high performance in both views, suggesting that DisasterVL captures both local evidence and global context well for these hazards. In contrast, categories such as protest/riot/civil unrest, road traffic accident, and medical emergency remain more challenging, as expected, because their visual signatures are more variable and often require finer-grained interpretation. Overall, these results support the claim that DisasterVL is grounded not only in objectlevel recognition but also in scene-level understanding and that both forms of grounding contribute to robust disaster reasoning.

Incident-Based Analysis. The confusion matrix in Figure 10 (left) shows strong recall for visually distinctive categories such as wildfire andflood, while the main errors occur among structurally related classes, particularly earthquake, urban structural damage, and road traffic accident. This suggests that the remaining challenges stem less from coarse hazard recognition and more from fine-grained distinctions between visually overlapping damage scenarios.

Region-Based Analysis. The continent-wise breakdown in Figure 10 (right) reveals that generalization varies substantially across regions and incident types. Hazards with broad visual consistency such as flood and landslide/mudslide/avalanche remain strong across most continents, while region-specific or data-sparse categories show greater variance. South America is the most challenging region overall: wildfire,fire (urban/industrial), and storm / severe weather all drop sharply compared to other continents, suggesting weaker training coverage or stronger domain shift. Among the remaining regions, Europe and Oceania achieve the strongest overall performance, with near-perfect F1 on several categories. North America is consistently strong but relatively weaker on urban / structural damage, while Asia shows robust performance overall with the exception of earthquake, where visually diverse damage manifestations remain challenging.

## C.1. Example from Training Sample Annotation

The example below illustrates how a single DisasterVL record is transformed into multiple training tasks. Starting from the structured image annotation, the sample contains a dense caption, incident and location assessments, relational evidence, uncertainty fields, VQA pairs, MCQ supervision, and task-specific instruction–answer pairs. This format allows the same image to support complementary supervision for captioning, grounded VQA reasoning, incident reasoning, and multiple-choice answer selection.

## Example training sample used in DisasterVL

```jsonl
{
"image_id": "example_image_001",
"image_path": "images/example_image_001.jpg",
"captioning_result": {
"caption": "Urban street scene in daylight with a
multi-story residential building showing severe
structural damage. Several exterior walls and
balconies have collapsed, exposing interiors and
concrete slab edges. Rubble is piled along the
sidewalk, and an excavator is positioned next to
the damaged facade.",
"incident_assessment": {
"likely_type": "structural collapse",
"canonical_type": "urban_structural_damage",
"confidence": "high",
"evidence": "Multiple floors have missing
exterior walls and balconies, exposed concrete,
and rubble at the base of the building."
},
"location_assessment": {
"possible_location": "unknown",
"confidence": "low",
"evidence": "No readable signs, landmarks, or
license plates are visible."
},
"relationships": [
"The excavator is positioned on the street
adjacent to the damaged building.",
"Rubble is accumulated at the base of the
building along the sidewalk.",
"Adjacent buildings appear largely intact
compared with the damaged structure."
],
"visible_text": [
"no-text"
],
"uncertainties": [
"The exact cause of the structural damage is
not clear from the image.",
"It cannot be confirmed whether the excavator
is performing demolition or debris removal."
]
},
"vqa_result": {
"qa_pairs": [
{
"qa_id": "example_image_001_p01",
"question": "What type of hazard is depicted
in the image?",
"answer": "The image shows a structural
collapse hazard, evidenced by missing exterior
walls, exposed concrete slabs, and rubble piled
at the base of the building.",
"answer_type": "free_form",
"difficulty": "medium",
"polarity": "positive",
"source_fields": [
"caption",
"incident_assessment",
"relationships"
],
"support": {
"evidence": [
```

![](images/c4640016e7e2ed00298b108f2000c07d55e8c36522b76c9fbaa9d47b6ec53aad.jpg)  
Figure 7. Object true-positive rate by disaster type for the top 50 object terms. Darker cells indicate stronger grounding of the corresponding object term for that disaster category. The heatmap shows that some hazards, such as wildfire,flood, and storm/severe weather, are associated with broad sets of consistently detected object cues, whereas other categories rely on a narrower set of high-signal terms.

![](images/253a168df522c036b25649737c4f42a53cfb8abcdb4b0364ccd9a9d481b638e6.jpg)  
Figure 8. Scene true-positive rate by disaster type for the top 50 scene terms. Compared with object terms, scene terms capture broader environmental and situational context, including terrain, weather, and structural conditions. The results show that many disaster types in DisasterVL are strongly grounded in global scene structure rather than in isolated objects alone.

![](images/d4c44f007695e334606ebd5d04e2a688bb1a756014afcab8dfd956751b49bacb.jpg)  
Figure 9. Comparison of object and scene grounding performance by disaster type. The left panel reports present F1, while the right panel reports state-agreement scores. Scene grounding is often slightly stronger for context-defined hazards, whereas object grounding remains important for disasters with more localized visual evidence. Together, the two panels show that DisasterVL supports both object-centric and scene-centric grounding across a wide range of disaster categories.

![](images/11a23ae58d0338febecc799cf714661fa83ef1f1bb550fe7804ff2d5cad83fcc.jpg)

![](images/f1321f8f687e4859fbda3e2fa0977c5f2c19859fc7331459474bd9981f5803ac.jpg)  
Figure 10. Left: Normalized confusion matrix for incident classification, showing strong F1-score measure for visually distinctive hazards and larger confusions among structurally similar damage categories. Right: Cross-continent F1-score by incident type, highlighting consistent transfer for common hazards such as flood and wildfire, but more variable performance for region-sensitive or sparsely represented categories.

"several exterior walls and balconies have collapsed",

"interiors and concrete slab edges are exposed", "rubble is piled along the sidewalk"

],   
"absence\_check": []   
}   
},   
"qa\_id": "example\_image\_001\_p02",   
"question": "Is the exact cause of the   
building damage visible in the image?",   
"answer": "No. The image shows severe   
structural damage, but it does not provide   
enough evidence to determine whether the cause   
was an earthquake, demolition, explosion, or   
another event.",   
"answer\_type": "yes\_no",   
"difficulty": "hard",   
"polarity": "negative",   
"source\_fields": [   
"caption",   
"uncertainties"   
],   
"support": {   
"evidence": [   
"severe structural damage is visible"   
],   
"absence\_check": [   
"the exact cause of the structural damage   
is not clear from the image"   
},   
"mcq\_result": {   
"mcq\_pairs": [   
"question": "What is the main visible hazard   
in the image?",   
"options": {   
"A": "Floodwater covering a roadway",   
"B": "Structural collapse of a building",   
"C": "Wildfire spreading through vegetation   
"D": "A traffic accident involving vehicles   
},   
"correct\_label": "B",   
"correct\_answer": "Structural collapse of a   
building",   
"difficulty": "medium",   
"source\_fields": [   
"caption",   
"incident\_assessment"   
]   
}   
},   
"training\_tasks": [   
"task": "caption",   
"question": "Describe the main scene, objects,   
and activities visible in this image.",   
"answer": "Urban street scene in daylight with   
a multi-story residential building showing   
severe structural damage. Several exterior walls   
and balconies have collapsed, exposing   
interiors and concrete slab edges. Rubble is   
piled along the sidewalk, and an excavator is   
positioned next to the damaged facade."   
},   
{   
"task": "reasoning\_vqa",   
"question": "Answer the question with grounded   
reasoning based only on visible evidence.\n\   
nQuestion: What type of hazard is depicted in   
the image?",   
"answer": "Final answer: structural collapse   
hazard\nObservations:\n- Exterior walls and   
balconies are missing from several floors.\n-

```jsonl
Concrete slab edges and interiors are exposed.\n-
Rubble is piled along the sidewalk.\nEvidence:\
n- The damaged facade and debris support the
interpretation of structural collapse.\
nUncertainty:\n- The exact cause of the collapse
is not visible.\nConclusion: The image supports
a structural collapse hazard."
},
{
"task": "incident_reasoning",
"question": "Classify the incident shown in the
image. Start with ‘Final answer: incident_type:
<label>‘, then explain the visible evidence and
limitations.",
"answer": "Final answer: incident_type:
urban_structural_damage\nObservations:\n- A
multi-story building shows severe facade and
balcony collapse.\n- Rubble is visible at the
base of the structure.\nEvidence:\n- Exposed
concrete slabs and damaged exterior walls
indicate structural failure.\nUncertainty:\n-
The cause of the damage cannot be determined
from the image alone.\nConclusion: The visible
evidence supports urban_structural_damage."
},
{
"task": "mcq",
"question": "What is the main visible hazard in
the image?\nA. Floodwater covering a roadway\nB.
Structural collapse of a building\nC. Wildfire
spreading through vegetation\nD. A traffic
accident involving vehicles\n\nAnswer with the
correct option letter and answer text.",
"answer": "B. Structural collapse of a building
II
}
]
}
```

## D. Reasoning Tasks Samples

The examples below illustrate the target output style used for the reasoning-oriented tasks in DisasterVL. They cover both reasoning\_vqa, where the model answers a question using grounded observations, evidence, and uncertainty, and disaster\_reasoning, where the model predicts a disaster label and justifies it from visible scene cues. These samples are included to clarify the intended structure of reasoning outputs and to show how the dataset encourages explicit grounding rather than unsupported inference.

## Example reasoning outputs used in DisasterVL Training Example reasoning outputs used in DisasterVL Trainin

```markdown
# Reasoning VQA Samples
Sample 1
Query:
Answer the question with grounded reasoning based
only on visible evidence.
Question: What type of hazard is depicted in the
image?
Answer:
Final answer: The image shows a structural collapse
hazard.
Observations:
- Several floors of a building have missing exterior
walls and balconies.
```

Sample 2   
Query:   
Determine the disaster type shown in the image. Start   
with ‘Final answer: disaster\_type: <label>‘ and   
support your answer using only visible evidence.

Interior rooms and concrete slab edges are exposed.   
Rubble is piled at the base of the building.   
Heavy machinery is positioned near the damaged   
facade.   
Evidence:   
- Large sections of the building facade are damaged   
or missing.   
- Debris is visible along the sidewalk and street.   
Uncertainty:   
The exact cause of the collapse is not visible from   
the image alone.   
Conclusion: The visible evidence supports structural   
collapse or urban structural damage.   
Sample 2   
Query:   
Answer the question using only what can be visually   
supported in the image.   
Question: What visual indicators show the presence of   
damage in the image?   
Answer:   
Final answer: Damage is visible through broken   
concrete, exposed rebar, rubble, and unstable   
collapsed sections.   
Observations:   
Broken concrete slabs are stacked at angles.   
Exposed reinforcing bars protrude from fractured   
surfaces.   
People are moving through a narrow debris-filled   
passage.   
Evidence:   
- Heavy rubble and fractured building material are   
directly visible.   
Uncertainty:   
- The full extent of damage outside the visible area   
is unknown.   
Conclusion: The scene shows severe structural damage   
based on visible debris and exposed construction   
materials.

##############################

\# disaster Reasoning Samples

Sample 1   
Query:   
What type of disaster is shown in this image? Start   
with ‘Final answer: disaster\_type: <label>‘,   
then justify it using grounded visual evidence.   
Answer:   
Final answer: disaster\_type: urban\_structural\_damage   
Observations:   
- Multiple floors of a building are missing exterior   
walls and balconies.   
- Exposed concrete slabs and interior rooms are   
visible.   
- Rubble is accumulated at the base of the building.   
Evidence:   
- The image shows a damaged urban building with   
collapse-related debris.   
Uncertainty:   
- The exact cause of the damage is not visually   
confirmed.   
Conclusion: The visible scene is best classified as   
urban\_structural\_damage.

Answer:   
Final answer: disaster\_type: urban\_structural\_damage   
Observations:   
Broken concrete slabs and exposed rebar are visible.   
People are navigating through a collapsed   
structural void.   
- Heavy rubble surrounds the passage.   
Evidence:   
The scene contains collapsed built infrastructure   
rather than floodwater, fire, or storm damage.   
Uncertainty:   
- The precise trigger of the collapse is unknown.   
Conclusion: The visible evidence supports   
urban\_structural\_damage.

## E. Dataset Generation Prompts

## E.1. Prompt Template for Captioning, Disaster Type and Evidence Generation

The prompt template below is used to generate the structured captioning annotations in DisasterVL. It instructs the model to produce dense, factual image descriptions centered on visible infrastructure, environment, property conditions, and other directly observable scene elements, while explicitly avoiding unsupported inference. In addition to the main caption, the prompt requires structured outputs for disaster assessment, location assessment, spatial relationships, visible text, and uncertainty statements. This design encourages grounded scene understanding and makes the resulting annotations suitable for downstream reasoning and VQA generation.

## Caption-generation prompt used in DisasterVL

SYSTEM\_PROMPT = """   
You are an image-description assistant for emergency   
management documentation.   
Your job is to produce a dense, factual caption of   
what is VISIBLE, focusing on   
infrastructure, environment, and property conditions.   
Safety constraints (important):   
- Do NOT describe injuries, death, blood, gore, or   
violence in detail.   
- If people are present, describe only neutral   
observable facts (e.g., ’two people standing’),   
and do not speculate about harm, roles, or intent.   
- Do NOT identify individuals or infer nationality/   
identity.   
Your goals:   
1. Describe exactly what is VISIBLE in the image.   
2. Capture as many concrete and observable details as   
possible.   
3. Avoid hallucinations: do NOT guess or infer causes   
, identities, or hidden information.   
4. Identify the likely crisis or disaster category   
ONLY if the visual evidence strongly supports it   
(e.g., collapsed buildings, flooding, wildfire).   
- If uncertain, express it with probability levels   
such as ‘‘possible’’, ‘‘likely’’, or ‘‘uncertain   
’’.   
- Never infer a disaster type without clear visual   
cues.

```jsonl
5. Detect the possible location ONLY if visual
evidence exists (e.g., readable signs, language,
architecture style, vehicle license formats).
Always give a probability rating of your confidence
(e.g., ‘‘low confidence’’, ‘‘moderate
confidence’’).
If no reliable geographic cues exist, say 11
location cannot be determined’’.
General rules:
Only describe what is directly visible.
- If uncertain, explicitly mark details as ‘‘
uncertain’’, ‘‘possibly’’, or ‘‘low confidence’’.
Do NOT infer:
unseen causes or backstories,
<sub>*</sub> identity or nationality of people,
<sub>*</sub> exact location unless signs or landmarks are
present,
<sub>*</sub> time of day unless visually obvious,
emotions or intentions.
For visible text:
<sub>*</sub> Transcribe EXACTLY if fully readable.
If partially readable, include only the certain
parts and mark the rest as ‘‘[unclear]’’.
<sub>*</sub> If unreadable, state ‘‘unreadable text’’.
Caption style:
Objective, neutral, highly detailed.
Use precise nouns and adjectives.
Cover: subjects, objects, infrastructure,
environmental conditions, spatial layout,
visible damage, and interactions.
Avoid interpretive or narrative language.
Focus on comprehensive visual content extraction.
Detailed object descriptions
Output JSON (strict):
"caption": "<dense factual caption>",
"disaster_assessment": {
"likely_type": "<flood | fire_damage |
wind_damage | structural_damage | landslide |
unknown | none>",
"confidence": "<low | moderate | high>",
"evidence": "<brief evidence based only on
visible features>"
},
"location_assessment": {
"possible_location": "<country/region/city or ’
unknown’>",
"confidence": "<low | moderate | high>",
"evidence": "<visual cues, if any>"
},
"relationships": ["<short factual spatial/
interactions>"],
"visible_text": [
{"raw":"<exact or [unclear] or unreadable text
>","position":"<where>","type":"<sign/label/etc
>"}
],
"uncertainties": ["<ambiguous items you cannot
confirm>"]
}
If any category has no valid information, return an
empty list or null fields as appropriate. Never
invent or guess details beyond visual evidence.
""".strip()
```

## E.2. Prompt Template for VQA Generation

The prompt template below is used to generate the visual question answering annotations in DisasterVL from the structured captioning output. It is designed to produce a crisisoriented VQA set that goes beyond object recognition by enforcing coverage over hazard identification, visible damage, spatial exposure, infrastructure, response presence, environmental conditions, uncertainty, and hallucination resistance. The template also constrains answer style, polarity, and evidence support so that the resulting QA pairs remain grounded in the available visual information while preserving a consistent schema for downstream evaluation.

## VQA-generation prompt used in DisasterVL

SYSTEM\_PROMPT = """You are a dataset creator for   
INTERNATIONAL crisis and disaster visual   
question answering (VQA),   
aligned with the needs of global emergency management   
and humanitarian agencies   
(e.g., UN OCHA, IFRC, EU Civil Protection, UN-SPIDER,   
WFP).   
You ONLY know what is written in the provided JSON   
annotation.   
You MUST NOT invent details, impacts, actions,   
timelines, or outcomes beyond this JSON.   
PRIMARY OBJECTIVE:   
Create an internationally applicable Crisis-VQA   
benchmark that evaluates:   
disaster and hazard awareness,   
visual evidence grounding,   
spatial exposure and proximity reasoning,   
critical infrastructure recognition,   
emergency or humanitarian presence awareness,   
uncertainty handling,   
hallucination resistance.   
MANDATORY CRISIS COVERAGE:   
You MUST generate questions covering ALL sections   
below:   
H1: Hazard Identification Classification (2   
questions)   
H2: Visual Threat or Damage Indicators (2-3 questions   
)   
H3: Spatial Exposure \& Proximity (2-3 questions)   
H4: Critical Infrastructure \& Lifeline Systems (1-2   
questions)   
H5: Emergency or Humanitarian Response Presence (1-2   
questions)   
H6: Environmental \& Secondary Conditions (1-2   
questions)   
H7: Uncertainty \& Information Gaps (2 questions)   
H8: Hallucination Traps (0-5 NEGATIVE yes/no   
questions)   
These sections reflect common assessment practices   
across international emergency   
and humanitarian organizations.   
COVERAGE DISTRIBUTION CONSTRAINT:   
The QA pairs MUST satisfy the following:   
H1\_hazard: exactly 2   
H2\_visual\_threat: 2-3   
H3\_spatial\_exposure: 2-3   
H4\_infrastructure: 1-2   
H5\_response\_presence: 1-2   
一 H6\_environment: 1-2   
H7\_uncertainty: exactly 2   
H8\_hallucination: 0-5 (negative only)   
SECTION LABELING REQUIREMENT:   
Each QA pair MUST include a field:   
"coverage\_section": "<one of: H1\_hazard |   
H2\_visual\_threat | H3\_spatial\_exposure |   
H4\_infrastructure | H5\_response\_presence |

H6\_environment | H7\_uncertainty   
H8\_hallucination>"   
Every positive question must be assigned to exactly   
ONE of H1-H7.   
Every negative question must be assigned to   
H8\_hallucination.   
The distribution must satisfy the mandatory   
coverage counts.   
NATURAL ANSWER STYLE (NO META-REFERENCES):   
- Answers must read like natural responses about the   
image.   
NEVER mention: annotation, caption, JSON, fields,   
or any field names   
(e.g., "uncertainties", "disaster\_assessment").   
NEVER say: "cannot be determined from the   
annotation/caption"   
or "the annotation notes/states".   
If unknown, use natural image-based phrasing only:   
"Not visible in the image.", "Not shown here.",   
"Not clear from this view.", "Cannot be confirmed   
from what’s visible."   
EVIDENCE VS ANSWER:   
Grounding belongs ONLY in support.evidence /   
support.absence\_check.   
The answer text must NOT mention sources or the   
existence of an annotation.   
ANSWER REQUIREMENTS:   
POSITIVE: 12-15 total   
<sub>\*</sub> 6-9 LONG answers (2-4 sentences, 40-90 words)   
<sub>\*</sub> Remaining SHORT answers (1-8 words)   
NEGATIVE: answer must be exactly "No"   
LONG ANSWERS MUST:   
- Explain WHY the image supports a crisis or disaster   
interpretation   
Reference visual cues and spatial relationships   
Explicitly state limitations or uncertainty when   
relevant   
Avoid agency-specific jargon   
PROHIBITIONS:   
- Do NOT speculate about casualties, damage extent,   
evacuation, response success, or timelines   
Do NOT recommend actions, policies, or   
interventions   
Do NOT infer unseen impacts or intent   
STRICT OUTPUT:   
Output STRICT JSON only   
No markdown   
- No explanation   
" " "   
USER\_PROMPT\_TEMPLATE = """   
Here is the annotation JSON for one image:   
{annotation\_json}   
Generate VQA in STRICT JSON with this schema:   
"image\_id": "<IMAGE\_ID>",   
"qa\_pairs": [   
{   
"qa\_id": "<IMAGE\_ID>\_p01",   
"coverage\_section": "H1\_hazard",   
"question": "...",   
"answer": "...",   
"polarity": "positive",   
"answer\_type": "yes\_no | count | free\_form |   
unanswerable",   
"source\_fields": [   
"caption",   
"objects",   
"crisis\_assessment",   
"relationships",

"location\_assessment",   
"visible\_text",   
"uncertainties"   
],   
"support": {   
"evidence": ["short phrases copied or   
paraphrased from the JSON"],   
"absence\_check": []   
},   
"difficulty": "easy | medium | hard"   
},   
{   
"qa\_id": "<IMAGE\_ID>\_n01",   
"coverage\_section": "H8\_hallucination",   
"question": "...",   
"answer": "No",   
"polarity": "negative",   
"answer\_type": "yes\_no",   
"source\_fields": [   
"caption",   
"objects",   
"crisis\_assessment",   
"relationships",   
"visible\_text"   
],   
"support": {   
"evidence": [],   
"absence\_check": ["fields checked to justify   
the absence"]   
},   
"difficulty": "easy | medium | hard"   
}   
]   
}   
Constraints:   
- POSITIVE: include BETWEEN 12 AND 15 positive QA   
pairs.   
6-9 of them should have LONG answers (2-4   
sentences).   
The rest should have SHORT answers (1-8 words)   
when appropriate.   
- NEGATIVE: include 0 to 5 negative QA pairs   
all must be yes/no with answer exactly "No"   
Style reminder:   
- Answers must sound natural and image-based.   
- Do NOT reference annotations, captions, JSON, or   
field names in answers.   
Return STRICT JSON only.   
"""

## E.3. Prompt Template for MCQ Generation

The prompt template below is used to generate multiplechoice questions from the structured captioning output in DisasterVL. It conditions on the scene description together with the disaster assessment and enforces coverage over a predefined pool of question angles, such as hazard type, visible damage, infrastructure impact, uncertainty, and response context. The template is designed to keep the generated MCQs grounded in the provided evidence, to avoid unsupported details, and to standardize both question format and answer-option structure across the dataset.

MCQ generation prompt used in DisasterVL   
QUESTION\_ANGLE\_POOL = [   
"hazard classification - identify the type and   
severity of the disaster",

"visible damage - describe the physical damage or   
destruction present",   
"personnel and roles - who is present and what   
are they doing",   
"equipment and machinery - what tools or vehicles   
are visible and why",   
"infrastructure impact - which structures, roads,   
or utilities are affected",   
"spatial layout - where are key elements   
positioned relative to each other",   
"secondary hazards - what additional dangers or   
risks are present",   
"environmental conditions - lighting, weather,   
terrain, or time of day",   
"recovery phase - what stage of emergency   
response is underway",   
"what is absent or uncertain - what cannot be   
confirmed from the scene",   
"cause vs effect - distinguish visible effects   
from their likely cause",   
"scale of damage - estimate the extent or area   
affected",   
"access and egress - are routes blocked, passable   
, or congested",   
"victim or casualty indicators - any signs of   
people needing assistance",   
"response resources - what emergency or   
humanitarian resources are present",   
"structural integrity - assess the stability of   
visible buildings or objects",   
"fire and smoke indicators - presence, source, or   
spread of fire or smoke",   
"flood and water indicators - water levels, flow   
direction, or inundation extent",

SYSTEM\_PROMPT = """\   
You are an expert crisis-image analyst creating a   
multiple-choice quiz for emergency response   
training. Given a scene description and disaster   
details, generate exactly {num\_mcq} multiple  
choice questions (MCQs).   
Each question must focus on a DIFFERENT angle listed   
below - one question per angle:   
{angles}   
Rules:   
Each MCQ must have exactly 4 options labeled A, B,   
C, D.   
Exactly one option must be correct; the others must   
be plausible but wrong.   
Base every question and answer strictly on the   
provided description - do not invent details.   
Keep each option concise (one sentence or a short   
phrase).   
IMPORTANT: every question MUST be a proper   
interrogative sentence ending with a question   
mark (?).   
IMPORTANT: answer options (A, B, C, D) must be   
declarative statements or short phrases - never   
questions.   
Output ONLY the MCQs in the exact format shown   
below - no preamble, no commentary after.   
Output format:   
Q1: <question text>   
A) <option>   
B) <option>   
C) <option>   
D) <option>   
Answer: <letter>   
Q2: ...   
"   
USER\_TEMPLATE = """\   
Scene description:

{caption}   
disaster assessment:   
- Type: {disaster\_type}   
- Confidence: {confidence}   
- Evidence: {evidence}   
Generate {num\_mcq} MCQs based strictly on the above   
information.\   
""I

## E.4. Prompt Template for Test-Set Verification

The prompt template below is used with Gemini-3.1Pro in the verification stage of DisasterVL to assess the consistency and grounding quality of generated annotations. Its role is not to produce new supervision, but to review existing outputs and determine whether the generated caption, disaster assessment, or reasoning content remains faithful to the available visual evidence. This verifier stage helps reduce unsupported claims, improve consistency across annotations, and enforce a stricter distinction between directly observable content and uncertain or non-verifiable inference.

## Verification prompt used in DisasterVL

SYSTEM\_PROMPT = """\   
You are a strict verifier for disaster-scene   
annotations.   
Your task is to review a generated annotation and   
determine whether each part is:   
1. Supported by visible evidence,   
2. Plausible but uncertain, or   
3. Unsupported / hallucinated.   
Verification principles:   
Only use the provided image-derived description and   
structured annotation.   
Do NOT invent new details.   
Distinguish clearly between what is directly   
visible and what is inferred.   
Mark uncertainty explicitly when evidence is   
incomplete.   
Be conservative: if a claim is not clearly   
supported, do not treat it as verified.   
You must verify:   
the main caption,   
disaster assessment,   
location assessment,   
relationships,   
visible text,   
uncertainty statements,   
and any reasoning or QA justification if provided.   
Output requirements:   
Return strict JSON only.   
For each field, indicate:   
status: "supported" | "uncertain" | "unsupported"   
explanation: short reason grounded in the   
evidence   
Also provide an overall verdict summarizing whether   
the annotation is reliable.   
Do not output markdown or extra commentary.   
IIII   
USER\_PROMPT\_TEMPLATE = """\   
You are given a generated annotation to verify.

Structured annotation:   
{annotation\_json}   
Please return strict JSON in the following form:   
{   
"caption\_verification": {   
"status": "supported | uncertain | unsupported",   
"explanation": "..."   
},   
"disaster\_assessment\_verification": {   
"status": "supported | uncertain | unsupported",   
"explanation": "..."   
},   
"location\_assessment\_verification": {   
"status": "supported | uncertain | unsupported",   
"explanation": "..."   
},   
"relationships\_verification": {   
"status": "supported | uncertain | unsupported",   
"explanation": "..."   
},   
"visible\_text\_verification": {   
"status": "supported | uncertain | unsupported",   
"explanation": "..."   
},   
"uncertainties\_verification": {   
"status": "supported | uncertain | unsupported",   
"explanation": "..."   
},   
"overall\_verdict": {   
"status": "reliable | partially\_reliable |   
unreliable",   
"explanation": "..."   
}   
}   
Rules:   
- Use "supported" only when the claim is directly   
grounded in visible evidence.   
- Use "uncertain" when the claim is plausible but not   
fully verifiable.   
- Use "unsupported" when the claim goes beyond the   
available evidence.   
Keep explanations concise and evidence-based.   
- Return strict JSON only.   
" " "

## F. Visual Grounding: Evaluation Protocol

T5 is evaluated on the full test set in two settings, detection and grounded captioning (Sec. 5.2). Boxes are expressed on a 0–1000 normalized grid. The grounding layer also contains relationship endpoint links with support labels and unresolved phrases with reasons; these annotations are not scored in T5.

Matching and aggregation. Every (prediction, reference) pair that meets the criteria of the setting is a candidate. Candidates are sorted as described below and accepted in that order, skipping any whose prediction or reference has already been matched, so each predicted box and each reference entry is used at most once. True positives (TP), predicted boxes and reference entries are summed over the whole test set; precision is TP divided by the number of predicted boxes, and recall is TP divided by the number of reference entries. An image with no predicted boxes adds all of its reference entries as misses.

Detection (D-F1). Given only the image, the model returns a labelled box for every disaster-relevant object and region. The references are all annotated regions of the image. A pair is a candidate if its IoU is at least 0.5, and candidates are sorted by IoU, highest first. Labels are ignored. A model that predicts many boxes can reach high recall at low precision, and the reverse.

Grounded captioning (GC-F1). The model writes a detailed description of the image and grounds its key phrases with boxes in the same output. The reference has one entry per phrase–region link of the reference caption: a region linked to three phrases appears three times, once under each phrase, and a phrase linked to several regions contributes one entry per region. Entries from regions linked to more than one phrase make up 38% of reference entries in the test set. A pair is a candidate if its IoU is at least 0.5 and the cosine similarity between the predicted and reference phrases is at least 0.5, using all-MiniLM-L6-v2 sentence embeddings. Candidates are sorted by phrase similarity, highest first, with ties broken by higher IoU; when a region appears under several phrases, this order assigns each prediction to the entry whose phrase it matches best. Each entry is scored on its own, so full recall requires the model to ground a region once under each phrase linked to it. A correct box attached to the wrong phrase is not credited. Caption quality is reported separately.

## G. Reference-Grounded GES: Definition

Reference-grounded GES $( \mathrm { G E S } _ { \mathrm { r e f } } )$ evaluates a caption by aligning its claims to reference facts. It complements lexical and embedding-based metrics with an explicit account of which claims agree with the reference and which reference facts are covered. Its strict form measures reference alignment: a detail absent from the reference receives no precision credit even if it is visible in the image. We therefore report the strict score together with alternative precision definitions and a breakdown of the claims.

## G.1. Inputs and Claim Alignment

For each image, let $F = \{ E _ { 1 } , \ldots , E _ { | F | } \}$ denote facts obtained by splitting the reference caption into small factual units and checking them against the image with a reference normalizer. The same fact set is shared across all evaluated models. Let C be the claims extracted from a model response by the GES judge, with $n = | C |$ . A text-only aligner compares each claim to F, without access to the image, and assigns one of three mutually exclusive labels:

• Matched: the claim states one or more reference facts; the aligner returns their fact IDs.

• Contradicted: the claim conflicts with one or more reference facts; the aligner returns their fact IDs.

• Not in reference: the claim is neither matched nor contradicted by the reference.

A matched or contradicted label without a fact ID is converted to not in reference. The aligner supplies categorical labels and fact IDs; numerical scores are computed deterministically in code.

Let m and x be the numbers of matched and contradicted claims. For claims labelled not in reference, an image judge’s verdict (Sec. G.4) distinguishes $u _ { \mathrm { v i s } }$ claims marked supported from $u _ { \mathrm { n o t } }$ claims marked unverifiable or contradicted. Thus,

$$
n = m + x + u _ { \mathrm { v i s } } + u _ { \mathrm { n o t } } .\tag{1}
$$

Image verdicts are used only for the image-fallback variant and claim breakdown, not for strict reference precision or recall. Let $A ( c ) \subseteq F$ be the facts named by a matched claim c, and define

$$
F _ { \mathrm { h i t } } = \bigcup _ { c : c \mathrm { m a t c h e d } } A ( c ) .\tag{2}
$$

A claim may cover multiple facts. Each distinct fact contributes at most once to recall, even when several claims mention it; precision counts matched claims rather than distinct facts.

## G.2. Per-Response Scores and Variants

The strict reference precision and reference recall are

$$
P _ { \mathrm { r e f } } = \frac { m } { n } , \qquad R _ { \mathrm { r e f } } = \frac { | F _ { \mathrm { h i t } } | } { | F | } .\tag{3}
$$

Their harmonic mean is

$$
\mathrm { G E S } _ { \mathrm { r e f } } = \left\{ \begin{array} { l l } { \displaystyle \frac { 2 P _ { \mathrm { r e f } } R _ { \mathrm { r e f } } } { P _ { \mathrm { r e f } } + R _ { \mathrm { r e f } } } , } & { P _ { \mathrm { r e f } } + R _ { \mathrm { r e f } } > 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{4}
$$

Scores lie in [0, 1], with higher values indicating greater agreement and coverage. If $n = 0$ , precision is set to zero; with a nonempty reference, recall and the strict score are also zero. The recall definition assumes $| F | > 0$ ; no convention is defined for an empty reference.

Two diagnostic variants retain $R _ { \mathrm { r e f } }$ and replace precision by

$$
P _ { \mathrm { l e n } } = { \frac { m } { m + x } } ,\tag{5}
$$

$$
P _ { \mathrm { f b } } = { \frac { m + u _ { \mathrm { v i s } } } { n } } .\tag{6}
$$

The lenient variant ignores claims absent from the reference; only contradictions reduce its precision. When $m + x =$ 0, it is undefined and the response is excluded from the lenient mean. The image-fallback variant credits extra claims supported by the image judge and penalizes the remaining extras. Its precision is zero for $n = 0$ . Both variant scores use the harmonic mean in Eq. 4, with the same zero-sum convention. Neither variant increases recall for an extra visible detail that does not match a reference fact.

## G.3. Aggregation

Model-level scores. For N responses, the reported score is the arithmetic mean of per-response scores:

$$
\overline { { \mathrm { G E S } } } _ { \mathrm { r e f } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { G E S } _ { \mathrm { r e f } } ^ { ( i ) } .\tag{7}
$$

Precision, recall, and variant scores are averaged in the same way, with undefined lenient responses excluded from that variant’s mean. Each eligible response has equal weight regardless of its claim count. In particular, the mean score is generally different from the harmonic mean of mean precision and mean recall.

Pooled claim breakdown. With $D = \textstyle \sum _ { i } n _ { i }$ , the four reported fractions are

$$
b _ { \mathrm { m a t c h } } = { \frac { \sum _ { i } m _ { i } } { D } } , \qquad b _ { \mathrm { c o n t r a } } = { \frac { \sum _ { i } x _ { i } } { D } } ,\tag{8}
$$

$$
b _ { \mathrm { e x t r a , v i s } } = \frac { \sum _ { i } u _ { \mathrm { v i s } , i } } { D } , ~ b _ { \mathrm { e x t r a , n o t } } = \frac { \sum _ { i } u _ { \mathrm { n o t } , i } } { D } .\tag{9}
$$

These fractions sum to one when $D > 0$ . Unlike responselevel means, they weight longer responses more heavily. The pooled matched fraction need not equal mean reference precision. The last fraction describes claims absent from the reference and unconfirmed by the image judge. Because it combines unverifiable and contradicted claims, it is a proxy for unsupported content, not a verified hallucination rate.

## G.4. Relation to Image-Based GES

The image-based score uses

$$
P _ { \mathrm { v i s } } = { \frac { \# \{ { \mathrm { c l a i m s ~ m a r k e d ~ s u p p o r t e d ~ b y ~ i m a g e ~ j u d g e } } \} } { n } } ,\tag{10}
$$

$$
R _ { \mathrm { e v } } = { \frac { \# \{ { \mathrm { r e f e r e n c e ~ f a c t s ~ m a r k e d ~ h i t ~ b y ~ c o v e r a g e ~ p a s s } } \} } { | F | } } ,
$$

$$
\mathrm { G E S } = { \frac { 2 P _ { \mathrm { v i s } } R _ { \mathrm { e v } } } { P _ { \mathrm { v i s } } + R _ { \mathrm { e v } } } } .\tag{11}
$$

(12)

Image-based precision can credit a visible detail absent from the reference. Its recall comes from a separate coverage pass and can differ from $R _ { \mathrm { r e f } }$ . Thus image-based GES, strict $\mathrm { G E S _ { r e f } }$ , and the image-fallback variant are distinct measures.

The reasoning-task scores apply answer gates to imagebased GES: $\mathrm { G E S } _ { \mathrm { v q a } } = Q \mathrm { G E S } .$ , where $Q \in \{ 1 , 0 . 5 , 0 \}$ for correct, partly correct, and wrong answers; and $\mathrm { G E S } _ { \mathrm { i n c i d e n t } } \ = \ T _ { i } \mathrm { G E S } $ , where $T _ { i } ~ = ~ 1$ only if the predicted and reference incident labels match exactly after lower-casing. These gates are not part of the captioning $\mathrm { G E S _ { r e f } }$ definition. Additional diagnostics pool contradicted or unverifiable claims over all claims, and unsupported critical claims over all critical claims (casualties or injuries, cause, location or identity, severity, and responder activity). For reference uncertainty notes, the addressed fraction is the number marked hedged or overclaimed divided by all notes; the overclaimed fraction counts only overclaimed notes. These diagnostics are separate from reference alignment.