# Clips2Story: Training-Free Video Storyboarding and Editing Using Multimodal Retrieval-Embedded Generation

Junnuo Wang New York University jw6641@nyu.edu

Weihan Xu University of Washington wx35@uw.edu

Hao-Wen Dong University of Michigan hwdong@umich.edu

## Abstract

Video editors often work with large collections of fragmented footage and must select and organize relevant clips into a coherent story. However, many existing video editing and summarization methods assume a temporally ordered source video and aim to preserve its original narrative and temporal structure. In this paper, we introduce Clips2Story, a training-free, LLM-based pipeline that enables users to create keyword-specified short videos from a pool of video clips. Given a pool of clips and a user-specified keyword, we first extract text-based multimodal descriptors for each clip and retrieve the clips most relevant to the keyword. We then provide the textual descriptors of these candidate clips to an LLM, which selects and arranges them into an editable storyboard with connective narrations. Finally, we assemble the selected clips and retrieve accompanying visuals for the narrations to compose the output video. To evaluate the proposed pipeline, we construct a synthetic evaluation dataset from 100 long-form videos across five genres by segmenting each video into short clips using a scene-change detection model and shuffling the resulting clips to remove their temporal order. Through objective evaluation and qualitative examples, we demonstrate that Clips2Story can effectively retrieve keyword-relevant clips and compose coherent storyboards based on user-specified keywords. Human evaluation further shows that Clips2Story outperforms a strong baseline in terms of narrative coherence, keyword alignment, and overall realism.

## 1 Introduction

Video editing is a laborious yet critical task for video production. Video editors oftentimes need to look at a large amount of raw video footage, select useful video clips, and arrange them into the desired narrative. Yet, existing video summarization models require a long input video with a coherent storyline [Lin et al., 2023b]; existing long-form video generation models cannot retrieve and integrate raw video footage into their outputs; existing video editing approaches can only generate videos that follow the exact storyline of the input video [Xu et al., 2025]. They are therefore not designed for a common editing problem: constructing a new, user-directed story from a repository of existing video materials.

In this work, we aim to explore AI-assisted video editing that can 1) work with real video footage, 2) adapt to the user’s creative intention, and 3) generate a human-editable intermediate output that allows human intervention. Among the many complex tasks involved in professional video editing workflows, we focus on two central steps: selecting relevant video clips and organizing the selected video clips into a narrative specified by a keyword provided by the user. We introduce the clips-tostoryboard generation task and present Clips2Story, a training-free, LLM-based framework that takes a pool of video clips and a user-specified keyword as input. We first extract text-based descriptors for each video clip using speech recognition, video captioning, and object detection models. Next, we retrieve the video clips that are most relevant to the keyword provided by the user. Finally, we instruct a large language model (LLM) to arrange them into a coherent storyboard based on the multimodal information extracted from the candidate video clips. Unlike conventional video summarization models that produce a fixed summary that follows the original storyline, Clips2Story can generate diverse storyboards that differ from the input video based on user-specified keywords.

![](images/a10bcb437c32ce8e797eb394d7b52cfc34ad99f2d387bb806746e233c90d2773.jpg)  
Figure 1: Overview of the proposed Clips2Story-ND pipeline. Given a pool of raw video footage and a user-specified keyword, we first extract textual descriptors from the video clips and retrieve those most relevant to the keyword. We then use an LLM to generate an editable storyboard with connective narration. Finally, we compose the output video by assembling the selected clips and retrieving accompanying visuals for the narration.

With the generated storyboard, we then assemble the selected clips to compose the final output video. We study two variants: the Clips2Story-NF model sequences existing clips without connective narration; the Clips2Story-ND model generates connective narration to improve coherence between video clips. Rather than directly producing a finalized video, both Clips2Story variants output a human-readable storyboard containing the proposed clip sequence and narration, allowing creators to inspect and revise the narrative plan before final assembly.

To evaluate the proposed models, we construct a synthetic evaluation dataset from 100 long-form videos spanning five genres: documentary, film, lecture, vlog, and news. We first segment each video into short clips using a scene-change detection model at both the clip and scene levels. We then shuffle the resulting clips before sending their textual descriptors to the LLM, thereby removing the original temporal order and preventing leakage of the input video’s storyline. Our goal is to approximate the fragmented organization of real footage repositories that video editors often work with. We evaluate Clips2Story using automatic metrics, human evaluation, and ablations. The metrics assess repetition, visual and keyword alignment, temporal granularity, and consistency with the source order as a diagnostic reference. Across both settings, Clips2Story-NF achieves the strongest order preservation and lower repetition than the baselines, while Clips2Story-ND achieves the strongest overall keyword alignment. For human evaluation, Clips2Story-ND significantly outperforms the baseline model in terms of coherence, keyword alignment, realness, narration effectiveness, and clip-selection effectiveness.

## Our main contributions can be summarized as follows:

• We formulate keyword-guided clips-to-storyboard generation as a task that aims to edit a set of video clips into a storyboard with a coherent narrative specified by the user, representing a step toward practical video editing with existing footage.

• We introduce Clips2Story, a training-free framework that combines keyword-guided retrieval, multimodal descriptor extraction, and LLM-based storyboard planning. Its intermediate storyboard makes the proposed clip sequence and narration human-readable and editable.

• We construct a synthetic evaluation dataset from 100 real long-form videos across five genres and develop clip-level and scene-level settings for evaluating storyboard generation from fragmented video materials.

All source code and implementation details will be released upon paper acceptance. Video samples can be found on our demo website.<sup>1</sup>

## 2 Related Work

LLM-based video editing. Recent studies have explored LLMs as active agents for video editing and storytelling. Wang et al. [2024a] introduced LAVE, an LLM-powered assistant for languageguided linear video editing. For automated narrative construction, Xu et al. [2024] proposed Teaser-Gen, which generates narration and retrieves visually aligned clips to create documentary teasers. Extending this narration-driven paradigm, Xu et al. [2025] introduced REGen, which inserts retrieved video clips into explicit placeholders in an LLM-generated script. Although these systems demonstrate the potential of narration-guided and retrieval-augmented video editing, they assume that the input footage is chronologically ordered. In contrast, our method operates on a fragmented and randomly ordered pool of candidate clips, requiring the model to construct a globally coherent editorial structure from temporally unordered multimodal evidence, as summarized in Appendix A.

Video summarization. As a special case of video editing, video summarization shares the objective of distilling visual footage into a concise narrative with clips-to-storyboard generation. Existing approaches have used scene recognition and cross-modal alignment to generate condensed videos and textual summaries [Bose et al., 2023, Lin et al., 2023b, He et al., 2023]. More recently, Argaw et al. [2024] leverage LLMs to extract salient sentences from video transcripts, while Singh et al. [2024] use a hierarchical framework to identify sub-stories within structured television episodes. Unlike conventional video summarization, which condenses an existing narrative, our work composes a new storyline from a pool of candidate video clips.

LLM-based video generation. Recent advances have explored the use of LLMs for high-level semantic planning in video synthesis. Lian et al. [2023] use LLMs to expand user prompts into structured multi-scene plans. Similarly, Yeh et al. [2026] and Kondratyuk et al. [2023] frame video authoring as a context-aware expansion process that bridges narrative gaps. These methods primarily focus on planning layouts or generating new visual content to realize predefined storylines. In contrast, Clips2Story formulates video creation as a multimodal editing and sequencing problem. Rather than generating new pixels, our method extracts rich textual descriptors from existing video assets and uses them as surrogate representations. This formulation allows the LLM to act as an editor by retrieving, arranging, and narrating existing content, thereby shifting the focus from pixel-level synthesis to high-level narrative assembly.

## 3 Method

As illustrated in Figure 1, our training-free, LLM-based video editing framework consists of two stages: (1) shot-level multimodal analysis and (2) keyword-based clip retrieval. These stages are followed by two alternative pipelines for planning and rendering: Clips2Story-NF (narration-free) and Clips2Story-ND (narration-driven).

## 3.1 Shot-level Multimodal Analysis

For each clip in the candidate pool, we first extract the transcript and speaker information using a pretrained automatic speech recognition model; our implementation adopts WhisperX [Bain et al., 2023]. We then select the middle frame of the clip and generate a visual caption using a visionlanguage foundation model, instantiated as BLIP [Li et al., 2022]. Further, we extract entity and background information using an open-vocabulary object detector and a contrastive vision-language model, implemented with GroundingDINO [Liu et al., 2024b] and CLIP ViT-B/32 [Radford et al., 2021], respectively. As a result, each clip is represented by a multimodal description containing its duration, transcript, caption, entities, and background cues. These textual descriptors serve as inputs for both semantic retrieval and downstream planning.

Table 1: Statistics of the evaluation datasets.
<table><tr><td rowspan="2">Genre</td><td rowspan="2">Video duration (min)  $\mathrm { ( m e a n \pm s t d . \ d e v . ) }$ </td><td rowspan="2">Word density (# words/min)</td><td colspan="2">Clip-level shuffling</td><td colspan="2">Scene-level shuffling</td></tr><tr><td></td><td>Count Duration (sec)</td><td></td><td>Count Duration (sec)</td></tr><tr><td>Film</td><td> $8 6 . 4 5 \pm 1 3 . 3 1$ </td><td>152.43</td><td>656.55</td><td>9.66</td><td>164.95</td><td>38.33</td></tr><tr><td>Documentary</td><td> $4 8 . 4 3 \pm 4 . 6 8$ </td><td>161.27</td><td>730.75</td><td>4.13</td><td>183.55</td><td>16.42</td></tr><tr><td>Lecture</td><td> $4 1 . 8 1 \pm 1 1 . 2 5$ </td><td>168.50</td><td>99.05</td><td>165.22</td><td>25.70</td><td>300.32</td></tr><tr><td>Vlog</td><td> $2 7 . 0 4 \pm 6 . 3 5$ </td><td>186.46</td><td>52.65</td><td>74.36</td><td>14.05</td><td>198.39</td></tr><tr><td>News</td><td> $2 4 . 0 5 \pm 4 . 7 4$ </td><td>187.95</td><td>198.65</td><td>8.44</td><td>50.50</td><td>32.99</td></tr><tr><td>All</td><td> $4 5 . 5 6 \pm 8 . 0 7$ </td><td>171.32</td><td>347.53</td><td>52.36</td><td>87.75</td><td>117.29</td></tr></table>

## 3.2 Keyword-based Clip Retrieval

A major challenge in AI-assisted video editing [Lin et al., 2023a, Wang et al., 2024a] is scaling to a large number of input videos. Directly feeding the multimodal descriptions of an entire film or documentary into an LLM overloads its effective reasoning capabilities. Without filtering, the model struggles to manage the massive context [Liu et al., 2024a], resulting in poorly sequenced timelines and generated videos. To make the editing task tractable, we construct a planning pool by retrieving clips based on their semantic relevance to a user-specified keyword. Specifically, we rank clip relevance by computing the cosine similarity between the sentence-transformer embeddings [Reimers and Gurevych, 2019] of the keyword and each clip’s concatenated transcript and caption. We then retrieve a subset of candidate clips with the highest similarity scores, up to a total duration of 15 minutes. This subset forms the planning pool presented to the LLM for timeline planning, which is necessary for handling long-form content.

## 3.3 Narration-Free Timeline Sequencing

Using the keyword-specific clip pool, we prompt an LLM, instantiated as GPT-5.2 [OpenAI, 2025], to generate a video timeline centered on the user-specified keyword. We first introduce the Clips2Story NF pipeline for timeline planning. In this pipeline, the LLM is tasked exclusively with selecting and sequencing footage segments from the retrieved clip pool. Clips2Story-NF produces timelines composed entirely of the original audiovisual footage, without generating additional media such as synthetic narration. It is therefore suitable for lightweight assistance with organizing raw footage or drafting storyboards. The prompt used for Clips2Story-NF is provided in Appendix B.

## 3.4 Narration-Driven Visual Selection and Rendering

Although Clips2Story-NF sequences relevant clips, directly assembling disjointed segments can result in abrupt transitions and weak logical flow. To bridge semantic gaps between clips, we introduce Clips2Story-ND. As illustrated in Figure 1, we instruct the LLM to generate a sequence of short connective narrations that support the selected video clips. We then apply a visual selection stage to determine which footage should be displayed during each generated narration segment. Following Xu et al. [2024], we synthesize the generated narration segments into audio using a text-to-speech model; our implementation uses VITS [Kim et al., 2021]. We then extract frames from the candidate clip pool and use a VTGHLS-based score to rank and select visually relevant footage whose duration matches that of the corresponding narration.

The final Clips2Story-ND video combines three components: (1) synthesized narration audio; (2) retrieved clips played under the narration with their original audio muted; and (3) LLM-selected clips that retain both their original audio and visual content. Clips2Story-ND is suitable for editing scenarios in which the source footage alone is insufficient to produce a longer video with a coherent narrative, such as explanatory videos or short stories assembled from heterogeneous raw clips. We provide the specific prompt used for Clips2Story-ND in Appendix C.

![](images/1375431844fcde0a8c55b11cd7911fe763aaebe8e4f9da368ad56ff1534438f3.jpg)  
Figure 2: Illustration of the dataset construction process. The original input video is processed into two distinct candidate pools: a fine-grained clip-level pool and a grouped scene-level pool.

## 4 Experimental Setup

## 4.1 Evaluation Dataset

Because paired data for clips-to-storyboard generation are unavailable, we construct a controlled evaluation dataset from 100 long-form YouTube videos, evenly distributed across five genres: documentary, film, lecture, vlog, and news. We segment each source video into candidate units and shuffle them before inference to approximate real-world editing scenarios, in which footage is often stored as fragmented, unordered material rather than presented as a coherent sequence. This removes direct access to the original chronology while retaining it as a diagnostic reference. As summarized in Table 1, the genres vary substantially in video duration and word density, allowing evaluation across different temporal structures and content densities.

We construct two versions of the dataset, as illustrated in Figure 2, to evaluate performance under different levels of footage fragmentation.

• Clip-level setting: We divide each source video into short shots using the automated transition detector TransNetV2 [Soucek and Lokoc, 2024] and shuffle the resulting clips. The large pool of short, fragmented candidates provides limited temporal context and constitutes the more challenging setting.

• Scene-level setting: To approximate footage in which multiple camera shots capture the same semantic event [Bose et al., 2023], we merge adjacent shots using a multimodal similarity score based on visual, textual, and audio features, as described in Appendix D. This produces fewer, longer candidate units with greater temporal context and therefore represents a less fragmented setting. Together, these settings provide a controlled means of examining how candidate granularity affects keyword-guided storyboard planning.

## 4.2 Keyword Prompt Generation

To simulate user-directed editing, we use an LLM, instantiated as GPT-5.2 [OpenAI, 2025], to generate five representative keywords for each source video based on its full transcript and captions. Each keyword is incorporated into a shared prompt template, allowing us to evaluate whether the systems can produce distinct, semantically focused storyboards from the same candidate footage.

## 4.3 Baseline Models

We compare our method against three baseline models. For consistency, all baselines are adapted to operate on the same shuffled candidate pool as our method.

• A2Summ [He et al., 2023] is an extractive multimodal video summarization model that selects segments based on text–vision temporal correspondence. To accommodate long inputs, we apply it to 300-second chunks and concatenate the resulting selections.

• TeaserGen [Xu et al., 2024] is a narration-centric, LLM-based framework for teaser generation. It divides the transcript into ten segments, summarizes and rewrites them into a cohesive narration, and retrieves visual clips aligned with the synthesized audio.

• REGen [Xu et al., 2025] is a retrieval-embedded framework for long-to-short video editing. It generates a narration script containing visual placeholders and fills them with retrieved clips to produce the final video.

![](images/af474bcafcf462fbe5a4be4766a1a2a8558dc167e2f599151bf8bc15ad233e27.jpg)  
Figure 3: Qualitative storyboards generated by Clips2Story-ND, with each clip labeled by its original start time in the source video (left), alongside generated storyboard duration distributions under the scene-level (middle) and clip-level (right) settings.

## 4.4 Evaluation Metrics

Following TeaserGen [Xu et al., 2024] and REGen [Xu et al., 2025], we report Repetitiveness (REP), CLIPScore, and VTGHLS. To further measure the semantic alignment and temporal structure of the generated storyboards, we introduce three task-specific metrics:

• Mean clip duration (MCD) is the average duration of the selected clips and generated narration segments, capturing the pacing and temporal granularity of the final sequence.

• Order preservation coefficient (OPC) measures the consistency between the generated clip order and the original chronological order of the source video using Kendall’s τ. Although our task does not aim to reconstruct the source chronology, the original order provides an informative proxy for natural temporal and narrative progression. A higher OPC therefore indicates that the generated timeline more closely follows the underlying narrative flow of the source video.

• Keyword relevance (KR) measures the semantic alignment between the generated timeline and its target keyword. We compute three modality-specific scores based on (1) narration and transcripts, (2) video frames, and (3) visual captions. Each modality is divided into approximately uniform textual or temporal units, and its KR score is computed as the mean cosine similarity between the keyword and the corresponding units.

Formal definitions and implementation details are provided in Appendix E.

## 4.5 Subjective Evaluation

We conduct a human evaluation with 15 participants using a 5-point Likert scale. We consider three genres: documentary, film, and lecture. For each genre, we randomly sample six videos and ask participants to evaluate the short videos (3–5 minutes) generated by Clips2Story-NF, Clips2Story-ND, and REGen [Xu et al., 2025]. The study evaluates the generated videos across five dimensions: coherence, keyword alignment, realness, narration effectiveness, and clip-selection effectiveness. Additionally, we compare the textual storyboards generated by REGen and Clips2Story-ND across three dimensions: organization, keyword informativeness, and engagingness. The exact questionnaire items are provided in Appendices F and G.

## 5 Results

## 5.1 Qualitative Examples

Figure 3 presents two example storyboards generated by Clips2Story-ND for the same documentary under the scene-level setting, conditioned on the keywords “mammal” and “fossil.” The selected clips exhibit clear semantic alignment with their respective keywords. For “mammal,” the storyboard includes dolphins, primates, and humans, whereas “fossil” retrieves footage of fossilized skulls and bones. Furthermore, the timestamps below the clips denote their original start times in the source video. We observe that the generated storyboards can reconstruct parts of the original timeline while preserving narrative and logical flow. These examples demonstrate the system’s ability to customize storyboards according to user-specified keywords while organizing semantically related clips into coherent sequences.

Table 2: Objective evaluation results under the scene-level and clip-level settings. Error bars indicate 95% confidence intervals.
<table><tr><td colspan="2">Model</td><td>MCD (sec)</td><td>REP (%)</td><td> $\mathbf { C L I P S c o r e }$ </td><td>VTGHLS</td><td>OPC</td><td>KR-Narration ↑</td><td></td><td>KR-Frame ↑ KR-Caption ↑</td></tr><tr><td rowspan="4">Sce-vel shuiing</td><td>A2Summ [He et al., 2023]</td><td> $1 5 . 5 8 \pm 3 . 8 6$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 5 4 \pm 0 . 0 1$ </td><td> $1 . 0 4 \pm 0 . 0 2$ </td><td> $0 . 1 8 \pm 0 . 0 6$ </td><td> $0 . 1 6 \pm 0 . 0 1$ </td><td> $0 . 1 7 \pm 0 . 0 0$ </td><td> $0 . 2 1 \pm 0 . 0 1$ </td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td> $4 . 1 9 \pm 0 . 1 5$ </td><td> $1 8 . 9 5 \pm 2 . 2 9$ </td><td> $0 . 6 0 \pm 0 . 0 2$ </td><td>1.41 ± 0.02</td><td> $0 . 0 5 \pm 0 . 0 5$ </td><td> $0 . 1 8 \pm 0 . 0 1$ </td><td> $0 . 1 8 \pm 0 . 0 1$ </td><td> $0 . 2 2 \pm 0 . 0 2$ </td></tr><tr><td>REGen [Xu et al., 2025]</td><td> $4 . 8 5 \pm 0 . 9 0$ </td><td> $2 0 . 3 9 \pm 6 . 0 7$ </td><td> $0 . 5 8 \pm 0 . 0 2$ </td><td>0.68 ± 0.10</td><td> $0 . 1 1 \pm 0 . 1 1$ </td><td> $0 . 1 8 \pm 0 . 0 2$ </td><td> $0 . 1 8 \pm 0 . 0 1$ </td><td> $0 . 2 3 \pm 0 . 0 2$ </td></tr><tr><td>Clips2Story-NF</td><td> $2 8 . 7 5 \pm 2 . 6 7$ </td><td> $0 . 8 9 \pm 0 . 4 6$ </td><td> $0 . 5 6 \pm 0 . 0 2$ </td><td> $1 . 0 6 \pm 0 . 0 3$ </td><td> $0 . 5 6 \pm 0 . 0 4$ </td><td> $0 . 2 2 \pm 0 . 0 1$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 0 }$ </td><td> $0 . 2 3 \pm 0 . 0 2$ </td></tr><tr><td rowspan="5">Cip-vel shiing</td><td>Clips2Story-ND</td><td> $1 0 . 5 9 \pm 0 . 7 9$ </td><td> $1 3 . 1 1 \pm 2 . 1 4$ </td><td>0.56 ± 0.01 1.20 ± 0.01</td><td></td><td> $0 . 2 7 \pm 0 . 0 4$ </td><td> $\mathbf { 0 . 2 5 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 0 }$ </td><td>0.24 ± 0.01</td></tr><tr><td>A2Summ [He et al., 2023]</td><td> $1 6 . 7 6 \pm 3 . 9 6$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 5 5 \pm 0 . 0 1$ </td><td>1.05 ± 0.02</td><td> $0 . 0 5 \pm 0 . 0 4$ </td><td> $0 . 1 7 \pm 0 . 0 1$ </td><td> $0 . 1 7 \pm 0 . 0 0$ </td><td>0.21 ± 0.01</td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td> $4 . 1 1 \pm 0 . 1 4$ </td><td>16.86 ± 2.44</td><td> $0 . 6 0 \pm 0 . 0 1$ </td><td>1.41 ± 0.02</td><td> $0 . 0 7 \pm 0 . 0 6$ </td><td> $0 . 1 8 \pm 0 . 0 1$ </td><td> $0 . 1 8 \pm 0 . 0 1$ </td><td>0.23 ± 0.02</td></tr><tr><td>REGen [Xu et al., 2025]</td><td> $4 . 9 0 \pm 0 . 5 0$ </td><td> $2 0 . 2 1 \pm 6 . 3 8$ </td><td> $0 . 5 8 \pm 0 . 0 2$ </td><td>0.78 ± 0.10</td><td> $0 . 0 7 \pm 0 . 0 8$ </td><td> $0 . 1 8 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 1 }$ </td><td>0.23 ± 0.02</td></tr><tr><td>Clips2Story-NF</td><td> $1 5 . 2 2 \pm 2 . 1 9$ </td><td> $0 . 6 6 \pm 0 . 2 9$ </td><td> $0 . 5 5 \pm 0 . 0 2$ </td><td>1.05 ± 0.03</td><td> $0 . 3 0 \pm 0 . 0 6$ </td><td> $0 . 2 4 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 0 }$ </td><td> $0 . 2 3 \pm 0 . 0 2$ </td></tr><tr><td rowspan="2"></td><td>Clips2Story-ND</td><td> $7 . 6 4 \pm 0 . 7 4$ </td><td> $1 5 . 8 0 \pm 2 . 0 5$ </td><td>0.55 ± 0.01</td><td>1.24 ± 0.01</td><td> $0 . 1 5 \pm 0 . 0 4$ </td><td> ${ \bf 0 . 2 6 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 2 4 \pm 0 . 0 1 }$ </td></tr><tr><td>Input video clips</td><td></td><td> $0 . 0 2 \pm 0 . 0 0$ </td><td>0.50 ± 0.01</td><td>1.00 ± 0.02</td><td>一</td><td> $0 . 1 5 \pm 0 . 0 1$ </td><td>0.17 ± 0.00</td><td>0.20 ± 0.01</td></tr></table>

![](images/4befcd088228081a08b954264ae72a2c0d534ab888e592fa1b89996eb7679120.jpg)  
Timeline progress (% of input duration)

![](images/86fea63dadac8751688d9e7ecf2c17fb478b0c47d2718eca5f0a80bc8e2c9bb2.jpg)  
Timeline progress (% of output duration)  
Figure 4: KR-Narration over the normalized timeline for the documentary Black and Jewish America: An Interwoven History. Left: The input video evaluated with five target keywords. Right: Outputs generated by A2Summ, TeaserGen, REGen, Clips2Story-NF, and Clips2Story-ND for the target keyword “Strange Fruit.”

## 5.2 Objective Evaluation Results

Table 2 presents the overall performance of our proposed pipelines and the baselines under both the scene-level and clip-level settings.

Narrative coherence and redundancy. Although explicit temporal reconstruction is not the primary objective of our task, the chronological order of the source video serves as a proxy for natural narrative progression. As shown in Table 2, Clips2Story-NF achieves the highest OPC in both settings, suggesting that LLM-based planning can recover a coherent narrative trajectory without access to explicit temporal priors. In comparison, the three baselines, A2Summ, TeaserGen, and REGen, obtain OPC scores below 0.20. Moreover, TeaserGen and REGen exhibit higher repetitiveness (REP), indicating that they reuse the same content more frequently. Both Clips2Story variants achieve lower REP values than these baselines. Between our two variants, the narration-free Clips2Story-NF produces substantially less repetition than the narration-driven Clips2Story-ND. One possible explanation is that the narration-to-visual matching stage in Clips2Story-ND, which follows the retrieval approach introduced by TeaserGen, may retrieve the same visually relevant clips for semantically similar narration segments.

Duration distribution. Figure 3 shows the distribution of generated storyboard durations. In our experiments, the target duration is set to three to five minutes. A2Summ exhibits relatively broad distributions, reflecting variation in source-video length. In contrast, TeaserGen and REGen generally produce shorter videos because they are designed specifically for teaser generation. Both Clips2Story variants tend to exceed the target duration. Clips2Story-NF remains closer to the desired range, whereas Clips2Story-ND exceeds it by a larger margin because its planning process also accounts for the estimated duration of the generated narration. Under the finer-grained clip-level setting, the duration distributions of both variants shift closer to the target range, as shorter candidate clips provide the LLM with more precise control over the final storyboard duration.

Keyword alignment. As shown in Table 2, Clips2Story achieves its largest gains in KR-Narration. Clips2Story-ND ranks highest in both settings, followed by Clips2Story-NF. Both variants also achieve the highest or tied-highest KR-Frame, while Clips2Story-ND obtains the highest KR-Caption. Compared with the input clip pool, both variants improve relevance across all modalities. Figure 4 further illustrates these results over time. For the documentary Black and Jewish America: An Interwoven History, the target keywords exhibit distinct relevance profiles in the input video, while the Clips2Story outputs for “Strange Fruit” maintain higher narration relevance than the baseline outputs across much of their normalized timelines. Together, the results indicate that keyword-based retrieval effectively anchors the generated storyboards to the target theme.

![](images/b461a53953681f944c7bc4b747bc735be28b2d4d7308a5b09b6a9a7b2b1b3b34.jpg)  
Figure 5: Genre-wise comparison of CLIPScore, VTGHLS, OPC, KR-Narration, KR-Frame, and KR-Caption under the scene-level setting. Error bars indicate 95% confidence intervals. Specific numbers can be found in Table 6.

Table 3: Subjective evaluation results under the scene-level setting on a Likert scale from 1 to 5. Higher values are better. All values are reported with 95% confidence intervals.
<table><tr><td></td><td colspan="5">Generated videos</td><td colspan="3">Generated textual storyboards</td></tr><tr><td>Model</td><td>Coherence</td><td>Keyword alignment</td><td>Realness</td><td>Narration effectiveness</td><td>Clip-selection effectiveness</td><td></td><td>Keyword</td><td>Organization informativeness Engagingness</td></tr><tr><td>REGen [Xu et al., 2025]</td><td> $3 . 1 7 \pm 0 . 1 1$ </td><td> $3 . 1 1 \pm 0 . 1 6$ </td><td> $2 . 8 2 \pm 0 . 1 7$ </td><td>3.27 ± 0.15</td><td>3.03 ± 0.17 4.07 ± 0.19</td><td></td><td> $3 . 6 9 \pm 0 . 2 3$ </td><td> $3 . 7 3 \pm 0 . 2 7$ </td></tr><tr><td>Clips2Story-NF</td><td> $3 . 6 9 \pm 0 . 1 3$ </td><td> $3 . 5 2 \pm 0 . 1 5$ </td><td>3.58 ± 0.15</td><td></td><td>3.58 ± 0.15</td><td></td><td></td><td></td></tr><tr><td>Clips2Story-ND</td><td>4.43 ± 0.14 4.42 ± 0.13 4.28 ± 0.15 4.29 ± 0.14 4.17 ± 0.16 4.07 ± 0.19 4.09 ± 0.18 4.07 ± 0.21</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Cross-genre robustness. Figure 5 compares performance across five genres with different temporal structures and speech densities. Clips2Story-NF achieves the highest OPC in every genre, with particularly strong results for the transcript-dense lecture and news categories; corresponding worddensity statistics are reported in Table 1. Both Clips2Story variants also maintain relatively stable VTGHLS across genres, whereas REGen drops substantially for films, lectures, and vlogs despite comparable CLIPScores. Keyword alignment is more modality-dependent: Clips2Story-ND achieves the highest or near-highest KR-Narration across genres, while KR-Frame differences are small and KR-Caption gains are less consistent. Overall, the proposed methods generalize across genres, with the clearest advantages in narration-based relevance and temporal-order preservation.

## 5.3 Subjective Evaluation Results

Table 3 shows that both Clips2Story variants consistently outperform REGen across all evaluated criteria, with the narration-driven Clips2Story-ND outperforming the narration-free Clips2Story-NF. Clips2Story-ND also achieves high coherence and keyword alignment scores, showing that narration-driven planning significantly improves logical flow and semantic faithfulness. Even without voiceovers, Clips2Story-NF surpasses REGen in coherence and realness, confirming that our retrievalfirst approach selects higher-quality visual sequences. Conversely, REGen achieves a significantly lower realness score than the Clips2Story variants, yielding outputs that viewers perceive as artificial or disjointed. While Clips2Story-ND achieves parity with REGen in organization, it significantly outperforms the baseline in keyword informativeness and engagingness. These large margins confirm that storyboard generation using a filtered clip pool and text-based descriptors can yield compelling, contextually anchored stories that the baseline struggles to achieve.

## 5.4 Ablation Studies

We conduct ablation studies on a 50-video subset under the clip-level setting using Clips2Story-NF. We evaluate the effects of removing visual descriptors or transcripts, replacing keyword-based retrieval with random retrieval, replacing middle-frame captioning with video captioning, and replacing GPT-5.2 [OpenAI, 2025] with either the open-weight Qwen3-32B [Yang et al., 2025] or the vision-language model Qwen2.5-VL [Bai et al., 2025]. For random retrieval, we shuffle the full clip pool and sample

Table 4: Ablation results under the clip-level setting. All variants are defined relative to the full Clips2Story-NF pipeline. All values are reported with 95% confidence intervals.
<table><tr><td>Model</td><td>MCD (sec)</td><td>REP (%)</td><td>CLIPScore</td><td>VTGHLS</td><td>KR-Narration ↑</td><td>KR-Frame ↑</td><td>KR-Caption ↑</td><td>OPC</td></tr><tr><td>Clips2Story-NF (full model)</td><td>14.52 ± 2.90</td><td>0.63 ± 0.36</td><td>0.54 ± 0.02</td><td> $1 . 0 5 \pm 0 . 0 3$ </td><td> ${ \bf 0 . 2 4 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 1 }$ </td><td> $0 . 2 2 \pm 0 . 0 2$ </td><td>0.27 ± 0.09</td></tr><tr><td>w/o visual descriptors</td><td>15.95 ± 3.20</td><td>0.63 ± 0.38</td><td>0.49 ± 0.04</td><td> $0 . 9 7 \pm 0 . 0 8$ </td><td> $0 . 2 3 \pm 0 . 0 2$ </td><td> $0 . 1 7 \pm 0 . 0 1$ </td><td> $0 . 2 0 \pm 0 . 0 2$ </td><td>0.38 ± 0.06</td></tr><tr><td>w/o transcript</td><td>11.39 ± 2.38</td><td>0.71 ± 0.32</td><td>0.52 ± 0.03</td><td> $1 . 0 4 \pm 0 . 0 7$ </td><td> $0 . 1 7 \pm 0 . 0 2$ </td><td> $0 . 1 8 \pm 0 . 0 1$ </td><td> ${ \bf 0 . 2 3 \pm 0 . 0 2 }$ </td><td> $0 . 0 1 \pm 0 . 0 6$ </td></tr><tr><td>w/o keyword-based clip retrieval</td><td>12.52 ± 2.40</td><td>0.75 ± 3.90</td><td>0.51 ± 0.04</td><td> $0 . 9 9 \pm 0 . 0 8$ </td><td> $0 . 1 7 \pm 0 . 0 2$ </td><td> $0 . 1 7 \pm 0 . 0 1$ </td><td> $0 . 2 1 \pm 0 . 0 2$ </td><td> $0 . 2 7 \pm 0 . 0 7$ </td></tr><tr><td>w/ video captioning</td><td> $1 5 . 6 0 \pm 3 . 1 4$ </td><td>0.78 ± 0.42</td><td>0.51 ± 0.04</td><td> $1 . 0 0 \pm 0 . 0 8$ </td><td> $0 . 2 3 \pm 0 . 0 2$ </td><td> $0 . 1 9 \pm 0 . 0 1$ </td><td> $0 . 2 2 \pm 0 . 0 2$ </td><td>0.28 ± 0.08</td></tr><tr><td>w/ Qwen3-32B [Yang et al., 2025]</td><td> $9 . 5 7 \pm 1 . 7 2$ </td><td>10.23 ± 3.21</td><td>0.49 ± 0.04</td><td> $0 . 9 8 \pm 0 . 0 8$ </td><td> $0 . 2 0 \pm 0 . 0 2$ </td><td> $0 . 1 7 \pm 0 . 0 1$ </td><td> $0 . 2 0 \pm 0 . 0 2$ </td><td>0.18 ± 0.08</td></tr><tr><td>w/ Qwen2.5-VL-7B-Instruct [Bai et al., 2025] 17.72 ± 4.66</td><td></td><td>5.54 ± 2.16</td><td>0.47 ± 0.05 0.92 ± 0.09</td><td></td><td> $0 . 2 3 \pm 0 . 0 3$ </td><td> $0 . 1 8 \pm 0 . 0 1$ </td><td> $0 . 2 0 \pm 0 . 0 2$ </td><td>0.19 ± 0.09</td></tr></table>

clips using the same duration budget as keyword-based retrieval to satisfy the planner’s context limit.   
All results are reported with 95% confidence intervals.

As shown in Table 4, visual and textual descriptors serve complementary roles. Removing visual descriptors decreases CLIPScore from 0.54 to 0.49, VTGHLS from 1.05 to 0.97, and frame- and caption-based keyword relevance, but increases OPC from 0.27 to 0.38, suggesting that visual descriptors primarily support semantic grounding. In contrast, removing transcripts reduces KR-Narration from 0.24 to 0.17 and OPC to 0.01 while largely preserving CLIPScore and VTGHLS, indicating that transcripts support narrative and temporal ordering. Random retrieval reduces keyword relevance across all modalities without improving mean OPC, showing that the planner cannot fully compensate for a less relevant candidate pool. The video-captioning variant performs similarly to the full model, with only modest decreases in CLIPScore and VTGHLS, suggesting that middleframe captioning efficiently represents the temporally consistent clips produced by our segmentation pipeline.

Finally, replacing GPT-5.2 with Qwen3-32B or Qwen2.5-VL lowers CLIPScore, VTGHLS, and OPC while increasing REP. Qwen3-32B yields the highest REP at 10.23%, while Qwen2.5-VL retains comparable KR-Narration but shows weaker semantic alignment and temporal ordering. These results show that Clips2Story supports both open-weight and vision-language planners, although neither matches GPT-5.2 across the evaluated metrics. Taken together, transcripts support narrative ordering, visual descriptors strengthen semantic grounding, and keyword-based retrieval improves candidate relevance. Middle-frame captioning provides an efficient visual representation, while planner strength remains critical for coherent, non-repetitive storyboards.

## 5.5 Computational Cost

We evaluate inference latency on the full dataset of 100 videos using a single NVIDIA L40S GPU. Clips2Story-NF requires an average of 6.18 minutes per video, equivalent to 16.68% of the input duration. Clips2Story-ND requires 7.96 minutes per video on average, including 1.77 minutes for narration–visual matching. Multimodal metadata is extracted only once for each input video and can be reused across different editing prompts. Overall, both variants complete inference in substantially less time than the duration of the input video, demonstrating the computational feasibility of the proposed training-free pipelines.

## 6 Discussion & Limitations

While Clips2Story demonstrates robust performance, it has several limitations. First, relying solely on text-based descriptors means that the system overlooks critical non-textual elements, such as fine grained visual details and background music continuity. This limits overall fidelity and can result in visually or acoustically jarring transitions. Future work could leverage native multimodal foundation models to process raw audiovisual streams directly. Second, our hard-filtering retrieval strategy, designed to manage LLM context limits, may discard structurally useful but semantically tangential B-roll. A dynamic retrieval mechanism could enable more complex narrative arcs. Moreover, by optimizing for narrative coherence, the LLM risks recontextualizing factual footage into misleading storylines. This potential for contextual manipulation and misinformation is a critical ethical concern, particularly in sensitive domains such as news and documentaries.

Finally, we show in Figure 6 a prototype interface for the proposed Clips2Story system. The interface allows the user to enter a keyword that determines the theme of the video. The system will automatically suggest the most relevant clips from the video library for inclusion in the second stage, but the user can review and adjust the selection. The selected video clips will be passed to the second stage, where the system will suggest a storyboard with a live preview. The user can review, reorder, replace, or remove the video clips before the final rendering. The interface is designed to integrate directly into professional video editing workflows while preserving human agency and creativity. A demo of the prototype can be found on our demo website.

![](images/b235967bd79bba05787d7e1eb3a8e084a6fa1c3450f7d505cd3ebc3afe9cc6e3.jpg)  
Figure 6: Prototype Clips2Story interface. (a) Given a user-input keyword, the system will suggest the most relevant video clips from the video library. The user may review and adjust the selection. (b) Given the selected video clips, the system will generate a storyboard with clip details and a live preview. The user may review, reorder, replace, and remove the video clips before the final rendering. A demo of the prototype can be found on our demo website.<sup>1</sup>

## 7 Conclusion

We introduced Clips2Story, a training-free, LLM-driven framework that generates coherent video storyboards from a non-chronologically ordered pool of video clips. To evaluate our approach, we developed a synthetic data generation pipeline that removes the original temporal order of the clips. We show that, given a pool of video clips and a user-specified keyword, the proposed system can effectively generate a coherent video storyboard by selecting and rearranging the input clips. Our human evaluation further shows that Clips2Story outperforms REGen in terms of coherence, keyword alignment, and realness. Our framework highlights the potential of LLMs as active editorial agents, paving the way for more intelligent and scalable AI-assisted video editing tools.

## References

Dawit Mureja Argaw, Seunghyun Yoon, Fabian Caba Heilbron, Hanieh Deilamsalehy, Trung Bui, Zhaowen Wang, Franck Dernoncourt, and Joon Son Chung. Scaling up video summarization pretraining with large language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8332–8341, 2024.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

Max Bain, Jaesung Huh, Tengda Han, and Andrew Zisserman. Whisperx: Time-accurate speech transcription of long-form audio. arXiv preprint arXiv:2303.00747, 2023.

Digbalay Bose, Rajat Hebbar, Krishna Somandepalli, Haoyang Zhang, Yin Cui, Kree Cole-McLaughlin, Huisheng Wang, and Shrikanth Narayanan. Movieclip: Visual scene recognition in movies. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2083–2092, 2023.

Bo He, Jun Wang, Jielin Qiu, Trung Bui, Abhinav Shrivastava, and Zhaowen Wang. Align and attend: Multimodal summarization with dual contrastive losses. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 14867–14878, 2023.

Jaehyeon Kim, Jungil Kong, and Juhee Son. Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech. In International conference on machine learning, pages 5530–5540. PMLR, 2021.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.

Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley. Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:2880–2894, 2020.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022.

Long Lian, Baifeng Shi, Adam Yala, Trevor Darrell, and Boyi Li. Llm-grounded video diffusion models. arXiv preprint arXiv:2309.17444, 2023.

Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091, 2023a.

Jingyang Lin, Hang Hua, Ming Chen, Yikang Li, Jenhao Hsiao, Chiuman Ho, and Jiebo Luo. Videoxum: Cross-modal visual and textural summarization of videos. IEEE Transactions on Multimedia, 26:5548–5560, 2023b.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the associationfor computational linguistics, 12:157–173, 2024a.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024b.

OpenAI. Introducing GPT-5.2. https://openai.com/index/introducing-gpt-5-2/, December 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 3982–3992, 2019.

Aditya Kumar Singh, Dhruv Srivastava, and Makarand Tapaswi. Previously on... from recaps to story summarization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13635–13646, 2024.

Tomás Soucek and Jakub Lokoc. Transnet v2: An effective deep network architecture for fast shot transition detection. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11218–11221, 2024.

Bryan Wang, Yuliang Li, Zhaoyang Lv, Haijun Xia, Yan Xu, and Raj Sodhi. Lave: Llm-powered agent assistance and language augmentation for video editing. In Proceedings of the 29th International Conference on Intelligent User Interfaces, pages 699–714, 2024a.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024b.

Weihan Xu, Paul Pu Liang, Haven Kim, Julian McAuley, Taylor Berg-Kirkpatrick, and Hao-Wen Dong. Teasergen: Generating teasers for long documentaries. arXiv preprint arXiv:2410.05586, 2024.

Weihan Xu, Yimeng Ma, Jingyue Huang, Yang Li, Wenye Ma, Taylor Berg-Kirkpatrick, Julian McAuley, Paul Pu Liang, and Hao-Wen Dong. Regen: Multimodal retrieval-embedded generation for long-to-short video editing. arXiv preprint arXiv:2505.18880, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Catherine Yeh, Anh Truong, Mira Dontcheva, and Bryan Wang. Vidmento: Creating video stories through context-aware expansion with generative video. arXiv preprint arXiv:2601.22013, 2026.

## A Comparisons to Prior Work

Table 5 compares Clips2Story with related methods.

Table 5: Comparison of Clips2Story with related methods. Unlike existing baselines, our approach operates on an unordered pool of video clips and supports keyword-guided clip retrieval, narrative ordering, and connective narration. Here, Summ. = summarization, Edit. = editing, Gen. = generation, Ord. = ordered, Unord. = unordered, retr. = retrieval, Narr. = narrative, Conn. = connective, and narr. = narration. <sup>∗</sup>Achieved via our Clips2Story-ND pipeline.
<table><tr><td rowspan="2"></td><td colspan="2">Task type</td><td colspan="3">Inputs</td><td colspan="3">Capabilities</td></tr><tr><td>Summ. / Edit.</td><td>Gen.</td><td>Ord.</td><td>Unord.</td><td>Keyword guided</td><td>Clip Narr. retr.</td><td>ord.</td><td>Conn. narr.</td></tr><tr><td>MovieClip [Bose et al., 2023]</td><td></td><td></td><td></td><td>X</td><td>X</td><td></td><td></td><td>X</td></tr><tr><td>A2Summ [He et al., 2023]</td><td></td><td></td><td></td><td></td><td>X</td><td></td><td></td><td>X</td></tr><tr><td>LfVS [Argaw et al., 2024]</td><td></td><td></td><td></td><td>X</td><td>X</td><td></td><td></td><td>X</td></tr><tr><td>TaleSumm [Singh et al., 2024]</td><td></td><td></td><td></td><td></td><td>X</td><td></td><td></td><td>X</td></tr><tr><td>VideoXUM [Lin et al., 2023b]</td><td></td><td></td><td></td><td></td><td>X</td><td></td><td></td><td></td></tr><tr><td>LAVE [Wang et al., 2024a]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td></td><td></td><td></td><td>X</td><td>X</td><td></td><td></td><td></td></tr><tr><td>REGen [Xu et al., 2025]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLM-grounded [Lian et al., 2023]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vidmento [Yeh et al., 2026]</td><td></td><td></td><td></td><td>X</td><td></td><td></td><td></td><td></td></tr><tr><td>VideoPoet [Kondratyuk et al., 2023]</td><td></td><td></td><td></td><td>X</td><td></td><td>X</td><td></td><td>X</td></tr><tr><td>Clips2Story (ours)</td><td></td><td></td><td>X</td><td></td><td></td><td></td><td></td><td></td></tr></table>

## B Prompt Template for the Narration-Free Variant

The prompt used for the Narration-Free pipeline is provided below. In practice, the AVAILABLE CLIPS (UNORDERED POOL) field is populated dynamically with the retrieved candidate clips for each example.

## System / Instruction Prompt

You are an expert AI video editor and planning assistant. Convert the user’s editing request into a storyboard (sequence of footage segments) in strict JSON.

## USER REQUEST

Create a catchy 3–5 minute mini-story about {keyword}. Open with a strong hook, build curiosity and progression, and end with a satisfying payoff. Select and sequence footage that naturally builds this narrative.

## AVAILABLE CLIPS (UNORDERED POOL)

Each clip is independent raw footage. There is no implied chronological relationship by list order. Use only the provided clip\_id strings when referencing clips in your timeline. Each clip includes transcript, caption, background, and entities.

- clip\_id={clip\_id\_1}, duration={duration\_1}   
transcript="{transcript\_1}"   
caption="{caption\_1}"   
background="{background\_1}"   
entities="{entities\_1}"   
- clip\_id={clip\_id\_2}, duration={duration\_2}   
transcript="{transcript\_2}"   
caption="{caption\_2}"   
background="{background\_2}"   
entities="{entities\_2}"

## TASK

Produce an ordered sequence of footage segments. Each item in the sequence has a "segments" list containing the footage segments that tell the story.

You may select, skip, trim, or reorder clips as needed. When trimming, define the exact start\_time and end\_time to extract the specific part of the clip you want to use, while respecting transcript sentence boundaries. Choose as many sequence items as needed so that the total runtime is approximately 3–5 minutes.

## RULES

1. Compose a coherent story arc with a strong hook, clear progression, and a satisfying payoff. Infer intent from the user request, including topic, tone, and pacing.

2. Select footage using transcript, caption, background, and entity evidence; prefer strong matches. Ensure that the sequence of clips flows naturally to tell the story.

3. Choose clips and precisely define their start\_time/end\_time to trim the footage to the exact moments needed. Do not trim mid-sentence. If a clip contains speech, the start\_time and end\_time must exactly match transcript timestamp brackets, never intermediate points. Do not feel constrained to use the full clip duration.

4. Choose as many segments as needed so that the total runtime, defined as the sum of all selected clip durations, is approximately 3–5 minutes.

5. Do not invent any clips. Do not invent or output timestamps outside the bounds of the original clip duration.

6. For each selected segment, include "duration" (computed as end\_time - start\_time) and "transcript" containing only the transcript text that falls within the selected time span, preserving speaker labels and relative timestamps when present.

• Do not paraphrase, summarize, or invent transcript text.

• If no transcript is available for that clip or trimmed segment, output "transcript": "".

## HARD CONSTRAINTS

• Each segment’s clip\_id must exactly match an available clip.

• Do not invent clips.

• Output must be valid JSON only, with no markdown or extra text.

## OUTPUT SCHEMA (STRICT)

```json
[
{
"segments": [
{
"clip_id": "c_91f2a3",
"start_time": 0.0,
"end_time": 0.0,
"duration": 0.0,
"transcript": ""
}
]
}
]
```

Output only the JSON array, with no markdown and no extra text.

## C Prompt Template for the Narration-Driven Variant

The prompt used for the Narration-Driven pipeline is provided below. In practice, the AVAILABLE CLIPS (UNORDERED POOL) field is populated dynamically with the retrieved candidate clips for each example.

## System / Instruction Prompt

You are an expert AI video editor and planning assistant. Convert the user’s editing request into a storyboard (narration + supporting footage segments) in strict JSON.

## USER REQUEST

Create a catchy 3–5 minute mini-story about {keyword}. Open with a strong hook, build curiosity and progression, and end with a satisfying payoff. Write short narrations that guide

the viewer, and for each narration select footage that will visually support what the narration just set up.

## CRITICAL – RENDERING ORDER

• Narration is played first; then the selected footage segments for that narration play. Narration is not played over footage.

• Therefore, write each narration as a lead-in for the clips that follow it. The footage should visually support what the narration just set up using transcript, caption, background, and entity evidence when available.

## AVAILABLE CLIPS (UNORDERED POOL)

Each clip is independent raw footage. There is no implied chronological relationship by list order. Use only the provided clip\_id strings when referencing clips in your timeline. Each clip includes transcript, caption, background, and entities.

```javascript
- clip_id={clip_id_1}, duration={duration_1}
transcript="{transcript_1}"
caption="{caption_1}"
background="{background_1}"
entities="{entities_1}"
```

```javascript
- clip_id={clip_id_2}, duration={duration_2}
transcript="{transcript_2}"
caption="{caption_2}"
background="{background_2}"
entities="{entities_2}"
```

## TASK

Produce a list of sequence items. Each item in the sequence has: (1) "narration": short text for that moment (no timestamps; narration duration will be determined later by TTS), and (2) "segments": footage segments that play after this narration and visually support it. You may select, skip, trim, or reorder clips as needed. When trimming, define the exact start\_time and end\_time to extract the specific part of the clip you want to use (respecting transcript sentence boundaries). Choose as many sequence items as needed so that the total runtime (narration + clip durations) is approximately 3–5 minutes.

## RULES

1. Compose a coherent story arc with a strong hook, clear progression, and a satisfying payoff. Infer intent from the user request, including topic, tone, and pacing.

2. Select footage using transcript, caption, background, and entity evidence; prefer strong matches. Each narration’s segments should visually support what the narration just set up.

3. Choose clips and precisely define their start\_time/end\_time to trim the footage to the exact moments needed. Do not trim mid-sentence. If a clip contains speech, the start\_time and end\_time must exactly match transcript timestamp brackets, never intermediate points. Do not feel constrained to use the full clip duration.

4. Choose as many sequence items as needed so that the total runtime (TTS narration duration + all selected clip durations) is approximately 3–5 minutes. Do not add timestamps for narration; only provide the narration text.

5. Do not invent any clips. Do not invent or output timestamps outside the bounds of the original clip duration.

6. For each selected segment, include "duration" (computed as end\_time - start\_time) and "transcript" containing only the transcript text that falls within the selected time span, preserving speaker labels and relative timestamps when present. • Do not paraphrase, summarize, or invent transcript text.

• If no transcript is available for that clip or trimmed segment, output "transcript": "".

## HARD CONSTRAINTS

• Each segment’s clip\_id must exactly match an available clip.

• Do not invent clips.

• Output must be valid JSON only, with no markdown or extra text.

## OUTPUT SCHEMA (STRICT)

[   
{   
"narration": "string",   
"segments": [   
{   
"clip\_id": "c\_91f2a3",   
"start\_time": 0.0,   
"end\_time": 0.0,   
"duration": 0.0,   
"transcript": ""   
}   
]   
}   
]   
Output only the JSON array, with no markdown and no extra text.

## D Dataset Construction Details

Below, we provide the technical specifications and hyperparameters used to process the source videos into the clip-level and scene-level candidate pools.

Clip-level extraction. In the clip-level setting, initial shots are detected using TransNetV2 [Soucek and Lokoc, 2024], generating the finest level of temporal granularity used in our experiments.

Scene-level grouping. In the scene-level setting, similarity is computed only between neighboring shots using multimodal features: visual similarity from CLIP ViT-B/32 [Radford et al., 2021] applied to the middle frame of each shot, caption similarity from CLIP ViT-B/32 applied to BLIP-generated captions [Li et al., 2022], audio similarity from PANNs [Kong et al., 2020], and transcript similarity from Sentence-Transformer embeddings [Reimers and Gurevych, 2019] of ASR text. These four signals are combined with weights 0.30 (visual), 0.25 (caption), 0.25 (audio), and 0.20 (ASR) to produce a single similarity score for each adjacent pair.

For each cut, we define a boundary value as 1 − similarity, where a higher boundary value indicates a stronger semantic or perceptual change between adjacent shots. To reduce noisy segmentation, the boundary sequence is smoothed with a median filter. For each video, a threshold $\mathbf { \bar { \rho } } _ { T }$ is set as the 75th percentile of that video’s boundary distribution. Neighboring shots are merged into the same group while their boundary value is less than or equal to $\bar { T } ;$ when the boundary exceeds $T ,$ a new group is started. To prevent excessively long grouped units, we impose a maximum group duration of 3 minutes.

## E Detailed Evaluation Metric Formulations

We provide the mathematical definitions and technical implementation details for the evaluation metrics introduced in the Experimental Setup section of the main paper.

Order preservation coefficient (OPC). Let $O = \left( o _ { 1 } , \ldots , o _ { n } \right)$ be the deduplicated sequence of selected shots in the outputs and let $p ( o _ { i } )$ be the original position of shot $o _ { i }$ in the source video. Excluding ties, we compute OPC as:

$$
{ \mathrm { O P C } } = \left\{ { \begin{array} { l l } { 1 , } & { n \leq 1 { \mathrm { ~ o r ~ } } P = 0 , } \\ { \displaystyle { \frac { C - D } { P } } , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{1}
$$

where $C$ and $D$ are the numbers of concordant and discordant shot pairs, and $P = C + D$

Keyword relevance (KR). We measure keyword relevance separately for narration/transcripts, video frames, and visual captions. Each modality is evaluated over its own set of approximately uniform textual or temporal units. The corresponding KR score is the unweighted mean cosine similarity between the target keyword and these units.

Narration and transcript relevance. For each segment in the generated timeline, we use the generated narration for narration-driven segments and the ASR transcript of the selected source clip otherwise. We concatenate the resulting text in timeline order and divide it at sentence boundaries into chunks of approximately 50 words. Let $\mathbf { k } ^ { \mathrm { { s t } } }$ denote the sentence-transformer embedding of the target keyword [Reimers and Gurevych, 2019], and let ${ \bf d } _ { j }$ denote the embedding of the j-th text chunk. Given J chunks, spoken-text relevance is defined as

$$
\mathrm { K R } _ { \mathrm { N a r r a t i o n } } = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \cos \left( \mathbf { k } ^ { \mathrm { s t } } , \mathbf { d } _ { j } \right) .\tag{2}
$$

Frame relevance. We uniformly sample the output timeline at 0.1 FPS. Let $\mathbf { k } ^ { \mathrm { c l i p } }$ denote the OpenCLIP ViT-B/32 embedding of the target keyword [Radford et al., 2021], and let $\mathbf { f } _ { m }$ denote the embedding of the m-th sampled frame. Given M sampled frames, frame relevance is computed as

$$
\mathrm { K R } _ { \mathrm { F r a m e } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \cos \left( \mathbf { k } ^ { \mathrm { c l i p } } , \mathbf { f } _ { m } \right) .\tag{3}
$$

Caption relevance. At each frame-sampling timestamp, we caption a 10-second source-video window centered on that timestamp using Qwen2-VL-7B-Instruct [Wang et al., 2024b]. Let $\mathbf { c } _ { m }$ denote the OpenCLIP text embedding of the caption associated with the m-th sampled window. Caption relevance is computed as

$$
\mathrm { K R } _ { \mathrm { C a p t i o n } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \cos \left( { \bf k } ^ { \mathrm { c l i p } } , { \bf c } _ { m } \right) .\tag{4}
$$

## F Subjective Evaluation Questions for Clips-to-storyboard Generation

In our subjective evaluation survey, we ask the following questions to evaluate our proposed pipelines for storyboard generation.

• Coherence: To what extent does the video maintain coherence and a smooth flow, such that each segment transitions logically and the overall viewing experience feels seamless?

• Keyword alignment: To what extent does the video align with the target keyword?

• Realness: To what extent does the video appear realistic and convincing overall?

• Narration effectiveness: To what extent do the narrations help you understand the connections between clips and guide your understanding of how the video progresses?

• Selected clips effectiveness: To what extent do the selected video clips provide useful and informative content for the video?

## G Subjective Evaluation Questions for Generated Textual Storyboards

In our subjective evaluation survey, we ask the following questions to evaluate our proposed pipelines for narration generation.

• Organization: To what extent does the narration present a well-organized and consistent storyline?

• Keyword informativeness: To what extent does the narration clearly introduce the main characters or events related to the target keyword?

• Engagingness: To what extent does the narration keep you engaged and build curiosity about what happens next?

![](images/36513fe9e026f26f0c10b5dde3dd46c7fdaa26f473a739f14f8b251ced87d332.jpg)  
Figure 7: Genre-wise comparison of CLIPScore, VTGHLS, OPC, KR-Narration, KR-Frame, and KR-Caption under the clip-level setting. Error bars indicate 95% confidence intervals. Specific numbers can be found in Table 7.

## H Additional Quantitative Results

In this section, we provide a more granular breakdown of our model’s performance. Specifically, we analyze the results across five distinct video genres under both scene-level and clip-level settings.

Table 6: Performance of A2Summ, TeaserGen, REGen, Clips2Story-NF, and Clips2Story-ND across five video genres under the scene-level setting, compared with the input video clips. Values are reported with 95% confidence intervals.
<table><tr><td>Model</td><td>MCD (sec)</td><td>REP (%)</td><td>CLIPScore</td><td>VTGHLS</td><td>OPC</td><td>KR-Narration ↑</td><td>KR-Frame ↑</td><td>KR-Caption ↑</td></tr><tr><td colspan="9">Documentary</td></tr><tr><td>Input video clips</td><td>4.13 ± 0.39</td><td>0.00 ± 0.00</td><td>0.49 ± 0.01</td><td>1.02 ± 0.03</td><td>1.00 ± 0.00</td><td>0.17 ± 0.02</td><td>0.17 ± 0.01</td><td>0.23 ± 0.02</td></tr><tr><td>A2Summ</td><td>3.99 ± 0.38</td><td>0.00 ± 0.00</td><td>0.50 ± 0.01</td><td>1.07 ± 0.02</td><td>0.05 ± 0.04</td><td>0.19 ± 0.02</td><td>0.17 ± 0.01</td><td>0.23 ± 0.02</td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td>3.48 ± 0.10</td><td>19.99 ± 3.28</td><td>0.67 ± 0.02</td><td>1.48 ± 0.02</td><td>0.04 ± 0.07</td><td>0.21 ± 0.02</td><td>0.19 ± 0.01</td><td>0.24 ± 0.02</td></tr><tr><td>REGen [Xu et al., 2025]</td><td>3.56 ± 0.41</td><td>20.50 ± 12.76</td><td>0.59 ± 0.04</td><td>1.00 ± 0.12</td><td>−0.07 ± 0.12</td><td>0.22 ± 0.03</td><td>0.19 ± 0.01</td><td>0.23 ± 0.03</td></tr><tr><td>Clips2Story-NF</td><td>19.03 ± 2.29</td><td>0.29 ± 0.29</td><td>0.57 ± 0.02</td><td>1.15 ± 0.03</td><td>0.48 ± 0.07</td><td>0.27 ± 0.02</td><td>0.19 ± 0.01</td><td>0.26 ± 0.03</td></tr><tr><td>Clips2Story-ND</td><td>7.78 ± 0.47</td><td>10.07 ± 1.08</td><td>0.58 ± 0.02</td><td>1.24 ± 0.03</td><td>0.31 ± 0.05</td><td>0.30 ± 0.02</td><td>0.20 ± 0.01</td><td>0.26 ± 0.03</td></tr><tr><td colspan="9">Film</td></tr><tr><td>Input video clips</td><td>9.66 ± 1.83</td><td>0.00 ± 0.00</td><td>0.47 ± 0.00</td><td>0.90 ± 0.02</td><td>1.00 ± 0.00</td><td>0.11 ± 0.01</td><td>0.16 ± 0.01</td><td>0.20 ± 0.02</td></tr><tr><td>A2Summ</td><td>9.18 ± 1.60</td><td>0.00 ± 0.00</td><td>0.49 ± 0.01</td><td>0.96 ± 0.01</td><td>0.10 ± 0.09</td><td>0.12 ± 0.01</td><td>0.16 ± 0.01</td><td>0.20 ± 0.01</td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td>4.08 ± 0.19</td><td>8.43 ± 2.25</td><td>0.59 ± 0.02</td><td>1.43 ± 0.01</td><td>0.01 ± 0.06</td><td>0.16 ± 0.02</td><td>0.17 ± 0.01</td><td>0.18 ± 0.02</td></tr><tr><td>REGen [Xu et al., 2025]</td><td>4.61 ± 0.57</td><td>10.55 ± 9.08</td><td>0.55 ± 0.04</td><td>0.66 ± 0.22</td><td>−0.06 ± 0.20</td><td>0.18 ± 0.02</td><td>0.17 ± 0.01</td><td>0.19 ± 0.04</td></tr><tr><td>Clips2Story-NF</td><td>27.14 ± 4.87</td><td>0.15 ± 0.10</td><td>0.51 ± 0.02</td><td>0.99 ± 0.03</td><td>0.40 ± 0.07</td><td>0.17 ± 0.02</td><td>0.18 ± 0.01</td><td>0.20 ± 0.02</td></tr><tr><td>Clips2Story-ND</td><td>9.93 ± 0.84</td><td>6.33 ± 0.69</td><td>0.51 ± 0.02</td><td>1.15 ± 0.02</td><td>0.28 ± 0.05</td><td>0.20 ± 0.02</td><td>0.18 ± 0.01</td><td>0.22 ± 0.02</td></tr><tr><td colspan="9">Lecture</td></tr><tr><td>Input video clips</td><td>165.22 ± 129.49</td><td>0.01 ± 0.01</td><td>0.50 ± 0.02</td><td>1.02 ± 0.02</td><td>1.00 ± 0.00</td><td>0.21 ± 0.02</td><td>0.19 ± 0.01</td><td>0.15 ± 0.02</td></tr><tr><td>A2Summ</td><td>28.21 ± 15.19</td><td>0.00 ± 0.00</td><td>0.55 ± 0.03</td><td>1.05 ± 0.05</td><td>0.33 ± 0.16</td><td>0.23 ± 0.02</td><td>0.19 ± 0.01</td><td>0.18 ± 0.03</td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td>4.82 ± 0.45</td><td>29.31 ± 6.42</td><td>0.60 ± 0.03</td><td>1.35 ± 0.07</td><td>0.16 ± 0.18</td><td>0.22 ± 0.02</td><td>0.21 ± 0.01</td><td>0.25 ± 0.04</td></tr><tr><td>REGen [Xu et al., 2025]</td><td>8.23 ± 3.95</td><td>14.74 ± 13.58</td><td>0.59 ± 0.04</td><td>0.55 ± 0.23</td><td>0.27 ± 0.29</td><td>0.22 ± 0.03</td><td>0.21 ± 0.01</td><td>0.26 ± 0.05</td></tr><tr><td>Clips2Story-NF</td><td>41.54 ± 8.15</td><td>0.47 ± 0.26</td><td>0.52 ± 0.06</td><td>0.98 ± 0.11</td><td>0.71 ± 0.09</td><td>0.26 ± 0.04</td><td>0.20 ± 0.01</td><td>0.19 ± 0.04</td></tr><tr><td>Clips2Story-ND</td><td>14.17 ± 2.63</td><td>17.85 ± 8.63</td><td>0.56 ± 0.03 1.18 ± 0.03</td><td></td><td>0.32 ± 0.13</td><td>0.30 ± 0.03</td><td>0.21 ± 0.01</td><td>0.22 ± 0.04</td></tr><tr><td colspan="9">News</td></tr><tr><td>Input video clips</td><td>8.44 ± 1.96</td><td>0.05 ± 0.01</td><td>0.53 ± 0.03</td><td>1.05 ± 0.03</td><td>1.00 ± 0.00</td><td>0.13 ± 0.02</td><td>0.15 ± 0.01</td><td>0.17 ± 0.02</td></tr><tr><td>A2Summ</td><td>8.51 ± 1.10</td><td>0.00 ± 0.00</td><td>0.58 ± 0.04</td><td>1.13 ± 0.04</td><td>0.21 ± 0.09</td><td>0.14 ± 0.02</td><td>0.15 ± 0.01</td><td>0.18 ± 0.02</td></tr><tr><td>TeaserGen [Xu et al., 2024] REGen [Xu et al., 2025]</td><td>4.26 ± 0.28 4.48 ± 1.12</td><td>19.38 ± 4.77 16.20 ± 12.32</td><td>0.60 ± 0.03</td><td>1.41 ± 0.03</td><td>−0.02 ± 0.11 0.13 ± 0.24</td><td>0.15 ± 0.02 0.14 ± 0.03</td><td>0.16 ± 0.01</td><td>0.18 ± 0.02</td></tr><tr><td></td><td></td><td></td><td>0.62 ± 0.05</td><td>0.79 ± 0.24</td><td></td><td></td><td>0.17 ± 0.01</td><td>0.20 ± 0.03</td></tr><tr><td>Clips2Story-NF Clips2Story-ND</td><td>24.87 ± 4.35 8.98 ± 1.33</td><td>3.20 ± 1.99 20.51 ± 3.70</td><td>0.62 ± 0.04 0.59 ± 0.03</td><td>1.14 ± 0.04 1.23 ± 0.03</td><td>0.61 ± 0.09 0.26 ± 0.05</td><td>0.24 ± 0.02 0.26 ± 0.02</td><td>0.20 ± 0.01 0.20 ± 0.01</td><td>0.22 ± 0.03 0.23 ± 0.03</td></tr><tr><td colspan="9"></td></tr><tr><td>Input video clips 74.36 ± 62.64</td><td></td><td>0.02 ± 0.01</td><td>Vlog 0.53 ± 0.02</td><td>0.99 ± 0.03</td><td>1.00 ± 0.00</td><td>0.13 ± 0.02</td><td>0.18 ± 0.01</td><td>0.25 ± 0.04</td></tr><tr><td>A2Summ</td><td>27.99 ± 6.63</td><td>0.00 ± 0.00</td><td>0.57 ± 0.03</td><td>1.01 ± 0.04</td><td>0.22 ± 0.15</td><td>0.13 ± 0.02</td><td>0.18 ± 0.01</td><td>0.24 ± 0.04</td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td>4.30 ± 0.23</td><td>17.64 ± 3.56</td><td>0.56 ± 0.04</td><td>1.39 ± 0.03</td><td>0.06 ± 0.14</td><td>0.16 ± 0.01</td><td>0.18 ± 0.01</td><td>0.24 ± 0.04</td></tr><tr><td>REGen [Xu et al., 2025]</td><td>3.37 ± 0.86</td><td>39.97 ± 16.51</td><td>0.53 ± 0.04</td><td>0.41 ± 0.23</td><td>0.26 ± 0.26</td><td>0.14 ± 0.03</td><td>0.19 ± 0.01</td><td>0.26 ± 0.04</td></tr><tr><td>Clips2Story-NF</td><td>31.18 ± 3.72</td><td>0.34 ± 0.24</td><td>0.58 ± 0.03</td><td>1.03 ± 0.03</td><td>0.58 ± 0.10</td><td>0.16 ± 0.02</td><td>0.19 ± 0.01</td><td>0.26 ± 0.04</td></tr><tr><td>Clips2Story-ND</td><td>12.07 ± 1.21</td><td>10.79 ± 1.26</td><td>0.56 ± 0.02</td><td>1.22 ± 0.03</td><td>0.19 ± 0.09</td><td>0.18 ± 0.02</td><td>0.19 ± 0.01</td><td>0.27 ± 0.04</td></tr></table>

Vlog  
News  
Table 7: Performance of A2Summ, TeaserGen, REGen, Clips2Story-NF, and Clips2Story-ND across five video genres under the clip-level setting, compared with the input video clips. Values are reported with 95% confidence intervals.
<table><tr><td>Model</td><td>MCD (sec)</td><td>REP (%)</td><td>CLIPScore</td><td>VTGHLS</td><td>OPC</td><td>KR-Narration ↑</td><td></td><td>KR-Frame ↑ KR-Caption ↑</td></tr><tr><td colspan="9">Documentary</td></tr><tr><td>Input video clips</td><td>4.13 ± 0.39</td><td>0.00 ± 0.00</td><td>0.49 ± 0.01</td><td>1.02 ± 0.03</td><td>1.00 ± 0.00</td><td>0.17 ± 0.02</td><td>0.17 ± 0.01</td><td>0.23 ± 0.02</td></tr><tr><td>A2Summ</td><td>3.87 ± 0.41</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.50 ± 0.01</td><td>1.07 ± 0.02</td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td> $0 . 2 1 \pm 0 . 0 2$ </td><td>0.17 ± 0.01</td><td>0.23 ± 0.02</td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td>3.48 ± 0.11</td><td>15.18 ± 3.37</td><td>0.65 ± 0.01</td><td>1.46 ± 0.02</td><td> $0 . 1 1 \pm 0 . 0 6$ </td><td> $0 . 2 1 \pm 0 . 0 2$ </td><td>0.19 ± 0.01</td><td>0.24 ± 0.02</td></tr><tr><td>REGen [Xu et al., 2025]</td><td>3.69 ± 0.23</td><td>9.80 ± 10.58</td><td>0.62 ± 0.02</td><td>0.94 ± 0.19</td><td>0.02 ± 0.07</td><td> $0 . 2 3 \pm 0 . 0 2$ </td><td>0.20 ± 0.01</td><td>0.23 ± 0.03</td></tr><tr><td>Clips2Story-NF</td><td> $5 . 3 8 \pm 0 . 5 1$ </td><td> $0 . 3 8 \pm 0 . 3 1$ </td><td>0.55 ± 0.02</td><td>1.13 ± 0.03</td><td> $0 . 2 9 \pm 0 . 0 8$ </td><td> $0 . 3 1 \pm 0 . 0 2$ </td><td> $0 . 1 9 \pm 0 . 0 1$ </td><td> $0 . 2 5 \pm 0 . 0 3$ </td></tr><tr><td>Clips2Story-ND</td><td> $4 . 2 8 \pm 0 . 2 4$ </td><td> $1 6 . 1 3 \pm 1 . 7 8$ </td><td>0.57 ± 0.01 1.29 ± 0.02</td><td></td><td> $0 . 1 3 \pm 0 . 0 4$ </td><td> ${ \bf 0 . 3 3 \pm 0 . 0 2 }$ </td><td></td><td>0.20 ± 0.01 0.27 ± 0.03</td></tr></table>

<table><tr><td>Input video clips</td><td>9.66 ± 1.83</td><td>0.00 ± 0.00</td><td>0.47 ± 0.00</td><td>0.90 ± 0.02</td><td>1.00 ± 0.00</td><td>0.11 ± 0.01</td><td>0.16 ± 0.01</td><td>0.20 ± 0.02</td></tr><tr><td>A2Summ</td><td>9.29 ± 1.75</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.49 ± 0.01</td><td>0.95 ± 0.01</td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td> $0 . 1 2 \pm 0 . 0 1$ </td><td>0.17 ± 0.01</td><td>0.20 ± 0.02</td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td>3.95 ± 0.20</td><td>7.65 ± 2.24</td><td>0.59 ± 0.02</td><td>1.42 ± 0.01</td><td> $0 . 0 4 \pm 0 . 0 6$ </td><td> $0 . 1 6 \pm 0 . 0 2$ </td><td>0.17 ± 0.01</td><td>0.19 ± 0.02</td></tr><tr><td>REGen [Xu et al., 2025]</td><td>4.48 ± 0.70</td><td> $1 9 . 8 7 \pm 1 2 . 7 7$ </td><td>0.53 ± 0.04 0.85 ± 0.14</td><td></td><td>0.08 ± 0.16</td><td>0.15 ± 0.03</td><td> $\mathbf { 0 . 1 8 \pm 0 . 0 1 }$ </td><td>0.18 ± 0.03</td></tr><tr><td>Clips2Story-NF</td><td> $1 0 . 7 6 \pm 1 . 5 4$ </td><td> $0 . 2 2 \pm 0 . 1 7$ </td><td>0.50 ± 0.01 0.99 ± 0.02</td><td></td><td> $0 . 2 7 \pm 0 . 0 6$ </td><td> $0 . 1 9 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 1 8 \pm 0 . 0 1 }$ </td><td>0.21 ± 0.02</td></tr><tr><td>Clips2Story-ND</td><td> $6 . 1 7 \pm 0 . 5 1$ </td><td> $8 . 9 9 \pm 0 . 9 3$ </td><td>0.52 ± 0.01 1.22 ± 0.01</td><td></td><td> $0 . 1 7 \pm 0 . 0 3$ </td><td> ${ \bf 0 . 2 2 \pm 0 . 0 2 }$ </td><td>0.18 ± 0.01</td><td>0.22 ± 0.02</td></tr></table>

<table><tr><td>Input video clips</td><td>165.22 ± 129.49</td><td>0.01 ± 0.01</td><td>0.50 ± 0.02</td><td>1.02 ± 0.02</td><td>1.00 ± 0.00</td><td>0.21 ± 0.02</td><td>0.19 ± 0.01</td><td>0.15 ± 0.02</td></tr><tr><td>A2Summ</td><td>31.32 ± 14.68</td><td>0.00 ± 0.00</td><td>0.57 ± 0.04</td><td>1.06 ± 0.05</td><td>0.17 ± 0.14</td><td> $0 . 2 3 \pm 0 . 0 2$ </td><td>0.19 ± 0.01</td><td>0.17 ± 0.03</td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td>4.72 ± 0.45</td><td>31.86 ± 5.99</td><td>0.61 ± 0.04</td><td>1.36 ± 0.07</td><td>0.13 ± 0.21</td><td>0.22 ± 0.02</td><td>0.21 ± 0.01</td><td>0.26 ± 0.05</td></tr><tr><td>REGen [Xu et al., 2025]</td><td> $7 . 4 0 \pm 1 . 7 3$ </td><td>19.38 ± 14.61</td><td>0.58 ± 0.05</td><td>0.42 ± 0.22</td><td>0.32 ± 0.19</td><td>0.21 ± 0.04</td><td>0.21 ± 0.01</td><td>0.27 ± 0.06</td></tr><tr><td>Clips2Story-NF</td><td> $2 3 . 2 0 \pm 5 . 8 7$ </td><td> $0 . 3 8 \pm 0 . 4 2$ </td><td>0.54 ± 0.06 0.99 ± 0.11</td><td></td><td> $0 . 1 9 \pm 0 . 2 5$ </td><td> $0 . 2 7 \pm 0 . 0 4$ </td><td> $0 . 2 0 \pm 0 . 0 1$ </td><td> $0 . 1 9 \pm 0 . 0 4$ </td></tr><tr><td>Clips2Story-ND</td><td> $1 0 . 6 8 \pm 2 . 0 4$ </td><td> $2 0 . 0 9 \pm 8 . 0 8$ </td><td>0.56 ± 0.03 1.20 ± 0.04</td><td></td><td> $0 . 1 7 \pm 0 . 1 6$ </td><td> $\mathbf { 0 . 3 2 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 2 1 \pm 0 . 0 1 }$ </td><td>0.22 ± 0.03</td></tr></table>

<table><tr><td>Input video clips</td><td>8.44 ± 1.96</td><td>0.05 ± 0.01</td><td>0.53 ± 0.03</td><td>1.05 ± 0.03</td><td>1.00 ± 0.00</td><td>0.13 ± 0.02</td><td>0.15 ± 0.01</td><td>0.17 ± 0.02</td></tr><tr><td>A2Summ</td><td>8.92 ± 1.84</td><td>0.00 ± 0.00</td><td>0.58 ± 0.04</td><td>1.13 ± 0.03</td><td> $0 . 0 2 \pm 0 . 0 6$ </td><td> $0 . 1 5 \pm 0 . 0 2$ </td><td>0.16 ± 0.01</td><td>0.18 ± 0.02</td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td>4.12 ± 0.25</td><td>16.14 ± 3.92</td><td>0.59 ± 0.03 1.41 ± 0.03</td><td></td><td>0.03 ± 0.11</td><td>0.14 ± 0.02</td><td>0.16 ± 0.01</td><td>0.20 ± 0.02</td></tr><tr><td>REGen [Xu et al., 2025]</td><td>4.87 ± 0.75</td><td>9.44 ± 7.42</td><td> $0 . 5 7 \pm 0 . 0 4 1 . 0 3 \pm 0 . 1 7$ </td><td></td><td> $- 0 . 0 3 \pm 0 . 1 1$ </td><td> $0 . 1 6 \pm 0 . 0 3$ </td><td> $0 . 1 7 \pm 0 . 0 1$ </td><td>0.19 ± 0.03</td></tr><tr><td>Clips2Story-NF</td><td> $1 0 . 4 1 \pm 2 . 7 9$ </td><td> $1 . 8 9 \pm 1 . 1 8$ </td><td> $0 . 5 8 \pm 0 . 0 3 1 . 1 2 \pm 0 . 0 4$ </td><td></td><td> $0 . 3 7 \pm 0 . 0 6$ </td><td> $0 . 2 5 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 2 0 \pm 0 . 0 1 }$ </td><td> $0 . 2 2 \pm 0 . 0 3$ </td></tr><tr><td>Clips2Story-ND</td><td>6.19 ± 1.07</td><td> $2 2 . 1 1 \pm 3 . 6 0$ </td><td>0.55 ± 0.03 1.25 ± 0.02</td><td></td><td> $0 . 1 3 \pm 0 . 0 3$ </td><td> ${ \bf 0 . 2 7 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 2 0 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 2 3 \pm 0 . 0 3 }$ </td></tr></table>

<table><tr><td>Input video clips</td><td>74.36 ± 62.64</td><td>0.02 ± 0.01</td><td>0.53 ± 0.02</td><td>0.99 ± 0.03</td><td>1.00 ± 0.00</td><td>0.13 ± 0.02</td><td>0.18 ± 0.01</td><td>0.25 ± 0.04</td></tr><tr><td>A2Summ</td><td>30.40 ± 6.97</td><td>0.00 ± 0.00</td><td>0.58 ± 0.02</td><td>1.03 ± 0.04</td><td>0.05 ± 0.07</td><td> $0 . 1 3 \pm 0 . 0 2$ </td><td>0.18 ± 0.01</td><td>0.25 ± 0.04</td></tr><tr><td>TeaserGen [Xu et al., 2024]</td><td>4.30 ± 0.15</td><td>13.48 ± 4.64</td><td>0.57 ± 0.04</td><td>1.38 ± 0.03</td><td>0.05 ± 0.14</td><td>0.17 ± 0.01</td><td>0.19 ± 0.01</td><td>0.26 ± 0.04</td></tr><tr><td>REGen [Xu et al., 2025]</td><td>4.08 ± 0.83</td><td>42.58 ± 19.27</td><td>0.58 ± 0.05</td><td>0.66 ± 0.25</td><td>−0.04 ± 0.22</td><td>0.16 ± 0.04</td><td> $0 . 1 9 \pm 0 . 0 2$ </td><td>0.26 ± 0.06</td></tr><tr><td>Clips2Story-NF</td><td>26.36 ± 3.71</td><td>0.42 ± 0.42</td><td>0.58 ± 0.03</td><td>1.02 ± 0.03</td><td>0.37 ± 0.11</td><td> $0 . 1 6 \pm 0 . 0 2$ </td><td>0.20 ± 0.01</td><td>0.27 ± 0.04</td></tr><tr><td>Clips2Story-ND</td><td>10.89 ± 1.15</td><td>11.68 ± 1.57</td><td>0.56 ± 0.02</td><td>1.23 ± 0.03</td><td>0.12 ± 0.09</td><td>0.18 ± 0.01</td><td>0.19 ± 0.01</td><td>0.28 ± 0.04</td></tr></table>