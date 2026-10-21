![](images/ae204ac85e9464990b4abaa93b7f967adbf710ff9d9054ecddd3e7b5eeaf5a13.jpg)

# EvaOmni: An Evidence-Grounded Benchmark for Audio-Visual Reasoning

Xiaoxing Hu<sup>1,2∗†</sup>, Wenxi Yao<sup>1∗</sup>, Chaoyuan Shi<sup>3∗</sup>, Ziqi Ye<sup>4,2∗</sup>, Xiaorong Guo<sup>4</sup>, Peixuan Li<sup>4</sup>, Zhuoran Sun<sup>5</sup>, Ziyang Gong<sup>1</sup>, Peiyuan Zhang<sup>1</sup>, Mingxin Liu<sup>1</sup>, Shuran Ma<sup>1</sup>, Hongbo Liu<sup>6</sup>, Xiaomin Yu<sup>7</sup>, Yu Tian<sup>8</sup>, Wentao Wang<sup>1</sup>, Zhenjie Yang<sup>9</sup>, Jian Hu<sup>10</sup>, Yunhang Shen<sup>10</sup>, Haoyu Cao<sup>10</sup>, Xue Yang<sup>1,B</sup>

<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>Shanghai Innnovation Institute, <sup>3</sup>Inner Mongolia University of Technology, <sup>4</sup>Fudan University, <sup>5</sup>Beijing Institute of Technology, <sup>6</sup>Tongji University, <sup>7</sup>The Hong Kong University of Science and Technology (Guangzhou), <sup>8</sup>Tsinghua University, <sup>9</sup>The University of Hong Kong, <sup>10</sup>Tencent Youtu Lab

<sup>†</sup>Project Lead, <sup>∗</sup>Core Contributors, <sup>B</sup>Corresponding Author

## Abstract

Omni LLMs can naturally locate key audio and visual evidence and integrate information across modalities for audiovisual reasoning. Existing audio-visual reasoning benchmarks that mainly rely on multiple-choice questions usually score models solely based on answer correctness, making it dificult to assess whether a model truly forms a reasoning chain from evidence to the correct answer. In addition, textual or unimodal shortcuts may allow models to answer correctly without analyzing the necessary audio-visual evidence, further weakening the reliability of evaluation benchmarks. To address these issues, we introduce EvaOmni, an evidence-grounded benchmark containing 392 videos and 1,027 purely human-annotated question-answer pairs, covering 15 capabilities and three levels: retrieval and localization, basic understanding, and complex reasoning. We design a series of rigorous annotation procedures in which experts annotate audio and visual evidence and construct video-based distractors; humans review whether questions meet strict inspection criteria, and finally models filter textual and unimodal shortcuts. Beyond high-quality question-answer pairs, we require models to simultaneously return the answer and supporting evidence intervals in order to evaluate the complete reasoning chain. We propose the Coverage Grounded Score (C.G.) to measure how much annotated evidence a model identifies, and the Bottleneck Grounded Score (B.G.) to penalize poor temporal localization in either modality; both are incorporated into answer correctness. Across 7 open-weight and 17 proprietary Omni LLMs, Gemini3.7-Flash achieved the highest overall accuracy (79.4%), C.G. (70.8%), and B.G. (35.1%). At Level 3, it reached 79.1% accuracy, but its B.G. was only 12.3%. These results show that even the most advanced models still have room for development in grounded audio-visual reasoning.

![](images/cdd1701ef48672e7f8d7506a5b0d3fc03524c1248fcb1b0a3f575bfce5bacd77.jpg)  
Figure 1 Existing audio-visual reasoning benchmarks sufer from shortcut issues, which weaken the reliability of benchmark evaluation. EvaOmni greatly reduces this phenomenon through a series of rigorous annotation procedures and a grounded evaluation protocol.

## 1 Introduction

Omni-modal large language models (Omni LLMs) [5, 25, 32, 33] can jointly process text, audio, images, and video, integrating cross-modal cues to interpret multimodal inputs and reason over the evidence they provide. Audio-visual reasoning in video scenarios is particularly dificult because it requires models to connect and analyze auditory and visual signals simultaneously. This involves identifying relevant evidence in each modality and integrating cues from diferent moments, abilities that support applications such as sports event understanding and the analysis of films and television programs. With the rapid development of Omni LLMs, accurately and reliably evaluating the capabilities of existing Omni LLMs has become particularly important, and is of great significance for identifying model shortcomings and driving their development.

Existing benchmarks for joint audio-visual reasoning [2, 14, 16, 22, 31, 36] cover everyday activities, real-world scenarios, and diverse tasks, providing valuable ways to assess Omni LLMs. However, our analysis reveals a key limitation of dominant multiple-choice question (MCQ) evaluations: a correct answer does not necessarily demonstrate joint use of audio and visual evidence. As illustrated in Fig. 1, some questions can be answered using a single modality or even textual cues alone. This ambiguity has two sources. First, answer-only scoring cannot distinguish evidence-supported answers from lucky guesses. Second, question and answer-option design can introduce unimodal or logical (text) shortcuts. For example, the correct option may be inferred from common knowledge, audio-only input, or vision-only input. Quantitative visualizations from existing benchmarks are provided in the Appendix D.1. Together, these limitations motivate an evaluation protocol that requires models to identify the audio and visual evidence supporting each answer, enabling a complete evaluation of the entire reasoning chain.

Motivated by these observations, we introduce EvaOmni, an EVidence-grounded benchmark for Audio-visual reasoning. It comprises 392 videos and 1,027 question-answer pairs spanning 15 fine-grained capabilities at three levels of reasoning and evidence complexity: Level 1-Retrieval & Grounding requires locating relevant audio and visual evidence; Level 2-Basic Understanding requires interpreting and combining complementary cues; and Level 3-Complex Reasoning requires understanding with complex evidence distribution, or performing multi-step cross-modal reasoning. Experts need to annotate the audio and visual intervals supporting each answer, and models must return both an answer and its supporting intervals. This design both greatly reduces annotation errors and prevents models from using shortcuts when answering questions. To ensure question quality, we designed a series of rigorous annotation procedures: systematic video selection, expert annotation, cross-checking, and model-assisted filtering. These procedures help reduce text and unimodal shortcuts. Such procedures ensure the high quality and reliability of our questions. We also propose two new evaluation metrics: the Coverage Grounded Score (C.G.), which measures the coverage of annotated evidence identified by the model, and the Bottleneck Grounded Score (B.G.), which penalizes inaccurate temporal localization in either modality. Both incorporate answer correctness to comprehensively evaluate the model’s reasoning ability. Additional grounded metrics are reported in the Appendix D.2. We conduct a systematic evaluation of EvaOmni, involving seven open-weight models and 17 proprietary Omni LLMs. The best-performing proprietary model, Gemini3.7-Flash, achieves 79.4% accuracy, 70.8% C.G., and 35.1% B.G. overall. The strongest open-weight model, MiMo-V2.5-310B-A15B, achieves 56.8% accuracy, 48.4% C.G., and 20.5% B.G. Gemini3.7-Flash leads by 22.6, 22.4, and 14.6 percentage points, respectively. Relative to Qwen3-Omni-Instruct-30B-A3B (25.2% accuracy, 4.0% C.G., and 0.4% B.G.), MiMo-V2.5-310B-A15B scores 2.3, 12.1, and 51.3 times as high on these metrics, respectively. Despite this progress, the strongest proprietary models remain well below the human reference in localization quality: Gemini3.7-Flash achieves 35.1% B.G., compared with 94.3% for human experts. These findings demonstrate the importance of grounded evaluation, which cannot be captured by previous MCQ-based benchmarks. We hope EvaOmni brings some new insights to the development of omni-modal models: future models should not only pursue correct answers but also be able to precisely localize the evidence supporting their answers, so that their conclusions can be verified against the audio-visual input. We have publicly released EvaOmni, including its data, annotations, and evaluation framework, to support reliable, interpretable, and hierarchical evaluation of audio-visual reasoning.

![](images/3f63aa0d4df9d7fb7989226dec19f13b2a383552f46d27316838ce12520b4ae9.jpg)  
Figure 2 Top: We designed a series of rigorous annotation procedures to ensure the quality and reliability of EvaOmni questions. Bottom: An example of Grounded Annotation, where, building on MCQ, we require models to output the audio and visual evidence intervals supporting the answer, in order to evaluate the complete reasoning chain and prevent blind guessing and shortcuts.

## 2 Related Work

Omni LLMs. Omni LLMs process text, images, video, and audio, enabling cross-modal understanding and reasoning [6, 7, 17]. ImageBind [8] learned shared representations across text, audio, depth, and images; PandaGPT [21] connected such representations to an LLM; and NExT-GPT [29] extended omni-modal systems to image, audio, and video generation. GPT-4o [15] introduced streaming speech generation for natural interaction. The Gemini [4, 23, 24] and Qwen [25, 32, 33] families are leading proprietary and open-weight Omni LLMs, respectively, advancing cross-modal understanding and reasoning. SeedRealtime [1] further integrates audio, video, and text in a full-duplex architecture for continuous scene understanding, interaction-timing awareness, and responsive behavior. Together, these systems mark a shift toward joint reasoning and natural interaction.

Benchmarks for Omni LLMs. Unlike benchmarks of individual image, audio, or video understanding [3, 20, 27, 28, 34, 35], Omni LLM benchmarks assess joint cross-modal reasoning. DailyOmni [36] covers everyday scenarios; WorldSense [14] emphasizes audio-visual understanding in real-world videos; and OmniVideoBench [16] requires reasoning over both modalities. Yet text and unimodal shortcuts can undermine answer-only scores, and existing benchmarks do not jointly score answer correctness and the localization of supporting audio-visual evidence. Another evaluation approach, Open-Ended, relies on judge models and is thus insuficiently objective and convenient. Therefore, the research community currently urgently needs an evaluation system that is objective, convenient, and rigorous.

## 3 EvaOmni: A New Evidence-Driven Paradigm for Omni-Modal Evaluation

## 3.1 Rethinking Audio-Visual Reasoning Benchmarks

Before introducing EvaOmni, let us revisit the existing audio-visual reasoning benchmarks. Table 1 compares seven representative benchmarks.

In the shortcut-checks column, T/A/V/I/Tr denote text-, audio-, video-, image-, and transcript-based tests; U denotes an unspecified unimodal test, D a dependency and logic check, and H human shortcut review. In the taxonomy column, cat., dims., dif., and caps. denote categories, dimensions, dificulty tiers, and capabilities. OmniBench uses static images, while SocialOmni focuses on social interaction rather than video question answering. Most existing benchmarks use multiple-choice questions (MCQs) and score responses by answer correctness. This raises RQ1: Can answer-only MCQ accuracy establish whether a correct answer is grounded in both audio and visual evidence? It cannot: the same credit is assigned to evidence-supported answers and unsupported correct choices. For four-option questions, uniform random guessing yields an expected accuracy of 25%, illustrating one way to obtain correct answers without using either modality. Answer accuracy also provides no measure of whether a model can locate the evidence supporting its prediction.

![](images/b2d26891b5855e171240cfa85231c18e8ef11e366c71f31f5bc4c58c25d47380.jpg)  
Figure 3 Grounded evaluation.

Table 1 Comparison of omni-modal benchmark designs. We compare shortcut checks, evidence annotation, grounded evaluation, and task taxonomies. Grounded Ann. requires question-level audio and visual evidence intervals; Grounded Eval. scores answer correctness and evidence grounding.
<table><tr><td rowspan="2">Benchmark</td><td colspan="2">Reliability Control</td><td colspan="2">Evaluation Design</td></tr><tr><td>Shortcut checks</td><td>Grounded Ann.</td><td>Grounded Eval.</td><td>Taxonomy</td></tr><tr><td>OmniBench</td><td>T/I/Tr/H</td><td>No</td><td>No</td><td>3 cat./8 tasks</td></tr><tr><td>Daily-Omni</td><td>T/H</td><td>No</td><td>No</td><td>6 tasks</td></tr><tr><td>WorldSense</td><td>V/H</td><td>No</td><td>No</td><td>3 levels/26 tasks</td></tr><tr><td>OmniVideoBench</td><td>T/U/H</td><td>No</td><td>No</td><td>13 tasks</td></tr><tr><td>JointAVBench</td><td>D/H</td><td>No</td><td>No</td><td>5 dims./15 tasks</td></tr><tr><td>LVOmniBench</td><td>T/A/V/H</td><td>No</td><td>No</td><td>3 diff./9 types</td></tr><tr><td>SocialOmni</td><td></td><td>No</td><td>No</td><td>3 tasks</td></tr><tr><td>EvaOmni (ours)</td><td>T/A/V/H</td><td>Yes</td><td>Yes</td><td>3 levels/15 caps.</td></tr></table>

We also ask RQ2: To what extent can existing questions be answered through textual, logical, or unimodal shortcuts? We evaluate three representative video question-answering benchmarks—Daily-Omni, WorldSense, and OmniVideoBench—under text-, audio-, and video-only inputs. Figure 1 illustrates shortcut cases; the full protocol and aggregate results appear in Appendix D.1. We found that a considerable proportion of questions in existing benchmarks can still be answered correctly without complete audio-visual input. For example, in WorldSense, 14.4% of questions can be answered correctly by looking only at the question and options, 14.2% can be answered correctly by listening only to the audio, and 15.7% can be answered correctly by watching only the video. Note that this does not represent a strict shortcut ratio (it depends on our test models), but it still indicates that many shortcut problems exist in existing benchmarks. These observations motivate RQ3: How can we build a benchmark that retains the objectivity and convenience of MCQs while effectively and reliably evaluating model performance in audio-visual reasoning? We address this question with the following EvaOmni.

## 3.2 Introducing EvaOmni

To address RQ3, we introduce EvaOmni, an evidence-grounded benchmark for audio-visual reasoning. For each question, expert annotators identify the audio and visual intervals supporting the correct answer. At evaluation time, models must provide both an answer and timestamped evidence intervals for each modality. EvaOmni evaluates answer correctness together with evidence coverage and temporal grounding, thereby limiting the credit awarded to correct answers that lack matching evidence. Text-only, audio-only, and vision-only checks during question construction further reduce shortcut susceptibility, as described below. EvaOmni organizes 15 fine-grained capabilities into three levels of reasoning and evidence complexity:

Level 1: Retrieval and Grounding. Tests temporal alignment, event binidng, and entity binding, such as identifying the speaker of an utterance or linking a sound to its visible source.

Level 2: Basic Understanding. Requires models to interpret and integrate complementary audio-visual cues to understand events, scene context, and their relationships.

Level 3: Complex Reasoning. Tests long-range reasoning, multi-evidence chains, and multi-hop reasoning. These tasks require models to integrate evidence from diferent moments, perform multiple inference steps, or both.

The three levels contain 3, 9, and 3 capabilities, respectively, as summarized in Fig. 4(a). The detailed capability distribution is provided in the Appendix A.5.

## 3.3 Data Collection

To mitigate potential training data contamination, we collect recent videos from open platforms such as YouTube and Bilibili. The selected 1–12-minute videos span diverse real-world scenarios and contain informative visual events and audio cues. We exclude videos with hard-coded subtitles or prominent textual overlays to limit OCR-based shortcuts. Detailed selection and screening criteria are provided in Appendix A.

## 3.4 Annotation Pipeline

Expert annotators construct questions requiring joint use of audio and visual information and mark the temporal evidence for each modality. Each sample then undergoes human cross-validation for capability alignment, evidence quality, and potential textual or unimodal shortcuts. Disputed pairs receive independent review by two additional experts and are discarded unless both approve them. Model-assisted screening under text-only, audio-only, and visual-only inputs identifies additional shortcuts; pairs flagged at this stage are revised and re-evaluated. Appendix A provides the full annotation and quality-control procedure.

## 3.5 Evaluation Metrics

EvaOmni adopts a composite evaluation metric that simultaneously assesses a model’s answer correctness and evidence localization ability. We primarily use answer accuracy (Acc.) as well as additional grounded metrics, which are introduced below. For each question, the model returns an answer and audio and visual evidence intervals specified by start and end timestamps. We report two grounded scores that combine the answer score s (s = 1 for a correct option and s = 0 otherwise) with evidence localization quality:

![](images/4dbb0ff77ba5cf853ce192962e67cc6686ca86d9fcaaf64f4eec0f2cb500df73.jpg)  
(a) Hierarchical Capability Taxonomy

![](images/2229e4e5e9f9f02869185aa24f35b4e696684f805376c4d3e2687ed0787c4fe7.jpg)  
(b) Video Duration Statistic

![](images/13d1e7ded2289addfa9bc6bd3042555626710ab1ea149a7ea1ab4f6864ba297a.jpg)  
(c) Token Length Statistic

![](images/d44732fdaf8ba5912705bc5ab741acd8760a7b65ab98da5ead63f03938252c0c.jpg)  
(d) Evidence Distribution Statistic

![](images/8c0dcb3e59314b81209a862226f89f5d9b0b355595b9f32428ce55bf9cb236e4.jpg)  
(e) Video Type Distribution  
Figure 4 Overview and statistics of EvaOmni. (a) Hierarchical capability taxonomy. (b) Video duration statistics. (c) Mean question and answer-option lengths in tokens by level. (d) Mean counts and durations of annotated audio and visual evidence segments. (e) Video type distribution.

Coverage Grounded Score (C.G.). This metric measures how much annotated audio and visual evidence is covered by the predictions. Let audio<sub>I</sub> and visual<sub>I</sub> be the numbers of human-annotated audio and visual evidence segments that have a positive-duration temporal overlap with at least one predicted segment in the same modality. Each annotated segment is counted at most once. Let $\mathrm { \ a u d i o _ { a l l } }$ and $\mathrm { \ v i s u a l _ { a l l } }$ denote the total numbers of annotated segments in the respective modalities. We define

$$
\mathrm { C . G . } = \frac { s } { 2 } \left( \frac { \mathrm { a u d i o } _ { I } } { \mathrm { a u d i o } _ { \mathrm { a l l } } } + \frac { \mathrm { v i s u a l } _ { I } } { \mathrm { v i s u a l } _ { \mathrm { a l l } } } \right) .\tag{1}
$$

C.G. averages the two modality-specific evidence hit rates and weights them by answer correctness. We interpret C.G. as a lenient evidence hit-rate metric, not a measure of precise temporal localization: any positive-duration overlap counts as a hit, and overly broad predictions are not directly penalized for imprecise interval boundaries.

Bottleneck Grounded Score (B.G.). Let $\mathrm { I o U _ { a u d i o } }$ and $\mathrm { { I o U } _ { \mathrm { v i s u a l } } }$ denote the temporal intersection over union between predicted and human-annotated evidence for the audio and visual modalities, respectively. For each modality, we first take the union of all predicted intervals and the union of all annotated intervals; their IoU is the duration of the intersection of these two sets divided by the duration of their union. We define

$$
\mathrm { B . G . } = s \times \operatorname* { m i n } ( \mathrm { I o U } _ { \mathrm { a u d i o } } , \mathrm { I o U } _ { \mathrm { v i s u a l } } ) .\tag{2}
$$

Taking the minimum makes B.G. sensitive to the less accurately localized modality: strong localization in one modality cannot compensate for poor localization in the other. Thus, C.G. measures broad evidence coverage, whereas B.G. requires accurate localization in both modalities. Additional grounded metrics are provided in Appendix D.2.

## 4 Experiments

## 4.1 Evaluated Models

We evaluate 24 Omni LLMs on EvaOmni, comprising 7 open-weight and 17 proprietary models. The open-weight models include Qwen2.5-Omni-7B [32], Qwen3-Omni-Instruct and Qwen3-Omni-Thinking [33],

Table 2 Overall performance on EvaOmni. We report Acc., C.G., and B.G. for each capability level and overall. All values are in %. The best results are shown in bold, and the second-best results are underlined.
<table><tr><td rowspan="2">Model</td><td colspan="3">Level1 R.&amp;G.</td><td colspan="3">Level2 B.U.</td><td colspan="3">Level3 C.R.</td><td colspan="3">Overall</td></tr><tr><td>Acc.</td><td>C.G. B.G.</td><td></td><td>Acc.</td><td>C.G. B.G.</td><td></td><td>Acc.</td><td>C.G.</td><td>B.G.</td><td>Acc.</td><td>C.G.</td><td>B.G.</td></tr><tr><td colspan="9">Open-Weight Omni LLMs</td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-Omni-7B</td><td>16.6</td><td>1.2</td><td>0.6</td><td>17.4</td><td>0.5</td><td>0.0</td><td>13.7</td><td>0.3</td><td>0.0</td><td>16.4 0.8</td><td></td><td>0.3</td></tr><tr><td>女 Qwen3-Omni-Instruct-30B-A3B</td><td>28.6</td><td>5.1</td><td>0.6</td><td>22.4</td><td>3.4</td><td>0.4</td><td>22.2</td><td>2.4</td><td>0.1</td><td>25.2</td><td>4.0</td><td>0.4</td></tr><tr><td>女 Qwen3-Omni-Thinking-30B-A3B</td><td>26.3</td><td>11.4</td><td>4.8</td><td>24.4</td><td>6.2</td><td>1.2</td><td>21.6</td><td>4.1</td><td>0.6</td><td>24.8</td><td>8.3</td><td>2.8</td></tr><tr><td>MIDOA Nemotron3-Nano-Omni-30B-A3B</td><td>30.7</td><td>15.0</td><td>5.3</td><td>22.1</td><td>13.1</td><td>3.9</td><td>10.5</td><td>4.7</td><td>0.7</td><td>24.2</td><td>12.6</td><td>4.0</td></tr><tr><td>Xia MiMo-V2.5-310B-A15B</td><td>65.4</td><td>61.3</td><td>30.8</td><td>52.3</td><td>42.3</td><td>15.1</td><td>45.1</td><td>29.6</td><td>6.7</td><td>56.8</td><td>48.4</td><td>20.5</td></tr><tr><td>MiP MiMo-V2.6-Flash-310B-A15B</td><td>61.3</td><td>55.3</td><td>29.0</td><td>48.6</td><td>41.4</td><td>14.5</td><td>45.8</td><td>28.6</td><td>5.9</td><td>53.6</td><td>45.3</td><td>19.3</td></tr><tr><td>Xini MiMo-V2.6-Pro-1T-A42B</td><td>60.1</td><td>51.3</td><td>28.8</td><td>44.3</td><td>35.2</td><td>12.0</td><td>49.7</td><td>26.6</td><td>5.0</td><td>51.8</td><td>40.7</td><td>18.1</td></tr><tr><td colspan="11">Proprietary Omni LLMs</td></tr><tr><td>Doubao-Seed-2.0-Lite</td><td>54.1</td><td>36.4</td><td>10.8</td><td>54.1</td><td>36.1</td><td>11.7</td><td>62.7</td><td>32.2</td><td>5.9</td><td>55.4</td><td>35.6</td><td>10.4</td></tr><tr><td>Doubao-Seed-2.1-Lite</td><td>78.8</td><td>74.8</td><td>38.0</td><td>67.0</td><td>61.5</td><td>22.6</td><td>59.5</td><td>42.7</td><td>9.4</td><td>70.9</td><td>64.3</td><td>27.2</td></tr><tr><td>立 Qwen3.5-Omni-Flash</td><td>47.5</td><td>41.9</td><td>15.5</td><td>30.9</td><td>24.0</td><td>7.2</td><td>29.4</td><td>15.7</td><td>2.7</td><td>37.7</td><td>30.4</td><td>10.0</td></tr><tr><td>女 Qwen3.5-Omni-Plus</td><td>58.1</td><td>52.0</td><td>22.1</td><td>42.3</td><td>26.0</td><td>8.6</td><td>42.5</td><td>19.8</td><td>3.3</td><td>49.0</td><td>36.0</td><td>13.5</td></tr><tr><td>女 Qwen3.8-Omni-Flash</td><td>65.0</td><td>53.6</td><td>19.3</td><td>52.3</td><td>41.4</td><td>11.8</td><td>56.2</td><td>35.9</td><td>7.8</td><td>58.2</td><td>45.7</td><td>14.4</td></tr><tr><td>Gemini2.5-Flash</td><td>59.2</td><td>55.2</td><td>26.7</td><td>39.3</td><td>31.4</td><td>11.8</td><td>41.8</td><td>24.6</td><td>4.0</td><td>48.1</td><td>40.4</td><td>16.9</td></tr><tr><td>Gemini2.5-Flash-Lite</td><td>53.0</td><td>47.2</td><td>22.1</td><td>30.9</td><td>22.4</td><td>8.0</td><td>27.5</td><td>14.8</td><td>2.2</td><td>39.7</td><td>31.8</td><td>13.1</td></tr><tr><td>Gemini2.5-Pro</td><td>76.5</td><td>73.8</td><td>42.3</td><td>65.0</td><td>55.0</td><td>22.5</td><td>72.5</td><td>45.8</td><td>9.4</td><td>71.0</td><td>61.6</td><td>28.9</td></tr><tr><td>Gemini3.1-Flash-Lite</td><td>59.4</td><td>52.8</td><td>24.2</td><td>36.6</td><td>30.7</td><td>10.5</td><td>32.0</td><td>19.0</td><td>3.0</td><td>45.6</td><td>38.3</td><td>15.2</td></tr><tr><td>Gemini3.1-Flash-Lite-Preview</td><td>59.7</td><td>53.6</td><td>25.2</td><td>39.3</td><td>31.1</td><td>10.3</td><td>38.6</td><td>23.2</td><td>3.3</td><td>47.8</td><td>39.4</td><td>15.6</td></tr><tr><td>Gemini3.1-Pro-Preview</td><td>74.7</td><td>72.8</td><td>41.5</td><td>68.6</td><td>59.8</td><td>24.8</td><td>76.5</td><td>48.7</td><td>11.2</td><td>72.3</td><td>63.7</td><td>29.8</td></tr><tr><td>Gemini3.5-Flash</td><td>64.1</td><td>59.4</td><td>28.8</td><td>69.5</td><td>61.4</td><td>27.4</td><td>77.8</td><td>47.8</td><td>11.3</td><td>68.5</td><td>58.6</td><td>25.6</td></tr><tr><td>Gemini3.5-Flash-Lite</td><td>60.6</td><td>55.6</td><td>26.5</td><td>38.2</td><td>27.9</td><td>10.7</td><td>42.5</td><td>24.2</td><td>3.5</td><td>48.3</td><td>39.1</td><td>16.3</td></tr><tr><td>Gemini3.6-Flash</td><td>80.9</td><td>78.0</td><td>44.7</td><td>68.4</td><td>61.3</td><td>26.1</td><td>73.2</td><td>44.2</td><td>10.6</td><td>74.4</td><td>65.8</td><td>31.7</td></tr><tr><td>Gemini3.7-Flash</td><td>85.3</td><td>84.0</td><td>49.1</td><td>73.6</td><td>65.3</td><td>29.2</td><td>79.1</td><td>49.1</td><td>12.3</td><td>79.4</td><td>70.8</td><td>35.1</td></tr><tr><td>Gemini3.8-Flash</td><td>83.4</td><td>81.9</td><td>48.4</td><td>73.9</td><td>63.3</td><td>27.4</td><td>74.5</td><td>43.1</td><td>9.5</td><td>78.0</td><td>68.2</td><td>33.6</td></tr><tr><td>Gemini4-Argon</td><td>67.1</td><td>55.9</td><td>29.3</td><td>52.0</td><td>33.7</td><td>13.0</td><td>72.5</td><td>33.5</td><td>7.5</td><td>61.4</td><td>43.0</td><td>19.1</td></tr><tr><td>Human Expert</td><td>97.8</td><td>97.8</td><td>95.8</td><td>96.5</td><td>94.8</td><td>93.2</td><td>94.4</td><td>93.8</td><td>93.0</td><td>96.8</td><td>96.0</td><td>94.3</td></tr></table>

![](images/307151487610fb07065f9fa252b49d15aad212dd3ffe9bc6002fbe9da7afd58d.jpg)

![](images/a8e15a02398dea9e8425dac512e368be9b28329af3cf20542970cae5cbcc6607.jpg)

![](images/0bb5cdc57e38815ee239e17b4354588cf7d843a968e3a1b6a88ff83801912311.jpg)  
Figure 5 Overall performance rankings for selected models. We select the top two Gemini models and the top Doubao, Qwen, and MiMo models by overall Acc. in Table 2, then rank the same five models independently by Acc., C.G., and B.G. Model1: Gemini3.7-Flash; Model2: Gemini3.8-Flash; Model3: Doubao-Seed-2.1-Lite; Model4: Qwen3.8-Omni-Flash; Model5: MiMo-V2.5-310B-A15B. Model IDs remain fixed across panels, which share the same score scale.

Nemotron3-Nano-Omni [5], MiMo-V2.5-310B-A15B [30], MiMo-V2.6-Flash-310B-A15B, and MiMo-V2.6- Pro-1T-A42B [18]. The proprietary models include Qwen3.5-Omni-Flash and Qwen3.5-Omni-Plus [25], Qwen3.8-Omni-Flash [19], Doubao-Seed-2.0-Lite [26], Doubao-Seed-2.1-Lite [26], and 12 Gemini variants spanning the Pro, Flash, and Flash-Lite series [4, 9–13]. The complete model list is provided in Table 2. We also report human expert performance as a reference for comparing answer correctness and evidence grounding; its protocol is described in Appendix B.2.

## 4.2 Experimental Settings

We evaluate all 24 models on 1,027 question-answer pairs drawn from 392 videos, using the same prompt and preprocessed audio-visual content. Video frames are uniformly sampled at 1.0 FPS and resized so that their longer side is 480 pixels while preserving the aspect ratio. Open-weight models outside the MiMo family are evaluated on a single server equipped with eight NVIDIA H20 GPUs, using the inference frameworks and settings recommended in their model cards. The open-weight MiMo models and all proprietary models are evaluated through their providers’ oficial APIs. The prompt, input preprocessing procedures, and further evaluation details are provided in the Appendix.

## 4.3 Main Findings

Table 2 reports performance across capability levels and overall, while Figure 6 compares two representative models with human performance. We summarize the key findings below.

Overall Performance Analysis. (1) A substantial gap remains between the strongest evaluated open-weight and proprietary models. MiMo-V2.5-310B-A15B achieves overall Acc., C.G., and B.G. scores of 56.8%, 48.4%, and 20.5%, respectively, whereas Gemini3.7-Flash reaches 79.4%, 70.8%, and 35.1%. The respective gaps of 22.6, 22.4, and 14.6 percentage points span answer correctness, evidence coverage, and precise dual-modal localization. (2) The gap to human experts is especially pronounced in precise grounding. Human experts achieve 96.8% Acc., 96.0% C.G., and 94.3% B.G., exceeding Gemini3.7-Flash by 17.4, 25.2, and 59.2 percentage points, respectively. The model’s 79.4% accuracy and 70.8% C.G. indicate strong answering performance and substantial evidence coverage, yet its 35.1% B.G. reveals persistent dificulty in precisely localizing evidence in both modalities. Although grounded scores also incorporate answer correctness, the much smaller decline from Acc. to B.G. for humans (2.5 points, versus 44.3 for Gemini3.7-Flash) highlights the remaining grounding challenge. (3) Grounding limitations are more severe among the four open-weight models with fewer than 100B total parameters. Their overall C.G. ranges from 0.8% to 12.6% and B.G. from 0.3% to 4.0%, compared with 30.4%–70.8% and 10.0%–35.1% for the proprietary models. For example, Qwen2.5-Omni-7B achieves 16.4% accuracy but only 0.8% C.G. and 0.3% B.G. Jointly requiring correct answers and grounded evidence thus exposes limitations that answer-only evaluation can obscure. These models leave substantial room for improvement in consistently satisfying both dual-modal localization and answer correctness. In Figure 5, we compare the rankings of five models. Model rankings by grounding score broadly follow those by Acc., although a few shifts in position highlight diferences in evidence localization among models with similar answer accuracy.

Finding 1: Grounded evaluation shows that even leading models trail humans in evidence localization, while open-weight models under 100B parameters lag behind proprietary models in evidence perception and localization.

Analysis Across Capability Levels. Answer accuracy decreases from Level 1 to Level 2 for 21 of the 24 models, but increases from Level 2 to Level 3 for 13 models. For example, Gemini3.7-Flash improves from 73.6% to 79.1% accuracy between Levels 2 and 3, while its C.G. drops from 65.3% to 49.1% and B.G. from 29.2% to 12.3%. This divergence is consistent with the design of Level 3, which emphasizes long-range reasoning, combinations of multiple evidence segments, and multi-hop inference. These tasks place greater demands on assembling and localizing supporting evidence, without necessarily producing a monotonic decline in multiple-choice accuracy. One plausible explanation is that partial evidence can help eliminate distractors and narrow the candidate set, allowing a model to select the correct answer without fully localizing the supporting evidence. The aggregate results are compatible with this explanation, but do not directly establish the models’ answer-selection strategies. In contrast, Level 3 yields the lowest B.G. for all 24 models, including one tie, and the lowest C.G. for all 24 models. This broad trend across open-weight and proprietary models highlights the value of grounded evaluation: C.G. rewards evidence coverage alongside correct answers, whereas B.G. rewards accurate temporal localization in both modalities alongside correct answers. Reporting Acc. together with these two metrics therefore provides a more complete assessment of answering, evidence coverage, and precise grounding across capability levels.

![](images/448c4f3996928c648857178d86f487dd5b88081cf13e74bbd3f6efb62d3c2f1e.jpg)  
Figure 6 Overall performance.

Table 3 Cost efficiency. Cost eficiency is USD per unit of normalized score (↓), computed from the Overall results in Table 2; the lowest value in each column is bold.
<table><tr><td colspan="6">Total cost Model</td></tr><tr><td></td><td>(USD)</td><td>Acc.</td><td>C.G.</td><td>B.G.</td></tr><tr><td>0 Doubao-Seed-2.1-Lite</td><td>13.71</td><td>0.0188</td><td>0.0208</td><td>0.0491</td></tr><tr><td>女 Qwen3.8-Omni-Flash</td><td>11.54</td><td>0.0193</td><td>0.0246</td><td>0.0780</td></tr><tr><td>Gemini3.1-Flash-Lite</td><td>18.67</td><td>0.0399</td><td>0.0475</td><td>0.1196</td></tr><tr><td>Gemini3.7-Flash </td><td>44.41</td><td>0.0545</td><td>0.0611</td><td>0.1232</td></tr><tr><td>女 Qwen3.5-Omni-Flash</td><td>41.90</td><td>0.1082</td><td>0.1342</td><td>0.4080</td></tr><tr><td>1 Gemini3.5-Flash</td><td>88.93</td><td>0.1264</td><td>0.1478</td><td>0.3382</td></tr><tr><td>Gemini2.5-Flash</td><td>67.68</td><td>0.1370</td><td>0.1631</td><td>0.3900</td></tr><tr><td>Gemini3.1-Pro-Preview 1</td><td>118.57</td><td>0.1597</td><td>0.1812</td><td>0.3874</td></tr><tr><td>女 Qwen3.5-Omni-Plus</td><td>130.55</td><td>0.2594</td><td>0.3531</td><td>0.9416</td></tr><tr><td>Gemini2.5-Pro</td><td>224.48</td><td>0.3079</td><td>0.3548</td><td>0.7563</td></tr></table>

Finding 2: As the task level increases, evidence-grounding quality declines, while answer accuracy does not follow the same trend.

Cost-Efficiency Analysis. Table 3 reports cost-eficiency calculations for 10 selected proprietary models. We calculate total costs from provider API prices under a common setting of 1,027 requests, each with a 10-minute audio-visual input. Cost eficiency is calculated separately using the Overall Acc., C.G., and B.G. in Table 2: for total cost C, request count N, and percentage score s, the corresponding cost eficiency is C/(Ns/100). This gives the USD cost per unit of accumulated normalized score; lower values indicate better eficiency in answer correctness, evidence coverage, and precise dual-modal localization, respectively. Doubao-Seed-2.1-Lite achieves the best cost eficiency on all three metrics, at \$0.0188, \$0.0208, and \$0.0491, respectively, with a total cost of \$13.71. Qwen3.8-Omni-Flash has the lowest total cost (\$11.54) and ranks second on all three metrics, at \$0.0193, \$0.0246, and \$0.0780. Gemini3.7-Flash achieves the highest Overall scores among these models on all three metrics, but ranks fourth in each corresponding cost-eficiency column, at \$0.0545, \$0.0611, and \$0.1232. Nevertheless, it outperforms Gemini2.5-Pro across all three metrics while reducing total cost from \$224.48 to \$44.41 and lowering the respective costs per unit of normalized score by 82.3%, 82.8%, and 83.7%. These results show that stronger performance does not necessarily require higher spending or imply better cost eficiency. Model selection should jointly consider the budget and the required levels of answer correctness, evidence coverage, and localization precision.

Finding 3: Higher scores do not necessarily imply better cost eficiency across Acc., C.G., and B.G. Model selection should balance answering and grounding performance with budget constraints.

## 4.4 Further Analysis

Input Modality Ablation. In Table 4 and the following analysis, we abbreviate MiMo-V2.5-310B-A15B, Qwen3VL-8B-Instruct, and Qwen3VL-Instruct-30B-A3B as MiMo-V2.5, Qwen3VL-8B, and Qwen3VL-A3B, respectively. Table 4 compares three models under diferent input conditions, yielding three observations. (1)

Table 4 Input modality ablation. Bold marks the best input setting for each model and metric.
<table><tr><td>Model</td><td>Input</td><td>Acc.</td><td>C.G. B.G.</td><td></td></tr><tr><td rowspan="4">MiMo-V2.5</td><td>Audio</td><td>30.0</td><td>10.0</td><td>0.0</td></tr><tr><td>ASR</td><td>23.8</td><td>7.1</td><td>0.0</td></tr><tr><td>Video</td><td>26.8</td><td>9.3</td><td>0.0</td></tr><tr><td>Video+ASR Video+Audio</td><td>48.7</td><td>38.1</td><td>11.6</td></tr><tr><td rowspan="2">Qwen3VL-8B</td><td>Video</td><td>56.8 13.8</td><td>48.4 0.3</td><td>20.5 0.0</td></tr><tr><td>Video+ASR</td><td>23.2</td><td>9.7</td><td>1.9</td></tr><tr><td rowspan="2">Qwen3VL-A3B</td><td>Video</td><td>8.5</td><td>0.7</td><td>0.0</td></tr><tr><td>Video+ASR</td><td>19.3</td><td>7.0</td><td>0.8</td></tr></table>

Table 5 Answer–grounding diagnostics. A/V: unweighted audio/visual IoU; C: correct answer; G: both IoUs $\geq \mathbf { 0 . 5 }$ . Here ¬ means “not” and a comma means “and”. Bold marks the highest A, V, and C, G values and the lowest values in the three error columns.
<table><tr><td>Model</td><td>A</td><td>V</td><td>C,G</td><td>C,¬G</td><td>¬C,G</td><td>¬C,¬G</td></tr><tr><td>MiMo-V2.5</td><td>41.1</td><td>39.3</td><td>21.5</td><td>35.3</td><td>8.3</td><td>34.7</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>30.5</td><td>29.8</td><td>13.1</td><td>36.6</td><td>5.5</td><td>44.7</td></tr><tr><td>Qwen3.8-Omni-Flash</td><td>27.4</td><td>34.3</td><td>11.8</td><td>46.5</td><td>4.4</td><td>37.3</td></tr><tr><td>Gemini2.5-Flash</td><td>40.2</td><td>36.7</td><td>18.2</td><td>31.3</td><td>8.8</td><td>41.5</td></tr><tr><td>Gemini2.5-Pro</td><td>47.8</td><td>46.2</td><td>32.4</td><td>38.9</td><td>7.2</td><td>21.5</td></tr><tr><td>Gemini3.5-Flash</td><td>45.6</td><td>41.9</td><td>27.0</td><td>41.0</td><td>6.8</td><td>25.2</td></tr><tr><td>Gemini3.7-Flash</td><td>52.6</td><td>51.1</td><td>41.8</td><td>38.2</td><td>5.8</td><td>14.3</td></tr></table>

Native audio outperforms ASR transcripts in the evaluated MiMo-V2.5 settings. With audio alone, MiMo-V2.5 achieves 30.0% Acc. and 10.0% C.G., compared with 23.8% and 7.1% using ASR alone, respectively. The advantage persists when video is available: Video+Audio achieves 56.8% Acc., 48.4% C.G., and 20.5% B.G., exceeding Video+ASR by 8.1, 10.3, and 8.9 percentage points. Thus, ASR transcripts do not fully substitute for audio in these settings. Non-speech sounds, prosody, and transcription or timestamp errors may contribute to this gap. (2) Removing either modality substantially degrades answering and grounding. Relative to Video+Audio, MiMo-V2.5’s accuracy drops by 26.8 percentage points with Audio alone and 30.0 points with Video alone. C.G. similarly falls from 48.4% to 10.0% and 9.3%, respectively, while B.G. drops from 20.5% to 0.0% in both cases. These declines highlight the value of complementary audio-visual evidence. (3) ASR improves vision-language models, but substantial grounding limitations remain. Adding ASR to video improves Qwen3VL-8B’s accuracy from 13.8% to 23.2% and Qwen3VL-A3B’s from 8.5% to 19.3%, gains of 9.4 and 10.8 percentage points. Their C.G. scores also increase from 0.3% to 9.7% and from 0.7% to 7.0%, respectively. Nevertheless, their B.G. scores remain at only 1.9% and 0.8%, compared with MiMo-V2.5’s 11.6% under the same Video+ASR condition. Transcripts therefore provide useful additional evidence, but do not by themselves ensure efective evidence integration and precise localization.

Finding 4: Both modalities improve answer accuracy. Native audio outperforms ASR, while ASR improves the evaluated VLMs but leaves their precise grounding performance low.

Error Analysis. Table 5 decomposes answering and grounding outcomes for seven models; details are in Appendix D.3. Audio IoU exceeds visual IoU by 0.7–3.7 points for six of the seven models; Qwen3.8-Omni-Flash is the only exception (27.4% versus 34.3%). The two failure modes are asymmetric. Requiring both modality IoUs to reach 0.5, 48.1%–79.9% of correct answers fail the criterion (C, ¬G): Gemini3.7-Flash answers 79.4% correctly but satisfies both criteria on only 41.8%, and Qwen3.8-Omni-Flash drops from 58.2% to 11.8%. Conversely, 12.2%–32.6% of grounding-passing predictions still choose the wrong option (¬C, G), and ¬C, G never exceeds 8.8% of all questions, so accurate dual-modal localization is rarely reached where answers are wrong. Therefore, neither answer correctness nor temporal overlap alone is suficient to establish reliable evidence-grounded answering.

Finding 5: Answer correctness does not guarantee accurate dual-modal grounding, and grounding success does not guarantee a correct answer.

## 5 Conclusion

We introduce EvaOmni, a new evidence-grounded benchmark for audio-visual reasoning. Rather than evaluating only whether a model selects the correct answer, EvaOmni also requires the model to identify the supporting evidence necessary to justify its prediction. Through a series of rigorous annotation and qualitycontrol procedures, EvaOmni comprises 1,027 questions organized into three hierarchical levels, providing diverse scenarios and rich data for comprehensive evaluation. We further introduce multiple Grounded Score metrics to evaluate the accuracy of evidence localization. Among them, the Coverage Grounded Score rewards evidence coverage, while the Bottleneck Grounded Score prevents strong localization in one modality from masking failure in the other. Additional Grounded Score variants are provided in the Appendix. Large-scale evaluation shows that combining grounded evaluation with accuracy reveals insights missed by accuracy alone. EvaOmni provides a reproducible, interpretable basis for measuring progress and identifies precise temporal grounding as the capability current Omni LLMs most clearly lack. We have fully released the data, annotations, and evaluation framework, and we hope they will serve as a valuable resource for the research community to advance the development of Omni LLMs.

## References

[1] ByteDance Seed. Seedrealtime audio-visual full-duplex llm released: Toward omni-modal natural interaction. https://seed.bytedance.com/en/blog/seedrealtime-audio-visual-full-duplex-llm-released-toward-o mni-modal-natural-interaction, August 2026.

[2] Jianghan Chao, Wenhui Tan, Yuchong Sun, Ruihua Song, Liyun Ru, et al. Jointavbench: A benchmark for joint audio-visual reasoning evaluation. In International Conference on Learning Representations, volume 2026, pages 84035–84062, 2026.

[3] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.

[4] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[5] Amala Sanjay Deshmukh, Kateryna Chumachenko, Tuomas Rintamaki, Matthieu Le, Tyler Poon, Danial Mohseni Taheri, Ilia Karmanov, Guilin Liu, Jarno Seppanen, Arushi Goel, et al. Nemotron 3 nano omni: eficient and open multimodal intelligence. arXiv preprint arXiv:2604.24954, 2026.

[6] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Yuhang Dai, Meng Zhao, Yi-Fan Zhang, Shaoqi Dong, Yangze Li, Xiong Wang, et al. Vita: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024.

[7] Chaoyou Fu, Haojia Lin, Xiong Wang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, Long MA, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. Advances in Neural Information Processing Systems, 38:75300–75320, 2026.

[8] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind one embedding space to bind them all. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15180–15190. IEEE, 2023.

[9] Google Deepmind. Gemini 3.1 Pro . https://deepmind.google/models/gemini/pro/, 2026.

[10] Google Deepmind. Gemini 3.6 Flash . https://blog.google/innovation-and-ai/models-and-research/gemi ni-models/gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber/, 2026.

[11] Google Deepmind. Introducing Gemini 3.7 Flash. https://blog.google/innovation-and-ai/models-and-res earch/gemini-models/introducing-gemini-3-7-flash/, 2026.

[12] Google Deepmind. Introducing Gemini 3.8 Flash and 3.8 Flash Cyber. https://blog.google/innovation-and -ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/, 2026.

[13] Google Deepmind. Gemini 4 Argon: our next era of frontier intelligence . https://blog.google/innovation-a nd-ai/models-and-research/gemini-models/gemini-4-argon/, 2026.

[14] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. In International Conference on Learning Representations, volume 2026, pages 52423–52443, 2026.

[15] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[16] Caorui Li, Yu Chen, Yiyan Ji, Jin Xu, Zhenyu Cui, Shihao Li, Yuanxing Zhang, Zhenghao Song, Dingling Zhang, Ying He, et al. Omnivideobench: Towards audio-visual understanding evaluation for omni mllms. In International Conference on Learning Representations, volume 2026, pages 138214–138236, 2026.

[17] Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model. arXiv preprint arXiv:2502.04328, 2025.

[18] Zongming Qiao, Ziyue Hua, Zirui Ou, Zihao Yue, Zihan Jiang, Zhuo Huang, Zhiyang Chen, Zhixian Zheng, Zhipeng Xu, Zhengrui Ma, et al. Mimo-v2. 6: Scaling reinforcement learning towards self-improvement. arXiv preprint arXiv:2610.11959, 2026.

[19] Qwen Team. Qwen3.8-omni-flash: Omni senses. agentic delivery., September 2026. URL https://qwen.ai/blog ?id=qwen3.8-omni-flash.

[20] Sakshi Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. In International Conference on Learning Representations, volume 2025, pages 84929–84964, 2025.

[21] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. In Proceedings of the 1st Workshop on Taming Large Language Models: Controllability in the Era of Interactive Assistants!, pages 11–23, 2023.

[22] Keda Tao, Yuhua Zheng, Jia Xu, Wenjie Du, Kele Shao, Hesong Wang, Xueyi Chen, Xin Jin, Junhan Zhu, Bohan Yu, et al. Lvomnibench: Pioneering long audio-video understanding evaluation for omnimodal llms. arXiv preprint arXiv:2603.19217, 2026.

[23] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[24] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

[25] Qwen Team. Qwen3.5-omni technical report. arXiv preprint arXiv:2604.15804, 2026.

[26] Volcano Engine. Coding plan personal model release record, September 2026. URL https://ark.volcengine.c om/region:cn-beijing/model/detail?name=doubao-seed-2-1-lite.

[27] Dingdong Wang, Junan Li, Jincenzi Wu, Dongchao Yang, Xueyuan Chen, Tianhua Zhang, and Helen Meng. Mmsu: A massive multi-task spoken language understanding and reasoning benchmark. In International Conference on Learning Representations, volume 2026, pages 31374–31410, 2026.

[28] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. Lvbench: An extreme long video understanding benchmark. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 22958–22967. IEEE, 2025.

[29] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023.

[30] Xiaomi MiMo. MiMo-V2.5. https://mimo.xiaomi.com/mimo-v2-5/, 2026.

[31] Tianyu Xie, Jinfa Huang, Yuexiao Ma, Rongfang Luo, Yan Yang, Qingchuan Ma, Wang Chen, Yuhui Zeng, Yixuan Zou, Zhiqiang Lu, et al. Socialomni: Benchmarking audio-visual social interactivity in omni models. arXiv preprint arXiv:2603.16859, 2026.

[32] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025.

[33] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.

[34] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.

[35] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, et al. Mlvu: Benchmarking multi-task long video understanding. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13691–13701. IEEE, 2025.

[36] Ziwei Zhou, Rui Wang, Zuxuan Wu, and Yu-Gang Jiang. Daily-Omni: Towards audio-visual reasoning with temporal alignment across modalities. arXiv preprint arXiv:2505.17862, 2025.

## Appendix Contents

A Dataset Construction and Annotation Details 14   
A.1 Video Collection and Filtering . 14   
A.2 Annotation Principles and Schema 15   
A.3 Human Cross-Validation 15   
A.4 Model-Assisted Shortcut Screening 15   
A.5 EvaOmni Structure 16   
B Experimental Details 16   
B.1 Input Preprocessing and Inference Setup 16   
B.2 Human Expert Baseline 17   
C Prompt Templates . 17   
D Additional Quantitative Results and Analysis 17   
D.1 Shortcut Evaluation and Case Studies 17   
D.2 Additional Grounded Evaluation Results 31   
D.3 Details of Failure Mode Analysis 31   
D.4 Analysis Across Fine-Grained Capabilities 33   
E Limitations and Future Work 33   
E.1 Limitations 33   
E.2 Future Work 33

## A Dataset Construction and Annotation Details

## A.1 Video Collection and Filtering

We select candidate videos according to four criteria, aiming to reduce potential training data contamination and textual shortcuts while providing suficient cues for constructing audio-visual reasoning questions.

Temporal freshness and potential contamination screening. We consider only videos uploaded after January 2025. As an additional screening step, we randomly sample three frames from each video and provide them to Gemini3.1-Pro-Preview, asking it to describe the video content. Videos for which the model can reconstruct the full content from these frames are flagged as potential contamination risks and excluded from the candidate pool. This procedure is a heuristic: neither the upload date nor the model’s ability to reconstruct the content establishes whether a video was present in its training data.

Avoiding subtitle-based textual shortcuts. We exclude videos with hard-coded subtitles to prevent models from substituting on-screen text recognition for audio understanding. This criterion supports the intended requirement that questions depend jointly on audio and visual information.

Manageable video duration. We select videos lasting 1–12 minutes to balance temporal scope, model input budgets, and annotation feasibility. At a sampling rate of 1.0 FPS, this range corresponds to approximately 60–720 frames, helping control the visual input size while retaining temporally structured event sequences.

Extending the collection to longer videos is a direction for future work on audio-visual reasoning over longer time spans.

Sufficient diversity of entities and events. We prioritize videos with varied participants, actions, or event developments. Videos with highly repetitive content, such as a single person continuously performing one action, may provide too few cues for constructing plausible distractors. We manually exclude videos that do not support adequate distractor construction, reducing opportunities to answer questions through unimodal cues or simple elimination.

## A.2 Annotation Principles and Schema

Five senior PhD students with research backgrounds in computer science and specialized knowledge of linguistics participate in annotation. Each sample records the video URL, question, answer options, capability level, and fine-grained capability category. Annotators separately mark the visual and audio evidence supporting the answer, with each evidence segment containing start and end timestamps in mm:ss format and a content description. For Level 3 multi-hop reasoning questions, the annotation additionally includes a step-by-step reasoning chain.

Question and distractor design. Each question must match its assigned capability category and level and be designed to require joint use of audio and visual information. Distractors draw on visual and audio cues present in the video so that a single modality is insuficient to reliably distinguish the correct option from the alternatives. They also represent logically plausible alternative explanations, limiting opportunities to eliminate incorrect options solely through wording or logical cues in the question and answer choices.

## A.3 Human Cross-Validation

After initial annotation, each expert’s question-answer pairs are randomly divided into four parts, which are randomly assigned, one to each of the other four experts, for cross-validation. Each expert therefore reviews approximately as many pairs as they initially annotated. The reviewer checks three dimensions:

1. Capability alignment: whether the question matches its assigned fine-grained capability and capability level.

2. Shortcut susceptibility: whether the question can be reliably answered using only the question and options, audio alone, or vision alone through logical or unimodal cues.

3. Evidence quality: whether the audio and visual evidence intervals and descriptions are accurate and support the annotated answer.

Approximately 78% of the initially submitted pairs pass this first review. The remaining disputed pairs are independently reviewed by two additional experts, neither of whom is the original annotator or the first reviewer. A pair is discarded if either additional expert identifies a problem; it passes only if both approve it. Overall, approximately 82% of the initially submitted pairs pass human cross-validation, while the remaining 18% are discarded at this stage.

## A.4 Model-Assisted Shortcut Screening

Following human validation, we use Gemini3.1-Pro-Preview and Qwen3-Omni-Instruct-30B-A3B for additional shortcut screening under text-only, audio-only, and vision-only conditions. We inspect both the answers and their supporting rationales, as well as the evidence from the available modality in unimodal settings. If a model can reliably answer a question correctly and provide a plausible supporting rationale despite the absence of one or both modalities, the sample is returned for revision. Revised samples must pass model-assisted screening again before inclusion.

The resulting EvaOmni benchmark contains 1,027 question-answer pairs curated through this annotation and quality-control process.

## A.5 EvaOmni Structure

EvaOmni contains 392 videos and 1,027 question-answer pairs across three capability levels. The main text defines the three levels; here we define the fine-grained capabilities within each one. Level 1 (436 questions).

1. Temporal Alignment (165 questions): Identify the main content of one modality using information from the other modality.

2. Event Binding (143 questions): Summarize the main event in one modality using information from the other modality.

3. Entity Binding (128 questions): Link the main content of one modality to the corresponding content in the other modality.

Level 2 (438 questions).

1. Causal Reasoning (127 questions): Infer the causes of events.

2. Emotional Intelligence (87 questions): Interpret emotional states from audio-visual cues.

3. Intent Understanding (56 questions): Infer a person’s intent from audio-visual cues.

4. Atmosphere Understanding (28 questions): Interpret the atmosphere of a scene from audio-visual cues.

5. Implicit Event Reasoning (18 questions): Infer unstated events, such as an omitted or likely future event.

6. Semantic Disambiguation (30 questions): Resolve the specific meaning of ambiguous speech or behavior using context from both modalities.

7. Temporal Understanding (31 questions): Determine the order, overlap, or duration relationships between audio and visual events.

8. Spatial Understanding (30 questions): Reason about spatial locations and relationships.

9. Conditional Counting (31 questions): Count entities or events that satisfy specified conditions.

## Level 3 (153 questions).

1. Multi-hop Reasoning (83 questions): Answer questions that require multiple linked inference steps.

2. Long-range Reasoning (35 questions): Integrate dispersed evidence whose most widely separated start times are at least four minutes apart.

3. Multi-Evidence Chain (35 questions): Reason over at least five temporally separated pieces of evidence.

## B Experimental Details

## B.1 Input Preprocessing and Inference Setup

Input preprocessing. For evaluation, all videos are resized to 480 pixels on the longer side while preserving their original aspect ratios. Frames are uniformly sampled at 1.0 FPS and provided to the Omni LLMs together with the corresponding audio. For the 1–12-minute videos in EvaOmni, this configuration keeps the audio-visual inputs within the context windows of the evaluated models.

Inference setup. All open-weight models except those in the MiMo family are evaluated on a single server equipped with eight NVIDIA H20 GPUs, using the inference frameworks and parameter settings recommended in their respective model cards. The three MiMo models are evaluated through MiMo’s oficial API platform with the same input preprocessing settings described above. Since the weights of these three models have been publicly released, we classify them as open-weight models.

## B.2 Human Expert Baseline

The human expert baseline is obtained from test participants who are independent of the annotators who constructed the questions and marked the reference evidence. They answer the same questions using the same test videos as the models and report supporting audio and visual intervals; their responses are scored using the same Acc., C.G., and B.G. metrics. Participants may pause and replay the videos when answering and locating evidence. This allowance helps compensate for human memory, which is much more limited than a model’s access to the supplied video context, but it also makes the viewing protocol diferent from model inference. Even with replay, the need to retain and revisit details across a video may disadvantage humans, especially for long or temporally dispersed evidence. We therefore treat the human scores as a reference baseline rather than a strictly matched comparison of memory or viewing conditions.

## C Prompt Templates

Figures 7, 8, 9, 10, 11, and 12 show the prompt templates used for the six input settings.

## D Additional Quantitative Results and Analysis

## D.1 Shortcut Evaluation and Case Studies

Figure 1 summarizes the shortcut evaluation results for three representative benchmarks for Omni LLMs, alongside EvaOmni. Here, we provide further details of the evaluation protocol and report the complete results in Table 6.

Evaluation protocol. For the visual-only condition, denoted as Video Only in the table, we use Qwen3-Omni-30B-A3B-Instruct<sup>1</sup> and Qwen3-VL-30B-A3B-Instruct<sup>2</sup>. For the audio-only condition, we use Qwen3-Omni-30B-A3B-Instruct and Audio-Flamingo-3<sup>3</sup>. For the text-only condition, we use Gemini3.1-Pro-Preview<sup>4</sup> and Qwen3.8-Flash<sup>5</sup>, providing only the question and answer options. Under each input condition, we evaluate each model twice, yielding four sets of predictions, and take the intersection of the sets of correctly answered questions across all four runs. The Intersection column reports the percentage of all evaluated questions answered correctly in every run. This design aims to reduce the influence of chance successes in a single run and biases specific to an individual model, although it does not eliminate these factors.

Interpreting the results. These results serve as diagnostic indicators of potential shortcut susceptibility. Consistently correct answers from multiple models under unimodal or text-only inputs do not necessarily imply that a question is flawed, nor do they alone establish that the models exploit a shortcut. Instead, they indicate the extent to which correct answers can be repeatedly obtained without full audio-visual input under the tested models and settings, highlighting a potential vulnerability of answer-only evaluation. The reported rates should therefore not be interpreted directly as the proportion of flawed questions or the probability that a benchmark can be exploited.

Illustrative Shortcut Cases. Figures 13–18, 19–24, and 25–30 show two text-only, two visual-only, and two audio-only shortcut cases from each of WorldSense, OmniVideoBench, and Daily-Omni, respectively.

![](images/dac386bb450869a4298900320f0b06aab436dc322324ea3cec4d3c5a7d6981ed.jpg)

Figure 7 Prompt used for audio-only inputs.  
![](images/789b7d1396b5161f1c324fe1f013d55c0afaa85b976115785ddce92540cb8e9a.jpg)  
Figure 8 Prompt used for text-only inputs.

![](images/82829a78cdad016e06f335d31b1589d26ea9c5874978067ab77bf853598d90c3.jpg)

Figure 9 Prompt used for video-only inputs.  
![](images/9380a6cbce2ebe88baf780b64db5e3035f80825ed775c785f7efed729bc77f50.jpg)  
Figure 10 Prompt used for Video+Audio inputs.

![](images/de860e834f79705e0f4eb6f1552f1c9f2530830cacc691b622a03ad7eafe6b2e.jpg)

Figure 11 Prompt used for ASR-only inputs.  
![](images/ae8b947713c26b4dcd94f546be01a1b997d10ec3d6d456de3b2a597a34dc0cfa.jpg)  
Figure 12 Prompt used for Video+ASR inputs.

![](images/31ddc877e211f4160081cbca7b9fff6c366b764e910df948d12d0aecc69692de.jpg)

Figure 13 Text-only shortcut case 1 from WorldSense.  
![](images/16e6bd3e28b46fae8512ebe579b9a9340abb26221056ab90de4496a4bea7d193.jpg)  
Figure 14 Text-only shortcut case 2 from WorldSense.

![](images/2ad3000800e8058aee62055e78114da2b8d0200ce4c1569400ce08f0381094bc.jpg)

Figure 15 Visual-only shortcut case 1 from WorldSense.  
![](images/bf208d093621c3f17714532d5d4a2ab315bdc11773cc2c8e40a357bc1103b373.jpg)  
Figure 16 Visual-only shortcut case 2 from WorldSense.

![](images/e1cd91d0e18d6a3c8801aa2e3d54660a26f2c7634418e3f8dfd4a3b2a345cba8.jpg)

Figure 17 Audio-only shortcut case 1 from WorldSense.  
![](images/9c61f9ad6fd0098013102b303e62d431c343a674bb8c0a0185e00dd87073647f.jpg)  
Figure 18 Audio-only shortcut case 2 from WorldSense.

Table 6 Benchmark comparison under text-only, audio-only, and video-only inputs. All values are percentages. Each model is evaluated twice, and an answer is considered correct only if both independent runs produce the correct answer.; Intersection denotes the proportion of evaluated questions answered correctly in all four runs under the corresponding input condition.  
(a) Daily-Omni
<table><tr><td>Input</td><td>Model 1</td><td>Model 2</td><td>Intersection</td></tr><tr><td>Text Only</td><td>45.1</td><td>38.6</td><td>28.4</td></tr><tr><td>Audio Only</td><td>30.2</td><td>57.9</td><td>25.6</td></tr><tr><td>Video Only</td><td>41.2</td><td>45.2</td><td>29.4</td></tr></table>

(b) WorldSense
<table><tr><td>Input</td><td>Model 1</td><td>Model 2</td><td>Intersection</td></tr><tr><td>Text Only</td><td>24.7</td><td>24.5</td><td>14.4</td></tr><tr><td>Audio Only</td><td>20.7</td><td>46.0</td><td>14.2</td></tr><tr><td>Video Only</td><td>25.5</td><td>39.8</td><td>15.7</td></tr></table>

(c) OmniVideoBench
<table><tr><td>Input</td><td>Model 1</td><td>Model 2</td><td>Intersection</td></tr><tr><td>Text Only</td><td>29.2</td><td>25.8</td><td>16.0</td></tr><tr><td>Audio Only</td><td>13.6</td><td>30.0</td><td>8.2</td></tr><tr><td>Video Only</td><td>22.8</td><td>26.8</td><td>12.1</td></tr></table>

(d) EvaOmni
<table><tr><td>Input</td><td>Model 1</td><td>Model 2</td><td>Intersection</td></tr><tr><td>Text Only</td><td>9.7</td><td>3.8</td><td>1.8</td></tr><tr><td>Audio Only</td><td>3.2</td><td>18.2</td><td>1.0</td></tr><tr><td>Video Only</td><td>7.3</td><td>12.5</td><td>3.6</td></tr></table>

Model 1 / Model 2 by input condition:  
Text Only: Gemini3.1-Pro-Preview / Qwen3.8-Flash.  
Audio Only: Audio-FlamingGo-3 / Qwen3-Omni-Instruct-30B-A3B.  
Video Only: Qwen3VL-Instruct-30B-A3B / Qwen3-Omni-Instruct-30B-A3B.

![](images/6d2cb7878a1ef60c7ac06090e6af9c4eef6def2f1aa0bbf4f404bd7c8fa0644d.jpg)

Figure 19 Text-only shortcut case 1 from OmniVideoBench.  
![](images/d0e97f554bbd6bce9672988f9053700a2132878ddb7c38baafcb2c9c364cbf74.jpg)  
Figure 20 Text-only shortcut case 2 from OmniVideoBench.

![](images/d31cbfd89f8ad7d99d0e3ec14f8bdb9125a50cdcfd4c0f0e33cbebd7e9142a41.jpg)

Figure 21 Visual-only shortcut case 1 from OmniVideoBench.  
![](images/ff555a4904b44774765388c9a5742c3ef0d96add7a29eee004fcda2c7c08c45e.jpg)  
Figure 22 Visual-only shortcut case 2 from OmniVideoBench.

![](images/6d7b103192664b4ec702c132dd3f53054bba2e0e5b199038ee0b52d39c91968c.jpg)

Figure 23 Audio-only shortcut case 1 from OmniVideoBench.  
![](images/ffc548c308426cc4183477adcf9dc5205c1169013a2c91276f106de7619c1d81.jpg)  
Figure 24 Audio-only shortcut case 2 from OmniVideoBench.

![](images/1ebd93902839c77a4667bf3755d462ccc3d1f1adef8d01b12bdb8df619d239cd.jpg)

Figure 25 Text-only shortcut case 1 from Daily-Omni.  
![](images/c238663aae014ec5ad55dc7031d994717fb2bb57201dca833a181f1385ef1d1a.jpg)  
Figure 26 Text-only shortcut case 2 from Daily-Omni.

![](images/465fae695aa84c12191cd901d05d1ca478c273b0be6bc77af4721204e9be85fa.jpg)

Figure 27 Visual-only shortcut case 1 from Daily-Omni.  
![](images/60ab54e7d619a5873268174236eba356c5fecc8e303d32db47762c9f46a5ef00.jpg)  
Figure 28 Visual-only shortcut case 2 from Daily-Omni.

![](images/f998f66537e03765bf960073bd1efd638bd2f55c0afeb0c64ca04524535e9454.jpg)

Figure 29 Audio-only shortcut case 1 from Daily-Omni.  
![](images/2e6010e34df8e6da6ec528bc746d8047ea2a91e27cd8d77e25410d79a52c22d7.jpg)  
Figure 30 Audio-only shortcut case 2 from Daily-Omni.

## D.2 Additional Grounded Evaluation Results

In addition to the Bottleneck Grounded Score (B.G.) and Coverage Grounded Score (C.G.) reported in the main text, we report a continuous score and threshold-based scores derived from the average temporal IoU across audio and visual evidence. These metrics provide complementary views of performance under diferent localization requirements.

For sample i, let $s _ { i }$ denote the answer score defined in the main text, and let $a _ { i }$ and $v _ { i }$ denote the temporal IoUs for audio and visual evidence, respectively. Their average is

$$
m _ { i } = { \frac { a _ { i } + v _ { i } } { 2 } } .\tag{3}
$$

For an evaluation set of N samples, we define

$$
\mathrm { M e a n } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } m _ { i } ,\tag{4}
$$

$$
\mathrm { T h r @ } \tau = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } { \bf 1 } \{ m _ { i } \ge \tau \} , \qquad \tau \in \{ 0 . 3 , 0 . 5 , 0 . 7 \} ,\tag{5}
$$

where 1{·} is the indicator function. All results are reported as percentages.

Mean continuously measures the joint performance of answering and average evidence localization across modalities; it is not a localization metric independent of answer correctness. For MCQs with binary answer scores, Thr@τ is the fraction of all samples that are answered correctly and have an average IoU of at least τ , rather than a pass rate computed only over correct answers. Higher thresholds impose stricter requirements on average localization. Unlike B.G., which is sensitive to the weaker modality, average IoU allows stronger localization in one modality to compensate for weaker localization in the other. Passing a threshold on average IoU therefore does not require both modality-specific IoUs to exceed that threshold.

Overall performance. Table 7 shows that Gemini3.7-Flash achieves the highest overall Mean (43.7%), followed by Gemini3.8-Flash (42.3%) and Gemini3.6-Flash (40.4%). Among the updated open-weight models, MiMo-V2.5 obtains the highest overall Mean at 26.3%, trailing Gemini3.7-Flash by 17.4 percentage points. Thus, a substantial gap between the leading open-weight and proprietary models persists when the weaker-modality IoU used in B.G. is replaced by average IoU.

Performance across capability levels. Mean decreases from Level 1 to Level 2 and again to Level 3 for all 20 models. For example, Gemini3.7-Flash scores 57.0%, 37.6%, and 23.5% across the three levels, while MiMo-V2.5 scores 37.2%, 20.9%, and 10.5%. This pattern is consistent with the greater demands on evidence integration and temporal localization at higher levels. Level 3 remains challenging even when localization quality can be traded of across modalities. Because these scores also incorporate answer correctness, however, the decline cannot be attributed entirely to localization errors.

Sensitivity to localization thresholds. Stricter average-IoU thresholds substantially reduce scores. Gemini3.7- Flash obtains overall scores of 63.7%, 49.9%, and 26.8% at thresholds of 0.3, 0.5, and 0.7, respectively; the corresponding scores for MiMo-V2.5 are 39.4%, 27.0%, and 13.6%. Thus, many correctly answered samples satisfying a permissive localization criterion fail to meet a stricter one. Rankings also depend on the threshold: Gemini3.7-Flash leads in overall Mean, Thr@0.3, and Thr@0.5, whereas Gemini3.8-Flash has a slightly higher Thr@0.7 score (27.8% versus 26.8%). Continuous scores and results at multiple thresholds therefore provide complementary information about average performance and success under stricter localization requirements.

## D.3 Details of Failure Mode Analysis

Scope and validation. We analyze the available predictions from Gemini3.7-Flash, Gemini3.5-Flash, Qwen3.5- Omni-Plus, Gemini2.5-Pro, Gemini2.5-Flash, Qwen3.8-Omni-Flash, and MiMo-V2.5-310B-A15B. Each model is evaluated on the same subset of the available annotations, matched by question ID. We also verify video IDs, questions, options, levels, and reference answers for every record.

Table 7 Additional grounded evaluation results across capability levels. Mean averages the answer score weighted by the mean audio and visual IoU. Thr@τ averages the answer score after gating it by whether that mean IoU reaches τ, for $\tau \in \{ 0 . 3 , 0 . 5 , 0 . 7 \}$ . All values are percentages; higher is better. As in Table 2, the first four open-weight models retain results from the earlier evaluation. Within each model group, bold and underlined values indicate the best and second-best distinct scores in each column, respectively; ties receive the same formatting.
<table><tr><td rowspan="2">Model</td><td colspan="4">Level 1</td><td colspan="4">Level 2</td><td colspan="4">Level 3</td><td colspan="4">Overall</td></tr><tr><td>Mean Thr@0.3 Thr@0.5 Thr@0.7 Mean Thr@0.3 Thr@0.5 Thr@0.7 Mean Thr@0.3 Thr@0.5 Thr@0.7 Mean Thr@0.3 Thr@0.5 Thr@0.7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Open-Weight Omni LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-Omni-7B</td><td>0.7</td><td>0.9</td><td>0.9</td><td>0.5</td><td>0.1</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.4</td><td>0.4</td><td>0.0</td><td>0.0</td></tr><tr><td>Qwen3-Omni-Instruct-30B-A3B</td><td>2.8</td><td>5.3</td><td>2.5</td><td>0.0</td><td>1.2</td><td>1.1</td><td>0.6</td><td>0.0</td><td>0.5</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1.8</td><td>2.9</td><td>1.4</td><td>0.0 1.5</td></tr><tr><td>Qwen3-Omni-Thinking-30B-A3B</td><td>6.5</td><td>10.2 11.3</td><td>7.4 7.6</td><td>2.8</td><td>2.5</td><td>3.4</td><td>1.1</td><td>0.6</td><td>1.0</td><td>0.7</td><td>0.0</td><td>0.0</td><td>4.1</td><td>6.0</td><td>3.8</td><td>2.1</td></tr><tr><td>i Nemotron3-Nano-Omni-30B-A3B</td><td>7.7</td><td>56.7</td><td></td><td>3.2</td><td>5.9</td><td>9.5</td><td>5.3</td><td>1.7</td><td>1.7</td><td>2.0</td><td>0.0</td><td>0.0</td><td>6.1</td><td>9.1</td><td>5.5</td><td></td></tr><tr><td>MiMo-V2.5-310B-A15B</td><td>37.2</td><td></td><td>42.4</td><td>22.4</td><td>20.9</td><td>31.1</td><td>20.5</td><td>9.3</td><td>10.5</td><td>14.4</td><td>2.0</td><td>1.3</td><td>26.3</td><td>39.4</td><td>27.0</td><td>13.6</td></tr><tr><td>MiMo-V2.6-Flash-310B-A15B 1</td><td>35.3</td><td>52.3</td><td>43.8</td><td>20.3</td><td>21.0</td><td>32.5</td><td>20.5</td><td>8.4</td><td>11.0</td><td>17.0</td><td>2.0</td><td>0.0</td><td>25.5</td><td>38.6</td><td>27.6</td><td>12.2</td></tr><tr><td>MiMo-V2.6-Pro-1T-A42B</td><td>33.0</td><td>48.6</td><td>40.6</td><td>20.3</td><td>17.5</td><td>27.5</td><td>16.4</td><td>7.0</td><td>9.5</td><td>9.8</td><td>2.0</td><td>0.7</td><td>22.8</td><td>33.8</td><td>24.4</td><td>11.7</td></tr><tr><td colspan="10">Proprietary Omni LLMs</td><td colspan="7"></td></tr><tr><td>Doubao-Seed-2.0-Lite 0 Doubao-Seed-2.1-Lite</td><td>17.8</td><td>30.2</td><td>15.4</td><td>4.4</td><td>18.7</td><td>28.6</td><td>16.8</td><td>8.9</td><td>12.6</td><td>17.0</td><td>2.6</td><td>0.7</td><td>17.4</td><td>27.6</td><td>14.1</td><td>5.7</td></tr><tr><td>Qwen3.5-Omni-Flash</td><td>47.5</td><td>71.9</td><td>56.5</td><td>27.0</td><td>30.8</td><td>46.6</td><td>30.0</td><td>14.1</td><td>16.9</td><td>23.5</td><td>6.5</td><td>1.3</td><td>35.8</td><td>53.8</td><td>37.7</td><td>17.6</td></tr><tr><td></td><td>20.7</td><td>34.3</td><td>18.4</td><td>6.9</td><td>10.2</td><td>15.2</td><td>8.9</td><td>3.2</td><td>4.9</td><td>3.9</td><td>1.3</td><td>0.7</td><td>13.9</td><td>21.6</td><td>11.8</td><td>4.4</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>28.5</td><td>45.9</td><td>30.0</td><td>11.8</td><td>12.1</td><td>18.2</td><td>10.7</td><td>3.9</td><td>6.8</td><td>7.8</td><td>2.0</td><td>0.7</td><td>18.2</td><td>28.3</td><td>17.5</td><td>6.7</td></tr><tr><td>Qwen3.8-Omni-Flash</td><td>26.7</td><td>44.5</td><td>27.4</td><td>7.1</td><td>18.7</td><td>30.7</td><td>14.3</td><td>5.0</td><td>13.8</td><td>17.0</td><td>5.9</td><td>1.3</td><td>21.4</td><td>34.5</td><td>18.6</td><td>5.4 11.1</td></tr><tr><td>一 Gemini2.5-Flash Gemini2.5-Flash-Lite</td><td>33.4</td><td>51.6</td><td>38.9</td><td>17.3</td><td>16.5</td><td>24.3</td><td>15.9</td><td>8.9</td><td>7.5</td><td>7.2</td><td>1.3</td><td>0.0</td><td>22.3</td><td>33.3</td><td>23.5</td><td></td></tr><tr><td>Gemini2.5-Pro</td><td>27.4</td><td>41.7 70.3</td><td>30.9</td><td>13.4</td><td>10.8</td><td>15.5</td><td>8.6</td><td>5.7</td><td>4.3</td><td>2.6</td><td>0.7</td><td>0.0</td><td>16.9</td><td>24.6</td><td>16.8</td><td>8.1</td></tr><tr><td>Gemini3.1-Flash-Lite</td><td>49.3</td><td></td><td>60.6</td><td>34.3</td><td>30.6</td><td>45.9</td><td>32.3</td><td>14.5</td><td>17.8</td><td>22.9</td><td>7.2</td><td>1.3</td><td>36.6</td><td>52.8</td><td>40.5</td><td>20.9</td></tr><tr><td></td><td>31.1</td><td>47.9</td><td>36.2</td><td>15.7</td><td>15.0</td><td>23.6</td><td>13.4</td><td>6.1</td><td>5.3</td><td>3.9</td><td>0.7</td><td>0.7</td><td>20.4</td><td>31.0</td><td>21.1</td><td>9.3</td></tr><tr><td>Gemini3.1-Flash-Lite-Preview Gemini3.1-Pro-Preview</td><td>32.1</td><td>50.5 68.9</td><td>37.6</td><td>16.8</td><td>15.2</td><td>23.4</td><td>12.7</td><td>6.1</td><td>6.5</td><td>6.5</td><td>0.0</td><td>0.0</td><td>21.1</td><td>32.3</td><td>21.3</td><td>9.7</td></tr><tr><td>Gemini3.5-Flash</td><td>48.6 36.8</td><td>56.0</td><td>60.8 44.2</td><td>31.6 21.7</td><td>33.0</td><td>48.0</td><td>32.7</td><td>16.6</td><td>20.1</td><td>28.8</td><td>5.2</td><td>2.6</td><td>37.7</td><td>53.9</td><td>40.5</td><td>20.8</td></tr><tr><td></td><td>33.5</td><td>51.6</td><td>38.7</td><td>18.4</td><td>35.5</td><td>52.0</td><td>36.8</td><td>20.0</td><td>20.6</td><td>27.5</td><td>9.8</td><td>1.3</td><td>33.8</td><td>50.0</td><td>35.9</td><td>17.9 11.3</td></tr><tr><td>Gemini3.5-Flash-Lite</td><td>52.5</td><td>75.1</td><td>66.1</td><td>36.9</td><td>15.1</td><td>23.4</td><td>13.9</td><td>8.0</td><td>8.3 20.5</td><td>9.8</td><td>0.7</td><td>0.7</td><td>21.9</td><td>33.3</td><td>22.4</td><td>24.1</td></tr><tr><td>一 Gemini3.6-Flash Gemini3.7-Flash</td><td>57.0</td><td>81.8</td><td>72.4</td><td>40.6</td><td>35.3 37.6</td><td>53.0 55.5</td><td>36.1 40.7</td><td>19.3 21.4</td><td>23.5</td><td>31.4 35.9</td><td>8.5 12.4</td><td>2.0 3.3</td><td>40.4 43.7</td><td>59.1 63.7</td><td>44.7 49.9</td><td>26.8</td></tr><tr><td></td><td>56.4</td><td>79.3</td><td>69.4</td><td>43.5</td><td>36.6</td><td>53.0</td><td>38.9</td><td>21.1</td><td>18.3</td><td>26.1</td><td>5.9</td><td>2.0</td><td>42.3</td><td>60.1</td><td>46.8</td><td>27.8</td></tr><tr><td>Gemini3.8-Flash Gemini4-Argon</td><td>35.6</td><td>51.6</td><td>43.3</td><td>21.4</td><td>17.5</td><td>28.6</td><td>17.0</td><td>6.4</td><td>14.5</td></table>

Scoring protocol. Ground-truth evidence is read from the annotation file for every model. For each modality, we compute temporal IoU between the unions of the predicted and annotated intervals, without multiplying by answer correctness. Failed inference, response parsing, or truncated output receives zero answer and localization scores and remains in the denominator. Comma-based decimal timestamps are normalized numerically. These choices preserve a common evaluation set without granting localization credit to malformed predictions.

Let C denote answer correctness and $G _ { \tau } = \mathcal { H } [ \operatorname* { m i n } ( \mathrm { I o U } _ { \mathrm { a u d i o } } , \mathrm { I o U } _ { \mathrm { v i s u a l } } ) \ge \tau ]$ denote successful dual-modal localization. Unlike the mean-IoU threshold metric above, this criterion requires each modality to meet the threshold. Table 5 reports the four outcomes at $\tau = 0 . 5 ;$ each outcome is expressed as a percentage of all questions.

Modality-specific localization. Six of the seven models in Table 5 obtain higher mean audio than visual IoU overall. Among these six, the diference ranges from 0.7 percentage points for Qwen3.5-Omni-Plus to 3.7 points for Gemini3.5-Flash. Qwen3.8-Omni-Flash shows the opposite pattern, with 27.4% audio IoU and 34.3% visual IoU.

Answer–grounding disagreement. Across the seven models, 47.8%–79.8% of correct answers fail the dual-modal localization criterion at $\tau = 0 . 5$ . For Gemini3.7-Flash, 47.8% of correct answers fail grounding, whereas 12.2% of grounding-passing predictions answer incorrectly. The converse error also occurs in the other models: for MiMo-V2.5, 27.9% of grounding-passing predictions answer incorrectly. These outcomes distinguish temporal localization from answer selection; temporal overlap alone establishes neither semantic understanding nor causal use of the reported evidence.

Threshold sensitivity. For Gemini3.7-Flash, the fraction of correct answers for which either audio or visual IoU falls below the grounding threshold is 32.9%, 47.8%, and 78.2% at thresholds of 0.3, 0.5, and 0.7, respectively. These proportions depend on the threshold and are not direct evidence that correct answers were guessed.

![](images/92fa88f20f2d844088ec5c1810668525f7c2919949447483851116243035a96f.jpg)  
Figure 31 Performance across fine-grained capabilities. Lines show answer accuracy (Acc.); solid and hatched bars show Coverage Grounded Score (C.G.) and Bottleneck Grounded Score (B.G.), respectively. All values are percentages. Task abbreviations follow the capability taxonomy in Appendix A.5: Level 1: TA = Temporal Alignment, EvB = Event Binding, EnB = Entity Binding; Level 2: CR = Causal Reasoning, EI = Emotional Intelligence, IU = Intent Understanding, AU = Atmosphere Understanding, IEV = Implicit Event Reasoning, TU = Temporal Understanding, CC = Conditional Counting, SD = Semantic Disambiguation, SU = Spatial Understanding; Level 3: LRR = Long-range Reasoning, MEC = Multi-Evidence Chain, MHR = Multi-hop Reasoning.

## D.4 Analysis Across Fine-Grained Capabilities

Figure 31 compares the two representative models across 15 fine-grained capabilities. On the three Level 1 alignment tasks, Gemini3.7-Flash achieves 82.3%–88.8% accuracy and 47.5%–49.9% B.G., compared with 60.4%–68.8% accuracy and 28.1%–34.0% B.G. for MiMo-V2.5. Both models show a larger separation between answering and precise grounding on complex tasks. On multi-hop reasoning, Gemini3.7-Flash reaches 80.7% accuracy but only 10.6% B.G., while MiMo-V2.5 reaches 44.6% accuracy and 4.9% B.G.; on multi-evidence chains, the corresponding scores are 82.9%/15.6% and 51.4%/8.2%. These comparisons suggest that selecting an answer is substantially easier for both models than locating the dispersed evidence needed to support it.

## E Limitations and Future Work

## E.1 Limitations

Dataset coverage. EvaOmni contains 1,027 question-answer pairs drawn from 392 videos lasting 1–12 minutes with suficient audio-visual cues. Excluding hard-coded subtitles and prioritizing varied entities and events support the construction of questions requiring both modalities, but also constrain the data distribution. The results therefore do not establish generalization to substantially longer videos, subtitle-rich content, scenes with sparse events, or all linguistic and cultural contexts. The three capability levels characterize diferent task requirements; they do not guarantee a strictly increasing dificulty ordering in answer accuracy across diferent question sets.

## E.2 Future Work

Broader data and task coverage. Future extensions could include longer videos, a wider range of linguistic and cultural contexts, and scenes with sparser audio-visual cues while retaining checks for the necessity of both modalities. Subtitle-rich content would also enable explicit separation of text reading from audio understanding. Broader evaluation of open-ended question answering could test whether the current findings hold without cues from answer options.