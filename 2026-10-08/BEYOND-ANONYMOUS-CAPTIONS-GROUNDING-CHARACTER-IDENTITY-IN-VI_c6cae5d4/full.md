# BEYOND ANONYMOUS CAPTIONS: GROUNDING CHARACTER IDENTITY IN VIDEO CAPTIONING AND QUESTION ANSWERING

Anas Filali Razzouki<sup>1,2</sup> Killian Steunou<sup>1,2</sup> Khalil Guetari<sup>2</sup> Thomas Kling<sup>2</sup> Mounˆım El-Yacoubi<sup>1</sup> Yannis Tevissen<sup>1,2</sup>

<sup>1</sup>Tel´ ecom SudParis, Institut Polytechnique de Paris, France´

<sup>2</sup>Moments Lab Research, France

## ABSTRACT

Linking people’s appearance and actions to character identities is essential for understanding video narratives. We present a framework for identity-aware video captioning and person-centric question answering that combines automatic character identification, explicit spatial grounding, and task-specific adaptation. Starting from LSMDC v2 movie clips, our pipeline matches detected faces to actor reference images, tracks characters across frames, and builds inputs with identitylinked bounding boxes. A strong vision-language model generates identity-aware captions and questions, which are manually verified and filtered to create a benchmark of 750 captioned clips and 3,000 person-centric questions. We study five grounding strategies combining textual coordinates with visual face or estimated person boxes across Video-MLLM families at roughly 2B, 4B, and 8B parameters and larger frontier models. Combining visual face boxes with textual coordinates yields the most consistent performance across scales and significantly improves overall performance over coordinates alone. Smaller models tend to over-assign known identities when the queried person is not grounded, while larger models better recognize such UNIDENTIFIED cases. We introduce BAC by LoRA finetuning Qwen models at 2B, 4B, and 8B scales on about 32K identity-aware captioned clips. Across all scales, BAC outperforms every other evaluated model family of comparable size. BAC-8B reaches 93.20% overall QA accuracy, ranking behind only GPT-5.6 Sol among the frontier models evaluated in our study. Overall, explicitly communicating who is where, together with lightweight taskspecific adaptation, substantially improves identity-aware video understanding without changing the underlying architecture. We release the benchmark, training data, code, and BAC checkpoints at https://github.com/momentslab/ beyond-anonymous-captions.

## 1 INTRODUCTION

Understanding human-centered video requires more than recognizing what happens in a scene. Video-captioning models can describe events, appearances, actions, and interactions in natural language (Xu et al., 2016; Yang et al., 2023). However, these descriptions are typically identityagnostic: a model may correctly understand what each visible person is doing without determining which character that person corresponds to. In narrative videos such as movies, this distinction is important because scene understanding depends not only on recognizing actions and interactions, but also on attributing them to the correct characters.

At the same time, visual grounding has made substantial progress in associating textual references with corresponding visual regions (Kamath et al., 2021; Ma et al., 2024). Given a textual reference to a person or object, grounding models can localize the referred entity, typically through spatial regions such as bounding boxes. Video captioning therefore provides a mechanism for describing what is happening in a scene, while visual grounding provides complementary information about where a referred entity is located.

This observation motivates the central question of our work: can these two capabilities be combined to support identity-aware video understanding? More specifically, if a Video-MLLM is provided not only with the visual content of a video, but also with explicit information indicating the identity and spatial location of the people appearing in it, can it correctly associate each identity with the corresponding person in the scene? Such an association could allow the model to preserve its existing understanding of appearances, actions, spatial relations, and interactions while grounding that understanding in the correct character identities. More generally, we investigate whether explicit information about who is where can help a Video-MLLM determine more reliably who is doing what, where, and with whom.

Prior work on identity-aware visual description has approached the problem in several ways. Early movie-naming methods such as M-VAD Names (Pini et al., 2019) associate face tracks with charac ter identities and replace generic SOMEONE mentions in existing captions with the corresponding names, rather than generating identity-aware descriptions directly. A related post-processing strat egy is proposed by Tevissen et al. (Tevissen et al., 2024), who first generate a generic image caption and then use attention maps and identified face regions to replace person mentions with names. In both cases, identity is introduced after the caption content has largely been determined, so people omitted from the original caption cannot naturally be recovered. M-VAD Names indeed formulates its main task as replacing SOMEONE tags in existing captions with proper character names.

More recent work integrates identity more directly into multimodal understanding. MICap (Raajesh et al., 2024) maintains anonymous person identities across a videoset of five consecutive clips by clustering faces and jointly learning identity assignment and caption generation. IDA-VLM (Ji et al., 2025), instead, conditions a vision-language model on reference images of known identities and requires the model to recognize those identities in new scenes before performing tasks such as localization, question answering, or captioning. ISYV (Gao et al., 2026) extends this referenceconditioned setting to video, requiring the model to recognize and reason about a target person across time and shot transitions. Thus, these methods either learn identity consistency jointly with generation or require the multimodal model itself to infer the correspondence between a reference identity and its occurrence in the visual input.

A complementary line of work studies region-level multimodal grounding. Omni-RGPT (Heo et al., 2025) associates user-specified boxes or masks with language references to support region-specific reasoning over images and videos, while VideoGLaMM (Munasinghe et al., 2025) grounds language in video at the pixel level through spatio-temporally consistent segmentation masks. These approaches demonstrate that explicit links between language and visual regions can support finegrained reasoning, but they are not designed specifically to determine how known character identities should be represented to a general-purpose Video-MLLM.

Despite these advances, existing approaches do not address our central question: how should already known character identities be spatially represented to a general-purpose Video-MLLM so that it can reason about them throughout a video? We introduce this formulation by providing explicit identity-linked spatial cues whenever reliable detections and tracks are available, rather than requiring the model to recover identity solely from reference images or appearance. Importantly, these cues may be absent in some frames, requiring the Video-MLLM to propagate the available identity information across time and associate each character with their appearances, actions, locations, and interactions. We therefore investigate how different spatial representations of known identities enable identity-aware video captioning and person-centric question answering, without designing a dedicated region-aware architecture.

To study this question, we develop a pipeline on movie shots from LSMDC v2 in which detected faces are matched to actor reference images and identified characters are spatially grounded across sampled frames. We construct a manually verified benchmark of 750 captioned shots and 3,000 person-centric questions, and compare five identity-grounding representations ranging from textual coordinates to visual face or person boxes and their combinations. We evaluate them on identityaware video captioning and person-centric question answering, then use the resulting grounding formulation to introduce BAC at three model scales (2B, 4B, and 8B). We compare BAC with Video-MLLMs of similar sizes from multiple model families and extend the evaluation to frontier open-weight and proprietary video-language models. We release approximately 32K synthetically captioned training clips and a manually verified benchmark of 750 clips and 3,000 questions, all with face bounding-box annotations, plus source code and trained checkpoints.

## 2 METHODS

## 2.1 DATASET

## 2.1.1 CHARACTER IDENTIFICATION

We build our dataset from the LSMDC v2 collection (Rohrbach et al., 2017; Torabi et al., 2016; Maharaj et al., 2017), downloading 92 movies comprising 46,756 video clips. Following castsupervised character-identification approaches for movies and TV series (Xu et al., 2010; Nagrani & Zisserman, 2018; Bamman et al., 2024), we construct character-linked face tracks by combining cast information, actor reference images, face recognition, and temporal tracking.

For each movie, we retrieve actor and character names from IMDb and collect up to 50 reference images per actor from DuckDuckGo. We filter low-quality or ambiguous images using Insight-Face (Deng et al., 2019) based on image resolution, face size, detection confidence, and face dominance. The remaining face embeddings are clustered to remove identity outliers, and the 5 most representative images are retained for each actor, forming a reliable reference gallery.

Each clip is then processed frame-by-frame with InsightFace to detect faces and extract 512- dimensional embeddings. A DeepSORT-inspired online tracker (Wojke et al., 2017) links detections across consecutive frames into temporally consistent tracks. Each track is represented by a robust aggregate embedding and matched against the movie-specific reference gallery. The predicted identity is accepted when the matching confidence exceeds 0.30 and is propagated across the track together with its face bounding boxes.

Finally, because the original LSMDC clips may contain multiple camera shots with abrupt changes in scene or viewpoint, we split them into visually coherent shots using PySceneDetect. This produces 54,133 shots, with an average duration of 2.50 s, and an average of 1.7 shots per clip.

## 2.1.2 CAPTION AND QUESTION GENERATION

Given the short shot duration, averaging approximately 2.5 s at 24 fps, we uniformly sample 8 frames per shot. As illustrated in Figure 1, these frames provide sufficient temporal coverage of the main actions. From the 54,133 detected shots, we retain only those containing at least one identity-linked bounding box in at least one sampled frame. Preserving the original LSMDC v2 movie-level split, this yields 31,935 shots from 72 training movies and 6,219 shots from 12 test movies. The training set is further divided into 28,742 training and 3,193 validation shots, with the validation split used for BAC hyperparameter tuning.

For both training and test shots, the sampled frames are provided to Gemini 2.5 Flash to generate identity-aware captions. For test shots, the model is additionally instructed to generate four personcentric questions and to assess shot validity, visual and temporal coherence, and identity-grounding difficulty. We retain shots that are valid, high-quality, and challenging, with coherence scores of at least 9 and challenge scores of at least 8.5. This favors non-trivial cases involving multiple people, distractors, occlusions, interactions, or viewpoint changes, yielding 2,008 shots.

Finally, from the 2,008 retained shots, we manually review and correct the captions, questions, and answers. We remove redundant or ambiguous examples and refine overly simple questions to require more precise identity grounding. This curation produces a final evaluation benchmark of 750 captioned shots and 3,000 person-centric questions.

## 2.2 IDENTITY-GROUNDING REPRESENTATIONS

Once character identities and their locations have been established, we investigate how these identity–location associations should be represented to a general-purpose Video-MLLM. We consider five grounding strategies that differ in whether spatial information is conveyed through textual coordinates, visual annotations, or a combination of both. Each identified character is assigned an identifier $P _ { i }$ , which is consistently mapped to the same identity throughout the shot.

Face Textual Coordinates (FTC). The sampled frames are provided without visual bounding-box annotations. The prompt specifies the identity associated with each $P _ { i }$ together with the textual coordinates of its face bounding box for every frame in which the character is localized. This

![](images/934f35544dd06784b234058731cda97112c6e7479506f497f8172dab27ef319f.jpg)

![](images/69dc25ef4658025d55f8ade02374023bcfd110f89887e34496d487a6eebf7b8d.jpg)  
Identity mapping: P1 ↔ Cam Gigandet; P2 ↔ Minka Kelly; P3 ↔ Leighton Meester.

![](images/cad501fb452f9cba22fc8a6ec65e4384d3bc5935baf26037e719fc5b65baf4fa.jpg)

![](images/17ac11d557fd889bbafbc34aed66ad8c38c86a7fb2a4f272e0e025ca89850d5b.jpg)  
Caption: Cam Gigandet, in a green shirt, and Minka Kelly, in a red top, stand close together and share a kiss while Leighton Meester, in the car, watches.

Person-centric QA: Who is in the car? (Position) → Leighton Meester; Who is kissing Minka Kelly? (Interaction) → Cam Gigandet; Who is wearing a green shirt? (Appearance) → Cam Gigandet; Who is wearing a red top? (Appearance) → Minka Kelly.

Figure 1: Example of identity-aware captioning and person-centric QA using character identities grounded across sampled video frames.

representation evaluates whether the Video-MLLM can establish the identity–location association using textual spatial information alone.

For strategies involving visual grounding, a bounding box is drawn around each localized character, with the corresponding P<sub>i</sub> displayed at its upper-left corner. For a given identity, the box and label use the same color across frames, providing a consistent identity-linked visual cue. The $P _ { i }$ label size is scaled with the bounding box to remain visible while limiting occlusion. Figure 1 illustrates this visual identity-grounding process.

Face Visual Boxes (FVB). Each identified character is visually grounded using its face bounding box. No bounding-box coordinates are provided in the textual prompt, such that the identity–location association is conveyed exclusively through the visual face annotations.

Person Visual Boxes (PVB). Each identified character is visually grounded using an estimated person-level bounding box. Rather than requiring an additional person detector, we extrapolate the person box from the detected face using a lightweight anthropometric heuristic based on fixed faceto-body proportions (Jaruenpunyasak et al., 2022). Details of the estimation procedure are provided in Appendix A. In this method, no spatial coordinates are included in the prompt; identity–location associations are conveyed solely through person-level visual annotations.

Face Visual Boxes + Face Textual Coordinates (FVB+FTC). Each identified character is visually grounded using its face bounding box. In addition to the visual annotation, the prompt provides the textual coordinates of the corresponding face bounding box for each frame.

Person Visual Boxes + Person Textual Coordinates (PVB+PTC). Each identified character is visually grounded using an estimated person-level bounding box. In addition to the visual annotation, the prompt provides the textual coordinates of the corresponding estimated person bounding box for each frame.

Together, these five representations allow us to compare textual grounding, visual grounding, and their combination, while also examining whether face-level or person-level spatial cues are more effective for identity-aware video understanding.

## 2.3 EXPERIMENTAL PROTOCOL

We evaluate the proposed identity-grounding representations on two complementary tasks: identityaware video captioning and person-centric question answering. All experiments are conducted on the manually verified evaluation set of 750 video clips, comprising 3,000 identity-related questions.

We conduct three complementary sets of experiments. First, we study the effect of the grounding representation while keeping the Video-MLLM family fixed. We compare the five representations introduced above: FTC, FVB, PVB, FVB+FCT, and PVB+PTC. This comparison is performed with Qwen3 models at approximately 2B, 4B, and 8B parameters, allowing us to examine how the effectiveness of each grounding representation evolves with model scale.

Second, we investigate whether the benefits of identity grounding generalize across different Video-MLLM families. Models are grouped into three approximate parameter scales: 2B, 4B, and 8B–9B. At the 2B scale, we evaluate Qwen3, InternVL3.5; at the 4B scale, we evaluate Qwen3, InternVL3.5, MiniCPM-V, and Ovis2.5; and at the 8B–9B scale, we evaluate Qwen3, Ovis2, InternVL3.5, Eagle2.5, and Molmo2. The same clips, questions, prompts, identity annotations, and grounding representations are used across models within each comparison.

Third, we introduce BAC, obtained by fine-tuning the Qwen model family at three scales (2B, 4B, and 8B) using Low-Rank Adaptation (LoRA). The models are trained on our training split using ≈ 32K identity-aware captions generated by Gemini as supervision, as described in Section 2.1.2, and are evaluated on the manually verified evaluation set. This stage assesses whether task-specific, parameter-efficient adaptation can further improve identity-aware video understanding when combined with explicit identity grounding.

For captioning, each model receives the grounded video shot and is instructed to generate a description referring to identified characters using their corresponding $P _ { i }$ identifiers. For question answering, the model receives the same grounded shot together with one person-centric question at a time and returns the corresponding $P _ { i } ,$ or UNIDENTIFIED (UNID) when the queried person cannot be associated with any provided identity. The predicted $P _ { i }$ identifiers are subsequently mapped to their corresponding character names for evaluation and qualitative presentation.

Except for the LoRA adaptation experiments, all models are evaluated without task-specific finetuning. We use a consistent prompting protocol within each task and keep the generation configuration fixed for each model across the compared grounding representations.

## 2.4 EVALUATION METRICS

Person-centric question answering: QA predictions are compared directly with the ground-truth identity. We report QA Overall Accuracy over all 3,000 questions, of which 2,074 target a known character and 926 correspond to UNIDENTIFIED cases. QA Named Accuracy is computed over the 2,074 questions targeting a known character, while QA UNIDENTIFIED Accuracy is computed over the 926 questions referring to a person who cannot be associated with any of the provided character identities. We additionally report QA Clip-Level Accuracy, defined as the percentage of clips for which all four associated questions are answered correctly.

Identity-aware captioning: Evaluating identity accuracy against a single ground-truth caption is unreliable because two correct captions may describe different visible attributes of the same character, such as a shirt, skirt, or hat. Manual evaluation is also impractical at scale. We therefore use a VLM-as-a-judge protocol based on Qwen/Qwen3.8-27B-FP8, an open-weight 27B vision-language model achieving 94% accuracy on our person-centric QA task. The judge receives the video shot, identity grounding, and generated caption, and evaluates identity correctness, hallucination, visual faithfulness, and naturalness. We report Caption Overall, the mean quality score on a 0–10 scale; Caption Identity, the percentage of shots where every queried character is mentioned and correctly identified; and Caption Hallucination, the percentage containing an unsupported person, identity, or absent object.

## 3 RESULTS

## 3.1 EFFECT OF IDENTITY GROUNDING STRATEGIES

Table 1 shows that explicit visual grounding substantially improves person-centric QA over FTC alone. QA Overall increases from 53.87% to 68.90% at 2B, from 62.57% to 79.73% at 4B, and from 59.77% to 88.53% at 8B. Paired McNemar tests confirm that FTC is significantly worse than every visually grounded variant at all model sizes $( p < 0 . 0 0 1 )$ ). Excluding FTC, differences among the grounded variants become smaller with scale. At 2B, FVB+FTC achieves the highest accuracy and significantly outperforms PVB+PTC and PVB, while remaining statistically indistinguishable from FVB. At 4B and 8B, FVB+FTC, FVB, and PVB+PTC consistently form the top statistical group, while PVB remains significantly weaker.

For captioning, the effect of the grounding representation is most pronounced at small model scale. At 2B, FVB+FTC clearly outperforms the other strategies in identity accuracy, with all pairwise differences being statistically significant $( p < 0 . 0 0 1 )$ . As model size increases, these differences shrink substantially: at 4B, the visually grounded variants are mostly statistically indistinguishable, with only FVB+FTC outperforming FVB, while at 8B all four visually grounded strategies are statistically tied. In contrast, FTC remains significantly worse than every visually grounded variant at 4B and 8B $( p < 0 . 0 0 1 )$ .

Overall, FVB+FTC provides the most consistent captioning and QA performance across scales, particularly at 2B. We therefore analyze whether its remaining errors depend on grounded face size. For QA, face-box area correlates positively with correctness, increasing from $r _ { s } = 0 . 1 0 3$ at 2B to $r _ { s } = 0 . 2 4 5$ at 8B, while captioning shows a stronger and stable association $( r _ { s } = 0 . 2 3 8 – 0 . 2 5 6 )$ PVB+PTC is less sensitive to box size, likely because the full-person box covers a much larger visual region than the face alone.

Results grouped by size confirm this pattern. At 2B (Table 2), PVB+PTC performs better for small faces $( < 4 , \bar { 0 0 0 } \mathrm { p } \mathrm { \bar { x } } ^ { 2 } )$ , reaching 91.8% versus 80.0% for FVB+FTC $( p = 0 . 0 2 1 )$ . For medium and large faces, FVB+FTC performs better, reaching 92.8% vs. 91.3% and 96.8% vs. 91.1%, respectively. These results suggest that full-person grounding can compensate when the face is very small, whereas face-based grounding becomes more effective once the face is sufficiently visible; expanding the grounding to the full person may then introduce less precise spatial information. Qualitative examples of both cases are provided in Appendix B.

The Named and UNIDENTIFIED results further reveal how grounding affects identity assignment. At 2B and 4B, the combined visual–textual strategies favor assigning known identities: FVB+FTC and PVB+PTC achieve the highest Named accuracy, while remaining substantially weaker on UNIDENTIFIED cases. Visual-only grounding generally reduces this imbalance, improving the rejection of unidentified characters at the cost of lower Named accuracy. This asymmetry decreases markedly with model scale. At 8B, FVB+FTC achieves nearly balanced performance between Named and UNIDENTIFIED identities (88.04% and 86.93%), indicating that larger models are better able to use the grounding signal without systematically forcing a known identity when the evidence is insufficient.

Table 1: Comparison of identity-grounding strategies across Qwen3-VL model sizes. Captioning is evaluated using overall quality, identity accuracy, and hallucination rate. QA performance is reported over all questions, separately for Named and UNIDENTIFIED (UNID) identities.
<table><tr><td colspan="2"></td><td colspan="3">Captioning</td><td colspan="3">Person-Centric QA</td></tr><tr><td>Size</td><td>Grounding</td><td>Overall ↑</td><td>Identity ↑</td><td>Halluc. ↓</td><td>Overall ↑</td><td>Named ↑</td><td>UNID ↑</td></tr><tr><td rowspan="5">2B</td><td>FTC</td><td>5.74</td><td>47.47</td><td>7.60</td><td>53.87</td><td>75.60</td><td>5.18</td></tr><tr><td>FVB</td><td>5.92</td><td>36.80</td><td>10.27</td><td>68.27</td><td>86.55</td><td>27.32</td></tr><tr><td>PVB</td><td>5.52</td><td>27.54</td><td>8.28</td><td>66.13</td><td>80.71</td><td>33.48</td></tr><tr><td>FVB+FTC</td><td>7.61</td><td>76.53</td><td>3.07</td><td>68.90</td><td>92.48</td><td>16.09</td></tr><tr><td>PVB+PTC</td><td>7.29</td><td>69.87</td><td>2.67</td><td>67.13</td><td>91.27</td><td>13.07</td></tr><tr><td rowspan="5">4B</td><td>FTC</td><td>6.32</td><td>52.80</td><td>4.53</td><td>62.57</td><td>71.75</td><td>42.01</td></tr><tr><td>FVB</td><td>8.03</td><td>85.20</td><td>9.73</td><td>79.13</td><td>92.29</td><td>49.68</td></tr><tr><td>PVB</td><td>8.19</td><td>87.33</td><td>4.40</td><td>77.83</td><td>91.22</td><td>47.84</td></tr><tr><td>FVB+FTC</td><td>8.32</td><td>88.80</td><td>1.07</td><td>79.73</td><td>93.11</td><td>49.78</td></tr><tr><td>PVB+PTC</td><td>8.22</td><td>87.07</td><td>1.60</td><td>79.20</td><td>94.21</td><td>45.57</td></tr><tr><td rowspan="5">8B</td><td>FTC</td><td>6.32</td><td>54.27</td><td>3.60</td><td>59.77</td><td>51.69</td><td>77.86</td></tr><tr><td>FVB</td><td>8.45</td><td>92.00</td><td>3.87</td><td>88.53</td><td>91.85</td><td>81.10</td></tr><tr><td>PVB</td><td>8.44</td><td>91.73</td><td>1.87</td><td>85.37</td><td>90.60</td><td>73.65</td></tr><tr><td>FVB+FTC</td><td>8.45</td><td>91.60</td><td>0.93</td><td>87.70</td><td>88.04</td><td>86.93</td></tr><tr><td>PVB+PTC</td><td>8.37</td><td>90.67</td><td>1.60</td><td>87.43</td><td>92.24</td><td>76.67</td></tr></table>

![](images/e2baaa78c283cfb998a0bf02fbd7dcccdc963a9a1f629bf83499231ad66ff313.jpg)

![](images/83407748bd6af38ead00d66ed2260eec532b5e9b81cad462348c46b621997807.jpg)  
(a) Identity mapping: P1 ↔ Leighton Meester.  
Q4 [Action]: Who walks away toward the door? GT: UNIDENTIFIED.

![](images/c90289545a8b4421ae7e389e616931f861f480503b591858fd263b42ef50dc22.jpg)

![](images/82f80b3d1fd6cc68b06608057dcf193328a355a6191ae4c77f93d525df6855c1.jpg)

<table><tr><td>Model</td><td>Pred.</td><td>Model</td><td></td><td>Pred.</td><td>Model</td><td></td><td>Pred.</td></tr><tr><td>BAC-2B (ours) Qwen3-VL 2B</td><td>UNID. Leighton</td><td>7</td><td>BAC-4B (ours) Qwen3-VL 4B</td><td>UNID. Leighton</td><td>BAC-8B (ours) L Qwen3-VL 8B</td><td>UNID. Leighton</td><td>V X</td></tr><tr><td>GPT-5.6 Sol</td><td></td><td>X</td><td></td><td></td><td>X</td><td></td><td></td></tr><tr><td></td><td>UNID.</td><td></td><td>GPT-5.6 Sol*</td><td>UNID.</td><td>V</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Claude Sonnet 4.6</td><td>UNID.</td><td></td></tr><tr><td>Qwen3-VL 235B</td><td>UNID.</td><td></td><td>Gemini 2.5 Flash</td><td>UNID.</td><td></td><td></td><td></td></tr></table>

![](images/91730fb782ee19505cdbe808ca39f18267e063c35eff31f2bb15ba74fece036d.jpg)

![](images/f6245de503b62ca84ab042a16a94608d6af462ab426b0b1915e5f9ad5d4c2952.jpg)

![](images/1390ea240256649aa2cd2739adf7af079cc1dbcd8b93f7b458c53d17b3fdc396.jpg)

![](images/061efae243504f5152e9f30cd721e08cd7cd8362f9063f7486f533d46170e7d6.jpg)

(b) Identity mapping: P1 ↔ Kat Graham; P2 ↔ Minka Kelly; P3 ↔ Aly Michalka. Q3 [Action]: Who stands upfrom the brown sofa? GT: Aly Michalka (P3).
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td></tr><tr><td>BAC-2B (ours) Qwen3-VL 2B</td><td>Aly Kat</td><td>√ X</td><td>BAC-4B (ours) Qwen3-VL 4B</td><td>Aly Kat</td><td>√ X</td><td>BAC-8B (ours) Qwen3-VL 8B</td><td>Aly Minka</td><td>√ X</td></tr><tr><td>GPT-5.6 Sol</td><td></td><td>√</td><td>GPT-5.6 Sol*</td><td></td><td>√</td><td>Claude Sonnet 4.6</td><td></td><td></td></tr><tr><td></td><td>Aly</td><td></td><td></td><td>Aly</td><td></td><td></td><td>UNID.</td><td>×</td></tr><tr><td>Qwen3-VL 235B</td><td>UNID.</td><td>X</td><td>Gemini 2.5 Flash</td><td>UNID.</td><td>X</td><td></td><td></td><td></td></tr></table>

![](images/f9731f29cdc26f57a6d784621bdac76768c36eda343f04933a200d4085b8d6df.jpg)

![](images/88ba2e83814572e25f8b219ff3416ff30397deaa15d3df15bac2160097eb8e53.jpg)

![](images/8bc7af80e0329c4a2398a74304a498b23855107e90d8c9e7dcfbc3a86d769de5.jpg)

![](images/3d042ee99fbc3e67a92a27ee916204341b3daa26c2eba3ab14aa2008c0e5bcbc.jpg)

(c) Identity mapping: P1 ↔ J. K. Simmons; P2 ↔ Elliot Page; P3 ↔ Olivia Thirlby; P4 ↔ Allison Janney. Q4 [Position]: Whofollows behind Olivia Thirlby and Elliot Page? GT: Allison Janney (P4).
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td></tr><tr><td>BAC-2B (ours) Qwen3-VL 2B</td><td>J. K. Elliot</td><td>X X</td><td>BAC-4B (ours) Qwen3-VL 4B</td><td>J. K. J. K.</td><td>X X</td><td>BAC-8B (ours) Qwen3-VL 8B</td><td>J. K. X UNID.</td></tr><tr><td>GPT-5.6 Sol</td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td></tr><tr><td></td><td>Allison</td><td>√</td><td>GPT-5.6 Sol*</td><td>Allison</td><td>√</td><td>Claude Sonnet 4.6</td><td>J. K. X</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL 235B</td><td>UNID.</td><td>X</td><td>Gemini 2.5 Flash</td><td>UNID.</td><td>×</td><td></td><td></td></tr></table>

Figure 2: Qualitative identity-aware QA examples. Identity-mapping colors match the boundingbox colors in each frame. (a) BAC correctly predicts UNIDENTIFIED; (b) BAC correctly identifies the queried named person; (c) BAC fails to identify the queried named person, while only GPT-5.6 Sol succeeds. <sup>∗</sup>: GPT-5.6 Sol with thinking. 7

Table 2: QA accuracy of FVB+FTC and PVB+PTC grounding methods with Qwen3-2B, stratified by face bounding-box area. $\Delta$ denotes the accuracy difference between PVB+PTC and FVB+FTC.
<table><tr><td>Face-area bin</td><td>N</td><td>FVB+FTC</td><td>PVB+PTC</td><td>∆ (P-F)</td><td>p</td></tr><tr><td>Small  $< 4 , 0 0 0 \mathrm { p x } ^ { 2 }$ </td><td>85</td><td>80.0%</td><td>91.8%</td><td>+11.8</td><td>0.021</td></tr><tr><td>Medium  $\mathrm { 4 , 0 0 0 { - } 2 6 9 , 9 9 9 \ p x ^ { 2 } }$ </td><td>1,865</td><td>92.8%</td><td>91.3%</td><td>-1.5</td><td>0.009</td></tr><tr><td> $\mathbf { L a r g e } \geq 2 7 0 , 0 0 0 \mathbf { p x } ^ { 2 }$ </td><td>124</td><td>96.8%</td><td>91.1%</td><td>-5.6</td><td>0.016</td></tr></table>

## 3.2 BENCHMARKING ACROSS MODEL FAMILIES

Table 3 compares BAC with several VLM families at comparable parameter scales using the FVB+FTC grounding method. We first focus on the base models to identify the strongest backbone before examining the effect of BAC fine-tuning. Among the base models, Qwen3 provides the strongest overall balance, achieving the highest QA accuracy within each size group together with consistently strong captioning performance. Its QA advantage over the strongest competing base model is statistically significant at 2B, 4B, and 8–9B $( p < 0 . 0 0 1$ , paired McNemar tests). At 8–9B, Ovis2.5 attains higher caption identity accuracy than Qwen3 (93.73% vs. 91.60%), but the difference is not significant $( p = 0 . 0 8 9 )$ .

We then examine the effect of BAC fine-tuning on the Qwen3 backbone. The best configuration identified by our fine-tuning sweep uses a learning rate of $5 \times 1 0 ^ { - 5 }$ , LoRA rank $r = 1 6 .$ , scaling factor $\alpha = 3 2$ , and dropout 0.05. This configuration shows stable training and validation convergence across all three model scales, with full optimization details and learning curves provided in Appendix C.

BAC substantially improves the corresponding Qwen3 baseline at every scale. Although BAC is trained only on the captioning task, it generalizes strongly to person-centric QA without QA-specific supervision. QA accuracy increases from 68.90% to 80.33% at 2B, from 79.73% to 90.13% at 4B, and from 87.70% to 93.20% at 8B; all three improvements are statistically significant $( p < 0 . 0 0 1 )$ The gains are especially large for UNIDENTIFIED questions at 2B and 4B. Figure 2(a) qualitatively illustrates this improvement, with BAC correctly predicting UNIDENTIFIED across all three model scales while the corresponding Qwen3-VL baselines fail. Named accuracy also remains high, indicating that adaptation improves the model’s ability to distinguish grounded identities from people whose identity is not provided.

Captioning follows the same trend: BAC improves overall quality and identity accuracy at all three scales while maintaining a low hallucination rate. BAC-2B already slightly exceeds the strongest 4B base model in QA (80.33% vs. 79.73%), while BAC-4B surpasses all evaluated 8–9B base models (90.13% vs. 87.70% for the strongest baseline). The relative improvement decreases as model size grows, suggesting that task-specific adaptation is particularly beneficial for smaller models, where identity grounding is more challenging.

## 3.3 COMPARISON WITH FRONTIER VLMS

Table 4 compares BAC with substantially larger open-weight and proprietary frontier VLMs. GPT-5.6 Sol achieves the highest overall performance, reaching 97.07% with thinking and 96.30% without thinking; the difference between the two settings is small but statistically significant $( p = 0 . 0 0 7 )$ BAC-8B reaches 93.20% overall, with balanced performance on Named (93.88%) and UNIDEN-TIFIED (91.68%) questions. Although it remains significantly below GPT-5.6 Sol $( p \textless 0 . 0 0 1 )$ BAC-8B significantly outperforms Qwen3-VL-235B-A22B (86.10%), Gemini 2.5 Flash (82.63%), and Claude Sonnet 4.6 (76.10%), all with $p < 0 . 0 0 1$

This advantage is already visible at smaller scales. BAC-4B reaches 90.13% overall accuracy and significantly surpasses the much larger Qwen3-VL-235B-A22B by 4.03 percentage points $( p < 0 . 0 0 1 )$ , while achieving 96.38% accuracy on Named questions. BAC-2B also reaches 80.33%, significantly outperforming Claude Sonnet 4.6 $( p < 0 . 0 0 1 )$ , although it remains below Gemini 2.5 Flash and Qwen3-VL-235B-A22B. Overall, these results show that BAC substantially narrows the gap to frontier proprietary models while remaining compact, open-weight, and self-hostable. Figure 2 illustrates representative BAC successes and failures: (a) correct UNIDENTIFIED prediction, (b) correct named-person identification, and (c) a challenging failure where only GPT-5.6 Sol succeeds. Additional qualitative examples and frontier-model failures are provided in Appendix E.

Category-wise results show that BAC-8B is particularly strong on appearance and object questions, reaching 95.58% and 93.99%, respectively. Its lowest performance is observed on position questions (87.54%), followed by interaction questions (91.28%), suggesting that spatial and relational reasoning remain more challenging than attribute-based recognition. A detailed breakdown across all BAC model scales and frontier Video-MLLMs is provided in Appendix D.

Table 3: Benchmark comparison across model families and sizes with the FVB+FTC grounding method. Captioning is evaluated using overall quality, identity accuracy, and hallucination rate. QA performance is reported over all questions, separately for Named and UNIDENTIFIED (UNID).
<table><tr><td colspan="2"></td><td colspan="3">Captioning</td><td colspan="3">Person-Centric QA</td></tr><tr><td>Size</td><td>Model</td><td>Overall ↑</td><td>Identity ↑</td><td>Halluc. ↓</td><td>Overall ↑</td><td>Named ↑</td><td>UNID↑</td></tr><tr><td rowspan="3">2B</td><td>BAC (ours)</td><td>8.09</td><td>85.33</td><td>3.20</td><td>80.33</td><td>95.90</td><td>45.46</td></tr><tr><td>Qwen3</td><td>7.61</td><td>76.53</td><td>3.07</td><td>68.90</td><td>92.48</td><td>16.09</td></tr><tr><td>InternVL3.5</td><td>5.59</td><td>24.86</td><td>4.48</td><td>63.90</td><td>91.18</td><td>2.81</td></tr><tr><td rowspan="5">4B</td><td>BAC (ours)</td><td>8.48</td><td>92.40</td><td>0.80</td><td>90.13</td><td>96.38</td><td>76.13</td></tr><tr><td>Qwen3</td><td>8.32</td><td>88.80</td><td>1.07</td><td>79.73</td><td>93.11</td><td>49.78</td></tr><tr><td>InternVL3.5</td><td>7.49</td><td>76.93</td><td>6.80</td><td>74.87</td><td>70.38</td><td>84.99</td></tr><tr><td>MiniCPM-V</td><td>6.61</td><td>60.93</td><td>15.47</td><td>69.60</td><td>80.23</td><td>45.79</td></tr><tr><td>Ovis2</td><td>6.40</td><td>47.87</td><td>10.80</td><td>65.27</td><td>71.12</td><td>52.16</td></tr><tr><td rowspan="6">8-9B</td><td>BAC (ours)</td><td>8.59</td><td>94.03</td><td>1.06</td><td>93.20</td><td>93.88</td><td>91.68</td></tr><tr><td>Qwen3</td><td>8.45</td><td>91.60</td><td>0.93</td><td>87.70</td><td>88.04</td><td>86.93</td></tr><tr><td>Ovis2.5 9B</td><td>8.39</td><td>93.73</td><td>1.87</td><td>83.10</td><td>85.39</td><td>77.97</td></tr><tr><td>InternVL3.5</td><td>7.68</td><td>81.73</td><td>6.27</td><td>79.00</td><td>85.28</td><td>65.12</td></tr><tr><td>Eagle2.5</td><td>7.42</td><td>74.13</td><td>4.80</td><td>77.63</td><td>79.17</td><td>74.19</td></tr><tr><td>Molmo2</td><td>5.66</td><td>52.40</td><td>28.13</td><td>69.17</td><td>63.36</td><td>82.18</td></tr></table>

Table 4: Person-centric QA performance and model accessibility. Mean denotes the average of Named and UNIDENTIFIED (UNID) accuracy. API costs are reported in USD per million input/output tokens; All models use the FVB+FTC grounding method.
<table><tr><td>Model</td><td>Overall</td><td>Named</td><td>UNID</td><td>Mean</td><td>Clip all-correct</td><td>Open-weight</td><td>Cost ($/1M in/out)</td></tr><tr><td>GPT-5.6 Sol (think)</td><td>97.07</td><td>97.06</td><td>97.08</td><td>97.07</td><td>90.13</td><td>No</td><td>4.00 / 20.00</td></tr><tr><td>GPT-5.6 Sol (no think)</td><td>96.30</td><td>96.14</td><td>96.65</td><td>96.40</td><td>87.60</td><td>No</td><td>4.00 / 20.00</td></tr><tr><td>BAC-8B (ours)</td><td>93.20</td><td>93.88</td><td>91.68</td><td>92.78</td><td>76.93</td><td>Yes</td><td>Self-hosted</td></tr><tr><td>BAC-4B (ours)</td><td>90.13</td><td>96.38</td><td>76.13</td><td>86.26</td><td>66.40</td><td>Yes</td><td>Self-hosted</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>86.10</td><td>80.67</td><td>98.27</td><td>89.47</td><td>60.93</td><td>Yes</td><td>0.287 / 1.147</td></tr><tr><td>Gemini 2.5 Flash</td><td>82.63</td><td>75.80</td><td>97.95</td><td>86.87</td><td>51.60</td><td>No</td><td>0.30 / 2.50</td></tr><tr><td>BAC-2B (ours)</td><td>80.33</td><td>95.90</td><td>45.46</td><td>70.68</td><td>43.73</td><td>Yes</td><td>Self-hosted</td></tr><tr><td>Claude Sonnet 4.6</td><td>76.10</td><td>66.44</td><td>97.73</td><td>82.09</td><td>41.87</td><td>No</td><td>3.00 / 15.00</td></tr></table>

## 4 CONCLUSION

We introduced a framework for identity-aware video understanding that combines automatic character identification with explicit spatial grounding for video captioning and person-centric question answering. Using LSMDC v2, we constructed a manually verified benchmark of 750 captioned clips and 3,000 questions, and systematically evaluated five identity-grounding representations across multiple model scales and families. Our results show that explicit visual grounding substantially improves identity association compared with textual coordinates alone, with face-based visual grounding combined with textual coordinates providing the most consistent performance across scales. Building on this formulation, we introduced BAC through LoRA-based fine-tuning of Qwen models at 2B, 4B, and 8B scales. BAC-8B achieves 93.20% QA accuracy and remains competitive with substantially larger frontier Video-MLLMs, while maintaining strong performance across question categories. Overall, the results show that explicitly communicating who is where, together with taskspecific adaptation, provides an effective approach to improving identity-aware video understanding without requiring changes to the underlying Video-MLLM architecture.

## 5 ETHICAL CONSIDERATIONS

Our approach combines face recognition, temporal tracking, and spatial grounding to associate known identities with their appearances, actions, and interactions across video frames. Although we study this capability only in the controlled setting of movie understanding, using predefined cast identities and movie clips, similar techniques could potentially be adapted to identify and track individuals in real-world footage, raising concerns related to privacy, surveillance, and unauthorized profiling. Our work is intended to advance character-centric video understanding rather than person identification in unconstrained real-world environments, and we do not evaluate or advocate its use for surveillance or monitoring applications. We therefore encourage future applications of identityaware video understanding to consider appropriate consent, privacy protections, and restrictions on the use of biometric identity information.

## 6 ACKNOWLEDGMENTS

This project was provided with computing AI and storage resources by GENCI at IDRIS thanks to the grant 20XX-AD011017029 on the supercomputer Jean Zay’s H100 partition.

## REFERENCES

David Bamman, Rachael Samberg, Richard Jean So, and Naitian Zhou. Measuring diversity in hollywood through the large-scale computational analysis of film. Proceedings of the National Academy ofSciences, 121(46):e2409770121, 2024.

Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4690–4699, 2019.

Shibo Gao, Chongxiao Wang, Chenglong Huang, Jie Ma, Haolin Shi, Fei Ding, Jing Li, Qiang Lyu, Yangyang Liu, Yang Liu, et al. I seek you in videos: Identity-conditioned queries for personcentric video reasoning. arXiv preprint arXiv:2608.07417, 2026.

Miran Heo, Min-Hung Chen, De-An Huang, Sifei Liu, Subhashree Radhakrishnan, Seon Joo Kim, Yu-Chiang Frank Wang, and Ryo Hachiuma. Omni-rgpt: Unifying image and video region-level understanding via token marks. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3919–3930. IEEE, 2025.

Jermphiphut Jaruenpunyasak, Alba Garc´ıa Seco de Herrera, and Rakkrit Duangsoithong. Anthropometric ratios for lower-body detection based on deep learning and traditional methods. Applied Sciences, 12(5):2678, 2022.

Yatai Ji, Shilong Zhang, Jie Wu, Peize Sun, Weifeng Chen, Xuefeng Xiao, Sidi Yang, Yujiu Yang, and Ping Luo. Ida-vlm: towards movie understanding via id-aware large vision-language model. In International Conference on Learning Representations, volume 2025, pp. 52639–52652, 2025.

Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr - modulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1780–1790, October 2021.

Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization for grounding multimodal large language models. In European Conference on Computer Vision, pp. 417–435. Springer, 2024.

Tegan Maharaj, Nicolas Ballas, Anna Rohrbach, Aaron C Courville, and Christopher Joseph Pal. A dataset and exploration of models for understanding video data through fillin-the-blank question-answering. In Computer Vision and Pattern Recognition (CVPR), 2017. URL http://openaccess.thecvf.com/content\_cvpr\_2017/papers/ Maharaj\_A\_Dataset\_and\_CVPR\_2017\_paper.pdf.

Shehan Munasinghe, Hanan Gani, Wenqi Zhu, Jiale Cao, Eric Xing, Fahad Shahbaz Khan, and Salman Khan. Videoglamm: A large multimodal model for pixel-level visual grounding in videos. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19036– 19046. IEEE, 2025.

Arsha Nagrani and Andrew Zisserman. From benedict cumberbatch to sherlock holmes: Character identification in tv series without a script. arXiv preprint arXiv:1801.10442, 2018.

Stefano Pini, Marcella Cornia, Federico Bolelli, Lorenzo Baraldi, and Rita Cucchiara. M-vad names: a dataset for video captioning with naming. Multimedia Tools and Applications, 78(10):14007– 14027, 2019.

Haran Raajesh, Naveen Reddy Desanur, Zeeshan Khan, and Makarand Tapaswi. Micap: A unified model for identity-aware movie descriptions. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14011–14021. IEEE, 2024.

Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Chris Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele. Movie description. International Journal of Computer Vision, 2017. URL http://link.springer.com/ article/10.1007/s11263-016-0987-1?wt\_mc=Internal.Event.1.SEM. ArticleAuthorOnlineFirst.

Yannis Tevissen, Khalil Guetari, Marine Tassel, Erwan Kerleroux, and Fred´ eric Petitpont. In-´ serting faces inside captions: image captioning with attention guided merging. arXiv preprint arXiv:2405.02305, 2024.

Atousa Torabi, Niket Tandon, and Leon Sigal. Learning language-visual embedding for movie understanding with natural-language. arXiv:1609.08124, 2016. URL http://arxiv.org/ pdf/1609.08124v1.pdf.

Nicolai Wojke, Alex Bewley, and Dietrich Paulus. Simple online and realtime tracking with a deep association metric. In 2017 IEEE international conference on image processing (ICIP), pp. 3645– 3649. IEEE, 2017.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5288–5296, 2016.

Mengdi Xu, Xiaotong Yuan, Jialie Shen, and Shuicheng Yan. Cast2face: Character identification in movie with actor-character correspondence. In Proceedings of the 18th ACM international conference on Multimedia, pp. 831–834, 2010.

Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10714–10726, June 2023.

## APPENDIX

## A ESTIMATING A PERSON BOUNDING BOX FROM A FACE DETECTION

To obtain person-level spatial grounding without requiring an additional person detector, we estimate a full-person bounding box directly from each detected face. Given a face bounding box $B _ { \mathrm { f a c e } } =$ $[ x _ { f } , y _ { f } , w _ { f } , h _ { f } ]$ , we exploit approximate human body proportions to construct an estimated person box $\bar { B _ { \mathrm { b o d y } } } \bar { \bf \Phi } = \bar { [ { x _ { b } , y _ { b } , w _ { b } , h _ { b } } ] }$ . Specifically, the body height and width are obtained by scaling the detected face dimensions,

$$
h _ { b } = K _ { h } h _ { f } , \qquad w _ { b } = K _ { w } w _ { f } ,
$$

where $K _ { h }$ and $K _ { w }$ control the expected body-to-face height and width ratios, respectively. We assume that the body is approximately horizontally aligned with the face. We therefore compute the horizontal center of the face,

$$
x _ { c , f } = x _ { f } + { \frac { w _ { f } } { 2 } } ,
$$

and center the estimated body box around this position,

$$
x _ { b } = x _ { c , f } - \frac { w _ { b } } { 2 } .
$$

Vertically, the person box begins slightly above the detected face in order to include the top of the head,

$$
y _ { b } = y _ { f } - K _ { \mathrm { o f f s e t } } h _ { f } ,
$$

where $K _ { \mathrm { o f f s e t } }$ controls the upward extension. The resulting box therefore preserves the spatial location of the detected face while extending it according to approximate human proportions to cover the person’s visible body. This procedure is heuristic rather than a learned body-detection method and provides the estimated person regions used in our person-box grounding representations.

Implementation parameters. We use the same fixed parameters for all experiments. Specifically, we empirically set the body-height scaling factor to $K _ { h } ~ = ~ 6 . 5$ , the body-width scaling factor to $K _ { w } = 2 . 5$ , and the vertical offset to $K _ { \mathrm { o f f s e t } } = 0 . 1 5$ . Thus, the estimated person box has a height of $6 . 5 h _ { f }$ and a width of $2 . 5 w _ { f }$ , and starts $0 . 1 5 h _ { f }$ above the detected face box. These parameters are applied uniformly to all detected faces without additional person detection or per-frame adaptation.

## B EFFECT OF FACE AND PERSON SIZE

As shown in the main results, the relative effectiveness of FVB+FTC and PVB+PTC depends partly on target size. When the face is very small, the full-person box can provide a more informative spatial cue. In Figure 3, the target face occupies only 2,222 $\mathrm { p x } ^ { 2 }$ with a height of 52 px; PVB+PTC correctly identifies the target across all Qwen3-VL sizes, whereas FVB+FTC fails.

Conversely, when the face is large and visually informative, FVB+FTC can be more reliable. In Figure 4, the face occupies $3 9 1 , { \overset { \smile } { 8 9 } } 7 ~ \mathrm { p x } ^ { 2 }$ with a height of 760.5 px; FVB+FTC succeeds across all model sizes, while PVB+PTC fails for Qwen3-VL 2B. The full-person boxes also overlap in this example, introducing visual clutter that can weaken the identity–location association.

These examples support the quantitative results: PVB+PTC is particularly useful for very small faces, whereas FVB+FTC becomes more reliable as face size increases. For already informative faces, full-person boxes may add little benefit and can even interfere with grounding when they overlap.

![](images/bde489e49d0fc3b3c6e01f18361e45c09f548ef5d9cf305edd0144f3768b8993.jpg)

![](images/b6fe8c6c5b66c547bc2fec3356754892c45d9918dd25f8bd55011d29454d54a1.jpg)

![](images/9dc1b2a2866aa623934099ee450892ae38cc9fb3af9e94706fcfbd6b66229631.jpg)

![](images/f064bc961442223e50a2146aaf62fe44e9d31056c5e6081e0f59dbe9e981ab1b.jpg)

![](images/1c2c8f2591bb3f563d7530f3b66c0c887d274bbdea5e046e902292930c27199f.jpg)

![](images/c4454c6ae51734484dccd92fc252a73f5dcd434df69ace05e9c006a0f8cdf26e.jpg)

![](images/6163c7c14db85c30cb602431dc9c3d7c3bcae652563687cce5d6172da4e257f4.jpg)  
(a) Face grounding

![](images/0fbe1c8d22e19b68a064c6c7ff4dbea39399ffcab031711f396d3fe85336b39d.jpg)

![](images/f77239339d4170805bc216ec88e03bf5cf6bdd3c5823697df9cb248ef2f44103.jpg)

![](images/3174e0d5c4449f5154896ea030885c7a84f9089d0de8d90f125a89dd42cde430.jpg)

![](images/4d4ffd6637e3d60902e5003d6a34f42c25864c9f84ab7c754fc4762b8ad90e62.jpg)

![](images/106f2eeccf4bcee8552feb8e9188e80b28477c4c6429a9288acf7d7208437640.jpg)

![](images/a525e35493504cc8f932af3e82edf364ad7bef2d4b8c6c7b958e769e4d4f2094.jpg)

![](images/bb4a5778c2a8938f72eb0076d5ece9c5e4aec13f6e80408001f2cb288f2817e6.jpg)

![](images/d2a4e73aa0c81679d6d3a6eec90f76f5e769520556eac28336df239ac4769dcf.jpg)  
(b) Person grounding

![](images/ee85c75df519fd9f79ded125e23aecd81580064c806a474202d69c3cfd338fb8.jpg)

Identity mapping: P1 ↔ Tyrese Gibson; P2 ↔ Dennis Quaid; P3 ↔ Lucas Black; P4 ↔ Adrianne Palicki. Q2 [Appearance]: Who is pregnant? GT: Adrianne Palicki. P4 face size: $2 , 2 2 2 { \mathrm { p x } } ^ { 2 } ( 5 2$ px height).
<table><tr><td>Model</td><td>FVB+FTC</td><td>PVB+PTC</td></tr><tr><td>Qwen3-VL 2B</td><td>× Lucas</td><td>√Adrianne</td></tr><tr><td>Qwen3-VL 4B</td><td>× Lucas</td><td>√Adrianne</td></tr><tr><td>Qwen3-VL 8B</td><td>X UNIDENTIFIED</td><td>√Adrianne</td></tr></table>

Figure 3: Qualitative comparison of face- and person-level grounding for a small-face example. The target character P4 (Adrianne Palicki) has a very small face area of 2,222 $\mathrm { p x } ^ { 2 }$ and a face height of 52 px. PVB+PTC grounding method correctly identifies Adrianne Palicki across all model sizes, whereas FVB+FTC grounding method predicts Lucas Black at 2B and 4B and UNIDENTIFIED at 8B.

![](images/6cc4fd7220ccaf44f4e9288e2bc4c345e04afde16e5d8a749f7bde04d5a6275d.jpg)

![](images/2571e959342b04ff83d2a098afc9aa272e391714d94c0ae55ed3e949a95c4590.jpg)

![](images/02ffd0b26ea9b7fdf6159e644bee91c218a8eb390e1cf647771c3ab0abb8e201.jpg)  
(a) Face grounding

![](images/35d546bd17b5861f14f1e6c3fd5c24ea2fff89ccaa46281a7e8183b338e4d6eb.jpg)

![](images/a1928e162c07cc1ac370d8c5ec5c0b0a146e238d372756e096e1fa939d4693b6.jpg)

![](images/7d72a3db9c452a7d4e42bd9704ed40edc4d6b17b89b9166d9a6e4753e9286fa4.jpg)

![](images/dc61ee5a001f28466a4370c2c4215b29621b8b13922718e301c02c6cc4f6496d.jpg)  
(b) Person grounding

![](images/5495e6f77db8b4ca6db48825b2c69ef9c077fe6bd171962ba07140e78d516581.jpg)

Identity mapping: P1 ↔ Joaquin; P2 ↔ Abigail.  
Q2 [Action]: Who opens their eyes with a frightened expression? GT: Joaquin. P1 face size: 391,897 px<sup>2</sup> (760.5 px height).
<table><tr><td>Model</td><td>FVB+FTC</td><td>PVB+PTC</td></tr><tr><td>Qwen3-VL 2B</td><td>√ Joaquin</td><td>× Abigail</td></tr><tr><td>Qwen3-VL 4B</td><td>√ Joaquin</td><td>√ Joaquin</td></tr><tr><td>Qwen3-VL 8B</td><td>√ Joaquin</td><td>√ Joaquin</td></tr></table>

Figure 4: Qualitative comparison of face- and person-level grounding for an action question. The target character P1 (Joaquin) has a very large face area of $3 9 1 , 8 9 7 \mathrm { p x } ^ { 2 }$ and a face height of 760.5 px. FVB+FTC grounding method correctly identifies Joaquin across all three model sizes. PVB+PTC grounding method predicts Abigail at 2B, but correctly identifies Joaquin at 4B and 8B.

## C BAC FINE-TUNING AND MODEL SCALING

LoRA fine-tuning. Our objective is to fine-tune Qwen3-VL base models at different scales on the training data we build as described in Section 2.1.2. We consider the 2B, 4B, and 8B variants and perform parameter-efficient adaptation using LoRA. We refer to the resulting fine-tuned models as BAC-2B, BAC-4B, and BAC-8B. The adapters are applied only to the language-model projection layers, while the vision encoder and multimodal projector remain frozen.

To determine an effective LoRA configuration, we conduct a five-setting hyperparameter sweep for one epoch, varying the learning rate, LoRA rank r, scaling factor α, and dropout. The best configuration is obtained with a learning rate of $5 \times 1 0 ^ { - 5 } , r = \mathrm { \bar { 1 6 } } , \alpha = 3 2$ , and a dropout of 0.05. At this rank, LoRA introduces 17.4M trainable parameters for BAC-2B, 33.0M for BAC-4B, and 43.6M for BAC-8B. Figure 5 shows the training and validation losses for the 2B, 4B, and 8B models using the best configuration, with both curves reported together for each model size. For each BAC variant, the checkpoint achieving the lowest validation loss is used for inference.

![](images/7d4cd4c555171ff55a2cbca03e80060f234c03559ae88871bd2be7b3315f2a93.jpg)  
(a) BAC-2B

![](images/f009b60ca72bf2204f136d5cdf437c970083f9e3ca4840a292ec29f44c35f6d4.jpg)  
(b) BAC-4B

![](images/2e621a5973d6a01292833252e14cfa7b10994a748ca6d5dbf21dcd4d7ae8c673.jpg)  
(c) BAC-8B  
Figure 5: Training dynamics of the BAC models at three scales. For each model, we report the training loss, evaluation loss, and learning-rate schedule during LoRA fine-tuning. The dashed vertical line indicates the checkpoint with the best evaluation loss.

## D CATEGORY-WISE PERSON-CENTRIC QA PERFORMANCE

Figure 6 provides a category-wise breakdown of person-centric QA performance across both frontier and similarly sized models. Overall, BAC-8B maintains strong performance across all question types, indicating that task-specific adaptation benefits a broad range of identity-aware reasoning categories.

Among frontier models, GPT-5.6 Sol thinking achieves the highest accuracy across all categories, while BAC-8B remains competitive, with its strongest absolute results on appearance, object, and posture questions and its lowest performance on position. BAC-4B follows a similar pattern, whereas BAC-2B substantially larger variation across categories. Among 8–9B models, BAC-8B achieves the highest accuracy in every category and consistently outperforms its Qwen3-VL 8B base, with the largest gains over the base model observed for object, appearance, and posture questions.

![](images/11290abbb7bebb33db9e298e2658822d72936aaa97f65cc21a600889d0242b98.jpg)

![](images/c17643eb2e0f5bb5944aa92f0979bec9203201c40ce5b894fffd21eaa0515a8e.jpg)  
Figure 6: Person-centric QA accuracy across question categories. Top: comparison with frontier models, with categories ordered by GPT-5.6 Sol\* performance. Bottom: comparison between BAC-8B and other 8–9B models, with categories ordered by BAC-8B performance. GPT-5.6 Sol\* denotes GPT-5.6 Sol with thinking enabled.

## E ADDITIONAL QUALITATIVE RESULTS

![](images/0c38c3c15a24a6d71dd0a97734724fdf92f942b3652bded910647e1b055c1a8a.jpg)

![](images/7a2026059ac5a292ab90a8adbd683ff5649dbc58129e335a8a5cd89e2d6e5de1.jpg)  
(a) Identity mapping: P1 ↔ Leslie Mann; P2 ↔ Paul Rudd.

![](images/3e6a234a8904035da83abc4bac79e592a276169b3f65b09f46e2f3dac7901a27.jpg)

![](images/c00462ba670930ee1fd6a330f0aad7b21d79394f39311e5d0aff15dfc45ca03b.jpg)

Q2 [Posture]: Who is lying in a hospital bed? GT: Paul Rudd (P2).  
Q3 [Action]: Who gathers her dress as she walks past the hospital bed? GT: Leslie Mann (P1).
<table><tr><td>Model</td><td>Q2</td><td>Q3</td><td>Model</td><td>Q2</td><td>Q3</td><td>Model</td><td>Q2</td><td>Q3</td></tr><tr><td>BAC-2B (ours)</td><td>V</td><td>V</td><td>BAC-4B (ours)</td><td>V</td><td>V</td><td>BAC-8B (ours)</td><td>V</td><td>V</td></tr><tr><td>Qwen3-VL 2B</td><td>7</td><td>5</td><td>Qwen3-VL 4B</td><td>V</td><td>V</td><td>Qwen3-VL 8B</td><td>√</td><td>V</td></tr><tr><td>GPT-5.6 Sol</td><td>Leslie X</td><td>Paul </td><td>GPT-5.6 Sol*</td><td>Leslie X</td><td>Paul ×</td><td>Claude Sonnet 4.6 Leslie × Paul ×</td><td></td><td></td></tr></table>

![](images/e155c230de5c67e3870e0d9de1ebfcb59d3214e92d9a67951a4adc0294118a89.jpg)

![](images/dfd8dd904e0457d8ec830c2b37281c22913f5962d612798e4d064aef3bebe2f3.jpg)  
(b) Identity mapping: P1 ↔ Chris Pine.

![](images/937fd7f8f492b825b739932d2e065a15eed9655ed20afa612af0d688ae10653a.jpg)

![](images/ca16e07792557f7fe3443c0cba9607733943a1d7d4c9510bd4a612503a02ba26.jpg)

Q3 [Interaction]: Who is being punched in theface? GT: UNIDENTIFIED.
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td></tr><tr><td>BAC-2B (ours)</td><td>Chris</td><td>×</td><td>BAC-4B (ours)</td><td>Chris</td><td>×</td><td>BAC-8B (ours)</td><td>Chris</td><td>X</td></tr><tr><td>Qwen3-VL 2B</td><td>Chris</td><td>X</td><td>Qwen3-VL 4B</td><td>Chris ×</td><td></td><td>Qwen3-VL 8B</td><td>Chris</td><td>×</td></tr><tr><td>GPT-5.6 Sol</td><td>UNID.</td><td>V</td><td>GPT-5.6 Sol*</td><td>Chris ×</td><td></td><td>Claude Sonnet 4.6 UNID. √</td><td></td><td></td></tr><tr><td>Qwen3-VL 235B</td><td>Chris</td><td>×</td><td>Gemini 2.5 Flash</td><td>Chris ×</td><td></td><td></td><td></td><td></td></tr></table>

![](images/9fcf4a8a39b10d3a9e9bf412b056f4b17bde47017f5a5ab08453720554e7efff.jpg)

![](images/b60b0c0c9279ab077c6fae48ac069233cf4fc11b0b7b86cd0b18d532ebff9b10.jpg)  
(c) Identity mapping: P1 ↔ Elliot Page.

![](images/b98bc108aca4a3156465f542c724d490d30ea265407a8aa639561d27b99b36ed.jpg)

![](images/bf6e482c1a7d013a12e0080f2dab5467114c0ae4e86d25723a3379233bb595fb.jpg)

Q4 [Action]: Who holds up an ultrasound image to look at it? GT: UNIDENTIFIED.
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td>Model</td><td>Pred.</td><td></td></tr><tr><td>BAC-2B (ours)</td><td>UNID.</td><td>V</td><td>BAC-4B (ours)</td><td>UNID.</td><td></td><td>BAC-8B (ours)</td><td>UNID.</td></tr><tr><td>Qwen3-VL 2B</td><td>Elliot</td><td>X</td><td>Qwen3-VL 4B</td><td>UNID.</td><td>y</td><td>Qwen3-VL 8B</td><td>UNID. V</td></tr><tr><td>GPT-5.6 Sol</td><td>Elliot</td><td>X</td><td>GPT-5.6 Sol*</td><td>Elliot</td><td>X Claude Sonnet 4.6</td><td>UNID.</td><td></td></tr><tr><td>Qwen3-VL 235B</td><td>UNID.</td><td>」</td><td>Gemini 2.5 Flash</td><td>UNID.</td><td>V</td><td></td><td>y</td></tr></table>

![](images/c2f044d5738e88c4d762c77c95b128ab9c882ee3770ad88b87b3e5d4998cdded.jpg)

![](images/e7c2328a3edc20999ea22f85337d4327ccd52aa14457ccf4b5670f5f1c4106c4.jpg)  
(a) Identity mapping: P1 ↔ Paul Bettany.

![](images/b0445ab952337a6dec0c42f2f7e38ddf37c5dad9338e2af88480e22f3cbfcd05.jpg)

![](images/14d31381a90b611b6c218f5520a87078626005022df14b387204b1698fdc3afb.jpg)

Q1 [Appearance]: Who is shirtless with blood dripping down his back? GT: Paul Bettany (P1).
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td></tr><tr><td>BAC-2B (ours)</td><td>Paul</td><td>L</td><td>BAC-4B (ours)</td><td>Paul</td><td></td><td>BAC-8B (ours)</td><td>Paul</td><td></td></tr><tr><td>Qwen3-VL 2B</td><td>Paul</td><td>V</td><td>Qwen3-VL 4B</td><td>Paul</td><td></td><td>Qwen3-VL 8B</td><td>Paul</td><td>V</td></tr><tr><td>GPT-5.6 Sol</td><td>UNID.</td><td>X</td><td>GPT-5.6 Sol*</td><td>Paul</td><td>√</td><td>Claude Sonnet 4.6</td><td>UNID.</td><td>X</td></tr><tr><td>Qwen3-VL 235B</td><td>UNID.</td><td>X</td><td>Gemini 2.5 Flash</td><td>UNID.</td><td>X</td><td></td><td></td><td></td></tr></table>

![](images/48651a141b39eeea4a9abe0b3dfc21703c20aa636037f9d038a3b774bab48384.jpg)

![](images/c4555f02d2dec7c8d4b98a7d18903c9fd10e5dfdaaf8aa5e02fd27bad87c119c.jpg)

![](images/5adb93747b9a7e97d7ea77276d9cc4e2a3afe8a8c3b76393263d790d563de870.jpg)

![](images/629e13d6dd0c9f824dcc91bf3ca1bc1ade8efa8a1813ff774a3ecf7f68ee3e6c.jpg)

(b) Identity mapping: P1 ↔ Michelle Rodriguez; P2 ↔ Bridget Moynahan; P3 ↔ Michael Pena˜ . Q1 [Posture]: Who is leaning against the refrigerator or cabinet? GT: Bridget Moynahan (P2).
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td></tr><tr><td>BAC-2B (ours) Qwen3-VL 2B</td><td>Bridget Bridget</td><td>V V</td><td>BAC-4B (ours) Qwen3-VL 4B</td><td>Michael</td><td>X V</td><td>BAC-8B (ours) Qwen3-VL 8B</td><td>Bridget</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Bridget</td><td></td><td></td><td>Bridget</td><td></td></tr><tr><td>GPT-5.6 Sol</td><td>Michael</td><td>X</td><td>GPT-5.6 Sol*</td><td>Michael</td><td>×</td><td>Claude Sonnet 4.6</td><td>Bridget</td><td></td></tr><tr><td>Qwen3-VL 235B</td><td>Bridget</td><td>V</td><td>Gemini 2.5 Flash</td><td>UNID.</td><td>X</td><td></td><td></td><td></td></tr></table>

![](images/ef136df88018e33faad4d013ea35fc785143889e5eb61c9bd7448945e38e5720.jpg)

![](images/bc31bf585f91d30a6155310943eac548089c0a832260daf4e8966f3b12e770c7.jpg)

![](images/ee630ee2e638e97746669a1936bd66e335ff81767aebe8f40a2206330d0e4f5a.jpg)  
(c) Identity mapping: P1 ↔ Jon Tenney; P2 ↔ Charles S. Dutton.

![](images/55f97cc1c9e4cd66165aeba1df3454414dd707902e73f4ba60063169181a13d9.jpg)

Q3 [Interaction]: Who tries to pull Jon back into the vehicle? GT: Charles S. Dutton (P2).
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td>Model</td><td></td><td>Pred.</td><td></td></tr><tr><td>BAC-2B (ours) Qwen3-VL 2B</td><td>Charles Charles</td><td></td><td>BAC-4B (ours) Qwen3-VL 4B</td><td>Charles UNID.</td><td>√ X</td><td>BAC-8B (ours) Qwen3-VL 8B</td><td>Charles Charles</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.6 Sol</td><td>Charles</td><td>V</td><td>GPT-5.6 Sol*</td><td>UNID.</td><td>X</td><td>Claude Sonnet 4.6</td><td>UNID.</td><td>×</td></tr><tr><td>Qwen3-VL 235B</td><td>UNID.</td><td>X</td><td>Gemini 2.5 Flash</td><td>UNID.</td><td>X</td><td></td><td></td><td></td></tr></table>

![](images/5738ac8031d347d7221b992f2f2be182fb1521c43ea0b2afb2c816bc3c7cd772.jpg)

![](images/7bf202083d8a9bb04c3848b858ec19136662455f003775f4e0ce3da7e4b96b30.jpg)  
(a) Identity mapping: P1 ↔ Anjali Jay; P2 ↔ Sendhil Ramamurthy.

![](images/2d22b52b9c82d4564b05fd6f81dde91cb5739eca5e9f003872ada95eae8c4915.jpg)

![](images/7fd1c105b06b1903822940c5c8691f714ab6fd63fd83e71cf6071f137113f88e.jpg)

Q2 [Appearance]: Who is wearing a red sleeveless dress? GT: Anjali Jay (P1).
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td></tr><tr><td>BAC-2B (ours) Qwen3-VL 2B</td><td>Anjali</td><td>V</td><td>BAC-4B (ours)</td><td>Anjali</td><td>√</td><td>BAC-8B (ours)</td><td>UNID. X</td></tr><tr><td></td><td>Anjali</td><td>V</td><td>Qwen3-VL 4B</td><td>UNID.</td><td>X</td><td>Qwen3-VL 8B</td><td>UNID. X</td></tr><tr><td>GPT-5.6 Sol</td><td>UNID.</td><td>X</td><td>GPT-5.6 Sol*</td><td>Anjali</td><td>V</td><td>Claude Sonnet 4.6</td><td>UNID. X</td></tr><tr><td>Qwen3-VL 235B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>UNID.</td><td>X</td><td>Gemini 2.5 Flash</td><td>UNID.</td><td>X</td><td></td><td></td></tr></table>

![](images/3acc8ec17c0168648a2de1207015f8fd08d342eb518436169539e1f3f89be9fc.jpg)

![](images/9985943bd351293df020161a5a0cb977edc6abdd180cef4157450f09c1de32af.jpg)  
(b) Identity mapping: P1 ↔ Adrianne Palicki.

![](images/34b70af5df31eae70cb869f2e82e966d5d277d504e4cd3cb53ce8af8b48c8df6.jpg)

![](images/ae472ad61756bf21025e8bbb162fa3eeba098f268877ffd569db38e774082151.jpg)

Q3 [Posture]: Who is leaning against the sink? GT: Adrianne Palicki (P1).
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td></tr><tr><td>BAC-2B (ours)</td><td>Adrianne</td><td></td><td>BAC-4B (ours)</td><td>Adrianne</td><td></td><td>BAC-8B (ours)</td><td>Adrianne</td><td></td></tr><tr><td>Qwen3-VL 2B</td><td>Adrianne</td><td>V</td><td>Qwen3-VL 4B</td><td>Adrianne</td><td></td><td>Qwen3-VL 8B</td><td>Adrianne</td><td>V</td></tr><tr><td>GPT-5.6 Sol</td><td>UNID.</td><td>X</td><td>GPT-5.6 Sol*</td><td>Adrianne</td><td>V</td><td>Claude Sonnet 4.6</td><td>UNID.</td><td>×X</td></tr><tr><td>Qwen3-VL 235B</td><td>Adrianne</td><td>V</td><td>Gemini 2.5 Flash</td><td>UNID.</td><td>×</td><td></td><td></td><td></td></tr></table>

![](images/7505881724096654ecba8a7fb13e4ff00f323603528bfb562755850a50534351.jpg)

![](images/08a660194ca9b4c586c1a8b43a10d95b34a82503bcbe4ccc5aeb8a95de896669.jpg)

![](images/2a2ace733fb4d2540fb9de5df99a7435b7853515860d42f1ecb78f471bb14b79.jpg)  
(c) Identity mapping: P1 ↔ Martin Lawrence; P2 ↔ Tracy Morgan.

![](images/ce333ccb16e9ceb01988689784e334a90c805024b5c17bbec6d7313d64d113fd.jpg)

Q4 [Interaction]: Who is holding a young man from behind, supporting him under the arms? GT: Martin Lawrence (P1).
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td></tr><tr><td>BAC-2B (ours) Qwen3-VL 2B</td><td>Martin</td><td></td><td>BAC-4B (ours)</td><td>Martin</td><td></td><td>BAC-8B (ours)</td><td>Martin</td><td></td></tr><tr><td></td><td>Martin</td><td>y</td><td>Qwen3-VL 4B</td><td>Martin</td><td></td><td>Qwen3-VL 8B</td><td>Martin</td><td></td></tr><tr><td>GPT-5.6 Sol Qwen3-VL 235B</td><td>Martin UNID.</td><td>V ×</td><td>GPT-5.6 Sol* Gemini 2.5 Flash</td><td>UNID. Martin</td><td>X V</td><td>Claude Sonnet 4.6</td><td>UNID.</td><td>×</td></tr></table>

![](images/7f68147ed335dc2668a0529a3d9151a2c793fac059f226a52061d73c7478af41.jpg)

![](images/15ced32ea14c2a6758f537025ee3ba5ea5f848bb4fb179daf98bf61ba186e69f.jpg)  
(a) Identity mapping: P1 ↔ Michael Cera; P2 ↔ Elliot Page.

![](images/e380105913187a45064f647cc816345d1ed76a095d7b68f3917494a532e74ce4.jpg)

![](images/e6f6caa04073e3e9d97f562771eae69e336fdbeb76892e7508fedefcf36be76c.jpg)

Q2 [Appearance]: Who is wearing a light blue shirt? GT: Elliot Page (P2). Q4 [Posture]: Who rests against a pillow with closed eyes before opening them? GT: Elliot Page (P2).
<table><tr><td>Model</td><td>Q2</td><td>Q4</td><td>Model</td><td>Q2</td><td>Q4</td><td>Model</td><td>Q2</td><td>Q4</td></tr><tr><td>BAC-2B (ours)</td><td></td><td>V</td><td>BAC-4B (ours)</td><td>V</td><td>V</td><td>BAC-8B (ours)</td><td></td><td>V</td></tr><tr><td>Qwen3-VL 2B</td><td>Michael ×</td><td>V</td><td>Qwen3-VL 4B</td><td>V</td><td>V</td><td>Qwen3-VL 8B</td><td>V</td><td>V</td></tr><tr><td>GPT-5.6 Sol</td><td>Michael × Michael ×</td><td></td><td>GPT-5.6 Sol*</td><td>1</td><td></td><td>Claude Sonnet 4.6</td><td>V</td><td>Michael ×</td></tr><tr><td>Qwen3-VL 235B</td><td></td><td></td><td>Gemini 2.5 Flash UNID. ×</td><td></td><td>V</td><td></td><td></td><td></td></tr></table>

![](images/8c27441666b1bc3c3fe0978d0f4936653a95b372e2c1e76b11b8a178e0a2c6ef.jpg)

![](images/6b555d678aea92983103202a1d4ada06eb8e8e27b39f326b8d9c1d1b71e21fc6.jpg)

![](images/690a78bb90c62d03d839972f65b99d2f7282d534163209ee62560c58bb349b41.jpg)  
(b) Identity mapping: P1 ↔ Marc John Jefferies; P2 ↔ Jessica Lucas.

![](images/67d39d691439c64922ca15865915d473f200124e8d92d0d74f6ffb528112003a.jpg)

Q4 [Interaction]: Who turns back to look while walking away hand-in-hand with the man in the gray hoodie? GT: Jessica Lucas (P2).
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td></tr><tr><td>BAC-2B (ours)</td><td>Jessica</td><td>V</td><td>BAC-4B (ours)</td><td>Jessica</td><td></td><td>BAC-8B (ours)</td><td>Jessica</td><td></td></tr><tr><td>Qwen3-VL 2B</td><td>Marc</td><td>X</td><td>Qwen3-VL 4B</td><td>Jessica</td><td></td><td>Qwen3-VL 8B</td><td>Jessica</td><td></td></tr><tr><td>GPT-5.6 Sol</td><td>UNID.</td><td>×</td><td>GPT-5.6 Sol*</td><td>Jessica</td><td>V</td><td>Claude Sonnet 4.6</td><td>Jessica</td><td></td></tr><tr><td>Qwen3-VL 235B</td><td>Jessica</td><td>L</td><td>Gemini 2.5 Flash</td><td>UNID.</td><td>X</td><td></td><td></td><td></td></tr></table>

![](images/7f54b478a22ddd22dda1d38e46fe30b51f57332fe0ec63e144d397336d7a629b.jpg)

![](images/ebf374ce1128958ebe6b99eee4a00e295403bb17869685fbe27e6dc1c6b60536.jpg)

![](images/f0e3e85be6d2d6e21edb3222640d899064e3d8ad340433351815bbbd655eeb0d.jpg)

![](images/38ecb0096f25a744e65d78793e8199e4b8867448baaa92e0039585330308db1b.jpg)  
(c) Identity mapping: P1 ↔ Olivia Thirlby; P2 ↔ Allison Janney; P3 ↔ Elliot Page.

Q1 [Appearance]: Who is reclining with an exposed pregnant belly? GT: Elliot Page (P3).
<table><tr><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td><td>Model</td><td>Pred.</td><td></td></tr><tr><td>BAC-2B (ours)</td><td>Elliot</td><td></td><td>BAC-4B (ours)</td><td>Elliot</td><td>V</td><td>BAC-8B (ours)</td><td>Allison</td><td>X</td></tr><tr><td>Qwen3-VL 2B</td><td>Elliot</td><td>V</td><td>Qwen3-VL 4B</td><td>Allison</td><td>X</td><td>Qwen3-VL 8B</td><td>Allison</td><td>X</td></tr><tr><td>GPT-5.6 Sol</td><td>UNID.</td><td>X</td><td>GPT-5.6 Sol*</td><td>Elliot</td><td>L</td><td>Claude Sonnet 4.6</td><td>UNID.</td><td>×</td></tr><tr><td>Qwen3-VL 235B</td><td>Elliot</td><td>V</td><td>Gemini 2.5 Flash</td><td>Elliot</td><td></td><td></td><td></td><td></td></tr></table>