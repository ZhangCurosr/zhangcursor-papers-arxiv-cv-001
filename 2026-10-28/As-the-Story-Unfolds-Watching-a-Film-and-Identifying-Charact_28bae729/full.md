# As the Story Unfolds: Watching a Film and Identifying Characters as a Human Does

Zhongrui Gui <sup>1</sup> Junyu Xie<sup>1</sup> Tengda Han<sup>1</sup> Weidi Xie<sup>1,</sup> <sup>2</sup> Andrew Zisserman<sup>1</sup>   
<sup>1</sup>Visual Geometry Group, Department of Engineering Science, University of Oxford, UK <sup>2</sup>School of Artificial Intelligence (SAI), Shanghai Jiao Tong University, China {zhongrui,jyx,htd,weidi,az}@robots.ox.ac.uk

![](images/857bc40adb8c72e9a4e12893904ef4c4b5b307ada0c2418abee5d51964aa2b74.jpg)  
Figure 1:<sub>Direct</sub> <sub>Address</sub> <sub>(Self-Introduction)</sub>Character naming task. We associate character names to their facial appearances in movies,<sub>Direct</sub> <sub>Address</sub> <sub>(Eyeline</sub> <sub>Match) Third-Party</sub> <sub>Reference</sub> mirroring how viewers infer who is who while watching. Characters may be introduced through diverse cues, including self-introduction, direct address, and third-party reference. We show three representative examples from Four Weddings and a Funeral (1994). Correctly linking a character to their name is central to movie understanding, enabling coherent character tracking across scenes as the story unfolds.

## Abstract

Films introduce characters incrementally as they progress. We study the problem of identifying principal characters and assigning their names directly from the movie itself, without using external cast lists, actor photographs, or pre-built character banks. We introduce the character naming task: given a name mentioned in dialogue and the associated video context, the model must determine which visible character bears that name. This task is challenging because name resolution often depends on cinematic cues such as shot–reverse-shot structure, self-introduction, direct address, third-party reference, gaze, and temporal continuity. We provide a dataset and annotations for evaluating character naming and first character appearances, develop a video-anddialogue model for the task, and compare it with proprietary multimodal large language models. We further introduce a progressive framework for building a character bank as the film unfolds. Together, these contributions support online, human-like movie understanding in which character identities are inferred from the film itself. Website: https://www.robots.ox.ac.uk/\~vgg/research/StreamCast/.

## 1 Introduction

The objective of this paper is to watch a film as a human would – ingesting the film frame-by-frame in an online mode, and identifying the principal characters and their names as the story unfolds. Traditional films, such as a romantic comedy, consist of characters and their story arcs. In order for the viewer to quickly understand who these principal characters are, the film must introduce and name them towards the start of the film. This is achieved by a combination of cinematic devices (such as the camera following the person, and close-ups) and using the dialogue or other information to name them, as illustrated in Fig. 1.

Obtaining character names explicitly only from the film is in stark contrast to the methods employed recently in the literature for tasks like generating audio descriptions and transcripts [21, 24– 26, 43, 67, 68]. In these works, a character bank is first constructed using a cast list and actor photos from IMDB, Fandom and other online sources. The character bank is then used to label the principal characters throughout the film using face recognition (or voice recognition for animated films [21]).

To operationalize our goal, we first investigate how the viewer is presented with the name of a character as they watch the film – almost always it is from the dialogue, and very occasionally from other sources, such as name badges. This enables the character naming task to be automated and evaluated – given a name in the dialogue, the associated shots can be used to ask the question: which of the people appearing in this clip has this name? To this end, the first contribution of this paper is to build a dataset from MovieNet [28] and Movie101 [73] films that can be used to train and evaluate this task. We develop a method to solve this task, using the video and dialogue, and compare its performance to proprietary Multimodal LLMs (MLLMs), such as Gemini [19]. This is a challenging task, as correct inference often requires an understanding of film editing style (as the person addressed by name may be shown in a different shot) and also character gaze and eyeline (to know who they are looking at when saying a name).

This establishes that characters can be named in films, using the information presented by the screenwriter and director. Often, the name of a principal character is presented multiple times across different dialogues throughout the film, providing opportunities for error correction (by both human and machine) if the first presentation had been missed or incorrectly assigned. Once the principal characters are named (possibly multiple times), they can be recognized throughout the rest of the movie, and retrospectively named from their first appearance (using human or machine memory). To this end, we introduce two further tasks: (i) annotating the movies with the first appearance of the principal characters (which may well be before they are named in the dialogue); and (ii) progressively building up a character bank as the film progresses. The character bank records the appearance, voice and names of the principal characters (as in previous work). We provide annotations to evaluate the character first appearance over a number of movies. For example, in the movie Juno, the protagonist Juno is introduced in the first shot of the film, and appears in almost every scene, but is not named until 7.2 minutes into the film.

In summary, we make the following four contributions: (i) we introduce the new task of character naming from dialogue alone, a key enabling step for watching and understanding a film as a human does; (ii) we provide the Movie Character Naming (MCN) dataset with 23,792 instances of multiple-choice character naming over 407 movies, and also 20 movies with the first appearance of the principal characters annotated; (iii) we develop a character naming model using open-source MLLMs, and compare its performance to proprietary MLLMs, analysing what is difficult to solve; and (iv) we introduce the first automatic method to progressively build up a character bank for a feature-length movie. These capabilities put in place the foundation for understanding character arcs and, ultimately, the story of a film. The dataset and models are available at: https://www.robots.ox.ac.uk/\~vgg/research/StreamCast/.

## 2 Movie Character Naming (MCN) Dataset

In this section, we introduce the Movie Character Naming (MCN) Dataset, which provides annotations for two complementary tasks: Character Naming and First Occurrence. The Character Naming annotations consist of video segments in which character names are uttered in the dialogue, each paired with ground-truth name-to-track mappings. The First Occurrence annotations, by contrast, are designed to evaluate the streaming pipeline for character bank construction, and provide two (possibly coinciding) timestamps per principal character: the moment they first appear on screen, and the moment their name is first revealed.

Fig. 2 shows a representative MCN sample for the character Carrie from Four Weddings and a Funeral. The movie timeline is annotated with Character Naming and First Occurrence labels. Additional examples are provided in Appendix H. In the following, we first describe how the dataset is annotated given character tracks and dialogue. The implementation details for obtaining the character face-tracks and dialogue are given in Sec. 2.3.

## 2.1 Data Collection and Curation

We build MCN on top of MovieNet [28] and Movie101 [73], which already provide ground-truth mappings between character names and body bounding-box coordinates.

![](images/5c96c73f445da157c7de627c7323b0402ca2f1d954479ae36179fe0bf51881f3.jpg)  
Figure 2: Example annotations from the MCN dataset. The MCN dataset provides manually annotated timestamps for each character’s first visual occurrence and first naming, along with ground-truth name-to-face correspondences for the Character Naming task.

Table 1: Statistics of the Character Naming (CN) and First Occurrence (FO) annotations. Each question corresponds to a name-to-track mapping query or a first-occurrence timestamp annotation.
<table><tr><td>Task</td><td>Split</td><td># Movies</td><td>Duration (h)</td><td># Characters</td><td># Questions</td></tr><tr><td>CN</td><td>Train</td><td>377</td><td>108.13</td><td>1853</td><td>22210/21150</td></tr><tr><td>CN</td><td>Test</td><td>30</td><td>6.51</td><td>226</td><td>1582/1399</td></tr><tr><td>FO</td><td>Test</td><td>20</td><td>39.98</td><td>287</td><td>574</td></tr></table>

Annotations for Character Naming (CN). We construct the associations between character names and face tracks in three steps. First, since MovieNet and Movie101 already provide ground-truth name–body associations, we align character body boxes with their face tracks in space and time, yielding name–face associations for all characters appearing on screen. Second, we keep only those associations whose corresponding names are also mentioned in the dialogue, since these are the only ones inferable from the audio–visual signal we use. Third, we handle the case where a name is mentioned in the dialogue but the corresponding face track is unavailable (e.g., when the character is referred to in absentia). In this case, we label the association as none, since no in-clip cues support the assignment. As a result, we provide the final name–face associations as ground-truth labels, along with all character-level face tracks in the video clip as candidate options, to construct a multiple-choice instance.

Annotations for First Occurrence (FO). For every character in the movie, we manually annotate two timestamps: thefirst visual occurrence, i.e. the moment the character first appears on screen, and thefirst naming, i.e. the moment at which a human viewer can first associate the character’s visual appearance with their name based solely on the preceding content.

Quality Control. To ensure a high-quality test set, we manually inspect every test sample and correct errors in face-track clustering and name labelling. We then run state-of-the-art proprietary models on the curated set and examine cases where their outputs disagree with the annotation, fixing any remaining errors (or confirming the annotation is correct but the model fails).

## 2.2 Dataset Statistics

The statistics of both annotation types in the MCN dataset are summarised in Tab. 1. For the CN task, the training splits comprise clips from 377 training movies (108.13 hours), covering 1,853 characters and 22,210 name-to-track questions, of which 21,150 have non-none answers. The training movies are disjoint from those in both the CN and FO test sets, ensuring no movie-level leakage across splits. The CN test split contains clips from 30 movies (6.51 hours), covering 226 characters and 1,582 manually verified questions, of which 1,399 have non-none answers. In particular, the CN questions in the test set can be broken down by evidence type: approximately 12% involve self-introduction, 70% involve direct address by other characters, 16% involve third-party reference, and 2% require reasoning from textual cues in the movie, such as name badges.

The FO annotations form a test set of 20 movies (39.98 hours) with 287 annotated characters, yielding 574 timestamp annotations in total. Each character has two annotations, corresponding to the first visual occurrence and the first naming. Together, the MCN dataset spans a diverse range of genres, casts, and narrative styles, ensuring that our evaluation captures performance across varied cinematic settings.

## 2.3 Preprocessing – Character-Level Face-Tracks and Associated Speech

To obtain multiple-choice instances for Character Naming, we need to extract character-level tracks and speech from video clips around name occurrences in the dialogue, therefore enabling us to ask models to select one of them for each query name. This involves extracting a face-track cluster $f _ { i }$ for each character and linking it with the corresponding voice-track cluster $v _ { j }$ . Note, this pre-processing is done at the clip level when annotating the dataset, but results are also made available over the entire movie for downstream tasks (e.g., character bank construction).

Face Track Extraction. For each recurring character i, we detect and track faces across shots and group the tracks into a character-level face-track cluster $f _ { i }$ . The detailed procedure is given in Appendix A.1.

Linking Voices to Face Tracks. For the clean voice segments in the video, we first apply speaker diarization to group them into voice clusters, with each cluster denoted $v _ { j }$ . To match each voice cluster to an existing face track (i.e. to identify who is speaking on screen), we apply TalkNet [57] to obtain synchronization scores $s _ { i , j }$ between faces $f _ { i }$ and voices $v _ { j }$ . Each voice cluster is then assigned to the face-track cluster with the highest synchronization score, yielding complete character identities with paired faces and voices.

As a further preprocessing step, we transcribe the voice tracks into textual dialogue with Parakeet [47]. Since each voice track is linked to a face cluster, this naturally yields visually grounded subtitles. Finally, the character names n emerge as a natural by-product of subtitle extraction. We retain clips whose subtitles contain at least one character name, passing them to the subsequent character naming step.

## 3 Assigning Names to Characters

The task of assigning names to characters, i.e., Character Naming (CN), is performed naturally by humans while watching a film, but is not trivial for deep learning systems owing to multiple challenges: first, a character may be addressed directly by name, or referred to in the third person while off-screen, or revealed only through what others say about them; second, who is being addressed may only be indicated by gaze or by the editing structure, for example, as one character says a name, and the named character then reacts in the subsequent shot. We therefore treat CN as a new multimodal task that requires joint reasoning over textual cues (e.g., conversational logic, vocative address) and visual cues (e.g., eye contact, on-screen presence).

Formally, given a video clip $V$ containing a set of face-track clusters $\{ f _ { i } \}$ (each optionally linked to a voice cluster $v _ { j }$ , as described in Sec. 2.3) and a candidate name n extracted from the dialogue, the Character Naming (CN) task is to predict an assignment $\psi ( n ) \in \{ f _ { i } \} \cup \{ \emptyset \}$ }. Here, $\psi ( n ) = f _ { i }$ indicates that the name n refers to the character represented by face-track cluster $f _ { i }$ , while $\psi ( n ) = \emptyset$ indicates that none of the visible characters corresponds to n (e.g., the named character is off-screen).

In this section, we first describe how we operationalize Character Naming for MLLMs through visual prompting alone (Sec. 3.1). We then introduce a fine-tuning recipe that improves the MLLMs’ reasoning on this task, enabling more reliable assignments of character names (Sec. 3.2).

## 3.1 MLLM-Based Character Naming

To adapt MLLMs for character naming in a zero-shot manner, we use the following prompting strategies.

Prompting the MLLM with Visual Identities. Rather than providing the MLLM with bounding-box coordinates of the face tracks, we adopt the color-based visual prompting scheme of Xie et al. [67]. Each face track is assigned a unique color, and its bounding boxes are drawn in that color on the raw video frames. The boxes are then referred to by color in the text prompt, allowing the MLLM to associate identities with their spatial locations.

Prompting the MLLM with Visually Grounded Subtitles. Subtitles are embedded directly into the text prompt, with each speaker referred to by the color tag of their associated face track. We additionally include timestamps to enable more temporally accurate inference. For example, a line spoken by the character with blue face bounding boxes would be formatted as <693.2s> [BLUE] “Any idea who the girl in the black hat is?” <695.0s>.

Prompting the MLLM for Character Naming. Given the visual identities and visually grounded subtitles, we directly ask the MLLM to match each character name from the subtitles with a color, $\mathrm { e . g . }$ , “Which color corresponds to John Constantine? $\lesssim$ The color code thus bridges each visual identity’s spatial position to its corresponding name.

## 3.2 Fine-tuning the MLLM for Character Naming

Following the prompting strategies described above, we adopt a two-stage R1-style fine-tuning recipe [22] to further improve the naming accuracy.

Stage I: Supervised Fine-tuning (SFT). The first stage is a cold start that supervises the MLLM with synthetic reasoning traces. These traces are generated by a teacher MLLM, which is given the video frames, visually grounded subtitles, and ground-truth name-to-color mapping, and asked to produce a reasoning trace that leads to the correct answer. This teaches the student MLLM to follow the desired output format (i.e. a reasoning trace followed by the answer) before the reinforcement learning stage.

Stage II: Reinforcement Learning. The second stage applies GRPO [22] with a composite reward $R = r _ { f } + r _ { a }$ , where $r _ { f }$ rewards adherence to the expected response structure and $r _ { a }$ measures the correctness of the predicted name–face pairs. Letting $\hat { i } ( n )$ and $i ^ { * } ( n )$ denote the face track assigned to name n under the prediction $\hat { \psi }$ and the ground truth $\psi ^ { * }$ , we set $\begin{array} { r } { r _ { a } = \sum _ { n \in N } \mathbf { 1 } [ \hat { i } ( n ) = i ^ { * } ( n ) ] } \end{array}$ . The GRPO advantage and policy update follow the standard formulation [22], summarized in Appendix B.

A naive application of GRPO to this task suffers from a subtle shortcut: the model can latch onto color-specific cues. For instance, it may always associate the protagonist with “red” simply because $\mathbf { \tilde { r e d } } ^ { * }$ is the most frequently used color in the training distribution. We address this with a simple but effective color-permutation augmentation: during RL training, we randomly permute the color assignment for each sample. The ground-truth name–face mapping is invariant under such permutations, so consistent rewards across permuted variants are achievable only by reasoning about identity itself. This regularizes the policy toward color-agnostic reasoning and, empirically, accounts for a significant share of our gains over the cold-start baseline.

## 4 Streaming Pipeline for Character Bank Construction

Given a full movie, our goal is to construct a character bank in an online manner, taking the shot as the elementary unit. The character bank $B _ { t }$ serves as a memory of all characters observed so far. At each shot $t ,$ it aggregates the corresponding character information as triplets $( f , n , v )$ , where $f$ is a face-track cluster, n a character name, and v a voice cluster.

Formally, the bank is updated auto-regressively (as illustrated in Fig. 3):

$$
B _ { t } = \phi _ { B } ( B _ { t - 1 } , \psi ( V _ { t } ) ) ,\tag{1}
$$

where $V _ { t }$ denotes the video clip at shot $t ,$ and ψ is the MLLM fine-tuned for CN, which outputs character names and corresponding face tracks and voices $\{ ( n , f , v ) \}$ . We also retain identities with no assigned name, i.e. $\{ ( \emptyset , f , \bar { v } ) \}$ , mirroring how a human viewer can visually remember a character even before they are introduced by name. These triplets are then merged into the previous bank by the update operator ϕ<sub>B</sub>.

Overall, this streaming pipeline maintains and auto-regressively updates the character bank B. We detail the bank update process in Sec. 4.1 and introduce an error-correction mechanism in Sec. 4.2 that keeps the update robust to error accumulation over multiple steps.

## 4.1 Character Bank Update

Given the names and corresponding face and voice tracks $\{ ( n , f , v ) \}$ extracted from a new shot (with n possibly empty), we merge them into the existing entries of the previous bank in two steps.

Track Assignment. We treat face tracks as the primary signal and assign each $f _ { i }$ to an existing face cluster in the bank. We extract the corresponding face embedding $e _ { i }$ and find the closest cluster by distance to its embedding centroid $\bar { e } _ { b }$ as in Eq. 2. A smaller distance indicates greater similarity between the face embedding $e _ { i }$ and the cluster centroid $\bar { e } _ { b }$

$$
b ^ { * } = \arg \operatorname* { m i n } _ { b \in \mathcal { B } _ { t - 1 } } d _ { \cos } ( e _ { i } , \bar { e } _ { b } ) .\tag{2}
$$

If $d _ { \mathrm { c o s } } ( e _ { i } , \bar { e } _ { b ^ { \ast } } ) < \tau _ { f }$ , the track is assigned to entry $b ^ { * }$ . Otherwise, we fall back to voice as the secondary signal: if the shot’s speaker embedding $u _ { i }$ matches an entry’s voice centroid $\bar { u } _ { b }$ under a tighter distance threshold $\tau _ { v } < \tau _ { f }$ Ida N/A Stevie, the track is assigned there. If neither threshold is satisfied, a new entry is instantiated Bank Bt−1in the bank. Upon assignment, the matched centroids are updated.

![](images/46120065a368b42d1f76982b60bc52dfa8e5d66472cddae8689d754f970e334a.jpg)  
Figure 3: Streaming pipeline for character bank construction. As new shots arrive, we extract observations (character tracks, subtitles, and mentioned names) and reason jointly over them to link <sub>think,</sub> <sub>Stevie?</sub>face tracks to names. Each step incrementally updates the bank through three operations: naming an existing cluster (Stevie, first cluster), enriching a cluster with additional face exemplars (second <sup>Tell</sup> <sup>me</sup> <sup>what's</sup> <sup>going</sup> <sup>to</sup> I think it's ours for the taking. cluster), and instantiating a new entry for a previously unseen character (fourth cluster). The sample is drawn from The Ides ofMarch (2011).

Name Binding. We then bind a name to each entry in the character bank. For each entry, we maintain a histogram over the names associated with its face tracks across all observed shots, and use the 1 (Name: Unknown → Stevie)majority-voted name as its canonical name. This handles aliases and nicknames; for example, in Juno, What do you the character Juno may be addressed as Mama Bear, Home Skillet, or General MacGuff. Majority <sup>think,</sup> <sup>Stevie?</sup>voting helps recover the most representative name for each character. It also improves robustness to noisy predictions from the character naming module, since occasional erroneous name assignments <sub>happen</sub> <sub>on</sub> <sub>the</sub> <sub>15th?</sub>are unlikely to dominate the histogram.

Retrospective Assignment is a special case of name binding: when the CN module assigns a name <sup>Character</sup> <sup>Naming</sup> <sup>→</sup> <sup>Character</sup> <sup>Notebook</sup> <sup>Nt</sup> ϕto a previously unnamed face-track cluster, the new name is added to that cluster’s histogram with a count of one, and the canonical name is updated accordingly. Thus, clusters that are initially unnamed Bank B<sub>t−1</sub> Bank B<sub>t</sub>can still acquire names once sufficient naming evidence becomes available in later shots.

## 4.2 Error Correction during Character Bank Construction

The Markov structure of the update process $\phi _ { B }$ makes it susceptible to error accumulation from upstream components (e.g. transcription, active-speaker detection, and identity grounding). To mitigate this, we introduce three error correction mechanisms deployed along the streaming chain. First, we use an MLLM to filter out noisy or ambiguous face tracks before clustering, ensuring high-quality embeddings. Second, we use an LLM to filter out query names from the subtitles that are unlikely to refer to on-screen characters. Third, we resolve cross-entry name conflicts after each shot using a greedy voting scheme that enforces one-to-one name assignment. Together, these mechanisms address distinct error modes and stabilize the evolution of $B _ { t }$ without revisiting past observations. We refer the reader to Appendix C for further details.

## 5 Experiments

This section presents our evaluation. We first introduce the metrics (Sec. 5.1), then evaluate each task. Sec. 5.2 evaluates our character naming (CN) module on the MCN test set against state-of-the-art openand closed-source models. Sec. 5.3 evaluates the full streaming character bank pipeline, measuring how well our method tracks each character’s first visual appearance and the first moment it can be linked to a name, and compares the final character bank against the IMDB ground truth. We ablate key design choices, including the two-stage fine-tuning strategy and data augmentation for CN, and the error correction mechanisms for the character bank. Implementation details and ablations are provided in Appendix D and Appendix E.

## 5.1 Evaluation Metrics

Character Naming Accuracy. We evaluate for two settings. Overall accuracy is computed over all name queries, including those labelled none when the mentioned character cannot be linked to any on-screen face track. Known accuracy is computed only over queries whose ground-truth label is a valid face track, thereby examining the model’s ability to correctly ground on-screen characters.

Table 2: Character naming accuracy on the MCN test set. We report overall accuracy (including none labels) and known accuracy (restricted to queries with a valid face track). Our fine-tuned Qwen3VL-4B-Thinking closes most of the gap to proprietary frontier models despite being substantially smaller in terms of model size.
<table><tr><td>Model</td><td>Propriet.</td><td>Fine-Tuned</td><td>Overall Acc.</td><td>Known Acc.</td></tr><tr><td>Random Baseline</td><td>X</td><td>X</td><td>29.90</td><td>33.74</td></tr><tr><td>Gemini-3-Flash-Preview</td><td>√</td><td>X</td><td>79.57</td><td>86.62</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>√</td><td>X</td><td>82.47</td><td>89.69</td></tr><tr><td>Qwen3VL-4B-Thinking</td><td>×</td><td>X</td><td>52.15</td><td>56.68</td></tr><tr><td>Qwen3VL-8B-Instruct</td><td>×</td><td>X</td><td>53.60</td><td>58.83</td></tr><tr><td>Qwen3.5-9B</td><td>X</td><td>X</td><td>61.57</td><td>67.69</td></tr><tr><td>Qwen3VL-8B-Instruct (FT)</td><td>X</td><td>√</td><td>64.92</td><td>71.77</td></tr><tr><td> $\mathrm { Q w e n 3 V L - 4 B \mathrm { - T h i n k i n g } \left( S F T + R L \right) }$ </td><td>X</td><td>√</td><td>72.76</td><td>81.06</td></tr></table>

Character Bank Mean Temporal IoU. To evaluate the character bank streaming pipeline, we propose a Mean Temporal IoU (mTIoU) metric that measures how closely the bank tracks ground-truth human knowledge of each character over time.

For a movie of duration T and character $c ,$ let $H _ { c } \subseteq [ 0 , T ]$ denote the interval during which a human viewer is aware of c, and $P _ { c } \subseteq [ 0 \mathcal { T } ]$ the interval during which the model correctly identifies c in the bank. We define $\mathrm { T I o U } ( c ) = | H _ { c } \cap P _ { c } ^ { * } | / | \dot { H } _ { c } \cup P _ { c } | ,$ , which penalizes both late identification and incorrect identification (e.g. an entry later overwritten with a wrong name). To reflect narrative importance, we weight each character by their subtitle mention frequency $w _ { c }$ and report mTI $\textstyle { \mathrm { { J } } } = \sum _ { c } w _ { c } ^ { \bullet } { \mathrm { { T I o U } } } ( c ) / \sum _ { c } w _ { c } ^ { \smile }$

We instantiate two variants from the MCN timestamps: the visual variant m $\mathrm { T I o U } _ { v }$ uses $H _ { c } = [ t _ { c } ^ { v } , T ]$ with $t _ { c } ^ { v }$ the first visual occurrence of c, and the name variant m $\mathrm { T I o U } _ { n }$ uses $H _ { c } = [ t _ { c } ^ { n } , T ]$ with $\hat { t } _ { c } ^ { n }$ the first naming timestamp (when a human can first link a face to a name).

Principal Character Recall (PCR@K). To evaluate the quality of the final character bank, we propose Principal Character Recall at K (PCR@K), measuring how reliably the pipeline recovers a movie’s most important characters with both correct names and correct visual exemplars.

For each movie, we take the top-K principals from IMDB, ranked by their narrative importance. A ground-truth principal c is recovered iff some single entry $b \in B$ satisfies both of the following conditions: (i) the canonical name of b phonetically matches the IMDB name of $c ,$ and (ii) the distance between its face centroid $\bar { e } _ { b }$ and the IMDB reference embedding $\bar { e } _ { c } ^ { * }$ is below the threshold $\tau _ { \mathrm { e v a l } }$ . We report per-movie recall as the fraction of recovered principals and average across movies.

## 5.2 Results on Character Naming

We evaluate Character Naming (CN) with three sets of methods. First, we compare against several baselines, including a random baseline that uniformly selects from the candidate colors, non-proprietary open-source MLLMs (Qwen3VL-4B-Thinking, Qwen3VL-8B-Instruct, and Qwen3.5-9B [2]), and a naive fine-tuning baseline obtained by directly fine-tuning Qwen3VL-8B-Instruct on the MCN training set. Second, we report the result of our CN model, which is obtained by two-stage R1-style fine-tuning of Qwen3VL-4B-Thinking on the MCN training set. Third, we compare against state-of-the-art proprietary MLLMs: Gemini-3-Flash-Preview and Gemini-3.1-Pro-Preview [19]. All models receive the same inputs with a visually prompted video clip, visually grounded subtitles, and queried character names, and are asked to output the corresponding box colors.

Results. We report Character Naming results in Tab. 2. The pretrained Qwen3VL-8B-Instruct achieves 53.60% overall accuracy, improving to 64.92% via naive fine-tuning. In contrast, our two-stage SFT+GRPO approach with color-permutation augmentation, applied to a smaller 4B-Thinking backbone, yields a +7.84% gain over the naive 8B baseline. Despite the smaller model size, our approach closes most of the gap to Gemini-3.1-Pro-Preview and clearly outperforms all open-source alternatives. Qualitative examples are shown in Fig. 4.

Discussion. Compared with open-source models such as Qwen3.5-9B, our two-stage fine-tuned model more accurately names characters in direct address scenarios (e.g. shot–reverse-shot, telephone conversations), better exploits visual cues (e.g. pointing, handshakes), and improves temporal grounding between subtitles and frames. A qualitative comparison is provided in Appendix I.

![](images/6ea13ec88a1cc6afecc21213ee846773e3180b5eec13e25fcf97305168dde409.jpg)  
Figure 4: Qualitative results for Character Naming. Examples are from Four Weddings and a Funeral and Jerry Maguire. In each case, our default “Qwen3VL-4B-Thinking (SFT + RL)” model reasons jointly over subtitles and video frames to link a queried name to the correct face — in the first, by following a verbal introduction to a later podium shot; in the second, by resolving direct address across a shot–reverse-shot exchange.

Table 3: Character naming accuracy on the MCN test subset from unseen movies. We report the accuracy results on a test subset derived from unseen movies (released after the models were trained) by all evaluated models to isolate the effects of knowledge leakage. Overall accuracy includes none labels, and known accuracy is restricted to queries with a valid face track.
<table><tr><td>Model</td><td>Propriet.</td><td>Fine-Tuned</td><td>Overall Acc.</td><td>Known Acc.</td></tr><tr><td>Random Baseline</td><td>X</td><td>X</td><td>18.42</td><td>20.93</td></tr><tr><td>Gemini-3-Flash-Preview</td><td>√</td><td>X</td><td>78.62</td><td>81.34</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>√</td><td>X</td><td>80.50</td><td>78.36</td></tr><tr><td>Qwen3VL-8B-Instruct</td><td>X</td><td>X</td><td>45.28</td><td>45.52</td></tr><tr><td>Qwen3.5-9B</td><td>X</td><td>X</td><td>50.94</td><td>54.48</td></tr><tr><td>Qwen3VL-4B-Thinking (SFT + RL)</td><td>X</td><td>√</td><td>64.78</td><td>71.64</td></tr></table>

Nevertheless, a gap remains to the strongest proprietary model, likely driven by errors in transcription and active-speaker detection. We hypothesise that models such as Gemini [19] can partially recover from these errors by leveraging lip motion, whereas our model is more sensitive to such upstream noise.

There are also cases where both Gemini [19] and our model fail. In particular, both struggle with (i) subtle cues (e.g. brief pointing among multiple characters), (ii) complex scenes with motion, and (iii) over-predicting on-screen characters when the ground-truth character is absent. We refer readers to Appendix I for further analysis of these failure modes.

Evaluation on Unseen Movies. We also observe that proprietary models and pre-trained open-source models might have seen the movies that we test on. Therefore, they could infer the character identities without using cues from the clips but from their previous knowledge. This introduces a risk of data contamination, potentially leading to an overestimation of the models’ actual capabilities on our benchmark. To address this concern, we collect 102 manually annotated clips from four films released from March 2026 onward (Voicemailsfor Isabelle, 19 June 2026; In the Grey, 15 May 2026; The Death of Robin Hood, 19 June 2026; Heartstopper Forever, 17 July 2026), after the reported training cut-offs of both Gemini-3.1-Pro (February 2026) and Qwen3-VL (September 2025), yielding 159 character-naming questions, 25 of which have none as the answer. Neither model can therefore have encountered these films during pre-training.

In Tab. 3, the relative ordering of methods from Tab. 2 is unchanged on both metrics. Our fine-tuned 4B model again outperforms all open-source baselines by a wide margin and trails only the proprietary Gemini models. Accuracy drops for all methods by a margin comparable to that of the random baseline, reflecting the greater difficulty of these clips, which frequently feature fast-moving action and dense background crowds, compared to the MCN test set used in Tab. 2. Our model retains most of its original accuracy, confirming that it relies on in-clip reasoning rather than prior familiarity with the films.

Table 4: Results on the MCN First Occurrence dataset. We report mean Temporal IoU (mTIoU) against two reference intervals, first visual occurrence and first naming, and Principal Character Recall (PCR@K) on the final character bank for $K \in \{ 5 , 1 0 , 1 5 \}$ . Our method operates in a streaming setting, whereas proprietary baselines are given full-movie access.
<table><tr><td>Method</td><td>Streaming</td><td> $m T I o U _ { v }$ </td><td> $m T I o U _ { n }$ </td><td>PCR@5</td><td>PCR@10</td><td>PCR@15</td></tr><tr><td>Gemini-3-Flash-Preview</td><td>X</td><td>51.81</td><td>30.92</td><td>35.00</td><td>32.04</td><td>29.84</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>X</td><td>49.05</td><td>30.10</td><td>44.00</td><td>37.15</td><td>34.93</td></tr><tr><td>Ours</td><td>√</td><td>84.90</td><td>54.40</td><td>63.00</td><td>54.54</td><td>48.19</td></tr></table>

## 5.3 Results on Streaming Character Bank Construction

Baseline. For streaming character bank construction, we use Gemini-3-Flash-Preview and Gemini-3.1-Pro-Preview [19] as baselines, prompting them to output each character’s name, first visual occurrence timestamp, and first naming timestamp. To obtain face clusters, we ask the models to identify timestamps with close-up or clear solo shots, from which we perform face detection, extract features, and cluster faces, selecting the largest connected component as the predicted cluster. As these models are not designed to directly output face crops or clusters, we consider a prediction correct if any face in the cluster matches the ground-truth identity.

Results. As shown in Tab. 4, our pipeline substantially outperforms both proprietary baselines across all metrics, despite operating in the more challenging streaming setting. These improvements reflect more accurate identification of when characters first appear and are named (mTIoU), as well as stronger alignment of the final character bank with canonical identities from external sources (PCR). These gains are especially notable given that the baselines have full-movie access while ours processes the video in a single streaming pass. Appendix J shows two examples of the final constructed character bank.

Discussion. Our pipeline performs well on movies such as Juno and Meet the Parents, which feature relatively simple scenes with few participants and a limited set of recurring characters. In these cases, identity clustering is generally reliable, aided by our face-track filtering. Even when Character Naming errors occur, our method can often recover as additional evidence emerges over time.

Performance degrades, however, in movies with many scenes containing numerous minor or background characters. In Harry Potter and the Goblet ofFire, for example, background characters introduce noise into clustering, which distracts the Character Naming process. Movies with very large named casts, such as Four Weddings and a Funeral, with over 50 named characters among hundreds of recurring faces, are also particularly challenging, as the model must maintain and disambiguate many identities over long temporal contexts.

## 6 Related Work

Character-Centric Movie Understanding. Movie understanding has emerged as a milestone task for video understanding, owing to its long temporal span and rich multimodal narrative structure, motivating a variety of datasets [3, 28, 55, 59, 61]. A representative line of research is Audio Description (AD) generation [21, 24–26, 31, 48, 67, 68], where Han et al. [25] identifies character recognition as central to the task. Existing work typically takes one of two shortcuts: relying on curated sources like IMDB cast lists [24–26, 43, 67, 68] or fan wikis [21], which are often incomplete or unavailable for new films, or sidestepping naming entirely with placeholders like “Person A” [18, 56, 75]. A separate line of work links names to faces directly from video. Early approaches align shooting scripts with subtitles for weak labels and use active-speaker detection to associate them with face tracks [15, 54]. Extensions add prob abilistic models over face, speech, and clothing cues [58], coreference resolution [45], and script-free dialogue-based naming for recurring TV casts [12, 27, 40]. A complementary line assumes identities are known and analyses interactions via gaze and eyelines [20, 36, 37, 46]. We revisit this problem in the era of MLLMs. Prior dialogue-based approaches, including Haurilet et al. [27], cast naming as an optimisation over many mentions across clips or episodes of a recurring TV cast; we instead resolve identity from the single clip surrounding the name as a human viewer does, generalising to feature-length movies in the wild where names are typically mined from one-off dialogue. In concurrent work, Pu et al. [44] track identities via textual appearance descriptions, a lossy representation that fails to distinguish visually similar characters and degrades under costume or lighting changes. Our character bank instead grounds each identity in visual exemplars, retaining the fine-grained cues text abstracts away.

Visual Reasoning in Vision-Language Models (MLLMs). Recent work extends reasoning from Large Language Models (LLMs) [32, 65] to the multimodal setting. Early approaches prompt frozen MLLMs to produce intermediate reasoning steps [39, 62], while more recent methods train explicit reasoning via reinforcement learning. Inspired by [22], Group Relative Policy Optimization (GRPO) has been adapted to MLLMs, showing that RL with verifiable rewards improves reasoning ability [29, 38, 51, 74], including across perception tasks such as detection [35], segmentation [34], and fine-grained perception [72]. For video understanding specifically, prior work incorporates temporal reasoning into this paradigm, e.g. via temporal-aware GRPO [16], joint spatio-temporal reasoning [33], and long-form temporal grounding [63]. In parallel, visual prompting introduces structured cues, such as colored masks [69], boxes [6, 53], region tokens [10, 71], or identity markers [67], to link visual entities with language. Our approach builds on both lines of work, combining color-based visual prompting with a two-stage SFT + GRPO fine-tuning strategy to enable an MLLM to reason about character identity from visual and dialogue cues jointly.

Person Clustering in Long Videos. Grouping person appearances by identity is a long-standing problem in video understanding, with applications to cast listing, character-based browsing, and story understanding [11, 15, 17, 60]. Early work focuses on face clustering, using metric or representation learning to group face tracks under variations in pose, lighting, and expression [50, 52, 64, 70]. These methods exploit spatio-temporal structure in edited videos via co-occurrence constraints [4, 30, 66], or apply nearest-neighbor clustering over pretrained embeddings [42, 49]. However, they rely on visible faces, limiting performance when characters are occluded, off-screen, or viewed from behind. To address this, Brown et al. [5] introduce Video Person-Clustering, grouping face, body, and voice cues via multimodal hierarchical clustering, with later work extending this direction using graph-based formulations [41]. Our pipeline follows a similar multimodal approach, using face embeddings as the primary signal with voice as fallback. Unlike prior work, we operate in a streaming setting, where clusters are instantiated, updated, and named as the film unfolds, extending online clustering [1, 7] to feature-length videos.

## 7 Discussion

In this paper, we introduce the task of Character Naming from dialogue and close the gap between open-source models and the best proprietary models. Building on this task, we further introduce a streaming pipeline to construct a character bank, enabling online, human-like movie understanding. For both tasks, we compare our method with Gemini [19], though neither task is fully solved, even by proprietary models. A natural next step is to develop models or an agentic framework to overcome the current failure modes, and thereby improve performance.

## Acknowledgements

This research is supported by UK EPSRC Programme Grant VisualAI (EP/T028572/1), a Royal Society Research Professorship RSRP\R\241003, and Clarendon Scholarships.

## References

[1] Charu C Aggarwal, S Yu Philip, Jiawei Han, and Jianyong Wang. A framework for clustering evolving data streams. In Proceedings 2003 VLDB conference, pages 81–92. Elsevier, 2003.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[3] Max Bain, Arsha Nagrani, Andrew Brown, and Andrew Zisserman. Condensed movies: Story based retrieval with contextual embeddings. In Proceedings ofthe Asian Conference on Computer Vision, 2020.

[4] Martin Bauml, Makarand Tapaswi, and Rainer Stiefelhagen. Semi-supervised learning with constraints for person identification in multimedia data. In Proceedings ofthe IEEE conference on computer vision andpattern recognition, pages 3602–3609, 2013.

[5] Andrew Brown, Vicky Kalogeiton, and Andrew Zisserman. Face, body, voice: Video person-clustering with multiple modalities. In proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3184–3194, 2021.

[6] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Vip-llava: Making large multimodal models understand arbitrary visual prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12914–12923, 2024.

[7] Feng Cao, Martin Estert, Weining Qian, and Aoying Zhou. Density-based clustering over an evolving data stream with noise. In Proceedings of the 2006 SIAM international conference on data mining, pages 328–339. SIAM, 2006.

[8] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

[9] Brandon Castellano. PySceneDetect: Video scene cut detection and analysis tool. https: //github.com/Breakthrough/PySceneDetect, 2024. Accessed: 2026-04-19.

[10] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.

[11] Ramazan Gokberk Cinbis, Jakob Verbeek, and Cordelia Schmid. Unsupervised metric learning for face identification in tv video. In 2011 International Conference on Computer Vision, pages 1559–1566. IEEE, 2011.

[12] Timothee Cour, Benjamin Sapp, Akash Nagle, and Ben Taskar. Talking pictures: Temporal grouping and dialog-supervised person recognition. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 1014–1021. IEEE, 2010.

[13] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multi-level face localisation in the wild. In CVPR, 2020.

[14] Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck. Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification. arXiv preprint arXiv:2005.07143, 2020.

[15] Mark Everingham, Josef Sivic, and Andrew Zisserman. Hello! my name is... buffy”–automatic naming of characters in tv video. In BMVC, volume 2, page 6, 2006.

[16] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.

[17] Andrew Fitzgibbon and Andrew Zisserman. On affine invariant clustering and automatic cast listing in movies. In European Conference on Computer Vision, pages 304–320. Springer, 2002.

[18] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision andpattern recognition, pages 24108–24118, 2025.

[19] Google DeepMind. Gemini 3 Pro model card. Technical report, Google, December 2025. URL https:// storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf.

[20] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, et al. Ava: A video dataset of spatio-temporally localized atomic visual actions. In Proceedings of the IEEE conference on computer vision andpattern recognition, pages 6047–6056, 2018.

[21] Zhongrui Gui, Junyu Xie, Tengda Han, Weidi Xie, and Andrew Zisserman. Character-centric understanding of animated movies. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 3300–3309, 2025.

[22] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[23] Jiangyu Han, Petr Pálka, Marc Delcroix, Federico Landini, Johan Rohdin, Jan Cernocky, and Lukáš Burget.\` Efficient and generalizable speaker diarization via structured pruning of self-supervised models. arXiv preprint arXiv:2506.18623, 2025.

[24] Tengda Han, Max Bain, Arsha Nagrani, Gül Varol, Weidi Xie, and Andrew Zisserman. Autoad: Movie description in context. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18930–18940, 2023.

[25] Tengda Han, Max Bain, Arsha Nagrani, Gul Varol, Weidi Xie, and Andrew Zisserman. Autoad ii: The sequel-who, when, and what in movie audio description. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13645–13655, 2023.

[26] Tengda Han, Max Bain, Arsha Nagrani, Gül Varol, Weidi Xie, and Andrew Zisserman. Autoad iii: The prequel-back to the pixels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18164–18174, 2024.

[27] Monica-Laura Haurilet, Makarand Tapaswi, Ziad Al-Halah, and Rainer Stiefelhagen. Naming tv characters by watching and analyzing dialogs. In 2016 IEEE winter conference on applications of computer vision (WACV), pages 1–9. IEEE, 2016.

[28] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In European conference on computer vision, pages 709–727. Springer, 2020.

[29] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Xu Tang, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

[30] Vicky Kalogeiton and Andrew Zisserman. Constrained video face clustering using 1nn relations. https://ora.ox.ac.uk/objects/uuid:23f369b3-794b-42a6-a6af-ab4b0d726ba4/files/spn89d688d, 2020.

[31] Eshika Khandelwal, Junyu Xie, Tengda Han, Max Bain, Arsha Nagrani, Andrew Zisserman, Gül Varol, and Makarand Tapaswi. More than a moment: Towards coherent sequences of audio descriptions. arXiv preprint arXiv:2510.25440, 2025.

[32] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

[33] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025.

[34] Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Seg-zero: Reasoning-chain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520, 2025.

[35] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 2034–2044, 2025.

[36] Manuel J Marin-Jimenez, Vicky Kalogeiton, Pablo Medina-Suarez, and Andrew Zisserman. Laeo-net: revisiting people looking at each other in videos. In Proceedings ofthe IEEE/CVF conference on computer vision andpattern recognition, pages 3477–3485, 2019.

[37] Manuel Jesús Marin-Jimenez, Andrew Zisserman, Marcin Eichner, and Vittorio Ferrari. Detecting people looking at each other in videos. International Journal of Computer Vision, 106(3):282–296, 2014.

[38] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

[39] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-of-thought prompting for large multimodal models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14420–14431, 2024.

[40] Arsha Nagrani and Andrew Zisserman. From benedict cumberbatch to sherlock holmes: Character identification in tv series without a script. arXiv preprint arXiv:1801.10442, 2018.

[41] Arsha Nagrani, Samuel Albanie, and Andrew Zisserman. Learnable pins: Cross-modal embeddings for person identity. In Proceedings ofthe European conference on computer vision (ECCV), pages 71–88, 2018.

[42] Charles Otto, Dayong Wang, and Anil K Jain. Clustering millions of faces by identity. IEEE transactions on pattern analysis and machine intelligence, 40(2):289–303, 2017.

[43] Stefano Pini, Marcella Cornia, Federico Bolelli, Lorenzo Baraldi, and Rita Cucchiara. M-vad names: a dataset for video captioning with naming. Multimedia Tools and Applications, 78(10):14007–14027, 2019.

[44] Junfu Pu, Yuxin Chen, Teng Wang, and Ying Shan. Omniscript: Towards audio-visual script generation for long-form cinematic video. arXiv preprint arXiv:2604.11102, 2026.

[45] Vignesh Ramanathan, Armand Joulin, Percy Liang, and Li Fei-Fei. Linking people in videos with “their” names using coreference resolution. In European conference on computer vision, pages 95–110. Springer, 2014.

[46] Adria Recasens, Carl Vondrick, Aditya Khosla, and Antonio Torralba. Following gaze in video. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 1435–1443, 2017.

[47] Dima Rekesh, Nithin Rao Koluguri, Samuel Kriman, Somshubra Majumdar, Vahid Noroozi, He Huang, Oleksii Hrinchuk, Krishna Puvvada, Ankur Kumar, Jagadeesh Balam, et al. Fast conformer with linearly scalable attention for efficient speech recognition. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8. IEEE, 2023.

[48] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In Proceedings ofthe IEEE conference on computer vision andpattern recognition, pages 3202–3212, 2015.

[49] Saquib Sarfraz, Vivek Sharma, and Rainer Stiefelhagen. Efficient parameter-free clustering using first neighbor relations. In Proceedings ofthe IEEE/CVF conference on computer vision andpattern recognition, pages 8934–8943, 2019.

[50] Vivek Sharma, Makarand Tapaswi, M Saquib Sarfraz, and Rainer Stiefelhagen. Self-supervised learning of face representations for video face clustering. In 2019 14th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2019), pages 1–8. IEEE, 2019.

[51] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

[52] Shuai Shen, Wanhua Li, Zheng Zhu, Guan Huang, Dalong Du, Jiwen Lu, and Jie Zhou. Structure-aware face clustering on a large-scale graph with 107 nodes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9085–9094, 2021.

[53] Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. What does clip know about a red circle? visual prompt engineering for vlms. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 11987–11997, 2023.

[54] Josef Sivic, Mark Everingham, and Andrew Zisserman. “who are you?”-learning person specific classifiers from video. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 1145–1152. IEEE, 2009.

[55] Mattia Soldan, Alejandro Pardo, Juan León Alcázar, Fabian Caba, Chen Zhao, Silvio Giancola, and Bernard Ghanem. Mad: A scalable dataset for language grounding in videos from movie audio descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5026–5035, 2022.

[56] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024.

[57] Ruijie Tao, Zexu Pan, Rohan Kumar Das, Xinyuan Qian, Mike Zheng Shou, and Haizhou Li. Is someone speaking? exploring long-term temporal features for audio-visual active speaker detection. In Proceedings of the 29th ACM international conference on multimedia, pages 3927–3935, 2021.

[58] Makarand Tapaswi, Martin Bäuml, and Rainer Stiefelhagen. “knock! knock! who is it?” probabilistic person identification in tv-series. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 2658–2665. IEEE, 2012.

[59] Makarand Tapaswi, Yukun Zhu, Rainer Stiefelhagen, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. Movieqa: Understanding stories in movies through question-answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4631–4640, 2016.

[60] Makarand Tapaswi, Marc T Law, and Sanja Fidler. Video face clustering with unknown number of clusters. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 5027–5036, 2019.

[61] Paul Vicol, Makarand Tapaswi, Lluis Castrejon, and Sanja Fidler. Moviegraphs: Towards understanding human-centric situations from videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8581–8590, 2018.

[62] Yaoting Wang, Shengqiong Wu, Yuecheng Zhang, Shuicheng Yan, Ziwei Liu, Jiebo Luo, and Hao Fei. Multimodal chain-of-thought reasoning: A comprehensive survey. arXiv preprint arXiv:2503.12605, 2025.

[63] Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025.

[64] Zhongdao Wang, Liang Zheng, Yali Li, and Shengjin Wang. Linkage based face clustering via graph convolution network. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1117–1125, 2019.

[65] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[66] Baoyuan Wu, Siwei Lyu, Bao-Gang Hu, and Qiang Ji. Simultaneous clustering and tracklet linking for multi-face tracking in videos. In Proceedings of the IEEE international conference on computer vision, pages 2856–2863, 2013.

[67] Junyu Xie, Tengda Han, Max Bain, Arsha Nagrani, Gül Varol, Weidi Xie, and Andrew Zisserman. Autoad-zero: A training-free framework for zero-shot audio description. In Proceedings of the Asian Conference on Computer Vision, pages 2265–2281, 2024.

[68] Junyu Xie, Tengda Han, Max Bain, Arsha Nagrani, Eshika Khandelwal, Gül Varol, Weidi Xie, and Andrew Zisserman. Shot-by-shot: Film-grammar-aware training-free audio description generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16503–16513, 2025.

[69] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023.

[70] Lei Yang, Xiaohang Zhan, Dapeng Chen, Junjie Yan, Chen Change Loy, and Dahua Lin. Learning to cluster faces on an affinity graph. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2298–2306, 2019.

[71] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704, 2023.

[72] En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954, 2025.

[73] Zihao Yue, Qi Zhang, Anwen Hu, Liang Zhang, Ziheng Wang, and Qin Jin. Movie101: A new movie understanding benchmark. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4669–4684, 2023.

[74] Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1-zero’s" aha moment" in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025.

[75] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, et al. Mlvu: Benchmarking multi-task long video understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13691–13701, 2025.

## Appendix

This Appendix contains the following sections:

• In Appendix A, we elaborate on the preprocessing stage, covering face track extraction and clustering as well as the linking of voice clusters to face tracks via active-speaker synchronization.

• In Appendix B, we present the full GRPO formulation used to fine-tune our character-naming MLLM, including the composite reward, advantage normalization, and policy update.

• In Appendix C, we provide detailed descriptions of the error correction mechanism during streaming character bank construction.

• In Appendix D, we specify implementation details and hyperparameter settings for clustering, fine-tuning, and streaming character bank construction.

• In Appendix E, we conduct ablation studies on the fine-tuning strategy, color-permutation augmentation, and error-correction mechanisms to verify key design choices.

• In Appendix F, we introduce an additional baseline for streaming character bank construction that natively produces face exemplars, and is therefore better matched to our output format than the MLLM baselines.

• In Appendix G, we provide additional details on the MCN dataset collection, including the coarse-to-fine alignment pipeline used to recover MovieNet annotations at native resolution.

• In Appendix H, we present representative dataset examples illustrating the diversity of films and naming cues covered by the MCN benchmark.

• In Appendices I and J, we provide additional qualitative results for character naming and streaming character bank construction, including comparisons with baselines and failure cases.

• In Appendix K, we describe the compute resources used for the experiments, including MLLM fine-tuning and inference.

• In Appendix L, we discuss the broader impacts of our work, including potential risks and the scope of the released artefacts.

• We also include a supplementary video that demonstrates our character naming results and visualizes the streaming character bank construction process on the movie Juno.

## A Details on Preprocessing

This section provides additional details on the preprocessing stage of Sec. 2.3. We first describe how face tracks are detected, extracted, and merged into face clusters (Sec. A.1), and then explain how voice clusters are linked to these face clusters via active-speaker synchronization (Sec. A.2).

## A.1 Face Track Extraction and Clustering

To obtain complete and accurate face tracks, we detect shot boundaries using SceneDetect [9] and query SAM 3 [8] with the text prompt “face” at the beginning, middle, and end of each shot to handle occlusions. We then extract face features with InsightFace [13] by sampling 15 frames per track. Following Brown et al. [5], we cluster the tracks into face clusters $\left\{ f _ { i } \right\}$ under two constraints: (1) a spatio-temporal cannot-link constraint, preventing tracks with (partial) temporal overlap from being grouped, since co-occurring tracks in the same frame must correspond to distinct characters; and (2) a nearest-neighbor distance constraint, which requires the distance between the features of merged tracks to fall below a threshold τ<sub>cluster</sub>.

## A.2 Linking Voices to Face Tracks

We transcribe movie dialogue using Parakeet [47] and group voice segments into voice clusters $\{ v _ { j } \}$ via speaker diarization [23]. To match each voice cluster to a face cluster, we apply TalkNet [57] to compute a per-frame active-speaker synchronization score $s _ { k , t } ,$ indicating how likely face track k is speaking at timestamp t. We rely on diarization because TalkNet, which is predominantly trained on frontal faces, fails when speakers are off-screen or not facing the camera.

For each voice cluster $v _ { j }$ with subtitle lines $\{ e \}$ (each e has start and end times $t _ { e } ^ { \mathrm { s } }$ and $t _ { e } ^ { \mathrm { e } } )$ , we aggregate the per-frame scores at the cluster level to obtain an association score $S _ { i , j }$ between face cluster $f _ { i }$ and voice cluster $v _ { j } { \mathrm { : } }$

$$
s _ { e , i } = \operatorname* { m a x } _ { \mathrm { t r a c k } } \frac { 1 } { k \in f _ { i } } \sum _ { | \mathcal { T } _ { k , e } | } s _ { k , t } , \quad S _ { i , j } = \operatorname* { m a x } _ { e \in v _ { j } } s _ { e , i } , \quad \mathcal { T } _ { k , e } = \{ t : t _ { e } ^ { s } \leq t \leq t _ { e } ^ { \mathrm { e } } \} .\tag{3}
$$

We assign each $v _ { j }$ to the face cluster $f _ { i ^ { * } ( j ) }$ with the highest score, $i ^ { * } ( j ) = \mathrm { a r g m a x } _ { i } S _ { i , j }$ , provided that $S _ { i ^ { * } ( j ) , j }$ exceeds a threshold $\tau _ { \mathrm { A S D } } . \mathrm { A s }$ an additional override, whenever $s _ { k , t }$ exceeds $\tau _ { \mathrm { o v e r i d e } }$ for some face track k within the time span of subtitle line $e ,$ we directly assign e to the face cluster containing $k ,$ bypassing the diarization-based grouping. This handles cases where TalkNet provides high-confidence per-frame evidence that contradicts the diarization output.

## B GRPO Reinforcement Learning Fine-Tuning

This section provides the full GRPO formulation summarised in Sec. 3.2.

Reward. We apply GRPO [22] with a composite reward that combines a format reward $r _ { f }$ and an accuracy reward $r _ { a } .$ . The format reward $r _ { f }$ is granted whenever the output $o _ { i }$ adheres to the expected response structure. The accuracy reward $r _ { a }$ measures how well the predicted set of name–identity pairs ${ \hat { \psi } } ( V )$ agrees with the ground-truth pairs $\psi ^ { * } ( V )$ . Letting $\hat { i } ( n )$ and $i ^ { * } ( n )$ denote the identities paired with name n under $\hat { \psi }$ and $\psi ^ { * }$ , respectively, we set

$$
r _ { a } = \sum _ { n \in N } \mathbf { 1 } \Big [ \widehat { i } ( n ) = i ^ { * } ( n ) \Big ] ,\tag{4}
$$

and define the full reward as $R _ { i } = r _ { f } + r _ { a }$

Advantage and Policy Update. The advantage $A _ { i }$ is computed by normalizing rewards within each group of $\breve { G }$ sampled responses:

$$
A _ { i } { = } \frac { R _ { i } { - } \mathrm { m e a n } ( \{ R _ { j } \} ) } { \mathrm { s t d } ( \{ R _ { j } \} ) } .\tag{5}
$$

Following Guo et al. [22], the policy is updated by maximising

$$
\mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { q , \{ o _ { i } \} } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \operatorname* { m i n } ( \rho _ { i } ( \theta ) A _ { i } , \operatorname { c l i p } ( \rho _ { i } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { i } ) - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) \right] ,\tag{6}
$$

where $\rho _ { i } ( \theta ) { = } { \pi } _ { \theta } ( o _ { i } \vert q ) / { \pi } _ { \theta _ { \mathrm { o l d } } } ( o _ { i } \vert q )$ is the importance ratio, ϵ is the clipping threshold, and $\beta$ controls the strength of the KL regularisation against the reference policy $\pi _ { \mathrm { r e f } }$

## C Error Correction for Character Bank Construction

Additional Details on Error Mitigation. As outlined in Sec. 4.2, the Markovian update process ϕ<sub>B</sub> is inherently sensitive to upstream noise, where errors from components such as transcription, active-speaker detection, and identity grounding can propagate over time. To address this, we elaborate on the three interventions applied along the streaming pipeline: face-track filtering, intra-entry name merging, and inter-entry name disambiguation.

Face-Track Filtering. Prior to clustering, we introduce an MLLM-based filtering stage to discard unreliable or ambiguous face tracks, thereby maintaining clean and identifiable clusters in the bank. This step removes low-quality observations, including blurred faces and extreme profile views. Empirically, embeddings derived from the retained high-quality tracks are sufficiently stable, rendering clustering errors negligible under our operating thresholds.

Query Name Filtering. As previously discussed, when no on-screen character actually corresponds to the candidate name, both our model and Gemini tend to over-assign one of the visible faces to an off-screen name rather than abstaining. To circumvent this, we filter the candidate name pool upstream so that off-screen names rarely reach the naming step in the first place. Concretely, we use an LLM (Qwen3.5-9B) to inspect each subtitle window and keep only those candidate names likely to refer to a character currently on screen. This requires reasoning over the flow of the surrounding dialogue rather than classifying each line in isolation, since whether a named person is actually present or merely being discussed is rarely decidable from a single utterance.

Inter-Entry Name Disambiguation. Ideally, each name should be uniquely associated with a single entry; however, MLLM-induced errors can lead to conflicts across entries. We resolve these after each shot using a greedy voting procedure: for names assigned to multiple entries, the entry with the highest vote retains the name, while others revert to their next-best candidates. This process is repeated until convergence.

Overall, these mechanisms address complementary failure modes in the streaming process and prevent error accumulation as $B _ { t }$ evolves, without requiring access to past observations.

## D Implementation Details

We set the face-clustering distance threshold to $\tau _ { \mathrm { c l u s t e r } } { = } 0 . 4$ , the active-speaker association threshold to $\tau _ { \mathrm { A S D } } = - 0 . 0 7 5$ , and the per-frame override threshold to $\tau _ { \mathrm { o v e r r i d e } } = 0 . 1$ . We use DiariZen-Large-s80-v2 [23] as our diarization model.

For training the MLLM for speaker identity recognition, we adopt Qwen3-VL-4B-Thinking [2] as our base model. To construct the first-stage supervised fine-tuning (SFT) dataset, we sample approximately 10% of the training set and prompt Gemini-3-Flash-Preview [19] to generate reasoning traces that lead to the given ground-truth name-to-track mappings. For SFT, we fine-tune only the language model component of Qwen3-VL-4B-Thinking while keeping both the vision encoder and the multimodal projector frozen, for one epoch with a batch size of 4, a learning rate of $1 \times 1 0 ^ { - 5 }$ , and a cosine schedule with warmup ratio 0.03. Then, we perform GRPO fine-tuning on the remaining training set for one epoch, using a learning rate of $1 \times \bar { 1 0 } ^ { - 6 }$ , a batch size of 4, $G = 4$ sampled responses per prompt, and a KL coefficient of $\beta { = } 0 . 0 0 4$

For streaming character bank construction, we set the face matching threshold to $\tau _ { f } = 0 . 5 5$ and the voice fallback threshold to $\tau _ { v } = 0 . 2 5$ . Voice embeddings are extracted with ECAPA-TDNN [14] over speaker-aligned subtitle segments; both face and voice centroids are maintained as L2-normalized running means over all cluster members. Candidate names are extracted from subtitles using spaCy transformer NER pipelines (en\_core\_web\_trf), and off-screen mentions are filtered out by Qwen3.5-9B [2]. The remaining names are linked to face clusters by our fine-tuned MLLM, which receives frames sampled at 2 FPS. Low-quality faces are filtered by Qwen3-VL-8B-Instruct [2] with stricter prompting before they are committed to the bank.

For PCR@K evaluation, we set the face-matching threshold $\tau _ { \mathrm { e v a l } } = 0 . 5 5$ and use the Double Metaphone algorithm for phonetic matching between predicted canonical names and IMDB ground-truth names. The IMDB reference embedding $\bar { e } _ { c } ^ { * }$ is the L2-normalized InsightFace embedding extracted from the actor’s IMDB headshot.

No hyperparameter mentioned above was tuned on the test set. In the absence of a validation split, all thresholds were set following prior work: $\tau _ { \mathrm { c l u s t e r } } , \tau _ { f }$ and $\tau _ { v }$ follow the multimodal clustering steup of Brown et al. [5], and $\tau _ { \mathrm { A S D } }$ and $\tau _ { \mathrm { o v e r r i d e } }$ follow TalkNet [57]. $\tau _ { \mathrm { e v a l } }$ is set equal to $\tau _ { f }$ (i.e. to 0.55), the value conventionally used to decide whether two InsightFace embeddings correspond to the same identity.

## E Ablation Study

We ablate the key design choices of our method on the MCN dataset to isolate their individual contributions. Using Qwen3-VL-4B-Thinking as the base model, we evaluate the fine-tuning strategy and data augmentation for character naming. We also ablate on error correction mechanisms to measure their effects on the full streaming pipeline.

Comparing with Direct SFT on an Instruct Model. To assess whether our two-stage recipe can be replaced by directly fine-tuning a larger Instruct model, we compare against Qwen3VL-8B-Instruct trained with SFT on the answer labels. While SFT alone on the 4B Thinking model performs slightly worse than the 8B baseline (64.16% vs. 64.92%), adding the RL stage pushes the smaller model to 72.76% overall accuracy, clearly surpassing the 8B Instruct model. Reasoning-based training therefore provides gains that cannot be recovered by model scale alone.

Table 5: Ablation study on Character Naming. We compare our two-stage (SFT + RL) fine-tuning of Qwen3VL-4B-Thinking against direct SFT on a larger Qwen3VL-8B-Instruct model, and further ablate the contributions of the RL stage and color-permutation augmentation.
<table><tr><td>Model</td><td>SFT</td><td>RL</td><td>Augment.</td><td>Overall Acc.</td><td>Known Acc.</td></tr><tr><td>Qwen3VL-8B-Instruct</td><td>√</td><td>X</td><td>X</td><td>64.92</td><td>71.77</td></tr><tr><td>Qwen3VL-4B-Thinking</td><td>√</td><td>X</td><td>X</td><td>62.39</td><td>67.98</td></tr><tr><td>Qwen3VL-4B-Thinking</td><td>√</td><td>X</td><td>√</td><td>64.16</td><td>70.41</td></tr><tr><td>Qwen3VL-4B-Thinking</td><td>√</td><td>√</td><td>X</td><td>71.37</td><td>79.56</td></tr><tr><td>Qwen3VL-4B-Thinking</td><td>√</td><td>√</td><td>√</td><td>72.76</td><td>81.06</td></tr></table>

Effect of Color-Permutation Augmentation. As shown in Tab. 5, perturbing the cluster-to-color mapping during training consistently improves performance across both training stages. At the SFT stage, it lifts overall accuracy from 62.39% to 64.16% and known accuracy from 67.98% to 70.41%. The gains persist after RL, with overall accuracy improving from 71.37% to 72.76% and known accuracy from 79.56% to 81.06%. These consistent improvements indicate that the augmentation prevents the model from exploiting spurious shortcuts tied to specific color assignments, encouraging it to genuinely reason about character identity from visual and dialogue cues.

Table 6: Ablation of error-correction mechanisms. We start from our full model and disable one mechanism at a time, measuring the impact on mTIoUs and PCR@Ks.
<table><tr><td>Variant</td><td>mTIoUv</td><td> $m T I o U _ { n }$ </td><td>PCR@5</td><td>PCR@10</td><td>PCR@15</td></tr><tr><td>Ours (Full)</td><td>84.90</td><td>54.40</td><td>63.00</td><td>54.54</td><td>48.19</td></tr><tr><td>w/o Face-Track Filtering</td><td>83.30</td><td>52.27</td><td>60.00</td><td>53.01</td><td>47.47</td></tr><tr><td>w/o Query Name Filtering</td><td>85.13</td><td>52.16</td><td>57.00</td><td>49.46</td><td>44.99</td></tr><tr><td>w/o Inter-Entry Name Disambiguation</td><td>84.90</td><td>52.40</td><td>62.00</td><td>53.00</td><td>47.48</td></tr></table>

Error-Correction Mechanisms. Tab. 6 summarises the contribution of each error-correction stage. Removing face-track filtering degrades both mTIoU (from 84.90 to 83.30) and mTIoU (from 54.40 to 52.27), suggesting that noisy or unreliable tracks introduce low-quality face features that weaken face clustering and subsequently affect name assignment. Removing query name filtering does not reduce mTIoU (from 84.90 to 85.13), but substantially reduces mTIoU (from 54.40 to 52.16) and PCR@10 (from 54.54 to 49.46). This indicates that off-screen mentions mainly contribute spurious candidate names, which interfere with the MLLM’s name-binding step rather than character localisation. Finally, removing inter-entry name disambiguation primarily affects name assignment: mTIoU decreases from 54.40 to 52.40, while mTIoU remains unchanged and PCR@K decreases slightly (e.g., PCR@5: 63.00 → 62.00). This shows that inter-entry disambiguation improves the consistency of name assignments across entries without changing localising characters’ first occurrences. Overall, the ablation distinguishes two input-stage filtering mechanisms that suppress noisy evidence before name binding from an inter-entry resolution mechanism that refines the final name assignments.

## F Additional Baseline for Streaming Character Bank Construction

In Sec. 5.3, we use Gemini-3-Flash-Preview and Gemini-3.1-Pro-Preview [19] as baselines for streaming character bank construction. We note that these baselines are disadvantaged by our requirement that they emit face exemplars, a form of output they are not designed to produce. No existing method targets this setting directly: prior work either presupposes a pre-built character bank or assumes a recurring television cast. Proprietary MLLMs are therefore the closest available point of comparison, notwithstanding the mismatch in output format. The lenient scoring criterion was adopted to compensate that.

To provide a comparison that does not rely on lenient scoring to compensate for the output-format mismatch, we have additionally implemented a classical weakly-supervised baseline [27]. It uses exactly the same face tracking, clustering, and majority-vote name binding as our pipeline, replacing only the mechanism by which names are linked to faces, and is given access to the full film rather than operating in a streaming pass. The weakly-supervised baseline uses subtitle text alone, with no audio, no pixels, and no vision-language model. A character vocabulary is extracted from the transcript by NER, and each mention is typed as first-, second-, or third-person using the five grammar rules in [27], so that first- and second-person mentions ("I’m Harry", "Hello, Harry") imply presence while third-person mentions ("Where is Harry?") imply absence. Each typed mention yields a positive or negative bag of face tracks from a surrounding window, here three shots centred on the mention’s shot. A per-character linear MIL model (Normalized Set Kernel, the best-performing solver in [27]) is trained on InsightFace embeddings, and a second-stage classifier, trained on each model’s most confident tracks, scores every track in the film and assigns the argmax name.

Table 7: Results of the MIL baseline on the MCN First Occurrence dataset. We report an additional baseline for streaming character bank construction.
<table><tr><td>Method</td><td>Streaming</td><td> $m T I o U _ { v }$ </td><td> $m T I o U _ { n }$ </td><td>PCR@5</td><td>PCR@10</td><td>PCR@15</td></tr><tr><td>Gemini-3-Flash-Preview</td><td>X</td><td>51.81</td><td>30.92</td><td>35.0</td><td>32.0</td><td>29.8</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>X</td><td>49.05</td><td>30.10</td><td>44.0</td><td>37.2</td><td>34.9</td></tr><tr><td>MIL [27]</td><td>X</td><td>82.85</td><td>33.38</td><td>34.0</td><td>19.7</td><td>16.7</td></tr><tr><td>Ours</td><td>√</td><td>84.90</td><td>54.40</td><td>63.0</td><td>54.5</td><td>48.2</td></tr></table>

In Tab. 7, since the baseline shares our face tracking and clustering, it attains a comparable mTIoU (82.85 against 84.90), confirming that the two systems recover visual identities equally well. On the naming metrics, however, it falls far behind despite full-movie access, reaching 33.38 m $\mathrm { \bar { I } I o U } _ { n }$ against our 54.40 and 34.0 PCR@5 against our 63.0. Dialogue cues under classical weak supervision are therefore insufficient to name characters reliably at feature-length scale, which is the gap our fine-tuned naming model closes.

## G Details on Dataset Collection

MovieNet [28] provides body bounding-box annotations on sparse 240P keyframes indexed by $^ { ( s , k ) }$ a shot index and within-shot image index. These indices do not identify frames in the native-resolution movie file, and MovieNet’s shot segmentation does not coincide with ours. To build the MCN dataset, we must (i) recover the exact frame number $f ^ { \star } \in \{ 0 , . . . , N - 1 \}$ } in the full video for each annotated keyframe, and (ii) rescale every box from $2 4 0 { \dot { \mathsf { P } } }$ to native resolution. We address this with a three-stage coarse-to-fine pipeline: CLIP retrieval for coarse localisation, longest-increasing-subsequence (LIS) filtering to remove inconsistent matches, and perceptual hashing for exact frame recovery.

CLIP Feature Extraction. For every MovieNet keyframe and for every 24th frame (≈ 1 fps at 24 fps) of each full-resolution video, we extract a 512-D CLIP ViT-B/32 image embedding. The keyframes form a query pool Q and the video samples a source pool S. Both pools are cached per movie so that the subsequent stages run independently per title.

Coarse Alignment via FAISS Retrieval and LIS Filtering. Within each movie we run exact L2 nearest-neighbor search (FAISS IndexFlatL2) of every query against S and take the top-1 source frame as a coarse anchor $\widehat { f } _ { i }$ for keyframe i. CLIP retrieval is robust to the resolution gap and to codec and color differences between annotation crops and rendered video frames, but it occasionally returns a visually similar shot from elsewhere in the movie. We prune such errors by exploiting the fact that MovieNet keyframes are listed in temporal order: the sequence $( \widehat { f } _ { i } ) _ { i }$ <sub>i</sub> must be monotonically increasing. We compute its longest increasing subsequence in $O ( n \mathrm { l o g } n )$ and mark every keyframe outside the LIS as unreliable $( \widehat { f } _ { i } \gets - 1 )$ . The criterion is strict but conservative: any rogue match must drop out of the LIS together with whichever neighbor it conflicts with, and the LIS retains the longer consistent chain.

Fine Alignment via Perceptual Hashing. The coarse stage anchors each keyframe to within ±24 frames. To recover the exact frame we scan a window $\boldsymbol { \overline { { \widehat { f } _ { i } } } } - \boldsymbol { W } , \boldsymbol { \widehat { f } _ { i } } + \boldsymbol { W } \boldsymbol { ] }$ of the full video with $W { = } 1 5 0$ . For each candidate frame and for the query we compute two 64-bit perceptual hashes on the CLAHE-normalized grayscale image: pHash (low-frequency DCT coefficients thresholded against their median) and dHash (horizontal gradient sign). Candidates are ranked by the combined Hamming distance $H _ { \mathrm { p } } { + } H _ { \mathrm { d } }$ , and we accept the lowest-ranked frame satisfying both $\dot { H } _ { \mathfrak { p } } { \le } 1 2$ and $H _ { \mathrm { d } } { \leq } 1 0$ . If no candidate meets both thresholds we fall back to the lowest combined-distance frame in the window. An optional ORB+RANSAC inlier rerank on the top-M candidates disambiguates near-duplicates within the same shot. When a keyframe’s coarse anchor was discarded by the LIS filter, we replace its window with one bracketed by the nearest two LIS-passing neighbors in the same movie, $f _ { \mathrm { p r e v } }$ and $f _ { \mathrm { n e x t } } .$ we use $[ \operatorname* { m i n } ( f _ { \mathrm { p r e v } } , f _ { \mathrm { n e x t } } ) , \dot { \operatorname* { m a x } } ( f _ { \mathrm { p r e v } } , f _ { \mathrm { n e x t } } ) ]$ when both exist, $\mathbf { a } \pm W$ window around the single available neighbor when only one exists, and drop the keyframe entirely when neither exists.

Bounding Box Rescaling and Output. With the exact frame index $f _ { i } ^ { \star }$ recovered, each MovieNet body box is rescaled from the per-row annotation resolution $( W _ { q } , H _ { q } )$ to native video resolution $( W _ { f } , H _ { f } )$ via axis-independent scale factors $s _ { x } { = } W _ { f } / W _ { q }$ and $s _ { y } { = } H _ { f } { / } H _ { q } ^ { \cdot }$ The resulting aligned records, of the form $( s , k , f _ { i } ^ { \star } , \mathbf { b } _ { i } , \mathrm { p i d } )$ where $\mathbf { b } _ { i }$ is the rescaled body box and pid is the MovieNet character identifier, serve as the automatic ground truth for the MCN dataset.

## H Dataset Examples

Fig. 5 shows six representative annotations from the MCN dataset, drawn from films that span genres, eras, and visual styles — from the period crowds of Gangs ofNew York to the romantic comedy of Sleep less in Seattle, the political drama of The Ides ofMarch, and the sports biopic Moneyball. This diversity is by design: a character-naming benchmark that generalizes beyond a single genre or production style must capture the full range of cinematographic conventions through which characters are introduced.

The examples also illustrate the variety of cues a model must exploit to resolve a name to the correct face. The most common case is direct address, in which one character names another in dialogue, but the visual realisation of this act is far from uniform. The named character may be identified through the canonical shot–reverse-shot structure, through an eyeline match when speaker and addressee share the frame, or through a telephone conversation where the named character is entirely off-screen at the moment of naming and must be linked to a face seen elsewhere in the clip. A separate class of examples requires reasoning from on-screen text: the character’s name appears as a written cue — a name tag, a letter, a sign, or a screen — and the model must perform OCR and then ground the recovered string to the correct visual identity. Cases like these are unsolvable from dialogue alone and require multimodal reasoning over both spoken and written language alongside visual evidence.

## I Qualitative Results for Character Naming

Qualitative Examples. Fig. 6 shows three representative examples covering the main cue types our model handles: direct address, where a speaker calls another character by name; third-party reference, where a name is mentioned about someone else in the scene; and visual reasoning, where the spoken cue is ambiguous and the mapping must be inferred from scene role. In each case, our model grounds the name to the correct colored box by jointly leveraging the spoken cue and the surrounding visual context.

Comparing with Baseline. We compare our output against a strong baseline on the same scene in Fig. 7. Our model produces a concise, well-grounded explanation and arrives at the correct identity, whereas the baseline generates a long chain of thought with repeated self-verifications yet commits to an incorrect mapping. Longer reasoning traces do not guarantee better grounding: our model achieves more accurate association with substantially fewer tokens.

Failure Cases. We analyse representative failure cases of our model and Gemini-3.1-Pro-Preview in Fig. 8 and Fig. 9. Fig. 8 shows cases where Gemini-3.1-Pro-Preview fails despite being the strongest proprietary baseline, suggesting that it can struggle to use subtle visual actions and interactions that are crucial for character naming. Fig. 9 further compares common failure modes across both models. Overall, our model appears better at leveraging fine-grained visual cues, speaker activity, and cross-turn dialogue context, while remaining errors are often caused by subtle actions, ambiguous speaker grounding, or noisy subtitles.

## J Qualitative Results for Streaming Character Bank Construction

We present qualitative results for streaming character bank construction on two films, Juno and Meet the Parents. As shown in Fig. 10, our pipeline constructs clean character entries for the major characters in Juno, grouping representative face tracks and linking them to canonical names while also preserving discovered aliases. Fig. 11 shows a second example on Meet the Parents. The bank again identifies the most prominent characters, but also reveals typical failure cases in complex multi-person scenes, where ambiguous visual context and transcription errors can lead to incorrect or noisy name bindings. These examples demonstrate that the proposed bank construction procedure is effective across different films, while remaining errors are primarily due to challenging name evidence rather than face-track clustering. Moreover, we refer readers to the supplementary video, which shows the full streaming character bank construction processfor Juno.

## K Experiment Compute Requirements

Fine-tuning of the MLLM is performed using four NVIDIA Ada 6000, A40, or L40S GPUs, each with 48 GB of memory. All inference on the MCN character-naming test set is conducted on a single NVIDIA Ada 6000, A40, or L40S GPU with 48 GB of memory. Construction of the streaming character bank also uses four NVIDIA Ada 6000, A40, or L40S GPUs.

## L Broader Impacts

Our method identifies characters in movies using cinematic and dialogue cues. This technique can be used to generate accurate audio descriptions and speaker-attributed subtitles, improving accessibility for audiences with visual or hearing impairments. This method does not perform open-world face recognition or release any identity gallery from personal video sources. We therefore do not foresee significant negative societal impacts from this work.

![](images/9a1cf26bcf2871ead9cd12293d4dcf85da7104cd8024808757f4daabb2e0177c.jpg)  
Q: Which colour corresponds to “Billy” and “David"? A: {Billy: Green; David:Magenta}.

![](images/fcfc5445c60832a8250d05a12b4fa11bccc6f64d0ae6a40a94a9b3947beb9bcb.jpg)  
Q: Which colour corresponds to "Bill"? A: {Bill: Yellow}.

![](images/a383425bb3d49eb9a87f46e2fd8b12a4c7b8904deef0fb9a35deab3e64a9ebaf.jpg)  
Q: Which colour corresponds to “Jonah” and “Sam”? A: {Jonah: Teal; Sam: Yellow}.

![](images/554f77b9b9b3d3cacd7f5891c54b1462a466949847b697af2f98fb599c3322ab.jpg)

![](images/fa346e8f5eca6fd2bc164d13451fdf855b700c6210a43642755389c3080c5894.jpg)  
Q: Which colour corresponds to “Laura"? A: {Laura: Blue}.

![](images/301cf2f4415cacd9a07e2cc8881f214d03b98b91d948011e323a58e61647ec52.jpg)  
Q: Which colour corresponds to “Paul Zara”? A: {Paul Zara: Pink}.

Figure 5: Examples from the MCN dataset. Six character-naming annotations illustrating the range of problem types covered: direct address with shot–reverse-shot structure, direct address with eyeline match, direct address over the telephone, on-screen text, and OCR-based reasoning. Clips are drawn from Moneyball, Gangs ofNew York, Sleepless in Seattle, The Help, Four Weddings and a Funeral, and The Ides of March.

![](images/859a8d02048380593da8ef33ad479e7ca8d09cb66b793594cf8c2b46521129ba.jpg)  
Figure 6: Qualitative results for character naming. We associate character names with their facial appearances in movies, mirroring how audiences infer who is who while watching. Characters may be introduced through diverse cues, including direct address by other characters (top), visual reasoning with common knowledge (middle), and third-party reference (bottom). In each case, the model grounds the name to the correct colored box by combining the spoken cue with visual evidence such as gaze, gesture, and scene role. Correctly linking character identities is central to movie understanding, enabling coherent character tracking across scenes. The examples are taken from Sleepless in Seattle, Four Weddings and a Funeral and Meet the Parents.

![](images/b81ecfe2961d8f99359f3f9bdef53e3346f49781847d7bd48bdace9d8c69c207.jpg)  
Figure 7: Comparison of reasoning efficiency on character naming. We compare our model against Qwen3.5-9B on the same scene. Our model produces a concise, well-grounded answer that directly links the spoken cue (“Congratulations, Walter”) to visual evidence (gaze and handshake direction), correctly identifying the orange box as Walter. In contrast, the baseline generates a lengthy chain of thought with repeated re-examinations and self-verifications, yet arrives at an incorrect mapping (green). This illustrates that longer reasoning does not guarantee better grounding: our model achieves more accurate character naming with substantially fewer tokens.

![](images/b2a7504970311235e59768891e5117a7e8cb9819072d3823e39b0cfb448d5f75.jpg)  
Figure 8: Examples of failure cases for both our model and Gemini-3.1-Pro-Preview. We show three examples in which Gemini-3.1-Pro-Preview, the strongest proprietary baseline, fails. These cases suggest that it struggles to capture subtle actions that are important for character naming. In the last example, Gemini is given the original audio rather than the erroneous subtitles for inference. The examples are taken from Four Weddings and a Funeral and Jerry Maguire.

![](images/02f5c0afbdbb0d9e2d151e817a67c7095391de99c028d41e22aa6f9a2690ca37.jpg)  
Figure 9: Examples of failure cases for either our model or Gemini-3.1-Pro-Preview. We show three examples in which Gemini-3.1-Pro-Preview or our model fails under common failure modes. These cases suggest that our model is better at leveraging fine-grained visual cues, speaker activity, and cross-turn dialogue context to resolve character names, while remaining errors often arise from subtle actions, ambiguous speaker grounding, or noisy subtitles. The examples are taken from Crazy, Stupid, Love., Star Wars: Episode IV- A New Hope, and The Big Lebowski.

![](images/db59049eca4c0ab6193432fdb35b38804b4a2355c74a154d6d0928ca299ed4ea.jpg)  
Figure 10: Example of the constructed character bank for Juno. We show the character bank constructed by our streaming pipeline. Each row (apart from the last two) corresponds to one character entry and includes representative face tracks, the predicted canonical name with confidence, and discovered aliases. The example illustrates that the bank captures major characters with clean visual clusters and resolves name variations, such as mapping Mama Bear to Juno, Bren to Brenda, and McGovern to Mac. Some naming errors remain, typically due to noisy or ambiguous character-name associations.

![](images/3bf7f16d388908461c813135134ab5813fec6b0ce733d6982644e4d0732bded4.jpg)  
Figure 11: Example of the constructed character bank for Meet the Parents. We show another example of a character bank constructed by our streaming pipeline. Our method correctly identifies the most prominent characters, but errors remain for the bottom three entries. These characters often appear in complex multi-person scenes, making reliable name binding more difficult. In addition, transcription errors can introduce noisy name variants; for example, Fonker and Falker should both be resolved to Focker for Greg Focker.