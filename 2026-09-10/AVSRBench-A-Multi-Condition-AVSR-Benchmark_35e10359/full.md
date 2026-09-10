# AVSRBench: A Multi-Condition AVSR Benchmark

Rishabh Jain

Sigmedia Group, School of Engineering

Trinity College Dublin, Ireland

rijain@tcd.ie

Naomi Harte

Sigmedia Group, School of Engineering

Trinity College Dublin, Ireland

nharte@tcd.ie

Abstract—While AVSR has achieved sub-1% word error rates on the standard LRS3 benchmark, its reliance on broadcast speech obscures whether this reflects true generalization or just domain adaptation. To investigate this gap, we evaluate three AVSR architectures across six conditions: controlled broadcast speech, fixed-grammar utterances, hyper-articulated Lombard speech, read speech from professional lipspeakers and nonprofessional speakers, and spontaneous multi-party video conversations. We find that visual-only performance deteriorates rapidly beyond broadcast domains, and audio-video fusion mainly benefits Lombard speech environments. Visual understanding degrades sharply at 90° profile views, with multimodal systems relying largely on acoustic fallback. Additionally, speaker articulation proves more critical than minor camera shifts, and LLM-based architectures suffer from poor out-of-domain generalization. Our work highlights a significant generalization gap in current AVSR research. To address this, we also introduce RoomReader-AV as a new benchmark for AVSR and release a unified data preprocessing pipeline to make comprehensive multi-condition evaluation accessible.

Index Terms—AVSR, Lipreading, LRS Datasets, Multicondition evaluation, Benchmarking AVSR, RoomReader

## I. INTRODUCTION

Audio-visual speech recognition (AVSR) has made major progress on standard benchmarks over the past several years [1]–[5]. Models such as AV-HuBERT [6], [7], Auto-AVSR [8], and Llama-AVSR [9] have achieved word error rates (WER) below 2% on LRS2 [10] and LRS3 [11] broadcast corpora, the two datasets that dominate AVSR research. With performance approaching near-perfect on these controlled broadcast benchmarks, it is easy to get the impression that AVSR is close to a solved problem [12].

Some recent studies [5], [13]–[15] have attempted to move beyond controlled broadcast corpora by simulating more challenging conditions, such as cocktail-party noise [16]–[18], in-the-wild speech [19]–[21], video conferencing [22], and Lombard speech [23]–[25]. Cocktail-party evaluations show that models can degrade sharply from 7% to over 69% WER when interfering speakers and background noise are present [26]. In-the-wild benchmarks also report an average 30% absolute WER spike across VSR models relative to LRS3, suggesting that broadcast-trained representations do not generalize well beyond controlled lip movements [21]. Video conferencing scenarios show a similar pattern, with Auto-AVSR’s audio-visual (AV) WER rising from under 1% to over 33% on Zoom [22]. The LRS-VoxMM benchmark [20] confirms that real-world audio distortions like reverberation make conversational speech significantly harder to transcribe than LRS3. Lombard speech [27] adds another challenge, since noise-induced hyperarticulation [28] changes both acoustic and visual speech production in ways that broadcasttrained models are not typically exposed to [25]. Even recent LLM-based decoders [29], [30] mainly improve performance through lexical decoding, rather than better visual feature representation, which leaves the visual encoder as a major bottleneck [31]. This limitation stems from models relying heavily on training word frequencies rather than actual visual perception [32]. Lin et al. [33] use MultiVSR [34] subsets to confirm that this vocabulary reliance impacts performance severely, even in LRS3-matching environments. More broadly, these studies show that AVSR performance is highly sensitive to out-of-domain conditions, but they still examine each factor in isolation [35], [36].

Most AVSR research still evaluates models on broadcast speech corpora, where speakers face the camera, lighting is good, and speech is clearly articulated. The LRS3 [11] test set is less than an hour, which can inflate performance and hide how sharply models degrade outside the broadcast setting. Many AVSR models ship with evaluation code that is tightly coupled to the LRS2 [10] and LRS3 benchmark datasets, and running them on new corpora often requires custom data preparation, cleaning, and decoding scripts that are not standardized or easy to reuse. This makes it hard to compare models fairly across datasets and increases the effort required to move beyond LRS2 and LRS3 in a systematic way.

To address these gaps, this paper makes three core contributions. First, we release a standardized audio-visual and lipreading data preparation toolkit designed for common evaluation across multiple datasets. This toolkit integrates with Auto-AVSR and AV-HuBERT codebases allowing researchers to streamline multi-condition evaluations. Second, we introduce RoomReader [37] as a benchmark for spontaneous, multi-party video conferencing conversations. It captures spontaneous Zoom interactions rather than controlled broadcast speech and thus provides a challenging setting for conversational AVSR. Third, we conduct a systematic evaluation of three modern AVSR systems across six publicly available datasets that have not previously been benchmarked together. This evaluation tests variables that earlier work often overlooks, including nonfrontal camera angles, speaker articulation, Lombard speech, and video conferencing conditions.

TABLE I  
DATASET SUMMARY FOR EVALUATION
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Hrs.</td><td rowspan=1 colspan=1>Utter.</td><td rowspan=1 colspan=1>Spk.</td><td rowspan=1 colspan=1>Condition</td></tr><tr><td rowspan=1 colspan=1>LRS2</td><td rowspan=1 colspan=1>0.80h</td><td rowspan=1 colspan=1>1,243</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>Broadcast (BBC TV)</td></tr><tr><td rowspan=1 colspan=1>LRS3</td><td rowspan=1 colspan=1>0.84h</td><td rowspan=1 colspan=1>1,321</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>Broadcast (TED/TEDx)</td></tr><tr><td rowspan=1 colspan=1>GRID</td><td rowspan=1 colspan=1>23.0h</td><td rowspan=1 colspan=1>32,895</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>Fixed-vocabulary syntax</td></tr><tr><td rowspan=1 colspan=1>LombardGrid</td><td rowspan=1 colspan=1>7.35h</td><td rowspan=1 colspan=1>10,749</td><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=1>Lombard speech in noise</td></tr><tr><td rowspan=1 colspan=1>TCD-TIMIT</td><td rowspan=1 colspan=1>22.0h</td><td rowspan=1 colspan=1>13,826</td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=1>Phonetically rich read speech</td></tr><tr><td rowspan=1 colspan=1>RoomReader</td><td rowspan=1 colspan=1>6.49h</td><td rowspan=1 colspan=1>10,324</td><td rowspan=1 colspan=1>118</td><td rowspan=1 colspan=1>Spontaneous conversational</td></tr></table>

Hrs. = Hours, Utter. = Utterances; Spk. = Speakers.

Our evaluation under visual-only (VO), audio-only (AO), and audio-visual (AV) settings shows several clear patterns. VO recognition fails badly outside broadcast conditions, and neither more training data nor an LLM-based decoder solves this problem. AV fusion helps mainly in Lombard speech environments; otherwise, it offers little to no advantage over AO recognition. We also find that minor visual changes have little effect, but extreme profile views and speaker articulation matter much more. Our work shows that current AVSR systems remain brittle outside controlled broadcast data.

## II. DATASETS AND PROCESSING

## A. Dataset Overview

We evaluate across six datasets spanning a range from broadcast speech to fully spontaneous conversation, as summarized in Table I.

LRS2 [10] consists of clips from BBC television broadcasts. We use only the test set in this work.

LRS3 [11] consists of TED and TEDx talk clips. It is the primary training domain for AV-HuBERT and Auto-AVSR and serves as the main in-domain benchmark throughout this paper.

GRID [38] is a 34-speaker corpus with a fixed six-word command grammar. The audio is clean and the recordings are frontal, but the vocabulary is unseen during training. This makes GRID a useful test of generalization beyond the training vocabulary and sentence structure.

LombardGrid [39] contains 54 speakers recorded while listening to 80 dB background noise, which elicits Lombard speech: louder voice, higher pitch, and more exaggerated lip movements. Crucially, synchronized frontal (0°) and profile (90°) cameras allow direct comparison of camera angle effects within the same recording session.

TCD-TIMIT [40] contains 62 speakers recording 6,913 phonetically rich TIMIT sentences. The dataset is first divided by articulation expertise into three professional lipspeakers (trained to articulate clearly for hearing-impaired audiences) and 59 non-professional volunteers. Each of these groups is then further subdivided by camera angle into straightcam (0°) and 30degcam (30°) subsets.

RoomReader [37] contains 30 Zoom-based tutorial sessions with 118 participants and manually corrected transcriptions with word-level boundaries. Speech is fully spontaneous and conversational. The dataset provides distinct audio streams: a session-level audio file containing all participants speech (which inherently includes overlapping speech from other speakers), and individual audio tracks containing each speaker’s isolated audio. While our preprocessing pipeline supports both formats, we report results exclusively on the individual participant stream to evaluate noisy, single-speaker Zoom-based speech. For our evaluation, we further divide the data into Easy and Hard subsets, the specific criteria for which are detailed later in Section IV-E. RoomReader has not previously been used for AVSR evaluation and offers a novel and challenging out-of-domain benchmark for AVSR.

![](images/035ee362c588dc0492ec3cf65e11dcf379b7b57a8e5e933479e6151d7405306b.jpg)  
Fig. 1. Examples from the datasets used in this study. First row: professional lipspeaker and volunteer from TCD-TIMIT at 0<sup>◦</sup> and 30<sup>◦</sup> angles. Second row: LombardGrid speakers at 0<sup>◦</sup> and 90<sup>◦</sup> views. Third row: RoomReader Zoom conversations. GRID, LRS2, and LRS3 omitted due to space constraints.

Figure 1 presents examples from the TCD-TIMIT, LombardGrid, and RoomReader datasets. For datasets other than LRS2 and LRS3, which use predefined test sets, we use the entire corpus for inference.

## B. Preprocessing Pipeline

A practical barrier to multi-condition AVSR evaluation is not the absence of suitable datasets, but format incompatibility. Adapting new and differently formatted datasets to widely used frameworks like AV-HuBERT and Auto-AVSR (which serve as the foundation for most modern AVSR systems) requires substantial, undocumented engineering effort. As a result, researchers often avoid this bottleneck by evaluating only on standard benchmarks like LRS2 and LRS3, whose test sets are less than 1 hour. To address this, we release a preprocessing pipeline that converts GRID, LombardGrid, TCD-TIMIT, and RoomReader into formats directly compatible with both frameworks. This pipeline requires no modification to either codebase, allowing users to seamlessly integrate these datasets into existing research workflows.

The pipeline applies a consistent set of preprocessing steps across all datasets. First, faces are detected and mouth regions of interest (ROIs) are extracted using RetinaFace [41] with 68-point facial landmarks, producing 96×96 pixel crops. Audio is standardized by downsampling to 16 kHz mono, while transcripts are normalized by removing punctuation and converting text to lowercase. For GRID dataset [38], which lacks explicit transcripts, we generate them by mapping them to their fixed six-word vocabulary pattern. For RoomReader [37], although disfluency markers (\$ and #) are removed, we explicitly retain backchannels and filled pauses (e.g., ”yeah,” ”hmm”) to preserve the natural flow of spontaneous conversation. Retaining these elements directly reflects realworld use cases and significantly increases the challenge of the benchmark. Finally, dataset manifests are generated in formats compatible with AV-HuBERT [6] and Auto-AVSR [8]. The newly revamped RoomReader dataset produced by this pipeline is referred to as RoomReader-AV in this work. The complete pipeline, along with comprehensive documentation detailing the data cleaning steps required for each dataset, is publicly available on our GitHub<sup>1</sup>.

TABLE II  
WER(%) COMPARISON ON THE LRS2 AND LRS3 DATASETS ACROSS DIFFERENT MODALITIES
<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=2>LRS2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>LRS3</td></tr><tr><td rowspan=1 colspan=1>VO ↓</td><td rowspan=1 colspan=1>AO↓</td><td rowspan=1 colspan=1>AV↓</td><td rowspan=1 colspan=1>VO ↓</td><td rowspan=1 colspan=1>AO↓</td><td rowspan=1 colspan=1>AV ↓</td></tr><tr><td rowspan=2 colspan=1>Auto-AVSRAV-HuBERTLlama-AVSR</td><td rowspan=1 colspan=1>14.6538.00</td><td rowspan=1 colspan=1>1.468.26</td><td rowspan=1 colspan=1>1.767.25</td><td rowspan=1 colspan=1>19.1028.69</td><td rowspan=1 colspan=1>1.001.95</td><td rowspan=2 colspan=1>0.901.470.79</td></tr><tr><td rowspan=1 colspan=1>41.57</td><td rowspan=1 colspan=1>5.07</td><td rowspan=1 colspan=1>4.58</td><td rowspan=1 colspan=1>26.20</td><td rowspan=1 colspan=1>0.74</td></tr></table>

Results are reported using the best checkpoint released by the authors of each model. ↓: Lower is better.

## III. AVSR MODELS

We evaluate three state-of-the-art AVSR models using their official pretrained checkpoints. All decoding settings follow each model’s official implementation and were held constant across datasets and modalities. Performance is measured in WER across VO, AO, and AV settings.

Auto-AVSR [8] is trained on 3,448 hours of AV data (LRW [42], LRS2, LRS3, VoxCeleb2 [43], AVSpeech [44]) using a ResNet-18 visual frontend, Conformer encoder [45], and hybrid CTC/attention decoder [46]. It represents the largescale supervised approach. For the scale analysis in Section IV-F, we also evaluate a checkpoint trained on 1,759 hours (LRS3 + VoxCeleb2), which matches AV-HuBERT’s and Llama-AVSR’s pretraining data volume and allows for a fairer comparison.

AV-HuBERT Large [6] uses masked multimodal prediction for self-supervised pretraining [47] using VoxCeleb2 and LRS3 datasets, followed by fine-tuning on LRS3. It processes multi-stream AV input with a shared Transformer. The noiseaugmented checkpoint is used throughout this work.

Llama-AVSR [9] combines pretrained AV-HuBERT visual features with Whisper [48] acoustic features and uses a Llama3.1-8B-based [49] LLM decoder. It is finetuned with a LoRA [50] adapter on the LRS3 and VoxCeleb2 datasets.

## IV. EXPERIMENT AND RESULTS

## A. Large-Scale Lip-Reading Corpora: LRS2 and LRS3

Table II shows the AVSR results on LRS2 and LRS3. LRS3 is the shared training dataset across all three models. LRS2 is in-domain for Auto-AVSR but unseen for AV-HuBERT and Llama-AVSR. AV performance is at, or near, AO for all models on LRS3, confirming that visual fusion works when test conditions match training. Llama-AVSR achieves the best LRS3 AV (0.79%) and AO (0.74%), with Auto-AVSR close behind (0.90% AV). On LRS2, Auto-AVSR achieves the strongest VO due to direct LRS2 training; however, for the models where LRS2 is unseen, AV-HuBERT outperforms Llama-AVSR in the VO setting (38.00% vs. 41.57%), showing stronger visual generalization. Even so, VO remains dramatically weaker than AO and AV for all models.

TABLE III  
BENCHMARKING WER (%) RESULTS ON GRID DATASET.
<table><tr><td>Model</td><td>VO ↓</td><td>AO↓</td><td>AV ↓</td><td>∆ (AO-AV)</td></tr><tr><td>Auto-AVSR</td><td>66.53</td><td>12.57</td><td>21.47</td><td>-8.90</td></tr><tr><td>AV-HuBERT</td><td>83.80</td><td>42.42</td><td>42.98</td><td>-0.56</td></tr><tr><td>Llama-AVSR</td><td>116.39</td><td>42.26</td><td>84.14</td><td>-41.88</td></tr></table>

∆ (AO–AV) denotes the difference between AO and AV WER. Negative values indicate degradation from AV fusion. ↓: Lower is better.

TABLE IV  
ERROR EXAMPLES FOR AV-HUBERT ON GRID DATASET.
<table><tr><td rowspan=1 colspan=1>Reference</td><td rowspan=1 colspan=1>AO</td><td rowspan=1 colspan=1>VO</td></tr><tr><td rowspan=1 colspan=1>place blue in b 7 pleaselay blue with r 2 pleaseplace white in i 6 nowlay white in c 1 again</td><td rowspan=1 colspan=1>place blew and be 7 pleaselay blue with are 2 pleaseplease wide in icex nowlay white in sea 1 again</td><td rowspan=1 colspan=1>but it&#x27;s blue and pizzaval beeslight blewer beer 3 songyeah i&#x27;m here&#x27;s why i sit downplease wine i&#x27;ll see what</td></tr></table>

\*AV-HuBERT AV results are omitted as they mirror the AO results.

## B. Structured Read Speech: GRID

The GRID corpus consists of fixed six-word command grammar (e.g., place green at B 4 now), which is entirely absent from all three models’ training data, creating a domain mismatch in both vocabulary and sentence structure. As seen in Table III, VO performance collapses for all models, reaching 66.53% for Auto-AVSR, 83.80% for AV-HuBERT, and exceeding 116% for Llama-AVSR. More critically, AV performance is worse than AO for Auto-AVSR (-8.90%) and substantially worse for Llama-AVSR (-41.88%), while AV-HuBERT shows no change. Overall, the visual stream contributes little benefit under this domain shift and in most cases actively degrades recognition performance. Auto-AVSR achieves the best generalization across all modalities, reflecting its more diverse training data.

The differences between the AO and VO errors in Table IV illustrate the contrasting failure modes of acoustic versus visual recognition on this structural grammar. The AO model relies entirely on sound, so its errors are primarily homophones. It captures the correct phonemic structure but outputs alternative spellings, such as mistaking the letters and numbers b 7 for be 7, r 2 for are 2, or c 1 for sea 1. Conversely, the VO model relies entirely on visual cues, so it struggles with words that look identical on the lips (homophenes). To compensate for visual ambiguity, the model hallucinates phrases to satisfy its language decoder. This causes it to alter the syntax (turning place blue into but it’s blue), drop articulatory cues (white becoming why), or introduce completely unrelated vocabulary (c 1 again becoming wine i’ll see what).

## C. Lombard Speech Under Noise: LombardGrid

LombardGrid represents the only out-of-domain scenario where AV consistently improves over AO, as shown in Table V. Results include the entire dataset, combining both the frontal and profile viewing conditions. While the recorded audio remains clean, speakers heard 80 dB noise through headphones to induce the Lombard effect [51], [52]. This causes speakers to naturally hyper-articulate and exaggerate their facial movements, making the dataset well-suited for demonstrating the benefits of visual fusion. In this scenario, Auto-AVSR benefits clearly from visual information, with AV WER (11.73%) improving over AO (14.02%). AV-HuBERT shows a small but consistent gain as well. In contrast, Llama-AVSR does not follow this trend, with AV performing slightly worse than AO. Furthermore, Llama-AVSR occasionally produces nonsensical outputs, such as severe repetitions and incoherent transcriptions, which is a characteristic failure mode of LLM-based decoders prone to hallucination when facing outof-domain distribution shifts. Notably, Auto-AVSR’s +2.29% AV improvement directly reverses its -8.90% degradation on the identical grammar of the standard GRID corpus (Table III), demonstrating that hyper-articulated visual cues can compensate for severe syntactic mismatch.

TABLE V  
BENCHMARKING WER (%) PERFORMANCE ON LOMBARDGRID
<table><tr><td>Model</td><td>VO ↓</td><td>AO↓</td><td>AV ↓</td><td>∆ (AO-AV)</td></tr><tr><td>Auto-AVSR</td><td>78.69</td><td>14.02</td><td>11.73</td><td>+2.29</td></tr><tr><td>AV-HuBERT</td><td>94.80</td><td>34.04</td><td>33.64</td><td>+0.40</td></tr><tr><td>Llama-AVSR</td><td>145.11</td><td>36.41</td><td>39.20</td><td>-2.79</td></tr></table>

↓: Lower is better. ∆ (AO–AV) denotes the difference between AO and AV WER. Positive values indicate improvement from audiovisual fusion, while negative values indicate degradation.

TABLE VI  
WER(%) PERFORMANCE ON LOMBARDGRID UNDER FRONTAL (0<sup>◦</sup>) AND SIDE-PROFILE (90<sup>◦</sup>) VIEWS.
<table><tr><td rowspan="2">Model</td><td colspan="2">VO ↓</td><td colspan="2">AO↓</td><td colspan="2">AV↓</td></tr><tr><td>0º</td><td>90°</td><td>0º</td><td>90°</td><td>0º</td><td>90°</td></tr><tr><td>Auto-AVSR</td><td>64.88</td><td>92.50</td><td>14.03</td><td>14.03</td><td>11.59</td><td>11.87</td></tr><tr><td>AV-HuBERT</td><td>87.06</td><td>102.53</td><td>34.04</td><td>34.04</td><td>33.11</td><td>34.17</td></tr><tr><td>Llama-AVSR</td><td>111.01</td><td>180.62</td><td>36.41</td><td>36.41</td><td>38.72</td><td>39.68</td></tr></table>

↓: Lower is better. 0<sup>◦</sup> denotes frontal (head-on) view, while 90<sup>◦</sup> denotes profile (side-view) condition.

VO performance degrades significantly at the 90<sup>◦</sup> (profile) view for all models (see Table VI). Auto-AVSR WER increases from 64.88% to 92.50%, AV-HuBERT from 87.06% to 102.53%, and Llama-AVSR from 111.01% to 180.62%, indicating a strong sensitivity of visual recognition to head pose. In contrast, AO performance remains unchanged across views, as expected, since it does not depend on visual input. AV performance shows only minor changes, with Auto-AVSR moving from 11.59% to 11.87% and AV-HuBERT from 33.11% to 34.17%. This suggests that AV robustness under profile views is not due to reliable visual recognition at extreme poses, but rather due to the dominance of the audio stream in fusion when visual information becomes unreliable, with the exception of Auto-AVSR (90° AV 11.87% vs. AO 14.03%).

## D. Camera Angle and Speaker Articulation: TCD-TIMIT

TCD-TIMIT enables us to independently examine the effects of camera angle, using its $0 ^ { \circ }$ and 30° recordings, and articulation quality, by comparing professional lipspeakers with everyday volunteers (non-professional speakers).

TABLE VII  
WER (%) ON TCD-TIMIT ACROSS VIEWING ANGLES (0<sup>◦</sup> AND 30<sup>◦</sup>).
<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=2 colspan=1>Mod.</td><td rowspan=2 colspan=3>Lipspeakers0°↓30° ↓ Δ</td><td rowspan=1 colspan=3>Volunteers</td></tr><tr><td rowspan=1 colspan=1>30°↓</td><td rowspan=1 colspan=1>Δ</td><td rowspan=1 colspan=1>0°↓</td><td rowspan=1 colspan=1>30°↓</td><td rowspan=1 colspan=1>Δ</td></tr><tr><td rowspan=2 colspan=1>Auto-AVSRAV-HuBERTLlama-AVSR</td><td rowspan=2 colspan=1>VO</td><td rowspan=2 colspan=1>25.9138.7343.90</td><td rowspan=2 colspan=1>26.5438.7645.51</td><td rowspan=1 colspan=1>0.63</td><td rowspan=2 colspan=1>45.5763.7387.95</td><td rowspan=2 colspan=1>47.6865.1590.78</td><td rowspan=2 colspan=1>2.111.422.83</td></tr><tr><td rowspan=1 colspan=1>0.031.61</td></tr><tr><td rowspan=1 colspan=1>Auto-AVSRAV-HuBERTLlama-AVSR</td><td rowspan=1 colspan=1>AO</td><td rowspan=1 colspan=1>4.1411.085.54</td><td rowspan=1 colspan=1>4.1611.135.60</td><td rowspan=1 colspan=1>0.020.050.06</td><td rowspan=1 colspan=1>6.5916.838.47</td><td rowspan=1 colspan=1>6.4516.377.88</td><td rowspan=1 colspan=1>0.140.460.59</td></tr><tr><td rowspan=2 colspan=1>Auto-AVSRAV-HuBERTLlama-AVSR</td><td rowspan=2 colspan=1>AV</td><td rowspan=2 colspan=1>7.0611.8910.98</td><td rowspan=2 colspan=1>7.2011.8210.99</td><td rowspan=2 colspan=1>0.140.070.01</td><td rowspan=1 colspan=1>10.8818.24</td><td rowspan=2 colspan=1>10.2417.8514.64</td><td rowspan=2 colspan=1>0.640.390.61</td></tr><tr><td rowspan=1 colspan=1>15.25</td></tr></table>

↓: lower is better. Mod. = Modalities. 0<sup>◦</sup> = frontal view; 30<sup>◦</sup> = semi-profile view. $\Delta = | 0 ^ { \circ }$ − 30<sup>◦</sup>| (absolute performance gap between angles).

TABLE VIII  
WER (%) IN TCD-TIMIT ACROSS VARYING MODALITY FOR LIPSPEAKERS, VOLUNTEERS, AND MATCHED VOLUNTEERS.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Mod.</td><td rowspan=1 colspan=1>LS ↓</td><td rowspan=1 colspan=1>V↓</td><td rowspan=1 colspan=1>MV↓</td><td rowspan=1 colspan=1>∆ (LS-MV)</td></tr><tr><td rowspan=2 colspan=1>Auto-AVSRAV-HuBERTLlama-AVSR</td><td rowspan=2 colspan=1>VO</td><td rowspan=2 colspan=1>26.2238.7543.90</td><td rowspan=2 colspan=1>46.6364.4489.46</td><td rowspan=1 colspan=1>43.15</td><td rowspan=2 colspan=1>-16.93-21.64-44.05</td></tr><tr><td rowspan=1 colspan=1>60.3987.95</td></tr><tr><td rowspan=1 colspan=1>Auto-AVSRAV-HuBERTLlama-AVSR</td><td rowspan=1 colspan=1>AO</td><td rowspan=1 colspan=1>4.1511.085.54</td><td rowspan=1 colspan=1>6.7216.608.16</td><td rowspan=1 colspan=1>6.0816.838.47</td><td rowspan=1 colspan=1>-1.93-5.75-2.93</td></tr><tr><td rowspan=3 colspan=1>Auto-AVSRAV-HuBERTLlama-AVSR</td><td rowspan=3 colspan=1>AV</td><td rowspan=3 colspan=1>7.1311.8910.98</td><td rowspan=1 colspan=1>10.56</td><td rowspan=1 colspan=1>9.49</td><td rowspan=3 colspan=1>-2.36-6.35-4.27</td></tr><tr><td rowspan=1 colspan=1>18.04</td><td rowspan=1 colspan=1>18.24</td></tr><tr><td rowspan=1 colspan=1>14.98</td><td rowspan=1 colspan=1>15.25</td></tr></table>

↓: lower is better. Mod. = Modalities. LS = Lipspeakers, V = Volunteers, MV = Matched Volunteers. ∆ (LS−MV) represents the performance gap between LS and MV.

1) Effect of Camera Angle in TCD-TIMIT: We evaluate all models under two viewing conditions, frontal (0<sup>◦</sup>) and semi-side profile (30<sup>◦</sup>), across Lipspeakers and Volunteers as presented in Table VII. Across all configurations, the change in WER between $0 ^ { \circ }$ and $3 0 ^ { \circ }$ remains small, consistently below 3% absolute difference. AO performance is unchanged across viewing angles, as expected since it does not depend on visual input. Similarly, AV performance shows no meaningful variation with pose change in this moderate range. This suggests that small frontal offsets are within the robustness range of current visual encoders. When combined with the LombardGrid 90<sup>◦</sup> results in Section IV-C, a consistent pattern emerges. While moderate pose changes have negligible impact, extreme profile views lead to substantial degradation in visual recognition. In those cases, AV performance is largely maintained through reliance on the audio stream rather than true visual pose invariance. This aligns with findings by Lan et al. [53], who showed that for computer lip-reading systems, moderate angles preserve the visibility of important articulatory gestures.

2) Speaker Articulation in TCD-TIMIT: All three professional lipspeakers in TCD-TIMIT are female. To enable a fair comparison with the volunteer group, we define a matched volunteers (MV) subset of female volunteers with identical sentences, durations, and utterance counts to the lipspeakers. This subset helps us isolate the effect of articulation, as professional lipspeakers are trained to produce clear visible lip movements while volunteers are everyday speakers with no such training.

TABLE IX WER (%) ON ROOMREADER-AV
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>VO ↓</td><td rowspan=1 colspan=1>AO ↓</td><td rowspan=1 colspan=1>AV ↓</td><td rowspan=1 colspan=1>∆(AO-AV)</td></tr><tr><td rowspan=1 colspan=1>Auto-AVSR</td><td rowspan=1 colspan=1>110.07</td><td rowspan=1 colspan=1>25.40</td><td rowspan=1 colspan=1>26.39</td><td rowspan=1 colspan=1>-0.99</td></tr><tr><td rowspan=1 colspan=1>AV-HuBERT</td><td rowspan=1 colspan=1>92.54</td><td rowspan=1 colspan=1>36.18</td><td rowspan=1 colspan=1>34.44</td><td rowspan=1 colspan=1>+1.74</td></tr><tr><td rowspan=1 colspan=1>Llama-AVSR</td><td rowspan=1 colspan=1>311.48</td><td rowspan=1 colspan=1>88.01</td><td rowspan=1 colspan=1>128.54</td><td rowspan=1 colspan=1>-40.53</td></tr></table>

↓: lower is better. ∆ (AO–AV) = AO–AV WER difference; positive = improvement from AV fusion, negative = degradation from AV fusion.

![](images/7e0a280223e747dc9b178d0fc9a8116ec9708fb4050b70e0d8b16507c395dd3a.jpg)  
Fig. 2. WER (%) of the Auto-AVSR model across utterance durations (in seconds) for the VO, AO, and AV modalities on RoomReader-AV.

The articulation gap between Lipspeakers and Matched Volunteers is large and highly consistent across all settings (Table VIII). The precise articulation of the professional lipspeakers is directly reflected in the VO performance. For example, at 0<sup>◦</sup>, Auto-AVSR achieves a 25.91% WER on lipspeakers compared to 45.57% on the matched volunteers. AV-HuBERT and Llama-AVSR exhibit similar, substantial performance gaps between the two groups, confirming that professional articulation significantly aids visual recognition. Volunteers remain significantly harder to recognize in the VO setting, with performance drops ranging from approximately 17% to 44% depending on the model. In contrast, AO performance shows only minor degradation, typically between 1% and 6%, indicating that the acoustic signal is similarly clean across both speaker groups. However, this visual advantage does not translate to improved AV performance. Across all models, AV performance is consistently worse than AO performance. The TCD-TIMIT audio recordings are exceptionally clean and AO performance is already highly accurate. In this scenario, fusing the less reliable visual modality actively degrades the nearperfect audio stream, resulting in a negative AV fusion effect regardless of the speaker’s articulation quality.

## E. Spontaneous Conversational Speech: RoomReader-AV

RoomReader-AV is the most challenging subset in this evaluation, with no in-domain exposure for any model. For our evaluation, we use the individual audio stream, featuring each speaker’s isolated audio. Several patterns emerge from the results in Table IX. VO collapses for all models. AO performance is also challenged but remains viable for Auto-AVSR (25.40%) and AV-HuBERT (36.18%), whereas Llama-AVSR struggles significantly with the AO condition (88.01%). Auto-AVSR AV closely follows its AO performance, indicating it degrades without overly relying on the visual stream. AV-HuBERT is the only model to show an AV improvement, suggesting it can extract some complementary visual cues even in adverse conditions. Llama-AVSR suffers a severe AV penalty, with its WER increasing by over 40 compared to its AO result. Llama-AVSR’s extreme VO error rate of 311.48% reflects severe hallucinations. For short backchannel inputs like ”yeah” or ”hmm,” the model generates long, unrelated sequences, as pretrained language model knowledge overrides both visual and acoustic evidence. This represents a fundamentally different failure mode from other architectures.

TABLE X  
WER (%) ON ROOMREADER-AV SUBSETS: EASY AND HARD
<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=3>RR-AV_Easy (≥ 2s)</td><td rowspan=1 colspan=3>RR-AV_Hard (&lt; 2s)</td></tr><tr><td rowspan=1 colspan=1>VO ↓</td><td rowspan=1 colspan=1>AO ↓</td><td rowspan=1 colspan=1>AV↓</td><td rowspan=1 colspan=1>VO ↓</td><td rowspan=1 colspan=1>AO ↓</td><td rowspan=1 colspan=1>AV ↓</td></tr><tr><td rowspan=2 colspan=1>Auto-AVSRAV-HuBERT</td><td rowspan=2 colspan=1>105.3589.72</td><td rowspan=2 colspan=1>20.6028.61</td><td rowspan=2 colspan=1>21.7226.90</td><td rowspan=1 colspan=1>126.95</td><td rowspan=1 colspan=1>42.58</td><td rowspan=1 colspan=1>43.14</td></tr><tr><td rowspan=1 colspan=1>96.65</td><td rowspan=1 colspan=1>53.75</td><td rowspan=1 colspan=1>52.24</td></tr></table>

↓: lower is better. Llama-AVSR is excluded due to high error rates.

Why is VO much worse than AO on RoomReader-AV? Both modalities struggle, but audio degrades less significantly. Several factors explain the visual collapse. First, spontaneous meeting speech is informal, with casual vocabulary, fillers, incomplete sentences, and frequent topic shifts, which differs significantly from the scripted or semi-scripted speech the models were trained on. Second, Zoom participants often look away from the camera, either to their screens or while gesturing, leading to suboptimal mouth ROI quality. Third, many RoomReader-AV turns are short (”yeah,” ”okay,” ”right”), offering minimal visual context, even for human lipreaders. Finally, the AVSR models were trained primarily on broadcast-style speech, which differs significantly from the more dynamic facial movements seen in meeting scenarios.

To further investigate the impact of these short turns, in Figure 2, we plot the Auto-AVSR model WER for VO, AO, and AV modes against utterance duration. The plot clearly shows that WERs are much higher for short clips and drop rapidly as clips get longer. This observation is strongly supported by recent research from Djilali et al. [21], which highlights that utterances shorter than 2 seconds present an inherent recognition challenge due to limited temporal context. Driven by this observation, we partition the RoomReader-AV into two difficulty subsets using this established 2-second threshold. Clips shorter than 2 seconds form the hard subset: RR-AV Hard (6,752 utterances, 1.6 hours), while clips of 2 seconds or longer form the easy subset: RR-AV Easy (3,572 utterances, 4.9 hours). Clips in the RR-AV Hard average just 2.3 words, compared to 15.9 words in the RR-AV Easy.

Table X benchmarks the models across these subsets. The Hard subset presents a severe recognition challenge across all configurations. Most notably, AO WER effectively doubles for both models when evaluated on the Hard subset. While the visual modality remains extremely challenging across both subsets, it exhibits a similarly pronounced degradation on the Hard subset. These subsets are intended to provide a more challenging evaluation benchmark for AVSR research, reducing over-reliance on controlled datasets such as LRS2 and LRS3 where models have largely saturated performance.

A-3.4k = Auto-AVSR (3,448h); A-1.7k = Auto-AVSR (1,759h); AVH = AV-HuBERT. The TCD-TIMIT value is the averaged score across all Lipspeakers and Volunteers at both 0<sup>◦</sup> and 30<sup>◦</sup> angles. ↓: lower is better.  
TABLE XI  
CROSS-DATASET PERFORMANCE WER (%) ACROSS MODELS AND TRAINING SCALES FOR VIDEO-ONLY MODE.
<table><tr><td>Dataset</td><td>A-3.4k↓</td><td>A-1.7k ↓</td><td>AVH↓</td><td>Llama-AVSR↓</td></tr><tr><td>LRS2</td><td>14.65</td><td>35.05</td><td>38.00</td><td>41.57</td></tr><tr><td>LRS3</td><td>19.10</td><td>31.14</td><td>34.04</td><td>26.20</td></tr><tr><td>GRID</td><td>66.53</td><td>83.92</td><td>88.53</td><td>116.39</td></tr><tr><td>LombardGrid</td><td>78.69</td><td>94.69</td><td>94.80</td><td>145.11</td></tr><tr><td>TCD-TIMIT</td><td>43.18</td><td>59.09</td><td>60.11</td><td>81.91</td></tr><tr><td>RoomReader-AV</td><td>110.07</td><td>113.20</td><td>92.54</td><td>311.48</td></tr></table>

## F. Scaling and Generalization in Visual Speech Recognition

We compare models trained on varying data scales to assess the impact of dataset size and architecture on VO WER. To ensure a fair baseline, we include Auto-AVSR trained on 1,759h, which uses the exact same training data as AV-HuBERT and Llama-AVSR. This setup isolates how well visual features generalize across different datasets and training conditions. As shown in Table XI, data scale heavily impacts VSR performance. Increasing Auto-AVSR’s training data from 1.7k to 3.4k hours leads to consistent, significant reductions in VO WER across most datasets. This improvement plateaus for RoomReader-AV, where performance remains almost unchanged despite doubling the training data. This indicates that while scaling data helps in structured, in-domain scenarios, generalizing to diverse, real-world environments is a fundamental problem, not merely a matter of scale. As doubling broadcast data fails to resolve this out-of-domain visual bottleneck, achieving true generalization will likely require training on conversational ”in-the-wild” data or explicitly decoupling visual representation learning from AV fusion. At the 1.7k-hour scale, the supervised Auto-AVSR demonstrates stronger out-of-domain robustness compared to AV-HuBERT, with RoomReader being the only exception, whereas Llama-AVSR only performed well with seen LRS3.

## V. DISCUSSION

VO does not generalize outside its training domain: Across all tested conditions, VO yields low error rates for in-domain (19–44% WER) and completely collapses on every out-of-domain dataset (66–313% WER). Visual encoders heavily overfit to the specific articulation and vocabulary of professional TED-style broadcast speakers, failing to transfer to other datasets and conditions. A model can achieve a 19% VO WER on LRS3, while failing completely on every other condition; therefore, this performance does not equate to generalized visual speech understanding.

AV fusion is beneficial in two situations: AV mainly improves upon AO in-domain (LRS2, LRS3) and when lip movements are hyper-articulated (LombardGrid). In all other conditions, AV offers little benefit and can significantly worsen AO performance. Current fusion mechanisms blindly trust visual features regardless of quality, injecting noise into the audio pathway rather than down-weighting or ignoring them.

The severe RoomReader AV failure for Llama-AVSR (+40 WER) provides clear evidence of this, while Auto-AVSR’s stability stems from its large-scale supervised training yielding a more robust visual encoder.

Camera pose tolerance is audio dominance, not visual robustness: The LombardGrid angle comparison shows that AV tolerance at a 90<sup>◦</sup> profile view comes mainly from audio fallback. When VO performance collapses at this extreme angle, the AV system relies on audio, maintaining near-identical performance across both views. TCD-TIMIT confirms this: a 30<sup>◦</sup> offset is tolerated because it remains within the visual encoder’s frontal range, whereas a 90° angle renders visual features uninformative and the model falls back to audio.

Speaker articulation matters much more than camera position: The 17 to 44% error gap between professional lipspeakers and everyday volunteers is much larger than the minimal 2% change caused by a 30<sup>◦</sup> camera offset. Since benchmark tests use trained professionals, they do not reflect how normal people talk. Therefore, any claims about realworld AVSR performance must account for this major difference in speaking style.

LLM-based decoders: In-domain gains vs. deployment costs: While Llama-AVSR achieves superior in-domain results, its LLM decoder introduces severe out-of-domain instability, often producing hallucinated, repetitive outputs on short spontaneous inputs. Beyond these accuracy drops, the architecture imposes a massive computational penalty. Since Llama-AVSR shares Auto-AVSR’s codebase, this hardware comparison is direct. Llama-AVSR requires 17.91 GB of GPU memory for AV inference compared to the 2.29 GB needed by Auto-AVSR. Furthermore, while model training costs are outside our current scope, the resource disparity in that phase would be significantly greater. This extreme overhead, coupled with slow autoregressive generation, renders the model largely unsuitable for low-latency, real-time applications.

## VI. CONCLUSION AND FUTURE WORK

This paper presents a unified multi-condition evaluation of AVSR across six diverse datasets, assessing supervised, selfsupervised, and LLM-based architectures. Our work demonstrates that current AVSR progress is narrow in scope. Regardless of architecture or training scale, visual-only models fail to generalize outside of broadcast speech, and apparent visual robustness to extreme camera angles is largely driven by acoustic fallback. Additionally, the use of LLMs introduces new challenges to domain shifts. Evaluation on standard datasets like LRS2 and LRS3 is insufficient to characterize real-world system generalization. Thus, to address this gap and drive the field toward genuine robustness, we introduce RoomReader-AV as a new benchmark for spontaneous multiparty conversational AVSR. Furthermore, we release a preprocessing pipeline that converts this novel dataset, alongside GRID, LombardGrid, and TCD-TIMIT, into formats directly compatible with existing AVSR frameworks, making comprehensive multi-condition evaluation highly accessible and extension to newer ASR/VSR/AVSR systems straightforward.

## AI-GENERATED CONTENT DISCLOSURE

During the preparation of this work, Perplexity (Claude Sonnet 4.6) was used exclusively for minor English grammar corrections and improving the clarity of the written text.

## REFERENCES

[1] C. Sheng, G. Kuang, L. Bai, C. Hou, Y. Guo, X. Xu, M. Pietikainen,¨ and L. Liu, “Deep learning for visual speech analysis: A survey,” IEEE transactions on pattern analysis and machine intelligence, vol. 46, no. 9, pp. 6001–6022, 2024.

[2] M. Kit Khinn Teng, H. Zhang, and T. Saitoh, “Phoneme-level visual speech recognition via point-visual fusion and language model reconstruction,” arXiv e-prints, pp. arXiv–2507, 2025.

[3] K. R. Prajwal, T. Afouras, and A. Zisserman, “Speech Recognition Models are Strong Lip-readers,” in Interspeech 2024, 2024, pp. 2425– 2429.

[4] P. H. Seo, A. Nagrani, and C. Schmid, “Avformer: Injecting vision into frozen speech models for zero-shot av-asr,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 22 922–22 931.

[5] Y. J. Ahn, J. Park, S. Park, J. Choi, and K.-E. Kim, “SyncVSR: Data-Efficient Visual Speech Recognition with End-to-End Crossmodal Audio Token Synchronization,” in Interspeech 2024, 2024, pp. 867–871.

[6] B. Shi, W.-N. Hsu, K. Lakhotia, and A. Mohamed, “Learning audiovisual speech representation by masked multimodal cluster prediction,” in International Conference on Learning Representations (ICLR), 2022.

[7] B. Shi, W.-N. Hsu, and A. Mohamed, “Robust Self-Supervised Audio-Visual Speech Recognition,” in Interspeech 2022, 2022, pp. 2118–2122.

[8] P. Ma, S. Petridis, and M. Pantic, “Auto-avsr: Audio-visual speech recognition with automatic labels,” in 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[9] U. Cappellazzo et al., “Large language models are strong audio-visual speech recognition learners,” in 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[10] T. Afouras, J. S. Chung, A. Senior, O. Vinyals, and A. Zisserman, “ Deep Audio-Visual Speech Recognition ,” IEEE Transactions on Pattern Analysis & Machine Intelligence, vol. 44, no. 12, pp. 8717–8727, Dec. 2022. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/TPAMI.2018.2889052

[11] T. Afouras, J. S. Chung, and A. Zisserman, “Lrs3-ted: a largescale dataset for visual speech recognition,” in arXiv preprint arXiv:1809.00496, 2018.

[12] L. Xia, G. Chen, X. Xu, J. Cui, and Y. Gao, “Audiovisual speech recognition: A review and forecast,” International Journal of Advanced Robotic Systems, vol. 17, no. 6, p. 1729881420976082, 2020.

[13] X. Liu, E. Lakomkin, K. Vougioukas, P. Ma, H. Chen, R. Xie, M. Doulaty, N. Moritz, J. Kolar, S. Petridis et al., “Synthvsr: Scaling up visual speech recognition with synthetic supervision,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18 806–18 815.

[14] F. Su, C. Li, M. Li, and J. Liu, “M2s-avsr: Modality-aware multi-view self-supervised representation for robust audio-visual speech recognition,” 2026. [Online]. Available: https://arxiv.org/abs/2606.05763

[15] W. Tian, M. Shao, B. Mu, X. Geng, C. Wang, Y. Liao, Z. Zhao, Z. Zhang, J. Hu, M. Wei, and L. Xie, “Seeing the context: Rich visual context-aware speech recognition via multimodal reasoning,” 2026. [Online]. Available: https://arxiv.org/abs/2603.07263

[16] Y. Wu, C. Li, S. Yang, Z. Wu, and Y. Qian, “Audio-Visual Multi-Talker Speech Recognition in a Cocktail Party,” in Interspeech 2021, 2021, pp. 3021–3025.

[17] T.-B. Nguyen, T. V. Nguyen, Q. T. Do, and C. M. Luong, “ViCocktail: Automated Multi-Modal Data Collection for Vietnamese Audio-Visual Speech Recognition,” in Interspeech 2025, 2025, pp. 166–170.

[18] T.-B. Nguyen, K. Zmolikova, P. Ma, N. Q. Pham, C. Fuegen, and A. Waibel, “A cocktail-party benchmark: Multi-modal dataset and comparative evaluation results,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 19 502–19 506.

[19] P. Ma, S. Petridis, and M. Pantic, “Visual speech recognition for multiple languages in the wild,” Nature Machine Intelligence, vol. 4, no. 11, pp. 930–939, 2022. [Online]. Available: https://doi.org/10.1038/s42256- 022-00550-z

[20] D. Kwak, J. Choi, S. Lee, and J. S. Chung, “Lrs-voxmm: A benchmark for in-the-wild audio-visual speech recognition,” arXiv preprint arXiv:2604.27866, 2026.

[21] Y. A. D. Djilali, S. Narayan, E. LeBihan, H. Boussaid, E. Almazrouei, and M. Debbah, “Do vsr models generalize beyond lrs3?” in Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2024, pp. 6635–6644.

[22] Y. Huang, J. Xue, L. Jiajun, D. Li, T. Zhang, Z. Yi, Y. Ren, and K. Li, “When avsr meets video conferencing: Dataset, degradation, and the hidden mechanism behind performance collapse,” arXiv preprint arXiv:2603.22915, 2026.

[23] J. Junqua, “The lombard reflex and its role on human listeners and automatic speech recognizers,” The Journal of the Acoustical Society of America, vol. 93, no. 1, pp. 510–524, 01 1993. [Online]. Available: https://doi.org/10.1121/1.405631

[24] S. Nandakishor and D. Pati, “Analysis of lombard effect by using hybrid visual features for asr,” in Pattern Recognition and Machine Intelligence, A. Ghosh, I. King, M. Bhattacharyya, S. Sankar Ray, and S. K. Pal, Eds. Cham: Springer International Publishing, 2024, pp. 328–335.

[25] P. Ma, S. Petridis, and M. Pantic, “Investigating the Lombard Effect Influence on End-to-End Audio-Visual Speech Recognition,” in Interspeech 2019, 2019, pp. 4090–4094.

[26] T.-B. Nguyen, N.-Q. Pham, and A. Waibel, “Cocktail-Party Audio-Visual Speech Recognition,” in Interspeech 2025, 2025, pp. 1828–1832.

[27] H. McGurk and J. MacDonald, “Hearing lips and seeing voices,” Nature, vol. 264, no. 5588, pp. 746–748, 1976.

[28] J. Simko,<sup>ˇ</sup> S. Be<sup>ˇ</sup> nuˇ s, and M. Vainio, “Hyperarticulation in lombardˇ speech: Global coordination of the jaw, lips and the tongue,” The Journal of the Acoustical Society of America, vol. 139, no. 1, pp. 151–162, 01 2016. [Online]. Available: https://doi.org/10.1121/1.4939495

[29] F. Su, C. Li, J. Liu, W. Ju, H. Suo, and M. Li, “Robust llm-based audiovisual speech recognition with sparse modality alignment and visual unit-guided refinement,” arXiv preprint arXiv:2603.03811, 2026.

[30] J. H. Yeo, H. Rha, S. J. Park, and Y. M. Ro, “MMS-LLaMA: Efficient LLM-based audio-visual speech recognition with minimal multimodal speech tokens,” in Findings of the Association for Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 20 724–20 735. [Online]. Available: https://aclanthology.org/2025.findings-acl.1065/

[31] R. Jain and N. Harte, “From hype to insight: Rethinking large language model integration in visual speech recognition,” in Proceedings of 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026.

[32] ——, “The lipreading gap: Do vsr models perceive visual speech like human lipreaders?” 2026. [Online]. Available: https://arxiv.org/abs/2606.07435

[33] Z. Lin, S. Petridis, M. Pantic, and N. Harte, “Assessing true generalisability of audio-visual speech recognisers,” 2026. [Online]. Available: https://arxiv.org/abs/2606.07259

[34] K. R. Prajwal, S. Hegde, and A. Zisserman, “Scaling multilingual visual speech recognition,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[35] Z. Lin and N. Harte, “Uncovering the visual contribution in audiovisual speech recognition,” in 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[36] J. Hong, M. Kim, J. Choi, and Y. M. Ro, “Watch or listen: Robust audiovisual speech recognition with visual corruption modeling and reliability scoring,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18 783–18 794.

[37] J. Reverdy, S. O. Russell, L. Duquenne, D. Garaialde, B. R. Cowan, and N. Harte, “Roomreader: A multimodal corpus of online multiparty conversational interactions,” in Proceedings of the Thirteenth Language Resources and Evaluation Conference, 2022, pp. 2517–2527.

[38] M. Cooke, J. Barker, S. Cunningham, and X. Shao, “An audio-visual corpus for speech perception and automatic speech recognition,” The Journal of the Acoustical Society ofAmerica, vol. 120, no. 5, pp. 2421– 2424, 2006.

[39] N. Alghamdi, S. Maddock, R. Marxer, J. Barker, and G. J. Brown, “A corpus of audio-visual lombard speech with frontal and profile views,” The Journal of the Acoustical Society of America, vol. 143, no. 6, pp. EL523–EL529, 2018.

[40] N. Harte and E. Gillen, “Tcd-timit: An audio-visual corpus of continuous speech,” IEEE Transactions on Multimedia, vol. 17, no. 5, pp. 603–615, 2015.

[41] J. Deng, J. Guo, E. Ververas, I. Kotsia, and S. Zafeiriou, “RetinaFace: Single-shot multi-level face localisation in the wild,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020, pp. 5202–5211.

[42] J. S. Chung and A. Zisserman, “Lip reading in the wild,” in Asian conference on computer vision. Springer, 2016, pp. 87–103.

[43] J. S. Chung et al., “Voxceleb2: Deep speaker recognition,” in Interspeech, 2018, pp. 1086–1090.

[44] A. Ephrat, I. Mosseri, O. Lang, T. Dekel, K. Wilson, A. Hassidim, W. T. Freeman, and M. Rubinstein, “Looking to listen at the cocktail party: a speaker-independent audio-visual model for speech separation,” ACM Trans. Graph., vol. 37, no. 4, Jul. 2018. [Online]. Available: https://doi.org/10.1145/3197517.3201357

[45] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer: Convolutionaugmented Transformer for Speech Recognition,” in Interspeech 2020, 2020, pp. 5036–5040.

[46] S. Watanabe, T. Hori, S. Kim, J. R. Hershey, and T. Hayashi, “Hybrid ctc/attention architecture for end-to-end speech recognition,” IEEE Journal of Selected Topics in Signal Processing, vol. 11, no. 8, pp. 1240–1253, 2017.

[47] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM transactions on audio, speech, and language processing, vol. 29, pp. 3451–3460, 2021.

[48] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28 492–28 518.

[49] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[50] E. J. Hu et al., “Lora: Low-rank adaptation of large language models.” Proceedings of the International Conference on Learning Representations (ICLR), 2022.

[51] J. H. L. Hansen and V. Varadarajan, “Analysis and compensation of lombard speech across noise type and levels with application to inset/out-of-set speaker recognition,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 17, no. 2, pp. 366–378, 2009.

[52] B. Lindblom, “Explaining phonetic variation: A sketch of the h&h theory,” in Speech production and speech modelling. Springer, 1990, pp. 403–439.

[53] Y. Lan, B.-J. Theobald, and R. Harvey, “View independent computer lip-reading,” in 2012 IEEE International Conference on Multimedia and Expo, 2012, pp. 432–437.