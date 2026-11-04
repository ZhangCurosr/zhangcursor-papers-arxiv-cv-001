# GROUNDSEEK-CXR: RETROSPECTIVE EVALUATION-GUIDED ACTIVE OBSERVATION FRAMEWORK FOR MULTI-LABEL CHEST RADIOGRAPHY DIAGNOSIS

Yihang Fu<sup>1,∗</sup>, Xinyu Liu<sup>2,∗</sup>, Qing Xu<sup>3,∗</sup>, Xiaofeng Yang<sup>4</sup>, Zhen Chen<sup>5,✉</sup>

<sup>1</sup>Yale University <sup>2</sup>A\*STAR <sup>3</sup>University of Nottingham <sup>4</sup>University of Chicago <sup>5</sup>The Hong Kong Polytechnic University

## ABSTRACT

Chest radiography (CXR) diagnosis is critical for thoracic disease assessment, yet challenged by the need to identify subtle lesions and cooccurring abnormalities across regions. While recent vision-language models can actively observe local regions, existing preference-based methods supervise each observation by its immediate effect on the current diagnosis, failing to capture how each observation and stopping decision contributes to the final diagnosis. To address this issue, we propose GroundSeek-CXR, an active CXR diagnosis framework that learns observation and stopping decisions by retrospectively evaluating them with the final diagnosis. Specifically, we first devise a Retrospective Diagnostic Evaluation (RDE) that scores observation and stop candidates from offline Monte Carlo tree search through independent continuations. Moreover, we design a Retrospective Gap-weighted Preference Optimization (RGPO) that learns from retrospective-score preferences weighted by their score gaps, prioritizing actions that improve diagnostic accuracy. Extensive experiments demonstrate that our GroundSeek-CXR framework outperforms state-of-the-art methods, and improves abnormal-image sample-F1 by 4.48% over immediate-score preference learning.

Index Terms— Chest X-ray, thoracic disease diagnosis, visionlanguage model, preference optimization

## 1. INTRODUCTION

Chest radiography (CXR) is one of the most widely used imaging examinations for thoracic disease diagnosis, and has attracted significant research attention [1]. Accurate CXR interpretation requires detecting diverse abnormalities, ranging from diffuse findings (e.g., consolidation and pleural effusion) to subtle focal lesions (e.g., small nodules) [2]. However, the projective nature of radiography superimposes ribs, vessels and the mediastinum onto the lung fields, so subtle lesions are easily obscured and overlooked even by experienced radiologists [3]. Moreover, a radiograph often contains co-occurring abnormalities across regions, demanding both global assessment and careful regional inspection [4]. These challenges have given rise to the field of multi-label CXR diagnosis, which seeks to comprehensively identify co-occurring abnormalities.

The classical CXR diagnosis methods [5, 6] formulated this task as multi-label image classification, mapping the entire radiograph to abnormality labels in a single forward pass. More recently, medical vision-language models (VLMs) [7, 8, 9] generated diagnoses with textual reasoning, and grounding methods [10, 11] further associated findings with anatomical regions, but they still perceived the radiograph only once at a fixed resolution. To enable regional observation, visual chain-of-thought methods [12, 13] allowed models to crop and zoom into suspicious regions during reasoning, and ViGoRL and DeepEyes [14, 15] further trained such observation behaviors with reinforcement learning. Moreover, UV-CoT [16] learned region-conditioned reasoning through preference optimization, and ClinCoT [17] introduced clinical hypotheses and score-aware preferences for medical visual reasoning, while MIRA [18] explored reflective regional inspection.

![](images/bbdcfd04451c3026f00c91e7b15163a98567e25f4e2bd801b47f382a162639db.jpg)  
Fig. 1. Illustration of our GroundSeek-CXR for multi-label CXR diagnosis. At each turn, GroundSeek-CXR looks back at the radiograph and zooms into a suspicious region to observe local evidence (Turns 1-2), and stops to give the final diagnosis once the evidence is sufficient (Turn 3). Both observation and stopping decisions are learned by retrospectively evaluating them with the final diagnosis.

Despite these advancements, a fundamental limitation of existing methods lies in how observations are supervised: preferencebased methods [16, 17] evaluate each observation by its immediate effect on the current diagnosis, while reinforcement learning methods [14, 15] share a single outcome reward across all decisions in a trajectory. However, in multi-label CXR diagnosis, an observation without immediate change may still guide subsequent observations toward co-occurring abnormalities. In fact, radiologists follow a hypothetico-deductive process [19], iteratively inspecting regions to confirm or refute hypotheses until sufficient evidence is gathered. Therefore, both the value of each observation and the decision to stop should be judged by the final diagnosis. These insights motivate us to develop a unified framework that retrospectively evaluates both decisions through their subsequent continuations.

![](images/c9982dd21bdaf8fbd639157856ff9e82853d3c320b30b0a22a7de410d4c3bbd7.jpg)  
Fig. 2. The overview of our GroundSeek-CXR framework, consisting of the RDE and the RGPO. The RDE retrospectively scores observation and stop candidates at each search state through their independent continuations from offline MCTS. The RGPO trains the student with score-gap weighted stopping and observation preferences, enabling observation and stopping.

To overcome this bottleneck, we propose GroundSeek-CXR, an active CXR diagnosis framework that learns observation and stopping decisions by retrospectively evaluating them with the final diagnosis, as shown in Fig. 2. Specifically, we design a Retrospective Diagnostic Evaluation (RDE) that leverages offline Monte Carlo tree search (MCTS) with a frozen teacher to propose observation and stop candidates, and evaluates each candidate by the diagnostic utility and observation cost of its independent continuations. We further introduce a Retrospective Gap-weighted Preference Optimization (RGPO) that ranks candidate actions by retrospective scores and weights each preference pair by its score gap, enabling the student to favor actions that improve the final diagnosis. Extensive experiments on VinDr-CXR and CheXpert demonstrate that the proposed GroundSeek-CXR framework outperforms state-of-the-art VLMs, and improves abnormal-image sample-F1 by 4.48% over immediatescore preference learning with higher precision and recall.

## 2. METHODOLOGY

## 2.1. Overview and Problem Formulation

We present GroundSeek-CXR in Fig. 2, and achieve multi-label CXR diagnosis by learning observation and stopping decisions through retrospective evaluation with the final diagnosis. To this end, we design the RDE to propose observation and stop candidates via offline MCTS with a frozen teacher and evaluate them through independent continuations, and the RGPO to train the student with score-gap weighted preference pairs, which enables the student to perform observation and stopping directly at inference without tree search.

Specifically, given a chest radiograph I, the student $\pi _ { \theta }$ maintains a diagnostic state $s _ { t } = ( \hat { Y } _ { t } , \mathcal { U } _ { t } , \mathcal { E } _ { t } , B _ { t } )$ at turn t. Here, $\hat { Y } _ { t }$ is the predicted label set, $\mathcal { U } _ { t }$ contains unresolved hypotheses, and $\mathcal { E } _ { t }$ contains acquired crops. The remaining observation budget $B _ { t }$ is initialized to

K. At each turn, π selects an action $a _ { t } \in { \mathcal { A } } :$

$$
\mathcal A = \{ \mathrm { s t o p } \} \cup \{ \mathrm { o b s e r v e } ( p , z ) \} ,\tag{1}
$$

where $p = ( x , y )$ is an integer-pixel center in the original radiograph, and z describes the intended observation target in free text. Requests call image\_zoom\_in\_tool with coordinates and label. Centered at $p ,$ the tool dynamically crops a rectangle from the original-resolution radiograph. Each observation consumes one budget unit and adds the crop to $\mathcal { E } _ { t + 1 }$ . Action selection and state updates use $\pi _ { \theta } ,$ conditioned on the original radiograph and acquired crops. $\mathbf { A }$ stop action or budget exhaustion ends the episode and returns $\hat { Y } _ { T }$ . Malformed actions or failed crops invalidate the episode.

## 2.2. Retrospective Diagnostic Evaluation

Existing methods supervise observations with either immediate effects [16, 17] or trajectory-level outcome rewards [14, 15], which cannot reveal how a single observation or stopping decision contributes to the final diagnosis. To address this limitation, we design the RDE that retrospectively evaluates each observation and stop candidate by the final diagnosis of its independent continuations.

Specifically, the frozen teacher runs offline MCTS on each training radiograph with upper-confidence selection, progressive widening, and mean-score backup. Each search retains up to M search states, at each of which a stop candidate and at most J observation candidates are proposed. Given the reference labels Y, the diagnostic utility of a predicted label set $\hat { Y }$ and the retrospective score of a continuation τ from turn t are defined as follows:

$$
\begin{array} { r l } & { D ( \hat { Y } , Y ) = \alpha F _ { 1 } ( \hat { Y } , Y ) + ( 1 - \alpha ) \mathrm { E M } ( \hat { Y } , Y ) , } \\ & { \quad \quad U _ { t } ( \tau ) = v D ( \hat { Y } _ { T } , Y ) - \lambda _ { c } N _ { t : T } - \lambda _ { v } ( 1 - v ) , } \end{array}\tag{2}
$$

where $\operatorname { E M } ( { \hat { Y } } , Y )$ equals $1 \mathrm { i f } \hat { Y }$ exactly matches Y and 0 otherwise, $\alpha$ balances the F1 score and the exact match, $v \in \{ 0 , 1 \}$ indicates episode validity, $N _ { t : T }$ counts the observation attempts from turn t onward including the candidate itself, and $\lambda _ { c }$ and $\lambda _ { v }$ weight the observation cost and the invalidity penalty, respectively. The three terms of $U _ { t }$ reward the final diagnosis, impose the observation cost, and penalize invalid episodes. Note that a stop candidate incurs no further observation cost, and the reference labels are only used by this offline evaluator.

Table 1. Comparison with external models on multi-label CXR diagnosis. Macro and micro precision (Pre.), recall (Rec.) and F1 (%) are reported on VinDr-CXR, and macro metrics on CheXpert.
<table><tr><td rowspan="2">Methods</td><td>VinDr (Macro)</td><td>VinDr (Micro)</td><td></td><td rowspan="2">CheXpert</td></tr><tr><td>Pre. Rec. F1</td><td>Pre. Rec.</td><td>F1 Pre. Rec.</td></tr><tr><td>Qwen3.5-4B [20]</td><td>34.82 37.82 36.13</td><td>55.26</td><td>55.88 55.57</td><td>F1 37.98 31.42 33.79</td></tr><tr><td>MedGemma-1.5 [7]</td><td>42.72 46.17 44.31</td><td>61.00</td><td>62.42 61.70</td><td>52.03 45.40 47.94</td></tr><tr><td>CheXagent-2 [9]</td><td>47.93 50.44 49.09</td><td>65.45 66.45</td><td>65.94</td><td>56.28 54.43 54.82</td></tr><tr><td>Lingshu [8]</td><td>46.27 49.02 47.55</td><td>63.87 65.02</td><td>64.44</td><td></td></tr><tr><td>RAD-DINO [6]</td><td>62.13 65.38 63.46</td><td>71.61 69.61</td><td>70.60</td><td>53.45 51.31 51.80 61.12 59.74 60.02</td></tr><tr><td>GroundSeek-CXR</td><td>60.21 61.47 59.75</td><td></td><td>73.06</td><td></td></tr><tr><td></td><td></td><td>72.71</td><td>73.42</td><td>65.13 65.46 64.20</td></tr></table>

Table 2. Comparison with different training strategies on multi-label CXR diagnosis. I and T denote immediate and retrospective ranking, and U denotes uniform weights. Abnormal denotes sample-level metrics on 949 abnormal VinDr-CXR images.
<table><tr><td rowspan="2">Methods</td><td colspan="2">VinDr (Macro)</td><td colspan="2">CheXpert</td><td colspan="2">Abnormal</td></tr><tr><td>Pre. Rec.</td><td>F1</td><td>Pre. Rec.</td><td>F1</td><td>Pre. Rec.</td><td>F1</td></tr><tr><td>Trajectory SFT</td><td>54.21</td><td>55.46 53.76</td><td>58.42 57.63</td><td>56.96</td><td>53.41</td><td>54.17 50.54</td></tr><tr><td>DPO (I/U)</td><td>55.39 56.82 54.97</td><td></td><td>59.37 58.85</td><td>58.12</td><td>54.62 55.48</td><td>51.84</td></tr><tr><td>Immediate DPO (I/I)</td><td>56.48</td><td>58.08 56.21</td><td>60.86 60.15</td><td>59.62</td><td>55.91 56.84</td><td>53.24</td></tr><tr><td>DPO (T/U)</td><td>58.32 59.94</td><td>57.95</td><td>62.89 62.57</td><td>61.96</td><td>58.26 59.12</td><td>55.47</td></tr><tr><td>GroundSeek-CXR (T/T)</td><td>60.21 61.47</td><td>59.75</td><td>65.13</td><td>65.46 64.20</td><td>60.41</td><td>60.66 57.72</td></tr></table>

Moreover, each candidate a at state s receives R independent continuations [21] of the teacher, separate from the tree search. For the r-th continuation, we read out the immediate score $S _ { \mathrm { i m m } } ^ { ( r ) } ( s , a )$ after the state update of a, and the retrospective score $S _ { \mathrm { r e t r o } } ^ { ( r ) } ( s , a )$ after the full continuation, where both apply validity and observation cost up to the readout. The score of each candidate under horizon h ∈ {imm, retro} is averaged as:

$$
\bar { Q } _ { h } ( s , a ) = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } S _ { h } ^ { ( r ) } ( s , a ) .\tag{3}
$$

Since no further action follows a stop candidate, its immediate and retrospective scores are identical. Candidates truncated by the context or generation limit are excluded. In this way, our RDE effectively evaluates each observation and stop candidate by the final diagnosis of its continuations, providing retrospective supervision for the subsequent preference optimization.

## 2.3. Retrospective Gap-weighted Preference Optimization

Although recent methods [16, 17] incorporate score differences into preference optimization, their preferences are derived from the immediate quality of each response, and the decision to stop is rarely compared against further observation. To address this, we design the RGPO that ranks observation and stop candidates by their retrospective scores and weights each preference pair by its score gap.

Specifically, we first initialize the student through trajectory supervised fine-tuning (SFT) on the highest-scoring valid trajectory of each radiograph whose final label set is exactly correct, and adopt it as the frozen reference model $\pi _ { \mathrm { r e f } } .$ . For two candidates a and $a ^ { \prime }$ at the same search state s, including states from unsuccessful trajectories, we compute the score gap $\bar { \delta } _ { h } = \bar { Q } _ { h } ( s , a ) - \bar { Q } _ { h } ( s , a ^ { \prime } )$ for $h \in \{ \mathrm { i m m } , \mathrm { r e t r o } \}$ . A pair is retained only if, under both horizons, $| \bar { \delta } _ { h } | > \epsilon$ and at least κ of the R indexed score differences agree with the sign of $\bar { \delta } _ { h }$ . This shared screening ensures that immediate and retrospective supervision are compared on common candidate pairs, which cover both observation–observation and observation–stop comparisons. Since stop pairs may constitute only a small fraction p of the dataset D, we assign each pair i a base weight $\eta _ { q } ( i )$ and normalize it with the retrospective score gap, as follows:

Table 3. Comparison of false positives, observations per image and mean retrospective score on VinDr-CXR, including 2,051 normal and 949 abnormal images.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">FPR (%)↓</td><td colspan="3">Obs./Image</td><td rowspan="2">Score↑</td></tr><tr><td>Normal Abnormal</td><td></td><td>All</td></tr><tr><td>Trajectory SFT</td><td>5.51</td><td>0.190</td><td>1.096</td><td>0.477</td><td>0.77555</td></tr><tr><td>DPO (I/U)</td><td>5.75</td><td>0.205</td><td>1.212</td><td>0.523</td><td>0.77703</td></tr><tr><td>Immediate DPO (I/I)</td><td>6.00</td><td>0.210</td><td>1.264</td><td>0.543</td><td>0.78004</td></tr><tr><td>DPO (T/U)</td><td>6.58</td><td>0.232</td><td>1.623</td><td>0.672</td><td>0.78084</td></tr><tr><td>GroundSeek-CXR (T/T)</td><td>6.24</td><td>0.219</td><td>1.506</td><td>0.626</td><td>0.79134</td></tr></table>

Table 4. Analysis of successive observations on VinDr-CXR. Abnormal-image sample-level metrics (%) are read out at the initial state, after the first observation and at termination.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Readout</td><td colspan="2">Abnormal</td><td rowspan="2">FPR (%) Obs./Image</td></tr><tr><td>Pre. Rec.</td><td>F1</td></tr><tr><td rowspan="3">Immediate DPO</td><td>Initial</td><td>53.13 52.84</td><td>49.52 5.36</td><td>0.000</td></tr><tr><td>First observation</td><td>54.78 55.32</td><td>52.04 5.85</td><td>0.320</td></tr><tr><td>Final</td><td>55.91 56.84 53.24</td><td>6.00</td><td>0.543</td></tr><tr><td rowspan="3"></td><td>Initial</td><td>53.41 53.18 49.80</td><td>5.41</td><td>0.000</td></tr><tr><td>GroundSeek-CXR First observation</td><td>57.48 58.05 54.84</td><td>6.14</td><td>0.333</td></tr><tr><td>Final</td><td>60.41 60.6657.72</td><td>6.24</td><td>0.626</td></tr></table>

$$
\begin{array} { r } { \eta _ { q } ( i ) = \{ \begin{array} { l l } { q / p , } & { \mathrm { s t o p ~ p a i r } , } \\ { ( 1 - q ) / ( 1 - p ) , } & { \mathrm { o t h e r w i s e } , } \end{array}  } \\ { w _ { i } = \frac { \eta _ { q } ( i ) | \bar { \delta } _ { \mathrm { r e t r o } , i } | } { \mathbb { E } _ { j \sim \mathcal { D } } [ \eta _ { q } ( j ) | \bar { \delta } _ { \mathrm { r e t r o } , j } | ] } , } \end{array}\tag{4}
$$

where $q$ is the target share of stop pairs, and $q = p$ recovers unit base weights. The student $\pi _ { \theta }$ is then optimized with the weighted direct preference optimization (DPO) [22] objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R G P O } } = - \mathbb { E } _ { i \sim \mathcal { D } } \Big [ w _ { i } \log \sigma \big ( \beta \big [ \ell _ { \theta } \big ( a _ { i } ^ { + } \big | s _ { i } \big ) - \ell _ { \theta } \big ( a _ { i } ^ { - } \big | s _ { i } \big ) \big ] \big ) \Big ] , } \end{array}\tag{5}
$$

where $a _ { i } ^ { + }$ and $\boldsymbol { a } _ { i } ^ { - }$ denote the preferred and dispreferred actions ranked by $\bar { \delta } _ { \mathrm { r e t r o } , i } , \ell _ { \theta } = \log \pi _ { \theta } - \log \pi _ { \mathrm { r e f } }$ is computed only over action and termination tokens, and β controls the deviation from $\pi _ { \mathrm { r e f } } .$ . Replacing the retrospective scores with immediate scores in both ranking and weighting yields the immediate-score preference learning baseline. Although only actions receive the preference loss, action selection and state update share the same decoder parameters. In this way, our RGPO effectively aligns both the direction and the strength of action preferences with the final diagnosis, enabling the student to favor observation and stopping decisions that improve multi-label CXR diagnosis under a limited observation budget.

## 3. EXPERIMENTS

## 3.1. Datasets and Implementations

Datasets: To validate the effectiveness of the proposed GroundSeek-CXR, we conduct experiments on two benchmarks: the VinDr-CXR dataset [2], containing 18,000 radiographs annotated with 28 abnormality labels, and the CheXpert dataset [23], from which 202 frontal radiographs of 200 patients are used with five conditions (i.e., atelectasis, cardiomegaly, consolidation, edema and pleural effusion). VinDr-CXR adopts the official split, where 10% of the training set is reserved for development and the test set contains 3,000 images (949 abnormal and 2,051 normal). The CheXpert is only used for external evaluation without training in our experiments.

![](images/474d9ef2b3961a14024fead65333a1e3e648456856368585a1b2f291f3b72271.jpg)  
Fig. 3. Visualization of our GroundSeek-CXR for multi-label CXR diagnosis. GroundSeek-CXR zooms into the suspicious right lung region to observe local evidence before stopping, and correctly identifies pneumonia with lung opacity.

Implementation Details: We adopt Qwen3.5-4B [20] as the student and Qwen3.5-122B-A10B as the frozen teacher on NVIDIA H200 GPUs. After adapting the vision encoder and projector on MIMIC-CXR [24] and VinDr-CXR, only LoRA [25] on the LLM decoder is trained. MCTS runs 24 iterations with $M = 8 , J = 3$ and R = 4 continuations at temperature 0.8. We set $( \alpha , \lambda _ { c } , \lambda _ { v } , \epsilon , \kappa , \beta )$ to (0.75, 0.02, 0.20, 0.01, 3, 0.1) and select q on the development set. K is sampled from {1, 2, 4, 6} in training and set to 6 at inference. RAD-DINO [6] adopts a VinDr-CXR-trained classifier. I, T and U denote immediate ranking, retrospective ranking and uniform weights. Results are averaged over three seeds, with abnormal-image sample-F1 as the primary endpoint and normal-image false-positive rate (FPR) for specificity.

## 3.2. Comparison with State-of-the-art Methods

To comprehensively evaluate the performance of GroundSeek-CXR, we conduct extensive comparisons with state-of-the-art methods on multi-label CXR diagnosis. As shown in Table 1, general and medical VLMs [20, 7, 9, 8] exhibit limited performance, since they perceive the radiograph only once at a fixed resolution. RAD-DINO [6] with a VinDr-CXR-trained classifier achieves the highest VinDr-CXR macro-F1 of 63.46%. In contrast, our GroundSeek-CXR outperforms all VLMs in precision, recall and F1 across three benchmark summaries, and surpasses RAD-DINO by 2.46% micro-F1 on VinDr-CXR and 4.18% macro-F1 on CheXpert. Notably, CheXpert is not used for training, demonstrating the generalization of the learned observation and stopping decisions. As illustrated in Fig. 3, GroundSeek-CXR zooms into the suspicious lung region before the final diagnosis, providing interpretable evidence for its predictions.

Moreover, we compare our GroundSeek-CXR with different training strategies under the same student and SFT initialization, as shown in Table 2. Immediate-score preference learning (Immediate DPO) evaluates each action by its immediate effect, and improves abnormal-image sample-F1 by 2.70% over trajectory SFT. In contrast, the proposed GroundSeek-CXR framework achieves the best performance across all metrics, improving abnormal-image sample-F1 by 4.48% over Immediate DPO with 4.50% and 3.82% gains in precision and recall, respectively. It further improves VinDr-CXR and CheXpert macro-F1 by 3.54% and 4.58%, respectively. These comparisons validate that retrospective evaluation with the final diagnosis provides more effective supervision for observation and stopping decisions than their immediate effects.

## 3.3. Ablation Study

To investigate the effectiveness of the RGPO, we conduct ablation studies on the preference ranking and gap weighting, as illustrated in Table 2 $( 2 ^ { n \hat { d } } - 5 ^ { t h }$ rows), where all variants share the same pairs and training budget. Replacing immediate with retrospective ranking improves abnormal-image sample-F1 by 3.63% and 4.48% under uniform and score-gap weights, respectively, proving that retrospective scores identify more reliable preferred actions. Notably, retrospective ranking with uniform weights even outperforms Immediate DPO by 2.23%. Moreover, score-gap weighting improves sample-F1 by 1.40% and 2.25% under immediate and retrospective ranking, respectively. As shown in Table 3, under retrospective ranking, gap weighting further reduces the observations per image from 0.672 to 0.626 and the normal FPR from 6.58% to 6.24%. These results prove that coordinating retrospective ranking and gap weighting plays a pivotal role in our RGPO, aligning both the direction and the strength of action preferences with the final diagnosis.

## 3.4. Analysis of Successive Observations

To further analyze how observations contribute to the final diagnosis, we read out predictions at the initial state, after the first observation and at termination, as shown in Table 4. Both policies start from similar abnormal-image sample-F1 of 49.52% and 49.80%. GroundSeek-CXR reaches 54.84% after the first observation, and subsequent observations bring a further 2.88% gain, versus 1.20% for Immediate DPO. Moreover, as shown in Table 3, GroundSeek-CXR allocates more observations to abnormal images (1.506 vs. 1.264 per image) while keeping similar observations on normal images (0.219 vs. 0.210), with a slight normal FPR increase from 6.00% to 6.24% and the highest mean retrospective score. These results demonstrate that retrospective evaluation encourages observations whose value emerges in subsequent inspection, consistent with our motivation.

## 4. CONCLUSION

In this work, we identify that existing preference-based active observation methods evaluate each observation by its immediate effect, and propose GroundSeek-CXR, an active CXR diagnosis framework that learns observation and stopping decisions by retrospectively evaluating them with the final diagnosis. The framework integrates the RDE that leverages offline MCTS with a frozen teacher to evaluate observation and stop candidates through independent continuations, and the RGPO that ranks candidate actions by retrospective scores and weights each preference pair by its score gap. Extensive experiments demonstrate that the proposed GroundSeek-CXR framework outperforms state-of-the-art VLMs and immediate-score preference learning with higher precision and recall, validating the effectiveness of retrospective evaluation for multi-label CXR diagnosis.

## 5. ACKNOWLEDGMENTS

This work is supported by PolyU Start-up Fund (P0060371).

## 6. REFERENCES

[1] Erdi Çallı, Ecem Sogancioglu, Bram van Ginneken, Kicky G van Leeuwen, and Keelin Murphy, “Deep learning for chest Xray analysis: A survey,” Med. Image Anal., vol. 72, pp. 102125, 2021.

[2] Ha Q Nguyen, Khanh Lam, Linh T Le, Hieu H Pham, Dat Q Tran, Dung B Nguyen, Dung D Le, Chi M Pham, Hang TT Tong, Diep H Dinh, et al., “VinDr-CXR: An open dataset of chest X-rays with radiologist’s annotations,” Sci. Data, vol. 9, no. 1, pp. 429, 2022.

[3] Trafton Drew, Melissa L-H Võ, and Jeremy M Wolfe, “The invisible gorilla strikes again: Sustained inattentional blindness in expert observers,” Psychol. Sci., vol. 24, no. 9, pp. 1848– 1853, 2013.

[4] Harold L Kundel and Calvin F Nodine, “Interpreting chest radiographs without visual search,” Radiology, vol. 116, no. 3, pp. 527–532, 1975.

[5] Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Robyn L Ball, Curtis Langlotz, et al., “CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning,” arXiv preprint arXiv:1711.05225, 2017.

[6] Fernando Pérez-García, Harshita Sharma, Sam Bond-Taylor, Kenza Bouzid, Valentina Salvatelli, Maximilian Ilse, Shruthi Bannur, Daniel C Castro, Anton Schwaighofer, Matthew P Lungren, et al., “Exploring scalable medical image encoders beyond text supervision,” Nat. Mach. Intell., vol. 7, pp. 119– 130, 2025.

[7] Andrew Sellergren, Chufan Gao, Fereshteh Mahvar, Timo Kohlberger, Fayaz Jamil, Madeleine Traverse, Alberto Tono, Bashir Sadjad, Lin Yang, Charles Lau, et al., “MedGemma 1.5 technical report,” arXiv preprint arXiv:2604.05081, 2026.

[8] Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, et al., “Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning,” arXiv preprint arXiv:2506.07044, 2025.

[9] Zhihong Chen, Maya Varma, Jean-Benoit Delbrouck, Magdalini Paschali, Louis Blankemeier, Dave Van Veen, Jeya Maria Jose Valanarasu, Alaa Youssef, Joseph Paul Cohen, Eduardo Pontes Reis, et al., “CheXagent: Towards a foundation model for chest X-ray interpretation,” arXiv preprint arXiv:2401.12208, 2024.

[10] Huihui Xu, Yuanpeng Nie, Hualiang Wang, Ying Chen, Wei Li, Junzhi Ning, Lihao Liu, Hongqiu Wang, Lei Zhu, Jiyao Liu, et al., “MedGround-R1: Advancing medical image grounding via spatial-semantic rewarded group relative policy optimization,” in MICCAI, 2025.

[11] Razi Mahmood, Diego Machado-Reyes, Joy Wu, Parisa Kaviani, Ken CL Wong, Niharika D’Souza, Mannudeep Kalra, Ge Wang, Pingkun Yan, and Tanveer Syeda-Mahmood, “Phrasegrounded fact-checking for automatically generated chest X-ray reports,” in MICCAI, 2025.

[12] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li, “Visual CoT: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning,” in NeurIPS, 2024.

[13] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna, “Visual Sketchpad: Sketching as a visual chain of thought for multimodal language models,” in NeurIPS, 2024.

[14] Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J Tarr, Aviral Kumar, and Katerina Fragkiadaki, “Grounded reinforcement learning for visual reasoning,” in NeurIPS, 2025.

[15] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu, “DeepEyes: Incentivizing “thinking with images” via reinforcement learning,” in ICLR, 2026.

[16] Kesen Zhao, Beier Zhu, Qianru Sun, and Hanwang Zhang, “Unsupervised visual chain-of-thought reasoning via preference optimization,” in ICCV, 2025.

[17] Xiwei Liu, Yulong Li, Xinlin Zhuang, Xuhui Li, Jianxu Chen, Haolin Yang, Imran Razzak, and Yutong Xie, “ClinCoT: Clinical-aware visual chain-of-thought for medical vision language models,” arXiv preprint arXiv:2603.01124, 2026.

[18] Shengzhi Wang, Jun Yang, Kai Wu, Xiaozhong Ji, Yiwen Ye, Ziyang Chen, Mingliang Xiong, Wen Fang, Mingqing Liu, Mengyuan Xu, et al., “MIRA: Medical image reflection for agentic diagnosis,” arXiv preprint arXiv:2608.10827, 2026.

[19] Arthur S Elstein, Lee S Shulman, and Sarah A Sprafka, Medical problem solving: An analysis of clinical reasoning, Harvard University Press, 1978.

[20] Qwen Team, “Qwen3.5: Towards native multimodal agents,” https://www.alibabacloud.com/blog/ qwen3-5-towards-native-multimodal-agents\_ 602894, 2026.

[21] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui, “Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations,” in ACL, 2024, pp. 9426–9439.

[22] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn, “Direct preference optimization: Your language model is secretly a reward model,” in NeurIPS, 2023.

[23] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al., “CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison,” in AAAI, 2019.

[24] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng, “MIMIC-CXR, a deidentified publicly available database of chest radiographs with free-text reports,” Sci. Data, vol. 6, no. 1, pp. 317, 2019.

[25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, “LoRA: Low-rank adaptation of large language models,” in ICLR, 2022.