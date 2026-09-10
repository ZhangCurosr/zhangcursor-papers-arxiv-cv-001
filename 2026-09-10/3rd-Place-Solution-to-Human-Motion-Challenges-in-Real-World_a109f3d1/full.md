# 3rd Place Solution to Human Motion Challenges in Real-World and Clinical Settings (MoCha) @ECCV2026: Language-Aligned Motion Representations for Domain-Generalizable UPDRS-Gait Severity Estimation

Soojie Kim, Muhammad Munsif, Minkyung Kim, and Seungryul Baek

UNIST, South Korea

{soojie,munsif,minn0219,srbaek}@unist.ac.kr

Abstract. In this work, we introduce language-aligned motion representations for domain-generalizable UPDRS-Gait severity estimation, aiming to learn semantically structured motion features that generalize across heterogeneous clinical domains. We first learn motion representations using a Bi-GRU backbone that captures the temporal dynamics of SMPL sequences. Prior to model training, motion captions are generated ofline using Qwen2.5-7B-Instruct. The backbone is then trained with both classification and text-alignment objectives to learn discriminative and semantically structured motion representations while accounting for the class imbalance present in the training data. We subsequently adapt the learned backbone independently to each source domain so that the model can capture domain-specific motion characteristics. The resulting source-specific models are then merged at the parameter level to consolidate complementary knowledge across source domains into a single domain-generalized model. To further mitigate class imbalance, we perform GPT-5.5-based pseudo labeling, and our final merged models for each site do not use any class-prior correction during inference. The resulting model is evaluated under the unseen-site setting of the MoCha Challenge, using Macro F1 as the primary evaluation metric. Our method achieves a macro-F1 of 0.57 on the hidden test set with only 637K active parameters at inference, ranking 3rd among 58 leaderboard entries in the MoCha 2026 Challenge<sup>1</sup>. The challenge attracted 1,669 submissions from 112 participants and ofered monetary prizes sponsored by Machine Medicine Technologies<sup>2</sup>.

Keywords: Gait analysis, motion analysis, action recognition, cross-site generalization, language-aligned motion representation

## 1 Introduction

Human motion recognition in real-world and clinical environments is challenging because motion distributions can vary across subjects, cohorts, and acquisition sites. Such domain shifts can cause models trained on observed domains to rely on domain-specific patterns and consequently degrade when applied to unseen environments [3,6,8]. To address this issue, we investigate language-aligned motion representations that encourage the learned features to capture semantically meaningful gait characteristics while remaining transferable across heterogeneous clinical domains. The MoCha Challenge addresses this problem through cross-site evaluation on the CARE-PD dataset [1], which provides harmonized SMPL-based gait sequences collected across multiple cohorts and clinical centers.

![](images/f0e06b32b7079fc6899ca83c600958b999ba68e70fbff35853d7663d3096254e.jpg)  
Fig. 1: Comparison of information dependencies among the top-ranked MoCha challenge entries. Unlike the higher-ranked methods, our model does not rely on test-subject metadata, class-weighted training, or test-time adjustment, and predicts directly from motion observations alone. This setting imposes fewer assumptions on the unseen test domain and more closely reflects realistic deployment conditions.

We approach this problem from the perspective of fine-grained motion representation learning. Parkinsonian gait abnormalities often involve subtle temporal and kinematic diferences, making it important to learn representations that capture discriminative motion patterns while remaining robust to domainspecific variations. To this end, we employ a Bi-GRU-based shared backbone to encode temporal dynamics from SMPL sequences. The backbone is optimized using classification and text-alignment objectives, allowing the learned embedding to capture both class-discriminative information and semantic relationships between motion classes. This is particularly relevant in the presence of class imbalance, where conventional supervised learning may become dominated by frequent classes.

After learning the shared representation, we separately adapt the backbone to each source domain. These domain-specific models can capture complementary motion characteristics associated with diferent cohorts, but relying on a single source model may limit generalization to unseen domains. We therefore perform parameter-level model merging [4] to integrate knowledge from the independently adapted models. Rather than treating source domains as interchangeable samples from a single distribution, we regard them as complementary sources of motion knowledge and aim to consolidate this knowledge into a unified model.

Beyond the proposed representation learning framework, our approach also difers from the higher-ranked challenge entries in the information required for prediction. The comparative pipelines in Figure. 1 highlight the key distinctions of our model. While the higher-ranked approaches rely on additional datasetspecific information or adjustment strategies, such as test-subject grouping, class-weighted training, or test-time calibration, our method operates under a more constrained information setting. In particular, we assume that such auxiliary information is unavailable and perform prediction directly from motion observations alone. This setting more closely reflects realistic deployment scenarios, where metadata, class-distribution priors, and test-set-level statistics may not be accessible for previously unseen subjects or clinical sites.

Thus, our overall framework consists of three stages: (1) shared motion representation learning, (2) source-domain-specific adaptation, and (3) parameterlevel model merging. Through this formulation, we investigate whether combining complementary source-domain knowledge can produce a more transferable representation for fine-grained human motion recognition under the unseen-site setting of the MoCha Challenge.

## 2 Methods

Our framework Figure. 2 consists of three stages: caption-aligned motion pretraining, GPT-5.5-based pseudo labeling, and source-specific fine-tuning followed by parameter-level model merging for domain generalization.

## 2.1 Caption-aligned Bi-GRU motion encoder

Each SMPL motion sequence is resampled to 25 FPS and converted into a 263- dimensional HumanML3D representation. Sequences are cropped or padded to 200 frames and normalized before being processed by a single-layer bidirectional GRU. The forward and backward hidden states are concatenated into a 256- dimensional sequence embedding, which is used for four-class UPDRS-gait classification.

To encourage the encoder to focus on clinically meaningful gait characteristics, we additionally align the motion embedding with textual gait descriptions. The captions describe severity-related motion patterns such as walking speed, stride and foot movement, arm swing, limb mobility, and postural stability. A frozen text encoder provides the caption representation, while a projection head maps the Bi-GRU embedding into the corresponding semantic space. A motion reconstruction decoder is also used only during training to preserve fine-grained temporal information.

Captions are generated ofline and once. For each sequence we extract perside knee flexion range, hip swing range and foot-lift height, their left–right asymmetries, cadence, step count and step-time variability, express each relative to its population tertile rather than as an absolute value, and pass the result to Qwen2.5-7B-Instruct [9], which writes one clinical sentence per walk (2,914 distinct captions from 2,915 sequences; no caption names a severity level).

![](images/345f3ad3af1bb91480db56b03f298f2f2783bc55f6278f11f091bb330eba5d70.jpg)  
Fig. 2: Overview of the proposed three-stage framework. (1) Bi-GRU motion representation learning trains two complementary branches with classification together with motion reconstruction or caption-based semantic alignment. (2) GPT-5.5-based pseudo labeling augments unlabeled CARE-PD motions with pseudo UPDRS-gait labels to mitigate class imbalance. (3) Source-specific fine-tuning and model merging independently adapts each pretrained branch to the source domains and consolidates the resulting models through SVD-based parameter merging. The predictions of the two merged branches are averaged to produce the final UPDRS-gait prediction.

## 2.2 GPT-5.5-based pseudo labeling

CARE-PD exhibits substantial class imbalance across UPDRS-gait severity levels. We therefore use additional unlabeled CARE-PD sequences by generating pseudo labels with GPT-5.5.

GPT-5.5 analyzes the SMPL motion and assigns an UPDRS-gait pseudo label based on motion characteristics learned from the labeled CARE-PD examples. The labeling process considers clinically relevant cues including walking speed, step and foot movement, arm swing, body posture, and overall gait restriction. The resulting pseudo-labeled samples are added to their original source datasets to reduce class imbalance, while generated motion descriptions are also used as additional text supervision when available.

## 2.3 Source-specific fine-tuning and model merging

Starting from the same pretrained Bi-GRU model, we independently fine-tune the network on the four CARE-PD source datasets: PD-GaM, BMCLab, T-SDU-PD, and 3DGait. This allows each model to adapt to the characteristics of its own clinical domain while maintaining a common initialization.

After source-specific training, we merge the four independently adapted models at the parameter level. For each source, we compute the parameter change from the common pretrained model and apply singular value decomposition to matrix-valued model deltas. The source updates are represented in a shared basis, where unusually large domain-specific components are suppressed before reconstructing the merged parameters. This procedure is designed to preserve parameter changes that are consistently useful across domains while reducing conflicting source-specific updates [4].

The two motion branches are merged independently. During inference, both merged models predict four-class UPDRS-gait probabilities, and their probability distributions are averaged to obtain the final prediction.

## 3 Experiments

## 3.1 Experimental setup

For each source dataset, namely PD-GaM, BMCLab, T-SDU-PD, and 3DGait, the pretrained weights of Model A and Model B were used as identical initializations. Each source model was then independently fine-tuned for 15 epochs using both ground-truth and pseudo-labeled samples with equal sample weights. We used AdamW optimization with a weight decay of $1 \times 1 0 ^ { - 4 }$ . The learning rate was set to $2 \times 1 0 ^ { - 6 }$ for the Bi-GRU backbone, $7 \times 1 0 ^ { - 5 }$ for the classification and projection heads, and $1 \times 1 0 ^ { - 4 }$ for the training-only decoder and CLIP adapter. The batch size was set to 96 and the random seed to 42. A cosine annealing learning-rate scheduler and gradient clipping with a maximum norm of 5.0 were applied during training.

We did not use validation-based best-checkpoint selection or early stopping. Instead, the final checkpoint from the 15th epoch was retained for each source domain. The resulting source-specific models were merged using SCORE [4] with τ = 1.96 and a merge scale of 1.0. During final inference, the class probabilities predicted by the merged Model A and Model B were averaged with equal weights of 1:1. No post-hoc logit adjustment or class-prior correction was applied.

## 3.2 Results

We achieved 3rd place overall, and the oficial leaderboard results for the top five teams are reported in Table. 1.

In Table. 2, the winning entry [7] exploits the anonymized subject grouping provided with the hidden test set for subject-level posterior aggregation and further applies label-free transductive adjustment using unlabeled test-set statistics. In contrast, the runner-up [2] does not rely on test-set information, but trains its severity classifier using class-weighted cross-entropy. Our final model uses neither test-subject grouping nor test-time adjustment and does not employ class-weighted training. At inference, predictions are obtained directly from the input motion sequence, without auxiliary information from the unseen test cohort.

<table><tr><td>Participants</td><td>F1</td><td>Precision</td><td>Recall</td><td>Acc</td><td>QWK</td></tr><tr><td>1st place (JLShen) [7]</td><td>0.69</td><td>0.72</td><td>0.68</td><td>0.66</td><td>0.60</td></tr><tr><td>2nd place (brady kinesia) [2]</td><td>0.58</td><td>0.65</td><td>0.55</td><td>0.53</td><td>0.41</td></tr><tr><td>3rd place (unist_visionlab)</td><td>0.57</td><td>0.56</td><td>0.59</td><td>0.54</td><td>0.43</td></tr><tr><td>4th place (Nottingham_RVCE)</td><td>0.56</td><td>0.60</td><td>0.53</td><td>0.54</td><td>0.42</td></tr><tr><td>5th place (tuananh1007)</td><td>0.55</td><td>0.60</td><td>0.52</td><td>0.49</td><td>0.43</td></tr></table>

Table 1: Performance comparison on the challenge leaderboard. Our entry achieved 3rd place, alongside results from the top 5 entries. Metrics include Macro-F1, Macro-Precision, Macro-Recall, Accuracy, and Quadratic Weighted Kappa (QWK).

<table><tr><td>Method</td><td>Subject grouping</td><td>Class-weighted</td><td>Test-time adj</td></tr><tr><td>1st place (JLShen) [7]</td><td>√</td><td>x</td><td>√</td></tr><tr><td>2nd place (brady_kinesia) [2]</td><td>x</td><td>√</td><td>x</td></tr><tr><td>3rd place (unist visionlab)</td><td>X</td><td>x</td><td>X</td></tr></table>

Table 2: Comparison of benchmark-specific information and adjustment strategies used by the final top-ranked challenge entries. The winning entry uses the released anonymized subject grouping for subject-level posterior aggregation and applies labelfree transductive adjustment using statistics of the hidden test set [7]. The runner-up trains its severity classifier using class-weighted cross-entropy [2]. In contrast, our final model uses neither the released test-subject grouping nor test-time adjustment, and does not employ class-weighted training. This places our method in a more constrained information setting, where auxiliary metadata and test-set-level statistics are assumed to be unavailable, better reflecting deployment scenarios in which predictions must be made directly from motion observations alone. The row highlighted in blue corresponds to our model.

Table. 3 compares the challenge performance of the pretrained backbone and our full framework. The pretrained model is trained only on the original ground-truth data, which exhibit substantial class imbalance. To compensate for this imbalance at inference time, we apply post-hoc logit adjustment [5] using the class prior estimated from the training set. In contrast, our full framework augments the training data through pseudo labeling, resulting in a substantially more balanced class distribution. Therefore, no post-hoc logit adjustment or class-prior correction is applied to our final model, and predictions are directly obtained from the merged model outputs.

The proposed model consists of two independent lightweight Bi-GRU branches, with approximately 637K parameters actively used during inference. In contrast, the Simple ReCon baseline, which adopts a simple encoder-decoder architecture, uses approximately 17.66M parameters along its inference path. Therefore, the proposed model performs prediction with approximately 27.7× fewer active parameters, reducing both model storage requirements and inference computation. Furthermore, in settings with limited training data and substantial distribution shifts across source domains, excessive model capacity may increase overfitting to source-specific characteristics. Thus, the lightweight architecture may also be beneficial for generalization.

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Pseudo Merging Logit-adj</td><td rowspan="2"></td><td colspan="5">Hidden-site evaluation</td></tr><tr><td>F1</td><td>Precision Recall</td><td></td><td>Acc</td><td>QWK</td></tr><tr><td>Simple ReCon</td><td>√</td><td>x x</td><td>0.53</td><td>0.54</td><td>0.53</td><td>0.53</td><td>0.37</td></tr><tr><td>Simple ReCon</td><td>X</td><td>x √</td><td>0.54</td><td>0.60</td><td>0.51</td><td>0.54</td><td>0.41</td></tr><tr><td>Simple ReCon</td><td>√ √</td><td>x</td><td>0.53</td><td>0.56</td><td>0.53</td><td>0.55</td><td>0.38</td></tr><tr><td>Backbone</td><td>x</td><td>x √</td><td>0.57</td><td>0.60</td><td>0.55</td><td>0.52</td><td>0.41</td></tr><tr><td>Ours</td><td></td><td>X</td><td>0.57</td><td>0.56</td><td>0.59</td><td>0.54</td><td>0.43</td></tr></table>

Table 3: Challenge hidden-site evaluation. F1-score, precision, and recall are reported as macro-averaged metrics across the four UPDRS-gait classes. We additionally compare the efects of pseudo-labeled data, domain-specific model merging, and post-hoc logit adjustment on the final challenge performance.

## 4 Conclusion

In this work, we address UPDRS-gait classification under multi-domain distribution shifts and severe class imbalance. Our framework combines captionaligned Bi-GRU motion pretraining, GPT-5.5-based pseudo labeling, and sourcespecific fine-tuning followed by parameter-level model merging. While the pretrained backbone requires post-hoc logit adjustment to compensate for the imbalanced ground-truth training set, the proposed framework directly performs inference without class-prior correction by constructing a more balanced training set through pseudo labeling. Overall, the proposed approach aims to learn more robust gait representations and reduce source-specific bias for improved generalization to unseen clinical domains.

## References

1. Adeli, V., Klabucar, I., Rajabi, J., Filtjens, B., Mehraban, S., Wang, D., Seo, H., Hoang, T.H., Do, M.N., Muller, C., Oliveira, C., Coelho, D.B., Ginis, P., Gilat, M., Nieuwboer, A., Spildooren, J., Mckay, L., Kwon, H., Cliford, G., Esper, C., Factor, S., Genias, I., Dadashzadeh, A., Shum, L., Whone, A., Mirmehdi, M., Iaboni, A., Taati, B.: Care-pd: A multi-site anonymized clinical dataset for parkinson’s disease gait assessment. In: NeurIPS (2025)

2. Caiola, M., Weitz, A.C.: More motion is not always better motion: Corpus composition governs whether augmentation helps smpl-based parkinsonian gait severity estimation. arXiv preprint arXiv:2608.23730 (2026)

3. Castro, D.C., Walker, I., Glocker, B.: Causality matters in medical imaging. Nature Communications 11, 3673 (2020). https://doi.org/10.1038/s41467-020-17478- w

4. Chaves, L., Zhou, C., Burkholz, R., Valle, E., Avila, S.: Bridging domains through subspace-aware model merging. In: CVPR (2026)

5. Menon, A.K., Jayasumana, S., Rawat, A.S., Jain, H., Veit, A., Kumar, S.: Long-tail learning via logit adjustment. In: ICLR (2021)

6. Pitawela, D., Carneiro, G., Chen, H.T.: Cloc: Contrastive learning for ordinal classification with multi-margin n-pair loss. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 15538–15548 (2025)

7. Shen, J.: Aggregate, don’t adapt: Subject-level posterior aggregation and transductive calibration for cross-site parkinsonian gait severity. arXiv preprint arXiv:2608.20587 (2026)

8. Su, W., Tang, S., Liu, X., Yi, X., Ye, M., Zu, C., Li, J., Zhu, X.: Domain adaptive diabetic retinopathy grading with model absence and flowing data. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 28337–28346 (2025)

9. Team, Q.: Qwen2.5: A party of foundation models (September 2024), https:// qwenlm.github.io/blog/qwen2.5/