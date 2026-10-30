# ANNOTATION-FREE REPRESENTATION LEARNING FOR POLYP RETRIEVAL IN COLONOSCOPY VIDEOS

Luca Parolari<sup>1</sup> Lamberto Ballan<sup>1</sup> Carlo Biffi<sup>2,†</sup> Loic Le Folgoc<sup>3,†</sup>

<sup>1</sup> Department of Mathematics, University of Padova, Padova, Italy <sup>2</sup> Cosmo Intelligent Medical Devices, Dublin, Ireland <sup>3</sup> LTCI, Telecom Paris, Institut Polytechnique de Paris, Palaiseau, France

## ABSTRACT

Colonoscopy is inherently comparative: reliable visual assessment of polyps often depends on complementary views acquired under different viewpoints and imaging conditions. Retrieving such views requires polyp tracklet representations, which recent contrastive methods learn from temporal cues but still build from manually annotated bounding boxes. We propose an annotation-free framework that constructs tracklet views directly from raw colonoscopy videos, enabling temporal self-supervised learning on full-procedure videos, including frames without visible polyps. On REAL-Colon, our method achieves state-of-the-art results on tracklet retrieval and re-identification, outperforming both self-supervised and supervised baselines. We then introduce few-shot polyp retrieval, a task closer to clinical use, where a few observations of a target polyp are used to retrieve its remaining tracklets: with only two examples, our method surpasses most competing methods given ten. Finally, we show that a modest number of unlabeled procedures suffices for competitive performance, making on-site model development more practical. Code is available at https://github.com/lparolari/freerep.

Index Terms— Contrastive learning, self-supervised learning, annotation-free, retrieval, colonoscopy videos

## 1. INTRODUCTION

Colonoscopy is the gold standard for the prevention and early detection of colorectal cancer [1]. Deep learning increasingly supports it through computer-aided polyp detection [2] and characterization [3]. Polyp retrieval aims to recover additional views of a target lesion from colonoscopy videos, starting from one or a few observations. Aggregating information across these complementary views could support polyp characterization and facilitate lesion assessment and reporting during post-procedure review [4, 5]. However, learning representations of polyp tracklets—sequences of consecutive detections of the same polyp—is challenging: different polyps may look similar, while the same polyp may vary substantially in appearance over time [6]. Representations must therefore be robust to appearance changes while remaining discriminative across polyps.

Supervised contrastive learning on polyp entity annotations, i.e., labels linking observations of the same polyp, yields strong tracklet representations [7]. Yet, such labels are costly to collect, as they require expert clinicians. Temporal self-supervised methods reduce this cost by deriving positive pairs from temporal continuity: by splitting tracklets into temporal segments [8, 9], by integrating textual reports through momentum contrast [10], or by sampling pseudo-positives among temporally close tracklets of the same procedure [11].

These approaches nevertheless rely on ground-truth bounding boxes to construct polyp-centered training inputs, providing explicit spatial supervision during representation learning. Such annotations are costly and time-consuming to obtain. They also restrict training to annotated polyp observations, excluding the large majority of each procedure, namely frames without visible polyps, which still carry useful information about normal colonic appearance.

In this work, we learn polyp tracklet representations directly from raw colonoscopy videos, without bounding boxes or polyp identity labels. Since tracklets can no longer be polyp-centered, we redefine them as sequences of consecutive frames. However, our experiments show that global frame-level views underperform because of the lack of locality. Therefore, we adopt a temporally consistent random cropping strategy that restores locality without annotations. This enables temporal self-supervision on full-procedure videos and yields state-of-the-art results on REAL-Colon for tracklet retrieval and re-identification. We then introducefewshot polyp retrieval, which retrieves the remaining tracklets of a lesion from a few examples, as in post-procedure review. With only two support tracklets, our method outperforms most baselines that use ten.

Our contributions are: (i) an annotation-free framework that achieves state-of-the-art results by learning directly from raw full-procedure videos; (ii) a novel few-shot polyp retrieval protocol; (iii) analyses of the trade-off between spatial supervision and data scale, and of cross-cohort robustness.

## 2. ANNOTATION-FREE REPRESENTATION LEARNING FRAMEWORK

We train a visual encoder by contrastive learning without the two sources of supervision of polyp-centric settings: groundtruth polyp detections and polyp entity annotations. Temporally consistent crops of raw frames recover the locality of polyp-centered views (Sec. 2.1), extending training to fullprocedure videos, including frames without visible polyps (Sec. 2.2). Pseudo-positive associations derived from temporal proximity within each video replace polyp entity labels (Sec. 2.3), enabling annotation-free representation learning.

## 2.1. Annotation-Free Tracklet Views

Polyp-centric frameworks build each training sample from ground-truth polyp detections. Without them, we redefine a tracklet as a short sequence of consecutive frames from the same video, whether or not a polyp is visible.

The most direct view is then a sequence of full frames, which removes the spatial prior that focused the encoder on the polyp. However, this full-frame baseline loses 15% (relative) with respect to the crop-based state-of-the-art trained with ground-truth detections. Full frames shift the visual domain toward global endoscopic views with large background regions, instruments, specular highlights, and blur.

We therefore keep local views but sample them without detections: a random crop is drawn once per tracklet and applied to all its frames, preserving spatial coherence. This yields a 27% relative improvement over full frames, surpasses the crop-based state-of-the-art by 8%, and performs comparably to a synthetic box crop based on dataset-level bounding-box statistics, which, unlike ours, requires annotations (Tab. 2).

## 2.2. Full-Procedure Training

Since views no longer depend on detections, training is not confined to annotated polyp tracklets and extends to raw full-procedure videos, including the mucosal patterns, lumen views, and other colonoscopy-specific content that polypcentric methods discard.

We build the training set from REAL-Colon, which contains 60 full-procedure videos from four acquisition cohorts. We follow the split of [11], but retain the full content of its 27 training videos with polyps and add the 14 videos without visible polyps that it omitted. The resulting 41-video training set, a superset of the previously used annotated data, contains 1,725,537 frames, of which only 171,957 (approximately 10%) show at least one polyp.

## 2.3. Temporal Self-Supervision

We optimize the tracklet encoder with the temporal selfsupervised framework of [11], which derives pseudo-positive associations from the temporal structure of colonoscopy videos rather than from polyp identity labels. We leave the objective unchanged and modify only its input domain: tracklets are no longer required to contain a polyp, which allows training on full-procedure videos.

Formally, each training tracklet $x _ { i }$ has a source video $v _ { i }$ and a temporal position $p _ { i }$ . For an anchor $a _ { i } ,$ the candidate pseudo-positives are the tracklets of the same video $( v _ { j } = v _ { i } )$ ranked by temporal distance. We sample κ of them without replacement from an exponential distribution over ranks with temperature τ, obtaining a bag $B _ { i } = \{ b _ { i k } \} _ { k = 1 } ^ { \kappa }$ a curriculum increases τ in training, from close to broader context.

We minimize the noise-aware contrastive loss [11]:

$$
\mathcal { L } = - \sum _ { i } \log \frac { \sum _ { k } e ^ { s ( z _ { i k } , y _ { i } ) } } { \sum _ { k } e ^ { s ( z _ { i k } , y _ { i } ) } + \sum _ { j \neq i } \sum _ { k } e ^ { s ( z _ { i k } , y _ { j } ) } } ,
$$

where $y _ { i } = f _ { \theta } ( a _ { i } )$ and $z _ { i k } = f _ { \theta } ( b _ { i k } )$ are the encoder representations of the anchor and of the k-th element of its bag, and $s ( \cdot , \cdot )$ is a similarity function. Summing over the bag lets the model weight the most relevant pseudo-positives instead of forcing all temporal neighbors to match the anchor equally.

## 3. EXPERIMENTAL RESULTS

## 3.1. Dataset and Downstream Setup

We use REAL-Colon [19], which contains 60 full-procedure colonoscopy videos from four acquisition cohorts recorded at six clinical centers on three continents with equipment from two manufacturers. We follow the setup of [11]. Training uses 41 videos without any annotation (Sec. 2.2); evaluation uses the 19-video test set (47 polyps), which contains only tracklets with visible polyps.

Tracklet retrieval and ReID Following [11], each test tracklet is used in turn as a query, and the remaining tracklets are ranked by the cosine similarity of their embeddings to it; we report mAP and HR@1, the fraction of queries whose topranked tracklet depicts the same polyp. For re-identification, the cosine similarity of each tracklet pair is thresholded to separate same-polyp from different-polyp pairs, and performance is summarized by AUROC and AUPR.

Few-shot polyp retrieval Tracklet retrieval treats each tracklet as an independent query, whereas in clinical review an expert may hold one or a few observations of a polyp and search for its other appearances. We therefore retrieve at the polyp level: for each target polyp, we randomly sample a support set $\boldsymbol { S } = \{ s _ { 1 } , \ldots , s _ { K } \}$ of $K \in \{ 1 , 2 , 3 , 5 , 1 0 \}$ of its tracklets and rank each gallery tracklet $g ~ \in ~ \mathcal { G }$ by sim ${ \it \Psi } ( g , S ) = \mathrm { m a x } _ { s _ { i } \in S } \cos ( f _ { \theta } ( g ) , f _ { \theta } ( s _ { i } ) )$ , keeping the support tracklets as separate exemplars. Random sampling models an average review scenario rather than a curated selection; we repeat it R = 50 times and report the mean mAP.

Table 1: Comparison with State-of-the-Art: our approach outperforms self-supervised, weakly supervised, and fully supervised baselines. Bold: best; underline: second best.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Params (M)</td><td rowspan="2">FLOPs (G)</td><td colspan="2">Tracklet Retrieval mAP</td><td colspan="2">Tracklet ReID AUROC AUPR</td><td rowspan="2"></td><td colspan="4">Few-Shot Polyp Retrieval</td></tr><tr><td>(↑)</td><td>HR@1 (↑)</td><td>(↑)</td><td>(↑) K=1</td><td>K=2</td><td>mAP (↑) K=3</td><td>K=5</td><td>K=10</td></tr><tr><td colspan="2">General Purpose FM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DinoV2-S/14 [12]</td><td>22.1</td><td>44.2</td><td>47.31</td><td>89.81</td><td>80.07</td><td>29.77</td><td>47.91</td><td>57.04</td><td>62.73</td><td>65.64</td><td>72.57</td></tr><tr><td>DinoV2-g/14 [12]</td><td>1136</td><td>2331</td><td>48.29</td><td>90.35</td><td>78.23</td><td>27.04</td><td>48.08</td><td>57.20</td><td>62.40</td><td>66.46</td><td>72.13</td></tr><tr><td>DinoV3-S/16 [13]</td><td>21.6</td><td>34.7</td><td>48.95</td><td>88.99</td><td>80.47</td><td>31.77</td><td>47.60</td><td>57.36</td><td>62.95</td><td>66.80</td><td>76.59</td></tr><tr><td>V-JEPA2 [14]</td><td>326.0</td><td>272.1</td><td>29.01</td><td>70.79</td><td>69.84</td><td>17.14</td><td>22.29</td><td>28.46</td><td>32.70</td><td>33.38</td><td>39.93</td></tr><tr><td colspan="2">Endoscopy FM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EndoFM [15]</td><td>121.3</td><td>190.1</td><td>45.75</td><td>87.64</td><td>79.71</td><td>31.91</td><td>45.12</td><td>56.28</td><td>61.80</td><td>65.46</td><td>73.03</td></tr><tr><td>EndoFM-LV [16]</td><td>121.3</td><td>190.1</td><td>28.21</td><td>63.18</td><td>67.56</td><td>19.28</td><td>22.54</td><td>28.22</td><td>28.92</td><td>30.95</td><td>38.22</td></tr><tr><td>EndoViT [17]</td><td>86.6</td><td>134.9</td><td>31.12</td><td>68.34</td><td>68.26</td><td>17.51</td><td>28.18</td><td>35.60</td><td>38.68</td><td>40.77</td><td>47.77</td></tr><tr><td>SurgeNet [18]</td><td>24.3</td><td>31.0</td><td>44.75</td><td>85.87</td><td>75.86</td><td>25.23</td><td>47.14</td><td>58.25</td><td>63.37</td><td>68.18</td><td>75.87</td></tr><tr><td colspan="2">Fully supervised</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sup [7]</td><td>25.6</td><td>33.1</td><td>54.72</td><td>81.93</td><td>93.85</td><td>41.58</td><td>50.60</td><td>55.75</td><td>58.23</td><td>59.60</td><td>63.22</td></tr><tr><td>TA [7]</td><td>25.6</td><td>33.1</td><td>57.52</td><td>84.24</td><td>94.33</td><td>43.81</td><td>53.12</td><td>57.94</td><td>60.49</td><td>62.33</td><td>68.52</td></tr><tr><td colspan="2">Weakly supervised</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MVE [8]</td><td>25.6</td><td>33.1</td><td>42.05</td><td>81.25</td><td>79.73</td><td>27.47</td><td>48.41</td><td>56.25</td><td>58.47</td><td>60.58</td><td>65.89</td></tr><tr><td>TPC [9]</td><td>25.6</td><td>33.1</td><td>43.25</td><td>75.82</td><td>78.40</td><td>27.58</td><td>51.15</td><td>59.46</td><td>63.00</td><td>65.51</td><td>69.90</td></tr><tr><td>NTSSL [11]</td><td>25.6</td><td>33.1</td><td>61.49</td><td>91.17</td><td>93.89</td><td>51.78</td><td>59.24</td><td>67.64</td><td>71.36</td><td>74.69</td><td>81.34</td></tr><tr><td colspan="2">Self-supervised</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MVE [8]</td><td>25.6</td><td>33.1</td><td>46.22</td><td>80.43</td><td>89.74</td><td>31.93</td><td>44.69</td><td>51.67</td><td>54.03</td><td>54.62</td><td>59.30</td></tr><tr><td>TPC [9]</td><td>25.6</td><td>33.1</td><td>49.94</td><td>84.65</td><td>91.60</td><td>35.66</td><td>46.96</td><td>54.73</td><td>57.49</td><td>58.79</td><td>62.83</td></tr><tr><td>Ours</td><td>25.6</td><td>33.1</td><td>66.47</td><td>92.80</td><td>95.03</td><td>54.91</td><td>63.46</td><td>73.10</td><td>75.58</td><td>78.80</td><td>85.72</td></tr></table>

## 3.2. Implementation Details

The tracklet encoder follows prior work [8, 9, 7, 11]: an ImageNet-pretrained ResNet-50 [20] extracts frame features, which a three-layer transformer encoder (8 heads, $d _ { \mathrm { m o d e l } } = 2 5 6 )$ aggregates and a two-layer MLP (256 → 128) projects for contrastive learning. We use κ = 4 and a batch size of 60. The curriculum schedules τ in the exponential distribution from $\tau _ { \mathrm { m i n } } = 0 . 3 \ \mathrm { t o } \ \tau _ { \mathrm { m a x } } = 1 2$ , using a cosine scheduler [11]. Videos are temporally subsampled by a factor of four, and a tracklet spans L = 8 subsampled frames (about one second); views are random resized crops (scale range (0.08, 1), 232 × 232 pixels) sampled once per tracklet, with no further augmentation. We train for 25 epochs with AdamW (learning rate $1 0 ^ { - 5 }$ , weight decay 10<sup>−4</sup>) on a single NVIDIA A100 GPU. At test time, as in [9, 7, 11], tracklets are built by linking ground-truth detections with an IoU of at least 0.1, each enlarged to 5× the bounding-box diagonal.

## 3.3. Comparison with State-of-the-Art

Tab. 1 compares our approach with the state-of-the-art. Selfsupervised baselines are re-trained on our training set, while weakly and fully supervised baselines use its annotated subset. Our framework outperforms previous self-supervised approaches by 33.10% in retrieval mAP and by 3.74% and 53.98% in ReID AUROC and AUPR (relative). MVE [8] and TPC [9] draw positive views from distinct segments of a tracklet, presuming a polyp that stays visible throughout, which rarely holds for generic frame sequences.

Our method also surpasses NTSSL [11], a weakly supervised method that shares our objective but relies on bounding boxes, and the fully supervised baselines on all metrics; against TA [7], our method slightly improves AUROC and improves AUPR by 25.34%, i.e., with fewer false positives. Foundation models give strong generic features but are not optimized for the temporal, low-semantics structure of colonoscopy; endoscopy-specific ones rarely beat generalpurpose ones, consistent with their smaller scale and surgical training data.

Few-shot polyp retrieval All methods benefit from additional support observations, confirming that single views of a polyp are ambiguous. Moving from K = 1 to K = 2 raises our mAP from 63.46 to 73.10, which already exceeds the K = 10 performance of nearly all baselines; the only exceptions are DinoV3-S/16, SurgeNet, and NTSSL, which uses ground-truth boxes during training. With K = 10, our method reaches 85.72 mAP.

## 3.4. Supervision vs. Scaling

We next ask how much unlabeled full-procedure video is needed to offset the removal of spatial supervision. We train on nested subsets of the 41-video training split (4, 8, 16, 24, 32, and 41 videos), with five independent subsets per size to average out the choice of videos. Fig. 1 compares the resulting models with NTSSL [11], which uses ground-truth boxes on the full training set.

Performance grows steadily with the number of videos: the annotation-free model matches NTSSL at around 24 videos on retrieval and AUPR, and at 32 videos on AUROC, and keeps improving up to the full 41-video set. In line with scaling results on natural images [21], more unlabeled videos can thus be more effective than fewer spatially annotated polyp observations. This break-even point is specific to REAL-Colon and should not be read as a universal threshold.

![](images/d8cb09553b75eb58f76edb716fd9b04ecf46d0aa896db2012e0def049323b27e.jpg)  
Fig. 1: Scaling of annotation-free training with the number of training videos (nested subsets; shaded band: ± std over five subsets per size). Dashed lines: NTSSL [11]. The break-even point is reached between 24 and 32 videos, depending on the metric, and performance keeps improving up to 41 videos.

## 3.5. Robustness to Cohort Removal

We assess robustness to unseen acquisition domains by retraining our model from scratch with each of the four REAL-Colon cohorts withheld in turn (leave-one-cohort-out, LOCO) and evaluating it on the unchanged test set. As a matched control, we use the 32-video random subsets of Sec. 3.4, which remove a comparable amount of data drawn from all cohorts. In Fig. 2, LOCO averages 62.90 mAP, above NTSSL (61.49), which uses ground-truth boxes and every cohort, and close to the control (64.11). Re-identification is slightly more sensitive (92.32 against 93.95 AUROC for the control). Withholding a cohort thus costs about as much as removing the same amount of data, and in mAP it still surpasses NTSSL trained on all cohorts, suggesting resilience to cohort removal.

## 3.6. Ablation Study

Tab. 2 traces the path from polyp-centric supervised learning to our annotation-free setting. Replacing identity labels with temporal self-supervision and the noise-aware loss raises retrieval mAP from 57.52 to 61.49 and ReID AUPR from 43.81 to 51.78, showing that temporal associations are an effective supervisory signal when their noise is addressed. Switching to random crops on full-procedure videos with the same loss improves every metric further, reaching 66.47 mAP and 54.91

![](images/2c8ba4c2e70efc03ea7a3cff6d5b612ba5c2ca429ac0ea35f70de370091166a5.jpg)  
Fig. 2: Robustness to cohort removal: random subsets (five 32-video subsets drawn from all cohorts) against leave-onecohort-out (LOCO) training. Differences between the two protocols are not significant (Welch’s t-test, p-values below each panel), indicating no large effect of removing a cohort.

Table 2: Ablation toward annotation-free training. Annot.: polyp identities and boxes (ID+Box), boxes (Box), box statistics (Stats), or none (–); Loss: supervised contrastive (SupCon) or noise-aware (Noise-Aw.); Views: ground-truth box crops (BBox), synthetic box crops (Synth.), full frames (Frame), or random crops (Crop); Full: full-procedure training. Gray: annotation-dependent; bold: best annotation-free.
<table><tr><td rowspan="2">Annot.</td><td colspan="3">Training setup</td><td rowspan="2">mAP</td><td rowspan="2">Tracklet Retrieval HR@1</td><td rowspan="2">Tracklet ReID AUROC AUPR</td></tr><tr><td>Loss</td><td>Views</td><td>Full</td></tr><tr><td>ID+Box</td><td>SupCon</td><td>BBox</td><td>x</td><td>57.52</td><td>84.24</td><td>94.33 43.81</td></tr><tr><td rowspan="3">Box Stats</td><td>Noise-Aw.</td><td>BBox</td><td>x</td><td>61.49</td><td>91.17</td><td>93.89 51.78</td></tr><tr><td>Noise-Aw.</td><td>Synth.</td><td>√</td><td>65.60</td><td>91.71</td><td>95.14 57.72</td></tr><tr><td></td><td></td><td>√</td><td></td><td></td><td>43.74</td></tr><tr><td>一</td><td>Noise-Aw.</td><td>Frame</td><td></td><td>52.27</td><td>84.10</td><td>89.24</td></tr><tr><td>一</td><td>SupCon</td><td>Crop</td><td>√</td><td>64.31</td><td>89.54</td><td>94.97 49.83</td></tr><tr><td>1</td><td>Noise-Aw.</td><td>Crop</td><td>√</td><td>66.47</td><td>92.80</td><td>95.03 54.91</td></tr></table>

AUPR. A supervised contrastive loss over the same temporal bags remains competitive (64.31 mAP), so the temporal sampling itself provides useful structure, but it lowers HR@1 and AUPR. Full frames perform worst on every metric, while synthetic boxes, which rely on annotation statistics, trade lower retrieval scores for higher ReID AUPR.

## 4. CONCLUSION

We presented an annotation-free framework that learns polyp tracklet representations from raw full-procedure colonoscopy videos. Without bounding-box or identity annotations, it achieves state-of-the-art results on REAL-Colon for tracklet retrieval, re-identification, and few-shot polyp retrieval, using only a modest number of unlabeled procedures. Validation is currently confined to REAL-Colon: although it spans six centers on three continents and two manufacturers, it is the only public full-procedure dataset with persistent polyp identities, and we hope new datasets will enable broader validation.

## 5. DECLARATIONS

Ethics. This retrospective study uses open-access human subject data from REAL-Colon [19]; ethical approval was not required, as confirmed by the data license.

Competing interests. C.B. is affiliated with Cosmo Intelligent Medical Devices, developer of the GI Genius device.

Acknowledgments. We acknowledge ISCRA for awarding this project access to the LEONARDO supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CINECA (Italy). AI disclosure. Claude (Anthropic) was used to improve the writing and to prepare the charts; the authors reviewed and take responsibility for all content.

## 6. REFERENCES

[1] Cesare Hassan, Marco Spadaccini, Andrea Iannone, et al., “Performance of artificial intelligence in colonoscopy for adenoma and polyp detection: A systematic review and meta-analysis,” Gastrointest. Endosc., vol. 93, no. 1, pp. 77–85, 2021.

[2] Jason A. Dominitz, Andrew J. Gawron, Grace B. Mc-Kee, et al., “Impact of availability of computeraided detection devices on adenoma detection during colonoscopy: A cluster randomized study,” Gastroenterology, vol. 171, no. 5, pp. 870–884, 2026.

[3] Cesare Hassan, Tommy Rizkala, Yuichi Mori, et al., “Computer-aided diagnosis for the resect-and-discard strategy for colorectal polyps: A systematic review and meta-analysis,” Lancet Gastroenterol. Hepatol., vol. 9, no. 11, pp. 1010–1019, 2024.

[4] Tatsushi Naito, Takuto Nosaka, Tomoko Tanaka, et al., “Usefulness of an artificial intelligence-based colonoscopy report generation support system,” Clin. Endosc., vol. 58, no. 2, pp. 327–330, 2025.

[5] Qingzhong Chen, Shilun Cai, Crystal Cai, et al., “Colo-SCRL: Self-supervised contrastive representation learning for colonoscopic video retrieval,” in IEEE Int. Conf. Multimedia Expo (ICME), 2023, pp. 1056–1061.

[6] Samuel Gussman-Toh, Wei-Chung Cheng, et al., “Assessing AI polyp detection delay using tracklets,” in IEEE Int. Symp. Biomed. Imaging (ISBI), 2026, pp. 1–4.

[7] Luca Parolari, Andrea Cherubini, Lamberto Ballan, et al., “Temporally-aware supervised contrastive learning for polyp counting in colonoscopy,” in Med. Image Comput. Comput. Assist. Interv. (MICCAI), 2025.

[8] Yotam Intrator, Natalie Aizenberg, Amir Livne, et al., “Self-supervised polyp re-identification in colonoscopy,” in Med. Image Comput. Comput. Assist. Interv. (MICCAI), 2023, pp. 590–600.

[9] Luca Parolari, Andrea Cherubini, Lamberto Ballan, et al., “Towards polyp counting in full-procedure colonoscopy videos,” in IEEE Int. Symp. Biomed. Imaging (ISBI), 2025, pp. 1–5.

[10] Suncheng Xiang, Cang Liu, et al., “VT-ReID: Learning discriminative visual-text representation for polyp re-identification,” in IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2024, pp. 3170–3174.

[11] Luca Parolari, Pietro Gori, Lamberto Ballan, et al., “Contrastive learning under noisy temporal selfsupervision for colonoscopy videos,” in Med. Image Comput. Comput. Assist. Interv. (MICCAI), 2026.

[12] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni,´ et al., “DINOv2: Learning robust visual features without supervision,” TMLR, vol. 2024, 2024.

[13] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, et al.,´ “DINOv3,” arXiv preprint arXiv:2508.10104, 2025.

[14] Mido Assran, Adrien Bardes, David Fan, et al., “V-JEPA 2: Self-supervised video models enable understanding, prediction and planning,” arXiv preprint arXiv:2506.09985, 2025.

[15] Zhao Wang, Chang Liu, Shaoting Zhang, et al., “Foundation model for endoscopy video analysis via largescale self-supervised pre-train,” in Med. Image Comput. Comput. Assist. Interv. (MICCAI), 2023, pp. 101–111.

[16] Zhao Wang, Chang Liu, et al., “Improving foundation model for endoscopy video analysis via representation learning on long sequences,” IEEE J. Biomed. Health Inform., vol. 29, no. 5, pp. 3526–3536, 2025.

[17] Dominik Batic, Felix Holm, Ege Ozsoy, et al., “En-<sup>¨</sup> doViT: Pretraining vision transformers on a large collection of endoscopic images,” Int. J. Comput. Assist. Radiol. Surg., vol. 19, no. 6, pp. 1085–1091, 2024.

[18] Tim J. M. Jaspers, Ronald L. P. D. de Jong, Yiping Li, et al., “Scaling up self-supervised learning for improved surgical foundation models,” Med. Image Anal., vol. 108, pp. 103873, 2026.

[19] Carlo Biffi, Giulio Antonelli, et al., “REAL-Colon: A dataset for developing real-world AI applications in colonoscopy,” Sci. Data, vol. 11, no. 1, pp. 539, 2024.

[20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al., “Deep residual learning for image recognition,” in IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.

[21] Xiaohua Zhai, Alexander Kolesnikov, et al., “Scaling vision transformers,” in IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 1204–1213.