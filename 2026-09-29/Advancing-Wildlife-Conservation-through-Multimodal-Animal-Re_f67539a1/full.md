# Advancing Wildlife Conservation through Multimodal Animal Re-Identification with Environmental Metadata

Yuzhuo Li<sup>1</sup>\*, Di Zhao<sup>1</sup>\*<sup>†</sup>, Tingrui Qiao<sup>1</sup>, Yihao Wu<sup>1</sup>, Bo Pang<sup>1</sup>, Yun Sing Koh<sup>1</sup>

<sup>1</sup>School of Computer Science, University of Auckland

Auckland, New Zealand

{yil708, tqia361, ywu840, bpan882}@aucklanduni.ac.nz {di.zhao, y.koh}@auckland.ac.nz

## Abstract

Identifying individual animals is crucial for effective wildlife monitoring and conservation efforts. Recent advancements in computer vision have shown promise in animal reidentification (Animal ReID) by leveraging data from camera traps. However, existing Animal ReID datasets rely exclusively on visual data, overlooking environmental metadata that ecologists have identified as highly correlated with animal behavior and identity, such as temperature and circadian rhythms. Meanwhile, modern vision–language models (VLMs) offer rich multimodal reasoning capabilities, but existing resources underutilize their text-processing potential. To address these limitations, we propose MetaWild, a multimodal Animal ReID dataset comprising 20,890 images across six species, paired with environmental metadata extracted from embedded camera trap overlays and scene contexts. Additionally, to facilitate the use of metadata in existing ReID methods, we propose the Meta-Feature Adapter (MFA), a lightweight module that can be incorporated into existing VLM-based ReID methods, allowing ReID models to leverage both environmental metadata and visual information to improve ReID performance. Experiments on MetaWild show that combining baseline ReID models with MFA to incorporate metadata consistently improves performance compared to using visual information alone, validating the effectiveness of incorporating metadata in re-identification.

## Introduction

Animal re-identification (Animal ReID) aims to recognize individual animals across images to support wildlife research such as population monitoring, movement ecology, and conservation management (Schneider et al. 2019; Wu et al. 2026a; Schofield et al. 2022). Compared to traditional tagging-based approaches, computer vision offers a scalable, non-invasive solution that avoids disturbing endangered species (Beery 2023; Xu et al. 2024). However, most existing Animal ReID datasets are purely visual, captured by surveillance or camera traps without contextual information (Adam et al. 2024; Li et al. 2020; Gao et al. 2021). This visual-only design introduces two limitations. First, it omits environmental metadata (e.g., temperature and circadian rhythms), which ecologists have shown to be highly correlated with animal behavior and appearance (Leliveld et al. 2022). Figure 1(a) shows that different individuals tend to appear under distinct environmental conditions, showing the potential of metadata to provide identity-discriminative cues. Without such information, existing datasets cannot support the evaluation of the impact of incorporating environmental metadata on ReID performance. Second, with the rapid advancement of multimodal models capable of jointly processing images and text (Radford et al. 2021; Qiao et al. 2026a; Pang et al. 2025a), image-only datasets underutilize their text-encoding capabilities, limiting the full potential of multimodal representations in Animal ReID.

![](images/fafa256670bf4dba1667afffa03596094cbff8e2c2ad5262abfb49376868e6a2.jpg)  
Figure 1: Overview of multimodal Animal ReID framework and the role of metadata. (a) Example images of two deer individuals showing distinct preferences across environmental conditions. (b) The model integrates visual features with textual environmental metadata.

To address these limitations, we present MetaWild, a dataset designed to enable systematic evaluation of metadata integration and multimodal learning in Animal ReID. MetaWild is constructed from the publicly available New Zealand Trail Camera (NZ-TrailCams) dataset (LILA BC Project 2024), a wildlife image collection captured using camera traps deployed in diverse natural habitats across New Zealand. We select six representative species, including invasive and native animals. We have carefully curated images that are visually clear and contain reliable metadata overlays, comprising a total of 20,890 images. In addition to visual data, MetaWild provides curated environmental metadata, including temperature, circadian rhythms, and face orientation, which are chosen based on their availability from overlays and their influence on animal appearance. This design supports multimodal ReID by jointly exploiting visual and textual cues, as illustrated in Figure 1(b). Furthermore, since existing Animal ReID methods are generally designed to operate solely on visual inputs and cannot directly incorporate the textual metadata, we propose the Meta-Feature Adapter (MFA), a lightweight module that can be incorporated into existing VLM-based Animal ReID methods (Jiao et al. 2024), allowing the integration of environmental metadata into visual representations.

Our contributions are summarized as follows. Firstly, we create and release MetaWild, a new dataset that pairs visual animal images with curated environmental metadata to evaluate the impact of incorporating environmental metadata on ReID performance, thereby facilitating the development of multimodal ReID methods. Secondly, we propose MFA, a lightweight module that enables the integration of metadata into existing VLM-based Animal ReID models, allowing for performance evaluation with MetaWild without modifying the model architecture. Finally, extensive experiments on MetaWild show that incorporating environmental metadata alongside visual data within a multimodal learning framework consistently improves Animal ReID performance.

## Related Work

Animal ReID aims to recognize and match individual animals across images for wildlife monitoring (Wu et al. 2026b). Existing methods include: (1) Globalfeature learning that treats the entire image as input, directly extracting global features for ReID (He et al. 2023). (2) Speciesspecificfeature extraction that relies on distinctive local patterns (e.g., elephant ears (Weideman et al. 2020)), but they are sensitive to occlusions and viewpoint variations, which hinder their generalizability across species. and (3) Auxiliary information integration: Methods such as pose key point estimation (Li et al. 2020, 2025) incorporate additional visual cues to refine feature extraction. Despite their success, these methods remain constrained to image-based data.

Animal ReID Datasets. Existing datasets range from controlled farm or lab environments (Kern et al. 2024; Wang et al. 2026; Wahltinez and Wahltinez 2024) to challenging in-the-wild camera-trap collections (Li et al. 2020). WildlifeDatasets (Cerm<sup>ˇ</sup> ak et al. 2024) provide a unified li-´ brary and standardized tools for visual ReID benchmarking. While these datasets have advanced visual-based ReID, they omit environmental metadata, which has been shown to influence animal behavior and appearance, limiting the evaluation of its impact on ReID performance.

Vision-Language Model learn aligned visual–textual representations (Jia et al. 2021; Pang et al. 2025b; Zhao et al. 2025) and show strong generalization ability (Zhao et al. 2024, 2026; Qiao et al. 2026b). While some recent works have adapted VLMs to ReID tasks (Li, Sun, and Li 2023; Jiao et al. 2024), they primarily leverage the image encoder and use the text encoder only for static, category-level descriptions. In this paper, we leverage environmental metadata as a semantically rich textual information source to more fully utilize the text encoder for Animal ReID.

![](images/fa5b6b64da0ae2c40dc257331fd62841794ee04ea9ff0514cc37aac54486608d.jpg)  
Figure 2: Examples of the MetaWild dataset.

## The MetaWild Dataset

## Dataset Composition

MetaWild is designed to facilitate the evaluation of multimodal learning approaches in Animal ReID by pairing visual data with contextual environmental metadata. It is constructed from the publicly available NZ-TrailCams dataset (LILA BC Project 2024). To ensure the applicability and diversity of metadata integration across various animal species, MetaWild comprises 20,890 images spanning six representative species: Deer, Hare, Penguin, Pukeko, Stoat,¯ and Wallaby. Each image is paired with environmental metadata extracted from embedded camera trap overlays.

<table><tr><td></td><td colspan="2">Train</td><td colspan="2">Gallery</td><td colspan="2">Query</td><td colspan="2">Total</td></tr><tr><td>Datasets</td><td>Imgs</td><td>s IDs</td><td>Imgs IDs</td><td></td><td>Imgs IDs</td><td></td><td>Imgs IDs</td><td></td></tr><tr><td>Deer</td><td>1,631</td><td>21</td><td>586</td><td>17</td><td>216</td><td>17</td><td>2,433</td><td>38</td></tr><tr><td>Hare</td><td>1,820</td><td>31</td><td>926</td><td>29</td><td>306</td><td>29</td><td>3,052</td><td>60</td></tr><tr><td>Penguin</td><td>1,431</td><td>34</td><td>725</td><td>43</td><td>296</td><td>643</td><td>2,452</td><td>77</td></tr><tr><td>Pūkeko</td><td>1,854</td><td>11</td><td>800</td><td>19</td><td>411</td><td>19</td><td>3,065</td><td>30</td></tr><tr><td>Stoat</td><td>4,067</td><td>151</td><td>1,649</td><td>9102</td><td>1,017 102</td><td></td><td>6,733</td><td>253</td></tr><tr><td>Wallaby</td><td>1,888</td><td>25</td><td>964</td><td>22</td><td>303</td><td>22</td><td>3,155</td><td>47</td></tr></table>

Table 1: Details of Benchmark Datasets.

Figure 2 presents examples of the species and environmental metadata included in the MetaWild dataset. The selection of species reflects conservation priorities, including predators (Stoat), pests (Wallaby, Hare), endangered native species (Yellow-eyed Penguin), and native animals (Deer and Pukeko) (Department of Conservation 2024, 2025).¯ Stoats compete with native birdlife for food and habitat, also eat the eggs and young, and attack the adults, posing a significant threat to native wildlife. The Yellow-eyed Penguin, classified as endangered with a rapidly declining population, exemplifies a vulnerable native species in urgent need of protection. Wallabies and hares are considered agricultural pests, causing significant damage to native vegetation and ecosystems. Deer and Pukeko, both native to New Zealand,¯ exhibit unique visual and behavioral patterns that introduce diversity and realism, further enriching the dataset’s applicability. The dataset was organized according to standard ReID protocols, comprising 60% for training, 25% for the gallery, and 15% for the query sets (Li et al. 2018), and detailed statistics are shown in Table 1.

## Dataset Construction

We ensured high data quality by filtering visually clear images, performing reliable identity annotations, and extracting consistent metadata. Both identity labels and metadata were independently verified by at least three annotators. The detailed procedure is described below.

Image Selection. We first conducted a filtering process to remove low-quality images in which the target animals appeared as unrecognizable, blurry blobs due to motion blur or poor lighting. We further selected a representative subset of images for each species to ensure sufficient intra-species variation across individuals and conditions (e.g., viewpoints, lighting, and environments) and maintain balanced distribution across environmental metadata.

Identity Annotation. We employed a combination of temporal analysis and visual verification to assign identities within each species. For temporal analysis, we used the camera trap images’ time-stamped nature to track animals across sequential frames. Animals captured within narrow time windows (e.g., a few seconds) at the same camera location were likely to belong to the same individual. To complement this, manual visual inspection was conducted to confirm or correct identity groupings based on distinct physical characteristics, such as markings, size, and shape.

Metadata Extraction. We focus on three metadata features: temperature, circadian rhythms, and face orientation, as they directly influence the animal’s appearance and behavior (Cade et al. 2021; McVey et al. 2023). Temperature is read from embedded overlays, circadian rhythm (day/night) is inferred from timestamps and lighting, and face orientation is manually annotated to support geometric reasoning in ReID. Other metadata types were excluded due to redundancy or unavailability (e.g., missing geolocation). All metadata is standardized and stored in structured JSON files for each image.

Image Preprocessing. To focus on the target animal and reduce background noise, we used a YOLO-based detector to generate bounding boxes for cropping, followed by manual verification to correct detection errors. Furthermore, each cropped image was renamed using a structured format, id camera-id count (e.g., 11 CT-GIG-03 27, where 11 denotes the individual identity, CT-GIG-03 represents the camera ID, and 27 indicates the 28th image for identity

![](images/01a3399022a105b5f5cbf3ab93d17c87eae9919fc473504ee39e40a83a058b02.jpg)  
Figure 3: Overview of the proposed Meta-Feature Adapter (MFA) module

11). This systematic naming convention facilitates efficient data management and traceability.

## Methodology

We propose a lightweight Meta-Feature Adapter (MFA) to integrate environmental metadata into existing VLM-based Animal ReID models without modifying their backbone architectures. As shown in Figure 3, MFA consists of two main components: (1) Feature Experts, employed adapters (Gao et al. 2024) as experts that refine visual and textual embeddings into metadata-aware representations. (2) Gated Cross-Attention, which fuses visual and metadata features by weighting relevant information from each modality.

## Feature Experts

To enable effective fusion between visual features and environmental metadata, we incorporate feature experts in both the text and image branches. In the text branch, we convert metadata into natural language descriptions using a fixed prompt template: $\mathbf { \ddot { \theta } } _ { \mathbf { A } }$ photo of a {species} {individual id} in {freezing, cold, chilly, cool, warm, hot} temperature, with face direction {front, back, left, right}, captured during the {day, night}.” This template mimics natural language captions used during VLM pretraining, maximizing compatibility with the text encoder. The metadata-augmented prompt $P _ { \mathbf { M } }$ is encoded using the pretrained text encoder $\tau ( \cdot )$ to obtain text embedding $T _ { \mathrm { M } } \ = \ T ( P _ { \mathrm { M } } )$ , which is then refined by a Textual Metadata Expert (TME) $E _ { T }$ into a metadataaware embedding $T _ { \mathrm { M } } ^ { \prime } = E _ { \mathrm { T } } ( T _ { \mathrm { M } } )$ . Similarly, in the image branch, we introduce a Visual Feature Expert (VFE) $E _ { I }$ to transform the raw visual embedding $I _ { \mathrm { x } }$ into metadata-aware representations $I _ { \mathrm { x } } ^ { \prime } = E _ { I } ( I _ { \mathrm { x } } )$ . Both experts $E _ { T }$ and $E _ { I }$ are trained end-to-end using ReID loss functions including the identity classification loss $\mathcal { L } _ { i d }$ for encouraging separability across identities and the triplet loss $\mathcal { L } _ { t r i }$ for enforcing relative distance constraints (Wang and Liu 2021). By allowing gradients from the ReID objective to propagate through the feature experts, $E _ { T }$ and $E _ { I }$ learn task-relevant transformations that directly improve ReID performance.

## Gated Cross-Attention

We utilize a cross-attention mechanism to integrate metadata-aware text embeddings $T _ { \mathbf { M } } ^ { \prime }$ with visual features $I _ { \mathrm { x } } ^ { \prime } .$

<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>DeermAP CMC-1</td><td rowspan=1 colspan=1>HaremAP CMC-1</td><td rowspan=1 colspan=1>PenguinmAP CMC-1</td><td rowspan=1 colspan=1>PūkekomAP CMC-1</td><td rowspan=1 colspan=1>StoatmAP CMC-1</td><td rowspan=1 colspan=1>WallabymAP CMC-1</td></tr><tr><td rowspan=1 colspan=1>CLIP-ZS (Radford et al. 2021)</td><td rowspan=1 colspan=1>50.0±.0 92.1±.0|</td><td rowspan=1 colspan=1>|33.8±.0 81.1±.0|</td><td rowspan=1 colspan=1>34.0±.0 57.8±.0</td><td rowspan=1 colspan=1>|33.0±.0 69.6±.0</td><td rowspan=1 colspan=1>30.1±.0 72.7±.0|</td><td rowspan=1 colspan=1>|46.2±.0 85.5±.0</td></tr><tr><td rowspan=1 colspan=1>CLIP-FTCLIP-FT+MFA</td><td rowspan=1 colspan=1>63.2±.1 95.4±.4|66.7±.2 95.7±.3</td><td rowspan=1 colspan=1>|56.7±.292.5±.458.4±.3 92.6±.2</td><td rowspan=1 colspan=1>44.0±.364.9±.246.0±.264.9±.3</td><td rowspan=1 colspan=1>56.8±.180.3±.558.2±.281.6±.3</td><td rowspan=1 colspan=1>|68.6±.1 92.3±.369.8±.2 91.8±.2</td><td rowspan=1 colspan=1>|55.5±.1 92.4±.456.8±.4 90.7±.3</td></tr><tr><td rowspan=1 colspan=1>CLIP-ReID (Li, Sun, and Li 2023)CLIP-ReID+MFA</td><td rowspan=1 colspan=1>65.2±.4 95.8±.369.4±.2 98.1±.1</td><td rowspan=1 colspan=1>60.0±.6 95.1±.363.2±.1 95.4±.1</td><td rowspan=1 colspan=1>|44.8±.4 67.9±.450.3±.468.6±.2</td><td rowspan=1 colspan=1>57.6±.282.0±.159.8±.183.7±.2</td><td rowspan=1 colspan=1>67.5±.1 91.5±.371.5±.2 92.0±.1</td><td rowspan=1 colspan=1>56.9±.4 88.8±.261.8±.2 92.1±.1</td></tr><tr><td rowspan=1 colspan=1>ReID-AW (Jiao et al. 2024)ReID-AW+MFA</td><td rowspan=1 colspan=1>67.5±.3 96.0±.272.4±.2 97.0±.2</td><td rowspan=1 colspan=1>63.3±.4 95.6±.366.2±.3 96.8±.2</td><td rowspan=1 colspan=1>48.8±.569.4±.355.3±.470.8±.4</td><td rowspan=1 colspan=1>58.5±.382.0±.461.8±.2 86.7±.3</td><td rowspan=1 colspan=1>69.5±.3 93.5±.574.1±.4 95.0±.2</td><td rowspan=1 colspan=1>|58.4±.3 91.8±.263.5±.1 92.7±.2</td></tr></table>

Table 2: Intra-species re-identification performance on the MetaWild dataset across six species, we report mAP and CMC-1 accuracy (%) with 95% confidence intervals. CLIP-ZS shows zero variance due to its deterministic zero-shot inference nature.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>DeermAP CMC-1</td><td rowspan=1 colspan=1>HaremAP CMC-1</td><td rowspan=1 colspan=1>PenguinmAPCMC-1</td><td rowspan=1 colspan=1>PūkekomAP CMC-1</td><td rowspan=1 colspan=1>StoatmAP CMC-1</td><td rowspan=1 colspan=1>WallabymAP CMC-1</td></tr><tr><td rowspan=1 colspan=1>CLIP-ZS (Radford et al. 2021)</td><td rowspan=1 colspan=1>40.3±.0 80.5±.0|</td><td rowspan=1 colspan=1>|25.1±.0 75.1±.0|</td><td rowspan=1 colspan=1>|27.5±.0 50.4±.0|</td><td rowspan=1 colspan=1>|24.6±.0 64.5±.0|</td><td rowspan=1 colspan=1>|23.0±.0 62.3±.0|</td><td rowspan=1 colspan=1>|40.6±.0 80.3±.0</td></tr><tr><td rowspan=1 colspan=1>CLIP-FTCLIP-FT+MFA</td><td rowspan=1 colspan=1>55.1±.283.0±.456.7±.2 86.4±.4</td><td rowspan=1 colspan=1>|41.5±.281.4±.243.9±.481.6±.2</td><td rowspan=1 colspan=1>38.7±.359.7±.240.8±.163.5±.2</td><td rowspan=1 colspan=1>41.3±.276.9±.242.3±.376.9±.4</td><td rowspan=1 colspan=1>45.4±.379.3±.246.2±.4 79.6±.2</td><td rowspan=1 colspan=1>49.0±.2 72.2±.150.0±.3 72.8±.3</td></tr><tr><td rowspan=1 colspan=1>CLIP-ReID (Li, Sun, and Li 2023)CLIP-ReID+MFA</td><td rowspan=1 colspan=1>|56.3±.384.4±.360.2±.4 89.2±.3</td><td rowspan=1 colspan=1>43.5±.186.3±.344.1±.188.6±.2</td><td rowspan=1 colspan=1>39.5±.260.3±.441.3±.264.1±.4</td><td rowspan=1 colspan=1>43.8±.477.9±.145.1±.378.3±.2</td><td rowspan=1 colspan=1>45.8±.379.3±.247.9±.180.3±.2</td><td rowspan=1 colspan=1>50.1±.4 82.5±.252.3±.2 82.8±.2</td></tr><tr><td rowspan=1 colspan=1>ReID-AW (Jiao et al. 2024)ReID-AW+MFA</td><td rowspan=1 colspan=1>59.3±.3 89.0±.262.5±.4 92.4±.4</td><td rowspan=1 colspan=1>47.6±.290.2±.450.2±.3 90.8±.2</td><td rowspan=1 colspan=1>40.8±.5 63.9±.344.2±.4 64.6±.3</td><td rowspan=1 colspan=1>50.4±.4 80.3±.153.6±.3 83.6±.1</td><td rowspan=1 colspan=1>53.3±.3 83.9±.156.1±.1 84.6±.1</td><td rowspan=1 colspan=1>|51.7±.3 84.2±.253.1±.2 86.0±.3</td></tr></table>

Table 3: Leave-one-domain-out inter-species ReID performance on the MetaWild dataset, we report mAP and CMC-1 accuracy (%) with 95% confidence intervals for each target species.

Unlike conventional fusion methods, cross-attention enables context-aware integration, allowing image features to selectively attend to relevant metadata cues. Given image embeddings $I _ { \mathrm { x } } ^ { \prime } \in \mathbb { R } ^ { N \times d }$ and metadata-aware text embeddings $T _ { \mathbf { M } } ^ { \prime } \in \breve { \mathbb { R } } ^ { M \times d }$ , we compute Query (Q), Key (K), and Value $( \overrightharpoon { V } )$ matrices as $Q = { \mathrm { \bar { } { } { } ^ { \prime } { } W _ { Q } , \bar { K ^ { \prime } } = \bar { T } _ { M } ^ { \prime } \bar { W _ { K } } , \bar { V } = T _ { M } ^ { \prime } { \cal { W } _ { V } } } }$ where $W _ { Q } , W _ { K }$ , and $W _ { V }$ are learnable projection matrices. To address the fact that metadata relevance varies across images, we employ a gating mechanism to selectively adjust metadata contribution. A gating value $\gamma \in [ 0 , 1 ]$ is computed as $\gamma = \mathrm { G a t e } ( I _ { \mathrm { x } } ^ { \prime } , T _ { \mathrm { M } } ^ { \prime } ) = \bar { \sigma } ( \mathrm { M L P } ( [ I _ { \mathrm { x } } ^ { \prime } ; T _ { \mathrm { M } } ^ { \prime } ] ) )$ , where $[ I _ { \mathrm { x } } ^ { \prime } ; T _ { \mathrm { M } } ^ { \prime } ]$ denotes concatenation, MLP is a multi-layer perceptron with layer normalization, and σ is the sigmoid activation. The final meta-augmented image embedding $I _ { \mathrm { m e t a } }$ is computed as $I _ { \mathrm { { m e t a } } } = \gamma { \bar { A V } } + I _ { \mathrm { { x } } } ^ { \prime } $ , where A is the cross-attention weight matrix (Shi et al. 2022). The loss function is defined as:

$$
\mathcal { L } _ { \mathcal { A } } ^ { i } = - \log \frac { \exp \left( s \left( T _ { \mathrm { M } } ^ { \prime i } , I _ { \mathrm { m e t a } } ^ { i } \right) / \tau \right) } { \sum _ { j = 1 } ^ { B } \exp \left( s \left( T _ { \mathrm { M } } ^ { \prime i } , I _ { \mathrm { m e t a } } ^ { j } \right) / \tau \right) }\tag{1}
$$

where B is the batch size, $( T _ { M } ^ { \prime i } , I _ { \mathrm { m e t a } } ^ { i } )$ is the i-th matched pair, and τ is the temperature parameter.

## Experiments

We evaluate existing Animal ReID models under visual-only and visual+metadata settings on MetaWild using two protocols: (1) Intra-species ReID (Varghese, Jawahar, and Prince 2023), where training and testing are performed on different individuals within the same species; and (2) Inter-species ReID (Heiling et al. 2016), where we adopt a leave-onedomain-out (LODO) strategy (Yu et al. 2024), training on five species and testing on the remaining unseen species to reflect real-world scenarios where collecting labeled data for every species is impractical (Jiao et al. 2024).

## Experimental Results

Intra-species ReID. Table 2 shows that incorporating environmental metadata consistently improves ReID performance across all six species. CLIP-ReID achieves mAP gains of 5.5% on Penguin, 4.9% on Wallaby, and 4.2% on Deer, while ReID-AW shows improvements of 6.5% on Penguin, 5.1% on Wallaby, and 4.9% on Deer, demonstrating the effectiveness of metadata integration.

Inter-species ReID. Table 3 summarizes LODO evaluations where one species is held out for testing while the remaining five are used for training. Incorporating metadata consistently improves all baseline methods: CLIP-ReID+MFA achieves mAP gains of 3.9% on Deer, 3.3% on Pukeko,¯ and 1.8% on Penguin, while ReID-AW shows improvements of 3.4% on Penguin, 3.2% on Deer, and 2.6% on Hare. These results demonstrate that environmental metadata provides complementary cues that enhance model generalization across species boundaries and improve transferable identity representation learning.

## Conclusion

We present MetaWild, a multimodal Animal ReID dataset pairing visual data with environmental metadata to evaluate metadata’s impact on ReID performance. To support this investigation without architectural changes, we further propose the Meta-Feature Adapter (MFA), a lightweight module that enables the integration of metadata into VLM-based Animal ReID methods. Extensive experiments demonstrate that incorporating metadata alongside visual information consistently improves ReID accuracy, confirming the value of contextual environmental metadata. We hope this work can inspire broader exploration of environmental metadata and multimodal approaches in wildlife ReID and beyond.

## References

Adam, L.; Cerm<sup>ˇ</sup> ak, V.; Papafitsoros, K.; and Picek, L. 2024.´ SeaTurtleID2022: A long-span dataset for reliable sea turtle re-identification. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 7146– 7156.

Beery, S. M. 2023. Where the Wild Things Are: Computer Visionfor Global-Scale Biodiversity Monitoring. California Institute of Technology.

Cade, D. E.; Gough, W. T.; Czapanskiy, M. F.; Fahlbusch, J. A.; Kahane-Rapport, S. R.; Linsky, J. M.; Nichols, R. C.; Oestreich, W. K.; Wisniewska, D. M.; Friedlaender, A. S.; et al. 2021. Tools for integrating inertial sensor data with video bio-loggers, including estimation of animal orientation, motion, and position. Animal Biotelemetry, 9(1): 34.

Cerm <sup>ˇ</sup> ak, V.; Picek, L.; Adam, L.; and Papafitsoros, K. 2024. ´ WildlifeDatasets: An open-source toolkit for animal reidentification. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 5953–5963.

Department of Conservation. 2024. New Zealand’s Unique Biodiversity is at Risk from Pests, Weeds, and Other Threats. https://www.doc.govt.nz/nature/pests-and-threats/. Accessed: 2024.

Department of Conservation. 2025. New Zealand’s Native Animals. https://www.doc.govt.nz/nature/native-animals/. Accessed: 2025.

Gao, J.; Burghardt, T.; Andrew, W.; Dowsey, A. W.; and Campbell, N. W. 2021. Towards self-supervision for video identification of individual Holstein-Friesian cattle: The Cows2021 dataset. arXiv preprint arXiv:2105.01938.

Gao, P.; Geng, S.; Zhang, R.; Ma, T.; Fang, R.; Zhang, Y.; Li, H.; and Qiao, Y. 2024. CLIP-Adapter: Better visionlanguage models with feature adapters. International Journal ofComputer Vision, 132(2): 581–595.

He, Z.; Qian, J.; Yan, D.; Wang, C.; and Xin, Y. 2023. Animal re-identification algorithm for posture diversity. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing, 1–5. IEEE.

Heiling, S.; Khanal, S.; Barsch, A.; Zurek, G.; Baldwin, I. T.; and Gaquerel, E. 2016. Using the knowns to discover the unknowns: MS-based dereplication uncovers structural diversity in 17-hydroxygeranyllinalool diterpene glycoside production in the Solanaceae. The Plant Journal, 85(4): 561– 577.

Jia, C.; Yang, Y.; Xia, Y.; Chen, Y.-T.; Parekh, Z.; Pham, H.; Le, Q.; Sung, Y.-H.; Li, Z.; and Duerig, T. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, 4904–4916. PMLR.

Jiao, B.; Liu, L.; Gao, L.; Wu, R.; Lin, G.; Wang, P.; and Zhang, Y. 2024. Toward re-identifying any animal. Advances in Neural Information Processing Systems, 36.

Kern, D.; Schiele, T.; Klauck, U.; and Ingabire, W. 2024. Towards Automated Chicken Monitoring: Dataset and Machine Learning Methods for Visual, Noninvasive Reidentification. Animals, 15(1): 1.

Leliveld, L. M.; Riva, E.; Mattachini, G.; Finzi, A.; Lovarelli, D.; and Provolo, G. 2022. Dairy cow behavior is affected by period, time of day and housing. Animals, 12(4): 512.

Li, D.; Zhang, Z.; Chen, X.; and Huang, K. 2018. A richly annotated pedestrian dataset for person retrieval in real surveillance scenarios. IEEE Transactions on Image Processing, 28(4): 1575–1590.

Li, S.; Li, J.; Tang, H.; Qian, R.; and Lin, W. 2020. ATRW: A Benchmark for Amur Tiger Re-identification in the Wild. In Proceedings of the 28th ACM International Conference on Multimedia, 2590–2598.

Li, S.; Sun, L.; and Li, Q. 2023. CLIP-ReID: exploiting vision-language model for image re-identification without concrete text labels. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, 1405–1413.

Li, Y.; Zhao, D.; Qiao, T.; Wu, Y.; Pang, B.; and Koh, Y. S. 2025. MetaWild: A multimodal dataset for animal reidentification with environmental metadata. In Proceedings of the 33rd ACM international conference on multimedia, 13009–13015.

LILA BC Project. 2024. Trail Camera Images of New Zealand Animals. https://lila.science/datasets/nz-trailcams. Accessed: 2024.

McVey, C.; Hsieh, F.; Manriquez, D.; Pinedo, P.; and Horback, K. 2023. Invited Review: Applications of unsupervised machine learning in livestock behavior: Case studies in recovering unanticipated behavioral patterns from precision livestock farming data streams. Applied Animal Science, 39(2): 99–116.

Pang, B.; Qiao, T.; Walker, C.; Cunningham, C.; and Koh, Y. S. 2025a. CABIN: Debiasing Vision-Language Models Using Backdoor Adjustments. In IJCAI, 484–492.

Pang, B.; Qiao, T.; Walker, C.; Cunningham, C.; and Koh, Y. S. 2025b. Libra: Measuring bias of large language model from a local context. In European Conference on Information Retrieval, 1–16. Springer.

Qiao, T.; Zhao, D.; Li, Y.; Pang, B.; Walker, C.; Cunningham, C.; and Koh, Y. S. 2026a. Multiple Images Distract Large Multimodal Models via Attention Fragmentation. In European Conference on Computer Vision, 361– 379. Springer.

Qiao, T.; Zhao, D.; Walker, C.; Cunningham, C.; and Koh, Y. S. 2026b. Zero-Shot Domain Generalisation via Prompt-Driven Feature Refinement. In 2026 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 6184– 6193. IEEE.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 8748–8763. PMLR.

Schneider, S.; Taylor, G. W.; Linquist, S.; and Kremer, S. C. 2019. Past, present and future approaches using computer vision for animal re-identification from camera trap data. Methods in Ecology and Evolution, 10(4): 461–470.

Schofield, G.; Papafitsoros, K.; Chapman, C.; Shah, A.; Westover, L.; Dickson, L. C.; and Katselidis, K. A. 2022. More aggressive sea turtles win fights over foraging resources independent of body size and years of presence. Animal Behaviour, 190: 209–219.

Shi, X.; Wei, D.; Zhang, Y.; Lu, D.; Ning, M.; Chen, J.; Ma, K.; and Zheng, Y. 2022. Dense cross-query-and-support attention weighted mask aggregation for few-shot segmentation. In European Conference on Computer Vision, 151– 168. Springer.

Varghese, A.; Jawahar, M.; and Prince, A. A. 2023. Finetuning ConvNets with novel leather image data for species identification. In Fifteenth International Conference on Machine Vision, volume 12701, 150–157. SPIE.

Wahltinez, O.; and Wahltinez, S. J. 2024. An open-source general purpose machine learning framework for individual animal re-identification using few-shot learning. Methods in Ecology and Evolution, 15(2): 373–387.

Wang, D.; Zhou, C.; Zhao, D.; Liu, X.; Ma, M. C.; Ushaw, G.; and Davison, R. 2026. TowerMind: A tower defence game learning environment and benchmark for LLM as agents. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 26151–26159.

Wang, F.; and Liu, H. 2021. Understanding the behaviour of contrastive loss. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2495– 2504.

Weideman, H.; Stewart, C.; Parham, J.; Holmberg, J.; Flynn, K.; Calambokidis, J.; Paul, D. B.; Bedetti, A.; Henley, M.; Pope, F.; et al. 2020. Extracting identifying contours for African elephants and humpback whales using a learned appearance model. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 1276– 1285.

Wu, Y.; Zhao, D.; Getz, W. M.; Liu, L.; Dobbie, G.; Wilson, D.; and Koh, Y. S. 2026a. Region-Aware Multimodal Interleaving for Animal Re-Identification. In European Conference on Computer Vision, 76–94. Springer.

Wu, Y.; Zhao, D.; Li, Y.; Alajas, M.; Glen, A. S.; Zhang, J.; Dobbie, G.; Wilson, D.; and Koh, Y. S. 2026b. Overcoming fine-grained visual challenges in animal re-identification via semantic feature alignment. In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 371– 381. IEEE.

Xu, P.; Zhang, Y.; Ji, M.; Guo, S.; Tang, Z.; Wang, X.; Guo, J.; Zhang, J.; and Guan, Z. 2024. Advanced intelligent monitoring technologies for animals: A survey. Neurocomputing, 585: 127640.

Yu, H.; Zhang, X.; Xu, R.; Liu, J.; He, Y.; and Cui, P. 2024. Rethinking the evaluation protocol of domain generalization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21897–21908.

Zhao, D.; Koh, Y. S.; Dobbie, G.; Hu, H.; and Fournier-Viger, P. 2024. Symmetric Self-Paced Learning for Domain Generalization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 16961–16969.

Zhao, D.; Zhang, J.; Hu, H.; Fournier-Viger, P.; Dobbie, G.; and Koh, Y. S. 2025. Balancing Invariant and Specific Knowledge for Domain Generalization with Online Knowledge Distillation. In IJCAI, 2440–2448.

Zhao, D.; Zhang, J.; Hu, H.; Fournier-Viger, P.; Dobbie, G.; and Koh, Y. S. 2026. Unlearning during training: Domainspecific gradient ascent for domain generalization. In The Fourteenth International Conference on Learning Representations.