# Data-Driven Vision-Language Models for Remote Sensing: A Survey

Yue Zhou, Member, IEEE, Shujun Zhao, Xue Yang\*, Member, IEEE, Ruigang Li, Tianwen Zhang, Mengcheng Lan, Chaofeng Chen, Member, IEEE, Lingfei Ma, Member, IEEE, Hongjie He, Member, IEEE, Jonathan Li\* Fellow, IEEE

Abstract—During the deep learning era, innovations in remote sensing (RS) vision models primarily focused on optimizing network architectures for specific tasks and conducting end-to-end training. However, with the advent of the Vision-Language Models (VLMs) era, constrained by substantial computational resource demands, the mainstream research paradigm has shifted towards adapting general-purpose VLMs using RS vertical domain data for fine-tuning, rather than modifying their core architectures. This transition marks a change in the innovation pathway from model-centric to data-centric. Despite the increasing importance of data, the academic community still lacks a systematic, data-centric survey to guide researchers in constructing high-quality RS visionlanguage datasets for various application scenarios and in conducting effective model fine-tuning and performance evaluation based on these datasets. To address this gap, this paper presents a systematic review of the development status of RS datasets tailored for VLMs. It further provides a comprehensive practical guide for RS researchers in the large model era, covering the entire pipeline from data construction to fine-tuning and evaluation. This paper aims to encourage the academic community to build more

diverse and higher-quality vision-language datasets for specialized RS sub-fields. Ultimately, the goal is to converge these efforts into a comprehensive RS visionlanguage corpus that offers extensive task coverage and a wide geographical distribution, thereby laying a solid data foundation for advancing towards general artificial intelligence in RS.

Index Terms—Vision-Language Model, Remote Sensing, Dataset, Data Pipeline, Benchmark.

## I. INTRODUCTION

The advent of large-scale pretraining techniques [1]–[3] has ushered in artificial intelligence’s era of foundation models [4]–[6]. Within this context, Vision-Language Models (VLMs) [7]–[9] have demonstrated remarkable capabilities in crossmodal understanding and generation, establishing themselves as foundational architectures across numerous research domains. However, the enormous parameter counts and substantial computational requirements of these models render direct architectural modifications for specific vertical domains [10]–[12], such as remote sensing (RS) [13], [14], prohibitively expensive and technically demanding [15], [16]. Consequently, a significant paradigm shift is underway in the research community: from a model-centric paradigm focused on designing and optimizing model architectures toward a data-centric paradigm centered on domain-specific data curation and efficient adaptive fine-tuning of general-purpose foundation models [17], [18]. This transformation implies that high-quality domainspecific data, rather than model architecture itself, has become the critical enabling factor for unlocking the potential of foundation models and addressing specialized tasks [19], [20]. In particular, fine-tuning general VLMs with large-scale imagetext RS data has become the primary method for achieving advanced functionalities [21]–[23].

![](images/fb8f9b6bf428a4bd0ffe9a08712db01f8454ac829710a1ba23a87e190ce63723.jpg)  
Fig. 1. Representative Vision-Language Datasets in RS.

Broadly, VLMs for RS cover three major technical branches: 1) contrastive learning-based crossmodal representation models (e.g., RemoteCLIP [24]); 2) text-guided RS image generation models (e.g., Text2Earth [25]); and 3) Multimodal Large Language Models (MLLMs). The core research boundary of this survey is the third category, MLLMs, as this class of models has driven a paradigm shift in RS data construction from static annotations to free-form instruction dialogues, which is fully aligned with the data-centric perspective of this work. Since 2023, a paradigm shift driven by VLMs has gained significant momentum [26]–[28]. Academic institutions and research organizations worldwide are actively engaging in this interdisciplinary field, leading to the creation of numerous large-scale, multi-task, and multimodal vision-language datasets [29]–[31]. The field has evolved from foundational works, such as RSVQA [32] in 2023, to more versatile datasets, like GeoChat-Instruct [33] in 2024. By 2025, the development of relevant datasets experienced explosive growth [34]–[36], with expectations to enter an intensive launch phase from 2026 onward.

As shown in Figure 1, the accelerating pace of innovation is vividly illustrated by a timeline where icons transition from sparse to densely packed, metaphorically shifting from pedestrians and bicycles to cars, aircraft, and rockets, indicating exponential progress. Concurrently, community engagement has become a core driver [37]. The increasing number of GitHub stars for relevant projects reflects a vibrant open-source ecosystem characterized by continuous release, validation, and iteration cycles. A fundamental shift in task paradigms is also evident: dataset nomenclature has evolved from conventional detection [38]–[43] and segmentation [44]–[48] to encompass advanced tasks requiring complex cognition, such as instruction tuning [49]– [53], visual reasoning [54]–[57], video understanding [58]–[62], and geospatial tasks [63]–[67]. These trends collectively signal an emerging consensus: the integration of visual and linguistic reasoning is transforming satellite perception from mere seeing to genuine insight, heralding a new era of generalpurpose RS intelligence. This data-driven research surge not only underscores the maturation of multimodal intelligence in RS but also provides a dynamic, rapidly evolving platform for global scholars to collectively advance the field toward higher-level universal perception and comprehension [68]–[70].

Existing surveys on RS VLMs [7], [13], [71]– [73] have predominantly focused on methodological analyses at the model level. However, it is essential to acknowledge that, compared to the deep learning era [74]–[78], the design space for architectural innovation in the foundation model era is relatively constrained, with most contribu tions adopting modular augmentation and integration strategies [79]–[81]. The fundamental reason lies in the fact that substantial architectural modifications typically necessitate training from scratch on large-scale data to recover the generalization capabilities acquired from massive general-domain corpora [82], [83]. The scale of datasets constructed by researchers for specific scenarios is generally insufficient to support such from-scratch training of foundation models. Even when models trained from scratch achieve satisfactory fitting performance on scenario-specific datasets, they often forfeit the powerful zero-shot generalization capabilities inherent to general-purpose models, arguably one of the most valuable properties of foundation models [84], [85]. Although theoretically feasible to incorporate general-domain data for from-scratch training of novel architectures, such research imposes extraordinary computational resource requirements that remain beyond the reach of most academic laboratories [16], [86]. Therefore, we contend that in the foundation model era, the core competitive advantage of small-to-medium research teams resides in data rather than models. Furthermore, while several surveys on RS datasets exist, their treatment of RS vision-language datasets lacks sufficient depth [71], [87]. Compounding this issue, the explosive growth of RS vision-language datasets in recent years has rendered existing surveys incapable of providing comprehensive coverage of the current landscape [7], [73].

To address these gaps, this paper presents the first systematic review and taxonomy of RS datasets designed for VLMs fine-tuning, aiming to help researchers accurately grasp the developmental trajectory and frontier trends in this field. Beyond serving as an evolutionary chronicle of RS visionlanguage datasets, this work aspires to provide a practically valuable reference manual that can genuinely assist researchers in overcoming research bottlenecks. Specifically, researchers urgently need clarity on how to systematically construct, annotate, evaluate, and leverage high-quality vision-language datasets for diverse RS application scenarios [88]– [90], as well as how to select effective fine-tuning strategies [91]–[95] and conduct rigorous performance evaluation [8], [28], [96], [97] based on such high-quality data. The absence of such guiding knowledge has, to a certain extent, led to redundant data construction efforts, ad hoc fine-tuning practices, and fragmented evaluation standards, thereby hindering effective knowledge accumulation and efficient technological iteration within the field.

To address the aforementioned challenges, this paper proposes a novel Data Curation research framework, GeoChef, which draws a systematic analogy between the development process of RS VLMs and culinary procedures. Within this framework, we reconceptualize the role of the researcher as a Chef who must not only possess keen observational acuity for iterative model optimization but also engage deeply in every stage of data selection and processing. The framework encompasses four key phases. The first phase, Ingredient Atlas, systematically establishes a taxonomic structure for RS vision-language datasets and analyzes the characteristics and evolutionary patterns of different data types. Just as a chef must thoroughly understand the properties of each ingredient, researchers must familiarize themselves with the task attributes and applicable scenarios of various RS data modalities. The second phase, Ingredient Processing, provides a comprehensive methodology for constructing the dataset. Analogous to ingredient preparation in cooking, raw RS data must undergo multiple processing stages to ultimately yield refined, well-annotated, high-quality datasets that preserve the professional characteristics of RS data while conforming to the format specifications required for MLLM training [98], [99]. The third phase, Cooking Manual, elaborates on the selection and implementation of dataset finetuning strategies, enabling researchers to choose optimal fine-tuning approaches based on specific task requirements and resource constraints. The fourth phase, Gourmet Tasting, establishes a comprehensive evaluation framework that introduces commonly used benchmarks and multidimensional evaluation metrics for different tasks [100]–[102]. This evaluation approach resembles the professional assessment of culinary critics, attending to both overall performance and fine-grained details. Throughout the entire research cycle, the Chef (researcher) consistently assumes the central role of observation and iterative optimization, progressively mastering the complete methodology to customize high-performance VLMs for specific RS application scenarios through continuous experimentation, evaluation, and refinement. Through this framework, this paper provides RS researchers with end-to-end guidance from data preparation to model evaluation, facilitating the transformation of researchers from passive data consumers to proactive GeoChef, ultimately enabling qualitative leaps in RS MLLM performance. The main contributions of this paper are threefold:

• More than 120 publicly available visionlanguage datasets in the field of Earth observation are systematically surveyed, with detailed categorization and in-depth analysis conducted according to task types.

• A dynamic, community-driven data warehouse is constructed, support the addition of independent datasets by researchers and enabling multidimensional retrieval, which significantly improves the efficiency of discovering and accessing RS vision-language datasets.

• A step-by-step practical guide is provided for researchers in RS, covering the entire process from data preparation to model evaluation, thereby promoting the development of RS vision-language models toward greater specialization and practicality.

![](images/64e4340e86c62b5cb4985faa2e23c9db7f25340d1b88a0aa8c59aae52a11c46c.jpg)  
Fig. 2. Publication of RS Vision-Language Datasets

The remainder of this paper is organized as follows. Section II introduces the taxonomy and a tree diagram of remote sensing vision-language datasets. Section III elaborates on the details of task-oriented datasets and their evolution trends. Section IV proposes a community-driven repository to facilitate efficient data access for researchers. Section V reviews three mainstream methodologies for constructing datasets. Section VI discusses common fine-tuning strategies for vision-language models. Section VII outlines benchmarks and evaluation metrics for assessing the model. Section VIII presents future research directions. Section IX concludes the paper. The overall framework follows a logical progression from fundamental concepts and methodologies to evaluation and outlook.

## II. THE TAXONOMY OF RS VISION-LANGUAGE DATASETS

This survey provides a systematic analysis of over 100 collected vision-language datasets. Figure 1 illustrate the growth trend of RS vision-language datasets over the past five years. The emergence of MLLMs has significantly accelerated the expansion of RS vision-language data. Notably, the number of RS vision-language datasets doubled from 2023 to 2024. Furthermore, the number of new datasets introduced solely in 2025 has surpassed the cumulative total of all such datasets created before that year. The growth rate observed in 2025 exceeds an exponential pattern, indicating that the field of RS vision-language datasets is currently experiencing a phase of rapid expansion. Additionally, Figure 2 illustrates the distribution of publications related to existing RS vision-language datasets. Aside from works posted on the arXiv preprint server, TGRS is the preferred journal for such papers, accounting for 19 publications. It is followed by journals and conferences such as ISPRS and NeurIPS. Moreover, leading venues, including AAAI, CVPR, GRSM, ICCV, ICLR, and ECCV, have also published research on RS vision-language datasets. This publication distribution suggests that peer-reviewed work in this field generally maintains high quality and attracts broad attention from reputable forums.

![](images/10c8e462039a99f490a3ca36e569bc4d33d0b78e9dc3405d44e5dd79d3d73a26.jpg)  
Fig. 3. Taxonomy of RS Vision-Language Data.

## A. Taxonomy

In existing surveys on RS VLMs, a common practice is to categorize tasks based on model capabilities. However, we observe that such a capabilityoriented taxonomy is inherently non-exhaustive and lacks completeness. Whenever new research emerges, it necessitates continuous supplementation of new capability types, resulting in a classification system that remains in a state of passive expansion. To address this limitation, this paper shifts perspective to the forms of data input and output, proposing a data-centric taxonomy for RS visionlanguage tasks.

As shown in Figure 3, this framework categorizes tasks into three major types, distinguished by their conventional definitions in the RS community and distinct evaluation metrics:

• Image Captioning is primarily evaluated using text similarity metrics, since there often exist multiple valid and natural descriptions for an image, making a simple binary “correct/incorrect” judgment infeasible.

• VQA is uniquely evaluated by objective binary correctness, judging whether the model’s response to multiple-choice, true/false, or open factual questions is definitively right or wrong.

• Visual grounding inherits the evaluation paradigm of classic object detection and segmentation tasks, using spatial overlap metrics to quantify alignment between predictions and ground-truth annotations.

Furthermore, each task category can be subdivided according to the granularity of the image region being processed into image-level, regionlevel, and pixel-level subcategories. Additionally, by extending the input from a single image to multiple images, corresponding temporal tasks can be constructed to accommodate the needs of dynamic scene analysis and change detection. Based on these three dimensions – task type, regional granularity, and temporal dynamics – we systematically organize existing RS vision-language tasks into three major categories and 16 subcategories.

## B. Tree Diagram of Existing RS Vision-Language Datasets

To facilitate researchers in quickly locating datasets of each subcategory, we have created a tree diagram as illustrated in Figure 4. This diagram, comprising three hierarchical levels, vividly showcases the developmental panorama of the current RS visual language domain, and its structure inherently reveals several profound patterns. The most prominent characteristic is the blurring and restructuring of task boundaries. A single dataset frequently appears under multiple seemingly distinct task branches. For instance, EarthDial-Instruct [103] is simultaneously present under description, classification, and localization. This is not a classification error, but rather reveals a core trend. The research paradigm in the field is shifting from serving single tasks towards constructing multitask general-purpose datasets capable of handling description, question answering, localization, and classification. The aim is to train or evaluate more general RS foundation models.

Within this trend, VQA has indisputably become the hub and main battleground of the field. In the diagram, the VQA branch not only has the finest subcategories and contains the most datasets, but has also spurred the creation of numerous specialized evaluation sets often named with terms like Bench or Eval. This signifies the field’s entry into a mature stage of systematically assessing model capabilities, where the research focus shifts from whether a function can be implemented to how capable and comprehensive the model is. Simultaneously, the organizational principle of the tree diagram is distinctly marked by the domain-specific characteristics of RS. It is not simply divided by image-text modalities but is rooted in tasks, and particularly emphasizes two dimensions less standard in other visual fields: Geo-referencing and Temporal sequencing. This intuitively suggests that understanding spatial location and change perception are the core challenges and value propositions of RS visual language research. Finally, the names of the datasets themselves form a concise history of the field’s development. High-frequency terms such as RS, SAR, Geo, and Earth indicate data sources. Terms like Cap, VQA, and VG define core tasks, while Instruct, Bench, and size indicators such as 1M or 10M reveal the current hotspots of instruction tuning, performance benchmarking, and large-scale pre-training, respectively. The entire map clearly delineates a developmental trajectory from specialized to general-purpose, from single-task to multi-task, and from training-driven to evaluation-driven.

## III. THE EVOLUTION OF RS VISION-LANGUAGE DATASETS

This section provides a detailed sequential introduction to five categories of datasets, Image Captioning, VQA, VG, Comprehensive Data, and Benchmark, along with an analysis of their developmental trends. In each subsection, we present the detailed information of the corresponding existing RS vision-language datasets in a table.

## A. Evolution of RS Captioning Datasets

Image captioning, one of the earliest visionlanguage multimodal tasks in RS, aims to describe the content of RS images using natural language. Existing RS image captioning datasets can be broadly categorized into five classes, as outlined in the taxonomy of Figure 5. Each of these classes will be elaborated on in this section. The development of these datasets, as chronicled in Table I, shows a trajectory from small-scale, single-modality collections to large-scale, diverse, and complex data corpora.

![](images/a27c7daf822a66c42ca5de2dfe67d444b40ce0c35dc0c9c79c9f4f9cc9c4b2dd.jpg)  
Fig. 4. The main categorization of this survey.

1) Captioning & Retrieval: In original image captioning tasks, the model is required to generate corresponding textual descriptions from input images. For cross-modal retrieval tasks, the model must identify the most relevant text descriptions given images (image-to-text retrieval) or retrieve the

![](images/d5856061159446878a565912e9aadcd3ce8d515160e97be6a0a75dea6cfe6179.jpg)  
Fig. 5. Rapid Distinction of RS Captioning Tasks. The difficulty level increases with the rightward direction.  
TABLE I

MAINSTREAM RS IMAGE CAPTIONING DATASETS. ABBREVIATIONS: MULTITEMPORAL (MT), INFRARED (IR),   
NEAR-INFRARED (NIR), MULTISPECTRAL (MS), DIGITAL ELEVATION MODEL (DEM), SYNTHETIC APERTURE RADAR   
(SAR), MANUALLY DATA PIPELINE (MANUAL), AUTOMATICALLY DATA PIPELINE (AUTO), SEMI-AUTOMATICALLY DATA PIPELINE (SEMI-AUTO), GROUND SAMPLE DISTANCE (GSD), NOVEL ANNOTATION RATIO (NAR)

<table><tr><td>Name</td><td>Year</td><td>Venue</td><td>#Samples</td><td>Modality</td><td>GSD (m)</td><td>Method</td><td>NAR</td></tr><tr><td>UCM-Captions [136]</td><td>2016</td><td>CITS</td><td>2,100 images, 10,500 captions</td><td>RGB</td><td>0.3048</td><td>Auto</td><td>1.0</td></tr><tr><td>Sydney-Captions [136]</td><td>2016</td><td>CITS</td><td>613 images, 3,065 captions</td><td>RGB</td><td>0.7</td><td>Auto</td><td>1.0</td></tr><tr><td>RSICD [127]</td><td>2017</td><td>TGRS</td><td>10,921 images, 54,605 captions</td><td>RGB</td><td>Diverse</td><td>Manual</td><td>1.0</td></tr><tr><td>RSITMD [128]</td><td>2021</td><td>TGRS</td><td>4,743 images, 23,715 captions</td><td>RGB</td><td>Diverse</td><td>Manual</td><td>1.0</td></tr><tr><td>UAV-Captions [205]</td><td>2021</td><td>TGRS</td><td>2,628 images, 7,884 captions</td><td>RGB</td><td>0.02</td><td>Manual</td><td>1.0</td></tr><tr><td>NWPU-Captions [124]</td><td>2022</td><td>TGRS</td><td>31,500 images, 157,500 captions</td><td>RGB</td><td>0.2~30</td><td>Manual</td><td>1.0</td></tr><tr><td>LEVIR-CC [153]</td><td>2022</td><td>TGRS</td><td>20,154 images, 50,358 captions</td><td>MT</td><td>0.5</td><td>Manual</td><td>1.0</td></tr><tr><td>CapERA [151]</td><td>2022</td><td>RS</td><td>2,864 videos, 1,420 captions</td><td>Video</td><td>Diverse</td><td>Semi-Auto</td><td>1.0</td></tr><tr><td>RSICap [145]</td><td>2023</td><td>ISPRS</td><td>2,585 captions</td><td>RGB</td><td>Diverse</td><td>Manual</td><td>1.0</td></tr><tr><td>LAIOÑ-EO [111]</td><td>2023</td><td>ICCV</td><td>24,933 captions</td><td>Video</td><td>Diverse</td><td>Auto</td><td>0.0</td></tr><tr><td>SkyScript [135]</td><td>2023</td><td>AAAI</td><td>2.6M captions</td><td>RGB</td><td>0.1~30</td><td>Semi-Auto</td><td>1.0</td></tr><tr><td>RemoteCLIP [24]</td><td>2024</td><td>TGRS</td><td>828,725 captions</td><td>RGB</td><td>Diverse</td><td>Auto</td><td>1.0</td></tr><tr><td>RS5M [126]</td><td>2024</td><td>TGRS</td><td>5M captions</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>0.4</td></tr><tr><td>MMM-RS [120]</td><td>2024</td><td>NeurIPS</td><td>2,103,273 captions</td><td>RGB,SAR,NIR</td><td>Diverse</td><td>Semi-Auto</td><td>1.0</td></tr><tr><td>LuoJiaHOG [142]</td><td>2024</td><td>ISPRS</td><td>94,856 images</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>1.0</td></tr><tr><td>MOCO [121]</td><td>2024</td><td>Drone</td><td>7,449 images, 37,245 captions</td><td>RGB</td><td>RGB</td><td>Manual</td><td>1.0</td></tr><tr><td>LRS2M [206]</td><td>2025</td><td>arXiv</td><td>2,023,688 captions</td><td>Diverse</td><td>Diverse</td><td>Auto</td><td>0.5</td></tr><tr><td>RSTeller [146]</td><td>2025</td><td>ISPRS</td><td>1,309,926 images, 2,619,852 captions</td><td>RGB</td><td>0.6</td><td>Auto</td><td>1.0</td></tr><tr><td>GeoLangBind-2M [109]</td><td>2025</td><td>arXiv</td><td>2,050,983 captions</td><td>RGB,SAR,MS,DEM,IR</td><td>Diverse</td><td>Auto</td><td>0.2</td></tr><tr><td>Git-10M [25]</td><td>2025</td><td>GRSM</td><td>10.5M captions</td><td>RGB</td><td>0.5~128</td><td>Semi-Auto</td><td>1.0</td></tr><tr><td>GAIA [108]</td><td>2025</td><td>arXiv</td><td>41,030 images, 205,150 captions</td><td>MS,NIR,SAR,RGB</td><td>Diverse</td><td>Auto</td><td>1.0</td></tr><tr><td>CVG-Text [105]</td><td>2025</td><td>ICCV</td><td>30,000 coordinate points</td><td>RGB</td><td>0.12</td><td>Semi-Auto</td><td>1.0</td></tr><tr><td>Llama3-SSL4EOS12 [207]</td><td>2025</td><td>PKDD</td><td>975,000 images, 1M captions</td><td>MS</td><td>10</td><td>Semi-Auto</td><td>1.0</td></tr><tr><td>SPIE [115]</td><td>2025</td><td>arXiv</td><td>403,432 images</td><td>MS</td><td>Diverse</td><td>Auto</td><td>1.0</td></tr><tr><td>GLEAM [110]</td><td>2025</td><td>arXiv</td><td>504 images, 202,832 captions</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>1.0</td></tr><tr><td>Street2Sat-TEXT [104]</td><td>2025</td><td>GRSM</td><td>36,000 coordinate points</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>1.0</td></tr><tr><td>CVAVT-TEXT [104]</td><td>2025</td><td>GRSM</td><td>44,000 image pairs</td><td>RGB</td><td>Diverse</td><td>Auto</td><td>1.0</td></tr><tr><td>SAR-TEXT [133]</td><td>2025</td><td>arXiv</td><td>136,584 images</td><td>SAR</td><td>Diverse</td><td>Auto</td><td>1.0</td></tr><tr><td>RSCC [144]</td><td>2025</td><td>NeurIPS</td><td>124,630 images, 62,315 captions</td><td>MT</td><td>Diverse</td><td>Semi-Auto</td><td>0.3</td></tr><tr><td>Landsat30-AU-CAP [117]</td><td>2025</td><td>AAAI</td><td>196,262 captions</td><td>RGB</td><td>30</td><td>Semi-Auto</td><td>1.0</td></tr><tr><td>SARCAP [130]</td><td>2025</td><td>ISPRS</td><td>400,000 captions</td><td>SAR</td><td>0.1~20</td><td>Auto</td><td>1.0</td></tr><tr><td>MMSAR [122]</td><td>2025</td><td>TGRS</td><td>105,685 images, 304K captions</td><td>SAR</td><td>Diverse</td><td>Auto</td><td>1.0</td></tr><tr><td>SARVLM-1M [134]</td><td>2025</td><td>arXiv</td><td>365,984 images, 1,727,408 captions</td><td>SAR</td><td>Diverse</td><td>Auto</td><td>0.0</td></tr></table>

most matching images based on text queries (textto-image retrieval). Early RS datasets did not distinguish between these two tasks, with each sample typically comprising a single RS image paired with a concise textual caption. Foundational datasets like UCM-Captions and Sydney-Captions [136], created via automatic pipelines, and the manually annotated RSICD [127] exemplify this category. They established the basic paradigm of single-sentence, imagelevel description for RGB imagery, with scales ranging from a few thousand to tens of thousands of image-caption pairs.

2) Detailed Captioning: With the increasing adoption of LLMs, the captions generated in new datasets have exhibited significant improvements in both textual length and semantic richness (incorporating multiscale spatial relationships and domainspecific terminology). This evolution reflects the growing complexity of vision language alignment requirements in RS applications. Recent large-scale datasets are pivotal for pre-training robust visionlanguage models. For instance, RS5M [126] provides 5 million captions, and RSTeller [146] offers over 2.6 million image-caption pairs. Furthermore, datasets like GeoLangBind-2M [109] and MMM-RS [120] have expanded beyond RGB to include diverse modalities (e.g., SAR, MS, DEM, IR, NIR), capturing richer geospatial semantics. This trend towards massive scale and multi-modality, as seen in SkyScript (2.6M captions) [135] and Remote-CLIP [24], is crucial for developing foundational models with deep domain understanding.

3) Region Captioning: Region captioning distinguishes itself from conventional image captioning by focusing on generating textual descriptions for specific image regions, typically delineated through bounding boxes. This localized approach enables the precise characterization of individual objects or regions of interest, which is particularly valuable for complex visual analysis tasks in RS applications. Datasets supporting this task, e.g, Falcon SFT [106] and GeoLLaVA [152], require fine-grained annotations that link phrases or sentences to specific areas within an image. This represents a move towards more structured and object-aware textual descriptions compared to global image captioning.

4) Grounding Captioning: Grounding Captioning represents an advanced form of detailed image captioning that not only generates comprehensive descriptions of visual content, but also provides precise spatial locations (bounding boxes or masks) for each described object in the image. This technique imposes significant challenges, as it requires the model to establish an accurate point-to-point correspondence between textual descriptions and their corresponding visual elements in the image space. This task can be performed at either the region level or the pixel level, depending on whether the target objects are represented by bounding boxes or masks. It is a natural extension of region captioning with an explicit localization requirement. The creation of such datasets is an active research area, e.g, EarthDial-Instruct [103] and GeoPixelD [154], and often builds upon the foundational data from large-scale captioning collections.

5) Temporal Captioning: Temporal captioning tasks process sequential image inputs (either multiple still frames or video sequences) to generate coherent descriptions of temporal dynamics. In RS applications, the most prominent implementation is change captioning derived from change detection datasets, which requires models to summarize object-level transformations (e.g., construction progress, land cover changes, or disaster impacts) across temporal observations. This task presents unique challenges in spatiotemporal reasoning. Datasets like LEVIR-CC [153], a multitemporal change captioning dataset with over 50k captions, are specifically designed for this purpose. Other resources like CapERA [151], a video captioning dataset, also support modeling temporal sequences. The evolution in this category focuses on capturing what has changed in addition to what is there.

In summary, the development of RS image captioning datasets has evolved from early, small-scale collections with brief, image-level captions for basic tasks towards large-scale, semantically rich corpora that support complex multimodal understanding. This progression is marked by a significant expansion in data volume and caption complexity, driven by advances in foundation models. The focus has shifted to more sophisticated objectives, including fine-grained region description, precise visual grounding, and temporal dynamics analysis. Concurrently, datasets have increasingly incorporated diverse data modalities beyond optical imagery, and annotation methodologies have transitioned from manual efforts to automated, model-assisted processes, collectively enabling deeper and more structured geospatial language-vision alignment.

![](images/19a1ecf0cbfc205afac0c0edda89bb5ac4a4cbe160e12297d21f4a1f7a8fa262.jpg)  
Fig. 6. Rapid Distinction of RS VQA Tasks. The difficulty level increases with the rightward direction.

## B. Evolution of RS VQA Datasets

VQA is one of the most common tasks in visionlanguage multimodal research. It is generally categorized into three types: multiple-choice, true/false (binary), and open-ended questions. Among these, multiple-choice and true/false questions are objective and can be evaluated using accuracy as the primary metric. In contrast, open-ended questions often have non-unique answers, necessitating a variety of evaluation methods. For short-form answers consisting of only one or two words, contentbased accuracy can be used, i.e., the prediction is considered correct if it includes any groundtruth word; otherwise, it is marked incorrect. In addition, synonym-based similarity metrics, such as those used in Next-QA [209], can be employed to account for semantic equivalence. However, for longer open-ended responses, these metrics become less reliable. In such cases, it is common to resort to evaluation using LLMs [145] or human judges [148] to assess the quality and relevance of the generated answers. We categorize existing RS VQA datasets into five primary types, which are described in detail below. This categorization aligns with the taxonomy of vision-language data shown in Figure 6. The mainstream datasets for these categories are summarized in Table II, showing a clear evolution in scale, modality, and task complexity over the years. Below, we describe each category in detail, referencing key datasets from the table.

1) Scene Classification: Scene classification is a domain-specific task in RS, aimed at identifying the land cover type or land use category of an entire image. In the context of vision-language data, scene classification datasets can be viewed as a special form of VQA, where each image is associated with a predefined questiontypically fixed across the dataset (e.g., What is the scene type?) and a corresponding answer label. Unlike general-purpose VQA tasks that involve diverse and image-specific questions, RS scene classification VQA is characterized by its standardized question format and categorical answer space. This structure enables the integration of scene classification into multimodal learning frameworks under the visionlanguage paradigm. While not all datasets in Table II are exclusively for scene classification, several early and foundational datasets, such as the RSVQA series (RSVQA-LR/HR) [32] and RSIVQA [175], incorporate true/false questions that can be framed as binary classification queries about high-level scene attributes (e.g., Is there a residential area?). This serves as a bridge connecting traditional scene classification to the VQA paradigm.

TABLE II  
MAINSTREAM RS VQA DATASETS. ABBREVIATIONS: MULTITEMPORAL (MT), INFRARED (IR), NEAR-INFRARED (NIR), MULTISPECTRAL (MS), DIGITAL ELEVATION MODEL (DEM), SYNTHETIC APERTURE RADAR (SAR), HYPERSPECTRAL (HS), FREE FORMAT (FRE), MULTIPLE CHOICE (CHO), TRUE OR FALSE (T/F), MANUALLY DATA PIPELINE (MANUAL), AUTOMATICALLY DATA PIPELINE (AUTO), SEMI-AUTOMATICALLY DATA PIPELINE (SEMI-AUTO), GROUND SAMPLE DISTANCE (GSD), NOVEL ANNOTATION RATIO (NAR)
<table><tr><td>Name</td><td>Year</td><td>Venue</td><td>#Samples</td><td>Modality</td><td>GSD (m)</td><td>Method</td><td>Type</td><td>NAR</td></tr><tr><td>RSVQA-LR [32]</td><td>2020</td><td>TGRS</td><td>772 images, 77,232 Q&amp;As</td><td>RGB</td><td>10</td><td>Auto</td><td>FRE, T/F</td><td>1.0</td></tr><tr><td>RSVQA-HR [32]</td><td>2020</td><td>TGRS</td><td>10,659 images, 1,066,316 Q&amp;As</td><td>RGB</td><td>0.15</td><td>Auto</td><td>FRE, T/F</td><td>1.0</td></tr><tr><td>RSVQAxBEN [176]</td><td>2021 IGARSS</td><td></td><td>590,326 images, 14,758,150 triples</td><td>RGB</td><td>Diverse</td><td>Auto</td><td>FRE, T/F</td><td>1.0</td></tr><tr><td>FloodNet [164]</td><td>2021</td><td>Access</td><td>2,343 images, 4,500 Q&amp;As</td><td>RGB</td><td>0.015</td><td>Manual</td><td>FRE, T/F</td><td>1.0</td></tr><tr><td>RSIVQA [175]</td><td>2021</td><td>TGRS</td><td>37,264 images, 111,134 Q&amp;As</td><td>RGB</td><td>0.1~8</td><td>Semi-Auto</td><td>T/F</td><td>0.1</td></tr><tr><td>CDVQA [186]</td><td>2022</td><td>TGRS</td><td>2,968 Q&amp;As</td><td>RGB, MT</td><td>0.5~3</td><td>Auto</td><td>FRE, T/F</td><td>1.0</td></tr><tr><td>VQA-TextRS [182]</td><td>2022</td><td>IJRS</td><td>2,144 images, 6,245 Q&amp;As</td><td>RGB</td><td>0.3</td><td>Manual</td><td>FRE, T/F</td><td>1.0</td></tr><tr><td>CRSVQA [160]</td><td>2023</td><td>TGRS</td><td>4,639 images, 4,644 Q&amp;As</td><td>RGB</td><td>0.5~3</td><td>Manual</td><td>FRE</td><td>1.0</td></tr><tr><td>HRVQA [167]</td><td>2023</td><td>ISPRS</td><td>53,512 images, 1,070,240 Q&amp;As</td><td>RGB</td><td>0.08</td><td>Auto</td><td>FRE</td><td>1.0</td></tr><tr><td>EarthVQA [162]</td><td>2023</td><td>AAAI</td><td>6,000 images, 208,593 Q&amp;As</td><td>MS, HS</td><td>0.3~0.5</td><td>Manual</td><td>FRE, T/F</td><td>0.6</td></tr><tr><td>QAG-360K [185]</td><td>2024</td><td>arXiv</td><td>6,810 images, 360K Q&amp;As</td><td>MT</td><td>0.1~3.0</td><td>Semi-Auto</td><td>FRE, T/F</td><td>1.0</td></tr><tr><td>RemoteCount [24]</td><td>2024</td><td>TGRS</td><td>165,745 images, 828,725 Q&amp;As</td><td>RGB</td><td>Diverse</td><td>Auto</td><td>FRE</td><td>1.0</td></tr><tr><td>OSVQA [174]</td><td>2024</td><td>ISPRS</td><td>14,216 images, 1,131,730 Q&amp;As</td><td>SAR, RGB</td><td>Diverse</td><td>Semi-Auto</td><td>CHO, T/F</td><td>1.0</td></tr><tr><td>GeoLLaVA [152]</td><td>2025</td><td>arXiv</td><td>100,000 train, 6,042 test Q&amp;As</td><td>RGB, MS, MT</td><td>0.3~3.7</td><td>Semi-Auto</td><td>FRE</td><td>1.0</td></tr><tr><td>MME-RealWorld [172]</td><td>2025</td><td>ICLR</td><td>1,298 images, 3,738 Q&amp;As</td><td>RGB</td><td>0.07~38.22</td><td>Manual</td><td>CHO</td><td>1.0</td></tr><tr><td>LRS-VQA [171]</td><td>2025</td><td>arXiv</td><td>1,657 images, 7,333 Q&amp;As</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>FRE, T/F</td><td>1.0</td></tr><tr><td>AgroMind [156]</td><td>2025</td><td>arXiv</td><td>20,850 images, 28,482 Q&amp;As</td><td>RGB, NIR, MS</td><td>Diverse</td><td>Auto</td><td>FRE, T/F, CHO</td><td>1.0</td></tr><tr><td>AVI-Math [155]</td><td>2025</td><td>ISPRS</td><td>360 images, 3,773 Q&amp;As</td><td>RGB</td><td>Diverse</td><td></td><td>Semi-Auto FRE, CHO, T/F</td><td>1.0</td></tr><tr><td>Earth-Bench [161]</td><td>2025</td><td>arXiv</td><td>13,729 images, 248 Q&amp;As</td><td>RGB</td><td>Diverse</td><td>Auto</td><td>CHO</td><td>1.0</td></tr><tr><td>VICot-HRSC [181]</td><td></td><td>2025 NerulRS</td><td>364 images, 4,400 Q&amp;As</td><td>RGB</td><td>Diverse</td><td>Manual</td><td>FRE</td><td>1.0</td></tr><tr><td>GeoPlan-Bench [165]</td><td>2025</td><td>arXiv</td><td>1,244 Q&amp;As</td><td>MS, SAR, RGB, MT</td><td>Diverse</td><td>Semi-Auto</td><td>FRE</td><td>1.0</td></tr><tr><td>LRS-GRO [170]</td><td>2025</td><td>arXiv</td><td>1,224 images, 13,245 Q&amp;As</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>FRE</td><td>1.0</td></tr><tr><td>Landsat30-AU-VQA [117] 2025</td><td></td><td>AAAI</td><td>17,725 Q&amp;As</td><td>RGB</td><td>30</td><td>Semi-Auto</td><td>CHO</td><td>1.0</td></tr><tr><td>CitySet [90]</td><td>2025</td><td>ISPRS</td><td>20,589,110, 800K Q&amp;As</td><td>RGB</td><td>0.5</td><td>Manual</td><td>FRE</td><td>1.0</td></tr><tr><td>UnivEARTH [178]</td><td>2025</td><td>ICML</td><td>140 Q&amp;As</td><td>Diverse</td><td>RGB</td><td>Semi-Auto</td><td>T/F</td><td>1.0</td></tr><tr><td>Geo3DVQA [166]</td><td>2025</td><td>arXiv</td><td>110K Q&amp;As</td><td>RGB</td><td>Diverse</td><td>Auto</td><td>FRE, CHO</td><td>1.0</td></tr><tr><td>UrbanVideo-Bench [179]</td><td>2025</td><td>arXiv</td><td>5.2K Q&amp;As</td><td>Video</td><td>Diverse</td><td>Semi-Auto</td><td>CHO</td><td>1.0</td></tr><tr><td>BEDI [158]</td><td>2025</td><td>arXiv</td><td>185 images, 3,097 Q&amp;As</td><td>RGB, Video</td><td>Diverse</td><td>Manual</td><td>CHO, FRE</td><td>1.0</td></tr><tr><td>KnowFlow-Bench [168]</td><td>2025</td><td>arXiv</td><td>324 real RS workflows</td><td>RGB</td><td>Diverse</td><td>Auto</td><td>FRE</td><td>1.0</td></tr><tr><td>MM-UAVBench [173]</td><td>2025</td><td>arXiv</td><td>1.5K videos, 2.8K images, 5.7K instances</td><td>RGB, Video</td><td>Diverse</td><td>Manual</td><td>CHO</td><td>0.8</td></tr><tr><td>LRS-VQA-Zoom [208]</td><td>2025</td><td>arXiv</td><td>302K Q&amp;As</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>FRE, CHO</td><td>1.0</td></tr></table>

2) VQA: In RS image-level VQA tasks, each data sample typically consists of an image, a question, and a corresponding answer. This encompasses the most common form of VQA, dealing with questions about the image content without explicit region reference. As shown in Table II, a vast majority of datasets focus on this task. The field has evolved significantly from early datasets like RSVQA-LR (77k Q&As) [32] to recent large-scale ones such as HRVQA (1.07M Q&As) [167] and OSVQA (1.13M Q&As) [174], demonstrating a clear trend towards increased data volume. The questions are typically free-form (FRE), true/false (T/F), or multiplechoice (CHO). Another notable trend is the expansion of data modalities, from primarily RGB in early works to incorporating multispectral (MS), synthetic aperture radar (SAR), and multi-temporal (MT) data in datasets such as EarthVQA [162], OSVQA [174], and CDVQA [186]. Automated and semi-automated data generation methods have been crucial in scaling up dataset sizes, as seen in RSVQAxBEN (14.76M triples) [176]. A prominent and challenging subtype within image-level VQA is counting, which involves estimating the number of objects (e.g., vehicles, airplanes) in a scene. Dedicated datasets like RemoteCount (828k Q&As) [24] have been created for this purpose. Since the answers are integers, counting is considered a form of short-form open-ended VQA. However, standard accuracy is an inadequate metric for evaluation, as it treats all incorrect counts equally, failing to reflect the magnitude of error. Therefore, regression-based metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) are more appropriate, as used in REO-VLM [210].

3) Region VQA: Region-level VQA differs from image-level VQA in that the question explicitly refers to a specific region within the image. This explicit reference can be conveyed in two primary ways: (1) through natural language questions that include coordinate references in textual form [106], or (2) via visual prompts such as bounding boxes or markers overlaid on the image [147]. The explicit specification of regions acts as a strong visual cue, which can mitigate the referential ambiguity present in standard image-level VQA. Recent datasets are increasingly incorporating this capability. For instance, GeoLLaVA [152] and GeoPlan-Bench [165] are constructed to support region-level understanding and reasoning, where questions or instructions are tied to specific spatial areas within the RS imagery, enabling more precise and interpretable queries.

4) Grounding VQA: Grounding VQA refers to a task that combines VG and VQA. The model is required not only to provide an answer to a given question but also to localize the corresponding answer region within the image, as illustrated in Figure 6. Compared to conventional VQA, this imposes greater demands on the model, as it must both infer the correct answer and accurately identify the specific image region (bounding boxes) that supports it. Models trained on such data exhibit enhanced interpretability, as the localized regions provide visual evidence. In RS, datasets such as LRS-GRO [170] and QAG-360K [185] are designed explicitly for this joint task, requiring the model to generate a bounding box or mask corresponding to the answer for a given question. This represents a more advanced and spatially-grounded form of VQA.

5) Temporal VQA: With the growing capability of vision-language models to process sequential data, temporal VQA tasks have emerged. These tasks take multiple images (often forming a video or a temporal sequence) as input and require models to derive answers by analyzing temporal relationships, changes, or dynamics. In RS, most existing temporal VQA datasets are constructed from change detection data, focusing on queries about pre- and post-disaster comparisons or urban development. Reflecting this trend, recent datasets begin to incorporate temporal or video data. For example, CDVQA [186] uses bi-temporal images for change-related Q&A. More recent datasets, such as BEDI [158], are built upon aerial video imagery, requiring mathematical reasoning over dynamic scenes. Similarly, UrbanVideo-Bench [179] and MM-UAVBench [173] utilize video data to support tasks that require temporal understanding, such as tracking and action recognition, thereby pushing the boundaries of RS VQA from static imagery to spatiotemporal analysis.

Overall, the development of RS VQA datasets is characterized by increased scale, richer modalities, and greater task complexity. The field has progressed from foundational, manually-constructed datasets to large-scale, automatically generated benchmarks, as evidenced by the growth in the number of samples. There is a significant diversification in data modalities, expanding from standard RGB to include multispectral, SAR, and video data. Concurrently, the VQA task formulations have evolved beyond basic image-level questionanswering to incorporate more complex types such as region-referenced, grounded, and temporal VQA.

## C. Evolution of RS Visual Grounding Datasets

RS VG aims to use natural language expressions to locate specific objects (in the form of bounding boxes or segmentation masks) in RS images, thereby enhancing human interaction with intelligent RS interpretation systems. As shown in Figure 7, the evolution of datasets has been pivotal in driving progress in this field. In practical applications, different objects and scenarios necessitate different grounding representations: an HBB can coarsely localize an object’s position, and a mask precisely depicts its shape. As summarized in Table III, a variety of large-scale RS visual grounding datasets have emerged recently, spanning from 2022 to 2025. Early research primarily focused on HBBs, as seen in datasets like RSVG [191] and DIOR-RSVG [188]. However, as the field advances, more diverse grounding types are being adopted to meet complex application needs. Newer datasets increasingly support segmentation masks, such as Ref-SegRS [204], RRSIS-D [201], and the large-scale GeoSeg-1M [196]. This shift reflects the growing demand for more precise, pixel-level localization. The annotation methodologies have also evolved, ranging from manual and semi-automatic to fully automatic data pipelines, significantly improving annotation efficiency and scale. Concurrently, the diversity in GSD and the sheer volume of samples in these datasets have greatly enhanced the robustness and applicability of the developed models.

![](images/b0fe80d4407083865403e2afebebd3157edb25519426682d539878ceb799a5a8.jpg)  
Fig. 7. Rapid Distinction of RS VG Tasks. The three tasks on the left operate at the region level, while those on the right are at the pixel level, with difficulty increasing progressively from left to right on both sides.  
TABLE III  
MAINSTREAM RS VG DATASETS. ABBREVIATIONS: HORIZONTAL BOUNDING BOX (HBB), SEGMENTATION MASK (MASK), MANUALLY DATA PIPELINE (MANUAL), AUTOMATICALLY DATA PIPELINE (AUTO), SEMI-AUTOMATICALLY DATA PIPELINE (SEMI-AUTO), GROUND SAMPLE DISTANCE (GSD), NOVEL ANNOTATION RATIO (NAR)

<table><tr><td>Name</td><td>Year</td><td>Venue</td><td>#Samples</td><td>Modality</td><td>GSD (m)</td><td>Method</td><td>Type</td><td>NAR</td></tr><tr><td>RSVG [191]</td><td>2022</td><td>MM</td><td>4,329 images, 7,933 refers</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>HBB</td><td>1.0</td></tr><tr><td>DIOR-RSVG [188]</td><td>2022</td><td>TGRS</td><td>17,402 images, 38,320 refers</td><td>RGB</td><td>0.5~30</td><td>Semi-Auto</td><td>HBB</td><td>1.0</td></tr><tr><td>GeoText-1652 [189]</td><td>2023</td><td>ECCV</td><td>316,335 refers</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>HBB</td><td>1.0</td></tr><tr><td>RSVG-HR [192]</td><td>2024</td><td>TGRS</td><td>2,650 refers</td><td>RGB</td><td>Diverse</td><td>Manual</td><td>HBB</td><td>1.0</td></tr><tr><td>OPT-RSVG [190]</td><td>2024</td><td>TGRS</td><td>25,452 images, 48,952 refers</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>HBB</td><td>1.0</td></tr><tr><td>RefSegRS [204]</td><td>2024</td><td>TGRS</td><td>4,420 refers across 285 scenes</td><td>RGB</td><td>0.13</td><td>Auto</td><td>Mask</td><td>1.0</td></tr><tr><td>RRSIS-D [201]</td><td>2024</td><td>CVPR</td><td>17,402 refers</td><td>RGB</td><td>0.5~30</td><td>Semi-Auto</td><td>Mask</td><td>1.0</td></tr><tr><td>AAVG [187]</td><td>2025</td><td>arXiv</td><td>26,465 refers</td><td>RGB</td><td>0.007~0.04</td><td>Auto</td><td>Mask</td><td>1.0</td></tr><tr><td>RISBench [199]</td><td>2025</td><td>arXiv</td><td>52,472 refers</td><td>RGB</td><td>0.1~30</td><td>Semi-Auto</td><td>Mask</td><td>1.0</td></tr><tr><td>AerialVG [193]</td><td>2025</td><td>ICCV</td><td>50,000 images, 50,000 refers</td><td>RGB</td><td>Diverse</td><td>Manual</td><td>HBB</td><td>1.0</td></tr><tr><td>EarthReason [202]</td><td>2025</td><td>arXiv</td><td>5,434 masks, 30,000 refers</td><td>RGB</td><td>0.5~153</td><td>Semi-Auto</td><td>Mask</td><td>1.0</td></tr><tr><td>GRES [198]</td><td>2025</td><td>arXiv</td><td>9,205 images, 1M refers</td><td>RGB</td><td>0.3</td><td>Semi-Auto</td><td>Mask</td><td>1.0</td></tr><tr><td>RefDrone [195]</td><td>2025</td><td>arXiv</td><td>8,536 images, 17,900 refers</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>HBB</td><td>1.0</td></tr><tr><td>RemoteSAM-270K [194]</td><td>2025</td><td>MM</td><td>270K refers</td><td>RGB</td><td>Diverse</td><td>Auto</td><td>Mask</td><td>0.7</td></tr><tr><td>NWPU-Refer [203]</td><td>2025</td><td>arXiv</td><td>15,003 images, 49,745 refers</td><td>RGB</td><td>0.12~0.5</td><td>Manual</td><td>Mask</td><td>1.0</td></tr><tr><td>GeoSeg-1M [196]</td><td>2025</td><td>arXiv</td><td>590K images, 1.1M refers</td><td>RGB</td><td>0.05~153</td><td>Auto</td><td>Mask</td><td>1.0</td></tr><tr><td>RIS-LAD [200]</td><td>2025</td><td>AAAI</td><td>13,871 refers</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>Mask</td><td>1.0</td></tr><tr><td>GRASP-1K [197]</td><td>2025</td><td>arXiv</td><td>1,071 refers</td><td>RGB</td><td>0.08~30</td><td>Semi-Auto</td><td>Mask</td><td>1.0</td></tr><tr><td>LaSeRS [169]</td><td>2025</td><td>arXiv</td><td>40K masks, 30K refers</td><td>RGB</td><td>Diverse</td><td>Semi-Auto</td><td>Mask</td><td>1.0</td></tr></table>

1) Referring Expression Comprehension (REC): The input to REC is an RS image paired with a referring expression, and the output is a single corresponding bounding box of the uniquely referred object in the image. Most early datasets (e.g., RSVG [191], DIOR-RSVG [188], and GeoText-1652 [189]) typically describe objects using their relative positions within the image, as well as appearance attributes such as color, size, and shape.

Subsequently, AVVG [187] has shifted its focus toward referential description based on 3D spatial locations and fine-grained attributes of vehicles. This evolution addresses the need for precise object localization and identification in complex urban environments. For the REC task, the IoU between the predicted bounding box and the ground truth box is directly employed as the evaluation metric. Alternatively, a threshold (e.g., IoU 0.5) can be set to compute accuracy, where a prediction is correct if its IoU exceeds the threshold.

2) Generalized REC (GREC): GREC extends REC to multi-object localization and hallucination detection, where no referred object exists. This formulation is more aligned with real-world applications. GeoChat-Instruct [33] represents an early example of a GREC dataset in the RS domain. Notably, conventional RS object detection can be viewed as a specialized GREC task where the referring expressions are predefined object categories. Common object detection metrics, such as Mean Average Precision (mAP), require confidence scores, which many MLLMs do not provide natively. To address this, LMMRotate [211] introduces $\mathrm { m A P } _ { n c }$ and $\mathrm { m F _ { 1 } }$ metrics designed for models without confidence scores, which are recommended for evaluating GREC. The development of GREC reflects the field’s progression towards handling more complex and realistic scenarios, where models must not only identify relevant objects but also recognize when no object matches the description, thereby reducing false positives in practical applications.

3) Temporal GREC: T-GREC extends the input from a single image to multi-image sequences. A canonical example is change detection, where given bi-temporal images, the model must localize changed regions by reasoning across time. This demands spatio-temporal understanding, making it more challenging than standard GREC. To our knowledge, TEOChatlas [183] is the first and only RS dataset designed to support this specific task. This task requires the model to understand both spatial relationships within individual images and temporal dynamics across image sequences, enabling applications such as urban development monitoring, disaster assessment, and agricultural growth tracking. The complexity of T-GREC lies in the need for joint spatial-temporal reasoning, where models must distinguish actual changes from seasonal variations, illumination differences, and other confounding factors.

4) Referring Expression Segmentation (RES): Similar to REC, RES aims to localize a uniquely referred object, but it outputs a precise segmentation mask instead of a bounding box. The development of datasets supporting mask output, as shown in Figure 7, has facilitated research in this direction. Representative datasets include RefSegRS [204], AAVG [187], and RRSIS-D [201]. Influenced by the reasoning segmentation tasks [79] in the field of general MLLMs, research in this domain has begun to exhibit a marked shift from simple explicit reference towards tasks that require reasoning for implicit reference, with notable examples including GRASP-1K [197], and GeoSeg-1M [196]. Evaluation can use an IoU-based accuracy threshold. Standard segmentation metrics such as mean IoU and overall IoU are also applicable for a comprehensive assessment, where IoU is calculated between masks. The transition from bounding boxes to segmentation masks represents a significant advancement in localization precision, enabling applications that require detailed shape information, such as building footprint extraction, crop boundary delineation, and infrastructure monitoring.

5) Generalized RES (GRES): GRES extends RES to support multiple objects and empty referrals, corresponding to a more generalized node in the task taxonomy. Analogous to how object detection is a form of GREC, RS semantic segmentation can be viewed as a specialized GRES task. The recently proposed RemoteSAM-270K dataset [194] is a representative benchmark for this task. Since it involves potentially multiple disconnected regions, it generalizes beyond the single-mask output of RES. GRES addresses the practical need to localize multiple objects described in complex natural language expressions, which is essential for applications like inventory counting, damage assessment, and environmental monitoring. The development of GRES datasets and methods reflects the field’s growing maturity in handling more sophisticated referring expressions that may involve multiple objects, complex relationships, and potentially no relevant objects in the image.

6) Temporal GRES: T-GRES advances the paradigm by requiring segmentation outputs from multiple input images. The primary application is in change detection, where the goal is to segment the changed regions between images taken at different times. This task combines temporal reasoning with precise mask prediction, presenting greater complexity than T-GRES, which only outputs bounding boxes for changes. Datasets ChangeChat-87K [53] and Falcon SFT [106] are currently identified as the two primary benchmarks supporting the T-GRES task. This task requires models to perform finegrained pixel-level segmentation of changes while ignoring irrelevant variations caused by factors like illumination, season, or sensor characteristics. The emergence of T-GRES represents the cutting edge of RS visual grounding, integrating temporal analysis with precise segmentation to support applications such as monitoring construction progress, assessing disaster impacts, and tracking land use changes over time.

In summary, the field of RS VG has undergone rapid evolution. As evidenced by the survey of mainstream datasets, there is a clear trend from early models using horizontal bounding boxes towards recent, more precise segmentation masks, alongside advancements in annotation methods from manual to automatic pipelines. Correspondingly, the task taxonomy has expanded from the foundational REC to include more complex and practical scenarios, such as generalized, temporal, and segmentation-based variants, reflecting the growing sophistication and breadth of application in the research domain.

## D. Evolution of Comprehensive Datasets

Beyond the aforementioned single-task RS vision-language datasets, numerous studies have proposed comprehensive datasets and benchmarks supporting multiple tasks to enhance the overall capabilities of MLLMs, as exemplified in Table IV and Table V. Training data are undergoing explosive growth in scale, rapidly escalating from hundreds of thousands of instructions in 2023 to tens of millions (e.g., 11.11 million instructions in EarthDial-Instruct) and even nearly a hundred million (e.g., 78 million instructions in Falcon SFT) by 2025. The construction methodologies heavily rely on automated pipelines to support this massive scale. Concurrently, data modalities are rapidly diversifying from the early dominance of RGB, with SAR, infrared, multispectral, and even hyperspectral data being widely integrated, fostering specialized datasets for domains like change detection and disaster assessment .

Currently, RS vision-language data stands on the cusp of a transition from quantitative accumulation to qualitative breakthrough. Its development trajectory is clear: continuous expansion in scale, increasing modality fusion, and more specialized tasks. Future evolution directions are even more defined: First, a balanced emphasis on both scale expansion and quality improvement, enhancing the complexity, diversity, and accuracy of instructions while shifting the paradigm from ”scale-first” to ”quality-and-efficiency parity.” Second, deep multimodal fusion, moving beyond simple juxtaposition of data types towards pursuing cross-modal alignment and synergistic understanding, enabling complementary use of multi-source data in complex tasks like disaster response and land cover change detection. Third, task design is trending towards complexity and practicality, evolving from basic description and Q&A to comprehensive tasks requiring spatiotemporal reasoning, causal analysis, and decision support.

These trends collectively propel comprehensive datasets to become the cornerstone for training the next generation of powerful and general-purpose RS MLLMs. The core of future evolution will shift from pursuing sheer data volume towards building a new generation of data infrastructure characterized by high quality, deep fusion, strong reasoning capabilities, and effective evaluation. This necessitates closer collaboration among data builders, algorithm researchers, and domain experts to jointly forge a next-generation data system capable of empowering large RS models to achieve deep visual understanding, complex spatiotemporal reasoning, and reliable

TABLE IV  
MAINSTREAM RS COMPREHENSIVE DATA. ABBREVIATIONS: MULTITEMPORAL (MT), INFRARED (IR), MULTISPECTRAL (MS), SYNTHETIC APERTURE RADAR (SAR), MANUALLY DATA PIPELINE (MANUAL), AUTOMATICALLY DATA PIPELINE (AUTO), SEMI-AUTOMATICALLY DATA PIPELINE (SEMI-AUTO), GROUND SAMPLE DISTANCE (GSD), NOVEL ANNOTATION RATIO (NAR)
<table><tr><td>Name</td><td>Year</td><td>Venue</td><td>#Samples</td><td>Modality</td><td>Method</td><td>GSD (m)</td><td>NAR</td></tr><tr><td>GeoChat-Instruct [33]</td><td>2023</td><td>CVPR</td><td>318,000 instructions</td><td>RGB</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>MMRS-1M [119]</td><td>2024</td><td>TGRS</td><td>1M instructions</td><td>RGB, SAR, IR</td><td>Auto</td><td>Diverse</td><td>0.0</td></tr><tr><td>VRSBench-Train [149]</td><td>2024</td><td>NeurIPS</td><td>142K instructions</td><td>RGB</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>RS-GPT4V [143]</td><td>2024</td><td>arXiv</td><td>107,936 images</td><td>RGB</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>RSVP [147]</td><td>2024</td><td>TGRS</td><td>3.65M instructions</td><td>RGB</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>LHRS-Align/Instruct [112]</td><td>2024</td><td>ECCV</td><td>1.1898M instructions</td><td>RGB</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>FIT-RS [139]</td><td>2024</td><td>arXiv</td><td>1,800,851 instructions</td><td>RGB, IR, NIR, HS</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>VersaD/HnstD/VariousRS [148]</td><td>2024</td><td>AAAI</td><td>1.5214M instructions</td><td>RGB</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>LHRS-Align-Recap/Instruct [113]</td><td>2024</td><td>ISPRS</td><td>1.18M instructions</td><td>RGB</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>ChangeChat-87K [53]</td><td>2024</td><td>ICASSP</td><td>87,195 instructions</td><td>MT</td><td>Semi-Auto</td><td>Diverse</td><td>0.4</td></tr><tr><td>EarthDial-Instruct [103]</td><td>2025</td><td>CVPR</td><td>11.11M instructions</td><td>RGB, SAR, MS</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>SkyEye-968K [212]</td><td>2025</td><td>ISPRS</td><td>968K instructions</td><td>RGB</td><td>Semi-Auto</td><td>Diverse</td><td>0.03</td></tr><tr><td>DDFAV [138]</td><td>2025</td><td>RS</td><td>1,755 images, 27,650 instructions</td><td>RGB</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>GeoPixelD [154]</td><td>2025</td><td>ICML</td><td>53,816 instructions</td><td>RGB</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>Falcon_SFT [106]</td><td>2025</td><td>arXiv</td><td>5.6M images, 78M instructions</td><td>RGB, MS, SAR</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>TEOChatlas [183]</td><td>2025</td><td>ICLR</td><td>1,244,393 images, 554,071 instructions</td><td>MT</td><td>Auto</td><td>Diverse</td><td>0.6</td></tr><tr><td>GeoPixInstruct [141]</td><td>2025</td><td>GRSM</td><td>65,463 images, 140,412 instructions</td><td>RGB</td><td>Semi-Auto</td><td>Diverse</td><td>0.7</td></tr><tr><td>SARLANG-1M [132]</td><td>2025</td><td>ICLR</td><td>118,331 images, 1,126,277 instructions</td><td>SAR</td><td>Semi-Auto</td><td>0.1~25</td><td>1.0</td></tr><tr><td>M-RSVP [213]</td><td>2025</td><td>arXiv</td><td>650,000 instructions</td><td>RGB, SAR, IR</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>RS-VL3M [129]</td><td>2025</td><td>arXiv</td><td>3M instructions</td><td>SAR, IR, RGB</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>FUSAR-GEOVL-1M [107]</td><td>2025</td><td>arXiv</td><td>120K images, 1M instructions</td><td>SAR</td><td>Auto</td><td>0.2~3</td><td>1.0</td></tr><tr><td>ScoreRS [114]</td><td>2025</td><td>NeurIPS</td><td>131K instructions</td><td>RGB</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>DisasterM3 [88]</td><td>2025</td><td>NeurIPS</td><td>26,988 image pairs</td><td>RGB, SAR</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>RS-Eot-4K [163]</td><td>2025</td><td>arXiv</td><td>4.3K instructions</td><td>RGB, SAR, IR</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr></table>

TABLE V

MAINSTREAM RS COMPREHENSIVE BENCHMARKS. ABBREVIATIONS: MULTITEMPORAL (MT), MULTISPECTRAL (MS), SYNTHETIC APERTURE RADAR (SAR), BITEMPORAL (BT), PANCHROMATIC (PAN), MANUALLY DATA PIPELINE (MANUAL), AUTOMATICALLY DATA PIPELINE (AUTO), SEMI-AUTOMATICALLY DATA PIPELINE (SEMI-AUTO), GROUND SAMPLE DISTANCE (GSD), NOVEL ANNOTATION RATIO (NAR)
<table><tr><td>Name</td><td>Year</td><td>Venue</td><td>#Samples</td><td>Modality</td><td>Method</td><td>GSD (m)</td><td>NAR</td></tr><tr><td>RSIEval [145]</td><td>2023</td><td>ISPRS</td><td>100 images, 100 captions, 936 Q&amp;As</td><td>RGB</td><td>Manual</td><td>Diverse</td><td>1.0</td></tr><tr><td>GeoChat-Bench [33]</td><td>2023</td><td>CVPR</td><td>105K images, 318K instructions</td><td>RGB, MS</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>LHRS-Bench [112]</td><td>2024</td><td>ECCV</td><td>108 questions, 690 Q&amp;As</td><td>MS, PAN</td><td>Manual</td><td>Diverse</td><td>1.0</td></tr><tr><td>GEOBench-VLM [140]</td><td>2025</td><td>ICCV</td><td>10,000 instructions</td><td>RGB, MS, SAR, BT, MT</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>UrBench [184]</td><td>2025</td><td>AAAI</td><td>11,600 instructions</td><td>RGB</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>XLRS-Bench [150]</td><td>2025</td><td>CVPR</td><td>1,400 images, 91K instructions</td><td>RGB</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>CHOICE [214]</td><td>2025</td><td>NeurIPS</td><td>10,507 instructions</td><td>RGB, MT, BT, MI</td><td>Semi-Auto</td><td>0.1~30</td><td>1.0</td></tr><tr><td>AirSpatial [215]</td><td>2025</td><td>TGRS</td><td>14K refers, 17K Q&amp;As</td><td>RGB</td><td>Auto</td><td>0.007~0.04</td><td>1.0</td></tr><tr><td>SARChat-Bench-2M [131]</td><td>2025</td><td>arXiv</td><td>2,063,548 instructions</td><td>SAR</td><td>Auto</td><td>0.3~10</td><td>1.0</td></tr><tr><td>REOBench [125]</td><td>2025</td><td>arXiv</td><td>37,400 instructions</td><td>RGB</td><td>Auto</td><td>Diverse</td><td>1.0</td></tr><tr><td>ThinkGeo [177]</td><td>2025</td><td>arXiv</td><td>486 tasks, 1773 instructions</td><td>RGB, SAR</td><td>Semi-Auto</td><td>0.015~15</td><td>1.0</td></tr><tr><td>A2Seek [157]</td><td>2025</td><td>arXiv</td><td>542 videos</td><td>Video, RGB, IR</td><td>Semi-Auto</td><td>Diverse</td><td>1.0</td></tr></table>

geospatial decision-making.

## E. Demarcation Between Atomic and Composite Datasets

With the rapid advancement of MLLMs, a growing number of composite datasets have been introduced, which are constructed by aggregating existing image-text datasets and reformatting them into standardized instruction-following formats. In contrast to atomic datasets, which are built entirely with de novo novel annotations from scratch, composite datasets represent a distinctive data paradigm unique to the era of MLLM.

![](images/06f0167d0338c39ab87369a18871099d88a4353771cd920e35b9fdbee3a0c315.jpg)  
Fig. 8. A Practical Guide to Data Choice, Curation, Fine-Tuning, and Evaluation. With GeoChef, researchers can master an iterative observe-refine cycle to train models that excel in practical RS scenarios.

For image captioning datasets, LAION-EO [111] is constructed by filtering RS images from general image-text pairs, resulting in a Novel Annotation Ratio (NAR) of 0; it is a purely composite dataset. For the LRS2M [206] dataset, half of its samples are obtained by integrating existing datasets, while the other half are generated via model-based augmentation from short captions, giving it a NAR of 0.5. For VQA datasets, a large number of questions in the RSIVQA [175] task are generated via template-based transformationfor example, by directly converting RS image classification tasks into VQA tasksresulting in a NAR of only 0.1. For VG datasets, 30% of the samples in RemoteSAM-270K [194] are derived from the integration of existing datasets, yielding a NAR of 0.7. For comprehensive multi-modal datasets, the MMRS-1M [119] and SkyEye-986k [212] datasets are constructed almost entirely through the integration or transformation of existing datasets. In contrast, all corresponding comprehensive benchmarks have a NAR of 1.

Notably, these two dataset categories are neither mutually exclusive nor mutually antagonistic; instead, they constitute a synergistic, complementary ecosystem that underpins the full development lifecycle of RS VLM. When evaluated solely on their intrinsic contribution at the dataset level, composite datasets do not outperform atomic datasets, yet low-NAR composite datasets deliver a scalable data foundation for large-scale pre-training and model initialization.

## IV. COMMUNITY-DRIVEN DATA WAREHOUSE

After familiarizing ourselves with existing RS vision-language datasets, we devised a fourstep framework, illustrated in Figure 8, to support researchers in iterative development, culminating in the creation of GeoChat. For a chef, the first and foremost step is to select the appropriate ingredients. Similarly, for a researcher, the first step is to select the appropriate data according to their specific needs. As shown in Figure 9, this section begins by presenting the foundational component of the project: a communitydriven data warehouse.

![](images/9a83d0954135f8d626717662a1358e1192d5d2031cb3e2bb60a2ca80fbc84bfa.jpg)  
Fig. 9. Our Open-source Project maintained on Github.

To address the challenges of fragmented data resources and the discontinuation of maintenance for existing open-source repositories in the field of RS vision-language research, this paper introduces a community-driven open-source data warehouse, Awesome-RS-VL-Data<sup>1</sup>, to provide sustained and comprehensive data support for related studies. As an emerging paradigm for data management, the core of a community-driven data warehouse lies in its approach to aggregating, maintaining, and updating data resources through open collaboration, thereby overcoming the limitations in sustainability and scalability inherent in traditional closed databases.

The open source of this data warehouse followed a prolonged period of preparation and internal refinement, including an initial trial operation within internal scenarios for over six months. The primary objective was to accumulate a substantial volume of high-quality dataset resources. Through continuous expansion, the repository now hosts more than 120 datasets, establishing a solid foundation for open sharing with the community. Furthermore, recognizing that other open-source repositories<sup>2345</sup> on similar topics have suffered from update stagnation – often due to shifts in research team focus, graduation of core maintainers, or other commitments – leading to data resources failing to keep pace with research advancements, we decided to formally open-source our internally curated and mature dataset inventory. This initiative aims to fill the gap for a sustainably maintained data warehouse in this domain. Moving forward, the core development team will continue to participate in the repository’s updates and improvements, while also leveraging the power of the community to achieve long-term, iterative data refinement.

<table><tr><td>Open 1,388</td><td>Closed 15</td></tr><tr><td> the rain #648 - zytx121 opened on Jun 20, 2018</td><td>Author 口1</td></tr><tr><td>happy endings</td><td>1</td></tr><tr><td>#647 - zytx121 opened on Jun 20, 2018</td><td></td></tr><tr><td>Anistemi #637 · zytx121 opened on Jun 19, 2018</td><td>口1</td></tr><tr><td>□ Buds Crown #636 - zytx121 opened on Jun 19, 2018</td><td>1</td></tr><tr><td>□ninelie</td><td>口1</td></tr><tr><td>#644 - zytx121 opened on Jun 20, 2018</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Asphodelus</td><td>1 #643 · zytx121 opened on Jun 19, 2018</td></tr></table>

Fig. 10. A community-driven harmonica score library. Currently, 1,388 harmonica scores have been collected, 99% of which were contributed by community users.

Compared to traditional private data warehouses, the community-driven model represents the core advantage of this repository. Private data warehouses typically rely on maintenance by specific research teams or individual personnel, making their continuity and updates highly vulnerable to factors such as changes in research direction or turnover of key personnel, often resulting in dataset obsolescence and project abandonment. In contrast, our data warehouse adopts a community-self-maintenance model, opening both the permissions and responsibilities for data upkeep to the entire community. This fundamentally mitigates the sustainability risks associated with a single maintaining entity. The feasibility of this model has been successfully validated through another open-source project we initiated: a harmonica score library<sup>6</sup>. By establishing a unified data format and initially manually extracting and standardizing tablature information before transitioning to community self-maintenance after open-sourcing, this project has stably operated for eight years. The community has spontaneously contributed and maintained nearly 1,400 harmonica tablatures, fully demonstrating the significant advantages of the community-driven model in resource accumulation and project sustainability. Even if the core development team suspends direct maintenance due to unforeseen circumstances, the community can still propel the continuous development of the data warehouse.

![](images/e9d5610090094ab0bf14e388f538cb6c8947171ced5ba06e89ce0998a9c5a2e6.jpg)  
Fig. 11. A mobile app developed based on our data warehouse. It can help researchers real-time grasp the dynamics of RS visionlanguage datasets.

In terms of functional design and user experience, this data warehouse incorporates several distinctive features. First, it employs a multi-dimensional label classification system to finely annotate the included datasets. It enables researchers to quickly locate resources that meet their needs. Second, it adheres to a mechanism of continuous updates, closely tracking research trends in the RS vision-language field and consistently incorporating new datasets and related information. This ensures the timeliness and comprehensiveness of the data resources, providing up-to-date support for research and applications. Third, leveraging the GitHub Issues ecosystem and the official GitHub REST API v3, the warehouse possesses powerful extensibility, supporting thirdparty platform development. The adaptive website GeoCef, built upon this data warehouse, achieves cross-device compatibility and can be seamlessly used on computers, mobile phones, tablets, and other terminals, as shown in Figure 11. This allows researchers to utilize fragmented time to track the latest dataset dynamics and search for target resources, thereby providing inspiration and highquality corpus support for research topic selection and model development.

In addition to the GitHub-based open-source data repository functionality, the README interface of this project aggregates resources for RS visual language datasets, including access links to papers and data, as well as corresponding references and data links to the original RS datasets. Furthermore, we have compiled a list of recently published largescale RS vision-language models and incorporated other high-quality Awesome resource lists in this field, aiming to provide researchers with one-stop information support. This community-driven data warehouse not only establishes an efficient datasharing platform for RS vision-language research but also fosters a virtuous cycle of community contribution, resource accumulation, research empowerment. In the future, we anticipate broader participation from researchers in maintaining and refining the data warehouse, collectively enriching the data ingredients to provide solid support for the innovative development of next-generation RS intelligent models.

![](images/ab13f601f00c05ff2e207c6992b1de479433f2b9dbbb70b8517faff474246da8.jpg)  
Fig. 12. Distribution of Data Construction Methods Across the Entire and Different Types of Datasets.

## V. CONSTRUCTION OF RS VISION-LANGUAGE DATASETS

Dataset construction is the second step to becoming a GeoChef. After selecting the suitable raw data materials, preliminary processing must be performed to prepare for the subsequent cooking process. The construction methodology of RS vision-language datasets is pivotal to their quality and applicability. Based on the level of human involvement and algorithmic automation, mainstream data construction pipelines can be broadly categorized into three paradigms: the completely manual pipeline [124], [128], [150], [153], the completely automatic pipeline [21], [24], [108], [109], [115], [136], [146], and the semi-automatic pipeline [104], [110], [117], [135], [144], [151], [207]. The choice among these paradigms involves a fundamental trade-off between data quality, production efficiency, diversity, and the requirement for expert knowledge. The following subsections detail the characteristics, advantages, disadvantages, and typical use cases of each approach.

## A. Manually Data Pipeline

The manually data construction pipeline relies entirely on human experts throughout the entire process. In this paradigm, there is no algorithmic involvement; the generation, cross-validation, and iterative optimization of data samples are all performed by humans. This approach is particularly suited for constructing high-precision, small-sample datasets that demand deep domain expertise, such as benchmark datasets designed for rigorous evaluation. Its foremost advantage lies in the exceptional diversity and high quality of the curated data, as expert knowledge ensures accuracy and richness. However, the primary drawback is its low efficiency and high dependency on skilled human resources, making it prohibitively expensive and impractical for large-scale captioning or generative tasks. A representative successful case of this paradigm is the XLRS-Bench [150], a high-quality benchmark for evaluating RS vision-language models.

## B. Automatically Data Pipeline

In contrast, the automatic data construction pipeline operates without any human intervention. It typically starts from existing datasets and employs predefined rules or algorithms to automatically generate templated data samples. This method is highly suitable for building large-scale datasets for repetitive or structured tasks. Its core advantage is high efficiency; once the generation rules and algorithms are verified to be correct, the produced data theoretically requires no manual inspection, enabling rapid scaling. The main disadvantage is the potential lack of data diversity, as the output is constrained by the initial templates and rules. Nevertheless, this limitation can be mitigated by employing LLMs to paraphrase and enrich the generated text. A notable example is the GeoChat [33] dataset, constructed automatically for geo-localized dialogue.

## C. Semi-automatically Data Pipeline

The semi-automatic pipeline represents a synergistic approach that balances human expertise and algorithmic efficiency, making it the most commonly adopted method in industrial practice.

![](images/255b8677e4e1ee8ba5828b812b9dc69b8dae057f38dd64ae22596ea238a97630.jpg)  
Fig. 13. Distribution of Data Generation Models in Papers Using Semi-Automatic Data Construction.

![](images/5cba1b116515cda07147702b0cbb86172297fbb4d1b01e06e35a575e1a112fe8.jpg)  
Fig. 14. Multi-Agent Annotation Systems and Quality Control Paradigms.

In this paradigm, a model (e.g., a large visionlanguage model) is primarily responsible for the initial data generation, while human annotators are tasked with verification, correction, and optimization. This collaborative workflow effectively balances speed with quality control. The advantages are significant: it yields diverse data quickly by leveraging the generative capacity of models, while human oversight ensures a reliable quality floor. The key disadvantage is the continued requirement for manual verification, which can be applied either to the full dataset or through sampling, thus incurring non-negligible but manageable labor costs. This approach is exemplified by the construction of the VRSBench [149] dataset, where model generation was coupled with human review.

## D. Analysis about Data Construction Methods

Figure 12 quantifies the distribution of multimodal data construction methods. Semi-automation dominates overall at 56.25%, peaking at 65.52% in captioning datasets. Manual annotation varies drastically, reaching its highest share (31.48%) in VQA datasets, driven by the tasks inherent reasoning complexity and demand for precise visual-text alignment. Fully automatic methods remain secondary and stable (15.52%26.09%), with maximum adoption in the VG dataset, indicating structuredoutput tasks are more amenable to automation. These patterns reveal a clear correlation between task nature and construction paradigm: descriptionheavy tasks prioritize semi-automation for scalability, while reasoning-intensive tasks require greater manual intervention. The persistent minority status of automatic methods underscores the fundamental challenge of high-quality unsupervised multimodal data generation, highlighting the need for advanced human-in-the-loop frameworks or generative models to improve efficiency for complex tasks.

Figure 13 further analyzes model adoption in recent semi-automatic data construction studies. GPT-4o [15] emerges as the undisputed dominant tool, used in nearly 20 surveyed papers, owing to its robust multimodal capabilities, high instruction fidelity, and reliable API performancecritical for generating coherent, task-aligned synthetic data. Beyond this dominant model, adoption becomes highly fragmented. A secondary tier includes GPT-4V, GPT-4 [15], Qwen2-VL [86], and GPT-3.5 [216]. The broader ecosystem remains in an active exploratory phase, with researchers evaluating tradeoffs between the cost-effectiveness of smaller models, customizability of open-source options, and specialized capabilities for specific data types. This has led to adoption of alternatives including Claude-3.5 [217], Gemini variants [218], and leading opensource vision-language models [16]–[18], [219].

## E. AI Agent-Driven Data Construction Paradigm and Quality Control

The AI agent-driven paradigm represents the next evolutionary stage of semi-automatic data construction, replacing the traditional “single-model generation + human verification” workflow with a collaborative multi-agent system [114], [171] that automates complex, multi-step reasoning and curation processes. Unlike single-model approaches plagued by inherent biases and limited reasoning capabilities, multi-agent systems leverage specialized division of labor to achieve higher annotation quality and scalability. As shown in the Figure 14, the most prevalent and practical implementation is the dual-agent architecture [171], consisting of a generator agent producing initial annotations and an independent evaluator agent filtering low-quality or incorrect samples. This separation of generation and evaluation effectively mitigates confirmation bias and overconfidence issues.

Building on dual-agent systems, recent research has proposed tri-agent architectures to address the challenging task of constructing high-quality chainof-thought (CoT) reasoning data, which is critical for training RS-VL models with advanced spatial and logical reasoning abilities. A representative example is the RS-EoT-4K dataset pipeline [114], which employs a Socratic questioning framework involving three specialized agents: Reasoner, Perceiver, and Verifier. The Reasoner is blinded to image content and prompted that its collaborator has ”weak perceptual capabilities”, forcing it to decompose complex spatial problems into simple incremental questions based on local visual features. Conversely, the Perceiver with full image access is prompted that its collaborator has ”weak reasoning capabilities”, compelling it to provide precise, concise answers without extraneous information. Their iterative Q&A exchange naturally unfolds into step-by-step reasoning trajectories, which are finally validated for logical consistency and factual accuracy by the Verifier agent. This self-play-based Socratic approach produces more interpretable and logically sound reasoning traces than direct singlemodel CoT generation, setting a new benchmark for reasoning-focused RS-VL dataset construction.

Despite these advancements, MLLMs are not infallible. In the RS domain with highly specialized knowledge and ambiguous visual patterns, modelgenerated annotations inevitably contain errors. Thus, human-in-the-loop quality control remains indispensable for high-quality RS-VL datasets. The traditional human-centric quality control pipeline [220] consists of four sequential stages: annotator training, initial human annotation, cross-validation by multiple annotators, and expert final confirmation, which relies almost entirely on human labor and is extremely time-consuming and expensive. In contrast, the agent-augmented pipeline shifts the majority of the workload to AI agents, evolving into four new stages: prompt engineering, MLLM initial generation, multi-agent cross-validation, and expert final confirmation, with human intervention limited to only the first and last critical stages. The required intensity of human verification varies by data usage. For training sets where minor noise is tolerable, random sampling checks are sufficient; failed samples can be manually corrected or discarded. However, for test sets used to evaluate model performance and draw scientific conclusions, 100% full manual verification by domain experts is mandatory, as any errors can lead to misleading performance comparisons and invalid research findings.

Nevertheless, the AI agent-driven paradigm still faces several limitations: current multi-agent systems struggle with highly specialized RS knowledge, their performance is highly dependent on prompt engineering quality, and standardized metrics for evaluating agent-generated reasoning data are lacking. Future research should focus on developing domain-specialized agents, exploring dynamic role assignment architectures, and establishing rigorous benchmarks for agent-constructed datasets, which will further accelerate the development of next-generation RS MLLMs.

![](images/4612032c533b8f70f3fbd511895b21b4e162e9538d60b1a278422121e25b5f11.jpg)  
Fig. 15. Frequency of Major RS Meta Datasets Used to Construct Vision-Language Datasets.

## F. Best Practices for Constructing High-Quality RS Vision-Language Dataset

In RS vision-language research, redundant dataset construction and inconsistent quality standards waste valuable resources and hinder model development. As illustrated in Figure 15, a vast amount of RS metadata is repeatedly repurposed to build RS vision-language datasets. Among them, the classic RS object detection dataset DOTA [38] has been repeatedly constructed as a visionlanguage dataset in 17 studies, resulting in significant redundancy. To address these pervasive challenges, this guide introduces a streamlined threestep workflow to help researchers efficiently construct high-quality datasets.

• Step 1: Pre-construction Redundancy Check: Before construction, consult Table VI to map base RS datasets to their existing vision-language extensions. If similar annotations exist, extend the current dataset or focus on underrepresented domains instead of rebuilding from scratch to reduce redundant workload.

• Step 2: Task-Specific Semi-Automatic Annotation: Full manual annotation is costly and unscalable, so adopt a task-specific semiautomatic pipeline: use RS-fine-tuned VLMs for image-level captioning with human verification; combine pre-trained RS object detectors with region-specific caption generators for region-level descriptions; use template-based methods for factual QA and general MLLMs for reasoning QA, with mandatory human validation for all annotations.

• Step 3: Rigorous Quality Control : Please follow the two pipelines shown in Figure 14 for quality management. All datasets should include a transparent public quality report recording construction details, standards and limitations.

This standardized workflow minimizes redundancy, ensures dataset reliability, and accelerates RS vision-language model development. For users who find manual table consultation cumbersome, our GeoChef agent provides query services: after receiving your needs, it generates personalized dataset construction recommendations.

## VI. FINE-TUNING OF RS VISION-LANGUAGE DATASETS

The third step towards becoming a GeoChef involves fine-tuning the model using the prepared vision-language data. This step can be likened to a chef cooking a dish, as it critically determines the final performance of the model. Although suitable ingredients have been carefully selected and preprocessed, improper control of the culinary process could undermine previous efforts. Thus, this phase is of great importance.

TABLE VI  
A QUICK LOOKUP TABLE FOR PRE-CONSTRUCTION REDUNDANCY CHECK.
<table><tr><td>RS Dataset</td><td>Cap</td><td>VQA</td><td>VG</td><td>Comprehensive Data</td></tr><tr><td>DOTA [38]</td><td>LRS2M, RSICap</td><td>RSIVQA, RemoteCount</td><td>GRSAP-1K, GeoSeg- 1M, RemoteSAM-270K, RISBench</td><td>RSVP, LHRS-Align-Recap &amp; LHRS-Instruct-Plus, SkyEye- 968K, VRSBench-Train, DDFAV, VersaD &amp; HnstD &amp; VariousRS- Instruct, MMRS-1M, UrBench, REOBench</td></tr><tr><td>DIOR [221]</td><td>LRS2M, Git-10M</td><td>RemoteCount</td><td>RemoteSAM-270K, RISBench, GRSAP-1K, GeoSeg-1M, OPT-RSVG, DIOR-RSVG</td><td>VRSBench-Train, GeoChat- Instruct, DDFAV, MMRS-1M, SARCHat-Bench-2M</td></tr><tr><td>FAIR1M [40]</td><td></td><td>LRS-VQA, MME-RealWorld</td><td>GeoSeg-1M, GRSAP-1K</td><td>LHRS-Align-Recap &amp; LHRS- Instruct-Plus, DDFAV, GeoChat- Instruct, VersaD &amp; HnstD &amp; VariousRS-Instruct, MMRS-1M</td></tr><tr><td>HRRSD [222]</td><td>LRS2M</td><td>RSIVQA, RemoteCount</td><td>OPT-RSVG, RemoteSAM- 270K</td><td>RSVP, MMRS-1M</td></tr><tr><td>Potsdam [223]</td><td>LRS2M</td><td>MME-RealWorld</td><td>GeoSeg-1M, RemoteSAM- 270K</td><td>RSVP, VersaD &amp; HnstD &amp; VariousRS-Instruct, SARCHat- Bench-2M, UrBench</td></tr><tr><td>FMoW [224]</td><td>RS5M, MMM-RS</td><td>GeoLLaVA</td><td>EarthReason</td><td>VersaD &amp; HnstD &amp; VariousRS- Instruct, EarthDial-Instruct, TEOCHatlas</td></tr><tr><td>RSITMD [128]</td><td></td><td>RemoteCount</td><td></td><td>RSVP, SkyEye-968K, LHRS-Align &amp; LHRS-Instruct, RS-GPT4V, Ver- saD &amp; HnstD &amp; VariousRS-</td></tr><tr><td>LOVEDA [225]</td><td>LRS2M</td><td>EarthVQA, RemoteCount</td><td>GeoSeg-1M, RemoteSAM- 270K</td><td>Instruct, MMRS-1M VersaD &amp; HnstD &amp; VariousRS- Instruct</td></tr></table>

Given that this paper primarily targets universities and small-to-medium-sized research institutions with limited computational resources, which often struggle to bear the high costs of training large models from scratch, it does not cover large-scale pre-training [226], [227]. Instead, it focuses on deploying effective fine-tuning strategies based on pretrained general-purpose MLLMs [228], [229]. As an efficient transfer learning approach, fine-tuning can rapidly adapt large models to specific domains or tasks under constrained resources [230], [231]. This chapter will first clarify the core differences and applicable scenarios between two technical paths:

full-parameter fine-tuning (FPFT) [232], [233] and parameter-efficient fine-tuning (PEFT) [234], [235]. Subsequently, it systematically introduces the basic methods of supervised fine-tuning (SFT) [236], [237] and discusses the principles and advantages of reinforcement fine-tuning (RFT) [238], [239] techniques that have emerged in this year. By systematically reviewing different fine-tuning strategies, this chapter aims to guide researchers in flexibly selecting appropriate methods based on data characteristics, fully leveraging the potential of data to construct higher-performance MLLMs.

## A. FPFT v.s. PEFT

FPFT represents the most classical approach to model adaptation. Its core principle involves unlocking all parameters of a pre-trained model and performing end-to-end training updates on a taskspecific dataset [240]. This method aims to meticulously adjust the weights of each layer according to the requirements of the new task, thereby maximizing the model’s potential to achieve optimal taskspecific performance. In the domain of RS, VHM [148] performed FPFT on 16 NVIDIA A100-80G GPUs, while EarthDial [103] employed 8 identical GPUs for its training. However, this comprehensive adjustment comes at a significant cost in computational resources and time, making it particularly unsuitable for environments with limited resources, such as universities and small-to-medium-sized research institutions. Furthermore, each model finetuned in this manner requires storing a complete parameter copy, leading to substantial storage overhead. It also tends to be prone to overfitting when the dataset is small.

To address the challenges above, PEFT techniques have emerged. The core idea is to keep the main parameters of the pre-trained model fixed, introducing and training only a minimal adapter to learn task-specific knowledge. Among various PEFT techniques, LoRA (Low-Rank Adaptation) [241] stands out as one of the most mainstream and practical methods. For example, GeoChat [33] freezes the visual encoder and projection layer, finetuning only the LLM with LoRA to inject RS knowledge into LLaVA. SkyEyeGPT [212] freezes the visual encoder and fine-tunes both the LLM and projection layer with LoRA. RS-LLaVA [242], by contrast, fine-tunes all three components simultaneously with LoRA. Some methods combine both approaches, such as LHRS-Bot [112] which uses FPFT for connection layers and LoRA for the LLM.

FPFT and PEFT offer complementary strengths. FPFT is preferred for peak-performance tasks, GPU-rich environments, large domain shifts, and data-abundant scenarios. PEFT, by contrast, is optimal for resource-limited settings, rapid deployment, multi-task learning, and data-scarce applications where overfitting is a major concern [243]–[245].

## B. SFT v.s. RFT

SFT is fundamentally an imitative learning paradigm [246], [247]. Its core mechanism involves guiding models to learn task-specific answer distributions by minimizing the loss between predictions and ground-truth labels using large-scale highquality input-output annotated data, analogous to students learning from examples. In RS applications, the FPFT and PEFT methods, which were introduced earlier, all fall under the category of SFT. It excels in simple implementation, stable training, low computational cost, rapid achievement of excellent stable performance on well-defined tasks with sufficient high-quality data, and strong output style controllability. Its main limitations include heavy reliance on costly annotated data, confinement to training data distributions, lack of active innovation, poor generalization to dynamically changing or open-ended tasks, and significant overfitting risks.

RFT is an advanced fine-tuning technique built on SFT [248]. It introduces a reinforcement learning (RL) framework that uses reward signalsrather than fixed input-output annotationsto drive model optimization. Recent RS research shows substantial gains from RFT. The Geo-R1 [249] paradigm enforces a ”reason first, then act” process for fewshot geospatial referring expression understanding, where the model first generates explicit reasoning chains before localizing target objectsconsistently and substantially outperforming SFT baselines across three few-shot benchmarks while demonstrating strong cross-dataset generalization. DVG-Bench employs RFT to train an implicit-to-explicit reasoning mechanism for UAV RS imagery, significantly enhancing visual reasoning grounding capabilities. RFT’s core mechanism is trial-and-error learning: a preset reward function quantifies how well the model output meets task objectives, and RL algorithms like policy gradients iteratively update the model to maximize rewards. RFT overcomes the annotation limitations of SFT, improving output quality, consistency, task alignment, generalization, and reliability. However, it is ineffective for samples that are always wrong or always correct; its main benefit is on near-decision-boundary samples that are sometimes correct, sometimes not. Therefore, selecting such samples as training data maximizes RFTs optimization gains.

In practical application scenarios, SFT and RFT are not opposing optimization options but form a progressive relationship of basic laying and advanced improvement in the large model iteration process. For most large model deployment scenarios, it is recommended to first complete SFT training using high-quality annotated data to quickly drive the model to adapt to the core needs of the task. GeoZero [250] employs a two-stage finetuning strategy: first SFT, then RFT. It first applies SFT to simple problems to build a solid foundation for the model, and then uses RFT on challenging problems to further unlock the models potential.

However, if it is necessary to carry out research related to Chain-of-Thought (CoT) [181], building on SFT is an inevitable path. Taking exploration research on RS CoT as an example, the general technical route usually first uses a large amount of high-quality CoT data to perform SFT, enabling the model to master basic question-answering capabilities and the ability to output thinking processes, thus becoming a qualified model with basic task adaptability. On this basis, the RFT mechanism is introduced to further optimize the models answer quality and CoT logic through a carefully designed reward function [251]. Nevertheless, the lack of high-quality CoT-annotated data in the RS domain poses a significant challenge, as directly employing data without CoT annotations for SFT tends to diminish the MLLMs ability to produce coherent reasoning chains [252]. To mitigate this issue, two primary strategies have been explored: one leverages advanced closed-source models to automatically generate CoT-annotated SFT samples, while the other applies RFT directly to general-purpose MLLMs, bypassing domain-specific SFT entirely [253]. SocraticAgent [163] developed a self-play system to generate high-quality CoT-annotated SFT data. The system comprises two AI MLLMs: the Reasoner and the Perceiver, which produce training data by simulating Socratic dialogues. DroneVG-R1 [254] did not perform SFT and directly used manually annotated implicit-to-explicit CoT for RFT.

## VII. EVALUATION OF RS MLLMS

Model evaluation serves as the final step in the GeoChef pipeline, marking both the completion of a new iterative cycle and the commencement of a new one. Just as the quality of culinary dishes must be judged by customer feedback, the performance of a trained model must be evaluated against standardized benchmarks. Based on the evaluation results, researchers can refine data selection strategies, data preprocessing procedures, and finetuning approaches, thereby continuously improving the modelmuch like honing ones culinary skills. We summarize the most common public benchmarks and their corresponding evaluation metrics for 4 RS vision-language tasks in Tables VII, VIII, IX, and X, which can serve as a reference for newcomers to the field. In addition to utilizing existing benchmarks, researchers may also construct specialized benchmarks to verify the superiority of their proposed methods for specific problems under fair testing conditions.

## A. Evaluation on RS Image Captioning Tasks

To ensure a reasonable and well-grounded evaluation, we select five complementary metrics tailored to the multi-dimensional requirements of RS image captioning (lexical accuracy, semantic fidelity, fluency, and informational richness). BLEU-4 [256] measures 4-gram precision to assess domain terminology accuracy and syntactic correctness, while METEOR [257] compensates for its limitations by emphasizing semantic equivalence and aligning with human judgments. ROUGE-1 (unigram coverage) and ROUGE-L (longest common subsequence) [258] evaluate the capture of key geographic features and spatial coherence, respectively. CIDEr [259], designed specifically for image captioning, uses TF-IDF weighting to highlight informational salience by prioritizing rare, discriminative terms.

Table VII reveals a complex and nuanced competitive landscape within the field of RS image captioning. The GeoZero [250] model emerges as a particularly notable contender. It demonstrates performance approaching the performance ceiling on the UCM-Captions [136] dataset, exemplified by its remarkably high METEOR score of 83.75. This indicates an exceptional capability in generating fluent descriptions with near-perfect semantic alignment, suggesting a powerful fitting ability in specific, well-structured scenarios. However, a clear specialization bias is observed across the models. The RS-CapRet [255] model achieves a standout CIDEr score of 250.20 on the most challenging RSICD [127] dataset, significantly surpassing all other models. This result highlights its unique advantage in producing highly informative and salient captions. Yet, this superiority is not entirely sustained on the Sydney dataset [136], implying potential boundaries to its generalizability.

TABLE VII  
COMPARISON OF MAINSTREAM RS MLLMS ON 3 RS IMAGE CAPTIONING BENCHMARKS.
<table><tr><td>Dataset</td><td>Model</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE_1</td><td>ROUGE_L</td><td>CIDEr</td></tr><tr><td rowspan="6">RSICD</td><td>RSGPT [145]</td><td>36.83</td><td>30.10</td><td></td><td>53.34</td><td>102.94</td></tr><tr><td>GeoChat [33]</td><td></td><td>13.48</td><td>11.59</td><td></td><td>12.39</td></tr><tr><td>SkyEyeGPT [212]</td><td>59.99</td><td>35.35</td><td></td><td>62.63</td><td>83.65</td></tr><tr><td>RS-CapRet [255]</td><td>43.40</td><td>37.00</td><td></td><td>63.30</td><td>250.20</td></tr><tr><td>Earthdial [103]</td><td></td><td>56.18</td><td>33.77</td><td>27.61</td><td></td></tr><tr><td>GeoZero [250]</td><td>35.35</td><td>52.31</td><td>60.16</td><td>54.47</td><td>114.82</td></tr><tr><td rowspan="9">UCM-Captions</td><td>RSGPT [145]</td><td>65.74</td><td>42.21</td><td></td><td>78.34</td><td>333.23</td></tr><tr><td>GeoChat [33]</td><td></td><td>14.40</td><td></td><td>13.22</td><td>14.27</td></tr><tr><td>SkyEyeGPT [212]</td><td>78.41</td><td>46.24</td><td></td><td>79.49</td><td>236.75</td></tr><tr><td>RS-CapRet [255]</td><td>67.00</td><td>47.20</td><td></td><td>81.70</td><td>354.80</td></tr><tr><td>RS-LLaVA [242]</td><td>72.84</td><td>47.98</td><td></td><td>85.17</td><td>349.43</td></tr><tr><td>Earthdial [103]</td><td></td><td>51.42</td><td>40.00</td><td>34.15</td><td></td></tr><tr><td>RingMo-Agent [129]</td><td>77.63</td><td>51.79</td><td></td><td>85.51</td><td>373.68</td></tr><tr><td>GeoZero [250]</td><td>75.72</td><td>83.75</td><td>84.60</td><td>83.75</td><td>393.57</td></tr><tr><td>RSGPT [145]</td><td>62.23</td><td>41.37</td><td></td><td>74.77</td><td></td></tr><tr><td rowspan="6">Sydney-Captions</td><td></td><td></td><td>10.63</td><td>12.00</td><td></td><td>273.08</td></tr><tr><td>GeoChat [33]</td><td>77.40</td><td></td><td></td><td>11.26</td><td></td></tr><tr><td>SkyEyeGPT [212]</td><td>56.40</td><td>46.62 38.80</td><td></td><td>77.74 70.70</td><td>181.06 239.20</td></tr><tr><td>RS-CapRet [255] Earthdial [103]</td><td></td><td>57.31</td><td>49.39</td><td>41.00</td><td></td></tr><tr><td></td><td>64.08</td><td></td><td>75.72</td><td></td><td></td></tr><tr><td>GeoZero [250]</td><td></td><td>75.12</td><td></td><td>74.12</td><td>293.49</td></tr></table>

In summary, no single model establishes absolute dominance across all datasets and evaluation metrics, underscoring the current absence of a universally superior solution. Model performance is highly dependent on the alignment between its training objectives and the distribution of the test data. The future competitive focus will likely center on developing a general-purpose model that exhibits balanced and robust capabilities across diverse and challenging real-world scenarios.

## B. Evaluation on RS VQA Tasks

In RS VQA tasks, accuracy [260] is generally regarded as the primary evaluation metric. For counting-oriented questions or regression-based tasks, regression metrics such as mean absolute error (MAE) [261] are commonly adopted to quantify prediction errors. In this survey, we uniformly use Overall Accuracy (OA) [32] as the core quantitative evaluation metric across all four RS VQA benchmarks, as it has become the de facto standard in the RS vision-language community. Formally, the OA metric is defined as the percentage of test samples where the models predicted answer is completely consistent with the human-annotated ground truth.

As shown in Table VIII, leading models have achieved scores approaching or exceeding 90 on the conventional RSVQA-LR and HR datasets [32], indicating that the VQA task for RS images at standard resolutions is nearing maturity. However, a drastic performance decline is observed on the ultrahigh-resolution benchmarks, XLRS-Bench [150] and LRS-VQA [171], where most reported scores fall within the 2050 range. This significant gap highlights the fundamental challenges posed by ultra-high-resolution imagery, including perceiving fine-grained visual patterns, filtering critical information from massive pixel data, and comprehending complex long-range spatial relationships. These challenges likely push the capabilities of existing mainstream model architectures to their limits.

TABLE VIII  
COMPARISON OF MAINSTREAM RS MLLMS ON 4 RS VQA BENCHMARKS. AMONG THEM, XLRS-BENCH AND LRS-VQA BELONG TO THE ULTRA-HIGH-RESOLUTION DATASET.
<table><tr><td>Model</td><td>RSVQA-LR RSVQA-HR XLRS-Bench LRS-VQA</td><td></td><td></td><td></td></tr><tr><td>RSGPT [145]</td><td>92.29</td><td>89.78</td><td></td><td></td></tr><tr><td>GeoChat [33]</td><td>91.81</td><td>70.82</td><td>22.03</td><td>19.38</td></tr><tr><td>SkyEyeGPT [212]</td><td>84.19</td><td>81.89</td><td></td><td></td></tr><tr><td>EarthGPT [119]</td><td></td><td>71.15</td><td></td><td></td></tr><tr><td>VHM [148]</td><td>89.12</td><td>74.35</td><td></td><td>26.69</td></tr><tr><td>RSUniVLM [262]</td><td>92.05</td><td>90.85</td><td></td><td></td></tr><tr><td>RS-LLaVA [242]</td><td>93.04</td><td></td><td></td><td></td></tr><tr><td>LHRS-Bot-Nova [113]</td><td></td><td>88.45</td><td></td><td></td></tr><tr><td>Earthdial [103]</td><td></td><td>71.00</td><td></td><td></td></tr><tr><td>RingMo-Agent [129]</td><td></td><td>79.58</td><td></td><td></td></tr><tr><td>ScoreRS [114]</td><td></td><td>81.55</td><td></td><td></td></tr><tr><td>GeoZero [250]</td><td></td><td>88.92</td><td>48.1</td><td></td></tr><tr><td>ImageRAG [263]</td><td></td><td></td><td></td><td>27.92</td></tr><tr><td>ZoomSearch [264]</td><td></td><td></td><td></td><td>32.92</td></tr></table>

Furthermore, model development exhibits a clear trend toward specialization. One group of models aims to achieve state-of-the-art performance on specific tasks, serving as domain-specific experts. For instance, RS-LLaVA [242] achieves the highest score on RSVQA-LR, GeoZero [250] leads by a significant margin on XLRS-Bench, and Zoom-Search [264] performs best on LRS-VQA.

Notably, GeoZero presents a promising exception. It maintains high competitiveness on the traditional high-resolution task RSVQA-HR while achieving a commanding lead on the ultra-highresolution XLRS-Bench. This outcome suggests that its technical approach may possess superior generalization and robustness in handling visual information across different scales and complexities, offering valuable insights for developing nextgeneration, more powerful and general RS visionlanguage models.

## C. Evaluation on RS VG Tasks

For RS VG tasks, different metrics are adopted based on the number of target objects. Specifically, IoU@0.5 accuracy [266] is used for single-object referring grounding to assess the localization accuracy of the target object. For multi-object referring grounding, AP@50 [38] is employed to comprehensively evaluate the models grounding performance. It should be noted that a higher IoU threshold imposes stricter requirements on grounding precision, and IoU@0.5 is widely recognized as the standard threshold in the field of RS VG.

TABLE IX  
COMPARISON OF MAINSTREAM RS MLLMS ON 4 RS REGION-LEVEL VG BENCHMARKS. AMONG THEM, AVVG BELONGS TO THE HIGH-RESOLUTION DATASET.
<table><tr><td>Model</td><td>RSVG</td><td>DIOR-RSVG</td><td>VRSBench</td><td>AVVG</td></tr><tr><td>GeoChat [33]</td><td>2.04</td><td>24.05</td><td>11.52</td><td>0.28</td></tr><tr><td>LHRS-Bot [112]</td><td>1.56</td><td>17.59</td><td>1.19</td><td>0.00</td></tr><tr><td>SkySenseGPT [139]</td><td></td><td>12.87</td><td></td><td></td></tr><tr><td>LHRS-Bot-Nova [113]</td><td></td><td>31.51</td><td></td><td></td></tr><tr><td>VHM [148]</td><td></td><td>56.17</td><td></td><td></td></tr><tr><td>GeoGround [187]</td><td>26.65</td><td>77.73</td><td>66.04</td><td>21.58</td></tr><tr><td>RSUniVLM [262]</td><td></td><td>72.47</td><td>69.31</td><td></td></tr><tr><td>ScoreRS [114]</td><td></td><td>64.52</td><td></td><td></td></tr><tr><td>GeoZero [250]</td><td>50.04</td><td>79.43</td><td>73.32</td><td></td></tr><tr><td>RSCoVLM [208]</td><td>53.79</td><td>84.55</td><td>79.73</td><td>29.40</td></tr></table>

TABLE X

COMPARISON OF MAINSTREAM RS MLLMS ON 2 RSPIXEL-LEVEL VG BENCHMARKS. AMONG THEM,EARTHREASON BELONGS TO THE DATASET OF REASONINGSEGMENTATION.
<table><tr><td rowspan="2">Model</td><td colspan="3">RRSIS-D</td><td colspan="2">EarthReason</td></tr><tr><td>P@0.5</td><td>oIoU</td><td>mIoU</td><td>cIoU</td><td>gIoU</td></tr><tr><td>GeoGround [187]</td><td>67.50</td><td></td><td>60.50</td><td>-</td><td>-</td></tr><tr><td>RemoteSAM [194]</td><td></td><td>80.04</td><td>71.75</td><td></td><td></td></tr><tr><td>GeoPixel [154]</td><td>83.33</td><td>84.90</td><td>67.30</td><td>53.90</td><td>52.53</td></tr><tr><td>SegEarth-R1 [202]</td><td>76.96</td><td>78.01</td><td>66.40</td><td>68.25</td><td>70.75</td></tr><tr><td>RemoteReasoner [265]</td><td>54.29</td><td></td><td>50.97</td><td>69.13</td><td>70.96</td></tr><tr><td>UniGeoSeg [196]</td><td></td><td>81.35</td><td>72.23</td><td>75.67</td><td>73.12</td></tr><tr><td>SegEarth-R2 [169]</td><td>=</td><td></td><td>67.90</td><td>69.50</td><td>73.50</td></tr></table>

Table IX reveals a significant performance gap between advanced and early MLLMs in regionlevel VG. Leading models, such as RSCoVLM [208] and GeoZero [250], score above 50 points across datasets, including RSVG [191], DIOR-RSVG [188], and VRSBench [149], demonstrating strong geolocation capabilities. In contrast, earlier models, such as GeoChat [33] and LHRS-Bot [112], generally score below 25 points, with performance on the high-resolution AVVG dataset [187] approaching zero. This gap highlights rapid technological progress in the field. The AVVG dataset serves as a challenging benchmark, where all models perform significantly worse; even the top model, RSCoVLM, achieves only 29.40 points.

Table X further illustrates a trend of capability differentiation in pixel-level RS MLLMs, marked by a competition between basic segmentation accuracy and advanced reasoning. UniGeoSeg [196] excels in both tasks, particularly on the challenging EarthReason dataset [202], demonstrating strong generalization and reasoning abilities for complex instructions that involve geographical knowledge. It can be considered an all-round performer. Conversely, GeoPixel [154] leads in traditional segmentation on the RRSIS-D [201] dataset, particularly in the precise localization metric P@0.5; however, its performance declines on the EarthReason task. This suggests that models optimized for traditional segmentation may not be designed for the highlevel semantic understanding required by reasoningintensive tasks. Additionally, SegEarth shows minimal change in basic segmentation metrics from version R1 to R2 [169], but a clear improvement in reasoning, indicating a focused effort on enhancing reasoning capabilities, which aligns with the field’s move toward a deeper geographical understanding.

In summary, the two tables above clearly outline a watershed in the current development of RS MLLMs: models are no longer solely pursuing accurate vision but are also diverging into a new track of deep reasoning.

## D. Limitations of Current Evaluation

The rapid proliferation of RS MLLMs has exposed fundamental flaws in existing evaluation paradigms, where inflated performance metrics often bear little relation to a model’s true generalization ability. A core and pervasive issue is information leakage, which undermines the validity of benchmark comparisons and misguides research directions. We first systematically categorize three distinct yet interconnected information leakage pathways prevalent in RS-VLM benchmarks:

• Pre-training leakage has become particularly prominent as large-scale pre-trained VLMs [24], [122] have become a standard paradigm. Current general-domain pre-training corpora such as CLIP [2] incorporate billions of RS images and text descriptions [21], leading downstream task evaluation to essentially become a seen-answer test that severely overestimates the models zero-shot and few-shot performance. However, the risk of pre-training leakage is often overlooked in the era of large models.

• Metadata leakage occurs because the visuallanguage datasets used for training and testing are constructed based on the same RS metadatasets. As shown in Fig. 15, as many as 17 RS visual-language datasets have been built on DOTA [38], resulting in severe metadata leakage risks among these datasets. The presence of identical images in the test and training sets can still inflate evaluation accuracy, even if the text annotations are not fully consistent.

• Reasoning grounding leakage arises when reasoning referring texts are directly annotated on the basis of original referring expressions. With the development of visual reasoning, an increasing number of reasoning grounding datasets [169], [197], [202], [254] have emerged. Some of these datasets directly generate reasoning texts from existing VG datasets without changing the referred objects, allowing MLLMs fine-tuned on non-reasoning VG datasets to blindly predict the answers to reasoning grounding tasks.

Beyond the three types of information leakage discussed previously, discrepancies in evaluation implementation details can also undermine the impartiality of benchmarking. In what follows, we elaborate on three evaluation criteria that are frequently neglected:

• Prompting Consistency Criteria: For benchmark-focused papers, evaluate representative MLLMs zero-shot performance on the proposed benchmark [150]. Ensure task prompts are consistent with the original literature, as prompt variations significantly affect task performance – e.g., Qwen2.5-VLs support for absolute and 1000 [86] – scaled normalized coordinates causes large precision gaps in VG.

• Answer Extraction Criteria: For VQA tasks with stringent output format constraints, final answers can be extracted via regular expression matching or constrained through multi-turn dialogue to enforce standardized output formats [155]. Uniform answer extraction strategies must be applied across all evaluated models to ensure consistent and comparable results.

• SFT Data Consistency Criteria: For finetuning studies [249], [250], ensure all baseline MLLMs are fine-tuned on the same training dataset for fair comparison. Test sets often align with training data, so fine-tuned models gain advantages that may obscure architectural or strategy contributions, a detail sometimes overlooked to highlight method superiority.

## E. Future Direction of Benchmarks

At this stage, RS vision-language benchmarks have gradually evolved from single-temporal and single-modal to multi-temporal and multi-modal settings [183], from image chips to large-scale RS scenes [150], and from shallow semantic matching to deep cognitive reasoning [254].

However, two promising trends can be foreseen in future benchmark development. First, benchmarks are shifting from general capability evaluation to domain specialization. They are moving beyond generic scene understanding toward professional scenarios such as disaster response [88], agricultural monitoring [156], and photogrammetry [155], enabling fine-grained assessment of domain semantics, and expert knowledge alignment. Second, evaluation paradigms are transitioning from model-centric testing to agent-oriented assessment [267], driven by the explosive rise and widespread adoption of AI agent technologies. Unlike standalone large models that are mostly constrained to static reasoning within fixed knowledge boundaries, AI agents can execute far more complex, multi-step real-world tasks through flexible tool invocation and modular collaborative workflows [177].

## VIII. CHALLENGES AND FUTURE DIRECTIONS

## A. Reasoning Tasks

The research focus of RS vision-language models is shifting from fundamental perception tasks to more challenging, complex reasoning tasks. This transition stems from a core challenge: conventiona tasks such as object detection [268] and semantic segmentation [269] can often be effectively accomplished without leveraging the deep comprehension and reasoning capabilities of ML’LMs, making it difficult to fully exploit the advantages of MLLMs. To break through this bottleneck, researchers have begun constructing a series of new benchmark datasets that require deep intent understanding and logical reasoning, aiming to evaluate and promote model performance at higher cognitive levels. These emerging reasoning tasks primarily present challenges across several dimensions. Implicit VG [202] requires models to understand instructions that imply specific attributes or functional relationships rather than directly referring to object appearance. Temporal reasoning [183] demands that models analyze multi-temporal RS images to understand dynamic processes such as urban expansion and disaster evolution. Relational reasoning [139] challenges models to comprehend intricate interactive relationships among multiple ground objects within an image. 3D spatial reasoning [215] elevates the challenge from the two-dimensional image plane to the three-dimensional spatial level, requiring an understanding of object height, volume, and 3D spatial relationships. Multimodal mathematical reasoning [155] extends the challenge to integrating mathematical logic and visual information, testing the model’s ability to transform visual information into mathematical models and perform calculations. In summary, the frontier challenges in RS visionlanguage data for reasoning tasks are centered on advancing models beyond basic seeing and recognizing towards deeper cognitive capabilities. These challenges are guiding RS vision-language research towards more profound and general spatial intelli gence.

## B. Ultra-High-Resolution RS Imagery

The processing of very large-scale RS imagery poses a systemic challenge for current RS MLLMs, spanning the dimensions of data, algorithms, and computational resources. From a data foundation perspective, the cost of constructing high-quality benchmarking and training datasets remains prohibitively high. Such imagery requires fine-grained annotation reliant on specialized RS domain knowledge. While purely manual annotation can ensure quality, it suffers from efficiency limitations. For instance, the development of benchmarks like XLRS-Bench [150] necessitates annotating dozens of tasks for a single image; this extremely time-consuming nature directly constrains the accumulation of largescale, high-quality, contextually rich data. Through manual annotation of 4K UAV imagery, the AVVG [187] created a high-resolution vehicle VG dataset, which provides fine-grained attributes including vehicle color, brand, and model.

At the model input and information retention level, a fundamental contradiction exists. Drastically compressing imagery with tens of thousands of pixels down to the standard size acceptable for models results in a severe loss of rich spatial contextual information and fine object details [263]. Conversely, avoiding compression is often infeasible as the original image dimensions far exceed the processing limits of most models, preventing them from effectively capturing long-range dependencies. Since the core of RS interpretation is precise measurement of surface information, this contradiction ultimately results in subpar model performance in critical tasks such as object counting and finegrained object identification.

Furthermore, analyzing internal processing efficiency highlights the pronounced challenge posed by the low semantic density characteristic of RS imagery. Vast areas of ineffective background, such as oceans and forests, prompt the visual encoder to generate a massive number of redundant visual tokens [171]. These tokens contain minor meaningful semantics, consume scarce computational resources, and can interfere with the modeling of crucial targets.

In summary, addressing these challenges requires continued and in-depth exploration from researchers in areas such as data and image construction, model algorithm innovation [264], and the development of evaluation benchmarks.

## C. Multi-source RS Imagery

Currently, a critical limitation in the field is that the vast majority of existing automatic and semi-automatic vision-language data construction pipelines are still largely restricted to optical imagery, due to the lack of powerful and reliable intelligent agents capable of processing multi-spectral [108], synthetic aperture radar (SAR) [132], and other specialized remote sensing (RS) modalities. This makes it difficult for existing general-purpose MLLMs to directly and accurately interpret their visual content, thereby severely degrading the reliability of automatically and semi-automatically generated image-text paired data. Particularly for SAR images, their nature as syntheses of radar echoes necessitates deep domain expertise for object recognition, posing substantial difficulties for automated data annotation.

To address this challenge, the community has explored various innovative solutions. One effective strategy is modality transfer. For instance, the SARLANG-1M [132] dataset adopts this approach by leveraging paired SAR and optical images: a general vision-language model first generates descriptions for the more interpretable optical images, and these descriptions are then transferred to the corresponding SAR images as textual annotations. This method skillfully circumvents the difficulty of directly interpreting specialized imagery, though it usually still requires expert manual review to ensure final quality. Another approach involves constructing more sophisticated automated annotation frameworks to improve description quality. For example, SAR-TEXT [133] developed a pipeline that first uses a model to generate preliminary captions from optical images, simultaneously converting segmented images into textual descriptions via algorithms, and then feeds both pieces of information into a large language model. Leveraging its incontext learning capability, the framework produces high-quality, detailed descriptions. Although this method demands higher input requirements (paired SAR images, optical images, and segmentation maps), it can yield very fine-grained annotations. When paired optical images are unavailable, researchers have also explored methods that do not rely on external paired data. The SARCAP [130] dataset provides a typical example: it directly utilizes existing object detection or semantic segmentation annotations of SAR images. It generates corresponding textual descriptions based on predefined rules and templates. Similarly, the OSVQA [174] dataset first annotates basic attributes of images under expert guidance and then uses templates to generate diverse visual question-answer pairs. This approach fully exploits the potential of existing specialized datasets, enabling the construction of valuable vision-language data even with a single data modality. Furthermore, by automatically associating large-scale unlabeled RS imagery with rich semantic information from open geoinformation databases, large-scale and semantically diverse datasets can also be built, as successfully demonstrated by the SkyScript [135] dataset.

It is worth noting that the core ideas behind the above methodologies are not limited to addressing data construction for SAR imagery. They possess high generalizability and transfer value and can be effectively applied to other specialized RS data domains. For instance, in modalities such as nighttime light [270] and thermal infrared [271] RS imagery, which also face annotation difficulties, these approaches can provide important references for constructing high-quality vision-language datasets.

## D. UAV Imagery

Embodied intelligence has emerged as a prominent frontier in AI research [12]. UAVs, often referred to as aerial robots, situate themselves at the intersection of embodied intelligence and RS technologies. Existing vision-language datasets for UAVs are expanding beyond foundational capabilities, such as navigation [189], [272], to progressively encompass diverse competency dimensions, including 3D spatial perception [215], [273], mathematical reasoning [155], complex decision-making [158], multi-agent collaboration [274], and security assurance [275]. As a critical enabler for the development of aerial embodied intelligence, these UAV vision-language datasets are constructing a comprehensive capability framework that spans perception, cognition, decision-making, collaboration, and safety, thereby propelling the field from basic environmental awareness towards deep human-UAV interaction and sophisticated decision-making.

Through the efficient integration of simulated and real-world data [272], [276], coupled with deep learning from multimodal and multi-agent technologies [277], UAVs hold significant potential to evolve from passive aerial cameras that merely execute tasks into proactive, embodied agents capable of autonomous environmental perception, intent understanding, interactive collaboration, and independent decision-making.

## E. Agentic AI in RS

One of the most remarkable advancements in the current AI domain is the rapid development of AI agents. These agents not only represent a key direction of technological evolution, but are also regarded by industries as a critical force for fostering new productive forces. However, a significant bottleneck that hinders the progress of AI agents from RS lies in the acquisition and construction of high-quality task-specific datasets. This process faces three core challenges. First, the field involves extremely high expertise barriers. RS interpretation relies on profound knowledge of Earth sciences. Tasks such as land surface temperature retrieval or nighttime light analysis [270] involve complex physical models and specialized interpretation rules. Consequently, data annotation must be performed by domain experts, leading to high costs and prolonged cycles. Second, task logic is inherently complex. Practical RS workflows, like disaster assessment or environmental monitoring, often require multi-step quantitative reasoning and the coordinated invocation of various specialized tools, for example, spectral index calculation [267] or change detection models [53]. This imposes stringent requirements on the design and validation of task sequences within datasets. Third, data and task formats are highly heterogeneous. The diversity of RS data sources and the dispersion of task objectives make standardized, large-scale annotation exceptionally difficult.

To address these challenges, current research primarily depends on manual construction paradigms.

For instance, datasets like UnivEarth [178] and GeoPlan-Bench [165] employ a reverse engineering approach, deriving questions from existing scientific reports or predefined tool pathways to ensure answer accuracy and verifiability. The ThinkGeo [177] dataset comprises expert-designed and validated reasoning chains, encompassing thousands of steps to ensure logical rigor. Earth-Bench further encapsulates domain knowledge into hundreds of standardized tools, constructing problems around tool combinations. These methods collectively form the mainstream benchmarking frameworks, providing valuable high-quality standards for evaluating the fundamental capabilities of AI agents. However, their fundamental limitation is the deep reliance on expert-led design, annotation, and validation. This results in constrained dataset scales, typically limited to a few hundred or thousands of samples, poor scalability, difficulty in covering real-world long-tail scenarios, and a nature more suited for testing rather than training sets.

Therefore, breaking through the data scale bottleneck is crucial for advancing this field. Establishing sustainable, low-cost, and high-quality data augmentation mechanisms represents an urgent and promising research direction to unlock the practical application potential of RS AI agents.

## F. RS Application Scenarios

Building high-quality, domain-specific datasets is pivotal for advancing RS technology from mere perception to deeper cognition and informed decisionmaking. Their core value lies in bridging the gap between general-purpose MLLMs and the specialized demands of the field. By infusing models with expert domain knowledge, these datasets enable an understanding of the unique perspectives, professional geo-objects, and technical terminology characteristic of RS imagery. This capability is fundamental for supporting complex analytical and decision-making tasks, such as disaster assessment [164] and urban planning [162]. Furthermore, such datasets form the cornerstone of natural humanmachine interaction, allowing experts to efficiently query data using intuitive instructions like analyzing road blockages after an earthquake, thereby significantly boosting operational efficiency [88].

However, the construction process is fraught with challenges. The primary hurdle is achieving annotation expertise and consistency. Interpreting RS images requires profound domain knowledge, making precise annotation inherently costly. To ensure quality, a human-machine collaborative strategy is often essential, typically involving AI-powered initial screening followed by verification by domain experts, supplemented by rigorous annotator training, cross-validation, and expert sampling audits [220]. Another critical challenge is balancing task design, scientificity, and practicality. Datasets must not only pursue high-precision metrics but also effectively address real-world industry pain points. This demands a deep understanding of business logic from builders, who must design diverse problems, correlate multi-source economic data, and even simulate real disaster scenarios to ensure the data underpins effective decision-making. Ultimately, striking a balance between data scale, quality, and cost is crucial. A feasible path currently involves leveraging AI-assisted generation with expert oversight to strike a balance between efficiency and quality.

## IX. CONCLUSION

This paper concludes by presenting a comprehensive, data-centric guide for tailoring Multimodal Large Language Models (MLLMs) in RS, centered on the novel GeoChef framework that systematizes the entire pipeline from dataset curation to model evaluation. Through a systematic survey of over 120 vision-language datasets, a detailed taxonomy, and a community-driven data warehouse, we provide practical methodologies for constructing, fine-tuning, and assessing MLLMs, empowering researchers to overcome resource constraints and achieve specialized model performance. We anticipate that this work can provide practical guidance for academia and industry, promoting the development of more diverse and higher-quality vision-language datasets for specialized RS fields. Ultimately, our long-term vision is to enable the organic integration of these scattered data resources, gradually forming a comprehensive, geographically widespread, and annotation-standardized super RS vision-language corpus. This will not only serve as the foundation for efficient and reproducible research and development but also lay a solid and indispensable data cornerstone for advancing toward a GeoAI with robust generalization and cognitive capabilities.

## REFERENCES

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[2] M. Hafner, M. Katsantoni, T. Koster, J. Marks, J. Mukher-¨ jee, D. Staiger, J. Ule, and M. Zavolan, “Clip and complementary methods,” Nature Reviews Methods Primers, vol. 1, no. 1, p. 20, 2021.

[3] Z. Sun, Y. Fang, T. Wu, P. Zhang, Y. Zang, S. Kong, Y. Xiong, D. Lin, and J. Wang, “Alpha-clip: A clip model focusing on wherever you want,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 13 019–13 029.

[4] M. Moor, O. Banerjee, Z. S. H. Abad, H. M. Krumholz, J. Leskovec, E. J. Topol, and P. Rajpurkar, “Foundation models for generalist medical artificial intelligence,” Nature, vol. 616, no. 7956, pp. 259–265, 2023.

[5] Y. Qin, S. Hu, Y. Lin, W. Chen, N. Ding, G. Cui, Z. Zeng, X. Zhou, Y. Huang, C. Xiao et al., “Tool learning with foundation models,” ACM Computing Surveys, vol. 57, no. 4, pp. 1–40, 2024.

[6] M. Awais, M. Naseer, S. Khan, R. M. Anwer, H. Cholakkal, M. Shah, M.-H. Yang, and F. S. Khan, “Foundation models defining a new era in vision: a survey and outlook,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[7] X. Li, C. Wen, Y. Hu, Z. Yuan, and X. X. Zhu, “Visionlanguage models in remote sensing: Current progress and future trends,” IEEE Geoscience and Remote Sensing Magazine, vol. 12, no. 2, pp. 32–66, 2024.

[8] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE transactions on pattern analysis and machine intelligence, vol. 46, no. 8, pp. 5625–5644, 2024.

[9] Z. Zhang, J. Wang, F. Wen, Y. Guo, X. Zhao, X. Fang, S. Ding, Z. Jia, J. Xiao, Y. Shen et al., “Large multimodal models evaluation: a survey,” Science China Information Sciences, vol. 68, no. 12, p. 221301, 2025.

[10] Y. Zeng, X. Zhang, H. Li, J. Wang, J. Zhang, and W. Zhou, “X2-vlm: All-in-one pre-trained model for vision-language tasks,” IEEE transactions on pattern analysis and machine intelligence, vol. 46, no. 5, pp. 3156–3168, 2023.

[11] Y. Zhao, E. Zhong, C. Yuan, Y. Li, M. Zhao, C. Li, J. Hu, W. Liu, and C. Liu, “Med-vlm: Enhancing medical image segmentation accuracy through vision-language model,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2025, pp. 7283–7293.

[12] X. Hao, L. Zhou, Z. Huang, Z. Hou, Y. Tang, L. Zhang, G. Li, Z. Lu, S. Ren, X. Meng, Y. Zhang, J. Wu, J. Lu, C. Dang, J. Guan, J. Wu, Z. Hou, H. Li, S. Xia, M. Zhou, Y. Zheng, Z. Yue, S. Gu, H. Tian, Y. Shen, J. Cui, W. Zhang, S. Xu, B. Wang, H. Sun, Z. Zhu, Y. Jiang, Z. Guo, C. Gong, C. Zhang, W. Ding, K. Ma, G. Chen, R. Cai, D. Xiang, H. Qu, F. Luo, H. Ye, and L. Chen, “Mimo-embodied: X-embodied foundation model technical report,” 2025.

[13] X. Weng, C. Pang, and G.-S. Xia, “Vision-language modeling meets remote sensing: Models, datasets, and perspectives,” IEEE Geoscience and Remote Sensing Magazine, 2025.

[14] X. Xue and X. X. Zhu, “Regression in earth observation: Are vision–language models up to the challenge?” IEEE Geoscience and Remote Sensing Magazine, 2025.

[15] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[16] J. Chen, D. Zhu, X. Shen, X. Li, Z. Liu, P. Zhang, R. Krishnamoorthi, V. Chandra, Y. Xiong, and M. Elhoseiny, “Minigpt-v2: large language model as a unified interface for vision-language multi-task learning,” arXiv preprint arXiv:2310.09478, 2023.

[17] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, “Qwen-vl: A frontier large visionlanguage model with versatile abilities,” arXiv preprint arXiv:2308.12966, 2023.

[18] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 24 185–24 198.

[19] M. Y. Lu, B. Chen, D. F. Williamson, R. J. Chen, I. Liang, T. Ding, G. Jaume, I. Odintsov, L. P. Le, G. Gerber et al., “A visual-language foundation model for computational pathology,” Nature medicine, vol. 30, no. 3, pp. 863–874, 2024.

[20] H. Chen, H. Chen, Z. Zhao, K. Han, G. Zhu, Y. Zhao, Y. Du, W. Xu, and Q. Shi, “An overview of domainspecific foundation model: key technologies, applications and challenges,” Science China Information Sciences, vol. 69, no. 1, p. 111301, 2026.

[21] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion-5b: An open large-scale dataset for training next generation image-text models,” Advances in neural information processing systems, vol. 35, pp. 25 278–25 294, 2022.

[22] Z. Luo, P. Zhao, C. Xu, X. Geng, T. Shen, C. Tao, J. Ma, Q. Lin, and D. Jiang, “Lexlip: Lexicon-bottlenecked language-image pre-training for large-scale image-text sparse retrieval,” in Proceedings ofthe IEEE/CVF international conference on computer vision, 2023, pp. 11 206– 11 217.

[23] Q. Yu, Q. Sun, X. Zhang, Y. Cui, F. Zhang, Y. Cao, X. Wang, and J. Liu, “Capsfusion: Rethinking imagetext data at scale,” in Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 022–14 032.

[24] F. Liu, D. Chen, Z. Guan, X. Zhou, J. Zhu, Q. Ye, L. Fu, and J. Zhou, “Remoteclip: A vision language foundation model for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–16, 2024.

[25] C. Liu, K. Chen, R. Zhao, Z. Zou, and Z. Shi, “Text2earth: Unlocking text-driven remote sensing image generation with a global-scale dataset and a foundation model,” IEEE Geoscience and Remote Sensing Magazine, 2025.

[26] H. Wang, Y. Nie, Y. Ye, Y. Wang, S. Li, H. Yu, J. Lu, and C. Huang, “Dynamic-vlm: Simple dynamic visual token compression for videollm,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 20 812–20 823.

[27] D. Zhang, J. Lei, J. Li, X. Wang, Y. Liu, Z. Yang, J. Li, W. Wang, S. Yang, J. Wu et al., “Critic-v: Vlm critics help catch vlm errors in multimodal reasoning,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 9050–9061.

[28] V. Prabhu, S. Purushwalkam, A. Yan, C. Xiong, and R. Xu, “Trust but verify: Programmatic vlm evaluation in the wild,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 3258–3267.

[29] J. Wu, M. Zhong, S. Xing, Z. Lai, Z. Liu, Z. Chen, W. Wang, X. Zhu, L. Lu, T. Lu et al., “Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks,” Advances in Neural Information Processing Systems, vol. 37, pp. 69 925–69 975, 2024.

[30] T.-D. Truong, H.-Q. Nguyen, X.-B. Nguyen, A. Dowling, X. Li, and K. Luu, “Insect-foundation: A foundation model and large multimodal dataset for vision-language insect understanding,” International Journal of Computer Vision, vol. 133, no. 10, pp. 7128–7153, 2025.

[31] P. Xu, W. Shao, K. Zhang, P. Gao, S. Liu, M. Lei, F. Meng, S. Huang, Y. Qiao, and P. Luo, “Lvlm-ehub: A comprehensive evaluation benchmark for large visionlanguage models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[32] S. Lobry, D. Marcos, J. Murray, and D. Tuia, “Rsvqa: Visual question answering for remote sensing data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 12, pp. 8555–8566, 2020.

[33] K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. S. Khan, “Geochat: Grounded large visionlanguage model for remote sensing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 27 831–27 840.

[34] Z. Wei, G. Wang, X. Ma, K. Mei, H. Chen, Y. Jin, and F. Rao, “Hq-clip: Leveraging large vision-language models to create high-quality image-text datasets and clip models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 22 447–22 456.

[35] J. Chen, Q. Yu, X. Shen, A. Yuille, and L.-C. Chen, “Vitamin: Designing scalable vision models in the visionlanguage era,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 12 954–12 966.

[36] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, “Instructblip: Towards general-purpose vision-language models with instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 49 250–49 267, 2023.

[37] U. R. Pol, P. S. Vadar, and T. T. Moharekar, “Hugging face: revolutionizing ai and nlp,” International Journalfor Research in Applied Science and Engineering Technology, vol. 12, no. 8, pp. 1121–1124, 2024.

[38] G.-S. Xia, X. Bai, J. Ding, Z. Zhu, S. Belongie, J. Luo, M. Datcu, M. Pelillo, and L. Zhang, “Dota: A largescale dataset for object detection in aerial images,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3974–3983.

[39] K. Li, G. Wan, G. Cheng, L. Meng, and J. Han, “Object detection in optical remote sensing images: A survey and a new benchmark,” ISPRSjournal ofphotogrammetry and remote sensing, vol. 159, pp. 296–307, 2020.

[40] X. Sun, P. Wang, Z. Yan, F. Xu, R. Wang, W. Diao, J. Chen, J. Li, Y. Feng, T. Xu et al., “Fair1m: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 184, pp. 116–130, 2022.

[41] X. Yang, G. Zhang, X. Yang, Y. Zhou, W. Wang, J. Tang, T. He, and J. Yan, “Detecting rotated objects as gaussian distributions and its 3-d generalization,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 4, pp. 4335–4354, 2022.

[42] Y. Zhou, X. Yang, G. Zhang, J. Wang, Y. Liu, L. Hou, X. Jiang, X. Liu, J. Yan, C. Lyu et al., “Mmrotate: A rotated object detection benchmark using pytorch,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 7331–7334.

[43] X. Yang, J. Yan, Z. Feng, and T. He, “R3det: Refined single-stage detector with feature refinement for rotating object,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 4, 2021, pp. 3163–3171.

[44] T. Su and S. Zhang, “Local and global evaluation for remote sensing image segmentation,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 130, pp. 256– 276, 2017.

[45] M. Volpi and D. Tuia, “Deep multi-task learning for a geographically-regularized semantic segmentation of aerial images,” ISPRS journal of photogrammetry and remote sensing, vol. 144, pp. 48–60, 2018.

[46] D. Wang, J. Zhang, B. Du, M. Xu, L. Liu, D. Tao, and L. Zhang, “Samrs: Scaling-up remote sensing segmentation dataset with segment anything model,” Advances in Neural Information Processing Systems, vol. 36, pp. 8815–8827, 2023.

[47] Q. He, X. Sun, W. Diao, Z. Yan, F. Yao, and K. Fu, “Multimodal remote sensing image segmentation with intuition-inspired hypergraph modeling,” IEEE Transactions on Image Processing, vol. 32, pp. 1474–1487, 2023.

[48] J. Li, Y. Cai, Q. Li, M. Kou, and T. Zhang, “A review of remote sensing image segmentation by deep learning methods,” International Journal of Digital Earth, vol. 17, no. 1, p. 2328827, 2024.

[49] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction

tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892–34 916, 2023.

[50] S. Zhang, L. Dong, X. Li, S. Zhang, X. Sun, S. Wang, J. Li, R. Hu, T. Zhang, G. Wang et al., “Instruction tuning for large language models: A survey,” ACM Computing Surveys, 2023.

[51] Z. Shi, A. X. Yang, B. Wu, L. Aitchison, E. Yilmaz, and A. Lipani, “Instruction tuning with loss over instructions,” Advances in Neural Information Processing Systems, vol. 37, pp. 69 176–69 205, 2024.

[52] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26 296–26 306.

[53] P. Deng, W. Zhou, and H. Wu, “Changechat: An interactive model for remote sensing change analysis via multimodal instruction tuning,” in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[54] T. Gupta and A. Kembhavi, “Visual programming: Compositional visual reasoning without training,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 14 953–14 962.

[55] M. Małkinski and J. Ma´ ndziuk, “A review of emerging´ research directions in abstract visual reasoning,” Information Fusion, vol. 91, pp. 713–736, 2023.

[56] L. Chen, B. Li, S. Shen, J. Yang, C. Li, K. Keutzer, T. Darrell, and Z. Liu, “Large language models are visual reasoning coordinators,” Advances in Neural Information Processing Systems, vol. 36, pp. 70 115–70 140, 2023.

[57] Z. Chen, Q. Zhou, Y. Shen, Y. Hong, Z. Sun, D. Gutfreund, and C. Gan, “Visual chain-of-thought prompting for knowledge-based visual reasoning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 2, 2024, pp. 1254–1262.

[58] B. Huang, X. Wang, H. Chen, Z. Song, and W. Zhu, “Vtimellm: Empower llm to grasp video moments,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 271–14 280.

[59] M. Nie, D. Ding, C. Wang, Y. Guo, J. Han, H. Xu, and L. Zhang, “Slowfocus: Enhancing fine-grained temporal understanding in video llm,” Advances in Neural Information Processing Systems, vol. 37, pp. 81 808–81 835, 2024.

[60] F. Shu, L. Zhang, H. Jiang, and C. Xie, “Audio-visual llm for video understanding,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 4246–4255.

[61] Y. Yuan, H. Zhang, W. Li, Z. Cheng, B. Zhang, L. Li, X. Li, D. Zhao, W. Zhang, Y. Zhuang et al., “Videorefer suite: Advancing spatial-temporal object understanding with video llm,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 18 970– 18 980.

[62] Z. Wang, S. Yu, E. Stengel-Eskin, J. Yoon, F. Cheng, G. Bertasius, and M. Bansal, “Videotree: Adaptive treebased video representation for llm reasoning on long videos,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 3272–3283.

[63] S. Hou, H. Jiao, J. Liang, Z. Shen, A. Zhao, and H. Wu, “Geocogent: an llm-based agent for geospatial

code generation,” International Journal of Geographical Information Science, pp. 1–34, 2025.

[64] Y. Zhang, J. Li, Z. Wang, Z. He, Q. Guan, J. Lin, and W. Yu, “Geospatial large language model trained with a simulated environment for generating tool-use chains autonomously,” International Journal of Applied Earth Observation and Geoinformation, p. 104312, 2024.

[65] C. Yu, Y. Yao, X. Zhang, G. Zhu, Y. Guo, X. Shao, M. Shibasaki, Z. Hu, L. Dai, Q. Guan et al., “Monkuu: a llm-powered natural language interface for geospatial databases with dynamic schema mapping,” International Journal of Geographical Information Science, pp. 1–22, 2025.

[66] S. Singh, M. Fore, and D. Stamoulis, “Geollm-engine: A realistic environment for building geospatial copilots,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 585–594.

[67] H. Zhu, Z. Li, and J. Jin, “Ger-llm: Efficient and effective geospatial entity resolution with large language model,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 23 272– 23 288.

[68] S. Scheider and K.-F. Richter, “Geoai,” KI-Kunstliche¨ Intelligenz, vol. 37, no. 1, pp. 5–9, 2023.

[69] K. Janowicz, S. Gao, G. McKenzie, Y. Hu, and B. Bhaduri, “Geoai: spatially explicit artificial intelligence techniques for geographic knowledge discovery and beyond,” pp. 625–636, 2020.

[70] G. Mai, K. Janowicz, Y. Hu, S. Gao, B. Yan, R. Zhu, L. Cai, and N. Lao, “A review of location encoding for geoai: methods and applications,” International Journal of Geographical Information Science, vol. 36, no. 4, pp. 639–673, 2022.

[71] Y. Zhou, L. Feng, Y. Ke, X. Jiang, J. Yan, X. Yang, and W. Zhang, “Towards vision-language geo-foundation model: A survey,” arXiv preprint arXiv:2406.09385, 2024.

[72] H. Zhang, J.-J. Xu, H.-W. Cui, L. Li, Y. Yang, C.-S. Tang, and N. Boers, “When geoscience meets foundation models: Toward a general geoscience artificial intelligence system,” IEEE geoscience and remote sensing magazine, 2024.

[73] A. Xiao, W. Xuan, J. Wang, J. Huang, D. Tao, S. Lu, and N. Yokoya, “Foundation models for remote sensing and earth observation: A survey,” IEEE Geoscience and Remote Sensing Magazine, 2025.

[74] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” nature, vol. 521, no. 7553, pp. 436–444, 2015.

[75] N. Rusk, “Deep learning,” Nature Methods, vol. 13, no. 1, pp. 35–35, 2016.

[76] L. Zhang, L. Zhang, and B. Du, “Deep learning for remote sensing data: A technical tutorial on the state of the art,” IEEE Geoscience and remote sensing magazine, vol. 4, no. 2, pp. 22–40, 2016.

[77] Y. Li, H. Zhang, X. Xue, Y. Jiang, and Q. Shen, “Deep learning for remote sensing image classification: A survey,” Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, vol. 8, no. 6, p. e1264, 2018.

[78] J. Li, D. Hong, L. Gao, J. Yao, K. Zheng, B. Zhang, and J. Chanussot, “Deep learning in multimodal remote sensing data fusion: A comprehensive review,” International

Journal of Applied Earth Observation and Geoinformation, vol. 112, p. 102926, 2022.

[79] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, “Lisa: Reasoning segmentation via large language model,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 9579–9589.

[80] A. Zhang, L. Zhao, C.-W. Xie, Y. Zheng, W. Ji, and T.-S. Chua, “Next-chat: An lmm for chat, detection and segmentation,” arXiv preprint arXiv:2311.04498, 2023.

[81] W. Wang, Z. Chen, X. Chen, J. Wu, X. Zhu, G. Zeng, P. Luo, T. Lu, J. Zhou, Y. Qiao et al., “Visionllm: Large language model is also an open-ended decoder for visioncentric tasks,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[82] S. Uchida, “Using early llms for corpus linguistics: Examining chatgpt’s potential and limitations,” Applied Corpus Linguistics, vol. 4, no. 1, p. 100089, 2024.

[83] Y. Yu, Z. Dai, Z. Wang, W. Wang, R. Chen, and J. Pei, “Opencsg chinese corpus: A series of highquality chinese datasets for llm training,” arXiv preprint arXiv:2501.08197, 2025.

[84] U. K. Mall, C. P. Phoo, M. Liu, C. Vondrick, B. Hariharan, and K. Bala, “Remote sensing vision-language foundation models without annotations via ground remote alignment,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 49 294–49 314.

[85] Z. Xu, Y. Zhu, S. Deng, A. Mittal, Y. Chen, M. Wang, P. Favaro, J. Tighe, and D. Modolo, “Benchmarking zeroshot recognition with vision-language models: Challenges on granularity and specificity,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1827–1836.

[86] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.

[87] Z. Xiong, F. Zhang, Y. Wang, Y. Shi, and X. X. Zhu, “Earthnets: Empowering ai in earth observation,” arXiv preprint arXiv:2210.04936, 2022.

[88] J. Wang, W. Xuan, H. Qi, Z. Liu, K. Liu, Y. Wu, H. Chen, J. Song, J. Xia, Z. Zheng et al., “Disasterm3: A remote sensing vision-language dataset for disaster damage assessment and response,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[89] H. Wu, Z. Du, D. Zhong, Y. Wang, and C. Tao, “Fsvlm: A vision-language model for remote sensing farmland segmentation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–13, 2025.

[90] J. Wang, W. Xuan, H. Qi, Z. Chen, H. Chen, Z. Zheng, J. Xia, Y. Zhong, and N. Yokoya, “Cityvlm: Towards sustainable urban development via multi-view coordinated visionlanguage model,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 232, pp. 62–74, 2026.

[91] X. Lin, W. Wang, Y. Li, S. Yang, F. Feng, Y. Wei, and T.-S. Chua, “Data-efficient fine-tuning for llm-based recommendation,” in Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, 2024, pp. 365–374.

[92] H. Zheng, L. Shen, A. Tang, Y. Luo, H. Hu, B. Du, Y. Wen, and D. Tao, “Learning from models beyond fine-

tuning,” Nature Machine Intelligence, vol. 7, no. 1, pp. 6–17, 2025.

[93] D. Anisuzzaman, J. G. Malins, P. A. Friedman, and Z. I. Attia, “Fine-tuning llms for specialized use cases,” Mayo Clinic Proceedings: Digital Health, 2024.

[94] C. Tian, Z. Shi, Z. Guo, L. Li, and C.-Z. Xu, “Hydralora: An asymmetric lora architecture for efficient fine-tuning,” Advances in Neural Information Processing Systems, vol. 37, pp. 9565–9584, 2024.

[95] J. Bian, L. Wang, L. Zhang, and J. Xu, “Lora-fair: Federated lora fine-tuning with aggregation and initialization refinement,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 3737–3746.

[96] T. Lee, H. Tu, C. H. Wong, W. Zheng, Y. Zhou, Y. Mai, J. S. Roberts, M. Yasunaga, H. Yao, C. Xie, and P. Liang, “Vhelm: A holistic evaluation of vision language models,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 140 632–140 666.

[97] S. Shool, S. Adimi, R. Saboori Amleshi, E. Bitaraf, R. Golpira, and M. Tara, “A systematic review of large language model (llm) evaluations in clinical medicine,” BMC Medical Informatics and Decision Making, vol. 25, no. 1, p. 117, 2025.

[98] J. Hu, R. Liu, D. Hong, A. Camero, J. Yao, M. Schneider, F. Kurz, K. Segl, and X. X. Zhu, “Mdas: A new multimodal benchmark dataset for remote sensing,” Earth System Science Data, vol. 15, no. 1, pp. 113–131, 2023.

[99] D. Hong, B. Zhang, H. Li, Y. Li, J. Yao, C. Li, M. Werner, J. Chanussot, A. Zipf, and X. X. Zhu, “Cross-city matters: A multimodal remote sensing benchmark dataset for cross-city semantic segmentation using high-resolution domain adaptation networks,” Remote Sensing of Environment, vol. 299, p. 113856, 2023.

[100] M. Hossin and M. N. Sulaiman, “A review on evaluation metrics for data classification evaluations,” International journal ofdata mining & knowledge management process, vol. 5, no. 2, p. 1, 2015.

[101] E. K. Dey and M. Awrangjeb, “A robust performance evaluation metric for extracted building boundaries from remote sensing data,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 13, pp. 4030–4043, 2020.

[102] G. Naidu, T. Zuva, and E. M. Sibanda, “A review of evaluation metrics in machine learning algorithms,” in Computer science on-line conference. Springer, 2023, pp. 15–25.

[103] S. Soni, A. Dudhane, H. Debary, M. Fiaz, M. A. Munir, M. S. Danish, P. Fraccaro, C. D. Watson, L. J. Klein, F. S. Khan et al., “Earthdial: Turning multi-sensory earth observations to interactive dialogues,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 14 303–14 313.

[104] J. Ye, J. He, X. Zhang, Y. Lin, H. Lin, C. He, and W. Li, “Satellite image synthesis from street view with fine-grained spatial textual guidance: A novel framework,” IEEE Geoscience and Remote Sensing Magazine, vol. 13, no. 3, pp. 395–414, 2025.

[105] J. Ye, H. Lin, L. Ou, D. Chen, Z. Wang, Q. Zhu, C. He, and W. Li, “Where am i? cross-view geo-localization

with natural language descriptions,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 5890–5900.

[106] K. Yao, N. Xu, R. Yang, Y. Xu, Z. Gao, T. Kitrungrotsakul, Y. Ren, P. Zhang, J. Wang, N. Wei et al., “Falcon: A remote sensing vision-language foundation model,” arXiv preprint arXiv:2503.11070, 2025.

[107] Y. Yang, X. Zhang, Q. Fang, J. Liu, Z. Ye, R. Li, L. Liu, and H. Wang, “Fusar-klip: Towards multimodal foundation models for remote sensing,” 2025.

[108] A. Zavras, D. Michail, X. X. Zhu, B. Demir, and I. Papoutsis, “Gaia: A global, multimodal, multiscale vision– language dataset for remote sensing image analysis,” IEEE Geoscience and Remote Sensing Magazine, 2026.

[109] Z. Xiong, Y. Wang, W. Yu, A. J. Stewart, J. Zhao, N. Lehmann, T. Dujardin, Z. Yuan, P. Ghamisi, and X. X. Zhu, “Geolangbind: Unifying earth observation with agglomerative vision-language foundation models,” arXiv preprint arXiv:2503.06312, 2025.

[110] X. Lu, Z. Zheng, Y. Wan, Y. Yao, A. Wang, R. Zhang, P. Xia, Q. Wu, Q. Li, W. Lin, X. Zhao, P. Ma, X. Yang, and H. Li, “Gleam: Learning to match and explain in cross-view geo-localization,” 2025.

[111] M. Czerkawski and A. Francis, “From laion-5b to laion-eo: Filtering billions of images using anchor datasets for satellite image extraction,” arXiv preprint arXiv:2309.15535, 2023.

[112] D. Muhtar, Z. Li, F. Gu, X. Zhang, and P. Xiao, “Lhrsbot: Empowering remote sensing with vgi-enhanced large multimodal language model,” in European Conference on Computer Vision. Springer, 2024, pp. 440–457.

[113] Z. Li, D. Muhtar, F. Gu, Y. He, X. Zhang, P. Xiao, G. He, and X. Zhu, “Lhrs-bot-nova: Improved multimodal large language model for remote sensing vision-language interpretation,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 227, pp. 539–550, 2025.

[114] D. Muhtar, E. Zhang, Z. Li, F. Gu, Y. He, P. Xiao, and X. Zhang, “Quality-driven curation of remote sensing vision-language data via learned scoring models,” Advances in Neural Information Processing Systems, vol. 38, pp. 171 088–171 133, 2026.

[115] D. Si, D. Wang, E. Gao, X. Qin, L. Zhao, J. Zhang, M. Xu, J. Zhan, J. Wang, L. Liu et al., “Spex: a vision-language model for land cover extraction on spectral remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, 2026.

[116] C. T. Marimo, B. Blumenstiel, M. Nitsche, J. Jakubik, and T. Brunschwiler, “Beyond the visible: Multispectral vision-language learning for earth observation,” in Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 2025, pp. 359– 375.

[117] S. Ma, Z. Li, and J. A. Taylor, “Landsat30-au: A visionlanguage dataset for australian landsat imagery,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 10, 2026, pp. 7809–7817.

[118] W. Chen, J. Chen, Y. Deng, J. Chen, Y. Feng, Z. Xi, D. Liu, K. Li, and Y. Meng, “Lrsclip: A vision-language foundation model for aligning remote sensing image with longer text,” arXiv preprint arXiv:2503.19311, 2025.

[119] W. Zhang, M. Cai, T. Zhang, Y. Zhuang, and X. Mao, “Earthgpt: A universal multimodal large language model for multisensor image comprehension in remote sensing domain,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–20, 2024.

[120] J. Luo, Y. Wang, Z. Gu, Y. Qiu, S. Yao, F. Wang, C. Xu, W. Zhang, D. Wang, and Z. Cui, “Mmm-rs: a multimodal, multi-gsd, multi-scene remote sensing dataset and benchmark for text-to-image generation,” in Proceedings of the 38th International Conference on Neural Information Processing Systems, 2024, pp. 12 151–12 163.

[121] L. Pan, C. Song, X. Gan, K. Xu, and Y. Xie, “Military image captioning for low-altitude uav or ugv perspectives,” Drones, vol. 8, no. 9, 2024.

[122] P. Wang, Z. Lu, Y. Li, B. Ding, and D. Zhang, “Sarclip: The first visionlanguage foundation model for sar image,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–11, 2025.

[123] W. Zhang, M. Cai, Y. Ning, T. Zhang, Y. Zhuang, H. Chen, J. Li, and X. Mao, “Earthgpt-x: Enabling mllms to flexibly and comprehensively understand multi-source remote sensing imagery,” arXiv preprint arXiv:2504.12795, 2025.

[124] Q. Cheng, H. Huang, Y. Xu, Y. Zhou, H. Li, and Z. Wang, “Nwpu-captions dataset and mlca-net for remote sensing image captioning,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–19, 2022.

[125] X. Li, Y. Tao, S. Zhang, S. Liu, Z. Xiong, C. Luo, L. Liu, M. Pechenizkiy, X. Zhu, and T. Huang, “Reobench: Benchmarking robustness of earth observation foundation models,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[126] Z. Zhang, T. Zhao, Y. Guo, and J. Yin, “Rs5m and georsclip: A large-scale vision- language dataset and a large vision-language model for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–23, 2024.

[127] X. Lu, B. Wang, X. Zheng, and X. Li, “Exploring models and data for remote sensing image caption generation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 4, pp. 2183–2195, 2018.

[128] Z. Yuan, W. Zhang, K. Fu, X. Li, C. Deng, H. Wang, and X. Sun, “Exploring a fine-grained multiscale method for cross-modal remote sensing image retrieval,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–19, 2022.

[129] H. Hu, P. Wang, Y. Feng, K. Wei, W. Yin, W. Diao, M. Wang, H. Bi, K. Kang, T. Ling et al., “Ringmo-agent: A unified remote sensing foundation model for multiplatform and multi-modal reasoning,” arXiv preprint arXiv:2507.20776, 2025.

[130] C. Jiang, C. Wang, F. Wu, P. Ma, L. Zou, T. Li, J. Ning, and Y. Tang, “Sarclip: a multimodal foundation framework for sar imagery via contrastive language-image pretraining,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 231, pp. 17–34, 2026.

[131] Z. Ma, X. Xiao, S. Dong, P. Wang, H. Wang, and Q. Pan, “Sarchat-bench-2m: A multi-task vision-language benchmark for sar image interpretation,” arXiv preprint arXiv:2502.08168, 2025.

[132] Y. Wei, A. Xiao, Y. Ren, Y. Zhu, H. Chen, J. Xia, and N. Yokoya, “Sarlang-1m: A benchmark for visionlanguage modeling in sar image understanding,” IEEE Transactions on Geoscience and Remote Sensing, 2026.

[133] Y. He, X. Cheng, J. Zhu, C. Qiu, J. Wang, X. Zhang, Q. Huang, and K. Yang, “Sar-text: A large-scale sar image-text dataset built with sar-narrator and a progressive learning strategy for downstream tasks,” 2025.

[134] Q. Ma, Z. Wang, W. Liu, X. Lu, B. Deng, P. Duan, X. Kang, and S. Li, “Sarvlm: A vision language foundation model for semantic understanding and target recognition in sar imagery,” 2025.

[135] Z. Wang, R. Prabha, T. Huang, J. Wu, and R. Rajagopal, “Skyscript: A large and semantically diverse visionlanguage dataset for remote sensing,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 6, 2024, pp. 5805–5813.

[136] B. Qu, X. Li, D. Tao, and X. Lu, “Deep semantic understanding of high resolution remote sensing image,” in 2016 International Conference on Computer, Information and Telecommunication Systems (CITS), 2016, pp. 1–5.

[137] G. Hoxha and F. Melgani, “A novel svm-based decoder for remote sensing image captioning,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2022.

[138] H. Li, X. Zhang, and H. Qu, “Ddfav: Remote sensing large vision language models dataset and evaluation benchmark,” Remote Sensing, vol. 17, no. 4, p. 719, 2025.

[139] J. Luo, Z. Pang, Y. Zhang, T. Wang, L. Wang, B. Dang, J. Lao, J. Wang, J. Chen, Y. Tan et al., “Skysensegpt: A fine-grained instruction tuning dataset and model for remote sensing vision-language understanding,” arXiv preprint arXiv:2406.10100, 2024.

[140] M. Danish, M. A. Munir, S. R. A. Shah, K. Kuckreja, F. S. Khan, P. Fraccaro, A. Lacoste, and S. Khan, “Geobenchvlm: Benchmarking vision-language models for geospatial tasks,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 7132–7142.

[141] R. Ou, Y. Hu, F. Zhang, J. Chen, and Y. Liu, “Geopix: A multimodal large language model for pixel-level image understanding in remote sensing,” IEEE Geoscience and Remote Sensing Magazine, 2025.

[142] Y. Zhao, M. Zhang, B. Yang, Z. Zhang, J. Kang, and J. Gong, “Luojiahog: A hierarchy oriented geo-aware image caption dataset for remote sensing image–text retrieval,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 222, pp. 130–151, 2025.

[143] L. Xu, L. Zhao, W. Guo, Q. Li, K. Long, K. Zou, Y. Wang, and H. Li, “Rs-gpt4v: A unified multimodal instruction-following dataset for remote sensing image understanding,” arXiv preprint arXiv:2406.12479, 2024.

[144] Z. Chen, C. Wang, N. Zhang, and F. Zhang, “Rscc: A large-scale remote sensing change caption dataset for disaster events,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[145] Y. Hu, J. Yuan, C. Wen, X. Lu, Y. Liu, and X. Li, “Rsgpt: A remote sensing vision language model and benchmark,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 224, pp. 272–286, 2025.

[146] J. Ge, X. Zhang, Y. Zheng, K. Guo, and J. Liang, “Rsteller: Scaling up visual language modeling in remote

sensing with rich linguistic semantics from openly available data and large language models,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 226, pp. 146– 163, 2025.

[147] W. Zhang, M. Cai, T. Zhang, Y. Zhuang, J. Li, and X. Mao, “Earthmarker: A visual prompting multimodal large language model for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–19, 2025.

[148] C. Pang, X. Weng, J. Wu, J. Li, Y. Liu, J. Sun, W. Li, S. Wang, L. Feng, G.-S. Xia et al., “Vhm: Versatile and honest vision language model for remote sensing image analysis,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 6, 2025, pp. 6381– 6388.

[149] X. Li, J. Ding, and M. Elhoseiny, “Vrsbench: A versatile vision-language benchmark dataset for remote sensing image understanding,” Advances in Neural Information Processing Systems, vol. 37, pp. 3229–3242, 2024.

[150] F. Wang, H. Wang, Z. Guo, D. Wang, Y. Wang, M. Chen, Q. Ma, L. Lan, W. Yang, J. Zhang et al., “Xlrs-bench: Could your multimodal llms understand extremely large ultra-high-resolution remote sensing imagery?” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 14 325–14 336.

[151] L. Bashmal, Y. Bazi, M. M. Al Rahhal, M. Zuair, and F. Melgani, “Capera: Captioning events in aerial videos,” Remote Sensing, vol. 15, no. 8, p. 2139, 2023.

[152] H. Elgendy, A. Sharshar, A. Aboeitta, Y. Ashraf, and M. Guizani, “Geollava: Efficient fine-tuned visionlanguage models for temporal change detection in remote sensing,” arXiv preprint arXiv:2410.19552, 2024.

[153] C. Liu, R. Zhao, H. Chen, Z. Zou, and Z. Shi, “Remote sensing image change captioning with dual-branch transformers: A new method and a large scale dataset,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–20, 2022.

[154] A. Shabbir, M. Zumri, M. Bennamoun, F. S. Khan, and S. Khan, “Geopixel: Pixel grounding large multimodal model in remote sensing,” in International Conference on Machine Learning. PMLR, 2025, pp. 54 095–54 111.

[155] Y. Zhou, L. Feng, M. Lan, X. Yang, Q. Li, Y. Ke, X. Jiang, and W. Zhang, “Multimodal mathematical reasoning embedded in aerial vehicle imagery: Benchmarking, analysis, and exploration,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 230, pp. 289–303, 2025.

[156] Q. Li, Y. Zhang, Z. Mai, Y. Chen, H. Huang, J. Zhang, Z. Zhang, Y. Wen, W. Li, H. Fu et al., “Can large multimodal models understand agricultural scenes? benchmarking with agromind,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[157] M. Mo, X. Tong, M. Tan, J. Leng, J. Zheng, Y. Liu, H. Chen, J. Gan, W. Li, and X. Gao, “A2seek: Towards reasoning-centric benchmark for aerial anomaly understanding,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[158] M. Guo, M. Wu, J. He, S. Li, H. Li, and C. Tao, “Bedi: A comprehensive benchmark for evaluating embodied agents on uavs,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 232, pp. 910–936, 2026.

[159] X. An, J. Sun, Z. Gui, and W. He, “Coreval: A comprehensive and objective benchmark for evaluating the remote sensing capabilities of large vision-language models,” arXiv preprint arXiv:2411.18145, 2024.

[160] M. Zhang, F. Chen, and B. Li, “Multistep questiondriven visual question answering for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–12, 2023.

[161] P. Feng, Z. Lv, J. Ye, X. Wang, X. Huo, J. Yu, W. Xu, W. Zhang, L. Bai, C. He, and W. Li, “Earth-agent: Unlocking the full landscape of earth observation with agents,” 2025.

[162] J. Wang, Z. Zheng, Z. Chen, A. Ma, and Y. Zhong, “Earthvqa: Towards queryable earth via relational reasoningbased remote sensing visual question answering,” in Thirtieth AAAI Conference on Artificial Intelligence, 2024.

[163] R. Shao, Z. Li, Z. Zhang, L. Xu, X. He, H. Yuan, B. He, Y. Dai, Y. Yan, Y. Chen, W. Guo, and H. Li, “Asking like socrates: Socrates helps vlms understand remote sensing images,” 2025.

[164] M. Rahnemoonfar, T. Chowdhury, A. Sarkar, D. Varshney, M. Yari, and R. R. Murphy, “Floodnet: A high resolution aerial imagery dataset for post flood scene understanding,” IEEE Access, vol. 9, pp. 89 644–89 654, 2021.

[165] K. Li, J. Wang, Z. Wang, H. Qiao, W. Zhang, D. Meng, and X. Cao, “Designing domain-specific agents via hierarchical task abstraction mechanism,” 2025.

[166] M. Tsujimoto, J. Wang, W. Xuan, and N. Yokoya, “Geo3dvqa: Evaluating vision-language models for 3d geospatial reasoning from aerial imagery,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2026, pp. 4994–5003.

[167] K. Li, G. Vosselman, and M. Y. Yang, “Hrvqa: A visual question answering benchmark for high-resolution aerial images,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 214, pp. 65–81, 2024.

[168] Z. Chen, H. Wang, J. Yao, P. Ghamisi, J. Zhou, P. M. Atkinson, and B. Zhang, “Cangling-knowflow: A unified knowledge-and-flow-fused agent for comprehensive remote sensing applications,” 2025.

[169] Z. Xin, K. Li, L. Chen, W. Li, Y. Xiao, H. Qiao, W. Zhang, D. Meng, and X. Cao, “Segearth-r2: Towards comprehensive language-guided segmentation for remote sensing images,” 2025.

[170] R. Liu, B. Fu, J. Song, K. Li, W. Li, L. Xue, H. Qiao, W. Zhang, D. Meng, and X. Cao, “Zoomearth: Active perception for ultra-high-resolution geospatial visionlanguage tasks,” 2025.

[171] J. Luo, Y. Zhang, X. Yang, K. Wu, Q. Zhu, L. Liang, J. Chen, and Y. Li, “When large vision-language model meets large remote sensing imagery: Coarse-to-fine textguided token pruning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 9206–9217.

[172] Y. Zhang, H. Zhang, H. Tian, C. Fu, S. Zhang, J. Wu, F. Li, K. Wang, Q. Wen, Z. Zhang et al., “Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans?” in International Conference on Learning Representations, vol. 2025, 2025, pp. 89 655–89 701.

[173] S. Dai, Z. Ma, Z. Luo, X. Yang, Y. Huang, W. Zhang, C. Chen, Z. Guo, W. Xu, Y. Sun, and M. Sun, “Mmuavbench: How well do multimodal large language models see, think, and plan in low-altitude uav scenarios?” 2025.

[174] Z. Zhao, C. Zhou, Y. Zhang, C. Li, X. Ma, and J. Tang, “Text-guided coarse-to-fine fusion network for robust remote sensing visual question answering,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 230, pp. 1– 17, 2025.

[175] X. Zheng, B. Wang, X. Du, and X. Lu, “Mutual attention inception network for remote sensing visual question answering,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2021.

[176] S. Lobry, B. Demir, and D. Tuia, “Rsvqa meets bigearthnet: A new, large-scale, visual question answering dataset for remote sensing,” in 2021 IEEE International Geoscience and Remote Sensing Symposium (IGARSS). IEEE, 2021, pp. 1218–1221.

[177] A. Shabbir, M. A. Munir, A. Dudhane, M. U. Sheikh, M. H. Khan, P. Fraccaro, J. B. Moreno, F. S. Khan, and S. Khan, “Thinkgeo: Evaluating tool-augmented agents for remote sensing tasks,” 2025.

[178] C. H. Kao, W. Zhao, S. Revankar, S. Speas, S. Bhagat, R. Datta, C. P. Phoo, U. Mall, C. Vondrick, K. Bala, and B. Hariharan, “Towards llm agents for earth observation,” 2025.

[179] B. Zhao, J. Fang, Z. Dai, Z. Wang, J. Zha, W. Zhang, C. Gao, Y. Wang, J. Cui, X. Chen, and Y. Li, “Urbanvideobench: Benchmarking vision-language models on embodied intelligence with video data in urban spaces,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2025, p. 3240032423.

[180] C. Zhang and S. Wang, “Good at captioning bad at counting: Benchmarking gpt-4v on earth observation data,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7839–7849.

[181] C. Wang, Z. Luo, R. Liu, C. Ran, S. Fan, X. Chen, and C. He, “Vicot-agent: A vision-interleaved chain-ofthought framework for interpretable multimodal reasoning and scalable remote sensing analysis,” 2025.

[182] M. M. Al Rahhal, Y. Bazi, S. O. Alsaleh, M. Al-Razgan, M. L. Mekhalfi, M. Al Zuair, and N. Alajlan, “Openended remote sensing visual question answering with transformers,” International Journal of Remote Sensing, vol. 43, no. 18, pp. 6809–6823, 2022.

[183] J. Irvin, E. Liu, J. Chen, I. Dormoy, J. Kim, S. Khanna, Z. Zheng, and S. Ermon, “Teochat: A large visionlanguage assistant for temporal earth observation data,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 68 883–68 911.

[184] B. Zhou, H. Yang, D. Chen, J. Ye, T. Bai, J. Yu, S. Zhang, D. Lin, C. He, and W. Li, “Urbench: A comprehensive benchmark for evaluating large multimodal models in multi-view urban scenarios,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 10, 2025, pp. 10 707–10 715.

[185] K. Li, F. Dong, D. Wang, S. Li, Q. Wang, X. Gao, and T.- S. Chua, “Show me what and where has changed? ques-

tion answering and grounding for remote sensing change detection,” arXiv preprint arXiv:2410.23828, 2024.

[186] Z. Yuan, L. Mou, Z. Xiong, and X. X. Zhu, “Change detection meets visual question answering,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–13, 2022.

[187] Y. Zhou, M. Lan, X. Li, Y. Ke, X. Jiang, L. Feng, and W. Zhang, “Geoground: A unified large visionlanguage model. for remote sensing visual grounding,” arXiv preprint arXiv:2411.11904, 2024.

[188] Y. Zhan, Z. Xiong, and Y. Yuan, “Rsvg: Exploring data and models for visual grounding on remote sensing data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–13, 2023.

[189] M. Chu, Z. Zheng, W. Ji, T. Wang, and T.-S. Chua, “Towards natural language-guided drones: Geotext-1652 benchmark with spatial relation matching,” in European Conference on Computer Vision. Springer, 2024, pp. 213–231.

[190] K. Li, D. Wang, H. Xu, H. Zhong, and C. Wang, “Language-guided progressive attention for visual grounding in remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–13, 2024.

[191] Y. Sun, S. Feng, X. Li, Y. Ye, J. Kang, and X. Huang, “Visual grounding in remote sensing images,” in Proceedings ofthe 30th ACM International Conference on Multimedia, 2022, pp. 404–412.

[192] M. Lan, F. Rong, H. Jiao, Z. Gao, and L. Zhang, “Language query-based transformer with multiscale crossmodal alignment for visual grounding on remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–13, 2024.

[193] J. Liu, Q. Chen, Z. Wang, Y. Tang, Y. Zhang, C. Yan, D. Wang, X. Li, and B. Zhao, “Aerialvg: A challenging benchmark for aerial visual grounding by exploring positional relations,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 5177–5187.

[194] L. Yao, F. Liu, D. Chen, C. Zhang, Y. Wang, Z. Chen, W. Xu, S. Di, and Y. Zheng, “Remotesam: Towards segment anything for earth observation,” in Proceedings ofthe 33rd ACM International Conference on Multimedia, 2025, pp. 3027–3036.

[195] Z. Sun, Y. Liu, Z. Su, H. Zhu, Y. Gu, Y. Zou, Z. Liu, G.-S. Xia, B. Du, and Y. Xu, “Refdrone: A challenging benchmark for referring expression comprehension in drone scenes,” 2025.

[196] S. Ni, D. Wang, H. Chen, H. Guo, N. Zhang, and J. Zhang, “Unigeoseg: Towards unified open-world segmentation for geospatial scenes,” 2025.

[197] C. Jiang, Y. Zhou, J. Yan, J. Li, J. Li, Y. Zhou, H. He, and J. Li, “Grasp: Geospatial pixel reasoning via structured policy learning,” 2025.

[198] J. Quenum, W.-H. Hsieh, T.-H. P. Wu, R. Gupta, T. Darrell, and D. Chan, “Lisat: Language-instructed segmentation assistant for satellite imagery,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[199] Z. Dong, Y. Sun, Y. Gu, and T. Liu, “Cross-modal bidirectional interaction model for referring remote sensing

image segmentation,” arXiv preprint arXiv:2410.08613, 2024.

[200] K. Ye, Y. Luan, Z. Chen, G. Meng, P. Dai, and L. Cao, “Ris-lad: A benchmark and model for referring lowaltitude drone image segmentation,” 2025.

[201] S. Liu, Y. Ma, X. Zhang, H. Wang, J. Ji, X. Sun, and R. Ji, “Rotated multi-scale interaction network for referring remote sensing image segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 658–26 668.

[202] K. Li, Z. Xin, L. Pang, C. Pang, Y. Deng, J. Yao, G. Xia, D. Meng, Z. Wang, and X. Cao, “Segearth-r1: Geospatial pixel reasoning via large language model,” arXiv preprint arXiv:2504.09644, 2025.

[203] Z. Yang, H. Yao, L. Tian, X. Zhao, Q. Li, and Q. Wang, “A large-scale referring remote sensing image segmentation dataset and benchmark,” 2025.

[204] Z. Yuan, L. Mou, Y. Hua, and X. X. Zhu, “Rrsis: Referring remote sensing image segmentation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–12, 2024.

[205] G. Hoxha and F. Melgani, “A novel svm-based decoder for remote sensing image captioning,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2022.

[206] W. Chen, Y. Deng, W. Jin, J. Chen, J. Chen, Y. Feng, Z. Xi, D. Liu, K. Li, and Y. Meng, “Dgtrsd and dgtrsclip: A dual-granularity remote sensing image–text dataset and vision–language foundation model for alignment,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 29 113–29 130, 2025.

[207] C. T. Marimo, B. Blumenstiel, M. Nitsche, J. Jakubik, and T. Brunschwiler, “Beyond the visible: Multispectral vision-language learning for earth observation,” pp. 359– 375, 2025.

[208] Q. Li, S. Ma, J. Luo, Y. Yu, Y. Zhou, F. Wang, X. Lu, X. Wang, X. He, Y. Chen et al., “Co-training visionlanguage models for remote sensing multi-task learning,” Remote Sensing, vol. 18, no. 2, p. 222, 2026.

[209] J. Xiao, X. Shang, A. Yao, and T.-S. Chua, “Next-qa: Next phase of question-answering to explaining tempora actions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 9777– 9786.

[210] X. Xue, G. Wei, H. Chen, H. Zhang, F. Lin, C. Shen, and X. X. Zhu, “Reo-vlm: Transforming vlm to meet regression challenges in earth observation,” arXiv preprint arXiv:2412.16583, 2024.

[211] Q. Li, X. He, X. Shu, Y. Yu, D. Chen, Y. Chen, and X. Yang, “A simple aerial detection baseline of multimodal language models,” in IGARSS 2025-2025 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2025, pp. 6833–6837.

[212] Y. Zhan, Z. Xiong, and Y. Yuan, “Skyeyegpt: Unifying remote sensing vision-language tasks via instruction tuning with large language model,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 221, pp. 64– 77, 2025.

[213] W. Zhang, M. Cai, Y. Ning, T. Zhang, Y. Zhuang, S. Lu, H. Chen, J. Li, and X. Mao, “Earthgpt-x: A spatial mllm for multilevel multisource remote sensing imagery

understanding with visual prompting,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–21, 2025.

[214] X. An, J. Sun, Z. Gui, and W. He, “Choice: benchmarking the remote sensing capabilities of large vision-language models,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[215] Y. Zhou, R. Ding, X. Yang, X. Jiang, and X. Liu, “Airspatialbot: A spatially aware aerial agent for finegrained vehicle attribute recognition and retrieval,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–12, 2025.

[216] J. C. Lin, D. N. Younessi, S. S. Kurapati, O. Y. Tang, and I. U. Scott, “Comparison of gpt-3.5, gpt-4, and human user performance on a practice ophthalmology written examination,” Eye, vol. 37, no. 17, pp. 3694–3695, 2023.

[217] W. L. Benzon, “From llm mechanisms to ringcomposition: A conversation with claude 3.5 working paper,” Available at SSRN, 2025.

[218] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805, 2023.

[219] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892–34 916, 2023.

[220] J. Wang, W. Xuan, H. Qi, Z. Chen, H. Chen, Z. Zheng, J. Xia, Y. Zhong, and N. Yokoya, “Cityvlm: Towards sustainable urban development via multi-view coordinated visionlanguage model,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 232, pp. 62–74, 2026.

[221] K. Li, G. Wan, G. Cheng, L. Meng, and J. Han, “Object detection in optical remote sensing images: A survey and a new benchmark,” ISPRS J. Photogramm. Remote Sens., vol. 159, pp. 296–307, 2020.

[222] Y. Zhang, Y. Yuan, Y. Feng, and X. Lu, “Hierarchical and robust convolutional neural network for very highresolution remote sensing object detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 8, pp. 5535–5548, 2019.

[223] A. Song and Y. Kim, “Semantic segmentation of remotesensing imagery using heterogeneous big data: International society for photogrammetry and remote sensing potsdam and cityscape datasets,” ISPRS International Journal of Geo-Information, vol. 9, no. 10, p. 601, 2020.

[224] G. Christie, N. Fendley, J. Wilson, and R. Mukherjee, “Functional map of the world,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 6172–6180.

[225] J. Wang, Z. Zheng, A. Ma, X. Lu, and Y. Zhong, “Loveda: A remote sensing land-cover dataset for domain adaptive semantic segmentation,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, J. Vanschoren and S. Yeung, Eds., vol. 1. Curran Associates, Inc., 2021. [Online]. Available: https://datasets-benchmarks-proceedings. neurips.cc/paper files/paper/2021/file/ 4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf

[226] B. Zoph, G. Ghiasi, T.-Y. Lin, Y. Cui, H. Liu, E. D. Cubuk, and Q. Le, “Rethinking pre-training and self-

training,” Advances in neural information processing systems, vol. 33, pp. 3833–3845, 2020.

[227] F.-L. Chen, D.-Z. Zhang, M.-L. Han, X.-Y. Chen, J. Shi, S. Xu, and B. Xu, “Vlp: A survey on vision-language pretraining,” Machine Intelligence Research, vol. 20, no. 1, pp. 38–56, 2023.

[228] W. Ma, R. Zhang, X. Ning, H. Zhang, and Y. Chen, “Exploring the potential of vlms in remote sensing through prompt optimization,” The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. 48, pp. 219–226, 2025.

[229] J. Lin, H. Yin, W. Ping, P. Molchanov, M. Shoeybi, and S. Han, “Vila: On pre-training for visual language models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26 689–26 699.

[230] J. Yang, Z. Li, S. Xie, W. Zhu, W. Yu, and S. Li, “Crossmodal adapter: Parameter-efficient transfer learning approach for vision-language models,” in 2024 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2024, pp. 1–6.

[231] J. Li and H. Sun, “Lift: Transfer learning in visionlanguage models for downstream adaptation and generalization,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 4678–4687.

[232] K. Lv, Y. Yang, T. Liu, Q. Guo, and X. Qiu, “Full parameter fine-tuning for large language models with limited resources,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 8187–8198.

[233] Y. Liu, Y. Zhang, Q. Li, T. Liu, S. Feng, D. Wang, Y. Zhang, and H. Schutze, “Hift: A hierarchica¨ full parameter fine-tuning strategy,” arXiv preprint arXiv:2401.15207, 2024.

[234] N. Ding, Y. Qin, G. Yang, F. Wei, Z. Yang, Y. Su, S. Hu, Y. Chen, C.-M. Chan, W. Chen et al., “Parameter-efficient fine-tuning of large-scale pre-trained language models,” Nature machine intelligence, vol. 5, no. 3, pp. 220–235, 2023.

[235] L. Wang, S. Chen, L. Jiang, S. Pan, R. Cai, S. Yang, and F. Yang, “Parameter-efficient fine-tuning in large language models: a survey of methodologies,” Artificial Intelligence Review, vol. 58, no. 8, p. 227, 2025.

[236] G. Dong, H. Yuan, K. Lu, C. Li, M. Xue, D. Liu, W. Wang, Z. Yuan, C. Zhou, and J. Zhou, “How abilities in large language models are affected by supervised finetuning data composition,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 177–198.

[237] R. Wang, Y. Hao, L. Hu, J. Chen, M. Chen, and D. Wu, “Self-supervised learning with data-efficient supervised fine-tuning for crowd counting,” IEEE Transactions on Multimedia, vol. 25, pp. 1538–1546, 2023.

[238] Y. Fan, O. Watkins, Y. Du, H. Liu, M. Ryu, C. Boutilier, P. Abbeel, M. Ghavamzadeh, K. Lee, and K. Lee, “Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models,” Advances in Neural Information Processing Systems, vol. 36, pp. 79 858–79 885, 2023.

[239] S. Zhai, H. Bai, Z. Lin, J. Pan, P. Tong, Y. Zhou, A. Suhr, S. Xie, Y. LeCun, Y. Ma et al., “Fine-tuning large vision-language models as decision-making agents via

reinforcement learning,” Advances in neural information processing systems, vol. 37, pp. 110 935–110 971, 2024.

[240] I. Ziegler, A. Koksal, D. Elliott, and H. Sch¨ utze, “Craft¨ your dataset: Task-specific synthetic dataset generation through corpus retrieval and augmentation,” Transactions of the Association for Computational Linguistics, vol. 13, pp. 1693–1721, 2025.

[241] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” ICLR, vol. 1, no. 2, p. 3, 2022.

[242] Y. Bazi, L. Bashmal, M. M. Al Rahhal, R. Ricci, and F. Melgani, “Rs-llava: A large vision-language model for joint captioning and question answering in remote sensing imagery,” Remote Sensing, vol. 16, no. 9, 2024.

[243] J. Pan, C. Liu, J. Wu, F. Liu, J. Zhu, H. B. Li, C. Chen, C. Ouyang, and D. Rueckert, “Medvlm-r1: Incentivizing medical reasoning capability of vision-language models (vlms) via reinforcement learning,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2025, pp. 337–347.

[244] J. Rodriguez-Juan, D. Ortiz-Perez, J. Garcia-Rodriguez, D. Tomas, and G. J. Nalepa, “Integrating advanced vision-´ language models for context recognition in risks assessment,” Neurocomputing, vol. 618, p. 129131, 2025.

[245] H. Wu, Z. Du, D. Zhong, Y. Wang, and C. Tao, “Fsvlm: A vision-language model for remote sensing farmland segmentation,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[246] E. Boisseau, “Imitation and large language models,”<sup>´</sup> Minds and Machines, vol. 34, no. 4, p. 42, 2024.

[247] R. C. Alvarado, “Turings children: Large language models and imitative intelligence,” Social Science Computer Review, p. 08944393251364293, 2025.

[248] R. Agnihotri and A. Pandey, “Efficient online rft with plug-and-play llm judges: Unlocking state-of-the-art performance,” arXiv preprint arXiv:2506.05748, 2025.

[249] Z. Zhang, Z. Guan, T. Zhao, H. Shen, Y. Cai, Z. Su, Y. Shang, Z. Liu, J. Yin, and X. Li, “Geo-r1: Improving few-shot geospatial referring expression understanding with reinforcement fine-tuning,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 237, pp. 113–129, 2026.

[250] D. Wang, S. Liu, W. Jiang, F. Wang, Y. Liu, X. Qin, Z. Luo, C. Zhou, H. Guo, J. Zhang, B. Du, D. Tao, and L. Zhang, “Geozero: Incentivizing reasoning from scratch on geospatial scenes,” 2025.

[251] K. Stechly, K. Valmeekam, and S. Kambhampati, “Chain of thoughtlessness? an analysis of cot in planning,” Advances in Neural Information Processing Systems, vol. 37, pp. 29 106–29 141, 2024.

[252] Y. Xia, R. Wang, X. Liu, M. Li, T. Yu, X. Chen, J. McAuley, and S. Li, “Beyond chain-of-thought: A survey of chain-of-x paradigms for llms,” in Proceedings of the 31st International Conference on Computational Linguistics, 2025, pp. 10 795–10 809.

[253] Q. Lyu, S. Havaldar, A. Stein, L. Zhang, D. Rao, E. Wong, M. Apidianaki, and C. Callison-Burch, “Faithful chainof-thought reasoning,” in The 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Associa-

tionfor Computational Linguistics (IJCNLP-AACL 2023), 2023.

[254] Y. Zhou, J. Chen, Z. Zhang, P. Huang, R. Ding, Z. Zou, P. F. Gao, Y. Wei, K. Li, and X. Yang, “Dvgbench: Implicit-to-explicit visual grounding benchmark in uav imagery with large visionlanguage models,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 232, no. c, pp. 831–847, 2026.

[255] J. D. Silva, J. Magalhes, D. Tuia, and B. Martins, “Large language models for captioning and retrieving remote sensing images,” 2024.

[256] K. Papineni, S. Roukos, T. Ward, and W. J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proc Meeting of the Association for Computational Linguistics, 2002.

[257] B. Satanjeev, “Meteor: An automatic metric for mt evaluation with improved correlation with human judgments,” ACL-2005, pp. 228–231, 2005.

[258] C. Y. Lin, “Rouge: A package for automatic evaluation of summaries,” 2004.

[259] R. Vedantam, C. L. Zitnick, and D. Parikh, “Cider: Consensus-based image description evaluation,” 2014.

[260] S. Antol, A. Agrawal, J. Lu, M. Mitchell, and D. Parikh, “Vqa: Visual question answering,” International Journa of Computer Vision, vol. 123, no. 1, pp. 4–31, 2015.

[261] M. Ren, R. Kiros, and R. Zemel, “Exploring models and data for image question answering,” Litoral Revista De La Poesa Y El Pensamiento, pp. 2953–2961, 2015.

[262] X. Liu and Z. Lian, “Rsunivlm: A unified vision-language model for remote sensing via granularity-oriented moe,” Pattern Recognition, p. 113717, 2026.

[263] Z. Zhang, H. Shen, T. Zhao, Z. Guan, B. Chen, Y. Wang, X. Jia, Y. Cai, Y. Shang, and J. Yin, “Enhancing ultrahigh resolution remote sensing imagery analysis with imagerag: A new framework,” IEEE Geoscience and Remote Sensing Magazine, vol. 13, no. 3, pp. 369–394, 2025.

[264] Y. Zhou, C. Jiang, C. Yuan, and J. Li, “Look where it matters: Training-free ultra-hr remote sensing vqa via adaptive zoom search,” 2025.

[265] L. Yao, F. Liu, H. Lu, C. Zhang, R. Min, S. Xu, S. Di, and P. Peng, “Remotereasoner: Towards unifying geospatial reasoning workflow,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 14, 2026, pp. 11 883–11 891.

[266] J. Mao, J. Huang, A. Toshev, O. Camburu, and K. Murphy, “Generation and comprehension of unambiguous object descriptions,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[267] M. Zhou, H. Guo, X. Ouyang, D. Gunasekera, and Z. Sun, “Land surface temperature retrieval from sdgsat-1: assessment of different retrieval algorithms with different atmospheric reanalysis data,” International Journal of Digital Earth, vol. 18, no. 1, 2025.

[268] J. Yang, S. Yang, A. W. Gupta, R. Han, L. Fei-Fei, and S. Xie, “Thinking in space: How multimodal large language models see, remember, and recall spaces,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 10 632–10 643.

[269] K. He, G. Gkioxari, P. Dollar, and R. Girshick, “Mask r-´

cnn,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2961–2969.

[270] B. Wu, H. Huang, Y. Wang, S. Shi, J. Wu, and B. Yu, “Global spatial patterns between nighttime light intensity and urban building morphology,” International Journal of Applied Earth Observation and Geoinformation, vol. 124, p. 103495, 2023.

[271] J. Zhu, Y. Xu, L. Hoegner, and U. Stilla, “Generation of thermal point clouds from uncalibrated thermal infrared image sequences and mobile laser scans,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–16, 2023.

[272] Y. Gao, C. Li, Z. You, J. Liu, Z. Li, P. Chen, Q. Chen, Z. Tang, L. Wang, P. Yang et al., “Openfly: A versatile toolchain and large-scale benchmark for aerial visionlanguage navigation,” arXiv e-prints, pp. arXiv–2502, 2025.

[273] L. Zhang, Y. Zhang, H. Li, H. Fu, Y. Tang, H. Ye, L. Chen, X. Liang, X. Hao, and W. Ding, “Is your vlm sky-ready? a comprehensive spatial intelligence benchmark for uav navigation,” arXiv preprint arXiv:2511.13269, 2025.

[274] J. Zha, Y. Fan, T. Zhang, G. Chen, Y. Chen, C. Gao, and X. Chen, “Aircopbench: A benchmark for multidrone collaborative embodied perception and reasoning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 2, 2026, pp. 1507–1515.

[275] Y. Wang, J. Zhao, Z. Fan, X. Zhang, X. Wu, Y. Zhang, L. Jin, X. Li, G. Wang, M. Jia et al., “Jtd-uav: Mllmenhanced joint tracking and description framework for anti-uav systems,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1633– 1644.

[276] X. Wang, D. Yang, Y. Liao, W. Zheng, B. Dai, H. Li, S. Liu et al., “Uav-flow colosseo: A real-world benchmark for flying-on-a-word uav imitation learning,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[277] W. Zhang, Z. Zhou, X. Zeng, L. Xuchen, J. Fang, C. Gao, J. Cui, Y. Li, X. Chen, and X.-P. Zhang, “Open3d-vqa: A benchmark for embodied spatial concept reasoning with multimodal large language model in open space,” ser. MM ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 1278412791.