# AgroVisNet: A lightweight Convolutional Network and the BD-PlantDX Expert-Validated Benchmark for Radish, Potato and Pointed Gourd Disease Classification

Md. Abdullah Mandal<sup>b,∗</sup>, Saad Ahmed<sup>a</sup>, Md. Khalid Syfullah<sup>b</sup>

<sup>a</sup>Department of Computer Science and Engineering, Rajshahi University of Engineering & Technology, Rajshahi, 6204, Bangladesh

<sup>b</sup>Department of Computer Science and Engineering, Bangladesh Army University of Science & Technology, Saidpur, Bangladesh

## Abstract

Automated plant disease diagnosis is increasingly deployed on farmer-held devices in regions where agronomic expertise is scarce and network connectivity is unreliable. Three obstacles limit its practical value: public benchmarks are dominated by a small set of non-native crops, region-specific datasets are rarely validated by domain experts, and the architectures that reach competitive accuracy carry parameter budgets that are unsuited to low-cost hardware. We propose AgroVisNet, a compact convolutional network trained from scratch, together with BD-PlantDX, an expert-validated benchmark of 12,432 field images spanning 12 classes of radish, potato and pointed gourd in healthy and diseased states, collected across the Bogura and Nilphamari districts of Bangladesh. AgroVisNet couples grouped bottleneck residual blocks carrying sequential channel and spatial attention with multi-scale depthwise blocks and a dual-pooling classification head, reaching 290,572 trainable parameters. On BD-PlantDX the model attains 99.52% test accuracy and 99.52% weighted F1, exceeding all six ImageNet-pretrained lightweight backbones evaluated under an identical protocol while using 8.7 to 16.8 times fewer parameters and 1.3 to 8.5 times fewer multiply– accumulate operations. Exported for deployment, the model quantises to a 0.46 MB full-integer network at a 0.22 percentage-point accuracy cost and classifies an image in 8.40 ms on a single CPU. Across five random seeds accuracy remains at 99.57 ± 0.10%, a ten-variant ablation isolates the contribution of each component, and the same architecture transfers without redesign to two independently collected datasets at 98.71% and 99.05% accuracy. Grad-CAM evidence indicates that predictions rest on lesion-bearing leaf regions rather than on background cues.

Keywords: plant disease classification, lightweight convolutional network, expert-validated dataset, attention mechanism, explainable artificial intelligence

## 1. Introduction

Timely identification of crop disease governs both yield and the quantity of chemical input a grower applies. Radish (Raphanus sativus) is a commonly grown vegetable, especially in Asian nations, and holds significant importance in human diet, local farming, and economic activities [37]. Late detection of potato disease in Bangladesh was associated with average yield losses of 25 to 57% over the 2016 to 2019 seasons, placing these pathologies among the most economically damaging in the country [24]. Manual scouting, the prevailing practice, depends on visual inspection by trained personnel and delivers neither the coverage nor the response time that early intervention requires [57]. Domain specialists able to diferentiate fungal lesions, insect damage and nutrient disorders are scarce and expensive to reach from remote holdings, which leaves growers to apply broad-spectrum pesticide on the basis of an uncertain diagnosis [42]. Computer vision ofers a route around this constraint. Convolutional neural networks trained on leaf imagery now report accuracies above 99% on public benchmarks, and a smartphone camera is suficient to acquire the input [7, 13]. Molecular and spectroscopic alternatives, including DNA-based point-of-care assays [29], mass spectrometry imaging [2] and near-infrared spectroscopy [35], achieve high specificity, yet their cost and procedural complexity keep them outside the reach of a smallholder. Image-based diagnosis therefore remains the only modality that combines diagnostic value with a deployment cost a grower can absorb. Two gaps persist across the published work in this area. The first concerns data. A large share of reported results is obtained on PlantVillage [18, 48, 50], a laboratory-acquired collection of 54,305 images covering 14 crop species, none of which is native to the cropping systems of northern Bangladesh. Models fitted to that distribution encounter a diferent pathogen profile, a diferent cultivar set and a diferent imaging environment once deployed locally, and the few region-specific collections that exist are typically limited to one crop and carry no record of agronomist verification [6, 16]. The second gap concerns capacity. Reported accuracy gains are frequently obtained by fine-tuning backbones of 5 to 25 million parameters [39, 49], a budget that resists deployment on the low-cost handsets that dominate rural Bangladesh, and the compact backbones proposed as an alternative are pretrained on ImageNet rather than designed for the fine-grained, texture-driven discrimination that lesion classification demands. Existing work addresses at most one of these gaps at a time. Region-specific datasets are released without a matching eficient architecture [17], and compact architectures are validated only on laboratory imagery of non-native crops [64]. To the best of our knowledge, no prior study jointly delivers an expert-validated multi-crop dataset for this cropping system and a sub-megabyte model that matches ImageNet-pretrained backbones on it. This paper closes both gaps together. The contributions of this paper are fivefold:

• We introduce BD-PlantDX, an expert-validated benchmark of 12,432 images spanning 12 classes drawn from radish, potato and pointed gourd, each crop represented in healthy and diseased states. Acquisition covered the Bogura and Nilphamari districts of Bangladesh, class assignment was verified by practicing agronomists, and the collection includes root as well as foliar presentations.

• We propose AgroVisNet, a convolutional network trained from scratch that combines grouped bottleneck residual blocks carrying sequential channel and spatial attention, multi-scale depthwise blocks and a dual-pooling classification head within 290,572 trainable parameters. The individual components are drawn from established literature; the contribution lies in the composition and in the parameter budget it achieves.

• We benchmark AgroVisNet against six ImageNet-pretrained lightweight backbones under an identical data pipeline, split and evaluation protocol, showing that a model trained from scratch at 0.29M parameters exceeds every baseline while using 8.7 to 16.8 times fewer parameters and 1.3 to 8.5 times fewer multiply–accumulate operations.

• We verify that the architecture transfers without redesign to two independently collected Bangladeshi datasets, reaching 98.71% accuracy over 21 vegetable classes and 99.05% accuracy over 5 radish classes.

• We report a five-seed reproducibility study, an eleven-variant component ablation and Grad-CAM interpretability evidence, establishing that the reported accuracy is stable across initialization and that predictions rest on lesion-bearing regions.

The study is organized around five research questions:

• RQ1: Can a convolutional network trained from scratch with fewer than 0.3M parameters match ImageNet-pretrained lightweight backbones on a region-specific, multi-crop disease benchmark?

• RQ2: Which architectural components of AgroVisNet contribute to its accuracy, and what parameter cost does each one carry?

• RQ3: Does the architecture generalize to independently collected plant disease datasets without any change to its structure or its training recipe?

• RQ4: How stable is the reported performance across random initialization and stochastic optimization?

• RQ5: Do the learned representations attend to disease-bearing leaf regions rather than to background or acquisition artifacts?

The remainder of this paper is organized as follows. Section 2 introduces the necessary background, Section 3 reviews related work, Section 4 describes the construction and composition of BD-PlantDX, Section 5 presents AgroVisNet and the experimental protocol, Section 6 reports the results, and Section 7 concludes.

## 2. Background

This section introduces the concepts on which the remainder of the paper depends, at a descriptive level; the formal treatment of each mechanism, together with the notation used throughout, is deferred to Section 5. Figure 1 places the concepts within the end-to-end pipeline of an image-based plant disease diagnosis system.

## 2.1. Plant Disease Classification as a Supervised Vision Task

Plant disease classification is posed as multi-class image classification. A color photograph of a leaf or a root is presented to a trained network, which returns a confidence score for every healthy and diseased condition in the label set and assigns the image to the condition receiving the highest score. The task is fine grained rather than merely multi-class: several conditions difer only in lesion texture, in the morphology of the lesion margin or in the pattern of yellowing, while sharing leaf shape, venation and overall color envelope. This visual proximity between classes, rather than the raw number of classes, determines the representational demand placed on the network and motivates every design decision described in Section 5.

## 2.2. Residual and Bottleneck Convolutional Blocks

Residual learning adds a direct path from the input of a group of convolutional layers to its output, allowing the group to learn a correction to its input rather than a complete transformation of it. The direct path carries gradients back to early layers without attenuation, which permits substantial depth without the accuracy degradation that plagued earlier deep stacks [19]. The bottleneck variant narrows the representation before the spatial convolution and widens it again afterward, confining the expensive spatial operation to a reduced channel space and lowering its cost sharply. AgroVisNet adopts the bottleneck form throughout.

## 2.3. Grouped and Depthwise Separable Convolution

The weight count of a conventional convolution grows with the product of its input and output channel counts, which makes it the dominant cost in a deep network. Grouped convolution partitions the channels into disjoint groups and convolves within each group independently, dividing that cost by the number of groups [28]. Depthwise separable convolution takes this partitioning to its limit, filtering each channel in isolation and then recombining the channels with a pointwise projection, the construction underlying MobileNet [21]. Both mechanisms trade a degree of cross-channel mixing for parameter economy, and AgroVisNet applies both at diferent depths of the network.

![](images/366ca41a1e041a148e2b11442236ec28f88ee62fa2f242df95b54e07dc81c0eb.jpg)  
Figure 1: Conceptual pipeline of an image-based plant disease diagnosis system, from field acquisition through preprocessing, representation learning and interpretability to the diagnostic decision returned to the grower.

## 2.4. Channel and Spatial Attention

Attention modules rescale intermediate feature maps by a gate that the network learns alongside its filters. Channel attention summarizes each feature map into a single descriptor, passes the descriptors through a small perceptron and emits one multiplier per channel, which amplifies informative filters and suppresses the rest, the mechanism introduced by squeeze-and-excitation networks [22]. Spatial attention instead summarizes across channels and emits one multiplier per spatial location, which concentrates response on the region carrying the evidence. The convolutional block attention module composes the two in sequence and reports consistent gains at negligible parameter cost [68]. Lesion classification benefits from both forms: the channel gate emphasizes texture-selective filters, and the spatial gate directs the response toward the lesion rather than the surrounding healthy tissue.

## 2.5. Regularization and Activation

Batch normalization standardizes the intermediate activations within each mini-batch, which stabilizes optimization and permits higher learning rates [23]. Dropout counters co-adaptation between units by removing a random subset of them at each training step, and its structured variant spatial dropout removes entire feature maps instead, which is the appropriate form for convolutional tensors whose neighboring activations are strongly correlated [58, 61]. The swish activation is smooth and non-monotonic, and it is reported to outperform the rectified linear unit on deeper convolutional stacks [47]. AgroVisNet uses all four.

## 2.6. Gradient-Weighted Class Activation Mapping

Grad-CAM produces a class-discriminative localization map by weighting the activations of a late convolutional layer according to how strongly each one influences the score of the predicted class, then overlaying the result on the input image [53]. In agricultural diagnosis its role is diagnostic rather than decorative. A model that separates classes on background soil, on illumination gradients or on artifacts of the acquisition setup will produce maps that fall away from the lesion, and this failure mode remains invisible in aggregate accuracy, surfacing only once the model meets imagery from a diferent collection.

## 3. Literature Review

We organize prior work along five research lines: transfer learning on public benchmarks, purpose-built architectures for radish, potato and pointed gourd, eficient architectures for on-device inference, regionspecific data collection, and interpretability. Table 1 summarizes the studies most directly comparable to ours.

## 3.1. Transfer Learning on Public Benchmarks

The dominant methodology fine-tunes an ImageNet-pretrained backbone on PlantVillage. Hassan et al. [18] evaluated InceptionV3, InceptionResNetV2, MobileNetV2 and EficientNetB0 across the 38 classes of the benchmark, with EficientNetB0 reaching 99.56%. Pandian et al. [39] scaled the setting to 147,500 images over 59 classes with a 14-layer network and three augmentation strategies, reporting 99.97%. Harakannanavar et al. [15] contrasted classical pipelines built on discrete wavelet transform, principal component analysis and gray-level co-occurrence matrix features against convolutional models on tomato imagery, with the convolutional model reaching 99.6% against 88% for a support vector machine. Ali et al. [5] pushed an ensemble of deep architectures to 99.89% on the same benchmark. These studies establish that the PlantVillage distribution is close to saturated, yet accuracy on it carries limited information about behavior on imagery acquired under a diferent pathogen profile and a diferent acquisition protocol.

## 3.2. Purpose-Built Architectures for radish and potato

A second line designs architectures for a single crop. Rozaqi and Sunyoto [50], Sanjeev et al. [52] and Barman et al. [8] each proposed a network for the three-class potato problem of early blight, late blight and healthy leaf, reporting 92%, 96.5% and 96.75% respectively, all trained on PlantVillage. Rashid et al. [48] raised this to 99.75% with a multi-level formulation. Reis and Turk [49] introduced MDSCIRNet, which composes depthwise separable convolution with multi-head attention and reaches 99.24% alone and 99.33% when its features are passed to a support vector machine. Dame et al. [11] classified 4,200 smartphone potato images into three conditions at 99% accuracy and additionally graded severity into six levels at 96%. Radish has received markedly less attention: Banerjee et al. [6] reported 92% over five radish diseases with a convolutional and support-vector hybrid, and Alam et al. [3] applied a modified Mask R-CNN to radish organ segmentation rather than to disease. Pointed gourd, to the best of our knowledge, has not been treated in this literature at all.

## 3.3. Eficient Architectures for On-Device Inference

Deployment on grower-held hardware has motivated a family of compact backbones that falls into three design lineages. The purely convolutional line comprises GhostNetV2, which generates part of its feature maps through cheap linear operations [59], RepGhostNet, which reparameterizes the same idea for hardware eficiency [10], ConvNeXt, which modernizes the plain convolutional stack with large kernels and inverted bottlenecks [32], and MobileNetV4, whose universal inverted bottleneck is searched for Pareto-optimal la tency across mobile processors and accelerators [43]. A second line applies structural reparameterization to hybrid designs, with FastViT removing skip connections from its token mixer to lower memory access cost [63] and RepViT transferring the block-level, macro and micro design choices of lightweight transformers back onto a pure convolutional network [66]. A third line reduces the cost of attention itself: EdgeNeXt and MobileViTv2 interleave convolution with separable self-attention [33, 34], SwiftFormer replaces quadratic attention with an additive formulation [55], and EficientFormerV2 searches latency and parameter count jointly to reach MobileNet-scale budgets [31]. Within agriculture, Vishnoi et al. [64] reduced layer count to classify apple disease at 98% with lower storage and execution cost, and Rakesh et al. [46] deployed ResNet50 and DenseNet121 on a Raspberry Pi 4B for root crop classification at 99.60% and 97.60%. These architec tures establish that competitive accuracy is attainable in the one to five million parameter range, however all of them are pretrained on ImageNet and none is designed against the texture-driven discrimination that lesion classification demands.

## 3.4. Region-Specific Data Collection

A smaller line of work addresses the data gap directly. Wäldchen et al. [65] argued that automated species and disease identification requires collections representative of the target climatic region. Hasan et al. [17] released 25 categories of Bangladeshi vegetable leaf imagery and Hasan et al. [16] followed with 2,801 radish leaf images across one healthy and four diseased classes. Qin et al. [44] evaluated ResNet variants on 15,207 field images covering 201 species, reporting 91.83% to 92.95%. These releases confirm the value of native imagery, yet each covers a single crop family, none reports formal agronomist validation of the class assignments, and none is paired with an architecture designed for the resulting distribution.

## 3.5. Interpretability in Agricultural Diagnosis

Interpretability has entered the field as a trust requirement. Alhammad et al. [4] combined transfer learning with Grad-CAM on 2,152 potato images, reaching 98% test accuracy with visual explanations of each decision. Paul et al. [40] applied LIME and SHAP to a deep ensemble over four potato diseases collected in West Bengal. These studies treat interpretability qualitatively, presenting selected heatmaps without a quantitative agreement measure against expert annotation.

## 3.6. Research Gap

Across these five lines, existing methods satisfy at most two of the three requirements of our setting: an expert-validated dataset representative of the local cropping system, a parameter budget compatible with low-cost hardware, and evidence that accuracy survives a change of dataset. Transfer learning studies satisfy none of the three, purpose-built potato architectures satisfy the second alone, eficient backbones satisfy the second while relying on ImageNet pretraining, and region-specific data releases satisfy the first while ofering no matching model. BD-PlantDX and AgroVisNet are constructed to satisfy all three jointly.

## 4. The BD-PlantDX Dataset

BD-PlantDX is an expert-validated collection of 12,432 field images covering 12 classes drawn from three vegetable crops of northern Bangladesh, each represented in healthy and diseased condition. This section reports the agronomic rationale for the crop selection, the acquisition and curation protocol, the expert validation procedure, the composition of the release and the properties that distinguish it from existing collections. Figure 2 summarizes the workflow from field survey to released split.

## 4.1. Crop Selection and Survey Regions

Vegetable production occupies 2.57% of the arable land of Bangladesh and yields 3.73 million tons annually, a return per hectare that exceeds that of the staple cereals [14]. Within this sector the Bogura and Nilphamari districts are among the most intensive winter vegetable regions of the country [60]. Consultation with local agronomists identified radish (Raphanus sativus), potato (Solanum tuberosum) and pointed gourd (Trichosanthes dioica, known locally as potol) as the three crops that combine the largest cultivated area with the highest reported disease-driven loss in these districts. Radish and potato are represented in the existing literature, however almost always through a single foliar presentation and on imagery acquired outside South Asia; pointed gourd, to the best of our knowledge, is absent from the published disease classification literature entirely.

Table 1: Overview of deep learning approaches to radish and potato disease classification. Accuracies are quoted from the original publications and were obtained under difering splits and protocols.
<table><tr><td>Study</td><td>Crop Classes</td><td></td><td>Dataset</td><td>Method</td><td>Accuracy (%)</td></tr><tr><td>Rozaqi and Sunyoto [50]</td><td>Potato</td><td>3</td><td>PlantVillage</td><td>CNN</td><td>92.00</td></tr><tr><td>Sanjeev et al. [52]</td><td>Potato</td><td>3</td><td>PlantVillage</td><td>Feed-forward network</td><td>96.50</td></tr><tr><td>Barman et al. [8]</td><td>Potato</td><td>3</td><td>PlantVillage</td><td>Self-built CNN</td><td>96.75</td></tr><tr><td>Rashid et al. [48]</td><td>Potato</td><td>3</td><td>PlantVillage</td><td>Multi-level CNN</td><td>99.75</td></tr><tr><td>Lee et al. [30]</td><td>Potato</td><td>3</td><td>PlantVillage</td><td>CNN</td><td>99.00</td></tr><tr><td>Jha et al. [26]</td><td>Potato</td><td>3</td><td>Field-collected</td><td>Stacking ensemble</td><td>98.86</td></tr><tr><td>Reis and Turk [49]</td><td>Potato</td><td>4</td><td>Public composite</td><td>MDSCIRNet with SVM</td><td>99.33</td></tr><tr><td>Alam et al. [3]</td><td>Radish</td><td>2</td><td>Field-collected</td><td>Mask R-CNN</td><td>96.30</td></tr><tr><td>Banerjee et al. [6]</td><td>Radish</td><td>5</td><td>Field-collected</td><td>CNN with SVM</td><td>92.00</td></tr><tr><td>Ullah et al. [62]</td><td>Multiple</td><td>8</td><td>PlantVillage</td><td>DeepPlantNet</td><td>98.49</td></tr><tr><td>Geetharamani and Pandian [13]</td><td>Multiple</td><td>39</td><td>PlantVillage</td><td>Nine-layer CNN</td><td>96.46</td></tr><tr><td>Zhang et al. [69]</td><td>Tomato</td><td>4</td><td>AIChallenger</td><td>Improved Faster R-CNN</td><td>97.10</td></tr><tr><td>Qin et al. [44]</td><td>Multiple</td><td>201</td><td>Field-collected</td><td>Res2Net-101</td><td>92.95</td></tr><tr><td>Shafik et al. [54]</td><td>Multiple</td><td>15</td><td>TPPD</td><td>ResNet-9</td><td>97.40</td></tr><tr><td>Nawaz et al. [36]</td><td>Potato</td><td>3</td><td>PlantVillage</td><td>PotatoGuardNet</td><td>99.41</td></tr><tr><td>Sanchez-Sanchez et al. [51]</td><td>Potato</td><td>2</td><td>Field-collected</td><td>ResNet50</td><td>93.00</td></tr><tr><td>Dame et al. [11]</td><td>Potato</td><td>3</td><td>Field-collected</td><td>CNN</td><td>99.00</td></tr><tr><td>Sinamenye et ai. [56]</td><td>Potato</td><td>7</td><td>Field-collected</td><td>Hybrid deep learning</td><td>85.06</td></tr><tr><td>Quoc et al. [45]</td><td>Radish</td><td>5</td><td>Bangladesh</td><td>SCOLD</td><td>94.37</td></tr><tr><td>Ji et al. [27]</td><td>Radish</td><td>6</td><td>Field-collected</td><td>Hybrid CNN-Transformer</td><td>91.00</td></tr><tr><td>This work</td><td>Radish, pointed gourd,</td><td>12</td><td>BD-PlantDX</td><td>AgroVisNet (0.29M params)</td><td>99.52</td></tr></table>

## 4.2. Acquisition Protocol

Field surveys were conducted across several agricultural sites in the two districts over the winter cropping season. Local agronomists accompanied each survey and identified the diseased plots in advance, which ensured that the sampled plants carried the pathology of interest rather than an incidental stress symptom. Specimens were harvested from the identified plots and cataloged against the plot record before imaging. Two consumer smartphones, a POCO M3 and an OPPO F16, served as the acquisition devices, which reflects the sensor class a grower would realistically use at inference time. Each specimen was photographed from five angles under a controlled illumination arrangement that suppresses direct sunlight, preserves color fidelity and avoids the saturation that removes lesion texture. Specimen position was rotated manually between exposures to capture the adaxial surface, the abaxial surface and the lesion margin. Images were stored as JPEG at the native resolution of the capture device, a format chosen for its storage economy and its universal support in mobile acquisition pipelines.

## 4.3. Curation and Expert Validation

Raw captures were screened for motion blur, severe defocus and duplicate framing, and the surviving images were assigned to one of the 12 classes. Each class assignment was then reviewed by practicing agronomists, who confirmed the pathology, the afected organ and the severity band on a per-image basis; images on which the reviewers disagreed were removed rather than resolved by majority. The signed validation record accompanies the release. This step distinguishes BD-PlantDX from the region-specific collections currently available, which report the label taxonomy without documenting an independent verification of the assignments [16, 17]. Filtering was additionally applied to normalize brightness across acquisition sessions, and no synthetic or generative augmentation entered the released images; all augmentation reported in Section 5 is applied at training time only.

![](images/49e25c4c679859f3d9b21496ff2f2da2cf8bf44473fec574c13060a6552560d4.jpg)

Figure 2: Construction workflow of BD-PlantDX, from agronomist-guided field survey and multi-angle acquisition through curation, expert validation of the class assignment and stratified partition into training, validation and test splits.  
![](images/54d20f8034ab5ac130946ac0b9b0146cd84ee9b0af52f10aed4f2f7c88123912.jpg)  
Figure 3: Representative images from each of the 12 BD-PlantDX classes, arranged by crop, with healthy presentations in the first column of each crop group.

## 4.4. Composition

The release contains 12,432 images across 12 classes. Radish contributes six classes: two healthy presentations covering the root and the leaf, together with Alternaria brassicae blight, flea beetle damage, scab and white mold. Potato contributes four classes: a healthy leaf, late blight fungus, leaf mosaic and nutrient deficiency. Pointed gourd contributes two classes: a healthy leaf and downy mildew. The taxonomy therefore spans four distinct etiologies, namely fungal infection, viral infection, insect damage and abiotic nutrient disorder, which is a wider causal range than single-pathogen benchmarks provide and which forces the model to separate visually similar chlorotic patterns of diferent origin. Inclusion of the radish root alongside the radish leaf further requires the model to operate across two organ morphologies within one label space. Table 2 lists the per-class composition and Figure 4 shows the distribution over the three splits. Class counts range from 1,002 to 1,158 images, giving a maximum imbalance ratio of 1.16 and removing the need for resampling or class-weighted loss. Figure 3 presents representative images for each class.

## 4.5. Partitioning

The collection is partitioned once into 8,694 training, 1,868 validation and 1,870 test images under a stratified 70/15/15 scheme, and this fixed partition is reused without modification by every experiment in Section 6. Partitioning is applied at the specimen level rather than at the image level, which prevents the five exposures of one physical leaf from being distributed across splits and removes the optimistic bias that imagelevel partitioning introduces into multi-view collections. The test split is touched once per configuration, after model selection has been completed on the validation split.

Table 2: Composition of BD-PlantDX. Counts follow a stratified 70/15/15 partition applied within each class.
<table><tr><td>Crop</td><td>Class</td><td>Condition</td><td>Train</td><td>Val.</td><td>Test</td><td>Total</td></tr><tr><td>Potato</td><td>Potato Healthy</td><td>Healthy</td><td>708</td><td>152</td><td>152</td><td>1,012</td></tr><tr><td>Pointed gourd</td><td>Gourd Healthy</td><td>Healthy</td><td>706</td><td>152</td><td>152</td><td>1,010</td></tr><tr><td>Radish</td><td>Radish Root Healthy</td><td>Healthy</td><td>810</td><td>174</td><td>174</td><td>1,158</td></tr><tr><td>Radish</td><td>Radish Leaf Healthy</td><td>Healthy</td><td>727</td><td>156</td><td>157</td><td>1,040</td></tr><tr><td>Potato</td><td>Potato Late Blight</td><td>Diseased</td><td>700</td><td>151</td><td>151</td><td>1,002</td></tr><tr><td>Potato</td><td>Potato Mosaic</td><td>Diseased</td><td>709</td><td>152</td><td>152</td><td>1,013</td></tr><tr><td>Potato</td><td>Potato Nutrient Deficiency</td><td>Diseased</td><td>707</td><td>152</td><td>152</td><td>1,011</td></tr><tr><td>Pointed gourd</td><td>Gourd Downy Mildew</td><td>Diseased</td><td>710</td><td>152</td><td>153</td><td>1,015</td></tr><tr><td>Radish</td><td>Radish Alternaria brassicae</td><td>Diseased</td><td>718</td><td>154</td><td>154</td><td>1,026</td></tr><tr><td>Radish</td><td>Radish Flea Beetle Damage</td><td>Diseased</td><td>707</td><td>152</td><td>152</td><td>1,011</td></tr><tr><td>Radish</td><td>Radish Scab</td><td>Diseased</td><td>781</td><td>168</td><td>168</td><td>1,117</td></tr><tr><td>Radish</td><td>Radish White Mold</td><td>Diseased</td><td>711</td><td>153</td><td>153</td><td>1,017</td></tr><tr><td>Total</td><td colspan="2"></td><td>8,694</td><td>1,868</td><td>1,870</td><td>12,432</td></tr></table>

![](images/37ef27d650cf5c0c67a8b847c9cde61526e472394894d6b64969b22d67acb750.jpg)  
Figure 4: Per-class image counts in BD-PlantDX across the training, validation and test partitions. The maximum imbalance ratio between the largest and the smallest class is 1.16.

## 4.6. Comparison with Existing Collections

Table 3 positions BD-PlantDX against the public collections most closely related to it. PlantVillage remains the largest, however its imagery is laboratory acquired and covers no crop native to the target cropping system. The two Bangladeshi releases of Hasan et al. [17] and Hasan et al. [16] are region appropriate, yet the first covers foliar presentations only and the second is restricted to radish. BD-PlantDX is, to the best of our knowledge, the first collection to combine multi-crop coverage of this cropping system, both foliar and root presentations, four distinct disease etiologies and documented per-image expert validation.

## 5. Methodology

Section 2 described the mechanisms on which AgroVisNet rests in words; this section states them formally, derives the network block by block against the architecture of Figure 5, and fixes the training and evaluation

Table 3: BD-PlantDX against related public collections.
<table><tr><td>Dataset</td><td>Region</td><td>Images</td><td>Classes</td><td>Crops</td><td>Organs</td><td>Expert validated</td></tr><tr><td>PlantVillage [18]</td><td>Not region specific</td><td>54,305</td><td>38</td><td>14</td><td>Leaf</td><td>Not reported</td></tr><tr><td>VegNet-BD [17]</td><td>Bangladesh</td><td>12,786</td><td>21</td><td>6</td><td>Leaf</td><td>Not reported</td></tr><tr><td>RadishLeaf-BD [16]</td><td>Bangladesh</td><td>2,801</td><td>5</td><td>1</td><td>Leaf</td><td>Not reported</td></tr><tr><td>Qin et al. [44]</td><td>China</td><td>15,207</td><td>201</td><td>201</td><td>Leaf</td><td>Not reported</td></tr><tr><td>BD-PlantDX (ours)</td><td>Bangladesh</td><td>12,432</td><td>12</td><td>3</td><td>Leaf and root</td><td>Yes, per image</td></tr></table>

protocol. Every symbol introduced here is collected in Table 4, which is self-contained and carries the full notation of the paper.

## 5.1. Problem Formulation

Let $\mathcal { D } = \{ ( \mathbf { X } _ { i } , \mathbf { y } _ { i } ) \} _ { i = 1 } ^ { N }$ denote a labeled collection in which $\mathbf { X } _ { i } \in \mathbb { R } ^ { H \times W \times 3 }$ is an RGB image of a leaf or a root and $\mathbf { y } _ { i } \in \{ 0 , 1 \} ^ { K }$ is a one-hot encoding over the $K = 1 2$ classes of BD-PlantDX. We seek parameters $\pmb \theta$ of a network $f _ { \theta }$ that maps an image onto the probability simplex $\Delta ^ { K - 1 } = \{ \mathbf { p } \in \mathbb { R } ^ { K } : p _ { c } \geq 0 , \sum _ { c } p _ { c } = 1 \}$ and minimizes the expected categorical cross-entropy over the data distribution. The design constraint distinguishing this work from the standard formulation is a budget on $| \pmb \theta |$ : the model must remain below 0.3M trainable parameters while matching backbones one to two orders of magnitude larger.

## 5.2. Preprocessing and Augmentation

Images are decoded, resized to 224 × 224 by bilinear interpolation and rescaled to unit range,

$$
\widetilde { \mathbf { X } } = \frac { 1 } { 2 5 5 } \mathbf { X } .\tag{1}
$$

Augmentation is embedded inside the network graph as a stochastic operator A<sub>τ</sub> that is active during training and reduces to the identity at inference,

$$
\widehat { \mathbf { X } } = \left\{ \begin{array} { l l } { \mathcal { A } _ { \tau } ( \widetilde { \mathbf { X } } ) , } & { \mathrm { t r a i n i n g } , } \\ { \widetilde { \mathbf { X } } , } & { \mathrm { i n f e r e n c e } . } \end{array} \right.\tag{2}
$$

The operator composes random rotation within $\pm 1 5 ^ { \circ }$ , random zoom within ±10%, random translation within ±10% of each spatial extent and random contrast jitter of ±10%, each with reflective boundary fill and a fixed seed. Horizontal and vertical flips are deliberately excluded: lesion distribution on a leaf is not mirror symmetric with respect to the petiole, and flipping was found to degrade the separation of the two radish healthy presentations. Embedding $\mathbf { \mathcal { A } } _ { \tau }$ in the graph rather than in the input pipeline guarantees that the deployed model and the trained model share one preprocessing definition.

## 5.3. Architecture Overview

Figure 5 presents AgroVisNet in full. The network comprises a preprocessing front end, a strided stem, four feature stages and a dual-pooling classification head. Each feature stage $l \in \{ 1 , 2 , 3 , 4 \}$ pairs two Enhanced Residual Blocks (ERB) with one Multi-Scale Depthwise Block (MSDB), and is characterized by a channel width $F _ { l }$ and a group count $g _ { l }$ with $( F _ { l } , g _ { l } )$ taking the values (32, 2), (64, 4), (96, 8) and (128, 16). The group count is doubled together with the channel width at every stage, which holds the cost of the spatial convolution approximately constant as the representation deepens. Spatial resolution follows $2 2 4 \to 5 6 \to 2 8 \to 1 4 \to 1 4$ , with downsampling performed by the stem and by the max-pooling layers that terminate stages one and two; stages three and four operate at a fixed 14 × 14 grid, which preserves the spatial detail on which lesion margin discrimination depends. The complete network holds 296,780 parameters, of which 290,572 are trainable and 6,208 are the non-trainable running statistics of the batch normalization layers.

![](images/d86cbe0ec67f560440fb907c5ef6584daeba99c45fcf284ca5517f9964a184ff.jpg)  
Figure 5: Architecture of AgroVisNet. The upper row traces the forward path from preprocessing through the stem, four feature stages and the dual-pooling classification head, annotating each stage with its channel width $F _ { l } ,$ group count g<sub>l</sub> and output resolution. The lower row expands the two recurring blocks: the Enhanced Residual Block, which applies a grouped bottleneck followed by sequential channel and spatial attention, and the Multi-Scale Depthwise Block, which fuses 3 × 3 and $5 \times 5$ depthwise responses.

## 5.4. Stem

The stem reduces the input resolution by a factor of four before any residual computation, which removes the dominant cost of full-resolution processing. It applies a strided $5 \times 5$ convolution with 32 filters, batch normalization and swish activation, followed by strided max pooling,

$$
{ \bf U } ^ { ( 1 ) } = \mathrm { M a x P o o l } _ { 3 , 2 } \left( \delta \left( \mathrm { B N } \left( { \bf W } _ { \mathrm { s t e m } } \circledast _ { 5 , 2 } \widehat { \bf X } \right) \right) \right) ,\tag{3}
$$

where $\circledast _ { k , s }$ denotes convolution with kernel size k and stride $s , \operatorname { M a x P o o l } _ { k , s }$ denotes max pooling with window k and stride $s ,$ and $\mathbf { U } ^ { ( 1 ) } \in \mathbb { R } ^ { 5 6 \times 5 6 \times 3 2 }$ $\mathrm { ~ A ~ 5 ~ } \times 5$ kernel is preferred over the more common $3 \times 3$ at this depth since the receptive field it establishes covers a full lesion nucleus at the native scale of the imagery.

## 5.5. Enhanced Residual Block

The ERB is the primary representational unit and corresponds to the orange panel of Figure 5. It realizes the residual bottleneck of Section 2 with grouped spatial convolution and two attention gates. Given an input U with C channels and a target width $F ,$ the block first contracts to $F / 2$ channels with a pointwise convolution, applies the spatial convolution in the contracted space with g groups, and expands back to $F$ channels,

$$
\begin{array} { r } { \mathbf { Z } ^ { ( 1 ) } = \delta ( \mathrm { B N } ( \mathbf { W } _ { r 1 } \circledast _ { 1 , 1 } \mathbf { U } ) ) , } \end{array}\tag{4}
$$

$$
\begin{array} { r } { \mathbf { Z } ^ { ( 2 ) } = \delta \Big ( \mathrm { B N } \Big ( \mathbf { W } _ { r 2 } \circledast _ { 3 , g } \mathbf { Z } ^ { ( 1 ) } \Big ) \Big ) , } \end{array}\tag{5}
$$

$$
\begin{array} { r } { \mathbf { Z } ^ { ( 3 ) } = \mathrm { B N } \Big ( \mathbf { W } _ { r 3 } \circledast _ { 1 , 1 } \mathbf { Z } ^ { ( 2 ) } \Big ) , } \end{array}\tag{6}
$$

with $\mathbf { W } _ { r 1 } \ \in \ \mathbb { R } ^ { 1 \times 1 \times C \times F / 2 } , \ \mathbf { W } _ { r 2 } \ \in \ \mathbb { R } ^ { 3 \times 3 \times ( F / 2 g ) \times F / 2 }$ and $\mathbf { W } _ { r 3 } \in \mathbb { R } ^ { 1 \times 1 \times F / 2 \times F }$ The joint efect of the bottleneck and of grouping is multiplicative. A dense $3 \times 3$ convolution at width $F$ carries $9 F ^ { 2 }$ weights, whereas Eq. (5) carries

$$
\Omega ( \mathbf { W } _ { r 2 } ) = \frac { 9 F ^ { 2 } } { 4 g } ,\tag{7}
$$

a reduction of $4 g ,$ which reaches a factor of 64 in the final stage where $g = 1 6$ . This single design choice accounts for the majority of the parameter economy of the network. The expanded tensor is then gated along the channel axis. Two global descriptors are formed by average and maximum pooling, both are passed through a perceptron whose weights are shared across the two paths, and the summed responses are squashed to (0, 1),

$$
{ \bf z } _ { \mathrm { a v g } } = \mathrm { G A P } \left( { \bf Z } ^ { ( 3 ) } \right) , \quad { \bf z } _ { \mathrm { m a x } } = \mathrm { G M P } \left( { \bf Z } ^ { ( 3 ) } \right) ,\tag{8}
$$

$$
\mathbf { M } _ { c } = \sigma \big ( \mathbf { W } _ { 1 } ^ { \top } \boldsymbol { \rho } \big ( \mathbf { W } _ { 0 } ^ { \top } \mathbf { z } _ { \mathrm { a v g } } \big ) + \mathbf { W } _ { 1 } ^ { \top } \boldsymbol { \rho } \big ( \mathbf { W } _ { 0 } ^ { \top } \mathbf { z } _ { \mathrm { m a x } } \big ) \big ) ,\tag{9}
$$

$$
\mathbf { Z } ^ { ( 4 ) } = \mathbf { M } _ { c } \odot \mathbf { Z } ^ { ( 3 ) } ,\tag{10}
$$

where GAP and GMP reduce a tensor over its spatial axes, $\mathbf { W } _ { 0 } \in \mathbb { R } ^ { F \times F / r }$ and $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { F / r \times F }$ with reduction ratio $r = 1 6$ , both without bias, and $\odot$ broadcasts the per-channel multiplier across space. Weight sharing across the two pooling paths halves the parameter cost of the module relative to independent branches, and the two descriptors are complementary: average pooling reports the mean activation of a filter over the organ, while maximum pooling reports its peak response, which is the more informative statistic when a lesion occupies a small fraction of the frame. A spatial gate follows. A single $7 \times 7$ convolution collapses the gated tensor to one channel and produces a per-location multiplier,

$$
\mathbf { M } _ { s } = \sigma \Big ( \mathbf { W } _ { s } \circledast _ { 7 , 1 } \mathbf { Z } ^ { ( 4 ) } \Big ) ,\tag{11}
$$

$$
\mathbf { Z } ^ { ( 5 ) } = \mathbf { M } _ { s } \odot \mathbf { Z } ^ { ( 4 ) } .\tag{12}
$$

The large kernel is intentional: the gate must integrate context well beyond a lesion boundary in order to separate a genuine necrotic region from a shadow or a soil speckle of comparable local appearance. Ordering the two gates channel first and spatial second follows the convolutional block attention module [68], in which this order is reported to outperform the reverse and the parallel arrangement. The block closes with the residual addition,

$$
\mathbf { Y } = \delta \Big ( \mathbf { Z } ^ { ( 5 ) } + \mathcal { S } ( \mathbf { U } ) \Big ) ,\tag{13}
$$

where $s$ is the identity when the input and output widths agree and a batch-normalized $1 \times 1$ projection otherwise. Placing the activation after the addition rather than before it preserves an unmodified gradient path from the block output to the block input.

## 5.6. Multi-Scale Depthwise Block

The MSDB, shown in the purple panel of Figure 5, terminates every stage and supplies scale diversity that the fixed $3 \times 3$ kernel of the ERB cannot provide on its own. Lesion size varies substantially within a single class, from the pinhole punctures of flea beetle damage to the confluent necrotic sectors of late blight, and a single receptive field resolves only one end of that range. The block expands the input pointwise, applies two depthwise convolutions of diferent kernel size in parallel, concatenates their responses and projects back,

$$
\begin{array} { r } { \mathbf { P } = \delta ( \mathrm { B N } ( \mathbf { W } _ { p } \circledast _ { 1 , 1 } \mathbf { Y } ) ) , } \end{array}\tag{14}
$$

$$
\begin{array} { r } { { \bf D } _ { 3 } = \delta \left( \mathrm { B N } \left( { \bf W } _ { d 3 } \circledast _ { 3 } ^ { \mathrm { d w } } { \bf P } \right) \right) , } \end{array}\tag{15}
$$

$$
\mathbf { D } _ { 5 } = \delta \big ( \mathrm { B N } \big ( \mathbf { W } _ { d 5 } \circledast _ { 5 } ^ { \mathrm { d w } } \mathbf { P } \big ) \big ) ,\tag{16}
$$

$$
\begin{array} { r } { \mathbf { Q } = \delta ( \mathrm { B N } ( \mathbf { W } _ { q } \circledast _ { 1 , 1 } \left[ \mathbf { D } _ { 3 } ; \mathbf { D } _ { 5 } \right] ) ) , } \end{array}\tag{17}
$$

$$
\begin{array} { r } { \mathbf { U } ^ { ( l + 1 ) } = \mathrm { S p a t i a l D r o p o u t } ( \mathbf { Q } , p _ { s } ) , } \end{array}\tag{18}
$$

where $[ \cdot ; \cdot ]$ denotes concatenation along the channel axis and $\mathbf { \Phi } _ { ( * ) } ^ { \mathrm { d w } }$ a depthwise convolution that filters each channel independently. The two depthwise branches together carry only $( 9 + 2 5 ) F$ weights, which is negligible against the ${ \bar { \boldsymbol { F } } } ^ { 2 }$ scaling of the pointwise projections. Spatial dropout is applied at a stage-dependent rate $p _ { s } \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 \}$ , increasing with depth, and drops whole feature maps for the reason given in Section 2 [61].

## 5.7. Dual-Pooling Classification Head

The head collapses the final 14×14×128 tensor through two pooling operators in parallel and concatenates the results,

$$
\mathbf { v } = \Big [ \mathrm { G M P } \Big ( \mathbf { U } ^ { ( 5 ) } \Big ) ; \mathrm { G A P } \Big ( \mathbf { U } ^ { ( 5 ) } \Big ) \Big ] \in \mathbb { R } ^ { 2 5 6 } .\tag{19}
$$

Global average pooling summarizes the prevalence of a pattern over the whole organ, which suits difuse conditions such as nutrient deficiency, while global maximum pooling retains the strongest local evidence, which suits focal conditions such as scab. Retaining both removes the need to select between the two failure modes. Classification proceeds through two regularized dense layers and a softmax,

$$
\begin{array} { r } { \mathbf { h } _ { 1 } = \mathrm { B N } \big ( \delta \big ( \mathbf { W } _ { f 1 } ^ { \top } \mathrm { D r o p o u t } ( \mathbf { v } , p _ { d } ) \big ) \big ) , } \end{array}\tag{20}
$$

$$
\begin{array} { r } { \mathbf { h } _ { 2 } = \mathrm { B N } \big ( \delta \big ( \mathbf { W } _ { f 2 } ^ { \top } \mathrm { D r o p o u t } ( \mathbf { h } _ { 1 } , p _ { d } ) \big ) \big ) , } \end{array}\tag{21}
$$

$$
\widehat { \mathbf { y } } = \mathrm { s o f t m a x } \big ( \mathbf { W } _ { o } ^ { \top } \mathbf { h } _ { 2 } \big ) ,\tag{22}
$$

with $\mathbf { W } _ { f 1 } \ \in \ \mathbb { R } ^ { 2 5 6 \times 1 2 8 }$ $\mathbf { W } _ { f 2 } \ \in \ \mathbb { R } ^ { 1 2 8 \times 6 4 } , \ \mathbf { W } _ { o } \ \in \ \mathbb { R } ^ { 6 4 \times K }$ and $p _ { d } \ = \ 0 . 3 .$ . The head accounts for 41,536 parameters, which is 14.3% of the trainable total.

## 5.8. Objective and Optimization

Training minimizes categorical cross-entropy with an $\ell _ { 2 }$ penalty applied to the convolutional and dense kernels,

$$
\mathcal { L } ( \pmb { \theta } ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { K } y _ { i , c } \log \widehat { y } _ { i , c } + \lambda \sum _ { \mathbf { W } \in \pmb { \theta } _ { \mathrm { k e r } } } \Vert \mathbf { W } \Vert _ { F } ^ { 2 } ,\tag{23}
$$

with $\lambda = 1 0 ^ { - 5 }$ and $\theta _ { \mathrm { k e r } } \subset \theta$ the set of convolutional and dense kernels, which excludes the batch normalization scales and ofsets. Parameters are updated by stochastic gradient descent with Nesterov momentum,

$$
\begin{array} { r } { \mathbf { m } _ { t } = \mu \mathbf { m } _ { t - 1 } + \nabla _ { \theta } \mathcal { L } \big ( \theta _ { t - 1 } - \eta _ { t } \mu \mathbf { m } _ { t - 1 } \big ) , } \end{array}\tag{24}
$$

$$
\pmb { \theta } _ { t } = \pmb { \theta } _ { t - 1 } - \eta _ { t } \mathbf { m } _ { t } ,\tag{25}
$$

with $\mu = 0 . 9$ and an initial rate $\eta _ { 0 } = 1 0 ^ { - 2 }$ . Momentum-based descent is preferred over adaptive methods here since the network is trained from random initialization rather than fine-tuned, a regime in which the flatter minima associated with plain momentum are reported to generalize better. The learning rate is halved whenever the training loss fails to improve for five consecutive epochs, with a floor of $1 0 ^ { - 8 }$ , and optimization stops when the training loss stalls for thirty consecutive epochs, at which point the weights of the best epoch are restored. The maximum budget is $E = 3 0 0$ epochs at a batch size of $B = 3 2$ ; no run reached the cap, with convergence occurring between epoch 80 and epoch 157 across all configurations. Algorithm 1 states the complete forward pass.

## 5.9. Baselines

Six ImageNet-pretrained lightweight backbones serve as baselines: EficientFormerV2-S0 [31], RepViT-M1.0 [66], ConvNeXt-Atto [32], FastViT-T8 [63], MobileNetV4-Conv-Small [43] and GhostNetV2-1.0 [59]. The selection spans the three design lineages that currently define the sub-five-million parameter regime, namely the purely convolutional stack represented by ConvNeXt-Atto, MobileNetV4-Conv-Small and GhostNetV2-1.0, the reparameterized hybrid represented by RepViT-M1.0 and FastViT-T8, and the eficient-attention transformer represented by EficientFormerV2-S0, so that the comparison is not confined to a single architectural family. All six are instantiated from the ImageNet-1k checkpoints distributed with the timm library. Every baseline consumes the identical data pipeline, the identical split and the identical resolution, and each is fine-tuned end to end with a freshly initialized classification head using AdamW at a learning rate of $5 \times 1 0 ^ { - 5 }$ with weight decay $1 0 ^ { - 3 }$ , head dropout 0.3 and stochastic depth 0.1, for at most 100 epochs at batch size 32 with early stopping after 15 stagnant epochs and the same halving schedule on the validation loss. The comparison therefore isolates architecture and pretraining, holding data and protocol fixed.

Algorithm 1 AgroVisNet forward pass   
Require: Image $\overline { { \textbf { X } \in \mathbb { R } ^ { 2 2 4 \times 2 2 4 \times 3 } ; } }$ widths $F \ = \ ( 3 2 , 6 4 , 9 6 , 1 2 8 )$ ; groups $g \ = \ ( 2 , 4 , 8 , 1 6 )$ ; rates $\begin{array} { r l } { p _ { s } } & { { } = } \end{array}$   
$( 0 . 0 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 )$   
Ensure: Class distribution $\widehat { \mathbf { y } } \in \Delta ^ { K - 1 }$   
1: $\widehat { \mathbf { X } } \gets \mathcal { A } _ { \tau } ( \mathbf { X } / 2 5 5 )$ ▷ Eq. (1), (2)   
2: $\mathbf { U } \gets \mathrm { M a x P o o l } _ { 3 , 2 } ( \delta ( \mathrm { B N } ( \mathbf { W } _ { \mathrm { s t e m } } \circledast _ { 5 , 2 } \widehat { \mathbf { X } } ) ) )$   
3: for l = 1 to 4 do   
4: $\mathbf { U } \gets \mathrm { E R B } ( \mathbf { U } , F _ { l } , g _ { l } , \mathrm { p r o j e c t i o n } )$ ▷ Eq. (4)–(13)   
5: $\mathbf { U } \gets \mathrm { E R B } ( \mathbf { U } , F _ { l } , g _ { l } , \mathrm { i d e n t i t y } )$   
6: $\mathbf { U } \gets \mathrm { M S D B } ( \mathbf { U } , F _ { l } , p _ { s , l } )$ ▷ Eq. (14)–(18)   
7: if $l \leq 2$ then   
8: $\mathbf { U } \gets \mathrm { M a x P o o l } _ { 2 , 2 } ( \mathbf { U } )$   
9: end if   
10: end for   
11: $\mathbf { v } \gets [ \mathrm { G M P } ( \mathbf { U } ) ; \mathrm { G A P } ( \mathbf { U } ) ]$ ▷ Eq. (19)   
12: return softmax $( \mathbf { W } _ { o } ^ { \top } \mathbf { h } _ { 2 } ( \mathbf { v } ) )$ ▷ Eq. (20)–(22)

## 5.10. Evaluation Protocol

We report accuracy together with weighted precision, weighted recall and weighted F1, all computed on the held-out test split after model selection on the validation split. Four experiments follow. The seed study repeats the full training procedure at seeds 42 through 46 and reports the mean and the sample standard deviation; the six pretrained baselines are additionally retrained at these same five seeds, and the significance of the accuracy margin is assessed by a two-sided Wilcoxon signed-rank test on the five paired per-seed diferences against each baseline, complemented by a paired Student t-test, with both families of p-values corrected for multiplicity by the Holm step-down procedure [20, 67]. The component ablation trains eleven variants that each remove one element of the architecture or of the regularization while holding the seed, the split and the schedule fixed, and reports the resulting accuracy change in percentage points. The crossdataset study retrains the unmodified architecture on two independently collected datasets, adjusting only the width of the output layer. Interpretability is assessed with Grad-CAM applied to the last convolutional tensor, whose k-th activation map is denoted $\mathbf { A } _ { k } \in \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } }$ ; the importance of that map for class c is the spatially averaged gradient of the class score, and the localization map is the rectified weighted sum of the maps,

$$
\alpha _ { k } ^ { c } = \frac { 1 } { H ^ { \prime } W ^ { \prime } } \sum _ { u } \sum _ { w } \frac { \partial \widehat { y } _ { c } } { \partial A _ { u , w , k } } , \qquad \mathbf { L } ^ { c } = \rho \left( \sum _ { k } \alpha _ { k } ^ { c } \mathbf { A } _ { k } \right) .\tag{26}
$$

All experiments were run in TensorFlow 2 with Keras 3.13 on a single NVIDIA P100 accelerator, and the baseline runs additionally used PyTorch with the timm library.

## 6. Experimental Results

## 6.1. Classification Performance on BD-PlantDX

Figure 6 traces optimization on BD-PlantDX. Training and validation accuracy separate by less than 0.05 percentage points from epoch 100 onward, and the validation loss tracks the training loss without the upward inflection that marks overfitting, which indicates that the combined efect of spatial dropout, dense dropout, weight decay and embedded augmentation is suficient at this parameter budget. The best validation epoch is 102 of 145, at which point training accuracy stands at 99.78% and validation accuracy at 99.73%.

Table 5 reports the per-class test performance and Figure 7 presents the same quantities graphically. AgroVisNet attains 99.52% test accuracy, 99.53% weighted precision, 99.52% weighted recall and 99.52% weighted F1 over the 1,870 held-out images, with a macro F1 of 99.52% that confirms the aggregate figure is not carried by the larger classes. Five of the twelve classes are classified without error, namely Potato Healthy, Radish Root Healthy, Potato Nutrient Deficiency, Gourd Downy Mildew and Radish Alternaria brassicae. The lowest per-class F1 values are 97.75% for Radish Flea Beetle Damage and 98.39% for Radish Leaf Healthy, and precision on Radish Flea Beetle Damage falls to 95.60% while its recall remains at 100%.

![](images/a9a2d2c0284e1114bae3b059e1a5a4103827ef53d4ee1905bda581d11fadeab0.jpg)

![](images/a3d8ee7c0ed0b38ce06d103c1845f4fa1e74dc5d10ff85054313f76545f55d81.jpg)  
Figure 6: Training and validation trajectories of AgroVisNet on BD-PlantDX. Left: accuracy. Right: categorical cross-entropy on a logarithmic scale.

![](images/e1407c68734b71580423400371bf18a7d21688a60b5f1f1d44dc71abfd988792.jpg)  
Figure 7: Per-class precision, recall and F1 of AgroVisNet on the BD-PlantDX test split.

Figure 8 localizes the residual error. Nine of 1,870 test images are misclassified. Seven of the nine fall into the Radish Flea Beetle Damage class, comprising four Radish Leaf Healthy images, two Potato Mosaic images and one Gourd Healthy image; the remaining two are one Potato Late Blight image assigned to Radish Leaf Healthy and one Radish White Mold image assigned to Radish Scab. The dominant confusion is therefore directional and interpretable: flea beetle damage manifests as sparse pinhole punctures of one to two millimeters, and a healthy leaf carrying incidental mechanical damage or a small necrotic fleck presents nearly the same signature at 224 × 224 resolution. This failure mode is a resolution limitation rather than a representational one, and higher acquisition resolution over the afected region is the natural remedy.

## 6.2. Comparison with Pretrained Lightweight Backbones

Table 6 compares AgroVisNet against six ImageNet-pretrained backbones trained under an identical protocol, and Figures 9 and 10 present the accuracy against the parameter budget and the full metric set respectively. AgroVisNet reaches 99.52% accuracy at 0.29M trainable parameters. The strongest baseline is EficientFormerV2-S0 at 99.25% with 3.25M parameters, followed by RepViT-M1.0 at 99.20% with 4.72M, ConvNeXt-Atto at 99.14% with 3.38M, FastViT-T8 at 99.04% with 3.27M, MobileNetV4-Conv-Small at 98.98% with 2.51M, and GhostNetV2-1.0 at 98.93% with 4.89M. The proposed model therefore exceeds the best baseline by 0.27 percentage points while using 11.2 times fewer parameters, and exceeds the smallest baseline by 0.54 percentage points while using 8.7 times fewer parameters; against GhostNetV2-1.0 the parameter ratio reaches 16.8.

![](images/0d092252543d0af0adc565b8829d33d0c8c77f168b962fe009e64246a1ed4159.jpg)  
Figure 8: Confusion matrix of AgroVisNet on the BD-PlantDX test split. Nine of 1,870 images are misclassified, seven of them into the Radish Flea Beetle Damage class.

An ensemble of four fully fine-tuned heavyweight backbones provides a further reference point. Averaging the softmax outputs of VGG16, ResNet50, InceptionV3 and MobileNetV2 yields 99.36% accuracy at a macro-averaged one-versus-rest area under the receiver operating characteristic curve of 0.9973, matching the single strongest lightweight baseline while requiring over 250M parameters and four separate forward passes. AgroVisNet exceeds that ensemble by 0.16 percentage points with a single forward pass through 0.29M parameters, which supports the position that architectural fit to the target distribution matters more than raw capacity on this class of problem.

## 6.3. Comparison with Published Plant-Disease Methods

The pretrained backbones above are general-purpose designs adapted to the task; a complementary question is how AgroVisNet compares against architectures published specifically for plant-disease classification. We reproduce two recent methods: Mob-Res, a MobileNetV2 feature extractor augmented with residual blocks [38], and EDL10, a soft-voting ensemble of four convolutional networks trained with the Adam, SGD, RMSprop and Adamax optimizers [25]. Both were retrained on BD-PlantDX under their own originally published hyperparameters rather than the protocol of Section 5, so that each method is evaluated with the recipe its authors tuned, and both were assessed on the same 1,870-image test split (Table 7). EDL10’s reported parameter count is the total across its four sub-models, and the AgroVisNet accuracy and F1 figures repeat Table 6.

Mob-Res reaches 99.20% accuracy and F1 at 3.47M parameters and EDL10 reaches 96.52% accuracy at 8.49M parameters across its four sub-models, against AgroVisNet’s 99.52% at 0.29M. The proposed model is therefore 0.32 and 3.00 percentage points more accurate while using 12 and 29 times fewer parameters than the two published designs, and the ensemble in particular incurs four forward passes for the lowest accuracy of the three. The headline figure each method reports on its own dataset is listed for context but is not comparable across rows, because the datasets and class inventories difer; the informative comparison is the reproduced column, measured on a common dataset and test split, which reinforces the pattern established against the pretrained backbones, that a compact architecture matched to the target distribution is more accurate and far smaller than heavier general-purpose or ensemble alternatives.

![](images/b4f4dcb781cec6934c3e2257585b7168553684cc3a68fbce08c5203d3e56e3de.jpg)

Figure 9: Test accuracy against trainable parameter count on BD-PlantDX.  
![](images/9aa7665a1dacd097e022491a449353c843a16f8be2c27d1822bbc5943c61c230.jpg)  
Figure 10: Accuracy, precision, recall and F1 of AgroVisNet and the six pretrained baselines on BD-PlantDX, annotated with the trainable parameter count of each model.

## 6.4. Deployment and Computational Eficiency

Parameter count alone does not determine whether a model is deployable; the arithmetic executed per image and the memory the model occupies matter as much on the low-cost hardware that is the intended deployment target. Table 8 therefore reports the multiply–accumulate (MAC) cost, the single-precision ondisk size and the test accuracy of AgroVisNet against the six pretrained backbones at 224 × 224 resolution. MAC counts are measured directly, single-precision size follows the parameter count at four bytes per weight, and the parameter and accuracy columns repeat Table 6; of AgroVisNet’s 296,780 total parameters, 290,572 are trainable. The final row of Table 8 reports the baseline-to-AgroVisNetratio from the least to the most demanding baseline on each axis, so that a larger value favours AgroVisNet. Because wall-clock latency is a property of the measurement platform rather than of the architecture, the MAC count serves as the hardware-independent compute proxy across models, and measured latency is reported for AgroVisNet alone.

At 132.70M MACs, AgroVisNet is the cheapest model in the comparison, from 1.3 times below the next-lightest baseline, GhostNetV2-1.0, to 8.5 times below the heaviest, RepViT-M1.0; its single forward pass costs 265.40M floating-point operations, twice the MAC count. It is also the smallest on disk at 1.13 MB, against 9.70 to 24.56 MB for the baselines, a direct consequence of a parameter budget that is 8.7 to 16.8 times below the 2.51 to 4.89M parameters of the pretrained backbones. AgroVisNet is thus at once the most accurate model in the comparison and the least expensive to evaluate, so that on this dataset no competing model is simultaneously more accurate and cheaper to run.

![](images/6578c3fc12a868fa556d8256d2beeb68abeae191fc45f9b87d90da5573a5bc81.jpg)

![](images/914bb45a5afce0806fbdfd9a1ff5d161e0c65f23882d520d3311e0877fcedab3.jpg)  
Figure 11: Export and quantization behaviour of AgroVisNet. (a) On-disk size across export formats, annotated with test accuracy. (b) Single-image CPU latency of the TensorFlow Lite formats through the XNNPACK delegate.

Table 9 and Figure 11 report how the model behaves once exported for on-device use. The single-precision TensorFlow Lite export reproduces the Keras accuracy exactly at 1.17 MB; dynamic-range quantization preserves accuracy to the fourth decimal while reducing the model to 0.43 MB; and full-integer INT8 quantization retains accuracy to within 0.22 percentage points at 0.46 MB, small enough to sit in the flash budget of a low-cost microcontroller. The INT8 model is additionally the fastest of the three at 8.40 ms per image, below both the 10.44 ms of the single-precision export and the 9.00 ms single-image latency measured on a GPU, because integer kernels map cleanly onto the vector units of a commodity CPU; the dynamic-range variant is the slowest at 15.12 ms, as its weights are dequantised on the fly. Taken together, these measurements turn the parameter budget into a concrete deployment envelope: a sub-half-megabyte model that classifies an image in under ten milliseconds on a CPU with no dedicated accelerator.

Compute cost and quantized size predict deployability; a direct measurement confirms it. We exported AgroVisNet to TensorFlow Lite and benchmarked all three formats on a mid-range Android handset (realme RMX3853, Qualcomm Snapdragon 7+ Gen 3, 12 GB RAM, Android 16) with the benchmark\_model tool pinned to the five performance cores, over 100 runs per measurement and three repeats. Each configuration was run for 100 measurements after 20 warmup runs, with the standard deviation across the three repeats remaining below 0.9 ms in every case; the GPU delegate was also requested but fell back to CPU execution on this device and is therefore omitted from the comparison. Table 10 and Figure 12 report single-image latency across the XNNPACK CPU delegate at one, two and four threads and the Android Neural Networks API (NNAPI). The full-integer INT8 model is the fastest configuration at every thread count, reaching 2.33 ms per image at four threads, approximately 429 images per second, against 4.38 ms for the dynamic-range model and 5.47 ms for the single-precision export. Latency scales cleanly with the thread count, the INT8 model falling from 6.02 ms at one thread to 3.47 ms at two and 2.33 ms at four, a 2.6-fold reduction across the four-core span, while NNAPI improves on none of the four-thread XNNPACK paths and reads 5.85 ms for INT8 as the delegate falls back to CPU execution, so the CPU delegate is the operative deployment path on this system-on-chip. The 2.33 ms on-device INT8 latency is below the 8.40 ms desktop-CPU figure of Table 9, because integer kernels map directly onto the mobile vector units, and it places recognition far above video frame rate on a handset with no dedicated accelerator. Together, these measurements substantiate the deployability claim in the paper’s title: the model is not merely small in parameter count but cheap to compute, sub-megabyte on disk, and real-time on a commodity smartphone, matching or exceeding the accuracy of backbones nearly an order of magnitude larger on every eficiency axis reported above.

## 6.5. Stability Across Random Seeds

Table 11 and Figure 13 report five independent repetitions of the full training procedure at seeds 42 through 46. Test accuracy averages 99.57% with a sample standard deviation of 0.10 percentage points and a range of 99.47% to 99.68%; weighted F1 tracks it at $9 9 . 5 7 \pm 0 . 1 0 \%$ . The lower bound of the observed range exceeds the accuracy of every pretrained baseline, which indicates that the margin reported in Table 6 is not an artifact of a favorable initialization. Convergence is less stable than accuracy: the best epoch varies from 50 to 110 and training time from 56.6 to 98.7 minutes, a spread driven by the training-loss-triggered learning rate schedule, which reaches its final rate after two halvings in some runs and four in others. Figure 14 shows the corresponding validation trajectories, all of which exceed 99% by epoch 60 despite difering sharply during the first 40 epochs.

![](images/6e9556d9349b0463b491247eaecc348d7479ded2e367329239a06a789e3271ef.jpg)

![](images/0d682dc348890d22649e744463641d97dd7b8c418ce52a28d7b06487def55193.jpg)

Figure 12: On-device deployment of AgroVisNet on the Android handset of Table 10. (a) Single-image latency across the XNNPACK CPU delegate and NNAPI, grouped by export format; error bars are the standard deviation across three repeats. (b) Throughput of the headline four-thread XNNPACK configuration.  
![](images/8cbc940bf600167bea0ef1e6cd784f4aeb8d37e8ee531786e068e03105429d67.jpg)

![](images/b12062ecb0dbf4aeb7bb136576caf42e35618cd825d97011ad8a49842ea3a1f9.jpg)  
Figure 13: Seed-to-seed stability of AgroVisNet on BD-PlantDX. Left: test accuracy per seed against the mean and the onestandard-deviation band. Right: mean and standard deviation of the four aggregate metrics.

## 6.6. Statistical Significance of the Margin over the Baselines

A margin of a few tenths of a percentage point invites the question of whether it reflects a genuine diference or a favourable draw of the single reference seed. To settle this, each of the six pretrained backbones was retrained under its own protocol at the same five seeds used for the seed study of Table 11, namely 42 through 46, and the test accuracy of AgroVisNet was paired with that of each baseline seed by seed. The five paired diferences per baseline were tested with a two-sided Wilcoxon signed-rank test, the nonparametric procedure standardly recommended for classifier comparison, and the six resulting p-values were corrected for multiplicity with the Holm step-down procedure [12, 20, 67]. A paired two-sided Studen t-test, Holm-corrected across the same six comparisons, is reported alongside as a parametric complement. Table 12 and Figure 15 report the outcome.

AgroVisNet is more accurate than every baseline on all five of the five seeds, with no ties, and the mean paired diference ranges from 0.33 percentage points against RepViT-M1.0 to 0.61 against MobileNetV4- Conv-Small. The 95% confidence interval on that diference excludes zero for all six baselines, the tightest being [0.31, 0.42] against ConvNeXt-Atto and the widest [0.14, 0.52] against RepViT-M1.0 (Figure 15). The paired t-test rejects the null of equal accuracy for all six after Holm correction, with adjusted p-values from $3 . 2 \times 1 0 ^ { - 4 }$ against ConvNeXt-Atto to 0.013 against EficientFormerV2-S0 and RepViT-M1.0, so the margin is significant at the 0.05 level against every backbone.

![](images/42ac05babff66d267127867b94a313961f6bd264250e52c3d49055185e936e06.jpg)  
Figure 14: Validation accuracy trajectories of the five seeds. All runs exceed 99% by epoch 60 despite difering convergence behavior over the first 40 epochs.

![](images/22ed8f1504990b90a0a8eba14ab9c372233971dc34959e62839c8357ea14a618.jpg)  
Figure 15: Paired per-seed test-accuracy diference between AgroVisNet and each pretrained backbone on BD-PlantDX, over the five seeds of Table 11. Each marker is the mean diference (AgroVisNet minus baseline) with its 95% CI whisker; the dashed line marks equality.

The Wilcoxon signed-rank test, by contrast, reaches significance against none of the six, and the reason is a property of the sample size rather than of the efect. With five nonzero paired diferences that all share a sign, the exact two-sided Wilcoxon p-value attains its smallest possible value, $2 / 2 ^ { 5 } = 0 . 0 6 2 5$ , which already exceeds 0.05 before any correction; every baseline therefore records a raw p of exactly 0.0625, which Holm inflates to 0.375. The rank test is thus at its resolution floor and cannot certify a diference at five seeds no matter how large or how consistent it is. The convergent evidence of a clean five-of-five sweep, six confidence intervals that exclude zero and a significant paired t-test establishes that the accuracy advantage of AgroVisNet over the pretrained backbones is reliable rather than an artifact of initialization; a larger seed budget is the only requirement for the rank test to reach the same conclusion.

## 6.7. Component Ablation

Table 13 and Figures 16 and 17 report ten single-component variants against the full-model baseline while holding the seed, the split and the schedule fixed. The ablation protocol re-seeds the entire stack before every variant and follows a single fixed trajectory, under which the full model reaches 99.04% rather than the 99.52% of the main run; all comparisons below are therefore made within the ablation protocol.

Substituting the rectified linear unit for swish is the single most damaging change at −0.27 percentage points, which is consistent with the smoothness of swish being useful at this depth. Removing both attention gates, the multi-scale depthwise block or the grouped convolutions each costs −0.16 percentage points, and removing either gate individually costs −0.11. The parameter accounting in Figure 17 is the more consequential reading. Removing the Multi-Scale Depthwise Block raises the parameter count from 296,780 to 465,420, an increase of 56.8%, while lowering accuracy; removing the grouped convolutions raises it to 418,316, an increase of 41.0%, again while lowering accuracy. The two mechanisms therefore contribute accuracy and parameter economy jointly rather than trading one against the other, which is the central design claim of the architecture. The two attention gates together cost 39,040 parameters, which is 13.2% of the total, for a combined 0.16 percentage point contribution.

![](images/e54a86e1c4b56f30f76e874cf7efce4646f25e3c0c472eb7be2383e0c8bef291.jpg)

Figure 16: Test accuracy of ten ablation variants and the full-model baseline. The dashed line marks the full model and each bar is annotated with its change in percentage points.  
![](images/ec5381689e9aa6b2cb4cbacb3f666971c287d11e4ba7b2b46301f509c5311f83.jpg)  
Figure 17: Parameter count and test accuracy of each ablation variant. Removing the multi-scale depthwise block (V4) or the grouped convolutions (V5) increases the parameter count while lowering accuracy.

Two variants improve on the full model within this protocol. Removing augmentation and removing weight decay each yield 99.20%, an increase of 0.16 percentage points, and removing augmentation additionally cuts training time from 108.5 to 32.2 minutes. We report this without adjustment. BD-PlantDX is class balanced and was acquired under a controlled illumination protocol, and on a test split drawn from that same acquisition process the regularizers restrict the fit at a small cost in clean accuracy. Both are retained in the released configuration since the deployment target is field imagery under illumination, viewpoint and background conditions absent from the current test split, and the field-condition evaluation listed among the outstanding items is the experiment that will settle this trade-of quantitatively.

## 6.8. Cross-Dataset Generalization

Table 14 and Figure 18 report the unmodified architecture retrained on two independently collected Bangladeshi datasets, with the output width as the only change. On VegNet-BD, comprising 12,786 images across 21 classes of six vegetable crops [17], AgroVisNet attains 98.71% accuracy and 98.71% weighted F1 at 291,157 parameters. On RadishLeaf-BD, comprising 2,801 images across five radish classes [16], it attains 99.05% accuracy and 99.05% weighted F1 at 290,117 parameters. Neither transfer required a change to the block structure, the widths, the group counts, the augmentation or the optimizer, which addresses the concern that a small architecture tuned on one collection encodes a dataset-specific inductive bias.

![](images/9f75830003eb8e04eb2f7b6cbd88d78476df44ccc284bd3f4b38e8f958c82df6.jpg)  
Figure 18: Accuracy, precision, recall and F1 of the unmodified AgroVisNet architecture across BD-PlantDX, VegNet-BD, RadishLeaf-BD and the merged 36-class corpus of all three.

Figure 19 decomposes the VegNet-BD result. Twelve of the 21 classes are classified without error, and the aggregate is limited by four bitter gourd classes: Fusarium wilt at 87.18% F1, mosaic virus at 88.76%, downy mildew at 98.84% and the healthy leaf at 98.80%. Fusarium wilt and mosaic virus both present as interveinal chlorosis on bitter gourd foliage and are separated in the field by root inspection rather than by leaf appearance, and the residual error concentrates where the visual evidence is genuinely ambiguous. Every class of the remaining five crops exceeds 98.9% F1. Figure 20 shows the RadishLeaf-BD confusion matrix, in which the only error is four Radish Black Leaf Spot images assigned to Radish Fresh Leaf out of 422 test images.

A separate-retraining protocol establishes that the architecture transfers to each collection in isolation but leaves open whether it scales to a single heterogeneous label space that spans all three crops at once. To settle this, the three datasets were merged into one corpus and the architecture retrained on it with, again, only the output width altered. The two classes common to BD-PlantDX and RadishLeaf-BD under matching definitions, the healthy radish leaf and the radish flea beetle damage, were unified under one label each, so that the union of the 12, 21 and 5 class inventories collapses to a 36-class taxonomy over 28,019 images. On the resulting 4,223-image stratified test split AgroVisNet attains 99.19% accuracy, 99.21% weighted precision and 99.19% weighted F1 at 292,132 parameters (Table 14, Figure 18), an increase of only 1,560 parameters, or 0.5%, over the BD-PlantDX configuration and attributable entirely to the wider softmax. Sixteen of the 36 classes are classified without error and 34 of the 4,223 test images are misclassified. The merged accuracy exceeds the model’s own standalone figures on VegNet-BD and RadishLeaf-BD and falls 0.33 percentage points short of the standalone BD-PlantDX result. That gap does not originate in the BD-PlantDX classes, which remain nearly intact at 99.46% accuracy within the merged model against 99.52% standalone; it is carried by a single pre-existing ambiguous pair from VegNet-BD, bitter gourd Fusarium wilt at 89.51% F1 with 12 of its 76 test images assigned to bitter gourd mosaic virus, the same confusion reported for that pair in the standalone VegNet-BD experiment where their F1 was 87.18% and 88.76% respectively. The architecture therefore scales to a substantially larger, heterogeneous multi-crop label space without any structural change, at a negligible parameter cost, and the accuracy it concedes is confined to genuinely ambiguous classes rather than spread across the taxonomy.

## 6.9. Robustness to Input Corruption

Field deployment exposes the model to capture conditions absent from the curated test split. To characterize this without new data collection, we applied five synthetic corruptions, Gaussian blur, additive noise, JPEG compression, and brightness decrease and increase, at five increasing severities to the test split and re-evaluated the frozen model; severity 0 is the uncorrupted image at 99.57% accuracy. Table 15 and Figure 21 report the sweep.For each corruption, the mean drop is computed as the clean accuracy minus the mean accuracy over severities 1 through 5, and Table 15 orders the five corruptions from the most to the least robust by this measure.

![](images/f6fe584100feef19be6cf5f1ce0849c677d95d2d454832d7823978efdc65a8b0.jpg)  
Figure 19: Per-class F1 of AgroVisNet on the 21 classes of VegNet-BD. Twelve classes are classified without error and the aggregate is limited by four bitter gourd classes.

The model is most robust to the two corruptions closest to its acquisition pipeline. JPEG compression and Gaussian blur cost only 7.72 and 9.18 percentage points of mean accuracy across the five severities and hold above 95% and 90% respectively through severity 4, collapsing only at the most extreme severity 5 (67.81% and 64.06%); this is expected, since the corpus is stored as JPEG and moderate blur leaves lesion texture intact. Robustness to additive noise and to illumination shift is markedly weaker. Additive noise holds to severity 2 (97.11%) but then falls of sharply, to 67.33% at severity 3 and 20.16% at severity 5. Increasing brightness is the single most damaging corruption, with additive noise and decreasing brightness following: increasing brightness costs accuracy from the first severity (90.21%) and reaches 23.85% at severity 5 for a 43.18-point mean drop, and decreasing brightness follows a similar trajectory to 23.21%. This asymmetric profile is the signature of the controlled-illumination acquisition protocol of Section 4: the model never saw strong sensor noise or large exposure changes during training, so these are the genuine out-of-distribution shifts, whereas compression and mild blur preserve the chromatic and textural cues the decision rests on and are tolerated. Test-time augmentation over brightness and noise, or a modest amount of the same augmentation at training time, is the natural mitigation and is deferred to the field-condition study.

## 6.10. Interpretability

Accuracy on a held-out split does not reveal which pixels a model uses, and because BD-PlantDX is acquired against a uniform background a recognizer that keyed on the background could score well and fail on field imagery, a failure mode that accuracy alone cannot detect (RQ5). We therefore combine qualitative attribution maps with a quantitative faithfulness analysis and a sanity check, organized below as method, qualitative attribution, its stage-wise refinement, a failure analysis, a faithfulness study and a model-sensitivity sanity check.

![](images/0b695b071bee240c5654b417510be6ecff4cdfb13f854aa9ee6c08bf54c09f71.jpg)  
Figure 20: Confusion matrix of AgroVisNet on the RadishLeaf-BD test split. Four of 422 images are misclassified.

![](images/6e7aa4a2273c469321bc51c8310e590d15ad21bbb3f5589740c05c9b1bd6251f.jpg)  
Figure 21: Test accuracy of AgroVisNet against corruption severity for five synthetic corruptions applied to the BD-PlantDX test split; severity 0 is the uncorrupted image.

(i) Method. We apply Grad-CAM [53] and its generalization Grad-CAM++ [9] to the last convolutional tensor, whose k-th activation map is $\mathbf { A } _ { k } .$ . Grad-CAM weights each map by the spatially averaged gradient of the class score and forms the localization map as the rectified weighted sum of $\operatorname { E q . }$ (26), which is min–max normalized and bilinearly upsampled to the 224×224 input grid. Grad-CAM++ replaces the uniform spatial average of that channel weight with a positive, pixel-wise weighting derived from higher-order derivatives of the class score, which sharpens localization when several disjoint regions support the same class. The two methods therefore probe the same evidence at diferent spatial granularity, and the faithfulness study of part (v) determines which is the more representative of how this backbone aggregates it.

(ii) Qualitative attribution. Figure 22 shows Grad-CAM and Grad-CAM++ maps for eight representative classes, each predicted with essentially full confidence. In every case the attribution peak falls on the leaf lamina, concentrating on the chlorotic, necrotic or mildew-covered tissue that carries the class signal and falling away over the uniform acquisition background. This is direct evidence that the model has not latched onto the controlled background as a class-correlated shortcut, the failure mode to which laboratory-acquired collections are most exposed, and it helps explain the cross-dataset transfer of Section 6.

![](images/663b9209723b4391bf00b01d04ebfaa34e9d3e21103d290e32dd9c240d219e33.jpg)

Figure 22: Grad-CAM and Grad-CAM++ localization maps of AgroVisNet for eight representative BD-PlantDX classes, with predicted-class probability.  
![](images/fb78edb05543879cfc5d52b9dc6174dde6ff1ebbfa00b3deeaef402cdb61e9c3.jpg)  
Figure 23: Stage-wise evolution of the Grad-CAM map for three classes.

Grad-CAM++ produces the more compact maps and Grad-CAM the broader ones; the faithfulness analysis of part (v) shows the broader maps to be the more representative of how this backbone aggregates evidence.

(iii) Where the attribution refines. Figure 23 traces the map through the four stages of the network. The first stage responds broadly along leaf contours and venation with little class preference; the second begins to separate the afected lamina from the background; and the last two stages consolidate the response onto the symptomatic region. The progression from a generic contour response to a single consolidated region indicates that the class-discriminative representation is built in the deeper stages rather than read of low-level edges.

(iv) Failure analysis. Figure 24 renders, for five misclassified test images, the Grad-CAM map for the true class alongside that for the predicted class. In all five the two maps fall on the same leaf, so the model localizes correctly and fails at discrimination rather than at attention. The errors are dominated by a single directional confusion: of the nine misclassified test images, seven are assigned to Radish Flea Beetle Damage, four of them from Radish Leaf Healthy and the remainder from Potato Mosaic and Gourd Healthy (Figure 8), because flea-beetle feeding manifests as sparse millimetre-scale pinholes that a healthy leaf carrying incidental mechanical damage closely mimics at 224 × 224 resolution. The model is confidently rather than marginally wrong: mean probability 0.89 on the predicted class against 0.09 on the true class, with five of the nine errors above 0.9 and four above 0.99 on the wrong class. The residual error is therefore a genuine visual near-degeneracy between the flea-beetle and healthy-leaf signatures rather than a backgroundattention failure, and higher acquisition resolution over the afected region, not stronger attention, is the appropriate remedy.

(v) Faithfulness. A plausible-looking map is not itself evidence of faithfulness. We therefore compute the deletion and insertion measures of Petsiuk et al. [41] over 200 test images, progressively removing pixels from or restoring them to a blurred baseline in order of decreasing attribution and recording the area under the resulting predicted-probability curve, together with the average confidence drop and increasein-confidence rate of Chattopadhay et al. [9] obtained by masking the input with the normalized map (Table 16, Figure 25). Lower average confidence drop, higher increase-in-confidence, and lower deletion AUC together with higher insertion AUC indicate a more faithful explanation, and no degenerate (uniform or empty) attribution maps were produced across the 200 evaluated images. On insertion, both methods recover the prediction faster than a random-ordering control (insertion AUC 0.932 and 0.925 against 0.902), so the highlighted region is suficient to support the decision. Grad-CAM is the more faithful of the two attribution methods on the masking metrics, with the lower average confidence drop (41.5% against 61.6%) and the higher increase-in-confidence rate, consistent with a compact backbone that aggregates evidence over the whole lamina rather than a few isolated points; all qualitative panels above therefore use Grad-CAM. The deletion measure is the less discriminating here: removing the highest-attribution pixels first does not collapse the prediction faster than random removal (deletion AUC 0.356 and 0.340 against 0.225 for the control), because the leaf fills most of the frame and the evidence is distributed across the lamina, so a compact high-attribution region can be excised while most of the discriminative texture remains. Insertion and the confidence-drop metric, which probe the suficiency of the highlighted region rather than the efect of a small compact deletion, are the more informative criteria on this leaf-dominated imagery, and both favour the attribution maps over the control.

![](images/b5f2d98ea9c9e6b9f851c63fe73ddee9bd0db5cf884c2413a3d8ad389c2ee4cf.jpg)

Figure 24: Failure analysis of five misclassified test images: Grad-CAM maps for the ground-truth and predicted class, with class probabilities.  
![](images/02d0d1136f79755a946ec302ed1048b1760137931bdea8eb44c1254a95b700cd.jpg)

![](images/bc63142d368cf9b99d78afcc56a3f66743454e797a51d325a7f3421a64b5787d.jpg)  
Figure 25: Deletion and insertion faithfulness curves for Grad-CAM, Grad-CAM++ and a random control, averaged over 200 test images.

(vi) Sanity check. Finally, following Adebayo et al. [1], we verify that the attribution depends on the learned parameters rather than acting as an edge detector. After cascading randomization of the classifier head and the final stage, the Grad-CAM maps collapse to a near-uniform field with no correspondence to the leaf (Figure 26), unlike the sharply localized maps of the trained model, and they notably do not fall back on the contour structure that dominates the first stage. Localization is therefore a property of the learned weights rather than of the input’s edge content, so the explanations satisfy a model-sensitivity criterion that many published saliency analyses fail.

![](images/ff68fd7981a7fae477c6b7671f95362351e7460fbfb196af48e14b7d6afd90a2.jpg)  
Figure 26: Sanity check by cascading weight randomization [1]. input images (top), trained-model Grad-CAM maps (middle), and maps after randomization (bottom).

## 6.11. Discussion and Limitations

For RQ1, a convolutional network trained from scratch at 290,572 parameters exceeds every ImageNetpretrained lightweight backbone evaluated here, reaching 99.52% against 99.25% for the strongest lightweight baseline while using 11.2 times fewer parameters, and the five-seed lower bound of 99.47% remains above all six baselines. Pretraining is therefore not a prerequisite for competitive accuracy at this scale once the architecture matches the structure of the target distribution.

For RQ2, the ablation isolates the swish activation as the largest single accuracy contributor at 0.27 percentage points, and identifies the multi-scale depthwise block and the grouped convolutions as the components that contribute accuracy and parameter economy at once, each of whose removal raises the parameter count by 41.0% or more while lowering accuracy. The dual-pooling head is accuracy neutral within the ablation protocol at a cost of 16,384 parameters, and augmentation together with weight decay each cost 0.16 percentage points of clean-test accuracy under the current acquisition protocol.

For RQ3, the unmodified architecture reaches 98.71% on 21 classes of VegNet-BD and 99.05% on five classes of RadishLeaf-BD, with only the output width altered, which indicates that the design encodes a general inductive bias for foliar lesion discrimination rather than a fit to BD-PlantDX. The same architecture further sustains 99.19% accuracy when the three datasets are merged into a single 36-class corpus of 28,019 images at a 0.5% parameter increase, which confirms that the design scales to a heterogeneous multi-crop label space and not merely to each collection in isolation.

For RQ4, accuracy over five seeds is 99.57±0.10% within a 0.21 percentage point range, while the epoch of best validation performance varies by a factor of 2.2, showing that the reported accuracy is reproducible even though the optimization trajectory is not; a paired analysis over the same five seeds further confirms that the margin over every pretrained baseline is statistically reliable, with AgroVisNet winning all five seeds against each backbone and a Holm-corrected paired t-test significant in every case (Section 6.6).

For RQ5, the attribution maps localize onto lesion-bearing lamina rather than onto background across representative classes and misclassifications alike; the insertion and confidence-drop faithfulness measures confirm that the highlighted region carries the evidence the model uses, and a weight-randomization sanity check confirms that the localization depends on the learned parameters rather than on input edges. The deletion measure does not separate the maps from a random control on this leaf-dominated imagery, and a quantitative agreement against agronomist-drawn lesion masks remains outstanding.

Four limitations bound these conclusions. All BD-PlantDX imagery was acquired under a controlled illumination arrangement against a uniform background, and accuracy under field illumination, cluttered backgrounds and partial occlusion is not yet measured; the synthetic corruption sweep of Table 15 is a proxy that shows tolerance to compression and blur but sensitivity to additive noise and brightness shift, and a genuine field-condition subset remains the outstanding test. The eficiency comparison now covers multiplyaccumulate cost, single-precision and quantized model size, desktop CPU inference latency and, in Table 10, on-device latency and throughput on a mid-range Android handset, but sustained-load thermal behaviour and peak activation memory remain open, the latter reported neither on-device nor for the baselines. The paired margin over the baselines is now established by a Holm-corrected t-test and a clean five-of-five seed sweep with all six confidence intervals excluding zero (Section 6.6); the residual caveat is only that the Wilcoxon signed-rank test is underpowered at five seeds, its exact two-sided p-value bounded below by 0.0625, so a larger seed budget would let the rank test corroborate the same margin. Finally, the twelve classes cover the pathologies prevalent in two districts over one cropping season, and coverage of additional agro-ecological zones and seasons remains open.

## 7. Conclusion

We presented AgroVisNet, a convolutional network of 290,572 trainable parameters that composes grouped bottleneck residual blocks carrying sequential channel and spatial attention, multi-scale depthwise blocks fusing 3×3 and 5×5 responses, and a dual-pooling classification head, together with BD-PlantDX, an expert-validated benchmark of 12,432 field images spanning 12 healthy and diseased classes of radish, potato and pointed gourd collected across the Bogura and Nilphamari districts of Bangladesh. On BD-PlantDX the model attains 99.52% accuracy and 99.52% weighted F1, exceeding all six ImageNet-pretrained lightweight backbones evaluated under an identical protocol while using 8.7 to 16.8 times fewer parameters and the fewest multiply-accumulate operations of any model compared, and it exceeds a four-model heavyweight transfer ensemble by 0.16 percentage points at less than one percent of its parameter count. Exported for deployment, the model quantises to a 0.46 MB full-integer network at a 0.22 percentage-point accuracy cost and classifies an image in under ten milliseconds on a single CPU. Accuracy is stable at 99.57 ± 0.10% over five seeds and its margin over every ImageNet-pretrained baseline holds across all five seeds under a paired significance analysis, an eleven-variant ablation shows that the multi-scale depthwise block and the grouped convolutions supply accuracy and parameter economy jointly, the unmodified architecture transfers to two independently collected datasets at 98.71% and 99.05% accuracy, and Grad-CAM evidence indicates that predictions rest on lesion-bearing tissue rather than on background cues. The present evaluation rests on imagery acquired under controlled illumination, and the eficiency characterization, while it now spans compute, model size, quantization, and CPU latency, does not yet include measurements on the target hardware itself.

Future work will close these gaps: a field-condition evaluation subset with natural backgrounds and variable illumination will establish whether the accuracy reported here survives deployment conditions, the synthetic corruption sweep of Section 6 being a first step that shows strong tolerance to compression and blur but sensitivity to additive noise and illumination shift; sustained-load latency and memory characterisation on the target embedded hardware will carry the quantised model from a laboratory eficiency profile to a verified on-device deployment claim; and severity grading annotation will extend the label space from disease presence toward the intervention decision a grower actually faces.

Declaration of Generative AI and AI-Assisted Technologies in the Manuscript Preparation Process

During the preparation of this manuscript, large language models (LLMs) were employed only as supporting tools for improving language and for assisting with code debugging. All AI-assisted material was subsequently examined, verified, and corrected by the authors, who assume full responsibility for the accuracy, originality, and integrity of the work presented here.

Originality. The role of the LLMs was strictly editorial. Any passages that were drafted or polished with such assistance were fact-checked and rewritten by the authors to ensure consistency with the actual research findings. Every figure, table, and quantitative outcome, together with all technical descriptions and reported numerical values, was generated solely by the authors without any AI contribution.

Transparency. LLMs were additionally applied during code development, mainly for identifying and resolving implementation errors. Even so, the experimental pipeline demanded considerable manual design, incremental correction, and iterative testing, since AI-generated recommendations on their own proved inadequate. Each reported result was produced by executing author-verified code and was checked against expected behaviour prior to inclusion. The underlying methodology, the model architecture, and the Agro-VisNet experimental protocol were designed and validated by the authors independently of any LLM.

Responsibility. At no point during the writing or debugging process was any sensitive, private, or proprietary information such as dataset images or participant data disclosed to an AI tool, and every interaction adhered to ethical principles concerning data ownership and intellectual property. The use of LLMs remained limited to general writing support and code debugging, and it had no bearing on the scientific contributions or the claims advanced in this paper.

## CRediT authorship contribution statement

Md. Abdullah Mandal: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Project administration, Software, Visualization, Validation, Writing - original draft. Saad Ahmed: Formal analysis, Resources, Supervision, Validation, Writing - review & editing. Md. Khalid Syfullah: Formal analysis, Project administration, Resources, Supervision, Writing - review & editing.

## Declaration of Competing Interest

The authors declare that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this article.

## Acknowledgments

The authors gratefully acknowledge Professor Dr. Md. Harun Ar Rashid, Department of Horticulture, Bangladesh Agricultural University, Mymensingh, for his expert validation of the BD-PlantDX dataset. His independent, per-class review of the radish, potato and pointed gourd images, recorded in a signed Certificate of Data Integrity, confirmed the accuracy and consistency of the healthy and diseased class assignments across all twelve classes. The authors further thank the local agronomists who guided the field surveys and identified the diseased plots across the Bogura and Nilphamari districts, and the farmers and landholders who permitted specimen collection from their fields. Their agronomic expertise and cooperation were essential to the construction of BD-PlantDX.The authors received no external funding for this study.

## Data and Code Availability

The BD-PlantDX dataset, the trained AgroVisNet weights, and the figure-generation notebook will be released at (https://github.com/Abdullah2104/agrovisnet).

## References

[1] Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B., 2018. Sanity checks for saliency maps, in: Advances in Neural Information Processing Systems, pp. 9505–9515.

[2] Ajith, A., Milnes, P.J., Johnson, G.N., Lockyer, N.P., 2022. Mass spectrometry imaging for spatial chemical profiling of vegetative parts of plants. Plants 11, 1234.

[3] Alam, N., Sagar, A.S., Dang, L.M., Zhang, W., Park, H.Y., Hyeonjoon, M., 2025. Deep learning based radish and leaf segmentation for phenotype trait measurement. Signal, Image and Video Processing 19, 178.

[4] Alhammad, S.M., Khafaga, D.S., El-Hady, W.M., Samy, F.M., Hosny, K.M., 2025. Deep learning and explainable AI for classification of potato leaf diseases. Frontiers in Artificial Intelligence 7, 1449329.

[5] Ali, A.H., Youssef, A., Abdelal, M., Raja, M.A., 2024. An ensemble of deep learning architectures for accurate plant disease classification. Ecological Informatics 81, 102618.

[6] Banerjee, D., Kukreja, V., Aeri, M., Hariharan, S., Garg, N., 2023. Integrated CNN-SVM approach for accurate radish leaf disease classification: a comparative study and performance analysis, in: Proceedings of the International Conference on Disruptive Technologies, IEEE. pp. 1–6.

[7] Barman, U., Choudhury, R.D., Sahu, D., Barman, G.G., 2020a. Comparison of convolution neural networks for smartphone image based real time classification of citrus leaf disease. Computers and Electronics in Agriculture 177, 105661.

[8] Barman, U., Sahu, D., Barman, G.G., Das, J., 2020b. Comparative assessment of deep learning to detect the leaf diseases of potato based on data augmentation, in: Proceedings of the International Conference on Computational Performance Evaluation, IEEE. pp. 682–687.

[9] Chattopadhay, A., Sarkar, A., Howlader, P., Balasubramanian, V.N., 2018. Grad-CAM++: generalized gradient-based visual explanations for deep convolutional networks, in: Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV), IEEE. pp. 839–847.

[10] Chen, C., Guo, Z., Zeng, H., Xiong, P., Dong, J., 2022. RepGhost: a hardware-eficient ghost module via re-parameterization. arXiv preprint arXiv:2211.06088 .

[11] Dame, T.A., Adera, G.B., Girmaw, D.W., 2025. Deep learning-based potato leaf disease classification and severity assessment. Discover Applied Sciences 7, 649.

[12] Demšar, J., 2006. Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research 7, 1–30.

[13] Geetharamani, G., Pandian, A., 2019. Identification of plant leaf diseases using a nine-layer deep convolutional neural network. Computers and Electrical Engineering 76, 323–338.

[14] Haque, M.M., Hoque, M.Z., 2021. Vegetable production and marketing channels in Bangladesh: present scenario, problems, and prospects. Technical Report. Bangladesh Agricultural Research Council.

[15] Harakannanavar, S.S., Rudagi, J.M., Puranikmath, V.I., Siddiqua, A., Pramodhini, R., 2022. Plant leaf disease detection using computer vision and machine learning algorithms. Global Transitions Proceedings 3, 305–310.

[16] Hasan, M., Gani, R., Rashid, M.R.A., Isty, M.N., Kamara, R., Tarin, T.K., 2025. Smartphone image dataset for radish plant leaf disease classification from bangladesh. Data in Brief 58, 111263.

[17] Hasan, M., Gani, R., Rashid, M.R.A., Tarin, T.K., Kamara, R., Mou, M.Y., Rabbi, S.F., 2024. Comprehensive smart smartphone image dataset for plant leaf disease detection and freshness assessment from bangladesh vegetable fields. Data in Brief 56, 110775.

[18] Hassan, S.M., Maji, A.K., Jasiński, M., Leonowicz, Z., Jasińska, E., 2021. Identification of plant-leaf diseases using cnn and transfer-learning approach. Electronics 10, 1388.

[19] He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778.

[20] Holm, S., 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6, 65–70.

[21] Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H., 2017. MobileNets: eficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 .

[22] Hu, J., Shen, L., Sun, G., 2018. Squeeze-and-excitation networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7132–7141.

[23] Iofe, S., Szegedy, C., 2015. Batch normalization: accelerating deep network training by reducing internal covariate shift, in: Proceedings of the International Conference on Machine Learning, pp. 448–456.

[24] Islam, M.H., Islam, S., Masud, M.M., Mita, M.M., Islam, M.S., Islam, M.R., 2022. Identification of potential chemical fungicides with diverse groups of active ingredient for controlling late blight of potato in bangladesh. Asian-Australasian Journal of Bioscience and Biotechnology 7, 23–35.

[25] Jain, A., Dubey, A.K., Singh, S.K., Panwar, A., Gupta, N., Kumar, S., Arya, V., Alhalabi, W., Pan, S.H., Alsulami, B.S., Gupta, B.B., 2026. Optimized CNN-based ensemble deep learning approach for potato leaf disease detection with data augmentation. Scientific Reports 16, 22379. doi:10.1038/s415 98-026-50480-8.

[26] Jha, P., Dembla, D., Dubey, W., 2024. Deep learning models for enhancing potato leaf disease prediction: implementation of transfer learning based stacking ensemble model. Multimedia Tools and Applications 83, 37839–37858.

[27] Ji, M., Zhou, Z., Wang, X., Tang, W., Li, Y., Wang, Y., Zhou, C., Lv, C., 2024. Implementing real-time image processing for radish disease detection using hybrid attention mechanisms. Plants 13, 3001.

[28] Krizhevsky, A., Sutskever, I., Hinton, G.E., 2012. ImageNet classification with deep convolutional neural networks, in: Advances in Neural Information Processing Systems, pp. 1097–1105.

[29] Lau, H.Y., Botella, J.R., 2017. Advanced DNA-based point-of-care diagnostic methods for plant diseases detection. Frontiers in Plant Science 8, 2016.

[30] Lee, T.Y., Yu, J.Y., Chang, Y.C., Yang, J.M., 2020. Health detection for potato leaf with convolutional neural network, in: Proceedings of the Indo-Taiwan International Conference on Computing, Analytics and Networks, IEEE. pp. 289–293.

[31] Li, Y., Hu, J., Wen, Y., Evangelidis, G., Salahi, K., Wang, Y., Tulyakov, S., Ren, J., 2023. Rethinking vision transformers for MobileNet size and speed, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16889–16900.

[32] Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S., 2022. A ConvNet for the 2020s, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 11976–11986.

[33] Maaz, M., Shaker, A., Cholakkal, H., Khan, S., Zamir, S.W., Anwer, R.M., Khan, F.S., 2022. EdgeNeXt: eficiently amalgamated CNN-transformer architecture for mobile vision applications, in: Proceedings of the European Conference on Computer Vision Workshops, pp. 3–20.

[34] Mehta, S., Rastegari, M., 2022. Separable self-attention for mobile vision transformers. arXiv preprint arXiv:2206.02680 .

[35] Mohd Hilmi Tan, M.I.S., et al., 2021. Ganoderma boninense disease detection by near-infrared spectroscopy classification: a review. Sensors 21, 3052.

[36] Nawaz, M., Javed, A., Saudagar, A.K.J., 2026. PotatoGuardNet: a refined deep learning framework for potato leaf disease detection. Frontiers in Plant Science 17, 1720276.

[37] Nishio, T., 2017. Economic and academic importance of radish, in: The radish genome. Springer, pp. 1–10.

[38] Pal, C., Karmakar, S., Mukherjee, I., Chakrabarti, P.P., 2025. A lightweight and explainable CNN model for empowering plant disease diagnosis. Scientific Reports 15, 30720. doi:10.1038/s41598-025 -94083-1.

[39] Pandian, J.A., Kumar, V.D., Geman, O., Hnatiuc, M., Arif, M., Kanchanadevi, K., 2022. Plant disease detection using deep convolutional neural network. Applied Sciences 12, 6982.

[40] Paul, H., Ghatak, S., Chakraborty, S., Pandey, S.K., Dey, L., Show, D., Maity, S., 2024. A study and comparison of deep learning based potato leaf disease detection and classification techniques using explainable ai. Multimedia Tools and Applications 83, 42485–42518.

[41] Petsiuk, V., Das, A., Saenko, K., 2018. RISE: randomized input sampling for explanation of black-box models, in: Proceedings of the British Machine Vision Conference (BMVC).

[42] Qadri, S.A.A., Huang, N.F., Wani, T.M., Bhat, S.A., 2024. Advances and challenges in computer vision for image-based plant disease detection: a comprehensive survey of machine and deep learning approaches. IEEE Transactions on Automation Science and Engineering 22, 2639–2670.

[43] Qin, D., Leichner, C., Delakis, M., Fornoni, M., Luo, S., Yang, F., Wang, W., Banbury, C., Ye, C., Akin, B., Aggarwal, V., Zhu, T., Moro, D., Howard, A., 2024. MobileNetV4: universal models for the mobile ecosystem, in: Computer Vision – ECCV 2024, Springer. doi:10.1007/978-3-031-73661-2\_5.

[44] Qin, X., Shi, Y., Huang, X., Li, H., Huang, J., Yuan, C., Liu, C., 2021. Attention-based deep multi-scale network for plant leaf recognition, in: International Conference on Intelligent Computing, Springer. pp. 302–313.

[45] Quoc, K.N., Thu, L.L.T., Quach, L.D., 2025. A vision-language foundation model for leaf disease identification. Expert Systems with Applications , 130084.

[46] Rakesh, M., Jeevankumar, M., Rudraswamy, S., 2025. Implementation of real time root crop leaf classification using CNN on Raspberry Pi microprocessor. Smart Agricultural Technology 10, 100714.

[47] Ramachandran, P., Zoph, B., Le, Q.V., 2017. Searching for activation functions. arXiv preprint arXiv:1710.05941 .

[48] Rashid, J., Khan, I., Ali, G., Almotiri, S.H., AlGhamdi, M.A., Masood, K., 2021. Multi-level deep learning model for potato leaf disease recognition. Electronics 10, 2064.

[49] Reis, H.C., Turk, V., 2024. Potato leaf disease detection with a novel deep learning model based on depthwise separable convolution and transformer networks. Engineering Applications of Artificial Intelligence 133, 108307.

[50] Rozaqi, A.J., Sunyoto, A., 2020. Identification of disease in potato leaves using convolutional neural network (CNN) algorithm, in: Proceedings of the International Conference on Information and Communications Technology, IEEE. pp. 72–76.

[51] Sanchez-Sanchez, J., Minguillo-Rubio, C., Arcila-Diaz, J., 2026. Development of a machine learning technique for the detection of late blight in potato leaves. Journal of Agriculture and Food Research , 103109.

[52] Sanjeev, K., Gupta, N.K., Jeberson, W., Paswan, S., 2021. Early prediction of potato leaf diseases using ANN classifier. Oriental Journal of Computer Science and Technology 13, 129–134.

[53] Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D., 2017. Grad-CAM: visual explanations from deep networks via gradient-based localization, in: Proceedings of the IEEE International Conference on Computer Vision, pp. 618–626.

[54] Shafik, W., Tufail, A., De Silva, L.C., Haji Mohd Apong, R.A.A., Kim, K.H., 2025. Deep learning technique for plant disease classification and pest detection and model explainability elevating agricultural sustainability. BMC Plant Biology 25, 1491.

[55] Shaker, A., Maaz, M., Rasheed, H., Khan, S., Yang, M.H., Khan, F.S., 2023. SwiftFormer: eficient additive attention for transformer-based real-time mobile vision applications, in: Proceedings of the IEEE International Conference on Computer Vision, pp. 17425–17436.

[56] Sinamenye, J.H., Chatterjee, A., Shrestha, R., 2025. Potato plant disease detection: leveraging hybrid deep learning models. BMC Plant Biology 25, 647.

[57] Sladojevic, S., Arsenovic, M., Anderla, A., Culibrk, D., Stefanovic, D., 2016. Deep neural networks based recognition of plant diseases by leaf image classification. Computational Intelligence and Neuroscience 2016, 3289801.

[58] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R., 2014. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research 15, 1929–1958.

[59] Tang, Y., Han, K., Guo, J., Xu, C., Xu, C., Wang, Y., 2022. GhostNetV2: enhance cheap operation with long-range attention, in: Advances in Neural Information Processing Systems, pp. 9969–9982.

[60] The Financial Express, 2016. Bogra, Nilphamari on way to massive production of winter vegetables. URL: https://today.thefinancialexpress.com.bd/print/bogra-nilphamari-on-way-to-massi ve-production-of-winter-vegetables. accessed 6 October 2025.

[61] Tompson, J., Goroshin, R., Jain, A., LeCun, Y., Bregler, C., 2015. Eficient object localization using convolutional networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 648–656.

[62] Ullah, N., Khan, J.A., Almakdi, S., Alshehri, M.S., Al Qathrady, M., El-Rashidy, N., El-Sappagh, S., Ali, F., 2023. An efective approach for plant leaf diseases classification based on a novel DeepPlantNet deep learning model. Frontiers in Plant Science 14, 1212747.

[63] Vasu, P.K.A., Gabriel, J., Zhu, J., Tuzel, O., Ranjan, A., 2023. FastViT: a fast hybrid vision transformer using structural reparameterization, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5785–5795.

[64] Vishnoi, V.K., Kumar, K., Kumar, B., Mohan, S., Khan, A.A., 2022. Detection of apple plant diseases using leaf images through convolutional neural network. IEEE Access 11, 6594–6609.

[65] Wäldchen, J., Rzanny, M., Seeland, M., Mäder, P., 2018. Automated plant species identification: trends and future directions. PLoS Computational Biology 14, e1005993.

[66] Wang, A., Chen, H., Lin, Z., Han, J., Ding, G., 2024. RepViT: revisiting mobile CNN from ViT perspective, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15909–15920.

[67] Wilcoxon, F., 1945. Individual comparisons by ranking methods. Biometrics Bulletin 1, 80–83.

[68] Woo, S., Park, J., Lee, J.Y., Kweon, I.S., 2018. CBAM: convolutional block attention module, in: Proceedings of the European Conference on Computer Vision, pp. 3–19.

[69] Zhang, Y., Song, C., Zhang, D., 2020. Deep learning-based object detection improvement for tomato disease. IEEE Access 8, 56607–56614.

Table 4: List of Notations.
<table><tr><td rowspan=1 colspan=1>Symbol</td><td rowspan=1 colspan=1>Description</td><td rowspan=1 colspan=1>Symbol</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>Raw RGB input image, $\overline { { \mathbf { X } \in \mathbb { R } ^ { H \times W \times 3 } } }$ </td><td rowspan=1 colspan=1> $\sigma ( \cdot ) , \rho ( \cdot )$ </td><td rowspan=1 colspan=1>Sigmoid and rectified linear activation</td></tr><tr><td rowspan=1 colspan=1> $H , W$ </td><td rowspan=1 colspan=1>Input height and width, both set to 224</td><td rowspan=1 colspan=1>BN(·)</td><td rowspan=1 colspan=1>Batch normalization</td></tr><tr><td rowspan=1 colspan=1> $\overline { { H ^ { \prime } , W ^ { \prime } } }$ </td><td rowspan=1 colspan=1>Height and width of an intermediate fea-ture map</td><td rowspan=1 colspan=1> $_ { \mathrm { G A P , G M P } }$ </td><td rowspan=1 colspan=1>Global average and global maximumpooling</td></tr><tr><td rowspan=1 colspan=1> $C$ </td><td rowspan=1 colspan=1>Channel count at the input of a block</td><td rowspan=1 colspan=1> $\mathbf { W } _ { \mathrm { s t e m } }$ </td><td rowspan=1 colspan=1>Kernel of the strided stem convolution</td></tr><tr><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>Rescaled image with entries in [0, 1]</td><td rowspan=1 colspan=1> $\mathbf { W } _ { r 1 } , \mathbf { W } _ { r 2 } , \mathbf { W } _ { r 3 }$ </td><td rowspan=1 colspan=1>Kernels of the grouped residual bottle-neck</td></tr><tr><td rowspan=1 colspan=1> $\widehat { \mathbf { x } }$ </td><td rowspan=1 colspan=1>Augmented network input</td><td rowspan=1 colspan=1> $\mathbf { W } _ { 0 } , \mathbf { W } _ { 1 }$ </td><td rowspan=1 colspan=1>Weights of the shared attention percep-tron</td></tr><tr><td rowspan=1 colspan=1> $K$ </td><td rowspan=1 colspan=1>Number of classes, K = 12 on BD-PlantDX</td><td rowspan=1 colspan=1> $\mathbf { W } _ { s }$ </td><td rowspan=1 colspan=1>Kernel of the spatial attention gate</td></tr><tr><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>Number of images in a split</td><td rowspan=1 colspan=1> $\mathbf { W } _ { p } , \mathbf { W } _ { q }$ </td><td rowspan=1 colspan=1>Pointwise kernels of the depthwise block</td></tr><tr><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>Labeled collection $\left\{ \left( \mathbf { X } _ { i } , \mathbf { y } _ { i } \right) \right\} _ { i = 1 } ^ { N }$ </td><td rowspan=1 colspan=1> $\mathbf { W } _ { d 3 } , \mathbf { W } _ { d 5 }$ </td><td rowspan=1 colspan=1>Depthwise kernels at size 3 and size 5</td></tr><tr><td rowspan=1 colspan=1> $\mathbf { y } , \widehat { \mathbf { y } }$ </td><td rowspan=1 colspan=1>One-hot label and predicted class distri-bution</td><td rowspan=1 colspan=1> $\mathbf { W } _ { f 1 } , \mathbf { W } _ { f 2 } , \mathbf { W } _ { o }$ </td><td rowspan=1 colspan=1>Kernels of the classification head</td></tr><tr><td rowspan=1 colspan=1> $\overline { { { \Delta } ^ { K } } }$ -1</td><td rowspan=1 colspan=1>Probability simplex over the K classes</td><td rowspan=1 colspan=1> $\mathbf { M } _ { c }$ </td><td rowspan=1 colspan=1>Channel attention vector, $\overline { { \mathbf { M } _ { c } \in \mathbb { R } ^ { 1 \times 1 \times F } } }$ </td></tr><tr><td rowspan=1 colspan=1>θ</td><td rowspan=1 colspan=1>Trainable parameters of the network</td><td rowspan=1 colspan=1> $\mathbf { M } _ { s }$ </td><td rowspan=1 colspan=1>Spatial attention map, $\overline { { \mathbf { M } _ { s } \in \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } \times 1 } } }$ </td></tr><tr><td rowspan=1 colspan=1> $f _ { \theta }$ </td><td rowspan=1 colspan=1>Network mapping an image to $\overline { { \Delta ^ { K - 1 } } }$ </td><td rowspan=1 colspan=1> ${ \bf z } _ { \mathrm { a v g } } , { \bf z } _ { \mathrm { m a x } }$ </td><td rowspan=1 colspan=1>Globally averaged and maximized chan-nel descriptors</td></tr><tr><td rowspan=1 colspan=1> $A _ { \tau }$ </td><td rowspan=1 colspan=1>Stochastic training-time augmentationoperator</td><td rowspan=1 colspan=1> $r$ </td><td rowspan=1 colspan=1>Channel attention reduction ratio, $r =$ 16</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathbf { U } ^ { ( l ) } } }$ </td><td rowspan=1 colspan=1>Input tensor of stage l</td><td rowspan=1 colspan=1> $s ( \cdot )$ </td><td rowspan=1 colspan=1>Shortcut mapping, identity or 1 × 1 pro-jection</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathbf { z } ^ { ( i ) } } }$ </td><td rowspan=1 colspan=1>i-th intermediate tensor inside a residualblock</td><td rowspan=1 colspan=1> $\mathbf { P }$ </td><td rowspan=1 colspan=1>Pointwise-expandedtensor inside adepthwise block</td></tr><tr><td rowspan=1 colspan=1> $\mathbf { Y }$ </td><td rowspan=1 colspan=1>Output tensor of an Enhanced ResidualBlock</td><td rowspan=1 colspan=1> $\mathbf { D } _ { 3 } , \mathbf { D } _ { 5 }$ </td><td rowspan=1 colspan=1>Depthwise responses at kernel size 3 and5</td></tr><tr><td rowspan=1 colspan=1> $F _ { l }$ </td><td rowspan=1 colspan=1>Output channel width of stage l</td><td rowspan=1 colspan=1> $\mathbf { Q }$ </td><td rowspan=1 colspan=1>Fused output of a Multi-Scale DepthwiseBlock</td></tr><tr><td rowspan=1 colspan=1> $g _ { l }$ </td><td rowspan=1 colspan=1>Number of convolution groups in stage l</td><td rowspan=1 colspan=1> $p _ { s }$ </td><td rowspan=1 colspan=1>Spatial dropout rate of a stage</td></tr><tr><td rowspan=1 colspan=1> $k , s$ </td><td rowspan=1 colspan=1>Convolution kernel size and stride</td><td rowspan=1 colspan=1> $p _ { d }$ </td><td rowspan=1 colspan=1>Dense dropout rate, $p _ { d } = 0 . 3$ </td></tr><tr><td rowspan=1 colspan=1> $\circledast { k , s }$ </td><td rowspan=1 colspan=1>Convolution with kernel size k and strideS</td><td rowspan=1 colspan=1> $\mathbf { v }$ </td><td rowspan=1 colspan=1>Dual-pooled feature vector, v $\overline { { \in \mathbb { R } ^ { 2 F _ { 4 } } } }$ </td></tr><tr><td rowspan=1 colspan=1> $\circledast , g$ </td><td rowspan=1 colspan=1>Grouped convolution over g groups</td><td rowspan=1 colspan=1> $\mathbf { h } _ { 1 } , \mathbf { h } _ { 2 }$ </td><td rowspan=1 colspan=1>Hidden activations of the classificationhead</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \boldsymbol { \mathfrak { E } } _ { k } ^ { \mathrm { d w } } } }$ </td><td rowspan=1 colspan=1>Depthwise convolution with kernel size k</td><td rowspan=1 colspan=1> $\mathcal { L }$ </td><td rowspan=1 colspan=1>Total training objective</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Broadcast Hadamard product</td><td rowspan=1 colspan=1> $\lambda$ </td><td rowspan=1 colspan=1>Weight decay coefficient, $\lambda = 1 0 ^ { - 5 }$ </td></tr><tr><td rowspan=1 colspan=1> $[ \cdot ; \cdot ]$ </td><td rowspan=1 colspan=1>Concatenation along the channel axis</td><td rowspan=1 colspan=1> $\eta _ { t } , \mathbf { m } _ { t }$ </td><td rowspan=1 colspan=1>Learning rate and momentum buffer atstep t</td></tr><tr><td rowspan=1 colspan=1>|1·∥|F</td><td rowspan=1 colspan=1>Frobenius norm of a kernel</td><td rowspan=1 colspan=1> $\mu , B , E$ </td><td rowspan=1 colspan=1>Momentum 0.9, batch size 32, epoch cap300</td></tr><tr><td rowspan=1 colspan=1> $\delta ( \cdot )$ </td><td rowspan=1 colspan=1>Swish activation, δ(x) = x σ(x)</td><td rowspan=1 colspan=1> $\Omega ( \cdot )$ </td><td rowspan=1 colspan=1>Parameter count of a layer or a block</td></tr><tr><td rowspan=1 colspan=1> $\mathbf { A } _ { k }$ </td><td rowspan=1 colspan=1>k-th activation map of the last convolu-tional tensor</td><td rowspan=1 colspan=1> $\overline { { \alpha _ { k } ^ { c } , { \bf L } ^ { c } } }$ </td><td rowspan=1 colspan=1>Grad-CAM channel weight and localiza-tion map for class c</td></tr></table>

Table 5: Per-class test performance of AgroVisNet on BD-PlantDX.
<table><tr><td>Class</td><td>Prec. (%)</td><td>Rec. (%)</td><td>F1 (%)</td><td>Supp.</td></tr><tr><td>Potato Healthy</td><td>100.00</td><td>100.00</td><td>100.00</td><td>152</td></tr><tr><td>Gourd Healthy</td><td>100.00</td><td>99.34</td><td>99.67</td><td>152</td></tr><tr><td>Radish Root Healthy</td><td>100.00</td><td>100.00</td><td>100.00</td><td>174</td></tr><tr><td>Radish Leaf Healthy</td><td>99.35</td><td>97.45</td><td>98.39</td><td>157</td></tr><tr><td>Potato Late Blight</td><td>100.00</td><td>99.34</td><td>99.67</td><td>151</td></tr><tr><td>Potato Mosaic</td><td>100.00</td><td>98.68</td><td>99.34</td><td>152</td></tr><tr><td>Potato Nutrient Deficiency</td><td>100.00</td><td>100.00</td><td>100.00</td><td>152</td></tr><tr><td>Gourd Downy Mildew</td><td>100.00</td><td>100.00</td><td>100.00</td><td>153</td></tr><tr><td>Radish Alternaria brassicae</td><td>100.00</td><td>100.00</td><td>100.00</td><td>154</td></tr><tr><td>Radish Flea Beetle Damage</td><td>95.60</td><td>100.00</td><td>97.75</td><td>152</td></tr><tr><td>Radish Scab</td><td>99.41</td><td>100.00</td><td>99.70</td><td>168</td></tr><tr><td>Radish White Mold</td><td>100.00</td><td>99.35</td><td>99.67</td><td>153</td></tr><tr><td>Macro average</td><td>99.53</td><td>99.51</td><td>99.52</td><td>1,870</td></tr><tr><td>Weighted average</td><td>99.53</td><td>99.52</td><td>99.52</td><td>1,870</td></tr></table>

Table 6: AgroVisNet against six ImageNet-pretrained lightweight backbones on BD-PlantDX, all under an identical split and protocol.
<table><tr><td>Model</td><td>Params (M)</td><td>Acc. (%)</td><td>Prec. (%)</td><td>Rec. (%)</td><td>F1 (%)</td></tr><tr><td>AgroVisNet (ours)</td><td>0.29</td><td>99.52</td><td>99.53</td><td>99.52</td><td>99.52</td></tr><tr><td>EfficientFormerV2-S0</td><td>3.25</td><td>99.25</td><td>99.25</td><td>99.25</td><td>99.25</td></tr><tr><td>RepViT-M1.0</td><td>4.72</td><td>99.20</td><td>99.20</td><td>99.20</td><td>99.20</td></tr><tr><td>ConvNeXt-Atto</td><td>3.38</td><td>99.14</td><td>99.15</td><td>99.14</td><td>99.14</td></tr><tr><td>FastViT-T8</td><td>3.27</td><td>99.04</td><td>99.04</td><td>99.04</td><td>99.04</td></tr><tr><td>MobileNetV4-Conv-Small</td><td>2.51</td><td>98.98</td><td>98.99</td><td>98.98</td><td>98.98</td></tr><tr><td>GhostNetV2-1.0</td><td>4.89</td><td>98.93</td><td>98.94</td><td>98.93</td><td>98.93</td></tr></table>

Table 7: Comparison of AgroVisNet against two methods published specifically for plant-disease classification, each retrained on BD-PlantDX under its own published hyperparameters and evaluated on the shared 1,870-image test split (Section 6.3).
<table><tr><td>Method</td><td>Params</td><td>Orig. acc. (own data)</td><td>Acc. (%)</td><td>F1 (%)</td></tr><tr><td>Mob-Res [38]</td><td>3.47M</td><td>99.47</td><td>99.20</td><td>99.20</td></tr><tr><td>EDL10 [25]</td><td>8.49M</td><td>97.00</td><td>96.52</td><td>96.53</td></tr><tr><td>AgroVisNet (proposed)</td><td>0.29M</td><td></td><td>99.52</td><td>99.52</td></tr></table>

Table 8: Computational eficiency of AgroVisNet against the six pretrained backbones on BD-PlantDX at 224 × 224 resolution (Section 6.4).
<table><tr><td>Model</td><td>Params (M)</td><td>MACs (M)</td><td>FP32 (MB)</td><td>Acc. (%)</td></tr><tr><td>AgroVisNet (ours)</td><td>0.29</td><td>132.70</td><td>1.13</td><td>99.52</td></tr><tr><td>EfficientFormerV2-S0</td><td>3.25</td><td>406.70</td><td>12.43</td><td>99.25</td></tr><tr><td>RepViT-M1.0</td><td>4.72</td><td>1124.58</td><td>24.56</td><td>99.20</td></tr><tr><td>ConvNeXt-Atto</td><td>3.38</td><td>551.75</td><td>12.92</td><td>99.14</td></tr><tr><td>FastViT-T8</td><td>3.27</td><td>539.11</td><td>12.54</td><td>99.04</td></tr><tr><td>MobileNetV4-Conv-Small</td><td>2.51</td><td>188.75</td><td>9.70</td><td>98.98</td></tr><tr><td>GhostNetV2-1.0</td><td>4.89</td><td>176.34</td><td>18.79</td><td>98.93</td></tr><tr><td>ratio (min-max)</td><td colspan="4">8.7-16.8× 1.3-8.5× 8.6–21.7×</td></tr></table>

Table 9: Export and post-training quantization of AgroVisNet, measured through the TensorFlow Lite XNNPACK delegate (Section 6.4).
<table><tr><td>Export format</td><td>Size (MB)</td><td>CPU lat. (ms)</td><td>Acc. (%)</td></tr><tr><td>Keras FP32</td><td>1.13</td><td></td><td>99.52</td></tr><tr><td>TFLite FP32</td><td>1.17</td><td>10.44</td><td>99.52</td></tr><tr><td>TFLite dynamic-range</td><td>0.43</td><td>15.12</td><td>99.52</td></tr><tr><td>TFLite INT8 (full)</td><td>0.46</td><td>8.40</td><td>99.30</td></tr></table>

Table 10: On-device inference of AgroVisNet on a mid-range Android handset (realme RMX3853, Snapdragon 7+ Gen 3, Android 16), measured with the TensorFlow Lite benchmark\_model tool (Section 6.4). Throughput is for the four-thread XNNPACK configuration.
<table><tr><td rowspan="2">Export format</td><td rowspan="2">(MB)</td><td colspan="3">Size XNNPACK (ms)</td><td rowspan="2">NNAPI (ms)</td><td rowspan="2">Thr. (img/s)</td></tr><tr><td>1t</td><td>2t</td><td>4t</td></tr><tr><td>TFLite FP32</td><td>1.17</td><td>12.83</td><td>7.61</td><td>5.47</td><td>17.64</td><td>182.9</td></tr><tr><td>TFLite dynamic-range</td><td>0.43</td><td>10.10</td><td>6.06</td><td>4.38</td><td>9.57</td><td>228.4</td></tr><tr><td>TFLite INT8 (full)</td><td>0.46</td><td>6.02</td><td>3.47</td><td>2.33</td><td>5.85</td><td>429.1</td></tr></table>

Table 11: Five-seed repetition of the full training procedure on BD-PlantDX.
<table><tr><td></td><td>Seed Epochs run</td><td>Best epoch</td><td>Val. acc. (%)</td><td>Test acc. (%)</td><td>F1 (%)</td></tr><tr><td>42</td><td>140</td><td>110</td><td>99.79</td><td>99.52</td><td>99.52</td></tr><tr><td>43</td><td>136</td><td>106</td><td>99.79</td><td>99.52</td><td>99.52</td></tr><tr><td>44</td><td>80</td><td>50</td><td>99.63</td><td>99.68</td><td>99.68</td></tr><tr><td>45</td><td>121</td><td>91</td><td>99.84</td><td>99.68</td><td>99.68</td></tr><tr><td>46</td><td>84</td><td>54</td><td>99.63</td><td>99.47</td><td>99.47</td></tr><tr><td> $\mathrm { M e a n } \pm \mathrm { S D }$ </td><td></td><td></td><td> $9 9 . 7 4 \pm 0 . 1 0$ </td><td> $9 9 . 5 7 \pm 0 . 1 0$ </td><td> $9 9 . 5 7 \pm 0 . 1 0$ </td></tr></table>

Table 12: Paired per-seed significance of the AgroVisNet accuracy margin over the six pretrained backbones (Table 11 seeds). ∆: mean accuracy diference with 95% CI; W/L: win/loss record; $p _ { \mathrm { H } } ^ { W } , \bar { p } _ { \mathrm { H } } ^ { t } \colon$ Holm-adjusted Wilcoxon and paired t-test p-value (Section 6.6).
<table><tr><td>Baseline</td><td>∆(pp)</td><td>95% CI (pp)</td><td> $\mathrm { W / L }$ </td><td> $p _ { \mathrm { H } } ^ { W }$ </td><td> $p _ { \mathrm { H } } ^ { t }$ </td></tr><tr><td>EfficientFormerV2-S0</td><td>+0.39</td><td>[0.18, 0.59]</td><td>5/0</td><td>0.375</td><td>0.013</td></tr><tr><td>RepViT-M1.0</td><td>+0.33</td><td>[0.14, 0.52]</td><td>5/0</td><td>0.375</td><td>0.013</td></tr><tr><td>ConvNeXt-Atto</td><td>+0.37</td><td>[0.31, 0.42]</td><td>5/0</td><td>0.375 &lt; 0.001</td><td></td></tr><tr><td>FastViT-T8</td><td>+0.48</td><td>[0.29, 0.68]</td><td>5/0</td><td>00.375</td><td>0.010</td></tr><tr><td>MobileNetV4-Conv-Small</td><td>+0.61</td><td>[0.50, 0.72]</td><td></td><td>5/0 0.375 &lt; 0.001</td><td></td></tr><tr><td>GhostNetV2-1.0</td><td>+0.43</td><td>[0.23, 0.63]</td><td></td><td>5/00.375</td><td>0.012</td></tr></table>

Table 13: Component ablation on BD-PlantDX. Accuracy change is measured against the V0 full model within the ablation protocol.
<table><tr><td>ID</td><td>Removed component</td><td>Params</td><td>Acc. (%)</td><td>∆(pp)</td></tr><tr><td>V0</td><td>Full model (baseline)</td><td>296,780</td><td>99.04</td><td>reference</td></tr><tr><td>V1</td><td>Channel attention</td><td>289,100</td><td>98.93</td><td>-0.11</td></tr><tr><td>V2</td><td>Spatial attention</td><td>265,420</td><td>98.93</td><td>-0.11</td></tr><tr><td>V3</td><td>Both attentions</td><td>257,740</td><td>98.88</td><td>-0.16</td></tr><tr><td>V4</td><td>Multi-Scale Depthwise Block</td><td>465,420</td><td>98.88</td><td>-0.16</td></tr><tr><td>V5</td><td>Grouped convolutions</td><td>418,316</td><td>98.88</td><td>-0.16</td></tr><tr><td>V6</td><td>Dual pooling</td><td>280,396</td><td>99.04</td><td>0.00</td></tr><tr><td>V7</td><td>Swish, replaced by ReLU</td><td>296,780</td><td>98.77</td><td>-0.27</td></tr><tr><td>V8</td><td>Augmentation</td><td>296,780</td><td>99.20</td><td>+0.16</td></tr><tr><td>V9</td><td>l2 regularization</td><td>296,780</td><td>99.20</td><td>+0.16</td></tr><tr><td>V10</td><td>Dropout</td><td>296,780</td><td>98.93</td><td>-0.11</td></tr></table>

Table 14: Generalization of the unmodified AgroVisNet architecture to two independently collected datasets and to the merged 36-class corpus of all three.
<table><tr><td>Dataset</td><td></td><td></td><td>Cls. Images Params Acc. (%)</td><td>F1 (%)</td></tr><tr><td>BD-PlantDX (ours)</td><td></td><td>12 12,432 290,572</td><td>99.52</td><td>99.52</td></tr><tr><td>VegNet-BD</td><td>21</td><td>12,786291,157</td><td>98.71</td><td>98.71</td></tr><tr><td>RadishLeaf-BD</td><td>5</td><td>2,801 290,117</td><td>99.05</td><td>99.05</td></tr><tr><td>Merged corpus (3 datasets)</td><td>36</td><td>28,019 292,132</td><td>99.19</td><td>99.19</td></tr></table>

Table 15: Robustness of AgroVisNet to five synthetic input corruptions on the BD-PlantDX test split (Section 6.9). Severity 0 is the uncorrupted image at 99.57% accuracy.
<table><tr><td>Corruption</td><td>Acc. at sev. 5 (%)</td><td>Mean acc. sev. 1–5 (%)</td><td>Mean drop (pp)</td></tr><tr><td>JPEG compression</td><td>67.81</td><td>91.85</td><td>7.72</td></tr><tr><td>Gaussian blur</td><td>64.06</td><td>90.40</td><td>9.18</td></tr><tr><td>Brightness decrease</td><td>23.21</td><td>65.88</td><td>33.69</td></tr><tr><td>Additive noise</td><td>20.16</td><td>63.37</td><td>36.20</td></tr><tr><td>Brightness increase</td><td>23.85</td><td>56.40</td><td>43.18</td></tr></table>

Table 16: Faithfulness metrics for attribution maps (200 test images; 50 for random control).
<table><tr><td>Method</td><td>(%) ↓</td><td>(%) ↑</td><td>Avg. drop Increase Del. AUC Ins. AUC ↓</td><td>↑</td></tr><tr><td>Grad-CAM</td><td>41.52</td><td>0.50</td><td>0.356</td><td>0.932</td></tr><tr><td>Grad-CAM++</td><td>61.62</td><td>0.00</td><td>0.340</td><td>0.925</td></tr><tr><td>Random (control)</td><td></td><td></td><td>0.225</td><td>0.902</td></tr></table>