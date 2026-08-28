# Learning Woody Clearing with Loss Alignment for Zero-Shot Regrowth and Woody Segmentation

Kal Backman, Jared Wood and Adam Roff

Abstract—Detecting woody clearing is vital for managing biodiversity. Deep learning models can detect change in woody vegetation from bitemporal remote sensing imagery, however generated products may not meet end-user specifications due to unaligned loss definitions. Further limitations of deep learning models are the reliance on large datasets which can be difficult to attain for spatially rare and ambiguous events such as regrowth detection. In this work we train a model to detect woody change using bitemporal Sentinel-2 imagery consisting of 7 years’ worth of annual imagery across the state of New South Wales, Australia. To align the objective of the model with end-user metrics, we introduce the loss scaling coefficient α which transforms the objective to optimize for specific F<sub>β</sub> scores. Introducing α was found to increase precision by 1.85× or recall by 1.12×. We propose input imagery augmentation and generation techniques that allow the woody change detection model to zero-shot transfer to regrowth and woody segmentation tasks. For woody segmentation, image generation techniques using activation maximization with low α values for stability and image generation techniques derived from handcrafted features utilizing a mosaic of clearing patches and artificial trees for contextual grounding were found to outperform prior woody segmentation works of the study area, reducing the overall error by up to 18.2%. For zero-shot woody regrowth, creating pseudo-post and prior images resulted in the model achieving an F1 score of 0.845, creating a foundation for future regrowth detection work.

Index Terms—Change detection, zero-shot transfer, loss alignment, class imbalance, remote sensing

## I. INTRODUCTION

Land clearing is the leading cause of terrestrial biodiversity loss [1]–[4]. With the continual growth in the demand for agricultural land [5], [6], natural resources [1] and urbanization [7], [8], it is more imperative than ever that we enact land clearing reform to protect biodiversity [9], [10]. In order to protect current biodiversity, timely and accurate data of land clearing trends and extent are required to make informed decisions [11], [12].

Land clearing detection within the context of remote sensing stems from the broader domain of change detection, which aims to denote changes in the semantic classes of pixels across time within remote sensing imagery. Change detection has shown success in many applications including post disaster assessment [13]–[15], urbanization and infrastructure expansion [16]–[18] and land cover / land use change monitoring [19]–[22].

Traditional change detection methods tend to operate on a single pixel level, comparing spectral indices and individual bands across the time-series to make decisions [23]–[26]. However such traditional single pixel methods ignore key contextual information, resulting in noisy predictions that contain holes in continuous regions denoted as change [27], [28].

Deep learning approaches circumvent this singular pixel context window limitation by pooling information spatially at different spatial resolutions to attain a greater context window. Common deep learning methods focus on using convolutional neural networks (CNN) [29]–[31], self-attention [32]–[34] or Mamba [35], [36] architectures to detect change within an image.

The behavior of deep learning approaches are defined by the dataset they are trained on and the loss function used to map input imagery to output segmentation masks. The cross-entropy loss is a commonly used loss function for remote sensing change detection tasks [33], [34], [37], [38]. However for change detection tasks focusing on heavily class imbalanced datasets, a combination of the cross-entropy loss and dice loss is preferred [29], [32], [39] due to the dice loss not being strongly biased towards the majority class [40].

However such loss functions may not produce input-output mappings that are aligned with the objectives of the enduser, where an essential quality for successful remote sensing products is that they provide the relevant and accurate insights to the user [41], [42]. These relevant end-user accuracy definitions can be biased towards optimizing a specific quality such as recall at the cost of precision, as seen in operational programs utilizing region proposal models for spatially rare events that require skilled interpreters to edit, verify and evaluate all published remote sensing products [43]. By default, the aforementioned loss functions are unable to bias the model to prioritize recall or precision over the other. Although confidence thresholding can be used as a post processing operation to bias predictions, neural networks are known to suffer from overconfidence issues [44]–[46]. This overconfidence causes the bulk of predictions to be concentrated in the tails of the confidence distribution, leading to confidence thresholds with extremely low margins of error to attain the desired model performance.

Another issue with deep learning approaches for change detection is the vast amounts of data required to train a suitable model. This is further challenging for change detection tasks that aim to detect rare events due to the limited data available. Although transfer learning can be used to minimize data requirements [47], [48], an initial dataset is still required. Obtaining an initial dataset for tasks such as woody regrowth is challenging, unlike woody clearing which often involves discrete abrupt events, woody regrowth is a visually subtle and continuous process occurring over several years. Despite decades of efforts, comprehensive and consistent regrowth mapping remains limited [49], [50], preventing deep learning approaches from being applied to regrowth mapping due to the absence of sufficient training data [51].

Zero-shot learning aims to circumvent the initial dataset requirement by transferring trained models to unseen tasks without additional training or data. Zero-shot learning allows deep learning models to complete tasks where training data is insufficient such as woody regrowth detection by leveraging the abundance of similarly distributed datasets such as woody clearing.

In this work we aim to train a deep learning model that detects woody vegetation clearing using bitemporal geospatially aligned Sentinel-2 imagery. For the purpose of this work, woody vegetation is defined as vegetation over the height of 2m. The objective of the model is to optimize for $F _ { \beta }$ scores of large $\beta$ values, i.e. prioritize recall of woody clearing over precision. To align the model’s learning objective with the end user’s objective of maximizing recall, the loss scaling coefficient α was introduced to allow the learning process to explicitly target a specific metric during training.

The proposed model comprises of a CNN twin encoder decoder network that outputs a segmentation mask denoting woody vegetation that is absent in the post image relative to the prior image. To further improve woody vegetation monitoring, the proposed model was transferred in a zero-shot manner to woody regrowth and woody segmentation tasks by generating and augmenting input imagery to extract latent information from within the network. The proposed work aims to build a foundation for further regrowth detection work by establishing a method that can initially propose regrowth regions to address the difficulty in attaining woody regrowth data.

The main contributions of the proposed work are:

1) The development of a woody clearing network trained and validated on a large dataset consisting of 7 years worth of imagery covering a landmass of 801,137 km<sup>2</sup> and containing 54 billion data points.

2) The proposal of the loss scaling coefficient α that modifies the binary cross entropy and dice loss to align the training objective towards end user orientated metrics. Capable of increasing precision or recall by a factor of 1.85× or 1.12× respectively in the woody clearing task and increasing the F1 score by 1.32× for the zero-shot woody segmentation task.

3) Input imagery augmentation and generation techniques that allow a model trained purely on woody change detection data to zero-shot transfer to woody regrowth and woody segmentation tasks. Zero-shot woody segmentation was found to outperform prior woody segmentation maps of the study area with a reduction of the overall error by 18.2%, and the first work to build the foundation for woody regrowth for the study area due to the difficulties of establishing a curated dataset.

## II. METHODOLOGY

This section introduces the woody change detection architecture, the proposed loss function and the woody change detection dataset used for training and validation.

## A. Model architecture

The woody change detection architecture follows an encoder-decoder architecture comprising of three components: the encoder, the decoder and the head. The objective of the encoder is to extract multiscale features from bitemporal images pairs to describe the structure of the scene. The objective of the decoder is to fuse information across different temporal and spatial scales to detect regions of woody change. The objective of the head is to produce output confidence scores that denote whether a corresponding pixel in the bitemporal input pair contains woody clearing. An overview of the model architecture can be seen in Fig. 1.

1) Encoder: The encoder employs a Siamese convolutional neural network that independently extracts multiscale features from the bitemporal image pair. The encoder receives two spatially aligned images: $X _ { t 0 } \notin \mathrm { \& \ } X _ { t 1 }$ , where $X _ { t 0 }$ denotes the image taken prior to $X _ { t 1 }$ while $X _ { t 1 }$ denotes the image post of $X _ { t 0 }$ . Both $X _ { t 0 } \notin { \cal X } _ { t 1 }$ are passed through identically weighted backbones which output a total of 5 feature maps denoted as $F _ { t i } ^ { j } .$ where i refers to the input imagery time i.e. $X _ { t i }$ and j denoting the depth of the feature map. Features $[ F _ { t i } ^ { 1 } , F _ { t i } ^ { 2 } , F _ { t i } ^ { 3 } .$ $F _ { t i } ^ { 4 } , F _ { t i } ^ { 5 } ]$ contain [64, 128, 256, 512, 1028] output channels at resolution $[ \frac { 1 } { 2 } , \ \frac { 1 } { 4 } , \ \frac { 1 } { 8 } , \ \frac { 1 } { 1 6 } , \ \frac { 1 } { 3 2 } ]$ relative to the input imagery.

The initial feature map is produced by the backbone’s stem which receives the input imagery and applies an initial convolutional layer with stride length of 2, followed by 3 consecutive convolutional blocks which comprise of batchnorm, SiLU activation and convolutional layers. Successive feature maps are generated by receiving the prior feature map as input and applying a max pooling operation followed by 4 consecutive bottleneck residual blocks [52] for parameter efficiency.

![](images/bac2072ab39fc84a0c5774658b805ff6b45282a26269f3ca77f264e19734bfe9.jpg)  
Fig. 1. Overview of the woody change detection model architecture. The encoder individually extracts features from the prior and post images at different resolution scales using convolution-based bottleneck blocks. The extracted features are subsequently fused to encode woody change at each resolution scale. The lowest resolution feature is progressively upscaled and fused with the subsequent higher resolution features to produce woody change detection predictions at an identical resolution as the input imagery.

2) Decoder: The decoder receives the 5 feature maps from both bitemporal images $( \boldsymbol { F } _ { t 0 } ^ { j } \ \& \ \boldsymbol { F } _ { t 1 } ^ { j } )$ and initially fuses $F _ { t 0 } ^ { 5 }$ & $F _ { t 1 } ^ { 5 }$ via a fusion block comprising of an initial concatenation layer followed by a bottleneck residual block with a convolutional block to facilitate a skip connection due to the reduction in channel dimensionality. The result is fed through the decoder block which comprises of an initial up-sampling operation to match the resolution of the next feature map, followed by 2 consecutive bottleneck residual blocks, where the intermediate output is denoted as $\hat { F } _ { t 0 , t 1 } ^ { 5 }$ The decoder subsequently fuses the next set of bitemporal features which are then fused with the prior feature stage i.e. Fuse(Fuse( $F _ { t 0 } ^ { j - 1 } , F _ { t 1 } ^ { j - 1 } ) , \bar { F } _ { t 0 , t 1 } ^ { j } )$ . The fused features are passed through the decoder block and the process is repeated until the subsequent feature map resolution matches the original input image resolution.

3) Head: The head receives the final output from the decoder and applies a single convolutional block followed by a sigmoid activation to generate confidence scores that a pixel in image $X _ { t 1 }$ contains woody clearing relative to image $X _ { t 0 }$ which is denoted as $\hat { Y } _ { t 0 , t 1 }$ . Output $\hat { Y } _ { t 0 , t 1 }$ can be subsequently thresholded to a set confidence value to produce a binary mask indicating woody clearing.

## B. Loss function

The greatest challenge in training a network to segment woody clearing from remote sensing imagery is the extreme class imbalance from the scarcity of woody clearing. Woody clearing can constitute 0.05% of total pixels as seen in section II-C3, resulting in a class imbalance of 1 positive pixel per 2,000 negative pixels. Despite its popularity for segmentation tasks [33], [34], [37], [38], the binary cross-entropy loss heavily biases the objective function towards the majority class due to equally weighting the minimization of log-probabilities across all pixels. This leads to the network getting stuck in a sub-optimal local minima due to prioritizing the minimization of a large sum of small errors in the majority class, over a small sum of large errors in the minority class. The dice loss on the other hand computes the overlap between the target and the prediction, making it more robust to class imbalances [40] due to penalizing the network on false positives and false negatives. However the dice loss is prone to over fixating on erroneous labels in batches containing few positive samples. Therefore a combination of the binary cross-entropy loss $( L _ { B C E } )$ and dice loss $( L _ { D i c e } )$ is commonly used for class imbalanced applications [29], [32], [39], which can further be weighted for cases of extreme class disparity:

$$
L _ { T o t a l } = w _ { B C E } \cdot L _ { B C E } + w _ { D i c e } \cdot L _ { D i c e } .\tag{1}
$$

The dice loss is defined as:

$$
L _ { D i c e } = 1 - \frac { 2 \displaystyle \sum _ { i = 1 } ^ { n } Y _ { i } \hat { Y _ { i } } } { \displaystyle \sum _ { i = 1 } ^ { n } Y _ { i } + \sum _ { i = 1 } ^ { n } \hat { Y _ { i } } } ,\tag{2}
$$

which can be further generalized to the expression:

$$
L _ { D i c e } = 1 - \frac { 2 \displaystyle \sum _ { i = 1 } ^ { n } Y _ { i } \hat { Y _ { i } } w _ { i } + \epsilon } { \displaystyle \sum _ { i = 1 } ^ { n } Y _ { i } w _ { i } + \sum _ { i = 1 } ^ { n } \hat { Y _ { i } } w _ { i } + \epsilon } ,\tag{3}
$$

where $w _ { i } \in [ 0 , 1 ]$ denotes a weighting score which can be used to reflect the confidence in the validity of the given label $Y _ { i }$ while $\epsilon = 0 . 0 0 1$ is used for stability to prevent small division errors. Similarly the generalized binary cross-entropy loss can be expressed as:

$$
L _ { B C E } = - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( Y _ { i } \log ( \hat { Y _ { i } } ) + ( 1 - Y _ { i } ) \log ( 1 - \hat { Y _ { i } } ) ) w _ { i } .\tag{4}
$$

Observing equations 3 & 4, it can be seen that both loss functions equally weight the optimization of both precision and recall metrics. For the dice loss this is achieved via dividing by the sum of ground truths $( Y _ { i } )$ which restricts the prevalence of false negatives, therefore boosting recall. Dividing by the sum of predictions $( \hat { Y } _ { i } )$ restricts the prevalence of false positives, thereby boosting precision. For the binary cross-entropy loss this is achieved via multiplying the ground truth with the log probability of the prediction $( Y _ { i } \log ( \hat { Y } _ { i } ) )$ which aims to enhance true positives, therefore boosting recall. The subsequent term $( \left( 1 - Y _ { i } \right) \log ( 1 - \hat { Y } _ { i } ) )$ aims to enhance true negatives, therefore boosting precision.

The equal optimization of both precision and recall aims to maximize the $F _ { 1 }$ score, a common metric to rank the performance of segmentation models and is defined in equation 9 in section III-A2. However for certain applications it may be crucial to prioritize precision or recall over the other. In such cases, higher confidence thresholds can be applied to attain greater precision whilst the inverse can be used to enhance recall. However neural networks are known to suffer from overconfidence issues [44]–[46], leading to an invalid representation of the true confidence and therefore making threshold adjustments unreliable. Instead, defining a loss function that allows users to explicitly bias the learning objective towards precision or recall to suit their application is required.

To allow for a tunable loss function such as to optimize for specific $F _ { \beta }$ scores, we introduce the tunable parameter $\alpha \in ( 0 , \infty )$ into the loss functions depicted in equations 3 & 4:

$$
L _ { D i c e } = 1 - \frac { 2 \displaystyle \sum _ { i = 1 } ^ { n } Y _ { i } \hat { Y _ { i } } w _ { i } + \epsilon } { \displaystyle \sum _ { i = 1 } ^ { n } Y _ { i } w _ { i } \alpha + \sum _ { i = 1 } ^ { n } \hat { Y _ { i } } w _ { i } \frac { 1 } { \alpha } + \epsilon } ,\tag{5}
$$

$$
L _ { B C E } = - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( Y _ { i } \log ( \hat { Y _ { i } } ) \alpha + ( 1 - Y _ { i } ) \log ( 1 - \hat { Y _ { i } } ) \frac { 1 } { \alpha } ) w _ { i } .\tag{6}
$$

Terms that favour recall are multiplied by α whilst terms that favour precision are multiplied by $\textstyle { \frac { 1 } { \alpha } }$ , such that for values $\alpha >$ 1, a greater priority is given to enhancing recall whilst for values $\alpha < 1$ , precision is favored. The use of α and $\textstyle { \frac { 1 } { \alpha } }$ allows for modifying each term by an identical factor to simplify the tuning process.

## C. Data

1) Raw input imagery: The input imagery used to train and validate the woody clearing architecture was based on 10m Sentinel-2 imagery captured across mainland New South Wales, Australia covering a total landmass of 801,137 km<sup>2</sup> per year. Sentinel-2 imagery was captured across the summer period, where the majority of images were sampled between November to early March across a 7 year period from 2018 to 2024.

The Sentinel-2 imagery was split into 109 roughly 100km × 100km scenes which had radiometric corrections applied to compute the JRSRP Sentinel-2 surface reflectance outlined in [53], [54], and cloud masking used to rank image quality. For each year, a new post image was selected based on the availability of cloud free imagery, taking the closest image in date to January 1, while the prior image for a given era was selected from the post image of the previous era. Some images were created from a mosaic of two or three images where cloud or incomplete captures meant clean imagery was not available. The Sentinel-2 imagery was subsequently cropped to a common image size and spatial boundary for each scene, such that all images within a scene could be stacked with each pixel corresponding to an identical physical location across the images.

2) Woody clearing labels: To generate woody clearing labels, regions of interest were initially proposed via a clearing regression index outlined in [23] and [43] which compares individual pixels in the prior and post images to compute a probability that the particular pixel contains woody clearing. The region of interest proposal model was designed to highly emphasize recall at the cost of precision as to not exclude woody clearing events.

A team of operators manually checked through all proposed regions of interest across the state for a given era and denoted which pixels within these regions of interest contained woody clearing and for what purpose were they cleared for (e.g. agricultural). A secondary check was subsequently performed by an independent operator to validate that the indicated woody clearing pixels were correct. During both manual woody clearing checks, operators are able to reference higher resolution auxiliary data sources to disambiguate woody clearing events. The process was repeated yearly for a total of 6 eras’ worth of labels beginning from (2018-2019) to (2023-2024) were obtained.

The labels across all eras were subsequently cropped to an identical image and spatial boundary such that they overlap with the clipped Sentinel-2 imagery.

3) Training and validation datasets: A dataset was generated to train the woody clearing segmentation model by initially extracting all 10m bands: 2 (blue 490nm), 3 (green 560nm), 4 (red 665nm) and 8 (NIR 842nm) and a single 20m band 12 (SWIR 2190nm) which was resampled to stack on top of the 10m bands. The extracted bands were normalized between [-1, 1] by computing a histogram and clipping the lowest and highest 2.5% of values. The new minimum and maximum values were used to linearly scale each band between -1 and 1. The normalized extracted bands were subsequently split into 400w × 400h tiles. The tiles were generated such that for all years for a given scene, each tile could be stacked so that each pixel corresponds to an identical physical location.

Labels were similarly split into 400w × 400h tiles which overlapped with the input imagery. Binary labels were formulated where pixels denoted as clearing were assigned a value of 1 whilst non-clearing pixels were assigned a value of 0. As woody clearing resultant from pine plantations were not required to be reported on, operators did not encode plantation clearing as woody clearing due to time constraints, resulting in woody clearing from plantations erroneously being encoded with a value of 0.

An additional weighting tile was generated to scale the loss functions outlined in equations 5 & 6. The purpose of the weight tile is to zero weight invalid regions where data may not be present and to weight the loss function to account for the confidence in the label source to minimize the impact of erroneous labels [55]. As all labels denoted as woody clearing have been manually validated by at least two human operators, it provides a highly trusted point of truth. Regions not flagged by the initial clearing regression index are not required to be manually validated and therefore may contain a greater rate of errors. To account for this, human verified clearing events are assigned a weighting of 1.0, whilst nonverified non clearing events are assigned a weighting of 0.1. To minimize the influence of the incorrect classification of woody clearing deriving from pine plantations, non-clearing pixels which returned a high-probability for clearing from the initial clearing regression index had their weight values further decayed by 50%. Invalid pixels such as those obstructed by cloud or pixels that fall outside of the boundaries of New South Wales were assigned a value of 0.0 due to being excluded from the labeling process.

Labels associated with eras (2018-2019) to (2022-2023) were assigned to the training dataset whilst the remaining era (2023-2024) was excluded and reserved for validation purposes only. The total dataset consisted of 284,215 tiles for training and 56,851 for validation, where the woody clearing present in the dataset was 0.05% of all pixels.

## III. EXPERIMENT DESIGN

This section introduces the experiment procedures to validate the proposed woody change detection architecture. The initial subsection outlines the model training and evaluation procedure to segment woody clearing. The subsequent subsection explores the model’s response to different loss scaling coefficients and their ability to optimize for specific $F _ { \beta }$ scores. The remaining two subsections explore methods that allow the model to be applied to zero-shot tasks such as woody segmentation and woody regrowth detection without the need for further training.

## A. Woody clearing detection

1) Training: The woody change detection model was trained by sampling prior and post image pairs alongside the accompanied label using a batch size of 128. To account for the scarcity of woody clearing within the dataset, 50% of data points within a batch were randomly sampled from a known set of tiles that contained woody clearing, whilst the remaining 50% were sampled randomly from the entire training dataset.

The sampled training tiles had a random combination of data augmentation techniques applied. Data augmentation techniques used can be grouped into 3 categories: those that alter the appearance of the base imagery, those that add noise to the imagery and those that geometrically distort the imagery. Data augmentation techniques used to alter the appearance of the base imagery include gamma correction transforms to artificially darken or brighten the image and band-shift transforms to artificially bring out different colors within the image. Noising transforms include adding normally distributed noise to the image and applying Gaussian blur operations. Geometric augmentations include flipping the image along the vertical and horizontal axis, rotating the image, scaling the image to become larger or smaller and shearing the image to skew its perspective. After data augmentations were applied, a randomly selected point on the image was chosen and cropped to the training input image size of 256w × 256h.

The model was trained for a total of 200,000 optimization iterations split into 64 epochs containing 3,125 optimization iterations. Loss scaling coefficient α was set to 1.0, whilst the binary cross entropy loss scaling coefficient w<sub>BCE</sub> was set to 0.1 and the dice loss scaling coefficient $w _ { D i c e }$ was set to 1.0. The Adam optimizer was used with an initial learning rate of 2e-4 and was linearly decayed to the final learning rate of 1e-5. Gradient clipping was performed using a maximum gradient norm of 1.0 and gradient scaling for mixed precision layers. The model was trained using a NVIDIA A100 80GB GPU.

2) Evaluation: The trained model was tasked with generating woody clearing predictions for entire scenes across the (2023-2024) era. The model received an input image size of 1024w × 1024h using a batch size of 32. The input image patch was swept across the image using an image overlap of 50%. A weighted average of prediction confidences was used to fuse overlapping segments, where pixels far away from the center of the image patch were weighted lower than those in the center due to the lack of contextual information around the borders.

The model was evaluated under 5 metrics: precision, recall, F1-score, F2-score and intersection over union (IoU) which are defined as follows:

$$
\mathrm { P r e c i s i o n } = { \frac { T P } { T P + F P } }\tag{7}
$$

$$
{ \mathrm { R e c a l l } } = { \frac { T P } { T P + F N } }\tag{8}
$$

$$
\mathrm { F 1 } = \frac { \mathrm { 2 } \cdot \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } }\tag{9}
$$

$$
\mathrm { F } 2 = { \frac { 5 \cdot \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { 4 \cdot \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } }\tag{10}
$$

$$
{ \mathrm { I o U } } = { \frac { T P } { T P + F P + F N } }\tag{11}
$$

T P denotes the number of true positives, where a true positive is defined as a prediction that is above a threshold confidence level that corresponds to a ground truth label of woody change.

FP, TN and FN denote the number of false positives, true negatives and false negatives respectively.

To better understand the distribution of errors, the aforementioned metrics were recomputed whilst considering segment matching and distance thresholds. Segment analysis aims to test the model’s ability to detect patches of clearing compared to individual pixels. Segments were computed by buffering all woody clearing pixels by 20m (2 pixels), where adjacently connected buffered pixels were grouped and assigned unique segment IDs. The segment IDs were subsequently transferred to the unbuffered pixels. Segments are computed for both the model’s predictions and the ground truth labels, resulting in each pixel in the unbuffered prediction and ground truth corresponding to a unique segment ID.

For segment analysis, true positives (TP) are defined as ground truth pixels that belong to a segment that overlaps with a predicted segment. False positives $( F P )$ are predicted pixels whose segment does not overlap with a ground truth segment. False negatives (FN) are ground truth pixels whose segment does not overlap with a prediction segment.

Distance analysis aims to test the model’s ability to detect areas containing woody clearing and to assess the spatial error in the predictions. For distance analysis, a true positive (T P) is defined as a ground truth pixel that is within a threshold distance to any predicted woody clearing pixel. False positives (FP) are predicted pixels that are not within a threshold distance to any ground truth pixel. False negatives (F N) are ground truth pixels that are not within a threshold distance to any predicted pixel.

Prior image  
Post image  
Prediction  
Overlay  
![](images/99a3292fe480e8796b4176d8f6753818fa5797f95d2744a666c928454e29197d.jpg)  
Fig. 2. Example predictions of the woody clearing model.

Example predictions of the trained woody clearing detection model can be seen in Fig. 2. Results of the woody clearing evaluation can be seen in section IV-A.

## B. Woody clearing loss scaling

1) Training: To assess the model’s response to the loss scaling coefficient α outlined in equations 5 & 6, the model trained in section III-A was further fine tuned using various α values ranging from $\frac { 1 } { 1 0 }$ to 10. For each of the 15 selected α values, the base model was trained for an additional 32 epochs (100,000 optimization iterations) using an identical methodology described in section III-A1, with the exclusion of the total epochs trained for and loss scale α.

2) Evaluation: Each of the 15 models were evaluated on the (2023-2024) era following the same procedure outlined in section III-A2. For each model, a confidence threshold value of 50% was used to compute TP, FP & FN. To better understand the relationship between precision and recall for various α values, the F1 and F2 score can be generalized to an $F _ { \beta }$ score defined as:

$$
F _ { \beta } = \frac { ( 1 + \beta ^ { 2 } ) \cdot \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { \beta ^ { 2 } \cdot \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } }\tag{12}
$$

Where $( \beta = 1 )$ is equivalent to an F1-score which equally weights precision and recall, whilst for values $( \beta > 1 )$ recall is favored over precision and conversely for $( \beta < 1 )$ where precision is favored over recall.

To establish a guideline for determining what α value best optimizes for a specific target metric, a curve was fit to the observed precision and recall values for each α conditioned model. The proposed models for estimating precision (P<sup>ˆ</sup>) and recall (R<sup>ˆ</sup>) with respect to α are defined as:

$$
\hat { P } = \operatorname { s i g m o i d } ( - \operatorname { s i g n } ( \alpha ^ { \prime } ) \cdot \log _ { 2 } ( \sqrt { | \alpha ^ { \prime } | + 1 } ) - \gamma _ { P } )\tag{13}
$$

and

$$
\hat { R } = \mathrm { s i g m o i d } ( \mathrm { s i g n } ( \alpha ^ { \prime } ) \cdot \log _ { 2 } ( \sqrt { | \alpha ^ { \prime } | + 1 } ) - \gamma _ { R } ) .\tag{14}
$$

Where $\alpha ^ { \prime }$ is a linearization transformation applied to α defined as

$$
\alpha ^ { \prime } = { \left\{ \begin{array} { l l } { \alpha - 1 , } & { { \mathrm { i f ~ } } \alpha > 1 } \\ { 1 - { \frac { 1 } { \alpha } } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{15}
$$

$\gamma _ { P }$ and $\gamma _ { R }$ are curve origin offsets defined as:

$$
\gamma _ { P } = \log _ { e } ( \frac { 1 } { P _ { \alpha 1 } } - 1 )\tag{16}
$$

and

$$
\gamma _ { R } = \log _ { e } ( \frac { 1 } { R _ { \alpha 1 } } - 1 ) ,\tag{17}
$$

where $P _ { \alpha 1 }$ and $R _ { \alpha 1 }$ denote the precision and recall scores for the default $\alpha = 1$ model.

Results of the loss scaling experiment can be seen in section IV-B.

## C. Woody clearing comparison

To evaluate the proposed model’s ability to generalize to regions outside of the study area, a comparison study was performed using the MapBiomas [49] dataset. The MapBiomas [49] project continuously maps vegetation, land use change and deforestation across Brazil, and provides date encoded polygons of deforestation alerts to signal regions experiencing woody clearing. A total of 6 Sentinel-2 scenes spread across Brazil were chosen for evaluation. Sentinel-2 image pairs were chosen based on the least cloudy image within a 3- month window from the end of 2022 and end of 2023 for the prior and post images respectively. Due to the absence of cloud free imagery, OmniCloudMask [56] was used to generate cloud masks which excluded cloudy pixels in the prior or post image from statistical analysis. Deforestation alert polygons were filtered to those spatially and temporally contained within the evaluation scenes’ prior and post images and were subsequently rasterized to overlay with the Sentintel-2 imagery. Model predictions for evaluation scenes were generated following the process outlined in section III-A2

Prior work that maps woody vegetation clearing is the Global Forest Change [57] program, which provides annual global maps of tree clearing. The Global Forest Change [57] clearing map was evaluated on the proposed validation dataset outlined in section II-C3 and the evaluation scenes used within the MapBiomas [49] dataset. The 30m Global Forest Change [57] clearing rasters corresponding to evaluation scenes were merged, clipped and projected to their respective scenes such that ground truth and Global Forest Change [57] predictions could be overlayed. To account for temporal discrepancies between ground truth labels and predictions, clearing predictions within the current and prior annual time period were used to compute evaluation metrics. The true positives for Global Forest Change [57] evaluation metrics used the sum of true positives contained within the annual clearing maps for both the current and prior time periods relative to the ground truth label. The false positives and false negatives for Global Forest Change [57] evaluation metrics used only the current annual clearing map for the ground truth label, using the true positives to mask potential false positives or false negatives. The approach ensures that the Global Forest Change [57] evaluation is not disadvantaged due to the ground truth label time period not aligning with Global Forest Change [57] annual reporting dates.

Results of the woody clearing comparison experiments can be seen in section IV-C.

## D. Zero-shot woody segmentation

To assess the model’s ability to generalize to unseen tasks, the model was deployed in a zero-shot fashion to create binary segmentation masks of woody vegetation without any training through visual prompting. To extract the model’s internal representation of woody vegetation which is formulated when independently analysing the prior and post images for the woody clearing task, the prior image $X _ { t 0 }$ was given a reference image for which the woody segmentation mask should be generated for, whilst the post image $X _ { t 1 }$ was generated to extract the model’s latent representation of woody vegetation.

To assess the optimal post image generation technique to extract latent woody vegetation information from the model, an experiment was performed which benchmarked 9 different post image generation techniques and their efficacy in zeroshot woody segmentation. The 9 image generation techniques include: (1) zero image input, where the post image was replaced as an empty zero array. (2) Zero feature input, where all features $( F _ { t 1 } ^ { 1 } , \ \bar { F } _ { t 1 } ^ { 2 } , \ F _ { t 1 } ^ { 3 } , \ F _ { t 1 } ^ { \bar { 4 } } , \ F _ { t 1 } ^ { 5 } )$ from the encoder were replaced with zero arrays. Both image generation techniques (1) & (2) aim to assess whether the absence of all features is sufficient to extract latent woody vegetation information from the network, or if additional context from the post image is required.

Image generation technique (3), empty scene, replaces the post image with an empty dirt patch which was tiled to fit the input resolution of 256w × 256h. (4) Clearing scene, replaces the post image with a collection of clearing patches which were rotated, flipped and stitched together to fit the input resolution of 256w × 256h. Image generation techniques (3) & (4) both aim to test if features in the post image are necessary to extract woody information compared to the absence of all features in methods (1) & (2). Method (4) tests whether features explicitly encoding clearing are required or if a generic scene with the absence of woody vegetation is sufficient as shown in method (3). Both image generation methods (3) & (4) utilize image patch tiling due to difficulties of obtaining 2.56km × 2.56km continuous regions of the desired features.

Image generation methods (5) & (6) aim to augment the post images in methods (3) & (4) respectively by replacing a 16-pixel wide border around the post image with the contents of the prior image. Image generation methods (7) & (8) alternatively augment methods (3) & (4) by pasting small, isolated patches of woody vegetation on top of the post images from a collection of 4 woody vegetation samples taken from a single scene. Methods (5) to (8) aim to assess whether additional contextual grounding of the post image is required by providing examples of woody vegetation to compare to. For methods (5) - (8), an additional weight mask was used to zero-weight predictions resultant from the border or from pasted woody vegetation samples. For methods (7) & (8), a total of 8 post images were generated with different placement of pasted woody vegetation such that each pixel across the 8 generated images contained a non-zero weighting.

Image generation methods (1) – (8) all demonstrate methods utilizing handcrafted features, which requires the operator to have a deep understanding of the dataset and model to construct an appropriate post image. To propose a more generalizable post image generation method that is not dependent on the specific context of the task, a post image was generated using activation maximization optimization.

Unlike traditional model training where the model’s weights are treated as parameters which are to be optimized to satisfy a loss function, images generated via activation maximization keep the model weights fixed and treat the input image as an array of parameters needing to be optimized. The objective of the optimization procedure is to find a post image $X _ { t 1 }$ that produces the maximum value of the model’s output prediction $\hat { Y } _ { i } ,$ , which is achieved using the mean squared error between the prediction $\hat { Y _ { i } }$ and the largest confidence value the model can output:

![](images/ad3f5e5813128eadd14f08f91dfc78a40a42bb212c7f4a0d1f5cc6b8a7ba359d.jpg)  
Fig. 3. Examples of the different post image generation techniques used to generate zero-shot woody predictions. Predictions were generated using a confidence threshold of 50% using a masked weighted average and 50% image overlap.

$$
L _ { A c t i v a t i o n } = \frac { \sum _ { i = 1 } ^ { n } ( \hat { Y } _ { i } - 1 . 0 ) ^ { 2 } } { n } .\tag{18}
$$

To constrain the optimization to prevent post images being generated containing large pixel intensities, a regularization

term is introduced to the loss function which applies the mean squared error to the current generated image pixel intensities:

$$
L _ { I n p u t } = \frac { \displaystyle \sum _ { i = 1 } ^ { n } ( X _ { t 1 i } ) ^ { 2 } } { n } .\tag{19}
$$

A post image was generated using the input data tiles collected from the training dataset described in Section II-C3. The input tiles were cropped to $2 5 6 \mathrm { w } \times 2 5 6 \mathrm { h }$ and fed into the model as the prior image, whilst the generated image was fed into the model as the post image using a batch size of 64. The loss and gradients were subsequently computed and the generated image’s pixel intensities were updated using stochastic gradient descent using an initial learning rate of 0.01 which was linearly decayed to 0.0001 across 20,000 optimization iterations.

Examples of the 9 proposed post image generation methods can be seen in Fig. 3.

1) Zero-shot woody segmentation evaluation: To assess the 9 proposed post image generation methods, a dataset of woody vegetation was created. 6 candidate scenes were chosen from spatially diverse regions across the state of New South Wales, Australia. From the 6 candidate scenes, randomly sampled patches were hand labeled denoting pixels containing woody vegetation and those that do not. Over 1M pixels were labeled, covering $1 0 8 \mathrm { k m ^ { 2 } }$ of landmass where 28% of pixels were considered woody vegetation.

Each of the 9 proposed post image generation methods were used to generate predictions of woody vegetation on the woody vegetation dataset using an input image size of $2 5 6 \mathrm { w } \times 2 5 6 \mathrm { h }$ A masked weighted average was used to generate predictions where masked regions include the prior image border applied to the post image, and the pasted woody vegetation within the post image. Predictions were generated using a 50% image overlap where the border post image generation methods (5) and (6) used a 50% image overlap of the valid unmasked region.

All handcrafted image generation approaches (1) – (8) used the default $\alpha = 1$ model to generate predictions. For image generation approach (9) activation maximization, due to the optimization process being explicitly aimed at generating a post image that denotes all pixels to be considered woody vegetation, using the default α = 1 model would lead to small precision scores. To counteract such adverse behaviour, the α $= \ \frac { 1 } { 1 0 }$ model which enhances precision at the cost of recall was used to attain a more even balance between precision and recall.

The models were evaluated under the metrics defined in III-A2 with the exception of the F2 score being replaced with the overall accuracy metric (OA) which is defined as:

$$
\mathrm { O A } = { \frac { T P + T N } { T P + F P + T N + F N } } .\tag{20}
$$

The F2 score is replaced by OA due to the woody clearing task being heavily class imbalanced, leading to non-meaningful high OA scores. As the woody segmentation dataset has a more even class balance and the woody segmentation task not focused on maximizing recall, the OA metric is favored.

Results of the woody segmentation experiment can be seen in section IV-D1.

2) Zero-shot woody segmentation comparison: To compare the proposed model’s zero-shot woody segmentation performance against prior work, SamGeo [58] was evaluated on the woody vegetation dataset. SamGeo’s [58] text-based pipeline utilizes GroundingDino [59] for zero-shot visual grounding coupled with SAM [60] object segmentation. Each image within the woody vegetation dataset was passed through the SamGeo’s [58] text-based pipeline with the prompt “Woody vegetation” to segment only woody vegetation.

Results of the zero-shot woody segmentation comparison can be seen in section IV-D1.

3) Zero-shot woody segmentation Fisher et al. point dataset: To benchmark the performance of the model’s ability to zero-shot the woody segmentation task in relation to prior works, an experiment was performed using the labelled point dataset from [61]. The [61] Fisher et al. dataset contained 6,648 points sampled using a stratified random approach across the state of New South Wales, Australia where each point was manually classified through visual interpretation using a Leica ADS40 airborne digital camera at 0.5m resolution captured in 2011.

Each of the 9 proposed post image generation methods were used to generate a complete woody vegetation map of New South Wales, Australia, covering 801,137 km<sup>2</sup> using the 2018 Sentinel-2 imagery. Each of the generation methods alongside the [61] Fisher et al. model were evaluated under 3 criteria: woody accuracy, non-woody accuracy and overall accuracy defined as:

$$
{ \mathrm { w o o d y ~ a c c u r a c y } } = { \frac { \displaystyle \sum _ { i = 1 } ^ { n } { \hat { Y } } _ { i } Y _ { i } } { \displaystyle \sum _ { i = 1 } ^ { n } { Y _ { i } } } } ,\tag{21}
$$

$$
\begin{array} { r } { \mathrm { n o n – w o o d y ~ a c c u r a c y } = \frac { \displaystyle \sum _ { i = 1 } ^ { n } \big ( 1 - \hat { Y } _ { i } \big ) \big ( 1 - Y _ { i } \big ) } { \displaystyle \sum _ { i = 1 } ^ { n } \big ( 1 - Y _ { i } \big ) } } \end{array}\tag{22}
$$

and

$$
{ \mathrm { o v e r a l l ~ a c c u r a c y } } = { \frac { \displaystyle \sum _ { i = 1 } ^ { n } { \hat { Y } } _ { i } Y _ { i } + \sum _ { i = 1 } ^ { n } { ( 1 - { \hat { Y } } _ { i } ) ( 1 - Y _ { i } ) } } { n } } ,\tag{23}
$$

where $\hat { Y _ { i } }$ and $Y _ { i }$ denote the binary prediction of the model using a 50% confidence threshold and the ground truth label respectively.

The results of the zero-shot woody segmentation on the [61] Fisher et al. dataset can be seen in Section IV-D2.

## E. Zero-shot woody regrowth detection

To further assess the model’s ability to perform binary zero-shot segmentation, the model was deployed with the objective to detect woody regrowth. For this work, binary woody regrowth is defined as pixels in the present that satisfy the woody definition criteria of vegetation over the height of 2m, which are resultant from clearing events in the past. To extract woody regrowth information from the network, the

Pseudo-prior image

input pairs $X _ { t 0 }$ and $X _ { t 1 }$ were swapped such that the original prior image became the post image and the original post image became the prior image. These switched input pairs are defined as the pseudo-prior and pseudo-post images.

To assess the model’s ability to zero-shot woody regrowth detection, an evaluation dataset was created. Unlike woody clearing, woody regrowth is visually subtle in appearance due to being a continuous process occurring over several years. In contrast to woody clearing which is derived from discrete events occurring on a specific date. Therefore to provide sufficient time for woody regrowth to occur, the pseudo-prior images were taken from the Sentinel-2 images for the year 2024, whilst the pseudo-post images were taken from 2018.

Despite the rarity of woody clearing, woody regrowth is even less prevalent to observe. Combined with the ambiguity of its non-discrete appearance, it becomes difficult to randomly assign regions for manual labeling due to the low probability of overlap and the difficulty for humans to naturally detect it whilst scanning the image. Therefore to systematically propose regions for human labeling, the woody clearing model trained with loss scaling coefficient α = 4 was used to propose scenes with high woody regrowth present. The $\alpha = 4$ model was chosen as an initial region proposal method due to its higher recall scores compared to the standard α = 1 model, making it more suitable for alerting operators which scenes to investigate for manual labeling.

A total of 5 scenes across New South Wales, Australia were proposed. Manual labeling focused on regions within the scene that were flagged by the model to contain woody regrowth. Over 490,000 pixels were labeled, covering $4 9 \mathrm { k m } ^ { \mathrm { { 2 } } }$ of landmass where 16% of pixels were considered woody regrowth.

Examples of the woody regrowth dataset can be seen in Fig. 4. Results of the zero-shot woody regrowth detection experiment can be seen in section IV-E.

1) Zero-shot woody regrowth comparison: To compare the proposed model’s zero-shot woody regrowth segmentation performance against prior work, Segment Any Change [62] was evaluated on the woody regrowth dataset. Segment Any Change [62] utilizes SAM [60] latent vectors from bitemporal image pairs to detect semantic changes in input imagery. Change predictions for each image within the woody regrowth dataset were generated for evaluation.

To derive additional insights into how much spatial contextual information the proposed approach utilizes when performing zero-shot woody regrowth segmentation, a rudimentary NDVI difference threshold baseline was proposed to compute the reliance on individual pixel spectral values. For each image pair within the woody regrowth dataset, the NDVI vegetation index for each pixel in each image was computed as:

$$
N D V I = \frac { N I R – R } { N I R + R } ,\tag{24}
$$

where a binary regrowth prediction for each pixel was determined by thresholding the NDVI difference between the post and prior images:

$$
\hat { Y } = ( N D V I _ { p o s t } – N D V I _ { p r i o r } ) > T h r e s h o l d .\tag{25}
$$

Ground truth  
![](images/396f465e4102e8f46106cecde8fe190b137694506158bd99908c9e622388f626.jpg)  
Fig. 4. Example images used to evaluate woody regrowth, where the pseudopost image derives from 2018 Sentinel-2 imagery whilst the pseudo-prior image derives from 2024 Sentinel-2 imagery.

To determine a suitable threshold to apply for an image pair, threshold values beginning from 0.0 to 2.0 using increments of 0.1 were applied to all images excluding the target image pair. From the threshold values, the threshold corresponding to the highest F1 score was subsequently applied to the excluded target image pair. The process of excluding a new target image pair was repeated until all image pairs had been assigned a threshold.

The results of the zero-shot regrowth comparison experiments can be seen in section IV-E.

## IV. RESULTS

## A. Woody clearing detection

The results for woody clearing detection are shown in table I at confidence thresholds ranging from 10% to 90%. The model attains a precision of 35.2% with a recall of 79.7% at a confidence level of 50% on a pixel-wise metric basis.

When comparing the pixel-wise results to the segment-wise results it can be observed that precision and recall increase to 56.9% and 97.5% respectively, resulting in an increase of precision by 1.6× and a reduction in missed detections by a factor of 8. This large decrease in missed detections indicates that the model excels at proposing regions where woody clearing is present, but experiences difficulties in precisely matching individual pixel predictions to human labels.

TABLE I  
PERFORMANCE METRICS FOR WOODY CLEARING DETECTION ON A PIXEL-WISE AND SEGMENT-WISE BASIS
<table><tr><td>Confidence</td><td>Precision</td><td>Recall</td><td>F1</td><td>F2</td><td>IoU</td></tr><tr><td colspan="6">Pixel-wise metrics</td></tr><tr><td>10%</td><td>0.289</td><td>0.830</td><td>0.429</td><td>0.604</td><td>0.273</td></tr><tr><td>25%</td><td>0.312</td><td>0.819</td><td>0.452</td><td>0.618</td><td>0.292</td></tr><tr><td>50%</td><td>0.352</td><td>0.797</td><td>0.488</td><td>0.636</td><td>0.323</td></tr><tr><td>75% 90%</td><td>0.390</td><td>0.775</td><td>0.519</td><td>0.647</td><td>0.351</td></tr><tr><td></td><td>0.409</td><td>0.763</td><td>0.533</td><td>0.651</td><td>0.363</td></tr><tr><td colspan="6">Segment-wise metrics</td></tr><tr><td>10%</td><td>0.463</td><td>0.983</td><td>0.630</td><td>0.803</td><td>0.459</td></tr><tr><td>25%</td><td>0.503</td><td>0.980</td><td>0.665</td><td>0.824</td><td>0.498</td></tr><tr><td>50%</td><td>0.569</td><td>0.975</td><td>0.718</td><td>0.853</td><td>0.561</td></tr><tr><td>75%</td><td>0.626</td><td>0.970</td><td>0.761</td><td>0.874</td><td>0.614</td></tr><tr><td>90%</td><td>0.653</td><td>0.967</td><td>0.780</td><td>0.882</td><td>0.639</td></tr></table>

C)  
TABLE II

PERFORMANCE METRICS FOR WOODY CLEARING DETECTION USING DISTANCE THRESHOLDS
<table><tr><td>Distance</td><td>Precision</td><td>Recall</td><td>F1</td><td>F2</td><td>IoU</td></tr><tr><td>0m</td><td>0.352</td><td>0.797</td><td>0.488</td><td>0.636</td><td>0.323</td></tr><tr><td>10m</td><td>0.451</td><td>0.871</td><td>0.594</td><td>0.734</td><td>0.423</td></tr><tr><td>20m</td><td>0.478</td><td>0.891</td><td>0.622</td><td>0.760</td><td>0.452</td></tr><tr><td>30m</td><td>0.492</td><td>0.901</td><td>0.636</td><td>0.772</td><td>0.467</td></tr><tr><td>40m</td><td>0.503</td><td>0.909</td><td>0.648</td><td>0.783</td><td>0.479</td></tr><tr><td>50m</td><td>0.510</td><td>0.914</td><td>0.655</td><td>0.789</td><td>0.487</td></tr><tr><td>60m</td><td>0.515</td><td>0.918</td><td>0.660</td><td>0.794</td><td>0.493</td></tr><tr><td>70m</td><td>0.520</td><td>0.923</td><td>0.665</td><td>0.799</td><td>0.498</td></tr><tr><td>80m</td><td>0.524</td><td>0.926</td><td>0.670</td><td>0.803</td><td>0.503</td></tr><tr><td>90m</td><td>0.528</td><td>0.930</td><td>0.674</td><td>0.807</td><td>0.508</td></tr><tr><td>100m</td><td>0.531</td><td>0.932</td><td>0.677</td><td>0.810</td><td>0.511</td></tr></table>

Metrics were computed under a confidence threshold of 50%. Spatial resolution of input imagery was 10m per-pixel.

This is further supported by observing the recall metric across different distance thresholds as shown in table II. By applying a 10m distance threshold which equates to a 1-pixel buffer, the amount of missed woody clearing by the model reduces by a factor of 1.6 and is further reduced by a factor of 1.9 when applying a 20m distance threshold.

The disparity between the model’s achieved precision and recall metric is further discussed in section V-A.

## B. Woody clearing loss scaling

The results of applying the loss scaling coefficient α with respect to model performance can be seen in Fig. 5. Observing Fig. 5-A & Fig. 5-B, as the loss scaling coefficient α increases, the model attains a greater recall at the cost of reduced precision. Whilst the inverse is achieved by reducing α to boost precision at the expense of recall. However observing both Fig. 5-A & Fig. 5-B it can be seen that a greater overall recall or precision can be attained by optimizing the loss scaling coefficient α compared to traditional confidence thresholding. A full table of precision and recall scores conditional on α and confidence thresholds can be seen in tables VII and VIII in appendix A.

Each model trained on a given α parameter was assessed on various $F _ { \beta }$ scores where $\beta$ ranges from $[ \frac { 1 } { 1 5 }$ , 15] which can be seen in Fig. 5-C. It can be see that as $\beta$ increases, the relative performance of models with greater α values also increases. This is due to higher $\beta$ values more heavily weight recall when assessing model performance which higher α models optimize for. However given a specific target metric e.g. $F _ { 4 }$ score, there exists an optimal α value that optimizes for the target metric. The data derived optimal α values for given $\beta$ values are shown as the green curve in Fig. 5-C.

![](images/a86b08c3c9afa4ac6bfff50825c0710235abd6a7220d436ac0d5d2f013009568.jpg)

![](images/dfc2a895601759fff8e97a3302eecd5ba11ad59ca70a3edb1d00dfa06ad57db4.jpg)

![](images/45a6933823d667ec73eb6e1ec9a511f81098248aed85bbccc9e312886efe36c4.jpg)  
Fig. 5. Results from the woody clearing loss scaling experiment. Figures A) and B) plot the attained precision and recall metrics across confidence thresholds ranging from 5% to 95% against various α conditioned models where α ranges from $\textstyle { \frac { 1 } { 1 0 } }$ to 10. Figure C) plots the $F _ { \beta }$ scores attained for each α conditioned model where $\beta$ ranges from $\frac { 1 } { \frac { 1 5 } { / 3 } }$ to 15. The green line indicates the maximum $F _ { \beta }$ score for a specific β value, highlighting the optimal α value which maximizes the particular $F _ { \beta }$ score.

![](images/40aec317b8cc112cdf8f7ddb54abe868c4ab7ea872a362c97e63a7cc90883b68.jpg)

![](images/511ccf434d724552ea9832256d9046b4763c39ef4f49cf335ddd11f48f05a9cd.jpg)  
Fig. 6. Plots comparing the observed precision and recall metrics for the woody clearing detection task to that of the estimated precision and recall using equations 13 and 14

A comparison between the observed precision and recall scores attained by the α conditioned models and the estimated scores via equations 13 and 14 can be seen in Fig. 6. Terms $P _ { \alpha 1 }$ and $R _ { \alpha 1 }$ are assigned a value of 0.352 and 0.797 respectively and are derived from table I.

From Fig. 6, it can be observed that the curves estimating the precision and recall scores closely align to the shape of the observed values. The average error between each of the observed points and predicted points for precision and recall are 0.025 and 0.016 respectively, where the majority of the error deriving from lower value α<sup>′</sup> estimates.

Additional plots comparing observed precision and recall scores against predicted precision and recall scores for the zero-shot woody segmentation and regrowth detection tasks can be seen in Fig. 10 and Fig. 11 in appendix B. Further plots comparing the observed precision and recall scores against alternative scaled loss functions can be found in Fig 12 in appendix C.

## C. Woody clearing comparison

The results of the proposed model and Global Forest Change [57] on the MapBiomas [49] dataset can be seen in table III.

The proposed model achieved a recall score of 0.735, a 1.5× improvement compared to the Global Forest Change [57] recall score of 0.494. However, both models attained relatively low precision scores of 0.166 and 0.084, reducing the F1 scores of the proposed model and Global Forest Change [57] to 0.270 and 0.144 respectively. The lower precision score is attributable to missed clearing events within the MapBiomas [49] dataset, where the majority of false positives from both the proposed model and Global Forest Change [57] were the result of correct detections of clearing events being mislabeled.

Global Forest Change [57] attained a precision, recall and F1 score of 0.158, 0.268 and 0.199 respectively across the New South Wales study area. Compared to the proposed model which achieved a 2.2×, 3.0× and 2.45× increased performance in precision, recall and F1 scores respectively.

A comparison of generated predictions and missed clearing events on the MapBiomas [49] dataset can be seen in Fig. 7 in section V-E.

## D. Zero-shot woody segmentation

1) Zero-shot woody segmentation evaluation: The model’s performance using different post image generation techniques for the zero-shot woody segmentation task can be seen in table IV. The greatest performing model was the clearing scene + trees post image generation method with an F1 score of 0.899. Handcrafted post image generation techniques that used real reference scenes performed greater than those that used zeroed inputs. On average, the post image generation techniques utilizing a clearing scene as reference attained an F1 score twice as great compared to those using the empty scene. Adding additional contextual woody vegetation further improved model performance, whereby randomly placing trees across the post image was shown to increase the base clearing scene F1 performance by a factor of 2.

TABLE III  
WOODY CLEARING PERFORMANCE ON MAPBIOMAS DATASET
<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td>F1</td><td>F2</td><td>IoU</td></tr><tr><td>Proposed approach</td><td>0.166</td><td>0.735</td><td>0.270</td><td>0.436</td><td>0.156</td></tr><tr><td>Global Forest Change</td><td>0.084</td><td>0.494</td><td>0.144</td><td>0.251</td><td>0.078</td></tr></table>

Bold values indicate the most optimal value for a given metric.

TABLE IV  
PERFORMANCE METRICS ON ZERO-SHOT WOODY SEGMENTATION
<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td>F1</td><td>IoU</td><td>OA</td></tr><tr><td>Zero image</td><td>0.829</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.723</td></tr><tr><td>Zero feature</td><td>0.666</td><td>0.058</td><td>0.106</td><td>0.056</td><td>0.731</td></tr><tr><td>Empty-scene</td><td>0.591</td><td>0.136</td><td>0.221</td><td>0.124</td><td>0.734</td></tr><tr><td>Empty scene + border</td><td>0.587</td><td>0.189</td><td>0.286</td><td>0.167</td><td>0.738</td></tr><tr><td>Empty scene + trees</td><td>0.684</td><td>0.375</td><td>0.485</td><td>0.320</td><td>0.779</td></tr><tr><td>Clearing scene</td><td>0.921</td><td>0.298</td><td>0.451</td><td>0.291</td><td>0.798</td></tr><tr><td>Clearing scene + border</td><td>0.874</td><td>0.524</td><td>0.655</td><td>0.487</td><td>0.847</td></tr><tr><td>Clearing scene + trees</td><td>0.904</td><td>0.895</td><td>0.899</td><td>0.817</td><td>0.944</td></tr><tr><td>Activation maximization</td><td>0.867</td><td>0.867</td><td>0.867</td><td>0.764</td><td>0.928</td></tr><tr><td>SamGeo</td><td>0.683</td><td>0.688</td><td>0.685</td><td>0.521</td><td>0.792</td></tr></table>

Metrics were computed under a confidence threshold of 50%. Bold values indicate the most optimal value for the given metric.

TABLE V  
ZERO-SHOT WOODY SEGMENTATION PERFORMANCE METRICS ON FISHER ET AL. POINT DATASET
<table><tr><td>Model</td><td>Woody accuracy</td><td>Non-woody accuracy</td><td>Overall accuracy</td></tr><tr><td>Zero image</td><td>0.000</td><td>1.000</td><td>0.536</td></tr><tr><td>Zero feature</td><td>0.132</td><td>0.978</td><td>0.585</td></tr><tr><td>Empty-scene</td><td>0.220</td><td>0.959</td><td>0.616</td></tr><tr><td>Empty scene + border</td><td>0.356</td><td>0.910</td><td>0.653</td></tr><tr><td>Empty scene + trees</td><td>0.468</td><td>0.891</td><td>0.695</td></tr><tr><td>Clearing scene</td><td>0.295</td><td>0.974</td><td>0.659</td></tr><tr><td>Clearing scene + border</td><td>0.756</td><td>0.930</td><td>0.849</td></tr><tr><td>Clearing scene + trees</td><td>0.883</td><td>0.899</td><td>0.892</td></tr><tr><td>Activation maximization</td><td>0.846</td><td>0.900</td><td>0.873</td></tr><tr><td>Fisher et al.</td><td>0.747</td><td>0.930</td><td>0.868</td></tr></table>

Metrics were computed under a confidence threshold of 50%. Bold values indicate the most optimal value for the given metric.

Compared to the automated post image generation technique activation maximization, the best performing handcrafted post image generation method shared comparable performance to each other. The best handcrafted approached outperformed the activation maximization approach by 1.6 percentage points in terms of the overall accuracy.

Further insights into the post image generation techniques and their shared difficulties are discussed in section V-C.

Compared to alternative works, SamGeo [58] achieved an F1 score of 0.685, where the best performing zero-shot configuration outperformed it by a factor of 1.3×. Comparison of generated predictions between SamGeo [58] and the Clearing scene + trees method can be seen in Fig. 8 in section V-E.

2) Zero-shot woody segmentation Fisher et al. point dataset: The results of the model’s zero-shot woody segmentation performance on the [61] Fisher et al. dataset can be seen in table V. Similar to the zero-shot woody segmentation results in Section IV-D1, the post image generation techniques which utilized a clearing scene and those that added additional contextual woody vegetation performed better than those which did not.

Relative to the [61] Fisher et al. model which was explicitly trained to detect woody vegetation, the proposed model using the post image generation techniques; activation maximization and clearing scene with isolated tree images, outperformed the [61] Fisher et al. model despite being zero-shot transferred. The [61] Fisher et al. model attained an overall accuracy of 86.8%, whilst the zero-shot woody segmentation model achieved an overall accuracy of 87.3% using activation maximization and 89.2% using the clearing scene with trees, a reduction in the error by 3.8% and 18.2% respectively.

TABLE VI  
PERFORMANCE METRICS ON ZERO-SHOT WOODY REGROWTH
<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td>F1</td><td>IoU</td><td>OA</td></tr><tr><td>Proposed approach</td><td>0.815</td><td>0.878</td><td>0.845</td><td>0.731</td><td>0.947</td></tr><tr><td>Segment Any Change</td><td>0.474</td><td>0.322</td><td>0.383</td><td>0.237</td><td>0.828</td></tr><tr><td>NDVI threshold</td><td>0.659</td><td>0.556</td><td>0.603</td><td>0.432</td><td>0.878</td></tr></table>

Bold values indicate the most optimal value for a given metric.

## E. Zero-shot woody regrowth detection

The model’s performance for the zero-shot woody regrowth segmentation task can be seen in table VI. At a confidence threshold level of 50% the model attains an F1 score of 0.845 and an overall accuracy of 94.7%.

Compared to Segment Any Change [62] and NDVI difference thresholding which attained an F1 score of 0.383 and 0.603 respectively, the proposed approach achieved a performance improvement by a factor of 2.2× and 1.4× respectively. Comparison of generated predictions between Segment Any Change [62] and the NDVI difference thresholding method can be seen in Fig. 9 in section V-E.

## V. DISCUSSION

## A. Woody clearing precision performance

When comparing precision and recall for the woody clearing detection task, table I shows a large disparity between both metrics’ performances. It was found that recall was greater than precision by a factor of 2.26 at a 50% confidence threshold, without applying any α scaling to bias the model towards either metric. Observing the precision and recall scores for the zero-shot woody segmentation and woody regrowth tasks in tables IV & VI, the large disparity between precision and recall does not exist, despite using identical model weights.

The cause of the poor performance of the precision metric in the woody clearing detection task is due to discrepancies of what is required to be reported as woody clearing when formulating the dataset’s labels. Woody clearing resultant from plantations were not required to be reported on and therefore were not present in the dataset’s labels. As plantations consist of large dense examples of woody clearing that comprise of many pixels, the model’s precision score is adversely affected due to any predicted clearing being incorrectly flagged as a false positive inside of plantations.

An additional factor that negatively impacts precision is the requirement that operators are only required to validate pixels derived from the initial woody clearing index. Despite the clearing index being calibrated to have a high recall at the cost of precision, small traces of false negatives can be found which causes operators to miss examples of woody clearing when formulating labels.

This discrepancy in what is reported as woody clearing from an operational perspective is further demonstrated in the MapBiomas [49] dataset. Both the proposed work and Global Forest Change [57] attained low precision scores within the MapBiomas [49] dataset due to non-reported clearing events deriving from minimum reporting area requirements, as seen in Fig. 7.

Compared to the woody segmentation and woody regrowth datasets, such semantic exclusion criteria from ignoring plantations do not exist and therefore present a more homogeneous performance between precision and recall.

## B. α loss scaling

Conditioning the binary cross-entropy and dice loss on the loss scaling coefficient α demonstrated to be an effective method for targeting the optimization of specific $F _ { \beta }$ scores. The ability to train a model that maximizes recall for a given minimal precision threshold allows for a remote sensing product aimed towards end user specifications.

Applying confidence thresholding alone was able to attain a precision score of 0.409 at 90% confidence threshold and a recall score of 0.830 at 10% confidence threshold as shown in table I. Further extreme confidence thresholding resulted in a precision score of 0.421 at 99.5% confidence threshold and a recall score of 0.838 at 0.5% confidence threshold, an increase in the precision and recall with respect to the 90% and 10% confidence thresholds of 2.9% and 0.9% respectively. Thus demonstrating confidence thresholding alone being an ineffective method to tune precision and recall scores as fine numerical precision is required due to deep learning models producing overconfident predictions.

By applying α loss scaling, precision was able to be increased from 0.359 at an α value of 1 and 50% confidence threshold to 0.785 at an α value of $\textstyle { \frac { 1 } { 1 0 } }$ and 50% confidence threshold. Recall was able to be increased from 0.778 at an α value of 1 and 50% confidence threshold to 0.937 at an α value of 10 and 50% confidence threshold. Resulting in an increase in the default precision and recall scores of 118.7% and 20.4% respectively at a 50% confidence threshold. Compared to confidence thresholding alone, introducing the α loss scaling coefficient was able to result in a model with a 1.85× higher precision or 1.12× higher recall.

α loss scaling was further found to be useful during zeroshot woody segmentation where the activation maximization post image generation technique alone resulted in a precision and recall score of 49.2% and 99.1% respectively when using the α = 1 model. By counteracting the activation maximization optimization objective of prioritising high recall scores with the prioritisation of precision scores for $\alpha < 1$ models, the $\textstyle \alpha = { \frac { 1 } { 1 0 } }$ model achieved a precision and recall score of 86.7%, a 1.32× increase in the F1 score when compared to the default α = 1 model.

However to align the model’s objective with end user’s objectives, an appropriate α value must first be estimated to condition the training process. Conditioning the model to optimize for a specific $F _ { \beta }$ score is dependent on the task and the dataset the model is deployed on as seen in Fig. 6, 10 & 11. Equations 13 & 14 demonstrate the general relationship between model performance and α where an initial precision and recall score on a validation dataset is required at the origin $\alpha = 1$ to calibrate for the specific task and dataset difficulty. An α value can be estimated to achieve a specific precision or recall score by solving for $\alpha ^ { \prime }$ in equations 13 & 14 which results in:

$$
\begin{array} { r } { \alpha _ { P } ^ { \prime } = \left\{ \begin{array} { l l } { 2 ^ { 2 \log _ { e } ( \frac { 1 - \hat { P } } { \hat { P } } ) - 2 \gamma _ { P } } - 1 } & { \mathrm { i f ~ } \alpha _ { P } ^ { \prime } > 0 } \\ { - 2 ^ { - 2 \log _ { e } ( \frac { 1 - \hat { P } } { \hat { P } } ) + 2 \gamma _ { P } } + 1 } & { \mathrm { e l s e } } \end{array} \right. } \end{array}\tag{26}
$$

and

$$
\alpha _ { R } ^ { \prime } = \left\{ \begin{array} { l l } { 2 ^ { - 2 \log _ { e } ( \frac { 1 - \hat { R } } { \hat { R } } ) + 2 \gamma _ { R } } - 1 } & { \mathrm { i f ~ } \alpha _ { R } ^ { \prime } > 0 } \\ { - 2 ^ { 2 \log _ { e } ( \frac { 1 - \hat { R } } { \hat { R } } ) - 2 \gamma _ { R } } + 1 } & { \mathrm { e l s e } } \end{array} \right. ,\tag{27}
$$

where $\alpha _ { P } ^ { \prime } \ \& \ \alpha _ { R } ^ { \prime }$ denotes the estimated $\alpha ^ { \prime }$ value required to attain the desired precision and recall scores respectively at a 50% confidence threshold. The resultant α value can be used to subsequently fine-tune the $\alpha = 1$ model to optimize for the intended metric, where further confidence thresholding can be performed to account for small discrepancies between the predicted α equations and the observed metrics.

## C. Zero-shot woody segmentation

For the zero-shot woody segmentation task, a common difficulty experienced across all the post image generation techniques were large areas of continuous woody vegetation not being detected. All generation techniques excluding the zero image method excelled at detecting isolated groups or rows of trees adjacent to non-woody scenery as seen in Fig. 3 in section III-D. However as the surrounding contextual information denoting the appearance of non-woody vegetation diminishes in the prior image, the predictions of woody vegetation within these areas also diminish.

This phenomenon is theorized to be caused due to the training dataset for woody clearing not containing examples of large continuous extents of clearing. These large continuous woody extents do not provide sharp distinct features such as edges which are commonly extracted by convolutional kernels to help discriminate between classes. Instead the homogeneous textures present in thick woody vegetation are similar to the homogeneity of wet grass paddocks and large bodies of water which can subsequently dry up between years, leading to the misleading appearance of woody clearing within confined image context windows. This visual similarity towards the negative class and the lack of large continuous examples of the positive class leads the model to resist producing predictions in regions of non-distinct features. By introducing isolated woody vegetation patches into the post image as reference, the model is grounded by providing many local comparisons as to what woody and non-woody features look like within the current context window.

The success of the greatest performing zero-shot woody segmentation approach, clearing scene + trees, is dependent on the user’s ability to construct artificial features to extract latent information from the model. Constructing an artificial post-image to extract woody segmentation information from a woody change detection model is feasible due to woody clearing leaving distinct land scar features and that discrete examples of woody vegetation for grounding the model can be easily extracted from existing imagery. However for more abstract change detection tasks such as post disaster assessment [13]–[15] or general land cover / land use change [19]–[22], creating such post-images to extract latent information may not be practical due to the difficulty associated with quantifying features such as infrastructure damage or a specific land use type into a discrete image form. In order for such zeroshot post-image augmentation techniques to not be limited by the user’s ability to construct an artificial image capable of extracting the desired latent information, the non-handcrafted approach using activation maximization was proposed at the cost of a reduction in zero-shot performance.

## D. Zero-shot woody regrowth detection

The proposed zero-shot approach enabled the detection of woody vegetation regrowth by utilizing a model trained exclusively on land clearing data, circumventing the need to acquire regrowth labels.

The benefits of the zero-shot approach for global forest monitoring are significant. Given the availability of land clearing datasets and Sentinel-2 imagery, the proposed methodology offers a solution replicable globally. Regions with an established land clearing monitoring program can expand their monitoring capabilities from deforestation to reforestation in a cost-effective and scalable approach without the need to collect new regrowth annotations.

Accurate regrowth detection is critical for climate mitigation strategies, as it supports assessments of carbon sequestration, biodiversity recovery, and forest resilience, potentially correcting for underestimates in current carbon accumulation rates [63].

## E. Model comparison

Images comparing the proposed approach, Global Forest Change [57] and MapBiomas [49] can be seen in Fig. 7. The proposed approach excelled at small, thin areas of clearing resultant from roads and paths which often go undetected by the prior methods. Compared to prior works which tend to propose large continuous regions as clearing, the proposed work only flags pixels associated with woody clearing in areas that have undergone clearing operations. It was observed that the proposed model experienced a degradation in performance around areas of cloud cover. Although a cloud mask was used to ignore predictions associated with cloud cover and cloud shadow, the out of distribution features associated with the high and low intensity pixel values from bright clouds and dark shadows resulted in predictions in the immediate neighboring area to be incorrect. The exclusive use of high-quality cloud free imagery during the training stage resulted in the model becoming unrobust to such perturbations. Future work should supplement training datasets with a small sample of images

Proposed approach Global Forest Change

Prior image  
Post image  
MapBiomas  
![](images/4b17e60dc9b59754f165f7caa03e1dac995939802b5dd87b963e87b45bf0d3d8.jpg)  
Fig. 7. Example woody clearing predictions of the proposed approach, Global Forest Change [57] and MapBiomas [49] overlayed onto the Sentinel-2 post image.

containing cloudy imagery to climatize the model to wouldbe out of distribution features.

Images comparing the predictions of the proposed approach and SamGeo [58] in the zero-shot woody segmentation task can be seen in Fig. 8. The main limitation observed in the predictions generated by the SamGeo [58] model was the tendency to favor macro features such as dense forested regions whilst ignoring smaller features such as individual tree crowns. The proposed approach did not suffer from such discrepancies in the generated predictions and was able to correctly segment woody vegetation across all spatial scales.

Images comparing the predictions of the proposed approach, Segment Any Change [62] and NDVI difference thresholding in the zero-shot woody regrowth task can be seen in Fig. 9. Similar to SamGeo [58], Segment Any Change [62] was observed to favor macro features and over segment large continuous regions. The baseline NDVI difference thresholding method demonstrated spatial consistency issues with salt and pepper like noise in its generated predictions due to the lack of spatial information pooling. The NDVI difference thresholding method required a large discrepancy between NDVI pixel intensities from image pairs containing recently cleared areas and subsequently high density regrowth from plantations to operate effectively.

Across the zero-shot woody segmentation and zero-shot woody regrowth tasks, both SamGeo [58] and Segment Any Change [62] experienced difficulties segmenting small scale features. Both models utilize SAM [60] to implement zeroshot segmentation which is trained on a large dataset of 11 million images sourced from photographers. Despite SAM’s [60] impressive performance on traditional datasets, remote sensing imagery presents alternative challenges not contained within ground-level photography, where coarse 10m Sentinel-2 imagery does not provide the sharp distinct features shown within ground-level photography. Furthermore ground-level photography is constrained to traditional RGB image bands whereas remote sensing imagery leverages hyperspectral imagery to compute indices such as NDVI, NDWI and NDBI to aid in vegetation, water and urbanization classification. Higher resolution imagery that can provide greater detail and more complex features to match ground-level photography may aid such models in attaining greater performance.

Input image  
Proposed approach  
SamGeo  
![](images/8a3ad97ea8f894074c6c9c340aedc213324f9ba01e0b526c08391dc23f9f03d2.jpg)  
Fig. 8. Example woody segmentation predictions of the proposed approach and SamGeo [58] overlayed on Sentinel-2 imagery.

Pseudo-post image Pseudo-prior image Proposed approach Segment Any Change NDVI threshold  
![](images/18226b806d090b5e1bb9bcb8df67c774622b878d92bb1315894c62fa2828f8de.jpg)  
Fig. 9. Example woody regrowth predictions of the proposed approach, Segment Any Change [62] and NDVI difference thresholding overlayed onto the Sentinel-2 Pseudo-prior image.

## F. Limitations

The main limitation of the proposed work is that it is trained within a constrained geographical location, the state of New South Wales, Australia. Although the proposed work was evaluated outside the training study area, further work is required to both train and evaluate the proposed work in a greater global context to detect woody clearing, regrowth and woody segmentation.

Further limitations of the woody clearing architecture include the exclusion of clearing events deriving from cultivated vegetation such as plantations as discussed in section V-A. Further work is required to filter such data points to attain a more accurate understanding of the model’s precision statistic and to further promote the model’s ability to detect woody clearing events.

An additional limitation of the proposed work is the relatively small sample size evaluation dataset used to assess the zero-shot performance. Compared to the evaluation dataset for woody clearing which used an entire years’ worth of imagery, the evaluation dataset for zero-shot regrowth and woody segmentation amounted to less than 1% of the total samples used for woody clearing. Further work is required to establish a greater evaluation dataset for woody regrowth and segmentation to more rigorously assess the reliability and performance of the proposed work to operate in zero-shot conditions.

A further limitation of the proposed work was found within the zero-shot woody segmentation task where the model experienced difficulties predicting regions of large continuous forested regions due to the lack of contextual information. To alleviate this issue, future work should employ self-attention layers as seen in vision transformer architectures [32]–[34] to more efficiently pool spatial information, alongside greater training image sizes so that broadscale features can be captured.

The final limitation of the proposed work is the expected performance of zero-shot models compared to supervised models of comparable model complexity and dataset size. The main benefit of zero-shot transfer is the absence of needing to attain high quality input output annotation pairs to train a separate model, which may be costly or practically infeasible to attain. It is recommended that zero-shot transfer should not be used as a main source of generating remote sensing products, with the exception for cases where it is infeasible to attain sufficient training labels, but as a method of providing additional supplementary products at no additional cost.

## VI. CONCLUSION

In this work we propose a model to detect woody change from input bitemporal Sentinel-2 imagery. We introduce the loss scaling coefficient α that modifies the binary crossentropy and dice loss to favor the optimization of precision or recall to better align with end user specifications. Compared to traditional confidence thresholding, the α conditioned models were able to attain a 1.85× higher precision or 1.12× higher recall on the woody change dataset.

The proposed work demonstrates input image generation techniques that allow the woody change detection model to zero-shot transfer to woody segmentation tasks with no additional training through visual prompting. The most optimal handcrafted post image generation techniques were those that used a mosaic of clearing patches whilst introducing additional artificially pasted trees for contextual grounding. Automated post image generation techniques using activation maximization performed comparatively to the best performing handcrafted post image generation technique. Both handcrafted and automated post image generation techniques outperformed the [61] Fisher et al. model on the [61] Fisher et al. dataset, reducing the error by 18.2% and 3.8% respectively.

The proposed woody change detection model was further zero-shot transferred to the woody regrowth detection task where it attained an F1 score of 0.845. The proposed work provides a foundation for future woody regrowth detection work, where due to the rarity and ambiguity of woody regrowth, acquiring datasets to formulate and validate models are difficult.

## VII. ACKNOWLEDGMENTS

Labels and support provided by Vegetation Monitoring & Reporting Team, Remote Sensing and Regulatory Mapping. Computational power was provided by the Science Data and Compute facility. Both are part of the Science and Insights Division of the New South Wales Department of Climate Change, Energy, the Environment and Water.

## APPENDIX A

## WOODY CLEARING α-CONFIDENCE TABLES

The precision and recall scores for each model conditioned on α across confidence thresholds ranging from 5% and 95% can be seen in tables VII and VIII respectively.

## APPENDIX B

## ZERO-SHOT REGROWTH AND WOODY SEGMENTATION α CONDITIONED PRECISION AND RECALL

Plots comparing the observed precision and recall scores against predicted precision and recall scores for the zero-shot woody segmentation and regrowth detection tasks can be seen in Fig. 10 and Fig. 11 respectively.

The average error between the observed points and the predicted points for precision and recall for the zero-shot woody segmentation task was 0.017 and 0.079. The average error between the observed points and the predicted points for precision and recall for the zero-shot woody regrowth task was 0.043 and 0.025. The average error across all 3 tasks in estimating precision and recall was 0.028 and 0.040 respectively.

TABLE VII WOODY CLEARING PRECISION
<table><tr><td></td><td>10</td><td></td><td>118</td><td>1-6 1-4</td><td>1-3</td><td>1-2</td><td>3-4</td><td>1.00</td><td>4-3</td><td>2.0</td><td>3.0</td><td></td><td>4.0</td><td>6.0 8.0</td><td>10.0</td></tr><tr><td>10%</td><td>5%</td><td>0.706</td><td>0.673 0.640</td><td>0.562</td><td>0.503</td><td>0.423</td><td>0.331</td><td>0.286</td><td>0.240</td><td>0.182</td><td>0.135</td><td>0.107</td><td>0.077</td><td>0.063</td><td>0.052</td></tr><tr><td></td><td></td><td>0.719</td><td>0.687 0.652</td><td>0.570</td><td>0.510</td><td>0.429</td><td>0.337</td><td>0.291</td><td>0.245</td><td>0.186</td><td>0.139</td><td>0.110</td><td>0.080</td><td>0.066</td><td>0.055</td></tr><tr><td rowspan="12">Conce</td><td>15%</td><td>0.728 0.698</td><td>0.662</td><td>0.580</td><td>0.520</td><td>0.439</td><td>0.345</td><td>0.299</td><td>0.252</td><td>0.192</td><td>0.144</td><td>0.114</td><td>0.084</td><td>0.069</td><td>0.058</td></tr><tr><td>20%</td><td>0.737</td><td>0.708 0.671</td><td>0.588</td><td>0.528</td><td>0.447</td><td>0.353</td><td>0.306</td><td>0.258</td><td>0.198</td><td>0.148</td><td>0.118</td><td>0.087</td><td>0.072</td><td>0.061</td></tr><tr><td>25%</td><td>0.745</td><td>0.718 0.682</td><td>0.599</td><td>0.540</td><td>0.458</td><td>0.364</td><td>0.316</td><td>0.268</td><td>0.206</td><td>0.155</td><td>0.124</td><td>0.092</td><td>0.076</td><td>0.065</td></tr><tr><td>30%</td><td>0.758</td><td>0.734 0.699</td><td>0.616</td><td>0.558</td><td>0.475</td><td>0.378</td><td>0.330</td><td>0.280</td><td>0.215</td><td>0.163</td><td>0.131</td><td>0.097</td><td>0.080</td><td>0.068</td></tr><tr><td>35%</td><td>0.766</td><td>0.742 0.708</td><td>0.624</td><td>0.566</td><td>0.483</td><td>0.385</td><td>0.336</td><td>0.286</td><td>0.221</td><td>0.168</td><td>0.135</td><td>0.100</td><td>0.083</td><td>0.071</td></tr><tr><td>40%</td><td>0.772</td><td>0.749 0.715</td><td>0.631</td><td>0.573</td><td>0.490</td><td>0.392</td><td>0.342</td><td>0.291</td><td>0.225</td><td>0.172</td><td>0.139</td><td>0.103</td><td>0.086</td><td>0.074</td></tr><tr><td>45%</td><td>0.778</td><td>0.755</td><td>0.722 0.639</td><td>0.582</td><td>0.499</td><td>0.400</td><td>0.350</td><td>0.299</td><td>0.232</td><td>0.177</td><td>0.143</td><td>0.107</td><td>0.089</td><td>0.077</td></tr><tr><td>50%</td><td>0.785</td><td>0.764</td><td>0.731 0.649</td><td>0.592</td><td>0.508</td><td>0.409</td><td>0.359</td><td>0.307</td><td>0.239</td><td>0.183</td><td>0.148</td><td>0.111</td><td>0.093</td><td>0.080</td></tr><tr><td>55%</td><td>0.791</td><td>0.771</td><td>0.739 0.657</td><td>0.601</td><td>0.519</td><td>0.419</td><td>0.369</td><td>0.316</td><td>0.247</td><td>0.189</td><td>0.154</td><td>0.116</td><td>0.097</td><td>0.084</td></tr><tr><td>60%</td><td>0.799</td><td>0.780</td><td>0.748 0.667</td><td>0.612</td><td>0.529</td><td>0.429</td><td>0.378</td><td>0.325</td><td>0.255</td><td>0.196</td><td>0.160</td><td>0.121</td><td>0.101</td><td>0.087</td></tr><tr><td>65%</td><td>0.806</td><td>0.787</td><td>0.755 0.674</td><td>0.620</td><td>0.537</td><td>0.436</td><td>0.385</td><td>0.331</td><td>0.260</td><td>0.200</td><td>0.164</td><td>0.124</td><td>0.104</td><td>0.090</td></tr><tr><td>70%</td><td>0.810</td><td>0.792</td><td>0.760</td><td>0.679</td><td>0.625</td><td>0.542</td><td>0.441</td><td>0.390</td><td>0.336</td><td>0.265</td><td>0.204</td><td>0.168</td><td>0.127 0.107</td><td></td><td>0.093</td></tr><tr><td>75%</td><td>0.817</td><td>0.799</td><td>0.768</td><td>0.688</td><td>0.635</td><td>0.552</td><td>0.451</td><td>0.399</td><td>0.345</td><td>0.272</td><td>0.211</td><td>0.173</td><td>0.132</td><td>0.111</td><td>0.097</td></tr><tr><td>80%</td><td>0.822</td><td>0.805</td><td>0.774</td><td>0.696</td><td>0.643</td><td>0.561</td><td>0.460</td><td>0.408</td><td>0.353</td><td>0.279</td><td>0.217</td><td>0.179</td><td>0.136</td><td>0.115</td><td>0.100</td></tr><tr><td>85%</td><td>0.827</td><td>0.810</td><td>0.780</td><td>0.702</td><td>0.649</td><td>0.567</td><td>0.466</td><td>0.414</td><td>0.359</td><td>0.285</td><td>0.221</td><td>0.183</td><td>0.140</td><td>0.118</td><td>0.104</td></tr><tr><td>90%</td><td>0.831</td><td>0.815</td><td>0.785</td><td>0.706</td><td>0.654</td><td>0.573</td><td>0.471</td><td>0.419</td><td>0.364</td><td>0.289</td><td>0.225</td><td>0.187</td><td>0.143</td><td>0.122</td><td>0.108</td></tr><tr><td>95%</td><td>0.835</td><td>0.822</td><td>0.792</td><td>0.711</td><td>0.659</td><td>0.577</td><td>0.475</td><td>0.423</td><td>0.368</td><td>0.293</td><td>0.229</td><td>0.191</td><td>0.147</td><td>0.128</td><td>0.118</td></tr></table>

TABLE VIII WOODY CLEARING RECALL
<table><tr><td></td><td>1</td><td></td><td>118</td><td>#</td><td># 1#</td><td>1-2</td><td>24</td><td>1.00</td><td>4-3</td><td>2.0</td><td>3.0</td><td></td><td>4.0</td><td>6.0 8.0</td><td></td><td>10.0</td></tr><tr><td></td><td>5%</td><td>0.421</td><td>0.450</td><td>0.502</td><td>0.596</td><td>0.653</td><td>0.720</td><td>0.785</td><td>0.816</td><td>0.842</td><td>0.876</td><td>0.904</td><td>0.920</td><td>0.942</td><td>0.949</td><td>0.955</td></tr><tr><td rowspan="12">Conhnce</td><td>10%</td><td>0.409</td><td>0.438</td><td>0.491 0.590</td><td>0.648</td><td></td><td>0.716</td><td>0.782</td><td>0.813</td><td>0.839</td><td>0.874</td><td>0.902</td><td>0.918</td><td>0.941</td><td>0.947</td><td>0.953</td></tr><tr><td>15%</td><td>0.402</td><td>0.430 0.483</td><td>0.584</td><td>0.643</td><td></td><td>0.712</td><td>0.778</td><td>0.809</td><td>0.836</td><td>0.871</td><td>0.900</td><td>0.916</td><td>0.939</td><td>0.945</td><td>0.952</td></tr><tr><td>20%</td><td>0.396</td><td>0.423</td><td>0.476 0.578</td><td>0.638</td><td></td><td>0.707</td><td>0.774</td><td>0.806</td><td>0.833</td><td>0.868</td><td>0.897</td><td>0.914</td><td>0.937</td><td>0.944</td><td>0.950</td></tr><tr><td>25%</td><td>0.390</td><td>0.416</td><td>0.469 0.572</td><td>0.631</td><td>0.701</td><td></td><td>0.769</td><td>0.801</td><td>0.829</td><td>0.864</td><td>0.894</td><td>0.911</td><td>0.935</td><td>0.941</td><td>0.948</td></tr><tr><td>30%</td><td>0.383</td><td>0.408</td><td>0.461 0.563</td><td>0.623</td><td></td><td>0.693</td><td>0.762</td><td>0.794</td><td>0.822</td><td>0.859</td><td>0.889</td><td>0.907</td><td>0.931</td><td>0.938</td><td>0.945</td></tr><tr><td>35%</td><td>0.377</td><td>0.402</td><td>0.455 0.558</td><td>0.619</td><td></td><td>0.689</td><td>0.759</td><td>0.791</td><td>0.820</td><td>0.856</td><td>0.887</td><td>0.905</td><td>0.929</td><td>0.936</td><td>0.943</td></tr><tr><td>40%</td><td>0.373</td><td>0.397</td><td>0.450</td><td>0.553</td><td>0.614</td><td>0.685</td><td>0.755</td><td>0.788</td><td>0.816</td><td>0.853</td><td>0.884</td><td>0.903</td><td>0.927</td><td>0.934</td><td>0.941</td></tr><tr><td>45%</td><td>0.367</td><td>0.390</td><td>0.443</td><td>0.546</td><td>0.607</td><td>0.678</td><td>0.749</td><td>0.783</td><td>0.811</td><td>0.849</td><td>0.881</td><td>0.900</td><td>0.925</td><td>0.932</td><td>0.939</td></tr><tr><td>50%</td><td>0.361</td><td>0.384</td><td>0.436</td><td>0.539</td><td>0.601</td><td>0.672</td><td>0.744</td><td>0.778</td><td>0.807</td><td>0.846</td><td>0.877</td><td>0.897</td><td>0.922</td><td>0.930</td><td>0.937</td></tr><tr><td>55%</td><td>0.355</td><td>0.378</td><td>0.430</td><td>0.532</td><td>0.593</td><td>0.665</td><td>0.738</td><td>0.772</td><td>0.802</td><td>0.841</td><td>0.873</td><td>0.893</td><td>0.919</td><td>0.927</td><td>0.935</td></tr><tr><td>60%</td><td>0.349</td><td>0.372</td><td>0.423</td><td>0.526</td><td>0.587</td><td>0.659</td><td>0.732</td><td>0.768</td><td>0.798</td><td>0.838</td><td>0.869</td><td>0.890</td><td>0.917</td><td>0.925</td><td>0.933</td></tr><tr><td>65%</td><td>0.344</td><td>0.367</td><td>0.418</td><td>0.521</td><td>0.583</td><td>0.656</td><td>0.729</td><td>0.765</td><td>0.795</td><td>0.835</td><td>0.867</td><td>0.888</td><td>0.915</td><td>0.923</td><td>0.932</td></tr><tr><td>70%</td><td>0.340</td><td>0.361</td><td>0.413</td><td>0.516</td><td>0.578</td><td>0.651</td><td>0.726</td><td>0.761</td><td>0.792</td><td>0.832</td><td>0.865</td><td>0.885</td><td>0.913</td><td>0.921</td><td>0.930</td></tr><tr><td>75%</td><td>0.334</td><td>0.355</td><td>0.406</td><td>0.510</td><td>0.572</td><td>0.645</td><td>0.720</td><td>0.756</td><td>0.787</td><td>0.828</td><td>0.860</td><td>0.881</td><td>0.909</td><td>0.918</td><td>0.927</td></tr><tr><td>80%</td><td>0.328</td><td>0.349</td><td>0.400</td><td>0.503</td><td>0.565</td><td>0.639</td><td>0.715</td><td>0.751</td><td>0.783</td><td>0.823</td><td>0.856</td><td>0.877</td><td>0.906</td><td>0.916</td><td>0.925</td></tr><tr><td>85%</td><td>0.323</td><td>0.342</td><td>0.393</td><td>0.498</td><td>0.560</td><td>0.635</td><td>0.711</td><td>0.747</td><td>0.779</td><td>0.820</td><td>0.853</td><td>0.875</td><td>0.903</td><td>0.914</td><td>0.923</td></tr><tr><td>90%</td><td>0.318</td><td>0.335</td><td>0.386</td><td>0.493</td><td>0.556</td><td>0.631</td><td>0.708</td><td>0.744</td><td>0.777</td><td>0.818</td><td>0.851</td><td>0.872</td><td>0.901</td><td>0.911</td><td>0.920</td></tr><tr><td>95%</td><td></td><td>0.311</td><td>0.325 0.377</td><td>0.488</td><td>0.551</td><td>0.627</td><td>0.705</td><td>0.742</td><td>0.774</td><td>0.816</td><td>0.849</td><td>0.870</td><td>0.899</td><td>0.907</td><td>0.915</td></tr></table>

![](images/9bb0ca374f99b25c819a5663b37d8ab1789ec596491d4861bb7977be2ed999fb.jpg)

![](images/e99c213908e42dacf0e4410413f14fb2180db34f45a79792bf4ceb9815c14859.jpg)  
Fig. 10. Plots comparing the observed precision and recall metrics for the zero-shot woody segmentation task to that of the estimated precision and recall. Observed values were taken using the clearing scene + trees post image generation technique

![](images/396661fca585123b23e2c2dff0903d08a6a9d66a3178e53979b3252771d03e0b.jpg)

![](images/0d1ba4f77f38b7f3fd4153e1b266b4d7d2e2573dabcec2f89063b0572321f045.jpg)  
Fig. 11. Plots comparing the observed precision and recall metrics for the zero-shot woody regrowth detection task to that of the estimated precision and recall.

The majority of the observed errors for the predicted recall originate from lower $\alpha ^ { \prime }$ values from the woody segmentation task. Observing Fig. 10 it can be seen that at $\alpha ^ { \prime }$ values less than $^ { - 2 , }$ a sharp decrease in the recall performance can be observed. This is due to $\alpha ^ { \prime }$ values less than 0 favouring precision over recall, making the model less susceptible to the input generation techniques used to extract latent woody vegetation information from the model. This sudden drop in recall performance due to the breakdown of the input image generation technique is further exacerbated by the experienced difficulties generating predictions in large areas of continuous woody vegetation as discussed in section V-C. As these large continuous woody vegetation areas constitute a large proportion of the positive class, a slight decrease in their performance results in a large decrease in overall recall.

## APPENDIX C SCALED LOSS COMPARISON

The Tversky loss is an alternative loss function that scales the false positives and false negatives to favor precision or recall and is defined as:

$$
L _ { T v e r k y } = 1 - \frac { \displaystyle { \sum _ { i = 1 } ^ { n } Y _ { i } \hat { Y } _ { i } } } { \displaystyle { \sum _ { i = 1 } ^ { n } Y _ { i } \hat { Y } _ { i } + \varphi \sum _ { i = 1 } ^ { n } ( 1 - Y _ { i } ) \hat { Y } _ { i } + \gamma \sum _ { i = 1 } ^ { n } Y _ { i } ( 1 - \hat { Y } _ { i } ) } } ,\tag{28}
$$

where larger $\varphi$ values promote a reduction in false positives leading to higher precision scores whilst higher $\gamma$ values promote a reduction in false negatives leading to higher recall scores. The use of two parameters $\varphi$ and $\gamma$ results in a greater hyperparameter search space to find an optimal configuration, where altering either $\varphi$ or $\gamma$ impacts both precision and recall due to precision and recall being a tradeoff of one another. Compared to the proposed loss function defined in equations 5 & $^ { 6 , }$ a single parameter α is used to control the preference of recall or precision, leading to a clear signal as to what direction the hyperparameter α should be moved in to attain the desired performance.

To observe the difference in behavior between the proposed loss function and the Tversky loss, additional models were trained following the process outlined in section III-B1 where the dice loss was replaced with the Tversky loss. Due to the difficulty of searching a 2-dimensional state space compared to a 1-dimensional state space, the Tversky loss parameters $\varphi$ and $\gamma$ were constrained such that such that $\textstyle \varphi = { \frac { 1 } { \alpha } }$ and $\gamma = \alpha$ using the α values used in section III-B1. The results of the loss comparison experiment can be seen in Fig. 12.

Observing Fig. 12 it can be seen that the proposed loss function attains marginally greater recall scores whilst experiences significantly greater precision scores for lower $\alpha ^ { \prime }$ values.

![](images/05f45ec2fd7fc6c353eb2d1b60453b2d24dac16c47b38d5faa3ec1f5034e9fdc.jpg)

![](images/04ba5126c51c62e2823c5da5e330a98815ab69ed353c715ac1b54b96984784fd.jpg)  
Fig. 12. Plots comparing the proposed loss and the Tversky loss for the observed precision and recall metrics in the woody clearing detection task.

## REFERENCES

[1] S. D´ıaz, J. Settele, E. S. Brond´ızio, H. T. Ngo, J. Agard, A. Arneth, P. Balvanera, K. A. Brauman, S. H. M. Butchart, K. M. A. Chan, L. A. Garibaldi, K. Ichii, J. Liu, S. M. Subramanian, G. F. Midgley, P. Miloslavich, Z. Molnar, D. Obura, A. Pfaff, S. Polasky, A. Purvis,´ J. Razzaque, B. Reyers, R. R. Chowdhury, Y.-J. Shin, I. Visseren-Hamakers, K. J. Willis, and C. N. Zayas, “Pervasive human-driven decline of life on earth points to the need for transformative change,” Science, vol. 366, no. 6471, p. eaax3100, 2019.

[2] S. Legge, L. Rumpff, S. T. Garnett, and J. C. Woinarski, “Loss of terrestrial biodiversity in australia: magnitude, causation, and response,” Science, vol. 381, no. 6658, pp. 622–631, 2023.

[3] P. Jaureguiberry, N. Titeux, M. Wiemers, D. E. Bowler, L. Coscieme, A. S. Golden, C. A. Guerra, U. Jacob, Y. Takahashi, J. Settele et al., “The direct drivers of recent global anthropogenic biodiversity loss,” Science advances, vol. 8, no. 45, p. eabm9982, 2022.

[4] C. W. Davison, C. Rahbek, and N. Morueta-Holme, “Land-use change and biodiversity: Challenges for assembling evidence on the greatest threat to nature,” Global Change Biology, vol. 27, no. 21, pp. 5414– 5429, 2021.

[5] D. R. Williams, M. Clark, G. M. Buchanan, G. F. Ficetola, C. Rondinini, and D. Tilman, “Proactive conservation to prevent habitat losses to agricultural expansion,” Nature sustainability, vol. 4, no. 4, pp. 314– 322, 2021.

[6] D. Tilman, M. Clark, D. R. Williams, K. Kimmel, S. Polasky, and C. Packer, “Future threats to biodiversity and pathways to their prevention,” Nature, vol. 546, no. 7656, pp. 73–81, 2017.

[7] V. Kati, C. Kassara, Z. Vrontisi, and A. Moustakas, “The biodiversitywind energy-land use nexus in a global biodiversity hotspot,” Science of The Total Environment, vol. 768, p. 144471, 2021.

[8] P. De Marco, S. Villen, P. Mendes, C. N´ obrega, L. Cortes, T. Castro,´ and R. Souza, “Vulnerability of cerrado threatened mammals: an integrative landscape and climate modeling approach,” Biodiversity and Conservation, vol. 29, pp. 1637–1658, 2020.

[9] C. Fletcher, W. J. Ripple, T. Newsome, P. Barnard, K. Beamer, A. Behl, J. Bowen, M. Cooney, E. Crist, C. Field et al., “Earth at risk: An urgent call to end the age of destruction and forge a just and sustainable future,” PNAS nexus, vol. 3, no. 4, p. pgae106, 2024.

[10] A. E. Reside, J. Beher, A. J. Cosgrove, M. C. Evans, L. Seabrook, J. L. Silcock, A. S. Wenger, and M. Maron, “Ecological consequences of land clearing and policy reform in queensland,” Pacific Conservation Biology, vol. 23, no. 3, pp. 219–230, 2017.

[11] S. E. McCord, N. P. Webb, J. W. Van Zee, S. H. Burnett, E. M. Christensen, E. M. Courtright, C. M. Laney, C. Lunch, C. Maxwell, J. W. Karl et al., “Provoking a cultural shift in data quality,” BioScience, vol. 71, no. 6, pp. 647–657, 2021.

[12] S. S. Farley, A. Dawson, S. J. Goring, and J. W. Williams, “Situating ecology as a big-data science: Current advances, challenges, and solutions,” BioScience, vol. 68, no. 8, pp. 563–576, 2018.

[13] D. Brunner, G. Lemoine, and L. Bruzzone, “Earthquake damage assessment of buildings using vhr optical and sar imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 48, no. 5, pp. 2403–2420, 2010.

[14] Z. Zheng, Y. Zhong, J. Wang, A. Ma, and L. Zhang, “Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: From natural disasters to manmade disasters,” Remote Sensing of Environment, vol. 265, p. 112636, 2021.

[15] Y. Qing, D. Ming, Q. Wen, Q. Weng, L. Xu, Y. Chen, Y. Zhang, and B. Zeng, “Operational earthquake-induced building damage assessment using cnn-based direct remote sensing change detection on superpixel level,” International Journal of Applied Earth Observation and Geoinformation, vol. 112, p. 102899, 2022.

[16] M. Papadomanolaki, M. Vakalopoulou, and K. Karantzalos, “A deep multitask learning framework coupling semantic segmentation and fully convolutional lstm networks for urban change detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 9, pp. 7651–7668, 2021.

[17] H. Guo, Q. Shi, A. Marinoni, B. Du, and L. Zhang, “Deep building footprint update network: A semi-supervised method for updating existing building footprint from bi-temporal remote sensing images,” Remote Sensing of Environment, vol. 264, p. 112589, 2021.

[18] X. Huang, Y. Cao, and J. Li, “An automatic change detection method for monitoring newly constructed building areas using time-series multiview high-resolution optical satellite images,” Remote Sensing of Environment, vol. 244, p. 111802, 2020.

[19] H. Chen, C. Lan, J. Song, C. Broni-Bediako, J. Xia, and N. Yokoya, “Objformer: Learning land-cover changes from paired osm data and optical high-resolution imagery via object-guided transformer,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[20] M. M. H. Seyam, M. R. Haque, and M. M. Rahman, “Identifying the land use land cover (lulc) changes using remote sensing and gis approach: A case study at bhaluka in mymensingh, bangladesh,” Case Studies in Chemical and Environmental Engineering, vol. 7, p. 100293, 2023.

[21] Q. Zhu, X. Guo, W. Deng, S. Shi, Q. Guan, Y. Zhong, L. Zhang, and D. Li, “Land-use/land-cover change detection based on a siamese global learning framework for high spatial resolution remote sensing imagery,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 184, pp. 63–78, 2022.

[22] S. Das and D. P. Angadi, “Land use land cover change detection and monitoring of urban growth using remote sensing and gis techniques: A micro-level study,” GeoJournal, vol. 87, no. 3, pp. 2101–2123, 2022.

[23] P. Scarth, S. Gillingham, and J. Muir, “Assimilation of spectral information and temporal history into a statewide woody cover change classification,” in Proceedings of 14th Australasian Remote Sensing and Photogrammetry Conference, Darwin, NT, Australia, vol. 29, 2008.

[24] P. R. Coppin and M. E. Bauer, “Digital change detection in forest ecosystems with remote sensing imagery,” Remote sensing reviews, vol. 13, no. 3-4, pp. 207–234, 1996.

[25] M. Gianinetto and P. Villa, “Mapping hurricane katrina’s widespread destruction in new orleans using multisensor data and the normalized difference change detection (ndcd) technique,” International Journal of Remote Sensing, vol. 32, no. 7, pp. 1961–1982, 2011.

[26] G. Xian and C. Homer, “Updating the 2001 national land cover database impervious surface products to 2006 using landsat imagery change detection methods,” Remote sensing of environment, vol. 114, no. 8, pp. 1676–1686, 2010.

[27] R. J. Radke, S. Andra, O. Al-Kofahi, and B. Roysam, “Image change detection algorithms: a systematic survey,” IEEE transactions on image processing, vol. 14, no. 3, pp. 294–307, 2005.

[28] S. Bontemps, P. Bogaert, N. Titeux, and P. Defourny, “An object-based change detection method accounting for temporal dependences in time series with medium to coarse spatial resolution,” Remote sensing of environment, vol. 112, no. 6, pp. 3181–3191, 2008.

[29] K. Isaienkov, M. Yushchuk, V. Khramtsov, and O. Seliverstov, “Deep learning for regular change detection in ukrainian forest ecosystem with sentinel-2,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 364–376, 2020.

[30] M. Lin, G. Yang, and H. Zhang, “Transition is a process: Pair-tovideo change detection networks for very high resolution remote sensing images,” IEEE Transactions on Image Processing, vol. 32, pp. 57–71, 2022.

[31] Z. Lv, H. Huang, L. Gao, J. A. Benediktsson, M. Zhao, and C. Shi, “Simple multiscale unet for change detection with heterogeneous remote sensing images,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2022.

[32] Z. Li, C. Yan, Y. Sun, and Q. Xin, “A densely attentive refinement network for change detection based on very-high-resolution bitemporal remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–18, 2022.

[33] S. Fang, K. Li, and Z. Li, “Changer: Feature interaction is what you need for change detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–11, 2023.

[34] H. Chen, Z. Qi, and Z. Shi, “Remote sensing image change detection with transformers,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2021.

[35] H. Zhang, K. Chen, C. Liu, H. Chen, Z. Zou, and Z. Shi, “Cdmamba: Incorporating local clues into mamba for remote sensing image binary change detection,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[36] J. N. Paranjape, C. De Melo, and V. M. Patel, “A mamba-based siamese network for remote sensing change detection,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 1186–1196.

[37] J. Pan, Y. Bai, Q. Shu, Z. Zhang, J. Hu, and M. Wang, “M-swin: Transformer-based multi-scale feature fusion change detection network within cropland for remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[38] L. Mou, L. Bruzzone, and X. X. Zhu, “Learning spectral-spatialtemporal features via a recurrent convolutional neural network for change detection in multispectral imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 2, pp. 924–935, 2018.

[39] X. Li, M. He, H. Li, and H. Shen, “A combined loss-based multiscale fully convolutional network for high-resolution remote sensing image change detection,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2021.

[40] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-net: Fully convolutional neural networks for volumetric medical image segmentation,” in 2016 fourth international conference on 3D vision (3DV). Ieee, 2016, pp. 565–571.

[41] R. E. Kennedy, P. A. Townsend, J. E. Gross, W. B. Cohen, P. Bolstad, Y. Wang, and P. Adams, “Remote sensing change detection tools for natural resource managers: Understanding concepts and tradeoffs in the design of landscape monitoring projects,” Remote sensing of environment, vol. 113, no. 7, pp. 1382–1396, 2009.

[42] R. L. Czaplewski and P. L. Patterson, “Classification accuracy for stratification with remotely sensed data,” Forest Science, vol. 49, no. 3, pp. 402–408, 2003.

[43] Queensland Department of Environment and Science, “Statewide landcover and trees study (slats): Overview of methods,” 2018.

[44] Y. Hwang, W. Jo, J. Hong, and Y. Choi, “Overcoming overconfidence for active learning,” IEEE Access, 2024.

[45] C. Aliferis and G. Simon, “Overfitting, underfitting and general model overconfidence and under-performance pitfalls and best practices in machine learning and ai,” Artificial intelligence and machine learning in health care and medical sciences: Best practices and pitfalls, pp. 477–524, 2024.

[46] H. Wei, R. Xie, H. Cheng, L. Feng, B. An, and Y. Li, “Mitigating neural network overconfidence with logit normalization,” in International conference on machine learning. PMLR, 2022, pp. 23 631–23 644.

[47] Y. Ma, S. Chen, S. Ermon, and D. B. Lobell, “Transfer learning in environmental remote sensing,” Remote Sensing of Environment, vol. 301, p. 113924, 2024.

[48] K. Weiss, T. M. Khoshgoftaar, and D. Wang, “A survey of transfer learning,” Journal of Big data, vol. 3, pp. 1–40, 2016.

[49] C. M. Souza Jr, J. Z. Shimbo, M. R. Rosa, L. L. Parente, A. A. Alencar, B. F. Rudorff, H. Hasenack, M. Matsumoto, L. G. Ferreira, P. W. Souza-Filho et al., “Reconstructing three decades of land use and land cover changes in brazilian biomes with landsat archive and earth engine,” Remote Sensing, vol. 12, no. 17, p. 2735, 2020.

[50] Queensland Department of Environment and Science, “2021–22 slats report,” 2024.

[51] X. Yuan, J. Shi, and L. Gu, “A review of deep learning methods for semantic segmentation of remote sensing imagery,” Expert Systems with Applications, vol. 169, p. 114417, 2021.

[52] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[53] N. Flood, T. Danaher, T. Gill, and S. Gillingham, “An operational scheme for deriving standardised surface reflectance from landsat tm/etm+ and spot hrg imagery for eastern australia,” Remote Sensing, vol. 5, no. 1, pp. 83–109, 2013.

[54] N. Flood, “Comparing sentinel-2 surface reflectance from the european space agency level 2a processing and the joint remote sensing research program (australia),” 2020. [Online]. Available: https://doi.org/10.6084/m9.figshare.12838085.v1

[55] K. Backman, B. Beck, and D. Kulic, “Classifying bicycle infrastructure´ using on-bike street-level images,” in 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC), 2024, pp. 3166–3173.

[56] N. Wright, J. M. Duncan, J. N. Callow, S. E. Thompson, and R. J. George, “Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with omnicloudmask,” Remote Sensing of Environment, vol. 322, p. 114694, 2025.

[57] M. C. Hansen, P. V. Potapov, R. Moore, M. Hancher, S. A. Turubanova, A. Tyukavina, D. Thau, S. V. Stehman, S. J. Goetz, T. R. Loveland et al., “High-resolution global maps of 21st-century forest cover change,” science, vol. 342, no. 6160, pp. 850–853, 2013.

[58] L. P. Osco, Q. Wu, E. L. De Lemos, W. N. Gonc¸alves, A. P. M. Ramos, J. Li, and J. M. Junior, “The segment anything model (sam) for remote sensing applications: From zero to one shot,” International Journal of Applied Earth Observation and Geoinformation, vol. 124, p. 103540, 2023.

[59] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European conference on computer vision. Springer, 2024, pp. 38–55.

[60] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.

[61] A. Fisher, M. Day, T. Gill, A. Roff, T. Danaher, and N. Flood, “Largearea, high-resolution tree cover mapping with multi-temporal spot5 imagery, new south wales, australia,” Remote Sensing, vol. 8, no. 6, p. 515, 2016.

[62] Z. Zheng, Y. Zhong, L. Zhang, and S. Ermon, “Segment any change,” Advances in Neural Information Processing Systems, vol. 37, pp. 81 204–81 224, 2024.

[63] S. C. Cook-Patton, S. M. Leavitt, D. Gibbs, N. L. Harris, K. Lister, K. J. Anderson-Teixeira, R. D. Briggs, R. L. Chazdon, T. W. Crowther, P. W. Ellis et al., “Mapping carbon accumulation potential from global natural forest regrowth,” Nature, vol. 585, no. 7826, pp. 545–550, 2020.