# DATA LEAKAGE IN PATCH-BASED HYPERSPECTRAL IMAGE CLASSIFICATION: QUANTIFYING THE IMPACT OF SPATIAL OVERLAP

Mohammed Q. Alkhatib D

College of Engineering and IT, University of Dubai, Dubai, 14143, UAE mqalkhatib@ieee.org

## ABSTRACT

Patch-based learning improves hyperspectral image (HSI) classification by exploiting local spectral-spatial information, but random train-test sampling from the same image can cause spatial patch overlap, leading to data leakage and optimistic performance estimates. This paper investigates same-class train-test spatial overlap in patch-based HSI classification using two measures: overlap percentage (OP), which quantifies the global amount of overlapped testing patch pixels, and average overlap ratio (AOR), which measures the local severity among affected testing patches. Experiments on the Pavia University dataset compare random and non-random spatial sampling using SVM, MLP, 2D-CNN, 3D-CNN, ViT, and MorpMamba. The results show that deep patch-based models achieve high accuracy under random sampling, with 3D-CNN reaching 96.17% Overall Accuracy (OA), but drop substantially under non-random spatial sampling, where 3D-CNN decreases to 55.20% and ViT and 2D-CNN drop by 40.71 and 38.81 percentage points (PP), respectively. Patch-size analysis further shows that increasing the patch size from 5 × 5 to 19 × 19 raises the random-sampling overlap percentage from 23.28% to 77.02%. These findings demonstrate that random patchbased evaluation can substantially inflate classification performance, especially for models that strongly exploit spatial context. The code associated with this paper is available at: https://github.com/mqalkhatib/Data\_ Leakage\_in\_HSI\_Classification.

Index Terms— HSI classification, Data Leakage, Spatial Overlap, Deep Learning, Patch-Based Learning

## 1. INTRODUCTION

Hyperspectral image (HSI) classification assigns semantic labels to pixels using their spectral responses across many contiguous bands. Compared with Optical imagery, HSI captures richer material-specific information, which improves the discrimination of visually similar land-cover classes. This makes HSI classification important for fine-grained scene understanding, particularly when different materials exhibit similar spatial appearance but distinct spectral signatures. However, the task remains challenging due to high dimensionality, spectral redundancy, spatial variability within the same class, and the limited availability of labeled samples.

Traditional methods, such as Support Vector Machines (SVMs) [1] and Multilayer Perceptrons (MLPs) [2], have been widely used for HSI classification, but they often depend mainly on spectral features and have limited ability to exploit spatial context. Deep learning methods address this limitation by learning spectral-spatial representations directly from HSI data [3]. Convolutional neural network (CNN)- based models have progressed from spectral 1D-CNNs [4] to spatial 2D-CNNs [5], joint spectral-spatial 3D-CNNs [6], and hybrid 2D-3D CNNs [7]. More recently, transformer- and Mamba-based models have been introduced to capture longrange dependencies and improve spectral-spatial modeling [8, 9].

Despite these advances, many HSI classifiers still rely on patch-based learning, where a spatial neighborhood around each labeled pixel is used as input. Although this improves spectral-spatial representation, it can introduce data leakage when training and testing samples are randomly selected from the same image. In such cases, their patches may spatially overlap, causing shared pixels to appear in both sets, weakening train-test independence and producing overly optimistic performance estimates [10, 11]. This leakage becomes more severe with larger patch sizes and higher training ratios, and is further amplified by spatial autocorrelation, where neighboring pixels often have similar spectra [11]. Although spatially disjoint sampling can reduce this issue [12], it may further reduce the already limited number of labeled training samples.

To better understand and quantify data leakage, this paper investigates the relationship between same-class train-test spatial overlap and classification performance across six classifiers and two sampling protocols. Unlike prior studies that mainly identify or mitigate leakage, this study jointly quantifies its global extent and local severity using overlap percentage and average overlap ratio. SVM, MLP, 2D-CNN, 3D-CNN, Vision Transformer (ViT), and MorpMamba are evaluated under random and non-random spatial sampling to assess model sensitivity and quantify the resulting accuracy inflation.

## 2. RELATED WORK

## 2.1. Patch-Based Learning in HSI

Patch-based learning is widely used to incorporate spatial context by extracting a local neighborhood around each labeled pixel and using it as the model input. This strategy allows classifiers to exploit local spectral-spatial correlations and has become common in CNN, Transformer, and Mambabased HSI classification frameworks [5, 6, 7, 8, 9].

However, patch extraction introduces dependencies between samples from the same scene. Since neighboring patches may share many pixels, training and testing samples can become highly correlated even when their center pixels are different [10, 11]. This dependency is mainly controlled by the patch size and sampling strategy. Larger patches increase spatial coverage and therefore raise the probability of train-test overlap, while random sampling from a single image can place training and testing pixels close to each other due to spatial continuity [10, 11].

## 2.2. Data Leakage in Remote Sensing and Computer Vision

Data leakage occurs when testing information becomes indirectly available during training, leading to overly optimistic performance estimates. In patch-based HSI classification, random sampling from the same image may place training and testing samples in spatially correlated regions, causing their patches to overlap and share pixels [10, 11]. Spatially disjoint, iterative, and augmentation-based sampling strategies have been proposed to reduce this dependence [12, 13, 10], but they may reduce the number of usable labeled samples and primarily focus on mitigation.

Comparatively little attention has been given to jointly quantifying the global extent and local severity of sameclass train-test patch overlap and relating these characteristics to performance degradation across different classifier families under matched experimental conditions. This work addresses this gap using two complementary overlap measures evaluated across traditional machine-learning, CNN-based, Transformer-based, and Mamba-based classifiers under random and leakage-reduced spatial sampling protocols.

## 3. METHODOLOGY

## 3.1. Patch-Based HSI Classification

The general architecture of most deep learning frameworks for hyperspectral image classification is illustrated in Fig. 1. The process begins with a hyperspectral image $\mathbf { I } _ { \mathrm { O r i g i n a l } } \ \in$ $\mathbb { R } ^ { H \times W \times C }$ To reduce spectral redundancy and computational cost, Principal Component Analysis (PCA) is applied, resulting in a compressed image $\mathbf { I } _ { \mathrm { R e d u c e d } } \in \mathbb { R } ^ { H \times W \times P }$ , where $P \ \ll \ C$ Spatial patches of size $M \times M \times P$ are then extracted and used as inputs to a deep learning model, such as CNN, ViT, or Mamba-based architectures. The extracted features are subsequently passed through a classification head with a softmax activation function to produce the final class predictions.

![](images/5da3cb6247b96177a1dcc6dc208fd852129c9060e143575d5678ed2f6ca8ff4d.jpg)  
Fig. 1: General Patch-based hyperspectral image classification framework using PCA and deep learning models.

## 3.2. Quantifying Train-Test Spatial Overlap

To measure the amount of same-class spatial overlap between training and testing patches, two simple measures are used. Let $T _ { \mathrm { t r } }$ and $T _ { \mathrm { t e } }$ denote the training and testing center pixels, respectively. For any pixel $p ,$ the patch extracted around it is denoted by $\Omega _ { M } ( \boldsymbol { p } )$ , where M is the patch size.

For a class $c ,$ the spatial coverage of its training patches is defined as

$$
\Omega _ { \mathrm { t r } } ^ { ( c ) } = \bigcup _ { \stackrel { p \in T _ { \mathrm { t r } } } { c _ { p } = c } } \Omega _ { M } ( p ) ,\tag{1}
$$

where $c _ { p }$ denotes the class label of the training center pixel p.

A testing patch is considered overlapped when it shares pixels with the training patch coverage of the same class. This same-class condition is used because overlap between patches from different classes does not represent direct classconsistent leakage.

The first measure is the overlap percentage (OP). It measures how many pixels inside the testing patches are also covered by training patches from the same class. It is computed as:

$$
O P = \frac { { \displaystyle \sum _ { q \in T _ { \mathrm { t e } } } { \left| \Omega _ { M } ( q ) \cap \Omega _ { \mathrm { t r } } ^ { ( c _ { q } ) } \right| } } } { { \displaystyle \sum _ { q \in T _ { \mathrm { t e } } } { \left| \Omega _ { M } ( q ) \right| } } } \times 1 0 0 ,\tag{2}
$$

where $c _ { q }$ is the class label of the testing center pixel $q .$ A higher value of $O P$ means that a larger portion of all testingpatch pixels is also contained in same-class training patches.

To compute the second measure, the overlap ratio of each testing patch q is first calculated as:

$$
O R ( \boldsymbol { q } ) = \frac { \left| \Omega _ { M } ( \boldsymbol { q } ) \cap \Omega _ { \mathrm { t r } } ^ { ( c _ { q } ) } \right| } { \left| \Omega _ { M } ( \boldsymbol { q } ) \right| } .\tag{3}
$$

The second measure, termed the average overlap ratio $( A O R )$ , is then obtained by averaging $O R ( q )$ over only the testing patches affected by same-class overlap:

$$
A O R = \frac { 1 } { | Q | } \sum _ { q \in Q } O R ( q ) ,\tag{4}
$$

![](images/9f28990964a5a55da529ad99005f3fd97aade363d755e9733d7fd49d197183d8.jpg)  
Fig. 2: Examples of the sampling settings used in this study: (a) random sampling, where training and testing patches may overlap; (b) non-random spatial sampling, where training and testing regions are selected with greater spatial separation; (c) Class Labels.

where $Q ~ = ~ \{ q ~ \in ~ T _ { \mathrm { t e } } ~ : ~ O R ( q ) ~ > ~ 0 \}$ is the set of testing patches with non-zero same-class overlap.

Therefore, $O P$ measures the global prevalence of leakage across the complete testing set, whereas AOR measures the conditional severity of leakage among only the affected testing patches. Consequently, a sampling protocol may produce a low OP but a relatively high AOR when only a small number of testing patches overlap substantially with same-class training patches.

## 3.3. Non-Random Spatial Sampling Protocol

To examine the effect of spatial overlap in patch-based HSI classification, two sampling protocols are considered, as illustrated in Fig. 2. The first follows the conventional random strategy, where labeled pixels are divided into training, validation, and testing subsets. Since each pixel is represented by a spatial patch, nearby training and testing samples may share pixels, introducing spatial dependency and potentially inflating classification performance.

The second protocol uses non-random spatial sampling, where samples are selected with greater spatial separation. This reduces direct mixing between training and testing regions and provides a stricter evaluation setting. Since testing pixels are not explicitly excluded after patch extraction, this setting is considered leakage-reduced rather than fully leakage-free. Training samples are selected first. Validation samples are then drawn from the remaining labeled pixels and used only for model monitoring and selection, while all other labeled pixels form the test set used for the final performance evaluation.

Comparing the two protocols allows the sensitivity of each classifier to train-test spatial overlap to be evaluated. A large performance drop under non-random spatial sampling suggests that the model may have benefited from spatial dependency under random patch-based evaluation.

## 3.4. Experimental Setup

The experiments are conducted on the Pavia University hyperspectral dataset, which contains nine labeled land-cover classes [14]. Before patch extraction, PCA is applied to reduce the spectral dimensionality to $P = 1 5$ components, and each labeled pixel is represented by ${ \textrm { a 9 } } \times { \textrm { 9 } } \times { \textrm { 1 5 } }$ spatialspectral patch.

For both sampling protocols, a class-balanced training set is constructed by selecting 94 samples per class, corresponding to approximately one-tenth of the smallest class, Shadow, which contains 947 labeled samples. In addition, 18 samples per class are used for validation and reserved only for model monitoring and selection, while all remaining labeled pixels are used for testing.

The evaluated classifiers include SVM [1], MLP [2], 2D-CNN [5], 3D-CNN [6], ViT [15], and MorpMamba [9]. The same training configuration is used under both sampling protocols so that performance differences are mainly attributed to the sampling strategy. All experiments are repeated over 10 independent runs, and the results are reported as mean±standard deviation using Overall Accuracy (OA), Average Accuracy (AA), Kappa coefficient, and class-wise accuracy.

## 4. RESULTS AND DISCUSSION

Tables 1 and 2 present the classification results under random and non-random patch-based sampling, respectively. Under random sampling, deep spectral-spatial models clearly outperform traditional classifiers. The 3D-CNN achieves the highest OA of 96.17%, followed by ViT, 2D-CNN, and MorpMamba with 91.78%, 86.97%, and 81.07%, respectively, whereas SVM and MLP obtain 62.15% and 69.50%. A similar trend is observed for AA and Kappa, indicating that deep patch-based models benefit strongly when training and testing samples are randomly distributed within the same image.

This advantage decreases substantially under the nonrandom spatial setting. As summarized in Table 3, all models experience an OA reduction, with the largest drops observed for 3D-CNN, ViT, and 2D-CNN at 40.97, 40.71, and 38.81 percentage points, respectively. MorpMamba decreases by 32.62 percentage points, while SVM and MLP show smaller drops of 10.90 and 19.35 percentage points. These results indicate that models explicitly exploiting spatial neighborhoods are more sensitive to the sampling protocol.

The class-wise results show a similar pattern. For 3D-CNN, accuracy decreases from 94.28% to 46.27% for Asphalt, from 97.63% to 49.78% for Meadows, from 91.59% to 43.99% for Gravel, and from 96.38% to 47.89% for Bare Soil. These large reductions mainly occur in spatially variable classes. In contrast, compact and spectrally distinctive classes, such as Metal Sheet and Shadow, remain comparatively easier to classify under non-random sampling.

Table 1: Class-wise and overall classification performance under the random patch-based evaluation setting with potential same-class train-test spatial overlap. The reported values are presented as mean±standard deviation over 10 runs.
<table><tr><td>No</td><td>Name</td><td>Train</td><td>Val</td><td>Test</td><td>Total</td><td>SVM</td><td>MLP</td><td>2D-CNN</td><td> $\overline { { 3 \mathrm { D } \mathrm { - } C \mathrm { N N } } }$ </td><td>ViT</td><td> $\overline { { \mathrm { \mathbf { M o r p M a m b a } } } }$ </td></tr><tr><td>1</td><td>Asphalt</td><td>94</td><td>18</td><td>6,519</td><td>6,631</td><td> $\overline { { 6 2 . 8 3 \pm 1 . 0 1 } }$ </td><td> $\overline { { 6 4 . 5 1 \pm 3 . 1 1 } }$ </td><td> $\overline { { 8 3 . 0 7 \pm 8 . 1 9 } }$ </td><td> $\overline { { 9 4 . 2 8 { \pm } 1 . 9 5 } }$ </td><td> $\overline { { 8 9 . 2 6 { \pm } 3 . 3 2 } }$ </td><td> $7 5 . 6 7 { \scriptstyle \pm 6 . 8 6 }$ </td></tr><tr><td>2</td><td>Meadows</td><td>94</td><td>18</td><td>18,537</td><td>18,649</td><td> $4 9 . 4 6 { \pm } 4 . 9 1 $ </td><td> $6 4 . 1 7 { \pm } 8 . 4 5$ </td><td> $8 4 . 9 8 { \pm } 5 . 4 7$ </td><td> $9 7 . 6 3 { \pm } 1 . 1 7 $ </td><td> $9 1 . 3 2 { \pm } 4 . 0 4$ </td><td> $7 8 . 0 5 { \pm } 7 . 5 0 $ </td></tr><tr><td>3</td><td>Gravel</td><td>94</td><td>18</td><td>1,987</td><td>2,099</td><td> $2 1 . 4 0 { \pm } 8 . 3 4$ </td><td> $6 9 . 2 6 { \pm } 1 2 . 6 2$ </td><td> $8 3 . 0 4 \pm 4 . 2 8 $ </td><td> $9 1 . 5 9 { \pm } 3 . 1 0 \ $ </td><td> $8 8 . 9 4 \pm 3 . 8 6 $ </td><td> $7 4 . 4 9 { \pm } 1 6 . 5 7$ </td></tr><tr><td>4</td><td>Trees</td><td>94</td><td>18</td><td>2,952</td><td>3,064</td><td> $9 5 . 3 2 { \pm } 1 . 4 8 $ </td><td> $9 6 . 1 4 { \pm } 1 . 5 0 $ </td><td> $9 6 . 9 8 { \pm } 1 . 0 4 $ </td><td> $9 7 . 7 4 { \pm } 1 . 1 1 $ </td><td> $9 7 . 7 1 { \pm } 1 . 1 1 $ </td><td> $9 5 . 4 5 { \pm } 1 . 6 4 $ </td></tr><tr><td>5</td><td>Metal Sheet</td><td>94</td><td>18</td><td>1,233</td><td>1,345</td><td> $9 9 . 6 3 { \pm } 0 . 1 5 $ </td><td> $9 9 . 6 5 { \scriptstyle \pm 0 . 2 0 } $ </td><td> $9 9 . 5 0 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $9 9 . 9 9 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $9 9 . 9 9 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $9 9 . 9 4 { \pm } 0 . 0 6 $ </td></tr><tr><td>6</td><td>Bare Soil</td><td>94</td><td>18</td><td>4,917</td><td>5,029</td><td> $6 7 . 1 8 { \pm } 5 . 1 6 $ </td><td> $6 4 . 7 1 { \scriptstyle \pm 1 1 . 9 4 }$ </td><td> $8 8 . 5 1 { \pm } 3 . 4 5 $ </td><td> $9 6 . 3 8 { \pm } 1 . 5 6 $ </td><td> $8 9 . 4 4 { \pm } 4 . 8 2 $ </td><td> $8 8 . 4 1 { \scriptstyle \pm 6 . 4 4 }$ </td></tr><tr><td>7</td><td>Bitumen</td><td>94</td><td>18</td><td>1,218</td><td>1,330</td><td> $8 7 . 4 0 { \pm } 1 . 5 5 $ </td><td> $8 7 . 2 3 { \pm } 5 . 4 5$ </td><td> $9 2 . 1 3 { \pm } 3 . 6 5 $ </td><td> $9 7 . 0 4 \pm 1 . 6 0 $ </td><td> $9 6 . 4 9 { \pm } 1 . 6 8 $ </td><td> $8 5 . 3 4 \pm 7 . 4 3$ </td></tr><tr><td>8</td><td>Brick</td><td>94</td><td>18</td><td>3,570</td><td>3,682</td><td> $8 4 . 7 4 \pm 5 . 0 7$ </td><td> $6 7 . 4 3 { \pm } 8 . 0 4 $ </td><td> $8 7 . 2 0 { \pm } 4 . 0 4 $ </td><td> $9 0 . 5 1 { \pm } 4 . 1 6 $ </td><td> $9 2 . 4 4 { \pm } 2 . 0 3 $ </td><td> $7 6 . 1 5 { \pm } 1 1 . 9 4$ </td></tr><tr><td>9</td><td>Shadow</td><td>94</td><td>18</td><td>835</td><td>947</td><td> $9 9 . 7 7 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $9 9 . 7 2 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $9 9 . 6 4 { \pm } 0 . 2 3 $ </td><td> $9 9 . 9 2 { \pm } 0 . 1 1 $ </td><td> $9 9 . 7 1 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $9 9 . 0 8 { \pm } 0 . 7 9 $ </td></tr><tr><td colspan="5"></td><td></td><td> $6 2 . 1 5 { \pm } 1 . 6 8$ </td><td> $\overline { { 6 9 . 5 0 { \pm } 3 . 2 2 } }$ </td><td> $\overline { { 8 6 . 9 7 { \pm } 3 . 2 7 } }$ </td><td> $\overline { { 9 6 . 1 7 { \scriptstyle \pm 0 . 8 0 } } }$ </td><td> $9 1 . 7 8 { \pm } 1 . 3 0 $ </td><td> $\overline { { 8 1 . 0 7 { \pm } 3 . 6 5 } }$ </td></tr><tr><td colspan="5">AA (%)</td><td></td><td> $7 4 . 1 9 { \pm } 0 . 5 4 $ </td><td> $7 9 . 2 0 { \pm } 1 . 9 0 $ </td><td> $9 0 . 5 6 { \pm } 1 . 6 0 $ </td><td> $9 6 . 1 2 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $9 3 . 9 2 { \pm } 0 . 8 3 $ </td><td> $8 5 . 8 4 \pm 2 . 4 3$ </td></tr><tr><td colspan="5">Kappa × 100</td><td> $5 4 . 0 1 { \pm } 1 . 5 6 $ </td><td> $6 1 . 7 7 { \scriptstyle \pm 3 . 3 7 }$ </td><td></td><td> $8 3 . 1 0 { \pm } 4 . 0 2 $ </td><td> $9 4 . 9 0 { \pm } 1 . 0 5 $ </td><td> $8 9 . 1 9 { \pm } 1 . 6 1$ </td><td> $7 5 . 8 3 { \pm } 4 . 3 0 $ </td></tr></table>

Table 2: Class-wise and overall classification performance under the non-random spatial sampling setting designed to reduce same-class train-test spatial overlap. The reported values are presented as mean±standard deviation over repeated runs.
<table><tr><td>No</td><td>Name</td><td>Train</td><td>Val</td><td>Test</td><td>Total</td><td>SVM</td><td>MLP</td><td>2D-CNN</td><td>3D-CNN</td><td>ViT</td><td>MorpMamba</td></tr><tr><td>1</td><td>Asphalt</td><td>94</td><td>18</td><td>6,519</td><td>6,631</td><td> $\overline { { 6 1 . 0 4 \pm 1 2 . 6 6 } }$ </td><td> $\overline { { 6 4 . 4 3 \pm 9 . 4 4 } }$ </td><td> $\overline { { 5 0 . 3 0 { \pm } 1 8 . 1 0 } }$ </td><td> $\overline { { 4 6 . 2 7 \pm 1 6 . 3 2 } }$ </td><td> $\overline { { 4 6 . 6 3 \pm 1 4 . 6 1 } }$ </td><td> $\overline { { 4 9 . 1 1 \pm 1 5 . 4 8 } }$ </td></tr><tr><td>2</td><td>Meadows</td><td>94</td><td>18</td><td>18,537</td><td>18,649</td><td> $3 8 . 7 2 { \scriptstyle \pm 1 2 . 4 1 }$ </td><td> $3 3 . 6 6 { \pm } 1 4 . 4 6$ </td><td> $4 1 . 5 6 { \pm } 1 0 . 4 5$ </td><td> $4 9 . 7 8 { \pm } 1 8 . 8 8$ </td><td> $5 0 . 2 6 { \pm } 1 7 . 1 9$ </td><td> $3 7 . 6 0 { \scriptstyle \pm 1 3 . 6 1 }$ </td></tr><tr><td>3</td><td>Gravel</td><td>94</td><td>18</td><td>1,987</td><td>2,099</td><td> $2 9 . 8 5 { \pm } 1 2 . 7 4 $ </td><td> $3 6 . 5 1 { \pm } 2 2 . 8 1 $ </td><td> $3 4 . 8 3 { \pm } 2 2 . 7 2$ </td><td> $4 3 . 9 9 { \pm } 2 0 . 9 3 $ </td><td> $3 1 . 3 6 \pm 1 7 . 4 6$ </td><td> $3 0 . 9 7 { \scriptstyle \pm 2 0 . 5 4 }$ </td></tr><tr><td>4</td><td>Trees</td><td>94</td><td>18</td><td>2,952</td><td>3,064</td><td> $7 6 . 8 8 { \pm } 1 9 . 9 5$ </td><td> $8 0 . 3 9 { \pm } 1 3 . 0 5$ </td><td> $6 2 . 1 7 { \pm } 1 8 . 3 1$ </td><td> $8 2 . 3 6 { \pm } 1 1 . 1 2$ </td><td> $5 7 . 6 2 { \pm } 2 1 . 1 3 $ </td><td> $7 3 . 1 7 { \pm } 1 9 . 3 5 $ </td></tr><tr><td>5</td><td>Metal Sheet</td><td>94</td><td>18</td><td>1,233</td><td>1,345</td><td> $9 9 . 6 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $9 9 . 4 5 { \pm } 0 . 4 9$ </td><td> $9 2 . 9 9 { \pm } 4 . 9 6 $ </td><td> $9 9 . 7 6 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $9 8 . 9 0 { \pm } 1 . 5 7 \ $ </td><td> $9 7 . 3 1 { \pm } 2 . 9 3 $ </td></tr><tr><td>6</td><td>Bare Soil</td><td>94</td><td>18</td><td>4,917</td><td>5,029</td><td> $4 5 . 8 1 \pm 1 1 . 6 1$ </td><td> $4 8 . 6 5 { \pm } 1 2 . 3 5 $ </td><td> $3 8 . 4 1 \pm 6 . 5 5$ </td><td> $4 7 . 8 9 { \pm } 1 3 . 4 7$ </td><td> $3 9 . 5 1 { \pm } 1 1 . 0 1 $ </td><td> $4 2 . 4 6 { \pm } 1 4 . 8 8$ </td></tr><tr><td>7</td><td>Bitumen</td><td>94</td><td>18</td><td>1,218</td><td>1,330</td><td> $6 6 . 7 9 { \scriptstyle \pm 1 9 . 2 1 }$ </td><td> $7 4 . 5 2 { \pm } 1 7 . 4 5$ </td><td> $7 2 . 0 0 { \scriptstyle \pm 1 6 . 2 0 }$ </td><td> $8 9 . 9 2 { \pm } 5 . 3 8 $ </td><td> $6 4 . 7 8 { \pm } 2 0 . 5 7$ </td><td> $7 7 . 7 8 { \pm } 1 5 . 3 4$ </td></tr><tr><td>8</td><td>Brick</td><td>94</td><td>18</td><td>3,570</td><td>3,682</td><td> $6 3 . 3 6 { \pm } 1 2 . 0 2$ </td><td> $5 7 . 3 8 { \pm } 2 4 . 1 6$ </td><td> $5 3 . 6 3 { \pm } 1 7 . 6 6$ </td><td> $5 6 . 5 5 { \pm } 2 0 . 7 2$ </td><td> $5 6 . 7 5 { \pm } 1 7 . 0 2$ </td><td> $6 4 . 9 4 { \pm } 1 5 . 1 3$ </td></tr><tr><td rowspan="3">9</td><td colspan="5">Shadow 94</td><td> $9 9 . 4 0 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $9 9 . 7 0 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $9 3 . 0 8 { \pm } 3 . 7 4 $ </td><td> $9 6 . 5 1 { \pm } 5 . 3 6$ </td><td> $8 0 . 8 5 { \pm } 1 2 . 7 6$ </td><td> $8 8 . 2 8 { \pm } 7 . 0 1 $ </td></tr><tr><td colspan="4">OA (%)</td><td></td><td> $\overline { { 5 1 . 2 5 { \pm } 5 . 4 3 } }$ </td><td> $\overline { { 5 0 . 1 5 { \pm 6 . 9 1 } } }$ </td><td> $4 8 . 1 6 { \pm } 5 . 9 6 $ </td><td>55.20±9.21</td><td> $\overline { { 5 1 . 0 7 { \pm } 7 . 4 2 } }$ </td><td> $\overline { { 4 8 . 4 5 { \pm 9 . 2 6 } } }$ </td></tr><tr><td colspan="4">AA (%)</td><td></td><td> $6 4 . 6 0 { \scriptstyle \pm 3 . 7 6 }$ </td><td> $6 6 . 0 8 { \scriptstyle \pm 3 . 2 2 }$ </td><td> $5 9 . 8 9 { \pm } 4 . 7 8 $ </td><td> $6 8 . 1 2 { \scriptstyle \pm 3 . 0 2 }$ </td><td> $5 8 . 5 2 { \pm } 3 . 5 7 $ </td><td> $6 2 . 4 0 { \scriptstyle \pm 4 . 9 2 }$ </td></tr><tr><td colspan="5"> $\mathrm { K a p p a } \times 1 0 0$ </td><td> $4 1 . 8 4 \pm 5 . 0 9$ </td><td> $4 1 . 2 7 { \pm } 6 . 4 4$ </td><td> $3 8 . 0 6 { \pm } 6 . 4 5 $ </td><td> $4 5 . 9 6 \pm 8 . 6 4$ </td><td></td><td>40.54±6.81</td><td> $3 9 . 1 5 { \pm } 9 . 0 6 $ </td></tr></table>

Table 3: Overall accuracy degradation, in percentage points, from random sampling to non-random spatial sampling.
<table><tr><td>Model</td><td>Random OA (%)</td><td>Non-Random OA (%)</td><td>OA Drop (pp)</td></tr><tr><td>SVM</td><td>62.15</td><td>51.25</td><td>10.90</td></tr><tr><td>MLP</td><td>69.50</td><td>50.15</td><td>19.35</td></tr><tr><td>2D-CNN</td><td>86.97</td><td>48.16</td><td>38.81</td></tr><tr><td>3D-CNN</td><td>96.17</td><td>55.20</td><td>40.97</td></tr><tr><td>ViT</td><td>91.78</td><td>51.07</td><td>40.71</td></tr><tr><td>MorpMamba</td><td>81.07</td><td>48.45</td><td>32.62</td></tr></table>

Table 4: Effect of patch size on same-class overlap percentage and average overlap ratio under random and non-random spatial sampling settings. The values are computed from the training and testing patch locations, not from model predictions. Only mean values are reported.

Overall, Tables 1 and 2 indicate that the high accuracy obtained under random patch-based evaluation is partly influenced by same-class train-test spatial dependency. Neighboring patches may share pixels or highly similar local structures, allowing information from training regions to be indirectly reflected in the test set. Random sampling may therefore produce optimistic performance estimates, whereas non-random spatial sampling provides a stricter assessment of model generalization.

<table><tr><td rowspan="2">Patch Size</td><td colspan="2">Overlap Percentage (%)</td><td colspan="2">Average Overlap Ratio (%)</td></tr><tr><td>Random</td><td>Non-Random</td><td>Random</td><td>Non-Random</td></tr><tr><td> $\overline { { 5 \times 5 } }$ </td><td>23.28</td><td>0.83</td><td>47.29</td><td>45.26</td></tr><tr><td> $7 \times 7$ </td><td>35.03</td><td>1.28</td><td>52.98</td><td>43.79</td></tr><tr><td> $9 \times 9$ </td><td>45.29</td><td>1.75</td><td>58.45</td><td>42.26</td></tr><tr><td> $1 1 \times 1 1$ </td><td>54.04</td><td>2.26</td><td>63.61</td><td>40.50</td></tr><tr><td> $1 3 \times 1 3$ </td><td>61.48</td><td>2.80</td><td>68.31</td><td>38.98</td></tr><tr><td> $1 5 \times 1 5$ </td><td>67.73</td><td>3.39</td><td>72.70</td><td>38.86</td></tr><tr><td> $1 7 \times 1 7$ </td><td>72.86</td><td>4.01</td><td>76.60</td><td>39.13</td></tr><tr><td> $1 9 \times 1 9$ </td><td>77.02</td><td>4.66</td><td>79.91</td><td>39.42</td></tr></table>

## 4.1. Effect of Patch Size on Same-Class Spatial Leakage

Table 4 shows the effect of patch size on the same-class OP and AOR, computed using Eqs. (2) and (4), respectively. Under random sampling, OP increases from 23.28% with a 5×5 patch to 77.02% with a $1 9 \times 1 9$ patch, showing that larger patches substantially increase the portion of testing regions covered by training patches. AOR follows the same trend, increasing from 47.29% to 79.91%, which indicates that larger patches also increase the severity of overlap among the affected testing patches. In contrast, the non-random spatial setting maintains a much lower overlap percentage, increasing only from 0.83% to 4.66% as the patch size grows from 5 × 5 to 19 × 19. This confirms that the non-random sampling strategy greatly reduces the global amount of same-class train-test overlap. However, AOR remains relatively high, decreasing from 45.26% to around 39% for larger patch sizes. This is because Eq. (4) is computed only over testing patches with non-zero overlap; thus, although few testing patches are affected, the remaining overlap cases may still contain considerable local overlap.

Overall, patch size directly controls both the extent and severity of spatial leakage. OP measures the global amount of leakage, whereas AOR provides complementary information about the severity of the affected cases. This distinction is important in leakage-reduced settings, where the number of affected testing patches may be small even when their local overlap remains high.

## 5. CONCLUSION

This paper investigated same-class train-test spatial overlap in patch-based hyperspectral image classification using two measures: OP and AOR. The results showed that random patch-based sampling can substantially inflate classification performance, especially for deep models that exploit spatial context. Under non-random spatial sampling, 3D-CNN, ViT, and 2D-CNN showed large OA drops of 40.97, 40.71, and 38.81 percentage points, respectively, while SVM and MLP showed smaller reductions. The patch-size analysis further confirmed that larger patches increase spatial leakage, with the random-sampling overlap percentage rising from 23.28% for 5 × 5 patches to 77.02% for $1 9 \times 1 9$ patches.

Future work will extend this analysis to additional hyperspectral datasets and different training ratios to further assess the generality of the observed leakage effects. More rigorous spatial separation strategies will also be investigated to reduce train-test dependency while preserving sufficient labeled samples for training. In addition, future studies may examine how leakage affects newer deep architectures and whether model design choices can improve robustness under leakage-reduced evaluation settings.

## 6. REFERENCES

[1] Farid Melgani and Lorenzo Bruzzone, “Classification of hyperspectral remote sensing images with support vector machines,” IEEE Transactions on geoscience and remote sensing, vol. 42, no. 8, pp. 1778–1790, 2004.

[2] Javier Plaza, Antonio Plaza, Rosa Perez, and Pablo´ Mart´ınez, “Parallel classification of hyperspectral images using neural networks,” in Computational intelligencefor remote sensing, pp. 193–216. Springer, 2008.

[3] Mohammed Q. Alkhatib, “ConvViTMamba: efficient multiscale convolution, Transformer, and Mamba-based sequence modelling for hyperspectral image classification,” International Journal of Remote Sensing, vol. 0, no. 0, pp. 1–40, 2026.

[4] Wei Hu, Yangyu Huang, Li Wei, Fan Zhang, and Hengchao Li, “Deep convolutional neural networks for hyperspectral image classification,” Journal of Sensors, vol. 2015, pp. 1–12, 2015.

[5] Konstantinos Makantasis, Konstantinos Karantzalos, Anastasios Doulamis, and Nikolaos Doulamis, “Deep supervised learning for hyperspectral data classification through convolutional neural networks,” in 2015 IEEE international geoscience and remote sensing symposium (IGARSS). IEEE, 2015, pp. 4959–4962.

[6] Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, and Chokri Ben Amar, “3-D deep learning approach for remote sensing image classification,” IEEE Transactions on geoscience and remote sensing, vol. 56, no. 8, pp. 4420–4434, 2018.

[7] Swalpa Kumar Roy, Gopal Krishna, Shiv Ram Dubey, and Bidyut B Chaudhuri, “HybridSN: Exploring 3-D–2- D CNN feature hierarchy for hyperspectral image classification,” IEEE Geoscience and Remote Sensing Letters, vol. 17, no. 2, pp. 277–281, 2019.

[8] Mohammed Q Alkhatib and Ali Jamali, “HSIFormer: An Efficient Vision Transformer Framework for Enhanced Hyperspectral Image Classification Using Local Window Attention,” in 2024 14th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS). IEEE, 2024, pp. 1–5.

[9] Muhammad Ahmad, Muhammad Hassaan Farooq Butt, Adil Mehmood Khan, Manuel Mazzara, Salvatore Distefano, Muhammad Usama, Swalpa Kumar Roy, Jocelyn Chanussot, and Danfeng Hong, “Spatial–spectral morphological mamba for hyperspectral image classification,” Neurocomputing, vol. 636, pp. 129995, 2025.

[10] Hao Feng, Yongcheng Wang, Zheng Li, Ning Zhang, Yuxi Zhang, and Yunxiao Gao, “Information leakage in deep learning-based hyperspectral image classification: A survey,” Remote Sensing, vol. 15, no. 15, pp. 3793, 2023.

[11] Kaizhe Zhan, Zijie Huo, Zhijian Yin, and Zhen Yang, “Rethinking data leakage in patch-based hyperspectral image classification with traditional deep networks,” Remote Sensing, vol. 18, no. 7, pp. 979, 2026.

[12] Muhammad Ahmad, Usman Ghous, Danfeng Hong, Adil Mehmood Khan, Jing Yao, Shaohua Wang, and Jocelyn Chanussot, “A disjoint samples-based 3d-cnn with active transfer learning for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–16, 2022.

[13] Chein-I Chang, Chia-Chen Liang, and Peter Fuming Hu, “Iterative random training sampling convolutional neural network for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–26, 2023.

[14] “Hyperspectral remote sensing scenes,” https: //www.ehu.eus/ccwintco/index.php/ Hyperspectral\_Remote\_Sensing\_Scenes, Accessed: 31-May-2026.

[15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.