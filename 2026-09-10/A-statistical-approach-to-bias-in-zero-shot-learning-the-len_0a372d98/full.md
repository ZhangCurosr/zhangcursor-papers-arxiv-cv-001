# A statistical approach to bias in zero-shot learning: the lens of handwriting recognition

Clarence Chew<sup>1</sup>

Department of Mathematics

National University of Singapore

clarence\_chew\_@nus.edu.sg

Sukalpa Chanda

Dept of Comp Sc & Comm

Østfold University College

sukalpa@ieee.org

Soumendu Sundar Mukherjee

Statistics and Mathematics Unit

Indian Statistical Institute

ssmukherjee@isical.ac.in

Gim Siang Chia<sup>1</sup>

Department of Mathematics

National University of Singapore

e0310077@u.nus.edu

Subhroshekhar Ghosh

Department of Mathematics

National University of Singapore

subhrowork@gmail.com

## Abstract

Generalized zero-shot learning (GZSL) has emerged as an important paradigm for visual recognition systems that must generalize to classes that were not observed during training. Traditional GZSL techniques are limited by their applicability to a relatively small number of such unseen classes, scalability beyond which is challenging due to its wellknown misclassification bias towards classes observed during training. In this work, we investigate the GZSL paradigm through the lens of zero-shot handwritten word recognition over extremely large vocabularies, which brings into sharp focus the central problem of inherent bias of GZSL methods towards the training data. We propose a statistical approach to rectifying this bias, which views any classical GZSL feature learner as a black box mechanism whose intrinsic bias is reflected in its ability to identify the training status (seen vs. unseen) of a typical data point, which may be viewed as an out of distribution inferential problem. Our method leverages a simple two-stage hierarchical architecture, combining a classical GZSL blackbox in the first stage and an ensemble of light-weight Monte Carlo based bias-correctors in the second, wherein the dual tasks of bias estimation and threshold calibration are performed. Once debiased, the classification of test data is undertaken only restricted to its predicted training status, using well-founded approaches with solid statistical underpinnings (such as nearest neighbour, logistic regression and random forests). In doing so, we can achieve relative accuracy improvements of over 20% in the classification of unseen words compared to established techniques. A key outcome of our investigations is that word recognition over large scale vocabularies can be well-represented by a much lower dimensional approximation (with as low as 15 dimensions), which renders practicable classical statistical methodologies as well as very large scale Monte Carlo techniques. Our approach is underpinned by mathematical analysis that captures the essence of the statistical approach to bias correction. It may be recognized that our statistical approach to bias rectification can be combined in a turn-key fashion with potentially any classical GZSL learner as a blackbox, thereby suggesting a wide scope of applicability of this method for a wide variety of GZSL implementations in diferent domains.

Contents   
1 Introduction 4   
2 Problem Formulation and Methodology 7   
3 The Algorithmic Pipeline 10   
4 Results & Discussion 13   
5 Theoretical Underpinnings of Ensembling for Bias Reduction 16   
6 Concluding Remarks 18   
7 Acknowledgements 18   
A Supporting Algorithms 21   
B Further Theoretical Details 24   
B.1 Why Use $E _ { m }$ over a Consensus Voting? . 24   
B.2 Proofs of Various Results . . 24   
B.3 A Result under Exchangeability of the Predictors $\psi _ { k }$ 27

## 1 Introduction

Generalized Zero-shot Learning (ZSL) has emerged as an important paradigm for visual recognition systems that must generalize to classes that were not observed during training [18]. In existing approaches, semantic relationships between classes are often captured through attribute annotations or language embeddings, with the ultimate objective of transferring knowledge from seen classes to unseen ones. Most existing work on zero-shot recognition has focused on object classification benchmarks such as Animals with Attributes (AwA), CUB-200, and SUN, where the number of unseen classes typically ranges from tens to a few hundred [5, 17, 15, 16].

A well-known issue with existing GZSL frameworks is a persistent bias towards seen classes, i.e. those classes whose data have been used for training the learning model [5]. This problem of inherent bias towards seen classes is brought into sharp focus in the substantially more challenging setting of zeroshot handwritten word recognition over extremely large vocabularies. Unlike object recognition datasets, where visual classes correspond to semantically meaningful object categories, handwritten word recognition operates over a combinatorial space of word forms. Even moderately sized vocabularies can contain thousands of distinct classes, and new word classes can be constructed through arbitrary combinations of characters.

This fundamental diference introduces several challenges that are not present in conventional zero-shot object recognition. First, the number of potential classes is significantly larger, leading to a much denser and more complex class manifold. Second, visual diferences between classes can be extremely subtle; words may difer by only a single character or by a small modification in stroke structure. Third, handwritten text exhibits substantial intra-class variation due to diferences in writing style, stroke connectivity, and character spacing. Together, these factors make large-vocabulary zeroshot word recognition a significantly more dificult problem than standard object-level zero-shot classification [12, 4].

Most modern handwriting recognition systems rely on Connectionist Temporal Classification (CTC) based sequence models that predict character sequences [7]. While efective for transcription tasks, these models are known to exhibit several limitations which render them a structurally misaligned for zero-shot word recognition problems; these include segmentation dificulties, accumulation of errors across characters, and focus on local (as opposed to global) structures in complex words.

To address these dificulties, we leverage a structured word embedding paradigm, namely the so-called Pho(SC)Net [12], the proposed new representation called Pyramidal Histogram of Shapes, captures both the positional information of characters and their visual shape appearance, making it more discriminative than traditional PHOC embeddings [12]. Handwritten word images are mapped directly into this embedding space, and recognition is performed by matching the predicted embedding against embeddings of candidate. Combined with a classical nearest neighbor classifier, this approach is known to prominently capture the well-known phenomenon of bias towards seen classes in GZSL, with accuracy rates for correct identification of unseen classes at a lowly 77% (compared to a seen class accuracy of around 93%) [12].

Our Contributions. In this paper, we investigate the problem of mitigating the inherent bias towards seen classes in GZSL through the prism of handwritten word recognition over large vocabularies.

We propose a statistical approach to rectifying bias towards seen classes, which views any classical GZSL feature learner as a black box mechanism whose intrinsic bias is reflected in its ability (or lack thereof) to identify the training status (seen vs. unseen) of the class of a typical test data point. This latter issue may in fact be viewed as an out of distribution inferential problem; although in view of complete class identification (as opposed to mere training status), our problem is much more challenging and goes far beyond classical out of distribution learning.

Our method leverages a simple two-stage hierarchical architecture, combining a classical GZSL feature learning blackbox in the first stage and an ensemble of light weight Monte Carlo based bias-correctors in the second. While the first stage may be envisaged to generate a structured embedding of a give data point, it is the second stage where the dual tasks of bias estimation and threshold calibration are performed. To wit, using a Monte Carlo aggregation approach, the classifiers in the second phase certify an input data point to from a seen class only if the proportion of seen certificates from the individual second stage classifiers crosses a suitable threshold that is calibrated to capture the inherent bias of the GZSL feature generator in the first stage. The overall output of this two stage architecture is thus two-fold: (i) the structured word embedding (from the first stage) and (ii) a debiased training status predictor (seen/unseen) of the class of the input data point (from the second stage).

Subsequent to processing via this two-stage architecture, the classification of test data is undertaken only restricted to its predicted training status, using well-founded approaches with solid statistical underpinnings (such as nearest neighbour classifiers, logistic regression and random forests). This design allows the system to dynamically restrict recognition to either seen or unseen candidate vocabularies, mitigating the well-known bias toward seen classes in generalized zero-shot learning. Our approach is well-motivated by an underpinning mathematical analysis that unveils the potential of such a method under a simplified analytical setup.

A key observation augmenting our approach is a highly revealing geometric property of this embedding space. Although the structured word representation has dimensionality d = 769, we find that the embeddings produced by trained models lie on a remarkably low-dimensional manifold. Empirically, applying principal component analysis (PCA) reveals that the essential structure of the embedding space can be captured using only 15 dimensions. Despite this dramatic reduction in dimensionality, the resulting representations remain highly informative for distinguishing between thousands of word classes. This phenomenon suggests that the embedding space possesses a strong underlying geometric structure that can be exploited for recognition. In particular, we can leverage this property to construct highly lightweight classifiers that operate in the reduced-dimensional space to determine whether a predicted embedding corresponds to a word class observed during training.

By combining structured word embeddings with a principled statistical approach to bias correction, the proposed framework enables recognition across vocabularies containing thousands of word classes, including a large number of unseen words. Augmented with dimensionality reduction leading to a parsimonious representation of the embedding manifold geometry, we are able to provide significant advances (relative improvements of over 20%) in accurately identifying unseen classes in the challenging problem of zero short handwritten word recognition, at a very manageable computational cost. The proposed method leverages the structured word embedding Pho(SC)Net as a black box, and can thus be potentially augmented in a turn-key fashion to any classical GZSL feature learning mechanism to achieve superior outcomes, especially for correct identification of unseen data categories.

Related Work. Zero-shot learning (ZSL) has been studied extensively in object recognition, where models learns a mapping from seen to unseen categories through attributes or semantic embeddings [9, 11, 6, 13, 1]. In generalized zero-shot learning, inference during test phase involves recognition over both seen and unseen classes [5, 17, 15, 16]. While these methods have led to strong progress on benchmarks such as AwA, CUB, and SUN, the underlying setting typically involves tens or hundreds of semantically separated object classes. But Handwritten word recognition is fundamentally diferent: the label space is combinatorial, vocabularies can contain thousands of classes, and neighboring classes may difer by only a single character or minor stroke variation. This makes direct transfer of standard ZSL assumptions and methods non-trivial.

Embedding-based word spotting and recognition. In many real-world applications such as searching large digital archives, it is customary to represent both word images and text strings in a shared embedding space, where retrieval can be performed via similarity-based matching. The PHOC representation introduced by Almazán et al. [2] is a foundational approach in this direction, enabling segmentation-free word spotting and lexicon-based recognition. Subsequent work showed that deep architectures can predict such embeddings efectively from word images [14, 8], and PHOC-style methods have remained attractive because they support lexicon search through nearest-neighbor matching rather than explicit sequence decoding.

Zero-shot handwritten word recognition. Closer to our setting, recent work has explored zero-shot handwritten word recognition using structured embeddings that encode both character occurrence and shape information. Pho(SC)Net introduced a hybrid PHO(SC) representation for zero-shot word image recognition in historical documents, showing that shape-aware structured embeddings can improve generalization to unseen words [12]. Pho(SC)- CTC later combined these representations with a CTC-based decoding framework to improve recognition further [4]. These studies established structured word embeddings as a viable basis for zero-shot handwritten recognition, but they do not explicitly address the large-scale generalized setting in which test data contain both seen and unseen words and the model must manage the strong bias toward seen classes.

## 2 Problem Formulation and Methodology

Problem Formulation. Let X denote a space of handwritten word images and W denote the corresponding vocabulary of word classes. Let $\chi \stackrel { w } {  } \mathcal { W }$ $x \mapsto w ( x )$ denote a map taking the word image to the word class. Each word class $w \in \mathcal W$ is associated with an embedding vector $e ( w ) \in \mathbb { R } ^ { d }$

We partition the vocabulary into two disjoint sets $\mathcal { W } = \mathcal { W } _ { s } \cup \mathcal { W } _ { u }$ , where ${ \mathcal { W } } _ { s }$ represents the set of word classes available during training and $\mathcal { W } _ { u }$ represents unseen word classes that appear only during testing. The data points (in the form of handwritten word images) are envisaged to be generated from a distribution $P$ on the space $\mathcal { X }$ . The training and test datasets are defined respectively as

$$
\begin{array} { r l } & { \mathcal { D } _ { \mathrm { t r a i n } } = \{ ( x _ { i } , w _ { i } \mid x _ { i } \in \mathcal { X } , \ w _ { i } = w ( x _ { i } ) \in \mathcal { W } _ { s } \} ; } \\ & { \mathcal { D } _ { \mathrm { t e s t } } = \{ ( x _ { j } , w _ { j } ) \mid x _ { j } \in \mathcal { X } , \ w _ { j } \in \mathcal { W } _ { s } \cup \mathcal { W } _ { u } \} . } \end{array}
$$

We learn a function $f _ { \theta } : \mathcal { X }  \mathbb { R } ^ { d }$ , that maps a word image to a structured word embedding; suitably parametrized by θ which may symbolize, for instance, the weights of a neural network which may drive the predictor $f _ { \theta }$ Classically, recognition of a word is performed by nearest neighbor matching; in other words, $\begin{array} { r } { \hat { w } ( x ) = \arg \operatorname* { m i n } _ { w \in { \mathcal W } } \| f _ { \theta } ( x ) - e ( w ) \| } \end{array}$

Methodology. Our framework consists of two phases: a first phase consisting of $L$ copies of embedding models $( M _ { [ 1 ] } ^ { ( k ) } ) _ { k = 1 } ^ { L }$ , and a second phase comprising of seen/unseen classifiers $\left( \left[ M _ { [ 2 ] , i } ^ { ( k ) } \right] _ { i = 1 } ^ { n _ { k } } \right) _ { k = 1 } ^ { \bar { L } ^ { \prime } } ( n _ { k }$ copies for each $M _ { [ 1 ] } ^ { ( k ) }$ from the first phase).

Phase I: Embedding Models $( M _ { [ 1 ] } ^ { ( k ) } )$ . Each embedding model learns a mapping $M _ { [ 1 ] } ^ { ( k ) } : \mathcal { X }  \mathbb { R } ^ { d }$ that converts a word image into a structured embedding vector. The embedding encodes compositional properties of words including local letter shape descriptors, segment-level character presence and bigram statistics across diferent parts of a word. For each $1 \le k \le L$ , the model $M _ { [ 1 ] } ^ { ( k ) }$ is trained on a subset of the seen vocabulary ${ \mathcal W } _ { s } ^ { ( k ) } \subset { \mathcal W } _ { s }$ , while the remaining classes $( \mathrm { i . e . } \ \mathcal { W } _ { s } \cup \mathcal { W } _ { u } \setminus \mathcal { W } _ { s } ^ { ( k ) } )$ are treated as unseen for the k-th model $M _ { \left\lceil 1 \right\rceil } ^ { ( k ) }$ from Phase I. Training $1 \le k \le L$ such models produces a diverse ensemble where each model has partial knowledge of the vocabulary.

Phase II: Seen/Unseen Classifier $( M _ { [ 2 ] , i } ^ { ( k ) } )$ . Embedding models tend to be biased toward classes observed during training. In other words, the misidentification error for the training status of a data point x (generated from the distribution $P )$ , captured by the quantity $\mathbb { P } [ \hat { w } ( x ) \in \mathcal { W } _ { s } ^ { ( k ) } | w ( x ) \in \mathcal { W } _ { u } ^ { ( k ) } ]$ is high (for comparison, in a perfect classification mechanism, this probability should be 0). We define the indicator random variables $Z _ { k } ( \boldsymbol { x } ) : = \mathbb { 1 } [ \{ \boldsymbol { w } ( \boldsymbol { x } ) \in \mathcal { W } _ { s } ^ { ( k ) } \} ]$ and $\hat { Z } _ { k } ( x ) : = \mathbb { 1 } [ \{ M _ { \lceil 1 \rceil } ^ { ( k ) } ( x ) \in \mathcal { W } _ { s } ^ { ( k ) } \} ]$

To mitigate bias, in Phase II we introduce an ensemble of $n _ { k }$ classifiers (for each fixed $M _ { [ 1 ] } ^ { ( k ) }$ from Phase I). At the fine-grained level, each $\left( M _ { [ 2 ] , i } ^ { ( k ) } \right) _ { i = 1 } ^ { n _ { k } }$ is trained to learn the training category $Z _ { k } ( x )$ , with a 0-1 valued output $\ddot { Z } _ { k , i } ( x )$ The $M _ { [ 2 ] , i } ^ { ( k ) } – \mathrm { s }$ are typically trained using a (random) subset of the training data $\mathcal { W } _ { s }$ (say, around 80%). So their 0-1 valued outputs will be diferent and will embody their inherent randomness. In an unbiased scenario, a natural way to aggregate the $\hat { Z } _ { k , i } ( x )$ would be to take a majority vote among the $M _ { [ 2 ] , i } ^ { ( k ) } { } ^ { - \mathrm { s } }$ in order to determine the training status seen/unseen. But in view of the inherent bias of $Z _ { k }$ towards being 1, we compute declare the training status as “seen” only if the total output $\left( \Sigma _ { i = 1 } ^ { n _ { k } } \hat { Z } _ { k , i } ( x ) \right)$ crosses a threshold thres (to be thought of as significantly higher than $n _ { k } / 2 )$ . In practice, the choice of thres can reasonably be carried out via cross validation.

Low Dimensional Approximation. For practical purposes, we can additionally modify the the scheme outlined above by using a low dimensional projection $M _ { [ 1 ] } ^ { ( k ) } ( x )$ , via a standard dimension reduction methodology, such as Principal Component Analysis (PCA), and use this as an input to the Phase II classifiers $\left( \Pi [ M _ { [ 2 ] , i } ^ { ( k ) } ] \right) _ { i = 1 } ^ { n _ { \bar { k } } }$ . The ability to operate on a low-dimensional problem makes each $M _ { [ 2 ] , i } ^ { ( k ) }$ to be computationally light, which entails that we are able to deploy a very large number $n _ { k }$ of such classifiers leading to improved accuracy of the Monte Carlo method for estimation of training status. The trade-of is the approximation incurred due to working with the low-dimensional representation $\Pi [ M _ { [ 1 ] } ^ { ( k ) } ( x ) ]$ instead of the full word embedding $M _ { [ 1 ] } ^ { ( k ) } ( x )$

Inference. Given an input word image x, we will compute a prediction for the word $w ( x )$ for each $1 \leq k \leq L .$ , and subsequently aggregate these predictions. For a fixed $k ,$ , the embedding model $M _ { [ 1 ] } ^ { ( k ) }$ in Phase I produces $v _ { k } ( x ) : = M _ { [ 1 ] } ^ { ( k ) } ( x )$ . The $\left( M _ { [ 2 ] , i } ^ { ( k ) } \right) _ { i = 1 } ^ { n _ { k } }$ ensemble predicts whether the embedding corresponds to a seen or unseen class (for that Phase I model $M _ { [ 1 ] } ^ { ( k ) }$ . We denote this predicted training status as ${ \hat { T } } ^ { ( k ) }$ , which can take two values – seen/unseen. If ${ \hat { T } } ^ { ( k ) } =$ seen (resp., unseen), then the predicted word is the “best approximation” to $M _ { [ 1 ] } ^ { ( k ) } ( x )$ from the set ${ \mathcal W } _ { s } ^ { ( k ) }$ (resp. from the set ${ \ww W } \backslash { \boldsymbol { W } } _ { s } ^ { ( k ) } )$ .

The “best approximation” above can be computed via nearest neighbour

![](images/98de0e2e3e2b8a6cabc1c53d33b5971c4e51ae26bef228d7dad0b09cfc0a694f.jpg)

Figure 1: Schematic of the algorithmic pipeline for predicting words for IAM handwriting dataset

matching within the relevant subset of the vocabulary. For the seen class ${ \mathcal W } _ { s } ^ { ( k ) }$ This can optionally be substituted by a random forest or logistic regression based classifier trained on the corresponding data.

Predictions for $1 \le k \le L$ are aggregated (via most frequent prediction, with ties being broken via nearest neighbour matching) to obtain the final word prediction.

## 3 The Algorithmic Pipeline

We train the entire system of predicting $w ( x )$ in these stages:

• Training embedding model $M _ { [ 1 ] } ^ { ( k ) }$ . Algorithm 1 does sample splitting by taking a subset of the 250-by-50 pixel images x with $w ( x ) \in \mathcal { W } _ { s } ^ { ( k ) }$ . These splits are used to train $M _ { [ 1 ] } ^ { ( k ) }$ . For the values of the splitting fractions, we typically take $s \approx 0 . 1$ and $t \approx 0 . 7$

• Training ensemble seen/unseen classifier $M _ { [ 2 ] , - } ^ { ( k ) }$ . Algorithm 4 (c.f. Section A in the Appendix) is used to build a dataset to train one ensemble of $M _ { [ 2 ] , - } ^ { ( k ) }$ models for each k. The dataset includes two parts: synthetic data ‘far’ from the seen classes as unseen examples, and remaining training data $( X , X ^ { \prime } )$ not used for training $M _ { [ 1 ] } ^ { ( k ) }$ . Each $M _ { [ 2 ] , i } ^ { ( k ) }$ is a is a neural network that is trained with 80% of the dataset, so that each $M _ { [ 2 ] , i } ^ { ( k ) }$ in the ensemble produces meaningfully diferent results that can be combined. This ensemble predicts if the $M _ { [ 1 ] } ^ { ( k ) }$ has been trained on the word, i.e. $w ( x ) \in \mathcal { W } _ { s } ^ { ( k ) }$ , given $M _ { [ 1 ] } ^ { ( k ) } ( x )$

• (Optional) Training word predictor. Algorithm 5 (c.f. Section A in the Appendix) produces a dataset with two parts: training data with words in ${ \mathcal W } _ { s } ^ { ( k ) }$ that were not used to train $M _ { [ 1 ] } ^ { ( k ) }$ , and synthetic data. This is used for training a random forest (RF) or logistic regression (LR) model that could predict the word among ${ \dot { \mathcal W } } _ { s } ^ { ( k ) }$ (RF or LR could replace nearest-neighbour matching).

• Word prediction from single $M _ { [ 1 ] } ^ { ( k ) }$ . Algorithm 2 produces a prediction for each $M _ { [ 1 ] } ^ { ( k ) }$ for the images in the test set. We use $M _ { [ 1 ] } ^ { ( k ) } ( x )$ to get the estimated feature vector, then use $M _ { [ 2 ] , \cdot } ^ { ( k ) }$ to predict whether $w ( x ) \in \mathcal { W } _ { s } ^ { ( k ) }$ If it is predicted that $w ( x ) \in \mathcal { W } _ { s } ^ { ( k ) }$ , then Algorithm 6 (c.f. Section A in the Appendix) does nearest-neighbour matching among ${ \mathcal W } _ { s } ^ { ( k ) }$ (or uses the random forest/logistic regression) to give a prediction for the word. If it If is predicted that $w ( x ) \in \mathcal { W } \backslash \mathcal { W } _ { s } ^ { ( k ) }$ , then nearest-neighbour matching among ${ \ww W } \backslash { \boldsymbol { \mathcal { W } } _ { s } ^ { ( k ) } }$ is used.

• Aggregate word prediction from all $M _ { [ 1 ] } ^ { ( k ) }$ . Algorithm 3 takes the prediction from each $M _ { [ 1 ] } ^ { ( k ) }$ to form a final prediction. Algorithm 3 does this by taking the most frequent prediction, breaking ties by nearest-neighbour matching of $M _ { [ 1 ] } ^ { ( k ) } ( x )$

To train the $M _ { [ 1 ] } ^ { ( \bar { k } ) } \mathrm { s }$ in Algorithm 1, we use the same PhoS and PhoC vectors given in the original Pho(SC)Net approach [4]. In particular, for each of 8407 words, we form a 769-dimensional word embedding vector as follows:

• Shapes: The word is split into 1 to 5 parts (for 15 parts in total), each with 11 dimensions representing the shapes of the letters, for a total of 165 dimensions

• Letters: The word is split into 2 to 5 parts (for 14 parts in total), each with 36 dimensions representing letters appearing in that part, for a total of 504 dimensions

• Bigrams: The word is split into halves, each with 50 dimensions representing bigrams, for a total of 100 dimensions

Algorithm 1 Producing dataset splits and training $M _ { [ 1 ] } ^ { ( k ) }$   
Input: $\mathcal { D } _ { \mathrm { t r a i n } } ,$ the training data; $s , t \in ( 0 , 1 )$ , the data splitting frac  
tions.   
Output: $M _ { [ 1 ] } ^ { ( k ) } , T , X , X ^ { \prime } ;$ , where T represents the reduced training data   
for $M _ { [ 1 ] } ^ { ( k ) }$ , X will be the ‘seen’ training data for $M _ { [ 2 ] , i } ^ { ( k ) }$ , and $X ^ { \prime }$ will be the   
‘unseen’ training data for $M _ { [ 2 ] , i } ^ { ( k ) }$   
1: In $\mathcal { D } _ { \mathrm { t r a i n } } ,$ we randomly select the fraction s of the labels to be denoted as   
unseen, then filter out the training data with such labels and name the   
resulting dataframe $X ^ { \prime }$   
2: We further split the remaining $\mathcal { D } _ { \mathrm { t r a i n } } \backslash X ^ { \prime }$ , into $t : ( 1 - t )$ based on per-class   
splitting, naming the fraction t split as $T _ { i }$ , and the fraction $1 - t$ split as   
X.   
3: Train $M _ { [ 1 ] } ^ { ( k ) }$ on T.   
4: return ${ \dot { M } } _ { [ 1 ] } ^ { ( k ) } , T , X , X ^ { \prime } .$   
Algorithm 2 Predicting classes for each $M _ { [ 1 ] } ^ { ( k ) }$   
Input: d a test datapoint, $M _ { [ 1 ] } ^ { ( k ) } , M _ { [ 2 ] , - } ^ { ( k ) } , \mathcal { W } _ { s } ^ { ( k ) } , \mathcal { W } _ { u } ^ { ( k ) }$ , thres a threshold   
Output: Predictions[k] and ${ \dot { F } } [ k ]$ , prediction for word class and feature   
vector (resp.) for d from $M _ { [ 1 ] } ^ { \dot { ( k ) } }$   
1: $F [ k ] \gets M _ { [ 1 ] } ^ { ( k ) } ( \mathfrak { d } )$ ▷ Predicted feature vector   
2: count ← number of $M _ { [ 2 ] , - } ^ { ( k ) }$ that predict F corresponds to a ‘seen’ class   
(i.e., in ${ \mathcal W } _ { s } ^ { ( k ) } )$   
3: if count $\geq$ thres then   
4: Use Algorithm 6 to predict class of F among ${ \mathcal W } _ { s } ^ { ( k ) }$ ; store class to   
Predictions[k]   
5: else   
Predictions[k] ← nearest distance class from $\mathcal { W } _ { u } ^ { ( k ) }$ ▷ Predicted   
word class   
7: end if   
8: return Predictions[k], F[k]

Algorithm 3 Combining predictions   
Input: d a test datapoint, all the $F [ - ]$ , Predictions[−] (via Alg. 2)   
Output: Final\_Prediction, a finalized word prediction for d   
1: List the most frequent labels among Predictions[−]   
2: if there is a unique most frequent label then   
3: Final\_Prediction ← the unique most frequent label   
4: else   
5: For each of the most frequent labels c =Predictions[l], compute   
$\| e ( c ) - F [ l ] \|$   
6: Let c<sup>∗</sup> = argmin $_ { c } \| e ( c ) - F [ l ] \|$ in the above step   
7: Final\_Prediction $ c ^ { * }$   
8: end if   
9: return Final\_Prediction   
4 Results & Discussion

The IAM handwriting data [10] consists of 250 pixel by 50 pixel images of handwritten words, each which are one of 8407 words. 7898 of the words are given in the training set, while the other 509 words are kept as unseen classes. The task is to determine which of 8407 words is written.

We store the accuracy on unseen classes $A _ { u }$ , accuracy on seen classes $A _ { s }$ We compute the harmonic mean $\begin{array} { r } { h : = \frac { 2 A _ { s } A _ { u } } { A _ { s } + A _ { u } } } \end{array}$ to measure performance at both prediction tasks. As an alternate metric of efectiveness, we also investigate the balanced accuracy $A _ { b } : = \operatorname* { m i n } ( A _ { u } , A _ { s } )$

The original Pho(SC)Net approach involved training a model to predict the feature vector for image handwriting samples. As the table shows, there is a bias towards predicting seen classes, with significantly lower unseen class accuracy (0.77) than seen class accuracy (0.93).

The other experiments are improvements over the original Pho(SC)Net approach via the paradigm of using many small models $M _ { [ 2 ] , - } ^ { ( k ) }$ to determine if the feature vector corresponds to a seen class ${ \mathcal W } _ { s } ^ { ( k ) }$ or an unseen class ${ \mathcal W } _ { u } ^ { ( k ) }$ Afterwards, a threshold count ‘thres’ (in Algorithm 2) is chosen. If there are more models which report the feature vector as seen, compared to ‘thres’, the feature vector is considered seen, and prediction will only be done among seen classes. ‘thres’ was roughly chosen to maximize h, the harmonic mean of the unseen and seen accuracy.

We consider class selection methods: nearest-neighbour (NN), random forest (RF) and logistic regression (LR).

Table 1: Implementation details
<table><tr><td></td><td> $L$ </td><td> $n _ { k }$ </td><td>Dim. for  $M _ { [ 2 ] }$ </td><td>thres for h thres for</td><td> $A _ { b }$ </td></tr><tr><td> $M _ { [ 1 ] }$  (Original Pho(SC)Net) [4]</td><td>1</td><td>一</td><td></td><td></td><td>一</td></tr><tr><td>(21)  $M _ { [ 1 ] } + 5 0 M _ { [ 2 ] }$  (769 dim, NN)</td><td>21</td><td>50</td><td>769</td><td>49</td><td>49</td></tr><tr><td>(5)  $M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$  (30 dim, NN)</td><td>5</td><td>500</td><td>30</td><td>445</td><td>430</td></tr><tr><td>(5)  $M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$  (15 dim, NN)</td><td>5</td><td>500</td><td>15</td><td>440</td><td>400</td></tr><tr><td>(5)  $M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$  (15 dim, RF)</td><td>5</td><td>500</td><td>15</td><td>425</td><td>405</td></tr><tr><td>(5)  $M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$  (15 dim, LR)</td><td>5</td><td>500</td><td>15</td><td>435</td><td>420</td></tr></table>

Table 2: Experimental results on handwritten test dataset
<table><tr><td></td><td> $A _ { u }$ </td><td> $A _ { s }$ </td><td> $h$ </td><td> $A _ { b }$ </td></tr><tr><td> $M _ { [ 1 ] }$  (Original Pho(SC)Net) [4]</td><td>0.77</td><td>0.93</td><td>0.84</td><td></td></tr><tr><td>(21)  $M _ { [ 1 ] } + 5 0 M _ { [ 2 ] }$  (769 dim, NN)</td><td>0.931</td><td>0.903</td><td>0.917</td><td>0.903</td></tr><tr><td>(5)  $M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$  (30 dim, NN)</td><td>0.916</td><td>0.857</td><td>0.886</td><td>0.877</td></tr><tr><td>(5)  $M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$  (15 dim, NN)</td><td>0.905</td><td>0.850</td><td>0.877</td><td>0.864</td></tr><tr><td>(5)  $M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$  (15 dim, RF)</td><td>0.905</td><td>0.847</td><td>0.875</td><td>0.867</td></tr><tr><td>(5)  $M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$  (15 dim, LR)</td><td>0.900</td><td>0.829</td><td>0.863</td><td>0.847</td></tr></table>

(21) $M _ { [ 1 ] } + 5 0 M _ { [ 2 ] }$ (769 dim, NN) is trained without dimension reduction, and achieves the best performance, with the highest $h = 0 . 9 1 7 .$ Each $M _ { [ 2 ] , i } ^ { ( k ) }$ has 10 ReLU hidden layers of 256 nodes.

![](images/33f5521d11a152cd83bf50d0337a673c0a66fb02eb370f5e3f10c45dfb8c59b0.jpg)

![](images/f87f04f7ebb882ec911077876b7bc2fad6a8d606cbdc7d33f83831218c9e6b42.jpg)  
Figure 2: Plot of cumulative sum of eigenvalues, ROCs for ensemble $M _ { [ 2 ] , } ^ { ( k ) }$ model

We can consider the covariance matrix of the class embedding vectors. We see that the top 15 eigenvalues account for 0.6224 of the sum of all eigenvalues while the top 30 eigenvalues account for 0.7592 of the sum of all eigenvalues. Figure 2 shows the the sum of largest eigenvalues of the covariance matrix, as a fraction the sum of all the eigenvalues of the covariance matrix.

The ROCs for the ensemble $\Breve { M } _ { [ 2 ] , } ^ { ( k ) }$ model (for each k) are computed by weighting the test set, to give equal total weight to the tests where $w ( x ) \in \mathcal { W } _ { s }$ and the tests where $w ( x ) \in \mathcal { W } _ { u } ,$ of which there are a significantly diferent number of both tests. The ensembles used here were of 500 30-dimensional $M _ { [ 2 ] , i } ^ { ( k ) }$ . We can see the 5 diferent ensembles have very similar curves.

For both $( 5 ) M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$ (15 dim, NN) and $( 5 ) M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$ (30 dim, NN), each $M _ { [ 2 ] , i } ^ { ( k ) }$ has 10 ReLU hidden layers of 32 nodes.

$( 5 ) M _ { [ 1 ] } + \overleftarrow { 5 } 0 0 M _ { [ 2 ] }$ (15 dim, RF). It is the same as $( 5 ) M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$ (15 dim, NN), but we add a RF, trained on 15-dimensional representations with scikit-learn with a depth of 10. The seen RF turns out to be 5GB per M1. One limitation is that the computational cost is intensive in the height of the tree.

(5) $M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$ (15 dim, LR). It is the same as (5) $M _ { [ 1 ] } + 5 0 0 M _ { [ 2 ] }$ (15 dim, NN), but we add a LR, trained on 15-dimensional representations with scikit-learn with 1000 iterations. One limitation is numerical stability.

All the variations of our proposed methodology provide a significant improvement over the Pho(SC)Net (c.f. Table 2). In particular, for the crucial problem of unseen class detection, we are able achieve an accuracy improvement of 0.931 (from 0.77). The overall GZSL accuracy (h) also improves 0.917 from 0.84. While the (21) $M _ { [ 1 ] } + 5 0 M _ { [ 2 ] }$ ensemble approach is expectedly the most accurate $( h = 0 . 9 1 7 )$ , the other approaches involving dimension reduction are also much more accurate than the original Pho(SC)Net. Thus, low-dimensional ensemble approaches provide a computationally lighter alternative to reduce bias for practical applications in various GZSL problems.

Note: All the data analyses reported in this section were conducted on a shared compute cluster with 1 NVIDIA A40 (48GB), 36 cores Intel 9452Y CPUs and 256GB RAM. The most expensive computation, (21) $M _ { [ 1 ] } + 5 0 M _ { [ 2 ] }$ (769 dimensions, NN) took approximately a week of compute, whereas the other experiments can be run in 3 days, for a total of roughly 3 weeks of compute.

## 5 Theoretical Underpinnings of Ensembling for Bias Reduction

In this section, we theoretically justify the idea of using an ensembling strategy for reducing the bias towards seen classes by considering a simplified model which is amenable to mathematical analysis. Suppose we have a dataset consisting of pairs $( x _ { i } , w _ { i } )$ , where each $w _ { i } \in \mathcal { W } _ { s }$ . Suppose we have $L$ binary predictors $\psi _ { k } , 1 \le k \le L$ , where $\psi _ { k }$ is trained on a subset of the available data by restricting to certain class labels ${ \mathcal W } _ { s } ^ { ( k ) } \subset { \mathcal W } _ { s }$ . Given an new observation $( x , w ) , \psi _ { k }$ is supposed to identify whether the true class label w of x is from among the classes it has seen during its training. As such it gives binary outputs ‘seen’ or ‘unseen’. We shall assume that $\mathcal { W } _ { s } = \cup _ { k = 1 } ^ { L } \mathcal { W } _ { s } ^ { ( k ) }$ . Then $\mathcal { W } _ { u } = \mathcal { W } \backslash \mathcal { W } _ { s }$ are the common unseen classes. We note here that by sample splitting, one may ensure that the predictors $\psi _ { k }$ are independent (albeit at the loss of some eficiency). In the supplementary material, we analyse the case where the $\psi _ { k } ? $ s are only assumed to be exchangeable.

A simple example of such binary predictors would be as follows. For each $k ,$ we fit a Gaussian mixture model to the corresponding training data. Denoting by $\hat { \phi } _ { k }$ the fitted the probability density function, we declare $\psi _ { k } ( x ) = \cdot _ { \mathrm { u n s e e n } } ,$ if $\hat { \phi } _ { k }$ has a ‘low’ value at x.

Let us introduce shorthands for the ‘Type 1’ and ‘Type $2 ^ { \cdot }$ errors of the binary predictors:

$$
\alpha _ { k } : = \mathbb { P } \big ( \psi _ { k } ( x ) = \mathrm { \mathfrak { s } _ { u n s e e n } } ^ { \prime } \mid w \in \mathcal { W } _ { s } ^ { ( k ) } \big ) , \qquad \beta _ { k } : = \mathbb { P } \big ( \psi _ { k } ( x ) = \mathrm { \mathfrak { s } _ { s e e n } } ^ { \prime } \mid w \notin \mathcal { W } _ { s } ^ { ( k ) } \big ) .
$$

We shall assume that $\alpha _ { k } + \beta _ { k } \le 1$ . Indeed, a random predictor has $\alpha _ { k } + \beta _ { k } = 1$ so we are assuming the predictor $\psi _ { k }$ to be no worse than random.

Now let $E _ { m }$ be the ensemble predictor that outputs ‘seen’ if and only if at least m of the classifiers $\psi _ { k }$ output ‘seen’. We shall show that for appropriate choices of $m , E _ { m }$ achieves high accuracy in classifying both ‘seen’ and ‘unseen classes.

We shall present our results on the accuracy of $E _ { m }$ under two diferent class-allocation mechanisms.

(i) Arbitrary allocation. Seen classes are assigned to ${ \mathcal W } _ { s } ^ { ( k ) }$ in some arbitrary manner.

(ii) Random allocation. Each seen class $w \in \mathcal { W } _ { s }$ is included in ${ \mathcal W } _ { s } ^ { ( k ) }$ with probability $p > 0$ , and this procedure is repeated independently across k.

We define a few key quantities to state our results. For $w ^ { \prime } \in \mathcal { W } _ { s }$ , let $\begin{array} { r l } { \textstyle } & { { } \mathcal { L } ( w ^ { \prime } ) : = } \end{array}$ $\{ k \mid w ^ { \prime } \in \mathcal { W } _ { s } ^ { ( k ) } \}$ . Let $\begin{array} { r } { \gamma _ { k } : = 1 - \alpha _ { k } - \beta _ { k } \ge 0 , \Delta ( w ^ { \prime } ) : = \sum _ { k \in \mathcal { L } ( w ^ { \prime } ) } } \end{array}$ γ<sub>k</sub> and $\Delta : = \mathrm { m i n } _ { w ^ { \prime } \in { \mathcal W } _ { s } } \Delta ( w ^ { \prime } )$

Theorem 1 (Arbitrary allocation). Assume that $\gamma _ { k } > 0$ for all $k \in [ L ] ,$ , so that $\Delta > 0$ , Then, $f o r m = \lceil \mu ( 1 + \delta ) \rceil$ , where $\begin{array} { r } { \mu = \sum _ { k \in [ L ] } \beta _ { k } } \end{array}$ and $\begin{array} { r } { \delta = \frac { \Delta } { 2 \mu + \Delta } , } \end{array}$ we have

$$
\begin{array} { r l } & { \mathbb { P } ( E _ { m } ( x ) =  \vphantom { \sum _ { i } }  u n s e e n ^ { \prime } | w \in \mathcal { W } _ { s } ) \leq e ^ { - \frac { \delta ^ { 2 } ( \mu + \Delta ) } { 2 } } , } \\ & { \quad \mathbb { P } ( E _ { m } ( x ) =  \vphantom { \sum _ { i } }  s e e n ^ { \prime } | w \in \mathcal { W } _ { u } ) \leq e ^ { - \frac { \delta ^ { 2 } \mu } { 2 + \delta } } . } \end{array}
$$

When $\Delta$ is bounded away from zero, and $\mu = \Theta ( L )$ , we see from Theorem 1 that $E _ { m }$ is able to classify both seen and unseen classes with high accuracy.

Theorem 2 (Random allocation). Let $\Delta _ { 0 } > 0$ be a deterministic threshold. $F o r m = \lceil \mu ( 1 + \delta ) \rceil$ , where $\begin{array} { r } { \mu = \sum _ { k \in [ L ] } \beta _ { k } } \end{array}$ and $\begin{array} { r } { \delta = \frac { \Delta _ { 0 } } { 2 \mu + \Delta _ { 0 } } } \end{array}$ , we have

$$
\begin{array} { r l } & { \mathbb { P } \big ( E _ { m } ( x ) = \left. \mathrm { \# } n s e e n ^ { \prime } \right| w \in \mathcal { W } _ { s } \big ) \leq \mathbb { P } \big ( \Delta < \Delta _ { 0 } \big ) + \mathbb { E } \big [ e ^ { - \delta ^ { 2 } ( \mu + \Delta _ { 0 } ) / 2 } \big ] , } \\ & { \quad \mathbb { P } \big ( E _ { m } ( x ) = \left. \mathrm { \# } _ { s e e n ^ { \prime } } \right| w \in \mathcal { W } _ { u } \big ) \leq \mathbb { P } \big ( \Delta < \Delta _ { 0 } \big ) + \mathbb { E } \big [ e ^ { - \delta ^ { 2 } \mu / ( 2 + \delta ) } \big ] . } \end{array}
$$

If we further assume that the $\alpha _ { k }$ ’s and the $\beta _ { k }$ ’s are statistically independent of the class allocation mechanism, then, with $\eta \in ( 0 , 1 )$ and $\Delta _ { 0 } : = ( 1 -$ $\begin{array} { r } { \eta ) p \sum _ { k \in [ L ] } \gamma _ { k } } \end{array}$ , we have

$$
\begin{array} { r l } & { \mathbb { P } \big ( E _ { m } ( x ) = { \mathit { \iota } } _ { u n s e e n } { \mathrm { \iota } } | { \mathit { \omega } } w \in { \mathcal { W } } _ { s } \big ) \le | { \mathcal { W } } _ { s } | e ^ { - \frac { 2 \eta ^ { 2 } p ^ { 2 } ( \sum _ { k \in [ L ] } \gamma _ { k } ) ^ { 2 } } { \sum _ { k \in [ L ] } \gamma _ { k } ^ { 2 } } } + e ^ { - \delta ^ { 2 } ( \mu + \Delta _ { 0 } ) / 2 } , } \\ & { \quad \mathbb { P } \big ( E _ { m } ( x ) = { \mathit { \iota } } _ { s e e n } { \mathrm { \iota } } | { \mathit { \omega } } w \in { \mathcal { W } } _ { u } \big ) \le | { \mathcal { W } } _ { s } | e ^ { - \frac { 2 \eta ^ { 2 } p ^ { 2 } ( \sum _ { k \in [ L ] } \gamma _ { k } ) ^ { 2 } } { \sum _ { k \in [ L ] } \gamma _ { k } ^ { 2 } } } + e ^ { - \delta ^ { 2 } \mu / ( 2 + \delta ) } . } \end{array}
$$

We see that random allocations, assuming independence of the $\alpha _ { k } \mathrm { ^ { * } s }$ and the $\beta _ { k } \mathrm { ^ { * } s }$ from the allocation scheme, the ensemble classifier $E _ { m }$ is able to identify both seen and unseen classes with high accuracy, if $\mu$ is large, $\delta$ is bounded away from 0, and $\frac { ( \sum _ { k \in [ L ] } \gamma _ { k } ) ^ { 2 } } { \sum _ { k \in [ L ] } \gamma _ { k } ^ { 2 } }$ is large. For instance, if we assume a uniform lower bound $\gamma _ { k } = 1 - \overset { \cdot } { \alpha _ { k } } - \beta _ { k } \geq \varepsilon > 0$ , then $\Delta _ { 0 } \geq ( 1 - \eta ) p \varepsilon L$ and $\begin{array} { r } { \frac { ( \sum _ { k = 1 } ^ { L } \gamma _ { k } ) ^ { 2 } } { \sum _ { k = 1 } ^ { L } \gamma _ { k } ^ { 2 } } \geq \varepsilon ^ { 2 } L } \end{array}$

We can see Algorithm 3 using multiple predictions to form a single predic tion by taking the most common-occurring prediction. Each $M _ { [ 1 ] } ^ { ( k ) }$ is biased towards predicting words in ${ \mathcal W } _ { s } ^ { ( k ) }$ , afecting the accuracy of words in ${ \mathcal W } \backslash { \mathcal W } _ { s } ^ { ( k ) }$ especially words not in the training set $( \mathrm { i n } \ \mathcal { W } _ { u } )$ . In our approach, multiple $\bar { M } _ { \mathrm { [ 1 ] } } ^ { ( k ) }$ trained on diferent ${ \mathcal W } _ { s } ^ { ( k ) }$ give some robustness against an incorrect prediction for words inside $\mathcal { W } _ { u }$

## 6 Concluding Remarks

In this work, we have devised a novel methodology for rectifying the inherent bias toward seen classes in genralised zero shot learning in the context of hand written word recognition. Broadly speaking, it is a statistical methodology based on an ensembling strategy that leverages a two stage architecture, with the second stage explicitly focusing on bias mitigation. The proposed method is very general and can in principle be augmented to any classical GZSL learner in a turn-key fashion to implement bias correction. The present work is focused on the handwritten word recognition appplication, and extensions of this approach to other GZSL scenarios is a natural area for follow-up investigation. We provide a foundational theoretical structure to complement our methodology, and extensions of this theory to cover wider application domains and weaker hypothesis setups is another natural direction of research.

## 7 Acknowledgements

The authors would like to acknowledge that computational work involved in this research work is partially supported by NUS IT’s Research Computing group using grant number NUSREC-HPC-00001. SG was supported in part by the NUS Dean’s Chair Associate Professorship E-146-00-0037-01 and the Singapore MOE grants A-8002014-00-00 and A-8003802-00-00.

## References

[1] Zeynep Akata, Scott Reed, Daniel Walter, Honglak Lee, and Bernt Schiele. Evaluation of output embeddings for fine-grained image classification. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2927–2936. IEEE, 2015.

[2] Jon Almazán, Albert Gordo, Alicia Fornés, and Ernest Valveny. Word spotting and recognition with embedded attributes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(12):2552–2566, 2014.

[3] Rina Foygel Barber. Hoefding and bernstein inequalities for weighted sums of exchangeable random variables. Electronic Communications in Probability, 2024.

[4] Ravi Bhatt, Anuj Rai, Narayanan C. Krishnan, and Sukalpa Chanda. Pho(sc)-CTC: a hybrid approach towards zero-shot word image recognition in historical documents. International Journal on Document Analysis and Recognition (IJDAR), 25(4):357–374, 2022.

[5] Wei-Lun Chao, Soravit Changpinyo, Boqing Gong, and Fei Sha. An empirical study and analysis of generalized zero-shot learning for object recognition in the wild. In European Conference on Computer Vision (ECCV), pages 52–68. Springer, 2016.

[6] Andrea Frome, Greg S. Corrado, Jonathon Shlens, Samy Bengio, Jef Dean, Tomas Mikolov, et al. DeViSE: A deep visual-semantic embedding model. In Advances in Neural Information Processing Systems 26 (NeurIPS 2013), pages 2121–2129. Curran Associates, Inc., 2013.

[7] Alex Graves, Santiago Fernández, Faustino Gomez, and J"urgen Schmidhuber. Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd International Conference on Machine Learning (ICML), pages 369–376. ACM, 2006.

[8] Praveen Krishnan, Kartik Dutta, and C. V. Jawahar. Deep feature embedding for accurate recognition and retrieval of handwritten text. In 2016 15th International Conference on Frontiers in Handwriting Recognition (ICFHR), pages 289–294. IEEE, 2016.

[9] Christoph H. Lampert, Hannes Nickisch, and Stefan Harmeling. Learning to detect unseen object classes by between-class attribute transfer. In 2009 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 951–958. IEEE, 2009.

[10] U.-V. Marti and H. Bunke. The iam-database: an english sentence database for ofline handwriting recognition. International Journal on Document Analysis and Recognition, 5(1):39–46, November 2002.

[11] Mark Palatucci, Dean Pomerleau, Geofrey E. Hinton, and Tom M. Mitchell. Zero-shot learning with semantic output codes. In Advances in Neural Information Processing Systems 22 (NeurIPS 2009), pages 1410–1418. Curran Associates, Inc., 2009.

[12] Anuj Rai, Narayanan C. Krishnan, and Sukalpa Chanda. Pho(sc)net: An approach towards zero-shot word image recognition in historical documents. In Document Analysis and Recognition – ICDAR 2021 Workshops, volume 12918 of Lecture Notes in Computer Science, pages 17–30. Springer, 2021.

[13] Richard Socher, Milind Ganjoo, Christopher D. Manning, and Andrew Ng. Zero-shot learning through cross-modal transfer. In Advances in Neural Information Processing Systems 26 (NeurIPS 2013), pages 935– 943. Curran Associates, Inc., 2013.

[14] Sebastian Sudholt and Gernot A. Fink. Evaluating word string embeddings and loss functions for CNN-based word spotting. In 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), pages 915–920. IEEE, 2017.

[15] Vinay Kumar Verma, Gundeep Arora, Ashish Mishra, and Piyush Rai. Generalized zero-shot learning via synthesized examples. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4281–4289. IEEE, 2018.

[16] Vinay Kumar Verma, Dhanajit Brahma, and Piyush Rai. Meta-learning for generalized zero-shot learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 6062–6069, 2020.

[17] Yongqin Xian, Christoph H. Lampert, Bernt Schiele, and Zeynep Akata. Zero-shot learning—a comprehensive evaluation of the good, the bad and the ugly. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(9):2251–2265, 2019.

[18] Yongqin Xian, Bernt Schiele, and Zeynep Akata. An empirical study and analysis of generalized zero-shot learning for object recognition in

the wild. In European Conference on Computer Vision (ECCV), pages 52–68. Springer, 2016.

## A Supporting Algorithms

Algorithm 4 is used to generate unseen data for $M _ { [ 2 ] , - } ^ { ( k ) }$ . This is run after Algorithm 1 and before Algorithm 2. This uses the datasets $X , X ^ { \prime }$ (which were not used to train $M _ { [ 1 ] } ^ { ( k ) } )$ as well as additional synthetic data.

```latex
Algorithm 4 Synthetically generating unseen data for $M _ { [ 2 ] , - } ^ { ( k ) }$
Input: $M _ { [ 1 ] } ^ { ( k ) } , X , X ^ { \prime } , L , R \triangleright L$ and $R$ are the lower and upper bounds of
the extremes respectively
Output: $\hat { \hat { X } } .$ , consisting of feature vectors derived from $X \cup X ^ { \prime }$ and
synthetically generated data
1: $N \gets \mathrm { l e n g t h } ( X \cup X ^ { \prime } )$ ▷ number of instances to generate
2: dim ← size of feature vector
3: Initialize datasets $Z , Z ^ { \prime }$
4: for each $X _ { i }$ of X do
5: $Z [ i ]  M _ { [ 1 ] } ^ { ( k ) } ( X _ { i } ) \triangleright Z$ contains feature vectors of $X ,$ predicted by $M _ { [ 1 ] } ^ { ( k ) }$
6: end for
7: for each $X _ { i } ^ { \prime }$ of $X ^ { \prime }$ do
8: $Z ^ { \prime } [ i ]  M _ { [ 1 ] } ^ { ( k ) } ( X _ { i } ^ { \prime } ) \triangleright Z ^ { \prime }$ contains the feature vectors of $X ^ { \prime } ,$ predicted
by $M _ { [ 1 ] } ^ { ( k ) }$
9: end for
10: Calculate mean $( \mu )$ and standard deviation $( \sigma )$ for each coordinate of
feature vectors in $Z .$
11: $\hat { X ^ { \prime } } \gets \mathrm { a r r a y } ( N ,$ dim)
▷ Store the feature vectors of the synthetically generated training unseen,
used to train $M _ { [ 2 ] , - } ^ { ( k ) }$
12: for each $X ^ { \prime } [ i ]$ of $X ^ { \prime }$ do
13: 95% of the coordinates of $\hat { X } ^ { \prime } [ i ]$ are given an extreme sample
▷ from $\leq L \%$ quantile or $\geq R \%$ quantile of $N ( \mu _ { j } , \sigma _ { j } )$
14: The rest 5% of coordinates of $\hat { X } ^ { \prime } [ i ]$ are given middle samples
▷ from between $L \%$ quantile and R% quantile of $N ( \mu _ { j } , \sigma _ { j } )$
15: end for
16: return a dataframe $\hat { X }$ that consists of $Z , Z ^ { \prime }$ and ${ \hat { X } } ^ { \prime } ,$ attaching 1 to the
feature vectors of $Z ,$ and 0 to $Z ^ { \prime } , \hat { X } ^ { \prime } . \ \triangleright 1$ means it is seen, 0 means it is
unseen.
```

Algorithm 5 (optional) is used to generate data for random forest/logistic regression approaches. This is run after Algorithm 1 and before Algorithm 2. This uses the dataset X (which was not used to train $M _ { [ 1 ] } ^ { ( k ) } )$ as well as additional synthetic data.

Algorithm 5 Synthetically generating seen data for random forest/logistic   
regression   
Input: $M _ { [ 1 ] } ^ { ( k ) } , X$   
Output: Dataset $X _ { \mathrm { s e e n } }$ for training random forest   
1: m ← maximum number of training samples among ${ \mathcal W } _ { s } ^ { ( k ) }$ in $X$   
2: D stores diferences between feature vectors and the corresponding class   
3: for each $X _ { i }$ in X do   
4: $F  M _ { [ 1 ] } ^ { ( k ) } ( X _ { i } )$   
5: Add feature vector $F$ and class $w ( X _ { i } )$ to $X _ { \mathrm { s e e n } }$   
6: Add the diference $F - e ( w ( X _ { i } ) )$ to D   
7: end for   
8: for each class $c \in \mathcal { W } _ { s } ^ { ( k ) }$ do   
9: while class has $<$ m samples do   
10: v ← Uniform sample from D   
11: Add $\boldsymbol { v } + e ( c )$ and class to $X _ { \mathrm { s e e n } }$   
12: end while   
13: end for   
14: return $X _ { \mathrm { s e e n } }$

Algorithm 6 is used to apply the random forest/logistic regression (if available) otherwise it applies the nearest-neighbour approach. This is run during Algorithm 2.

Algorithm 6 Predict class of feature vector among seen classes   
Input: F a feature vector, ${ \mathcal W } _ { s } ^ { ( k ) }$ , RF an optional random forest, LR an   
optional logistic regression.   
Output: Class prediction for feature vector   
1: if RF available then   
2: return $\operatorname { R F } ( F )$   
3: else if LR available then   
4: return LR(F)   
5: else   
6: return nearest distance class from ${ \mathcal W } _ { s } ^ { ( k ) }$   
7: end if

## B Further Theoretical Details

## B.1 Why Use $E _ { m }$ over a Consensus Voting?

To remove the bias toward seen classes, the simplest ensembling method that comes to mind is to declare the class of a new example as ‘seen‘ if all the classifiers $\psi _ { k }$ call it ‘seen’. Let us call this consensus voting rule $E _ { \mathrm { a l l } } ( x )$

Then, assuming independence of the predictors $\psi _ { k }$ , it follows that

$$
\mathbb { P } ( E _ { \mathrm { a l l } } ( x ) = \mathrm { ` u n s e e n ' } \mid w \in \mathcal { W } _ { u } ) = 1 - \prod _ { k \in [ L ] } \beta _ { k } \geq 1 - \beta ^ { L } ,
$$

provided a mild upper bound $\beta _ { k } \le \beta < 1$ holds on the Type II errors of the predictors.

Thus the consensus rule $E _ { \mathrm { a l l } }$ is able to remove the bias toward ‘seen classes. However, it is too stringent when it comes to correctly identifying ‘seen’ classes. Recall that $\mathcal { L } ( w ) : = \{ k \ | \ w \in \mathcal { W } _ { s } ^ { ( k ) } \}$ and set $\ell ( w ) : = \# \mathcal { L } ( w )$ In other words, ℓ(w) many predictors had w as a seen class in their training sets. Then,

$$
\begin{array} { r l } & { \mathbb { P } ( E _ { \mathrm { a l l } } ( x ) = \mathrm { \ ` s e e n ' } | w \in \mathcal { W } _ { s } ) = \mathbb { P } ( \mathrm { a l l ~ } \psi _ { k } \mathrm { \normalfont ~ c a l l ~ } x \mathrm { \normalfont ~ s e e n } ) } \\ & { \qquad = \displaystyle \prod _ { k : w \in \mathcal { W } _ { s } ^ { ( k ) } } ( 1 - \alpha _ { k } ) \prod _ { k : w \notin \mathcal { W } _ { s } ^ { ( k ) } } \beta _ { k } } \\ & { \qquad \geq ( 1 - \alpha ) ^ { \ell ( w ) } \beta ^ { L - \ell ( w ) } , } \end{array}
$$

where the last inequality holds if $\alpha _ { k } \leq \alpha$ and $\beta _ { k } \ge \beta$ for each $k \in [ L ]$ . The takeaway is that the accuracy in identifying ‘seen’ classes can be quite far from 1.

The above computation suggests the use of a less stringent voting rule such as $E _ { m }$ with an appropriate choice of m.

## B.2 Proofs of Various Results

Recall that

$$
\Delta ( w ) : = \sum _ { k \in \mathcal { L } ( w ) } \gamma _ { k } , \qquad \Delta : = \operatorname* { m i n } _ { w \in \mathcal { W } _ { s } } \Delta ( w ) .
$$

Proof of Theorem 1. Let

$$
X : = \sum _ { k \in [ L ] } \mathbb { 1 } ( \psi _ { k } ( x ) = { \mathrm { ' s e e n ' } } ) .
$$

Then X is a sum of independent Bernoulli random variables. Note that

$$
\begin{array} { l } { \displaystyle \mu _ { s } : = \operatorname { \mathbb E } [ X \mid w \in \mathcal { W } _ { s } ] = \sum _ { k \in \mathcal { L } ( w ) } ( 1 - \alpha _ { k } ) + \sum _ { k \in [ L ] \backslash \mathcal { L } ( w ) } \beta _ { k } } \\ { = \displaystyle \sum _ { k \in \mathcal { L } ( w ) } ( 1 - \alpha _ { k } - \beta _ { k } ) + \sum _ { k \in [ L ] } \beta _ { k } } \\ { = \Delta ( w ) + \mu } \\ { \geq \Delta + \mu . } \end{array}
$$

On the other hand,

$$
\mu _ { u } : = \mathbb { E } [ X \mid w \in \mathcal { W } _ { u } ] = \sum _ { k \in [ L ] } \beta _ { k } = \mu
$$

For our choice of $\delta ,$ it may be seen that

$$
( 1 + \delta ) \mu = \frac { 2 \mu ( \mu + \Delta ) } { 2 \mu + \Delta } \leq \frac { 2 \mu \mu _ { s } } { 2 \mu + \Delta } = ( 1 - \delta ) \mu _ { s } .
$$

Applying the Chernof bounds, we have

$$
\begin{array} { r l } & { \mathbb { P } ( E _ { m } ( x ) = \mathrm { \langle u n s e e n ' ~ | ~ } w \in \mathcal { W } _ { s } ) = \mathbb { P } ( X < m \mid w \in \mathcal { W } _ { s } ) } \\ & { \qquad = \mathbb { P } ( X < ( 1 + \delta ) \mu \mid w \in \mathcal { W } _ { s } ) } \\ & { \qquad \le \mathbb { P } [ X < ( 1 - \delta ) \mu _ { s } \mid w \in \mathcal { W } _ { s } ] } \\ & { \qquad \le e ^ { - \delta ^ { 2 } \mu _ { s } / 2 } . } \end{array}
$$

Again by the Chernof bounds,

$$
\begin{array} { r l } & { \mathbb { P } ( E _ { m } ( x ) = \mathrm { \langle s e e n ' ~ \vert ~ } w \notin \mathcal { W } _ { s } ) = \mathbb { P } ( X \geq m \mid w \in \mathcal { W } _ { u } ) } \\ & { \qquad = \mathbb { P } ( X \geq ( 1 + \delta ) \mu \mid w \in \mathcal { W } _ { u } ) } \\ & { \qquad \leq e ^ { - \delta ^ { 2 } \mu / ( 2 + \delta ) } . } \end{array}
$$

This completes the proof.

Proof of Theorem 2. We note that under the random allocation model, the indicators $X _ { w , k } : = \mathbb { 1 } ( w \in \mathcal { W } _ { s } ^ { ( k ) } )$ are i.i.d. Ber $( p )$ random variables. Now

$$
\Delta ( w ) = \sum _ { k \in [ L ] } \gamma _ { k } X _ { w , k } ,
$$

where $\gamma _ { k } = 1 - \alpha _ { k } - \beta _ { k }$ . Note that the $\gamma _ { k } \mathrm { ^ { * } s }$ a priori depend on the allocated classes $\mathcal { A } : = ( \mathcal { W } _ { s } ^ { ( k ) } ) _ { k \in [ L ] }$ . Given an allocation ${ \mathcal { A } } ,$ we may repeat the argument used in the proof of Theorem 1. Let $\Delta _ { 0 }$ denote a deterministic threshold such that $\mathbb { P } ( \Delta \leq \Delta _ { 0 } )$ is small. Then, with $\begin{array} { r } { \delta = \frac { \Delta _ { 0 } } { 2 \mu + \Delta _ { 0 } } } \end{array}$ and $m = \lceil \mu ( 1 + \delta ) \rceil$ , we have

$$
\begin{array} { r l } & { \mathbb { P } ( E _ { m } ( x ) = \mathrm { \ " u n s e e n ~ " } | ~ w \in \mathcal { W } _ { s } ) = \mathbb { E } \left[ \mathbb { P } ( E _ { m } ( x ) = \mathrm { \ " u n s e e n ~ " } | ~ w \in \mathcal { W } _ { s } , \mathcal { A } ) \right] } \\ & { \qquad = \mathbb { E } \left[ \mathbb { P } ( E _ { m } ( x ) = \mathrm { \ " u n s e e n ~ " } | ~ w \in \mathcal { W } _ { s } , \mathcal { A } ) \mathbb { 1 } ( \Delta < \Delta _ { 0 } ) \right] } \\ & { \qquad + \mathbb { E } \left[ \mathbb { P } ( E _ { m } ( x ) = \mathrm { \ " u n s e e n ~ " } | ~ w \in \mathcal { W } _ { s } , \mathcal { A } ) \mathbb { 1 } ( \Delta \geq \Delta _ { 0 } ) \right] } \\ & { \qquad \leq \mathbb { P } ( \Delta < \Delta _ { 0 } ) + \mathbb { E } \left[ e ^ { - \delta ^ { 2 } ( \mu + \Delta _ { 0 } ) / 2 } \right] . } \end{array}
$$

Similarly,

$$
\mathbb { P } \left( E _ { m } ( x ) = \mathrm { \ " u n s e e n " } | w \in \mathcal { W } _ { s } \right) \le \mathbb { P } ( \Delta < \Delta _ { 0 } ) + \mathbb { E } \left[ e ^ { - \delta ^ { 2 } \mu / ( 2 + \delta ) } \right] .
$$

If we further assume that the $\alpha _ { k } \mathrm { { ' s } }$ and the $\beta _ { k } \mathrm { ^ { * } s }$ do not depend on the class allocations ${ \mathcal { A } } ,$ then the $\Delta ( w ) \mathrm { s }$ are independent random variables, and hence by Hoefding’s inequality, for any $\eta \in ( 0 , 1 )$ ,

$$
\begin{array} { r } { \mathbb { P } \bigg ( \Delta ( w ) \leq ( 1 - \eta ) p \sum _ { k \in [ L ] } \gamma _ { k } \bigg ) = \mathbb { P } ( \Delta ( w ) \leq ( 1 - \eta ) \mathbb { E } [ \Delta ( w ) ] ) } \\ { \leq \exp \bigg ( - \frac { 2 \eta ^ { 2 } p ^ { 2 } \left( \sum _ { k \in [ L ] } \gamma _ { k } \right) ^ { 2 } } { \sum _ { k \in [ L ] } \gamma _ { k } ^ { 2 } } \bigg ) . } \end{array}
$$

It follows by a union bound that

$$
\mathbb { P } \bigg ( \Delta \leq ( 1 - \eta ) p \sum _ { k \in [ L ] } \gamma _ { k } \bigg ) \leq | \mathcal { W } _ { s } | \exp \bigg ( - \frac { 2 \eta ^ { 2 } p ^ { 2 } ( \sum _ { k \in [ L ] } \gamma _ { k } ) ^ { 2 } } { \sum _ { k \in [ L ] } \gamma _ { k } ^ { 2 } } \bigg ) .
$$

Thus we may set $\begin{array} { r } { \Delta _ { 0 } : = ( 1 - \eta ) p \sum _ { k \in [ L ] } \gamma _ { k } } \end{array}$ and obtain

$$
\mathbb { P } ( E _ { m } ( x ) = \mathrm { \mathrm { : } u n s e e n " | } ~ w \in \mathcal { W } _ { s } ) \le | \mathcal { W } _ { s } | \exp \bigg ( - \frac { 2 \eta ^ { 2 } p ^ { 2 } ( \sum _ { k \in [ L ] } \gamma _ { k } ) ^ { 2 } } { \sum _ { k \in [ L ] } \gamma _ { k } ^ { 2 } } \bigg ) + e ^ { - \delta ^ { 2 } ( \mu + \Delta _ { 0 } ) / 2 } .
$$

Similarly,

$$
\mathbb { P } ( E _ { m } ( x ) = \mathrm { \mathfrak { s e e n } } ^ { \prime } \mid w \in \mathcal { W } _ { s } ) \le | \mathcal { W } _ { s } | \exp \bigg ( - \frac { 2 \eta ^ { 2 } p ^ { 2 } ( \sum _ { k \in [ L ] } \gamma _ { k } ) ^ { 2 } } { \sum _ { k \in [ L ] } \gamma _ { k } ^ { 2 } } \bigg ) + e ^ { - \delta ^ { 2 } \mu / ( 2 + \delta ) } .
$$

This completes the proof.

## B.3 A Result under Exchangeability of the Predictors $\psi _ { k }$

One issue with the previous analysis is that we have been assuming independence of the predictors $\psi _ { k }$ (for instance, by sample splitting, which comes at the cost of some statistical eficiency). In practice, the diferent predictors in the ensemble may be assigned common training samples, rendering them not independent of each other. It turns out that we can prove similar results under the weaker assumption of exchangeability.

Theorem 3. Fix $x , w ( x )$ . Suppose $( \alpha _ { 1 } , \beta _ { 1 } ) , \dots ( \alpha _ { L } , \beta _ { L } )$ are exchangeable random variables with common mean $( \alpha , \beta )$

Then, for $m = L \left( \beta + \textstyle { \frac { p } { 2 } } ( 1 - \alpha - \beta ) \right)$ , the ensemble classifier $E _ { m }$ is such that

$$
\mathbb { P } ( E _ { m } ( x ) = { \mathopen { : } } u n s e e n ^ { \prime } | ~ w \in \mathcal { W } _ { s } ) \leq \exp \left( - \frac { p ^ { 2 } L ( 1 - \alpha - \beta ) ^ { 2 } ( L - H _ { L } ) } { 2 ( L - 1 ) } \right)
$$

$$
\mathbb { P } ( E _ { m } ( x ) = { \mathit { \iota } } _ { s e e n } ^ { } | ~ w \in \mathcal { W } _ { u } ) \leq \exp \left( - \frac { p ^ { 2 } L ( 1 - \alpha - \beta ) ^ { 2 } ( L - H _ { L } ) } { 2 ( L - 1 ) } \right)
$$

Here, $( \alpha _ { 1 } , \beta _ { 1 } ) , \dots ( \alpha _ { L } , \beta _ { L } )$ being exchangeable means that for any permutation σ of $[ L ]$ , we have

$$
\begin{array} { r } { ( \alpha _ { 1 } , \ldots , \alpha _ { L } , \beta _ { 1 } , \ldots , \beta _ { L } ) \stackrel { d } { = } ( \alpha _ { \sigma ( 1 ) } , \ldots , \alpha _ { \sigma ( L ) } , \beta _ { \sigma ( 1 ) } , \ldots , \beta _ { \sigma ( L ) } ) . } \end{array}
$$

Proof. As before, let

$$
X = \sum _ { k \in [ L ] } \mathbb { 1 } ( \psi _ { k } ( x ) = { \mathrm { ' s e e n ' } } ) .
$$

We analyze this variable in the cases $w \in \mathcal { W } _ { s }$ and $w \in \mathcal { W } _ { u }$

From [3, Theorem 4], let $v \in \mathbb { R } ^ { L }$ and $X _ { 1 } , \dots , X _ { L } \in [ - 1 , 1 ]$ be exchangeable, where $L \geq 2$ . Then,

$$
\mathbb { P } \left\{ \sum _ { i = 1 } ^ { L } v _ { i } ( X _ { i } - \mathbb { E } [ X _ { i } ] ) \geq \| v \| _ { 2 } { \sqrt { 2 \left( { \frac { L - 1 } { L - H _ { L } } } \right) \log ( 1 / \delta ) } } \right\} \leq \delta
$$

where $\begin{array} { r } { H _ { L } = \sum _ { i = 1 } ^ { L } \frac { 1 } { i } \sim O ( \log L ) } \end{array}$

We apply this in the case where $X _ { i }$ is a Rademacher random variable as follows:

$$
X _ { i } = { \left\{ \begin{array} { l l } { 1 } & { \psi _ { k } { \mathrm { ~ c a l l s ~ } } x { \mathrm { ~ s e e n } } } \\ { - 1 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

so we get:

$$
\mathbb { E } [ X _ { i } ] = \left\{ { \begin{array} { l l } { 2 p ( 1 - \alpha ) + 2 ( 1 - p ) \beta - 1 } & { w \in \mathscr { W } _ { s } } \\ { 2 \beta - 1 } & { w \in \mathscr { W } _ { u } } \end{array} } \right.
$$

$$
X - \mathbb { E } [ X ] = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { L } X _ { i } - \mathbb { E } [ X _ { i } ] .
$$

Setting $v = ( 1 / 2 , \dots , 1 / 2 )$ , we obtain

$$
\mathbb { P } \left\{ X - \mathbb { E } [ X ] \geq \frac { 1 } { 2 } \sqrt { 2 L \left( \frac { L - 1 } { L - H _ { L } } \right) \log ( 1 / \delta ) } \right\} \leq \delta .
$$

Setting $v = ( - 1 / 2 , \dots , - 1 / 2 )$ , we obtain

$$
\mathbb { P } \left\{ X - \mathbb { E } [ X ] \leq - \frac { 1 } { 2 } \sqrt { 2 L \left( \frac { L - 1 } { L - H _ { L } } \right) \log ( 1 / \delta ) } \right\} \leq \delta .
$$

Hence, choosing our threshold $m = L \left( \beta + \textstyle { \frac { p } { 2 } } ( 1 - \alpha - \beta ) \right)$ , we can get

$$
\frac { 1 } { 2 } \sqrt { 2 L \left( \frac { L - 1 } { L - H _ { L } } \right) \log ( 1 / \delta ) } = \frac { p L } { 2 } ( 1 - \alpha - \beta ) ,
$$

i.e.

$$
\delta = \exp { \left( - \frac { p ^ { 2 } L ( 1 - \alpha - \beta ) ^ { 2 } ( L - H _ { L } ) } { 2 ( L - 1 ) } \right) }
$$

which is our false positive rate/false negative rate. We notice $\delta \sim \exp ( - c L )$ for some constant c. □

We can see that the prediction errors shrink exponentially in $L ,$ making the ensemble predictor a viable approach.