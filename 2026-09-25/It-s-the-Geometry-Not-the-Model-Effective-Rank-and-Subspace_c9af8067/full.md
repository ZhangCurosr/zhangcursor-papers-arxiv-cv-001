# It’s the Geometry, Not the Model: Effective Rank and Subspace Alignment in Functional Connectivity Classification

Xiao Fan, Jingyuan Li, Yubo Han, Hongbin Guo, Guanya Li, Yang Hu,

Wenchao Zhang, Weibin Ji, Yi Zhang

xiufan@stu.xidian.edu.cn

Abstract—Resting-state functional connectivity (FC) is a widely used representation for classifying brain phenotypes and disorders, and most FC classification pipelines feed the full connectome into a model and place the burden of progress on the model. We instead argue that the geometry of FC itself is the dominant factor. Because FC is computed from wholeseries correlations that are highly reliable within individuals and vary along relatively few directions across them, its across-subject variation is effectively low-dimensional, concentrating in a small leading effective subspace. Within a cohort, this implies that most of the nominal FC dimensions are redundant and that the discriminative structure available in the data is confined to a few directions. More importantly, across cohorts the orientation of this subspace need not agree, and such misalignment can drive cross-site transfer to fail even when the subspaces are of comparable size. Across 2,330 subjects from HCP, ABIDE, and ADHD-200, effective-rank analysis shows strong spectral concentration, and projecting onto only the leading effectiverank-scale components recovers most of the full-FC classification performance. On ABIDE, site-specific effective subspaces are only weakly aligned, and their principal-angle overlap predicts pairwise transfer after covariate adjustment, even though per-site effective ranks are comparable. Crucially, controlled rotations that perturb subspace orientation while preserving the mean and covariance spectrum drive transfer toward chance, whereas displacement-matched label-orthogonal rotations do not, isolating orientation as the factor that degrades transfer. Rather than proposing a new architecture, this study offers a diagnostic view of FC generalization, suggesting that cross-site evaluation and harmonization be judged by whether they align effective subspaces rather than only improve within-dataset accuracy.

Index Terms—functional connectivity, effective rank, representation geometry, subspace alignment, cross-site generalization

## I. INTRODUCTION

Functional connectivity (FC) derived from resting-state fMRI is widely used to characterize brain functional organization and to classify brain phenotypes and disorders. Recent FC classification research has increasingly treated the highdimensional connectome as a modeling problem, leading to graph convolutional networks (GCNs), connectome-specific convolutional neural networks (CNNs) [1], and Transformerbased models [2]. This line of work rests on a common assumption that FC contains discriminative structure which more expressive architectures can extract more fully, so progress is expected to come primarily from architecture design.

However, this model-centered framing does not explain two recurring observations in FC classification. First, simple multilayer perceptrons (MLPs), and even linear models, often match or exceed more elaborate graph-based architectures on standard FC classification tasks [3]. Second, FC classifiers remain fragile under cross-site transfer, a limitation that more sophisticated architectures do not resolve [4]. Together these observations raise a common representation-level question: how can the same FC geometry make decoding easy within a dataset yet fragile across sites?

Prior work has examined these observations only in isolation. Some studies show that functional-connectivity variation often has low-dimensional structure [5], [6], and learningtheoretic results suggest that spectrally concentrated covariance can favor simple predictors [7], [8]. Yet this literature treats strong within-dataset decoding mainly as a modelcomparison issue and cross-site failure mainly as a domainshift issue, leaving the two conceptually separate. We instead argue that both stem from a single representational property, the leading effective subspace of FC, that is, the small set of covariance directions that captures most across-subject FC variation.

This property also distinguishes FC from other neuroimaging inputs and clarifies why the representation, rather than the architecture, deserves attention. Standard medical images are raw high-dimensional signals for which expressive architectures are genuinely required to extract structure, and the architecture-centered framing above is largely inherited from that setting. FC, by contrast, is computed from whole-series correlations that are highly reliable within individuals and vary along relatively few directions across them [9], so its acrosssubject variation concentrates in far fewer directions than its nominal edge count.

Within a dataset, this leading effective subspace contains most of the discriminative FC signal, so most of the nominal FC dimensions are redundant and the discriminative structure is confined to a few directions. More importantly, across sites the same concentration becomes a transfer bottleneck. Effective rank sets the scale of the subspace, but transfer depends on whether site-specific effective subspaces are similarly oriented. When orientations diverge, a classifier trained on one site relies on directions that the other barely carries.

We evaluate this representation-geometry account on three public resting-state fMRI datasets, HCP, ABIDE autism, and ADHD-200, together spanning 2,330 subjects. Our claim is representation-centered. FC should not be treated merely as high-dimensional input, but as a low-effective-rank representation whose leading effective subspace shapes both withindataset decoding and cross-site transfer. Our contributions are:

• We reframe FC classification as an effective-subspace problem. This view links two observations usually treated separately, strong within-dataset decoding by simple models and fragile cross-site transfer.

• We show that within-dataset FC decoding is largely supported by the leading effective subspace. Effectiverank measurements and top-PC decoding localize most of the label-relevant FC signal to this subspace, where simple classical and shallow classifiers match or exceed more elaborate baselines.

• We show that the same effective subspace constrains cross-site transfer through its orientation. Across ABIDE sites, principal-angle alignment predicts pairwise transfer after covariate adjustment, and controlled rotations of that subspace, at matched perturbation magnitude, isolate orientation as the factor that drives transfer down.

## II. ANALYSIS FRAMEWORK AND EXPERIMENTAL PROTOCOL

We organize the study around a representation-geometry question: why can FC be readily decoded within a dataset while remaining fragile under cross-site transfer? Our hypothesis is that both behaviors stem from the same effectivesubspace geometry, which we make testable through the two quantities below.

We operationalize this account with two label-free geometric quantities (Fig. 1). Effective rank measures the scale of the subject-level FC covariance, indicating how many covariance directions are effectively used. Principal-angle overlap measures the alignment between site-specific effective subspaces, indicating whether different sites represent FC variation along similar directions. Together, these quantities allow us to test two linked predictions of the same geometry: if discriminative FC signal is concentrated in the leading effective subspace, top principal component (PC) decoding should recover most full-FC performance and simple classifiers should be strong competitors; if cross-site transfer depends on subspace orientation, principal-angle overlap should predict transfer and controlled rotations of the leading subspace should degrade it. Robustness checks and raw blood-oxygen-level-dependent (BOLD) time series controls further rule out memorization, high dimensionality alone, relatedness, and label easiness as simpler explanations.

## A. Datasets and FC Feature Construction

We use three public resting-state fMRI datasets covering healthy and clinical phenotypes. For HCP sex classification, we retain 1,002 subjects from the S1200 release with available sex labels and use the Brainnetome atlas with $V ~ = ~ 2 4 6$ regions [10], [11]. In all tables and figures, HCP denotes this HCP sex-classification task. For ABIDE autism classification, we use 787 subjects from 22 sites. This full ABIDE cohort is used for within-dataset classification and effective-rank analyses. For cross-site subspace alignment, pairwise transfer, and leave-one-site-out (LOSO) analyses, we exclude the six sites with fewer than 20 subjects, leaving 16 sites with $n _ { s } \geq 2 0$ for site-wise subspace estimation. The controlled orientationrotation intervention (Section II-E) additionally requires persite split-half estimation and is therefore run on a further subset, the six largest sites $( n _ { s } \geq 4 2 )$ . These site sets are nested (22 sites for within-dataset $\mathrm { a n a l y s e s } \ \supseteq \ 1 6$ sites for per-site cross-site analyses $\supseteq 6$ sites for the rotation intervention), with each threshold set by the per-site sample the corresponding analysis needs for stable estimation. For ADHD-200 classification, we use only the first session and retain subjects with valid AAL116 ROI time series, at least 80 time points, and a site label in the phenotypic table. We exclude sites with shorter time series, including Pittsburgh, WashU, and OHSU, leaving 541 subjects from 4 sites. ABIDE and ADHD-200 are both represented with the AAL atlas with V = 116 regions [12]– [14].

![](images/7a2850dc90a746e8d14fc893f96186b7fd644f69bf74fd50fb2a2d9573f51270.jpg)  
Fig. 1. The effective-rank and subspace-alignment analysis framework. Vectorized FC features are nominally high-dimensional but may occupy a much smaller leading effective subspace. The within-dataset branch tests whether this subspace contains most discriminative FC signal and explains the competitiveness of simple classifiers. The cross-site branch tests whether site-specific effective subspaces are similarly oriented, using principal-angle overlap to quantify subspace alignment and relate it to transferability.

For each subject $i ,$ we compute the Pearson correlation matrix across ROI BOLD time series, apply Fisher’s ztransform, and vectorize the upper-triangular entries excluding the diagonal to obtain $\mathbf { x } _ { i } \in \bar { \mathbb { R } } ^ { D }$ , where $D = V ( V - 1 ) / 2$ Stacking all subjects gives $X = [ \mathbf { x } _ { 1 } ^ { \top } , \ldots , \mathbf { x } _ { N } ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { N \times D }$

## B. Effective Rank of Subject-Level FC Covariance

Effective rank quantifies the effective dimensionality of the subject-level FC covariance, which sets the scale of the leading effective subspace. Given the FC feature matrix X, we center it across subjects to obtain $\widetilde { X }$ . Let $\lambda _ { i }$ denote the nonzero eigenvalues of $\widetilde { X } ^ { \top } \widetilde { X } / ( N - 1 )$ . We compute the Shannon effective rank as

$$
\operatorname { E R } ( X ) = \exp \left( - \sum _ { i } p _ { i } \log p _ { i } \right) , \qquad p _ { i } = { \frac { \lambda _ { i } } { \sum _ { j } \lambda _ { j } } } .\tag{1}
$$

We adopt the Shannon (entropy-based) effective rank rather than a hard-threshold or numerical rank because it is a smooth, threshold-free functional of the eigenvalue spectrum that responds directly to how concentrated the spectrum ${ \mathrm { i s } } ,$ which makes the resulting scale comparable across datasets and sites. A smaller ER indicates that across-subject FC variation is concentrated in fewer covariance directions, whereas a larger ER indicates that variation is spread across more directions.

## C. Within-Dataset Decodability and Representation Controls

We evaluate whether the leading effective subspace contains the discriminative FC signal that supports within-dataset decoding. We first perform top-k principal-component decoding. Principal component analysis (PCA) is fitted on the training fold only, FC features are projected onto the leading k components, and L2-regularized logistic regression (L2-LogReg) is evaluated on the held-out fold. We vary k from small values to the ER and 2ER scales, using rounded integer values for non-integer ER. When reporting recovery relative to full $\mathrm { F C } ,$ we use chance-normalized AUROC recovery,

$$
{ \frac { \mathrm { A U R O C } _ { k } - 0 . 5 } { \mathrm { A U R O C } _ { \mathrm { f u l l } } - 0 . 5 } } .
$$

We also include a mid-band control using components between ER and 2ER to test whether performance is specific to the leading covariance directions rather than to dimensionality reduction alone. The upper bound is set to 2ER so that this control spans a band of the same width as the leading effective subspace, giving a width-matched comparison between the leading ER directions and the next ER directions; it is chosen for this matched-control purpose rather than tuned.

We then compare classical baselines and representative neural architectures under shared preprocessing, splits, and metrics. The comparison covers classical linear and kernel models, shallow MLPs, graph-based neural networks, connectomespecific CNN-style architectures, and Transformer-based models. This design tests whether more elaborate architectures provide consistent gains over simple FC decoders when the leading effective subspace already contains most discriminative signal. Unless otherwise stated, evaluations use three random seeds and three stratified subject-level folds, and results are reported as mean±std over matched seed-fold runs. Hyperparameters for tunable models are selected by validation AUROC within each training fold.

We add three safeguards against simpler explanations. Label shuffling tests whether the pipeline can memorize arbitrary labels, random Gaussian features test whether high nominal dimensionality alone is sufficient, and family- versus subjectlevel CV on HCP tests whether relatedness drives performance.

Finally, we compare FC with raw BOLD time series from the same fMRI source. We evaluate flattened BOLD with ridge classification and structured BOLD with a compact temporal CNN under matched folds. This representation control asks whether simple FC classifiers succeed because the labels are intrinsically easy, or because FC makes much of the discriminative signal accessible to simple decoders.

## D. Cross-Site Subspace Alignment

Cross-site transfer requires more than within-site concentration of discriminative signal. If site-specific effective subspaces are poorly aligned, a classifier trained on one site may rely on directions that are weakly represented, or absent, in another. We test this idea on the 16-site ABIDE subset by estimating a site-specific effective subspace for each site.

For each site $s ,$ let $X _ { s }$ denote its centered FC feature matrix. We obtain the top-k principal components by singular value decomposition, with sensitivity evaluated over $k \in \{ 5 , 1 0 , 2 0 \}$ For a pair of sites $( s , t )$ , let $U _ { s } , U _ { t } ~ \in ~ \mathbb { R } ^ { D \times k }$ denote the corresponding orthonormal subspace bases. We define their principal-angle overlap as

$$
\operatorname { o v e r l a p } ( s , t ) = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \cos ^ { 2 } ( \theta _ { i } ) = \frac { 1 } { k } \left\| U _ { s } ^ { \top } U _ { t } \right\| _ { F } ^ { 2 } ,\tag{2}
$$

where $\theta _ { i }$ are the principal angles obtained from the singular values of $U _ { s } ^ { \top } U _ { t }$ . Larger overlap means the two sites represent FC variation along similar directions, whereas smaller overlap means these directions diverge, the geometric condition under which a source-site classifier leans on directions weakly represented at the target site and transfer is expected to degrade.

Pairwise transfer is measured by training L2-LogReg on site s and testing on site t for all ordered site pairs $( s , t ) , s \neq t .$ This gives a directional transfer matrix, whereas principalangle overlap is symmetric. We therefore compute the Pearson correlation between overlap and transfer over ordered nondiagonal site pairs, and assess significance using a Mantelstyle site-label permutation test with 5,000 permutations. In each permutation, site labels of the overlap matrix are jointly permuted along rows and columns, preserving the dyadic dependence structure while testing whether subspace alignment predicts directional transfer. To ensure that the overlap– transfer association is not driven by site-level confounds, we residualize both overlap and transfer against five sitepair covariates using multiple regression quadratic assignment (MRQAP). For each covariate we form a symmetric sitepair distance matrix: absolute differences in sample size, class balance (patient fraction), mean age, and sex ratio, together with the site-mean FC distance

$$
\left\| \bar { \mathbf { x } } _ { s } - \bar { \mathbf { x } } _ { t } \right\| _ { 2 } ,
$$

where $\bar { \mathbf { x } } _ { s }$ is the mean FC vector of site s. Overlap and transfer are regressed on these five covariate matrices, and the partial association between the overlap and transfer residuals is assessed by MRQAP with row–column permutations (5,000 permutations), which preserves the dyadic dependence structure. Motion and mean framewise displacement were not included because they were unavailable in the ABIDE phenotypic table used here.

## E. Rank Control and Orientation Intervention

To separate subspace orientation from rank magnitude, we control for per-site effective rank in the dyadic analysis and repeat the comparison after matching site sample size. We then perform a controlled orientation-rotation intervention. For a target site $t ,$ let $\smash { \widetilde { X } } _ { t }$ denote its centered FC feature matrix and let $Q _ { \theta } ~ \in ~ \mathbb { R } ^ { D \times D }$ be an orthogonal transformation that rotates selected FC directions by angle θ. The rotated target representation is

$$
X _ { t } ^ { ( \theta ) } = \mathbf { 1 } \bar { \mathbf { x } } _ { t } ^ { \top } + \widetilde { X } _ { t } Q _ { \theta } , \qquad Q _ { \theta } ^ { \top } Q _ { \theta } = I .\tag{3}
$$

A classifier trained on the unrotated source site is then evaluated on $X _ { t } ^ { ( \theta ) }$

Because $Q _ { \theta }$ is orthogonal and the original site mean $\bar { \mathbf { x } } _ { t }$ is restored, the intervention preserves the feature mean and the covariance eigenvalue spectrum:

$$
\operatorname { c o v } \left( X _ { t } ^ { ( \theta ) } \right) = Q _ { \theta } ^ { \top } \operatorname { c o v } ( X _ { t } ) Q _ { \theta } .\tag{4}
$$

Thus, the intervention perturbs orientation without changing lower-order structure. We rotate the leading effective subspace as the primary test and use displacement-matched discriminative and label-orthogonal rotations to distinguish label-relevant orientation effects from perturbation magnitude alone. Because this within-site design splits each site in half (one half trains the source classifier, the other is rotated and used as the target), it needs roughly twice the subjects of the whole-site analyses, so the rotation intervention is run on the six largest ABIDE sites $( n _ { s } ~ \ge ~ 4 2 )$ , using 10 split-halves per site and seven rotation angles from $0 ^ { \circ }$ to 90<sup>◦</sup>. Effects are aggregated at the level of independent sites, giving a per-site paired comparison $( n = 6 )$ between the discriminative and label-orthogonal arms.

a) Evaluation protocols.: Within-dataset analyses, including top-PC decoding, classifier comparisons, representation controls, and within-dataset robustness checks, use three stratified subject-level folds and three random seeds (seeds 1– 3), yielding $3 \times 3$ seed–fold evaluations. Results are reported as mean±std over these evaluations unless otherwise stated. Cross-site analyses use site-level evaluation on the 16-site ABIDE subset. Leave-one-site-out (LOSO) trains on all but one site and tests on the held-out site, whereas pairwise transfer trains on one site s and tests on another site t for all ordered pairs $( s , t ) , s \neq t .$

## III. RESULTS

## A. Low Effective Rank of FC Features

We begin by establishing the structural property that underpins the within-dataset analysis. Effective-rank measurements across datasets are summarized in Table I. Although the FC feature vectors are nominally high-dimensional, their subjectlevel covariance is strongly concentrated: $\mathrm { E R } / D < 3 \%$ for all three datasets. This means that across-subject variation in FC features is distributed over only a small number of effective covariance directions relative to the nominal feature dimension. The ratio ER/N ranges from 6% to 32%, indicating that the effective dimensionality is also small relative to the available sample size. We therefore use ER/D and ER/N as empirical diagnostics of spectral concentration in each cohort.

TABLE I  
EFFECTIVE RANK OF VECTORIZED FC FEATURES. D IS THE NOMINAL FEATURE DIMENSION, ER THE SHANNON EFFECTIVE RANK OF THE SUBJECT-LEVEL FC COVARIANCE (EQ. 1), AND ER/D, ER/N ITS RATIOS TO THE FEATURE DIMENSION AND THE SAMPLE SIZE. ER/D < 3% ON ALL THREE DATASETS INDICATES STRONG SPECTRAL CONCENTRATION.
<table><tr><td>Dataset</td><td> $D$ </td><td>ER</td><td>ER/D (%)</td><td>ER/N (%)</td></tr><tr><td>HCP</td><td>30,135</td><td>62.8</td><td>0.21</td><td>6.3</td></tr><tr><td>ABIDE</td><td>6,670</td><td>108.5</td><td>1.63</td><td>13.8</td></tr><tr><td>ADHD-200</td><td>6,670</td><td>174.2</td><td>2.61</td><td>32.2</td></tr></table>

## B. Top-PC Decoding in the Leading Effective Subspace

We next test whether the low-effective-rank structure of FC is directly relevant to supervised classification. After fitting PCA within each training fold, we project FC features onto the top-k principal components and evaluate L2-LogReg on the held-out fold (Fig. 2). The leading effective subspace recovers most of the full-FC classification performance. ${ \mathrm { A t ~ } } k = { \mathrm { E R } }$ chance-normalized AUROC recovery is 88% on HCP, 103% on ABIDE, and 87% on ADHD-200, so a subspace at the ER scale already reproduces most of the discriminative signal on all three datasets.

The three datasets differ in how the recovery curve behaves around this scale, and these differences follow from a single distinction. ER is computed from the unlabeled covariance spectrum and measures how widely across-subject variation is spread, which need not match how widely the labelrelevant signal is spread. The two can therefore align, fall short, or slightly exceed one another. ABIDE is the aligned case: recovery reaches the full-FC level right at the ER scale (103%), so the discriminative signal occupies about as many directions as the covariance does. ADHD-200 is the concentrated case: it has the largest ER (Table I) yet the weakest discriminative signal (Table II), because its acrosssubject variation is spread widely while the label-relevant part is confined to a few leading components. Recovery therefore peaks at a very small k and then declines, since the remaining high-variance directions are largely label-irrelevant and adding them only injects noise. HCP is the mildly extended case: recovery is already 88% at ER but rises to 94% at 2ER, indicating that a small amount of label-relevant signal extends just beyond the ER scale while the bulk lies within it. This is consistent with our claim that the leading effective subspace carries most, not all, of the label-relevant signal.

![](images/68fd0165ed3c097105f87ee417f2631f397715ad8e191bafc572163994c2e6ed.jpg)

![](images/62aeff7e4a10ba658e324a7ab608d94a490ef39ae33555a6ddbbd673881812fa.jpg)

![](images/367bc5dc6a520ceaaee517c75526999affaa8e872ea87108093baf715a68cfe5.jpg)  
Fig. 2. (A) HCP (B) ABIDE (C) ADHD-200. Top-k PC classification of FC features. PCA is fitted within each training fold, and L2-LogReg is evaluated on held-out subjects after projecting FC features onto the leading k principal components. Dashed horizontal lines show full-FC classification performance, and vertical dotted lines mark the rounded ER and 2ER scales. The ER-scale leading effective subspace recovers most full-FC performance. ADHD-200 reaches its peak before ER, suggesting that its label-relevant signal is more concentrated than the covariance-defined ER scale.

The non-leading PC control confirms that this is a property of the leading directions rather than of dimensionality reduction alone. Components between ER and 2ER yield AUROC values of 0.669, 0.545, and 0.508 on HCP, ABIDE, and ADHD-200, respectively, well below the leading-subspace performance. The label-relevant FC signal thus resides in the leading effective subspace, directly linking low-effective-rank geometry to the competitiveness of simple FC classifiers.

## C. Raw-BOLD Representation Control

Fig. 3 tests whether the competitiveness of simple FC classifiers is specific to the FC representation rather than to intrinsically easy labels. Raw BOLD and FC differ in a way that bears directly on their geometry. Raw BOLD is a time-resolved signal that fluctuates from frame to frame, so its discriminative structure is temporal and distributed rather than concentrated in a few static directions. FC, by contrast, summarizes each subject by whole-series correlations that are stable within individuals, which is why its across-subject variation collapses onto a small leading effective subspace. Consistent with this, using the same fMRI source, flattened raw BOLD with ridge classification remains weak, whereas a compact temporal CNN recovers signal on HCP and ABIDE. This contrast shows that discriminative information is present in the underlying fMRI data but is not equally accessible across representations, and that the low-dimensional structure is a property of how FC is constructed rather than of the parcellation or the labels. FC makes much of this information accessible to simple decoders, whereas raw BOLD requires temporal modeling, which is why analyzing the FC representation itself, rather than importing architectural complexity, is the appropriate lens here.

## D. Classifier Comparison under Low Effective Rank

The effective-subspace result suggests that much of the FC decoding signal is already accessible to simple decoders.

![](images/73fa95774634992c8a811aed6a05846b7a2e2fb00e187d7a95b3f134444887de.jpg)

![](images/69c9b072302b3e774cc0d338eae701caddd17f41c97158b6ce0c60983d30b8d3.jpg)  
Fig. 3. Representation choice shapes decoder complexity. (A) Effective-rankto-sample-size ratio for FC and raw BOLD representations. (B) Test AUROC of FC with L2-LogReg, flattened raw BOLD with ridge classification, and raw BOLD with a temporal CNN. Significance marks compare temporal CNN against flattened ridge over the 3 × 3 seed–fold runs.

Table II evaluates this expectation using representative linear, kernel, MLP, connectome-specific, and Transformer-based models under shared splits and metrics.

Across HCP, ABIDE, and ADHD-200, simple FC decoders are consistently competitive. L2-LogReg achieves the best mean AUROC on HCP and ABIDE, while RBF SVM gives the best mean AUROC on ADHD-200. MLPs remain close to the best-performing models on HCP and ABIDE, but they are not uniquely superior. Representative graph, connectomespecific, and Transformer-based models do not provide consistent gains over these simpler baselines.

These results show that, for FC representations whose leading effective subspace already supports decoding, increasing architectural complexity is not automatically beneficial.

TABLE II  
ARCHITECTURE COMPARISON. TEST AUROC IS REPORTED AS MEAN±STD OVER 3 × 3 SEED–FOLD RUNS. BOLDFACE MARKS THE BEST MEAN PER DATASET. L2-LOGREG DENOTES L2-REGULARIZED LOGISTIC REGRESSION.
<table><tr><td>Family</td><td>Model</td><td>HCP</td><td>ABIDE</td><td>ADHD-200</td></tr><tr><td>Classical</td><td>L2-LogReg</td><td>0.914±.002</td><td>0.724±.006</td><td>0.587±.009</td></tr><tr><td>Classical</td><td>RBF SVM</td><td>0.834±.009</td><td>0.670±.017</td><td>0.634±.006</td></tr><tr><td>Neural</td><td> $\mathrm { M L P } \ ( h = 6 4 )$ </td><td>0.896±.007</td><td>0.707±.011</td><td>0.594±.006</td></tr><tr><td>Neural</td><td>MLP (h = 1024)</td><td>0.894±.005</td><td>0.710±.010</td><td>0.582±.014</td></tr><tr><td>Graph</td><td>GCN</td><td>0.802±.008</td><td>0.632±.013</td><td>0.569±.016</td></tr><tr><td>Graph</td><td>GAT-style</td><td>0.807±.018</td><td>0.663±.006</td><td>0.574±.017</td></tr><tr><td>Connectome</td><td>BrainNetCNN [1]</td><td>0.846±.035</td><td>0.686±.008</td><td>0.596±.016</td></tr><tr><td>Transformer</td><td>BNT [2]</td><td></td><td>0.881±.006 0.676±.010</td><td>0.585±.018</td></tr></table>

## E. Robustness and Leakage Checks

The leading-effective-subspace analysis establishes the main link between low-effective-rank geometry and FC classification performance. The remaining within-dataset checks rule out simpler artifacts (Table III). Label shuffling reduces performance to chance, random Gaussian features with the same nominal dimensionality fail to reproduce true-label performance, and family-level cross-validation on HCP changes performance only marginally. Although the ADHD-200 Gaussian control is slightly above chance (0.545), it remains below the true-label reference (0.587). Together, these checks indicate that the observed FC performance is not explained by memorization, nominal dimensionality, or family relatedness.

TABLE III  
CONTROL EXPERIMENTS AND LEAKAGE CHECKS. REPORTED VALUES ARE MEAN TEST AUROC. ALL CONTROLS ARE WITHIN-DATASET.
<table><tr><td>Control</td><td>HCP</td><td>ABIDE</td><td>ADHD-200</td></tr><tr><td>True labels (reference)</td><td>0.914</td><td>0.724</td><td>0.587</td></tr><tr><td>Label shuffle</td><td>0.514</td><td>0.491</td><td>0.495</td></tr><tr><td>Random Gaussian features</td><td>0.531</td><td>0.535</td><td>0.545</td></tr><tr><td>Subject-level CV</td><td>0.914</td><td>一</td><td>一</td></tr><tr><td>Family-level CV</td><td>0.908</td><td>一</td><td>一</td></tr></table>

## F. Cross-Site Subspace Alignment

We next examine whether cross-site transfer follows the alignment of site-specific effective subspaces. As a preliminary evaluation check, ABIDE leave-one-site-out (LOSO) crossvalidation yields lower AUROC than random 3-fold crossvalidation $( 0 . 7 0 9 \pm 0 . 0 8 5 ~ \mathrm { v s . } ~ 0 . 7 2 4 )$ , while remaining above chance. This shows that site structure affects evaluation, but that the classification signal is still detectable. We then quantify principal-angle overlap between the 16 ABIDE sitespecific effective subspaces (Fig. 4A). At $k = 1 0 ,$ the mean overlap is 0.195, indicating substantial orientation differences across sites within the same ABIDE classification task. Higher overlap is associated with higher ordered pairwise transfer AUROC (Fig. 4B): Pearson $r ~ = ~ 0 . 4 1 8$ with Mantel-style site-label permutation, $p = 0 . 0 0 2$ , with stable results across $k \in \{ 5 , 1 0 , 2 0 \}$ . The association remains after residualizing both overlap and transfer against site-pair differences in sample size, class balance, mean age, sex ratio, and site-mean FC distance (partial $\beta = 0 . 3 9 4$ , MRQAP $p = 0 . 0 4 4 ;$ Fig. 4C). Thus, site-specific effective subspaces that are more similarly oriented support better cross-site transfer.

We finally examine the role of effective-rank magnitude. Adding per-site effective rank to the dyadic regression preserves the overlap association $( \beta ~ = ~ 0 . 5 0 0 , ~ p ~ = ~ 0 . 0 3 2 )$ After subsampling every site to $n \ = \ 2 0 ,$ per-site effective ranks concentrate around ≈ 13 with low cross-site variation $\mathrm { ( S D ~ \approx ~ 1 . 2 ) }$ . These controls show that, after sample-size matching, effective-rank magnitude is nearly uniform across sites, whereas subspace orientation remains informative for transfer.

## G. Controlled Subspace-Orientation Rotation

We next test whether subspace misalignment can directly degrade transfer, rather than merely correlate with it. We rotate selected FC directions in the target site while preserving the feature mean and covariance eigenvalue spectrum, and then evaluate a source-site classifier on the rotated target. This gives a controlled dose–response test of orientation mismatch.

Rotating the leading effective subspace degrades transfer monotonically toward chance (Fig. 5A). As the rotation drives source–target overlap down across its realistic range, mean transfer AUROC falls from $\approx 0 . 6 3$ at $\theta = 0 \mathrm { ~ t o ~ } \approx 0 . 5 3$ at $\theta = 9 0 ^ { \circ }$ (leading-subspace arm, $\Delta \approx 0 . 0 9 6 ) ;$ the swept band covers the empirical ABIDE overlap (≈ 0.195 at $k = 1 0 )$ Because the rotation is orthogonal and the site mean is restored, it leaves the feature mean (change $\sim 7 \times 1 0 ^ { - 1 6 } )$ and the covariance eigenvalue spectrum (change $\sim 2 \times 1 0 ^ { - 1 3 } )$ intact, so this decline reflects a change in orientation alone rather than any shift in mean or variance.

Matched-displacement controls show the effect is labelrelevant rather than a generic consequence of perturbing the data (Fig. 5B). At equal displacement $( \| \Delta \mathbf { x } \| = 6 . 8 2 )$ , rotating the discriminative direction reduces transfer by $\Delta \approx 0 . 1 0$ $( \mathrm { A U R O C } ~ \approx ~ 0 . 6 3 ~  ~ 0 . 5 3 )$ , whereas rotating a labelorthogonal direction leaves it essentially flat $( \Delta \approx 0 . 0 0 3 )$ Evaluated at the level of independent sites, the discriminative rotation lowers transfer at 5 of the 6 sites, against 0 of 6 for the label-orthogonal control (per-site paired ttest $\begin{array} { r l r } { p } & { { } = } & { 0 . 0 3 4 } \end{array}$ ; Wilcoxon $\begin{array} { r c l } { p } & { = } & { 0 . 0 6 3 ) } \end{array}$ . With only six sites this is a controlled demonstration rather than a highpowered test; the population-level cross-site evidence remains the 16-site alignment–transfer association of Section III-F. Taken together, the active–placebo dissociation under a meanand spectrum-preserving rotation indicates that reorienting the leading effective subspace ${ \mathrm { i s } } ,$ on its own, sufficient to degrade cross-site transfer, with perturbation magnitude held fixed.

## IV. DISCUSSION

## A. A Geometric Account of Two Behaviors

The contribution of this study is not that FC has low effective rank, which is already known, but that a single geometric property, the leading effective subspace, accounts for two behaviors usually explained separately. Strong withindataset decoding by simple models and fragile cross-site transfer are often treated as a model-comparison issue and a domain-shift issue, respectively. Our results tie both to the same subspace. Within a site it concentrates the discriminative signal, so simple decoders suffice and added architectural capacity has little structure left to exploit. Across sites the same concentration makes transfer depend on whether that subspace is similarly oriented, which the rotation intervention shows is sufficient, on its own, to drive transfer toward chance. Viewed this way, the two behaviors are not separate phenomena but two consequences of the same low-dimensional geometry.

## B. Implications for Future FC Studies

This account is diagnostic rather than architectural, and it suggests concrete changes for how FC classification should be studied. First, model comparisons should be calibrated against the representation. When ER-scale components already recover most full-FC performance, a strong simple baseline is the expected consequence of the representation, not a weak control, and an architecture should be credited only for signal it captures beyond the leading effective subspace. Otherwise, a comparison between a complex model and a simple classifier mainly reflects how much discriminative signal the FC representation has already exposed.

A  
![](images/b547d628c30a38fd54d05aa12484186ad5319490232e7e1004eb1ce6f41996e4.jpg)

![](images/feed513196476953b96aa36b4e3cd5f9b7a624713690afd1d322d2f41bc5b757.jpg)

![](images/6a2e13bd205dc0d96d9279eeed56981f4cfd73c09adcb46fbb5ad667cbdd863b.jpg)  
Fig. 4. Cross-site subspace alignment on ABIDE $( k = 1 0 )$ . (A) Principal-angle overlap between the 16 site-specific effective subspaces. (B) Ordered pairwise transfer AUROC increases with subspace overlap; significance is assessed by Mantel-style site-label permutation. (C) The overlap–transfer association remain after residualizing both axes against available site-pair covariates, including sample size, class balance, mean age, sex ratio, and site-mean FC distance.

![](images/e6cdb082e8a97f3f697014bfc3d8d42b6269cc92f9c163ef20f5fff42b927d42.jpg)

![](images/1eb342434487a19cd92ec1d11140422a886ad78515148d61f7fb97f4bca372cb.jpg)  
Fig. 5. Controlled rotation of subspace orientation, on the six largest ABIDE sites $( n _ { s } \geq 4 2$ ; 10 split-halves per site, seven rotation angles $0 ^ { \circ } { - } 9 0 ^ { \circ } ) . ~ ( \mathrm { A } )$ On-thesis test: rotating the leading effective subspace $( k = 1 0 )$ ) drives source– target overlap down and degrades transfer AUROC monotonically toward chance; the shaded vertical band marks the empirical ABIDE overlap. (B) Variance/displacement control: rotating the discriminative direction degrades transfer, whereas rotating a label-orthogonal direction of matched displacement $( \| \Delta \mathbf { x } \| = 6 . 8 2 )$ does not. The rotation is orthogonal and restores the site mean, preserving the feature mean and the covariance eigenvalue spectrum (changes $\stackrel { \cdot } { \sim } 1 0 ^ { - 1 5 } \mathrm { a n d } \sim 1 0 ^ { - 1 3 } )$ , so only orientation is perturbed. Bands denote ±SD across sites and split-halves; effect sizes and the per-site test are reported in the text.

Second, cross-site evaluation should be treated as a geometric question. Poor transfer should not be attributed by default to model capacity or generic domain shift; when sitespecific subspaces are misaligned, the source-site decision directions are simply under-expressed at the target. We therefore suggest that harmonization and domain-adaptation methods be assessed by whether they increase subspace alignment, not only by whether they raise within-dataset AUROC, and that principal-angle overlap serve as a lightweight diagnostic for anticipating transfer before models are trained. Travelingsubject or harmonized multi-site cohorts would allow this alignment view to be tested where acquisition differences are controlled.

These implications are representation-specific. The raw-

BOLD control shows that when a representation does not expose the discriminative signal to a linear model, temporal modeling still recovers information that linear decoding misses, so complex models remain necessary in general. For vectorized FC, however, the leading effective subspace is the object that governs both model comparison and transfer, and treating it as such is more informative than adding architectural complexity alone.

## C. Limitations

Two limitations bound these conclusions. First, the crosssite analysis adjusts for sample size, class balance, mean age, sex ratio, and site-mean FC distance, but not for motion or mean framewise displacement, which were unavailable in the phenotypic table used here; with 16 sites the adjusted dyadic analysis also has limited power. Second, the rotation intervention is a controlled synthetic perturbation on the six largest sites, so its strength lies in the mean- and spectrumpreserving active-placebo dissociation rather than in the persite sample size. Natural site differences are not purely rotations, and traveling-subject or harmonized multi-site cohorts would be needed to test whether the same subspace mechanism holds once population and acquisition differences are tightly controlled.

## V. CONCLUSION

FC classification is strongly shaped by representation geometry. Within a dataset, the leading effective subspace contains much of the discriminative FC signal, allowing simple linear and shallow classifiers to compete with graph, connectomespecific, and Transformer architectures. Across sites, the same concentration becomes a vulnerability: transfer depends on whether different sites orient that subspace the same way, and controlled orientation perturbations degrade transfer toward chance. Together these results trace both behaviors, strong within-dataset performance of simple models and fragile crosssite transfer, to a single geometric origin. FC should therefore be interpreted not merely as a high-dimensional connectome input, but as a low-effective-rank representation whose leading subspace governs decoding and generalization alike.

## CODE AVAILABILITY

Anonymized code and figure scripts are available at: https://anonymous.4open.science/r/LowRank A-CBA0/.

## REFERENCES

[1] J. Kawahara, C. J. Brown, S. P. Miller, B. G. Booth, V. Chau, R. E. Grunau, J. G. Zwicker, and G. Hamarneh, “BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment,” NeuroImage, vol. 146, pp. 1038–1049, 2017.

[2] X. Kan, W. Dai, H. Cui, Z. Zhang, Y. Guo, and C. Yang, “Brain network transformer,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[3] K. Han, Y. Su, L. He, L. Zhan, S. Plis, V. D. Calhoun, and C. Yang, “Rethinking functional brain connectome analysis: do graph deep learning models help?” npj Artificial Intelligence, vol. 2, no. 1, p. 19, 2026.

[4] D. Klepl, B. Rehak Bu´ ckovˇ a, J. Svoboda, D. Tome´ cek, F.ˇ Spaniel, and<sup>ˇ</sup> J. Hlinka, “Domain adaptation enables cross-site classification of firstepisode schizophrenia from multimodal neuroimaging data,” bioRxiv, 2026, feb. 2026.

[5] R. Kashyap et al., “Individual-specific fmri-subspaces improve functional connectivity prediction of behavior,” NeuroImage, vol. 189, pp. 804–812, 2019.

[6] D. S. Margulies et al., “Situating the default-mode network along a principal gradient of macroscale cortical organization,” Proceedings of the National Academy of Sciences, vol. 113, no. 44, pp. 12 574–12 579, 2016.

[7] P. L. Bartlett, P. M. Long, G. Lugosi, and A. Tsigler, “Benign overfitting in linear regression,” Proceedings of the National Academy of Sciences, vol. 117, no. 48, pp. 30 063–30 070, 2020.

[8] A. Tsigler and P. L. Bartlett, “Benign overfitting in ridge regression,” Journal of Machine Learning Research, vol. 24, 2023.

[9] E. S. Finn, X. Shen, D. Scheinost, M. D. Rosenberg, J. Huang, M. M. Chun, X. Papademetris, and R. T. Constable, “Functional connectome fingerprinting: identifying individuals using patterns of brain connectivity,” Nature neuroscience, vol. 18, no. 11, pp. 1664–1671, 2015.

[10] D. C. Van Essen et al., “The WU-Minn human connectome project: an overview,” NeuroImage, vol. 80, pp. 62–79, 2013.

[11] L. Fan et al., “The human brainnetome atlas: A new brain atlas based on connectional architecture,” Cerebral Cortex, vol. 26, no. 8, pp. 3508– 3526, 2016.

[12] A. Di Martino et al., “The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism,” Molecular Psychiatry, vol. 19, pp. 659–667, 2014.

[13] ADHD-200 Consortium, “The ADHD-200 consortium: a model to advance the translational potential of neuroimaging in clinical neuroscience,” Frontiers in Systems Neuroscience, vol. 6, p. 62, 2012.

[14] N. Tzourio-Mazoyer et al., “Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain,” NeuroImage, vol. 15, no. 1, pp. 273–289, 2002.