# CardioDrift: Automatic Cardiac Motion Tracking on a 3D Atlas Mesh via Shape-aware Coherent Point Drift with Slice Shift Correction

Reza Akbari Movahed, Abuzar Rezaee, Shatha Alamri, Edmond S.L. Ho, Fani Deligianni, Ali Gooya

Abstract—Automatic cardiac motion tracking from cine cardiac magnetic resonance (CMR) imaging is essential for quantifying myocardial motion abnormalities associated with cardiovascular disease. Existing methods fall into two groups: those that represent motion without anatomical correspondence across subjects and therefore limit inter-subject motion assessment; and those that represent motion with globally corresponding points, which typically require a time-consuming and error-prone semiautomatic step to fit a cardiac atlas to the subject space. To address these limitations, this study proposes CardioDrift, a fully automatic, atlas-based probabilistic framework that represents subject-specific cardiac motion on a 3D atlas mesh with globally corresponding points. The Coherent Point Drift (CPD) formulation is generalised to jointly register the atlas surfaces of multiple cardiac regions to the anatomically corresponding contours extracted from cine CMR segmentation maps throughout the cardiac cycle. Slice-shift correction is embedded in the same probabilistic model, estimating slice-wise translations that suppress the misalignment artefact limiting atlas-fitting accuracy. Evaluation on 1,301 randomly selected UK Biobank subjects and the M&M dataset, CardioDrift attains Dice similarity coefficients (DSCs) of 92.23±8.70%, 82.11±7.56%, and 88.69±4.13% for the left ventricle (LV), myocardium (Myo), and right ventricle (RV), exceeding the strongest baseline on every distance and overlap metric of both datasets.

Index Terms—Cardiac motion tracking, cine CMR, point-set registration, coherent point drift, slice misalignment.

## I. INTRODUCTION

cause of mortality worldwide [1]. It is typically performed through visual assessment of cine cardiac magnetic resonance (CMR) imaging [2], a process that is subjective, poorly reproducible, and time-consuming [3], motivating considerable effort to automate tracking and quantify motion abnormalities.

Automated methods face two obstacles. Those operating in the image domain represent motion in a subject-specific coordinate space, establishing no anatomical correspondence across individuals and precluding the population-level comparison that assessing pathological motion depends on. Those using a cardiac atlas mesh achieve such correspondence but require a semi-automatic step to fit the atlas to the subject’s space, a process that is error-prone and time-consuming. Moreover, these methods first extract cardiac motion in image space, where it may include motion fields unrelated to the heart, and only then interpolate the motion fields from the image domain onto the mesh, attenuating them in the process.

This work addresses these obstacles with CardioDrift, a nonlearning-based point set registration framework formulated using the Coherent Point Drift (CPD) approach [4] that automatically fits a cardiac atlas mesh into the subject’s cardiac space and propagates it over the cardiac cycle, using only contour points from cine short-axis (SAX) segmentation maps. A rigid stage first aligns the atlas at the end-diastolic (ED) frame while estimating per-slice translations to correct slice misalignment. A non-rigid stage then deforms the aligned atlas to the sliceshift-corrected contours at each subsequent frame through a shared deformation field. While previous methods lack global correspondence in the derived motion, rely on semi-automatic atlas fitting, capture irrelevant motion through image-domain motion extraction, and attenuate motion through interpolation, CardioDrift automatically fits a cardiac atlas to the subject space accurately and propagates it precisely throughout the cardiac cycle. The main contributions are:

• Developing a probabilistic point-set registration framework for tracking cardiac motion on a 3D atlas mesh that provides anatomical correspondence across subjects, requiring only cine SAX segmentation maps as input.

• Generalising CPD to jointly register the left-ventricular endocardium (Endo), left-ventricular epicardium (Epi), and right-ventricular (RV) atlas surfaces to their corresponding contours using a single shared transformation.

• Embedding slice-shift correction in the rigid stage by estimating SAX-slice translations within the same probabilistic model to suppress inter-slice misalignment and improve atlas-fitting performance.

• Evaluating the framework on the UK Biobank and Multi-Centre, Multi-Vendor and Multi-Disease (M&M) datasets, demonstrating more accurate atlas fitting and propagation than existing methods, without manual alignment to the subject space.

## II. BACKGROUND

Most state-of-the-art methods formulate tracking as image registration, estimating dense pixel- or voxel-wise motion fields from cine CMR. Classical approaches build on free-form deformation [5], [6] or the Demons algorithm and its diffeomorphic reformulation [7], [8], while deep-learning methods regress the field in a single forward pass [9]–[11]. Bayesian formulations register long-axis sequences [12] or jointly segment and register SAX slices [13]. They may capture motions unrelated to the heart because they rely on intensity-based image registration metrics. CardioMorphNet addresses this by guiding motion estimation towards the regions identified by a segmentation branch [14]. These methods operate in a subject-specific image space and establish no anatomical correspondence across individuals [15], [16], limiting intersubject motion assessment.

Recent studies have formulated cardiac motion tracking in the mesh/point-set domain to enable both intra-subject and inter-subject motion assessment. For instance, Zakeri et al. proposed a deep motion model that predicts the cardiac point set at each time frame and identifies outlier points resulting from excessive deviations from the model prediction [17]. Although this approach enables intra-subject assessment through outlier detection, its output point sets lack intersubject correspondence, limiting abnormality assessment to the intra-subject setting. To enable inter-subject correspondence, DeepMesh fits an atlas mesh whose points correspond anatomically across individuals to the subject’s cardiac space and warps it across the cycle using motion fields estimated from multi-view CMR [18]. This method represents cardiac motion with globally corresponding points under several constraints: the fitting is semi-automatic and requires manual landmark selection; the motion fields are estimated in image space and interpolated onto the mesh in physical coordinates, attenuating extracted motion; and the intensity-based similarity term admits motions unrelated to the heart. Automating atlas fitting is complicated by the heart’s variable position and orientation in physical space [19] and by slice misalignment [20].

Point-set registration could enable atlas-based motion tracking without manual atlas fitting or image-to-mesh interpolation by operating directly in physical coordinates. Here, contours extracted from cine CMR segmentation maps serve as the fixed point sets, while the corresponding atlas surfaces are the moving point sets. Estimating transformations over the cardiac cycle that fit the atlas surfaces with the subject’s contours represents cardiac motion on the atlas mesh. Non-learningbased methods derive transformations by estimating point-wise correspondences through iterative optimisation or probabilistic modelling [21]. Among these, CPD performs strongly [4], formulating registration as probability density estimation, where moving points parameterise a Gaussian mixture model (GMM) centroids and correspondences are expressed probabilistically and jointly refined with the transformation. Learning-based methods instead predict transformations from latent pointwise features, but generalise poorly beyond their training distribution and degrade under outliers, noise, and sparsity [22]. These limitations are particularly relevant to CMRderived contours, which are sparse through-plane, affected by slice misalignment, and prone to segmentation errors. CPD therefore offers a reliable alternative, but its standard formulation neither enforces correspondence between each atlas surface and its anatomical counterpart nor corrects slice misalignment, which limits fitting accuracy.

## III. MATERIALS AND METHODS

## A. Proposed Framework

The framework registers the Endo, Epi, and RV atlas surfaces to the subject’s corresponding contour points throughout the cardiac cycle. Let $g \in \{ \mathrm { E n d o } , \mathrm { E p i } , \mathrm { R V } \}$ denote the cardiac anatomical region and $l = 1 , \ldots , L$ the frames in the cardiac sequence, with $l \ = \ 1$ denoting the ED time point. For each region, $\mathbf { X } _ { I } ^ { g } \in \mathbb { R } ^ { N _ { g } \times D }$ is the subject’s contour point set at frame l and $\mathbf { Y } ^ { g } ~ \in ~ \mathbb { R } ^ { M _ { g } \times D }$ is the atlas point set, where $N _ { g }$ and $M _ { g }$ are their numbers of points and $D = 3$ Row-wise concatenation over the three regions in a fixed order, $\mathrm { C o n c a t } _ { g } ( \cdot )$ , gives ${ \bf { X } } _ { l } = { \bf { \Psi } }$ Conca $\mathrm { t } _ { g } ( \mathbf { X } _ { l } ^ { \bar { g } } ) \in \mathbb { R } ^ { N \times D }$ and Y = Conca $\hat { \mathbf { \Psi } } _ { p } ( \mathbf { Y } ^ { g } ) \mathbf { \Psi } \in \mathbf { \Psi } \mathbb { R } ^ { M \times D }$ , with $\begin{array} { r } { \boldsymbol { N } ^ { \mathrm { ~ * ~ } } = \sum _ { g } \boldsymbol { N } _ { g } } \end{array}$ and $\begin{array} { r } { M \ = \ \sum _ { q } M _ { g } } \end{array}$ . Given $\{ \mathbf { X } _ { l } \} _ { l = 1 } ^ { L }$ and Y, we aim to obtain $\{ \mathbf { Z } _ { l } \} _ { l = 1 } ^ { L }$ , where $\mathbf { Z } _ { l } ~ \in ~ \mathbb { R } ^ { M \times D }$ is the registered atlas in the subject’s cardiac space at frame l.

Fig. 1 shows the diagram of the proposed framework, whose input is a sequence of L SAX volumes $\{ \mathbf { I } _ { l } \} _ { l = 1 } ^ { L }$ . Contour extraction proceeds identically at every frame: each SAX volume is segmented into the left ventricle (LV), myocardium (Myo), and RV, the Endo, Epi, and RV contours are extracted from the segmentation masks by the marching squares algorithm together with logical operations, and the result is mapped from voxel to physical coordinates using the affine transformation in the DICOM header, outputting $\mathbf { X } _ { l }$

Tracking proceeds in two phases. In the alignment phase (Fig. 1a), the rigid parameters and the per-slice translations $\{ \mathbf { t } _ { k } \in \mathbb { R } ^ { D \times 1 } \} _ { k = 1 } ^ { K }$ over the K SAX slices are estimated jointly from the atlas points Y and the ED contour points X<sub>ED</sub>; applying the rigid transformation to Y yields the fitted atlas $\widetilde { \mathbf Y }$ in the subject’s cardiac space. In the propagation phase (Fig. 1b), the contour points $\mathbf { X } _ { l }$ at each frame are translated by the estimated slice-shift corrections to output $\widetilde { \mathbf { X } } _ { l } ,$ and the nonrigid registration deforms $\widetilde { \mathbf Y }$ to $\widetilde { \mathbf { X } } _ { l }$ , yielding the frame-specific fitted atlas $\mathbf { Z } _ { l }$ . Repeating this over all $L$ frames provides $\{ \mathbf { Z } _ { l } \} _ { l = 1 } ^ { L }$

## B. Shape-aware CPD Generalisation

Both stages build on the following generalisation of CPD, which registers the Endo, Epi, and RV atlas surfaces to their anatomically corresponding contour point sets under a single transformation. Following the original CPD formulation [4], registration is cast as probability density estimation: the atlas points of each region are the centroids of a GMM and the contour points are observations drawn from it, so that $p ( \mathbf { x } _ { n } ^ { g } ) =$ $\sum _ { m = 1 } ^ { M _ { g } + 1 } \bar { p } ( m ^ { g } ) p ( { \bf x } _ { n } ^ { g } \mid m ^ { g } )$ , where $\mathbf { x } _ { n } ^ { g } \in \mathbb { R } ^ { 1 \times D }$ is the nth contour point and $\mathbf { y } _ { m } ^ { g } \in \mathbb { R } ^ { 1 \times D }$ the mth atlas point of region g. Each component likelihood is an isotropic Gaussian,

$$
p ( \mathbf { x } _ { n } ^ { g } \mid m ^ { g } ) = \mathcal { N } \big ( \mathbf { x } _ { n } ^ { g } ; \mathbf { y } _ { m } ^ { g } , \sigma _ { g } ^ { 2 } \mathbf { I } \big ) ,\tag{1}
$$

![](images/5eacfa945d36c5f1999d34d9923e43a94c200212bf4c51d5e421029a6264463b.jpg)  
Fig. 1. Overview of CardioDrift. (a) Alignment phase: the atlas Y is fitted to the subject’s cardiac space at ED, which also estimates the per-slice translations. (b) Propagation phase: $\widetilde { \mathbf Y }$ is deformed to the corrected contours $\widetilde { \mathbf { X } } _ { l }$ at each frame, providing ${ \bf Z } _ { l } . ~ \{ { \bf R } , { \bf t } , s \}$ are the rigid parameters, $\mathbf { t } _ { k }$ are the slice shift correction translations, and ϕ is the non-rigid deformation field.

with equal priors $p ( m ^ { g } ) = 1 / M _ { g }$ and an additional uniform component $p ( \mathbf { x } _ { n } ^ { g } \mid M _ { g } + 1 ) = 1 / N _ { g }$ , weighted by $\omega _ { g } \in ( 0 , 1 )$ accounting for outliers.

The GMM centroids are re-parameterised by a transformation parameter set $\theta ,$ estimated by minimising the weighted negative log-likelihood

$$
E ( \theta , \sigma _ { g } ^ { 2 } ) = - \sum _ { g } \alpha _ { g } \sum _ { n = 1 } ^ { N _ { g } } \log \sum _ { m = 1 } ^ { M _ { g } + 1 } p ( m ^ { g } ) p ( \mathbf { x } _ { n } ^ { g } \mid m ^ { g } ) ,\tag{2}
$$

where $\alpha _ { g } ~ \in ~ [ 0 , 1 ]$ weights region $g .$ The posterior that the mth component generated the nth observation, $P _ { m n } ^ { g } \triangleq p ( m ^ { g } \ |$ $\mathbf { x } _ { n } ^ { g } )$ representing the soft correspondence between $\mathbf { x } _ { n } ^ { g }$ and $\mathbf { y } _ { m } ^ { g }$ follows from Bayes’ rule as

$$
P _ { m n } ^ { g } = \frac { \exp \biggl ( - \frac { 1 } { 2 \sigma _ { g } ^ { 2 } } \| \mathbf { x } _ { n } ^ { g } - \tau ( \mathbf { y } _ { m } ^ { g } ; \theta ) \| ^ { 2 } \biggr ) } { \displaystyle \sum _ { j = 1 } ^ { M _ { g } } \exp \biggl ( - \frac { 1 } { 2 \sigma _ { g } ^ { 2 } } \| \mathbf { x } _ { n } ^ { g } - \tau ( \mathbf { y } _ { j } ^ { g } ; \theta ) \| ^ { 2 } \biggr ) + c } ,\tag{3}
$$

where $\tau ( \cdot ; \theta )$ is the transformation parameterised by $\theta$ and $\begin{array} { r } { c = ( 2 \pi \dot { \sigma _ { g } ^ { 2 } } ) ^ { D / 2 } \frac { w _ { g } } { 1 - w _ { a } } \frac { M _ { g } } { N _ { a } } } \end{array}$

The optimal θ and $\sigma _ { g } ^ { 2 }$ are derived by expectationmaximisation (EM): the E-step computes $P _ { m n } ^ { g }$ from the current estimates, and the M-step minimises the expected complete-data negative log-likelihood evaluated at the previous values $P _ { m n } ^ { g ( \mathrm { o l d } ) }$ , which upper-bounds E. Dropping terms independent of θ and $\sigma _ { g } ^ { 2 } .$

$$
\begin{array} { c } { { \displaystyle Q ( \theta , \sigma _ { g } ^ { 2 } ) = \sum _ { g } \alpha _ { g } \bigg ( \frac { 1 } { 2 \sigma _ { g } ^ { 2 } } \sum _ { n , m } P _ { m n } ^ { g ( \mathrm { o l d } ) } \| \mathbf { x } _ { n } ^ { g } - \tau ( \mathbf { y } _ { m } ^ { g } ; \theta ) \| ^ { 2 } } } \\ { { + \frac { N _ { p } ^ { g } D } { 2 } \log \sigma _ { g } ^ { 2 } \bigg ) , } } \end{array}\tag{4}
$$

where $\begin{array} { r } { N _ { p } ^ { g } \ = \ \sum _ { n , m } P _ { m n } ^ { g ( \mathrm { o l d } ) } } \end{array}$ . The form of $\tau ( \cdot ; \theta )$ is given separately for the rigid and non-rigid stages below.

## C. Rigid and Slice-Shift-Correction Transformation

This stage fits the atlas surfaces to their corresponding contours while estimating per-slice translations. The transformation is

$$
\tau ( \mathbf { y } _ { m } ^ { g } ; \mathbf { t } _ { k } , \mathbf { R } , \mathbf { t } , s ) = \mathbf { t } _ { k } + s \mathbf { R } \mathbf { y } _ { m } ^ { g } + \mathbf { t } ,\tag{5}
$$

where $\mathbf { x } _ { n k } ^ { g }$ is the nth contour point of region g on the kth SAX slice, $\mathbf { t } _ { k }$ denotes slice-shift translations for the point on the kth slice. $\mathbf { R } \in \mathbb { R } ^ { D \times D } , \mathbf { t } \in \mathbb { R } ^ { D \times 1 }$ , and $s \in \mathbb { R } ^ { + }$ are the rotation, global translation, and isotropic scaling. Substituting into (4) gives

$$
\begin{array} { l } { { \displaystyle Q ( { \bf t } _ { k } , { \bf R } , { \bf t } , s , \sigma _ { g } ^ { 2 } ) = \sum _ { g } \alpha _ { g } \bigg ( \frac { 1 } { 2 \sigma _ { g } ^ { 2 } } \sum _ { k , n , m } P _ { m n k } ^ { g ( \mathrm { o l d } ) } } } \\ { { \displaystyle ~ \times ~ \| { \bf x } _ { n k } ^ { g } - { \bf t } _ { k } - s { \bf R } { \bf y } _ { m } ^ { g } - { \bf t } \| ^ { 2 } + \frac { N _ { p } ^ { g } D } { 2 } \log \sigma _ { g } ^ { 2 } \bigg ) } , } \end{array}\tag{6}
$$

the sum runs over $k = 1 , \ldots , K , n = 1 , \ldots , N _ { g } ^ { k }$ , and $m =$ $1 , \ldots , M _ { g }$ with $N _ { g } ^ { k }$ the number of contour points of region g on slice $k ,$ and $\begin{array} { r } { \dot { N } _ { p } ^ { g } = \sum _ { k , n , m } P _ { m n k } ^ { g ( \mathrm { o l d } ) } } \end{array}$ . The corresponding posterior is

$$
P _ { m n k } ^ { g } = \frac { \displaystyle \exp \left( - \frac { 1 } { 2 \sigma _ { g } ^ { 2 } } \| \mathbf { x } _ { n k } ^ { g } - \mathbf { t } _ { k } - s \mathbf { R } \mathbf { y } _ { m } ^ { g } - \mathbf { t } \| ^ { 2 } \right) } { \displaystyle \sum _ { j = 1 } ^ { M _ { g } } \exp \left( - \frac { 1 } { 2 \sigma _ { g } ^ { 2 } } \| \mathbf { x } _ { n k } ^ { g } - \mathbf { t } _ { k } - s \mathbf { R } \mathbf { y } _ { j } ^ { g } - \mathbf { t } \| ^ { 2 } \right) + c } .\tag{7}
$$

Minimising (6) jointly over $\{ \mathbf { R } , \mathbf { t } , s , \sigma _ { g } ^ { 2 } , \mathbf { t } _ { k } \}$ is intractable, as the rigid parameters and slice-shift translations enter the objective in a coupled manner. We therefore alternate two stages within each EM iteration: the rigid parameters are updated on contour points translated by the current $\mathbf { t } _ { k }$ , then $\mathbf { t } _ { k }$ is re-estimated from the residual misalignment between the warped atlas $\widetilde { \mathbf { Y } } ^ { g }$ and the original contours $\mathbf { X } ^ { g }$ . Both derivations follow below.

## D. Rigid Parameters Estimation

This step estimates R, t, and s from the atlas point sets and the contour points after the current slice-shift translation has been applied. Writing $\tilde { \mathbf { x } } _ { n } ^ { g } = \mathbf { x } _ { n k } ^ { g } - \mathbf { t } _ { k }$ , the objective (6) reduces to (4) under $\tau ( \mathbf { y } _ { m } ^ { g } ) = s \mathbf { R } \mathbf { y } _ { m } ^ { g } + \mathbf { t }$ with $\widetilde { \mathbf { X } } ^ { g }$ in place of $\mathbf { X } ^ { g } ;$ the posteriors $P _ { m n } ^ { g }$ following from (3) under the same substitutions. Minimising this objective over R is constrained by $\mathbf { R } ^ { \top } \mathbf { R } = \mathbf { I }$ and det(R) = 1, and admits a closed form via Lemma 1 of [23]: the rotation maximising $\operatorname { t r } ( \mathbf { A } ^ { \top } \mathbf { R } )$ for a known real $\mathbf { A } \in \bar { \mathbb { R } } ^ { D \times D }$ is $\mathbf { R } = \mathbf { U } \mathbf { C } \mathbf { V } ^ { \top }$ , with $\mathbf { U S V } ^ { \top } =$ $\operatorname { S V D } ( \mathbf { A } ) , \mathbf { C } \ = \ \mathrm { d i a g } ( 1 , \dots , \mathrm { d e t } ( \mathbf { U V } ^ { \top } ) )$ ), and diag(·), tr(·) the diagonal matrix formed from its arguments and the trace.

To reach this form, t is eliminated by setting $\partial Q / \partial \mathbf { t } = 0 .$ giving

$$
\mathbf { t } = \sum _ { g } w _ { g } \big ( \pmb { \mu } _ { x } ^ { g } - s \mathbf { R } \pmb { \mu } _ { y } ^ { g } \big ) = \bar { \pmb { \mu } } _ { x } - s \mathbf { R } \bar { \pmb { \mu } } _ { y } ,\tag{8}
$$

where $\begin{array} { r } { \pmb { \mu } _ { x } ^ { g } \ = \ \frac { 1 } { N _ { n } ^ { g } } ( \widetilde { \mathbf { X } } ^ { g } ) ^ { \top } ( \mathbf { P } ^ { g } ) ^ { \top } \mathbf { 1 } } \end{array}$ and $\begin{array} { r } { \pmb { \mu } _ { y } ^ { g } \ = \ \frac { 1 } { N _ { n } ^ { g } } ( \mathbf { Y } ^ { g } ) ^ { \top } \mathbf { P } ^ { g } \mathbf { 1 } } \end{array}$ $\mathbf { P } ^ { g } \in \mathbb { R } ^ { M _ { g } \times N _ { g } }$ collects the posteriors $P _ { m n } ^ { g }$ , and 1 is a column vector of ones of the appropriate length. The region weights

$$
w _ { g } = \frac { \gamma _ { g } N _ { p } ^ { g } } { \sum _ { h } \gamma _ { h } N _ { p } ^ { h } } , \qquad \gamma _ { g } \triangleq \frac { \alpha _ { g } } { \sigma _ { g } ^ { 2 } } ,\tag{9}
$$

sum to one over the three regions, h indexing the same set as $^ { g , }$ so that t is the offset between the pooled centroids $\bar { \pmb { \mu } } _ { x } =$ $\sum _ { g } w _ { g } \pmb { \mu } _ { x } ^ { g }$ and $\begin{array} { r } { \bar { \pmb { \mu } } _ { y } = \sum _ { g } w _ { g } \pmb { \mu } _ { y } ^ { g } } \end{array}$

Writing $\overline { { \mathbf { X } } } ^ { g } = \widetilde { \mathbf { X } } ^ { g } - \mathbf { 1 } \hat { \mu } _ { x } ^ { \top }$ and $\overline { { \mathbf { Y } } } ^ { g } = \mathbf { Y } ^ { g } - \mathbf { 1 } \bar { \mu } _ { y } ^ { \top }$ , and substituting (8) back leaves no region-wise residual offset and reduces $Q$ to $c _ { 1 } \operatorname { t r } ( \mathbf { A } ^ { \top } \mathbf { R } ) + c _ { 2 }$ , with $c _ { 1 } > 0 , c _ { 2 }$ independent of R, and

$$
\mathbf { A } = \sum _ { g } \gamma _ { g } \left( \overline { { \mathbf { X } } } ^ { g } \right) ^ { \top } \left( \mathbf { P } ^ { g } \right) ^ { \top } \overline { { \mathbf { Y } } } ^ { g } .\tag{10}
$$

Minimising $Q$ over R is therefore equivalent to maximising $\operatorname { t r } ( \mathbf { A } ^ { \top } \mathbf { R } )$ , and the optimal rotation follows from Lemma 1. Setting the remaining partial derivatives to zero gives

$$
\begin{array} { c } { { \displaystyle s = \frac { \mathrm { t r } ( { \bf A } ^ { \top } { \bf R } ) } { \sum _ { g } \gamma _ { g } \mathrm { t r } \left[ ( \overline { { { { \bf Y } } } } ^ { g } ) ^ { \top } \mathrm { d i a g } ( { \bf P } ^ { g } { \bf 1 } ) \overline { { { { \bf Y } } } } ^ { g } \right] } , } } \\ { { \displaystyle \sigma _ { g } ^ { 2 } = \frac { 1 } { D N _ { p } ^ { g } } \Big ( \mathrm { t r } \Big [ ( \overline { { { { \bf X } } } } ^ { g } ) ^ { \top } \mathrm { d i a g } ( ( { \bf P } ^ { g } ) ^ { \top } { \bf 1 } ) \overline { { { { \bf X } } } } ^ { g } \Big ] - 2 s \mathrm { t r } ( { \bf A } _ { g } ^ { \top } { \bf R } ) } } \\ { { \displaystyle ~ + s ^ { 2 } \mathrm { t r } \Big [ ( \overline { { { { \bf Y } } } } ^ { g } ) ^ { \top } \mathrm { d i a g } ( { \bf P } ^ { g } { \bf 1 } ) \overline { { { { \bf Y } } } } ^ { g } \Big ] \Big ) , } } \end{array}\tag{11}
$$

where $\mathbf { A } _ { g } = ( \overline { { \mathbf { X } } } ^ { g } ) ^ { \top } ( \mathbf { P } ^ { g } ) ^ { \top } \overline { { \mathbf { Y } } } ^ { g }$ . Each EM iteration of this stage computes $\mathbf { P } ^ { g }$ via (3), forms $\gamma _ { g }$ from the current $\sigma _ { g } ^ { 2 } .$ , then applies (8)–(11) in closed form.

## E. Slice-wise Translation Estimation

This step estimates the per-slice translations $\{ \mathbf { t } _ { k } \} _ { k = 1 } ^ { K }$ , given the rigidly warped atlas points $\widetilde { \mathbf { y } } _ { m } ^ { g } \ = \ s \mathbf { R } \mathbf { y } _ { m } ^ { g } + \mathbf { t }$ and the original contour points $\mathbf { X } ^ { g }$ . Holding the rigid parameters fixed, (6) reduces to

$$
Q ( \mathbf { t } _ { k } ) = \sum _ { g } \frac { \alpha _ { g } } { 2 \sigma _ { g } ^ { 2 } } \sum _ { k , n , m } P _ { m n k } ^ { g ( \mathrm { o l d } ) } \| \mathbf { x } _ { n k } ^ { g } - \mathbf { t } _ { k } - \widetilde { \mathbf { y } } _ { m } ^ { g } \| ^ { 2 } ,\tag{12}
$$

with $P _ { m n k } ^ { g ( \mathrm { o l d } ) }$ computed via (7).

Each $\mathbf { t } _ { k }$ must lie in the SAX imaging plane. The plane orientation is estimated from the data: the contour points on the middle SAX slice $\{ \mathbf { p } _ { r } \} _ { r = 1 } ^ { N _ { m } } \mathbf { p } _ { r } \in \mathbb { R } ^ { 1 \times D }$ where $N _ { m }$ is the total number of such points, are centred to give pe . Then, the eigenvectors of the scatter matrix $\begin{array} { r } { \mathbf { H } = \sum _ { r } \widetilde { \mathbf { p } } _ { r } ^ { \top } \mathbf { \widetilde { p } } _ { r } \in \mathbb { R } ^ { D \times D } } \end{array}$ are obtained by SVD. The eigenvector of smallest eigenvalue $\mathbf { v } _ { 1 }$ approximates the slice-stack normal, while the remaining $\mathbf { v } _ { 2 } , \ldots , \mathbf { v } _ { D }$ span the imaging plane and form the projection matrix $\mathbf { E } = [ \mathbf { v } _ { 2 } , \ldots , \mathbf { v } _ { D } ] ^ { \top } \in \mathbb { R } ^ { ( D - 1 ) \times D }$ . Each translation is then parameterised as $\mathbf t _ { k } = \mathbf E ^ { \top } \mathbf a _ { k }$ with $\mathbf { a } _ { k } \in \mathbb { R } ^ { ( D - 1 ) \times 1 }$ guaranteeing $\mathbf { t } _ { k }$ lies in the estimated plane.

Substituting this parameterisation into (12), retaining only the terms in ${ \bf a } _ { k }$ , and adding the regularisation term $\begin{array} { r } { \frac { \lambda } { 2 } \sum _ { k } \mathbf { a } _ { k } ^ { \top } \mathbf { a } _ { k } } \end{array}$ that prevents unrealistically large translations, where $\lambda \ >$ 0 is the regularisation coefficient, the stationary condition ${ \partial Q } / { \partial { \bf a } _ { k } } = 0$ together with $\mathbf t _ { k } = \mathbf E ^ { \top } \mathbf a _ { k }$ yields

$$
\mathbf { t } _ { k } = \frac { \mathbf { E } ^ { \top } \mathbf { E } \sum _ { g , n , m } \frac { \alpha _ { g } } { \sigma _ { g } ^ { 2 } } P _ { m n k } ^ { g ( \mathrm { o l d } ) } \left( \mathbf { x } _ { n k } ^ { g } - \widetilde { \mathbf { y } } _ { m } ^ { g } \right) } { \displaystyle \sum _ { g , n , m } \frac { \alpha _ { g } } { \sigma _ { g } ^ { 2 } } P _ { m n k } ^ { g ( \mathrm { o l d } ) } + \lambda } ,\tag{13}
$$

applied in the M-step once $P _ { m n k } ^ { g ( \mathrm { o l d } ) }$ has been computed in the E-step.

## F. Non-rigid Transformation

The non-rigid stage deforms the fitted atlas to the contours at each frame. With the fitted atlas points $\widetilde { \mathbf Y } = \mathrm { C o n c a t } _ { g } ( \widetilde { \mathbf Y } ^ { g } )$ the deformed point set is $\textbf { Z } = \widetilde { \textbf { Y } } + \phi ( \widetilde { \textbf { Y } } )$ , where $\phi$ is the deformation field applied uniformly across all regions. Following [4], $\phi ( \widetilde { \mathbf { Y } } ) = \mathbf { G } \mathbf { W }$ , with $\mathbf { W } \in \mathbb { R } ^ { M \times D }$ an unknown weight matrix and $\mathbf { G } ~ \in ~ \mathbb { R } ^ { M \times M }$ a Gaussian kernel matrix constructed by,

$$
G _ { i j } = \exp \left( - \frac { \| \mathbf { y } _ { i } - \mathbf { y } _ { j } \| ^ { 2 } } { 2 \rho ^ { 2 } } \right) , \quad i , j = 1 , \ldots , M ,\tag{14}
$$

whose bandwidth $\rho > 0$ controls the spatial extent of the deformation. Substituting $\tau (  { { \widetilde { \mathbf { Y } } } } ,  { \mathbf { W } } ) \ = \  { { \widetilde { \mathbf { Y } } } } +  { { \mathbf { G } } }  { { \mathbf { W } } }$ into (4) and adding the penalty $\begin{array} { r } { \frac { \beta } { 2 } \operatorname { t r } [ \mathbf { W } ^ { \top } \mathbf { G } \mathbf { W } ] , } \end{array}$ , in which $\beta ~ > ~ 0$ governs the smoothness of the deformation field, specialises the objective to a function $Q ( \mathbf { W } , \sigma _ { q } ^ { 2 } )$ , with the posteriors P<sup>g</sup> following from (3) under this τ. Writing $\mathbf { G } ( \boldsymbol { m } , : )$ for the mth row of G, setting $\partial Q / \partial \mathbf { W } = 0$ and $\partial Q / \partial \sigma _ { g } ^ { 2 } = 0$ yields, as shown in [4],

$$
( \mathbf { G } + \beta \mathbf { D } _ { \sigma } ) \mathbf { W } = \mathrm { d i a g } ( \mathbf { P 1 } ) ^ { - 1 } \mathbf { P X } - \widetilde { \mathbf { Y } } ,\tag{15}
$$

$$
\sigma _ { g } ^ { 2 } = \frac { 1 } { N _ { p } ^ { g } D } \sum _ { n , m } { P } _ { m n } ^ { g ( \mathrm { o l d } ) } \left\| \tilde { \mathbf { x } } _ { n } ^ { g } - \tilde { \mathbf { y } } _ { m } ^ { g } - \mathbf { G } ( m , : \mathbf { \alpha } ) \mathbf { W } \right\| ^ { 2 } ,\tag{16}
$$

where $\widetilde { \mathbf { X } } = \mathrm { C o n c a t } _ { q } ( \widetilde { \mathbf { X } } ^ { g } )$ , P = blkdiag (P<sup>g</sup>), and ${ \bf D } _ { \sigma } = { \bf \Phi }$ blkdia $\underline { { \boldsymbol { \mathfrak { r } } } } _ { q } \bigl ( \sigma _ { q } ^ { 2 } / \alpha _ { g } \mathrm { d i a g } ( \mathbf { P } ^ { g } \mathbf { 1 } ) ^ { - 1 } \bigr )$ . Initialised with $\mathbf { W } = \mathbf { 0 }$ and $\begin{array} { r } { \sigma _ { g } ^ { 2 } = \frac { { \bf \tilde { \Delta } } _ { 1 } { \bf \tilde { \Delta } } ^ { s } } { D N _ { g } M _ { g } } \sum _ { n , m } \lVert \tilde { \bf x } _ { n } ^ { g } - { \bf y } _ { m } ^ { g } \rVert ^ { 2 } } \end{array}$ , the EM alternates between computing $\dot { P } _ { m n } ^ { g ( \mathrm { o l d } ) }$ via (3) and solving (15)–(16) until convergence.

## IV. EXPERIMENTS AND RESULTS

## A. Experimental Setup

The framework was evaluated on two datasets. The first comprises cine CMR SAX volumes from a random sample of 1,301 UK Biobank subjects [24], segmented into the LV, Myo, and RV regions at every frame of the cardiac cycle with the automated tool of Bai et al. [25]. The second is the M&M dataset [26], which includes 134 subjects; it was acquired at several clinical centres with scanners from four vendors and includes both healthy and pathological anatomy. Expertannotated masks are provided for M&M only at the ED and end-systolic (ES) time points. The bi-ventricular atlas of [27] served as the template, with the Endo, Epi, and RV surfaces decimated to 359, 459, and 402 points, respectively.

The rigid stage is compared against two classical methods, iterative closest point (ICP) and rigid CPD, and two learningbased methods, PCRNet [28] and PointNetLK [29]. The non rigid stage is compared against non-rigid CPD, CPD-Net [30], a learning-based extension of CPD, and DeepMesh [18], which estimates motion fields from the cine images and interpolates them onto the mesh vertices. All non-rigid methods deform the atlas fitted by CardioDrift’s rigid stage and receive contour points pre-translated by its slice-shift corrections, so the comparison isolates the deformation model; for DeepMesh, this replaces the semi-automatic initialisation of the original formulation. Ablations of the slice-shift correction and the shape-aware modelling are also reported. Both regularisation coefficients were selected by sweeps on the two datasets, given in full in the supplementary material. We adopt $\lambda = 5 \times 1 0 ^ { 3 }$ which maximises UK Biobank accuracy at a mean per-slice shift of 2.6 mm, within the range reported by Tarroni et al. [20], and $\beta = 3 . 1 6 ,$ , which balances accuracy against the mean Laplacian of the registered mesh; both are used unchanged for the two datasets.

Performance is quantified by the mean surface distance (MSD) and the 95th-percentile Hausdorff distance (HD95), both in millimetres, computed between the fitted atlas surfaces and the subject contour points, and by the Dice similarity coefficient (DSC) between the reference masks and those obtained by rasterising the fitted surfaces onto the SAX volume grid. The rigid stage is evaluated at the ED frame, at which the atlas is fitted; the non-rigid stage is evaluated across all frames on UK Biobank and, following annotation availability, at ED and ES only on M&M.

## B. Rigid Registration Performance

The UK Biobank rigid block of Table I reports the quantitative comparison on that cohort. Against the baselines, CardioDrift achieves the lowest HD95 and MSD for all three regions and the highest DSC for the LV, Myo, and RV. CPD outperforms ICP on every metric but still yields substantially larger distances and lower overlap than CardioDrift, while ICP obtains the largest geometric errors. PCRNet improves on PointNetLK in HD95 and MSD and attains a competitive RV DSC of 75.04%, yet both lag behind throughout, reflecting the difficulty of deriving correspondences in a single forward pass from sparse and heterogeneous contours.

The same comparison on M&M is reported in Table I. The ordering of the methods is preserved, with CPD again the strongest baseline and ICP and PointNetLK the weakest, but absolute performance is lower for every method, while the margin of CardioDrift over the strongest baseline holds up, at 28.2 percentage points of LV DSC and 28.3 points of Myo DSC against 22.6 and 28.5 on UK Biobank, and remains substantial on all nine metrics. The learning-based methods degrade most in relative terms: PointNetLK loses 15.3 percentage points of RV DSC (51.48% to 36.16%) and PCRNet 11.0 points (75.04% to 64.06%). This is consistent with the domain-shift sensitivity of single-pass learned registration.

Fig. 2(a) shows a representative UK Biobank case. The surfaces fitted by CardioDrift closely follow the contour points across all three regions, and the rasterised masks closely match the corresponding regions in the ground-truth masks, whereas the competing methods show noticeable misalignment at the ventricular boundaries and in overall orientation.

## C. Non-rigid Registration Performance

The UK Biobank non-rigid block of Table I reports atlas propagation over the cardiac cycle. CardioDrift achieves the lowest HD95 and MSD for all regions and the highest DSC throughout. The largest gain is in the Myo DSC, which rises from $6 4 . 9 1 \pm 9 . 9 8 \%$ (CPD), $6 0 . 4 1 \pm 1 2 . 5 7 \%$ (CPD-Net), and $6 0 . 9 0 \pm 1 5 . 7 9 \%$ (DeepMesh) to $8 2 . 1 1 \pm 7 . 5 6 \%$ . DeepMesh records the largest surface distances of the four methods, an

Endo HD95 of $1 7 . 5 6 \pm 5 . 4 6$ mm against $9 . 4 7 \pm 3 . 1 4$ mm for CardioDrift.

The M&M non-rigid block reports the corresponding results, which are confined to ED and ES. CardioDrift is best on every metric, and the Myo again shows the widest margin: $7 6 . 8 1 \pm 1 2 . 4 3 \%$ against $5 5 . 8 0 \pm 1 6 . 1 8 \%$ for CPD, 48.35 ± 17.29% for CPD-Net, and $3 8 . 0 4 { \pm } 1 4 . 8 7 \%$ for DeepMesh, gaps of 21.0, 28.5, and 38.8 percentage points that exceed those on UK Biobank. The two learned methods are the most affected by the change of cohort: CPD-Net falls behind plain CPD for every metric except LV DSC, while on UK Biobank the two remain close, and DeepMesh loses 22.9 percentage points of Myo DSC and 18.6 of RV DSC between the datasets, against 5.3 and 4.2 for CardioDrift.

Fig. 2(b) shows a representative UK Biobank tracking result. The atlases propagated by CardioDrift stay closer to the contour points throughout the cycle and delineate the ED-to-ES transition more faithfully than the three baselines, most visibly in the Myo at ES (frames 20 and 30), indicating that it captures the large inward displacement of the myocardial wall while keeping the mesh smooth and anatomically plausible.

## D. Slice-Shift Correction Performance

The two CardioDrift rows of the UK Biobank rigid block in Table I isolate the effect of slice-shift correction. Enabling it improves every LV and Myo metric, most notably Myo DSC (55.76±12.68% to $7 0 . 2 5 \pm 1 2 . 2 7 \% )$ , while the three RV metrics are marginally better without it. These population-level differences are modest, but the subject-level effect matters: without correction the fitted atlas is visibly offset in a subset of cases (Fig. 3(a)–(d)), an error that would propagate into the non-rigid stage.

The corresponding M&M rows show a weaker but consistent effect. Correction improves seven of the nine metrics, again with Myo DSC the most responsive $( 4 4 . 4 5 \pm 1 8 . 5 6 \% ~ 1$ o $6 0 . 2 1 \pm 1 9 . 1 0 \% )$ ; the exceptions are RV HD95 and RV MSD, marginally lower without correction, matching the pattern on UK Biobank. The reduced benefit follows from the acquisition protocols: UK Biobank stacks are acquired under a standardised sequential breath-hold protocol in which misalignment is systematic and well described by one in-plane translation per slice, whereas the multi-protocol M&M stacks leave more residual misalignment outside the model.

Figs. 3(e)–(h) show coronal and sagittal views of the original and corrected SAX images. In the sagittal view the Endo region at the apex, discontinuous in the original data, becomes connected after correction, though a minor disconnection remains at the base; in the coronal view, correction removes inter-slice discontinuities in the Endo and RV. The estimated translations therefore reduce the misalignment artefact in the stack itself.

## E. Shape-aware Ablation Study

Table II reports the shape-aware ablation. Enabling the terms for all regions provides the best atlas fitting on both datasets: the best overall HD95 of $1 7 . 8 2 \pm 3 . 8 4$ mm and overall DSC of $8 0 . 6 6 \pm 9 . 8 4 \%$ on UK Biobank, and the best overall HD95 of 22.13±6.65 mm, the best Myo DSC, and the best overall DSC of 73.02 ± 18.07% on M&M. The regionspecific configurations confirm that each term benefits its corresponding structure, with the same ordering on both datasets: the Endo-only configuration obtains the lowest Endo and Epi HD95 and the highest LV DSC, and on UK Biobank the highest Myo DSC as well, whereas the RV-only configuration achieves the best RV HD95 and RV DSC. Disabling a region’s term degrades that region most.

TABLE I  
CARDIODRIFT AGAINST BASELINE REGISTRATION METHODS ON BOTH DATASETS, WITH THE SLICE-SHIFT-CORRECTION (SSC) ABLATION. HD95 AND MSD (MM) ARE COMPUTED AGAINST THE SUBJECT CONTOUR POINTS, DSC (%) AGAINST THE REFERENCE MASKS. ON UK BIOBANK, THE RIGID STAGE IS EVALUATED AT ED AND THE NON-RIGID STAGE ACROSS ALL FRAMES; ON M&M, THE RIGID STAGE IS EVALUATED AT ED AND THE NON-RIGID STAGE AT ED AND ES. BOLD: BEST PER COLUMN WITHIN EACH STAGE OF EACH DATASET.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Stage</td><td rowspan="2">Method</td><td colspan="3">HD95 (mm)</td><td colspan="3">MSD (mm)</td><td colspan="3">DSC (%)</td></tr><tr><td>Endo</td><td>Epi</td><td>RV</td><td>Endo</td><td>Epi</td><td>RV</td><td>LV</td><td>Myo</td><td>RV</td></tr><tr><td rowspan="9">UK Biobank</td><td rowspan="9">Rigid</td><td>CardioDrift CardioDrift w/o SSC</td><td>14.18±4.43 17.19±6.94</td><td>16.98±5.06 18.93±7.42</td><td>22.30±4.78 21.16±4.83</td><td>4.50±1.21 5.62±2.60</td><td>4.77±1.39 5.80±2.48</td><td>5.72±1.52 4.96±1.16</td><td>90.72±10.56 86.52±11.90</td><td>70.25±12.27 55.76±12.68</td><td>81.01±9.28 86.19±8.82</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CPD</td><td>21.45±11.30 38.10±6.46</td><td>23.09±11.12</td><td>32.66±15.98</td><td>7.51±3.88</td><td>8.11±4.71</td><td>10.29±6.58</td><td>68.13±29.96</td><td>41.76±27.89</td><td>57.96±31.88</td></tr><tr><td>ICP</td><td></td><td>36.75±6.28</td><td>44.35±6.42</td><td>11.00±1.64</td><td>10.76±1.74</td><td>15.70±2.09</td><td>34.19±17.39</td><td>11.70±5.01</td><td>50.32±5.94</td></tr><tr><td>PCRNet</td><td></td><td>24.79±4.62 25.37±5.08</td><td>40.29±6.85</td><td>8.50±1.56</td><td>7.71±1.82</td><td>9.93±1.54</td><td>56.01±14.77</td><td>25.03±11.18</td><td>75.04±5.76</td></tr><tr><td></td><td>PointNetLK</td><td>36.38±8.67 34.47±8.60</td><td>42.85±7.59</td><td>9.28±1.74</td><td>8.64±1.87</td><td>14.61±2.66</td><td>20.87±19.01</td><td>13.66±8.62</td><td>51.48±10.47</td></tr><tr><td rowspan="3">Non-rigid</td><td>CardioDrift</td><td>9.47±3.14</td><td>10.62±3.26</td><td>14.33±2.92</td><td>3.01±0.35</td><td>3.06±0.35</td><td>2.89±0.33</td><td>92.23±8.70</td><td>82.11±7.56</td><td>88.69±4.13</td></tr><tr><td>CPD</td><td>13.73±4.87</td><td>12.12±5.26</td><td>17.31±5.28</td><td>4.10±1.18</td><td>4.15±1.30</td><td>4.22±2.29</td><td>85.23±13.02</td><td>64.91±9.98</td><td>83.15±10.51</td></tr><tr><td>CPD-Net</td><td>12.94±4.91</td><td>14.31±5.49</td><td>19.45±4.90</td><td>4.58±1.63</td><td>4.69±1.71</td><td>5.16±2.08</td><td>84.40±12.41</td><td>60.41±12.57</td><td>79.71±12.35</td></tr><tr><td rowspan="9">M&amp;M</td><td rowspan="9"></td><td>DeepMesh</td><td>17.56±5.46</td><td>19.88±6.02</td><td>19.92±4.31</td><td>5.91±1.69</td><td>5.84±1.57</td><td>5.27±1.20</td><td>81.19±12.25</td><td>60.90±15.79</td><td>76.46±10.96</td></tr><tr><td>CardioDrift</td><td>17.23±6.98</td><td>22.15±7.29</td><td>27.00±8.62</td><td>5.50±2.36</td><td>6.19±2.72</td><td>7.35±3.66</td><td>83.82±20.34</td><td>60.21±19.10</td><td>75.01±18.54</td></tr><tr><td>CardioDrift w/o SSC</td><td>21.96±9.30</td><td>25.61±9.50</td><td>25.10±7.85</td><td>8.28±4.24</td><td>8.16±4.06</td><td>7.16±2.31</td><td>70.86±22.19</td><td>44.45±18.56</td><td>67.03±19.70</td></tr><tr><td>CPD Rigid</td><td>24.76±10.62</td><td>27.85±9.76</td><td>37.64±18.30</td><td>8.53±3.73</td><td>9.51±4.51</td><td>12.20±6.80</td><td>55.67±33.68</td><td>31.88±26.19</td><td>48.01±30.61</td></tr><tr><td>ICP</td><td>37.38±8.44</td><td>35.45±7.97</td><td>45.07±8.73</td><td>10.80±2.26</td><td>10.86±2.21</td><td>15.66±2.80</td><td>26.11±18.24</td><td>9.03±7.02</td><td>46.92±7.59</td></tr><tr><td>PCRNet</td><td>27.44±5.79</td><td>28.95±6.37</td><td>36.52±8.59</td><td>8.88±2.04</td><td>8.06±2.26</td><td>10.71±2.47</td><td>44.92±21.08</td><td>22.69±11.19</td><td>64.06±10.54</td></tr><tr><td>PointNetLK</td><td>33.45±7.83</td><td>31.32±7.79</td><td>47.21±10.74</td><td>12.58±2.30</td><td>11.09±2.13</td><td>16.94±4.39</td><td>21.32±17.73</td><td>10.86±6.91</td><td>36.16±10.53</td></tr><tr><td>CardioDrift</td><td>10.95±4.32</td><td>13.40±4.55</td><td>14.57±3.55</td><td>3.12±0.64</td><td>3.35±0.69</td><td>2.66±0.46</td><td>86.66±14.22</td><td>76.81±12.43</td><td>84.53±9.33</td></tr><tr><td>CPD Non-rigid</td><td>15.59±8.21</td><td>16.08±7.65</td><td>20.46±9.75</td><td>5.16±2.24</td><td>5.14±2.31</td><td>5.33±3.78</td><td>73.13±24.15</td><td>55.80±16.18</td><td>74.94±21.52</td></tr><tr><td rowspan="3"></td><td>CPD-Net</td><td>17.65±9.22</td><td>20.81±8.78</td><td>22.21±7.86</td><td>6.26±3.05</td><td>6.98±3.24</td><td>6.65±3.78</td><td>73.61±22.50</td><td>48.35±17.29</td><td>68.75±21.69</td></tr><tr><td>DeepMesh</td><td>28.76±11.04</td><td>31.79±10.58</td><td>30.48±10.81</td><td>9.22±3.71</td><td>9.45±3.57</td><td>8.76±2.41</td><td>66.04±22.25</td><td>38.04±14.87</td><td>57.88±18.23</td></tr></table>

On M&M, the two configurations without the RV term raise RV HD95 from 27.00 ± 8.62 mm to 41.70 ± 12.78 (Endo only) and 48.48 ± 15.88 (Epi only), and reduce RV DSC from 75.01 ± 18.54% to 38.85 ± 16.37% and 34.60 ± 15.68%; conversely, retaining only the RV term raises Endo HD95 to 31.88 ± 17.30 mm and lowers LV DSC to 48.92 ± 37.16%. Because no single-region configuration holds up across all regions, their joint incorporation is the most robust as the overall columns reflect on both datasets.

## V. DISCUSSION AND CONCLUSION

This study introduced CardioDrift, a fully automatic, atlasbased framework for representing subject-specific cardiac motion on a 3D atlas mesh with globally corresponding points, requiring only cine CMR SAX segmentation maps. Its core design generalises CPD by jointly registering the Endo, Epi, and RV atlas surfaces to their anatomically corresponding contours under a shared transformation, establishing correspondence within each region rather than across an undifferentiated point set. Moreover, it incorporates slice-shift correction during atlas fitting to the subject’s cardiac space, reducing inter-slice misalignment and improving atlas-fitting accuracy. Compared with state-of-the-art methods, CardioDrift represents subjectspecific cardiac motion on an atlas with globally corresponding points, without requiring manual atlas alignment in the subject’s cardiac space. Moreover, it avoids capturing irrelevant motion and the need for image-to-mesh interpolation, reducing attenuation of the derived motion because it operates entirely in the mesh/point-set domain and is based on cardiac contours.

CardioDrift outperformed all baselines on every metric of both datasets. In the rigid stage, the probabilistic formulation expresses correspondence as a soft assignment refined iteratively, suiting the sparse, noisy, and heterogeneous cardiac contour sets, precisely the conditions under which single-pass learning-based methods perform poorly. Embedding slice-shift correction in the same EM procedure reduces the inter-slice misalignment that would otherwise bias the alignment. These translations are estimated only at ED and held fixed thereafter, since the full cine sequence for a slice is acquired within a single breath-hold and the misalignment arises from diaphragm position varying between breath-holds, that is, between slices [24], [31]; it is an artefact of acquisition rather than cardiac phase. Unlike DeepMesh [18], the closest prior work, which requires semi-automatic initialisation and interpolates imagespace motion fields onto the mesh, CardioDrift automatically fits the atlas to the subject’s cardiac space using only the contour points, removing both manual intervention and the associated motion attenuation.

In the non-rigid stage, the largest gain was in Myo DSC, the most difficult region to track owing to its thin structure and large inward displacement at ES. The shape-aware mechanism is decisive: treating the three regions as distinct within a shared deformation field resolves correspondence anatomically rather than geometrically. The comparison with DeepMesh is the most direct of the three baselines, since it likewise propagates a corresponding atlas mesh but differs in how deformation is obtained: its motion fields are estimated from the image domain and interpolated onto the mesh space, and its DSC is 11.0 percentage points below CardioDrift for the LV and 12.2 for the RV, indicating that image-to-mesh motion interpolation degrades tracking.

TABLE II  
ABLATION STUDY OF THE SHAPE-AWARE MODELLING ACROSS ANATOMICAL REGIONS (RIGID STAGE), EVALUATED OVER UK BIOBANK AND M&M SUBJECTS. ✓ AND ✗ DENOTE WHETHER THE SHAPE-AWARE TERM IS ENABLED FOR THE CORRESPONDING REGION. ALL METRICS ARE REPORTED AS MEAN±STD. BOLD: BEST PER COLUMN WITHIN EACH DATASET.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Config</td><td rowspan="2">Endo</td><td rowspan="2">Epi</td><td rowspan="2">RV</td><td colspan="4">HD95 (mm)</td><td colspan="4">DSC(%)</td></tr><tr><td>Endo</td><td>Epi</td><td>RV</td><td>All Regions</td><td>LV</td><td>Myo</td><td>RV</td><td>All Regions</td></tr><tr><td rowspan="4">UK Biobank</td><td>Full</td><td>√</td><td>√</td><td>√</td><td>14.18±4.43</td><td> $1 6 . 9 8 { \pm } 5 . 0 6 $ </td><td>22.30±4.78</td><td>17.82±3.84</td><td>90.72±10.56</td><td> $7 0 . 2 5 { \pm } 1 2 . 2 7$ </td><td> $8 1 . 0 1 { \pm } 9 . 2 8 $ </td><td> $\mathbf { 8 0 . 6 6 } { \pm 9 . 8 4 }$ </td></tr><tr><td>Endo only</td><td>√</td><td>x</td><td>x</td><td>11.79±3.30</td><td>15.84±4.56</td><td>37.38±7.93</td><td>21.67±4.01</td><td>92.59±4.72</td><td>72.29±6.43</td><td> $3 5 . 0 3 { \pm } 1 0 . 1 8$ </td><td> $6 6 . 6 4 \pm 5 . 3 3$ </td></tr><tr><td>Epi only</td><td>x</td><td>√</td><td>x</td><td>15.67±4.98</td><td>16.33±4.71</td><td>39.37±9.56</td><td>23.79±5.29</td><td>87.54±13.43</td><td> $6 6 . 9 2 \pm 1 3 . 3 3$ </td><td> $3 3 . 5 1 { \pm } 1 0 . 0 6$ </td><td> $6 2 . 6 6 { \pm } 1 0 . 3 6$ </td></tr><tr><td>RV only</td><td>x</td><td>x</td><td>√</td><td>41.00±17.61</td><td>41.24±16.52</td><td>17.19±3.21</td><td>33.14±11.68</td><td>55.13±37.03</td><td>30.56±20.94</td><td>81.46±9.44</td><td> $5 5 . 7 2 { \scriptstyle \pm 2 1 . 8 6 }$ </td></tr><tr><td rowspan="4">M&amp;M</td><td>Full</td><td>√</td><td>√</td><td>√</td><td>17.23±6.98</td><td>22.15±7.29</td><td>27.00±8.62</td><td>22.13±6.65</td><td>83.82±20.34</td><td>60.21±19.10</td><td>75.01±18.54</td><td> ${ \bf 7 3 . 0 2 \pm 1 8 . 0 7 }$ </td></tr><tr><td>Endo only</td><td>√</td><td>x</td><td>x</td><td>15.85±6.06</td><td>21.31±6.82</td><td>41.70±12.78</td><td>26.28±7.42</td><td>84.92±19.04</td><td>58.76±18.02</td><td> $3 8 . 8 5 { \pm } 1 6 . 3 7$ </td><td> $6 0 . 8 4 \pm 1 4 . 9 0 $ </td></tr><tr><td>Epi only</td><td>x</td><td>√</td><td>x</td><td> $1 8 . 9 4 \pm 5 . 9 0 $ </td><td>23.98±7.50</td><td>48.48±15.88</td><td>30.47±8.47</td><td> $6 7 . 4 8 { \pm } 2 9 . 2 6 $ </td><td>47.07±23.44</td><td> $3 4 . 6 0 { \pm } 1 5 . 6 8$ </td><td> $4 9 . 7 1 \pm 2 0 . 7 9$ </td></tr><tr><td>RV only</td><td>x</td><td>x</td><td>√</td><td> $3 1 . 8 8 \pm 1 7 . 3 0 $ </td><td>35.00±16.32</td><td>21.08±3.97</td><td>29.32±11.55</td><td> $4 8 . 9 2 \pm 3 7 . 1 6$ </td><td>29.13±22.22</td><td>79.04±9.49</td><td> $5 2 . 3 6 { \pm } 2 2 . 1 2$ </td></tr></table>

![](images/45900b5cff7543111bbee98af718eca9831f11cc655baaa0979a45bf27547b2a.jpg)  
Fig. 2. Qualitative results. (a) Rigid atlas fitting for CardioDrift and the four baselines, with the mid-ventricular SAX image, ground-truth mask, and extracted contour points on the left; per method, the fitted atlas over the contour points and the rasterised mask. (b) Motion tracking across the cardiac cycle for CardioDrift, CPD, CPD-Net, and DeepMesh; per method, the propagated atlas and its rasterised mask, against the SAX image and ground-truth mask.

The M&M evaluation tests these mechanisms outside the standardised UK Biobank acquisition. Likewise UK Biobank, CardioDrift obtains the best rigid and non-rigid registration performance among the baselines, whereas the learned meth ods lose most, CPD-Net falling below CPD and DeepMesh, which additionally depends on vendor-dependent image intensities, showing the largest drop of any non-rigid method. Because correspondence is re-estimated per subject by EM rather than predicted from learned features, CardioDrift has no training distribution to depart from, and its behaviour is governed by the geometry of the contours presented to it; the regularisation coefficients, selected on UK Biobank and retained unchanged for M&M, support the same point. The one mechanism whose benefit diminishes is slice-shift correction, since a single in-plane translation per slice describes systematic breath-hold misalignment more completely than the heterogeneous residual misalignment of a multi-protocol cohort.

Several limitations motivate future work. The first is computational cost: (15) must be solved independently at each frame; replacing the iterative EM with a learned model is a natural remedy, though a hybrid coupling a network with a few EM iterations is more promising than a single forward pass, given the sparsity and heterogeneity of cardiac contours. Second, the method depends on the input segmentation, whose errors propagate directly into the contour points; jointly refining the contours from cine CMR image features, or modelling segmentation uncertainty end-to-end, would reduce this sensitivity. The third is the scope of the M&M evaluation, where reference masks exist only at ED and ES, so temporal consistency over the intermediate frames rests on UK Biobank only. Finally, the formulation includes no term controlling the temporal smoothness of the output mesh sequences. Addressing these would extend the accuracy and anatomical interpretability demonstrated here into a computationally efficient framework for large-scale inter-subject cardiac motion assessment.

![](images/35b2fe64f4d5ff4aad8c817b0f5b24387709c5ff029499f07f640fe4343c44b5.jpg)  
Fig. 3. Slice-shift correction. Atlas fitted (a) with and (b) without correction, over the subject’s contour points. Axial view of the original middle slice SAX image overlaid with the rasterised mask (c) with and (d) without slice shift correction. Coronal (e, f) and Sagittal (g, h) views of the original and corrected SAX images, with ground-truth masks overlaid.

## CODE AVAILABILITY

The source code is publicly available at GitHub.

## REFERENCES

[1] P. Claus, A. M. S. Omar, G. Pedrizzetti, P. P. Sengupta, and E. Nagel, “Tissue tracking technology for assessing cardiac mechanics: principles, normal values, and clinical applications,” JACC: Cardiovascular Imaging, vol. 8, no. 12, pp. 1444–1460, 2015.

[2] M. Salerno, B. Sharif, H. Arheden, A. Kumar, L. Axel, D. Li, and S. Neubauer, “Recent advances in cardiovascular magnetic resonance: techniques and applications,” Circulation: Cardiovascular Imaging, vol. 10, no. 6, p. e003951, 2017.

[3] S. E. Luijnenburg, D. Robbers-Visser, A. Moelker, H. W. Vliegen, B. J. Mulder, and W. A. Helbing, “Intra-observer and interobserver variability of biventricular function, volumes and mass in patients with congenital heart disease measured by cmr imaging,” The international journal of cardiovascular imaging, vol. 26, pp. 57–64, 2010.

[4] A. Myronenko and X. Song, “Point set registration: Coherent point drift,” IEEE transactions on pattern analysis and machine intelligence, vol. 32, no. 12, pp. 2262–2275, 2010.

[5] D. Rueckert, L. I. Sonoda, C. Hayes, D. L. Hill, M. O. Leach, and D. J. Hawkes, “Nonrigid registration using free-form deformations: application to breast mr images,” IEEE transactions on medical imaging, vol. 18, no. 8, pp. 712–721, 1999.

[6] M. De Craene, G. Piella, O. Camara, N. Duchateau, E. Silva, A. Doltra, J. D’hooge, J. Brugada, M. Sitges, and A. F. Frangi, “Temporal diffeomorphic free-form deformation: Application to motion and strain estimation from 3d echocardiography,” Medical Image Analysis, vol. 16, no. 2, pp. 427–450, 2012.

[7] J.-P. Thirion, “Image matching as a diffusion process: an analogy with maxwell’s demons,” Medical image analysis, vol. 2, no. 3, pp. 243–260, 1998.

[8] T. Vercauteren, X. Pennec, A. Perchant, and N. Ayache, “Non-parametric diffeomorphic image registration with the demons algorithm,” in International conference on medical image computing and computer-assisted intervention. Springer, 2007, pp. 319–326.

[9] G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, and A. V. Dalca, “Voxelmorph: a learning framework for deformable medical image registration,” IEEE transactions on medical imaging, vol. 38, no. 8, pp. 1788–1800, 2019.

[10] C. Qin, W. Bai, J. Schlemper, S. E. Petersen, S. K. Piechnik, S. Neubauer, and D. Rueckert, “Joint learning of motion estimation and segmentation for cardiac mr image sequences,” in Medical Image Computing and Computer Assisted Intervention–MICCAI 2018: 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part II 11. Springer, 2018, pp. 472–480.

[11] Q. Meng, C. Qin, W. Bai, T. Liu, A. De Marvao, D. P. O’Regan, and D. Rueckert, “Mulvimotion: Shape-aware 3d myocardial motion tracking from multi-view cardiac mri,” IEEE transactions on medical imaging, vol. 41, no. 8, pp. 1961–1974, 2022.

[12] A. Zakeri, A. Hokmabadi, N. Bi, I. Wijesinghe, M. G. Nix, S. E. Petersen, A. F. Frangi, Z. A. Taylor, and A. Gooya, “Dragnet: Learningbased deformable registration for realistic cardiac mr sequence generation from a single frame,” Medical Image Analysis, vol. 83, p. 102678, 2023.

[13] N. Bi, A. Zakeri, Y. Xia, N. Cheng, Z. A. Taylor, A. F. Frangi, and A. Gooya, “Segmorph: Concurrent motion estimation and segmentation for cardiac mri sequences,” IEEE transactions on medical imaging, 2024.

[14] R. A. Movahed, A. Rezaee, A. Zakeri, C. Berry, E. S. Ho, and A. Gooya, “Cardiomorphnet: Cardiac motion prediction using a shape-guided bayesian recurrent deep network,” Medical Image Analysis, vol. 113, p. 104149, 2026. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S1361841526002185

[15] E. Puyol-Anton, M. Sinclair, B. Gerber, M. S. Amzulescu, H. Langet, M. De Craene, P. Aljabar, P. Piro, and A. P. King, “A multimodal spatiotemporal cardiac motion atlas from mr and ultrasound data,” Medical image analysis, vol. 40, pp. 96–110, 2017.

[16] A. H. A. W. G. on Myocardial Segmentation, R. for Cardiac Imaging:, M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs, S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan et al., “Standardized myocardial segmentation and nomenclature for tomographic imaging of the heart: a statement for healthcare professionals from the cardiac imaging committee of the council on clinical cardiology of the american heart association,” Circulation, vol. 105, no. 4, pp. 539–542, 2002.

[17] A. Zakeri, A. Hokmabadi, N. Ravikumar, A. F. Frangi, and A. Gooya, “A probabilistic deep motion model for unsupervised cardiac shape anomaly assessment,” Medical Image Analysis, vol. 75, p. 102276, 2022.

[18] Q. Meng, W. Bai, D. P. O’Regan, and D. Rueckert, “Deepmesh: Meshbased cardiac motion tracking using deep learning,” IEEE transactions on medical imaging, vol. 43, no. 4, pp. 1489–1500, 2023.

[19] J. Duan, G. Bello, J. Schlemper, W. Bai, T. J. Dawes, C. Biffi, A. de Marvao, G. Doumoud, D. P. O’Regan, and D. Rueckert, “Automatic 3d biventricular segmentation of cardiac images by a shape-refined multi-task deep learning approach,” IEEE transactions on medical imaging, vol. 38, no. 9, pp. 2151–2164, 2019.

[20] G. Tarroni, W. Bai, O. Oktay, A. Schuh, H. Suzuki, B. Glocker, P. M. Matthews, and D. Rueckert, “Large-scale quality control of cardiac imaging in population studies: application to uk biobank,” Scientific reports, vol. 10, no. 1, p. 2408, 2020.

[21] X. Huang, G. Mei, J. Zhang, and R. Abbas, “A comprehensive survey on point cloud registration,” arXiv preprint arXiv:2103.02690, 2021.

[22] D. Bojanic, K. Bartol, J. Forest, T. Petkovi´ c, and T. Pribani´ c, “Addressing´ the generalization of 3d registration methods with a featureless baseline and an unbiased benchmark,” Machine vision and applications, vol. 35, no. 3, p. 41, 2024.

[23] A. Myronenko and X. Song, “On the closed-form solution of the rotation matrix arising in computer vision problems,” arXiv preprint arXiv:0904.1613, 2009.

[24] S. E. Petersen, P. M. Matthews, J. M. Francis, M. D. Robson, F. Zemrak, R. Boubertakh, A. A. Young, S. Hudson, P. Weale, S. Garratt et al., “Uk biobank’s cardiovascular magnetic resonance protocol,” Journal of cardiovascular magnetic resonance, vol. 18, no. 1, p. 8, 2015.

[25] W. Bai, M. Sinclair, G. Tarroni, O. Oktay, M. Rajchl, G. Vaillant, A. M. Lee, N. Aung, E. Lukaschuk, M. M. Sanghvi et al., “Automated cardiovascular magnetic resonance image analysis with fully convolutional networks,” Journal of cardiovascular magnetic resonance, vol. 20, no. 1, p. 65, 2018.

[26] V. M. Campello, P. Gkontra, C. Izquierdo, C. Mart´ın-Isla, A. Sojoudi, P. M. Full, K. Maier-Hein, Y. Zhang, Z. He, J. Ma et al., “Multicentre, multi-vendor and multi-disease cardiac segmentation: the M&Ms challenge,” IEEE Transactions on Medical Imaging, vol. 40, no. 12, pp. 3543–3554, 2021.

[27] W. Bai, W. Shi, A. de Marvao, T. J. Dawes, D. P. O’Regan, S. A. Cook, and D. Rueckert, “A bi-ventricular cardiac atlas built from 1000+ high resolution mr images of healthy subjects and an analysis of shape and motion,” Medical image analysis, vol. 26, no. 1, pp. 133–145, 2015.

[28] V. Sarode, X. Li, H. Goforth, Y. Aoki, R. A. Srivatsan, S. Lucey, and H. Choset, “Pcrnet: Point cloud registration network using pointnet encoding,” arXiv preprint arXiv:1908.07906, 2019.

[29] Y. Aoki, H. Goforth, R. A. Srivatsan, and S. Lucey, “Pointnetlk: Robust & efficient point cloud registration using pointnet,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 7163–7172.

[30] L. Wang, X. Li, J. Chen, and Y. Fang, “Coherent point drift networks: Unsupervised learning of non-rigid point set registration,” arXiv preprint arXiv:1906.03039, 2019.

[31] A. G. Chandler, R. J. Pinder, T. Netsch, J. A. Schnabel, D. J. Hawkes, D. L. Hill, and R. Razavi, “Correction of misaligned slices in multi-slice cardiovascular magnetic resonance using slice-to-volume registration,”

Journal of cardiovascular magnetic resonance, vol. 10, no. 1, p. 13, 2008.