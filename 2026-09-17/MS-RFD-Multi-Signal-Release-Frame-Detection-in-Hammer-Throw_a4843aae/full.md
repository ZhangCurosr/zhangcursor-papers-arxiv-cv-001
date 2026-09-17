# MS-RFD: Multi-Signal Release Frame Detection in Hammer Throw from Reconstructed 3D Trajectories

Ahmed Endris Hasen<sup>1,\*</sup>, Nikolaos Passalis<sup>2</sup>, Tomi Vänttinen<sup>3</sup>, Jenni Raitoharju<sup>1</sup>

<sup>1</sup>Faculty of Information Technology, University of Jyväskylä, Jyväskylä, Finland

<sup>2</sup> Faculty of Sciences, Aristotle University of Thessaloniki, Thessaloniki, Greece

<sup>3</sup>Finnish Institute of High Performance Sport KIHU, Jyväskylä, Finland

Corresponding author: ahmed.e.hasen@jyu.fi

Email: ahmed.e.hasen@jyu.fi, passalis@csd.auth.gr, tomi.vanttinen@kihu.fi, jenni.k.raitoharju@jyu.fi

Abstract—Recent advances in artificial intelligence and computer vision are reshaping sports performance analysis by enabling automated detection, tracking, and performance analysis. In hammer throw, performance is strongly determined by the kinematic conditions at release, particularly release speed, release angle, and release height. However, identifying the release instant from video typically requires manual frame-by-frame inspection, which is subjective and cumbersome in real-world training scenarios. In this paper, we present a fully automatic multisignal release frame detection (MS-RFD) method for hammer throw using reconstructed 3D hammer trajectories. The proposed method integrates four complementary kinematic signals: speed dynamics, angular velocity transition, radial distance relative to the rotation center, and post-release trajectory linearity. These signals are fused to score and verify candidate release frames. MS-RFD is evaluated through the throwing-distance estimation error obtained from the release parameters estimated at the detected frame. An ablation study analyzes the contribution of each signal and compares alternative candidate selection strategies. The results show that speed dynamics and radial expansion provide the strongest signals for release frame detection, while angular velocity and post-release linearity provide smaller refinements.

Index Terms—Hammer throw, release frame detection, sports biomechanics, 3D trajectory analysis.

## I. INTRODUCTION

Recent advances in computer vision and artificial intelligence are rapidly transforming sports performance analysis [1], enabling tasks such as player and ball detection [2], tracking and trajectory based analysis [3], and action or event detection [4]. In biomechanics-driven sports, such as hammer throw, performance analysis often depends on accurate estimation of release parameters, including release speed, release angle, and release height [5], [6]. These parameters define the subsequent ballistic motion and strongly influence the final throwing distance [7]. Previous biomechanical studies have examined release conditions and the radius of curvature during turns as important quantities in hammer throw analysis [7], [8].

Video-based analysis provides a practical way to estimate release parameters and predict throwing distance, particularly in training environments where direct measurement may be difficult. In our previous work [9], we introduced a videobased hammer throw distance estimation and trajectory analysis pipeline that combines synchronized multi-view video, DLT-based camera calibration, hammer-head detection, and stereo triangulation to reconstruct 3D trajectories from video inputs. Despite the promising results, automatic release frame identification remained a limiting component of the pipeline. However, identifying the release frame is a critical step for estimating release parameters, extracting meaningful biomechanical insights and enabling trajectory-based distance estimation from videos.

In our previous work [9], release frame identification was mainly performed manually through frame-by-frame video inspection, which is cumbersome and dependent on expert judgment. This limits scalability and introduces subjectivity into performance analysis workflows. A simple automatic baseline was also included, where the release frame was estimated from a weighted combination of speed and angular velocity computed from the reconstructed 3D trajectory. Although this provided an initial automatic solution, the accuracy was limited. The method relied only on two local kinematic cues and did not explicitly model the broader transition from rotational motion to ballistic flight. These limitations motivate the need for a more systematic and reliable automatic release frame detection method based on the reconstructed 3D trajectories.

In this paper, we focus on release frame detection and propose a multi-signal approach operating on reconstructed 3D hammer trajectories. The proposed multi-signal release frame detection (MS-RFD) method integrates multiple physicsinformed temporal signals, including speed dynamics, radial distance variation relative to the rotational center, angular velocity transition, and post-release trajectory linearity. These signals are combined within a unified scoring framework and followed by biomechanical candidate validation and finalframe selection. Unlike the previous method that relies on limited signal rules, the proposed method uses multiple complementary cues to better detect the transition from rotational motion to ballistic flight. The main contributions of this paper are as follows:

• We formulate hammer throw release frame identification from multi-view videos as an automatic release frame detection problem based on the reconstructed 3D trajectories.

• We propose MS-RFD, a multi-signal release frame detection method that combines speed dynamics, radial expansion, angular velocity transition, and post-release trajectory linearity.

• We conduct an ablation study to analyze the contribution of individual signals and to compare alternative candidate selection strategies for release frame detection.

• We evaluate the proposed method using the biomechanical consistency of the release parameters and throw distance estimation accuracy.

## II. METHOD

## A. Overview of the Proposed Method

The proposed method builds on the 3D trajectory reconstruction pipeline introduced in [9]. Given synchronized dualview video, the hammer head is detected in each camera view, cameras are calibrated using Direct Linear Transformation (DLT), and the 3D hammer positions are reconstructed through stereo triangulation and the resulting trajectory is smoothed before release analysis to reduce noise and improve temporal consistency.

Let

$$
\mathcal { T } = \{ \mathbf { p } _ { 1 } , \mathbf { p } _ { 2 } , \ldots \{  , \mathbf { p } _ { n } \}
$$

denote the reconstructed 3D hammer trajectory, where $\mathbf { p } _ { i } =$ $( x _ { i } , y _ { i } , z _ { i } ) \ \in \ \mathbb { R } ^ { 3 }$ represents the estimated hammer position at frame i. The objective is to estimate the release frame $r ^ { * }$ , corresponding to the transition from rotational motion to ballistic flight. This release-frame detection problem can be written generally as

$$
r ^ { * } = F ( { \mathcal { T } } ) ,
$$

where $F$ is an automatic release-detection function that maps the reconstructed trajectory to a release-frame index.

Fig. 1 illustrates the proposed MS-RFD framework. In this work, F is implemented through three main stages: candidate search-window selection, multi-signal release scoring, and candidate validation/selection. The method first selects a candidate search window R from the final rotational phase of the trajectory. Since release is expected to occur during the last complete rotational cycle, this window is estimated from the radial-distance behavior of the reconstructed trajectory and placed around the final circular-motion region. Candidate frames inside R are then evaluated using four normalized kinematic signals: speed dynamics, angular transition, radial expansion, and post-release linearity. These signals are fused into a multi-signal release score $S ( r )$ , which is used to rank candidate frames within the search window as described in Section II-B. Candidate frames are then validated using biomechanical release-parameter constraints, and the final release frame is selected from the validated candidate set using one of the selection strategies described in Section II-C.

## B. Multi-Signal Release Scoring

The release event is characterized by a transition from circular motion to ballistic motion. To capture this, four complementary kinematic cues are extracted from the reconstructed trajectory: speed dynamics, angular transition, radial expansion, and post-release linearity. The method evaluates multiple candidate frames r within the search window $\mathcal { R } .$

For each signal, unnormalized score $g _ { j } ( r )$ is first computed from the trajectory for each candidate frame r. The final signal score $\phi _ { j } ( \boldsymbol { r } )$ is then obtained by normalizing the signal values within the search window:

$$
\phi _ { j } ( \boldsymbol { r } ) = \mathcal { N } _ { \mathcal { R } } ( g _ { j } ( \boldsymbol { r } ) ) ,
$$

where $\mathcal { N } _ { \mathcal { R } } ( \cdot )$ rescales the values in R to $[ 0 , 1 ] .$ . Thus, larger $\phi _ { j } ( \boldsymbol { r } )$ values indicate stronger release evidence.

1) Speed Dynamics $( \phi _ { 1 } ) .$ : Velocity is computed using finite differences. A central-difference is used to obtain a smoother estimate:

$$
\mathbf { v } _ { i } = \frac { \mathbf { p } _ { i + 1 } - \mathbf { p } _ { i - 1 } } { 2 \Delta t } , \quad \Delta t = \frac { 1 } { f _ { \mathrm { f p s } } } .\tag{1}
$$

The speed magnitude is $s _ { i } = \| \mathbf { v } _ { i } \|$ . Hammer speed typically increases through the turns and reaches a high value near release [10], [11], [12]. The speed-dynamics signal therefore captures positive increases in smoothed hammer speed near release while reducing sensitivity to short-term noise and fluctuations in the reconstructed trajectory. The speed signal is defined from the positive temporal increase of the smoothed speed:

$$
g _ { 1 } ( r ) = \operatorname* { m a x } \left( 0 , \frac { d \tilde { s } _ { r } } { d t } \right) , \qquad \phi _ { 1 } ( r ) = \mathcal { N } _ { \mathcal { R } } ( g _ { 1 } ( r ) ) ,
$$

where $\tilde { s } _ { r }$ is the smoothed speed at candidate frame r.

2) Angular Transition $( \phi _ { 2 } ) \colon$ : The transition from rotational to more linear motion can also be characterized through changes in the direction of velocity. The angular change between consecutive velocity vectors is

$$
\theta _ { i } = \operatorname { a r c c o s } \left( \frac { \mathbf { v } _ { i - 1 } \cdot \mathbf { v } _ { i } } { \| \mathbf { v } _ { i - 1 } \| \| \mathbf { v } _ { i } \| } \right)\tag{2}
$$

During rotation, the velocity direction changes rapidly. After release, the horizontal trajectory becomes more linear, and directional change decreases. This signal evaluates the transition comparing angular behavior before and after each candidate frame:

$$
g _ { 2 } ( r ) = \mathrm { m a x } \left( 0 , \bar { \theta } _ { r } ^ { \mathrm { p r e } } - \bar { \theta } _ { r } ^ { \mathrm { p o s t } } \right) , \qquad \phi _ { 2 } ( r ) = \mathcal { N } _ { \mathcal { R } } ( g _ { 2 } ( r ) ) ,
$$

where $\bar { \theta } _ { r } ^ { \mathrm { p r e } }$ and $\bar { \theta } _ { r } ^ { \mathrm { p o s t } }$ denote the mean angular change before and after $r ,$ respectively.

3) Radial Distance Expansion $( \phi _ { 3 } ) \colon$ During the turns, the hammer follows an approximately circular path in the horizontal plane. The rotation center c is estimated from stable mid-trajectory points in the horizontal xy plane using a robust RANSAC [13] based approximation. The radial distance is defined as

$$
\rho _ { i } = \| \mathbf { p } _ { i } ^ { x y } - \mathbf { c } \| ,\tag{3}
$$

where $\mathbf { p } _ { i } ^ { x y } \ : = \ : ( x _ { i } , y _ { i } )$ denotes the horizontal projection. At release, the hammer departs from the circular path, resulting in sustained radial expansion. The radial-expansion signal

![](images/88f0095d208e83f36d848f377fc234c5123d3f7348308e9eceb55120aa8b6ee6.jpg)  
Fig. 1. Overview of the proposed MS-RFD framework for automatic release-frame detection from reconstructed 3D hammer trajectories.

measures positive consistent outward radial progression over a future window:

$$
\begin{array} { r } { g _ { 3 } ( r ) = \operatorname* { m a x } \left( 0 , \rho _ { r + L } - \rho _ { r } \right) , \qquad \phi _ { 3 } ( r ) = \mathcal { N } _ { \mathcal { R } } ( g _ { 3 } ( r ) ) , } \end{array}
$$

where L is the future window length.

4) Post-Release Linearity $( \phi _ { 4 } ) \mathrm { : }$ After release, the hammer follows projectile motion, and its trajectory becomes locally linear over a short interval. For each candidate frame r, a line is fitted to a short future segment of the 3D trajectory from $r$ to $r + L ,$ and linearity is evaluated using a coefficient of determination:

$$
R _ { r } ^ { 2 } = 1 - \frac { \sum _ { k = r } ^ { r + L } \| \mathbf { q } _ { k } - \hat { \mathbf { q } } _ { k } \| ^ { 2 } } { \sum _ { k = r } ^ { r + L } \| \mathbf { q } _ { k } - \bar { \mathbf { q } } \| ^ { 2 } } ,\tag{4}
$$

where k indexes the frames in the future segment, $\mathbf { q } _ { k }$ denotes the 3D trajectory point at frame $k , { \hat { \mathbf { q } } } _ { k }$ is its projection onto the fitted 3D line, and $\bar { \bf q }$ is the segment mean. Higher values of $R _ { r } ^ { 2 }$ indicate stronger linearity and therefore stronger consistency with post-release ballistic motion. The linearity signal is defined as

$$
g _ { 4 } ( r ) = R _ { r } ^ { 2 } , \qquad \phi _ { 4 } ( r ) = \mathcal { N } _ { \mathcal { R } } \bigl ( g _ { 4 } ( r ) \bigr ) .
$$

5) Score Fusion: The multi-signal release score for each candidate frame is computed by combining the normalized signals:

$$
S ( r ) = \sum _ { j = 1 } ^ { 4 } w _ { j } \phi _ { j } ( r ) , \quad \sum _ { j = 1 } ^ { 4 } w _ { j } = 1 ,\tag{5}
$$

where $S ( r )$ is the fused release score for candidate frame r and $w _ { j }$ is the weight for signal score $\phi _ { j } ( \boldsymbol { r } )$ . The fused score $S ( r )$ is used to rank candidate frames within the search window ${ \mathcal { R } } ,$ with higher values indicating stronger release likelihood. The weights are specified as $w _ { 1 } ~ = ~ 0 . 4 0 , ~ w _ { 2 } ~ = ~ 0 . 2 0$ $w _ { 3 } = 0 . 3 0$ , and $w _ { 4 } = 0 . 1 0$ , corresponding to speed dynamics, angular transition, radial expansion, and post-release linearity, respectively. The weights were empirically selected based on the relevance of each cue and analyzed through ablation experiments.

## C. Candidate Selection and Validation

After using the multi-signal score $S ( r )$ to rank candidate frames within the search window R, the frames with the highest release scores are retained as candidate release frames and are filtered using biomechanical constraints. To ensure biomechanical consistency and to reduce false detections, candidate frames are retained only if their estimated release speed, release angle, and release height fall within realistic hammer throw ranges:

$$
\begin{array} { r } { ( s _ { \operatorname* { m i n } } \leq s _ { r } \leq s _ { \operatorname* { m a x } } ) \wedge ( \alpha _ { \operatorname* { m i n } } \leq \alpha _ { r } \leq \alpha _ { \operatorname* { m a x } } ) } \\ { \wedge ( h _ { \operatorname* { m i n } } \leq h _ { r } \leq h _ { \operatorname* { m a x } } ) , } \end{array}\tag{6}
$$

where $s _ { r } , \alpha _ { r } ,$ and $h _ { r }$ denote release speed, angle, and height at candidate frame $^ { r , }$ respectively. The parameter bounds are chosen according to plausible hammer throw release conditions reported in biomechanics literature [14], [15], [5]. If no valid candidate remains, the search window is expanded and the validation step is repeated. Let ${ \mathcal { C } } \subseteq { \mathcal { R } }$ denote the validated candidate-frame set. Thus, frames in $\mathcal { C }$ are selected based on the multi-signal score $S ( r )$ and satisfy the biomechanical plausibility constraints. The final release estimate is then obtained from C using one of the selection strategies described below.

1) Weighted Mean Aggregation $( S I ) .$ : Each candidate frame $r \in { \mathcal { C } }$ is assigned a weight $\lambda _ { r }$ based on its speed proximity to the robust candidate speed center $\mu _ { s } \colon$

$$
\lambda _ { r } = \exp \left( - \frac { ( s _ { r } - \mu _ { s } ) ^ { 2 } } { 2 \sigma _ { s } ^ { 2 } } \right) .\tag{7}
$$

where $s _ { r }$ denotes the speed at candidate frame r, while $\mu _ { s }$ and $\sigma _ { s }$ are computed from the speeds of the validated candidates in C. The weighted mean frame is first computed, and then final

release frame is selected as the validated candidate closest to this weighted mean:

$$
r ^ { * } = \arg \operatorname* { m i n } _ { r \in \mathcal { C } } \left| r - \frac { \sum _ { u \in \mathcal { C } } \lambda _ { u } u } { \sum _ { u \in \mathcal { C } } \lambda _ { u } } \right| ,
$$

2) Representative Frame Selection (S2): The representative-frame strategy selects the candidate whose speed is closest to the median candidate speed:

$$
r ^ { * } = \underset { r \in \mathcal { C } } { \arg \operatorname* { m i n } } \left| s _ { r } - \mathrm { m e d i a n } ( \{ s _ { i } \} _ { i \in \mathcal { C } } ) \right|\tag{8}
$$

This approach selects a physically existing frame while reducing sensitivity to extreme candidate values.

3) Multi-Criteria Selection (S3): The multi-criteria strategy evaluates each validated candidate using release speed, release angle, and release height. Let $k \in \{ s , \alpha , h \}$ index these release parameters, with $p _ { s , r } = s _ { r } , p _ { \alpha , r } = \alpha _ { r } ,$ and $p _ { h , r } = h _ { r }$ . The multi-criteria score $M ( r )$ for candidate frame r is defined as:

$$
M ( r ) = \sum _ { k \in \{ s , \alpha , h \} } \beta _ { k } \exp \left( - \frac { ( p _ { k , r } - \tilde { p } _ { k } ) ^ { 2 } } { 2 \sigma _ { k } ^ { 2 } } \right) ,\tag{9}
$$

where $\tilde { p } _ { k }$ is the median of parameter $k$ across the validated candidates, $\sigma _ { k }$ is the corresponding scale estimate, and $\beta _ { k }$ controls its relative importance. In this work, the weights for speed, angle, and height are set to 0.50, 0.30, and 0.20, respectively. The final release frame is then selected as the validated candidate with the highest multi-criteria score:

$$
r ^ { * } = \arg \operatorname* { m a x } _ { r \in { \mathcal { C } } } M ( r ) .
$$

## III. EXPERIMENTS AND RESULTS ANALYSIS

## A. Dataset and Experimental Setup

The experimental evaluation was conducted with the same dataset used in [9], which was collected by the Finnish Institute of High Performance Sport (KIHU) between 2017 and 2024. The recordings were captured using a dual-camera setup (side and back views) at frame rates of 240 fps and 180 fps with a resolution of $1 9 2 0 \times 1 0 8 0$ . The reconstructed 3D trajectories used in this work are obtained using the pipeline introduced in [9].

The proposed release frame detection method is evaluated through the performance and biomechanical consistency of release parameters and distance estimation accuracy. Specifically, each detected release frame is used to estimate the release parameters, including: release speed (m/s), release angle (degrees), and release height (m). These parameters are then used to compute the predicted throw distance using a physics-based trajectory model used in [9]. The predicted distance $d _ { n } ^ { \mathrm { p r e d } }$ is compared against the ground-truth measured distance $d _ { n } ^ { \mathrm { G T } }$ to compute the error. The evaluation metrics we used are mean absolute error (MAE), median absolute error (MedAE), and mean absolute percentage error (MAPE):

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left| d _ { n } ^ { \mathrm { p r e d } } - d _ { n } ^ { \mathrm { G T } } \right| ,\tag{10}
$$

$$
\begin{array} { r } { \mathrm { { M e d A E } } = \mathrm { { m e d i a n } } \left( \left| d _ { n } ^ { \mathrm { { p r e d } } } - d _ { n } ^ { \mathrm { { G T } } } \right| \right) , } \end{array}\tag{11}
$$

TABLE I  
PERFORMANCE COMPARISON ACROSS SIGNAL CONFIGURATIONS AND CANDIDATE SELECTION STRATEGIES.
<table><tr><td>Conf.</td><td>Strategy</td><td>MAE (m)</td><td>MedAE (m)</td><td>MAPE (%)</td></tr><tr><td>C1</td><td>S1</td><td>5.16</td><td>4.47</td><td>7.31</td></tr><tr><td>Cl</td><td>S2</td><td>5.19</td><td>4.45</td><td>7.34</td></tr><tr><td>Cl</td><td>S3</td><td>5.13</td><td>4.44</td><td>7.26</td></tr><tr><td>C2</td><td>S1</td><td>5.11</td><td>4.41</td><td>7.24</td></tr><tr><td>C2</td><td>S2</td><td>5.13</td><td>4.44</td><td>7.27</td></tr><tr><td>C2</td><td>S3</td><td>4.72</td><td>4.34</td><td>6.69</td></tr><tr><td>C3</td><td>S1</td><td>4.49</td><td>4.00</td><td>6.37</td></tr><tr><td>C3</td><td>S2</td><td>4.39</td><td>4.42</td><td>6.21</td></tr><tr><td>C3</td><td>S3</td><td>4.73</td><td>3.75</td><td>6.70</td></tr><tr><td>C4</td><td>S1</td><td>4.37</td><td>4.06</td><td>6.19</td></tr><tr><td>C4</td><td>S2</td><td>4.38</td><td>4.42</td><td>6.20</td></tr><tr><td>C4</td><td>S3</td><td>4.71</td><td>3.75</td><td>6.67</td></tr></table>

$$
\mathrm { M A P E } = \frac { 1 0 0 } { N } \sum _ { n = 1 } ^ { N } \left| \frac { d _ { n } ^ { \mathrm { p r e d } } - d _ { n } ^ { \mathrm { G T } } } { d _ { n } ^ { \mathrm { G T } } } \right| ,\tag{12}
$$

where n indexes the evaluated throws, and N is the total number of throws.

In addition, we perform a comprehensive ablation analysis to assess the contribution of each signal using the following signal combinations: $\mathrm { C 1 } = \phi _ { 1 } , \mathrm { C 2 } = \mathrm { C 1 } + \phi _ { 2 } , \mathrm { C 3 } = \mathrm { C 2 } + \phi _ { 3 } ,$ and ${ \bf C } 4 = { \bf C } 3 + \phi _ { 4 }$ . Three candidate selection strategies S1-S3 were evaluated for each configuration. This enables a deeper understanding of which cues are most informative for practical automatic release frame detection and how they influence the overall system performance.

## B. Overall Performance Analysis

The quantitative results for different variants of the proposed MS-RFD framework are summarized in Table I. The best result is obtained by the full MS-RFD configuration using weighted aggregation (C4-S1), which achieves an MAE of 4.37 m, a MedAE of 4.06 m, and a MAPE of 6.19%. The representative-frame based selection strategy under the same full configuration (C4-S2) gives nearly the same performance, with an MAE of 4.38 m and a MAPE of 6.20%. The ablation study shows a clear improvement as additional release cues are introduced. The speed-only configuration C1 provides a strong initial baseline, and adding angular transition information in C2 improves the candidate ranking by incorporating directional-change evidence. A more substantial improvement is observed, when radial expansion is introduced in C3, indicating that outward motion is a particularly important cue for separating true release from continued rotation. The final C4 configuration adds post-release linearity as an additional consistency cue. Although the gain from C3 to C4 is smaller, this cue improves the physical interpretation of the selected release frame by favoring candidates followed by a locally linear trajectory segment.

Fig. 2 illustrates the MAE comparison across the four signal configurations and shows the overall reduction in error from the speed-only setting toward the full C4 configuration. The figure also shows a comparison against the previous manual and automatic baselines from [9]. Compared with the previous automatic baseline, which obtains an MAE of 7.46 m, C4- S1 reduces the error by approximately 41.4%. The manual baseline still provides the lowest error, as expected, but the proposed method closes a substantial part of the gap between manual release frame selection and automatic detection.

![](images/1f3e9e2b87f5bad3a28c9150089a8e4689650a95dbb728a55edd2affa4ac2775.jpg)

Fig. 2. Mean absolute error (MAE) comparison across the four signal configurations and three candidate selection strategies. The bars represent the average error for each strategy, while the vertical error bars indicate the perthrow error variability. The dashed and dotted horizontal lines show the MAE of the previous manual and automatic baselines, respectively.  
![](images/5914f04a547426c59d5de43028adbbc264e3a9ff00aa4ebfb5142fd04cafe510.jpg)  
Fig. 3. Distribution of absolute distance errors for the manual baseline, previous automatic baseline, and the three C4 selection techniques. The boxplots show the median, interquartile range, mean, and outliers.

## C. Per-Throw Error Analysis

Table II presents the detailed per-throw predictions for the full C4 configuration, together with the manual and automatic baselines. The manual baseline achieves the lowest MAE of 2.95 m, which is expected because it benefits from human interpretation of the release event. Among automatic methods, the proposed C4 variants substantially outperform the previous automatic baseline. The automatic baseline obtains an MAE of 7.46 m, while C4-S1 and C4-S2 reduce the automatic error to approximately 4.38 m. Fig. 3 shows the distribution of absolute errors for the manual baseline, previous automatic baseline, and C4, highlighting that C4 reduces the error spread compared with the previous automatic method.

Several throws are estimated with high accuracy. For example, IDs 12, 13, 22, 33, 36, and 37 achieve low errors. These cases indicate that MS-RFD is effective when the reconstructed trajectory contains a clear rotational-to-ballistic transition. The largest remaining errors occur for IDs 31, 25, 24, 21, and 11. These throws dominate the MAE and suggest that the main limitation is not the average behavior of the detector, but a small number of difficult trajectories where the release transition is either noisy, temporally ambiguous, or affected by reconstruction uncertainty.

![](images/4b8ebfbe2cd9a1c60a277c5535a18e13eea650d8691ff989feed5b4949917244.jpg)  
Fig. 4. Reconstructed 3D hammer trajectory showing the estimated release point corresponding to the detected release frame.

The comparison against the manual baseline is also informative. The proposed method does not yet match manual selection, but it closes a large part of the gap between previous automatic release detection and manual method.

## D. Qualitative Analysis

Fig. 4 shows an example of a reconstructed 3D hammer trajectory and the estimated release point corresponding to the detected release frame, highlighting the transition from rotational motion to linearity using our MS-RFD method, where speed, angular change, radial expansion, and future trajectory linearity are combined to identify a physically plausible release frame.

## IV. CONCLUSION

This paper presented MS-RFD, a multi-signal framework for automatic release-frame detection in hammer throw using reconstructed trajectories. The method combines speed dynamics, angular transition, radial expansion, and post-release linearity to identify physically plausible release frames. The results show that explicitly modeling hammer-throw mechanics improves automatic release detection. The ablation results further show that each signal contributes complementary information, with the strongest performance obtained when all cues are combined. The full C4 configuration achieved the best automatic performance, reducing MAE from 7.46m for the previous automatic baseline to 4.37 m. Although the proposed method does not yet match manual release-frame selection, it substantially closes the gap between previous automatic detection and human-guided analysis. The current study has some limitations. First, MS-RFD depends on the quality of the reconstructed 3D trajectory, so errors from detection, calibration, triangulation, or smoothing can affect the trajectory near release and influence the estimated release parameters and predicted distance. In addition, the method uses fixed fusion weights and temporal windows, which were kept constant across experiments rather than optimized for individual athletes or recording setups. Furthermore, the evaluation is mainly based on downstream throwing-distance error rather than direct frame-level release annotations.

TABLE II  
PER-THROW COMPARISON OF PREDICTED THROWING DISTANCES AND ABSOLUTE ERRORS FOR THE MANUAL BASELINE, PREVIOUS AUTOMATIC BASELINE, AND THE PROPOSED C4 SELECTION STRATEGIES.
<table><tr><td rowspan="2">ID</td><td rowspan="2">GT</td><td colspan="2">Manual Ref.</td><td colspan="2">Automatic Ref.</td><td colspan="2">C4-S1</td><td colspan="2">C4-S2</td><td colspan="2">C4-S3</td></tr><tr><td>Pred.</td><td>Error</td><td>Pred.</td><td>Error</td><td>Pred.</td><td>Error</td><td>Pred.</td><td>Error</td><td>Pred.</td><td>Error</td></tr><tr><td>11</td><td>70.26</td><td>73.22</td><td>2.96</td><td>72.38</td><td>2.12</td><td>77.10</td><td>6.84</td><td>76.70</td><td>6.44</td><td>76.70</td><td>6.44</td></tr><tr><td>12</td><td>73.51</td><td>74.49</td><td>0.98</td><td>76.79</td><td>3.28</td><td>72.85</td><td>0.66</td><td>74.52</td><td>1.01</td><td>71.39</td><td>2.12</td></tr><tr><td>13</td><td>68.25</td><td>69.87</td><td>1.62</td><td>82.22</td><td>13.97</td><td>69.14</td><td>0.89</td><td>69.89</td><td>1.64</td><td>68.93</td><td>0.68</td></tr><tr><td>21</td><td>70.19</td><td>79.66</td><td>9.47</td><td>87.89</td><td>17.70</td><td>79.27</td><td>9.08</td><td>79.75</td><td>9.56</td><td>74.53</td><td>4.34</td></tr><tr><td>22</td><td>70.68</td><td>69.51</td><td>1.17</td><td>62.08</td><td>8.60</td><td>71.43</td><td>0.75</td><td>75.10</td><td>4.42</td><td>67.22</td><td>3.46</td></tr><tr><td>23</td><td>71.03</td><td>71.34</td><td>0.31</td><td>73.21</td><td>2.18</td><td>66.97</td><td>4.06</td><td>74.27</td><td>3.24</td><td>62.21</td><td>8.82</td></tr><tr><td>24</td><td>69.89</td><td>66.18</td><td>3.71</td><td>74.64</td><td>4.75</td><td>77.19</td><td>7.30</td><td>78.08</td><td>8.19</td><td>75.97</td><td>6.08</td></tr><tr><td>25</td><td>70.36</td><td>60.12</td><td>10.24</td><td>45.62</td><td>24.74</td><td>61.57</td><td>8.79</td><td>64.17</td><td>6.19</td><td>56.05</td><td>14.31</td></tr><tr><td>26</td><td>71.07</td><td>73.30</td><td>2.23</td><td>65.83</td><td>5.24</td><td>75.67</td><td>4.60</td><td>76.25</td><td>5.18</td><td>76.01</td><td>4.94</td></tr><tr><td>27</td><td>72.76</td><td>73.09</td><td>0.33</td><td>79.63</td><td>6.87</td><td>78.94</td><td>6.18</td><td>79.64</td><td>6.88</td><td>78.33</td><td>5.57</td></tr><tr><td>31</td><td>70.07</td><td>75.09</td><td>5.02</td><td>77.72</td><td>7.65</td><td>81.17</td><td>11.10</td><td>80.78</td><td>10.71</td><td>79.70</td><td>9.63</td></tr><tr><td>32</td><td>70.17</td><td>70.37</td><td>0.20</td><td>58.48</td><td>11.69</td><td>73.50</td><td>3.33</td><td>73.67</td><td>3.50</td><td>73.74</td><td>3.57</td></tr><tr><td>33</td><td>72.56</td><td>68.49</td><td>4.07</td><td>81.06</td><td>8.50</td><td>73.61</td><td>1.05</td><td>72.19</td><td>0.37</td><td>72.19</td><td>0.37</td></tr><tr><td>34</td><td>71.32</td><td>70.05</td><td>1.27</td><td>74.29</td><td>2.97</td><td>68.63</td><td>2.69</td><td>70.06</td><td>1.26</td><td>70.06</td><td>1.26</td></tr><tr><td>35</td><td>73.82</td><td>73.43</td><td>0.39</td><td>69.48</td><td>4.34</td><td>78.16</td><td>4.34</td><td>78.80</td><td>4.98</td><td>77.57</td><td>3.75</td></tr><tr><td>36</td><td>65.28</td><td>68.70</td><td>3.42</td><td>71.16</td><td>5.88</td><td>64.83</td><td>0.45</td><td>66.12</td><td>0.84</td><td>63.66</td><td>1.62</td></tr><tr><td>37</td><td>69.36</td><td>66.68</td><td>2.68</td><td>83.22</td><td>13.86</td><td>67.09</td><td>2.27</td><td>69.26</td><td>0.10</td><td>66.19</td><td>3.17</td></tr><tr><td>MAE</td><td>一</td><td>一</td><td>2.95</td><td></td><td>7.46</td><td></td><td>4.37</td><td></td><td>4.38</td><td></td><td>4.71</td></tr></table>

## ACKNOWLEDGMENT

This work was part of Finland’s Ministry of Education and Culture’s Doctoral Education Pilot under Decision No. VN/3137/2024-OKM-6 (The Finnish Doctoral Program Network in Artificial Intelligence, AI-DOC).

## REFERENCES

[1] B. T. Naik, M. F. Hashmi, and N. D. Bokde, “A comprehensive review of computer vision in sports: Open issues, future trends and research directions,” Applied Sciences, vol. 12, no. 9, p. 4429, 2022.

[2] M. Buric, M. Pobar, and M. Ivaši´ c-Kos, “Adapting yolo network for ball´ and player detection,” in Proceedings of the 8th International Conference on Pattern Recognition Applications and Methods, vol. 1, 2019, pp. 845– 851.

[3] L. Torres-Ronda, E. Beanland, S. Whitehead, A. Sweeting, and J. Clubb, “Tracking systems in team sports: a narrative review of applications of the data and sport specific analysis,” Sports medicine-open, vol. 8, no. 1, p. 15, 2022.

[4] H. Xu, A. A. Baniya, S. Well, M. R. Bouadjenek, R. Dazeley, and S. Aryal, “Deep learning for sports video event detection: Tasks, datasets, methods, and challenges,” arXiv preprint arXiv:2505.03991, 2025.

[5] G. M. Castaldi, R. Borzuola, V. Camomilla, E. Bergamini, G. Vannozzi, and A. Macaluso, “Biomechanics of the hammer throw: narrative review,” Frontiers in Sports and Active Living, vol. 4, p. 853536, 2022.

[6] Y. Ding, W. Liu, and J. Li, “A kinematic analysis of the hammer throw technique in elite female athletes: a comparative study between chinese and international competitors,” Frontiers in Physiology, vol. 16, p. 1590350, 2025.

[7] J. Dapena, M. Gutiérrez-Dávila, V. M. Soto, and F. J. Rojas, “Prediction of distance in hammer throwing,” Journal of Sports Sciences, vol. 21, no. 1, pp. 21–28, 2003.

[8] K. Murofushi, S. Sakurai, K. Umegaki, and K. Kobayashi, “Development of a system to measure radius of curvature and speed of hammer head during turns in hammer throw,” International Journal of Sport and Health Science, vol. 3, pp. 116–128, 2005.

[9] A. E. Hasen, N. Passalis, T. Vanttinen, and J. Raitoharju, “Hammer throw distance estimation using deep learning and physics-based modeling,” Expert Systems with Applications, p. 132022, 2026.

[10] K. Murofushi, S. Sakurai, K. Umegaki, and J. Takamatsu, “Hammer acceleration due to thrower and hammer movement patterns,” Sports biomechanics, vol. 6, no. 3, pp. 301–314, 2007.

[11] M. Bandou, S. Tanabe, and A. Ito, “Relationship between hammer throw performance and hammer head velocity,” Japan J. Phys. Educ. Hlth. Sport Sci, vol. 46, pp. 505–514, 2006.

[12] S. M. Brice, K. F. Ness, and D. Rosemond, “An analysis of the relationship between the linear hammer speed and the thrower applied forces during the hammer throw for male and female throwers,” Sports biomechanics, vol. 10, no. 3, pp. 174–184, 2011.

[13] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381–395, 1981.

[14] R. Pavlovic, “Biomechanical analysis hammer throw: The influence of´ kinematic parameters on the results of finalists world championships,” American Journal of Sports Science and Medicine, vol. 8, no. 2, pp. 36–46, 2020.

[15] R. Isele and E. Nixdorf, “Biomechanical analysis of the hammer throw at the 2009 iaaf world championships in athletics,” New Studies in Athletics, vol. 25, no. 3/4, pp. 37–60, 2010.