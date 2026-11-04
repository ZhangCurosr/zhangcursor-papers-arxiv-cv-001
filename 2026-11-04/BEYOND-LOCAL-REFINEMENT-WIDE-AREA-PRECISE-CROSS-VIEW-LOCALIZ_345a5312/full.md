# BEYOND LOCAL REFINEMENT: WIDE-AREA PRECISE CROSS-VIEW LOCALIZATION

Qingguo Hu<sup>1</sup>, Xinyu Ma<sup>3</sup>, Yujie Lin<sup>1</sup>, Yaoxiang Wang<sup>1</sup>, Qi Shi<sup>2</sup>, Zonghao Guo<sup>2</sup>, Chi Chen<sup>2,∗</sup>, Jinsong Su<sup>1,∗</sup>

<sup>1</sup>Xiamen University <sup>2</sup>Tsinghua University <sup>3</sup>University of Macau

huqingguo@stu.xmu.edu.cn jssu@xmu.edu.cn

Project Page: https://qghuxmu.github.io/RingMatch/

## ABSTRACT

Fine-grained cross-view localization aims to precisely localize a ground camera against geo-referenced aerial imagery, but current methods typically assume a narrow geographic prior, substantially limiting their applicability when upstream localization remains coarse. To bridge this gap, we study this task over much larger aerial regions. However, this seemingly simple extension fundamentally changes the localization problem: the growing disparity in spatial extent between ground and aerial views makes reliable cross-view matching difficult to establish directly, while the increasing number of look-alike locations in larger regions introduces substantial visual ambiguity. To address these challenges, we propose RingMatch, a search-and-verification framework for wide-area cross-view localization. Ring-Match first leverages predictive uncertainty to uncover plausible locations and progressively zooms into their surroundings to obtain increasingly fine-grained evidence, yielding a compact set of promising candidates. We then introduce the depth ring, a novel geometric representation that reveals subtle variations in the spatial layout around a location and serves as a view-shared spatial signature for an LVLM verifier to resolve residual ambiguity. To benchmark this setting, we construct VIGOR-Pro by composing VIGOR aerial tiles into contiguous regions with spatial extents of up to 400 m. Experiments show that RingMatch maintains strong localization performance as the aerial region expands, consistently outperforming representative end-to-end and retrieval-based methods.

## 1 INTRODUCTION

Determining where an image was captured from visual observations is a fundamental problem with broad applications in navigation, robotics, and public safety. At the global scale, image geolocalization provides a coarse location estimate for a ground image from geographic cues in the scene (Vivanco Cepeda et al., 2023; Li et al., 2025). To achieve finer localization, aerial imagery has been widely used as a reference owing to its broad coverage and rich scene details (Deuser et al., 2023). Within this paradigm, ongoing research has increasingly focused on the final stage of the overall pipeline, known as fine-grained cross-view localization (Zhu et al., 2021), which seeks to estimate the ground camera’s precise metric position within a geo-referenced aerial image. By leveraging advances in cross-view representation learning (Xia et al., 2022; Fervers et al., 2023; Xia et al., 2024) and fine-grained spatial matching (Wang et al., 2023; Lentsch et al., 2023; Xia & Alahi, 2025), existing approaches have achieved notable improvements in metric localization accuracy.

Despite these advances, a critical gap remains before existing fine-grained cross-view localization methods can be reliably deployed in real-world applications. Specifically, current methods largely operate in a local-refinement regime, assuming that the camera has been narrowed down to a tightly bounded aerial region before estimating its precise position. On VIGOR (Zhu et al., 2021), a widely used benchmark, the commonly adopted setting restricts the camera to a prior area of only about 35 × 35 meters (Lentsch et al., 2023; Xia et al., 2024; Xia & Alahi, 2025). Such a narrow range requires the upstream localization system to reduce its uncertainty to a comparable scale. In practice, however, upstream localization from GPS/GNSS or image retrieval may remain coarse, and such a precise prior may not be available (Xia et al., 2022; Shi et al., 2023). This mismatch limits the integration of cross-view localization into practical visual localization systems.

![](images/a569c9695895cdc60a83aac1aa9d7902c83d69321517c65e02004e3b83f5b49f.jpg)  
(a) Degradation over larger aerial regions.

![](images/03e9a8c72c3b23b6187aa8cb567f853edc9ab30173c4e5673b2eee1d97a9e895.jpg)  
(b) Ambiguity from look-alike locations.  
Figure 1: (a) Existing fine-grained cross-view localization methods incur substantially larger errors as the aerial region expands from 35 × 35 m to $4 0 0 \times 4 0 0 \mathrm { m }$ . (b) A large aerial region can contain multiple look-alike locations compatible with the same ground observation (inset). In this example, locations • A and • B share similar appearance with the true location (⋆). The main differences lie in subtle spatial details, such as building outlines and the arrangement of surrounding vegetation.

A natural way to accommodate coarser upstream localization priors is to enlarge the aerial region. However, this seemingly straightforward extension fundamentally changes the localization problem. As the aerial region expands, the ground observation covers only a small area within it, making it difficult to directly establish reliable cross-view matching. As shown in Figure 1a, expanding the aerial region can severely degrade existing methods. This growing spatial-extent disparity inherently shifts the task from local refinement to wide-area spatial search. Meanwhile, larger aerial regions often contain more recurring structures, creating multiple look-alike locations compatible with the same ground observation. Such locations often share similar appearance but differ in subtle details of their spatial layouts (see Figure 1b). Resolving this ambiguity requires recognizing and comparing these spatial cues across large viewpoint disparities, yet such fine-grained cross-view spatial reasoning remains challenging for current multimodal models (Jeong et al., 2026; Liu et al., 2026).

To address these challenges, we propose RingMatch, a search-and-verification framework for precise cross-view localization over large aerial regions. Rather than localizing directly over the full region, RingMatch first performs a multi-hypothesis branch-and-zoom search that explores the aerial region from coarse to fine. We use predictive uncertainty to expose distinct plausible hypotheses, retain them as separate search branches, and recursively zoom into their surroundings to access increasingly fine-grained evidence. By avoiding premature commitment under sparse cross-view correspondences, the search produces a compact shortlist of promising candidates.

Verification then focuses on resolving ambiguity among these look-alike candidates. To this end, we introduce the depth ring, a view-shared geometric representation that encodes camera-centered distances to surrounding scene surfaces across horizontal azimuths. Its azimuth-wise depth pattern captures subtle variations in the surrounding spatial layout, forming a distinctive location-specific signature that can be compared across ground and aerial views. We derive the ground ring from estimated depth and train an aerial ring predictor to reconstruct the corresponding ring from each candidate-centered aerial patch. A large vision-language model (LVLM) verifier then examines the overall ring patterns together with their visual context to select the final location.

To evaluate precise localization over large aerial regions, we construct VIGOR-Pro by automatically composing VIGOR aerial tiles into contiguous regions of approximately 100, 200, and 400 m. Across both same-area and cross-area evaluations, RingMatch consistently outperforms representative end-to-end and retrieval-based localization methods. At 400 m, RingMatch achieves mean localization errors of 12.36 m and 19.81 m on same-area and cross-area evaluation, respectively, compared with 15.53 m and 26.38 m for the strongest retrieval-based baseline. Beyond this, further analysis shows that our depth ring can serve as explicit spatial evidence for improving zero-shot cross-view spatial reasoning in frontier multimodal models such as GPT-5.6 Terra and Gemini-3.7 Flash, highlighting its potential for broader applications.

In summary, our main contributions are as follows:

• We study fine-grained cross-view localization beyond the conventional local-refinement regime, extending the task to substantially larger aerial regions. We further construct VIGOR-Pro to enable systematic evaluation under this setting.

• We propose a multi-hypothesis branch-and-zoom search that uses predictive uncertainty to preserve multiple plausible locations and progressively refine them, avoiding premature commitment under sparse cross-view correspondences.

• We introduce the depth ring, a view-shared geometric representation that captures the finegrained spatial structure surrounding a location. It enables effective disambiguation among look-alike locations and serves as explicit spatial evidence for cross-view reasoning.

## 2 PROBLEM SETUP AND CHALLENGES

## 2.1 PROBLEM FORMULATION

Following the standard problem formulation of fine-grained cross-view localization (Xia et al., 2022; 2024), we are given a ground panorama $I _ { g }$ and a geo-referenced aerial image $I _ { a }$ depicting an aerial region that contains the camera location. In practice, $I _ { a }$ can be obtained from an upstream localization system as a coarse geographic prior. Let $\Omega _ { a }$ denote the image domain of $I _ { a }$ . Our goal is to estimate the camera position

$$
\hat { \mathbf p } = ( \hat { x } , \hat { y } ) \in \Omega _ { a } .\tag{1}
$$

Unlike conventional fine-grained cross-view localization, we consider a broader aerial region to accommodate a coarser upstream localization prior. In this work, we consider the known-orientation setting, where the ground panorama has a known heading relative to the north-up aerial image, allowing us to isolate the challenges introduced by wide-area cross-view localization.

Since $I _ { a }$ is geo-referenced, let $\mathcal { G } _ { a } : \Omega _ { a } \to \mathbb { R } ^ { 2 }$ map each image coordinate to its corresponding geographic coordinate. The localization error can then be measured directly in meters:

$$
e _ { \mathrm { g e o } } ( \hat { \mathbf { p } } , \mathbf { p } ^ { \star } ) = d _ { \mathrm { g e o } } ( \mathcal { G } _ { a } ( \hat { \mathbf { p } } ) , \mathcal { G } _ { a } ( \mathbf { p } ^ { \star } ) ) ,\tag{2}
$$

where $\mathbf { p } ^ { \star } \in \Omega _ { a }$ denotes the ground-truth camera position and $d _ { \mathrm { g e o } }$ denotes the geographic distance.

## 2.2 WHAT CHALLENGES DOES WIDE-AREA CROSS-VIEW LOCALIZATION POSE?

Expanding the aerial region introduces challenges beyond simply enlarging the coordinate range.   
We identify two that are particularly critical.

Sparse ground–aerial correspondences. A ground observation captures only the local surroundings of the camera. As the aerial extent grows, the mismatch in spatial extent between the ground observation and the aerial region becomes increasingly pronounced, leaving only a small portion of the region with evidence that corresponds to the ground view. As shown in Figure 1a, expanding the aerial region can severely degrade existing end-to-end methods, particularly those relying on local cross-view alignment, including HC-Net (Wang et al., 2023) and $\mathrm { F G ^ { 2 } }$ (Xia & Alahi, 2025). This degradation stems from the fact that reliable cross-view evidence exists only within an unknown local area of the aerial region, making direct localization over the full region inherently challenging. The task therefore shifts from local refinement to a wide-area spatial search problem.

Visual ambiguity in ground–aerial matching. When cross-view localization is extended to larger aerial regions, recurring scene layouts and visual patterns can give rise to multiple look-alike locations. As shown in Figure 1b, such look-alike locations can share similar semantic content and visual layouts, making them difficult to distinguish based on appearance alone. Resolving this ambiguity requires fine-grained cross-view spatial reasoning: the ability to recognize and compare subtle spatial differences under large viewpoint disparities, including spatial configuration, relative geometry, and structural organization of scene elements. Recent studies show that multimodal models struggle with spatial reasoning, particularly egocentric–allocentric transformations (Yang et al., 2025). These limitations become more pronounced across viewpoints, where models must establish spatial correspondence and preserve object and layout consistency (Jeong et al., 2026; Liu et al., 2026).

![](images/f5ac510f7b0798e849c476d8adff454811747cc28aff55df2bd77b7c84ccc1bc.jpg)  
Figure 2: Overview of RingMatch. RingMatch first performs multi-hypothesis branch-and-zoom search over the aerial region, using predictive uncertainty to expose plausible hypotheses and recursively zoom into them to produce a compact candidate shortlist. It then resolves the remaining ambiguity with depth-ring-grounded verification. Ground observations and candidate aerial patches are mapped to depth rings that capture subtle details of the surrounding spatial layout. The correct candidate exhibits strong consistency with the ground observation, whereas a look-alike candidate does not. An LVLM verifier leverages this evidence to select the final location.

## 3 METHODOLOGY

## 3.1 OVERVIEW OF RINGMATCH

Section 2.2 leaves two coupled difficulties: cross-view correspondences over a large aerial region are sparse, and coarse visual evidence may point to multiple look-alike locations. Rather than aligning the ground view to the full aerial region in one step, we propose RingMatch, a search-andverification framework that first explores the aerial region for plausible locations and then disambiguates them using fine-grained spatial cues, as illustrated in Figure 2. Search explores the aerial region from coarse to fine, retaining multiple plausible hypotheses and progressively zooming into their surroundings to produce a candidate shortlist. Verification then distinguishes the remaining look-alike candidates. Since raw appearance is difficult to compare reliably across the ground–aerial viewpoint gap, we represent each view with a camera-centered depth ring, a view-shared geometric representation that captures fine-grained spatial structure around a location. An LVLM verifier then leverages this evidence together with visual context to select the most consistent candidate.

## 3.2 MULTI-HYPOTHESIS BRANCH-AND-ZOOM SEARCH

Sparse cross-view correspondences make a single prediction unreliable over a large aerial region. We therefore maintain multiple plausible hypotheses rather than committing prematurely. We represent the search as a tree, where each node corresponds to a location hypothesis and its associated search region. Starting from the root covering the full aerial region, we use predictive uncertainty to branch each node into multiple child hypotheses and zoom into their surroundings. Repeating this branchand-zoom process progressively refines plausible locations while preserving alternative search paths. The resulting leaf hypotheses are ranked to produce a shortlist for verification.

Uncertainty-guided branching. Let node v correspond to a search region $R _ { v } \subseteq \Omega _ { a }$ , with $R _ { \mathrm { r o o t } } =$ $\Omega _ { a }$ . We feed the aerial crop $I _ { a } ^ { K _ { v } }$ and $I _ { g }$ to a localizer parameterized by θ, which predicts a camera coordinate $\mathbf { u } \in R _ { v }$ . To expose multiple plausible modes, we draw M stochastic predictions

$$
\mathcal { U } _ { v } = \{ \mathbf { u } _ { v , m } \} _ { m = 1 } ^ { M } , \qquad \mathbf { u } _ { v , m } \sim p _ { \theta } ( \mathbf { u } \mid I _ { g } , I _ { a } ^ { R _ { v } } ) .\tag{3}
$$

We then apply single-linkage clustering (Gower & Ross, 1969) to group nearby predictions. Each retained cluster $C _ { v ^ { \prime } } ^ { \mathbf { ^ { \scriptscriptstyle - } } } \subseteq \mathcal { U } _ { v }$ defines a child node $v ^ { \prime } ,$ with its centroid as the corresponding hypothesis for a separate branch. Branching is therefore adaptive: concentrated predictions produce few branches, whereas multiple separated modes produce more. In this way, the search expands according to predictive uncertainty, allocating additional branches only when distinct hypotheses emerge.

Recursive zooming. For each retained hypothesis, we form a search region centered on it according to a predefined zoom schedule. Progressively narrowing the search region allows the localizer to access increasingly fine-grained evidence around each hypothesis. We repeat uncertainty-guided branching within each region, continuing this branch-and-zoom cycle until the final zoom level.

Hypothesis scoring and candidate shortlist. We score each hypothesis by aggregating the cluster sizes along its search path, based on the intuition that a promising hypothesis should remain well supported as the search progresses. For a hypothesis h at level ℓ with path $( v _ { 0 } , v _ { 1 } , \ldots , v _ { \ell } )$ , we define $\begin{array} { r } { S ( { \bf h } ) = \sum _ { t = 1 } ^ { \ell } | C _ { v _ { t } } | } \end{array}$ , where $C _ { v _ { t } }$ is the cluster that produced node $v _ { t }$ . To bound the search cost, we rank hypotheses by S after each level and retain at most B for further expansion. At the final level, the top-k leaf hypotheses $\{ { \bf h } _ { i } \} _ { i = 1 } ^ { k }$ form the candidate shortlist for verification.

## 3.3 DEPTH-RING-GROUNDED VERIFICATION

Search returns a shortlist of k plausible locations, but residual ambiguity may remain among them. As noted in Section 2.2, distinguishing them requires fine-grained spatial reasoning across large viewpoint disparities. To resolve this ambiguity, we introduce the depth ring, a view-shared geometric representation that encodes the estimated distance from a location to surrounding scene surfaces across horizontal azimuths. By expressing fine-grained spatial structure from both views in a shared representation, depth rings enable verification based on geometric consistency rather than direct appearance comparison across the viewpoint gap.

Depth ring as a spatial signature of a place. For a location p, we define its depth ring as the distance from p to the first scene surface encountered along each horizontal direction:

$$
r _ { \mathbf { p } } ( \phi ) = \operatorname* { i n f } \{ \rho > 0 : { \bf p } + \rho { \bf d } ( \phi ) \mathrm { ~ r e a c h e s ~ a ~ s c e n e ~ s u r f a c e } \} , \qquad \phi \in [ 0 , 2 \pi ) ,\tag{4}
$$

where ϕ denotes the azimuth, and $\mathbf { d } ( \phi ) = ( \cos \phi , \sin \phi ) ^ { \top }$ is the corresponding unit vector in the horizontal plane. Sampling the ring at A azimuths gives $\mathbf { r _ { p } } = [ r _ { \mathbf { p } } ( \phi _ { 1 } ) , \hdots , r _ { \mathbf { p } } ( \phi _ { A } ) ] \in \mathbb { R } ^ { A }$

Although one-dimensional, the depth ring faithfully captures subtle variations in the surrounding spatial layout (see Figure 3). Its azimuth-wise depth pattern therefore forms a distinctive spatial signature of a location. As shown in Figure 2, the aerial ring at the true location closely agrees with the ground ring, whereas a look-alike location with a different surrounding layout produces an inconsistent profile. Since the depth ring is indexed by azimuth, crossview comparison requires a common directional reference. Under the known-orientation setting, ground and aerial rings can be directly aligned; for unknown headings, the comparison can be extended by first estimating and compensating for the relative yaw.

This representation relates to classical isovist-based descriptions of surrounding visibility (Benedikt, 1979) and LiDAR place descriptors that compactly encode surrounding geometry (Kim & Kim, 2018; Lu et al., 2022). However, since no range scans are directly available in our setting, depth rings must be inferred from ground and aerial observations.

![](images/de12baf5eee724ec082b7aa4c7d4c33ff55ddfd5c897513ea3523c235fc1170e.jpg)  
Figure 3: A depth ring summarizes azimuthwise distances to surrounding scene surfaces in the horizontal plane. Its shape reveals subtle variations in the surrounding spatial layout, forming a compact spatial signature for cross-view comparison. The ring is derived from the ground observation and overlaid on the aerial patch only for visualization.

Constructing rings from both views. Ground-to-ring. The ground panorama observes the surrounding scene, with its horizontal coordinates naturally corresponding to azimuths. We use a pretrained metric depth model (Yang et al., 2024) to estimate a depth map $D _ { g }$ from $I _ { g } .$ For each azimuth $\phi _ { a }$ , we then pool depth values within a narrow vertical band $\bar { B } ( \phi _ { a } )$ around the horizon:

$$
r _ { g } ( \phi _ { a } ) = \mathrm { P o o l } \left\{ D _ { g } ( x , y ) \mid ( x , y ) \in \mathcal { B } ( \phi _ { a } ) \right\} , \qquad a = 1 , \ldots , A .\tag{5}
$$

Aerial-to-ring. Unlike the ground panorama, the aerial view does not directly provide azimuth-wise depth around a location. We therefore learn an aerial-to-ring predictor $f _ { \varphi }$ that maps a local aerial patch $I _ { a } ^ { \mathbf { p } }$ together with its specified location p to the corresponding azimuth-wise depth ring:

$$
\hat { \mathbf { r } } _ { a } ( \mathbf { p } ) = f _ { \varphi } \left( I _ { a } ^ { \mathbf { p } } , \mathbf { p } \right) .\tag{6}
$$

We implement $f _ { \varphi }$ as azimuth-wise metric depth estimation. Training requires no additional annotation: for each ground–aerial pair with a known camera position, we construct the ground ring using Eq. (5) and use it to supervise the aerial prediction at the corresponding location. Architectural and training details are provided in Appendix C.2.

LVLM-based candidate verification. Although depth rings provide explicit spatial evidence, direct numerical comparison can be brittle under local depth errors. We therefore use a learned LVLM verifier to compare overall ring patterns rather than relying on point-wise agreement. For each candidate $\mathbf { h } _ { i } ,$ we align the ground ring $\mathbf { r } _ { g }$ and predicted aerial ring $\hat { \mathbf { r } } _ { a } ( \mathbf { h } _ { i } )$ in a shared polar coordinate system, and provide them together with the ground panorama and the corresponding candidate aerial patch to the verifier. It then selects the most plausible candidate from $\{ { \bf h } _ { i } \} _ { i = 1 } ^ { k ^ { \star } }$ as the final prediction.

## 3.4 TRAINING STRATEGY FOR RINGMATCH

Multi-scale training for search. We use a single cross-view localizer $p _ { \theta }$ across all levels of branch-and-zoom, requiring it to operate over aerial regions at different scales. We therefore aggregate ground–aerial training pairs across scales and jointly train the localizer. Leveraging the visual understanding and flexible generation capabilities of LVLMs, we implement $p _ { \theta }$ within an LVLM framework rather than using conventional cross-view localization models (Xia et al., 2024; Xia & Alahi, 2025). Given a ground panorama and an aerial image, the model is trained to generate the camera coordinate in text form using the ground-truth coordinate as supervision.

Hard-candidate training for verifier. After training the search model, we run multi-hypothesis search on the ground–aerial training pairs and use the retained hypotheses as hard candidates for verifier training. The verifier is fine-tuned to output the selected candidate in text form, using the candidate closest to the ground-truth location as supervision. Because these candidates are drawn from the hypotheses retained after search, they naturally preserve the residual visual ambiguity, exposing the verifier to challenging cases similar to those encountered at inference.

## 4 EXPERIMENTS

## 4.1 SETUP

Dataset and evaluation protocol. Existing benchmarks for fine-grained cross-view localization are mainly designed for narrow geographic priors, where localization is performed within a relatively small aerial region. To evaluate precise localization over larger aerial regions, we construct VIGOR-Pro from VIGOR (Zhu et al., 2021) by composing neighboring aerial tiles into contiguous mosaics while preserving the original ground panoramas and camera locations. No additional imagery or manual annotation is introduced. We construct three scales with approximate side lengths of 100, 200, and 400 m, corresponding to increasingly coarse geographic priors. Following the standard VIGOR protocol, we evaluate under both same-area and cross-area settings. The former uses all cities for both training and testing, while the latter trains on two cities and evaluates on the remaining two unseen cities. All experiments use the known-orientation setting, and we report mean and median localization errors in meters following prior work (Xia & Alahi, 2025). Detailed construction procedures and dataset statistics are provided in Appendix B.

Table 1: Main results on VIGOR-Pro under same-area and cross-area settings across increasingly large aerial regions. Mean and median localization errors are reported in meters. The best results are highlighted in bold, and the second-best results are underlined.
<table><tr><td rowspan="2">Area</td><td rowspan="2">Method</td><td colspan="2">100m</td><td colspan="2">200 m</td><td colspan="2">400 m</td></tr><tr><td>Mean ↓</td><td>Median ↓</td><td>Mean ↓</td><td>Median ↓</td><td>Mean ↓</td><td>Median ↓</td></tr><tr><td rowspan="13">Sam-rea</td><td>End-to-End</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HC-Net (Wang et al., 2023)</td><td>10.52</td><td>3.70</td><td>29.54</td><td>10.53</td><td>85.62</td><td>58.49</td></tr><tr><td>CCVPE (Xia et al., 2024)</td><td>9.63</td><td>2.11</td><td>14.07</td><td>2.64</td><td>27.90</td><td>3.48</td></tr><tr><td>FG2 (Xia &amp; Alahi, 2025)</td><td>9.76</td><td>4.63</td><td>18.38</td><td>8.14</td><td>38.41</td><td>17.00</td></tr><tr><td>Qwen3-VL-4B</td><td>5.36</td><td>2.27</td><td>12.16</td><td>3.52</td><td>23.03</td><td>3.93</td></tr><tr><td>Retrieval-Based</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sample4Geo (Deuser et al., 2023)</td><td>23.84</td><td>20.85</td><td>22.80</td><td>16.85</td><td>23.84</td><td>15.89</td></tr><tr><td>Sample4Geo+FG²</td><td>13.35</td><td>5.17</td><td>12.80</td><td>3.68</td><td>15.53</td><td>4.72</td></tr><tr><td>CIPÈR (Jeon et al., 2026)</td><td>18.69</td><td>10.48</td><td>25.78</td><td>8.61</td><td>40.79</td><td>8.65</td></tr><tr><td>Our Search-Based Framework</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Single-Hypothesis</td><td>5.00</td><td>2.00</td><td>10.29</td><td>2.27</td><td>21.42</td><td>2.46</td></tr><tr><td>RingMatch (Search-only)</td><td>4.32</td><td>1.65</td><td>8.03</td><td>1.82</td><td>15.60</td><td>1.92</td></tr><tr><td>RingMatch</td><td>4.02</td><td>1.65</td><td>6.57</td><td>1.79</td><td>12.36</td><td>1.88</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="10">Crosarea</td><td>End-to-End HC-Net (Wang et al., 2023)</td><td>14.02</td><td>5.10</td><td>44.12</td><td>18.49</td><td>121.71</td><td></td></tr><tr><td>CCVPE (Xia et al., 2024)</td><td></td><td></td><td></td><td></td><td></td><td>110.38</td></tr><tr><td></td><td>19.29</td><td>3.58</td><td>41.65</td><td>4.67</td><td>94.18</td><td>7.69</td></tr><tr><td>FG2 (Xia &amp; Alahi, 2025) Qwen3-VL-4B</td><td>10.77 7.12</td><td>5.22 3.87</td><td>21.68</td><td>10.23 5.45</td><td>62.21</td><td>35.56</td></tr><tr><td>Retrieval-Based</td><td></td><td></td><td>17.25</td><td></td><td>34.64</td><td>6.16</td></tr><tr><td>Sample4Geo (Deuser et al., 2023)</td><td>25.02</td><td>21.63</td><td>27.78</td><td>18.01</td><td>34.80</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>17.25</td></tr><tr><td>Sample4Geo+FG²</td><td>14.00</td><td>4.86</td><td>17.66</td><td>3.36</td><td>26.38</td><td>4.78</td></tr><tr><td>CIPÈR (Jeon et al., 2026)</td><td>21.57</td><td>12.46</td><td>40.41</td><td>13.85</td><td>80.84</td><td>22.59</td></tr><tr><td>Our Search-Based Framework</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Single-Hypothesis</td><td>5.79</td><td>2.48</td><td>12.90</td><td>2.29</td><td>29.19</td><td>2.40</td></tr><tr><td>RingMatch (Search-only)</td><td>4.95</td><td>1.95</td><td>11.01</td><td>1.99</td><td>23.99</td><td>2.09</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>19.81</td><td></td></tr><tr><td>RingMatch</td><td>4.81</td><td>1.94</td><td>9.31</td><td>1.96</td><td></td><td>2.04</td></tr></table>

Implementation details. The search model is initialized from Qwen3-VL-4B-Instruct (Bai et al., 2025) and jointly trained on 50, 100, 200, and 400 m regions to support localization across varying aerial extents. The additional 50 m regions are generated from the 100 m training regions and used only for multi-scale training. During inference, we follow the {400, 200, 100, 50} m zoom schedule starting from each evaluation scale. Unless specified otherwise, we draw M = 16 predictions per node at temperature 1 and use a pruning budget of B = 4. After the final search stage, we retain the top-2 hypotheses for verification. Ground depth rings are constructed with Depth Anything V2 (Yang et al., 2024), and the aerial ring predictor is trained with ground-derived ring supervision. The verifier is initialized from the search model and further fine-tuned on hard candidates generated under the same search configuration. Further implementation details are provided in Appendix C.

Baselines. We compare RingMatch against representative end-to-end and retrieval-based localization methods, and further report controlled variants within our search-based framework. (1) Endto-end localization directly predicts the camera position from a ground–aerial pair. This category includes representative cross-view localization methods, such as HC-Net (Wang et al., 2023), CCVPE (Xia et al., 2024), and $\mathrm { F G ^ { 2 } }$ (Xia & Alahi, 2025). We additionally report direct prediction results from our fine-tuned Qwen3-VL-4B search model. (2) Retrieval-based localization treats the original aerial tiles composing each VIGOR-Pro mosaic as the retrieval gallery. We adopt Sample4Geo (Deuser et al., 2023), a Sample4Geo+FG<sup>2</sup> cascade, and CIPER (Jeon et al., 2026). Sample4Geo uses the center of the top-1 retrieved tile as the prediction; Sample4Geo+FG<sup>2</sup> further refines the location within that tile using $\mathrm { F G ^ { 2 } }$ , a recent strong method for local refinement, while CIPER jointly performs retrieval and local pose estimation. (3) Our search-basedframework progressively explores and zooms into the aerial region. Within this framework, we introduce Single-Hypothesis, an ablation variant that uses the same search model as RingMatch but retains only one hypothesis per level. Each existing end-to-end baseline is retrained independently at every VIGOR-Pro scale using its official configuration. For Sample4Geo and CIPER, we use the released checkpoints and follow their official inference procedures. Further details are provided in Appendix D.

Table 2: Verifier ablation on VIGOR-Pro under same-area evaluation. Mean localization error (m) is reported on all test samples (ALL) and the verification subset (VFY). “Search init.” denotes initialization from the trained search model.
<table><tr><td rowspan="2"></td><td colspan="3">Verifier components</td><td colspan="2">100 m</td><td colspan="2">200 m</td><td colspan="2">400 m</td></tr><tr><td>Local patch</td><td>Depth ring</td><td>Search init.</td><td>ALL</td><td>VFY</td><td>ALL</td><td>VFY</td><td>ALL</td><td>VFY</td></tr><tr><td>Search-only</td><td>一</td><td>一</td><td></td><td>4.32</td><td>22.59</td><td>8.03</td><td>41.64</td><td>15.60</td><td>70.58</td></tr><tr><td rowspan="3">w/ Verifier</td><td>√</td><td></td><td></td><td>4.60</td><td>26.89</td><td>8.67</td><td>48.11</td><td>17.05</td><td>81.06</td></tr><tr><td>√</td><td>√</td><td></td><td>4.26</td><td>21.67</td><td>7.36</td><td>34.77</td><td>13.89</td><td>58.17</td></tr><tr><td>√</td><td>√</td><td>V</td><td>4.02</td><td>17.59</td><td>6.57</td><td>26.69</td><td>12.36</td><td>47.12</td></tr></table>

## 4.2 MAIN RESULTS

Localization over increasingly large aerial regions. Table 1 shows that conventional end-to-end localizers degrade sharply as the aerial region expands. On same-area evaluation, the mean error of CCVPE rises from 9.63 m at 100 m to 27.90 m at 400 m, with even larger increases for HC-Net and FG<sup>2</sup>. In contrast, RingMatch achieves the lowest errors across all scales, reaching 12.36/1.88 m mean/median error at 400 m, compared with 15.53/4.72 m for the strongest retrieval-based baseline.

Cross-area generalization. The advantage of RingMatch persists when generalizing to unseen geographic areas. At 400 m, RingMatch achieves mean/median errors of 19.81/2.04 m, outperforming the strongest retrieval-based baseline Sample4Geo+FG<sup>2</sup> with 26.38/4.78 m. This demonstrates the robustness of RingMatch to geographic distribution shifts.

Broader regions demand multi-hypothesis exploration. Compared with Single-Hypothesis, RingMatch (Search-only) reduces the same-area mean error by 13.6%, 22.0%, and 27.2% at 100, 200, and 400 m, respectively. The increasing gain shows that premature commitment becomes more costly as the aerial region expands, underscoring the need for multi-hypothesis exploration.

Depth-ring-grounded verification corrects costly mis-selections. Compared with search alone, verification further reduces the mean error at 400 m from 15.60 to 12.36 m under same-area evaluation and from 23.99 to 19.81 m under cross-area evaluation. In contrast, the corresponding median errors change only slightly. This suggests that verification primarily corrects costly mis-selections among plausible look-alike candidates rather than refining already accurate predictions.

## 4.3 ANALYSIS

Depth rings enable stronger zero-shot cross-view spatial reasoning. Table 3 evaluates whether depth rings can help frontier multimodal models resolve visual ambiguity across ground–aerial views without task-specific fine-tuning. With visual evidence alone, GPT-5.6 Terra and Gemini-3.7 Flash remain limited in recognizing fine-grained spatial differences, with GPT-5.6 Terra performing only slightly above chance. Ring-only already achieves 67.29% and 69.42% accuracy at 200 and 400 m, showing that depth rings carry discriminative spatial information. Providing depth-ring evidence improves GPT-5.6 Terra by 10.65 and 6.74 points and Gemini-3.7 Flash by 3.22 and 2.73 points. Notably, Gemini with depth-ring evidence further surpasses Ring-only, suggesting that depth rings serve not only as geometric descriptors for direct crossview matching, but also as explicit spatial evidence for improving cross-view reasoning.

Table 3: Verification accuracy (%) on VIGOR-Pro same-area verification samples. FT denotes task-specific fine-tuning. Ringonly selects the candidate with the higher Pearson correlation between its predicted aerial depth ring and the ground depth ring.
<table><tr><td>Method</td><td>FT</td><td>200 m</td><td>400 m</td></tr><tr><td>Random</td><td></td><td>50.00</td><td>50.00</td></tr><tr><td>Ring-only</td><td>一</td><td>67.29</td><td>69.42</td></tr><tr><td>GPT-5.6 Terra w/ depth ring Gemini-3.7 Flash w/ depth ring</td><td>一 一</td><td>56.93 67.58 (+10.65) 66.70 69.92 (+3.22)</td><td>59.47 66.21 (+6.74) 72.56 75.29 (+2.73)</td></tr><tr><td>RingMatch</td><td>√</td><td>81.92</td><td>82.81</td></tr></table>

(a) Level L  
![](images/29f9257266368bd2d63ee26cffde6005642e721270989013b5d44e44a3e6d329.jpg)  
Candidate count k

(b) Samples M  
![](images/e835764f7151fe3cd5def289c4563e2b2a2987c9d09296bb58f7c3a8d4f99a2f.jpg)  
Candidate count k

(c) Budget B  
![](images/d819054f9fa74e4255c9350a62f8816f669c3f4a95c533ba79a2b14b2474968c.jpg)  
Candidate count k  
Figure 4: Analysis of multi-hypothesis search on VIGOR-Pro under same-area evaluation at the 400 m scale. We report top-k oracle mean localization error when varying (a) search level L, (b) sampling count M, and (c) pruning budget B, where the oracle selects the closest location among the top-k candidates. Unless otherwise varied, we use L = 4, M = 16, and B = 4.

Table 4: Generalization beyond VIGOR to four unseen cities at the 200 m scale.
<table><tr><td rowspan="2">Method</td><td colspan="2">London</td><td colspan="2"> $\mathrm { T o k y o }$ </td><td colspan="2">São Paulo</td><td colspan="2">Cape Town</td><td colspan="2">Overall</td></tr><tr><td>Mean ↓</td><td>Median ↓</td><td>Mean↓</td><td>Median ↓</td><td>Mean↓</td><td>Median ↓</td><td>Mean↓</td><td>Median↓</td><td>Mean↓</td><td>Median ↓</td></tr><tr><td>FG2</td><td>14.98</td><td>7.24</td><td>36.26</td><td>20.87</td><td>30.91</td><td>17.13</td><td>23.52</td><td>12.41</td><td>26.42</td><td>13.41</td></tr><tr><td>Sample4Geo+FG²</td><td>9.47</td><td>2.22</td><td>24.38</td><td>4.72</td><td>19.09</td><td>3.92</td><td>14.14</td><td>2.82</td><td>16.77</td><td>3.20</td></tr><tr><td>RingMatch (Search-only)</td><td>9.65</td><td>1.23</td><td>21.64</td><td>1.04</td><td>19.95</td><td>1.49</td><td>9.57</td><td>1.02</td><td>15.20</td><td>1.19</td></tr><tr><td>RingMatch</td><td>8.58</td><td>1.21</td><td>17.42</td><td>1.00</td><td>18.48</td><td>1.45</td><td>7.89</td><td>0.98</td><td>13.09</td><td>1.16</td></tr></table>

Depth-ring evidence improves candidate verification. Table 2 isolates the verifier components. A verifier using visual patches alone fails to improve over search and can even degrade localization, confirming the difficulty of resolving look-alike candidates from appearance alone. Adding depthring evidence reduces the mean error on verification samples from 81.06 to 58.17 m at 400 m, while initialization from the trained search model further lowers it to 47.12 m, showing the complementarity of explicit spatial evidence and cross-view representations learned during search.

Multi-hypothesis search preserves promising locations. To isolate search quality from verification, Figure 4 reports top-k oracle mean error on VIGOR-Pro under same-area evaluation at 400 m. With the default setting (L = 4, M = 16, B = 4), increasing k from 1 to 2 reduces the error from 15.60 to 10.43 m, while k = 5 further reaches 8.52 m. The second candidate captures 73.0% of the total gain from k = 1 to k = 5, supporting a compact top-2 shortlist for verification.

Sensitivity to search parameters. Search quality improves with deeper exploration and sufficient stochastic sampling. At k = 2, increasing L from 1 to 4 reduces the oracle error from 12.49 to 10.43 m, while increasing M from 8 to 16 reduces it from 12.94 to 10.43 m, with diminishing gains at larger M. Performance is comparatively insensitive to the pruning budget when enough branches are retained, with B = 4 and 8 giving similar results. More results are provided in Appendix G.

Generalization beyond VIGOR. To examine whether the observed gains extend beyond VIGOR, we further collect evaluation data from four cities outside the benchmark—London, Tokyo, São Paulo, and Cape Town—and evaluate at the 200 m scale. For all methods, we directly use models trained under the VIGOR same-area setting without target-city adaptation. As shown in Table 4, RingMatch consistently outperforms both baselines across all four cities, achieving 13.09/1.16 m mean/median error overall compared with 16.77/3.20 m for Sample4Geo+FG<sup>2</sup>. These results show that the gains of RingMatch extend beyond the geographic and domain characteristics of VIGOR.

## 5 RELATED WORK

Precise cross-view localization. Cross-view retrieval matches ground observations against georeferenced aerial galleries (Zhu et al., 2021; 2022; Deuser et al., 2023), while metric localization estimates continuous camera coordinates or poses within an aerial reference. Existing metric methods use dense pose predictions, geometric alignment, or fine-grained feature correspondences (Wang et al., 2023; Xia et al., 2024; Xia & Alahi, 2025). Although effective for local refinement, these methods are typically evaluated with tightly bounded aerial priors. We study precise localization within substantially larger contiguous aerial regions, where sparse ground–aerial correspondences make direct localization increasingly difficult.

Wide-area cross-view localization and hierarchical search. UnifyGeo (Song et al., 2025) and CIPER (Jeon et al., 2026) integrate large-scale retrieval and metric localization within unified frameworks. Just Zoom In (Erzurumlu et al., 2026) and GeoMoE (Fan et al., 2026) progressively narrow geographic search spaces through hierarchies of satellite cells or gallery entries. These approaches address related coarse-to-fine localization challenges, but emphasize retrieval–localization integration or hierarchical geographic narrowing. Our work focuses on recovering precise continuous coordinates within a given coarse aerial prior. We form spatially distinct hypotheses from stochastic coordinate predictions and recursively zoom into their neighborhoods, followed by explicit verification of the remaining ambiguous candidates.

Geometry and cross-view spatial reasoning. Cross-view matching has long leveraged geometric transformations, geometry-guided aggregation, and local correspondences (Shi et al., 2019; Lentsch et al., 2023; Xia & Alahi, 2025). Meanwhile, LVLM-based geolocation exploits scene semantics and geographic knowledge (Li et al., 2024; 2025), yet fine-grained spatial reasoning remains challenging across viewpoints (Jeong et al., 2026; Liu et al., 2026). To complement existing geometric matching approaches, we introduce the depth ring, a camera-centered representation of the surrounding spatial layout. By deriving ground rings from estimated depth and predicting corresponding aerial rings from overhead imagery, we provide explicit geometric evidence that enables an LVLM verifier to distinguish visually similar locations through fine-grained spatial differences.

Appendix A provides an extended review of related work and a more detailed discussion of the connections between prior work and our approach.

## 6 CONCLUSION

We study fine-grained cross-view localization beyond the conventional local-refinement regime to better accommodate the coarse localization priors. To address the sparse cross-view correspondences and visual ambiguity arising over larger aerial regions, RingMatch combines multihypothesis branch-and-zoom search with depth-ring-grounded verification, preserving plausible locations while exploiting fine-grained spatial structure for disambiguation. Experiments on VIGOR-Pro show consistent improvements across increasing aerial scales. Further analysis shows that depth rings provide explicit spatial evidence for cross-view reasoning in frontier multimodal models.

## REFERENCES

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Michael L. Benedikt. To take hold of space: Isovists and isovist fields. Environment and Planning B: Planning and Design, 6(1):47–65, 1979.

Fabian Deuser, Konrad Habel, and Norbert Oswald. Sample4Geo: Hard negative sampling for crossview geo-localisation. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 16847–16856, 2023.

Yunus Talha Erzurumlu, Jiyong Kwag, and Alper Yilmaz. Just zoom in: Cross-view geo-localization via autoregressive zooming. arXiv preprint arXiv:2603.25686, 2026.

Ruijie Fan, Junyan Ye, Qi Zhu, and Weijia Li. One query, many scales: Sparse mixture-of-experts for efficient hierarchical cross-view geo-localization. arXiv preprint arXiv:2608.01060, 2026.

Florian Fervers, Sebastian Bullinger, Christoph Bodensteiner, Michael Arens, and Rainer Stiefelhagen. Uncertainty-aware vision-based metric cross-view geolocalization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21621–21631, 2023.

J. C. Gower and G. J. S. Ross. Minimum spanning trees and single linkage cluster analysis. Journal ofthe Royal Statistical Society. Series C (Applied Statistics), 18(1):54–64, 1969.

Sixing Hu, Mengdan Feng, Rang M. H. Nguyen, and Gim Hee Lee. CVM-Net: Cross-view matching network for image-based ground-to-aerial geo-localization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7258–7267, 2018.

Yurim Jeon, Dongseong Seo, and Seung-Woo Seo. CIPER: A unified framework for cross-view image-retrieval and pose-estimation. arXiv preprint arXiv:2606.05011, 2026.

Suchae Jeong, Jaehwi Song, Haeone Lee, Hanna Kim, Jian Kim, Dongjun Lee, Dong Kyu Shin, Changyeon Kim, Dongyoon Hahm, Woogyeol Jin, Juheon Choi, and Kimin Lee. Learning multiview spatial reasoning from cross-view relations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Pengyue Jia, Yiding Liu, Xiaopeng Li, Yuhao Wang, Yantong Du, Xiao Han, Xuetao Wei, Shuaiqiang Wang, Dawei Yin, and Xiangyu Zhao. G3: an effective and adaptive framework for worldwide geolocalization using large multi-modality models. Advances in neural information processing systems, 37:53198–53221, 2024.

Giseop Kim and Ayoung Kim. Scan context: Egocentric spatial descriptor for place recognition within 3D point cloud map. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 4802–4809, 2018. doi: 10.1109/IROS.2018.8593953.

Ted Lentsch, Zimin Xia, Holger Caesar, and Julian F. P. Kooij. SliceMatch: Geometry-guided aggregation for cross-view pose estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17225–17234, 2023.

Ling Li, Yu Ye, Bingchuan Jiang, and Wei Zeng. GeoReasoner: Geo-localization with reasoning in street views using a large vision-language model. In International Conference on Machine Learning (ICML), volume 235, pp. 29222–29233, 2024.

Ling Li, Yao Zhou, Yuxuan Liang, Fugee Tsung, and Jiaheng Wei. Recognition through reasoning: Reinforcing image geo-localization with large vision-language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pp. 69058–69085, 2025.

Ruixun Liu, Lingyu Zhang, Lanxuan Xue, Kaiyu Li, Bowen Fu, and Xiangyong Cao. Cvsbench: A comprehensive benchmark for cross-view spatial reasoning and dreaming. In European Conference on Computer Vision, pp. 496–514. Springer, 2026.

Sha Lu, Xuecheng Xu, Huan Yin, Zexi Chen, Rong Xiong, and Yue Wang. One RING to rule them all: Radon sinogram for place recognition, orientation and translation estimation. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 2778–2785, 2022. doi: 10.1109/IROS47612.2022.9981308.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Juhye Park, Wooju Lee, Dasol Hong, Changki Sung, Youngwoo Seo, Dongwan Kang, and Hyun Myung. VIRD: View-invariant representation through dual-axis transformation for cross-view pose estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 33693–33702, 2026.

Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 12179–12188, 2021.

Yujiao Shi and Hongdong Li. Beyond cross-view image retrieval: Highly accurate vehicle localization using satellite image. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17010–17020, 2022.

Yujiao Shi, Liu Liu, Xin Yu, and Hongdong Li. Spatial-aware feature aggregation for image based cross-view geo-localization. Advances in Neural Information Processing Systems, 32, 2019.

Yujiao Shi, Fei Wu, Akhil Perincherry, Ankit Vora, and Hongdong Li. Boosting 3-dof groundto-satellite camera localization accuracy via geometry-guided cross-view transformer. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 21459–21469. IEEE, 2023.

Zhuo Song, Ye Zhang, Kunhong Li, Longguang Wang, and Yulan Guo. A unified hierarchical framework for fine-grained cross-view geo-localization over large-scale scenarios. arXiv preprint arXiv:2505.07622, 2025.

Vicente Vivanco Cepeda, Gaurav Kumar Nayak, and Mubarak Shah. GeoCLIP: Clip-inspired alignment between locations and images for effective worldwide geo-localization. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Nam Vo, Nathan Jacobs, and James Hays. Revisiting im2gps in the deep learning era. In Proceedings ofthe IEEE international conference on computer vision, pp. 2621–2630, 2017.

Wei Wang, Yuqian Yuan, Tianwei Lin, Wenqiao Zhang, Siliang Tang, Jun Xiao, and Yueting Zhuang. CrossView Suite: Harnessing cross-view spatial intelligence of MLLMs with dataset, model and benchmark. arXiv preprint arXiv:2605.18621, 2026.

Xiaolong Wang, Runsen Xu, Zhuofan Cui, Zeyu Wan, and Yu Zhang. Fine-grained cross-view geolocalization using a correlation-aware homography estimator. Advances in Neural Information Processing Systems, 36:5301–5319, 2023.

Xingrui Wang, Wufei Ma, Tiezheng Zhang, Celso M. de Melo, Jieneng Chen, and Alan L. Yuille. Spatial457: A diagnostic benchmark for 6D spatial reasoning of large multimodal models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Tobias Weyand, Ilya Kostrikov, and James Philbin. PlaNet – photo geolocation with convolutional neural networks. In European Conference on Computer Vision (ECCV), 2016.

Zimin Xia and Alexandre Alahi. FG<sup>2</sup>: Fine-grained cross-view localization by fine-grained feature matching. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6362–6372, 2025.

Zimin Xia, Olaf Booij, Marco Manfredi, and Julian F. P. Kooij. Visual cross-view metric localization with dense uncertainty estimates. In European Conference on Computer Vision (ECCV), pp. 90– 106, 2022. doi: 10.1007/978-3-031-19842-7\_6.

Zimin Xia, Olaf Booij, and Julian F. P. Kooij. Convolutional cross-view pose estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 46(5):3813–3831, 2024. doi: 10.1109/TPAMI.2023.3346924.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10632–10643. IEEE, 2025.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Junyan Ye, Zhutao Lv, Weijia Li, Jinhua Yu, Haote Yang, Huaping Zhong, and Conghui He. Crossview image geo-localization with panorama-BEV co-retrieval network. In European Conference on Computer Vision (ECCV), 2024.

Sijie Zhu, Taojiannan Yang, and Chen Chen. VIGOR: Cross-view image geo-localization beyond one-to-one retrieval. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3640–3649, 2021. doi: 10.1109/CVPR46437.2021.00364.

Sijie Zhu, Mubarak Shah, and Chen Chen. TransGeo: Transformer is all you need for cross-view image geo-localization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1162–1171, 2022.

## APPENDIX

This appendix provides extended related work, dataset documentation, implementation and evaluation details, and additional qualitative and quantitative analyses for the main paper:

• § A extends the related-work discussion on visual geo-localization, precise cross-view localization, geometry-based representations, and LVLM reasoning.

• § B documents VIGOR-Pro construction, coordinate transformations, benchmark scales, and dataset statistics, with representative wide-area examples in Figure 8.

• § C details training and inference for the search model, depth-ring predictor, and verifier, including supervision, optimization, and inference procedures.

• § D describes baseline adaptation and the common evaluation protocols for end-to-end, retrieval-based, and search-based methods.

• § E specifies the zero-shot cross-view reasoning experiments and the generalization evalu ation beyond VIGOR, with verifier prompts in Figures 13 and 14.

• § F analyzes qualitative depth-ring examples and their ability to distinguish look-alike locations, illustrated in Figure 9.

• § G extends the analysis of search levels, sampling counts, and pruning budgets across all VIGOR-Pro scales and both evaluation settings in Figures 10–12.

## A EXTENDED RELATED WORK

This section expands the discussion in Section 5 with broader background and detailed comparisons.

Image geo-localization and cross-view geo-localization. Visual geo-localization spans a broad range of spatial extents. Global methods infer geographic locations through classification, retrieval, or joint image–coordinate representations (Weyand et al., 2016; Vo et al., 2017; Vivanco Cepeda et al., 2023), while recent LVLM-based systems exploit semantic cues and geographic knowledge for reasoning-based image geo-localization (Li et al., 2024; Jia et al., 2024; Li et al., 2025). When geo-referenced aerial imagery is available, cross-view geo-localization matches ground observations against an aerial gallery to identify the corresponding geographic location (Hu et al., 2018; Shi et al., 2019; Zhu et al., 2022; Deuser et al., 2023; Ye et al., 2024). These methods provide broad geographic coverage and can serve as an upstream stage that narrows the localization prior.

Fine-grained cross-view localization. A complementary line of work focuses on precise metric localization within a local aerial reference. Existing methods predict continuous 2D positions or 3-DoF poses using dense pose distributions, geometric transformations, correspondence matching, or pose-conditioned features (Xia et al., 2022; Fervers et al., 2023; Lentsch et al., 2023; Wang et al., 2023; Xia et al., 2024; Xia & Alahi, 2025; Park et al., 2026). While these approaches achieve high metric accuracy, they largely operate in a local-refinement regime whose localization range is determined by the supplied aerial prior. On VIGOR (Zhu et al., 2021), commonly used protocols assume only a small neighborhood around the camera (Lentsch et al., 2023; Xia et al., 2024; Xia & Alahi, 2025). Our work instead studies precise cross-view localization when this upstream prior remains much broader.

Bridging global and fine-grained localization. Recent systems increasingly connect these two stages. UnifyGeo integrates retrieval, re-ranking, and metric localization, while CIPER jointly performs city-scale retrieval and local 3-DoF pose estimation (Song et al., 2025; Jeon et al., 2026). Just Zoom In and GeoMoE instead perform hierarchical coarse-to-fine geographic narrowing (Erzurumlu et al., 2026; Fan et al., 2026). Their retrieval and narrowing stages are complementary to ours and can provide the coarse geographic prior assumed by our setting. Although hierarchical narrowing is related to our progressive search, these hierarchical methods primarily navigate predefined discrete cells or gallery entries. In contrast, we directly localize continuous camera coordinates within a contiguous aerial region, using predictive uncertainty to retain multiple plausible hypotheses and progressively refine them. Our goal is therefore to extend precise cross-view localization to broader geographic priors rather than replace the upstream localization stage.

Cross-view geometry and spatial representations. Across ground–aerial cross-view tasks, geometry has long been used to bridge the viewpoint gap through polar transformations, bird’s-eyeview projections, homography estimation, geometry-guided matching, and point–plane correspondence (Shi et al., 2019; Shi & Li, 2022; Lentsch et al., 2023; Wang et al., 2023; Ye et al., 2024; Xia & Alahi, 2025; Park et al., 2026). These designs support cross-view matching or pose estimation by making the two views more geometrically compatible. Our depth ring instead provides a view-shared geometric interface that brings spatial structure from both views into a common comparison space, exposing subtle layout differences among look-alike locations. It is conceptually related to vantage-point representations such as isovists and LiDAR place descriptors such as Scan Context and RING (Benedikt, 1979; Kim & Kim, 2018; Lu et al., 2022). Unlike LiDAR descriptors, our image-only setting provides no direct range measurements. We therefore derive ground ring from estimated depth and learn to predict corresponding aerial rings from overhead imagery using ground-derived supervision.

LVLMs for geolocation and spatial reasoning. LVLMs have recently been explored for image geo-localization through semantic and geographic reasoning (Li et al., 2024; Jia et al., 2024; Li et al., 2025). However, fine-grained spatial reasoning remains challenging for current multimodal models (Wang et al., 2025; Yang et al., 2025). Recent cross-view studies further highlight challenges in establishing correspondence and maintaining spatial consistency across viewpoints (Jeong et al., 2026; Wang et al., 2026; Liu et al., 2026). Rather than relying on LVLMs to infer cross-view geometry directly from raw imagery, we use their predictive uncertainty for multi-hypothesis search and provide explicit depth-ring evidence for resolving look-alike locations.

## B VIGOR-PRO DATASET CONSTRUCTION

VIGOR-Pro turns an existing VIGOR panorama–aerial pair into a large-area localization example. The construction changes only the aerial context and coordinate frame, while preserving the original panorama and camera location. It introduces neither newly collected imagery nor manual location labels. Figure 5 illustrates the overall construction process.

1 Source aerial tile  
![](images/9092f10a600eb2e1d84530eeaa7f13057a84b2f26249f4fe6698ac0e4d2bc458.jpg)

2 Choose anchor  
![](images/86b4e04c553d1ef76649460a4854ad3a0de9484317eae3b1ba6c817175eac434.jpg)

3 Retrieve neighbors  
![](images/4df700b949b267b36d67b5197b61c9f731ec90a8a38d0132995c1753fac0288e.jpg)

4 Update label  
![](images/55e62b2e7d9b1db350cdd01bca727d9fa9236b427fa12dbf99f8b79f0cf6a9ef.jpg)  
Figure 5: VIGOR-Pro construction retains the original camera location while enlarging its aerial context. The source tile is assigned to a sampled lattice cell, geographic neighbors complete the overlapping mosaic, and the camera offset is transformed into the mosaic frame. The empty grid in Step 2 denotes logical tile slots; completed tiles overlap by half their width.

## B.1 CONSTRUCTION PROTOCOL

Tile lattice and mosaic geometry. VIGOR aerial tiles are $6 4 0 \times 6 4 0$ pixels with 50% overlap, hence a 320-pixel lattice stride. An n × n mosaic has native side length

$$
L _ { n } = 6 4 0 + 3 2 0 ( n - 1 ) .\tag{7}
$$

Tiles are pasted in a deterministic order at this stride. In overlap regions, later tiles overwrite earlier pixels; no interpolation or blending is applied. Thus every output pixel is copied from an original VIGOR tile.

Geographic retrieval and rejection. Each VIGOR query is associated with one positive and three semi-positive aerial tiles. We independently use each of these four tiles as a source anchor. For a query–source pair, the source tile is placed at a uniformly sampled lattice cell $( r , c ) \in \{ 0 , \ldots , n -$ $\mathrm { i } \} ^ { 2 }$ . Expected centers for the remaining cells are obtained by stepping on the city-specific VIGOR lattice. A spatial index returns the nearest tile, which is accepted only when the geodesic distance between its center and the expected center is below 10 m. A mosaic is valid only if all $n ^ { 2 }$ cells pass this test. The query is retained when at least one of its four source anchors yields a complete mosaic.

Coordinate transformation. Let $( \Delta x , \Delta y )$ denote the camera offset from the native aerial tile center, and let $q = ( n - 1 ) / 2$ . With image x increasing eastward and image y increasing southward, the offset in the native mosaic frame is

$$
\Delta x ^ { \prime } = \Delta x + 3 2 0 ( q - c ) , \qquad \Delta y ^ { \prime } = \Delta y + 3 2 0 ( r - q ) .\tag{8}
$$

We express the target in an image-size-independent $[ 0 , 1 0 0 0 ] ^ { 2 }$ coordinate system as

$$
x = 1 0 0 0 \left( { \frac { 1 } { 2 } } - { \frac { \Delta x ^ { \prime } } { L _ { n } } } \right) , \qquad y = 1 0 0 0 \left( { \frac { 1 } { 2 } } + { \frac { \Delta y ^ { \prime } } { L _ { n } } } \right) .\tag{9}
$$

The inverse used by evaluation is

$$
\Delta x ^ { \prime } = L _ { n } \left( { \frac { 1 } { 2 } } - { \frac { x } { 1 0 0 0 } } \right) , \qquad \Delta y ^ { \prime } = L _ { n } \left( { \frac { y } { 1 0 0 0 } } - { \frac { 1 } { 2 } } \right) .\tag{10}
$$

Equations (9) and (10) form an exact round trip before decimal rounding. The center of a mosaic maps to (500, 500). Algorithm 1 summarizes the complete construction procedure, including geographic retrieval, mosaic validation, stitching, and coordinate transformation.

Algorithm 1: Construct one query’s VIGOR-Pro examples   
Input: panorama $I _ { g } ;$ source tuples $\{ ( T _ { j } , \Delta x _ { j } , \Delta y _ { j } ) \} _ { j = 1 } ^ { 4 } \mathrm { . }$ ; grid size n.   
Output: zero or more complete mosaic–label pairs.   
for source anchor $j = 1 , \dots , 4$ do   
Sample anchor cell $( r _ { j } , c _ { j } )$ uniformly.   
For every cell, compute its expected geographic center and retrieve the nearest city tile.   
Reject anchor j if any center discrepancy is at least 10 m.   
Otherwise stitch at stride 320 and transform the label with Eqs. (8)–(9).   
end for   
Return all valid pairs; discard $I _ { g }$ only if none remain.

## B.2 BENCHMARK SCALES

We instantiate $n \in \{ 2 , 5 , 1 0 \}$ , yielding native mosaic resolutions $L _ { n } \in \{ 9 6 0 , 1 9 2 0 , 3 5 2 0 \}$ pixels. Using the city-specific ground sampling distance (GSD) reported by SliceMatch (Lentsch et al., 2023), the mean side lengths are approximately 106 m, 213 m, and 390 m, which we refer to as the 100, 200, and 400 m scales, respectively. The city-specific GSD values and corresponding physical side lengths are reported in Table 5, and representative examples are shown in Figure 8.

Dataset split. The benchmark inherits the official VIGOR query split. Neighbor retrieval is performed only within the query city and does not alter the original training/test city partition. All mosaics originating from the same panorama remain in the same split.

Dataset statistics. Table 6 reports statistics after complete-mosaic filtering. Original panoramas counts ground panoramas in the source VIGOR annotations, and original pairs counts their associated panorama–tile pairs before mosaic construction. Because each panorama has four source tiles, the 105,214 original panoramas yield 420,856 original pairs at every scale. Complete mosaics counts original pairs for which all $n ^ { 2 }$ aerial tiles are recovered. The completion rate is the number of complete mosaics divided by the number of original pairs. Retained panoramas counts unique ground panoramas with at least one complete mosaic.

Table 5: Aerial-region side lengths induced by the source GSD and native mosaic geometry. Values are side lengths in meters.
<table><tr><td>City</td><td>GSD (m/pixel)</td><td>100 m</td><td>200 m</td><td>400 m</td></tr><tr><td rowspan="4">New York Seattle San Francisco Chicago</td><td>0.113248</td><td>108.72</td><td>217.44</td><td>398.63</td></tr><tr><td>0.100817</td><td>96.78</td><td>193.57</td><td>354.88</td></tr><tr><td>0.118141</td><td>113.42</td><td>226.83</td><td>415.86</td></tr><tr><td>0.111262</td><td>106.81</td><td>213.62</td><td>391.64</td></tr><tr><td>Mean</td><td></td><td>106.43</td><td>212.87</td><td>390.25</td></tr></table>

Table 6: VIGOR-Pro geometry and construction statistics after complete-mosaic filtering.
<table><tr><td>Scale</td><td>Grid n × n</td><td>Native resolution</td><td>Mean side (m)</td><td>Original panoramas</td><td>Original pairs</td><td>Complete mosaics</td><td>Completion rate (%)</td><td>Retained panoramas</td></tr><tr><td>100 m</td><td>2 × 2</td><td>9602</td><td>106</td><td>105,214</td><td>420,856</td><td>417,375</td><td>99.17</td><td>105,214</td></tr><tr><td>200m</td><td>5×5</td><td>19202</td><td>213</td><td>105,214</td><td>420,856</td><td>400,740</td><td>95.22</td><td>104,098</td></tr><tr><td>400 m</td><td>10 × 10</td><td>35202</td><td>390</td><td>105,214</td><td>420,856</td><td>372,532</td><td>88.52</td><td>101,053</td></tr></table>

## C IMPLEMENTATION DETAILS

This section describes the training and inference details of the multi-scale search model, depth-ring predictor, and learned verifier, including their supervision, inputs, and optimization settings.

## C.1 SEARCH MODEL: TRAINING AND INFERENCE

Supervision format. Each supervised example contains exactly two images in a fixed order—the ground panorama followed by the aerial region—and one user/assistant turn. The template below gives both roles; the two image inputs are inserted in this order.

User   
<image><image>   
Given a ground-level panoramic image and a satellite image, please   
predict the coordinates [x, y] of the ground-level camera location in   
the satellite image.   
Assistant   
[x, y]

The assistant target is the bare string [x, y], with each value rounded to one decimal place in the normalized coordinate system of Eq. (9); no explanation or auxiliary token is appended. Keeping the same coordinate grammar at every scale permits one model to operate throughout the tree.

Multi-scale training set. Separate search models are trained for same-area and cross-area evaluation using their respective training splits. The training mixture contains direct VIGOR-Pro examples with approximately 100, 200, and 400 m coverage. We additionally introduce a 50 m scale to provide finer-grained supervision for the search stage. These examples are derived by cropping 1/2-scale regions from the 100 m training examples. To simulate realistic search refinement, the crop center is displaced from the ground-truth location using independent zero-mean Gaussian jitter with a standard deviation equal to 0.1 of the crop half-size. The jitter is truncated to ensure that the ground truth location remains inside the crop whenever the parent label lies within the image. At an image boundary, the crop is translated inward rather than reduced in size. Let $( a _ { x } , a _ { y } , b _ { x } , b _ { y } )$ denote the crop box in the normalized parent-image coordinates, and (x, y) the ground-truth camera location. The in-crop label is $x ^ { \prime } = 1 0 \dot { 0 } 0 ( x - a _ { x } \big ) / ( b _ { x } - a _ { x } )$ and $y ^ { \prime } \overset { } { = } 1 0 0 0 ( y \overset { - } { - } a _ { y } ) / ( b _ { y } - a _ { y } )$

Table 7: Search-model training configuration.
<table><tr><td>Configuration</td><td>Value</td><td>Configuration</td><td>Value</td></tr><tr><td>Initialization</td><td>Qwen3-VL-4B-Instruct</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Trainable</td><td>Language model</td><td>Batch size</td><td>256</td></tr><tr><td>Frozen</td><td>Vision, projector</td><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Training scales</td><td> $5 0 / 1 0 0 / 2 0 0 / 4 0 0 \mathrm { m }$ </td><td>Model input</td><td> $4 8 0 ^ { 2 } / 5 1 2 ^ { 2 } / 5 1 2 ^ { 2 } / 7 6 8 ^ { 2 } \mathrm { p x }$ </td></tr><tr><td>Training epochs</td><td>2</td><td>LR schedule</td><td>Cosine</td></tr><tr><td>Warmup ratio</td><td>0.1</td><td></td><td></td></tr></table>

Table 8: Branch-and-zoom inference schedule. The search-region schedule denotes the approximate side length of the aerial region presented at each level.
<table><tr><td>Initial scale</td><td>Search-region schedule (m)</td><td>Model-input size (px)</td></tr><tr><td>100 m</td><td> $1 0 0  5 0 $ </td><td> $5 1 2  4 8 0$ </td></tr><tr><td>200 m</td><td> $2 0 0 \to 1 0 0 \to 5 0$ </td><td> $5 1 2  5 1 2  4 8 0$ </td></tr><tr><td>400 m</td><td> $4 0 0 {  } 2 0 0 {  } 1 0 0 {  } 5 0$ </td><td> $7 6 8 \to 5 1 2 \to 5 1 2 \to 4 8 0$ </td></tr></table>

Optimization. We initialize from Qwen3-VL-4B-Instruct. During training, only the language model parameters are updated, while the vision tower and multimodal projector remain frozen. All models are trained on 8× NVIDIA A100 (80GB) GPUs. Table 7 summarizes the optimization configuration.

Branch-and-zoom inference. Table 8 summarizes the branch-and-zoom schedule used at each evaluation scale. At each node, we draw M textual coordinate samples. The coordinates are clustered by single linkage in the normalized crop-local coordinate system, using a Euclidean distance threshold of 50. Clusters with fewer than two members are discarded, and at most the three largest clusters are retained. Each retained cluster is represented by its centroid, which defines a child hypothesis. After expanding all active nodes at each level, we rank the resulting hypotheses using the path-aggregated cluster-size score defined in Section 3.2 and retain at most B for further expansion. For each retained hypothesis, we form the corresponding child search region and extract its aerial crop directly from the original mosaic. If the crop would exceed an image boundary, the search region is translated inward without changing its size. To express predictions from different search regions in a common coordinate frame, we record each region as $( a _ { x } , a _ { y } , b _ { x } , b _ { y } )$ in the normalized root-mosaic coordinate system. A local prediction $\mathbf { u } = ( u _ { x } , u _ { y } )$ is then mapped back to the root frame as

$$
\mathcal { T } ( \mathbf { u } ) = \Big ( a _ { x } + \frac { u _ { x } } { 1 0 0 0 } ( b _ { x } - a _ { x } ) , a _ { y } + \frac { u _ { y } } { 1 0 0 0 } ( b _ { y } - a _ { y } ) \Big ) .\tag{11}
$$

After the final level, the leaf hypotheses are ranked by the same score, and the top-k are retained for verification.

## C.2 DEPTH RING: CONSTRUCTION, LEARNING, AND INFERENCE

Ground panorama to depth ring. VIGOR panoramas are oriented with the center facing north. We sample the depth ring at A uniformly spaced clockwise azimuths $\phi _ { a } = 3 6 0 ^ { \circ } ( a - 1 ) / A$ , starting from the leftmost panorama direction as $\phi = 0 ^ { \circ }$ . Since Depth Anything V2 is designed for perspective images, we first apply equirectangular-to-perspective (E2P) projection to render four $5 \mathrm { { i } 2 \times \mathrm { { 5 1 2 } } }$ rectilinear views with view directions $\psi _ { j } \in \{ \dot { 0 } ^ { \circ } , \dot { 9 } 0 ^ { \circ } , 1 8 0 ^ { \circ } , 2 7 0 ^ { \circ } \}$ , pitch $0 ^ { \circ }$ , and horizontal field of view 90<sup>◦</sup>. We then apply Depth Anything V2 Metric Outdoor Large to each view independently. Eq. (5) is implemented by mapping each perspective-view column back to its panorama azimuth. For a view of width $W _ { f }$ and focal length $\bar { f } = \bar { W } _ { f } / [ 2 \tan ( 4 5 ^ { \circ } ) ]$ ], column x of view j corresponds to

$$
\phi _ { j } ( x ) = \left[ { \psi _ { j } } + \tan ^ { - 1 } \left( \frac { x - W _ { f } / 2 } { f } \right) \right] \bmod { 3 6 0 ^ { \circ } } .\tag{12}
$$

For each target azimuth $\phi _ { a } .$ , we pool the estimated metric depth values in the corresponding horizon band to obtain $r _ { g } ( \phi _ { a } )$ as in Eq. (5). For robustness to isolated depth artifacts, each perspective-view column is represented by the median estimated depth within a ±8-row band around the horizontal center. After mapping the samples from all four views to panorama azimuths using Eq. (12), we aggregate them into the A uniformly spaced azimuths of the depth ring, averaging samples assigned to the same azimuth. The Metric Outdoor Large model uses an 80 m depth range, preserving a common nominal metric scale across panoramas. Figure 6 illustrates the conversion from the panorama to the four perspective views and the resulting depth ring construction.

![](images/97eacb8f7e1596e4bdf108c19d36a25c6d61ae3870698ac15b91109a09ccc045.jpg)  
Figure 6: Ground depth-ring construction. The panorama is projected into four rectilinear views using equirectangular-to-perspective (E2P) projection. Depth Anything V2 Metric Outdoor Large estimates depth for each view, from which we pool the highlighted horizon band and map the resulting samples to their world azimuths. Samples from all views are then aggregated into the A uniformly spaced azimuths of the ground depth ring.

![](images/8b53198235a95e6b2d38e12620b460d5b730124a84540d38ae1a7b1a66f73600.jpg)  
Figure 7: Architecture of the aerial-to-ring predictor. DINOv2 and DPT encode the local aerial patch into a dense feature map. The candidate location p conditions the A azimuth queries through FiLM modulation. For each azimuth, a directional beam is sampled from p, and the corresponding query attends only to features along that beam. Each updated query predicts a metric-depth class, producing the aerial depth ring over all azimuths. During training, the predictor is supervised by the ground depth ring extracted from the corresponding panorama.

Training recipe of the aerial-to-ring predictor. Figure 7 illustrates the architecture of the aerialto-ring predictor. It uses a frozen DINOv2-Base encoder (Oquab et al., 2023) with multi-level features from transformer blocks 3, 6, 9, and 12, which are fused by a DPT decoder (Ranftl et al., 2021) into a dense aerial feature map. It maintains A learned azimuth queries. The candidate coordinate p, represented in the normalized local-patch coordinate system, is encoded by an MLP and used to condition these queries through feature-wise linear modulation (FiLM) (Perez et al., 2018), while its mapped feature-map location serves as the origin for directional sampling. For each azimuth $\phi _ { a }$ , we sample a thin directional beam from the aerial feature map starting at p to exclude irrelevant context outside that direction. Each azimuth query is then updated by four cross-attention blocks using only the features sampled along its corresponding beam. The prediction head maps each query to logits $z _ { a , k }$ over K metric-depth classes corresponding to uniformly spaced depth values in [0, 80] m. Each ground-ring value is quantized to its nearest metric-depth class and used as the cross-entropy target. The predictor is supervised by the ground depth ring constructed above and trained with azimuth-wise cross-entropy.

Table 9: Aerial-to-ring model and training configuration.
<table><tr><td>Configuration</td><td>Value</td><td>Configuration</td><td>Value</td></tr><tr><td>Model</td><td></td><td></td><td></td></tr><tr><td>Aerial patch</td><td> $6 4 0 ^ { 2 }$ </td><td>FFN width</td><td>1,024</td></tr><tr><td>Azimuth samples A</td><td>360</td><td>Depth classes K</td><td>80</td></tr><tr><td>Metric depth range</td><td>[0, 80] m</td><td>Backbone</td><td>frozen DINOv2-B</td></tr><tr><td>DPT output width</td><td>256</td><td>Cross-attn. layers / heads</td><td>4/8</td></tr><tr><td>Training</td><td></td><td></td><td></td></tr><tr><td>Training loss</td><td>azimuth-wise CE</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Batch size</td><td>64</td><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td>0.05</td><td>Warmup</td><td>500 steps</td></tr><tr><td>LR schedule</td><td>cosine</td><td>Training epochs</td><td>1</td></tr></table>

Training uses the VIGOR-Pro training sets at the 100, 200, and 400 m scales, with separate predictors trained for same-area and cross-area evaluation using their respective training splits. Each panorama–aerial pair provides one example. From each aerial mosaic, we extract a 640 × 640 local patch around the annotated camera location. At image boundaries, the crop is shifted inward while preserving its size, and the camera coordinate is updated accordingly.

At inference, the same aerial preprocessing is applied to each search hypothesis. For each azimuth, we select the highest-scoring depth class and use its corresponding depth value as the prediction.

Table 9 summarizes the training and model configuration of the aerial-to-ring predictor.

## C.3 VERIFIER: HARD-CANDIDATE TRAINING AND INFERENCE

Hard-candidate training set. Separate verifiers are trained for same-area and cross-area evaluation using their respective training splits. Hard candidates from the 100, 200, and 400 m scales are combined for verifier training. For each ground–aerial pair in the training split, we run the same branch-and-zoom procedure used at test time, with M = 16 stochastic predictions per node, sampling temperature 1, pruning budget $B = 4$ , and the corresponding zoom schedule. We then take the two highest-scoring hypotheses at the final search level. We discard examples for which fewer than two valid hypotheses are available or the two candidates are separated by less than 10 m. The latter removes near-duplicate alternatives that provide little useful selection supervision. For each retained pair, the two hypotheses are randomly assigned to Candidate A and Candidate B, preventing the verifier from exploiting candidate order. The candidate with the smaller Euclidean distance to the ground-truth camera position is used as the supervision target. Importantly, the competing candidate is generated by the search model itself rather than independently sampled, exposing the verifier to plausible hard alternatives that remain after search.

Multimodal input and supervision. Each verifier example contains six images in a fixed order: (1) the ground panorama; (2) the full aerial region with Candidates A and B marked; (3–4) local aerial patches around Candidates A and B; and (5–6) candidate-specific depth-ring comparisons. For each candidate, the ground depth ring and the predicted aerial ring are rendered together in a common polar representation. The two candidate-specific comparisons use the same radial scale, enabling direct comparison of their geometric consistency. The candidate coordinates in the root aerial region are additionally provided in the normalized coordinate system. The supervision target contains only \boxed{A} or \boxed{B}, without an auxiliary rationale or coordinate-regression target. We provide the complete prompt in Figure 13.

Optimization and inference. We initialize the verifier from the trained search model, retaining the cross-view representations learned during multi-scale search training. As in search-model training, the vision tower and multimodal projector are frozen, and only the language-model parameters are updated. All models are trained on 8× NVIDIA A100 (80GB) GPUs. Table 10 summarizes the optimization configuration.

At inference, branch-and-zoom search first returns the two highest-scoring final hypotheses. We apply verification only when the two candidates are at least 10 m apart; otherwise, the top-ranked search hypothesis is retained as the final prediction. For verified samples, we construct the corresponding verifier input. The verifier performs greedy decoding with temperature 0 and selects Candidate A or B. The selected hypothesis, expressed in the root aerial coordinate frame, is returned directly as the final localization result.

Table 10: Verifier training configuration.
<table><tr><td>Configuration</td><td>Value</td><td>Configuration</td><td>Value</td></tr><tr><td>Initialization Trainable</td><td>Search model</td><td>Training epochs</td><td>2</td></tr><tr><td></td><td>Language model</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Frozen</td><td>Vision, projector</td><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Batch size</td><td>256</td><td>LR schedule</td><td>Cosine</td></tr><tr><td>Warmup ratio</td><td>0.1</td><td>Candidate shortlist 2</td><td></td></tr><tr><td>Candidate crop</td><td> $6 4 0 \times 6 4 0$ </td><td></td><td></td></tr></table>

## D BASELINE ADAPTATION AND EVALUATION PROTOCOLS

Common evaluation protocol. For each VIGOR-Pro scale and split, all methods are evaluated on the same retained examples and receive the same underlying geographic prior defined by the corresponding aerial mosaic. End-to-end and search-based methods operate on the complete contiguous mosaic, whereas retrieval-based methods search over the original VIGOR tiles composing the same region. All predictions are mapped to the native mosaic coordinate frame before evaluation and converted to metric errors using the city-specific GSD in Table 5.

End-to-end localization. HC-Net (Wang et al., 2023), CCVPE (Xia et al., 2024), and $\mathrm { F G ^ { 2 } }$ (Xia & Alahi, 2025) are retrained independently at the 100, 200, and 400 m scales of VIGOR-Pro. At each scale, the complete aerial mosaic is provided as input and resized according to the preprocessing of the corresponding official implementation. Camera labels and model predictions are transformed consistently between the resized input and the native mosaic coordinate frame. Apart from these scale-dependent adaptations, we follow the official architectures, objectives, training recipes, and inference settings. For Qwen3-VL-4B, we directly evaluate our fine-tuned search model by using its single-pass prediction without applying the search procedure.

Retrieval-based localization. For each VIGOR-Pro example, the retrieval gallery consists only of the original VIGOR tiles composing the corresponding mosaic, rather than the full VIGOR gallery. This preserves the native tile scale used by the retrieval models while restricting the search to the same geographic region available to the other localization paradigms. For Sample4Geo (Deuser et al., 2023), we use the officially released checkpoint and inference configuration, taking the center of the top-ranked tile as the localization prediction. For Sample4Geo+ $\mathrm { F } \bar { \mathrm { G } } ^ { 2 }$ , the same retrieved tile is further processed by an $\mathrm { F G ^ { 2 } }$ model fine-tuned on the original VIGOR dataset for within-tile localization, and the resulting prediction is mapped to the native mosaic coordinate frame according to the tile position. For CIPER (Jeon et al., 2026), we likewise use the officially released checkpoint and inference configuration on the same gallery, and map the pose prediction associated with the selected aerial tile to the native mosaic coordinate frame.

Our search-based framework. Single-Hypothesis uses the same trained search model and inference configuration as RingMatch, but retains only the highest-scoring hypothesis at each search level for subsequent expansion. We also report RingMatch (Search-only), which follows the same search procedure but directly selects the top-ranked hypotheses without applying verification.

## E EXPERIMENTAL DETAILS FOR ADDITIONAL ANALYSES

## E.1 ZERO-SHOT CROSS-VIEW SPATIAL REASONING

We evaluate zero-shot cross-view spatial reasoning on the VIGOR-Pro same-area verification samples at the 200 and 400 m scales. Following the verification protocol in Appendix C.3, the two highest-scoring final search hypotheses form Candidates A and B, and samples with candidate separation below 10 m are excluded. The candidate closer to the ground-truth camera location is treated as the correct answer. We evaluate GPT-5.6 Terra and Gemini-3.7 Flash without task-specific finetuning. For each model, we compare two input settings on the same candidate pairs. In the visualonly setting, the model receives the ground panorama, the full aerial region with both candidates marked, and a local aerial patch centered on each candidate, using the prompt in Figure 14. The depth-ring setting additionally provides the ground ring paired with the predicted aerial ring for each candidate, together with the corresponding instructions in Figure 13. We report candidate selection accuracy. As a direct geometric baseline, Ring-only selects the candidate whose predicted aerial ring has the higher Pearson correlation with the ground ring. We also report the verifier in RingMatch, which is fine-tuned for this task, on the same samples for reference.

## E.2 GENERALIZATION BEYOND VIGOR

Dataset construction. We construct an additional evaluation set of 4,000 panorama–aerial pairs from four cities outside the VIGOR benchmark: London, Tokyo, São Paulo, and Cape Town, with 1,000 examples per city. Ground panoramas are collected from Google Street View, sampled at least 75 m apart, and converted to equirectangular images with the central viewing direction point ing north. Aerial imagery is collected through the Google Maps Static API. Following the same 5 × 5 mosaic geometry used at the 200 m VIGOR-Pro scale, each aerial region is composed of 25 overlapping 640 × 640 satellite images queried at zoom level 20 with 50% overlap, yielding a 1920 × 1920 mosaic. To avoid center bias, the camera-containing tile is evenly distributed across the 25 grid positions, with 40 examples per position in each city. No examples from these cities are used for task-specific training.

Evaluation protocol. We evaluate FG<sup>2</sup>, Sample4Geo+FG<sup>2</sup> and Ring-Match without target-city fine-tuning or adaptation, using models trained under the VIGOR same-area setting. Localization errors are measured in the native mosaic coordinate system and converted to meters using the corresponding city-specific ground sampling distances, which are summarized in Table 11.

Table 11: Aerial-region side lengths induced by the cityspecific GSD and the native mosaic geometry.
<table><tr><td>City</td><td>GSD (m/pixel)</td><td>Side length (m)</td></tr><tr><td>London</td><td>0.092883</td><td>178.33</td></tr><tr><td>Tokyo</td><td>0.121224</td><td>232.75</td></tr><tr><td>São Paulo</td><td>0.136834</td><td>262.72</td></tr><tr><td>Cape Town</td><td>0.123851</td><td>237.79</td></tr><tr><td>Mean</td><td></td><td>227.90</td></tr></table>

## F QUALITATIVE ANALYSIS OF DEPTH RINGS

Figure 9 provides qualitative examples of depth rings for cross-view matching. Across diverse scene layouts, aerial rings predicted near the ground-truth camera location preserve the characteristic spatial structure of the corresponding ground ring, including its dominant directions, relative extents, and local shape variations. In contrast, rings predicted at incorrect locations exhibit clear structural discrepancies, even when the candidate aerial regions contain similar roads or surrounding buildings. These examples illustrate how the depth ring maps ground and aerial observations into a shared representation of surrounding spatial layout, providing discriminative geometric evidence beyond direct cross-view appearance matching.

## G EXTENDED ANALYSIS OF MULTI-HYPOTHESIS SEARCH

Figures 10–12 extend the search analysis in Figure 4 to all VIGOR-Pro scales under both same-area and cross-area evaluation.

Search level. Increasing the number of search levels becomes increasingly important as the aerial region expands. At 400 m and k = 2, increasing L from 1 to 4 reduces the mean oracle error from 12.49 to 10.43 m on same-area and from 19.26 to 15.64 m on cross-area; the corresponding median errors decrease from 3.05 to 1.76 m and from 4.68 to 1.99 m. The same trend holds at smaller scales, but the gains become more pronounced over broader regions, supporting the need for iterative zooming when the initial localization prior is coarse.

Sampling count. Increasing M provides the largest gain from 8 to 16, with further sampling yielding diminishing improvements. At 400 m and k = 2, the same-area mean error decreases from 12.94 to 10.43 m and further to 9.54 m at M = 32, while cross-area error decreases from 18.29 to 15.64 m and 15.17 m, respectively. Median errors change much less and are not consistently improved by larger M, suggesting that additional samples primarily help recover difficult cases by exposing alternative plausible modes.

Pruning budget. A small budget of B = 2 degrades search quality, particularly at larger scales, whereas B = 4 and 8 yield similar top-2 performance across both evaluation settings. Larger budgets become more beneficial mainly when considering deeper candidate lists. These results indicate that once enough competing branches are retained, search is comparatively insensitive to the exact pruning budget, supporting our default choice of B = 4 for the compact top-2 shortlist used by the verifier.

![](images/c73697fc7b5cf0882fe02ca15364730041c0ecaa452793269b114a01bacb2137.jpg)  
Figure 8: Examples of VIGOR-Pro at the 400 m scale. Each row compares an original VIGOR aerial tile (left) with the corresponding VIGOR-Pro 10 × 10 mosaic (right). The orange point denotes the ground-camera location, and the dashed orange box indicates the footprint of the original tile within the enlarged aerial region.

Ground view

![](images/505fd41dafd19087f21457fbdc78e226658eac60a136b2a9f751ff8b6211f8a7.jpg)

Candidate A  
![](images/e2ed55ffbbea82b8ea650d069d7fb0c14ecd508e449486e57b1f1ce78f891e04.jpg)

Candidate B  
![](images/ef25ff7ff6153a2a163c0a0048c65878d01b852f1b27072672537d24f09fe703.jpg)

![](images/4ad920e63cd9e22f7808dde4a1659b1ab5975063e5faf14f699eec944f967ba2.jpg)

![](images/304916a0ecf61536b4d3879f923edb9a1896b296691d94ab61583c689f9f814c.jpg)  
(a)

![](images/c876d9f89bbaebc7917b90b9e067419ab65d731e173f0de7e56ad09f428e05ab.jpg)

Ground view  
Candidate A  
![](images/acd47764453cc4371e3f117d9140476156abdb50c9bc2e37783343f9ba8d4f80.jpg)

![](images/f0391635d68caa0a3969b60e75e28a56db72afa05bd1a3674d1cc92ebd8c1b88.jpg)

Candidate B  
![](images/241daadccbe5c93463297f165ef14ab14d4937f91dd44f44fbdb2d8e3088344c.jpg)

![](images/6e4c18a64c5839ca6ce4898b20eba27f73321ce6efbe824c5a8cd647b848f602.jpg)

![](images/3d8dcfa8905e3162c8ddc60f451db5da7ea4db62e95cf89be289f8f3f34b506e.jpg)  
(b)

![](images/081d60639e8efebd4b12b3eb2f0de67164971feaa54d315b70c726ca9264b501.jpg)  
Figure 9: Qualitative examples of depth rings for cross-view matching. Each panel compares the ground-view depth ring with predicted aerial depth rings at two candidate locations. Candidates closer to the ground-truth camera location exhibit stronger correspondence in surrounding spatial layout, while incorrect candidates show clear geometric discrepancies.

Ground view

![](images/3d1a6eb9a0a7fe458aa4dedbe1c48b88a93a0a03d62e135994ad01efeb0a9d8e.jpg)

Candidate A  
![](images/045768d31fbffcd262c376b7386d69a5994f8b8594a33efa4eb99dc4f6986959.jpg)

Candidate B  
![](images/bf279269871a9b2de9d655fa7a9cc311a883e1afc17d883eed0d3b922faef57e.jpg)

![](images/8972ac430e8e67abf4a7bf6585b6a005ab7cfe89a56bf7941e30fa341d46ff7e.jpg)

![](images/c0cb125b3e3751ed07cf318f4727fd993e8365d22bb65bdafc343c3359b62d2f.jpg)  
(c)

![](images/a7863eac36f6045e4d8354a2b9463df66e2fee2614f617067a37d7fb720ee017.jpg)

Ground view  
Candidate A  
![](images/140f3efc4a39c08237f9d6f0a9623b68b1043003c642ac8860fb554afafe782d.jpg)

![](images/e934e0fd01cf3b45d30dac6a990059f6d93c992a3132bc7268e690f9c9872059.jpg)

Candidate B  
![](images/f3c2fe566c9122047ec05af3d4ac52126388421fd07567b5e860c898e9fa679c.jpg)

![](images/042497633fc96f79cc4450a07212560337f315194439ac5ebdb20524e399439a.jpg)

![](images/2cd176479c5c793313d8ba93c3b8aa5eb875195e6861e0304fe5bd59d7c728af.jpg)  
(d)

![](images/0c77a4aa163351181f329126885c357136c32ca6bf20184300877337e2f2236b.jpg)  
Figure 9: Qualitative examples of depth rings for cross-view matching (continued).

![](images/eb7ace33986993834e3dd04a59ad529473cf24cb8d1924c8450cef3553d1646d.jpg)  
(b) Cross-area  
Figure 10: Sensitivity to search level L. Columns correspond to the 100, 200, and 400 m scales, which contain two, three, and four search levels, respectively; rows report mean and median top-k oracle error. Results are reported for both same-area and cross-area evaluation at all three scales.

![](images/cae33a8cac2f8d27c792b6e540b329960d646edecbe1fb8b38a9481878763bba.jpg)  
(b) Cross-area  
Figure 11: Sensitivity to sampling count M. Columns correspond to the 100, 200, and 400 m scales; rows report mean and median top-k oracle error. Results are reported for both same-area and crossarea evaluation at all three scales.

![](images/4fc6ccfcac38d0c27b37938007cdd0a2e33445682adc10394f4dda0dc0a29fc1.jpg)  
(b) Cross-area  
Figure 12: Sensitivity to pruning budget B. Columns correspond to the 100, 200, and 400 m scales; rows report mean and median top-k oracle error. Results are reported for both same-area and crossarea evaluation at all three scales.

![](images/5e7da4e0a0704135181e4f7a257634ad1d7b651343172b62c66806226858d4a8.jpg)  
Figure 13: Prompt for verification with depth-ring inputs.

![](images/348db132398c57b80d7402f960158f09f6b95effa560f934fdac28bc9658baaa.jpg)  
Figure 14: Prompt for verification without depth-ring inputs.