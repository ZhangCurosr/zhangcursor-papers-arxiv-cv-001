# CLAD: Keypoint Matching Using Global Line Correspondences

Masaki Satoh

Morpho, Inc.

Tokyo, Japan

m-satoh@morphoinc.com

Abstract—Traditional keypoint descriptors, such as SIFT (Scale-Invariant Feature Transform) and ORB (Oriented FAST and Rotated BRIEF), are computed from local image patches. They may be unreliable in scenes with repetitive patterns or weak textures, where patches at different locations can look similar. We propose the Contextual Line-Assisted Descriptor (CLAD), a keypoint matching method that addresses this problem by using global geometric context from line correspondences. CLAD represents each keypoint with a vector whose elements indicate its proximity to matched lines. It can therefore use the keypoint’s spatial relationships to the lines in the scene. We use HOME (Hough-space One-dimensional Matching of Extrema) to detect both clear and weak linear structures and to establish enough line-to-keypoint associations to represent this context. Evaluations on the TUM RGB-D SLAM dataset and outdoor video sequences show that CLAD provides geometrically consistent matches that complement those obtained with traditional local descriptors. In repetitive scenes, where local descriptors may not distinguish similar patches, CLAD uses global structure to improve matching reliability. The method is computationally lightweight and practical to implement.

## I. INTRODUCTION

Keypoint detection and matching are fundamental to many computer vision applications, including image registration, 3D reconstruction, and simultaneous localization and mapping (SLAM). A keypoint detector identifies distinctive, repeatable points in an image, such as corners and blobs. A matching algorithm then finds correspondences between images by comparing the feature vectors, or descriptors, associated with the keypoints. The number of correct matches directly affects the accuracy and robustness of later processing. We can increase this number either by improving the detector to find more repeatable points or by improving the descriptors to make matching more reliable. This paper focuses on the second approach.

In standard methods, such as the Scale-Invariant Feature Transform (SIFT) [1] and Oriented FAST and Rotated BRIEF (ORB) [2], each keypoint descriptor is computed only from the local image patch around that keypoint. Local information alone may not distinguish keypoints at different locations if their patches look similar, making it difficult to match such keypoints correctly. Even in these cases, human observers can often establish correct correspondences by considering the scene as a whole and the geometric relationships between keypoints. For example, standard descriptors may not distinguish individual checkerboard corners, as illustrated in Fig. 1, whereas a human observer can identify them from the overall grid pattern.

![](images/1c64b18025ea5e8a4381781564a5322732d56b9b35c8c27fcf2f0851ef28c4a1.jpg)  
Fig. 1. Examples of keypoint matches obtained with ORB (top) and CLAD (bottom). Each pair of corresponding keypoints and its connecting line are shown in the same color. These examples illustrate a limitation of descriptors based only on local image information: in challenging scenes, many local patches may look alike, causing methods such as ORB to confuse different keypoints. In contrast, CLAD uses the global geometric context provided by line correspondences, which helps it find more reliable matches in these cases.

We propose CLAD (Contextual Line-Assisted Descriptor), a method that uses line correspondences to capture global geometric structure and improve keypoint matching. The following simplified example illustrates the main idea. Suppose that we establish line correspondences between two images, where line $\ell _ { i }$ in the first image corresponds to line $\ell _ { i } ^ { \prime }$ in the second. Let N be the number of line correspondences. We represent the jth keypoint in the first image and the kth keypoint in the second image with N-dimensional binary vectors $\boldsymbol { v } _ { j }$ and $\pmb { v } _ { k } ^ { \prime }$ , respectively. The ith element of $\boldsymbol { v } _ { j }$ is 1 if $\ell _ { i }$ passes through the $j \mathrm { t h }$ keypoint in the first image and 0 otherwise. We define $\pmb { v } _ { k } ^ { \prime }$ in the same way for the kth keypoint in the second image. These binary descriptors thus encode the global linear context. We use the Hamming distance to compare CLAD descriptors and match keypoints, as in standard methods. Fig. 1 shows that CLAD can match keypoints in challenging scenes where local descriptors such as ORB may fail.

## II. METHOD

CLAD matches input keypoints by using global context derived from line correspondences. This section first defines the CLAD descriptor and then describes the keypoint detector and the line detection and matching algorithms used in our implementation. In principle, CLAD can be used with any keypoint detector and any line detection and matching algorithm.

## A. Contextual Line-Assisted Descriptor

For $i = 1 , 2 , \dots , N$ , line $\ell _ { i } = ( a _ { i } , b _ { i } , c _ { i } ) ^ { \mathrm { T } }$ in the first image corresponds to line $\ell _ { i } ^ { \prime } = ( a _ { i } ^ { \prime } , b _ { i } ^ { \prime } , c _ { i } ^ { \prime } ) ^ { \mathrm { T } }$ in the second image. The distance from a keypoint x in the first image to line $\ell _ { i }$ is

$$
d _ { i } ( \pmb { x } ) = \frac { \lvert a _ { i } \ b { x } + b _ { i } \ b { y } + c _ { i } \rvert } { \sqrt { a _ { i } ^ { 2 } + b _ { i } ^ { 2 } } } .\tag{1}
$$

For this keypoint, we define an N-dimensional binary descriptor vector v(x) as

$$
\pmb { v } ( \pmb { x } ) \equiv ( \psi ( d _ { 1 } ( \pmb { x } ) ) , \psi ( d _ { 2 } ( \pmb { x } ) ) , \dots , \psi ( d _ { N } ( \pmb { x } ) ) ) ^ { \mathrm { T } } ,\tag{2}
$$

where $\psi ( d )$ indicates whether the keypoint is near the line: it is 1 for small distances and 0 for large distances. We use the parameter $\sigma _ { \psi }$ to set the size of the proximity region, such that $\psi ( d ) = 1$ when $d < \sigma _ { \psi }$ and $\psi ( d ) = 0$ otherwise. For a keypoint x, we call the set of lines satisfying $\psi ( d _ { i } ( { \pmb x } ) ) = 1$ its associated lines. We define the descriptor for a keypoint $\mathbf { x } ^ { \prime }$ in the second image in the same way.

For keypoints x and $\mathbf { x } ^ { \prime }$ in the first and second images to match, corresponding lines should have similar proximity relationships to the two keypoints. That is, if line $\ell _ { i }$ passes near x, its counterpart $\ell _ { i } ^ { \prime }$ should also pass near $\mathbf { { \boldsymbol { x } } ^ { \prime } } ;$ if it does not pass near x, $\ell _ { i } ^ { \prime }$ should not pass near $\mathbf { x } ^ { \prime }$ either. Thus, a sufficient number of descriptor elements should agree. We compute the Hamming distance between descriptors and use it for keypoint matching. For efficient storage and computation, we pad the descriptors with zeros so that their lengths are multiples of 8.

## B. Keypoint Detection

As noted above, CLAD can be used with any keypoint detector. We consider the Features from Accelerated Segment Test (FAST) [3] for two reasons. First, it is very fast and is therefore suitable for many applications. Second, a singlescale detector such as FAST avoids assigning scale information that CLAD does not need. Multiscale detection, as used by SIFT, is especially useful when images differ substantially in scale. However, CLAD does not require it, and scale estimates may instead introduce noise by placing keypoints close to one another. Although CLAD could also be applied to multiscale keypoints, we leave this for future work. In our implementation, however, we use ORB rather than FAST for keypoint detection. ORB’s keypoint detector is based on FAST, so the detected keypoints are similar. Using ORB keypoints also lets us compare ORB and CLAD on the same set of keypoints. We disable ORB’s multiscale feature detection.

## C. Line Detection and Matching

For CLAD to match keypoints successfully, we need enough line correspondences between the images. A standard approach to line detection and matching is to detect line segments with the Line Segment Detector (LSD) [4] and match them with the Line Band Descriptor (LBD) [5]. LSD efficiently detects straight line segments in an image, and LBD computes a robust descriptor for each segment and uses it to find matches across images. For CLAD, we can extend LSD-detected segments to infinite lines and use their LBD correspondences. LSD and LBD are suitable for many applications, including SLAM. However, they are not ideal for CLAD because the detected segments represent only a limited part of the scene’s global linear context. In addition, LSD can be computationally expensive, which may limit its use in real-time applications.

Recently, Hough-space-based methods have been proposed for line detection and matching [6], [7]. One such method is HOME [7]. It is lightweight compared with LSD and LBD and is therefore suitable for real-time applications. HOME is based on the texture-preserving Hough transform [6], a continuous version of the Hough transform. Unlike methods that detect only visible line segments, this transform accumulates image information along lines. It can therefore respond to weak linear structures even when they are not visible as continuous line segments. HOME can thus capture weak and implicit structures that may help represent global context in CLAD. We therefore use HOME in this paper. In Section III, we compare LSD and HOME in terms of how much line-based context each method provides to CLAD.

## III. EXPERIMENTS

The main purpose of our experiments is to show that CLAD provides information that complements traditional keypoint descriptors, such as ORB and SIFT, in keypoint matching. We do not claim that CLAD can replace these descriptors.

Before describing the experiments, we specify the parameters used in our method. For ORB, we use OpenCV’s default parameters, except that we disable multiscale feature detection. For HOME, we use the parameters from the original paper. We set the proximity-region size to $\sigma _ { \psi } = 4 .$ . To improve matching reliability, we use Lowe’s ratio test [1] with a threshold of 0.75 for ORB, HOME, and CLAD matching. For each keypoint in the first image, we apply the ratio test to its best and secondbest matches in the second image.

Because HOME is designed to process video frames, we use video sequences in our experiments. First, to illustrate the importance of global context, we use three checkerboard sequences from the TUM RGB-D SLAM dataset [8]: “freiburg1 rgb calibration,” “freiburg2 rgb calibration,” and “freiburg3 calibration rgb depth.” These sequences are listed in the “Calibration Files” category; we refer to them as the checkerboard sequences. Second, to represent a typical use case, we use three indoor sequences from the “Handheld SLAM” category of the TUM dataset: “freiburg1 360,” “freiburg1 floor,” and “freiburg1 desk.” We refer to these as the indoor sequences. Third, as a sanity check, we prepare seven texture-rich outdoor sequences captured by smartphones and refer to them as the outdoor sequences. Each outdoor sequence is a four-second video recorded at 30 frames per second (120 frames total) at a resolution of $6 4 0 \times 3 6 0$ pixels. Sample frames are shown in Fig. 2.

![](images/77614419e1ba8d6c180902a3c0b237688497c0c4ca31dd71bd8df02c2571cf90.jpg)

![](images/ab0bfa427c05f2af80687117a3005e262c32ac64fe5b5a89b98ff6c441611fb5.jpg)

![](images/8a96d7be80b3a75797cb0be040eb298331fcd56d18d8ed020d2a92541f2489d7.jpg)  
Fig. 2. Example frames from the checkerboard (left), indoor (middle), and outdoor (right) sequences.

## A. LSD vs HOME

For CLAD to work effectively, each frame must contain enough detected lines, and each keypoint must have enough associated lines. Because only some detected lines can be matched across images, the number of line correspondences available to CLAD is lower than the total number of detected lines. It is therefore important for each frame to have many detected lines and for each keypoint to have many associated line candidates. We therefore compare LSD, using OpenCV’s default settings, with HOME on the checkerboard, indoor, and outdoor sequences. For CLAD, we extend each LSDdetected line segment to an infinite line. We measure the average number of detected lines per frame and the average number of associated line candidates per keypoint. The results are reported in TABLE I. In the table, “Keypoints,” “Detected,” and “Associated” denote the average numbers of detected ORB keypoints, detected lines, and associated line candidates, respectively. HOME consistently detects more lines and provides more associated line candidates than LSD for all three sequence types. This result supports our choice of HOME as the source of line features for CLAD. HOME reaches its configured maximum of 2000 lines in most sequences.

TABLE I  
AVERAGE NUMBERS OF DETECTED LINES AND ASSOCIATED LINE CANDIDATES FOR LSD AND HOME.
<table><tr><td rowspan="2"></td><td colspan="2">Checkerboard (4799 frames)</td><td colspan="2">Indoor (2611 frames)</td><td colspan="2">Outdoor (840 frames)</td></tr><tr><td>LSD</td><td>HOME</td><td>LSD</td><td>HOME</td><td>LSD</td><td>HOME</td></tr><tr><td>Keypoints</td><td colspan="2">497.5</td><td colspan="2">378.4</td><td colspan="2">500</td></tr><tr><td>Detected</td><td>825.7</td><td>1999.9</td><td>629.7</td><td>1985.3</td><td>929.7</td><td>2000</td></tr><tr><td>Associated</td><td>20.1</td><td>43.2</td><td>15.9</td><td>41.2</td><td>19.7</td><td>42.8</td></tr></table>

## B. ORB vs CLAD

We compare CLAD and ORB by matching pairs of video frames. For the checkerboard and indoor sequences, we match adjacent frames. Because adjacent-frame matching is too easy for the outdoor sequences, we instead match pairs of outdoor frames separated by four frames. Ground-truth correspondences for ORB keypoints are difficult to obtain in real-world scenes. We therefore use a practical evaluation to test whether CLAD provides matches that differ from those found by ORB.

Because ORB and CLAD use the same keypoints, we expect some of their matches to overlap. We consider overlapping matches more reliable than non-overlapping matches because the two methods use fundamentally different information. We compute a fundamental matrix from the overlapping matches and use it to evaluate the two-view geometric consistency of the overlapping, ORB-only, and CLAD-only matches. Although the estimated matrix is not guaranteed to be perfect, it provides a reasonable basis for comparing the statistical properties of these three match sets. We also evaluate the overlapping matches to check that they are reliable or at least consistent with the two-view geometry. For this experiment, we measure the distances from keypoints to the corresponding epipolar lines in both images. For inlier thresholds θ = 1, 3, 9, we report the average number and ratio of inlier matches, the root-mean-square error (RMSE) of the residuals, and the average total number of matches in each category. We consider a match inlier if the distances are below the threshold θ in both images. We skip a frame pair if it has fewer than 16 overlapping matches or if fundamental matrix estimation fails. We estimate the fundamental matrix using OpenCV’s findFundamentalMat function with the LMEDS algorithm.

The comparison results are shown in TABLE II, where “ORB-o” and “CLAD-o” denote ORB-only and CLAD-only matches, respectively. Overall, the overlapping matches have higher inlier ratios and lower RMSE values, indicating that matches found by both ORB and CLAD are consistent with the two-view geometry. The checkerboard sequences most clearly demonstrate the value of global context. The average total number of ORB-only matches is 52.3, much lower than the 198.3 CLAD-only matches. Moreover, even at the widest threshold, θ = 9, the inlier ratio of the ORB-only matches is only 54.7%. This result shows that ORB has difficulty distinguishing locally similar patterns. In contrast, the CLAD-only matches achieve an inlier ratio of 99.9% at the same threshold. Thus, CLAD can use global context to match keypoints reliably, even when their local patterns are ambiguous. Fig. 1 shows example ORB and CLAD matches from the checkerboard sequences. The example illustrates how ORB can confuse locally similar patterns, whereas CLAD can distinguish keypoints using the global grid structure.

The ORB-only and CLAD-only matches also show different characteristics on the indoor and outdoor sequences. CLADonly matches are more numerous than ORB-only matches, but the ORB-only matches have lower residuals. The ORB-only inlier ratio is higher than the CLAD-only inlier ratio at strict thresholds, such as θ = 1, while the ordering reverses at the relaxed threshold of θ = 9. In other words, CLAD finds more matches, but some are close to, rather than exactly on, the correct correspondences. This behavior is consistent with the expected characteristics of global and local methods.

## C. ORB+SIFT’ vs CLAD

Our goal is to show that CLAD provides matches that complement traditional methods based on local descriptors. A comparison with other methods, such as SIFT, would therefore be useful. However, this comparison is difficult because each method normally uses its own keypoint detector. A meaningful comparison requires fixing the keypoint set and varying only the matching algorithm. To make this comparison, we compute SIFT descriptors at the keypoints detected by ORB and match them using Lowe’s ratio test with a threshold of 0.75, as for ORB and CLAD. We use OpenCV’s SIFT implementation with its default parameters and set the keypoint size to 5 pixels. We then combine these pseudo-SIFT matches with the standard ORB matches and compare the result with CLAD. We call this combination ORB+SIFT’. The results are shown in TABLE III, where “ORB+S’-o” denotes matches found only by ORB+SIFT’. Adding pseudo-SIFT matches increases the numbers of overlapping and ORB+SIFT’-only matches and decreases the number of CLAD-only matches. However, the overall trend is similar to that in the ORB-versus-CLAD comparison.

TABLE II  
COMPARISON OF KEYPOINT MATCHING PERFORMANCE BETWEEN ORB AND CLAD FOR DIFFERENT THRESHOLD VALUES θ.
<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">Checkerboard (4796 pairs)</td><td colspan="3">Indoor (2472 pairs)</td><td colspan="3">Outdoor (812 pairs)</td></tr><tr><td>Overlapped</td><td>ORB-0</td><td>CLAD-0</td><td>Overlapped</td><td>ORB-0</td><td>CLAD-0</td><td>Overlapped</td><td>ORB-0</td><td>CLAD-0</td></tr><tr><td rowspan="2"></td><td>Total</td><td>233.1</td><td>52.3</td><td>198.3</td><td>170.2</td><td>35.5</td><td>95.4</td><td>233.2</td><td>47.1</td><td>109.0</td></tr><tr><td>θ = 1</td><td>196.7</td><td>15.6</td><td>116.9</td><td>136.6</td><td>20.7</td><td>41.9</td><td>200.3</td><td>34.3</td><td>52.6</td></tr><tr><td rowspan="3">Inlier</td><td>θ = 3</td><td>228.8</td><td>23.5</td><td>183.1</td><td>168.9</td><td>31.2</td><td>81.1</td><td>232.3</td><td>45.2</td><td>96.0</td></tr><tr><td>θ = 9</td><td>233.0</td><td>28.6</td><td>198.2</td><td>170.1</td><td>32.9</td><td>94.8</td><td>233.1</td><td>46.1</td><td>108.0</td></tr><tr><td>θ = 1</td><td>84.4%</td><td>29.9%</td><td>59.0%</td><td>80.3%</td><td>58.4%</td><td>43.9%</td><td>85.9%</td><td>72.8%</td><td>48.2%</td></tr><tr><td rowspan="3">Inlier ratio</td><td>θ = 3</td><td>98.2%</td><td>45.0%</td><td>92.3%</td><td>99.3%</td><td>88.1%</td><td>85.0%</td><td>99.6%</td><td>95.9%</td><td>88.1%</td></tr><tr><td>θ = 9</td><td>100.0%</td><td>54.7%</td><td>99.9%</td><td>100.0%</td><td>92.7%</td><td>99.3%</td><td>100.0%</td><td>97.9%</td><td>99.1%</td></tr><tr><td>θ = 1</td><td>0.280</td><td>0.396</td><td>0.484</td><td>0.443</td><td>0.514</td><td>0.539</td><td>0.406</td><td>0.484</td><td>0.531</td></tr><tr><td rowspan="2">RMSE</td><td>θ = 3</td><td>0.628</td><td>1.091</td><td>1.063</td><td>0.761</td><td>1.061</td><td>1.357</td><td>0.622</td><td>0.868</td><td>1.281</td></tr><tr><td>θ = 9</td><td>0.855</td><td>2.488</td><td>1.572</td><td>0.830</td><td>1.477</td><td>2.076</td><td>0.675</td><td>1.080</td><td>1.847</td></tr></table>

TABLE III

COMPARISON OF KEYPOINT MATCHING PERFORMANCE BETWEEN ORB+SIFT’ AND CLAD FOR DIFFERENT THRESHOLD VALUES θ.
<table><tr><td rowspan="3" colspan="2"></td><td colspan="3">Checkerboard (4796 pairs)</td><td colspan="3">Indoor</td><td colspan="3">Outdoor</td></tr><tr><td colspan="3"></td><td colspan="3">(2472 pairs)</td><td colspan="3">(812 pairs)</td></tr><tr><td>Overlapped</td><td>ORB+S&#x27;-0</td><td>CLAD-0</td><td>Overlapped</td><td>ORB+S&#x27;-0</td><td>CLAD-0</td><td>Overlapped</td><td>ORB+S&#x27;-0</td><td>CLAD-0</td></tr><tr><td rowspan="3"></td><td>Total</td><td>254.8</td><td>101.2</td><td>176.5</td><td>195.6</td><td>52.4</td><td>67.9</td><td>264.2</td><td>65.7</td><td>77.9</td></tr><tr><td>θ = 1</td><td>209.6</td><td>19.8</td><td>106.5</td><td>151.6</td><td>27.0</td><td>28.6</td><td>219.2</td><td>41.5</td><td>36.6</td></tr><tr><td>θ = 3</td><td>249.7</td><td>34.0</td><td>165.0</td><td>192.8</td><td>44.5</td><td>55.7</td><td>262.2</td><td>61.4</td><td>66.9</td></tr><tr><td rowspan="3">Inlier ratio</td><td>θ = 9</td><td>254.8</td><td>44.8</td><td>176.4</td><td>195.5</td><td>48.3</td><td>67.3</td><td>264.2</td><td>64.6</td><td>77.1</td></tr><tr><td>θ = 1</td><td>82.3%</td><td>19.5%</td><td>60.3%</td><td>77.5%</td><td>51.4%</td><td>42.1%</td><td>83.0%</td><td>63.1%</td><td>46.9%</td></tr><tr><td>θ = 3</td><td>98.0%</td><td>33.6%</td><td>93.5%</td><td>98.6%</td><td>84.9%</td><td>82.0%</td><td>99.2%</td><td>93.4%</td><td>85.8%</td></tr><tr><td rowspan="3"></td><td>θ = 9</td><td>100.0%</td><td>44.2%</td><td>99.9%</td><td>100.0%</td><td>92.2%</td><td>99.1%</td><td>100.0%</td><td>98.3%</td><td>98.9%</td></tr><tr><td>θ = 1</td><td>0.289</td><td>0.433</td><td>0.480</td><td>0.453</td><td>0.522</td><td>0.537</td><td>0.415</td><td>0.499</td><td>0.530</td></tr><tr><td>θ = 3 θ = 9</td><td>0.673 0.892</td><td>1.256 2.792</td><td>1.023 1.472</td><td>0.835 0.948</td><td>1.193 1.717</td><td>1.376 2.220</td><td>0.709 0.779</td><td>1.068 1.384</td><td>1.290 1.954</td></tr></table>

These results show that CLAD provides matches that complement traditional local-feature matching methods. Although ORB generally provides more accurate matches, CLAD can find additional, globally consistent matches that ORB alone may miss.

## D. Processing Time

Finally, we report the processing time of the current CLAD implementation, measured on a laptop with an AMD Ryzen 7 7735U CPU and 32 GB of RAM. We use the outdoor sequences because they contain relatively many ORB and HOME features and thus provide a demanding test of processing time. The average processing times are 1.580 ms for ORB keypoint detection, 3.919 ms for HOME keypoint detection and descriptor extraction, 4.321 ms for HOME matching, 3.545 ms for CLAD descriptor extraction, and 1.072 ms for CLAD matching. Although we have not thoroughly optimized the implementation, these results indicate that it can operate at practical processing rates and may run faster with further optimization.

## IV. CONCLUSION

In this paper, we introduced CLAD, a keypoint matching method that uses line correspondences to incorporate global context and complement traditional local-feature matching methods. Line correspondences capture linear structures in a scene and can provide geometric information, such as a homography. However, they have fewer applications than point correspondences; for example, line correspondences alone cannot be used to compute the fundamental matrix [9]. CLAD associates this global context with local keypoints detected by traditional methods, making global linear context available to standard keypoint-matching pipelines. This design may allow CLAD to be used in a wide range of applications.

Several directions remain for future research. The most important next step is to evaluate CLAD in computer vision tasks such as 3D reconstruction and visual localization. Another direction is to improve HOME or evaluate other linebased methods to further enhance CLAD. Because HOME is designed to process video frames, improving or replacing it could extend CLAD to general keypoint-matching tasks. FAST and other traditional detectors are designed for local-feature matching and may not be optimal for CLAD. Developing or adapting keypoint detectors better suited to CLAD could therefore further improve its performance. Because CLAD provides information that complements traditional local-feature matching methods, designing descriptors that combine local and global information could further improve matching performance.

## ACKNOWLEDGMENT

The author thanks Takeshi Miura for fruitful discussions and Masaki Hilaga for his support and for permitting this research as part of the author’s corporate duties. The author used OpenAI’s ChatGPT, Google Gemini, and GitHub Copilot to edit the language, organize the manuscript, and improve its presentation. The author reviewed and verified all AI-assisted content and is solely responsible for the final manuscript.

## REFERENCES

[1] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” International Journal of Computer Vision, vol. 60, no. 2, pp. 91–110, 2004.

[2] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “Orb: An efficient alternative to sift or surf,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV). IEEE, 2011, pp. 2564–2571.

[3] E. Rosten and T. Drummond, “Machine learning for high-speed corner detection,” in European Conference on Computer Vision (ECCV). Springer, 2006, pp. 430–443.

[4] R. G. von Gioi, J. Jakubowicz, J.-M. Morel, and G. Randall, “Lsd: A line segment detector,” Image Processing On Line, vol. 2, pp. 35–55, 2012.

[5] L. Zhang and R. Koch, “An efficient and robust line segment matching approach based on lbd descriptor and pairwise geometric consistency,” Journal of Visual Communication and Image Representation, vol. 24, no. 7, pp. 794–805, 2013.

[6] M. Satoh, “Hough-sift: Robust image registration for linear structures via hough space,” 2026. [Online]. Available: https://arxiv.org/abs/2607.14598

[7] ——, “Home: Robust hough-space matching method for structured and textureless videos,” 2026. [Online]. Available: https://arxiv.org/abs/2607. 25389

[8] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of rgb-d slam systems,” in Proc. of the International Conference on Intelligent Robot Systems (IROS), Oct. 2012.

[9] R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed. Cambridge University Press, 2003.