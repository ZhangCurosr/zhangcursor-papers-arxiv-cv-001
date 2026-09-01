# HorizonNet for visual terrain navigation

Bertil Grelsson\*<sup>1</sup>, Andreas Robinson\*<sup>1</sup>, Michael Felsberg<sup>1</sup>, Fahad Shahbaz Khan<sup>1,2</sup>

<sup>1</sup>Computer Vision Laboratory, Department of Electrical Engineering, Linkoping University, Sweden¨

<sup>2</sup>Inception Institute of Artificial Intelligence, Abu Dhabi, UAE

{bertil.grelsson, andreas.robinson, michael.felsberg, fahad.khan}@liu.se

Abstract—This paper investigates the problem of position estimation of unmanned surface vessels (USVs) operating in coastal areas or in the archipelago. We propose a position estimation method where the horizon line is extracted in a 360 degree panoramic image around the USV. We design a CNN architecture to determine an approximate horizon line in the image and implicitly determine the camera orientation (the pitch and roll angles). The panoramic image is warped to compensate for the camera orientation and to generate an image from an approximately level camera. A second CNN architecture is designed to extract the pixelwise horizon line in the warped image. The extracted horizon line is correlated with digital elevation model (DEM) data in the Fourier domain using a MOSSE correlation filter. Finally, we determine the location of the maximum correlation score over the search area to estimate the position of the USV. Comprehensive experiments are performed in a field trial in the archipelago. Our approach provides promising results by achieving position estimates with GPS-level accuracy.

Index Terms—position estimation, horizon, registration, CNN, DEM

## I. INTRODUCTION

In recent years, unmanned systems such as Unmanned Aerial Vehicles (UAVs), Unmanned Ground Vehicles (UGVs), and Unmanned Surface Vessels (USVs) have become increasingly popular providing safe and secure operations in remote environments. Within unmanned systems, the aim of USVs is to perform various ocean sensing tasks in a variety of cluttered sea environments. Generally, these USVs (autonomous or tele-operated) are reliant on accurate position measurements provided by the Global Positioning System (GPS) for safe navigation. In hostile ocean scenarios, the GPS signal can be unreliable or not available at all, creating the need for alternative position sensors. In this paper, we tackle the problem of providing accurate position measurements for a USV operating in challenging coastal areas or in the archipelago.

Terrain navigation, without any need for externally controlled signals and sensors, is suitable as an alternative position sensor in these scenarios. In terrain navigation, measurements of the surrounding terrain are correlated with a terrain spatial database to provide position measurements. Terrain navigation is well established as a position sensor and the underlying sensor measurements could originate from various techniques such as radars, sonars, altimeters, lidars, and cameras [1]–[3]. For the position estimation in our scenario, the USV can utilize information obtained from an omnidirectional camera, a digital compass, and a high-resolution digital elevation model (DEM) of the operational area, see Figure 1.

![](images/2cb16adcd9252e15a1de7163a0525af2270e2869a7e016e52c85eb163bb367dd.jpg)  
Fig. 1. Left: Image showing the USV equipped with a Ladybug omnidirectional camera. Right: CNN training image positions (cyan); Test image position estimation with the GPS (red) and the proposed method (yellow). Our proposed method provides position estimation as accurate as the GPS.

In recent years, deep learning and convolutional neural networks (CNNs) have significantly boosted the level of performance for various computer vision tasks including image classification, object segmentation, and object tracking. For unmanned systems, CNNs have e.g. been used for object recognition in UAV images [4], road lane guidance for autonomous cars [5], and collision avoidance manoeuvers for USVs [6]. In this paper, we employ CNNs to extract terrain information from USV images to generate accurate position measurements for the USV.

To aggregate terrain information from consecutive images and to align the image content with DEM data for USV position estimation, object tracking and registration are essential components. In recent years, the best performing visual object trackers apply a discriminatively trained correlation filter (DCF) on top of multidimensional features [7]. The foundation for DCF-based trackers is the MOSSE correlation filter [8]. A filter is trained to model the appearance of the tracked object in some example images. To search for the object in the next frame or, in our case, to register the object with DEM data, a correlation score is computed over the search area. For computational speed, the correlation is performed in the Fourier domain.

For terrain navigation of a manned surface vessel in coastal areas and in the archipelago, humans would intuitively use readily observable characteristic landmarks in a few directions, project the directions on a sea chart or map, and use cross bearing to determine the vessel position. The horizon line most often constitutes a spatially extended characteristic landmark for islands and the shore. Matching of the complete horizon line around the vessel with a map is mathematically a more robust cross bearing measurement than just using a few directions. To obtain highly accurate position estimates, the horizon line must be captured with high angular resolution, which camera sensors can provide. These insights are the motivations behind our proposed position estimation method.

We propose a position estimation method where two CNNs are employed to extract the camera orientation and horizon line, respectively, in a 360<sup>◦</sup> panoramic image around the USV. The horizon line is correlated with DEM data in the Fourier domain using a MOSSE correlation filter. Finally, we determine the location of the maximum correlation score over the search area to estimate the position of the USV. Our approach is evaluated in a field trial in the archipelago and achieves promising results with the average global position accuracy of 2.5±1.3 meters relative to the GPS ground truth data, i.e. we achieve position estimates with GPS-level accuracy. Figure 1 shows position estimates with GPS and the proposed method.

Our contributions are: 1) The proposed method for USV terrain navigation, 2) CNNs designed for camera orientation estimation and horizon segmentation in a marine environment, 3) Horizon line registration with a MOSSE correlation filter, 4) Comprehensive experiments that demonstrate the GPS-level accuracy of the proposed method.

## II. RELATED WORK

Automatic extraction of the horizon line and water line (which is the first water to land/sky transition seen from the vessel) in the image is equivalent to segmentation of the image into sky/land/water groups. Previous works [9], [10] have investigated the use of CNNs to determine straight lines and sky segmentation in images. Lee et al. [9] propose an approach where a CNN is trained to find straight lines in order to extract semantically meaningful information from the image. Verbickas and Whitehead [10] propose a CNN-based method for sky segmentation. In their approach a two convolutional layer network is trained from scratch on the authors’ own dataset. A comparison of deep learning methods for horizon/sky line segmentation is presented by Ahmad et al. [11]. However, their evaluation is only performed on sky/mountain images. To the best of our knowledge, we have not encountered any public benchmark or CNN-based segmentation method with respect to marine images and USVs.

Most previous work on horizon registration for position estimation has been performed in the spatial domain. Dumble and Gibbens [12] precompute reference horizon profiles from DEM data in the Alps in a set of 3D grid points. They extract the reference profile at the grid point closest to the assumed position of their aerial vehicle. To refine the estimated location, they use gradient descent to iteratively minimize the error between the horizon line in the image and the transformed reference profile. The method requires a large horizon profile variance, which prevents its use in our scenario in the archipelago with low altitude islands. A method for accurate registration of low variance horizon profiles is proposed by Grelsson et al. [13], but they only estimate the camera orientation and not the position. Another method suitable for low variance horizon lines is proposed by Chiodini et al. [14], where a Mars rover is localized by matching the detected skyline with DEM data. For position estimation, they do a grid search over the location and the viewing angle, and minimize the least-square error between the detected and the rendered skyline. Baatz et al. [15] use a database with extracted features from the horizon line called contourlets, and employ a Bag-of-Words-like approach for large scale localization of mountainous images. We have not encountered any previous work where registration of the horizon line with DEM data has been carried out in the Fourier domain.

![](images/a38ac6daab723878e3ca706bb8242c1cbc8f992fdaaacbdb05dd1cfeae51e6a1.jpg)  
Fig. 2. Flowchart for position estimation with classical methods. Predicted horizon line (red) overlaid on image (1). Geometric horizon line (green) and water line (red) overlaid on distance function (3) and warped image (4) before registration, after registration (5), and after fine tuning (6).

The introduction of the MOSSE correlation filter [8] showed that image object tracking with adaptive correlation filters in the Fourier domain is significantly faster and also more robust to variations in target appearance than previous trackers working in the spatial domain. A major advantage with the proposed method compared to appearance based approaches [16] is that it does not require images to have been captured in the test area prior to navigation. It is sufficient to have images from a similar environment to provide accurate position estimates.

## III. CLASSICAL METHODS FOR POSITION ESTIMATION

The approach for our position estimation method is based on registration of the horizon line with DEM data. In this section, we sketch an algorithm with classical computer vision methods for position estimation. This algorithm is used to create the ground truth labels for the CNNs in the proposed method. It also provides a baseline for the position estimate accuracy that can be obtained by registration in the spatial domain. An algorithm flowchart is shown in Figure 2 together with cropped sample images for illustration of each step.

The camera used in our field trials generates panoramic images in a cylindrical projection. First, to determine an approximate camera orientation from the image, we use Canny edge detection [17] and Hough voting [18]. We search for the approximate horizon plane on the cylinder, which will be an S-shaped curve in the panoramic view. We adapt the method in Grelsson et al. [13] and vote for the normal vector of the horizon plane parameterized with the pitch and roll angles.

The second step in the algorithm is to warp the image to compensate for the camera orientation. The warping will create an image corresponding to an approximately level camera, suitable for the subsequent registration process with the geometric horizon. As a third step, we compute a Canny edge image on the warped image and a distance function D based on the edge image. Another input to the registration is the geometric horizon line from the digital elevation model. The DEM for the test site was provided by Vricon [19]. The DEM is computed from recent satellite imagery and has a pixel resolution on the ground of 0.5m. The altitude accuracy is in the same order as the pixel resolution.

To create the ground truth labels for training the CNNs, we use the position given by the GPS and the heading angle from the digital compass or, if not available, the tangent vector from the logged GPS trajectory. The geometric horizon is generated by ray-tracing using DEM data. For the desired number of image columns around the vessel, we extract the altitude profile from the DEM. In each direction, we compute the elevation angle of all objects along the ray. The maximum elevation angle is taken as the horizon point in that direction. The water line point, the first water to land/sky transition in each direction, is taken as the vertical viewing angle to the point where the DEM makes a step larger than a 0.2m threshold along the ray direction. The search radius is set to 6km in all directions. The ideal horizon, assuming a flat and spherical earth, is at 3.2km for a camera at 1.0m height, but we add some margin to cope with the topography.

Ideally, the complete geometric horizon line would be located where the distance function D is zero. To find the horizon line in the image, we search for a rotation of the geometric horizon line points on the cylinder, such that when projected onto the image, their summed distance function values will be a minimum, i.e. we minimize the score

$$
s = \underset { \theta , \phi , \psi } { \mathrm { a r g m i n } } \sum _ { i } \mathrm { D } \left\{ \mathbf { R } ( \psi ) \pi ( \mathbf { R } ( \theta , \phi ) \pi ^ { - 1 } ( h _ { i } ) ) \right\}\tag{1}
$$

where $h _ { i }$ are the horizon line points, π is the projection from the cylinder to the image plane, θ, ϕ, and ψ are the pitch, roll and heading angles and R is a rotation matrix.

For registration, we perform a grid search over the pitch, roll and heading angles. For the first two angles we need to compute the rotation matrix and project the transformed points onto the image plane. The heading angle rotation simply corresponds to a horizontal shift on the image plane. The step size in pitch and roll is set to $0 . 2 5 ^ { \circ }$ and we search over the range ±2<sup>◦</sup>. We extract the rotation angles for the minimum score and project the geometric horizon line and water line onto the warped image after transformation with the said rotation angles. In general, there is a good fit between the geometric horizon line and water line with the image content, but occasionally there are small deviations. The main reason is that the DEM is not a perfect representation of the real world.

![](images/4ce5f4cf03cd276273b3242d0fe1bcf6985f01212a6a79d91e12fe1992cd8b2f.jpg)  
Fig. 3. Flowchart for position estimation with the proposed method.

To adjust for the discrepancies we perform a final tuning step of the geometric horizon line and water line in the image.

For final tuning we first convolve the warped image with a Sobel filter to enhance gradients in the vertical direction. For each image column we search for a local maximum of the gradient, exceeding a threshold, in a small region close to the horizon and water lines obtained in the registration step. If no gradient maximum is detected in the search region, the horizon line from the registration is retained. For position estimation, the algorithm is run for positions over a grid and the minimum score obtained according to (1) is recorded for each position.

The main drawback with this classical method for position estimation is that it is prohibitively slow for real-time applications. In our implementation, the spatial registration took more than 1s per position grid point, whereas our proposed method below achieved more than 40 grid points per second.

## IV. CNN-BASED POSITION ESTIMATION METHOD

Prior to designing our proposed method, we experimented with an end-to-end CNN to output the position estimate directly from the input image. We tried a modified version of Posenet [20]. We trained it to learn the camera position and orientation, but the training failed completely. The reason for this behavior is quite simple. Only a small part of the image (the land objects) contain information that is useful for position estimation. The sky and sea change appearance over time and will only distract the position estimation if using the full image content. Hence, prior knowledge about what is the relevant part of the image for position estimation is required.

This insight motivates the design of our proposed method for position estimation, which is similar to the classical method in its architecture. The proposed algorithm consists of seven steps, which are explained in detail in this section and shown in a flow chart in Figure 3.

HorizonFinder CNN. To determine a rough camera orientation (pitch and roll angles), i.e. to find an approximate horizon line in the panoramic image, we employ a convolutional neural network called HorizonFinder. This CNN replaces the Canny detector and the Hough voting in the classical method.

We use a ResNet50 network [21] pretrained on ImageNet to provide feature descriptors. We then add two stacks with a convolutional layer, vertical pooling, and a Leaky Rectified Linear Unit (Lrelu) [22] activation function. Finally, we have a fully connected layer to output the camera orientation angles. We found that the network training was more accurate when the output was the cosine and sine of the pitch and roll angles, θ and ϕ, rather than having the attitude angles by themselves as output. The network design is shown in Figure 4.

As a loss function we use the L1 loss between the predicted and ground truth values for the parameters cos θ, sin θ, cos ϕ, and sin ϕ. This network design and loss function gave satisfactory results as judged by viewing the backprojected horizon line on the panoramic image. Since we did not have access to any exact ground truth for the camera orientation from external sensors, we could not perform any quantitative comparison for different network designs and loss functions. We trained the network for 100 epochs. The initial learning rate was set to 0.0001 and it was then reduced by a factor of two every nine epochs.

![](images/c03165f85e29139d1104f8d072dbe21e3aa2260c96c17548115788738f3dfb4a.jpg)  
Fig. 4. Network design for HorizonFinder (top) and HorizonSegmenter (bottom). The numbers denote the input and output size (H×W×C) for each layer.

Image warping. Based on the predicted pitch and roll angles from the HorizonFinder CNN, we warp the panoramic image to compensate for the camera orientation. Since the horizon now will be almost vertically centered in the image, we only warp the central part of the image. The size of the original panoramic image is 2048×1024, whereas the warped image is 2048×384.

HorizonSegmenter CNN. To predict the location of the horizon line and water line in the warped image, we design a second CNN called HorizonSegmenter. This CNN has no one-to-one counterpart in the classical method. We use a similar network design as for HorizonFinder. We start with a pretrained ResNet50 to generate feature descriptors. We use three stacks of layers, each comprising a convolutional layer, vertical pooling, and a Lrelu activation function. To obtain the same horizontal resolution as the ground truth, we insert two horizontal upsampling layers with bilinear interpolation. Finally, a fully connected layer is added to output the vertical pixel location of the horizon line and water line for all image columns. As a loss function we use the absolute pixel difference between the predicted and ground truth horizon line and water line summed over the training image. The network design is shown in Figure 4. To avoid considerable overfitting during training due to relatively few images, we randomly generate training images from the original warped images. We make a random horizontal crop of a 512×384 image from the original image during training. We used the same learning rate scheme as for the HorizonFinder network.

Compute geometric horizon. The geometric horizon is computed in exactly the same manner as described in the classical position estimation method.

Phase correlation for relative heading angle measurements. For the registration with the MOSSE correlation filter in the next step, we need the camera heading angle for each image in a global coordinate system as an input. The absolute heading angle of the first image in a sequence can be obtained from a digital compass (without relying on GPS) with an accuracy better than 2<sup>◦</sup> [23]. In our field trial we did not have access to a digital compass. Instead, we replaced the heading for the first image with the ground truth heading from the GPS trajectory plus some noise simulating the digital compass.

To find the relative change in heading angles from one video frame to the next, we employ phase correlation matching [24] of the predicted horizon line output from the HorizonSegmenter network for successive images.

MOSSE correlation filter. We adapt the MOSSE correlation filter [8], originally developed for visual object tracking. The MOSSE filter is designed to generate a desired output signal (typically a Gaussian) shifted to the spatial location most closely corresponding to a set of learned reference signals. For efficiency, the filter is evaluated in the Fourier domain. The filter is trained with multiple references to improve its robustness against changes in appearance and noise.

In our case we use the segmented horizon lines from an image sequence as reference. We align the segmented horizon lines in the spatial domain in accordance with their estimated heading angle. In the Fourier transform domain, we denote the horizon lines in the sequence with $F _ { i }$ and the Gaussian target signal with G. We compute the MOSSE filter K as

$$
K ^ { \star } = \frac { \sum _ { i } G _ { i } \cdot F _ { i } ^ { \star } } { \sum _ { i } F _ { i } \cdot F _ { i } ^ { \star } } \mathrm { ~ , ~ }\tag{2}
$$

and the MOSSE filter signal response as

$$
r = \mathcal { F } ^ { - 1 } \left\{ H \cdot K ^ { \star } \right\} \ \mathrm { ~ , ~ }\tag{3}
$$

where H is the Fourier transform of the geometric horizon line from the DEM, <sup>⋆</sup> is the complex conjugate and · denotes element-wise multiplication. Ideally, r will be the Gaussian target signal with a zero phase shift if the heading angle estimate is correct. Since the digital compass gives the absolute heading with an accuracy around 2<sup>◦</sup>, we search for the peak signal response within $\mathrm { ~ a ~ } \pm 4 ^ { \circ }$ band from the center to have some margin. As a quality measure of the signal response, i.e. our MOSSE filter correlation score, we use the peak-tooutput-energy ratio [25]. We suppress the response within the expected Gauss signal bandwidth around the detected peak and compute the average energy over the remainder of the signal response. The score is the peak signal over the square root of the average energy.

Position estimation. To generate a position estimate, we compute the MOSSE filter correlation score for various positions in a XY grid. We extract the maximum score over the search area and apply a second order polynomial fit to obtain a subgrid position estimate.

![](images/83565ac5e6b82b783bd7911b25aebe495d538234ab9c5779d169ff0b43320613.jpg)  
Fig. 5. HorizonFinder result (top): CNN output (green) and ground truth (red). HorizonSegmenter result (bottom): predicted horizon line (green) and water line (red). HorizonFinder predicts the approximate horizon line well and HorizonSegmenter outputs the essential shape of the horizon line.

## V. EXPERIMENTS AND RESULTS

## A. Field trial

A field trial was performed on a single day in the archipelago. Omnidirectional images were captured with a Ladybug3 camera [26] mounted on a tele-operated USV (4m long), see Figure 1. A total of 20k images were captured at 10 fps during a 45 minute trial. The rear view (roughly 30<sup>◦</sup>) was occluded by other sensors on the vessel. The vessel position was measured with a U-blox EVK-8 GPS receiver [27] acquired at 1 fps.

## B. CNNs for horizon detection and segmentation

40 image sequences, each containing 100 consecutive images acquired at 10 fps, were selected for evaluation of the CNNs and the full position estimation method. Ground truth labels were generated with the method described in Section III. 35 image sequences were used for training of the two CNNs and 5 sequences were used for testing, see Figure 1.

HorizonFinder. The HorizonFinder CNN robustly locates the horizon line in the panoramic image. The test error relative to the ground truth is on average less than 0.1<sup>◦</sup>. Figure 5 shows the result for one test image.

One observation that was made is that the HorizonFinder network tends to give a visually better approximation of the horizon line than the ground truth generated with Canny detection and Hough voting. The reason is probably that the network has learned to generalize and average the errors produced in the ground truth generation. We have no ground truth camera orientation from other sensors to provide a quantitative evidence for this observation.

HorizonSegmenter. The HorizonSegmenter network segments the horizon line and water line well in the warped image, but the segmentation is not perfect. Figure 5 shows the result for one test image. The CNN learns to predict the general shape of the horizon line but loses the high frequency variations in the horizon line. Single trees and other thin structures in the true horizon line are missed. This observation must be kept in mind when using the segmented horizon line in the registration process for position estimation, which is our main goal.

## C. Position estimates

For position estimation we have evaluated the results achieved with the proposed method (Section IV) and make a comparison with baseline results obtained with registration in the spatial domain (Section III).

We first assume that we have gained the prior knowledge that the vessel is located within a 80m×80m region. We compute the MOSSE filter correlation score every 4m within this region. We extract the grid position of the maximum score and perform a second order polynomial fit of the scoring function to interpolate a sub-grid resolution position estimate. To compute the MOSSE filter, we use the segmented horizon line for 10 consecutive images and correlate it with the geometric horizon line computed in one of the grid points. The USV travels about 6m during these 10 frames (1s), i.e. slightly more than the grid size. Typical results for the correlation score are illustrated in Figure 6.

The position estimates obtained with the proposed method are listed in Table I and shown on the map in Figure 7 together with the measured GPS position. The estimated position is the average position for the USV while capturing the 10 images. The average deviation between the position estimates with our proposed method and the GPS measurements is 2.47±1.26m. Assuming uncorrelated position measurements, these deviations imply that our method provides position measurements with GPS-level accuracy, according to e.g. [28].

The key contribution in our proposed method is the adaptation of the MOSSE filter to perform the registration of the horizon line in the Fourier domain. As a comparison, we perform the registration in the spatial domain, where we use the method described in Section III. For each position grid point, we record the lowest summed distance according to (1). We detect the location of the lowest distance over the grid and make the same sub-grid resolution interpolation as for the proposed method. The average position error obtained with spatial registration is 3.1m. A t-test at 95% significance level shows that the position errors achieved with the proposed method are significantly lower than those obtained with spatial registration. The Fourier method is more robust against deviations between the horizon line from the image and the geometric horizon line from the DEM. As noted, the segmented horizon line from the CNN does not catch all high frequency variations in the image. Furthermore, not even a high-resolution DEM will be a perfect representation of the real world. These discrepancies are more or less inevitable, and for the Fourier registration, the lower frequency content is sufficient to produce highly accurate position estimates whereas the discrepancies distract the spatial registration more.

TABLE I  
POSITION ESTIMATION ACCURACY ACHIEVED WITH THE MOSSE FILTER REGISTRATION AND WITH THE SPATIAL REGISTRATION.
<table><tr><td>Registration method</td><td>Input</td><td>Mean error (m)</td><td>Std (m)</td></tr><tr><td>MOSSE</td><td>Horizon</td><td>2.47</td><td>1.26</td></tr><tr><td>Spatial, full search</td><td>Horizon</td><td>3.09</td><td>1.91</td></tr><tr><td>Spatial, shift only</td><td>Horizon</td><td>6.45</td><td>5.95</td></tr><tr><td>MOSSE</td><td>Horizon+Water</td><td>3.02</td><td>1.40</td></tr></table>

![](images/70fe572b9e955c3e2c1c1f4e9612dae42aa9a58eb76747c4c6fbaa986be2d8c8.jpg)

![](images/44ada97b38fc9996124a2f470e149df27101e55f59707d3ee45565b961dd3d34.jpg)

![](images/bd20533918f0bef6c7883c1cee0348b27a9625d81a4c35f0e36df59dc1667838.jpg)

Fig. 6. The MOSSE filter correlation score (blue) in the search region for three example positions. The geometric horizon lines are displayed in green and the water lines in red (as seen from the true position). The search region is extended to 1km×1km for better illustration. The highest correlation score is clearly obtained for the true position.  
![](images/e42f7bc77a80a2ace73a6216696e6bef3a72d139d79a1e2bddbcc98dc734999b.jpg)  
Fig. 7. Position estimation results: GPS (red), proposed method (yellow). MOSSE filter registration (left) and spatial registration (right).

The spatial registration is also by far slower than registration in the Fourier domain. To speed up the spatial registration considerably, we searched only in the heading angle (a shift in the image domain), and assumed that the pitch and roll angle estimates from the Hough voting were correct. This speed-up degrades the average position error to 6.5m. Again, this shows that the spatial registration is more sensitive to angular errors than MOSSE registration, which handles these angular errors in the Fourier domain matching process.

The HorizonSegmenter CNN additionally extracts the water line from the image. The motivation is that we wanted to investigate if the water line could aid and improve the position estimates. We compute a MOSSE filter for the difference between the horizon line and the water line in the image. This results in an average position error of 3.0m, i.e. inferior to using merely the horizon line. Our explanation is that for cameras at low height with small gracing angles to the sea, the water line position in the image is very sensitive to the actual height of the camera above the sea level. Also, the variance of the water line position in the image is small unless the USV is very close to land. These two factors make water line registration inaccurate for cameras on small USVs.

## VI. CONCLUSIONS

In this paper, we have investigated the problem of providing accurate position measurements for a USV operating in coastal areas or in the archipelago when GPS signals are denied or failing. We propose a position estimation method where the horizon line is extracted in a 360<sup>◦</sup> panoramic image around the USV. We design two CNN architectures. The first CNN determines an approximate horizon line in the image and implicitly determines the camera orientation. The image is warped to compensate for the camera orientation. The second CNN extracts the pixelwise horizon line and water line in the warped image. The extracted horizon line is correlated with DEM data in the Fourier domain using a MOSSE correlation filter. Finally, we determine the location of the maximum correlation score over the search area to estimate the position of the USV. Our approach provides promising results by achieving position estimates with GPS-level accuracy.

We have shown that the proposed method with registration of the horizon line in the Fourier domain outperforms a classical method with registration in the spatial domain, both in terms of position accuracy and computational speed. We found that for a small USV, with the camera nominally located only 1m above the sea level, registration of the water line with DEM data was not robust and degraded the position estimates obtained when using registration of the horizon line only. The observation that horizon line registration is sufficient to provide very accurate position estimates suggests that our proposed method can be generalized to other domains and also be employed by UGVs and UAVs operating on land.

## ACKNOWLEDGMENT

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.

This work was supported by the Swedish Foundation for Strategic Research (Smart Systems: RIT 15-0097).

This research is supported by CENIIT grant (18.14), and VR starting grant (2016-05543).

## REFERENCES

[1] D. Vaman, “Trn history, trends and the unused potential,” in Digital Avionics Systems Conference (DASC), 2012 IEEE/AIAA 31st. IEEE, 2012, pp. 1A3–1.

[2] J. Han, J. Park, J. Kim, and N.-s. Son, “Gps-less coastal navigation using marine radar for usv operation,” IFAC-PapersOnLine, vol. 49, no. 23, pp. 598–603, 2016.

[3] J. Melo and A. Matos, “Survey on advances on terrain based navigation for autonomous underwater vehicles,” Ocean Engineering, vol. 139, pp. 250–264, 2017.

[4] M. Radovic, O. Adarkwa, and Q. Wang, “Object recognition in aerial images using convolutional neural networks,” Journal ofImaging, vol. 3, no. 2, p. 21, 2017.

[5] B. T. Nugraha, S.-F. Su et al., “Towards self-driving car using convolutional neural network and road lane detector,” in Automation, Cognitive Science, Optics, Micro Electro-Mechanical System, and Information Technology (ICACOMIT), 2017 2nd International Conference on. IEEE, 2017, pp. 65–69.

[6] Q. Xu, C. Zhang, and L. Zhang, “Deep convolutional neural network based unmanned surface vehicle maneuvering,” in Chinese Automation Congress (CAC), 2017. IEEE, 2017, pp. 878–881.

[7] M. Kristan, A. Leonardis, J. Matas, M. Felsberg, R. Pflugfelder, L. C. Zajc, T. Vojir, G. Hager, A. Lukezic, A. Eldesokey, G. Fernandez,¨ and et al., “The Visual Object Tracking VOT2017 challenge results,” in 2017 IEEE INTERNATIONAL CONFERENCE ON COMPUTER VI-SION WORKSHOPS (ICCVW 2017), ser. IEEE International Conference on Computer Vision Workshops. IEEE, 2017, pp. 1949–1972.

[8] D. S. Bolme, J. R. Beveridge, B. A. Draper, and Y. M. Lui, “Visual object tracking using adaptive correlation filters,” in Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on. IEEE, 2010, pp. 2544–2550.

[9] J.-T. Lee, H.-U. Kim, C. Lee, and C.-S. Kim, “Semantic line detection and its applications,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 3229–3237.

[10] R. Verbickas and A. Whitehead, “Sky and ground detection using convolutional neural networks,” in Proceedings of the International Conference on Machine Vision and Machine Learning, vol. 64, 2014.

[11] T. Ahmad, P. Campr, M. Cadik, and G. Bebis, “Comparison of semantic<sup>ˇ</sup> segmentation approaches for horizon/sky line detection,” in Neural Networks (IJCNN), 2017 International Joint Conference on. IEEE, 2017, pp. 4436–4443.

[12] S. J. Dumble and P. W. Gibbens, “Efficient terrain-aided visual horizon based attitude estimation and localization,” Journal of Intelligent & Robotic Systems, vol. 78, no. 2, pp. 205–221, 2015.

[13] B. Grelsson, M. Felsberg, and F. Isaksson, “Highly accurate attitude estimation via horizon detection,” Journal of Field Robotics, vol. 33, no. 7, pp. 967–993, 2016.

[14] S. Chiodini, M. Pertile, S. Debei, L. Bramante, E. Ferrentino, A. G. Villa, I. Musso, and M. Barrera, “Mars rovers localization by matching local horizon to surface digital elevation models,” in Metrology for AeroSpace (MetroAeroSpace), 2017 IEEE International Workshop on. IEEE, 2017, pp. 374–379.

[15] G. Baatz, O. Saurer, K. Koser, and M. Pollefeys, “Large scale visual¨ geo-localization of images in mountainous terrain,” in Computer Vision– ECCV 2012. Springer, 2012, pp. 517–530.

[16] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, and J. Sivic, “Netvlad: Cnn architecture for weakly supervised place recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 5297–5307.

[17] J. Canny, “A computational approach to edge detection.” PAMI, vol. 8, pp. 679–698, 1986.

[18] P. Hough, “Method and means for recognizing complex patterns.” U.S. Patent 3069654, 1962.

[19] Vricon, https://www.vricon.com/, 2018.

[20] A. Kendall, M. Grimes, and R. Cipolla, “Posenet: A convolutional network for real-time 6-dof camera relocalization,” in Computer Vision (ICCV), 2015 IEEE International Conference on. IEEE, 2015, pp. 2938–2946.

[21] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[22] A. L. Maas, A. Y. Hannun, and A. Y. Ng, “Rectifier nonlinearities improve neural network acoustic models,” in Proc. ICML, vol. 30, no. 1, 2013.

[23] Specification Airmar GH2183, http://www.airmar.com/uploads/ brochures/gh2183.pdf, 2018.

[24] G. Meneghetti, M. Danelljan, M. Felsberg, and K. Nordberg, “Image alignment for panorama stitching in sparsely structured environments,” in Scandinavian Conference on Image Analysis. Springer, 2015, pp. 428–439.

[25] B. Javidi and J. Wang, “Design of filters to detect a noisy target in nonoverlapping background noise,” JOSA A, vol. 11, no. 10, pp. 2604– 2612, 1994.

[26] Specification FLIR Ladybug3, https://www.ptgrey.com/ ladybug3-360-degree-firewire-spherical-camera-systems, 2018.

[27] Specification U-blox EVK-8, https://www.u-blox.com/en/product/ evk-8evk-m8, 2018.

[28] William J. Hughes Technical Center, “Global Positioning System (GPS) Standard Positioning Service (SPS) Performance Analysis Report,” Tech. Rep. 86, Jul. 2014.