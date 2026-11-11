# From Depth to Safe Depth: Uncertainty-Aware Monocular Perception for Mobile Robot Navigation

Saeid Firouzi Daghigh and Saeed Ayat

Abstract—Monocular depth foundation models make it tempting to replace depth sensors on mobile robots with a single RGB camera. We show that, used off the shelf, this is unsafe: in a simulated indoor navigation benchmark, a standard Nav2 stack driven by Depth Anything V2 (metric) collides in 63% of 120 episodes, because the model systematically over-estimates the distance to nearby obstacles. We propose a lightweight plug-in module that turns predicted depth into safe depth. A 0.58 M-parameter head on top of the frozen depth model predicts, per pixel, a log-depth correction $\mu$ and an uncertainty σ, trained with a β-weighted Gaussian negative log-likelihood. The navigation stack consumes the one-sided bound $\hat { d } _ { \bf s a f e } = d e ^ { \mu - k \sigma }$ , where k is set by conformal calibration on a held-out world, and uncertainty also decides which pixels may be treated as floor. The correction halves the depth error (AbsRel 0.262 → 0.134) and reduces near-range over-estimation from 59.8% to 12.0%. In navigation, safe depth raises the success rate from 37% to 70% and lowers collisions from 63% to 3%, close to the 2% of ground-truth depth. A constant safety margin calibrated to the same pixel coverage reaches only 30% success because it freezes the robot; uncertainty-aware safety wins 52 of 56 discordant paired episodes against it. Gains hold on glass, texture-less walls and dark regions. Source code, pre-trained models, and test data results are publicly available at: https://github.com/saeed5959/depth safe robot

Index Terms—Collision avoidance, monocular depth estimation, uncertainty quantification, mobile robot navigation.

## I. INTRODUCTION

D <sup>EPTH</sup> <sup>perception</sup> <sup>is</sup> <sup>the</sup> <sup>basis</sup> <sup>of</sup> <sup>obstacle</sup> <sup>avoidance</sup> <sup>for</sup>mobile robots. LiDAR and RGB-D sensors provide it mobile robots. LiDAR and RGB-D sensors provide it directly, but add cost, weight and power, which is why cameraonly navigation has long been attractive for small or low-cost platforms [1], [2]. Monocular depth estimation has progressed from early multi-scale networks [6] through training on mixed datasets [7] to foundation models such as Depth Anything V2 [8] and Metric3D v2 [9], which predict metric depth zeroshot in unseen scenes [13]. Navigation systems increasingly build on such models, from monocular reconstruction for micro aerial vehicles [2] and risk-aware model predictive control [4] to off-road stacks [5] and safety layers for learned navigation policies [3], [12].

Foundation-model depth, however, is not equally reliable everywhere. Errors concentrate on texture-less surfaces, transparent objects, dark regions and, importantly, scenes outside the training distribution. For navigation the direction of the error matters: under-estimating depth makes a robot cautious, over-estimating it makes obstacles appear farther than they are and leads to collisions. A cost map built directly from predicted depth treats every pixel as equally trustworthy, which MonoMPC [4] identified as a main cause of collisions of standard Nav2 pipelines on Depth Anything point clouds.

Uncertainty quantification offers a principled way out. Aleatoric uncertainty can be learned by predicting a variance with a likelihood loss [14]; evidential regression [15] and post-hoc heads on frozen networks such as BayesCap [10] avoid sampling at test time. A recent study combined several uncertainty methods with Depth Anything V2 and concluded that likelihood-based heads give useful pixel-wise uncertainty at little cost [11]. What is less studied is how such uncertainty should enter a classical navigation stack, and whether it actually improves closed-loop safety compared with simply being more conservative everywhere.

This paper addresses that question with a deliberately simple design. Our contributions are:

• A plug-in safe-depth module: a small head on a frozen depth foundation model that predicts a per-pixel bias correction µ and uncertainty σ, and outputs the one-sided bound $d e ^ { \mu - k \sigma }$ for an unchanged Nav2 stack. k is set by conformal calibration on a held-out world.

• An analysis of how depth uncertainty interacts with a downward-looking camera: shrinking depth along rays lifts the floor into phantom obstacles. We resolve this with an uncertainty-aware ground test.

• A paired closed-loop evaluation with 120 episodes per method in procedurally generated indoor worlds with glass, blank walls and dark regions, comparing groundtruth depth, raw monocular depth, a constant margin with the same calibrated coverage, the corrected depth alone, and the full method, with McNemar tests on paired outcomes.

## II. RELATED WORK

Monocular depth estimation. Eigen et al. [6] introduced deep networks for single-image depth. MiDaS [7] showed that mixing diverse datasets yields robust zero-shot transfer, but predicts relative depth. Depth Anything V2 [8] trains a large teacher on synthetic data and distils it through pseudolabelled real images, and provides metric variants fine-tuned on indoor (Hypersim) and outdoor data. Metric3D v2 [9] resolves scale ambiguity with a canonical camera space. A recent survey [13] summarises this progress and notes that robustness and reliability remain open problems.

Monocular navigation. Camera-only navigation has been approached with learned traversability classifiers and generated depth [1], with monocular metric reconstruction feeding a motion-primitive planner [2], and with end-to-end diffusion policies [3]. MonoMPC [4] conditions a learned probabilistic collision model on the estimated point cloud and plans with risk-aware MPC. CARE [12] adds a repulsive-force safety layer to visual navigation foundation models using monocular depth. Marsal et al. [5] compare LiDAR with monocular depth inside an open-source off-road stack. Unlike these works, we keep the standard planner and cost map, and change only the depth that is fed to them.

Uncertainty in depth. Kendall and Gal [14] distinguished aleatoric and epistemic uncertainty and learned heteroscedastic variance for depth regression. Seitzer et al. [16] showed that the Gaussian NLL lets networks “give up” on hard samples and proposed the ${ \beta \mathrm { - N L L } } .$ . Evidential regression [15] predicts both uncertainty types in one pass. BayesCap [10] learns calibrated uncertainty for frozen networks, the setting we adopt. Landgraf et al. [11] benchmarked uncertainty methods on Depth Anything V2. Conformal prediction [17] provides distribution-free coverage guarantees from a calibration set; we use it to set the safety factor k.

## III. METHOD

## A. Overview

Fig. 1 summarises the pipeline. An RGB frame is passed through a frozen metric Depth Anything V2 model (ViT-S, Hypersim checkpoint, input 518 px). Its depth $d _ { \mathrm { D A } } .$ decoder features and the image feed a small head that outputs $\mu$ and $s = \log \sigma$ per pixel. The resulting depth image is converted to a point cloud and consumed by an unmodified Nav2 [18] voxel cost map, planner and controller.

## B. Correction and uncertainty head

We hook the 32-channel feature map that enters the final convolution of the DPT decoder. Together with log $d _ { \mathrm { D A } }$ and the normalised RGB image it forms a 36-channel input at half the network resolution. The head is a small U-Net: two convolution blocks, two stride-2 stages with residual dilated convolutions (dilations up to 8, for context on large uniform surfaces), two up-sampling stages with skip connections, and a $1 \times 1$ output layer, 0.58 M parameters in total. The output layer is initialised to $\mu = 0 , \sigma = 0 . 2 { \mathrm { . } }$ , so training starts exactly from the raw model. Working in log-depth makes errors multiplicative, which matches how monocular depth errors scale with distance. The corrected depth and its uncertainty are

$$
\log \hat { d } = \log d _ { \mathrm { D A } } + \mu , \qquad \sigma = e ^ { s } .\tag{1}
$$

## C. Training objective

With the residual $r = \log d _ { \mathrm { G T } } - \log \hat { d } ,$ the Gaussian negative log-likelihood is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { N L L } } = \frac { 1 } { 2 } r ^ { 2 } e ^ { - 2 s } + s . } \end{array}\tag{2}
$$

Its minimum over s lies at $\sigma = | \boldsymbol { r } |$ , so the head learns, without any uncertainty labels, the typical error size of each kind of pixel, while $\mu$ learns the systematic part of the error. Because the gradient of $\mu$ is scaled by $\sigma ^ { - 2 }$ , plain NLL tends to inflate σ on hard pixels instead of correcting them. We therefore use the β-NLL [16], multiplying the loss by $\mathrm { s g } ( \sigma ^ { 2 \beta } )$ with $\beta \ : = \ : 0 . 5$ , where sg stops the gradient. Pixels are weighted by $w = 1 / \operatorname* { m a x } ( d _ { \mathrm { G T } } , 0 . 5 \mathrm { m } )$ , since nearby obstacles cause collisions, and pixels with invalid ground truth or depth outside [0.1, 10] m are masked.

## D. Safe depth and calibration

The depth passed to navigation is the one-sided lower bound

$$
\hat { d } _ { \mathrm { s a f e } } = d _ { \mathrm { D A } } e ^ { \mu - k \sigma } .\tag{3}
$$

Instead of fixing $k = 2 ,$ , which assumes a perfectly calibrated Gaussian, we choose k by split-conformal calibration [17]: on a held-out validation world we compute the normalised scores $z = ( \log \hat { d } - \log d _ { \mathrm { G T } } ) / \sigma$ of pixels closer than 3 m and set k to their 97.7% quantile, so that $d _ { \mathrm { G T } } \geq \hat { d } _ { \mathrm { s a f e } }$ holds for 97.7% of near pixels on that world. This gave $k = 3 . 0$ . For the constantmargin baseline $d _ { \mathrm { D A } } e ^ { - c }$ , c is calibrated the same way to the same coverage, which gave $c = 0 . 7 3$ (depth scaled by 0.48). Both baselines are therefore equally safe at the pixel level; they differ only in where the margin is spent.

## E. Integrating safe depth with the cost map

Phantom floor. The camera looks down at the floor. Shrinking depth along a ray moves a floor point towards the camera and upwards: for a camera at height H, a floor pixel whose depth is under-estimated by a fraction ϵ is lifted by $H \epsilon$ at any range. With $H = 0 . 5 5 \mathrm { { m } }$ , a 14% under-estimate already exceeds the cost map’s obstacle threshold, and safe depth turned the whole floor into lethal cells. We therefore classify ground pixels explicitly. Each pixel’s ray meets the floor plane at depth $d _ { \mathrm { f l o o r } } .$ obtained from the camera intrinsics and its pose above the floor. A pixel is ground if

$$
\log d _ { \mathrm { e s t } } - \log d _ { \mathrm { f l o o r } } \geq - \tau ,\tag{4}
$$

where $d _ { \mathrm { e s t } }$ is the method’s best estimate $( d _ { \mathrm { D A } }$ for the raw and margin variants, $\hat { d }$ otherwise), and its depth is replaced by $d _ { \mathrm { f l o o r } } ,$ so it clears but never marks the cost map. We use $\tau ~ = ~ 0 . 1 5$ for all variants and, for the safe variant, $\tau = \operatorname* { m a x } ( 0 . 1 5 , k \sigma )$ : a pixel is treated as floor if the floor lies within its own uncertainty band. Uncertainty thus decides both where obstacles are and whether a pixel is an obstacle at all.

Range and timing. Monocular depth is reliable at short range, so for all monocular variants obstacles are marked within 2.5 m and cleared by ray tracing up to 5 m (4 m and 5 m for ground truth). Because inference finishes about one second after the image was captured, the depth node republishes the camera intrinsics with the depth image’s time stamp; otherwise the point-cloud conversion silently drops most frames.

## IV. EXPERIMENTAL SETUP

## A. Simulation and robot

Experiments run in Gazebo Fortress with ROS 2 Humble. The robot is a differential-drive base $( 0 . 4 0 \times 0 . 3 0 \mathrm { m }$ , maximum speed 0.3 m/s) with a 640×480 RGB-D camera $( 6 0 ^ { \circ }$ horizontal field of view) mounted 0.55 m above the floor and pitched down by 0.2 rad. The depth channel of the same camera provides ground truth. To isolate perception, all methods use ground-truth odometry, the same Nav2 configuration (NavFn planner, DWB controller, voxel cost maps, robot radius 0.25 m, inflation 0.55 m), and identical simulation settings.

b  
![](images/8b208a0eecb5616fc727ccc22357fe69a49cbfbc4b346c801f3fd0e5dba2e67b.jpg)  
Fig. 1. Safe-depth pipeline. A frozen Depth Anything V2 model predicts metric depth $d _ { \mathrm { D A } }$ . A small head, fed with the decoder features, log d and th RGB image, predicts a per-pixel log-depth correction $\mu$ and uncertainty σ. The lower bound $d _ { \mathrm { D A } } e ^ { \mu - k \sigma }$ , with k set by conformal calibration on a validation world, passes through an uncertainty-aware ground test before an unmodified Nav2 stack builds its cost map. Only the orange components are ours; the blue model and green navigation stack are unchanged.

## B. Worlds

A procedural generator builds indoor worlds from a seed: an 8 m corridor (1.8 m wide) with doors, pictures and cardboard boxes, opening into a 7 × 8 m hall with furniture from the AWS RoboMaker small-house set (Fig. 2). Three hard cases can be enabled: a free-standing glass partition (tinted pane with a frosted safety band and rails), blank texture-less white walls, and a dark half of the hall lit at about 10%. Five training worlds use different subsets of hard cases; one of them (train\_03) is held out for validation and calibration. Four test worlds are never used for training or tuning: test\_clean without hard cases, and test\_01–03 with all three.

## C. Episodes and metrics

Each test world has 30 episodes: the four canonical goals from the spawn pose and 11 sampled start-goal pairs, each run twice with small start perturbations (±0.1 m, ±10<sup>◦</sup>), giving 120 paired episodes per method (Fig. 3). Starts and goals keep at least 0.45 m clearance and are connected by an A\* path for a 0.30 m robot. An episode is labelled easy with line of sight, hard if its path passes a hard case or a narrow gap, and medium otherwise. It ends with success when Nav2 reaches the goal within 0.3 m, collision on any contact of the robot body (bumper sensor), timeout after $2 . 5 L _ { A ^ { * } } / v _ { \mathrm { m a x } } + 2 0 s ,$ , stuck after 40 s without 0.1 m progress, or failure when Nav2 aborts. We report success rate with Wilson 95% intervals, outcome rates, SPL, time to goal and minimum clearance. Because all methods run the same episodes, we compare them with exact McNemar tests on the discordant pairs.

a  
![](images/ae06936b0f2e7f43795fb44bbc68b344c69562b00e8ad6ee8e409cef198e7698.jpg)

![](images/02fb52006077974af9ca872ec152e564aad11edc4a1f49ad8e080f91dca09eb5.jpg)  
Fig. 2. Simulation environment. (a) Corridor seen from behind the robot. (b) View from the corridor into the hall with furniture, boxes and the glass partition. (c) Top view of test world test\_01 with walls, blank panels, boxes, furniture, the glass partition, the dark region and the four canonical goals; grey marks the area reachable by a robot of radius 0.30 m.

## D. Training data

RGB images and ground-truth depth were recorded on the five training worlds while the robot executed episodes with both ground-truth and raw monocular depth, so that the data include the close approaches that precede monocular collisions. A frame was stored every 0.2 m or $1 0 ^ { \circ }$ of motion. Training uses 7,045 frames from four worlds; train\_03 is used for validation and calibration. The head was trained for 5 epochs with AdamW (learning rate $1 0 ^ { - 3 }$ , one-cycle schedule), batch size 6, horizontal flips and brightness augmentation, on a single GTX 1650 (4 GB) with the depth model frozen in half precision.

![](images/61034e73e894aab4b2653e44452dd9983953b40b5e8db07c5cb8918a5c0b6027.jpg)  
Fig. 3. Start poses (arrows), goals (stars) and A\* reference paths of the sampled episodes in $\tt t e s t \_ 0 1$ , coloured by difficulty: easy (green), medium (orange), hard (red). Cyan and magenta outline the glass partition and the blank panel.

TABLE I  
DEPTH ACCURACY ON THE TEST WORLDS. NEAR OVER-ESTIMATION: FRACTION OF PIXELS CLOSER THAN 3 M WHOSE PREDICTION EXCEEDS THE TRUTH BY MORE THAN 0.2 M.
<table><tr><td>Method</td><td>AbsRel ↓</td><td> $\delta _ { 1 . 2 5 } \uparrow$ </td><td>Near over-est. ↓</td><td>Mean σ</td></tr><tr><td>DA2 raw</td><td>0.262</td><td>62.5%</td><td>59.8%</td><td></td></tr><tr><td> $\mathrm { D A } 2 + \mathrm { h e a d } \left( \mu \right)$ </td><td>0.134</td><td>91.9%</td><td>12.0%</td><td>0.131</td></tr></table>

## E. Compared methods

GT: ground-truth depth (upper bound). Raw: $d _ { \mathrm { D A } } .$ . Margin: $d _ { \mathrm { D A } } e ^ { - c } , c = 0 . 7 3 .$ . µ: corrected depth <sup>ˆ</sup>d. Safe: $\hat { d } _ { \mathrm { s a f e } }$ with $k =$ 3.0. All monocular variants share the same depth node, ground test and cost-map settings.

## V. RESULTS

## A. Depth accuracy

Table I reports depth accuracy on the test worlds. Raw Depth Anything V2 is far from its benchmark accuracy in this setting: the mean relative error is 26%, only 62.5% of pixels lie within 25% of the truth, and 59.8% of pixels closer than 3 m are over-estimated by more than 0.2 m. The error is not a constant scale factor, as the ratio of predicted to true depth varied between roughly 0.8 and 2 across scenes. The head halves AbsRel, raises $\delta _ { 1 . 2 5 }$ to 91.9% and cuts dangerous nearrange over-estimation five-fold. On the validation world the area under the sparsification error curve was 0.028, indicating that σ ranks errors well.

## B. Navigation

Table II summarises the 120 paired episodes. With groundtruth depth the stack succeeds in 93% of episodes. Raw monocular depth collides in 63%; it never times out or gets stuck, which is the signature of over-confidence: obstacles are placed too far away and the robot drives into them. Its minimum clearance of 0.25 m is exactly the contact distance of the body.

The constant margin removes most collisions (5%) but leaves the robot stuck or timed out in 65% of episodes.

Shrinking every depth by half narrows the 1.8 m corridor and the gaps between furniture until no path remains. It is less successful than raw depth. Correcting the bias (µ) raises success to 60% and lowers collisions to 15%. Adding uncertainty gives the best monocular result: 70% success with 3% collisions, close to the 2% of ground truth, and a minimum clearance of 0.45 m. The safe variant takes slightly longer than ground truth (26 s vs. 24.5 s) because it keeps more distance where it is unsure.

Table III shows that the differences are significant on paired outcomes. The safe variant wins 45 of 50 discordant episodes against raw depth and 52 of 56 against the margin. Adding σ to the correction helps on its own (20 vs. 8, $p = 0 . 0 3 6 )$ . Ground truth remains clearly better (30 vs. 2).

## C. Hard cases

Table IV breaks success down by the hard case an episode’s path passes; an episode can count in several groups. The ordering raw $< \mu <$ safe holds in every group, the margin is never better than $\mu ,$ and the safe variant improves over raw depth by 33-37 percentage points in each group, including the scenes without hard cases. Glass remains the hardest case for all monocular variants: the safety band and rails provide only weak cues, and the glass groups are small (16 episodes), so their intervals are wide. Note that the simulated depth camera sees glass as solid, which makes the ground-truth baseline optimistic compared with real depth sensors.

## VI. DISCUSSION AND LIMITATIONS

Where the margin goes matters. The margin and the safe variant are calibrated to the same pixel coverage, yet their outcomes differ by 40 percentage points. A uniform margin spends caution everywhere, including on well-textured walls and furniture that the model sees reliably. σ spends it where errors actually occur. Bias correction alone is not sufficient either: $\mu$ still collides in 15% of episodes, because its remaining errors are not systematic and only the uncertainty term can cover them.

Uncertainty and geometry. Lowering depth along camera rays has side effects that a cost map does not expect: it lifts the floor and pulls side walls inwards. The first we resolve with the uncertainty-aware ground test. The second remains: with large k, narrow passages shrink, and part of the remaining 27% of non-collision failures are such conservative stops. A calibration target defined on navigation outcomes, or applying the bound only to nearby obstacles, could trade these off better than the pixel-level coverage used here.

Limitations. All experiments are in simulation with one robot, one camera configuration and one depth model (ViT-S). Ground-truth odometry isolates perception but hides localisation error. The ground test assumes a flat floor. The head is trained on simulated data from the same generator family as the test worlds, although on different layouts, so transfer to real images needs to be verified. Monocular inference at about 1 Hz on a 4 GB GPU also limits the speed at which the robot can safely drive.

TABLE II  
CLOSED-LOOP NAVIGATION ON 120 PAIRED TEST EPISODES. SUCCESS WITH WILSON 95% INTERVAL; OUTCOME RATES; SPL; MEAN TIME OF SUCCESSFUL EPISODES; MEDIAN MINIMUM CLEARANCE.
<table><tr><td>Method</td><td>Success [95% CI] ↑</td><td>Collision↓</td><td>Stuck/timeout/fail↓</td><td> $\mathrm { { S P L } \uparrow }$ </td><td>Time (s)</td><td>Min. clear. (m)</td></tr><tr><td>GT depth</td><td>0.93 [0.87, 0.96]</td><td>0.02</td><td>0.05</td><td>0.92</td><td>24.5</td><td>0.49</td></tr><tr><td>Mono raw</td><td>0.37 [0.29, 0.46]</td><td>0.63</td><td>0.00</td><td>0.36</td><td>14.7</td><td>0.25</td></tr><tr><td>Mono + margin  $( c = 0 . 7 3 )$ </td><td>0.30 [0.23, 0.39]</td><td>0.05</td><td>0.65</td><td>0.28</td><td>18.0</td><td>0.45</td></tr><tr><td> $\mathbf { M o n o } + \mu$ </td><td>0.60 [0.51, 0.68]</td><td>0.15</td><td>0.25</td><td>0.55</td><td>22.0</td><td>0.38</td></tr><tr><td> $\mathrm { M o n o } + \mu - k \sigma \ \mathrm { ( o u r s , } \ k = 3 . 0 )$ </td><td>0.70 [0.61, 0.77]</td><td>0.03</td><td>0.27</td><td>0.63</td><td>26.0</td><td>0.45</td></tr></table>

TABLE III  
EXACT MCNEMAR TESTS ON PAIRED EPISODES. b: FIRST METHOD SUCCEEDS AND THE SECOND FAILS; c: THE REVERSE.
<table><tr><td>Comparison</td><td>b</td><td>C</td><td>p</td></tr><tr><td>Safe vs. raw</td><td>45</td><td>5</td><td> $4 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>Safe vs. margin</td><td>52</td><td>4</td><td> $1 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>µ vs. raw</td><td>33</td><td>5</td><td> $4 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Safe vs. µ</td><td>20</td><td>8</td><td> $0 . 0 3 6$ </td></tr><tr><td>Safe vs. GT</td><td>2</td><td>30</td><td> $2 \times 1 0 ^ { - 7 }$ </td></tr></table>

TABLE IV  
SUCCESS RATE BY HARD CASE ON THE PATH.
<table><tr><td>Hard case</td><td>GT</td><td>Raw</td><td>Margin</td><td>μ</td><td>Safe</td></tr><tr><td>Glass</td><td>0.94</td><td>0.00</td><td>0.25</td><td>0.31</td><td>0.37</td></tr><tr><td>Blank wall</td><td>0.98</td><td>0.31</td><td>0.25</td><td>0.54</td><td>0.67</td></tr><tr><td>Dark</td><td>0.96</td><td>0.25</td><td>0.21</td><td>0.50</td><td>0.58</td></tr><tr><td>None</td><td>0.88</td><td>0.39</td><td>0.35</td><td>0.65</td><td>0.76</td></tr></table>

## VII. CONCLUSION

Off-the-shelf monocular depth from a foundation model is not a safe drop-in replacement for a depth sensor: in our benchmark it made a standard navigation stack collide in most episodes. A small head on the frozen model, trained to predict a correction and an uncertainty, and a conformally calibrated lower bound turn this depth into safe depth that an unmodified Nav2 stack can use. It nearly doubles the success rate and brings collisions close to those of ground-truth depth, while a constant margin with the same calibrated safety level leaves the robot unable to move. Future work will validate the module on real robots, use navigation-aware calibration, and combine aleatoric with epistemic uncertainty for unseen environments.

## REFERENCES

[1] H. Ghasemi, E. Balazadeh, S. Malekpour, A. Kalhor, and M. Tale Masouleh, “Autonomous unmanned aerial vehicle indoor navigation based on deep learning approaches,” in Proc. Int. Conf. Robotics and Mechatronics (ICRoM), 2023.

[2] N. Simon and A. Majumdar, “MonoNav: MAV navigation via monocular depth estimation and reconstruction,” in Proc. Int. Symp. Experimental Robotics (ISER), 2023, arXiv:2311.14100.

[3] A. Sridhar, D. Shah, C. Glossop, and S. Levine, “NoMaD: Goal masked diffusion policies for navigation and exploration,” in Proc. IEEE Int. Conf. Robotics and Automation (ICRA), 2024, arXiv:2310.07896.

[4] B. Sharma, P. Jadhav, P. Paul, K. M. Krishna, and A. K. Singh, “MonoMPC: Monocular vision based navigation with learned collision model and risk-aware model predictive control,” 2025, arXiv preprint.

[5] R. Marsal, Q. Picard, A. Poire, S. Kerbourc’h, T. Toralba, C. Yver,´ A. Chapoutot, and D. Filliat, “An open-source LiDAR and monocular off-road autonomous navigation stack,” 2026, arXiv:2604.03096.

[6] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” in Advances in Neural Information Processing Systems (NeurIPS), 2014.

[7] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards robust monocular depth estimation: Mixing datasets for zero-shot crossdataset transfer,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 3, pp. 1623-1637, 2022.

[8] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth Anything V2,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.

[9] M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen, K. Wang, G. Yu, C. Shen, and S. Shen, “Metric3D v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation,” IEEE Trans. Pattern Anal. Mach. Intell., 2024.

[10] U. Upadhyay, S. Karthik, Y. Chen, M. Mancini, and Z. Akata, “BayesCap: Bayesian identity cap for calibrated uncertainty in frozen neural networks,” in Proc. Eur. Conf. Computer Vision (ECCV), 2022.

[11] S. Landgraf, R. Qin, and M. Ulrich, “A critical synthesis of uncertainty quantification and foundation models in monocular depth estimation,” 2025, arXiv:2501.08188.

[12] J. Kim, J. Sim, W. Kim, K. Sycara, and C. Nam, “Enhancing safety of foundation models for visual navigation through collision avoidance via repulsive estimation,” 2025, arXiv:2506.03834.

[13] M. Liu, X. Lyu, Y.-T. Sun, Y.-H. Huang, Z. Yang, P. Dai, and X. Qi, “Monocular depth estimation from a single image: Progress and opportunities,” 2026, arXiv:2609.01172.

[14] A. Kendall and Y. Gal, “What uncertainties do we need in Bayesian deep learning for computer vision?” in Advances in Neural Information Processing Systems (NeurIPS), 2017.

[15] A. Amini, W. Schwarting, A. Soleimany, and D. Rus, “Deep evidential regression,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[16] M. Seitzer, A. Tavakoli, D. Antic, and G. Martius, “On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks,” in Proc. Int. Conf. Learning Representations (ICLR), 2022.

[17] A. N. Angelopoulos and S. Bates, “Conformal prediction: A gentle introduction,” Found. Trends Mach. Learn., vol. 16, no. 4, pp. 494-591, 2023.

[18] S. Macenski, F. Mart´ın, R. White, and J. Gines Clavero, “The Marathon´ 2: A navigation system,” in Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS), 2020.