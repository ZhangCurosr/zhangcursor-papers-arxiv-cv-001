# Edge-Assisted Crowdsensing for Explainable Cyclist Safety Perception

Hongsen Cao   
School of EECS   
Queen Mary University of London   
London, UK   
hong.cao@qmul.ac.uk

Chia-Yen Chiang School of EECS Queen Mary University of London London, UK

Mona Jaber School of EECS Queen Mary University of London London, UK

Meihui Wang   
School of Public Health   
Imperial College London   
London, UK

Abstract—Cyclist fatality rates remain high despite policy support for active travel, and autonomous vehicles add complexity to mixed traffic. Neither crash statistics nor multi-model detection pipelines scale to continuous monitoring of perceived safety along a route. We present an edge-assisted crowdsensing system: handlebar cameras supply street-level frames, a vision–language model (VLM) turns each frame into structured traffic attributes through a single prompt, and an explainable boosting machine (EBM) combines them with OpenStreetMap road infrastructure and on-bike kinematics to predict a crowdsourced 1–9 safety score. Extraction belongs at the network edge: the fastest selfhosted model needs 2.49 s per frame and scoring takes 0.4 ms on a CPU. On 1,362 annotated frames from two London routes, the EBM regressor reaches 72.2% binary accuracy and a mean absolute error of 1.011 under five-fold cross-validation. It is within one accuracy point of black-box ensembles. Its perfeature shape functions give exact attribution: physical lane separation is worth +0.97 rating points, a painted lane +0.70 and a construction site −0.65. Planners can therefore price an infrastructure change in rating points, and the results enter cost-benefit analysis directly. We also report inter-annotator agreement (α = 0.4695), an attribute audit, and a sensitivity analysis linking attribute error to accuracy. With re-encoding, a 20 s sampling interval serves 1,000 riders at 54 Mbit/s at a cost of 1.4 accuracy points, making continuous monitoring practical.

Index Terms—cyclist safety, perceived safety, mobile crowdsensing, vision–language models, explainable boosting machine, edge computing, mobile edge generation, spatial cross-validation

## I. INTRODUCTION

Cycling brings well-documented environmental, health and economic co-benefits [1]. Yet urban road networks were designed around motorised transport, exposing cyclists to high vehicle volumes and large speed differentials with little physical protection. Autonomous vehicles add further complexity, raising new safety challenges for vulnerable road users [2].

Crash statistics capture only part of the problem. Perceived safety is how safe a cyclist feels. It shapes route choice and ridership at least as strongly as objective risk [3], [4].

Yasmin Fathy   
Department of Engineering   
University of Cambridge   
Cambridge, UK

Questionnaire surveys [5] do not scale to continuous routelevel monitoring, and vision pipelines [6] treat safety as an objective property of the scene. Neither approach measures perceived safety along a route.

Continuous measurement is a mobile crowdsensing problem because the riders carry the sensors and the riders make the judgement. The VLM that describes each scene belongs at the network edge, and downstream stages see only its compact structured output. Mobile edge generation (MEG) formalises this split for generative models: the model runs at the edge and only a sketch is transmitted [7], [8]. The cost of such a measurement lies in uplink, edge compute and sampling rate, and any saving on them has to be weighed against accuracy. This paper contributes:

• An edge-deployable crowdsensing pipeline that turns handlebar video into structured traffic attributes through a single vision–language model (VLM) prompt and fuses them with OpenStreetMap (OSM) road infrastructure and rider kinematics. The prompt replaces a specialised multi-model detection pipeline, and the extraction stage fits in 9.2 GB of GPU memory on open-weight models, so it can be hosted on a roadside or edge server with only the attribute vector sent onward.

• An interpretable scoring stage built on an explainable boosting machine (EBM). It runs in 0.4 ms on a CPU and can run at any tier of the system. It matches the black-box baselines to within one accuracy point and gives planners per-feature contributions in rating points.

• A per-stage deployment budget for city-scale operation, measuring uplink, edge compute and sampling rate: re-encoding cuts the payload 23×, and a 20 s interval serves a 1,000-rider fleet at 54 Mbit/s at a cost of 1.4 accuracy points.

## II. RELATED WORK

Mobile crowdsensing and edge inference. Mobile crowdsensing treats participant-carried sensing as a systems problem in which coverage, participant burden and uplink cost are design variables [9], [10]. Karakaya et al. [11] treat cyclingsurface sensing this way, splitting processing between rider phones and the cloud. Serving large vision–language models at the network edge is an active research area [12], and MEG places the generative model itself at the edge so that only compact sketches traverse the network [7], [8].

Cycling safety assessment. Recent reviews address cyclist safety broadly [13] and perceived safety in particular [14]. Chataway et al. [4] find that infrastructure quality shapes how cyclists interpret risk. Questionnaires measure perceived risk under varying conditions [3], [5], but they scale poorly and cannot resolve variation along a route. Crash databases [15] record outcomes only and cannot separate objective risk from subjective experience.

Sensing and vision for scene-level safety. Street-level imagery has been used to score bikeability [16], to rate cycling safety from geospatial and visual cues [17] and to relate perceived safety to accident records [18]. Costa et al. [6] rank environments by pairwise image comparison, and the ranking does not identify which scene elements drive it. Christie et al. [19] link helmet-mounted video to nearmiss frequency, and Chiang et al. [20] release an ego-view cyclist dataset; both motivate the on-bike design. Wang et al. [21] enrich Global Positioning System (GPS) trajectories with road-network context, and we build on the same routelevel fusion.

Explainable AI in transportation. Post-hoc saliency methods explain predictions at the pixel level, whereas planners work with per-variable quantities. Generalised additive models (GAMs) [22] and their boosted variant the EBM [23] give inherently interpretable predictions and have been applied to cycling demand [24].

VLMs for structured scene understanding. Vision– language pre-training such as Contrastive Language–Image Pre-training (CLIP) [25] gives strong zero-shot transfer to new visual domains. However, traffic applications have targeted open-ended question answering and narration [26], and downstream modelling needs structured extraction. We use structured prompting to extract safety-relevant scene attributes as JavaScript Object Notation (JSON), which replaces a specialised multi-model detection pipeline with a single maintainable query. Section III-B validates the resulting attributes against human labels.

## III. METHODOLOGY

Fig. 1 summarises the pipeline in three stages: capture on the bicycle, attribute extraction at the network edge, and scoring wherever the attribute vector is consumed.

## A. Data Collection and Safety Scoring

Two cyclists rode two London routes with handlebarmounted GoPro cameras that log video, inertial data and GPS fixes. Route A covers 11 km near Queen Mary University of London and Route B 2.9 km in Farringdon. We sampled one frame every five seconds, yielding 1,419 usable frames after quality filtering (1,220 from Route A, 199 from Route B), each 5312×2988 pixels and a 3.13 MB payload. Thirty-two recordings span 21.9 km along-track: 19.1 km on Route A over 28 recordings that revisit the same streets, and 2.8 km on Route B over four. The 1,362 frames carrying a GPS fix form the modelling set, a density of 62.3 frames per riderkilometre.

![](images/c2f8899448e67bfb063408b4c91b037a833ed1c47d43be5479de842677039d58.jpg)  
Fig. 1. Three-stage pipeline: data preparation, VLM-based traffic description at the network edge, and EBM-based safety modelling with interpretable analysis.

We collected annotations on a custom crowdsourcing platform [27]. Ninety-nine annotators rated frames on a 1–9 ordinal scale from incredibly unsafe to incredibly safe. They were screened for cycling at least three times a week and sampled across age groups. Annotators supplied 7,045 ratings over the 1,252 frames for which per-annotator records were retained (5.63 ratings per frame on average, median 5, range 1–29); 1,165 of those frames were rated twice or more and 87 exactly once. The ground-truth score y for each frame is the mean of its available ratings, and we derive a binary label with a threshold τ = 5:

$$
c _ { i } = { \left\{ \begin{array} { l l } { 1 } & { y _ { i } > \tau , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{1}
$$

where $c _ { i } = 1$ denotes a safe environment. The resulting dataset is near-balanced: of the 1,362 frames in the modelling set, 45.2% are labelled safe and 54.8% unsafe. Averaged scores have a median of 4.88 and quartiles of 4.00 and 6.14, a slight skew towards the unsafe side. The quartiles bracket τ = 5, so the boundary falls inside the middle 50% of the data and splits the classes more evenly than a cut into either tail.

Agreement and rater profile. Over the multiply-rated frames, Krippendorff’s α under an ordinal difference function is 0.4695 with a 95% confidence interval (CI) of [0.4361, 0.4985], and the median across-annotator variance is 1.25. Agreement is moderate, as expected for a judgement that depends on individual risk tolerance, and it is the reason we model the mean rating. Familiarity also shifts ratings. On the 980 frames rated both by annotators who had ridden the location and by annotators who had not, a within-frame paired comparison shows that familiarity lowers the rating by 0.121 points (95% CI [0.031, 0.208]).

## B. Multimodal Feature Extraction

VLM-based traffic attributes. For each frame we issued a single prompt to Grok-2-Vision-1212 requesting a structured

TABLE I  
VLM COMPARISON ON HUMAN-AUDITED ATTRIBUTES (% ACCURACY) AND PER-TOKEN PRICE, WITH THE ALWAYS-ABSENT BASELINE FOR THE THREE ATTRIBUTES THAT CARRY CORPUS-WIDE LABELS.
<table><tr><td>Attribute</td><td>Claude Sonnet</td><td>GPT-40</td><td>Grok-2-Vision</td><td>Baseline</td></tr><tr><td>is_cycling_lane_present</td><td>68.9</td><td>55.8</td><td>83.6</td><td>67.0</td></tr><tr><td>is_cycling_lane_separation_present</td><td>83.8</td><td>55.0</td><td>86.3</td><td>80.4</td></tr><tr><td>is_intersection</td><td>38.6</td><td>63.4</td><td>53.5</td><td>85.6</td></tr><tr><td>is_construction_site_present</td><td>59.9</td><td>57.6</td><td>58.2</td><td></td></tr><tr><td>immediate_crash_risk_from_traffic</td><td>50.4</td><td>60.8</td><td>53.9</td><td></td></tr><tr><td>is_high_traffic_density</td><td>62.3</td><td>67.1</td><td>64.9</td><td></td></tr><tr><td>Average accuracy</td><td>60.7</td><td>60.0</td><td>66.7</td><td></td></tr><tr><td>$/M input tokens</td><td>3.00</td><td>2.50</td><td>2.00</td><td></td></tr><tr><td>$/M output tokens</td><td>15.00</td><td>10.00</td><td>10.00</td><td></td></tr></table>

Baseline: accuracy of always predicting the majority (absent) class,  
computed on the 1,362 frames carrying human labels for that attribute.

JSON response, and the model returned 14 attributes. Five enter the models: cycling lane, lane separation, construction site, immediate crash risk from traffic and high traffic density. The rest cover intersections, traffic signals, buses, pedestrians, surface wetness, parks and visibility, and none of them is a model input. Table I compares three VLMs against humanaudited ground truth on six visually verifiable attributes. Grok-2-Vision reaches the highest average accuracy (66.7%) at the lowest per-token cost and supplies every attribute used below.

Attribute audit. Three of the six attributes carry corpuswide human labels on 1,362 frames, and comparison against the always-absent baseline in Table I shows which of them can be trusted. Cycling lane and lane separation clear their baselines by 16.6 and 5.9 points, and we retain both. Intersection does not: at 53.5% it falls below its 85.6% baseline, because the extractor reports an intersection in 57.1% of frames against a human rate of 14.4%. We exclude it from every model. The same over-reporting affects the two retained attributes more mildly. The extractor marks them present in 43.2% and 25.0% of frames, against human rates of 33.0% and 19.6%. We audited the other three on a sample only, so they carry no baseline. Section IV-C bounds the cost of residual attribute error.

OSM road infrastructure. We map-matched GPS-tagged frames to the OSM road network by projecting each onto the nearest candidate road segment; the largest matched distance was 46.4 m. We aggregated the OSM road class (highway) to six levels, giving route-level context independent of the frames. The residual other level covers 46.9% of Route A.

Kinematic features. GoPro inertial metadata provides three-dimensional speed (speed3d) and acceleration (accl3d), capturing dynamics the frames cannot show. Eight features enter the models: the five attributes named above, the aggregated road class, and the two kinematic channels.

## C. Explainable Boosting Machine

We adopt EBMs [23], which extend GAMs [22] with gradient boosting over shallow decision trees and a small set of selected pairwise interaction terms:

$$
g ( \mathbb { E } [ Y \mid { \bf x } ] ) = \beta _ { 0 } + \sum _ { j = 1 } ^ { p } f _ { j } ( x _ { j } ) + \sum _ { i < j } f _ { i j } ( x _ { i } , x _ { j } ) .\tag{2}
$$

Here Y is the target, the score $y \in [ 1 , 9 ]$ for regression and the binary label c for classification. The vector x holds the $p = 8$ features, $\beta _ { 0 }$ is the intercept, and g(·) is the link, the identity for regression and the logit for classification. Each $f _ { j }$ is a univariate shape function learned by boosting singlefeature residuals. The terms $f _ { i j }$ capture pairwise effects, with candidate pairs selected by interaction strength [23]. The regressor minimises mean squared error and the classifier minimises cross-entropy. Each term in Eq. (2) plots as a shape function that shows how a feature raises or lowers the predicted score, and the attribution is read directly from the fitted terms.

Protocol and settings. We evaluated all models by five-fold cross-validation over the GPS-tagged frames, using KFold with shuffling and random\_state=42. Each fold holds 272–273 test frames. Folds are not stratified, so class balance varies across folds: the safe share spans 0.404–0.484. Seed 1 governs model-internal randomness. We fixed every setting in advance, so no model was tuned against the evaluation data. The configurations are EBM at the defaults of the interpret library; logistic regression max\_iter=1000; decision tree max\_depth=5; random forest 200 trees, max\_depth=8; XGBoost 200 trees, max\_depth=4, learning rate 0.05. The EBM consumes raw features; the remaining models use label encoding with median imputation, and the encoder and imputer are fitted on training folds only.

## IV. RESULTS

## A. Exploratory Feature Analysis

Kinematic features. Perceived safety increases with cycling speed up to approximately 4 m/s and then declines at higher values. Annotators rated still images and never saw rider speed, so the curve reflects the infrastructure that permits higher speeds.

Feature combination analysis. Fig. 2 shows mean scores for cycling infrastructure combined with three adverse conditions. Construction sites show the strongest negative contrast, about −1.0 points even where physical separation is present. Separation reduces the penalty for immediate crash risk from traffic to −0.1 points, compared with −0.3 under lane marking alone. A physical barrier reduces perceived danger beyond what lane marking alone achieves. Intersections cost about 0.2 points regardless of infrastructure.

Categorical feature ranking. Ranking the binary attributes by the raw contrast $\Delta = \mathrm { { m e a n } _ { Y e s } - \mathrm { { m e a n } _ { N o } } }$ agrees in direction with the combinations above. Cycling lane separation $( \Delta > 1 . 0 )$ and lane presence show the largest positive contrasts, while construction sites and immediate crash risk show the largest negative ones $( \Delta < - 0 . 4 )$ . These contrasts are marginal; the additive model of Section IV-D conditions on the remaining features.

## B. Model Performance

Classification. We compare the EBM with two interpretable baselines, logistic regression and a decision tree, and with two black-box references, a random forest and

![](images/8bafb168b85adfc3f4e54efaad1f829ac74d1db6db80ff9f057239f67ba7e57e.jpg)  
Fig. 2. Mean safety score for each combination of cycling infrastructure (column groups: cycling lane present, cycling lane separation) and adverse condition (row panels: construction site, crash risk from traffic, intersection). Each cell gives the mean score and the proportion of observations in brackets. The no-infrastructure, no-hazard cell of each column group is the baseline, and every other cell adds its signed change from it.

XGBoost. Table II reports the binary task under random fivefold cross-validation. Among interpretable models the EBM regressor attains the highest accuracy (72.2%) and weighted F1 (0.719), 5.7 points above logistic regression. It is within 1.0 point of both XGBoost (71.5%) and random forest (73.2%). Section IV-C shows that no model pair separates under a tighter protocol. The EBM regressor’s unsafe-class recall (0.815) exceeds its safe-class recall (0.623), so it errs on the side of caution.

Regression. The EBM regressor achieves a mean absolute error (MAE) of 1.011 and a root-mean-square error of 1.303 on the 1–9 scale, so the average prediction error is about one point. Within-tolerance accuracy reaches approximately 60% at ±1 and 87% at ±2. The median annotation variance is 1.25, so two individual raters routinely differ by more than a point. The target is a mean of several ratings and correspondingly less noisy, so ±2 is a practical tolerance.

## C. Robustness: Spatial Blocking and Attribute Noise

Samples from a moving sensor are spatially autocorrelated. Frames five seconds apart along a route are near-duplicates, so random folds can place almost identical frames in training and test. To quantify how much this inflates accuracy, we reran every model under a second five-fold protocol differing only in grouping. We cut each recording into contiguous 200 m along-track blocks, 124 in total, and formed folds with GroupKFold so that no block spans the train/test boundary; we call this spatial blocking. We assigned blocks from coordinates alone, before fitting any transformation. Neither protocol stratifies, so grouping is the only manipulated difference. Spatial blocking widens the per-fold safe share to 0.335–0.531, against 0.404–0.484 under random folds.

TABLE II  
MODEL COMPARISON FOR BINARY SAFETY CLASSIFICATION UNDER RANDOM FIVE-FOLD CROSS-VALIDATION. DIFFERENCES OF ABOUT ONE POINT AT THIS SAMPLE SIZE DO NOT ESTABLISH AN ORDERING; SEE SECTION IV-C.
<table><tr><td>Model</td><td>Acc</td><td>F1(W)</td><td>Rec(U)</td><td>Rec(S)</td><td>MAE</td></tr><tr><td>Interpretable</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EBM regressor (ours)</td><td>0.722</td><td>0.719</td><td>0.815</td><td>0.623</td><td>1.011</td></tr><tr><td>EBM classifier (ours)</td><td>0.711</td><td>0.710</td><td>0.799</td><td>0.633</td><td></td></tr><tr><td>Logistic regression</td><td>0.665</td><td>0.664</td><td>0.761</td><td>0.580</td><td></td></tr><tr><td>Decision tree</td><td>0.669</td><td>0.662</td><td>0.836</td><td>0.520</td><td>0.999</td></tr><tr><td>Black-box reference</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random forest</td><td>0.732</td><td>0.732</td><td>0.813</td><td>0.660</td><td>0.988</td></tr><tr><td>XGBoost</td><td>0.715</td><td>0.715</td><td>0.754</td><td>0.680</td><td>1.016</td></tr></table>

Acc: accuracy. F1(W): weighted F1. Rec(U)/Rec(S): recall for the unsafe/safe class. Mean absolute error  
(MAE) on the 1–9 scale, regression-capable models only.

Measured accuracy is higher under random folds for five of the six models. Under frame-paired comparison, the gain is 3.7 points for random forest (95% CI [1.8, 5.7], Holm $p < 0 . 0 0 1 )$ and 2.6 points for the EBM regressor ([0.8, 4.7], $p = 0 . 0 0 4 )$ The remaining gains are too small for the test to resolve, and logistic regression shows none, since a linear model cannot exploit near-duplicate neighbours. Under spatial blocking, no pair of models separates in any of the 15 exact McNemar tests with Holm correction. The ordering in Table II is therefore not a ranking, and its accuracy level holds for the random-fold protocol that produced it.

An end-to-end baseline. On the same spatially blocked folds, a linear probe on CLIP ViT-L/14 embeddings does not separate from the EBM (Holm $p = 0 . 3 6 7 6$ over seven regularisation settings), so the comparison is unresolved at this sample size. Its best setting leads by 2.6 points, and that setting was chosen on the evaluation folds. The probe also has no additive attributions.

Attribute noise. Corrupting the five VLM inputs at rate p and retraining under spatial blocking degrades accuracy roughly linearly, by about 3.3 points per ten points of added attribute error, reaching 9.8 points at $p = 0 . 3 0 .$ . The five inputs average 69.4% accuracy in Table I, so their 30.6% error rate is already at the top of the swept range. Attribute quality is the main limit on accuracy here. The model tolerates that error because it combines several weak, partly independent cues and depends on none of them alone.

![](images/376d90859be1c4469338db3c00ba13bd1f47b1e96aa613530c9e1f8c57f4fc45.jpg)  
Fig. 3. Global feature importance from the EBM, as the mean absolute weighted score. Every main-effect term outranks every pairwise interaction term.

## D. Feature Importance and Explainability

As Fig. 3 shows, individual shape functions dominate pairwise interaction terms in mean absolute contribution. In descending order, the five most influential features are physically separated cycling lane, painted cycling lane, cyclist acceleration, cyclist speed and construction site.

On the rating scale, physical separation is worth +0.97 points and a painted lane +0.70, while a construction site costs 0.65 points. A construction site reduces the benefit of protective infrastructure but does not cancel it. Road class has the widest marginal contrast of any categorical feature (∆ = 1.52), but ranks below these inside the model. Across 150 bootstrap resamples of the 124 spatial blocks, the two cycling-lane terms appear in the top three every time. Road class, construction and speed exchange positions, so we treat those three as one tier.

## V. DEPLOYMENT BUDGET: UPLINK, COMPUTE AND SAMPLING

City-scale monitoring is constrained less by accuracy than by uplink and compute cost. Scoring a frame with the trained EBM takes 0.4 ms on CPU; the fastest VLM configuration needs 2.49 s on an A100-80GB, a factor of roughly 6,000. Extraction should therefore be offloaded, as in mobile edge computing [28] and in recent work on vision–language inference at the edge [12]. In MEG terms the attribute vector is the sketch, and downstream stages work from it without running the heavy model [7]. In the proposed deployment each frame enters the edge as a 0.13 MB re-encoded image and leaves it as a 14-field JSON record.

Self-hosted extraction. A commercial endpoint charges per frame and requires rider video and GPS to leave the operator’s control. Table III benchmarks five open-weight VLMs.

TABLE III  
SELF-HOSTED EXTRACTION ON A 309-FRAME STRATIFIED SUBSET (A100-80GB, BF16, BATCH 1, 1024 PX LONG EDGE). ACCURACIES ARE ON CYCLING LANE AND LANE SEPARATION, THE TWO ATTRIBUTES THE AUDIT RETAINS.
<table><tr><td>Model</td><td>Lat. (s)</td><td>VRAM</td><td>Frames/GPU h</td><td>Lane / Sep.</td></tr><tr><td>Qwen3-VL 4B</td><td>2.74</td><td>9.2 GB</td><td>1312</td><td>0.854 / 0.816</td></tr><tr><td>Qwen3-VL 8B</td><td>2.99</td><td>17.8 GB</td><td>1205</td><td>0.819 / 0.816</td></tr><tr><td>Qwen3-VL 32B</td><td>8.94</td><td>67.2 GB</td><td>403</td><td>0.767  / 0.822</td></tr><tr><td>InternVL3 8B</td><td>2.49</td><td>16.1 GB</td><td>1444</td><td>0.877 / 0.825</td></tr><tr><td>Llama-3.2 11B</td><td>3.63</td><td>22.3 GB</td><td>992</td><td>0.796 / 0.793</td></tr></table>

Lat.: mean latency per frame. VRAM: peak GPU memory. Frames/GPU h: sustained throughput. Lane / Sep.: accuracy on cycling lane and lane separation.

![](images/acab37c76d0bcd83a5b4deecc16c4a67fc9124a0e923e8afbde070e27841aca8.jpg)  
Fig. 4. Sampling interval against accuracy under spatial blocking, with the uplink each interval implies relative to 5 s capture on the same axis. Shading marks the fold-to-fold spread of the 5 s baseline. The size-matched control keeps the 5 s cadence while using the same number of frames as the thinned setting. Bars span folds.

Accuracy on cycling lane is comparable to the commercial extractor’s, and accuracy does not rise with model size. The 4B model exceeds the 32B by 8.7 points on cycling lane (Holm $p = 0 . 0 0 3 6 )$ at 3.3× the speed in one-seventh the memory. On lane separation all five are close to the 80.4% always-absent baseline of Table I, so only the cycling-lane column discriminates. Fifteen paired tests cover five models and the three attributes with corpus-wide labels at both input resolutions. The 1024 px input is favoured in seven and native capture in none. These extractors suit a new deployment; checking equivalent downstream attribution would require reextracting the corpus.

Uplink and sampling interval. At 3.13 MB per frame and one frame every 5 s, raw capture implies 5.0 Mbit/s per rider, so a 1,000-rider fleet would demand 5.0 Gbit/s. Re-encoding to a 1024 px long edge at JPEG quality 85 cuts the payload by 23×, and a 20 s interval removes another factor of four, bringing the fleet requirement to 54 Mbit/s. Fig. 4 sweeps the interval from 5 s to 60 s against the uplink it implies. Accuracy under spatial blocking stays inside the fold-to-fold spread of the 5 s baseline out to 20 s, where the uplink is already four times smaller, and declines from 30 s. The 20 s setting costs 1.4 accuracy points. A size-matched control at the 5 s cadence tracks the thinned curve over the same range. Compute then becomes the limit: the same fleet emits 180,000 frames per hour, and extracting them at the fastest measured throughput would occupy about 125 A100-class accelerators continuously. At city scale, extraction is a datacentre-class workload at the network edge; the 0.4 ms scoring step can run wherever the attribute vector arrives, including the bicycle.

## VI. DISCUSSION AND LIMITATIONS

Experience weighting. Weighting ratings 0.8 to 0.2 in favour of annotators who had ridden the location improves regression MAE from 1.011 to 0.95 and raises accuracy by about two points, but reduces safe-class recall by roughly ten points. It also changes the target scores, so the two are not scored on identical labels; we report the unweighted model as primary.

Limitations. The VLM processes each frame independently and may miss cross-frame context; contrastive prompting across consecutive frames could address this. The dataset covers two routes in inner London, and cross-city generalisation remains to be demonstrated. Annotators cycle at least three times weekly, and familiarity alone moves a rating by 0.121 points, so the model reflects the judgement of frequent local riders. Extending the annotator pool to occasional and noncyclists is the next step.

## VII. CONCLUSION

We presented an edge-assisted crowdsensing system that turns handlebar video into a structured sketch of each scene at the network edge, fuses it with OSM road infrastructure and rider kinematics, and scores perceived safety with an explainable boosting machine. The EBM regressor achieves 72.2% binary accuracy and MAE 1.011 on a 1–9 scale. It stays within one point of the black-box ensembles under random folds, and under spatial blocking no model pair separates. With scoring some 6,000× cheaper than extraction and a 20 s interval serving a 1,000-rider fleet at 54 Mbit/s, the pipeline is a practical basis for continuous safety monitoring. The system gives planners a scale of perceived safety on which to weigh candidate interventions. Neither the structured prompt nor the additive model is specific to cycling, so the same edge-assisted pattern extends to other street-level qualities judged by the people who use the street.

## REFERENCES

[1] D. Ding et al., “The co-benefits of active travel interventions beyond physical activity: a systematic review,” Lancet Planetary Health, vol. 8, no. 10, pp. e790–e803, 2024.

[2] A. Reyes-Munoz and J. Guerrero-Ib ˜ a´nez, “Vulnerable road users and˜ connected autonomous vehicles interaction: A survey,” Sensors, vol. 22, no. 12, p. 4614, 2022.

[3] S. Cafiso, G. Pappalardo, and N. Stamatiadis, “Observed risk and user perception of road infrastructure safety assessment for cycling mobility,” Infrastructures, vol. 6, no. 11, p. 154, 2021.

[4] E. S. Chataway et al., “Safety perceptions and reported behaviour related to cycling in mixed traffic: A comparison between Brisbane and Copenhagen,” Transportation Research Part F, vol. 23, pp. 32–43, 2014.

[5] T. Gotschi¨ et al., “Towards a comprehensive safety evaluation of cycling infrastructure including objective and subjective measures,” Journal of Transport & Health, vol. 8, pp. 44–54, 2018.

[6] M. Costa et al., “Which cycling environment appears safer? Learning cycling safety perceptions from pairwise image comparisons,” IEEE Transactions on Intelligent Transportation Systems, vol. 26, pp. 1689– 1700, 2025.

[7] R. Zhong, X. Mu, Y. Zhang, M. Jaber, and Y. Liu, “Mobile edge generation: A new era to 6G,” IEEE Network, vol. 38, no. 5, pp. 47–55, 2024.

[8] R. Zhong, X. Mu, M. Jaber, and Y. Liu, “A mobile edge generation approach,” in Proc. IEEE Global Communications Conf. (GLOBECOM), 2024, pp. 4173–4178.

[9] R. K. Ganti, F. Ye, and H. Lei, “Mobile crowdsensing: current state and future challenges,” IEEE Communications Magazine, vol. 49, no. 11, pp. 32–39, 2011.

[10] B. Guo et al., “Mobile crowd sensing and computing: the review of an emerging human-powered sensing paradigm,” ACM Computing Surveys, vol. 48, no. 1, pp. 1–31, 2015.

[11] A.-S. Karakaya, L. Thomas, D. Koljada, and D. Bermbach, “A crowdsensing approach for deriving surface quality of cycling infrastructure,” in Proc. 11th IEEE Int. Conf. Cloud Engineering (IC2E), 2023.

[12] A. Sharshar, L. U. Khan, W. Ullah, and M. Guizani, “Visionlanguage models for edge networks: A comprehensive survey,” 2025, arXiv:2502.07855.

[13] A. Scarano, M. Aria, F. Mauriello, M. R. Riccardi, and A. Montella, “Systematic literature review of 10 years of cyclist safety research,” Accident Analysis & Prevention, vol. 184, p. 106996, 2023.

[14] M. C. Ferreira et al., “Identifying the determinants and understanding their effect on the perception of safety, security and comfort by pedestrians and cyclists,” Transportation Research Part F, vol. 91, pp. 136–163, 2022.

[15] CycleStreets, “BikeData: Cycling Data Map,” 2024. [Online]. Available: https://bikedata.cyclestreets.net/

[16] K. Ito and F. Biljecki, “Assessing bikeability with street view imagery and computer vision,” Transportation Research Part C, vol. 132, p. 103371, 2021.

[17] J. De Bock and S. Verstockt, “Road cycling safety scoring based on geospatial analysis, computer vision and machine learning,” Multimedia Tools and Applications, vol. 82, pp. 8359–8380, 2023.

[18] Y. Ye, C. Zhong, and E. Suel, “Unpacking the perceived cycling safety of road environment using street view imagery and cycle accident data,” Accident Analysis & Prevention, vol. 205, p. 107677, 2024.

[19] N. Christie et al., “Spatiotemporal and behavioural correlates of cycling near misses: evidence from helmet-mounted video data,” Accident Analysis & Prevention, vol. 224, p. 108272, 2025.

[20] C.-Y. Chiang et al., “AllTheDocks road safety dataset: A cyclist’s perspective and experience,” in Proc. IEEE 99th Vehicular Technology Conf. (VTC2024-Spring), 2024, pp. 1–5.

[21] M. Wang et al., “CycleTrajectory: an end-to-end pipeline for enriching and analysing GPS trajectories to understand cycling behaviour,” in Proc. 2nd ACM SIGSPATIAL Workshop on Sustainable Urban Mobility, 2024, pp. 27–32.

[22] T. Hastie and R. Tibshirani, “Generalized additive models,” Statistical Science, vol. 1, no. 3, pp. 297–318, 1986.

[23] Y. Lou, R. Caruana, J. Gehrke, and G. Hooker, “Accurate intelligible models with pairwise interactions,” in Proc. 19th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2013, pp. 623–631.

[24] E. H. Lee, “Understanding the gender gap in bike-sharing services via explainable artificial intelligence,” Transportation Research Record, vol. 2679, pp. 622–633, 2025.

[25] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. 38th Int. Conf. Machine Learning (ICML), 2021, pp. 8748–8763.

[26] C. Sima et al., “DriveLM: Driving with graph visual question answering,” in Proc. European Conf. Computer Vision (ECCV), 2024.

[27] Digital Twins for Sustainable Development Goals, QMUL, “CyclingSafety.co.uk: Crowdsourced Cyclist Safety Perception Platform,” 2024.

[28] Y. Mao, C. You, J. Zhang, K. Huang, and K. B. Letaief, “A survey on mobile edge computing: the communication perspective,” IEEE Communications Surveys & Tutorials, vol. 19, no. 4, pp. 2322–2358, 2017.