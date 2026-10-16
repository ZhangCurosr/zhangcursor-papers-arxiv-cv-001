# Bridging the Markerless Tracking Validation Gap: Intraoperative X-ray Workflow Integration

Connor Daly<sup>1(B)</sup>, Salvatore Russo<sup>2</sup>, Jinendra Ekanayake<sup>3</sup>, Daniel S. Elson<sup>1</sup>, and Ferdinando Rodriguez y Baena<sup>1</sup>

<sup>1</sup> The Hamlyn Centre for Robotic Surgery, Imperial College London, London, UK, cd1723@ic.ac.uk

2 Department of Neurosurgery, Imperial College Healthcare NHS Trust, London, UK 3 Stanford University School of Medicine, Stanford University, California, USA

Abstract. Markerless intraoperative tracking of spinal anatomy via RGB D sensors is a promising alternative to invasive bone-anchored fiducials, yet no existing clinical dataset provides independent subsurface ground truth for validating these methods. We address this gap with a clinically viable pipeline that derives per-vertebra 6-DoF pose from routine intraoperative C-arm fluoroscopy, co-registered to the RGB-D camera frame through a multimodal fiducial array detectable in both imaging modalities. Critically, the pipeline requires no additional radiation, instrumentation, or deviation from standard surgical workflows, relying solely on the fluoroscopic images already acquired during routine level checks. A calibrated transform chain links the vertebral pose recovered via multi-view 2D–3D CT-to-fluoroscopy registration to the RGB-D coordinate frame, enabling direct comparison of any markerless registration estimate against a subsurface reference. Ex vivo validation against optically tracked ovine vertebrae confirms end-to-end accuracy of 2.06 ± 0.29 mm and 3.35 ± 0.93°, suficient to serve as independent ground truth for methods with substantially larger errors. We demonstrate the pipeline’s utility by evaluating two markerless registration strategies on clinical data spanning 11 vertebrae across 1,500 frames from three surgical sequences: the first such evaluation against an independent subsurface reference in a clinical setting. The proposed framework provides reusable infrastructure for benchmarking emerging markerless tracking methods on real surgical data.

Keywords: RGB-D Tissue Tracking · Orthopedic Surgery · 2D–3D Registration

## 1 Introduction and Motivation

Markerless intraoperative tracking of orthopedic anatomy via RGB-D sensors is a promising emerging direction for surgical guidance [11, 15]. Current surgical navigation systems depend on invasive bone-anchored fiducials to register (and subsequently track) preoperative imaging with the surgical scene [26, 27], introducing added operative time and complication risk [8]. Markerless approaches instead extract anatomically meaningful correspondences directly from intraoperative point clouds via learned segmentation, enabling registration and tracking without physical instrumentation [11, 15]. Daly et al. [5] advanced this direction with the first clinical markerless tracking dataset for spine surgery. However, their ground truth is inherently approximate: alignment relies on surgeon-indicated surface landmarks refined against intraoperative topology, neither of which references the pose of the subsurface anatomy. For clinical adoption, markerless systems must be validated against a gold-standard measurement ofering quantifiable translational and rotational error metrics [1], a requirement no existing clinical dataset satisfies. Cadaveric studies have obtained accurate vertebral pose through rigidly fixed optical tracking arrays [17, 16]. Liebmann et al. [16], for example, fixed 3D-printed constructs to vertebral bodies via K-wires, achieving sub-2 mm accuracy but requiring anterior exposure incompatible with standard posterior approaches, additional tracking hardware within the operative field, and an intraoperative CT. These requirements are not permissible within a clinical observational study, leaving a methodological gap: there is currently no clinically viable means of obtaining precise vertebral pose ground truth for evaluating markerless registration in real surgical settings.

To address this, we present a pipeline that leverages routine intraoperative C-arm imaging to derive ground-truth vertebral pose within the RGB-D camera frame. Our contributions include:

1. A multimodal data capture and labelling pipeline that co-registers intraoperative RGB-D recordings with ground-truth vertebral poses derived from routine intraoperative fluoroscopy, requiring no additional radiation, no instrumentation within the surgical field, or deviation from standard surgical workflows.

2. The first quantitative evaluation of markerless spine registration methods on clinical data against an independent subsurface reference, reporting pervertebra translational and rotational error across three surgical sequences.

3. A clinically grounded evaluation framework, validated ex vivo to 2.06±0.29 mm and 3.35±0.93°, which can serve as reusable infrastructure for benchmarking markerless registration methods against an independent subsurface reference. Upon publication we will release CAD files for the intraoperative marker array and calibration phantom, along with the evaluation code and annotated data.

This work reports initial results from a large-scale ongoing clinical data collection study; the pipeline and evaluation framework are designed to scale as further cases are captured, enabling progressively larger-scale benchmarking of markerless methods.

## 2 Methodology

## 2.1 Data Capture

In contrast to the use of coarse surface features to approximate the underlying pose of the spine by Daly et al. [5], we integrated routine intraoperative C-arm fluoroscopy, taken alongside the RGB-D recording, to locate the patient’s preoperative anatomy within the intraoperative scene. C-arm fluoroscopy is broadly accessible across clinical settings [3, 19], ensuring our pipeline generalizes across institutions without requiring intraoperative CT [14]. We used established multiview 2D–3D registration algorithms, as implemented in the ImFusion software suite (ImFusion GmbH, Germany) [12], requiring at least two fluoroscopic images from separate poses with known projection geometry.

![](images/ecf2df9e90ef7b9811267f6bb57566de34c158c27db66e10a9504d1f344baa9e.jpg)  
Fig. 1: (A) Intraoperative setup with the marker inside a sterile cover, secured taut with a rubber band by the scrub nurse to maintain detectability. The overlay shows automatic pose detection and projection of the preoperative CT into the surgical scene using the transform chain in (I); (B, C) Automatic marker pose detection in C-arm images acquired during a fixation case, with projection errors (pixels) overlaid; (D) AP DRR images, showing CT registered to C-arm images in ImFusion; (E) Calibration phantom under fluoroscopy; (F) Marker array under fluoroscopy; (G) Marker array under fluoroscopy with ovine vertebrae; (H) Data processing overview, showing how transforms needed to project preoperative CT into the intraoperative RGB-D stream are acquired. (I) Solid lines depict our method’s full transform chain enabling projection of vertebra j’s CT into the Azure Kinect RGB-D camera frame (dashed line, $\mathbf { T } _ { \mathbf { V } _ { j } } ^ { \mathbf { R G B D } } ,$ ).

We modeled each C-arm source as a pinhole camera [23, 10, 18] and calibrated intrinsics using a custom multi-plane phantom [4, 6] with nine radioopaque ArUco markers [9] at known depths, estimating parameters via bundle adjustment [2, 23]. Calibration images were acquired prior to surgery from AP and oblique angles within ±30°, reflecting the range of intraoperative C-arm positioning.

## 2.2 Intraoperative Capture

This study received approval from the relevant national research ethics authority (reference withheld for anonymization). The cases we captured within our surgical study were taken from the clinical caseload of a tertiary neurosurgery department. The captured procedures are all lumbar-thoracic fixation cases. For recording RGB-D sequences, we used an Azure Kinect camera (Microsoft Corporation, USA), positioned directly over the incision using a boom arm fixed to a tripod.

To co-register fluoroscopic and RGB-D coordinate frames without external tracking hardware, we designed a multimodal fiducial array, automatically detectable in both imaging modalities. Five radiopaque ArUco markers, each 30 × 30 mm, are inset on a radio-translucent PETG body. The insets were 3D-printed using PLA XRS filament (Nanovia Tech, France). The non-coplanar array layout ensured no marker overlap occurs at either AP or oblique viewing angles. The array was held on a boom arm adjacent to the exposition site, beneath the C-arm detector and within the RGB-D camera field of view, and covered with an ultrasound probe cover to maintain the sterile field.

An intraoperative C-arm is routinely used to provide confirmation of the vertebra level being operated on, typically from an AP view with adjacent oblique views. As the marker array was adjacent to the exposition, it is visible in the Xray field of view during these level check shots, allowing both the determination of relative C-arm pose and the co-registration of X-ray and RGB-D coordinate frames to be performed with the same array. For each clinical case, we also collected the accompanying preoperative CT scan. We segmented the relevant bone tissue from CT using the network developed by Meng et al. [21].

## 2.3 CT Imaging to RGB-D Registration

For each C-arm shot, using the acquired intrinsic parameters from our multi-view calibration, the radiopaque ArUco markers are automatically detected within the image, and we then recovered the board-to-detector transform via PnP pose estimation [7, 13]. Designating the first acquired image as the reference frame, we computed relative transforms between shots through the common marker array coordinate system, and combined these with the previously calibrated intrinsic parameters to yield projection matrices suitable for multi-view 2D–3D registration. Multi-view 2D-3D X-ray to CT registration is a well-established technique capable of achieving sub-millimeter accuracy [20]. We then performed intensity-based 2D–3D registration, optimizing over six rigid-body parameters by maximizing mutual information between digitally reconstructed radiographs (DRRs) and the acquired fluoroscopic images, using the ImFusion software suite.

This process was repeated for each segmented vertebra $\mathbf { V } _ { j }$ , yielding the transform from $\mathbf { V } _ { j }$ to the detector coordinate frame, $\mathbf { D } _ { r e f }$ , at the reference position. The transforms $\mathbf { T } _ { B } ^ { \mathbf { D } _ { r e f } }$ and $\mathbf { T } _ { B } ^ { \mathbf { R G B D } }$ , locating the marker array in the reference detector frame and the RGB-D camera frame respectively, were both recovered via ArUco detection and PnP pose estimation.

Thus, we obtained the complete transform chain from the coordinate system of the vertebra $\mathbf { V } _ { j }$ to the camera C:

$$
\mathbf { T } _ { \mathbf { V } _ { j } } ^ { \mathbf { R G B D } } = \mathbf { T } _ { \mathcal { B } } ^ { \mathbf { R G B D } } \cdot \left( \mathbf { T } _ { \mathcal { B } } ^ { \mathbf { D } _ { r e f } } \right) ^ { - 1 } \cdot \mathbf { T } _ { \mathbf { V } _ { j } } ^ { \mathbf { D } _ { r e f } }
$$

## 3 Evaluation and Results

## 3.1 Ex Vivo Validation

To demonstrate the accuracy and repeatability of our data collection and labeling pipeline, we conducted several validation experiments. We first established the level of error introduced by each individual component of the transform chain. To do this, we first rigidly fixed a passive infrared tracking array to the multimodal marker array. We then directly compared how well the marker array itself could be tracked in both the C-arm imaging, as well as the Azure Kinect RGB frames, with respect to a highly accurate optical tracker (Polaris Vicra, Northern Digital Inc, Canada; 0.25 mm RMS volumetric accuracy, 0.50 mm 95% CI) [24]. For each imaging system, the marker array was moved between 4 distinct points in front of the sensor. We present the results in Tab. 1.

Subsequently, to validate the end-to-end accuracy of the full transform chain, we compared the diference in pose of four excised ovine vertebrae determined by our pipeline, with that found by directly tracking ArUco markers, rigidly fixed to the vertebrae. We mounted two plastic ArUco markers to each of the vertebrae with plastic screws, minimizing the markers’ visibility within the C-arm imaging. We then performed CT scans of each vertebra. Within the coordinate space of the generated scans, we then localized the ArUco positions, meaning then that each vertebra could be tracked directly via the RGB-D camera, by-passing the transform chain shown in Fig. 1 (I) entirely.

To compute registration diferences, we selected two points (the tip of the spinous process, and a point on the superior laminar margin) for establishing a Total Registration Error (TRE), with the optically tracked pose representing the target. The vertebrae and radio-opaque marker array were positioned between the source and detector of a TAU 2020 C-arm (Ziehm-Orthoscan Inc, USA), as described in section 2.2. With the radio-opaque marker array visible from the RGB-D camera, we captured 2 C-arm shots with an angular disparity of $3 0 ^ { \circ }$ . This is comparable to the mean angular disparity of $2 5 . 6 ^ { \circ }$ seen in clinical cases we captured. Following the process described in 2.3, we projected the vertebrae from the CT coordinate space to the RGB-D camera coordinate space. We present the TRE for the point landmarks as well as the rotational diference in Tab. 2. The independent validation of the RGB-D and X-ray pose track-

Table 1: Independent validation of each coordinate recovery stage against optical tracker ground truth. RGB-D tracking is evaluated with the sterile probe cover in place. Transitions 1–3 represent lateral displacement within the image plane; transition 4 represents displacement along the sensor optical axis.
<table><tr><td>Modality</td><td> $\overline { { 1 \to 2 } }$  (mm) (°)</td><td> $\overline { { 1 \to 3 } }$  Trans. Rot.Trans. Rot. (mm)</td><td>(°)</td><td> $\overline { { 1 \to 4 } }$  Trans. Rot. (mm)</td><td>(°)</td></tr><tr><td>RGB-D tracking X-ray tracking (AP) X-ray tracking (oblique)</td><td>0.40 0.58 0.46</td><td>0.31 0.03 0.18</td><td>0.70 1.19 0.96</td><td>0.61 0.07 0.50</td><td>0.10 1.85 0.02 0.35 0.66 0.38</td></tr></table>

Table 2: Ex vivo registration accuracy across ovine thoracic vertebrae. Combined TRE represents the mean of both anatomical landmark measurements.
<table><tr><td>Vertebra</td><td></td><td>[Combined TRE (mm) Rotational Error (°)</td></tr><tr><td>T1</td><td>1.96</td><td>4.06</td></tr><tr><td>T2</td><td>2.32</td><td>3.40</td></tr><tr><td>T3</td><td>1.62</td><td>3.94</td></tr><tr><td>T4</td><td>2.32</td><td>1.98</td></tr><tr><td>Agg.</td><td> $\mathbf { \overline { { { 2 . 0 6 \pm 0 . 2 9 } } } }$ </td><td> $\mathbf { \overline { { { 3 . 3 5 \pm 0 . 9 3 } } } }$ </td></tr></table>

ing against an optical tracker reference confirms sub-millimeter translational agreement and sub-degree rotational agreement. The end-to-end pipeline, validated ex vivo against fiducial-tracked vertebral pose, yields a combined TRE of 2.06±0.29 mm and rotational error of $3 . 3 5 { \pm } 0 . 9 3 ^ { \circ }$ , approaching the $\leq 2$ mm and $\leq 5 ^ { \circ }$ thresholds commonly cited for spinal navigation systems [22]. Critically, this level of precision is suficient to serve as independent ground truth for evaluating markerless tracking methods, whose registration errors are substantially larger, particularly in rotation.

## 3.2 Clinical Data Registration

With the pipeline validated in the lab, we demonstrated its utility by evaluating two markerless registration strategies on clinical data from three surgical sequences, encompassing 11 vertebrae across 1,500 frames. To our knowledge, this represents the first evaluation of markerless spine registration against an independent subsurface gold standard reference in a clinical setting.

![](images/dc7a43ef2aa55e908b9e125820929c57b4d524761a406bae6448aabe897611bf.jpg)  
Fig. 2: (A) The pose of ovine CT meshes directly tracked via the Azure Kinect in blue, compared to those tracked via fluoroscopy and projected into the Kinect camera space in yellow; (B) Per vertebra patches predicted by the network in Sec. 3.2; (C, D) Intraoperative pose estimates via X-ray in yellow, and through per vertebra ICP in blue.

The first strategy follows [5], deforming the preoperative spine mesh via a kinematic chain model before rigidly registering it to segmented intraoperative points using ICP. This approach relies on semi-automated surface-based optimization of the kinematic chain parameters, with manual refinement, to align the deformed mesh with the intraoperative point cloud: a workflow dificult to scale to real-time intraoperative use. We therefore additionally trialed a fully automated approach inspired by [15], in which CT meshes of individual vertebrae are independently registered to per-vertebra segmentation patches. These patches were obtained via a two-stage pipeline: a PointNet++ [25] heatmap regression network, trained on manually annotated spinous process tip positions, first localizes tips within the point cloud (recall 0.95, F1 0.89, localization error 3.98 ± 1.99 mm), after which a PointNet segmentation network isolates points within 40 mm of each detection (IoU 0.71, F1 0.83). Per-vertebra segmentation labels were generated by computing mesh-to-point-cloud proximity for individual vertebral CT meshes from [5], following the same approach used therein for global spine masks. The results of both registration approaches, evaluated against our fluoroscopic pose estimates with a correspondence distance of 3 mm for both methods, are shown in Tab. 3.

The rigid method of [5] outperforms per-vertebra ICP on rotation $( 4 . 5 2 ^ { \circ } \pm$ $1 . 0 0 ^ { \circ } \ \mathrm { v s . } \ 8 . 3 6 ^ { \circ } \pm 3 . 6 9 ^ { \circ } )$ and exhibits substantially lower variance across all metrics, while mean translation errors are comparable with overlapping distributions. The per-vertebra approach, though fully automated, lacks the geometric constraints aforded by registering the full spine as a single structure, limiting its ability to resolve orientation. While both methods achieve mean translational errors below 5 mm, neither yet meets the ≤ 2 mm and $\leq 5 ^ { \circ }$ thresholds cited for spinal navigation [22], underscoring both the value of independent pose evaluation and the headroom remaining for emerging methods.

Table 3: Comparison of [5]’s rigid registration method across three sequences of clinical data, versus individual ICP. Translation (Trans.) errors are reported in mm and rotation (Rot.) errors in degrees. Each sequence is 500 frames. "Exposure" represents the exposed vertebra count.
<table><tr><td>Exposure</td><td>Per Vertebra ICP Trans. (mm) Rot. (o)</td><td></td><td>Rigid ICP Trans. (mm) Rot. (o)</td></tr><tr><td>5</td><td> $\overline { { 5 . 1 5 \pm 3 . 6 5 } }$ </td><td> $\overline { { 8 . 2 1 \pm 3 . 7 4 } }$ </td><td> $\overline { { 5 . 7 0 \pm 1 . 5 9 } }$   $\overline { { 4 . 6 1 \pm 1 . 5 5 } }$ </td></tr><tr><td>3</td><td> $3 . 9 8 \pm 1 . 8 1$ </td><td> $9 . 1 0 \pm 3 . 6 8$ </td><td> $4 . 6 6 \pm 0 . 8 2$   $3 . 5 3 \pm 0 . 7 8$ </td></tr><tr><td>3</td><td> $4 . 0 2 \pm 3 . 1 0$ </td><td> $7 . 8 6 \pm 3 . 6 3$ </td><td> $3 . 3 8 \pm 0 . 5 8$   $5 . 3 6 \pm 0 . 3 0$ </td></tr><tr><td>Overall</td><td>4.52 ± 3.00</td><td> $\mathbf { \overline { { 8 . 3 6 \pm 3 . 6 9 } } }$ </td><td>4.78 ± 1.10  $\overline { { { \bf { 4 . 5 2 \pm 1 . 0 0 } } } }$ </td></tr></table>

## 4 Discussion

The clinical registration comparison in Tab. 3 demonstrates the core contribution of this work: for the first time, competing markerless registration strategies can be evaluated on real surgical data against an independent subsurface reference. Prior clinical evaluation was limited to surrogate surface metrics [5]; our pipeline provides per-vertebra 6-DoF pose estimates from routine intraoperative fluoroscopy without additional hardware or workflow disruption.

Registration errors in Tab. 3 are measured against our fluoroscopic estimates, which carry an intrinsic uncertainty of 2.06 ± 0.29 mm and $3 . 3 5 \pm 0 . 9 3 ^ { \circ }$ (Tab. 2); these values thus represent upper bounds on true registration error. Alas, the rotational diference between the two methods (3.84<sup>◦</sup>) exceeds our reference uncertainty, confirming that the rigid method of [5] is meaningfully more accurate in rotation. Translational errors are not statistically diferent. Overall, the rigid approach is clearly superior, but its reliance on manual deformation tuning remains a barrier to clinical deployment, and thus an avenue for future work.

Key limitations remain. The study reports three clinical sequences and ovine ex vivo validation; further labeled cases, drawn from a larger ongoing clinical study, will broaden the evaluation. Accuracy is also constrained by the 2D–3D registration stage, which depends on image quality, angular separation, and the number of fluoroscopic views.

Clinically, classical surface registration is intrinsically limited: the posterior vertebral surface exposed during surgery is geometrically ambiguous, leaving ICP unable to robustly resolve pose, particularly rotation. This is unlikely to improve with initialization alone. With suficient fluoroscopic ground truth, a learned mapping from intraoperative point clouds to vertebral pose could instead leverage appearance, anatomical context, and priors beyond ICP’s scope.

## References

1. Balachandran, R., Fitzpatrick, J.M., Labadie, R.F.: Accuracy of image-guided surgical systems at the lateral skull base as clinically assessed using bone-anchored hearing aid posts as surgical targets. Otology & Neurotology 29(8), 1050–1055 (2008)

2. Bradski, G., Kaehler, A.: Opencv: Open source computer vision library. https://opencv.org/ (2000), accessed: 2026-02-08

3. Chan, Y., Banza, L., Martin Jr, C., Harrison, W.J.: Essential fracture and orthopaedic equipment lists in low resource settings: consensus derived by survey of experts in africa. BMJ open 8(9), e023473 (2018)

4. Cho, Y., Moseley, D.J., Siewerdsen, J.H., Jafray, D.A.: Accurate technique for complete geometric calibration of cone-beam computed tomography systems. Medical physics 32(4), 968–983 (2005)

5. Daly, C., Marconi, E., Riva, M., Ekanayake, J., Elson, D.S., Rodriguez y Baena, F.: Towards markerless intraoperative tracking of deformable spine tissue. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 627–637. Springer (2025)

6. Daly, M.J., Siewerdsen, J.H., Cho, Y.B., Jafray, D.A., Irish, J.C.: Geometric calibration of a mobile c-arm for intraoperative cone-beam ct. Medical physics 35(5), 2124–2136 (2008)

7. Fischler, M.A., Bolles, R.C.: Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM 24(6), 381–395 (1981)

8. Gao, Z., Zhang, X., Xu, Z., Jiang, C., Hu, W., Zhang, H., Hao, D.: Mazor x robotassisted upper and lower cervical pedicle screw fixation: a case report and literature review. BMC Geriatrics 24(1), 916 (2024)

9. Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F.J., Marín-Jiménez, M.J.: Automatic generation and detection of highly reliable fiducial markers under occlusion. Pattern Recognition 47(6), 2280–2292 (2014)

10. Gorges, S., Kerrien, E., Berger, M.O., Trousset, Y., Pescatore, J., Anxionnat, R., Picard, L.: Model of a vascular c-arm for 3d augmented fluoroscopy in interventional radiology. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 214–222. Springer (2005)

11. Hu, X., Nguyen, A., Baena, F.R.y.: Occlusion-robust visual markerless bone tracking for computer-assisted orthopedic surgery. Transactions on Instrumentation and Measurement 71, 1–11 (2022). https://doi.org/10.1109/TIM.2021.3134764

12. ImFusion: ImFusion Suite: Visual computing for medical imaging. https://imfusion.com/products-overview/imfusion-suite/ (2025), accessed: 2025-11-14

13. Jain, A.K., Mustafa, T., Zhou, Y., Burdette, C., Chirikjian, G.S., Fichtinger, G.: Ftrac—a robust fluoroscope tracking fiducial. Medical physics 32(10), 3185–3198 (2005)

14. Ji, Z.W., Shen, M.J., Sun, J.J., Wang, J.L., Deng, Y.K., Zhang, Y., Wu, X.X., Chen, K.W., Mao, H.Q.: Enhanced safety and eficacy of o-arm navigation over carm guidance in percutaneous kyphoplasty for patients with osteoporotic vertebral compression fractures and spinal deformity: A comparative study. Orthopaedic Surgery 17(1), 125–132 (2025)

15. Liebmann, F., von Atzigen, M., Stütz, D., Wolf, J., Zingg, L., Suter, D., Cavalcanti, N.A., Leoty, L., Esfandiari, H., Snedeker, J.G., et al.: Automatic registration

with continuous pose updates for marker-less surgical navigation in spine surgery. Medical Image Analysis 91, 103027 (2024)

16. Liebmann, F., Stütz, D., Suter, D., Jecklin, S., Snedeker, J.G., Farshad, M., Fürnstahl, P., Esfandiari, H.: Spinedepth: a multi-modal data collection approach for automatic labelling and intraoperative spinal shape reconstruction based on rgb-d data. Journal of Imaging 7(9), 164 (2021)

17. Liu, H., Baena, F.R.Y.: Automatic markerless registration and tracking of the bone for computer-assisted orthopaedic surgery. IEEE Access 8, 42010–42020 (2020)

18. Liu, W.P., Otake, Y., Azizian, M., Wagner, O.J., Sorger, J.M., Armand, M., Taylor, R.H.: 2d–3d radiograph to cone-beam computed tomography (cbct) registration for c-arm image-guided robotic surgery. International journal of computer assisted radiology and surgery 10(8), 1239–1252 (2015)

19. MacKechnie, M.C., Shearer, D.W., Verhofstad, M.H., Martin, C., Graham, S.M., Pesantez, R., Schuetz, M., Hüttl, T., Kojima, K., Bernstein, B.P., et al.: Establishing consensus on essential resources for musculoskeletal trauma care worldwide: a modified delphi study. JBJS 106(1), 47–55 (2024)

20. Markelj, P., Tomaževič, D., Likar, B., Pernuš, F.: A review of 3d/2d registration methods for image-guided interventions. Medical image analysis 16(3), 642–661 (2012)

21. Meng, D., Boyer, E., Pujades, S.: Vertebrae localization, segmentation and identification using a graph optimization and an anatomic consistency cycle. Computerized Medical Imaging and Graphics 107, 102235 (2023)

22. Miller, C.A., Ledonio, C.G., Hunt, M.A., Siddiq, F., Polly, D.W.: Reliability of the planned pedicle screw trajectory versus the actual pedicle screw trajectory using intra-operative 3d ct and image guidance. International Journal of Spine Surgery 10 (2016)

23. Navab, N., Heining, S.M., Traub, J.: Camera augmented mobile c-arm (camc): calibration, accuracy study, and clinical applications. IEEE Transactions on Medical Imaging 29(7), 1412–1423 (2010)

24. Northern Digital Inc.: Polaris Vicra Technical Specifications (2018), https://www.ndigital.com/products/polaris-vicra/, p/N 10001248 (Rev 001)

25. Qi, C.R., Yi, L., Su, H., Guibas, L.J.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems 30 (2017)

26. Stealthstation s8 surgical navigation system. https://europe.medtronic.com/xden/healthcare-professionals/products/neurological/surgical-navigationsystems/stealthstation/stealthstation-s8.html, accessed: 2026-02-26

27. Wilson Jr, J.P., Fontenot, L., Stewart, C., Kumbhare, D., Guthikonda, B., Hoang, S.: Image-guided navigation in spine surgery: from historical developments to future perspectives. Journal of Clinical Medicine 13(7), 2036 (2024)