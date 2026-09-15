# Integrating Multi-view Multi-light Surface Reconstruction into Cultural Heritage Workflows

Baptiste Brument<sup>1</sup> , Robin Bruneau<sup>2</sup> , Benjamin Coupry<sup>1</sup> , Vincent Demoulin<sup>3</sup>, Jean Mélou<sup>4</sup> , Antoine Laurent<sup>5</sup> , Fabien Castan<sup>3</sup> , Jean-Denis Durou<sup>1</sup> , and Lilian Calvet<sup>6</sup>

<sup>1</sup> IRIT, UMR 5505, CNRS, Toulouse, France

<sup>2</sup> University of Zurich, Zurich, Switzerland

<sup>3</sup> The Mill, France

<sup>4</sup> Fittingbox, Labège, France

<sup>5</sup> TRACES, UMR 5608, CNRS, Toulouse, France

<sup>6</sup> ROCS, Balgrist University Hospital, University of Zurich, Zurich, Switzerland

Abstract. Cultural heritage documentation increasingly relies on imagebased 3D surface reconstruction, with photogrammetry software making such workflows accessible to archaeologists, conservators, and heritage technicians. These tools have been successful for conventional multiview acquisition, but they do not routinely exploit richer multi-view, multi-light data, despite its potential for improving fine-scale surface reconstruction. This limitation is particularly relevant in heritage contexts, where controlled-light acquisition devices such as RTI domes are already used to capture illumination-varying image sets. The challenge is therefore to connect these existing acquisition practices with recent computer vision methods in a form that can be used within operational heritage workflows. In this work, we address this need by integrating state-of-the-art components from computer vision for multiview, multi-light surface reconstruction into Meshroom, an open-source photogrammetry framework. Rather than proposing a new reconstruction algorithm, our contribution is to assemble and expose existing advanced methods, namely a complete photometric stereo ecosystem (calibrated, self-calibrated and universal), automatic object masking, and multi-view normal-and-reflectance integration, within a usable heritageoriented workflow. The proposed system thus provides an intermediate software layer between computer vision research code and practical cultural heritage applications, making recent techniques easier to use and evaluate.

Keywords: Multi-view Photometric Stereo · Photometric Stereo · Cultural Heritage · Photogrammetry

## 1 Introduction

High-fidelity 3D replicas have become important tools across cultural heritage practice, and advances in sensors, acquisition, and reconstruction algorithms now produce increasingly detailed digital representations of heritage objects. This precision matters because, for many artifacts, the relevant information lies in the fine surface relief, namely engravings, tool marks, and micro-relief, as much as in the overall shape. Such objects are also among the hardest to digitise, since they are frequently poorly textured, as with lithic, bone or uniform ceramics.

![](images/7dcb9bc3ea72add3fba39572d7ec63633c38f18f1a16f2f49a359e075f31bc8f.jpg)  
Fig. 1: Our contribution: integrating a user-friendly multi-view, multi-light surface reconstruction pipeline into Meshroom [20] for cultural heritage practitioners. Top: reconstructing a lithic biface; the multi-view, multi-light inputs in the gallery (left), a per-view normal map from photometric stereo in the image viewer (centre), the final surface in the 3D viewer (right), and the processing graph (bottom; see Fig. 3). Bottom: the photometric stereo methods integrated as interchangeable Meshroom nodes [37,38, 40].

To record this surface from photographs, the heritage community relies on two complementary families. Photogrammetry recovers 3D geometry from many viewpoints through mature, widely used software [12, 19, 30], but it is most reliable on global shape, struggles with uniform and non-Lambertian surfaces [25], and bakes the capture-time illumination into its texture (Fig. 2). Reflectance transformation imaging (RTI), by contrast, photographs a fixed viewpoint under many light directions [17, 34, 46]: varying the light reveals fine relief that flat photographs miss, and is widely used to read faint engravings and tool marks [35, 54], yet it yields an image-space, single-viewpoint representation, not a 3D surface.

These two families capture complementary cues, multi-view geometry and illumination-varying appearance, yet they are mostly acquired and analysed separately, and combining them remains a recognised open challenge in the heritage community [51,52]. The computer vision community has long sought to reconcile them in a single representation: multi-view, multi-light surface reconstruction, known as multi-view photometric stereo (MVPS), recovers per-view normal and reflectance maps from the multi-light images of each viewpoint (photometric stereo) and fuses them across views into one 3D surface (multi-view normal integration). Its key practical appeal is to reuse the illumination-varying images the heritage community already captures for RTI, now from several viewpoints, building on familiar acquisition practice rather than a new modality. However, this reconstruction long proved dificult: early methods had to reconcile potentially conflicting depth and normal objectives and tended to erase the very detail they were meant to preserve [26,27]. Recent neural surface reconstruction removes this conflict through a single optimisation objective [5,6,9], while photometric stereo itself now handles non-Lambertian materials and recovers per-pixel normals without prior light calibration [11, 22, 23]. What remains is that these advances live as research prototypes and unpackaged code, out of reach of most heritage practitioners.

This lack of integration is itself the barrier to adoption, in a community that has nonetheless already embraced photogrammetric software. We address it by integrating state-of-the-art computer vision components for multi-view, multi-light surface reconstruction into Meshroom [20], an existing open-source photogrammetry software, so that classical and recent photometric stereo and multi-view normal integration methods become usable and evaluable by heritage practitioners. Moving these methods out of the laboratory means supporting many acquisition setups rather than a single one, as the constraints vary widely (access to the artefacts, availability and cost of a calibrated dome, indoor versus outdoor capture); we therefore expose a modular ecosystem of interchangeable nodes that adapts to each setup, from a calibrated dome to a hand-held flash in the field. Concretely, our contributions are:

– (C1) A complete photometric stereo ecosystem exposed as Meshroom nodes, covering calibrated photometric stereo with automatic calibration-sphere detection and light calibration, self-calibrated near-light photometric stereo [13], and universal neural methods that need no prior light calibration, namely UniMS-PS [22], SDM-UniPS [23] and LINO-UniPS [11];

(C2) Automatic object masking [10,39] and a multi-view normal integration node that optionally fuses the estimated reflectance, provided both as the reference RNb-NeuS2 [6] and as our permissively licensed OpenRNb [41] reimplementation, which together close the chain from multi-light images to a single textured, high-fidelity surface mesh;

– (C3) An open-source release of all the nodes and of ready-to-use multiview, multi-light pipelines, with sample code that promotes the integration of further photometric stereo and multi-view normal integration methods;   
– (C4) A qualitative validation on real archaeological objects.

## 2 Related Work

Photogrammetry for cultural heritage. Imagebased 3D reconstruction has become the dominant photographic technique in heritage documentation [56,57], largely because computer vision methods have been translated into robust, easy-to-use software that fits heritage workflows, from single artefacts to large campaigns such as archaeological excavations [14], monument documentation [68], and the post-fire digitisation of Notre-Dame de Paris [12]. Packages such as Metashape [30], RealityScan [19], 3DF Zephyr [1], Pix4D [59], Meshroom [20,45], and COLMAP [60,61] provide graphical interfaces for structure-from-motion and multiview stereo. At their core, these pipelines map posed images to a dense 3D surface, typically through multi-view stereo, although the exact reconstruction back-end of the proprietary packages is not publicly disclosed. They are most reliable on global shape: fine relief can be recovered, but usually only under strong assumptions on viewpoint coverage and surface appearance, and the highestfrequency detail tends to be smoothed out by the multi-view stereo and meshing. A further limita-

![](images/683afef969d6e52b32679d0b21c29324c16e7a267d3b3327d13bb2dfb1fb2c3f.jpg)  
Fig. 2: Texture from Meshroom’s standard photogrammetry pipeline bakes in the acquisition lighting: lit more at the front, this Socrates bust reconstructs darker at the back despite an identical material.

tion concerns texture: because each surface point samples colour directly from the input photographs, capture illumination, including shadows and directional shading, is baked into the texture map (Fig. 2). The resulting model therefore carries the acquisition lighting, which is undesirable when texture is meant to document intrinsic object appearance.

Reflectance transformation imaging. Multi-light acquisition is already familiar in the heritage community through reflectance transformation imaging (RTI) and polynomial texture maps [17, 34, 46], surveyed in [35, 54]. In RTI, an object is photographed from a single fixed viewpoint while lit successively from many directions, typically with an LED dome (Fig. 4) or a hand-held flash [49, 63]. The result is a stack of images with a fitted per-pixel reflectance model, from which the surface can be interactively relit [58] and faint features such as inscriptions or tool marks enhanced. In its standard form, however, it remains a single-viewpoint, image-space technique: it neither produces 3D information nor explicitly disentangles the surface normals from the reflectance, the very quantities a multi-view fusion would need to build a single 3D representation.

Combining multi-view and multi-light in cultural heritage. Photogrammetry and RTI thus capture complementary cues, yet are mostly acquired and analysed separately. Early surveys already captured both photogrammetry and RTI of the same object [43], and later eforts registered RTI onto a photogrammetric 3D model for joint viewing [51, 52]; this shows both the interest of the combination and its dificulty, such multimodal integration requiring careful registration and remaining a recognised open challenge rather than a routine operation. Other lines combine the two cues only post-hoc: one blends high-frequency dome photometric-stereo normals onto a low-frequency photogrammetric or laser height field [33], while another bakes RTI normal maps onto a coarse photogrammetric mesh, leaving the relief in a texture rather than the geometry [44]; both keep the modalities in separate pipelines, registered and merged afterwards. The approach we follow sidesteps this registration problem: instead of fusing RTI with photogrammetry as two separate modalities, we feed the multi-light images directly into photometric stereo and integrate the resulting per-view normals across viewpoints into a single 3D surface.

Photometric stereo. Photometric stereo recovers, at every pixel, the surface normal together with the reflectance from images of a fixed viewpoint under varying illumination [66]. Classical formulations assume known, calibrated lighting; over the last few years the field has progressed rapidly on two fronts that matter for heritage. First, the acquisition setting has been relaxed from the fully controlled laboratory dome towards semi-controlled, in-the-field capture: selfcalibrated near-light methods recover the lighting on the fly from a coarse geometric proxy [13], while universal deep methods operate under entirely unknown, spatially-varying illumination [11, 22, 23]. Second, these methods increasingly handle complex materials, estimating not only a difuse reflectance but richer, physically-based reflectance (PBR) [7,23]. By estimating a normal at every pixel from the illumination variation, photometric stereo reaches a higher level of surface detail than single-shot methods, including on non-Lambertian materials, sparser viewpoints, and weakly textured surfaces where photogrammetry is inherently ambiguous [25]. These advances make photometric stereo applicable well beyond the controlled dome, where it has begun to be benchmarked on dedicated data [16].

Multi-view photometric stereo. In computer vision, MVPS was classically cast as a joint optimisation problem, deforming a mesh so that it both renders consistently across views and agrees with the per-view photometric-stereo normals [53]. Later methods estimated reflectance, normals, and lighting together for more general materials [29]. These formulations had to balance conflicting objectives, such as a depth term against a normal term, and tended to smooth away the high-frequency relief encoded by the normals [26, 27]. Neural volume rendering changed this by supervising a neural signed distance function through a volumetric rendering loss [64,65]; neural MVPS methods then gained markedly in detail by encouraging this field to agree with photometric-stereo normals [31,67], without re-parametrising the photometric quantities, since normals-only integration already recovers most of the relief [9]. RNb-NeuS [5, 6] keeps a single objective while additionally admitting reflectance as an optional cue, by encoding the heterogeneous per-view normals and reflectance into a common, radiance-like quantity that directly supervises the signed distance function.

![](images/4971135afee3856bec8cf816a285e4467410a5e76c9d63a587f9bf15cbcbbe22.jpg)  
Fig. 3: End-to-end data flow of the proposed pipeline. SAM 3 segments the multiview, multi-light images, restricting the rest of the pipeline to the object of interest. Structure-from-motion recovers the camera parameters, optionally guided by the masks (e.g. discard turntable-induced multiple motions), and an optional coarse geometry proxy that may be subsequently used by self-calibrated photometric stereo methods. Photometric stereo (calibrated, self-calibrated, or universal) turns the masked images into per-view normal maps, and optionally reflectance maps. Multi-view normal integration then fuses the per-view normals and reflectance with the camera parameters into a single mesh textured with the recovered reflectance.

Reflectance as shading-free texture. Because photometric stereo separates reflectance from shading, the colour map applied to the reconstructed mesh is the intrinsic reflectance of the surface, free of the shadows and shading baked in by single-shot photogrammetric texturing [6]. Recovering such a relightable, intrinsic appearance jointly with geometry is the shared goal of inverse-rendering methods, whether from single-illumination [48, 62, 69] or multi-light photometric [4] capture; here it comes for free as a by-product of the chain. For heritage documentation, this yields an illumination-neutral texture that does not lock the object into the lighting conditions of the acquisition [28].

## 3 Proposed Pipeline in Meshroom

We now present the proposed pipeline: a multi-view, multi-light surface reconstruction chain assembled as nodes inside the open-source photogrammetry software Meshroom [20], turning illumination-varying image sets into a single textured, high-fidelity mesh. Figure 3 gives an overview of the data flow, from acquisition to the final mesh, and the rest of this section follows it stage by stage.

## 3.1 Multi-view Multi-light Acquisition

Multi-view, multi-light acquisition couples several viewpoints with several illuminations per viewpoint, the latter already familiar to the heritage community through RTI domes (Fig. 4) or a hand-held flash (Sec. 2). The pipeline accepts data organised per viewpoint and per light, captured with such a dome, an open-hardware camera-and-turntable setup, or a hand-held flash in the field, and places no constraint on image resolution (45.7 MP in our experiments).

![](images/a3b7817dcb235eecec8a57aeee3a04a38355e233b03dac9c583602c03d70f8c5.jpg)  
Fig. 4: Multi-light acquisition with an RTI dome (here a Mercurio Imaging dome [36]). Left: the dome seen from the front, its 105 LEDs distributed over a hemisphere around a central camera; firing the LEDs one at a time yields one image per light direction. Right: such a dome in use on the Panneau des Chevaux in the Chauvet cave.

## 3.2 A Nodal Pipeline in Meshroom

We build on the nodal architecture of Meshroom [20]: a reconstruction pipeline is a directed acyclic graph of nodes, each wrapping a tool, with an intermediate cache, automatic invalidation of downstream nodes, and standard file formats exchanged between nodes. Meshroom (version 2025.1 and above) already reads the formats heritage acquisition produces, from camera RAW (e.g. Canon CR2, Nikon NEF, Sony ARW) to high-dynamic-range OpenEXR through the Open-ImageIO library, and can fuse bracketed exposures into HDR [15]; it scales to the very-high-resolution datasets typical of photogrammetry and can dispatch its stages, including the per-pose photometric stereo computations, across a render farm through a pluggable submitter mechanism (e.g. Tractor [55]). We chose Meshroom for this extensibility and its mature, widely used interface: it lets us graft a complete photometric stereo and multi-view normal integration chain onto the existing structure-from-motion stage, making the methods accessible to end users while keeping the pipeline adaptable to each acquisition setup. Every intermediate result, masks, camera parameters, per-view normal and reflectance maps, and the final mesh, can be visualised and reused.

Our contributions are the photometric stereo nodes, the multi-view normal integration node and the automatic masking, but also their orchestration with the existing components: feeding the structure-from-motion coarse proxy to selfcalibrated photometric stereo, inserting light-calibration nodes where a method requires them, and deriving the metric scaling and centering of the neural volume from the object masks or the sparse point cloud. We release all these nodes, together with several ready-to-use MVPS pipelines, as open-source Meshroom plugins implemented in Python: practitioners with no computer vision background can run the prebuilt pipelines as is, while more technical users can recompose the nodes, or implement new ones in Python, for their own acquisitions.

Automatic object masking. Heritage objects are usually imaged on a turntable, in a light tent, or on a stand for objects that need several faces to be captured, and a binary mask separating the object from the background serves two distinct purposes. First, on a turntable the background and the object move diferently from one shot to the next, a standard failure case for structure-from-motion; masking the background removes these conflicting motions. Second, many neural surface reconstruction methods, including the one we rely on, take per-view object masks as input to bound the reconstructed volume [6,64]. Producing such masks by hand, on every view, is tedious. We therefore integrate automatic semantic segmentation as a Meshroom node, ImageSegmentationSam3 [39], built on the Segment Anything model with concepts (SAM 3) [10]. From a lightweight prompt naming the object, optionally refined with positive and negative bounding boxes, it produces a per-view mask that then propagates through the whole chain (structure-from-motion, multi-view stereo, photometric stereo and integration), making object-centric reconstruction efectively automatic.

Structure-from-motion and multi-view stereo. These stages are already integrated into Meshroom through AliceVision [20], which has evolved substantially beyond the initial OpenMVG [45] and CMPMVS [24] implementations. They detect and match image features across the input views, recover the camera parameters together with a sparse point cloud, and can optionally build a dense surface proxy. Fiducial markers (CCTag [8] or AprilTag [50]) can also be placed in the scene, for two purposes. They improve robustness on objects that are hard for structure-from-motion: for a weakly textured piece on a turntable, markers on the turntable anchor the camera poses through reliable correspondences rather than uncertain matches on the object. Given their known 3D coordinates, Meshroom also uses them to set the metric scale, via its SfMTransform node. The recovered camera parameters anchor the subsequent multi-view normal integration; the optional coarse surface proxy is used only by the self-calibrated near-light photometric stereo node [13] for its light estimation.

The photometric stereo ecosystem. We implement the photometric stereo stage as interchangeable Meshroom nodes, covering calibrated lighting recovered from calibration spheres, self-calibrated near-light illumination, and unknown lighting handled by universal neural methods. Exposing these alternatives behind a common interface lets practitioners select and compare the method that best matches the acquisition at hand. All nodes output per-view normal maps, with reflectance maps provided when supported by the selected method.

Calibrated photometric stereo, the most classical and elementary setting, assumes the light directions and intensities are known. In our pipeline, a calibration sphere of known radius is placed in the field of view (Fig. 5) and detected automatically (SphereDetection [18]); the LightingCalibration node locates the specular highlight on the sphere and infers each light direction (directional-light assumption in the current implementation), and the PhotometricStereo node then applies classical calibrated photometric stereo [66] to recover the per-pixel normals and reflectance.

![](images/037d73e5b788243aacaa6bd998fc969300d86936540eee463db0e40dd11b5ed0.jpg)  
(a) Automatic sphere detection

![](images/e72603a4f7ee1e8fc7ae17564a95d18a76d803e1fcf44277548ac5efce74694e.jpg)  
(b) Calibrated photometric-stereo sub-pipeline  
Fig. 5: Calibrated photometric stereo in Meshroom. (a) A calibration sphere placed in the scene, here beside the Cypriot warrior head, is detected automatically (green circle) by the SphereDetection node. (b) The calibrated sub-pipeline, SphereDetection → LightingCalibration → PhotometricStereo, recovers the perimage light directions from the sphere and then the per-pixel normals and reflectance.

Self-calibrated near-light photometric stereo [13] removes this calibration step for nearby light sources: it recovers the position, intensity and anisotropy parameter of each light on the fly from a geometric proxy of the scene, either the sparse point cloud or, preferably, the coarse surface mesh of the multi-view stereo node [24], whose denser geometry yields more accurate surface normals for the light estimation. This fits in-the-field acquisition, for example with a hand-held flash, where nothing about the lighting is measured. As the state of the art in its category and based on an explicit physical model rather than learned from data, it introduces no learned-dataset bias and scales well, though it is best suited to near-Lambertian surfaces.

Unlike the calibrated and self-calibrated near-light approaches, universal photometric stereo requires neither measured lighting nor a coarse geometric proxy: it relies on learned neural models to recover surface normals, and reflectance when supported, under unknown, spatially varying illumination and complex, non-Lambertian materials. We integrate three universal photometricstereo methods that together represent the state of the art with complementary trade-ofs, each wrapped as a Meshroom node: UniMS-PS [22, 40] handles very-high-resolution inputs at multiple scales, at a higher computational cost; SDM-UniPS [23, 38] additionally recovers a physically-based reflectance; and LINO-UniPS [11, 37], the most recent, explicitly decouples illumination to preserve high-frequency detail, running in minutes where UniMS-PS takes hours on the same GPU.

Taken together, these nodes span the full range of lighting regimes, from calibrated to fully uncalibrated, and all feed the same multi-view normal integration. Their value lies in this interoperability and in making research-prototype methods available, and directly comparable, to practitioners. The calibrated node is kept for a reason: when the acquisition can be controlled, the more the lighting is constrained and known, the better-posed the reconstruction, which matters for metrology-oriented documentation.

Multi-view normal-and-reflectance integration. The final node fuses the per-view normal and reflectance maps into a single surface. Following recent neural surface reconstruction, it represents the scene as a signed distance function (SDF) whose zero level set is the surface, and optimises it so that the SDF gradient agrees, in each view, with the per-view normal maps from photometric stereo [6, 9]; the reflectance is folded into the same objective by re-parametrising normals and reflectance as a simulated radiance field. The mesh is then extracted by marching cubes [32]. We integrate RNb-NeuS2 [6, 42], which adapts the instant neural surface reconstruction of NeuS2 [65] to normal-and-reflectance inputs: its single objective includes the normals-only integration of SuperNormal [9] as a special case (see [6], Sec. 5.3), runnable as an option, while additionally integrating the reflectance, for instance from SDM-UniPS, that a normals-only integrator would discard. On the implementation side, its multiresolution hash encoding [47] and CUDA backend cut the per-scene optimisation from about fifteen hours in the original RNb-NeuS [5] to roughly five minutes. The result is a single high-fidelity mesh textured with the reflectance, free of baked-in shading. RNb-NeuS2 carries an NVIDIA license restricting it to academic use; for commercial settings, we contribute OpenRNb [41] (MPL 2.0), a from-scratch reimplementation of the integration backend exposed as a drop-in alternative.

## 4 Results on Real Archaeological Objects

We evaluate the pipeline qualitatively on four real archaeological objects, chosen so that each exercises a diferent instantiation of the pipeline, namely a diferent photometric stereo method.<sup>7</sup> In these experiments, all multi-view photometric stereo results are produced with the RNb-NeuS2 integration node [6] (Sec. 3). All datasets were captured with a full-frame 45.7 MP camera, and the full chain, raw development, structure-from-motion, photometric stereo and integration, runs in a few hours per object on a single GPU, comparable to a conventional photogrammetric reconstruction. Because fine engravings and tool marks are often central to documentation, we focus on the recovered surface relief, rendering the reconstructed geometry under controlled shading rather than its colour texture. A structured-light scan [2, 3], when available, is included as an additional comparison modality and never as a target: it is one digitisation technique among others, relying on dedicated hardware rather than the accessible, image-based setting we address.

Reference  
![](images/70adbb86e46e6602acb96e8a393f3de1858199b127a781e6ee383ba4ab5f81ed.jpg)

![](images/01c535d2187cdfebf3eef527d53b9098475238d5bbea51c6a9de98fbb5ac819d.jpg)

![](images/50bf44fc79df1eb9d6c755f621892794656b0146535d8d90d5df906139449e6c.jpg)  
COLMAP

![](images/39ec26116c7b2dd3dd5a9302aa9eac38e1a89573dcfdf0023ded240c3a55125b.jpg)  
Meshroom

![](images/30c148707803f65d9cfa33817fcadbc23467bb8022edf63ec19bff24d560af47.jpg)  
Proposed  
Fig. 6: Head of a Cypriot warrior (terracotta, 6th century BC), under a matched viewpoint. From left to right: a reference photograph, three photogrammetric reconstructions from 121 views, namely Metashape [30], COLMAP [60, 61] and the Meshroom photogrammetry pipeline [20], and the proposed pipeline with self-calibrated near-light PS [13], integrated by RNb-NeuS2 [6] (13 views under 12 illuminations each, namely 156 images). The black insets magnify the eye (top) and the braided beard (bottom). The reference is consistent with the recovered relief, supporting that it is genuine rather than hallucinated; the proposed pipeline restores sharper relief, notably the separated strands of the beard, at a comparable image budget.

## 4.1 Head of a Cypriot Warrior

The first object is a terracotta head whose fine, high-frequency relief, concentrated in the hair and beard, is the kind of structure known to be hard to reconstruct faithfully. We compare the proposed pipeline against photogrammetry at a comparable image budget, rendering the geometry alone (Fig. 6), using self-calibrated near-light photometric stereo [13] for the per-view normals, with a 35 mm lens and a hand-held flash for the multi-light captures. For this almost identical number of images, the proposed pipeline recovers markedly finer relief: as the eye and beard insets show, it separates the strands of the braided beard that dense matching tends to fuse and smooth. The reference photograph, shown alongside, confirms that this relief is genuinely present on the object and not a neural hallucination.

## 4.2 Engraved Bone of Gourdan

The second object is an engraved bone whose surface bears shallow incised engravings, the kind of faint, high-frequency relief that a flat photograph or a smoothed mesh would lose. It is acquired from 23 viewpoints under 35 illuminations each, with a 50 mm lens and an RTI dome [36], and reconstructed without any photogrammetric prior: the per-view normals are estimated by the universal method UniMS-PS [22] and integrated into a single mesh by RNb-NeuS2 [6]. The recovered normal map already makes the engravings legible, and the integrated surface preserves them in 3D (Fig. 7), showing that the universal photometric stereo nodes make the pipeline able to digitise surfaces without any light calibration. By contrast, photogrammetric reconstructions from the 23 views alone (RealityScan [19], Metashape [30] or COLMAP [60, 61]) leave the engravings barely visible, confirming that single-light multi-view stereo does not resolve this faint relief.

![](images/a0b5993bd3915fb416c445f45b4c63b22540797cd4ad1fe9dcfd5e783845674a.jpg)

Fig. 7: Engraved bone from the Gourdan cave, rendered under grazing light: a reference photograph, the UniMS-PS [22] per-view normals, three photogrammetric reconstructions (RealityScan [19], Metashape [30] and COLMAP [60, 61]), and the surface integrated from these normals by RNb-NeuS2 [6]. The engravings are barely visible with photogrammetry but recovered by the proposed pipeline.  
![](images/ad36a8602ef48bb6439030dfab480ba13ea0e45125707e211025e490f1ad7c7e.jpg)  
Fig. 8: Lithic biface, from left to right: a reference photograph, RealityScan [19], the proposed pipeline, and an Artec Micro 2 scan [2]. For each method, a diagonal splits the relit texture (upper left) from the geometry alone (lower right); the circular inset magnifies the fine flake-scar relief of the boxed region.

## 4.3 Lithic Biface

The third object is a flint biface whose surface is covered by shallow, intersecting flake scars: a low-amplitude, weakly textured relief that is hard to recover. It also appears in Fig. 1, reconstructed inside the Meshroom interface. We compare the reconstructions in both geometry and textured appearance, relit under the illumination of a reference photograph (Fig. 8): each method panel is split along its diagonal, with the relit texture in the upper-left and the shading-only geometry in the lower-right. The multi-light set comprises 12 viewpoints under 12 illuminations each (50 mm lens, hand-held flash), against 162 photogrammetric views. The proposed pipeline, self-calibrated near-light photometric stereo [13] integrated by RNb-NeuS2 [6], recovers the sharp ridges of the shallow flake scars and reproduces the textured appearance of the real object, at a level of detail very close to the structured-light scan (Artec Micro 2) [2].

## 4.4 Woman with a Calathos

Finally, we turn to a ceramic head whose smooth, fine-grained surface carries a low-amplitude micro-relief that is easily lost. We reconstruct it and show it next to a reference photograph (Fig. 9): a photogrammetry pipeline (RealityScan [19]), a structured-light scan (Artec Spider, with 3D point accuracy down to 50 µm) [3], and the proposed pipeline. Photogrammetry uses 48 views, and the proposed pipeline uses 8 viewpoints, each imaged under 35 illuminations with a 50 mm lens and an RTI dome [36], self-calibrated near-light photometric stereo [13] providing the per-view normals and reflectance. Rendered under identical shading (Fig. 9, top), the proposed pipeline brings out the fine granularity of the ceramic surface that the other reconstructions, including the structuredlight scan, do not render at this scale. To check that this relief is genuine and not an artefact, we relight each surface under a single grazing light matched to the reference photograph (Fig. 9, bottom): the micro-relief recovered by the proposed pipeline closely tracks the shadows cast on the real object, whereas the photogrammetric and scanned surfaces remain comparatively smooth.

## 5 Discussion and Limitations

Limitations. Our integration still has limitations. First, only a subset of the relevant methods is wrapped so far: further photometric stereo and integration methods, and emerging approaches that reconstruct the surface directly from the multi-light images without an explicit photometric-stereo-and-fusion split, are not yet available as nodes. Second, several stages, the universal photometric stereo networks and parts of the multi-view stereo back-end, require an NVIDIA/CUDA GPU, which remains a deployment constraint for some heritage users. Third, the multi-view integration node comes in two implementations that are not yet on par. RNb-NeuS2 [6] is the reference one but, building on NeuS2, it inherits a non-commercial NVIDIA research license; OpenRNb [41] is a fromscratch, permissively licensed (MPL-2.0) reimplementation built on the open instant-nsr-pl framework [21], and is therefore usable in commercial settings. On the five DiLiGenT-MV objects the two recover essentially the same submillimetre surfaces, yet OpenRNb’s mean Chamfer distance is currently higher (0.24 mm against 0.18 mm on UniMS-PS normals, and 0.30 mm against 0.22 mm on SDM-UniPS). This is promising for a fully open, commercially usable alternative, but closing this gap needs further investigation. Finally, our evaluation on real archaeological objects is qualitative. Unlike the controlled DiLiGenT-MV benchmark reported above, these unique heritage objects have no metric ground truth, and the structured-light scans shown alongside are an alternative digitisation modality rather than a reference against which to compute an error. Quantifying reconstruction errors on such objects is one of the natural next steps to consolidate these findings.

Reference  
RealityScan  
Proposed  
Artec Spider  
![](images/ff75cf02afd121a7527adb8a083f9c96875e24a14e2d8c912207cf4308e6ca81.jpg)  
Fig. 9: Woman with a calathos (ceramic, 4th century BC), with columns in correspondence. Top: shading renders of the reconstructed geometry, next to a reference photograph. Bottom: the same surfaces relit under a single grazing light matched to the reference, so that fine relief is revealed by the cast shadows; the black box magnifies the right corner of the mouth (inset, top left of each panel). Columns: reference, RealityScan [19], the proposed pipeline (PS [13], 8 views under 35 illuminations each) and a structured-light scan [3]; photogrammetry uses 48 views. Among the shown modalities, the proposed pipeline best brings out the fine granularity of the ceramic micro-relief, including next to the structured-light scan at this resolution, and the grazing relighting matches the relief seen on the reference.

Outlook. Several directions follow naturally. The most immediate is to broaden the ecosystem, wrapping further photometric stereo and multi-view normal integration methods behind the same interface so that practitioners can compare them on their own data. A more ambitious direction is to propagate a per-pixel reliability of the photometric estimates through the pipeline and expose them as a confidence map in the mesh texture, so that downstream analysis can flag regions to interpret with caution. Because every intermediate output is exposed in open formats, the pipeline is also a natural basis for an open evaluation platform and for shared, frequency-resolved quality metrics. Finally, since the system is embedded in Meshroom rather than introduced as a separate application, it inherits an interface and execution model already familiar from photogrammetric practice. A complementary user study with conservators and archaeologists would help refine presets, documentation, and acquisition guidelines for diferent heritage settings.

## 6 Conclusion

We have presented an end-to-end multi-view, multi-light pipeline assembled inside Meshroom, a widely used open-source photogrammetry framework. Our contribution is the integration that turns a high-fidelity but lab-bound approach into an operational tool: a complete photometric-stereo ecosystem, calibrated, self-calibrated near-light and universal, automatic object masking, and a multiview normal-and-reflectance integration node. On real archaeological objects, the pipeline recovers fine surface relief and produces a shading-free texture, while reusing the interface and open formats that heritage practitioners already use for photogrammetry. The system lowers the barrier to detailed 3D reconstruction within a tool already adopted in heritage. We hope this encourages wider use of multi-view photometric stereo, and its extension to a broader range of surface materials.

## Acknowledgements

Baptiste Brument was funded by the OPEN-DOPAMIn project (CNRS Innovation). This work was granted access to the Juliet computing platform, supported by a French government grant managed by the Agence Nationale de la Recherche under the “Investissements d’avenir” program (reference “ANR-21-ESRE-0051”). It has been supported by the University of Zurich and University Hospital Balgrist.

## References

1. 3Dflow: 3DF Zephyr. https://www.3dflow.net, accessed 2026

2. Artec 3D: Artec Micro II: desktop 3D scanner for small objects. https://www. artec3d.com/portable-3d-scanners/artec-micro, processed with Artec Studio 19. Accessed 2026

3. Artec 3D: Artec Spider: high-resolution structured-light 3D scanner. https:// www.artec3d.com/portable-3d-scanners/artec-spider, hand-held scanner, accuracy 50 microns. Accessed 2026

4. Bao, J., Chen, G., Cui, S.: PIR: Photometric Inverse Rendering with Shading Cues Modeling and Surface Reflectance Regularization. In: 3DV (2025)

5. Brument, B., Bruneau, R., Quéau, Y., Mélou, J., Lauze, F.B., Durou, J.D., Calvet, L.: RNb-NeuS: Reflectance and Normal-Based Multi-View 3D Reconstruction. In: CVPR (2024)

6. Bruneau, R., Brument, B., Quéau, Y., Mélou, J., Lauze, F.B., Durou, J.D., Calvet, L.: Multi-View Surface Reconstruction Using Normal and Reflectance Cues. IJCV 134(2) (2026)

7. Burley, B.: Physically-Based Shading at Disney. In: ACM SIGGRAPH (2012)

8. Calvet, L., Gurdjos, P., Griwodz, C., Gasparini, S.: Detection and Accurate Localization of Circular Fiducials under Highly Challenging Conditions. In: CVPR (2016)

9. Cao, X., Taketomi, T.: SuperNormal: Neural Surface Reconstruction via Multi-View Normal Integration. In: CVPR (2024)

10. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., Rädle, R., Afouras, T., Mavroudi, E., Xu, K., Wu, T.H., Zhou, Y., Momeni, L., Hazra, R., Ding, S., Vaze, S., Porcher, F., Li, F., Li, S., Kamath, A., Cheng, H.K., Dollár, P., Ravi, N., Saenko, K., Zhang, P., Feichtenhofer, C.: SAM 3: Segment Anything with Concepts (2025), https: //arxiv.org/abs/2511.16719

11. Chen, H., Li, H., Ye, C., Chen, Z., Li, B., Xu, S., Guo, X., Liu, X., Wang, Y., Zhang, B., Ikehata, S., Shi, B., Rao, A., Zhao, H.: Light of Normals: Unified Feature Representation for Universal Photometric Stereo. In: ICLR (2026), arXiv:2506.18882

12. Comte, F., Pamart, A., Réby, K., De Luca, L.: Strategies and Experiments for Massive 3D Digitalization of the Remains After the Notre Dame de Paris’ Fire. In: 3D Virtual Reconstruction and Visualization of Complex Architectures. ISPRS, vol. XLVIII-2/W4 (2024)

13. Coupry, B., Lauze, F., Laurent, A., Mélou, J., Quéau, Y., Durou, J.D.: Self-Calibrated Near-Light Photometric Stereo Using a Geometric Proxy. In: SSVM. LNCS, vol. 15668 (2025)

14. De Reu, J., De Smedt, P., Herremans, D., Van Meirvenne, M., Laloo, P., De Clercq, W.: On Introducing an Image-Based 3D Reconstruction Method in Archaeological Excavation Practice. J. Archaeol. Sci. 41, 251–262 (2014)

15. Debevec, P.E., Malik, J.: Recovering High Dynamic Range Radiance Maps from Photographs. In: ACM SIGGRAPH (1997)

16. Dulecha, T.G., Pintus, R., Gobbetti, E., Giachetti, A.: SynthPS: a Benchmark for Evaluation of Photometric Stereo Algorithms for Cultural Heritage Applications. In: GCH. pp. 13–22 (2020)

17. Earl, G., Martinez, K., Malzbender, T.: Archaeological Applications of Polynomial Texture Mapping: Analysis, Conservation and Representation. J. Archaeol. Sci. 37(8), 2040–2050 (2010)

18. Fainsin, L., Mélou, J., Calvet, L., Carlier, A., Durou, J.D.: Neural Detection of Spheres in Images for Lighting Calibration. In: QCAV. Proc. SPIE, vol. 12749 (2023)

19. Games, E.: RealityScan (formerly RealityCapture). https://www.realityscan. com, accessed 2026

20. Griwodz, C., Gasparini, S., Calvet, L., Gurdjos, P., Castan, F., Maujean, B., De Lillo, G., Lanthony, Y.: AliceVision Meshroom: An Open-Source 3D Reconstruction Pipeline. In: ACM MMSys (2021)

21. Guo, Y.C.: Instant Neural Surface Reconstruction. https : / / github . com / bennyguo/instant-nsr-pl, accessed 2026

22. Hardy, C., Quéau, Y., Tschumperlé, D.: UniMS-PS: A Multi-Scale Encoder-Decoder Transformer for Universal Photometric Stereo. CVIU 248, 104093 (2024)

23. Ikehata, S.: Scalable, Detailed and Mask-Free Universal Photometric Stereo. In: CVPR (2023)

24. Jancosek, M., Pajdla, T.: Multi-View Reconstruction Preserving Weakly-Supported Surfaces. In: CVPR (2011)

25. Karami, A., Menna, F., Remondino, F.: Investigating 3D Reconstruction of Non-Collaborative Surfaces Through Photogrammetry and Photometric Stereo. In: Imaging Today, Foreseeing Tomorrow. ISPRS, vol. XLIII-B2 (2021)

26. Kaya, B., Kumar, S., Oliveira, C., Ferrari, V., Van Gool, L.: Uncertainty-aware deep multi-view photometric stereo. In: CVPR (2022)

27. Kaya, B., Kumar, S., Oliveira, C., Ferrari, V., Van Gool, L.: Multi-View Photometric Stereo Revisited. In: WACV (2023)

28. Laurent, A., Coupry, B., Brument, B., Mélou, J., Quéau, Y., Fritz, C., Durou, J.D.: Combining Geometric and Photometric 3D Reconstruction Techniques for Cultural Heritage. JCH 73, 43–51 (2025)

29. Li, M., Zhou, Z., Wu, Z., Shi, B., Diao, C., Tan, P.: Multi-View Photometric Stereo: A Robust Solution and Benchmark Dataset for Spatially Varying Isotropic Materials. IEEE TIP 29, 4159–4173 (2020)

30. LLC, A.: Agisoft Metashape. https://www.agisoft.com, accessed 2026

31. Logothetis, F., Budvytis, I., Cipolla, R.: NPL-MVPS: Neural Point-Light Multi-View Photometric Stereo. In: WACV (2025)

32. Lorensen, W.E., Cline, H.E.: Marching Cubes: A High Resolution 3D Surface Construction Algorithm. In: ACM SIGGRAPH (1987)

33. MacDonald, L., Toschi, I., Nocerino, E., Hess, M., Remondino, F., Robson, S.: Accuracy of 3D Reconstruction in an Illumination Dome. In: XXIII ISPRS Congress. ISPRS, vol. XLI-B5 (2016)

34. Malzbender, T., Gelb, D., Wolters, H.: Polynomial Texture Maps. In: ACM SIG-GRAPH (2001)

35. Mansouri, A.: Reflectance Transformation Imaging (RTI) for Cultural Heritage: Principles, Data Processing, and Applications. In: Cultural Heritage at a Crossroads: Sustainable Development and Digital Transformation (2026)

36. Mercurio Imaging: Mercurio Imaging RTI Domes. https://mercurioimaging. com/offres/, accessed 2026

37. Meshroom contributors: mrLINOUniPS: LINO-UniPS Photometric Stereo Node for Meshroom. https://github.com/meshroomHub/mrLINOUniPS, accessed 2026

38. Meshroom contributors: mrSDMUniPS: SDM-UniPS Photometric Stereo Node for Meshroom. https://github.com/meshroomHub/mrSDMUniPS, accessed 2026

39. Meshroom contributors: mrSegmentation: Segmentation Nodes for Meshroom. https://github.com/meshroomHub/mrSegmentation, accessed 2026

40. Meshroom contributors: mrUniMSPS: UniMS-PS Photometric Stereo Node for Meshroom. https://github.com/meshroomHub/mrUniMSPS, accessed 2026

41. Meshroom contributors: OpenRNb: An MPL-2.0 Implementation of Reflectanceand-Normal-Based Multi-View Surface Reconstruction. https://github.com/ meshroomHub/mrOpenRNb, accessed 2026

42. Meshroom contributors: mrRNbNeuS2: RNb-NeuS2 Multi-View Integration Node for Meshroom. https://github.com/RobinBruneau/RNb- NeuS2/tree/main/ meshroom (2026)

43. Miles, J., Pitts, M., Pagi, H., Earl, G.: Photogrammetry and RTI Survey of Hoa Hakananai’a Easter Island Statue. In: Across Space and Time, pp. 144–155. Amsterdam University Press (2015)

44. Morita, M.M., Loaiza Carvajal, D.A., González Bagur, I.L., Bilmes, G.M.: A Combined Approach of SfM-MVS Photogrammetry and Reflectance Transformation Imaging to Enhance 3D Reconstructions. JCH 68, 38–46 (2024)

45. Moulon, P., Monasse, P., Perrot, R., Marlet, R.: OpenMVG: Open Multiple View Geometry. In: Int. Worksh. Reproducible Research Pattern Recog. (2016)

46. Mudge, M., Malzbender, T., Schroer, C., Lum, M.: New Reflection Transformation Imaging Methods for Rock Art and Multiple-Viewpoint Display. In: VAST (2006)

47. Müller, T., Evans, A., Schied, C., Keller, A.: Instant Neural Graphics Primitives with a Multiresolution Hash Encoding. ACM TOG 41(4), 1–15 (2022)

48. Munkberg, J., Hasselgren, J., Shen, T., Gao, J., Chen, W., Evans, A., Müller, T., Fidler, S.: Extracting Triangular 3D Models, Materials, and Lighting From Images. In: CVPR (2022)

49. Nam, G., Lee, J.H., Gutierrez, D., Kim, M.H.: Practical SVBRDF Acquisition of 3D Objects with Unstructured Flash Photography. ACM TOG 37(6), 1–12 (2018)

50. Olson, E.: AprilTag: A robust and flexible visual fiducial system. In: ICRA (2011)

51. Pamart, A., Ponchio, F., Abergel, V., Alaoui M’Darhri, A., Corsini, M., Dellepiane, M., Morlet, F., Scopigno, R., De Luca, L.: A Complete Framework Operating Spatially-Oriented RTI in a 3D/2D Cultural Heritage Documentation and Analysis Tool. In: 3D Virtual Reconstruction and Visualization of Complex Architectures. ISPRS, vol. XLII-2/W9 (2019)

52. Pamart, A., Morlet, F., De Luca, L., Véron, P.: A Robust and Versatile Pipeline for Automatic Photogrammetric-Based Registration of Multimodal Cultural Heritage Documentation. Remote Sensing 12(12), 2051 (2020)

53. Park, J., Sinha, S.N., Matsushita, Y., Tai, Y.W., Kweon, I.S.: Multiview Photometric Stereo Using Planar Mesh Parameterization. In: ICCV (2013)

54. Pintus, R., Dulecha, T.G., Ciortan, I., Gobbetti, E., Giachetti, A.: State-of-the-art in Multi-Light Image Collections for Surface Visualization and Analysis. Comput. Graph. Forum 38(3), 909–934 (2019)

55. Pixar: Tractor. https://renderman.pixar.com/tractor, accessed 2026

56. Remondino, F.: Heritage Recording and 3D Modeling with Photogrammetry and 3D Scanning. Remote Sensing 3(6), 1104–1138 (2011)

57. Remondino, F., El-Hakim, S.: Image-Based 3D Modelling: A Review. The Photogrammetric Record 21(115), 269–291 (2006)

58. Righetto, L., Khademizadeh, M., Giachetti, A., Ponchio, F., Gigilashvili, D., Bettio, F., Gobbetti, E.: Eficient and User-Friendly Visualization of Neural Relightable Images for Cultural Heritage Applications. ACM J. Comput. Cult. Herit. 17(4), 1–24 (2024)

59. SA, P.: Pix4Dmapper. https://www.pix4d.com, accessed 2026

60. Schönberger, J.L., Frahm, J.M.: Structure-from-Motion Revisited. In: CVPR (2016)

61. Schönberger, J.L., Zheng, E., Frahm, J.M., Pollefeys, M.: Pixelwise View Selection for Unstructured Multi-View Stereo. In: ECCV (2016)

62. Sun, C., Cai, G., Li, Z., Yan, K., Zhang, C., Marshall, C., Huang, J.B., Zhao, S., Dong, Z.: Neural-PBIR Reconstruction of Shape, Material, and Illumination. In: ICCV (2023)

63. Tetzlaf, M., Meyer, G.: Using Flash Photography and Image-Based Rendering to Document Cultural Heritage Artifacts. In: GCH. pp. 137–146 (2016)

64. Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., Wang, W.: NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-View Reconstruction. In: NeurIPS (2021)

65. Wang, Y., Han, Q., Habermann, M., Daniilidis, K., Theobalt, C., Liu, L.: NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-View Reconstruction. In: ICCV (2023)

66. Woodham, R.J.: Photometric Method for Determining Surface Orientation from Multiple Images. Opt. Engineering 19(1), 139–144 (1980)

67. Yang, W., Chen, G., Chen, C., Chen, Z., Wong, K.Y.K.: PS-NeRF: Neural Inverse Rendering for Multi-view Photometric Stereo. In: ECCV (2022)

68. Yastikli, N.: Documentation of Cultural Heritage Using Digital Photogrammetry and Laser Scanning. JCH 8(4), 423–427 (2007)

69. Zhang, K., Luan, F., Li, Z., Snavely, N.: IRON: Inverse Rendering by Optimizing Neural SDFs and Materials From Photometric Images. In: CVPR (2022)