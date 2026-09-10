# Morphological Decoupling-Based Skeletal Classification for Clinical Assessment of Malocclusion

Zhichun Jin<sup>a,b,c,1</sup>, Zhicheng He<sup>d,1</sup>, Hao Xu<sup>a,b,c</sup>, Dongyang Li<sup>a,b,c</sup>, Lin Wang<sup>a,b,c</sup>, Hongliang Ren<sup>d</sup>, Long Bai<sup>d,∗</sup>

<sup>a</sup>The Afiliated Stomatological Hospital ofNanjing Medical University, Nanjing, China <sup>b</sup>State Key Laboratory Cultivation Base of Research, Prevention and Treatment for Oral Diseases, Nanjing Medical University, Nanjing, China <sup>c</sup>Jiangsu Province Engineering Research Center ofStomatological Translational Medicine, Nanjing, China <sup>d</sup>Department ofElectronic Engineering, The Chinese University ofHong Kong, Hong Kong SAR, China

## Abstract6

Malocclusion skeletal grading is a fundamental task in orthodontics, critical for diagnosis and treatment planning. Traditionally, cone-beam computed tomography (CBCT) is used for visual measurement, and the reconstructed lateral cephalograms are handed2 over to expert dentists for diagnosis. However, manual review is time-consuming, labor-intensive, and subject to inter-operator variability. Therefore, an automatic CBCT-based system is needed for reliable malocclusion skeletal grading. In this case, wee develop TeethGNN, a novel graph-based framework designed to combine CBCT image features with morphological information forS accurate and eficient malocclusion grading. TeethGNN utilizes a decoupled learnable decoder to directly predict key morphological indicators from CBCT images, eliminating the need for manual measurements. These morphological features are then fused with image features using a graph neural network (GNN), which efectively models the relationships between the modalities. To further enhance robustness and calibration, we introduce a collaborative calibration strategy. This strategy combines multi-scale graph adversarial perturbation for explicit calibration and nonlinear topological graph calibration for implicit confidence adjustment.. Extensive experiments and ablation studies on our collected clinical dataset demonstrate that our malocclusion measurement system<sup>s</sup> achieves 77.08% in accuracy and 89.61% in AUC, outperforming the compared state-of-the-art methods. These results validate thee efectiveness of graph-based multimodal fusion and collaborative calibration in improving malocclusion grading performance. Our<sup>e</sup> system shows strong potential for advancing computer-aided orthodontic diagnosis, providing an accurate and reliable solution for vision-based clinical measurement and diagnosis.<sup>1</sup>

Keywords:

Skeletal Classification, Malocclusion Diagnosis, Graph Neural Network, Multimodal Learning, Network Calibration

## 1. Introduction.

The malocclusion skeletal grading is a widely used diagnostic framework used to categorize malocclusion cases into three<sup>6</sup> classes (Class I, II, and III) based on the positional relationship between the maxilla and mandibula [1, 2, 3, 4, 5]. Malocclu-v sion afects patients’ oral functions, facial aesthetics, and mental health, making early intervention and treatment essential for achieving better outcomes. Skeletal grading represents difer-a ent malocclusion patterns and their severity. Accurate classification results are crucial for orthodontists to develop appropriate treatment plans. Traditionally, this classification relies on medical measurement systems with cone-beam computed tomography (CBCT), where expert orthodontists perform visual inspections and manual measurements on reconstructed lateral cephalograms [6]. While efective, these methods are timeconsuming and prone to inter-operator variability, particularly in complex cases [7].

Recently, artificial intelligence (AI) algorithms have been increasingly integrated with vision-based measurement systems in healthcare [8, 9, 10]. In the field of dentistry, numerous approaches based on convolutional neural networks (CNNs) and Transformers have been combined with X-ray- or CBCTbased medical measurement systems to assist clinicians in diagnosis [11, 12, 13]. Existing medical measurement algorithms typically train a feature extractor to extract visual features from images, use a classification layer to output logits, and extract the class with the highest probability. While this approach performs eficient representation learning on medical images and often achieves good results, the black-box nature of such methods leaves room for improvement. Specifically, in clinical scenarios, the malocclusion skeletal grading is often based on specific morphological indicators of the CBCT reconstructed lateral cephalograms, such as the angle between nasion-subspinale and nasion-supramental (ANB), and the angle between the mandibular plane and the Frankfort horizontal plane (MP-FH). These indicators are typically measured manually or through morphological methods, with the classification results determined accordingly. Inspired by this process, our network aims to follow a similar approach: first decoupling the original input image to extract morphological information, and then integrating this information with the original image to assist in downstream orthodontic diagnosis tasks.

In practical applications of our measurement algorithm, although we aim to efectively integrate morphological information with image data, we intend for the model’s input to consist solely of the original image. Otherwise, if the input still requires manually measured morphological information, it would impose a significant burden on medical practitioners. To address this, we first design a decoupling learnable decoder to model the ANB and MP-FH morphological indicators. With the input of the model remaining in the single CBCT reconstructed lateral cephalogram, the network shall directly predict ANB and MP-FH through these decoders. Subsequently, we explored which model architecture is best suited for multimodal information fusion between the morphological indicators and the image features in this context. Simple approaches, such as concatenation, summation, or attention-based fusion, have been widely studied for multimodal tasks. However, we believe these approaches are not well-suited to our scenario because the morphological features are simple numerical values, making it dificult to extract meaningful features and efectively fuse them with image features.

In this case, an efective measurement solution is to use the Graph Neural Network (GNN) [14, 15]. GNN is a deep learning model designed for processing graph-structured data and can efectively model information about nodes, edges, and their connections [16, 17]. The core principle of GNN is based on a message-passing mechanism, where each node aggregates information from its neighbors and updates its representation by combining it with its own features. This process allows GNN to gradually capture both local and global features within the graph structure. Through multi-layer propagation, GNN can learn embeddings for nodes, edges, or the entire graph. In our case, GNNs shall represent information from diferent modalities (i.e., morphological information and image features) as distinct graph nodes and learn the node relationships. This approach ensures that morphological information, even when represented as simple numerical values, is not overlooked. It also enables efective fusion without requiring complex feature extraction. Therefore, we construct separate graph nodes for the decoupled morphological information and image representations. Through graph representation learning (GRL), we establish meaningful connections between graph nodes formed by the decoupled morphological information and image features, facilitating better interaction between modalities and improving fusion performance.

However, although graph-based measurement approaches can efectively model relationships between diferent modalities, the shallow architecture of GNNs typically limits their depth compared to traditional CNNs [18, 19, 20]. Studies have shown that this relatively shallow architecture may lead to the model being uncalibrated. In addition, GNNs rely heavily on the quality of neighboring information and graph structure, as well as smaller training datasets and dynamic graph characteristics, which further exacerbate the mismatch between predicted confidence and true probability. These factors make the output probabilities of GNNs less aligned with true confidence, making calibration crucial in practical applications [21, 22]. An uncalibrated model results in a mismatch between predicted probabilities and actual accuracy, undermining reliability and performance. The model’s confidence can no longer serve as a trustworthy indicator of correctness, rendering probability outputs less meaningful. In tasks involving threshold-based decision-making, uncalibrated models may lead to improper threshold settings, increasing false-positive or false-negative rates. These issues reduce the robustness and applicability of the model, limiting its use in real-world scenarios.

In our vision-based medical measurement system, in addition to innovatively using a decoupling GNN to integrate morphological indicators and image representations for malocclusion skeletal grading, we further address the issue of GNN miscalibration described above. Specifically, we perform network cal ibration at diferent stages of the GNN, thereby enhancing the accuracy and robustness of our model in diagnosis. First, during the construction of graph nodes, we introduce gradient-based adversarial perturbations into the embedding of graph nodes. Without altering the topology of the graph, we inject multiscale perturbations into the embedding space of the nodes. By employing an eficient adversarial training strategy that integrates perturbation generation with model optimization, we improve the robustness of the model and achieve explicit calibration. Subsequently, when generating the logits output of the GNN, we adjust the confidence of the nodes by leveraging the graph’s topology and nonlinear transformations. This ensures that the confidence of neighboring nodes becomes more consistent, achieving implicit calibration and improving the overall performance. Our contributions can be summarized as:

We propose an intelligent CBCT-based measurement system for malocclusion skeletal grading, in which we develop TeethGNN, a decoupled graph-based framework by modeling oral morphological information and visual representations. This enables precise and eficient measurement and diagnosis progress, assisting dentists in the process of decision-making and treatment planning.

We apply explicit and implicit calibration techniques to the diferent stages of our TeethGNN. We inject adversarial perturbations as explicit multi-scale adversarial calibration while constructing graph nodes, and apply implicit graph nonlinear topological calibration during the prediction stage. The proposed collaborative graph calibration strategy leads to robust and accurate diagnosis predictions of our TeethGNN.

– We conduct extensive comparative and ablation experiments. The results demonstrate that our intelligent measurement system achieves superior performance in malocclusion skeletal grading, consistently outperforming existing classification solutions. Furthermore, our calibration design efectively improves the performance. These findings highlight the strong potential of our CBCT-based measurement system in computer-aided diagnosis.

## 2. Related Work

## 2.1. Algorithm-assisted Skeletal Gradingfor Malocclusion

Recent advancements in AI have considerably improved the eficiency and accuracy of oral measurement and malocclusion grading [23, 24, 25, 26]. Various types of data have been utilized for this purpose, including occlusal contact information and traditional classification methods such as skeletal classification. For instance, Yao et al. developed a device for measuring bite force integrated with a random forest model, achieving 87.83% accuracy [27]. This approach ofers a non-invasive diagnostic tool that reduces radiation exposure. Additionally, multi-modal frameworks combining CBCT, panoramic X-rays, and 3D dental arch models are showing promise in improving diagnostic accuracy and robustness [28, 29], enabling a more comprehensive analysis using multiple data sources.

Deep learning models, particularly CNNs, have emerged as the cornerstone of automated systems for malocclusion classification [30, 31, 32]. CNN algorithms trained on lateral cephalograms have achieved impressive performance, with landmark detection rates exceeding 98% [33]. Other studies using cascaded CNNs on posteroanterior (PA) cephalograms have demonstrated point-to-point error as low as 1.26mm [34], while CNNs have also been applied to optimize performance with other imaging modalities, such as PA cephalograms [35] and lateral facial photographs [36]. Furthermore, recent studies have incorporated graph-based approaches, such as GNNs, to model neighborhood relationships in medical imaging, including dentistry [37, 38]. GNNs capture both spatial and relational information, improving classification systems by integrating local feature extraction with relational modeling [39]. This hybrid approach enhances the accuracy and robustness of malocclusion classification systems, marking notable progress in AI applications within orthodontics.

Despite advancements in AI, several limitations persist in CNN-based methods for malocclusion grading, including small sample sizes, inability to predict the direction of landmark changes, and reliance on 2D images like lateral cephalograms, which omit 3D information [40]. Additionally, variability in landmark definitions and input data quality afect accuracy. Our work addresses these challenges by applying GNNs to skeletal grading, improving the capture of complex spatial relationships and enhancing the robustness, accuracy, and generalizability of AI systems for orthodontics, especially in cases where traditional methods fall short.

## 2.2. Graph Representation Learning in Medical Scenarios

Building upon the foundational advancements, GRL has demonstrated immense potential in biomedicine. In genomics, GRL employs graph-based modeling of gene interactions to predict gene functions, identify associations between genes and diseases, and improve protein function prediction in proteinprotein interaction networks, facilitating the identification of drug targets critical in drug discovery eforts [41, 42, 43]. Medical knowledge graphs (MKGs) represent another transformative application. Constructed from heterogeneous data sources, GNNs facilitate applications like drug interaction prediction and personalized treatment recommendations [44, 45]. GNNs have also demonstrated utility in predicting adverse drug reactions by learning from large-scale MKGs [46, 47, 48]. In the domain of medical imaging analysis, GNNs extend their capabilities beyond traditional Convolutional Neural Networks (CNNs) by modeling the intricate spatial and topological relationships inherent in medical images. By representing pixels or regions as nodes and their spatial relationships as edges, GNNs have achieved state-of-the-art results in tasks such as tumor detection [49, 50]. By integrating data from modalities such as CT, MRI, and PET scans, GNNs construct comprehensive representations that account for the complementary strengths of each modality, which has been particularly beneficial in early cancer detection and the assessment of neurodegenerative diseases [51]. The modeling of organ and tissue structures also uses anatomical graphs, where nodes represent anatomical regions or landmarks, and edges capture the relationships between them. This representation enables organ shape analysis and automated diagnosis in congenital heart defect scenarios [52].

GNNs have been extensively explored in dental applications. In particular, TeethGNN employs GNNs to directly process non-Euclidean 3D mesh data, enhancing the eficiency and robustness of tooth segmentation against challenges such as malocclusions and scanning noise [53]. Similarly, TEANet introduces a node-erasable adaptive graph specifically designed for tooth arrangement in orthodontic treatments, addressing complex cases like tooth extraction and overcrowded dentition [54]. Furthermore, comparative studies on panoramic X-rays have demonstrated that GNNs outperform traditional convolutional neural networks (CNNs) by capturing non-Euclidean spatial relationships, resulting in superior diagnostic accuracy for dental diseases [55]. However, most GNN applications in orthodontics are limited, with fewer studies focusing on their use in skeletal classification.

## 3. Methods

## 3.1. AI-assisted System

As shown in Fig. 1, the proposed malocclusion skeletal grading system is based on the CBCT (NewTom VGi, Italy) imaging machine. This CBCT machine is equipped with a high sensitivity flat-panel detector and a micro-focus X-ray tube, supporting various field-of-view (FOV) options to meet diferent clinical needs. For orthodontic evaluation, full cranial imaging was used with exposure parameters set to the FOV of 15 × 15 cm, a tube voltage of 110 kV, a tube current of 5–10 mA, and a focal spot size of 0.3 mm, enabling the reconstruction of lateral cephalograms. During the scan, the patient is positioned with the midsagittal plane parallel to the scanning plane, lips gently closed, and the upper and lower teeth in a natural occlusion position. Finally, the images are processed using our TeethGNN, and the prediction results are provided to dentists as diagnostic references.

![](images/205185629b8fdc0b092c13c6c02fae231a91b7b44367ace16313f2ce949c7f35.jpg)  
Figure 1: Overview of our computer-assisted diagnosis system. TeethGNN will serve as the intelligent hub linking dentists and the imaging system.

## 3.2. Preliminaries: GNNs

Graph Neural Networks are designed to process graphstructured data. A graph is typically represented as $G = ( V , E )$ where V is the set of nodes and E is the set of edges. For each node $\nu \in V ,$ a feature vector $\mathbf { X } _ { \nu }$ is associated. The goal of GNNs is to learn meaningful node or graph representations by aggregating information from neighboring nodes. A common formulation of a GNN layer is:

$$
\begin{array} { r } { \mathbf { h } _ { \nu } ^ { ( k ) } = \sigma \big ( \mathbf { W } ^ { ( k ) } \cdot \mathrm { A G G } \left( \left\{ \mathbf { h } _ { u } ^ { ( k - 1 ) } : u \in N ( \nu ) \right\} \cup \left\{ \mathbf { h } _ { \nu } ^ { ( k - 1 ) } \right\} \right) \big ) } \end{array}\tag{1}
$$

where $\mathbf { h } _ { \nu } ^ { ( k ) }$ is the node embedding of v at the k-th layer, $N ( \nu )$ denotes the neighbors of v, AGG(·) represents the aggregation function, $\mathbf { W } ^ { ( k ) }$ is a trainable weight matrix, and $\sigma ( \cdot )$ is a nonlinear activation function. The node embeddings are initialized as ${ \bf h } _ { \nu } ^ { ( 0 ) } = { \bf x }$ <sub>v</sub> using the input features. By stacking multiple GNN layers, the model captures higher-order neighborhood information, enabling efective representation learning for tasks such as node classification, link prediction, and graph classification.

## 3.3. Proposed Framework: TeethGNN

## 3.3.1. Multi-modality Teeth Graph

Our model architecture is illustrated in Fig. 2. TeethGNN processes the input image I through two separate pathways: the image pathway and the morphological information pathway. These pathways are designed to capture complementary information: the image pathway focuses on extracting global visual features, and the morphological pathway learns specific morphological information critical for diagnosis.

In the image pathway, a standard ResNet50 [56] is employed as a feature extractor. ResNet50 is widely used for its ability to capture hierarchical features through residual connections, making it well-suited for visual representation learning. The extracted feature map $F$ can be formulated as:

$$
F = \mathrm { R e s N e t } 5 0 ( I ) ,\tag{2}
$$

where F represents the high-dimensional feature representation of the input image I. These features provide a rich description of the image’s global context, which is crucial for downstream diagnosis tasks.

In the morphological information pathway, we focus specifically on the morphological information ANB and MP-FH, which are essential for capturing the structural relationships within the image. To process these morphological features, we utilize a ResNet50 [56] as the encoder for each angular feature. The encoder is followed by a lightweight linear decoder to predict the respective angular values. This process can be described as:

$$
\hat { y } _ { \mathrm { A N B } } = \mathrm { D e c o d e r } _ { \mathrm { A N B } } ( \mathrm { E n c o d e r } _ { \mathrm { A N B } } ( I ) ) ,\tag{3}
$$

$$
\hat { y } _ { \mathrm { M P - F H } } = \mathrm { D e c o d e r } _ { \mathrm { M P - F H } } ( \mathrm { E n c o d e r } _ { \mathrm { M P - F H } } ( I ) ) ,\tag{4}
$$

where $\hat { y } _ { \mathrm { A N B } }$ and ˆy denote the predicted angular values for ANB and MP-FH, respectively. Each encoder-decoder pair is trained to specialize in extracting and predicting the corresponding angular feature, ensuring accurate modeling of the morphological information.

To optimize the learning of these angular features, we adopt the Mean Squared Error (MSE) loss. The MSE loss is commonly used for regression tasks due to its simplicity and efectiveness in penalizing large deviations. The loss function for the morphological pathway is defined as:

$$
\mathcal { L } _ { \mathrm { m o r p h } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( ( y _ { \mathrm { A N B } } ^ { ( i ) } - \hat { y } _ { \mathrm { A N B } } ^ { ( i ) } ) ^ { 2 } + ( y _ { \mathrm { M P - F H } } ^ { ( i ) } - \hat { y } _ { \mathrm { M P - F H } } ^ { ( i ) } ) ^ { 2 } \right) ,\tag{5}
$$

![](images/e48b61c569cf9a766cffbabc51bdf81337c8d17adc25c83f389c9887ca20d660.jpg)  
Figure 2: Overview of the proposed framework. (a) Warm-up stage: the image encoder is trained using MSE loss to initialize morphological information extraction. (b) Full training stage: the extracted morphological information and image features are jointly fed into a graph network for representation learning. The explicit multi-scale adversarial calibration and implicit graph nonlinear topological calibration are applied to achieve accurate and robust diagnostic results. Trainable and frozen modules are marked separately.

where $y _ { \mathrm { A N B } } ^ { ( i ) }$ and $y _ { \mathrm { M P - F H } } ^ { ( i ) }$ are the ground truth values for the i-th sample, and N is the total number of training samples. This loss ensures that the predicted angular values closely match the ground truth, providing the model with precise morphological feature representations.

After obtaining ˆy<sub>ANB</sub>, ˆy<sub>MP-FH</sub>, and F, we represent these complementary sources of information as a task-specific multimodal graph $G = ( V , E )$ , where $V = \{ \nu _ { F } , \nu _ { \mathrm { A N B } } , \nu _ { \mathrm { M P - F H } } \}$ . The three nodes correspond to the image feature $F ,$ the predicted ANB measurement $\hat { y } _ { \mathrm { A N B } }$ , and the predicted MP-FH measurement ˆy<sub>MP-FH</sub>, respectively. A fully connected topology with self-loops is adopted so that each node can exchange information with both the image representation and the morphologyderived skeletal indicators. This compact topology provides an explicit fusion structure for the two scalar morphological measurements and the high-dimensional image feature, reducing the risk that the morphological information is diluted by direct feature concatenation. The GNN follows a graph-convolutional message-passing scheme, in which AGG(·) aggregates neighboring node embeddings and node-specific trainable transformations update each node representation. To learn the graph representations, the node embeddings in the GNN are itera-

tively updated as follows:

$$
\begin{array} { r } { \begin{array} { r l } & { \mathbf { h } _ { F } ^ { \left( k \right) } = \sigma \left( \mathbf { W } _ { F } ^ { \left( k \right) } \cdot \mathrm { A G G } \left( \mathbf { h } _ { \mathrm { A N B } } ^ { \left( k - 1 \right) } , \mathbf { h } _ { \mathrm { M P - F H } } ^ { \left( k - 1 \right) } , \mathbf { h } _ { F } ^ { \left( k - 1 \right) } \right) \right) , } \\ & { \mathbf { h } _ { \mathrm { A N B } } ^ { \left( k \right) } = \sigma \left( \mathbf { W } _ { \mathrm { A N B } } ^ { \left( k \right) } \cdot \mathrm { A G G } \left( \mathbf { h } _ { \mathrm { M P - F H } } ^ { \left( k - 1 \right) } , \mathbf { h } _ { F } ^ { \left( k - 1 \right) } , \mathbf { h } _ { \mathrm { A N B } } ^ { \left( k - 1 \right) } \right) \right) , } \\ & { \mathbf { h } _ { \mathrm { M P - F H } } ^ { \left( k \right) } = \sigma \left( \mathbf { W } _ { \mathrm { M P - F H } } ^ { \left( k \right) } \cdot \mathrm { A G G } \left( \mathbf { h } _ { \mathrm { A N B } } ^ { \left( k - 1 \right) } , \mathbf { h } _ { F } ^ { \left( k - 1 \right) } , \mathbf { h } _ { \mathrm { M P - F H } } ^ { \left( k - 1 \right) } \right) \right) , } \end{array} } \end{array}\tag{6}
$$

where ${ \bf h } _ { F } ^ { ( k ) } , { \bf h } _ { \mathrm { A N B } } ^ { ( k ) } .$ , and ${ \bf h } _ { \mathrm { M P - F H } } ^ { ( k ) }$ represent the embeddings of the image feature node, the ANB node, and the MP-FH node at the k-th layer, respectively. $\mathbf { W } _ { F } ^ { ( k ) } , \mathbf { W } _ { \mathrm { A N B } } ^ { ( \dot { k } ) } ,$ , and $\mathbf { W } _ { \mathrm { M P - F H } } ^ { ( k ) }$ are trainable weight matrices associated with each node type at the k-th layer. The initial node embeddings ${ \bf h } _ { F } ^ { ( 0 ) } , { \bf h } _ { \mathrm { A N B } } ^ { ( 0 ) } ,$ , and ${ \bf h } _ { \mathrm { M P - F H } } ^ { ( 0 ) }$ are initialized directly from their respective features, such that:

$$
{ \bf h } _ { F } ^ { ( 0 ) } = F , \quad { \bf h } _ { \mathrm { A N B } } ^ { ( 0 ) } = \widehat { y } _ { \mathrm { A N B } } , \quad { \bf h } _ { \mathrm { M P - F H } } ^ { ( 0 ) } = \widehat { y } _ { \mathrm { M P - F H } } .\tag{7}
$$

By iteratively updating the node embeddings, the GNN effectively captures the interactions between the image feature and the morphological information, producing a unified representation of their relationships. After the GNN layers, the final embeddings of the nodes are concatenated into a single vector, which is passed through a linear layer for diagnosis classification. The diagnosis task is supervised using the cross-entropy loss, defined as:

$$
\mathcal { L } _ { \mathrm { c l s } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { C } y _ { i } ^ { c } \log ( \hat { y } _ { i } ^ { c } ) ,\tag{8}
$$

where N is the total number of samples, C is the number of classes, $y _ { i } ^ { c }$ is the ground truth label for class c of sample i, and $\hat { y } _ { i } ^ { c }$ is the predicted probability for the same class. During the graph node construction and network inference, we further apply Multi-scale Graph Adversarial Perturbation and Nonlinear Topological Graph Calibration to help the model achieve more accurate and robust performance. The specific details are provided in Section 3.3.2 and Section 3.3.3.

## 3.3.2. Multi-scale Graph Adversarial Perturbation

To improve the robustness and calibration of the GNN in malocclusion grading, we propose a multi-scale graph adversarial perturbation strategy inspired by [21], which introduces gradient-based perturbations into the graph node embeddings. Unlike methods that modify graph topology, our approach operates directly in the embedding space, preserving the graph’s structural integrity while enhancing feature diversity. Given the constructed graph, we define the adversarial perturbation δ as additive noise applied to the initial node embeddings. At the t-th iteration, the perturbed embeddings are formulated as:

$$
\begin{array} { r } { \begin{array} { c } { { \mathbf { h } _ { F } ^ { ( 0 , t ) } = F + \delta _ { F } ^ { ( t ) } , } } \\ { { \mathbf { h } _ { \mathrm { A N B } } ^ { ( 0 , t ) } = \hat { y } _ { \mathrm { A N B } } + \delta _ { \mathrm { A N B } } ^ { ( t ) } , } } \\ { { \mathbf { h } _ { \mathrm { M P - F H } } ^ { ( 0 , t ) } = \hat { y } _ { \mathrm { M P - F H } } + \delta _ { \mathrm { M P - F H } } ^ { ( t ) } , } } \end{array} } \end{array}\tag{9}
$$

where $\delta _ { F } ^ { ( t ) } , \delta _ { \mathrm { A N B } } ^ { ( t ) }$ , and $\delta _ { \mathrm { M P - F H } } ^ { ( t ) }$ are perturbations introduced to the corresponding node embeddings. These perturbations are optimized using a projected gradient ascent method. At each step, the perturbations are iteratively updated to maximize the training loss:

$$
\begin{array} { r } { \delta _ { \nu } ^ { ( t + 1 ) } = \mathrm { P r o j } \left( \delta _ { \nu } ^ { ( t ) } + \lambda \cdot \mathrm { s i g n } \left( \nabla _ { \delta _ { \nu } ^ { ( t ) } } \mathcal { L } ( \mathbf { h } _ { \nu } ^ { ( t ) } , y ) \right) \right) , } \end{array}\tag{10}
$$

where $\delta _ { \nu } ^ { ( t ) }$ represents the perturbation for node v at iteration $t ,$ λ is the step size, and L is the training loss function. The projection operator Proj(·) ensures that the perturbations remain within a predefined norm ball to prevent excessive deviation from the original embeddings.

Additionally, we apply a multi-scale strategy by introducing perturbations with varying magnitudes. For each node type, the perturbations are scaled by factors {1 2 S } during training to create a diverse set of adversarial examples, which can be formulated as:

$$
\begin{array} { r } { \delta _ { \nu } ^ { ( t , m ) } = s \cdot \lambda \cdot \mathrm { s i g n } \left( \nabla _ { \delta _ { \nu } ^ { ( t ) } } \mathcal { L } ( \mathbf { h } _ { \nu } ^ { ( t ) } , y ) \right) , \quad s \in \{ 1 , 2 , \dots , S \} . } \end{array}\tag{11}
$$

Therefore, this multi-scale approach improves the model’s ability to generalize by exposing it to a broad spectrum of input variations.

By eficiently integrating perturbation generation and model optimization, this approach minimizes additional computational costs while improving robustness. The multi-scale adversarial perturbations enhance the GNN’s resilience to noisy inputs, and explicitly calibrate the network, ensuring accurate and reliable predictions for malocclusion grading.

## 3.3.3. Nonlinear Topological Graph Calibration

To enhance the trustworthiness and accuracy of malocclusion grading, we propose a nonlinear topological graph calibration mechanism. This module adjusts the output logits of the GNN by propagating confidence values across the graph topology, ensuring that neighboring nodes have more consistent confidence levels. By using nonlinear transformations, this approach implicitly calibrates the model while preserving its classification accuracy.

Given the uncalibrated logits of each node, denoted as v<sub>ANB</sub>, v<sub>MP-FH</sub>, and $\mathbf { v } _ { F }$ for the morphological information ˆy<sub>ANB</sub>, ˆy<sub>MP-FH</sub>, and the image feature F, the calibration process applies a nodespecific transformation:

$$
{ \bf v } _ { i } ^ { \prime } = \frac { { \bf v } _ { i } } { t _ { i } } , \quad \mathrm { f o r } i \in \{ F , \mathrm { A N B , M P \mathrm { - } F H } \} ,\tag{12}
$$

where $t _ { i } > 0$ is a learned temperature parameter that scales the logits. This transformation ensures that the relative ranking of class probabilities remains unchanged, thus preserving the classification accuracy.

To further ensure consistency between the confidence values of connected nodes, we add a regularization term that minimizes the total variation of confidence across edges:

$$
\mathcal { L } _ { \mathrm { c a l } } = \sum _ { ( i , j ) \in E } \left\| \mathbf { v } _ { i } ^ { \prime } - \mathbf { v } _ { j } ^ { \prime } \right\| _ { 2 } ^ { 2 } ,\tag{13}
$$

where E represents the edges of the graph. This term encourages nodes with similar features or logits to have closer confidence levels after calibration. This nonlinear calibration method ensures that the confidence values of nodes are consistent with the graph’s structure, enhancing both the robustness and reliability of the GNN’s predictions.

## 3.3.4. Training and Inference Recipe

To efectively train the model, we adopt a two-stage training strategy, including the warm-up and full training stages. We further describe the inference behavior to clarify the use of ground-truth and predicted morphological values.

Warm-up Training Stage. In the warm-up stage, we exclusively train the morphological information pathway. This stage allows the model to focus solely on learning the morphological features y<sub>ANB</sub> and y<sub>MP-FH</sub> without interference from the classification task. By minimizing ${ \mathcal { L } } _ { \mathrm { m o r p h } }$ during this phase, the model is equipped with the ability to extract accurate angular features. The optimization function during the warm-up stage shall be represented as:

$$
\mathcal { L } _ { \mathrm { w a r m } } = \mathcal { L } _ { \mathrm { m o r p h } } ,\tag{14}
$$

This is the only phase in which the ground-truth angular values are used, namely as the regression targets of ${ \mathcal { L } } _ { \mathrm { m o r p h } } .$

Full Training Stage. In the full training stage, the morphological pathway is no longer supervised with ground-truth angular values. Instead, it operates as an auxiliary module to support the image pathway: for each image it produces the predicted angles ˆy<sub>ANB</sub> and ˆy<sub>MP-FH</sub>, which are used to initialize the morphological graph nodes (Eq. 7). The training objective shifts to optimizing the diagnostic classification task, where the loss function is:

$$
\mathcal { L } _ { \mathrm { f u l l } } = { \mathcal { L } } _ { \mathrm { c l s } } + \alpha { \mathcal { L } } _ { \mathrm { c a l } } ,\tag{15}
$$

where α is a hyperparameter that balances the trade-of between classification accuracy and calibration smoothness. This stage ensures that the model leverages both global image features and auxiliary morphological information to improve its classification performance. Since the graph is built only from the predicted angles, our model uses only the image and diagnostic information as input and output, without including additional ground-truth morphological information, thereby eliminating any suspicion of ‘cheating.’ By combining the strengths of the image pathway and the morphological pathway, our model achieves a robust representation of both global and structural information, which is crucial for accurate diagnosis.

Inference Stage. During the inference stage, the behavior is consistent with the full training stage. The model takes only the lateral cephalogram as input, predicts ˆy<sub>ANB</sub> and yˆ<sub>MP-FH</sub> through the morphological pathway, initializes the morphological nodes with these predicted values, builds the graph, and outputs the skeletal class. No manual measurement is required, which matches the intended clinical use where ground-truth angular values are not available.

## 4. Results

## 4.1. Dataset

The orthodontic data collection was conducted at the Anonymous Hospital, which contains CBCT-reconstructed lateral cephalograms from 476 patients undergoing malocclusion diagnosis. Specifically, the training and testing sets contain 428 and 48 samples, respectively, with cross-validation to make the results reliable. The dataset includes detailed coordinate annotations and cephalometric measurements such as ANB, MP-FH, etc. Expert dentists manually annotate the skeletal grading labels, which can be categorized into Skeletal Class I, Skeletal Class II (angle class II, division 1), Skeletal Class II (angle class II, division 2), and Skeletal Class III, as presented in Fig. 3. The study was approved by the Ethical Committee Department of the Anonymous Hospital.

## 4.2. Implementation Details

Our proposed solution is evaluated against the following comparison methods:

• CNN: ResNet18<sup>2</sup> [56], ResNet50<sup>2</sup> [56], RegNet<sup>2</sup> [57], EficientNet<sup>2</sup> [58], MobilenetV2<sup>2</sup> [59], ShufleNet<sup>2</sup> [60], DenseNet121<sup>2</sup> [61], ConvNeXtV2<sup>3</sup> [62];

• Transformer: ViT<sup>3</sup> [63], Swin<sup>3</sup> [64], RIFormer<sup>4</sup> [65], MaxVit<sup>3</sup> [66], HiViT<sup>4</sup> [67], EficientFormer<sup>3</sup> [68];

• Foundation Models: DINOv2<sup>2</sup> [69], EVA-02<sup>3</sup> [70];

• Mamba: VMamba<sup>5</sup> [71];

• GNN: VisionGNN<sup>4</sup> [72], GCN<sup>6</sup> [73], GIN<sup>7</sup> [74], GAT<sup>8</sup> [17].

All models were trained and evaluated under the same fourclass label definition, training/test split, and test-time protocol. Data augmentation was applied only to the training images, while the test set is kept unchanged. The 12 augmentation operations were applied independently to the original images rather than compounded with one another, including resize, rotation, afine transformation, blur, color jitter, resized crop, grayscale conversion, horizontal flip, vertical flip, sharpness adjustment, random erase, and random perspective transformation.

TeethGNN was optimized with Adam using a learning rate of 0.0001, a batch size of 16, and up to 1000 epochs. The code for all baseline methods is sourced from the public repositories indicated in the footnotes. For image-only baselines (CNN, Transformer, Mamba, and GNN), the final classification heads were set as the four-class classifiers, and the models were fine-tuned end-to-end from oficial ImageNet-pretrained releases when such releases were publicly available. When official releases specified architecture-dependent input resolution, normalization, or fine-tuning settings, those settings were followed; when no specific fine-tuning recipe was provided, the common fine-tuning setting used Adam with a learning rate of 0.0001, a batch size of 16, and up to 1000 epochs. If ImageNetpretrained weights were not publicly released for the exact variant, we used the oficial implementation or the released initialization provided by the model authors and trained the model under the same common schedule. For the foundation-model baselines, DINOv2 and EVA-02 were initialized from their official pretrained releases and fine-tuned with a four-class classifier. No hyperparameter was selected using the test labels.

## 4.3. Experimental Results

The experimental results in Table 1 demonstrate the performance advantages of our proposed TeethGNN model compared with the baseline methods across the major evaluation metrics. Among the CNN-based models, ResNet50 achieves the highest accuracy (0.7500), surpassing the other CNN models. However, even ResNet50 remains lower than TeethGNN: TeethGNN achieves an accuracy of 0.7708 and an F1-score of 0.6516, exceeding ResNet50 (0.7500 accuracy and 0.5600 F1-score) by gaps of 0.0208 and 0.0916, respectively. This highlights the limitations of traditional convolutional architectures in fully capturing the structural and relational features inherent in our dataset. Transformer-based models generally perform worse than the best CNN baseline: ViT, MaxVit, and Swin reach accuracies of 0.5625, 0.6458, and 0.6667, respectively, all below ResNet50 (0.7500). This reflects the challenges of applying Transformers directly to medical imaging tasks without suficient adaptation. Although Swin is the strongest Transformer baseline in accuracy (0.6667), it still falls short of TeethGNN (0.7708 accuracy and 0.6516 F1-score), with gaps of 0.1041 and 0.1529. Besides, foundation models show competitive results, particularly EVA-02, with an accuracy of 0.7083 and an F1-score of 0.5415. However, these models still fall below TeethGNN, which is higher by 0.0625 in accuracy and 0.1101 in F1-score, suggesting that pre-trained large-scale models may not be fully optimized for the specific needs of malocclusion grading.

When compared with VisionGNN, the only other graphbased model in the comparison, TeethGNN shows notable improvements. TeethGNN achieves an accuracy that is higher by 0.1041 and an F1-score that is higher by 0.2016 than VisionGNN (0.6667 accuracy and 0.4500 F1-score). This performance gap highlights the efectiveness of our graph-based design in leveraging structural information, which is critical for understanding the relationships in lateral cephalograms. Furthermore, TeethGNN also achieves the highest AUC value (0.8961), indicating its superior ability to distinguish between classes with high confidence. In summary, TeethGNN achieves the best overall results, supporting its ability to address the challenges of malocclusion skeletal grading. These results emphasize the importance of incorporating graph-based representations and customized modeling strategies for this complex orthodontic diagnosis task.

## 4.4. Ablation Studies

Efects of the Calibration Components. Table 2 presents the results of the ablation study on the two proposed modules: Multi-scale Graph Adversarial Perturbation (MGAP) and Nonlinear Topological Graph Calibration (NTGC). In addition to the classification metrics, we report Expected Calibration Error (ECE) and Brier score to assess confidence calibration. The complete model with both MGAP and NTGC achieves the lowest ECE (0.0910) and Brier score (0.4987), while retaining the best accuracy (0.7708), F1-score (0.6516), and AUC (0.8961). These results suggest that the two calibration components improve confidence alignment, without compromising classification performance.

Efects of the Loss Ratio α. Table 3 shows the results of the ablation study on the loss ratio α, which balances the cross-entropy loss and the calibration loss. When α = 0 05, the model achieves the highest accuracy of 0.8125 and a competitive AUC of 0.8843, indicating that a small contribution from the calibration loss improves the model’s overall performance. As α increases to 0.1 and 0.5, the performance slightly decreases in terms of accuracy, recall, and F1-score. However, the AUC increases to 0.8951 at α = 0 1, suggesting that calibration loss may enhance the model’s ability to distinguish classes at certain thresholds. Interestingly, when α = 1, the model achieves the highest precision of 0.8178 and F1-score of 0.7171, but the accuracy drops to 0.7708 and the AUC reduces to 0.8403. This suggests that overly emphasizing calibration loss may lead to overfitting the calibration objective, reducing generalization. Overall, these results demonstrate the importance of tuning α to balance the trade-of between the primary classification task and the calibration objectives efectively. We ultimately selected α = 1 as the final setting because it provides the most balanced performance across all metrics.

Efects of the Scale Factor s. Table 4 presents the results of the ablation study on the scale factor s in the multi-scale perturbation strategy. The scale factor s determines how many levels of scale are included in the model. When s = 1, only scale 1 is used, leading to limited performance with an accuracy of 0.7292 and an AUC of 0.8472. Increasing s to 2 results in moderate improvements, with an accuracy of 0.7500 and an AUC of 0.8841. The best performance is observed when s = 3, where scales 1, 2, and 3 are utilized. In this case, the model achieves the highest accuracy of 0.7708 and AUC of 0.8961, along with a recall of 0.6273, indicating a strong balance across the metrics. However, increasing s beyond 3 does not further improve performance and even reduces precision and recall. These results highlight that using the 3-level scale strikes the optimal balance between the model’s complexity and its performance.

Efects of Diferent Fusion Strategies. As shown in Table 5, compared to other multimodal fusion techniques, our proposed graph-based fusion gives the best results across these metrics. First, our model showcases the top Accuracy of 0.7708, which is about 12% higher than the best non-graph competitor (Self-Attention, 0.6458). On the other balanced metrics, our model also performs well, with a precision of 0.6778 and a recall of 0.6273. These results reveal that our model not only makes more correct predictions but also misses fewer true cases. The area under the ROC curve rises to 0.8961, giving an extra safety margin of roughly 0.14 over the strongest baseline. Such consistent improvement across five diferent measures shows that modeling the relations between modalities as a graph network helps the network capture complementary information and suppress noise that plain concatenation, attention, or gating cannot handle. In other words, the graph lets the model learn both local and global context, while still keeping the information flow simple and straightforward, which yields better generalization. Therefore, our model maintains a good balance between precision and recall and produces more stable predictions than the other fusion schemes.

## 5. Discussion

Efectiveness and Clinical Relevance. The experimental results show that our measurement solution outperforms the compared methods on most of the key evaluation metrics, highlighting the efectiveness of the proposed framework in malocclusion skeletal grading. The model’s ability to capture both global and structural relationships ofers an improvement over conventional CNN-based and transformer-based approaches used in the field [82, 83, 84]. The incorporation of multi-scale adversarial perturbation and topological calibration further enhances robustness and generalizability. These advantages align with recent trends in multimodal orthodontic AI, such as the multiphase processing methods advocated by Bardideh et al. [85], reinforcing the value of combining diverse data sources for complex clinical tasks. Beyond these technical gains, the proposed system is also of practical clinical relevance. Designed as a decision-support tool rather than an autonomous diagnostic system, it predicts the skeletal class directly from the CBCT-reconstructed lateral cephalogram and can thus serve as a fast second reader that reduces manual measurement time and the inter-operator variability commonly seen in cephalometric analysis. Such a role is particularly useful in primary-care or community settings where orthodontic specialists are not always available, as the system can support early screening and timely referral, promote consistent diagnosis and treatment planning across operators, and aid the longitudinal monitoring of patients during treatment.

Table 1: Comparison of experimental results against existing and state-of-the-art methods.
<table><tr><td>Models</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td><td>AUC</td></tr><tr><td>ResNet18 [56]</td><td>0.6667</td><td>0.4542</td><td>0.4103</td><td>0.4311</td><td>0.7949</td></tr><tr><td>ResNet50 [56]</td><td>0.7500</td><td>0.5506</td><td>0.5697</td><td>0.5600</td><td>0.8681</td></tr><tr><td>MobilenetV2 [59]</td><td>0.6250</td><td>0.4194</td><td>0.4325</td><td>0.4258</td><td>0.8016</td></tr><tr><td>EfficientNet [58]</td><td>0.6458</td><td>0.4520</td><td>0.4333</td><td>0.4425</td><td>0.8108</td></tr><tr><td>RegNet [57]</td><td>0.6667</td><td>0.4542</td><td>0.4103</td><td>0.4311</td><td>0.7460</td></tr><tr><td>DenseNet121 [61]</td><td>0.6875</td><td>0.4821</td><td>0.4533</td><td>0.4673</td><td>0.7803</td></tr><tr><td>ShuffleNet [60]</td><td>0.6875</td><td>0.5783</td><td>0.5451</td><td>0.5612</td><td>0.7799</td></tr><tr><td>ConvNeXtV2 [62]</td><td>0.5625</td><td>0.4988</td><td>0.4501</td><td>0.4732</td><td>0.6183</td></tr><tr><td>ViT [63]</td><td>0.5625</td><td>0.3900</td><td>0.4098</td><td>0.3997</td><td>0.6986</td></tr><tr><td>Swin [64]</td><td>0.6667</td><td>0.5790</td><td>0.4380</td><td>0.4987</td><td>0.7894</td></tr><tr><td>MaxVit [66]</td><td>0.6458</td><td>0.5088</td><td>0.3984</td><td>0.4469</td><td>0.7079</td></tr><tr><td>HiViT [67]</td><td>0.5000</td><td>0.3163</td><td>0.3284</td><td>0.3222</td><td>0.5134</td></tr><tr><td>RIFormer [65]</td><td>0.7292</td><td>0.5417</td><td>0.4845</td><td>0.5115</td><td>0.7082</td></tr><tr><td>EfficientFormer [68]</td><td>0.6875</td><td>0.4617</td><td>0.4533</td><td>0.4575</td><td>0.7854</td></tr><tr><td>EVA-02 (Finetuned) [70]</td><td>0.7083</td><td>0.5931</td><td>0.4982</td><td>0.5415</td><td>0.7223</td></tr><tr><td>DINOv2 (Finetuned) [69]</td><td>0.5455</td><td>0.4211</td><td>0.3385</td><td>0.3753</td><td>0.6821</td></tr><tr><td>VMamba [71]</td><td>0.7083</td><td>0.3946</td><td>0.3803</td><td>0.3873</td><td>0.6847</td></tr><tr><td>VisionGNN [72]</td><td>0.6667</td><td>0.4548</td><td>0.4453</td><td>0.4500</td><td>0.7385</td></tr><tr><td>GAT [17]</td><td>0.6042</td><td>0.5595</td><td>0.5948</td><td>0.5766</td><td>0.7521</td></tr><tr><td>GCN [73]</td><td>0.6875</td><td>0.6670</td><td>0.6097</td><td>0.6371</td><td>0.8287</td></tr><tr><td>GIN [74]</td><td>0.6667</td><td>0.5944</td><td>0.6260</td><td>0.6098</td><td>0.7791</td></tr><tr><td>TeethGNN (Ours)</td><td>0.7708</td><td>0.6778</td><td>0.6273</td><td>0.6516</td><td>0.8961</td></tr></table>

![](images/d122ab12c3d076cf4a4e469fd7dd292d177bbd4614e2a3094933df5e6a88d24b.jpg)  
Figure 3: Examples of four malocclusion categories. The orange lines show the position of the ANB angle.

Limitations and Radiation Considerations. Despite its promising performance, several limitations remain. First, the clinical cost of an error depends on which classes are confused. Confusions within the same skeletal type, such as Class II Division 1 versus Division 2, share the same anteroposterior relationship and have a limited impact on the overall treatment direction. In contrast, confusions across types, especially those involving Class III, are more consequential because they may correspond to diferent strategies such as growth modification, camouflage treatment, or orthognathic surgery. To mitigate this, the framework keeps a clinician in the loop, and the proposed calibration improves the reliability of the predicted confidence so that low-confidence or borderline cases can be flagged for expert review rather than acted upon directly.

Table 2: Ablation study on the calibration components. We (i) remove the Multi-scale Graph Adversarial Perturbation (MGAP), and (ii) remove the Nonlinear Topological Graph Calibration (NTGC). ECE and Brier score are reported as calibration metrics.
<table><tr><td>MGAP</td><td>NTGC</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td><td>AUC</td><td>ECE</td><td>Brier Score</td></tr><tr><td>X</td><td>X</td><td>0.7500</td><td>0.6051</td><td>0.5743</td><td>0.5892</td><td>0.8653</td><td>0.1051</td><td>0.4910</td></tr><tr><td>√</td><td>X</td><td>0.7292</td><td>0.6375</td><td>0.5010</td><td>0.5611</td><td>0.8916</td><td>0.2123</td><td>0.5098</td></tr><tr><td>X</td><td>√</td><td>0.7708</td><td>0.6368</td><td>0.6365</td><td>0.6366</td><td>0.8732</td><td>0.2058</td><td>0.5080</td></tr><tr><td>√</td><td>√</td><td>0.7708</td><td>0.6778</td><td>0.6273</td><td>0.6516</td><td>0.8961</td><td>0.0910</td><td>0.4987</td></tr></table>

Table 3: Ablation study on the loss ratio α.
<table><tr><td>MODEL</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td><td>AUC</td></tr><tr><td> $\alpha = 0 . 0 1$ </td><td>0.7708</td><td>0.6778</td><td>0.6273</td><td>0.6516</td><td>0.8961</td></tr><tr><td> $\alpha = 0 . 0 5$ </td><td>0.8125</td><td>0.6033</td><td>0.5944</td><td>0.5988</td><td>0.8843</td></tr><tr><td> $\alpha = 0 . 1$ </td><td>0.7500</td><td>0.6839</td><td>0.5936</td><td>0.6356</td><td>0.8951</td></tr><tr><td> $\alpha = 0 . 5$ </td><td>0.7917</td><td>0.7349</td><td>0.6300</td><td>0.6784</td><td>0.8786</td></tr><tr><td> $\alpha = 1$ </td><td>0.7708</td><td>0.8178</td><td>0.6385</td><td>0.7171</td><td>0.8403</td></tr></table>

Table 4: Ablation study on the scale factor s in the graph adversarial perturbation.
<table><tr><td>MODEL</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td><td>AUC</td></tr><tr><td> $s = 1$ </td><td>0.7292</td><td>0.7632</td><td>0.6000</td><td>0.6718</td><td>0.8472</td></tr><tr><td> $s = 2$ </td><td>0.7500</td><td>0.7025</td><td>0.5771</td><td>0.6337</td><td>0.8841</td></tr><tr><td> $s = 3$ </td><td>0.7708</td><td>0.6778</td><td>0.6273</td><td>0.6516</td><td>0.8961</td></tr><tr><td> $s = 4$ </td><td>0.7708</td><td>0.5981</td><td>0.5559</td><td>0.5762</td><td>0.8847</td></tr><tr><td> $s = 5$ </td><td>0.7292</td><td>0.6750</td><td>0.5651</td><td>0.6152</td><td>0.8744</td></tr></table>

Second, most errors arise in clinically ambiguous situations. These include borderline cases in which the morphological indicators lie close to the decision thresholds, such as an ANB value near the Class I/Class II or Class II/Class III boundary, the subtle distinction between Class II Division 1 and Division 2 that relies on incisor inclination, and classes with fewer training samples. Notably, such cases are also frequent sources of disagreement among human experts. For this reason, TeethGNN is positioned to complement rather than replace specialist judgment. This view is consistent with broader AI-assisted orthodontic research, where Vaughan et al. [86] reported only fair agreement between an AI tool and clinicians in evaluating malocclusions, a result that reflects the intrinsic dificulty of these cases and the variability among experts themselves as much as the behavior of any single system. Accordingly, the system is intended to support clinical decision-making and to highlight cases that merit closer expert attention.

Table 5: Ablation study on the fusion strategy. ‘Attn’ denotes Attention.
<table><tr><td>FUSION</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td><td>AUC</td></tr><tr><td>Concat [75]</td><td>0.6250</td><td>0.5724</td><td>0.5964</td><td>0.5842</td><td>0.7279</td></tr><tr><td>JCA [76]</td><td>0.5833</td><td>0.4702</td><td>0.4817</td><td>0.4759</td><td>0.7097</td></tr><tr><td>MAT [77]</td><td>0.5833</td><td>0.5491</td><td>0.4817</td><td>0.5132</td><td>0.7501</td></tr><tr><td>Gated [78]</td><td>0.5625</td><td>0.5218</td><td>0.5651</td><td>0.5426</td><td>0.7077</td></tr><tr><td>Self-Attn [79]</td><td>0.6458</td><td>0.5426</td><td>0.5203</td><td>0.5312</td><td>0.7565</td></tr><tr><td>Guided-Attn [80]</td><td>0.5625</td><td>0.4722</td><td>0.4801</td><td>0.4761</td><td>0.7478</td></tr><tr><td>Co-Attn (Bi) [80]</td><td>0.5625</td><td>0.4668</td><td>0.4621</td><td>0.4644</td><td>0.7153</td></tr><tr><td>Co-Attn (V2T) [80]</td><td>0.5417</td><td>0.4631</td><td>0.4950</td><td>0.4785</td><td>0.7263</td></tr><tr><td>Co-Attn (T2V) [80]</td><td>0.6042</td><td>0.4875</td><td>0.4297</td><td>0.4568</td><td>0.6238</td></tr><tr><td>CAT-ViL [81]</td><td>0.5833</td><td>0.5249</td><td>0.5070</td><td>0.5158</td><td>0.7344</td></tr><tr><td>Ours</td><td>0.7708</td><td>0.6778</td><td>0.6273</td><td>0.6516</td><td>0.8961</td></tr></table>

In addition, while TeethGNN leverages graph-based integration of image and morphological features, it has not yet incorporated dynamic functional parameters such as bite force or soft-tissue dynamics, which are important aspects of comprehensive diagnosis. Given the high variability and physiological complexity inherent in orthodontic conditions, integrating such multimodal data is itself challenging. As emphasized by Farook et al. [87], the standardization of digital occlusion assessment remains dificult due to diferences in expert philosophy and the multifactorial nature of occlusal dysfunction. Moreover, this study is based on a single-center dataset, which inevitably limits the diversity of imaging devices, acquisition protocols, and patient populations, so the cross-center generalizability of TeethGNN cannot yet be fully guaranteed. We therefore interpret our results as evidence of feasibility and internal validity on a single-center cohort rather than a claim of broad generalizability, and we regard external validation on independent cohorts as an essential next step toward clinical translation.

Lastly, radiation exposure is another important factor when applying CBCT-based systems in orthodontic diagnosis, and this concern is greater in growing patients and in cases that require repeated follow-up examinations, because the biological efects of ionizing radiation can accumulate over time. From this point of view, radiation-free 3D imaging is attractive, as it allows clinicians to perform more frequent and repeatable examinations with lower biological risk, which is useful for monitoring treatment progress and growth changes. Radiation-free 3D techniques, such as 3D facial scanning, have shown value for soft-tissue evaluation and longitudinal monitoring, and have been proposed as an alternative to CBCT volumetric rendering for soft-tissue assessment [88]. However, the present study targets skeletal malocclusion grading, which depends on craniofacial skeletal relationships and on morphological indicators such as ANB and MP-FH. These indicators are defined on bony landmarks and cannot yet be fully derived from facial surface scans, because surface scans capture soft-tissue shape rather than the underlying skeletal structure. For this reason, radiation-free 3D imaging cannot yet replace CBCT-derived skeletal information for our current task, although it remains promising for soft-tissue evaluation and follow-up. We therefore regard TeethGNN as a decisionsupport tool for cases in which CBCT is already clinically indicated, rather than as a reason to acquire additional radiation.

Future Works. Future works will proceed along several directions. We will enlarge the dataset and improve its diversity and class balance to strengthen generalizability, particularly for rare classes and borderline cases. In particular, we will prioritize external validation on independent, multi-center cohorts spanning diferent institutions, CBCT devices, and patient populations, and explore federated learning to enable cross-institutional training and validation without centralizing patient data. We will also explore efective ways to embed additional modalities, such as biomechanical signals, soft-tissue dynamics, and electronic health records, into the graph learning framework to support more comprehensive diagnosis. In particular, we will combine the strengths of diferent imaging sources, so that CBCT can provide accurate skeletal information when it is clinically needed, while radiation-free 3D facial and intraoral scans can support soft-tissue analysis and frequent follow-up. Integrating CBCT with radiation-free 3D imaging within the same graph-based framework may reduce the dependence on ionizing radiation, allow more frequent and safer monitoring, and still maintain reliable diagnostic and prognostic performance. Finally, we plan to explicitly model boundary uncertainty and to strengthen human-in-the-loop mechanisms that account for the subjectivity and variability of clinical standards, so that the system can ofer more reliable and clinically trustworthy decision support.

## 6. Conclusion

This paper presents a novel vision-based medical measurement system for malocclusion skeletal grading. We develop TeethGNN, which integrates morphological information and CBCT image features using a graph-based approach. The proposed framework incorporates two key modules: explicit multi-scale graph adversarial perturbation and implicit nonlinear topological graph calibration, which enhance model robustness, calibration, and diagnostic accuracy. Extensive experiments and ablation studies demonstrate the efectiveness of our system and its individual components, achieving strong overall performance compared with state-of-the-art methods. Future work will focus on expanding the dataset to improve generalizability and exploring advanced graph-based strategies to further enhance the multimodal information integration. Additionally, eforts will be made to adapt the measurement system for broader applications in medical imaging and automatic diagnosis.

## Author Contribution Statement

Zhichun Jin: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Visualization, Writing – original draft. Zhicheng He: Conceptualization, Data curation, Formal analysis, Methodology, Software, Validation, Writing – original draft. Hao Xu: Data curation, Visualization, Software, Writing – review & editing. Dongyang Li: Data curation, Visualization. Lin Wang: Conceptualization, Funding acquisition, Project administration, Resources, Supervision, Writing – review & editing. Hongliang Ren: Conceptualization, Project administration, Resources, Supervision, Writing – review & editing. Long Bai: Conceptualization, Investigation, Methodology, Project administration, Resources, Supervision, Writing – original draft.

## Acknowledgment

This work was supported by the Postgraduate Research & Practice Innovation Program 597 of Jiangsu Province (KYCX22\_1801), and Natural Science Foundation of Jiangsu Province (BK20250581).

## References

[1] S. Campbell, G. Goldstein, Angle’s classification–a prosthodontic consideration: Best evidence consensus statement, Journal of Prosthodontics 30 (2021) 67–71.

[2] I. L. Nielsen, A comprehensive diagnostic system for orthodontists-beyond angle’s classification, Taiwanese Journal of Orthodontics 31 (2019) 3.

[3] D. J. Rinchuse, D. J. Rinchuse, Ambiguities of angle’s classification, The Angle Orthodontist 59 (1989) 295– 298.

[4] D. Wastell, J. Johnson, J. Jones, N. Bennett, Orthodontic analysis and treatment planning: a suite of programs for performing centroid cephalometrics, Computer Methods and Programs in Biomedicine 26 (1988) 259–265.

[5] H. Kim, E. Shim, J. Park, Y.-J. Kim, U. Lee, Y. Kim, Web-based fully automated cephalometric analysis by deep learning, Computer methods and programs in biomedicine 194 (2020) 105513.

[6] D. K. Hardy, Y. P. Cubas, M. F. Orellana, Prevalence of angle class III malocclusion: A systematic review and meta-analysis, Open Journal of Epidemiology 2 (2012) 75–82. doi:10.4236/ojepi.2012.24012.

[7] R. S. Gracea, N. Winderickx, M. Vanheers, J. Hendrickx, F. Preda, S. Shujaat, M. C. de Llano Perula, R. Jacobs, Artificial intelligence for orthodontic diagnosis and treatment planning: A scoping review, Journal of Dentistry (2024) 105442.

[8] L. Bai, M. Islam, L. Seenivasan, H. Ren, Surgicalvqla:transformer with gated vision-language embedding for visual question localized-answering in robotic surgery, in: 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 6859–6865.

[9] T. Sellers, T. Lei, C. Luo, Z. Bi, G. E. Jan, Human autonomy teaming-based safety-aware navigation through bioinspired and graph-based algorithms, Biomimetic Intelligence and Robotics 4 (2024) 100189.

[10] S. Zhao, L. Bai, K. Yuan, F. Li, J. Yu, W. Dong, G. Wang, M. I. Hoque, N. Padoy, N. Navab, et al., Rethinking data imbalance in class incremental surgical instrument segmentation, Medical Image Analysis 105 (2025) 103728.

[11] L. Schneider, L. Arsiwala-Scheppach, J. Krois, H. Meyer-Lückel, K. Bressem, S. Niehues, F. Schwendicke, Benchmarking deep learning models for tooth structure segmentation, Journal of dental research 101 (2022) 1343–1349.

[12] G. Chandrashekar, S. AlQarni, E. E. Bumann, Y. Lee, Collaborative deep learning model for tooth segmentation and identification using panoramic radiographs, Computers in Biology and Medicine 148 (2022) 105829.

[13] J.-H. Lee, D.-h. Kim, S.-N. Jeong, S.-H. Choi, Diagnosis and prediction of periodontally compromised teeth using a deep learning-based convolutional neural network algorithm, Journal of periodontal & implant science 48 (2018) 114–123.

[14] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, G. Monfardini, The graph neural network model, IEEE transactions on neural networks 20 (2008) 61–80.

[15] X. Diao, W. Chi, J. Wang, Graph neural network based method for robot path planning, Biomimetic Intelligence and Robotics 4 (2024) 100147.

[16] T. Lei, T. Sellers, C. Luo, D. W. Carruth, Z. Bi, Graphbased robot optimal path planning with bio-inspired algorithms, Biomimetic Intelligence and Robotics 3 (2023) 100119.

[17] P. Velickovi ˇ c, G. Cucurull, A. Casanova, A. Romero, ´ P. Lio, Y. Bengio, Graph attention networks, arXiv preprint arXiv:1710.10903 (2017).

[18] L. Bai, B. Ma, R. Wang, G. Wang, B. Cui, Z. Jiang, M. Islam, Z. Min, J. Lai, N. Navab, et al., Multimodal graph representation learning for robust surgical workflow recognition with adversarial feature disentanglement, Information Fusion 123 (2025) 103290.

[19] M. Wang, H. Yang, Q. Cheng, Gcl: Graph calibration loss for trustworthy graph neural network, in: Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 988–996.

[20] K. Yuan, T. Chen, S. Li, J. L. Lavanchy, C. Heiliger, E. Özsoy, Y. Huang, L. Bai, N. Navab, V. Srivastav, et al., Recognizing surgical phases anywhere: Few-shot testtime adaptation and task-graph guided refinement, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2025, pp. 467– 477.

[21] K. Kong, G. Li, M. Ding, Z. Wu, C. Zhu, B. Ghanem, G. Taylor, T. Goldstein, Robust optimization as data augmentation for large-scale graphs, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 60–69.

[22] X. Wang, H. Liu, C. Shi, C. Yang, Be confident! towards trustworthy graph neural networks via confidence calibration, Advances in Neural Information Processing Systems 34 (2021) 23768–23779.

[23] J. H. Park, Y.-J. Kim, J. Kim, J. Kim, I.-H. Kim, N. Kim, N. R. Vaid, Y.-A. Kook, Use of artificial intelligence to predict outcomes of nonextraction treatment of class ii malocclusions, in: Seminars in Orthodontics, volume 27, Elsevier, 2021, pp. 87–95.

[24] E. C. Olivetti, F. Marcolin, S. Moos, E. Vezzetti, C. Borbon, E. Zavattero, G. Ramieri, How to predict the future face? a 3d methodology to forecast the aspect of patients after orthognathic surgeries, Computer Methods and Programs in Biomedicine 265 (2025) 108757.

[25] I.-H. Kim, J.-S. Kim, J. Jeong, J.-W. Park, K. Park, J.- H. Cho, M. Hong, K.-H. Kang, M. Kim, S.-J. Kim, et al., Orthognathic surgical planning using graph cnn with dual embedding module: External validations with multi-hospital datasets, Computer Methods and Programs in Biomedicine 242 (2023) 107853.

[26] H. Han, C. Zhang, L. Zhu, P. Singh, R. T.-C. Hsung, Y. Y. Leung, T. Komura, W. Wang, M. Gu, Facial surgery preview based on the orthognathic treatment prediction, Computer Methods and Programs in Biomedicine (2025) 108781.

[27] Z. Yao, P. Wang, Y. Li, X. Yang, X. Zhou, Y. Wang, W. Xu, Y. Sun, Automatic angle’s classification based on the occlusal contact information, in: 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC), IEEE, 2022, pp. 761–767.

[28] I. Kim, D. Misra, L. Rodriguez, M. Gill, D. K. Liberton, K. Almpani, J. S. Lee, S. Antani, Malocclusion classification on 3d cone-beam ct craniofacial images using multi-channel deep learning models, in: 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), IEEE, 2020, pp. 1294–1298.

[29] A. A. Nayak, P. Venugopala, B. Ashwini, G. Padmashree, A study on ai applications for orthodontics and malocclusion detection approaches, in: 2024 IEEE International Conference on Distributed Computing, VLSI, Electrical Circuits and Robotics (DISCOVER), IEEE, 2024, pp. 273–278.

[30] M. Juneja, S. K. Saini, H. Kaur, P. Jindal, Application of convolutional neural networks for dentistry occlusion classification, Wireless Personal Communications 136 (2024) 1749–1767.

[31] F. A. N. M. Sabri, A. M. Ali, A. N. A. Abd Rahman, M. A. M. Zurin, A. S. A. Salam, N. A. C. Din, Classification of malocclusion using convolutional neural network and knowledge-based systems, in: 2023 IEEE 8th International Conference on Recent Advances and Innovations in Engineering (ICRAIE), IEEE, 2023, pp. 1–4.

[32] J.-N. Zhang, H.-P. Lu, J. Hou, Q. Wang, F.-Y. Yu, C. Zhong, C.-Y. Huang, S. Chen, Deep learning-based prediction of mandibular growth trend in children with anterior crossbite using cephalometric radiographs, BMC Oral Health 23 (2023) 28.

[33] Z. Jiao, Z. Liang, Q. Liao, S. Chen, H. Yang, G. Hong, H. Gui, Deep learning for automatic detection of cephalometric landmarks on lateral cephalometric radiographs using the mask region-based convolutional neural network: a pilot study, Oral Surgery, Oral Medicine, Oral Pathology and Oral Radiology 137 (2024) 554–562.

[34] S.-H. Han, J. Lim, J.-S. Kim, J.-H. Cho, M. Hong, M. Kim, S.-J. Kim, Y.-J. Kim, Y. H. Kim, S.-H. Lim, et al., Accuracy of posteroanterior cephalogram landmarks and measurements identification using a cascaded convolutional neural network algorithm: A multicenter study, Korean Journal of Orthodontics 54 (2024) 48.

[35] S. Takeda, Y. Mine, Y. Yoshimi, S. Ito, K. Tanimoto, T. Murayama, Landmark annotation and mandibular lateral deviation analysis of posteroanterior cephalograms using a convolutional neural network, Journal of Dental Sciences 16 (2021) 957–963.

[36] Y. Shimamura, C. Tachiki, K. Takahashi, S. Matsunaga, T. Takaki, M. Hagiwara, Y. Nishii, Accuracy of cephalometric landmark and cephalometric analysis from lateral facial photograph by using cnn-based algorithm, Scientific Reports 14 (2024) 31089. doi:10.1038/ s41598-024-82230-z.

[37] S. G. Paul, A. Saha, M. Z. Hasan, S. R. H. Noori, A. Moustafa, A systematic review of graph neural network in healthcare-based applications: Recent advances, trends, and future directions, IEEE Access 12 (2024) 15145–15170. doi:10.1109/ACCESS.2024.3354809.

[38] S. Mohammadi, Medical Image Analysis Based on Graph Machine Learning and Variational Methods, Ph.D. thesis,

Chapman University, Orange, CA, 2024. doi:10.36837/ chapman.000601.

[39] H. Zhang, C. Liu, P. Yang, S. Yang, Q. Yu, R. Liu, The concept of ai-assisted self-monitoring for skeletal malocclusion, Health Informatics Journal 30 (2024) 14604582241274511.

[40] S. Koval, B. A. Sutter, S. Koval, Occlusal characteristics in pre-orthodontic patients with deep overbite: An obser vational study, Advanced Dental Technologies & Techniques (2023) 1–8.

[41] M. M. Li, K. Huang, M. Zitnik, Graph representation learning in biomedicine and healthcare, Nature Biomedical Engineering 6 (2022) 1353–1369.

[42] H. Zhou, W. Wang, J. Jin, Z. Zheng, B. Zhou, Graph neural network for protein–protein interaction prediction: a comparative study, Molecules 27 (2022) 6135.

[43] K. Jha, S. Saha, H. Singh, Prediction of protein–protein interaction using graph neural networks, Scientific Reports 12 (2022) 8360.

[44] X. Wu, J. Duan, Y. Pan, M. Li, Medical knowledge graph: Data sources, construction, reasoning, and applications, Big Data Mining and Analytics 6 (2023) 201–217.

[45] C. Gao, S. Yin, H. Wang, Z. Wang, Z. Du, X. Li, Medicalknowledge-based graph neural network for medication combination prediction, IEEE Transactions on Neural Networks and Learning Systems 35 (2024) 13246–13257. doi:10.1109/TNNLS.2023.3266490.

[46] Y. Li, W. Zhao, B. Dang, X. Yan, M. Gao, W. Wang, M. Xiao, Research on adverse drug reaction prediction model combining knowledge graph embedding and deep learning, in: 2024 4th International Conference on Machine Learning and Intelligent Systems Engineering (MLISE), IEEE, 2024, pp. 322–329.

[47] J. Patel, R. Patel, Adverse drug reaction prediction: Graph neural networks and causal inference techniques, in: 2024 4th Interdisciplinary Conference on Electrics and Computer (INTCEC), IEEE, 2024, pp. 1–5.

[48] F. Zhang, B. Sun, X. Diao, W. Zhao, T. Shu, Prediction of adverse drug reactions based on knowledge graph embedding, BMC Medical Informatics and Decision Making 21 (2021) 1–11.

[49] M. Lotfy, A. Alperovich, T. Giannantonio, B. Barz, X. Zhang, F. Holm, N. Navab, F. Boehm, C. Schwamborn, T. K. Hofmann, et al., Robust tumor segmentation with hyperspectral imaging and graph neural networks, arXiv preprint arXiv:2311.11782 (2023).

[50] M. Ravinder, G. Saluja, S. Allabun, M. S. Alqahtani, M. Abbas, M. Othman, B. O. Soufiene, Enhanced brain tumor classification using graph convolutional neural network architecture, Scientific Reports 13 (2023) 14938.

[51] L. Zhang, Y. Zhao, T. Che, S. Li, X. Wang, Graph neural networks for image-guided disease diagnosis: A review, iRADIOLOGY 1 (2023) 151–166.

[52] X.-M. Zhang, L. Liang, L. Liu, M.-J. Tang, Graph neural networks and their current applications in bioinformatics, Frontiers in genetics 12 (2021) 690049.

[53] Y. Zheng, B. Chen, Y. Shen, K. Shen, Teethgnn: semantic 3d teeth segmentation with graph neural networks, IEEE Transactions on Visualization and Computer Graphics 29 (2022) 3158–3168.

[54] X. Li, M. Chung, L. Bi, J. Huang, Y. Pan, D. D. Feng, D. Chen, L. Jiang, B. Sheng, J. Kim, Teanet: Automated tooth extraction and arrangement with tooth-level graph spatial transformation network, Available at SSRN 4996981 (????).

[55] S. M. D. Thumati, K. Dhanya, H. Sathish, K. S. Madan, S. Rani, A comparative study on the working of gnn and cnn on panoramic x-rays in prediction of dental diseases, in: 2023 8th International Conference on Communication and Electronics Systems (ICCES), IEEE, 2023, pp. 755– 762.

[56] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[57] J. Xu, Y. Pan, X. Pan, S. Hoi, Z. Yi, Z. Xu, Regnet: Self-regulated network for image classification, IEEE Transactions on Neural Networks and Learning Systems 34 (2022) 9562–9567.

[58] M. Tan, Q. Le, Eficientnet: Rethinking model scaling for convolutional neural networks, in: International conference on machine learning, PMLR, 2019, pp. 6105–6114.

[59] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, Mobilenetv2: Inverted residuals and linear bottlenecks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4510– 4520.

[60] X. Zhang, X. Zhou, M. Lin, J. Sun, Shuflenet: An extremely eficient convolutional neural network for mobile devices, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 6848– 6856.

[61] G. Huang, Z. Liu, L. Van Der Maaten, K. Q. Weinberger, Densely connected convolutional networks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 4700–4708.

[62] S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, S. Xie, Convnext v2: Co-designing and scaling convnets with masked autoencoders, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 16133–16142.

[63] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An image is worth 16x16 words: Transformers for image recognition at scale, in: International Conference on Learning Representations, 2021.

[64] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo, Swin transformer: Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10012–10022.

[65] J. Wang, S. Zhang, Y. Liu, T. Wu, Y. Yang, X. Liu, K. Chen, P. Luo, D. Lin, Riformer: Keep your vision backbone efective while removing token mixer, arXiv preprint arXiv:2304.05659 (2023).

[66] Z. Tu, H. Talebi, H. Zhang, F. Yang, P. Milanfar, A. Bovik, Y. Li, Maxvit: Multi-axis vision transformer, in: European conference on computer vision, Springer, 2022, pp. 459–479.

[67] X. Zhang, Y. Tian, W. Huang, Q. Ye, Q. Dai, L. Xie, Q. Tian, Hivit: Hierarchical vision transformer meets masked image modeling, arXiv preprint arXiv:2205.14949 (2022).

[68] Y. Li, G. Yuan, Y. Wen, J. Hu, G. Evangelidis, S. Tulyakov, Y. Wang, J. Ren, Eficientformer: Vision transformers at mobilenet speed, Advances in Neural Information Processing Systems 35 (2022) 12934–12949.

[69] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., Dinov2: Learning robust visual features without supervision, arXiv preprint arXiv:2304.07193 (2023).

[70] Y. Fang, Q. Sun, X. Wang, T. Huang, X. Wang, Y. Cao, Eva-02: A visual representation for neon genesis, Image and Vision Computing 149 (2024) 105171.

[71] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, J. Jiao, Y. Liu, Vmamba: Visual state space model, Advances in neural information processing systems 37 (2025) 103031–103063.

[72] K. Han, Y. Wang, J. Guo, Y. Tang, E. Wu, Vision gnn: An image is worth graph of nodes, Advances in neural information processing systems 35 (2022) 8291–8303.

[73] T. N. Kipf, M. Welling, Semi-supervised classification with graph convolutional networks, arXiv preprint arXiv:1609.02907 (2016).

[74] K. Xu, W. Hu, J. Leskovec, S. Jegelka, How powerful are graph neural networks?, arXiv preprint arXiv:1810.00826 (2018).

[75] L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, K.-W. Chang, Visualbert: A simple and performant baseline for vision and language, arXiv preprint arXiv:1908.03557 (2019).

[76] R. G. Praveen, W. C. de Melo, N. Ullah, H. Aslam, O. Zeeshan, T. Denorme, M. Pedersoli, A. L. Koerich, S. Bacon, P. Cardinal, et al., A joint cross-attention model for audio-visual fusion in dimensional emotion recognition, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 2486– 2495.

[77] Z. Wu, L. Liu, Y. Zhang, M. Mao, L. Lin, G. Li, Multimodal crowd counting with mutual attention transformers, in: 2022 IEEE International Conference on Multimedia and Expo (ICME), IEEE, 2022, pp. 1–6.

[78] J. Arevalo, T. Solorio, M. Montes-y Gómez, F. A. González, Gated multimodal units for information fusion, arXiv preprint arXiv:1702.01992 (2017).

[79] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, Advances in neural information processing systems 30 (2017) 5998–6008.

[80] Z. Yu, J. Yu, Y. Cui, D. Tao, Q. Tian, Deep modular coattention networks for visual question answering, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 6281–6290.

[81] L. Bai, M. Islam, H. Ren, Cat-vil: Co-attention gated vision-language embedding for visual question localizedanswering in robotic surgery, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2023, pp. 397–407.

[82] Y. M. Bichu, B. Zou, P. K. Chaudhari, S. M. Adel, N. Vaiid, Artificial intelligence applications in orthodontics, in: Artificial Intelligence for Oral Health Care: Applications and Future Prospects, Springer, 2025, pp. 81– 97.

[83] R. S. Gracea, N. Winderickx, M. Vanheers, J. Hendrickx, F. Preda, S. Shujaat, M. C. de Llano-Pérula, R. Jacobs, Artificial intelligence for orthodontic diagnosis and treatment planning: A scoping review, Journal of Dentistry 152 (2025) 105442.

[84] S. Sahim, M. Boutissante, F. El Quars, Applications of artificial intelligence and machine learning for orthodontic diagnosis, in: Cranio-Maxillofacial Surgery – Orthognathic and Orthodontic Techniques, IntechOpen, 2025. doi:10.5772/intechopen.1009333.

[85] E. Bardideh, F. L. Alizadeh, M. Amiri, M. Ghorbani, Designing an artificial intelligence system for dental occlusion classification using intraoral photographs: A comparative analysis between artificial intelligence-based and clinical diagnoses, American Journal of Orthodontics and Dentofacial Orthopedics 166 (2024) 125–137.

[86] M. Vaughan, S. Mheissen, M. Cobourne, F. Ahmed, Diagnostic accuracy of artificial intelligence for dental and occlusal parameters using standardized clinical photographs, American Journal of Orthodontics and Dentofacial Orthopedics 167 (2025) 733–740. doi:10.1016/j.ajodo. 2025.01.017.

[87] T. H. Farook, F. Rashid, S. Ahmed, J. Dudley, Clinical machine learning in parafunctional and altered functional occlusion: A systematic review, The Journal of Prosthetic Dentistry 133 (2025) 124–128.

[88] G. Perrotti, R. Reda, O. Rossi, I. D’Apolito, T. Testori, L. Testarelli, et al., A radiation free alternative to cbct volumetric rendering for soft tissue evaluation, Brazilian Dental Science 26 (2023) e3726. doi:10.4322/bds. 2023.e3726.