# Enhancing Table Structure Recognition via Bounding Box Guidance

Lei Hu and Shuangping Huang<sup>⋆</sup>

South China University of Technology, Guangzhou, China eehulei@mail.scut.edu.cn, eehsp@scut.edu.cn

Abstract. Table Structure Recognition (TSR) aims to extract the bound ing boxes of cells and table structure (e.g., HTML) from table images. Although current approaches have made significant progress, the latest image-to-sequence methods overlook the explicit utilization of the bounding box information when predicting HTML sequences, leading to error predictions in complex scenes. In this paper, we introduce a novel framework BGTR (Bounding Box-Guided Table Recognizer). To more efectively utilize bounding box information, we first predict the bounding boxes of cells and then use this information to guide the generation of HTML sequences. While utilizing bounding box information can enhance the accuracy of HTML sequences, for natural scene tables, the data volume is too small to allow for suficient training of bbox-guided HTML generation. In response, we adopt a progressive training method for natural scene tables and introduce SNSTab, a synthetically generated natural scene table dataset. Our experiments on five benchmark datasets demonstrate SOTA performance.

Keywords: Table structure recognition · Image-to-sequence · Bounding box guidance · Dataset.

## 1 Introduction

Tables are a crucial medium for structured information dissemination. Table detection (TD) aims to extract the position of tables from document images, and many methods [1,8,22] have shown excellent results. Table Structure Recognition (TSR) aims to transform images containing tables into structured data, which is both crucial and challenging. Leveraging the advancements of transformer [25], which have proven highly efective in various fields [7], image-to-sequence methods [20,13,29,3] have demonstrated promising results in TSR. These methods employ an encoder-decoder architecture to simultaneously predict the HTML (Hyper Text Markup Language) sequence and the bounding box (bbox) of table cells. In predicting HTML sequences, they rely solely on image information and overlook the explicit utilization of bbox information. However, table structure and formatting can be highly complex, and bbox information is essential for parsing the structure of tables. Therefore, exclusive reliance on image information may lead to error predictions in complex scenes, like spanning cells (Fig. 1 (a)). In this paper, we introduce a novel framework BGTR (Bounding Box-Guided Table Recognizer). Unlike previous image-to-sequence methods [20,13,29,3], we explicitly utilize bbox information to obtain accurate HTML sequences. We first use a Bbox Predictor to predict bboxes. Then, during HTML sequence decoding, we enable the Bbox-Guided Structure Decoder to perceive both the image and bbox information of the table, resulting in accurate HTML sequences.

<sub>w/o</sub> <sub>bbox-guided</sub>   
![](images/9975f4e4af896192f34a22b883e967d0d52d253f0f8c0511e0c0b90583074401.jpg)

![](images/0bf46d8f424d76431e045764d2230c2034f455770e40082a4b17cd64a1a3df9c.jpg)

<table><tr><td colspan="6">Maximum time to record video and photo size</td></tr><tr><td>I Memory card(Gb)</td><td>VIEO(minutes)</td><td></td><td>PHOTO 3264*2448(8M)</td><td>2560*1920(5M)</td><td>1920*1080(2M)</td></tr><tr><td>32G</td><td>1920*1080P(FulI HD) 1280*720P(HD)</td><td>4000*3000(12M) 3648*2736(10M)</td><td>11056</td><td>13584</td><td></td></tr><tr><td>16G</td><td>368 640</td><td>9232</td><td>6792</td><td>21616</td><td>47552 23776</td></tr><tr><td>8G</td><td>184 320</td><td>4616 5528</td><td>3396</td><td>10808 5404</td><td>11888</td></tr><tr><td>4G</td><td>92 160 46 80</td><td>2308 1154</td><td>2764 1382 1698</td><td>2702</td><td>5944</td></tr></table>

(a)
<table><tr><td colspan="4">Digital Document Tables</td></tr><tr><td>Dataset</td><td>Samples</td><td>Datasets</td><td>Samples</td></tr><tr><td>PubTabNet [33]</td><td>500K</td><td>FinTabNet [32]</td><td>100K</td></tr><tr><td>SynthTabNet [20]</td><td>600K</td><td>PubTables-1M [23]</td><td>758K</td></tr><tr><td colspan="4">Natural Scene Tables</td></tr><tr><td>TabRecSet [28]</td><td>38K</td><td>WTW [17]</td><td>14K</td></tr><tr><td>iFLYTAB [30]</td><td>17K</td><td>TAL [5]</td><td>15K</td></tr></table>

(b)  
Fig. 1: The motivation behind the proposed method. (a) Comparison of HTML visualization with and without bbox-guided generation on TabRecSet [28], the red dotted boxes indicate error results, the blue dotted boxes indicate spanning cells. The results indicate that using bboxes for guidance achieves better performance in spanning cells. (b) A comparison between two types of datasets reveals that the sample size of the digital document table dataset is significantly larger than that of the natural scene table dataset.

Although utilizing bbox information can improve the accuracy of the HTML sequence, for natural scene tables, as shown in Fig. 1 (b), the data volume is small, and the structure and style of tables in natural scenes are complex, making it insuficient for adequate training of bbox-guided HTML generation. In response, we adopt a progressive training method for natural scene tables and introduce SNSTab. Progressive training method includes a foundation training stage and a advancement training stage. In the foundation training stage, we aim to train the model with a large number of tables from natural scenes, thereby enabling it to learn how to more efectively utilize bbox information for guiding the generation of HTML sequences, this approach leads to improve the model’s foundational understanding of tables. In the advancement training stage, training is conducted on a specific natural scene table dataset (e.g., TabRecSet [28] and iFLYTAB [30]). SNSTab is a synthetically generated natural scene table dataset containing 500k table images for the foundation training stage. It includes wired tables, wireless tables, inclined tables, and curved tables, featuring diverse table structures and backgrounds. This variety enables the model to comprehensively learn various aspects of table knowledge during the foundation training stage, thereby achieving better results in complex scenes like spanning cells and deformed tables, as shown in Fig. 7.

Extensive experiments demonstrate the efectiveness of our proposed BGTR and the progressive training method, achieving state-of-the-art performance on five public benchmarks.

To sum up, our contributions are as follows:

– We propose BGTR, a novel framework that explicitly utilizes bbox information for guiding HTML sequence generation, which aims at enhancing structural recognition accuracy in challenging table scenes.

To ensure that bbox-guided HTML generation is adequately trained in natural scenes, we adopt a progressive training method and introduce SNSTab, a synthetically generated natural scene table dataset for the foundation training stage.

Our experiments on five benchmark datasets demonstrate state-of-the-art performance.

## 2 Related Work

## 2.1 Table Structure Recognition

With the rapid development of deep learning, a variety of table structure recognition methods have emerged, which can be divided into three categories: graphbased methods, split-and-merge methods, and image-to-sequence methods.

Graph-based Methods. These methods utilize cells or text boxes as the basic elements of the table, employing a graph network to determine the row and column relationships between them. GraphTSR [4] utilized graph attention networks to the TSR task, determining the row and column relationships of adjacent cells through graph edge classification. TabStruct-Net [21] implemented a unified end-to-end framework for cell detection and cell relationship analysis. GFTE [14] employed a graph-based convolutional network that integrates image features, position features, and textual features to predict relationships between cells. NCGM [16] enabled cooperation among geometry, appearance, and content modalities, leveraging their interaction to enhance multi-modal representation in intricate situations. However, these methods are limited by their reliance on additional bbox data or OCR accuracy, leading to potential errors in table structure recognition, and additionally, they need complex post-processing methods.

Split-and-merge Methods. Typically, these methods comprise two models: the split model and the merge model. The split model initially detects the row and column regions of the table and then intersects them to obtain the grid cells of the table. Subsequently, the merge model is employed to determine which adjacent grid cells need to be merged. SPLERGE [24] became the first to use the split-and-merge framework for the TSR task, addressing an issue where previous methods struggled with resolving spanning cells. By utilizing textual information, SEM [31] achieved enhanced results on complex tables with spanning cells. To address geometric distortion in table images, TSRFormer [15] approached the detection of row and column regions as a linear regression problem. However, two-stage training can be complex and resource-intensive, potentially leading to longer training times and dificulties in optimization compared to more streamlined, end-to-end methods.

Image-to-sequence Methods. These methods treat the table as a structured sequence (e.g., HTML or LATEX), using an encoder-decoder framework to convert the table image into a structured sequence that fully describes the table structure. EDD [33] employed a CNN-based encoder to extract the visual features from table images and utilized two LSTM-based decoders to simultaneously recognize the table structure and cell content. TableMaster [29] introduced a transformer-based [25] architecture, achieving significant progress in the TSR task by recognizing the table structure and cell bboxes simultaneously. Based on TableMaster [29], VAST [13] treated bbox prediction as a coordinate sequence generation task and introduced a visual-alignment loss that significantly improved bbox accuracy. However, bbox information is essential for parsing the structure of table, unlike previous methods that produce inaccurate HTML sequence predictions in complex table scenes due to the lack of bbox information, this paper utilizes bbox information to guide the generation of HTML sequences, resulting in more accurate HTML sequences.

## 2.2 Existing Datasets

While the size of table datasets has significantly increased, existing datasets primarily focus on digital documents [20,33,32,23], such as PDF files. Building a digital document table dataset is relatively straightforward because annotated information can be directly extracted from PDF files. However, tables captured in natural scenes through cameras cannot be automatically annotated, and manual annotation is a time-consuming process. Additionally, natural scene tables are more complex, often inclined, rotated, and curved, further increasing the annotation dificulty. Due to these challenges, there is a substantial disparity in the number of table datasets between natural scenes and digital documents, as illustrated in Fig. 1 (b). To address this issue, we propose a large-scale synthetically generated natural scene table dataset.

![](images/3723a671868728fa4aa8ce6747677a23d9c5c4fdfb2acc4ed33b73804a4df78f.jpg)  
Fig. 2: The production process of SNSTab: (a) table generation. (b) table transformation. (c) background synthesis.

## 3 SNSTab

SNSTab contains 500k synthetic images of natural scene tables, including wired tables, wireless tables, provincial line tables, inclined tables, curved tables and large tables. Although image generation has achieved significant success in other fields [6], its application in table recognition remains quite limited. To our knowledge, SNSTab is the first large-scale natural scene table synthesis dataset. SNSTab’s annotations contain the coordinates of the table cells, the text inside the cells, and the HTML sequence that describes the table structure. The creation of the SNSTab dataset involves three phases: table generation, table transformation, and background synthesis, as shown in Fig. 2.

Table generation. This step is to generate digital document table images. We randomly generate table images based on the open source tool Table Generation<sup>1</sup>. First, we will generate a grid with a random number of rows and columns; Then, we will randomly merge the adjacent grids to get spanning cells, and generate random text for each grid; Finally, we convert the above table into HTML sequences, and get the final table image through the browser rendering.

Table transformation. Since tables in the nature scene tend to be inclined or rotated. Therefore, after automatically generating tables, we apply thin plate spline (TPS) [2] to randomly transform them. This simulation captures the complexities observed in natural scenes. As shown in Fig. 2 (b), we take the four vertices of the table image and the midpoints of the four sides as the source points, the target points are then obtained by randomly moving the source points within the range of the red dotted line. After the TPS transformation, the coordinates of the cells are also transformed.

Background synthesis. In addition to their complex structures, tables in natural scenes often exhibit a variety of backgrounds. We captured 400 background images of natural scenes, including paper, walls, daily-life items, and more. For each table image, a random background image is first selected, and then a random area of the same size as the table image is extracted from the background image. Finally, the table image is merged with the selected background to produce the final image.

![](images/0b3aaa5fad19b32b0cfaec64907d6e8c207188c823bb804ffa0ea3e8cd020db5.jpg)  
Fig. 3: A simple example of using HTML sequences to represent a table structure, diferent colors in the figure represent diferent rows of the table.

For more details of the dataset and for additional dataset samples, please refer to the supplementary materials.

## 4 Method

## 4.1 Preliminary

In this paper, we utilize HTML sequences to represent the table structure, as shown in Fig. 3. Given a table image, our model outputs HTML sequences of the table and the corresponding cell bboxes. To better facilitate prediction, we tokenize HTML sequences into HTML tokens. For cells without spanning, nonempty cells and empty cells are denoted by $< t d > < / t d > \mathrm { a n d } < e m > <$ $/ e m > ,$ , respectively. In the case of spanning cells, the tokens are divided into three ${ \mathrm { p a r t s } } \colon < t d ,$ colspan = N or rowspan = N, and $> < / t d >$ . Here, $< t d$ indicates the beginning of the spanning cells, N specifies the count of cells that are spanning, and $> < / t d >$ marks the end of the spanning cells. $< t r >$ and $< / t r >$ respectively represent the beginning and the end of each row in a table. We use $H _ { N } = \{ h _ { i } \} _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times 1 }$ to denote HTML sequences, where N is the sequence length and $h _ { i }$ denotes the i-th HTML token. We use $B _ { N } = \{ b _ { j } \} _ { j = 1 } ^ { N } \in$ $\mathbb { R } ^ { N \times 4 }$ to denote the bbox of table cells. For each cell, its bbox is represented as $[ x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } ]$ , where $[ x _ { 1 } , y _ { 1 } ]$ denotes the coordinates of the top-left corner, and $[ x _ { 2 } , y _ { 2 } ]$ represents the coordinates of the bottom-right corner. Moreover, the HTML tokens have a one-to-one correspondence with the bboxes, and the bbox value is non-zero only if the HTML token $\mathrm { i s } < t d > < / t d > \mathrm { a n d } < t d .$

## 4.2 Overall Architecture

The overall framework of BGTR is illustrated in Fig. 4. Given a table image, denoted as $\mathbf { P } \in \mathbb { R } ^ { H \times W \times 3 }$ , where H and W represent the height and width of the image, respectively. We employ an Image Encoder to extract image features, resulting in the feature map $\mathbf { F } _ { e n c o d e r } \in \mathbb { R } ^ { \frac { H } { 8 } \times \frac { W } { 8 } \times d }$ , where d denotes the dimension of the features. After applying 2D positional encoding, the flattened image features are obtained as $\mathbf { F } _ { i m a g e } \in \mathbb { R } ^ { \frac { H W } { 6 4 } \times d }$ . The image features $\mathbf { F } _ { i m a g e }$ are further fed into the Shared Decoder for decoding, resulting in decoded features $\mathbf { F } _ { s h a r e } ^ { 1 } \in \mathbb { R } ^ { N \times d }$ . The Shared Decoder is used to reduce the gap between the image and the sequence, making it more aligned with the sequential features. $\mathbf { F } _ { s h a r e } ^ { 1 }$ is first fed into the Bbox Decoder to obtain bboxes $B _ { N }$ . Then, $\mathbf { F } _ { s h a r e } ^ { 1 }$ is sent into the Bbox-Guided Structure Decoder (Sec. 4.3), using the predicted bbox information to guide the generation of HTML sequences $H _ { N }$ . For additional details regarding the Image Encoder, Shared Decoder, and Bbox Decoder, please refer to Sec. 5.2.

![](images/67e96f68d759a41d26e06931d519fb551f1e05f5cbf8ea02f1f9954cb54d9da9.jpg)  
Fig. 4: Architecture of BGTR. The Bbox Predictor aims to acquire the bboxes of table cells and consists of three parts: an image encoder, a shared decoder, and a bbox decoder. The Bbox-Guided Structure Decoder generates the HTML sequence.

## 4.3 Bbox-Guided Structure Decoder

To more efectively utilize bbox information, we first predict the bboxes of cells and then utilize the bbox information to enhance the accuracy of HTML sequence prediction. Since we employ an autoregressive decoding approach, we utilize parallel training methods during training to accelerate the training speed. Specifically, the Bbox-Guided Structure Decoder receives $\mathbf { F } _ { s h a r e } ^ { 1 } \in \mathbb { R } ^ { N \times d }$ from the Shared Decoder and $\mathbf { F } _ { b b o x } \in \mathbb { R } ^ { N \times d }$ from the Bbox Decoder as input. In the Bbox Decoder, after $\mathbf { F } _ { b b o x }$ passes through a linear layer and a sigmoid layer, the bboxes $B _ { N } \in \mathbb { R } ^ { N \times 4 }$ are obtained. $\mathbf { F } _ { s h a r e } ^ { 1 }$ initially passes through a masked self-attention layer, resulting in ${ \bf F } _ { s h a r e } ^ { 2 } \in \mathbb { R } ^ { N \times d }$ . Here, the mask refers to the prediction of the current time step HTML token being based on the output of previous time steps. Subsequently, $\mathbf { F } _ { s h a r e } ^ { 2 }$ and $\mathbf { F } _ { b b o x }$ are fed into a masked crossattention layer. Here, the mask indicates that the prediction of the current time step HTML token is based on the bbox outputs of both the current and previous time steps. $\mathbf { F } _ { s h a r e } ^ { 2 }$ serves as the query vector, while $\mathbf { F } _ { b b o x }$ serve as the key/value vectors. By utilizing the cross-attention mechanism, bbox information $\mathbf { F } _ { b b o x }$ becomes efectively integrated into $\mathbf { F } _ { s h a r e } ^ { 1 } .$ The use of $\mathbf { F } _ { b b o x }$ for decoding allows the model to comprehensively understand the position and relative relationships of each cell while predicting HTML sequences. This process allows the model to generate HTML sequences guided by bbox information. After passing through a linear layer and a softmax layer, the decoder’s output yields the final HTML sequences $H _ { N } = \{ h _ { i } \} _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times 1 }$

![](images/8108674b313c4e461b97fd4950196b6565d7b840faf06aab91a0a9c216682e94.jpg)  
Fig. 5: An overview of the progressive training method. (a) Foundation training stage training on SNSTab. (b) Advancement training stage training on natural scene table datasets (e.g., TabRecSet [28] and iFLYTAB [30]).

## 4.4 Progressive Training Method

As illustrated in Fig. 5, this section will discuss the implementation of the progressive training method.

Foundation training stage. In the foundation training stage, the purpose is for the model to acquire common knowledge about tables. Due to the diverse types and varied structures of tables in natural scenes, the foundation training stage requires a large number of data samples. Based on this, we introduce SNSTab, a large synthetic table dataset in natural scenes. For further details about SNSTab, please refer to Sec. 3 and the supplementary material. After completing the foundation training stage on SNSTab, the model develops a foundational capability for recognizing table structures in natural scenes. Additionally, it can learn how to efectively utilize bbox information to guide the generation of HTML sequences, particularly in complex scenes such as spanning cells and deformed tables (Fig. 7).

Advancement training stage. Building on the foundation training stage, the advancement training stage is conducted on a specific natural scene table dataset. With the common knowledge acquired in the foundation training stage, the model demonstrates improved convergence speed and enhanced overall training efectiveness in the advancement training stage. And in this stage, the Shared Decoder and the Bbox-Guided Structure Decoder are initialized using the training from the foundation training stage. Due to certain diferences in the data between the two stages, the Bbox decoder is trained from scratch.

Through the progressive training process, the issue of insuficient data leading to inadequate training of bbox-guided HTML generation in natural scenes has been significantly alleviated.

## 4.5 Loss Functions

Our model adopts an end-to-end training approach and includes two loss functions. For the Bbox Decoder, $L _ { 1 }$ loss is employed to supervise the prediction of bboxes, which is denoted as $\mathcal { L } _ { b b o x }$ . For Bbox-Guided Structure Decoder, crossentropy loss is utilized to supervise the prediction of HTML tokens, which is denoted as $\mathcal { L } _ { h t m l }$ . The final loss function is formulated as follows:

$$
\begin{array} { r } { \mathcal { L } = \lambda \mathcal { L } _ { h t m l } + \mathcal { L } _ { b b o x } , } \end{array}\tag{1}
$$

where λ is the hyperparameter.

## 5 Experiments

## 5.1 Datasets and Evaluation Metric

Datasets. Our method is evaluated on five popular public benchmarks, including TabRecSet [28], iFLYTAB [30], PubTabNet [33], FinTabNet [32] and SynthTabNet [20].

TabRecSet [28] is a natural scene table dataset featuring tables from diverse scenes with various forms. It has 32.07K images and 38.17K tables, the number of images is not equal to the number of tables because some images contain multiple tables. As TabRecSet did not provide a predefined split, we randomly divided the dataset into train and test splits(80%,20%), resulting in 30.6k training table images and 7.5k testing table images.

iFLYTAB [30] has 12,104 training samples and 5,187 testing samples. It contains both wired and wireless tables from natural scenes and digital documents.

PubTabNet [33] contains 500,777 training images and 9,115 validating images, each accompanied by annotation information detailing the table structure and text content along with their positions. All the tables are extracted from the scientific articles, and annotations are automatically obtained from the PDF source files.

FinTabNet [32] is a large-scale dataset containing 91596 training tables, 10,635 validating tables and 10,656 testing tables. All the tables are sourced from the annual reports of the S&P 500 companies. Following [32,20,13,19], we use validating sets for testing.

SynthTabNet [20] is a synthetically generated dataset with diverse table styles, complex structures, and an increased number of rows and columns. It contains 480k training images, 60k validating images, and 60k testing images. In addition to the bounding boxes of the non-empty cell, it also has the bounding boxes of the empty cell.

Table 1: Comparison with state-of-the-art methods. PT indicates that progressive training method is used. S indicates simple tables. C indicates complex tables. Bold indicates the best performance, while underline indicates the second-best performance. ⋆ indicates the image-to-sequence method. † means pre-training on PubTabNet [33].
<table><tr><td rowspan="3">Method</td><td colspan="2">PubTab</td><td colspan="2">FinTab SynthTab</td><td colspan="3">TabRecSet</td><td>iFLYTAB</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">TEDS-S TEDS TEDS-S</td><td rowspan="2">TEDS-S</td><td colspan="3">TEDS-S</td><td rowspan="2">TEDS-S</td></tr><tr><td>S</td><td>C</td><td>All</td></tr><tr><td>EDD  [33]</td><td>89.90</td><td>88.30</td><td>90.06</td><td>1</td><td>95.01</td><td>77.71</td><td>91.03†</td><td>一</td></tr><tr><td>GTE [32]</td><td>93.01</td><td></td><td>87.10</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TableMaster ★ [29]</td><td>96.04</td><td>96.16</td><td></td><td></td><td>97.20</td><td>84.11</td><td>94.14</td><td>84.63</td></tr><tr><td>SEM [31]</td><td>一</td><td>93.70</td><td>1</td><td>I</td><td>一</td><td></td><td></td><td>75.90</td></tr><tr><td>NCGM [16]</td><td></td><td>95.40</td><td></td><td></td><td>一</td><td></td><td></td><td>一</td></tr><tr><td>TableFormer ★ [20]</td><td>96.75</td><td>93.60</td><td>96.80</td><td>96.70</td><td></td><td></td><td></td><td></td></tr><tr><td>VAST ★ [13]</td><td>97.23</td><td>96.31</td><td>98.63</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GridFormer [19]</td><td>97.00</td><td>95.84</td><td>98.63</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SEMv2 [30]</td><td>97.50</td><td>=</td><td>一</td><td>一</td><td></td><td></td><td></td><td>92.00</td></tr><tr><td>TSRFormer [15]</td><td>97.50</td><td>一</td><td>一</td><td></td><td>一</td><td></td><td>一</td><td>一</td></tr><tr><td>BGTR ★</td><td>97.63</td><td>96.57</td><td>98.89</td><td>99.11</td><td>98.35</td><td>89.27</td><td>96.23</td><td>91.02</td></tr><tr><td>BGTR (PT) ★</td><td></td><td></td><td></td><td>一</td><td>98.65</td><td>92.47</td><td>97.21</td><td>92.00</td></tr></table>

Table 2: Comparison of cell bbox detection results on PubTabNet. PP indicates the post-processing.
<table><tr><td>Method</td><td>mAP</td><td>mAP(PP)</td></tr><tr><td>EDD + BBox [20]</td><td>79.2</td><td>82.7</td></tr><tr><td>TableFormer [20]</td><td>82.1</td><td>86.8</td></tr><tr><td>BGTR</td><td>91.9</td><td>-</td></tr></table>

Evaluation Metric. The Tree-Edit-Distance-based Similarity (TEDS) [33] is employed as the evaluation metric, treating tables as tree structures. To mitigate the impact of OCR errors on the final score, we also utilize TEDS-S to assess the accuracy of the table structure without the table content.

## 5.2 Implementation Details

In this paper, the experimental settings are as follows: the table images are resized to 480 × 480, and the flattened image sequence length is 3600. The dimension of the features d is 512. The multi-head number is 8. The maximum HTML sequence length is 500. We used Ranger [27] as the optimizer, the minibatch size is set to 8. For TabRecSet, PubTabNet, FinTabNet and SynthTabNet, we trained 25 epochs, the initial learning rate is established at 1e-3, and divided by 10 at 17 and 22 epochs. For iFLYTAB, we trained 120 epochs, the initial learning rate is established at 1e-3, and divided by 10 at 75 and 105 epochs. For the foundation training stage on SNSTab, we trained 3 epochs, the initial learning rate is established at 1e-3. Experiments are conducted using 2 NVIDIA GeForce RTX 3090 GPUs with 24GB of RAM memory.

We use the ResNet-50 [10] combined with the Multi-Aspect GCA [18] module and 2D positional encoding to form the Image Encoder. To enhance the model’s understanding of the 2D topology of table images, we employ 2D positional encoding to encode image features. The Shared Decoder comprises two identical stacked transformer [25,12,11] decoding layers. The Bbox Decoder comprises a single transformer [25] decoding layer. The Bbox-Guided Structure Decoder comprises two identical stacked transformer [25] decoding layers.

## 5.3 Comparison with Previous State-of-the-arts

As shown in Table 1, our method not only outperforms non-image-to-sequence methods, but also outperforms the best image-to-sequence method.

Results on natural scene tables. We evaluate the performance of our model on two natural scene table datasets: TabRecSet [28] and iFLYTAB [30]. Given the absence of a baseline method in TabRecSet, TableMaster [29] is adopted as the baseline. We divide the dataset into two categories: simple (S) and complex (C). A table is considered complex if it contains spanning cells, otherwise, it is classified as a simple table. On TabRecSet, a TEDS-S score of 98.65% for simple tables and 92.47% for complex tables is achieved. Compared with baseline TableMaster, our method demonstrates improvements of 1.45% on simple tables, 8.36% on complex tables, and 3.07% overall. On iFLYTAB, a TEDS-S accuracy of 92.00% is achieved by our method, comparable to SEMv2 [30] and outperforms other methods.

Results on digital document tables. The performance of our model is also evaluated on three digital document table datasets: PubTabNet [33], FinTabNet [32] and SynthTabNet [20]. For PubTabNet, similar to previous methods [13,19,9], the OCR results are from the text detection method PSENet [26] and text recognition method MASTER [18], and we match the text bboxes to the cell bboxes as described in [29]. A TEDS-S score of 97.63% and a TEDS score of 96.57% are achieved on PubTabNet which outperforms other methods. For FinTabNet and SynthTabNet, TEDS-S scores of 98.89% and 99.11% are achieved, respectively. Compared with TableFormer [20], our method exhibits improvements of 2.09% and 2.41% on FinTabNet and SynthTabNet, respectively.

In addition, we evaluate the performance of cell bbox detection on PubTab-Net [33] using the PASCAL VOC mAP metric. As shown in Table 2, our method outperforms TableFormer [20] by 5.1% even without using post-processing.

The results on five datasets validate the efectiveness of using bbox to guide the generation of HTML sequences.

Table 3: Ablation studies of module design. BG signifies bbox-guided HTML generation. PT signifies the progressive training method.
<table><tr><td colspan="2">Methods</td><td colspan="3">TEDS-S</td></tr><tr><td>BG</td><td>PT</td><td>Simple</td><td>Complex</td><td>All</td></tr><tr><td></td><td></td><td>98.30</td><td>86.50</td><td>95.54</td></tr><tr><td>√</td><td></td><td>98.35</td><td>89.27</td><td>96.23</td></tr><tr><td>√</td><td>V</td><td>98.65</td><td>92.47</td><td>97.21</td></tr></table>

Table 4: Ablation studies of advancement training stage training method. SD indicates Share Decoder. BD indicates Bbox Decoder. BGD indicates Bbox-Guided Structure Decoder.
<table><tr><td colspan="3">Methods</td><td colspan="3">TEDS-S</td></tr><tr><td>SD</td><td>BD</td><td>BGD</td><td>Simple</td><td>Complex</td><td>All</td></tr><tr><td>√</td><td></td><td></td><td>98.60</td><td>92.21</td><td>97.11</td></tr><tr><td>√</td><td>√</td><td></td><td>98.56</td><td>91.96</td><td>97.02</td></tr><tr><td>√</td><td></td><td>√</td><td>98.65</td><td>92.47</td><td>97.21</td></tr><tr><td>√</td><td>√</td><td>√</td><td>98.60</td><td>92.18</td><td>97.10</td></tr></table>

## 5.4 Visualization

We illustrate some visualization of BGTR in PubTabNet [33], FinTabNet [32], SynthTabNet [20], TabRecSet [28] and iFLYTAB [30]. As shown in Fig. 6, BGTR is adept at handling a wide range of scenarios and complex table structures. This includes tables with row and column spans, those containing multi-line text, as well as instances with empty cells. Moreover, it demonstrates strong robustness in both digital documents and natural scene environments.

## 5.5 Ablation Studies

For simplicity, we conduct ablation experiments on TabRecSet [28]. Several experiments were conducted to validate the efectiveness of our methods.

Efectiveness of module design. As indicated in Table 3, BG signifies bbox-guided HTML generation. PT signifies the progressive training method, we constructed the baseline experiment following the previous methods [20,13,29,3] which overlook the explicit utilization of bbox information when predicting HTML sequences. Utilizing BG significantly improves the TEDS-S score by 2.77% on complex tables, indicating the efectiveness of guiding HTML sequence generation with bbox information in complex table scenes. Meanwhile, PT enhances the model’s generalization capabilities, particularly in handling complex tables, proving the efectiveness of the progressive training method. As shown in Fig. 7, using the progressive training method can yield better results on spanning cells and deformed tables.

![](images/24de7e3bfa23da87be4e191a23e784ec0771c9c780469038958a6cd5293daa74.jpg)

![](images/a0333a4a6768728943c4ccf2e9d7ea75b1990ee2bb8e903cc3a1b05a7346b2ec.jpg)

![](images/9e175e433110c8857b82f937267e3e4437a9140397fa7283bd96234524764394.jpg)  
(a) PubTabNet

![](images/a6aebec0c68878a8991483b9aeadca13104af0097aa9a0005c4d379da2b91c17.jpg)

![](images/92cd51e68095fe1ccccc500c6996db0b0f8160da1d1c9ffc487d051b3d077840.jpg)

![](images/baa885af869a6dc4eecddcc993292eb802da39a73c65cb7b66212e4008f63f0c.jpg)

![](images/a3c147716d0d804d6e757642dd6367c8a952fbcd7f6644a5a08005e8f55f89d4.jpg)

![](images/0d1f9e75658c7cc4c34dedc9d1d8e96e4a9ecdbb25cacab06a8fd37f1324cd5d.jpg)

![](images/b0c0bbe247a8cb2f93be91778579b50bca8ad0252ed6633b93edf8b7af34e959.jpg)

(b) FinTabNet  
![](images/949f897517389762f91abac8c9d6019ab25c6488088651f1e10eb4ef66d4ed95.jpg)

![](images/841214326beee037ec0c409a5f174371c3dfa43329121df64fc4dcb736bf9b49.jpg)  
(c) SynthTabNet

![](images/64beab8f959ba6c2b3698a2d0f332479b568a64af5fbb2a975656bdda5cf3c0b.jpg)

![](images/b97357d88ebd59f366e6e047d693ab893e4e1f3c983388da712b8cb8b448b2e5.jpg)

![](images/e98992bcca611ad656b55c70351e66b766079a6093fcb8406ccb7920a1eb8ea9.jpg)

![](images/b8490f7e48ce888b22899de689333eb718239ca9447fd8eaa8ee5938c18f507d.jpg)

(d) TabRecSet  
![](images/abe0c876a2ceac226f5cc503aac0947824e7dcfa91d29481ca229b7af613d845.jpg)

![](images/bf5998800e3cbd037847c497c85bd7b6c7243fb5100b02769ec86694fddf1145.jpg)

![](images/1d1f1535e9ab0129f05a23752c4ae5ace9b7b50da1998a4bc41795c87a7cdca2.jpg)

![](images/67823b96ccfee75d42cdbbefa8663dd7b66631ed8e4e0321b9ae3c8a92ce33cc.jpg)  
(e) iFLYTAB  
Fig. 6: Visualization efects of BGTR. The predicted cell bounding boxes are depicted using green polygonal frames. From top to bottom, the sequence is PubTabNet [33], FinTabNet [32], SynthTabNet [20], TabRecSet [28] and iFLY-TAB [30]. The first and third columns show the visualization of bounding boxes, while the second and fourth columns display the visualization of HTML sequences.

![](images/1e466e54166da932f7551adb5ea02d114d125b75c7179bf0985a8d000f71d425.jpg)  
Fig. 7: Comparison of HTML visualization w/ and w/o progressive training on TabRecSet [28], the red dotted boxes indicate error results, the blue dotted boxes indicate spanning cells.

Table 5: Ablation studies of λ in loss function.
<table><tr><td rowspan="2">λ</td><td colspan="3">TEDS-S</td></tr><tr><td>Simple</td><td>Complex</td><td>All</td></tr><tr><td>0.5</td><td>98.49</td><td>91.82</td><td>96.94</td></tr><tr><td>1</td><td>98.65</td><td>92.47</td><td>97.21</td></tr><tr><td>2</td><td>98.57</td><td>91.92</td><td>97.02</td></tr></table>

Efectiveness of advancement training stage training method. As indicated in Table 4, placing a check mark (✓) signifies that the module continues to the advancement training stage of training, building upon the foundation training stage, while its absence indicates starting the training anew. From the results, we can see that SD (Share Decoder) and BGD (Bbox-Guided Structure Decoder) are very helpful for the training in the advancement training stage. This indicates that training in the foundation training stage with a large amount of data enables the model to learn a wide variety of table structures. However, due to the data diferences between two stages, BD (Bbox Decoder) is not much of a help for the training in the advancement training stage.

Efectiveness of λ in loss function. As indicated in Table 5, the table indicates that deep supervision positively impacts performance. However, the numerical results demonstrate a notable consistency across various trade-of parameter settings. For simplicity in model training, we recommend using λ = 1 in practical applications.

## 6 Conclusion

In this paper, we introduced BGTR, a novel framework that explicitly use bbox information to guide the generation of HTML sequences. Besides, to alleviate the problem of insuficient data leading to inadequate training of bbox-guided HTML generation in natural scenes, we adopted a progressive training method for natural scene tables and introduced SNSTab, a large synthetic table dataset in natural scenes. Experimental results on five benchmark datasets demonstrate that the proposed method achieves state-of-the-art performance.

Acknowledgements The research is partially supported by National Key R&D Program of China (2023YFC3502900), National Natural Science Foundation of China (No. 62176093, 61673182), Key Realm R&D Program of Guangzhou (No. 202206030001), Guangdong Provincial Science and Technology Plan (No. 2023A0505030016).

## References

1. Agarwal, M., Mondal, A., Jawahar, C.: Cdec-net: Composite deformable cascade network for table detection in document images. In: 2020 25th international conference on pattern recognition (ICPR). pp. 9491–9498. IEEE (2021)

2. Bookstein, F.L.: Principal warps: Thin-plate splines and the decomposition of deformations. IEEE Transactions on pattern analysis and machine intelligence 11(6), 567–585 (1989)

3. Chen, B., Peng, D., Zhang, J., Ren, Y., Jin, L.: Complex table structure recognition in the wild using transformer and identity matrix-based augmentation. In: International Conference on Frontiers in Handwriting Recognition. pp. 545–561. Springer (2022)

4. Chi, Z., et al.: Complicated table structure recognition. arXiv preprint arXiv:1908.04729 (2019)

5. Contributors, T.: Tal\_ocr\_table: A scene table structure recognition benchmark (2021), https://ai.100tal.com/dataset

6. Dai, G., Zhang, Y., Ke, Q., Guo, Q., Huang, S.: One-shot difusion mimicker for handwritten text generation. In: European Conference on Computer Vision (2024)

7. Dai, G., Zhang, Y., Wang, Q., Du, Q., Yu, Z., Liu, Z., Huang, S.: Disentangling writer and character styles for handwriting generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5977–5986 (2023)

8. Gemelli, A., Vivoli, E., Marinai, S.: Graph neural networks and representation embedding for table extraction in pdf documents. In: 2022 26th International Conference on Pattern Recognition (ICPR). pp. 1719–1726. IEEE (2022)

9. Guo, Z., et al.: Trust: An accurate and end-to-end table structure recognizer using splitting-based transformers. arXiv preprint arXiv:2208.14687 (2022)

10. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

11. Hu, L.: Immoe: Incomplete multi-view anomaly detection via mixture of view experts fusion. arXiv preprint arXiv:2607.19032 (2026)

12. Hu, L., Gan, Z., Deng, L., Liang, J., Liang, L., Huang, S., Chen, T.: Replaycad: Generative difusion replay for continual anomaly detection. In: Kwok, J. (ed.) Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25. pp. 2946–2954. International Joint Conferences on Artificial

Intelligence Organization (8 2025). https://doi.org/10.24963/ijcai.2025/328, https://doi.org/10.24963/ijcai.2025/328, main Track

13. Huang, Y., Lu, N., Chen, D., Li, Y., Xie, Z., Zhu, S., Gao, L., Peng, W.: Improving table structure recognition with visual-alignment sequential coordinate modeling. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11134–11143 (2023)

14. Li, Y., et al.: Gfte: graph-based financial table extraction. In: Pattern Recognition. ICPR International Workshops and Challenges: Virtual Event, January 10–15, 2021, Proceedings, Part II. pp. 644–658. Springer (2021)

15. Lin, W., et al.: Tsrformer: Table structure recognition with transformers. In: Proceedings of the 30th ACM International Conference on Multimedia. pp. 6473–6482 (2022)

16. Liu, H., Li, X., Liu, B., Jiang, D., Liu, Y., Ren, B.: Neural collaborative graph machines for table structure recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4533–4542 (2022)

17. Long, R., et al.: Parsing table structures in the wild. In: ICCV. pp. 944–952 (2021)

18. Lu, N., et al.: Master: Multi-aspect non-local network for scene text recognition. Pattern Recognition 117, 107980 (2021)

19. Lyu, P., et al.: Gridformer: Towards accurate table structure recognition via grid prediction. In: Proceedings of the 31st ACM International Conference on Multimedia. pp. 7747–7757 (2023)

20. Nassar, A., Livathinos, N., Lysak, M., Staar, P.: Tableformer: Table structure understanding with transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4614–4623 (2022)

21. Raja, S., Mondal, A., Jawahar, C.: Table structure recognition using top-down and bottom-up cues. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVIII 16. pp. 70–86. Springer (2020)

22. Shehzadi, T., et al.: Towards end-to-end semi-supervised table detection with deformable transformer. In: International Conference on Document Analysis and Recognition. pp. 51–76. Springer (2023)

23. Smock, B., Pesala, R., Abraham, R.: Pubtables-1m: Towards comprehensive table extraction from unstructured documents. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4634–4642 (2022)

24. Tensmeyer, C., Morariu, V.I., Price, B., Cohen, S., Martinez, T.: Deep splitting and merging for table structure decomposition. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 114–121. IEEE (2019)

25. Vaswani, A.: Attention is all you need. arXiv preprint arXiv:1706.03762 (2017)

26. Wang, W., et al.: Shape robust text detection with progressive scale expansion network. In: CVPR (2019)

27. Wright, L., Demeure, N.: Ranger21: a synergistic deep learning optimizer. arXiv preprint arXiv:2106.13731 (2021)

28. Yang, F., Hu, L., Liu, X., Huang, S., Gu, Z.: A large-scale dataset for end-to-end table recognition in the wild. Scientific Data 10(1), 110 (2023)

29. Ye, J., Qi, X., He, Y., Chen, Y., Gu, D., Gao, P., Xiao, R.: Pingan-vcgroup’s solution for icdar 2021 competition on scientific literature parsing task b: table recognition to html. arXiv preprint arXiv:2105.01848 (2021)

30. Zhang, Z., Hu, P., Ma, J., Du, J., Zhang, J., Yin, B., Yin, B., Liu, C.: Semv2: Table separation line detection based on instance segmentation. Pattern Recognition 149, 110279 (2024)

31. Zhang, Z., Zhang, J., Du, J., Wang, F.: Split, embed and merge: An accurate table structure recognizer. Pattern Recognition 126, 108565 (2022)

32. Zheng, X., Burdick, D., Popa, L., Zhong, X., Wang, N.X.R.: Global table extractor (gte): A framework for joint table identification and cell structure recognition using visual context. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 697–706 (2021)

33. Zhong, X., ShafieiBavani, E., Jimeno Yepes, A.: Image-based table recognition: data, model, and evaluation. In: European conference on computer vision. pp. 564–580. Springer (2020)

## A SNSTab

## A.1 Dataset creation

The creation of the SNSTab dataset is divided into three phases: table generation, table transformation, and background synthesis. In this section we will cover the details of each step.

Table generation. When generating digital document tables, we set the maximum number of rows in the table to 20, the maximum number of columns to 15, the minimum number of rows to 2, and the minimum number of columns to 2. In addition, in order to get spanning cells, we will randomly merge the rows and columns of the cells, which will not exceed 40% of the total number of cells. The text in the cells is mainly from some common words and phrases in Chinese and English, and the length of the text will not exceed 10. Since table image generation using browser rendering is time consuming, it takes about 20 hours on average to generate 10,000 table images.

Table transformation. After the source points are selected, the source points move randomly to form the target points. The movement of the source points do not exceed 10% of the width of the image horizontally and 10% of the height of the image vertically. In fact, not all tables in the natural scene are inclined and rotated, and in order to better simulate this situation, 20% of the tables are not transformed.

Background synthesis. As shown in Fig. 8, in order to reduce the impact of the background image on the table text, we did not select a background with text when obtaining the background image. In the background synthesis, we will first obtain the mask region of the table image, then replace the mask region in the background image with the table image, and finally get the final sample.

## A.2 Samples

SNSTab comprises a diverse range of tables within the dataset, including wired tables, Wireless tables, Provincial line tables, inclined tables, curved tables and large tables. Partial sample data is illustrated in Fig. 9 for reference.

## A.3 Statistics

To give a more complete picture of SNSTab, we have listed the following statistics:

Cell number: this represents the number of cells contained in each table, as shown in Fig. 10

Row number: this represents the number of rows in each table, as shown in Fig. 11

Column number: this represents the number of columns in each table, as shown in Fig. 12

Length of cell content: this represents the text length of each cell, as shown in Fig. 13

Rowspan number: this represents the number of rows that each cell spans, as shown in Fig. 14

Colspan number: this represents the number of columns that each cell spans, as shown in Fig. 15

![](images/a2c969d2c11e3d9c3a2e5452b9f0eeca6238e710256b3c67b661b0af3cbf0094.jpg)  
Fig. 8: Samples in the background images. Including paper, walls and daily-life items.

![](images/eac91583bf62235bb429c11499d8da0f19556af7827ddecbee45a3b113a2c21a.jpg)  
Fig. 9: Samples in the SNSTab dataset. Including wired tables, wireless tables, provincial line tables, inclined tables, curved tables and large tables.

![](images/b3b5a1901bc3ca433e5dd6459b1a0422f49b8360fdb80dc2565213fd7045b535.jpg)  
Fig. 10: Statistics of cell number.

![](images/35ee337b8870aefa2497a286e235c0e6563459c700af6234bfcabb1209f9e1d2.jpg)  
Fig. 11: Statistics of row number.

![](images/47cac2e8ebbd33d2fe423ab9e298b57ccd4d9fab2d89fb9249565ec060fbfe1e.jpg)  
Fig. 12: Statistics of column number.

![](images/91b6d1a80c6e10b16b9d2b966998ec5a8839bea300cdd4a29c5a7fc21393ec1e.jpg)  
Fig. 13: Statistics of length of cell content.

![](images/e472ca3c48c78f81360f8a7d48b3771aabfb086d4d26b7fea68f287b2894437a.jpg)  
Fig. 14: Statistics of rowspan number.

![](images/2b58aded3e2a10a52f64e7338331a7ff7eaffe9eaf3f8bcbcc628570acd8aa8c.jpg)  
Fig. 15: Statistics of colspan number.