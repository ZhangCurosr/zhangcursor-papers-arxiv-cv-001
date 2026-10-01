# Comparative study of adapting pre-trained models for driving behavior video captioning<sup>∗</sup>

Sayak Mallick<sup>2</sup> Philipp Geiger<sup>1</sup> Augustin Kelava<sup>2</sup> <sup>1</sup>Bosch Center for Artificial Intelligence <sup>2</sup>University of Tübingen sayak.mallick@student.uni-tuebingen.de Philipp.W.Geiger@de.bosch.com augustin.kelava@uni-tuebingen.de

## Abstract

This report examines and compares some of the many fine tuning and prompting methods existing, applying them within the domain of autonomous driving. The idea is to compare these methods by adapting a Large Language Model (LLM) on a video dataset. LLM’s have become extremely good at achieving a good understanding of different forms of data and this study aims to induce a low dimensional understanding of driving situations into our primary test model SpaceTimeGPT. Experiments on BDD-X (Berkeley DeepDrive eXplanation) dataset demonstrate good performance of the full fine tuning framework on some automatic metrics, and in some metrics, it even surpasses the baseline. We also try Low-Rank Adaptation (LoRA) and prompt engineering on VideoLLaVA model and discuss its limitations.

## 1 Introduction

Driving video captioning is an important problem. Despite being effective, the lack of interpretability often limits our understanding of what goes on under the hood in the neural network approaches of these captioning models. An effective method of captioning and understanding can have several use cases in domain of Autonomous Driving. Firstly, this would allow text-based searching for interesting scenarios in a database. Secondly, the ability to express understanding in natural language would bridge the gap between an end user, who has limited knowledge of intelligent systems, and the machine’s behaviour. Thirdly, language can be looked at as a high-level representation for rule learning. There is a need for such representations and language is one such way to have short representations which nonetheless contain the things important for the decision making. Therefore, this could be an important step also towards better (representation) learning for driving. Fourth, this could make it easier to check to what extent a dataset of given scenarios covers a given operational design domain. We achieve this by using multiple encoder-decoder couples and cutting-edge LLM’s for feature understanding in the videos. We also experiment with different adaptation methods and create a nice spectra of possible ways to achieve our objectives.

## 2 Related work

Notable related work in the domain includes ADAPT [1] which also uses the same dataset we use, BDD-X [2], for their experimentation.

ADAPT or Action-aware Driving Caption Transformer introduces a transformer architecture for captioning of driving videos. Just like our experiments, ADAPT addresses explainability and interpretability in this domain. They use a novel approach of jointly training driving caption task along with vehicular control prediction task by using a shared video representation. Since it is then expressed linguistically, it makes it easy to understand what is going on in the videos when compared to previous vision or LiDAR based approaches. The framework is then assessed on the BDD-X dataset. It results in state-of-the-art performance on automatic metrics and human evaluation. The paper includes a detailed ablation study analysing the impact of the different aspects of the design of this framework, different sampling rates and different control signal types. The authors also talk about the pipeline they developed to deploy this ADAPT framework in both simulator environments and the real world. This can facilitate real time conversion of raw driving video input into natural language. ADAPT is a direct motivation for us on this project. There were certain hurdles (eg. the docker setup limitations) which did not allow us to directly work with the ADAPT code and update it.

Wayve has released a research paper titled "GAIA-1: A Generative World Model for Autonomous Driving" [3] that explores the challenges of building autonomous driving systems, while focusing primarily on the problem of effectively predicting outcomes in response to the vehicle’s actions in an evolving world. The paper introduces GAIA-1, a generative world model that leverages video, text, and action inputs to generate realistic driving scenarios while offering control over ego-vehicle behavior and scene characteristics. Further, Wayve also released LINGO-2 which according to their website, is "the first closed-loop vision-language-action driving model (VLAM) tested on public roads". However, there is no publication available as of now. There is a previous paper on LingoQA [4], which works towards the same direction.

## 3 Setting and problem formulation

Setting: In this paper, we consider video-to-text models having

• the inputs X which can be broken into visual signals $X _ { p i x }$ and further, optionally, textual input $X _ { t e x }$

• a set of textual labels Y .

## Given:

• a general-purpose video captioning (video-to-text) model M pre-trained on other large datasets (or individual components of them which can then be somehow connected, for eg. video encoder ENC and text decoder DEC)

– θ are all trainable parameters

– the model follows a probability distribution $p _ { \theta }$ as its conditional density

– the density factorises as follows,

$$
p ( { \cal Y } \mid { \cal X } _ { p i x } , { \cal X } _ { t e x } ) = \prod _ { i = 1 } ^ { L } p _ { \theta } ( { \cal Y } ^ { [ i ] } \mid { \cal Y } ^ { [ 1 : i - 1 ] } , { \cal X } _ { p i x } , { \cal X } _ { t e x } ^ { [ 1 : i - 1 ] } )
$$

• a dataset for the target task of driving video captioning D (and potentially related side tasks) and $D \sim ( X , Y )$ pairs.

![](images/612ab5396776896fc117f28184654bc6aebe933f120bb28440e2db641cf38f2b.jpg)  
Figure 1: Summary of approach studied in this paper of how to get from dataset and pre-trained model to solution of the driving video captioning task (pictures taken from BDD / BDD-X).

One of the goals: to train a model that performs well on driving video behaviour captioning task. This can be achieved by maximizing the probability in the above equation. The objectives of the experiment change according to the approach taken.

The overview of this document is summarised in Figure 1.

## 4 Approaches

The most important underlying mechanism for our experiments is the transformer architecture. Transformers have achieved state-of-the-art performance in various NLP tasks, including language translation, text summarization, and sentiment analysis, because of their ability to capture long-range dependencies and contextual information efficiently.

In the ViViT paper[5], using these kind of transformer models for video classification was studied. In order to handle the long sequences of tokens in video data, they propose many efficient variants of the model to factorise the spatial- and temporal-dimensions of the input.

The table in Figure 2 in the supplementary material sums up some key differences between classic text transformers and video transformers.

The approaches taken are discussed in two different sections: Pre-trained Models and Adaptation Methods.

## 4.1 Pre-trained Models:

A pre-trained model is a machine learning model that has already been trained on a large dataset for a particular task beforehand. They can be used as a starting point for other tasks by transferring their previous knowledge into the new problem at hand. This can be effectively achieved by fine-tuning them either completely or by freezing certain layers.

In our experimentation, we work with TimesFormer GPT-2, which is pretrained on the VaTeX (public dataset containing diverse video captioning and description instances)[6]. The other model (VideoLLaVA) gets its pretraining datasets from multiple sources. The image pretraining is done on the LLAVA[7] dataset and the video pretraining is done on the VALLEY[8] dataset. The next sections summarises the different pretrained models that we use either off the shelf or with some sort of adaptation in our experimentation.

## 4.1.1 TimeSformer GPT-2

TimeSformer GPT-2 or TFGPT2 is an extension of the GPT-2 architecture integrating temporal information into its model, which is done by using the TimeSformer[9] model. By incorporating time dependent structure $X _ { p i x }$ alongside textual data, This model enhances the understanding of sequential data. This makes it particularly useful for tasks involving time-series data or sequential information such as stock market prediction, video analysis, or natural language generation. The main idea is to develop upon the success of GPT-2 in generating coherent and contextually relevant text while also accounting for temporal dynamics, leading to improved performance in tasks which can benefit with understanding of time. In essence, this model is basically an encoder-decoder couple, where the encoder is "facebook/timesformer-base-finetuned-k600" from the Huggingface model repository and the decoder is "gpt2". In the TimeSformer paper[9], the authors have included a comparison between five space-time self-attention schemes: Space Attention, Joint Space-Time Attention, Divided Space-Time Attention, Sparse Local Global Attention and Axial Attention. Out of these, Divided Space-Time Attention achieves the best results. Compared to 3D CNN’s, the TimeSformer is a much improved model because CNN’s have strong inductive biases like local connectivity and translation invariance. These characteristics are beneficial for small datasets but not in our use case. Local connectivity means that the neurons in a layer are connected only to a local region of input. Translation invariance means that a shift in the input image pixel values results in a proportionate shift in output pixel values. The TimeSformer has a large learning capacity in spite of its low inference cost. This facilitates increasing model capacity and means that the TimeSformer scales fairly well and is quicker to train compared to CNN’s[10].

## 4.1.2 TimeSformer BERT

We also use the TimeSformer BERT model, which is basically the TimeSformer model in conjunction with the BERT language model[11]. It is slightly less powerful than the GPT-2 version, as we will see later in the experiments section.

## 4.1.3 Video-LLaVA

Video-LLaVA[12], also known as Large-scale visual language is a model to better understand and process images and videos and is a powerful AI model designed to bridge the gap between how computers see and understand the world compared to humans. By converting video information into a format similar to text, Video-LLaVA can analyze videos like a large language model analyzes text. This allows it to deeply understand and consequently, grasp the content of a video as present in $X _ { p i x } ,$ including the actions and objects within it much more effectively than previous models. Additionally, a prompt $X _ { t e x }$ can be included. This allows for improved understanding which is key to applications like video summarization, generating captions, and answering questions based on video content. The training pipeline consists of two separate stages : image/video understanding and then instruction tuning.

## 4.2 Adaptation Methods

The two major adaptation methods we experiment with are the full fine tuning and the LoRA. We also try to engineer prompts for the bigger Video-LLaVA model.

## 4.2.1 Full-fine tuning

Classic full fine-tuning comprises of adjustment of the parameters θ of a pre-trained model to adapt it to a specific task by training it on the dataset D for target task of driving video captioning. This process begins with a pre-trained model M that has preciously been trained on a large and diverse dataset. Then, some layers may be frozen to encourage the model to remember key broad details from previous task. After this, the final layers (which presumably contain finer and task specific information) are replaced to fit the new task. This new network is then trained, using the new dataset D. This gives us a new set of weights and biases which are eventually used by the model to perform target task. An essential part of this is the use of an appropriate optimization algorithm (such as stochastic gradient descent) and the right loss function. Classic full fine tuning results in better performance on the new task than pretrained model[13]. However, the process is often slow and requires significant compute power, especially for large models like the modern LLM’s. There is also a high risk of overfitting in case the dataset is not large enough.

For classic fine tuning, the objective is

$$
m a x _ { \theta } \sum _ { D : ( X , Y ) } \sum _ { i = 1 } ^ { L } l o g ( p _ { \theta } ( Y ^ { [ i ] } \mid Y ^ { [ 1 : i - 1 ] } , X _ { p i x } , X _ { t e x } ^ { [ 1 : i - 1 ] } )
$$

Let $\theta _ { 0 }$ be all θ that are already present and ∆θ be the new additional parameters to be trained. Usually, in any kind of fine tuning, the model is initialized with pretrained weights $\theta _ { 0 }$ and updated to $\theta _ { 0 } + \Delta \theta$ In the case of full fine tuning, we do not learn a different set of parameters $\Delta \theta$ . Therefore, |∆θ| here is $| \theta _ { 0 } |$ as the initial parameters itself are further trained.

## 4.2.2 Low-Rank Adaptation fine tuning

LoRA fine-tuning[14] is a method that fine tunes more efficiently by introducing low-rank matrices into selected layers, while keeping most parameters fixed. This approach starts with a pre-trained model M. Most of the parameters stay fixed and hold the model’s knowledge from the previous task. Then, we introduce low-rank matrices into some layers. This new architecture is then trained on the dataset D. These matrices capture the information that helps the model perform better on the new task. Like full fine-tuning, it is of prime importance to use the right optimization algorithm and loss function for the target task. Since we train only the newly added low rank matrices, while keeping the rest of the parameters fixed, LoRA is much more efficient computationally and also takes lesser time and memory. Further, there is also a lesser risk of over-fitting, since majority of the parameters are unchanged. In this way, LoRA allows us to quickly adapt pretrained models for new target tasks.

In the second part of the experiments, we implement LoRA on our target dataset D. This is a much more parameter-efficient approach, which allows the task specific parameter increment $\Delta \theta = \Delta \theta ( \alpha )$ to be further encoded by a small number of parameters α. The dimensionality of the parameter vector α is much smaller than the dimensionality of the original parameter vector $\theta _ { 0 }$

$$
\left. \alpha \right. < < \left. \theta _ { 0 } \right.
$$

Then, the revised objective is

$$
m a x _ { \alpha } \sum _ { D : ( X , Y ) } \sum _ { i = 1 } ^ { L } l o g \big ( p _ { \theta _ { 0 } + \Delta \theta ( \alpha ) } \big ( Y ^ { [ i ] } \mid Y ^ { [ 1 : i - 1 ] } , X _ { p i x } , X _ { t e x } ^ { [ 1 : i - 1 ] } \big )
$$

## 4.2.3 Prompt Engineering and In-context learning

In-context learning is an adaptation method where a pre-trained model learns how to perform better on a new target task by looking at examples of how its output should be. There are many different forms of in-context learning, like zero shot learning, one shot learning or many shot learning. In each of these, there is either zero, one or many examples included in the prompt, for the model to learn from. This, along with the query for the new task, leads the model to the right output. The approach allows the model to understand the context and generate responses based on the formats of the example/s provided. This method is frequently used as it hardly takes any further computational resources. The strong limitation of this method is that it is bound by the pre-existing knowledge of the model and the quality of examples in the prompt.

Prompt engineering is an adaption method that improves performance on a task by customising the input prompts. In this type of adaptation, there is no change in the underlying parameters; instead, we leverage the pre-existing knowledge and logical understanding of the model. The prompt may be crafted to contain particular instructions or examples that nudges the LLM to outputting the desired behaviour based on task requirement. Prompt engineering requires no further computational resources as it only involves manipulation of the input text. However, as we see later on, the success of prompt engineering can be limited by the model’s inherent capabilities and the quality of the crafted prompts.

On the TimeSformer-GPT2, we experiment with the classic fine tuning and the LoRA while on the larger Video-LLaVA model, we tried a couple of different prompt engineering techniques. The full fine tuning and LoRA for Video-LLaVA could not be completed due to time and computational restraints.

## 5 Experiments

We experiment with the BDD100K and the BDDX datasets for our tasks. They are both widely used datasets in the domain of autonomous driving.

## 5.1 BDD100K

BDD100K[15], or Berkeley Deep Drive 100K, is an important resource in the realm of computer vision and autonomous driving. It contains 100,000 videos, each about 40s long, and the format is 720p at 60 frames per second. The videos are collected and recorded all over the United States. This gives us a great mix of different circumstances such as city streets to countryside scenes. The dataset is labeled at a key-frame from every 10th second and have detailed annotations attached. This includes image tagging, bounding boxes and image segmentation details. There is also further information on road object detection. For our auxiliary task, we use this dataset to try and predict weather, surroundings and time of day columns as present in the dataset. We do this to gauge the level of understanding of the TimeSformer model. The results of this auxiliary task is presented in the supplementary section.

## 5.2 BDDX

BDDX or Berkeley Deep Drive X [2], is a unique dataset. It consists of video-label pairs and is one of the few datasets available that allows a model to learn captioning in the autonomous driving domain. BDDX has over 77 hours of driving videos spread across 7000 videos. These videos are tapes under a wide range of weather conditions such as rain, snow, sun. Just like the BDD100K, there is a plethora of videos for different situations, such as, overtakes, intersections, and simple speeding up. There is a pre-defined training/validation/testing split of the dataset. To process these videos in an easier manner, we preprocess dataset D to contain video names directly and make each row a segment. In this form, dataset D has 5 total columns: a column containing video name, and the corresponding segment labels (start time, end time, caption, reasoning).

We can introduce BDDX as our primary dataset D and break it down into its contents. The contents are: - the videos, X (the dataset itself contains video links which directly correspond to video names from a complete data pool of videos from BDD100K) - the corresponding labels, Y (consists of start time, end time, caption and reasoning for each segment of the video)

For our primary task, we try to predict captions that not only generate the action taken by the vehicle but also some sort of understanding as to why it takes it.

The four primary metrics we test on are the BLEU-4, CIDEr, METEOR and ROUGE-L. The CIDEr metric lies between 0 and 1000 while the rest are between 0 and 100.

Our first try is with the TFGPT2. In Table 1, we find the comparison between all the methods we experimented on.

Table 1: Metric scores
<table><tr><td>Model used</td><td>Method of adaptation</td><td>Rank</td><td>Prompt Number</td><td>BLEU4</td><td>CIDEr</td><td>METEOR</td><td>ROUGE-L</td></tr><tr><td>ADAPT - Narration</td><td></td><td></td><td></td><td>34.6</td><td>247.5</td><td>30.6</td><td>62.8</td></tr><tr><td>ADAPT - Reasoning</td><td></td><td></td><td></td><td>11.4</td><td>102.6</td><td>15.2</td><td>32.0</td></tr><tr><td>TF-GPT2</td><td>Pretrained model</td><td></td><td></td><td>0.02</td><td>3.9</td><td>7.9</td><td>12.4</td></tr><tr><td>TF-BERT</td><td>Pretrained model</td><td></td><td></td><td>0.001</td><td>1.9</td><td>5.4</td><td>10.3</td></tr><tr><td>TF-GPT2</td><td>Full fine-tuning</td><td></td><td></td><td>10.0</td><td>105.9</td><td>34.9</td><td>44.3</td></tr><tr><td>TF-BERT</td><td>Full fine-tuning</td><td></td><td></td><td>5.5</td><td>69.6</td><td>32.9</td><td>41.5</td></tr><tr><td>TF-GPT2</td><td>LoRA fine tuning</td><td>16</td><td></td><td>9.7</td><td>83.4</td><td>34.4</td><td>44.7</td></tr><tr><td>TF-GPT2</td><td>LoRA fine tuning</td><td>128</td><td></td><td>9.0</td><td>75.4</td><td>33.9</td><td>43.6</td></tr><tr><td>TF-GPT2</td><td>LoRA fine tuning</td><td>768</td><td>=</td><td>9.3</td><td>96.9</td><td>33.9</td><td>43.2</td></tr><tr><td>Video-LLaVA</td><td>Pretrained model</td><td></td><td>P1</td><td>0.02</td><td>1.6</td><td>19.3</td><td>17.9</td></tr><tr><td>Video-LLaVA</td><td>Prompt Engineering</td><td></td><td>P2</td><td>0.12</td><td>8.2</td><td>27.3</td><td>32.9</td></tr><tr><td>Video-LLaVA</td><td>Prompt Engineering</td><td></td><td>P3</td><td>0.09</td><td>6.7</td><td>26.5</td><td>31.2</td></tr></table>

Initially, we compare the TF-GPT2 and TF-BERT pretrained models.Then comes the fully finetuned versions of the TFGPT2 and TF-BERT models. In the next part, we compare the different LoRA configurations for the TFGPT2 model. LoRA models can be modified by changing rank r and scaling factor alpha. We take a constant alpha and play around with the rank.

Then, we examine different prompt engineering techniques for the Video-LLaVA. The prompts that we experimented on and used are :

Prompt number 1 P1: "What action does the car take and why do you think it takes this action?" Prompt number 2 P2: "What action does the car take and why? Please respond in one sentence in this format: ’The car takes (action) because (reasoning).’ Include specific details about the traffic signals, road conditions, and any relevant environmental factors."

Prompt number 3 P3: "Describe the car’s action and reasoning in one sentence in the following format: ’The car takes (action) because (reasoning).’ Include details about the surrounding conditions and factors influencing the action."

The discussion of comparison between the methods is in the next section.

## 6 Discussion and analysis

## 6.1 Discussion of methods

## 6.1.1 Pretrained models

As expected, the pretrained models fall well short in the driving captioning and understanding task. This is because the data it is pretrained on is very different and having not looked at the new dataset, the model has no idea what to look for in the videos and structure of the video shot. In driving, video is always shot from the ego vehicle point of view. Here, we also see that the GPT2 version of TimeSformer is seemingly more expressive, because of the higher number of parameters present.

## 6.1.2 Full fine tuning

After first epoch of fine-tuning, the model seems to act as if the task is binary classification. With time and some more epochs, the model becomes more expressive and it becomes more like a multi-class classification task. It learns the most frequent outputs first, and the expressiveness and detail come only with the last few epochs of training.

## 6.1.3 LoRA fine tuning

Not all layers are trainable using LoRA. The first iterations using linear layers did not yield the desired outputs, as we saw repetitive outputs and unexpected sentence structures pop up. Finally, we settled on using linear, embedding and 1D and 2D convolution layers and this improved the outputs.

## 6.1.4 Prompt engineering

Prompt engineering in general seems to not perform too well here. We tried 3 different prompts ranging from simple to more complex. The performance definitely improves but it does not even come close to the outputs of the fine tuning.

## 6.2 Discussion of results

As expected, the full fine tuning outperforms the other experiments in all the metrics. It even outperforms some of the ADAPT metrics, which is appreciable because we achieve it with limited hardware and nonspecialist models made to understand the domain purely by adaptation. For the different LoRA, it seems that the model has issues with fitting as the different configurations seem to give similar results.

## 7 Conclusion

We see a glimpse of the power of the modern language models. Inspite of TimeSformer-GPT2 being much smaller and less complex than the ADAPT architecture, it performs comparably on many of the metrics. With further data, the capability of these models can be maximised. Also, more capable models like Video-LLaVA can be inspected and fine-tuned for driving video captioning and understanding. Although that does require significant computational resources and time, it should yield appreciable results.

We find that classical full fine-tuning, although computationally intensive yields the best results. The choice of adaptation method then becomes a conscious decision based on computational resources and time available to the user. Sometimes, simple prompt engineering can already lead to good results.

## A Appendix

## A.1 Background on metrics

BLEU-4 : Bilingual Evaluation Understudy[16] is an inexpensive and language independent method of automatic machine translation evaluation that is used to evaluate quality of translated text by comparing it to one or more references. This is done by comparing n-grams in candidate translations to the references. BLEU-4 is the metric that focuses on 4-grams. It then takes the geometric mean and adds a brevity penalty to discourage shorter translation. Higher scores indicate better quality translations that closely match the reference texts in terms of word sequences.

CIDEr : Consensus-based Image Description Evaluation[17] is a metric developed specifically for the evaluation of image captions generated by machine learning models. It measures the harmony, in a sense, between multiple reference captions and the generated caption. Unlike BLEU, which primarily focuses on n-gram overlaps, CIDEr considers both the similarity of individual words and their semantic relevance within the context of the image. In a way, it is n-gram similarity weighted with the TF-IDF score. Thus, this metric tends to correlate better with human judgments of caption quality, especially in scenarios where diversity and relevance are crucial.

ROUGE-L : Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence[18] is a metric commonly used in natural language processing tasks, such as summarization or translation. It evaluates the quality of summaries or translations by computing the "longest common subsequence" between the candidate summary and the reference summaries. ROUGE-L emphasizes on recall by measuring exactly what part of the reference summary is captured in the candidate string. A higher scores means that there is a larger overlap between candidate and reference, and consequently results in a better quality of translation.

METEOR : Metric for Evaluation of Translation with Explicit Ordering[19] is a metric that evaluates a translation by computing a score based on word-to-word matches between the translation and a reference. It considers both the lexical and the semantic similarities. It does so, by computing a harmonic mean of precision and recall, weighted by a measure of alignment between words in the candidate and reference. Additionally, METEOR takes into account stemming, synonyms and word orders while evaluating the final score, making it highly robust to variations in language expression. A higher METEOR scores indicate better quality translations that capture both surface-level and semantic aspects of the reference text.

## References

[1] Bu Jin, Xinyu Liu, Yupeng Zheng, Pengfei Li, Hao Zhao, Tong Zhang, Yuhang Zheng, Guyue Zhou, and Jingjing Liu. Adapt: Action-aware driving caption transformer, 2023. URL https: //arxiv.org/abs/2302.00673.

[2] Jinkyu Kim, Anna Rohrbach, Trevor Darrell, John Canny, and Zeynep Akata. Textual explanations for self-driving vehicles. Proceedings of the European Conference on Computer Vision (ECCV), 2018.

[3] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving, 2023. URL https://arxiv.org/abs/2309.17080.

[4] Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, Elahe Arani, and Oleg Sinavski. Lingoqa: Video question answering for autonomous driving, 2024. URL https://arxiv.org/abs/2312.14115.

[5] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luciˇ c, and Cordelia´ Schmid. Vivit: A video vision transformer, 2021. URL https://arxiv.org/abs/2103. 15691.

[6] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In The IEEE International Conference on Computer Vision (ICCV), October 2019.

[7] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.

[8] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability, 2023.

[9] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding?, 2021. URL https://arxiv.org/abs/2102.05095.

[10] Silvio Olivastri, Gurkirt Singh, and Fabio Cuzzolin. End-to-end video captioning, 2019. URL https://arxiv.org/abs/1904.02628.

[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. URL https://arxiv.org/ abs/1810.04805.

[12] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection, 2023. URL https: //arxiv.org/abs/2311.10122.

[13] Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping, 2020. URL https://arxiv.org/abs/2002.06305.

[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https://arxiv.org/abs/2106.09685.

[15] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. Bdd100k: A diverse driving dataset for heterogeneous multitask learning, 2020. URL https://arxiv.org/abs/1805.04687.

[16] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin, editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https://aclanthology.org/P02-1040.

[17] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4566–4575, 2015. doi: 10.1109/CVPR.2015.7299087.

[18] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013.

[19] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Jade Goldstein, Alon Lavie, Chin-Yew Lin, and Clare Voss, editors, Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72, Ann Arbor, Michigan, June 2005. Association for Computational Linguistics. URL https://aclanthology.org/ W05-0909.