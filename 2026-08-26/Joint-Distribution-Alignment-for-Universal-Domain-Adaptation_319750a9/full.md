# Joint Distribution Alignment for Universal Domain Adaptation

Shizhe Li, Hongshan Pu, Mengying Xie, Yi Xiang, and Xiaowei Yang

Abstract—Unsupervised domain adaptation (UDA) has been widely concerned in the fields of machine learning, pattern recognition, and computer vision. Traditional UDA learning usually assumes that the label spaces of the source and target domains are exactly the same and only needs to solve the problem of sample distribution drift existing between two domains. However, in real world applications, the label spaces between two domains may be different. In this case, there are both sample distribution drift and class spatial difference between domains, namely Universal Domain Adaptation (UniDA) learning scenario. At present, existing works rarely offer theoretical analysis for universal domain adaptation. In this paper, we provide an upper bound of the generalization error for universal domain adaptation. According to the proposed generalization error bound, we propose a novel UniDA algorithm called Joint Distribution Alignment for Universal Domain Adaptation (JAUA), which aligns the joint distributions by minimizing the distribution discrepancy calculated by Chi-Square divergence. Furthermore, we propose a progressive pseudo-labeling method to assign the pseudo labels to unlabeled target samples. The experiment results on six public image datasets demonstrate the superiority of JAUA in handling the UniDA problem.

Index Terms—Universal domain adaptation, progressive pseudo-labeling method, generalization error bound.

## I. INTRODUCTION

RADITIONAL machine learning follows the basic assumption that both the training data and the test data follow the same joint probability distribution [1]. However, in practical applications such as biological imaging, medical imaging, and industrial defect detection, the data distribution may shift due to environmental conditions, camera angles, or device status, making it challenging to satisfy this identical distribution assumption [2]. Under such conditions, classifiers with excellent performance on source images cannot be directly adapted to target image classification tasks. For efficiency reasons, transferring a trained model in one domain to another can save training costs significantly. To maintain the model performance in new domains, it is often necessary to recollect and relabel training data, which incurs significant human, material, and time costs. To address this issue, transfer learning treats the training data as the source domain and the test data as the target domain, aiming to identify transferable or reusable features, parameters, or models across domains. As a critical category in transfer learning, domain adaptation (DA) focuses on scenarios where the data distributions in the source and target domains are inconsistent, known as domain shift [3], [4], as shown in Fig. 1.

![](images/0af28a5fadea9e9eb4d35a15f08551ff64e16cf40039e1af51fdb0a1a0f32b11.jpg)  
Fig. 2. Different setting of DA

Unsupervised domain adaptation (UDA) improves the model performance on unlabeled target domain by using knowledge learned in labeled source domain [5]–[7]. Tradi tional unsupervised domain adaptation methods assume that both domains share the same label set. However, in real-world scenarios, since the data of the target domain are usually unlabeled, the difference of label spaces between domains cannot be predicted [8], [9]. To tackle this limitation, the Universal Domain Adaptation (UniDA) framework has been proposed. In UniDA, both the source and target domains may contain exclusive classes, as shown in Fig. 2. In this way, UniDA can be closer to practical application and has higher practicality and scalability. At present, UniDA has been widely applied in many fields such as object detection [10], semantic clustering [11], time series analysis [12], remote sensing image classification [13], [14] and machine fault diagnosis [15].

Under the UniDA framework, two key challenges must be addressed: 1) Label space discrepancy: identifying and labeling target-domain-specific classes as ”unknown” while correctly classifying the common classes. 2) Distribution shift: aligning the joint distributions of the common classes across domains.

To address the first challenge, this study uses a progressive separation mechanism based on the idea of [16], which gradually distinguishes the common class samples from unknown class samples by selecting samples with higher confidence as common class samples. To address the second challenge, the upper bound of the target generalization error is first given based on the Chi-Square divergence, which estimates the difference of the joint distribution of common classes in the source and target domains. And then a UniDA model is built by minimizing this upper bound.

The contributions of this paper can be summarized as:

• An upper bound of the generalization error on the target domain is first proposed based on the Chi-Square divergence.

• A novel UniDA model is built by minimizing the above upper bound of the generalization error.

• A progressive pseudo labeling method is designed to assign the pseudo labels to the unlabeled target samples, which helps to improve the classification performance of the unlabeled samples on the target domain. Based on this progressive pseudo labeling strategy, a novel UniDA algorithm called Joint Distribution Alignment for Universal Domain Adaptation (JAUA) is proposed.

• To verify the effectiveness of JAUA, JAUA is compared with twenty-four state-of-the-art methods on six public datasets. The results of the experiment show the superiority of JAUA in solving UniDA problems.

The rest of this paper is organized as follows. Section II introduces the related work. Section III gives the whole model of our work. Section IV shows the complete experimental results and analysis. Section V gives conclusions and future work.

## II. RELATED WORK

In this section, we introduce briefly the related work including universal domain adaptation, some UniDA methods based on progressive separation and theoretical analysis of open-set domain adaptation.

## A. Universal Domain Adaptation

Most of the existing UniDA methods rely on thresholdbased criteria to detect unknown-class samples in the target domain. To solve the UniDA problem, You [17] proposed the Universal Adaptation Network (UAN), which assigns higher weights to samples from common classes by measuring sample-level transferability and employs a domain discriminator trained adversarially to align cross-domain features. Fu [18] introduced a Calibrated Multiple Uncertainty (CMU) metric, which estimates the probability of target samples belonging to common classes through hybrid uncertainty estimation. Li [19] developed the Domain Consensus Clustering (DCC) method, leveraging domain consensus knowledge to cluster target-domain data while separating unknown classes from common classes. The DANCE framework [20] avoids relying solely on source supervision for discriminative representations; instead, it employs self-supervised learning to capture the target domain’s clustering structure and learns feature representations through neighbor clustering. Lv [21] introduced a GAN-style autoencoder to transform features into latent representations and performed adversarial domain adaptation in the latent space, thereby providing additional guidance to the backbone network to enhance feature compactness and domain discriminability.

These methods have been widely adopted in UniDA research and provided valuable insights for addressing domain adaptation challenges in real-world applications. However, from the point of view of the principle, the key problem of UniDA is how to reduce the distribution difference between the source domain and the target domain and separate the unknown classes from the target domain.

## B. Methods Based on Progressive Separation

For the UniDA problem, there exist unknown classes in the target domain. When the source domain is aligned with the target domain, if these unknown classes are not excluded, it will result in negative transferring due to the mismatch between the common classes and the unknown classes. Therefore, it is especially important to effectively separate the common class samples and the unknown class samples. To deal with this problem, Liu proposed Separate To Adaptive (STA) [16] which is one of the most representative methods. STA uses a progressive separation mechanism to separate the common class samples and the unknown class samples by two-stage method. In the coarse separation stage, STA uses a multi-binary classifier to estimate the similarity be tween the target domain samples and each source domain class. In the fine separation stage, it selects samples with very high and very low similarity as common and unknown class data, respectively to train binary classifiers. Based on the observation that due to the similarity of weights of the common class and unknown class samples in the early training stage, a large number of unknown class samples will be involved in the distributional alignment of the common class and such a negative effect may not be ameliorated in the later stage of training, Gao [8] proposed Threshold Domain Adversarial Network (ThDAN), which adaptively computes the threshold by quantifying the average transferability scores of the source domain samples and progressively selecting the common class to update the classifier during adversarial training. To eliminate false positive samples in the unknown classes, Dai [22] proposed open-set domain adaptation GCL-OSDA for graph cooperative learning. In GCL-OSDA, an evidence network is utilized to output the confidence of the target domain samples based on the Dempster-Shafer (D-S) theory of evidence and the reliability of selecting the unknown samples is improved by asymptotically selecting the samples with a very high level of uncertainty. Du [23] proposed the Self Separation and Misseparation impact Minimization (SSMM) method, which progressively selects the target domain samples with low predictive entropy and high predictive entropy as the common class and unknown class, respectively, and maximizes the distribution difference between the common class and the unknown class. In this way, SSMM can minimize the effect of mis-separation of the target domain samples and reduce the algorithm’s dependence on the similarity between the source and target domains.

Although the above algorithms are designed to solve the domain adaptive problem and can achieve good experimental results, they are not based on the target generalization error bounds and their theoretical interpretability is lacked.

## C. Generalization Error Bound

In order to fill the theoretical gap of the adaptive problem in the open set domain, Fang [24] proposed the generalization error bound of the target domain:

$$
\begin{array} { r l r } {  { \frac { R ^ { t } ( h ) } { 1 - \pi _ { C + 1 } ^ { t } } \le R ^ { s } ( h ) + 2 d _ { \mathcal { H } } ^ { \ell } ( P _ { X ^ { t } | Y ^ { s } } , P _ { X ^ { s } } ) } } \\ & { } & \\ & { } & { \qquad + ( \frac { R _ { u , C + 1 } ^ { t } ( h ) } { 1 - \pi _ { C + 1 } ^ { t } } - R _ { u , C + 1 } ^ { s } ( h ) ) + \Lambda } \end{array}\tag{1}
$$

where $R ^ { s } ( h )$ is the generalization error on the source domain, and $d _ { \mathcal { H } } ^ { \ell } \left( \vec { P } _ { X ^ { t } \mid Y ^ { s } } , \vec { P } _ { X ^ { s } } \right)$ is the distribution difference between the common classes of the target domain and the source domain, which converts the MMD distance into a suitable regularization term for aligning marginal distributions and class-conditional distributions of the common classes among the source domain and target domain. $\begin{array} { r } { \triangle _ { 0 } \ = \ \frac { R _ { u , C + 1 } ^ { t } ( h ) } { 1 - \pi _ { C + 1 } ^ { t } } \ - \ } \end{array}$ $R _ { u , C + 1 } ^ { s } ( h )$ is the open-set difference which is considered to be closely related to the risk of the unknown classes. Λ is the stream regularization term, which makes the similar samples closer and can be used to represent the geometrical relationship between the source and target domains.

Although the above work addresses the open-set domain adaptation problem by investigating the generalized error bound theory, it ignores the negative transferable effects of separating all samples at once. At present, there is a lack of similar theoretical discussion for universal domain adaptation.

## III. SYMBOLS AND PROBLEM SETTING

In this section, we first introduce briefly some mathematical symbols and notations. And then, we give problem setting.

## A. Symbols and Notations

We list all of the mathematical symbols and notations used in this study in Table I.

## B. Problem Setting

Problem: (Universal Domain Adaptation) Given source joint probability distribution $P _ { X ^ { s } Y ^ { s } } ( x , y )$ and target joint probability distribution $P _ { X ^ { t } Y ^ { t } } ( x , y )$ , where $P _ { X ^ { t } Y ^ { t } } ( x , y )$ is different from $P _ { X ^ { s } Y ^ { s } } ( x , y )$ . The label space of source domain is $\mathcal { V } ^ { s } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { c } , y ^ { u _ { s } } \}$ . The label space of target domain is $y ^ { t } ~ = ~ \{ y _ { 1 } , y _ { 2 } , \ldots , y _ { c } , y ^ { u _ { t } } \}$ which is different from $\mathcal { V } ^ { s }$ obviously. Given the source domain $D ^ { s } = \{ x _ { i } ^ { s } , y _ { i } ^ { s } \} _ { i = 1 } ^ { m _ { s } }$ drawn from $P _ { X ^ { s } Y ^ { s } } ( x , y )$ i.i.d. and the target domain $D ^ { t } = \{ x _ { i } ^ { t } \} _ { i = 1 } ^ { m _ { t } }$ drawn from $P _ { X ^ { t } } ( x )$ i.i.d.. Our goal is to find an optimal classifier $h : \mathcal { X } \overset { } { \to } \mathcal { Y } ^ { t }$ and make the expected risk $R ^ { t } ( h ) =$ $\begin{array} { r } { E _ { ( x , y ) \sim P _ { X ^ { t } Y ^ { t } } ( x , y ) } [ l ( h ( x ) , y ) ] = \int l ( h ( x ) , y ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y } \end{array}$ smallest, where $l ( h ( x ) , y )$ is a loss function to measure the accuracy of the classifier.

TABLE I SYMBOLS AND NOTATIONS
<table><tr><td rowspan=1 colspan=4>Symbols</td><td rowspan=1 colspan=1>Descriptions</td></tr><tr><td rowspan=1 colspan=4> $\dot { \overline { { \mathcal { X } \in \mathcal { R } ^ { D } } } }$ </td><td rowspan=1 colspan=1>Feature space</td></tr><tr><td rowspan=1 colspan=4>C</td><td rowspan=1 colspan=1>The number of common classes</td></tr><tr><td rowspan=1 colspan=2> $\mathcal { V } ^ { u _ { s } } = \{</td><td rowspan=1 colspan=1>y ^ { u _ { s } } \} = \{ C + 1 \}$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Private label space</td></tr><tr><td rowspan=1 colspan=2> $\overrightarrow { \mathcal { V } ^ { u _ { t } } } = \hat { \{ \boldsym</td><td rowspan=1 colspan=1>bol { y } ^ { u _ { t } } \} } = \hat { \{ \boldsymbol { C } + 2 \} }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Unknown label space</td></tr><tr><td rowspan=1 colspan=4> $\overline { { \mathcal { V } ^ { u _ { G } } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { c } \} } }$ </td><td rowspan=1 colspan=1>Common label space</td></tr><tr><td rowspan=1 colspan=1> $\mathcal { V } ^ { s</td><td rowspan=1 colspan=2>} = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { c } , y ^ { u _ { s } } \}$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Source label space</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathcal { V</td><td rowspan=1 colspan=2>} ^ { t } = \left\{ y _ { 1 } , y _ { 2 } , . . . , y _ { c } , y ^ { u _ { t } } \right\} } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Target label space</td></tr><tr><td rowspan=1 colspan=4> $\overline { { P _ { X ^ { s } Y ^ { s } } ( x , y ) } }$ </td><td rowspan=1 colspan=1>Source joint probability distribution</td></tr><tr><td rowspan=1 colspan=4> $\overline { { P _ { X ^ { t } Y ^ { t } } ( x , y ) } }$ </td><td rowspan=1 colspan=1>Target joint probability distribution</td></tr><tr><td rowspan=1 colspan=4> $\overline { { P _ { X ^ { s } } ( x ) } }$ </td><td rowspan=1 colspan=1>Source margin probability distribution</td></tr><tr><td rowspan=1 colspan=4> $\overline { { P _ { X ^ { t } } ( x ) } }$ </td><td rowspan=1 colspan=1>Target margin probability distribution</td></tr><tr><td rowspan=1 colspan=4> $\underline { m } _ { s }$ </td><td rowspan=1 colspan=1>The size of source domain</td></tr><tr><td rowspan=1 colspan=4>mt</td><td rowspan=1 colspan=1>The size of target domain</td></tr><tr><td rowspan=1 colspan=4>Ds = {xi , yi }i=1</td><td rowspan=1 colspan=1>Source domain</td></tr><tr><td rowspan=1 colspan=4>Dt = {xi}i=1</td><td rowspan=1 colspan=1>Target domain</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>The i-th class samplesfrom source domain</td></tr><tr><td rowspan=1 colspan=4> $\overline { { D _ { i } ^ { t } } }$ </td><td rowspan=1 colspan=1>The ¿-th class samplesfrom target domain</td></tr><tr><td rowspan=1 colspan=4> $\overline { { D _ { u } ^ { t } } }$ </td><td rowspan=1 colspan=1>The unknown class samplesfrom target domain</td></tr><tr><td rowspan=1 colspan=4> $\overline { { D _ { k } ^ { s } } }$ </td><td rowspan=1 colspan=1>The common class samplesfrom source domain</td></tr><tr><td rowspan=1 colspan=4> $\overline { { D _ { k } ^ { t } } }$ </td><td rowspan=1 colspan=1>The common class samplesfrom target domain</td></tr><tr><td rowspan=1 colspan=4> $m _ { t u }$ </td><td rowspan=1 colspan=1>The size of $\overline { { D _ { u } ^ { t } } }$ </td></tr><tr><td rowspan=1 colspan=4> $m _ { s k }$ </td><td rowspan=1 colspan=1>The size of $\overline { { D _ { k } ^ { s } } }$ </td></tr><tr><td rowspan=1 colspan=4> $m _ { t k }$ </td><td rowspan=1 colspan=1>The size of $\frac { \kappa } { D _ { k } ^ { t } }$ </td></tr><tr><td rowspan=1 colspan=4> $\overline { { h \in \mathcal { H } } }$ </td><td rowspan=1 colspan=1>Classification model</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R ^ { s } ( h ) } }$ </td><td rowspan=1 colspan=1>Source risk</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R ^ { t } ( h ) } }$ </td><td rowspan=1 colspan=1>Target risk</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R ^ { t u } ( h ) } }$ </td><td rowspan=1 colspan=1>Target unknown risk</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R ^ { t k } ( h ) } }$ </td><td rowspan=1 colspan=1>Target common risk</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R ^ { s u } ( h ) } }$ </td><td rowspan=1 colspan=1>Source private risk</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R ^ { s k } ( h ) } }$ </td><td rowspan=1 colspan=1>source common risk</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R _ { u _ { s } } ^ { s } ( h ) } }$ </td><td rowspan=1 colspan=1>Loss of samples of the source domainbeing classified as private class</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R _ { u _ { t } } ^ { t } ( h ) } }$ </td><td rowspan=1 colspan=1>Loss of samples of the target domainbeing classified as unknown class</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R _ { u _ { t } } ^ { s k } ( h ) } }$ </td><td rowspan=1 colspan=1>Loss of common class samplesof the source domain beingmisclassified as unknown class</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R _ { u _ { s } } ^ { t k } \left( h \right) } }$ </td><td rowspan=1 colspan=1>Loss of common class samplesof the target domain beingmisclassified as private class</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R _ { u _ { t } } ^ { t u } ( h ) } }$ </td><td rowspan=1 colspan=1>Loss of unknown class samplesof the target domain beingclassified as unknown class</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R _ { u _ { t } } ^ { t k } ( h ) } }$ </td><td rowspan=1 colspan=1>Loss of common class samplesof the target domain beingmisclassified as unknown class</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R _ { u _ { s } } ^ { s u } ( h ) } }$ </td><td rowspan=1 colspan=1>Loss of private class samplesof the source domain beingclassified as private class</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R _ { u _ { s } } ^ { s k } ( h ) } }$ </td><td rowspan=1 colspan=1>Loss of common class samplesof the source domain beingmisclassified as private class</td></tr><tr><td rowspan=1 colspan=4> $\overline { { R _ { k } ^ { t k } ( h ) } }$ </td><td rowspan=1 colspan=1>Loss of common class samplesof the target domain beingclassified as common class</td></tr></table>

## IV. PROPOSED MODEL AND ALGORITHM

In this section, we firstly give the upper bound of the generalization error on the target domain. Secondly, a novel mathematical model is built by minimizing the upper bound of the generalization error. Finally, a progressive pseudo labeling

algorithm is designed.

## A. Upper Bound of Generalization Error

The definitions of $R ^ { t } { ( h ) } , R ^ { s } ( h ) , R _ { u _ { t } } ^ { t } ( h ) , R _ { u _ { s } } ^ { s } ( h ) , R _ { u _ { t } } ^ { s k } ( h )$ and $R _ { u _ { s } } ^ { t k } ( h )$ are as follows:

$$
R ^ { s } ( h ) = \int _ { \chi * y ^ { s } } l ( h ( x ) , y ) P _ { X ^ { s } Y ^ { s } } ( x , y ) d x d y\tag{2}
$$

$$
R ^ { t } ( h ) = \int _ { \chi * y ^ { t } } l ( h ( x ) , y ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y\tag{3}
$$

$$
\begin{array} { l } { { \displaystyle R _ { u _ { t } } ^ { t } ( h ) = \int _ { \mathcal X } l ( h ( x ) , y ^ { u _ { t } } ) P _ { X ^ { t } } ( x ) d x } } \\ { { \displaystyle \quad \quad = \int _ { \mathcal X \ast \mathcal Y ^ { u _ { t } } } l ( h ( x ) , y ^ { u _ { t } } ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y } } \\ { { \displaystyle \quad \quad \quad + \int _ { \mathcal X \ast \mathcal Y ^ { u _ { G } } } l ( h ( x ) , y ^ { u _ { t } } ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y } } \\ { { \displaystyle \quad \quad \triangleq R _ { u _ { t } } ^ { t u } ( h ) + R _ { u _ { t } } ^ { t k } ( h ) } } \end{array}\tag{4}
$$

$$
\begin{array} { l } { \displaystyle R _ { u _ { s } } ^ { s } ( h ) = \int _ { \mathcal { X } } l ( h ( x ) , y ^ { u _ { s } } ) P _ { X ^ { s } } ( x ) d x } \\ { \displaystyle \quad \quad = \int _ { \mathcal { X } \ast \mathcal { Y } ^ { u _ { s } } } l ( h ( x ) , y ^ { u _ { s } } ) P _ { X ^ { s } Y ^ { s } } ( x , y ) d x d y } \\ { \displaystyle \quad \quad \quad + \int _ { \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } l ( h ( x ) , y ^ { u _ { s } } ) P _ { X ^ { s } Y ^ { s } } ( x , y ) d x d y } \\ { \displaystyle \quad \quad \triangleq R _ { u _ { s } } ^ { s u } ( h ) + R _ { u _ { s } } ^ { s k } ( h ) } \end{array}\tag{5}
$$

$$
R _ { u _ { t } } ^ { s k } ( h ) = \int _ { \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } l ( h ( x ) , y ^ { u _ { t } } ) P _ { X ^ { s } Y ^ { s } } ( x , y ) d x d y\tag{6}
$$

$$
R _ { u _ { s } } ^ { t k } ( h ) = \int _ { \mathcal { X } * y ^ { u _ { G } } } l ( h ( x ) , y ^ { u _ { s } } ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y\tag{7}
$$

In this paper, we use Chi-Square divergence to calculate the difference between the joint distributions of the source domain and the target domain. Based on its following definition:

$$
\begin{array} { l } { \displaystyle D _ { C S } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } | | P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } ) } \\ { \displaystyle = \int _ { \mathcal { X } _ { * \mathcal { Y } ^ { u _ { G } } } } P _ { X ^ { s } Y ^ { s } } ( x , y ) \frac { P _ { X ^ { s } Y ^ { s } } ( x , y ) } { P _ { X ^ { t } Y ^ { t } } ( x , y ) } d x d y - 1 } \\ { \displaystyle \approx \frac { 1 } { m _ { s k } } \sum _ { i = 1 } ^ { m _ { s k } } \frac { P _ { X ^ { s } Y ^ { s } } ( x _ { i } ^ { s } , y _ { i } ^ { s } ) } { P _ { X ^ { t } Y ^ { t } } ( x _ { i } ^ { s } , y _ { i } ^ { s } ) } - 1 } \end{array}\tag{8}
$$

we can give the following Theorem 1.

Theorem 1: Given a bounded loss function l. If there exist positive real numbers $M , \ N _ { s }$ and $N _ { T }$ , which makes $| l ( h ( x ) , y ) | \leq M$ hold for any $( x , y ) \in \mathcal { X } { * } \mathcal { Y } ^ { u _ { G } } , | l ( h ( x ) , y ) | \leq$ $N _ { s }$ hold for any $( x , y ) \in \mathcal { X } * y ^ { u _ { s } } , | l ( h ( x ) , y ) | \leq N _ { T }$ hold for any $( x , y ) \in \mathcal { X } * \mathcal { Y } ^ { u _ { t } }$ , then for any $h \in \mathcal H$ , the following inequality holds:

$$
\begin{array} { r l } & { R ^ { t } ( h ) \leq R ^ { s } ( h ) + \lambda _ { t t } R _ { u _ { t } } ^ { t u } ( h ) + ( \lambda _ { t } + \lambda _ { s } ) R _ { k } ^ { t k } ( h ) } \\ & { \qquad - \lambda _ { g s t } R _ { u _ { t } } ^ { s k } ( h ) - \lambda _ { g s s } R _ { u _ { s } } ^ { s k } ( h ) + ( 3 M + N _ { T } + N _ { S } ) } \\ & { \qquad \ast \sqrt { 2 D _ { C S } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } \ast \mathcal { Y } ^ { u } G }  P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } \ast \mathcal { Y } ^ { u } G } ) } + A + B } \end{array}\tag{9}
$$

where $\lambda _ { t t } , \lambda _ { g s t }$ and $\lambda _ { g s s }$ are hyperparameters, A and B are positive real numbers which satisfy $\begin{array} { r } { \bar { R } _ { u _ { t } } ^ { t k } ( h ) \leq \frac { \lambda _ { t } } { \lambda _ { t t } } R _ { k } ^ { t k } ( h ) + \frac { A } { \lambda _ { t t } } } \end{array}$ and $R _ { u _ { \mathrm { s } } } ^ { t k } ( h ) \leq \lambda _ { s } R _ { k } ^ { t k } ( h ) + B , \lambda _ { t }$ and $\lambda _ { s }$ are hyperparameters which bound $R _ { u _ { t } } ^ { t k } ( h )$ and $R _ { u _ { s } } ^ { t k } ( h ) . R _ { k } ^ { t k } ( h )$ is the loss of common classes of the target domain being classified as common class and its definition is as follows:

$$
R _ { k } ^ { t k } ( h ) = \int _ { \chi _ { * } \ y ^ { u _ { G } } } l ( h ( x ) , y ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y\tag{10}
$$

The detailed proof of Theorem 1 is given in the Appendix A.

## B. Proposed Model and Solution Method

Theorem 1 shows that the generalization error of the target domain is affected by the source domain risk $R ^ { s } ( h )$ loss of unknown class samples of the target domain being classified as unknown class $R _ { u _ { t } } ^ { t u } ( h )$ , loss of common class samples of the target domain being classified as common class $R _ { k } ^ { t k } ( h )$ , loss of common class samples of the source domain being misclassified as unknown class $R _ { u _ { t } } ^ { s k } ( h )$ , loss of common class samples of the source domain being misclassified as private class $R _ { u _ { s } } ^ { s k } ( h )$ and the difference between the joint probability distributions of two domains. In order to minimize the generalization error of the target domain, we give the following optimization model based on inequality (9)

$$
\begin{array} { r l } {  { \operatorname* { m i n } _ { h \in \mathcal { H } } \hat { R } ^ { s } ( h ) + \lambda _ { t t } \hat { R } _ { u _ { t } } ^ { t u } ( h ) + ( \lambda _ { t } + \lambda _ { s } ) \hat { R } _ { k } ^ { t k } ( h ) } } \\ & { - \lambda _ { g s t } \hat { R } _ { u _ { t } } ^ { s k } ( h ) - \lambda _ { g s s } \hat { R } _ { u _ { s } } ^ { s k } ( h ) + ( 3 M + N _ { s } + N _ { t } ) } \\ & { * \sqrt { 2 D _ { C S } ( P _ { X } s _ { Y } s _ { \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \vert \vert P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } ) } } \end{array}\tag{11}
$$

On the one hand, from the definition of Chi-Square divergence $D _ { C S } ( P _ { X ^ { S } Y ^ { S } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } | | P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } )$ , we know that its estimation needs label information of the target domain samples which will be provided by h. On the other hand, in order to obtain the optimal $h ,$ we need to find the optimal $D _ { C S } ( P _ { X ^ { S } Y ^ { S } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } | | P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } )$ . Based on the above considerations, we solve the optimization problem (11) based on alternative iteration scheme. First, we estimate h based on Representer Theorem [25]. Then we estimate $D _ { C S } ( P _ { X ^ { S } Y ^ { S } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } | | P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } )$ based on h.

(1) Estimation of hypothesis h

Based on (11), we estimate h by solving the following problem:

$$
\begin{array} { r } { h ^ { * } = \arg \underset { h \in \mathcal { H } } { \operatorname* { m i n } } \hat { R } ^ { s } ( h ) + \lambda _ { t t } \hat { R } _ { u _ { t } } ^ { t u } ( h ) + ( \lambda _ { t } + \lambda _ { s } ) \hat { R } _ { k } ^ { t k } ( h ) } \\ { - \lambda _ { g s t } \hat { R } _ { u _ { t } } ^ { s k } ( h ) - \lambda _ { g s s } \hat { R } _ { u _ { s } } ^ { s k } ( h ) + \eta \| h \| _ { \mathcal { H } } ^ { 2 } } \end{array}\tag{12}
$$

Using representer theorem, let h be represented as:

$$
h ^ { * } ( x ) = \sum _ { i = 1 } ^ { m _ { s } + m _ { t } } \alpha _ { i } k ( x , x _ { i } )\tag{13}
$$

where $\alpha ~ = ~ ( \alpha _ { 1 } , \alpha _ { 2 } , . . . , \alpha _ { m _ { s } + m _ { t } } ) ~ \in ~ \mathcal { R } ^ { ( m _ { s } + m _ { t } ) * ( C + 2 ) }$ are coefficients and $( K ) _ { i j } = k ( x _ { i } , x _ { j } ) , K \in \mathcal { R } ^ { ( m _ { s } + m _ { t } ) * ( m _ { s } + m _ { t } ) }$ is a kernel matrix.

Inspired by [26], in this study, the following quadratic loss function l is used:

$$
l ( h ( x _ { i } ) , y _ { i } ) = ( y _ { i } - h ( x _ { i } ) ) ^ { 2 }\tag{14}
$$

Compared to the other loss, on the one hand, quadratic loss function satisfies the Lipschitz condition in any finite closed interval and this character provides the theoretical basis for proving effectiveness of the proposed progressive pseudo labeling method. On the other hand, it can be easily derived to yield an analytical solution for $h .$

Let $Y , Y _ { u _ { t } } , Y _ { k } \in R ^ { ( C + 2 ) * ( m _ { s } + m _ { t } ) }$ be 0-1 matrices, $W , V _ { t u } ,$ $V _ { k } , V _ { s k } \in \tilde { R ^ { ( m _ { s } + m _ { t } ) * ( m _ { s } + m _ { t } ) } }$ and $Y _ { u _ { s } } \in R ^ { ( C + 2 ) * ( m _ { s } + m _ { t } ) }$ be a diagonal matrices. Their specific definitions are as follows:

$$
( Y ) _ { i j } = { \Bigg \{ } 1 , \quad x _ { j } \in D _ { i } ^ { s }\tag{15}
$$

$$
( Y _ { u _ { t } } ) _ { i j } = { \left\{ \begin{array} { l l } { 1 , } & { i = C + 2 } \\ { 0 , } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }\tag{16}
$$

$$
( Y _ { k } ) _ { i j } = { \Bigg \{ } 1 , ~ x _ { j } \in D _ { i } ^ { t }\tag{17}
$$

$$
( W ) _ { i i } = \left\{ { \begin{array} { l l } { { \sqrt { \frac { 1 } { m _ { s } } } } , } & { x _ { i } \in D ^ { s } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{18}
$$

$$
( V _ { t u } ) _ { i i } = \left\{ \begin{array} { l l } { \sqrt { \frac { 1 } { m _ { t u } } } , } & { x _ { i } \in D _ { u } ^ { t } } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{19}
$$

$$
( V _ { k } ) _ { i i } = { \left\{ \begin{array} { l l } { { \sqrt { \frac { 1 } { m _ { t k } } } } , } & { x _ { i } \in D _ { k } ^ { t } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{20}
$$

$$
( V _ { s k } ) _ { i i } = \left\{ \begin{array} { l l } { \sqrt { \frac { 1 } { m _ { s k } } } , } & { x _ { i } \in D _ { k } ^ { s } } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{21}
$$

$$
( Y _ { u _ { s } } ) _ { i j } = { \left\{ \begin{array} { l l } { 1 , } & { i = C + 1 } \\ { 0 , } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }\tag{22}
$$

Then, based on the law of large numbers and matrix representation, we can easily obtain the empirical estimations of $R ^ { s } ( h ) , R _ { u _ { t } } ^ { t u } ( h ) , R _ { k } ^ { t k } ( h ) , R _ { u _ { t } } ^ { s k } ( h )$ and $R _ { u _ { s } } ^ { s \bar { k } } ( h )$ as follows:

$$
\hat { R } ^ { s } ( h ) = \| ( Y - \alpha ^ { \top } K ) W \| _ { F } ^ { 2 }\tag{23}
$$

$$
\hat { R } _ { u _ { t } } ^ { t u } ( h ) = \| ( Y _ { u _ { t } } - \alpha ^ { \top } K ) V _ { t u } \| _ { F } ^ { 2 }\tag{24}
$$

$$
\hat { R } _ { k } ^ { t k } ( h ) = \| ( Y _ { k } - \alpha ^ { \top } K ) V _ { k } \| _ { F } ^ { 2 }\tag{25}
$$

$$
\hat { R } _ { u _ { t } } ^ { s k } ( h ) = \| ( Y _ { u _ { t } } - \alpha ^ { \top } K ) V _ { s k } \| _ { F } ^ { 2 }\tag{26}
$$

$$
\hat { R } _ { u _ { s } } ^ { s k } ( h ) = \| ( Y _ { u _ { s } } - \alpha ^ { \top } K ) V _ { s k } \| _ { F } ^ { 2 }\tag{27}
$$

Based on the above matrix representations, the optimization problem (11) can be rewritten as follows:

$$
\begin{array} { r l } & { \quad ( \alpha ^ { * } ) } \\ & { = \arg \underset { \alpha \in \mathcal { R } ^ { ( m _ { s } + m _ { t } ) * ( C + 2 ) } } { \operatorname* { m i n } } \eta t r ( \alpha ^ { \top } K \alpha ) + \| ( Y - \alpha ^ { \top } K ) W \| _ { 2 } ^ { 2 } } \\ & { \quad + \lambda _ { t t } \| \big ( Y _ { u _ { t } } - \alpha ^ { \top } K ) V _ { t u } \| _ { 2 } ^ { 2 } + \big ( \lambda _ { s } + \lambda _ { t } \big ) \| \big ( Y _ { k } - \alpha ^ { \top } K \big ) V _ { k } \| _ { 2 } ^ { 2 } } \\ & { \quad - \lambda _ { g s t } \| \big ( Y _ { u _ { t } } - \alpha ^ { \top } K \big ) V _ { s k } \| _ { 2 } ^ { 2 } - \lambda _ { g s s } \| \big ( Y _ { u _ { s } } - \alpha ^ { \top } K \big ) V _ { s k } \| _ { 2 } ^ { 2 } } \end{array}\tag{28}
$$

Theorem 2: If and only if $\begin{array} { l c l } { { \lambda _ { g s t } + \lambda _ { g s s } } } & { { \le } } & { { { \frac { m _ { s k } } { m _ { s } } } } } \end{array}$ , the objective function in the optimization problem (28) has unique minimum.

The proof of Theorem 2 is provided in the Appendix B. The closed solution of α is as follows:

$$
\begin{array} { r l r } { \alpha ^ { * } = ( ( W ^ { 2 } + \lambda _ { t t } V _ { t u } ^ { 2 } + ( \lambda _ { t } + \lambda _ { t } ) V _ { k } ^ { 2 } - ( \lambda _ { g s t } + \lambda _ { g s s } ) V _ { s k } ^ { 2 } ) K } & { } & \\ { + \eta I ) ^ { - 1 } ( W ^ { 2 } Y ^ { \top } + \lambda _ { t t } V _ { t u } ^ { 2 } Y _ { u _ { t } } ^ { 2 } + ( \lambda _ { t } + \lambda _ { s } ) V _ { k } ^ { 2 } Y _ { k } ^ { 2 } } & { } \\ { - \lambda _ { g s t } V _ { s k } ^ { 2 } Y _ { u _ { t } } ^ { 2 } - \lambda _ { g s s } V _ { s k } ^ { 2 } Y _ { u _ { s } } ^ { 2 } ) } & { \quad } & { { \mathrm { ~ o ~ o ~ } } } \end{array}\tag{29}
$$

(2) Estimation of $D _ { C S } ( P _ { X ^ { S } Y ^ { S } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } | | P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } )$

After obtaining h, we use it to classify the samples of the target domain and estimate (8). Inspired by $[ 2 7 ] - [ 3 4 ]$ , we use kernel function $r ( x , y ; \rho )$ to approximate $\frac { { \bar { P } } _ { X ^ { s } Y ^ { s } } ( x _ { i } ^ { s } , y _ { i } ^ { s } ) } { P _ { X ^ { t } Y ^ { t } } ( x _ { i } ^ { s } , y _ { i } ^ { s } ) }$ . Let $\begin{array} { r } { r ( x , y ; \rho ) = \frac { P _ { X ^ { s } Y ^ { s } } ( x _ { i } ^ { s } , y _ { i } ^ { s } ) } { P _ { X ^ { t } Y ^ { t } } ( x _ { i } ^ { s } , y _ { i } ^ { s } ) } } \end{array}$ be expressed as

$$
r ( x , y ; \rho ) = \sum _ { i = 1 } ^ { m _ { t k } + m _ { s k } } \rho _ { i } \Phi _ { i } ( x , y )\tag{30}
$$

where $\rho _ { i }$ is the weight parameter. $\Phi _ { i } ( x , y ) = k ( x , x _ { i } ) l ( y , y _ { i } )$ 2 is the kernel function, where $\begin{array} { r } { k ( x , x _ { i } ) = e x p ( - \frac { | | x - x _ { i } | | ^ { 2 } } { \sigma _ { r } } ) , \sigma _ { x } } \end{array}$ is the median inter-sample distance of $m _ { t k } + m _ { s k }$ samples. $l ( y , y _ { i } )$ is the delta kernel function, $l ( y , y _ { i } ) = 1$ if and only if $y = y _ { i }$ , otherwise $l ( y , y _ { i } ) = 0$

$$
= \frac { 1 } { 2 } \int _ { \substack { \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } } [ r ( x , y ; \rho ) - \frac { P _ { X ^ { s } Y ^ { s } } ( x , y ) } { P _ { X ^ { t } Y ^ { t } } ( x , y ) } ] ^ { 2 } P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y
$$

$$
\begin{array} { l } { { = \displaystyle \frac { 1 } { 2 } \int _ { \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } r ^ { 2 } ( x , y ; \rho ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y } } \\ { { - \displaystyle \int _ { \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } r ( x , y ; \rho ) P _ { X ^ { s } Y ^ { s } } ( x , y ) d x d y + C } } \end{array}
$$

$$
= \frac { 1 } { 2 } \sum _ { i , j = 1 } ^ { m _ { t k } + m _ { s k } } \rho _ { i } \rho _ { j } \int _ { \chi _ { * } y ^ { u _ { G } } } \Phi _ { i } ( x , y ) \Phi _ { j } ( x , y ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y
$$

$$
- \sum _ { i = 1 } ^ { m _ { t k } + m _ { s k } } \rho _ { i } \int _ { \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \Phi _ { i } ( x , y ) P _ { X ^ { s } Y ^ { s } } ( x , y ) d x d y + C
$$

$$
= \frac { 1 } { 2 } \rho ^ { \top } H \rho - b ^ { \top } \rho + C\tag{31}
$$

$$
\mathrm { w h e r e ~ } H \in R ^ { ( m _ { t k } + m _ { s k } ) * ( m _ { t k } + m _ { s k } ) } , b \in R ^ { m _ { t k } + m _ { s k } } , \mathrm { a n d ~ }
$$

$$
( H ) _ { i j } = \int _ { \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \Phi _ { i } ( x , y ) \Phi _ { j } ( x , y ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y\tag{32}
$$

$$
( b ) _ { i } = \int _ { \chi _ { * } \ y ^ { u _ { G } } } \Phi _ { i } ( x , y ) P _ { X ^ { s } Y ^ { s } } ( x , y ) d x d y\tag{33}
$$

We can estimate $\rho$ by solving the following optimization problem:

$$
\rho ^ { * } = \arg \operatorname* { m i n } _ { \rho \in \mathbb { R } ^ { m _ { t k } + m _ { s k } } } \frac { 1 } { 2 } \rho ^ { \top } \hat { H } \rho - \hat { b } ^ { \top } \rho + \frac { \lambda } { 2 } \rho ^ { \top } \rho = ( \hat { H } + \lambda I ) ^ { - 1 } \hat { b }\tag{34}
$$

where ${ \scriptstyle { \frac { \lambda } { 2 } } } \rho ^ { \top } \rho$ is the penalty item avoiding overfitting,

$$
( \hat { H } ) _ { i j } = \frac { 1 } { m _ { t k } } \sum _ { k = 1 } ^ { m _ { t k } } \Phi _ { i } ( x _ { k } ^ { t } , y _ { k } ^ { t } ) \Phi _ { j } ( x _ { k } ^ { t } , y _ { k } ^ { t } )\tag{35}
$$

$$
( \hat { b } ) _ { i } = \frac { 1 } { m _ { s k } } \sum _ { k = 1 } ^ { m _ { s k } } \Phi _ { i } ( x _ { k } ^ { s } , y _ { k } ^ { s } )\tag{36}
$$

Hence,

$$
\begin{array} { c } { { D _ { C S } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } ) } } \\ { { \approx \displaystyle \frac { 1 } { m _ { s k } } \sum _ { i = 1 } ^ { m _ { s k } } m a x \{ r ( x _ { i } ^ { s } , y _ { i } ^ { s } ; ( \hat { H } + \lambda I ) ^ { - 1 } \hat { b } ) , \varepsilon \} - 1 } } \end{array}\tag{37}
$$

From (37), we know that the sample features of the source domain and target domain have important effects on $r ( x , y ; \rho )$ To obtain the better $r ( x , y ; \rho )$ , we first use a pair of projection matrices $W _ { s }$ and $W _ { t }$ to project the source samples and the target samples into the high-dimensional space, respectively. Then, we minimize $D _ { C S } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } | | P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } )$ to align the joint distribution of the source and target domains. Obviously, the optimization problem (37) can be rewritten as follows:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { W _ { s } , W _ { t } } f ( W _ { s } , W _ { t } ) } \\ { = \displaystyle \operatorname* { m i n } _ { W _ { s } , W _ { t } } \frac { 1 } { m _ { s k } } \sum _ { i = 1 } ^ { m _ { s k } } \operatorname* { m a x } \left\{ r \left( W _ { s } ^ { \top } x _ { i } ^ { s } , y _ { i } ^ { s } ; ( \hat { H } + \lambda I ) ^ { - 1 } \hat { b } \right) , \varepsilon \right\} } \end{array}
$$

$$
\it s . t . \ W _ { s } ^ { \top } W _ { s } = I , \ W _ { t } ^ { \top } W _ { t } = I\tag{38}
$$

where $W _ { s } , W _ { t } \in \mathcal { R } ^ { D * d } , d \leq D$ and

$$
\hat { H } = \frac { 1 } { m _ { s k } } \sum _ { k = 1 } ^ { m _ { s k } } \Phi _ { i } ( \boldsymbol W _ { s } ^ { \top } \boldsymbol x _ { k } ^ { s } , y _ { k } ^ { s } ) \Phi _ { j } ( \boldsymbol W _ { s } ^ { \top } \boldsymbol x _ { k } ^ { s } , y _ { k } ^ { s } )\tag{39}
$$

$$
\hat { b } = \frac { 1 } { m _ { t k } } \sum _ { k = 1 } ^ { m _ { t k } } \Phi _ { i } ( W _ { t } ^ { \top } x _ { k } ^ { t } , y _ { k } ^ { t } )\tag{40}
$$

In this study, we use Steepest Descent (SD) [35] to solve the optimization problem (38). After obtaining $r ( x , y ; \rho )$ , we fix it to calculate α by (29).

## C. Progressive Pseudo Labeling Method

In order to solve the optimization problem (11), the pseudo labels of the target samples need to be provided. In the previous researches [8], [16], [22], [23], [36], [37], progressive labeling methods have been proposed to assign high confidential pseudo labels into the target samples. Specifically, in the initial stage, they first use the source samples to train a classifier and then assign the target samples into the specific classes. In the subsequent iteration stages, based on the confidential degrees of the pseudo labels, they build the optimization model by progressively selecting a part of the target samples. Generally speaking, since there are no unknown class labels in the label space of the source domain, the classifier cannot assign unknown class labels to the target samples directly. Previous methods have largely relied on tools such as entropy and pre-classifiers as criteria for sample separation, but these approaches may overlook the spatial structure of samples, resulting in outcomes that lack interpretability. Notice that samples that have the same label tend to be more similar, so the labels of the samples can be classified based on the similarity among the samples. Specifically, if some target samples are significantly different from all the source samples, then the classifier can assign them into unknown class. In this study, SVM [38] is used as classifier. Although SVM cannot assign unknown class labels to target samples directly, SVM can provide the confidence $E ( x ^ { t } ) = \{ E _ { 1 } ( \bar { x ^ { t } } ) , E _ { 2 } ( x ^ { t } ) , . . . , E _ { C } ( x ^ { t } ) \}$ where $E _ { i }$ represents the confidence score which represents the signed distance of the target sample $x ^ { t }$ to the i-th hyperplane. Then the classes of the target samples $\{ x _ { j } ^ { t } \} _ { j = 1 } ^ { m _ { t } }$ can be determined by

$$
\hat { y } _ { j } ^ { t } = \left\{ { S V M ( x _ { j } ^ { t } ) } , \ : \ : { m a x E ( x _ { j } ^ { t } ) } \geq \tau \right.\tag{41}
$$

where $\tau$ is the confidence threshold.

In practice, if SVM is used only once, the prediction will be inaccurate because there will still be some common class samples predicted as unknown class. To deal with this problem, we design a progressive labeling method. Specifically, let the total iteration number and the current iteration number be $T$ and $t ,$ respectively. In the t-th iteration, the target domain samples whose pseudo-labels are common classes will be selected at the ratio $t / T$ . The reason why only the common class samples are selected is that the purpose of optimizing $D _ { C S } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } | | P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } )$ is to align the joint probability distributions of common classes between the source and target domains. In the t-th iteration, $M _ { t k }$ samples in each class are selected based on the ascending order of confidence as the common class samples where $M _ { t k }$ is as follows:

$$
M _ { t k } = c e i l ( m a x ( m _ { 1 } , m _ { 2 } , . . . , m _ { C } ) ( \frac { t } { T } ) )\tag{42}
$$

In particular, if $m _ { c } \le M _ { t k }$ , there will be only $m _ { c }$ samples are selected. The confidence sequence of the selected samples in the c-th class can be expressed as:

$$
\begin{array} { r l } & { l _ { c } = \left\{ \begin{array} { l l } { E ( x _ { c , 1 } ^ { t } ) , E ( x _ { c , 2 } ^ { t } ) , . . . , E ( x _ { c , m _ { c } } ^ { t } ) , } & { \mathrm { i f ~ } m _ { c } \leq M _ { t k } } \\ { E ( x _ { c , 1 } ^ { t } ) , E ( x _ { c , 2 } ^ { t } ) , . . . , E ( x _ { c , M _ { t k } } ^ { t } ) , } & { \mathrm { i f ~ } m _ { c } > M _ { t k } } \end{array} \right. } \\ & { ~ s . t . ~ E ( x _ { c , i } ^ { t } ) > E ( x _ { c , j } ^ { t } ) ~ \mathrm { i f ~ } i < j } \end{array}\tag{43}
$$

In this way, some common class samples are selected at each iteration to participate in the alignment process of the joint distributions between the source and target domains. In order to make the reader understand our idea clearly, the framework of JAUA is shown in Fig. 3.

In the previous research, Chen [39] pointed out that during the progressive pseudo-labeling process, errors in the initial pseudo-labels may propagate iteratively throughout training, potentially impacting algorithmic performance. For JAUA, the initial pseudo labels are usually also unreliable. However, they can be iteratively updated by the progressive pseudo labeling in the training phase to enhance their reliability. The theoretical analysis and discussions on error propagation issue from pseudo-labels to distribution alignment are provided in Supplementary Material. The related results show that the proposed progressive pseudo labeling method can adaptively reduce error propagation and align the joint distributions between two domains.

![](images/cc1d91f4229ff7cd5556c0fc6b7627a738604575479507a5c28fe1df6d0f4691.jpg)  
Fig. 3. The framework of JAUA. The input dataset is first classified into $1 , 2 , \ldots , C + 1$ classes using SVM. Through progressive pseudo-labeling, the dataset’s classification is progressively updated. Target domain samples with confidence E below τ are classified as $C + { \overline { { 2 } } } ,$ , and the label set is updated. This iterative training process is repeated until the final target domain classification results are obtained.

## D. Proposed Algorithm

Based on the solution procedure of the optimization problem (11), we first use SVM to obtain the pseudo labels of the target domain samples. And then Principal Component Analysis (PCA) [40] is used to reduce the dimension of $D _ { s }$ and $D _ { t }$ from $D$ to d and obtain initial $W _ { s }$ and $W _ { t }$ . According to (28), we can use $W _ { s }$ and $W _ { t }$ to get the updated $\alpha ,$ the updated classification model $h$ can be obtained based on (12). In this way, the updated pseudo-label $\hat { y } ^ { t }$ can be obtained. In the subsequent iterations, the new common class samples can be selected by the new pseudo-labels to optimize (38) and get the new $W _ { s }$ and $W _ { t }$

Based on the above analysis, the whole procedure of JAUA is summarized as 1.

Since JAUA needs to compute kernel matrix and matrix inversion in each iteration, the total time complexity and space complexity of JAUA are $O ( T ( m ^ { 3 } + S m _ { k } ^ { 3 } ) )$ and $O ( m ^ { 2 } )$ respectively. For large scale data set, we choose to use Random Fourier Features [41] and Conjugate Gradient [42] to compute kernel matrix and matrix inversion respectively. The total time complexity and space complexity can be reduced to $O ( T ( m d p + m p ^ { 2 } ) )$ and $O ( m p )$ , respectively. $p \ < < \ m$ represents the random feature dimension.

## V. EXPERIMENT

In this section, we conduct the experiments to verify the effectiveness of JAUA. JAUA algorithm is written in Python 3.9. The experiments are conducted on a computer with 13th Gen Intel(R) Core(TM) i7-13650HX.

Algorithm 1 JAUA   
Input: Source labeled dataset: $D _ { s } ~ = ~ \{ x _ { i } ^ { s } , y _ { i } ^ { s } \} _ { i = 1 } ^ { m _ { x } }$ ; Target   
unlabeled dataset: $D _ { t } ~ = ~ \{ x _ { i } ^ { t } \} _ { i = 1 } ^ { m _ { t } } ; ~ m ~ = ~ m _ { s } ~ + ~ m _ { t } ;$   
$m _ { k } = m _ { s k } + m _ { t k } ;$ Hyperparameters $\lambda _ { t t } , \lambda _ { s } , \lambda _ { t } , \lambda _ { g s s } .$   
$\lambda _ { g s t } ;$ Regular norm hyperparameters: λ, $\eta ;$ threshold of   
confidence: $\tau ;$ PCA dimension $d ;$ Total numbers of itera  
tions: $T ;$ Iterations of aligning joint distribution: S; Bound   
of relative loss $\mathcal { E } .$   
Output: Target domain predict labels: $\{ \hat { y } _ { i } ^ { t } \} _ { i = 1 } ^ { m _ { t } }$   
1: Init pseudo labels of target domain samples $\{ \hat { y } _ { i } ^ { t } \} _ { i = 1 } ^ { m _ { t } }$ by   
$S V M ( D _ { s } , D _ { t } ) ;$   
2: Divide the dataset $D _ { t }$ by (41) to obtain $D _ { t k }$ and $D _ { t u } ;$   
3: Init $W _ { s } , W _ { t }$ by $P C A ( ( X _ { s } , X _ { t } ) , d ) ;$   
4: Set the current total iteration number t to 1;   
5: Compute $M _ { t k }$ by (42);   
6: Obtain the confidence sequence $l _ { c }$ by (43);   
7: Set the current total loss $\varepsilon _ { \mathrm { 0 } }$ to $0 ;$   
8: while $t < T$ and $\varepsilon < \mathcal { E }$ do   
9: Set the iteration number of aligning joint distribution s   
to 1;   
10: while $s < S$ do   
11: Use $l _ { c }$ to optimize (38) by SD algorithm to update   
$W _ { s } , W _ { t } ;$   
12: $s \gets s + 1 ;$   
13: end while   
14: Map samples $X _ { s } , \ X _ { t }$ to high dimensional samples   
$W _ { s } X _ { s } , W _ { t } X _ { t } ;$   
15: Use $W _ { s } X _ { s } , W _ { t } X _ { t }$ to calculate $\alpha ^ { * }$ by (28);   
16: Compute the total loss $\varepsilon _ { t }$ by (37);   
17: Update $\{ \hat { y } _ { i } ^ { t } \} _ { i = 1 } ^ { m _ { t } }  \alpha ^ { * } K ;$   
18: Update $M _ { t k }$ by (42);   
19: Update the confidence sequence $l _ { c }$ by (43);   
20: $t \gets t + 1 , \varepsilon \gets ( \varepsilon _ { t } - \varepsilon _ { t - 1 } ) / \varepsilon _ { t } ;$   
21: end while   
22: return $\{ \hat { y } _ { i } ^ { t } \} _ { i = 1 } ^ { m _ { t } } ;$

DATASET DESCRIPTION
<table><tr><td rowspan=1 colspan=1>datasets</td><td rowspan=1 colspan=1>Domain</td><td rowspan=1 colspan=1>Sample number</td><td rowspan=1 colspan=1>Category</td></tr><tr><td rowspan=3 colspan=1>Office-31</td><td rowspan=1 colspan=1>Amazon</td><td rowspan=1 colspan=1>2,817</td><td rowspan=1 colspan=1>31</td></tr><tr><td rowspan=1 colspan=1>Dslr</td><td rowspan=1 colspan=1>795</td><td rowspan=1 colspan=1>31</td></tr><tr><td rowspan=1 colspan=1>Webcam</td><td rowspan=1 colspan=1>498</td><td rowspan=1 colspan=1>31</td></tr><tr><td rowspan=4 colspan=1>Office-Home</td><td rowspan=1 colspan=1>Art</td><td rowspan=1 colspan=1>2,427</td><td rowspan=1 colspan=1>65</td></tr><tr><td rowspan=1 colspan=1>Clipart</td><td rowspan=1 colspan=1>4,365</td><td rowspan=1 colspan=1>65</td></tr><tr><td rowspan=1 colspan=1>Product</td><td rowspan=1 colspan=1>4,439</td><td rowspan=1 colspan=1>65</td></tr><tr><td rowspan=1 colspan=1>RealWorld</td><td rowspan=1 colspan=1>4,357</td><td rowspan=1 colspan=1>65</td></tr><tr><td rowspan=2 colspan=1>Visda</td><td rowspan=1 colspan=1>Synthetic</td><td rowspan=1 colspan=1>150,000</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>RealWorld</td><td rowspan=1 colspan=1>50,000</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=3 colspan=1>Domainnet</td><td rowspan=1 colspan=1>Painting</td><td rowspan=1 colspan=1>151,518</td><td rowspan=1 colspan=1>345</td></tr><tr><td rowspan=1 colspan=1>Real</td><td rowspan=1 colspan=1>350,654</td><td rowspan=1 colspan=1>345</td></tr><tr><td rowspan=1 colspan=1>Sketch</td><td rowspan=1 colspan=1>140,772</td><td rowspan=1 colspan=1>345</td></tr><tr><td rowspan=3 colspan=1>ImageCLEF</td><td rowspan=1 colspan=1>Caltech</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>ImageNet</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>PASCAL VOC</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=3 colspan=1>PACS</td><td rowspan=1 colspan=1>Cartoon</td><td rowspan=1 colspan=1>4,688</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>Photo</td><td rowspan=1 colspan=1>3,340</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>Sketch</td><td rowspan=1 colspan=1>7,858</td><td rowspan=1 colspan=1>7</td></tr></table>

## A. Datasets

Six public image datasets including Office-31 [43], Office-Home [44], Visda [45], Domainnet [46], ImageCLEF [47] and PACS [48], are used for the experimental study and their overviews are provided in Table II. The relevant settings for the datasets have been kept consistent with those of the baseline algorithms.

## B. Baseline Algorithms

We use four categories of baseline algorithms for comparison: 1) algorithms based on adversarial learning and feature alignment:UAN [17], CMU [18], DANCE [20], DCC [19], OSBP [49], IWAN [50], LaFea [21], Tanet [51], GUDA [52], Mahalanobis [53], BSP-WSA [54] and GATE [55]; 2) algorithms based on classifier design and threshold strategies: OVANet [56], CPR [57], KLS [58], PCL [59], STUN [60] and UACP [61]; 3) algorithms based on mutual learning and graph structures: MLNET [62], E-MLNET [47], UniDA-CGL [63] and LIWUDA [64]; and 4) algorithms based on contrastive learning and clustering: CAN [65] and MOEO [66].

## C. Experimental evaluation metrics

The performances of all algorithms are evaluated by the average accuracy of common class ACC, the accuracy of unknown class UNK and H-score (HOS). They are defined as:

$$
A C C = \frac { 1 } { C } \sum _ { i = 1 } ^ { C } \frac { | \{ x | x \in D _ { t } ^ { i } \ a n d \ h ( x ) = y _ { i } \} | } { | \{ x | x \in D _ { t } ^ { i } \} | }\tag{44}
$$

$$
U N K = \frac { | \{ x | x \in D _ { t } ^ { C + 2 } ~ a n d ~ h ( x ) = y _ { C + 2 } \} | } { | \{ x | x \in D _ { t } ^ { C + 2 } \} | }\tag{45}
$$

$$
H O S = { \frac { 2 * A C C * U N K } { A C C + U N K } }\tag{46}
$$

where $h ( x )$ represents the predict label of the sample $x . \mid \cdot \mid$ denotes the number of elements in the set. From the definitions of these metrics, we know that their ranges are [0,1]. Based on these definitions, we know that the closer the metrics are to 1, the better the algorithm performs.

## D. Implementation Details

In this study, we use the baseline network ResNet50 to extract the data features and the initial dimension of the feature is 2048. The parameter settings in JAUA are listed in Table III. Except for T and E, the optimal values of the other parameters are obtained by the grid search.

## E. Experimental Results and Analysis

Due to the page limitation, the average results of ACC, UNK and HOS on the datasets Visda, Office-31, Office-Home, Domainnet, ImageCLEF and PACS are listed in Table IV, Table V, Table VI, Table VII, Table VIII and Table IX, respectively. Except for Visda, the detailed results on each domain adaptation task are provided in Table S-3, Table S-4, Table S-5, Table S-6 and Table S-7, respectively, which are placed in Supplementary Material. In these Tables, the bold types and the underlined types represent the best results and the second-best results of different domain adaptation tasks among the comparison algorithms, respectively. “-” indicates the experimental results which are not provided by the original algorithms. In order to show JAUA’s scalability, the runtime of JAUA on six datsets are reported in Table S-2 of Supplementary Material.

From Table IV to IX, we can find that compared with the baseline algorithms, the average ACC, UNK and HOS of JAUA on all of the datasets are optimal. The main reason why JAUA obtains the better results is as follows. On the one hand, JAUA aligns the joint probability distributions between the common classes of the source and target domains by minimizing Chi-Square divergence and learns more excellent feature representation. On the other hand, the baseline algorithms usually identify the unknown class samples by aligning the marginal distributions or the conditional distributions of the common classes.

From Table S-3 of Supplementary Material, we can find that on the tasks of D→A and D→W of the dataset Office-31 , in terms of UNK, JAUA shows more significant improvements. The main reason is that D only includes low-noise samples. When taking it as the source domain, JAUA can easily identify the unknown class samples.

From Table S-4 of Supplementary Material, we can find that on the tasks of Ar→Cl, Ar→Pr and Ar→Rw of the dataset Office-Home , in terms of UNK, JAUA also shows more significant improvements. The main reason is that the samples in Ar primarily consist of sketches. When taking it as the source domain, JAUA can more effectively identify latent features of the unknown class samples. On the tasks of Ar→Pr, Cl→Pr and Rw→Pr of the dataset Office-Home , in terms of ACC, JAUA obtains the best results. This may be attributed to Pr only containing background-free images with minimal redundant information. When taking it as the target domain, JAUA can easily recognize the common classes. When taking Rw as the target domain, all metrics of JAUA are improved. This indicates JAUA’s strong classification capability for realworld images, reflecting its superior practical application potential in real-world scenarios.

TABLE III  
THE PARAMETER SETTINGS IN JAUA
<table><tr><td rowspan=1 colspan=1>Parameters</td><td rowspan=1 colspan=1>Office-31</td><td rowspan=1 colspan=1>Office-Home</td><td rowspan=1 colspan=1>Visda</td><td rowspan=1 colspan=1>Domainnet</td><td rowspan=1 colspan=1>ImageCLEF</td><td rowspan=1 colspan=1>PACS</td></tr><tr><td rowspan=1 colspan=1> $\lambda _ { t t }$ </td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \lambda _ { s } } }$ </td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \lambda _ { t } } }$ </td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1> $\lambda _ { g s s }$ </td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>0.3</td></tr><tr><td rowspan=1 colspan=1> $\lambda _ { g s t }$ </td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>0.3</td></tr><tr><td rowspan=1 colspan=1>λ</td><td rowspan=1 colspan=1>0.0001</td><td rowspan=1 colspan=1>0.0001</td><td rowspan=1 colspan=1>0.0001</td><td rowspan=1 colspan=1>0.0001</td><td rowspan=1 colspan=1>0.0001</td><td rowspan=1 colspan=1>0.0001</td></tr><tr><td rowspan=1 colspan=1>η</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1>0.00015</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1>0.00005</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1>0.005</td></tr><tr><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>-0.1</td><td rowspan=1 colspan=1>-0.5</td><td rowspan=1 colspan=1>-0.1</td><td rowspan=1 colspan=1>-0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>S</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>30</td></tr><tr><td rowspan=1 colspan=1>d</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>300</td></tr><tr><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>20</td></tr><tr><td rowspan=1 colspan=1>ε</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>1%</td></tr></table>

TABLE IV  
COMPARISON OF THE RESULTS ON VISDA DATASET(UNIT:%)
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>ACC</td><td rowspan=1 colspan=1>UNK</td><td rowspan=1 colspan=1>HOS</td></tr><tr><td rowspan=1 colspan=1>UAN</td><td rowspan=1 colspan=1>60.8</td><td rowspan=1 colspan=1>20.4</td><td rowspan=1 colspan=1>30.5</td></tr><tr><td rowspan=1 colspan=1>CMU</td><td rowspan=1 colspan=1>61.4</td><td rowspan=1 colspan=1>24.1</td><td rowspan=1 colspan=1>34.6</td></tr><tr><td rowspan=1 colspan=1>DANCE</td><td rowspan=1 colspan=1>69.2</td><td rowspan=1 colspan=1>31.0</td><td rowspan=1 colspan=1>42.8</td></tr><tr><td rowspan=1 colspan=1>DCC</td><td rowspan=1 colspan=1>64.2</td><td rowspan=1 colspan=1>32.3</td><td rowspan=1 colspan=1>43.0</td></tr><tr><td rowspan=1 colspan=1>OSBP</td><td rowspan=1 colspan=1>30.3</td><td rowspan=1 colspan=1>49.9</td><td rowspan=1 colspan=1>37.7</td></tr><tr><td rowspan=1 colspan=1>IWAN</td><td rowspan=1 colspan=1>58.7</td><td rowspan=1 colspan=1>18.0</td><td rowspan=1 colspan=1>27.6</td></tr><tr><td rowspan=1 colspan=1>OVANet</td><td rowspan=1 colspan=1>75.3</td><td rowspan=1 colspan=1>41.0</td><td rowspan=1 colspan=1>53.1</td></tr><tr><td rowspan=1 colspan=1>LIWUDA</td><td rowspan=1 colspan=1>64.2</td><td rowspan=1 colspan=1>27.6</td><td rowspan=1 colspan=1>38.6</td></tr><tr><td rowspan=1 colspan=1>CPR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>58.2</td></tr><tr><td rowspan=1 colspan=1>LaFea</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>58.9</td></tr><tr><td rowspan=1 colspan=1>Tanet</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>60.1</td></tr><tr><td rowspan=1 colspan=1>UniDA-CGL</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>57.9</td></tr><tr><td rowspan=1 colspan=1>Mahalanobis</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>56.8</td></tr><tr><td rowspan=1 colspan=1>BSP-WSA</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>57.1</td></tr><tr><td rowspan=1 colspan=1>CAN</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>61.6</td></tr><tr><td rowspan=1 colspan=1>MOEO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>60.7</td></tr><tr><td rowspan=1 colspan=1>GATE</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>56.4</td></tr><tr><td rowspan=1 colspan=1>KLS</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>54.7</td></tr><tr><td rowspan=1 colspan=1>UACP</td><td rowspan=1 colspan=1>I</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>60.1</td></tr><tr><td rowspan=1 colspan=1>JAUA</td><td rowspan=1 colspan=1>75.5</td><td rowspan=1 colspan=1>52.6</td><td rowspan=1 colspan=1>62.0</td></tr></table>

TABLE V  
COMPARISON OF THE RESULTS ON OFFICE-31 DATASET(UNIT:%)
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>ACC</td><td rowspan=1 colspan=1>UNK</td><td rowspan=1 colspan=1>HOS</td></tr><tr><td rowspan=1 colspan=1>UAN</td><td rowspan=1 colspan=1>89.3</td><td rowspan=1 colspan=1>49.3</td><td rowspan=1 colspan=1>63.5</td></tr><tr><td rowspan=1 colspan=1>CMU</td><td rowspan=1 colspan=1>91.1</td><td rowspan=1 colspan=1>61.1</td><td rowspan=1 colspan=1>73.1</td></tr><tr><td rowspan=1 colspan=1>DANCE</td><td rowspan=1 colspan=1>93.0</td><td rowspan=1 colspan=1>78.2</td><td rowspan=1 colspan=1>84.7</td></tr><tr><td rowspan=1 colspan=1>DCC</td><td rowspan=1 colspan=1>93.2</td><td rowspan=1 colspan=1>70.7</td><td rowspan=1 colspan=1>80.2</td></tr><tr><td rowspan=1 colspan=1>IWAN</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>36.7</td><td rowspan=1 colspan=1>51.6</td></tr><tr><td rowspan=1 colspan=1>OVANet</td><td rowspan=1 colspan=1>84.2</td><td rowspan=1 colspan=1>88.0</td><td rowspan=1 colspan=1>85.9</td></tr><tr><td rowspan=1 colspan=1>LIWUDA</td><td rowspan=1 colspan=1>91.6</td><td rowspan=1 colspan=1>74.5</td><td rowspan=1 colspan=1>82.1</td></tr><tr><td rowspan=1 colspan=1>PCL</td><td rowspan=1 colspan=1>76.5</td><td rowspan=1 colspan=1>83.3</td><td rowspan=1 colspan=1>78.4</td></tr><tr><td rowspan=1 colspan=1>CPR</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>88.8</td></tr><tr><td rowspan=1 colspan=1>LaFea</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>90.5</td></tr><tr><td rowspan=1 colspan=1>STUN</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>89.8</td></tr><tr><td rowspan=1 colspan=1>KLS</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>88.5</td></tr><tr><td rowspan=1 colspan=1>UACP</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>91.0</td></tr><tr><td rowspan=1 colspan=1>JAUA</td><td rowspan=1 colspan=1>93.4</td><td rowspan=1 colspan=1>89.0</td><td rowspan=1 colspan=1>91.1</td></tr></table>

## F. Parameter Sensitivity Analysis

JAUA includes five hyperparameters $\lambda _ { t t } , \lambda _ { s } , \lambda _ { t } , \lambda _ { g s s }$ and $\lambda _ { g s t }$ . In this subsection, we explore the effects of these hyperparameters on JAUA by fixing three parameters and adjusting the others. The experiments are conducted on the dataset Office-31.

Due to space limitation, we only give the results for the task

COMPARISON OF THE RESULTS ON OFFICE-HOME DATASET(UNIT:%)  
TABLE VI
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>ACC</td><td rowspan=1 colspan=1>UNK</td><td rowspan=1 colspan=1>HOS</td></tr><tr><td rowspan=1 colspan=1>UAN</td><td rowspan=1 colspan=1>77.0</td><td rowspan=1 colspan=1>45.2</td><td rowspan=1 colspan=1>56.6</td></tr><tr><td rowspan=1 colspan=1>CMU</td><td rowspan=1 colspan=1>77.2</td><td rowspan=1 colspan=1>52.0</td><td rowspan=1 colspan=1>61.6</td></tr><tr><td rowspan=1 colspan=1>DANCE</td><td rowspan=1 colspan=1>80.3</td><td rowspan=1 colspan=1>54.0</td><td rowspan=1 colspan=1>63.9</td></tr><tr><td rowspan=1 colspan=1>DCC</td><td rowspan=1 colspan=1>78.9</td><td rowspan=1 colspan=1>68.4</td><td rowspan=1 colspan=1>72.0</td></tr><tr><td rowspan=1 colspan=1>OSBP</td><td rowspan=1 colspan=1>64.6</td><td rowspan=1 colspan=1>34.5</td><td rowspan=1 colspan=1>44.5</td></tr><tr><td rowspan=1 colspan=1>IWAN</td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=1>33.0</td><td rowspan=1 colspan=1>45.3</td></tr><tr><td rowspan=1 colspan=1>OVANet</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>71.2</td></tr><tr><td rowspan=1 colspan=1>LIWUDA</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>62.7</td></tr><tr><td rowspan=1 colspan=1>PCL</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>70.3</td></tr><tr><td rowspan=1 colspan=1>CPR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>72.3</td></tr><tr><td rowspan=1 colspan=1>LaFea</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>73.3</td></tr><tr><td rowspan=1 colspan=1>STUN</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>73.8</td></tr><tr><td rowspan=1 colspan=1>KLS</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>74.6</td></tr><tr><td rowspan=1 colspan=1>UACP</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>74.7</td></tr><tr><td rowspan=1 colspan=1>GUDA</td><td rowspan=1 colspan=1>70.4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>JAUA</td><td rowspan=1 colspan=1>80.4</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>75.4</td></tr></table>

![](images/d4384309f13e2c1a711f67128571cddf4d88a76d8f11eb6962c2accd9a8dbf8a.jpg)  
Fig. 4. Influence of the parameters of JAUA for task: D→A on Office-31

D→A. We fix $\lambda _ { t t } = 1 , \lambda _ { s } = 0 . 7$ and $\lambda _ { t } = 0 . 7$ . The changes of HOS with respect to $\lambda _ { g s s }$ and $\lambda _ { g s t }$ are illustrated in Fig. 4. Fig. 4 shows a three-dimensional histogram of HOS as parameters $\lambda _ { g s s }$ and $\lambda _ { g s t }$ vary. Higher bars indicate HOS closer to 1, signifying better domain adaptation performance. From Fig. 4 we find that JAUA is sensitive to the parameters $\lambda _ { g s s }$ and $\lambda _ { g s t } .$ The reason is that the parameter $\lambda _ { g s t }$ affects the importance of loss of the common class samples being misclassified as unknown class on the source domain while the parameter $\lambda _ { g s s }$ affects the importance of loss of the common class samples being misclassified as the private class on the source domain. Fig. 4 shows that the HOS value increases as $\lambda _ { g s s }$ and $\lambda _ { g s t }$ increase. However, when $\lambda _ { g s t } = 0 . 4$ and $\lambda _ { g s s } = 0 . 1$ , HOS drops dramatically. This occurs because this setup violates the restriction $\lambda _ { g s t } + \lambda _ { g s s } \le m _ { s k } / m _ { s }$ in Theorem 2. On the other tasks of the other datasets, we find that JAUA is also sensitive to the other parameters. In real-world applications, we usually use the grid search to obtain the optimal hyperparameters.

COMPARISON OF THE RESULTS ON DOMAINNET DATASET(UNIT:%)
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>HOS</td></tr><tr><td rowspan=1 colspan=1>UAN</td><td rowspan=1 colspan=1>41.0</td></tr><tr><td rowspan=1 colspan=1>CMU</td><td rowspan=1 colspan=1>48.3</td></tr><tr><td rowspan=1 colspan=1>DANCE</td><td rowspan=1 colspan=1>33.5</td></tr><tr><td rowspan=1 colspan=1>DCC</td><td rowspan=1 colspan=1>49.2</td></tr><tr><td rowspan=1 colspan=1>OSBP</td><td rowspan=1 colspan=1>32.0</td></tr><tr><td rowspan=1 colspan=1>IWAN</td><td rowspan=1 colspan=1>32.8</td></tr><tr><td rowspan=1 colspan=1>OVANet</td><td rowspan=1 colspan=1>50.7</td></tr><tr><td rowspan=1 colspan=1>LIWUDA</td><td rowspan=1 colspan=1>48.6</td></tr><tr><td rowspan=1 colspan=1>PCL</td><td rowspan=1 colspan=1>41.3</td></tr><tr><td rowspan=1 colspan=1>LaFea</td><td rowspan=1 colspan=1>51.9</td></tr><tr><td rowspan=1 colspan=1>Tanet</td><td rowspan=1 colspan=1>57.4</td></tr><tr><td rowspan=1 colspan=1>CAN</td><td rowspan=1 colspan=1>52.1</td></tr><tr><td rowspan=1 colspan=1>MOEO</td><td rowspan=1 colspan=1>52.6</td></tr><tr><td rowspan=1 colspan=1>GATE</td><td rowspan=1 colspan=1>52.1</td></tr><tr><td rowspan=1 colspan=1>KLS</td><td rowspan=1 colspan=1>51.8</td></tr><tr><td rowspan=1 colspan=1>UACP</td><td rowspan=1 colspan=1>50.3</td></tr><tr><td rowspan=1 colspan=1>JAUA</td><td rowspan=1 colspan=1>58.1</td></tr></table>

TABLE VIII  
COMPARISON OF THE RESULTS ON IMAGECLEF DATASET(UNIT:%)
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>HOS</td></tr><tr><td rowspan=1 colspan=1>OVANet</td><td rowspan=1 colspan=1>74.3</td></tr><tr><td rowspan=1 colspan=1>MLNET</td><td rowspan=1 colspan=1>83.3</td></tr><tr><td rowspan=1 colspan=1>E-MLNET</td><td rowspan=1 colspan=1>85.4</td></tr><tr><td rowspan=1 colspan=1>JAUA</td><td rowspan=1 colspan=1>89.4</td></tr></table>

TABLE IX  
COMPARISON OF THE RESULTS ON PACS DATASET(UNIT:%)
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>ACC</td></tr><tr><td rowspan=1 colspan=1>UAN</td><td rowspan=1 colspan=1>30.7</td></tr><tr><td rowspan=1 colspan=1>CMU</td><td rowspan=1 colspan=1>21.0</td></tr><tr><td rowspan=1 colspan=1>DANCE</td><td rowspan=1 colspan=1>17.6</td></tr><tr><td rowspan=1 colspan=1>OVANet</td><td rowspan=1 colspan=1>58.5</td></tr><tr><td rowspan=1 colspan=1>GUDA</td><td rowspan=1 colspan=1>58.2</td></tr><tr><td rowspan=1 colspan=1>JAUA</td><td rowspan=1 colspan=1>64.2</td></tr></table>

## G. Convergence Analysis

In this subsection, we discuss the convergence of JAUA. Based on Algorithm 1, JAUA must set a maximum number of iterations $T$ to ensure the execution of the algorithm. In order to explore the convergence of the JAUA algorithm, a larger maximum number of iterations T is determined in advance. The algorithm will stop when the relative loss change of two consecutive iterations is less than 1%. Fig. 5 shows the convergence of JAUA for task: D→A of the dataset Office-31. We can also obtain similar conclusion for the other tasks on the other datasets.

![](images/f00a8f6ecc7b3bad65cf314f1ce964e1f6ca093ad0b05bd3fc90fa7c1f5ba0d4.jpg)  
Fig. 5. The change of the whole loss of JAUA for task: D→A on Office-31

![](images/4242335ef0103594f4bbf60c7ae1f3f22918636cb39d9aede3fef60023428996.jpg)  
Fig. 6. Ablation experiment for task: D→A on Office-31

## H. Ablation Study

To study the contribution of each component in the proposed model, we conduct the ablation experiment for task: D→A on the Office-31 dataset. The results are illustrated in Fig. 6, where “wo X” means removing the loss term X from the whole loss function.

From Fig. 6, we can find that all of the components contribute to improving HOS of JAUA, where $R _ { u _ { t } } ^ { t u }$ and $R _ { u _ { t } } ^ { s k }$ are the two most important components for JAUA. This result implies that compared to UDA, the presence of unknown class samples makes it crucial to distinguish common and private class samples from unknown ones as much as possible, which in turn helps improve the performance of JAUA.

## VI. CONCLUSIONS AND FUTURE WORK

In this paper, based on joint distribution alignment, a generalization error bound of the target domain is for the first time proposed for UniDA. A novel UniDA model is built by minimizing this bound. Based on a progressive pseudolabeling strategy, JAUA is designed to identify known, private, and unknown class samples. The experiments show that JAUA can solve UniDA problem well.

In future work, we would like to further explore the intraclass structure of the unknown classes in the target domain.

We expect to provide some novel error bounds and models by classifying the unknown class samples in the target domain into different unknown classes, which may be able to further improve the generalization ability of the model.

## APPENDIX A. PROOF OF THEOREM 1

Proof:

We adopt an assumption below:

Assumption 1 [67]: For any loss function l, l is upper bounded, i.e. for any space $( { \mathcal { X } } * { \mathcal { Y } } ) , { \mathcal { \exists } } M > 0 , { \forall } x \in { \mathcal { X } } ,$ $y \in \mathcal { V } , l ( h ( x ) , y ) \leq M$

The specific definitions of $R ^ { t u } ( h ) , \ R ^ { t k } ( h ) , \ R ^ { s u } ( h )$ and $R ^ { s k } ( h )$ are as follows:

$$
R ^ { t u } ( h ) = \int _ { \chi _ { * } y ^ { u _ { t } } } l ( h ( x ) , y ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y\tag{47}
$$

$$
\begin{array} { l } { { R ^ { t k } ( h ) = R ^ { t } ( h ) - R ^ { t u } ( h ) } } \\ { { \displaystyle \qquad = \int _ { \chi _ { * } \ y ^ { u _ { G } } } l ( h ( x ) , y ) P _ { X ^ { t } Y ^ { t } } ( x , y ) d x d y } } \end{array}\tag{48}
$$

$$
R ^ { s u } ( h ) = \int _ { \chi _ { * } \ y ^ { u _ { s } } } l ( h ( x ) , y ) P _ { X ^ { s } Y ^ { s } } ( x , y ) d x d y\tag{49}
$$

$$
\begin{array} { l } { { R ^ { s k } ( h ) = R ^ { s } ( h ) - R ^ { s u } ( h ) } } \\ { { \displaystyle \quad = \int _ { \chi _ { * } y ^ { u _ { G } } } l ( h ( x ) , y ) P _ { X ^ { s } Y ^ { s } } ( x , y ) d x d y } } \end{array}\tag{50}
$$

Based on Assumption 1, according to (48) and (50), we have

$$
\begin{array} { r l } & { | R ^ { t k } ( h ) - R ^ { s k } ( h ) | } \\ & { = \Big | \int _ { \mathcal { X } * y ^ { u _ { G } } } l ( h ( x ) , y ) \left[ P _ { X ^ { t } Y ^ { t } } ( x , y ) - P _ { X ^ { s } Y ^ { s } } ( x , y ) \right] d x d y \Big | } \\ & { \leq M \left| \int _ { \mathcal { X } * y ^ { u _ { G } } } \left[ P _ { X ^ { t } Y ^ { t } } ( x , y ) - P _ { X ^ { s } Y ^ { s } } ( x , y ) \right] d x d y \right| } \\ & { \leq M D _ { T V } \left( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \right) } \end{array}\tag{51}
$$

where $D _ { T V }$ represents the Total Variation.

From (51), we know that the following two inequalities hold:

$$
\begin{array} { r l } & { R ^ { t k } ( h ) } \\ & { \leq R ^ { s k } ( h ) + M D _ { T V } \left( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \Vert P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \right) } \end{array}\tag{52}
$$

$$
\begin{array} { r l } & { R ^ { s k } ( h ) } \\ & { \leq R ^ { t k } ( h ) + M D _ { T V } \left( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \Vert P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \right) } \\ & { \mathrm { C o m b i n i n g ~ ( 4 ) , ~ ( 6 ) ~ w i t h ~ ( 4 7 ) , ~ w e ~ c a n ~ o b t a i n } } \end{array}\tag{53}
$$

$$
\begin{array} { l } { { \displaystyle R ^ { t u } ( h ) } } \\ { { = \int _ { \boldsymbol { \mathcal { X } } _ { \ast \boldsymbol { y } } \boldsymbol { y } \boldsymbol { u } _ { t } }  { l } ( h ( \boldsymbol { x } ) , \boldsymbol { y } ) P _ { X ^ { t } Y ^ { t } } ( \boldsymbol { x } , \boldsymbol { y } ) d \boldsymbol { x } d \boldsymbol { y }  } } \\ { { \displaystyle = R _ { \boldsymbol { u } _ { t } } ^ { t } ( h ) - R _ { \boldsymbol { u } _ { t } } ^ { s k } ( h ) + } } \\ { { \displaystyle  \int _ { \boldsymbol { \mathcal { X } } _ { \ast \boldsymbol { y } } \boldsymbol { y } ^ { u } \boldsymbol { G } }  { l } ( h ( \boldsymbol { x } ) , \boldsymbol { y } ^ { u _ { t } } ) [ P _ { X ^ { s } Y ^ { s } } ( \boldsymbol { x } , \boldsymbol { y } ) - P _ { X ^ { t } Y ^ { t } } ( \boldsymbol { x } , \boldsymbol { y } ) ] d \boldsymbol { x } d \boldsymbol { y }  } } \\ { { \le \lambda _ { t t } R _ { \boldsymbol { u } _ { t } } ^ { t } ( h ) - \lambda _ { g s t } R _ { \boldsymbol { u } _ { t } } ^ { s k } ( h ) } } \\ { { \displaystyle  + N _ { T } D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } \ast \boldsymbol { y } ^ { u _ { G } } } ) [ P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } \ast \boldsymbol { y } ^ { u _ { G } } } )  } } \end{array}\tag{54}
$$

where $\lambda _ { g s t } \le 1$ and $\lambda _ { t t } \geq 1$ are two hyperparameters which control the importance of $R _ { u _ { t } } ^ { s k } ( h )$ and $R _ { u _ { t } } ^ { t } ( h )$

Combining (48), (52) with (54), we can obtain

$$
\begin{array} { r l } & { R ^ { t } ( h ) = R ^ { t u } ( h ) + R ^ { t k } ( h ) } \\ & { \qquad \leq \lambda _ { t t } R _ { u } ^ { t } ( h ) - \lambda _ { g s t } R _ { u _ { t } } ^ { s k } ( h ) } \\ & { \qquad + N _ { T } D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } }  ) + } \\ & { \qquad R ^ { s k } ( h ) + M D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } }  )  } \\ & { \qquad = \lambda _ { t t } R _ { u _ { t } } ^ { t } ( h ) - \lambda _ { g s t } R _ { u _ { t } } ^ { s k } ( h ) + R ^ { s k } ( h )  \qquad \quad } \\ & { \qquad +  ( M + N _ { T } ) D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } )  } \end{array}\tag{55}
$$

Similarly, combining (7), (5) with (49), we have

$$
\begin{array} { r l } & { \quad R ^ { s u } ( h ) } \\ & { = \displaystyle \int _ { \mathcal { X } * \mathcal { Y } ^ { u _ { s } } } l ( h ( x ) , y ) P _ { X ^ { s } Y ^ { s } } ( x , y ) d x d y } \\ & { = R _ { u _ { s } } ^ { s } ( h ) - R _ { u _ { s } } ^ { t k } ( h ) } \\ & { \quad - \displaystyle \int _ { \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } l ( h ( x ) , y ^ { u _ { s } } ) \left[ P _ { X ^ { s } Y ^ { s } } ( x , y ) - P _ { X ^ { t } Y ^ { t } } ( x , y ) \right] d x d y } \\ & { \geq \lambda _ { g s } R _ { u _ { s } } ^ { s } ( h ) - R _ { u _ { s } } ^ { t k } ( h ) } \\ & { \quad - N _ { S } D _ { T V } \left( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \right) \left[ P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \right) } \end{array}\tag{56}
$$

where $\lambda _ { g s s } ~ \leq ~ 1$ is a hyperparameter which controls the importance of $R _ { u _ { \mathrm { s } } } ^ { s } ( h )$

Combining (50), (52) with (56), we have

$$
\begin{array} { r l } & { \quad R ^ { t k } ( h ) } \\ & { \leq R ^ { s k } ( h ) + M D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } + \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } + \mathcal { Y } ^ { u _ { G } } } ) } \\ & { \leq R ^ { s k } ( h ) + M D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } + \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } ) } \\ & { \quad + R ^ { s u } ( h ) - \lambda _ { g s } \ / R _ { u _ { s } } ^ { s } ( h ) + R _ { u _ { s } } ^ { t k } ( h ) } \\ & { \quad + N _ { S } D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } ) } \\ & { = R ^ { s } ( h ) + R _ { u _ { s } } ^ { t k } ( h ) - \lambda _ { g s } \ / s _ { u _ { s } } \ / R _ { u _ { s } } } \\ & { \quad \quad + ( M + N _ { S } ) D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } \ast \mathcal { Y } ^ { u _ { G } } } ) } \end{array}\tag{57}
$$

From [68], we know that the following inequality holds:

$$
\begin{array} { r l } & { D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } ) } \\ & { \leq \sqrt { 2 D _ { C S } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } * \mathcal { Y } ^ { u _ { G } } } ) } } \end{array}\tag{58}
$$

Using (53),(55),(57) and (58), we can obtain

$$
\begin{array} { r l } & { \quad R ^ { t } ( h ) } \\ & { \leq \lambda _ { t t } R _ { u _ { t } } ^ { t } ( h ) - \lambda _ { g s t } R _ { u _ { t } } ^ { s k } ( h ) + R ^ { s k } ( h ) } \\ & { \quad + ( M + N _ { T } ) D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \lambda _ { s } \gamma ^ { s } u _ { G } } \| P _ { X ^ { t } Y ^ { t } \sim X ^ { s } \times \mathcal { Y } ^ { u _ { G } } } ) } \\ & { \leq \lambda _ { t t } R _ { u _ { t } } ^ { t } ( h ) - \lambda _ { g s t } R _ { u _ { t } } ^ { s k } ( h ) + R ^ { t k } ( h ) } \\ & { \quad + ( 2 M + N _ { T } ) D _ { T V } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } ^ { s } \times \mathcal { Y } ^ { u _ { G } } } \| P _ { X ^ { t } Y ^ { t } \sim \mathcal { X } \times \mathcal { Y } ^ { u _ { G } } } ) } \\ & { \leq R ^ { s } ( h ) + \lambda _ { t t } R _ { u _ { t } } ^ { t } ( h ) - \lambda _ { g s t } R _ { u _ { t } } ^ { s k } ( h ) } \\ & { \quad + R _ { u _ { s } } ^ { t k } ( h ) - \lambda _ { g s s } R _ { u _ { s } } ^ { s } ( h ) + ( 3 M + N _ { T } + N _ { S } ) } \\ & { \quad * \sqrt { 2 D _ { C S } ( P _ { X ^ { s } Y ^ { s } \sim \mathcal { X } ^ { s } \times \mathcal { Y } ^ { u _ { G } } } )  { P } _ { X ^ { t } Y ^ { t } \sim \mathcal { Y } ^ { u _ { G } } } ) ^ { u _ { G } } } } \end{array}\tag{59}
$$

As stated in the Problem Setting, the goal of UniDA is to minimize $R ^ { t } ( h )$ . Based on (4) and (5), we know that in order to minimize $R ^ { t } ( h )$ , we need to minimize $R ^ { s } ( h ) , R _ { u _ { t } } ^ { t u } ( h )$ $R _ { u _ { t } } ^ { t k } ( h )$ and $R _ { u _ { s } } ^ { t k } ( h )$ and maximize $R _ { u _ { t } } ^ { s k } ( h ) , R _ { u _ { s } } ^ { s u } ( h )$ and $R _ { u _ { s } } ^ { s \tilde { k } } ( h )$ . Generally speaking, minimizing $R ^ { s } ( h )$ and $R _ { u _ { t } } ^ { t u } ( h )$

can enable the classifier to more accurately classify source samples and unknown class samples. Similarly, maximizing $R _ { u _ { t } } ^ { s k } ( h )$ and $R _ { u _ { s } } ^ { s k } ( h )$ can enable the classifier to more accurately classify the common class samples. Unfortunately, minimizing $R _ { u _ { t } } ^ { t k } ( h )$ and $R _ { u _ { s } } ^ { t k } ( h )$ ) may result in that the classifier classifies the common class samples from the target domain into unknown class or private classes. In the same way, maximizing $R _ { u _ { s } } ^ { s u } ( h )$ may result in that the classifier classifies the private class samples from the source domain into common classes. In order to avoid these cases, in the following, we update the upper bound in (59). Based on the law of large numbers, we can easily find that $R _ { u _ { t } } ^ { t k } ( h )$ and $R _ { u _ { \mathrm { s } } } ^ { t k } ( h )$ are finite, there exist positive real numbers A and B which satisfy $\begin{array} { r } { R _ { u _ { t } } ^ { t k } ( h ) \leq \frac { \lambda _ { t } } { \lambda _ { t t } } R _ { k } ^ { t \hat { k } } ( h ) + \frac { A } { \lambda _ { t t } } } \end{array}$ and $R _ { u _ { s } } ^ { t k } ( h ) \leq \lambda _ { s } R _ { k } ^ { t k } ( h ) + \dot { B }$ where $\lambda _ { t }$ and $\lambda _ { s }$ are hyperparameters which bound $R _ { u _ { t } } ^ { t k } ( h )$ and $R _ { u _ { s } } ^ { t k } ( h )$

Based on the above analysis, the novel bound of $R ^ { t } ( h )$ can be rewritten as:

$$
\begin{array} { r l } & { ~ H ^ { \{ \bar { t } } } ( h )  \\ & { \leq H ^ { s } ( h ) + \lambda _ { \mathrm { e } } R _ { n _ { \mathrm { s } } } ^ { t } ( h ) - \lambda _ { \mathrm { o } \varepsilon } R _ { n _ { \mathrm { s } } } ^ { s , k } ( h ) } \\ & { \quad + R _ { n _ { \mathrm { s } } } ^ { t } ( h ) - \lambda _ { \mathrm { J } , \mathrm { s } } R _ { n _ { \mathrm { s } } } ^ { s } ( h ) + ( 3 M + N _ { T } + N _ { S } ) } \\ & { \quad + \sqrt { 2 D _ { C } \delta ( P _ { X } \gamma \times \gamma _ { \mathrm { s } } , X _ { \mathrm { e } } \gamma _ { \mathrm { s } } ) \gamma _ { \mathrm { s } } } } \\ & { = R ^ { s } ( h ) + \lambda _ { \mathrm { t } } R _ { n _ { \mathrm { t } } } ^ { t } ( h ) + \lambda _ { \mathrm { t } } R _ { n _ { \mathrm { t } } } ^ { s } ( h ) - \lambda _ { \mathrm { g } \varepsilon } R _ { n _ { \mathrm { s } } } ^ { t } ( h ) + R _ { n _ { \mathrm { s } } } ^ { t k } ( h ) } \\ & { \quad - \lambda _ { \mathrm { g } \times B } R _ { n _ { \mathrm { s } } } ^ { s , u } ( h ) - \lambda _ { \mathrm { g } \times B } R _ { n _ { \mathrm { s } } } ^ { s , k } ( h ) + ( 3 M + N _ { T } + N _ { S } ) } \\ & { \quad + \sqrt { 2 D _ { C } \delta ( P _ { X } \gamma \times \gamma _ { \mathrm { s } } , X _ { \mathrm { g } } \gamma _ { \mathrm { s } } ) \gamma _ { \mathrm { s } } } } \\ & { \leq H ^ { s } ( h ) + \lambda _ { \mathrm { t } } R _ { n _ { \mathrm { t } } } ^ { s } ( h ) + ( \lambda _ { \mathrm { t } } + \lambda _ { \mathrm { s } } ) R _ { n _ { \mathrm { t } } } ^ { s } ( h ) } \\ &  \quad - \lambda _ { \mathrm { g } \times B } R _ { n _ { \mathrm { s } } } ^  s \end{array}\tag{60}
$$

Therefore, Theorem 1 has been proved.

## APPENDIX B. PROOF OF THEOREM 2

Proof: Let

$$
\begin{array} { r l } & { \mathcal L ( \alpha ) } \\ & { \ = \underset { \alpha \in \mathcal R ^ { ( m _ { s } + m _ { t } ) * ( C + 2 ) } } { \mathrm { m i n } } \eta t r ( \alpha ^ { \top } K \alpha ) + \| ( Y - \alpha ^ { \top } K ) W \| _ { 2 } ^ { 2 } } \\ & { + \lambda _ { t t } \| ( Y _ { u _ { t } } - \alpha ^ { \top } K ) V _ { t u } \| _ { 2 } ^ { 2 } + ( \lambda _ { s } + \lambda _ { t } ) \| ( Y _ { k } - \alpha ^ { \top } K ) V _ { k } \| _ { 2 } ^ { 2 } } \\ & { - \lambda _ { g s t } \| ( Y _ { u _ { t } } - \alpha ^ { \top } K ) V _ { s k } \| _ { 2 } ^ { 2 } - \lambda _ { g s s } \| ( Y _ { u _ { s } } - \alpha ^ { \top } K ) V _ { s k } \| _ { 2 \epsilon } ^ { 2 } . } \end{array}\tag{61}
$$

if and only if $\frac { \partial ^ { 2 } { \mathcal { L } } ( \alpha ) } { \partial \alpha ^ { 2 } } \geq 0 ,$ , the solution of $\begin{array} { r } { \frac { \partial \mathcal { L } ( \alpha ) } { \partial \alpha } = 0 } \end{array}$ can make $\mathcal { L } ( \alpha )$ to reach minimum.

From

$$
\begin{array} { l } { \displaystyle \frac { \partial \mathcal { L } ( \alpha ) } { \partial \alpha } = - 2 K W ^ { 2 } Y ^ { \top } + 2 K W ^ { 2 } K \alpha - 2 \lambda _ { t t } K V _ { t u } ^ { 2 } Y _ { u _ { t } } ^ { \top } } \\ { \displaystyle \phantom { \frac { \partial \mathcal { L } ( \alpha ) } { \partial \alpha } = - 2 k { W } V _ { t u } ^ { 2 } K \alpha - 2 ( \lambda _ { s } + \lambda _ { t } ) K V _ { k } ^ { 2 } Y _ { k } ^ { \top } } } \\ { \displaystyle \phantom { \frac { \partial \mathcal { L } ( \alpha ) } { \partial \alpha } = - 2 ( \lambda _ { s } + \lambda _ { t } ) K V _ { k } ^ { 2 } K \alpha + 2 \lambda _ { g s t } K V _ { s k } ^ { 2 } Y _ { u _ { t } } ^ { \top } } } \\ { \displaystyle \phantom { \frac { \partial \mathcal { L } ( \alpha ) } { \partial \alpha } = - 2 k { W } V _ { s k } ^ { 2 } K \alpha + 2 \lambda _ { g s s } K V _ { s k } ^ { 2 } Y _ { u _ { s } } ^ { \top } } } \\ { \displaystyle \phantom { \frac { \partial \mathcal { L } ( \alpha ) } { \partial \alpha } = - 2 { \lambda _ { g s s } K } V _ { s k } ^ { 2 } K \alpha + 2 \eta K \alpha } } \end{array}\tag{62}
$$

we have

$$
\begin{array} { r l r } {  { \frac { \partial ^ { 2 } \mathcal { L } ( \alpha ) } { \partial \alpha ^ { 2 } } = 2 ( I \otimes K W ^ { 2 } K ) + 2 \lambda _ { t t } ( I \otimes K V _ { t u } ^ { 2 } K ) } } \\ & { } & { ~ + 2 ( \lambda _ { s } + \lambda _ { t } ) ( I \otimes K V _ { k } ^ { 2 } K ) - 2 \lambda _ { g s t } ( I \otimes K V _ { s k } ^ { 2 } K ) } \\ & { } & { ~ - 2 \lambda _ { g s s } ( I \otimes K V _ { s k } ^ { 2 } K ) + 2 \eta ( I \otimes K ) } \end{array}\tag{63}
$$

where $\otimes$ represents Kronecker product. Based on (18) and (21), we can find that $\frac { \partial ^ { 2 } \mathcal { L } ( \alpha ) } { \partial \alpha ^ { 2 } } \geq 0$ always holds if and only if $\begin{array} { r } { \frac { 1 } { m _ { s } } - \frac { \lambda _ { g s t } + \lambda _ { g s s } } { m _ { s k } } \ge 0 } \end{array}$ . Hence, Theorem 2 has been proved.

## REFERENCES

[1] S. J. Pan and Q. Yang, “A survey on transfer learning,” IEEE Trans. Knowl. Data Eng., vol. 22, no. 10, pp. 1345–1359, Oct. 2010.

[2] Y. Zhang, S. Miao, T. Mansi, and R. Liao, “Task driven generative modeling for unsupervised domain adaptation: Application to x-ray image segmentation,” 2018, arXiv: 1806.07201.

[3] K. Zhang, B. Scholkopf, K. Muandet, and Z. Wang, “Domain adaptation¨ under target and conditional shift,” in Int. Conf. Mach. Learn., Jun. 2013 pp. 819–827.

[4] N. Adams, “Dataset shift in machine learning,” Journal of the Royal Statistical Society Series A: Statistics in Society, vol. 173, no. 1, pp. 274–274, Dec. 2009.

[5] S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan, “A theory of learning from different domains,” Mach. Learn., vol. 79, no. 1–2, pp. 151–175, Oct. 2009.

[6] A. Dutta, R. Lal, Y. Garg, C.-K. Ta, D. S. Raychaudhuri, and A. K. Roy-Chowdhury, “Pose guided unsupervised domain adaptation for human body part segmentation,” IEEE Trans. Image Process., 2026.

[7] X. Liu, Y. Huang, H. Wang, Z. Xiao, and S. Zhang, “Universal and scalable weakly-supervised domain adaptation,” IEEE Trans. Image Process., vol. 33, pp. 1313–1325, 2024.

[8] Y. Gao, A. J. Ma, Y. Gao, J. Wang, and Y. Pan, “Adversarial open set domain adaptation via progressive selection of transferable target samples,” Neurocomputing, vol. 410, pp. 174–184, Oct. 2020.

[9] C.-X. Ren, Z.-X. Huang, and H. Yan, “Open set domain adaptation via target-relaxed optimal transport,” IEEE Trans. Image Process., 2026.

[10] Y. Zheng, J. Wu, W. Li, and Z. Chen, “Universal domain adaptive object detection via dual probabilistic alignment,” Proc. AAAI Conf. Artif. Intell., vol. 39, no. 10, pp. 10 644–10 652, Apr. 2025.

[11] W. He, Z. Wang, and Y. Zhang, “Target semantics clustering via text representations for robust universal domain adaptation,” Proc. AAAI Conf. Artif. Intell., vol. 39, no. 16, pp. 17 132–17 140, Apr. 2025.

[12] R. Mussard, F. Pacheco, M. Berar, G. Gasso, and P. Honeine, “Deep joint distribution optimal transport for universal domain adaptation on time series,” 2025, arXiv: 2503.11217.

[13] J. Guo, Y. Lai, J. Zhang, J. Zheng, H. Fu, L. Gan, L. Hu, G. Xu, and X. Che, “C³da: A universal domain adaptation method for scene classification from remote sensing imagery,” IEEE Geosci. Remote Sens. Lett., vol. 21, pp. 1–5, Mar. 2024.

[14] Q. Xu, Y. Shi, X. Yuan, and X. X. Zhu, “Universal domain adaptation for remote sensing image scene classification,” IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1–15, Feb. 2023.

[15] Q. Qian, J. Luo, and Y. Qin, “Adaptive intermediate class-wise distribution alignment: A universal domain adaptation and generalization method for machine fault diagnosis,” IEEE Trans. Neural Networks Learn. Syst., vol. 36, no. 3, pp. 4296–4310, Mar. 2025.

[16] H. Liu, Z. Cao, M. Long, J. Wang, and Q. Yang, “Separate to adapt: Open set domain adaptation via progressive separation,” in Proc IEEE Comput Soc ConfComput Vision Pattern Recognit, Jun. 2019, pp. 2922– 2931.

[17] K. You, M. Long, Z. Cao, J. Wang, and M. I. Jordan, “Universal domain adaptation,” in Proc IEEE Comput Soc Conf Comput Vision Pattern Recognit, Jun. 2019, pp. 2715–2724.

[18] B. Fu, Z. Cao, M. Long, and J. Wang, “Learning to detect open classes for universal domain adaptation,” in Proc. Eur. Conf. Comput. Vis.(ECCV), Nov. 2020, pp. 567–583.

[19] G. Li, G. Kang, Y. Zhu, Y. Wei, and Y. Yang, “Domain consensus clustering for universal domain adaptation,” in Proc IEEE Comput Soc Conf Comput Vision Pattern Recognit, Jun. 2021, pp. 9757–9766.

[20] K. Saito, D. Kim, S. Sclaroff, and K. Saenko, “Universal domain adaptation through self-supervision,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Dec. 2020, pp. 16 282–16 292.

[21] Q. Lv, Y. Li, J. Dong, and Z. Guo, “Lafea: Learning latent representation beyond feature for universal domain adaptation,” IEEE Trans. Circuits Syst. Video Technol., vol. 33, no. 11, pp. 6733–6746, Nov. 2023.

[22] Y. Dai, H. Zhu, S. Yang, and H. Zhang, “Gcl-osda: Uncertainty prediction-based graph collaborative learning for open-set domain adaptation,” Knowledge-Based Syst., vol. 256, pp. 109 850–109 850, Nov. 2022.

[23] Y. Du, Y. Cao, Y. Zhou, Y. Chen, R. Zhang, and C. Wang, “Self separation and misseparation impact minimization for open-set domain adaptation,” in International Conference on Database Systems for Advanced Applications, Apr. 2021, pp. 400–409.

[24] Z. Fang, J. Lu, F. Liu, J. Xuan, and G. Zhang, “Open set domain adaptation: Theoretical bound and algorithm,” IEEE Trans. Neural Networks Learn. Syst., vol. 32, no. 10, pp. 4309–4322, Oct. 2021.

[25] B. Scholkopf, R. Herbrich, and A. J. Smola, “A generalized representer¨ theorem,” in International conference on computational learning theory. Springer, 2001, pp. 416–426.

[26] S. Chen, L. Han, X. Liu, Z. He, and X. Yang, “Subspace distribution adaptation frameworks for domain adaptation,” IEEE Trans. Neural Networks Learn. Syst., vol. 31, no. 12, pp. 5204–5218, 2020.

[27] C.-X. Ren, Y.-W. Luo, and D.-Q. Dai, “Buresnet: Conditional bures metric for transferable representation learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 4, pp. 4198–4213, 2022.

[28] S. Chen, “Multi-source domain adaptation with mixture of joint distributions,” Pattern Recogn., vol. 149, p. 110295, 2024.

[29] S. Chen, M. Harandi, X. Jin, and X. Yang, “Domain adaptation by joint distribution invariant projections,” IEEE Trans. Image Process., vol. 29, pp. 8264–8277, 2020.

[30] L. Wen, S. Chen, Z. Hong, and L. Zheng, “Maximum likelihood weight estimation for partial domain adaptation,” Inf. Sci., vol. 676, p. 120800, 2024.

[31] X. Jin, X. Yang, B. Fu, and S. Chen, “Joint distribution matching embedding for unsupervised domain adaptation,” Neurocomputing, vol. 412, pp. 115–128, 2020.

[32] S. Chen, M. Harandi, X. Jin, and X. Yang, “Semi-supervised domain adaptation via asymmetric joint distribution matching,” IEEE Trans. Neural Networks Learn. Syst., vol. 32, no. 12, pp. 5708–5722, 2020.

[33] S. Chen, Z. Hong, M. Harandi, and X. Yang, “Domain neural adaptation,” IEEE Trans. Neural Networks Learn. Syst., vol. 34, no. 11, pp. 8630–8641, 2022.

[34] S. Chen, L. Wang, Z. Hong, and X. Yang, “Domain generalization by joint-product distribution alignment,” Pattern Recogn., vol. 134, p. 109086, 2023.

[35] J. C. Meza, “Steepest descent,” Wiley Interdiscip. Rev. Comput. Stat., vol. 2, no. 6, pp. 719–722, 2010.

[36] Y. Luo, Z. Wang, Z. Chen, Z. Huang, and M. Baktashmotlagh, “Sourcefree progressive graph learning for open-set domain adaptation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 9, pp. 11 240–11 255, 2023.

[37] Y. Luo, Z. Wang, Z. Huang, and M. Baktashmotlagh, “Progressive graph learning for open-set domain adaptation,” in Int. Conf. Mach. Learn. PMLR, 2020, pp. 6468–6478.

[38] H. Xue, Q. Yang, and S. Chen, “Svm: Support vector machines,” in The top ten algorithms in data mining. Chapman and Hall/CRC, 2009, pp. 51–74.

[39] Y. Chen, C. Wei, A. Kumar, and T. Ma, “Self-training avoids using spurious features under domain shift,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020, pp. 21 061–21 071.

[40] H. Abdi and L. J. Williams, “Principal component analysis,” Wiley Interdiscip. Rev. Comput. Stat., vol. 2, no. 4, pp. 433–459, 2010.

[41] A. Rahimi and B. Recht, “Random features for large-scale kernel machines,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2007, pp. 1177–1184.

[42] M. R. Hestenes, E. Stiefel et al., “Methods of conjugate gradients for solving linear systems,” J. Res. Natl. Bur. Stand., vol. 49, no. 6, pp. 409–436, 1952.

[43] K. Saenko, B. Kulis, M. Fritz, and T. Darrell, “Adapting visual category models to new domains,” in Proc. Eur. Conf. Comput. Vis.(ECCV), Sep. 2010, pp. 213–226.

[44] H. Venkateswara, J. Eusebio, S. Chakraborty, and S. Panchanathan, “Deep hashing network for unsupervised domain adaptation,” in Proc IEEE Comput Soc Conf Comput Vision Pattern Recognit, Jul. 2017, pp. 5385–5394.

[45] X. Peng, B. Usman, N. Kaushik, J. Hoffman, D. Wang, and K. Saenko, “Visda: The visual domain adaptation challenge,” 2017, arXiv: 1710.06924.

[46] X. Peng, Q. Bai, X. Xia, Z. Huang, K. Saenko, and B. Wang, “Moment matching for multi-source domain adaptation,” in Proc IEEE Int Conf Comput Vision, 2019, pp. 1406–1415.

[47] S. F. dos Santos, T. A. De Almeida, and J. Almeida, “E-mlnet: Enhanced mutual learning for universal domain adaptation with sample-specific weighting,” in Braz. Symp. Comput. Graph. Image Process. IEEE, 2025, pp. 1–6.

[48] D. Li, Y. Yang, Y.-Z. Song, and T. M. Hospedales, “Deeper, broader and artier domain generalization,” in Proc IEEE Int Conf Comput Vision, 2017, pp. 5542–5550.

[49] K. Saito, S. Yamamoto, Y. Ushiku, and T. Harada, “Open set domain adaptation by backpropagation,” in Proc. Eur. Conf. Comput. Vis.(ECCV), Sep. 2018, pp. 153–168.

[50] J. Zhang, Z. Ding, W. Li, and P. Ogunbona, “Importance weighted adversarial nets for partial domain adaptation,” in Proc IEEE Comput Soc Conf Comput Vision Pattern Recognit, Jun. 2018, pp. 8156–8164.

[51] H. Wu, Z. Feng, Q. Zhang, J. Wu, and J. Lai, “Tanet: Adversarial network via tokens transformer for universal domain adaptation,” in International Conference on Image and Graphics, Oct. 2023, pp. 180– 191.

[52] W. Su, Z. Han, X. Liu, and Y. Yin, “Generalized universal domain adaptation,” Knowledge-Based Syst., vol. 302, p. 112344, 2024.

[53] C. Fan, P. Liu, and W. Zhao, “Mahalanobis distance-guided conditional adversarial learning for universal domain adaptation,” Knowledge-Based Syst., p. 113850, 2025.

[54] W. Wang, C. Huang, J. Wen, C. Wang et al., “Batch singular value polarization and weighted semantic augmentation for universal domain adaptation,” in Int. Conf. Mach. Learn., 2024.

[55] L. Chen, Y. Lou, J. He, T. Bai, and M. Deng, “Geometric anchor correspondence mining with uncertainty modeling for universal domain adaptation,” in Proc IEEE Comput Soc Conf Comput Vision Pattern Recognit, 2022, pp. 16 134–16 143.

[56] K. Saito and K. Saenko, “Ovanet: One-vs-all network for universal domain adaptation,” in Proc IEEE Int Conf Comput Vision, Oct. 2021, pp. 8980–8989.

[57] S. Hur, I. Shin, K. Park, S. Woo, and I. S. Kweon, “Learning classifiers of prototypes and reciprocal points for universal domain adaptation,” in Proc. IEEE Winter Conf. Appl. Comput. Vis., Jan. 2023, pp. 531–540.

[58] Y. Wang, L. Zhang, R. Song, H. Li, P. L. Rosin, and W. Zhang, “Exploiting inter-sample affinity for knowability-aware universal domain adaptation,” Int. J. Comput. Vision, vol. 132, no. 5, pp. 1800–1816, Dec. 2024.

[59] X. Shan, T. Ma, and Y. Wen, “Prediction of common labels for universal domain adaptation,” Neural Networks, vol. 165, pp. 463–471, Aug. 2023.

[60] S. K. Jain and S. Das, “Stochastic binary network for universal domain adaptation,” in Proc. IEEE Winter Conf. Appl. Comput. Vis., Jan. 2024, pp. 106–115.

[61] Y. Wang, Y. Liu, and S. Chen, “Towards adaptive unknown authentication for universal domain adaptation by classifier paradox,” Mach. Learn., vol. 113, no. 4, pp. 1623–1641, 2024.

[62] Y. Lu, M. Shen, A. J. Ma, X. Xie, and J.-H. Lai, “Mlnet: Mutual learning network with neighborhood invariance for universal domain adaptation,” in Proc. AAAI Conf. Artif. Intell., vol. 38, no. 4, 2024, pp. 3900–3908.

[63] C. Fan, P. Liu, and W. Zhao, “Curriculum adaptation method based on graph neural networks for universal domain adaptation,” Expert Sys Appl, vol. 255, p. 124509, 2024.

[64] J. Zhu, F. Ye, Q. Xiao, P. Guo, Y. Zhang, and Q. Yang, “A versatile framework for unsupervised domain adaptation based on instance weighting,” IEEE Trans. Image Process., 2024.

[65] S. Yu, Y. Huang, T. Yang, J. Lin, and R. Luo, “Novel category discovery across domains with contrastive learning and adaptive classifier,” in Proc. Int. Jt. Conf. Neural Networks. IEEE, 2024, pp. 1–9.

[66] W. Ai, Z. Yang, Z. Chen, and X. Hu, “Maximum open-set entropy optimization via uncertainty measure for universal domain adaptation,” J Visual Commun. Image Represent., vol. 101, p. 104169, 2024.

[67] L. Wen, S. Chen, L. Zheng, and P. Xuan, “Open-set domain adaptation by joint distribution alignment and unknown risk minimization,” Pattern Recogn., p. 113013, 2025.

[68] I. Sason and S. Verdu, “ ´ f-divergence inequalities,” IEEE Trans. Inf. Theory, vol. 62, no. 11, pp. 5973–6006, 2016.