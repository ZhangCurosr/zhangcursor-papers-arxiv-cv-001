# Beyond Readability: Evaluating Task Information Recoverability

Yiwei Liu School of Science and Engineering The Chinese University of Hong Kong, Shenzhen yiweiliu1@link.cuhk.edu.cn

## Abstract

Direct visual readability and task-information recoverability are diferent quantities. Failure to decode a target from a fixed observation need not eliminate access to that target through another recovery route. We develop an evaluation perspective that makes the observation, query, target, and available knowledge explicit and measures the overlap between routes’ success sets. For information available on the original visible surface under suitable imaging conditions, direct optical recovery reads the target from the image, optionally after restoration; entity-linked recovery uses residual visual evidence to identify the depicted entity and accesses its target through an entity–attribute relation in a specified knowledge resource. Such access can draw on stored knowledge or an external source. A controlled book-cover study instantiates external access with a fixed title–author catalog, comparing optical author recovery with visual title resolution and deterministic lookup under resolution degradation. Entity-linked successes persist across the tested vision–language models, revealing information access beyond the tested direct visual frontier despite substantial diferences in absolute performance. A substantial optical-only region remains. These complementary outcomes show why visual degradation should be evaluated through the task information accessible along specified routes and knowledge resources, alongside direct readability.

## 1. Introduction

Direct visual readability and task-information recoverability answer diferent questions. Readability concerns decoding a target from its visible representation; recoverability concerns access to that target from an observation through a specified route. A fixed image may frustrate direct optical decoding while retaining cues that identify the depicted entity. That identity can index an entity–attribute relation in stored knowledge or an external resource, making the requested target accessible without decoding its surface representation. An evaluation of perception therefore needs to compare complete recovery routes on the same observation as well as measure their individual accuracy.

Human perception motivates this distinction. Sensory evidence is interpreted through experience, context, and knowledge: an incomplete image can still support a meaningful account of the object that produced it [22, 28]. Theories of object recognition explain how visual structure supports identity across changes in appearance, while studies of the ventral visual pathway characterize representations that support this stability [8, 19, 23]. People can also extract category or scene meaning from very brief presentations [39, 44]. These findings motivate a task-centered evaluation question: when fine detail becomes dificult to decode, what useful information remains accessible from the same observation?

Context makes this question especially important. Experiments on object identification show that surrounding scenes and the consistency of object–scene relations influence recognition [9, 17, 37]. Context constrains plausible interpretations, allowing an object’s appearance to be understood in relation to its environment and learned associations [4, 36]. Top-down influences can facilitate visual recognition, including through information available before detailed identification is complete [6, 21]. Experience also guides the allocation of attention: learned spatial regularities, scene knowledge, and task goals shape which evidence is selected [15, 24, 48]. Together, these perspectives suggest that evaluating perception requires attention to the relationship between available visual cues and the knowledge that makes those cues useful.

Predictive and inferential accounts make this relationship explicit. Predictive coding describes interactions between sensory signals and predictions about their causes [16, 20, 41], while work on associative prediction and perceptual expectation examines how prior experience shapes interpretation and decisions [5, 18, 43]. Findings on expectationdependent visual representations and recurrent object recognition further emphasize that recognition involves more than a single pass through the incoming signal [27, 29]. For evaluation, the resulting motivation is straightforward: the detail that a particular decoder can extract and the information that a system can access using that observation are distinct quantities.

Semantic memory provides a further connection between recognizing an object and knowing something about it. Research on semantic representation describes how perceptual inputs relate to stored knowledge of entities, attributes, and relationships [10, 11, 38]. Accounts of semantic cognition and grounded cognition connect this knowledge to perceptual experience and its use in context [7, 30]. A book illustrates the distinction: its printed author name may be dificult to transcribe, yet its title, artwork, or layout may identify the book. Its identity can then provide access to the author through existing knowledge or an external lookup, as illustrated in Figure 1.

Modern vision–language systems ofer a practical setting for studying such access. Large-scale image–text learning connects visual representations to language, and multimodal models combine visual inputs with pretrained language capabilities [1, 26, 40]. Visual entity recognition and information-seeking question answering already use images to address information about particular entities [14, 25, 33].

We operationalize this distinction through two ways of accessing the same target. Direct optical recovery decodes the target from its visual representation, optionally after restoration. Entity-linked knowledge recovery uses residual visual evidence to identify the entity and then accesses the target through a specified entity–attribute resource. We restrict attention to information available on the original visible surface under suitable imaging conditions. In our controlled demonstration, a model resolves a book title from the image and a fixed external title–author catalog supplies the author. This choice makes knowledge access deterministic after entity linking while controlling the source and retrieval variability of open search. A recoverability gap appears when the entity-linked route accesses a target that the tested direct decoders fail to recover from the same observation. Joint outcomes expose this gap and the complementary optical-only region, which marginal accuracy alone leaves unresolved.

We contribute an evaluation framework that treats direct decoding and entity-linked knowledge access as complete routes from a fixed observation to the same target. A controlled external catalog makes the entity–attribute relation explicit and auditable, so success requires both the correct identity and the correct target. A Strong Optical Frontier summarizes success across the tested direct decoders; joint outcome counts and conditional rescue rates then measure which observations support either route or both.

We demonstrate this perspective with a controlled evaluation of 5,000 books across seven degradation levels, providing 35,000 observations and using a fixed catalog for external knowledge access. The primary Qwen3.7-Plus [2] evaluation shows entity-linked recovery within the opticalfailure region alongside a substantial optical-only region. The optical frontier has higher overall accuracy, but the entity-linked route exceeds it at the strongest degradation levels. Replacing the title-resolution model with Qwen2.5- VL-32B-Instruct [3] or InternVL3.5-14B-Instruct [46] preserves a nontrivial entity-linked-only region under the same frozen protocol, despite substantial variation in absolute performance. Complementary recoverability is thus observed across all three tested VLMs: direct visual decoding and visual identification followed by catalog access expose partially distinct regions of task information access.

## 2. Related Work

## 2.1. Restoration and text recovery

General super-resolution methods provide plausible routes to improve the input of a text decoder. HAT combines channel and window-based self-attention with overlapping crossattention [13], while Real-ESRGAN learns blind restoration using synthetic degradations [47]. We evaluate their restored images through the same OCR stack as the native input. Image fidelity itself is not the scored target.

Text-specific restoration is a relevant alternative to these general routes. TextZoom and Scene Text Telescope study reconstruction for reading scene text [12, 45]; more recently, TIGER separates glyph restoration from full-image enhancement [31]. These approaches emphasize the importance of evaluating restoration through the content that a downstream decoder actually recovers. Our evaluation compares the selected general restoration routes using a common targetrecovery criterion.

## 2.2. Visual entity recognition and structured knowledge access

Knowledge-based visual question answering connects image understanding with information beyond directly visible content. OK-VQA and A-OKVQA study questions requiring external or world knowledge [32, 42], while InfoSeek and Encyclopedic VQA emphasize detailed information about fine-grained entities [14, 33]. Entity recognition provides a link between visual evidence and such information: OVEN maps an image and query to a Wikipedia entity, and WikiCLIP develops a contrastive approach to open-domain visual entity recognition [25, 35].

Our study uses this recognition-to-knowledge connection to evaluate recovery of a surface-available target under degradation. Entity identity can index an attribute in existing knowledge or in an external resource; the controlled experiment instantiates the latter with exact lookup in a fixed title–author catalog. This choice holds coverage, title ambiguity, and answer mapping constant rather than introducing variation from open-web retrieval. Residual visual evidence supports book-title resolution, and successful linking provides access to the author. Comparing this complete route with direct optical decoding on the same observations reveals where entity-linked knowledge recovers information that the tested optical routes fail to read.

![](images/147793e0750507c8fa695c2a3b52f79034b292515cc45cbefbd6c53d1574ba4c.jpg)  
Figure 1. Visual cues support book identification, which in turn enables access to an associated author through existing knowledge or an external resource. The illustration motivates evaluation of the information accessible from a fixed observation through a complete recovery route.

## 3. Recovery Framework

We evaluate recovery of task-relevant information from a single fixed RGB observation and a natural query. The query requests a textual or text-serializable target with its own readable or decodable representation on the depicted entity’s original visible surface under suitable imaging conditions. A complete recovery route may decode the target directly or use the observation to identify the entity and access its target through a specified knowledge resource. The output is the requested information or an abstention.

## 3.1. Direct Optical Recovery

Direct optical recovery obtains the target by decoding its surface representation in the native image or a resized or restored version, using learned or conventional processing. Success is measured by recovery of that content, independently of the visual quality of an intermediate image.

## 3.2. Entity-linked Knowledge Recovery

The entity-linked semantic route uses residual visual evidence to identify the depicted entity and then accesses its associated target through an entity–attribute relation. Partial text, appearance, and layout can support identification even when direct target decoding fails. The knowledge resource may be internal to the recognizer or external to it; in the controlled study, a fixed catalog serves as the external resource. The model resolves a book title from the degraded image, and an exact catalog match supplies the author. Entity ambiguity, incorrect linkage, insuficient evidence, and abstention are possible outcomes of this complete route.

The scientific distinction is how the requested information is accessed: direct decoding of its surface representation or visual identification followed by access through an entity-to-target relation. Joint outcomes measure where these routes succeed together and where either route alone recovers the target. This comparison captures complementary recoverability from the same observation.

## 4. Controlled Evaluation

The controlled evaluation measures author recovery from 5,000 book covers across seven resolution levels, yielding 35,000 observations with entity-linked scoring. The observation set and title–author catalog are fixed throughout; the supplement specifies construction, inference settings, hardware, and scoring in detail.

## 4.1. Evaluation design

Native, HAT x4, and Real-ESRGAN x4 start from the same RGB observation and use the same PaddleOCR stack. OCR processes each full route image at relative scales 1 and 2, with LANCZOS resizing and deterministic transcript concatenation. The target matcher tests normalized substring containment, ignoring case, spacing, and punctuation. For evaluation, we define the Strong Optical Frontier as the union of successes across the tested optical routes. Intendedset rescue divides semantic successes by all optical-frontier failures; no-response cases are reported separately.

## 4.2. Observation set and degradation

The target is the AUTHOR printed on a book cover. OCR-VQA metadata [34] are pooled into a catalog of 184,809 titleunique entries, excluding ambiguous normalized titles without author-based disambiguation. Deterministic ordering selects 8,000 candidates; clean-cover OCR identifies 5,785 eligible books whose normalized author occurs in the transcript. The evaluation freezes 5,000 unique books before degradation, with no model training or adaptation.

Each book yields seven observations at scales 1.0, 0.75, 0.5, 0.35, 0.25, 0.18, and 0.125. BOX/area downsampling is followed by aspect-preserving LANCZOS resizing and centering on a black 512-by-512 RGB canvas, without JPEG recompression. Scale 1.0 includes canvas resizing and difers from the clean-screen image. All routes share the 35,000 observations, with equal weight across levels.

## 4.3. Entity-linked semantic recovery

Qwen3.7-Plus through the oficial Alibaba Cloud service receives only the degraded image and query and returns a title hypothesis, visible anchors, and abstention. Exact title linking applies Unicode NFKC, case folding, and removal of non-alphanumeric characters; a unique match in the fixed external catalog supplies the author. Thus the model identifies the book from the image while the catalog provides the entity–attribute mapping. Success requires both the correct book and author, so a wrong book sharing the author fails. Evaluation entities are intentionally in the catalog, and incomplete or variant titles can remain unresolved.

Semantic evaluation yields 34,934 valid responses and 66 infrastructure/API failures. Incorrect valid responses are retained without retries.

To assess robustness to the semantic model, we also evaluate Qwen2.5-VL-32B-Instruct and InternVL3.5-14B-Instruct on the same 35,000 observations. Both use imagebased title resolution followed by the same deterministic catalog lookup and entity-linked scoring.

## 5. Results

Recovery regions. Table 1 gives marginal target recovery on the controlled observation set for the primary Qwen3.7- Plus evaluation. The optical frontier improves only slightly on Native, since restoration adds relatively few successful cases. Semantic recovery has lower overall accuracy, with entity-correct and catalog-derived author-correct counts coinciding. The joint outcomes reveal how this marginal ordering relates to recovery on individual observations.

The joint outcomes in Table 2 expose the distinction. Visual identification followed by catalog access recovers 5,158 of the 16,836 observations where all tested optical routes fail, giving 30.64% intended-set rescue. A substantial opticalonly region remains as well, so neither observed success set contains the other. This is evidence of complementary taskinformation recovery despite the entity-linked route’s lower marginal accuracy.

Table 1. Controlled evaluation: overall target recovery. A semantic success requires the correct entity and author.
<table><tr><td>Route</td><td>Correct Total</td><td>Accuracy</td></tr><tr><td>Native + OCR</td><td>17,940 35,000</td><td>51.26%</td></tr><tr><td>HAT + OCR</td><td>13,532 35,000</td><td>38.66%</td></tr><tr><td>Real-ESRGAN + OCR</td><td>13,221 35,000</td><td>37.77%</td></tr><tr><td>Strong frontier</td><td>18,16435,000</td><td>51.90%</td></tr><tr><td>Semantic, full set</td><td>14,31535,000</td><td>40.90%</td></tr><tr><td>Semantic, valid only</td><td>14,31534,934</td><td>40.98%</td></tr></table>

Table 2. Controlled evaluation: joint target-recovery outcomes. The first two columns form the two-by-two table for the 34,934 observations with valid semantic responses.
<table><tr><td></td><td colspan="2">Semantic outcome</td></tr><tr><td>Optical frontier</td><td>Correct</td><td>Incorrect No response</td></tr><tr><td>Correct</td><td>9,157</td><td>8,967</td></tr><tr><td>Incorrect</td><td>5,158</td><td>11,652</td></tr></table>

Efect of degradation. Figure 2 and Table 3 show how the observed regions change with resolution. Optical recovery falls sharply as scale decreases, while semantic recovery initially declines more slowly and exceeds the frontier at the three smallest scales. Both routes fail frequently at the strongest degradation, indicating that semantic access also depends on the surviving evidence.

Conditional rescue peaks at scale 0.35, whereas the largest absolute rescue count occurs at 0.25. The distinction reflects the changing size and composition of the opticalfailure subset, which contains only 39 observations at scale 1.0. Declining rescue at the smallest scales is consistent with increasingly ambiguous identity evidence.

Semantic failure modes. The outcome taxonomy contains 14,315 successes, 19,243 unresolved titles/entities, 707 wrong entities, 669 abstentions, and 66 infrastructure/API failures. Unresolved titles dominate scientific errors even at scale 1.0. Unresolved outcomes may arise from incomplete title recovery or unsuccessful entity linking. Abstention rises sharply at the strongest degradation, adding a distinct failure mode as the image becomes less informative.

Qualitative evidence. Figure 3 illustrates one frozen rescue at scale 0.25. The three optical transcripts retain title fragments but miss the author. From the same degraded input, the entity-linked route resolves the full title, enabling exact catalog lookup of the correct author. This case illustrates how surviving visual evidence can identify an entity and thereby index the target in an external resource when direct decoding fails.

(a) Optical and semantic recovery  
![](images/c42611e02ce1eec8ccf3159b4df1b82fef794942c69c005642a448494b5b5c47.jpg)

(b) Conditional semantic rescue  
![](images/1e3a0352653b3a957d91742123d826763640a14d90f1963cf32d5fc8be3eb51d.jpg)  
Figure 2. Controlled recovery under fixed degradation, using Qwen3.7-Plus for the semantic route. Left: all routes on the full intended set of 5,000 observations per level. Right: semantic success conditioned on failure of the Strong Optical Frontier, with the number of observed rescues over all frontier failures shown at each point. Smaller scales mean stronger degradation; points denote the seven tested levels.

Target: author | degradation scale 0.25  
Native + OCR  
![](images/a928f2ecec10d8aebb610220426224375cce0445ee0067cf34c887070a4dd36e.jpg)

HAT + OCR  
![](images/ed3aa7ef832989560f3a6011f4d733f311ae30d53619008816c1c118a819d6da.jpg)

Real-ESRGAN + OCR  
![](images/85993f0ca5a385db31aea48285652c5ae025473b4e1621c1bcb5036ac6c9ce96.jpg)  
Semantic title from the same native image:  
Beautiful You: A Daily Guide to Radical Self-Acceptance Catalog lookup: Rosie Molinary | SUCCESS  
Figure 3. Controlled evaluation: author recovery for Beautiful You. All three full archived route images are shown, with matched author region enlargements. Native, HAT, and Real-ESRGAN OCR each fail the frozen author matcher. Semantic title resolution uses the same native degraded image; exact linking to the frozen catalog supplies Rosie Molinary, with both entity and author correct.

Table 3. Controlled evaluation by degradation level, with 5,000 observations per row. Accuracies use the full intended set. Rescue is the semantic-success count divided by all frontier failures at that level.
<table><tr><td>Scale</td><td>Native</td><td>HAT</td><td>Real-ESRGAN</td><td>Frontier</td><td>Semantic</td><td>Frontier failures</td><td>Rescue</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1.000 0.750</td><td>98.86% 93.50%</td><td>90.58% 80.32%</td><td>88.94% 79.08%</td><td>99.22% 94.34%</td><td>51.18% 51.06%</td><td>39 283</td><td>13/39 (33.33%) 111/283 (39.22%)</td></tr><tr><td>0.500</td><td>79.56%</td><td>55.14%</td><td>55.20%</td><td>80.80%</td><td>50.08%</td><td>960</td><td>397/960 (41.35%)</td></tr><tr><td>0.350</td><td>48.14%</td><td>26.52%</td><td>25.98%</td><td>49.20%</td><td>46.66%</td><td>2,540</td><td>1,095/2,540 (43.11%)</td></tr><tr><td>0.250</td><td>25.48%</td><td>11.36%</td><td>9.56%</td><td>25.90%</td><td>41.06%</td><td>3,705</td><td>1,460/3,705 (39.41%)</td></tr><tr><td>0.180</td><td>10.40%</td><td>5.06%</td><td>4.30%</td><td>10.76%</td><td>29.60%</td><td>4,462</td><td>1,286/4,462 (28.82%)</td></tr><tr><td>0.125</td><td>2.86%</td><td>1.66%</td><td>1.36%</td><td>3.06%</td><td>16.66%</td><td>4,847</td><td>796/4,847 (16.42%)</td></tr></table>

Table 4. Semantic-model robustness on the same 35,000 observations and frozen optical frontier. AUTHOR correctness requires the correct book and catalog-derived author. Rescue uses all 16,836 frontier failures.
<table><tr><td colspan="3">AUTHOR Semantic model</td><td rowspan="2">Rescue rate</td></tr><tr><td></td><td>correct</td><td>Rescues</td></tr><tr><td>Qwen3.7-Plus Qwen2.5-VL-</td><td>14,315</td><td>5,158</td><td>30.64%</td></tr><tr><td>32B-Instruct InternVL3.5-</td><td>11,105</td><td></td><td>3,693 21.94%</td></tr><tr><td>14B-Instruct</td><td>6,873</td><td></td><td>2,27913.54%</td></tr></table>

Semantic-model robustness. Table 4 tests whether semantic-only recovery persists when the title-resolution model changes. Absolute author recovery and rescue rates vary substantially, yet every tested VLM recovers the target on a nontrivial subset of the same optical-frontier failures.

## 6. Discussion

## 6.1. Interpreting recovery routes

The recoverability gap changes how failure under visual degradation should be interpreted. Failure of the tested optical routes identifies a boundary of direct decoding; it leaves open whether the same target is accessible through another route from the same observation. The controlled study makes this distinction observable: an entity-linkedonly region persists across all three tested VLMs, with its size depending strongly on the model, while a substantial optical-only region remains. These complementary success sets make route overlap essential to interpreting information access, even when one route has higher marginal accuracy.

The relevant unit of evaluation is the complete recovery route, with its observation, query, and permitted knowledge resource. Here, residual visual evidence supports book identification, and a fixed external catalog supplies the author through an entity–attribute relation. The book-cover evaluation thus gives a controlled instance of the broader principle that readability alone cannot determine task-information recoverability. Evaluating degraded observations through the information still accessible along specified routes avoids equating poorer visual detail with automatic loss of the requested information.

## 6.2. Open-web knowledge access

Open-web search services such as Jina Search and Duck-DuckGo could provide broader and more current entitylinked evidence than a fixed catalog, extending access when a target relation is absent from a curated resource. Their use also makes controlled evaluation harder: queries generated from the image, result ranking, page availability and content, and a model’s selection and interpretation of retrieved evidence can all afect the final answer. We therefore use a fixed catalog as the external knowledge source in the present study, making entity coverage and title–author mapping explicit and lookup deterministic. A useful next step is to evaluate open-web retrieval under the same fixed-observation protocol, varying the search service, query strategy, and evidence selection while recording the retrieved sources and their efect on information recovery.

## 6.3. Scope of surface-grounded targets

Our evaluation focuses on targets with a decodable representation on the original visible surface under suitable imaging conditions. A plain cup, for example, can be recognized from its shape even though the word “cup” is not printed on it; asking what the object is poses an object-recognition question without a corresponding surface field to read. Such questions are important, but they require a diferent reference route and controls for comparing visual recognition with knowledge access. We restrict the present study to surface-grounded textual or text-serializable targets so that direct decoding and entity-linked access can be evaluated against the same target from the same fixed observation.

## 7. Conclusion

Readability is not recoverability: failure of direct optical decoding can coexist with task information accessible through another route from the same fixed observation. Visual identification can index an entity–attribute relation in a knowledge resource, enabling access to a target whose surface representation is dificult to decode. Our controlled bookcover study instantiates this route through title resolution and external lookup in a fixed catalog, alongside direct optical author recovery. Entity-linked-only successes persist across the tested VLMs, while a substantial optical-only region remains. The broader evaluation lesson is to measure task information through complete routes and their specified knowledge resources rather than equate visual degradation with its loss.

## AI Use Disclosure

As English is not the authors’ native language, AI-assisted tools were used to improve language clarity, grammar, and overall readability of the manuscript. The use of AI was limited to writing assistance and does not extend to the development of the research ideas, experimental design, data analysis, or scientific conclusions. All AI-assisted revisions were reviewed and verified by the authors, who are fully responsible for the final manuscript.

## References

[1] Jean-Baptiste Alayrac, Jef Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikoł aj Bińkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, pages 23716–23736. Curran Associates, Inc., 2022. 2

[2] Alibaba Cloud. Alibaba Cloud Model Studio: Recommended Models, 2026. Accessed September 20, 2026. 2

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923, 2025. 2

[4] Moshe Bar. Visual Objects in Context. Nature Reviews Neuroscience, 5(8):617–629, 2004. 1

[5] Moshe Bar. The proactive brain: using analogies and associations to generate predictions. Trends in Cognitive Sciences, 11(7):280–289, 2007. 1

[6] M. Bar, K. S. Kassam, A. S. Ghuman, J. Boshyan, A. M. Schmid, A. M. Dale, M. S. Hämäläinen, K. Marinkovic, D. L. Schacter, B. R. Rosen, and E. Halgren. Top-down facilitation of visual recognition. Proceedings of the National Academy ofSciences, 103(2):449–454, 2006. 1

[7] Lawrence W. Barsalou. Grounded Cognition. Annual Review of Psychology, 59(1):617–645, 2008. 2

[8] Irving Biederman. Recognition-by-components: A theory of human image understanding. Psychological Review, 94(2): 115–147, 1987. 1

[9] Irving Biederman, Robert J. Mezzanotte, and Jan C. Rabinowitz. Scene perception: Detecting and judging objects undergoing relational violations. Cognitive Psychology, 14(2): 143–177, 1982. 1

[10] Jefrey R. Binder and Rutvik H. Desai. The neurobiology of semantic memory. Trends in Cognitive Sciences, 15(11): 527–536, 2011. 2

[11] Jefrey R. Binder, Rutvik H. Desai, William W. Graves, and Lisa L. Conant. Where Is the Semantic System? A Critical Review and Meta-Analysis of 120 Functional Neuroimaging Studies. Cerebral Cortex, 19(12):2767–2796, 2009. 2

[12] Jingye Chen, Bin Li, and Xiangyang Xue. Scene Text Telescope: Text-Focused Scene Image Super-Resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12026–12035, 2021. 2

[13] Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, and Chao Dong. Activating More Pixels in Image Super-Resolution Transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22367–22377, 2023. 2

[14] Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions? In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 14948–14968, 2023. 2

[15] Marvin M. Chun and Yuhong Jiang. Contextual Cueing: Implicit Learning and Memory of Visual Context Guides Spatial Attention. Cognitive Psychology, 36(1):28–71, 1998. 1

[16] Andy Clark. Whatever next? Predictive brains, situated agents, and the future of cognitive science. Behavioral and Brain Sciences, 36(3):181–204, 2013. 1

[17] Jodi L. Davenport and Mary C. Potter. Scene Consistency in Object and Background Perception. Psychological Science, 15(8):559–564, 2004. 1

[18] Floris P. de Lange, Micha Heilbron, and Peter Kok. How Do Expectations Shape Perception? Trends in Cognitive Sciences, 22(9):764–779, 2018. 1

[19] James J. DiCarlo, Davide Zoccolan, and Nicole C. Rust. How Does the Brain Solve Visual Object Recognition? Neuron, 73 (3):415–434, 2012. 1

[20] Karl Friston. A theory of cortical responses. Philosophical Transactions of the Royal Society B: Biological Sciences, 360 (1456):815–836, 2005. 1

[21] Charles D. Gilbert and Wu Li. Top-down influences on visual processing. Nature Reviews Neuroscience, 14(5):350–363, 2013. 1

[22] Richard Langton Gregory. Perceptions as hypotheses. Philosophical Transactions of the Royal Society of London. B, Biological Sciences, 290(1038):181–197, 1980. 1

[23] Kalanit Grill-Spector and Kevin S. Weiner. The functional architecture of the ventral temporal cortex and its role in categorization. Nature Reviews Neuroscience, 15(8):536–548, 2014. 1

[24] John M. Henderson. Human gaze control during real-world scene perception. Trends in Cognitive Sciences, 7(11):498– 504, 2003. 1

[25] Hexiang Hu, Yi Luan, Yang Chen, Urvashi Khandelwal, Mandar Joshi, Kenton Lee, Kristina Toutanova, and Ming-Wei Chang. Open-domain Visual Entity Recognition: Towards Recognizing Millions of Wikipedia Entities. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12065–12075, 2023. 2

[26] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 4904–4916. PMLR, 2021. 2

[27] Kohitij Kar, Jonas Kubilius, Kailyn Schmidt, Elias B. Issa, and James J. DiCarlo. Evidence that recurrent circuits are critical to the ventral stream’s execution of core object recognition behavior. Nature Neuroscience, 22(6):974–983, 2019. 2

[28] Daniel Kersten, Pascal Mamassian, and Alan Yuille. Object Perception as Bayesian Inference. Annual Review ofPsychology, 55(1):271–304, 2004. 1

[29] Peter Kok, Janneke F.M. Jehee, and Floris P. de Lange. Less Is More: Expectation Sharpens Representations in the Primary Visual Cortex. Neuron, 75(2):265–270, 2012. 2

[30] Matthew A. Lambon Ralph, Elizabeth Jeferies, Karalyn Patterson, and Timothy T. Rogers. The neural and computational bases of semantic cognition. Nature Reviews Neuroscience, 18(1):42–55, 2017. 2

[31] Minxing Luo, Linlong Fan, Qiushi Wang, Ge Wu, Yiyan Luo, Yuhang Yu, Jinwei Chen, Yaxing Wang, Qingnan Fan, and Jian Yang. Restore Text First, Enhance Image Later: Two-Stage Scene Text Image Super-Resolution with Glyph Structure Guidance. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 30553–30563, 2026. 2

[32] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3195–3204, 2019. 2

[33] Thomas Mensink, Jasper Uijlings, Lluis Castrejon, Arushi Goel, Felipe Cadar, Howard Zhou, Fei Sha, André Araujo, and Vittorio Ferrari. Encyclopedic VQA: Visual Questions About Detailed Properties of Fine-Grained Categories. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3113–3124, 2023. 2

[34] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. OCR-VQA: Visual Question Answering by Reading Text in Images. In 2019 International Conference on Document Analysis and Recognition, pages 947–952, 2019. 4

[35] Shan Ning, Longtian Qiu, Jiaxuan Sun, and Xuming He. WikiCLIP: An Eficient Contrastive Baseline for Open-domain Visual Entity Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1596–1605, 2026. 2

[36] Aude Oliva and Antonio Torralba. The role of context in object recognition. Trends in Cognitive Sciences, 11(12):520– 527, 2007. 1

[37] Stephen E. Palmer. The efects of contextual scenes on the identification of objects. Memory & Cognition, 3(5):519– 526, 1975. 1

[38] Karalyn Patterson, Peter J. Nestor, and Timothy T. Rogers. Where do you know what you know? The representation of semantic knowledge in the human brain. Nature Reviews Neuroscience, 8(12):976–987, 2007. 2

[39] Mary C. Potter, Brad Wyble, Carl Erick Hagmann, and Emily S. McCourt. Detecting meaning in RSVP at 13 ms per picture. Attention, Perception, & Psychophysics, 76(2): 270–279, 2014. 1

[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 2

[41] Rajesh P. N. Rao and Dana H. Ballard. Predictive coding in the visual cortex: a functional interpretation of some extraclassical receptive-field efects. Nature Neuroscience, 2(1): 79–87, 1999. 1

[42] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-OKVQA: A Benchmark for Visual Question Answering Using World Knowledge. In Computer Vision – ECCV 2022, pages 146– 162. Springer, 2022. 2

[43] Christopher Summerfield and Floris P. de Lange. Expectation in perceptual decision making: neural and computational mechanisms. Nature Reviews Neuroscience, 15(11):745–756, 2014. 1

[44] Simon Thorpe, Denis Fize, and Catherine Marlot. Speed of processing in the human visual system. Nature, 381(6582): 520–522, 1996. 1

[45] Wenjia Wang, Enze Xie, Xuebo Liu, Wenhai Wang, Ding Liang, Chunhua Shen, and Xiang Bai. Scene Text Image Super-Resolution in the Wild. In Computer Vision – ECCV 2020, pages 650–666. Springer, 2020. 2

[46] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen

Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Eficiency. arXiv preprint arXiv:2508.18265, 2025. 2

[47] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-ESRGAN: Training Real-World Blind Super-Resolution With Pure Synthetic Data. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 1905–1914, 2021. 2

[48] Jeremy M. Wolfe and Todd S. Horowitz. Five factors that guide attention in visual search. Nature Human Behaviour, 1 (3):0058, 2017. 1

# Beyond Readability: Evaluating Task Information Recoverability Supplementary Material

Yiwei Liu School of Science and Engineering The Chinese University of Hong Kong, Shenzhen yiweiliu1@link.cuhk.edu.cn

## S1. Controlled-set construction

## S1.1. Catalog and candidate selection

The controlled task recovers the author printed on a book cover using metadata from OCR-VQA. Metadata from the train, validation, and test splits are pooled for evaluation; no model is trained on these splits. For repeated image identifiers, the representative is the first record in lexicographic split order and then integer row order. Records with an empty title, author, or normalized title are removed.

Title normalization applies Unicode NFKC, case folding, and retention of alphanumeric characters. Every normalized title associated with multiple book identifiers is excluded, with no author-based disambiguation. The resulting catalog contains 184,809 title-unique entries and maps each retained title to its book identity and author. It serves as the fixed external knowledge resource: once a title is linked, access to the author is deterministic rather than dependent on changing search results or source ranking.

Candidate selection uses the seed ocr-vqa-controll ed-v1-20260909. Books are ordered by the SHA-256 digest of this UTF-8 seed, followed by a null separator and the book identifier. The first 8,000 records form the candidate pool. The same ordering selects the final books after the clean-cover eligibility screen.

## S1.2. Clean-cover eligibility

Eligibility requires the normalized catalog author to occur as a substring of the normalized OCR transcript from the original clean cover. The screen uses the controlled experiment’s OCR configuration described in Section S2.3, including full-image processing at relative scales 1 and 2. Of the 8,000 candidates, 5,785 pass this screen; the first 5,000 eligible unique books in the deterministic order form the fixed evaluation set.

This screen establishes that the requested attribute is recoverable from the original surface under the clean-image condition. Membership is fixed before degraded-image evaluation.

## S2. Degradation and optical recovery

## S2.1. Resolution degradation

Each selected cover is decoded as RGB and processed independently at scales 1.0, 0.75, 0.5, 0.35, 0.25, 0.18, and 0.125. At each scale, the original width and height are multiplied by the scale and rounded to the nearest integer using Python’s rounding convention, with each dimension clamped to at least one pixel. BOX/area downsampling produces the reduced image.

The reduced image is then resized with LANCZOS interpolation to fit a 512-by-512 canvas while preserving its aspect ratio. The fit factor is the smaller of the canvaswidth and canvas-height ratios; the resized dimensions use the same rounding and minimum-pixel rule. The image is centered on a black RGB canvas using integer ofsets obtained by floor division of the remaining width and height by two. Outputs are saved losslessly as PNG, with no JPEG recompression. Scale 1.0 retains the canvas-resizing step and therefore difers from the image used for eligibility screening.

All recovery routes receive the same resulting observation for a given book and scale. The design contains 35,000 observations, with 5,000 at each level; overall observationlevel rates weight the seven levels equally. Each image is evaluated independently, including across diferent levels of the same book.

## S2.2. Restoration and hardware

The optical routes process the native observation directly, apply HAT x4, or apply Real-ESRGAN x4 before OCR. HAT uses the pretrained Real HAT GAN x4 checkpoint, and Real-ESRGAN uses the pretrained x4plus checkpoint. Both use tile size 256; padding is 32 pixels for HAT and 10 pixels for Real-ESRGAN. Restoration and OCR models are used without training or adaptation to the evaluation set.

The optical restoration experiments use a workstation with five NVIDIA GeForce RTX 2080 Ti GPUs, each with

Table S1. OCR configuration for native and restored route images.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Document orientation classification</td><td>Disabled</td></tr><tr><td>Document unwarping</td><td>Disabled</td></tr><tr><td>Text-line orientation classification</td><td>Enabled</td></tr><tr><td>Maximum detection side length</td><td>4,000 pixels</td></tr><tr><td>Detection threshold</td><td>0.3</td></tr><tr><td>Box threshold</td><td>0.6</td></tr><tr><td>Unclip ratio</td><td>1.5</td></tr><tr><td>Recognition score threshold</td><td>0.0</td></tr><tr><td>Relative full-image scales</td><td>1.0 and 2.0</td></tr><tr><td>Scale-2 interpolation</td><td>OpenCV LANCZOS4</td></tr></table>

11,264 MiB of memory, and NVIDIA driver 570.124.04.   
Controlled OCR evaluation runs on the CPU.

## S2.3. OCR configuration

The optical routes use PaddleOCR 3.7.0 with the PP-OCRv6 medium detector and recognizer, configured for Chinese and English text through the Chinese language setting, with PaddlePaddle 3.3.1. Table S1 gives the OCR processing settings shared by all routes.

OCR processes each complete route image first at its native route resolution and then at twice that resolution, using the same preprocessing and matching policy for every observation. Recognized lines are concatenated in scale order and, within each scale, the model’s reading order.

For optical target matching, normalization applies Unicode NFKC and uppercase conversion, removes whitespace, and retains alphanumeric characters and code points U+3400–U+9FFF. A nonempty normalized target must occur as a substring of the normalized OCR transcript.

## S3. Controlled semantic recovery

## S3.1. Primary model and service settings

The controlled semantic route uses Qwen3.7-Plus through the oficial Alibaba Cloud Bailian/DashScope service in the Beijing region. Requests use temperature zero, reasoning disabled, and a maximum output budget of 1,800 tokens. The evaluation was completed in September 2026.

Each request contains only the degraded image and query, asking for a title hypothesis, visible anchors, and an abstention decision. The model uses residual visual evidence to resolve the book identity; subsequent deterministic lookup in the fixed external title–author catalog supplies the author. The complete route therefore combines visual entity identification with controlled knowledge access.

## S3.2. Title linking and target scoring

The predicted title is normalized using Unicode NFKC, case folding, and retention of alphanumeric characters, matching the catalog construction rule. An exact normalized-title match to a unique record links the observation to that book and supplies its author. Incomplete subtitles and alternative title forms can therefore remain unresolved even when they resemble a catalog title.

For a valid semantic response, success requires a nonabstained prediction linked to the correct book and yielding the correct author. A link to a diferent book is a wrong-entity outcome even if that book shares the target author. The semantic outcome categories are success, unresolved title or entity, wrong entity, abstention, and infrastructure/API failure. Incorrect valid responses are retained without retries; provider or transport failures with no valid response are counted separately.

## S4. Local VLM robustness

The local evaluations use Qwen2.5-VL-32B-Instruct at revision 7cfb30d71a1f4f49a57592323337a4a4727301da and InternVL3.5-14B-Instruct at revision 72c82460a2c0 5a3bc2b47be230c44edc88e86ed4, with BF16 inference. Both use deterministic decoding (do\_sample=false, temperature=0, max\_new\_tokens=1800), the same semantic prompt, title-linking rule, and scoring protocol as the primary evaluation. We retain first-pass results only and perform no targeted retry. The local VLM evaluations used four NVIDIA A100-SXM4-40GB GPUs. Table S2 reports the first-pass response accounting for both local models.

Table S2. First-pass response accounting for the two local VLMs, with 35,000 observations per model.
<table><tr><td>Outcome</td><td>Qwen2.5-VL- 32B-Instruct</td><td>InternVL3.5- 14B-Instruct</td></tr><tr><td>Valid responses</td><td>34,968</td><td>34,972</td></tr><tr><td>Correct book and author</td><td>11,105</td><td>6,873</td></tr><tr><td>Unresolved title/entity</td><td>21,683</td><td>26,584</td></tr><tr><td>Wrong entity</td><td>740</td><td>1,137</td></tr><tr><td>Abstention</td><td>1,440</td><td>378</td></tr><tr><td>Infrastructure/</td><td></td><td></td></tr><tr><td>runtime failures</td><td>32</td><td>0</td></tr><tr><td>Malformed outputs</td><td>0</td><td>28</td></tr></table>