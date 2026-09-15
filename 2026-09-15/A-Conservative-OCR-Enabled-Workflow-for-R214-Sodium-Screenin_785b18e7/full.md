# A Conservative OCR-Enabled Workflow for R214 Sodium Screening of South African Packaged Foods

Mayimunah Nagayi

Alice Scaria Khan

Tamryn Frank

Department of Computer Science

University of the Western Cape

School of Public Health

School of Public Health

University of the Western Cape

University of the Western Cape

Cape Town, South Africa

Cape Town, South Africa

Cape Town, South Africa

4163113@myuwc.ac.za

askhan@uwc.ac.za

tfrank@uwc.ac.za

Rina Swart

Department of Dietetics and Nutrition

University of the Western Cape

Cape Town, South Africa

rswart@uwc.ac.za

Clement Nyirenda

Department of Computer Science and eResearch Office

University of the Western Cape

Cape Town, South Africa

cnyirenda@uwc.ac.za

Abstract—Using food package images to monitor sodium and salt content against South Africa’s R214 sodium limits is challenging when screening decisions require product identity, nutrition facts panel evidence, reporting basis, and categoryspecific thresholds. This study presents a conservative imagebased workflow that combines region detection, optical character recognition (OCR), product identity and sodium evidence extraction, R214 category assignment, deterministic threshold comparison, and independent vision language model comparison. The evaluation used 442 packaged food products and 3 929 full package images from a real-world South African food packaging dataset. A YOLO26s small detector generated 4 195 region crops, and strict post-processing produced one sodium evidence row per product. The integrated workflow produced 290 OUTSIDE R214 SCOPE, 139 REVIEW, seven SCREEN-PASS, and six SCREEN-FAIL outcomes. The independent Qwen2.5- VL 7B vision language model workflow produced 387 OUTSIDE R214 SCOPE, 31 REVIEW, twenty SCREEN-PASS, and four SCREEN-FAIL outcomes. The workflows agreed on exact R214 category assignment for 415 of 442 products (93.9%) and on whether the assigned category was within R214 scope for 416 of 442 products (94.1%). Final screening outcome agreement was 307 out of 442 products, or 69.5%. Manual verification on 60 products showed lower strict outcome agreement than regulated status agreement, while all manual INSUFFICIENT DATA cases were kept out of SCREEN-PASS and SCREEN-FAIL by both automated workflows. The findings show that conservative imagebased screening can organise package evidence, identify clear cases, and assign uncertain cases to REVIEW rather than forcing SCREEN-PASS or SCREEN-FAIL decisions.

Index Terms—Food package images, optical character recognition, sodium regulation, vision language models, compliance screening

## I. INTRODUCTION

Food package labels are an important source of productlevel nutrition, ingredient, and marketing information. Nutrition facts panels, ingredient lists, product names, quantity declarations, and other label elements are used in food composition databases, consumer tools, nutrition research, and food supply monitoring systems [1], [2]. The Food Label Information Program (FLIP) shows how product-level label information can be structured for monitoring and research [1], while FoodSwitch shows how branded food data and barcode scanning can support consumer nutrition information [2]. In image-based work, this information must be recovered from package photographs, where text may be small, dense, multilingual, curved, or visually mixed with other package content [3], [4]. South African studies based on package photographs also show that printed label evidence can support regulation-related assessment [5]. For products that are not already available in a structured database, the physical package remains evidence that must be read and checked. This makes automated information extraction from food package images useful for regulation-related screening.

High sodium intake is associated with raised blood pressure and increased risk of cardiovascular disease [6]. South Africa introduced mandatory sodium limits through Regulation R214, which sets maximum total sodium values per 100 g foodstuff for regulated processed food categories [7]. The limits are category specific. This means that sodium screening cannot be done by reading a sodium value alone. The product type must be identified, the correct regulatory category must be assigned, the sodium value and its basis must be checked, and the value must then be compared with the correct category threshold [7], [8]. This makes R214 screening a structured evidence problem, not only a text extraction problem.

Prior South African sodium monitoring work has shown that R214 assessment depends on correct product categorisation, comparable sodium values, and expert checking [7]–[9]. Manual, laboratory-based, and database-based approaches remain important, but they are difficult to scale across large image collections and repeated market monitoring. Package images also create technical challenges, as useful label evidence may appear across several views and may be affected by small text, glare, curved surfaces, dense tables, multilingual text, and partial occlusion [3], [4]. Optical character recognition (OCR) and extraction methods based on large language models can recover useful label text [3], [12], but R214 screening still requires sodium or salt evidence, reporting basis checks, product type, category assignment, and threshold comparison [7], [8]. In this study, conservative screening means that the workflow assigns REVIEW when product identity, category evidence, sodium or salt evidence, or reporting basis is unclear, rather than forcing a SCREEN-PASS or SCREEN-FAIL decision from incomplete evidence.

This paper proposes and evaluates a conservative imagebased workflow for assessing packaged food products against official R214 sodium limits. The workflow links package evidence extraction, product identity evidence, sodium category classification, sodium and salt evidence checks, basis checking, and deterministic decision logic, with an independent Qwen2.5-VL 7B vision language model comparison used as a separate screening view. Each product is assigned a conservative screening outcome, with the outcome definitions and decision rules described in Section IV. The evaluation-set size was determined by the contents of three selected 2023 dataset folders, as described in Section III-A.

1) An image-based sodium screening workflow that combines package region detection, OCR, sodium and salt evidence selection, product identity extraction, R214 category classification, basis checking, and threshold comparison.

2) A conservative decision framework that assigns products to SCREEN-PASS, SCREEN-FAIL, REVIEW, or OUTSIDE R214 SCOPE under the R214 screening logic without forcing unclear cases into SCREEN-PASS or SCREEN-FAIL decisions.

3) An evaluation on 442 real-world South African packaged food products and 3 929 package images, reporting evidence recovery, category assignment, and conservative screening outcomes.

4) A comparison with an independent vision language model workflow and targeted manual checking to identify agreement patterns, disagreement types, and main failure sources.

The rest of this paper is organised as follows. Section II reviews related work on sodium regulation monitoring, food package OCR, product recognition, food label databases, and vision language models. Section III describes the dataset and regulation context. Section IV presents the proposed methodology. Section V reports the results and evaluation. Section VI discusses the findings, limitations, and error sources. Section VII concludes the paper.

## II. RELATED WORK

## A. Sodium Regulation and Food Label Monitoring

South African sodium regulation provides the policy context for this study. Regulation R214 defines maximum total sodium values per 100 g foodstuff for regulated processed food categories, with category-specific targets applied to product type rather than to sodium values alone [7]. Prior South African studies have assessed sodium content and industry compliance using processed food data, label-declared values, nutrition information panel data, laboratory analysis, and monitoring of regulated foodstuffs [8]–[11]. Together, these studies show that R214 screening requires correct product categorisation, comparable sodium values, and expert checking where category, panel values, and measurement basis must be interpreted together [7], [8]. Food label databases, package-based systems, and South African package photograph studies also show that label evidence can support nutrition and regulationrelated monitoring [1], [2], [5]. However, they rely mainly on structured, manual, or laboratory-supported data rather than automatically extracted package-image evidence.

## B. Food Package Text Extraction and Product Evidence

Food package images are difficult to process automatically when labels appear on curved, reflective, multilingual, or visually crowded surfaces. Nagayi et al. [3] evaluated optical character recognition (OCR) performance on real-world South African food packaging labels and reported challenges linked to dense nutrition panels, ingredient text, multilingual content, glare, curved surfaces, and varied layouts. This work shows that OCR can recover useful label text, but text recovery alone does not create a regulation-ready screening decision. The recovered text still has to be linked to product identity, sodium or salt evidence, reporting basis, and the applicable R214 category. Guimaraes et al. [4] reviewed grocery label˜ detection and recognition work and highlighted the difficulty of detecting and reading text from retail product packaging. Assiri et al. [12] studied extraction of nutritional elements and values from bilingual food labels using large language models, while Pettersson et al. [13] studied fine-grained grocery product recognition using product images and OCR text. These studies address important extraction and recognition tasks, but they do not combine food category assignment, per 100 g basis checking, sodium threshold comparison, and conservative review logic in one traceable screening workflow.

## C. Multimodal Models and Related Work Synthesis

Vision language models (VLMs) and large language models (LLMs) are increasingly used for nutrition-related tasks. Ma et al. [14] integrated VLMs into a high-throughput nutrition screening workflow and showed that multimodal models can support faster processing of nutrition-related information. This work is relevant to package-based nutrition screening, but multimodal interpretation alone does not provide the controlled evidence checks and decision logic needed for regulationspecific screening. Across sodium monitoring, food label databases, package image text extraction, product recognition, and multimodal nutrition studies, existing work addresses important parts of the task, but the reviewed studies do not present a traceable image-based workflow that connects realworld package images to category assignment, reporting-basis checking, category-specific threshold comparison, and conservative screening outcomes. This gap motivates the workflow proposed in this paper.

## III. DATASET AND REGULATION CONTEXT

## A. Dataset Source and Package Views

This study uses a real-world image dataset from the Department of Dietetics and Nutrition at the University of the Western Cape. The dataset forms part of the South African Nutrition Facts Panel project. The images were collected in 2023 by trained personnel and accessed through the Department of Dietetics and Nutrition Research project site via SharePoint. The subset used in this study was drawn from three image folders: 20230608 04 03, 20230608 04 18, and 20230612 04 18. The folders were selected after initial inspection of 2023 folders for use as the study subset, without a formal random or category-based sampling procedure. The selected folders contained a range of packaged food products and were not filtered in advance to include only products from R214 categories. There were no duplicate products in the selected subset. Each product had a unique product identifier, and all products and their associated package images in the selected folders were included, giving 442 packaged food products and 3 929 full package images. The study size was therefore determined by the contents of the selected folders rather than by a predefined sample-size target. Each product was represented by multiple package views rather than a single image. The number of images per product ranged from three to twenty-seven, with an average of 8.9 images per product. This multi-view structure was important for R214 screening, as product identity, sodium values, reporting basis information, ingredients, and quantity declarations may appear on different parts of the package. These package views therefore provided the visual evidence used for product identity extraction, sodium and salt evidence extraction, category assignment, and screening.

## B. R214 Sodium Screening Context

Regulation R214 defines maximum total sodium values per 100 g foodstuff for thirteen regulated processed food categories [7]. Since the package images used in this study were collected in 2023, the workflow used the 2019 R214 limits for screening. The official R214 categories were used as the regulatory category space. The working label space used codes 1 to 13 for the official R214 sodium categories and an operational code 14 for products outside the R214 sodium category scope based on the available evidence. Code 14 is not an official R214 foodstuff category. An OUTSIDE R214 SCOPE outcome therefore indicates that the available evidence did not support assignment to one of the thirteen R214 categories, not that sodium was absent from the product. Products with unclear identity or unclear category evidence were assigned to REVIEW during the decision stage rather than being forced into code 14. R214 screening therefore requires both category evidence and comparable sodium evidence, as described in the methodology section.

## IV. METHODOLOGY

This section describes the staged image-based workflow used to convert full package images into product-level R214 sodium screening outcomes. As shown in Fig. 1, the study used an integrated workflow, an independent workflow, and manual verification. The integrated workflow combined region detection, crop generation, OCR, evidence construction, product identity extraction, sodium and salt evidence extraction, R214 category assignment, and deterministic threshold comparison. The independent workflow used full package images with Qwen2.5-VL 7B, a vision-language model from the Qwen2.5- VL model family, to provide a separate screening view [15]. Manual verification was used on a fixed 60 product subset to check agreement patterns and error cases.

## A. Region Detection and Crop Generation

Detector training used an NVIDIA GeForce RTX 3070 graphics processing unit (GPU) with Ultralytics 8.4.47 while the full study workflow ran on the University of the Western Cape high performance computing (HPC) cluster using Slurm GPU jobs on an NVIDIA L4 GPU node with four CPU cores and 48 GB memory. The computer vision stage used a YOLO26s small detector from the You Only Look Once (YOLO) model family, implemented with the Ultralytics YOLO software [16], to locate four evidence regions: nutrition facts panel, ingredients, product name, and quantity. The detector was trained on a separate labelled image set from folder $2 0 2 3 0 6 1 4 \_ 0 4 \_ 1 8$ , so the detector training images did not overlap with the 442-product study set. Annotations were prepared in Microsoft Visual Object Tagging Tool using four tags: nfp, ingredients, Product\_name, and Quantity [17]. After a product-based split, the labelled dataset contained 176 products and 718 images: 141 products (80.1%) for training, eighteen products (10.2%) for validation, and seventeen products (9.7%) for testing. Training used YOLO26s pretrained weights, image size 896, batch size 4, 150 maximum epochs, patience 30, and seed 42. Inference on the 3 929 study images used image size 896, confidence threshold 0.35, and intersection over union threshold 0.60, with class-specific padding before crop saving.

## B. Text Recognition and Evidence Construction

Text recognition was applied to the detected region crops using PaddleOCR 3.5.0, paddlepaddle-gpu 3.3.0, and the latin\_PP-OCRv5\_mobile\_rec recognition model. Previous OCR accuracy evaluation against manually transcribed package text from the same broader image collection is reported by Nagayi et al. [3]. Text recognition was run on the HPC cluster using GPU resources. Recognised text was cleaned by normalising Unicode, removing hidden characters, replacing newlines with spaces, and collapsing repeated spaces. Weak crop-level outputs were retried with an enhanced crop variant, and the retry output was retained only when it improved the candidate score. Crop-level outputs were grouped by product and region to build product-level evidence. Region candidates were scored using task-specific cues. Nutrition facts panel candidates were scored using nutrition terms, sodium or salt terms, numeric content, per 100 g or per 100 ml cues, and table-like content. Ingredients, product name, and quantity candidates were scored using region-specific text patterns. Nutrition facts panel candidates were retained for sodium and salt parsing, and the strict evidence step selected one sodium evidence row per product.

![](images/4022ffc273c483c9783f1d59fcd42fcca4bbe1bea75bad04c2c0f02f85d1eec2.jpg)  
Fig. 1. Overview of the image-based R214 sodium screening and validation workflow. The integrated workflow produces product-level screening outcomes, the independent workflow provides a separate comparison, and manual verification checks a 60 product subset for agreement and error analysis.

## C. Product Identity, Sodium Evidence, and Category Assignment

Product identity was extracted from full package images using Qwen2.5-VL 7B through Ollama, with the model identifier qwen2.5vl:7b. Full images were used for this step to capture stylised and brand-led product names, front-of-pack text, visual context, and multiple package views that OCR crops alone may miss. The model returned product name, product type, quantity, visible brand, visible text, and a short reason. Image-level outputs were scored, and the highestscoring candidate was selected as product-level identity evidence; low-scoring or incomplete outputs were flagged for review.

Sodium and salt evidence was extracted from nutrition facts panel candidate text. The parser searched for sodium and salt labels, nearby numeric values, mg or g units, and the order of per 100 g and serving information. Values were converted to sodium in mg per 100 g where supported. Strict automatic evidence was limited to direct sodium in mg per 100 g or salt in g per 100 g with clear column order and no quality-assurance flag. Other conversions or unclear cases were assigned to REVIEW. Salt conversion used standard sodium– salt conversion factors [18]. Products without usable values, comparable units, or a clear per 100 g basis were assigned to REVIEW.

The R214 category assignment used product-level identity evidence. The classifier used strict rules first, followed by a Groq API assisted fallback with llama-3.3-70b-versatile at temperature 0 for uncertain cases involving brand-specific or non-standard product wording [19]. The category space used official R214 categories 1 to 13 and operational code 14 for products not applicable to R214 based on available evidence. Products with unclear category evidence were assigned to category review rather than being forced into code 14.

## D. Screening Outcomes, Decision Rules, and Evaluation Measures

The study used four conservative screening outcomes. These outcomes were used for screening and prioritisation, not for final legal determination.

• SCREEN-PASS. The available evidence supported an applicable R214 category, comparable sodium evidence, and a value at or below the relevant R214 limit.

• SCREEN-FAIL. The available evidence supported an applicable R214 category, comparable sodium evidence, and a value above the relevant R214 limit.

• REVIEW. The available evidence was incomplete, unclear, conflicting, or not reliable enough for a SCREEN-PASS or SCREEN-FAIL decision. This included unclear product identity, uncertain category assignment, weak sodium or salt extraction, unclear reporting basis, or noncomparable units.

OUTSIDE R214 SCOPE. The available product evidence did not support assignment to an applicable R214 sodium category. In this study, OUTSIDE R214 SCOPE is only an R214 screening outcome, not a statement about all possible food labelling or food composition regulations.

The final decision step combined category evidence, sodium evidence, and the 2019 R214 limits. If category evidence required review, the final decision was REVIEW. If a product was assigned operational code 14 without category review, the final decision was OUTSIDE R214 SCOPE. If a product was assigned an R214 category but sodium evidence was not READY, the final decision was REVIEW. If a product had both an applicable R214 category and READY sodium evidence, the sodium value was compared with the relevant 2019 R214 limit. Values at or below the limit were assigned SCREEN-PASS, and values above the limit were assigned SCREEN-FAIL. Additional safety checks flagged laboratorymethod text, extreme or implausibly low sodium values, unclear column order, unusual salt conversions, and uncertain category assignments. Where comparison was possible but further checking was still needed, provisional SCREEN-PASS or SCREEN-FAIL labels were retained for audit, while these products were reported as REVIEW.

Detector performance was reported using precision, recall, and mean average precision (mAP). Agreement between automated workflows was measured using category agreement, regulated status agreement, final decision group agreement, and direct SCREEN-PASS versus SCREEN-FAIL conflict. Category agreement counted products where both workflows assigned the same R214 category or the same outside-scope grouping. Regulated status agreement counted products where both workflows agreed on whether the product was assigned to an applicable R214 sodium category. Final decision group agreement compared the four reported screening outcomes directly. Direct SCREEN-PASS versus SCREEN-FAIL conflict counted cases where one workflow assigned SCREEN-PASS and the other assigned SCREEN-FAIL.

## E. Independent Workflow and Manual Verification

The independent workflow used Qwen2.5-VL 7B, a vision language model run through Ollama at temperature 0, on full package images only, and did not use integrated workflow outputs such as YOLO crops, PaddleOCR text, selected sodium evidence, category assignments, or final decisions. Qwen performed image evidence extraction, product consolidation, R214 category assessment, sodium assessment, and final screening decision, while Python handled execution, structured-output parsing, validation checks, retries, and result logging. Generic category rules were used to reduce forced assignment of out-of-scope products into official R214 categories. SCREEN-PASS and SCREEN-FAIL decisions were kept only when the sodium or salt evidence was internally valid and clearly comparable per 100 g; otherwise, the product was assigned to REVIEW.

Manual verification was conducted on a fixed subset of 60 products selected at product level. Products were matched across the manual file, integrated workflow output, and independent workflow output using the product identifier. Manual checking used the original package images, including evidence across package views where needed, and extracted evidence files to confirm product identity, R214 category, sodium or salt evidence, reporting basis, and screening outcome. The subset focused on important screening cases, including regulated or priority products, uncertain cases, and disagreement cases between the two automated workflows.

## V. RESULTS AND EVALUATION

This section reports the processing outputs, workflow screening outcomes, automated agreement, and manual verification results. Detector performance was evaluated using precision, recall, and mean average precision (mAP), while workflow agreement was evaluated using the product-level agreement measures defined in the methodology section.

## A. Processing and Evidence Recovery

The detector provided usable localisation for the downstream R214 screening workflow. On the validation split, it achieved 0.744 precision, 0.693 recall, 0.718 mAP at 0.5, and 0.480 mAP at 0.5:0.95. On the held-out test split, it achieved 0.897 precision, 0.877 recall, 0.887 mAP at 0.5, and 0.638 mAP at 0.5:0.95. The nutrition facts panel class had the strongest test result, with 0.995 mAP at 0.5, while quantity was the weakest class, with 0.778 mAP at 0.5. The trained detector was applied to all 3 929 full package images and generated 4 195 region crops from 2 901 images. PaddleOCR processed all crops with no processing errors. Product-level text evidence was constructed for all 442 products, with 952 nutrition facts panel candidates retained for sodium and salt parsing. The sodium evidence step produced one strict sodium evidence row per product. Of the 442 products, 207 were marked READY for automatic sodium comparison and 235 were assigned to REVIEW. Category assignment was rule-based for 302 products, used the Groq fallback for 49 products, and required category review for 91 products. Table I summarises detector performance, crop recovery, and productlevel evidence preparation.

TABLE I  
PROCESSING AND EVIDENCE RECOVERY SUMMARY
<table><tr><td>Measure Result</td></tr><tr><td>Detector performance</td></tr><tr><td>Validation precision / recall 0.744 / 0.693</td></tr><tr><td>Validation mAP@0.5 / mAP@0.5:0.95 0.718 / 0.480</td></tr><tr><td>Test precision / recall 0.897  / 0.877</td></tr><tr><td>Test mAP@0.5 / mAP@0.5:0.95 0.887  / 0.638</td></tr><tr><td>Nutrition facts panel test mAP@0.5 0.995</td></tr><tr><td>Quantity test mAP@0.5 0.778 Image processing and crop recovery</td></tr><tr><td>Images processed / with crops 3 929 /  2 901</td></tr><tr><td>Total crops generated 4195</td></tr><tr><td>Product name / ingredients crops 1 461 /  1 093</td></tr><tr><td>Nutrition facts panel / quantity crops 952  /  689</td></tr><tr><td>Evidence and category preparation</td></tr><tr><td>Strict sodium evidence rows 442</td></tr><tr><td>Sodium READY / REVIEW products 207  /  235</td></tr><tr><td>Salt converted products 15</td></tr><tr><td>Rejected sodium candidates 150</td></tr><tr><td>Category assignment: rule / Groq / review 302 / 49 / 91</td></tr><tr><td></td></tr></table>

## B. Workflow Outcomes and Automated Agreement

The integrated workflow produced six detailed decision labels. For reporting and comparison with the independent workflow, provisional SCREEN-PASS and SCREEN-FAIL labels were retained for audit but reported as REVIEW when further checking was required. The integrated workflow therefore assigned 290 products to OUTSIDE R214 SCOPE, 139 to REVIEW, seven to SCREEN-PASS, and six to SCREEN-FAIL. The independent Qwen2.5-VL 7B workflow assigned 387 products to OUTSIDE R214 SCOPE, 31 to REVIEW, twenty to SCREEN-PASS, and four to SCREEN-FAIL. Agreement values were calculated at product level across all 442 products. Category agreement between the two automated workflows was 415 out of 442 products, or 93.9%. Regulated status agreement was 416 out of 442 products, or 94.1%. Final decision group agreement was 307 out of 442 products, or 69.5%. One direct SCREEN-PASS versus SCREEN-FAIL conflict was found between the integrated and independent workflows. Table II summarises the reported outcomes and automated agreement.

TABLE II  
WORKFLOW OUTCOMES AND AUTOMATED AGREEMENT
<table><tr><td>Measure</td><td>Integrated</td><td>Independent</td></tr><tr><td>Reported screening outcomes</td><td></td><td></td></tr><tr><td>OUTSIDE R214 SCOPE</td><td>290</td><td>387</td></tr><tr><td>REVIEW</td><td>139</td><td>31</td></tr><tr><td>SCREEN-PASS</td><td>7</td><td>20</td></tr><tr><td>SCREEN-FAIL</td><td>6</td><td>4</td></tr><tr><td>Total</td><td>442</td><td>442</td></tr><tr><td>Agreement between automated workflows</td><td></td><td></td></tr><tr><td>Category agreement</td><td>415/442 (93.9%)</td><td></td></tr><tr><td>Regulated status agreement</td><td></td><td>416/442 (94.1%)</td></tr><tr><td>Final decision group agreement</td><td></td><td>307/442 (69.5%)</td></tr><tr><td>Direct SCREEN-PASS versus SCREEN-FAIL conflict</td><td></td><td>1</td></tr></table>

## C. Manual Verification

Manual verification was conducted on a fixed, targeted subset of 60 products selected at product level. Manual review assigned twenty-six products to SCREEN-PASS, twentyone to SCREEN-FAIL, eight to INSUFFICIENT DATA, and five to OUTSIDE R214 SCOPE. For the same products, the integrated workflow assigned seven to SCREEN-PASS, six to SCREEN-FAIL, and 47 to REVIEW. The independent workflow assigned sixteen to SCREEN-PASS, four to SCREEN-FAIL, 24 to REVIEW, and sixteen to OUTSIDE R214 SCOPE. The 60-product subset covered 20 of the 24 SCREEN-PASS/SCREEN-FAIL products in the independent workflow, including all four independent SCREEN-FAIL products. Manual verification agreement was calculated at product level across the subset. Strict outcome agreement with manual review was 12 out of 60 products, or 20.0%, for the integrated workflow, and 20 out of 60 products, or 33.3%, for the independent workflow. Of the thirteen SCREEN-PASS or SCREEN-FAIL decisions retained by the integrated workflow after conservative review, twelve agreed with the manual screening direction. Regulated status agreement was 55 out of 60 products, or 91.7%, for manual versus the integrated workflow, and 47 out of 60 products, or 78.3%, for manual versus the independent workflow. All eight manual INSUFFICIENT DATA cases were kept out of SCREEN-PASS and SCREEN-FAIL by both automated workflows. Direct SCREEN-PASS versus SCREEN-FAIL conflicts were limited to one case for manual versus the integrated workflow and three cases for manual versus the independent workflow. Table III summarises the manual verification checks.

TABLE III  
MANUAL VERIFICATION CHECKS ON THE 60 PRODUCT SUBSET
<table><tr><td>Check</td><td>Integrated</td><td>Independent</td></tr><tr><td>Strict agreement with manual</td><td>12/60 (20.0%)</td><td>20/60 (33.3%)</td></tr><tr><td>Regulated status agreement</td><td>55/60 (91.7%)</td><td>47/60 (78.3%)</td></tr><tr><td>INSUFFICIENT DATA not SCREEN-</td><td>8/8</td><td>8/8</td></tr><tr><td>PASS/SCREEN-FAIL Direct SCREEN-PASS versus SCREEN-FAIL</td><td>1</td><td>3</td></tr></table>

## VI. DISCUSSION

The results support the conservative decision design described in Section IV. The detector recovered useful package regions, with stronger nutrition facts panel localisation than quantity localisation, but downstream uncertainty remained after text recognition, sodium parsing, basis checking, and category assignment. This shows that the main difficulty was not only reading label text, but converting package evidence into reliable product-level evidence for R214 screening. Relevant evidence could also occur across different package views and had to be linked to the correct reporting basis and regulatory category.

The comparison between workflows shows that broad R214 relevance was more stable than exact screening outcome. Category agreement and regulated status agreement were higher than final decision group agreement, while the integrated workflow assigned more products to REVIEW than the independent workflow. Products requiring further checking remained under REVIEW even when a threshold direction could be calculated. Manual verification showed the same conservative pattern. Of the thirteen SCREEN-PASS or SCREEN-FAIL decisions retained by the integrated workflow, twelve agreed with the manual screening direction, while manual checking could inspect evidence across package views where needed. All eight manual INSUFFICIENT DATA cases were kept out of SCREEN-PASS and SCREEN-FAIL by both automated workflows. The workflow is therefore best understood as a screening and prioritisation approach that retains clearer decisions and directs uncertain products to review.

## VII. CONCLUSION

This paper presented a conservative image-based workflow for screening packaged food products against South Africa’s R214 sodium limits. On 442 products and 3 929 full package images, the integrated workflow showed a more conservative decision pattern than the independent Qwen2.5-VL 7B workflow, with more products assigned to REVIEW and only one direct SCREEN-PASS versus SCREEN-FAIL conflict between the two automated workflows. Manual verification showed stronger alignment on regulated status than on exact screening outcome, and all manual INSUFFICIENT DATA cases were kept out of SCREEN-PASS and SCREEN-FAIL. The workflow is not intended to replace human judgement or official compliance assessment; its value is in organising package image evidence, identifying clear screening cases, and prioritising uncertain products for manual review. Future work should expand manual verification and extend the workflow to additional labelling regulations.

## ACKNOWLEDGMENT

Most computational processing in this study was performed using the University of the Western Cape’s High Performance Computing and Research Cloud facilities (https://eresearch. uwc.ac.za).

## REFERENCES

[1] M. Ahmed, A. Schermel, J. J. Lee, M. Weippert, B. Franco-Arellano, and M. R. L’Abbe, “Development of the Food Label Information Program: a´ comprehensive Canadian branded food composition database,” Frontiers in Nutrition, vol. 8, Art. no. 825050, 2022, doi: https://doi.org/10.3389/ fnut.2021.825050.

[2] E. Dunford, H. Trevena, C. Goodsell, K. H. Ng, J. Webster, A. Millis, S. Goldstein, O. Hugueniot, and B. Neal, “FoodSwitch: a mobile phone app to enable consumers to make healthier food choices and crowdsourcing of national food composition data,” JMIR mHealth and uHealth, vol. 2, no. 3, Art. no. e37, 2014, doi: https://doi.org/10.2196/mhealth.3230.

[3] M. Nagayi, A. S. Khan, T. Frank, R. Swart, and C. Nyirenda, “Evaluating OCR performance on food packaging labels in South Africa,” in Artificial Intelligence Research: 6th Southern African Conference, SACAIR 2025, Cape Town, South Africa, December 1–5, 2025, Proceedings, A. Gerber and A. W. Pillay, Eds. Cham, Switzerland: Springer, 2026, pp. 127–143, doi: https://doi.org/10.1007/978-3-032-11733-5 8.

[4] V. Guimaraes, J. Nascimento, P. Viana, and P. Carvalho, “A review˜ of recent advances and challenges in grocery label detection and recognition,” Applied Sciences, vol. 13, no. 5, Art. no. 2871, 2023, doi: https://doi.org/10.3390/app13052871.

[5] S. Abdool Karim, T. Frank, A. S. Khan, M. G. Tlhako, S. K. Joni, E. C. Swart, et al., “An assessment of compliance with proposed regulations to restrict on package marketing of packaged foods to improve nutrition in South Africa,” BMC Nutrition, vol. 11, Art. no. 17, 2025, doi: https: //doi.org/10.1186/s40795-025-01007-3.

[6] World Health Organization, Sodium Intake for Adults and Children. Geneva, Switzerland: World Health Organization, 2012. Available: https: //www.who.int/publications/i/item/9789241504836

[7] Department of Health, South Africa, “Regulations relating to the reduction of sodium in certain foodstuffs and related matters,” Government Notice R.214, Government Gazette No. 36274, 2013. Available: https://www.gov.za/sites/default/files/gcis document/201409/ 36274rg9934gon214.pdf

[8] K. E. Charlton, B. Pretorius, R. Shakhane, P. Naidoo, H. Cimring, K. Hussain, B. Nojilana, and J. Webster, “Compliance of the food industry with mandated salt target levels in South Africa: towards development of a monitoring and surveillance framework,” Journal ofFood Composition and Analysis, vol. 126, Art. no. 105908, pp. 1–7, 2024, doi: https://doi. org/10.1016/j.jfca.2023.105908.

[9] B. van der Westhuizen, T. Frank, S. Abdool Karim, and E. C. Swart, “Determining food industry compliance to mandatory sodium limits: successes and challenges from the South African experience,” Public Health Nutrition, vol. 26, no. 11, pp. 2551–2558, 2023, doi: https://doi. org/10.1017/S1368980023000757.

[10] S. A. E. Peters, E. Dunford, L. J. Ware, T. Harris, A. Walker, M. Wicks, T. van Zyl, B. Swanepoel, K. E. Charlton, M. Woodward, J. Webster, and B. Neal, “The sodium content of processed foods in South Africa during the introduction of mandatory sodium limits,” Nutrients, vol. 9, no. 4, Art. no. 404, 2017, doi: https://doi.org/10.3390/nu9040404.

[11] B. Swanepoel, L. Malan, P. H. Myburgh, R. Schutte, K. Steyn, and E. Wentzel-Viljoen, “Sodium content of foodstuffs included in the sodium reduction regulation of South Africa,” Journal of Food Composition and Analysis, vol. 63, pp. 73–78, 2017, doi: https://doi.org/10.1016/j.jfca. 2017.07.040.

[12] F. Y. Assiri, M. D. Alahmadi, M. A. Almuashi, and A. M. Almansour, “Extract nutritional information from bilingual food labels using large language models,” Journal ofImaging, vol. 11, no. 8, Art. no. 271, 2025, doi: https://doi.org/10.3390/jimaging11080271.

[13] T. Pettersson, M. Riveiro, and T. Lofstr¨ om, “Multimodal fine-grained¨ grocery product recognition using image and OCR text,” Machine Vision and Applications, vol. 35, Art. no. 79, 2024, doi: https://doi.org/10.1007/ s00138-024-01549-9.

[14] P. Ma, Y. Wu, N. Yu, Y. Zhang, M. Backes, Q. Wang, and C. Wei, “Integrating Vision-Language Models for Accelerated High-Throughput Nutrition Screening,” Advanced Science, vol. 11, no. 34, Art. no. e2403578, 2024, doi: https://doi.org/10.1002/advs.202403578.

[15] S. Bai et al., “Qwen2.5-VL Technical Report,” arXiv preprint arXiv:2502.13923, 2025. Available: https://arxiv.org/abs/2502.13923

[16] G. Jocher, J. Qiu, and A. Chaurasia, “Ultralytics YOLO,” Zenodo, version 8.4.47, May 2026, doi: https://doi.org/10.5281/zenodo.20052427.

[17] Microsoft, “Visual Object Tagging Tool (VoTT),” GitHub repository. Available: https://github.com/microsoft/VoTT

[18] World Health Organization, “Sodium reduction,” Fact sheet. Available: https://www.who.int/news-room/fact-sheets/detail/sodium-reduction

[19] Groq, “Groq API Reference,” Groq Docs. Available: https://console.groq.com/docs/api-reference