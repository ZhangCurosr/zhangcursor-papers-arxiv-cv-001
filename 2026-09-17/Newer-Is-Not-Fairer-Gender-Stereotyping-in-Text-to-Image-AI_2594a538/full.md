# Newer Is Not Fairer: Gender Stereotyping in Text-to-Image AI Across Model Generations

Shesh Narayan Gupta<sup>1</sup> and Nik Bear Brown<sup>1</sup>

<sup>1</sup>College of Engineering, Northeastern University, Boston, MA 02115, USA

{gupta.shes, ni.brown}@northeastern.edu

## Abstract

Text-to-image generative models are widely used in professional and creative settings, yet how they represent gender across occupations—and whether newer models are fairer—remains poorly understood across multiple generations. We evaluate gender representation across 20 occupations, 5 prompt templates, and 4 Stable Difusion model generations (SD 1.5, SD 2.1, SDXL, SD 3 Medium), generating 8,000 images with n = 100 per occupation-model cell (5 prompts × 20 images), and classifying all with DeepFace. Across the 8,000 open-source images, 76.4% show male subjects (95% CI [75.1%, 78.7%], p < 2.2 × 10<sup>−16</sup>, Benjamini–Hochberg adjusted). More strikingly, 57.6% of images for historically female-coded occupations show male subjects (raw p = 3.43 × 10<sup>−22</sup>, BH-adjusted p = 1.71 × 10<sup>−21</sup>). All nine significant tests reported in this paper survive BH correction across 10 tests. When compared against U.S. Bureau of Labor Statistics workforce data, models underrepresent women by 20–46pp on average, with particularly large deviations for near gender-balanced occupations: scientist (48% female in BLS, 82–99% male in model outputs) and cleaner (46% female in BLS, 80–92% male in outputs). Model generations do not improve steadily: bias worsens from SD 1.5 to SDXL before partially recovering in SD 3 Medium. A preliminary comparison with GPT-image-1 on five occupations suggests lower bias than open-source models, though the practical efect is small (Cramér’s V = 0.080) and the comparison is exploratory. No model achieves gender parity.

Keywords: text-to-image generation, gender bias, occupational stereotyping, bias amplification, Stable Difusion, prompt sensitivity, DeepFace, workforce demographics

## 1 Introduction

Text-to-image (T2I) models are now embedded in tools used for content creation, marketing, education, and design. When someone prompts one of these systems with “a photo of a surgeon” or “a photo of a nurse,” the model draws on patterns learned during training to decide what that person looks like. If those patterns reflect historical stereotypes rather than the actual demographics of who holds these jobs today, the images produced can reinforce those stereotypes at scale.

The question of whether T2I model bias improves across generations is practically important. The Stable Difusion family is among the most widely adopted open-source T2I systems, with successive versions representing substantial architectural and training advances. If widely-used models are more biased than their predecessors, users and developers need to know. Prior work has documented gender bias in T2I systems, mostly in earlier models and with limited cross-generation comparisons [1, 2, 3, 4, 5]. This paper fills that gap with a controlled longitudinal benchmark.

We generate 8,000 images across 20 occupations, 5 prompt phrasings, and 4 Stable Difusion generations (n = 100 per occupation-model cell), classify all with DeepFace, and compare outputs against BLS workforce demographics. All reported tests are corrected for multiple comparisons using the Benjamini–Hochberg procedure.

Our main contributions are:

1. The first controlled longitudinal comparison of occupational gender stereotyping across four Stable Difusion generations in a single unified experiment, with confidence intervals and BH multiple comparisons correction reported throughout.

2. A bias amplification analysis comparing model outputs against BLS workforce demographics, showing that models underrepresent women by 20–46pp on average, with the largest deviations occurring for near gender-balanced occupations.

3. Evidence of a non-linear bias trajectory in which SDXL shows higher gender skew than both SD 1.5 and SD 3 Medium, revealing a deployment gap in which model progress does not guarantee fairer outputs for users of established model versions.

4. A prompt sensitivity analysis showing that output gender composition varies substantially with prompt phrasing and that this sensitivity increases across model generations.

## 2 Related Work

## 2.1 Gender Bias in Text-to-Image Models

Gender stereotypes in AI systems have been documented across modalities. Bolukbasi et al. [6] showed that word embeddings encode occupational gender associations that propagate into downstream tasks. Zhao et al. [7] found that visual question answering models amplify gender bias beyond what is present in training data. In the T2I space, Bianchi et al. [1] examined DALL-E 2 and Stable Difusion and found that occupational prompts produce images reflecting historical stereotypes rather than current workforce demographics. Cho et al. [2] built DALL-Eval, a benchmark for social bias in DALL-E systems. Mandal et al. [3] showed that neutral prompts can produce more stereotyped results than explicitly gendered ones.

Most directly related to our work, Luccioni et al. [4] systematically audited gender and ethnicity representation across 150 occupations in Stable Difusion and other T2I models, finding pervasive male and white dominance—a finding our study corroborates and extends across four model generations. Friedrich et al. [5] proposed Fair Difusion, a method for steering T2I outputs toward more equitable demographic representation. Our study complements this work by providing the first controlled longitudinal comparison across four Stable Difusion generations and by introducing bias amplification analysis—comparing outputs against BLS workforce data—which reveals large deviations for near-balanced occupations that aggregate stereotype metrics do not capture.

## 2.2 Occupational Stereotyping and Representation

Many professions remain gender-segregated: women are underrepresented in STEM and overrepresented in caregiving and education [8]. AI systems trained on internet data absorb these patterns [9]. Buolamwini and Gebru [10] demonstrated that commercial face analysis systems were substantially less accurate for darker-skinned women due to skewed training data. Our work extends this tradition to T2I generation, asking what faces models create for occupational prompts and how far those outputs deviate from demographic reality.

## 2.3 Prompt Phrasing and Model Output

How a prompt is worded can substantially change T2I model outputs. Oppenlaender [11] catalogued prompt engineering strategies and their efects on image content. Weidinger et al. [12] identified stereotype perpetuation as a key risk dimension for large-scale AI systems. Our prompt sensitivity analysis quantifies this directly across four model generations, finding that sensitivity increases even as average bias partially decreases in newer models.

## 3 Methodology

## 3.1 Experimental Design

The experiment uses a $4 \times 2 0 \times 5 \times 2 0$ design: 4 models × 20 occupations × 5 prompt templates × 20 images per prompt = 8,000 open-source model images. Each occupation-model cell therefore contains $n = 1 0 0$ images (5 prompts × 20 images), giving 95% confidence intervals of approximately ±9.8pp at $p = 0 . 5$ for occupation-level estimates, and ±1.8pp at the model level $( n = 2 , 0 0 0$ per model). An additional 100 GPT-image-1 images cover 5 spotlight occupations using a single prompt template (n = 20 per occupation) for a preliminary exploratory comparison only.

## 3.2 Occupation Selection

We selected 20 occupations to represent a range of professional domains, income levels, and historical gender patterns, following the approach of prior bias benchmarks [1, 4]. The goal was to include occupations with strong historical male skew, strong historical female skew, and at least two nearbalanced occupations to test whether models deviate from a roughly equal baseline. The final set was not drawn from prior work verbatim but was selected to cover diverse sectors—healthcare, law, education, skilled trades, technology, service, and creative fields—while keeping the total manageable for a controlled experiment.

Historical gender distributions are drawn from the U.S. Bureau of Labor Statistics Current Population Survey, Table 11 (2023 annual averages) [8]. The ten historically male-skewed occupations and their BLS mappings are: engineer (SOC 17-2000, 16% female), CEO/chief executive (SOC 11-1011, 31% female), surgeon (SOC 29-1248, 22% female), pilot (SOC 53-2011, 9% female), construction worker/laborer (SOC 47-2061, 4% female), scientist (SOC 19-0000, 48% female), judge (SOC 23-1023, 34% female), firefighter (SOC 33-2011, 8% female), mechanic/automotive technician (SOC 49-3023, 4% female), and programmer/computer programmer (SOC 15-1251, 22% female). The ten historically female-skewed occupations are: nurse/registered nurse (SOC 29-1141, 87% female), teacher/elementary teacher (SOC 25-2000, 74% female), receptionist (SOC 43-4171, 90% female), cleaner (see note below), babysitter (SOC 39-9011 childcare workers, 94% female), librarian (SOC 25-4022, 84% female), social worker (SOC 21-1029, 82% female), florist/floral designer (SOC 27-1023, 65% female), hair stylist/cosmetologist (SOC 39-5012, 92% female), and preschool teacher (SOC 25-2010, 95% female).

Three BLS mapping decisions warrant explicit acknowledgment. First, the prompt “cleaner” is ambiguous: BLS distinguishes janitors and building cleaners (SOC 37-2011, approximately 29% female) from maids and housekeeping cleaners (SOC 37-2012, approximately 89% female). These have very diferent gender compositions. We used 46% female as a weighted average across both categories, but this figure should be treated with caution; the actual reference value depends on which mental image “cleaner” evokes, which likely varies across individuals and model training corpora. Second, BLS has no “babysitter” occupational category; we mapped this to childcare workers (SOC 39-9011, 94% female), which is the closest available category. The prompt “babysitter” may evoke a narrower role than this category captures. Third, “scientist” maps to the broad BLS category “life, physical, and social science occupations” (SOC 19-0000, 48% female), which aggregates subfields with very diferent gender compositions—physicists are approximately 20% female, biologists approximately 50%, and psychologists approximately 75%. The 48% figure represents the aggregate and may not reflect the subfield that any given generated image evokes. These ambiguities do not afect the direction of our findings but should be borne in mind when interpreting occupation-level amplification gaps.

## 3.3 Prompt Templates

Five prompt templates per occupation measure sensitivity to phrasing:

1. “a photo of a {occupation}”

2. “a professional photograph of a {occupation} at work”

3. “a realistic image of a {occupation} in a workplace setting”

4. “a person working as a {occupation}”

5. “a headshot of a {occupation}”

All templates are gender-neutral. For the GPT-image-1 preliminary comparison, only the first template was used, applied to the 5 most stereotyped occupations identified after all four open-source models completed generation.

## 3.4 Models and Generation Settings

Four Stable Difusion variants are evaluated: SD 1.5 (runwayml/stable-difusion-v1-5), SD 2.1 Base (stabilityai/stable-difusion-2-1-base), SDXL (stabilityai/stable-difusion-xl-base-1.0), and SD 3 Medium (stabilityai/stable-difusion-3-medium-difusers). All four were run locally on a consumer GPU (NVIDIA RTX 4060, 8 GB VRAM) using Hugging Face Difusers in float16 precision: 30 inference steps, guidance scale 7.5, 512×512 resolution, 20 fixed random seeds applied consistently across all models and occupations. GPT-image-1 was generated via the OpenAI API at 1024×1024 resolution.

## 3.5 Demographic Classification and Validation

Gender is classified using DeepFace [13] with enforce\_detection=False, assigning each image a dominant gender label (Man or Woman). DeepFace was trained primarily on real photographs rather than AI-generated images, which may produce a distribution shift afecting classification accuracy. To assess this, one of the authors manually reviewed a random sample of 50 images drawn across all models and occupations, labeling each for apparent gender and comparing against DeepFace output. Agreement was 96% (48/50 images). We acknowledge the limitations of this validation: the sample is small relative to 8,100 total images, and a single reviewer provides no inter-rater reliability measure. These are genuine weaknesses. We argue, however, that the primary findings—involving male classification rates of 76.4% against a 50% baseline—would require systematic, directionally consistent classifier error of an implausible magnitude to explain away. Of 80 occupation-model cells across all four open-source models, 16 fall within the ±9.8pp CI of 50% and are explicitly flagged as indicative throughout the Results section.

Apparent-gender classification is a proxy for perceived gender presentation rather than selfidentified gender, and accuracy varies across skin tones and image styles. All 8,100 images had faces detected.

## 3.6 Metrics and Statistical Tests

Stereotype Score: |male\_pct − 50|, ranging from 0 (balanced) to 50 (fully skewed).

Amplification Gap: model\_female\_pct−BLS\_female\_pct in percentage points. Negative values mean the model shows more men than the actual workforce.

Prompt Sensitivity Index: Standard deviation of male\_pct across the 5 prompt templates per occupation-model cell (n = 20 per prompt).

Statistical tests: Binomial tests for overall male dominance vs. 50% chance. Chi-square tests with Cramér’s V for pairwise model comparisons at image level. All p-values are adjusted using the Benjamini–Hochberg (BH) false discovery rate procedure applied to all 10 tests reported in this paper. All findings described as significant survive this correction. The one non-significant comparison (SD 1.5 vs. SD 2.1, raw $p = 0 . 7 9 2 )$ also does not survive correction and is reported as such throughout.

## 4 Results

## 4.1 Overall Male Dominance

Across all 8,000 open-source model images, 76.4% are classified as male (95% CI [75.1%, 78.7%], binomial test $p < 2 . 2 \times 1 0 ^ { - 1 6 }$ , BH-adjusted). This holds for every model and every occupation category. Figure 1 shows the stereotype score heatmap.

![](images/bfceed51f556dab06f2e1339281f190a6e747925bc48a0c69cb9abac077297fd.jpg)  
Figure 1: Stereotype score heatmap across 20 occupations and 4 open-source models. Red = stronger gender skew. = historically male-skewed, = historically female-skewed.

## 4.2 Gender Distribution by Occupation

Figure 2 shows male percentage per occupation and model $( n = 1 0 0 \ \mathrm { p e r }$ cell, 95% $\mathrm { C I } \approx \pm 9 . 8 \mathrm { p p } )$ For strongly skewed occupations, CIs do not afect the conclusions: programmer (97–100% male across all models), construction worker (97–100%), firefighter (94–100%), engineer (96–97%), and scientist (82–99%) all have lower CI bounds well above 50%. Sixteen of 80 occupation-model cells fall within ±9.8pp of 50% and are treated as indicative rather than definitive; these are flagged in Section 4.4 and listed in the Limitations.

![](images/b980bb3493d4c4fe2864cf72002a563be9dfa8b938d51abc56d0f1a5228bb449.jpg)  
Figure 2: Percentage of male-presenting images by occupation and model (n = 100 per cell, 95% $\mathrm { C I } \approx \pm 9 . 8 \mathrm { p p } )$ . Dashed line = 50% balance.

## 4.3 Bias Amplification Against Workforce Reality

Figure 3 shows how far model outputs deviate from BLS workforce demographics. The amplification gap is substantial across all models and occupations.

![](images/4a378f63ee3710868091362ec064aa9dad57482992b555c1b46365cb6e220e7c.jpg)  
Figure 3: Amplification gap heatmap: percentage points by which each model underrepresents women relative to BLS data. Nearly all values are negative. Exceptions are mechanic and pilot in some models.

Mean amplification gaps: SD 1.5 (−27.1pp overall, −41.1pp on female-coded occupations), SD 2.1 (−27.6pp, −39.8pp), SDXL (−31.2pp, −45.9pp), SD 3 Medium (−20.4pp, −26.0pp). The five worst individual cases are social worker in SD 2.1 (BLS: 82% female, model: 22% female, $\mathrm { g a p } \colon - 6 2 \mathrm { p p } )$ , social worker in SDXL (−62pp), hair stylist in SDXL (BLS: 92% female, model: 31% female, −61pp), social worker in SD 1.5 (−60pp), and teacher in SD 2.1 (BLS: 74% female, model: 17% female, −57pp). These gaps are 4–6 times the occupation-level CI of ±9.8pp and are unambiguous.

Near-balanced occupations show particularly striking deviations. Scientist is 48% female in BLS data yet models generate only 1–18% female scientist images (gap 30–47pp). Cleaner is 46% female in BLS data yet models generate only 8–20% female cleaner images (gap 26–42pp). These deviations are 3–5 times the CI width. Note that the 46% cleaner reference is a weighted average across two BLS subcategories with very diferent gender compositions—janitors and building cleaners (SOC 37-2011, ∼29% female) and maids and housekeeping cleaners (SOC 37-2012, ∼89% female). The direction of the finding holds regardless of which subcategory is used as the reference: even against the more male-skewed janitors figure of 29% female, models generating 8–20% female cleaner images still show a gap of 9–21pp in the same direction. Figure 4 shows BLS reality alongside model outputs for all 20 occupations.

![](images/4de3506cd9ac82c2961b675edcefef12987519fd34fec1c5ed84462fa4a87339.jpg)  
Figure 4: BLS workforce female percentage (dark blue) alongside model outputs for all 20 occupations. Near-balanced occupations such as scientist and cleaner show some of the largest deviations.

## 4.4 Cross-Directional Bias

Figure 5 shows male percentages for all 10 historically female-skewed occupations. Across all 4,000 images of these occupations, 57.6% are classified as male (raw $p = 3 . 4 3 \times 1 0 ^ { - 2 2 }$ , BH-adjusted $p = 1 . 7 1 \times 1 0 ^ { - 2 1 } )$ Six occupations show male majority across all four models with CI lower bounds clearly above 50%: cleaner (80–92% male), teacher (76–83% in SD 1.5–SD 2.1–SDXL), social worker (71–80% in SD 1.5–SD 2.1–SDXL), librarian (33–79%), florist (51–69%), and hair stylist (52– 70%). Of the 16 boundary cells identified across all open-source results, the most notable are SD 3 Medium’s teacher (45% male, ±9.8pp) and social worker (45% male, ±9.8pp), which straddle 50% and should be read as indicative of improvement rather than definitive female majority.

Cross-Directional Bias: Historically Female Occupations Values above 50% indicate reversed gender stereotyping  
![](images/3c98a50b63693b40c4235fa8d555aeb96fb72ac5619c4ab8dd8112989cfe3b63.jpg)  
Figure 5: Male percentage for historically female-skewed occupations $( n = 1 0 0 \ \mathrm { p e r }$ cell, 95% CI $\approx \pm 9 . 8 \mathrm { p p } )$ . Values above 50% indicate reversed gender stereotyping.

On female-skewed occupations, SD 3 Medium generates 45.2% male images versus SDXL’s 65.5% $( \chi ^ { 2 } = 8 2 . 5 5 , p < 0 . 0 0 1$ , Cramér’s V = 0.199, BH-adjusted). SD 3 Medium shows female majority for 7 of 10 historically female occupations versus 2–4 for earlier models, though pilot (98% male in SD 3 Medium, up from 80–88% in earlier models) is a notable regression.

## 4.5 Model Evolution and the Deployment Gap

Figure 6 shows the bias trajectory. Mean stereotype scores: SD $1 . 5 \ ( 2 8 . 9 )  \mathrm { S D ~ 2 . 1 ~ ( 3 1 . 5 ) } $ SDXL $( 3 2 . 5 )  \mathrm { S D }$ 3 Medium (29.1). Table 1 shows pairwise chi-square tests with BH-adjusted p-values and Cramér’s V, all computed from image-level contingency tables. SD 1.5 and SD 2.1 are statistically indistinguishable (raw $p = 0 . 7 9 2 , V = 0 . 0 0 4$ , does not survive BH correction). All five other pairwise comparisons survive. The largest efect is SDXL vs. SD 3 Medium $( V = 0 . 1 2 6 )$

![](images/e3a7b6f7ad92db1343558228b82f6e4d05e7e4f0a393243f29d9843504cc72a3.jpg)

![](images/72786a0cb2232602436b16acd73734d4d163964bd271b7ba4311be5eed142f9a.jpg)  
Figure 6: Left: mean stereotype score by model. Right: percentage of occupations showing male majority—peaks at SDXL before recovering in SD 3 Medium.

Table 1: Pairwise image-level chi-square tests (N = 2,000 per model). All p-values BH-adjusted across 10 tests.
<table><tr><td>Model A</td><td>Model B</td><td>Male % (A)</td><td>Male % (B)</td><td> $\chi ^ { 2 }$ </td><td>p (BH-adj.)</td><td>V</td></tr><tr><td>SD 1.5</td><td>SD 2.1</td><td>77.0%</td><td>77.3%</td><td>0.05</td><td>0.792 (n.s.)</td><td>0.004</td></tr><tr><td>SD 1.5</td><td>SDXL</td><td>77.0%</td><td>81.0%</td><td>10.18</td><td>0.002√</td><td>0.050</td></tr><tr><td>SD 1.5</td><td>SD 3M</td><td>77.0%</td><td>70.1%</td><td>24.83</td><td>&lt; 0.001 √</td><td>0.076</td></tr><tr><td>SD 2.1</td><td>SDXL</td><td>77.3%</td><td>81.0%</td><td>8.57</td><td>0.005√</td><td>0.045</td></tr><tr><td>SD 2.1</td><td>SD 3M</td><td>77.3%</td><td>70.1%</td><td>27.51</td><td>&lt; 0.001 √</td><td>0.081</td></tr><tr><td>SDXL</td><td>SD 3M</td><td>81.0%</td><td>70.1%</td><td>66.84</td><td>&lt; 0.001 √</td><td>0.126</td></tr></table>

✓: significant after BH correction. V = Cramér’s V efect size. n.s. = not significant, does not survive BH correction. All values computed from image-level contingency tables $( N = 2 , 0 0 0$ per model).

This non-linear trajectory illustrates what we term a deployment gap: a situation in which a widely adopted model version shows higher bias than both its predecessor and its successor. SDXL is one of the most broadly adopted open-source Stable Difusion variants, yet it is the most genderbiased of the four generations tested. SD 3 Medium partially corrects this but is newer and not yet as broadly integrated.

## 4.6 Prompt Sensitivity

Figure 7 shows prompt sensitivity per occupation and model. Mean sensitivity increases: SD 1.5 (9.6) → SD 2.1 (12.0) → SDXL (12.0) → SD 3 Medium (13.3). SD 3 Medium produces less biased outputs on average but is more sensitive to prompt phrasing. Female-skewed occupations show consistently higher sensitivity than male-skewed ones across all models.

![](images/6127fa028c0c819acd2878b4a5d34e298709cd9de092d8bc81e5d844e804186d.jpg)  
Figure 7: Prompt sensitivity index (std dev of male % across 5 prompt templates, $n = 2 0$ per prompt) by occupation and model. GPT-image-1 excluded (single prompt design). Sensitivity increases across generations.

## 4.7 Racial Composition

Figure 8 shows the racial composition of generated images across all models and occupations. Across all 8,000 open-source images, white presentation is the dominant category at 59.1% overall (SD 1.5: 56.7%, SD 2.1: 57.1%, SDXL: 60.7%, SD 3 Medium: 62.0%). Asian presentation is the second most common across all four models (SD 1.5: 18.0%, SD 2.1: 16.1%, SDXL: 13.3%, SD 3 Medium: 15.2%), followed by Black (9.7% overall), Middle Eastern (8.9%), Latino Hispanic (6.0%), and Indian (0.7%). No model produces substantially diverse racial outputs by any reasonable measure.

White presentation is somewhat lower for male-skewed occupations (56.3% white on average) than female-skewed ones (62.0%), a diference driven primarily by higher Asian classification rates in technical roles such as engineer, programmer, and scientist. The occupations with the lowest white presentation rates are social worker (46.8% white), construction worker (49.5%), mechanic (50.2%), engineer (50.5%), and scientist (54.2%). The least racially varied are judge (68.8% white), librarian (67.2%), receptionist (66.8%), and hair stylist (65.5%). White presentation increases slightly from SD 1.5 to SD 3 Medium, suggesting that newer models may generate less racially diverse images for these occupational prompts despite other improvements. A full statistical analysis of racial bias—including occupational amplification gaps against BLS racial composition data—is beyond the scope of this paper and is left for dedicated follow-up work.

![](images/6aeeaf750992327b71654656805f95f156bdd187a34335870a24ee8d71a39ddd.jpg)  
Figure 8: Racial composition of generated images by occupation and model. White presentation dominates across all conditions.

## 4.8 Preliminary Comparison: GPT-image-1

As a preliminary exploratory comparison, we generated 100 GPT-image-1 images across 5 spotlight occupations (n = 20 per occupation, single prompt). This comparison is intentionally limited: it covers only 5 occupations, uses one prompt template, and involves a large sample size asymmetry $( n = 1 0 0$ for GPT-image-1 vs. n = 2,000 per open-source model). Open-source models generate 84.5% male images for these roles; GPT-image-1 generates 68.5% male images $( \chi ^ { 2 } = 1 5 . 6 1$ , BHadjusted $p < 0 . 0 0 1$ , Cramér’s $V = 0 . 0 8 0 )$ . A Cramér’s V of 0.080 is a small efect by conventional standards. Despite statistical significance, the practical diference is modest.

GPT-image-1 generates 80% male images for programmer (vs. 97–100% for open-source models), 70% for firefighter (vs. 94–100%), and 65% for cleaner (vs. 80–92%). For nurse, GPT-image-1 at 35% male is comparable to SD 1.5 (35%) and better than SD 2.1, SDXL, and SD 3 Medium (40– 42%). Construction worker is 100% male across all models. Figure 9 shows the comparison. Note that three GPT-image-1 cells—cleaner $( 6 5 \% \pm 2 0 . 9 \mathrm { p p } )$ , firefighter $( 7 0 \% \pm 2 0 . 1 \mathrm { p p } )$ , and nurse (35% ±20.9pp)—have wide CIs due to $n = 2 0$ , and should be treated as indicative. A proper follow-up would apply the same multi-prompt, multi-occupation design used for the open-source models.

![](images/012e7f1451e5938d09b1d166a9da96983963597893d9290ea04486b5255f132a.jpg)  
Figure 9: Preliminary GPT-image-1 comparison (purple, bold border) vs open-source models on 5 occupations. Cramér’s $V = 0 . 0 8 0$ (small efect). $n = 2 0$ for GPT-image-1 per occupation; CIs are approximately ±20pp.

Table 2: Male percentage on 5 spotlight occupations. GPT-image-1 results are preliminary $( n = 2 0$ single prompt, Cramér’s $V = 0 . 0 8 0 )$ .
<table><tr><td>Occupation</td><td>SD 1.5</td><td>SD 2.1</td><td>SDXL</td><td>SD 3M</td><td> $\mathbf { G P T - 1 } ^ { * }$ </td><td>Hist.</td></tr><tr><td>Programmer</td><td>100</td><td>98</td><td>97</td><td>100</td><td>80</td><td> $\vec { \mathrm { ~ \bigtriangledown ~ } }$ </td></tr><tr><td>Construction Worker</td><td>99</td><td>100</td><td>97</td><td>100</td><td>100</td><td> $\vec { \mathrm { ~ \bigtriangledown ~ } }$ </td></tr><tr><td>Firefighter</td><td>98</td><td>94</td><td>98</td><td>100</td><td>70†</td><td> $\vec { \mathrm { ~ \bigtriangledown ~ } }$ </td></tr><tr><td>Cleaner</td><td>86</td><td>80</td><td>91</td><td>92</td><td>65†</td><td></td></tr><tr><td>Nurse</td><td>35</td><td>41</td><td>40</td><td>42</td><td>35†</td><td></td></tr></table>

= historically male-skewed.  = historically female-skewed. Open-source: $n = 1 0 0$ per occupation. ♂ ♀<sub>\*GPT-image-1:</sub> <sub>n</sub> <sub>=</sub> <sub>20</sub> <sub>per</sub> <sub>occupation,</sub> <sub>single</sub> <sub>prompt.</sub> $^ \dag \mathrm { \ C I } \approx \pm 2 0 \mathrm { p p } ;$ treat as indicative. Overall: $\chi ^ { 2 } = 1 5 . 6 1$ BH-adjusted $p < 0 . 0 0 1$ , Cramér’s $V = 0 . 0 8 0$

## 5 Discussion

## 5.1 Large and Robust Deviations From Workforce Demographics

The primary findings are robust regardless of where you look. At the model level, CIs are tight (±1.8pp) and the 76.4% overall male rate sits 26pp above the 50% baseline. At the occupation level, the most important results—programmer, construction worker, firefighter, engineer, and scientist— all have lower CI bounds well above 50% despite the ±9.8pp occupation-level CI. The amplification gaps for near-balanced occupations (−30 to $- 4 7 \mathrm { p p }$ for scientist, −26 to −42pp for cleaner) are three to five times the CI width and cannot be attributed to sampling variation.

Of 80 occupation-model cells, 16 fall within the $\pm 9 . 8 \mathrm { p p }$ CI of 50% and are treated as indicative rather than definitive. The most notable are SD 3 Medium’s teacher (45% male) and social worker (45% male), which straddle 50% and indicate improvement relative to earlier models but cannot be called definitively female-majority without a larger sample.

## 5.2 The Deployment Gap

The non-linear trajectory—SD 1.5 and SD 2.1 statistically indistinguishable $( V = 0 . 0 0 4 )$ , SDXL significantly worse than both $( V = 0 . 0 5 0$ and 0.045 respectively), SD 3 Medium significantly better than SDXL $( V = 0 . 1 2 6 )$ —reveals what we term a deployment gap: a situation in which a widely adopted model version shows higher bias than both its predecessor and its successor. Users relying on SDXL get no benefit from SD 3 Medium’s improvements unless they actively switch, and there is no reason to assume they know to.

## 5.3 Cross-Directional Bias Is Systematic

57.6% of images for historically female occupations are classified as male—a reversal that is statistically overwhelming (raw $p = 3 . 4 3 \times 1 0 ^ { - 2 2 }$ , BH-adjusted $p = 1 . 7 1 \times 1 0 ^ { - 2 1 } , n = 4 , 0 0 0 )$ is robust. SD 3 Medium shows meaningful improvement on this dimension $( V = 0 . 1 9 9 ~ \mathrm { v s . \ S D X I }$ on femalecoded occupations) but the improvement is selective—teacher and social worker improve while pilot regresses—suggesting targeted rather than systematic correction.

## 5.4 Prompt Sensitivity Grows With Generation

More capable models are less consistent in their demographic representations across prompt phrasings, not more. Female-coded occupations show consistently higher sensitivity than male-coded ones, suggesting less stable learned representations for these roles. Evaluating a model on one prompt per occupation misses most of the variation it can produce—a practical reason to test multiple phrasings in any bias audit.

## 5.5 GPT-image-1: A Preliminary Observation

The GPT-image-1 comparison yields a small practical efect (Cramér’s $V = 0 . 0 8 0 )$ despite statistical significance (BH-adjusted $p < 0 . 0 0 1 )$ . Three of five GPT-image-1 cells have CIs of approximately $\pm 2 0 \mathrm { p p }$ due to $n = 2 0$ , making those occupation-level comparisons indicative only. A proper followup would apply the same multi-prompt, multi-occupation design used for the open-source models.

## 5.6 Practical Recommendations

Based on these findings:

• Evaluate each model version independently. Bias does not improve monotonically. SDXL is more biased than SD 1.5. Every deployed model version should be evaluated for demographic representation.

• Test multiple prompt phrasings. A single prompt underestimates output range. Evaluate at least three to five phrasings across all relevant occupations.

• Compare against demographic ground truth. Comparing outputs against BLS or equivalent national workforce data reveals deviations that internal 50/50 balance metrics do not capture.

## 6 Limitations

Statistical power and confidence intervals. Each occupation-model cell contains $n = 1 0 0$ images (5 prompts × 20), giving 95% CIs of approximately $\pm 9 . 8 \mathrm { p p }$ at $p = 0 . 5$ . Model-level estimates (n = 2,000) have CIs of approximately ±1.8pp. Of 80 open-source occupation-model cells, 16 fall within ±9.8pp of 50% and are flagged as indicative throughout. Multiple comparisons are corrected using BH across all 10 tests; all nine significant findings survive correction.

Classifier validation. DeepFace was trained on real photographs. Our single-reviewer manual validation on 50 images (96% agreement) is limited in sample size and lacks inter-rater reliability. We argue the primary findings involving gaps of 26pp or more above baseline are robust to plausible classifier error, but occupation-level findings near 50% should be interpreted with caution.

Binary gender classification. DeepFace classifies apparent gender as Man or Woman, missing non-binary, non-conforming, and androgynous presentations. Apparent gender is a proxy for perceived presentation, not self-identified gender.

Resolution diferences. Open-source models were run at 512×512 pixels while GPT-image-1 images were generated at 1024×1024. DeepFace face detection and attribute classification accuracy may difer across resolutions. This provides a further reason to treat the GPT-image-1 comparison as exploratory, as some diferences between GPT-image-1 and open-source outputs could reflect resolution-driven classification variation rather than model behavior alone.

Default inference settings and ecological validity. All models were run with default settings and no negative prompts. Many users of Stable Difusion models employ negative prompts (e.g., “bad anatomy, blurry, low quality”) and modified guidance scales that can substantially afect demographic outputs. Results reflect out-of-the-box behavior rather than the full distribution of real-world use, which limits ecological validity.

BLS data specificity and occupational mapping. Three occupational mappings involve meaningful ambiguity. “Cleaner” maps to a combined estimate across janitors and building cleaners (SOC 37-2011, ∼29% female) and maids and housekeeping cleaners (SOC 37-2012, ∼89% female), which have very diferent gender compositions; the 46% figure used is a weighted average. “Babysitter” has no BLS SOC code and maps to childcare workers (SOC 39-9011, 94% female). “Scientist” maps to the broad category of life, physical, and social science occupations (SOC 19-0000, 48% female), which aggregates subfields ranging from ∼20% female (physics) to ∼75% female (psychology). These ambiguities afect the precision of the BLS reference values for these three occupations but do not change the direction of the findings. BLS figures reflect U.S. 2023 patterns and may not transfer to other cultural contexts.

Occupational scope. Twenty occupations is a reasonable but non-exhaustive starting point. Gender-neutral and emerging roles are not included.

Model access. SD 3.5 and later variants were not evaluated due to hardware constraints.

GPT-image-1 comparison. Five occupations, one prompt, n = 20 per occupation; three cells have CIs ≈ ±20pp. Combined with the resolution diference from open-source models, Cramér’s V = 0.080 should be treated as a preliminary observation only.

## 7 Conclusion

Across 8,000 generated images and four model generations, one pattern holds without exception: models consistently depict professional roles as more male than the actual workforce—and newer models are not reliably fairer. We measured this systematically across 20 occupations, 5 prompt phrasings, and four Stable Difusion generations, with n = 100 per occupation-model cell, confidence intervals throughout, and Benjamini–Hochberg correction across all 10 tests.

Male dominance is pervasive and statistically robust: 76.4% of 8,000 open-source images show male subjects (95% CI [75.1%, 78.7%], $p < 2 . 2 \times 1 0 ^ { - 1 6 }$ , BH-adjusted). 57.6% of images for histori cally female occupations show male subjects (raw $p = 3 . 4 3 \times 1 0 ^ { - 2 2 }$ , BH-adjusted $p = 1 . 7 1 \times 1 0 ^ { - 2 1 } )$ .

Models underrepresent women by 20–46pp on average relative to BLS workforce demographics, with deviations of 26–47pp for near-balanced occupations such as scientist and cleaner.

Bias does not improve steadily across generations. SDXL—one of the most broadly adopted open-source Stable Difusion variants—shows higher gender skew than both SD 1.5 and SD 3 Medium (V = 0.050 and 0.126 respectively). We introduce the term deployment gap to describe this pattern. A preliminary comparison with GPT-image-1 suggests lower bias on four of five spotlight occupations (V = 0.080, small efect), warranting follow-up study. Code and generation configurations are available at https://github.com/SheshNGupta/GenderSterotype.

## Ethics Statement

This study does not involve human subjects. All images analyzed were generated by AI models in response to occupational text prompts; no images of real individuals were collected, used, or stored. DeepFace classifications assign apparent gender labels (Man or Woman) to synthetic faces. We acknowledge that this binary classification does not capture the full spectrum of gender identity and presentation, and we discuss this limitation explicitly in Section 6.

The images generated in this study depict professional roles and were used solely to measure demographic patterns in model outputs. No images were generated with the intent to produce harmful, demeaning, or discriminatory content. Raw generated images are not released publicly to avoid potential misuse; only aggregate statistics and figures derived from them are shared.

The findings of this paper document gender bias in widely deployed AI systems. We believe transparency about these patterns is necessary for the research community and for developers to address them. We are aware that detailed documentation of model biases could in principle be used to exploit or exacerbate those biases, but we judge this risk to be low given that the patterns we identify are already observable to any user of these systems. The benefit of public documentation substantially outweighs this risk.

All generation configurations, classification scripts, and aggregate results are made publicly available to support reproducibility and future audits.

## Acknowledgments

The authors thank the developers of the Hugging Face Difusers library and the DeepFace framework, which made this large-scale evaluation feasible on consumer hardware.

## References

[1] Federico Bianchi, Pratyusha Kalluri, Esin Durmus, Faisal Ladhak, Myra Cheng, Debora Nozza, Tatsunori Hashimoto, Dan Jurafsky, James Zou, and Aylin Caliskan. Easily accessible text-toimage generation amplifies demographic stereotypes at large scale. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency (FAccT), pages 1493–1504, 2023.

[2] Jaemin Cho, Abhay Zala, and Mohit Bansal. DALL-Eval: Probing the reasoning skills and social biases of text-to-image generation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3043–3054, 2023.

[3] Abhishek Mandal, Susan Leavy, and Suzanne Little. Multimodal composite association score: Measuring gender bias in generative multimodal models. arXiv preprint arXiv:2304.13855, 2023.

[4] Alexandra Sasha Luccioni, Christopher Akiki, Margaret Mitchell, and Yacine Jernite. Stable bias: Analyzing societal representations in difusion models. Advances in Neural Information Processing Systems, 36, 2023.

[5] Felix Friedrich, Patrick Schramowski, Manuel Brack, Lukas Struppek, Dominik Hintersdorf, Sasha Luccioni, and Kristian Kersting. Fair difusion: Instructing text-to-image generation models on fairness. arXiv preprint arXiv:2302.10893, 2023.

[6] Tolga Bolukbasi, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam T. Kalai. Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. In Advances in Neural Information Processing Systems, volume 29, 2016.

[7] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Men also like shopping: Reducing gender bias amplification using corpus-level constraints. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2979–2989, 2017.

[8] U.S. Bureau of Labor Statistics. Labor force statistics from the current population survey, table 11: Employed persons by detailed occupation, sex, race, and hispanic or latino ethnicity. https://www.bls.gov/cps/cpsaat11.htm, 2023. Accessed: March 2025.

[9] Antonio Torralba and Alexei A. Efros. Unbiased look at dataset bias. In Proceedings of the 2011 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1521– 1528. IEEE, 2011.

[10] Joy Buolamwini and Timnit Gebru. Gender shades: Intersectional accuracy disparities in commercial gender classification. In Proceedings of the 1st Conference on Fairness, Accountability and Transparency, volume 81, pages 77–91. PMLR, 2018.

[11] Jonas Oppenlaender. The creativity of text-to-image generation. In Proceedings of the 25th International Academic Mindtrek Conference, pages 192–202, 2022.

[12] Laura Weidinger, John Mellor, Maribeth Rauh, Conor Grifin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, Zac Kenton, Sasha Brown, Will Hawkins, Tom Stepleton, Courtney Biles, Abeba Birhane, Julia Haas, Laura Rimell, Lisa Anne Hendricks, William Isaac, Sean Legassick, Geofrey Irving, and Iason Gabriel. Ethical and social risks of harm from language models. arXiv preprint arXiv:2112.04359, 2021.

[13] Sefik Ilkin Serengil and Alper Özpinar. HyperExtended LightFace: A facial attribute analysis framework. In 2021 International Conference on Engineering and Emerging Technologies (ICEET), pages 1–4. IEEE, 2021.