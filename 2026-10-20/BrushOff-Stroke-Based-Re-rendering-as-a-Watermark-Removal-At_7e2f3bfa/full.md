# BrushOff: Stroke-Based Re-rendering as a Watermark Removal Attack on Paintings

Hao Chen

haoc.research@gmail.com

Junnan Xu

junnanxu25@gmail.com

Abstract—Invisible watermarks are used to mark who owns a painting. We show that a simple attack removes the most robust one we test. The attack redraws the watermarked painting with brush strokes, using a pretrained stroke-based renderer. We call this attack BrushOff. It needs no training, no knowledge of the watermark, and no generative model. We test four watermarks on 500 paintings from five styles. VINE, a recent watermark built to survive image editing, survives all 14 conventional distortions we test, with at least 97.8% detection. Repainting lowers its detection to 1.8%, while the output keeps a no-reference quality score close to that of the clean paintings. Diffusion regeneration can also remove VINE, but at the same removal level it moves the image twice as far from the painting in DreamSim distance, and it changes what the painting shows. Because the renderer draws a countable number of strokes, we can control the attack strength step by step. Finally, we test what kind of signal survives. A simple probe that stores bits in the relative brightness of neighboring regions survives repainting, while a probe that uses absolute brightness does not. The same probe fails under diffusion regeneration, while VINE fails under repainting. Robustness to these two attacks is therefore a separate property, and benchmarks should test both.

## I. INTRODUCTION

Invisible watermarks are a common tool to record who owns an image. Artists, museums, and platforms can embed an owner ID into a painting before sharing it online. For this to work, the mark must survive the ways people copy and modify images.

Recent watermarks are robust to many changes. VINE [1] is designed to survive not only compression and blur, but also generative editing with diffusion models. In our tests on 500 paintings, it survives all 14 conventional distortions, including JPEG quality 5 and strong blur.

A painting can be copied in another way: it can be painted again. A human copyist does not reproduce pixels. They reproduce the composition, the colors, and the main shapes with brush strokes. Stroke-based renderers [2], [3] do the same thing automatically. They describe an image with a set of strokes, each with a position, size, angle, and color. We ask a simple question: does a watermark survive when a painting is redrawn with strokes?

For VINE, the most robust watermark we test, the answer is no. We redraw each watermarked painting with Paint Transformer [3], a pretrained renderer, without any training. We call this attack BrushOff. VINE drops from 100% to 1.8% detection. The repainted images keep a MUSIQ quality score of 58.2, close to 59.2 for the clean paintings, and keep the composition and subject (Fig. 1).

Repainting differs from the standard generative attack. Diffusion regeneration [4] also removes watermarks, but it relies on a model that has learned what images look like. It can replace parts of the image with new content. Repainting has no such prior. Each stroke copies its color from the input. At the same removal level, diffusion regeneration moves the image twice as far from the painting in DreamSim distance (0.559 vs. 0.265), and it visibly changes the content.

Repainting also gives a scale that diffusion attacks do not have. The number of strokes is an explicit and countable bottleneck. We can increase it step by step and see when a watermark starts to survive.

Finally, we ask what kind of signal survives repainting. We test two simple low-frequency probes. A probe that stores bits in the relative brightness of neighboring regions survives repainting with 83% detection. A probe that uses absolute brightness fails. The surviving probe, however, fails under diffusion regeneration, while VINE fails under repainting. Robustness to the two attacks is therefore a separate property.

Our contributions are:

• We show that stroke-based repainting removes VINE, a watermark that survives all conventional distortions we test, while keeping image quality close to the clean paintings.

• We compare repainting with diffusion regeneration on 500 paintings from five styles. At the same removal level, repainting changes the painting about half as much in DreamSim distance and keeps its content.

• We use the stroke count as a measurable bottleneck, and show with two probes that relative brightness survives repainting while absolute brightness does not. Robustness to repainting and to regeneration are separate properties.

## II. RELATED WORK

a) Image watermarking: Classical methods embed bits in a transform domain, such as DWT and DCT coefficients [5], [6]. Learned methods train an encoder and a decoder together, often with distortions in the training loop [7]–[9]. Trust-Mark [10] adds error correction and works at any resolution. VINE [1] uses a generative prior and is trained to survive image editing, including diffusion-based editing. Other methods embed the watermark during generation, such as Stable Signature [11], Tree-Ring [12], and Gaussian Shading [13]. We focus on post-hoc watermarks, because a painting already exists before it is marked.

![](images/d7cd842d38969e7ec69dd4e7136889ef979f710c4df8f5610675d10a7ec63d1d.jpg)  
Fig. 1. Examples from three styles. JPEG and blur keep the VINE watermark but leave strong artifacts. Diffusion regeneration keeps the watermark at strength 0.5, and removes it at strength 0.8 only by changing the content. Repainting (RP-2048/L6) keeps the composition and removes the watermark. Below each image: whether VINE is detected, its bit accuracy, and MUSIQ.

b) Watermark removal: The standard removal attacks are conventional distortions and regeneration. Zhao et al. [4] show that pixel-level watermarks can be removed by encoding an image into a generative model and decoding it back. Benchmarks such as WAVES [14] and W-Bench [1] test watermarks against distortions, regeneration, and image editing. Recent work uses diffusion-based image editing to remove or launder watermarks [15]. All of these attacks rely on a learned generative prior. Our attack does not. This lets us separate removal by re-rendering from removal by a learned prior.

c) Protection of artworks: Several works watermark artworks or protect artistic style [16]–[18]. Adversarial protections such as Glaze [19] can be removed by simple steps such as noisy upscaling [20]. We study a different target: invisible marks that record ownership.

d) Stroke-based rendering: Stroke-based renderers describe an image with a sequence of brush strokes. Learning to Paint [2] uses reinforcement learning. Paint Transformer [3] predicts strokes in one pass with a transformer and paints from coarse to fine. Later work predicts where to paint at each step [21]. StegoNPR [22] hides data in the arrangement of non-photorealistic strokes. That work embeds data through strokes. We use strokes to remove data. To our knowledge, stroke-based rendering has not been studied as a watermark removal attack.

## III. THREAT MODEL AND REPAINTING ATTACK

## A. Threat model

We consider an attacker who wants to remove the provenance mark from a painting. The attacker has one watermarked image, does not know the method, key, or payload, cannot query the detector, and does not train any model. A typical case: a painting is shared online with an invisible owner mark, and someone downloads it, re-renders it, and posts the result as their own.

The attack succeeds if two conditions hold. First, the detector no longer finds the watermark at a fixed false positive rate. Second, the output still looks like the same painting, like a copy made by hand. We therefore always report detection together with fidelity (Sec. IV-E).

## B. Repainting with a stroke-based renderer

Our attack redraws the watermarked image with brush strokes. We use Paint Transformer [3], a pretrained strokebased renderer. We use the released model as is, without any training or tuning.

Paint Transformer paints from coarse to fine. At layer k, it resizes the target image to 32 · 2<sup>k</sup> pixels per side and splits it into 32 × 32 patches. For each patch, the network looks at the target and the current canvas, and predicts 8 strokes. Each stroke has five shape parameters (center x, y, width, height, angle) and one RGB color. The color is sampled from the target image at the stroke center. A binary decision per stroke controls whether it is drawn. For a canvas of size S, there are

K + 1 layers with $K = \lceil \log _ { 2 } ( S / 3 2 ) \rceil$ . After the last layer, the official code adds one border pass. This pass shifts the patch grid by half a patch to cover the seams between patches.

We save the canvas after every layer. This gives a sequence of outputs from one painting run, which we call stroke levels. Level Lℓ for $\ell \leq K + 1$ is the canvas after layer $\ell - 1$ . The last level, $\mathrm { L } ( K + 2 )$ , adds the border pass and equals the official output. One run therefore gives all levels at no extra cost.

a) Border pass: The border pass draws all of its strokes. In the official code, the stroke decision in this pass is not binarized, so every predicted stroke is drawn. We keep this behavior so that our last level matches the official output. As a result, the last level has many more strokes than the level before it. For this reason we use the number of strokes actually drawn, not the level index, when we study how detection depends on the attack strength (Sec. V-C).

b) Canvas size: We control stroke size by changing the canvas size. For canvas size $S \in \{ 1 0 2 4 , 2 0 4 8 \}$ , we upsample the $5 1 2 \times 5 1 2$ input with bicubic interpolation, paint it, and downsample the result back to $5 1 2 \times 5 1 2$ with antialiasing. A larger canvas adds more layers, so strokes become smaller relative to the image. The 512, 1024, and 2048 canvases give 6, 7, and 8 levels. We denote a setting by canvas size and level, for example RP-2048/L6.

## C. Why repainting removes watermarks

Repainting replaces every pixel with strokes. The output is a function of the stroke parameters only. Any signal that the strokes do not capture is lost.

a) A countable bottleneck: The strokes form an explicit and countable bottleneck. Each drawn stroke carries 8 numbers. At RP-2048/L6, the renderer draws about 6,000 strokes per image, or about 48,000 numbers. A 512×512 RGB image has 786,432 values. So the painting is described with about 6% of the original values. We can increase this budget step by step by moving to finer levels. This lets us plot watermark survival against the number of strokes (Sec. V-C). Diffusionbased attacks do not offer such a scale, because their latent space is not easy to interpret.

b) No generative prior: Paint Transformer has no generative prior over natural images. It does not invent content. Each stroke copies its color from the input at one location. This separates two possible reasons for watermark removal. Diffusion regeneration may remove a watermark because the model replaces image content with what it has learned. Repainting can only remove a watermark by re-rendering the existing content. If a watermark survives diffusion regeneration but not repainting, its weakness is the rendering itself, not a learned prior.

c) Cost: The attack is cheap: about 27 s per image at canvas size 2048 on one consumer GPU, with no training.

## IV. EXPERIMENTAL SETUP

## A. Data

We use 500 paintings from WikiArt [23], 100 from each of five styles: Baroque, Impressionism, Pointillism, Ukiyo-e, and Color Field Painting. We chose these styles because they differ in texture. Baroque paintings have smooth gradients and fine detail. Impressionism and Pointillism are made of visible brush marks and dots. Ukiyo-e prints have flat colors and sharp outlines. Color Field paintings have large flat regions with little texture. We sample images at random with a fixed seed, keep those with a short side of at least 512 pixels, center-crop them to a square, and resize them to 512 × 512. The WikiArt data is used for non-commercial research only.

TABLE I  
EMBEDDING QUALITY ON 500 PAINTINGS. PSNR: MEAN / MIN. DET.: DETECTED RIGHT AFTER EMBEDDING. FP: DETECTIONS ON THE 500 CLEAN PAINTINGS.
<table><tr><td>Method</td><td>Bits</td><td>PSNR (dB) Det. FP</td></tr><tr><td>VINE TrustMark</td><td>100 61+39</td><td>36.9 / 27.8 500 0 42.4 / 35.8 500 0</td></tr><tr><td>RivaGAN</td><td>32</td><td>40.5 / 39.8 485 0</td></tr><tr><td>DwtDctSvd</td><td>32</td><td>37.8 / 24.7 500 0</td></tr><tr><td>DwtDct</td><td>32</td><td>37.8 / 24.8 349 0</td></tr></table>

## B. Watermarks

We evaluate four post-hoc watermarks. VINE [1] is a recent learned watermark designed to survive image editing, including generative editing. We use the released VINE-R model with a 100-bit payload. TrustMark [10] is a learned watermark with error correction. We use the Q model with a 61-bit payload and the default BCH code, which gives 100 bits in total. RivaGAN [9] is a learned watermark with a 32-bit payload. DwtDctSvd is a classical frequency-domain method with a 32-bit payload. For RivaGAN and DwtDctSvd we use the invisible-watermark library [6]. Every image carries the same fixed payload.

We also tested DwtDct from the same library. It fails on 151 of the 500 paintings before any attack, and on 44 of the 100 Color Field paintings. We report it in Table I but leave it out of the attack evaluation.

Table I reports embedding quality. All four methods reach a mean PSNR above 36 dB. VINE has the lowest PSNR (36.9 dB) and TrustMark the highest (42.4 dB).

## C. Detection

We decide detection at a fixed false positive rate of $1 0 ^ { - 3 }$ For a payload of n bits, we assume that the decoded bits of an unwatermarked image are independent and uniform. We then pick the smallest bit accuracy τ such that the chance of reaching τ by luck is at most $1 0 ^ { - \dot { 3 } }$ . This gives $\tau = 2 6 / 3 2$ for RivaGAN and DwtDctSvd, and $\tau = 6 6 / 1 0 0$ for VINE. For TrustMark we use its own decision: the watermark is detected if error correction succeeds and recovers the payload exactly.

We check the false positive rate empirically. We run every detector on the 500 clean paintings. None of them fires on any clean painting.

We also check each watermark right after embedding. An image that is not detected before any attack cannot tell us anything about the attack. We drop such images for that watermark and report them separately. This affects only RivaGAN, which fails on 15 of 500 paintings before any attack. Fourteen of them are Color Field paintings, which have large flat regions and little texture to hide a signal in.

## D. Attacks

a) Repainting: We run the repainting attack of Sec. III-B with canvas sizes 512, 1024, and 2048. This gives 21 settings, from RP-512/L1 to RP-2048/L8. For VINE, we run all settings on all 500 paintings. Canvas size 2048 is slow (Sec. III-C). For the other three watermarks, we therefore run it on a subset of 100 paintings, the first 20 of each style.

b) Conventional distortions: We apply 14 standard distortions: JPEG compression with quality 90, 70, 50, 30, 10, and 5; Gaussian blur with radius 1, 2, 4, 6, and 8 pixels; and down-scaling by a factor of 0.5, 0.25, and 0.125 followed by up-scaling back to 512 × 512 with bicubic interpolation. We save all outputs as PNG, so no further compression is added.

c) Diffusion regeneration: We compare against diffusion regeneration [4], the standard generative removal attack. The image is encoded with the Stable Diffusion VAE, noised to a given strength, and denoised back. We use Stable Diffusion 2.1-base [24] in image-to-image mode with an empty prompt, no classifier-free guidance, 50 denoising steps, and a fixed seed. We test strengths 0, 0.1, 0.2, 0.3, and 0.5 on all four watermarks. Strength 0 is a VAE round trip without noise. For VINE, we add strengths 0.6, 0.7, and 0.8, because VINE still survives at 0.5. We denote a setting by its strength, for example Regen-0.5.

## E. Fidelity metrics

We measure how close the attacked image stays to the clean painting. All metrics compare against the clean painting, not the watermarked one. We use three metrics, because no single one captures what we need.

DISTS [25] measures structure and texture similarity. DreamSim [26] measures perceptual similarity at a higher level, closer to whether two images show the same thing. For both, lower means closer to the clean painting. MUSIQ [27] is a no-reference quality score. It measures whether an image looks free of artifacts such as blur, noise, and block patterns. Higher is better. The clean paintings have a mean MUSIQ of 59.2. We also report PSNR for reference.

MUSIQ does not check whether the content matches the painting: a coarse repainting with a few hundred strokes can score high. We therefore always read MUSIQ together with DreamSim.

In a pilot study, fidelity barely depended on which watermark the image carried. We therefore compute fidelity on the VINE images and use it for all methods.

a) Statistics: We report 95% confidence intervals for detection rates. We compute them with a bootstrap over images, using 2,000 resamples and the percentile method.

## V. RESULTS

Table II gives the main numbers.

## A. Repainting removes VINE

VINE survives every conventional distortion we test. Its detection rate stays at or above 97.8% under all 14 distortions, including JPEG quality 5 and blur radius 8 (Table II). These two distortions are already strong. JPEG q5 drops MUSIQ from 59.2 to 32.3, and blur radius 8 drops it to 15.1.

Repainting removes VINE. At RP-2048/L6, the detection rate drops to 1.8% (95% CI: 0.8–3.2%). At RP-1024/L6, it drops to 3.0%.

The repainted images still look like normal paintings. RP-2048/L6 keeps a MUSIQ of 58.2, close to the 59.2 of the clean paintings. Its DreamSim to the clean painting is 0.265. This is close to JPEG q5 (0.230) and lower than blur radius 8 (0.355). In other words, at a similar perceptual distance, JPEG and blur leave visible artifacts and keep the watermark, while repainting gives a clean-looking image and removes the watermark. Fig. 2 shows this for all settings. Fig. 1 shows examples.

Coarser levels remove all watermarks but change the painting too much. At levels L1 to L5, no watermark is detected in more than 6% of images at any canvas size. But DreamSim is above 0.35 at these levels, and the images lose much of their detail. We therefore focus on L6 and finer levels.

## B. Comparison with diffusion regeneration

Diffusion regeneration can also remove VINE, but it needs to change the image much more. At strength 0.5, VINE is still detected in 47.4% of images (95% CI: 42.8–51.8%). Detection only falls to 1.8%, the level reached by RP-2048/L6, at strength 0.8.

At the same removal level, repainting stays closer to the painting. Regen-0.8 has a DreamSim of 0.559, while RP-2048/L6 has 0.265. Its PSNR is 14.4 dB, while RP-2048/L6 has 19.9 dB. DISTS is similar for the two (0.325 and 0.349).

At a similar DreamSim, repainting removes far more. Regen-0.5 and RP-2048/L6 have almost the same DreamSim (0.256 and 0.265) and MUSIQ (59.1 and 58.2). VINE is detected in 47.4% of images after Regen-0.5, but in only 1.8% after RP-2048/L6.

Diffusion regeneration also changes what the painting shows. In Fig. 1, Regen-0.5 turns the dragon in the Ukiyo e print and the bush in the Pointillist painting into different shapes, while VINE is still detected. Repainting keeps the composition and the subject, while VINE is gone.

MUSIQ does not penalize this change. The MUSIQ of diffusion regeneration rises with strength, from 58.2 at strength 0 to 63.2 at 0.8. The model adds sharp details that were not in the painting, and MUSIQ rewards them. This is why we read MUSIQ together with DreamSim.

We see a simple reason for this result. VINE is trained to survive generative editing [1], so diffusion regeneration is close to what it was built for. Repainting has no generative prior and renders the image in a different way. VINE was not trained for this kind of change.

TABLE II  
FIDELITY AND DETECTION UNDER EACH ATTACK. FIDELITY IS MEASURED AGAINST THE CLEAN PAINTING. THE CLEAN PAINTINGS HAVE A MEAN MUSIQ OF 59.2. DETECTION RATES USE A FIXED FALSE POSITIVE RATE OF 10<sup>−3</sup>. <sup>†</sup>: 100-IMAGE SUBSET. –: NOT RUN.
<table><tr><td colspan="2"></td><td colspan="4">Fidelity</td><td colspan="4">Detection rate</td></tr><tr><td>Attack</td><td>Strokes</td><td>DISTS↓</td><td>DreamSim↓</td><td>MUSIQ↑</td><td>PSNR↑</td><td>VINE</td><td>TrustMark</td><td>RivaGAN</td><td>DwtDctSvd</td></tr><tr><td>JPEG q10</td><td></td><td>0.267</td><td>0.111</td><td>42.7</td><td>25.5</td><td>1.000</td><td>0.068</td><td>0.082</td><td>0.000</td></tr><tr><td>JPEG q5</td><td></td><td>0.339</td><td>0.230</td><td>32.3</td><td>23.1</td><td>0.992</td><td>0.000</td><td>0.002</td><td>0.000</td></tr><tr><td>Blur r4</td><td></td><td>0.353</td><td>0.168</td><td>16.7</td><td>23.4</td><td>1.000</td><td>0.954</td><td>0.705</td><td>0.562</td></tr><tr><td>Blur r8</td><td></td><td>0.470</td><td>0.355</td><td>15.1</td><td>21.5</td><td>1.000</td><td>0.050</td><td>0.000</td><td>0.058</td></tr><tr><td>Resize 0.125</td><td></td><td>0.337</td><td>0.163</td><td>17.9</td><td>23.7</td><td>0.978</td><td>0.826</td><td>0.695</td><td>0.890</td></tr><tr><td>Regen-0.1</td><td></td><td>0.124</td><td>0.061</td><td>57.7</td><td>23.0</td><td>0.998</td><td>0.000</td><td>0.002</td><td>0.092</td></tr><tr><td>Regen-0.3</td><td></td><td>0.168</td><td>0.122</td><td>57.9</td><td>20.6</td><td>0.950</td><td>0.000</td><td>0.000</td><td>0.012</td></tr><tr><td>Regen-0.5</td><td></td><td>0.221</td><td>0.256</td><td>59.1</td><td>18.3</td><td>0.474</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Regen-0.8</td><td>一</td><td>0.325</td><td>0.559</td><td>63.2</td><td>14.4</td><td>0.018</td><td></td><td></td><td></td></tr><tr><td>RP-512/L6</td><td>4,249</td><td>0.383</td><td>0.280</td><td>59.5</td><td>20.1</td><td>0.532</td><td>0.000</td><td>0.126</td><td>0.262</td></tr><tr><td>RP-1024/L6</td><td>6,045</td><td>0.349</td><td>0.267</td><td>57.9</td><td>19.9</td><td>0.030</td><td>0.000</td><td>0.196</td><td>0.158</td></tr><tr><td>RP-2048/L6</td><td>6,042</td><td>0.349</td><td>0.265</td><td>58.2</td><td>19.9</td><td>0.018</td><td>0.000†</td><td>0.226†</td><td>0.180†</td></tr><tr><td>RP-2048/L7</td><td>16,392</td><td>0.309</td><td>0.182</td><td>58.3</td><td>21.6</td><td>0.618</td><td>0.000†</td><td>0.656†</td><td>0.290†</td></tr><tr><td>RP-2048/L8</td><td>50,192</td><td>0.258</td><td>0.116</td><td>57.1</td><td>23.6</td><td>1.000</td><td>0.030†</td><td>0.849†</td><td>0.470†</td></tr></table>

JPEG Gaussian blur Down/up-scale Difusion regen. Repaint, canvas 512 Repaint, canvas 1024 Repaint, canvas 2048  
![](images/d61ca83bda742e05e165b37d6c236b90e23cef27f1ac4e52f95804a78e6dc803.jpg)

![](images/c6546bf6539c050b6f8da039f2d2c84a55d5ea60c2ed79534e52b9e6df65aef8.jpg)  
Fig. 2. (a) VINE detection rate and (b) MUSIQ against DreamSim distance to the clean painting. Each curve is one attack family at increasing strength. Error bars: 95% bootstrap confidence intervals. Dotted line in (b): mean MUSIQ of the clean paintings.

## C. Detection and the number of strokes

For the three weaker watermarks, detection follows the number of strokes. Fig. 3 plots detection against the number of strokes drawn. For RivaGAN and DwtDctSvd, the curves from the three canvas sizes almost overlap. Detection rises only after a few thousand strokes. TrustMark stays near zero at every level.

VINE shows a sharp transition. It is detected in at most 0.2% of images at about 1,900 strokes or fewer (L5 at every canvas size). It is detected in 62% of images at about 16,000 strokes (RP-2048/L7), and in all images at about 50,000 strokes (RP-2048/L8). So more strokes is not always better for the attacker: the best setting is the finest level that still removes the watermark, here RP-2048/L6.

For VINE, the stroke count is not the whole story. RP-512/L6 draws about 4,200 strokes, fewer than RP-2048/L6 with about 6,000. Yet VINE is detected in 53% of images after RP-512/L6, and in only 1.8% after RP-2048/L6. Likewise, RP-1024/L7 and RP-2048/L7 draw a similar number of strokes (14,800 and 16,400), but give 96% and 62%. In both pairs, the setting with the larger canvas, and therefore smaller strokes, removes more. Both RP-512/L6 and RP-1024/L7 are last levels, where the border pass draws all of its strokes (Sec. III-B). So the size and placement of strokes also matter, not only their number.

## D. Differences between styles

At RP-2048/L6, repainting removes VINE in every style. The detection rate is at most 7% in each of the five styles.

At the next level, styles differ. At RP-2048/L7, VINE is detected in 40% of Color Field paintings and in 87% of Pointillist paintings. The other three styles fall between 55% and 65%.

![](images/f77b1dbf1569515f0ed672f6ff3248be811f6b753e4c9215857973d77c41b756.jpg)  
Fig. 3. Detection rate against the number of strokes drawn, for each watermark and canvas size. Hollow markers: last level, where the border pass draws all of its strokes. Shaded bands: 95% bootstrap confidence intervals. TrustMark, RivaGAN, and DwtDctSvd at canvas size 2048 use a 100-image subset.

Diffusion regeneration shows a different pattern. At Regen-0.5, VINE survives best on Color Field (70%) and Baroque (67%), and worst on $\mathrm { U k i y o - e } \left( 3 0 \% \right)$ and Impressionism (29%).

## E. The weaker watermarks

Repainting changes the robustness ranking of the weaker watermarks. TrustMark survives blur radius 4 (95%) and down-scaling to 0.125 (83%), but every repainting setting removes it. Even at RP-2048/L8, its detection rate is 3%. RivaGAN behaves the other way. It fails under blur radius 8 (0%), but survives the finest repainting (85% at RP-2048/L8). Robustness to conventional distortions therefore does not predict robustness to repainting.

For these three watermarks, repainting is not the cheapest attack. JPEG q10 removes all three at a DreamSim of 0.111. Regen-0.1 lowers all three to at most 9% at a DreamSim of 0.061. Even a VAE round trip without noise (Regen-0) lowers RivaGAN to 32%. For RivaGAN and DwtDctSvd, repainting reaches similar removal only at the coarse levels, where DreamSim is above 0.35. TrustMark is the exception: every repainting setting removes it, down to a DreamSim of 0.116. The advantage of repainting is specific to VINE, the watermark built to resist generative editing.

## VI. WHAT SURVIVES REPAINTING?

Repainting keeps the composition of a painting: where things are, and which areas are lighter or darker. If a watermark is carried by this coarse structure, it may survive. We test this idea with two simple low-frequency probes. The probes are not new watermarking methods. They are textbook schemes that we use only to test the idea.

## A. Two probes

Both probes split the image into a $G \times G$ grid of blocks and change only the mean brightness of each block. The change is added through a smooth Hann window inside each block, so no block edges appear. The same change is added to R, G, and B, so only brightness changes. Both probes carry the same 32-bit payload, repeated over the blocks with a keyed permutation. We use the same detection threshold as for the other 32-bit methods $( 2 6 / 3 2 )$

a) QIM: absolute brightness: The first probe stores one bit in the absolute mean brightness of each block. It uses quantization index modulation [28] with step ∆. We test $G \in \{ 8 , 1 6 \}$ and $\Delta \in \{ 8 , 1 6 \}$

b) Pair: relative brightness: The second probe stores one bit in the brightness difference between two neighboring blocks. For bit 1, the left block must be brighter than the right block by at least T; for bit 0, darker by at least T. Pairs that already satisfy this are left unchanged. We cap the change of any block at $2 T .$ , so very large natural differences are not forced. We use $G = 1 6$ , which gives 128 pairs and 4 repetitions per bit, and test $T \in \{ 4 , 8 \}$ . At decoding, each pair gives one vote, $\mathrm { c l i p } ( d / T , - 1 , 1 )$ , where d is the measured difference.

We changed this decoder once. Our first version summed the raw differences, and large natural differences hid the embedded votes: only 36 of 100 paintings were detected right after embedding with $T \ = \ 8 .$ The clipped vote fixes this without changing the false positive rate, and we use it throughout.

## B. Setup

We embed both probes in 100 paintings, the first 20 of each style. We attack them with the same repainting, conventional, and regeneration settings as before. We count an image as missed if the probe fails right after embedding. So the detection rates below are over all 100 images.

Both probes have no false positives in practice. On the 500 clean paintings, five of the six configurations never fire. QIM with $G = 1 6 , \Delta = 8$ fires once, which is in line with the designed rate of $1 0 ^ { - 3 }$

![](images/fed940c02679cf5b039e4f10e75f685742c3a661bf92aa72b8584a293bcf132e.jpg)  
Fig. 4. Detection rate against embedding PSNR after (a) RP-2048/L6 and (b) Regen-0.5. Probes use 100 paintings, and embedding failures count as misses. Error bars: 95% bootstrap CIs.

The probes trade visibility for robustness. QIM with $\Delta = 8$ and pair with T = 4 reach a PSNR of about 37.5 dB, close to VINE (36.9 dB). QIM with $\Delta = 1 6$ and pair with T = 8 reach only 31.5 dB and 30.6 dB. Pair with T = 4 cannot be written into every painting: it is detected right after embedding in 60 of 100 paintings. With $T = 8$ this rises to 96 of 100.

## C. Results

Relative brightness survives repainting. Pair with $T = 8$ is detected in 83% of images after RP-2048/L6 (95% CI: 76– 89%), where VINE is detected in 1.8%. It is still detected in 77% of images at RP-512/L5, which draws only about 1,900 strokes. It also survives all conventional distortions, with at least 91% detection.

Absolute brightness does not survive repainting. All four QIM configurations are detected in at most 5% of images at L6 and finer levels. They are robust to conventional distortions, with 66–98% detection under blur radius 8 and 71–97% under JPEG q5. Their bit accuracy after repainting is often below chance. For example, QIM with G = 8, ∆ = 16 has a bit accuracy of 0.31 after RP-512/L6. A bit accuracy far below 0.5 means that many bits flip in the same direction. So repainting does not erase the block brightness. It shifts it, and absolute codes cannot tolerate this shift. The pair probe compares two neighboring blocks, so a shared shift cancels out.

The probe also holds at a matched PSNR. Pair with T = 4 has about the same PSNR as VINE. It is detected in 29% of images after RP-2048/L6 (95% CI: 22–36%), compared with 1.8% for VINE.

Robustness to repainting and to diffusion regeneration are separate properties. The pair probe that survives repainting fails under diffusion regeneration: 6% after Regen-0.5. VINE shows the opposite pattern: 1.8% after RP-2048/L6, but 47% after Regen-0.5. Fig. 4 shows this for all methods. No method we test survives both.

## VII. DISCUSSION AND LIMITATIONS

a) Robustness has more than one axis: Our results show that a watermark can be robust to one kind of re-rendering and fragile to another. VINE resists diffusion regeneration but not repainting. Our pair probe resists repainting but not diffusion regeneration. Robustness to conventional distortions does not predict either one: TrustMark survives blur but not repainting, and RivaGAN the other way round. Benchmarks that test only conventional distortions and generative regeneration [1], [14] therefore miss a real attack. We suggest adding nongenerative re-rendering, such as stroke-based repainting, to these benchmarks. A claim that a watermark is hard to remove should also be tested against repainting.

b) Where a robust signal could live: Our probe suggests that a signal carried by relative brightness between regions survives repainting. This is the level that strokes reproduce. The probe pays for this with visibility: with $T \ = \ 8$ it is about 6 dB lower in PSNR than VINE. It also fails under diffusion regeneration. Combining such a signal with a learned watermark may cover both attacks. We leave this to future work.

c) Limitations: Our study has several limits.

• We test one stroke-based renderer, Paint Transformer. Other renderers, such as Learning to Paint [2], may behave differently.

• Repainting changes the texture of a painting. The output has visible brush strokes, so it is a convincing copy, not an exact one. For uses that need an exact copy, this attack is less useful.

• The repainted images have small dark spots along the right and bottom borders. They come from the black initial canvas of the official code, where no stroke covers it. They do not affect our conclusions, but a stronger attacker could easily remove them.

• Canvas size 2048 for the three weaker watermarks uses 100 images, and fidelity is computed on the VINE images only.

• We changed the pair decoder once after seeing the first results (Sec. VI-A).

• We do not test watermarks embedded during image generation, such as Tree-Ring [12] or Gaussian Shading [13].

• We only study paintings. On photographs, the repainted output would look like a painting, so our threat model does not apply.

d) Ethics: This paper describes a way to remove watermarks from paintings. The attack uses only a public pretrained model, and anyone can run it already. We think it is better for watermark designers to know about it. We follow the licenses of the data and models: WikiArt and the VINE weights are for non-commercial research only.

## VIII. CONCLUSION

Repainting a painting with brush strokes removes VINE, which survives all conventional distortions we tested. The attack needs no training and no generative model, and the output still looks like the painting. At the same removal level, diffusion regeneration changes the painting about twice as much. No watermark we tested resists both attacks, so we hope future benchmarks include stroke-based repainting.

## REFERENCES

[1] S. Lu, Z. Zhou, J. Lu, Y. Zhu, and A. Kong, “Robust watermarking using generative priors against image editing: From benchmarking to advances,” in International Conference on Learning Representations, 2025.

[2] Z. Huang, W. Heng, and S. Zhou, “Learning to paint with modelbased deep reinforcement learning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 8709–8718.

[3] S. Liu, T. Lin, D. He, F. Li, R. Deng, X. Li, E. Ding, and H. Wang, “Paint Transformer: Feed forward neural painting with stroke prediction,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2021, pp. 6578–6587.

[4] X. Zhao, K. Zhang, Z. Su, S. Vasan, I. Grishchenko, C. Kruegel, G. Vigna, Y.-X. Wang, and L. Li, “Invisible image watermarks are provably removable using generative AI,” Advances in neural information processing systems, 2024.

[5] I. Cox, M. Miller, J. Bloom, J. Fridrich, and T. Kalker, Digital Watermarking and Steganography, 2nd ed. Morgan Kaufmann, 2008.

[6] ShieldMnt, “invisible-watermark,” https://github.com/ShieldMnt/ invisible-watermark, 2020.

[7] J. Zhu, R. Kaplan, J. Johnson, and L. Fei-Fei, “HiDDeN: Hiding data with deep networks,” in European conference on computer vision. Springer, 2018, pp. 682–697.

[8] M. Tancik, B. Mildenhall, and R. Ng, “StegaStamp: Invisible hyperlinks in physical photographs,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 2114–2123.

[9] K. A. Zhang, L. Xu, A. Cuesta-Infante, and K. Veeramachaneni, “Robust invisible video watermarking with attention,” arXiv preprint arXiv:1909.01285, 2019.

[10] T. Bui, S. Agarwal, and J. P. Collomosse, “TrustMark: Robust watermarking and watermark removal for arbitrary resolution images,” in IEEE/CVF International Conference on Computer Vision, ICCV. IEEE, 2025, pp. 18 629–18 639.

[11] P. Fernandez, G. Couairon, H. Jegou, M. Douze, and T. Furon, “The´ Stable Signature: Rooting watermarks in latent diffusion models,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 22 409–22 420.

[12] Y. Wen, J. Kirchenbauer, J. Geiping, and T. Goldstein, “Tree-Ring watermarks: Fingerprints for diffusion images that are invisible and robust,” arXiv preprint arXiv:2305.20030, 2023.

[13] Z. Yang, K. Zeng, K. Chen, H. Fang, W. Zhang, and N. Yu, “Gaussian Shading: Provable performance-lossless image watermarking for diffusion models,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 12 162–12 171.

[14] B. An, M. Ding, T. Rabbani, A. Agrawal, Y. Xu, C. Deng, S. Zhu, A. Mohamed, Y. Wen, T. Goldstein, and F. Huang, “WAVES: benchmarking the robustness of image watermarks,” in Forty-first International Conference on Machine Learning, ICML, 2024.

[15] J. Yang, Q. Li, W. Zong, Y.-W. Chow, W. Susilo, H. Yu, C. Wang, and S. Gao, “One prompt is enough: Watermark laundering through foundation image models,” arXiv preprint arXiv:2609.01249, 2026.

[16] Y. Zhang, D. Ye, S. Shen, and J. Wang, “StyleMark: a robust watermarking method for art style images against black-box arbitrary style transfer,” arXiv preprint arXiv:2412.07129, 2024.

[17] Y. Luo, T. Zhou, F. Liu, and Z. Cai, “IRWArt: Levering watermarking performance for protecting high-quality artwork images,” in Proceedings of the ACM Web Conference 2023. Association for Computing Machinery, 2023, p. 2340–2348.

[18] F. Shi, L. Li, K. Chen, G. Feng, and X. Zhang, “AuthenLoRA: Entangling stylization with imperceptible watermarks for copyright-secure LoRA adapters,” arXiv preprint arXiv:2511.21216, 2025.

[19] S. Shan, J. Cryan, E. Wenger, H. Zheng, R. Hanocka, and B. Y. Zhao, “Glaze: Protecting artists from style mimicry by text-to-image models,” in 32nd USENIX Security Symposium (USENIX Security 23), 2023, pp. 2187–2204.

[20] R. Honig, J. Rando, N. Carlini, and F. Tram¨ er, “Adversarial perturbations\` cannot reliably protect artists from generative AI,” in International Conference on Learning Representations, 2025.

[21] T. Hu, R. Yi, H. Zhu, L. Liu, J. Peng, Y. Wang, C. Wang, and L. Ma, “Stroke-based neural painting and stylization with dynamically predicted painting region,” in Proceedings ofthe 31st ACM International Conference on Multimedia, 2023, pp. 7470–7480.

[22] C. Lord and J. P. Collomosse, “StegoNPR: Information hiding in painterly renderings,” in Theory and Practice of Computer Graphics, 2013.

[23] B. Saleh and A. Elgammal, “Large-scale classification of fine-art paintings: Learning the right metric on the right feature,” arXiv preprint arXiv:1505.00855, 2015.

[24] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2022, pp. 10 674–10 685.

[25] K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, “Image quality assessment: Unifying structure and texture similarity,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.

[26] S. Fu, N. Tamir, S. Sundaram, L. Chai, R. Zhang, T. Dekel, and P. Isola, “DreamSim: Learning new dimensions of human visual similarity using synthetic data,” in Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS, 2023.

[27] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, “MUSIQ: multi-scale image quality transformer,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2021, pp. 5128–5137.

[28] B. Chen and G. Wornell, “Quantization index modulation: a class of provably good methods for digital watermarking and information embedding,” IEEE Transactions on Information Theory, 2001.