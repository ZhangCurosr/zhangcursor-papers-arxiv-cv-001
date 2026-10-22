# Clinical radar heart sounds with echocardiographic phenotypes of ventricular dysfunction and valvular disease

Jinye Li, Qingchao Chen <sup>✉</sup> <sup>∗†‡</sup>

## Abstract

Radar heart sounds capture cardiac mechanical activity without contact, while echocardiography characterizes ventricular and valvular function. Paired measurements connect these signals to structural heart disease, yet such data remain sparsely represented in public resources. Here we present 537 radar heart-sound recordings paired with one echocardiographic report for each of 162 patients. The cohort covers three overlapping phenotypes: left ventricular ejection fraction (LVEF) below 50%, regional wall-motion abnormality and at least moderate valvular regurgitation, present in 38, 43 and 39 patients, respectively. Original complex signals, processed radar heart sounds and temporal quality segmentation support signal-to-phenotype analysis. Technical validation retains 95.93% of signal duration and confirms reproducible waveform reconstruction. Patient-level five-fold evaluation achieves a mean outer-test AUROC of $0 . 7 7 0 \pm 0 . 1 3 8$ for reduced LVEF, identifying mechanical information associated with ventricular systolic dysfunction. This resource supports research linking non-contact cardiac mechanical signals to echocardiographic structural and functional phenotypes.

Keywords: radar heart sound; echocardiography; reduced LVEF; regional wall motion; valvular disease

## 1 Background & Summary

Structural heart disease, including ventricular dysfunction and valvular lesions, contributes to heart failure, disability and premature mortality, creating a substantial public-health burden[1, 2]. A US population-based study estimated moderate or severe valve disease in 2.5% of adults and 13.3% of those aged 75 years or older[2]. Echocardiography is a first-line imaging method and a clinical reference standard for characterizing cardiac structure and function. It quantifies left ventricular ejection fraction (LVEF), identifies regional wall-motion abnormalities and grades valvular regurgitation[3, 4], providing clinically defined phenotypes of ventricular and valvular dysfunction.

Public radar resources have advanced non-contact assessment of cardiac timing and rhythm through recordings paired with ECG and other physiological references[5, 6, 7]. Representative collections include radar heart sounds with digital-stethoscope recordings, hemodynamic responses measured with impedance cardiography and blood pressure, and millimeter-wave signals referenced to ECG, respiration and pulse waveforms. These collections were acquired from healthy volunteers and do not pair radar recordings with echocardiographic labels of ventricular dysfunction or valve disease. Thus, the missing link is between radar-recorded cardiac mechanical activity and clinically characterized structural abnormalities. Addressing this gap requires patient recordings paired with measurements of systolic function, regional contraction and valvular regurgitation.

Here we present 537 radar heart-sound recordings paired with one retained echocardiographic report for each of 162 patients assessed in the Department of Cardiology at Peking University First Hospital. The cohort covers three nonexclusive phenotypes: LVEF below 50%, regional wall-motion abnormality and at least moderate regurgitation of any valve, present in 38, 43 and 39 patients, respectively. Repeated recordings within patients capture cardiac mechanical waveforms, while original complex signals, processed radar heart sounds (RHS) and temporal quality segmentation support analyses at both signal and patient levels.

The paired resource supports investigation of how cardiac mechanical waveforms relate to ventricular systolic function, regional wall motion and valvular abnormalities. Technical validation retains 95.93% of nominal signal duration and confirms reconstruction of processed RHS from original complex samples. Patient-level five-fold evaluation achieves a mean outer-test AUROC of 0.770±0.138 for reduced LVEF, demonstrating identifiable mechanical information associated with systolic dysfunction. Together, clinical phenotyping, waveform traceability and patient-level evaluation provide a foundation for developing non-contact cardiac mechanical markers and investigating overlapping structural and functional abnormalities.

## 2 Methods

## 2.1 Participants and clinical assessment

The study was conducted in the Department of Cardiology at Peking University First Hospital among patients undergoing clinical cardiac assessment. Each patient underwent repeated oneminute radar measurements while resting supine. Measurements across visits were grouped within the same individual. The clinical cohort comprises 162 patients with numerical LVEF and binary classifications of regional wall-motion abnormality and valvular disease. Each patient is paired with one latest echocardiographic assessment, yielding 162 clinical assessments associated with 537 radar recordings.

Patients range from 18 to 89 years of age, with a mean of 63.22 years (SD 13.00); 109 are male and 53 female. LVEF is available for all 162 patients and ranges from 15.0% to 80.1%. Table 1 summarizes the demographic and functional characteristics of the cohort and its overlapping phenotype-positive groups.

## 2.2 Radar system and acquisition protocol

The radar system combined a Texas Instruments IWR1443 radio-frequency board with a DCA1000EVM acquisition board. One transmitting antenna and four receiving antennas were enabled (1TX4RX): TX0 was active, and TX1 and TX2 were disabled. Each chirp produced one slow-time observation. The processing path evaluated here uses the third receiving channel (RX3); its complex I/Q samples provide the input to heart-sound extraction. Figure 1 shows the clinical acquisition environment and patient–radar placement alongside the FMCW configuration and analysis workflow.

Table 1: Demographic and LVEF distribution in the clinical cohort and phenotype-positive subgroups. Percentages use the column-specific denominator. Phenotype groups are nonexclusive. Each patient contributes one echocardiographic assessment. SD is the sample standard deviation; IQR denotes the first and third quartiles.
<table><tr><td>Characteristic</td><td>All</td><td>LVEF &lt; 50%</td><td>RWMA</td><td>Valve disease</td></tr><tr><td>Patients</td><td>162</td><td>38</td><td>43</td><td>39</td></tr><tr><td>Male</td><td>109 (67.3%)</td><td>30 (78.9%)</td><td>29 (67.4%)</td><td>23 (59.0%)</td></tr><tr><td>Female</td><td>53 (32.7%)</td><td>8 (21.1%)</td><td>14 (32.6%)</td><td>16 (41.0%)</td></tr><tr><td>Age, mean (SD), years</td><td>63.22 (13.00)</td><td>62.21 (16.43)</td><td>64.19 (11.11)</td><td>68.08 (11.69)</td></tr><tr><td>LVEF, mean (SD), %</td><td>59.75 (14.58)</td><td>37.65 (8.68)</td><td>50.64 (8.80)</td><td>54.87 (17.96)</td></tr><tr><td>Age, median (IQR), years</td><td>66.5 (56.2–72.0)</td><td>68.0 (50.2–73.0)</td><td>67.0 (56.5–73.0)</td><td>69.0 (65.0–75.5)</td></tr><tr><td>LVEF, median (IQR), %</td><td>63.7 (50.8–71.2)</td><td>38.7 (33.5–44.6)</td><td>51.2 (46.7–57.5)</td><td>59.5 (41.0–69.1)</td></tr><tr><td>Age 18–39 years</td><td>7 (4.3%)</td><td>3 (7.9%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td></tr><tr><td>Age 40–49 years</td><td>19 (11.7%)</td><td>6 (15.8%)</td><td>8 (18.6%)</td><td>4 (10.3%)</td></tr><tr><td>Age 50–59 years</td><td>25 (15.4%)</td><td>6 (15.8%)</td><td>5 (11.6%)</td><td>3 (7.7%)</td></tr><tr><td>Age 60–69 years</td><td>54 (33.3%)</td><td>6 (15.8%)</td><td>12 (27.9%)</td><td>13 (33.3%)</td></tr><tr><td>Age 70–79 years</td><td>48 (29.6%)</td><td>12 (31.6%)</td><td>17 (39.5%)</td><td>14 (35.9%)</td></tr><tr><td>Age 80–98 years</td><td>9 (5.6%)</td><td>5 (13.2%)</td><td>1 (2.3%)</td><td>5 (12.8%)</td></tr><tr><td>LVEF &lt;40%</td><td>20 (12.3%)</td><td>20 (52.6%)</td><td>5 (11.6%)</td><td>8 (20.5%)</td></tr><tr><td>LVEF 40-&lt;50%</td><td>18 (11.1%)</td><td>18 (47.4%)</td><td>12 (27.9%)</td><td>8 (20.5%)</td></tr><tr><td>LVEF 50-&lt;60%</td><td>26 (16.0%)</td><td>0 (0.0%)</td><td>20 (46.5%)</td><td>4 (10.3%)</td></tr><tr><td>LVEF 60-&lt;70%</td><td>52 (32.1%)</td><td>0 (0.0%)</td><td>6 (14.0%)</td><td>10 (25.6%)</td></tr><tr><td>LVEF ≥70%</td><td>46 (28.4%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td><td>9 (23.1%)</td></tr></table>

The FMCW profile started at 77.000 GHz with a frequency slope of 63.343 MHz/µs. The ramp ended at 63.14 µs, giving the configured full-ramp bandwidth of approximately 3,999.48 MHz. Idle time was 10.00 µs, TX start time 1.00 µs and ADC start time 6.00 µs. Each chirp contained 512 complex ADC samples per receive channel at 9.121 MS/s. The sampled aperture therefore lasted approximately 56.13 µs and occupied approximately 3.556 GHz of the ramp. The nominal range resolution c/(2B) is approximately 3.75 cm when calculated from the full ramp, or 4.22 cm using the sampled aperture. The receive gain was 30 dB, and the first and second analog high-pass corner settings were 175 and 350 kHz, respectively.

Each frame contained one chirp, with a frame period of 1 ms. A recording consisted of 60,000 frames, corresponding to a slow-time sampling rate of 1,000 Hz and a duration of 60 s. Acquisition was initiated by software triggering with zero frame-trigger delay. The 9.121-MS/s fast-time rate resolves the sampled chirp, whereas the 1,000-Hz slow-time rate tracks the phase evolution of cardiac-related vibration.

A complex radar acquisition is represented by slow-time observations and fast-time samples. For the evaluated receive channel, each recording has 60,000 observations with 512 complex samples per observation. Heart-sound extraction transforms this two-dimensional signal into a 60,000-sample

a

b

vibration time series. Table 2 summarizes the acquisition parameters and the settings used for waveform and spectral analysis.

Patients rested supine during recordings of cardiac-related chest-surface vibration. The radar faced the anterior chest and was aimed approximately 2 cm inferolateral, towards the patient’s left, to the intersection of the chest midline and the line joining the nipples. The radar–chest distance was approximately 50–80 cm. This landmark served as the reference for repeated oneminute acquisitions over anterior chest regions. Clothing was unrestricted, and the protocol used resting measurements. Automatic target-cell selection used the largest time-averaged range-profile magnitude. The collection comprises radar recordings paired with echocardiographic reports; ECG was not acquired.

![](images/af0cd3a8fb55c3cfc0ed26d70322c585e6208d1c2f967a950a9befb27ca5e22b.jpg)

![](images/d069a20352b3711be31af703c61ac0bb6d0cba7ee1455983787dd9b14a71aae3.jpg)

![](images/7d4ed774bcf07db6ecc09944324645ea88d5603407d3d0c6471da034b8bcdf47.jpg)

![](images/f470671da8177da81ca9edc4d1b7ff08c6103904b91a29b88a84e66b17d96c3d.jpg)  
Figure 1: Clinical acquisition environment, FMCW configuration and radar heart-sound workflow. a, Photograph of the clinical acquisition setup, showing a supine patient and the arm-mounted IWR1443BOOST radar and DCA1000EVM capture board positioned above the chest. b, FMCW ramp and sampled aperture: 77-GHz start, $6 3 . 3 4 3 \mathrm { - M H z } / \mu \mathrm { s }$ slope, 63.14-µs ramp, 512 ADC samples at $9 . 1 2 1 ~ \mathrm { M S / s }$ , and one chirp per 1-ms frame. c, Complex-signal input, range FFT, maximummagnitude range-cell selection, final-cell phase extraction, filtering and smoothing, followed by outlier removal and manual quality review. Ofline reconstruction uses the final selected cell, preserving historical selection decisions. d, One retained report per selected patient supplies clinical context for three overlapping targets: LVEF below 50%, regional wall-motion abnormality and at least moderate regurgitation of any valve.

Table 2: TI IWR1443/DCA1000EVM acquisition configuration and ofline analysis settings. Fullramp bandwidth and sampled bandwidth are distinct; the 1-ms frame period defines the exported 1,000-Hz vibration time series.
<table><tr><td>Parameter</td><td>Setting</td><td>Interpretation</td></tr><tr><td>RF board / acquisition board</td><td>TI IWR1443 / DCA1000EVM</td><td>Non-contact FMCW front end and raw-data capture</td></tr><tr><td>Transmit / receive configuration</td><td>1TX4RX</td><td>TX0 enabled; TX1/TX2 disabled; four complex receive channels</td></tr><tr><td>Chirp start frequency</td><td>77.000 GHz</td><td>FMCW profile start</td></tr><tr><td>Frequency slope</td><td>63.343 MHz/µs</td><td>Linear frequency modulation</td></tr><tr><td>Idle / TX start time</td><td>10.00 /  1.00 µs</td><td>Chirp timing settings</td></tr><tr><td>ADC start / ramp-end time</td><td>6.00 / 63.14 µs</td><td>Sampled aperture within the ramp</td></tr><tr><td>ADC samples / fast-time rate</td><td>512 / 9.121 MS/s</td><td>Per chirp and per receive channel</td></tr><tr><td>Full-ramp / sampled bandwidth</td><td>3,999.48 / approx. 3,555.64</td><td>Full profile versus 56.13-µs sampled</td></tr><tr><td>Nominal range resolution</td><td>MHz 3.75 / 4.22 cm</td><td>aperture Full-ramp / sampled-aperture calculation</td></tr><tr><td>Receive gain / HPF corners</td><td>30 dB / 175 and 350 kHz</td><td>Receiver analog settings</td></tr><tr><td>Chirp / loop configuration</td><td>Index 0; one loop per frame</td><td>No transmitter multiplexing in the</td></tr><tr><td>Frame period / number of frames</td><td>1 ms / 60,000</td><td>exported recording Recording timing</td></tr><tr><td>Trigger / trigger delay</td><td>Software / 0 µs</td><td>Frame acquisition start</td></tr><tr><td>Slow-time rate / duration</td><td>1,000 Hz / 60 s</td><td>One temporal observation per frame</td></tr><tr><td>ADC quantization</td><td>16-bit per I/Q component</td><td>Digitized complex I/Q measurements</td></tr><tr><td>RHS length / reconstruction channel</td><td>60,000 / RX3, one-based</td><td>Continuous waveform; explicit</td></tr><tr><td>Target-cell selection</td><td>Maximum mean magnitude</td><td>single-channel reconstruction Peak within the configured search region;</td></tr><tr><td>Acquisition posture / state</td><td>Supine; resting</td><td>reconstruction uses the final selected cell Repeated chest acquisitions; no specific clothing requirement</td></tr><tr><td>Chest reference / radar distance</td><td>Approximately 2 cm</td><td>Patient&#x27;s left from the</td></tr><tr><td>Clinical correspondence</td><td>inferolateral / 50–80 cm One latest report per patient</td><td>midline-nipple-line intersection Every included radar recording is within</td></tr><tr><td>Quality-monitoring sequence</td><td>Outlier removal, then manual</td><td>three calendar months Final reviewed interval mask used for</td></tr><tr><td>Phase-amplitude conversion</td><td>inspection 79 GHz</td><td>analysis Phase-amplitude conversion</td></tr><tr><td>constant RHS band-pass / smoothing</td><td>Order 4, 16–80 Hz / 31</td><td>Forward-backward Butterworth and</td></tr><tr><td>Envelope / envelope smoothing</td><td>points, degree 3 Hilbert magnitude / 101</td><td>Savitzky-Golay Illustrative signal representation</td></tr><tr><td>Spectral window lengths / overlap</td><td>points, degree 3 2,048 and 4,096 points / 50%</td><td>Fully retained windows; edge exclusions</td></tr></table>

## 2.3 Extraction of radar heart sounds

Let $s [ n , m ]$ denote the complex $\mathrm { I / Q }$ signal for the evaluated receive channel, with slow-time index n and fast-time index m. The fast-time mean is removed independently for each chirp, and a 512-point Fourier transform yields the complex range profile $S [ n , k ]$ . The target cell maximizes the slow-time mean magnitude within the candidate region: $\begin{array} { r } { A [ k ] = N ^ { - 1 } \sum _ { n = 1 } ^ { N } | S [ n , k ] | } \end{array}$ and $k ^ { * } = \arg \operatorname* { m a x } _ { k \in \mathcal { R } } A [ k ]$ ， where R denotes the candidate cells. One cell is fixed for phase tracking throughout a recording. Historical manual adjustment of the automatically selected cell occurred in 60 of the 537 signalvalidation recordings; waveform reconstruction uses the final selected cell in each case.

For the selected range-cell trace $z [ n ]$ , the unwrapped phase is $\phi [ n ] = \mathrm { u n w r a p } ( \arg z [ n ] )$ . The amplitude convention used during processing is $a [ n ] = ( c / f _ { \mathrm { c o n v } } ) \phi [ n ] / ( 2 \pi )$ , with $f _ { \mathrm { c o n v } } = 7 9 ~ \mathrm { G H z }$ This conversion constant is retained separately from the 77-GHz start frequency of the acquisition profile. Waveform illustrations and reconstruction comparisons use normalized amplitude, with each compared signal independently mean-centred and RMS-normalized.

A fourth-order Butterworth band-pass design at 16–80 Hz is applied forward and backward to obtain a zero-net-phase-delay ofline waveform. The filtered signal is smoothed with a 31-point, third-degree Savitzky–Golay filter[8]. The resulting signed mechanical vibration series is the radar heart sound (RHS). Its Hilbert analytic-signal magnitude supplies an amplitude envelope, which is smoothed with a 101-point, third-degree Savitzky–Golay filter for illustration. The signed waveform and envelope provide separate signal representations.

Signal-quality monitoring combines automated outlier removal with subsequent manual inspection. The quality-review implementation identifies large deviations from the median using a medianabsolute-deviation-based robust scale, with a standard-deviation fallback when that scale is zero. Flagged isolated samples are interpolated on the original sampling grid for inspection. Reviewers then inspect the signals and mark excluded time intervals. Validation applies the reviewed interval mask to the RHS. Exclusion intervals are unioned, and the original time axis is maintained for all retained samples.

## 2.4 Echocardiographic measurements and phenotype definitions

Echocardiographic assessment characterizes chamber dimensions, ventricular volumes, LVEF, regional contraction, mitral inflow, tissue Doppler and valvular function. The latest examination is retained for each patient. LVEF measurements include Teichholz and biplane Simpson approaches, alongside two-dimensional/M-mode and Doppler assessment. The analysis uses the reported clinical measurements.

Reduced LVEF is defined by the strict threshold LVEF < 50%, applied to the reported clinical LVEF. This phenotype is present in 38 patients, with 124 above or at the threshold. Regional wall-motion abnormality is represented as a binary clinical phenotype and is present in 43 patients. These targets characterize global systolic function and regional contraction, respectively.

Valvular disease is defined as at least moderate regurgitation of any valve, including the aortic, mitral, tricuspid or another valve. It is present in 39 patients, with 123 negative for this target. Mild regurgitation alone and stenosis alone are negative under this definition. The three phenotypes are nonexclusive, reflecting the coexistence of ventricular and valvular abnormalities. Lesion-specific echocardiographic evaluation of regurgitation is described in clinical recommendations[4].

## 2.5 Pairing of radar and echocardiographic measurements

All repeated radar measurements from one patient are paired with that patient’s latest echocardiographic assessment. Patient-level grouping is maintained across visits, so that each individual contributes one clinical phenotype vector and multiple radar observations. The paired design contains 162 clinical assessments and 537 radar recordings.

The pairing protocol specifies an interval of no more than three calendar months between radar measurement and echocardiographic examination, in either direction. Clinical summaries use one observation per patient, whereas signal-level experiments use individual radar recordings.

## 2.6 Ofline validation experiments and statistical analysis

Signal-level technical validation examines 537 recordings to assess continuity, retained exposure, spectral sensitivity and processing reproducibility. Numerical checks evaluate RHS length, finite values and nonconstant content. Reviewed exclusion intervals are unioned and applied as a temporal mask on the original time axis. Retained duration is calculated before excluding the first and last second for spectral analysis. These edges are omitted uniformly to reduce edge influence. Excluded intervals include their start sample and omit their end sample.

Welch-style spectra[9] average periodic-Hann-windowed, constant-detrended periodograms of 2,048-sample windows with 50% overlap. Only windows entirely within retained data contribute. The 16–80-Hz centroid is $\begin{array} { r } { C = \sum f P ( f ) / \sum P ( f ) } \end{array}$ , with both sums restricted to that band. A bandpower fraction uses 16–80 Hz in the numerator and 1–200 Hz in the denominator. This fraction quantifies the spectral concentration of the filtered and smoothed RHS representation.

Within-record variation is evaluated by comparing centroids in the first and last 30 s under the same quality mask. Window-length sensitivity is evaluated with 2,048- and 4,096-sample windows. Absolute diferences and descriptive Spearman correlation quantify these comparisons. Processing reproducibility is evaluated in 12 deterministically selected recordings. The complex RX3 signal undergoes fast-time centring, range transformation, final-cell phase extraction and the specified filtering and smoothing. RHS and reconstructed waveforms are independently mean-centred and RMS-normalized after removing one second from each edge. Pearson correlation and normalized RMSE assess reproduction of normalized waveform shape.

Clinical summaries use one observation per patient and available finite values independently per variable. Standard deviations use the sample denominator, and quartiles use linear interpolation. Values are analyzed without imputation or winsorization. Binary targets retain valid zero-or-one entries, with other values classified as invalid or missing. The three disease axes are nonexclusive, and each numerical field has its own available-value denominator.

## 2.7 Patient-level evaluation of reduced LVEF

To evaluate whether radar cardiac mechanical signals characterize left ventricular systolic function, an LVEF classification experiment used echocardiographic LVEF as the clinical reference and defined $\mathrm { L V E F } < 5 0 \%$ as the positive target. The experiment evaluated all 162 patients in the study cohort: 124 with LVEF at or above 50% and 38 below 50%.

The RHS waveform was resampled to 500 Hz and divided into nonoverlapping 5-s windows. Within each outer fold, a global median and interquartile range were estimated exclusively from the training set and used for robust scaling of the training, validation and test signals. Applying the same training-derived scale across patients preserves their relative amplitude diferences. The model jointly encoded the RHS time-domain waveform, Hilbert envelope and logarithmic shorttime Fourier transform (STFT) representation. Recording-level multiple-instance learning (MIL) aggregated consecutive windows within each recording. Training used focal loss with automatically determined class weights to accommodate the lower prevalence of the positive target.

Evaluation used five outer cross-validation folds grouped by patient identity. In each fold, approximately 20% of patients formed the outer test set, while the remaining patients were divided into training and validation sets corresponding to approximately 70% and 10% of the full cohort, respectively. All recordings and windows from the same patient remained in one partition. The validation set was used for model selection, and each outer test set was used for final evaluation. Performance was assessed at the patient level using the area under the receiver operating characteristic curve (AUROC), area under the precision–recall curve (PR-AUC), macro-averaged F1 and accuracy.

## 2.8 Distribution and overlap of clinical phenotypes

The cohort contains 38 patients with LVEF below 50% (23.46%), 43 with regional wall-motion abnormalities (26.54%) and 39 with valvular disease (24.07%). All percentages use the same denominator of 162 patients, each with one retained report. Mean LVEF is 59.75% (SD 14.58), with median 63.65% and interquartile range 50.75–71.18%. Figure 2 shows the distribution and intersections of these labels.

The clinical population is predominantly middle-aged and older. The age bands 18–39, 40– 49, 50–59, 60–69, 70–79 and 80 years or older contain seven, 19, 25, 54, 48 and nine patients, respectively; 111 of 162 (68.52%) are aged at least 60 years. Male and female patients account for 67.28% and 32.72%. Numerical LVEF is below 40% in 20 patients and between 40% and strictly below 50% in 18; the remaining 26, 52 and 46 patients lie in the 50–below-60%, 60–below-70% and at-least-70% bands, respectively. Repeated radar acquisitions do not increase these patient-level denominators.

Table 1 summarizes demographic and LVEF variation across the overlapping phenotype-positive subgroups. Mean age is 62.21 years (SD 16.43) in the reduced-LVEF group, 64.19 (11.11) in the wall-motion group and 68.08 (11.69) in the valve group. Mean LVEF is 37.65%, 50.64% and 54.87%, respectively. These summaries describe each clinical axis within the same patient population.

Among the 38 patients with reduced LVEF, 17 also have wall-motion abnormality and 16 have valvular disease; six carry all three labels. Overall, 56 patients carry one target, 23 carry two and six carry all three. The remaining 77 form the target-negative group for these three phenotypes. This overlap motivates multilabel analysis of the clinical targets.

The echocardiographic assessments describe the same 162-patient population throughout the demographic summaries and phenotype plots. Each numerical variable uses the denominator of patients with an available measurement. Patient-level summaries preserve the distinction between

a

b

clinical variation and repeated radar observations.

d  
f  
![](images/ccc65410253fd70f9c833d8626295a8e1784a816e9897d91b658147dd54cd49b.jpg)

![](images/711cf01c2490075072f5eb91073f7cd2fef4a8db04e506221b731794fd102509.jpg)

![](images/4d1ffd0fa542ee4d0aea170a5544814b98ca3705f51fd0a127024b2f259f52ed.jpg)

![](images/e2a174b3f333ed6cabf6ab0c3943399358fc6a20f48e45883ae4e0f958f75113.jpg)

![](images/78978dd70225b1dbcab7c4ce0d228efa34cabf87e41ccfba3e58a459d36d5296.jpg)

![](images/b432d5862bc4f3142040ec6c63e6d48182c691b089f9d72f03140213ea668bf7.jpg)  
Figure 2: Demographic distribution and overlapping clinical phenotypes in 162 patients, each with one retained report. a, Age-band counts stacked by recorded sex; axis labels give the lower age bound. b, Reported clinical LVEF in 5-percentage-point bins, with the orange line denoting 50% and orange bars denoting values strictly below it. c, Positive counts and percentages for the three nonexclusive targets. Valvular disease means at least moderate regurgitation of any valve. d, Joint label counts; filled-dot rows EF, WM and V denote reduced LVEF, regional wall-motion abnormality and valvular disease. The empty intersection denotes the group with none of the three target labels. e, LVEF by nonexclusive group; boxes span the first to third quartile, horizontal lines show medians, and whiskers span the observed minimum to maximum. f, Sex composition of the whole cohort and phenotype-positive groups, with counts inside the bars. Clinical distributions use one observation per patient.

## 3 Data Records

The radar measurements are represented at three complementary levels: the original complex signal, the processed RHS and temporal quality segmentation (Table 3). These three components link the acquired I/Q signal to the vibration waveform and its analysis intervals. The original complex signal and processed RHS are provided in MATLAB format.

The complex signal retains the fast-time and slow-time dimensions used for range selection and phase tracking. The evaluated RX3 representation contains 60,000 slow-time observations and 512 complex fast-time samples per observation. This representation supports re-extraction of cardiac mechanical signals with alternative range-selection or filtering strategies.

Each RHS is a continuous 60,000-sample signed waveform at 1,000 Hz. Temporal quality segmentation marks retained and excluded intervals following outlier screening and manual review. These intervals remain aligned with the original 60-s time axis, enabling consistent calculation of retained exposure and quality-aware spectral features.

The measurements span the echocardiographic phenotypes described in the Methods. Repeated recordings share a patient’s clinical phenotype vector, supporting both within-patient signal analysis and comparisons across the overlapping clinical groups.

Table 3: Three complementary representations of each radar measurement. MATLAB format is used for the complex signal and processed RHS; quality segments retain their positions on the original time axis.
<table><tr><td>Representation</td><td>Extent</td><td>Analytical role</td></tr><tr><td>Original complex radar signal</td><td>60,000 by 512 per recording</td><td>Complex I/Q representation for the evaluated RX3 channel; supports range selection and phase-based signal extraction</td></tr><tr><td>Processed RHS</td><td>60,000 samples at 1,000 Hz per recording</td><td>Signed cardiac-related vibration waveform; normalized amplitude is used for waveform illustrations and reconstruction comparisons</td></tr><tr><td>Temporal quality segmentation</td><td>Reviewed intervals on each 60-s time axis</td><td>Retained and excluded intervals following outlier screening and manual review; supports quality-aware exposure and spectral analysis</td></tr></table>

## 4 Technical Validation

## 4.1 Signal continuity and quality-aware exposure

All 537 RHS vectors in the signal-validation set contain 60,000 finite, nonconstant samples. Efective excluded intervals lie within the one-minute record. After interval union, kept and deleted durations sum to the nominal duration for all 537 recordings within a tolerance of 2 ms. These checks establish numerical continuity and consistent temporal accounting of the reviewed quality intervals.

Quality exclusions afect 330 recordings (61.45%). The retained duration totals 30,908.45 s, or 8.59 h, corresponding to 95.93% of the nominal 32,220 s. Median retained duration is 58.77 s (interquartile range 56.25–60.00), and the minimum is 40.04 s. The time excluded at the two record edges for spectral analysis is additional to these coverage estimates. Figure 4a shows the cumulative retained-duration distribution.

## 4.2 Spectral sensitivity to quality masking and window length

Every selected recording contributes at least one complete quality-retained window to both spectral window-length analyses and both recording-half analyses. The quality-masked 16–80-Hz centroid has a median of 20.89 Hz (interquartile range 20.19–21.66). The median band-power fraction is 0.873 (0.826–0.905). The spectral distribution describes the RHS representation after band-pass filtering and smoothing.

![](images/6deffb767c78bd07d783897f3dd04c9a24f9372085b57c5d0ba6a7c58e7690e3.jpg)

![](images/5a0bd796a90f4c50a6311281ddd5371df1e54aff0269a793193de0a28a5a0a91.jpg)

![](images/cfff5f8b0cc7607279dffcd41394d30a5ac07eeec768d8fc578c5d64fce84249.jpg)

![](images/77a9b3d261893868376d4c15e8a212afdf6c07f5b1479437e4c9330294ee20ee.jpg)

![](images/ea569427b1cd540065309088275cb072f6ea8ec0135750e9c8b5557abeba8dfe.jpg)

![](images/e42aec3525e583b5a2280fb02262fd44574d24844c21fc45e37b47979acd0079.jpg)  
Figure 3: Representative radar heart-sound waveforms and their spectra. a, c and $\mathrm { e , }$ Six-second excerpts from three recordings selected at approximately the first quartile, median and third quartile of the quality-masked spectral-centroid distribution. Blue denotes the signed RHS waveform and orange its smoothed Hilbert envelope; both use the RMS of the displayed signed excerpt as the amplitude scale. Excerpts lie within retained intervals. b, d and $\mathrm { f , }$ Quality-masked spectra from the respective full recordings, normalized to their maximum power; shading indicates 16–80 Hz. R-identifiers denote anonymous analytical recordings. The examples illustrate variation across the spectral-centroid distribution.

The median absolute centroid change after applying the quality mask is 0.213 Hz (interquartile range 0–1.534), with a maximum of 7.828 Hz. These diferences demonstrate the sensitivity of energy-weighted features to temporal quality segmentation: excluding a small fraction of the duration can substantially change the spectral centroid.

Increasing the analysis window from 2.048 to 4.096 s changes the centroid by a median absolute 0.073 Hz (0.035–0.148), with a maximum of 1.671 Hz. The first and last 30-s centroids have descriptive Spearman correlation 0.815 and median absolute diference 0.279 Hz (0.138–0.600); the maximum half diference is 7.888 Hz. Window-length sensitivity is typically smaller than the change associated with temporal masking or within-record variation, but individual recordings can difer. These comparisons quantify analytical sensitivity and temporal variation within a saved recording.

## 4.3 Reconstruction from raw complex radar samples

The raw-data experiment reconstructs 12 selected recordings using the final selected range cell and the evaluated single-channel processing path. All 12 RHS and reconstructed normalized waveforms have Pearson correlation above 0.999999. Normalized RMSE ranges from 2.65×10<sup>−10</sup> to 1.23×10<sup>−6</sup> (Fig. 5). These results establish close agreement between the processed RHS and the waveform reconstructed from the original complex radar signal.

The reconstruction experiment evaluates decoding, final-cell phase extraction and filtering/smoothing in the 12 sampled records. Independent mean centring and RMS normalization make the comparison a test of waveform shape. Table 4 summarizes the executed experiments and their observation units.

## 4.4 Clinical reference and patient-level phenotypes

The paired design retains one latest echocardiographic assessment for each of 162 patients. Clinical LVEF and the strict 50% threshold identify 38 positive and 124 negative patients. Binary classifications identify 43 patients with regional wall-motion abnormality and 39 with at least moderate valvular regurgitation. Patient-level grouping links repeated mechanical observations to the corresponding clinical assessment.

## 4.5 Discrimination of reduced left ventricular systolic function

Patient-level five-fold cross-validation in the 162-patient study cohort yielded a mean outer-test AUROC of $0 . 7 7 0 \pm 0 . 1 3 8$ . The reported PR-AUC was 0.559, macro-averaged F1 was 0.652 and accuracy was 71.6% (Table 4). These results connect the radar-derived mechanical representation to an echocardiographically defined functional target: non-contact RHS contains information associated with reduced LVEF that can be identified at the patient level. This functional evaluation complements the signal-quality and reconstruction experiments, supporting the use of paired radar and echocardiographic measurements to investigate ventricular systolic dysfunction.

![](images/613c39ab85a8d13426438c28b77beac288eb197fb6b5c31d6b00e8e1061b0b41.jpg)

![](images/f50041871887cbbc6889a3bd5bfe9f5d20efe7b6804caade6dbf02038b760f53.jpg)

![](images/11b3a8cd94c630cc0d9d588a3ae81df46781c2940e8756725d597f5cf15599ef.jpg)

![](images/0b11658684050b6440f8b85382d9e0dd447296d0afc6b89a45ee9963ed348e60.jpg)  
Figure 4: Quality-aware signal validation and analysis sensitivity in 537 selected recordings. ${ \mathrm { a } } ,$ Empirical cumulative distribution of retained duration, calculated from the union of efective deleted intervals before record-edge exclusion. b, Centroid before and after quality masking, with the same one-second edge exclusions. ${ \mathrm { c } } ,$ Quality-masked centroids in the first and last 30 s. d, Quality-masked centroids using 2.048- and 4.096-s windows, each with 50% overlap. Centroids are restricted to 16– 80 Hz. Each point is one recording; gray dashed lines indicate equality. Comparisons use the final reviewed quality masks.

![](images/dd0f606ef15602a6c2fc4c834351dcb00cb9bb73b92ce402e4dd41ea75acf684.jpg)

![](images/4d774f23b4ad85ef7b530707c5edbcd9c57565319b56e24e2df3df46fc48919d.jpg)  
Figure 5: Reconstruction of RHS from the original complex radar signal. a, Pearson correlation of RHS and reconstructed waveforms in 12 deterministically selected recordings. b, Corresponding RMSE after independent mean centring and RMS normalization; the RMSE axis is logarithmic. Both comparisons omit the first and last second. Numbers 01–12 denote the sampled recordings used for normalized waveform-shape reconstruction.

Table 4: Signal and functional validation. Medians of signal metrics are followed by interquartile ranges. Signal-level observations use 537 recordings, including 12 sampled records for reconstruction; LVEF discrimination uses the 162-patient study cohort. AUROC summarizes the five outer test folds.
<table><tr><td>Experiment / metric</td><td>Unit / n</td><td>Observed result</td></tr><tr><td>RHS numerical integrity</td><td>Recordings / 537</td><td>All 60,000-point vectors finite and nonconstant</td></tr><tr><td>Quality interval accounting</td><td>Recordings / 537</td><td>No kept-plus-deleted duration discrepancy above 2 ms</td></tr><tr><td>Retained signal exposure</td><td>Recordings / 537</td><td>30,908.45 s total; 95.93% of nominal duration</td></tr><tr><td>Retained duration, s</td><td>Recordings / 537</td><td>58.77 (56.25–60.00); minimum 40.04</td></tr><tr><td>Quality-masked centroid, Hz</td><td>Recordings / 537</td><td>20.89 (20.19–21.66)</td></tr><tr><td>Processed band-power fraction</td><td>Recordings / 537</td><td>0.873 (0.826–0.905)</td></tr><tr><td>Absolute masking change, Hz</td><td>Recordings / 537</td><td>0.213 (0–1.534); maximum 7.828</td></tr><tr><td>Absolute half-record difference, Hz</td><td>Recordings / 537</td><td>0.279 (0.138–0.600); Spearman correlation 0.815</td></tr><tr><td>Absolute window-length change, Hz</td><td>Recordings / 537</td><td>0.073 (0.035–0.148); maximum 1.671</td></tr><tr><td>Raw-to-RHS normalized shape</td><td>Recordings / 12</td><td>Pearson correlation above 0.999999 in every sampled record</td></tr><tr><td>Raw-to-RHS normalized RMSE</td><td>Recordings / 12</td><td> $2 . 6 5 \times 1 0 ^ { - 1 0 } \mathrm { ~ t o ~ } 1 . 2 3 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Reduced-LVEF AUROC</td><td>Patients / 162</td><td>0.770 ± 0.138; mean across five outer test folds</td></tr><tr><td>Reduced-LVEF PR-AUC</td><td>Patients / 162</td><td>0.559</td></tr><tr><td>Reduced-LVEF macro-F1</td><td>Patients / 162</td><td>0.652</td></tr><tr><td>Reduced-LVEF accuracy</td><td>Patients / 162</td><td>71.6%</td></tr></table>

## 5 Usage Notes

The paired recordings support cardiac-mechanical research with echocardiographic clinical context. Applications include quality-aware feature extraction, comparison of processing methods and multilabel analysis of ventricular dysfunction, regional contraction and valvular regurgitation. Raw complex samples permit alternative signal representations to be evaluated against the processed RHS.

Quality-aware analysis should preserve the original time axis and use windows wholly contained in retained intervals. A minimum retained-duration criterion should be specified before supervised modelling. The supplied masks and retained-duration summaries provide the temporal information needed to apply such a criterion.

Clinical analyses should preserve the nonexclusive clinical phenotype definitions. Reduced LVEF uses the reported clinical value and the strict 50% threshold; valvular disease uses at least moderate regurgitation of any valve. The 77 patients with no target labels form the three-target-negative group. Report-level measurement methods, age, sex and coexisting phenotypes are relevant covariates for subsequent association analyses.

Training, validation and test partitions should be formed at patient level, with all recordings across visits from one patient assigned to the same partition. Subsequent modelling should report patient-level denominators, label overlap, missingness and class-specific performance with uncertainty. This grouping aligns model evaluation with the patient-level echocardiographic targets.

## 6 Code Availability

The study software implements radar control, complex-signal processing, range selection, phasebased RHS extraction, quality segmentation and the technical validation experiments. The evaluated operations and parameters are specified in the Methods. The acquisition and analysis software is maintained with the study materials.

## 7 Data Availability

The radar measurements are maintained by the investigators as original complex-signal MATLAB data, processed RHS MATLAB data and temporal quality segmentation. Each measurement is associated with the corresponding patient-level echocardiographic phenotype.

## References

[1] Gregory A. Roth, George A. Mensah, Catherine O. Johnson, et al. Global burden of cardiovascular diseases and risk factors, 1990–2019: Update from the GBD 2019 study. Journal of the American College of Cardiology, 76(25):2982–3021, 2020. doi: 10.1016/j.jacc.2020.11.010.

[2] Vuyisile T. Nkomo, Julius M. Gardin, Thomas N. Skelton, John S. Gottdiener, Christopher G. Scott, and Maurice Enriquez-Sarano. Burden of valvular heart diseases: a population-based study. The Lancet, 368(9540):1005–1011, 2006. doi: 10.1016/S0140-6736(06)69208-8.

[3] Roberto M. Lang, Luigi P. Badano, Victor Mor-Avi, et al. Recommendations for cardiac chamber quantification by echocardiography in adults: An update from the american society of echocardiography and the european association of cardiovascular imaging. Journal of the American Society of Echocardiography, 28:1–39.e14, 2015. doi: 10.1016/j.echo.2014.10.003.

[4] William A. Zoghbi, David Adams, Robert O. Bonow, et al. Recommendations for noninvasive evaluation of native valvular regurgitation. Journal of the American Society of Echocardiography, 30:303–371, 2017. doi: 10.1016/j.echo.2017.01.007.

[5] Kilin Shi, Sven Schellenberger, Christoph Will, Tobias Steigleder, Fabian Michler, Jonas Fuchs, Robert Weigel, Christoph Ostgathe, and Alexander Koelpin. A dataset of radar-recorded heart sounds and vital signs including synchronised reference sensor signals. Scientific Data, 7:50, 2020. doi: 10.1038/s41597-020-0390-1.

[6] Sven Schellenberger, Kilin Shi, Tobias Steigleder, Anke Malessa, Fabian Michler, Laura Hameyer, Nina Neumann, Fabian Lurz, Robert Weigel, Christoph Ostgathe, and Alexander Koelpin. A dataset of clinically recorded radar vital signs with synchronised reference sensor signals. Scientific Data, 7:291, 2020. doi: 10.1038/s41597-020-00629-5.

[7] Ruochen Wu, Laura Miro, Albert Aguasca, Antoni Broquetas, Cosme Garcia, and Montse Najar. A dataset of 120 GHz millimeter-wave radar vital signals with synchronized reference recordings. Scientific Data, 13:741, 2026. doi: 10.1038/s41597-026-07016-6.

[8] Abraham Savitzky and Marcel J. E. Golay. Smoothing and diferentiation of data by simplified least squares procedures. Analytical Chemistry, 36:1627–1639, 1964. doi: 10.1021/ac60214a047.

[9] Peter D. Welch. The use of fast fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. IEEE Transactions on Audio and Electroacoustics, 15:70–73, 1967. doi: 10.1109/TAU.1967.1161901.

b

## A Supplementary Information

Supplementary Figures S1 and S2 detail the acquisition timing and phase-based RHS extraction. The acquisition diagram distinguishes fast-time sampling within each chirp from slow-time observations across frames. The processing diagram connects the complex signal, the selected range-cell phase, the filtered RHS and the reviewed quality intervals.

Reconstruction uses the complex RX3 signal with 512 fast-time samples per observation, the final selected range cell, phase unwrapping, the 79-GHz amplitude conversion constant, forward– backward Butterworth filtering and Savitzky–Golay smoothing. Spectral analysis preserves the original time axis, applies the reviewed quality intervals and excludes one second from each recording edge. Only windows fully contained in retained intervals contribute to the spectrum.

![](images/0d26039c79631061d3f8c6153b1dcef47bb466e2f59efcdeb290a59b171f47c2.jpg)

![](images/37cc5bda152d70f618f19df421a875732c6b900075c437785a7249ed1a6b6da8.jpg)  
Figure S1: RF timing, sampled aperture and raw $\mathrm { I / Q }$ representation. a, The 77-GHz-start FMCW ramp, $6 3 . 3 4 3 \mathrm { - M H z } / \mu \mathrm { s }$ slope, $6 { - } \mu \mathrm { s }$ ADC delay and 512 samples at 9.121 $\mathrm { M S / s }$ distinguish the sampled aperture from the full ramp. b, The confirmed 1TX4RX configuration enables TX0 and records four receive channels. c, One chirp per 1-ms frame yields 60,000 temporal observations over 60 s; the evaluated receive-channel representation contains 60,000 by 512 complex $\mathrm { I / Q }$ samples.

![](images/ea606bf1d004b26d8664a259b58ccacd1567cfc5c171d57c467e0030a2223786.jpg)  
Forward–backward Butterworth, order 4; Savitzky–Golay 31 / 3; Hilbert envelope 101 / 3

![](images/7bd1cc957e93525b12d64198d9f7351a3cd4a7759881162fcaf5252a7c6d748d.jpg)

c  
![](images/a62ee913539594b1368c176686d6a4ee943b7e399a73e03de34c2d30942ed8ef.jpg)  
Figure S2: Phase-derived radar heart sounds and quality-aware spectral sampling. a, Complex RX3 input, fast-time centring and range FFT, maximum-mean-magnitude target-cell selection, final-cell phase extraction with the 79-GHz conversion constant, forward–backward 16–80-Hz filtering and Savitzky–Golay smoothing, followed by outlier removal and manual quality inspection. Reconstruction uses the final selected cell. The waveform window is 31 points; the illustrated Hilbert envelope uses 101 points. b, Final reviewed intervals in the selected recording with the shortest retained exposure. Blue denotes retained intervals and gray exclusions. The first three complete 2.048-s analysis windows appear in green on separate tracks at their original positions; spectral eligibility additionally excludes the first and last second. No fragments are concatenated. ${ \mathrm { c } } ,$ Periodic Hann windows of 2,048 and 4,096 samples at 1,000 Hz, evaluated with 50% overlap. The windows specify the spectral-analysis weighting functions.