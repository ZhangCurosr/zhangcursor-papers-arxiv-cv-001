# Design and Implementation of a Kalman Filter-Infused Algorithm for Tilt Estimation

Yuehan Ma   
Independent Researcher   
yuehanma.com   
lucasmahjkl@gmail.com

Hongji Dai Independent Research terback@gmail.com

## Abstract

Accurate tilt angle estimation is important in many engineering applications, such as robotics, motion tracking, and embedded control systems. However, measurements from low-cost inertial sensors are often degraded by noise and drift. This paper presents a single-axis tilt angle estimation system based on the MPU6050 inertial measurement unit, implemented on an RP2040 microcontroller platform, with sensor fusion achieved through a Kalman filter.

The accelerometer provides a direct estimate of tilt angle from gravity but is sensitive to noise and short-term fluctuations. The gyroscope provides smooth angular rate measurements, but integration over time introduces drift. To overcome these limitations, a Kalman filter is used to combine measurements from both sensors, leveraging the long-term stability of the accelerometer and the short-term smoothness of the gyroscope.

Both simulation and hardware experiments are performed. In simulation, sensor noise and drift are modeled to evaluate the filter performance under control conditions. In the hardware implementation, real-time MPU6050 data is acquired and processed by the RP2040 platform, and the estimated tilt angle is compared with accelerometer-only and gyroscope-only outputs. The results show that the proposed method effectively reduces noise measurements and suppresses long-term drift while preserving good dynamic response.

Overall, the system provides more stable and accurate tilt estimation than either sensor alone, demonstrating a practical and accessible approach for Kalman filter based sensor fusion in embedded application. This manuscript is a preprint version of the work.

## 1. Introduction

Accurate tilt angle estimation plays a critical role in a wide range of engineering applications, including robotics, motion tracking, and embedded control systems. In these systems, reliable orientation information is essential for tasks such as stabilization, navigation, and human–machine interaction. With the increasing availability of low-cost inertial measurement units (IMUs), it has become feasible to implement tilt estimation in compact and affordable embedded platforms. However, achieving high accuracy using such sensors remains a significant challenge.

Low-cost IMUs, such as the MPU6050 sensor module, typically combine an accelerometer and a gyroscope to provide motion and orientation-related measurements. Each of these sensing modalities has inherent limitations. The accelerometer can estimate tilt angle based on the direction of gravity, but it is highly sensitive to noise and external vibrations, resulting in significant short-term fluctuations. On the other hand, the gyroscope provides smooth angular velocity measurements, but estimating angle through time integration introduces cumulative errors, leading to long-term drift. As a result, using either sensor independently does not provide sufficiently accurate or stable tilt estimation.

To overcome these limitations, sensor fusion techniques are commonly employed to combine the complementary characteristics of accelerometers and gyroscopes. Among these techniques, the Kalman Filter is widely used due to its ability to optimally estimate system states in the presence of noise and uncertainty. By integrating prediction from the gyroscope with correction from the accelerometer, the Kalman Filter can achieve a balance between short-term smoothness and long-term stability, making it particularly suitable for embedded tilt estimation systems.

In this work, a tilt angle estimation system based on the MPU6050 IMU is developed and implemented on an

RP2040-based microcontroller platform, specifically the STEPico RP2040 development board. A real-time Kalman Filter is designed to fuse accelerometer and gyroscope measurements, enabling improved accuracy and stability in tilt estimation. The system is designed with a focus on accessibility and practicality, making it suitable for educational and low-cost embedded applications.

This work makes the following contributions:

Design and implementation of a low-cost tilt angle estimation system using the MPU6050 IMU on an RP2040 microcontroller platform.

Development of a real-time Kalman Filter-based sensor fusion algorithm for embedded systems.

Comprehensive quantitative evaluation of the system, including noise analysis, drift analysis, and accuracy assessment under controlled conditions.

● A qualitative user-level evaluation demonstrating the impact of improved estimation on interaction smoothness in an embedded display application.

## 2. System Design

## 2.1 Hardware Architecture

The tilt estimation system is built using a low-cost inertial sensing module, the MPU6050 sensor module, and an RP2040-based microcontroller platform, specifically the STEPico RP2040 development board. The MPU6050 integrates a 3-axis accelerometer and a 3-axis gyroscope, providing raw motion and orientation-related data required for tilt estimation.

The MPU6050 communicates with the microcontroller via the Inter-Integrated Circuit (I2C) interface. The microcontroller periodically acquires accelerometer and gyroscope measurements at a fixed sampling rate. In this implementation, a sampling frequency of 100 Hz is used to balance computational load and estimation accuracy.

The sensor module is mounted on a rigid platform to ensure consistent orientation during experiments. This reduces unintended disturbances and improves measurement repeatability. The microcontroller processes the incoming sensor data in real time and outputs the estimated tilt angle through a serial interface for logging and analysis. In addition, the system can optionally interface with a small display module for real-time visualization in interactive applications.

![](images/33352e281cc4f9ac5687f74e77e2421c8e73d8fb2d649be1d7bf0d6965388f81.jpg)  
Fig I: Data acquisition and angle-estimation workflow between the RP2040 Stepico microcontroller and MPU6050 sensor.

## 2.2 Software Architecture

The software system is organized as a real-time processing pipeline that converts raw sensor measurements into a stable tilt angle estimate. The overall data flow is shown conceptually as:

![](images/027aef94305176c02840d196a3ab8cfb75360f1984909b03da274dfe2e8e88c0.jpg)  
Figure II. Block diagram of the Kalman filter–based angle estimation process

At each sampling step, raw accelerometer and gyroscope data are first read from the MPU6050. These

measurements are then processed to obtain two intermediate angle estimates:

● An accelerometer-based angle derived from the gravity vector

● A gyroscope-based angle obtained through angular rate integration

These intermediate estimates are then fused using a Kalman Filter, which produces the final tilt angle output. The resulting angle is transmitted via serial

$$
\theta _ { a c c } = a r c t a n ( \frac { a _ { y } } { a _ { z } } )\tag{[1]}
$$

communication for data logging and can also be used to drive real-time applications such as display-based interaction.

## 2.3 Data Processing Pipeline

The processing pipeline consists of four main stages:

## 1. Sensor Data Acquisition

Raw data from the accelerometer and gyroscope are retrieved via I2C. The measurements are converted into physical units and timestamped to ensure accurate time integration.

## 2. Angle Computation

The accelerometer data are used to compute a tilt angle based on the direction of gravity. The gyroscope data are integrated over time to obtain a second estimate of the tilt angle.

## 3. Kalman Filter-Based Sensor Fusion

A Kalman Filter is applied to combine the accelerometer and gyroscope estimates. The gyroscope data are used for prediction, while the accelerometer data are used for correction. This process reduces noise and compensates for drift.

## 4. Output and Logging

The filtered tilt angle is output through a serial interface and recorded in CSV format for further analysis. This data is later used for quantitative evaluation, including noise, drift, and accuracy analysis. The resulting data will be plotted using R for a more visual demonstration of the experiments

## 2.4 Implementation Details

The system is implemented using an embedded programming environment compatible with the RP2040 platform. Real-time constraints are considered to ensure that all computations, including sensor reading and Kalman filtering, are completed within each sampling interval.

A consistent sampling interval is maintained using a timing control mechanism, and the time step (dt) is calculated dynamically to improve integration accuracy. Basic sensor calibration is performed to reduce bias, particularly for the gyroscope, which is critical for minimizing drift.

## 2.5 Reproducibility

To facilitate reproducibility and further development, the full hardware configuration and source code are publicly available. Refer to Appendix.

## 3. Methodology

## 3.1 Accelerometer-Based Angle Estimation

The accelerometer measures the projection of gravitational acceleration along its sensing axes. When the sensor is stationary or moving at a constant velocity, the measured acceleration can be used to estimate the tilt angle relative to gravity.

For a single-axis tilt estimation, the angle can be computed using the ratio of two acceleration components. For example, the tilt angle can be expressed as:

Where $a _ { _ y }$ and $a _ { _ { z } }$ are the accelerometer readings along the corresponding axes.

Although this method provides an absolute reference based on gravity, it is highly sensitive to noise and external disturbances, resulting in significant short-term fluctuations in the estimated angle.

## 3.2 Gyroscope-Based Angle Estimation

The gyroscope measures angular velocity, which can be integrated over time to estimate the tilt angle. The angle at time step � can be computed as:

$$
\Theta _ { g y r o } ( t ) = \Theta _ { g y r o } ( t - 1 ) + \mathbf { \omega } \omega \mathbf { \cdot } d t\tag{[2]}
$$

where ω is the angular velocity measured by the gyroscope, and �� is the time interval between successive measurements.

This approach provides smooth and responsive angle estimation in the short term. However, due to sensor bias and noise, the integration process accumulates errors over time, leading to drift in the estimated angle.

## 3.3 Kalman Filter for Sensor Fusion

To address the limitations of individual sensors, a Kalman Filter is used to combine accelerometer and gyroscope measurements. The filter estimates the true tilt angle by recursively updating its state based on prediction and correction steps.

## 3.3.1 Prediction Step

In the prediction step, the gyroscope measurement is used to estimate the current angle based on the previous state:

$$
\theta _ { k } ^ { - } = \theta _ { k - 1 } ^ { - } + \ \omega \cdot d t\tag{[3]}
$$

This prediction leverages the short-term smoothness of the gyroscope data.

## 3.3.2 Update Step

In the update step, the accelerometer-based angle is used to correct the predicted value:

$$
\ u _ { \theta _ { k } } ^ { - } = \ u _ { \theta _ { k - 1 } } ^ { - } + K ( \ u _ { \theta _ { a c c } } - \ u _ { \theta _ { k } } )\tag{[4]}
$$

where � is the Kalman gain, which determines the relative weighting between the prediction and the measurement.

The Kalman gain is dynamically adjusted based on the estimated uncertainty of the system, allowing the filter to rely more on the gyroscope during rapid motion and more on the accelerometer during stable conditions.

## 3.4 Filter Characteristics

The Kalman Filter effectively combines the complementary strengths of the two sensors:

The gyroscope provides smooth short-term estimation but suffers from drift

● The accelerometer provides long-term reference but is noisy

By integrating both sources, the filter produces an angle estimate that is both stable and responsive.

In practical implementation, the filter parameters are tuned to balance noise reduction and responsiveness.

Proper tuning is essential to achieve optimal performance under different motion conditions.

3.5 Implementation Considerations To ensure accurate estimation in an embedded environment, several practical considerations are addressed:

Time step accuracy: The time interval �� is measured dynamically to improve integration precision

Sensor calibration: Initial bias in the gyroscope is estimated and compensated

Numerical stability: The filter is implemented using a simplified model suitable for real-time execution on a microcontroller

Computational efficiency: The algorithm is optimized to run within the fixed sampling interval without introducing latency

## 4. Experimental Setup

The following 5 experiments will be used to test our Kalman filter methodology, by putting the filter in both static and kinetic experiments alongside the other two signals: accelerometer and gyroscope. Data will be collected over a short term period (10-30 seconds) or a long term span(120 seconds). These data will then be analyzed to determine whether the Kalman filter-infused algorithm is able to generate a smoother tracking by combining the strengths of the accelerometer and gyroscope signal.

## 4.1 Experiment 1 Setup

![](images/0dc14aa3518a20558f61dff9295dae19942d2f2dc61593a1404cf87423936def.jpg)  
Fig III: The hardware setup ofExperiment I on a breadboard.  
4.1.1 Setup description

A Raspberry Pi RP2040 connected to a MPU6050 sensor on a standard breadboard. The RP2040 is then connected to Thonny in which the code will be run.

## 4.1.2 Experiment description

The purpose of this experiment is to evaluate the short-term noise characteristics of the MPU6050 sensor under stationary conditions and to compare the stability of different angle estimation methods in the sensor.

The sensor is placed on a flat surface and kept completely still during the experiment. Data are continuously recorded over a fixed duration, including:

Accelerometer-based angle estimation $( a n g l e _ { a c c } )$

Gyroscope-integrated angle estimation $( a n g l e _ { g y r o } )$

Kalman filter fused angle $( a n g l e _ { _ { k a l m a n } } )$

By analyzing the extent of variation of these three measurements, this experiment aims to identify the effectiveness of a Kalman Filter-infused algorithm as a means to reduce noise from the individual components of the accelerometer and the gyroscope.

## 4.2 Experiment 2 Setup

## 4.2.1 Setup description

Experiment 2 will follow the same setup as Experiment 1. For setup diagrams and instructions refer to section 4.1.1 and Fig III.

## 4.1.2 Experiment description

The purpose of this experiment is to evaluate the long-term characteristics of the MPU6050 sensor under stationary conditions and to investigate the limitations of regular angle estimation methodologies over extended periods.

The rest of the experiment is kept constant as experiment 1. Refer to section 4.1.2

## 4.3 Experiment 3 Setup

To facilitate moving the sensor at an incline, the breadboard is attached onto a digital protractor angle finder. The protractor would be opened at different angles for the sensor to collect data upon. The protractor has extreme precision and lays a strong foundation for collecting data at an angle.

![](images/6ed75d94b789a845efed918732b4e4f56361188c84524e1d710fa3a420351039.jpg)  
Fig IV: The setup of Experiment 3 with protractor-determined angle at 30.0°

![](images/c19231a069b6bfa919399fe550801fb391152870d5f7e8aefcf1c62f67b59dff.jpg)  
Fig V: The setup ofExperiment 3 with protractor-determined angle at $\it 6 0 . 0 ^ { \circ }$

## 4.3.1 Setup description

Experiment 3 will follow the same hardware setup as Experiments 1 and 2. For setup information refer to section 4.1.1. This experiment will also feature a digital protractor angle finder, which the breadboard is tied to.

## 4.3.2 Experiment description

The purpose of this experiment is to evaluate the accuracy of the estimation system under known, fixed-angle conditions and to compare the performance of different estimation methods.

The MPU6050 sensor is placed at several predefined angles using external references using a leveling tool. At each angle, the sensor is kept completely stationary, and data is recorded over a short duration. The same signals are measured:

Accelerometer-based angle estimation $( a n g l e _ { a c c } )$

Gyroscope-integrated angle estimation $( a n g l e _ { g y r o } )$

Kalman filter fused angle $( a n g l e _ { _ { k a l m a n } } )$

For this experiment, we will use $a n g l e _ { a c c }$ and $a n g l e _ { _ { k a l m a n } }$ as viable data, as a gyroscope finds difficulty in measuring angles at an incline. At each angle measured, an average is taken to determine the measured data vs. the assumed ideal value. This seeks to compare the accuracy of the accelerometer data as opposed to the data given by the Kalman algorithm.

This experiment seeks to determine the accuracy of the different measuring methods without a completely flat surface, which makes the measurement prone to extra noise and destabilization in the data obtained.

## 4.4 Experiment 4 Setup

This experiment requires the sudden movement of the sensor between different angle measures in order to measure the speed of adjustment and its stability. Therefore, 3D printed angles are made as a guide for where to move the digital protractor angle finder, leading to the setup as shown below.

![](images/bd2cd93768ff53580a664f8cbd22bcd32e0aa4038d5ecbd75242f7ec9391bb7c.jpg)  
Fig. VI: Hardware setup and external tools for Experiment 4.

## 4.4.1 Setup description

The same hardware setup is kept. For setup information refer to section 4.1.1. This experiment will also feature 3d printed angle measures that will be used as a reference to adjust the angle in the experiment.

## 4.4.2 Experiment description

This experiment introduces a set of fixed angle supports. These supports allowed the sensor to be quickly repositioned between predefined angles $( 0 ^ { \circ } , 3 0 ^ { \circ } , 4 5 ^ { \circ }$ and 60°), enabling near instantaneous transitions followed by stable holding periods.

This setup reduces uncertainties caused by slow manual adjustment and ensures more well defined steps like changes in orientation. Consequently, it allows for clearer observation of system response, including

responsiveness, stability, and noise characteristics of the accelerometer, gyroscope, and Kalman filter outputs.

The recorded signals during this experiment includes:

Accelerometer based angle estimation $( a n g l e _ { a c c } )$

Gyroscope integrated angle estimation $( a n g l e _ { g y r o } )$

Kalman filter fused angle $( a n g l e _ { _ { k a l m a n } } )$

The purpose of this experiment is to test the algorithm’s ability to adjust under rapid and sudden movements, while attempting to filter out introduced noises due to inconsistencies in the moving process.

## 4.5 Experiment 5 Setup

This experiment will differ from the setups of Experiments 1-4 as it requires data collection during a rolling motion, therefore the microcontroller cannot be connected to the computer via a wire. For this experiment, we will use a MEGO portable battery as our power source, paired with a green LED to determine when data collection has begun. The RP2040 will automatically run the program once the battery is turned on.

![](images/f1466fd2a46f9ae20b84ef4c75e04f117704e1bcf0e05037e1ca2094bcdbcf20.jpg)  
Fig. VII: New portable supply and hardware setup for Experiment 5.

![](images/cc0eda99baac885aa88601b9b1e79377120699bb153909db9cfdee78f89e7fc5.jpg)  
Fig. VIII: The circuit system on the wheel which will be rotated to gather rotational data.

## 4.5.1 Setup description

A Raspberry Pi RP2040 connected to a MPU6050 sensor on a shortened breadboard, using a MEGO Portable Battery Supply. A Green LED connected to the to indicate when data has started to collect. The entire contraption is then glued to a 3d printer filament holder in order to conduct smooth circular motion.

## 4.5.2 Experiment description

This experiment seeks to test the stability of the Kalman Filter-infused algorithm under circular periodic motion. The same three signals: $a n g l e _ { a c c } , a n g l e _ { g y r o }$ , and $a n g l e _ { _ { k a l m a n } }$ are measured, and we will examine whether the signals provide a smooth, repeatable, and periodic tracking.

Due to the limited space available on the half breadboard, the arrangement of the portable power supply, microcontroller module, and MPU6050 sensor had to be optimized carefully. As a result, the MPU6050 communication pins were fixed to the following GPIO configuration:

![](images/9bb8b084a26afa82c1aa192d5b2303aaf52b4326715f5443643c39f9e59d136e.jpg)  
Fig. IX: The GPIO configuration of the MPU 6050 to the RP2040 STEPico in Experiment 5

A green LED indicator was introduced to provide visual feedback during data acquisition. The LED turns on only while the system is actively recording and saving data. When the LED is off, no data recording is taking place. This feature helps ensure clear synchronization between experimental motion and the data collection period during rotational testing.

## 5. Quantitative Results and Analysis

## 5.1 Experiment 1

## 5.1.1 Experiment 1 Graphical Results

The following graphs (Figures X - XII) illustrate the short-term fluctuations of the 3 signals $a n g l e _ { a c c } ,$ $a n g l e _ { g y r o }$ , and $a n g l e _ { _ { k a l m a n } }$ across 3 repeated trials in the same environment. The experiments were conducted back to back to minimize any bias caused by non-human factors.

![](images/94ec53059a861b80c405c51fec55419547be9ac82ae9396b2527fe8a4311c333.jpg)  
Figure X: Experiment 1 Trial 1 data

![](images/9f59cf745f96bab204b3733c550c64432cae45d4e124bcad511e1fb0bc75e09e.jpg)  
Figure XI: Experiment 1 Trial 2 data

![](images/6970de0ff7d5dc0cd455314d2297041e567ffbdf287c82e9a0714067b232b5ea.jpg)  
Figure XII: Experiment 1 Trial 3 data

## 5.1.2 Experiment 1 Result Analysis

## (1) Trial 1 Result Analysis

The experimental results show that the accelerometer-based angle exhibits noticeable fluctuations, with a standard deviation of 0.166°, indicating significant measurement noise. The gyroscope-based angle is relatively smooth but still shows variability, with a standard deviation of 0.181°.

After the Kalman filter adjusted to the true value\*, the remaining data reached a standard deviation of 0.077°, demonstrating highly improved stability. This confirms that the Kalman filter effectively reduces noise while maintaining a consistent angle estimate.

\* The standard deviation is calculated using the data points that are not considered an outlier by the standard statistics methodology, where an outlier is considered as a point that is less than Q1-1.5(IQR) or more than Q3+1.5(IQR).

## (2) Trial 2 Result Analysis

For trial 2, similar results are shown. The standard deviation of the accelerometer and the gyroscope was recorded to be 0.168° and 0.100°, respectively. While the Kalman-Filter infused algorithm recorded a standard deviation of 0.066° after its initial adjustments towards the true value.

The gyroscope drifted upwards overtime instead of downwards in Experiment 1.1, giving the Kalman-filter infused algorithm a slightly smaller recovery distance to find the true value. The standard deviation of the gyroscope is also recorded to be much lower than Trial 1, which the accelerometer stays constant throughout.

## (3) Trial 3 Result Analysis

For trial 3, the stability of the Kalman-filter infused algorithm is again showcased; whilst the accelerometer and gyroscope signals received a standard deviation of 0.241° and 0.127°. The Kalman algorithm recorded a

0.083°, which again shows a stability much greater than both other signals.

For this trial specifically, it is well noted that the accelerometer suffered a significant spike at around 14 seconds, which is most likely due to a non-human factor. It can be noted that the Kalman filter algorithm fluctuated very little under that influence.

## 5.2 Experiment 2

## 5.2.1 Experiment 1 Graphical Results

The following graphs (Figures XIII - XV) illustrate the long-term fluctuations of the 3 signals $a n g l e _ { a c c } \mathrm { ~ }$ $a n g l e _ { g y r o }$ , and $a n g l e _ { _ { k a l m a n } }$ across 3 repeated trials in the same environment. The experiments were conducted back to back to minimize any bias caused by non-human factors.

![](images/be8865d46b531487f5c486e0521ee44447a389628a7295c2d2503f8b4b0c94c5.jpg)  
Figure XIII: Experiment 2 Trial 1 data

![](images/77f6f3be2857cca783af1e84500ab7bbd8a66616070cc027ee4116c24cc03e9a.jpg)  
Figure XIV: Experiment 2 Trial 2 data

![](images/a87714e4dccaff4fd6b6ac4f01456dbed0d2352ac3fc2c1210cfc667a7d40ba9.jpg)  
Figure XV: Experiment 2 Trial 3 data

## 5.2.2 Experiment 2 Result Analysis

## (1) Trial 1 Result Analysis

This 2 minute long-term tracking of the three signals has shown a trend similar to trial 1: While the accelerometer and gyroscope had a standard deviation of 0.171° and 0.194°, the Kalman algorithm again showed its long term tracking prowess by recording a standard deviation of 0.065° after initial adjustment.

Another point worth noting is that the long term drift of the gyroscope became quite lucid in this experiment; it measurement drifts in undesired directions and received a standard deviation higher than angle $a n g l e _ { a c c }$ for the first time. This proves that the gyroscope is generally troublesome during long term tracking as compared to the accelerometer.

## (2) Trial 2 Result Analysis

This trial presents a similar result as trial 1. With the accelerometer and gyroscope posting a standard deviation of 0.166° and 0.375°, while the Kalman-filter infused algorithm giving a standard deviation of 0.062°. The gyroscope’s long-term drift becomes even more evident as its standard deviation is recorded to be more than double of the result in 6.2.1.

## (3) Trial 3 Result Analysis

Trial 3 poses similar results as trial 1 where the gyroscope drifts in seemingly random patterns. In this trial, despite the fluctuation of the gyroscope being significantly lowered to a standard deviation of 0.082°, it did not hinder the Kalman algorithm’s performance at 0.062° as compared to 0.178° of the accelerometer.

## 5.3 Experiment 3

## 5.3 Experiment 3 Graphical Results

The following graphs (Figures XVI-XVII) are used to demonstrate the extent of deviations from the actual value of the accelerometer and Kalman filter-infused algorithm in set, non-zero angle conditions. Note that the gyroscope data will not be used for this experiment as a gyroscope cannot measure data if the sensor is at an incline.

Theoretical vs. Actual Angle Measurement  
![](images/f54b6188de36a49713882db28abc033bf0da67cbbc8296d158f8289f26ca23cf.jpg)  
Figure XVI: Experiment 3 Trial 1 data

Theoretical vs. Actual Angle Measurement  
![](images/aa43d18e0d7c6c8a25988dfaa3e3432c7ae62732e40f73a58fecb784b567069d.jpg)  
Figure XVII: Experiment 3 Trial 2 data

Theoretical vs. Actual Angle Measurement  
![](images/6c82428f3f402c17db9f296f24a251bf57f85e8faefab2739f4ea2286c28b12a.jpg)  
Figure XVIII: Experiment 3 Trial 3 data

## 5.3.2 Experiment 3 Result Analysis

Experiment 3 presents data that are very consistent across the three trials, most likely due to the methodology of taking the average across 10 seconds for multiple angles. Again, due to the Kalman Filter’s initial calibration, the outliers in the data set are again discarded.

However, in this experiment, the Kalman data drifts further away from the actual data as the angle increases. This can be explained by a process of overcalibration at each individual angle, which is not accounted for while outliers were discarded. For the graphs at each angle, refer to Appendix.

## 5.4 Experiment 4

## 5.4 Experiment 4 Graphical Results

The following graphs (Figures XIX-XXI) will attempt to determine the stability of the three signals under abrupt movements of angles to 30 , , and . 3 trials will be45 60 conducted and the same pre-experiment conditions and environment are kept and assumed.

![](images/0920cb75d5b7089b96612bc0e0b018bbd9905c71f097df168f9fe4d84c8e00c3.jpg)  
Figure XIX: Experiment 4 Trial 1 data

![](images/26f71ed7d5e0a2aaeb07f35409f50ddbb99ea72d8bb13f3875659ea9f1f71f94.jpg)  
Figure XX: Experiment 4 Trial 2 data

![](images/90fe6e51f92fa50cc1439a4335ec3f03731048229332086dbc65d049443c7ce9.jpg)  
Figure XXI: Experiment 4 Trial 3 data

## 6.4.2 Experiment 4 Result Analysis

In this experiment, The Kalman algorithm demonstrated its ability to adjust to the actual value through rounding off noises. While the accelerometer and gyroscope data contains significant spikes that fluctuate between the actual value and those around it, the Kalman algorithm again presents a smooth tracking that is more accurate and less prone to errors in measurement.

Despite the smooth tracking of the Kalman filter, it is observed that at every sudden jolt the Kalman’s line needs time to recover to the actual data. This delay may cause minor concerns, however once the calibration is finished, the signal steadies and presents a smooth line.

## 5.5 Experiment 5

## 5.5.1 Experiment 5 Graphical Results

The following graphs (Figures XIII - XV) demonstrate the changes in the angle measured by the three sensors during periodic motion. 3 trials will be conducted and the same pre-experiment conditions and environment are kept and assumed.

Again, the gyroscope would provide little use here as it only measures rotation only along the axis parallel to the wheel’s axle. Consequently, examining a different gyroscope axis, particularly the axis perpendicular to the sensor board, produced little or no change.

![](images/2525271c8c63c1497c648e7de0a4d07d6198e045b19f2fdf75046f11b26264e1.jpg)

Figure XXII: Experiment 5 Trial 1 data  
![](images/eb36348b4fe6f900fae4431b9d04289d11dc326d814ad82e8eca23ece3032e98.jpg)

Figure XXIII: Experiment 5 Trial 2 data  
![](images/dc60d16aca8aeb979b5b13455772d08134089fdf66bfb7c76e5b83381a92301b.jpg)  
Figure XXIV: Experiment 5 Trial 3 data

## 6.5 Experiment 5 Result Analysis

In each experiment, the gyroscope was not able to recognize circular motion, and therefore presented data that was flawed and did not reflect the true value.

The accelerometer’s data resembles a sine wave, however there are many instances of spikes that may be caused by measurement noise. However the Kalman filter was able to smooth out the graph into almost a perfect sinusoidal curve. This again demonstrated that the Kalman algorithm is able to maintain smooth tracking even during rotations.

However, the source of noise is currently unknown; the noise could also be caused by the human’s unintentional movement of the hand or an imperfect rolling process. This may cause problems if intentional microadjustments are made, but was omitted by the Kalman system as measurement noise.

## 6. Discussion

Through the 5 experiments, the Kalman filter-infused algorithm for object tracking produced a smooth result that allowed the noise during the tracking process to be massively reduced. The purpose of combining the strengths of the accelerometer and gyroscope data has been successfully achieved, generating results that are significantly better than the two original signals.

A potential improvement to this study is that the angle estimation is limited to a single axis, assuming motion within a planar configuration. The tilt angle is computed using accelerometer measurements based on a two-dimensional projection, and the Kalman filter is applied to fuse accelerometer and gyroscope data along this axis.

This approach simplifies the problem and allows clear observation of noise reduction and drift suppression effects. However, it does not account for full three-dimensional orientation, where cross-axis coupling and more complex motion dynamics may affect the estimation accuracy.

Future work may include extending this approach to full 3D orientation estimation using multi-axis sensor fusion methods, such as quaternion-based filters or complementary filters. This allows us to discover a new dimension in which the Kalman Filter-infused algorithm’s utility is testified.

The MPU6050 must be attached to the breadboard, and while our experiments look to adjust the breadboard’s angle, we cannot be sure that the sensor is perfectly parallel with the breadboard, which contributes to small errors in data generation.

For experiment 5 specifically, future work may include powering the mechanism with a motor with a set RPM instead of pushing it manually. Despite the fact that the filament holder is extremely smooth, human errors during pushing may contribute to potential errors in the signals received.

Also, the weight was not balanced on the wheel during the experiment. The MEGO portable power supply weighs 1 side of the wheel down significantly, contributing to a flawed sinusoid.

## 7. Conclusion

This study successfully demonstrated the utility of a Kalman-Filter infused algorithm that effectiveness to reduce measurement noise. The components of a low-cost MPU6050 sensor includes an accelerometer and a gyroscope, which both demonstrate their shortcomings. The accelerometer is prone to short-term fluctuations due to its high sensitivity towards noise and measurement errors; on the other hand, the gyroscope faces the challenge of cumulative error, which causes long-term drift.

The incorporation of a Kalman-Filter algorithm that combines the advantages of both signals was used in 5 separate experiments to test its performance under both static and kinetic conditions. Overall, the algorithm illustrates a much smoother tracking and was far less influenced by noise.

These results demonstrate the value of a Kalman filtering algorithm in real-world motion-tracking systems, where measurements are rarely perfect and multiple noisy data sources must be combined in order to piece a more holistic picture of the entire object tracking mechanism.

## Appendix

Raspberry Pi STEPico Pinout Map  
![](images/6e7b044c68c7613d43c31e9ded2047e030bba8bcad0d02349953c0d3fb2966f5.jpg)

MPU6050 Pinout Map  
![](images/c24c9129dd0efd69f811efbb777a883a62de79f0ea259ecda66f693a32a179fc.jpg)

Hardware Configuration Pin Mapping for Experiment  
1-4
<table><tr><td rowspan=1 colspan=1>Function</td><td rowspan=1 colspan=1>RP2040 Pin</td><td rowspan=1 colspan=1>MPU6050 Pin</td></tr><tr><td rowspan=1 colspan=1>I2C SDA</td><td rowspan=1 colspan=1>GP</td><td rowspan=1 colspan=1>SDA</td></tr><tr><td rowspan=1 colspan=1>I2C SCL</td><td rowspan=1 colspan=1>GP</td><td rowspan=1 colspan=1>SCL</td></tr><tr><td rowspan=1 colspan=1>Power</td><td rowspan=1 colspan=1>Vsys</td><td rowspan=1 colspan=1>Vcc</td></tr><tr><td rowspan=1 colspan=1>Ground</td><td rowspan=1 colspan=1>Gnd</td><td rowspan=1 colspan=1>Gnd</td></tr></table>

Hardware Configuration Pin Mapping for Experiment 5
<table><tr><td rowspan=1 colspan=1>Function</td><td rowspan=1 colspan=1>RP2040 Pin</td><td rowspan=1 colspan=1>MPU6050 Pin</td></tr><tr><td rowspan=1 colspan=1>I2C SDA</td><td rowspan=1 colspan=1>GP</td><td rowspan=1 colspan=1>SDA</td></tr><tr><td rowspan=1 colspan=1>I2C SCL</td><td rowspan=1 colspan=1>GP</td><td rowspan=1 colspan=1>SCL</td></tr><tr><td rowspan=1 colspan=1>Power</td><td rowspan=1 colspan=1>Vsys</td><td rowspan=1 colspan=1>Vcc</td></tr><tr><td rowspan=1 colspan=1>Ground</td><td rowspan=1 colspan=1>Gnd</td><td rowspan=1 colspan=1>Gnd</td></tr><tr><td rowspan=1 colspan=1>Data SavingIndicator</td><td rowspan=1 colspan=1>GP18</td><td rowspan=1 colspan=1>N/A</td></tr></table>

Software Infusion Guide   
For installation guides and software codes (Thonny, R   
plots, etc), refer to GitHub Repository:   
https://github.com/KingofSaltyFish/Kalman-Filter-Resea   
rch

## Reference

[1] Bashir, M. A., Malik, F. M., Akbar, Z. A., & Uzair, M. (2020). Kalman filter based sensor fusion for altitude estimation of aerial vehicle. IOP Conference Series: Materials Science and Engineering, 853, 012034. https://doi.org/10.1088/1757-899X/853/1/012034

[2] Malik, N. N., & Hyder, M. J. (2019). Analytical review on the techniques to improve the performance oftilt measurement by MPU 6050 [Manuscript].

[3] Zhang, T., & Liao, Y. (2017). Attitude measure system based on extended Kalman filter for multi-rotors. Computers and Electronics in Agriculture, 134, 19–26. https://doi.org/10.1016/j.compag.2016.12.021

[4] Bashir, M. A., Malik, F. M., Akbar, Z. A., & Uzair, M. (2020). Kalman filter based sensor fusion for altitude estimation of aerial vehicle. IOP Conference Series: Materials Science and Engineering, 853, 012034. https://doi.org/10.1088/1757-899X/853/1/012034

[5] Ikhsan Alfian, R., Ma’arif, A., & Sunardi, S. (2021). Noise Reduction in the Accelerometer and Gyroscope Sensor with the Kalman Filter Algorithm. Journal ofRobotics and Control (JRC), 2(3), 180–189. https://doi.org/10.18196/jrc.2375

[6] Raspberry Pi Ltd. (2025). RP2040 datasheet: A microcontroller by Raspberry Pi. https://pip.raspberrypi.com/documents/RP-008371-DS

[7] InvenSense, Inc. (2013). MPU-6000 and MPU-6050 product specification (Rev. 3.4; Document No. PS-MPU-6000A-00). https://www.es.co.th/Schemetic/PDF/MPU6000.PDF