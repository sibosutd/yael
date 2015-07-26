Multimodal Sensors Project Notes
=========

> Song Sibo (Email:sibo_song@mymail.sutd.edu.sg)

## List of activities

1. walking 
1. walking upstairs 
1. walking downstairs 
1. riding elevator up
1. riding elevator down
1. riding escalaator down
1. riding escalaator up
1. sitting 
1. eating 
1. drinking 
1. texting 
1. making phone calls
1. working at PC 
1. reading 
1. writing sentences
1. organizing files
1. running
1. doing push-ups 
1. doing sit-ups 
1. cycling 

## Categories of activities
### Ambulation
1. walking 
1. walking upstairs 
1. walking downstairs 
1. riding elevator up
1. riding elevator down
1. riding escalaator down
1. riding escalaator up
1. sitting 

### Daily activities
1. eating 
1. drinking 
1. texting 
1. making phone calls

### Office work
1. working at PC 
1. reading 
1. writing sentences
1. organizing files

### Exercise/fitness
1. running
1. doing push-ups 
1. doing sit-ups 
1. cycling 

## Sensor Data format
+ Brown Glass:
	+ 21 dimension: Date, Timestamp, Accelerometer1, Accelerometer2, Accelerometer3, Gravity1, Gravity2, Gravity3, Gyroscope1, Gyroscope2, Gyroscope3, LinearAcceleration1, LinearAcceleration2, LinearAcceleration3, MagneticField1, MagneticField2, MagneticField3, RotationVector1, RotationVector2, RotationVector3, RotationVector4
	+ Sampling rate: 30Hz

+ Black Glass: 
	+ 27 dimension: Date, Timestamp, Accelerometer1, Accelerometer2, Accelerometer3, Gravity1, Gravity2, Gravity3, Gyroscope1, Gyroscope2, Gyroscope3, **Light**, LinearAcceleration1, LinearAcceleration2, LinearAcceleration3, MagneticField1, MagneticField2, MagneticField3, RotationVector1, RotationVector2, RotationVector3, RotationVector4, **Latitude, Longitude, Altitude, Bearing, Speed**
	+ Sampling rate: 10Hz

## Dataset preview
![dataset preview in .gif][dataset preview in .gif]

## Pipeline
+ Video data processing: 
	+ Downsample the videos 
	+ Generate trajectory features
	+ Sample ten percent data to form the codebook
	+ Build Guassian Mixure Model (GMM)
	+ Generate fisher vector
	+ Classify fisher vectors using SVM

+ Sensor data processing:
	+ Use sliding windows to pre-process the data
	+ Sample ten percent data to form the codebook
	+ Build Guassian Mixure Model (GMM)
	+ Generate fisher vector
	+ Classify fisher vectors using SVM

## Running time
+ Trajectory Feature Extraction:
	+ type = video, size = 320x180, fps = 10: **1:30:00, (5400s)**
	+ type = video, size = 430x240, fps = 15: **2:27:53, (8873s)**
+ Fisher Vector Generation
	+ type = video, size = 320x180, fps = 10: **1:26:19**
	+ type = video, size = 430x240, fps = 15: 

## Experiment
+ Sensor data:
	+ data dimension: 19 (after removing Data, Timestamp)
	+ parameter:
		+ Window size: 10
		+ PCA dimensionality reduction: from 190 to 95
		+ Fisher Vector: 
			+ k = 25
			+ number of samples: k*100 = 2500
			+ fisher vector dimension: 2kd = 4750
		+ SVM:
			+ C = 10
			+ Cross-validation: training set(90%), test set(10%)
			+ Times: 100

+ Video data:
	+ data dimension: 426
	+ parameter:
		+ PCA dimensionality reduction: from 426 to 213
		+ Fisher Vector: 
			+ k = 25
			+ number of samples: 1 percent
			+ fisher vector dimension: 2kd = 10650
		+ SVM:
			+ C = 10
			+ Cross-validation: training set(90%), test set(10%)
			+ Times: 100

## Results

+ Sensor data
	+ Overall accuracy: 
		+ Linear SVC:  0.42
	+ Confusion matrix

+ Video data
	+ Overall accuracy: 
		+ Linear SVC:  0.7975
	+ Confusion matrix

+ All data
	+ Overall accuracy:
		+ Linear SVC:  0.792
	+ Confusion matrix

<!-- Links -->

[dataset preview in .gif]: https://http://sibosutd.github.io/yael/ "dataset preview"
