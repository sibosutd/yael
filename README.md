# yael
Generate fisher vector for video and sensor data using yael library.

> This project is for multi-modal human activity recognition

## Requirement

+ Yael library
 + http://yael.gforge.inria.fr/gettingstarted.html
  + Setup python interface of Yael 
   + http://yael.gforge.inria.fr/python_interface.html

> Note: Need configure with `--enable-numpy` to allow some functions like fvec_to_numpy.

+ Trajectories code
 + http://lear.inrialpes.fr/people/wang/dense_trajectories

## Usage
```shell
testfile_video.sh filename
testfile_sensor.sh filename
```

## Output
The program will output a label which indicates an activity.
### Label
1. walking 
2. walking upstairs 
3. walking downstairs 
4. eating 
5. drinking 
6. push-ups 
7. runing in the gym 
8. working at PC 
9. reading 
10. sitting  

> Note: Not finished.

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

## Results
+ TBA

> Singapore University of Technology and Design (SUTD)