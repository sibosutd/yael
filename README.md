yael
=========

This project is for multi-modal human activity recognition by generating fisher vector for video and sensor data. 

> Note: If you have any issue running the code, please feel free to contact Sibo. (Email:sibo_song@mymail.sutd.edu.sg)

## Requirement
+ Linux environment is required.
+ Yael library
	+ http://yael.gforge.inria.fr/gettingstarted.html
+ Setup python interface of Yael 
	+ http://yael.gforge.inria.fr/python_interface.html

> Note: Need configure with `--enable-numpy` to allow some functions like fvec_to_numpy.

+ Trajectories code
	+ http://lear.inrialpes.fr/people/wang/dense_trajectories

> Note: `OpenCV` and `ffmpeg` are required to run trajectories code.

## Usage

### Test library
Please do some simple test to make sure yael library and trajectory code are installed and configured successfully.

### Set path
Before tesing, you are required to modify the `PATH` variables in `testfile_sensor`, `testfile_video` and `config.py` file.

### Test video
+ Make shell scripts executable before running these commands.
```shell
chmod +x testfile_video.sh
chmod +x testfile_sensor.sh
```
+ Test video and sensor data file
```shell
testfile_video.sh ../test_data/filename.mp4
testfile_sensor.sh ../test_data/filename.txt
```
> Note: processing video data might take a bit longer.

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