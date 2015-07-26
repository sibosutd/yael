#!/bin/bash

# Test video file
# Input: video
# Output: label of different activities

# initialize
start_time=`date +%s`
BASE=/home/sibo/Documents/Projects
TRAJ_PATH=$BASE/multimodal_sensors/code/dense_trajectory_release_v1.2/release
REBUILD_PATH=$BASE/yael/rebuild
FEATURE_PATH=$BASE/yael/feature

# read the first argument as file name
file=$1

# process file name
echo $file
fn=$file
fn=${fn##*/}
fn=${fn%.mp4}

# downsample the raw video into 430*240 dimension
ffmpeg -i ${file} -r 25 -s 430*240 -an $REBUILD_PATH/${fn}.mp4
# ffmpeg -i ${file} -r 25 -s 100*50 -an $REBUILD_PATH/${fn}.mp4

# generate dense trajectory features
$TRAJ_PATH/DenseTrack $REBUILD_PATH/${fn}.mp4 > $FEATURE_PATH/${fn}.feature

# generate fisher vector
python testfile_video.py $FEATURE_PATH/${fn}.feature

# check execution time
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.
