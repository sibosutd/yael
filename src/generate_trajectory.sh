#!/bin/bash

# Generate dense trajectory features

# initialize
start_time=`date +%s`
BASE=/home/sibo/Documents/Projects
TRAJ_PATH=$BASE/multimodal_sensors/code/dense_trajectory_release_v1.2/release
REBUILD_PATH=$BASE/yael/rebuild_size_430_240_fps_15
FEATURE_PATH=$BASE/yael/feature_size_430_240_fps_15

# generate dense trajectory features
for file in $REBUILD_PATH/*.mp4; do
fn=$file
fn=${fn##*/}
fn=${fn%.mp4}

$TRAJ_PATH/DenseTrack $REBUILD_PATH/${fn}.mp4 > $FEATURE_PATH/${fn}.feature
echo ${file##*/}
done

# check execution time
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.