#!/bin/bash

# initialize
start_time=`date +%s`
BASE=/home/sibo/Documents/Projects
INPUT_PATH=$BASE/yael/sequence_20_selected
OUTPUT_PATH=$BASE/yael/rebuild_size_430_240_fps_15

# resize the data
for file in $INPUT_PATH/*.mp4; do
# ffmpeg -i $file -s 320*180 -r 10 -an $OUTPUT_PATH/${file##*/}
ffmpeg -i $file -s 430*240 -r 15 -an $OUTPUT_PATH/${file##*/}
echo ${file##*/}
done

# check execution time
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.