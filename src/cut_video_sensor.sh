#!/bin/bash

# Cut video data into 15s and select sensor data.
# Input: raw video and sensor data
# Output: 15s video and sensor data
# Usage: ./cut_video_sensor.sh 13 20150629_181213_181213_1 act16seq01

# initialize
start_time=`date +%s`
BASE=/home/sibo/Documents/Projects
INPUT_PATH=$BASE/raw_datasets/raw_multimodal
OUTPUT_PATH=$BASE/raw_datasets/raw_sequence

# read arguments
start=$1
input=$2
output=$3

# cut video data
# ffmpeg example:
# ffmpeg -i 20150629_181213_181213_1.mp4 -ss 00:00:10 -t 00:00:15 -s 320*180 -r 10 -an test.mp4
ffmpeg -ss $start -i $INPUT_PATH/$input.mp4 -c copy -t 00:00:15 $OUTPUT_PATH/$output.mp4

# cut video data
python parse_sensor.py -i $INPUT_PATH/$input.txt -o $OUTPUT_PATH/$output -s $start -t black

# check execution time
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.