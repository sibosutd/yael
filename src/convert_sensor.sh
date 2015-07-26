#!/bin/bash

# Convert 30Hz sensor data(red glass) to 10Hz data(black glass).
# Input: 30Hz sensor data
# Output: 10Hz sensor data

# initialize
start_time=`date +%s`
BASE=/home/sibo/Documents/Projects
INPUT_PATH=$BASE/multimodal_sensors/sensor_data/data_txt
OUTPUT_PATH=$BASE/yael/data_black

for file in $INPUT_PATH/*.txt; do
	# process file name
	# echo $file
	fn=$file
	fn=${fn##*/}
	fn=${fn%.txt}
	# generate fisher vector
	python parse_sensor.py -i ${file} -o $OUTPUT_PATH/${fn} -t red
done
# check execution time
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.