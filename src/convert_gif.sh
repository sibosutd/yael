#!/bin/bash

# Convert .mp4 video data into gif 
# Input: rebuild video data
# Output: gif
# Usage: 

# initialize
start_time=`date +%s`
BASE=/home/sibo/Documents/Projects
INPUT_PATH=$BASE/yael/rebuild
OUTPUT_PATH=$BASE/yael/gif

for file in $INPUT_PATH/*.mp4; do
	# process file name
	echo $file
	fn=$file
	fn=${fn##*/}
	fn=${fn%.mp4}
	# generate gif
	ffmpeg -ss 0 -t 3 -i ${file} -vf fps=5,scale=150:-1:flags=lanczos,palettegen $OUTPUT_PATH/${fn}palette.png
	ffmpeg -ss 0 -t 3 -i ${file} -i $OUTPUT_PATH/${fn}palette.png \
	-filter_complex "fps=5,scale=150:-1:flags=lanczos[x];[x][1:v]paletteuse" $OUTPUT_PATH/${fn}.gif
done
# ffmpeg -ss $start -i $INPUT_PATH/$input.mp4 -c copy -t 00:00:15 $OUTPUT_PATH/$output.mp4

# check execution time
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.