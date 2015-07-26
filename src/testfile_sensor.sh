#!/bin/bash

# Test sensor file
# Input: sensor
# Output: label of different activities

# initialize
start_time=`date +%s`
BASE='/home/sibo/Documents/Projects'

# read the first argument as file name
file=$1

# process file name
echo $file
fn=$file
fn=${fn##*/}
fn=${fn%.mp4}

# generate fisher vector
python testfile_sensor.py ${file}

# check execution time
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.
