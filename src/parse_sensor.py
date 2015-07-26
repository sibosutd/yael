'''
Parse sensor data and write data into csv file.
'''
__author__ = '1000892'

import os, sys, csv, getopt, time, datetime
import numpy as np
from numpy import genfromtxt

######################################################
# command usage
# example:
# python parse_sensor.py -i ../test_data/act05seq10.txt -o ../test_data/act05seq10_converted -t red
# python parse_sensor.py -i ../test_data/20150720_193742_193742_1.txt -o ../test_data/20150720_193742_193742_1_parsed -s 00:00:00 -t black
input_file = ''
output_file = ''
start_time = ''
glass_type = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], 'hi:s:t:o:')
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print 'Usage: convert_sensor.py -i <input_file> -o <output_file> -s <start_time> -t <glass_type>'
        sys.exit()
    elif opt == '-i':
        input_file = arg
    elif opt == '-o':
        output_file = arg
    elif opt == '-s':
        start_time = arg
    elif opt == '-t':
        glass_type = arg
    else:
        assert False, 'unhandled option'

######################################################
# parse data
print 'Convert sensor file:\n\t', input_file
if start_time == '':
    start_time = '00:00:00'
else: 
    # start_time = int(start_time)
    print 'Start time is:\n\t', start_time

t = time.strptime(start_time,'%H:%M:%S')
start_seconds = datetime.timedelta(hours=t.tm_hour,minutes=t.tm_min,seconds=t.tm_sec).total_seconds()

sensor_data = genfromtxt(input_file)

# black glass
if glass_type == 'black':
    # select 10s sensor data, remove first 2 and last 5 columns
    sensor_data = sensor_data[1+start_seconds*10:1+(start_seconds+15)*10,2:-5]
    sensor_data = np.delete(sensor_data, 9, axis=1)
elif glass_type == 'red':
    # select 10s sensor data, convert 30Hz data to 10Hz data
    sensor_data = sensor_data[::3,2:]
else:
    assert False, 'wrong option'

# check sensor data dimension
print 'Sensor data dimension is:\n\t', sensor_data.shape

# write csv file
if (os.path.isfile(output_file+'.csv')):
    assert False, 'file exists'
else:
    np.savetxt(output_file+'.csv', sensor_data, fmt='%.7e', delimiter=',')

print '######################################################'