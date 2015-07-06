% Read sensor data into .mat file
clear all; 

PROJECT_PATH = '/home/sibo/Documents/Projects/multimodal_sensors/sensor_data/';
SIZE = 10;

list = dir([PROJECT_PATH 'data_txt/*.txt']);
num = length(list);
all_feature_num = 0;

for i_file = 1 : num
    % Get file name
    name_string = list(i_file, 1).name;
    disp(['Reading ' name_string ' ...']);

    % read in data
    fid = fopen([PROJECT_PATH 'data_txt/' name_string]);
    data = textscan(fid,'%f_%f\t%f_%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f');
    fid = fclose(fid);

    [a, b] = size(data{1,1});
    feature = [];
    for i = 1 : SIZE : (floor(a/SIZE)*SIZE)
        if (i == 1)
            for j = 5 : 23
                feature = [feature data{1,j}(i:i+SIZE-1,1)'];
            end
        else
            feature_line = [];
            for j = 5 : 23
                feature_line = [feature_line data{1,j}(i:i+SIZE-1,1)'];
            end
            feature = [feature; feature_line];
        end
    end
    featureNum = size(feature,1);
    all_feature_num = all_feature_num + featureNum;
    save([PROJECT_PATH 'data_mat/' name_string '.mat'], 'feature');
end

save([PROJECT_PATH 'results/all_feature_num.mat'],'all_feature_num');
