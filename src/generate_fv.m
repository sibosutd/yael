%% Generate the Fisher Vector for each video
clear all;

% Initialization and Parameters
PROJECT_PATH = '/home/sibo/Documents/Projects/multimodal_sensors/sensor_data/';
NUM_OF_CLUSTERS = 25;
DIM_AFTER_PCA = 100;

% Load codebook
disp('########## Loading codebook ##########');
load([PROJECT_PATH 'codebook/traj_select_reduced.mat']);
load([PROJECT_PATH 'codebook/traj_select_coeff.mat']);
load([PROJECT_PATH 'codebook/traj_select_mean.mat']);

% GMM clustering
disp('########## GMM clustering ##########');
[traj_means, traj_covariances, traj_priors] = vl_gmm(traj_select_reduced', NUM_OF_CLUSTERS);

save([PROJECT_PATH 'codebook/traj_GMM.mat'],'traj_means','traj_covariances','traj_priors');

disp('########## Fisher encoding ##########');
list = dir([PROJECT_PATH 'data_mat/*.mat']);
num = length(list);
fisher_data = [];
for i = 1 : num
    % Get file name
    name_string = list(i, 1).name;
    disp(name_string);
    % Initialization
    clear feature;
    load([PROJECT_PATH 'data_mat/' name_string]);
    feature_num = size(feature,1);
    % num_data_to_be_encoded = feature_num;
    % Remove the selected samples mean value (Important!)
    traj_sub_sample_mean = feature - ones(feature_num,1) * mean(feature);
    % Save data for each descriptor
    traj_data_to_be_encoded = traj_sub_sample_mean * traj_select_coeff(:,1:DIM_AFTER_PCA);
    % Fisher encoding
    traj_encoding = vl_fisher(traj_data_to_be_encoded', traj_means, traj_covariances, traj_priors, 'Improved');

    all_encoding = traj_encoding;
    % Save fisher vectors for each video
    save([PROJECT_PATH 'fv_combined/fv_' name_string], 'all_encoding');
    % Combine all fisher data into one variable
    fisher_data = [fisher_data; all_encoding'];
end

% Save fisher data file
disp('########## Save Fisher Vector data ##########');
save([PROJECT_PATH 'results/fisher_data.mat'],'fisher_data');
csvwrite([PROJECT_PATH 'results/fv_sensor.csv'], fisher_data);
csvwrite([PROJECT_PATH '../results/fv_sensor.csv'], fisher_data);
