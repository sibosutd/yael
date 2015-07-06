%% Generate Codebook for Fisher Vector
clear all; 

% Initialization and Parameters
PROJECT_PATH = '/home/sibo/Documents/Projects/multimodal_sensors/sensor_data/';
NUM_OF_CODEBOOK = 2560;
DIM_AFTER_PCA = 100;

% Building codebooks
disp('########## Building codebooks ##########');
load([PROJECT_PATH 'results/all_feature_num.mat']);
list = dir([PROJECT_PATH 'data_mat/*.mat']);
num = length(list);
select_rate = NUM_OF_CODEBOOK / all_feature_num;
traj_select = [];

for i = 1 : num
    % Get file name
    name_string = list(i, 1).name;
    disp(name_string);
    % Initialization
    clear feature;
    load([PROJECT_PATH 'data_mat/' name_string]);
    % Randomly select features
    feature_num = size(feature,1);
    select_num = ceil(feature_num * select_rate);
    rand_select = ceil(feature_num * rand(1, select_num));
    % Trajectory
    traj_select = [traj_select; feature(rand_select,:)];
end

disp('########## Reduce dimensionality using PCA ##########');
% Calculate the mean value of select samples
traj_select_mean = mean(traj_select);
% Save the mean value
save([PROJECT_PATH 'codebook/traj_select_mean.mat'],'traj_select_mean');
% Reduce dimensionality by two using PCA
[traj_select_coeff, traj_select_score] = pca(traj_select);
% Save coefficient matrix
save([PROJECT_PATH 'codebook/traj_select_coeff.mat'],'traj_select_coeff');
% Save reduced principal component score
traj_select_reduced = traj_select_score(:,1:DIM_AFTER_PCA);
save([PROJECT_PATH 'codebook/traj_select_reduced.mat'],'traj_select_reduced');
