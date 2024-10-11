close all; clc; clear all;

% Define the folder containing the .mat files
folder_name = 'Datasets_splits_005'; 

% Get a list of all .mat files in the folder
mat_files = dir(fullfile(folder_name, '*.mat'));

%%

% Initialize arrays for combined trajectories
ExpData_fr = [];
ExpData_I = [];
ExpData_t = [];
ExpData_I_seq = [];
ExpData_t_seq = [];

% Loop through each .mat file and combine the trajectories
for k = 1:length(mat_files)
    % Load the current .mat file
    mat_file_path = fullfile(folder_name, mat_files(k).name);
    disp(mat_file_path)
    data = load(mat_file_path);
    
    % Concatenate the trajectories with the combined arrays
    ExpData_t = [ExpData_t, data.data_t];
    ExpData_fr = [ExpData_fr, data.data_fr];
    ExpData_I = [ExpData_I, data.data_I];
    ExpData_I_seq = [ExpData_I_seq, data.data_I_seq];
    ExpData_t_seq = [ExpData_t_seq, data.data_t_seq];
end

%Randomize the trajectories
randOrder = randperm(length(ExpData_t));

ExpData_t = ExpData_t(randOrder);
ExpData_fr = ExpData_fr(randOrder);
ExpData_I = ExpData_I(randOrder);
ExpData_I_seq = ExpData_I_seq(randOrder);
ExpData_t_seq = ExpData_t_seq(randOrder);


% Save the combined trajectories into a new .mat file
combined_mat_file = fullfile(folder_name, 'Exp_all_trajectories.mat');
save(combined_mat_file, 'ExpData_t','ExpData_fr', 'ExpData_I', 'ExpData_I_seq', 'ExpData_t_seq');

fprintf('All trajectories have been combined and saved to %s\n', combined_mat_file);
