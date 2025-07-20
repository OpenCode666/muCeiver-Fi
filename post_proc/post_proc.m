%% Post Processing for Wi-Fi Human Pose Estimation
clear
clc
close all
%% load
% The following two lines assume that the results and code are located in the same directory.
% If not, please manually update the corresponding paths.
% currentPath = fileparts(mfilename('fullpath'));
% parentPath = [fileparts(currentPath) '\\experiments']; 
% ================= change as the folder path of the experiments!!!
parentPath = '...\\experiments';
% ================= change as the folder path of the experiments!!!
dataPath = [parentPath '\\deterministic\\training\\predictions\\pre_result.mat'];
load(dataPath)
pre_result = squeeze(pre_result);
%% Extract keypoints
dv = 0.03;
grid_s = 0.01;
X = 0.2:grid_s:0.8;
Y = grid_s:grid_s:1.29;
fprintf('Post Processing')
for i = 1:size(pre_result,1)
    if mod(i,20) == 0
        fprintf('.')
    end
    samples = squeeze(pre_result(i,:,:));
    [path_info_output,ind_spe,~] = Det_peaks(samples,50,Y,X);        % Detect peak values
    out_point = mergeClosePoints(path_info_output, dv, 15);          % Merge close points
    out_point = out_point(:,[3,2,1]);                                % [Z,X,Y]
    out_point = out_point';
    [out_sort(:,:,i),err(i,:)] = sortPoints(out_point);       % Sort points
end
fprintf('DONE \n')
fprintf('Keypoint Average Localization Error is %.1f cm\n',mean(err))
save_path = [fileparts(mfilename('fullpath')) '\\HPE.mat'];
save(save_path,'out_sort','err')