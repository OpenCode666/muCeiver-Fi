%% Human Pose Estimation Case visualization
clear
clc
close all
%% Load data
load_path = [fileparts(mfilename('fullpath')) '\\HPE.mat'];
load(load_path)
%% Plot
flag.line = 1;     % add line
% Walking
body3D_show(out_sort(:,:,105),flag,'Walking');
% Pointing
body3D_show(out_sort(:,:,245),flag,'Pointing');
% Hands Up
body3D_show(out_sort(:,:,480),flag,'Hands Up');
% Hands Open
body3D_show(out_sort(:,:,670),flag,'Hands Open');
% Sitting Down
body3D_show(out_sort(:,:,870),flag,'Sitting Down');
% Standing
body3D_show(out_sort(:,:,1300),flag,'Standing');


