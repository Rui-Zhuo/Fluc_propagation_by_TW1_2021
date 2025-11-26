clear; close all;
%% import data
file_dir = 'E:\Research\Work\tianwen_IPS\base_line_multiple\';
file_name = 'cases.xlsx';
data = readmatrix([file_dir, file_name]);
%% extract data
date = data(:,1);
time = data(:,2);
stat_x = data(:,3); % [km]
stat_y = data(:,4);
stat_z = data(:,5);
scale = data(:,6); % [s]
vp_magni = data(:,7); % [km/s]
theta_vp = data(:,8); % [deg.], refer to x-axis
%% calculate wave length
lambda = vp_magni .* scale; % [km]
%% calculate wave vector
% sky plane
theta_sp = atan2d(stat_y, stat_x); % [deg.]
% wave vector
wv_magni = 2*pi./lambda; % [km-1]
wv_x = wv_magni .* cosd(theta_vp) .* cosd(theta_sp);
wv_y = wv_magni .* cosd(theta_vp) .* sind(theta_sp);
wv_z = wv_magni .* sind(theta_vp);