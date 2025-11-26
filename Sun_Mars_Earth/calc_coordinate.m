clear; close all;
% SH (ShangHai): 121.47 E, 31.23 N
% JS (JiaMuSi): 130.78 E, 46.50 N
% KS (KaShi): 75.99 E, 39.47 N
% YG (Yarragadee, Geraldton): 114.60 E, 28.76 S
lon = 75.99; % [deg.]
lat = 39.47; % [deg.]

Re = 6371000; % [m]
x_coor = round(Re * cosd(lat) * cosd(lon), 3);
y_coor = round(Re * cosd(lat) * sind(lon), 3);
z_coor = round(Re * sind(lat), 3);

disp([num2str(x_coor), ' ', num2str(y_coor), ' ', num2str(z_coor)]);