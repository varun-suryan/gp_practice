clear all;
clc;
close all;


points = [1 : 0.5 : 100]';

l = 10/sqrt(2);


signal_std = 1;

noise = 0.0;

kernel = signal_std^2 * exp (- squareform(pdist(points)).^2 / (2 * l^2) ) + noise^2 * eye(numel(points));

R = mvnrnd( zeros(numel(points), 1) , kernel, 1);


plot(points, R);
hold on;
% axis([1 100 -20 20]);


