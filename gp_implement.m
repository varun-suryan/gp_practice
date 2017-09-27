clear all;
clc;
close all;

points = [-10 : 0.1 : 10]';



l = 20/sqrt(2);

signal_std = 1;
noise = 1.0;
kernel = signal_std^2 * exp (- squareform(pdist(points)).^2 / (2 * l^2) ) + noise^2 * eye(numel(points));
R_prior = mvnrnd( zeros(numel(points), 1) , kernel, 1);

plot(points, R_prior,'r');
hold on;
% axis([1 100 -20 20]);

points_train = [points 100 * sign(points)];

post_kernel = kernel(151:end, 151:end) - kernel(151:end, 1:150) * pinv(kernel(1:150, 1:150) + noise^2 * eye(size(150))) * kernel(1:150, 151:end);

post_mean = kernel(151:end, 1:150) * pinv(kernel(1:150, 1:150) + noise^2 * eye(size(150))) * points_train(1:150, 2);

R_post = mvnrnd( post_mean , round(abs(post_kernel),3), 1);

plot(post_mean, 'g');

