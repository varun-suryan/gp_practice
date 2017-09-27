clear all;
clc;
close all;

points = [-30 : 0.1 : 30]';


l = 6/sqrt(2);

signal_std = 2;
noise = 0.1;

kernel_ = signal_std^2 * exp (- squareform(pdist(points)).^2 / (2 * l^2) ) + noise^2 * eye(numel(points));
mean_ = zeros(numel(points), 1);

R_prior = mvnrnd( mean_ , kernel_, 1);

plot(points, R_prior,'r');
hold on;
% axis([1 100 -20 20]);


frequency = 40;
points_train = points(1 : frequency : end);
points_train_target = sin(points_train) + 0.05 * randn(size(points_train));



% Sequential Implementation
count = 1;
for indexing = 1 : frequency : size(points, 1);
	coeff = 1 / kernel_(indexing, indexing) + noise^2 * eye(numel(indexing));
	mean_ = mean_ + kernel_(:, indexing) * coeff * (points_train_target(count) - mean_(indexing));	
	kernel_ = kernel_ - kernel_(:, indexing) * coeff * kernel_(indexing, :) + noise^2;
	count = count + 1;
end

% % Implementation in one Stroke
% indexing = 1 : frequency : size(points, 1);
% coeff = pinv(kernel_(indexing, indexing) + noise^2 * eye(numel(indexing)));
% mean_ = mean_ + kernel_(:, indexing) * coeff * (points_train_target - mean_(indexing));	
% kernel_ = kernel_ - kernel_(:, indexing) * coeff * kernel_(indexing, :) + noise^2;


% R_post = mvnrnd( mean_, kernel_, 1);

plot(points, mean_, 'g')
% plot(points, R_post,'g');
hold on;
scatter(points_train, points_train_target)

