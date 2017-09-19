clear all;
clc;
close all;


[pointsX, pointsY] =  meshgrid([1 : 0.1 : 5], [1 : 0.1 : 5]);

l = 0.5/sqrt(2);


signal_std = 1;

noise = 0;

kernel = signal_std^2 * exp (- squareform(pdist([pointsX(:), pointsY(:)])).^2 / (2 * l^2) ) + noise^2 * eye(numel(pointsX));


R = mvnrnd( zeros(numel(pointsX), 1) , kernel, 1);

R = reshape(R, size(pointsX));

surf(R)

% plot(points, R);


% hold on;
% axis([1 100 -20 20]);


