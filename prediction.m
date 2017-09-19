clear all;
clc;
close all;


x = linspace(-7, 7, 10)';                 % 20 training inputs
y =   4 * sin(x);

xs = linspace(-9, 9, 61)';                  % 61 test inputs 

l = 1; signal_std = 1; noise_std = .90;

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [log(l) log(signal_std)], 'lik', log(noise_std));


hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

% hyp2 =
% mean: []
% cov: [-0.6352 -0.1045]
% lik: -2.3824

[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
hold on; plot(xs, mu); plot(x, y, '+')