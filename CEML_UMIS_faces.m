% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Luis Gonzalo Sanchez Giraldo, 2014

clear all
close all
clc

%% Load face Data
% Modify this path to the actual location of your data folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_path = './data';  %% Your path to data goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(data_path);
load facedata %% umas faces

%% Preprocessing (centering, PCA and variance normalization)
X = bsxfun(@minus, X, mean(X,1));
[~,Xeig,V] = svd(X);
n_comp = min(size(X));
X = X*V(:,1:n_comp); %% no whitening
X = X/sqrt(trace(cov(X))/n_comp);


%% Run Conditional Entropy Metric Learning 
no_dims = 2;
d_y = no_dims;
sigma = sqrt(d_y);
alpha = 1.01;
param.n_iter = 300;
param.mu_in = 0.01;
param.mu_fin = 0.05;
[M, A, Y] = CondEntropyMetricLearning(X, labels, no_dims, sigma, alpha, param);


%% Plot Results
new_sigma = sqrt(2*trace(A'*A));
K_y = real(guassianMatrix(Y, new_sigma));
figure
subplot(121)
scatter(Y(:,1),Y(:,2),12,labels)
set(gca,'DataAspectRatio',[1 1 1])
title('Projected Features')
subplot(122)
imshow(K_y,[])
title('Gram Matrix Projected Features')
colormap jet
drawnow
