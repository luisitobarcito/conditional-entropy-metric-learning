% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Luis Gonzalo Sanchez Giraldo, 2018

clear all
close all
clc

%% Load MADELON Data
% Modify this path to the actual location of your data folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_path = '/home/lgsanchez/work/Data/UCI_data/MADELON/';  %% Your path to data goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(data_path);
load('madelon_train.data', '-ascii')%% load train data
X.train = madelon_train;
clear madelon_train;

load('madelon_train.labels', '-ascii')%% load train labels
labels.train = madelon_train;
clear madelon_train;  

load('madelon_valid.data', '-ascii')%% load train data
X.valid = madelon_valid;
clear madelon_valid;

load('madelon_valid.labels', '-ascii')%% load train labels
labels.valid = madelon_valid;
clear madelon_valid;  

%% Preprocessing (centering, PCA and variance normalization)
DO_PCA = false;
X.valid = bsxfun(@minus, X.valid, mean(X.train,1));
X.train = bsxfun(@minus, X.train, mean(X.train,1));

if DO_PCA
    [~,Xeig,V] = svd(X.train);
    n_comp = min(size(X.train));
    X.train = X.train*V(:,1:n_comp); %% no whitening
    X.valid = X.valid*V(:,1:n_comp);
else
   n_comp = size(X.train,2);
end
X.valid = X.valid/sqrt(trace(cov(X.train))/n_comp);
X.train = X.train/sqrt(trace(cov(X.train))/n_comp);


%% Run Conditional Entropy Metric Learning 
no_dims = 100;
d_y = no_dims;
sigma = sqrt(d_y);
alpha = 1.01;
param.n_iter = 2000;
param.mu_in = 0.01;
param.mu_fin = 0.01;
[M, A, Y] = CondEntropyMetricLearning(X.train, labels.train, no_dims, sigma, alpha, param);


%% Plot Results
new_sigma = sqrt(2*trace(A'*A));
K_y = real(guassianMatrix([Y(labels.train == 1,:); Y(labels.train == -1,:)], new_sigma));
figure
subplot(121)
scatter(Y(:,1),Y(:,2),12,labels.train)
set(gca,'DataAspectRatio',[1 1 1])
title('Projected Features')
subplot(122)
imshow(K_y,[])
title('Gram Matrix Projected Features')
colormap jet
drawnow
