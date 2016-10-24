% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Luis Gonzalo Sanchez Giraldo, 2014

clear all
close all
clc

%% Create Stripes
N = 200;
n_stripes = 5;
X = [round(rand(N,1)*(n_stripes-1)) randn(N,1)];
labels = mod(X(:,1),2)+1;
X(:,1) = X(:,1)+0.1*randn(N,1);
X(:,2) = X(:,2)+labels(:);
[L,labels] = labelMatrix(labels); 
%%% Preprocess
X = bsxfun(@minus, X, mean(X,1));
stdX = std(X);
X = bsxfun(@rdivide, X, stdX);
X(:,stdX == 0) = 0;
 

%% Random Initialization
d_y = 2;
d_x = size(X,2);


%% Run CEML for n_reps number of repetitions per alpha value
param.mu_in =0.5;
param.n_iter =150;
param.mu_fin = 0.5;
no_dims = 2;
sigma = sqrt(2);
reg_par = 1;
count = 1;
%% Generate grid of alpha values with repetitions
n_reps = 60;
alphas = [1.01 1.3 2 5];
alphaV = ones(n_reps,1)*alphas;
alphaV = alphaV(:);
allS = zeros(2,length(alphaV));
for i1= 1:length(alphaV)
    alpha = alphaV(i1);  
    [~, A, ~] = CondEntropyMetricLearning(X, labels, no_dims, sigma, alpha, param);
    %%% Use SVD to unvcover dominant diraction in solution
    [~,~,v] = svd(A);
    allS(:,i1) = A*v(:,1);
end
allS = abs(allS);

%% Display example figure
figure
scatter(X(:,1),X(:,2),12,labels)

%% Calculate number of times algorithms converges to horizontal axis vs vertical axis  
feat_angles = atan(allS(2,:)./allS(1,:));
angles_threshold = feat_angles > pi/4;
angles_threshold = reshape(angles_threshold, n_reps, length(alphas));
angle_counts = zeros(2, length(alphas));
angle_counts(2, :) = sum(angles_threshold, 1);
angle_counts(1, :) = n_reps - angle_counts(2, :);


