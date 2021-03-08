function [M, A, Y] = CondEntropyMetricLearningMiniBatch(X, labels, no_dims, sigma, alpha, varargin) 
% [M, A, Y] = CondEntropyMetricLearning(X, labels, no_dims, sigma, alpha, param)
%
% Metric learning using matrix-based conditonal entropy. This is an
% implementation of the metric learning algorithm described in the
% paper:
% "Luis G Sanchez Giraldo and Jose Principe, Information Theoretic Learning Using 
% Infinitely Divisible Kernels. ICLR 2013." 
%
% INPUTS:
%
% X: (N x d_x) data matrix -each row corresponds to a single instance. 
% 
% labels: array of length N where the i-th entry denotes the class of the
%         i-th instance. labels can be a vector of integers indicating
%         the class, or a cell array with the name of the class.
%
% no_dims: positive integer that denotes the dimensionality of the
%          projected inputs.
% sigma: kernel size in k(y_i, y_j) = exp(-(|| y_i - y_j ||^2)/(2*sigma^2)).
%        
% alpha: order of Renyi's entropy (alpha > 1). 
%
% param: (optional) struct optimization parameters
% param.mu_in: Initial stepsize (default 0.5)
% param.mu_fin: Final stepsize  (default param.mu_in/10)
% param.n_iter: number of iterations (default 150)
%
% OUTPUTS:
%
% M: (d_x x d_x) positive semidefinite matrix to compute the semimetric
%    d_M(x_i, x_j) = (x_i - x_j)'*M*(x_i - x_j).	 
% 
% A: (d_x x d_y) transformation matrix. 
%  
% Y: (N x d_y) set of transformed inputs.
% 
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Luis Gonzalo Sanchez Giraldo, 2014
 

  %% RANDOM INITIALIZATION 
  N = size(X,1);        % Number of exemplars
  d_x = size(X,2);      % Input dimensionality
  N_batch = size(X, 3); % Number of batches
  d_y = no_dims;        % Dimensionality of the Projected space
  A = randn(d_x,d_y);   % Transformation Matrix
  A = sqrt(d_y)*A/sqrt(trace(A'*A));

  [L, ~] = labelMatrix(labels);  % Label matrix (one-hot encoding)
  K_l = L* L'/N;                 % Gram matrix for labels
  Y = X(:,:,1)*A;                       % Initial transformed inputs read the first batch
  
  %% OPTIMIZATION PROCEDURE
  % Checking paramters
  if isempty(varargin)
    param.mu_in =0.5;  % Initial stepsize
    param.n_iter =150; % Number of iterations 
    param.mu_fin = 0.05; % Final stepsize 
  else
    if(length(varargin) == 1)
      param = varargin{1};
      if ~isfield(param, 'mu_in')
        param.mu_in = 0.5;
      end
      if ~isfield(param, 'mu_fin')
        param.mu_fin = param.mu_fin/10;
      end
      if ~isfield(param, 'n_iter')
        param.n_iter = 150;
      end
    else
      error('CondEntropyMetricLearning error. Invalid number of input arguments.');
    end
  end
  if (param.mu_in <  param.mu_fin)
     warning('CondEntropyMetricLearning warning. Final stepsize is larger than initial stepsize.')
  end
  % Initialization of optimization parameters
  mu = param.mu_in;  % step size
  reg_par = 1; % Lagrange multiplier of the trace constraint on A
  
  batch_count = 1;
  for i= 1 : param.n_iter
    mu = mu - (param.mu_in - param.mu_fin)/param.n_iter;  % Step size schedulling  
    K_y = real(guassianMatrix(Y,sigma))/N;  % Gram matrix of tranformed inputs
    K_ly = K_l.*K_y*N;                      % Gram matrix joint-space
    
    %%% Compute spectrum of K_y and K_ly
    [V_y,L_y] = eig(K_y);
    V_y = real(V_y);
    lambda_y = abs(diag(L_y));

    [V_ly,L_ly] = eig(K_ly);
    V_ly = real(V_ly);
    lambda_ly = abs(diag(L_ly));
   

    %%% Compute gradient terms
    Grad_y = V_y*diag((alpha/(1 - alpha))*(1/(sum(lambda_y.^alpha)))*(lambda_y.^(alpha - 1)))*V_y';
    Grad_ly = V_ly*diag((alpha/(1 - alpha))*(1/(sum(lambda_ly.^alpha)))*(lambda_ly.^(alpha - 1)))*V_ly';
    Grad = N*Grad_ly.*K_l - Grad_y;
    P = Grad.*K_y;

    %%% Updating the transformation matrix (Plain Gradient Descend)
    A = A - mu*(X(:,:,batch_count)'*(P - diag(P*ones(N,1)))*(X(:,:,batch_count)*A) + reg_par*A);

    %%% Fixed point update of the lagrange multiplier
    reg_par = -trace((A'*X(:,:,batch_count)')*(P - diag(P*ones(N,1)))*(X(:,:,batch_count)*A))/(d_y);

    %%% Update Transformed inputs
    batch_count = batch_count + 1;
    if batch_count > N_batch
        batch_count =1;
    end
    Y = X(:,:,batch_count)*A;

    %%% Display progress 
    if (~mod(i,30))
      F =  (1/(1 - alpha))*log( (sum(lambda_ly.^alpha))/(sum(lambda_y.^alpha)) );
      fprintf('CondEntropyMetricLearning at Iteration %d. CondEntropy = %f, \nLagrange multiplier = %f, trace value = %f \n', i, F, reg_par, trace(A'*A));
    end
    
  end

  M = A*A';

end
