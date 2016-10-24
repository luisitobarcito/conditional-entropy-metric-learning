%% RUN ALL THE METHODS
num_runs = 10;
num_folds = 2;
knn_neighbor_size = 4;


for i  = 1:num_runs
  
    
    %%% Conditional Entropy Metric Learning (Proposed Method)
    d_y = 3;
    sigma = sqrt(d_y);
    alpha = 1.01;
    acc(i).CEML = CrossValidateKNN(labels, X, @(labels, X) CondEntropyMetricLearning(X, labels, 3, sigma, alpha), num_folds, knn_neighbor_size);
    disp(sprintf('CEML kNN cross-validated accuracy = %f', acc(i).CEML));
    
    %%% Information Theoretic Metric Learning (Davis 2007)
    acc(i).ITML = CrossValidateKNN(labels, X, @(labels,X) MetricLearningAutotuneKnn(@ItmlAlg, labels, X), num_folds, knn_neighbor_size);
    disp(sprintf('ITML kNN cross-validated accuracy = %f', acc(i).ITML));
    
    %%% Neigbourghood Component Analys (Goldberger 2004)
    acc(i).NCA = CrossValidateKNN(labels, X, @(labels, X) ncaWrap(X, labels, 3), num_folds, knn_neighbor_size);
    disp(sprintf('NCA kNN cross-validated accuracy = %f', acc(i).NCA));
    
    %%% Maximally Collapsing Metric Learning (Globerson 2005)
    acc(i).MCML = CrossValidateKNN(labels, X, @(labels, X) mcmlWrap(X, labels, 3), num_folds, knn_neighbor_size);
    disp(sprintf('MCML kNN cross-validated accuracy = %f', acc(i).MCML));
    
    
    %%% Large Margin Nearest Neighbor (Weinberger 2005)
    acc(i).LMNN = CrossValidateKNN(labels, X, @(labels, X) lmnnWrap(X, labels, 3), num_folds, knn_neighbor_size);
    disp(sprintf('LMNN kNN cross-validated accuracy = %f', acc(i).LMNN));
    
    
    %%% Inverse Covariance (Whitening)
    acc(i).invCov = CrossValidateKNN(labels, X, @(labels, X) invCovWrap(X, labels), num_folds, knn_neighbor_size);
    disp(sprintf('InvCov kNN cross-validated accuracy = %f', acc(i).invCov));
    
    
    %%% Eucledian Distance
    acc(i).Euclidean = CrossValidateKNN(labels, X, @(labels, X) euclideanWrap(X, labels), num_folds, knn_neighbor_size);
    disp(sprintf('Euclidean kNN cross-validated accuracy = %f', acc(i).Euclidean));
    

end
