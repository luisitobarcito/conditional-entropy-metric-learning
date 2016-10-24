function [M, A, Y] = mcmlWrap(X, labels, no_dims)
mapping = mcml(X, labels, no_dims);
A = mapping.M;
Y = X*A;
M = A*A';