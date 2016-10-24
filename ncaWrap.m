function [M, A, Y] = ncaWrap(X, labels, no_dims)
[Y, mapping] = nca(X, labels, no_dims);
A = mapping.M;
M = A*A';