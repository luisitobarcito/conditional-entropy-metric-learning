function [M, A, Y] = lmnnWrap(X, labels, no_dims)
[M, A, Y] = lmnn(X, labels);
A = A(:,1:no_dims);
