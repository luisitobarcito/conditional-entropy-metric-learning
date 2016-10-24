function [M, A, Y] = invCovWrap(X, labels)
C = cov(X);
[v,d] = eig(C);
d = diag(d);
d_inv = 1./d;
d_inv(d == 0) = 0;
M = v*diag(d_inv)*v';
A = v*diag(sqrt(d_inv));
Y = X*A;