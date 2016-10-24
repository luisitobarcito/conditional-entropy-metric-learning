function [L,indices] = labelMatrix(categories)
[indices,names] = grp2idx(categories);
N = length(indices);
C = max(indices);
L = zeros(N,C);
for i=1:C
    L(i==indices,i) = 1;
end
