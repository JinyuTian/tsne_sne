function [ Y ] = adding_dimensions( X, no_dims, type )
%ADDING_DIMENSIONS creates 'noise' or unstructured data in no_dims
%dimensions and adds these to the input X. The extra dimensions are sampled
%from the 'type' distribution with a
%   X - dimension N x D, the data to add dimensions to
%   no_dims - scalar 1 x 1, the number of dimensions to add to the data
%   type - string, the distribution type of the additional data. If
%   gaussian it is sampled with same variance and mean as the X matrix. If
%   uniform sampled between highest and lowest value.

if nargin < 2
    no_dims = 10;
    type = 'gaussian' ;
end

if nargin < 3
    type = 'gaussian';
end

[N,~ ] = size(X);

if type == 'gaussian'
    meanx = mean(mean(X));
    if N == 1
        varx = var(X);
    else
        varx = var(var(X));
    end
    X_add = meanx + varx*randn(no_dims, N);
    
elseif type == 'uniform'
    max_x = max(max(X));
    min_x = min(min(X));
    meanx = mean(mean(X));
    X_add = meanx + (max-min)*rand(no_dims,N);
    
else
    X_add = rand(N,no_dims);
end
Y = [X X_add'];
end


