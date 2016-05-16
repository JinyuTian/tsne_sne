%% Shows effect of plotting Kullback Leibler divergence
n = 100; d = 100;
kb = @(p,q) p./100 *log(p./q); 
Z = zeros(n,d);
for i = 1:n
    for j = 1:d
        Z(i,j) = kb(i,j);
    end
end
contour(Z,50)
% Add labels and stuff

surf(Z)
