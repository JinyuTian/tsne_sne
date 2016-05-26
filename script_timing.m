clc; clear all;
%% Real data
type='helix';
noise = 0.01;
nspace = 2.^(7:11);
timermat = zeros(length(nspace),2);
for i = 1:length(nspace)
    n = nspace(i);                 %the number of data points
    [x, labels]=generate_data(type,n,noise);
    initial_dims = size(x,2); 

    tic;
    [Y_tsne,~] = tsne_mod(x, labels, 3, initial_dims, 30);
    timermat(i,1)= toc;
    tic;
    [Y_sne,~] = sne_mod(x, labels, 3, 30);
    timermat(i,2)= toc;

end
plot(nspace,timermat);