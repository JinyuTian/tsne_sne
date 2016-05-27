%% Choose a datatype
clc; clear all; close all
%% toy data
n=1000;                 %the number of data points
noise=0.01;             % the noise on the generated data
type='swiss';     % 'swiss', 'changing_swiss' '3d_clusters' 'twinpeaks' 'difficult' 'intersect' 'helix'
types = {'swiss', 'changing_swiss', 'twinpeaks', '3d_clusters', 'difficult', 'intersect', 'helix'};
% Use dimensionality reduction toolbox to generate data
[x, labels]=generate_data(type,n,noise);
perp = 40;                  % Sets the perplexity to which the gaussians (on the higher dimensional data points) variance is set to get
initial_dims = size(x,2);   % If dims>initial_dims, then performs PCA to this dim.
no_dims = 2;                % the dimension of the lower dimensional space
%% tsne
% width = 2; height = 5;
Result_cell=cell(7,3);
for i = 1:length(types)
    string = types{i};
    [x, labels]=generate_data(string,n,noise);
    initial_dims = size(x,2);   % If dims>initial_dims, then performs PCA to this dim.
    %
    %     subplot(height,width,3*(i-1)+1)
    %     plot_sne(x,labels,initial_dims);
    %
    Y_tsne = tsne_mod(x, labels, no_dims, initial_dims, perp);
    Y_sne = sne_mod(x, labels, no_dims, perp);
    Result_cell{i,1} = Y_sne;
    Result_cell{i,2} = Y_tsne;
    Result_cell{i,3} = types{i};
    %     subplot(height,width,2*(i-1)+1)
    %     plot_sne(Y_tsne,labels,2);
    %     subplot(height,width,2*(i-1)+2)
    %     plot_sne(Y_sne,labels,2);
end
%% plotting
% create subplot of each to evaluate.
% (seems like too few iterations for more complex data sets)
for i = 1:7
    string = types{i};
    [x, labels]=generate_data(string,n,noise);
    
    figure(i)
    subplot(2,2,[1:2]);
    plot_sne(x,labels,3)
    set(gca,'FontSize',16)
    title('Original data')
    subplot(2,2,3);
    plot_sne(Result_cell{i,1},labels,2)
    title('SNE mapping')
    set(gca,'FontSize',16)
    subplot(2,2,4);
    plot_sne(Result_cell{i,2},labels,2)
    set(gca,'FontSize',16)
    title('t-SNE mapping')
    
end

%% Just one



type = 'SwissRoll';
perplexity = 30;
[x,dim,labels]=generateData(type);

[ytsne,~] = tsne_mod(x',labels,2,dim,perplexity);
[ysne,~] = sne_mod(x',labels,2,perplexity);
%%

subplot(2,2,[1:2]);
plot_sne(x',labels,3)
set(gca,'FontSize',16)
title('Original data')
subplot(2,2,3);

plot_sne(ytsne,labels,2)
title('t-SNE mapping')
set(gca,'FontSize',16)

subplot(2,2,4);
plot_sne(ysne,labels,2)
set(gca,'FontSize',16)
title('SNE mapping')