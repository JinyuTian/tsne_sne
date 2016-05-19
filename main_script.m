%% Choose a datatype
clc; clear all; close all
%% toy data
n=1000;                 %the number of data points
noise=0.01;             % the noise on the generated data
type='brokenswiss';     % 'swiss', 'changing_swiss' '3d_clusters' 'twinpeaks' 'difficult' 'intersect' 'helix'
   
% Use dimensionality reduction toolbox to generate data
[x, labels]=generate_data(type,n,noise);

% Add a "noisy dimension" consisting of an extra feature of gaussian noise
N = 1;      % number of extra dimensions
% X_embedd = adding_dimensions(x,1) 
%% Real data
dataset = 'iris' % 'mnist', 
[x, labels] = load_dataset(dataset); 

%% RUNNING T-SNE + SNE
% parameters 
perp = 40;                  % Sets the perplexity to which the gaussians (on the higher dimensional data points) variance is set to get
initial_dims = size(x,2);   % If dims>initial_dims, then performs PCA to this dim.
no_dims = 2;                % the dimension of the lower dimensional space
%% tsne
Y_tsne = tsne(x, labels, no_dims, initial_dims, perp);
%% sne
Y_sne = sne(x, labels, no_dims, perp);
%% CLUSTERING
[mus,stds]=cluster_evaluation(Y_tsne,labels,[10 1 5]);
[minFmeasure,numComponents]=min(mus(3,:));
[model,cluster_labels]=gmmFitting(Y_tsne,[numComponent,2]);
%% print and save data


