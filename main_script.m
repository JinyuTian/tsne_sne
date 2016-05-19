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
%evaluation of the algorithm
label=labels+1;
numComponents=max(label);
error_sne=zeros(10,1);
error_tsne=zeros(10,1);
for i=1:10
    [model_sne,cluster_labels_sne]=gmmFitting(Y_sne,[numComponents 2]);
    [model_tsne,cluster_labels_tsne]=gmmFitting(Y_tsne,[numComponents 2]);
    error_sne(i)=labelComparison(label,cluster_labels_sne)/length(label);
    error_tsne(i)=labelComparison(label,cluster_labels_tsne)/length(label);
end
disp('Error ratio of SNE')
mean(error_sne)
disp('Error ratio of tSNE')
mean(error_tsne)

disp('Press any key to continue');
pause

% ----validation of the clustering-------
label=labels+1;
[mus_sne,stds_sne]=cluster_evaluation(Y_sne,label,[10 1 4]);
[mus_tsne,std_tsne]=cluster_evaluation(Y_tsne,label,[10 1 4]);
cluster_comparison(mus_sne,stds_sne,mus_tsne,std_tsne);
%% print and save data


