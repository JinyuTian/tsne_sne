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
no_dims = 2;                % the dimension of the lower dimensional space
%% Creates an array of tsne-results with their corresponding KLD
[N, D] = size(x);
Perplexity = linspace(5,40,2); 
runs = 5;
Y_tsne_results = zeros(N,2,length(Perplexity),runs);
KLD_results = zeros(length(Perplexity),runs);
for i = 1:length(Perplexity)
    for j = 1:runs
        [Y_tsne, KLD]=tsne_mod(x, labels, no_dims, D, Perplexity(i));
        Y_tsne_results(:,:,i,j) = Y_tsne;
        KLD_results(i,j) = KLD;
    end
end
KLD_mean = mean(KLD_results,2);
errorbar(Perplexity, KLD_mean, std(KLD_results'));

%% SNE-results and corresponding KLD
[N, D] = size(x);
Perplexity = linspace(5,40,2); 
runs = 5;
Y_sne_results = zeros(N,2,length(Perplexity),runs);
KLD_sne_results = zeros(length(Perplexity),runs);
for i = 1:length(Perplexity)
    for j = 1:runs
        [Y_sne, KLD]=sne_mod(x, labels, no_dims, D, Perplexity(i));
        Y_sne_results(:,:,i,j) = Y_sne;
        KLD_sne_results(i,j) = KLD;
    end
end
figure()
KLD_mean = mean(KLD_sne_results,2);
errorbar(Perplexity, KLD_mean, std(KLD_sne_results'));

