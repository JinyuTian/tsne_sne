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
label=labels+1; %Only of Iris model
para='evaluation'; %validation or evaluation
cluster_evaluation(Y_sne,Y_tsne,label,para);
%% CLASSIFICATION
%training data
training_ratio=0.2;
ind=randperm(size(Y_sne,1));
train_sne=Y_sne(ind(1:ceil(size(Y_sne,1)*training_ratio)),:);
test_sne=Y_sne(ind(ceil(size(Y_sne,1)*training_ratio):end),:);
train_tsne=Y_tsne(ind(1:ceil(size(Y_tsne,1)*training_ratio)),:);
test_tsne=Y_tsne(ind(ceil(size(Y_tsne,1)*training_ratio):end),:);
train_labels=labels(ind(1:ceil(size(Y_tsne,1)*training_ratio)));
test_labels=labels(ind(ceil(size(Y_tsne,1)*training_ratio):end));

%nu-svm
model_sne=svmtrain(train_labels,train_sne,'-s 1 -t 2 -c 1 -g 0.1'); %-s method -t kernel type -c cost -g gamma -v n-fold cross validation
model_tsne=svmtrain(train_labels,train_tsne,'-s 1 -t 2 -c 1 -g 0.1' );
disp('training outcomes')
model_sne
model_tsne

%testing
disp('testing')
[predicted_label_sne, accuracy_sne, decision]=svmpredict(test_labels,test_sne,model_sne);
[predicted_label_tsne, accuracy_tsne, decision]=svmpredict(test_labels,test_tsne,model_sne);

%% print and save data


