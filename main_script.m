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
dataset = 'mnist' % 'mnist', 'iris'
[x, labels] = load_dataset(dataset); 

%% RUNNING T-SNE + SNE
% parameters 
perp = 30;                  % Sets the perplexity to which the gaussians (on the higher dimensional data points) variance is set to get
initial_dims =size(x,2);   % If dims>initial_dims, then performs PCA to this dim.
no_dims = 2;                % the dimension of the lower dimensional space
%% tsne
Y_tsne = tsne_mod(x, labels, no_dims, initial_dims, perp);
%% sne
Y_sne = sne_mod(x, labels, no_dims, perp);
%% CLUSTERING
label=labels+1; %Only of Iris model
para='validation'; %validation or evaluation
cluster_evaluation(Y_sne,Y_tsne,label,para);
%% CLASSIFICATION
%training and testing data
training_ratio=0.3;
[train_sne,train_tsne,test_sne,test_tsne,test_labels,train_labels]=train_test_generation(training_ratio,Y_sne,Y_tsne,labels);
%Hyperparameter evaluation
[accuracy_sne,accuracy_tsne,NrSv_sne,NrSv_tsne,NrClass_sne,NrClass_tsne]=nv_svm_evaluation(train_sne,test_sne,train_tsne,test_tsne,train_labels,test_labels);
classification_eval_plot(accuracy_sne,accuracy_tsne,NrSv_sne,NrSv_tsne);
%% Training ratio evaluation
training_ratio_eval(labels, Y_tsne,Y_sne);

%%
model_tsne=svmtrain(train_labels,train_tsne,'-s 1 -t 2 -c 0.5 -g 0.1 -n 0.1');
figure()
plotBoundary(Y_tsne,labels,model_tsne)
model_sne=svmtrain(train_labels,train_sne,'-s 1 -t 2 -c 0.5 -g 0.1 -n 0.1');
figure()
plotBoundary(Y_sne,labels,model_sne);
%% print and save data
%plot the clustering outcome
numComponents_tsne=16;
numComponents_sne=16;
[model_sne,cluster_labels_sne]=gmmFitting(Y_sne,[numComponents_sne 2]);
[model_tsne,cluster_labels_tsne]=gmmFitting(Y_tsne,[numComponents_tsne 2]);
figure()
scatter(Y_sne(:,1),Y_sne(:,2),10,labels,'filled')
hold on
h = ezcontour(@(x,y)pdf(model_sne,[x y]),[min(Y_sne(:,1))-2 max(Y_sne(:,1))+2],[min(Y_sne(:,2))-2 max(Y_sne(:,2))+2]);

figure()
scatter(Y_tsne(:,1),Y_tsne(:,2),10,labels,'filled')
hold on
h = ezcontour(@(x,y)pdf(model_tsne,[x y]),[min(Y_tsne(:,1))-2 max(Y_tsne(:,1))+2],[min(Y_tsne(:,2))-2 max(Y_tsne(:,2))+2]);


