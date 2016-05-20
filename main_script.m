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
Y_tsne = tsne_mod(x, labels, no_dims, initial_dims, perp);
%% sne
Y_sne = sne_mod(x, labels, no_dims, perp);
%% CLUSTERING
label=labels+1; %Only of Iris model
para='evaluation'; %validation or evaluation
cluster_evaluation(Y_sne,Y_tsne,label,para);
%% CLASSIFICATION
%training data
training_ratio=0.1;
ind=randperm(size(Y_tsne,1));
train_sne=Y_sne(ind(1:ceil(size(Y_sne,1)*training_ratio)),:);
test_sne=Y_sne(ind(ceil(size(Y_sne,1)*training_ratio):end),:);
train_tsne=Y_tsne(ind(1:ceil(size(Y_tsne,1)*training_ratio)),:);
test_tsne=Y_tsne(ind(ceil(size(Y_tsne,1)*training_ratio):end),:);
train_labels=labels(ind(1:ceil(size(Y_tsne,1)*training_ratio)));
test_labels=labels(ind(ceil(size(Y_tsne,1)*training_ratio):end));

accuracy_sne=zeros(10,10);
accuracy_tsne=zeros(10,10);
NrSv_sne=zeros(10,10);
NrClass_sne=zeros(10,10);
NrSv_tsne=zeros(10,10);
NrClass_tsne=zeros(10,10);

for i=1:1
    for j=1:1
        option=strcat('-s 1 -t 2 -c 0.5 -g ',num2str(i*0.1),' -n ',num2str(j*0.1));
        %nu-svm
        model_sne=svmtrain(train_labels,train_sne,option); %-s method -t kernel type -c cost -g gamma -v n-fold cross validation -n nu
        NrSv_sne(i,j)=model_sne.totalSV;
        NrClass_sne(i,j)=model_sne.nr_class;
        model_tsne=svmtrain(train_labels,train_tsne,option);
        NrSv_tsne(i,j)=model_tsne.totalSV;
        NrClass_tsne(i,j)=model_tsne.nr_class;        

        %testing
        [predicted_label_sne, temp, decision_sne]=svmpredict(test_labels,test_sne,model_sne);
        accuracy_sne(i,j)=temp(1);
        [predicted_label_tsne, temp, decision_tsne]=svmpredict(test_labels,test_tsne,model_tsne);
        accuracy_tsne(i,j)=temp(1);
    end
end
%%
figure()
plotBoundary(Y_sne,labels,model_sne);
figure()
plotBoundary(Y_tsne,labels,model_tsne)
%% print and save data


