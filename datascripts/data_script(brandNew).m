clear all
clc

%% Script for generating data to try out properties
n = 1000; % Number of data-points
noise = 0.05;
% ml_toolbox -> methods -> tooboxes -> drtoolbox -> generate_data
%	[X, labels, t] = generate_data(dataname, n, noise)

%% broken swiss with add dimensions
[X, labels, t] = generate_data('brokenswiss', n, noise);

%% swiss with add dimensions NOTE: do we want the hole to be in one cluster or not? 
[X, labels, t] = generate_data('swiss', n, noise);

%% swiss with varying density of sampling
[X, labels, t] = generate_data('changing_swiss', n, noise);

%% gaussians in various dimensions, higher D? 

%% gaussian mixture model data, clusters in 3d
% [X,labels,gmm] = ml_clusters_data(num_samples,dim,num_classes);
 [X, labels, t] = generate_data('3d_clusters', n, noise);

%% Generate helix
[X, labels, t] = generate_data('helix', n, noise);
%% Twinpeaks data
[X, labels, t] = generate_data('twinpeaks', n, noise);


%% Visualize the data
%plot_sne(X, labels, size(X,1))
%% Adding extra dimensions to the data (after this no visualization..) 

N=10; %noise dimension

X_embedd = adding_dimensions(X,N); %add extra dimension

% look at three of the added dimensions and the first dimensions
figure()
plot_sne(X_embedd(:,1:3), labels, size(X,2))


figure()
plot_sne(X_embedd(:,4:6), labels, size(X,2))


figure()
plot_sne(X_embedd(:,2:4), labels, size(X,2))


%% Running SNE and t-SNE

%rearrange the data clustering
%if want to do sne and tSne on the original data, replace X_embedd with X
% ind=randperm(size(X_embedd,1));
% train_X=X_embedd(ind(1:end),:);
ind=randperm(size(X,1));
train_X=X(ind(1:end),:);
train_labels=labels(ind(1:end));
no_dims=2;  %dimensionality reduction to no_dims
perplexity=30;
normalization= false; %whether normalize the orginal data

% Run SNE    
% Normalization the data
if normalization
    train_X_norm = train_X - min(train_X(:));
    train_X_norm = train_X_norm / max(train_X_norm(:));
    train_X_norm = bsxfun(@minus, train_X_norm, mean(train_X_norm, 1));
    train_X=train_X_norm;
end
mappedX1=sne(train_X, no_dims, perplexity);
figure()
if no_dims<=3
    plot_sne(mappedX1, train_labels, no_dims)
else
    disp('Dimensinality too high to visualize')
end

% Run tSNE
figure()
mappedX2=pure_tsne(train_X,train_labels,no_dims,perplexity);
if no_dims<=3
    plot_sne(mappedX2, train_labels, no_dims);
else
    disp('Dimensinality too high to visualize')
end

%%
%------------------------- Clustering--------------------------%
%% Kernel K-Means
% options.kernel='gauss'; %''gauss' for rbf kernel or use 'gauss-diag'
% options.kpar = 0.1;     % this is the variance of the Gaussian function

cluster_options             = [];
cluster_options.method_name = 'kernel-kmeans';
cluster_options.kernel='gauss'; %''gauss' for rbf kernel or use 'gauss-diag'
cluster_options.kpar = 0.1;     % this is the variance of the Gaussian function
repeats                     = 15;
Ks                          = 1:10;

[mus1, stds1]                 = ml_clustering_optimise(mappedX1(:,1:no_dims),Ks,repeats,cluster_options,'start','sample','Distance','sqeuclidean');

%% K-means
%
%   Lets see if we can find the number of clusters through RSS, AIC and BIC
%

cluster_options             = [];
cluster_options.method_name = 'kmeans';
repeats                     = 15;
Ks                          = 1:10;

% AIC BIC RSS calculation for SNE
[mus1, stds1]                 = ml_clustering_optimise(mappedX1(:,1:no_dims),Ks,repeats,cluster_options,'start','sample','Distance','sqeuclidean');
[mus2, stds2]                 = ml_clustering_optimise(mappedX2(:,1:no_dims),Ks,repeats,cluster_options,'start','sample','Distance','sqeuclidean');
%% Plot RSS, AIC and BIC
ml_plot_rss_aic_bic(mus1,stds1,Ks);

ml_plot_rss_aic_bic(mus2,stds2,Ks);


%% K-means F-measure

repeat = 15;    %Times running the k-means            
KRange = 1:10;  %Range of cluster number for F-measure checking
means=zeros(length(KRange),2);
stds=zeros(length(KRange),2);

options = [];
options.method_name  = 'kmeans';

for k = 1:length(KRange)

    options.K = KRange(k);
    
    F = zeros(repeat,1);

    for i = 1:repeat
        [result] = ml_clustering(mappedX1,options,'start','sample','Distance','sqeuclidean');
        F(i) = ml_Fmeasure(result.labels,train_labels);
    end

    means(k,1) = mean(F);
    stds(k,1) = std(F);
    
    for i = 1:repeat
        [result] = ml_clustering(mappedX2,options,'start','sample','Distance','sqeuclidean');
        F(i) = ml_Fmeasure(result.labels,train_labels);
    end

    means(k,2) = mean(F);
    stds(k,2) = std(F);

end
% Plot F-measure
figure()
errorbar(KRange,means(:,1),stds(:,1),'r--s');
hold on
errorbar(KRange,means(:,2),stds(:,2),'g--s');
legend('SNE','tSNE');
title('F-measure for different projections')
ylabel('FMeasure')
xlabel('Number of clusters')


%% GMM clustering

% Fit data to GMM model with full covariance matrix
KRange=1:10;
AIC=zeros(1,10);
BIC=zeros(1,10);

% run GMM for SNE
for i=1:10
    obj{i}=gmdistribution.fit(mappedX1,i,'Start','randSample','CovType','full');
    AIC(i)= obj{i}.AIC;
    BIC(i)= obj{i}.BIC;
end

% Plot AIC BIC
figure()
plot(KRange,AIC,'r--s');
hold on
plot(KRange,BIC,'g--s');
legend('SNE_AIC','SNE_BIC');
title('AIC and BIC for SNE using GMM')
ylabel('AIC & BIC')
xlabel('Number of clusters')


%Plot the best fitting
figure()
scatter(mappedX1(:,1),mappedX1(:,2),10,train_labels,'filled')
hold on
[minAIC,numComponents] = min(AIC);
h = ezcontour(@(x,y)pdf(obj{numComponents},[x y]),[min(mappedX1(:,1))-2 max(mappedX1(:,1))+2],[min(mappedX1(:,2))-2 max(mappedX1(:,2))+2]);



% run GMM for tSNE
for i=1:10
    obj{i}=gmdistribution.fit(mappedX2,i,'Start','randSample','CovType','full');
    AIC(i)= obj{i}.AIC;
    BIC(i)= obj{i}.BIC;
end

% Plot AIC BIC
figure()
plot(KRange,AIC,'r--s');
hold on
plot(KRange,BIC,'g--s');
legend('tSNE_AIC','tSNE_BIC');
title('AIC and BIC for tSNE using GMM')
ylabel('AIC & BIC')
xlabel('Number of clusters')



%Plot the best fitting
figure()
scatter(mappedX2(:,1),mappedX2(:,2),10,train_labels,'filled')
hold on
[minAIC,numComponents] = min(AIC);
h = ezcontour(@(x,y)pdf(obj{numComponents},[x y]),[min(mappedX2(:,1))-2 max(mappedX2(:,1))+2],[min(mappedX2(:,2))-2 max(mappedX2(:,2))+2]);


   








