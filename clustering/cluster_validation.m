function [mus,stds]=cluster_validation(X,labels,para)
%   input
%       labels: 0 for no-labels, labels >= 0
%       para:[repeats,Kmin, Kmax]
%           repeats: times for iteration in 
%           Kmin and Kmax: minimum and maximum numbers of clusters
%   
%   output:
%       mus: 3-by-repeats matrix of aic bic and f-measure mean
%       stds: 3-by-repeats matrix of aic bic and f-meausre standard
%       variance
%       if there exists points without label, only return f-measure
%       outcomes
% 


if ~exist('X', 'var')||~exist('labels','var')
    disp('Please enter the data and the labels')
end
if ~exist('para','var')
    disp('The default value could generate erros')
    para=[15 1 10]
end


%% GMM clustering
if min(labels)>=0
    % Fit data to GMM model with full covariance matrix
    repeats=para(1);
    KRange=para(2):para(3);
    mus=zeros(3,length(KRange));
    stds=zeros(3,length(KRange));
    temp=zeros(3,repeats);
    % run GMM 
    for i=para(2):para(3)
        for j=1:repeats
            obj{i}=gmdistribution.fit(X,i,'Start','randSample','CovType','full','Regularize',0.05);
            %AIC and BIC
            temp(1,j)= obj{i}.AIC;
            temp(2,j)= obj{i}.BIC;
            %F-measure
            post_p=[];
            for k=1:i
               gm=gmdistribution(obj{i}.mu(k,:),obj{i}.Sigma(:,:,k),1);
               post_p=[post_p pdf(gm,X)*obj{i}.PComponents(k)];
            end
            [Y,cluster_labels]=max(post_p,[],2);
            temp(3,j)=gmmFmeasure(cluster_labels,labels);
        end
        mus(1,i)=mean(temp(1,:));
        mus(2,i)=mean(temp(2,:));
        mus(3,i)=mean(temp(3,:));
        stds(1,i)=std(temp(1,:));
        stds(2,i)=std(temp(2,:));
        stds(3,i)=std(temp(3,:));
    end

elseif minlabels==0
        % Fit data to GMM model with full covariance matrix
    repeats=para(1);
    KRange=para(2):para(3);
    mus=zeros(1,length(KRange));
    stds=zeros(1,length(KRange));
    temp=zeros(1,repeats);
    % run GMM 
    for i=para(2):para(3)
        for j=1:repeats
            obj{i}=gmdistribution.fit(X,i,'Start','randSample','CovType','full','Regularize',0.05);
            %F-measure
            post_p=[];
            %genearte labels
            for k=1:i
               gm=gmdistribution(obj{i}.mu(k,:),obj{i}.Sigma(:,:,k),1);
               post_p=[post_p pdf(gm,X)*obj{i}.PComponents(k)];
            end
            [Y,cluster_labels]=max(post_p,[],2);
            temp(1,j)=gmmFmeasure(cluster_labels,labels);
        end
        mus(1,i)=mean(temp(1,:));
        stds(1,i)=std(temp(1,:));
    end
else
    disp('Invalid labels')
end


end