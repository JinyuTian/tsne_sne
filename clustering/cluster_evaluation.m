function [mus,stds]=cluster_evaluation(X,labels,para)
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
            obj{i}=gmdistribution.fit(X,i,'Start','randSample','CovType','full');
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
    % Plot AIC BIC
    figure()
    errorbar(KRange,mus(1,:),stds(1,:),'r--s');
    hold on
    errorbar(KRange,mus(2,:),stds(2,:),'g--s');
    legend('AIC','BIC');
    title('AIC and BIC using GMM')
    ylabel('AIC & BIC')
    xlabel('Number of clusters')
    figure()
    errorbar(KRange,mus(3,:),stds(3,:),'g--s')
    title('F1-Measure using GMM')
    ylabel('F1-Measure')
    xlabel('Number of clusters')

    %Plot the best fitting
    figure()
    [minAIC,numComponents] = min(mus(1,:));
    if size(X,2)==2
        scatter(X(:,1),X(:,2),10,labels,'filled')
        hold on
        h=ezcontour(@(x,y)pdf(obj{numComponents},[x y]),[min(X(:,1))-2, max(X(:,1))+2],[min(X(:,2))-2, max(X(:,2))+2]);
        title('Best fitting according to AIC')
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
            obj{i}=gmdistribution.fit(X,i,'Start','randSample','CovType','full');
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
    figure()
    errorbar(KRange,mus(1,:),stds(1,:),'g--s')
    title('F1-Measure using GMM')
    ylabel('F1-Measure')
    xlabel('Number of clusters')

    %Plot the best fitting
    figure()
    [minAIC,numComponents] = min(mus(1,:));
    if size(X,2)==2
        scatter(X(:,1),X(:,2),10,labels,'filled')
        hold on
        h=ezcontour(@(x,y)pdf(obj{numComponents},[x y]),[min(X(:,1))-2, max(X(:,1))+2],[min(X(:,2))-2, max(X(:,2))+2]);
        title('Best fitting according to F1-measure')
    end
else
    disp('Invalid labels')
end


end