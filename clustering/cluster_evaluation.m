function [cluster_labels,cluster_para]=cluster_evaluation(method,X,labels,gridSearch,para)
%   method: k-means and GMM
%   gridSearch is a booleam for whether do parameter grid search or not
%   if gridSearch is true, para is [repeats,Kmin, Kmax]
%   repeats: times for iteration in 
%   Kmin and Kmax: minimum and maximum numbers of clusters
%   if gridSearch is false, the para is formed as followed:
%   k-means: [K]: numbers of clustering
%   GMM:[K covariant_matrix_type]: covariant_matrix_type:1 for diagnal, 2 for full
%   
% 
%   output:
%       cluster_lables: labels got from culstering
%       cluster_para: relevant information from the specific algorithm, for the
%       gridSearch case, only return the cluster_para
%       k-means:
%       cluster_para{1}=[mus,std]; mus:3-by-K matrix
%       for rss-aic-bic mean; std: 3-by-K matrix for std of rss-aic-bic,
%       cluster_para{2} for the F-measure
%       GMM:
%       cluster_para=[mus,stds]: mus and stds are 2-by-k matrixes for aic-bic
%       If the gridSearch is not implemented:
%       k-means: center of the clusters
%       GMM: the GMM model 
%
% 

if ~exist('method', 'var')
    method='k-means';
    return;
end
if ~exist('X', 'var')||~exist('labels','var')
    disp('Please enter the data and the labels')
end
if ~exist('gridSearch','var')
    gridSearch=true;
    if strcmp(method,'k-means')
        para=[15 1 10];
    elseif strcmp(method,'GMM')
        para=[15 1 10];
    else
        disp('Invalid method.')
    end
end
if ~exist('para','var')
    disp('Please input hyperparameters');
end

cluster_labels=[];


if strcmp(method,'k-means')
    if gridSearch
        % grid search for the optimal hyperparameter for 
        cluster_options             = [];
        cluster_options.method_name = 'kmeans';
        repeats                     = para(1);
        Ks                          = para(2):para(3);

        % AIC BIC RSS calculation for SNE
        [mus, stds]                 = ml_clustering_optimise(X,Ks,repeats,cluster_options,'start','sample','Distance','sqeuclidean');
        cluster_para{1}=[mus,stds]
        %% Plot RSS, AIC and BIC
        ml_plot_rss_aic_bic(mus,stds,Ks);
        
       %% K-means F-measure
        repeat = para(1);    %Times running the k-means            
        KRange = para(2):para(3);  %Range of cluster number for F-measure checking
        means=zeros(length(KRange),1);
        stds=zeros(length(KRange),1);

        options = [];
        options.method_name  = 'kmeans';

        for k = 1:length(KRange)

            options.K = KRange(k);

            F = zeros(repeat,1);

            for i = 1:repeat
                [result] = ml_clustering(X,options,'start','sample','Distance','sqeuclidean');
                F(i) = ml_Fmeasure(result.labels,labels);
            end
            means(k,1) = mean(F);
            stds(k,1) = std(F);
        end
        % Plot F-measure
        figure()
        errorbar(KRange,means(:,1),stds(:,1),'r--s');
        title('F-measure')
        ylabel('FMeasure')
        xlabel('Number of clusters')
        cluster_para{2}=[means',stds'];
    else
        [cluster_labels,cluster_para]=kmeans(X,para(1));        
    end
elseif strcmp(method,'GMM')
%% GMM clustering
    if gridSearch
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
        cluster_para=[mus,stds];
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
        end
    else 
        switch para(2)
            case 1
                obj=gmdistribution.fit(X,para(1),'Start','randSample','CovType','diagonal');
            case 2
                obj=gmdistribution.fit(X,para(1),'Start','randSample','CovType','full');
        end
        cluster_para=obj;
        post_p=[];
        for i=1:para(1)
           gm=gmdistribution(obj.mu(i,:),obj.Sigma(:,:,i),1);
           post_p=[post_p pdf(gm,X)*obj.PComponents(i)];
        end
        [Y,cluster_labels]=max(post_p,[],2);
        
    end
end

end