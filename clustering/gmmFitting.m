function [model,cluster_labels]=gmmFitting(X,para)
%   input:
%       para=[Nr of clusters, covariance matrix type]
%       matrix type:
%           1 for full diagonal matrix, 2 for full covariance matrix
%   output: 
%       model is the gmm model


    switch para(2)
        case 1
            obj=gmdistribution.fit(X,para(1),'Start','randSample','CovType','diagonal','Regularize',0.05);
        case 2
            obj=gmdistribution.fit(X,para(1),'Start','randSample','CovType','full','Regularize',0.05);
        otherwise
            disp('Invalid covariance matrix type')
    end
    model=obj;
    post_p=[];
    for i=1:para(1)
       gm=gmdistribution(obj.mu(i,:),obj.Sigma(:,:,i),1);
       post_p=[post_p pdf(gm,X)*obj.PComponents(i)];
    end
    [Y,cluster_labels]=max(post_p,[],2);
end