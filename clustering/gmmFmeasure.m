function x=gmmFmeasure(cluster_labels, labels)
% This function is used to calculate the f-measure in clustering
% Input:
%     cluster_labels are the labels generated from clustering(cluster_labels>0)
%     labels are the labels of the original system, if equals to zero,
%     means this data points does not have label(labels>=0)

    if length(cluster_labels)~=length(labels)
        disp('The length of the labels and cluster_labels should be the same')
        return
    end
    M=length(cluster_labels);

    nc=max(max(labels)); %number of the classes
    nk=max(max(cluster_labels)); %number of the clusters
    if nc==0
        disp('Please label part of the data points');
        return
    end

    R=zeros(nc,nk); %recall
    P=zeros(nc,nk); %precision

    C=zeros(nc,1);  %number of points in each class
    K=zeros(nk,1);  %number of the points in each cluster
    N=zeros(nc,nk); %number of datapoint of class c belongs to cluster k

    for i=1:nc
       [V,L,temp]= find(labels==i);
       C(i)=length(temp);
    end

    for i=1:nk
       [V,L,temp]= find(cluster_labels==i);
       K(i)=length(temp);
    end
    
    for i=1:M
        if(labels(i)>0)
            N(labels(i),cluster_labels(i))=N(labels(i),cluster_labels(i))+1;
        end
    end


    for i=1:nc
        for k=1:nk
            R(i,k)=N(i,k)/C(i);
            P(i,k)=N(i,k)/K(k);
        end
    end
    R
    P
    F=zeros(nc,nk);
    for i=1:nc
        for k=1:nk
            F(i,k)=2*R(i,k)*P(i,k)/(R(i,k)+P(i,k));
        end
    end
    y=sum(C.*max(F,[],2))/M;
end

