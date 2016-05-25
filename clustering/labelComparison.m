function N=labelComparison(labels,cluster_labels)
%find out how many different labels there are
if size(labels,1)<size(labels,2)
    labels=labels';
end

if size(cluster_labels,1)<size(cluster_labels,2)
    cluster_labels=cluster_labels';
end

clustering=zeros(max(cluster_labels),max(labels));
for i=1:size(clustering,1)
    [rows,cols,vals]=find(cluster_labels==i);
    temp=labels(rows);
    for j=1:size(clustering,2)
        [temprows,tempclos,vals]=find(temp==j);
        clustering(i,j)=length(temprows);
    end
end
N=sum(sum(clustering,2)-max(clustering,[],2));
end