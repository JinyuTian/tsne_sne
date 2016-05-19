function N=labelComparison(labels,cluster_labels)
%find out how many different labels there are
clustering=zeros(max(labels),max(cluster_labels));
for i=1:size(clustering,1)
    [rows,cols,vals]=find(labels==i);
    for j=1:size(clustering,2)
        temp=cluster_labels(rows);
        [temprows,tempclos,vals]=find(temp==j);
        clustering(i,j)=length(vals);
    end
end
N=sum(sum(clustering,2)-max(clustering,[],2));
end