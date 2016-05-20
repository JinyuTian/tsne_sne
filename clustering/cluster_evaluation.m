function cluster_evaluation(Y_sne,Y_tsne,label,para)
%para is the utilization of the code: evaluation or validation

if strcmp(lower(para),'evaluation')
    %using the number of the labels to do clustering evaluation
%     numComponents_sne=max(label);
%     numComponents_tsne=max(label);

    %using the number of the labels we got from F1-measure to do clusering evaluation
    [mus_sne,stds_sne]=cluster_validation(Y_sne,label,[30 1 6]);
    [mus_tsne,std_tsne]=cluster_validation(Y_tsne,label,[30 1 6]);
    [temp,numComponents_tsne]=max(mus_tsne(3,:));
    [temp,numComponents_sne]=max(mus_sne(3,:));

    error_sne=zeros(10,1);
    error_tsne=zeros(10,1);
    for i=1:10
        [model_sne,cluster_labels_sne]=gmmFitting(Y_sne,[numComponents_sne 2]);
        [model_tsne,cluster_labels_tsne]=gmmFitting(Y_tsne,[numComponents_tsne 2]);
        error_sne(i)=labelComparison(label,cluster_labels_sne)/length(label);
        error_tsne(i)=labelComparison(label,cluster_labels_tsne)/length(label);
    end
    numComponents_sne
    numComponents_tsne
    disp('Error ratio of SNE')
    mean(error_sne)
    disp('Error ratio of tSNE')
    mean(error_tsne)
else
    % ----validation of the clustering-------
    [mus_sne,stds_sne]=cluster_validation(Y_sne,label,[20 1 6]);
    [mus_tsne,std_tsne]=cluster_validation(Y_tsne,label,[20 1 6]);
    cluster_comparison(mus_sne,stds_sne,mus_tsne,std_tsne);
end