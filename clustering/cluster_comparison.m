function cluster_comparison(mus_sne,stds_sne,mus_tsne,stds_tsne)
    figure()
    hold on
    errorbar(1:size(mus_sne,2),mus_sne(1,:),stds_sne(1,:),'r--s');
    errorbar(1:size(mus_sne,2),mus_sne(2,:),stds_sne(2,:),'g--s');
    errorbar(1:size(mus_tsne,2),mus_tsne(1,:),stds_tsne(1,:),'b--s');
    errorbar(1:size(mus_tsne,2),mus_tsne(2,:),stds_tsne(2,:),'k--s');
    legend('SNE_AIC','SNE_BIC','tSNE_AIC','tSNE_BIC');
    
    figure()
    hold on
    errorbar(1:size(mus_sne,2),mus_sne(3,:),stds_sne(3,:),'r--s');
    errorbar(1:size(mus_tsne,2),mus_tsne(3,:),stds_tsne(3,:),'g--s');
    legend('SNE_AIC','SNE_BIC');
end