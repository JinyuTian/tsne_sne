clc; clear all;
%% Real data
dataset = 'iris' % 'mnist', 
[x, labels] = load_dataset(dataset); 
no_dims = 2;
%% running tsne
% perp = 30;
% [N, D] = size(x)
% [Y_tsne, KLD]=tsne_mod(x, labels, no_dims, D, perp);
%% Options for classifying with nu-SVM
% Setting options for nu-SVM. Using fixed values. 
clear options
options.svm_type    = 1;    % 0: C-SVM, 1: nu-SVM
options.kernel_type = 0; %already default
options.nu          = 0.001; % nu \in (0,1) (upper-bound for misclassifications on margni and lower-bound for # of SV) for nu-SVM
options.sigma       = 0.8;    % radial basis function: exp(-gamma*|u-v|^2), gamma = 1/(2*sigma^2)

%% Setting upp indexes for CV 
% 5-fold crossvalidation
% K = 10;
% idxCV = CVsplit(N, K);
% 
% if (min(labels) < 0)|(min(labels)==1 ) 
%     fprintf('class labels has to start from 0 \n');
% else 
%     no_classes = max(labels)-min(labels)+1;
% end
% 
% [XTr, labelTr, XVal, labelVal] = getTrVal( Y_tsne, new_labels, 2, idxCV );
% %% Train a classifier
% % TSNE_model = svm_train(XTr,labelTr, options);
% 
% %model_tsne = svmtrain(labelTr,XTr,'-s 1 -t 2 -c 1 -g 0.1' );

%% Predict on test set
% [pred_label, acc, dec_values] =  svm_predict(XVal, labelVal, TSNE_model);
% % save values
% 
% ml_plot_svm_boundary( XTr, labelTr, TSNE_model, options, 'draw');


%% Predict for results on SNE and t-SNE:s depending on perplexity.
% 5-fold crossvalidation
K = 6;
idxCV = CVsplit(N, K);

if (min(labels) < 0)|(min(labels)==1 ) 
    fprintf('class labels has to start from 0 \n');
else 
    no_classes = max(labels)-min(labels)+1;
end
%% running tsne for desired ranges of perplexity
Perplexity = linspace(2,40,20); 
[N, D] = size(x);
Y_tsne_results = zeros(N,2,length(Perplexity));
for i = 1:length(Perplexity)
   Y_tsne_results(:,:,i) = tsne_mod(x, labels)
end

%% running sne for desired ranges of perplexity
Perplexity = linspace(2,40,20); 
[N, D] = size(x);
Y_sne_results = zeros(N,2,length(Perplexity));
for i = 1:length(Perplexity)
   Y_sne_results(:,:,i) = sne_mod(x, labels)
end

%% 
classification_error = zeros(K,no_classes,l);
for l = 1:length(Perplexity)
    Y_tsne = Y_tsne_results(:,:,l);
    for i = 1:no_classes
        new_labels = real(labels == i-1);
        for j = 1:K
            % Get training set and validation set
            [XTr, labelTr, XVal, labelVal] = getTrVal( Y_tsne, new_labels, j, idxCV );
            % Train a classifier
            [~, model] = svm_classifier(XTr, labelTr, options, []);
            % Predict on test set
            [pred_label, acc, dec_values] =  svm_predict(XVal, labelVal, model);
            % save values
            classification_error(j,i,l) = acc(2);
        end
    end
end
%errorbar(average_error, 0.1.*ones(size(average_error)));
mean_err =mean(mean(classification_error,2),1)
std_err = std(mean(classification_error,2),0,1)
errorbar(Perplexity, mean_err,std_err)

%%
classification_error_sne = zeros(K,no_classes,l);
for l = 1:length(Perplexity)
    Y_sne = Y_sne_results(:,:,l);
    for i = 1:no_classes
        new_labels = real(labels == i-1);
        for j = 1:K
            % Get training set and validation set
            [XTr, labelTr, XVal, labelVal] = getTrVal( Y_sne, new_labels, j, idxCV );
            % Train a classifier
            [~, model] = svm_classifier(XTr, labelTr, options, []);
            % Predict on test set
            [pred_label, acc, dec_values] =  svm_predict(XVal, labelVal, model);
            % save values
            classification_error_sne(j,i,l) = acc(2);
        end
    end
end
%errorbar(average_error, 0.1.*ones(size(average_error)));
mean_err =mean(mean(classification_error_sne,2),1)
std_err = std(mean(classification_error_sne,2),0,1)
errorbar(mean_err,std_err)
