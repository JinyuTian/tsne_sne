clc; clear all;
%% Real data
dataset = 'mnist' % 'mnist', 
[x, labels] = load_dataset(dataset); 
no_dims = 2;
%% running tsne
 perp = 30;
 [N, D] = size(x)
[Y_tsne, KLD]=tsne_mod(x, labels, no_dims, D, perp);
%% Options for classifying with nu-SVM
% Setting options for nu-SVM. Using fixed values. 
clear options
options.svm_type    = 1;    % 0: C-SVM, 1: nu-SVM
options.kernel_type = 0; %already default
options.nu          = 0.001; % nu \in (0,1) (upper-bound for misclassifications on margni and lower-bound for # of SV) for nu-SVM
options.sigma       = 0.8;    % radial basis function: exp(-gamma*|u-v|^2), gamma = 1/(2*sigma^2)

%% Setting upp indexes for CV 
%5-fold crossvalidation
%K = 10;
%idxCV = CVsplit(N, K);
% 
% if (min(labels) < 0)|(min(labels)==1 ) 
%     fprintf('class labels has to start from 0 \n');
% else 
%     no_classes = max(labels)-min(labels)+1;
% end
[pred, model] = svm_classifier(x, labels, options, []);


ml_plot_svm_boundary( x, labels, model, options, 'draw');

[XTr, labelTr, XVal, labelVal] = getTrVal( Y_tsne, labels, 2, idxCV );
%% Train a classifier
% TSNE_model = svm_train(XTr,labelTr, options);

%model_tsne = svmtrain(labelTr,XTr,'-s 1 -t 2 -c 1 -g 0.1' );

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
Perplexity = linspace(5,40,5); 
[N, D] = size(x);
Y_tsne_results = zeros(N,2,length(Perplexity));
for i = 1:length(Perplexity)
   Y_tsne_results(:,:,i) = tsne_mod(x, labels);
end

%% running sne for desired ranges of perplexity
Perplexity = linspace(2,40,5); 
[N, D] = size(x);
Y_sne_results = zeros(N,2,length(Perplexity));
for i = 1:length(Perplexity)
   Y_sne_results(:,:,i) = sne_mod(x, labels)
end

%% 
classification_error = zeros(K,no_classes,length(Perplexity));
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

%% change to multiclass, check if feasible to do adaboost for classification
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

%% Multiclass
K = 5;
idxCV = CVsplit(N, K);

supportv_count = zeros(length(Perplexity),K);
classification_error = zeros(length(Perplexity),K);
for i = 1:length(Perplexity)
    Y_tsne = Y_tsne_results(:,:,i);
    for j = 1:K
        % Get training set and validation set
        [XTr, labelTr, XVal, labelVal] = getTrVal( Y_tsne, labels, j, idxCV );
        % Train a classifier
        [~, model] = svm_classifier(XTr, labelTr, options, []);
        % Predict on test set
        [pred_label, acc, dec_values] =  svm_predict(XVal, labelVal, model);
        % save values
        classification_error(i,j) = acc(1);
        supportv_count(i,j) = model.totalSV;
    end
end
mean_err = mean(classification_error,2);
std_err = std(classification_error,0,2);
errorbar(Perplexity,mean_err,std_err)

figure()
SV_mean = mean(supportv_count,2);
sv_std_err = std(supportv_count,0,2);
errorbar(Perplexity,SV_mean, std_err)

%% SNE multiclass SV and ..

K = 5;
idxCV = CVsplit(N, K);

supportv_count = zeros(length(Perplexity),K);
classification_error = zeros(length(Perplexity),K);
for i = 1:length(Perplexity)
    Y_sne = Y_sne_results(:,:,i);
    for j = 1:K
        % Get training set and validation set
        [XTr, labelTr, XVal, labelVal] = getTrVal( Y_sne, labels, j, idxCV );
        % Train a classifier
        [~, model] = svm_classifier(XTr, labelTr, options, []);
        % Predict on test set
        [pred_label, acc, dec_values] =  svm_predict(XVal, labelVal, model);
        % save values
        classification_error(i,j) = acc(1);
        supportv_count(i,j) = model.totalSV;
    end
end
mean_err = mean(classification_error,2);
std_err = std(classification_error,0,2);
errorbar(Perplexity,mean_err,std_err)

figure()
SV_mean = mean(supportv_count,2);
sv_std_err = std(supportv_count,0,2);
errorbar(Perplexity,SV_mean, std_err)


%% GentleBoost with maxiterations

K = 3;
idxCV = CVsplit(N, K);

clear options
options.weaklearner  = 0;
options.epsi         = 0.1;
options.lambda       = 1e-2;
options.max_ite      = 100;
options.T            = 2;

train_error =  zeros(length(Perplexity),K);
class_error =  zeros(length(Perplexity),K);
classification_error = zeros(length(Perplexity),K);
for i = 1:length(Perplexity)
    Y_tsne = Y_tsne_results(:,:,i);
    for j = 1:K
        % Get training set and validation set
        [XTr, labelTr, XVal, labelVal] = getTrVal( Y_tsne, labels, j, idxCV );
        % Train a classifier
        [Ntrain, d] = size(XTr);
        [Ntest, d] = size(XVal);

        model = gentleboost_model(XTr' , labelTr' , options);

        % Predict on test set
        [labelVal_est, fxtrain]     = gentleboost_predict(XVal' , model);
        [labelTr_est, fxtrain]      = gentleboost_predict(XTr' , model);
       
        % save values
        train_error(i,K)        = sum(labelTr_est~=labelTr')/Ntrain;
        class_error(i,K)        = sum(labelVal_est~=labelVal')/Ntest;
    end
end

mean_err = mean(train_error,2);
std_err = std(train_error,0,2);
errorbar(Perplexity,mean_err,std_err)
hold on;
mean_err = mean(class_error,2);
std_err = std(class_error,0,2);
errorbar(Perplexity,mean_err,std_err)