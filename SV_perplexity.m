%% Real data
dataset = 'mnist' % 'mnist', 
[x, labels] = load_dataset(dataset); 
x=x(1:800,:);
labels=labels(1:800,:);

%% RUNNING T-SNE + SNE
% parameters 
initial_dim=size(x,2);
no_dims = 3;                % the dimension of the lower dimensional space
N=15;    %number of perplexity
%% Creates an array of tsne-results with their corresponding KLD
Y_tsne={};
Perplexity = linspace(5,40,N); 
for i = 1:length(Perplexity)
    [Y_tsne{i}, KLD]=tsne_mod(x, labels, no_dims, initial_dim, Perplexity(i));
end

%% SNE-results and corresponding KLD
Y_sne={};
for i = 1:length(Perplexity)
    [Y_sne{i}, KLD]=tsne_mod(x, labels, no_dims, initial_dim, Perplexity(i));
end
%% CLASSIFICATION
%training and testing data
training_ratio=[0.05:0.05:0.5];
accuMean1=[]; %log of accuracy of sne
accuMean2=[]; %log of accuracy of t-sne
per=[];
svMean=[];
K= 15;
for i=1:N
    means=zeros(2,10);
    stds=zeros(2,10);
    meanSV=zeros(2,10);
    stdSV=zeros(2,10);
    temp=zeros(4,10);
    for k=1:K
        for j=1:5
            %training and testing data
            training_ratio=0.03*k;
            [train_sne,train_tsne,test_sne,test_tsne,test_labels,train_labels]=train_test_generation(training_ratio,Y_sne{i},Y_tsne{i},labels);

            %nu-svm on sne
            model_sne=svmtrain(train_labels,train_sne,'-s 1 -t 2 -c 0.5 -g 0.1 -n 0.1'); %-s method -t kernel type -c cost -g gamma -v n-fold cross validation -n nu
            temp(3,j)=length(model_sne.SVs);
            [predicted_label_sne, accu, decision_sne]=svmpredict(test_labels,test_sne,model_sne);
            temp(1,j)=accu(1); 

            %nu-svm on t-sne
            model_tsne=svmtrain(train_labels,train_tsne,'-s 1 -t 2 -c 0.5 -g 0.01 -n 0.1'); %-s method -t kernel type -c cost -g gamma -v n-fold cross validation -n nu
            temp(4,j)=length(model_tsne.SVs);
            [predicted_label_tsne, accu, decision_tsne]=svmpredict(test_labels,test_tsne,model_tsne);
            temp(2,j)=accu(1); 
        end
        means(1,k)=mean(temp(1,:));
        means(2,k)=mean(temp(2,:));
        stds(1,k)=std(temp(1,:));
        stds(2,k)=std(temp(2,:));
        meanSV(1,k)=mean(temp(3,:));
        meanSV(2,k)=mean(temp(4,:));
        stdSV(1,k)=std(temp(3,:));
        stdSV(2,k)=std(temp(4,:));
    end
    accuMean1=[accuMean1; means(1,:)];
    accuMean2=[accuMean2; means(2,:)];
    svMean=[svMean meanSV];
    per=[per Perplexity(i)*ones(1,K)];
end
%%
figure()
plot3(svMean(1,:),per,accuMean1(:),'o','MarkerFaceColor','r')
xlabel('Number of SV')
ylabel('Perplexity')
zlabel('Accuracy')
hold on
plot3(svMean(2,:),per,accuMean2(:),'o','MarkerFaceColor','b')
legend('SNE','t-SNE')

