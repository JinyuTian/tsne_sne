%% Choose a datatype
clc; clear all; close all
%% Real data
dataset = 'mnist' % 'mnist', 
[x, labels] = load_dataset(dataset); 
no_dims = 2;
perp = 30;

%% use a subset of MNIST for computational reasons
labels = labels(1:250);
x = x(1:250,:);
initial_dims = size(x,2);

%% Run on data

% Ranges of     eta - learning rate for SNE
%               epsilon - initial learning rate for t-SNE
eta = linspace(0.01,3,30);
epsilon = linspace(10,1200,30);

Result_cell=cell(30,2);

for i = 1:length(eta)
    fprintf('Iteration number %d \n',i)
    fprintf('       Starting SNE \n')
    [Y_sne, ~] = LR_sne_mod( x, labels, no_dims, perp, eta(i) );
    Result_cell{i,1} = Y_sne;
    fprintf('       Starting t-SNE \n')
    [Y_tsne,~] = LR_tsne_mod( x, labels, no_dims, initial_dims, perp, epsilon(i));
                           
    Result_cell{i,2} = Y_tsne;
end
%% Plotting for visual evaluation
h = 5;
w = 3;
for i = 1:length(Result_cell)
    figure(1)
    subplot( 5,3,i)
    plot_sne(Result_cell{i,1},labels,2)
    figure(2)
    subplot(5,3,i)
    plot_sne(Result_cell{i,2},labels,2)
end

