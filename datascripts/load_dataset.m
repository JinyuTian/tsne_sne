function [ X,labels ] = load_dataset( name )
%LOAD_DATASET Summary of this function goes here
%   Detailed explanation goes here

switch name
    case: 'iris'
        load('datafiles/iris_data.mat');
        load('datafiles/iris_labels.mat');
        X=iris_data;
        labels=iris_labels;
    case: 'mnist'
        load('datafiles/mnist2500_X.txt')
        load('datafiles/mnist2500_labels.txt')
        X=mnist2500_X;
        labels=mnist2500_labels;
    case: 'abalone'
        load('datafiles/abalone.data')
        X=abalone(:,1:8);
        labels=abalone(:,end);
    case: 'movement'
        
    case: 'COIL-20? oliviettI?'

end

end