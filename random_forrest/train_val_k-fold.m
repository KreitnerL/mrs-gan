% Script to train a random forest on MRS data using k-fold cross validation
% (c) Linus Kreitner
% (c) Dhritiman Das
% Munich, 2020

% INPUT:
% X - List of spectra. N1 x L double, N1 = number of spectra, L length of spectra
% Y - List of quantifications. M x N2, M = number of metabolites, N2 = number of spectra
% labels - List of metabolite labels. M x 1
% nTrees - Number of trees in the forest. int

% OUTPUT:
% avg_est - M x 1 double. The mean error per metabolite
function [avg_est] = trainRF(X, Y, nTrees)

    num_metabolites = size(Y,1)

    %% Prepare for k-fold cross validation
    k = 20 % for k-fold cross validation
    N = size(X,1) % compute total size of dictionary
    c = cvpartition(N, 'KFold', k) % create k disjoint partitions

    %% RUN UNIVARIATE REGRESSION
    for i = 1:k
        for metabolite=1:num_metabolites
            % TRAIN 
            model{metabolite} = TreeBagger(nTrees, X(training(c,i), :), Y(metabolite,test(c,i), :)), 'Method','regression'); 
            % VALIDATE
            Est{metabolite}{i} = predict(model{metabolite}{i}, X(test(c,i), :))';
        end
    end

    %TODO
    for metabolite=1:num_metabolites
        avg_est{metabolite} = mean(avg_est{metabolite}, 2);
    end
end
