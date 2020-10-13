% Script to train a random forest model to quantify the given number of metabolites of an MR spectrum.
% The model is trained on the given training set and validated on a given validation set.
% (c) Linus Kreitner
% (c) Dhritiman Das
% Munich, 2020

% INPUT:
% X - List of spectra. N1 x L double, N1 = number of spectra, L length of spectra
% Y - List of quantifications. M x N2, M = number of metabolites, N2 = number of spectra
% labels - List of metabolite labels. M x 1
% nTrees - Number of trees in the forest. int

% OUTPUT:
% err_rel - List of relative prediction error per metabolite. M x N2 double
% figure - Boxplot that shows the error per metabolite

function [err_rel] = train_val(X_train, Y_train, X_test, Y_test, labels, nTrees)
    num_metabolites = size(Y_train,1);
    num_tests = size(X_test,1);

    %% RUN UNIVARIATE REGRESSION
    for metabolite=1:num_metabolites
        % TRAIN
        model{metabolite} = TreeBagger(nTrees, X_train, Y_train(metabolite,:), 'Method','regression'); 
        % VALIDATE
        Est{metabolite} = predict(model{metabolite}, X_test)';
    end

    for metabolite=1:num_metabolites
        err_rel(metabolite,:) = (abs(Est{metabolite} - Y_test(metabolite,:))) ./ (abs(Y_test(metabolite,:)));
        avg_err_rel(metabolite,:) = mean(err_rel(metabolite,:),2);
        fprintf('Relative error %s: %f\n', labels(metabolite), avg_err_rel(metabolite,:));
    end
	
	figure, boxplot(err_rel','Labels',labels, 'Notch','on'), ylim([0, 1]);
end