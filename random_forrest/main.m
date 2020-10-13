% Main file.
% Define how training and validation with random forest should be carried out.
function err = main()

    synthetic_spectra = 'D:\Datasets\Synthetic_data\dataset_magnitude.mat';
    synthetic_parameter = 'D:\Datasets\Synthetic_data\dataset_parameters.mat';
    labels = ["cho", "cre", "glx", "ins", "lip", "mac", "naa"];
    nTrees = 100; % Number of trees

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Prepare dataset
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    load(synthetic_spectra);
    load(synthetic_parameter);

    % Prepare X
    X = eval('mag');
    X = squeeze(X);
    N2 = size(eval(labels(1)),2);

    % Prepare Y
    for i=1:size(labels,2)
        Y(i,1:N2) = double(eval(labels(i)));
    end
    clearvars -except X Y labels nTrees
    train_size = 1000;
    test_size = 100;

    X_train = X(1:train_size,:);
    Y_train = Y(:,1:train_size);
    X_test = X(train_size+1 : train_size+test_size, :);
    Y_test = Y(:, train_size+1 : train_size+test_size);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Train & Validate
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err = train_val(X_train, Y_train, X_test, Y_test, labels, nTrees);
end