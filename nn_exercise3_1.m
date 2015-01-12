%% Dimensionality Reduction by PCA Analysis %% 

% Load Dataset
load cho_dataset;


% Number of Performance Interations to Run
nperf = 20;
% Create vector for storing training iterations
output_perf = zeros(nperf,2);

% Standardized the Input values and the Output values.
[pn, p_sd] = mapstd(choInputs);
[tn, t_sd] = mapstd(choTargets);

% maxfrac shrinks the number of relevant columns from 21 (complete) to 4
[pp, p_pca] = processpca(pn, 'maxfrac', 0.001);

for i = 1:nperf
    % Store Time of the start of process
    now = cputime;


    % Get dimensions of output PCA
    [m, n] = size(pp);

    %% Define Training, Validation, Test, Apply LM algorithm %%

    % indicies for test, validation, training sets
    i_tests = 2:4:n; % Start at 2, take every 4th value until n
    i_valid = 4:4:n;
    i_train = [1:4:n 3:4:n];

    % Configure a network
    net = fitnet(5); % single layer, 5 nodes
    net.trainFcn = 'trainlm'; % Default LM Algorithm
    net.trainFcn = 'trainbr'; % Bayesian Regularization
    net.divideFcn = 'divideind';
    net.divideParam = struct('trainInd', i_train, 'valInd', i_valid, 'testInd', i_tests); % ... are just line breaks
    [net, tr] = train(net, pn, tn); % Using original data
%     [net, tr] = train(net, pp, tn); % Using PCA output
    % Get Predictions on training and test
%     yh_train = net( pp(:,i_train) );
%     yh_tests = net( pp(:,i_tests) );
    yh_train = net( pn(:,i_train) );
    yh_tests = net( pn(:,i_tests) );

    % Capture Performance of Model
    model_performance = perform(net, tn(:,i_tests), yh_tests);
    
    % Append to Output Array
    output_perf(i,1) = model_performance;
    % Store time of completed model
    later = cputime - now;
    output_perf(i,2) = later
end

performance = mean(output_perf);
disp(performance)





