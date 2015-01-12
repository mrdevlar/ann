%% Pema Indians Diabetes Classification

% Input Data Conversion
%import'pidstart.mat';
data = mapstd(Xnorm,0.5)';
targets = Y';
targets(targets==-1) = 0;

n_neuron = 10;

net = patternnet(n_neuron, 'trainrp');
net.performParam.regularization = 1e-6;
%net.layers{1}.transferFcn = 'logsig';
net.outputs{1}.transferFcn = 'logsig';
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', 1:616, ...
                         'valInd', [], ...
                         'testInd', 616:768);
                     

[net, tr] = train(net, data, targets);
train_yhat = net(data);
disp(n_neuron);
disp(tr.best_perf);
disp(tr.best_tperf);
