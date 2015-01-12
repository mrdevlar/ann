%% Single Dimension Case

% Generate training, validation, test data
train_x = linspace(-5,5,1000);
train_y = sinc(train_x);
val_x = linspace(-4.8,4.8,200);
val_y = sinc(val_x);
test_x = linspace(-5,5,200);
test_y = sinc(test_x);

% Combine training and validation data
x = [train_x, val_x, test_x];
y = [train_y, val_y, test_y];


% % net = fitnet(10, 'trainbfg'); 
net = fitnet(10, 'trainlm'); % winner!
% net = fitnet(10, 'trainscg');
net.performParam.regularization = 1e-6;
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', 1:1000, ...
                         'valInd', 1001:1200, ...
                         'testInd', 1201:1400);
[net, tr] = train(net, x, y);
test_yhat = net(test_x);

tr.best_perf