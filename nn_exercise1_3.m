%% Non-Linear Case with Noise

% Example 1

rng(47)

% Parameters 
n_training  = 1000;          % Size of training set
p_valid     = 1;            % Validation set proportion [0,1]
n_neurons   = 40;            % Number of neurons in hidden layer
l_noise     = 0.1;          % Amount of noise
train_algo  = 'trainrp';   % Training Algorithm
c_regular   = 0.1;          % Regularization Constant

% Options, 1 == TRUE, 0 == FALSE
q_validate   = 0;
q_regular    = 0;

% True function (for MSE)
true_x = linspace(-1,1,n_training);
true_y = sin(2*pi*true_x);

% Training set
train_x = linspace(-1,1,n_training);
train_y = sin(2*pi*train_x) + l_noise*randn(size(train_x));

% MSE between noisy training set and true function
noise_mse = mean(power(true_y - train_y,2));

% Validation set
val_x = linspace(-0.9,0.9,(p_valid*n_training) );
val_y = sin(2*pi*val_x) + l_noise*randn(size(val_x));

% Combine training and validation sets
x = [train_x, val_x];
y = [train_y, val_y];

% Define the ANN
net = fitnet(n_neurons, train_algo);
net.divideFcn = 'divideind';

% Early Stopping
if q_validate == 1
    net.divideParam = struct('trainInd', 1:size(train_x,2), ...
                         'valInd', (size(train_x,2)+1):(size(val_x,2)+size(train_x,2)), ...
                         'testInd', []);
else
    net.divideParam = struct('trainInd', 1:size(train_x,2), ...
                         'valInd', [], ...
                         'testInd', []);
end

% Regularization
if q_regular == 1
    net.performParam.regularization = c_regular;
end

[net, tr] = train(net, x, y);
train_yhat = net(train_x);

figure;
colormap gray;
plot(train_x, train_y, 'g*');
hold on;
plot(train_x, train_yhat, 'b-');
plot(train_x, sin(2*pi*train_x), 'b:');
hold off;
legend('Training Set', 'Approximated Function', 'True Function');

% Generate MSE

net_error = mse(net, true_y, train_yhat);
disp(net_error);
net_error = mse(net, train_y, train_yhat);
disp(net_error);


%net.initFcn = 'initlay';
%disp(net.inputWeights{1,1})
%net.layers{1}.initFcn = 'initwb';
%net.inputWeights{1,1}.initFcn = 'randnr';