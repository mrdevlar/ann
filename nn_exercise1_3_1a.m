%% Non-Linear Case with Noise

% Example 1

rng(47)

ns_training = [50,200,1000];
ns_neurons  = [5, 10, 20, 40];
ls_noise    = [0.1, 0.5, 1.0];


algos       = {'trainrp'; 'traingdx'; 'trainscg'; 'trainbfg'; 'trainlm'};

early_stop  = [0,1];
regular     = [0,1];
qs_regular  = [0.5, 0.1, 1e-6];

% Parameters 
n_training  = 1000;          % Size of training set
p_valid     = 1;            % Validation set proportion [0,1]
n_neurons   = 5;            % Number of neurons in hidden layer
l_noise     = 0.1 ;          % Amount of noise
train_algo  = 'trainbfg';   % Training Algorithm
c_regular   = 0.5;          % Regularization Constant

% Options, 1 == TRUE, 0 == FALSE
q_validate   = 0;
q_regular    = 0;

full_output = cell(2801,13);
% full_output(1,:) = {'n_training', 'p_valid', 'n_neurons', 'l_noise', 'train_algo', ...
%     'q_validate', 'q_regular', 'c_regular', 'net_true_error', 'net_error'};

full_output(1,:) = {'n_training', 'n_neurons', 'l_noise', 'train_algo', ...
    'q_validate', 'q_regular', 'c_regular', 'epochs', 'stop', ...
    'best_perf', 'best_vperf', 'best_tperf', 'net_true_error'};
counter = 2;



  
    if q_regular == 0 & c_regular == 0.1
        break
    end
% disp(train_algo)

disp(counter)

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

% Train
[net, tr] = train(net, x, y);
train_yhat = net(train_x);


% Generate MSE
net_true_error = mse(net, true_y, train_yhat);
% disp(net_true_error);
%net_error = mse(net, train_y, train_yhat);
% disp(net_error);

image_filename = strcat('noisy_', train_algo, '_nt_', num2str(n_training), ...
    '_nn_', num2str(n_neurons), '_ln_', strrep(num2str(l_noise), '.','-'), ...
    '_cr_', strrep(num2str(c_regular), '.','-'), '_qes_', num2str(q_validate), ...
    '_qrg_', num2str(q_regular));

% Plot
figure('Visible','off');
%figure;
colormap gray;
plot(train_x, train_y, 'g*');
hold on;
plot(train_x, train_yhat, 'b-');
plot(train_x, sin(2*pi*train_x), 'b:');
hold off;
legend('Training Set', 'Approximated Function', 'True Function');
saveas(gcf, image_filename, 'png')



% Create Output
outplace = {n_training, n_neurons, l_noise, train_algo, one2true(q_validate),...
    one2true(q_regular), c_regular, tr.num_epochs, tr.stop, ...
    tr.best_perf, tr.best_vperf, tr.best_tperf, net_true_error};

% outplace = {n_training, p_valid, n_neurons, l_noise, train_algo, ...
%    one2true(q_validate), one2true(q_regular), c_regular, net_true_error, net_error};

full_output(counter,:) = outplace;
counter = counter + 1;
disp(outplace(1,:));


%output_table = cell2table(f_data);


%disp(full_output);

%net.initFcn = 'initlay';
%disp(net.inputWeights{1,1})
%net.layers{1}.initFcn = 'initwb';
%net.inputWeights{1,1}.initFcn = 'randnr';