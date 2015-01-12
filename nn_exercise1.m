
%%%%% Exercise Session 1 %%%%%

%% Set seed for consistent reproduction, sorta
rng(47)

%% Linear Case (sourced from hiddenlayer.m on toledo)
% Create linear data set
x = 0:0.05:1;
y1 = 3*x + 5*x + 8*x+ 1.5;

% Train Network with One Layer 
lin = linearlayer;                          % single output layer
lin = configure(lin, x, y1);                % define output length
lin.inputs{1}.processFcns = {};             % remove default scaling
lin.outputs{1}.processFcns = {};            % remove default scaling
lin.trainFcn = 'trainlm'                    % linear training
lin.divideFcn = 'dividetrain';              % assign all to train
[lin_trained, ~] = train(lin, x, y1);       % train
plot(x, lin_trained(x), 'b-');              % plot
[b, w] = output_layer_weights(lin_trained)  % capture bias and weights


%% Non-linear Case
x = linspace(0,1,21);                       
y = sin(0.7 * pi * x);


% Plot the Figure
figure;
plot(x,y, '*-');
xlabel('x');
ylabel('y');


% Fit a ANN with two hidden units
net = fitnet(2);
net = configure(net, x, y);
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net,x,y);

% Plot output against input
plot(x, net(x), '*-');
hold on;
plot(x, y, 'r-');
xlabel('x');
ylabel('y');
legend('Estimated', 'True');
hold off;


% Retrieve Biases and Weights from each hidden unit
[biases, weights] = hidden_layer_weights(net);
% Retrieve the transfer function from the hidden units
[fun] = hidden_layer_transfer_function(net);

% Display the Functions
disp(biases)
disp(weights)
disp(fun)

% Generate values for activation function
act_x1 = tansig(weights(1) * x + biases(1));
act_x2 = tansig(weights(2) * x + biases(2));

% Retrieve Biases and Weights from the output layer
[o_biases, o_weights] = output_layer_weights(net);
% Retrieve the output transfer function
[fn2] = output_layer_transfer_function(net);

% Display Output Functions
disp(o_biases)
disp(o_weights)
disp(fn2)

% Plot output and weights
output = (o_weights(1) * act_x1) + (o_weights(2) * act_x2) + o_biases;
figure;
colormap(gray);
plot(x,y, '*');
hold on;
plot(x, output, '-');
hold on;
plot(x,act_x1, ':');
hold on;
plot(x,act_x2, '--');
xlabel('x');
ylabel('y');
h = legend('True', 'Estimated', 'x^1_p', 'x^2_p');
set(h,'Interpreter','Tex')
hold off;

% Sufacemap plot (sourced from hiddenlayer.m)
u1s = linspace(min(act_x1), max(act_x1), 10);
u2s = linspace(min(act_x2), max(act_x2), 10);

% Evaluate the function in the output layer on the regularly spaced grid defined by u1s and u2s
outplane = zeros(length(u2s), length(u1s));
for ii=1:length(u1s)
    for jj=1:length(u2s)
        outplane(jj,ii) = o_biases + o_weights(1)*u1s(ii) + o_weights(2)*u2s(jj);
    end
end

% Plot the plane defined by the function of the output neuron
figure;
colormap(gray);
h = surf(u1s, u2s, outplane);
alpha(h, 0.7);
hold on;
plot3(act_x1, act_x2, y, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
xlabel('x^1_p','Interpreter', 'Tex');
ylabel('x^2_p','Interpreter', 'Tex');
zlabel('y_p', 'Interpreter', 'Tex');
hold off;



%% Clearing Interlude
clearvars



%% Non-Linear Case with Noise

% Example 1

n_train = 100;



























