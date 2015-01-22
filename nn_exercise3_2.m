%% Automatic Relevance Determination
clear all;
% Load Dataset
load ionstart.mat;
inputs = Xnorm;
targets = Y;
targets(targets==-1) = 0;

% Set up network parameters.
nin = 33;			            % Number of inputs.
nhidden = 10;			        % Number of hidden units.
nout = 1;			            % Number of outputs.
aw1 = 0.01*ones(1, nin);	% First-layer ARD hyperparameters.
ab1 = 0.01;			          % Hyperparameter for hidden unit biases.
aw2 = 0.01;			          % Hyperparameter for second-layer weights.
ab2 = 0.01;			          % Hyperparameter for output unit biases.
beta = 50.0;			        % Coefficient of data error.

% Create and initialize network.
prior = mlpprior(nin, nhidden, nout, aw1, ab1, aw2, ab2);
net = mlp(nin, nhidden, nout, 'logistic', prior, beta);

% Set up vector of options for the optimiser.
nouter = 2;			% Number of outer loops
ninner = 10;		        % Number of inner loops
options = zeros(1,18);		% Default options vector.
options(1) = 1;			% This provides display of error values.
options(2) = 1.0e-7;	% This ensures that convergence must occur
options(3) = 1.0e-7;
options(14) = 1000;		% Number of training cycles in inner loop. 

% Train using scaled conjugate gradients, re-estimating alpha and beta.
for k = 1:nouter
  net = netopt(net, options, inputs , targets, 'scg');
  [net, gamma] = evidence(net, inputs, targets, ninner);
  fprintf(1, '  alpha =  %8.5f\n', net.alpha);
  fprintf(1, '  beta  =  %8.5f\n', net.beta);
  fprintf(1, '  gamma =  %8.5f\n\n', gamma);
end

% Get the most relevant values
m_w=zeros(1,33);
for i = 1:33
    m_w(i) = mean(abs(net.w1(i,:)));
end
m_w = [m_w ; 1:33]';
m_w = sortrows(m_w,-1);
relevant = m_w(1:10,2);
r_inputs = inputs(:,relevant);

n=length(targets');

% Fit all variables
net = patternnet(10, 'trainscg');
net.layers{1}.transferFcn='logsig';
net.layers{2}.transferFcn='logsig';
%net.performParam.regularization = 1e-6;
net.divideFcn ='divideind';
net.divideParam = struct('trainInd', [1:4:n 3:4:n],...
                        'valInd', 4:4:n,...
                        'testInd', 2:4:n);
[net,tr] = train(net, inputs', targets');
disp(tr.best_perf);
disp(tr.best_tperf);


% Fit only relevant inputs
% Fit all variables
net = patternnet(10, 'trainscg');
net.layers{1}.transferFcn='logsig';
net.layers{2}.transferFcn='logsig';
%net.performParam.regularization = 1e-6;
net.divideFcn ='divideind';
net.divideParam = struct('trainInd', [1:4:n 3:4:n],...
                        'valInd', 4:4:n,...
                        'testInd', 2:4:n);
[net,tr] = train(net, r_inputs', targets');
disp(tr.best_perf);
disp(tr.best_tperf);



