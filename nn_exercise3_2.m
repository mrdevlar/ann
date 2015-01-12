%% Automatic Relevance Determination

% Load Dataset
load ionstart.mat;

% Generate the data set.
ndata = 100;
noise = 0.05;
x1 = rand(ndata, 1) + 0.002*randn(ndata, 1);
x2 = x1 + 0.02*randn(ndata, 1);
x3 = 0.5 + 0.2*randn(ndata, 1);
x = [x1, x2, x3];
t = sin(2*pi*x1) + noise*randn(ndata, 1);

% Plot the data and the original function.
h = figure;
plotvals = linspace(0, 1, 200)';
plot(x1, t, 'ob')
hold on
axis([0 1 -1.5 1.5])
[fx, fy] = fplot('sin(2*pi*x)', [0 1]);
plot(fx, fy, '-g', 'LineWidth', 2);
legend('data', 'function');



% Set up network parameters.
nin = 3;			% Number of inputs.
nhidden = 2;			% Number of hidden units.
nout = 1;			% Number of outputs.
aw1 = 0.01*ones(1, nin);	% First-layer ARD hyperparameters.
ab1 = 0.01;			% Hyperparameter for hidden unit biases.
aw2 = 0.01;			% Hyperparameter for second-layer weights.
ab2 = 0.01;			% Hyperparameter for output unit biases.
beta = 50.0;			% Coefficient of data error.

% Create and initialize network.
prior = mlpprior(nin, nhidden, nout, aw1, ab1, aw2, ab2);
net = mlp(nin, nhidden, nout, 'linear', prior, beta);

% Set up vector of options for the optimiser.
nouter = 2;			% Number of outer loops
ninner = 10;		        % Number of inner loops
options = zeros(1,18);		% Default options vector.
options(1) = 1;			% This provides display of error values.
options(2) = 1.0e-7;	% This ensures that convergence must occur
options(3) = 1.0e-7;
options(14) = 300;		% Number of training cycles in inner loop. 

% Train using scaled conjugate gradients, re-estimating alpha and beta.
for k = 1:nouter
  net = netopt(net, options, x, t, 'scg');
  [net, gamma] = evidence(net, x, t, ninner);
  fprintf(1, '\n\nRe-estimation cycle %d:\n', k);
  disp('The first three alphas are the hyperparameters for the corresponding');
  disp('input to hidden unit weights.  The remainder are the hyperparameters');
  disp('for the hidden unit biases, second layer weights and output unit')
  disp('biases, respectively.')
  fprintf(1, '  alpha =  %8.5f\n', net.alpha);
  fprintf(1, '  beta  =  %8.5f\n', net.beta);
  fprintf(1, '  gamma =  %8.5f\n\n', gamma);
  disp(' ')
  disp('Press any key to continue.')
  pause
end

% Plot the function corresponding to the trained network.
figure(h); hold on;
[y, z] = mlpfwd(net, plotvals*ones(1,3));
plot(plotvals, y, '-r', 'LineWidth', 2)
legend('data', 'function', 'network');


