%For training set
n   = 100;
v_n = round(n*.2);
t_n = round(n*.2);

y_s   = zeros(n);
y_v = zeros(v_n);
y_t = zeros(t_n);

x1 = linspace(-5,5,n);
xv = linspace(-4.9,4.9,v_n);
xt = linspace(-5,5,t_n);

% Generate Surface of Training Set
for i = 1:n;
    for j = 1:n;
        y_s(i,j) = sinc( sqrt(x1(i)^2 + x1(j)^2) );
    end
end


% Training
train_x      = zeros(2,n^2);
train_x(1,:) = rep(x1,n);
train_x(2,:) = rep(x1,n,1);
train_y      = y_s(:)';

for i = 1:v_n;
    for j = 1:v_n;
        y_v(i,j) = sinc( sqrt(xv(i)^2 + xv(j)^2) );
    end
end

% Validation
val_x       = zeros(2,v_n^2);
val_x(1,:)  = rep(xv, v_n);
val_x(2,:)  = rep(xv, v_n,1);
val_y       = y_v(:)';

for i = 1:t_n;
    for j = 1:t_n;
        y_t(i,j) = sinc( sqrt(xt(i)^2 + xt(j)^2) );
    end
end

% Test
test_x      = zeros(2,t_n^2);
test_x(1,:) = rep(xt, t_n);
test_x(2,:) = rep(xt, t_n,1);
test_y      = y_t(:)';


x = [train_x, val_x, test_x];
y = [train_y, val_y, test_y];




%net = fitnet(40, 'trainbfg'); % faster but higher mse

% neurons = 38:50;

% for neuron = neurons
    
net = fitnet(40, 'trainlm');
% net = fitnet(10, 'trainscg');
net.performParam.regularization = 1e-6;
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', 1:size(train_x,2), ...
'valInd', (size(train_x,2)+1):(size(train_x,2)+size(val_x,2)), ...
'testInd', (size(train_x,2)+size(val_x,2+1)):(size(train_x,2)+size(val_x,2+size(test_x,2))));


[net, tr] = train(net, x, y);
test_yhat = net(test_x);

tr.best_perf
% end

% Plot Function Surface
figure;
colormap(gray);
h = surf(x1,x1,y_s);
alpha(h, 0.7);
hold on;
plot3(test_x(1,:), test_x(2,:), test_yhat, 'c-*');

xlabel('x_1','Interpreter', 'Tex');
ylabel('x_2','Interpreter', 'Tex');
zlabel('y', 'Interpreter', 'Tex');
hold off;