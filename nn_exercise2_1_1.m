load lasertrain.dat;
load laserpred.dat;
input = [lasertrain; laserpred];

p = 71;
nodes = 5;
%5, 6, 9,10,20


u_nrow = size(input, 1) - p;


[x, y] = time_series(input, p);

net = fitnet(nodes, 'trainscg');
net.divideFcn = 'divideind';
net.performParam.regularization = 1e-6;
net.divideParam = struct('trainInd', (1:u_nrow-100), ...
                         'valInd', (1:5:u_nrow-100), ...
                         'testInd', (u_nrow-100):u_nrow);

[net, tr] = train(net, x, y);

yhat = net(x);


disp(p);
disp(nodes);
disp(tr.best_perf);
disp(tr.best_tperf);

plot(1:u_nrow, yhat, '*-');
hold on;
plot(1:u_nrow, y, 'r-');
xlabel('x');
ylabel('y');
legend('Estimated', 'True');
hold off;

pause;


plot(1:u_nrow, yhat, '*-');
hold on;
plot(1:u_nrow, y, 'r-');
xlim([(u_nrow-100), u_nrow]);
xlabel('x');
ylabel('y');
legend('Estimated', 'True');
hold off;




