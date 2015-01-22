N = 10;
x1 = linspace(-5,5,N);

y = zeros(N);
for i = 1:N;
    for j = 1:N;
        for k = 1:N;
            for l = 1:N;
                for m = 1:N;
                    y(i,j,k,l,m) = sinc(sqrt(x1(i)^2 + x1(j)^2 + x1(k)^2 + x1(l)^2 + x1(m)^2));
                end
            end
        end
    end
end
 
train_x = zeros(5,N*N*N*N*N);
for i = 1:N;
    for j = 1:N;
        for k = 1:N;
            for l = 1:N;
                for m = 1:N;
                    train_x(1,(i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = x1(i);
                    train_x(2,(i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = x1(j);
                    train_x(3,(i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = x1(k);
                    train_x(4,(i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = x1(l);
                    train_x(5,(i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = x1(m);
                end
            end
        end
    end
end

train_y = zeros(1,N*N*N*N*N);
for i = 1:N;
    for j = 1:N;
        for k = 1:N;
            for l = 1:N;
                for m = 1:N;
                    train_y((i-1)*N^4+(j-1)*N^3+(k-1)*N^2+(l-1)*N+m) = y(i,j,k,l,m);
                end
            end
        end
    end
end

% Remove excess parameters
varlist = {'x', 'y', 'x1'};
clear(varlist{:});
 
net = fitnet(127, 'trainscg');
net = configure(net, train_x, train_y);
net.performParam.regularization = 1e-6;

 
[net, tr] = train(net, train_x, train_y);
