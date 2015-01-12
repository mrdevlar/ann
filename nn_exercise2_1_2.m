%% Alphabet Recognition

%appcr1

[X,T] = prprob;
c = X(:,3);
e = X(:,5);

en = e - 0.6 * rand(35,1);
ea = e;
ea(1) = 0.5;
ea(31) = 0.5;
ea(15:20) = 0.5;
ea = ea - 0.2 * randn(35,1);
plotchar(ea)

plotchar(c);
plotchar(e);


