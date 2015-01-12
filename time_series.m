function [x_train, y_train] = time_series(x, p)



nrow = size(x,1);
    for i = 1:nrow-p

        a_row = x(i:i+p-1, 1)';

        if i == 1
            m = [a_row];
        else    
            m = [m ; a_row];
        end
    end

x_train = m';
y_train = x(p+1:nrow,1)';



end

