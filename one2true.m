function[bool] = one2true(num)
    if num == 1
        bool = 'True';
    elseif num == 0
        bool = 'False';
    else
        disp('Something went wrong')
    end
end