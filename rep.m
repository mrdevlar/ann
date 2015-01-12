function [result]=rep(array, count, transpose)
if nargin < 3
    transpose = 0;
end



if transpose == 0
    matrix = repmat(array, count,1);
    result = matrix(:)';
elseif transpose == 1
    matrix = repmat(array, count,1)';
    result = matrix(:)';
end