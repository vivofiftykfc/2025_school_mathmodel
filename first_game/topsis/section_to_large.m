function X_new = section_to_large(X, a, b)
% section_to_large 对矩阵按列进行分段正向化转换
%   输入 X 为待处理矩阵，a、b 为区间下限和上限
%   对每一列，先计算 M = max{a - min, max - b}，再根据 x_i 所处区间进行转换

    [n, m] = size(X);
    X_new = zeros(size(X));
    for j = 1:m
        col = X(:, j);
        col_min = min(col);
        col_max = max(col);
        M = max(a - col_min, col_max - b);
        % x_i < a
        idx = col < a;
        X_new(idx, j) = 1 - (a - col(idx)) / M;
        % a <= x_i <= b
        idx = (col >= a) & (col <= b);
        X_new(idx, j) = 1;
        % x_i > b
        idx = col > b;
        X_new(idx, j) = 1 - (col(idx) - b) / M;
    end
end