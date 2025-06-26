function X_new = small_to_large(X)
% small_to_large - 对矩阵每列进行正向化处理
% 计算公式: x_new = (max - x_i) / (max - min)
% 输入:
%   X - 待转换的矩阵，每列代表一个指标
% 输出:
%   X_new - 正向化后的矩阵

    [n, m] = size(X);
    X_new = zeros(size(X));
    for j = 1:m
        col_max = max(X(:, j));
        col_min = min(X(:, j));
        if col_max == col_min
            % 当该列数据均相等时，统一赋值为1
            X_new(:, j) = ones(n, 1);
        else
            X_new(:, j) = (col_max - X(:, j)) / (col_max - col_min);
        end
    end
end