function Z = standarder(X)
% standarder - 对矩阵每列进行归一化，计算公式:
%   z_ij = x_ij / sqrt(sum(x_ij^2)) (按列求和)
%
% 输入:
%   X - 待标准化的矩阵，每列代表一个指标
% 输出:
%   Z - 标准化后的矩阵

    [n, m] = size(X);
    Z = zeros(size(X));
    for j = 1:m
        col_norm = norm(X(:, j), 2); % 计算欧式范数
        if col_norm == 0
            Z(:, j) = X(:, j);
        else
            Z(:, j) = X(:, j) / col_norm;
        end
    end
end