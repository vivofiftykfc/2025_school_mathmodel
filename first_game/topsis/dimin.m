function D_minus = dimin(Z, weights)
% dimin 计算每个方案与理想最劣解的加权距离
%   输入：
%       Z       - 标准化矩阵，每行一个方案，每列一个指标
%       weights - 权重向量，应为行向量，长度与 Z 的列数相同
%   输出：
%       D_minus - 每个方案与理想最劣解的距离

    [n, m] = size(Z);
    if length(weights) ~= m
        error('权重向量长度必须与指标个数相同');
    end
    % 计算理想最劣解向量
    Zmin = min(Z, [], 1);
    D_minus = zeros(n, 1);
    for i = 1:n
        diff = (Z(i, :) - Zmin).^2;
        D_minus(i) = sqrt(sum(weights .* diff));
    end
end
