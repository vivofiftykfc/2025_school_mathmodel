function D_plus = dimax(Z, weights)
% dimax 计算每个方案与理想最优解的加权距离
%   输入：
%       Z       - 标准化矩阵，每行一个方案，每列一个指标
%       weights - 权重向量，应为行向量，长度与 Z 的列数相同
%   输出：
%       D_plus  - 每个方案与理想最优解的距离

    [n, m] = size(Z);
    if length(weights) ~= m
        error('权重向量长度必须与指标个数相同');
    end
    % 计算理想最优解向量
    Zplus = max(Z, [], 1);
    D_plus = zeros(n, 1);
    for i = 1:n
        diff = (Zplus - Z(i, :)).^2;
        D_plus(i) = sqrt(sum(weights .* diff));
    end
end
