function S = topsis(X, weights, types, a, b)
% topsis 使用TOPSIS方法对多个评价对象进行排序
%   输入：
%       X       - 原始数据矩阵，每行代表一个评价对象，每列代表一个指标
%       weights - 权重向量，长度与X的列数相同
%       types   - 指标类型向量，取值为：
%                   1：极大型指标（越大越好）
%                   2：极小型指标（越小越好）
%                   3：中间型指标（最佳值在区间[a, b]内）
%       a, b    - 中间型指标的区间下限和上限，仅在types中存在3时需要提供
%   输出：
%       S       - 每个评价对象的相对接近度得分，值越大表示越优

    % 数据正向化
    [n, m] = size(X);
    X_new = zeros(n, m);
    for j = 1:m
        if types(j) == 1
            % 极大型指标，无需处理
            X_new(:, j) = X(:, j);
        elseif types(j) == 2
            % 极小型指标，调用small_to_large函数
            X_new(:, j) = small_to_large(X(:, j));
        elseif types(j) == 3
            % 中间型指标，调用section_to_large函数
            if nargin < 5
                error('对于中间型指标，请提供区间下限a和上限b。');
            end
            X_new(:, j) = section_to_large(X(:, j), a, b);
        else
            error('未知的指标类型。');
        end
    end

    % 数据标准化
    Z = standarder(X_new);

    % 计算与理想最优解和最劣解的距离
    D_plus = dimax(Z, weights);
    D_minus = dimin(Z, weights);

    % 计算相对接近度得分
    S = fin(D_plus, D_minus);
end
