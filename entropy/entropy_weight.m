function weights = entropy_weight(X, types, a, b)
% entropy_weight 使用熵值法计算指标权重
%   输入：
%       X       - 原始数据矩阵，每行代表一个评价对象，每列代表一个指标
%       types   - 指标类型向量，取值为：
%                   1：极大型指标（越大越好）
%                   2：极小型指标（越小越好）
%                   3：中间型指标（最佳值在区间[a, b]内）
%       a, b    - 中间型指标的区间下限和上限，仅在types中存在3时需要提供
%   输出：
%       weights - 各指标的权重向量

    [n, m] = size(X);
    
    % ==================== 数据正向化处理 ====================
    X_pos = zeros(n, m);
    for j = 1:m
        if types(j) == 1
            X_pos(:, j) = X(:, j);
        elseif types(j) == 2
            X_pos(:, j) = small_to_large(X(:, j));
        elseif types(j) == 3
            if nargin < 4
                error('对于区间型指标，请提供区间下限a和上限b。');
            end
            X_pos(:, j) = section_to_large(X(:, j), a, b);
        else
            error('未知的指标类型。');
        end
    end
    
    % ==================== 数据标准化处理 ====================
    % 极差标准化（0-1标准化）
    X_norm = zeros(n, m);
    valid_columns = true(1, m);  % 标记有效列（存在差异的列）
    
    for j = 1:m
        col_min = min(X_pos(:, j));
        col_max = max(X_pos(:, j));
        range = col_max - col_min;
        
        if range == 0
            X_norm(:, j) = 0;
            valid_columns(j) = false;  % 标记无差异列
        else
            X_norm(:, j) = (X_pos(:, j) - col_min) / range;
        end
    end
    
    % ==================== 计算概率矩阵 ====================
    p = zeros(n, m);
    for j = 1:m
        if valid_columns(j)
            col_sum = sum(X_norm(:, j));
            if col_sum == 0
                valid_columns(j) = false;
            else
                p(:, j) = X_norm(:, j) / col_sum;
            end
        end
    end
    
    % ==================== 计算熵值 ====================
    k = 1 / log(n);  % 熵值计算系数
    e = ones(1, m);  % 初始化熵值
    
    for j = 1:m
        if valid_columns(j)
            entropy_sum = 0;
            for i = 1:n
                if p(i, j) ~= 0
                    entropy_sum = entropy_sum + p(i, j) * log(p(i, j));
                end
            end
            e(j) = -k * entropy_sum;
        else
            e(j) = 1;  % 无差异列的熵设为1
        end
    end
    
    % ==================== 计算权重 ====================
    d = 1 - e;          % 计算信息效用值
    total_d = sum(d);
    
    if total_d == 0     % 处理所有指标都无差异的特殊情况
        weights = ones(1, m) / m;
        warning('所有指标没有差异，返回等权重');
    else
        weights = d / total_d;
    end
    
    % 确保输出为行向量
    weights = weights(:)';
end