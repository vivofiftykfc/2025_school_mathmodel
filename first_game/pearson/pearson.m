function r = pearson(X, Y)
% pearson_corr 计算向量 X 和 Y 的皮尔逊相关系数
%   r = pearson_corr(X, Y) 返回 X 和 Y 的皮尔逊相关系数。
%
%   输入:
%     X, Y - 数值向量，长度必须相同。
%
%   输出:
%     r - 皮尔逊相关系数

    if length(X) ~= length(Y)
        error('输入向量的长度必须相同。');
    end

    meanX = mean(X);
    meanY = mean(Y);

    numerator = sum((X - meanX) .* (Y - meanY));

    denominator = sqrt(sum((X - meanX).^2) * sum((Y - meanY).^2));

    % 防止除以0的情况
    if denominator == 0
        error('分母为零，无法计算相关系数。');
    end

    r = numerator / denominator;
end
