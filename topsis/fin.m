function S = fin(D_plus, D_minus)
% fin 计算 TOPSIS 的相对接近度得分 S
%   输入：
%       D_plus  - 每个方案与理想最优解的距离
%       D_minus - 每个方案与理想最劣解的距离
%   输出：
%       S       - 相对接近度得分向量

    if length(D_plus) ~= length(D_minus)
        error('D_plus 和 D_minus 的长度必须相同');
    end
    S = D_minus ./ (D_plus + D_minus);
end
