function X_concat = taylor(A, B)
% taylor 将两个矩阵横向连接
%   要求 A 和 B 的行数相同，每一行均为特征向量

    if size(A,1) ~= size(B,1)
        error('矩阵 A 和 B 的列数必须相同');
    end
    X_concat = [A, B];
end