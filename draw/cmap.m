function cmap = cmap(n, color)
% redblue 生成蓝-白-红发散色图
%   cmap = redblue(n) 返回一个 n x 3 的颜色矩阵，n 默认为 64
%   color: 目标色，要求输入一个行向量标志其RGB值，以255的形式
%   颜色从蓝色过渡到白色再过渡到红色，0 对应白色
% 美化：将其更改为由#2885b5向白色再向#2885b5

    if nargin < 1
        n = 64;
    end

    baseColor = color / 255;

    cmap = zeros(n,3);
    for i = 1:n
        t = (i-1)/(n-1);
        if t < 0.5
            % 从蓝色到白色
            t2 = t * 2;
            cmap(i,:) = (1 - t2)*baseColor + t2*[1 1 1];
        else
            % 从白色到蓝色
            t2 = (t - 0.5)*2;
            cmap(i,:) = (1 - t2)*[1 1 1] + t2*baseColor;
        end
    end
end
