function plotRadar(values, colors, index)
% plotRadar 绘制多个样本的雷达图（每列为一个样本）
%   values : 3×n 矩阵，每列为一个样本的三个指标值
%   colors : n×3 矩阵，每行定义对应样本的 RGB 颜色（取值范围[0,1]）
%   index  : 文件命名编号，保存为 output/index.png

    % 验证输入尺寸
    [numIndicators, numSamples] = size(values);
    if numIndicators ~= 3
        error('输入数据必须为3行矩阵，每列为一个样本。');
    end
    if size(colors,1) ~= numSamples || size(colors,2) ~= 3
        error('颜色矩阵维度应为 [样本数 x 3]，且颜色值应在 [0,1] 范围内。');
    end

    % 定义雷达图角度（单位：弧度），补充闭合点
    angles = [0, 2*pi/3, 4*pi/3, 0];
    
    % 创建图形窗口，背景为白色
    figure('Color','w');
    hold on;
    
    % 计算所有样本数据中最大值，用于确定参考线和网格的半径
    maxVal = max(values(:)) * 1.1;
    
    % 存储每个样本 patch 的句柄，用于图例显示
    legendHandles = gobjects(numSamples,1);
    
    % 绘制每个样本的填充区域
    for i = 1:numSamples
         % 取出第 i 个样本数据，并补充闭合点
         vals = [values(:,i)' values(1,i)];
         % 转换为笛卡尔坐标
         x = vals .* cos(angles);
         y = vals .* sin(angles);
         % 绘制填充图形，设置边界线宽，FaceAlpha 为 0.6 实现60%浓度
         legendHandles(i) = patch(x, y, colors(i,:), 'FaceAlpha', 0.6, ...
             'EdgeColor', colors(i,:), 'LineWidth', 2);
    end

    % 绘制三个参考射线（从原点到 maxVal 处，虚线）
    for k = 1:3
         xRef = [0, maxVal * cos(angles(k))];
         yRef = [0, maxVal * sin(angles(k))];
         plot(xRef, yRef, '--k', 'LineWidth', 0.5);
    end

    % 添加圆形网格，便于判断数值大小（4 个同心圆）
    numCircles = 4;
    theta = linspace(0, 2*pi, 200);
    for r = linspace(maxVal/numCircles, maxVal, numCircles)
         plot(r*cos(theta), r*sin(theta), ':', 'Color', [0.5 0.5 0.5]);
    end

    % 设置指标标签与数值显示在图形外围
    labels = {'Livability','Toughness','Wisdom'};
    for k = 1:3
         % 设置标签位置，放在比 maxVal 稍外侧的位置
         labelRadius = maxVal * 1.2;
         xLabel = labelRadius * cos(angles(k));
         yLabel = labelRadius * sin(angles(k));
         text(xLabel, yLabel, labels{k}, 'FontName', 'Cambria', 'FontAngle', 'italic', ...
              'HorizontalAlignment','center', 'FontSize', 12);
    end

    % 添加图例，默认将样本命名为 Sample 1, Sample 2, ..., 放置在右上角
    sampleNames = {'Beijing','Xiaan','Guangzhou'};
    legend(legendHandles, sampleNames, 'Location', 'northeast', 'FontName', 'Cambria', 'FontAngle', 'italic');

    % 调整坐标轴，使图形居中且比例相等
    axis equal;
    axis off;
    hold off;

    % 保存图片到当前目录下的 output 文件夹中
    folder = fullfile(pwd, 'output');
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    filename = fullfile(folder, sprintf('%d.png', index));
    print(gcf, filename, '-dpng', '-r300');
    close(gcf);
end
