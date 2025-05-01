function plotHeatmap(mat, index)
% plotHeatmap 绘制矩阵的热力图
%   mat   : 待绘制的矩阵
%   index : 文件命名编号，保存为 output/index.png
%
% 以 0 为基底，矩阵中绝对值越大颜色越“热”
% 使用自定义的蓝-白色图，并按照 300 dpi 分辨率保存图片

    figure('Color', 'w', 'Position', [100, 100, 1200, 400]);
    
    imagesc(mat);
    colorbar;

    % 设置颜色轴对称，使 0 为中性
    mAbs = max(abs(mat(:)));
    clim([-mAbs, mAbs]);
   
    colormap(cmap(256, [40 133 181]));

    % 添加横纵坐标标题，并设置字体为 Cambria, 斜体加粗
    title('Loading coefficients of the original indices in each principal component of PCA', 'FontName', 'Cambria', 'FontAngle', 'italic','FontSize',22)
    xlabel('Loading coefficients for the original metrics', 'FontName', 'Cambria', 'FontAngle', 'italic','FontSize',18);
    ylabel('Principal component numbering', 'FontName', 'Cambria', 'FontAngle', 'italic','FontSize',18);
    
    % 更换坐标轴刻度数字字体为 Cambria, 斜体加粗
    set(gca, 'FontName', 'Cambria', 'FontAngle', 'italic','FontSize',18);
    
    % 隐藏坐标轴标尺（刻度线），仅保留坐标轴数字
    set(gca, 'TickLength', [0 0]);
    
    % 保存图片到当前根目录下的 output 文件夹中
    folder = fullfile(pwd, 'output');
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    filename = fullfile(folder, sprintf('%d.png', index));
    print(gcf, filename, '-dpng', '-r300');
    close(gcf);
end
