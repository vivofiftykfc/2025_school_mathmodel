function browse_bsds500()
    clc; close all;
    
    % 根目录
    root_dir = 'D:\temp_code\2025_school_mathmodel\third_game\data\BSDS500';
    split = 'val';  % 可设为 'val' 或 'test'
    img_dir = fullfile(root_dir, 'images', split);
    gt_dir = fullfile(root_dir, 'groundTruth', split);
    
    % 获取文件列表
    img_files = dir(fullfile(img_dir, '*.jpg'));
    if isempty(img_files)
        error('未找到图像文件，请确认路径是否正确。');
    end
    
    % 创建删除文件夹
    deleted_img_dir = fullfile(img_dir, 'deleted');
    deleted_gt_dir  = fullfile(gt_dir, 'deleted');
    if ~exist(deleted_img_dir, 'dir'); mkdir(deleted_img_dir); end
    if ~exist(deleted_gt_dir, 'dir'); mkdir(deleted_gt_dir); end
    
    % 图形界面
    fig = figure('Name', 'BSDS500 浏览器', 'NumberTitle', 'off', 'Position', [200,200,1200,600]);

    ax1 = subplot(1,3,1); title('原图');
    ax2 = subplot(1,3,2); title('标注');
    ax3 = subplot(1,3,3); title('cl/cs 自动判别');

    % 控制变量
    idx = 1;
    total = numel(img_files);
    
    % 按钮
    uicontrol('Style', 'pushbutton', 'String', '保留', ...
        'Position', [460, 20, 80, 30], 'FontSize', 12, ...
        'Callback', @keep_image);
    
    uicontrol('Style', 'pushbutton', 'String', '删除', ...
        'Position', [560, 20, 80, 30], 'FontSize', 12, ...
        'Callback', @delete_image);
    
    % 显示第一个图像
    show_image();

    %% 显示图像
    function show_image()
        if idx > total
            msgbox('所有数据已浏览完毕。', '完成');
            close(fig);
            return;
        end
        file = img_files(idx);
        [~, name, ~] = fileparts(file.name);

        % 读取图像
        I_rgb = imread(fullfile(img_dir, file.name));
        I_gray = im2double(rgb2gray(I_rgb));
        axes(ax1); imshow(I_rgb); title(['原图: ', file.name]);

        % 加载GT
        gt_file = fullfile(gt_dir, [name, '.mat']);
        if exist(gt_file, 'file')
            S = load(gt_file);
            if isfield(S, 'groundTruth') && iscell(S.groundTruth)
                % 显示边缘
                BW = S.groundTruth{1}.Boundaries;
                axes(ax2); imshow(BW); title('标注图 (第1标记者)');
                
                % 获取Segmentation并执行自动cl/cs分析
                Seg = S.groundTruth{1}.Segmentation;
                label_mask = generate_clcs_mask(I_gray, Seg);
                
                % 显示cl/cs标记图
                cmap = [0 0 0; 0 1 0; 1 0 0]; % 0=black, 1=cl=green, 2=cs=red
                RGB = label2rgb(label_mask, cmap);
                axes(ax3); imshow(RGB); title('cl/cs 自动判别');
            else
                axes(ax2); imshow(zeros(size(I_gray))); title('标注格式异常');
                axes(ax3); imshow(zeros(size(I_gray))); title('无cl/cs');
            end
        else
            axes(ax2); imshow(zeros(size(I_gray))); title('未找到标注文件');
            axes(ax3); imshow(zeros(size(I_gray))); title('无cl/cs');
        end
    end

    %% 保留
    function keep_image(~, ~)
        idx = idx + 1;
        show_image();
    end

    %% 删除
    function delete_image(~, ~)
        file = img_files(idx);
        [~, name, ext] = fileparts(file.name);

        % 移动图像
        movefile(fullfile(img_dir, file.name), ...
                 fullfile(deleted_img_dir, file.name));
        % 移动GT
        gt_file = fullfile(gt_dir, [name, '.mat']);
        if exist(gt_file, 'file')
            movefile(gt_file, fullfile(deleted_gt_dir, [name, '.mat']));
        end

        idx = idx + 1;
        show_image();
    end
end

%% 工具函数：生成cl/cs掩码
function label_mask = generate_clcs_mask(I_gray, Seg)
    mean_gray = mean(I_gray(:));
    label_mask = zeros(size(Seg));  % 0=未标记，1=cl, 2=cs
    region_labels = unique(Seg(:));
    region_labels(region_labels == 0) = [];

    for i = 1:length(region_labels)
        rid = region_labels(i);
        mask = (Seg == rid);
        region_mean = mean(I_gray(mask));
        if region_mean >= mean_gray
            label_mask(mask) = 1;  % cl
        else
            label_mask(mask) = 2;  % cs
        end
    end
end
