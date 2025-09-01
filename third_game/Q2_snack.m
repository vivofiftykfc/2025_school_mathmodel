% =========================================================================  
%                   Snakes：Active Contour Models - BSDS500 Testing
% =========================================================================  
% 基于原始Snake算法，增加BSDS500数据集批量测试功能
% 保持原有算法逻辑不变，添加自动化测试和性能评估
% =========================================================================  

clc; clf; clear all;

% =========================================================================  
%                      BSDS500数据集路径设置
% =========================================================================  
data_root = 'D:/temp_code/2025_school_mathmodel/third_game/data/BSDS500';
test_image_dir = fullfile(data_root,  'images', 'test');
test_gt_dir = fullfile(data_root, 'groundTruth', 'test');

% 检查路径是否存在
if ~exist(test_image_dir, 'dir')
    error('测试图像目录不存在: %s', test_image_dir);
end
if ~exist(test_gt_dir, 'dir')
    error('真实标签目录不存在: %s', test_gt_dir);
end

% 获取测试图像列表
image_files = dir(fullfile(test_image_dir, '*.jpg'));
num_images = length(image_files);

if num_images == 0
    error('在目录 %s 中未找到测试图像', test_image_dir);
end

fprintf('找到 %d 个测试图像\n', num_images);

% 初始化性能指标数组
all_accuracies = zeros(num_images, 1);
all_precisions = zeros(num_images, 1);
all_recalls = zeros(num_images, 1);
all_f1_scores = zeros(num_images, 1);
all_dices = zeros(num_images, 1);

% =========================================================================  
%                      Snake算法参数设置
% =========================================================================  
NIter = 500; % 迭代次数（减少以加快批量处理）
alpha = 0.2; beta = 0.2; gamma = 20; kappa = 0.1;
wl = 0; we = 0.4; wt = 0;
sigma = 1.0;

% =========================================================================  
%                      批量处理每个测试图像
% =========================================================================  
for img_idx = 1:num_images
    fprintf('处理图像 %d/%d: %s\n', img_idx, num_images, image_files(img_idx).name);
    
    % 读取图像
    image_path = fullfile(test_image_dir, image_files(img_idx).name);
    I = imread(image_path);
    
    % 转化为双精度型并转换为灰度
    if size(I, 3) == 3
        I = rgb2gray(I);
    end
    I = im2double(I);
    
    [row, col] = size(I);
    
    % 高斯滤波
    H = fspecial('gaussian', ceil(3*sigma), sigma);
    Igs = filter2(H, I, 'same');
    
    % 自动生成初始Snake轮廓（圆形）
    center_x = col / 2;
    center_y = row / 2;
    radius = min(row, col) / 4;
    
    % 生成圆形初始轮廓
    num_points = 50;
    theta = linspace(0, 2*pi, num_points);
    x = center_x + radius * cos(theta);
    y = center_y + radius * sin(theta);
    
    % 构建Snake环
    xy = [x; y];
    c = length(x);
    xy(:, c+1) = xy(:, 1);
    
    % 样条曲线插值
    t = 1:(c+1);
    ts = 1:0.1:(c+1);
    xys = spline(t, xy, ts);
    xs = xys(1, :)';
    ys = xys(2, :)';
    
    % =========================================================================  
    %                     Snakes算法实现部分（保持原有逻辑）
    % =========================================================================  
    
    % 图像力-线函数
    Eline = Igs;
    
    % 图像力-边函数
    [gx, gy] = gradient(Igs);
    Eedge = -1 * sqrt((gx.*gx + gy.*gy));
    
    % 图像力-终点函数
    m1 = [-1 1];   
    m2 = [-1; 1];  
    m3 = [1 -2 1];   
    m4 = [1; -2; 1];  
    m5 = [1 -1; -1 1];  
    
    cx = conv2(Igs, m1, 'same');
    cy = conv2(Igs, m2, 'same');
    cxx = conv2(Igs, m3, 'same');
    cyy = conv2(Igs, m4, 'same');
    cxy = conv2(Igs, m5, 'same');
    
    Eterm = zeros(row, col);
    for i = 1:row
        for j = 1:col
            denominator = (1 + cx(i,j)^2 + cy(i,j)^2)^1.5;
            if denominator > eps
                Eterm(i,j) = (cyy(i,j)*cx(i,j)^2 - 2*cxy(i,j)*cx(i,j)*cy(i,j) + cxx(i,j)*cy(i,j)^2) / denominator;
            end
        end
    end
    
    % 外部力
    Eext = wl*Eline + we*Eedge + wt*Eterm;
    [fx, fy] = gradient(Eext);
    
    [m, n] = size(xs);
    
    % 计算五对角状矩阵
    b(1) = beta;
    b(2) = -(alpha + 4*beta);
    b(3) = (2*alpha + 6*beta);
    b(4) = b(2);
    b(5) = b(1);
    
    A = b(1)*circshift(eye(m), 2);
    A = A + b(2)*circshift(eye(m), 1);
    A = A + b(3)*circshift(eye(m), 0);
    A = A + b(4)*circshift(eye(m), -1);
    A = A + b(5)*circshift(eye(m), -2);
    
    % 计算矩阵的逆
    [L, U] = lu(A + gamma.*eye(m));
    Ainv = inv(U) * inv(L);
    
    % 迭代计算
    for iter = 1:NIter
        % 边界检查
        xs = max(1, min(col, xs));
        ys = max(1, min(row, ys));
        
        ssx = gamma*xs - kappa*interp2(fx, xs, ys, 'linear', 0);
        ssy = gamma*ys - kappa*interp2(fy, xs, ys, 'linear', 0);
        
        % 计算snake的新位置
        xs = Ainv * ssx;
        ys = Ainv * ssy;
    end
    
    % =========================================================================  
    %                      生成分割结果并计算性能指标
    % =========================================================================  
    
    % 创建分割掩码
    segmentation_mask = poly2mask(xs, ys, row, col);
    
    % 加载真实标签
    [~, img_name, ~] = fileparts(image_files(img_idx).name);
    gt_file = fullfile(test_gt_dir, [img_name, '.mat']);
    
    if exist(gt_file, 'file')
        gt_data = load(gt_file);
        
        % BSDS500数据集可能有多个标注，取第一个
        if isfield(gt_data, 'groundTruth')
            gt_boundaries = gt_data.groundTruth{1}.Boundaries;
        else
            % 如果格式不同，尝试其他字段
            fields = fieldnames(gt_data);
            gt_boundaries = gt_data.(fields{1});
        end
        
        % 将边界转换为掩码
        if size(gt_boundaries, 1) ~= row || size(gt_boundaries, 2) ~= col
            gt_boundaries = imresize(gt_boundaries, [row, col]);
        end
        
        % 创建真实标签掩码（假设边界内为前景）
        gt_mask = imfill(gt_boundaries, 'holes');
        
        % 计算性能指标
        [accuracy, precision, recall, f1_score, dice] = calculate_metrics(segmentation_mask, gt_mask);
        
        all_accuracies(img_idx) = accuracy;
        all_precisions(img_idx) = precision;
        all_recalls(img_idx) = recall;
        all_f1_scores(img_idx) = f1_score;
        all_dices(img_idx) = dice;
        
        fprintf('图像 %d - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, Dice: %.4f\n', ...
                img_idx, accuracy, precision, recall, f1_score, dice);
    else
        fprintf('警告: 未找到图像 %s 的真实标签\n', image_files(img_idx).name);
        % 设置默认值
        all_accuracies(img_idx) = 0;
        all_precisions(img_idx) = 0;
        all_recalls(img_idx) = 0;
        all_f1_scores(img_idx) = 0;
        all_dices(img_idx) = 0;
    end
    
    % 显示结果（可选）
    if img_idx <= 5  % 只显示前5个图像的结果
        figure;
        subplot(1,3,1); imshow(I); title('原始图像');
        subplot(1,3,2); imshow(segmentation_mask); title('Snake分割结果');
        if exist('gt_mask', 'var')
            subplot(1,3,3); imshow(gt_mask); title('真实标签');
        end
        drawnow;
    end
end

% =========================================================================  
%                      输出平均性能指标
% =========================================================================  
fprintf('\n--- Average Metrics Across %d Images ---\n', num_images);
fprintf('Average Accuracy:   %.4f\n', mean(all_accuracies));
fprintf('Average Precision:  %.4f\n', mean(all_precisions));
fprintf('Average Recall:     %.4f\n', mean(all_recalls));
fprintf('Average F1-Score:   %.4f\n', mean(all_f1_scores));
fprintf('Average Dice:       %.4f\n', mean(all_dices));

% =========================================================================  
%                      性能指标计算函数
% =========================================================================  
function [accuracy, precision, recall, f1_score, dice] = calculate_metrics(predicted, ground_truth)
    % 确保输入为逻辑类型
    predicted = logical(predicted);
    ground_truth = logical(ground_truth);
    
    % 计算混淆矩阵
    TP = sum(predicted & ground_truth, 'all');
    TN = sum(~predicted & ~ground_truth, 'all');
    FP = sum(predicted & ~ground_truth, 'all');
    FN = sum(~predicted & ground_truth, 'all');
    
    % 计算性能指标
    accuracy = (TP + TN) / (TP + TN + FP + FN);
    
    if (TP + FP) > 0
        precision = TP / (TP + FP);
    else
        precision = 0;
    end
    
    if (TP + FN) > 0
        recall = TP / (TP + FN);
    else
        recall = 0;
    end
    
    if (precision + recall) > 0
        f1_score = 2 * precision * recall / (precision + recall);
    else
        f1_score = 0;
    end
    
    if (2*TP + FP + FN) > 0
        dice = 2*TP / (2*TP + FP + FN);
    else
        dice = 0;
    end
end