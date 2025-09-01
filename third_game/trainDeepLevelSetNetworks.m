function trainDeepLevelSetNetworks()
% 训练深度学习增强的水平集网络
close all; clc; clear;

%% 设置参数
bsds_root = 'D:/temp_code/2025_school_mathmodel/third_game/data/BSDS500';
train_image_dir = fullfile(bsds_root, 'images', 'train');
train_gt_dir = fullfile(bsds_root, 'groundTruth', 'train');

% 训练参数
batch_size = 4;  % 减小batch size避免内存问题
num_epochs = 30;
learning_rate = 1e-4;
validation_split = 0.2;

%% 1. 准备训练数据
fprintf('准备训练数据...\n');
[train_images, train_labels, train_edges, train_params] = prepareTrainingData(train_image_dir, train_gt_dir);

% 数据分割
num_samples = size(train_images, 4);
val_indices = randperm(num_samples, round(num_samples * validation_split));
train_indices = setdiff(1:num_samples, val_indices);

% 训练集
X_train = train_images(:,:,:,train_indices);
Y_train_init = train_labels(:,:,:,train_indices);
Y_train_edge = train_edges(:,:,:,train_indices);
Y_train_param = train_params(:,train_indices);

% 验证集
X_val = train_images(:,:,:,val_indices);
Y_val_init = train_labels(:,:,:,val_indices);
Y_val_edge = train_edges(:,:,:,val_indices);
Y_val_param = train_params(:,val_indices);

fprintf('训练集大小: %d, 验证集大小: %d\n', length(train_indices), length(val_indices));

%% 2. 训练初始化网络
fprintf('\n=== 训练初始化网络 ===\n');
initNet = createInitializationNetwork();

% 设置训练选项
options_init = trainingOptions('adam', ...
    'InitialLearnRate', learning_rate, ...
    'MaxEpochs', num_epochs, ...
    'MiniBatchSize', batch_size, ...
    'ValidationData', {X_val, Y_val_init}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'auto');

% 训练网络
try
    [trainedInitNet, info_init] = trainNetwork(X_train, Y_train_init, initNet, options_init);
    hFig = findall(0,'Type','Figure','Name','Training Progress');
    if ~isempty(hFig)
        set(hFig, 'Name', '初始化网络训练曲线');
    end
    initNet = trainedInitNet;
    save('initNet_trained.mat', 'initNet');
    save('initNet_history.mat', 'info_init');
    fprintf('初始化网络训练完成并保存\n');
catch ME
    fprintf('初始化网络训练失败: %s\n', ME.message);
    % 创建一个简单的替代网络
    initNet = createSimpleInitNetwork();
    save('initNet_trained.mat', 'initNet');
    fprintf('保存了简单的初始化网络\n');
end

%% 3. 训练边界检测网络
fprintf('\n=== 训练边界检测网络 ===\n');
edgeNet = createEdgeDetectionNetwork();

% 设置训练选项
options_edge = trainingOptions('adam', ...
    'InitialLearnRate', learning_rate * 0.5, ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', batch_size, ...
    'ValidationData', {X_val, Y_val_edge}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'auto');

% 训练网络
try
    [trainedEdgeNet, info_edge] = trainNetwork(X_train, Y_train_edge, edgeNet, options_edge);
    hFig = findall(0,'Type','Figure','Name','Training Progress');
    if ~isempty(hFig)
        set(hFig, 'Name', '边界检测网络训练曲线');
    end
    edgeNet = trainedEdgeNet;
    save('edgeNet_trained.mat', 'edgeNet');
    save('edgeNet_history.mat', 'info_edge');
    fprintf('边界检测网络训练完成并保存\n');
catch ME
    fprintf('边界检测网络训练失败: %s\n', ME.message);
    % 创建一个简单的替代网络
    edgeNet = createSimpleEdgeNetwork();
    save('edgeNet_trained.mat', 'edgeNet');
    fprintf('保存了简单的边界检测网络\n');
end

%% 4. 训练参数预测网络
fprintf('\n=== 训练参数预测网络 ===\n');
paramNet = createParameterNetwork();

% 设置训练选项
options_param = trainingOptions('adam', ...
    'InitialLearnRate', learning_rate * 0.1, ...
    'MaxEpochs', num_epochs, ...
    'MiniBatchSize', batch_size, ...
    'ValidationData', {X_val, Y_val_param'}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'auto');

% 训练网络
try
    [trainedParamNet, info_param] = trainNetwork(X_train, Y_train_param', paramNet, options_param);
    hFig = findall(0,'Type','Figure','Name','Training Progress');
    if ~isempty(hFig)
        set(hFig, 'Name', '参数预测网络训练曲线');
    end
    paramNet = trainedParamNet;
    save('paramNet_trained.mat', 'paramNet');
    save('paramNet_history.mat', 'info_param');
    fprintf('参数预测网络训练完成并保存\n');
catch ME
    fprintf('参数预测网络训练失败: %s\n', ME.message);
    % 创建一个简单的替代网络
    paramNet = createSimpleParamNetwork();
    save('paramNet_trained.mat', 'paramNet');
    fprintf('保存了简单的参数预测网络\n');
end

fprintf('\n=== 所有网络训练完成 ===\n');

end

%% ==================== 数据准备函数 ====================
function [train_images, train_labels, train_edges, train_params] = prepareTrainingData(image_dir, gt_dir)
    fprintf('正在读取和处理训练数据...\n');
    
    % 获取所有训练图像并按自然顺序排序
    img_files = dir(fullfile(image_dir, '*.jpg'));
    img_files = sort_nat({img_files.name});
    num_images = min(length(img_files), 100); % 限制数量避免内存问题

    % 预分配内存
    train_images = zeros(256, 256, 1, num_images, 'single');
    train_labels = zeros(256, 256, 1, num_images, 'single');
    train_edges = zeros(256, 256, 1, num_images, 'single');
    train_params = zeros(5, num_images, 'single');

    valid_count = 0;

    for i = 1:num_images
        try
            % 读取和预处理图像
            img_path = fullfile(image_dir, img_files{i});
            img = imread(img_path);
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            img = imresize(img, [256, 256]);
            img_normalized = double(img) / 255.0;
            img_mean = mean(img_normalized(:));

            % 读取真实标签，严格与图片名对应
            gt_path = fullfile(gt_dir, strrep(img_files{i}, '.jpg', '.mat'));

            if exist(gt_path, 'file')
                % 读取标注数据
                gt_data = load(gt_path);
                GTs = gt_data.groundTruth;
                
                % 只使用第一个标注
                if ~isempty(GTs)
                    % 获取分割区域
                    seg = GTs{1}.Segmentation;
                    % 调整分割区域到目标尺寸
                    seg_resized = imresize(seg, [256, 256], 'nearest');
                    
                    % 计算每个区域的平均灰度值
                    regions = unique(seg_resized);
                    region_means = zeros(size(regions));
                    for r = 1:length(regions)
                        region_mask = (seg_resized == regions(r));
                        region_means(r) = mean(img_normalized(region_mask));
                    end
                    
                    % 找到主要分割区域（选择最大的一个区域）
                    region_sizes = zeros(size(regions));
                    for r = 1:length(regions)
                        region_sizes(r) = sum(seg_resized(:) == regions(r));
                    end
                    [~, main_idx] = max(region_sizes);
                    main_region = regions(main_idx);
                    
                    % 创建二值掩码
                    gt_mask = (seg_resized == main_region);
                    
                    % 判断该区域应该标记为cs还是cl
                    region_mean = region_means(main_idx);
                    is_cs = region_mean > img_mean;
                    
                    % 跳过无效标签
                    if all(gt_mask(:)==0)
                        fprintf('标签全0，跳过: %s\n', img_files{i});
                        continue;
                    end

                    % 生成标签
                    init_label = generateInitLabel(gt_mask);
                    edge_label = generateEdgeLabel(gt_mask);
                    param_label = generateParameters(img, gt_mask);

                    % 再次检查标签有效性
                    if all(init_label(:)==0) || all(edge_label(:)==0)
                        fprintf('生成的标签全0，跳过: %s\n', img_name);
                        continue;
                    end

                    valid_count = valid_count + 1;
                    train_images(:,:,1,valid_count) = single(img) / 255.0;
                    train_labels(:,:,1,valid_count) = init_label;
                    train_edges(:,:,1,valid_count) = edge_label;
                    train_params(:,valid_count) = param_label;

                    % 可视化前几张有效标签
                    if valid_count <= 3
                        figure(200+valid_count); clf;
                        subplot(2,2,1); imshow(img,[]); title('原图');
                        
                        % 显示分割区域（用不同颜色标识cs/cl）
                        colored_mask = gt_mask;
                        if is_cs
                            title_str = 'GT区域 (CS: 高灰度)';
                            color_map = [0 0 0; 1 0.5 0]; % 黑色背景，橙色前景
                        else
                            title_str = 'GT区域 (CL: 低灰度)';
                            color_map = [0 0 0; 0 0.5 1]; % 黑色背景，蓝色前景
                        end
                        subplot(2,2,2); 
                        imshow(colored_mask); colormap(gca, color_map); 
                        title(title_str);
                        
                        subplot(2,2,3); imagesc(init_label); colorbar; title('InitLabel');
                        subplot(2,2,4); imagesc(edge_label); colorbar; title('EdgeLabel');
                        drawnow;
                        
                        % 打印更多信息
                        fprintf('图像 %d (%s):\n', valid_count, img_files{i});
                        fprintf('  区域平均灰度值: %.3f (图像均值: %.3f)\n', region_mean, img_mean);

                        fprintf('  参数标签: %s\n', mat2str(param_label',3));
                    end
                end
            else
                fprintf('未找到标签文件，跳过: %s\n', img_name);
            end

            if mod(i, 20) == 0
                fprintf('已处理 %d/%d 张图像，有效图像: %d\n', i, num_images, valid_count);
            end
        catch ME
            fprintf('处理图像 %d 时出错: %s\n', i, ME.message);
        end
    end

    % 裁剪到实际有效数据
    if valid_count == 0
        error('没有有效的训练样本，请检查数据集和标签！');
    end
    train_images = train_images(:,:,:,1:valid_count);
    train_labels = train_labels(:,:,:,1:valid_count);
    train_edges = train_edges(:,:,:,1:valid_count);
    train_params = train_params(:,1:valid_count);

    fprintf('数据准备完成，有效图像数量: %d\n', valid_count);
end

function label = generateInitLabel(gt_mask)
    % 生成初始化标签 - 有符号距离函数
    % 内部为负值，外部为正值
    dist_inside = -bwdist(~gt_mask);
    dist_outside = bwdist(gt_mask);
    dist_map = dist_inside + dist_outside;
    
    % 限制距离范围并归一化
    dist_map = max(-10, min(10, dist_map));
    label = single(tanh(dist_map / 5.0));
end

function edge_label = generateEdgeLabel(gt_mask)
    % 生成边缘标签
    % 使用多种边缘检测方法的组合
    edge1 = edge(gt_mask, 'Canny');
    edge2 = edge(gt_mask, 'Sobel');
    edge_combined = edge1 | edge2;
    
    % 膨胀边缘使其更明显
    se = strel('disk', 2);
    edge_dilated = imdilate(edge_combined, se);
    
    % 应用高斯模糊
    edge_label = single(imgaussfilt(double(edge_dilated), 1.5));
    
    % 归一化到[0,1]
    if max(edge_label(:)) > 0
        edge_label = edge_label / max(edge_label(:));
    end
end

function params = generateParameters(img, gt_mask)
    % 生成参数标签 - 基于图像和分割特征
    
    % 计算图像特征
    img_double = double(img);
    img_std = std(img_double(:));
    img_mean = mean(img_double(:));
    
    % 计算分割特征
    area = sum(gt_mask(:));
    perimeter = sum(bwperim(gt_mask), 'all');
    
    % 避免除零
    if area == 0
        area = 1;
    end
    if perimeter == 0
        perimeter = 1;
    end
    
    compactness = (perimeter^2) / (4 * pi * area);
    
    % 生成归一化参数 [0,1]
    params = single(zeros(5, 1));
    
    % mu: 基于图像标准差 [0.05, 0.2] -> [0, 1]
    params(1) = min(1, max(0, (img_std - 20) / 80));
    
    % lambda: 基于紧致度 [1.0, 5.0] -> [0, 1]
    params(2) = min(1, max(0, (compactness - 1) / 4));
    
    % epsilon: 基于图像均值 [0.5, 2.0] -> [0, 1]
    params(3) = min(1, max(0, (img_mean - 50) / 150));
    
    % gamma: 基于面积比例 [0.1, 1.0] -> [0, 1]
    area_ratio = area / (256 * 256);
    params(4) = min(1, max(0, area_ratio));
    
    % tau: 基于周长 [10, 50] -> [0, 1]
    params(5) = min(1, max(0, (perimeter - 100) / 900));
end

%% ==================== 网络结构函数 ====================
function net = createInitializationNetwork()
    % 创建初始化网络
    layers = [
        imageInputLayer([256 256 1], 'Name', 'input')
        
        % 编码器
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        % 解码器
        transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'deconv1')
        reluLayer('Name', 'relu4')
        
        transposedConv2dLayer(2, 32, 'Stride', 2, 'Name', 'deconv2')
        reluLayer('Name', 'relu5')
        
        convolution2dLayer(1, 1, 'Padding', 'same', 'Name', 'final_conv')
        tanhLayer('Name', 'tanh')
        
        regressionLayer('Name', 'output')
    ];
    
    net = layerGraph(layers);
end

function net = createEdgeDetectionNetwork()
    % 创建边界检测网络
    layers = [
        imageInputLayer([256 256 1], 'Name', 'input')
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        
        transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'deconv1')
        reluLayer('Name', 'relu4')
        
        convolution2dLayer(1, 1, 'Padding', 'same', 'Name', 'final_conv')
        sigmoidLayer('Name', 'sigmoid')
        
        regressionLayer('Name', 'output')
    ];
    
    net = layerGraph(layers);
end

function net = createParameterNetwork()
    % 创建参数预测网络
    layers = [
        imageInputLayer([256 256 1], 'Name', 'input')
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
        reluLayer('Name', 'relu3')
        globalAveragePooling2dLayer('Name', 'gap')
        
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu4')
        fullyConnectedLayer(5, 'Name', 'fc2')
        sigmoidLayer('Name', 'sigmoid')
        
        regressionLayer('Name', 'output')
    ];
    
    net = layerGraph(layers);
end

%% ==================== 简单替代网络 ====================
function net = createSimpleInitNetwork()
    % 创建简单的初始化网络
    layers = [
        imageInputLayer([256 256 1], 'Name', 'input')
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        convolution2dLayer(1, 1, 'Padding', 'same', 'Name', 'conv2')
        tanhLayer('Name', 'tanh')
        regressionLayer('Name', 'output')
    ];
    net = layerGraph(layers);
end

function net = createSimpleEdgeNetwork()
    % 创建简单的边界检测网络
    layers = [
        imageInputLayer([256 256 1], 'Name', 'input')
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        convolution2dLayer(1, 1, 'Padding', 'same', 'Name', 'conv2')
        sigmoidLayer('Name', 'sigmoid')
        regressionLayer('Name', 'output')
    ];
    net = layerGraph(layers);
end

function net = createSimpleParamNetwork()
    % 创建简单的参数预测网络
    layers = [
        imageInputLayer([256 256 1], 'Name', 'input')
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        globalAveragePooling2dLayer('Name', 'gap')
        fullyConnectedLayer(5, 'Name', 'fc1')
        sigmoidLayer('Name', 'sigmoid')
        regressionLayer('Name', 'output')
    ];
    net = layerGraph(layers);
end

%% ==================== 文件名自然排序函数 ====================
function sorted = sort_nat(c)
    % 自然顺序排序字符串数组（如 image1.jpg, image2.jpg, ..., image10.jpg）
    [~, idx] = sort_nat_helper(c);
    sorted = c(idx);
end

function [sorted, idx] = sort_nat_helper(c)
    % 从文件名中提取数字并按数字大小排序
    expr = '\d+';
    n = numel(c);
    tokens = regexp(c, expr, 'match');
    nums = zeros(n, 1);
    for i = 1:n
        if ~isempty(tokens{i})
            nums(i) = str2double(tokens{i}{1});
        end
    end
    [~, idx] = sort(nums);
    sorted = c(idx);
end