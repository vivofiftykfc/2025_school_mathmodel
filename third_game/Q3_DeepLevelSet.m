function Q3_DeepLevelSet_Fixed
close all
clc,clear

%% 检查深度学习工具箱
if ~exist('trainNetwork', 'file')
    error('需要MATLAB深度学习工具箱 (Deep Learning Toolbox)');
end

%% Step 1: 设置路径和参数
bsds_root = 'D:/temp_code/2025_school_mathmodel/third_game/data/BSDS500';
image_dir = fullfile(bsds_root, 'images', 'test');
groundtruth_dir = fullfile(bsds_root, 'groundTruth', 'test');

image_files = dir(fullfile(image_dir, '*.jpg'));
num_images = length(image_files);

% 初始化性能指标数组
all_accuracies = zeros(num_images, 1);
all_precisions = zeros(num_images, 1);
all_recalls = zeros(num_images, 1);
all_f1_scores = zeros(num_images, 1);
all_dices = zeros(num_images, 1);

%% Step 2: 构建深度学习网络
fprintf('构建深度学习网络...\n');

% 网络1: 初始化网络 (预测初始phi)
initNet = createInitializationNetwork();

% 网络2: 边界检测网络
edgeNet = createEdgeDetectionNetwork();

% 网络3: 参数预测网络
paramNet = createParameterNetwork();

% 加载预训练模型标志
use_pretrained = true;
init_net_loaded = false;
edge_net_loaded = false;
param_net_loaded = false;

% 如果已有预训练模型，加载它们
if use_pretrained && exist('initNet_trained.mat', 'file')
    try
    %     load('initNet_trained.mat', 'initNet');
    %     fprintf('已加载预训练的初始化网络\n');
    %     init_net_loaded = true;
    % catch
        fprintf('预训练的初始化网络加载失败\n');
    end
end

if use_pretrained && exist('edgeNet_trained.mat', 'file')
    try
        load('edgeNet_trained.mat', 'edgeNet');
        fprintf('已加载预训练的边界检测网络\n');
        edge_net_loaded = true;
    catch
        fprintf('预训练的边界检测网络加载失败\n');
    end
end

if use_pretrained && exist('paramNet_trained.mat', 'file')
    try
        load('paramNet_trained.mat', 'paramNet');
        fprintf('已加载预训练的参数预测网络\n');
        param_net_loaded = true;
    catch
        fprintf('预训练的参数预测网络加载失败\n');
    end
end

%% Step 3: 主循环处理每张图像
for img_idx = 1:num_images
    img_name = image_files(img_idx).name;
    fprintf('处理图像 %d/%d: %s\n', img_idx, num_images, img_name);
    
    % 读取图像
    Img = imread(fullfile(image_dir, img_name));
    if size(Img, 3) == 3
        Img = rgb2gray(Img);
    end
    
    % 加载真实标签
    gt_file_name = strrep(img_name, '.jpg', '.mat');
    gt_data = load(fullfile(groundtruth_dir, gt_file_name));
    if isfield(gt_data, 'groundTruth') && ~isempty(gt_data.groundTruth)
        original_gt_mask = logical(gt_data.groundTruth{1}.Segmentation);
        [img_rows, img_cols] = size(Img);
        [gt_rows, gt_cols] = size(original_gt_mask);
        if img_rows ~= gt_rows || img_cols ~= gt_cols
            gt_mask = imresize(original_gt_mask, [img_rows, img_cols], 'nearest');
        else
            gt_mask = original_gt_mask;
        end
    else
        warning('Ground truth not found or empty for %s. Skipping metrics for this image.', img_name);
        continue;
    end
    
    %% Step 4: 深度学习增强的预处理
    
    % 4.1 图像预处理
    img_normalized = double(Img) / 255.0;
    img_resized = imresize(img_normalized, [256, 256]);
    

    % 4.2 使用深度学习网络预测初始phi
    fprintf('  使用深度学习预测初始轮廓...\n');
    if init_net_loaded
        try
            img_input = reshape(img_resized, [256, 256, 1, 1]);
            phi_init_pred = deepLearningPredict(initNet, img_input);
            phi_init_pred = imresize(phi_init_pred, [img_rows, img_cols]);
            phi_init_pred = phi_init_pred * 4 - 2; % 调整到[-2, 2]范围
            fprintf('  深度学习初始化成功\n');
            % 可视化预测初始phi
            figure(100); clf; imagesc(phi_init_pred); colorbar; title('预测初始phi'); drawnow;
        catch ME
            fprintf('  深度学习预测失败：%s，使用传统初始化\n', ME.message);
            phi_init_pred = traditionalInitialization(img_normalized, img_rows, img_cols);
        end
    else
        fprintf('  使用传统初始化\n');
        phi_init_pred = traditionalInitialization(img_normalized, img_rows, img_cols);
    end


    % 4.3 使用深度学习增强边界检测
    fprintf('  使用深度学习增强边界检测...\n');
    if edge_net_loaded
        try
            img_input = reshape(img_resized, [256, 256, 1, 1]);
            edge_prob = deepLearningPredict(edgeNet, img_input);
            edge_prob = imresize(edge_prob, [img_rows, img_cols]);
            g_deep = 1 - edge_prob;
            fprintf('  深度学习边界检测成功\n');
            % 可视化边界概率
            figure(101); clf; imagesc(edge_prob); colorbar; title('深度学习边界概率'); drawnow;
        catch ME
            fprintf('  深度边界检测失败：%s，使用传统方法\n', ME.message);
            g_deep = [];
        end
    else
        fprintf('  使用传统边界检测\n');
        g_deep = [];
    end


    % 4.4 使用深度学习预测最优参数
    fprintf('  使用深度学习预测最优参数...\n');
    if param_net_loaded
        try
            img_input = reshape(img_resized, [256, 256, 1, 1]);
            params_pred = deepLearningPredict(paramNet, img_input);
            if length(params_pred) >= 5
                mu = 0.05 + params_pred(1) * 0.15;      % [0.05, 0.2]
                lambda = 1.0 + params_pred(2) * 4.0;    % [1.0, 5.0]
                epsilon = 0.5 + params_pred(3) * 1.5;   % [0.5, 2.0]
                gamma = 0.1 + params_pred(4) * 0.9;     % [0.1, 1.0]
                tau = 10 + params_pred(5) * 40;         % [10, 50]
                fprintf('  深度学习参数预测成功\n');
                % 打印参数预测结果
                fprintf('    mu=%.4f, lambda=%.4f, epsilon=%.4f, gamma=%.4f, tau=%.4f\n', mu, lambda, epsilon, gamma, tau);
            else
                error('参数预测输出维度不正确');
            end
        catch ME
            fprintf('  参数预测失败：%s，使用默认参数\n', ME.message);
            mu = 0.1; lambda = 2; epsilon = 1.0; gamma = 0.5; tau = 24;
        end
    else
        fprintf('  使用默认参数\n');
        mu = 0.1; lambda = 2; epsilon = 1.0; gamma = 0.5; tau = 24;
    end
    
    % 固定参数
    timestep = 1;
    c0 = 2;
    maxiter = 600;
    sigma = 3;
    w = 3;
    
    %% Step 5: 传统水平集方法（使用深度学习增强）
    
    % 5.1 高斯平滑
    G = fspecial('gaussian', 2*round(3*sigma)+1, sigma);
    I_sigma = conv2(double(Img), G, 'same');
    
    % 5.2 预拟合函数计算
    [rows, cols] = size(Img);
    f_med = zeros(rows, cols);
    cl = zeros(rows, cols);
    cs = zeros(rows, cols);
    
    for i = 1:rows
        for j = 1:cols
            r_min = max(1, i-w);
            r_max = min(rows, i+w);
            c_min = max(1, j-w);
            c_max = min(cols, j+w);
            
            neighborhood = double(I_sigma(r_min:r_max, c_min:c_max));
            f_med(i,j) = median(neighborhood(:));
            
            larger_pixels = neighborhood(neighborhood > f_med(i,j));
            smaller_pixels = neighborhood(neighborhood < f_med(i,j));
            
            if ~isempty(larger_pixels)
                cl(i,j) = mean(larger_pixels);
            else
                cl(i,j) = f_med(i,j);
            end
            
            if ~isempty(smaller_pixels)
                cs(i,j) = mean(smaller_pixels);
            else
                cs(i,j) = f_med(i,j);
            end
        end
    end
    
    % 5.3 边界指示函数
    if ~isempty(g_deep)
        % 传统方法
        beta = 2 * std(I_sigma(:));
        [Gx, Gy] = gradient(G);
        I_conv_Gx = conv2(double(Img), Gx, 'same');
        I_conv_Gy = conv2(double(Img), Gy, 'same');
        grad_magnitude_squared = I_conv_Gx.^2 + I_conv_Gy.^2;
        g_traditional = 1 - tanh(grad_magnitude_squared / beta);
        
        % 融合深度学习和传统方法
        alpha_fusion = 1;
        g = alpha_fusion * g_deep + (1 - alpha_fusion) * g_traditional;
    else
        % 仅使用传统方法
        beta = 2 * std(I_sigma(:));
        [Gx, Gy] = gradient(G);
        I_conv_Gx = conv2(double(Img), Gx, 'same');
        I_conv_Gy = conv2(double(Img), Gy, 'same');
        grad_magnitude_squared = I_conv_Gx.^2 + I_conv_Gy.^2;
        g = 1 - tanh(grad_magnitude_squared / beta);
    end
    
    % 5.4 初始化phi
    phi = phi_init_pred;
    
    % 调试：检查初始化是否合理
    fprintf('  初始phi统计：min=%.3f, max=%.3f, mean=%.3f\n', ...
        min(phi(:)), max(phi(:)), mean(phi(:)));
    
    %% Step 6: 水平集演化主循环
    for k = 1:maxiter
        % 边界条件
        phi = NeumannBoundCond(phi);
        
        % 距离正则化项
        distRegTerm = distReg_p3(phi);
        
        % 计算演化项
        diracPhi = Dirac(phi, epsilon);
        [phi_x, phi_y] = gradient(phi);
        s = sqrt(phi_x.^2 + phi_y.^2);
        Nx = phi_x ./ (s + 1e-10);
        Ny = phi_y ./ (s + 1e-10);
        
        % 长度项
        [g_x, g_y] = gradient(g);
        div_g_normal = div(g.*Nx, g.*Ny);
        edgeTerm = diracPhi .* (g_x.*Nx + g_y.*Ny + div_g_normal);
        
        % 面积项
        adaptive_sign_func = gamma * atan((I_sigma - (cl + cs) / 2) / tau);
        areaTerm = adaptive_sign_func .* g .* diracPhi;
        
        % 调试：打印边界项和面积项统计
        if k == 1 && img_idx == 1
            fprintf('  [调试] edgeTerm: min=%.4f, max=%.4f, mean=%.4f\n', min(edgeTerm(:)), max(edgeTerm(:)), mean(edgeTerm(:)));
            fprintf('  [调试] areaTerm: min=%.4f, max=%.4f, mean=%.4f\n', min(areaTerm(:)), max(areaTerm(:)), mean(areaTerm(:)));
        end
        
        % 更新phi
        phi = phi + timestep * (mu * distRegTerm + lambda * edgeTerm + areaTerm);
        
        % 显示结果（仅第一张图像）
        if mod(k,100)==1 && img_idx == 1
            figure(5);
            set(gcf,'color','w');
            subplot(1,2,1);
            II = Img;
            II(:,:,2) = Img; II(:,:,3) = Img;
            imshow(II); axis off; axis equal; hold on;
            contour(phi, [0,0], 'r', 'LineWidth', 2);
            title(['深度学习增强结果, 迭代=' num2str(k)]);
            
            subplot(1,2,2);
            mesh(-phi);
            hold on; contour(phi, [0,0], 'r','LineWidth',2);
            view([180-30 65]);
            title(['水平集函数 φ, 迭代=' num2str(k)]);
            drawnow;
        end
        
        % 调试信息
        if mod(k,200)==1
            fprintf('  迭代 %d: 正则项=%.6f, 边界项=%.6f, 面积项=%.6f\n', ...
                k, mean(abs(distRegTerm(:))), mean(abs(edgeTerm(:))), mean(abs(areaTerm(:))));
        end
        
        % 收敛检查
        if k > 100 && mod(k, 50) == 0
            change = mean(abs(timestep * (mu * distRegTerm + lambda * edgeTerm + areaTerm)), 'all');
            if change < 1e-6
                fprintf('  收敛于迭代 %d\n', k);
                break;
            end
        end
%% ==================== 训练数据标签可视化辅助 ====================
% 建议在trainDeepLevelSetNetworks.m中，训练前后可加如下调试：
% 随机抽几张图片，画出generateInitLabel、generateEdgeLabel、generateParameters的输出，确认标签没问题。
% 例如：
% figure; imagesc(generateInitLabel(gt_mask)); colorbar; title('InitLabel');
% figure; imagesc(generateEdgeLabel(gt_mask)); colorbar; title('EdgeLabel');
% disp('参数标签:'); disp(generateParameters(img, gt_mask));
    end
    
    %% Step 7: 计算性能指标
    final_segmentation = phi <= 0;
    
    if any(size(final_segmentation) ~= size(gt_mask))
        final_segmentation = imresize(final_segmentation, size(gt_mask), 'nearest');
    end
    
    metrics = calculate_segmentation_metrics(final_segmentation, gt_mask);
    
    all_accuracies(img_idx) = metrics.Accuracy;
    all_precisions(img_idx) = metrics.Precision;
    all_recalls(img_idx) = metrics.Recall;
    all_f1_scores(img_idx) = metrics.F1_Score;
    all_dices(img_idx) = metrics.Dice;
    
    fprintf('  指标: Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f, Dice=%.4f\n', ...
        metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1_Score, metrics.Dice);
    
    % 显示最终结果
    if img_idx <= 3
        figure(6+img_idx);
        subplot(2,2,1);
        imshow(Img, []); title('原始图像');
        
        subplot(2,2,2);
        imshow(gt_mask); title('真实标签');
        
        subplot(2,2,3);
        imshow(final_segmentation); title('分割结果');
        
        subplot(2,2,4);
        imshow(labeloverlay(Img, final_segmentation, 'Colormap', [1 0 0]));
        title(sprintf('覆盖显示 (Dice: %.4f)', metrics.Dice));
    end
end

%% Step 8: 显示平均性能指标
fprintf('\n--- %d张图像的平均性能指标 ---\n', num_images);
fprintf('平均准确率:   %.4f\n', mean(all_accuracies));
fprintf('平均精确率:   %.4f\n', mean(all_precisions));
fprintf('平均召回率:   %.4f\n', mean(all_recalls));
fprintf('平均F1得分:   %.4f\n', mean(all_f1_scores));
fprintf('平均Dice系数: %.4f\n', mean(all_dices));

% 保存结果
save('deep_levelset_results.mat', 'all_accuracies', 'all_precisions', ...
    'all_recalls', 'all_f1_scores', 'all_dices');

end

%% ==================== 修复的深度学习预测函数 ====================

function output = deepLearningPredict(net, input)
    % 修复的预测函数，避免与系统辨识工具箱冲突
    try
        % 确保输入是single类型
        if ~isa(input, 'single')
            input = single(input);
        end
        
        % 使用深度学习工具箱的predict函数
        output = predict(net, input);
        
        % 确保输出是double类型
        if ~isa(output, 'double')
            output = double(output);
        end
        
    catch ME
        % 如果还是有问题，尝试使用activations函数
        try
            output = activations(net, input, net.Layers(end).Name);
            if ~isa(output, 'double')
                output = double(output);
            end
        catch
            rethrow(ME);
        end
    end
end

%% ==================== 深度学习网络构建函数 ====================

function net = createInitializationNetwork()
    % 创建初始化网络，预测初始phi
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
        
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
        batchNormalizationLayer('Name', 'bn4')
        reluLayer('Name', 'relu4')
        
        % 解码器
        transposedConv2dLayer(2, 128, 'Stride', 2, 'Name', 'deconv1')
        reluLayer('Name', 'relu5')
        
        transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'deconv2')
        reluLayer('Name', 'relu6')
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv5')
        reluLayer('Name', 'relu7')
        
        convolution2dLayer(1, 1, 'Padding', 'same', 'Name', 'conv6')
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
        
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'edge_conv1')
        reluLayer('Name', 'edge_relu1')
        convolution2dLayer(1, 1, 'Padding', 'same', 'Name', 'edge_conv2')
        sigmoidLayer('Name', 'edge_sigmoid')
        
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

%% ==================== 辅助函数 ====================

function phi_init = traditionalInitialization(img, rows, cols)
    % 传统初始化方法
    c0 = 2;
    phi_init = c0 * ones(rows, cols);
    img_mean = mean(img(:));
    initial_mask = img > img_mean;
    phi_init(initial_mask) = -c0;
end

%% ==================== 保持原有的辅助函数 ====================

function f = distReg_p3(phi)
    [phi_x, phi_y] = gradient(phi);
    s = sqrt(phi_x.^2 + phi_y.^2);
    
    dp3 = zeros(size(s));
    idx1 = (s >= 0) & (s <= 1);
    s1 = s(idx1);
    dp3(idx1) = s1 + (1/pi) * sin(pi * (s1 - 1)) - 1;
    
    idx2 = s > 1;
    s2 = s(idx2);
    dp3(idx2) = 1 - (2 ./ (1 + s2.^4));
    
    s_safe = s + 1e-10;
    dphi_x = dp3 .* phi_x ./ s_safe;
    dphi_y = dp3 .* phi_y ./ s_safe;
    
    f = div(dphi_x, dphi_y) + 4*del2(phi);
end

function f = div(nx,ny)
    [nxx,~] = gradient(nx);
    [~,nyy] = gradient(ny);
    f = nxx + nyy;
end

function f = Dirac(x, sigma)
    f = (1/2/sigma)*(1+cos(pi*x/sigma));
    b = (x<=sigma) & (x>=-sigma);
    f = f.*b;
end

function g = NeumannBoundCond(f)
    [nrow,ncol] = size(f);
    g = f;
    g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
    g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
    g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);
end

function metrics = calculate_segmentation_metrics(segmented_mask, ground_truth_mask)
    if ~islogical(segmented_mask)
        segmented_mask = logical(segmented_mask);
    end
    if ~islogical(ground_truth_mask)
        ground_truth_mask = logical(ground_truth_mask);
    end
    
    if any(size(segmented_mask) ~= size(ground_truth_mask))
        error('掩码尺寸不匹配');
    end
    
    seg_flat = segmented_mask(:);
    gt_flat = ground_truth_mask(:);
    
    TP = sum(seg_flat & gt_flat);
    TN = sum(~seg_flat & ~gt_flat);
    FP = sum(seg_flat & ~gt_flat);
    FN = sum(~seg_flat & gt_flat);
    
    metrics.Accuracy = (TP + TN) / (TP + TN + FP + FN);
    
    if (TP + FP) > 0
        metrics.Precision = TP / (TP + FP);
    else
        metrics.Precision = 0;
    end
    
    if (TP + FN) > 0
        metrics.Recall = TP / (TP + FN);
    else
        metrics.Recall = 0;
    end
    
    if (metrics.Precision + metrics.Recall) > 0
        metrics.F1_Score = 2 * (metrics.Precision * metrics.Recall) / (metrics.Precision + metrics.Recall);
    else
        metrics.F1_Score = 0;
    end
    
    if (2 * TP + FP + FN) > 0
        metrics.Dice = (2 * TP) / (2 * TP + FP + FN);
    else
        metrics.Dice = 0;
    end
end