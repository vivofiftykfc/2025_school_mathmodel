function Q2
close all
clc,clear
%% step1, Set paths for BSDS500 dataset
% Please adjust these paths according to your BSDS500 dataset location
bsds_root = 'D:/temp_code/2025_school_mathmodel/third_game/data/BSDS500'; % Replace with your BSDS500 root path
image_dir = fullfile(bsds_root, 'images', 'test'); % Using test images
groundtruth_dir = fullfile(bsds_root, 'groundTruth', 'test'); % Corresponding ground truth

image_files = dir(fullfile(image_dir, '*.jpg')); % List all JPG images
num_images = length(image_files);

% Initialize arrays to store metrics for all images
all_accuracies = zeros(num_images, 1);
all_precisions = zeros(num_images, 1);
all_recalls = zeros(num_images, 1);
all_f1_scores = zeros(num_images, 1);
all_dices = zeros(num_images, 1);

%% Main loop for processing each image in BSDS500
for img_idx = 1:num_images
    img_name = image_files(img_idx).name;
    
    fprintf('Processing image %d/%d: %s\n', img_idx, num_images, img_name);
    
    % Read grayscale image
    Img = imread(fullfile(image_dir, img_name));
    if size(Img, 3) == 3
        Img = rgb2gray(Img);
    end
    
    % Load ground truth
    % BSDS500 ground truth files are .mat files containing 'groundTruth' variable
    % Each groundTruth is a struct with multiple segmentations. We take the first one.
    gt_file_name = strrep(img_name, '.jpg', '.mat');
    gt_data = load(fullfile(groundtruth_dir, gt_file_name));
    % The groundTruth variable is a cell array, each cell contains a struct
    % with .Boundaries and .Segmentation. We'll use .Segmentation.
    % We typically use the first human annotated segmentation.
    if isfield(gt_data, 'groundTruth') && ~isempty(gt_data.groundTruth)
        % Ensure the ground truth segmentation is a logical mask
        % And resize it to match the image if necessary (BSDS images are 481x321 or 321x481)
        original_gt_mask = logical(gt_data.groundTruth{1}.Segmentation);
        
        % Resize GT to match the current image dimensions if they are different
        [img_rows, img_cols] = size(Img);
        [gt_rows, gt_cols] = size(original_gt_mask);
        
        if img_rows ~= gt_rows || img_cols ~= gt_cols
            % It's generally better to resize the image to GT dimensions,
            % or ensure consistent sizing. For this example, let's resize GT.
            % However, ensure your algorithm is robust to different image sizes.
            gt_mask = imresize(original_gt_mask, [img_rows, img_cols], 'nearest');
        else
            gt_mask = original_gt_mask;
        end
    else
        warning('Ground truth not found or empty for %s. Skipping metrics for this image.', img_name);
        continue; % Skip to next image if no ground truth
    end
    
    frm=0; % Reset frame counter for GIF if you plan to save per image
    
    %% step2, set params (按照论文Algorithm 1)
    timestep=1; % Δt = 1
    mu=0.1; % μ = 0.1
    lambda=2; % λ = 2
    epsilon=1.0; % ε = 1
    c0=2; % c0 = 2
    maxiter=800; % Reduced iterations for faster testing on a dataset
    sigma=3; % σ = 3 (论文中α参数，但这里用于高斯核)
    w=3;
    tau=24; % τ = 24 (论文Algorithm 1)
    gamma=0.5; % γ值需要根据实验调整，论文中未给出具体值
    %% step3 smooth image with gaussian filter
    G=fspecial('gaussian',2*round(3*sigma)+1,sigma);
    I_sigma=conv2(double(Img),G,'same'); % 得到I_σ
    
    %% step3.5, Pre-fitting functions calculation (严格按照论文Eq. 7-8)
    disp('Starting pre-fitting calculation...');
    tic;
    [rows, cols] = size(Img);
    f_med = zeros(rows, cols);
    cl = zeros(rows, cols);
    cs = zeros(rows, cols);
    for i = 1:rows
        for j = 1:cols
            % 定义邻域Ω_x
            r_min = max(1, i-w);
            r_max = min(rows, i+w);
            c_min = max(1, j-w);
            c_max = min(cols, j+w);
            
            neighborhood = double(I_sigma(r_min:r_max, c_min:c_max));
            
            % Step 1: 计算f_med(x) = median(I(y) | y ∈ Ω_x)
            f_med(i,j) = median(neighborhood(:));
            
            % Step 2: 定义Ω_l和Ω_s (严格按照论文)
            larger_pixels = neighborhood(neighborhood > f_med(i,j));
            smaller_pixels = neighborhood(neighborhood < f_med(i,j));
            
            % Step 3: 计算c_l和c_s
            if ~isempty(larger_pixels)
                cl(i,j) = mean(larger_pixels);
            else
                cl(i,j) = f_med(i,j); % 如果没有更大的像素，使用中值
            end
            
            if ~isempty(smaller_pixels)
                cs(i,j) = mean(smaller_pixels);
            else
                cs(i,j) = f_med(i,j); % 如果没有更小的像素，使用中值
            end
        end
    end
    toc;
    disp('Pre-fitting calculation finished.');
    %% step4 calculate edge indicator according to Eq. 10-11
    % β(I) = 2S(I_σ) where S is standard deviation
    beta = 2 * std(I_sigma(:)); % Eq. 10
    % 计算∇G_σ ★ I的梯度 (更精确的方法)
    [Gx, Gy] = gradient(G);
    I_conv_Gx = conv2(double(Img), Gx, 'same');
    I_conv_Gy = conv2(double(Img), Gy, 'same');
    grad_magnitude_squared = I_conv_Gx.^2 + I_conv_Gy.^2;
    g = 1 - tanh(grad_magnitude_squared / beta); % Eq. 11: g_β
    
    %% step5, set initial phi (按照论文方法)
    phi = c0 * ones(rows, cols);
    % 简单的初始化：基于图像强度
    img_mean = mean(I_sigma(:));
    initial_mask = I_sigma > img_mean;
    phi(initial_mask) = -c0;
    
    %% 主循环
    for k=1:maxiter
        %% step6, check boundary conditions
        phi=NeumannBoundCond(phi);
        
        %% step 7 calculate differential of regularized term
        distRegTerm = distReg_p3(phi); % 使用p3函数
        
        %% step8-10 update phi according to Eq.16
        diracPhi = Dirac(phi, epsilon);
        [phi_x, phi_y] = gradient(phi);
        s = sqrt(phi_x.^2 + phi_y.^2);
        Nx = phi_x ./ (s + 1e-10);
        Ny = phi_y ./ (s + 1e-10);
        
        % 1. 长度项 (Length Term) - 基于新的g_β
        [g_x, g_y] = gradient(g);
        div_g_normal = div(g.*Nx, g.*Ny);
        edgeTerm = diracPhi .* (g_x.*Nx + g_y.*Ny + div_g_normal);
        
        % 2. 面积项 (Area Term) - 使用正确的自适应符号函数
        % φ(I_σ, c_l, c_s) = γ * arctan[(I_σ - (c_l + c_s)/2)/τ]
        adaptive_sign_func = gamma * atan((I_sigma - (cl + cs) / 2) / tau); % 修正：使用I_σ
        areaTerm = adaptive_sign_func .* g .* diracPhi;
        
        % 3. 组合演化方程 (Eq. 16)
        phi = phi + timestep * (mu * distRegTerm + lambda * edgeTerm + areaTerm);
        
        %% 显示结果 (可以根据需要注释掉，以加快批量处理)
        if mod(k,50)==1 && img_idx == 1 % Only show for the first image or comment out
            h=figure(5);
            set(gcf,'color','w');
            subplot(1,2,1);
            II=Img;
            II(:,:,2)=Img;II(:,:,3)=Img;
            imshow(II); axis off; axis equal; hold on;
            contour(phi, [0,0], 'r', 'LineWidth', 2);
            msg=['Contour result, iteration=' num2str(k)];
            title(msg);
            
            subplot(1,2,2);
            mesh(-phi);
            hold on; contour(phi, [0,0], 'r','LineWidth',2);
            view([180-30 65]);
            msg=['Level set φ, iteration=' num2str(k)];
            title(msg);
            pause(0.01)
            
            % If you want to save GIF for each image, you need to adjust file names
            % frame = getframe(h);
            % im = frame2im(frame);
            % [imind,cm] = rgb2ind(im,256);
            % if frm == 1
            %     imwrite(imind,cm,'femur_corrected.gif','gif', 'Loopcount',inf);
            % else
            %     imwrite(imind,cm,'femur_corrected.gif','gif','WriteMode','append');
            % end
            % frm = frm + 1;
        end
        
        % 输出调试信息
        if mod(k,100)==1
            fprintf('  Iter %d: RegTerm=%.6f, EdgeTerm=%.6f, AreaTerm=%.6f\n', ...
                k, mean(abs(distRegTerm(:))), mean(abs(edgeTerm(:))), mean(abs(areaTerm(:))));
        end
    end
    
    %% Final segmentation and metrics calculation
    final_segmentation = phi <= 0; % Inside the contour is the segmented region
    
    % Ensure final_segmentation is the same size as gt_mask
    if any(size(final_segmentation) ~= size(gt_mask))
        warning('Final segmentation size mismatch with ground truth. Resizing...');
        final_segmentation = imresize(final_segmentation, size(gt_mask), 'nearest');
    end
    
    % Calculate performance metrics
    metrics = calculate_segmentation_metrics(final_segmentation, gt_mask);
    
    all_accuracies(img_idx) = metrics.Accuracy;
    all_precisions(img_idx) = metrics.Precision;
    all_recalls(img_idx) = metrics.Recall;
    all_f1_scores(img_idx) = metrics.F1_Score;
    all_dices(img_idx) = metrics.Dice;
    
    fprintf('  Metrics for %s: Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f, Dice=%.4f\n', ...
        img_name, metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1_Score, metrics.Dice);
    
    % Optional: Display final result for current image
    figure(6);
    subplot(1,2,1);
    imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;
    contour(phi, [0,0], 'r', 'LineWidth', 2);
    title(sprintf('Final segmentation for %s', img_name));
    
    subplot(1,2,2);
    imshow(labeloverlay(Img, final_segmentation, 'Colormap', [1 0 0])); % Overlay segmentation
    title(sprintf('Segmentation Overlay (Dice: %.4f)', metrics.Dice));
    drawnow; % Update figure
end

%% Display average metrics
fprintf('\n--- Average Metrics Across %d Images ---\n', num_images);
fprintf('Average Accuracy:   %.4f\n', mean(all_accuracies));
fprintf('Average Precision:  %.4f\n', mean(all_precisions));
fprintf('Average Recall:     %.4f\n', mean(all_recalls));
fprintf('Average F1-Score:   %.4f\n', mean(all_f1_scores));
fprintf('Average Dice:       %.4f\n', mean(all_dices));

% Optional: Save results to a file
save('bsds_segmentation_results.mat', 'all_accuracies', 'all_precisions', ...
    'all_recalls', 'all_f1_scores', 'all_dices');

end

%% Auxiliary Functions (remain unchanged)
function f = distReg_p3(phi)
    [phi_x, phi_y] = gradient(phi);
    s = sqrt(phi_x.^2 + phi_y.^2);
    
    % Eq. 15: d_p3(s)
    dp3 = zeros(size(s));
    % case 1: s in [0, 1]
    idx1 = (s >= 0) & (s <= 1);
    s1 = s(idx1);
    dp3(idx1) = s1 + (1/pi) * sin(pi * (s1 - 1)) - 1;
    
    % case 2: s in (1, inf)
    idx2 = s > 1;
    s2 = s(idx2);
    dp3(idx2) = 1 - (2 ./ (1 + s2.^4));
    
    % 避免除零
    s_safe = s + 1e-10;
    dphi_x = dp3 .* phi_x ./ s_safe;
    dphi_y = dp3 .* phi_y ./ s_safe;
    
    f = div(dphi_x, dphi_y) + 4*del2(phi);
end
function f = div(nx,ny)
    [nxx,~]=gradient(nx);
    [~,nyy]=gradient(ny);
    f=nxx+nyy;
end
function f = Dirac(x, sigma)
    f=(1/2/sigma)*(1+cos(pi*x/sigma));
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

%% New auxiliary function for metric calculation
function metrics = calculate_segmentation_metrics(segmented_mask, ground_truth_mask)
% Calculates common segmentation performance metrics.
% Input:
%   segmented_mask: A logical mask representing the segmentation result.
%   ground_truth_mask: A logical mask representing the ground truth.
% Output:
%   metrics: A struct containing Accuracy, Precision, Recall, F1-Score, Dice.

% Ensure both masks are logical and of the same size
if ~islogical(segmented_mask)
    segmented_mask = logical(segmented_mask);
end
if ~islogical(ground_truth_mask)
    ground_truth_mask = logical(ground_truth_mask);
end

if any(size(segmented_mask) ~= size(ground_truth_mask))
    error('Mask sizes do not match for metric calculation.');
end

% Flatten the masks for easier comparison
seg_flat = segmented_mask(:);
gt_flat = ground_truth_mask(:);

% True Positives (TP): Pixels correctly identified as foreground
TP = sum(seg_flat & gt_flat);
% True Negatives (TN): Pixels correctly identified as background
TN = sum(~seg_flat & ~gt_flat);
% False Positives (FP): Pixels incorrectly identified as foreground (Type I error)
FP = sum(seg_flat & ~gt_flat);
% False Negatives (FN): Pixels incorrectly identified as background (Type II error)
FN = sum(~seg_flat & gt_flat);

% Accuracy: (TP + TN) / (TP + TN + FP + FN)
metrics.Accuracy = (TP + TN) / (TP + TN + FP + FN);

% Precision: TP / (TP + FP)
if (TP + FP) > 0
    metrics.Precision = TP / (TP + FP);
else
    metrics.Precision = 0; % Avoid division by zero
end

% Recall (Sensitivity): TP / (TP + FN)
if (TP + FN) > 0
    metrics.Recall = TP / (TP + FN);
else
    metrics.Recall = 0; % Avoid division by zero
end

% F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
if (metrics.Precision + metrics.Recall) > 0
    metrics.F1_Score = 2 * (metrics.Precision * metrics.Recall) / (metrics.Precision + metrics.Recall);
else
    metrics.F1_Score = 0; % Avoid division by zero
end

% Dice Similarity Coefficient (DSC): 2 * TP / (2 * TP + FP + FN)
% Also can be written as 2 * |A intersect B| / (|A| + |B|)
if (2 * TP + FP + FN) > 0
    metrics.Dice = (2 * TP) / (2 * TP + FP + FN);
else
    metrics.Dice = 0; % Avoid division by zero
end

end