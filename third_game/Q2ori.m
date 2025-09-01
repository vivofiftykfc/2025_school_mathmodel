function Q2
close all
clc,clear
%% step1, read grayscale image
Img=imread('test.jpg');
Img=rgb2gray(Img);
frm=0;

%% step2, set params (按照论文Algorithm 1)
timestep=2; % Δt = 1
mu=0.05; % μ = 0.1
lambda=25; % λ = 2
epsilon=1.0; % ε = 1
c0=2; % c0 = 2
maxiter=1002;
% 论文参数
sigma=3; % σ = 3 (论文中α参数，但这里用于高斯核)
w=5; 
tau=28; % τ = 24 (论文Algorithm 1)
gamma=0.5; % γ值需要根据实验调整，论文中未给出具体值

%% step3 smooth image with gaussian filter
G=fspecial('gaussian',2*round(3*sigma)+1,sigma);
I_sigma=conv2(double(Img),G,'same'); % 得到I_σ
figure(1);
imagesc(I_sigma,[0, 255]); axis off; axis equal; colormap(gray);
title('I_σ: Gaussian smoothed image');

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

figure(2);
imagesc(g); axis off; axis equal; colorbar;
title('g_β: Adaptive edge indicator');

%% step5, set initial phi (按照论文方法)
phi = c0 * ones(rows, cols);
% % 简单的初始化：基于图像强度
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
    
    %% 显示结果
    if mod(k,50)==1
        frm=frm+1;
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
        pause(0.1)
        
        % 保存gif
        frame = getframe(h);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if frm == 1
            imwrite(imind,cm,'femur_corrected.gif','gif', 'Loopcount',inf);
        else
            imwrite(imind,cm,'femur_corrected.gif','gif','WriteMode','append');
        end
    end
    
    % 输出调试信息
    if mod(k,100)==1
        fprintf('Iter %d: RegTerm=%.6f, EdgeTerm=%.6f, AreaTerm=%.6f\n', ...
            k, mean(abs(distRegTerm(:))), mean(abs(edgeTerm(:))), mean(abs(areaTerm(:))));
    end
end

%% 最终结果
figure(6);
imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;
contour(phi, [0,0], 'r', 'LineWidth', 2);
title(['Final segmentation result, iterations=' num2str(k)]);

% 可视化各个组件
figure(7);
subplot(2,3,1); imagesc(f_med); title('f_{med}'); colorbar;
subplot(2,3,2); imagesc(cl); title('c_l'); colorbar;
subplot(2,3,3); imagesc(cs); title('c_s'); colorbar;
subplot(2,3,4); imagesc(adaptive_sign_func); title('φ(I_σ,c_l,c_s)'); colorbar;
subplot(2,3,5); imagesc(g); title('g_β'); colorbar;
subplot(2,3,6); imagesc(I_sigma); title('I_σ'); colorbar;

end

%% 辅助函数保持不变
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