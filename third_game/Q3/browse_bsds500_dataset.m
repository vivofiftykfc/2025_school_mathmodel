function visualize_bsds500_sample(root_dir, split, index, annotator_idx)
% 在原图上以红色边缘叠加显示 BSDS500 某一位标注者的边界结果
% root_dir: 数据集根目录（应包含 BSDS500/data/images 和 BSDS500/data/groundTruth）
% split: 'train' | 'val' | 'test'
% index: 图像索引
% annotator_idx: 标注者索引（1~5）

if nargin < 3
    index = 1;
end
if nargin < 4
    annotator_idx = 1;
end

img_dir = fullfile(root_dir, 'BSDS500', 'images', split);
gt_dir  = fullfile(root_dir, 'BSDS500', 'groundTruth', split);

img_files = dir(fullfile(img_dir, '*.jpg'));
img_files = sort_nat({img_files.name}); % 自然顺序排序

img_path = fullfile(img_dir, img_files{index});
gt_path = fullfile(gt_dir, strrep(img_files{index}, '.jpg', '.mat'));

% 读取原始图像
img = imread(img_path);
if size(img, 3) == 1
    img = repmat(img, [1, 1, 3]); % 转为 RGB
end

% 读取标注数据
gt_data = load(gt_path);
GTs = gt_data.groundTruth;

% 获取指定标注者的边界图
boundaries = GTs{annotator_idx}.Boundaries;

% 创建红色边界图
overlay = img;
overlay(:,:,1) = uint8(overlay(:,:,1)) + uint8(255 * boundaries); % Red 通道增强
overlay(:,:,2) = uint8(overlay(:,:,2) .* uint8(~boundaries));     % Green 通道抹黑
overlay(:,:,3) = uint8(overlay(:,:,3) .* uint8(~boundaries));     % Blue 通道抹黑

figure('Name', ['Overlay GT ', num2str(annotator_idx), ': ', img_files{index}], 'NumberTitle', 'off');
imshow(overlay);
title(['Image with Red Edges (GT ', num2str(annotator_idx), ')']);

end

function sorted = sort_nat(c)
% 自然顺序排序字符串数组（如 image1.jpg, image2.jpg, ..., image10.jpg）
[~, idx] = sort_nat_helper(c);
sorted = c(idx);
end

function [sorted, idx] = sort_nat_helper(c)
expr = '\\d+';
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