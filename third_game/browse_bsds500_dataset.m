function browse_bsds500_dataset(root_dir, split, index)
% 显示 BSDS500 数据集中指定划分的图像和其所有标注
% root_dir: 数据集根目录（应包含 BSDS500/data/images 和 BSDS500/data/groundTruth）
% split: 'train' | 'val' | 'test'
% index: 图像索引

if nargin < 3
    index = 1;
end

img_dir = fullfile(root_dir, 'BSDS500', 'images', split);
gt_dir  = fullfile(root_dir, 'BSDS500', 'groundTruth', split);

img_files = dir(fullfile(img_dir, '*.jpg'));
img_files = sort_nat({img_files.name}); % 自然顺序排序

img_path = fullfile(img_dir, img_files{index});
gt_path = fullfile(gt_dir, strrep(img_files{index}, '.jpg', '.mat'));

% 读取原始图像
img = imread(img_path);

% 读取标注数据
gt_data = load(gt_path);
GTs = gt_data.groundTruth;

figure('Name', ['BSDS500 Sample: ', img_files{index}], 'NumberTitle', 'off');
subplot(1, numel(GTs)+1, 1);
imshow(img);
title('Original Image');

for i = 1:numel(GTs)
    seg = GTs{i}.Segmentation;
    subplot(1, numel(GTs)+1, i+1);
    imshow(label2rgb(seg));
    title(['GT ', num2str(i)]);
end

end

function sorted = sort_nat(c)
% 自然顺序排序字符串数组（如 image1.jpg, image2.jpg, ..., image10.jpg）
[~, idx] = sort_nat_helper(c);
sorted = c(idx);
end

function [sorted, idx] = sort_nat_helper(c)
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