clc
clear

data = [
46735	32.8	101.7	274079	22.7	32.17	243.86	7462.26	31.48	60.75	183583	2.2	85.5	41.6	93.2
50941	32.8	102.7	378136	22.4	33.98	242.41	7559.02	33.32	62.63	156534	2.41	84.7	41.8	94.2
55400	32.1	102.3	411985	22.6	33.07	248.9	7819.31	34.31	62.23	128815	2.4	80.5	42.54	95
59982	32.1	102.4	471352	22.6	34.22	266.01	14007.14	36.32	63.83	111857	1.9	80.5	45.13	95.5
65052	32	103	526666	23.1	34.37	288.22	14083.44	38.33	65.39	105997	2.15	80.3	45.5	97
68304	32.4	102.6	493317	23.1	34.61	308.18	14162.84	33.26	54.24	100807	2.53	90.4	45.52	97.9
74416	31.7	101.1	468599	24	34.28	330.86	14871.09	35.2	56.62	99630	2.22	88.5	43.6	98.3
];

data_std = zscore(data);  %由于不清楚自带pca是否标准化过，故直接再zscore一下，反正zscore多少次结果一样

[coeff, score, latent, tsquared, explained, mu] = pca(data_std);

cumExplained = cumsum(explained);
disp('各主成分的贡献率（百分比）：');
disp(explained);
disp('累计贡献率（百分比）：');
disp(cumExplained);

% 输出每个原始指标在各主成分中的加载系数
% coeff 是一个矩阵，每列对应一个主成分，每行对应一个原始指标
disp('原始指标的加载系数：');
disp(coeff);

coeff_new = zeros(15,4);
for i = 1:15
    for e =1:4
        coeff_new(i,e) = abs(coeff(i,e)) * explained(e);
    end
end

coeff_new1 = zeros(15,4);
for i = 1:15
    for e =1:4
        coeff_new1(i,e) = abs(coeff(i,e));
    end
end

% 计算前四个主成分中各原始指标的加载系数
coeff_sum = zeros(15,1);
for i = 1:15
    for e =1:4
        coeff_sum(i) = abs(coeff(i,e)) * explained(e)  + coeff_sum(i);
    end
end

disp('原始指标的加载系数按照四个主成分的权重的加权和：');
disp(coeff_sum);

fin = [coeff_new, coeff_sum];

% 根据计算结果绘制热力图并保存，前四行为前四个主成分中原始指标占比(已经加权)，第五列为总体占比
plotHeatmap(fin.', 1);

% 根据计算结果绘制热力图并保存，前四行为前四个主成分中原始指标占比(没有加权)
plotHeatmap(coeff_new1.', 2);