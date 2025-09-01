import pandas as pd
import numpy as np
from datetime import datetime
from chardet import detect
from draw_utils import *
import warnings

def get_model_eval_df():
    """
        用来做第一题评价模型构建，获取各模型指标

        输出：
        df
    """
    # 模型名称
    models = ['BC', 'RSF', 'LIF', 'LPF', 'RSF&LoG', 'OLPFI', 'PBC', 'LPF&FCM',
              'LGJD', 'ABC', 'PFE', 'APFJD']

    # 各项指标数据（顺序：T均值, N均值, IOU均值, IOU方差）
    data = {
        'T_mean': [8.05775, 14.152625, 7.6465, 7.347375, 7.7845, 4.825875, 5.21475, 4.224625, 5.75925, 5.97825, 4.041625, 2.83425],
        'N_mean': [161.25, 315, 181.25, 298.75, 197.5, 145.625, 219.375, 228.125, 233.75, 202.5, 208.75, 145],
        'IOU_mean': [0.767625, 0.56675, 0.6595, 0.62475, 0.735875, 0.745875, 0.779125, 0.840125, 0.65775, 0.773125, 0.875125, 0.83975],
        'IOU_var': [0.0200111, 0.0812382, 0.0655789, 0.0609779, 0.0291961, 0.0346793, 0.0284090, 0.0143390, 0.0607856, 0.0466050, 0.0029538, 0.0112819]
    }

    # 创建 DataFrame，索引为模型名称
    df = pd.DataFrame(data, index=models)
    print("=" * 20)
    print("第一问数据加载完毕，指标为")
    print(df)

    return df

def entropy_weight_topsis(df, column_big_better, column_small_better,
                          expert_weight_ratio: float = 0.0,
                          expert_weights: dict = None):
    """
    熵权法 + TOPSIS 综合评分
    暂只支持负向指标与正向指标。其他的可以继续写拓展。

    输入：
    df : DataFrame，数据帧
    column_big_better : list，正向指标列
    column_small_better : list，负向指标列
    expert_weight_ratio : float，可选，专家权重占最终权重的比例（0~1）
    expert_weights : dict，可选，形如 {'指标名1': 权重1, '指标名2': 权重2, ...} 的专家权重

    输出：
    一维 numpy 数组，表示每一行的综合评分
    """
    df_norm = pd.DataFrame(index=df.index)

    # 0. 检查数据正确性（是否最大值与最小值相等）
    for column in df.columns:
        col_max = df[column].max()
        col_min = df[column].min()

        if col_max == col_min:
            # 抛出带有详细信息的异常
            raise ValueError(
                f"错误！列 '{column}' 的最大值({col_max})与最小值({col_min})相等，无法归一化。"
                f"\n请检查数据质量或删除该列。"
            )

    # 1. 归一化（负向指标，越小越好）
    for column in df.columns:
        col_max = df[column].max()
        col_min = df[column].min()
        if column in column_small_better:
            # 负向指标归一化（值越小越好）
            df_norm[column] = (col_max - df[column]) / (col_max - col_min)
            print(f"列{column}为负向指标，已正向归一化")
        elif column in column_big_better:
            # 正向指标归一化（值越大越好）
            df_norm[column] = (df[column] - col_min) / (col_max - col_min)
            print(f"列{column}为正向指标，已归一化")
        else:
            print(f"{column}为不支持的数据类型或未设置该指标处理方式，，未对{column}列数据进行处理，可以进一步改变原熵权法函数或检查是否传入完全的参数")

        # 2. 熵权法
    P = df_norm / (df_norm.sum(axis=0))  # axis=0的意思为沿垂直方向（按列）计算
    E = -np.nansum(P * np.log(P + 1e-9), axis=0) / np.log(len(df_norm))
    d = 1 - E
    ew_weights = d / d.sum()

    indicators = df.columns.tolist()
    ew_weight_dict = dict(zip(indicators, ew_weights))

    print("原始全部指标权重（EWM）:\n", ew_weight_dict)

        # 3. 融合专家权重（若提供)
    if expert_weight_ratio > 0 and expert_weights is not None:
        indicators = df_norm.columns.tolist()
        expert_vector = np.array([expert_weights[ind] for ind in indicators])
        final_weights = (1 - expert_weight_ratio) * ew_weights + expert_weight_ratio * expert_vector
        weight_dict = dict(zip(indicators, final_weights))
    else:
        final_weights = ew_weights
        weight_dict = ew_weight_dict

    print("最终指标权重（EWM）:\n", weight_dict)

        # 4. 构建加权标准化矩阵（确保列名匹配）
    V = df_norm.copy()
    for col in V.columns:
        V[col] = V[col] * weight_dict[col]

    # 4. TOPSIS 理想解
    A_pos = V.max()
    A_neg = V.min()

    # 5. 计算距离
    D_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    print(f"\n已算出欧氏距离，D_pos={D_pos}\nD_neg={D_neg}")

    # 6. 计算综合得分
    y = D_neg / (D_pos + D_neg + 1e-16)  # 防止除以0

    print("\n熵权法-TOPSIS 分析结果")
    print("=" * 10)
    print(f"处理列数: {len(df_norm.columns)}")
    print(f"处理行数: {len(df)}")
    print("\n指标权重:")
    for col, weight in weight_dict.items():
        indicator_type = "负向" if col in column_small_better else "正向"
        print(f"  {col:<10} [{indicator_type}指标]: {weight:.6f}")
    print(f"最终打分：\n{y}")

    return y

def corr_get_save(df, output_dir="output/corr", method="pearson", show=True):
    """
    计算并保存 DataFrame 中每对数值特征之间的相关系数。

    输入:
    df : pandas.DataFrame
        仅包含数值特征的 DataFrame
    output_dir : str
        保存结果的文件夹路径，默认 'output/corr'
    method : str
        相关系数类型，可选 'pearson', 'spearman', 'kendall'

    返回:
    corr_df : pandas.DataFrame
        特征间相关系数矩阵
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    if os.path.exists(output_dir):
        print(f"输出文件夹已存在，将输出到 {output_dir} 下")
    else:
        os.makedirs(output_dir)
        print(f"输出文件夹不存在，已创建，将输出到 {output_dir} 下")

    # 生成文件名
    timemark = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"corr_{method}_{timemark}.csv"
    csv_path = os.path.join(output_dir, csv_name)

    # 计算相关系数矩阵
    corr_df = df.corr(method=method)
    print(f"使用 {method} 方法计算相关系数成功")

    # 保存到 CSV
    corr_df.to_csv(csv_path, index=True)
    print(f"相关系数矩阵已保存至: {csv_path}")

    if show is True:
        CorrelationHeatmap(np.abs(corr_df), figsize=(5, 5), cmap="Blues",
                           annot_size=8, title_size=14,
                           title_name='相关系数矩阵',
                           output_dir="output/corr")

    return corr_df
