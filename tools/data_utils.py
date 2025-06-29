# data_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from chardet import detect
import os
from datetime import datetime

def ReadData_robust(file_path, columns=None):
    """
    具有鲁棒性的表格文件读取器，最基础的一集，也是最多用的一个函数
    目前支持微软excel、csv文件。

    参数：
    file_path : str - 文件路径
    columns : list - 数据帧的目录

    返回：
    pandas.DataFrame - 包含原始数据的DataFrame
    """
    # 检测文件编码
    with open(file_path, 'rb') as f:
        rawdata = f.read(300)  # 读取前300字节用于检测编码
        result = detect(rawdata)

    file_ext = os.path.splitext(file_path)[1].lower()

    # 尝试用检测到的编码读取
    try:
        if file_ext is '.csv':
            df = pd.read_csv(file_path, encoding=result['encoding'])
            print(f'成功读取csv文件，其编码为：{result['encoding']}')

        elif file_ext in ['xlsx', '.xls']:
            df = pd.read_excel(file_path, encoding=result['encoding'])
            print(f'成功读取excel文件，其编码为：{result['encoding']}')

        else:
            print("不支持的文件格式，请返回修改ReadData函数使其支持该文件类型。")

    except:
        # 常见编码回退列表
        encodings = ['gb18030', 'gbk', 'utf-8-sig', 'latin1']
        for enc in encodings:
            try:
                if file_ext is '.csv':
                    df = pd.read_csv(file_path, encoding=enc)
                    print(f'成功读取csv文件，其编码为：{result['encoding']}')

                elif file_ext in ['xlsx', '.xls']:
                    df = pd.read_excel(file_path, encoding=enc)
                    print(f'成功读取excel文件，其编码为：{result['encoding']}')

                else:
                    print("不支持的文件格式，请返回修改ReadData函数使其支持该文件类型。")
                print(f"成功使用编码: {enc}")
                break
            except:
                continue
        else:
            raise ValueError("无法自动检测编码，请手动指定")

    print("数据已读取")
    if columns is None:
        print(f"column未传入")
    else:
        df.columns = columns
        print(f"已为df.column重命名：{df.columns}")

    print("前五行数据为："+df.head(5))
    print(df.info)
    return df

def vif_get_draw(df, wight, height, shownoshow=True, ci=90, palette='viridis'):
    """
    方差膨胀系数(variance inflation factor，VIF)
    是衡量多元线性回归模型中复 (多重)共线性严重程度的一种度量。
    它表示回归系数估计量的方差与假设自变量间不线性相关时方差相比的比值。
    相关程度越低，这个VIF值就越高。

    本函数用于计算输入的DataFrame中各特征的方差膨胀因子（VIF）。

    输入：DataFrame（仅数值特征）。
         wight 绘制宽度
         height 绘制高度
         shownoshow 最后绘制出来的图片会不会弹窗显示
         ci 绘图时的置信区间
         palette 颜色调色板
    输出：包含特征名称及其对应VIF值的DataFrame。
         同时绘制相关示意图、计算得到的数据，并保存在output文件夹下，没有的话会自己创建。

    参数：
    df : pandas.DataFrame - 仅包含数值特征的DataFrame

    返回：
    pandas.DataFrame - 包含各特征VIF值的DF
    """
    output_dir = "output/vif"
    if os.path.exists(output_dir):
        print(f"输出文件夹已存在，将输出到{output_dir}下")
    else:
        os.mkdir(output_dir)
        print(f"输出文件夹不存在，已经创建，将输出到{output_dir}下")

    timemark = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"vif_data_{timemark}.csv"
    png_name = f"vif_{timemark}.png"
    pdf_name = f"vif_{timemark}.pdf"

    # 计算vif，计算的函数依赖 from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_df = pd.DataFrame(columns=['feature', 'VIF'])
    vif_df['feature'] = df.columns
    vif_df['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    print("vif计算成功")

    # 保存数据
    vif_df.to_csv(os.path.join(output_dir,csv_name),index=True)
    print(f"VIF数据已保存至: {os.path.join(output_dir, csv_name)}")

    # 绘制图片
    plt.figure(figsize=(wight, height), dpi=300)
    sns.barplot(x='VIF', y='feature', data=vif_df.sort_values('VIF', ascending=False), ci=ci, palette=palette)
    plt.title('Feature VIF')
    plt.tight_layout()
    plt.savefig(png_name, dpi=300, bbox_inches='tight')  # 位图
    plt.savefig(pdf_name, dpi=300, bbox_inches='tight')  # 矢量图
    print(f"VIF图表已保存至: {output_dir}/{png_name} 和 {output_dir}/{pdf_name}")

    if shownoshow is True:
        plt.show()
        return vif_df
    else:
        return vif_df

def corr_get_save(df, output_dir="output/corr", method="pearson"):
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

    return corr_df

def get_timerelation_data(df, target_col, method='stats', window=24, time_colnum=0):
    """
        时间序列特征工程流水线

        未写好，泛用性不高，已废弃

        参数：
        df : pandas.DataFrame - 原始数据
        target_col : str - 目标列名称
        method : str - 特征生成方法（'lag'/'stats'）
        window : int - 时间窗口大小
        time_colnum ： int - 时间序列所在df里面的columns

        设计逻辑：
        1. 动态识别时间列（始终使用第一列）
        2. 构建时间索引并排序确保时序连续性
        3. 两种特征生成模式：
           - lag模式：生成滞后特征
           - stats模式：生成滑动统计量
        4. 自动对齐特征与目标变量，删除含空值记录
    """
    df = df.copy()
    """
    这一句刚开始是容易被误解的，但是实际上要理解为df实质是一个指向原始内存的标签，
    这一句相当于给标签所指向的内存复制一下多个副本，然后给这个标签从原始内存揭下来贴到副本上
    """
    time_col = df.columns[time_colnum]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    X = pd.DataFrame(index=df.index)
    if method == 'lag':
        for col in df.columns:
            for i in range(1, window + 1):
                X[f"{col}_t-{i}"] = df[col].shift(i)
    elif method == 'stats':
        for col in df.columns:
            X[f"{col}_mean"] = df[col].rolling(window).mean().shift(1)
            # X[f"{col}_max"]  = df[col].rolling(window).max().shift(1)
            # X[f"{col}_min"]  = df[col].rolling(window).min().shift(1)
            # X[f"{col}_std"]  = df[col].rolling(window).std().shift(1)
    else:
        raise ValueError("method 参数必须为 'lag' 或 'stats'")
    y_all = df[target_col]
    data = pd.concat([X, y_all], axis=1).dropna()
    return data.drop(columns=[target_col]), data[target_col]

def Clean_Odd(data, col_index, method='IQR'):
    """
    清洗数据中的异常值
    现支持IQR、3sigma方法
    目前空间复杂度还是较高，可下一步继续改进

    参数：
    data : numpy.ndarray - 原始数据矩阵，形状为(n_samples, n_features)
    col_index : list - 需要检测异常值的列索引
    method : char - 选择方法，支持'IQR'、'3sigma'

    返回：
    cleaned_data : numpy.ndarray - 清洗后的数据矩阵
    """
    # 验证列索引有效性
    if col_index < 0 or col_index >= data.shape[1]:
        raise ValueError(f"列索引{col_index}超出有效范围(0-{data.shape[1] - 1})")

    # 创建新np数组
    cleaned_data = np.empty_like(data)

    if method == 'IQR':
        for i in data.shape[1]:
            if i in col_index:
                # 提取目标列数据
                target_col = data[:, i]
                start_num = np.shape(target_col)

                # 计算四分位距
                Q1 = np.percentile(target_col, 25)
                Q3 = np.percentile(target_col, 75)
                IQR = Q3 - Q1

                # 计算异常值边界
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                print(f"异常值边界：{lower_bound}-{upper_bound}" + f"四分位距{Q1}、{Q3}")

                # 创建数据筛选掩模
                mask = (target_col >= lower_bound) & (target_col <= upper_bound)

                # 应用掩模筛选数据
                cleaned_data[i] = data[mask]
                finish = np.shape(cleaned_data)

                print(f"数据列：{i}清洗完毕，清洗前数据量为：{start_num}，清洗后：{finish}")

            else:

                cleaned_data[i] = data[i]
                print("不需清洗")

    elif method == '3sigma':
        for i in data.shape[1]:
            if i in col_index:
                # 提取目标列数据
                target_col = data[:, i]
                start_num = np.shape(target_col)

                # 计算均值与标准差
                mean = np.mean(target_col)
                std = np.std(target_col)

                # 计算异常值边界
                lower_bound = mean - 3 * std
                upper_bound = mean - 3 * std

                print(f"异常值边界：{lower_bound}-{upper_bound}" + f"均值{mean}标准差{std}")

                # 创建数据筛选掩模
                mask = (target_col >= lower_bound) & (target_col <= upper_bound)

                # 应用掩模筛选数据
                cleaned_data[i] = data[mask]
                finish = np.shape(cleaned_data)

                print(f"数据列：{i}清洗完毕，清洗前数据量为：{start_num}，清洗后：{finish}")

            else:

                cleaned_data[i] = data[i]
                print("不需清洗")

    return cleaned_data

def entropy_weight_topsis(df, column_big_better, column_small_better):
    """
    熵权法 + TOPSIS 综合评分
    暂只支持负向指标与正向指标。其他的可以继续写拓展。

    输入：
    df : DataFrame，数据帧
    column_big_better : list，正向指标列
    column_small_better : list，负向指标列

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
            df_norm[column] = df[column]
            print("不支持的数据类型，可以进一步改变原熵权法函数")

        # 2. 熵权法
    P = df_norm / (df_norm.sum(axis=0))  # axis=0的意思为沿垂直方向（按列）计算
    E = -np.nansum(P * np.log(P + 1e-9), axis=0) / np.log(len(df_norm))
    d = 1 - E
    weights = d / d.sum()

    indicators = df.columns.tolist()
    weight_dict = dict(zip(indicators, weights))

    print("原始全部指标权重（EWM）:", weight_dict)

    # 3. 构建加权标准化矩阵
    V = df_norm * weights

    # 4. TOPSIS 理想解
    A_pos = V.max()
    A_neg = V.min()

    # 5. 计算距离
    D_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    print(f"已算出欧氏距离，D_pos={D_pos}，D_neg={D_neg}")

    # 6. 计算综合得分
    y = D_neg / (D_pos + D_neg)  # 防止除以0

    print("\n熵权法-TOPSIS 分析结果")
    print("=" * 10)
    print(f"处理列数: {len(df_norm.columns)}")
    print(f"处理行数: {len(df)}")
    print("\n指标权重:")
    for col, weight in weight_dict.items():
        indicator_type = "负向" if col in column_small_better else "正向"
        print(f"  {col:<10} [{indicator_type}指标]: {weight:.6f}")
    print(f"最终打分：{y}")

    return y

def data_processing_gam():
    """数据预处理流程"""
    # 读取并处理附件1
    data1 = ReadData_robust('data/3-附件1-2025校赛第2轮A题边界层高度与天气数据.csv',
                            columns=['date', 'Uwind', 'Vwind', 'temp', 'pressure', 'surface_temp',
                     'latent_heat', 'sensible_heat', 'humidity', 'ABLH'])

    # 数据质量检查
    print("\n===== 附件1原始数据检查 =====")
    print("ABLH缺失值数量:", data1['ABLH'].isnull().sum())
    data1['ABLH'] = pd.to_numeric(data1['ABLH'], errors='coerce')

    # 计算日均值
    daily_ablh = data1.groupby('date')['ABLH'].mean().reset_index()

    # 读取并处理附件2
    data2 = ReadData_robust('data/4-附件2-2025校赛第2轮A题嘉兴空气质量数据.csv',
                        columns=['date', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO'])

    daily_ablh['date'] = pd.to_datetime(daily_ablh['date'], errors='coerce')
    data2['date'] = pd.to_datetime(data2['date'], errors='coerce')

    # 合并数据集
    merged_data = pd.merge(daily_ablh, data2, on='date', how='inner')

    # 构造特征矩阵
    X = merged_data[['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO']]
    y = merged_data['ABLH']

    return X, y, merged_data
