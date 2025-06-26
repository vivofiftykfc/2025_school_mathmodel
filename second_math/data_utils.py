# data_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from chardet import detect

def ReadExcel_robust(file_path, staff):
    """
    鲁棒性CSV文件读取器，最基础的一集

    参数：
    file_path : str - 文件路径
    staff : 1-2 - 选择器，第一题是1，第二题是2

    返回：
    pandas.DataFrame - 包含原始数据的DataFrame

    设计逻辑：
    1. 自动检测文件编码（使用chardet库）
    2. 备选编码回退机制（gb18030/gbk/utf-8-sig/latin1）
    3. 动态列名分配（根据数据列数匹配预定义方案）
    4. 确保返回DataFrame类型与其他模块兼容
    """
    # 检测文件编码
    with open(file_path, 'rb') as f:
        rawdata = f.read(1000)  # 读取前1000字节用于检测编码
        result = detect(rawdata)

    # 尝试用检测到的编码读取
    try:
        df = pd.read_csv(file_path, encoding=result['encoding'])
    except:
        # 常见编码回退列表
        encodings = ['gb18030', 'gbk', 'utf-8-sig', 'latin1']
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                print(f"成功使用编码: {enc}")
                break
            except:
                continue
        else:
            raise ValueError("无法自动检测编码，请手动指定")

    print("数据已读取")
    if(staff == 1):
        df.columns = ["date", "U向分量风/m/s", "V向分量风/m/s", "2m气温/K",
                  "表面气压/Pa", "地表温度/K", "潜热通量/W/(m^2)",
                  "感热通量/W/(m^2)", "相对湿度/%", "大气边界层高度/m"]
    if(staff == 2):
        df.columns = ["date", "AQI", "PM2.5", "PM10", "SO2", "NO2", "CO"]
    return df

def compute_vif(df):
    """
    计算DataFrame中各特征的方差膨胀因子（VIF）。
    输入：去除时间列的原始DataFrame（仅数值特征）。
    输出：包含特征名称及其对应VIF值的DataFrame。
    参数：
    df : pandas.DataFrame - 仅包含数值特征的DataFrame

    返回：
    pandas.DataFrame - 包含各特征VIF值的DF

    设计逻辑：
    1. 使用statsmodels的VIF实现
    2. 顺序计算各列膨胀因子
    3. 结果封装为易读的DataFrame格式
    """
    vif_df = pd.DataFrame(columns=['feature','VIF'])
    vif_df['feature'] = df.columns
    vif_df['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_df

def plot_vif(vif_df):
    """
    绘制VIF条形图，用于可视化多重共线性程度。
    """
    plt.figure(figsize=(8,6))
    sns.barplot(x='VIF', y='feature', data=vif_df.sort_values('VIF', ascending=False))
    plt.title('Feature VIF')
    plt.tight_layout()
    plt.savefig('output/vif.png', dpi=300, bbox_inches='tight')  # 位图
    plt.savefig('output/vif.pdf', format='pdf', bbox_inches='tight')  # 矢量图
    plt.show()

def Resort(data):
    """按最后一列（大气边界层高度）降序排序数据，但是好像没啥用"""
    sorted_indices = np.argsort(-data[:, -1])  # 获取降序排列的索引
    print("数据已重排")
    return data[sorted_indices]  # 返回排序后的数据

def generate_features(df, method='lag', window=24):
    X = pd.DataFrame(index=df.index)
    if method == 'lag':
        for col in df.columns:
            for i in range(1, window+1):
                X[f"{col}_t-{i}"] = df[col].shift(i)
    elif method == 'stats':
        for col in df.columns:
            X[f"{col}_mean"] = df[col].rolling(window).mean().shift(1)
            # X[f"{col}_max"]  = df[col].rolling(window).max().shift(1)
            # X[f"{col}_min"]  = df[col].rolling(window).min().shift(1)
            # X[f"{col}_std"]  = df[col].rolling(window).std().shift(1)
    else:
        raise ValueError("method 参数必须为 'lag' 或 'stats'")
    return X

def prepare_training_data(df, target_col, method='stats', window=24):
    """
        时间序列特征工程流水线

        参数：
        df : pandas.DataFrame - 原始数据
        target_col : str - 目标列名称
        method : str - 特征生成方法（'lag'/'stats'）
        window : int - 时间窗口大小
        默认情况下是直接取前24小时均值

        设计逻辑：
        1. 动态识别时间列（始终使用第一列）
        2. 构建时间索引并排序确保时序连续性
        3. 两种特征生成模式：
           - lag模式：生成滞后特征
           - stats模式：生成滑动统计量
        4. 自动对齐特征与目标变量，删除含空值记录
    """
    df = df.copy()
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    X_all = generate_features(df.drop(columns=[target_col]), method, window)
    y_all = df[target_col]
    data = pd.concat([X_all, y_all], axis=1).dropna()
    return data.drop(columns=[target_col]), data[target_col]

def convert_uv_wind(data, u_col=1, v_col=2):    #没用
    """
    将UV风场分量转换为风速和风向角，并覆盖原数据列

    参数：
    data : numpy.ndarray - 原始数据矩阵（包含U/V分量）
    u_col : int - U分量所在列索引（默认第2列）
    v_col : int - V分量所在列索引（默认第3列）

    返回：
    numpy.ndarray - 转换后的数据矩阵

    气象学标准：
    - 风向角：从正北顺时针旋转的角度（0-360度）
    - 风速：单位m/s（与输入单位一致）

    但是好像也没啥用，懒得删了先放着吧
    """
    # 拷贝原始数据避免污染输入
    converted = data.copy().astype(np.float64)

    # 提取UV分量
    u = converted[:, u_col]
    v = converted[:, v_col]

    # 计算风速（向量模长）
    wind_speed = np.hypot(u, v)  # 等效于√(u² + v²)

    # 计算风向角（气象学标准）
    wind_dir = np.degrees(np.arctan2(u, v))  # 先计算数学角度
    wind_dir = (wind_dir + 180) % 360  # 转换为气象学风向

    # 替换原始UV列
    converted[:, u_col] = wind_speed
    converted[:, v_col] = wind_dir

    return converted

def IQR_Clean_single(data, col_index):
    """
    使用IQR方法清洗数据中的异常值

    参数：
    data : numpy.ndarray - 原始数据矩阵，形状为(n_samples, n_features)
    col_index : int - 需要检测异常值的列索引

    返回：
    cleaned_data : numpy.ndarray - 清洗后的数据矩阵
    """
    # 验证列索引有效性
    if col_index < 0 or col_index >= data.shape[1]:
        raise ValueError(f"列索引{col_index}超出有效范围(0-{data.shape[1] - 1})")

    # 提取目标列数据
    target_col = data[:, col_index]

    # 计算四分位距
    Q1 = np.percentile(target_col, 25)
    Q3 = np.percentile(target_col, 75)
    IQR = Q3 - Q1

    # 计算异常值边界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 创建数据筛选掩模
    mask = (target_col >= lower_bound) & (target_col <= upper_bound)

    # 应用掩模筛选数据
    cleaned_data = data[mask]

    return cleaned_data

def IQRClean(data, col):
    """
        使用IQR方法清洗数据中选中指标的异常值

        参数：
        data : numpy.ndarray - 原始数据矩阵，形状为(n_samples, n_features)
        col_index : list - 需要检测异常值的列索引的列表，比如[0,1,3]就是要按照第0 1 3 列的数据进行清洗

        返回：
        data : numpy.ndarray - 清洗后的数据矩阵
        """
    for i in col :
        data = IQR_Clean_single(data, i)

    print("数据已清洗")
    return data

def remove_columns(data, columns_to_remove):    #没用
    """
    从二维数组中删除指定列。但是看上去好像没啥用，也懒得删了就这吧开摆

    参数：
    data : numpy.ndarray - 原始二维数组
    columns_to_remove : int/list - 要删除的列索引(支持负数索引和重复处理)

    返回：
    numpy.ndarray - 删除指定列后的新数组

    示例：
    >>> arr = np.array([[1,2,3], [4,5,6]])
    >>> remove_columns(arr, [0,2])
    array([[2],
           [5]])
    """
    # 转换为numpy数组（确保输入为数组）
    arr = np.asarray(data)

    # 参数标准化处理
    if not isinstance(columns_to_remove, (list, tuple, np.ndarray)):
        columns = [columns_to_remove]
    else:
        columns = list(columns_to_remove)

    # 处理负数索引转换
    cols = [c % arr.shape[1] for c in columns]

    # 去重并排序（从右到左删除避免索引变化）
    unique_cols = sorted(set(cols), reverse=True)

    # 有效性验证
    if not arr.ndim == 2:
        raise ValueError("输入必须是二维数组")
    if any(c >= arr.shape[1] for c in unique_cols):
        raise IndexError(f"列索引超出范围(0-{arr.shape[1] - 1})")

    # 执行删除操作
    result = arr.copy()
    for col in unique_cols:
        result = np.delete(result, col, axis=1)

    return result

def preprocess_and_save(ablh_path, air_path, output_path, staff):
    """
        完整数据处理流程并保存带标签数据

        参数：
        ablh_path : str - ABLH数据路径
        air_path : str - 空气质量数据路径
        output_path : str - 输出文件路径（建议.csv或.xlsx后缀）
        staff : int - 选择时序处理模式，1为均值方差，2为不处理直接整1*24的时间向量
    """
    # 数据读取（假设ReadExcel_robust第二个参数是sheet编号）
    df_ablh = ReadExcel_robust(ablh_path, 1)  # 读取第一个sheet的ABLH数据
    df_air = ReadExcel_robust(air_path, 2)  # 读取第二个sheet的空气质量数据

    # 调试：打印列名
    print("ABLH数据列名:", df_ablh.columns.tolist())
    print("空气质量数据列名:", df_air.columns.tolist())

    if staff == 1:

        # 日期处理（需确认ABLH数据中的时间列名称是否为'data'）
        try:
            df_ablh['date'] = pd.to_datetime(df_ablh['date']).dt.date  # 注意列名改为'data'
            df_air['date'] = pd.to_datetime(df_air['date']).dt.date
        except KeyError as e:
            raise ValueError(f"列名错误，请确认数据包含正确的日期列: {e}")

        # ABLH日聚合
        df_ablh_daily = df_ablh.groupby('date')['大气边界层高度/m'].agg(['mean', 'max', 'min'])
        df_ablh_daily.columns = ['ABLH_mean', 'ABLH_max', 'ABLH_min']

        # 合并数据
        df = pd.merge(df_air, df_ablh_daily, on='date', how='inner')

        # 时序特征
        df['date'] = pd.to_datetime(df['date'])

        # 构建完整数据集（包含日期和所有标签）
        feature_cols = ['ABLH_mean', 'ABLH_max', 'ABLH_min']
        target_cols = ['AQI','PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
        full_data = df[['date'] + feature_cols + target_cols]

    elif staff == 2:

        try:
            df_ablh["datetime"] = pd.to_datetime(df_ablh["date"])
            df_ablh["hour"] = df_ablh["datetime"].dt.hour  # 提取小时信息
            df_ablh['date'] = pd.to_datetime(df_ablh['date']).dt.date
            df_air['date'] = pd.to_datetime(df_air['date']).dt.date
        except KeyError as e:
            raise ValueError(f"列名错误，请确认数据包含正确的列: {e}")


        # 数据重塑：将24小时数据转为列
        df_pivot = df_ablh.pivot(
            index="date",
            columns="hour",
            values="大气边界层高度/m"
        )
        df_pivot.columns = [f"ABLH_h{str(h).zfill(2)}" for h in range(24)]

        # 合并数据
        df = pd.merge(df_air, df_pivot, on="date", how="inner")
        df['date'] = pd.to_datetime(df['date'])
        feature_cols = df_pivot.columns.tolist()
        target_cols = ['AQI','PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
        full_data = df[['date'] + feature_cols + target_cols]


    # 保存数据（根据后缀自动选择格式）
    df = df.sort_values('date').reset_index(drop=True)
    if output_path.endswith('.csv'):
        full_data.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        full_data.to_excel(output_path, index=False)
    else:
        raise ValueError("仅支持.csv或.xlsx格式")
    print(full_data.head(3))
    print(f"数据已保存至 {output_path}，共 {len(full_data)} 条记录")
    return full_data

def entropy_weight_topsis_score(X):
    """
    熵权法 + TOPSIS 综合评分

    输入：
    df : DataFrame，仅包含需要评分的正向指标列（如 PM2.5、PM10、NO2、SO2、AQI）

    输出：
    一维 numpy 数组，表示每一行的综合评分
    """
    # 1. 归一化（负向指标，越小越好）
    X_norm = (X.max() - X) / (X.max() - X.min() + 1e-9)

    # 2. 熵权法
    P = X_norm / (X_norm.sum(axis=0) + 1e-9)
    E = -np.nansum(P * np.log(P + 1e-9), axis=0) / np.log(len(X))
    d = 1 - E
    weights = d / d.sum()

    # 可选：查看权重
    indicators = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
    weight_dict = dict(zip(indicators, weights))
    print("指标权重（EWM）:", weight_dict)

    # 3. 构建加权标准化矩阵
    V = X_norm * weights

    # 4. TOPSIS 理想解
    A_pos = V.max()
    A_neg = V.min()

    # 5. 计算距离
    D_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    # 6. 计算综合得分
    y = D_neg / (D_pos + D_neg)  # 防止除以0
    return y


def data_processing_gam():
    """数据预处理流程"""
    # 读取并处理附件1
    data1 = ReadExcel_robust('3-附件1-2025校赛第2轮A题边界层高度与天气数据.csv', 1)
    data1.columns = ['date', 'Uwind', 'Vwind', 'temp', 'pressure', 'surface_temp',
                     'latent_heat', 'sensible_heat', 'humidity', 'ABLH']

    # 数据质量检查
    print("\n===== 附件1原始数据检查 =====")
    print("ABLH缺失值数量:", data1['ABLH'].isnull().sum())
    data1['ABLH'] = pd.to_numeric(data1['ABLH'], errors='coerce')

    # 计算日均值
    daily_ablh = data1.groupby('date')['ABLH'].mean().reset_index()

    # 读取并处理附件2
    columns_2 = ['date', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO']
    data2 = pd.read_csv('4-附件2-2025校赛第2轮A题嘉兴空气质量数据.csv',
                        encoding='gbk', header=0, names=columns_2, parse_dates=['date'])

    daily_ablh['date'] = pd.to_datetime(daily_ablh['date'], errors='coerce')
    data2['date'] = pd.to_datetime(data2['date'], errors='coerce')

    # 合并数据集
    merged_data = pd.merge(daily_ablh, data2, on='date', how='inner')

    # 构造特征矩阵
    X = merged_data[['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO']]
    y = merged_data['ABLH']

    return X, y, merged_data
