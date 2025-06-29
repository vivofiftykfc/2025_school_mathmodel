import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
from datetime import datetime

def ScatterPlotPlant(sorted_data):
    """
    绘制各气象参数与大气边界层高度的散点图（当前未启用）

    参数：
    sorted_data : numpy.ndarray - 排序后的气象数据矩阵

    设计逻辑：
    1. 使用matplotlib创建4x2子图布局
    2. 根据预定义的特征索引配置绘制各参数散点图
    3. 固定x轴为边界层高度，y轴展示不同气象参数
    4. 预留扩展接口但当前未实际使用
    """
    plt.figure(figsize=(12, 8))

    # 需要绘制的参数及对应位置（索引根据实际数据调整）
    features = [
        (1, "U-component wind (m/s)"),
        (2, "V-component wind (m/s)"),
        (3, "2m temperature (K)"),
        (4, "Surface pressure (Pa)"),
        (5, "Surface temperature (K)"),
        (6, "Latent heat flux (W/m²)"),
        (7, "Sensible heat flux (W/m²)"),
        (8, "Relative humidity (%)")
    ]

    # 创建子图
    for i, (col_idx, title) in enumerate(features, 1):
        plt.subplot(4, 2, i)
        plt.scatter(sorted_data[:, -1], sorted_data[:, col_idx],
                    alpha=0.5, s=10)
        plt.xlabel("Atmospheric boundary layer height (m)")
        plt.ylabel(title)

    plt.tight_layout()

    plt.show()

def HexbinPlotPlant(data, x_col=-1):
    """
    绘制六边形箱密度图分析变量关系

    参数：
    data : numpy.ndarray - 原始气象数据矩阵
    x_col : int - 自变量列索引（默认最后一列为边界层高度）

    返回：无，直接生成图像文件

    设计逻辑：
    1. 自适应数据列数（10列或8列）匹配不同数据集格式
    2. 使用hexbin实现二维密度可视化，对数颜色标度增强分布显示
    3. 多项式拟合趋势线（仅当数据量>10时尝试）
    4. 自动保存位图/矢量图双格式输出
    5. 异常捕获机制防止拟合失败导致程序中断
    """
    plt.figure(figsize=(12, 18))

    # 需要绘制的参数及对应位置（索引根据实际数据调整）
    df = data
    num_cols = df.shape[1]
    if num_cols == 10:
        features = [
            (1, "U向分量风 (m/s)"),
            (2, "V向分量风 (m/s)"),
            (3, "2m气温 (K)"),
            (4, "表面气压 (Pa)"),
            (5, "地表温度 (K)"),
            (6, "潜热通量 (W/m²)"),
            (7, "感热通量 (W/m²)"),
            (8, "相对湿度 (%)")
        ]
    elif num_cols == 8:
        features = [
            (1, "U向分量风 (m/s)"),
            (2, "V向分量风 (m/s)"),
            (3, "地表温度 (K)"),
            (4, "潜热通量 (W/m²)"),
            (5, "感热通量 (W/m²)"),
            (6, "相对湿度 (%)")
        ]

    num_cols = df.shape[1]

    for i, (col_idx, title) in enumerate(features, 1):
        # 创建子图，布局为4行2列，
        if num_cols == 10:
            ax = plt.subplot(4, 2, i)
        if num_cols == 8:
            ax = plt.subplot(3, 2, i)

        # --- 数据准备阶段 ---
        # 提取自变量（边界层高度）和因变量（当前特征）
        x = data.iloc[:, x_col]  # 默认最后一列作为x轴
        y = data.iloc[:, col_idx]  # 当前特征列作为y轴

        # --- 绘图阶段 ---
        # 创建六边形分箱图对象
        hb = ax.hexbin(
            x=x,  # x轴数据
            y=y,  # y轴数据
            gridsize=50,  # 横向网格划分数量（控制六边形大小）
            bins='log',  # 使用对数颜色标度（更好显示数据分布）
            mincnt=1,  # 最小显示计数（至少有1个点才着色）
            edgecolors='none',  # 去除六边形边框
            alpha=0.85,  # 透明度设置（0=完全透明，1=不透明）
            cmap = 'viridis'
        )

        # --- 颜色条设置 ---
        cb = plt.colorbar(hb, ax=ax)  # 为当前子图添加颜色条


        # --- 趋势线拟合 ---
        # --- 核心改进点：LOWESS趋势线拟合 ---
        if len(x) > 50:  # 提高最小样本量要求
            try:
                # 动态调整平滑参数（数据量越大，平滑程度越高）
                lowess_frac = 0.2 if len(x) > 5000 else 0.3
                smoothed = lowess(y, x,
                                  frac=lowess_frac,  # 控制平滑程度（0-1）
                                  it=3,  # 稳健拟合迭代次数
                                  delta=0.01 * max(x))  # 加速参数

                # 排序保证线条连续性
                sort_idx = np.argsort(smoothed[:, 0])
                x_smooth = smoothed[:, 0][sort_idx]
                y_smooth = smoothed[:, 1][sort_idx]

                # 绘制趋势线（带置信区间效果）
                ax.plot(x_smooth, y_smooth,
                        color='#FF4500',  # 更醒目的橙色
                        lw=2.5,
                        linestyle='-',
                        zorder=5,
                        label='LOWESS趋势线')

                # 添加趋势线置信带（半透明）
                ax.fill_between(x_smooth,
                                y_smooth - 0.1 * np.std(y),
                                y_smooth + 0.1 * np.std(y),
                                color='#FFA07A',
                                alpha=0.3,
                                zorder=4)

            except Exception as e:
                print(f"LOWESS拟合失败 ({title}): {str(e)}")
        else:
            print(f"数据量不足 ({title}): {len(x)} 有效样本")

    # --- 其他可视化元素增强 ---
    # 添加参考线（零值线/平均值线）
        ax.axhline(y=np.nanmean(y), color='grey', linestyle=':', alpha=0.7)
        ax.axvline(x=np.nanmean(x), color='grey', linestyle=':', alpha=0.7)

    # 添加相关系数标注
        corr_coef = np.corrcoef(x, y)[0, 1]
        ax.text(0.95, 0.95,
            f'ρ = {corr_coef:.2f}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top')

        plt.xlabel("大气边界层高度/m")
        plt.ylabel(title)

    plt.tight_layout()
    print("六边形箱图已绘制")
    plt.savefig('output/hb.png', dpi=600, bbox_inches='tight')  # 位图
    plt.savefig('output/hb.pdf', format='pdf', bbox_inches='tight')  # 矢量图
    plt.show()

def PairwiseLOWESSPlot(data, sample_frac=0.2):
    """
    绘制数值变量两两关系的矩阵图，包含以下特性：
    - 对角线显示直方图
    - 非对角线显示散点图
    - 叠加LOWESS平滑曲线
    - 自动处理数值变量和缺失值

    参数：
    data : pd的df
    sample_frac : float - 数据采样比例（0-1），用于加速大规模数据绘图

    设计逻辑：
    1. 自动识别数据集格式（10列/8列）配置中文标签
    2. 使用Seaborn PairGrid构建矩阵图框架
    3. 对角线显示带KDE的直方图，非对角线显示散点图
    4. 采用LOWESS非参数回归拟合趋势线
    5. 自适应平滑参数（根据数据量自动调整frac值）
    6. 优化可视化样式（颜色方案、标签旋转、布局紧凑）
    """
    # 转换为DataFrame
    df = data
    num_cols = df.shape[1]

    # 选择数值型列（排除时间列）
    numeric_cols = df.columns[1:]
    df_numeric = df[numeric_cols].astype(float)

    # 数据采样（处理大数据）
    if sample_frac < 1:
        df_sampled = df_numeric.sample(frac=sample_frac, random_state=42)
    else:
        df_sampled = df_numeric

    # 创建绘图矩阵
    g = sns.PairGrid(df_sampled, diag_sharey=False,
                     height=1.8, aspect=1.2)

    # 设置对角线和非对角线图形
    g.map_diag(sns.histplot, color='#3498db', kde=True,
               edgecolor='w', linewidth=0.5)
    g.map_offdiag(sns.scatterplot, s=8, alpha=0.4,
                  color='#2ecc71', edgecolor='none')

    # 添加LOWESS曲线
    def add_lowess(x, y, **kwargs):
        # 过滤缺失值
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]

        # 计算LOWESS平滑（frac控制平滑程度）
        lowess_frac = 0.2 if len(x_clean) > 1000 else 0.3
        smoothed = lowess(y_clean, x_clean,
                          frac=lowess_frac, it=2)

        # 绘制平滑曲线
        plt.plot(smoothed[:, 0], smoothed[:, 1],
                 color='#e74c3c', lw=1.5,
                 linestyle='--', zorder=10)

    # 在所有散点图上叠加LOWESS
    g.map_offdiag(add_lowess)

    # 美化样式
    plt.subplots_adjust(top=0.95)
    g.fig.suptitle('变量关系矩阵 (含LOWESS趋势线)',
                   fontsize=18, y=0.98)

    # 设置刻度文字方向
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=45)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

    # 修改散点样式
    g.map_offdiag(sns.scatterplot, s=12, alpha=0.6,
                  color='#4C72B0', edgecolor='w', linewidth=0.3)
    # 修改趋势线样式
    # plt.plot(..., color='#D1495B', lw=2, linestyle='-', zorder=10)
    # 直方图优化
    # g.map_diag(sns.histplot, color='#55A868', kde=True,
    #            kde_kws={'linewidth': 2}, edgecolor='none')
    # 调整子图间距
    plt.subplots_adjust(top=0.92, hspace=0.15, wspace=0.1)

    # 显示图形
    print("这个图也画完了QAQ")
    plt.savefig('output/pl.png', dpi=600, bbox_inches='tight')  # 位图
    plt.savefig('output/pl.pdf', format='pdf', bbox_inches='tight')  # 矢量图
    plt.show()

def CorrelationHeatmap(df, figsize=(12, 10), cmap='viridis',
                       annot_size=8, title_size=14,
                       title_name='相关系数矩阵',
                       output_dir = "output/corr"):
    """
    绘制相关系数热力图并标注数值

    参数：
    df : DataFrame - 数据信息
    figsize : 元组 - 图形尺寸
    cmap : str - 颜色映射方案
    annot_size : int - 标注文字大小
    title_size : int - 标题文字大小
    title_name : char - 标题文字
    output_dir ： char - 存储文件夹
    """

    if os.path.exists(output_dir):
        print(f"输出文件夹已存在，将输出到{output_dir}下")
    else:
        os.mkdir(output_dir)
        print(f"输出文件夹不存在，已经创建，将输出到{output_dir}下")

    timemark = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_name = f"corr_heat_{timemark}.png"
    pdf_name = f"corr_heat_{timemark}.pdf"

    # 创建绘图画布
    plt.figure(figsize=figsize)

    # 绘制热力图
    heatmap = sns.heatmap(
        df,
        annot=True,  # 显示数值
        fmt=".2f",  # 数值格式（保留两位小数）
        vmin=-1, vmax=1,  # 固定颜色范围
        center=0,  # 中心值
        square=True,  # 单元格为正方形
        linewidths=0.5,  # 单元格边线宽度
        cmap=cmap,
        annot_kws={"size": annot_size, "weight": "bold"},  # 加粗标注
        cbar_kws={"label": "相关系数", "shrink": 0.6}  # 颜色条标签
    )

    # 设置标题和标签
    heatmap.set_title(title_name, fontsize=title_size, pad=20, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    # 添加网格线
    heatmap.grid(visible=True, linestyle='--', alpha=0.3)

    # 优化布局
    plt.tight_layout()
    print("相关系数矩阵已绘制")
    plt.savefig(png_name, dpi=300, bbox_inches='tight')  # 位图
    plt.savefig(pdf_name, dpi=300, bbox_inches='tight')  # 矢量图
    plt.show()

def plot_pred(y_true, y_pred, name):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    m = max(y_true.max(), y_pred.max())
    plt.plot([0,m],[0,m],'r--')
    plt.xlabel("实际 ABLH")
    plt.ylabel("预测 ABLH")
    plt.title(f"{name}: 实测 vs 预测")
    plt.tight_layout()
    plt.savefig(f"./logs/{name}_pred.png")
    plt.close()

def plot_imp(model, name, feats):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        plt.figure(figsize=(6,4))
        plt.bar(range(len(feats)), imp[idx])
        plt.xticks(range(len(feats)), np.array(feats)[idx], rotation=45)
        plt.title(f"{name} 特征重要性")
        plt.tight_layout()
        plt.savefig(f"./logs/{name}_imp.png")
        plt.close()


def visualize_gam_results(gam, X_train, y_train, y_test, y_pred, save_dir="./output"):
    """美化后的 GAM 模型结果可视化 + 图像保存功能"""
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("deep")

    # --- 各变量影响图 ---
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(14, 10))
    features = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO']
    for i, feat in enumerate(features):
        plt.subplot(3, 2, i + 1)
        XX = gam.generate_X_grid(term=i)
        plt.scatter(X_train.iloc[:, i], y_train,
                    color=palette[0], alpha=0.4, s=20, label='实际值')
        plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX),
                 color=palette[2], linewidth=3, label='拟合曲线')
        plt.title(f'{feat} 与 ABLH 的关系', fontsize=22)
        plt.xlabel(feat, fontsize=20)
        plt.ylabel('ABLH', fontsize=20)
        plt.legend(loc='best', frameon=True)

    plt.tight_layout()
    plt.suptitle('各污染物对 ABLH 的影响', fontsize=24, y=1.02)
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"{save_dir}/gam_effects.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/gam_effects.pdf", bbox_inches='tight')
    plt.show()

    # --- 残差图 ---
    plt.figure(figsize=(8, 5))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, s=30, color=palette[1])
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('预测值', fontsize=18)
    plt.ylabel('残差', fontsize=18)
    plt.title('残差分布图', fontsize=24)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/gam_residuals.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/gam_residuals.pdf", bbox_inches='tight')
    plt.show()

def draw_pred(y_true, y_pred, filename, figsize=(6,6), dpi=300):
    """
    绘制预测 vs 实际 散点图
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.7)
    m = max(max(y_true), max(y_pred))
    plt.plot([0, m], [0, m], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 实际值')
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=dpi)
    plt.savefig(f'{filename}.pdf')
    plt.close()

def draw_error(y_true, y_pred, filename, figsize=(6,4), dpi=300):
    """
    绘制残差分布直方图
    """
    import matplotlib.pyplot as plt
    import numpy as np

    residuals = y_true - y_pred
    plt.figure(figsize=figsize)
    plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('残差')
    plt.ylabel('频数')
    plt.title('残差分布图')
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=dpi)
    plt.savefig(f'{filename}.pdf')
    plt.close()

def draw_bar(models, scores, ylabel, filename, figsize=(6,4), dpi=300):
    """
    绘制模型指标柱状图
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    bars = plt.bar(models, scores, color='skyblue')
    plt.ylabel(ylabel)
    plt.title(f'不同模型 {ylabel} 比较')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=dpi)
    plt.savefig(f'{filename}.pdf')
    plt.close()

