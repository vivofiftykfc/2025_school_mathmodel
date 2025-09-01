import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def CorrelationHeatmap(df, figsize=(12, 10), cmap='Blues',
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
    png_path = os.path.join(output_dir, png_name)
    pdf_name = f"corr_heat_{timemark}.pdf"
    pdf_path = os.path.join(output_dir, pdf_name)

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
    plt.savefig(png_path, dpi=300, bbox_inches='tight')  # 位图
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')  # 矢量图
    plt.show()