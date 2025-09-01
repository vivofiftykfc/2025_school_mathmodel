import matplotlib
import matplotlib.pyplot as plt
def start_config():
    # 只能说基本上都可以通过这个来设置，也存在有在函数定义里规定格式的，具体还是看各个函数
    # 全局样式设置
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用白色网格背景
    # sns.set_palette("husl")  # 设置Seaborn默认配色为husl调色盘
    plt.rcParams.update({
        'font.size': 16,  # 全局字体大小
        'axes.titlesize': 20,  # 标题字号
        'axes.labelsize': 12,  # 坐标轴标签字号
        'xtick.labelsize': 14,  # X轴刻度字号
        'ytick.labelsize': 14,  # Y轴刻度字号
        'figure.dpi': 300,  # 输出分辨率
        'figure.facecolor': 'white',  # 画布背景色
        'savefig.bbox': 'tight',  # 保存时自动裁剪空白
    })

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False