from data_utils import *
from draw_utils import *
from model_utils import *
import os

def question_one():
    TARGET_COL = 'ABLH/m'  # 预测目标
    FEATURE_METHOD = 'stats'
    WINDOW_SIZE = 24
    TEST_SIZE = 0.2

    # ==== 读取数据，返回一个pandas的df的指针 ====
    df = ReadExcel_robust("3-附件1-2025校赛第2轮A题边界层高度与天气数据.csv",1)

    # # ==== 画个图来分别说明各变量间的相关性 ====
    # HexbinPlotPlant(df)
    # PairwiseLOWESSPlot(df, sample_frac=0.3)
    CorrelationHeatmap(df)

    # ==== 共线性分析 ====
    # vif_df = compute_vif(df.iloc[:,1:-1])
    # print(vif_df)
    # plot_vif(vif_df)

    # ==== 时序性特征构造有点复杂，第一题先不这么做了 ====
    # X, y = prepare_training_data(df, TARGET_COL, method=FEATURE_METHOD, window=WINDOW_SIZE)

    # ==== 模型训练与评估 ====
    results_df, coef_info, models_dict, best_params = evaluate_traditional_regression_models(df.iloc[:, [1,2,3,6,7,8]], df.iloc[:, -1], test_size=TEST_SIZE,
                                                         poly_degree=3)
    print(results_df)
    print(coef_info)
    print(best_params)
    # print(result_df)

def question_two(staff):
    print("快亖了喵~")

    # ==== 数据读取与预处理，还有训练集划分 ====
    data = preprocess_and_save('3-附件1-2025校赛第2轮A题边界层高度与天气数据.csv', '4-附件2-2025校赛第2轮A题嘉兴空气质量数据.csv', 'processed_data.csv', staff)
    feat_cols = [c for c in data.columns if c.startswith('ABLH')]
    X = data[feat_cols]

    # indicator_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
    # y = entropy_weight_topsis_score(data[indicator_cols])

    y = data[['AQI','PM2.5', 'PM10', 'NO2', 'SO2', 'CO']].mean(axis=1)

    # 选项3 层次分析法
    # weights = np.array([0.3, 0.14, 0.14, 0.14, 0.14, 0.14])
    # columns = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
    # y = data[columns].values @ weights

    print(X.head(3))
    print(y.head(3))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    yte, ypred, best, name = train_and_save_tree_basic(Xtr, Xte, ytr, yte)
    # 可视化
    plot_pred(yte, ypred, name)
    plot_imp(best, name, feat_cols)

def question_three():

    TEST_SIZE = 0.2
    save_path = 'output/bbbbiiiiggggmodel_stack.pkl'


    df = ReadExcel_robust('3-附件1-2025校赛第2轮A题边界层高度与天气数据.csv', staff=1)
    feat_cols = ["U向分量风/m/s", "V向分量风/m/s", "2m气温/K",
                  "表面气压/Pa", "地表温度/K", "潜热通量/W/(m^2)",
                  "感热通量/W/(m^2)", "相对湿度/%"]
    X = df[feat_cols]
    y = df["大气边界层高度/m"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    lin_models = train_linear_models_three(X_tr, y_tr, 3)
    print(lin_models)

    tree_models = train_tree_models_three(X_tr, X_te, y_tr, y_te)
    print(tree_models)

    all_models = {**lin_models, **tree_models}

    all_models = choose_best_model(all_models)

    stack = train_stacking(all_models, X_tr, y_tr)

    evaluate_and_save(stack, scaler, X_te, y_te, save_path)

    y_next = predict_next_day(X_te, save_path)

    print(y_next)

    df_next = pd.DataFrame({
        'ABLH_next': y_next
    })

    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'y_next_prediction.csv')
    df_next.to_csv(out_path, index=False)
    print(f"次日 ABLH 预测已保存到：{out_path}")

def question_two_gam():
    # 数据预处理
    X, y, merged_data = data_processing_gam()

    # 模型构建与评估
    gam, X_train, X_test, y_train, y_test, y_pred = build_gam(X, y)

    # 结果可视化
    visualize_gam_results(gam, X_train, y_train, y_test, y_pred)

def start_config():
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

start_config()

if __name__ == "__main__":

    start_config()
    # question_one()
    # question_two(2)
    # question_three()
    question_two_gam()


