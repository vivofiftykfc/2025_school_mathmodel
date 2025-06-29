import numpy as np
import pandas as pd
import time
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
import logging
from sklearn.ensemble import StackingRegressor
import scipy.sparse
def to_array(self):
    return self.toarray()
scipy.sparse.spmatrix.A = property(to_array)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from data_utils import *
import scipy.sparse
import numpy as np
np.int = int
def to_array(self):
    return self.toarray()
scipy.sparse.spmatrix.A = property(to_array)
from pygam import LinearGAM, s

# ========== 传统多元回归算法方案所用到的函数 ==========

def calculate_relative_error(y_true, y_pred):
    """
    计算相对误差比例（RMSE / 目标变量均值）

    参数：
    y_true : array-like 真实值
    y_pred : array-like 预测值

    返回：
    float : 百分比形式的相对误差比例
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_y = np.mean(y_true)
    return (rmse / mean_y) * 100  # 返回百分比

def evaluate_traditional_regression_models(X, y, test_size=0.2, poly_degree=1, param_search=False):
    """
    增强版回归模型评估框架

    参数：
    poly_degree : int - 多项式特征次数（1=无多项式）
    param_search : bool - 是否启用参数搜索

    返回：
    (评估结果DataFrame, 模型参数字典, 最优参数字典)
    """
    # 基础模型配置
    base_models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet()
    }

    # 动态参数空间（用于参数搜索）
    param_grids = {
        'Ridge': {'alpha': [0.01, 0.1, 1, 10]},
        'Lasso': {'alpha': [0.001, 0.01, 0.1]},
        'ElasticNet': {
            'alpha': [0.001, 0.01, 0.1],
            'l1_ratio': [0.3, 0.5, 0.7]
        }
    }

    # 多项式特征工程
    if poly_degree > 1:
        X = generate_poly_features(X, degree=poly_degree)

    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # 输入数据标准化

    # 数据分割（注意关闭shuffle以保持时间序列特性）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)

    results = []  # 存储各模型评估结果
    coef_dict = {}  # 存储各模型系数信息
    models_dict = {}
    best_params = {}  # 存储参数搜索得到的最佳参数

    for name in base_models:
        # 参数搜索逻辑
        if param_search and name in param_grids:
            model = GridSearchCV(
                estimator=base_models[name],
                param_grid=param_grids[name],
                cv=5,
                scoring='neg_mean_squared_error'
            )
            search_start = time.time()
            model.fit(X_train, y_train)
            search_time = time.time() - search_start
            best_params[name] = model.best_params_
            final_model = model.best_estimator_
        else:
            final_model = base_models[name]
            best_params[name] = None

        # 模型训练
        train_start = time.time()
        final_model.fit(X_train, y_train)
        train_time = time.time() - train_start

        # 结果记录
        y_pred = final_model.predict(X_test)
        relative_error = calculate_relative_error(y_test, y_pred)

        results.append({
            'Model': name,
            'TrainTime': round(train_time, 4),
            'R2': round(r2_score(y_test, y_pred), 4),
            'MSE': round(mean_squared_error(y_test, y_pred), 4),
            'RelativeError(%)': round(relative_error, 2),
            'Params': best_params.get(name, None)
        })

        # 系数记录
        coef_dict[name] = extract_model_coefficients(final_model,  feature_names)
        models_dict[name] = final_model

    return pd.DataFrame(results), coef_dict, models_dict, best_params
    # return pd.DataFrame(results), best_params

def generate_poly_features(X, degree=2):
    """生成多项式特征"""
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=True,
        interaction_only=False
    )
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    return pd.DataFrame(X_poly, columns=feature_names)

def extract_model_coefficients(model, feature_names):
    """提取模型系数并生成可读方程"""
    coef_info = {}

    # 处理管道模型（如参数搜索后的最佳估计器）
    if isinstance(model, Pipeline):
        final_step = model.steps[-1][1]
        coef = final_step.coef_ if hasattr(final_step, 'coef_') else None
        intercept = final_step.intercept_ if hasattr(final_step, 'intercept_') else None
    else:
        coef = model.coef_ if hasattr(model, 'coef_') else None
        intercept = model.intercept_ if hasattr(model, 'intercept_') else None

    # 构建系数字典
    coef_info['coefficients'] = dict(zip(feature_names, np.round(coef, 4))) if coef is not None else None
    coef_info['intercept'] = round(intercept, 4) if intercept is not None else None

    # 生成方程字符串
    equation = []
    if intercept is not None:
        equation.append(f"{round(intercept, 2)}")
    if coef is not None:
        for name, val in zip(feature_names, coef):
            equation.append(f"{round(val, 2)}*{name}")
    coef_info['equation'] = "y = " + " + ".join(equation) if equation else "No linear equation"

    # 添加正则化参数（如果适用）
    if hasattr(model, 'alpha'):
        coef_info['alpha'] = model.alpha
    if hasattr(model, 'l1_ratio'):
        coef_info['l1_ratio'] = model.l1_ratio

    return coef_info

# ========== 决策树方案所用到的函数 ==========

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_save_tree_basic(Xtr, Xte, ytr, yte):
    """
    训练模型、评估、保存，并记录日志
    """
    # 时间序列CV
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    tscv = TimeSeriesSplit(n_splits=10)

    # 定义基础模型与参数空间
    search_configs = {
        'RF_bootstrap': {
            'estimator': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200, 300, 500, 800, 1000],  # 更平滑的树数量增长
                'max_depth': [None, 5, 10, 15, 20, 30, 50, 70, 100],  # 增加中间深度
                'min_samples_split': [2, 3, 5, 10, 12, 15],  # 更多分裂限制
                'min_samples_leaf': [1, 2, 4, 8],  # 叶节点样本限制
                'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0],  # 特征采样策略
                'bootstrap': [True],  # 是否放回抽样
                'max_samples': [None, 0.5, 0.8]  # 样本采样比例
            }
        },
        'RF': {
            'estimator': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200, 300, 500, 800, 1000],  # 更平滑的树数量增长
                'max_depth': [None, 5, 10, 15, 20, 30, 50, 70, 100],  # 增加中间深度
                'min_samples_split': [2, 3, 5, 10],  # 更多分裂限制
                'min_samples_leaf': [1, 2, 4, 8],  # 叶节点样本限制
                'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0],  # 特征采样策略
                'bootstrap': [False],  # 是否放回抽样
            }
        },
        'XGB': {
            'estimator': XGBRegressor(objective='reg:squarederror', random_state=42),
            'params': {
                'n_estimators': [100, 200, 300, 500, 800],  # 更多基学习器数量
                'max_depth': [3, 4, 5, 6, 7, 9, 12],  # 更细粒度深度控制
                'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],  # 扩展学习率范围
                'subsample': [0.2, 0.4, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 样本采样比例
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 特征采样比例
                'gamma': [0, 0.1, 0.5, 1, 2],  # 分裂最小损失下降
                'reg_alpha': [0, 0.01, 0.05, 0.1, 1, 10],  # L1正则化
                'reg_lambda': [0, 0.1, 1, 10, 15, 20, 30]  # L2正则化
            }
        }
    }

    logging.basicConfig(
        filename="logs/tune.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    results = []
    # 对每个模型做随机搜索
    for name, cfg in search_configs.items():
        print(f"\n>> 调参模型: {name}")
        rs = RandomizedSearchCV(
            cfg['estimator'],
            cfg['params'],
            n_iter=20,
            scoring=rmse_scorer,
            cv=tscv,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        rs.fit(Xtr, ytr)
        best = rs.best_estimator_
        logging.info(f"{name} 最佳参数: {rs.best_params_}")
        print(f"{name} 最佳参数: {rs.best_params_}")

        # 用最优模型评估
        t0 = time.time()
        best.fit(Xtr, ytr)
        dt = time.time() - t0
        ypred = best.predict(Xte)
        r2 = r2_score(yte, ypred)
        rm = np.sqrt(mean_squared_error(yte, ypred))
        logging.info(f"{name} 最终评估: 用时{dt:.2f}s, R2={r2:.3f}, RMSE={rm:.3f}")
        print(f"{name} 最终评估: 用时{dt:.2f}s, R2={r2:.3f}, RMSE={rm:.3f}")

        # 保存模型
        joblib.dump(best, f'./models/{name}_best.pkl')



        results.append({'Model': name, 'Time': dt, 'R2': r2, 'RMSE': rm})

    # 汇总结果
    df_res = pd.DataFrame(results)
    df_res.to_csv("logs/model_summary_tuned.csv", index=False)
    print("\n调参后性能对比：")
    print(df_res)

    return yte, ypred, best, name

def choose_best_model(models):
    return models

# ==== 第三问融合算法所需的函数 ====

def train_linear_models_three(X_tr, y_tr, poly_degree=2):
    # 复用evaluate_traditional_regression_models框架，并提取训练好的estimator
    feature_names = ["U向分量风/m/s", "V向分量风/m/s", "2m气温/K",
                  "表面气压/Pa", "地表温度/K", "潜热通量/W/(m^2)",
                  "感热通量/W/(m^2)", "相对湿度/%"]
    X_tr = pd.DataFrame(X_tr, columns=feature_names)

    results_df, coef_dict, models_dict, best_params = evaluate_traditional_regression_models(
        X_tr, y_tr, test_size=0.2, poly_degree=poly_degree, param_search=False
    )
    print(results_df)
    print(coef_dict)
    print(best_params)
    return models_dict  # e.g. {'LinearRegression': lr_model, ...}

def train_tree_models_three(X_tr, X_te, y_tr, y_te):
    models = {}
    for _ in range(2):  # 内部RF_bootstrap、RF、XGB三选其优
        y_true, y_pred, best_model, name = train_and_save_tree_basic(X_tr, X_te, y_tr, y_te)
        models[name] = best_model
    return models  # e.g. {'RF': rf_model, 'XGB': xgb_model}

def train_stacking(models_dict, X_tr, y_tr):
    """
    基于已有的基学习器，训练一个堆叠融合模型（StackingRegressor）
    """
    # 1. 提取指定基学习器
    xgb_model = models_dict['XGB']  # XGBoost sklearn 接口回归器:contentReference[oaicite:2]{index=2}
    lasso_model = models_dict['Lasso']  # Lasso 回归器（L1 正则化）:contentReference[oaicite:3]{index=3}

    # 2. 按 (名称, 实例) 构造 estimators 列表
    estimators = [
        ('XGB', xgb_model),
        ('Lasso', lasso_model)
    ]  # estimators 参数期望 List[(str, estimator)]:contentReference[oaicite:4]{index=4}
    # 创建堆叠回归器：一级学习器列表 + 最终的线性回归元学习器 + 时间序列交叉验证
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=KFold(n_splits=5)
    )
    # 用训练集数据拟合堆叠模型
    stack.fit(X_tr, y_tr)
    # 返回训练好的堆叠模型
    return stack

def evaluate_and_save(stack, scaler, X_te, y_te, save_path):
    """
    对堆叠模型在测试集上评估，并将模型与标准化器保存到本地
    """
    # 用融合模型对测试集进行预测
    y_pred = stack.predict(X_te)
    # 计算并打印 R² 决定系数
    print("R2:", r2_score(y_te, y_pred))
    # 计算并打印均方误差
    print("MSE:", mean_squared_error(y_te, y_pred))
    # TODO: 如需输出相对误差，可在此调用自定义函数 calculate_relative_error
    # 将模型和标准化器封装后保存，方便后续加载预测
    joblib.dump({'model': stack, 'scaler': scaler}, save_path)
    print('模型已保存')

def predict_next_day(X_fc, model_path):
    """
    加载气象预报数据，用已保存的融合模型预测次日 ABLH
    """
    # 从本地加载之前保存的模型和标准化器
    tmp = joblib.load(model_path)
    scaler, stack = tmp['scaler'], tmp['model']
    # 对新特征做同样的标准化处理
    X_fc_scaled = scaler.transform(X_fc)
    # 返回次日 ABLH 的预测结果数组
    return stack.predict(X_fc_scaled)

# ==== 第二问补充算法GAM ====
def build_gam(X, y):
    """构建并训练GAM模型"""
    # 数据分割
    test_size = 0.42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print("\n===== 训练信息 =====")
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")

    # 模型训练
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5), fit_intercept=True)
    gam.fit(X_train, y_train)

    # 模型评估
    y_pred = gam.predict(X_test)
    print(f'R² Score: {r2_score(y_test, y_pred):.3f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}')

    return gam, X_train, X_test, y_train, y_test, y_pred