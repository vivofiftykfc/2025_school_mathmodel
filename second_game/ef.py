import numpy as np
from collections import defaultdict
import pprint
import re

model_info = {
    'LinearRegression': {'coefficients': {'1': np.float64(0.0), 'U向分量风/m/s': np.float64(-732.3342), 'V向分量风/m/s': np.float64(-10105.488), '2m气温/K': np.float64(23840.8564), '潜热通量/W/(m^2)': np.float64(-9760.7501), '感热通量/W/(m^2)': np.float64(-14730.6507), '相对湿度/%': np.float64(-13817.3368), 'U向分量风/m/s^2': np.float64(-553.3131), 'U向分量风/m/s V向分量风/m/s': np.float64(-353.3722), 'U向分量风/m/s 2m气温/K': np.float64(-1670.7047), 'U向分量风/m/s 潜热通量/W/(m^2)': np.float64(-431.2061), 'U向分量风/m/s 感热通量/W/(m^2)': np.float64(274.3239), 'U向分量风/m/s 相对湿度/%': np.float64(3083.5646), 'V向分量风/m/s^2': np.float64(-447.9044), 'V向分量风/m/s 2m气温/K': np.float64(18741.7708), 'V向分量风/m/s 潜热通量/W/(m^2)': np.float64(1313.1938), 'V向分量风/m/s 感热通量/W/(m^2)': np.float64(-525.9563), 'V向分量风/m/s 相对湿度/%': np.float64(554.4896), '2m气温/K^2': np.float64(-53872.218), '2m气温/K 潜热通量/W/(m^2)': np.float64(19214.275), '2m气温/K 感热通量/W/(m^2)': np.float64(28524.1245), '2m气温/K 相对湿度/%': np.float64(29159.1284), '潜热通量/W/(m^2)^2': np.float64(-1028.7586), '潜热通量/W/(m^2) 感热通量/W/(m^2)': np.float64(-668.0955), '潜热通量/W/(m^2) 相对湿度/%': np.float64(-2019.4897), '感热通量/W/(m^2)^2': np.float64(128.4513), '感热通量/W/(m^2) 相对湿度/%': np.float64(1769.6234), '相对湿度/%^2': np.float64(-1883.6226), 'U向分量风/m/s^3': np.float64(13.3051), 'U向分量风/m/s^2 V向分量风/m/s': np.float64(-4.6347), 'U向分量风/m/s^2 2m气温/K': np.float64(762.649), 'U向分量风/m/s^2 潜热通量/W/(m^2)': np.float64(45.4356), 'U向分量风/m/s^2 感热通量/W/(m^2)': np.float64(22.6329), 'U向分量风/m/s^2 相对湿度/%': np.float64(-95.4553), 'U向分量风/m/s V向分量风/m/s^2': np.float64(-11.408), 'U向分量风/m/s V向分量风/m/s 2m气温/K': np.float64(399.5011), 'U向分量风/m/s V向分量风/m/s 潜热通量/W/(m^2)': np.float64(-3.8471), 'U向分量风/m/s V向分量风/m/s 感热通量/W/(m^2)': np.float64(1.5015), 'U向分量风/m/s V向分量风/m/s 相对湿度/%': np.float64(-47.6934), 'U向分量风/m/s 2m气温/K^2': np.float64(2365.5191), 'U向分量风/m/s 2m气温/K 潜热通量/W/(m^2)': np.float64(237.0636), 'U向分量风/m/s 2m气温/K 感热通量/W/(m^2)': np.float64(-108.4217), 'U向分量风/m/s 2m气温/K 相对湿度/%': np.float64(-3070.1294), 'U向分量风/m/s 潜热通量/W/(m^2)^2': np.float64(-52.4854), 'U向分量风/m/s 潜热通量/W/(m^2) 感热通量/W/(m^2)': np.float64(-21.7209), 'U向分量风/m/s 潜热通量/W/(m^2) 相对湿度/%': np.float64(103.736), 'U向分量风/m/s 感热通量/W/(m^2)^2': np.float64(61.2251), 'U向分量风/m/s 感热通量/W/(m^2) 相对湿度/%': np.float64(-96.6446), 'U向分量风/m/s 相对湿度/%^2': np.float64(-16.9002), 'V向分量风/m/s^3': np.float64(31.8511), 'V向分量风/m/s^2 2m气温/K': np.float64(766.0456), 'V向分量风/m/s^2 潜热通量/W/(m^2)': np.float64(84.9243), 'V向分量风/m/s^2 感热通量/W/(m^2)': np.float64(27.574), 'V向分量风/m/s^2 相对湿度/%': np.float64(-170.1476), 'V向分量风/m/s 2m气温/K^2': np.float64(-8796.354), 'V向分量风/m/s 2m气温/K 潜热通量/W/(m^2)': np.float64(-1020.8734), 'V向分量风/m/s 2m气温/K 感热通量/W/(m^2)': np.float64(514.3724), 'V向分量风/m/s 2m气温/K 相对湿度/%': np.float64(-463.5555), 'V向分量风/m/s 潜热通量/W/(m^2)^2': np.float64(40.0243), 'V向分量风/m/s 潜热通量/W/(m^2) 感热通量/W/(m^2)': np.float64(31.1285), 'V向分量风/m/s 潜热通量/W/(m^2) 相对湿度/%': np.float64(-193.1097), 'V向分量风/m/s 感热通量/W/(m^2)^2': np.float64(19.3031), 'V向分量风/m/s 感热通量/W/(m^2) 相对湿度/%': np.float64(13.2146), 'V向分量风/m/s 相对湿度/%^2': np.float64(-26.8685), '2m气温/K^3': np.float64(30174.8061), '2m气温/K^2 潜热通量/W/(m^2)': np.float64(-10942.8355), '2m气温/K^2 感热通量/W/(m^2)': np.float64(-13985.9678), '2m气温/K^2 相对湿度/%': np.float64(-15972.7534), '2m气温/K 潜热通量/W/(m^2)^2': np.float64(-81.4226), '2m气温/K 潜热通量/W/(m^2) 感热通量/W/(m^2)': np.float64(741.1215), '2m气温/K 潜热通量/W/(m^2) 相对湿度/%': np.float64(3100.0284), '2m气温/K 感热通量/W/(m^2)^2': np.float64(-359.8906), '2m气温/K 感热通量/W/(m^2) 相对湿度/%': np.float64(-1781.7966), '2m气温/K 相对湿度/%^2': np.float64(2664.8174), '潜热通量/W/(m^2)^3': np.float64(-292.0685), '潜热通量/W/(m^2)^2 感热通量/W/(m^2)': np.float64(71.8244), '潜热通量/W/(m^2)^2 相对湿度/%': np.float64(495.7236), '潜热通量/W/(m^2) 感热通量/W/(m^2)^2': np.float64(-38.9215), '潜热通量/W/(m^2) 感热通量/W/(m^2) 相对湿度/%': np.float64(-76.7204), '潜热通量/W/(m^2) 相对湿度/%^2': np.float64(-211.6451), '感热通量/W/(m^2)^3': np.float64(-22.0088), '感热通量/W/(m^2)^2 相对湿度/%': np.float64(171.5434), '感热通量/W/(m^2) 相对湿度/%^2': np.float64(69.3981), '相对湿度/%^3': np.float64(-452.8205)}, 'intercept': np.float64(511.0318)
    }
}
import numpy as np
import re
from collections import defaultdict


# 变量名映射表：原始特征名 -> 简化符号
name_map = {
    'U向分量风/m/s': 'U',
    'V向分量风/m/s': 'V',
    '2m气温/K': 'T',
    '相对湿度/%': 'H',
    '潜热通量/W/(m^2)': 'LH',
    '感热通量/W/(m^2)': 'SH'
}

# 定义变量顺序，用于向量/矩阵索引
var_order = ['U', 'V', 'T', 'H', 'LH', 'SH']
var_index = {v: i for i, v in enumerate(var_order)}
n = len(var_order)

# 提取系数和截距
coeffs = model_info['LinearRegression']['coefficients']
intercept = float(model_info['LinearRegression']['intercept'])

# 解析特征名函数：仅匹配末尾^数字幂次，保留单位中^的合法部分
def parse_feature_name_to_terms(feature_name):
    terms = []
    for token in feature_name.split():
        match = re.match(r"^(.*?)(?:\^(\d+))?$", token)
        if not match:
            continue
        raw_var = match.group(1).strip()
        sym = name_map.get(raw_var)
        if sym is None:
            continue
        power = int(match.group(2)) if match.group(2) else 1
        terms.append((sym, power))
    return terms

# 初始化：一次项向量、二次项矩阵、三次项稀疏张量
C1 = np.zeros(n)
C2 = np.zeros((n, n))
C3 = defaultdict(float)

# 遍历所有特征项，填充矩阵/张量
for feat, coeff in coeffs.items():
    if np.isclose(coeff, 0):
        continue
    terms = parse_feature_name_to_terms(feat)
    # 展开如 [('U',2),('T',1)] -> ['U','U','T']
    expanded = []
    for var, p in terms:
        expanded.extend([var] * p)
    # 排序变量，确保一致的索引键
    expanded.sort()
    L = len(expanded)
    if L == 1:
        i = var_index[expanded[0]]
        C1[i] += coeff
    elif L == 2:
        i, j = var_index[expanded[0]], var_index[expanded[1]]
        # 对称矩阵填充
        if i == j:
            C2[i, i] += coeff
        else:
            C2[i, j] += coeff / 2
            C2[j, i] += coeff / 2
    elif L == 3:
        idx = tuple(var_index[v] for v in expanded)
        C3[idx] += coeff
    else:
        # 可以扩展到更高阶
        print(f"Warning: found order {L} term, skipping: {feat}")

# 输出结果
print(f"Intercept: {intercept:.4f}\n")
print("C1 (一次项向量)：")
for i, v in enumerate(var_order):
    print(f"  {v}: {C1[i]:+.4f}")

print("\nC2 (二次项矩阵)：")
print(C2)

print("\nC3 (三次项稀疏张量，部分展示)：")
for idx, val in sorted(C3.items())[:1000]:  # 只展示前20项
    vars_comb = ' * '.join(var_order[i] for i in idx)
    print(f"  {idx}: {val:+.4f}  ({vars_comb})")