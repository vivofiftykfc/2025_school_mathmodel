from data_utils import *
from config import *

def question_one():
    df = get_model_eval_df()
    corr_get_save(df, 'output/corr', method='pearson', show=True)
    print('显示iuo的均值与方差线性相关度过大，排除掉iuo的方差这一列数据')
    entropy_weight_topsis(df, ['IOU_mean'], ['T_mean', 'N_mean'],
                          expert_weight_ratio=0.3,
                          expert_weights={'IOU_mean':0.6,'T_mean':0.2, 'N_mean':0.2})

start_config()
question_one()