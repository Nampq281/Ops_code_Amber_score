import pandas as pd
import numpy as np
import pickle
from functools import wraps
from datetime import datetime as dt

def log_step_model(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.now() - tic)
        print(f"[{func.__name__}] Process time: {time_taken}s")
        return result
    return wrapper

@log_step_model
def score_scaling(offset, factor, event_p):
    ln_odds = np.log((1-event_p)/event_p)
    score = offset + factor*ln_odds
    score = np.round(score,0)
    return score

@log_step_model
def get_model(model_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model

@log_step_model
def cal_score(woe_df, loaded_model):
    woe_df['const'] = 1 # for statsmodels
    final_feat = ['cc_os_rate_avg_l25m',
                  'card_summary_renounces',
                  'contracts_summary_terminates', 
                  'in_ln_grp_max_l4m',
                  'cashL_mth_pmt_sum_l4m', 
                  'cc_os_rate_max_l25m', 
                  'od_utl_rate_max_l25m',
                  'pct_rm_term_lv', 
                  'consumerL_mth_pmt_sum_l25m']
    
    offset, factor = 487.122876205, 28.853900818
    beta = loaded_model.params
    n = len(final_feat)
    intercept = beta['const']

    predict = loaded_model.predict(exog=woe_df[final_feat+['const']])
    predict_score = score_scaling(offset, factor, predict) 
    score_feature = (-1*(woe_df[final_feat]*beta+intercept/n)*factor+offset/n) 
    score_feature = round(score_feature, 2) 
    score_feature = score_feature[final_feat].to_dict(orient='records')

    return predict, predict_score, score_feature