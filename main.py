import pandas as pd
import numpy as np
import time 

from parse_PCB import gen_id_level2, parse, handle_missing_column
from src.config import *
from generate_feature import *
from model_inference import get_model, cal_score
import json


def score_pipeline(pcb_json):
    begin = time.time()
    #______________________________Data Preprocessing______________________________

    id_customer = pcb_json['contract_info']['customer_id']
    created_time = pcb_json['contract_info']['disbursed_time']
    credit_history = pcb_json['pcb_info']['pcb_output']['CI_Req_Output']['CreditHistory']

    df = pd.DataFrame()
    df['customer_id'] = [id_customer]
    df['created_time'] = [created_time]
    df['credit_history'] = [credit_history]
    
    # ROOT level
    id_col_list = ['customer_id']
    field = ['credit_history']

    df_root = parse(df, field, id_col_list)
    df_root = handle_missing_column(df_root, df_root_col)  

    # Granted contracts level
    field1 = ['Contract.NonInstalments.GrantedContract']
    noninstall = parse(df_root, field1, id_col_list)
    noninstall = handle_missing_column(noninstall, contract_level_noninstall_col)
    noninstall = noninstall.replace('null', np.nan)

    field2 = ['Contract.Instalments.GrantedContract']
    install = parse(df_root, field2, id_col_list)
    install = install.replace('null', np.nan)
    install = handle_missing_column(install, contract_level_install_col)

    field3 = ['Contract.Cards.GrantedContract']
    card = parse(df_root, field3, id_col_list)
    card = handle_missing_column(card, contract_level_card_col)    
    card = card.replace('null', np.nan)


    # Generate loan_code_lv2
    noninstall['loan_code_lv2'] = gen_id_level2(noninstall, 'customer_id')
    install['loan_code_lv2'] = gen_id_level2(install, 'customer_id')
    card['loan_code_lv2'] = gen_id_level2(card, 'customer_id')
    
    # Time series level
    id_col_list_ts = ['loan_code_lv2', 'customer_id', 'CommonData.CBContractCode']
    field = ['Profiles']
    ts_card = parse(card, field, id_col_list_ts)
    ts_card = handle_missing_column(ts_card, ts_col)

    ts_noninstall = parse(noninstall, field, id_col_list_ts)
    ts_noninstall = handle_missing_column(ts_noninstall, ts_col)

    ts_install = parse(install, field, id_col_list_ts)
    ts_install = handle_missing_column(ts_install, ts_col)     


    # ______________________________Generate features______________________________

    # 1. contracts_summary_terminates
    df_tm = (df_root.
            pipe(cal_terminate_info)
            )

    # 2/3. cc_os_rate_max_l25m, cc_os_rate_avg_l25m
    df_cc_os = (ts_card.
            pipe(get_card_ts, card).
            pipe(get_lxm, df).
            pipe(cal_os_rate)
            )

    # 4. card_summary_renounces
    df_card_rn = (df_root.
                pipe(cal_renounces)
                )

    # 5. pct_rm_term_lv
    df_inst_rm_lv = (install.
                    pipe(get_living_inst).
                    pipe(cal_percent_remain)
                    )
    
    # 6. od_utl_rate_max_l25m
    df_od_utl = (ts_noninstall.
                    pipe(get_od, noninstall).
                    pipe(agg_ts_od).
                    pipe(get_lxm_od, df).
                    pipe(cal_od_rate)
                    )
    
    # 7/8. consumerL_mth_pmt_sum_l25m, cashL_mth_pmt_sum_l4m
    df_consumerL = (ts_install.
                        pipe(get_mthly_pmt, install).
                        pipe(get_by_loantype).
                        pipe(cal_mt_pmt_rate, df)
                        )

    # 9. in_ln_grp_max_l4m
    df_ins_lnGrp = (ts_install.
                        pipe(get_in_ln_grp, ts_card, ts_noninstall).
                        pipe(cal_ln_grp_lxm, df)
                    )
    
    # Finalize
    conso_list = [df_tm, df_cc_os, df_card_rn, df_inst_rm_lv, df_od_utl, df_consumerL, df_ins_lnGrp]
    df_fn = console_feature(conso_list)
    woe_feature = transform_WOE(df_fn)

    # ______________________________Model Inference______________________________
    loaded_model = get_model('artifacts/Fiza_PCB_score_10Mar25.sav')
    predict, predict_score, score_feature = cal_score(woe_feature, loaded_model)

    df_fn = df_fn.fillna(-1)
    feature_value = df_fn.to_dict(orient='records')
    print(round(time.time()  - begin,3),'seconds')
    return predict, predict_score, feature_value, score_feature

if __name__ == "__main__":
 
    with open('data_input/input.json') as json_data:
        pcb_json = json.load(json_data)

    predict, predict_score, feature_value, score_feature = score_pipeline(pcb_json)
    # To API-response
    print('Probability:', predict[0])
    print('Score:', predict_score[0])
    print('Features:', feature_value)
    print('Feature_scores:', score_feature)


    