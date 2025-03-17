import pandas as pd
import numpy as np
import pickle
import time 

from src.utils_ops import formatdate, to_dt_fmt, handle_missing_column
from src.config import *

from parse_PCB import clean_fmt, gen_id_level2, gen_id_cus2, parse, fmt_stringType
from generate_feature import cal_terminate_info, get_card_ts, get_lxm, cal_os_rate, \
                            cal_renounces, get_living_inst, cal_percent_remain, \
                            get_od, agg_ts_od, get_lxm_od, cal_od_rate, get_mthly_pmt,\
                            get_by_loantype, cal_mt_pmt_rate, get_in_ln_grp, \
                            cal_ln_grp_lxm, console_feature, transform_WOE
from model_inference import score_scaling


if __name__ == "__main__":
    begin = time.time()
    #______________________________Data Preprocessing______________________________
    df = pd.read_parquet(r'data_input/TO_TEST.parquet')

    df['created_on'] = df['created_time'].apply(lambda row: formatdate(row))
    df['credit_history'] = df['credit_history'].apply(lambda row: clean_fmt(row))
    df['created_time_fmt'] = df['created_time'].apply(lambda row: to_dt_fmt(row))
    df['id_customer2'] = df.apply(lambda row: gen_id_cus2(row.id, row.created_time_fmt), axis=1)
    dev_df = df.copy()

    # ROOT level
    id_col_list = ['id_customer2']
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
    noninstall['loan_code_lv2'] = gen_id_level2(noninstall, 'id_customer2')
    install['loan_code_lv2'] = gen_id_level2(install, 'id_customer2')
    card['loan_code_lv2'] = gen_id_level2(card, 'id_customer2')

    noninstall = fmt_stringType(noninstall, ['Profiles'])
    install = fmt_stringType(install, ['Profiles', 'InstGuarantees'])
    card = fmt_stringType(card, ['Profiles', 'CardsGuarantees'])

    
    # Time series level
    id_col_list_ts = ['loan_code_lv2', 'id_customer2', 'CommonData.CBContractCode']
    field = ['Profiles']
    ts_card = parse(card, field, id_col_list_ts)
    ts_card = handle_missing_column(ts_card, ts_col)    
    ts_noninstall = parse(noninstall, field, id_col_list_ts)
    ts_noninstall = handle_missing_column(ts_noninstall, ts_col)     
    ts_install = parse(install, field, id_col_list_ts)
    ts_install = handle_missing_column(ts_install, ts_col)     


    # ______________________________Generate features______________________________

    # 1. contracts_summary_terminates
    df_tm = cal_terminate_info(df_root)

    # 2/3. cc_os_rate_max_l25m, cc_os_rate_avg_l25m
    df_cc_os = (ts_card.
               pipe(get_card_ts, card).
               pipe(get_lxm, dev_df).
               pipe(cal_os_rate)
              )

    # 4. card_summary_renounces
    df_card_rn = cal_renounces(df_root)

    # 5. pct_rm_term_lv
    df_inst_rm_lv = (install.
                     pipe(get_living_inst).
                     pipe(cal_percent_remain)
                    )
    
    # 6. od_utl_rate_max_l25m
    df_od_utl = (ts_noninstall.
                     pipe(get_od, noninstall).
                     pipe(agg_ts_od).
                     pipe(get_lxm_od, dev_df).
                     pipe(cal_od_rate)
                    )
    
    # 7/8. consumerL_mth_pmt_sum_l25m, cashL_mth_pmt_sum_l4m
    df_consumerL = (ts_install.
                        pipe(get_mthly_pmt, install).
                        pipe(get_by_loantype).
                        pipe(cal_mt_pmt_rate, dev_df)
                        )

    # 9. in_ln_grp_max_l4m
    df_ins_lnGrp = (ts_install.
                        pipe(get_in_ln_grp).
                        pipe(cal_ln_grp_lxm, dev_df)
                    )
    
    # Finalize
    conso_list = [df_tm, df_cc_os, df_card_rn, df_inst_rm_lv, df_od_utl, df_consumerL, df_ins_lnGrp]
    df_fn = console_feature(conso_list)
    woe_feature = transform_WOE(df_fn)

    # ______________________________Model Inference______________________________
    model_path = 'artifacts/Fiza_PCB_score_10Mar25.sav'
    loaded_model = pickle.load(open(model_path, 'rb'))
    woe_feature['const'] = 1

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

    predict = loaded_model.predict(exog=woe_feature[final_feat+['const']])
    predict_score = score_scaling(offset, factor, predict) # Output
    score_feature = (-1*(woe_feature[final_feat]*beta+intercept/n)*factor+offset/n) 
    score_feature = round(score_feature, 2) # Output

    print('Score:', predict_score)
    print(score_feature[final_feat].to_dict(orient='records'))

    print(round(time.time()  - begin,3),'seconds')