from functools import wraps, reduce
from datetime import datetime as dt
import pandas as pd
import numpy as np
import pickle

from src.utils_ops import create_ym_format, month_diff, ratio
from src.f_generator_ops import agg_cal, generate_feature_lxm


def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.now() - tic)
        print(f"[{func.__name__}] Shape:{result.shape}. Process time: {time_taken}s")
        return result
    return wrapper

def concat_ym(ReferenceYear, ReferenceMonth):
    if ReferenceYear == None or ReferenceMonth == None:
        return None
    else:
        return ReferenceYear+ReferenceMonth
        
# ----------------------------------------------Feature 1
@log_step
def cal_terminate_info(root):
    tm_col = []
    for i in root.columns:
        if 'Terminated' in i:
            tm_col.append(i)

    df_tm = root[['id_customer2']+tm_col]
    try:
        df_tm['contracts_summary_terminates'] = df_tm[tm_col].sum(axis=1, min_count=1)
    except:
        df_tm['contracts_summary_terminates'] = np.nan
    return df_tm[['id_customer2','contracts_summary_terminates']]


# ----------------------------------------------Feature 2.3
@log_step
def get_card_ts(ts_card, contract_level_card):

    if contract_level_card.empty:
        contract_level_card = pd.DataFrame(columns=['id_customer2', 'loan_code_lv2', 'CreditLimit'])
    if ts_card.empty:
        ts_card = pd.DataFrame(columns=['loan_code_lv2', 'id_customer2',
                                        'ReferenceYear', 'ReferenceMonth', 
                                        'Granted', 'ResidualAmount', 'Utilization'])
    
    ts_card2 = pd.merge(ts_card, 
                        contract_level_card[['loan_code_lv2', 'CreditLimit']], 
                        on=['loan_code_lv2'], how='left')
    ts_card2['ResidualAmount'] = ts_card2['ResidualAmount'].astype(float)
    ts_card2['CreditLimit'] = ts_card2['CreditLimit'].astype(float)
    ts_card2['Utilization'] = ts_card2['Utilization'].replace('', np.nan)
    ts_card2['Utilization'] = ts_card2['Utilization'].astype(float)

    card_os = agg_cal(ts_card2,
                    group_col=['id_customer2','ReferenceYear', 'ReferenceMonth'],
                    val='ResidualAmount',
                    agg_fn=['sum'],
                    sub_fn=[]).reset_index()
    
    card_lmt = agg_cal(ts_card2,
                    group_col=['id_customer2','ReferenceYear', 'ReferenceMonth'],
                    val='CreditLimit',
                    agg_fn=['sum'],
                    sub_fn=[]).reset_index()
    
    card_agg = pd.merge(card_os, card_lmt, on=['id_customer2','ReferenceYear', 'ReferenceMonth'], how='outer')
    return card_agg

@log_step
def get_lxm(card_agg, dev_df):
    df_card = pd.merge(card_agg, dev_df[['id_customer2','created_time']], how="left", on=['id_customer2'])
    df_card['ts_ym'] = df_card[['ReferenceYear', 'ReferenceMonth']]\
                            .apply(lambda row: concat_ym(row.ReferenceYear,row.ReferenceMonth),
                                       axis=1)
    df_card['ts_ym_fmt'] = create_ym_format(df_card, 'ts_ym', fmt='%Y%m')
    df_card['last_x_months'] = df_card.apply(lambda row: month_diff(row['ts_ym_fmt'], row['created_time']), axis=1)
    rename_col = {'ResidualAmount_sum':'cc_os', 'CreditLimit_sum':'cc_lmt'}
    df_card.rename(columns=rename_col, inplace=True)
    return df_card

@log_step
def cal_os_rate(df_card):
    df_card['cc_os_rate'] = df_card.apply(lambda row: ratio(row['cc_os'], row['cc_lmt']), axis=1)
    df_card_final = generate_feature_lxm(df_card,
                        group_col=['id_customer2'],
                        val_col=['cc_os_rate'], 
                        agg_fn=['max', 'sum'],
                        sub_fn=['avg'],
                        LxM=[24]).reset_index()
    return df_card_final[['id_customer2', 'cc_os_rate_avg_l25m', 'cc_os_rate_max_l25m']]


# ----------------------------------------------Feature 4
@log_step
def cal_renounces(root):
    if 'Contract.Cards.Summary.NumberOfRenounced' not in root.columns:
        df_card_rn = root[['id_customer2']].copy()
        df_card_rn['card_summary_renounces'] = np.nan
    else:
        df_card_rn = root[['id_customer2', 'Contract.Cards.Summary.NumberOfRenounced']].copy()
    
    df_card_rn.columns = ['id_customer2','card_summary_renounces']
    return df_card_rn[['id_customer2', 'card_summary_renounces']]



# ----------------------------------------------Feature 5
@log_step
def get_living_inst(contract_level_install):
    get_col = ['id_customer2', 
        'RemainingInstalmentsAmount', 
        'TotalAmount',
        'RemainingInstalmentsNumber',
        'TotalNumberOfInstalments'
        ]

    if contract_level_install.empty:
        inst_info_lv = pd.DataFrame(columns=get_col)
    else:
        living_in = contract_level_install['CommonData.ContractPhase'].isin(['LV'])
        inst_info_lv = contract_level_install[living_in][get_col]
    return inst_info_lv

@log_step
def cal_percent_remain(inst_info_lv):
    agg_inst = inst_info_lv.groupby('id_customer2').agg(
        remain_term=('RemainingInstalmentsNumber','sum'),
        total_term=('TotalNumberOfInstalments','sum')
    ).reset_index()

    agg_inst['pct_remain_term'] = agg_inst['remain_term']/agg_inst['total_term']
    get_col_rm = ['id_customer2', 'pct_remain_term']
    inst_rm_lv = agg_inst[get_col_rm].rename(columns={'pct_remain_term':'pct_rm_term_lv'})
    return inst_rm_lv[['id_customer2', 'pct_rm_term_lv']]

# ----------------------------------------------Feature 6
@log_step
def get_od(ts_noninstall, contract_level_noninstall):
        if ts_noninstall.empty:
            ts_noninstall = pd.DataFrame(columns=['loan_code_lv2', 'id_customer2',
                                                'ReferenceYear', 'ReferenceMonth',
                                                'Granted', 'Utilization'])
        if contract_level_noninstall.empty:
            contract_level_noninstall = pd.DataFrame(columns=['loan_code_lv2', 'CommonData.TypeOfFinancing'])

        filter_ts = ~ts_noninstall['loan_code_lv2'].isna()
        filter_con = ~contract_level_noninstall['loan_code_lv2'].isna()
        ts_noninstall2 = pd.merge(ts_noninstall[filter_ts], 
                            contract_level_noninstall[filter_con][['loan_code_lv2','CommonData.TypeOfFinancing']], 
                            on=['loan_code_lv2'], how='left')
        ts_noninstall2 = ts_noninstall2[ts_noninstall2['CommonData.TypeOfFinancing']=='41']
        ts_noninstall2['Granted'] = ts_noninstall2['Granted'].astype(float)
        ts_noninstall2['Utilization'] = ts_noninstall2['Utilization'].replace('', np.nan)
        ts_noninstall2['Utilization'] = ts_noninstall2['Utilization'].astype(float)
        return ts_noninstall2

@log_step
def agg_ts_od(ts_noninstall2):
    nonInst_lmt = agg_cal(ts_noninstall2,
                    group_col=['id_customer2','ReferenceYear', 'ReferenceMonth'],
                    val='Granted',
                    agg_fn=['sum'],
                    sub_fn=[]).reset_index()

    nonInst_utl = agg_cal(ts_noninstall2,
                        group_col=['id_customer2','ReferenceYear', 'ReferenceMonth'],
                        val='Utilization',
                        agg_fn=['sum'],
                        sub_fn=[]).reset_index()
    nonInstInfo = pd.merge(nonInst_lmt, nonInst_utl, on=['id_customer2','ReferenceYear', 'ReferenceMonth'], how='outer').reset_index()
    return nonInstInfo

@log_step
def get_lxm_od(nonInstInfo, dev_df):
    nonInstInfo['ts_ym'] = nonInstInfo[['ReferenceYear', 'ReferenceMonth']]\
                            .apply(lambda row: concat_ym(row.ReferenceYear,row.ReferenceMonth),
                                       axis=1)
    nonInstInfo['ts_ym_fmt'] = create_ym_format(nonInstInfo, 'ts_ym', fmt='%Y%m')

    df_nonIns = pd.merge(dev_df[['id_customer2','created_time']], nonInstInfo, how="left", on=['id_customer2'])
    df_nonIns['last_x_months'] = df_nonIns.apply(lambda row: month_diff(row['ts_ym_fmt'], row['created_time']), axis=1)
    df_nonIns.rename(columns={'Granted_sum':'od_lmt'}, inplace=True)
    df_nonIns.rename(columns={'Utilization_sum':'od_utl'}, inplace=True)
    return df_nonIns

@log_step
def cal_od_rate(df_nonIns):
    df_nonIns['od_utl_rate'] = df_nonIns['od_utl']/df_nonIns['od_lmt']
    df_nonIns_final = generate_feature_lxm(df_nonIns,
                        group_col=['id_customer2'],
                        val_col=['od_lmt', 'od_utl', 'od_utl_rate'], 
                        agg_fn=['mean','min','max','sum'],
                        LxM=[24]).reset_index()
    df_nonIns_final = df_nonIns_final.replace(np.inf, 1)
    return df_nonIns_final[['id_customer2', 'od_utl_rate_max_l25m']]

# ----------------------------------------------Feature 7.8
@log_step
def get_mthly_pmt(ts_install, contract_level_install):
        if contract_level_install.empty:
            contract_level_install = pd.DataFrame(columns=['loan_code_lv2', 'TotalAmount', 'TotalNumberOfInstalments', 'CommonData.TypeOfFinancing'])
        if ts_install.empty:
            ts_install = pd.DataFrame(columns=['loan_code_lv2', 'id_customer2',
                                                'ReferenceYear', 'ReferenceMonth',
                                                'TotalAmount', 'TotalNumberOfInstalments'])
            
        contract_level_install['ins_mthly_pmt'] = contract_level_install['TotalAmount']/contract_level_install['TotalNumberOfInstalments']
        filter_ts = ~ts_install['loan_code_lv2'].isna()
        filter_con = ~contract_level_install['loan_code_lv2'].isna()

        ts_install2 = pd.merge(ts_install[filter_ts], 
                        contract_level_install[filter_con][['loan_code_lv2','ins_mthly_pmt', 'CommonData.TypeOfFinancing']],
                        on=['loan_code_lv2'], 
                        how='left')
        return ts_install2

@log_step
def get_by_loantype(ts_install2):
    ts_consumerLoan = ts_install2[ts_install2['CommonData.TypeOfFinancing']=='23']
    ts_cashLoan     = ts_install2[ts_install2['CommonData.TypeOfFinancing']=='22']

    group_col_inst =['id_customer2','ReferenceYear', 'ReferenceMonth']
    cashLoan = agg_cal(ts_cashLoan,
                       group_col=group_col_inst,
                       val='ins_mthly_pmt',
                       agg_fn=['sum'],
                       sub_fn=[]).reset_index()
    
    consumerLoan = agg_cal(ts_consumerLoan,
                        group_col=group_col_inst,
                        val='ins_mthly_pmt',
                        agg_fn=['sum'],
                        sub_fn=[]).reset_index()
    
    cashLoan = cashLoan.rename(columns={'ins_mthly_pmt_sum':'cashL_mth_pmt'})
    consumerLoan = consumerLoan.rename(columns={'ins_mthly_pmt_sum':'consumerL_mth_pmt'})
    inst_lmt = pd.merge(cashLoan, consumerLoan, on=group_col_inst, how='outer')
    return inst_lmt
    
@log_step
def cal_mt_pmt_rate(inst_lmt, dev_df) :    
    df_Ins = pd.merge(dev_df[['id_customer2', 'created_time']], inst_lmt, how="left", on=['id_customer2'])
    df_Ins['ts_ym'] = df_Ins.apply(lambda row: concat_ym(row.ReferenceYear,row.ReferenceMonth),
                                       axis=1)
    df_Ins['ts_ym_fmt'] = create_ym_format(df_Ins, 'ts_ym', fmt='%Y%m')

    df_Ins['last_x_months'] = df_Ins.apply(lambda row: month_diff(row['ts_ym_fmt'], row['created_time']), axis=1)
    df_Ins_final = generate_feature_lxm(df_Ins,
                            group_col=['id_customer2'],
                            val_col=['cashL_mth_pmt', 'consumerL_mth_pmt'], 
                            agg_fn=['sum'],
                            LxM=[3, 24]).reset_index()
    return df_Ins_final[['id_customer2', 'consumerL_mth_pmt_sum_l25m', 'cashL_mth_pmt_sum_l4m']]


# ----------------------------------------------Feature 9
@log_step
def get_in_ln_grp(ts_install):
    group_col = ['id_customer2','ReferenceYear', 'ReferenceMonth']
    if ts_install.empty:
        ts_install = pd.DataFrame(columns=['loan_code_lv2', 'id_customer2',
                                            'ReferenceYear', 'ReferenceMonth',
                                            'Status'])
    ts_install.replace('', np.nan, inplace=True)
    ts_install['Status'] = ts_install['Status'].replace('null', np.nan)
    ts_install['Status'] = ts_install['Status'].astype(float)
    InsStatus = agg_cal(ts_install,
                    group_col=group_col,
                    val='Status',
                    agg_fn=['max'],
                    sub_fn=[]).reset_index()
    InsStatus.rename(columns={'Status_max':'in_ln_grp'}, inplace=True)
    return InsStatus

@log_step
def cal_ln_grp_lxm(InsStatus, dev_df):
    dfStatus = pd.merge(dev_df[['id_customer2', 'created_time']], InsStatus, how="left", on=['id_customer2'])
    dfStatus['ts_ym'] = dfStatus['ReferenceYear'] + dfStatus['ReferenceMonth']
    dfStatus['ts_ym_fmt'] = create_ym_format(dfStatus, 'ts_ym', fmt='%Y%m')
    dfStatus['last_x_months'] = dfStatus.apply(lambda row: month_diff(row['ts_ym_fmt'], row['created_time']), axis=1)

    df_Status_final = generate_feature_lxm(dfStatus,
                            group_col=['id_customer2'],
                            val_col=['in_ln_grp'], 
                            agg_fn=['max'],
                            sub_fn=[],
                            LxM=[3]).reset_index()
    return df_Status_final[['id_customer2', 'in_ln_grp_max_l4m']]


# ----------------------------------------------Consolidate features
def console_feature(conso_list):
    join_col = ['id_customer2']
    conso_feature = reduce(lambda  left,right: pd.merge(left, right, on=join_col, how='outer'), conso_list)
    return conso_feature

def transform_WOE(conso_feature):
    bin_file_name = 'artifacts/BinProcess_10Mar25.sav'
    loaded_binning = pickle.load(open(bin_file_name, 'rb'))

    feature_col = list(set(conso_feature.columns) - set(['id_customer2']))
    woe_feature = loaded_binning.transform(conso_feature[feature_col], metric="woe")
    woe_feature['id_customer2'] = conso_feature['id_customer2']
    return woe_feature

