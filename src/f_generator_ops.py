from functools import reduce
import pandas as pd
import numpy as np
from src.utils_ops import create_ym_format, month_diff

def time_travel(df, last_n_months):
    start = (df['last_x_months'] >= 0) 
    end = (df['last_x_months'] <= (last_n_months))
    filter_last_n_mths = (start & end)
    return df[filter_last_n_mths]

def agg_cal(input_df:pd.DataFrame, 
            group_col:list, 
            val:str,
            mth_counts:float=np.nan,
            agg_fn:list = ['mean','min','max','sum','count'],
            sub_fn:list = ['avg']
           ):    
    group_df = input_df[[val]+group_col].groupby(group_col, dropna=False)
    df = group_df.agg({val:agg_fn})
    df.columns = agg_fn
    all_func = agg_fn

    if sub_fn:
        all_func = (agg_fn + sub_fn) 
        df['avg'] = df['sum']/mth_counts
            
    rename_cols = [val+'_'+F for F in (all_func)]
    df.columns = rename_cols
    return df

def generate_feature_lxm(df, 
                         group_col:list, 
                         val_col:list, 
                         agg_fn:list, 
                         sub_fn:list = ['avg'], 
                         LxM:list=[3,6,9]):
    """
        required columns: ['last_x_months']
    """
    result_lst = []
    for val in val_col:
        for mth_counts in LxM: 
            mth_counts += 1 # plus 1 month distance for PCB data to update
            df_lxm = time_travel(df, mth_counts)
            result = agg_cal(df_lxm, group_col, val, mth_counts, agg_fn, sub_fn)
            result.columns = [col+f'_l{mth_counts}m' for col in result.columns]
            result_lst.append(result)
        
    final_df = reduce(lambda  left,right: pd.merge(left, right, on=group_col, how='outer'), result_lst)
    return final_df



def rule_pcb_info(df, fil_col:list):
    """
    Only use report that last updated last 6 months until now
    input column: DateOfLastUpdate (PCB), CreditLimit, ResidualAmount
    """
    df['UpdateDateFmt'] = create_ym_format(df, 'CommonData.DateOfLastUpdate', fmt='%d%m%Y')
    df['mth_snc_last_update'] = df.apply(lambda row: month_diff(row['UpdateDateFmt'], row['process_date']), axis=1)

    df_filtered = df.copy()
    most_update_filter = (df_filtered['mth_snc_last_update'].isna())|(df_filtered['mth_snc_last_update']>6)
    # if contract is last updated in the past >6 months -> not include in calculation
    df_filtered.loc[most_update_filter, fil_col] = np.nan
    return df_filtered