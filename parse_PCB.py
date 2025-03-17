import pandas as pd
from datetime import datetime as dt
from src.utils_ops import *
import numpy as np

def clean_fmt(t):
    t2 = t.replace('""', '"')
    t2 = t2.replace('null', '"null"')
    t2 = t2.replace('"{', '{')
    t2 = t2.replace('}"', '}')
    return t2

def gen_id_level2(df, id_level1):
    try:
        id_lv2 = df[id_level1].astype(str) + '_' + df['CommonData.CBContractCode']
    except:
        id_lv2 = None
    return id_lv2

def fmt_stringType(df, json_cell_col):
    for col in json_cell_col:
        try:
            df[col] = df[col].astype(str)
        except:
            df[col] = None
        return df

def gen_id_cus2(id_customer, created_time_fmt:str):
    try: 
        id_customer = str(int(id_customer))
    except:
        id_customer = str(id_customer)
    return id_customer +'_'+ created_time_fmt



def parse(df_to_parse, field, id_col_list):
    df_holder = []
    try:
        df_to_parse[id_col_list+field].apply(lambda row: explode_nest(df_holder, 
                                                                    id_col_list, 
                                                                    row, 
                                                                    row[field[0]]), 
                                                                    axis=1)
        df_parsed = pd.concat(df_holder).reset_index(drop=True)
    except:
        df_parsed = pd.DataFrame()
    return df_parsed
