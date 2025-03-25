import pandas as pd
from src.utils_ops import *
import ast

# _______________________________________parse PCB____________________________________________________

def flatten(cell:list):
    if not isinstance(cell, (list, pd.Series)):
        cell = [cell]
    flat_lst = [pd.json_normalize(item) for item in cell]
    return flat_lst
    
def attach_id(id_col_list, df_parent, flat_lst):
    '''
    flat_lst list: list of dataframe
    id_col list: assign name of id column
    df_parent pd.DataFrame: parent node
    '''
    for table in flat_lst:
        for id_col in id_col_list:
            table[id_col] = df_parent[id_col]
    return flat_lst

def explode_nest(result:list, id_col_list:list, df_parent:pd.DataFrame, node):
    flat_lst = flatten(node)
    flat_lst_attached = attach_id(id_col_list, df_parent, flat_lst)
    result += flat_lst_attached    


def gen_id_level2(df, id_level1, id_level2='CommonData.CBContractCode'):
    try:
        id_lv2 = df[id_level1].astype(str) + '_' + df[id_level2]
    except:
        id_lv2 = None
    return id_lv2


def handle_missing_column(df, col_list, treat_value=np.nan):
    missing_cols = set(col_list) - set(df.columns)

    for col in missing_cols:
        df[col] = treat_value

    return df


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


# def to_json_fmt(node):
#     try:
#         fmt = str(node)
#         fmt = ast.literal_eval(fmt)
#     except:
#         fmt = {}
#     return fmt

# def clean_fmt(t):
#     t2 = t.replace('""', '"')
#     t2 = t2.replace('null', 'None')
#     t2 = t2.replace('"{', '{')
#     t2 = t2.replace('}"', '}')
#     return t2


# def fmt_stringType(df, json_cell_col):
#     for col in json_cell_col:
#         try:
#             df[col] = df[col].astype(str)
#         except:
#             df[col] = None
#         return df

# def gen_id_cus2(id_customer, created_time_fmt:str):
#     try: 
#         id_customer = str(int(id_customer))
#     except:
#         id_customer = str(id_customer)
#     return id_customer +'_'+ created_time_fmt