import json
import ast
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime as dt
from math import ceil



# _______________________________________parse PCB____________________________________________________
def to_json_fmt(node):
    try:
        fmt = str(node)
        fmt = ast.literal_eval(fmt)
    except:
        fmt = {}
    return fmt

def flatten(node:list):
    item_list = to_json_fmt(node)
    if not isinstance(item_list, list):
        item_list = [item_list]
    flat_lst = [pd.json_normalize(item) for item in item_list]
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



# _______________________________________Time format____________________________________________________
def ym_format(input_str, fmt):
    try: 
        output_dt = dt.strptime(str(input_str), fmt)
    except:
        output_dt = np.nan
    return output_dt

def create_ym_format(df, ym_col, fmt='%Y%m'):
    return df[ym_col].apply(lambda row: ym_format(row, fmt))


def month_diff(start, end):
  try:
      return ceil((end - start).days / 30.5)
  except:
      return np.nan      
  
def ratio(numerator, denominator):
    return np.nan if denominator == 0 else numerator/denominator


def get_dt_format(row):
    try:
        dt_type = dt.strptime(row, '%d%m%Y')
    except:
        dt_type = None
    return dt_type

def format_start_date(row):
    if len(str(row))<7:
        row = None
    else:
         row = '0'+str(row) if len(str(row))<8 else str(row)
    return row

def formatdate(row):
    row = str(row)
    row = dt.strptime(row, '%Y-%m-%d %H:%M:%S')
    return row

def to_dt_fmt(row, format_dt='%Y-%m-%d %H:%M:%S', format_string='%Y%m%d'):
    """ row Timestamp """
    row = str(row)
    row = dt.strptime(row, format_dt)
    return dt.strftime(row, format_string)


def concat_ym(ReferenceYear, ReferenceMonth):
    if ReferenceYear == None or ReferenceMonth == None:
        return None
    else:
        return ReferenceYear+ReferenceMonth