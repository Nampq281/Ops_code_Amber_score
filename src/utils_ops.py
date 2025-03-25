import numpy as np
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime as dt
from math import ceil

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
    

