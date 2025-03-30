import pandas as pd
from pandas import read_csv, DataFrame
from typing import List
import numpy as np
from datetime import datetime


def preprocess_data(fpath, save_path=None):
    data = read_csv(fpath)
    data = change_column_name(data, 'AddressID', 'eth_address')
    data = change_column_name(data, 'label', 'is_smart_money')
    
    # 处理时间列
    data['FirstTxTime'] = data['FirstTxTime'].apply(convert_time_to_number)
    data['LastTxTime'] = data['LastTxTime'].apply(convert_time_to_number)
    
    if save_path:
        data.to_csv(save_path, index=False)
    return data


def convert_time_to_number(time_str: str) -> int:
    """将时间字符串转换为数字格式"""
    dt = datetime.strptime(time_str, '%Y-%m-%d')
    return int(dt.timestamp())


def preprocess_special_column(data: DataFrame, col_name: str|List[str], method_type: str):
    if isinstance(col_name, str):
        if method_type == 'drop':
            data = data.drop(col_name, axis=1)
        elif method_type == 'PCA':
            raise NotImplementedError("PCA method is not implemented")
        return data
    elif isinstance(col_name, List):
        for col in col_name:
            data = preprocess_special_column(data, col, method_type)
        return data

def switch_columns(data: DataFrame, col_name1: str, col_name2: str):
    data[col_name1], data[col_name2] = data[col_name2], data[col_name1]
    return data


def change_column_name(data: DataFrame, col_name: str, new_col_name: str):
    data.rename(columns={col_name: new_col_name}, inplace=True)
    return data


def map_column_value(data: DataFrame, col_name: str, mapping_dict: dict):
    data[col_name] = data[col_name].map(mapping_dict)
    return data


if __name__ == '__main__':
    import sys
    preprocess_data(sys.argv[1], sys.argv[2])
