import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import secrets

# 配置随机种子（可选），保证结果可复现
np.random.seed(1000)
random.seed(1000)

# 定义各角色的参数配置
# 这里不直接输出二分类的smart money，而是在后续用连续指标进行计算
roles_config = {
    'Newbie': {
        'proportion': 0.25,   # 25%
        'first_tx_date_range': (datetime(2023, 1, 1), datetime(2025, 1, 1)),
        'active_duration_days_range': (0, 180),
        'total_tx_count_dist': ('lognormal_int', {'mean': 0.0, 'sigma': 0.8}),
        'total_in_value_dist': ('lognormal', {'mean': -2.3, 'sigma': 1.0}),
        'net_flow_fraction_range': (-0.1, 1.0),
        'avg_gas_price_dist': ('lognormal', {'mean': 3.4, 'sigma': 0.7}),
        'tx_per_day_range': (1, 3)
    },
    'SeasonedInvestor': {
        'proportion': 0.10,   # 10%
        'first_tx_date_range': (datetime(2015, 8, 1), datetime(2019, 1, 1)),
        'active_duration_days_range': (1000, 3500),
        'total_tx_count_dist': ('lognormal_int', {'mean': 3.0, 'sigma': 1.0}),
        'total_in_value_dist': ('lognormal', {'mean': 4.6, 'sigma': 1.3}),
        'net_flow_fraction_range': (0.0, 1.0),
        'avg_gas_price_dist': ('lognormal', {'mean': 3.7, 'sigma': 0.6}),
        'tx_per_day_range': (1, 3)
    },
    'HighFreqTrader': {
        'proportion': 0.10,   # 10%
        'first_tx_date_range': (datetime(2018, 1, 1), datetime(2022, 1, 1)),
        'active_duration_days_range': (200, 1200),
        'total_tx_count_dist': ('lognormal_int', {'mean': 6.0, 'sigma': 1.0}),
        'total_in_value_dist': ('lognormal', {'mean': 3.9, 'sigma': 1.2}),
        'net_flow_fraction_range': (-0.2, 0.2),
        'avg_gas_price_dist': ('lognormal', {'mean': 4.1, 'sigma': 0.7}),
        'tx_per_day_range': (5, 20)
    },
    'AirdropHunter': {
        'proportion': 0.10,   # 10%
        'first_tx_date_range': (datetime(2020, 1, 1), datetime(2023, 1, 1)),
        'active_duration_days_range': (100, 800),
        'total_tx_count_dist': ('lognormal_int', {'mean': 3.0, 'sigma': 1.0}),
        'total_in_value_dist': ('lognormal', {'mean': 1.6, 'sigma': 1.0}),
        'net_flow_fraction_range': (-0.3, 0.1),
        'avg_gas_price_dist': ('lognormal', {'mean': 3.4, 'sigma': 0.6}),
        'tx_per_day_range': (1, 5)
    },
    'NFTPlayer': {
        'proportion': 0.15,   # 15%
        'first_tx_date_range': (datetime(2021, 1, 1), datetime(2023, 1, 1)),
        'active_duration_days_range': (100, 1000),
        'total_tx_count_dist': ('lognormal_int', {'mean': 3.9, 'sigma': 1.1}),
        'total_in_value_dist': ('lognormal', {'mean': 2.3, 'sigma': 1.5}),
        'net_flow_fraction_range': (-0.5, 0.2),
        'avg_gas_price_dist': ('lognormal', {'mean': 4.5, 'sigma': 1.0}),
        'tx_per_day_range': (1, 10)
    },
    'ArbitrageBot': {
        'proportion': 0.10,   # 10%
        'first_tx_date_range': (datetime(2019, 1, 1), datetime(2023, 1, 1)),
        'active_duration_days_range': (100, 1000),
        'total_tx_count_dist': ('lognormal_int', {'mean': 5.3, 'sigma': 1.0}),
        'total_in_value_dist': ('lognormal', {'mean': 3.4, 'sigma': 1.0}),
        'net_flow_fraction_range': (-0.5, 0.1),
        'avg_gas_price_dist': ('lognormal', {'mean': 4.4, 'sigma': 0.8}),
        'tx_per_day_range': (5, 30)
    },
    'MEVBot': {
        'proportion': 0.05,   # 5%
        'first_tx_date_range': (datetime(2020, 1, 1), datetime(2024, 1, 1)),
        'active_duration_days_range': (100, 1500),
        'total_tx_count_dist': ('lognormal_int', {'mean': 6.9, 'sigma': 1.0}),
        'total_in_value_dist': ('lognormal', {'mean': 3.9, 'sigma': 1.2}),
        'net_flow_fraction_range': (-1.0, 0.1),
        'avg_gas_price_dist': ('lognormal', {'mean': 5.0, 'sigma': 1.0}),
        'tx_per_day_range': (10, 100)
    },
    'LongTermHolder': {
        'proportion': 0.05,   # 5%
        'first_tx_date_range': (datetime(2015, 8, 1), datetime(2018, 1, 1)),
        'active_duration_days_range': (0, 3500),
        'total_tx_count_dist': ('lognormal_int', {'mean': -0.5, 'sigma': 0.5}),
        'total_in_value_dist': ('lognormal', {'mean': 4.0, 'sigma': 1.5}),
        'net_flow_fraction_range': (0.0, 1.0),
        'avg_gas_price_dist': ('lognormal', {'mean': 3.0, 'sigma': 0.8}),
        'tx_per_day_range': (1, 2)
    }
}

# 计算总样本数（总量至少12000条记录）
total_addresses = 12000
role_counts = {role: int(conf['proportion'] * total_addresses) for role, conf in roles_config.items()}
sum_counts = sum(role_counts.values())
if sum_counts < total_addresses:
    last_role = list(roles_config.keys())[-1]
    role_counts[last_role] += (total_addresses - sum_counts)

# 帮助函数：根据配置的分布类型采样一个值
def sample_from_dist(dist_tuple):
    dist_type, params = dist_tuple
    if dist_type == 'lognormal':
        value = np.random.lognormal(mean=params['mean'], sigma=params['sigma'])
        return value
    elif dist_type == 'lognormal_int':
        value = np.random.lognormal(mean=params['mean'], sigma=params['sigma'])
        return max(1, int(round(value)))
    elif dist_type == 'uniform_int':
        return random.randint(params['low'], params['high'])
    elif dist_type == 'uniform':
        return random.uniform(params['low'], params['high'])
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

# 初始化数据列表
addresses = []
first_tx_times = []
active_durations = []
active_days_counts = []
avg_tx_intervals = []
total_tx_counts = []
total_in_values = []
total_out_values = []
net_flows = []
avg_gas_prices = []
# 这里先不直接输出智能资金标签，而是在后续计算连续得分
# 初始异常标记默认全为0
is_outlier_flags = []

for role, conf in roles_config.items():
    n = role_counts.get(role, 0)
    for i in range(n):
        # 生成地址（模拟以太坊地址）
        addr = '0x' + secrets.token_hex(20)
        
        # 随机生成首次交易日期
        start_date, end_date = conf['first_tx_date_range']
        total_days_range = (end_date - start_date).days
        offset_days = random.randint(0, total_days_range) if total_days_range > 0 else 0
        first_tx_date = start_date + timedelta(days=offset_days)
        first_tx_str = first_tx_date.strftime("%Y-%m-%d")
        
        # 随机生成活跃时长（天数）
        dur_min, dur_max = conf['active_duration_days_range']
        active_dur = random.randint(dur_min, dur_max) if dur_max > dur_min else dur_min
        last_tx_date = first_tx_date + timedelta(days=active_dur)
        now_date = datetime.now()
        if last_tx_date > now_date:
            last_tx_date = now_date
            active_dur = (last_tx_date - first_tx_date).days
        
        # 生成总交易次数
        total_tx = sample_from_dist(conf['total_tx_count_dist'])
        total_tx = max(1, total_tx)
        
        # 生成活跃天数（依据每天交易笔数区间估算）
        pad_min, pad_max = conf['tx_per_day_range']
        tx_per_day = random.randint(pad_min, pad_max) if pad_max > pad_min else pad_min
        active_days = int(np.ceil(total_tx / tx_per_day))
        if active_days > active_dur + 1:
            active_days = active_dur + 1
        active_days = min(active_days, total_tx)
        
        # 生成总流入金额
        total_in = sample_from_dist(conf['total_in_value_dist'])
        # 随机生成净流入比例（相对于总流入）
        frac_min, frac_max = conf['net_flow_fraction_range']
        net_frac = random.uniform(frac_min, frac_max)
        if net_frac < -0.9:
            net_frac = -0.9
        net_flow_val = net_frac * total_in
        total_out = total_in - net_flow_val
        if total_out < 0:
            total_out = 0
            net_flow_val = total_in
        
        # 生成平均Gas价格（Gwei）
        avg_gas = sample_from_dist(conf['avg_gas_price_dist'])
        
        # 计算平均交易间隔（天/笔）
        avg_interval = active_dur / total_tx if total_tx > 0 else 0.0
        
        # 添加数据
        addresses.append(addr)
        first_tx_times.append(first_tx_str)
        active_durations.append(active_dur)
        active_days_counts.append(active_days)
        avg_tx_intervals.append(round(avg_interval, 4))
        total_tx_counts.append(total_tx)
        total_in_values.append(round(total_in, 4))
        total_out_values.append(round(total_out, 4))
        net_flows.append(round(net_flow_val, 4))
        avg_gas_prices.append(round(avg_gas, 4))
        is_outlier_flags.append(0)

# 组装成DataFrame
df = pd.DataFrame({
    "Address": addresses,
    "FirstTxTime": first_tx_times,
    "ActiveDuration": active_durations,
    "ActiveDaysCount": active_days_counts,
    "AvgTxInterval": avg_tx_intervals,
    "TotalTxCount": total_tx_counts,
    "TotalInValue": total_in_values,
    "TotalOutValue": total_out_values,
    "NetFlow": net_flows,
    "AvgGasPrice": avg_gas_prices,
    "is_outlier": is_outlier_flags
})

# 注入异常值 (约2%的数据作为异常)
num_records = len(df)
num_outliers = max(1, int(0.02 * num_records))
outlier_indices = random.sample(range(num_records), num_outliers)

for idx in outlier_indices:
    outlier_type = random.choice(["tx_count", "gas_price", "value_flow"])
    if outlier_type == "tx_count":
        original_count = int(df.at[idx, "TotalTxCount"])
        new_count = original_count * random.randint(5, 10)
        df.at[idx, "TotalTxCount"] = new_count
        dur = df.at[idx, "ActiveDuration"]
        df.at[idx, "AvgTxInterval"] = round(dur / new_count if new_count > 0 else 0.0, 4)
        active_dur = int(dur)
        df.at[idx, "ActiveDaysCount"] = active_dur + 1 if active_dur >= 0 else 1
    elif outlier_type == "gas_price":
        df.at[idx, "AvgGasPrice"] = round(random.uniform(1000, 10000), 4)
    elif outlier_type == "value_flow":
        orig_in = df.at[idx, "TotalInValue"]
        orig_out = df.at[idx, "TotalOutValue"]
        if random.random() < 0.5:
            factor = random.uniform(10, 100)
            new_in = orig_in * factor
            new_out = orig_out
            df.at[idx, "TotalInValue"] = round(new_in, 4)
            df.at[idx, "TotalOutValue"] = round(new_out, 4)
        else:
            factor = random.uniform(5, 20)
            new_in = orig_in * factor
            new_out = orig_out * factor
            df.at[idx, "TotalInValue"] = round(new_in, 4)
            df.at[idx, "TotalOutValue"] = round(new_out, 4)
        df.at[idx, "NetFlow"] = round(df.at[idx, "TotalInValue"] - df.at[idx, "TotalOutValue"], 4)
    df.at[idx, "is_outlier"] = 1

# -------------------------------------------
# 计算“smart_money_score”：连续指标，模糊智能资金边界
# 采用 TotalTxCount, TotalInValue, AvgGasPrice, ActiveDuration 四个指标
# 先对每个指标做min-max归一化，再按权重加权（例如权重0.3,0.3,0.2,0.2），再加上小幅随机噪声
# 此得分在0~1之间，分数接近0.5的地址很难直接区分是否为smart money
# 如果需要二分类标签，可按阈值（如0.5）划分

features_for_score = ["TotalTxCount", "TotalInValue", "AvgGasPrice", "ActiveDuration"]
for feat in features_for_score:
    min_val = df[feat].min()
    max_val = df[feat].max()
    # 若最大值和最小值相等则全部归一化为0.5
    if max_val == min_val:
        df[feat + "_norm"] = 0.5
    else:
        df[feat + "_norm"] = (df[feat] - min_val) / (max_val - min_val)

# 设定各指标权重
w_tx = 0.3
w_in = 0.3
w_gas = 0.2
w_dur = 0.2

df["smart_money_score"] = (w_tx * df["TotalTxCount_norm"] +
                           w_in * df["TotalInValue_norm"] +
                           w_gas * df["AvgGasPrice_norm"] +
                           w_dur * df["ActiveDuration_norm"])

# 加入随机噪声，范围[-0.1,0.1]
noise = np.random.uniform(-0.1, 0.1, size=len(df))
df["smart_money_score"] = df["smart_money_score"] + noise
df["smart_money_score"] = df["smart_money_score"].clip(0, 1)

# 可选：按阈值0.5得到二分类标签，但由于得分连续且大部分在阈值附近，智能资金的边界较为模糊
df["is_smart_money"] = (df["smart_money_score"] > 0.2).astype(int)

# 删除归一化临时列
for feat in features_for_score:
    del df[feat + "_norm"]

# 保存最终结果为CSV文件
df.to_csv("ethereum_address_features_fuzzy.csv", index=False)
print(f"生成 {len(df)} 条地址数据，保存为 ethereum_address_features_fuzzy.csv")