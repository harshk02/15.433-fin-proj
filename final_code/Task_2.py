#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

def compute_risk_metrics(data, etf_name, confidence_level=0.95, drawdown_threshold=-0.01):
    metrics = {'ETF': etf_name}
    data['Daily Return per share'] = pd.to_numeric(data['Daily Return per share'], errors='coerce')
    data['Net Asset Value per Share'] = pd.to_numeric(data['Net Asset Value per Share'], errors='coerce')

    # 1. Volatility
    metrics['Volatility'] = data['Daily Return per share'].std()

    # 2. Absolute and Relative Drawdowns
    data['Running Max NAV'] = data['Net Asset Value per Share'].cummax()
    data['Absolute Drawdown'] = data['Net Asset Value per Share'] - data['Running Max NAV']
    data['Relative Drawdown'] = (data['Net Asset Value per Share'] / data['Running Max NAV']) - 1
    metrics['Max Relative Drawdown'] = data['Relative Drawdown'].min()

    # 3. Value at Risk 
    metrics['VaR'] = np.percentile(data['Daily Return per share'].dropna(), (1 - confidence_level) * 100)

    # 4. Drawdown Frequency and Average Drawdown Magnitude
    data['In Drawdown'] = data['Relative Drawdown'] < drawdown_threshold
    drawdown_periods = []
    drawdown_start_index = None

    for i, in_drawdown in enumerate(data['In Drawdown']):
        if in_drawdown:
            if drawdown_start_index is None:
                drawdown_start_index = i
        else:
            if drawdown_start_index is not None:
                drawdown_periods.append((drawdown_start_index, i - 1))
                drawdown_start_index = None

    if drawdown_start_index is not None:
        drawdown_periods.append((drawdown_start_index, len(data) - 1))

    drawdown_metrics = []
    for start, end in drawdown_periods:
        duration = end - start + 1
        magnitude = data['Relative Drawdown'][start:end + 1].min()
        drawdown_metrics.append({'Duration': duration, 'Magnitude': magnitude})

    drawdown_metrics_df = pd.DataFrame(drawdown_metrics)
    if not drawdown_metrics_df.empty:
        metrics['Drawdown Frequency'] = len(drawdown_metrics_df)
        metrics['Average Drawdown Magnitude'] = drawdown_metrics_df['Magnitude'].mean()
    else:
        metrics['Drawdown Frequency'] = 0
        metrics['Average Drawdown Magnitude'] = 0

    return metrics

file_path = 'data/AllFunds.xlsx'  
data = pd.read_excel(file_path)
data['Fund Identifier'] = data['Fund Identifier'].astype(str)

all_metrics = []
unique_funds = data['Fund Identifier'].unique()

for fund_id in unique_funds:
    fund_data = data[data['Fund Identifier'] == fund_id].copy()
    metrics = compute_risk_metrics(fund_data, etf_name=fund_id)
    all_metrics.append(metrics)

# print(all_metrics)
metrics_df = pd.DataFrame(all_metrics)

# Normalize the metrics and keep only normalized columns
normalized_columns = {}
for col in ['Volatility', 'Max Relative Drawdown', 'VaR', 'Drawdown Frequency', 'Average Drawdown Magnitude']:
    normalized_col = col + '_Normalized'
    normalized_columns[normalized_col] = (metrics_df[col] - metrics_df[col].min()) / (metrics_df[col].max() - metrics_df[col].min())

metrics_df = pd.DataFrame(normalized_columns)
metrics_df.insert(0, 'Fund ID', unique_funds)

# Assign weights and calculate the composite risk index
weights = {
    'Volatility_Normalized': 0.3,
    'Max Relative Drawdown_Normalized': 0.25,
    'VaR_Normalized': 0.25,
    'Drawdown Frequency_Normalized': 0.1,
    'Average Drawdown Magnitude_Normalized': 0.1
}

metrics_df['Composite Risk Index'] = sum(metrics_df[col] * weight for col, weight in weights.items())
# print(metrics_df)