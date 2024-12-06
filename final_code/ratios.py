#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

data = pd.read_csv("data/returns.csv")
funds = pd.read_excel("data/selected_etfs.xlsx")
all_funds = pd.read_excel("data/AllFunds.xlsx")


# In[3]:


# Set parameters
benchmark = 0  # Benchmark return (0%)
confidence_level = 0.95  # 95% confidence level

# 1.  Conditional VaR (CVaR)
def calculate_cvar(returns, confidence_level):
    var = np.nanpercentile(returns, 100 * (1 - confidence_level))
    cvar = returns[returns < var].mean()
    return cvar

# 2. Lower Partial Moment (LPM)
def calculate_lpm(returns, benchmark, order=2):
    below_benchmark = returns[returns < benchmark]
    lpm = ((benchmark - below_benchmark) ** order).mean()
    return lpm

# 3. Omega Ratio
def calculate_omega(returns, benchmark):
    gains = returns[returns > benchmark].sum()
    losses = (benchmark - returns[returns < benchmark]).sum()
    return gains / losses if losses != 0 else np.inf


# 4. Conditional Drawdown (CDaR)
def calculate_cdar(returns, confidence_level):
    cumulative_returns = (1 + returns).cumprod()
    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
    var_drawdown = np.nanpercentile(drawdowns, 100 * (1 - confidence_level))
    cdar = drawdowns[drawdowns < var_drawdown].mean()
    return cdar

# 6. Downside Deviation
def calculate_downside_deviation(returns, benchmark):
    below_benchmark = returns[returns < benchmark]
    downside_deviation = np.sqrt(((benchmark - below_benchmark) ** 2).mean())
    return downside_deviation


# In[4]:


# Daily
funds_list = list(funds["crsp_fundno"])
daily_summary = pd.DataFrame(columns=["Fund Name","Skewness","Kurtosis", "CVaR","LPM","Omega Ratio","CDaR","Downside Deviation"], index=funds_list)
for fund in funds_list:
    returns = all_funds.loc[all_funds["Fund Identifier"]==fund,"Daily Return per share"]
    daily_summary.loc[fund,"Fund Name"]=funds.loc[funds["crsp_fundno"]==fund, "fund_name"].iloc[0]
    daily_summary.loc[fund,"Skewness"]=returns.skew()
    daily_summary.loc[fund,"Kurtosis"]=returns.kurtosis()
    daily_summary.loc[fund,"CVaR"]= calculate_cvar(returns, confidence_level)
    daily_summary.loc[fund,"LPM"]=calculate_lpm(returns, benchmark)
    daily_summary.loc[fund,"Omega Ratio"]=calculate_omega(returns, benchmark)
    daily_summary.loc[fund,"CDaR"]=calculate_cdar(returns, confidence_level)
    daily_summary.loc[fund,"Downside Deviation"]=calculate_downside_deviation(returns, benchmark)
daily_summary


# In[5]:


# Monthly
funds_list = list(funds["crsp_fundno"])
monthly_summary = pd.DataFrame(columns=["Fund Name"], index=funds_list)
metrics_to_standardize = ["Skewness", "Kurtosis", "CVaR", "LPM", "Omega Ratio", "CDaR", "Downside Deviation"]

# First calculate raw metrics for standardization
raw_metrics = pd.DataFrame(index=funds_list)
for fund in funds_list:
    returns = pd.to_numeric(data.loc[data["crsp_fundno"]==fund,"mret"],errors='coerce')
    monthly_summary.loc[fund,"Fund Name"]=funds.loc[funds["crsp_fundno"]==fund, "fund_name"].iloc[0]
    raw_metrics.loc[fund,"Skewness"]=returns.skew()
    raw_metrics.loc[fund,"Kurtosis"]=returns.kurtosis()
    raw_metrics.loc[fund,"CVaR"]= calculate_cvar(returns, confidence_level)
    raw_metrics.loc[fund,"LPM"]=calculate_lpm(returns, benchmark)
    raw_metrics.loc[fund,"Omega Ratio"]=calculate_omega(returns, benchmark)
    raw_metrics.loc[fund,"CDaR"]=calculate_cdar(returns, confidence_level)
    raw_metrics.loc[fund,"Downside Deviation"]=calculate_downside_deviation(returns, benchmark)

# Standardize metrics
for metric in metrics_to_standardize:
    mean = raw_metrics[metric].mean()
    std = raw_metrics[metric].std()
    monthly_summary[f"{metric}_standardized"] = (raw_metrics[metric] - mean) / std

# Add Fund Name column to monthly summary
monthly_summary["Fund Name"] = funds.loc[funds["crsp_fundno"].isin(funds_list), "fund_name"].values

monthly_summary
# print(monthly_summary)