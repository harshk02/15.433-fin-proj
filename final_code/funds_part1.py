# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:04:04 2024

@author: i_bor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

raw_data = pd.read_csv('C:/Users/i_bor/OneDrive/Documentos/Ale_MFin/Fall 24/Fin Mkts/Final_project/funds_summary.csv')

#drop rows without allocation information and turnover ratio
raw_data= raw_data.dropna(subset=['per_com'])



# Identify crsp_fundno groups where tna_latest < 1000 for all rows
funds_to_drop = raw_data.groupby('crsp_fundno')['tna_latest'].transform(lambda x: (x < 10000).all() or x.isna().any())

# Filter out rows where crsp_fundno meets the drop condition
df = raw_data[~funds_to_drop]
df['crsp_fundno'].nunique()

#df[df['tna_latest'] <= 200000]['tna_latest'].dropna().plot(kind='hist', bins=30, color='blue', edgecolor='black', title='Histogram of tna_latest (<= 200,000)')
#df['index_fund_flag'].value_counts()
#df['et_flag'].value_counts()

#Create dummies
df["index_dummy"]=df['index_fund_flag'].notna().astype(int)
df["etf_dummy"]=df['et_flag'].notna().astype(int)

#Herfindahl concentration variable
df['concentration']=df.iloc[:, 9:22].pow(2).sum(axis=1)

#Change Y/N to 1/0
df[['open_to_inv','retail_fund', 'inst_fund']] = df[['open_to_inv','retail_fund', 'inst_fund']].replace({"Y": 1, "N": 0})

#Grouping by fund
df_funds = df.groupby('crsp_fundno')[['tna_latest', 'per_com', 'per_corp', 'per_govt', 'retail_fund', 'inst_fund', 'concentration', 'mgmt_fee', 'exp_ratio', 'turn_ratio', 'index_dummy', 'etf_dummy' ]].mean().reset_index()

#Dummy asset class
df_funds['per_govt'].plot(kind='hist', bins=30, color='blue', edgecolor='black', title='Histogram')
df_funds['com_dummy']=df_funds['per_com']>.5
df_funds['com_dummy']=df_funds['com_dummy'].astype(int)



#Getting fund data
unique_values = df['crsp_fundno'].unique()
# Write the unique values to a .txt file
with open('C:/Users/i_bor/OneDrive/Documentos/Ale_MFin/Fall 24/Fin Mkts/Final_project/unique_crsp_fundno.txt', 'w') as f:
    for value in unique_values:
        f.write(f"{value}\n")




returns = pd.read_csv('C:/Users/i_bor/OneDrive/Documentos/Ale_MFin/Fall 24/Fin Mkts/Final_project/returns.csv')
returns['crsp_fundno'].nunique()
returns['mret'] = pd.to_numeric(returns['mret'], errors='coerce')

df_funds_ret = returns.groupby('crsp_fundno').agg(mret_std=('mret', 'std'), mret_mean=('mret', 'mean')).reset_index()
df_funds_ret['mret_mean'].plot(kind='hist', bins=30, color='blue', edgecolor='black', title='Histogram of tna_latest (<= 200,000)')

#Join to the data frame
df_funds=df_funds.merge(df_funds_ret, on='crsp_fundno',how='inner')
#Decimals to per cent
df_funds['mret_std']=df_funds['mret_std']*100
df_funds['mret_mean']=df_funds['mret_mean']*100
df_funds['exp_ratio']=df_funds['exp_ratio']*100
df_funds['turn_ratio']=df_funds['turn_ratio']*100

# Histograms
bins = np.linspace(min(df_funds['mret_std']), max(df_funds['mret_std']), 30)

plt.hist(df_funds['mret_std'], bins=bins, alpha=0.5, color='blue', edgecolor='black')
plt.title('Histogram of Standard Deviation of all Funds')
plt.xlabel('Monthly Return Standard Deviation (%)')
plt.ylabel('Number of Funds')
plt.legend()
plt.show()

plt.hist(df_funds[df_funds['index_dummy'] == 0]['mret_std'], bins=bins, alpha=0.5, label='Index Fund', color='blue', edgecolor='black')
plt.hist(df_funds[df_funds['index_dummy'] == 1]['mret_std'], bins=bins, alpha=0.5, label='Not Index Fund', color='orange', edgecolor='black')
plt.title('Histogram of Standard Deviation for Index and Non-Index Funds')
plt.xlabel('Monthly Return Standard Deviation (%)')
plt.ylabel('Number of Funds')
plt.legend()
plt.show()

plt.hist(df_funds[df_funds['etf_dummy'] == 0]['mret_std'], bins=bins, alpha=0.5, label='ETF', color='blue', edgecolor='black')
plt.hist(df_funds[df_funds['etf_dummy'] == 1]['mret_std'], bins=bins, alpha=0.5, label='Not ETF', color='orange', edgecolor='black')
plt.title('Histogram of Standard Deviation for ETF and Non-ETF')
plt.xlabel('Monthly Return Standard Deviation (%)')
plt.ylabel('Number of Funds')
plt.legend()
plt.show()

plt.hist(df_funds[df_funds['inst_fund'] == 1]['mret_std'], bins=bins, alpha=0.5, label='Institutional Fund', color='blue', edgecolor='black')
plt.hist(df_funds[df_funds['retail_fund'] == 1]['mret_std'], bins=bins, alpha=0.5, label='Retail Fund', color='orange', edgecolor='black')
plt.title('Histogram of Standard Deviation for Retail and Institutional Funds')
plt.xlabel('Monthly Return Standard Deviation (%)')
plt.ylabel('Number of Funds')
plt.legend()
plt.show()

plt.hist(df_funds[df_funds['com_dummy'] == 0]['mret_std'], bins=bins, alpha=0.5, label='Less than 50% Stocks', color='blue', edgecolor='black')
plt.hist(df_funds[df_funds['com_dummy'] == 1]['mret_std'], bins=bins, alpha=0.5, label='More than 50% Stocks', color='orange', edgecolor='black')
plt.title('Histogram of Standard Deviation by Asset Class')
plt.xlabel('Monthly Return Standard Deviation (%)')
plt.ylabel('Number of Funds')
plt.legend()
plt.show()




# Define the dependent variable (y) and independent variables (X)
y = df_funds['mret_std']
X = df_funds[['per_com', 'per_corp', 'per_govt']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

X = df_funds[['per_com', 'per_corp', 'per_govt','tna_latest','retail_fund','index_dummy','etf_dummy','concentration','exp_ratio']]
X = sm.add_constant(X)
model = sm.OLS(y, X, missing='drop').fit()
print(model.summary())

X = df_funds[['per_com', 'per_corp', 'per_govt','tna_latest','retail_fund','index_dummy','etf_dummy','concentration','exp_ratio', 'mret_mean']]
X = sm.add_constant(X)
model = sm.OLS(y, X, missing='drop').fit()
print(model.summary())
