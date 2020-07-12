'''

Creating a temporary program to create the foreign currency billings forecast

'''

import pandas as pd
import numpy as np
import pickle
from openpyxl import load_workbook

output_dict = pickle.load(open('../data/processed/final_forecast.p', 'rb'))
print(output_dict.keys())


df = output_dict['final']
#df_fcst = output_dict['forecast']
print(df.columns)
print(df['book_1Y_DC'].describe())
df = df.groupby(['curr', 'period'], as_index=False).sum()
#df_fcst = df_fcst.groupby(['curr', 'period'], as_index=False).sum()

def build_one_curr_history(this_curr, df):
    this_curr_df = df[df['curr']==this_curr].copy()
    list_to_drop = ['curr', 'Period_Weeks']
    for col in this_curr_df.columns:
        if '_US' in col:
            list_to_drop.append(col)
    this_curr_df['deferred_1Y_DC'] = this_curr_df['deferred_1Y_DC'] + this_curr_df['book_1Y_DC']

    this_curr_df = this_curr_df[['period', 'is_forecast', 'deferred_3Y_DC', 'deferred_2Y_DC', 'deferred_1Y_DC',
                'deferred_6M_DC', 'deferred_3M_DC', 'deferred_1M_DC',
            'deferred_B_DC', 'service_DC', 'recognized_DC']]
    this_curr_df['Total_Billings'] = this_curr_df.sum(axis=1)
    this_curr_df['To_Revenue_in_1M_$'] = this_curr_df['deferred_1M_DC'] + this_curr_df['service_DC'] + this_curr_df['recognized_DC']

    this_curr_df['To_Revenue_in_1M_%'] = this_curr_df['To_Revenue_in_1M_$'] / this_curr_df['Total_Billings']
    this_curr_df['To_Revenue_in_1Y_%'] = this_curr_df['deferred_1Y_DC'] / this_curr_df['Total_Billings']
    return this_curr_df

curr_list = ['AUD', 'EUR', 'GBP', 'JPY']
df_list = ['AUD_hist', 'EUR_hist', 'GBP_hist', 'JPY_hist']
#df_fcst_list = ['AUD_fcst', 'EUR_fcst', 'GBP_fcst', 'JPY_fcst']

for i in range(len(curr_list)):
    df_list[i] = build_one_curr_history(curr_list[i], df)
    df_list[i].set_index('period', inplace=True)

    #df_fcst_list[i] = build_one_curr_history(curr_list[i], df_fcst)
    #df_fcst_list[i].set_index('period', inplace=True)

print(df_list[2])

#print(df_fcst_list[1])

with pd.ExcelWriter('../output/FOREIGN_CURR_BILLINGS.xlsx') as writer:
    for i in range(len(curr_list)):
        print(curr_list[i])
        this_fcst_sheetname = curr_list[i]+'_fcst'
        df_list[i].to_excel(writer, sheet_name=curr_list[i], startrow=4, startcol=1, header=True)
        #df_fcst_list[i].to_excel(writer, sheet_name = this_fcst_sheetname, startrow=4, startcol=1, header=True)

writer.save()

## ExcelWriter for some reason uses writer.sheets to access the sheet.
## If you leave it empty it will not know that sheet Main is already there
## and will create a new sheet.
'''
print(df_EUR_hist.tail(10))

print(df_EUR_hist.columns)

# Need to drop many of the columns
#df_EUR_hist.drop(columns = ['curr', 'Period_Weeks', 'is_forecast'], inplace=True)
print(df_EUR_hist.columns)

list_to_drop = []
for col in df_EUR_hist.columns:
    if '_US' in col:
        print('this one is in it:', col)
        list_to_drop.append(col)

print(list_to_drop)
df_EUR_hist = df_EUR_hist.drop(columns = list_to_drop)
print(df_EUR_hist.head(10))
print(df_EUR_hist.columns)

df_EUR_hist = df_EUR_hist[['period', 'deferred_3Y_DC', 'deferred_2Y_DC', 'deferred_1Y_DC',
            'deferred_6M_DC', 'deferred_3M_DC', 'deferred_1M_DC',
            'deferred_B_DC', 'service_DC', 'recognized_DC']]

print(df_EUR_hist.head(10))
'''


