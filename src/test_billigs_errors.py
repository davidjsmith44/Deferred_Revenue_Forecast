


import numpy as np
import pandas as pd
import pickle
import openpyxl


def df_to_xl(df, output_file, output_tab, start_row=0, start_col=0, header=False):
    '''
    Inputs
        df - the dataframe that will be sent to excel
        output_file - the path and filename of the output file
        output_tab - the tab in the output file to send the data to.
        start_row - the first row where the data will appear. NOTE: Python treats cell A1 as row zero.
        start_column - the first column where the data will appear. NOTE: Python treats cell A1 as row zero.
        header - if the header is True, it will send in the column names of the dataframe.
                If header is false, no column names will be sent to the excel file.
                If header contains a list of strings, the list of strings will be an alias to the column names

    This function opens the workbook output_file, reads in all of the sheets to that file and sends
    the dataframe df to a location on the output_tab sheet. The location is determined by the start_row and
    start_column sent to the function.

    Python Modules to be imported for this to work
        pandas
        openpyxl
    '''

    book = openpyxl.load_workbook(output_file)

    writer = pd.ExcelWriter(output_file, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    df.to_excel(writer,
                sheet_name=output_tab,
                startrow=start_row,
                startcol=start_col,
                header=header,
                index=False)

    writer.save()

    writer.close()

    print('DataFrame is written successfully to Excel File.')

    return None

df = pd.read_excel('../data/Data_2020_P06/all_billings_inputs.xlsx', sheet_name = 'base_billings')

print(df.head(10))

df.rename(index = str, columns = {'Document Currency': 'curr',
                                 'Enterprise BU Desc': 'BU',
                                 'Invoice Fiscal Year Period Desc': 'period',
                                 'Product Config Type': 'config',
                                 'Rev Rec Category': 'rev_req_type',
                                 'Rule For Bill Date': 'rebill_rule',
                                 'Completed Sales ( DC )': 'DC_amount',
                                 'Completed Sales': 'US_amount'}, inplace=True)


curr_list = ['AUD', 'EUR', 'GBP', 'JPY']
df = df.loc[df['curr'].isin(curr_list)].copy()

period_list = ['2020-04', '2020-05', '2020-06']
df = df.loc[df['period'].isin(period_list)].copy()

rec = df[df['Sales Type']=='RECOGNIZED'].copy()
svc = df[df['Sales Type']=='PRO-SVC-INV'].copy()
dfr = df[df['Sales Type']=='DEFERRED'].copy()
nr = df[df['Sales Type']=='NON-REV'].copy()

gb_rec = rec.groupby(['curr', 'period'], as_index = False).sum()
gb_rec.drop(labels='Subscription Term', axis=1,inplace =True)

gb_svc = svc.groupby(['curr', 'period'], as_index=False).sum()
gb_svc.drop(labels='Subscription Term', axis=1,inplace =True)

dfr_b = dfr[dfr['rev_req_type']=='B'].copy()
gb_b = dfr_b.groupby(['curr', 'period'], as_index=False).sum()
gb_b.drop(labels='Subscription Term', axis=1, inplace=True)

# Tyoe A billings
dfr_a = dfr[dfr['rev_req_type']=='A'].copy()
output_file = '../data/Data_2020_P06/FOREIGN_CURR_BILLINGS.xlsx'aw
df_to_xl(dfr_a, output_file, 'deferred_A', 0, 0, header=True)

gb_a = dfr_a.groupby(['curr', 'period', 'config'], as_index=False).sum()
gb_a.drop(labels='Subscription Term', axis=1, inplace = True)

config_list = ['1Y', '2Y', '3Y', 'MTHLY']
print(gb_a['config'].value_counts())
print('gb_a is below')
print(gb_a)
gb_a_config = gb_a[gb_a['config'].isin(config_list)]
gb_a_dropped = gb_a[~gb_a['config'].isin(config_list)]
gb_a_1Y = gb_a_config[gb_a_config['config']=='1Y'].copy()
gb_a_2Y = gb_a_config[gb_a_config['config']=='2Y'].copy()
gb_a_3Y = gb_a_config[gb_a_config['config']=='3Y'].copy()
gb_a_1M = gb_a_config[gb_a_config['config']=='MTHLY'].copy()


# type D billings
dfr_d = dfr[dfr['rev_req_type']=='D'].copy()
gb_d = dfr_d.groupby(['curr', 'period', 'rebill_rule'], as_index=False).sum()
gb_d.drop(labels='Subscription Term', axis=1, inplace = True)

gb_d_mthly = gb_d[gb_d['rebill_rule'].isin(['Y1', 'Y2', 'Y3', 'YM'])].copy()
gb_d_mthly.drop(labels='rebill_rule', axis=1,inplace=True)
gb_d_mthly = gb_d_mthly.groupby(['curr', 'period']).sum()
gb_d_mthly.reset_index(inplace=True)

gb_d_qtrly = gb_d[gb_d['rebill_rule'].isin(['YQ', 'YY', 'YT'])].copy()
gb_d_qtrly.drop(labels='rebill_rule', axis=1,inplace=True)
gb_d_qtrly = gb_d_qtrly.groupby(['curr', 'period']).sum()
gb_d_qtrly.reset_index(inplace=True)

gb_d_semi_ann = gb_d[gb_d['rebill_rule']=='YH']

gb_d_annual = gb_d[gb_d['rebill_rule'].isin(['YA', 'YC', 'YX'])].copy()
gb_d_annual.drop(labels='rebill_rule', axis=1,inplace=True)
gb_d_annual = gb_d_annual.groupby(['curr', 'period']).sum()
gb_d_annual.reset_index(inplace=True)

gb_d_two_yrs = gb_d[gb_d['rebill_rule']=='Y4']
gb_d_three_yrs = gb_d[gb_d['rebill_rule']=='Y7']


list_df = [gb_rec, gb_svc, gb_b,
           gb_a_1M,    gb_a_1Y,    gb_a_2Y,       gb_a_3Y,
           gb_d_mthly, gb_d_qtrly, gb_d_semi_ann, gb_d_annual, gb_d_two_yrs, gb_d_three_yrs]

list_columns = ['recognized', 'service', 'deferred_B',
        'deferred_1M_a', 'deferred_1Y_a', 'deferred_2Y_a', 'deferred_3Y_a',
        'deferred_1M_d', 'deferred_3M_d', 'deferred_6M_d', 'deferred_1Y_d', 'deferred_2Y_d', 'deferred_3Y_d']


def merge_new_dataframe(old_df, new_df, new_column):
    df_merged = pd.merge(old_df, new_df, how='outer',
                         left_on=['curr', 'period'],
                         right_on=['curr', 'period'])
    df_merged.rename(index=str, columns={'DC_amount': new_column + '_DC', 'US_amount': new_column + '_US'},
                     inplace=True)

    # need to drop the product configtype id for merges where the new_df is of type A
    config_str = 'config'
    rule_str = 'rebill_rule'
    if config_str in df_merged.columns:
        df_merged.drop(columns=['config'], inplace=True)

    if rule_str in df_merged.columns:
        df_merged.drop(columns=['rebill_rule'], inplace=True)

    return df_merged


def merge_all_dataframes(list_df, list_columns):
    for i, df in enumerate(list_df):
        # print('This is i:', i)
        # print('referencing the column: ', list_columns[i])

        if i == 0:
            df_merged = list_df[0].copy()
            df_merged.rename(index=str, columns={'DC_amount': list_columns[i] + '_DC',
                                                 'US_amount': list_columns[i] + '_US'}, inplace=True)
        else:
            df_merged = merge_new_dataframe(df_merged, df, list_columns[i])

    return df_merged

df = merge_all_dataframes(list_df, list_columns)
df = df.fillna(0)
print(df.head(40))

output_file = '../data/Data_2020_P06/FOREIGN_CURR_BILLINGS.xlsx'
output_tab = 'test_base_bill'
start_row = 5
start_column = 1
header=True

df['deferred_A_DC'] = df['deferred_3Y_a_DC']+df['deferred_2Y_a_DC'] +\
                    df['deferred_1Y_a_DC']+df['deferred_1M_a_DC']

print(df['deferred_A_DC'].sum())
df['deferred_D_DC']  =df['deferred_3Y_d_DC']+df['deferred_2Y_d_DC'] +\
                    df['deferred_1Y_d_DC']+df['deferred_6M_d_DC']+\
                    df['deferred_3M_d_DC']+df['deferred_1M_d_DC']
print(df['deferred_D_DC'].sum())
print(df.columns)
out_columns = ['curr', 'period', 'recognized_DC', 'service_DC',
               'deferred_B_DC',
               'deferred_A_DC', 'deferred_3Y_a_DC', 'deferred_2Y_a_DC',
               'deferred_1Y_a_DC', 'deferred_1M_a_DC',
               'deferred_D_DC',
               'deferred_3Y_d_DC', 'deferred_2Y_d_DC',
               'deferred_1Y_d_DC', 'deferred_6M_d_DC',
               'deferred_3M_d_DC', 'deferred_1M_d_DC']

df_out = df[out_columns]
print(df_out.columns)

df_to_xl(df_out, output_file, output_tab, start_row, start_column, header)

df_to_xl(gb_a_dropped, output_file, output_tab, 30, start_column, header)
