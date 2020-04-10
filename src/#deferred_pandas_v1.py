# deferred_pandas_v1.py

'''
Describe data sources here




Describe process in steps here




output explained here



Later these will be made into easier to use functions and imported into a simpler program






'''

# list of functions
def remove_infreq_curr(df, count_threshold=10):
    '''
    There are many currencies that only appear a few times in the billings dataset and we are removing them due to their infrequenct billings.
    For example, there is one MXP transaction in the four year history of billings.

    NOTE: As of April 2020, the TRY currency does not have forward rates or exchange rates in our dataset,
    so we are temporarilly removing this currency as well.

    INPUT:
        df:                 the original billings database right after it is loaded into a dataframe

        count_threshold:    the cut off number of dates to remove a currency (default = 10)

    OUTPUT:

        df:          the billings dataframe with the less frequent  currencies removed

        model_dict:  a dictionary being used as the 'key' storing all relevant model information
                        'curr_removed' - is a list of the currencies removed by this function

    '''

    vc = df['curr'].value_counts()
    keep_these = vc.values > count_threshold
    keep_curr = vc[keep_these]
    a = keep_curr.index
    df = df[df['curr'].isin(a)]

    # keeping track of the currencies that have been removed in the model_dict
    remove_these = vc[vc.values <= 10].index
    model_dict = {'curr_removed': list(vc[remove_these].index)}
    delete_curr = list(remove_these)

    # removing the 'TRY' currency due to lack of FX data
    if 'TRY' not in model_dict:
        model_dict['curr_removed'].append('TRY')
        delete_curr.append('TRY')
        a = a.drop('TRY')

    # standard reporting metrics printed to screen
    print('Model dictionary', model_dict)
    print('Deleted Currencies', delete_curr)

    print("---Removing infrequent currencies from billings history---")
    print('Total number of currencies in the base billings file: ', len(vc))
    if len(model_dict['curr_removed'])==0:
        print('No currencies were removed, all contained 10 or more billings')
        print('Currencies in the base billings file')
        for item in a:
            print(a[item], end = " ")
    else:
        print('\n Currencies were removed: ', len(model_dict['curr_removed']))

        for item in remove_these:
            print(item, ', ', end="")

        print("\n\n{} Remaining currencies: ".format(len(a)))
        for item in a:
            print(item, ', ', end="")

    return df model_dict



#import statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from math import ceil

# load up the base billings file
df = pd.read_excel('../data/all_billings_inputs.xlsx', sheet_name='base_billings')

# Rename the columns
df.rename(index = str, columns = {'Document Currency': 'curr',
                                 'Enterprise BU Desc': 'BU',
                                 'Invoice Fiscal Year Period Desc': 'period',
                                 'Product Config Type': 'config',
                                 'Rev Rec Category': 'rev_req_type',
                                 'Rule For Bill Date': 'rebill_rule',
                                 'Completed Sales ( DC )': 'DC_amount',
                                 'Completed Sales': 'US_amount'}, inplace=True)


df, model_dict = remove_infreq_curr(df)

#remove any values that are zero
print('This is the length of the dataframe before removing zeros: ', len(df))
df = df[df['DC_amount']!=0]
print('This is the length of the dataframe AFTER removing zeros: ', len(df))

# Remove any billings that are 'NON-REV' sales type
print('Length of the dataframe before removing non-revenue billings: ', len(df))
df = df[df['Sales Type']!='NON-REV']
print('Length of the dataframe after removing non-revenue billings:  ', len(df))

# starting split - apply - combine in pandas
# split into sales type dataframes
rec = df[df['Sales Type']=='RECOGNIZED'].copy()
svc = df[df['Sales Type']=='PRO-SVC-INV'].copy()
dfr = df[df['Sales Type']=='DEFERRED'].copy()

# RECOGNIZED REVENUE
# NOTE: The subscription term is the only numeric field that stays after the groupby completed.
gb_rec = rec.groupby(['curr', 'BU', 'period'], as_index=False).sum()
gb_rec.drop(labels='Subscription Term', axis=1,inplace =True)

# SERVICE BASED BILLINGS
gb_svc = svc.groupby(['curr', 'BU', 'period'], as_index=False).sum()
gb_svc.drop(labels='Subscription Term', axis=1,inplace =True)

# DEFERRED BILLINGS
# Type B: filter out the type B first then do a group_by
dfr_b = dfr[dfr['rev_req_type']=='B'].copy()
gb_b = dfr_b.groupby(['curr', 'BU', 'period'], as_index=False).sum()
gb_b.drop(labels='Subscription Term', axis=1, inplace=True)

print('length of deferred billings : ', len(dfr))
print('length of the type B billings: ', len(dfr_b))

# Type A:
'''
Type A billings have a billing plan that specifies the time between billings. {'3Y', '2Y', '1Y', 'MTHLY'}
'''
dfr_a = dfr[dfr['rev_req_type']=='A'].copy()
gb_a = dfr_a.groupby(['curr', 'BU', 'period',
                     'config'], as_index=False).sum()
gb_a.drop(labels='Subscription Term', axis=1, inplace = True)

config_list = ['1Y', '2Y', '3Y', 'MTHLY']
df_temp1 = gb_a[gb_a['config'].isin(config_list)]

gb_a_1Y = df_temp1[df_temp1['config']=='1Y'].copy()
gb_a_2Y = df_temp1[df_temp1['config']=='2Y'].copy()
gb_a_3Y = df_temp1[df_temp1'config']=='3Y'].copy()
gb_a_1M = df_temp1[df_temp1['config']=='MTHLY'].copy()

print('this is the lenght of type A 1M billings: ', len(gb_a_1M))
print('this is the lenght of type A 1Y billings: ', len(gb_a_1Y))
print('this is the lenght of type A 2Y billings: ', len(gb_a_2Y))
print('this is the lenght of type A 3Y billings: ', len(gb_a_3Y))

# Type D:
'''
Type D billings have a rule for bill data that determines when the contract rebills
    Monthly:        {Y1, Y2, Y3, Y5}
    Quarterly:      YQ
    Every 4 months: YT     NOTE: These are treated as quarterly because they are small in number and amount
    Semi-annual:    YH
    Annual:         {YA, YC}
    2 years:        Y4
    3 years:        Y7
'''
