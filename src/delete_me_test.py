import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_bookings(config_dict):
    '''
    The bookings files come from FP&A and the structure of these files changes often

    :param config_dict:
    :return: df_bookings
    '''
    filename_DX = config_dict['path_to_data'] + config_dict['bookings']['filename_DX']
    sheetname_DX = config_dict['bookings']['sheetname_DX']

    filename_DME = config_dict['path_to_data'] + config_dict['bookings']['filename_DME']
    sheetname_DME = config_dict['bookings']['sheetname_DME']
    start_row_DME = config_dict['bookings']['start_row_DME']

    df_DX = load_DX_bookings(filename, sheetname, start)
    df_DME = load_DME_bookings(filename, sheetname)
    df = pd.concat([df_DME, df_DX])

    return df

def load_DME_bookings(filename, sheetname):

    df_DME = pd.read_excel(filename, sheetname)

    df_DME = df_DME.rename(columns={'Metrics': 'metrics',
                            'Profit center': 'profit_center',
                            'Market Area': 'market_area',
                            'Market Segement': 'segment',
                            'Q1 2021':'Q1_2021',
                            'Q2 2021':'Q2_2021',
                            'Q3 2021':'Q3_2021',
                            'Q4 2021':'Q4_2021'
                            })

    # Drop the columns that we will not use
    df_DME = df_DME.drop(columns = ['segment', 'GTM', '2021'])

    # only want the Net ACV bookings
    df_DME = df_DME[df_DME['metrics']=='Net ACV']

    # ADJUSTING THE profit_center to not double count numbers
    # creating the BU_ID
    df_DME['BU_id'] =  df_DME['profit_center'].apply(lambda st: st[0:st.find("-")])
    df_DME['BU_segment'] = df_DME['profit_center'].apply(lambda st: st[st.find("-")+1:])

    df_DME['BU_id'] = df_DME['BU_id'].str.strip()
    df_DME['BU_segment'] = df_DME['BU_segment'].str.strip()

    list_BU_keepers = ['EB10', 'EB15']
    df_DME = df_DME[df_DME['BU_id'].isin(list_BU_keepers)]

    # NOW WORKING ON GEO, REGION and MARKET AREA
    # identify the characters in a string
    df_DME['in_parens'] =  df_DME['market_area'].apply(lambda st: st[st.find("(")+1:st.find(")")])

    df_DME['market_area'] = df_DME['market_area'].apply(lambda st: st[0:st.find("(")-1])

    df_DME['pc_ID'] = df_DME['profit_center'].apply(lambda st: st[0:st.find('-')])
    df_DME['pc_descr'] = df_DME['profit_center'].apply(lambda st: st[st.find('-')+1:])

    df_DME['geo'] = df_DME[df_DME['in_parens']=='G']['market_area']
    df_DME['geo'] =df_DME['geo'].ffill()

    df_DME['region'] = df_DME[df_DME['in_parens']=='R']['market_area']
    df_DME['region'] = df_DME['region'].ffill()

    # filter to just include market area
    df_DME = df_DME[df_DME['in_parens']=='MA'].copy()

    # drop unnecessary columns and reorder the columns
    df_DME = df_DME.drop(columns=['profit_center', 'in_parens' ])

    # Rename pc_descr to be segment
    df_DME.rename(columns = {'pc_descr': 'segment'}, inplace=True)

    # Add the BU
    df_DME['BU'] = 'Digital Media'

    # We need to remove the segment data: it is not included in the DME bookings
    df_DME = df_DME.groupby(by = ['BU', 'segment', 'geo', 'region', 'market_area']).sum()
    df_DME = df_DME.reset_index()

    df_DME = df_DME[['BU', 'segment', 'geo', 'region', 'market_area', 'Q1_2021','Q2_2021', 'Q3_2021', 'Q4_2021']]


    print('Done with the DME dataframe:')
    print(df_DME.sum())

    return df_DME


def load_DX_bookings(filename, sheetname, start_row):
    df_DX = pd.read_excel(filename, sheetname, skiprows=start_row)


    df_DX = df_DX.rename(columns = {'Unnamed: 0': 'segment',
                                    'Unnamed: 1': 'market_area',
                                    'Unnamed: 2': 'profit_center',
                                    'Q1 2021':'Q1_2021',
                                    'Q2 2021':'Q2_2021',
                                    'Q3 2021':'Q3_2021',
                                    'Q4 2021':'Q4_2021'})

    df_DX = df_DX.drop(columns = ['segment', '2021'])

    # identify the characters in a string
    df_DX['in_parens'] =  df_DX['market_area'].apply(lambda st: st[st.find("(")+1:st.find(")")])

    df_DX['market_area'] = df_DX['market_area'].apply(lambda st: st[0:st.find("(")-1])

    df_DX['pc_ID'] = df_DX['profit_center'].apply(lambda st: st[0:st.find('-')])
    df_DX['pc_descr'] = df_DX['profit_center'].apply(lambda st: st[st.find('-')+1:])

    df_DX['geo'] = df_DX[df_DX['in_parens']=='G']['market_area']
    df_DX['geo'] =df_DX['geo'].ffill()

    df_DX['region'] = df_DX[df_DX['in_parens']=='R']['market_area']
    df_DX['region'] = df_DX['region'].ffill()

    # filter to just include market area
    df_DX = df_DX[df_DX['in_parens']=='MA'].copy()


    # Adding BU and Segment information (all Exp Cloud)
    df_DX['BU'] = 'Digital Experience'
    df_DX['segment'] = 'Experience Cloud'

    # drop unnecessary columns and reorder the columns
    df_DX = df_DX[['BU', 'segment', 'geo', 'region', 'market_area', 'Q1_2021','Q2_2021', 'Q3_2021', 'Q4_2021']]

    # We need to remove the segment data: it is not included in the DME bookings
    df_DX = df_DX.groupby(by = ['BU', 'segment', 'geo', 'region', 'market_area']).sum()
    df_DX = df_DX.reset_index()

    print('Done with the DX dataframe:')
    print(df_DX.sum())

    return df_DX



